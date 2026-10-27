"""Readers (Ultravox; MiniCPM-o 4.5): generation, teacher-forced gold log-probability, self-tags (writer C). GPU.

Llama weights come from `text_model_id`. Without an HF token for the gated meta-llama repo, the ungated
`unsloth/Llama-3.1-8B-Instruct` re-upload is used and its provenance check is recorded (PREREG §4).
"""
import functools
import json
import os
import sys
import time

from memory import Counter, MiniCPMCounter

M = "fixie-ai/ultravox-v0_6-llama-3_1-8b"
GATED, MIRROR = "meta-llama/Llama-3.1-8B-Instruct", "unsloth/Llama-3.1-8B-Instruct"
TAG_PROMPT = ("Describe only HOW this was said, not the words. Reply in one line of JSON with keys "
              "emotion, pitch, rate, loudness, voice_events, background_sound, speaker_description.")


def provenance():
    """Compare the mirror with Meta's repo (needs HF_TOKEN with Llama access); returns a dict for the run record."""
    from huggingface_hub import HfApi
    api, out = HfApi(), {}
    try:
        def lfs(repo):
            info = api.model_info(repo, files_metadata=True)
            return {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings
                    if s.rfilename.endswith((".safetensors", ".json"))}
        a, b = lfs(GATED), lfs(MIRROR)
        weights = sorted(k for k in a if k.endswith(".safetensors"))
        out["weights_match"] = bool(weights) and all(a[k] == b.get(k) for k in weights)
        out["mismatched_files"] = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def redirect_gated(src):
    """Ultravox's processor and config resolve `text_model_id` = the gated repo; send those loads to `src`."""
    from transformers import AutoConfig
    _orig = AutoConfig.from_pretrained.__func__
    AutoConfig.from_pretrained = classmethod(
        lambda cls, name, *a, **k: _orig(cls, src if name == GATED else name, *a, **k))


class Ultravox:
    def __init__(self):
        import torch
        import transformers
        from transformers import AutoConfig, AutoModel, AutoProcessor
        self.torch = torch
        self.prov = provenance()
        use_meta = bool(os.environ.get("HF_TOKEN")) and "error" not in self.prov
        src = GATED if use_meta else MIRROR
        redirect_gated(src)
        t0 = time.time()
        self.proc = AutoProcessor.from_pretrained(M, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(M, trust_remote_code=True)
        cfg.text_model_id = src
        # sdpa: eager attention materialises fp32 n x n scores per layer and ran out of memory at ~3K positions
        self.model = AutoModel.from_pretrained(M, config=cfg, trust_remote_code=True, attn_implementation="sdpa",
                                               torch_dtype=torch.bfloat16).to("cuda").eval()
        self.tok = self.proc.tokenizer
        self.counter = Counter(self.tok)
        self.info = dict(model=M, text_model_id=src, transformers=transformers.__version__, attn="sdpa",
                         load_s=round(time.time() - t0, 1), provenance=self.prov)

    def inputs(self, system, user, wavs, suffix="", msgs=None):
        """`user` is the flat single-message layout; `msgs` (chat layout, no system turn) replaces it when given."""
        text = (self.tok.apply_chat_template([dict(role="system", content=system)] + msgs, add_generation_prompt=True,
                                             tokenize=False) if msgs else self.counter.render(system, user)) + suffix
        kw = dict(text=text, return_tensors="pt", sampling_rate=16000)
        if len(wavs) == 1:
            kw["audio"] = wavs[0]
        else:
            kw["audios"] = wavs
        x = self.proc(**kw).to("cuda")
        if "audio_values" in x:
            x["audio_values"] = x["audio_values"].to(self.torch.bfloat16)
        return x

    def generate(self, system, user, wavs, max_new, msgs=None):
        torch = self.torch
        torch.cuda.synchronize(); t = time.perf_counter()
        x = self.inputs(system, user, wavs, msgs=msgs)
        # generate rejects logits_to_keep for Ultravox (its forward only takes **kwargs), so the prefill would
        # compute vocabulary logits at every input position; greedy decoding needs the last one only
        lm = self.model.language_model
        lm.forward = functools.partial(type(lm).forward, lm, logits_to_keep=1)
        try:
            with torch.no_grad():
                o = self.model.generate(**x, max_new_tokens=max_new, do_sample=False)
        finally:
            del lm.forward
        torch.cuda.synchronize()
        n_in = x["input_ids"].shape[1]
        new = o[0][n_in:]
        return dict(answer=self.tok.decode(new, skip_special_tokens=True), n_new=int(len(new)),
                    hit_max=bool(len(new) >= max_new), seconds=round(time.perf_counter() - t, 3), n_in=int(n_in))

    def gold_logprob(self, system, user, wavs, gold):
        torch = self.torch
        g = self.tok(gold, add_special_tokens=False)["input_ids"]
        x = self.inputs(system, user, wavs, suffix=gold)
        ids = x["input_ids"][0]
        assert ids[-len(g):].tolist() == g, "gold tokenisation shifted at the prompt boundary"
        with torch.no_grad():
            logits = self.model(**x, logits_to_keep=len(g) + 1).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[-len(g):, None]).squeeze(-1)
        return dict(lp_sum=round(float(lp.sum()), 4), lp_mean=round(float(lp.mean()), 4), n_gold=len(g))

    def selftags(self, wavs, batch=1, max_new=48):
        """One JSON tag line per clip. batch > 1 uses Ultravox's own collator with left padding."""
        torch = self.torch
        msgs = [dict(role="user", content="<|audio|>\n" + TAG_PROMPT)]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if batch == 1:
            out = []
            for w in wavs:
                x = self.proc(text=text, audio=w, return_tensors="pt", sampling_rate=16000).to("cuda")
                x["audio_values"] = x["audio_values"].to(torch.bfloat16)
                with torch.no_grad():
                    o = self.model.generate(**x, max_new_tokens=max_new, do_sample=False)
                out.append(self.tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            return out
        mod = sys.modules[type(self.proc).__module__]
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        coll = mod.DataCollatorForSeq2SeqWithAudio(tokenizer=self.tok, padding=True)
        feats = []
        for w in wavs:
            x = self.proc(text=text, audio=w, return_tensors="pt", sampling_rate=16000)
            feats.append({k: (v[0] if k in ("input_ids", "attention_mask") else v) for k, v in x.items()})
        b = coll(feats).to("cuda")
        b["audio_values"] = b["audio_values"].to(torch.bfloat16)
        with torch.no_grad():
            o = self.model.generate(**b, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
        L = b["input_ids"].shape[1]
        return [self.tok.decode(r[L:], skip_special_tokens=True).strip() for r in o]


class MiniCPM:
    """MiniCPM-o 4.5, audio only, greedy, through its own `chat()` (as in the Phase 0 smoke)."""
    NAME = "openbmb/MiniCPM-o-4_5"

    def __init__(self):
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        t0 = time.time()
        self.model = AutoModel.from_pretrained(self.NAME, trust_remote_code=True, attn_implementation="sdpa",
                                               torch_dtype=torch.bfloat16, init_vision=False, init_audio=True,
                                               init_tts=False).to("cuda").eval()
        self.tok = AutoTokenizer.from_pretrained(self.NAME, trust_remote_code=True)
        self.counter = MiniCPMCounter(self.tok)
        self.info = dict(model=self.NAME, transformers=transformers.__version__, load_s=round(time.time() - t0, 1))
        self._gold, self._lp, self.n_in = None, None, None
        m = self.model
        stops = [self.tok.convert_tokens_to_ids(t) for t in m.terminators]
        stop = stops[0]

        class Out(dict):
            __getattr__ = dict.__getitem__

        def decode(inputs_embeds, tokenizer, attention_mask, **kw):
            # chat() builds the input embeddings and hands them to _decode: record the exact input length, and for
            # gold_logprob score the gold continuation by teacher forcing instead of generating
            self.n_in = int(inputs_embeds.shape[1])
            if self._gold is None:
                # MiniCPM's own _decode asks for every layer's hidden states (for TTS) and its prefill computes
                # vocabulary logits at every input position (1.8 GiB at 6K positions: out of memory on F on dev);
                # greedy text decoding needs neither
                lm = m.llm
                lm.forward = functools.partial(type(lm).forward, lm, logits_to_keep=1)
                try:
                    return lm.generate(inputs_embeds=inputs_embeds, pad_token_id=0, eos_token_id=stops,
                                       attention_mask=attention_mask, return_dict_in_generate=True, **kw)
                finally:
                    del lm.forward
            g = torch.tensor([self._gold], device=inputs_embeds.device)
            e = torch.cat([inputs_embeds, m.llm.get_input_embeddings()(g).to(inputs_embeds.dtype)], 1)
            logits = m.llm(inputs_embeds=e, logits_to_keep=g.shape[1] + 1).logits[0, :-1].float()
            self._lp = torch.log_softmax(logits, -1).gather(-1, g[0][:, None]).squeeze(-1)
            return Out(sequences=torch.tensor([[stop]], device=inputs_embeds.device))
        m._decode = decode

    @staticmethod
    def content(text, wavs):
        """'a <|audio|> b' -> ['a', wav, 'b'] (empty strings dropped), consuming clips from the iterator `wavs`."""
        out = []
        for i, part in enumerate(text.split("<|audio|>")):
            if i:
                out.append(next(wavs))
            if part.strip():
                out.append(part.strip())
        return out

    def _chat(self, system, user, wavs, max_new, msgs=None):
        it = iter(wavs)
        conv = [dict(role="system", content=system)]
        for m in (msgs or [dict(role="user", content=user)]):
            conv.append(dict(role=m["role"], content=self.content(m["content"], it) if m["role"] == "user"
                             else [m["content"]]))
        assert next(it, None) is None, "unused audio clips"
        self.n_in = None
        with self.torch.no_grad():   # chat() defaults to sampling with 3 beams and right-truncates at 8192 tokens
            return self.model.chat(msgs=conv, tokenizer=self.tok, do_sample=False, num_beams=1, max_inp_length=None,
                                   max_new_tokens=max_new, use_tts_template=False, generate_audio=False)

    def generate(self, system, user, wavs, max_new, msgs=None):
        torch = self.torch
        torch.cuda.synchronize(); t = time.perf_counter()
        ans = self._chat(system, user, wavs, max_new, msgs)
        torch.cuda.synchronize()
        ans = ans if isinstance(ans, str) else str(ans)
        n_new = len(self.tok(ans, add_special_tokens=False)["input_ids"])
        return dict(answer=ans, n_new=n_new, hit_max=n_new >= max_new, seconds=round(time.perf_counter() - t, 3),
                    n_in=self.n_in)

    def gold_logprob(self, system, user, wavs, gold):
        """Teacher-forced log-probability of the gold answer right after the generation prompt (flat layout)."""
        self._gold = self.tok(gold, add_special_tokens=False)["input_ids"]
        try:
            self._chat(system, user, wavs, 1)
            lp = self._lp
        finally:
            self._gold, self._lp = None, None
        return dict(lp_sum=round(float(lp.sum()), 4), lp_mean=round(float(lp.mean()), 4), n_gold=len(lp))


def minicpm_positions_per_second(wavs):
    """Load MiniCPM-o 4.5 and count the LLM input length it builds for clips of known length (PREREG §8)."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "openbmb/MiniCPM-o-4_5"
    model = AutoModel.from_pretrained(name, trust_remote_code=True, attn_implementation="sdpa",
                                      torch_dtype=torch.bfloat16, init_vision=False, init_audio=True,
                                      init_tts=False).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    seen = []
    orig = model.generate

    def spy(*a, **k):
        for key in ("input_ids", "inputs_embeds"):
            v = k.get(key)
            if v is not None:
                seen.append(int(v.shape[1])); break
        return orig(*a, **k)
    model.generate = spy
    rows = []
    for w in wavs:
        for content in (["Hello."], [w, "Hello."]):
            seen.clear()
            model.chat(msgs=[dict(role="user", content=content)], tokenizer=tok, do_sample=False, num_beams=1,
                       max_new_tokens=1, use_tts_template=True, generate_audio=False, max_inp_length=None)
            rows.append(seen[0] if seen else None)
    pairs = [(len(w) / 16000, rows[2 * i + 1] - rows[2 * i]) for i, w in enumerate(wavs)
             if rows[2 * i] is not None and rows[2 * i + 1] is not None]
    secs = sum(p[0] for p in pairs)
    del model; torch.cuda.empty_cache()
    return dict(n=len(pairs), audio_s=round(secs, 1),
                positions_per_s=round(sum(p[1] for p in pairs) / secs, 2) if secs else None,
                per_clip=[(round(a, 2), b) for a, b in pairs][:20], note=json.dumps(dict(np_version=np.__version__)))
