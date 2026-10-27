"""Writer features per unique user turn (writer D / D' inputs). CPU or GPU.

    python features.py DATA_DIR OUT_DIR [--device cpu|cuda]

Reads DATA_DIR/items/*.json and the turn audio; writes OUT_DIR/features.jsonl (one line per turn key) and
OUT_DIR/emb_parts/<session>_<turn>.npz (float16 `spk` and `ast` embeddings).
Resumable: turns already in features.jsonl are skipped.

Models (all public, no ESC-50 heads — PREREGISTRATION §5 leakage rules):
  SenseVoiceSmall (ASR + emotion + audio events), emotion2vec_plus_large (utterance emotion),
  AST AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593; background labels + pooled embedding),
  WavLM-base-plus-sv (speaker x-vector), librosa pyin / rms (prosody).
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np

from common import SR
from data import load_audio

SPEECHY = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
           "Conversation", "Narration, monologue", "Speech synthesizer", "Babbling", "Inside, small room",
           "Silence", "Whispering", "Shout", "Yell", "Laughter", "Giggle", "Chuckle, chortle", "Sigh", "Sniff",
           "Cough", "Breathing", "Throat clearing"}


def user_turn_keys(data_dir):
    keys = []
    for f in sorted(glob.glob(os.path.join(data_dir, "items", "*.json"))):
        for s in json.load(open(f))["sessions"]:
            keys += [t["key"] for t in s["turns"] if t["role"] == "user"]
    return list(dict.fromkeys(keys))


class Writers:
    def __init__(self, device):
        import torch
        from funasr import AutoModel
        from transformers import (ASTFeatureExtractor, ASTForAudioClassification, AutoFeatureExtractor,
                                  WavLMForXVector)
        self.torch, self.device = torch, device
        self.sv = AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf", device=device, disable_update=True)
        self.e2v = AutoModel(model="emotion2vec/emotion2vec_plus_large", hub="hf", device=device, disable_update=True)
        self.ast_fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device).eval()
        self.spk_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.spk = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()

    def sensevoice(self, wav):
        r = self.sv.generate(input=wav, cache={}, language="auto", use_itn=True)[0]["text"]
        tags = re.findall(r"<\|([^|]*)\|>", r)
        text = re.sub(r"<\|[^|]*\|>", "", r).strip()
        return dict(lang=tags[0] if tags else "", emo=tags[1] if len(tags) > 1 else "",
                    event=tags[2] if len(tags) > 2 else "", text=text)

    def emotion(self, wav):
        r = self.e2v.generate(wav, granularity="utterance", extract_embedding=False)[0]
        labels = [str(x).split("/")[-1] for x in r["labels"]]
        scores = [float(x) for x in r["scores"]]
        order = np.argsort(scores)[::-1]
        return dict(labels=[labels[i] for i in order], scores=[round(scores[i], 4) for i in order])

    def audioset(self, wav):
        torch = self.torch
        x = self.ast_fe(wav, sampling_rate=SR, return_tensors="pt").to(self.device)
        with torch.no_grad():
            o = self.ast(**x, output_hidden_states=True)
        p = torch.sigmoid(o.logits[0]).float().cpu().numpy()
        lab = self.ast.config.id2label
        top = [(lab[i], round(float(p[i]), 4)) for i in np.argsort(p)[::-1][:8]]
        bg = [t for t in top if t[0] not in SPEECHY][:3]
        emb = o.hidden_states[-1][0].mean(0).float().cpu().numpy()
        return dict(top=top[:5], bg=bg), emb

    def speaker(self, wav):
        torch = self.torch
        x = self.spk_fe(wav, sampling_rate=SR, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            e = self.spk(**x).embeddings[0]
        e = torch.nn.functional.normalize(e, dim=-1)
        return e.float().cpu().numpy()


def prosody(wav, words):
    import librosa
    dur = len(wav) / SR
    f0, voiced, _ = librosa.pyin(wav, fmin=65, fmax=400, sr=SR, frame_length=1024)
    v = f0[voiced] if voiced is not None and voiced.any() else np.array([])
    rms = librosa.feature.rms(y=wav)[0]
    return dict(dur=round(dur, 2), f0_med=round(float(np.median(v)), 1) if len(v) else None,
                voiced=round(float(voiced.mean()), 3) if voiced is not None else 0.0,
                rms_db=round(float(20 * np.log10(rms.mean() + 1e-8)), 1),
                rate_wps=round(words / dur, 2) if dur > 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir"); ap.add_argument("out_dir"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    fpath = os.path.join(a.out_dir, "features.jsonl")
    done = {json.loads(l)["key"] for l in open(fpath)} if os.path.exists(fpath) else set()
    keys = [k for k in user_turn_keys(a.data_dir) if k not in done]
    print(f"{len(done)} cached, {len(keys)} to do", flush=True)
    W = Writers(a.device)
    emb_path = os.path.join(a.out_dir, "emb_parts")
    os.makedirs(emb_path, exist_ok=True)
    t0 = time.time()
    with open(fpath, "a") as f:
        for i, k in enumerate(keys):
            wav = load_audio(a.data_dir, k)
            sv = W.sensevoice(wav)
            ast, e_ast = W.audioset(wav)
            e_spk = W.speaker(wav)
            rec = dict(key=k, sv=sv, e2v=W.emotion(wav), ast=ast, pros=prosody(wav, len(sv["text"].split())))
            np.savez(os.path.join(emb_path, k.replace("#", "_") + ".npz"), spk=e_spk.astype(np.float16),
                     ast=e_ast.astype(np.float16))
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(keys)} turns in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
