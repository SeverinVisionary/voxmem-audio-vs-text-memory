"""Memory records, budgets and arm rendering (PREREGISTRATION §5, BYTE_ACCOUNTING §1-2).

Everything here is CPU-only and reader-agnostic except the counters: `Counter` mirrors the Ultravox processor's
position count and `MiniCPMCounter` MiniCPM-o 4.5's `chat()`, both using only the tokenizer. Budgets are per reader
(BYTE_ACCOUNTING): a Builder takes its positions from the reader's counter.
"""
import json
import math
import os
from functools import lru_cache

import numpy as np

from common import Q2, hkey

ANSWER_LINES = ["Answer the final question with only the answer, in a few words.",
                "Reply with just the final answer, as briefly as possible."]
WRITERS = ("asr", "C", "D", "Dp", "D-emo", "D-env", "D-spk")
FIELD_GROUPS = {"emo": ("emotion", "sv_emotion"), "env": ("event", "background"), "spk": ("speaker",)}
FORMATS = ("json", "kv", "prose")
EXTRAS = ("longer", "relational", "prosody")
GRULES = ("reservoir", "kcentre")
SPK_THRESHOLD = 0.86            # WavLM-base-plus-sv model-card same-speaker threshold


def system_prompt(answer_line=0):
    base = open(Q2 / "phase0/out/vox_evaluation__candidate_system_prompt.no_abstain.txt").read().strip()
    return base + "\n" + ANSWER_LINES[answer_line]


# ---------------------------------------------------------------- features
class Feats:
    def __init__(self, feat_dir, selftags=None):
        self.dir = feat_dir
        self.f = {}
        for line in open(os.path.join(feat_dir, "features.jsonl")):
            r = json.loads(line)
            self.f[r["key"]] = r
        self.selftags = selftags or {}

    @lru_cache(maxsize=None)
    def emb(self, key):
        with np.load(os.path.join(self.dir, "emb_parts", key.replace("#", "_") + ".npz")) as z:
            spk = z["spk"].astype(np.float32)
            ast = z["ast"].astype(np.float32)
        return spk / (np.linalg.norm(spk) + 1e-8), ast / (np.linalg.norm(ast) + 1e-8)


def audio_tokens(n_samples, hop=160, chunk=3000, ds=2, stack=8):
    """Ultravox audio positions for one clip (mirrors UltravoxProcessor: ceil frames, 30 s chunks)."""
    frames = math.ceil(max(n_samples, 2 * hop) / hop)
    full, rest = divmod(frames, chunk)
    return full * math.ceil(chunk / (ds * stack)) + (math.ceil(rest / (ds * stack)) if rest else 0)


def minicpm_audio_tokens(n_samples, hop=160, pool=5):
    """MiniCPM-o 4.5 positions for one clip in non-streaming `chat()`: audio_start + <unk> * n + audio_end
    (processing_minicpmo.get_audio_placeholder with chunk_input=False). Matches the 10 clips measured on dev."""
    f = (math.ceil(n_samples / hop) - 1) // 2 + 1
    return (f - pool) // pool + 1 + 2


class Counter:
    """Exact Ultravox input length from text with <|audio|> placeholders plus clip lengths."""

    def __init__(self, tok):
        self.tok = tok

    def audio(self, n_samples):
        return audio_tokens(n_samples)

    def text_len(self, s):
        return len(self.tok(s, add_special_tokens=False)["input_ids"])

    def render(self, system, user):
        return self.tok.apply_chat_template([dict(role="system", content=system), dict(role="user", content=user)],
                                            add_generation_prompt=True, tokenize=False)

    def count(self, system, user, clip_lens):
        parts = self.render(system, user).split("<|audio|>")
        assert len(parts) == len(clip_lens) + 1, (len(parts), len(clip_lens))
        return sum(self.text_len(p) for p in parts) + sum(self.audio(n) for n in clip_lens)


class MiniCPMCounter(Counter):
    """Exact MiniCPM-o 4.5 input length for the flat layout as `reader.MiniCPM.generate` sends it: the user text is
    split at <|audio|>, parts stripped, empty parts dropped, joined with newlines, each clip as <audio>./</audio>."""
    PATTERN = "<audio>./</audio>"

    def audio(self, n_samples):
        return minicpm_audio_tokens(n_samples)

    def render(self, system, user):
        items = []
        for i, part in enumerate(user.split("<|audio|>")):
            if i:
                items.append(self.PATTERN)
            if part.strip():
                items.append(part.strip())
        return self.tok.apply_chat_template(
            [dict(role="system", content=system), dict(role="user", content="\n".join(items))],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
            use_tts_template=self.PATTERN in items)

    def count(self, system, user, clip_lens):
        parts = self.render(system, user).split(self.PATTERN)
        assert len(parts) == len(clip_lens) + 1, (len(parts), len(clip_lens))
        return sum(self.text_len(p) for p in parts) + sum(self.audio(n) for n in clip_lens)


# ---------------------------------------------------------------- records
def fmt_fields(d, fmt):
    d = {k: v for k, v in d.items() if v not in (None, "", [])}
    if fmt == "json":
        return json.dumps(d, separators=(",", ":"), ensure_ascii=False)
    if fmt == "kv":
        return "; ".join(f"{k}={v}" for k, v in d.items())
    return " ".join(f"{k.replace('_', ' ')}: {v}." for k, v in d.items())


def r2(x):
    return None if x is None else round(float(x), 2)


class ItemMemory:
    """Causal per-turn records for one item. All derived state uses only turns up to and including the turn."""

    def __init__(self, item, feats: Feats):
        self.item, self.F = item, feats
        self.user = [t["key"] for s in item["sessions"] for t in s["turns"] if t["role"] == "user"]
        self._online_state()

    def _online_state(self):
        # D: session-local diarisation (state reset per session). D': causal item-global speaker state.
        self.spk_local = {}
        for s in self.item["sessions"]:
            keys = [t["key"] for t in s["turns"] if t["role"] == "user"]
            self.spk_local.update(self._cluster(keys))
        cents, counts, last_of, hist = [], [], {}, {}
        self.spk, self.rel = {}, {}
        for k in self.user:
            e, _ = self.F.emb(k)
            sims = [float(e @ c) for c in cents]
            j = int(np.argmax(sims)) if sims and max(sims) >= SPK_THRESHOLD else None
            if j is None:
                cents.append(e.copy()); counts.append(1); j = len(cents) - 1
            else:
                counts[j] += 1
                c = cents[j] + (e - cents[j]) / counts[j]
                cents[j] = c / (np.linalg.norm(c) + 1e-8)
            sid = f"S{j + 1}"
            prev = last_of.get(sid)
            p = self.F.f[k]["pros"]
            h = hist.setdefault(sid, [])
            dz = {}
            for name, val in (("f0", p.get("f0_med")), ("loud", p.get("rms_db")), ("rate", p.get("rate_wps"))):
                xs = [x[name] for x in h if x.get(name) is not None]
                if val is not None and len(xs) >= 2:
                    dz[f"dz_{name}"] = r2((val - np.mean(xs)) / max(np.std(xs), 1e-3))
            sc = np.array(self.F.f[k]["e2v"]["scores"], dtype=float)
            sc = sc / sc.sum() if sc.sum() > 0 else sc
            ent = float(-(sc[sc > 0] * np.log(sc[sc > 0])).sum() / np.log(len(sc))) if len(sc) > 1 else 0.0
            self.spk[k] = sid
            self.rel[k] = dict(same_spk_prev="yes" if prev else "no",
                               cos_prev=r2(e @ self.F.emb(prev)[0]) if prev else None, **dz,
                               emo_uncertainty=r2(ent))
            last_of[sid] = k
            h.append(dict(f0=p.get("f0_med"), loud=p.get("rms_db"), rate=p.get("rate_wps")))

    def _cluster(self, keys):
        cents, counts, out = [], [], {}
        for k in keys:
            e, _ = self.F.emb(k)
            sims = [float(e @ c) for c in cents]
            j = int(np.argmax(sims)) if sims and max(sims) >= SPK_THRESHOLD else None
            if j is None:
                cents.append(e.copy()); counts.append(1); j = len(cents) - 1
            else:
                counts[j] += 1
                c = cents[j] + (e - cents[j]) / counts[j]
                cents[j] = c / (np.linalg.norm(c) + 1e-8)
            out[k] = f"S{j + 1}"
        return out

    def fields(self, k, writer):
        if writer.startswith("D-"):                     # holdout ablation: transcript + one field group of D
            d = self.fields(k, "D")
            return {key: v for key, v in d.items() if key == "said" or key in FIELD_GROUPS[writer[2:]]}
        f = self.F.f[k]
        d = dict(said=f["sv"]["text"])
        if writer == "C":
            t = self.F.selftags.get(k)
            if t is None:
                raise KeyError(f"writer C: no self-tag for {k}")
            try:
                d.update(json.loads(t))
            except (json.JSONDecodeError, TypeError):
                d["tags"] = (t or "")[:200]
        elif writer in ("D", "Dp"):
            e = f["e2v"]
            d.update(speaker=self.spk[k] if writer == "Dp" else self.spk_local[k],
                     emotion=f"{e['labels'][0]} {e['scores'][0]:.2f}",
                     sv_emotion=f["sv"]["emo"].lower(),
                     event=f["sv"]["event"] if f["sv"]["event"] not in ("Speech", "") else None,
                     background=", ".join(f"{lab} {p:.2f}" for lab, p in f["ast"]["bg"]))
            if writer == "Dp":
                d.update(self.rel[k])
        return d

    def record(self, k, writer, fmt):
        return fmt_fields(self.fields(k, writer), fmt)

    def extra(self, k, kind):
        f = self.F.f[k]
        if kind == "longer":
            e = f["e2v"]
            return ("emotions " + ", ".join(f"{l} {s:.2f}" for l, s in list(zip(e["labels"], e["scores"]))[:3])
                    + "; sounds " + ", ".join(f"{l} {p:.2f}" for l, p in f["ast"]["top"]))
        if kind == "relational":
            return fmt_fields(dict(speaker=self.spk[k], **self.rel[k]), "kv")
        p = f["pros"]
        return fmt_fields(dict(pitch_hz=p.get("f0_med"), voiced=p.get("voiced"), loudness_db=p.get("rms_db"),
                               words_per_s=p.get("rate_wps"), seconds=p.get("dur")), "kv")

    def queue_order(self):
        """Bit-reversal order over user turns: every prefix is spread across the history."""
        n = len(self.user)
        bits = max(1, (n - 1).bit_length())
        idx = sorted(range(n), key=lambda i: (int(f"{i:0{bits}b}"[::-1], 2), i))
        return [self.user[i] for i in idx]


# ---------------------------------------------------------------- selection (online)
def select_reservoir(keys, cost, caps):
    """Online hash-priority eviction: on overflow drop the lowest-hash turn. Uses no content and no future
    turns; caps[t] is the capacity after turn t arrives (grows with audio seen so far). Not uniform under
    variable clip costs; recorded as 'hash-priority' in the preregistration."""
    kept = []
    for t, k in enumerate(keys):
        kept.append(k)
        while sum(cost[x] for x in kept) > caps[t]:
            kept.remove(min(kept, key=lambda x: hkey("res", x)))
    return [k for k in keys if k in kept]


def select_kcentre(keys, cost, caps, emb):
    """Online streaming k-centre: on overflow drop the member closest to the rest (older on ties)."""
    kept = []
    for t, k in enumerate(keys):
        kept.append(k)
        while sum(cost[x] for x in kept) > caps[t]:
            if len(kept) == 1:
                kept.pop(); break
            E = np.stack([emb(x) for x in kept])
            S = E @ E.T
            np.fill_diagonal(S, -np.inf)
            kept.pop(int(np.argmax(S.max(1))))
    return [k for k in keys if k in kept]


def select_recent(keys, cost, cap):
    kept, used = [], 0
    for k in reversed(keys):
        if used + cost[k] > cap:
            break
        kept.append(k); used += cost[k]
    return [k for k in keys if k in kept]


# ---------------------------------------------------------------- arms
class Builder:
    """Renders arms for one item and enforces the position budget exactly."""

    def __init__(self, item, mem: ItemMemory, counter: Counter, clip_len, writer="D", fmt="kv",
                 extra="longer", grule="reservoir", answer_line=0):
        self.item, self.mem, self.C, self.clip_len = item, mem, counter, clip_len
        self.writer, self.fmt, self.extra_kind, self.grule = writer, fmt, extra, grule
        self.system = system_prompt(answer_line)
        self.cost = {k: counter.audio(clip_len(k)) for k in mem.user}
        self.P_A = sum(self.cost.values())
        self.shared = self.positions(self.plan_none())
        m = self.plan_m()
        self.P_M = self.positions(m) - self.shared
        k0 = mem.user[0]                                   # text positions a placeholder adds (e.g. a space)
        self.ph = self.positions(self.with_audio(m, [k0])) - self.positions(m) - self.cost[k0]
        self.sel_cost = {k: c + self.ph for k, c in self.cost.items()}

    # plans: dict(rec={key: str}, audio={key: audio_key}, extra={key: [str]}, history=bool)
    def plan_none(self):
        return dict(history=False, rec={}, audio={}, extra={})

    def plan_m(self, writer=None, shuffled=False):
        w = writer or self.writer
        recs = {k: self.mem.record(k, w, self.fmt) for k in self.mem.user}
        if shuffled:
            ks = self.mem.user
            perm = sorted(ks, key=lambda x: hkey("shuf", x))
            recs = {k: recs[p] for k, p in zip(ks, perm)}
        return dict(history=True, rec=recs, audio={}, extra={})

    def render(self, plan):
        it = self.item
        clips = []
        if not plan["history"]:
            body = "Conversation history: (none)"
        else:
            lines = ["Conversation history:"]
            for s in it["sessions"]:
                lines.append(f"[Session {s['timestamp']}]")
                for t in s["turns"]:
                    k = t["key"]
                    if t["role"] != "user":
                        lines.append(f"Assistant: {t['text']}")
                        continue
                    seg = ["User:"]
                    if plan["rec"].get(k):
                        seg.append(plan["rec"][k])
                    if k in plan["audio"]:
                        seg.append("<|audio|>"); clips.append(plan["audio"][k])
                    seg += [f"+ {x}" for x in plan["extra"].get(k, [])]
                    if len(seg) == 1:
                        seg.append("[not stored]")
                    lines.append(" ".join(seg))
            body = "\n".join(lines)
        user = body + f"\n[Now {it['query_timestamp']}] Current question: <|audio|>"
        return user, clips + ["__question__"]

    def render_chat(self, plan):
        """The same plan in the multi-turn chat layout of the Phase 0 smoke (one message per turn; a
        `[Session <ts>]` / `OK.` exchange opens each session). Returns (messages without the system turn, clips);
        message contents mark audio with <|audio|>, clips in order."""
        it, msgs, clips = self.item, [], []
        if plan["history"]:
            for s in it["sessions"]:
                msgs += [dict(role="user", content=f"[Session {s['timestamp']}]"), dict(role="assistant", content="OK.")]
                for t in s["turns"]:
                    k = t["key"]
                    if t["role"] != "user":
                        msgs.append(dict(role="assistant", content=t["text"]))
                        continue
                    seg = []
                    if plan["rec"].get(k):
                        seg.append(plan["rec"][k])
                    if k in plan["audio"]:
                        seg.append("<|audio|>"); clips.append(plan["audio"][k])
                    seg += [f"+ {x}" for x in plan["extra"].get(k, [])]
                    msgs.append(dict(role="user", content=" ".join(seg) or "[not stored]"))
        msgs.append(dict(role="user", content=f"[Now {it['query_timestamp']}] <|audio|>"))
        return msgs, clips + ["__question__"]

    def positions(self, plan):
        user, clips = self.render(plan)
        return self.C.count(self.system, user, [self.clip_len(c) for c in clips])

    def target(self, f):
        return self.shared + self.P_M + math.ceil(f * self.P_A)

    def fill(self, plan, target):
        """Append extra-text lines (queue order) until the plan uses `target` positions; last line cut at token level."""
        plan = dict(plan, extra={k: list(v) for k, v in plan["extra"].items()})
        cur = self.positions(plan)
        if cur > target:
            raise ValueError(f"plan over budget before fill: {cur} > {target}")
        for k in self.mem.queue_order():
            line = self.mem.extra(k, self.extra_kind)
            plan["extra"].setdefault(k, []).append(line)
            new = self.positions(plan)
            if new <= target:
                cur = new
                if cur == target:
                    return plan
                continue
            plan["extra"][k].pop()                      # cut this line to the remaining positions
            ids = self.C.tok(line, add_special_tokens=False)["input_ids"]
            lo, hi = 0, len(ids)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                plan["extra"][k].append(self.C.tok.decode(ids[:mid]))
                ok = self.positions(plan) <= target
                plan["extra"][k].pop()
                lo, hi = (mid, hi) if ok else (lo, mid - 1)
            if lo:
                plan["extra"][k].append(self.C.tok.decode(ids[:lo]))
            return plan
        return plan                                        # queue exhausted (tiny histories)

    def with_audio(self, plan, keys, sources=None):
        p = dict(plan, audio=dict(plan["audio"]))
        for i, k in enumerate(keys):
            p["audio"][k] = sources[i] if sources else k
        return p

    @staticmethod
    def caps(f, keys, cost, final_cap):
        """Capacity after each arriving turn: ceil(f * selection cost seen so far), capped at final_cap; equals
        final_cap (the item budget) at the last turn. Discarded turns are never reconsidered."""
        seen, out = 0, []
        for k in keys:
            seen += cost[k]
            out.append(min(final_cap, math.ceil(f * seen)))
        out[-1] = final_cap
        return out

    def select(self, rule, cap, keys=None, cost=None, f=None):
        keys = keys or self.mem.user
        cost = cost or self.sel_cost
        caps = self.caps(f, keys, cost, cap) if f is not None else [cap] * len(keys)
        if rule == "reservoir":
            return select_reservoir(keys, cost, caps)
        if rule == "kcentre":
            return select_kcentre(keys, cost, caps, lambda k: np.concatenate(self.mem.F.emb(k)))
        if rule == "recent":
            return select_recent(keys, cost, cap)
        raise ValueError(rule)

    def budget_audio(self, rule, f, keys=None, cost=None, sources=None):
        """Audio turns for a budgeted arm, selected online with placeholder cost included.
        Returns (selected keys, target, capacity). A final exact check drops the newest pick if still over."""
        target = self.target(f)
        m = self.plan_m()
        cap = target - self.positions(m)
        sel = self.select(rule, cap, keys, cost, f=None if rule == "recent" else f)
        while sel and self.positions(self.with_audio(m, sel)) > target:
            sel = sel[:-1]
        return sel, target, cap

    def arm(self, name, f=None, donor=None):
        m = self.plan_m()
        if name == "N":
            return self.plan_none()
        if name.startswith("M:"):
            return self.plan_m(writer=name[2:])
        if name == "shuf":
            return self.plan_m(shuffled=True)
        if name == "T":
            return self.fill(m, self.target(f))
        if name in ("G", "E"):
            sel, target, _ = self.budget_audio(self.grule if name == "G" else "recent", f)
            return self.fill(self.with_audio(m, sel), target)
        if name == "Gmm":
            # Same turn count as G, in G's slots, audio from a donor family of the same evidence type: the donor's
            # own G selection, highest-hash clips kept down to G's count. `donor` is a list of candidate Builders,
            # tried in order; None if no donor fits the budget with the same count.
            own, target, _ = self.budget_audio(self.grule, f)
            if not own:
                return None
            for d in donor:
                d_sel, _, _ = d.budget_audio(self.grule, f)
                if len(d_sel) < len(own):
                    continue
                keep = set(sorted(d_sel, key=lambda x: -hkey("res", x))[:len(own)])
                srcs = [x for x in d_sel if x in keep]
                p = self.with_audio(m, own, srcs)
                if self.positions(p) <= target:
                    return self.fill(p, target)
            return None
        if name == "F":
            return self.with_audio(self.fill(m, self.target(0.25)), self.mem.user)
        if name == "A":
            return self.with_audio(dict(history=True, rec={}, audio={}, extra={}), self.mem.user)
        if name == "oracle":
            ev = set(donor)                 # for "oracle", `donor` is the evidence session id list (analysis only)
            keys = [k for k in self.mem.user if k.split("#")[0] in ev]
            return self.with_audio(m, keys)
        raise ValueError(name)

    def ledger(self, plan):
        """Positions by component (BYTE_ACCOUNTING §1); analysis fields added by the caller."""
        total = self.positions(plan)
        audio = sum(self.C.audio(self.clip_len(src)) for src in plan["audio"].values())
        return dict(total=total, memory=total - self.shared, audio=audio, text=total - self.shared - audio,
                    n_audio_turns=len(plan["audio"]), shared=self.shared, P_A=self.P_A, P_M=self.P_M)
