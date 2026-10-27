"""Q2 dev pilot, GPU part (PREREGISTRATION §8-9). Resumable; every stage appends to OUT.

    python run_dev.py DATA FEAT OUT --stage selftags|screen|arms|minicpm [--limit N]
                      [--reader ultravox|minicpm] [--arms dev|test] [--settings FILE]

DATA: output of data.py. FEAT: output of features.py. OUT: results directory (Drive).
--reader minicpm: budgets from MiniCPM-o's own position count; files get a `_minicpm` suffix. Self-tags (writer C)
always come from Ultravox. --arms test: the §8 test arms, generation only, settings from --settings (frozen file).
"""
import argparse
import glob
import json
import os
import time

import numpy as np

import memory as Mm
from common import gold_text, hkey, load_eval, load_item
from data import load_audio
from memory import Builder, Feats, ItemMemory

ARMS = (["N", "A", "F", "M:asr", "M:C", "M:D", "M:Dp", "shuf", "oracle"]
        + [f"{a}@{f}" for a in ("T", "G", "E") for f in (0.10, 0.25, 0.50)] + ["Gmm@0.25", "T@0.25#line1"])
HOLDOUT_ARMS = ["N", "A", "M:asr", "M:D-emo", "M:D-env", "M:D-spk", "M:D", "T@0.25", "G@0.25", "F"]
CEILING_ARMS = ["N", "A", "M:asr", "T@0.25", "G@0.25", "F"]
CEILING_PER_TYPE = 16
TEST_ARMS = ["N", "A", "F"] + [f"{a}@{f}" for a in ("T", "G", "E") for f in (0.25, 0.50, 0.10)] + ["Gmm@0.25"]
DEFAULTS = dict(writer="D", fmt="kv", extra="longer", grule="reservoir")
FACTORS = [("writer", ["C", "D", "Dp"], "T@0.25"), ("fmt", list(Mm.FORMATS), "T@0.25"),
           ("extra", list(Mm.EXTRAS), "T@0.25"), ("grule", list(Mm.GRULES), "G@0.25")]
MAX_NEW = 64
SCREEN = "screen_sdpa.jsonl"        # screen.jsonl holds eager-attention rows from the run that ran out of memory
SUFFIX = dict(ultravox="", minicpm="_minicpm")


def check_count(rows):
    """The counter must reproduce the reader's real input length up to one constant per item (a constant cancels in
    P_M, P_A and every budget). Raise on a mismatch that varies between plans of the same item."""
    gaps = {}
    for r in rows:
        if r.get("n_in") is not None and "total" in r:
            gaps.setdefault(r["qid"], set()).add(r["n_in"] - r["total"])
    bad = {q: sorted(g) for q, g in gaps.items() if len(g) > 1}
    if bad:
        raise RuntimeError(f"position counter disagrees with the reader by a varying amount: {bad}")


def jl_append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def jl_read(path):
    """Read a JSONL log; a truncated final line (interrupted write) is dropped, interior corruption raises."""
    if not os.path.exists(path):
        return []
    lines = open(path).read().splitlines()
    out = []
    for i, l in enumerate(lines):
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return out


class Ctx:
    def __init__(self, data, feat, out, limit):
        self.data, self.out = data, out
        self.qids = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(data, "items", "*.json")))
        if limit:
            self.qids = self.qids[:limit]
        self.items = {q: load_item(data, q) for q in self.qids}
        self.ev = {q: load_eval(data, q) for q in self.qids}
        tags = {r["key"]: r["tag"] for r in jl_read(os.path.join(out, "selftags.jsonl"))}
        self.F = Feats(feat, tags)
        self._wav = {}

    def wav(self, key, qid=None):
        if key == "__question__":
            return np.load(os.path.join(self.data, "qaudio", f"{qid}.npy")).astype(np.float32) / 32767
        if key not in self._wav:
            self._wav[key] = load_audio(self.data, key)
        return self._wav[key]

    def builder(self, q, reader, **settings):
        s = dict(DEFAULTS, **settings)
        mem = ItemMemory(self.items[q], self.F)
        return Builder(self.items[q], mem, reader.counter, lambda k: len(self.wav(k, q)), writer=s["writer"],
                       fmt=s["fmt"], extra=s["extra"], grule=s["grule"], answer_line=s.get("answer_line", 0))

    def donors(self, q):
        same = [x for x in self.qids if self.ev[x]["evidence_type"] == self.ev[q]["evidence_type"] and x != q]
        return sorted(same, key=lambda x: hkey("donor", q, x))

    def plan(self, b, q, arm, reader):
        name, _, f = arm.split("#")[0].partition("@")
        f = float(f) if f else None
        if name == "Gmm":
            return b.arm("Gmm", f, donor=[self.builder(d, reader, **self.settings) for d in self.donors(q)])
        if name == "oracle":
            return b.arm("oracle", donor=self.ev[q]["evidence_session_ids"])
        return b.arm(name, f)

    def wavs(self, b, plan, q):
        _, clips = b.render(plan)
        return [self.wav(c, q) for c in clips]


def pending_selftags(ctx, out):
    done = {r["key"] for r in jl_read(os.path.join(out, "selftags.jsonl"))}
    keys = [t["key"] for q in ctx.qids for s in ctx.items[q]["sessions"] for t in s["turns"]
            if t["role"] == "user"]
    return [k for k in dict.fromkeys(keys) if k not in done]


def stage_selftags(ctx, reader, out):
    path = os.path.join(out, "selftags.jsonl")
    keys = pending_selftags(ctx, out)
    tp = {}
    probe = keys[:16]
    if probe:                                            # batch 1 vs batch 8 on the same clips: speed and agreement
        w = [ctx.wav(k) for k in probe]
        t = time.perf_counter(); one = reader.selftags(w, batch=1); tp["b1_s_per_turn"] = (time.perf_counter() - t) / len(w)
        try:
            t = time.perf_counter(); eight = []
            for i in range(0, len(w), 8):
                eight += reader.selftags(w[i:i + 8], batch=8)
            tp["b8_s_per_turn"] = (time.perf_counter() - t) / len(w)
            tp["b8_agree"] = sum(a == b for a, b in zip(one, eight)) / len(w)
        except Exception as e:  # noqa: BLE001
            tp["b8_error"] = f"{type(e).__name__}: {e}"[:400]
        for k, tag in zip(probe, one):
            jl_append(path, dict(key=k, tag=tag))
        json.dump(tp, open(os.path.join(out, "selftag_throughput.json"), "w"), indent=1)
        print("throughput", tp, flush=True)
    rest = keys[len(probe):]
    use_b8 = tp.get("b8_agree", 0) >= 0.75 and tp.get("b8_s_per_turn", 9e9) < tp.get("b1_s_per_turn", 0)
    bs = 8 if use_b8 else 1
    t0 = time.time()
    for i in range(0, len(rest), bs):
        chunk = rest[i:i + bs]
        tags = reader.selftags([ctx.wav(k) for k in chunk], batch=bs)
        for k, tag in zip(chunk, tags):
            jl_append(path, dict(key=k, tag=tag))
        if i % 80 == 0:
            print(f"selftags {i}/{len(rest)} {time.time() - t0:.0f}s", flush=True)
    tp.update(batch_used=bs, n=len(keys), total_s=round(time.time() - t0, 1))
    json.dump(tp, open(os.path.join(out, "selftag_throughput.json"), "w"), indent=1)


def stage_screen(ctx, reader, out, sfx=""):
    """One factor at a time; selection score = mean teacher-forced gold log-probability (PREREG §9).
    A row that fails (e.g. out of memory) is logged with an error; each value is scored on the items where every
    value of that factor has a log-probability."""
    import torch
    path = os.path.join(out, SCREEN.replace(".jsonl", f"{sfx}.jsonl") if sfx else SCREEN)
    rows = jl_read(path)
    settings = dict(DEFAULTS)
    missing_tags = [t["key"] for q in ctx.qids for s_ in ctx.items[q]["sessions"] for t in s_["turns"]
                    if t["role"] == "user" and t["key"] not in ctx.F.selftags]
    for factor, values, arm in FACTORS:
        scores, got_by = {}, {}
        if factor == "writer" and missing_tags:
            print(f"writer C excluded: {len(missing_tags)} turns lack self-tags", flush=True)
            values = [v for v in values if v != "C"]
        for v in values:
            s = dict(settings, **{factor: v})
            ctx.settings = s
            got = [r for r in rows if r["factor"] == factor and r["value"] == v and r["settings"] == s
                   and r["qid"] in ctx.qids]
            have = {r["qid"] for r in got}
            for q in ctx.qids:
                if q in have:
                    continue
                b = ctx.builder(q, reader, **s)
                p = ctx.plan(b, q, arm, reader)
                ev = ctx.ev[q]
                r = dict(factor=factor, value=v, arm=arm, qid=q, settings=s, **b.ledger(p))
                try:
                    r.update(reader.gold_logprob(b.system, b.render(p)[0], ctx.wavs(b, p, q),
                                                 gold_text(ev["gold_json"], ev["answer_type"])))
                    r["n_in"] = getattr(reader, "n_in", None)
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache(); r["error"] = f"OOM: {e}"[:300]
                jl_append(path, r); got.append(r)
                check_count(got)
            got_by[v] = {r["qid"]: r["lp_sum"] for r in got if "lp_sum" in r}
        common = set.intersection(*(set(g) for g in got_by.values()))
        assert common, f"screen {factor}: no item scored under every value"
        for v in values:
            scores[v] = float(np.mean([got_by[v][q] for q in sorted(common)]))
        settings[factor] = max(values, key=lambda v: (scores[v], -values.index(v)))
        print("screen", factor, scores, f"on {len(common)}/{len(ctx.qids)} items ->", settings[factor], flush=True)
    json.dump(dict(settings=settings, rule="max mean gold lp_sum; ties -> earlier candidate"),
              open(os.path.join(out, f"frozen_settings{sfx}.json"), "w"), indent=1)
    return settings


def stage_arms(ctx, reader, out, sfx="", arms=ARMS, settings_file=None, gold_lp=True):
    import torch
    settings = json.load(open(settings_file or os.path.join(out, f"frozen_settings{sfx}.json")))["settings"]
    ctx.settings = settings
    path = os.path.join(out, f"arms{sfx}.jsonl")
    prior = jl_read(path)
    attempts = {}
    for r in prior:
        attempts[(r["qid"], r["arm"])] = attempts.get((r["qid"], r["arm"]), 0) + 1
    done = {(r["qid"], r["arm"]) for r in prior if "error" not in r or attempts[(r["qid"], r["arm"])] >= 2}
    for q in ctx.qids:
        new = []
        ev = ctx.ev[q]
        ev_keys = {t["key"] for s in ctx.items[q]["sessions"] if s["session_id"] in ev["evidence_session_ids"]
                   for t in s["turns"] if t["role"] == "user"}
        gold = gold_text(ev["gold_json"], ev["answer_type"])
        for arm in arms:
            if (q, arm) in done:
                continue
            line = 1 if arm.endswith("#line1") else 0
            b = ctx.builder(q, reader, **dict(settings, answer_line=line))
            rec = dict(qid=q, arm=arm, evidence_type=ev["evidence_type"], memory_operation=ev["memory_operation"],
                       answer_type=ev["answer_type"], gold=gold, settings=settings, answer_line=line,
                       attempt=attempts.get((q, arm), 0) + 1)
            try:
                p = ctx.plan(b, q, arm, reader)
                if p is None:
                    rec["error"] = "no plan (G empty or no donor with matching turn count)"
                    rec["attempt"] = 2                              # deterministic: no retry
                    jl_append(path, rec); new.append(rec)
                    continue
                srcs = list(p["audio"].values())
                rec.update(b.ledger(p), budget_gap=(b.target(float(arm.split("@")[1].split("#")[0])) - b.positions(p))
                           if "@" in arm else None, ev_turns=len(ev_keys),
                           ev_kept=sum(s in ev_keys for s in srcs), ev_any=any(s in ev_keys for s in srcs))
                user, wavs = b.render(p)[0], ctx.wavs(b, p, q)
                torch.cuda.reset_peak_memory_stats()
                rec.update(reader.generate(b.system, user, wavs, settings.get("max_new", MAX_NEW)))
                if gold_lp:
                    rec.update(reader.gold_logprob(b.system, user, wavs, gold))
                rec["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache(); rec["error"] = f"OOM: {e}"[:300]
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
            jl_append(path, rec)
            new.append(rec)
        print("arms done", q, flush=True)
        check_count(new)
        real = [r for r in new if not r.get("error", "").startswith("no plan")]
        if real and all("error" in r for r in real):    # a systematic failure: stop before spending the GPU budget
            raise RuntimeError(f"every arm failed on {q}: {real[0]['error']}")


def ceiling_subset(ctx):
    """16 families per evidence type, in hash order: the strong-reader ceiling subset (PREREG_HOLDOUT §4)."""
    out = []
    for t in ("speaker_information", "paralinguistic_information", "environmental_sound"):
        qs = sorted((q for q in ctx.qids if ctx.ev[q]["evidence_type"] == t), key=lambda q: hkey("ceiling", q))
        out += qs[:CEILING_PER_TYPE]
    return out


def stage_plans(ctx, out, settings_file):
    """Export the ceiling subset's rendered prompts (MiniCPM-o budgets) and its clips as 16 kHz WAV for a reader
    run elsewhere. Tokenizer only; no model weights."""
    import soundfile as sf
    from transformers import AutoTokenizer
    from memory import MiniCPMCounter
    from reader import MiniCPM

    class Tok:
        counter = MiniCPMCounter(AutoTokenizer.from_pretrained(MiniCPM.NAME, trust_remote_code=True))
    ctx.settings = json.load(open(settings_file))["settings"]
    d = os.path.join(out, "plans")
    os.makedirs(os.path.join(d, "audio"), exist_ok=True)
    with open(os.path.join(d, "plans.jsonl"), "w") as f:
        for q in ceiling_subset(ctx):
            b = ctx.builder(q, Tok, **ctx.settings)
            ev = ctx.ev[q]
            for arm in CEILING_ARMS:
                p = ctx.plan(b, q, arm, Tok)
                user, clips = b.render(p)
                names = []
                for c in clips:
                    name = f"{q}__question.wav" if c == "__question__" else c.replace("#", "_") + ".wav"
                    path = os.path.join(d, "audio", name)
                    if not os.path.exists(path):
                        sf.write(path, ctx.wav(c, q), 16000, subtype="PCM_16")
                    names.append(name)
                f.write(json.dumps(dict(qid=q, arm=arm, system=b.system, user=user, clips=names,
                                        evidence_type=ev["evidence_type"], answer_type=ev["answer_type"],
                                        gold=gold_text(ev["gold_json"], ev["answer_type"]), **b.ledger(p))) + "\n")
    print("plans", len(ceiling_subset(ctx)), "families", flush=True)


def stage_minicpm(ctx, out):
    from reader import minicpm_positions_per_second
    keys = [t["key"] for q in ctx.qids[:4] for s in ctx.items[q]["sessions"] for t in s["turns"]
            if t["role"] == "user"][:20]
    r = minicpm_positions_per_second([ctx.wav(k) for k in keys])
    json.dump(r, open(os.path.join(out, "minicpm_positions.json"), "w"), indent=1)
    print("minicpm", r, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("feat"); ap.add_argument("out")
    ap.add_argument("--stage", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reader", default="ultravox", choices=list(SUFFIX))
    ap.add_argument("--arms", default="dev", choices=["dev", "test", "holdout"]); ap.add_argument("--settings")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ctx = Ctx(a.data, a.feat, a.out, a.limit)
    if a.stage == "minicpm":
        if os.path.exists(os.path.join(a.out, "minicpm_positions.json")):
            return print("minicpm: already measured")
        return stage_minicpm(ctx, a.out)
    if a.stage == "plans":
        return stage_plans(ctx, a.out, a.settings)
    if a.stage == "selftags" and not pending_selftags(ctx, a.out):
        return print("selftags: every turn already tagged")
    sfx = SUFFIX[a.reader] if a.stage != "selftags" else ""
    if sfx:
        from reader import MiniCPM
        reader = MiniCPM()
    else:
        from reader import Ultravox
        reader = Ultravox()
    json.dump(reader.info, open(os.path.join(a.out, f"reader_{a.stage}{sfx}.json"), "w"), indent=1)
    if a.stage == "selftags":
        stage_selftags(ctx, reader, a.out)
    elif a.stage == "screen":
        stage_screen(ctx, reader, a.out, sfx)
    elif a.stage == "arms":
        test = a.arms != "dev"
        assert not test or a.settings, "--arms test/holdout needs the frozen --settings file"
        stage_arms(ctx, reader, a.out, sfx, dict(dev=ARMS, test=TEST_ARMS, holdout=HOLDOUT_ARMS)[a.arms], a.settings,
                   gold_lp=not test)


if __name__ == "__main__":
    main()
