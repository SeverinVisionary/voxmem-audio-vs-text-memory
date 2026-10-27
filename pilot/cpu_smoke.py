"""CPU smoke for the dev pilot (no GPU, no 8B weights). Run after data.py and features.py on a few items.

    python cpu_smoke.py DATA FEAT OUT

1. Counter (tokenizer-only position count) == real UltravoxProcessor input length on rendered arms.
2. run_dev screen + arms stages end to end with a stub reader (fake generation / log-probs).
Writes OUT/cpu_smoke.json; exits non-zero on any failed check.
"""
import collections
import json
import os
import sys
import traceback

import run_dev
from memory import Counter
from reader import GATED, M, MIRROR, redirect_gated


class StubReader:
    def __init__(self):
        from transformers import AutoProcessor
        redirect_gated(GATED if os.environ.get("HF_TOKEN") else MIRROR)
        self.proc = AutoProcessor.from_pretrained(M, trust_remote_code=True)
        self.tok = self.proc.tokenizer
        self.counter = Counter(self.tok)
        self.info = dict(stub=True)

    def real_len(self, system, user, wavs):
        text = self.counter.render(system, user)
        kw = dict(text=text, return_tensors="pt", sampling_rate=16000)
        kw["audio" if len(wavs) == 1 else "audios"] = wavs[0] if len(wavs) == 1 else wavs
        return int(self.proc(**kw)["input_ids"].shape[1])

    def generate(self, system, user, wavs, max_new):
        return dict(answer="stub", n_new=1, hit_max=False, seconds=0.0, n_in=self.real_len(system, user, wavs))

    def gold_logprob(self, system, user, wavs, gold):
        return dict(lp_sum=-float(len(user) % 97), lp_mean=-1.0, n_gold=len(self.tok(gold)["input_ids"]))


def main():
    data, feat, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    res, ok = {}, True
    try:
        ctx = run_dev.Ctx(data, feat, out, 6)          # 6 items keeps the smoke to a few minutes
        r = StubReader()
        checks = []
        for q in ctx.qids:
            ctx.settings = dict(run_dev.DEFAULTS)
            b = ctx.builder(q, r)
            for arm in ["N", "A", "F", "M:D", "M:Dp", "T@0.25", "G@0.25", "E@0.1", "G@0.5"]:
                p = ctx.plan(b, q, arm, r)
                user, clips = b.render(p)
                mine, real = b.positions(p), r.real_len(b.system, user, [ctx.wav(c, q) for c in clips])
                checks.append(dict(qid=q, arm=arm, mine=mine, real=real, target=b.target(0.25), P_A=b.P_A))
        res["counter_checks"] = checks
        bad = [c for c in checks if c["mine"] != c["real"]]
        res["counter_mismatches"] = bad
        ok &= not bad
        run_dev.FACTORS[0] = ("writer", ["D", "Dp"], "T@0.25")        # writer C needs GPU self-tags
        settings = run_dev.stage_screen(ctx, r, out)
        res["screen_settings"] = settings
        _patch_cuda()
        run_dev.stage_arms(ctx, r, out)
        rows = run_dev.jl_read(os.path.join(out, "arms.jsonl"))
        res["arms_rows"] = len(rows)
        # writer-C self-tags come from the GPU selftags stage, so M:C cannot run here
        no_tags = [x for x in rows if x["arm"] == "M:C" and "writer C: no self-tag" in x.get("error", "")]
        res["arms_errors"] = [x for x in rows if "error" in x and not x["error"].startswith("no plan")
                              and x not in no_tags][:10]
        res["arms_no_plan"] = sum(1 for x in rows if x.get("error", "").startswith("no plan"))
        res["arms_no_selftags"] = len(no_tags)
        res["arms_error_kinds"] = dict(collections.Counter(f'{x["arm"]}: {x["error"][:60]}' for x in rows if "error" in x))
        ok &= not res["arms_errors"] and len(rows) > 0
        res["example_render"] = b.render(ctx.plan(b, ctx.qids[0], "G@0.25", r))[0][:3000]
    except Exception:
        res["exception"] = traceback.format_exc()[-3000:]
        ok = False
    res["ok"] = ok
    json.dump(res, open(os.path.join(out, "cpu_smoke.json"), "w"), indent=1, default=str)
    print("CPU SMOKE", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def _patch_cuda():
    import torch
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.empty_cache = lambda *a, **k: None


if __name__ == "__main__":
    main()
