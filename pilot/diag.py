"""Instrument check after the dev pilot (DEV_RESULTS.md): does a reader use old audio at all, and is the flat
memory layout what holds Ultravox at floor? Dev split only; generation only. Resumable.

    python diag.py DATA FEAT OUT --reader ultravox|minicpm

Arms N, A, T*@0.25, F, oracle with the frozen-candidate settings from the pilot (OUT/frozen_settings.json).
Plans are built exactly as in the pilot (Ultravox position budgets); only the rendering and the reader change:
  ultravox: chat layout (one message per turn, as in the Phase 0 smoke)    — flat layout is in arms.jsonl
  minicpm:  flat layout and chat layout
Rows go to OUT/diag.jsonl with arm "<reader>:<layout>:<arm>".
"""
import argparse
import json
import os

from common import gold_text
from run_dev import Ctx, jl_append, jl_read

ARMS = ["N", "A", "T@0.25", "F", "oracle"]
LAYOUTS = dict(ultravox=["chat"], minicpm=["flat", "chat"])
MAX_NEW = 64


class Plans:
    """Ultravox tokenizer only: the position budgets that define T*, as in the pilot."""
    def __init__(self):
        from transformers import AutoProcessor
        from memory import Counter
        from reader import GATED, M, MIRROR, redirect_gated
        redirect_gated(GATED if os.environ.get("HF_TOKEN") else MIRROR)
        self.counter = Counter(AutoProcessor.from_pretrained(M, trust_remote_code=True).tokenizer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("feat"); ap.add_argument("out"); ap.add_argument("--reader", required=True)
    a = ap.parse_args()
    ctx = Ctx(a.data, a.feat, a.out, 0)
    ctx.settings = json.load(open(os.path.join(a.out, "frozen_settings.json")))["settings"]
    if a.reader == "ultravox":
        from reader import Ultravox
        reader = plans = Ultravox()
    else:
        from reader import MiniCPM
        plans, reader = Plans(), MiniCPM()
    json.dump(reader.info, open(os.path.join(a.out, f"reader_diag_{a.reader}.json"), "w"), indent=1)
    path = os.path.join(a.out, "diag.jsonl")
    done = {(r["qid"], r["arm"]) for r in jl_read(path) if "error" not in r}
    for q in ctx.qids:
        ev = ctx.ev[q]
        b = ctx.builder(q, plans, **ctx.settings)
        new = []
        for layout in LAYOUTS[a.reader]:
            for arm in ARMS:
                name = f"{a.reader}:{layout}:{arm}"
                if (q, name) in done:
                    continue
                rec = dict(qid=q, arm=name, evidence_type=ev["evidence_type"], answer_type=ev["answer_type"],
                           gold=gold_text(ev["gold_json"], ev["answer_type"]))
                try:
                    p = ctx.plan(b, q, arm, plans)
                    if layout == "chat":
                        msgs, clips = b.render_chat(p)
                        user = None
                    else:
                        (user, clips), msgs = b.render(p), None
                    rec.update(reader.generate(b.system, user, [ctx.wav(c, q) for c in clips], MAX_NEW, msgs=msgs),
                               n_audio=len(clips) - 1)
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {e}"[:500]
                jl_append(path, rec); new.append(rec)
        print("diag done", q, flush=True)
        if new and all("error" in r for r in new):       # a systematic failure: stop before spending the GPU budget
            raise RuntimeError(f"every diag arm failed on {q}: {new[0]['error']}")


if __name__ == "__main__":
    main()
