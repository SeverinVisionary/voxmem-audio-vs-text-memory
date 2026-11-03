"""MiniCPM-o 4.5 on sealed exported plans (PREREG_POSCTRL2.md). GPU. Resumable.

    python minicpm_plans_reader.py PLANS_DIR OUT_JSONL [--limit N]

PLANS_DIR holds plans.jsonl and audio/ (16 kHz PCM WAV; no gold). Each row goes through reader.MiniCPM.generate,
the same chat() call, flat layout and greedy decoding as the v3 MiniCPM-o arms, with max_new from the Phase 1
frozen settings. A row that fails is logged with an error and retried on the next start; one still failing after
the rerun stage is scored wrong.
"""
import argparse
import json
import os

from reader import MiniCPM

MAX_NEW = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_minicpm.json")))["settings"]["max_new"]


def main():
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("plans"); ap.add_argument("out"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    audio = os.path.join(a.plans, "audio")
    rows = [json.loads(l) for l in open(os.path.join(a.plans, "plans.jsonl"))]
    assert not any("gold" in r for r in rows), "gold in plans"
    done = set()
    if os.path.exists(a.out):
        done = {(r["qid"], r["arm"]) for r in map(json.loads, open(a.out)) if "error" not in r}
    todo = [r for r in rows if (r["qid"], r["arm"]) not in done][:a.limit]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    rd = MiniCPM()
    print(json.dumps(rd.info), flush=True)
    torch = rd.torch
    streak = 0
    for row in todo:
        rec = {k: row[k] for k in ("qid", "arm", "evidence_type", "answer_type")}
        rec.update(model=MiniCPM.NAME, n_clips=len(row["clips"]))
        try:
            wavs = []
            for c in row["clips"]:
                w, sr = sf.read(os.path.join(audio, c), dtype="float32")
                assert sr == 16000, (c, sr)
                wavs.append(w)
            torch.cuda.reset_peak_memory_stats()
            rec.update(rd.generate(row["system"], row["user"], wavs, MAX_NEW))
            rec["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        except Exception as e:  # noqa: BLE001  (OOM): logged, retried on the next start
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            torch.cuda.empty_cache()
        streak = streak + 1 if "error" in rec else 0
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(row["qid"], row["arm"], rec.get("seconds"), rec.get("n_in"), rec.get("error", "")[:80], flush=True)
        if streak >= 5:
            raise SystemExit("5 failures in a row: stopping (a broken setup, not a row problem)")


if __name__ == "__main__":
    main()
