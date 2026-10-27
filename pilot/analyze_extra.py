"""Exploratory analyses of the Phase 1 test run (written after the primary analysis; labelled exploratory).

    python analyze_extra.py ARMS_JSONL VERDICTS_JSONL CLUSTERS_JSON OUT_JSON

1. Frontier: accuracy against memory size on both BYTE_ACCOUNTING ledgers. KV-resident bytes are exact (memory
   positions × 147,456 B for MiniCPM-o). Storage-form bytes are approximate: text ≈ 4 B per text position
   (UTF-8), audio = audio positions / 10.55 positions-per-second × 32,000 B (16 kHz PCM16).
2. Equivalence: the smallest margin δ for which text memory is non-inferior to old audio (ΔG upper bound).
3. Strata: G − T*, F − T*, A − T* by evidence type, memory operation, answer type and evidence recency
   (VoxMem evidence_relative_positions: 0 = start of history, 1 = just before the question).
"""
import csv
import json
import statistics as st
import sys

import numpy as np

from analyze_test import TYPES, contrast, load
from common import Q2

KV_B = 147_456
AUDIO_POS_PER_S = 10.55
PCM_B_PER_S = 32_000
TEXT_B_PER_POS = 4


def main():
    arms_path, ver_path, cl_path, out = sys.argv[1:5]
    acc, types, _ = load(arms_path, ver_path)
    comp = json.load(open(cl_path))["component"]
    qids = sorted(types)
    rows = {}
    for line in open(arms_path):
        r = json.loads(line)
        if "error" not in r:
            rows[(r["qid"], r["arm"])] = r
    res = dict(frontier={})
    for a in sorted(acc):
        rs = [rows[(q, a)] for q in qids if (q, a) in rows]
        xs = [v for v in acc[a].values() if v is not None]
        mem = [r["memory"] for r in rs]
        aud = [r["audio"] for r in rs]
        txt = [r["memory"] - r["audio"] for r in rs]
        store = [t * TEXT_B_PER_POS + au / AUDIO_POS_PER_S * PCM_B_PER_S for t, au in zip(txt, aud)]
        m = float(np.mean(xs))
        res["frontier"][a] = dict(
            acc=round(100 * m, 2), se=round(100 * np.sqrt(m * (1 - m) / len(xs)), 2),
            memory_positions_median=st.median(mem), audio_share_median=round(st.median(
                [au / max(1, mm) for au, mm in zip(aud, mem)]), 3),
            kv_mib_median=round(st.median(mem) * KV_B / 2 ** 20, 1),
            storage_kib_median_approx=round(st.median(store) / 1024, 1))
    g = contrast(acc, "G@0.25", "T@0.25", qids, types, comp, 202)
    res["equivalence"] = dict(dG_U_pp=g["U_pp"], dG_L_pp=g["L_pp"],
                              statement=f"text memory non-inferior to old audio at margin {g['U_pp']} pp "
                                        f"(95% two-sided bootstrap-t upper bound of ΔG@0.25)")
    meta = {r["question_id"]: r for r in csv.DictReader(open(Q2 / "phase0/out/vox_items_8k.csv"))}
    strata = {}
    for q in qids:
        m = meta[q]
        pos = json.loads(m["evidence_relative_positions"] or "[]")
        rec = "recent (≥ 0.67)" if pos and max(pos) >= 0.67 else "middle" if pos and max(pos) >= 0.33 else "early (< 0.33)"
        strata[q] = dict(type=types[q], operation=m["memory_operation"], answer=m["answer_type"] or "none",
                         recency=rec)
    res["strata"] = {}
    for key in ("type", "operation", "answer", "recency"):
        for val in sorted({s[key] for s in strata.values()}):
            qs = [q for q in qids if strata[q][key] == val]
            if len(qs) < 15:
                continue
            res["strata"][f"{key}={val}"] = dict(n=len(qs), **{
                name: {k: v for k, v in contrast(acc, a, b, qs, types, comp, 11 + i).items()
                       if k in ("diff_pp", "L_pp", "U_pp", "discordance")}
                for i, (name, a, b) in enumerate([("G-T@0.25", "G@0.25", "T@0.25"), ("F-T@0.25", "F", "T@0.25"),
                                                  ("A-T@0.25", "A", "T@0.25")])},
                acc={a: round(100 * np.mean([acc[a][q] for q in qs if acc[a].get(q) is not None]), 1)
                     for a in ("N", "A", "T@0.25", "G@0.25", "F")})
    json.dump(res, open(out, "w"), indent=1)
    for a, v in res["frontier"].items():
        print(f"{a:10s} acc {v['acc']:5.1f}±{v['se']:.1f}  mem {v['memory_positions_median']:6}  audio "
              f"{v['audio_share_median']:.2f}  KV {v['kv_mib_median']:6.1f} MiB  store≈{v['storage_kib_median_approx']:7.1f} KiB")
    print(res["equivalence"]["statement"])
    for k, v in res["strata"].items():
        print(f"{k:42s} n={v['n']:3d} " + " ".join(f"{n} {c['diff_pp']:+.1f} [{c['L_pp']},{c['U_pp']}]"
                                                    for n, c in v.items() if isinstance(c, dict) and "diff_pp" in c),
              v["acc"])


if __name__ == "__main__":
    main()
