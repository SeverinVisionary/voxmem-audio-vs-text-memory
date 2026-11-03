"""Second positive control (PREREG_POSCTRL2.md §3), committed before the run.

    python analyze_posctrl2.py MINICPM_ARMS MINICPM_VERDICTS GEMINI_ARMS GEMINI_VERDICTS OUT_JSON   # run from pilot/

Per reader, each test passes when the type-stratified cluster bootstrap-t lower bound (one-sided 0.025) is above 0.
MiniCPM-o: P1 Ev-A − Ev-M, then P2 O − M:asr only if P1 passes (all arms from this run). Gemini: P1 only.
No correction across readers: a correction would make a pass harder, and the claim at stake is that none passes.
Errors after the rerun are scored wrong; a reader with any family missing an arm is reported, not analysed.
"""
import json
import sys

import numpy as np

from analyze_posctrl import CLUSTERS, unstratified
from analyze_test import TYPES, contrast, load

READERS = {"minicpm": ("N", "M:asr", "Ev-A", "Ev-M", "O"), "gemini": ("Ev-A", "Ev-M")}
V3_MINICPM = "out_holdout/arms_minicpm.jsonl"


def test(acc, a, b, qids, types, comp, seed):
    c = contrast(acc, a, b, qids, types, comp, seed)
    c["unstratified"] = unstratified(acc, a, b, qids, types, comp, seed)
    return c


def reader(arms, ver, names, comp, seed):
    acc, types, failed = load(arms, ver)
    qids = sorted(types)
    missing = {a: len(qids) - len(acc[a]) for a in names if len(acc[a]) != len(qids)}
    res = {"n_families": len(qids), "failed": {a: failed[a] for a in names}, "missing": missing,
           "acc": {a: round(100 * np.mean([acc[a][q] or 0 for q in qids]), 2) for a in names if a in acc},
           "acc_by_type": {t: {a: round(100 * np.mean([acc[a][q] or 0 for q in qids if types[q] == t]), 2)
                               for a in names if a in acc} for t in TYPES}}
    if missing:
        res["P1"] = "INCOMPLETE"
        return res, acc, qids
    p1 = test(acc, "Ev-A", "Ev-M", qids, types, comp, seed)
    pass1 = p1["L_pp"] > 0
    res.update({"P1 Ev-A-Ev-M": p1, "P1": "HEARS-OLD-CLIPS" if pass1 else "NOT-SHOWN",
                "by_type_P1": {t: contrast(acc, "Ev-A", "Ev-M", [q for q in qids if types[q] == t], types, comp,
                                           seed + 10 + i) for i, t in enumerate(TYPES)}})
    if "O" in names:
        p2 = test(acc, "O", "M:asr", qids, types, comp, seed + 1)
        res.update({"P2 O-M:asr": p2,
                    "P2": ("USES-RIGHT-CLIPS" if p2["L_pp"] > 0 else "NOT-SHOWN") if pass1 else "NOT-TESTED (estimate only)",
                    "Ev-A-N": contrast(acc, "Ev-A", "N", qids, types, comp, seed + 2),
                    "Ev-M-N": contrast(acc, "Ev-M", "N", qids, types, comp, seed + 3)})
    return res, acc, qids


def main():
    m_arms, m_ver, g_arms, g_ver, out = sys.argv[1:6]
    comp = json.load(open(CLUSTERS))["component"]
    res = {}
    res["minicpm"], _, _ = reader(m_arms, m_ver, READERS["minicpm"], comp, 4001)
    res["gemini"], _, _ = reader(g_arms, g_ver, READERS["gemini"], comp, 4101)
    # descriptive: this run's MiniCPM-o N and M:asr answers against v3's (same plans, WAV round trip)
    v3 = {(r["qid"], r["arm"]): r.get("answer") for r in map(json.loads, open(V3_MINICPM))}
    new = {}
    for r in map(json.loads, open(m_arms)):
        new[(r["qid"], r["arm"])] = r.get("answer")
    res["minicpm"]["same_answer_as_v3"] = {
        a: round(float(np.mean([str(new[k]).strip().lower() == str(v3.get(k)).strip().lower()
                                for k in new if k[1] == a])), 3) for a in ("N", "M:asr")}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
