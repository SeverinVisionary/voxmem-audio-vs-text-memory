"""Positive control (PREREG_POSCTRL.md §3), committed before the run.

    python analyze_posctrl.py POSCTRL_ARMS POSCTRL_VERDICTS OUT_JSON     # run from pilot/

Ev-A, Ev-M and O come from the positive-control run; N, A and M:asr from the v3 Qwen3-Omni run. Fixed sequence:
P1 Ev-A − Ev-M, then P2 O − M:asr only if P1 passes; each passes when the type-stratified cluster bootstrap-t
lower bound (one-sided 0.025) is above 0. Errors after the rerun are scored wrong.
"""
import json
import sys

import numpy as np

from analyze_sensitivity import boot
from analyze_test import TYPES, Clusters, cluster_sums, contrast, load, paired

V3 = ("out_omni/arms_omni.jsonl", "out_omni/judge/verdicts.jsonl")
CLUSTERS = "out_holdout/clusters.json"
ARMS = ["N", "A", "M:asr", "Ev-A", "Ev-M", "O"]


def unstratified(acc, a, b, qids, types, comp, seed):
    qs, d = paired(acc, a, b, qids)
    cl = Clusters(qs, types, comp)
    s, n = cluster_sums(d, cl.of, len(cl.names))
    r = boot(s, n, np.zeros(len(s), int), seed)
    return dict(L_pp=r["L_pp"], U_pp=r["U_pp"])


def test(acc, a, b, qids, types, comp, seed):
    c = contrast(acc, a, b, qids, types, comp, seed)
    c["unstratified"] = unstratified(acc, a, b, qids, types, comp, seed)
    return c


def main():
    arms, ver, out = sys.argv[1:4]
    comp = json.load(open(CLUSTERS))["component"]
    acc, types, failed = load(*V3)
    pacc, ptypes, pfailed = load(arms, ver)
    assert ptypes == types
    acc = {a: acc[a] for a in ("N", "A", "M:asr")} | {a: pacc[a] for a in ("Ev-A", "Ev-M", "O")}
    qids = sorted(types)
    assert all(len(acc[a]) == len(qids) for a in ARMS), {a: len(acc[a]) for a in ARMS}
    p1 = test(acc, "Ev-A", "Ev-M", qids, types, comp, 3001)
    p2 = test(acc, "O", "M:asr", qids, types, comp, 3002)
    pass1 = p1["L_pp"] > 0
    pass2 = pass1 and p2["L_pp"] > 0
    x = {q: (acc["Ev-A"][q] - acc["Ev-M"][q]) - (acc["A"][q] - acc["M:asr"][q]) for q in qids}
    res = {
        "failed": {a: pfailed[a] for a in ("Ev-A", "Ev-M", "O")},
        "acc": {a: round(100 * np.mean([acc[a][q] or 0 for q in qids]), 2) for a in ARMS},
        "acc_by_type": {t: {a: round(100 * np.mean([acc[a][q] or 0 for q in qids if types[q] == t]), 2) for a in ARMS}
                        for t in TYPES},
        "P1 Ev-A-Ev-M": p1, "P1": "HEARS-OLD-CLIPS" if pass1 else "NOT-SHOWN",
        "P2 O-M:asr": p2, "P2": "USES-RIGHT-CLIPS" if pass2 else ("NOT-SHOWN" if pass1 else "NOT-TESTED (estimate only)"),
        "length cost (Ev-A-Ev-M)-(A-M:asr)": contrast({"x": x, "y": {q: 0 for q in qids}}, "x", "y", qids, types, comp,
                                                       3003),
        "Ev-A-N": contrast(acc, "Ev-A", "N", qids, types, comp, 3004),
        "Ev-M-N": contrast(acc, "Ev-M", "N", qids, types, comp, 3005),
        "by_type": {t: {"P1": contrast(acc, "Ev-A", "Ev-M", [q for q in qids if types[q] == t], types, comp, 3010 + i),
                        "P2": contrast(acc, "O", "M:asr", [q for q in qids if types[q] == t], types, comp, 3020 + i)}
                    for i, t in enumerate(TYPES)},
    }
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
