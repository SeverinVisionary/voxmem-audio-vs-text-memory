"""Post hoc scoring sensitivity for the positive control (written after the results, not preregistered).

    python analyze_posctrl_scoring.py OUT_JSON      # run from pilot/; writes nothing else

P1 Ev-A − Ev-M and P2 O − M:asr with judge 2's verdict wherever judge 2 saw the answer, and with the
deterministic matcher alone, for both the v3 rows (N, A, M:asr) and the positive-control rows. Same seeds as
analyze_posctrl.py.
"""
import json
import sys

from analyze_posctrl import CLUSTERS, V3
from analyze_test import contrast, load

POSCTRL = ("out_posctrl/arms_posctrl_gold.jsonl", "out_posctrl/judge/verdicts.jsonl")
MODES = {"judge2_where_seen": lambda r: r["j1"] if r.get("j2") is None else r["j2"], "matcher": lambda r: r["det"]}


def rescored(arms, ver, key, tmp):
    with open(tmp, "w") as f:
        f.writelines(json.dumps(dict(r, j1=key(r))) + "\n" for r in map(json.loads, open(ver)))
    return load(arms, tmp)


def main():
    import os
    import tempfile
    out = sys.argv[1]
    comp = json.load(open(CLUSTERS))["component"]
    res = {}
    with tempfile.TemporaryDirectory() as d:
        for mode, key in MODES.items():
            acc, types, _ = rescored(*V3, key, os.path.join(d, "v3.jsonl"))
            pacc, _, _ = rescored(*POSCTRL, key, os.path.join(d, "posctrl.jsonl"))
            acc = {a: acc[a] for a in ("N", "A", "M:asr")} | {a: pacc[a] for a in ("Ev-A", "Ev-M", "O")}
            qids = sorted(types)
            res[mode] = {"P1 Ev-A-Ev-M": contrast(acc, "Ev-A", "Ev-M", qids, types, comp, 3001),
                         "P2 O-M:asr": contrast(acc, "O", "M:asr", qids, types, comp, 3002)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
