"""Post hoc sensitivity for P1-Gemini (PREREG_POSCTRL2.md §6): drop every family whose Ev-M run gave no answer.

    python analyze_posctrl2_sensitivity.py GEMINI_ARMS GEMINI_VERDICTS OUT_JSON   # run from pilot/

A non-answer is a row still failing after the retry (invalid run) or one with an empty reply. All 12 are Ev-M,
where scoring them wrong makes P1 easier to pass; this reruns P1 on the remaining families, same seed.
"""
import json
import os
import sys
import tempfile

from analyze_posctrl2 import READERS, reader
from analyze_posctrl import CLUSTERS


def main():
    arms, ver, out = sys.argv[1:4]
    rows = [json.loads(l) for l in open(arms)]
    last = {(r["qid"], r["arm"]): r for r in rows}
    bad = sorted({q for (q, a), r in last.items() if "error" in r or not r.get("answer")})
    with tempfile.TemporaryDirectory() as t:
        a2, v2 = os.path.join(t, "arms.jsonl"), os.path.join(t, "ver.jsonl")
        open(a2, "w").writelines(json.dumps(r) + "\n" for r in rows if r["qid"] not in bad)
        open(v2, "w").writelines(l for l in open(ver) if json.loads(l)["qid"] not in bad)
        res, _, _ = reader(a2, v2, READERS["gemini"], json.load(open(CLUSTERS))["component"], 4101)
    res = json.loads(json.dumps(dict(dropped=bad, **res), default=float))
    json.dump(res, open(out, "w"), indent=1)
    print(len(bad), "families dropped;", res["P1"], res["P1 Ev-A-Ev-M"]["diff_pp"], res["P1 Ev-A-Ev-M"]["L_pp"])


if __name__ == "__main__":
    main()
