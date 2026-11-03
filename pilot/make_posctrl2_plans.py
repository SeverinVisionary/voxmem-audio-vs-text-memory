"""Second positive control's plans (PREREG_POSCTRL2.md §2), from two sealed plan sets. CPU, no gold.

    python make_posctrl2_plans.py CLEAN_DIR POSCTRL_DIR OUT_DIR   # unzipped plans_clean.zip and plans_posctrl.zip

Writes OUT_DIR/minicpm (N, M:asr from the clean plans; Ev-A, Ev-M, O from the positive-control plans) and
OUT_DIR/gemini (Ev-A, Ev-M), each plans.jsonl plus audio/ with only the clips its rows use. Families are in hash
order and a family's arms are adjacent, so a run cut short leaves a seeded subset of whole families.
"""
import json
import os
import shutil
import sys

from common import hkey

SETS = {"minicpm": ("N", "M:asr", "Ev-A", "Ev-M", "O"), "gemini": ("Ev-A", "Ev-M")}


def main():
    clean, posctrl, out = sys.argv[1:]
    rows, src = {}, {}
    for d, arms in ((clean, ("N", "M:asr")), (posctrl, ("Ev-A", "Ev-M", "O"))):
        for r in map(json.loads, open(os.path.join(d, "plans.jsonl"))):
            assert "gold" not in r, "gold in plans"
            if r["arm"] in arms:
                rows[(r["qid"], r["arm"])] = r
                src[(r["qid"], r["arm"])] = d
    qids = sorted({q for q, _ in rows}, key=lambda q: hkey("posctrl2-order", q))
    assert len(qids) == 142 and len(rows) == 5 * 142
    for name, arms in SETS.items():
        d = os.path.join(out, name)
        os.makedirs(os.path.join(d, "audio"), exist_ok=True)
        with open(os.path.join(d, "plans.jsonl"), "w") as f:
            for q in qids:
                for arm in arms:
                    r = rows[(q, arm)]
                    f.write(json.dumps(r) + "\n")
                    for c in r["clips"]:
                        dst = os.path.join(d, "audio", c)
                        if not os.path.exists(dst):
                            shutil.copy(os.path.join(src[(q, arm)], "audio", c), dst)
        print(name, len(qids) * len(arms), "rows,", len(os.listdir(os.path.join(d, "audio"))), "clips")


if __name__ == "__main__":
    main()
