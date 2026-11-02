"""Clean Gemini rerun (PREREG_GEMINI_CLEAN.md): C1 primary A - M:asr; C2-C4 estimates; on all 142 holdout families
and on the 94 that the Phase 1b Gemini subset never touched.

    python analyze_gemini_clean.py GEMINI_ARMS GEMINI_VERDICTS OUT_JSON     # run from pilot/
"""
import json
import sys

from analyze_test import contrast, load

MINICPM = ("out_holdout/arms_minicpm.jsonl", "out_holdout/judge/verdicts.jsonl")
OLD_GEMINI = "out_holdout/gemini_arms.jsonl"
CLUSTERS = "out_holdout/clusters.json"
ARMS = ["N", "A", "M:asr", "T@0.25", "G@0.25"]


def acc_table(acc, qids):
    return {a: round(100 * sum(acc[a].get(q) or 0 for q in qids) / len(qids), 2) for a in ARMS if a in acc}


def did(gacc, macc, qids, types, comp, seed):
    both = [q for q in qids if all(d.get(q) is not None for d in (gacc["A"], gacc["M:asr"], macc["A"], macc["M:asr"]))]
    x = {q: (gacc["A"][q] - gacc["M:asr"][q]) - (macc["A"][q] - macc["M:asr"][q]) for q in both}
    return contrast({"x": x, "y": {q: 0 for q in both}}, "x", "y", both, types, comp, seed)


def matcher(ver_path):
    rows = [json.loads(l) for l in open(ver_path)]
    p = ver_path + ".matcher.jsonl"
    with open(p, "w") as f:
        f.writelines(json.dumps(dict(r, j1=r["det"])) + "\n" for r in rows)
    return p


def block(gacc, gtypes, macc, qids, comp, seed):
    if not qids:
        return None
    return {"n_families": len(qids), "acc": acc_table(gacc, qids),
            "C1 A-M:asr": contrast(gacc, "A", "M:asr", qids, gtypes, comp, seed),
            "C2 T@0.25-A": contrast(gacc, "T@0.25", "A", qids, gtypes, comp, seed + 1),
            "C3 G@0.25-T@0.25": contrast(gacc, "G@0.25", "T@0.25", qids, gtypes, comp, seed + 2),
            "A-N": contrast(gacc, "A", "N", qids, gtypes, comp, seed + 3),
            "C4 DiD (A-M:asr) gemini minus minicpm": did(gacc, macc, qids, gtypes, comp, seed + 4)}


def main():
    arms, ver, out = sys.argv[1:4]
    comp = json.load(open(CLUSTERS))["component"]
    gacc, gtypes, gfailed = load(arms, ver)
    macc, _, _ = load(*MINICPM)
    seen = {json.loads(l)["qid"] for l in open(OLD_GEMINI)}
    qids = sorted(gtypes)
    fresh = [q for q in qids if q not in seen]
    rows = [json.loads(l) for l in open(arms)]
    res = {"invalid_runs": dict(gfailed), "invalid_rate": round(sum(gfailed.values()) / max(1, len(rows)), 4),
           "all": block(gacc, gtypes, macc, qids, comp, 2001),
           "never_seen": block(gacc, gtypes, macc, fresh, comp, 2101)}
    # C4 under one common scorer: the deterministic matcher for both readers
    ga, _, _ = load(arms, matcher(ver))
    ma, _, _ = load(MINICPM[0], matcher(MINICPM[1]))
    res["C4 DiD under matcher (all)"] = did(ga, ma, qids, gtypes, comp, 2201)
    c1 = res["all"]["C1 A-M:asr"]
    res["C1_decision"] = "AUDIO-BEATS-TRANSCRIPT" if c1["L_pp"] > 0 else "NOT-REPLICATED"
    c2 = res["all"]["C2 T@0.25-A"]
    res["C2_reading"] = ("WITHIN-10" if c2["L_pp"] > -10 and c2["U_pp"] < 10 else
                         "TEXT-WORSE" if c2["U_pp"] < 0 else "TEXT-BETTER" if c2["L_pp"] > 0 else "UNRESOLVED")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:5000])


if __name__ == "__main__":
    main()
