"""Phase 1b analysis (PREREG_HOLDOUT.md §3-4), committed before the holdout run.

    python analyze_holdout.py HOLDOUT_ARMS VERDICTS CLUSTERS OUT_JSON [GEMINI_ARMS GEMINI_VERDICTS]

Same estimators as analyze_test.py: family-weighted paired differences, cluster sign-flip p (one-sided) and a
type-stratified cluster bootstrap-t 95% interval over clusters.py units.
"""
import json
import sys

from analyze_test import TYPES, contrast, load

MARGIN_PP = 10.0
ATTRIBUTION = {"speaker_information": "M:D-spk", "paralinguistic_information": "M:D-emo",
               "environmental_sound": "M:D-env"}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    arms, ver, cl, out = sys.argv[1:5]
    acc, types, failed = load(arms, ver)
    comp = json.load(open(cl))["component"]
    qids = sorted(types)
    res = dict(n_families=len(qids), failed=dict(failed),
               acc={a: round(100 * sum(v for v in d.values() if v is not None) /
                             max(1, sum(v is not None for v in d.values())), 2) for a, d in sorted(acc.items())})
    # B1 (primary): transcript-only memory non-inferior to full audio at margin 10 pp
    b1 = contrast(acc, "M:asr", "A", qids, types, comp, 1001)
    res["B1 M:asr-A"] = b1
    res["B1_decision"] = ("TRANSCRIPT-SUFFICES" if b1["L_pp"] > -MARGIN_PP else
                          "AUDIO-NEEDED" if b1["U_pp"] < 0 else "INCONCLUSIVE")
    # B2 (secondary, Holm over the three types): the matched field group adds to the transcript
    b2 = {}
    for i, t in enumerate(TYPES):
        qt = [q for q in qids if types[q] == t]
        b2[t] = contrast(acc, ATTRIBUTION[t], "M:asr", qt, types, comp, 1100 + i)
    for t, p in zip(TYPES, holm([b2[t]["p_one_sided"] for t in TYPES])):
        b2[t]["p_holm"] = round(p, 5)
    res["B2 field-minus-transcript"] = b2
    res["B2 all fields M:D-M:asr"] = contrast(acc, "M:D", "M:asr", qids, types, comp, 1200)
    # B3 (secondary): Phase 1 replication on the holdout, estimates only
    res["B3 F-T@0.25"] = contrast(acc, "F", "T@0.25", qids, types, comp, 1301)
    res["B3 G-T@0.25"] = contrast(acc, "G@0.25", "T@0.25", qids, types, comp, 1302)
    res["A-N"] = contrast(acc, "A", "N", qids, types, comp, 1303)
    # C (ceiling): Gemini on the 48-family subset, and the same contrasts for MiniCPM-o on those families
    if len(sys.argv) > 6:
        gacc, gtypes, gfailed = load(sys.argv[5], sys.argv[6])
        sub = sorted(gtypes)
        c = dict(n_families=len(sub), failed=dict(gfailed),
                 acc={a: round(100 * sum(v for v in d.values() if v is not None) /
                               max(1, sum(v is not None for v in d.values())), 2) for a, d in sorted(gacc.items())})
        for name, a, b in [("A-M:asr", "A", "M:asr"), ("F-T@0.25", "F", "T@0.25"), ("G-T@0.25", "G@0.25", "T@0.25"),
                           ("A-N", "A", "N")]:
            c[f"gemini {name}"] = contrast(gacc, a, b, sub, gtypes, comp, 1400)
            c[f"minicpm {name} (same families)"] = contrast(acc, a, b, sub, types, comp, 1401)
        # difference in differences: does the strong reader gain more from audio over transcript?
        both = [q for q in sub if all(d.get(q) is not None for d in (gacc["A"], gacc["M:asr"], acc["A"], acc["M:asr"]))]
        dd = {"dd": {q: (gacc["A"][q] - gacc["M:asr"][q]) - (acc["A"][q] - acc["M:asr"][q]) for q in both}}
        dd["zero"] = {q: 0 for q in both}
        c["DiD (A-M:asr) gemini minus minicpm"] = contrast(
            {"x": {q: v for q, v in dd["dd"].items()}, "y": dd["zero"]}, "x", "y", both, types, comp, 1500)
        res["C ceiling"] = c
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "acc"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
