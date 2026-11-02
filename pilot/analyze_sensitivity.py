"""Post hoc sensitivity checks added for v3 (written after all runs, after the release review). Not preregistered.

    python analyze_sensitivity.py OUT_JSON      # run from pilot/; writes nothing else

1. Qwen3-Omni scoring: C1 A - M:asr and C3 G - T@0.25 with judge 2's verdict wherever judge 2 saw the answer, and
   with the deterministic matcher alone; judge-1/matcher disagreement by arm; judge-2 coverage and agreement, by
   row and by distinct (question, response) pair.
2. Resampling: the three decision contrasts (MiniCPM-o test G - T@0.25, MiniCPM-o holdout B1 M:asr - A, Qwen C1)
   with an unstratified cluster bootstrap-t (same units, same seeds), and the share of stratified bootstrap draws
   whose standard error is zero (analyze_test.boot_t sets their pivot to 0).
"""
import json
import sys

import numpy as np

from analyze_test import ALPHA, B, Clusters, cluster_sums, contrast, est_se, load, paired

OMNI = ("out_omni/arms_omni.jsonl", "out_omni/judge/verdicts.jsonl")
DECISIONS = {  # name: (arms, verdicts, clusters, a, b, seed as in the preregistered script)
    "minicpm test G-T@0.25": ("out_test/arms_minicpm.jsonl", "out_test/judge/verdicts.jsonl",
                              "out_test/clusters.json", "G@0.25", "T@0.25", 202),
    "minicpm holdout B1 M:asr-A": ("out_holdout/arms_minicpm.jsonl", "out_holdout/judge/verdicts.jsonl",
                                   "out_holdout/clusters.json", "M:asr", "A", 1001),
    "qwen C1 A-M:asr": (*OMNI, "out_holdout/clusters.json", "A", "M:asr", 2001),
}


def rows(p):
    return [json.loads(l) for l in open(p)]


def acc_from(arms, ver_rows, key):
    last = {(r["qid"], r["arm"]): r for r in rows(arms)}
    v = {(r["qid"], r["arm"]): key(r) for r in ver_rows}
    acc, types = {}, {}
    for (q, a), r in last.items():
        types[q] = r["evidence_type"]
        acc.setdefault(a, {})[q] = 0 if "error" in r else int(bool(v[(q, a)]))
    return acc, types


def boot(s, n, strata, seed):
    """analyze_test.boot_t, also returning the share of draws with zero standard error."""
    rng = np.random.default_rng(seed)
    m, se = est_se(s, n)
    groups = [np.flatnonzero(strata == t) for t in sorted(set(strata))]
    ts, zero = np.empty(B), 0
    for b in range(B):
        pick = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g)])
        mb, seb = est_se(s[pick], n[pick])
        zero += seb == 0
        ts[b] = (mb - m) / seb if seb > 0 else 0.0
    lo, hi = np.quantile(ts, [1 - ALPHA, ALPHA])
    return dict(diff_pp=round(100 * m, 2), L_pp=round(100 * (m - lo * se), 2), U_pp=round(100 * (m - hi * se), 2),
                zero_se_share=round(zero / B, 4))


def main():
    out = sys.argv[1]
    comp = json.load(open("out_holdout/clusters.json"))["component"]
    vr = rows(OMNI[1])
    res = {"qwen_scoring": {}, "resampling": {}}
    for mode, key in (("judge2_where_seen", lambda r: r["j1"] if r.get("j2") is None else r["j2"]),
                      ("matcher", lambda r: r["det"])):
        acc, types = acc_from(OMNI[0], vr, key)
        qids = sorted(types)
        res["qwen_scoring"][mode] = {
            "C1 A-M:asr": contrast(acc, "A", "M:asr", qids, types, comp, 2001),
            "C3 G@0.25-T@0.25": contrast(acc, "G@0.25", "T@0.25", qids, types, comp, 2003)}
    arms = sorted({r["arm"] for r in vr})
    res["qwen_scoring"]["j1_matcher_disagreement_pct"] = {
        a: round(100 * np.mean([r["j1"] != r["det"] for r in vr if r["arm"] == a]), 2) for a in arms}
    res["qwen_scoring"]["j1_matcher_disagreement_pct_outside_N"] = round(
        100 * np.mean([r["j1"] != r["det"] for r in vr if r["arm"] != "N"]), 2)
    seen = [r for r in vr if r.get("j2") is not None]
    answers = {(r["qid"], r["answer"].strip().lower()) for r in rows(OMNI[0])}
    res["qwen_scoring"]["judge2"] = dict(
        rows=len(seen), rows_pct=round(100 * len(seen) / len(vr), 2),
        agree_rows=sum(r["j1"] == r["j2"] for r in seen), distinct_answers_all=len(answers))
    for name, (a_p, v_p, c_p, a, b, seed) in DECISIONS.items():
        acc, types, _ = load(a_p, v_p)
        qids = sorted(types)
        c = json.load(open(c_p))["component"]
        qs, d = paired(acc, a, b, qids)
        cl = Clusters(qs, types, c)
        s, n = cluster_sums(d, cl.of, len(cl.names))
        res["resampling"][name] = {"stratified": boot(s, n, cl.stratum, seed),
                                   "unstratified": boot(s, n, np.zeros(len(s), int), seed),
                                   "mixed_type_clusters": int(sum(len({types[q] for q, k in zip(qs, cl.of) if k == i}) > 1
                                                                  for i in range(len(cl.names)))),
                                   "clusters": len(cl.names)}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
