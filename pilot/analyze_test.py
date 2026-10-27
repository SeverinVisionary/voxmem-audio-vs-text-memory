"""Primary test analysis (PREREGISTRATION §6-7), committed before the test run.

    python analyze_test.py ARMS_JSONL VERDICTS_JSONL CLUSTERS_JSON OUT_JSON

ARMS_JSONL: run_dev --arms test output (every attempt; the last row per family and arm counts). VERDICTS_JSONL:
judge.py output (j1 = correctness). CLUSTERS_JSON: {"component": {qid: id}, ...} from clusters.py; families
missing from it are their own cluster. Reader errors after the rerun are scored incorrect; > 5% in any primary arm
makes the run INVALID.

Tests (fixed sequence): H1 ΔF = F − T*@0.25, then H2 ΔG = G@0.25 − T*@0.25 only if H1 passes, each one-sided
α = 0.025 by a cluster-level sign-flip permutation test (10,000 flips). Intervals: type-stratified cluster
bootstrap-t (10,000 resamples), 95% two-sided, so L/U are the one-sided 0.025 bounds.
"""
import collections
import json
import sys

import numpy as np

B = 10_000
ALPHA = 0.025
FUTILITY_PP = 10.0
PRIMARY = ["T@0.25", "G@0.25", "F", "Gmm@0.25"]
TYPES = ["speaker_information", "paralinguistic_information", "environmental_sound"]


def load(arms_path, ver_path):
    last = {}
    for line in open(arms_path):
        r = json.loads(line)
        last[(r["qid"], r["arm"])] = r
    ver = {}
    for line in open(ver_path):
        v = json.loads(line)
        ver[(v["qid"], v["arm"])] = v
    acc, types, failed = collections.defaultdict(dict), {}, collections.Counter()
    for (q, a), r in last.items():
        types[q] = r["evidence_type"]
        if "error" in r:
            failed[a] += 1
            acc[a][q] = 0 if not r["error"].startswith("no plan") else None
        else:
            acc[a][q] = int(bool(ver[(q, a)]["j1"]))
    return acc, types, failed


class Clusters:
    def __init__(self, qids, types, comp):
        ids = {q: comp.get(q, f"solo:{q}") for q in qids}
        self.names = sorted(set(ids.values()))
        idx = {c: i for i, c in enumerate(self.names)}
        self.of = np.array([idx[ids[q]] for q in qids])
        # stratum of a cluster = the type of most of its families (ties -> first in TYPES order)
        votes = collections.defaultdict(collections.Counter)
        for q, c in zip(qids, self.of):
            votes[c][types[q]] += 1
        self.stratum = np.array([max(TYPES, key=lambda t: (votes[c][t], -TYPES.index(t))) for c in range(len(self.names))])


def paired(acc, a, b, qids):
    qs = [q for q in qids if acc[a].get(q) is not None and acc[b].get(q) is not None]
    return qs, np.array([acc[a][q] - acc[b][q] for q in qs], float)


def cluster_sums(d, of, k):
    s, n = np.zeros(k), np.zeros(k)
    np.add.at(s, of, d); np.add.at(n, of, 1)
    return s, n


def est_se(s, n):
    """Family-weighted mean difference and its cluster-robust SE from per-cluster sums and sizes."""
    N = n.sum()
    m = s.sum() / N
    return m, np.sqrt(((s - n * m) ** 2).sum()) / N


def signflip_p(s, n, rng):
    obs = s.sum() / n.sum()
    flips = rng.choice([-1.0, 1.0], size=(B, len(s)))
    perm = (flips * s).sum(1) / n.sum()
    return float((1 + (perm >= obs - 1e-12).sum()) / (B + 1))


def boot_t(s, n, strata, rng):
    m, se = est_se(s, n)
    groups = [np.flatnonzero(strata == t) for t in sorted(set(strata))]
    ts = np.empty(B)
    for b in range(B):
        pick = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g)])
        mb, seb = est_se(s[pick], n[pick])
        ts[b] = (mb - m) / seb if seb > 0 else 0.0
    lo, hi = np.quantile(ts, [1 - ALPHA, ALPHA])
    return m, se, m - lo * se, m - hi * se


def contrast(acc, a, b, qids, types, comp, seed):
    qs, d = paired(acc, a, b, qids)
    cl = Clusters(qs, types, comp)
    s, n = cluster_sums(d, cl.of, len(cl.names))
    rng = np.random.default_rng(seed)
    m, se, L, U = boot_t(s, n, cl.stratum, rng)
    p = signflip_p(s, n, np.random.default_rng(seed + 1))
    loo = [(s.sum() - s[i]) / (n.sum() - n[i]) for i in range(len(s)) if n.sum() > n[i]]
    return dict(n=len(qs), clusters=len(cl.names), diff_pp=round(100 * m, 2), se_pp=round(100 * se, 2),
                L_pp=round(100 * L, 2), U_pp=round(100 * U, 2), p_one_sided=round(p, 5),
                discordance=round(float(np.mean(d != 0)), 3),
                loo_range_pp=[round(100 * min(loo), 2), round(100 * max(loo), 2)] if loo else None)


def main():
    arms_path, ver_path, cl_path, out = sys.argv[1:5]
    acc, types, failed = load(arms_path, ver_path)
    comp = json.load(open(cl_path))["component"] if cl_path != "-" else {}
    qids = sorted(types)
    res = dict(n_families=len(qids), clusters_file=cl_path, arms={})
    for a, d in sorted(acc.items()):
        xs = [v for v in d.values() if v is not None]
        res["arms"][a] = dict(n=len(d), failed=failed[a], fail_rate=round(failed[a] / max(1, len(d)), 4),
                              acc=round(100 * np.mean(xs), 2) if xs else None,
                              by_type={t: round(100 * np.mean([d[q] for q in d if types[q] == t and d[q] is not None]), 2)
                                       for t in TYPES if any(types[q] == t and d[q] is not None for q in d)})
    invalid = [a for a in PRIMARY if res["arms"].get(a, {}).get("fail_rate", 1) > 0.05]
    c = {}
    c["H1 F-T@0.25"] = contrast(acc, "F", "T@0.25", qids, types, comp, 101)
    c["H2 G-T@0.25"] = contrast(acc, "G@0.25", "T@0.25", qids, types, comp, 202)
    c["G-Gmm@0.25"] = contrast(acc, "G@0.25", "Gmm@0.25", qids, types, comp, 303)
    for f, sd in (("0.5", 404), ("0.1", 505)):
        if f"G@{f}" in acc:
            c[f"G-T@{f} (exploratory)" if f == "0.1" else f"G-T@{f} (secondary)"] = \
                contrast(acc, f"G@{f}", f"T@{f}", qids, types, comp, sd)
    c["F-E@0.25 (exploratory)"] = contrast(acc, "F", "E@0.25", qids, types, comp, 606)
    for t, sd in zip(TYPES, (707, 808, 909)):
        qt = [q for q in qids if types[q] == t]
        c[f"G-T@0.25 {t} (exploratory)"] = contrast(acc, "G@0.25", "T@0.25", qt, types, comp, sd)
    eq = [c[f"G-T@0.25 {t} (exploratory)"]["diff_pp"] for t in TYPES]
    res["equal_weight_by_type_dG_pp"] = round(float(np.mean(eq)), 2)
    res["contrasts"] = c
    h1, h2, gm = c["H1 F-T@0.25"], c["H2 G-T@0.25"], c["G-Gmm@0.25"]
    h1_pass = h1["p_one_sided"] < ALPHA
    if invalid:
        decision = f"INVALID (failures > 5% in {invalid})"
    elif not h1_pass:
        decision = "NO-CUE"
    elif h2["U_pp"] < FUTILITY_PP and not (h2["p_one_sided"] < ALPHA and h2["L_pp"] > 0):
        decision = "TEXT-WINS"
    elif h2["p_one_sided"] < ALPHA and h2["L_pp"] > 0 and gm["p_one_sided"] < ALPHA:
        decision = "GO"
    else:
        decision = "INCONCLUSIVE"
    res["H1_pass"], res["decision"] = h1_pass, decision
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(dict(decision=decision, H1=h1, H2=h2, G_Gmm=gm), indent=1))


if __name__ == "__main__":
    main()
