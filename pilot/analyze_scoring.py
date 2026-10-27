"""Scoring sensitivity (exploratory, written after both runs): re-run the primary analyses with judge 1's verdict
replaced by (a) judge 2's wherever judge 2 saw the answer (seeded 10% plus all judge-1 / matcher disagreements;
judge 1 elsewhere) and (b) the deterministic matcher alone; plus disagreement rates by arm. Also post hoc:
T*@0.25 − A for each reader, and the Gemini contrasts with the invalid (shell-using) runs scored as answered
instead of incorrect: those runs could read the gold answers, so this is the bound in the other direction.

    python analyze_scoring.py OUT_JSON      # run from pilot/, reads out_test/ and out_holdout/
"""
import json
import os
import subprocess
import sys
import tempfile

VER = {"test": "out_test/judge/verdicts.jsonl", "holdout": "out_holdout/judge/verdicts.jsonl",
       "gemini": "out_holdout/judge_gemini/verdicts.jsonl"}
MODES = {"judge2_where_seen": lambda r: r["j1"] if r.get("j2") is None else r["j2"], "matcher": lambda r: r["det"]}


def rows(p):
    return [json.loads(l) for l in open(p)]


def rescore(tmp, mode):
    paths = {}
    for k, p in VER.items():
        paths[k] = os.path.join(tmp, f"{k}_{mode}.jsonl")
        with open(paths[k], "w") as f:
            for r in rows(p):
                f.write(json.dumps(dict(r, j1=MODES[mode](r))) + "\n")
    t, h = os.path.join(tmp, f"test_{mode}.json"), os.path.join(tmp, f"holdout_{mode}.json")
    subprocess.run([sys.executable, "analyze_test.py", "out_test/arms_minicpm.jsonl", paths["test"],
                    "out_test/clusters.json", t], check=True, capture_output=True)
    subprocess.run([sys.executable, "analyze_holdout.py", "out_holdout/arms_minicpm.jsonl", paths["holdout"],
                    "out_holdout/clusters.json", h, "out_holdout/gemini_arms.jsonl", paths["gemini"]],
                   check=True, capture_output=True)
    t, h = json.load(open(t)), json.load(open(h))
    c = h["C ceiling"]
    return {"test": dict(decision=t["decision"], acc={a: v["acc"] for a, v in t["arms"].items()},
                         H1=t["contrasts"]["H1 F-T@0.25"], H2=t["contrasts"]["H2 G-T@0.25"]),
            "holdout": dict(B1=h["B1 M:asr-A"], B1_decision=h["B1_decision"], acc=h["acc"],
                            B3_F=h["B3 F-T@0.25"], B3_G=h["B3 G-T@0.25"],
                            B2_speaker=h["B2 field-minus-transcript"]["speaker_information"]),
            "gemini": dict(acc=c["acc"], A_minus_asr=c["gemini A-M:asr"], F_minus_T=c["gemini F-T@0.25"],
                           G_minus_T=c["gemini G-T@0.25"], DiD=c["DiD (A-M:asr) gemini minus minicpm"])}


def post_hoc(tmp):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_test import contrast, load
    res = {}
    for name, (a, v, cl) in {"test": ("out_test/arms_minicpm.jsonl", VER["test"], "out_test/clusters.json"),
                             "holdout": ("out_holdout/arms_minicpm.jsonl", VER["holdout"], "out_holdout/clusters.json"),
                             "gemini": ("out_holdout/gemini_arms.jsonl", VER["gemini"], "out_holdout/clusters.json")}.items():
        acc, types, _ = load(a, v)
        q = sorted(types)
        res[f"{name} T@0.25-A"] = contrast(acc, "T@0.25", "A", q, types, json.load(open(cl))["component"], 1400)
    arms, seen = rows("out_holdout/gemini_arms.jsonl"), {(r["qid"], r["arm"]) for r in rows(VER["gemini"])}
    ga, gv = os.path.join(tmp, "gemini_as_given.jsonl"), os.path.join(tmp, "gemini_as_given_verdicts.jsonl")
    with open(ga, "w") as f:
        for r in arms:
            f.write(json.dumps({k: v for k, v in r.items() if k != "error"}) + "\n")
    with open(gv, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows(VER["gemini"]))
        for r in arms:                       # every invalid run answered the gold exactly (checked by hand)
            if (r["qid"], r["arm"]) not in seen:
                f.write(json.dumps(dict(qid=r["qid"], arm=r["arm"], j1=r["answer"].lower().replace(" ", "_") ==
                                        r["gold"].lower(), det=None, j2=None)) + "\n")
    h = os.path.join(tmp, "holdout_as_given.json")
    subprocess.run([sys.executable, "analyze_holdout.py", "out_holdout/arms_minicpm.jsonl", VER["holdout"],
                    "out_holdout/clusters.json", h, ga, gv], check=True, capture_output=True)
    c = json.load(open(h))["C ceiling"]
    res["gemini invalid runs as answered"] = dict(failed=c["failed"], A_minus_asr=c["gemini A-M:asr"],
                                                  DiD=c["DiD (A-M:asr) gemini minus minicpm"])
    return res


def agreement(p):
    rs = rows(p)
    by = {}
    for r in rs:
        by.setdefault(r["arm"], []).append(r["j1"] != r["det"])
    non_n = [r for r in rs if r["arm"] != "N"]
    both = [r for r in rs if r.get("j2") is not None]
    return dict(judge1_vs_matcher_pct_by_arm={a: round(100 * sum(v) / len(v), 1) for a, v in by.items()},
                judge1_vs_matcher_pct_non_N=round(100 * sum(r["j1"] != r["det"] for r in non_n) / len(non_n), 1),
                judge2_coverage_pct=round(100 * len(both) / len(rs), 1),
                judge1_judge2_agree_pct=round(100 * sum(r["j1"] == r["j2"] for r in both) / len(both), 1))


def main():
    out = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        res = {m: rescore(tmp, m) for m in MODES}
        res["post_hoc"] = post_hoc(tmp)
    res["agreement"] = {k: agreement(p) for k, p in VER.items()}
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
