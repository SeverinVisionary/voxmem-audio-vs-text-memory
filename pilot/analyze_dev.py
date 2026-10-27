"""Dev-pilot measurements (PREREGISTRATION §7-8) from arms.jsonl and the judge verdicts.

    python analyze_dev.py ARMS_JSONL VERDICTS_JSONL OUT_JSON

Correctness = first judge; the deterministic matcher and the second judge are reported as agreement rates.
Evidence metadata (ev_any, ev_kept) is used here only for the coverage and expected-ΔG calculation.
"""
import collections
import json
import re
import statistics as st
import sys

from judge import latest_rows

VOXMEM_A = dict(speaker_information=42.2, paralinguistic_information=15.2, environmental_sound=15.4, pooled=22.3)
MDE_PP = (9.0, 9.9)                     # §7: discordance 0.25, ICC 0.05-0.10
REFUSAL = re.compile(r"not aware|don't (have|know|recall)|do not (have|know|recall)|no (record|information|previous)|"
                     r"can't|cannot|unable|not sure|not mentioned|no mention", re.I)


def pct(x):
    return round(100 * x, 1)


def main():
    rows = latest_rows(sys.argv[1])
    ver = {(v["qid"], v["arm"]): v for v in map(json.loads, open(sys.argv[2]))}
    for r in rows:
        r.update(ok=ver[(r["qid"], r["arm"])]["j1"], det=ver[(r["qid"], r["arm"])]["det"],
                 j2=ver[(r["qid"], r["arm"])]["j2"])
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["arm"]][r["qid"]] = r
    qids = sorted({r["qid"] for r in rows})
    types = {r["qid"]: r["evidence_type"] for r in rows}
    res = dict(n_families=len(qids), arms={})

    def acc(arm, qs=None):
        xs = [by[arm][q]["ok"] for q in (qs or qids) if q in by[arm]]
        return pct(st.mean(xs)) if xs else None

    for arm, d in sorted(by.items()):
        rs = list(d.values())
        res["arms"][arm] = dict(
            n=len(rs), acc=acc(arm), acc_by_type={t: acc(arm, [q for q in qids if types[q] == t]) for t in VOXMEM_A
                                                  if t != "pooled"},
            det_agree=pct(st.mean(r["ok"] == r["det"] for r in rs)),
            j2_agree=pct(st.mean(r["ok"] == r["j2"] for r in rs if r["j2"] is not None))
            if any(r["j2"] is not None for r in rs) else None,
            lp_sum=round(st.mean(r["lp_sum"] for r in rs), 3),
            refusal=pct(st.mean(bool(REFUSAL.search(r["answer"])) for r in rs)),
            new_gt32=pct(st.mean(r["n_new"] > 32 for r in rs)), new_gt48=pct(st.mean(r["n_new"] > 48 for r in rs)),
            hit_max=pct(st.mean(r["hit_max"] for r in rs)),
            n_in_median=st.median(r["n_in"] for r in rs), peak_gib=max(r["peak_gib"] for r in rs),
            seconds=round(st.mean(r["seconds"] for r in rs), 2))
        if rs[0].get("budget_gap") is not None:
            res["arms"][arm].update(budget_gap_max=max(abs(r["budget_gap"]) for r in rs),
                                    audio_turns_zero=pct(st.mean(r["n_audio_turns"] == 0 for r in rs)),
                                    ev_any=pct(st.mean(r["ev_any"] for r in rs)),
                                    ev_kept_frac=pct(st.mean(r["ev_kept"] / max(1, r["ev_turns"]) for r in rs)),
                                    audio_share=pct(st.mean(r["audio"] / max(1, r["memory"]) for r in rs)))

    def paired(a, b):
        qs = [q for q in qids if q in by[a] and q in by[b]]
        d = [by[a][q]["ok"] - by[b][q]["ok"] for q in qs]
        return dict(n=len(qs), diff_pp=pct(st.mean(d)), discordance=pct(st.mean(x != 0 for x in d)))

    res["contrasts"] = {"F-T@0.25 (H1)": paired("F", "T@0.25"),
                        **{f"G-T@{f} (H2 at {f})": paired(f"G@{f}", f"T@{f}") for f in ("0.1", "0.25", "0.5")},
                        "G-Gmm@0.25": paired("G@0.25", "Gmm@0.25"), "F-E@0.25": paired("F", "E@0.25"),
                        "oracle-T@0.25": paired("oracle", "T@0.25"), "A-N": paired("A", "N"),
                        "T@0.25-M:C": paired("T@0.25", "M:C"), "M:C-shuf": paired("M:C", "shuf")}
    gain = res["contrasts"]["oracle-T@0.25"]["diff_pp"]
    res["expected_dG"] = {f: round(res["arms"][f"G@{f}"]["ev_any"] / 100 * gain, 1) for f in ("0.1", "0.25", "0.5")}
    res["mde_pp"] = MDE_PP
    res["A_vs_voxmem"] = dict(ours=dict(res["arms"]["A"]["acc_by_type"], pooled=res["arms"]["A"]["acc"]),
                              voxmem_fig4=VOXMEM_A, note="VoxMem includes refusal items; ours are answerable dev only")
    t0, t1 = res["arms"]["T@0.25"], res["arms"]["T@0.25#line1"]
    res["answer_line"] = dict(line0_det_disagree=round(100 - t0["det_agree"], 1),
                              line1_det_disagree=round(100 - t1["det_agree"], 1), line0_acc=t0["acc"], line1_acc=t1["acc"])
    pa = sorted(r["P_A"] for r in by["T@0.25"].values())
    res["P_A"] = dict(min=pa[0], median=st.median(pa), max=pa[-1])
    pm = sorted(r["P_M"] for r in by["T@0.25"].values())
    res["P_M"] = dict(min=pm[0], median=st.median(pm), max=pm[-1])
    json.dump(res, open(sys.argv[3], "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
