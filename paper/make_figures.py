"""Figures for the mini paper, from the shipped outputs (run from paper/): python3 make_figures.py"""
import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = "../pilot/"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated, light)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

test = json.load(open(P + "out_test/TEST_RESULTS.json"))
hold = json.load(open(P + "out_holdout/HOLDOUT_RESULTS.json"))
score = json.load(open(P + "out_holdout/SCORING_RESULTS.json"))
arms = [json.loads(l) for l in open(P + "out_test/arms_minicpm.jsonl")]


def rd(x):
    """Round half up to one decimal, as in the text (26.95 -> +27.0)."""
    from decimal import ROUND_HALF_UP, Decimal
    return f"{Decimal(str(x)).quantize(Decimal('0.1'), ROUND_HALF_UP):+}"


def composition():
    """Mean memory size per arm on test, split into text and audio (means add up; medians would not)."""
    names = [("A", "$A$  full audio"), ("T@0.25", "$T^*$  text record"), ("G@0.25", "$G$  text + old clips"),
             ("F", "$F$  $T^*$ + all audio")]
    fig, ax = plt.subplots(figsize=(6.0, 2.0))
    for i, (arm, label) in enumerate(names):
        rs = [r for r in arms if r["arm"] == arm]
        text, audio = st.mean(r["text"] for r in rs), st.mean(r["audio"] for r in rs)
        ax.barh(i, text, color=BLUE, height=0.55, edgecolor="white", linewidth=1)
        ax.barh(i, audio, left=text, color=ORANGE, height=0.55, edgecolor="white", linewidth=1)
        ax.text(text + audio + 70, i, f"{text + audio:,.0f}", va="center", color=INK)
    ax.set_yticks(range(len(names)), [n for _, n in names])
    ax.invert_yaxis()
    ax.set_xlabel("mean memory size (MiniCPM-o positions), test split")
    ax.set_xlim(0, 6300)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["text", "audio"], frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig("fig_composition.pdf", metadata={"CreationDate": None})


def accuracy():
    """Accuracy by arm for the three reader/split rows (first judge)."""
    cols = [("N", "$N$\nquestion only"), ("M:asr", "$M_{\\rm asr}$\ntranscript"), ("A", "$A$\nfull audio"),
            ("T@0.25", "$T^*$\ntext record"), ("G@0.25", "$G$\ntext + clips"), ("F", "$F$\n$T^*$ + all audio")]
    rows = [("MiniCPM-o, test (300)", {a: v["acc"] for a, v in test["arms"].items()}, BLUE, ""),
            ("MiniCPM-o, holdout (142)", hold["acc"], ORANGE, "//"),
            ("Gemini 3.8 Flash, holdout subset (48)", hold["C ceiling"]["acc"], AQUA, "..")]
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    w = 0.27
    for k, (label, acc, c, hatch) in enumerate(rows):
        present = [i for i, (arm, _) in enumerate(cols) if arm in acc]
        for i, (arm, _) in enumerate(cols):
            if arm not in acc:
                continue
            n_here = sum(arm in r[1] for r in rows)
            slot = [j for j, r in enumerate(rows) if arm in r[1]].index(k)
            x = i + (slot - (n_here - 1) / 2) * w
            ax.bar(x, acc[arm], w * 0.92, color=c, hatch=hatch, edgecolor="white", linewidth=0.8,
                   label=label if i == present[0] else None)
            ax.text(x, acc[arm] + 0.8, rd(acc[arm])[1:], ha="center", fontsize=5.8, color=INK)
    ax.set_xticks(range(len(cols)), [c for _, c in cols], fontsize=7.5)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 55)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", fontsize=7, ncol=3, bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig("fig_accuracy.pdf", metadata={"CreationDate": None})


def neg(r):
    return dict(diff_pp=-r["diff_pp"], L_pp=-r["U_pp"], U_pp=-r["L_pp"])


def contrasts():
    """Key paired contrasts, all oriented audio minus text (right = audio better), 95% intervals."""
    c = hold["C ceiling"]
    groups = [
        ("MiniCPM-o 4.5", BLUE, [
            ("$F - T^*$, test (gate)", test["contrasts"]["H1 F-T@0.25"], False, False),
            ("$G - T^*$, test (matched cost)", test["contrasts"]["H2 G-T@0.25"], False, False),
            ("$G - T^*$, holdout (matched cost)", hold["B3 G-T@0.25"], False, False),
            ("$A - M_{\\rm asr}$, holdout (non-inferiority)", neg(hold["B1 M:asr-A"]), False, True),
            ("$A - M_{\\rm asr}$, the 48 Gemini families", c["minicpm A-M:asr (same families)"], False, False)]),
        ("Gemini 3.8 Flash (exploratory, 48 families)", AQUA, [
            ("$A - M_{\\rm asr}$", c["gemini A-M:asr"], False, False),
            ("$A - M_{\\rm asr}$, invalid runs scored right", score["post_hoc"]["gemini invalid runs as answered"]["A_minus_asr"],
             True, False),
            ("$G - T^*$ (matched in MiniCPM-o positions)", c["gemini G-T@0.25"], False, False),
            ("$A - T^*$", neg(score["post_hoc"]["gemini T@0.25-A"]), True, False)]),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    y, ticks, labels, heads = 0, [], [], []
    for title, col, items in groups:
        ticks.append(y); labels.append(title); heads.append(len(labels) - 1)
        y += 1
        for label, r, post_hoc, margin in items:
            ax.plot([r["L_pp"], r["U_pp"]], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r["diff_pp"], y, "o", color=col if not post_hoc else "white", mec=col, ms=6, mew=1.5)
            if margin:
                ax.plot([10, 10], [y - 0.35, y + 0.35], color=INK, lw=1.2)
                ax.text(10.6, y - 0.42, "margin +10", fontsize=6, color=MUTED, va="bottom")
            ax.text(36, y, f"{rd(r['diff_pp'])} [{rd(r['L_pp'])}, {rd(r['U_pp'])}]", va="center",
                    fontsize=7, color=INK)
            ticks.append(y); labels.append(label + (" (post hoc)" if post_hoc else ""))
            y += 1
        y += 0.3
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(ticks, labels, fontsize=7.5)
    for k in heads:
        t = ax.get_yticklabels()[k]
        t.set_fontweight("bold"); t.set_color(INK)
        ax.yaxis.get_major_ticks()[k].tick1line.set_visible(False)
    ax.set_ylim(y - 0.6, -0.7)
    ax.set_xlim(-22, 35)
    ax.set_xticks([-20, -10, 0, 10, 20, 30])
    ax.set_xlabel("audio minus text, accuracy difference (points), 95% interval")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("fig_contrasts.pdf", metadata={"CreationDate": None})


composition()
accuracy()
contrasts()
