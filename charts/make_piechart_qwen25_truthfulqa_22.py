# ============================================================
# make_piechart_qwen25_truthfulqa_22.py
# ============================================================
# Pie chart for the Qwen2.5-VL victim results on all 22
# TruthfulQA entries, using the ORIGINAL, unmodified attack
# prompt (no defense wrapping yet).
#
# Reads: outputs_victim_qwen_truthfulqa_22/entry_state.json
#
# Victim: Qwen2.5-VL 7B  |  Judge: Gemma-4-abliterated (unchanged)
# Dataset: TruthfulQA (domenicrosati/TruthfulQA, Lin et al. 2021)
# — same 22 entries already tested against Gemma as victim.
#
# Saves the chart to:
#   outputs_victim_qwen_truthfulqa_22/piechart_qwen25_truthfulqa_22.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

STATE_PATH = "outputs_victim_qwen_truthfulqa_22/entry_state.json"
CHART_PATH = ("outputs_victim_qwen_truthfulqa_22/"
              "piechart_qwen25_truthfulqa_22.png")

with open(STATE_PATH, "r") as f:
    state = json.load(f)

targeted   = sum(1 for s in state.values() if s.get("status") == "TARGETED")
untargeted = sum(1 for s in state.values() if s.get("status") == "UNTARGETED")
none_hall  = sum(1 for s in state.values() if s.get("status") == "NONE")
errors     = sum(1 for s in state.values() if s.get("status") == "ERROR")

scored_total = targeted + untargeted + none_hall
grand_total  = len(state)

print(f"Grand total entries : {grand_total}")
print(f"Errors (skipped)    : {errors}")
print(f"Scored total        : {scored_total}")
print()

if scored_total == 0:
    print("No scored entries found — nothing to chart.")
else:
    targeted_pct   = 100 * targeted / scored_total
    untargeted_pct = 100 * untargeted / scored_total
    none_pct       = 100 * none_hall / scored_total
    total_hall_pct = 100 * (targeted + untargeted) / scored_total

    print(f"TARGETED hallucination   : {targeted} ({targeted_pct:.1f}%)")
    print(f"UNTARGETED hallucination : {untargeted} ({untargeted_pct:.1f}%)")
    print(f"NOT hallucinating        : {none_hall} ({none_pct:.1f}%)")
    print(f"TOTAL hallucination      : {targeted + untargeted} "
          f"({total_hall_pct:.1f}%)")
    print()
    print(f"For comparison, Gemma as victim on these same 22 "
          f"TruthfulQA entries gave 63.6% total hallucination.")

    labels = [
        f"Targeted Hallucination\n({targeted} entries, {targeted_pct:.1f}%)",
        f"Untargeted Hallucination\n({untargeted} entries, {untargeted_pct:.1f}%)",
        f"No Hallucination\n({none_hall} entries, {none_pct:.1f}%)",
    ]
    sizes  = [targeted, untargeted, none_hall]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    explode = (0.05, 0.05, 0.05)

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(13)

    ax.set_title(
        f"Hallucination Classification — TruthfulQA, Qwen2.5-VL "
        f"Victim ({scored_total} Entries)\n"
        f"Victim: Qwen2.5-VL 7B  |  Judge: Gemma-4-abliterated  |  "
        f"Original Attack Prompt (no defense)\n"
        f"Same 22 entries as the Gemma-victim TruthfulQA test, "
        f"images reused, only Step 8 model changed\n"
        f"Total Hallucination Rate: {total_hall_pct:.1f}% "
        f"(Targeted + Untargeted)  |  vs. Gemma: 63.6%",
        fontsize=10, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved chart to: {CHART_PATH}")
