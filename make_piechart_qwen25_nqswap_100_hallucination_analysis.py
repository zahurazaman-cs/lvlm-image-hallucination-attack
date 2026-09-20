# ============================================================
# make_piechart_qwen25_nqswap_100_hallucination_analysis.py
# ============================================================
# Pie chart for the professor-approved Structured
# Hallucination-Analysis Defense condition on the 100-entry
# NQ-Swap real-world generalization dataset, with Qwen2.5-VL as
# the reprompted victim
# (outputs_qwen25_nqswap_100_hallucination_analysis).
#
# The ORIGINAL Step 8 instructions were left completely UNCHANGED
# (including the line stating the image is the PRIMARY source).
# The wrap is the same single tagged HALLUCINATION_ANALYSIS prompt
# block validated on qa_data.json and on Gemma-on-NQ-Swap,
# reviewed and approved by Prof. Serra.
#
# Saves the chart to:
#   outputs_qwen25_nqswap_100_hallucination_analysis/piechart_qwen25_nqswap_100_hallucination_analysis.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

STATE_PATH = "outputs_qwen25_nqswap_100_hallucination_analysis/entry_state.json"
CHART_PATH = ("outputs_qwen25_nqswap_100_hallucination_analysis/"
              "piechart_qwen25_nqswap_100_hallucination_analysis.png")

with open(STATE_PATH, "r") as f:
    state = json.load(f)

targeted   = sum(1 for s in state.values() if s.get("new_status") == "TARGETED")
untargeted = sum(1 for s in state.values() if s.get("new_status") == "UNTARGETED")
none_hall  = sum(1 for s in state.values() if s.get("new_status") == "NONE")
errors     = sum(1 for s in state.values() if s.get("status") == "ERROR")

scored_total = targeted + untargeted + none_hall
grand_total  = len(state)

changed_to_correct = sum(
    1 for s in state.values()
    if s.get("changed") and s.get("new_status") == "NONE"
)
changed_to_wrong = sum(
    1 for s in state.values()
    if s.get("changed") and s.get("new_status") in ("TARGETED", "UNTARGETED")
    and s.get("original_status") == "NONE"
)

print(f"Grand total entries       : {grand_total}")
print(f"Errors (skipped)          : {errors}")
print(f"Scored total              : {scored_total}")
print()

if scored_total == 0:
    print("No scored entries found — nothing to chart.")
else:
    targeted_pct   = 100 * targeted / scored_total
    untargeted_pct = 100 * untargeted / scored_total
    none_pct       = 100 * none_hall / scored_total
    total_hall_pct = 100 * (targeted + untargeted) / scored_total

    print(f"NEW TARGETED   : {targeted} ({targeted_pct:.1f}%)")
    print(f"NEW UNTARGETED : {untargeted} ({untargeted_pct:.1f}%)")
    print(f"NEW NONE       : {none_hall} ({none_pct:.1f}%)")
    print(f"NEW TOTAL HALLUCINATION: {targeted + untargeted} "
          f"({total_hall_pct:.1f}%)")
    print()
    print(f"Flipped hallucinating -> correct: {changed_to_correct}")
    print(f"Flipped correct -> hallucinating: {changed_to_wrong}")

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
        f"Hallucination Classification — NQ-Swap Real-World "
        f"Generalization Test\n"
        f"Structured Hallucination-Analysis Defense "
        f"({scored_total} Entries)\n"
        f"Victim (reprompted): Qwen2.5-VL 7B  |  Judge: "
        f"Gemma-4-abliterated\n"
        f"Total Hallucination Rate: {total_hall_pct:.1f}% "
        f"(Targeted + Untargeted)",
        fontsize=11, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved chart to: {CHART_PATH}")
