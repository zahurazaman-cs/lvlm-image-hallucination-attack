# ============================================================
# make_defense_piechart_qwen25_hallucination_analysis.py
# ============================================================
# Pie chart for the professor-approved Structured
# Hallucination-Analysis Defense condition, applied with
# Qwen2.5-VL as the reprompted victim
# (outputs_defense_qwen25_hallucination_analysis) — the ORIGINAL
# Step 8 instructions were left completely UNCHANGED (including
# the line stating the image is the PRIMARY source). The wrap is
# the same single tagged HALLUCINATION_ANALYSIS prompt block
# validated on Gemma, reviewed and approved by Prof. Serra.
#
# Shows, out of all entries that hallucinated under Qwen2.5-VL's
# OWN original attack (107 entries, 21.4% of the full 500-entry
# HaluEval dataset), what fraction were successfully "defended"
# (flipped to the correct answer) vs. still hallucinating after up
# to 5 reprompt attempts.
#
# Saves the chart to:
#   outputs_defense_qwen25_hallucination_analysis/piechart_qwen25_hallucination_analysis.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

STATE_PATH = "outputs_defense_qwen25_hallucination_analysis/entry_state.json"
CHART_PATH = ("outputs_defense_qwen25_hallucination_analysis/"
              "piechart_qwen25_hallucination_analysis.png")

with open(STATE_PATH, "r") as f:
    state = json.load(f)

succeeded = sum(1 for s in state.values() if s.get("defense_succeeded") is True)
failed    = sum(1 for s in state.values() if s.get("defense_succeeded") is False)
errors    = sum(1 for s in state.values() if s.get("status") == "ERROR")
total     = len(state)
scored_total = succeeded + failed

print(f"Total previously-hallucinated Qwen2.5-VL entries: {total}")
print(f"Errors (skipped)                                : {errors}")
print(f"Scored total                                    : {scored_total}")
print()

if scored_total == 0:
    print("No scored entries found — nothing to chart.")
else:
    succeeded_pct = 100 * succeeded / scored_total
    failed_pct    = 100 * failed / scored_total

    print(f"Defense SUCCEEDED (flipped to correct): {succeeded} "
          f"({succeeded_pct:.1f}%)")
    print(f"Defense FAILED (still hallucinating)  : {failed} "
          f"({failed_pct:.1f}%)")

    labels = [
        f"Defense Succeeded\n({succeeded} entries, {succeeded_pct:.1f}%)",
        f"Defense Failed\n({failed} entries, {failed_pct:.1f}%)",
    ]
    sizes  = [succeeded, failed]
    colors = ["#2ecc71", "#e74c3c"]
    explode = (0.05, 0.05)

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
        f"Defense Mechanism Result — Structured "
        f"Hallucination-Analysis Prompt\n"
        f"Dataset: HaluEval (derived from HotpotQA)  |  Victim "
        f"(reprompted): Qwen2.5-VL 7B  |  Judge: "
        f"Gemma-4-abliterated\n"
        f"Original instructions unchanged, tagged risk framing "
        f"wrapped around them\n"
        f"({scored_total} entries that hallucinated under "
        f"Qwen2.5-VL's own original attack, up to 5 attempts each)",
        fontsize=10, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved chart to: {CHART_PATH}")
