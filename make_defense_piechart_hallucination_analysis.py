# ============================================================
# make_defense_piechart_hallucination_analysis.py
# ============================================================
# Pie chart for the professor-approved HALLUCINATION_ANALYSIS
# defense condition (outputs_defense_gemma4_hallucination_analysis)
# — the ORIGINAL Step 8 instructions were left completely
# UNCHANGED (including the line stating the image is the PRIMARY
# source). This time, the wrap is a single strong tagged prompt
# block (RISK framing, VALIDATION, VERIFICATION, VISUAL_
# RELIABILITY_WARNING, GROUND_TRUTH_OVERRIDE before the original
# instructions; MANDATORY FINAL CHECK and FINAL_RULE after them;
# the whole thing wrapped in matching HALLUCINATION_ANALYSIS tags)
# reviewed and approved by Prof. Serra, replacing the earlier
# short CAUTION/REMINDER version.
#
# Shows, out of all 332 previously-hallucinated entries (TARGETED
# or UNTARGETED in the original Gemma-victim attack), what fraction
# were successfully "defended" (flipped to the correct answer) vs.
# still hallucinating after up to 5 reprompt attempts.
#
# Saves the chart to:
#   outputs_defense_gemma4_hallucination_analysis/piechart_hallucination_analysis.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

STATE_PATH = "outputs_defense_gemma4_hallucination_analysis/entry_state.json"
CHART_PATH = ("outputs_defense_gemma4_hallucination_analysis/"
              "piechart_hallucination_analysis.png")

with open(STATE_PATH, "r") as f:
    state = json.load(f)

succeeded = sum(1 for s in state.values() if s.get("defense_succeeded") is True)
failed    = sum(1 for s in state.values() if s.get("defense_succeeded") is False)
errors    = sum(1 for s in state.values() if s.get("status") == "ERROR")
total     = len(state)
scored_total = succeeded + failed

print(f"Total previously-hallucinated entries: {total}")
print(f"Errors (skipped)                     : {errors}")
print(f"Scored total                         : {scored_total}")
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
    print()
    print(f"For comparison, the earlier short CAUTION/REMINDER wrap "
          f"recovered 166/332 (50.0%). This new professor-approved "
          f"prompt recovered {succeeded}/332 ({succeeded_pct:.1f}%).")

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
        f"Defense Mechanism Result — Professor-Approved "
        f"HALLUCINATION_ANALYSIS Prompt\n"
        f"Victim (reprompted) + Judge: Gemma-4-abliterated  |  "
        f"Original instructions unchanged, tagged risk framing "
        f"wrapped around them\n"
        f"({scored_total} previously-hallucinated entries, "
        f"up to 5 reprompt attempts each)",
        fontsize=11, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved chart to: {CHART_PATH}")
