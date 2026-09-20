# ============================================================
# combine_piechart_nqswap_till_50_reclassified.py
# ============================================================
# Pie chart for the RECLASSIFIED first 50 NQ-Swap entries
# (outputs_strict_batch_nqswap_26_50/RECLASSIFIED_SUMMARY.json) —
# this replaces the earlier binary-scheme chart. Reads the
# corrected native 3-way TARGETED/UNTARGETED/NONE classification
# produced by reclassify_nqswap_till_50.py, covering both:
#   - outputs_strict_batch_nqswap_25       (entries 0-24)
#   - outputs_strict_batch_nqswap_26_50    (entries 25-49)
#
# Victim + Judge: Gemma-4-abliterated  |  Image Generator:
# Qwen-Image-Edit-2509
#
# Dataset: NQ-Swap (Longpre et al.) — independent of HaluEval/
# HotpotQA, testing generalization of the primary 66.4%
# hallucination-rate finding.
#
# Saves the chart to:
#   outputs_strict_batch_nqswap_26_50/combined_piechart_nqswap_till_50_reclassified.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

SUMMARY_PATH = "outputs_strict_batch_nqswap_26_50/RECLASSIFIED_SUMMARY.json"
CHART_PATH = ("outputs_strict_batch_nqswap_26_50/"
              "combined_piechart_nqswap_till_50_reclassified.png")

with open(SUMMARY_PATH, "r") as f:
    results = json.load(f)

targeted   = sum(1 for r in results if r["final_category"] == "TARGETED")
untargeted = sum(1 for r in results if r["final_category"] == "UNTARGETED")
none_hall  = sum(1 for r in results if r["final_category"] == "NONE")

scored_total = targeted + untargeted + none_hall
grand_total  = len(results)

print(f"Grand total entries : {grand_total}")
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
    print(f"For comparison, the primary qa_data.json (HaluEval) "
          f"result across 500 entries was 66.4% total hallucination.")

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
        f"Generalization Test ({scored_total} Entries)\n"
        f"Victim + Judge: Gemma-4-abliterated  |  Image Generator: "
        f"Qwen-Image-Edit-2509\n"
        f"Dataset: NQ-Swap (Longpre et al.) — independent of "
        f"HaluEval/HotpotQA (reclassified to native 3-way scheme)\n"
        f"Total Hallucination Rate: {total_hall_pct:.1f}% "
        f"(Targeted + Untargeted)  |  vs. Primary Dataset: 66.4%",
        fontsize=11, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved combined chart to: {CHART_PATH}")
