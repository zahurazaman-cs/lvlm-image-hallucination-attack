# ============================================================
# combine_piechart_nqswap_till_75.py
# ============================================================
# Combines results from ALL 75 NQ-Swap real-world generalization
# test entries:
#   - outputs_strict_batch_nqswap_26_50/RECLASSIFIED_SUMMARY.json
#     (entries 0-49 — reclassified from the original binary scheme
#     into the native 3-way scheme via reclassify_nqswap_till_50.py)
#   - outputs_strict_batch_nqswap_51_75/SUMMARY.json
#     (entries 50-74 — native 3-way scheme from the start, using
#     the correct batch_50-style template)
#
# Victim + Judge: Gemma-4-abliterated  |  Image Generator:
# Qwen-Image-Edit-2509
#
# Dataset: NQ-Swap (Longpre et al.) — independent of HaluEval/
# HotpotQA, testing generalization of the primary 66.4%
# hallucination-rate finding.
#
# Saves the chart to:
#   outputs_strict_batch_nqswap_51_75/combined_piechart_nqswap_till_75.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

RECLASSIFIED_PATH = "outputs_strict_batch_nqswap_26_50/RECLASSIFIED_SUMMARY.json"
BATCH3_SUMMARY_PATH = "outputs_strict_batch_nqswap_51_75/SUMMARY.json"
CHART_PATH = ("outputs_strict_batch_nqswap_51_75/"
              "combined_piechart_nqswap_till_75.png")

with open(RECLASSIFIED_PATH, "r") as f:
    reclassified = json.load(f)

with open(BATCH3_SUMMARY_PATH, "r") as f:
    batch3 = json.load(f)

statuses_0_49 = [r["final_category"] for r in reclassified]
statuses_50_74 = [r["status"] for r in batch3]

all_statuses = statuses_0_49 + statuses_50_74

targeted   = sum(1 for c in all_statuses if c == "TARGETED")
untargeted = sum(1 for c in all_statuses if c == "UNTARGETED")
none_hall  = sum(1 for c in all_statuses if c == "NONE")
skipped    = sum(1 for c in all_statuses if c == "SKIPPED_NO_PHOTO")

scored_total = targeted + untargeted + none_hall
grand_total  = len(all_statuses)

print(f"Entries 0-49 (reclassified)  : {len(statuses_0_49)}")
print(f"Entries 50-74 (native 3-way) : {len(statuses_50_74)}")
print(f"Grand total                  : {grand_total}")
print(f"Skipped (no photo)           : {skipped}")
print(f"Scored total                 : {scored_total}")
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
          f"result across 500 entries was 66.4% total hallucination "
          f"(45.8% TARGETED, 20.6% UNTARGETED).")

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
        f"HaluEval/HotpotQA, 3 batches (25+25+25), native 3-way scheme\n"
        f"Total Hallucination Rate: {total_hall_pct:.1f}% "
        f"(Targeted + Untargeted)  |  vs. Primary Dataset: 66.4%",
        fontsize=11, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved combined chart to: {CHART_PATH}")
