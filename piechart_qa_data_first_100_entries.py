# ============================================================
# piechart_qa_data_first_100_entries.py
# ============================================================
# Pie chart for exactly the FIRST 100 SCORED entries out of the
# 500-entry qa_data.json (HaluEval) Gemma-4 pipeline results —
# true apples-to-apples comparison against the 100-entry NQ-Swap
# generalization test.
#
# Since batches are defined by fixed BATCH_INDICES (not round
# scored-entry counts), reaching exactly 100 requires:
#   - outputs_strict_batch_28       — ALL 28 entries (reclassified
#     to native 3-way scheme)
#   - outputs_strict_batch_50       — ALL 23 entries
#   - outputs_strict_batch_151_200  — ALL 26 entries
#   - outputs_strict_batch_201_280  — FIRST 23 entries only (in
#     original processing order, matching ascending BATCH_INDICES)
#
# 28 + 23 + 26 + 23 = 100 entries exactly.
#
# Victim + Judge: Gemma-4-abliterated  |  Image Generator:
# Qwen-Image-Edit-2509
#
# Saves the chart to:
#   outputs_strict_batch_201_280/piechart_qa_data_first_100_entries.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

BATCH_28_RECLASSIFIED  = "outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json"
BATCH_50_SUMMARY       = "outputs_strict_batch_50/SUMMARY.json"
BATCH_151_200_SUMMARY  = "outputs_strict_batch_151_200/SUMMARY.json"
BATCH_201_280_SUMMARY  = "outputs_strict_batch_201_280/SUMMARY.json"
CHART_PATH = ("outputs_strict_batch_201_280/"
              "piechart_qa_data_first_100_entries.png")


def load_batch_28():
    with open(BATCH_28_RECLASSIFIED, "r") as f:
        data = json.load(f)
    return [r["final_category"] for r in data]


def load_native_batch(path, limit=None):
    with open(path, "r") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return [r["status"] for r in data]


categories_28      = load_batch_28()
categories_50      = load_native_batch(BATCH_50_SUMMARY)
categories_151_200 = load_native_batch(BATCH_151_200_SUMMARY)
categories_201_first23 = load_native_batch(BATCH_201_280_SUMMARY, limit=23)

all_categories = (
    categories_28 + categories_50 + categories_151_200 + categories_201_first23
)

targeted   = sum(1 for c in all_categories if c == "TARGETED")
untargeted = sum(1 for c in all_categories if c == "UNTARGETED")
none_hall  = sum(1 for c in all_categories if c == "NONE")
skipped    = sum(1 for c in all_categories if c == "SKIPPED_NO_PHOTO")

scored_total = targeted + untargeted + none_hall
grand_total  = len(all_categories)

print(f"Batch 28 entries               : {len(categories_28)}")
print(f"Batch 50 entries                : {len(categories_50)}")
print(f"Batch 151-200 entries           : {len(categories_151_200)}")
print(f"Batch 201-280 (first 23 only)   : {len(categories_201_first23)}")
print(f"Grand total                     : {grand_total}")
print(f"Skipped (no photo)              : {skipped}")
print(f"Scored total                    : {scored_total}")
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
        f"Hallucination Classification — qa_data.json (HaluEval), "
        f"First 100 Scored Entries\n"
        f"Victim + Judge: Gemma-4-abliterated  |  Image Generator: "
        f"Qwen-Image-Edit-2509\n"
        f"(28 + 23 + 26 + 23 entries, {skipped} skipped)\n"
        f"Total Hallucination Rate: {total_hall_pct:.1f}% "
        f"(Targeted + Untargeted)  |  Direct comparison to "
        f"NQ-Swap (n=100)",
        fontsize=11, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved combined chart to: {CHART_PATH}")
