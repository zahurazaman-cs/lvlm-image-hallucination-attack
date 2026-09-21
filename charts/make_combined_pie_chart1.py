# ============================================================
# make_combined_pie_chart.py
# ============================================================
# Combines results from BOTH strict batch runs:
#   - outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json
#     (28 entries, classified via the post-hoc reclassify pass,
#      field: "final_category")
#   - outputs_strict_batch_50/SUMMARY.json
#     (23 entries, classified NATIVELY during the run itself,
#      field: "status")
#
# Produces ONE combined pie chart across all 51 entries showing:
#   - TARGETED hallucination
#   - UNTARGETED confabulation
#   - NONE (no hallucination)
# (entries with SKIPPED_NO_PHOTO are excluded from percentages,
#  same convention as before, but reported separately)
#
# Saves the chart to:
#   outputs_strict_batch_50/combined_hallucination_pie_chart.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

BATCH_28_SUMMARY = "outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json"
BATCH_50_SUMMARY = "outputs_strict_batch_50/SUMMARY.json"
CHART_PATH        = "outputs_strict_batch_50/combined_hallucination_pie_chart.png"


def load_batch_28():
    """
    outputs_strict_batch_28 was reclassified post-hoc; each entry's
    result is keyed as "final_category": "TARGETED" | "UNTARGETED" | "NONE"
    """
    with open(BATCH_28_SUMMARY, "r") as f:
        data = json.load(f)
    categories = [r["final_category"] for r in data]
    return categories


def load_batch_50():
    """
    outputs_strict_batch_50 classified natively during the run;
    each entry's result is keyed as "status": "TARGETED" | "UNTARGETED"
    | "NONE" | "SKIPPED_NO_PHOTO"
    """
    with open(BATCH_50_SUMMARY, "r") as f:
        data = json.load(f)
    categories = [r["status"] for r in data]
    return categories


# ── Load and combine both batches ─────────────────────────────
categories_28 = load_batch_28()
categories_50 = load_batch_50()
all_categories = categories_28 + categories_50

skipped    = sum(1 for c in all_categories if c == "SKIPPED_NO_PHOTO")
targeted   = sum(1 for c in all_categories if c == "TARGETED")
untargeted = sum(1 for c in all_categories if c == "UNTARGETED")
none_hall  = sum(1 for c in all_categories if c == "NONE")

scored_total = targeted + untargeted + none_hall  # excludes skipped
grand_total  = len(all_categories)

print(f"Batch 28 entries : {len(categories_28)}")
print(f"Batch 50 entries : {len(categories_50)}")
print(f"Grand total      : {grand_total}")
print(f"Skipped (no photo) : {skipped}")
print(f"Scored total     : {scored_total}")
print()
print(f"TARGETED hallucination   : {targeted} "
      f"({100*targeted/scored_total:.1f}%)")
print(f"UNTARGETED hallucination : {untargeted} "
      f"({100*untargeted/scored_total:.1f}%)")
print(f"NOT hallucinating        : {none_hall} "
      f"({100*none_hall/scored_total:.1f}%)")
print(f"TOTAL hallucination      : {targeted + untargeted} "
      f"({100*(targeted+untargeted)/scored_total:.1f}%)")

# ── Build the combined pie chart ───────────────────────────────
targeted_pct   = 100 * targeted / scored_total
untargeted_pct = 100 * untargeted / scored_total
none_pct       = 100 * none_hall / scored_total
total_hall_pct = 100 * (targeted + untargeted) / scored_total

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
    f"Hallucination Classification — Combined Strict Batches "
    f"({scored_total} Entries)\n"
    f"(28 from first batch + 23 from second batch, "
    f"{skipped} skipped)\n"
    f"Total Hallucination Rate: {total_hall_pct:.1f}% (Targeted + Untargeted)",
    fontsize=12, fontweight="bold", pad=20
)

ax.axis("equal")
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved combined chart to: {CHART_PATH}")
