# ============================================================
# make_hallucination_pie_chart.py
# ============================================================
# Reads outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json
# (produced by reclassify_hallucinations.py) and generates a
# pie chart showing the breakdown of:
#   - TARGETED hallucination
#   - UNTARGETED confabulation
#   - NONE (no hallucination)
#
# Saves the chart as a PNG inside outputs_strict_batch_28/, so
# it lives alongside your other results and persists on disk.
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

BASE_OUTPUT_DIR = "outputs_strict_batch_28"
SUMMARY_PATH    = f"{BASE_OUTPUT_DIR}/RECLASSIFIED_SUMMARY.json"
CHART_PATH      = f"{BASE_OUTPUT_DIR}/hallucination_breakdown_pie_chart.png"

# ── Load the reclassified results ─────────────────────────────
with open(SUMMARY_PATH, "r") as f:
    results = json.load(f)

total      = len(results)
targeted   = sum(1 for r in results if r["final_category"] == "TARGETED")
untargeted = sum(1 for r in results if r["final_category"] == "UNTARGETED")
none_hall  = sum(1 for r in results if r["final_category"] == "NONE")

targeted_pct   = 100 * targeted / total
untargeted_pct = 100 * untargeted / total
none_pct       = 100 * none_hall / total
total_hall_pct = 100 * (targeted + untargeted) / total

print(f"Total entries            : {total}")
print(f"TARGETED hallucination   : {targeted} ({targeted_pct:.1f}%)")
print(f"UNTARGETED confabulation : {untargeted} ({untargeted_pct:.1f}%)")
print(f"NOT hallucinating        : {none_hall} ({none_pct:.1f}%)")
print(f"TOTAL hallucination      : {targeted + untargeted} ({total_hall_pct:.1f}%)")

# ── Build the pie chart ────────────────────────────────────────
labels = [
    f"Targeted Hallucination\n({targeted} entries, {targeted_pct:.1f}%)",
    f"Untargeted Confabulation\n({untargeted} entries, {untargeted_pct:.1f}%)",
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
    f"Hallucination Classification — Strict Batch ({total} Entries)\n"
    f"Total Hallucination Rate: {total_hall_pct:.1f}% (Targeted + Untargeted)",
    fontsize=13, fontweight="bold", pad=20
)

ax.axis("equal")
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved chart to: {CHART_PATH}")
