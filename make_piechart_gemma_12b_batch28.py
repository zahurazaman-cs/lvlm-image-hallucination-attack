# ============================================================
# make_piechart_gemma_12b_batch28.py
# ============================================================
# Pie chart for the scale-up experiment: Gemma-4-abliterated 12B
# parameter model as victim, on the same 28 entries from batch 28,
# Steps 1-7 fully reused, judge unchanged (original 8.0B model).
#
# Saves the chart to:
#   outputs_gemma_larger_batch28_reuse/piechart_gemma_12b_batch28.png
#
# Requirements: matplotlib
# ============================================================

import matplotlib.pyplot as plt

TARGETED, UNTARGETED, NONE = 3, 1, 24
TOTAL = 28
ORIGINAL_HALL_PCT = 60.7  # the original 8.0B model's result on these same 28 entries

total_hall = TARGETED + UNTARGETED
total_hall_pct = 100 * total_hall / TOTAL
targeted_pct = 100 * TARGETED / TOTAL
untargeted_pct = 100 * UNTARGETED / TOTAL
none_pct = 100 * NONE / TOTAL

labels = [
    f"Targeted Hallucination\n({TARGETED} entries, {targeted_pct:.1f}%)",
    f"Untargeted Hallucination\n({UNTARGETED} entries, {untargeted_pct:.1f}%)",
    f"No Hallucination\n({NONE} entries, {none_pct:.1f}%)",
]
sizes = [TARGETED, UNTARGETED, NONE]
colors = ["#e74c3c", "#f39c12", "#2ecc71"]
explode = (0.05, 0.05, 0.05)

fig, ax = plt.subplots(figsize=(9, 7))
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors, explode=explode,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11},
    pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_color("white")
    at.set_fontweight("bold")
    at.set_fontsize(13)

ax.set_title(
    f"Hallucination Classification — Gemma-4-abliterated 12B Parameter Model\n"
    f"(Scale-Up Test, {TOTAL} Entries)\n"
    f"Dataset: HaluEval (qa_data.json, batch 28)  |  Victim: "
    f"Gemma-4-abliterated 12B  |  Judge: Gemma-4-abliterated 8.0B (unchanged)\n"
    f"Same images reused from the original attack (Steps 1-7 untouched)\n"
    f"Total Hallucination Rate: {total_hall_pct:.1f}%  |  "
    f"vs. Original 8.0B Model on Same 28 Entries: {ORIGINAL_HALL_PCT:.1f}%",
    fontsize=10, fontweight="bold", pad=20
)

ax.axis("equal")
plt.tight_layout()
plt.savefig("outputs_gemma_larger_batch28_reuse/piechart_gemma_12b_batch28.png",
            dpi=200, bbox_inches="tight")
print("Saved chart to: outputs_gemma_larger_batch28_reuse/piechart_gemma_12b_batch28.png")
