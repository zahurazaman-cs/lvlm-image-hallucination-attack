# ============================================================
# make_baseline_comparison_chart.py
# ============================================================
# Bar chart for the results section: the crude text overlay
# baseline versus the main symbolic attack, on the exact same 20
# qa_data.json entries, same victim + judge model throughout.
#
# Saves the chart to:
#   baseline_comparison_chart.png
#
# Requirements: matplotlib
# ============================================================

import matplotlib.pyplot as plt

# From the completed, corrected comparison run
BASELINE_TARGETED, BASELINE_UNTARGETED, BASELINE_NONE = 13, 0, 7
SYMBOLIC_TARGETED, SYMBOLIC_UNTARGETED, SYMBOLIC_NONE = 6, 4, 10
TOTAL = 20

baseline_hall = BASELINE_TARGETED + BASELINE_UNTARGETED
symbolic_hall = SYMBOLIC_TARGETED + SYMBOLIC_UNTARGETED

baseline_pct = 100 * baseline_hall / TOTAL
symbolic_pct = 100 * symbolic_hall / TOTAL

fig, ax = plt.subplots(figsize=(9, 7))

labels = ["Text Overlay Baseline\n(crude, visible text)",
          "Main Symbolic Attack\n(subtle, name-free visual cue)"]
values = [baseline_pct, symbolic_pct]
colors = ["#e67e22", "#c0392b"]

bars = ax.bar(labels, values, color=colors, edgecolor="white",
              linewidth=1.5, width=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{val:.1f}%", ha="center", va="bottom",
             fontsize=15, fontweight="bold")

ax.set_ylabel("Total Hallucination Rate (%)", fontsize=12)
ax.set_ylim(0, 80)
ax.set_title(
    "Attack Design Ablation: Crude Text Overlay vs Subtle Symbolic Cue\n"
    "Dataset: HaluEval (qa_data.json)  |  Victim + Judge: Gemma-4-abliterated\n"
    "Same 20 entries, same questions, only the visual manipulation differs\n"
    "No defense applied to either condition",
    fontsize=11, fontweight="bold", pad=15
)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("baseline_comparison_chart.png", dpi=200, bbox_inches="tight")
print("Saved chart to: baseline_comparison_chart.png")
