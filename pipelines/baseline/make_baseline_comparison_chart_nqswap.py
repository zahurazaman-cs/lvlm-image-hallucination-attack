# ============================================================
# make_baseline_comparison_chart_nqswap.py
# ============================================================
# Bar chart for the results section: the crude text overlay
# baseline versus the main symbolic attack, on NQ-Swap, matched by
# question text (19 entries), same victim + judge model throughout.
#
# Saves the chart to:
#   baseline_comparison_chart_nqswap.png
#
# Requirements: matplotlib
# ============================================================

import matplotlib.pyplot as plt

# From the completed, matched comparison run
BASELINE_TARGETED, BASELINE_UNTARGETED, BASELINE_NONE = 4, 1, 14
SYMBOLIC_TARGETED, SYMBOLIC_UNTARGETED, SYMBOLIC_NONE = 0, 11, 8
TOTAL = 19

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
    "Dataset: NQ-Swap (Longpre et al.)  |  Victim + Judge: Gemma-4-abliterated\n"
    f"Same {TOTAL} entries, matched by question text, only the visual manipulation differs\n"
    "No defense applied to either condition",
    fontsize=11, fontweight="bold", pad=15
)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("baseline_comparison_chart_nqswap.png", dpi=200, bbox_inches="tight")
print("Saved chart to: baseline_comparison_chart_nqswap.png")
