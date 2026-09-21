# ============================================================
# make_comparison_three_defense_conditions_qwen25_truthfulqa.py
# ============================================================
# THREE-WAY comparison chart for the results section, showing the
# effect of the two prompt-level defenses on Qwen2.5-VL across all
# 22 TruthfulQA entries, with clear annotation of dataset
# identity, victim model, and total entries.
#
#   Condition 1: No Defense (the original attack) — read from
#     outputs_victim_qwen_truthfulqa_22.
#   Condition 2: Minimal Prompt Defense (short caution/reminder
#     wrap) — read from outputs_qwen25_truthfulqa_22_wrap.
#   Condition 3: Structured Hallucination-Analysis Defense (tagged
#     HALLUCINATION_ANALYSIS block) — read from
#     outputs_qwen25_truthfulqa_22_hallucination_analysis.
#
# All three conditions are read LIVE from their actual result
# files rather than hardcoded, so the chart is guaranteed accurate
# regardless of the exact final numbers.
#
# Saves the chart to:
#   outputs_qwen25_truthfulqa_22_hallucination_analysis/comparison_three_defense_conditions_qwen25_truthfulqa.png
#
# Requirements: matplotlib
# ============================================================

import json
import matplotlib.pyplot as plt

TOTAL_ENTRIES = 22

CHART_PATH = ("outputs_qwen25_truthfulqa_22_hallucination_analysis/"
              "comparison_three_defense_conditions_qwen25_truthfulqa.png")

COND1_STATE_PATH = "outputs_victim_qwen_truthfulqa_22/entry_state.json"
with open(COND1_STATE_PATH, "r") as f:
    cond1_state = json.load(f)

cond1_targeted   = sum(1 for s in cond1_state.values() if s.get("status") == "TARGETED")
cond1_untargeted = sum(1 for s in cond1_state.values() if s.get("status") == "UNTARGETED")
cond1_none       = sum(1 for s in cond1_state.values() if s.get("status") == "NONE")
cond1_total      = cond1_targeted + cond1_untargeted + cond1_none

COND2_STATE_PATH = "outputs_qwen25_truthfulqa_22_wrap/entry_state.json"
with open(COND2_STATE_PATH, "r") as f:
    cond2_state = json.load(f)

cond2_targeted   = sum(1 for s in cond2_state.values() if s.get("new_status") == "TARGETED")
cond2_untargeted = sum(1 for s in cond2_state.values() if s.get("new_status") == "UNTARGETED")
cond2_none       = sum(1 for s in cond2_state.values() if s.get("new_status") == "NONE")
cond2_total      = cond2_targeted + cond2_untargeted + cond2_none

COND3_STATE_PATH = "outputs_qwen25_truthfulqa_22_hallucination_analysis/entry_state.json"
with open(COND3_STATE_PATH, "r") as f:
    cond3_state = json.load(f)

cond3_targeted   = sum(1 for s in cond3_state.values() if s.get("new_status") == "TARGETED")
cond3_untargeted = sum(1 for s in cond3_state.values() if s.get("new_status") == "UNTARGETED")
cond3_none       = sum(1 for s in cond3_state.values() if s.get("new_status") == "NONE")
cond3_total      = cond3_targeted + cond3_untargeted + cond3_none

cond1_hall_pct = 100 * (cond1_targeted + cond1_untargeted) / cond1_total if cond1_total else 0
cond2_hall_pct = 100 * (cond2_targeted + cond2_untargeted) / cond2_total if cond2_total else 0
cond3_hall_pct = 100 * (cond3_targeted + cond3_untargeted) / cond3_total if cond3_total else 0

print(f"Dataset: TruthfulQA (domenicrosati/TruthfulQA, Lin et al. 2021)")
print(f"Victim model: Qwen2.5-VL 7B  |  Judge model: Gemma-4-abliterated")
print(f"Total entries: {TOTAL_ENTRIES}")
print()
print(f"Condition 1, No Defense                      : "
      f"{cond1_total} entries, {cond1_hall_pct:.1f}% hallucinated "
      f"({cond1_targeted} targeted, {cond1_untargeted} untargeted, "
      f"{cond1_none} none)")
print(f"Condition 2, Minimal Prompt Defense           : "
      f"{cond2_total} entries, {cond2_hall_pct:.1f}% hallucinated "
      f"({cond2_targeted} targeted, {cond2_untargeted} untargeted, "
      f"{cond2_none} none)")
print(f"Condition 3, Structured Hallucination-Analysis Defense: "
      f"{cond3_total} entries, {cond3_hall_pct:.1f}% hallucinated "
      f"({cond3_targeted} targeted, {cond3_untargeted} untargeted, "
      f"{cond3_none} none)")

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

colors_3way = ["#e74c3c", "#f39c12", "#2ecc71"]
explode_3   = (0.05, 0.05, 0.05)

panels = [
    (axes[0], cond1_targeted, cond1_untargeted, cond1_none, cond1_total,
     cond1_hall_pct, "Condition 1: No Defense\n(Original Attack)"),
    (axes[1], cond2_targeted, cond2_untargeted, cond2_none, cond2_total,
     cond2_hall_pct, "Condition 2: Minimal Prompt Defense\n"
                     "(short caution/reminder wrap)"),
    (axes[2], cond3_targeted, cond3_untargeted, cond3_none, cond3_total,
     cond3_hall_pct, "Condition 3: Structured Hallucination-\n"
                     "Analysis Defense (tagged risk framing)"),
]

for ax, t, u, n, total, hall_pct, title in panels:
    if total == 0:
        ax.text(0.5, 0.5, "No data found", ha="center", va="center")
        ax.set_title(title, fontsize=12, fontweight="bold")
        continue
    sizes = [t, u, n]
    labels = [
        f"Targeted\n({t}, {100*t/total:.1f}%)",
        f"Untargeted\n({u}, {100*u/total:.1f}%)",
        f"No Hallucination\n({n}, {100*n/total:.1f}%)",
    ]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_3way, explode=explode_3,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9},
        pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(11)
    ax.set_title(f"{title}\nTotal Hallucination: {hall_pct:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.axis("equal")

fig.suptitle(
    f"Effect of Prompt-Level Defenses on Hallucination Rate\n"
    f"Dataset: TruthfulQA (domenicrosati/TruthfulQA, Lin et al. "
    f"2021)  |  Victim Model: Qwen2.5-VL 7B  |  Judge Model: "
    f"Gemma-4-abliterated  |  Total Entries: {TOTAL_ENTRIES}\n"
    f"All three conditions reuse the same adversarial images; only "
    f"the Step 8 instruction wording differs across conditions",
    fontsize=13, fontweight="bold", y=1.08
)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved three-way comparison chart to: {CHART_PATH}")
