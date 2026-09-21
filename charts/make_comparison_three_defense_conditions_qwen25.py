# ============================================================
# make_comparison_three_defense_conditions_qwen25.py
# ============================================================
# THREE-WAY comparison chart for the results section of the
# paper, showing the effect of the two prompt-level defenses on
# QWEN2.5-VL across the full 500-entry dataset, with clear
# annotation of dataset identity, victim model, and total entries
# so the chart is self-explanatory to a reviewer with no other
# context.
#
#   Condition 1: No Defense (the original attack, no prompt-level
#     mitigation applied — the baseline). Only the combined total
#     hallucination count (107/500, 21.4%) is confirmed; the
#     targeted/untargeted split for this condition was not
#     separately saved, so this panel is shown as a two-way split.
#     If you have that split on hand, edit COND1_TARGETED /
#     COND1_UNTARGETED below and switch this panel to three-way.
#   Condition 2: Minimal Prompt Defense (a short caution sentence
#     added before the original Step 8 instructions and a short
#     reminder sentence added after them, with the original
#     instructions themselves left completely unchanged)
#   Condition 3: Structured Hallucination-Analysis Defense (a
#     tagged HALLUCINATION_ANALYSIS block wrapping the SAME
#     unchanged original instructions, including explicit risk
#     framing, a mandatory verification check, and a final
#     override rule)
#
# All three conditions use the SAME 500 entries, the SAME
# adversarial images (Steps 1-7 never regenerated across any
# condition), the SAME victim model (Qwen2.5-VL 7B), and the SAME
# judge model (Gemma-4-abliterated, unchanged throughout). The
# ONLY variable across the three conditions is the wording of the
# Step 8 instruction shown to the victim model.
#
# Condition 1 and Condition 2 results are hardcoded below since
# they were already computed in earlier runs. Condition 3 is read
# live from its results file.
#
# Saves the chart to:
#   outputs_step8_reclassify_all_qwen25_hallucination_analysis/comparison_three_defense_conditions_qwen25.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

TOTAL_ENTRIES = 500

# ── Condition 1: No Defense (original attack, already known) ────
# Only the combined total is confirmed (107/500, 21.4%). Shown as
# a two-way split below.
COND1_HALLUCINATED = 107
COND1_NONE          = TOTAL_ENTRIES - COND1_HALLUCINATED

# ── Condition 2: Minimal Prompt Defense (already known) ─────────
COND2_TARGETED   = 60
COND2_UNTARGETED = 36
COND2_NONE       = 404

# ── Condition 3: Structured Hallucination-Analysis Defense (read live) ───
COND3_STATE_PATH = "outputs_step8_reclassify_all_qwen25_hallucination_analysis/entry_state.json"

with open(COND3_STATE_PATH, "r") as f:
    cond3_state = json.load(f)

cond3_targeted   = sum(1 for s in cond3_state.values() if s.get("new_status") == "TARGETED")
cond3_untargeted = sum(1 for s in cond3_state.values() if s.get("new_status") == "UNTARGETED")
cond3_none       = sum(1 for s in cond3_state.values() if s.get("new_status") == "NONE")
cond3_total      = cond3_targeted + cond3_untargeted + cond3_none

CHART_PATH = ("outputs_step8_reclassify_all_qwen25_hallucination_analysis/"
              "comparison_three_defense_conditions_qwen25.png")

# ── Print summary to terminal ────────────────────────────────────
cond1_hall_pct = 100 * COND1_HALLUCINATED / TOTAL_ENTRIES
cond2_total    = COND2_TARGETED + COND2_UNTARGETED + COND2_NONE
cond2_hall_pct = 100 * (COND2_TARGETED + COND2_UNTARGETED) / cond2_total
cond3_hall_pct = 100 * (cond3_targeted + cond3_untargeted) / cond3_total if cond3_total else 0

print(f"Dataset: HaluEval (derived from HotpotQA)")
print(f"Victim model: Qwen2.5-VL 7B  |  Judge model: Gemma-4-abliterated")
print(f"Total entries: {TOTAL_ENTRIES}")
print()
print(f"Condition 1, No Defense                      : "
      f"{COND1_HALLUCINATED}/{TOTAL_ENTRIES} ({cond1_hall_pct:.1f}%) hallucinated")
print(f"Condition 2, Minimal Prompt Defense           : "
      f"{COND2_TARGETED + COND2_UNTARGETED}/{cond2_total} "
      f"({cond2_hall_pct:.1f}%) hallucinated")
print(f"Condition 3, Structured Hallucination-Analysis Defense: "
      f"{cond3_targeted + cond3_untargeted}/{cond3_total} "
      f"({cond3_hall_pct:.1f}%) hallucinated")

# ── Build the three-panel figure ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 7))

colors_3way = ["#e74c3c", "#f39c12", "#2ecc71"]
colors_2way = ["#e74c3c", "#2ecc71"]
explode_3   = (0.05, 0.05, 0.05)
explode_2   = (0.05, 0.05)

# Panel 1: No Defense (two-way split, exact breakdown not confirmed)
ax = axes[0]
sizes = [COND1_HALLUCINATED, COND1_NONE]
labels = [
    f"Hallucinated\n({COND1_HALLUCINATED}, {cond1_hall_pct:.1f}%)",
    f"No Hallucination\n({COND1_NONE}, {100-cond1_hall_pct:.1f}%)",
]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors_2way, explode=explode_2,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9},
    pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(11)
ax.set_title(f"Condition 1: No Defense\n(Original Attack)\n"
             f"Total Hallucination: {cond1_hall_pct:.1f}%",
             fontsize=12, fontweight="bold")
ax.axis("equal")

# Panel 2: Minimal Prompt Defense (3-way split)
ax = axes[1]
sizes = [COND2_TARGETED, COND2_UNTARGETED, COND2_NONE]
labels = [
    f"Targeted\n({COND2_TARGETED}, {100*COND2_TARGETED/cond2_total:.1f}%)",
    f"Untargeted\n({COND2_UNTARGETED}, {100*COND2_UNTARGETED/cond2_total:.1f}%)",
    f"No Hallucination\n({COND2_NONE}, {100*COND2_NONE/cond2_total:.1f}%)",
]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors_3way, explode=explode_3,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9},
    pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(11)
ax.set_title(f"Condition 2: Minimal Prompt Defense\n"
             f"(short caution/reminder wrap)\n"
             f"Total Hallucination: {cond2_hall_pct:.1f}%",
             fontsize=12, fontweight="bold")
ax.axis("equal")

# Panel 3: Structured Hallucination-Analysis Defense (3-way split)
ax = axes[2]
sizes = [cond3_targeted, cond3_untargeted, cond3_none]
labels = [
    f"Targeted\n({cond3_targeted}, {100*cond3_targeted/cond3_total:.1f}%)",
    f"Untargeted\n({cond3_untargeted}, {100*cond3_untargeted/cond3_total:.1f}%)",
    f"No Hallucination\n({cond3_none}, {100*cond3_none/cond3_total:.1f}%)",
]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors_3way, explode=explode_3,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9},
    pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(11)
ax.set_title(f"Condition 3: Structured Hallucination-Analysis Defense\n"
             f"(tagged risk framing + verification rule)\n"
             f"Total Hallucination: {cond3_hall_pct:.1f}%",
             fontsize=12, fontweight="bold")
ax.axis("equal")

fig.suptitle(
    f"Effect of Prompt-Level Defenses on Hallucination Rate\n"
    f"Dataset: HaluEval (derived from HotpotQA)  |  Victim Model: "
    f"Qwen2.5-VL 7B  |  Judge Model: Gemma-4-abliterated  |  "
    f"Total Entries: {TOTAL_ENTRIES}\n"
    f"All three conditions reuse the same adversarial images; only "
    f"the Step 8 instruction wording differs across conditions",
    fontsize=13, fontweight="bold", y=1.08
)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved three-way comparison chart to: {CHART_PATH}")
