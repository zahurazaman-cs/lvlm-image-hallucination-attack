# ============================================================
# make_transferability_chart_gemma_qwen25.py
# ============================================================
# Two-panel pie chart visualizing cross-model hallucination
# transferability between Gemma-4-abliterated and Qwen2.5-VL on
# qa_data.json (HaluEval), covering the FULL 500-entry dataset —
# all 332 Gemma-hallucinated entries and all 107 Qwen2.5-VL-
# hallucinated entries, combining the 449 reuse-method entries
# (same image by construction) with two small targeted cross
# tests covering the remaining 51 two-phase entries on each side.
#
#   Panel 1: Of the entries Gemma hallucinated on, what fraction
#            also fooled Qwen2.5-VL on the SAME image?
#   Panel 2: Of the entries Qwen2.5-VL hallucinated on, what
#            fraction also fooled Gemma on the SAME image?
#
# Each panel breaks the "transferred" wedge down further into
# whether the OTHER model's hallucination was itself TARGETED or
# UNTARGETED, not just a binary transferred/did-not-transfer
# split.
#
# Numbers are hardcoded from the completed
# transferability_analysis_full500_gemma_qwen25.py run — edit the
# values below if you rerun that analysis and get different
# numbers.
#
# Saves the chart to:
#   transferability_chart_gemma_qwen25.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import matplotlib.pyplot as plt

# ── Direction 1: Gemma-hallucinated -> Qwen2.5-VL outcome ────────
D1_TOTAL                   = 332
D1_TRANSFERRED_TARGETED    = 58
D1_TRANSFERRED_UNTARGETED  = 27
D1_DID_NOT_TRANSFER        = 247

# ── Direction 2: Qwen2.5-VL-hallucinated -> Gemma outcome ────────
D2_TOTAL                   = 107
D2_TRANSFERRED_TARGETED    = 72
D2_TRANSFERRED_UNTARGETED  = 19
D2_DID_NOT_TRANSFER        = 16

CHART_PATH = "transferability_chart_gemma_qwen25.png"

fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))

colors = ["#c0392b", "#e67e22", "#2ecc71"]
explode = (0.06, 0.06, 0.03)

# Panel 1
ax = axes[0]
sizes = [D1_TRANSFERRED_TARGETED, D1_TRANSFERRED_UNTARGETED, D1_DID_NOT_TRANSFER]
labels = [
    f"Transferred\n(Qwen: Targeted)\n{D1_TRANSFERRED_TARGETED} entries",
    f"Transferred\n(Qwen: Untargeted)\n{D1_TRANSFERRED_UNTARGETED} entries",
    f"Did Not Transfer\n(Qwen: Correct)\n{D1_DID_NOT_TRANSFER} entries",
]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors, explode=explode,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10},
    pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(12)
ax.set_title(
    f"Direction 1: Gemma \u2192 Qwen2.5-VL\n"
    f"Of {D1_TOTAL} entries Gemma hallucinated on\n"
    f"(same image shown to both models)",
    fontsize=12, fontweight="bold"
)
ax.axis("equal")

# Panel 2
ax = axes[1]
sizes = [D2_TRANSFERRED_TARGETED, D2_TRANSFERRED_UNTARGETED, D2_DID_NOT_TRANSFER]
labels = [
    f"Transferred\n(Gemma: Targeted)\n{D2_TRANSFERRED_TARGETED} entries",
    f"Transferred\n(Gemma: Untargeted)\n{D2_TRANSFERRED_UNTARGETED} entries",
    f"Did Not Transfer\n(Gemma: Correct)\n{D2_DID_NOT_TRANSFER} entries",
]
wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors, explode=explode,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10},
    pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(12)
ax.set_title(
    f"Direction 2: Qwen2.5-VL \u2192 Gemma\n"
    f"Of {D2_TOTAL} entries Qwen2.5-VL hallucinated on\n"
    f"(same image shown to both models)",
    fontsize=12, fontweight="bold"
)
ax.axis("equal")

fig.suptitle(
    f"Cross-Model Hallucination Transferability\n"
    f"Dataset: HaluEval (derived from HotpotQA)  |  Models: "
    f"Gemma-4-abliterated and Qwen2.5-VL 7B  |  Judge: "
    f"Gemma-4-abliterated\n"
    f"Full coverage across all 500 entries: 449 from twelve "
    f"reuse-method batches (same image by construction) plus 51 "
    f"from two dedicated cross tests covering the two two-phase "
    f"batches, closing the gap to complete 332/332 and 107/107",
    fontsize=12, fontweight="bold", y=1.07
)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"Saved transferability chart to: {CHART_PATH}")
