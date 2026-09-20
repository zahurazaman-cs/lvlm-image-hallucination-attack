# ============================================================
# make_human_eval_agreement_chart_v2.py
# ============================================================
# Pie chart summarizing human judge validation results (Question B
# — reflective agreement), now with 8 respondents, 160 total
# judgments.
#
# Saves the chart to: human_eval_agreement_chart_v2.png
# ============================================================

import matplotlib.pyplot as plt

TOTAL_RESPONDENTS = 8
ITEMS_PER_PERSON = 20
TOTAL_JUDGMENTS = TOTAL_RESPONDENTS * ITEMS_PER_PERSON  # 160

CORRECT = 127
WRONG = TOTAL_JUDGMENTS - CORRECT  # 33

agreement_pct = 100 * CORRECT / TOTAL_JUDGMENTS
disagreement_pct = 100 * WRONG / TOTAL_JUDGMENTS

fig, ax = plt.subplots(figsize=(9, 7))

labels = [
    f"Judge Classification Confirmed\n({CORRECT} of {TOTAL_JUDGMENTS} judgments, {agreement_pct:.1f}%)",
    f"Judge Classification Disputed\n({WRONG} of {TOTAL_JUDGMENTS} judgments, {disagreement_pct:.1f}%)",
]
sizes = [CORRECT, WRONG]
colors = ["#2ecc71", "#e74c3c"]
explode = (0.05, 0.05)

wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, colors=colors, explode=explode,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12},
    pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_color("white")
    at.set_fontweight("bold")
    at.set_fontsize(15)

ax.set_title(
    f"Human Validation of the Automated Judge\n"
    f"{TOTAL_RESPONDENTS} independent human annotators, {ITEMS_PER_PERSON} randomly "
    f"sampled entries each, {TOTAL_JUDGMENTS} total judgments\n"
    f"Overall Reflective Agreement Rate: {agreement_pct:.1f}%",
    fontsize=12, fontweight="bold", pad=20
)

ax.axis("equal")
plt.tight_layout()
plt.savefig("human_eval_agreement_chart_v2.png", dpi=200, bbox_inches="tight")
print(f"Saved chart to: human_eval_agreement_chart_v2.png")
