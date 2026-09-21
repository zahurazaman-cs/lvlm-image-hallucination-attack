# ============================================================
# make_human_eval_agreement_chart.py
# ============================================================
# Pie chart summarizing human judge validation results — how
# often the automated judge's classification was independently
# confirmed as correct by real human raters.
#
# Based on 7 respondents, each evaluating 20 items, 140 total
# human judgments collected via Google Form.
#
# Saves the chart to:
#   human_eval_agreement_chart.png
#
# Requirements: matplotlib
# ============================================================

import matplotlib.pyplot as plt

TOTAL_RESPONDENTS = 7
ITEMS_PER_PERSON = 20
TOTAL_JUDGMENTS = TOTAL_RESPONDENTS * ITEMS_PER_PERSON  # 140

CORRECT = 111
WRONG = TOTAL_JUDGMENTS - CORRECT  # 29

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
    f"Overall Agreement Rate: {agreement_pct:.1f}%",
    fontsize=12, fontweight="bold", pad=20
)

ax.axis("equal")
plt.tight_layout()
plt.savefig("human_eval_agreement_chart.png", dpi=200, bbox_inches="tight")
print(f"Saved chart to: human_eval_agreement_chart.png")
print()
print(f"Total respondents: {TOTAL_RESPONDENTS}")
print(f"Total judgments: {TOTAL_JUDGMENTS}")
print(f"Judge confirmed correct: {CORRECT} ({agreement_pct:.1f}%)")
print(f"Judge disputed: {WRONG} ({disagreement_pct:.1f}%)")
print()
print(f"NOTE: one respondent (Arpita Ghosh) was a notable outlier, agreeing")
print(f"with the judge on only 8/20 (40.0%) items, while the other six")
print(f"respondents ranged from 85.0% to 90.0% agreement. Excluding this")
print(f"outlier: {103}/{120} = {100*103/120:.1f}% agreement among the remaining six raters.")
