# ============================================================
# make_human_view_of_victim_chart.py
# ============================================================
# Pie chart showing, according to HUMAN annotators' own
# independent judgment (Question A — given BEFORE seeing the
# judge's classification), how often the victim model's actual
# answer matched the correct answer, was simply wrong, or matched
# the hallucinated answer.
#
# This is the human-eye view of victim model accuracy, entirely
# independent of what the automated judge decided.
#
# Based on 8 respondents x 20 items = 160 total human judgments.
#
# Saves the chart to: human_view_of_victim_chart.png
# ============================================================

import matplotlib.pyplot as plt

DOC, WRONG, ALT = 62, 65, 33
TOTAL = DOC + WRONG + ALT  # 160

doc_pct = 100 * DOC / TOTAL
wrong_pct = 100 * WRONG / TOTAL
alt_pct = 100 * ALT / TOTAL

labels = [
    f"Model's Answer Was Correct\n({DOC} judgments, {doc_pct:.1f}%)",
    f"Model's Answer Was Wrong,\nMatched Neither\n({WRONG} judgments, {wrong_pct:.1f}%)",
    f"Model's Answer Matched the\nHallucinated Answer\n({ALT} judgments, {alt_pct:.1f}%)",
]
sizes = [DOC, WRONG, ALT]
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
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
    f"Victim Model Accuracy — According to Independent Human Judgment\n"
    f"8 Annotators, 20 Entries Each, {TOTAL} Total Judgments\n"
    f"Judgment given BEFORE annotators saw the automated judge's classification\n"
    f"Total Human-Perceived Hallucination Rate: {wrong_pct+alt_pct:.1f}%",
    fontsize=11, fontweight="bold", pad=20
)

ax.axis("equal")
plt.tight_layout()
plt.savefig("human_view_of_victim_chart.png", dpi=200, bbox_inches="tight")
print(f"Saved chart to: human_view_of_victim_chart.png")
