# ============================================================
# make_mcnemar_table_image.py
# ============================================================
# Renders the full McNemar's test results as a clean table image,
# ready to drop directly into the results section of the paper.
#
# Reads MCNEMAR_TEST_RESULTS.json, produced by
# mcnemar_test_all_defenses.py.
#
# Saves the image to:
#   mcnemar_test_table.png
#
# Requirements: matplotlib
# ============================================================

import json
import matplotlib.pyplot as plt

with open("MCNEMAR_TEST_RESULTS.json", "r") as f:
    results = json.load(f)

rows = []
for r in results:
    label = r["label"]
    b, c, p = r["b"], r["c"], r["p_value"]

    if p is None:
        p_str = "undefined"
        sig = "no flips"
    elif p < 0.001:
        p_str = f"{p:.5f}"
        sig = "***"
    elif p < 0.01:
        p_str = f"{p:.4f}"
        sig = "**"
    elif p < 0.05:
        p_str = f"{p:.4f}"
        sig = "*"
    else:
        p_str = f"{p:.4f}"
        sig = "n.s."

    rows.append([label, str(b), str(c), p_str, sig])

col_labels = ["Condition", "b\n(fixed)", "c\n(broke)", "p-value", "Sig."]

fig, ax = plt.subplots(figsize=(14, len(rows) * 0.55 + 1.5))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    colWidths=[0.55, 0.1, 0.1, 0.15, 0.1],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

# Header styling
for col in range(len(col_labels)):
    cell = table[0, col]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# Row styling by significance, and left-align the condition column
for i, row in enumerate(rows, start=1):
    sig = row[4]
    if sig == "***":
        color = "#d5f5e3"
    elif sig == "**":
        color = "#eafaf1"
    elif sig == "*":
        color = "#fef9e7"
    else:
        color = "#fdedec" if sig == "n.s." else "#f2f3f4"
    for col in range(len(col_labels)):
        cell = table[i, col]
        cell.set_facecolor(color)
        if col == 0:
            cell.set_text_props(ha="left")
            cell._text.set_horizontalalignment("left")

ax.set_title(
    "McNemar's Exact Test — Paired Before/After Results for Every "
    "Prompt-Level Defense Condition\n"
    "b = entries flipped hallucinating \u2192 correct, "
    "c = entries flipped correct \u2192 hallucinating\n"
    "Significance: * p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant",
    fontsize=12, fontweight="bold", pad=20
)

plt.tight_layout()
plt.savefig("mcnemar_test_table.png", dpi=200, bbox_inches="tight")
print("Saved table image to: mcnemar_test_table.png")
