# ============================================================
# make_defense_comparison_three_way.py
# ============================================================
# Side-by-side comparison of THREE prompting conditions, all
# applied to the SAME 332 entries — every entry that successfully
# hallucinated under the original attack pipeline:
#
#   1. ONLY IMAGE (baseline)  — the ORIGINAL attack's Step 8 prompt,
#      which explicitly told the victim to treat the image as the
#      PRIMARY source of evidence. Since these 332 entries were
#      filtered specifically as "hallucinated under this condition,"
#      this baseline is 0% recovered / 100% hallucinated BY
#      DEFINITION — it is not measured here, it is the starting
#      point the other two conditions are compared against.
#
#   2. ONLY TEXT (outputs_defense_gemma4_textonly) — Step 8's
#      prompt told the victim the article's TEXT is authoritative
#      and to disregard the image when the two conflict.
#
#   3. BALANCED (outputs_defense_gemma4_balanced) — Step 8's prompt
#      told the victim the image MAY be misleading, but explicitly
#      instructed it not to disregard the photo — weigh both
#      sources together.
#
# All three conditions reuse the EXACT SAME adversarial image and
# newspaper per entry — only the Step 8 prompt wording differs.
#
# Saves the chart to:
#   outputs_defense_gemma4_balanced/defense_comparison_three_way.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

STATE_TEXTONLY = "outputs_defense_gemma4_textonly/entry_state.json"
STATE_BALANCED = "outputs_defense_gemma4_balanced/entry_state.json"
CHART_PATH     = "outputs_defense_gemma4_balanced/defense_comparison_three_way.png"


def load_counts(path):
    with open(path, "r") as f:
        state = json.load(f)
    succeeded = sum(1 for s in state.values() if s.get("defense_succeeded") is True)
    failed    = sum(1 for s in state.values() if s.get("defense_succeeded") is False)
    return succeeded, failed


textonly_succeeded, textonly_failed = load_counts(STATE_TEXTONLY)
balanced_succeeded, balanced_failed = load_counts(STATE_BALANCED)

textonly_total = textonly_succeeded + textonly_failed
balanced_total = balanced_succeeded + balanced_failed

# Condition 1: Only Image — the original attack baseline. These are
# EXACTLY the entries that hallucinated under the image-prioritizing
# prompt, so by definition 0 succeeded / all failed.
image_total     = textonly_total  # same entry pool
image_succeeded = 0
image_failed    = image_total

print(f"ONLY IMAGE (baseline)  : {image_succeeded}/{image_total} recovered "
      f"(0.0% by definition — these are the entries that hallucinated "
      f"under this exact condition)")
print(f"ONLY TEXT              : {textonly_succeeded}/{textonly_total} "
      f"recovered ({100*textonly_succeeded/textonly_total:.1f}%)")
print(f"BALANCED               : {balanced_succeeded}/{balanced_total} "
      f"recovered ({100*balanced_succeeded/balanced_total:.1f}%)")

colors = ["#2ecc71", "#e74c3c"]
explode = (0.05, 0.05)

conditions = [
    ("Only Image\n(Original Attack Baseline)", image_succeeded, image_failed, image_total),
    ("Only Text\n(Text-Focused Warning)", textonly_succeeded, textonly_failed, textonly_total),
    ("Balanced\n(Image + Text Warning)", balanced_succeeded, balanced_failed, balanced_total),
]

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, (title, succeeded, failed, total) in zip(axes, conditions):
    succeeded_pct = 100 * succeeded / total
    failed_pct    = 100 * failed / total

    labels = [
        f"Recovered\n({succeeded}, {succeeded_pct:.1f}%)",
        f"Still Hallucinating\n({failed}, {failed_pct:.1f}%)",
    ]
    sizes = [succeeded, failed]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10},
        pctdistance=0.75, wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(12)

    ax.set_title(f"{title}\n({total} entries)", fontsize=13,
                 fontweight="bold", pad=15)
    ax.axis("equal")

fig.suptitle(
    "Defense Mechanism Comparison — Warning Message Prompt Output\n"
    "Same 332 previously-hallucinated entries, same reused adversarial "
    "images — only Step 8's prompt wording differs across conditions",
    fontsize=15, fontweight="bold", y=1.05
)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved three-way comparison chart to: {CHART_PATH}")
