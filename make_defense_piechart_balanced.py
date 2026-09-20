# ============================================================
# make_defense_piechart_balanced.py
# ============================================================
# Pie chart for the BALANCED WARNING defense condition
# (outputs_defense_gemma4_balanced) — Step 8's prompt told the
# victim model the image MAY be misleading, but explicitly
# instructed it not to disregard the photo — weigh both the image
# and the text together rather than favoring either one.
#
# Shows, out of all previously-hallucinated entries (TARGETED or
# UNTARGETED in the original Gemma-victim attack), what fraction
# were successfully "defended" (flipped to the correct answer) vs.
# still hallucinating after up to 5 reprompt attempts.
#
# Saves the chart to:
#   outputs_defense_gemma4_balanced/defense_piechart_balanced.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

STATE_PATH = "outputs_defense_gemma4_balanced/entry_state.json"
CHART_PATH = "outputs_defense_gemma4_balanced/defense_piechart_balanced.png"

with open(STATE_PATH, "r") as f:
    state = json.load(f)

succeeded = sum(1 for s in state.values() if s.get("defense_succeeded") is True)
failed    = sum(1 for s in state.values() if s.get("defense_succeeded") is False)
errors    = sum(1 for s in state.values() if s.get("status") == "ERROR")
total     = len(state)
scored_total = succeeded + failed

print(f"Total previously-hallucinated entries: {total}")
print(f"Errors (skipped)                     : {errors}")
print(f"Scored total                         : {scored_total}")
print()
print(f"Defense SUCCEEDED (flipped to correct): {succeeded} "
      f"({100*succeeded/scored_total:.1f}%)")
print(f"Defense FAILED (still hallucinating)  : {failed} "
      f"({100*failed/scored_total:.1f}%)")

succeeded_pct = 100 * succeeded / scored_total
failed_pct    = 100 * failed / scored_total

labels = [
    f"Defense Succeeded\n({succeeded} entries, {succeeded_pct:.1f}%)",
    f"Defense Failed\n({failed} entries, {failed_pct:.1f}%)",
]
sizes  = [succeeded, failed]
colors = ["#2ecc71", "#e74c3c"]
explode = (0.05, 0.05)

fig, ax = plt.subplots(figsize=(9, 7))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    colors=colors,
    explode=explode,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 11},
    pctdistance=0.75,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)

for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontweight("bold")
    autotext.set_fontsize(13)

ax.set_title(
    f"Defense Mechanism Result — Warning Message Prompt Output\n"
    f"Warning Style: Balanced (image and text weighed together, "
    f"neither disregarded)\n"
    f"Victim (reprompted) + Judge: Gemma-4-abliterated  |  "
    f"({scored_total} previously-hallucinated entries, "
    f"up to 5 reprompt attempts each)",
    fontsize=12, fontweight="bold", pad=20
)

ax.axis("equal")
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved chart to: {CHART_PATH}")
