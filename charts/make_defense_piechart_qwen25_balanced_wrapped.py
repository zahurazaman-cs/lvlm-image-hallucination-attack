# ============================================================
# make_defense_piechart_qwen25_balanced_wrapped.py
# ============================================================
# Pie chart for the Qwen2.5-VL BALANCED WRAP-AROUND defense
# condition (outputs_defense_qwen25_balanced_wrapped) — the
# ORIGINAL Step 8 instructions were left completely UNCHANGED
# (including the line stating the image is the PRIMARY source). A
# short (1-2 sentence) CAUTION was added immediately BEFORE those
# instructions and a short REMINDER immediately AFTER them —
# identical methodology to the Gemma defense experiment, now
# applied with Qwen2.5-VL as the reprompted victim.
#
# Shows, out of all Qwen2.5-VL-hallucinated entries (TARGETED or
# UNTARGETED in the Qwen2.5-VL victim pipeline), what fraction were
# successfully "defended" (flipped to the correct answer) vs. still
# hallucinating after up to 5 reprompt attempts.
#
# Saves the chart to:
#   outputs_defense_qwen25_balanced_wrapped/defense_piechart_qwen25_balanced_wrapped.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

STATE_PATH = "outputs_defense_qwen25_balanced_wrapped/entry_state.json"
CHART_PATH = ("outputs_defense_qwen25_balanced_wrapped/"
              "defense_piechart_qwen25_balanced_wrapped.png")

with open(STATE_PATH, "r") as f:
    state = json.load(f)

succeeded = sum(1 for s in state.values() if s.get("defense_succeeded") is True)
failed    = sum(1 for s in state.values() if s.get("defense_succeeded") is False)
errors    = sum(1 for s in state.values() if s.get("status") == "ERROR")
total     = len(state)
scored_total = succeeded + failed

print(f"Total previously-hallucinated Qwen2.5-VL entries: {total}")
print(f"Errors (skipped)                                : {errors}")
print(f"Scored total                                    : {scored_total}")
print()

if scored_total == 0:
    print("No scored entries found — nothing to chart.")
else:
    succeeded_pct = 100 * succeeded / scored_total
    failed_pct    = 100 * failed / scored_total

    print(f"Defense SUCCEEDED (flipped to correct): {succeeded} "
          f"({succeeded_pct:.1f}%)")
    print(f"Defense FAILED (still hallucinating)  : {failed} "
          f"({failed_pct:.1f}%)")

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
        f"Victim (reprompted): Qwen2.5-VL 7B  |  Judge: "
        f"Gemma-4-abliterated\n"
        f"Warning Style: Balanced Wrap-Around (short CAUTION/REMINDER "
        f"added before and after the\nORIGINAL, unchanged Step 8 "
        f"instructions — image still stated as primary source)\n"
        f"({scored_total} previously-hallucinated Qwen2.5-VL entries, "
        f"up to 5 reprompt attempts each)",
        fontsize=11, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved chart to: {CHART_PATH}")
