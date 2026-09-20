# ============================================================
# combine_piechart_nqswap_till_50.py
# ============================================================
# Combines results from the NQ-Swap real-world generalization
# test — NOT qa_data.json/HaluEval — across both completed batches:
#   - outputs_strict_batch_nqswap_25/SUMMARY.json      (25 entries)
#   - outputs_strict_batch_nqswap_26_50/SUMMARY.json   (25 entries)
#
# Victim + Judge: Gemma-4-abliterated (same as the primary attack
# pipeline)  |  Image Generator: Qwen-Image-Edit-2509 (unchanged)
#
# NQ-Swap (pminervini/NQ-Swap, "Entity-Based Knowledge Conflicts in
# Question Answering", Longpre et al.) is a genuinely independent
# real-world dataset, NOT derived from HotpotQA/HaluEval — this
# tests whether the pipeline's original 66.4% hallucination-rate
# finding generalizes beyond the primary benchmark.
#
# Produces ONE combined pie chart across all 50 entries showing:
#   - TARGETED hallucination
#   - UNTARGETED hallucination
#   - NONE (no hallucination)
#
# Saves the chart to:
#   outputs_strict_batch_nqswap_26_50/combined_piechart_nqswap_till_50.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

BATCH_1_SUMMARY = "outputs_strict_batch_nqswap_25/SUMMARY.json"
BATCH_2_SUMMARY = "outputs_strict_batch_nqswap_26_50/SUMMARY.json"
CHART_PATH = ("outputs_strict_batch_nqswap_26_50/"
              "combined_piechart_nqswap_till_50.png")


def load_batch(path):
    with open(path, "r") as f:
        return json.load(f)


batch_1 = load_batch(BATCH_1_SUMMARY)
batch_2 = load_batch(BATCH_2_SUMMARY)

all_statuses = (
    [r["status"] for r in batch_1] +
    [r["status"] for r in batch_2]
)

hallucinated     = sum(1 for c in all_statuses if c == "HALLUCINATED")
did_not_hall     = sum(1 for c in all_statuses if c == "DID_NOT_HALLUCINATE")
skipped          = sum(1 for c in all_statuses if c == "SKIPPED_NO_PHOTO")

scored_total = hallucinated + did_not_hall
grand_total  = len(all_statuses)

print(f"Batch 1 (entries 0-24) entries  : {len(batch_1)}")
print(f"Batch 2 (entries 25-49) entries : {len(batch_2)}")
print(f"Grand total                     : {grand_total}")
print(f"Skipped (no photo)              : {skipped}")
print(f"Scored total                    : {scored_total}")
print()

if scored_total == 0:
    print("No scored entries found — nothing to chart.")
else:
    hall_pct    = 100 * hallucinated / scored_total
    no_hall_pct = 100 * did_not_hall / scored_total

    print(f"HALLUCINATED         : {hallucinated} ({hall_pct:.1f}%)")
    print(f"DID NOT HALLUCINATE  : {did_not_hall} ({no_hall_pct:.1f}%)")
    print()
    print(f"NOTE: this batch used the ORIGINAL BINARY classification "
          f"scheme (inherited from the batch-28 template), not the "
          f"native 3-way TARGETED/UNTARGETED/NONE scheme used in every "
          f"other batch. This chart shows a 2-way split (Hallucinated "
          f"vs. Did Not Hallucinate) rather than the usual 3-way one.")
    print(f"For comparison, the primary qa_data.json (HaluEval) "
          f"result across 500 entries was 66.4% total hallucination "
          f"(3-way scheme).")

    labels = [
        f"Hallucinated\n({hallucinated} entries, {hall_pct:.1f}%)",
        f"Did Not Hallucinate\n({did_not_hall} entries, {no_hall_pct:.1f}%)",
    ]
    sizes  = [hallucinated, did_not_hall]
    colors = ["#e74c3c", "#2ecc71"]
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
        f"Hallucination Classification — NQ-Swap Real-World "
        f"Generalization Test ({scored_total} Entries)\n"
        f"Victim + Judge: Gemma-4-abliterated  |  Image Generator: "
        f"Qwen-Image-Edit-2509\n"
        f"Dataset: NQ-Swap (Longpre et al.) — independent of "
        f"HaluEval/HotpotQA, 25+25 entries across two batches\n"
        f"(Binary classification scheme — see script comments)  |  "
        f"Total Hallucination Rate: {hall_pct:.1f}%",
        fontsize=11, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved combined chart to: {CHART_PATH}")
