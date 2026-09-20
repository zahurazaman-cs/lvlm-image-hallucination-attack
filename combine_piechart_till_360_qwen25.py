# ============================================================
# combine_piechart_till_360_qwen25.py
# ============================================================
# Combines results from ALL FIVE Qwen2.5-VL victim-model batches
# completed so far, covering the same entries as the first five
# Gemma-victim batches (lines 1-360 of qa_data.json):
#   - outputs_victim_qwen_28/entry_state.json       (28 entries,
#     two-phase pipeline, fresh image regeneration)
#   - outputs_victim_qwen_50/entry_state.json        (23 entries,
#     two-phase pipeline, fresh image regeneration)
#   - outputs_victim_qwen_151_200/entry_state.json   (26 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_201_280/entry_state.json   (48 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_281_360/entry_state.json   (50 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#
# Victim model (target model, Step 8): Qwen2.5-VL 7B (via Ollama)
# Judge model (Step 9 classifier): Gemma-4-abliterated (UNCHANGED
# across all five batches — only the victim model is the
# experimental variable)
#
# Produces ONE combined pie chart across all 175 entries showing:
#   - TARGETED hallucination
#   - UNTARGETED hallucination
#   - NONE (no hallucination)
#
# Saves the chart to:
#   outputs_victim_qwen_281_360/combined_piechart_till_360_qwen25.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

STATE_28      = "outputs_victim_qwen_28/entry_state.json"
STATE_50      = "outputs_victim_qwen_50/entry_state.json"
STATE_151_200 = "outputs_victim_qwen_151_200/entry_state.json"
STATE_201_280 = "outputs_victim_qwen_201_280/entry_state.json"
STATE_281_360 = "outputs_victim_qwen_281_360/entry_state.json"
CHART_PATH    = "outputs_victim_qwen_281_360/combined_piechart_till_360_qwen25.png"


def load_state(path):
    with open(path, "r") as f:
        return json.load(f)


state_28      = load_state(STATE_28)
state_50      = load_state(STATE_50)
state_151_200 = load_state(STATE_151_200)
state_201_280 = load_state(STATE_201_280)
state_281_360 = load_state(STATE_281_360)

all_statuses = (
    [s["status"] for s in state_28.values()] +
    [s["status"] for s in state_50.values()] +
    [s["status"] for s in state_151_200.values()] +
    [s["status"] for s in state_201_280.values()] +
    [s["status"] for s in state_281_360.values()]
)

targeted   = sum(1 for c in all_statuses if c == "TARGETED")
untargeted = sum(1 for c in all_statuses if c == "UNTARGETED")
none_hall  = sum(1 for c in all_statuses if c == "NONE")
skipped    = sum(1 for c in all_statuses if c == "SKIPPED_NO_PHOTO")

scored_total = targeted + untargeted + none_hall
grand_total  = len(all_statuses)

print(f"Batch 28 entries          : {len(state_28)}")
print(f"Batch 23 entries          : {len(state_50)}")
print(f"Batch 151-200 entries     : {len(state_151_200)}")
print(f"Batch 201-280 entries     : {len(state_201_280)}")
print(f"Batch 281-360 entries     : {len(state_281_360)}")
print(f"Grand total               : {grand_total}")
print(f"Skipped (no photo)        : {skipped}")
print(f"Scored total              : {scored_total}")
print()

if scored_total == 0:
    print("No scored entries found — nothing to chart.")
else:
    targeted_pct   = 100 * targeted / scored_total
    untargeted_pct = 100 * untargeted / scored_total
    none_pct       = 100 * none_hall / scored_total
    total_hall_pct = 100 * (targeted + untargeted) / scored_total

    print(f"TARGETED hallucination   : {targeted} ({targeted_pct:.1f}%)")
    print(f"UNTARGETED hallucination : {untargeted} ({untargeted_pct:.1f}%)")
    print(f"NOT hallucinating        : {none_hall} ({none_pct:.1f}%)")
    print(f"TOTAL hallucination      : {targeted + untargeted} "
          f"({total_hall_pct:.1f}%)")

    labels = [
        f"Targeted Hallucination\n({targeted} entries, {targeted_pct:.1f}%)",
        f"Untargeted Hallucination\n({untargeted} entries, {untargeted_pct:.1f}%)",
        f"No Hallucination\n({none_hall} entries, {none_pct:.1f}%)",
    ]
    sizes  = [targeted, untargeted, none_hall]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    explode = (0.05, 0.05, 0.05)

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
        f"Hallucination Classification — Victim Model Swap "
        f"({scored_total} Entries)\n"
        f"Victim Model: Qwen2.5-VL 7B  |  Judge: Gemma-4-abliterated\n"
        f"(28 + 23 + 26 + 48 + 50 entries across five batches, "
        f"{skipped} skipped)\n"
        f"Total Hallucination Rate: {total_hall_pct:.1f}% "
        f"(Targeted + Untargeted)",
        fontsize=12, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved combined chart to: {CHART_PATH}")
