# ============================================================
# combine_piechart_till_900_qwen25_seq.py
# ============================================================
# Combines results from ALL FOURTEEN Qwen2.5-VL victim-model
# batches — the FULL sequential reuse-method series, covering the
# same entries as all fourteen Gemma-victim batches (lines 1-900
# of qa_data.json, the complete 500-entry dataset):
#   - outputs_victim_qwen_28/entry_state.json          (28 entries,
#     two-phase pipeline, fresh image regeneration)
#   - outputs_victim_qwen_50/entry_state.json           (23 entries,
#     two-phase pipeline, fresh image regeneration)
#   - outputs_victim_qwen_151_200/entry_state.json      (26 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_201_280/entry_state.json      (48 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_281_360/entry_state.json      (50 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_361_440/entry_state.json      (50 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_441_520/entry_state.json      (49 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_521_600/entry_state.json      (47 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_601_650_reuse/entry_state.json (30 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run; NOTE
#     distinct folder name to avoid colliding with the earlier
#     abandoned two-phase attempt at this same batch range)
#   - outputs_victim_qwen_651_700/entry_state.json      (30 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_701_750/entry_state.json      (30 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_751_800/entry_state.json      (30 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_801_850/entry_state.json      (30 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#   - outputs_victim_qwen_851_900/entry_state.json      (30 entries,
#     reuse method — Steps 1-7 reused from Gemma-victim run)
#
# Victim model (target model, Step 8): Qwen2.5-VL 7B (via Ollama)
# Judge model (Step 9 classifier): Gemma-4-abliterated (UNCHANGED
# across all fourteen batches — only the victim model is the
# experimental variable)
#
# Produces ONE combined pie chart across all 501 entries showing:
#   - TARGETED hallucination
#   - UNTARGETED hallucination
#   - NONE (no hallucination)
#
# This is the FULL, complete counterpart to the Gemma-victim
# 500-entry combined pie chart — the final head-to-head comparison
# across the entire dataset.
#
# Saves the chart to:
#   outputs_victim_qwen_851_900/combined_piechart_till_900_qwen25_seq.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

STATE_28              = "outputs_victim_qwen_28/entry_state.json"
STATE_50              = "outputs_victim_qwen_50/entry_state.json"
STATE_151_200         = "outputs_victim_qwen_151_200/entry_state.json"
STATE_201_280         = "outputs_victim_qwen_201_280/entry_state.json"
STATE_281_360         = "outputs_victim_qwen_281_360/entry_state.json"
STATE_361_440         = "outputs_victim_qwen_361_440/entry_state.json"
STATE_441_520         = "outputs_victim_qwen_441_520/entry_state.json"
STATE_521_600         = "outputs_victim_qwen_521_600/entry_state.json"
STATE_601_650_REUSE   = "outputs_victim_qwen_601_650_reuse/entry_state.json"
STATE_651_700         = "outputs_victim_qwen_651_700/entry_state.json"
STATE_701_750         = "outputs_victim_qwen_701_750/entry_state.json"
STATE_751_800         = "outputs_victim_qwen_751_800/entry_state.json"
STATE_801_850         = "outputs_victim_qwen_801_850/entry_state.json"
STATE_851_900         = "outputs_victim_qwen_851_900/entry_state.json"
CHART_PATH = ("outputs_victim_qwen_851_900/"
              "combined_piechart_till_900_qwen25_seq.png")


def load_state(path):
    with open(path, "r") as f:
        return json.load(f)


state_28            = load_state(STATE_28)
state_50            = load_state(STATE_50)
state_151_200       = load_state(STATE_151_200)
state_201_280       = load_state(STATE_201_280)
state_281_360       = load_state(STATE_281_360)
state_361_440       = load_state(STATE_361_440)
state_441_520       = load_state(STATE_441_520)
state_521_600       = load_state(STATE_521_600)
state_601_650_reuse = load_state(STATE_601_650_REUSE)
state_651_700       = load_state(STATE_651_700)
state_701_750       = load_state(STATE_701_750)
state_751_800       = load_state(STATE_751_800)
state_801_850       = load_state(STATE_801_850)
state_851_900       = load_state(STATE_851_900)

all_statuses = (
    [s["status"] for s in state_28.values()] +
    [s["status"] for s in state_50.values()] +
    [s["status"] for s in state_151_200.values()] +
    [s["status"] for s in state_201_280.values()] +
    [s["status"] for s in state_281_360.values()] +
    [s["status"] for s in state_361_440.values()] +
    [s["status"] for s in state_441_520.values()] +
    [s["status"] for s in state_521_600.values()] +
    [s["status"] for s in state_601_650_reuse.values()] +
    [s["status"] for s in state_651_700.values()] +
    [s["status"] for s in state_701_750.values()] +
    [s["status"] for s in state_751_800.values()] +
    [s["status"] for s in state_801_850.values()] +
    [s["status"] for s in state_851_900.values()]
)

targeted   = sum(1 for c in all_statuses if c == "TARGETED")
untargeted = sum(1 for c in all_statuses if c == "UNTARGETED")
none_hall  = sum(1 for c in all_statuses if c == "NONE")
skipped    = sum(1 for c in all_statuses if c == "SKIPPED_NO_PHOTO")

scored_total = targeted + untargeted + none_hall
grand_total  = len(all_statuses)

print(f"Batch 28 entries              : {len(state_28)}")
print(f"Batch 23 entries              : {len(state_50)}")
print(f"Batch 151-200 entries         : {len(state_151_200)}")
print(f"Batch 201-280 entries         : {len(state_201_280)}")
print(f"Batch 281-360 entries         : {len(state_281_360)}")
print(f"Batch 361-440 entries         : {len(state_361_440)}")
print(f"Batch 441-520 entries         : {len(state_441_520)}")
print(f"Batch 521-600 entries         : {len(state_521_600)}")
print(f"Batch 601-650 (reuse) entries : {len(state_601_650_reuse)}")
print(f"Batch 651-700 entries         : {len(state_651_700)}")
print(f"Batch 701-750 entries         : {len(state_701_750)}")
print(f"Batch 751-800 entries         : {len(state_751_800)}")
print(f"Batch 801-850 entries         : {len(state_801_850)}")
print(f"Batch 851-900 entries         : {len(state_851_900)}")
print(f"Grand total                   : {grand_total}")
print(f"Skipped (no photo)            : {skipped}")
print(f"Scored total                  : {scored_total}")
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
        f"({scored_total} Entries, FULL Dataset)\n"
        f"Victim Model: Qwen2.5-VL 7B  |  Judge: Gemma-4-abliterated\n"
        f"(28+23+26+48+50+50+49+47+30+30+30+30+30+30 entries across "
        f"fourteen batches, {skipped} skipped)\n"
        f"Total Hallucination Rate: {total_hall_pct:.1f}% "
        f"(Targeted + Untargeted)",
        fontsize=12, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved combined chart to: {CHART_PATH}")
