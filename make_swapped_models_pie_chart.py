# ============================================================
# make_swapped_models_pie_chart.py
# ============================================================
# Reads outputs_swapped_models_28/SUMMARY.json (produced by
# pipeline_swapped_models.py — Qwen2.5-VL as victim/judge,
# FLUX.1-Kontext-dev as image generator) and generates a pie
# chart showing the breakdown of:
#   - TARGETED hallucination
#   - UNTARGETED hallucination
#   - NONE (no hallucination)
#
# Saves the chart as a PNG inside outputs_swapped_models_28/.
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

BASE_OUTPUT_DIR = "outputs_swapped_models_28"
SUMMARY_PATH    = f"{BASE_OUTPUT_DIR}/SUMMARY.json"
CHART_PATH      = f"{BASE_OUTPUT_DIR}/swapped_models_pie_chart.png"

# ── Load results ────────────────────────────────────────────────
with open(SUMMARY_PATH, "r") as f:
    results = json.load(f)

total      = len(results)
targeted   = sum(1 for r in results if r["status"] == "TARGETED")
untargeted = sum(1 for r in results if r["status"] == "UNTARGETED")
none_hall  = sum(1 for r in results if r["status"] == "NONE")
skipped    = sum(1 for r in results if r["status"] == "SKIPPED_NO_PHOTO")

scored_total = targeted + untargeted + none_hall  # excludes skipped

print(f"Total entries             : {total}")
print(f"Skipped (no reference photo) : {skipped}")
print(f"Scored total               : {scored_total}")
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

    # ── Build the pie chart ────────────────────────────────────
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
        f"Hallucination Classification — Swapped Models "
        f"({scored_total} Entries)\n"
        f"Victim/Judge: Qwen2.5-VL 7B  |  Image Generator: "
        f"FLUX.1-Kontext-dev\n"
        f"Total Hallucination Rate: {total_hall_pct:.1f}% "
        f"(Targeted + Untargeted)",
        fontsize=12, fontweight="bold", pad=20
    )

    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved chart to: {CHART_PATH}")
