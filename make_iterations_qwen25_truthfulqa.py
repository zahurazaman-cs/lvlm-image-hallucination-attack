# ============================================================
# make_iterations_qwen25_truthfulqa.py
# ============================================================
# Iteration distribution for Qwen2.5-VL's original attack on
# TruthfulQA. Same situation as NQ-Swap — every single one of the
# 22 entries used the reuse method, Qwen2.5-VL was never tested
# against a freshly-escalated image on this dataset at all, only
# against whichever image Gemma's own attack had already settled
# on. There is no genuine two-phase subset here, so this is a
# single distribution, not a two-panel split.
#
# The recorded "source_iteration" means "which escalation level of
# GEMMA's attack image also happened to transfer and fool
# Qwen2.5-VL," a transfer metric, not an effort metric spent
# against Qwen2.5-VL itself.
#
# Saves the chart to:
#   iterations_qwen25_truthfulqa.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import statistics
import matplotlib.pyplot as plt

CHART_PATH = "iterations_qwen25_truthfulqa.png"
STATE_PATH = "outputs_victim_qwen_truthfulqa_22/entry_state.json"

with open(STATE_PATH, "r") as f:
    state = json.load(f)

transferred_iterations = []
for s in state.values():
    if s.get("status") in ("TARGETED", "UNTARGETED"):
        si = s.get("source_iteration")
        if si is not None:
            transferred_iterations.append(si)

print(f"Total hallucinated entries: {len(transferred_iterations)}")

if not transferred_iterations:
    print("No entries found — nothing to chart.")
else:
    avg_iter = statistics.mean(transferred_iterations)
    median_iter = statistics.median(transferred_iterations)

    print(f"Average source iteration: {avg_iter:.2f}")
    print(f"Median source iteration : {median_iter}")
    for i in range(1, 6):
        c = transferred_iterations.count(i)
        pct = 100 * c / len(transferred_iterations)
        print(f"  Iteration {i}: {c} entries ({pct:.1f}%)")

    counts = [transferred_iterations.count(i) for i in range(1, 6)]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(range(1, 6), counts, color="#9b59b6", edgecolor="white",
                  linewidth=1.5, width=0.6)

    for bar, count in zip(bars, counts):
        pct = 100 * count / len(transferred_iterations)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{count}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    ax.set_xlabel("Gemma Escalation Iteration of the Reused Image", fontsize=12)
    ax.set_ylabel("Number of Entries", fontsize=12)
    ax.set_xticks(range(1, 6))
    ax.set_title(
        f"Transferred Escalation Level — Qwen2.5-VL Original Attack\n"
        f"Dataset: TruthfulQA (domenicrosati/TruthfulQA, Lin et al. "
        f"2021)  |  {len(transferred_iterations)} Hallucinated "
        f"Entries (of 22 total)\n"
        f"Average: {avg_iter:.2f}  |  Median: {median_iter}\n"
        f"NOTE: every entry used the reuse method — this shows which "
        f"escalation level of Gemma's image transferred, not attempts "
        f"against Qwen2.5-VL",
        fontsize=11, fontweight="bold", pad=15
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved chart to: {CHART_PATH}")
