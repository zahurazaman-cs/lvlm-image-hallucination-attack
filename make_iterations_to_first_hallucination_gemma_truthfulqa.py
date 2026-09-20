# ============================================================
# make_iterations_to_first_hallucination_gemma_truthfulqa.py
# ============================================================
# For the TruthfulQA entries whose FINAL status was a
# hallucination (TARGETED or UNTARGETED) under Gemma's original
# attack, scans per-iteration judge files to find the FIRST
# iteration at which each entry hallucinated.
#
# TruthfulQA's single batch (outputs_strict_batch_truthfulqa_22)
# was built directly from the correct native 3-way template from
# the start, so no reclassification is needed here — the original
# step9_hallucination_iterN.txt files are read directly.
#
# Saves the chart to:
#   iterations_to_first_hallucination_gemma_truthfulqa.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import glob
import re
import json
import statistics
import matplotlib.pyplot as plt

CHART_PATH = "iterations_to_first_hallucination_gemma_truthfulqa.png"
BATCH_DIR = "outputs_strict_batch_truthfulqa_22"


def classify_file(path):
    with open(path, "r") as f:
        text = f.read()
    if "HALLUCINATING_TARGETED: YES" in text:
        return "TARGETED"
    elif "HALLUCINATING_UNTARGETED: YES" in text:
        return "UNTARGETED"
    else:
        return "NONE"


def first_hallucination_iteration(categories):
    for i, c in enumerate(categories, start=1):
        if c in ("TARGETED", "UNTARGETED"):
            return i
    return None


with open(f"{BATCH_DIR}/SUMMARY.json", "r") as f:
    summary = json.load(f)

first_iterations = []
missing = []

for r in summary:
    if r["status"] not in ("TARGETED", "UNTARGETED"):
        continue
    entry_num = r["entry_num"]
    entry_dir = f"{BATCH_DIR}/entry_{entry_num}"
    iter_files = sorted(
        glob.glob(f"{entry_dir}/step9_hallucination_iter*.txt"),
        key=lambda p: int(re.search(r"iter(\d+)", p).group(1))
    )
    if not iter_files:
        missing.append(entry_num)
        continue
    categories = [classify_file(f) for f in iter_files]
    first_iter = first_hallucination_iteration(categories)
    if first_iter is not None:
        first_iterations.append(first_iter)
    else:
        missing.append(entry_num)

print(f"Total final-hallucinated TruthfulQA entries found: "
      f"{len(first_iterations)}")
if missing:
    print(f"Could not determine for entries: {missing}")

if not first_iterations:
    print("No entries found — nothing to chart.")
else:
    avg_iter = statistics.mean(first_iterations)
    median_iter = statistics.median(first_iterations)

    print()
    print(f"Average iteration of first hallucination: {avg_iter:.2f}")
    print(f"Median iteration of first hallucination : {median_iter}")
    for i in range(1, 6):
        c = first_iterations.count(i)
        pct = 100 * c / len(first_iterations)
        print(f"  Iteration {i}: {c} entries ({pct:.1f}%)")

    counts = [first_iterations.count(i) for i in range(1, 6)]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(range(1, 6), counts, color="#e74c3c", edgecolor="white",
                  linewidth=1.5, width=0.6)

    for bar, count in zip(bars, counts):
        pct = 100 * count / len(first_iterations)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{count}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    ax.set_xlabel("Iteration of First Hallucination", fontsize=12)
    ax.set_ylabel("Number of Entries", fontsize=12)
    ax.set_xticks(range(1, 6))
    ax.set_title(
        f"Iterations to First Hallucination — Gemma-4-abliterated\n"
        f"Dataset: TruthfulQA (domenicrosati/TruthfulQA, Lin et al. "
        f"2021)  |  {len(first_iterations)} Final-Hallucinated "
        f"Entries (of 22 total)\n"
        f"Average: {avg_iter:.2f}  |  Median: {median_iter}",
        fontsize=12, fontweight="bold", pad=15
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved chart to: {CHART_PATH}")
