# ============================================================
# make_iterations_to_first_hallucination_gemma_nqswap.py
# ============================================================
# For the NQ-Swap entries whose FINAL status was a hallucination
# (TARGETED or UNTARGETED) under Gemma's original attack, scans
# per-iteration judge files to find the FIRST iteration at which
# each entry hallucinated.
#
# IMPORTANT: entries 0-49 (outputs_strict_batch_nqswap_25 and
# outputs_strict_batch_nqswap_26_50) originally used the older
# binary judge scheme at the PER-ITERATION level too, not just in
# the final summary. The reclassification pass already produced
# corrected per-iteration files for these two batches
# (step9_reclassified_iterN.txt), which this script reads instead
# of the original step9_hallucination_iterN.txt files for those
# two batches specifically.
#
# Entries 50-99 (outputs_strict_batch_nqswap_51_75 and
# outputs_strict_batch_nqswap_76_100) used the correct native
# 3-way scheme from the start, so their original
# step9_hallucination_iterN.txt files are read directly.
#
# Saves the chart to:
#   iterations_to_first_hallucination_gemma_nqswap.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import glob
import re
import json
import statistics
import matplotlib.pyplot as plt

CHART_PATH = "iterations_to_first_hallucination_gemma_nqswap.png"

# batches using the RECLASSIFIED per-iteration files
RECLASSIFIED_BATCHES = [
    "outputs_strict_batch_nqswap_25",
    "outputs_strict_batch_nqswap_26_50",
]

# batches using the native per-iteration files directly
NATIVE_BATCHES = [
    "outputs_strict_batch_nqswap_51_75",
    "outputs_strict_batch_nqswap_76_100",
]

RECLASSIFIED_SUMMARY_PATH = "outputs_strict_batch_nqswap_26_50/RECLASSIFIED_SUMMARY.json"


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


first_iterations = []
missing = []

# ── Entries 0-49: use reclassified per-iteration files ────────────
with open(RECLASSIFIED_SUMMARY_PATH, "r") as f:
    reclassified = json.load(f)

for r in reclassified:
    if r["final_category"] not in ("TARGETED", "UNTARGETED"):
        continue
    batch_dir = r["batch_dir"]
    entry_num = r["entry_num"]
    entry_dir = f"{batch_dir}/entry_{entry_num}"
    iter_files = sorted(
        glob.glob(f"{entry_dir}/step9_reclassified_iter*.txt"),
        key=lambda p: int(re.search(r"iter(\d+)", p).group(1))
    )
    if not iter_files:
        missing.append((batch_dir, entry_num))
        continue
    categories = [classify_file(f) for f in iter_files]
    first_iter = first_hallucination_iteration(categories)
    if first_iter is not None:
        first_iterations.append(first_iter)
    else:
        missing.append((batch_dir, entry_num))

# ── Entries 50-99: use native per-iteration files directly ────────
for batch_dir in NATIVE_BATCHES:
    with open(f"{batch_dir}/SUMMARY.json", "r") as f:
        summary = json.load(f)
    for r in summary:
        if r["status"] not in ("TARGETED", "UNTARGETED"):
            continue
        entry_num = r["entry_num"]
        entry_dir = f"{batch_dir}/entry_{entry_num}"
        iter_files = sorted(
            glob.glob(f"{entry_dir}/step9_hallucination_iter*.txt"),
            key=lambda p: int(re.search(r"iter(\d+)", p).group(1))
        )
        if not iter_files:
            missing.append((batch_dir, entry_num))
            continue
        categories = [classify_file(f) for f in iter_files]
        first_iter = first_hallucination_iteration(categories)
        if first_iter is not None:
            first_iterations.append(first_iter)
        else:
            missing.append((batch_dir, entry_num))

print(f"Total final-hallucinated NQ-Swap entries found: "
      f"{len(first_iterations)}")
if missing:
    print(f"Could not determine for {len(missing)} entries:")
    for m in missing[:10]:
        print(f"  {m[0]} entry #{m[1]}")

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
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{count}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    ax.set_xlabel("Iteration of First Hallucination", fontsize=12)
    ax.set_ylabel("Number of Entries", fontsize=12)
    ax.set_xticks(range(1, 6))
    ax.set_title(
        f"Iterations to First Hallucination — Gemma-4-abliterated\n"
        f"Dataset: NQ-Swap (Longpre et al.)  |  "
        f"{len(first_iterations)} Final-Hallucinated Entries (of 100 total)\n"
        f"Average: {avg_iter:.2f}  |  Median: {median_iter}",
        fontsize=12, fontweight="bold", pad=15
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved chart to: {CHART_PATH}")
