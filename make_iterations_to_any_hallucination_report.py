# ============================================================
# make_iterations_to_any_hallucination_report.py
# ============================================================
# Analyzes the ORIGINAL Gemma-4 attack pipeline across all FOURTEEN
# completed batches, asking: "On average, after how many escalation
# iterations did an entry FIRST produce a hallucinated answer,
# whether TARGETED or UNTARGETED?"
#
# IMPORTANT DESIGN NOTE: this uses the FIRST iteration at which
# hallucination occurred, scanning every iteration's Step 9 file —
# NOT each entry's final "iterations_needed" field. That field only
# reflects when TARGETED stopped the loop early; UNTARGETED entries
# always run the full 5 iterations by design (only TARGETED breaks
# early), so their final iterations_needed is always 5 and would be
# meaningless here. Scanning per-iteration files instead captures
# the true first moment of hallucination for BOTH categories,
# including entries that hallucinated early but later "recovered"
# to NONE by their final iteration.
#
# Entries that NEVER hallucinated (all iterations NONE) are
# excluded from this average, since there's no "iteration to
# hallucination" to report for them.
#
# Produces:
#   - A bar chart showing how many entries FIRST hallucinated at
#     each iteration (1 through 5)
#   - The average and median iteration of first hallucination
#
# Saves the chart to:
#   iterations_to_any_hallucination_report.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import glob
import re
import statistics
import matplotlib.pyplot as plt

CHART_PATH = "iterations_to_any_hallucination_report.png"

# Batch 28 used a different iteration-file naming convention
# (post-hoc reclassification pass) than the other thirteen batches
# (native 3-way classification from the start).
BATCH_DIRS_NATIVE = [
    "outputs_strict_batch_50",
    "outputs_strict_batch_151_200",
    "outputs_strict_batch_201_280",
    "outputs_strict_batch_281_360",
    "outputs_strict_batch_361_440_combined",
    "outputs_strict_batch_441_520",
    "outputs_strict_batch_521_600",
    "outputs_strict_batch_601_650",
    "outputs_strict_batch_651_700",
    "outputs_strict_batch_701_750",
    "outputs_strict_batch_751_800",
    "outputs_strict_batch_801_850",
    "outputs_strict_batch_851_900",
]
BATCH_DIR_28 = "outputs_strict_batch_28"


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
    """Returns the 1-indexed iteration of the first TARGETED or
    UNTARGETED result, or None if the entry never hallucinated."""
    for i, c in enumerate(categories, start=1):
        if c in ("TARGETED", "UNTARGETED"):
            return i
    return None


all_first_hall_iterations = []

# ── Batch 28: uses step9_reclassified_iterN.txt files ────────────
entry_dirs = sorted(
    glob.glob(f"{BATCH_DIR_28}/entry_[0-9]*"),
    key=lambda p: int(re.search(r"entry_(\d+)", p).group(1))
)
for entry_dir in entry_dirs:
    iter_files = sorted(
        glob.glob(f"{entry_dir}/step9_reclassified_iter*.txt"),
        key=lambda p: int(re.search(r"iter(\d+)", p).group(1))
    )
    if not iter_files:
        continue
    categories = [classify_file(f) for f in iter_files]
    first_iter = first_hallucination_iteration(categories)
    if first_iter is not None:
        all_first_hall_iterations.append(first_iter)

# ── Remaining thirteen batches: native step9_hallucination_iterN.txt ──
for batch_dir in BATCH_DIRS_NATIVE:
    entry_dirs = sorted(
        glob.glob(f"{batch_dir}/entry_[0-9]*"),
        key=lambda p: int(re.search(r"entry_(\d+)", p).group(1))
    )
    for entry_dir in entry_dirs:
        iter_files = sorted(
            glob.glob(f"{entry_dir}/step9_hallucination_iter*.txt"),
            key=lambda p: int(re.search(r"iter(\d+)", p).group(1))
        )
        if not iter_files:
            continue
        categories = [classify_file(f) for f in iter_files]
        first_iter = first_hallucination_iteration(categories)
        if first_iter is not None:
            all_first_hall_iterations.append(first_iter)

# ── Compute statistics ────────────────────────────────────────────
total_hallucinated = len(all_first_hall_iterations)
average_iter = statistics.mean(all_first_hall_iterations)
median_iter = statistics.median(all_first_hall_iterations)

counts_per_iter = {i: 0 for i in range(1, 6)}
for it in all_first_hall_iterations:
    counts_per_iter[it] = counts_per_iter.get(it, 0) + 1

print(f"Total entries that EVER hallucinated (TARGETED or "
      f"UNTARGETED, at any iteration): {total_hallucinated}")
print(f"Average iteration of FIRST hallucination: {average_iter:.2f}")
print(f"Median iteration of FIRST hallucination : {median_iter}")
print()
for i in range(1, 6):
    pct = 100 * counts_per_iter[i] / total_hallucinated if total_hallucinated else 0
    print(f"  Iteration {i}: {counts_per_iter[i]} entries ({pct:.1f}%)")

# ── Build the bar chart ──────────────────────────────────────────
iterations = list(range(1, 6))
counts = [counts_per_iter[i] for i in iterations]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(iterations, counts, color="#e67e22", edgecolor="white",
              linewidth=1.5, width=0.6)

for bar, count in zip(bars, counts):
    pct = 100 * count / total_hallucinated if total_hallucinated else 0
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{count}\n({pct:.1f}%)", ha="center", va="bottom",
             fontsize=11, fontweight="bold")

ax.set_xlabel("Iteration of FIRST Hallucination (Targeted or Untargeted)",
              fontsize=12)
ax.set_ylabel("Number of Entries", fontsize=12)
ax.set_xticks(iterations)
ax.set_title(
    f"How Many Iterations Until Hallucination First Occurred?\n"
    f"Gemma-4-abliterated Victim + Judge, Qwen-Image-Edit-2509 "
    f"Generator — All 14 Batches\n"
    f"Includes BOTH Targeted and Untargeted hallucination  |  "
    f"Total: {total_hallucinated} entries  |  "
    f"Average: {average_iter:.2f} iterations  |  "
    f"Median: {median_iter} iterations",
    fontsize=12, fontweight="bold", pad=20
)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved chart to: {CHART_PATH}")
