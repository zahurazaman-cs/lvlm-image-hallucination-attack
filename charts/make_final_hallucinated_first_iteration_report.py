# ============================================================
# make_final_hallucinated_first_iteration_report.py
# ============================================================
# Restricts analysis to ONLY the 332 entries whose FINAL iteration
# was a hallucination (TARGETED or UNTARGETED) — the same set shown
# in the combined pie chart. For these entries specifically, checks
# whether they hallucinated FROM THE VERY FIRST ITERATION onward
# (consistently wrong throughout), or whether they were actually
# CORRECT (NONE) on an earlier iteration and only started
# hallucinating later — meaning the escalating misleading
# description eventually broke an answer that was initially right.
#
# This matters because only TARGETED stops the loop early; NONE and
# UNTARGETED both continue escalating. So it's possible for an
# entry to answer correctly at iteration 1, then get pushed into a
# hallucination by a more aggressive later iteration, and still end
# up counted among the "final hallucinated" 332 — without ever
# having been correctly resistant throughout.
#
# Produces:
#   - A bar chart showing, among the 332 final-hallucinated entries,
#     the iteration at which each one FIRST hallucinated
#   - A breakdown: how many were wrong from iteration 1 ("always
#     hallucinated") vs. correct earlier but later broke down
#     ("developed hallucination")
#
# Saves the chart to:
#   final_hallucinated_first_iteration_report.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import glob
import re
import statistics
import matplotlib.pyplot as plt

CHART_PATH = "final_hallucinated_first_iteration_report.png"

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
    for i, c in enumerate(categories, start=1):
        if c in ("TARGETED", "UNTARGETED"):
            return i
    return None


def get_entry_categories(batch_dir, pattern):
    entry_dirs = sorted(
        glob.glob(f"{batch_dir}/entry_[0-9]*"),
        key=lambda p: int(re.search(r"entry_(\d+)", p).group(1))
    )
    results = {}
    for entry_dir in entry_dirs:
        entry_num = int(re.search(r"entry_(\d+)", entry_dir).group(1))
        iter_files = sorted(
            glob.glob(f"{entry_dir}/{pattern}"),
            key=lambda p: int(re.search(r"iter(\d+)", p).group(1))
        )
        if not iter_files:
            continue
        results[(batch_dir, entry_num)] = [classify_file(f) for f in iter_files]
    return results


all_categories = get_entry_categories(BATCH_DIR_28, "step9_reclassified_iter*.txt")
for batch_dir in BATCH_DIRS_NATIVE:
    all_categories.update(
        get_entry_categories(batch_dir, "step9_hallucination_iter*.txt")
    )

# ── Restrict to entries whose FINAL iteration hallucinated ───────
final_hallucinated = {
    key: cats for key, cats in all_categories.items()
    if cats[-1] in ("TARGETED", "UNTARGETED")
}

total_final_hallucinated = len(final_hallucinated)

first_hall_iterations = []
always_hallucinated = 0        # first hallucination at iteration 1
developed_hallucination = 0    # was NONE earlier, hallucinated later
developed_entries = []

for key, cats in final_hallucinated.items():
    first_iter = first_hallucination_iteration(cats)
    first_hall_iterations.append(first_iter)
    if first_iter == 1:
        always_hallucinated += 1
    else:
        developed_hallucination += 1
        developed_entries.append((key, cats))

average_iter = statistics.mean(first_hall_iterations)
median_iter = statistics.median(first_hall_iterations)

print(f"Total entries with FINAL status = hallucination: "
      f"{total_final_hallucinated}")
print(f"Average iteration of FIRST hallucination (within this set): "
      f"{average_iter:.2f}")
print(f"Median iteration of FIRST hallucination (within this set) : "
      f"{median_iter}")
print()
print(f"Always hallucinated (wrong from iteration 1)     : "
      f"{always_hallucinated} "
      f"({100*always_hallucinated/total_final_hallucinated:.1f}%)")
print(f"Developed hallucination (correct earlier, broke  : "
      f"{developed_hallucination} "
      f"({100*developed_hallucination/total_final_hallucinated:.1f}%)")
print(f"  down in a later iteration)")
print()

counts_per_iter = {i: 0 for i in range(1, 6)}
for it in first_hall_iterations:
    counts_per_iter[it] = counts_per_iter.get(it, 0) + 1
for i in range(1, 6):
    pct = 100 * counts_per_iter[i] / total_final_hallucinated
    print(f"  First hallucinated at iteration {i}: "
          f"{counts_per_iter[i]} entries ({pct:.1f}%)")

if developed_entries:
    print(f"\nEntries that were CORRECT earlier but hallucinated in a "
          f"LATER iteration (first {min(10, len(developed_entries))} shown):")
    for (batch_dir, num), cats in developed_entries[:10]:
        print(f"  {batch_dir} / Entry #{num}: all_iterations = {cats}")

# ── Build the bar chart ──────────────────────────────────────────
iterations = list(range(1, 6))
counts = [counts_per_iter[i] for i in iterations]

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#e74c3c" if i == 1 else "#f39c12" for i in iterations]
bars = ax.bar(iterations, counts, color=colors, edgecolor="white",
              linewidth=1.5, width=0.6)

for bar, count in zip(bars, counts):
    pct = 100 * count / total_final_hallucinated if total_final_hallucinated else 0
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{count}\n({pct:.1f}%)", ha="center", va="bottom",
             fontsize=11, fontweight="bold")

ax.set_xlabel("Iteration of FIRST Hallucination", fontsize=12)
ax.set_ylabel("Number of Entries", fontsize=12)
ax.set_xticks(iterations)
ax.set_title(
    f"Among Entries That Ended Hallucinating, When Did It Start?\n"
    f"Gemma-4-abliterated Victim + Judge, Qwen-Image-Edit-2509 "
    f"Generator — All 14 Batches\n"
    f"Total (final-iteration hallucinated): {total_final_hallucinated} "
    f"entries  |  Always hallucinated: {always_hallucinated} "
    f"({100*always_hallucinated/total_final_hallucinated:.1f}%)  |  "
    f"Developed later: {developed_hallucination} "
    f"({100*developed_hallucination/total_final_hallucinated:.1f}%)",
    fontsize=11, fontweight="bold", pad=20
)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved chart to: {CHART_PATH}")
