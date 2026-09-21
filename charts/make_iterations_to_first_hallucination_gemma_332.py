# ============================================================
# make_iterations_to_first_hallucination_gemma_332.py
# ============================================================
# For the 332 entries whose FINAL status was a hallucination
# (TARGETED or UNTARGETED) across all 14 Gemma-victim batches,
# scans every step9_hallucination_iterN.txt file for that entry to
# find the FIRST iteration at which it hallucinated (which is not
# always the same as the final iterations_needed value, since an
# entry can hallucinate early and later recover, or hallucinate
# only on the final attempt).
#
# Batch 28 uses RECLASSIFIED_SUMMARY.json (the corrected native
# 3-way labels) to determine set membership, since its original
# SUMMARY.json used an older binary scheme.
#
# Produces a bar chart of the distribution across iterations 1-5,
# plus the average and median iteration of first hallucination.
#
# Saves the chart to:
#   iterations_to_first_hallucination_gemma_332.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import glob
import re
import json
import statistics
import matplotlib.pyplot as plt

CHART_PATH = "iterations_to_first_hallucination_gemma_332.png"

BATCHES = [
    {"name": "outputs_strict_batch_28",
     "classify_path": "outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json",
     "classify_field": "final_category"},
    {"name": "outputs_strict_batch_50"},
    {"name": "outputs_strict_batch_151_200"},
    {"name": "outputs_strict_batch_201_280"},
    {"name": "outputs_strict_batch_281_360"},
    {"name": "outputs_strict_batch_361_440_combined"},
    {"name": "outputs_strict_batch_441_520"},
    {"name": "outputs_strict_batch_521_600"},
    {"name": "outputs_strict_batch_601_650"},
    {"name": "outputs_strict_batch_651_700"},
    {"name": "outputs_strict_batch_701_750"},
    {"name": "outputs_strict_batch_751_800"},
    {"name": "outputs_strict_batch_801_850"},
    {"name": "outputs_strict_batch_851_900"},
]


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


# ── Determine which entries are in the final 332 set ─────────────
final_hallucinated_entries = []  # list of (batch_name, entry_num)

for batch in BATCHES:
    if "classify_path" in batch:
        with open(batch["classify_path"], "r") as f:
            data = json.load(f)
        for r in data:
            if r[batch["classify_field"]] in ("TARGETED", "UNTARGETED"):
                final_hallucinated_entries.append((batch["name"], r["entry_num"]))
    else:
        with open(f"{batch['name']}/SUMMARY.json", "r") as f:
            data = json.load(f)
        for r in data:
            if r["status"] in ("TARGETED", "UNTARGETED"):
                final_hallucinated_entries.append((batch["name"], r["entry_num"]))

print(f"Total entries in final-hallucinated set: {len(final_hallucinated_entries)}")

# ── For each, scan per-iteration files to find first hallucination ──
first_iterations = []
missing = []

for batch_name, entry_num in final_hallucinated_entries:
    entry_dir = f"{batch_name}/entry_{entry_num}"
    iter_files = sorted(
        glob.glob(f"{entry_dir}/step9_hallucination_iter*.txt"),
        key=lambda p: int(re.search(r"iter(\d+)", p).group(1))
    )
    if not iter_files:
        missing.append((batch_name, entry_num))
        continue
    categories = [classify_file(f) for f in iter_files]
    first_iter = first_hallucination_iteration(categories)
    if first_iter is not None:
        first_iterations.append(first_iter)
    else:
        missing.append((batch_name, entry_num))

print(f"Successfully determined first-hallucination iteration for: "
      f"{len(first_iterations)}")
if missing:
    print(f"Could not determine for {len(missing)} entries "
          f"(missing/unreadable iteration files):")
    for m in missing[:10]:
        print(f"  {m[0]} entry #{m[1]}")
    if len(missing) > 10:
        print(f"  ... and {len(missing)-10} more")

# ── Report distribution ───────────────────────────────────────────
avg_iter = statistics.mean(first_iterations)
median_iter = statistics.median(first_iterations)

print()
print(f"Average iteration of first hallucination: {avg_iter:.2f}")
print(f"Median iteration of first hallucination : {median_iter}")
for i in range(1, 6):
    c = first_iterations.count(i)
    pct = 100 * c / len(first_iterations)
    print(f"  Iteration {i}: {c} entries ({pct:.1f}%)")

# ── Build bar chart ────────────────────────────────────────────────
counts = [first_iterations.count(i) for i in range(1, 6)]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.bar(range(1, 6), counts, color="#e74c3c", edgecolor="white",
              linewidth=1.5, width=0.6)

for bar, count in zip(bars, counts):
    pct = 100 * count / len(first_iterations)
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{count}\n({pct:.1f}%)", ha="center", va="bottom",
             fontsize=11, fontweight="bold")

ax.set_xlabel("Iteration of First Hallucination", fontsize=12)
ax.set_ylabel("Number of Entries", fontsize=12)
ax.set_xticks(range(1, 6))
ax.set_title(
    f"Iterations to First Hallucination — Gemma-4-abliterated\n"
    f"Dataset: HaluEval (derived from HotpotQA)  |  "
    f"{len(first_iterations)} Final-Hallucinated Entries (of 500 total)\n"
    f"Average: {avg_iter:.2f}  |  Median: {median_iter}",
    fontsize=12, fontweight="bold", pad=15
)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved chart to: {CHART_PATH}")
