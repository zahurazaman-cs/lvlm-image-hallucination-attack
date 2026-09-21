# ============================================================
# make_iterations_to_first_hallucination_qwen25.py
# ============================================================
# Rebuilds the iteration analysis for Qwen2.5-VL's original attack
# results on qa_data.json (HaluEval), correctly split into the two
# methodologically distinct groups established earlier:
#
#   GROUP 1 — the two two-phase batches (outputs_victim_qwen_28,
#   outputs_victim_qwen_50, 51 entries), where a FRESH image was
#   genuinely escalated specifically to fool Qwen2.5-VL, up to 5
#   attempts. For these, "iteration of first hallucination" means
#   exactly what it means for Gemma — real escalation effort.
#
#   GROUP 2 — the twelve reuse-method batches (450 entries), where
#   Qwen2.5-VL was tested ONCE against whichever image Gemma's own
#   attack had already settled on. For these, the recorded
#   "source_iteration" does NOT mean "attempts against Qwen2.5-VL"
#   — it means "which escalation level of GEMMA's attack image
#   also happened to transfer and fool Qwen2.5-VL."
#
# Reporting these separately avoids the misleading conflation of
# blending a genuine-effort metric with a transfer metric.
#
# Saves the chart to:
#   iterations_to_first_hallucination_qwen25.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import glob
import re
import json
import statistics
import matplotlib.pyplot as plt

CHART_PATH = "iterations_to_first_hallucination_qwen25.png"

TWO_PHASE_BATCH_DIRS = [
    "outputs_victim_qwen_28",
    "outputs_victim_qwen_50",
]

REUSE_BATCH_DIRS = [
    "outputs_victim_qwen_151_200",
    "outputs_victim_qwen_201_280",
    "outputs_victim_qwen_281_360",
    "outputs_victim_qwen_361_440",
    "outputs_victim_qwen_441_520",
    "outputs_victim_qwen_521_600",
    "outputs_victim_qwen_601_650_reuse",
    "outputs_victim_qwen_651_700",
    "outputs_victim_qwen_701_750",
    "outputs_victim_qwen_751_800",
    "outputs_victim_qwen_801_850",
    "outputs_victim_qwen_851_900",
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


# ── Group 1: TRUE iteration-to-hallucination (two-phase batches) ──
true_iterations = []
for batch_dir in TWO_PHASE_BATCH_DIRS:
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
            true_iterations.append(first_iter)

# ── Group 2: TRANSFERRED escalation level (reuse-method batches) ──
transferred_iterations = []
for batch_dir in REUSE_BATCH_DIRS:
    with open(f"{batch_dir}/entry_state.json", "r") as f:
        state = json.load(f)
    for s in state.values():
        if s.get("status") in ("TARGETED", "UNTARGETED"):
            transferred_iterations.append(s.get("source_iteration"))
transferred_iterations = [i for i in transferred_iterations if i is not None]

# ── Report each group separately ─────────────────────────────────
print("=" * 65)
print("GROUP 1: TRUE iteration-to-hallucination (two-phase batches, "
      "fresh escalation)")
print("=" * 65)
if true_iterations:
    print(f"Total: {len(true_iterations)}")
    print(f"Average: {statistics.mean(true_iterations):.2f}")
    print(f"Median : {statistics.median(true_iterations)}")
    for i in range(1, 6):
        c = true_iterations.count(i)
        print(f"  Iteration {i}: {c} entries "
              f"({100*c/len(true_iterations):.1f}%)")
else:
    print("No hallucinated entries found in this group.")

print()
print("=" * 65)
print("GROUP 2: TRANSFERRED escalation level (reuse-method batches, "
      "no fresh escalation for Qwen2.5-VL)")
print("=" * 65)
if transferred_iterations:
    print(f"Total: {len(transferred_iterations)}")
    print(f"Average: {statistics.mean(transferred_iterations):.2f}")
    print(f"Median : {statistics.median(transferred_iterations)}")
    for i in range(1, 6):
        c = transferred_iterations.count(i)
        print(f"  Iteration {i}: {c} entries "
              f"({100*c/len(transferred_iterations):.1f}%)")
else:
    print("No hallucinated entries found in this group.")

# ── Combined (for a single overview number, clearly labeled) ─────
combined = true_iterations + transferred_iterations
print()
print("=" * 65)
print("COMBINED (both groups together — see caveat above before "
      "citing this as a single number)")
print("=" * 65)
print(f"Total: {len(combined)}")
if combined:
    print(f"Average: {statistics.mean(combined):.2f}")
    print(f"Median : {statistics.median(combined)}")

# ── Build a two-panel bar chart, one per group ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, data, title, color in [
    (axes[0], true_iterations,
     f"Group 1: TRUE Iteration-to-Hallucination\n"
     f"(Two-Phase Batches, {len(true_iterations)} entries)", "#3498db"),
    (axes[1], transferred_iterations,
     f"Group 2: Transferred Escalation Level\n"
     f"(Reuse-Method Batches, {len(transferred_iterations)} entries)",
     "#9b59b6"),
]:
    counts = [data.count(i) for i in range(1, 6)]
    bars = ax.bar(range(1, 6), counts, color=color, edgecolor="white",
                  linewidth=1.5, width=0.6)
    for bar, count in zip(bars, counts):
        pct = 100 * count / len(data) if data else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{count}\n({pct:.1f}%)", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Number of Entries", fontsize=11)
    ax.set_xticks(range(1, 6))
    avg = statistics.mean(data) if data else 0
    med = statistics.median(data) if data else 0
    ax.set_title(f"{title}\nAvg: {avg:.2f}  |  Median: {med}",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

fig.suptitle(
    "Qwen2.5-VL Original Attack — Iteration Analysis for Hallucination\n"
    "Dataset: HaluEval (derived from HotpotQA)  |  Targeted + "
    "Untargeted combined; two methodologically distinct groups shown "
    "separately",
    fontsize=13, fontweight="bold", y=1.03
)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved chart to: {CHART_PATH}")
