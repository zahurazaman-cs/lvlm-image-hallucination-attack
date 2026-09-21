# ============================================================
# make_gemma_escalation_vs_qwen_transfer_chart.py
# ============================================================
# Answers: "For images that took heavy escalation to fool Gemma,
# did that same image ALSO fool Qwen2.5-VL when reused, or did it
# fail to transfer?"
#
# Restricts to entries where Gemma's FINAL status was TARGETED
# (the only outcome where "iterations_needed" genuinely means
# "took N attempts to succeed" — UNTARGETED/NONE entries always
# show iterations_needed=5 regardless of what actually happened,
# so they aren't meaningful for this specific question).
#
# For each Gemma escalation level (1-5), shows:
#   - How many of those images ALSO hallucinated Qwen2.5-VL
#     (TARGETED or UNTARGETED) — i.e. the attack "transferred"
#   - How many did NOT (Qwen2.5-VL answered correctly, NONE) —
#     i.e. the attack "did not transfer"
#
# This directly visualizes whether heavily-escalated (aggressive,
# late-iteration) attack images transfer worse/better across victim
# models than mild, first-attempt images.
#
# Saves the chart to:
#   gemma_escalation_vs_qwen_transfer_chart.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import json
import matplotlib.pyplot as plt

# (qwen_reuse_dir, gemma_source_dir) pairs — all twelve reuse-method
# batches, mapped back to their original Gemma source batch.
BATCH_PAIRS = [
    ("outputs_victim_qwen_151_200",       "outputs_strict_batch_151_200"),
    ("outputs_victim_qwen_201_280",       "outputs_strict_batch_201_280"),
    ("outputs_victim_qwen_281_360",       "outputs_strict_batch_281_360"),
    ("outputs_victim_qwen_361_440",       "outputs_strict_batch_361_440_combined"),
    ("outputs_victim_qwen_441_520",       "outputs_strict_batch_441_520"),
    ("outputs_victim_qwen_521_600",       "outputs_strict_batch_521_600"),
    ("outputs_victim_qwen_601_650_reuse", "outputs_strict_batch_601_650"),
    ("outputs_victim_qwen_651_700",       "outputs_strict_batch_651_700"),
    ("outputs_victim_qwen_701_750",       "outputs_strict_batch_701_750"),
    ("outputs_victim_qwen_751_800",       "outputs_strict_batch_751_800"),
    ("outputs_victim_qwen_801_850",       "outputs_strict_batch_801_850"),
    ("outputs_victim_qwen_851_900",       "outputs_strict_batch_851_900"),
]

CHART_PATH = "gemma_escalation_vs_qwen_transfer_chart.png"

# iteration_level -> {"transferred": N, "not_transferred": N}
buckets = {i: {"transferred": 0, "not_transferred": 0} for i in range(1, 6)}

total_gemma_targeted = 0

for qwen_dir, gemma_dir in BATCH_PAIRS:
    with open(f"{gemma_dir}/SUMMARY.json", "r") as f:
        gemma_summary = {r["entry_num"]: r for r in json.load(f)}

    with open(f"{qwen_dir}/entry_state.json", "r") as f:
        qwen_state = json.load(f)

    for entry_num, g in gemma_summary.items():
        if g["status"] != "TARGETED":
            continue  # only meaningful for Gemma-TARGETED entries

        total_gemma_targeted += 1
        iter_level = g["iterations_needed"]

        qwen_entry = qwen_state.get(str(entry_num))
        if qwen_entry is None:
            continue  # skipped/missing in the Qwen reuse run

        qwen_status = qwen_entry.get("status")
        if qwen_status in ("TARGETED", "UNTARGETED"):
            buckets[iter_level]["transferred"] += 1
        elif qwen_status == "NONE":
            buckets[iter_level]["not_transferred"] += 1
        # SKIPPED_NO_PHOTO / ERROR entries are excluded silently

print(f"Total Gemma-TARGETED entries considered: {total_gemma_targeted}")
print()
for i in range(1, 6):
    t = buckets[i]["transferred"]
    nt = buckets[i]["not_transferred"]
    total_i = t + nt
    if total_i == 0:
        print(f"Iteration {i}: no entries")
        continue
    print(f"Iteration {i}: {total_i} entries total")
    print(f"  Transferred to Qwen2.5-VL     : {t} ({100*t/total_i:.1f}%)")
    print(f"  Did NOT transfer (Qwen correct): {nt} ({100*nt/total_i:.1f}%)")

# ── Build grouped bar chart ──────────────────────────────────────
iterations = list(range(1, 6))
transferred_counts     = [buckets[i]["transferred"] for i in iterations]
not_transferred_counts = [buckets[i]["not_transferred"] for i in iterations]

fig, ax = plt.subplots(figsize=(11, 7))
x = range(len(iterations))
width = 0.35

bars1 = ax.bar([p - width/2 for p in x], transferred_counts, width,
               label="Transferred (also fooled Qwen2.5-VL)",
               color="#e74c3c", edgecolor="white", linewidth=1.5)
bars2 = ax.bar([p + width/2 for p in x], not_transferred_counts, width,
               label="Did NOT transfer (Qwen2.5-VL answered correctly)",
               color="#2ecc71", edgecolor="white", linewidth=1.5)

for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                     f"{int(height)}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

ax.set_xlabel("Gemma Escalation Iteration Needed to Succeed", fontsize=12)
ax.set_ylabel("Number of Entries", fontsize=12)
ax.set_xticks(list(x))
ax.set_xticklabels([str(i) for i in iterations])
ax.set_title(
    f"Did Gemma's Attack Images Transfer to Fool Qwen2.5-VL Too?\n"
    f"Restricted to {total_gemma_targeted} entries where Gemma was "
    f"successfully TARGETED\n"
    f"(same reused image tested against Qwen2.5-VL as victim, "
    f"Gemma as judge in both cases)",
    fontsize=12, fontweight="bold", pad=20
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=200, bbox_inches="tight")
print(f"\nSaved chart to: {CHART_PATH}")
