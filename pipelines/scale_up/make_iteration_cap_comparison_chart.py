# ============================================================
# make_iteration_cap_comparison_chart.py
# ============================================================
# Grouped bar chart comparing the REAL hallucination rate (full
# 5-iteration escalation) against the SIMULATED rate under a
# hard cap of 2 iterations, across all four conditions where this
# comparison is meaningful:
#
#   1. qa_data.json (HaluEval) — Gemma-4-abliterated, 500 entries
#   2. NQ-Swap — Gemma-4-abliterated, 100 entries
#   3. TruthfulQA — Gemma-4-abliterated, 22 entries
#   4. qa_data.json (HaluEval) — Qwen2.5-VL, two-phase entries
#      only, 51 entries (the only Qwen2.5-VL entries anywhere in
#      this project with genuine fresh escalation)
#
# THIS MAKES NO NEW MODEL CALLS — recomputes both the real and
# simulated results directly from existing per-iteration judge
# files, using the same logic validated in the two earlier
# simulation scripts.
#
# Saves the chart to:
#   iteration_cap_comparison_chart.png
#
# Requirements: matplotlib
#   pip install matplotlib --break-system-packages
# ============================================================

import glob
import json
import matplotlib.pyplot as plt


def classify_file(path):
    with open(path, "r") as f:
        text = f.read()
    if "HALLUCINATING_TARGETED: YES" in text:
        return "TARGETED"
    elif "HALLUCINATING_UNTARGETED: YES" in text:
        return "UNTARGETED"
    else:
        return "NONE"


def simulate_cap(entry_dir, prefix="step9_hallucination_iter"):
    iter1_path = f"{entry_dir}/{prefix}1.txt"
    if not glob.glob(iter1_path):
        return None
    iter1_status = classify_file(iter1_path)
    if iter1_status == "TARGETED":
        return "TARGETED"
    iter2_path = f"{entry_dir}/{prefix}2.txt"
    if not glob.glob(iter2_path):
        return iter1_status
    return classify_file(iter2_path)


def compute_real_vs_sim(entries_with_status, prefix="step9_hallucination_iter"):
    """entries_with_status: list of (entry_dir, real_status)"""
    real_hall = sim_hall = total = 0
    for entry_dir, real_status in entries_with_status:
        total += 1
        if real_status in ("TARGETED", "UNTARGETED"):
            real_hall += 1
        sim_status = simulate_cap(entry_dir, prefix)
        if sim_status in ("TARGETED", "UNTARGETED"):
            sim_hall += 1
    return real_hall, sim_hall, total


results = {}

# ── Condition 1: qa_data.json — Gemma-4-abliterated (500 entries) ──
QA_DATA_BATCHES = [
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

entries = []
for batch in QA_DATA_BATCHES:
    if "classify_path" in batch:
        with open(batch["classify_path"], "r") as f:
            data = json.load(f)
        for r in data:
            entries.append((f"{batch['name']}/entry_{r['entry_num']}",
                             r[batch["classify_field"]],
                             "step9_reclassified_iter"))
    else:
        with open(f"{batch['name']}/SUMMARY.json", "r") as f:
            data = json.load(f)
        for r in data:
            if r["status"] == "SKIPPED_NO_PHOTO":
                continue
            entries.append((f"{batch['name']}/entry_{r['entry_num']}",
                             r["status"], "step9_hallucination_iter"))

real_hall = sim_hall = total = 0
for entry_dir, real_status, prefix in entries:
    total += 1
    if real_status in ("TARGETED", "UNTARGETED"):
        real_hall += 1
    sim_status = simulate_cap(entry_dir, prefix)
    if sim_status in ("TARGETED", "UNTARGETED"):
        sim_hall += 1

results["qa_data.json\n(Gemma, n=500)"] = (real_hall, sim_hall, total)

# ── Condition 2: NQ-Swap — Gemma-4-abliterated (100 entries) ──────
entries = []
with open("outputs_strict_batch_nqswap_26_50/RECLASSIFIED_SUMMARY.json", "r") as f:
    reclassified = json.load(f)
for r in reclassified:
    entries.append((f"{r['batch_dir']}/entry_{r['entry_num']}",
                     r["final_category"], "step9_reclassified_iter"))

for batch_name in ["outputs_strict_batch_nqswap_51_75",
                    "outputs_strict_batch_nqswap_76_100"]:
    with open(f"{batch_name}/SUMMARY.json", "r") as f:
        summary = json.load(f)
    for r in summary:
        if r["status"] == "SKIPPED_NO_PHOTO":
            continue
        entries.append((f"{batch_name}/entry_{r['entry_num']}",
                         r["status"], "step9_hallucination_iter"))

real_hall = sim_hall = total = 0
for entry_dir, real_status, prefix in entries:
    total += 1
    if real_status in ("TARGETED", "UNTARGETED"):
        real_hall += 1
    sim_status = simulate_cap(entry_dir, prefix)
    if sim_status in ("TARGETED", "UNTARGETED"):
        sim_hall += 1

results["NQ-Swap\n(Gemma, n=100)"] = (real_hall, sim_hall, total)

# ── Condition 3: TruthfulQA — Gemma-4-abliterated (22 entries) ────
with open("outputs_strict_batch_truthfulqa_22/SUMMARY.json", "r") as f:
    summary = json.load(f)

real_hall = sim_hall = total = 0
for r in summary:
    if r["status"] == "SKIPPED_NO_PHOTO":
        continue
    total += 1
    if r["status"] in ("TARGETED", "UNTARGETED"):
        real_hall += 1
    entry_dir = f"outputs_strict_batch_truthfulqa_22/entry_{r['entry_num']}"
    sim_status = simulate_cap(entry_dir, "step9_hallucination_iter")
    if sim_status in ("TARGETED", "UNTARGETED"):
        sim_hall += 1

results["TruthfulQA\n(Gemma, n=22)"] = (real_hall, sim_hall, total)

# ── Condition 4: qa_data.json — Qwen2.5-VL two-phase (51 entries) ──
TWO_PHASE_BATCH_DIRS = ["outputs_victim_qwen_28", "outputs_victim_qwen_50"]

real_hall = sim_hall = total = 0
for batch_dir in TWO_PHASE_BATCH_DIRS:
    with open(f"{batch_dir}/entry_state.json", "r") as f:
        state = json.load(f)
    for key, s in state.items():
        if s.get("status") not in ("TARGETED", "UNTARGETED", "NONE"):
            continue
        total += 1
        if s["status"] in ("TARGETED", "UNTARGETED"):
            real_hall += 1
        entry_num = s.get("entry_num", key)
        entry_dir = f"{batch_dir}/entry_{entry_num}"
        sim_status = simulate_cap(entry_dir, "step9_hallucination_iter")
        if sim_status in ("TARGETED", "UNTARGETED"):
            sim_hall += 1

results["qa_data.json\n(Qwen2.5-VL,\ntwo-phase, n=51)"] = (real_hall, sim_hall, total)

# ── Print summary ──────────────────────────────────────────────────
print(f"{'Condition':<35s} {'Real %':>10s} {'Sim %':>10s} {'Change':>10s}")
for label, (real_hall, sim_hall, total) in results.items():
    real_pct = 100 * real_hall / total
    sim_pct = 100 * sim_hall / total
    label_clean = label.replace("\n", " ")
    print(f"{label_clean:<35s} {real_pct:>9.1f}% {sim_pct:>9.1f}% "
          f"{sim_pct-real_pct:>+9.1f}pp")

# ── Build grouped bar chart ───────────────────────────────────────
labels = list(results.keys())
real_pcts = [100 * v[0] / v[2] for v in results.values()]
sim_pcts = [100 * v[1] / v[2] for v in results.values()]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 8))

bars1 = ax.bar([p - width/2 for p in x], real_pcts, width,
               label="Real (cap = 5 iterations)",
               color="#e74c3c", edgecolor="white", linewidth=1.5)
bars2 = ax.bar([p + width/2 for p in x], sim_pcts, width,
               label="Simulated (cap = 2 iterations)",
               color="#3498db", edgecolor="white", linewidth=1.5)

for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.8,
                 f"{height:.1f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

ax.set_ylabel("Total Hallucination Rate (%)", fontsize=12)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10)
ax.set_title(
    "Effect of Capping Escalation at 2 Iterations vs the Real 5-Iteration Limit\n"
    "Judge: Gemma-4-abliterated (unchanged throughout)\n"
    "No new model calls — simulated directly from existing per-iteration judge files",
    fontsize=12, fontweight="bold", pad=15
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.set_ylim(0, max(real_pcts + sim_pcts) * 1.2)

plt.tight_layout()
plt.savefig("iteration_cap_comparison_chart.png", dpi=200, bbox_inches="tight")
print(f"\nSaved chart to: iteration_cap_comparison_chart.png")
