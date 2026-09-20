# ============================================================
# simulate_max_iter_2_gemma_all_datasets.py
# ============================================================
# Simulates what the FINAL hallucination classification would
# have been for every Gemma-victim entry across all three
# datasets if the escalation loop had been hard-capped at 2
# iterations instead of 5.
#
# THIS MAKES NO NEW MODEL CALLS. Every entry that did not stop
# early at iteration 1 already has an iteration 2 judge file on
# disk, since the real escalation loop only stops early on a
# TARGETED result, otherwise it always tries again. So the correct
# simulated result under a cap of 2 is simply:
#
#   - If iteration 1 was TARGETED -> stays TARGETED (identical to
#     the real run, since the real run also stopped here)
#   - Otherwise -> whatever iteration 2's classification was
#     (TARGETED, UNTARGETED, or NONE), since a cap of 2 means the
#     loop would have stopped there regardless of what the real
#     run went on to do afterward
#
# Covers all 500 qa_data.json entries, all 100 NQ-Swap entries,
# and all 22 TruthfulQA entries, using the same reclassification-
# aware logic already established for each dataset (batch 28 and
# the first 50 NQ-Swap entries use their corrected per-iteration
# files).
# ============================================================

import glob
import re
import json


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
    """Returns the simulated final classification under a cap of 2,
    or None if iteration 1 file is missing."""
    iter1_path = f"{entry_dir}/{prefix}1.txt"
    if not glob.glob(iter1_path):
        return None
    iter1_status = classify_file(iter1_path)
    if iter1_status == "TARGETED":
        return "TARGETED"
    iter2_path = f"{entry_dir}/{prefix}2.txt"
    if not glob.glob(iter2_path):
        # Entry only ever had 1 iteration and it wasn't TARGETED —
        # under a cap of 2 the result is unchanged (loop would have
        # tried a second time, but if none was ever run in reality,
        # treat iteration 1's result as final)
        return iter1_status
    return classify_file(iter2_path)


def report(name, real_targeted, real_untargeted, real_none,
           sim_targeted, sim_untargeted, sim_none, total):
    real_hall = real_targeted + real_untargeted
    sim_hall = sim_targeted + sim_untargeted
    print(f"\n{'='*65}")
    print(f"{name}")
    print(f"{'='*65}")
    print(f"Total entries: {total}")
    print()
    print(f"REAL (cap=5)  : TARGETED {real_targeted} "
          f"({100*real_targeted/total:.1f}%), "
          f"UNTARGETED {real_untargeted} ({100*real_untargeted/total:.1f}%), "
          f"NONE {real_none} ({100*real_none/total:.1f}%)")
    print(f"              TOTAL HALLUCINATION: {real_hall} "
          f"({100*real_hall/total:.1f}%)")
    print()
    print(f"SIMULATED (cap=2): TARGETED {sim_targeted} "
          f"({100*sim_targeted/total:.1f}%), "
          f"UNTARGETED {sim_untargeted} ({100*sim_untargeted/total:.1f}%), "
          f"NONE {sim_none} ({100*sim_none/total:.1f}%)")
    print(f"              TOTAL HALLUCINATION: {sim_hall} "
          f"({100*sim_hall/total:.1f}%)")
    print()
    diff = sim_hall - real_hall
    direction = "INCREASE" if diff > 0 else ("DECREASE" if diff < 0 else "NO CHANGE")
    print(f"Effect of capping at 2 iterations: {direction} of "
          f"{abs(diff)} entries ({abs(100*diff/total):.1f} percentage points)")


# ============================================================
# DATASET 1: qa_data.json (HaluEval) — 500 entries
# ============================================================

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

real_t = real_u = real_n = 0
sim_t = sim_u = sim_n = 0
total = 0

for batch in QA_DATA_BATCHES:
    if "classify_path" in batch:
        with open(batch["classify_path"], "r") as f:
            data = json.load(f)
        entries = [(r["entry_num"], r[batch["classify_field"]]) for r in data]
        prefix = "step9_reclassified_iter"
    else:
        with open(f"{batch['name']}/SUMMARY.json", "r") as f:
            data = json.load(f)
        entries = [(r["entry_num"], r["status"]) for r in data
                   if r["status"] != "SKIPPED_NO_PHOTO"]
        prefix = "step9_hallucination_iter"

    for entry_num, real_status in entries:
        total += 1
        if real_status == "TARGETED":
            real_t += 1
        elif real_status == "UNTARGETED":
            real_u += 1
        else:
            real_n += 1

        entry_dir = f"{batch['name']}/entry_{entry_num}"
        sim_status = simulate_cap(entry_dir, prefix)
        if sim_status == "TARGETED":
            sim_t += 1
        elif sim_status == "UNTARGETED":
            sim_u += 1
        else:
            sim_n += 1

report("qa_data.json (HaluEval) — Gemma-4-abliterated",
       real_t, real_u, real_n, sim_t, sim_u, sim_n, total)


# ============================================================
# DATASET 2: NQ-Swap — 100 entries
# ============================================================

real_t = real_u = real_n = 0
sim_t = sim_u = sim_n = 0
total = 0

# entries 0-49 (reclassified)
with open("outputs_strict_batch_nqswap_26_50/RECLASSIFIED_SUMMARY.json", "r") as f:
    reclassified = json.load(f)

for r in reclassified:
    total += 1
    real_status = r["final_category"]
    if real_status == "TARGETED":
        real_t += 1
    elif real_status == "UNTARGETED":
        real_u += 1
    else:
        real_n += 1

    entry_dir = f"{r['batch_dir']}/entry_{r['entry_num']}"
    sim_status = simulate_cap(entry_dir, "step9_reclassified_iter")
    if sim_status == "TARGETED":
        sim_t += 1
    elif sim_status == "UNTARGETED":
        sim_u += 1
    else:
        sim_n += 1

# entries 50-99 (native)
for batch_name in ["outputs_strict_batch_nqswap_51_75",
                    "outputs_strict_batch_nqswap_76_100"]:
    with open(f"{batch_name}/SUMMARY.json", "r") as f:
        summary = json.load(f)
    for r in summary:
        if r["status"] == "SKIPPED_NO_PHOTO":
            continue
        total += 1
        if r["status"] == "TARGETED":
            real_t += 1
        elif r["status"] == "UNTARGETED":
            real_u += 1
        else:
            real_n += 1

        entry_dir = f"{batch_name}/entry_{r['entry_num']}"
        sim_status = simulate_cap(entry_dir, "step9_hallucination_iter")
        if sim_status == "TARGETED":
            sim_t += 1
        elif sim_status == "UNTARGETED":
            sim_u += 1
        else:
            sim_n += 1

report("NQ-Swap — Gemma-4-abliterated",
       real_t, real_u, real_n, sim_t, sim_u, sim_n, total)


# ============================================================
# DATASET 3: TruthfulQA — 22 entries
# ============================================================

real_t = real_u = real_n = 0
sim_t = sim_u = sim_n = 0
total = 0

with open("outputs_strict_batch_truthfulqa_22/SUMMARY.json", "r") as f:
    summary = json.load(f)

for r in summary:
    if r["status"] == "SKIPPED_NO_PHOTO":
        continue
    total += 1
    if r["status"] == "TARGETED":
        real_t += 1
    elif r["status"] == "UNTARGETED":
        real_u += 1
    else:
        real_n += 1

    entry_dir = f"outputs_strict_batch_truthfulqa_22/entry_{r['entry_num']}"
    sim_status = simulate_cap(entry_dir, "step9_hallucination_iter")
    if sim_status == "TARGETED":
        sim_t += 1
    elif sim_status == "UNTARGETED":
        sim_u += 1
    else:
        sim_n += 1

report("TruthfulQA — Gemma-4-abliterated",
       real_t, real_u, real_n, sim_t, sim_u, sim_n, total)
