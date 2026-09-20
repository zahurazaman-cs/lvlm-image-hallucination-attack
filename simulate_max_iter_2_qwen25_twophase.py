# ============================================================
# simulate_max_iter_2_qwen25_twophase.py
# ============================================================
# Simulates what the FINAL hallucination classification would
# have been for Qwen2.5-VL if its escalation loop had been capped
# at 2 iterations instead of 5.
#
# SCOPE: only the 51 entries from the two two-phase batches
# (outputs_victim_qwen_28, outputs_victim_qwen_50), since these
# are the ONLY Qwen2.5-VL entries anywhere in this project where a
# fresh image was genuinely escalated specifically against
# Qwen2.5-VL. Every other batch across all three datasets used the
# reuse method, where Qwen2.5-VL was shown one fixed image and
# never given a second attempt at all, so there is no iteration 2
# data to simulate from for those — a cap could not have changed
# anything that never had a second attempt to begin with.
#
# THIS MAKES NO NEW MODEL CALLS — reads existing per-iteration
# judge files only.
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


def simulate_cap(entry_dir):
    iter1_path = f"{entry_dir}/step9_hallucination_iter1.txt"
    if not glob.glob(iter1_path):
        return None
    iter1_status = classify_file(iter1_path)
    if iter1_status == "TARGETED":
        return "TARGETED"
    iter2_path = f"{entry_dir}/step9_hallucination_iter2.txt"
    if not glob.glob(iter2_path):
        return iter1_status
    return classify_file(iter2_path)


TWO_PHASE_BATCH_DIRS = [
    "outputs_victim_qwen_28",
    "outputs_victim_qwen_50",
]

real_t = real_u = real_n = 0
sim_t = sim_u = sim_n = 0
total = 0

for batch_dir in TWO_PHASE_BATCH_DIRS:
    with open(f"{batch_dir}/entry_state.json", "r") as f:
        state = json.load(f)
    for key, s in state.items():
        if s.get("status") not in ("TARGETED", "UNTARGETED", "NONE"):
            continue
        total += 1
        real_status = s["status"]
        if real_status == "TARGETED":
            real_t += 1
        elif real_status == "UNTARGETED":
            real_u += 1
        else:
            real_n += 1

        entry_num = s.get("entry_num", key)
        entry_dir = f"{batch_dir}/entry_{entry_num}"
        sim_status = simulate_cap(entry_dir)
        if sim_status == "TARGETED":
            sim_t += 1
        elif sim_status == "UNTARGETED":
            sim_u += 1
        else:
            sim_n += 1

real_hall = real_t + real_u
sim_hall = sim_t + sim_u

print(f"{'='*65}")
print(f"qa_data.json (HaluEval) — Qwen2.5-VL, two-phase entries only")
print(f"{'='*65}")
print(f"Total entries: {total}")
print()
print(f"REAL (cap=5)  : TARGETED {real_t} ({100*real_t/total:.1f}%), "
      f"UNTARGETED {real_u} ({100*real_u/total:.1f}%), "
      f"NONE {real_n} ({100*real_n/total:.1f}%)")
print(f"              TOTAL HALLUCINATION: {real_hall} "
      f"({100*real_hall/total:.1f}%)")
print()
print(f"SIMULATED (cap=2): TARGETED {sim_t} ({100*sim_t/total:.1f}%), "
      f"UNTARGETED {sim_u} ({100*sim_u/total:.1f}%), "
      f"NONE {sim_n} ({100*sim_n/total:.1f}%)")
print(f"              TOTAL HALLUCINATION: {sim_hall} "
      f"({100*sim_hall/total:.1f}%)")
print()
diff = sim_hall - real_hall
direction = "INCREASE" if diff > 0 else ("DECREASE" if diff < 0 else "NO CHANGE")
print(f"Effect of capping at 2 iterations: {direction} of "
      f"{abs(diff)} entries ({abs(100*diff/total):.1f} percentage points)")
print()
print(f"NOTE: this covers only the {total} entries where Qwen2.5-VL "
      f"genuinely escalated its own fresh images. The remaining "
      f"449 qa_data.json entries, and all 100 NQ-Swap and all 22 "
      f"TruthfulQA Qwen2.5-VL entries, used the reuse method — "
      f"Qwen2.5-VL was shown one fixed image and never given a "
      f"second attempt, so there is no iteration cap to simulate "
      f"there at all.")
