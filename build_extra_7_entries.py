# ============================================================
# build_extra_7_entries.py
# ============================================================
# Pulls 7 REAL entries from your actual completed results —
# TruthfulQA and NQ-Swap, both Gemma and Qwen2.5-VL original
# attacks. No fabricated data, everything pulled directly from
# your existing output files.
#
# Aims for roughly: 2 TruthfulQA-Gemma, 2 TruthfulQA-Qwen,
# 2 NQ-Swap-Gemma, 1 NQ-Swap-Qwen, randomly sampled.
#
# Saves to: extra_7_items.csv, same format as form_items.csv
# ============================================================

import json
import glob
import re
import random
import csv

random.seed(7)

LABEL_DISPLAY = {
    "TARGETED": "Matches the alternative answer (the model was fooled into "
                "giving the specific false answer)",
    "UNTARGETED": "Wrong, and doesn't match either answer shown "
                  "(the model was confused, but not in the intended way)",
    "NONE": "Matches the documented correct answer (the model was not fooled)",
}

def extract_question(step8_path):
    if not glob.glob(step8_path):
        return None
    with open(step8_path, "r") as f:
        first_line = f.readline().strip()
    if first_line.startswith("Question:"):
        return first_line[len("Question:"):].strip()
    return None

def get_victim_answer(step8_path):
    if not glob.glob(step8_path):
        return None
    with open(step8_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("Victim Answer:"):
            return line[len("Victim Answer:"):].strip()
    return None

pool = {"truthfulqa_gemma": [], "truthfulqa_qwen": [],
        "nqswap_gemma": [], "nqswap_qwen": []}

# ── TruthfulQA, Gemma (native scheme, single batch) ──────────────
try:
    with open("outputs_strict_batch_truthfulqa_22/SUMMARY.json", "r") as f:
        summary = json.load(f)
    for r in summary:
        if r["status"] == "SKIPPED_NO_PHOTO":
            continue
        entry_dir = f"outputs_strict_batch_truthfulqa_22/entry_{r['entry_num']}"
        step8_path = f"{entry_dir}/step8_victim_answer_iter{r['iterations_needed']}.txt"
        q = extract_question(step8_path)
        va = get_victim_answer(step8_path)
        if q and va:
            pool["truthfulqa_gemma"].append(
                ("TruthfulQA (Gemma)", q, r["right_answer"],
                 r["hallucinated_answer"], va, r["status"]))
except FileNotFoundError:
    print("WARNING: outputs_strict_batch_truthfulqa_22 not found")

# ── TruthfulQA, Qwen2.5-VL (entry_state.json already has everything) ──
try:
    with open("outputs_victim_qwen_truthfulqa_22/entry_state.json", "r") as f:
        state = json.load(f)
    for s in state.values():
        if s.get("status") not in ("TARGETED", "UNTARGETED", "NONE"):
            continue
        pool["truthfulqa_qwen"].append(
            ("TruthfulQA (Qwen2.5-VL)", s["question"], s["right_answer"],
             s["hallucinated_answer"], s["victim_answer"], s["status"]))
except FileNotFoundError:
    print("WARNING: outputs_victim_qwen_truthfulqa_22 not found")

# ── NQ-Swap, Gemma (batches 51-75, 76-100, native scheme) ────────
for batch_dir in ["outputs_strict_batch_nqswap_51_75",
                   "outputs_strict_batch_nqswap_76_100"]:
    try:
        with open(f"{batch_dir}/SUMMARY.json", "r") as f:
            summary = json.load(f)
        for r in summary:
            if r["status"] == "SKIPPED_NO_PHOTO":
                continue
            entry_dir = f"{batch_dir}/entry_{r['entry_num']}"
            step8_path = f"{entry_dir}/step8_victim_answer_iter{r['iterations_needed']}.txt"
            q = extract_question(step8_path)
            va = get_victim_answer(step8_path)
            if q and va:
                pool["nqswap_gemma"].append(
                    ("NQ-Swap (Gemma)", q, r["right_answer"],
                     r["hallucinated_answer"], va, r["status"]))
    except FileNotFoundError:
        continue

# ── NQ-Swap, Qwen2.5-VL (entry_state.json already has everything) ──
try:
    with open("outputs_victim_qwen_nqswap_100/entry_state.json", "r") as f:
        state = json.load(f)
    for s in state.values():
        if s.get("status") not in ("TARGETED", "UNTARGETED", "NONE"):
            continue
        pool["nqswap_qwen"].append(
            ("NQ-Swap (Qwen2.5-VL)", s["question"], s["right_answer"],
             s["hallucinated_answer"], s["victim_answer"], s["status"]))
except FileNotFoundError:
    print("WARNING: outputs_victim_qwen_nqswap_100 not found")

for k, v in pool.items():
    random.shuffle(v)
    print(f"{k}: {len(v)} candidates available")

sample = []
sample += pool["truthfulqa_gemma"][:2]
sample += pool["truthfulqa_qwen"][:2]
sample += pool["nqswap_gemma"][:2]
sample += pool["nqswap_qwen"][:1]

if len(sample) < 7:
    print(f"\nWARNING: only found {len(sample)} of 7 requested — "
          f"one or more source files may be missing or too small. "
          f"Check the WARNING lines above.")

with open("extra_7_items.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Dataset", "Question", "Documented Correct Answer",
                      "Alternative Answer", "Model's Actual Response",
                      "Judge's Classification (for Question B)"])
    for i, (dataset, q, right, hall, victim, label) in enumerate(sample, start=1):
        writer.writerow([f"extra_{i}", dataset, q, right, hall, victim,
                          LABEL_DISPLAY[label]])

print(f"\nSaved {len(sample)} real entries to extra_7_items.csv")
