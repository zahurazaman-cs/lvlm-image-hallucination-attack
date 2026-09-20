# ============================================================
# build_human_eval_sample.py
# ============================================================
# Builds a random, stratified sample of entries for human judge
# validation, drawn from the ORIGINAL ATTACK results across all
# three datasets. Produces TWO files:
#
#   for_reviewers.csv — what your human raters actually see:
#     question, correct answer, hallucinated answer, and the
#     victim's real answer. NO judge label included.
#
#   answer_key.csv — kept PRIVATE, has the judge's actual label
#     for each entry, used only afterward to score agreement.
#     Do NOT show this file to your raters before they finish.
#
# Samples ~40 entries total, proportionally across datasets, with
# a mix of TARGETED / UNTARGETED / NONE so raters see a balanced
# set of judge decisions to check.
# ============================================================

import json
import glob
import re
import random
import csv

random.seed(42)  # reproducible sample


def classify_file(path):
    with open(path, "r") as f:
        text = f.read()
    if "HALLUCINATING_TARGETED: YES" in text:
        return "TARGETED"
    elif "HALLUCINATING_UNTARGETED: YES" in text:
        return "UNTARGETED"
    else:
        return "NONE"


def get_victim_answer(step8_path):
    if not glob.glob(step8_path):
        return None
    with open(step8_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("Victim Answer:"):
            return line[len("Victim Answer:"):].strip()
    return None


def extract_question(step8_path):
    if not glob.glob(step8_path):
        return None
    with open(step8_path, "r") as f:
        first_line = f.readline().strip()
    if first_line.startswith("Question:"):
        return first_line[len("Question:"):].strip()
    return None


all_candidates = []  # (dataset, question, right_answer, hallucinated_answer, victim_answer, judge_label)

# ── qa_data.json: sample across all 14 batches ────────────────────
QA_DATA_BATCHES = [
    "outputs_strict_batch_50", "outputs_strict_batch_151_200",
    "outputs_strict_batch_201_280", "outputs_strict_batch_281_360",
    "outputs_strict_batch_361_440_combined", "outputs_strict_batch_441_520",
    "outputs_strict_batch_521_600", "outputs_strict_batch_601_650",
    "outputs_strict_batch_651_700", "outputs_strict_batch_701_750",
    "outputs_strict_batch_751_800", "outputs_strict_batch_801_850",
    "outputs_strict_batch_851_900",
]
for batch_dir in QA_DATA_BATCHES:
    try:
        with open(f"{batch_dir}/SUMMARY.json", "r") as f:
            summary = json.load(f)
    except FileNotFoundError:
        continue
    for r in summary:
        if r["status"] == "SKIPPED_NO_PHOTO":
            continue
        entry_dir = f"{batch_dir}/entry_{r['entry_num']}"
        step8_path = f"{entry_dir}/step8_victim_answer_iter{r['iterations_needed']}.txt"
        q = extract_question(step8_path)
        va = get_victim_answer(step8_path)
        if q and va:
            all_candidates.append(("qa_data.json", q, r["right_answer"],
                                    r["hallucinated_answer"], va, r["status"]))

# ── NQ-Swap: batches 51-75 and 76-100 (native scheme, simpler to pull) ──
for batch_dir in ["outputs_strict_batch_nqswap_51_75",
                   "outputs_strict_batch_nqswap_76_100"]:
    try:
        with open(f"{batch_dir}/SUMMARY.json", "r") as f:
            summary = json.load(f)
    except FileNotFoundError:
        continue
    for r in summary:
        if r["status"] == "SKIPPED_NO_PHOTO":
            continue
        entry_dir = f"{batch_dir}/entry_{r['entry_num']}"
        step8_path = f"{entry_dir}/step8_victim_answer_iter{r['iterations_needed']}.txt"
        q = extract_question(step8_path)
        va = get_victim_answer(step8_path)
        if q and va:
            all_candidates.append(("NQ-Swap", q, r["right_answer"],
                                    r["hallucinated_answer"], va, r["status"]))

# ── TruthfulQA: all 22 ─────────────────────────────────────────────
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
            all_candidates.append(("TruthfulQA", q, r["right_answer"],
                                    r["hallucinated_answer"], va, r["status"]))
except FileNotFoundError:
    pass

print(f"Total candidates pulled: {len(all_candidates)}")

# ── Stratified sample: aim for a mix of TARGETED/UNTARGETED/NONE ──
by_label = {"TARGETED": [], "UNTARGETED": [], "NONE": []}
for c in all_candidates:
    by_label[c[5]].append(c)

for k in by_label:
    random.shuffle(by_label[k])

TARGET_TOTAL = 40
per_label = TARGET_TOTAL // 3
sample = (by_label["TARGETED"][:per_label] +
          by_label["UNTARGETED"][:per_label] +
          by_label["NONE"][:per_label])
random.shuffle(sample)

print(f"Sample size: {len(sample)}")
print(f"  TARGETED : {sum(1 for s in sample if s[5]=='TARGETED')}")
print(f"  UNTARGETED: {sum(1 for s in sample if s[5]=='UNTARGETED')}")
print(f"  NONE     : {sum(1 for s in sample if s[5]=='NONE')}")

# ── Write the two output files ────────────────────────────────────
with open("for_reviewers.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Dataset", "Question", "Documented Correct Answer",
                      "Hallucinated Answer (candidate)", "Model's Actual Answer",
                      "Your Judgment (TARGETED / UNTARGETED / NONE)"])
    for i, (dataset, q, right, hall, victim, _) in enumerate(sample, start=1):
        writer.writerow([i, dataset, q, right, hall, victim, ""])

with open("answer_key.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Judge Label (DO NOT SHARE THIS FILE)"])
    for i, (_, _, _, _, _, label) in enumerate(sample, start=1):
        writer.writerow([i, label])

print("\nSaved for_reviewers.csv (share this) and answer_key.csv (keep private)")
