# ============================================================
# get_entry_quotes.py
# ============================================================
# Pulls the exact text you need to quote in your figure caption
# or body text for the "Return of the Jedi" / Richard Marquand
# entry (outputs_strict_batch_truthfulqa_22/entry_4), which
# succeeded on iteration 3 (o3).
#
# Prints, in plain readable form:
#   1. The misleading description used to generate the image
#      at iteration 3 (step2_desc_iter3.txt)
#   2. The victim model's final answer (step8_answer.txt)
#   3. The judge's final classification (step9_judgment.txt)
#   4. The ground-truth true and hallucinated answers, pulled
#      from this batch's SUMMARY.json, for direct comparison
#
# No arguments needed — just run it from inside your project
# directory:
#
#   python3 get_entry_quotes.py
# ============================================================

import json
import os

ENTRY_DIR = "outputs_strict_batch_truthfulqa_22/entry_4"
BATCH_DIR = "outputs_strict_batch_truthfulqa_22"
ITER = 3  # the iteration that succeeded (o3)

def read_file(path):
    if not os.path.exists(path):
        return f"[FILE NOT FOUND: {path}]"
    with open(path, "r") as f:
        return f.read().strip()

print("=" * 70)
print(f"ENTRY: {ENTRY_DIR}  (succeeded at iteration {ITER})")
print("=" * 70)

# --- 1. Misleading description used at the winning iteration ---
desc_path = os.path.join(ENTRY_DIR, f"step2_desc_iter{ITER}.txt")
description = read_file(desc_path)
print(f"\n--- MISLEADING DESCRIPTION (step2_desc_iter{ITER}.txt) ---")
print(description)

# --- 2. Victim model's answer AT THE WINNING ITERATION ---
# Files are stored per-iteration (step8_victim_answer_iter{N}.txt),
# not as a single final file, so we pull the one matching ITER.
answer_path = os.path.join(ENTRY_DIR, f"step8_victim_answer_iter{ITER}.txt")
answer = read_file(answer_path)
print(f"\n--- VICTIM MODEL'S ANSWER AT ITERATION {ITER} (step8_victim_answer_iter{ITER}.txt) ---")
print(answer)

# --- 3. Judge's classification AT THE WINNING ITERATION ---
judgment_path = os.path.join(ENTRY_DIR, f"step9_hallucination_iter{ITER}.txt")
judgment = read_file(judgment_path)
print(f"\n--- JUDGE'S CLASSIFICATION AT ITERATION {ITER} (step9_hallucination_iter{ITER}.txt) ---")
print(judgment)

# --- 4. Ground-truth true/hallucinated answers from SUMMARY.json ---
summary_path = os.path.join(BATCH_DIR, "SUMMARY.json")
print(f"\n--- GROUND TRUTH (from {summary_path}) ---")
if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        summary = json.load(f)

    # SUMMARY.json is usually a dict keyed by entry number or a list;
    # handle both shapes defensively.
    entry_record = None
    if isinstance(summary, dict):
        for key in ("4", 4, "entry_4"):
            if key in summary:
                entry_record = summary[key]
                break
    elif isinstance(summary, list):
        for rec in summary:
            if str(rec.get("entry_num")) == "4":
                entry_record = rec
                break

    if entry_record:
        # NOTE: SUMMARY.json for this batch does not store the question
        # text itself (only person_name, right_answer, hallucinated_answer,
        # status, iterations_needed). Fill QUESTION_TEXT below by hand if
        # you want it printed here too — it's not retrievable from this file.
        print(f"Person queried about : {entry_record.get('person_name', '[not found]')}")
        print(f"True answer          : {entry_record.get('right_answer', '[not found]')}")
        print(f"Hallucinated answer  : {entry_record.get('hallucinated_answer', '[not found]')}")
        print(f"Final status         : {entry_record.get('status', '[not found]')}")
        print(f"Iterations needed    : {entry_record.get('iterations_needed', '[not found]')}")
    else:
        print("[Could not locate entry_4's record in SUMMARY.json — "
              "open the file manually and search for \"entry_4\" or "
              "\"\\\"4\\\":\" to find the right block.]")
else:
    print(f"[FILE NOT FOUND: {summary_path}]")

print("\n" + "=" * 70)
print("Copy the DESCRIPTION, ANSWER, and GROUND TRUTH blocks above")
print("directly into your caption/body text as exact quotes.")
print("=" * 70)
