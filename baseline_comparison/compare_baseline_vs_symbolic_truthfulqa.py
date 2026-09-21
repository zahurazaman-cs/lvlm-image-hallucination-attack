# ============================================================
# compare_baseline_vs_symbolic_truthfulqa.py
# ============================================================
# Fair comparison: text overlay baseline vs the original symbolic
# attack, on TruthfulQA, matched by QUESTION TEXT. Question text
# is read from each entry's step8_victim_answer_iterN.txt file,
# since SUMMARY.json does not store it.
#
# THIS MAKES NO NEW MODEL CALLS.
# ============================================================

import glob
import json

BASELINE_SUMMARY_PATH = "outputs_baseline_text_overlay_truthfulqa_22/SUMMARY.json"
SYMBOLIC_BATCH_DIR = "outputs_strict_batch_truthfulqa_22"


def extract_question_from_step8(entry_dir, final_iter):
    path = f"{entry_dir}/step8_victim_answer_iter{final_iter}.txt"
    if not glob.glob(path):
        return None
    with open(path, "r") as f:
        first_line = f.readline().strip()
    if first_line.startswith("Question:"):
        return first_line[len("Question:"):].strip()
    return None


with open(f"{SYMBOLIC_BATCH_DIR}/SUMMARY.json", "r") as f:
    symbolic_summary = json.load(f)

symbolic_lookup = {}
for r in symbolic_summary:
    if r["status"] == "SKIPPED_NO_PHOTO":
        continue
    entry_dir = f"{SYMBOLIC_BATCH_DIR}/entry_{r['entry_num']}"
    q = extract_question_from_step8(entry_dir, r["iterations_needed"])
    if q:
        symbolic_lookup[q.strip()] = r["status"]

with open(BASELINE_SUMMARY_PATH, "r") as f:
    baseline = json.load(f)

baseline_t = baseline_u = baseline_n = 0
symbolic_t = symbolic_u = symbolic_n = 0
matched = 0
unmatched = []

for r in baseline:
    status = r.get("status")
    if status not in ("TARGETED", "UNTARGETED", "NONE"):
        continue
    if status == "TARGETED":
        baseline_t += 1
    elif status == "UNTARGETED":
        baseline_u += 1
    else:
        baseline_n += 1

    q = r.get("question", "").strip()
    sym_status = symbolic_lookup.get(q)
    if sym_status is None:
        unmatched.append(q)
        continue
    matched += 1
    if sym_status == "TARGETED":
        symbolic_t += 1
    elif sym_status == "UNTARGETED":
        symbolic_u += 1
    elif sym_status == "NONE":
        symbolic_n += 1

baseline_total = baseline_t + baseline_u + baseline_n
symbolic_total = symbolic_t + symbolic_u + symbolic_n

print(f"{'='*65}")
print(f"FAIR COMPARISON — TruthfulQA, matched by question text")
print(f"{'='*65}")
print(f"Symbolic lookup built from {len(symbolic_lookup)} original entries")
print(f"Baseline entries scored: {baseline_total}")
print(f"Matched to a symbolic attack result: {matched}")
if unmatched:
    print(f"Unmatched questions: {len(unmatched)}")
    for q in unmatched[:5]:
        print(f"  - {q}")
print()
print(f"BASELINE (crude text overlay):")
print(f"  n={baseline_total}  TARGETED={baseline_t} UNTARGETED={baseline_u} NONE={baseline_n}")
print(f"  Total hallucination: {baseline_t+baseline_u}/{baseline_total} "
      f"({100*(baseline_t+baseline_u)/baseline_total:.1f}%)")
print()
print(f"MAIN ATTACK (symbolic visual cue, same matched entries):")
print(f"  n={symbolic_total}  TARGETED={symbolic_t} UNTARGETED={symbolic_u} NONE={symbolic_n}")
if symbolic_total:
    print(f"  Total hallucination: {symbolic_t+symbolic_u}/{symbolic_total} "
          f"({100*(symbolic_t+symbolic_u)/symbolic_total:.1f}%)")
