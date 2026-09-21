# ============================================================
# find_flipped_entries.py
# ============================================================
# Finds specific entries where the baseline and the main symbolic
# attack DISAGREED on the same question, and prints out exactly
# which two image files to open and look at side by side.
#
# No technical knowledge needed to use this — just run it and
# read the plain-English output.
# ============================================================

import glob
import json


def extract_question_from_step8(entry_dir, final_iter):
    path = f"{entry_dir}/step8_victim_answer_iter{final_iter}.txt"
    if not glob.glob(path):
        return None
    with open(path, "r") as f:
        first_line = f.readline().strip()
    if first_line.startswith("Question:"):
        return first_line[len("Question:"):].strip()
    return None


def is_hallucinating(status):
    return status in ("TARGETED", "UNTARGETED")


def find_symbolic_image(entry_dir, final_iter):
    candidates = glob.glob(f"{entry_dir}/step5_image_o{final_iter}_s*.png")
    return candidates[0] if candidates else None


print("=" * 70)
print("SECTION 1: qa_data.json — where the BASELINE won")
print("(baseline fooled the model, but your real symbolic attack did not)")
print("=" * 70)

with open("outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json", "r") as f:
    reclassified = json.load(f)

symbolic_lookup = {}
for r in reclassified:
    entry_dir = f"outputs_strict_batch_28/entry_{r['entry_num']}"
    final_iter = r.get("final_iteration", r.get("iterations_needed"))
    q = extract_question_from_step8(entry_dir, final_iter)
    if q:
        symbolic_lookup[q.strip()] = (r["final_category"], entry_dir, final_iter)

with open("outputs_baseline_text_overlay_20/SUMMARY.json", "r") as f:
    baseline = json.load(f)

found_1 = 0
for r in baseline:
    q = r.get("question", "").strip()
    baseline_status = r.get("status")
    if q not in symbolic_lookup:
        continue
    symbolic_status, sym_entry_dir, sym_iter = symbolic_lookup[q]

    if is_hallucinating(baseline_status) and not is_hallucinating(symbolic_status):
        found_1 += 1
        baseline_img = f"outputs_baseline_text_overlay_20/entry_{r['entry']}/overlaid.png"
        symbolic_img = find_symbolic_image(sym_entry_dir, sym_iter)
        print(f"\nEXAMPLE {found_1}")
        print(f"Question: {q}")
        print(f"Baseline said (WRONG): tricked the model")
        print(f"Symbolic attack said (CORRECT): did not trick the model")
        print(f"Open and compare these two files:")
        print(f"  1. Baseline image:  {baseline_img}")
        print(f"  2. Symbolic image:  {symbolic_img}")
        if found_1 >= 3:
            break

if found_1 == 0:
    print("No flipped entries found in this direction.")


print()
print("=" * 70)
print("SECTION 2: NQ-Swap — where the SYMBOLIC ATTACK won")
print("(your real attack fooled the model, but the baseline did not)")
print("=" * 70)

symbolic_lookup = {}
with open("outputs_strict_batch_nqswap_26_50/RECLASSIFIED_SUMMARY.json", "r") as f:
    reclassified = json.load(f)
for r in reclassified:
    entry_dir = f"{r['batch_dir']}/entry_{r['entry_num']}"
    final_iter = r.get("final_iteration", r.get("iterations_needed"))
    q = extract_question_from_step8(entry_dir, final_iter)
    if q:
        symbolic_lookup[q.strip()] = (r["final_category"], entry_dir, final_iter)

for batch_name in ["outputs_strict_batch_nqswap_51_75",
                    "outputs_strict_batch_nqswap_76_100"]:
    with open(f"{batch_name}/SUMMARY.json", "r") as f:
        summary = json.load(f)
    for r in summary:
        if r["status"] == "SKIPPED_NO_PHOTO":
            continue
        entry_dir = f"{batch_name}/entry_{r['entry_num']}"
        q = extract_question_from_step8(entry_dir, r["iterations_needed"])
        if q:
            symbolic_lookup[q.strip()] = (r["status"], entry_dir, r["iterations_needed"])

with open("outputs_baseline_text_overlay_nqswap_20/SUMMARY.json", "r") as f:
    baseline = json.load(f)

found_2 = 0
for r in baseline:
    q = r.get("question", "").strip()
    baseline_status = r.get("status")
    if q not in symbolic_lookup:
        continue
    symbolic_status, sym_entry_dir, sym_iter = symbolic_lookup[q]

    if is_hallucinating(symbolic_status) and not is_hallucinating(baseline_status):
        found_2 += 1
        baseline_img = f"outputs_baseline_text_overlay_nqswap_20/entry_{r['entry']}/overlaid.png"
        symbolic_img = find_symbolic_image(sym_entry_dir, sym_iter)
        print(f"\nEXAMPLE {found_2}")
        print(f"Question: {q}")
        print(f"Baseline said (CORRECT): did not trick the model")
        print(f"Symbolic attack said (WRONG): tricked the model")
        print(f"Open and compare these two files:")
        print(f"  1. Baseline image:  {baseline_img}")
        print(f"  2. Symbolic image:  {symbolic_img}")
        if found_2 >= 3:
            break

if found_2 == 0:
    print("No flipped entries found in this direction.")


print()
print("=" * 70)
print("SECTION 3: TruthfulQA — where the SYMBOLIC ATTACK won")
print("(your real attack fooled the model, but the baseline did not)")
print("=" * 70)

with open("outputs_strict_batch_truthfulqa_22/SUMMARY.json", "r") as f:
    symbolic_summary = json.load(f)

symbolic_lookup = {}
for r in symbolic_summary:
    if r["status"] == "SKIPPED_NO_PHOTO":
        continue
    entry_dir = f"outputs_strict_batch_truthfulqa_22/entry_{r['entry_num']}"
    q = extract_question_from_step8(entry_dir, r["iterations_needed"])
    if q:
        symbolic_lookup[q.strip()] = (r["status"], entry_dir, r["iterations_needed"])

with open("outputs_baseline_text_overlay_truthfulqa_22/SUMMARY.json", "r") as f:
    baseline = json.load(f)

found_3 = 0
for r in baseline:
    q = r.get("question", "").strip()
    baseline_status = r.get("status")
    if q not in symbolic_lookup:
        continue
    symbolic_status, sym_entry_dir, sym_iter = symbolic_lookup[q]

    if is_hallucinating(symbolic_status) and not is_hallucinating(baseline_status):
        found_3 += 1
        baseline_img = f"outputs_baseline_text_overlay_truthfulqa_22/entry_{r['entry']}/overlaid.png"
        symbolic_img = find_symbolic_image(sym_entry_dir, sym_iter)
        print(f"\nEXAMPLE {found_3}")
        print(f"Question: {q}")
        print(f"Baseline said (CORRECT): did not trick the model")
        print(f"Symbolic attack said (WRONG): tricked the model")
        print(f"Open and compare these two files:")
        print(f"  1. Baseline image:  {baseline_img}")
        print(f"  2. Symbolic image:  {symbolic_img}")
        if found_3 >= 3:
            break

if found_3 == 0:
    print("No flipped entries found in this direction.")

print()
print("=" * 70)
print("HOW TO ACTUALLY LOOK AT THESE")
print("=" * 70)
print("Just open each pair of image files listed above like any normal")
print("picture file (double-click them in your file explorer, or use")
print("VS Code's built-in image viewer). Put them side by side on your")
print("screen and compare how the fake visual clue looks in each one.")
