# ============================================================
# compare_baseline_vs_symbolic_same20.py
# ============================================================
# Fair, apples-to-apples comparison: the text overlay baseline's
# result on its 20 entries against your ORIGINAL symbolic attack's
# result on those EXACT SAME 20 entries (not the full 500 average,
# which would not be a matched comparison).
#
# Uses batch 28's RECLASSIFIED_SUMMARY.json, restricted to just
# the 20 entry numbers the baseline script used.
#
# THIS MAKES NO NEW MODEL CALLS.
# ============================================================

import json

BASELINE_SUMMARY_PATH = "outputs_baseline_text_overlay_20/SUMMARY.json"
RECLASSIFIED_PATH = "outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json"

BATCH_INDICES = [2, 3, 5, 7, 15, 18, 22, 26, 30, 36, 44, 50, 53, 54, 58, 63, 65, 67, 68, 69]

with open(BASELINE_SUMMARY_PATH, "r") as f:
    baseline = json.load(f)

baseline_targeted = sum(1 for r in baseline if r.get("status") == "TARGETED")
baseline_untargeted = sum(1 for r in baseline if r.get("status") == "UNTARGETED")
baseline_none = sum(1 for r in baseline if r.get("status") == "NONE")
baseline_total = baseline_targeted + baseline_untargeted + baseline_none

with open(RECLASSIFIED_PATH, "r") as f:
    reclassified = json.load(f)

symbolic_lookup = {r["entry_num"]: r["final_category"] for r in reclassified}

symbolic_targeted = symbolic_untargeted = symbolic_none = 0
for idx in BATCH_INDICES:
    # entry_num in RECLASSIFIED_SUMMARY.json is 1-indexed, one higher
    # than the raw 0-indexed line numbers used in BATCH_INDICES
    status = symbolic_lookup.get(idx + 1)
    if status == "TARGETED":
        symbolic_targeted += 1
    elif status == "UNTARGETED":
        symbolic_untargeted += 1
    elif status == "NONE":
        symbolic_none += 1

symbolic_total = symbolic_targeted + symbolic_untargeted + symbolic_none

print(f"{'='*65}")
print(f"FAIR COMPARISON — same 20 entries, two different attack designs")
print(f"{'='*65}")
print()
print(f"BASELINE (crude text overlay, no diffusion editing):")
print(f"  n={baseline_total}  TARGETED={baseline_targeted} "
      f"UNTARGETED={baseline_untargeted} NONE={baseline_none}")
print(f"  Total hallucination: {baseline_targeted+baseline_untargeted}/"
      f"{baseline_total} ({100*(baseline_targeted+baseline_untargeted)/baseline_total:.1f}%)")
print()
print(f"MAIN ATTACK (symbolic visual cue via diffusion editing, same entries):")
print(f"  n={symbolic_total}  TARGETED={symbolic_targeted} "
      f"UNTARGETED={symbolic_untargeted} NONE={symbolic_none}")
print(f"  Total hallucination: {symbolic_targeted+symbolic_untargeted}/"
      f"{symbolic_total} ({100*(symbolic_targeted+symbolic_untargeted)/symbolic_total:.1f}%)")
print()

diff = (baseline_targeted+baseline_untargeted) - (symbolic_targeted+symbolic_untargeted)
print(f"Difference: {diff:+d} entries "
      f"({100*diff/baseline_total:+.1f} percentage points)")
