# ============================================================
# fix_batch28_gemma_status_in_crosstest.py
# ============================================================
# Batch 28 originally used an older binary classification scheme
# (HALLUCINATED / DID_NOT_HALLUCINATE) before being reclassified
# into the standard TARGETED / UNTARGETED / NONE scheme, stored in
# outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json.
#
# crosstest_qwen25_on_gemma_images_twophase.py read the plain
# SUMMARY.json for both batch 28 and batch 50, so every batch 28
# entry's stored "gemma_status" is wrong (still the old binary
# label), even though the actual Qwen answer and Qwen's own
# classification in that same file are completely correct and
# do NOT need to be recomputed.
#
# This script only patches the gemma_status field for batch 28
# entries, using the already-computed reclassified labels. No
# Ollama calls are made.
# ============================================================

import json

CROSSTEST_STATE_PATH = "outputs_crosstest_qwen25_on_gemma_images_twophase/entry_state.json"
RECLASSIFIED_PATH     = "outputs_strict_batch_28/RECLASSIFIED_SUMMARY.json"

with open(RECLASSIFIED_PATH, "r") as f:
    reclassified = json.load(f)

reclassified_map = {r["entry_num"]: r["final_category"] for r in reclassified}

with open(CROSSTEST_STATE_PATH, "r") as f:
    state = json.load(f)

fixed_count = 0
for key, s in state.items():
    if s.get("batch_dir") != "outputs_strict_batch_28":
        continue
    entry_num = s.get("entry_num")
    if entry_num in reclassified_map:
        old_status = s.get("gemma_status")
        new_status = reclassified_map[entry_num]
        if old_status != new_status:
            print(f"  Fixing {key}: gemma_status {old_status} -> {new_status}")
            s["gemma_status"] = new_status
            fixed_count += 1

with open(CROSSTEST_STATE_PATH, "w") as f:
    json.dump(state, f, indent=2)

print(f"\nFixed {fixed_count} batch 28 entries.")
print(f"Saved corrected file to: {CROSSTEST_STATE_PATH}")
