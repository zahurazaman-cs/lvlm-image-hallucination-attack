# ============================================================
# find_missing_gemma_qwen_pairs.py
# ============================================================
# Identifies the exact entries where Gemma hallucinated (TARGETED
# or UNTARGETED) in one of the twelve reuse-method batches, but no
# corresponding Qwen2.5-VL result exists for that same entry
# number (skipped or errored in the original Qwen reuse run).
#
# These are the 17 entries missing from Direction 1 of the full
# 500-entry transferability analysis. Prints the list and saves it
# to a JSON file that fill_missing_qwen_on_gemma_images.py reads
# directly, so nothing needs to be retyped.
# ============================================================

import json

REUSE_BATCH_PAIRS = [
    ("outputs_victim_qwen_151_200",       "outputs_strict_batch_151_200"),
    ("outputs_victim_qwen_201_280",       "outputs_strict_batch_201_280"),
    ("outputs_victim_qwen_281_360",       "outputs_strict_batch_281_360"),
    ("outputs_victim_qwen_361_440",       "outputs_strict_batch_361_440_combined"),
    ("outputs_victim_qwen_441_520",       "outputs_strict_batch_441_520"),
    ("outputs_victim_qwen_521_600",       "outputs_strict_batch_521_600"),
    ("outputs_victim_qwen_601_650_reuse", "outputs_strict_batch_601_650"),
    ("outputs_victim_qwen_651_700",       "outputs_strict_batch_651_700"),
    ("outputs_victim_qwen_701_750",       "outputs_strict_batch_701_750"),
    ("outputs_victim_qwen_751_800",       "outputs_strict_batch_751_800"),
    ("outputs_victim_qwen_801_850",       "outputs_strict_batch_801_850"),
    ("outputs_victim_qwen_851_900",       "outputs_strict_batch_851_900"),
]

OUTPUT_PATH = "missing_gemma_qwen_pairs.json"


def load_gemma_batch(path):
    with open(f"{path}/SUMMARY.json", "r") as f:
        return {r["entry_num"]: r["status"] for r in json.load(f)}


def load_qwen_batch(path):
    with open(f"{path}/entry_state.json", "r") as f:
        state = json.load(f)
    return {s["entry_num"]: s["status"] for s in state.values()}


missing = []

for qwen_dir, gemma_dir in REUSE_BATCH_PAIRS:
    gemma_labels = load_gemma_batch(gemma_dir)
    qwen_labels  = load_qwen_batch(qwen_dir)

    for entry_num, g_label in gemma_labels.items():
        if g_label not in ("TARGETED", "UNTARGETED"):
            continue
        if entry_num not in qwen_labels:
            missing.append({
                "gemma_dir": gemma_dir,
                "qwen_dir": qwen_dir,
                "entry_num": entry_num,
                "gemma_status": g_label,
            })

print(f"Found {len(missing)} Gemma-hallucinated entries with no "
      f"corresponding Qwen result:\n")
for m in missing:
    print(f"  {m['gemma_dir']} entry #{m['entry_num']} "
          f"(Gemma: {m['gemma_status']}) — missing from {m['qwen_dir']}")

with open(OUTPUT_PATH, "w") as f:
    json.dump(missing, f, indent=2)

print(f"\nSaved list to: {OUTPUT_PATH}")
