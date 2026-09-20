# ============================================================
# batch_config.py — shared config for the two-phase pipeline
# ============================================================
# Edit ONLY this file when moving to the next batch (28, 23, 26,
# 48, 50, 50). Both phase1 and phase2 scripts import from here.
# ============================================================

DATA_PATH = "qa_data.json"

# BATCH 1 of 6 — same 28 entries as the original outputs_strict_batch_28.
BATCH_INDICES = [
    2, 3, 5, 7, 15, 18, 22, 26, 30, 36, 44, 50, 53, 54, 58, 63,
    65, 67, 68, 69, 70, 76, 77, 79, 83, 85, 93, 94
]

BASE_OUTPUT_DIR = "outputs_victim_qwen_28"

MAX_STEP6_ITER  = 5     # image-generation coherence retries (per outer iter)
MAX_STEP10_ITER = 5     # max hallucination-escalation iterations per entry

JUDGE_MODEL     = "huihui_ai/gemma-4-abliterated"   # Steps 2,3,3.5,6,9
VICTIM_MODEL    = "qwen2.5vl:7b"                    # Step 8 ONLY

OLLAMA_URL      = "http://localhost:11434/api"

# Directories from all six prior batches — used to locate each
# entry's existing reference photo instead of calling SerpApi again.
PRIOR_BATCH_DIRS = [
    "outputs_strict_batch_28",
    "outputs_strict_batch_50",
    "outputs_strict_batch_151_200",
    "outputs_strict_batch_201_280",
    "outputs_strict_batch_281_360",
    "outputs_strict_batch_361_440_combined",
]

STATE_PATH = f"{BASE_OUTPUT_DIR}/entry_state.json"
