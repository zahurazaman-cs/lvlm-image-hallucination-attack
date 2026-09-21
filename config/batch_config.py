# ============================================================
# batch_config.py — shared config for the two-phase pipeline
# ============================================================
# Edit ONLY this file when moving to the next batch (28, 23, 26,
# 48, 50, 50). Both phase1 and phase2 scripts import from here.
# ============================================================

DATA_PATH = "qa_data.json"

# BATCH 2 of 6 — same 23 entries as the original outputs_strict_batch_50.
BATCH_INDICES = [
    100, 104, 105, 107, 109, 110, 112, 115, 119, 123, 124, 126,
    129, 130, 133, 134, 136, 137, 139, 140, 144, 147, 148
]

BASE_OUTPUT_DIR = "outputs_victim_qwen_50"

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
