# ============================================================
# batch_config.py — shared config for the two-phase pipeline
# ============================================================
# Edit ONLY this file when moving to the next batch (28, 23, 26,
# 48, 50, 50). Both phase1 and phase2 scripts import from here.
# ============================================================

DATA_PATH = "qa_data.json"

# BATCH: Qwen2.5-VL victim-model experiment, third batch — same 30
# entries as outputs_strict_batch_601_650, for direct comparison
# against the Gemma-victim results already recorded for these
# identical 30 people.
BATCH_INDICES = [
    600, 601, 602, 603, 604, 605, 606, 608, 609, 610, 614, 615,
    616, 617, 618, 620, 622, 624, 625, 626, 627, 628, 630, 632,
    633, 635, 636, 637, 641, 644
]

BASE_OUTPUT_DIR = "outputs_victim_qwen_601_650"

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
    "outputs_strict_batch_601_650",
]

STATE_PATH = f"{BASE_OUTPUT_DIR}/entry_state.json"
