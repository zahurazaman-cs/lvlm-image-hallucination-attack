#!/bin/bash
# ============================================================
# run_batch.sh — orchestrates Phase 1 / Phase 2 alternation
# ============================================================
# Runs, for iterations 1 through MAX_STEP10_ITER (5):
#   python3 pipeline_phase1_images.py --iter N     (then EXITS)
#   python3 pipeline_phase2_victim_judge.py --iter N   (then EXITS)
#
# Each phase is a completely separate process invocation, so Qwen-
# Image-Edit-2509's ~58GB is FULLY released before Ollama's model-
# switching (Gemma <-> Qwen2.5-VL) ever runs, and vice versa. This
# prevents the memory-pressure/machine-freeze issue that happens
# when both are resident simultaneously on this shared-memory box.
#
# Stops early if all entries have reached a final status (TARGETED /
# UNTARGETED / NONE / SKIPPED_NO_PHOTO) before hitting iteration 5.
#
# USAGE:
#   tmux new -s allucinat_victim_28
#   cd ~/Downloads/Allucination\ \(2\)/Allucination
#   source venv/bin/activate
#   bash run_batch.sh
#   (Ctrl+B, D to detach)
# ============================================================

set -e

# Fixes CUDA "out of memory" errors that can occur on model reload even
# when memory is technically free, by letting PyTorch's allocator use
# flexible memory segments instead of requiring one giant contiguous block.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAX_ITER=5
STATE_FILE="outputs_victim_qwen_601_650/entry_state.json"   # EDIT PER BATCH

for iter in $(seq 1 $MAX_ITER); do
    echo ""
    echo "============================================================"
    echo "ITERATION $iter — PHASE 1 (image generation)"
    echo "============================================================"
    python3 pipeline_phase1_images.py --iter $iter

    echo ""
    echo "============================================================"
    echo "ITERATION $iter — PHASE 2 (victim + judge)"
    echo "============================================================"
    python3 pipeline_phase2_victim_judge.py --iter $iter

    # Check if any entries are still PENDING (need another iteration)
    PENDING_COUNT=$(python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
pending = sum(1 for s in state.values() if s['status'] == 'PENDING')
print(pending)
")

    echo ""
    echo "After iteration $iter: $PENDING_COUNT entries still pending"

    if [ "$PENDING_COUNT" -eq "0" ]; then
        echo "All entries finalized — stopping early (no need for further iterations)."
        break
    fi
done

echo ""
echo "============================================================"
echo "BATCH COMPLETE"
echo "============================================================"
python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)

targeted = sum(1 for s in state.values() if s['status'] == 'TARGETED')
untargeted = sum(1 for s in state.values() if s['status'] == 'UNTARGETED')
none_hall = sum(1 for s in state.values() if s['status'] == 'NONE')
skipped = sum(1 for s in state.values() if s['status'] == 'SKIPPED_NO_PHOTO')
scored = targeted + untargeted + none_hall

print(f'Scored total: {scored}  (skipped: {skipped})')
if scored > 0:
    print(f'TARGETED   : {targeted} ({100*targeted/scored:.1f}%)')
    print(f'UNTARGETED : {untargeted} ({100*untargeted/scored:.1f}%)')
    print(f'NONE       : {none_hall} ({100*none_hall/scored:.1f}%)')
"
