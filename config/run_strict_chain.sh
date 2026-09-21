#!/bin/bash
# ============================================================
# run_strict_chain.sh
# ============================================================
# Waits for the currently-running pipeline_strict_batch.py
# (2-entry test: Peggy Seeger + Jonathan Stark) to finish, then
# AUTOMATICALLY launches pipeline_strict_batch_28.py (full
# 28-entry strict run) with the same tee-logging pattern.
#
# Run this in a SEPARATE tmux window/session from the one running
# pipeline_strict_batch.py — it does not touch that process, it
# just watches for it to exit.
#
# USAGE:
#   tmux new -s allucinat_chain
#   cd ~/Downloads/Allucination\ \(2\)/Allucination
#   source venv/bin/activate
#   bash run_strict_chain.sh
#   (then Ctrl+B, D to detach)
# ============================================================

echo "============================================================"
echo "Watching for pipeline_strict_batch.py to finish..."
echo "(checking every 30 seconds)"
echo "============================================================"

while pgrep -f "python3 pipeline_strict_batch.py" > /dev/null; do
    sleep 30
done

echo ""
echo "============================================================"
echo "pipeline_strict_batch.py has finished (2-entry test complete)."
echo "Starting pipeline_strict_batch_28.py (28-entry full run) now..."
echo "============================================================"
echo ""

# Give a brief pause so any final file writes from the previous
# run fully settle before starting the next one.
sleep 5

mkdir -p outputs_strict_batch_28

python3 pipeline_strict_batch_28.py 2>&1 | tee outputs_strict_batch_28/full_run_log.txt

echo ""
echo "============================================================"
echo "pipeline_strict_batch_28.py finished. Chain complete."
echo "============================================================"
