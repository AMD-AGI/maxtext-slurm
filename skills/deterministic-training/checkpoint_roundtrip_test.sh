#!/bin/bash
# Phase 2c: Checkpoint roundtrip determinism test.
#
# Run A: train 20 steps straight
# Run B: train 10 steps with checkpoint, restore, train 10 more
# Compare: step 20 loss should be identical
#
# Usage: bash checkpoint_roundtrip_test.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPARE="$SCRIPT_DIR/compare_runs.py"
OUTPUTS_DIR="$REPO_DIR/outputs"

DET_ARGS="_env_DETERMINISTIC_MODE=1 enable_dropout=False per_device_batch_size=1"

newest_output_dir() {
    ls -dt "$OUTPUTS_DIR"/local_* 2>/dev/null | grep -v '\.log$' | head -1
}

cd "$REPO_DIR"

echo "=== Checkpoint Roundtrip Determinism Test ==="
echo ""

rm -rf ~/jax_cache

echo "[1/3] Run A: straight 20 steps..."
if ! ./in_container_run.sh llama2-70b -- $DET_ARGS steps=20 enable_checkpointing=False 2>&1 | tail -3; then
    echo "FAIL: Run A crashed"
    exit 1
fi
DIR_A=$(newest_output_dir)

echo ""
echo "[2/3] Run B-part1: 10 steps with checkpoint..."
if ! ./in_container_run.sh llama2-70b -- $DET_ARGS steps=10 enable_checkpointing=True checkpoint_period=10 2>&1 | tail -3; then
    echo "FAIL: Run B-part1 crashed"
    exit 1
fi
DIR_B1=$(newest_output_dir)
CKPT_DIR="$DIR_B1/llama2-70b_train_test/checkpoints"

echo ""
echo "[3/3] Run B-part2: restore from step 10, train to step 20..."
if ! ./in_container_run.sh llama2-70b -- $DET_ARGS steps=20 enable_checkpointing=False \
    load_full_state_path="$CKPT_DIR" 2>&1 | tail -3; then
    echo "FAIL: Run B-part2 crashed"
    exit 1
fi
DIR_B2=$(newest_output_dir)

echo ""
echo "=== Comparing ==="
echo "  Run A (straight): $(basename "$DIR_A")"
echo "  Run B (restored): $(basename "$DIR_B2")"
echo ""

RESULT=$(python3 "$COMPARE" "$DIR_A" "$DIR_B2" 2>&1)
echo "$RESULT" | grep -E 'step|EXACT|DIFFER|RESULT|^-'

if echo "$RESULT" | grep -q 'BIT-EXACT'; then
    echo ""
    echo "PASS: Checkpoint roundtrip is deterministic"
    exit 0
else
    echo ""
    echo "FAIL: Checkpoint roundtrip breaks determinism"
    echo "Note: steps 0-9 may differ (Run A computed from scratch, Run B restored)."
    echo "      Only steps 10-19 should match if restore is deterministic."
    exit 1
fi
