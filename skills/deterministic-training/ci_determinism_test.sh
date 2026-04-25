#!/bin/bash
# CI smoke test: verify DETERMINISTIC_MODE=1 produces bit-exact results.
# Exit 0 if bit-exact, exit 1 if non-deterministic or error.
#
# Usage:
#   bash ci_determinism_test.sh                    # default: llama2-70b, 5 steps
#   bash ci_determinism_test.sh --steps 15         # more steps
#   bash ci_determinism_test.sh --model llama2-7b  # different model
#
# Requires: 8 GPUs, ~12 min for default config.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPARE="$SCRIPT_DIR/compare_runs.py"

STEPS=5
MODEL="llama2-70b"
BATCH_SIZE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

COMMON_ARGS="_env_DETERMINISTIC_MODE=1 steps=$STEPS enable_dropout=False per_device_batch_size=$BATCH_SIZE"
OUTPUTS_DIR="$REPO_DIR/outputs"

newest_output_dir() {
    ls -dt "$OUTPUTS_DIR"/local_* 2>/dev/null | grep -v '\.log$' | head -1
}

echo "=== Determinism CI Test ==="
echo "Model: $MODEL | Steps: $STEPS | Batch: $BATCH_SIZE"
echo ""

# Clear compilation cache
rm -rf ~/jax_cache

echo "[1/3] Running first training..."
cd "$REPO_DIR"
if ! ./in_container_run.sh "$MODEL" -- $COMMON_ARGS 2>&1 | tail -3; then
    echo "FAIL: First run crashed"
    exit 1
fi
DIR_A=$(newest_output_dir)

echo ""
echo "[2/3] Running second training..."
if ! ./in_container_run.sh "$MODEL" -- $COMMON_ARGS 2>&1 | tail -3; then
    echo "FAIL: Second run crashed"
    exit 1
fi
DIR_B=$(newest_output_dir)

if [[ -z "$DIR_A" || -z "$DIR_B" || "$DIR_A" == "$DIR_B" ]]; then
    echo "FAIL: Could not find two distinct output directories"
    echo "  DIR_A=$DIR_A"
    echo "  DIR_B=$DIR_B"
    exit 1
fi

echo ""
echo "[3/3] Comparing at full float32 precision..."
echo "  A: $(basename "$DIR_A")"
echo "  B: $(basename "$DIR_B")"
echo ""

RESULT=$(python3 "$COMPARE" "$DIR_A" "$DIR_B" 2>&1)
echo "$RESULT" | grep -E 'step|EXACT|DIFFER|RESULT|^-'

if echo "$RESULT" | grep -q 'BIT-EXACT'; then
    echo ""
    echo "PASS: Deterministic mode verified — all steps bit-exact"
    exit 0
else
    echo ""
    echo "FAIL: Non-determinism detected"
    exit 1
fi
