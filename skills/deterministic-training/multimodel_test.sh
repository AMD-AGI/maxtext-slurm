#!/bin/bash
# Phase 2a: Multi-model determinism test.
# Tests whether the defensive flags become necessary for different models.
#
# Usage:
#   bash multimodel_test.sh              # run all models
#   bash multimodel_test.sh llama2-7b    # run one model
#
# Requires: 8 GPUs for llama2-70b/7b, multi-node for llama3.1-405b

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPARE="$SCRIPT_DIR/compare_runs.py"
RESULTS_FILE="$SCRIPT_DIR/multimodel_results.md"
OUTPUTS_DIR="$REPO_DIR/outputs"

STEPS=15

newest_output_dir() {
    ls -dt "$OUTPUTS_DIR"/local_* 2>/dev/null | grep -v '\.log$' | head -1
}

run_pair() {
    local test_name="$1"
    local model="$2"
    shift 2
    local extra_args="$*"

    echo ""
    echo "================================================================="
    echo "TEST: $test_name (model=$model)"
    echo "ARGS: $extra_args"
    echo "================================================================="

    rm -rf ~/jax_cache

    echo "[$(date +%H:%M:%S)] Run 1..."
    if ! ./in_container_run.sh "$model" -- $extra_args 2>&1 | tail -3; then
        echo "| $test_name | $model | ERROR | run 1 failed |" >> "$RESULTS_FILE"
        return
    fi
    local dir1; dir1=$(newest_output_dir)

    echo "[$(date +%H:%M:%S)] Run 2..."
    if ! ./in_container_run.sh "$model" -- $extra_args 2>&1 | tail -3; then
        echo "| $test_name | $model | ERROR | run 2 failed |" >> "$RESULTS_FILE"
        return
    fi
    local dir2; dir2=$(newest_output_dir)

    if [[ -z "$dir1" || -z "$dir2" || "$dir1" == "$dir2" ]]; then
        echo "| $test_name | $model | ERROR | dirs not found |" >> "$RESULTS_FILE"
        return
    fi

    local result_line
    result_line=$(python3 "$COMPARE" "$dir1" "$dir2" 2>&1 | grep 'RESULT:' || echo "RESULT: UNKNOWN")

    if echo "$result_line" | grep -q 'BIT-EXACT'; then
        echo "| $test_name | $model | BIT-EXACT | |" >> "$RESULTS_FILE"
    else
        local first_diff
        first_diff=$(python3 "$COMPARE" "$dir1" "$dir2" 2>&1 | grep 'DIFFER' | head -1 | awk '{print "step "$1" delta "$NF}')
        echo "| $test_name | $model | DIFFER | $first_diff |" >> "$RESULTS_FILE"
    fi
}

cd "$REPO_DIR"

MODELS="${1:-all}"

cat > "$RESULTS_FILE" << 'EOF'
# Multi-Model Determinism Test Results

| Test | Model | Result | Detail |
|------|-------|--------|--------|
EOF

if [[ "$MODELS" == "all" || "$MODELS" == "llama2-7b" ]]; then
    # llama2-7b: smaller model, tests compute-only determinism
    # Uses ici_tensor_parallelism=8 since it fits differently than 70b
    run_pair "full-det-mode" "llama2-7b" \
        "_env_DETERMINISTIC_MODE=1 steps=$STEPS enable_dropout=False per_device_batch_size=1"

    run_pair "only-fused-attn-off" "llama2-7b" \
        "_env_NVTE_FUSED_ATTN=0 steps=$STEPS enable_dropout=False per_device_batch_size=1"
fi

if [[ "$MODELS" == "all" || "$MODELS" == "llama2-70b" ]]; then
    # llama2-70b: baseline model (already tested, included for completeness)
    run_pair "full-det-mode" "llama2-70b" \
        "_env_DETERMINISTIC_MODE=1 steps=$STEPS enable_dropout=False per_device_batch_size=1"
fi

echo ""
echo "================================================================="
echo "MULTI-MODEL TEST COMPLETE"
echo "================================================================="
{
    echo ""
    echo "Generated: $(date)"
    echo "Host: $(hostname)"
} >> "$RESULTS_FILE"
cat "$RESULTS_FILE"
