#!/bin/bash
# Ablation study: test each deterministic setting individually.
# Removes one setting at a time from the bit-exact baseline to check if it's necessary.
#
# Total: 7 test pairs × 2 runs × ~8 min = ~112 min (~2 hours)
# Output: summary table showing which settings are necessary for determinism.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPARE="$SCRIPT_DIR/compare_runs.py"
RESULTS_FILE="$SCRIPT_DIR/ablation_results.md"
STEPS=15
COMMON_ARGS="steps=$STEPS enable_dropout=False per_device_batch_size=1"
OUTPUTS_DIR="$REPO_DIR/outputs"

DOCKER_XLA_FLAGS="--xla_gpu_memory_limit_slop_factor=95 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_enable_latency_hiding_scheduler=True --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=0 --xla_gpu_enable_all_gather_combine_by_dim=FALSE"

export XLA_FLAGS="$DOCKER_XLA_FLAGS"

cd "$REPO_DIR"

newest_output_dir() {
    ls -dt "$OUTPUTS_DIR"/local_* 2>/dev/null | grep -v '\.log$' | head -1
}

run_pair() {
    local test_name="$1"
    shift
    local env_args="$*"

    echo ""
    echo "================================================================="
    echo "TEST: $test_name"
    echo "ARGS: $env_args $COMMON_ARGS"
    echo "================================================================="

    rm -rf ~/jax_cache

    echo "[$(date +%H:%M:%S)] Starting run 1..."
    if ! ./in_container_run.sh llama2-70b -- $env_args $COMMON_ARGS 2>&1 | tail -3; then
        echo "| $test_name | ERROR | run 1 failed |" >> "$RESULTS_FILE"
        return
    fi
    local dir1
    dir1=$(newest_output_dir)

    echo "[$(date +%H:%M:%S)] Starting run 2..."
    if ! ./in_container_run.sh llama2-70b -- $env_args $COMMON_ARGS 2>&1 | tail -3; then
        echo "| $test_name | ERROR | run 2 failed |" >> "$RESULTS_FILE"
        return
    fi
    local dir2
    dir2=$(newest_output_dir)

    if [[ -z "$dir1" || -z "$dir2" || "$dir1" == "$dir2" ]]; then
        echo "RESULT: ERROR finding output dirs"
        echo "  dir1=$dir1"
        echo "  dir2=$dir2"
        echo "| $test_name | ERROR | could not find two distinct output dirs |" >> "$RESULTS_FILE"
        return
    fi

    echo "[$(date +%H:%M:%S)] Comparing:"
    echo "  dir1=$(basename "$dir1")"
    echo "  dir2=$(basename "$dir2")"

    local compare_out
    compare_out=$(python3 "$COMPARE" "$dir1" "$dir2" 2>&1)
    local result_line
    result_line=$(echo "$compare_out" | grep 'RESULT:' || echo "RESULT: UNKNOWN")
    echo "$result_line"

    if echo "$result_line" | grep -q 'BIT-EXACT'; then
        echo "| $test_name | BIT-EXACT | Removing this setting has no effect → NOT needed |" >> "$RESULTS_FILE"
    elif echo "$result_line" | grep -q 'MISMATCH'; then
        local first_diff
        first_diff=$(echo "$compare_out" | grep 'DIFFER' | head -1 | awk '{print "step "$1" delta "$NF}')
        echo "| $test_name | DIFFER | Removing breaks determinism → NEEDED ($first_diff) |" >> "$RESULTS_FILE"
    else
        echo "| $test_name | UNKNOWN | $result_line |" >> "$RESULTS_FILE"
    fi
}

# --- Header ---
cat > "$RESULTS_FILE" << 'EOF'
# Deterministic Mode — Ablation Study Results

Each test removes ONE setting from the proven bit-exact baseline
(`DETERMINISTIC_MODE=1` which includes `NVTE_FUSED_ATTN=0`).
If removing a setting breaks determinism → that setting is necessary.

| Test | Result | Conclusion |
|------|--------|------------|
EOF

echo "Starting ablation study at $(date)"
echo "Steps per run: $STEPS"
echo "Results: $RESULTS_FILE"

# --- Test 0: Baseline (all flags, control) ---
run_pair "0-baseline-all-flags" \
    "_env_DETERMINISTIC_MODE=1"

# --- Test 1: Remove xla_gpu_deterministic_ops ---
export XLA_FLAGS="$DOCKER_XLA_FLAGS --xla_gpu_deterministic_ops=false"
run_pair "1-no-xla_gpu_deterministic_ops" \
    "_env_DETERMINISTIC_MODE=1"
export XLA_FLAGS="$DOCKER_XLA_FLAGS"

# --- Test 2: Remove TF_DETERMINISTIC_OPS ---
run_pair "2-no-TF_DETERMINISTIC_OPS" \
    "_env_DETERMINISTIC_MODE=1" "_env_TF_DETERMINISTIC_OPS=0"

# --- Test 3: Remove HIPBLASLT_DETERMINISTIC ---
run_pair "3-no-HIPBLASLT_DETERMINISTIC" \
    "_env_DETERMINISTIC_MODE=1" "_env_HIPBLASLT_DETERMINISTIC=0"

# --- Test 4: Remove NCCL_ALGO + RCCL_MSCCLPP_ENABLE ---
run_pair "4-no-RCCL-pinning" \
    "_env_DETERMINISTIC_MODE=1" "_env_NCCL_ALGO=" "_env_RCCL_MSCCLPP_ENABLE="

# --- Test 5: Remove JAX_DEFAULT_PRNG_IMPL ---
run_pair "5-no-PRNG-override" \
    "_env_DETERMINISTIC_MODE=1" "_env_JAX_DEFAULT_PRNG_IMPL="

# --- Test 6: ONLY NVTE_FUSED_ATTN=0 (no DETERMINISTIC_MODE) ---
run_pair "6-only-NVTE_FUSED_ATTN=0" \
    "_env_NVTE_FUSED_ATTN=0"

# --- Footer ---
{
    echo ""
    echo "Generated: $(date)"
    echo "Host: $(hostname)"
    echo "Steps: $STEPS | Batch: 1 | Dropout: off | Model: llama2-70b | GPUs: 8x gfx950"
} >> "$RESULTS_FILE"

echo ""
echo "================================================================="
echo "ABLATION STUDY COMPLETE"
echo "================================================================="
cat "$RESULTS_FILE"
