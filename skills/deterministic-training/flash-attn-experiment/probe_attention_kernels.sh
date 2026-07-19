#!/bin/bash
# Probe MaxText `attention=` config space for a deterministic AND fast kernel.
#
# The current deterministic-mode workaround forces NVTE_FUSED_ATTN=0, which
# falls back to TE's unfused attention path: O(seq^2) memory, batch=8 OOMs,
# forced batch reduction to 1, ~9.7x throughput penalty.
#
# MaxText already exposes `attention=<value>` as a config knob (every config
# in configs/*.gpu.yml sets `attention: "cudnn_flash_te"`). Other values may
# already be wired into MaxText's attention path on AMD/ROCm. If any of them
# is BIT-EXACT at production batch_size=8, the 9.7x penalty disappears
# without any C++ / FFI / monkey-patch / TE rebuild.
#
# This script tests the candidates side-by-side. Total runtime ~80 minutes
# single-node llama2-70b on 8x gfx950.
#
# Usage:
#   bash probe_attention_kernels.sh                   # run the full matrix
#   MODEL=llama2-7b bash probe_attention_kernels.sh   # smaller model
#   STEPS=10 bash probe_attention_kernels.sh          # faster (less warmup margin)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$SKILL_DIR/../.." && pwd)"
COMPARE="$SKILL_DIR/compare_runs.py"
RESULTS_FILE="$SCRIPT_DIR/results.md"
OUTPUTS_DIR="$REPO_DIR/outputs"

MODEL="${MODEL:-llama2-70b}"
STEPS="${STEPS:-15}"

newest_output_dir() {
    ls -dt "$OUTPUTS_DIR"/local_* 2>/dev/null | grep -v '\.log$' | head -1
}

# Median TFLOP/s/device from steps >= 1 (skip step 0 warmup).
extract_tflops() {
    local run_dir="$1"
    local log="$run_dir/log"
    [[ -e "$log" ]] || { echo "N/A"; return; }
    local values
    values=$(awk '
        match($0, /completed step:[[:space:]]*([0-9]+)/, st) &&
        match($0, /TFLOP\/s\/device:[[:space:]]*([0-9.]+)/, tf) {
            if (st[1]+0 >= 1) print tf[1]
        }' "$log" | sort -n)
    [[ -z "$values" ]] && { echo "N/A"; return; }
    local n
    n=$(echo "$values" | wc -l)
    echo "$values" | awk -v n="$n" 'NR == int((n+1)/2) { printf "%.0f\n", $1 }'
}

# Fingerprint a failure mode from the tail of a run log (best-effort).
classify_failure() {
    local out="$1"
    if echo "$out" | grep -qiE 'out of memory|RESOURCE_EXHAUSTED|HBM' ; then
        echo "OOM"
    elif echo "$out" | grep -qiE 'unknown attention|invalid attention|attention type' ; then
        echo "UNSUPPORTED-KERNEL"
    elif echo "$out" | grep -qiE 'NotImplementedError|not implemented' ; then
        echo "NOT-IMPLEMENTED"
    elif echo "$out" | grep -qiE 'Traceback|Error:|Exception:' ; then
        echo "ERROR"
    else
        echo "FAILED"
    fi
}

# run_pair NAME BATCH KERNEL DET
#   NAME    test row label
#   BATCH   per_device_batch_size
#   KERNEL  value for MaxText `attention=` (empty = default in YAML)
#   DET     yes|no — enable DETERMINISTIC_MODE and defensive flags
run_pair() {
    local name="$1" batch="$2" kernel="$3" det="$4"

    echo ""
    echo "================================================================="
    echo "TEST: $name"
    echo "  attention=${kernel:-<config-default>}  batch=$batch  det=$det"
    echo "================================================================="

    local args="steps=$STEPS enable_dropout=False per_device_batch_size=$batch"
    if [[ "$det" == "yes" ]]; then
        # Activate all defensive flags via DETERMINISTIC_MODE, then override
        # NVTE_FUSED_ATTN=1 so TE-side fused attention isn't disabled — the
        # `attention=$kernel` selector decides whether TE is even invoked.
        args="_env_DETERMINISTIC_MODE=1 _env_NVTE_FUSED_ATTN=1 $args"
    fi
    [[ -n "$kernel" ]] && args="$args attention=$kernel"

    rm -rf ~/jax_cache

    echo "[$(date +%H:%M:%S)] Run 1..."
    local out1; out1=$(./in_container_run.sh "$MODEL" -- $args 2>&1)
    if ! echo "$out1" | grep -q 'completed step:'; then
        local why; why=$(classify_failure "$out1")
        echo "RESULT: $why (run 1 produced no training steps)"
        echo "$out1" | tail -5
        echo "| $name | $why | $batch | — | ${kernel:-<default>} |" >> "$RESULTS_FILE"
        return
    fi
    local dir1; dir1=$(newest_output_dir)
    local tflops1; tflops1=$(extract_tflops "$dir1")
    echo "[$(date +%H:%M:%S)] Run 1 done: $tflops1 TFLOP/s/device  ($(basename "$dir1"))"

    echo "[$(date +%H:%M:%S)] Run 2..."
    local out2; out2=$(./in_container_run.sh "$MODEL" -- $args 2>&1)
    if ! echo "$out2" | grep -q 'completed step:'; then
        local why; why=$(classify_failure "$out2")
        echo "| $name | $why | $batch | $tflops1 | ${kernel:-<default>} |" >> "$RESULTS_FILE"
        return
    fi
    local dir2; dir2=$(newest_output_dir)
    if [[ -z "$dir1" || -z "$dir2" || "$dir1" == "$dir2" ]]; then
        echo "| $name | DIR-ERROR | $batch | $tflops1 | ${kernel:-<default>} |" >> "$RESULTS_FILE"
        return
    fi

    local cmp; cmp=$(python3 "$COMPARE" "$dir1" "$dir2" 2>&1 | grep 'RESULT:' || echo 'UNKNOWN')
    local verdict
    if   echo "$cmp" | grep -q 'BIT-EXACT';        then verdict='BIT-EXACT'
    elif echo "$cmp" | grep -qE 'MISMATCH|DIFFER'; then verdict='DIFFER'
    else                                                verdict='UNKNOWN'
    fi
    echo "RESULT: $verdict  TFLOP/s/device=${tflops1:-N/A}  batch=$batch"
    echo "| $name | $verdict | $batch | ${tflops1:-N/A} | ${kernel:-<default>} |" >> "$RESULTS_FILE"
}

# --- Header ---
cat > "$RESULTS_FILE" << EOF
# Attention-Kernel Probe — Results

Searches MaxText's \`attention=\` config space for alternatives to
\`NVTE_FUSED_ATTN=0\` that are simultaneously deterministic AND fast.

A WIN is BIT-EXACT at production batch=8 with TFLOP/s/device close to the
non-deterministic baseline (~968 on llama2-70b/MI355X). Such a kernel can
replace the current 9.7x-penalty workaround entirely.

| Test | Determinism | Batch | TFLOP/s/dev | attention= |
|------|-------------|-------|-------------|------------|
EOF

cd "$REPO_DIR"
echo "Probe started at $(date)"
echo "Model: $MODEL  Steps: $STEPS  Output: $RESULTS_FILE"

# --- Controls -------------------------------------------------------------
# C0: known bit-exact baseline. Validates compare_runs.py and the harness.
#     attention=<config-default>=cudnn_flash_te, with NVTE_FUSED_ATTN=0
#     (so override _env_NVTE_FUSED_ATTN=1 is NOT applied — use det=no
#     and rely on the literal NVTE_FUSED_ATTN=0 path via DETERMINISTIC_MODE).
echo ""
echo "[$(date +%H:%M:%S)] === Controls ==="
# Reproduce existing baseline by NOT passing _env_NVTE_FUSED_ATTN=1 — we
# emulate this with a custom args string instead of run_pair's det=yes branch.
{
    name="C0-current-workaround-NVTE_FUSED_ATTN=0"
    args="_env_DETERMINISTIC_MODE=1 steps=$STEPS enable_dropout=False per_device_batch_size=1"
    echo ""
    echo "================================================================="
    echo "TEST: $name"
    echo "  (current production deterministic path — reproduces 100 TFLOP/s baseline)"
    echo "================================================================="
    rm -rf ~/jax_cache
    out1=$(./in_container_run.sh "$MODEL" -- $args 2>&1)
    if ! echo "$out1" | grep -q 'completed step:'; then
        echo "| $name | $(classify_failure "$out1") | 1 | — | <default> |" >> "$RESULTS_FILE"
    else
        dir1=$(newest_output_dir); tflops1=$(extract_tflops "$dir1")
        out2=$(./in_container_run.sh "$MODEL" -- $args 2>&1)
        dir2=$(newest_output_dir)
        cmp=$(python3 "$COMPARE" "$dir1" "$dir2" 2>&1 | grep 'RESULT:' || echo 'UNKNOWN')
        verdict='UNKNOWN'
        echo "$cmp" | grep -q 'BIT-EXACT'        && verdict='BIT-EXACT'
        echo "$cmp" | grep -qE 'MISMATCH|DIFFER' && verdict='DIFFER'
        echo "RESULT: $verdict  TFLOP/s/device=${tflops1:-N/A}  batch=1"
        echo "| $name | $verdict | 1 | ${tflops1:-N/A} | <default> |" >> "$RESULTS_FILE"
    fi
}

# C1: known non-deterministic fast baseline. Validates DIFFER detection.
run_pair "C1-baseline-non-det-fast"  8  ""                "no"

# --- Candidates -----------------------------------------------------------
echo ""
echo "[$(date +%H:%M:%S)] === Candidate kernels ==="

# K1: MaxText pure-JAX dot-product attention. Pure JAX, no TE, no Pallas.
#     Expected deterministic; expected slow (similar to current workaround
#     but via a different code path so likely a different checksum).
run_pair "K1-dot_product-bs1"        1  "dot_product"      "yes"

# K2: jax.nn.dot_product_attention via XLA — on AMD this dispatches to
#     whatever fused-attention path XLA can find. May or may not be
#     deterministic depending on XLA's GPU lowering on ROCm.
run_pair "K2-cudnn_flash_jax"        8  "cudnn_flash_jax"  "yes"

# K3: Pallas/Splash flash attention. The most interesting candidate: this
#     is a JAX-native fused attention kernel. If it works on gfx950 and is
#     deterministic, this is the WIN.
run_pair "K3-flash-pallas"           8  "flash"            "yes"

# K4: MaxText multi-head-attention module — JAX-native, may be unfused.
run_pair "K4-mha"                    8  "mha"              "yes"

# K5: same as K3 but with batch=1, so we can isolate "the kernel works" from
#     "the kernel works at production batch". Only run if K3 OOMs.
# (Conditional — the user can re-run manually if needed.)

# --- Footer ---
{
    echo ""
    echo "## Legend"
    echo ""
    echo "- **BIT-EXACT** — 2 runs produced byte-identical loss curves at the precision \`compare_runs.py\` checks."
    echo "- **DIFFER** — runs diverged; kernel is non-deterministic on this stack."
    echo "- **OOM** — \`out of memory\` / \`RESOURCE_EXHAUSTED\` in run output. Try smaller batch."
    echo "- **UNSUPPORTED-KERNEL** — MaxText rejected this \`attention=\` value (kernel not present in this fork/version)."
    echo "- **NOT-IMPLEMENTED** — kernel exists but a code path raised \`NotImplementedError\` (often: ROCm-side gap)."
    echo "- **ERROR / FAILED** — generic failure, see per-run log under \`outputs/local_*\`."
    echo ""
    echo "Generated: $(date)"
    echo "Host: $(hostname)"
    echo "Model: $MODEL  Steps: $STEPS"
} >> "$RESULTS_FILE"

echo ""
echo "================================================================="
echo "PROBE COMPLETE"
echo "================================================================="
cat "$RESULTS_FILE"
