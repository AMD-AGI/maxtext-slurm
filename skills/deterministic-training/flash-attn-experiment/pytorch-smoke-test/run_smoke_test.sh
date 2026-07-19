#!/bin/bash
# Run the ROCm/flash-attention deterministic-mode smoke test in the same
# sort of container `run_local.sh` uses for MaxText, but with a PyTorch
# image (since flash-attn is a PyTorch extension). This is a STRICT
# PREREQUISITE for any MaxText/JAX integration of ROCm/flash-attention:
# if the kernel itself isn't bit-exact in deterministic mode on our GPU,
# no amount of JAX FFI work will fix it.
#
# Mirrors run_local.sh's launch pattern:
#   - resolves SCRIPT_DIR
#   - sources container_env.sh for DOCKER_REGISTRY etc.
#   - launches a container with /dev/kfd and /dev/dri mounted
#   - runs the test inside, streams logs to stdout
#
# It does NOT touch MaxText, the JAX training image, or any project state.
# Output dir is its own ($SCRIPT_DIR/runs/<timestamp>/).
#
# Usage:
#   bash run_smoke_test.sh                             # default config (llama2-70b shapes)
#   FA_IMAGE=rocm/pytorch:latest bash run_smoke_test.sh
#   FA_BATCH=8 FA_SEQ=8192 bash run_smoke_test.sh      # bigger config

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# ----------------------------------------------------------------------
# Container image. Default to rocm/pytorch:latest because flash-attn is
# a PyTorch extension and that image has the ROCm + PyTorch + build tools
# combo that flash-attn's setup.py expects (per ROCm/flash-attention's
# README: "We recommend the Pytorch container from ROCm").
#
# Override via FA_IMAGE if you have a pre-baked image with flash-attn
# already installed (recommended for repeated runs — avoids the ~30 min
# first-time install).
# ----------------------------------------------------------------------
FA_IMAGE="${FA_IMAGE:-rocm/pytorch:latest}"

# ----------------------------------------------------------------------
# Optional: load registry credentials the same way run_local.sh would.
# ----------------------------------------------------------------------
if [[ -f "$REPO_DIR/container_env.local.sh" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/container_env.local.sh"
fi

# ----------------------------------------------------------------------
# Per-run output dir (timestamped, sibling pattern to outputs/local_*).
# ----------------------------------------------------------------------
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$SCRIPT_DIR/runs/$RUN_TS"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/log"

echo "ROCm/flash-attention smoke test"
echo "  image:    $FA_IMAGE"
echo "  workdir:  $RUN_DIR"
echo "  log:      $LOG_FILE"
echo ""

# ----------------------------------------------------------------------
# Build the in-container command. We:
#   1. install flash-attn if it isn't already importable (tries pip, fails
#      with a clear message if no wheel and no toolchain)
#   2. exec the test script
# ----------------------------------------------------------------------
read -r -d '' IN_CONTAINER_CMD << 'EOF' || true
set -e
echo "[smoke] Container: $(hostname), $(uname -a)"
echo "[smoke] Python: $(python3 --version 2>&1)"
echo "[smoke] PyTorch: $(python3 -c 'import torch; print(torch.__version__, "rocm=" + str(torch.version.hip))' 2>&1 || echo 'NOT INSTALLED')"

# Install flash-attn if missing.
if ! python3 -c 'import flash_attn' 2>/dev/null; then
    echo "[smoke] flash_attn not present — installing (this can take 5-30 min on first run)..."
    if [[ -n "$FA_WHEEL_URL" ]]; then
        pip install --no-build-isolation "$FA_WHEEL_URL" 2>&1 | tail -20
    else
        # Fall back to source install. Picks up local PyTorch + ROCm.
        pip install --no-build-isolation flash-attn 2>&1 | tail -30
    fi
fi

python3 -c 'import flash_attn; print("[smoke] flash_attn version:", flash_attn.__version__)'

cd /smoketest
exec python3 -u test_flash_attn.py
EOF

# Pass through any FA_* env vars the user has set (BATCH, SEQ, HEADS, ...)
# plus FA_WHEEL_URL if they want a specific wheel.
DOCKER_ENV_ARGS=()
for v in FA_BATCH FA_SEQ FA_HEADS FA_HEAD_DIM FA_SEED FA_BENCH_ITERS FA_WHEEL_URL; do
    if [[ -n "${!v:-}" ]]; then
        DOCKER_ENV_ARGS+=(-e "$v=${!v}")
    fi
done

# ----------------------------------------------------------------------
# Launch. Same /dev/kfd + /dev/dri + --ipc=host pattern that
# rocm/pytorch documentation (and ROCm/flash-attention's README) shows.
# ----------------------------------------------------------------------
set -x
docker run --rm \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt=seccomp=unconfined \
    --ipc=host \
    --shm-size=16G \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "$SCRIPT_DIR:/smoketest:ro" \
    -v "$RUN_DIR:/runs/current:rw" \
    -w /smoketest \
    "$FA_IMAGE" \
    bash -c "$IN_CONTAINER_CMD" 2>&1 | tee "$LOG_FILE"

set +x
echo ""
echo "Smoke-test log: $LOG_FILE"

# ----------------------------------------------------------------------
# Promote the most recent log to a stable symlink for easier follow-up.
# ----------------------------------------------------------------------
ln -sf "$RUN_TS/log" "$SCRIPT_DIR/runs/latest.log" 2>/dev/null || true

# Exit with the in-container command's status (PIPESTATUS[0] from tee).
exit "${PIPESTATUS[0]:-0}"
