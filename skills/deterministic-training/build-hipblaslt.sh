#!/bin/bash
# Build hipBLASLt with the deterministic mode patch.
# Run as: bash build-hipblaslt.sh
# Or copy-paste each step block into a terminal.

# ── Step 1: Clone source ────────────────────────────────────────────────────

cd /workspace
if [[ ! -d rocm-libraries/projects/hipblaslt/library ]]; then
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/ROCm/rocm-libraries.git
    cd rocm-libraries
    git sparse-checkout set projects/hipblaslt shared/rocroller shared/origami shared/mxdatagenerator
else
    echo "Source already present, ensuring shared deps are checked out..."
    cd rocm-libraries
    git sparse-checkout add shared/rocroller shared/origami shared/mxdatagenerator 2>/dev/null || true
fi
cd projects/hipblaslt

# ── Step 2: Apply patch ─────────────────────────────────────────────────────

PATCH_FILE=/maxtext-slurm/skills/deterministic-training/hipblaslt-deterministic.patch
TARGET_FILE=library/src/amd_detail/rocblaslt/src/tensile_host.cpp

if grep -q "HIPBLASLT_DETERMINISTIC" "$TARGET_FILE" 2>/dev/null; then
    echo "Patch already applied"
else
    patch -p1 < "$PATCH_FILE"
fi
grep -n "setDeterministicMode" "$TARGET_FILE"

# ── Step 3: Configure ───────────────────────────────────────────────────────

export ROCM_PATH="${ROCM_PATH:-$(readlink -f /opt/rocm)}"
export HIP_PATH="$ROCM_PATH"

# Auto-detect GPU architecture (fall back to gfx942;gfx950 for broad compatibility)
if command -v rocminfo &>/dev/null; then
    GPU_TARGETS="$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | sort -u | paste -sd';')"
fi
GPU_TARGETS="${GPU_TARGETS:-gfx942;gfx950}"
GPU_TARGETS="gfx942;gfx950"
echo "Building for GPU_TARGETS=$GPU_TARGETS"

rm -rf build

cmake \
    -DGPU_TARGETS="$GPU_TARGETS" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}/lib/llvm;${ROCM_PATH}" \
    -DCMAKE_INSTALL_PREFIX="${ROCM_PATH}" \
    -DCMAKE_PACKAGING_INSTALL_PREFIX="${ROCM_PATH}" \
    -DROCM_PATH="${ROCM_PATH}" \
    -DCMAKE_C_COMPILER="${ROCM_PATH}/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="${ROCM_PATH}/llvm/bin/clang++" \
    -DCMAKE_BUILD_TYPE=Release \
    -DHIPBLASLT_ENABLE_CLIENT=OFF \
    -B build -S .

# ── Step 4: Build ───────────────────────────────────────────────────────────

export TENSILE_CPU_THREADS=$(nproc)
export MAX_JOBS=$(nproc)
export OMP_NUM_THREADS=1

cmake --build build --target package --parallel $(nproc)

# ── Step 5: Install ─────────────────────────────────────────────────────────

dpkg -i --force-overwrite build/hipblaslt_*.deb build/hipblaslt-dev_*.deb

# ── Step 6: Verify ──────────────────────────────────────────────────────────

dpkg -s hipblaslt | grep Version
python3 -c "import ctypes; ctypes.CDLL('/opt/rocm/lib/libhipblaslt.so'); print('OK')"
echo "Done. Use: export HIPBLASLT_DETERMINISTIC=1"
