# Deterministic training container: base image + patched hipBLASLt
#
# Build:
#   docker build -t rocm/jax-training:maxtext-v26.2-deterministic \
#     -f skills/deterministic-training/Dockerfile .
#
# The build takes ~30-60 min (hipBLASLt compilation for gfx942+gfx950).
# Requires the base image to be available (pull or load from tar first).

ARG BASE_IMAGE=rocm/jax-training:maxtext-v26.2
FROM ${BASE_IMAGE}

ARG GPU_TARGETS="gfx942;gfx950"

COPY skills/deterministic-training/hipblaslt-deterministic.patch /tmp/hipblaslt-deterministic.patch

RUN set -eux && \
    cd /workspace && \
    if [ ! -d rocm-libraries/projects/hipblaslt/library ]; then \
        git clone --depth 1 --filter=blob:none --sparse \
            https://github.com/ROCm/rocm-libraries.git && \
        cd rocm-libraries && \
        git sparse-checkout set projects/hipblaslt shared/rocroller shared/origami shared/mxdatagenerator; \
    else \
        cd rocm-libraries && \
        git sparse-checkout add shared/rocroller shared/origami shared/mxdatagenerator 2>/dev/null || true; \
    fi && \
    cd projects/hipblaslt && \
    TARGET_FILE=library/src/amd_detail/rocblaslt/src/tensile_host.cpp && \
    if ! grep -q "HIPBLASLT_DETERMINISTIC" "$TARGET_FILE" 2>/dev/null; then \
        patch -p1 < /tmp/hipblaslt-deterministic.patch; \
    fi && \
    grep -n "setDeterministicMode" "$TARGET_FILE" && \
    ROCM_PATH=$(readlink -f /opt/rocm) && \
    cmake \
        -DGPU_TARGETS="${GPU_TARGETS}" \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH}/lib/llvm;${ROCM_PATH}" \
        -DCMAKE_INSTALL_PREFIX="${ROCM_PATH}" \
        -DCMAKE_PACKAGING_INSTALL_PREFIX="${ROCM_PATH}" \
        -DROCM_PATH="${ROCM_PATH}" \
        -DCMAKE_C_COMPILER="${ROCM_PATH}/llvm/bin/clang" \
        -DCMAKE_CXX_COMPILER="${ROCM_PATH}/llvm/bin/clang++" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHIPBLASLT_ENABLE_CLIENT=OFF \
        -B build -S . && \
    TENSILE_CPU_THREADS=$(nproc) MAX_JOBS=$(nproc) OMP_NUM_THREADS=1 \
        cmake --build build --target package --parallel $(nproc) && \
    dpkg -i --force-overwrite build/hipblaslt_*.deb build/hipblaslt-dev_*.deb && \
    dpkg -s hipblaslt | grep Version && \
    rm -rf build /tmp/hipblaslt-deterministic.patch && \
    python3 -c "import ctypes; ctypes.CDLL('${ROCM_PATH}/lib/libhipblaslt.so'); print('hipBLASLt OK')"
