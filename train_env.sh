#!/bin/bash

# Training environment configuration.
# Edit this file to tune XLA, NCCL, ROCm, and other runtime settings.
#
# Sourced by _train.sh before launching training.
# Per-run overrides: pass _env_KEY=VALUE after -- in submit.sh.

# NOTE: the entire build logic is commented out
#       to use the Docker image's default XLA_FLAGS!
: <<'BLOCK_COMMENT_TO_USE_DOCKER_IMAGE_DEFAULT_XLA_FLAGS'
# ---- Build XLA_FLAGS safely with clear structure ----
XLA_FLAGS=""

# === Core compiler and dump options ===
XLA_FLAGS+=" --xla_gpu_enable_cublaslt=true"
XLA_FLAGS+=" --xla_gpu_graph_level=0"
XLA_FLAGS+=" --xla_gpu_autotune_level=0"
# === GEMM and codegen behavior ===
XLA_FLAGS+=" --xla_gpu_enable_triton_gemm=false"
XLA_FLAGS+=" --xla_gpu_triton_gemm_any=false"
XLA_FLAGS+=" --xla_gpu_enable_command_buffer=''"   # Leave empty to disable explicit command buffer use
# === Collective combination / decomposition ===
XLA_FLAGS+=" --xla_gpu_enable_all_gather_combine_by_dim=false"
#XLA_FLAGS+=" --xla_gpu_enable_reduce_scatter_combine_by_dim=false"
#XLA_FLAGS+=" --xla_gpu_all_gather_combine_threshold_bytes=8589934592"   # Fix OOM for llama3.1-405b (dcn_fsdp=8, ici_fsdp=8)
#XLA_FLAGS+=" --xla_gpu_all_reduce_combine_threshold_bytes=1073741824"
#XLA_FLAGS+=" --xla_gpu_collective_permute_decomposer_threshold=1073741824"
#XLA_FLAGS+=" --xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824"
# === Overlapping and pipelining ===
#XLA_FLAGS+=" --xla_gpu_enable_highest_priority_async_stream=true"
XLA_FLAGS+=" --xla_gpu_enable_latency_hiding_scheduler=true"
XLA_FLAGS+=" --xla_gpu_enable_pipelined_all_gather=true"
XLA_FLAGS+=" --xla_gpu_enable_pipelined_all_reduce=true"
#XLA_FLAGS+=" --xla_gpu_enable_pipelined_p2p=true"
XLA_FLAGS+=" --xla_gpu_enable_pipelined_reduce_scatter=true"
#XLA_FLAGS+=" --xla_gpu_enable_while_loop_double_buffering=true"  # May cause OOM for llama3.1-405b (dcn_fsdp=8, ici_fsdp=8) even setting --xla_gpu_all_gather_combine_threshold_bytes=8589934592
#XLA_FLAGS+=" --xla_gpu_experimental_parallel_collective_overlap_limit=2"  # May conflict with latency-hiding scheduler (LHS=true)
# === Misc. ===
#XLA_FLAGS+=" --xla_gpu_unsupported_use_all_reduce_one_shot_kernel=true"

# ---- Finalize and export XLA_FLAGS ----
export XLA_FLAGS
BLOCK_COMMENT_TO_USE_DOCKER_IMAGE_DEFAULT_XLA_FLAGS

# ---- XLA dump (enable via _env_ENABLE_XLA_DUMP=1 in PASSTHROUGH_ARGS) ----
# Scope dump to global rank 0 only to avoid concurrent writers racing on the
# same xla_dump/ directory (all SPMD ranks compile identical HLO — rank 0's
# dump is representative and keeps xla_dump/ a flat dir for downstream tools
# like analyze_job.py and IRLens).
ENABLE_XLA_DUMP="${ENABLE_XLA_DUMP:-${EXTRACTED_ENV_MAP[ENABLE_XLA_DUMP]:-0}}"
if [[ "${ENABLE_XLA_DUMP,,}" =~ ^(1|y|yes|true)$ ]]; then
    _dump_rank="${GLOBAL_RANK:-${NODE_RANK:-0}}"
    if [[ "$_dump_rank" == "0" ]]; then
        echo "[XLA dump] Enabled on rank 0 (ENABLE_XLA_DUMP=$ENABLE_XLA_DUMP)"
        XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_dump_hlo_as_text"
        XLA_FLAGS="$XLA_FLAGS --xla_dump_hlo_module_re=^jit_train_step$"
        XLA_FLAGS="$XLA_FLAGS --xla_dump_hlo_pipeline_re='(?i)gpu'"
        XLA_FLAGS="$XLA_FLAGS --xla_dump_to=${OUTPUT_PATH}/xla_dump"
        export XLA_FLAGS
        echo "[XLA dump] XLA_FLAGS=$XLA_FLAGS"
    fi
    unset _dump_rank
fi

# ---- Disable XLA's in-process one-shot ragged-all-to-all kernel (default OFF) ----
# Controls --xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel.
# Default 0 (kernel disabled) → we append --...=false, so XLA's ragged thunk falls
# back to its kNccl lowering — the same runtime path 1-GPU/proc gets automatically.
# For sparse MoE (sparse_matmul=true use_turbo_grouped_gemm=true) on 1-node/proc
# this is a ~3x TGS speedup at equal HBM budget; verified no-op on dense configs,
# on sparse-gmm-deepep, and on 1-GPU/proc.
# Set _env_ENABLE_RAGGED_ONESHOT_KERNEL=1 to restore XLA's one-shot kernel (debug only).
# Appends to XLA_FLAGS so the image's default tuning flags are preserved.
ENABLE_RAGGED_ONESHOT_KERNEL="${ENABLE_RAGGED_ONESHOT_KERNEL:-${EXTRACTED_ENV_MAP[ENABLE_RAGGED_ONESHOT_KERNEL]:-0}}"
if [[ "${ENABLE_RAGGED_ONESHOT_KERNEL,,}" =~ ^(0|n|no|false)$ ]]; then
    echo "[ENABLE_RAGGED_ONESHOT_KERNEL=0] Disabling --xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel"
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false"
    export XLA_FLAGS
fi

# ---- Fix for JAX-0.8.2 ----
XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer=''"
export XLA_FLAGS

# ---- Deterministic mode (enable via _env_DETERMINISTIC_MODE=1) ----
# Addresses: XLA autotuning, TE non-deterministic kernels, RCCL reduction
# order, and JAX PRNG. See also: utils/deterministic.py for the PRNG override
# (MaxText hardcodes unsafe_rbg, which must be overridden post-init).
DETERMINISTIC_MODE="${DETERMINISTIC_MODE:-${EXTRACTED_ENV_MAP[DETERMINISTIC_MODE]:-0}}"
if [[ "${DETERMINISTIC_MODE,,}" =~ ^(1|y|yes|true)$ ]]; then
    echo "[deterministic] Enabled (DETERMINISTIC_MODE=$DETERMINISTIC_MODE)"

    # XLA autotuning: ensure autotune_level=0 for reproducible kernel selection.
    # This is normally set by the Docker image and by train_env.sh's default
    # XLA_FLAGS, but we enforce it here defensively: if the user overrides
    # XLA_FLAGS (e.g. via _env_XLA_FLAGS=...), the autotune_level could be lost.
    if [[ "${XLA_FLAGS:-}" != *"xla_gpu_autotune_level"* ]]; then
        XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
        echo "[deterministic] XLA_FLAGS += --xla_gpu_autotune_level=0 (was missing; added)"
    fi
    export XLA_FLAGS
    echo "[deterministic] XLA autotune_level=0 confirmed"

    # rocBLAS + MIOpen: disable atomic reductions for deterministic GEMM/conv.
    # This is a SEPARATE code path from xla_gpu_deterministic_ops — the two flags
    # are disconnected. TF_DETERMINISTIC_OPS triggers rocblas_set_atomics_mode(
    # rocblas_atomics_not_allowed) and MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC.
    export TF_DETERMINISTIC_OPS=1
    echo "[deterministic] TF_DETERMINISTIC_OPS=1 (rocBLAS atomics disabled)"

    # hipBLASLt: filter out non-deterministic TensileLite solutions (atomic GSU,
    # StreamKAtomic). Requires the HIPBLASLT_DETERMINISTIC patch in tensile_host.cpp
    # (4-line change that calls setDeterministicMode). Without the patch, this env
    # var has no effect — hipBLASLt ignores it.
    export HIPBLASLT_DETERMINISTIC=1
    echo "[deterministic] HIPBLASLT_DETERMINISTIC=1 (requires patched hipBLASLt)"

    # TE: force deterministic fused attention kernels (fwd + bwd)
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    echo "[deterministic] NVTE_ALLOW_NONDETERMINISTIC_ALGO=0"

    # RCCL: pin collective algorithm for reproducible reduction order.
    # Use Ring for everything — safest for determinism and hardware compatibility.
    # Tree allreduce + Simple protocol can SIGSEGV on newer GPUs (gfx950/MI355X).
    # RCCL only supports per-function prefixes for NCCL_NUM_FUNCTIONS=5 entries
    # (AllGather, AllReduce, AllToAllPivot, Broadcast, Reduce — NOT ReduceScatter).
    # Disable MSCCL++ which has its own algorithm selection bypassing NCCL_ALGO.
    export NCCL_ALGO="Ring"
    export RCCL_MSCCLPP_ENABLE=0
    echo "[deterministic] NCCL_ALGO=Ring  RCCL_MSCCLPP_ENABLE=0"

    # JAX PRNG: signal deterministic.py to override MaxText's unsafe_rbg with threefry2x32
    export JAX_DEFAULT_PRNG_IMPL=threefry2x32
    echo "[deterministic] JAX_DEFAULT_PRNG_IMPL=threefry2x32 (applied post-init by deterministic.py)"

    # CK fused attention deterministic kernel:
    #   - On image rocm/jax-training:maxtext-v26.2-det-te508-aot (or any image
    #     containing TE >= 2.12.0.dev0+8943023d, i.e. ROCm/TransformerEngine PR #508):
    #     the CK fused-attention `_deterministic` kernel variants are wired through
    #     NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 (set above). No further override needed.
    #     Production batch (bs=8) works directly with ~6% throughput cost vs non-det.
    #     See harness/reports/2026-05-05_summary.md for the full validation matrix.
    #   - On any older image (pre-PR-508), the deterministic flag does NOT reach
    #     CK because TE hardcoded `false` at three call sites in fused_attn.cpp.
    #     If you must run on such an old image, opt into the legacy unfused
    #     workaround by passing _env_NVTE_FUSED_ATTN=0 explicitly. That falls
    #     back to JAX native attention (O(seq^2) memory, ~9.7x throughput loss,
    #     forces per_device_batch_size=1 to avoid OOM).
    echo "[deterministic] CK fused attention uses _deterministic variants via NVTE_ALLOW_NONDETERMINISTIC_ALGO=0"
    echo "[deterministic] (requires PR-#508 image; legacy unfused workaround = pass _env_NVTE_FUSED_ATTN=0)"
fi

export NCCL_CHECKS_DISABLE=1
export NCCL_DEBUG=WARN
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1  # For debugging if needed
export TF_CPP_MIN_LOG_LEVEL=2

# ---- Memory fraction ----
export XLA_PYTHON_CLIENT_MEM_FRACTION=.93

export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=512

# ---- Multi-rail network optimization ----
#export NCCL_CROSS_NIC=2  # For multi-rail networks
export NCCL_NCHANNELS_PER_NET_PEER=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=8

# ---- InfiniBand tuning ----
export NCCL_IB_QPS_PER_CONNECTION=4
#export NCCL_IB_RETRY_CNT=7
#export NCCL_IB_TIMEOUT=23

# ---- Auto-detected NCCL network settings (IB HCA, QoS, socket interface) ----
_TRAIN_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_TRAIN_ENV_DIR/utils/detect_nccl_env.sh"
unset _TRAIN_ENV_DIR

# ---- Protocol and algorithm selection ----
#export NCCL_ALGO=Ring,Tree  # Hybrid algorithm selection
#export NCCL_PROTO=Simple  # Better for large messages in MoE

# ---- Buffer management ----
#export NCCL_BUFFSIZE=8388608  # 8MB buffers
# Larger buffer sizes for massive models (e.g. 300B+ parameters)
#export NCCL_BUFFSIZE=16777216  # 16MB

# ---- GPU compute settings ----
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPU_MAX_HW_QUEUES=2

# ---- AMD-specific optimizations ----
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_IPC_MODE_LEGACY=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_NO_SCRATCH_RECLAIM=1

# ---- Transformer Engine optimizations ----
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_FRAMEWORK=jax
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1}
export NVTE_FUSED_ATTN_AOTRITON=0
export NVTE_FUSED_ATTN_CK=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_ROCM=1

# ---- Composable Kernel optimizations ----
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=2
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}
export NVTE_CK_HOW_V3_BF16_CVT=2
# Forces FP32 precision for atomic accumulation in CK V3 GEMM output writes.
# Critical for MoE convergence: BF16 atomics (=0) cause visibly slower loss
# descent vs FP32 atomics (=1) due to accumulated rounding errors across many
# experts and layers. Use default value from the docker image (likely =1).
#export NVTE_CK_IS_V3_ATOMIC_FP32=1

# ---- Compilation cache settings ----
#export JAX_COMPILATION_CACHE_DIR="$OUTPUT_PATH/../jax_cache"
#export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

# ---- PGLE (Profile-Guided Layout Optimization) - uncomment after first run ----
#export JAX_ENABLE_PGLE=true
#export JAX_PGLE_AGGREGATION_PERCENTILE=90
#export JAX_PGLE_PROFILING_RUNS=5

export IONIC_LOCKFREE=all

# DMABUF default: enabled for performance, with runtime safety fallback below.
# If /boot kernel metadata is unavailable in the container, this file
# automatically forces NCCL_DMABUF_ENABLE=0 to avoid known SIGSEGV cases.
export NCCL_DMABUF_ENABLE=1
# Safety guard for direct sourcing and non-container launch paths.
if [[ "${NCCL_DMABUF_ENABLE:-}" == "1" ]]; then
    _kernel_release="$(uname -r 2>/dev/null || true)"
    _has_boot_kernel_metadata=false
    if [[ -n "$_kernel_release" && -d /boot ]] && compgen -G "/boot/*${_kernel_release}*" >/dev/null; then
        _has_boot_kernel_metadata=true
    fi
    if [[ "$_has_boot_kernel_metadata" != "true" ]]; then
        echo "[WARN] NCCL_DMABUF_ENABLE=1 but /boot lacks host kernel metadata for kernel '$_kernel_release'."
        echo "[WARN] Forcing NCCL_DMABUF_ENABLE=0 (mount /boot read-only to keep DMABUF enabled)."
        export NCCL_DMABUF_ENABLE=0
    fi
fi

export NCCL_GDRCOPY_ENABLE=1
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_IB_ECE_ENABLE=0
# NOTE: NCCL_IB_TC and NCCL_IB_FIFO_TC are auto-detected above (see utils/detect_nccl_env.sh).
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_USE_INLINE=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_PXN_DISABLE=0
export NET_OPTIONAL_RECV_COMPLETION=1
export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
export RCCL_LL128_FORCE_ENABLE=${RCCL_LL128_FORCE_ENABLE:-1}
export RCCL_MSCCLPP_ENABLE=${RCCL_MSCCLPP_ENABLE:-1}

#export HSA_DISABLE_CACHE=1
#export IB_PCI_RELAXED_ORDERING=1
#export NCCL_IB_QPS=2
#export NCCL_IB_SL=0
#export NCCL_IB_SPLIT_DATA_ON_QPS=0
#export NCCL_NET_GDR_LEVEL=3
#export NCCL_OOB_NET_IFNAME=enp81s0f1.2026
#export NCCL_TOPO_DUMP_FILE=/tmp/system_run2.txt
#export UCX_LOG_LEVEL=INFO
