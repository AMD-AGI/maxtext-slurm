# Deterministic Training — Technical Reference

Detailed per-fix documentation, code traces, and implementation notes. For the overview and experimental results, see [SKILL.md](SKILL.md).

## Implementation Overview

### Where the code lives

| File | What it does |
|------|--------------|
| `train_env.sh` | `DETERMINISTIC_MODE` block — sets 8 env vars (XLA, TE, RCCL, PRNG, CK). Later defaults use `${VAR:-default}` to avoid clobbering. |
| `utils/deterministic.py` | `apply_patches()` — monkey-patches MaxText's `initialize()` to override hardcoded `unsafe_rbg` PRNG. `verify_env()` — runtime checks for all critical flags. `LossChecksumTracker` — SHA-256 over loss values for cross-run comparison. |
| `utils/mfu_tracker.py` | Training entry point. Imports `deterministic` module and delegates: calls `deterministic.apply_patches()` before training, `deterministic.extract_loss()` during stdout interception, and `deterministic.print_loss_checksum()` after training completes. |

### How it works in `train_env.sh`

```bash
DETERMINISTIC_MODE="${DETERMINISTIC_MODE:-${EXTRACTED_ENV_MAP[DETERMINISTIC_MODE]:-0}}"
if [[ "${DETERMINISTIC_MODE,,}" =~ ^(1|y|yes|true)$ ]]; then
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_deterministic_ops=true"
    export XLA_FLAGS
    export TF_DETERMINISTIC_OPS=1                  # rocBLAS atomics + MIOpen
    export HIPBLASLT_DETERMINISTIC=1                # hipBLASLt (needs patched library)
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0       # TE fused attention algo selection
    export NCCL_ALGO="Ring"                         # Pin collectives to Ring
    export RCCL_MSCCLPP_ENABLE=0                    # Disable MSCCL++ (bypasses NCCL_ALGO)
    export JAX_DEFAULT_PRNG_IMPL=threefry2x32       # PRNG (applied post-init)
    export NVTE_FUSED_ATTN=0                        # Disable CK fused attention (sole source of non-det)
fi
```

Later defaults avoid clobbering:
```bash
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}
export RCCL_MSCCLPP_ENABLE=${RCCL_MSCCLPP_ENABLE:-1}
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1}
```

### PRNG monkey-patch in `utils/deterministic.py`

MaxText's `train.py` unconditionally sets `jax.config.update("jax_default_prng_impl", "unsafe_rbg")` inside `initialize()`, overriding any env var. The patch wraps `initialize()` to re-apply the user's choice after MaxText's override:

```python
def apply_patches(maxtext_train):
    _orig_initialize = maxtext_train.initialize
    def _patched_initialize(argv):
        result = _orig_initialize(argv)
        import jax
        if prng_impl:
            jax.config.update("jax_default_prng_impl", prng_impl)
        verify_env()
        return result
    maxtext_train.initialize = _patched_initialize
```

Call order: `train_env.sh` sets `JAX_DEFAULT_PRNG_IMPL=threefry2x32` → `mfu_tracker.py` calls `deterministic.apply_patches()` → MaxText's `initialize()` sets `unsafe_rbg` → patch re-applies `threefry2x32`.

### Runtime verification

`deterministic.verify_env()` checks after `initialize()` completes:
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO` is `"0"`
- `XLA_FLAGS` contains `--xla_gpu_deterministic_ops`
- `TF_DETERMINISTIC_OPS` is `"1"`
- `HIPBLASLT_DETERMINISTIC` is `"1"`
- `NVTE_FUSED_ATTN` is `"0"`
- `jax.config.jax_default_prng_impl` is `"threefry2x32"`

Any mismatch logs a `WARNING`. This catches bugs like later config lines clobbering earlier overrides.

### Overriding individual knobs

The `_env_` mechanism exports variables *after* `train_env.sh`, so overrides take precedence:

```bash
# Re-enable CK fused attention (non-deterministic, but faster)
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 _env_NVTE_FUSED_ATTN=1

# Keep unsafe_rbg PRNG
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 _env_JAX_DEFAULT_PRNG_IMPL=unsafe_rbg
```

---

## Fix 1: Transformer Engine — `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`

| | |
|---|---|
| **Variable** | `NVTE_ALLOW_NONDETERMINISTIC_ALGO` |
| **Default** | `1` (non-deterministic allowed) |
| **Deterministic value** | `0` |
| **Scope** | TE fused attention forward, backward workspace, and backward lowering in `transformer_engine/jax/cpp_extensions/attention.py` |

When set to `0`, TE passes `deterministic=True` to CK/AOTriton fused attention kernels.

**Known limitation:** TE hardcodes `deterministic=false` at three call sites in `fused_attn.cpp` (lines 491, 680, 866), so this flag never reaches the CK backend. CK *does* implement `_deterministic` kernel variants (per-split dQ buffers + fixed-order reduction instead of `atomicAdd`), but TE never enables them. See Fix 7 and [upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md](upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md) for the full source audit.

**Source evidence:**

File: `transformer_engine/jax/cpp_extensions/attention.py`
```python
def is_non_deterministic_allowed():
    return bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))

deterministic=not FusedAttnHelper.is_non_deterministic_allowed(),
```

MaxText invokes TE without passing `deterministic=True`:
```python
dpa_layer = DotProductAttention(
    head_dim=head_dim, num_attention_heads=self.num_query_heads, ...
    # NOTE: no deterministic= parameter passed
)
```

**Ablation:** Redundant when `NVTE_FUSED_ATTN=0`. Kept defensively for future CK fix.

---

## Fix 2: XLA — `--xla_gpu_deterministic_ops=true`

| | |
|---|---|
| **Flag** | `--xla_gpu_deterministic_ops` (via `XLA_FLAGS`) |
| **Default** | `false` |
| **Scope** | GEMM autotuning, scatter ops, convolution algo selection, fMHA algo selection |

Controls:
1. **GEMM autotuning** — `select_first_config = true`
2. **Deterministic scatter** — non-atomic scatter for embeddings
3. **Deterministic convolutions** — `MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC=1`
4. **Deterministic fMHA** — `force_deterministic = RequireDeterminism()`

**Source evidence:**

```cpp
// xla/service/gpu/autotuning/autotuner_pass.cc
autotune_config.select_first_config =
    debug_options.xla_gpu_deterministic_ops() ||
    debug_options.xla_gpu_exclude_nondeterministic_ops() ||
    debug_options.xla_gpu_autotune_level() == 0;

// xla/service/gpu/stream_executor_util.cc
bool RequireDeterminism(const HloModuleConfig& config) {
    return config.debug_options().xla_gpu_deterministic_ops() ||
           config.debug_options().xla_gpu_exclude_nondeterministic_ops();
}
```

**Ablation:** Not needed — Docker image already sets `--xla_gpu_autotune_level=0`. Standard llama2-70b shapes don't trigger non-deterministic scatter. Kept defensively.

---

## Fix 3: RCCL — `NCCL_ALGO=Ring` + `RCCL_MSCCLPP_ENABLE=0`

| | |
|---|---|
| **Variables** | `NCCL_ALGO`, `RCCL_MSCCLPP_ENABLE` |
| **Deterministic values** | `NCCL_ALGO="Ring"`, `RCCL_MSCCLPP_ENABLE=0` |
| **Scope** | All RCCL collective operations |

- `NCCL_ALGO="Ring"` — forces Ring for all collectives (fixed reduction order)
- `RCCL_MSCCLPP_ENABLE=0` — disables MSCCL++ (bypasses `NCCL_ALGO`)

**Design note — why Ring-only, not Tree:** Earlier iterations used `NCCL_ALGO="Ring;allreduce:Tree"` with `NCCL_PROTO=Simple` and `RCCL_LL128_FORCE_ENABLE=0`. This caused SIGSEGV during `ncclCommInitRankConfig` on RCCL 2.27.7 / ROCm 7.1.1 (gfx950). Ring-only is stable.

**RCCL prefix parser limitation:** Only 5 collective types supported as prefixes: AllGather, AllReduce, AllToAllPivot, Broadcast, Reduce. `ReduceScatter` causes "Unrecognized prefix token" crash.

**Source evidence:**
```cpp
// rccl/src/graph/search.cc
str = getenv("NCCL_RINGS");
NCCLCHECK(parseGraph(str, system, graph, NULL, NULL,
    graph->pattern == NCCL_TOPO_PATTERN_RING ? system->hostIdx % 2 : 0));
```

**For stronger multi-node guarantees:**
```bash
export NCCL_RINGS="0 1 2 3 4 5 6 7"   # Pin ring order (cluster-specific)
```

**Intra-node vs inter-node:** Intra-node collectives use XGMI/PCIe direct GPU-to-GPU with fixed, symmetric topology. Inter-node uses InfiniBand/RoCE over the network, where `hostIdx % 2` alternates ring direction for even/odd hosts. MSCCL++ may use different code paths for each. Our tests covered both (single-node = intra only, 2-node = intra + inter) and found RCCL deterministic without pinning in both cases.

**Ablation:** Not needed single-node or 2-node (identical checksums with/without). May matter at larger scale (4+ nodes) where topology variance increases. Kept defensively.

---

## Fix 4: rocBLAS — `TF_DETERMINISTIC_OPS=1`

| | |
|---|---|
| **Variable** | `TF_DETERMINISTIC_OPS` |
| **Scope** | rocBLAS GEMM atomics (`rocblas_set_atomics_mode`) + MIOpen convolutions |

Triggers `rocblas_atomics_not_allowed` before every rocBLAS call (`xla/stream_executor/rocm/rocm_blas.cc`). Separate code path from `xla_gpu_deterministic_ops` — both needed for full coverage.

**Limitation:** Does NOT affect hipBLASLt — only legacy rocBLAS fallback.

**Ablation:** Not needed for llama2-70b (hipBLASLt handles GEMMs via `enable_cublaslt=True`). Kept defensively for shapes falling back to rocBLAS.

---

## Fix 5: hipBLASLt — `HIPBLASLT_DETERMINISTIC=1`

| | |
|---|---|
| **Variable** | `HIPBLASLT_DETERMINISTIC` |
| **Scope** | All hipBLASLt GEMM calls (XLA linear layers + TE internal GEMMs) |
| **Requires** | Patched `libhipblaslt.so` (12-line patch to `tensile_host.cpp`) |

### How TensileLite DeterministicMode works

TensileLite already has full infrastructure:
1. `ContractionProblemGemm::setDeterministicMode(bool)` / `deterministicMode()`
2. `DeterministicModeEqual` predicate filters solutions during kernel selection
3. Solutions using atomic GlobalSplitU or StreamKAtomic tagged `DeterministicMode=False`

```python
# Contractions.py (solution tagging):
if GlobalSplitU > 1 and _GlobalAccumulation != 'MultipleBuffer':
    predicates += [DeterministicMode == False]  # atomic GSU
if StreamK > 0 and StreamKAtomic == 1:
    predicates += [DeterministicMode == False]  # atomic partial tiles
```

The only gap: hipBLASLt never calls `setDeterministicMode(true)`. Our 12-line patch adds `getenv("HIPBLASLT_DETERMINISTIC")` in `ConstructTensileProblem` and `updateTensileProblem`.

### Solution library coverage (MI300X BF16)

```
Total solutions:                                     78
With DeterministicMode=False (atomic GSU, GA=3):     13  (17%)
Without predicate (deterministic-safe, GA=2):        65  (83%)
```

65 safe solutions use `MultipleBuffer` (fixed-order reduction). 13 filtered use atomic accumulation (perf optimization for tall-skinny matrices).

### Build requirements

| Component | Rebuild? |
|-----------|----------|
| hipBLASLt library | **Yes** — `tensile_host.cpp` compiled into `libhipblaslt.so` |
| TensileLite kernels | No — predicates already in `.dat` files |
| XLA/jaxlib | No (Option A env var) |
| MaxText | No — transparent through XLA |

### Option A (implemented) vs Option B (upstream)

| | Option A (env var, current) | Option B (public API, upstream) |
|---|---|---|
| Lines | 12 (1 file) | ~30 (5 files hipBLASLt + 1 XLA) |
| Scope | Global per-process | Per-matmul via descriptor attribute |
| Effort | Done | 3-5 days |

**Ablation:** Not needed for standard GEMM shapes (they don't select atomic-GSU). Kept defensively for non-standard shapes.

---

## Fix 6: JAX PRNG — `JAX_DEFAULT_PRNG_IMPL=threefry2x32`

| | |
|---|---|
| **Variable** | `JAX_DEFAULT_PRNG_IMPL` (consumed by `utils/deterministic.py`) |
| **Default** | MaxText hardcodes `unsafe_rbg` |
| **Scope** | All random operations: weight init, dropout, data shuffling |

`threefry2x32` is a deterministic counter-based hash. `unsafe_rbg` uses XLA's `RngBitGenerator` with `fold_in` as a simple XOR — JAX documents it as "not guaranteed to be stable/deterministic across backends."

**Source evidence:**
```python
# jax/_src/prng.py
# Notice that the RngBitGenerator operations are not guaranteed to be
# stable/deterministic across backends or compiler versions.

def _unsafe_rbg_fold_in(key, data):
    _, random_bits = lax.rng_bit_generator(_rbg_seed(data), (10, 4), dtype='uint32')
    return key ^ random_bits[-1]
```

**Ablation:**
- `unsafe_rbg` + `enable_dropout=False`: BIT-EXACT — deterministic on ROCm/gfx950 for fixed seeds
- `threefry2x32` + `dropout_rate=0.1`: BIT-EXACT — checksum `f3c76093cf1633e2`, real dropout (10% zeroed)
- Kept defensively because JAX warns `unsafe_rbg` may not be stable across versions

---

## Fix 7: CK Fused Attention — `NVTE_FUSED_ATTN=0`

| | |
|---|---|
| **Variable** | `NVTE_FUSED_ATTN` |
| **Default** | `1` (fused attention via CK backend) |
| **Deterministic value** | `0` (falls back to JAX native attention) |
| **Performance impact** | **~9.7x throughput drop** (batch reduction 8→1 plus unfused overhead) |

### Fused vs Unfused Architecture

```
CK Fused (NVTE_FUSED_ATTN=1)         JAX Native (NVTE_FUSED_ATTN=0)
──────────────────────────────        ──────────────────────────────
Single GPU kernel computes            Three separate XLA ops:
Q×K→softmax→×V in tiles              1. Q @ K^T       → hipBLASLt GEMM
                                      2. softmax(QK/√d) → element-wise
Memory: O(seq_len) per head           3. attn @ V       → hipBLASLt GEMM
  → never materializes full matrix    Backward: dQ, dK, dV via separate GEMMs

~968 TFLOP/s/device                   Memory: O(seq_len²) per head
                                        → full [batch, heads, seq, seq] matrix
                                        → llama2-70b: 8×64×4096²×2B = 137 GB/GPU

                                      ~100 TFLOP/s/device
```

The JAX native path is deterministic: standard GEMMs (hipBLASLt, proven deterministic) + element-wise ops. CK avoids materializing the attention matrix via tiling. The non-deterministic (`_ndeterministic`) backward kernel uses `atomicAdd` for dQ gradient accumulation — confirmed by `dk_dv_reduce`, `dk_or_dv_reduce_thd`, `dbias_reduce_*` symbols in `libck_fused_attn.so` and PR #2705 on ROCm/composable_kernel.

CK *does* implement `_deterministic` kernel variants (8,208 for gfx950 in `libmha_bwd.so`) that use per-split dQ buffers + `FmhaBwdConvertQGradKernel` for fixed-order reduction instead of `atomicAdd`. The `is_deterministic` flag is propagated through TE → C++ (`attention_hip.cpp` lines 481-611) → `libck_fused_attn.so`, but **TE hardcodes `false` at three call sites** in `fused_attn.cpp` (lines 491, 680, 866), so the `_ndeterministic` variant is always dispatched. See [upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md](upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md) for the full source audit and risk assessment (~70% confidence the `_deterministic` path is correct end-to-end — it has never been exercised in production).

### OOM math

Unfused attention materializes `[batch, heads, seq_len, seq_len]`:
- `per_device_batch_size=8`: `8 × 64 × 4096² × 2B = 137 GB` → OOM
- `per_device_batch_size=1`: `1 × 64 × 4096² × 2B = 17 GB` → fits

---

## Verification Plan

```bash
# 3 runs with fresh compilation
rm -rf ~/jax_cache
./in_container_run.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 steps=15 enable_dropout=False per_device_batch_size=1

rm -rf ~/jax_cache
./in_container_run.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 steps=15 enable_dropout=False per_device_batch_size=1

rm -rf ~/jax_cache
./in_container_run.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 steps=15 enable_dropout=False per_device_batch_size=1

# Compare at full precision
python3 skills/deterministic-training/compare_runs.py --glob '*DETERMINISTIC*steps_15*enable_dropout_False*'
```

Follow-up: repeat with `enable_dropout=True dropout_rate=0.1`.

## Runtime Log Output

```
[deterministic] Enabled (DETERMINISTIC_MODE=1)
[deterministic] XLA_FLAGS += --xla_gpu_deterministic_ops=true
[deterministic] TF_DETERMINISTIC_OPS=1 (rocBLAS atomics disabled)
[deterministic] HIPBLASLT_DETERMINISTIC=1 (requires patched hipBLASLt)
[deterministic] NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
[deterministic] NCCL_ALGO=Ring  RCCL_MSCCLPP_ENABLE=0
[deterministic] JAX_DEFAULT_PRNG_IMPL=threefry2x32 (applied post-init by deterministic.py)
[deterministic] NVTE_FUSED_ATTN=0 (CK attention disabled for bit-exact determinism)
...
[deterministic] PRNG override: jax_default_prng_impl=threefry2x32 (was unsafe_rbg)
[deterministic] All env-var checks passed.
...
[determinism] loss_checksum=637613d3a620c008 (steps=50)
```
