# Deterministic Training — Technical Reference

Detailed per-fix documentation, code traces, and implementation notes. For the overview and experimental results, see [SKILL.md](SKILL.md).

> **Status (2026-05):** Updated for the post-PR-508 image
> (`rocm/jax-training:maxtext-v26.2-det-te508-aot`, TE `2.12.0.dev0+8943023d`). The historical
> `NVTE_FUSED_ATTN=0` workaround is no longer the deterministic path — CK is. The
> `xla_gpu_deterministic_ops=true` flag is now **deliberately NOT set** because it serializes
> MoE scatter ops (see Fix 2 below). Where this document refers to pre-PR-508 behavior, it is
> marked explicitly as historical.

## Implementation Overview

### Where the code lives

| File | What it does |
|------|--------------|
| `train_env.sh` | `DETERMINISTIC_MODE` block — sets the deterministic env vars (TE, hipBLASLt, rocBLAS, RCCL, PRNG, XLA autotune). Later defaults use `${VAR:-default}` to avoid clobbering. |
| `utils/deterministic.py` | `apply_patches()` — monkey-patches MaxText's `initialize()` to override hardcoded `unsafe_rbg` PRNG. `verify_env()` — runtime checks for all critical flags. `LossChecksumTracker` — SHA-256 over loss values for cross-run comparison. |
| `utils/mfu_tracker.py` | Training entry point. Imports `deterministic` module and delegates: calls `deterministic.apply_patches()` before training, `deterministic.extract_loss()` during stdout interception, and `deterministic.print_loss_checksum()` after training completes. |

### How it works in `train_env.sh` (post-PR-508)

```bash
DETERMINISTIC_MODE="${DETERMINISTIC_MODE:-${EXTRACTED_ENV_MAP[DETERMINISTIC_MODE]:-0}}"
if [[ "${DETERMINISTIC_MODE,,}" =~ ^(1|y|yes|true)$ ]]; then
    # XLA autotune: ensure autotune_level=0 (image default, re-asserted defensively)
    if [[ "${XLA_FLAGS:-}" != *"xla_gpu_autotune_level"* ]]; then
        XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0"
    fi
    export XLA_FLAGS
    export TF_DETERMINISTIC_OPS=1                  # rocBLAS atomics + MIOpen
    export HIPBLASLT_DETERMINISTIC=1                # hipBLASLt (needs patched library)
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0       # The one flag that actually matters — wires TE → CK _deterministic
    export NCCL_ALGO="Ring"                         # Pin collectives to Ring (defensive)
    export RCCL_MSCCLPP_ENABLE=0                    # Disable MSCCL++ (defensive)
    export JAX_DEFAULT_PRNG_IMPL=threefry2x32       # PRNG (applied post-init)
    # NOTE: NVTE_FUSED_ATTN stays at its default (=1, CK fused enabled).
    # NOTE: xla_gpu_deterministic_ops=true is DELIBERATELY NOT SET (toxic on MoE — see Fix 2).
fi
```

Later defaults avoid clobbering:
```bash
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}
export RCCL_MSCCLPP_ENABLE=${RCCL_MSCCLPP_ENABLE:-1}
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1}
```

### Pre-PR-508 historical block (legacy images)

For reference, the pre-PR-508 `DETERMINISTIC_MODE` block also set
`XLA_FLAGS+=" --xla_gpu_deterministic_ops=true"` and `export NVTE_FUSED_ATTN=0`. Both were
required because TE hardcoded `deterministic=false` for CK (see Fix 1), forcing fallback to
JAX unfused attention. The unfused path has `O(seq²)` memory, OOMs at `bs=8`, and degrades
throughput ~9.7×. On the new image both lines are removed; on old images they remain
necessary.

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
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO` is `"0"` (the only required check)
- `TF_DETERMINISTIC_OPS` is `"1"` (defensive)
- `HIPBLASLT_DETERMINISTIC` is `"1"` (defensive)
- `NCCL_ALGO` is `"Ring"` (defensive)
- `RCCL_MSCCLPP_ENABLE` is `"0"` (defensive)
- `XLA_FLAGS` contains `--xla_gpu_autotune_level=0` (image default; image-aware sanity check)
- `jax.config.jax_default_prng_impl` is `"threefry2x32"` (defensive)
- `NVTE_FUSED_ATTN`: path-aware informational note only (no enforced value). `=1` + `NVTE_FUSED_ATTN_CK=1` → CK deterministic path (post-PR-508); `=0` → legacy unfused workaround.

Any mismatch on the required check logs a `WARNING`. Defensive flags log `INFO` if missing.
This catches bugs like later config lines clobbering earlier overrides.

### Overriding individual knobs

The `_env_` mechanism exports variables *after* `train_env.sh`, so overrides take precedence:

```bash
# Turn off the single required flag (back to non-deterministic CK)
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 _env_NVTE_ALLOW_NONDETERMINISTIC_ALGO=1

# Force the legacy unfused workaround (pre-PR-508 path; will OOM at bs=8)
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 _env_NVTE_FUSED_ATTN=0 per_device_batch_size=1

# Keep unsafe_rbg PRNG
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 _env_JAX_DEFAULT_PRNG_IMPL=unsafe_rbg
```

---

## Fix 1: Transformer Engine — `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (the one that matters)

| | |
|---|---|
| **Variable** | `NVTE_ALLOW_NONDETERMINISTIC_ALGO` |
| **Default** | `1` (non-deterministic allowed) |
| **Deterministic value** | `0` |
| **Scope** | TE fused attention forward, backward workspace, and backward lowering in `transformer_engine/jax/cpp_extensions/attention.py`. **Post-PR-508: also dispatches the `_deterministic` CK kernel variant.** |

When set to `0`, TE passes `deterministic=True` to CK fused attention kernels. **Post-PR-508
(TE ≥ `2.12.0.dev0+8943023d`), this flag reaches the CK kernel and dispatches the
`_deterministic` variant — per-split `dQ_accum` buffers + fixed-order reduce instead of
`atomicAdd`.**

**Source evidence:**

File: `transformer_engine/jax/cpp_extensions/attention.py`
```python
def is_non_deterministic_allowed():
    return bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))

deterministic=not FusedAttnHelper.is_non_deterministic_allowed(),
```

File: `transformer_engine/common/fused_attn_rocm/fused_attn.cpp` (post-PR-508, lines 491, 680, 866):
```cpp
- /*deterministic=*/false,    // pre-PR-508 (TE 2.8 and earlier)
+ /*deterministic=*/deterministic,  // post-PR-508 (TE 2.12.0.dev0+8943023d)
```

MaxText still invokes TE without explicit `deterministic=` (see action item #7 in [upstream-action-items.md](upstream-action-items.md)) — relies on the env var:

```python
dpa_layer = DotProductAttention(
    head_dim=head_dim, num_attention_heads=self.num_query_heads, ...
    # NOTE: no deterministic= parameter passed; env-var driven
)
```

**Ablation (post-PR-508):** This is the **sole functionally required** flag. Removing it
DIVERGES (~46 ULPs by step 2). All other flags in `DETERMINISTIC_MODE` are defensive.

**Verification at runtime:** `NVTE_LOG_CK_CONFIG=1` prints `attn_bwd(ck): ... deterministic: 1`
when the flag is correctly wired through.

---

## Fix 2: XLA — `--xla_gpu_deterministic_ops=true` (DELIBERATELY NOT SET — toxic on MoE)

| | |
|---|---|
| **Flag** | `--xla_gpu_deterministic_ops` (via `XLA_FLAGS`) |
| **Default** | `false` |
| **Our setting** | **NOT SET** (left at default `false`) |
| **Reason** | Activates XLA's `ScatterDeterminismExpander`, which serializes MoE scatter ops into 8192-trip-count while-loops → **152× throughput collapse** on ds-proxy-e128-h2048. No-op on dense. |

### What the flag controls

1. **Scatter determinism** — XLA runs `ScatterDeterminismExpander`, rewriting every scatter / scatter-add into a sequential while-loop with `trip_count = scatter_size` (8192 for MoE routing at seq=4096, top_k=2)
2. **GEMM autotuning** — `select_first_config = true` (already covered by `autotune_level=0`)
3. **Deterministic convolutions** — `MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC=1` (no convolutions in transformer training)
4. **Deterministic fMHA** — `force_deterministic = RequireDeterminism()` (TE custom_call handles this internally via `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`; XLA just passes the flag through)

### Why it's toxic on MoE

Each MoE layer has 2 scatter ops (token dispatch + expert-output combine). At `seq=4096`,
`top_k=2`, each scatter has `trip_count=8192`. Across 64 decoder layers with
fwd + remat-fwd + bwd ≈ 3-4 instances per layer, that's **~1M sequential scalar HBM
read-modify-write ops per training step**.

Measured impact (ds-proxy-e128-h2048, bs=4):

| Mode | sec/step | TFLOP/s/dev |
|---|---|---|
| det with `xla_gpu_deterministic_ops=true` | **264.5** | **1.6** |
| det without (control) | 1.90 | 219 |
| nondet | 1.74 | 240 |

**HLO evidence:** the DET HLO has 618 while-loops vs NONDET's 505 — `+113` extra while-loops,
all from `layers/scatter` and `layers/scatter-add` with `backend_config="known_trip_count: 8192"`.
Both HLOs contain *identical* `te_fused_attn_forward/backward_ffi` (80+53) and
`__cublas$lt$matmul` (207) — so TE custom_calls are NOT decomposed (my earlier hypothesis,
disproven by HLO). Full analysis: [`deterministic-proj/harness/reports/2026-05-06_moe_scatter_serialization.md`](../../deterministic-proj/harness/reports/2026-05-06_moe_scatter_serialization.md).

### Why it's safe to leave off on this stack

MoE scatter on gfx950 is conflict-free by construction (each token routes to a unique expert
slot — no `atomicAdd` race). Verified by 200-step dual-runs:

- ds-proxy-e128-h2048 std path, 200 steps, no `xla_gpu_deterministic_ops`: BIT-EXACT
- ds-proxy-e128-h2048 sparse_matmul path, 200 steps: BIT-EXACT (HLO has zero atomicAdd anywhere in the train step)
- ds-proxy-e256-h2048 sparse_matmul, 50 steps: BIT-EXACT

The flag's other three sub-functions are no-ops or covered by other flags on this stack:
GEMM autotune disabled via `xla_gpu_autotune_level=0`; no convolutions in transformer
training; fMHA determinism handled by TE custom_call.

### XLA prefix-scan alternative

NVIDIA developed a prefix-scan replacement for the sequential `ScatterDeterminismExpander`
(XLA [PR #17886](https://github.com/openxla/xla/pull/17886) + [PR #19275](https://github.com/openxla/xla/pull/19275)),
exposed via `xla_gpu_enable_scatter_determinism_expander`. **Still default=false upstream**
as of 2026-05 due to bug history (see [upstream-action-items.md](upstream-action-items.md) item 4).

Tested manually on ds-proxy-e128-h2048: **OOM at bs=1** (sort buffer ~16 GB exceeds HBM for
128-expert models). Not viable for production MoE on current hardware.

### Source evidence

```cpp
// xla/service/scatter_determinism_expander.cc
// If deterministic operations are required, rewrite Scatter operations
// to use a while-loop that sequentially processes each scatter index.

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

### Ablation (post-PR-508)

| Removed | Result |
|---|---|
| `xla_gpu_autotune_level=0` (let autotune run) | DIVERGE — required |
| `xla_gpu_deterministic_ops=true` (not set in DETERMINISTIC_MODE; tested with it enabled) | DENSE: BIT-EXACT (no impact). MoE: BIT-EXACT but 152× slower. |

The image's default `xla_gpu_autotune_level=0` is the only XLA-side guarantee needed.

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

Triggers `rocblas_atomics_not_allowed` before every rocBLAS call (`xla/stream_executor/rocm/rocm_blas.cc`). Independent code path from XLA's deterministic-ops machinery.

**Limitation:** Does NOT affect hipBLASLt — only legacy rocBLAS fallback. MIOpen conv determinism (`MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC=1`) is also gated by this flag; not applicable to transformer training (no convolutions in the HLO).

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

## Fix 7: CK Fused Attention — the `_deterministic` kernel variant (post-PR-508)

| | |
|---|---|
| **Variable** | `NVTE_FUSED_ATTN` (default `1`, leave at default on the new image) |
| **Companion variables** | `NVTE_FUSED_ATTN_CK=1`, `NVTE_FUSED_ATTN_AOTRITON=0` |
| **What activates determinism** | `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (see Fix 1) |
| **Performance impact** | **~6% throughput cost vs non-det** at max-pdbs (903 vs 958 TFLOP/s/dev on llama2-70B) |

### Fused vs unfused architecture

```
CK Fused (NVTE_FUSED_ATTN=1, default)    JAX Native (NVTE_FUSED_ATTN=0, legacy workaround)
──────────────────────────────────────   ──────────────────────────────────────────────
Single GPU kernel computes               Three separate XLA ops:
Q·K → softmax → ·V in tiles              1. Q @ K^T       → hipBLASLt GEMM
                                          2. softmax(QK/√d) → element-wise
Memory: O(seq_len) per head              3. attn @ V       → hipBLASLt GEMM
  → never materializes full matrix       Backward: dQ, dK, dV via separate GEMMs

~958 TFLOP/s/dev (nondet)                Memory: O(seq_len²) per head
~903 TFLOP/s/dev (deterministic)           → full [batch, heads, seq, seq] matrix
                                           → llama2-70B: 8×64×4096²×2B = 137 GB/GPU
                                           → OOM at bs=8

                                         ~100 TFLOP/s/dev at bs=1 (forced by OOM)
                                         9.0× slower than the deterministic CK path.
```

### Why the `_deterministic` variant exists

The non-deterministic CK backward (`_ndeterministic` symbols) uses `atomicAdd` for `dQ`
gradient accumulation. Multiple SMs partition the KV axis, each computes `dQ_local`, and
they race to atomically add into the shared `dQ_accum` HBM buffer. FP non-associativity →
final value depends on arrival order → run-to-run non-determinism. Confirmed by
`dk_dv_reduce`, `dk_or_dv_reduce_thd`, `dbias_reduce_*` symbols in `libck_fused_attn.so`
and PR #2705 on `ROCm/composable_kernel`.

The `_deterministic` variant pre-allocates `nsplits = ceil(s_kv / kN0)` per-split
`dQ_accum` buffers (where `kN0 = 128` if `d_qk ≤ 128` else `64`). Each SM writes to its
own split offset (no `atomicAdd`); a separate `FmhaBwdConvertQGradKernel` sums splits in
fixed block-index order (not runtime arrival order). Both variants compiled into
`libmha_bwd.so` (8,208 kernels for gfx950).

### What PR #508 fixed

Pre-PR-508 (TE 2.8.0.dev0 and earlier), `nvte_fused_attn_bwd` in
`transformer_engine/common/fused_attn_rocm/fused_attn.cpp` hardcoded
`/*deterministic=*/false` at three call sites (lines 491, 680, 866), discarding the
caller's parameter. The `is_deterministic` flag was propagated through TE Python → C++
(`attention_hip.cpp` lines 481-611) → CK API, but never actually reached the kernel call.

[ROCm/TransformerEngine PR #508](https://github.com/ROCm/TransformerEngine/pull/508) (merge
commit `8943023d`) replaces `false` with `deterministic` at those three lines. The CK
deterministic path that had been shipped for years is now actually dispatched.

### Verification at runtime

```bash
NVTE_LOG_CK_CONFIG=1 ./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 steps=5
```

The CK log shows:
```
attn_bwd(ck): ... deterministic: 1
```

If you see `deterministic: 0` instead, the image is pre-PR-508 — fall back to
`NVTE_FUSED_ATTN=0` (the legacy workaround documented at the end of this section).

### Memory cost — closed-form formula

`transformer_engine/common/fused_attn_rocm/fused_attn_ck.cpp:757-784`:

```cpp
size_t kN0 = (d_qk <= 128) ? 128 : 64;
size_t nsplits = deterministic ? ceil(1.0 * s_kv / kN0) : 1;
...
void* dq_acc_ptr = planner.allocate(nsplits * h * max_tokens_q * d_qk * sizeof(float));
```

```
extra_bytes = (ceil(s_kv / kN0) - 1) × num_heads_q × (b × s_q) × head_dim_qk × 4
              kN0 = 128 if d_qk ≤ 128 else 64
```

Validated against 13 TE unit-test configs (matches measured peak alloc within 0.25 MB).
TE team confirmed 2026-05-06 that this cost is intrinsic to the split-accumulator
algorithm and cannot be optimized without sacrificing determinism. The `convert_dQ`
reduction is HBM-bandwidth-bound; cost grows linearly with `nsplits`.

There is currently no `nsplits`-capping knob upstream — see [upstream-action-items.md](upstream-action-items.md)
item 2 for the proposed `NVTE_CK_DETERMINISTIC_MAX_SPLITS` env var.

### Production impact (llama2-70B)

| | det | non-det |
|---|---|---|
| max-pdbs (chi2868, 8× gfx950) | 11 | 16 |
| TFLOP/s/dev at max-pdbs | 903 | 958 |
| Mem per pdbs | +9.6 GB | +6.6 GB |
| Loss of batch headroom (det vs non-det) | 5 pdbs (31%) | — |
| Throughput slowdown at max-vs-max | 1.06× (6%) | baseline |
| Kernel-only slowdown (matched bs=10 or 11) | ~5.1% | baseline |

### Legacy fallback: `NVTE_FUSED_ATTN=0` (pre-PR-508 images only)

For images **without** PR #508 (TE < `2.12.0.dev0+8943023d`), the only deterministic path
is to disable CK entirely:

```bash
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 _env_NVTE_FUSED_ATTN=0 per_device_batch_size=1
```

This forces fallback to JAX native attention (3 separate ops: GEMM + softmax + GEMM).
- Memory: `O(seq²)` materializes `[batch, heads, seq, seq]`. At llama2-70B `seq=4096`:
  `bs=8 × 64 × 4096² × 2B = 137 GB` → OOM. Forces `per_device_batch_size=1`.
- Throughput: ~100 TFLOP/s/dev at bs=1 → **9.0× slower** than the deterministic CK path.

On the new image this fallback is wrong: it costs 9× perf for no benefit.

---

## Verification Plan

```bash
# Image with PR #508 wired in
docker pull rocm/jax-training:maxtext-v26.2-det-te508-aot

# 2 runs with fresh JAX cache, full production batch
rm -rf ~/jax_cache
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 steps=50 enable_dropout=False

rm -rf ~/jax_cache
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 steps=50 enable_dropout=False

# Compare at full precision (TensorBoard is the enforced source of truth)
python3 skills/deterministic-training/compare_runs.py --glob '*DETERMINISTIC_MODE_1*steps_50*'
```

Follow-ups:
- Dropout: `enable_dropout=True`
- MoE std path: `./submit.sh ds-proxy-e128-h2048 -- _env_DETERMINISTIC_MODE=1 steps=50 per_device_batch_size=6`
- MoE sparse_matmul: `./submit.sh ds-proxy-e128-h2048 -- _env_DETERMINISTIC_MODE=1 steps=50 sparse_matmul=True shardy=True`
- Failure-mode demo: omit `_env_DETERMINISTIC_MODE=1`, three runs should DIFFER from step 2

## Runtime Log Output (post-PR-508)

```
[deterministic] Enabled (DETERMINISTIC_MODE=1)
[deterministic] XLA autotune_level=0 confirmed
[deterministic] TF_DETERMINISTIC_OPS=1 (rocBLAS atomics disabled)
[deterministic] HIPBLASLT_DETERMINISTIC=1 (requires patched hipBLASLt)
[deterministic] NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
[deterministic] NCCL_ALGO=Ring  RCCL_MSCCLPP_ENABLE=0
[deterministic] JAX_DEFAULT_PRNG_IMPL=threefry2x32 (applied post-init by deterministic.py)
[deterministic] CK fused attention uses _deterministic variants via NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
[deterministic] (requires PR-#508 image; legacy unfused workaround = pass _env_NVTE_FUSED_ATTN=0)
...
[deterministic] PRNG override: jax_default_prng_impl=threefry2x32 (was unsafe_rbg)
[deterministic] All env-var checks passed.
...
[determinism] loss_checksum=e352e66f43134c70 (steps=1000)
```

(Pre-PR-508 images additionally print `XLA_FLAGS += --xla_gpu_deterministic_ops=true` and
`NVTE_FUSED_ATTN=0`. The new image deliberately omits both — see Fix 2 and Fix 7.)
