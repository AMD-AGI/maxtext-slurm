# Deterministic Training — Cheatsheet

Quick reference for all flags, settings, and concepts validated in this project.
For the full overview, see [SKILL.md](SKILL.md).

## One-Liner

```bash
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1
```

Requires image `rocm/jax-training:maxtext-v26.2-det-te508-aot` (or any image containing
[ROCm/TransformerEngine PR #508](https://github.com/ROCm/TransformerEngine/pull/508), i.e.
TE ≥ `2.12.0.dev0+8943023d`). No batch reduction needed. Works on dense + MoE.

## Env Vars Set by `DETERMINISTIC_MODE=1`

| Env Var | Value | What It Does | Needed? |
|---|---|---|---|
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | TE dispatches the `_deterministic` CK kernel variant | **YES — sole required setting** |
| `NVTE_FUSED_ATTN` | `1` (default) | Use CK fused path (deterministic variant) | Required for the new image; the CK kernel handles determinism internally |
| `NVTE_FUSED_ATTN_CK` | `1` | Pin to CK backend (vs AOTriton) | Required — AOTriton rejects GQA in TE 2.12 |
| `NVTE_FUSED_ATTN_AOTRITON` | `0` | Disable AOTriton fallback | Required |
| `--xla_gpu_autotune_level` | `0` | Pin GEMM kernel selection (no autotune) | Image default; re-asserted defensively |
| `TF_DETERMINISTIC_OPS` | `1` | Disables rocBLAS atomic reductions + MIOpen det conv | Defensive — most GEMMs use hipBLASLt not rocBLAS |
| `HIPBLASLT_DETERMINISTIC` | `1` | Filters atomic-GSU TensileLite solutions | Defensive — gfx950/ROCm 7.1.1 BF16 has no atomic-GSU |
| `NCCL_ALGO` | `Ring` | Pins RCCL collectives to Ring | Defensive — bit-exact without it at single-node + 2-node |
| `RCCL_MSCCLPP_ENABLE` | `0` | Disables MSCCL++ (bypasses NCCL_ALGO) | Defensive |
| `JAX_DEFAULT_PRNG_IMPL` | `threefry2x32` | Deterministic PRNG (overrides MaxText's `unsafe_rbg`) | Defensive — `unsafe_rbg` is empirically det on gfx950 |

### Not set by `DETERMINISTIC_MODE` (and why)

| Env Var | Why not |
|---|---|
| `NVTE_FUSED_ATTN=0` | **Legacy workaround** for pre-PR-508 images. On the new image this is wrong: it forces unfused attention (`O(seq²)` memory), OOMs at bs=8 on llama2-70B, and degrades to ~9.7× lower throughput. Pass `_env_NVTE_FUSED_ATTN=0` explicitly only if you must run on an old image. |
| `--xla_gpu_deterministic_ops=true` | **Harmful on MoE.** XLA's `ScatterDeterminismExpander` serializes each MoE scatter op to a sequential while-loop (8192 iterations per layer × 64 layers × fwd+bwd ≈ ~1M serial ops/step) — measured **152× throughput collapse** on ds-proxy-e128-h2048. No-op on dense. Bit-exactness is preserved without it because MoE scatter on gfx950 is conflict-free by construction. |

## Concepts

| Concept | Meaning |
|---|---|
| **CK (Composable Kernel)** | AMD's GPU kernel library. CK fused attention computes Q·K → softmax → ·V in one kernel without materializing the full attention matrix. |
| **`_deterministic` CK variant** | Per-split `dQ_accum` buffers + `convert_dQ` ordered-reduce kernel (replaces `atomicAdd`). Compiled into `libmha_bwd.so`; dispatched when `deterministic=true` reaches the CK call. |
| **TE PR #508** | Three-line fix in `transformer_engine/common/fused_attn_rocm/fused_attn.cpp` (lines 491, 680, 866): `/*deterministic=*/false` → `/*deterministic=*/deterministic`. The flag now actually reaches the CK kernel. |
| **`atomicAdd` race** | GPU instruction whose arrival order depends on SM completion — the only fundamental reduction race in transformer training (FA backward `dQ` accumulation). |
| **Split-K + ordered reduce** | Each block writes to its own split offset; a second kernel sums splits in fixed block-index order. Standard deterministic alternative to `atomicAdd`. |
| **`nsplits`** | `ceil(s_kv / kN0)` where `kN0 = 128` if `d_qk ≤ 128` else `64`. Determines per-split workspace size. |
| **ULP (Unit in Last Place)** | Smallest representable FP difference at a given magnitude. |
| **`xla_gpu_deterministic_ops`** | XLA flag that activates `ScatterDeterminismExpander` (serializes scatter ops), `select_first_config` for GEMM autotune, det MIOpen conv, fMHA `force_deterministic`. **Toxic on MoE.** |
| **`ScatterDeterminismExpander`** | XLA HLO pass that rewrites scatter/scatter-add into a sequential while-loop with `trip_count=8192` (the MoE scatter size). Pure overhead on gfx950 because MoE scatter has no `atomicAdd` race. |
| **TensileLite `DeterministicMode`** | hipBLASLt predicate that filters solutions using atomic-GSU or StreamKAtomic. Activated by `HIPBLASLT_DETERMINISTIC=1` env var (requires patched `libhipblaslt.so`). |
| **threefry2x32** | JAX's deterministic counter-based PRNG. |
| **unsafe_rbg** | JAX's fast PRNG via XLA `RngBitGenerator`. Documented as "not guaranteed deterministic across backends"; empirically deterministic on gfx950. |

## What Was Tested (post-PR-508 image)

### Models

| Model | Type | Params | Steps | Nodes | Result | Checksum |
|---|---|---|---|---|---|---|
| llama2-70B | Dense | 68.977B | 1000 | 1 | BIT-EXACT | `e352e66f43134c70` |
| llama2-70B | Dense | 68.977B | 200 (max-pdbs=11) | 1 | BIT-EXACT | `1d857e6d219848c5` |
| llama2-70B | Dense + seq=2048 | 68.977B | 50 | 1 | BIT-EXACT | `d5654b0dc37ec0f4` |
| llama2-70B | Dense + seq=8192 | 68.977B | 50 | 1 | BIT-EXACT | — |
| ds-proxy-e128-h2048 | MoE std | 128 experts top-2 | 200 | 1 | BIT-EXACT | — |
| ds-proxy-e128-h2048 | MoE sparse_matmul | 128 experts top-2 | 200 | 1 | BIT-EXACT | `8ba4d5eb7405f661` |
| ds-proxy-e256-h2048 | MoE sparse_matmul | 256 experts top-4 | 50 | 1 | BIT-EXACT (`remat_policy=full`) | `ee0bf8f94c025a5a` |

### Cross-host / topology

| Test | Hosts | Result |
|---|---|---|
| Cross-host bit-exact | chi2868 vs chi2774 (each 50 steps, single-node) | BIT-EXACT all 50 steps, max_delta=0.0; checksum `050fb73070bb9d91` |
| Grad-accum invariance | `pdbs=4 gas=2` × 2 independent runs | BIT-EXACT all 50 steps |
| 2-node distributed (pre-PR-508 baseline) | chi2870 + chi2872 | BIT-EXACT |
| 8-node MoE (pre-PR-508 baseline) | ds-proxy-se0-e256-h4096, 200 steps | BIT-EXACT, `7ff136bdcaf72e8a` |

### Dropout

| dropout_rate | PRNG | Model | Result |
|---|---|---|---|
| 0.0 (no-op) | threefry2x32 | llama2-70B | BIT-EXACT |
| 0.1 (real) | threefry2x32 | llama2-70B | BIT-EXACT (`f3c76093cf1633e2`) |
| 0.0 | unsafe_rbg | llama2-70B | BIT-EXACT |
| True | threefry2x32 | ds-proxy-e128-h2048 MoE | BIT-EXACT |

### Failure-mode demo

3 independent llama2-70B runs with `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1` (default):

| Step | max_delta across 3 runs |
|---|---|
| 0–1 | 0.0 (forward-only) |
| 2 | 5.0e-5 |
| 13 | **6.7e-3** (peak) |
| 49 | 9.2e-6 (~13 ULPs) |

Step-49 std-dev: 4.75e-6. MoE non-det matches this pattern (diverges from step 2, attention dQ dominates; HLO confirms zero atomicAdd in MoE scatter).

### Ablation (single-flag removal, post-PR-508)

| Removed | Result |
|---|---|
| `TF_DETERMINISTIC_OPS` | BIT-EXACT — not needed |
| `HIPBLASLT_DETERMINISTIC` | BIT-EXACT — not needed on gfx950/ROCm 7.1.1 |
| `NCCL_ALGO + RCCL_MSCCLPP_ENABLE` | BIT-EXACT — not needed single-node/2-node |
| `JAX_DEFAULT_PRNG_IMPL` | BIT-EXACT — `unsafe_rbg` deterministic on gfx950 |
| **Everything except `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`** | **BIT-EXACT — sole required** |

### Root cause isolation (historical, pre-PR-508)

| What was changed | Result | Conclusion |
|---|---|---|
| Cached XLA compilation (same kernels) | DIFFER | Not compilation |
| hipBLASLt disabled | DIFFER | Not hipBLASLt |
| CK V2 backward (not V3) | DIFFER | V2 also affected |
| CK disabled (`NVTE_FUSED_ATTN=0`) | BIT-EXACT | **CK backward is the sole source** |
| CK + PR #508 deterministic variant | **BIT-EXACT** | **PR #508 dispatches the existing `_deterministic` kernel** |

Root cause: TE hardcoded `deterministic=false` at three call sites in `fused_attn.cpp` (lines 491, 680, 866). CK already had `_deterministic` variants in `libmha_bwd.so` — they just weren't dispatched. PR #508 fixes the three lines.

### MoE-specific findings

| Test | Result |
|---|---|
| MoE with `xla_gpu_deterministic_ops=true` | **152× throughput collapse** (264 sec/step at 1.6 TFLOP/s) |
| MoE without `xla_gpu_deterministic_ops` (det-fixed) | BIT-EXACT, 229 TFLOP/s @ max-pdbs=6 (12% overhead vs nondet 260) |
| MoE sparse_matmul, all det flags | BIT-EXACT, 0% det overhead (zero atomicAdd in HLO) |
| MoE non-det 3-run divergence | Matches dense Exp B pattern (diverges from step 2, max-delta ~3e-4 at step 49) |
| HLO diff DET vs NONDET MoE | +113 scatter-serialization while-loops in DET; TE custom_calls (80+53) and `__cublas$lt$matmul` (207) identical |

### hipBLASLt GSU trigger test (historical)

Designed to force low CU occupancy where atomic-GSU would be selected:

| Config | HIPBLASLT_DETERMINISTIC | Runs | Result |
|---|---|---|---|
| ds-proxy-se0-e256-h4096, bs=1, seq=2048, topk=1 | `1` (patch active) | 3 | BIT-EXACT (`ada48a7345ff4e4a`) |
| Same config | `0` (patch disabled) | 3 | BIT-EXACT (`ada48a7345ff4e4a`) |

Conclusion: gfx950 / ROCm 7.1.1 BF16 solution library contains no atomic-GSU solutions. hipBLASLt is deterministic by default on this platform. The `HIPBLASLT_DETERMINISTIC` patch is a no-op (kept defensively).

## Performance

### Dense (llama2-70B, bs=8 unless noted)

| Path | Batch | TFLOP/s/dev | MFU | Tokens/s/dev | vs det |
|---|---|---|---|---|---|
| CK deterministic (max-pdbs) | 11 | **903** | 36% | — | baseline |
| CK deterministic | 8 | 895 | 35.8% | 2,094 | baseline @ bs=8 |
| CK non-deterministic (max-pdbs) | 16 | 958 | 38% | — | 1.06× faster |
| Pre-fix unfused (`NVTE_FUSED_ATTN=0`) | 1 (forced OOM at bs=8) | 100 | 4% | 233 | **9.0× slower** |

### MoE (ds-proxy-e128-h2048, bs as noted)

| Path | bs | TFLOP/s/dev | vs nondet |
|---|---|---|---|
| Standard, det-fixed (no `xla_gpu_deterministic_ops`) | 6 (max-pdbs) | **229** | 0.88× (12% overhead) |
| Standard, nondet | 7 (max-pdbs) | 260 | 1.00× |
| **Standard, det with `xla_gpu_deterministic_ops=true`** | 4 | **1.6** | **0.007× (152× slower)** |
| sparse_matmul, det | 1 (max-pdbs; OOM at bs=2) | 18.2 | 1.00× (~0% overhead) |

### MoE (ds-proxy-e256-h2048)

| Path | bs | TFLOP/s/dev |
|---|---|---|
| sparse_matmul, det (`remat_policy=full`) | 1 | 8.1 |

## Memory cost — closed-form formula

```
extra_bytes = (ceil(s_kv / kN0) - 1) × num_heads_q × (b × s_q) × head_dim_qk × 4
              kN0 = 128 if d_qk ≤ 128 else 64
```

Validated against 13 TE unit-test configs (matches measured peak alloc to within 0.25 MB).
On llama2-70B: ~9.6 GB/pdbs det vs ~6.6 GB/pdbs non-det → ~3 GB/pdbs det premium →
loss of 5 pdbs of headroom (max-pdbs det=11 vs non-det=16).

## Key Files

| File | Purpose |
|---|---|
| `train_env.sh` | Shell-level `DETERMINISTIC_MODE` block |
| `utils/deterministic.py` | Runtime patches, env verification, loss checksum |
| `utils/mfu_tracker.py` | Training entry point — imports `deterministic.py` |
| `skills/deterministic-training/SKILL.md` | Overview, results, gaps |
| `skills/deterministic-training/technical-reference.md` | Per-fix deep dives |
| `skills/deterministic-training/upstream-action-items.md` | What to file upstream |
| `skills/deterministic-training/compare_runs.py` | Full-precision loss comparison (TensorBoard) |
| `skills/deterministic-training/ablation_test.sh` | Automated ablation harness (pre-PR-508 era; see header) |
| `skills/deterministic-training/ci_determinism_test.sh` | CI smoke test (2-run comparison) |
| `skills/deterministic-training/Dockerfile` | Reproducible container build (historical; new image is `maxtext-v26.2-det-te508-aot`) |
| `skills/deterministic-training/hipblaslt-deterministic.patch` | 10-line hipBLASLt patch (no-op on gfx950) |
| `skills/deterministic-training/upstream-bugs/*.md` | Filed/draft bug reports |
| `deterministic-proj/harness/reports/` | Full phase-by-phase validation record |

## What's NOT Tested Yet

| Area | What's missing | Why it matters |
|---|---|---|
| 8-node bit-exact on new image | True distributed multirank run on `maxtext-v26.2-det-te508-aot` | Cross-host single-step is verified; 8-node distributed RCCL on the new image is the next coverage milestone |
| llama3.1-405B | Larger GEMMs, heavier RCCL | Likely fine — hipBLASLt proven det on all gfx950 BF16 shapes |
| FP8 quantized | Different GEMM kernels | Unknown non-determinism profile; Q3-2026 |
| MoE megablox=True | Block-sparse MoE GEMM path | Different kernel; needs verification |
| MoE DeepEP dispatch | Specialized AllToAll bypassing RCCL | Untested |
| MoE capacity_factor < 1 | Token-overflow drop | Drop-order determinism not separately tested |
| Real data pipelines | Shuffling, I/O ordering | Needs `data_shuffle_seed` study |
| Checkpoint roundtrip | Save/restore determinism | Blocked on MaxText API issue |
| Per-collective-op RCCL | AllReduce vs AllGather vs ReduceScatter individually | Verified in aggregate |

## The One Thing That Mattered

Three lines in [`transformer_engine/common/fused_attn_rocm/fused_attn.cpp`](https://github.com/ROCm/TransformerEngine/blob/8943023/transformer_engine/common/fused_attn_rocm/fused_attn.cpp) — lines 491, 680, 866:

```cpp
// Before (TE 2.8.0.dev0 and earlier)
- /*deterministic=*/false,
+ /*deterministic=*/deterministic,
// After (TE 2.12.0.dev0+8943023d, PR #508)
```

CK already shipped `_deterministic` kernel variants for years. TE just never passed the flag down. The fix is three lines. Throughput cost: 9.7× → 1.06×.

**Recipe to use it:** `./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1` on image `rocm/jax-training:maxtext-v26.2-det-te508-aot`.
