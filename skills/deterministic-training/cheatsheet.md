# Deterministic Training — Cheatsheet

Quick reference for all flags, settings, and concepts tested in this project.

## One-Liner

```bash
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 per_device_batch_size=1
```

## Env Vars Set by DETERMINISTIC_MODE=1

| Env Var | Value | What It Does | Needed? |
|---------|-------|-------------|---------|
| `NVTE_FUSED_ATTN` | `0` | Disables CK fused attention, falls back to JAX native (matmul+softmax) | **YES** — sole required setting |
| `--xla_gpu_deterministic_ops` | `true` | Disables autotuning, deterministic scatter/conv/fMHA | Defensive — Docker `autotune_level=0` already covers it |
| `TF_DETERMINISTIC_OPS` | `1` | Disables rocBLAS atomic reductions + MIOpen det conv | Defensive — most GEMMs use hipBLASLt not rocBLAS |
| `HIPBLASLT_DETERMINISTIC` | `1` | Filters atomic-GSU TensileLite solutions | Defensive — standard shapes don't select atomic-GSU |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | `0` | TE selects "deterministic" CK algorithm variants | Defensive — redundant when CK is disabled |
| `NCCL_ALGO` | `Ring` | Pins RCCL collectives to Ring algorithm | Defensive — not needed single-node or 2-node |
| `RCCL_MSCCLPP_ENABLE` | `0` | Disables MSCCL++ (bypasses NCCL_ALGO) | Defensive — not needed single-node or 2-node |
| `JAX_DEFAULT_PRNG_IMPL` | `threefry2x32` | Deterministic PRNG (overrides MaxText's `unsafe_rbg`) | Defensive — `unsafe_rbg` is actually det on ROCm/gfx950 |

## Concepts

| Concept | Meaning |
|---------|---------|
| **CK (Composable Kernel)** | AMD's GPU kernel library for fused operations. CK fused attention computes Q*K->softmax->*V in one kernel without materializing the full attention matrix. |
| **Fused vs Unfused Attention** | Fused: single kernel, O(seq_len) memory, fast (~968 TFLOP/s). Unfused: 3 separate ops (2 GEMMs + softmax), O(seq_len^2) memory, slow (~100 TFLOP/s). |
| **atomicAdd** | GPU instruction that adds to a memory location atomically. Thread scheduling determines the order of additions, producing different FP results each run. |
| **ULP (Unit in Last Place)** | Smallest representable difference at a given float magnitude. 46 ULPs at step 2 = the first divergence seed. |
| **Ablation** | Remove one setting at a time from a working config to test if it's necessary. |
| **Defensive flag** | A setting that's not needed on the current platform but protects against edge cases (different models, multi-node, future software). |
| **TensileLite** | hipBLASLt's GEMM kernel selection engine. `DeterministicMode` filters solutions that use atomic GlobalSplitU. |
| **GlobalSplitU (GSU)** | TensileLite technique that splits GEMM K-dimension across thread groups. Atomic GSU uses atomicAdd to accumulate — non-deterministic. MultipleBuffer GSU writes to separate buffers then reduces in order — deterministic. |
| **MSCCL++** | Microsoft's alternative collective communication library inside RCCL. Has its own algorithm selection that bypasses `NCCL_ALGO`. |
| **threefry2x32** | JAX's deterministic PRNG — cryptographic hash-based counter. Same seed always produces same sequence. |
| **unsafe_rbg** | JAX's fast PRNG — uses XLA RngBitGenerator. Documented as "not guaranteed deterministic across backends." Empirically deterministic on ROCm/gfx950. |

## What Was Tested

### Models
| Model | Type | Params | Steps | Nodes | Result |
|-------|------|--------|-------|-------|--------|
| llama2-70b | Dense | 68.977B | 500 | 1 | BIT-EXACT (checksum `b4235c1ffafada08`) |
| llama2-70b | Dense | 68.977B | 50 | 1 | BIT-EXACT (checksum `637613d3a620c008`) |
| llama2-7b | Dense | 6.738B | 50 | 1 | BIT-EXACT (checksum `97bce95508892dfc`) |
| ds-proxy-se0-e256-h4096 | MoE (256 experts) | ~70B | 200 | 8 | BIT-EXACT (checksum `7ff136bdcaf72e8a`) |

### Topologies
| Config | Nodes | GPUs | Result |
|--------|-------|------|--------|
| Single-node | 1 | 8x gfx950 | BIT-EXACT |
| Multi-node | 2 (chi2870+chi2872) | 16x gfx950 | BIT-EXACT |

### Dropout
| dropout_rate | PRNG | Result |
|-------------|------|--------|
| 0.0 (no-op) | threefry2x32 | BIT-EXACT |
| 0.1 (real) | threefry2x32 | BIT-EXACT (checksum `f3c76093cf1633e2`) |
| 0.0 | unsafe_rbg | BIT-EXACT |

### Ablation (each flag removed individually)
| Removed | Result |
|---------|--------|
| `xla_gpu_deterministic_ops` | BIT-EXACT -- not needed |
| `TF_DETERMINISTIC_OPS` | BIT-EXACT -- not needed |
| `HIPBLASLT_DETERMINISTIC` | BIT-EXACT -- not needed |
| `NCCL_ALGO + RCCL_MSCCLPP_ENABLE` | BIT-EXACT -- not needed |
| `JAX_DEFAULT_PRNG_IMPL` | BIT-EXACT -- not needed |
| Everything except `NVTE_FUSED_ATTN=0` | BIT-EXACT -- sole required |

### Root Cause Isolation
| What was changed | Result | Conclusion |
|-----------------|--------|------------|
| Cached XLA compilation (same kernels) | DIFFER | Not compilation |
| hipBLASLt disabled | DIFFER | Not hipBLASLt |
| CK V2 backward (not V3) | DIFFER | V2 also affected |
| CK disabled (NVTE_FUSED_ATTN=0) | BIT-EXACT | CK backward is the sole source |
| **Root cause found** | | **TE hardcodes `deterministic=false` for CK backend** (`fused_attn.cpp` lines 491, 680, 866). CK has a working `_deterministic` kernel variant but TE never enables it. See `upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md`. |

### hipBLASLt GSU Trigger Test
Designed a config to force low CU occupancy (8 tokens/expert, 16 tiles, 5% CU util) where atomic-GSU would be selected:
| Config | HIPBLASLT_DETERMINISTIC | Runs | Result |
|--------|----------------------|------|--------|
| ds-proxy-se0-e256-h4096, batch=1, seq=2048, topk=1 | `1` (patch active) | 3 | BIT-EXACT (`ada48a7345ff4e4a`) |
| Same config | `0` (patch disabled) | 3 | BIT-EXACT (`ada48a7345ff4e4a`) |
Conclusion: gfx950 ROCm 7.1.1 BF16 solution library does not contain atomic-GSU solutions. hipBLASLt is deterministic by default on this platform. The `HIPBLASLT_DETERMINISTIC` patch is a no-op.

### Multi-node RCCL
| RCCL config | Result |
|------------|--------|
| NCCL_ALGO=Ring + RCCL_MSCCLPP_ENABLE=0 | BIT-EXACT |
| No RCCL pinning (defaults) | BIT-EXACT (same checksum) |
| Only NVTE_FUSED_ATTN=0 (no det mode) | BIT-EXACT |

## Performance

| Config | TFLOP/s/device | MFU | Tokens/s/device | Batch |
|--------|---------------|-----|-----------------|-------|
| Normal (CK fused, batch=8) | 968 | 38.8% | 2,264 | 8 |
| Deterministic (unfused, batch=1) | 100 | 4.0% | 233 | 1 |
| **Slowdown** | **9.7x** | | **9.7x** | |

Slowdown = batch reduction (8x) * unfused attention overhead (1.2x).

## Key Files

| File | Purpose |
|------|---------|
| `utils/deterministic.py` | Runtime patches, env verification, loss checksum tracker |
| `utils/mfu_tracker.py` | Training entry point — imports and delegates to `deterministic.py` |
| `train_env.sh` | Shell-level `DETERMINISTIC_MODE` block (8 env vars) |
| `SKILL.md` | Overview, results, gaps |
| `technical-reference.md` | Per-fix deep dives, code traces |
| `upstream-action-items.md` | What to fix in ROCm repos |
| `hipblaslt-deterministic.patch` | 10-line patch for hipBLASLt (no-op on gfx950/ROCm 7.1.1) |
| `compare_runs.py` | Full-precision loss comparison (TensorBoard) |
| `ablation_test.sh` | Automated ablation harness |
| `ci_determinism_test.sh` | CI smoke test (2-run comparison) |
| `Dockerfile` | Reproducible container build |
| `upstream-bugs/*.md` | Ready-to-file bug reports |
| `utils/test_deterministic.py` | Unit tests for `deterministic.py` and MFU stream delegation |

## What's NOT Tested Yet

| Area | What's missing | Why it matters |
|------|---------------|---------------|
| ~~MoE models~~ | ~~Scatter/gather for expert routing~~ | **DONE** — BIT-EXACT with 256 experts, 8 nodes, 200 steps |
| llama3.1-405b | Larger GEMMs, heavier RCCL | Likely fine — hipBLASLt proven deterministic on gfx950 |
| FP8 quantized | Different GEMM kernels | Unknown non-determinism profile |
| 4+ nodes | More topology variance | 8-node MoE is now verified; larger scales untested |
| Intra vs inter-node RCCL | XGMI (intra) vs InfiniBand (inter) | Different transports, different code paths in RCCL |
| Real data pipelines | Shuffling, I/O ordering | May need data_shuffle_seed pinning |
| Checkpoint roundtrip | Save/restore determinism | MaxText API issue, not a determinism issue |

## The One Thing That Matters

**TransformerEngine hardcodes `deterministic=false` when calling the CK fused attention backward** (`fused_attn.cpp` lines 491, 680, 866). CK *does* implement a working `_deterministic` kernel variant (per-split dQ buffers + fixed-order reduction instead of `atomicAdd`), but TE never enables it. The `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` flag correctly propagates `deterministic=True` from Python through the C++ FFI, but `nvte_fused_attn_bwd` discards the value and substitutes `false` with a TODO comment: `"enable deterministic after CK team show us how"`.

**Fix**: Change `false` to `deterministic` at three call sites in `transformer_engine/common/fused_attn_rocm/fused_attn.cpp`. See `upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md`.

**Workaround**: `NVTE_FUSED_ATTN=0` (disables CK entirely, 9.7x throughput loss).
