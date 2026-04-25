---
name: deterministic-training
description: Analysis of non-determinism sources in MaxText training on AMD ROCm GPUs, and how to achieve reproducible (bit-exact) training runs. Covers the full stack from MaxText → JAX → XLA → Transformer Engine → hipBLASLt → rocBLAS → RCCL.
---

# Deterministic Training on AMD ROCm

Make MaxText training runs reproducible across repeated executions on the same hardware.

## Usage

**Bit-exact deterministic** — append `_env_DETERMINISTIC_MODE=1` with reduced batch size:

```bash
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 per_device_batch_size=1
```

This disables CK fused attention (the sole source of non-determinism) and sets defensive flags for XLA, hipBLASLt, rocBLAS, RCCL, and PRNG. Requires `per_device_batch_size=1` because unfused attention materializes the full `seq_len × seq_len` matrix. Works with both `enable_dropout=True` and `False`.

**Turn off** — omit the flag (default is off):

```bash
./submit.sh llama2-70b
```

**Performance note:** Unfused attention has ~9.7x lower throughput (~100 vs ~968 TFLOP/s/device). The 9.7x comes from batch reduction (8→1, forced by OOM) plus unfused attention overhead (1.2x). `DETERMINISTIC_MODE` is designed for validation/debugging, not production training. The CK upstream fix would eliminate this penalty — see [upstream-action-items.md](upstream-action-items.md).

## Quick Reference

| Source of Non-Determinism | Impact | Mitigation | Ablation Result |
|---------------------------|--------|------------|-----------------|
| **CK fused attention backward** | **HIGH** | `NVTE_FUSED_ATTN=0` (disables CK) | **REQUIRED** — sole source of non-determinism. Root cause: TE hardcodes `deterministic=false` for CK backend (`fused_attn.cpp:866`). CK has `_deterministic` kernels but TE never enables them. |
| XLA GEMM autotuning + scatter | MEDIUM | `--xla_gpu_deterministic_ops=true` | Not needed (Docker sets `autotune_level=0`). Set defensively. |
| hipBLASLt atomic-GSU solutions | LOW | `HIPBLASLT_DETERMINISTIC=1` | Not needed — gfx950/ROCm 7.1.1 BF16 solutions have no atomic-GSU. Verified even with extreme low-occupancy MoE shapes (8 tokens/expert). |
| rocBLAS atomic reductions | MEDIUM | `TF_DETERMINISTIC_OPS=1` | Not needed (hipBLASLt handles GEMMs). Set defensively. |
| JAX PRNG (`unsafe_rbg`) | MEDIUM | `threefry2x32` via monkey-patch | Not needed — `unsafe_rbg` deterministic on ROCm/gfx950. Set defensively. |
| RCCL collective order | LOW-MED | `NCCL_ALGO=Ring` + `RCCL_MSCCLPP_ENABLE=0` | Not needed single-node or 2-node. Set defensively. |
| TE algo selection | HIGH | `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | Redundant when `NVTE_FUSED_ATTN=0`. Set defensively. |

## Architecture: Where Non-Determinism Enters

```
MaxText train_step
  │
  ├─ Model forward pass
  │   ├─ Attention (cudnn_flash_te) ─► TE ─► CK kernels ──► BLOCKER: CK bwd non-deterministic
  │   │                                                       Workaround: NVTE_FUSED_ATTN=0
  │   ├─ Linear layers (dense) ─► XLA ─► hipBLASLt ──────► HIPBLASLT_DETERMINISTIC=1 (patch)
  │   ├─ RMSNorm ─► pure JAX (jnp.mean + lax.rsqrt) ────► deterministic (race-free reductions)
  │   ├─ Activations (SiLU) ─► element-wise ─────────────► deterministic
  │   └─ Dropout ─► JAX PRNG ────────────────────────────► threefry2x32
  │
  ├─ Loss: jnp.sum(cross_entropy) ──────────────────────► deterministic (race-free reductions)
  │
  ├─ Backward pass (same ops reversed) ─────────────────► same mitigations as forward
  │   └─ Scatter (embedding grad) ───────────────────────► xla_gpu_deterministic_ops
  │
  ├─ Gradient clipping (global norm) ───────────────────► deterministic (race-free reductions)
  ├─ AdamW optimizer ───────────────────────────────────► deterministic (element-wise)
  │
  └─ Gradient sync
      ├─ AllReduce ─► RCCL ──────────────────────────────► NCCL_ALGO=Ring
      ├─ AllGather ─► RCCL ──────────────────────────────► NCCL_ALGO=Ring
      └─ ReduceScatter ─► RCCL ─────────────────────────► NCCL_ALGO=Ring
```

## What IS Deterministic (End-to-End Audit)

| Component | Why Deterministic |
|-----------|-------------------|
| Synthetic data | Fixed `PRNGKey(0)`, generated once |
| Learning rate schedule | Pure function of `step` + config |
| Weight init | Fixed via `config.init_weights_seed` |
| XLA reductions (softmax, sum, logsumexp, global norm) | `TreeReductionRewriter` unconditionally splits to race-free — no atomics, no flag needed |
| RMSNorm | Pure JAX (`jnp.mean` + `lax.rsqrt`), compiles to race-free XLA reductions |
| Cross-entropy loss | Element-wise + reduction (`custom_vjp`, no scatter) |
| AdamW optimizer | Purely element-wise (EMA updates, weight decay, LR scaling) |
| Activation functions (SiLU, GELU) | Element-wise |
| MIOpen convolutions | `MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC` set by `xla_gpu_deterministic_ops` and `TF_DETERMINISTIC_OPS` |
| Triton GEMM | Disabled (`--xla_gpu_enable_triton_gemm=False`) |
| AOTriton attention | Disabled (`NVTE_FUSED_ATTN_AOTRITON=0`) |
| Remat recomputation | `scan_layers=True` → `prevent_cse=False`, forward computed once |

## Experimental Evidence

### Ablation Study (single-node)

Each setting individually removed from the bit-exact baseline. Config: llama2-70b, 8× gfx950, `per_device_batch_size=1`, `steps=15`.

| Test | What was removed | Result |
|------|-----------------|--------|
| 0 - Baseline | Nothing (all flags) | BIT-EXACT |
| 1 | `xla_gpu_deterministic_ops=true` | BIT-EXACT → not needed |
| 2 | `TF_DETERMINISTIC_OPS=1` | BIT-EXACT → not needed |
| 3 | `HIPBLASLT_DETERMINISTIC=1` | BIT-EXACT → not needed |
| 4 | `NCCL_ALGO` + `RCCL_MSCCLPP_ENABLE` | BIT-EXACT → not needed |
| 5 | `JAX_DEFAULT_PRNG_IMPL` (use `unsafe_rbg`) | BIT-EXACT → not needed |
| 6 | **Everything** (only `NVTE_FUSED_ATTN=0`) | BIT-EXACT → sole required setting |

### Root Cause Isolation

| Experiment | Config | Result | What it proved |
|-----------|--------|--------|----------------|
| A, B, C | Full det mode + CK fused attn | DIFFER (~46 ULPs) | Non-determinism exists |
| D, E | Same, cached XLA compilation | DIFFER | NOT compilation — it's runtime |
| F, G | hipBLASLt disabled + CK attn | DIFFER | NOT hipBLASLt |
| O, P | CK V2 backward (not V3) | DIFFER | V2 also affected |
| M, N | CK disabled (`NVTE_FUSED_ATTN=0`) | BIT-EXACT | **CK backward is the sole source** |

### Multi-Model Verification

| Model | Type | Steps | Nodes | Checksum | Result |
|-------|------|-------|-------|----------|--------|
| llama2-70b | Dense | 500 | 1 | `b4235c1ffafada08` | BIT-EXACT |
| llama2-70b | Dense | 50 | 1 | `637613d3a620c008` | BIT-EXACT |
| llama2-7b | Dense | 50 | 1 | `97bce95508892dfc` | BIT-EXACT |
| ds-proxy-se0-e256-h4096 | MoE (256 experts, 32L) | 200 | 8 | `7ff136bdcaf72e8a` | BIT-EXACT |

### Multi-Node Verification (2 nodes chi2870+chi2872, 50 steps)

| Test | Config | Checksum | Result |
|------|--------|----------|--------|
| Baseline | Full `DETERMINISTIC_MODE=1` | `1ab22b7e7a83b06e` | BIT-EXACT |
| No RCCL pinning | NCCL_ALGO + RCCL_MSCCLPP cleared | `1ab22b7e7a83b06e` | BIT-EXACT |
| Only NVTE_FUSED_ATTN=0 | No other det flags | `60b9c05b0a78e95b` | BIT-EXACT |
| Dropout ON | Full det mode + enable_dropout=True | `1ab22b7e7a83b06e` | BIT-EXACT |

### Dropout Validation

- `dropout_rate=0.1` (real dropout, 10% zeroed): BIT-EXACT, checksum `f3c76093cf1633e2`
- `dropout_rate=0.0` (no-op): BIT-EXACT, different checksum — confirms dropout changes computation
- `unsafe_rbg` + `enable_dropout=False`: BIT-EXACT — `unsafe_rbg` deterministic on this platform

### Long-Duration Verification (500 steps)

Two independent runs with fresh XLA compilation caches, each 500 steps of llama2-70b on 8× gfx950. Full float32 TensorBoard comparison showed `max_delta=0.0` at every step. Loss converged from 10.87 (step 0) to 0.0000001070 (step 499) identically in both runs. Checksum: `b4235c1ffafada08`. Date: 2026-03-27.

### Loss Checksum Monitor

`utils/deterministic.py` prints `[determinism] loss_checksum=<hex> (steps=N)` at exit. Identical checksums across runs prove bit-exact determinism from the log alone.

### Checkpoint Roundtrip

**BLOCKED.** MaxText's `load_full_state_path` crashes during restore (heartbeat timeout). Needs investigation into checkpoint API, not a determinism issue.

## Known Gaps and Roadmap

### Performance
| Gap | Impact | Path Forward |
|-----|--------|-------------|
| **~9.7x throughput loss** (CK disabled) | Blocks production use | Upstream TE fix: change `false` to `deterministic` in `fused_attn.cpp` ([upstream-action-items.md](upstream-action-items.md)) |

### Model Coverage
| Gap | What it would test | Why it matters |
|-----|-------------------|---------------|
| ~~MoE models~~ | ~~XLA scatter/gather for expert routing~~ | **DONE** — ds-proxy-se0-e256-h4096 (256 experts, 8 nodes, 200 steps) verified BIT-EXACT |
| llama3.1-405b | Larger GEMM shapes (16384x53248), heavier RCCL | Likely fine — hipBLASLt proven deterministic on gfx950 for all BF16 shapes |
| FP8 quantized training | FP8 TensileLite solutions, quantize/dequantize rounding | Completely untested non-determinism profile |

### RCCL Collective Determinism
| Gap | What it would test | Why it matters |
|-----|-------------------|---------------|
| 4+ node scale | More topology variance, more ring reordering potential | 2-node may not represent larger clusters |
| Intra-node vs inter-node isolation | XGMI (intra) vs InfiniBand (inter) transports | Different RCCL code paths, different reduction ordering |
| NCCL_RINGS pinning | Fixed ring order independent of topology discovery | Strongest guarantee for multi-node, never tested |
| Per-collective-op testing | AllReduce, AllGather, ReduceScatter individually | Different ops may have different determinism profiles |
| NCCL_PROTO/NCCL_BUFFSIZE | Protocol and buffer size affect chunking/reduction order | Untested knobs |

### Infrastructure
| Gap | Impact | Status |
|-----|--------|--------|
| Real data pipelines | Shuffling/I/O non-determinism | May need data_shuffle_seed pinning |
| Checkpoint roundtrip | Can't verify save/restore determinism | MaxText checkpoint API issue |
| CI not integrated | Regressions undetected | `ci_determinism_test.sh` ready, needs pipeline hook |

## Implementation Details

The deterministic-mode Python code lives in `utils/deterministic.py` (PRNG monkey-patch, runtime verification, loss checksum tracker). The shell-level env vars are set by the `DETERMINISTIC_MODE` block in `train_env.sh`. `mfu_tracker.py` is the training entry point that imports and delegates to `deterministic.py`.

See [technical-reference.md](technical-reference.md) for:
- Per-fix documentation (8 env vars, code traces, ablation evidence)
- CK fused vs unfused attention architecture and memory math
- PRNG monkey-patch implementation
- Runtime verification system
- RCCL design notes (why Ring-only, SIGSEGV history)
- hipBLASLt TensileLite internals (DeterministicMode, solution library analysis)

## Source Repos for Deep-Dive

Use the [docker-artifact-check](../docker-artifact-check/SKILL.md) skill to identify exact commits and clone the matching source code for JAX, XLA, rocm-jax, RCCL, and MaxText. The file paths referenced in this document are relative to each repo's root.

## Test Environment

- Host: chi2882 (single-node), chi2870+chi2872 (multi-node)
- Models: llama2-70b (68.977B), llama2-7b (6.738B), ds-proxy-se0-e256-h4096 (MoE, 256 experts, ~70B)
- GPUs: 8× gfx950 (MI355X) per node
- ROCm: 7.1.1, RCCL 2.27.7, JAX 0.8.2
- Images: `rocm/jax-training:maxtext-v26.2-deterministic-v2` (single-node), `maxtext-v26.2-deterministic-final` (multi-node)
- Data: synthetic
