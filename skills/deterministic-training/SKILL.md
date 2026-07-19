---
name: deterministic-training
description: How to run bit-exact reproducible MaxText training on AMD ROCm GPUs. Covers the full stack from MaxText → JAX → XLA → Transformer Engine → CK fused attention → hipBLASLt → rocBLAS → RCCL, including the MoE scatter-serialization warning.
---

# Deterministic Training on AMD ROCm

Make MaxText training runs reproducible across repeated executions on the same hardware.

> **Status (2026-05):** With image `rocm/jax-training:maxtext-v26.2-det-te508-aot`
> (ships [ROCm/TransformerEngine PR #508](https://github.com/ROCm/TransformerEngine/pull/508)),
> deterministic training is bit-exact at production batch with **~6% throughput overhead**
> on llama2-70B. The historical 9.7× `NVTE_FUSED_ATTN=0` workaround is no longer needed.

## Usage

**Bit-exact deterministic** — append `_env_DETERMINISTIC_MODE=1`:

```bash
./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1
```

This sets `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (the only flag that actually matters on this
stack — see Ablation below) and a small set of defensive flags for hipBLASLt, rocBLAS, RCCL,
and JAX PRNG. CK fused attention stays enabled (`NVTE_FUSED_ATTN=1`, `NVTE_FUSED_ATTN_CK=1`)
and runs the `_deterministic` kernel variant. No batch reduction needed.

**Turn off** — omit the flag (default is off):

```bash
./submit.sh llama2-70b
```

**Performance note.** CK deterministic backward uses per-split `dQ_accum` buffers and an
ordered `convert_dQ` reduce kernel instead of `atomicAdd`. On llama2-70B at max-pdbs the gap
is **958 → 903 TFLOP/s/dev (~6%)**; on the e128 MoE standard path it's **~12%**; on the e128
MoE `sparse_matmul=True` path it's **~0%** (zero atomicAdd in the compiled HLO). The trade-off
is a per-split workspace: see closed-form formula in [technical-reference.md](technical-reference.md).

## Quick Reference

| Source of Non-Determinism | Impact | Mitigation | Required? |
|---|---|---|---|
| **CK fused attention backward** (`atomicAdd` on `dQ_accum`) | **HIGH** | `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` → TE dispatches `_deterministic` CK kernel (PR #508 wires the flag through) | **REQUIRED — sole functional cause of non-determinism on this stack** |
| hipBLASLt atomic-GSU solutions | LOW | `HIPBLASLT_DETERMINISTIC=1` | Defensive — gfx950/ROCm 7.1.1 BF16 has no atomic-GSU solutions; verified bit-exact even at 5% CU util (8 tokens/expert) |
| rocBLAS atomic reductions | MEDIUM | `TF_DETERMINISTIC_OPS=1` | Defensive — hipBLASLt handles GEMMs; rocBLAS only on fallback |
| RCCL collective order | LOW-MED | `NCCL_ALGO=Ring` + `RCCL_MSCCLPP_ENABLE=0` | Defensive — bit-exact without pinning on single-node and 2-node; pin for safety at larger scale |
| JAX PRNG (`unsafe_rbg`) | MEDIUM | `JAX_DEFAULT_PRNG_IMPL=threefry2x32` via `utils/deterministic.py` monkey-patch | Defensive — `unsafe_rbg` is empirically deterministic on gfx950 but JAX warns it may not be |
| XLA GEMM autotuning | MEDIUM | `--xla_gpu_autotune_level=0` | Already set by Docker image; `DETERMINISTIC_MODE` re-asserts it defensively |
| **XLA scatter determinism** (`--xla_gpu_deterministic_ops=true`) | — | **DO NOT SET** | **HARMFUL on MoE.** See [MoE warning](#the-xla_gpu_deterministic_ops-warning-moe-only) below. Not set by `DETERMINISTIC_MODE`. |

## The `xla_gpu_deterministic_ops` warning (MoE only)

`--xla_gpu_deterministic_ops=true` activates XLA's `ScatterDeterminismExpander`, which
rewrites every scatter/scatter-add op into a sequential while-loop. On llama2-70B (dense, zero
scatter ops) this is a no-op. **On MoE models it is a 152× throughput collapse**: each routing
layer has 2 scatter ops × `trip_count = seq_len × top_k = 8192` iterations × 64 decoder layers
× fwd+bwd ≈ **~1M sequential scalar ops per step** (e.g. 264 sec/step at 1.6 TFLOP/s, vs 1.74 sec/step
at 240 TFLOP/s without the flag).

`DETERMINISTIC_MODE` deliberately **does not** set this flag. MoE scatter ops on gfx950 are
deterministic by construction (each token routes to a unique expert slot; no `atomicAdd` race) —
verified by 200-step dual-run on e128 standard path and 200-step on e128 sparse_matmul.

Full root-cause analysis and HLO evidence:
[`deterministic-proj/harness/reports/2026-05-06_moe_scatter_serialization.md`](../../deterministic-proj/harness/reports/2026-05-06_moe_scatter_serialization.md).

## Architecture: where non-determinism enters

```
MaxText train_step
  │
  ├─ Model forward pass
  │   ├─ Attention (TE DotProductAttention) ─► CK fused kernel ──► PR #508: `_deterministic`
  │   │                                                              variant dispatched when
  │   │                                                              NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
  │   ├─ Linear layers (dense)           ─► XLA ─► hipBLASLt   ──► HIPBLASLT_DETERMINISTIC=1
  │   ├─ MoE routing (dispatch+combine)  ─► XLA scatter ops    ──► deterministic by construction
  │   │                                                              on gfx950 (no atomicAdd race);
  │   │                                                              DO NOT enable xla_gpu_deterministic_ops
  │   ├─ RMSNorm           ─► pure JAX ─► race-free XLA reductions
  │   ├─ Activations (SiLU)─► element-wise
  │   └─ Dropout           ─► JAX PRNG ─► threefry2x32 (via deterministic.py)
  │
  ├─ Loss: sum(cross_entropy) ─► race-free reductions
  │
  ├─ Backward pass (mirrored)
  │   └─ Attention dQ ─► CK kernel writes to per-split dQ buffers,
  │                       then convert_dQ kernel reduces in static
  │                       block-index order (the PR #508 path)
  │
  ├─ Gradient clipping ─► race-free reductions
  ├─ AdamW             ─► element-wise
  │
  └─ Gradient sync
      ├─ AllReduce       ─► RCCL ─► NCCL_ALGO=Ring
      ├─ AllGather       ─► RCCL ─► NCCL_ALGO=Ring
      ├─ ReduceScatter   ─► RCCL ─► NCCL_ALGO=Ring
      └─ AllToAll (MoE)  ─► RCCL ragged_all_to_all
```

## What IS deterministic by construction (audit)

| Component | Why deterministic |
|---|---|
| Synthetic data | Fixed `PRNGKey(0)`, generated once |
| Weight init | Fixed via `config.init_weights_seed` |
| Learning-rate schedule | Pure function of `step` + config |
| XLA reductions (softmax, sum, logsumexp, global norm) | `TreeReductionRewriter` splits to race-free — no atomics |
| RMSNorm | Pure JAX (`jnp.mean` + `lax.rsqrt`) → race-free XLA reductions |
| Cross-entropy loss | Element-wise + reduction (`custom_vjp`, no scatter) |
| AdamW optimizer | Element-wise (EMA, weight decay, LR scaling) |
| Activation functions (SiLU, GELU) | Element-wise |
| Triton GEMM | Disabled (`--xla_gpu_enable_triton_gemm=False`) |
| MoE token dispatch / combine | XLA scatter/scatter-add — conflict-free by construction on gfx950 (each token → unique slot) |
| `convert_dQ` post-CK reduction | Fixed block-index order (not runtime arrival order) |
| Remat recomputation | `scan_layers=True` → `prevent_cse=False`, forward computed once |

## Experimental evidence

### Headline result (post-PR-508)

| Model | Type | Steps | Nodes | Image | Result | Checksum |
|---|---|---|---|---|---|---|
| llama2-70B | Dense | 1000 | 1 | det-te508-aot | **BIT-EXACT** float32 every step | `e352e66f43134c70` |
| llama2-70B | Dense | 200 | 1 | det-te508-aot | **BIT-EXACT** at max-pdbs=11 | `1d857e6d219848c5` |
| ds-proxy-e128-h2048 (MoE, 128 experts) | MoE std | 200 | 1 | det-te508-aot | **BIT-EXACT** at pdbs=6 | — |
| ds-proxy-e128-h2048 (MoE, 128 experts) | MoE sparse_matmul | 200 | 1 | det-te508-aot | **BIT-EXACT** at bs=1 | `8ba4d5eb7405f661` |
| ds-proxy-e256-h2048 (MoE, 256 experts top-4) | MoE sparse_matmul | 50 | 1 | det-te508-aot | **BIT-EXACT** at bs=1 (`remat_policy=full` required) | `ee0bf8f94c025a5a` |
| llama2-70B | Dense + dropout=0.1 | 50 | 1 | (prior image) | BIT-EXACT | `f3c76093cf1633e2` |
| ds-proxy-e128-h2048 | MoE std + dropout=True | 50 | 1 | det-te508-aot | **BIT-EXACT** all 50 steps | — |
| llama2-70B | Dense, cross-host | 50 | 1 (chi2868↔chi2774) | det-te508-aot | **BIT-EXACT**, max_delta=0.0 | `050fb73070bb9d91` |
| llama2-70B | Dense, grad-accum | 50 | 1 (`pdbs=4 gas=2`) | det-te508-aot | **BIT-EXACT** | — |
| llama2-70B | Dense, seq=2048 | 50 | 1 | det-te508-aot | BIT-EXACT (`pdbs=8`) | `d5654b0dc37ec0f4` |
| llama2-70B | Dense, seq=8192 | 50 | 1 | det-te508-aot | BIT-EXACT (`pdbs=2`) | — |
| ds-proxy-se0-e256-h4096 | MoE 256 experts | 200 | 8 | (prior image) | BIT-EXACT | `7ff136bdcaf72e8a` |

### Failure-mode demo (what non-determinism looks like)

Three independent llama2-70B runs with `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1` (the default):

| Step | max-delta across 3 runs |
|---|---|
| 0–1 | 0.0 (forward-only) |
| 2 | 5.0e-5 (first backward fires the `dQ` atomicAdd race) |
| 13 | **6.7e-3** (peak divergence) |
| 49 | 9.2e-6 (final step, ~13 ULPs at this loss magnitude) |

Std-dev across 3 runs at step 49: **4.75e-6**. Visually the loss curves look identical (synthetic
data converges fast); bit-exactness is the actual difference. MoE non-det matches this pattern
(diverges from step 2, attention `dQ` is the dominant source — confirmed by HLO showing zero
atomicAdd in MoE scatter ops).

### Ablation study — single-flag removal (llama2-70B, dense, 15 steps)

Each row removes one setting from the bit-exact baseline. **Only `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` is required.**

| Removed | Result | Conclusion |
|---|---|---|
| (baseline — all `DETERMINISTIC_MODE=1` flags) | BIT-EXACT | — |
| `TF_DETERMINISTIC_OPS` | BIT-EXACT | Not needed (defensive) |
| `HIPBLASLT_DETERMINISTIC` | BIT-EXACT | Not needed on gfx950/ROCm 7.1.1 BF16 |
| `NCCL_ALGO` + `RCCL_MSCCLPP_ENABLE` | BIT-EXACT (single-node / 2-node) | Not needed at small scale (defensive at 4+ nodes) |
| `JAX_DEFAULT_PRNG_IMPL` (use `unsafe_rbg`) | BIT-EXACT | `unsafe_rbg` deterministic on gfx950 (defensive) |
| `xla_gpu_autotune_level` (let autotune run) | DIFFER | Required — but image default already sets it |
| **Everything except `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`** | **BIT-EXACT** | **Sole required setting** |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | DIFFER (~46 ULPs by step 2) | Required |

Historical note: the pre-PR-508 ablation in `ablation_test.sh` showed the same shape but used
`NVTE_FUSED_ATTN=0` as the sole required flag because PR #508 was unmerged. Both ablations
identify the CK backward as the sole source — PR #508 changes the *fix* (kernel variant vs.
disabling CK), not the *cause*.

### Performance (llama2-70B at production)

| Path | Batch | TFLOP/s/dev | MFU | vs deterministic |
|---|---|---|---|---|
| **CK deterministic (PR #508)** at max-pdbs | 11 | **903** | 36% | **baseline** |
| CK deterministic at bs=8 | 8 | 895 | 36% | baseline @ bs=8 |
| CK non-deterministic at max-pdbs | 16 | 958 | 38% | 1.06× faster |
| Pre-fix unfused workaround (`NVTE_FUSED_ATTN=0`) | 1 (forced by OOM) | 100 | 4% | **9.0× slower** |

### Memory cost — closed-form formula

The deterministic CK backward allocates `nsplits = ceil(s_kv / kN0)` per-split dQ buffers:

```
extra_bytes = (ceil(s_kv / kN0) - 1) × num_heads_q × (b × s_q) × head_dim_qk × 4
              where kN0 = 128 if d_qk ≤ 128 else 64
```

Validated against 13 TE unit-test configs (head_dim {64,128,192,256}, seq {2048,4096,8192,16384},
batch {1,2,4,8}, MHA/GQA/MQA) — formula matches `torch.cuda.max_memory_allocated()` to within
0.25 MB allocator slack. Worked example for `hd128_s2048_b2_mha`: det 320 MB vs non-det 80 MB
(4×), matches `16×8×4096×128×4 = 256 MB` extra split buffer + 64 MB other bwd tensors.

This cost is intrinsic to the split-accumulator algorithm (TE team confirmed 2026-05-06).
On llama2-70B it costs 5 pdbs of batch headroom (max-pdbs det=11 vs non-det=16) → ~31% loss.

### Long-run stability

- llama2-70B, 1000 steps, dual-run, every step bit-exact at float32 precision (loss 10.87 → 0.0000001070), checksum `e352e66f43134c70`.
- llama2-70B, 200 steps at max-pdbs=11, bit-exact, checksum `1d857e6d219848c5`.

### Loss-checksum monitor

`utils/deterministic.py` prints `[determinism] loss_checksum=<hex> (steps=N)` at exit. Identical
checksums across runs prove bit-exactness from the log alone. Known flake: the line is dropped
in ~3 of 6 runs (mfu_tracker stdout-interception bug). **TensorBoard event-file comparison via
`compare_runs.py` is the enforced source of truth** — the checksum is a convenience.

### Checkpoint roundtrip

**BLOCKED.** MaxText `load_full_state_path` heartbeat timeout during restore. Upstream MaxText
issue, not a determinism bug.

## Known gaps and roadmap

### Performance / scale

| Gap | Status | Path forward |
|---|---|---|
| ~~9.7× throughput loss (CK disabled)~~ | **RESOLVED** by PR #508 (1.06× cost at max-pdbs) | — |
| Multi-node bit-exact on new image | 2-node single-step cross-host verified (chi2868↔chi2774); true distributed multirank training on the new image not yet exercised at 8 nodes | Run 8-node bit-exact on new image (planned, see `deterministic-proj/harness/experiments/10_multi_node_new_image/`) |
| FP8 deterministic on gfx950 | Untested | Q3-2026 follow-up |
| llama3.1-405B | Untested | Different GEMM-shape regime; likely fine (hipBLASLt det on all gfx950 BF16 shapes) |
| Per-split workspace cost (memory) | Documented (closed-form formula) | TE team confirmed intrinsic; no `nsplits`-capping knob exists upstream — see [`upstream-action-items.md`](upstream-action-items.md) |

### MoE coverage

| Path | Status |
|---|---|
| Standard (capacity-factor) | ✅ BIT-EXACT e128 200 steps + dropout; 12% det overhead at max-pdbs |
| sparse_matmul=True | ✅ BIT-EXACT e128 200 steps + e256 50 steps; ~0% det overhead. e256 requires `remat_policy=full` (minimal_flash triggers ROCm `HIP_ERROR_InvalidValue` in `loop_broadcast_fusion_16`) |
| megablox=True | Not yet tested |
| DeepEP dispatch | Not yet tested |
| CUTLASS grouped GEMM (`NVTE_USE_CUTLASS_GROUPED_GEMM=1`) | Not yet tested |
| capacity_factor < 1 (token-overflow drop) | Drop-order determinism not separately tested |
| Multi-node EP > 8 on new image | Not yet tested |

### Infrastructure

| Gap | Status |
|---|---|
| Real (non-synthetic) data pipelines | `data_shuffle_seed` study pending |
| Checkpoint roundtrip | Blocked on MaxText `load_full_state_path` issue |
| CI integration | `ci_determinism_test.sh` ready; needs pipeline hook |
| Per-collective-op RCCL (AllReduce vs AllGather vs ReduceScatter) | Verified in aggregate; not isolated |

## Implementation details

The deterministic-mode Python code lives in `utils/deterministic.py` (PRNG monkey-patch,
runtime env verification, loss checksum tracker). Shell env vars are set by the
`DETERMINISTIC_MODE` block in `train_env.sh`. `utils/mfu_tracker.py` is the training entry
point that imports and delegates to `deterministic.py`.

See [technical-reference.md](technical-reference.md) for:
- Per-fix documentation (every env var, code traces, source evidence)
- CK fused attention architecture: `atomicAdd` race vs PR #508's split-accumulator + ordered reduce
- PRNG monkey-patch and runtime verification
- RCCL design notes (Ring-only, SIGSEGV history, prefix-parser limitation)
- hipBLASLt TensileLite internals (DeterministicMode predicate, solution library analysis)
- MoE scatter HLO analysis (why `xla_gpu_deterministic_ops` is toxic and how to verify)

## Source repos for deep-dive

Use the [docker-artifact-check](../docker-artifact-check/SKILL.md) skill to identify exact
commits and clone the matching source for JAX, XLA, rocm-jax, RCCL, TransformerEngine, and
MaxText. File paths in this skill are relative to each repo's root.

## Test environment

- Image: `rocm/jax-training:maxtext-v26.2-det-te508-aot` (sha256 `235574444c70327d3453639d5da8bb50759821dac46882599fde50a637e6c547`, 94.8 GB; NFS tarball at `/mnt/vast/qiangh/clean/maxtext-v26.2-det-te508-aot.tar`)
- TE: `2.12.0.dev0+8943023d` (PR #508 merge commit `8943023d654d4b89accc1cc413989547808898dd`)
- Hosts: chi2868 (single-node, primary), chi2774 (cross-host); chi2870+chi2872 (legacy 2-node, pre-PR-508)
- GPUs: 8× MI355X (gfx950) per node
- ROCm 7.1.1, RCCL 2.27.7, JAX 0.8.2
- Models: llama2-70B (68.977B), llama2-7B (6.738B), ds-proxy-e128-h2048 (MoE 128 experts top-2), ds-proxy-e256-h2048 (MoE 256 experts top-4), ds-proxy-se0-e256-h4096 (MoE 256 experts, prior multi-node baseline)
- Data: synthetic
- Validation: see `deterministic-proj/harness/reports/` for the full phase-by-phase record (Phase 0 audit → Phase 7 blog validation; Exp 08 sparse_matmul; Exp 09 coverage gaps)
