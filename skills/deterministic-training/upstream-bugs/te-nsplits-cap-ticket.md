# TE — Deterministic CK dQ workspace OOM on frontier models; need either `nsplits` cap with two-level ordered reduction, or guidance on max feasible model scale

## Problem

The deterministic CK fused attention backward (active when
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` and TE ≥ `2.12.0.dev0+8943023d`, i.e. after
[PR #508](https://github.com/ROCm/TransformerEngine/pull/508)) allocates
`nsplits = ceil(s_kv / kN0)` per-split `dQ_accum` buffers in HBM, where
`kN0 = 128` if `d_qk ≤ 128` else `64`. Each buffer is
`num_heads_q × (b × s_q) × d_qk × sizeof(float)` bytes.

Source: `transformer_engine/common/fused_attn_rocm/fused_attn_ck.cpp:757-784`

```cpp
size_t kN0 = (d_qk <= 128) ? 128 : 64;
size_t nsplits = deterministic ? ceil(1.0 * s_kv / kN0) : 1;
...
void* dq_acc_ptr = planner.allocate(nsplits * h * max_tokens_q * d_qk * sizeof(float));
```

For long sequences and large models the workspace is **massive** — on frontier-class
models (405B+) it exceeds the entire HBM budget at production batch sizes, making
deterministic training infeasible without reducing pdbs to 1–2.

## Observed Behavior

### Unit-test level (TE microbenchmarks)

Measured peak attention workspace (`torch.cuda.max_memory_allocated()`, gfx950 BF16):

| Config | det (MB) | non-det (MB) | ratio | extra MB |
|---|---|---|---|---|
| `hd128_s4096_b1_mha` | 576.25 | 80.25 | 7.2× | 496 |
| `hd128_s8192_b1_mha` | 1088.25 | 80.25 | 13.6× | 1008 |
| `hd128_s16384_b1_mha` | 4224.50 | 160.50 | 26.3× | 4064 |
| `hd128_s4096_b1_gqa` (32q, 8kv) | 2273.00 | 289.00 | 7.9× | 1984 |

Matches closed-form formula to within allocator slack (≤ 0.25 MB):

```
extra_bytes = (ceil(s_kv / kN0) - 1) × num_heads_q × (b × s_q) × head_dim_qk × 4
```

### End-to-end training level (OOM on frontier models)

On MI355X (192 GB HBM per device), the workspace alone is a show-stopper:

| Model | seq | pdbs | h_q | nsplits | Extra workspace | Result |
|---|---|---|---|---|---|---|
| llama2-70B | 4096 | 8 | 64 | 32 | 33.3 GB | **Fits** (max-pdbs det=11) |
| llama3.1-405B | 8192 | 5 | 128 | 64 | **169.1 GB** | **OOM** (SLURM job 15184) |
| llama3.1-405B | 8192 | 2 | 128 | 64 | 67.6 GB | Tight — retry pending |
| llama3.1-405B | 8192 | 1 | 128 | 64 | 33.8 GB | Should fit |
| DeepSeek-V3 671B | 4096 | 16 | 128 | 32 | **133.1 GB** | **OOM** (SLURM job 15183) |
| DeepSeek-V3 671B | 4096 | 4 | 128 | 32 | 33.3 GB | Should fit |

The 405B non-deterministic path runs fine at pdbs=5 (966 TFLOP/s/dev). The deterministic
path OOMs at the same pdbs because the extra 169 GB workspace exceeds the ~80 GB of
free HBM after model weights + activations.

At the production-training level on llama2-70B, deterministic mode loses **5 pdbs of
batch headroom** (max-pdbs det=11 vs non-det=16) — a 31% reduction in effective batch
capacity. On frontier models, the reduction is **total** — deterministic mode cannot run
at any production batch size.

## Why a naive `nsplits` cap does NOT preserve determinism

An earlier draft of this ticket proposed a simple `NVTE_CK_DETERMINISTIC_MAX_SPLITS=N`
env var that caps `nsplits` at N. **This is incorrect — a simple cap reintroduces
non-determinism:**

When `MAX_SPLITS < ceil(s_kv/kN0)`, multiple CK thread blocks must share a single
`dQ_accum` slot. They combine via `atomicAdd`, and the arrival order of SMs is
non-deterministic across runs. Therefore:

| `MAX_SPLITS` value | Mechanism | Bit-exact across runs? |
|---|---|---|
| `≥ ceil(s_kv/kN0)` (full split) | Each block has its own slot → no atomicAdd | **Yes** |
| `< ceil(s_kv/kN0)` (capped) | Multiple blocks share slots → atomicAdd within groups | **No** — SM arrival order varies |
| `1` | All blocks share one slot → equivalent to non-det path | **No** |

A simple cap is therefore **not a determinism-vs-memory trade-off** — it is a
**determinism-off switch that happens to use less memory**. The original ticket's claim
that "training remains bit-exact across runs for the same `MAX_SPLITS` value" was wrong.

## Proposed Fix (revised): two-level ordered reduction

To actually cap workspace while preserving determinism, the kernel needs a **two-level
split-K** structure:

1. **Level 1 (inter-group):** partition the KV axis into `N` groups (where `N` =
   `MAX_SPLITS`). Each group gets its own `dQ_accum` slot, as today. The `convert_dQ`
   kernel sums the N group accumulators in static order → deterministic across groups.

2. **Level 2 (intra-group):** within each group, `ceil(s_kv/kN0) / N` blocks must
   combine their partials. Instead of `atomicAdd`, use **ordered sequential reduction**:
   each block within a group writes to a small per-block sub-buffer
   (`dQ_local[group][block_within_group]`), then a second reduction pass sums them in
   static block-index order.

This is equivalent to a two-level `convert_dQ`: first reduce within each group (ordered),
then reduce across groups (ordered). Total workspace becomes:

```
workspace = N × num_heads_q × (b × s_q) × d_qk × sizeof(float)
          + ceil(s_kv/kN0) × num_heads_q × (b × s_q) × d_qk × sizeof(float)
            ↑ group-level accumulators       ↑ block-level sub-buffers (transient)
```

Wait — that's the same total as the uncapped case. The workspace saving comes from
making the block-level sub-buffers **streaming** (written and consumed within a single
reduction pass, not all live simultaneously). This requires a kernel-level change to the
CK backward, not just a parameter tweak.

### Alternative: reduce `kN0` instead

If `kN0` were increased (e.g. from 128 to 256 or 512), fewer splits are needed for the
same sequence length. This trades **per-block work granularity** for **fewer splits**:

| `kN0` | nsplits for seq=8192 | Workspace (405B pdbs=5) |
|---|---|---|
| 128 (current) | 64 | 169.1 GB |
| 256 | 32 | 83.3 GB |
| 512 | 16 | 40.3 GB |
| 1024 | 8 | 18.8 GB |

This is architecturally simpler (single-level reduction, just coarser blocks) but
requires the CK kernel to support larger tile sizes, which may hit register pressure
or occupancy limits on gfx950.

### Practical alternative: just reduce pdbs (current workaround)

Until the kernel is modified, the only way to run deterministically on frontier models
is to reduce `per_device_batch_size` until the workspace fits:

| Model | seq | Target extra ≤ 40 GB | Required pdbs |
|---|---|---|---|
| llama3.1-405B | 8192 | 33.8 GB at pdbs=1 | **pdbs=1** |
| DeepSeek-V3 671B | 4096 | 33.3 GB at pdbs=4 | **pdbs ≤ 4** |

This is the approach we are using in our validation campaign. The throughput penalty
is severe (~5–10× vs production pdbs) but determinism is preserved.

## What we're asking TE to evaluate

We see three possible directions; TE kernel team is best positioned to assess feasibility:

1. **Two-level ordered reduction** (described above) — preserves determinism at capped
   workspace. Implementation complexity: high (kernel rewrite).

2. **Larger `kN0` tile size** — reduces nsplits linearly. Implementation complexity:
   medium (tile-size parameter, but register/occupancy trade-offs).

3. **Guidance on the maximum feasible model scale for deterministic mode** — if neither
   (1) nor (2) is practical near-term, document the memory ceiling explicitly so
   customers can plan. Suggested additions to `envvars.rst`:
   - The closed-form workspace formula (already in this ticket)
   - A table of nsplits × workspace for common model shapes
   - A recommendation: "For models with `extra_bytes > X% of HBM`, reduce
     `per_device_batch_size` until workspace fits, or disable deterministic mode."

Any of these would be a meaningful improvement over the current state where frontier
model users silently OOM with no diagnostic or workaround guidance.

## Environment

- Docker image: `rocm/jax-training:maxtext-v26.2-det-te508-aot`
- TransformerEngine `2.12.0.dev0+8943023d` (post-PR #508)
- ROCm 7.1.1, JAX 0.8.2, GPU: 8× gfx950 (MI355X)
- Validated against 13 TE unit-test configs (`te_test/`: head_dim {64,128,192,256}, seq
  {2048,4096,8192,16384}, batch {1,2,4,8}, MHA/GQA/MQA, three mask types)

## Impact

**Deterministic training is currently infeasible on frontier-class models at production
batch sizes.** This is the single largest gap in the deterministic-training story on
ROCm. The 70B-class models that we validated successfully represent the ceiling of what
the current kernel can handle — anything larger (405B, 671B, or longer sequences on 70B)
hits the workspace wall.

The TE team has confirmed (private correspondence, 2026-05-06) the underlying
split-accumulator approach is by-design and the `convert_dQ` cost grows linearly with
`nsplits`. A kernel-level change is needed to break the linear scaling.

## References

- Validation harness: `deterministic-proj/harness/reports/2026-05-05_phase6_te_unit_test.md`
- TE team confirmation thread: noted in `harness/MEMORY.md` 2026-05-06 entry
- PR #508 (the determinism wiring fix this builds on): https://github.com/ROCm/TransformerEngine/pull/508
- Phase 9 OOM evidence: SLURM jobs 15183 (671B, 504 GiB), 15184 (405B, 362 GiB), 15185 (671B global-attn, OOM)
- Workspace calculations: `harness/reports/2026-05-11_post_pr508_validation_plan.md`
