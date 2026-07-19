# XLA — `ScatterDeterminismExpander` causes 152× MoE throughput collapse; add MoE-aware fast path

## Problem

`xla_gpu_deterministic_ops=true` triggers the `ScatterDeterminismExpander` HLO pass
(`xla/service/scatter_determinism_expander.cc`), which rewrites every `scatter` and
`scatter-add` op into a sequential `while` loop:

```
// If deterministic operations are required, rewrite Scatter operations
// to use a while-loop that sequentially processes each scatter index.
```

For Mixture-of-Experts (MoE) models this is catastrophic. Each MoE routing layer has
2 scatter ops (token dispatch + expert-output combine) with `trip_count` equal to the
scatter element count:

```
trip_count = seq_len × num_experts_per_tok = 4096 × 2 = 8192   (for our model)
```

Across 64 decoder layers × fwd + remat-fwd + bwd (≈ 3-4 instances per layer) × 8192
sequential iterations per loop, **a single training step performs ~1M sequential scalar
HBM read-modify-write ops**. Each iteration is one element; there is no GPU parallelism
within the loop body.

## Observed Behavior

Measured on AMD MI355X (gfx950), ROCm 7.1.1, JAX 0.8.2, ds-proxy-e128-h2048 (128 experts,
top-2, 64 layers), `per_device_batch_size=4`:

| Mode | sec/step | TFLOP/s/dev | vs nondet |
|---|---|---|---|
| det with `xla_gpu_deterministic_ops=true` | **264.5** | **1.6** | **0.007× (152× slower)** |
| det without (control, all other det flags) | 1.90 | 219 | 0.92× (8% overhead) |
| nondet baseline | 1.74 | 240 | 1.00× |

HLO dump diff confirms the cause:
- DET HLO: 618 while-loops, +113 are `op_name="layers/scatter"` / `"layers/scatter-add"`
  with `backend_config={"known_trip_count":{"n":"8192"}}`
- NONDET HLO: 505 while-loops, none scatter-serialized
- TE custom_calls (`te_fused_attn_forward_ffi` 80, `te_fused_attn_backward_ffi` 53) and
  `__cublas$lt$matmul` (207) are **byte-identical** between DET and NONDET. The bug is
  purely the scatter serialization, not custom-call decomposition.

## Why MoE scatter is safe without serialization

On gfx950 MoE routing, each token is routed to a *unique* expert slot via
`argmax + one_hot + cumsum`. The destination indices are guaranteed conflict-free
by construction — no two tokens write to the same `dispatch_buffer` slot, and no two
expert outputs are combined into the same destination. There is no `atomicAdd` race to
defend against.

We verified this empirically: ds-proxy-e128-h2048, two independent 200-step runs without
`xla_gpu_deterministic_ops`, all `DETERMINISTIC_MODE=1` defensive flags: **BIT-EXACT all
200 steps, max_delta=0.0**. Same for ds-proxy-e128-h2048 with `sparse_matmul=True
shardy=True`: 200 steps BIT-EXACT, HLO contains **zero `atomicAdd`** in the entire
compiled train step.

So the sequential `while`-loop rewrite is pure overhead — it provides no determinism
benefit on this scatter pattern.

## The existing prefix-scan alternative

XLA's prefix-scan replacement
(`xla_gpu_enable_scatter_determinism_expander`, PRs
[#17886](https://github.com/openxla/xla/pull/17886) +
[#19275](https://github.com/openxla/xla/pull/19275)) is **still default=false** in upstream
XLA as of 2026-05 due to bug history:

| Date | Event |
|---|---|
| 2024-11-15 | PR #19275 merged, `xla_gpu_enable_scatter_determinism_expander` default=true |
| 2024-11-16 | Google internal test failure → [`badb11c`](https://github.com/openxla/xla/commit/badb11c) sets default=false |
| 2024-11-19 | [PR #19429](https://github.com/openxla/xla/pull/19429): bug fix + re-enable attempt; `@akuegel` finds more bugs same day; fix merged WITHOUT re-enable |
| Present | Still default=false |

We tested manually enabling it on ds-proxy-e128-h2048:

| pdbs | Result |
|---|---|
| 1 | **OOM** (sort buffer ~16 GB) |
| 2 | OOM |
| 4 | OOM |

The prefix-scan approach requires sort buffers proportional to the number of scatter
elements. For 128-expert × 4096 seq × top-2 = 8192 elements per layer × 64 layers ×
fwd+bwd, the cumulative sort buffer exceeds available HBM even at the smallest batch.

## Proposed Fix

Two-part:

### 1. MoE-aware conflict-free fast path in `ScatterDeterminismExpander`

Detect when scatter indices are statically guaranteed to be conflict-free, and **skip
the sequential while-loop rewrite entirely** in that case. Bit-exactness is preserved
because the underlying scatter is already deterministic by construction.

Detection heuristics (any one is sufficient):

- **Indices produced by `argmax + one_hot + cumsum` pattern.** This is the canonical MoE
  routing idiom in JAX. Match the HLO sub-graph and propagate a `conflict_free` attribute
  to the scatter op.
- **Indices producing a permutation of `[0, N)` in their output range.** Can be statically
  verified via integer-range analysis when the index source is provable.
- **User annotation.** Allow `jax.lax.scatter` to accept a `unique_indices=True` parameter
  (similar to `numpy.add.at` semantics) that propagates as an HLO attribute. JAX-side
  changes needed; the XLA-side change is opt-in.

In any of these cases, the pass replaces the rewrite with an identity pass-through.

### 2. Memory-bounded prefix-scan for cases that *do* need it

For genuinely conflict-prone scatters (e.g. embedding-gradient scatter where multiple
tokens may share a vocab index), chunk the sort buffer to bound memory. Trades a small
amount of perf for fitting in HBM at large MoE sizes. This re-enables
`xla_gpu_enable_scatter_determinism_expander` as a viable default.

## Expected Behavior After Fix

With `xla_gpu_deterministic_ops=true` set on an MoE model:

- The MoE routing scatter ops are detected as conflict-free.
- No while-loop rewrite is emitted.
- Training throughput is identical (within autotune noise) to the case without the flag.
- Bit-exactness is preserved (already guaranteed by the conflict-free property).

Customers can then set `xla_gpu_deterministic_ops=true` as a defensive flag on MoE without
fear, mirroring how it behaves on dense models today.

## Environment

- AMD MI355X (gfx950), ROCm 7.1.1, JAX 0.8.2, rocm-jax fork (`rocm/jax-training:maxtext-v26.2-det-te508-aot`)
- Model: ds-proxy-e128-h2048 (128 experts, top-2, 64 layers); ds-proxy-e256-h2048 (256
  experts, top-4)
- HLO dumps: `/tmp/hlo_dumps/{moe_det,moe_nondet,sparse_det}/` (preserved for repro)

## Impact

- **MoE customer experience.** The current state requires architecture-specific
  knowledge to know that `xla_gpu_deterministic_ops` is harmful on MoE. The flag-name
  itself implies safety. A documentation note is insufficient; the right fix is for the
  pass to know better.
- **Cross-vendor parity.** Other vendors set this flag defensively without consequence on
  their stack. AMD MoE users hit a non-obvious 152× cliff. This breaks the contract that
  "deterministic-mode flags compose safely."
- **Currently mitigated by:** documentation (the AMD ROCm deterministic-training skill
  set; harness `reports/2026-05-06_moe_scatter_serialization.md`) and *not* setting the
  flag in our `train_env.sh DETERMINISTIC_MODE` block. But this is fragile — any user
  who learns about the flag elsewhere will set it and hit the cliff.

## References

- Internal investigation report:
  `deterministic-proj/harness/reports/2026-05-06_moe_scatter_serialization.md`
- HLO evidence: `/tmp/hlo_dumps/`
- Prefix-scan expander history:
  - [`xla#17886`](https://github.com/openxla/xla/pull/17886)
  - [`xla#19275`](https://github.com/openxla/xla/pull/19275)
  - [`xla#19429`](https://github.com/openxla/xla/pull/19429)
  - [`xla commit badb11c`](https://github.com/openxla/xla/commit/badb11c)
- AMD ROCm deterministic-training skill set:
  `skills/deterministic-training/SKILL.md` "The xla_gpu_deterministic_ops warning" section
