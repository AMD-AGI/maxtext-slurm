# TE Fused attention backward does not respect deterministic mode

## Problem

`nvte_fused_attn_bwd`, `nvte_fused_attn_bwd_kvpacked`, and `nvte_fused_attn_bwd_qkvpacked` in `transformer_engine/common/fused_attn_rocm/fused_attn.cpp` hardcode `deterministic=false` when calling the CK backend, ignoring the caller's `deterministic` parameter:

```cpp
// Line 866 (also lines 491 and 680)
fused_attn_ck_bwd(
  ...
  false, // TODO: enable deterministic after CK team show us how
  ...
```

Setting `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` correctly propagates `deterministic=True` from Python through the C++ FFI layer, but the value is discarded at this call site. The CK `_ndeterministic` kernel variant (which uses `atomicAdd` for dQ gradient accumulation) is always dispatched. The `_deterministic` variant (per-split dQ buffers + fixed-order reduction) is compiled into `libmha_bwd.so` but never used.

## Observed Behavior

Running TE fused attention backward 5 times as separate processes with `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (bf16, batch=2, seq=1024, heads=8, head_dim=128, gfx950):

- **dQ**: 5 different checksums across 5 runs — non-deterministic
- **dK**: identical checksum across all 5 runs — deterministic

This is expected given the bug: dQ uses cross-block `atomicAdd` (non-deterministic), while dK is accumulated in-register within a single thread-block (deterministic by construction).

With `NVTE_LOG_CK_CONFIG=1`, the CK dispatch log confirms the flag is dropped:

```
deterministic: 0     ← always 0 regardless of NVTE_ALLOW_NONDETERMINISTIC_ALGO
```

## Expected Behavior After Fix

With `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`:

- `NVTE_LOG_CK_CONFIG=1` should show `deterministic: 1` in the CK dispatch log
- dQ checksums should be identical across separate process runs
- Workspace size should increase (deterministic mode requires `nsplits` copies of the dQ accumulator)

## Environment

- Docker image: `rocm/jax-training:maxtext-v26.2`
- TransformerEngine 2.8.0.dev0+aec00a7f
- ROCm 7.1.1, JAX 0.8.2, GPU: 8× gfx950 (MI355X)

## Impact

This is the sole blocker for deterministic training with CK fused attention on ROCm. The current workaround (`NVTE_FUSED_ATTN=0`) disables CK entirely and falls back to unfused JAX attention, causing ~9.7x throughput loss (968 → 100 TFLOP/s/device) due to OOM-forced batch reduction.
