# Bug: TransformerEngine hardcodes `deterministic=false` for CK fused attention backward

**Target repo:** `ROCm/TransformerEngine`
**File:** `transformer_engine/common/fused_attn_rocm/fused_attn.cpp`, lines 491, 680, 866

## Summary

The `nvte_fused_attn_bwd` family of functions hardcodes `deterministic=false` when calling the CK backend, ignoring the `deterministic` parameter received from the caller. This means `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` has no effect — the CK `_ndeterministic` kernel variant is always dispatched, and dQ gradients are always accumulated via `atomicAdd`.

```cpp
// fused_attn.cpp line 866 (also lines 491 and 680)
fused_attn_ck_bwd(
  ...
  false, // TODO: enable deterministic after CK team show us how
  ...
```

CK *does* implement a correct deterministic backward path (`_deterministic` kernel variants with per-split dQ buffers + fixed-order reduction). Both variants are compiled into `libmha_bwd.so`. The deterministic path has just never been enabled because this `false` was never changed to `deterministic`.

## Proposed Fix

Three identical one-line changes in `transformer_engine/common/fused_attn_rocm/fused_attn.cpp`:

```diff
--- a/transformer_engine/common/fused_attn_rocm/fused_attn.cpp
+++ b/transformer_engine/common/fused_attn_rocm/fused_attn.cpp
@@ -488,7 +488,7 @@ void nvte_fused_attn_bwd_qkvpacked(
     fused_attn_ck_bwd_qkvpacked(
       ...
       window_size_left, window_size_right,
-      false, // TODO: enable deterministic after CK team show us how
+      deterministic,
       ...

@@ -677,7 +677,7 @@ void nvte_fused_attn_bwd_kvpacked(
     fused_attn_ck_bwd_kvpacked(
       ...
       window_size_left, window_size_right,
-      false, // TODO: enable deterministic after CK team show us how
+      deterministic,
       ...

@@ -863,7 +863,7 @@ void nvte_fused_attn_bwd(
     fused_attn_ck_bwd(
       ...
       window_size_left, window_size_right,
-      false, // TODO: enable deterministic after CK team show us how
+      deterministic,
       ...
```

The `deterministic` variable is already a parameter of each enclosing function — the change is purely wiring it through instead of discarding it.

## Evidence

### 1. CK dispatch log confirms flag is dropped

With `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` and `NVTE_LOG_CK_CONFIG=1`:

```
Python: is_non_deterministic_allowed=False, deterministic=True
CK log: deterministic: 0    ← flag lost
```

The Python layer correctly computes `deterministic=True`, the C++ FFI correctly extracts it, but `nvte_fused_attn_bwd` replaces it with `false` before calling the CK backend.

### 2. Workspace sized for non-deterministic mode

For `batch=1, heads=4, seq=256, head_dim=128`:
- Non-deterministic workspace (nsplits=1): `1 × 4 × 256 × 128 × 4 + LSE = 528,384 bytes`
- Deterministic workspace (nsplits=2): `2 × 4 × 256 × 128 × 4 + LSE = 1,052,672 bytes`
- Actual workspace requested: **528,384** ← non-deterministic size

### 3. dQ non-deterministic in both modes

5 separate process runs per mode, `batch=2, seq=1024, heads=8, head_dim=128`:

```
NON-DETERMINISTIC MODE (NVTE_ALLOW_NONDETERMINISTIC_ALGO=1):
  dQ=da218326678f7542  dK=edd882395b065f81
  dQ=a946174d047d7634  dK=edd882395b065f81   ← dQ differs, dK identical
  dQ=96acc3ac67c9c8cd  dK=edd882395b065f81
  dQ=3307fb7a8268c442  dK=edd882395b065f81
  dQ=16ad64c388308af8  dK=edd882395b065f81

DETERMINISTIC MODE (NVTE_ALLOW_NONDETERMINISTIC_ALGO=0):
  dQ=d3d22199fc37b5da  dK=edd882395b065f81
  dQ=08a7153df5667222  dK=edd882395b065f81   ← dQ STILL differs, dK identical
  dQ=066df7c206c23633  dK=edd882395b065f81
  dQ=14217b70da1f148b  dK=edd882395b065f81
  dQ=23f1f938ce79bf44  dK=edd882395b065f81
```

**dQ is non-deterministic regardless of the env var setting.** dK is always deterministic — consistent with the source analysis: dK has no cross-block `atomicAdd`, while dQ does.

### 4. Full stack trace of the flag

```
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
  → TE Python attention.py:929 — deterministic=True                    ✓ correct
  → JAX XLA FFI lowering — embeds deterministic=true as attribute      ✓ correct
  → C++ attention_hip.cpp:363 — extracts deterministic=true            ✓ correct
  → C++ attention_hip.cpp:640 — passes deterministic=true to NVTE API  ✓ correct
  → fused_attn.cpp:866 — HARDCODES false, ignores parameter           ✗ BUG
  → ck_fused_attn_bwd.cpp — receives false, computes nsplits=1        (consequence)
  → aiter::mha_bwd — dispatches _ndeterministic variant               (consequence)
  → CK kernel — uses atomicAdd for dQ                                 (consequence)
```

### 5. CK deterministic path verified correct (source audit)

We audited every kernel and pipeline file. The CK deterministic path is architecturally correct:

| Component | Deterministic? | Evidence |
|-----------|---------------|----------|
| Forward kernel (`fmha_fwd_kernel.hpp`) | ✓ by construction | Zero `atomicAdd`, zero cross-block reductions |
| Backward dK/dV | ✓ by construction | One thread-block per K/V tile, register accumulation |
| Backward dQ (deterministic path) | ✓ by design | `store_tile` to per-split buffer, `FmhaBwdConvertQGradKernel` sums in fixed order |
| Backward dQ (non-deterministic path) | ✗ | `atomicAdd` — always used because of this bug |
| D computation (`dot_do_o`) | ✓ by construction | Single-block elementwise + register reduction |
| Pipeline inner loop | ✓ | All GEMMs, softmax, masking within single block |
| Compiled variants | ✓ | 8,208 `_deterministic` + 8,184 `_ndeterministic` gfx950 in `libmha_bwd.so` |
| Workspace calculation (`fused_attn_ck.cpp:867`) | ✓ | `nsplits = deterministic? ceil(s_kv/kN0):1` — correct but receives `false` |

## Environment

- TransformerEngine 2.8.0.dev0+aec00a7f (from `rocm/jax-training:maxtext-v26.2`)
- ROCm 7.1.1, RCCL 2.27.7
- JAX 0.8.2 / jaxlib 0.8.2.dev0+selfbuilt
- GPU: 8× gfx950 (MI355X), single-node
- Model: llama2-70b (MaxText), BF16, 8-way FSDP

## Reproduction

```python
# Minimal reproduction — run as 5 separate processes and compare dQ checksums
import os, hashlib, numpy as np
os.environ["NVTE_FUSED_ATTN"] = "1"
os.environ["NVTE_FUSED_ATTN_CK"] = "1"
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"  # request deterministic

import jax, jax.numpy as jnp
from jax import random
from transformer_engine.jax.attention import fused_attn, AttnBiasType, AttnMaskType, QKVLayout, SequenceDescriptor
from transformer_engine.jax.sharding import MeshResource, global_shard_guard

b, s, h, d = 2, 1024, 8, 128
key = random.PRNGKey(42)
q = random.normal(random.split(key, 4)[0], (b,s,h,d), dtype=jnp.bfloat16)
k = random.normal(random.split(key, 4)[1], (b,s,h,d), dtype=jnp.bfloat16)
v = random.normal(random.split(key, 4)[2], (b,s,h,d), dtype=jnp.bfloat16)
sd = SequenceDescriptor(seqlens=(jnp.array([s]*b, dtype=jnp.int32),)*2)

@jax.jit
def grad_fn(q, k, v):
    def fwd(q, k, v):
        return jnp.sum(fused_attn(qkv=(q,k,v), bias=None, sequence_descriptor=sd, seed=None,
            attn_bias_type=AttnBiasType.NO_BIAS, attn_mask_type=AttnMaskType.NO_MASK,
            qkv_layout=QKVLayout.BSHD_BSHD_BSHD, scaling_factor=1.0/(d**0.5),
            dropout_probability=0.0, is_training=True))
    return jax.grad(fwd, argnums=(0,1,2))(q, k, v)

with global_shard_guard(MeshResource()):
    dq, dk, dv = grad_fn(q, k, v)
    jax.block_until_ready(dq)
print(f"dQ={hashlib.sha256(np.array(dq).tobytes()).hexdigest()[:16]}")
# Expected: same checksum every run. Actual: different every run.
```

## Impact

This is the sole blocker for deterministic training at full throughput on ROCm.

| Config | TFLOP/s/device | Batch | Throughput |
|--------|---------------|-------|------------|
| CK fused (current — non-deterministic) | 968 | 8 | 2,264 tokens/s/device |
| CK disabled (deterministic workaround) | 100 | 1 | 233 tokens/s/device |
| **CK fused + fix (expected)** | **~968** | **8** | **~2,264 tokens/s/device** |

The workaround (`NVTE_FUSED_ATTN=0`) forces unfused attention which materializes `[batch, heads, seq, seq]` — for llama2-70b that's 137 GB/GPU → OOM, requiring batch reduction from 8 to 1.

## Risk Assessment

| Aspect | Confidence | Notes |
|--------|-----------|-------|
| Fix makes flag reach CK | 100% | Proven by CK log diagnostic |
| CK workspace calculation handles deterministic=true | 95% | Verified in `fused_attn_ck.cpp:867`: `nsplits = deterministic? ceil(s_kv/kN0):1` |
| CK dispatch selects `_deterministic` variant | 95% | Verified in codegen: `t.is_deterministic == {F_deterministic}` |
| CK `_deterministic` kernel produces correct deterministic results | ~70% | Source audit shows correct architecture (per-split write + fixed-order reduce), but this code path has **never been tested end-to-end** |

The 30% risk on the last point is because the `_deterministic` CK kernel path has never been exercised in production. We recommend the TE team also run the standalone CK `tile_example_fmha_bwd` with `deterministic=true` to validate the CK kernel independently before merging.
