# MaxText: Make PRNG configurable + pass deterministic param to TE

**Target repo:** `ROCm/maxtext`

## Issue 1: Hardcoded PRNG (`unsafe_rbg`)

**File:** `MaxText/train.py` line ~514 (in `initialize()`)

**Problem:** MaxText unconditionally sets `jax.config.update("jax_default_prng_impl", "unsafe_rbg")`, overriding any user configuration. JAX documents `unsafe_rbg` as "not guaranteed to be stable/deterministic across backends or compiler versions."

**Fix:**
1. Add to `MaxText/configs/base.yml`:
```yaml
jax_default_prng_impl: "unsafe_rbg"
```
2. In `train.py:initialize()`, change:
```python
# Before:
jax.config.update("jax_default_prng_impl", "unsafe_rbg")
# After:
jax.config.update("jax_default_prng_impl", config.jax_default_prng_impl)
```

Effort: ~4 lines. Backward compatible (default remains `unsafe_rbg`). Users can override via `jax_default_prng_impl=threefry2x32` in config or CLI.

**Note:** Ablation testing showed `unsafe_rbg` is actually deterministic on ROCm/gfx950 for fixed seeds. This fix is for code quality and cross-platform safety.

---

## Issue 2: Missing `deterministic` param in TE attention

**File:** `MaxText/layers/attention_op.py` (`cudnn_flash_attention` method)

**Problem:** MaxText creates TE's `DotProductAttention` without passing `deterministic=True`, relying entirely on the `NVTE_ALLOW_NONDETERMINISTIC_ALGO` env var.

**Fix:**
```python
# Before:
return dpa_layer(query, key, value, sequence_descriptor=attn_mask)
# After:
return dpa_layer(query, key, value, sequence_descriptor=attn_mask,
                 deterministic=not self.config.enable_dropout)
```

Effort: 1 line. Backward compatible. Makes deterministic intent explicit in code rather than relying on env vars.

**Note:** This alone does NOT fix CK backward non-determinism (separate bug in CK). It's a code quality improvement.
