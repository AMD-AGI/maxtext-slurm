# Deterministic Training — Upstream Action Items

Items that require changes in external repos. Ordered by priority. For implementation details of current mitigations, see [technical-reference.md](technical-reference.md).

---

## 1. TE Hardcodes `deterministic=false` for CK Backend (CRITICAL)

**Bug report:** [upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md](upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md)
**Ticket:** [upstream-bugs/te-deterministic-false-ticket.md](upstream-bugs/te-deterministic-false-ticket.md)

**Repo:** `ROCm/TransformerEngine`

**Root cause (confirmed):** `nvte_fused_attn_bwd` in `fused_attn.cpp` (lines 491, 680, 866) hardcodes `deterministic=false` when calling the CK backend, ignoring the caller's parameter:
```cpp
false, // TODO: enable deterministic after CK team show us how
```

CK *does* implement a working `_deterministic` kernel variant (per-split dQ buffers + fixed-order reduction instead of `atomicAdd`). Both `_deterministic` and `_ndeterministic` variants are compiled into `libmha_bwd.so`. The flag simply never reaches them.

**Evidence:**
- `NVTE_LOG_CK_CONFIG=1` shows `deterministic: 0` regardless of `NVTE_ALLOW_NONDETERMINISTIC_ALGO` setting
- dQ gradient checksums differ across runs in both modes; dK is always identical (dK has no `atomicAdd`)
- Workspace sized for `nsplits=1` (non-deterministic) even when deterministic is requested

**Fix:** Change `false` to `deterministic` at three call sites. The variable is already in scope.

**Impact:** Sole blocker for deterministic training at full throughput. Current workaround (`NVTE_FUSED_ATTN=0`) has ~9.7x throughput loss.

---

## 2. hipBLASLt Deterministic Mode — Public API (LOW PRIORITY)

**Details:** [upstream-bugs/hipblaslt-deterministic-api.md](upstream-bugs/hipblaslt-deterministic-api.md)
**Ticket:** [upstream-bugs/hipblaslt-deterministic-api-ticket.md](upstream-bugs/hipblaslt-deterministic-api-ticket.md)

**Repo:** `ROCm/rocm-libraries` (hipBLASLt), `ROCm/xla`

**Problem:** hipBLASLt has no public API to request deterministic GEMM. TensileLite internally supports `DeterministicMode` but hipBLASLt never exposes it.

**Current workaround:** 10-line env var patch (`HIPBLASLT_DETERMINISTIC=1`) in `tensile_host.cpp`. Applied as local modification in our Docker image.

**Updated finding (March 2026):** Testing on gfx950/ROCm 7.1.1 showed the patch is a **no-op**. The gfx950 BF16 solution library contains no atomic-GSU or StreamKAtomic solutions. hipBLASLt is deterministic by default on this platform. We verified this with an extreme MoE config designed to trigger low CU occupancy (8 tokens/expert, 16 tiles, 5% CU util) — results were bit-exact with and without the patch across 6 runs.

**Risk if not done:** None on gfx950/ROCm 7.1.1. Potential future risk if AMD adds atomic-GSU solutions for gfx950 in later ROCm releases, or for gfx942 which has 100 StreamK solution files.

---

## 3. MaxText PRNG Configuration (MINOR)

**PR draft:** [upstream-bugs/maxtext-prng-and-te-param.md](upstream-bugs/maxtext-prng-and-te-param.md) (Issue 1)

**Repo:** `ROCm/maxtext`

**Problem:** MaxText hardcodes `jax.config.update("jax_default_prng_impl", "unsafe_rbg")` in `train.py:initialize()`, overriding any env var or config setting. Users cannot change the PRNG without monkey-patching.

**Current workaround:** `utils/deterministic.py` wraps `initialize()` to re-apply `threefry2x32` after MaxText's override.

**Proposed fix:**
1. Add `jax_default_prng_impl: "unsafe_rbg"` to `MaxText/configs/base.yml`
2. In `train.py:initialize()`, read from config instead of hardcoding:

```python
jax.config.update("jax_default_prng_impl", config.jax_default_prng_impl)
```

**Effort:** ~4 lines. Backward compatible (default remains `unsafe_rbg`).

**Ablation finding:** `unsafe_rbg` is already deterministic on ROCm/gfx950 for fixed-seed weight init (`enable_dropout=False`). Separately, `threefry2x32` with `enable_dropout=True` was verified bit-exact. The monkey-patch works but is a safety net, not functionally required on this platform.

---

## 4. MaxText TE Deterministic Parameter (MINOR)

**PR draft:** [upstream-bugs/maxtext-prng-and-te-param.md](upstream-bugs/maxtext-prng-and-te-param.md) (Issue 2)

**Repo:** `ROCm/maxtext`

**Problem:** MaxText creates TE's `DotProductAttention` without passing `deterministic=True`, relying entirely on the `NVTE_ALLOW_NONDETERMINISTIC_ALGO` env var.

**Current workaround:** `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` env var (works for algorithm selection but doesn't fix CK kernel-level non-determinism — see item 1).

**Proposed fix:** In `MaxText/layers/attention_op.py`:

```python
return dpa_layer(query, key, value, sequence_descriptor=attn_mask,
                 deterministic=not self.config.enable_dropout)
```

**Effort:** 1 line. Backward compatible.

**Note:** This alone does NOT fix the CK non-determinism (item 1). It's a code quality improvement that makes the deterministic intent explicit rather than relying on env vars.

---

## Summary

| # | Item | Repo | Priority | Effort | Impact |
|---|------|------|----------|--------|--------|
| 1 | TE hardcodes `deterministic=false` for CK | TransformerEngine | **CRITICAL** | 3 lines | **Sole blocker** — would enable deterministic training at full throughput (currently 9.7x perf loss) |
| 2 | hipBLASLt public deterministic API | rocm-libraries + xla | Low | 3-5 days | Future-proofing only — gfx950/ROCm 7.1.1 BF16 has no atomic-GSU solutions |
| 3 | MaxText PRNG config | maxtext | Low | 1 day | Code quality — `unsafe_rbg` is already deterministic on ROCm/gfx950 |
| 4 | MaxText TE deterministic param | maxtext | Low | 1 day | Code quality — no functional impact until TE fix lands |

**Key finding:** Item 1 is a TE bug (not CK). CK already implements `_deterministic` kernels; TE just never enables them. Items 2-4 are strictly code quality / future-proofing — none are functionally required on gfx950/ROCm 7.1.1.
