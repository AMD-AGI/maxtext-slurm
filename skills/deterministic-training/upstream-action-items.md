# Deterministic Training — Upstream Action Items

Items that require changes in external repos. Ordered by priority.
For implementation details of current mitigations, see [technical-reference.md](technical-reference.md).

---

## 1. TE Hardcodes `deterministic=false` for CK Backend — **RESOLVED**

**Status:** ✅ **Fixed by [ROCm/TransformerEngine PR #508](https://github.com/ROCm/TransformerEngine/pull/508), merged 2026-04. Merge commit `8943023d654d4b89accc1cc413989547808898dd`. TE version `2.12.0.dev0+8943023d`.**

**Bug report (kept for history):** [upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md](upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md)

**What was wrong.** `nvte_fused_attn_bwd` in `fused_attn.cpp` (lines 491, 680, 866) hardcoded
`deterministic=false` when calling the CK backend, ignoring the caller's parameter:

```cpp
false, // TODO: enable deterministic after CK team show us how
```

CK already implemented `_deterministic` kernel variants (per-split `dQ_accum` buffers +
`convert_dQ` fixed-order reduce instead of `atomicAdd`) — both `_deterministic` and
`_ndeterministic` were compiled into `libmha_bwd.so`. The flag simply never reached them.

**The fix** (3 lines):

```cpp
- /*deterministic=*/false,
+ /*deterministic=*/deterministic,
```

**Verified on the new image** (`rocm/jax-training:maxtext-v26.2-det-te508-aot`):
- llama2-70B 1000-step dual-run BIT-EXACT, checksum `e352e66f43134c70`
- llama2-70B 200-step at max-pdbs=11 BIT-EXACT, checksum `1d857e6d219848c5`
- ds-proxy-e128-h2048 200-step BIT-EXACT (std and sparse_matmul paths)
- ds-proxy-e256-h2048 50-step BIT-EXACT (sparse_matmul, `remat_policy=full`)
- Cross-host BIT-EXACT chi2868 ↔ chi2774
- Throughput: 903 TFLOP/s/dev at max-pdbs (dense), 1.06× vs non-det baseline (was 9.7×)

Full validation record: [`deterministic-proj/harness/reports/2026-05-05_summary.md`](../../deterministic-proj/harness/reports/2026-05-05_summary.md).

---

## 2. TE — `NVTE_CK_DETERMINISTIC_MAX_SPLITS` env var to cap workspace (NEW, CRITICAL for long seqs)

**Repo:** `ROCm/TransformerEngine`
**Status:** Drafted, awaiting submission. See [`upstream-bugs/te-nsplits-cap-ticket.md`](upstream-bugs/te-nsplits-cap-ticket.md) (to be drafted).

**Problem.** The deterministic CK backward allocates `nsplits = ceil(s_kv / kN0)` per-split
`dQ_accum` buffers, where `kN0 = 128` if `d_qk ≤ 128` else `64`. Each buffer is
`h_q × b × s_q × d_qk × 4` bytes. This is intrinsic to the split-accumulator algorithm
(confirmed by TE team 2026-05-06; the `convert_dQ` reduction kernel is HBM-bandwidth-bound,
cost grows linearly with `nsplits`).

For long sequence lengths the cost is substantial:
- `hd128_s16384_b1_mha`: extra **4,064 MB per attention call per sample** (det 4,224 MB vs non-det 160 MB → 26.3×)
- On llama2-70B max-pdbs: 31% batch-headroom loss (det=11 vs non-det=16)

There is currently **no env-var override** to cap `nsplits` upstream. Users who want a
softer trade-off (e.g. cap `nsplits=4` for `s_kv ≥ 4096`, accepting a controlled atomicAdd
group within each cap) cannot do so without source changes.

**Proposed fix.** Add env var `NVTE_CK_DETERMINISTIC_MAX_SPLITS` read in
`fused_attn_ck.cpp:757-784` to cap `nsplits`:

```cpp
size_t kN0 = (d_qk <= 128) ? 128 : 64;
size_t nsplits_natural = deterministic ? ceil(1.0 * s_kv / kN0) : 1;
size_t max_splits = 0;
if (const char *env = getenv("NVTE_CK_DETERMINISTIC_MAX_SPLITS")) {
    max_splits = static_cast<size_t>(atoi(env));
}
size_t nsplits = (max_splits > 0 && nsplits_natural > max_splits) ? max_splits : nsplits_natural;
```

**Customer-facing reason.** Documentation gap too — `transformer_engine/docs/envvars.rst`
does not warn about the per-split workspace cost. Closed-form formula:

```
extra_bytes = (ceil(s_kv / kN0) - 1) × num_heads_q × (b × s_q) × head_dim_qk × 4
              kN0 = 128 if d_qk ≤ 128 else 64
```

**Effort.** ~5 lines C++ + 1 doc paragraph.

**Risk if not done.** Customers with `seq ≥ 8192` on small batches hit OOM in deterministic
mode and have no recourse but to disable determinism.

---

## 3. TE — Surface AOTriton fallback reason at info log level (NEW, LOW)

**Repo:** `ROCm/TransformerEngine`
**Status:** Drafted, awaiting submission. See [`upstream-bugs/te-aotriton-info-log-ticket.md`](upstream-bugs/te-aotriton-info-log-ticket.md) (to be drafted).

**Problem.** When `is_aotriton_backend_supported()` in
`transformer_engine/common/fused_attn_rocm/fused_attn_aotriton.cpp:76-77` returns false,
the only feedback is a generic `UserWarning: Fused attention is not enabled because there
is no available kernel.` This is buried among many TE warnings and gives no information
about *why* AOTriton was rejected (GQA? MLA? head_dim ≥ 512? SWA? mixed dtype?).

Users who set `NVTE_FUSED_ATTN_AOTRITON=1` expecting AOTriton silently get unfused fallback
with no diagnostic. This was already noted in PR #508 review by `@wangye805`.

**Proposed fix.** Add an info-level (or warning-level on first encounter) log line
explaining the rejection reason:

```cpp
NVTE_INFO("AOTriton backend rejected: " << reason << " (head_dim=" << d_qk
          << ", num_heads_q=" << h_q << ", num_heads_kv=" << h_kv
          << ", attn_mask=" << mask_str << ", ...)");
```

**Effort.** ~10 lines C++.

**Risk if not done.** Customers waste time debugging silent fallback. Doesn't block
production (CK covers GQA now, post-PR #508), but it's a sharp edge.

---

## 4. XLA — `xla_gpu_enable_scatter_determinism_expander` upstream re-enable + MoE-aware fast path (NEW, HIGH for MoE users)

**Repo:** [`openxla/xla`](https://github.com/openxla/xla)
**Status:** Drafted, awaiting submission. See [`upstream-bugs/xla-scatter-determinism-moe-ticket.md`](upstream-bugs/xla-scatter-determinism-moe-ticket.md) (to be drafted).

**Problem.** XLA's `ScatterDeterminismExpander` (activated by
`xla_gpu_deterministic_ops=true`) rewrites every scatter/scatter-add into a sequential
while-loop with `trip_count = scatter_size`. For MoE models this is catastrophic:

- 2 scatter ops per layer (token dispatch + expert-output combine)
- × ~3-4 instances per layer (fwd + remat-fwd + bwd)
- × 64 decoder layers
- × 8192 sequential iterations per loop (seq=4096 × top_k=2)
- ≈ **~1M sequential scalar HBM read-modify-write ops per training step**

Measured impact on ds-proxy-e128-h2048: **152× throughput collapse** (264 sec/step at
1.6 TFLOP/s, vs 1.74 sec/step at 240 TFLOP/s without the flag).

**The MoE scatter is conflict-free by construction.** Each token routes to a unique
expert slot (no two tokens write to the same destination), so there is no `atomicAdd`
race to begin with. The sequential serialization is pure overhead. HLO evidence:
[`deterministic-proj/harness/reports/2026-05-06_moe_scatter_serialization.md`](../../deterministic-proj/harness/reports/2026-05-06_moe_scatter_serialization.md).

NVIDIA's high-performance replacement — the prefix-scan `ScatterDeterminismExpander`
(PRs [#17886](https://github.com/openxla/xla/pull/17886) + [#19275](https://github.com/openxla/xla/pull/19275)) — is **still default=false in upstream XLA** as of 2026-05 due to bug history:

| Date | Event |
|---|---|
| 2024-11-15 | PR #19275 merged, default=true |
| 2024-11-16 | Google internal test failure → [`badb11c`](https://github.com/openxla/xla/commit/badb11c): default=false |
| 2024-11-19 | [PR #19429](https://github.com/openxla/xla/pull/19429): bug fix + re-enable attempt; `@akuegel` found more bugs same day; fix merged WITHOUT re-enable |
| Present | Still default=false |

Tested manually enabling on ds-proxy-e128-h2048: **OOM at bs=1** — prefix-scan's sort
buffer (~16 GB) is too large for 128-expert models.

**Proposed fix.** Two-part:

1. **MoE-aware fast path**: detect when scatter indices are statically known to be
   conflict-free (e.g. produced by MoE routing via `argmax` + `one_hot` + `cumsum` pattern,
   which guarantees unique destinations) and **skip the sequential expander entirely**.
   Bit-exactness is preserved without any rewrite.

2. **Prefix-scan expander memory cap**: optionally chunk the sort buffer to bound memory
   (trades a small amount of perf for fitting in HBM at large MoE sizes).

**Effort.** Substantial — XLA pass authorship. ~1-2 weeks for a credible engineer.

**Risk if not done.** Any customer setting `xla_gpu_deterministic_ops=true` on MoE
catastrophically regresses. Currently mitigated by documentation (see SKILL.md and the
removed flag in `train_env.sh`), but a non-obvious upstream gotcha.

---

## 5. hipBLASLt Deterministic Mode — Public API (LOW)

**Repo:** `ROCm/rocm-libraries` (hipBLASLt), `ROCm/xla`
**Status:** Unchanged. See [`upstream-bugs/hipblaslt-deterministic-api.md`](upstream-bugs/hipblaslt-deterministic-api.md).

**Problem.** hipBLASLt has no public API to request deterministic GEMM. TensileLite
internally supports `DeterministicMode` but hipBLASLt never exposes it.

**Current workaround.** 10-line env-var patch (`HIPBLASLT_DETERMINISTIC=1`) in
`tensile_host.cpp`. Applied as local modification in our Docker image.

**Empirical finding (March 2026, still valid).** Testing on gfx950/ROCm 7.1.1 BF16 shows
the patch is a **no-op**. The gfx950 BF16 solution library contains no atomic-GSU or
StreamKAtomic solutions. hipBLASLt is deterministic by default on this platform. Verified
with an extreme MoE config (8 tokens/expert, 5% CU util) — bit-exact with and without the
patch across 6 runs.

**Risk if not done.** None on gfx950/ROCm 7.1.1. Potential future risk if AMD adds
atomic-GSU solutions in later ROCm releases, or for gfx942 which has 100 StreamK solution
files.

---

## 6. MaxText PRNG Configuration (MINOR)

**Repo:** `ROCm/maxtext`
**Status:** Unchanged. See [`upstream-bugs/maxtext-prng-and-te-param.md`](upstream-bugs/maxtext-prng-and-te-param.md) (Issue 1).

**Problem.** MaxText hardcodes `jax.config.update("jax_default_prng_impl", "unsafe_rbg")`
in `train.py:initialize()`, overriding any env var or config. Users cannot change the
PRNG without monkey-patching.

**Current workaround.** `utils/deterministic.py` wraps `initialize()` to re-apply
`threefry2x32` after MaxText's override.

**Proposed fix.**
1. Add `jax_default_prng_impl: "unsafe_rbg"` to `MaxText/configs/base.yml`
2. In `train.py:initialize()`, read from config:

```python
jax.config.update("jax_default_prng_impl", config.jax_default_prng_impl)
```

**Effort.** ~4 lines. Backward compatible.

**Note.** `unsafe_rbg` is empirically deterministic on gfx950 for fixed-seed weight init.
The monkey-patch is a safety net, not functionally required on this platform.

---

## 7. MaxText TE Deterministic Parameter (MINOR)

**Repo:** `ROCm/maxtext`
**Status:** Unchanged. See [`upstream-bugs/maxtext-prng-and-te-param.md`](upstream-bugs/maxtext-prng-and-te-param.md) (Issue 2).

**Problem.** MaxText creates TE's `DotProductAttention` without passing
`deterministic=True`, relying on `NVTE_ALLOW_NONDETERMINISTIC_ALGO` env var.

**Current workaround.** `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (works for algorithm selection;
PR #508 now also wires it through to the CK kernel).

**Proposed fix.** In `MaxText/layers/attention_op.py`:

```python
return dpa_layer(query, key, value, sequence_descriptor=attn_mask,
                 deterministic=not self.config.enable_dropout)
```

**Effort.** 1 line. Backward compatible. Code-quality improvement.

---

## Summary

| # | Item | Repo | Priority | Status | Effort |
|---|---|---|---|---|---|
| 1 | TE hardcodes `deterministic=false` | TransformerEngine | — | **✅ RESOLVED** (PR #508) | — |
| 2 | TE `NVTE_CK_DETERMINISTIC_MAX_SPLITS` env var | TransformerEngine | **HIGH** | Draft pending | ~5 lines + docs |
| 3 | TE AOTriton fallback info log | TransformerEngine | LOW | Draft pending | ~10 lines |
| 4 | XLA MoE-aware scatter determinism | openxla/xla | **HIGH** (MoE users) | Draft pending | 1-2 wks |
| 5 | hipBLASLt public deterministic API | rocm-libraries + xla | Low | Unchanged | 3-5 days |
| 6 | MaxText PRNG config | maxtext | Low | Unchanged | 1 day |
| 7 | MaxText TE deterministic param | maxtext | Low | Unchanged | 1 day |

**Key finding (2026-05).** The CK deterministic path is now fully production-viable. Remaining
items are workspace-cost ergonomics (#2), diagnostic UX (#3), the MoE-side XLA hazard (#4),
and minor code-quality / future-proofing.
