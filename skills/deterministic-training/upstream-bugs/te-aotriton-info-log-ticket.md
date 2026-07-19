# TE — Surface AOTriton backend fallback reason at info log level

## Problem

When `is_aotriton_backend_supported()` in
`transformer_engine/common/fused_attn_rocm/fused_attn_aotriton.cpp:76-77` returns `false`,
TE silently falls back to the unfused JAX-native attention path. The only feedback is a
generic warning buried among many TE startup messages:

```
UserWarning: Fused attention is not enabled because there is no available kernel.
```

This is unactionable. Users who explicitly set `NVTE_FUSED_ATTN_AOTRITON=1` expecting
AOTriton dispatch get unfused fallback with no diagnostic explaining *why* — was it GQA?
MLA? `head_dim ≥ 512`? Sliding-window attention? Mixed dtype? Mixed `seqlen + causal`?
Non-vanilla softmax?

This was already flagged in PR #508 review by `@wangye805`.

## Observed Behavior

Running MaxText with `NVTE_FUSED_ATTN_AOTRITON=1 NVTE_FUSED_ATTN_CK=0` on a GQA model
(e.g. llama2-70B, ds-proxy-e128-h2048 — both use GQA):

1. TE accepts the env var.
2. `is_aotriton_backend_supported()` evaluates internally and returns `false` (because
   the `TODO` at `fused_attn_aotriton.cpp:76` excludes GQA).
3. The single `UserWarning` is logged.
4. TE silently falls back to unfused (JAX-native) attention.
5. Training proceeds with `O(seq²)` memory, OOMs at the configured `per_device_batch_size`,
   or runs ~9.7× slower if the batch is forcibly reduced.

The user only discovers AOTriton was skipped via downstream symptoms (OOM, throughput
collapse) rather than from the TE log.

## Expected Behavior After Fix

A single info- (or warning-) level log line stating the rejection reason, e.g.:

```
[TE] AOTriton backend rejected: GQA not supported (num_heads_q=64, num_heads_kv=8)
[TE]   Falling back to: <next backend in chain — CK, then unfused>
```

For each of the constraints in `is_aotriton_backend_supported()`, emit a distinct
reason string:

- "GQA not supported (num_heads_q=N, num_heads_kv=M)"
- "MLA not supported"
- "head_dim ≥ 512 not supported (d_qk=N)"
- "Sliding-window attention not supported"
- "Mixed dtype not supported (Q=..., K=..., V=...)"
- "Mixed seqlen + causal not supported"
- "Non-vanilla softmax not supported"
- etc.

The signal should fire on first encounter per backend, not on every call (avoid log spam).

## Proposed Fix Sketch

In `transformer_engine/common/fused_attn_rocm/fused_attn_aotriton.cpp`:

```cpp
bool is_aotriton_backend_supported(/* args */, std::string* reject_reason = nullptr) {
    if (is_gqa(num_heads_q, num_heads_kv)) {
        if (reject_reason) *reject_reason =
            "GQA not supported (num_heads_q=" + std::to_string(num_heads_q) +
            ", num_heads_kv=" + std::to_string(num_heads_kv) + ")";
        return false;
    }
    // ... other checks, each with a descriptive reject_reason string
    return true;
}
```

At the dispatch site (caller of `is_aotriton_backend_supported`):

```cpp
std::string reason;
if (!is_aotriton_backend_supported(/*...*/, &reason)) {
    NVTE_LOG_FIRST_OCCURRENCE(WARNING) << "AOTriton backend rejected: " << reason
                                       << "; falling back to next backend.";
}
```

Using a `NVTE_LOG_FIRST_OCCURRENCE` macro (or equivalent) prevents log spam — the
diagnostic fires once per process, which is enough to alert the user.

## Environment

- Docker image: `rocm/jax-training:maxtext-v26.2-det-te508-aot`
- TransformerEngine `2.12.0.dev0+8943023d`
- ROCm 7.1.1, JAX 0.8.2, GPU: 8× gfx950 (MI355X)
- AOTriton 0.11.x compiled into image (`libaotriton_TEprivate_v2.so`, 4.6 MB)
- Observation: setting `NVTE_FUSED_ATTN_AOTRITON=1 NVTE_FUSED_ATTN_CK=0` on
  llama2-70B (GQA: 64 q-heads, 8 kv-heads) silently dispatched the unfused path

## Impact

- **Diagnostic UX.** Customers waste hours of debugging chasing OOM symptoms instead of
  reading a one-line log explaining the silent fallback.
- **Roadmap clarity.** If AOTriton ever closes the GQA gap, customers should be able to
  toggle `NVTE_FUSED_ATTN_AOTRITON=1` with confidence and read the log to confirm dispatch.
- **No correctness impact** — this is purely a logging change.

## References

- The bug also informs upstream-action-item #3 in the deterministic-training skill set:
  `skills/deterministic-training/upstream-action-items.md`.
- PR #508 review comment by `@wangye805` (private GitHub thread).
- AOTriton rejection criteria source (post-PR-508):
  https://github.com/ROCm/TransformerEngine/blob/8943023/transformer_engine/common/fused_attn_rocm/fused_attn_aotriton.cpp#L76
