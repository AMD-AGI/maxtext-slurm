# Deterministic Training on ROCm — Status Report

> **2026-05 update — superseded by PR #508.** Item 1 ("file and land the 3-line TE fix") is now
> **DONE**: [ROCm/TransformerEngine PR #508](https://github.com/ROCm/TransformerEngine/pull/508)
> (merge commit `8943023d`) is shipped in image
> `rocm/jax-training:maxtext-v26.2-det-te508-aot` (TE `2.12.0.dev0+8943023d`). Deterministic
> training is now production-viable at **1.06× cost** (903 vs 958 TFLOP/s/dev at max-pdbs on
> llama2-70B), not 9.7×. The "What doesn't work" section below is historical; the post-PR-508
> performance is documented in [SKILL.md](SKILL.md), [cheatsheet.md](cheatsheet.md), and the
> harness validation record at `deterministic-proj/harness/reports/2026-05-05_summary.md`.
> One new finding post-2026-03: `xla_gpu_deterministic_ops=true` is **toxic on MoE** (152×
> throughput collapse from scatter serialization) and is now deliberately not set by
> `DETERMINISTIC_MODE` — see [SKILL.md](SKILL.md) "The xla_gpu_deterministic_ops warning"
> section and `deterministic-proj/harness/reports/2026-05-06_moe_scatter_serialization.md`.

**Date:** 2026-03-27 (updated from 2026-03-15; superseded by PR #508 — see 2026-05 update at top)

## What works

Bit-exact deterministic training is **verified working** across:

- **Dense models**: llama2-70b (500 steps, 1 node — checksum `b4235c1ffafada08`), llama2-7b (50 steps, 1 node)
- **MoE models**: ds-proxy-se0-e256-h4096 — 256 experts, 32 layers, 200 steps, 8 nodes
- **Multi-node**: 2-node (16× gfx950) verified bit-exact, with and without RCCL ring pinning
- **Dropout**: real dropout (rate=0.1) is bit-exact with `threefry2x32` PRNG
- **Full stack audited**: XLA reductions, hipBLASLt GEMMs, rocBLAS, RCCL collectives, JAX PRNG — all confirmed deterministic on gfx950/ROCm 7.1.1 without any special flags
- **Long-duration**: 500-step llama2-70b run verified bit-exact at float32 precision across every step (loss 10.87 → 0.0000001, zero ULP difference)

Usage: `./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 per_device_batch_size=1`

## What doesn't work

**Performance.** Deterministic mode has a **~9.7x throughput penalty** (968 → 100 TFLOP/s/device). This makes it validation/debug-only, not production-viable.

The penalty comes from disabling CK fused attention (`NVTE_FUSED_ATTN=0`), which forces unfused attention that materializes the full `[batch, heads, seq, seq]` matrix. For llama2-70b that's 137 GB/GPU at batch=8 → OOM → forced batch reduction to 1.

## Root cause and blocker

**Single root cause identified:** TransformerEngine hardcodes `deterministic=false` at 3 call sites in `fused_attn.cpp` (lines 491, 680, 866) when calling the CK backward kernel. CK already has working `_deterministic` kernel variants (per-split dQ buffers + fixed-order reduction instead of `atomicAdd`) — they're compiled into `libmha_bwd.so` but TE never enables them. The TODO comment in the code reads: *"enable deterministic after CK team show us how"*.

This is a **TE bug, not a CK bug.** The fix is changing `false` to `deterministic` at those 3 lines — the variable is already in scope.

## What to do next

| Priority | Action | Owner | Effort |
|----------|--------|-------|--------|
| **CRITICAL** | File and land the 3-line TE fix (`fused_attn.cpp`: wire `deterministic` param through to CK backend) | ROCm/TransformerEngine | 3 lines, ~1 day |
| Low | hipBLASLt public deterministic API | ROCm/rocm-libraries | 3-5 days (future-proofing only — no-op on gfx950 today) |
| Low | Make MaxText PRNG configurable (currently hardcodes `unsafe_rbg`) | ROCm/maxtext | ~4 lines |
| Low | Pass `deterministic=` param from MaxText to TE attention | ROCm/maxtext | 1 line |

## Code changes (2026-03-27)

The deterministic-mode Python logic was extracted from `utils/mfu_tracker.py` into a dedicated `utils/deterministic.py` module for separation of concerns. `mfu_tracker.py` remains the training entry point and delegates to `deterministic.py` for patches, verification, and loss checksums. Unit tests added in `utils/test_deterministic.py` (30 tests). The refactoring was validated with a 500-step bit-exact GPU run.

**Bottom line:** The full-stack analysis is complete. Every layer from MaxText through RCCL has been audited and verified deterministic. The one remaining blocker is a 3-line TE wiring fix. Once that lands, deterministic training should work at full throughput with zero performance penalty.
