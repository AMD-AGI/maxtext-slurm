# Deterministic Training on ROCm — Status Report

**Date:** 2026-03-27 (updated from 2026-03-15)

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
