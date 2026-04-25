JAX/MaxText Deterministic Training  
  
@Zhang, Zirui shared the experience on Megatron side:

- Megatron can achieve deterministic training by turning off TE fused attention. Turning off TE fused attention causes memory to explode (materializes the full seq×seq attention matrix) and requires decreasing batch size, so throughput drops to ~10–20% of normal. This matches what we see in MaxText — same root cause.
- Key customers, which had requested deterministic mode, include Meta, AWS, Microsoft.
- One customer uses ++[ROCm/flash-attention](https://github.com/ROCm/flash-attention)++, which has deterministic mode. Another option is to use Aiter attention.
- Customers would use deterministic mode to debug new features or add deterministic tests in nightly CI.
- RCCL, at least as used in training, should always be deterministic.
- ++[Training/Inference Deterministic Recipe - TAS-Team - Confluence](https://amd.atlassian.net/wiki/spaces/~712020ea4fade82ae94a95b7c0ba1cb554d2a8/pages/1335501341)++

---

# **Deterministic Training on ROCm — Status Report**

**Date:** 2026-03-15

## **What works**

Bit-exact deterministic training is **verified working** across:

- **Dense models**: llama2-70b (50 steps, 1 node), llama2-7b (50 steps, 1 node)
- **MoE models**: ds-proxy-se0-e256-h4096 — 256 experts, 32 layers, 200 steps, **8 nodes**
- **Multi-node**: 2-node (16× gfx950) verified bit-exact, with and without RCCL ring pinning
- **Dropout**: real dropout (rate=0.1) is bit-exact with `threefry2x32` PRNG
- **Full stack audited**: XLA reductions, hipBLASLt GEMMs, rocBLAS, RCCL collectives, JAX PRNG — all confirmed deterministic on gfx950/ROCm 7.1.1 without any special flags

Usage: `./submit.sh llama2-70b -- _env_DETERMINISTIC_MODE=1 per_device_batch_size=1`

## **What doesn't work**

**Performance.** Deterministic mode has a **~9.7x throughput penalty** (968 → 100 TFLOP/s/device). This makes it validation/debug-only, not production-viable.

The penalty comes from disabling CK fused attention (`NVTE_FUSED_ATTN=0`), which forces unfused attention that materializes the full `[batch, heads, seq, seq]` matrix. For llama2-70b that's 137 GB/GPU at batch=8 → OOM → forced batch reduction to 1.

## **Root cause and blocker**

**Single root cause identified:** TransformerEngine hardcodes `deterministic=false` at 3 call sites in `fused_attn.cpp` (lines 491, 680, 866) when calling the CK backward kernel. CK already has working `_deterministic` kernel variants (per-split dQ buffers + fixed-order reduction instead of `atomicAdd`) — they're compiled into `libmha_bwd.so` but TE never enables them. The TODO comment in the code reads: *"enable deterministic after CK team show us how"*.

This is a **TE bug, not a CK bug.** The fix is changing `false` to `deterministic` at those 3 lines — the variable is already in scope.

## **What to do next**


| Priority     | Action                                                                                               | Owner                  | Effort                                                  |
| ------------ | ---------------------------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------- |
| **CRITICAL** | File and land the 3-line TE fix (`fused_attn.cpp`: wire `deterministic` param through to CK backend) | ROCm/TransformerEngine | 3 lines, ~1 day                                         |
| Low          | hipBLASLt public deterministic API                                                                   | ROCm/rocm-libraries    | 3-5 days (future-proofing only — no-op on gfx950 today) |
| Low          | Make MaxText PRNG configurable (currently hardcodes `unsafe_rbg`)                                    | ROCm/maxtext           | ~4 lines                                                |
| Low          | Pass `deterministic=` param from MaxText to TE attention                                             | ROCm/maxtext           | 1 line                                                  |


**Bottom line:** The full-stack analysis is complete. Every layer from MaxText through RCCL has been audited and verified deterministic. The one remaining blocker is a 3-line TE wiring fix. Once that lands, deterministic training should work at full throughput with zero performance penalty.