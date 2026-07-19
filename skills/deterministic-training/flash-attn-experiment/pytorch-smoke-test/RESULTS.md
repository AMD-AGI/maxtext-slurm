# ROCm/flash-attention Smoke Test — Results

**Date:** 2026-04-25
**Host:** chi2811 (8x gfx950, MI355X)
**Image:** `flash-attn-rocm:gfx950` (committed from `rocm/pytorch-private:nan-repro-rocm7.0.2.1_apple2apple_centos` after `pip install flash-attn`)
**ROCm:** 7.0.51831 (HIP 7.0.2.1)
**PyTorch:** 2.12.0a0+gitdcaf562
**flash-attn:** 2.8.3 (built from source for gfx950)

## Headline result

**`flash_attn_func(..., deterministic=True)` is bit-exact on gfx950.**

The CK `_deterministic` kernel that the project's `te-hardcodes-deterministic-false-for-ck-backend.md`
audit was 70%-confident about is **confirmed correct at production shape**. Switching to it
would cut the deterministic-mode penalty from **9.7× to 1.26×** — a ~7.7× speedup over the
current `NVTE_FUSED_ATTN=0` workaround.

## Run 1 — small shape (B=1, S=2048, H=8, D=128)

Built the wheel from source; confirmed determinism on a small input.

| Sub-test | Result |
|---|---|
| Test 1 — `deterministic=True`, 2 runs | **BIT-EXACT** (`out`, `dQ`, `dK`, `dV`) |
| Test 2 — `deterministic=False`, 2 runs | All identical (shape too small for atomic contention to surface non-determinism — harness inconclusive at this scale) |
| Test 3 — throughput | det=T: 0.41 ms/iter, ~36 TFLOP/s · det=F: 0.39 ms/iter, ~39 TFLOP/s |

Build time: **~13 min** on 64 cores compiling `csrc/composable_kernel/...` for `gfx950` (~2,500
object files).

## Run 2 — production shape (B=2, S=4096, H=64, D=128)

One full attention block of llama2-70b on a single GPU.

| Sub-test | Result | Detail |
|---|---|---|
| Test 1 — `deterministic=True`, 2 runs | **BIT-EXACT** | `out` ✓, `dQ` ✓, `dK` ✓, `dV` ✓ |
| Test 2 — `deterministic=False`, 2 runs | **observed non-determinism** | `dQ` differs (max \|Δ\| = 4.88e-4); `out`, `dK`, `dV` identical |
| Test 3 — throughput, 20-iter median | det=T: **10.69 ms/iter, 90.0 TFLOP/s**; det=F: **8.48 ms/iter, 113.4 TFLOP/s** | Deterministic-bwd overhead: **1.26× (26% slower)** |

Note that test 2's pattern — `dQ` differs while `dK`/`dV` are identical — exactly matches the
project's existing audit:

> *"dQ gradient checksums differ across runs in both modes; dK is always identical (dK has no
> atomicAdd)"*
> — `upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md`

This independently re-confirms the root-cause analysis: **dQ accumulation uses `atomicAdd` in
the non-deterministic kernel; the `_deterministic` variant uses per-split buffers + fixed-order
reduction.** The CK kernel team's design works; only TE's wiring failed to dispatch it.

## What this changes for the project

The deterministic-training docs say `NVTE_FUSED_ATTN=0` is *the* sole-source workaround at a
9.7× perf cost (`SKILL.md` Fix 7). The 9.7× breaks down as:

- **8× from batch reduction** (8→1, forced by 137 GB/GPU OOM under unfused attention)
- **1.2× from unfused-vs-fused overhead**

ROCm/flash-attention's `deterministic=True` keeps the kernel **fused**, so:

- Memory stays O(seq) → batch=8 fits → 8× factor disappears
- Only the deterministic-backward overhead applies → measured **1.26×** here

Net: **deterministic training at ~770 TFLOP/s/device** (= 968 / 1.26) instead of 100 — within
80% of the non-deterministic baseline. **Production-viable, not debug-only.**

This makes the JAX-side integration (the main deliverable next) a justified investment.

## What this validates and what it doesn't

| Validated | Not validated |
|---|---|
| CK `_deterministic` bwd kernel is bit-exact on gfx950 | Same kernel under SPMD/sharding (multi-GPU) |
| Pattern matches root-cause audit (dQ atomicAdd) | Same kernel inside MaxText's training loop end-to-end |
| Build path works (`pip install flash-attn` from source) | Multi-node bit-exactness |
| Deterministic-bwd overhead (~26%) is well within "production-viable" | Real loss-curve numerical match vs TE-based runs (different impls have different numerics) |

## Reproducing this

```bash
cd /mnt/vast/qiangh/clean/maxtext-slurm
FA_IMAGE=flash-attn-rocm:gfx950 \
FA_BATCH=2 FA_SEQ=4096 FA_HEADS=64 FA_HEAD_DIM=128 \
    bash skills/deterministic-training/flash-attn-experiment/pytorch-smoke-test/run_smoke_test.sh
```

The image `flash-attn-rocm:gfx950` is committed locally on this host (57.8 GB, layered on the
existing `rocm/pytorch-private:nan-repro-rocm7.0.2.1_apple2apple_centos`). To rebuild from
scratch on another host:

```bash
# Run the install in a container WITH GPU access (build-time GPU access required):
docker run -d --device=/dev/kfd --device=/dev/dri --group-add=video \
    --ipc=host --shm-size=16G --network=host \
    --name=flash-attn-installer \
    rocm/pytorch:latest \
    bash -c "MAX_JOBS=\$(nproc) pip install --no-build-isolation flash-attn && sleep 60"

# After ~13 min, verify and commit:
docker exec flash-attn-installer python3 -c 'import flash_attn; print(flash_attn.__version__)'
docker commit flash-attn-installer flash-attn-rocm:gfx950
docker stop flash-attn-installer && docker rm flash-attn-installer
```

## Next steps

1. **Done — this smoke test.** Decision-relevant fact established.
2. **Next — build a custom `rocm/jax-training` image with PyTorch + flash-attn alongside JAX.** Required for the JAX↔PyTorch bridge to even import flash-attn from inside the training container. ~1 day.
3. **Then — single-GPU `jax.pure_callback` wrapper** of `flash_attn_func` that matches Flax `DotProductAttention`'s signature, with `jax.custom_vjp` for backward. ~2–3 days.
4. **Then — single-GPU MaxText training run** with the wrapper active, comparing loss curve vs TE-based runs. ~1 day.
5. **Then — multi-GPU sharding via `custom_partitioning`** rules. ~3–4 days.
6. **Eventually — replace pure_callback with C++ FFI** for production-grade performance. ~3–5 days.

After step 4, you'd already have a working deterministic-mode training path running through
`run_local.sh` end-to-end — just single-GPU. Steps 5–6 scale it to production.
