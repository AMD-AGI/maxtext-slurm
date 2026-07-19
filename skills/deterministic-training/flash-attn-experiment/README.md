# Flash-Attention Experiment

Two layered tests, cheapest first. Run them in order. Stop once you have
the answer.

| Layer | Folder / file | What it answers | Cost |
|---|---|---|---|
| 1. **MaxText config-knob probe** | [`probe_attention_kernels.sh`](probe_attention_kernels.sh) | Does any `attention=` value already wired into MaxText (`flash`, `cudnn_flash_jax`, `dot_product`, `mha`) give us deterministic+fast on gfx950? If yes, **no integration work is needed at all**. | ~80 min GPU |
| 2. **ROCm/flash-attention smoke test** ✅ **PASSED 2026-04-25** | [`pytorch-smoke-test/`](pytorch-smoke-test/) — see [`RESULTS.md`](pytorch-smoke-test/RESULTS.md) | Does ROCm/flash-attention's deterministic CK kernel work on this hardware? **Yes — bit-exact at production shape, 1.26× perf cost vs non-deterministic** (vs the current 9.7× with `NVTE_FUSED_ATTN=0`). | ~13 min build + seconds per run |
| 3. ~~Full MaxText↔ROCm-flash-attn integration~~ | (not yet shipped) | Does it train end-to-end? | 8–12 days; tracked outside this folder |

Layer 3 — the actual JAX FFI / PyTorch bridge integration into MaxText —
is multi-day work that requires GPU iteration cycles. It is **not** in
this folder. Once layers 1 and 2 are complete with informative results,
layer 3 either becomes unnecessary (if layer 1 wins) or properly justified
(if layer 1 loses and layer 2 passes).

---

## Layer 1: Attention-Kernel Probe

Quickest possible test of "can flash-attention give us deterministic + fast?"
on this MaxText/JAX/ROCm stack.

## Why this is the first thing to try

Every config in `configs/*.gpu.yml` sets `attention: "cudnn_flash_te"` — the
TE path that hits the CK fused-attention bug. But `attention=` is a MaxText
config knob, not a hardcode: MaxText already supports several alternative
attention modules. If any of them is **deterministic AND fast** at production
batch on gfx950, the 9.7x `NVTE_FUSED_ATTN=0` penalty disappears with **zero
new code, zero C++, zero Docker rebuilds, zero TE patches.**

That makes this probe strictly cheaper than the alternatives in
[../proposals](../upstream-action-items.md):

| Path | Effort | Risk |
|---|---|---|
| **This probe** | 1.5 hours of GPU time | None — read-only, uses existing infra |
| Local 3-line TE fix + custom wheel | 3–5 days | Custom Docker layer to maintain |
| LD_PRELOAD shim over `nvte_fused_attn_bwd` | 3–4 days | ABI fragility across TE versions |
| JAX FFI to ROCm/flash-attention CK | 8–12 days | New C++ component |

If the probe finds a winner, the rest of those efforts are unnecessary. If
it doesn't, the result narrows the search and tells us *why* MaxText's
in-tree options aren't sufficient — which is itself useful evidence for
the upstream conversation.

## What it tests

Six runs, two per "test", each at `steps=15` on `llama2-70b` single-node:

| # | `attention=` | Batch | Why |
|---|---|---|---|
| C0 | `cudnn_flash_te` + `NVTE_FUSED_ATTN=0` | 1 | **Control.** Reproduces the existing bit-exact baseline (~100 TFLOP/s). Validates the harness. |
| C1 | `cudnn_flash_te` (default) | 8 | **Control.** Non-deterministic fast baseline (~968 TFLOP/s). Validates that `compare_runs.py` correctly flags DIFFER. |
| K1 | `dot_product` | 1 | MaxText's pure-JAX dot-product attention. No TE. Expected deterministic, expected slow. Tells us if a non-TE deterministic path exists. |
| K2 | `cudnn_flash_jax` | 8 | `jax.nn.dot_product_attention` → XLA → whatever fused path XLA picks on ROCm. Determinism depends on XLA's GPU lowering. |
| K3 | `flash` | 8 | **The interesting one.** Pallas/Splash flash attention. JAX-native fused. If this is deterministic on gfx950, **this is the win.** |
| K4 | `mha` | 8 | MaxText's MHA module. Lower expectation than K3 but cheap to include. |

All candidates run with the project's defensive flags on (`DETERMINISTIC_MODE=1`)
but with `_env_NVTE_FUSED_ATTN=1` overriding the `=0` that DETERMINISTIC_MODE
forces — because for non-TE attention paths (K1–K4) the TE fused-attn flag is
irrelevant, and we don't want to inadvertently slow the comparison by leaving
TE in the unfused fallback shape.

## Decision tree

```
For each candidate K1..K4:
  RUN  ──► OOM ──────────────────────────► document; not viable at this batch
       ──► UNSUPPORTED ────────────────► kernel missing in this fork; report
       ──► ERROR ──────────────────────► investigate per-run log
       ──► success ──► DIFFER ─────────► non-deterministic; not viable
                  ──► BIT-EXACT
                       └► TFLOP/s/dev
                            ├─ ≥ 700 ─► WIN — this kernel replaces NVTE_FUSED_ATTN=0
                            ├─ 200–700 ► partial win — beats current 100 by Nx
                            └─ < 200 ──► no improvement over current workaround
```

## How to run

```bash
cd /maxtext-slurm   # or wherever this checkout lives
bash skills/deterministic-training/flash-attn-experiment/probe_attention_kernels.sh
```

Optional knobs:

```bash
MODEL=llama2-7b  bash ...probe_attention_kernels.sh   # smaller, ~half runtime
STEPS=10         bash ...probe_attention_kernels.sh   # tighter; less margin
```

## Output

- `results.md` — one row per test with verdict, batch, throughput, kernel.
- `outputs/local_*` — per-run dirs as usual; `compare_runs.py` reads them.

## What to do with the result

| If the winner is | Then |
|---|---|
| **K3 (flash, BIT-EXACT, fast)** | Switch every `configs/*.gpu.yml` from `cudnn_flash_te` to `flash`, or wire `ATTENTION_KERNEL` into `train_env.sh` (with `DETERMINISTIC_MODE=1` selecting `flash` automatically). Update `SKILL.md` Fix 7 and `cheatsheet.md`. The 9.7x penalty is gone. |
| **K2 (cudnn_flash_jax, BIT-EXACT, fast)** | Same as K3 but via `cudnn_flash_jax`. Note: this path goes through XLA's GPU lowering, which has historically been less stable across JAX versions; pin the JAX version. |
| **K1 (dot_product, BIT-EXACT, slow)** | Marginal — different code path, same perf class as `NVTE_FUSED_ATTN=0`. Document as a backup; don't switch. |
| **All BIT-EXACT but slow** | Confirms in-tree options can't deliver fused + deterministic together. Escalate to one of: local TE patch, LD_PRELOAD shim, or JAX FFI to ROCm/flash-attention. See `../upstream-action-items.md`. |
| **All DIFFER or ERROR** | The CK non-determinism reaches further than expected (e.g., into `cudnn_flash_jax`), or this MaxText fork doesn't expose enough alternatives. File the upstream TE bug; the 3-line fix becomes the only path. |

## Caveats

- Single-node only. Multi-node behavior is a follow-up; if a kernel passes
  here, re-run on chi2870+chi2872 (the existing 2-node test pair).
- `attention=flash` may require additional MaxText config (e.g.,
  `flash_block_sizes`) on AMD. If K3 fails with a config error, add the
  expected keys to `configs/llama2-70b.gpu.yml` and re-run K3 individually.
- The probe assumes `ROCm/maxtext` exposes the same `attention=` values as
  upstream `google/maxtext`. If a candidate fails with `UNSUPPORTED-KERNEL`,
  that fork hasn't wired it; the result is informative either way.
- `K3` (`flash`) on AMD goes through Pallas → Triton → aiter. The
  `FLASH_ATTENTION_TRITON_AMD_*` env vars (see ROCm/flash-attention's
  README) may need to be set for best perf; defaults are tuned for
  determinism per AMD's documentation, which is exactly what we want.

## If you find a winner

The next file to update is `../SKILL.md` Fix 7 and `../upstream-action-items.md`,
both of which currently treat `NVTE_FUSED_ATTN=0` as the only deterministic
path. Replace those with a section pointing here and describing the
production setup for the winning kernel.
