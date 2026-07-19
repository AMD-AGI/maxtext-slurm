# ROCm/flash-attention smoke test

A standalone PyTorch test that asks **one question**: does
`flash_attn_func(..., deterministic=True)` produce bit-identical output and
gradients across two runs on this hardware?

If yes → the JAX-side integration of ROCm/flash-attention into MaxText is
worth pursuing. If no → file the bug upstream against ROCm/flash-attention
and don't waste a week on FFI work.

This is **not** the MaxText integration. It is the prerequisite test that
determines whether the integration would even produce a correct result.

## What this is and isn't

| | This smoke test | The full MaxText integration |
|---|---|---|
| Framework | PyTorch | JAX |
| Image | `rocm/pytorch:latest` | `rocm/jax-training:maxtext-v26.2` |
| Goal | "Does the kernel work?" | "Does training work end-to-end?" |
| Lines of code | ~250 | ~2,000+ (C++ FFI + sharding + Flax wrapper) |
| Effort | 30 min on first run | 8–12 days |
| Touches MaxText | No | Yes |
| Sufficient for production | No | Yes |
| Required before integration | Yes (gates the work) | — |

## Files

| File | Purpose |
|---|---|
| `test_flash_attn.py` | The actual test. Three sub-tests. |
| `run_smoke_test.sh` | Wrapper that launches a container (mirrors `run_local.sh`'s pattern) and runs the test inside. |
| `runs/<timestamp>/log` | Per-run log (created by `run_smoke_test.sh`). |
| `runs/latest.log` | Symlink to the most recent log. |

## What the test does

Three sub-tests, llama2-70b-shaped (B=2, S=4096, H=8, D=128, bf16, causal):

1. **`deterministic=True` is bit-exact across 2 runs.** This is the
   actual decision-relevant test. PASS = green-light the FFI work.
   FAIL = the kernel has the same root issue as the TE bypass; FFI work is
   invalidated.
2. **`deterministic=False` may differ.** Sanity check that the comparison
   harness can detect non-determinism if it occurs.
3. **Throughput**, deterministic vs non-deterministic, median over 20
   iterations after warmup. Quantifies the cost of the deterministic
   backward kernel on this GPU.

The test reports `OVERALL: PASS` or `OVERALL: FAIL` based on test 1 alone.
Tests 2 and 3 are observational.

## How to run

```bash
cd /mnt/vast/qiangh/clean/maxtext-slurm
bash skills/deterministic-training/flash-attn-experiment/pytorch-smoke-test/run_smoke_test.sh
```

That mirrors `run_local.sh`'s pattern: resolves the script's own dir, sources
`container_env.local.sh` if present (for any private-registry credentials),
launches a Docker container with `/dev/kfd` + `/dev/dri` + `--ipc=host` (the
combination ROCm requires), and runs the test inside.

## Customizing

```bash
# Smaller/larger config:
FA_BATCH=8 FA_SEQ=8192 bash run_smoke_test.sh

# Use a pre-baked image with flash-attn already installed (avoids the ~30 min
# first-run install). You can build such an image once with:
#   docker build -t my/rocm-flash-attn:latest .
# (Dockerfile sketch: rocm/pytorch:latest + `pip install flash-attn --no-build-isolation`)
FA_IMAGE=my/rocm-flash-attn:latest bash run_smoke_test.sh

# Use a specific release wheel from ROCm/flash-attention's releases page:
FA_WHEEL_URL=https://github.com/ROCm/flash-attention/releases/download/v2.8.4.1-cktile/flash_attn-2.8.4.1-cp310-cp310-linux_x86_64.whl \
    bash run_smoke_test.sh

# Skip the throughput benchmark (faster):
FA_BENCH_ITERS=0 bash run_smoke_test.sh   # OR
python3 test_flash_attn.py --skip-bench   # if running outside the wrapper
```

## Why this is worth running before any integration work

The deterministic-training docs in this repo
([../upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md](../../upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md))
flag **~70% confidence** that the CK `_deterministic` kernel is correct
end-to-end on gfx950 — it's never been exercised in production. ROCm/flash-attention's
`deterministic=True` path dispatches to that same `_deterministic` kernel
in `libmha_bwd.so`. If it works here, it works for the TE fix and for any
future JAX FFI binding. If it doesn't, all of those paths fail for the same
reason.

This smoke test is the cheapest way to convert that 70% into either ~99% or
0%, with a clear next step in either case.

## Outcome interpretation

| Outcome | Next step |
|---|---|
| Test 1 PASS, throughput within ~1.4× of non-deterministic | Green-light JAX FFI integration. Estimated 8–12 days; tracked separately. |
| Test 1 PASS, throughput much worse than ~1.4× | CK deterministic bwd has unexpected overhead; investigate kernel selection before committing to FFI. |
| Test 1 FAIL | The CK `_deterministic` kernel itself is buggy on gfx950. File a bug at https://github.com/ROCm/flash-attention/issues with the test output. Pivot to one of: Triton (aiter) backend (set `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` and re-run the same test), Pallas/JAX-native flash-attention via the sibling `probe_attention_kernels.sh`, or wait for the upstream fix. |
| Test 1 FAIL with `flash_attn` import error | The container doesn't have a working flash-attn install. Either pre-build an image with it, or pass `FA_WHEEL_URL=...` to use a release wheel. |

## What I cannot deliver in this repo without GPU iteration

- A `jax.pure_callback` + `jax.dlpack` + `jax.custom_vjp` wrapper that
  matches Flax `DotProductAttention`'s signature (mask, dropout RNG keys,
  GQA, packed sequences). Days of iteration; needs a GPU to debug each
  edge case.
- A C++ FFI shim against ROCm/flash-attention's `csrc/flash_attn_rocm/`
  headers, registered via `jax.experimental.ffi.ffi_call`. Bigger lift.
- `custom_partitioning` rules so the new attention works under SPMD/sharding.
- Docker-image extension that bakes flash-attn into `rocm/jax-training:maxtext-v26.2`.

Each of those is doable; none is a single-chat-turn task. After this smoke
test passes, I'd recommend tackling them in this order, with a check-in
after each:

1. Bake `flash-attn` into a derivative training image (Dockerfile next to
   `skills/deterministic-training/Dockerfile`). ~1 day.
2. Single-GPU `pure_callback`-based wrapper of `flash_attn_func` for MaxText.
   ~2–3 days.
3. Verify single-GPU MaxText training is bit-exact with this path active.
   ~1 day.
4. Multi-GPU sharding via `custom_partitioning`. ~3–4 days.
5. Replace with C++ FFI for production performance. ~3–5 days.
