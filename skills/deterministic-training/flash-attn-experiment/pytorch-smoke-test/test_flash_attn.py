"""ROCm/flash-attention deterministic-mode smoke test.

What this validates
-------------------
1. ``flash_attn_func(..., deterministic=True)`` produces **bit-identical**
   output and gradients across two independent runs on this GPU. This is
   the single fact the rest of the MaxText integration depends on. If it
   fails, the FFI/bridge work is unjustified.
2. ``flash_attn_func(..., deterministic=False)`` is allowed to differ
   across runs (sanity check on the harness — confirms our equality test
   isn't trivially passing).
3. Median throughput at llama2-70b shapes, deterministic vs non-deterministic,
   so we know the cost of the deterministic backward kernel on this hardware.

What this does NOT validate
---------------------------
- MaxText / JAX integration (PyTorch only).
- Sharding / SPMD behavior (single GPU only).
- Numerical match against any other implementation (just self-consistency).
- Multi-layer / full-model correctness.

This is strictly a "does the kernel work deterministically on this hardware?"
test. Pass = green light to invest in JAX FFI / bridge work. Fail = file
the bug upstream against ROCm/flash-attention; the project's 3-line TE
fix is invalidated for the same reason.

Shapes match llama2-70b's per-device attention block on 8-way TP:
  B=2, H=8, S=4096, D=128, bf16, causal=True.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from contextlib import contextmanager


def _print(msg: str) -> None:
    print(msg, flush=True)


def _check_imports() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        sys.exit(f"FATAL: torch not importable: {e}")
    try:
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_func  # noqa: F401
    except ImportError as e:
        sys.exit(
            "FATAL: flash_attn not importable: "
            f"{e}\n"
            "Install with one of:\n"
            "  pip install flash-attn --no-build-isolation\n"
            "  pip install <wheel-url-from-https://github.com/ROCm/flash-attention/releases>\n"
        )


def _device_info() -> dict:
    import torch
    if not torch.cuda.is_available():
        sys.exit("FATAL: torch.cuda.is_available() == False (no ROCm device visible).")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    return {"name": name, "capability": cap, "count": torch.cuda.device_count()}


@contextmanager
def _timeit():
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0
    torch.cuda.synchronize()


def _make_inputs(B: int, S: int, H: int, D: int, seed: int):
    """Allocate fresh Q/K/V tensors with a fixed seed.

    Each call returns NEW tensors with NEW grad slots so we don't pollute
    state across runs. Uses CUDA's PRNG seeded explicitly.
    """
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, S, H, D, generator=g, dtype=torch.bfloat16,
                    device="cuda", requires_grad=True)
    k = torch.randn(B, S, H, D, generator=g, dtype=torch.bfloat16,
                    device="cuda", requires_grad=True)
    v = torch.randn(B, S, H, D, generator=g, dtype=torch.bfloat16,
                    device="cuda", requires_grad=True)
    return q, k, v


def _one_run(B: int, S: int, H: int, D: int, seed: int, deterministic: bool):
    """Run flash_attn fwd + bwd once. Returns (out, dQ, dK, dV) as bf16 tensors."""
    import torch
    from flash_attn import flash_attn_func

    q, k, v = _make_inputs(B, S, H, D, seed)
    out = flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        causal=True,
        deterministic=deterministic,
    )
    # Use a fixed deterministic upstream gradient so dQ/dK/dV are deterministic
    # functions of the kernel only (not of any extra randomness in `loss.backward()`).
    g = torch.ones_like(out)
    out.backward(g)
    return (out.detach().clone(),
            q.grad.detach().clone(),
            k.grad.detach().clone(),
            v.grad.detach().clone())


def _bytes_equal(a, b) -> bool:
    """Bit-exact comparison of two tensors (same dtype, same shape, same bits)."""
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    # bf16 is fine to compare via .equal — tolerates no NaN/Inf semantics here
    # because we don't expect either in this kernel under these inputs.
    return torch.equal(a, b)


def _max_abs_diff(a, b) -> float:
    import torch
    return float((a.float() - b.float()).abs().max().item())


def test_bit_exact_deterministic_true(B, S, H, D, seed) -> bool:
    _print("")
    _print("=" * 65)
    _print("Test 1 — deterministic=True, 2 runs, expect BIT-EXACT")
    _print("=" * 65)

    out1, dq1, dk1, dv1 = _one_run(B, S, H, D, seed, deterministic=True)
    out2, dq2, dk2, dv2 = _one_run(B, S, H, D, seed, deterministic=True)

    fields = [("out", out1, out2), ("dQ", dq1, dq2),
              ("dK", dk1, dk2), ("dV", dv1, dv2)]
    all_pass = True
    for name, a, b in fields:
        same = _bytes_equal(a, b)
        diff = _max_abs_diff(a, b) if not same else 0.0
        verdict = "BIT-EXACT" if same else f"DIFFER (max |Δ|={diff:g})"
        _print(f"  {name:<3s}: {verdict}")
        all_pass = all_pass and same

    _print(f"RESULT: {'PASS' if all_pass else 'FAIL'} — "
          f"{'kernel is deterministic on this hardware' if all_pass else 'kernel is NOT deterministic; FFI work is invalidated'}")
    return all_pass


def test_nondeterministic_can_differ(B, S, H, D, seed) -> bool:
    """Optional sanity check.

    Two ``deterministic=False`` runs *may* differ. We don't assert they
    do (some inputs / sizes can produce identical results by chance), but
    we report what we see so the operator can confirm the comparison
    machinery would catch real non-determinism if it existed.
    """
    _print("")
    _print("=" * 65)
    _print("Test 2 — deterministic=False, 2 runs, harness sanity check")
    _print("=" * 65)

    out1, dq1, dk1, dv1 = _one_run(B, S, H, D, seed, deterministic=False)
    out2, dq2, dk2, dv2 = _one_run(B, S, H, D, seed, deterministic=False)

    differs_anywhere = False
    for name, a, b in [("out", out1, out2), ("dQ", dq1, dq2),
                       ("dK", dk1, dk2), ("dV", dv1, dv2)]:
        same = _bytes_equal(a, b)
        diff = _max_abs_diff(a, b) if not same else 0.0
        _print(f"  {name:<3s}: "
              f"{'identical' if same else f'differs (max |Δ|={diff:g})'}")
        differs_anywhere = differs_anywhere or (not same)

    _print(f"RESULT: {'observed non-determinism (harness works)' if differs_anywhere else 'all identical (harness can not distinguish — try a larger config)'}")
    return True


def test_throughput(B, S, H, D, seed, n_iter, deterministic) -> float:
    """Median wall-time per fwd+bwd over `n_iter` iterations, after warmup."""
    import torch
    from flash_attn import flash_attn_func

    q, k, v = _make_inputs(B, S, H, D, seed)
    g = torch.ones(B, S, H, D, dtype=torch.bfloat16, device="cuda")

    # 3 warmup iterations
    for _ in range(3):
        if q.grad is not None:
            q.grad = None; k.grad = None; v.grad = None
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True,
                              deterministic=deterministic)
        out.backward(g)
    torch.cuda.synchronize()

    timings = []
    for _ in range(n_iter):
        if q.grad is not None:
            q.grad = None; k.grad = None; v.grad = None
        with _timeit() as elapsed:
            out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True,
                                  deterministic=deterministic)
            out.backward(g)
        timings.append(elapsed())

    return statistics.median(timings)


def _attn_flops(B, S, H, D) -> float:
    """Theoretical fwd+bwd FLOPs for causal attention.

    fwd: 2 * B * H * S^2 * D (Q@K^T + (softmax * V))
    bwd: 5 * B * H * S^2 * D (rough, accounts for dQ, dK, dV plus softmax bwd)
    Causal halves the fwd, but bwd recomputes; a common rule-of-thumb is
    causal ≈ fwd/2 + bwd ≈ 3.5 * B * H * S^2 * D for fwd+bwd.
    """
    return 3.5 * B * H * S * S * D


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=int(os.environ.get("FA_BATCH", "2")))
    ap.add_argument("--seq", type=int, default=int(os.environ.get("FA_SEQ", "4096")))
    ap.add_argument("--heads", type=int, default=int(os.environ.get("FA_HEADS", "8")))
    ap.add_argument("--head-dim", type=int, default=int(os.environ.get("FA_HEAD_DIM", "128")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("FA_SEED", "42")))
    ap.add_argument("--bench-iters", type=int, default=int(os.environ.get("FA_BENCH_ITERS", "20")))
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-nondet", action="store_true")
    args = ap.parse_args()

    _check_imports()

    info = _device_info()
    _print("ROCm/flash-attention smoke test")
    _print(f"Device: {info['name']}  (cap={info['capability']}, count={info['count']})")
    _print(f"Shapes: B={args.batch} S={args.seq} H={args.heads} D={args.head_dim}  "
          f"dtype=bf16 causal=True dropout=0.0")
    _print(f"Seed:   {args.seed}")

    # -----------------------------------------------------------------
    # Test 1: bit-exact determinism (the actual decision-relevant test)
    # -----------------------------------------------------------------
    pass1 = test_bit_exact_deterministic_true(
        args.batch, args.seq, args.heads, args.head_dim, args.seed)

    # -----------------------------------------------------------------
    # Test 2: harness sanity check
    # -----------------------------------------------------------------
    if not args.skip_nondet:
        test_nondeterministic_can_differ(
            args.batch, args.seq, args.heads, args.head_dim, args.seed)

    # -----------------------------------------------------------------
    # Test 3: throughput
    # -----------------------------------------------------------------
    if not args.skip_bench:
        flops_per_iter = _attn_flops(args.batch, args.seq, args.heads, args.head_dim)
        _print("")
        _print("=" * 65)
        _print(f"Test 3 — throughput, median over {args.bench_iters} iterations")
        _print("=" * 65)

        for det in (True, False):
            t = test_throughput(args.batch, args.seq, args.heads, args.head_dim,
                                args.seed, args.bench_iters, det)
            tflops = flops_per_iter / t / 1e12
            label = "deterministic=True " if det else "deterministic=False"
            _print(f"  {label}: {t*1000:8.2f} ms/iter   "
                  f"~{tflops:7.1f} TFLOP/s (causal fwd+bwd, bf16)")

    _print("")
    if pass1:
        _print("OVERALL: PASS — ROCm/flash-attention deterministic backward is "
              "bit-exact on this hardware. The JAX FFI / bridge integration is "
              "justified.")
        return 0
    else:
        _print("OVERALL: FAIL — ROCm/flash-attention deterministic backward is "
              "NOT bit-exact on this hardware. File a bug upstream; do NOT proceed "
              "with the JAX FFI work until the kernel itself is fixed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
