"""Deterministic training support for MaxText on ROCm.

Provides runtime patches and verification to achieve bit-exact reproducible
training.  Activated by DETERMINISTIC_MODE=1 (set in train_env.sh).

Usage from the training entry point (mfu_tracker.py):

    from utils import deterministic

    deterministic.apply_patches(maxtext_train)   # before maxtext_train.main()
    maxtext_train.main(...)
    deterministic.print_loss_checksum()          # after training completes
"""

import hashlib
import os
import re
import struct

_LOSS_RE = re.compile(r"completed step:\s*\d+,.*loss:\s+([\d.]+)")


# ---------------------------------------------------------------------------
# Loss checksum tracker
# ---------------------------------------------------------------------------

class LossChecksumTracker:
    """Accumulates loss values into a running SHA-256 hash.

    Identical checksums across runs prove bit-exact reproducibility without
    needing TensorBoard event parsing.
    """

    def __init__(self):
        self._hasher = hashlib.sha256()
        self._count = 0

    def update(self, loss_str: str):
        self._hasher.update(struct.pack("!f", float(loss_str)))
        self._count += 1

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()

    @property
    def count(self) -> int:
        return self._count


tracker = LossChecksumTracker()


def extract_loss(text: str):
    """If *text* contains a MaxText loss log line, feed it to the tracker."""
    m = _LOSS_RE.search(text)
    if m:
        tracker.update(m.group(1))


# ---------------------------------------------------------------------------
# Runtime patches
# ---------------------------------------------------------------------------

def apply_patches(maxtext_train):
    """Monkey-patch MaxText's initialize() for deterministic mode.

    Two things that cannot be fixed via env vars alone:
      1. MaxText hardcodes unsafe_rbg PRNG — must be overridden post-init.
      2. Runtime verification — confirm env vars survived initialization.
    """
    is_deterministic = os.environ.get("DETERMINISTIC_MODE", "0").strip()
    prng_impl = os.environ.get("JAX_DEFAULT_PRNG_IMPL", "").strip()

    if is_deterministic.lower() not in ("1", "y", "yes", "true") and not prng_impl:
        return

    _orig_initialize = maxtext_train.initialize

    def _patched_initialize(argv):
        result = _orig_initialize(argv)

        import jax

        if prng_impl:
            jax.config.update("jax_default_prng_impl", prng_impl)
            print(f"[deterministic] PRNG override: jax_default_prng_impl={prng_impl} "
                  f"(was unsafe_rbg)", flush=True)

        verify_env()
        return result

    maxtext_train.initialize = _patched_initialize


# ---------------------------------------------------------------------------
# Runtime verification
# ---------------------------------------------------------------------------

def verify_env():
    """Post-init sanity checks — warn if any deterministic flag was clobbered."""
    tag = "[deterministic]"
    warnings = []

    checks = [
        ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0",
         "TE fused attention may be non-deterministic."),
        ("TF_DETERMINISTIC_OPS", "1",
         "rocBLAS may use non-deterministic atomic reductions."),
        ("HIPBLASLT_DETERMINISTIC", "1",
         "hipBLASLt may select non-deterministic atomic-GSU GEMM solutions."),
    ]
    for var, expected, msg in checks:
        actual = os.environ.get(var, "")
        if actual != expected:
            warnings.append(f"{var}={actual!r} (expected {expected!r}). {msg}")

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_deterministic_ops" not in xla_flags:
        warnings.append(
            "XLA_FLAGS missing --xla_gpu_deterministic_ops. "
            "GEMM autotuning and scatter ops may be non-deterministic.")

    nvte_fused = os.environ.get("NVTE_FUSED_ATTN", "1")
    if nvte_fused != "0":
        warnings.append(
            f"NVTE_FUSED_ATTN={nvte_fused!r} (expected '0'). "
            "CK fused attention backward is non-deterministic — "
            "results will NOT be bit-exact.")

    try:
        import jax
        actual_prng = jax.config.jax_default_prng_impl
        if actual_prng != "threefry2x32":
            warnings.append(
                f"jax_default_prng_impl={actual_prng!r} (expected 'threefry2x32'). "
                "PRNG may not be deterministic across backends.")
    except Exception:
        pass

    if not warnings:
        print(f"{tag} All env-var checks passed.", flush=True)
    for w in warnings:
        print(f"{tag} WARNING: {w}", flush=True)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def print_loss_checksum():
    """Print the accumulated loss checksum if any steps were recorded."""
    if tracker.count > 0:
        print(f"[determinism] loss_checksum={tracker.hexdigest()[:16]} "
              f"(steps={tracker.count})", flush=True)
