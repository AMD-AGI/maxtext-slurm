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

    if "xla_gpu_autotune_level" not in xla_flags:
        warnings.append(
            "XLA_FLAGS missing --xla_gpu_autotune_level=0. "
            "XLA may select different kernels across runs, breaking reproducibility.")
    elif "xla_gpu_autotune_level=0" not in xla_flags:
        warnings.append(
            "XLA_FLAGS has xla_gpu_autotune_level != 0. "
            "Non-zero autotune levels can select different kernels across runs.")

    # CK fused attention determinism check.
    #   Post-PR-#508 image (TE >= 2.12.0.dev0+8943023d): NVTE_FUSED_ATTN=1
    #   uses the CK `_deterministic` kernel variants (per-split dQ + ordered
    #   reduce). NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 (already checked above)
    #   triggers the deterministic dispatch.
    #   Pre-PR-#508 image: TE hardcodes `false` so NVTE_FUSED_ATTN=1 is
    #   non-deterministic and only NVTE_FUSED_ATTN=0 (unfused JAX-native
    #   workaround) gives bit-exactness.
    # We can't programmatically tell which image the user is on, so emit an
    # informational note rather than a hard warning. The bit-exactness of any
    # given run is verified post-hoc by loss_checksum + compare_runs.py
    # regardless of which path is active.
    nvte_fused = os.environ.get("NVTE_FUSED_ATTN", "1")
    nvte_ck = os.environ.get("NVTE_FUSED_ATTN_CK", "1")
    if nvte_fused == "0":
        print(f"{tag} NVTE_FUSED_ATTN=0: legacy unfused workaround (works on any "
              f"image; ~9.7x slower; requires per_device_batch_size=1 to avoid OOM).",
              flush=True)
    elif nvte_fused == "1" and nvte_ck == "1":
        print(f"{tag} NVTE_FUSED_ATTN=1, NVTE_FUSED_ATTN_CK=1: CK deterministic "
              f"path (requires PR-#508 image; ~6% slower than non-deterministic).",
              flush=True)

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
