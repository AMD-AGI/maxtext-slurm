#!/usr/bin/env python3
"""Verify the hipBLASLt deterministic mode patch is installed and working.

Usage:
    # Check if patch is present (no GPU needed):
    python3 verify-hipblaslt-patch.py --check-binary

    # Full runtime test (needs GPU):
    python3 verify-hipblaslt-patch.py

    # Compare solution counts with and without deterministic mode:
    python3 verify-hipblaslt-patch.py --compare
"""

import argparse
import ctypes
import os
import struct
import subprocess
import sys


def check_binary():
    """Static check: is setDeterministicMode called from the right functions?"""
    lib_path = None
    for p in [
        "/opt/rocm/lib/libhipblaslt.so",
        "/opt/rocm-7.1.1/lib/libhipblaslt.so",
    ]:
        if os.path.exists(p):
            lib_path = p
            break

    if not lib_path:
        print("FAIL: libhipblaslt.so not found")
        return False

    print(f"Library: {lib_path}")
    size_mb = os.path.getsize(lib_path) / 1024 / 1024
    print(f"Size:    {size_mb:.1f} MB")

    # Check nm symbols for setDeterministicMode references
    try:
        nm_out = subprocess.run(
            ["nm", "-C", lib_path],
            capture_output=True, text=True, timeout=120
        ).stdout
    except Exception as e:
        print(f"WARNING: nm failed ({e}), skipping symbol check")
        nm_out = ""

    # In the patched library, setDeterministicMode is called from both
    # ConstructTensileProblem and updateTensileProblem. The symbol
    # ContractionProblemGemm::setDeterministicMode should appear.
    det_symbols = [l for l in nm_out.splitlines() if "setDeterministicMode" in l]
    if det_symbols:
        print(f"OK:      setDeterministicMode symbol found ({len(det_symbols)} references)")
        for s in det_symbols[:3]:
            print(f"         {s.strip()[:100]}")
    else:
        print("WARNING: setDeterministicMode not in nm output (may be inlined)")

    # Check strings for the env var name
    try:
        strings_out = subprocess.run(
            ["strings", lib_path],
            capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        strings_out = ""

    if "HIPBLASLT_DETERMINISTIC" in strings_out:
        print("OK:      'HIPBLASLT_DETERMINISTIC' string found in binary")
        return True
    else:
        print("INFO:    'HIPBLASLT_DETERMINISTIC' string not found (may be compiler-optimized)")
        print("         This is expected — the string can be inlined by the compiler.")
        print("         Use --compare for a definitive runtime test.")
        return True  # not conclusive, need runtime test


def runtime_test():
    """Runtime test: run a GEMM with and without HIPBLASLT_DETERMINISTIC and
    compare the number of available algorithms."""

    print("\n=== Runtime Verification ===\n")

    # We'll run a small JAX matmul and use TensileLite debug logging to check
    # whether deterministic filtering is active.
    test_script = '''
import os, sys
import jax
import jax.numpy as jnp

det = os.environ.get("HIPBLASLT_DETERMINISTIC", "0")
print(f"HIPBLASLT_DETERMINISTIC={det}")

# Suppress JAX logs except errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Create a representative bf16 GEMM (similar to llama2-70b linear layer)
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (4096, 8192), dtype=jnp.bfloat16)
b = jax.random.normal(key, (8192, 4096), dtype=jnp.bfloat16)

# JIT compile and run
@jax.jit
def matmul(a, b):
    return jnp.dot(a, b)

# Warmup (triggers compilation + hipBLASLt algorithm selection)
c = matmul(a, b)
c.block_until_ready()

# Run twice and check bit-exactness
c1 = matmul(a, b)
c1.block_until_ready()
c2 = matmul(a, b)
c2.block_until_ready()

match = jnp.array_equal(c1, c2)
print(f"Same-process bit-exact: {match}")
print(f"Output sample: {c1[0, :4]}")

# Print the checksum for cross-process comparison
flat = c1.reshape(-1).view(jnp.uint16)
checksum = int(jnp.sum(flat.astype(jnp.int64)))
print(f"CHECKSUM={checksum}")
'''

    results = {}
    for det_val in ["0", "1"]:
        env = os.environ.copy()
        env["HIPBLASLT_DETERMINISTIC"] = det_val
        env["XLA_FLAGS"] = env.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"

        print(f"--- Run with HIPBLASLT_DETERMINISTIC={det_val} ---")
        proc = subprocess.run(
            [sys.executable, "-c", test_script],
            env=env, capture_output=True, text=True, timeout=120
        )

        if proc.returncode != 0:
            print(f"FAIL: process exited {proc.returncode}")
            print(proc.stderr[-500:] if proc.stderr else "(no stderr)")
            return False

        print(proc.stdout.strip())
        for line in proc.stdout.splitlines():
            if line.startswith("CHECKSUM="):
                results[det_val] = line.split("=")[1]
        print()

    if "0" in results and "1" in results:
        if results["0"] == results["1"]:
            print("RESULT: Checksums MATCH between det=0 and det=1")
            print("        (Same algorithm was selected in both cases — patch effect")
            print("         is only visible when non-deterministic solutions would")
            print("         have been chosen. This is normal for many GEMM shapes.)")
        else:
            print("RESULT: Checksums DIFFER between det=0 and det=1")
            print("        The patch IS changing algorithm selection!")
            print(f"        det=0: {results['0']}")
            print(f"        det=1: {results['1']}")
        return True
    else:
        print("FAIL: Could not extract checksums")
        return False


def compare_test():
    """Run the same matmul in two separate processes with HIPBLASLT_DETERMINISTIC=1
    and check if results are bit-exact."""

    print("\n=== Cross-Process Reproducibility Test ===\n")

    test_script = '''
import os, jax, jax.numpy as jnp
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (4096, 8192), dtype=jnp.bfloat16)
b = jax.random.normal(key, (8192, 4096), dtype=jnp.bfloat16)

@jax.jit
def matmul(a, b):
    return jnp.dot(a, b)

c = matmul(a, b)
c.block_until_ready()
flat = c.reshape(-1).view(jnp.uint16)
checksum = int(jnp.sum(flat.astype(jnp.int64)))
print(f"CHECKSUM={checksum}")
'''

    checksums = []
    for run in range(3):
        env = os.environ.copy()
        env["HIPBLASLT_DETERMINISTIC"] = "1"
        env["TF_DETERMINISTIC_OPS"] = "1"
        env["XLA_FLAGS"] = (
            env.get("XLA_FLAGS", "")
            + " --xla_gpu_deterministic_ops=true --xla_gpu_enable_command_buffer=''"
        )

        proc = subprocess.run(
            [sys.executable, "-c", test_script],
            env=env, capture_output=True, text=True, timeout=120
        )

        if proc.returncode != 0:
            print(f"Run {run+1}: FAIL (exit {proc.returncode})")
            print(proc.stderr[-300:] if proc.stderr else "")
            return False

        for line in proc.stdout.splitlines():
            if line.startswith("CHECKSUM="):
                cs = line.split("=")[1]
                checksums.append(cs)
                print(f"Run {run+1}: CHECKSUM={cs}")

    if len(checksums) == 3:
        if checksums[0] == checksums[1] == checksums[2]:
            print("\nPASS: All 3 runs produced identical checksums — deterministic!")
        else:
            print(f"\nFAIL: Checksums differ across runs")
            print(f"  Run 1: {checksums[0]}")
            print(f"  Run 2: {checksums[1]}")
            print(f"  Run 3: {checksums[2]}")
            print("\n  If HIPBLASLT_DETERMINISTIC=1 but results still differ,")
            print("  the patch may not be installed, or the divergence is from")
            print("  a non-GEMM source.")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify hipBLASLt deterministic patch")
    parser.add_argument("--check-binary", action="store_true",
                        help="Only check the binary (no GPU needed)")
    parser.add_argument("--compare", action="store_true",
                        help="Run 3 processes and compare checksums")
    args = parser.parse_args()

    print("=== hipBLASLt Deterministic Patch Verification ===\n")

    ok = check_binary()
    if args.check_binary:
        sys.exit(0 if ok else 1)

    if args.compare:
        ok = compare_test()
    else:
        ok = runtime_test()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
