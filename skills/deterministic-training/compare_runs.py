#!/usr/bin/env python3
"""Compare loss values across deterministic training runs at full float32 precision.

Usage:
    # Compare the 2-3 most recent deterministic runs:
    python3 compare_runs.py

    # Compare specific runs by job ID prefix or directory:
    python3 compare_runs.py outputs/local_20260311_000810* outputs/local_20260311_001200*

    # Compare with explicit glob:
    python3 compare_runs.py --glob '*DETERMINISTIC*steps_15*'
"""

import argparse
import glob
import os
import struct
import sys

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


def find_tb_events(run_dir):
    """Find TensorBoard event files under a run directory."""
    pattern = os.path.join(run_dir, "**", "events.out.tfevents.*")
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def read_loss_from_tb(event_file, tag="learning/loss"):
    """Read scalar values from a TensorBoard event file.

    Returns list of (step, value, wall_time) tuples.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        sys.exit("ERROR: tensorboard not installed. Run: pip install tensorboard")

    ea = EventAccumulator(os.path.dirname(event_file))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        available = [t for t in tags if "loss" in t.lower()]
        print(f"  WARNING: tag '{tag}' not found. Available loss tags: {available}")
        return []

    events = ea.Scalars(tag)
    return [(e.step, e.value, e.wall_time) for e in events]


def read_loss_from_log(log_file):
    """Fallback: parse loss from log file (lower precision, 3 decimal places)."""
    import re
    pattern = re.compile(r"completed step:\s*(\d+),.*loss:\s*([\d.]+)")
    results = []
    with open(log_file) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                results.append((int(m.group(1)), float(m.group(2)), 0.0))
    return results


def find_recent_det_runs(outputs_dir, pattern="*DETERMINISTIC*", limit=3):
    """Find recent deterministic-mode run directories."""
    dirs = sorted(glob.glob(os.path.join(outputs_dir, pattern)),
                  key=os.path.getmtime, reverse=True)
    run_dirs = [d for d in dirs if os.path.isdir(d)]
    return run_dirs[:limit]


def format_value(v, precision=10):
    return f"{v:.{precision}f}"


def compare(runs_data, tolerance=0.0):
    """Compare loss values across runs. Returns True if all match within tolerance."""
    if len(runs_data) < 2:
        print("Need at least 2 runs to compare.")
        return False

    names = list(runs_data.keys())
    ref_name = names[0]
    ref_data = {step: val for step, val, _ in runs_data[ref_name]}

    all_steps = sorted(ref_data.keys())
    if not all_steps:
        print("No loss data found in reference run.")
        return False

    header = f"{'step':>5s}"
    for name in names:
        short = name.split("-")[0] if "-" in name else name[:20]
        header += f"  {short:>20s}"
    if len(names) > 1:
        header += f"  {'max_delta':>14s}  {'status':>8s}"
    print(header)
    print("-" * len(header))

    all_match = True
    for step in all_steps:
        values = []
        for name in names:
            step_map = {s: v for s, v, _ in runs_data[name]}
            values.append(step_map.get(step))

        row = f"{step:5d}"
        for v in values:
            if v is not None:
                row += f"  {v:20.10f}"
            else:
                row += f"  {'(missing)':>20s}"

        present = [v for v in values if v is not None]
        if len(present) >= 2:
            delta = max(present) - min(present)
            match = delta <= tolerance
            status = "EXACT" if delta == 0.0 else ("OK" if match else "DIFFER")
            row += f"  {delta:14.10f}  {status:>8s}"
            if not match:
                all_match = False
        print(row)

    return all_match


def hex_float(f):
    """Show the exact IEEE-754 representation."""
    return struct.pack("!f", f).hex()


def main():
    parser = argparse.ArgumentParser(description="Compare deterministic training runs")
    parser.add_argument("runs", nargs="*", help="Run directories or log files to compare")
    parser.add_argument("--glob", default=None,
                        help="Glob pattern to find runs in outputs/ (e.g. '*DETERMINISTIC*steps_15*')")
    parser.add_argument("--tag", default="learning/loss", help="TensorBoard scalar tag")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="Maximum allowed absolute difference (default: 0.0 = bit-exact)")
    parser.add_argument("--log-fallback", action="store_true",
                        help="Fall back to log parsing if TensorBoard events are missing")
    parser.add_argument("--hex", action="store_true",
                        help="Also print IEEE-754 hex representation")
    args = parser.parse_args()

    if args.runs:
        run_dirs = []
        for r in args.runs:
            expanded = sorted(glob.glob(r))
            run_dirs.extend(expanded)
    elif args.glob:
        run_dirs = find_recent_det_runs(OUTPUTS_DIR, args.glob)
    else:
        run_dirs = find_recent_det_runs(OUTPUTS_DIR)

    if not run_dirs:
        print("No runs found. Pass run directories as arguments or use --glob.")
        print(f"  Searched: {OUTPUTS_DIR}")
        sys.exit(1)

    # Resolve directories vs log files
    runs_data = {}
    for path in run_dirs:
        if os.path.isfile(path) and path.endswith(".log"):
            name = os.path.basename(path).replace(".log", "")
            print(f"[log] {name}")
            data = read_loss_from_log(path)
            if data:
                runs_data[name] = data
            continue

        if os.path.isdir(path):
            name = os.path.basename(path)
            tb_files = find_tb_events(path)
            if tb_files:
                print(f"[TB]  {name}  ({len(tb_files)} event file(s))")
                data = read_loss_from_tb(tb_files[0], args.tag)
                if data:
                    runs_data[name] = data
                    continue

            # Fallback to log
            log_file = path + ".log"
            if args.log_fallback and os.path.isfile(log_file):
                print(f"[log] {name}  (TB fallback)")
                data = read_loss_from_log(log_file)
                if data:
                    runs_data[name] = data
                    continue

            print(f"[SKIP] {name}  (no TB events{' or log' if args.log_fallback else '; use --log-fallback'})")

    if len(runs_data) < 2:
        print(f"\nOnly {len(runs_data)} run(s) with data. Need at least 2.")
        sys.exit(1)

    print(f"\nComparing {len(runs_data)} runs (tag={args.tag}, tolerance={args.tolerance}):\n")

    ok = compare(runs_data, args.tolerance)

    if args.hex and runs_data:
        print(f"\n{'step':>5s}", end="")
        for name in runs_data:
            short = name.split("-")[0] if "-" in name else name[:20]
            print(f"  {short:>20s}", end="")
        print()
        ref_steps = sorted({s for data in runs_data.values() for s, _, _ in data})
        for step in ref_steps:
            print(f"{step:5d}", end="")
            for name, data in runs_data.items():
                step_map = {s: v for s, v, _ in data}
                v = step_map.get(step)
                if v is not None:
                    print(f"  {hex_float(v):>20s}", end="")
                else:
                    print(f"  {'':>20s}", end="")
            print()

    print()
    if ok:
        if args.tolerance == 0.0:
            print("RESULT: All runs are BIT-EXACT (float32 identical)")
        else:
            print(f"RESULT: All runs match within tolerance={args.tolerance}")
    else:
        print("RESULT: MISMATCH detected — runs are NOT deterministic")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
