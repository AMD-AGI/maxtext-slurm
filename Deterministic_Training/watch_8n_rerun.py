#!/usr/bin/env python3
import glob
import os
import re
import subprocess
import time
from typing import Dict, List, Set, Tuple


WORKSPACE = "/mnt/vast/qiangh/clean/maxtext-slurm"
OUTPUTS_DIR = os.path.join(WORKSPACE, "outputs")
POLL_SECONDS = 60

# Candidate pool used in prior 8N rerun attempts.
CANDIDATE_NODES = [
    "chi2774",
    "chi2798",
    "chi2800",
    "chi2832",
    "chi2865",
    "chi2867",
    "chi2868",
    "chi2872",
    "chi2879",
]


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def healthy_nodes() -> List[str]:
    ok = []
    for node in CANDIDATE_NODES:
        rc, _, _ = run_cmd(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                node,
                f"test -d {WORKSPACE}",
            ]
        )
        if rc == 0:
            ok.append(node)
    return ok


def list_8n_jobs() -> List[Tuple[str, str, str, str, str]]:
    rc, out, _ = run_cmd(
        ["squeue", "-u", "root", "-h", "-o", "%i|%T|%D|%R|%j"]
    )
    if rc != 0:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        jid, state, nodes, reason, name = parts
        if any(tag in name for tag in ("8nA-", "8nB-", "8nC-")):
            rows.append((jid, state, nodes, reason, name))
    return rows


def sacct_state(job_id: str) -> str:
    rc, out, _ = run_cmd(
        [
            "sacct",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ExitCode",
            "-n",
            "-P",
        ]
    )
    if rc != 0:
        return "UNKNOWN"
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        jid, state, _exit = parts[:3]
        if jid == job_id:
            return state
    return "UNKNOWN"


def last_log_status(log_path: str) -> str:
    try:
        with open(log_path, "r", errors="ignore") as f:
            text = f.read()
    except OSError:
        return "UNREADABLE"
    matches = re.findall(r"^\s*Status:\s*(.+)$", text, flags=re.M)
    return matches[-1].strip() if matches else "NO_SUMMARY"


def main() -> None:
    # Baseline existing logs so we only alert on new rerun logs.
    seen_logs: Set[str] = set(glob.glob(os.path.join(OUTPUTS_DIR, "*8n*.log")))

    seen_job_state: Dict[str, str] = {}
    done_jobs: Set[str] = set()

    prev_ready = None
    prev_healthy = []
    prev_active_ids: Set[str] = set()

    print("WATCH_8N_STARTED")

    while True:
        # 1) Resource readiness.
        healthy = healthy_nodes()
        ready = len(healthy) >= 8
        if (ready != prev_ready) or (healthy != prev_healthy):
            if ready:
                picked = ",".join(healthy[:8])
                print(f"READY_8N healthy={len(healthy)} nodelist={picked}", flush=True)
            else:
                print(
                    f"NOT_READY_8N healthy={len(healthy)} need=8 nodes={','.join(healthy)}",
                    flush=True,
                )
            prev_ready = ready
            prev_healthy = healthy

        # 2) Queue/running state transitions.
        jobs = list_8n_jobs()
        active_ids = set()
        for jid, state, nodes, reason, name in jobs:
            active_ids.add(jid)
            prev_state = seen_job_state.get(jid)
            if prev_state != state:
                print(
                    f"JOB_8N jid={jid} state={state} nodes={nodes} reason={reason} name={name}",
                    flush=True,
                )
                seen_job_state[jid] = state

        # 3) Completed jobs from previous active set.
        for jid in sorted(prev_active_ids - active_ids):
            if jid in done_jobs:
                continue
            state = sacct_state(jid)
            if state == "COMPLETED":
                print(f"DONE_8N jid={jid} state={state}", flush=True)
            else:
                print(f"ALERT_8N jid={jid} terminal_state={state}", flush=True)
            done_jobs.add(jid)
        prev_active_ids = active_ids

        # 4) New 8N logs and status.
        current_logs = set(glob.glob(os.path.join(OUTPUTS_DIR, "*8n*.log")))
        new_logs = sorted(current_logs - seen_logs)
        for log_path in new_logs:
            status = last_log_status(log_path)
            name = os.path.basename(log_path)
            if "FAILED" in status or "CANCELLED" in status or "TIMEOUT" in status:
                print(f"ALERT_LOG_8N log={name} status={status}", flush=True)
            elif "SUCCESS" in status or "COMPLETED" in status:
                print(f"DONE_LOG_8N log={name} status={status}", flush=True)
            else:
                print(f"INFO_LOG_8N log={name} status={status}", flush=True)
        seen_logs = current_logs

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
