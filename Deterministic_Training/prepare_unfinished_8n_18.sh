#!/usr/bin/env bash
set -euo pipefail

# Prepare and optionally submit the unfinished 18-job 8-node campaign:
# - 6 smoke jobs (steps=5)
# - 12 feasibility jobs (steps=20)
#
# Controls:
# - deterministic:   DETERMINISTIC_MODE=1
# - non-deterministic control: DETERMINISTIC_MODE=0 + NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
#
# Usage:
#   ./Deterministic_Training/prepare_unfinished_8n_18.sh            # dry-run (default)
#   ./Deterministic_Training/prepare_unfinished_8n_18.sh --submit   # submit to Slurm
#   DOCKER_IMAGE=/path/to/image.tar ./Deterministic_Training/prepare_unfinished_8n_18.sh --submit

SUBMIT=0
if [[ "${1:-}" == "--submit" ]]; then
  SUBMIT=1
fi

DOCKER_IMAGE="${DOCKER_IMAGE:-/mnt/vast/qiangh/clean/maxtext-v26.2-det-te508-aot.tar}"
SMOKE_TO="${SMOKE_TO:-preflight:900,pull:1800,ecc:300,train:7200}"
FEAS_TO="${FEAS_TO:-preflight:900,pull:1800,ecc:300,train:14400}"
PARTITION="${PARTITION:-k8s}"

WORKSPACE="/mnt/vast/qiangh/clean/maxtext-slurm"
REQUIRED_NODES=8

# Candidate nodes from previous 8N apple-to-apple cohort.
CANDIDATES=(
  chi2774
  chi2798
  chi2800
  chi2832
  chi2865
  chi2867
  chi2868
  chi2872
  chi2879
)

choose_nodes() {
  local -a healthy=()
  local n
  for n in "${CANDIDATES[@]}"; do
    if ssh -o BatchMode=yes -o ConnectTimeout=8 "$n" "test -d \"$WORKSPACE\"" >/dev/null 2>&1; then
      healthy+=("$n")
    fi
  done

  if [[ ${#healthy[@]} -lt ${REQUIRED_NODES} ]]; then
    echo "ERROR: only ${#healthy[@]} healthy nodes found with workspace path visible." >&2
    echo "Healthy: ${healthy[*]:-<none>}" >&2
    echo "Need at least ${REQUIRED_NODES} nodes before submitting 8N jobs." >&2
    return 1
  fi

  local selected=("${healthy[@]:0:${REQUIRED_NODES}}")
  local nodelist
  nodelist="$(IFS=,; echo "${selected[*]}")"
  echo "$nodelist"
}

NODELIST="$(choose_nodes)"
SBATCH_COMMON=(-p "$PARTITION" -N 8 --nodelist="$NODELIST" --exclusive --gpus-per-task=0)

run_cmd() {
  local label="$1"
  shift
  echo
  echo "=== ${label} ==="
  echo "$*"
  if [[ $SUBMIT -eq 1 ]]; then
    eval "$@"
  fi
}

echo "Mode: $([[ $SUBMIT -eq 1 ]] && echo submit || echo dry-run)"
echo "Docker image: ${DOCKER_IMAGE}"
echo "Partition: ${PARTITION}"
echo "NodeList: ${NODELIST}"
echo "Smoke timeouts: ${SMOKE_TO}"
echo "Feasibility timeouts: ${FEAS_TO}"

# 6 smoke (steps=5)
run_cmd "A-smoke-nd-r7" \
  "DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS='${SMOKE_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'llama3.1-405b:8nA-smk-nd-r7' ${SBATCH_COMMON[*]} --time=02:30:00 -- steps=5 per_device_batch_size=2 enable_dropout=False"
run_cmd "A-smoke-det-r7" \
  "DETERMINISTIC_MODE=1 STAGE_TIMEOUTS='${SMOKE_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'llama3.1-405b:8nA-smk-det-r7' ${SBATCH_COMMON[*]} --time=02:30:00 -- steps=5 per_device_batch_size=2 enable_dropout=False"
run_cmd "B-smoke-nd-r7" \
  "DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS='${SMOKE_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nB-smk-nd-r7' ${SBATCH_COMMON[*]} --time=02:30:00 -- steps=5 per_device_batch_size=4 sparse_matmul=False enable_dropout=False"
run_cmd "B-smoke-det-r7" \
  "DETERMINISTIC_MODE=1 STAGE_TIMEOUTS='${SMOKE_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nB-smk-det-r7' ${SBATCH_COMMON[*]} --time=02:30:00 -- steps=5 per_device_batch_size=4 sparse_matmul=False enable_dropout=False"
run_cmd "C-smoke-nd-r7" \
  "DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS='${SMOKE_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nC-smk-nd-r7' ${SBATCH_COMMON[*]} --time=02:30:00 -- steps=5 per_device_batch_size=1 sparse_matmul=True shardy=True enable_dropout=False"
run_cmd "C-smoke-det-r7" \
  "DETERMINISTIC_MODE=1 STAGE_TIMEOUTS='${SMOKE_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nC-smk-det-r7' ${SBATCH_COMMON[*]} --time=02:30:00 -- steps=5 per_device_batch_size=1 sparse_matmul=True shardy=True enable_dropout=False"

# 12 feasibility (steps=20)
run_cmd "A-feas-nd-p3-r7" \
  "DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'llama3.1-405b:8nA-fes-nd-p3-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=3 enable_dropout=False"
run_cmd "A-feas-det-p3-r7" \
  "DETERMINISTIC_MODE=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'llama3.1-405b:8nA-fes-det-p3-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=3 enable_dropout=False"
run_cmd "A-feas-nd-p5-r7" \
  "DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'llama3.1-405b:8nA-fes-nd-p5-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=5 enable_dropout=False"
run_cmd "A-feas-det-p5-r7" \
  "DETERMINISTIC_MODE=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'llama3.1-405b:8nA-fes-det-p5-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=5 enable_dropout=False"

run_cmd "B-feas-nd-p3-r7" \
  "DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nB-fes-nd-p3-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=3 sparse_matmul=False enable_dropout=False"
run_cmd "B-feas-det-p3-r7" \
  "DETERMINISTIC_MODE=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nB-fes-det-p3-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=3 sparse_matmul=False enable_dropout=False"
run_cmd "B-feas-nd-p4-r7" \
  "DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nB-fes-nd-p4-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=4 sparse_matmul=False enable_dropout=False"
run_cmd "B-feas-det-p4-r7" \
  "DETERMINISTIC_MODE=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nB-fes-det-p4-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=4 sparse_matmul=False enable_dropout=False"

run_cmd "C-feas-nd-p1-r7" \
  "DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nC-fes-nd-p1-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=1 sparse_matmul=True shardy=True enable_dropout=False"
run_cmd "C-feas-det-p1-r7" \
  "DETERMINISTIC_MODE=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nC-fes-det-p1-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=1 sparse_matmul=True shardy=True enable_dropout=False"
run_cmd "C-feas-nd-p2-r7" \
  "DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nC-fes-nd-p2-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=2 sparse_matmul=True shardy=True enable_dropout=False"
run_cmd "C-feas-det-p2-r7" \
  "DETERMINISTIC_MODE=1 STAGE_TIMEOUTS='${FEAS_TO}' DOCKER_IMAGE='${DOCKER_IMAGE}' ./submit.sh 'deepseek3-671b:8nC-fes-det-p2-r7' ${SBATCH_COMMON[*]} --time=04:00:00 -- steps=20 per_device_batch_size=2 sparse_matmul=True shardy=True enable_dropout=False"

echo
echo "Prepared 18 commands."
if [[ $SUBMIT -eq 0 ]]; then
  echo "Dry-run complete. Re-run with --submit to enqueue."
fi
