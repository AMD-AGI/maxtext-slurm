#!/bin/bash

# Container environment configuration.
# Edit this file to switch images, paths, or deployment-specific settings.
# Sourced by _container.sh before launching the container
# and by in_container_run.sh for MAXTEXT_REPO_DIR and MAXTEXT_PATCH_BRANCH.
# All variables can be overridden from the command line, e.g.:
#   DOCKER_IMAGE=my/image:tag ./run_local.sh model_name -- ...

# ── Registry credentials (private images only) ────────────────────────────────
# For private images, copy the template and fill in your credentials:
#   cp container_env.local.template container_env.local.sh
# container_env.local.sh is gitignored — credentials are never committed.
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
if [[ -f "${BASH_SOURCE[0]%/*}/container_env.local.sh" ]]; then
    source "${BASH_SOURCE[0]%/*}/container_env.local.sh"
    echo "[INFO] Loaded registry credentials from container_env.local.sh"
fi
# ── end Registry credentials ──────────────────────────────────────────────────

# ── Docker image ──────────────────────────────────────────────────────────────
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/jax-training:maxtext-v26.4-jax0.9.1-te2.12.0}"
USE_DOCKER_IMAGE_AINIC_DRIVER="${USE_DOCKER_IMAGE_AINIC_DRIVER:-true}"    # Use the container's built-in AINIC driver; set to false to bind-mount host IB libs instead (needed when container libionic1 mismatches host firmware)
MAXTEXT_REPO_DIR="${MAXTEXT_REPO_DIR:-/workspace/maxtext}"                # MaxText location inside the container
MAXTEXT_PATCH_BRANCH="${MAXTEXT_PATCH_BRANCH-}"                           # Global patch branch (explicit empty = skip checkout, use image default); per-model .env.sh can override
# ── end Docker image ──────────────────────────────────────────────────────────

# ── Host paths to mount ───────────────────────────────────────────────────────
DATASET_DIR="${DATASET_DIR:-/mnt/vast/datasets}"                          # Host path to datasets (mounted read-only as /datasets inside the container)
# Extra coredump directories to probe (beyond JOB_WORKSPACE).
# First entry with >500GB free space wins.
# CLI override: comma-separated string, e.g. COREDUMP_EXTRA_DIRS="/path1,/path2"
if [[ -n "${COREDUMP_EXTRA_DIRS:-}" ]]; then
    IFS=',' read -ra COREDUMP_EXTRA_DIRS <<< "$COREDUMP_EXTRA_DIRS"
else
    COREDUMP_EXTRA_DIRS=(
        "/perf_apps/maxtext_coredump"             # DLC cluster
    )
fi
# ── end Host paths to mount ───────────────────────────────────────────────────
