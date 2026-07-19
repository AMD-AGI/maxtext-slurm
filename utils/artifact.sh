#!/bin/bash
# artifact.sh — Build a point-in-time artifact of the repository for batch jobs.
#
# Usage (sourced by submit.sh):
#   source utils/artifact.sh
#   build_artifact <repo_root> <artifact_dir> <maxtext_slurm_dir>
#
# The artifact mirrors the repo tree so that jobs run from an immutable copy
# of the code, allowing the user to edit and submit more jobs without
# affecting pending or running ones.

build_artifact() {
    local src_dir="$1"      # repository root
    local dst_dir="$2"      # target artifact directory
    local script_dir="$3"   # maxtext_slurm dir (for capturing git summary)

    echo "[ARTIFACT] Building artifact..."
    echo "[ARTIFACT]   Source : $src_dir"
    echo "[ARTIFACT]   Target : $dst_dir"

    local start_ts
    start_ts=$(date +%s)

    mkdir -p "$dst_dir"

    # Respect .gitignore at every level; also exclude large local build trees
    # that are not needed by submitted jobs.
    local -a rsync_args=(
        -a
        --exclude='.git/'
        --filter=':- .gitignore'
    )
    if [[ "${ARTIFACT_INCLUDE_BUILD_HIPBLASLT_SCANNER:-0}" != "1" ]]; then
        rsync_args+=(--exclude='build_hipblaslt_scanner/')
    fi
    rsync "${rsync_args[@]}" "$src_dir/" "$dst_dir/"

    # Capture git summary into the artifact (it has no .git/ directory).
    if [[ -d "$src_dir/.git" ]]; then
        if [[ "$src_dir" == "$script_dir" ]]; then
            local summary_dest="$dst_dir/git_summary.txt"
        else
            local summary_dest="$dst_dir/$(basename "$script_dir")/git_summary.txt"
        fi
        (cd "$script_dir" && bash "$script_dir/utils/git_summary.sh") \
            > "$summary_dest" 2>&1 || true
    fi

    local end_ts
    end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))
    local size
    size=$(du -sh "$dst_dir" 2>/dev/null | cut -f1)

    echo "[ARTIFACT] Done in ${elapsed}s (${size})"
}
