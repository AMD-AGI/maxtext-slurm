# Job submission

## Output directory (`JOB_WORKSPACE`)

Multi-node [Slurm](https://slurm.schedmd.com/) jobs require the output directory to reside on a shared filesystem (NFS, Lustre, GPFS, etc.) — all nodes read job artifacts from it at startup and write logs, metrics, and checkpoints to it throughout training. Two options:

- **Clone the repo onto a shared filesystem** — the default `outputs/` directory is already accessible to all nodes; nothing to configure.
- **Clone the repo elsewhere** — set `JOB_WORKSPACE` to a path on the shared filesystem.

By default, all outputs (logs, training artifacts, build artifacts) go to `outputs/` next to the scripts. To use a different location, set `JOB_WORKSPACE`:

```bash
# Single submission
JOB_WORKSPACE=/shared/maxtext_jobs submit.sh 70b -N 1

# Or export once for the session
export JOB_WORKSPACE=/shared/maxtext_jobs
submit.sh 70b -N 1
submit.sh 405b -N 8
```

A warning is printed if the path doesn't appear to be on a shared filesystem. The utilities (`tail_job_log.sh`, `tag_tgs.sh`) also respect `JOB_WORKSPACE` when no explicit path argument is given.

## Slurm jobs (`submit.sh`)

```
# submit.sh <model_name>[[:model_name_alias]:exp_tag] [sbatch_args...] -- [passthrough_args...]
#
# Arguments:
#   model_name                            Model config name (required)
#   model_name:exp_tag                    Model config + experiment tag
#   model_name:model_name_alias:          Model config + checkpoint dir alias (trailing colon)
#   model_name:model_name_alias:exp_tag   Model config + checkpoint dir alias + experiment tag
#   sbatch_args                           Arguments passed to sbatch (before --)
#   passthrough_args                      Arguments after -- can include:
#                                         - MaxText args: key=value (passed to MaxText training)
#                                         - Env vars: _env_KEY=VALUE (exported before launch)
#
# model_name        selects the run config (.gpu.yml file) and the Slurm job name.
# model_name_alias  overrides model_name in the checkpointing output directory so that
#                   parallel experiments using the same config get isolated checkpoint/
#                   tensorboard directories. Only takes effect with enable_checkpointing=true;
#                   without it, dirs already include the unique job ID.
#                   Alias requires two colons; use a trailing colon for alias without tag.
#
# The -- separator splits sbatch arguments from passthrough arguments.
#
# Examples (using $REPO as placeholder for wherever the repo is cloned):
#   submit.sh 70b -N 1                                      # llama2-70b on 1 node
#   submit.sh 70b:baseline -N 1                             # With experiment tag
#   submit.sh 70b -N 2 --time=24:00:00                      # Multiple sbatch args
#   submit.sh 70b -N 1 -- remat_policy=full                 # Both sbatch and MaxText args
#   submit.sh 70b:exp1 -N 1 -- remat_policy=full            # With tag and MaxText args
#   submit.sh 70b -N 1 -- _env_NCCL_DEBUG=INFO              # `export NCCL_DEBUG=INFO` before launch
#   submit.sh 70b -N 1 -- _env_NCCL_DEBUG=INFO steps=1      # Both env and MaxText args
#   submit.sh 70b -N 1 -- steps=1 _env_ENABLE_XLA_DUMP=1    # HLO IR dump
#   submit.sh 70b:run-a: -N 1                               # Checkpoint dir alias (see below)
#   submit.sh 70b:run-a:exp1 -N 1                           # Alias + experiment tag
#   JOB_WORKSPACE=/shared/maxtext_jobs submit.sh 70b -N 1   # Custom output dir (shared FS)
#   DOCKER_IMAGE=my/image:tag submit.sh 70b -N 1            # Override Docker image
```

`model_name` is resolved by matching against `.gpu.yml` files in `configs/` — only models with an existing config are supported. See [Model Configs: Adding a new config](model-configs.md#adding-a-new-config) to add your own.

## Local runs (`run_local.sh`)

Run a single-node training job locally without Slurm scheduling:

```bash
run_local.sh 70b -- steps=10
run_local.sh 70b:my-experiment -- per_device_batch_size=2
RAY=1 run_local.sh 70b -- steps=10                        # with observability
```

Or drop into an interactive shell inside the container (no GPU gate):

```bash
run_local.sh
```

The interface mirrors `submit.sh` (same model spec format, same `--` separator, same `_env_` overrides) but without sbatch args since there is no Slurm. The script:

1. **Checks GPU availability** (training mode only) — detects GPUs (AMD via `/dev/kfd`, NVIDIA via `nvidia-smi`) and warns if they appear occupied. Does not abort.
2. **Runs in a container** — launches the same container image as `submit.sh` via `_container.sh` ([Docker](https://www.docker.com/) or [Podman](https://podman.io/), auto-detected).
3. **Runs from the live repo** — no artifact copy, so code changes take effect immediately.

**On a Slurm cluster**, reserve a node first so Slurm won't schedule other jobs onto it:

```bash
srun -N 1 --exclusive --pty /bin/bash   # get an interactive shell on a compute node
run_local.sh 70b -- steps=10
```

`JOB_WORKSPACE`, `_env_`, and `container_env.sh` overrides (e.g. `DOCKER_IMAGE=…`) work the same as with `submit.sh`.

## Inside-container runs (`in_container_run.sh`)

Run training from inside the container — when you're already in an interactive shell, a `docker exec` session, a [Kubernetes](https://kubernetes.io/) pod, or any pre-built environment with the training image:

```bash
in_container_run.sh 70b -- steps=10
in_container_run.sh 70b:my-experiment -- per_device_batch_size=2
RAY=1 in_container_run.sh 70b -- steps=10
```

The interface is identical to `run_local.sh` (same model spec format, same `--` separator, same `_env_` overrides), but it skips the container launch — it runs `_train.sh` directly.

For Kubernetes job submission and direct-container usage, see [Kubernetes job submission](k8s-job-submission.md).

A common workflow is to enter the container interactively, then iterate:

```bash
# On the host:
run_local.sh                                           # drop into an interactive container shell

# Inside the container:
in_container_run.sh 70b -- steps=5                     # quick test
in_container_run.sh 70b -- steps=5 remat_policy=full   # try a different config
```

Environment variables (`JAX_COORDINATOR_IP`, `NNODES`, etc.) default to single-node local values but can be overridden for multi-node setups. For direct in-container runs, `JOB_WORKSPACE` defaults to `outputs/` next to the scripts unless explicitly set. `container_env.sh` overrides (e.g. `MAXTEXT_REPO_DIR=…`, `MAXTEXT_PATCH_BRANCH=…`) also work — `DOCKER_IMAGE` and other container-launch settings are ignored since the container is already running.

## Checkpointing

Enable checkpointing by passing `enable_checkpointing=true` as a CLI passthrough arg:

```bash
submit.sh 70b -N 1 -- enable_checkpointing=true
submit.sh 70b -N 1 -- enable_checkpointing=true checkpoint_period=200 enable_single_replica_ckpt_restoring=true
```

| Parameter | Default (base.yml) | Description |
|-----------|---------------------|-------------|
| `enable_checkpointing` | `true` | Save model checkpoints during training. **Must be passed via CLI** (see below) |
| `async_checkpointing` | `true` | Write checkpoints in the background so training continues during save. **Sync mode blocks training until the write completes with no timeout** — this can stall indefinitely on large models. MaxText does not expose a configurable timeout for synchronous checkpointing |
| `checkpoint_period` | `10000` | Save a checkpoint every N steps |
| `max_num_checkpoints_to_keep` | `None` | Maximum number of checkpoints retained on disk. `None` keeps all. Set a limit to control disk usage (older checkpoints are deleted when the limit is reached) |
| `enable_single_replica_ckpt_restoring` | `false` | One replica reads the checkpoint and broadcasts to the rest, reducing storage read contention and may accelerate restore. Should typically be set to `false` when `dcn_data_parallelism` is a power of 2, because the checkpoint is globally sliced across DP replicas |

**Why CLI, not YAML.** Checkpointing must be enabled via `--` passthrough, not just in the model config YAML. The launch system uses the presence of `enable_checkpointing=true` in the job name to switch the output directory from job-based (`outputs/12345-JAX-llama2-70b/`) to model-based (`outputs/llama2-70b/`). This is what makes checkpoints persist across job restarts — the same model always writes to the same directory. Setting it only in the YAML would checkpoint to a job-specific directory that is lost on restart.

For running multiple experiments with checkpointing on the same model, see [Model Name Alias](#model-name-alias) below.

## Model name alias

With `enable_checkpointing=true`, the output directory is based on the model name (e.g. `outputs/llama2-70b/`) so checkpoints persist across job restarts. This means multiple experiments using the same model config will write to the same directory, causing conflicts.

A **model name alias** overrides the model name in the checkpoint directory while still using the original config file. Without checkpointing, output directories already include the unique job ID (e.g. `outputs/12345-JAX-llama2-70b/`), so the alias has no effect.

The number of colons determines parsing — alias always requires **two colons**:

| Input | Config (.gpu.yml) | Checkpoint dir | Slurm job name |
|-------|-------------------|----------------|----------------|
| `70b` | llama2-70b | `outputs/llama2-70b/` | JAX-llama2-70b |
| `70b:exp1` | llama2-70b | `outputs/llama2-70b/` | JAX-llama2-70b-exp1 |
| `70b:my-run:` | llama2-70b | `outputs/my-run/` | JAX-llama2-70b |
| `70b:my-run:exp1` | llama2-70b | `outputs/my-run/` | JAX-llama2-70b-exp1 |

To set an alias without an experiment tag, use a **trailing colon** (e.g. `70b:my-run:`). The 1-colon form `model:value` is always interpreted as `model:exp_tag` for backward compatibility.

```bash
# Problem: both write checkpoints to outputs/llama2-70b/ -> CONFLICT
submit.sh 70b:run-a -N 1 -- enable_checkpointing=true
submit.sh 70b:run-b -N 1 -- enable_checkpointing=true

# Fix: alias gives each experiment its own checkpoint dir
submit.sh 70b:70b-run-a: -N 1 -- enable_checkpointing=true  # -> outputs/70b-run-a/
submit.sh 70b:70b-run-b: -N 1 -- enable_checkpointing=true  # -> outputs/70b-run-b/
```

## Environment configuration

Training behavior is controlled by two config files and an optional per-run override mechanism:

| File | What it controls | Edited by users |
|------|-----------------|-----------------|
| `container_env.sh` | Docker image, registry, host mount paths, optional hotfix/debug branch. All variables are CLI-overridable (e.g. `DOCKER_IMAGE=… ./submit.sh …`) | Yes |
| `container_env.local.sh` | Registry credentials (gitignored) | Yes (private images only) |
| `train_env.sh` | Runtime env vars (XLA, NCCL, ROCm, ...) | Yes |
| `configs/<model>.env.sh` | Per-model env overrides (sourced after `train_env.sh`) | Yes (optional) |
| `_env_` prefix | Per-run overrides | Via CLI |

### `container_env.sh` (Docker image and paths)

Sourced by `_container.sh` before launching the container (and by `in_container_run.sh` for `MAXTEXT_REPO_DIR` and `MAXTEXT_PATCH_BRANCH`). All variables use `${VAR:-default}` so they can be **overridden from the command line** without editing the file:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DOCKER_REGISTRY` | `docker.io` | Container registry for pulls |
| `DOCKER_IMAGE` | `rocm/jax-training:maxtext-v26.1` | Docker image to run |
| `USE_DOCKER_IMAGE_AINIC_DRIVER` | `true` | Use the container's built-in AINIC driver; set to `false` to bind-mount host IB libs instead (needed when container libionic1 mismatches host firmware) |
| `MAXTEXT_REPO_DIR` | `/workspace/maxtext` | MaxText location inside the container |
| `MAXTEXT_PATCH_BRANCH` | _(empty)_ | Global patch branch (empty = image default); per-model `.env.sh` can override |
| `DATASET_DIR` | `/mnt/vast/datasets` | Host path to datasets (mounted read-only as `/datasets` inside the container) |
| `COREDUMP_EXTRA_DIRS` | `"/perf_apps/maxtext_coredump"` | Extra coredump directories to probe (comma-separated for CLI; see [Debugging: Core Dumps](debugging.md#where-core-dumps-go)) |

```bash
# One-off image override
DOCKER_IMAGE=my/custom:tag ./run_local.sh 70b -- steps=10
DOCKER_IMAGE=my/custom:tag ./submit.sh 70b -N 1

# Override multiple settings at once
DOCKER_IMAGE=my/custom:tag USE_DOCKER_IMAGE_AINIC_DRIVER=false ./run_local.sh 70b -- steps=10

# Override the MaxText branch (works with all entry points)
MAXTEXT_PATCH_BRANCH=my-hotfix ./submit.sh 70b -N 1

# Override host paths
DATASET_DIR=/my/datasets ./run_local.sh 70b -- steps=10

# Override coredump dirs (comma-separated for multiple paths)
COREDUMP_EXTRA_DIRS="/fast-ssd/coredumps,/shared/coredumps" ./submit.sh 70b -N 8

# Override registry (e.g. private mirror)
DOCKER_REGISTRY=my-registry.example.com ./submit.sh 70b -N 1

# Export once for the session
export DOCKER_IMAGE=my/custom:tag
./submit.sh 70b -N 1
./submit.sh 405b -N 8
```

For `submit.sh`, overrides propagate through Slurm automatically: `sbatch` captures the submit-time environment, and `srun --export=ALL` propagates it to all worker nodes.

Both registry images and local `.tar` tarballs are supported. `USE_DOCKER_IMAGE_AINIC_DRIVER` controls whether to use the container's built-in AINIC driver (`true`) or bind-mount host IB libraries into the container (`false`). Set to `false` on partitions where the container's `libionic1` version doesn't match the host firmware. `MAXTEXT_PATCH_BRANCH`, when non-empty, checks out that branch inside the container's MaxText repo at startup. Per-model `.env.sh` files can also set `MAXTEXT_PATCH_BRANCH` to override at training time (see `_train.sh`).

**Private images.** `_container.sh` tries an anonymous pull first. If the image is private, log in to the registry once on the cluster (one-time setup). In most HPC environments, home directories are on a shared filesystem, so all worker nodes inherit the credentials automatically:

```bash
docker login docker.io    # or: podman login docker.io
```

Alternatively, create a local credentials file: `cp container_env.local.template container_env.local.sh`. This works directly with `run_local.sh`. For `submit.sh`, the artifact system excludes gitignored files by default, so credentials won't reach worker nodes. On a private cluster where this is acceptable, remove the `container_env.local.sh` line from `.gitignore`:

```bash
# in maxtext_slurm/.gitignore, remove or comment out:
container_env.local.sh
```

This lets the artifact include the credentials file, making it available to all Slurm worker nodes.

**Host paths to mount.** `DATASET_DIR` and `COREDUMP_EXTRA_DIRS` (see table above) define host paths that are bind-mounted into the container. Override them from the command line the same way as other variables.

### `train_env.sh` (runtime variables)

Sourced by `_train.sh` before every training launch. Any `export` in this file applies to all runs. See [Performance: Tuning](performance.md#tuning) for guidance on what to adjust.

| Category | Examples |
|----------|----------|
| XLA flags | `XLA_FLAGS`, `XLA_PYTHON_CLIENT_MEM_FRACTION` |
| XLA dump | `ENABLE_XLA_DUMP` (toggle via `_env_ENABLE_XLA_DUMP=1`; see [Performance: HLO IR dump](performance.md#hlo-ir-dump)) |
| NCCL / RCCL | `NCCL_DEBUG`, `NCCL_CROSS_NIC`, `NCCL_IB_*`, `RCCL_*` |
| Memory | `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` |
| InfiniBand | `NCCL_IB_QPS_PER_CONNECTION`, `NCCL_IB_TC` |
| GPU compute | `GPU_MAX_HW_QUEUES`, `CUDA_DEVICE_MAX_CONNECTIONS` |
| AMD / ROCm | `HIP_FORCE_DEV_KERNARG`, `HSA_*` |
| Transformer Engine | `NVTE_FUSED_ATTN`, `NVTE_FUSED_ATTN_CK`, `NVTE_USE_ROCM` |
| Composable Kernel | `CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT`, `NVTE_CK_*` |

### Per-model environment overrides (`configs/<model>.env.sh`)

If a file `configs/<model-name>.env.sh` exists, `_train.sh` sources it after `train_env.sh`. This lets individual models override global defaults without affecting other models or requiring CLI overrides on every submit.

The full override order is:

```
train_env.sh (global)  →  configs/<model>.env.sh (per-model)  →  _env_ CLI (per-run)
```

Each layer overrides the previous. For example, `train_env.sh` sets `XLA_PYTHON_CLIENT_MEM_FRACTION=.85`, a per-model env file can raise it to `.93` for a memory-hungry model, and a CLI `_env_` override can adjust it further for a one-off experiment. Per-model env files can also set `MAXTEXT_PATCH_BRANCH` to check out a branch that adds support for models not yet in the image's default MaxText.

Per-model env files are optional — models without one use the global `train_env.sh` defaults. See [Model Configs: Per-model environment overrides](model-configs.md#per-model-environment-overrides) for details.

### Per-run overrides (`_env_`)

Any argument after `--` that starts with `_env_` is treated as an environment variable override. The prefix is stripped and the remainder is exported **after** `train_env.sh` and per-model env are sourced, so overrides always take precedence:

```bash
submit.sh            70b -N 2 -- _env_NCCL_DEBUG=INFO           # exports NCCL_DEBUG=INFO
run_local.sh         70b      -- _env_NCCL_DEBUG=INFO steps=5   # same mechanism, local run
in_container_run.sh  70b      -- _env_NCCL_DEBUG=INFO steps=5   # same mechanism, inside container
# Multiple overrides and MaxText args in one command:
submit.sh            70b -N 2 -- _env_NCCL_DEBUG=INFO _env_XLA_PYTHON_CLIENT_MEM_FRACTION=.93 remat_policy=full
submit.sh            70b -N 1 -- steps=1 _env_ENABLE_XLA_DUMP=1 # HLO IR dump (see Performance)
```

**Example: [`XLA_PYTHON_CLIENT_MEM_FRACTION`](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)** — controls what fraction of GPU memory [JAX](https://jax.dev/) pre-allocates. The default in `train_env.sh` is `.85`, which works for most models. Large models need a higher value:

```bash
# Too low (0.80) — hangs or OOM:
submit.sh 405b -N 8 -- per_device_batch_size=5 _env_XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

# Just right (0.93) — works:
submit.sh 405b -N 8 -- per_device_batch_size=5 _env_XLA_PYTHON_CLIENT_MEM_FRACTION=0.93

# Too high (0.98) — RCCL/NCCL allocation failure:
submit.sh 405b -N 8 -- per_device_batch_size=5 _env_XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
```

**Too low** — JAX cannot allocate enough memory for model parameters/activations. Training hangs or crashes with:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to
allocate 221.71GiB.: while running replica 0 and partition 40 of a replicated
computation (other replicas may have failed as well).
```

Fix: increase the value (e.g. `.80` -> `.93`).

**Too high** — JAX grabs so much memory that RCCL/NCCL has none left for communication buffers:

```
rocdevice.cpp:3583: Aborting with error : HSA_STATUS_ERROR_OUT_OF_RESOURCES:
The runtime failed to allocate the necessary resources. Available Free mem : 24 MB
```

or:

```
RCCL operation ncclGroupEnd() failed: unhandled cuda error
  ... Failed to CUDA calloc 6291456 bytes
```

Fix: decrease the value (e.g. `.98` -> `.93`). The sweet spot is typically `.85`–`.93` depending on model size. For persistent use, edit `train_env.sh` directly instead of passing `_env_` every time.

## Stage timeouts

Each job stage (preflight, docker pull, ECC check, training) has a configurable timeout to prevent unresponsive nodes from blocking the job forever. Defaults: preflight 900s, pull 900s, ecc 300s, train none.

Override per-job via `STAGE_TIMEOUTS` — specify only the stages you want to change:

```bash
STAGE_TIMEOUTS="preflight:600,train:86400" submit.sh 70b -N 24
```

Use `none` or `-1` to disable a timeout. The resolved values and override syntax are printed at the top of every job log.

## Ray mode (observability)

Enable [Ray](https://www.ray.io/) for full-stack job observability — real-time dashboards and crash-proof data persistence:

```bash
RAY=1 submit.sh 70b -N 1 -- per_device_batch_size=2
```

See [Observability](observability.md) for the full story: dashboards, SSH tunnel setup, post-run diagnostics, and debug commands.

## 1-GPU-per-process mode

By default each Slurm task runs a single Python process that claims all local GPUs (1-node-per-process). Some libraries need one Python process per GPU instead (e.g. IPC/xGMI-based expert-parallel dispatch kernels, or libraries that require `jax.local_device_count() == 1`). Enable the per-GPU launcher with `_env_ONE_GPU_PER_PROCESS=true`:

```bash
submit.sh 70b -N 8 -- _env_ONE_GPU_PER_PROCESS=true
RAY=1 submit.sh 70b -N 8 -- _env_ONE_GPU_PER_PROCESS=true         # composes with Ray
```

When enabled, the container forks `LOCAL_WORLD_SIZE` Python subprocesses per node (one per local GPU). Each subprocess gets its own `LOCAL_RANK`, `GLOBAL_RANK`, and `NPROCS` env vars and calls `jax.distributed.initialize(..., local_device_ids=[LOCAL_RANK])` so exactly one GPU is visible to JAX. Slurm still only sees one task per node, so there is no change to the sbatch allocation.

**Profiler scoping.** With `profiler=xplane` + `upload_all_profiler_results=true` (the shipped default in most `configs/*.gpu.yml`), each local rank would normally race on the same `<host>.xplane.pb` filename. `utils/mfu_tracker.py` serializes the `stop_trace` write per host and tags each process's output as `<host>.proc<LOCAL_RANK>.*`, preserving per-GPU straggler visibility. No user action required.

---

## How artifacts work

Each `submit.sh` submission snapshots the repo into `$JOB_WORKSPACE/.artifacts/`, enabling rapid multi-submit and traceability. The output directory structure is unchanged:

```
$JOB_WORKSPACE/
  12345-JAX-llama2-70b.log                   # Job log (same as before)
  12345-JAX-llama2-70b/                      # Per-job output dir (same as before)
    artifact -> ../.artifacts/artifact_...   #   symlink to the artifact
    log -> ../12345-JAX-llama2-70b.log       #   symlink to the job log
    <training output files>
  .artifacts/                                # Artifacts (new)
    artifact_20260206_143052_a3f2/           #   immutable copy of the repo at submit time
      submit_cmd.txt                         #   exact submit command used
      git_summary.txt                        #   git status at submit time
```

- **Interactive mode** (`run_local.sh` with no args) does not build artifacts — it uses the live code.
- Artifacts exclude `.git/`, `outputs/`, `*.trace.json*`, and `build_hipblaslt_scanner/` by default, so they are small and fast to build.
- `rm -rf $JOB_WORKSPACE/<JOB_ID>*` cleans up the per-job dir and job log as before, without touching artifacts.
- To clean up orphaned artifacts (no longer referenced by any job):

```bash
utils/cleanup_artifacts.sh        # dry-run: list all artifacts and their status
utils/cleanup_artifacts.sh -c     # remove orphans (confirms one by one)
utils/cleanup_artifacts.sh -c -y  # remove orphans (skip confirmation)
```

For details on how the artifact system works internally, see [Architecture: Artifact System](architecture.md#artifact-system).
