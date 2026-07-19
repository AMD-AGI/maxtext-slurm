# 8-Node Deterministic Training Experiments — Cross-Cluster Handoff

Audience: an AI agent (or engineer) running these experiments on a **different cluster**.
This document is self-contained. Read it fully before submitting anything.

---

## 0. TL;DR

- Run **18 jobs** on **8 nodes each** (8x MI355X / gfx950 per node): 6 smoke + 12 feasibility.
- Two model families: `llama3.1-405b` (Dense) and `deepseek3-671b` (MoE).
- Each experiment is a **deterministic vs non-deterministic** pair (the core comparison).
- **Docker image is NOT on public Docker Hub.** You must transfer the image tar (~90 GB) to the new cluster, or push it to a registry the new cluster can reach. See Section 3.
- The repo working directory **must be on a shared filesystem visible to all 8 nodes** (this was the #1 failure cause on the origin cluster).

---

## 1. Background and Goal

Deterministic training means bit-identical loss streams across repeated runs with the same seed/config/hardware. It is not a single switch; it is a stack-wide contract across kernels, collectives, compiler, and PRNG.

These 18 jobs are the **P0 gate** before the real long-horizon (1000-step) A/B replay campaign. They exist to catch the two most common failures cheaply *before* burning hours on full runs:

1. **Startup failures** on 8 nodes (multi-node bring-up, image, collectives).
2. **OOM** — deterministic mode uses extra attention-backward workspace (split `dQ` buffers), so its feasible batch ceiling is lower than non-deterministic. We must find the feasible batch per mode first.

The three setup classes map to three different non-determinism risk profiles:

| Class | Model | Path characteristics |
|---|---|---|
| A: Dense | `llama3.1-405b` | attention backward dominates; no MoE routing |
| B: MoE dense_matmul | `deepseek3-671b` (`sparse_matmul=False`) | attention + top-k router + token dispatch/combine |
| C: MoE sparse_matmul | `deepseek3-671b` (`sparse_matmul=True`, `shardy=True`) | class B + ragged/grouped sparse expert execution |

Note: the sparse path (`sparse_matmul=True`) is gated on `shardy=True` in this stack, so C is always the `sparse_matmul + shardy` bundle.

---

## 2. What "deterministic mode" actually sets

Deterministic mode is triggered by `DETERMINISTIC_MODE=1` (an env var consumed by `train_env.sh`). When set, it enables the following (see `train_env.sh` lines ~86-152):

- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` — TE forces deterministic fused-attention (CK) kernels (fwd + bwd). This is the primary control.
- `TF_DETERMINISTIC_OPS=1` — disables rocBLAS atomic reductions.
- `HIPBLASLT_DETERMINISTIC=1` — filters non-deterministic hipBLASLt solutions (requires patched hipBLASLt, present in the image).
- `NCCL_ALGO=Ring` + `RCCL_MSCCLPP_ENABLE=0` — pins collective reduction order.
- `JAX_DEFAULT_PRNG_IMPL=threefry2x32` — applied post-init by `utils/deterministic.py` (MaxText hardcodes `unsafe_rbg`, which must be overridden).
- `xla_gpu_autotune_level=0` — reproducible kernel selection.

Non-deterministic mode = `DETERMINISTIC_MODE=0` **plus** explicitly `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1` (the fast/default path).

**Why the buffer gives determinism:** non-det attention-backward accumulates `dQ` via atomic adds in nondeterministic order; float addition is not associative, so results vary run-to-run. Deterministic mode allocates per-split `dQ` accumulation buffers and does an **ordered** reduction — fixed summation order → bit-identical results. Cost: extra HBM (so lower feasible batch) and ~6% throughput.

---

## 3. Docker image — YES, you must transfer it manually

### The situation
- Image name used on the origin cluster: `rocm/jax-training:maxtext-v26.2-det-te508-aot`
- This tag is **not available on public Docker Hub** (verified: `docker manifest inspect` fails; Hub API returns 404 for the tag).
- Local image tar on the origin cluster: `/mnt/vast/qiangh/clean/maxtext-v26.2-det-te508-aot.tar` (~90 GB).

This image is required because it contains **ROCm/TransformerEngine PR #508** (TE ≥ `2.12.0.dev0+8943023d`), which wires the CK fused-attention `_deterministic` kernel variants through `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`. Older images silently do NOT propagate the deterministic flag to CK.

### How the launcher resolves the image (`_container.sh` lines ~217-278)
- If `DOCKER_IMAGE` is a path ending in `.tar` and the file exists → it runs `docker load -i <tar>` on **each node** during the pull-only stage.
- Otherwise → `docker image inspect`; if missing, `docker pull $DOCKER_REGISTRY/$DOCKER_IMAGE` (default registry `docker.io`).

### Your options on the new cluster (pick one)

**Option A — transfer the tar to shared storage (recommended, simplest):**
1. Copy `maxtext-v26.2-det-te508-aot.tar` to a path on the **new cluster's shared filesystem** visible to all nodes, e.g. `/shared/images/maxtext-v26.2-det-te508-aot.tar`.
2. Pass `DOCKER_IMAGE=/shared/images/maxtext-v26.2-det-te508-aot.tar` on every submit.
3. The launcher auto-detects the `.tar` and `docker load`s it on each node (first run is slow: ~90 GB load per node).

**Option B — pre-load on each node, then use the image name:**
1. `scp` the tar to each node (or shared FS), then `docker load -i <tar>` once per node.
2. Then pass `DOCKER_IMAGE=rocm/jax-training:maxtext-v26.2-det-te508-aot` (inspect finds it locally, skips pull).

**Option C — push to a private registry the new cluster can access:**
1. `docker load -i <tar>` somewhere, `docker tag`, `docker push <your-registry>/...`.
2. Set `DOCKER_REGISTRY=<your-registry>` and `DOCKER_IMAGE=<repo:tag>`; put creds in `container_env.local.sh` (see `container_env.local.template`).

**Answer to "do I need to manually copy-paste the docker tar file?": Yes.** The tag is not publicly pullable, so you must transfer the ~90 GB tar (Option A/B) or push it to an accessible registry (Option C). Option A is the least error-prone.

---

## 4. Prerequisites on the new cluster (do these first)

1. **Hardware**: 8 nodes, each 8x AMD Instinct MI355X (gfx950). Confirm with `rocm-smi --showproductname`.
2. **Slurm partition** with those 8 nodes (adjust `-p <partition>` in the commands below).
3. **Shared filesystem**: the repo (`maxtext-slurm/`) must live on storage mounted at the **same path on every node**. The origin-cluster failures were all `couldn't chdir to /mnt/.../maxtext-slurm` → `preflight.sh: No such file or directory`, i.e. the working dir was not visible on some nodes. This is the single most important prerequisite.
4. **Image available** per Section 3.
5. **Disk headroom**: `docker load` of a 90 GB tar needs plenty of free space per node; the coredump mount wants >500 GB free (falls back gracefully if absent).
6. `configs/llama3.1-405b.gpu.yml` and `configs/deepseek3-671b.gpu.yml` exist in the repo (they do in this repo — no action needed unless you copy only a subset).

### Config baselines (already in repo; passthrough args below override per-run)
- `llama3.1-405b.gpu.yml`: `dcn_fsdp=-1`, `ici_fsdp=-1`, `max_target_length=8192`, `attention=cudnn_flash_te`, `quantization=fp8`, `dataset_type=synthetic`.
- `deepseek3-671b.gpu.yml`: `dcn_fsdp=8`, `ici_expert_parallelism=8`, `max_target_length=4096`, `sparse_matmul=False` (default; overridden to True for class C), `dataset_type=synthetic`.

Both use synthetic data — **no dataset download required**.

---

## 5. The 18 experiments

Common setup (run once per shell):

```bash
# Point this at wherever you placed the image tar on the NEW cluster's shared FS.
export DOCKER_IMAGE="/shared/images/maxtext-v26.2-det-te508-aot.tar"
# Stage timeouts (preflight/pull/ecc/train), in seconds.
SMOKE_TO="preflight:900,pull:1800,ecc:300,train:7200"
FEAS_TO="preflight:900,pull:1800,ecc:300,train:14400"
# Adjust to your partition name:
PART="k8s"
```

Submit form (from the repo root):

```
./submit.sh <model>:<tag> -p $PART -N 8 --exclusive --time=<hh:mm:ss> -- <passthrough args>
```

- `DETERMINISTIC_MODE` and `NVTE_ALLOW_NONDETERMINISTIC_ALGO` are passed as **env-var prefixes** (not `_env_`), to keep job names short.
- `enable_dropout=False` for all (removes a PRNG variable from the comparison).

### 5.1 Six smoke jobs (steps=5) — "does it start on 8 nodes?"

```bash
# A) Dense (llama3.1-405b), pdbs=2
DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS="$SMOKE_TO" \
./submit.sh "llama3.1-405b:8nA-smk-nd" -p $PART -N 8 --time=02:30:00 -- \
steps=5 per_device_batch_size=2 enable_dropout=False

DETERMINISTIC_MODE=1 STAGE_TIMEOUTS="$SMOKE_TO" \
./submit.sh "llama3.1-405b:8nA-smk-det" -p $PART -N 8 --time=02:30:00 -- \
steps=5 per_device_batch_size=2 enable_dropout=False

# B) MoE dense_matmul (deepseek3-671b, sparse_matmul=False), pdbs=4
DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS="$SMOKE_TO" \
./submit.sh "deepseek3-671b:8nB-smk-nd" -p $PART -N 8 --time=02:30:00 -- \
steps=5 per_device_batch_size=4 sparse_matmul=False enable_dropout=False

DETERMINISTIC_MODE=1 STAGE_TIMEOUTS="$SMOKE_TO" \
./submit.sh "deepseek3-671b:8nB-smk-det" -p $PART -N 8 --time=02:30:00 -- \
steps=5 per_device_batch_size=4 sparse_matmul=False enable_dropout=False

# C) MoE sparse_matmul + shardy (deepseek3-671b), pdbs=1
DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS="$SMOKE_TO" \
./submit.sh "deepseek3-671b:8nC-smk-nd" -p $PART -N 8 --time=02:30:00 -- \
steps=5 per_device_batch_size=1 sparse_matmul=True shardy=True enable_dropout=False

DETERMINISTIC_MODE=1 STAGE_TIMEOUTS="$SMOKE_TO" \
./submit.sh "deepseek3-671b:8nC-smk-det" -p $PART -N 8 --time=02:30:00 -- \
steps=5 per_device_batch_size=1 sparse_matmul=True shardy=True enable_dropout=False
```

### 5.2 Twelve feasibility jobs (steps=20) — "what batch fits per mode?"

```bash
# A) Dense — pdbs 3 and 5
DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "llama3.1-405b:8nA-fes-nd-p3" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=3 enable_dropout=False

DETERMINISTIC_MODE=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "llama3.1-405b:8nA-fes-det-p3" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=3 enable_dropout=False

DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "llama3.1-405b:8nA-fes-nd-p5" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=5 enable_dropout=False

DETERMINISTIC_MODE=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "llama3.1-405b:8nA-fes-det-p5" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=5 enable_dropout=False

# B) MoE dense_matmul — pdbs 3 and 4
DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "deepseek3-671b:8nB-fes-nd-p3" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=3 sparse_matmul=False enable_dropout=False

DETERMINISTIC_MODE=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "deepseek3-671b:8nB-fes-det-p3" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=3 sparse_matmul=False enable_dropout=False

DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "deepseek3-671b:8nB-fes-nd-p4" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=4 sparse_matmul=False enable_dropout=False

DETERMINISTIC_MODE=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "deepseek3-671b:8nB-fes-det-p4" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=4 sparse_matmul=False enable_dropout=False

# C) MoE sparse_matmul + shardy — pdbs 1 and 2
DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "deepseek3-671b:8nC-fes-nd-p1" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=1 sparse_matmul=True shardy=True enable_dropout=False

DETERMINISTIC_MODE=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "deepseek3-671b:8nC-fes-det-p1" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=1 sparse_matmul=True shardy=True enable_dropout=False

DETERMINISTIC_MODE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "deepseek3-671b:8nC-fes-nd-p2" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=2 sparse_matmul=True shardy=True enable_dropout=False

DETERMINISTIC_MODE=1 STAGE_TIMEOUTS="$FEAS_TO" \
./submit.sh "deepseek3-671b:8nC-fes-det-p2" -p $PART -N 8 --time=04:00:00 -- \
steps=20 per_device_batch_size=2 sparse_matmul=True shardy=True enable_dropout=False
```

### 5.3 Summary table

| # | Tag | Model | Mode | pdbs | steps | extra flags |
|---|---|---|---|---:|---:|---|
| 1 | 8nA-smk-nd | llama3.1-405b | non-det | 2 | 5 | — |
| 2 | 8nA-smk-det | llama3.1-405b | det | 2 | 5 | — |
| 3 | 8nB-smk-nd | deepseek3-671b | non-det | 4 | 5 | sparse_matmul=False |
| 4 | 8nB-smk-det | deepseek3-671b | det | 4 | 5 | sparse_matmul=False |
| 5 | 8nC-smk-nd | deepseek3-671b | non-det | 1 | 5 | sparse_matmul=True shardy=True |
| 6 | 8nC-smk-det | deepseek3-671b | det | 1 | 5 | sparse_matmul=True shardy=True |
| 7 | 8nA-fes-nd-p3 | llama3.1-405b | non-det | 3 | 20 | — |
| 8 | 8nA-fes-det-p3 | llama3.1-405b | det | 3 | 20 | — |
| 9 | 8nA-fes-nd-p5 | llama3.1-405b | non-det | 5 | 20 | — |
| 10 | 8nA-fes-det-p5 | llama3.1-405b | det | 5 | 20 | — |
| 11 | 8nB-fes-nd-p3 | deepseek3-671b | non-det | 3 | 20 | sparse_matmul=False |
| 12 | 8nB-fes-det-p3 | deepseek3-671b | det | 3 | 20 | sparse_matmul=False |
| 13 | 8nB-fes-nd-p4 | deepseek3-671b | non-det | 4 | 20 | sparse_matmul=False |
| 14 | 8nB-fes-det-p4 | deepseek3-671b | det | 4 | 20 | sparse_matmul=False |
| 15 | 8nC-fes-nd-p1 | deepseek3-671b | non-det | 1 | 20 | sparse_matmul=True shardy=True |
| 16 | 8nC-fes-det-p1 | deepseek3-671b | det | 1 | 20 | sparse_matmul=True shardy=True |
| 17 | 8nC-fes-nd-p2 | deepseek3-671b | non-det | 2 | 20 | sparse_matmul=True shardy=True |
| 18 | 8nC-fes-det-p2 | deepseek3-671b | det | 2 | 20 | sparse_matmul=True shardy=True |

---

## 6. Submission strategy (recommended)

Do **not** dump all 18 at once blindly. Gate in waves:

1. Submit the **6 smoke** jobs first. Wait for them to reach steady step output.
2. If all 6 pass (see Section 7), submit the **12 feasibility** jobs.
3. Only after feasibility confirms per-mode batch ceilings should the follow-up 1000-step A/B campaign be planned.

Node pinning: on the origin cluster we pinned `--nodelist=<8 healthy nodes>` after confirming the workspace path was visible on each. On the new cluster, either pin an explicit healthy 8-node list or rely on the partition if all nodes are healthy and share the FS.

---

## 7. How to tell a job succeeded

Each job log ends with a JOB SUMMARY block. Look for:

```
Status: SUCCESS (exit 0)
```

Logs are written to `outputs/<jobid>-JAX-<model>-<tag>-...log`.

- **Smoke pass** = reaches training loop and prints `completed step: N, ... loss: ...` lines for all 5 steps, plus `SUCCESS (exit 0)`.
- **Feasibility pass** = same, for 20 steps, no OOM. Deterministic runs also print a `[determinism] loss_checksum=<hex> (steps=N)` line at the end.
- **OOM** shows `ran out of memory trying to allocate ...` in the log and a non-zero exit — this is expected at some batch sizes and is a valid feasibility result (records the ceiling), not a bug.

Quick scan across all attempts:

```bash
for f in outputs/*8n*-*.log; do
  echo "$(basename "$f"): $(grep -E '^\s*Status:' "$f" | tail -1)"
done
```

Deterministic replay check (any two same-config runs should match):

```bash
grep -h 'loss_checksum' outputs/<jobA>*.log outputs/<jobB>*.log
```

The checksum method: per-step loss scalars are packed as IEEE754 float32 and fed into SHA-256; first 16 hex chars are reported. Same config + same steps + same loss stream ⇒ identical checksum. (Implemented in `utils/deterministic.py`.)

---

## 8. Known failure modes (seen on the origin cluster)

| Symptom in log | Root cause | Fix |
|---|---|---|
| `couldn't chdir to <repo>` then `preflight.sh: No such file or directory`, `Preflight FAILED (exit=143)` | repo working dir not visible on all nodes (not shared FS, or some nodes unhealthy) | put repo on shared FS; pin `--nodelist` to nodes where the path exists |
| `docker pull ... not found` / pull fails | image tag not on the target registry | use the tar (Section 3) |
| `ran out of memory trying to allocate` | batch too large for that mode (esp. deterministic) | expected at ceiling; record it, use next-lower pdbs |
| Job `CANCELLED (scancel / SIGTERM)` | manually cancelled or node preempted | resubmit on healthy nodes |
| slurmd `DOWN+NOT_RESPONDING` on a node | node-level Slurm agent died | exclude that node; needs cluster admin to restart slurmd |

Deterministic-specific note: if you are ever forced onto a **pre-PR-508** image, the deterministic flag will NOT reach CK fused attention. The only workaround there is `_env_NVTE_FUSED_ATTN=0` (JAX-native unfused attention: ~9.7x slower, forces `per_device_batch_size=1`). With the correct image, do **not** set this.

---

## 9. Environment overrides reference

- Per-run env override after `--`: `_env_KEY=VALUE` (e.g. `_env_NVTE_FUSED_ATTN=0`).
- Prefix env vars (before `./submit.sh`) also propagate: used here for `DETERMINISTIC_MODE`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO`, `STAGE_TIMEOUTS`, `DOCKER_IMAGE`.
- Image selection: `DOCKER_IMAGE` (registry name or `.tar` path); `DOCKER_REGISTRY` (default `docker.io`); private creds via `container_env.local.sh`.
- Memory fraction is `XLA_PYTHON_CLIENT_MEM_FRACTION=.93` (in `train_env.sh`); leave as-is for apples-to-apples comparison with origin data.

---

## 10. Files to copy to the new cluster

- The entire `maxtext-slurm/` repo (onto shared storage).
- The image: `maxtext-v26.2-det-te508-aot.tar` (~90 GB) — or push it to an accessible registry.
- No datasets needed (synthetic data).
