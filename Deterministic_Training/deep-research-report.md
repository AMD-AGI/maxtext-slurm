# Deterministic Mode for NVIDIA GPU LLM Training

## Executive summary

Deterministic mode in GPU training means **repeatable numerical results (often bitwise-identical) across runs** *when the same program is executed with the same inputs under the same hardware + software environment*, by eliminating (or rejecting) sources of nondeterminism such as **nondeterministic kernel algorithms (e.g., atomics), runtime autotuning, and unstable execution ordering**. citeturn22view0turn17search2turn2view1

In practice, “deterministic” is not a single switch; it is a **stack-wide contract** spanning:
- GPU math libraries (CUDA / cuBLAS / cuDNN / cuDNN frontend), where determinism can require **library-specific knobs** such as `CUBLAS_WORKSPACE_CONFIG` and requesting deterministic kernels. citeturn2view1turn15view1turn15view2  
- Framework behavior (entity["organization","PyTorch Foundation","ml framework governance"] and TensorFlow), where you must **force deterministic algorithm selection, disable autotuning, and fully seed RNG**. citeturn22view0turn17search2turn12view0  
- Attention kernels and fused transformer stacks (notably NVIDIA Transformer Engine / FlashAttention), where fastest kernels may be explicitly **non-deterministic by design**, and a deterministic path can reduce throughput. citeturn20view0turn20view1turn8academia37  
- Distributed communication (NCCL), where determinism can be impacted by **collective algorithm selection (e.g., NVLS vs Ring/Tree), topology, and overlap schedules**. citeturn19view0turn6search25turn19view2  

What you can realistically guarantee:
- **Single-node, fixed GPU model, fixed software stack**: bitwise-identical training is achievable for many (not all) LLM configurations—especially when you avoid known nondeterministic fused attention paths and enforce deterministic library settings. citeturn19view0turn20view0turn22view0turn2view1  
- **Multi-node**: even when you apply comprehensive deterministic settings, real-world reports show that “deterministic mode” can still fail under certain versions/configurations. Treat multi-node bitwise reproducibility as “possible but fragile,” and verify for your exact topology and software versions. citeturn19view2turn19view0turn17search2  

Performance impact is workload-dependent and can be large for LLM attention backprop:
- A 2026 paper focused on deterministic attention for LLM training reports that **deterministic FlashAttention-3 backward can incur up to a 37.9% throughput reduction** versus nondeterministic backward on NVIDIA H800 GPUs (kernel-level throughput impact). citeturn8academia37  
- Enabling cuBLAS determinism via `CUBLAS_WORKSPACE_CONFIG` is documented by NVIDIA as potentially limiting performance (and/or increasing memory footprint), and a public reproduction shows a **~14% throughput drop** in a real workload when this env var is enabled. citeturn2view1turn9view0  
- TensorFlow explicitly warns that determinism generally reduces performance; additionally, enabling op determinism does **not** make latency/throughput deterministic—only the *numerical outputs and side effects*. citeturn17search2turn11search1  

## Internal 1-node negative-control evidence (MaxText ROCm)

This section records our own 1-node non-deterministic A/B controls in MaxText, using the same analysis rule across setups.

### Protocol (same for all three setups)

- **Goal**: verify that the non-deterministic path actually diverges (negative control), so deterministic replay claims are not explained by logging artifacts.
- **Pairing rule**: A/B runs are pinned to the same node and matched on model/config knobs (only run label differs).
- **Non-deterministic setting**: `DETERMINISTIC_MODE=0` with `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1`.
- **Divergence definition**: first step where aligned `completed step` loss values differ.
- **Alignment rule**: compare only the common prefix (`0..min(steps_A, steps_B)-1`).

### Results (three setups, six jobs)

| Setup | A job | B job | Common steps compared | First divergence step | A loss @ divergence | B loss @ divergence | Terminal state |
|---|---:|---:|---:|---:|---:|---:|---|
| Dense (`llama2-70b`, `pdbs=11`) | `19749` | `19756` | 3 | **2** | 10.831 | 10.660 | A cancelled at step 740; B auto-cancelled on divergence |
| MoE `dense_matmul` (`ds-proxy-e128-h2048`, `pdbs=6`) | `19747` | `19748` | 1000 | **41** | 11.225 | 11.226 | both success |
| MoE `sparse_matmul` (`ds-proxy-e128-h2048`, `pdbs=1`, `shardy=True`) | `19753` | `19754` | 200 | **34** | 9.929 | 9.931 | both success |

### Interpretation

- All three setup families show early A/B loss-stream divergence under non-deterministic mode.
- Divergence appears at different horizons (`step 2`, `34`, `41`), which is expected because sensitivity depends on model path and kernel mix.
- This negative-control evidence strengthens the deterministic claim: when determinism is disabled, matched A/B runs do not remain numerically identical.

### Scope boundaries

- These six jobs establish **single-node** negative-control behavior only.
- They do not, by themselves, prove multi-node deterministic reproducibility; that still requires dedicated multi-node A/B replay evidence under fixed topology and software versions.

## Deterministic mode in the NVIDIA LLM training stack

### Practical definition

Across both PyTorch and TensorFlow, the most operationally useful definition is:

> **Given the same inputs, same seeds, and the same hardware + software environment, the same operators produce the exact same outputs across repeated runs** (often bitwise-identical), by forcing deterministic algorithms and stable execution ordering; if only nondeterministic algorithms exist, the system either errors or accepts nondeterminism explicitly. citeturn22view0turn17search2turn19view0  

Key nuance: determinism is typically scoped to *numerical results*, **not** runtime metrics. Throughput/latency can still vary run-to-run due to OS scheduling, CPU thread timing, thermal/power states, and network jitter. citeturn11search1turn17search2turn6search25  

### Where nondeterminism originates in NVIDIA GPU training

Determinism breaks when any layer of the stack introduces variable ordering or algorithm selection. Major sources include:

- **cuBLAS (GEMM / matmul) concurrency + workspace behavior**: NVIDIA documents that deterministic behavior with multiple concurrent streams sharing a cuBLAS handle may require explicitly configured workspace or `CUBLAS_WORKSPACE_CONFIG`. citeturn2view1  
- **cuDNN / cuDNN frontend kernel selection**: kernels may differ by heuristics/autotuning; deterministic kernels may exist but can be slower or have different support constraints. citeturn22view0turn15view2turn13search2  
- **Fused attention kernels (FlashAttention / Transformer Engine / cuDNN SDPA)**: fastest variants may use atomics or parallel reduction patterns that are not deterministic; deterministic variants exist but can reduce throughput (notably in backward). citeturn20view0turn20view1turn8academia37turn15view1  
- **Framework autotuning and compilation**:
  - PyTorch deterministic mode disables multiple Inductor autotuning/benchmark steps because they “affect numerics.” citeturn22view0  
  - `torch.compile` can introduce correctness/reproducibility issues even when deterministic settings and seeds are applied, per a high-priority issue report. citeturn18view0  
  - TensorFlow op determinism is more thoroughly supported when XLA is not used, historically; compilation-time autotuning can select different kernels unless persisted. citeturn12view0turn17search24  
- **Distributed nondeterminism (NCCL)**: the collective algorithm used for all-reduce can affect reproducibility; Megatron-core documentation calls out `NCCL_ALGO` as important, and real multi-node reports show determinism can still fail even with careful settings. citeturn19view0turn19view2turn6search25  

### Determinism levels that matter for LLM architects

A useful taxonomy for production LLM training:

1. **Seeded training (weak reproducibility)**: fixed RNG seeds, but allow nondeterministic kernels/autotuning. This often yields “similar” loss curves but not bitwise identity. citeturn17search2turn22view0  
2. **Single-node deterministic training (strong reproducibility)**: deterministic algorithms enforced for GPU kernels + deterministic backends (cuBLAS workspace config, disable autotuning) + stable data order and dataloader seeding. citeturn22view0turn2view1turn17search2  
3. **Multi-node deterministic training (fragile reproducibility)**: requires all above plus fixed collective algorithms/topology-sensitive settings; verify per software release and cluster fabric. citeturn19view0turn19view2turn6search25  
4. **Cross-hardware / cross-version reproducibility (rare)**: generally not guaranteed; even cuBLAS determinism is scoped to “same architecture and same number of SMs” and does not promise bitwise identity across differing environments. citeturn2view1turn17search2  

## Enable deterministic mode in PyTorch

This section provides **exact knobs** for PyTorch 1.13–2.x on CUDA 11–12, with NVIDIA library interactions.

### Mandatory environment variables (set before launching Python)

#### cuBLAS reproducibility (CUDA ≥ 10.2 behavior)

Set one of the documented `CUBLAS_WORKSPACE_CONFIG` modes to force deterministic behavior for certain cuBLAS-backed operations under determinism enforcement:

```bash
# Option A: smaller workspace; NVIDIA notes it "may limit overall performance"
export CUBLAS_WORKSPACE_CONFIG=:16:8

# Option B: larger workspace; NVIDIA notes ~24 MiB extra GPU memory footprint
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

NVIDIA explicitly documents the performance/memory tradeoff here. citeturn2view1  

PyTorch’s deterministic algorithms mode also explicitly references these `CUBLAS_WORKSPACE_CONFIG` settings as required to avoid runtime errors for some CUDA operations. citeturn22view0  

#### If you use NVIDIA Transformer Engine or FlashAttention via TE

If you use NVIDIA Transformer Engine attention backends, TE documents that FlashAttention uses a non-deterministic algorithm “for optimal performance,” and that deterministic behavior can be requested by setting:

```bash
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
# optionally:
export NVTE_FLASH_ATTN=0   # to disable flash-attn backend entirely
```

This is explicitly described in Transformer Engine PyTorch documentation and TE env var reference. citeturn20view0turn20view1  

Megatron-core reproducibility guidance also calls out `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` as required when using Transformer Engine in deterministic mode. citeturn19view0  

### Single-GPU deterministic recipe

#### Core deterministic switches

PyTorch provides a top-level switch:

```python
import os, random
import numpy as np
import torch

SEED = 1234

# Best practice: set PYTHONHASHSEED before interpreter start if possible
# (shown here for completeness; the most robust approach is exporting it in the shell)
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Force deterministic algorithms or error if unavailable
torch.use_deterministic_algorithms(True)

# Disable cuDNN autotuning that can change algorithm choice
torch.backends.cudnn.benchmark = False

# Prefer deterministic cuDNN kernels (some ops may still be unavailable)
torch.backends.cudnn.deterministic = True
```

PyTorch defines deterministic algorithms as those that “always produce the same output” on the same software/hardware given the same input; when enabled, nondeterministic-only ops error. citeturn22view0  

PyTorch further notes deterministic operations “tend to have worse performance,” and that enabling deterministic algorithms also turns on an Inductor deterministic mode that avoids several autotuning/benchmarking behaviors. citeturn22view0  

#### Attention backend control (LLM-relevant)

A frequent culprit in transformer stacks is attention kernel selection. PyTorch’s SDPA operator selects among implementations for performance; you typically must **disable fast but nondeterministic kernels** (or request deterministic variants if supported) when strict reproducibility is required. citeturn13search6turn20view0turn22view0  

If you are using Transformer Engine, prefer TE’s deterministic gating (`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`) and/or disabling flash attention. citeturn20view0turn20view1turn19view0  

### Multi-GPU deterministic recipe (single-node DDP using NCCL)

Below is a pragmatic baseline for **single-node, multi-GPU** determinism. It is not a guarantee for all models/ops, but it is the minimum viable configuration to start verifying determinism.

#### Launch command

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0      # if using Transformer Engine
export PYTHONHASHSEED=1234

# (optional) fix NCCL algo if your training stack documents sensitivity
export NCCL_ALGO=Ring

torchrun --standalone --nproc_per_node=8 train.py
```

Megatron-core explicitly states that the chosen NCCL all-reduce algorithm (`NCCL_ALGO`) can matter for reproducibility, and lists the algorithms they tested. citeturn19view0  

NCCL environment variables are documented by NVIDIA (including `NCCL_DEBUG`, which is useful to audit runtime choices). citeturn6search25  

#### Per-rank seeding pattern

For data-parallel training, you want each process to be deterministic but not identical in RNG stream (or you’ll synchronize dropout masks and data augmentation across ranks incorrectly). A common pattern:

```python
import os, random
import numpy as np
import torch
import torch.distributed as dist

BASE_SEED = 1234

def seed_everything_for_rank(base_seed: int):
    # Initialize process group first so rank is defined
    rank = dist.get_rank() if dist.is_initialized() else 0
    seed = base_seed + rank

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# After dist.init_process_group(...)
seed_everything_for_rank(BASE_SEED)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

PyTorch’s determinism guarantee is explicitly scoped to “same input” and “same software and hardware” producing the same output for deterministic algorithms. citeturn22view0  

### Graph vs eager: torch.compile and CUDA graphs

If strict determinism is required, treat PyTorch compilation as **high risk** unless you have validated it with your exact model and attention stack:

- A PyTorch issue report shows nondeterministic outputs across repeated runs **even with** `CUBLAS_WORKSPACE_CONFIG`, `torch.use_deterministic_algorithms(True)`, and explicit seeding—when using `torch.compile`. citeturn18view0  

For deterministic validation workflows, consider forcing eager execution (as the report did via `torch.compiler.set_stance("force_eager")`) until you have an audited, reproducible compilation pipeline. citeturn18view0  

## Enable deterministic mode in TensorFlow

TensorFlow provides a first-class API for deterministic operator execution, plus legacy environment variables used historically (and still common in production images).

### Core deterministic API (TF 2.8+)

TensorFlow’s API doc defines op determinism as: if an op is run multiple times with the same inputs on the same hardware, it has the exact same outputs each time, and notes the performance tradeoff. citeturn17search2  

Minimal deterministic skeleton:

```python
import os, random
import numpy as np
import tensorflow as tf

SEED = 1234

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Must be called early, before building/initializing substantial GPU state
tf.config.experimental.enable_op_determinism()
```

TensorFlow’s official “What’s new in TF 2.9” blog also highlights that determinism comes at the cost of lower performance when op determinism is enabled. citeturn10search8turn17search2  

### Legacy environment variables (still encountered in NVIDIA containers / older guidance)

Historically, deterministic op behavior has been enabled via environment variables such as `TF_DETERMINISTIC_OPS` and `TF_CUDNN_DETERMINISTIC`. The tensorflow-determinism guidance summarizes common approaches and version behaviors and notes that GPU determinism was historically more thoroughly supported when XLA was not used. citeturn12view0  

NVIDIA TensorFlow container release notes (example: 19.06) state that setting `TF_DETERMINISTIC_OPS=1` makes certain GPU behaviors deterministic and that you do not need to also set the cuDNN determinism variable in that case; they also warn this may reduce performance. citeturn10search32  

A conservative “compatibility mode” launch pattern (particularly for older codebases) is:

```bash
export TF_DETERMINISTIC_OPS=1
export TF_CUDNN_DETERMINISTIC=1   # often used historically; some docs say redundant with TF_DETERMINISTIC_OPS
export PYTHONHASHSEED=1234
python train_tf.py
```

The redundancy/behavior caveats and “do not set TF_USE_CUDNN_AUTOTUNE” guidance are discussed in the tensorflow-determinism material. citeturn12view0turn10search32  

### Multi-GPU TensorFlow (NCCL-backed collectives)

TensorFlow’s deterministic op setting applies to op implementations; distributed determinism also depends on **distribution strategy behavior** and collective algorithms underneath (commonly NCCL for GPU). TensorFlow determinism guidance emphasizes that determinism is about op outputs, not runtime properties, and the determinism story is historically more robust without XLA. citeturn17search2turn12view0  

For multi-worker or multi-GPU, you should:
- Enable op determinism early (`enable_op_determinism()`), seed all RNG sources, and ensure dataset sharding/order is stable. citeturn17search2turn12view0  
- Treat XLA compilation and distributed execution overlap as additional determinism risks unless you lock down compilation/autotuning behavior. citeturn17search24turn12view0  

### Graph vs eager: XLA and deterministic compilation

If you use XLA (`jit_compile=True` / XLA devices), determinism can break due to compilation-time autotuning and nondeterministic ops. OpenXLA’s determinism documentation states that compilation is deterministic if persisted autotuning is used; otherwise measurement fluctuations can select different kernels. citeturn17search24  

At the flag level, XLA exposes a `xla_gpu_deterministic_ops` option that “guarantees run-to-run determinism,” with limitations (Scatter and SelectAndScatter lacking deterministic implementations, causing compilation errors). citeturn17search27  

## Remaining nondeterminism factors and mitigation strategies

Even after following the recipes, several classes of nondeterminism commonly survive in LLM training. This section enumerates the main ones and how to mitigate them.

### Nondeterministic GPU kernels due to atomics and parallel reduction order

Many high-performance GPU kernels use **atomic operations** or parallel reductions where the accumulation order is not stable, producing small floating-point differences run-to-run. TensorFlow explicitly attributes many nondeterministic differences to asynchronous threads changing the order in which floating-point numbers are added. citeturn17search2  

Mitigations:
- Prefer deterministic kernel implementations exposed by the framework/library; if unavailable, **fail fast** by enabling PyTorch deterministic algorithms (erroring on nondeterministic ops). citeturn22view0  
- For attention, explicitly choose deterministic backends/flags (Transformer Engine `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`; cuDNN frontend SDPA backward `use_deterministic_algorithm=True` if you are using that API). citeturn20view0turn20view1turn15view1turn15view2  

### Autotuning and “benchmark” modes that select kernels dynamically

Autotuning/benchmarking selects kernels based on measured runtime, which can vary with system load and can change between runs, impacting numerical behavior (and determinism). PyTorch’s deterministic mode explicitly disables several Inductor autotuning and benchmarking behaviors, and `torch.backends.cudnn.benchmark=False` is a standard requirement for stable algorithm choice. citeturn22view0turn13search22  

Mitigations:
- PyTorch: `torch.backends.cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`. citeturn22view0  
- TensorFlow: use op determinism (`enable_op_determinism`) and avoid or tightly control cuDNN autotuning as described in determinism guidance. citeturn12view0turn17search2turn10search32  
- XLA: persist autotuning results or require reuse of complete results, per OpenXLA determinism guidance. citeturn17search24  

### Mixed precision, dynamic loss scaling, and “numerical branching”

Even with “deterministic” kernels, small numerical differences can trigger discrete control-path changes (e.g., overflow detection affecting loss scaling), leading to divergence later in training. TensorFlow and PyTorch define determinism largely as repeated execution producing identical outputs *given the same inputs and environment*; if your training includes numerically sensitive thresholds, determinism can be brittle. citeturn17search2turn22view0  

Mitigations:
- For reproducibility debugging, consider stabilizing scaling decisions (e.g., avoid adaptive heuristics when validating determinism), and verify determinism at short horizons (first N steps) before long training. citeturn19view2turn17search2  
- If using Transformer Engine attention, follow TE’s deterministic gating guidance, and verify that your chosen attention backend supports deterministic backward for your GPU architecture and head dimensions. citeturn20view0turn15view2turn13search2  

### Graph compilation and kernel fusion changes

Compilation and fusion can change numerics and can introduce nondeterminism if the compiler uses runtime benchmarking or unstable scheduling.

- PyTorch: deterministic algorithms mode turns on an Inductor deterministic mode that avoids multiple types of on-device benchmarking. citeturn22view0  
- However, a reported `torch.compile` reproducibility/correctness issue produced different outputs across repeated runs even with determinism flags and seeding. citeturn18view0  

Mitigations:
- Validate determinism in **eager mode first**, then re-validate after enabling compilation. citeturn18view0turn22view0  
- If compilation is required in production, treat determinism as a tested property pinned to specific compiler versions and configurations (containerize and lock versions). citeturn19view0turn17search24  

### Distributed nondeterminism: collective algorithms, topology, overlap schedules

Megatron-core explicitly states that bitwise reproducible training is possible and calls out three known optimizations that break reproducibility while still yielding nearly identical runs—one of which is NCCL all-reduce algorithm choice (`NCCL_ALGO`). citeturn19view0  

Yet a more recent Megatron-Core issue report shows multi-node nondeterminism persisting even with comprehensive settings (including `CUBLAS_WORKSPACE_CONFIG`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, disabled NVLS/Flash Attention, and constrained CUDA device connections). citeturn19view2  

Mitigations:
- **Fix the NCCL algorithm** if your stack documents sensitivity (e.g., set `NCCL_ALGO=Ring` or explicitly exclude NVLS where your training framework does so). citeturn19view0turn19view2  
- Log/audit NCCL configuration and runtime selection using NCCL’s documented env variables (e.g., `NCCL_DEBUG`) to ensure you are truly running the intended algorithm. citeturn6search25turn19view0  
- Treat multi-node determinism as a release-gated feature: verify on your exact cluster (hardware trays/racks, network fabric) and software image. citeturn19view2turn19view0  

### Merlin diagram: RNG and deterministic enforcement flow

```mermaid
flowchart TD
  A[Process start] --> B[Set env vars before import]
  B --> B1[CUBLAS_WORKSPACE_CONFIG]
  B --> B2[TF_DETERMINISTIC_OPS / TF_CUDNN_DETERMINISTIC (legacy)]
  B --> B3[NVTE_ALLOW_NONDETERMINISTIC_ALGO (if TE)]
  B --> C[Import framework]
  C --> D[Seed RNGs]
  D --> D1[Python random / PYTHONHASHSEED]
  D --> D2[NumPy RNG]
  D --> D3[Framework RNG: torch / tf]
  D --> D4[Per-rank seed offset (DDP)]
  C --> E[Force deterministic algorithms]
  E --> E1[PyTorch: torch.use_deterministic_algorithms(True)]
  E --> E2[TensorFlow: enable_op_determinism()]
  E --> F[Training step]
  F --> G[Forward (dropout/attention uses RNG)]
  G --> H[Backward (reductions/atomics may break determinism)]
  H --> I[All-reduce (NCCL algo selection impacts determinism)]
  I --> J[Optimizer update]
```

## Performance and memory impact evidence, plus production checklist and tradeoffs

### Evidence-backed performance impacts (what typically drops, and by how much)

Determinism costs come from (a) **slower deterministic kernels**, (b) **disabling autotuning that would otherwise pick faster kernels**, and (c) **serialization necessary to avoid atomic nondeterminism**.

Below are concrete, cited data points that illustrate real magnitude ranges:

- **Deterministic attention backward can be expensive (LLM training–relevant)**:  
  The 2026 DASH paper reports that in FlashAttention-3, making the backward pass deterministic can incur **up to a 37.9% throughput reduction** relative to nondeterministic backward, attributing the loss chiefly to serialization required for consistent gradient accumulation. citeturn8academia37  

- **cuBLAS deterministic workspace config can trade throughput vs memory**:  
  NVIDIA documents `CUBLAS_WORKSPACE_CONFIG=:16:8` as potentially limiting performance, and `:4096:8` as increasing cuBLAS GPU memory footprint by ~24 MiB. citeturn2view1  

- **Empirical throughput drop observed when `CUBLAS_WORKSPACE_CONFIG` is enabled**:  
  A public issue report showed a workload running at **10.02 it/s without** `CUBLAS_WORKSPACE_CONFIG` vs **8.57 it/s with** `CUBLAS_WORKSPACE_CONFIG=:4096:8` (≈14% slower in that reproduction), demonstrating that the workspace config can materially reduce throughput even outside “training strictness” switches. citeturn9view0turn2view1  

- **Framework-level deterministic ops can be slower by design**:  
  PyTorch states that deterministic operations “tend to have worse performance” and enumerates multiple Inductor behaviors disabled under deterministic mode specifically because they “affect numerics.” citeturn22view0  

- **TensorFlow explicitly warns of reduced performance and clarifies scope**:  
  TensorFlow’s API docs warn that op determinism generally comes at the expense of lower performance. citeturn17search2  
  Separate documentation notes that enabling op determinism does not make latency, memory consumption, or throughput deterministic—only outputs and side effects. citeturn11search1  

### Table: deterministic vs nondeterministic settings for NVIDIA LLM training

The table below emphasizes LLM-relevant knobs and the *expected* (not guaranteed) impact direction. Where an exact numeric delta is known from sources, it is shown; otherwise, the qualitative impact is evidence-backed.

| Layer | Default (fast / nondeterministic-allowed) | Deterministic setting | What becomes deterministic | Expected perf / memory delta |
|---|---|---|---|---|
| cuBLAS GEMM | Default workspace/handle behavior | `CUBLAS_WORKSPACE_CONFIG=:16:8` or `:4096:8` | Deterministic behavior for certain cuBLAS uses under concurrency constraints | `:16:8` may reduce performance; `:4096:8` adds ~24 MiB GPU memory citeturn2view1 |
| PyTorch global | No enforcement | `torch.use_deterministic_algorithms(True)` | Switch to deterministic kernels or error | Deterministic ops “tend to have worse performance”; Inductor disables autotuning/benchmark passes citeturn22view0 |
| cuDNN selection | `torch.backends.cudnn.benchmark=True` (common) | `benchmark=False`, `deterministic=True` | Stable cuDNN algorithm choice (where deterministic kernels exist) | Removes autotune benefits; can be slower depending on kernel choice citeturn22view0turn13search22 |
| Transformer Engine attention | Allow nondeterministic kernels | `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | Only deterministic TE algorithms allowed | TE warns deterministic behavior is “at the cost of performance” citeturn20view0turn20view1 |
| FlashAttention backward | Nondeterministic backward (often atomics) | Deterministic backward mode (where supported) | Bitwise-stable backward gradients | Up to **37.9% throughput reduction** reported for deterministic FA3 backward in LLM training citeturn8academia37 |
| cuDNN frontend SDPA | Default SDPA backward path | `use_deterministic_algorithm=True` in SDPA backward | Bitwise-identical SDPA backward gradients (subject to support matrix) | May be slower; determinism support varies by arch/layout/version citeturn15view1turn15view2turn13search2 |
| NCCL all-reduce | Auto-selected algorithm/topology-dependent | Fix `NCCL_ALGO` (e.g., `Ring`) and avoid known nondeterministic paths as required by framework | Stable collective behavior consistent with fixed selection | Can reduce performance vs auto-selection; Megatron-core calls algorithm choice important for reproducibility citeturn19view0turn6search25turn19view2 |
| TensorFlow ops | Op determinism off | `tf.config.experimental.enable_op_determinism()` | Deterministic op outputs on same HW given same inputs | “May run slower” (no numeric promised); only outputs determinized citeturn17search2turn11search1 |
| PyTorch compilation | `torch.compile` enabled | Prefer eager for determinism validation; compile only after verification | Avoid compiler-induced nondeterminism/correctness issues | A reported case shows nondeterministic outputs under compile despite deterministic settings citeturn18view0 |

### Mermaid diagram: distributed determinism risk points

```mermaid
flowchart LR
  A[Per-rank forward/backward] --> B[Gradient buckets]
  B --> C[NCCL all-reduce]
  C --> D[Reduced grads]
  D --> E[Optimizer step]

  subgraph Risks
    R1[Kernel nondeterminism\n(atomics / reduction order)]
    R2[Autotuning changes\n(cuDNN / Inductor / XLA)]
    R3[NCCL algo selection\n(NVLS vs Ring/Tree)]
    R4[Overlap scheduling\n(comm/compute timing)]
  end

  A -.-> R1
  B -.-> R4
  C -.-> R3
  C -.-> R4
  D -.-> R1
  E -.-> R1
  A -.-> R2
```

### Production checklist for LLM training pipelines

This is a **concise, production-oriented checklist** (for strict reproducibility test runs; you can relax for throughput runs).

**Environment and version pinning**
- Pin container image + CUDA toolkit + cuDNN + NCCL versions; record GPU model and driver. Megatron-core reproducibility is explicitly scoped to the same HW/SW environment and notes which NGC PyTorch containers were verified. citeturn19view0  
- Treat multi-node determinism as a tested property per release; real reports show regressions across versions. citeturn19view2turn19view0  

**Library-level determinism**
- Set `CUBLAS_WORKSPACE_CONFIG` before process start; choose `:16:8` (favours lower workspace) or `:4096:8` (adds ~24 MiB). citeturn2view1  
- If using cuDNN frontend SDPA backward directly, set `use_deterministic_algorithm=True` and ensure your arch/layout/version supports determinism. citeturn15view1turn15view2turn13search2  

**Framework-level determinism**
- PyTorch: `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.benchmark=False`, `torch.backends.cudnn.deterministic=True`. citeturn22view0turn13search22  
- TensorFlow: call `tf.config.experimental.enable_op_determinism()` very early and seed all RNGs. citeturn17search2turn10search8  

**Attention stack**
- If using Transformer Engine, set `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`; disable FlashAttention backend if necessary (`NVTE_FLASH_ATTN=0`). citeturn20view0turn20view1turn19view0  
- Expect that deterministic attention backward can be a major throughput hit; evidence shows up to 37.9% kernel throughput loss for deterministic FA3 backward. citeturn8academia37  

**Distributed (NCCL)**
- Fix `NCCL_ALGO` when your training stack indicates sensitivity (Megatron-core explicitly does). citeturn19view0  
- Audit NCCL runtime choices using NCCL’s documented env tooling (e.g., `NCCL_DEBUG`). citeturn6search25  

**Compilation / graphs**
- Validate determinism in eager mode first; treat `torch.compile` as opt-in only after you pass reproducibility tests, given reports of nondeterminism under compile despite deterministic settings. citeturn18view0turn22view0  
- If using XLA, lock down autotuning via persisted results; use deterministic XLA flags only with awareness of unsupported ops. citeturn17search24turn17search27  

### Recommended tradeoffs for production LLM programs

A pragmatic split that balances cost and confidence:

- **Tier A: Continuous integration determinism (small scale)**  
  Run a short-horizon test (e.g., 20–200 steps) in strict deterministic mode (all knobs on) to catch regressions in numerics and distributed scheduling. This aligns with Megatron-core’s claim that deterministic mode should yield identical checkpoints/losses in the same environment, while acknowledging real-world regressions can occur and must be detected. citeturn19view0turn19view2  

- **Tier B: Throughput-optimized training (production runs)**  
  Accept nondeterminism for speed (enable flash attention, allow autotuning), but keep **seed discipline** and **stable data ordering** for statistical reproducibility. This matches the reality that deterministic kernels may carry large costs (e.g., deterministic FA3 backward) and that determinism is not equivalent to stable performance metrics. citeturn8academia37turn11search1turn17search2  

- **Tier C: Forensic reproduction mode (bug triage)**  
  When debugging, disable compilation (`torch.compile`) due to reproducibility risk reports, enforce deterministic algorithms, and reduce distributed complexity (single node if possible) to isolate kernel-level nondeterminism. citeturn18view0turn22view0turn19view2  

### Evidence URLs (exact)

```text
# NVIDIA / CUDA / cuBLAS reproducibility
https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#results-reproducibility

# PyTorch deterministic algorithms (definition, behavior, performance notes, Inductor deterministic mode)
https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html

# TensorFlow op determinism API
https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism

# TensorFlow “What’s new in TF 2.9” (mentions determinism performance tradeoff)
https://blog.tensorflow.org/2022/05/whats-new-in-tensorflow-29.html

# Transformer Engine PyTorch API docs (FlashAttention nondeterminism and deterministic gating)
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html

# Transformer Engine environment variables (NVTE_ALLOW_NONDETERMINISTIC_ALGO, NVTE_FLASH_ATTN, etc.)
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/envvars.html

# cuDNN frontend SDPA / attention docs (use_deterministic_algorithm in backward)
https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html

# cuDNN backend release notes (deterministic SDPA backward support on Blackwell, perf notes)
https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.1/release-notes.html

# Megatron-core reproducibility statement and deterministic-mode notes
https://pypi.org/project/megatron-core/0.12.1/

# Megatron multi-node determinism regression report (example of failures despite deterministic flags)
https://github.com/nvidia/megatron-lm/issues/2369

# PyTorch compile reproducibility/correctness issue (example: nondeterministic outputs despite determinism settings)
https://github.com/pytorch/pytorch/issues/159855

# Example performance regression with CUBLAS_WORKSPACE_CONFIG enabled (throughput drop in reproduction)
https://github.com/ultralytics/ultralytics/issues/19127

# DASH paper: deterministic attention backward throughput drop up to 37.9%
https://arxiv.org/abs/2601.21824

# OpenXLA determinism guidance (persisted autotuning for deterministic compilation)
https://openxla.org/xla/determinism

# XLA flag definition for xla_gpu_deterministic_ops and limitations
https://android.googlesource.com/platform/external/tensorflow/+/632ff3f6169ef18a6947c53bd6f3cb5bf7fc26a6/tensorflow/compiler/xla/xla.proto

# NCCL environment variable reference
https://docs.nvidia.com/deeplearning/nccl/archives/nccl_21212/user-guide/docs/env.html
```