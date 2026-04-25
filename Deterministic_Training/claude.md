# claude.md

## Project goals
1. Find deterministic settings for AMD MaxText + JAX + ROCm LLM training
2. Build a practical deterministic training mode
3. Produce a high-quality technical blog based on evidence

## Stack components

Each layer in the determinism stack maps to specific software components:

| Layer | Component | Role | Repo / Docs |
|-------|-----------|------|-------------|
| Model/code | MaxText | Google's reference LLM training codebase | https://github.com/google/maxtext / https://maxtext.readthedocs.io |
| Model/code | Grain | Deterministic data pipeline for JAX | https://github.com/google/grain |
| Framework | JAX | Functional ML framework (PRNG, jit, pmap) | https://github.com/jax-ml/jax / https://jax.readthedocs.io |
| Framework | Flax | Neural network library for JAX | https://github.com/google/flax |
| Framework | TransformerEngine (TE) | Optimized transformer ops (attention, linear) | https://github.com/NVIDIA/TransformerEngine (AMD fork used in Primus) |
| Compiler | XLA (OpenXLA) | Compiler for JAX -> GPU kernels | https://github.com/openxla/xla / https://openxla.org |
| Communication | RCCL | AMD's collective communication library (NCCL equivalent) | https://github.com/ROCm/rccl |
| Runtime | ROCm / HIP | AMD GPU runtime and driver stack | https://github.com/ROCm/ROCm |
| Runtime | rocBLAS | AMD BLAS library for GPU | https://github.com/ROCm/rocBLAS |
| Runtime | hipBLASLt | AMD lightweight BLAS library (GEMM tuning) | https://github.com/ROCm/hipBLASLt |

### Key internal reference
- AMD-AGI/Primus: https://github.com/AMD-AGI/Primus -- AMD's training framework. Already has deterministic mode for Megatron-LM (PR #376). MaxText determinism is planned.

### Known determinism env vars (from Primus)
- `PRIMUS_DETERMINISTIC=1` (umbrella flag, sets the four below)
- `NCCL_ALGO=Ring` (communication layer -- fixed reduction order)
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` (framework layer -- TransformerEngine)
- `ROCBLAS_DEFAULT_ATOMICS_MODE=0` (runtime layer -- disable atomics in GEMM)
- `TORCH_COMPILE_DISABLE=1` (runtime layer -- avoid Triton race conditions)
- `XLA_FLAGS='--xla_gpu_deterministic_ops=true'` (compiler layer -- not yet in Primus, needs investigation for MaxText)

## Working style
- Always separate research, code investigation, experiment design, and writing
- Always identify which layer a setting belongs to:
  - model/code layer
  - framework layer
  - compiler/XLA layer
  - communication layer
  - runtime/system layer
- Do not mix speculative conclusions with verified conclusions
- Change one major variable at a time in experiments

## Required output format
For research or code investigation:
1. Question
2. Current hypothesis
3. Evidence
4. Relevant code locations
5. Confidence
6. Next verification step

For experiments:
1. Goal
2. Changed variable
3. Fixed variables
4. Expected deterministic effect
5. Expected performance effect
6. Validation method
7. Result
8. Conclusion

For blog writing:
1. Audience
2. Key message
3. Evidence to support it
4. Missing evidence
5. Proposed outline