# workflow.md

## Phase 1: Research
Goal:
- Understand existing determinism practices across vendors and frameworks

Tasks:
- Survey NVIDIA determinism resources (NVIDIA/framework-determinism repo, GTC talks, CUBLAS_WORKSPACE_CONFIG, CUDNN deterministic flags, etc.)
- Survey JAX/XLA determinism controls (jax.config flags, XLA compiler flags)
- Review AMD technical blog quality and style as reference for our blog output
- Collect determinism-related env vars and config flags across the full stack
- Identify gaps: what NVIDIA covers that AMD does not yet

Outputs:
- docs/research.md
- initial list of candidate settings
- list of hypotheses

## Phase 2: Code investigation
Goal:
- Map deterministic controls across MaxText / JAX / XLA / ROCm stack

Outputs:
- setting map
- code path notes
- unresolved questions

## Phase 3: Evaluation design
Goal:
- Define what "deterministic" means in this project

Required checks:
- same seed reproducibility
- same loss curve reproducibility
- same gradient/update reproducibility if feasible
- same checkpoint resume behavior if relevant
- step time / throughput impact

Outputs:
- docs/evaluation.md
- experiment matrix

## Phase 4: Experiments
Order:
1. single GPU
2. single node multi-GPU
3. multi-node

For each experiment:
- change one major factor
- record evidence
- classify result as:
  - deterministic gain
  - no effect
  - partial effect
  - unclear

Outputs:
- docs/experiments/*.md
- results tables

## Phase 5: Blog
Goal:
- Write an evidence-driven technical blog

Sections:
1. Why determinism matters
2. Determinism layers in AMD MaxText stack
3. Investigated settings and what they actually do
4. Recommended recipe
5. Performance tradeoffs
6. Open limitations