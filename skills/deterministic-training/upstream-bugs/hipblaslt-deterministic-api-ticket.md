# No public API to request deterministic GEMM in hipBLASLt

## Problem

hipBLASLt has no matmul descriptor attribute to request deterministic execution. TensileLite internally supports deterministic mode via `ContractionProblemGemm::setDeterministicMode(bool)` — this filters out solutions that use atomic GlobalSplitU or StreamKAtomic, which produce non-deterministic results due to floating-point reduction order varying across runs. However, hipBLASLt never exposes this to callers.

Consumers like XLA check `OpDeterminismRequired()` but have no way to communicate this to hipBLASLt. cuBLASLt on NVIDIA provides `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET` for this purpose; hipBLASLt has no equivalent.

## Current State

The TensileLite infrastructure is complete:

- `ContractionProblemGemm::setDeterministicMode(bool)` — sets the mode (`ContractionProblem.hpp:991`)
- `DeterministicModeEqual` predicate — filters solutions during kernel selection (`ContractionProblemPredicates.hpp:1218`)
- Solutions with atomic GlobalSplitU or `StreamKAtomic == 1` are tagged `DeterministicMode=False` (`Contractions.py:526,530`)
- Most BF16 solutions pass the filter — only atomic-GSU / StreamKAtomic variants are excluded

The gap is in hipBLASLt's `tensile_host.cpp`: `ConstructTensileProblem()` and `updateTensileProblem()` never call `setDeterministicMode(true)`, so the filter is never activated.

## Current Workaround

We apply a 10-line local patch in `tensile_host.cpp` that reads `getenv("HIPBLASLT_DETERMINISTIC")` and calls `setDeterministicMode()` at both `ConstructTensileProblem` and `updateTensileProblem`. This is applied in our Docker image but not upstream.

```cpp
// Added at end of ConstructTensileProblem() and updateTensileProblem()
static const bool s_deterministic = [] {
    const char* env = getenv("HIPBLASLT_DETERMINISTIC");
    return env && (env[0] == '1');
}();
tensileProblem.setDeterministicMode(s_deterministic);
```

## Observed Behavior

Without the patch, the same GEMM operation can produce different results across runs when TensileLite selects an atomic-GSU solution. With the patch and `HIPBLASLT_DETERMINISTIC=1`, results are bit-exact.

Our ablation testing on llama2-70b (BF16, gfx950) showed standard GEMM shapes do not select atomic-GSU solutions, so the patch has no effect for that model. However, non-standard shapes (tall-skinny matrices, MoE expert routing) may select atomic-GSU and would benefit from the filter.

## Expected Behavior

A public API — either a matmul descriptor attribute or an env var officially supported by hipBLASLt — that activates `setDeterministicMode(true)` in TensileLite. This would allow XLA and other consumers to request deterministic GEMM when needed.

## Environment

- Docker image: `rocm/jax-training:maxtext-v26.2`
- ROCm 7.1.1, hipBLASLt from `ROCm/rocm-libraries` (develop branch, commit `d055641a1`)
- GPU: 8× gfx950 (MI355X)

## Impact

Low urgency — this is a code quality / future-proofing request, not a current blocker. Standard training shapes on llama2-70b do not trigger atomic-GSU. But as models diversify (MoE, non-square GEMM shapes), a proper deterministic API prevents hard-to-diagnose non-determinism from surfacing.
