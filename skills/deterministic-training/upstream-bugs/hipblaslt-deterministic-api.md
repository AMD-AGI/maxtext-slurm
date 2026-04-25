# Feature Request: Add HIPBLASLT_MATMUL_DESC_DETERMINISTIC attribute

**Target repos:** `ROCm/rocm-libraries` (hipBLASLt), `ROCm/xla` (XLA ROCm backend)

## Summary

hipBLASLt has no public API to request deterministic GEMM execution. TensileLite internally supports `DeterministicMode` (filters out atomic-GSU solutions that produce non-deterministic results), but hipBLASLt never activates it. Adding a matmul descriptor attribute would allow XLA to request deterministic GEMM when `OpDeterminismRequired()` is true.

## Current State

TensileLite already has full infrastructure:
1. `ContractionProblemGemm::setDeterministicMode(bool)` / `deterministicMode()`
2. `DeterministicModeEqual` predicate filters solutions during kernel selection
3. Solutions using atomic GlobalSplitU or StreamKAtomic are tagged `DeterministicMode=False`
4. Most BF16 solutions pass the deterministic filter (only atomic-GSU / StreamKAtomic solutions are filtered)

The only gap: hipBLASLt never calls `setDeterministicMode(true)`.

## Current Workaround

A 10-line env var patch in `tensile_host.cpp` (at two call sites: `ConstructTensileProblem` and `updateTensileProblem`) reads `getenv("HIPBLASLT_DETERMINISTIC")` and calls `setDeterministicMode()`. Applied as a local modification in our Docker image (`rocm/jax-training:maxtext-v26.2`) but not upstream.

## Proposed Change

### hipBLASLt (5 files, ~30 lines)

1. `hipblaslt/include/hipblaslt.h` — Add `HIPBLASLT_MATMUL_DESC_DETERMINISTIC = 110` to `hipblasLtMatmulDescAttributes_t` enum
2. `hipblaslt/src/hipblaslt_internal.hpp` — Add `bool deterministic = false` field to `_rocblaslt_matmul_desc`
3. `hipblaslt/src/rocblaslt_mat_utils.cpp` — Handle get/set attribute for the new field
4. `hipblaslt/src/amd_detail/rocblaslt/src/tensile_host.cpp` — Read from `RocblasltContractionProblem` and call `setDeterministicMode()`
5. `hipblaslt/src/include/rocblaslt-types.h` — Add `deterministic` field to `RocblasltContractionProblem`

### XLA (1 file, ~5 lines)

6. `xla/stream_executor/rocm/hip_blas_lt.cc` — In `BlasLt::GetMatmulPlan`, set `HIPBLASLT_MATMUL_DESC_DETERMINISTIC` when `OpDeterminismRequired()`

## Risk

LOW:
- Only atomic-GSU / StreamKAtomic solutions are filtered — no "no solution found" risk for standard shapes
- Default is off — zero behavior change for existing users
- Worst case: ~5-10% perf loss for tall-skinny matrices that prefer atomic-GSU

## Experimental Evidence (March 2026)

### Ablation: patch not needed for standard shapes
Ablation testing on llama2-70b showed the env var patch has no effect — standard GEMM shapes don't select atomic-GSU solutions.

### GSU trigger test: patch not needed even for extreme MoE shapes
We designed a config to maximize GSU likelihood: ds-proxy-se0-e256-h4096 with `per_device_batch_size=1`, `max_target_length=2048`, `num_experts_per_tok=1` → 8 tokens/expert → MoE GEMM [8, 2048]×[4096, 2048] = 16 output tiles for 304 CUs (5% utilization). Results:

| HIPBLASLT_DETERMINISTIC | Runs | Checksum | Result |
|------------------------|------|----------|--------|
| `1` (patch active) | 3 | `ada48a7345ff4e4a` | BIT-EXACT |
| `0` (patch disabled) | 3 | `ada48a7345ff4e4a` | BIT-EXACT |

The gfx950/ROCm 7.1.1 BF16 solution library (23 `.dat` files, 0 StreamK files) contains no atomic-GSU solutions. hipBLASLt is **deterministic by default** on this platform. The patch is a no-op.

### gfx942 may differ
gfx942 (MI300X) has 93 BF16 `.dat` files and 100 StreamK files. Atomic-GSU and StreamKAtomic solutions may exist for gfx942, making the patch potentially necessary on that platform.

## Note

A proper API future-proofs against solution library changes in future ROCm releases. On the current gfx950/ROCm 7.1.1 platform, it is not functionally needed.
