---
description: Architecture-first deep dive on deterministic training in MaxText across Dense, MoE dense_matmul, and MoE sparse_matmul on MI355X, with code-path evidence and 1000-step replay validation.
title: Deterministic Training in MaxText on MI355X - Architecture, Kernel Mechanics, and 3-Setup Trade-offs
image: ./images/3-setup-deterministic-comparison-thumbnail.webp
---

# Deterministic Training in MaxText on MI355X - Architecture, Kernel Mechanics, and 3-Setup Trade-offs

June 2026 by [Author Name], [Co-Author Name]  
18 min read

Deterministic training is usually presented as a switch. In real LLM systems, it is a contract across multiple layers: model graph semantics, kernel implementation, collective behavior, compiler decisions, and PRNG handling. If one layer is unstable, replay divergence shows up even when seeds and hardware are fixed.

This article is intentionally architecture-first. We spend most of the post on implementation mechanics and root-cause paths, then close with the experiment matrix and deployment recommendations.

We answer four concrete questions:

1. Where does non-determinism enter in Dense vs MoE dense_matmul vs MoE sparse_matmul?
2. Which controls are functionally required, and which are defensive?
3. Why does deterministic overhead differ across the three setup classes?
4. What do 1-node MI355X experiments say about throughput, memory headroom, and long-run replay stability?

Terminology used in this post:

- `pdbs`: per-device batch size.
- `TGS`: tokens per second per device.
- `sec/step`: end-to-end wall-clock training step time.

---

## Executive Summary

- The strongest production signal in this campaign is envelope compression: Dense deterministic feasible batch tops out at `pdbs=11` vs non-deterministic `>=16`; MoE dense_matmul deterministic tops out at `pdbs=6` vs non-deterministic `>=7` (with a documented non-det `pdbs=8` run); sparse+shardy is capped at `pdbs=1` in both modes.
- On single-node MI355X, deterministic replay is stable for all three tested setups in feasible batch regions (1000-step loss-stream checksum matches).
- Single-node non-deterministic A/B negative-control is now included for all three setup classes and shows early divergence in every class (first divergence step: Dense `2`, MoE dense_matmul `41`, MoE sparse_matmul `34`).
- Overhead is reported as relative increase in `sec/step`: Dense is ~`+7%`, MoE dense_matmul is ~`+12%`, and sparse+shardy shows no measurable delta at its only feasible point (`pdbs=1`).
- Measurement caveat: each matched point is `N=1` run per mode, so these are point estimates, not confidence intervals.
- The dominant shared risk surface is attention backward determinism gating in TE/CK. MoE setups add router + dispatch/combine order sensitivity; sparse path adds extra ragged/grouped execution complexity.
- Deterministic mode impacts both kernel speed and feasible operating envelope (OOM ceilings), so policy should be based on matched-batch and roofline views together.

---

## 1) Determinism in LLM Training: What "Same Seed" Misses

In high-throughput GPU training, run-to-run divergence can be caused by:

- non-deterministic kernel algorithms (for example, atomic accumulation order),
- data-dependent graph behavior (for example, MoE token routing),
- collective execution order differences,
- PRNG implementation choices and where they are applied in init/runtime.

This is why deterministic policy cannot be reduced to one flag. It requires a chain of controls that map to each risk surface.

![Stack risk map: MaxText graph -> JAX/XLA lowering -> TE fused attention -> MoE routing/dispatch -> RCCL collectives -> PRNG/dropout.](./images/figure-01-system-risk-map.png)
*Figure 1. Non-determinism risk surfaces across the stack.*

---

## 2) Three Setup Classes, Three Different Mechanical Profiles

To understand deterministic overhead, we need path-level mechanics, not just model names.

### Dense (`llama2-70b`)

Core path:

- self-attention
- dense MLP matmuls
- standard all-reduce style gradient sync

Determinism implication:

- attention backward dominates sensitivity; no MoE routing pipeline exists.

### MoE dense_matmul (`ds-proxy-e128-h2048`, `sparse_matmul=False`)

Core path:

- attention
- MoE top-k router
- token dispatch/combine with fixed-capacity dense expert math path

Determinism implication:

- inherits attention risk and adds order-sensitive token movement surfaces.

### MoE sparse_matmul (`ds-proxy-e128-h2048`, `sparse_matmul=True`, `shardy=True`)

This setup is intentionally evaluated as a `sparse_matmul + shardy` bundle (`shardy=True`), because sparse path execution in this stack is gated on that partitioning mode.

Core path:

- attention
- MoE top-k router
- sparse/ragged grouped execution path in expert computation

Determinism implication:

- includes all MoE dense risks plus additional ragged/group-size/permutation complexity.

This setup-level difference is the main reason a single deterministic overhead number is misleading.

---

## 3) Deterministic Control Chain Implemented in This Stack

Deterministic behavior is enforced through a three-layer chain.

### Layer A: launcher/runtime env controls (`train_env.sh`)

Deterministic mode enables:

- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`
- `TF_DETERMINISTIC_OPS=1`
- `HIPBLASLT_DETERMINISTIC=1`
- `NCCL_ALGO=Ring`
- `RCCL_MSCCLPP_ENABLE=0`
- `JAX_DEFAULT_PRNG_IMPL=threefry2x32`
- XLA autotune safeguard (`xla_gpu_autotune_level=0`)

For this 1-node campaign, the RCCL knobs (`NCCL_ALGO=Ring`, `RCCL_MSCCLPP_ENABLE=0`) are carried for scale-out consistency but were not exercised as divergence triggers.

### Layer B: MaxText init override fix (`utils/deterministic.py`)

MaxText init sets `jax_default_prng_impl` to `unsafe_rbg`. The deterministic patch wraps init and re-applies the requested PRNG implementation post-init, then verifies key env values.

### Layer C: kernel gate application in TE attention path

TE JAX fused attention reads `NVTE_ALLOW_NONDETERMINISTIC_ALGO` and uses it to drive deterministic selection in fused attention forward/backward paths.

Without all three layers, deterministic assumptions can silently fail.

![Control chain: train_env.sh -> deterministic.py patch -> TE deterministic gate.](./images/figure-02-deterministic-control-chain.png)
*Figure 2. The enforced deterministic control chain.*

---

## 4) Required vs Defensive Controls (What Actually Matters)

From source tracing and operational judgment in this stack (the single-flag ablation grid is not shown in this post):

| Control | Evidence shown in this post | Assessed role for this stack |
|---|---|---|
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | TE gate path is source-traced; bundled replay is stable | Assessed primary required candidate; standalone toggle result not shown here |
| `xla_gpu_autotune_level=0` | Runtime enforcement is verified; image default is confirmed | Important guardrail; standalone required/not-required status not isolated in this post |
| `HIPBLASLT_DETERMINISTIC=1` | Runtime enforcement is verified; deterministic filtering intent is source-traced | Defensive candidate for this stack; standalone toggle result not shown here |
| `TF_DETERMINISTIC_OPS=1` | Runtime enforcement is verified; fallback path rationale is source-traced | Defensive candidate for this stack; standalone toggle result not shown here |
| `NCCL_ALGO=Ring`, `RCCL_MSCCLPP_ENABLE=0` | Config carried in runs; 1-node replay is stable | Scale-out defensive candidate; not validated as divergence trigger in 1-node runs |
| `JAX_DEFAULT_PRNG_IMPL=threefry2x32` + init patch | Post-init override is source-traced and runtime-verified | Reproducibility guardrail candidate; single-node same-partition necessity not isolated here |

This table is intentionally an assessed-role map, not a fully measured causal partition.

---

## 5) Attention Deep Dive: Why This Is the Highest-Risk Surface

Attention backward is the critical shared component across all three setup classes.

### Pre-fix behavior (historical problem)

In pre-fix TE lineage, deterministic intent did not fully propagate to CK fused attention backward; this forced a costly workaround path (`NVTE_FUSED_ATTN=0`) in older workflows.

### Post-fix behavior in the deterministic campaign image

With TE deterministic wiring fixed, deterministic mode keeps fused attention and dispatches deterministic CK behavior through `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`.

Evidence chain:

1) **TE JAX gate read (`transformer_engine/jax/cpp_extensions/attention.py`):**

```python
def is_non_deterministic_allowed():
    return bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
```

2) **Deterministic call-site argument (`transformer_engine/jax/cpp_extensions/attention.py`):**

```python
deterministic=not FusedAttnHelper.is_non_deterministic_allowed(),
```

3) **CK backward workspace growth (TE `fused_attn_ck.cpp`, as traced in the container source audit):**

```cpp
size_t kN0 = (d_qk <= 128) ? 128 : 64;
size_t nsplits = deterministic ? ceil(1.0 * s_kv / kN0) : 1;
void* dq_acc_ptr = planner.allocate(nsplits * h * max_tokens_q * d_qk * sizeof(float));
```

Mechanically, this implies:

- non-deterministic mode can rely on atomic accumulation order in backward,
- deterministic mode uses split accumulation and ordered reduction behavior.

That improves replay stability but adds extra workspace and some throughput cost.

### Why memory headroom drops in Dense deterministic mode

Deterministic CK backward allocates split `dQ` accumulation buffers. The memory increment grows with sequence length and split count:

`extra_bytes ~= (ceil(s_kv / kN0) - 1) x num_heads x (batch x s_q) x head_dim x 4`, with `kN0 = 128` if `d_qk <= 128` else `64`.

Worked example using nominal llama2-70b shapes (`s_q=s_kv=4096`, `num_heads=64`, `head_dim=128`) and campaign batch `pdbs=11` gives `nsplits=32` and an incremental `dQ` workspace of about `42.6 GiB` per device. Against a `.93` HBM pool budget, this is a meaningful (but not sole) contributor to the observed `pdbs=11` to `pdbs=12` feasibility cliff. This is directionally consistent with the observed deterministic headroom loss, but it is not a full allocator-state reconstruction.

In the accompanying technical reference, the same allocation relation (using the total `nsplits` term) was validated against 13 TE unit-test configurations, matching measured peak allocation within about `0.25 MB`. Those checks are on smaller test shapes; because the relation is linear in the relevant dimensions, this post uses it as an extrapolation to campaign-scale shapes (the worked example here uses the incremental `(nsplits - 1)` form).

From CK algorithm structure, deterministic `convert_dQ` reduction is expected to be HBM-bandwidth-bound, so overhead is expected to increase with `nsplits` alongside memory growth. Quantitative attribution of that effect is deferred to the planned rocprof annex.

There is currently no upstream `nsplits`-capping runtime knob exposed, so batch/headroom management remains the primary mitigation in production settings.

The deterministic campaign data reflects reduced max feasible batch compared with non-deterministic mode. Runtime dispatch sanity check is available via `NVTE_LOG_CK_CONFIG=1` (`attn_bwd(ck): ... deterministic: 1`). This post's kernel evidence is source-level plus runtime CK dispatch logs; a full rocprof kernel-timeline annex is left as follow-up material.

---

## 6) MoE Deep Dive: Router + Dispatch + Sparse Path Mechanics

MoE introduces deterministic complexity that Dense path does not have.

### Router behavior

In MoE code path, routing can be random or deterministic top-k:

```python
if self.config.use_random_routing:
    top_k_weights, top_k_indices = random_routing(...)
else:
    top_k_weights, top_k_indices = jax.lax.top_k(...)
```

For deterministic replay campaigns, random routing must stay off.

### Dispatch/combine and collective ordering

MoE introduces:

- explicit token reorder/permutation operations,
- inter-device token movement (for expert ownership),
- additional reduction/combine operations.

These are order-sensitive surfaces that increase deterministic risk versus Dense.

### Why `xla_gpu_deterministic_ops=true` is intentionally avoided for MoE

On this stack, forcing XLA scatter determinism can serialize MoE scatter/scatter-add behavior and create severe throughput collapse in MoE-heavy graphs, while adding no practical value for tested deterministic outcomes. In prior internal MoE ablation on this software stack (`ds-proxy-e128-h2048`, `pdbs=4`, outside the `18989-19030` campaign), enabling this flag increased steady-window `sec/step` mean (steps 5-14) from `1.90` to `264.5` (~139x), not including compile/startup effects. The deterministic policy therefore does not set this flag.

This is a key architecture decision: not all "deterministic-looking" flags improve practical deterministic workflows.

---

## 7) Pre-Experiment Cost Model (What We Expect Before Measuring)

Given path mechanics, expected behavior is:

- **Dense**: moderate deterministic overhead, mostly from attention path changes and workspace pressure.
- **MoE dense_matmul**: larger overhead due to attention + routing/dispatch/combine interaction.
- **MoE sparse_matmul**: small matched-batch delta possible at low feasible batch, but with tighter global headroom and broader complexity surface.
- **Quantitative directional prediction**: at their respective matched feasible batches, MoE dense_matmul overhead is expected to be larger than Dense overhead (an ordinal comparison, not a controlled same-batch cross-setup estimate).

The experiments below are consistent with this model.

---

## 8) Experimental Methodology

### Setup and controls

- Node: `chi2816` (single node, 8x MI355X)
- Image: `rocm/jax-training:maxtext-v26.2-det-te508-aot`
- Deterministic toggle: `_env_DETERMINISTIC_MODE=1`
- Performance/sweep campaign jobs: `18989` to `19015`
- Replay-correctness campaign jobs: `19023` to `19030`

### Evaluation lenses

1. **Matched-batch fairness**: deterministic vs non-deterministic at same successful `pdbs`
2. **Roofline-best by mode**: best achieved throughput per mode including OOM ceilings

### Statistics policy (to avoid over-claiming precision)

- Matched-batch comparisons are single run-pairs (`N=1` deterministic run, `N=1` non-deterministic run).
- Reported `sec/step` values are means over the steady window (steps 5-14), with within-run spread reported as coefficient of variation (CV).
- Within-run CV reflects step-to-step variability inside one run and should be treated as a lower bound on true run-to-run variability.
- No run-to-run confidence interval is claimed from this dataset.
- For tiny deltas smaller than observed jitter, interpretation is "no measurable difference in this campaign."

### Correctness criterion

Long-horizon deterministic A/B replay with 1000-step checksum matching per setup.

Checksum method:

- Input stream: per-step loss scalars parsed from training logs in step order.
- Encoding: each loss is packed as IEEE754 float32 (`struct.pack("!f", ...)`) and fed into SHA-256.
- Reported value: first 16 hex chars (64-bit prefix) of the SHA-256 digest.
- Interpretation: matching values indicate bit-identical logged loss streams for recorded steps; this is replay evidence, not a full parameter-tree hash.

---

## 9) Results

### 9.1 Matched-batch deterministic overhead

| Setup | Matched batch | Runs (det/non-det) | Det sec/step (10-step mean, steps 5-14) | Non-det sec/step (10-step mean, steps 5-14) | Deterministic overhead (`sec/step`) | Det TGS | Non-det TGS | Within-run CV (`sec/step`) det/non-det |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 11 | 1 / 1 | 21.534 | 20.108 | +7.1% | 2093.10 | 2240.67 | 2.06% / 0.26% |
| MoE dense_matmul | 6 | 1 / 1 | 2.753 | 2.462 | +11.8% | 8925.83 | 9984.58 | 0.46% / 1.49% |
| MoE sparse_matmul (+shardy) | 1 | 1 / 1 | 5.835 | 5.814 | +0.4% | 702.02 | 704.93 | 0.86% / 2.64% |

Interpretation:

- deterministic cost is setup-class dependent,
- MoE dense_matmul is the largest measured overhead among tested classes at matched batch,
- sparse+shardy is only feasible at `pdbs=1`; at that only feasible point, no measurable performance difference is observed,
- Dense sensitivity check at a less edge-constrained common point (`pdbs=8`) gives ~`+6.3%`, consistent in direction with the `pdbs=11` result,
- Dense CV asymmetry narrows at `pdbs=8` (det/non-det: `1.10% / 0.27%` vs `2.06% / 0.26%` at `pdbs=11`), which is consistent with stronger allocator-edge effects near deterministic batch ceiling.
- `sec/step` is the primary measured signal; TGS is a derived rate at matched batch and is shown only once for readability.

### 9.2 Roofline-best by mode

| Setup | Det best (sweep) | Non-det best (sweep) | Notes |
|---|---|---|---|
| Dense | `pdbs=10`, TGS `2115.76` (`18994`) | `pdbs=16`, TGS `2257.42` (`18992`) | Det OOM at `pdbs=12` |
| MoE dense_matmul | `pdbs=5`, TGS `9006.11` (`19003`) | `pdbs=7`, TGS `10193.35` (`19000`) | Det OOM at `pdbs=7`; non-det `pdbs=8` (`19001`) runs but is not throughput-optimal |
| MoE sparse_matmul | `pdbs=1`, TGS `702.02` (`19008`) | `pdbs=1`, TGS `704.93` (`19006`) | Both modes OOM at `pdbs=2` |

Interpretation:

- deterministic mode affects feasible operating envelope, not only per-step speed.
- Dense detail: deterministic throughput peaks at `pdbs=10` (`2115.76`), so the matched-point comparison at `pdbs=11` is slightly past the deterministic local optimum.
- absolute rates are not comparable across setup classes (`llama2-70b` vs MoE proxy); only deterministic vs non-deterministic deltas within the same setup are meaningful.

### 9.3 Memory boundary outcomes

Representative failed-allocation requests at OOM boundary runs:

- Dense deterministic OOM: `pdbs=12` (`18996`, alloc `180.88GiB`)
- MoE dense_matmul deterministic OOM: `pdbs=7` (`19005`, alloc `121.71GiB`)
- MoE sparse_matmul OOM at `pdbs=2` in both modes (`19007`, `19009`, alloc `133.96GiB`)

Interpretation:

- `alloc` is the single failed allocation request size from the XLA BFC allocator log (`ran out of memory trying to allocate ...`), not per-GPU HBM high-water usage.
- This stack uses `XLA_PYTHON_CLIENT_MEM_FRACTION=.93`, so allocator behavior reflects both preallocated pool limits and live-buffer/contiguity constraints at failure time.
- Feasible `pdbs` ceilings are the primary boundary signal; `alloc` sizes are secondary diagnostics.
- sparse_matmul remains headroom-constrained across modes; policy cannot rely on matched-batch delta alone.

### 9.4 1000-step loss-stream replay correctness

| Setup | Replay batch (`pdbs`) | Run A | Run B | Checksum result |
|---|---:|---|---|---|
| Dense | 11 | `19023`: `234ce488261af32b` | `19026`: `234ce488261af32b` | Match |
| MoE dense_matmul | 6 | `19027`: `b1f3004814ad6ba4` | `19028`: `b1f3004814ad6ba4` | Match |
| MoE sparse_matmul (+shardy) | 1 | `19029`: `665161ff2834e9f6` | `19030`: `665161ff2834e9f6` | Match |

All three setups match exactly at 1000 steps for tested configurations and batches.

### 9.5 Non-deterministic A/B negative-control replay (single-node)

To ensure deterministic replay evidence is not a logging artifact, we ran matched non-deterministic A/B controls per setup class (same node, same config family, only run label differs).

| Setup | A run | B run | Common aligned steps | First divergence step | A loss @ divergence | B loss @ divergence | Run status |
|---|---|---|---:|---:|---:|---:|---|
| Dense (`llama2-70b`, `pdbs=11`) | `19749` (`a2`, target 1000-step) | `19756` (`b200`, target 200-step) | 3 | **2** | 10.831 | 10.660 | both cancelled after divergence decision |
| MoE dense_matmul (`ds-proxy-e128-h2048`, `pdbs=6`) | `19747` | `19748` | 1000 | **41** | 11.225 | 11.226 | both success |
| MoE sparse_matmul (+`shardy`, `pdbs=1`) | `19753` (`a200`) | `19754` (`b200`) | 200 | **34** | 9.929 | 9.931 | both success |

Interpretation:

- All three setup classes show non-deterministic A/B divergence under aligned comparison.
- Dense divergence appears very early (step `2`), with step `0` and step `1` matching before separation.
- This control upgrades the replay claim from self-consistency (`det-A == det-B`) to controlled contrast (`non-det A != non-det B`) within the same measurement method.

![Results matrix: overhead, roofline best, OOM boundaries, replay checksums.](./images/figure-03-results-matrix.png)
*Figure 3. Consolidated view across performance, memory, and correctness.*

---

## 10) Engineering Interpretation and Policy

The measured pattern is consistent with the path-level model:

- Dense deterministic cost is moderate and plausibly attention-dominated in this stack.
- MoE dense_matmul cost is larger and plausibly reflects combined attention plus routing/dispatch effects.
- Sparse+shardy shows no measurable matched-point delta at `pdbs=1` but remains globally headroom-constrained.
- The most robust production impact in this campaign is feasible-batch envelope compression.

### Recommended posture

| Setup | Functional health | Perf impact (matched batch) | Primary risk | Recommendation |
|---|---|---|---|---|
| Dense | Healthy | ~7% overhead (`N=1` pair) | Reduced deterministic batch ceiling | **Go** |
| MoE dense_matmul | Healthy with reduced headroom | ~12% overhead (`N=1` pair) | Deterministic OOM at higher batch | **Go with caution** |
| MoE sparse_matmul (`shardy=True`) | Healthy in feasible region | no measurable delta at only feasible point (`pdbs=1`, `N=1`) | Tight memory headroom | **Coverage/specialized path** |

### Guardrails

- keep deterministic controls explicit and runtime-verified
- evaluate matched-batch and roofline views together
- require long-horizon replay checksums in sign-off
- include non-deterministic A/B negative-control replay in validation cycles when feasible
- keep random routing disabled in deterministic replay campaigns
- treat throughput and memory as a coupled optimization target

---

## Evidence Gaps and Next Validation

### Known scope boundaries (this campaign)

- Matched-batch overhead estimates are based on single run-pairs (`N=1` per mode), with no run-to-run confidence interval.
- Checkpoint->resume deterministic replay was not tested.
- Dropout-on replay was not a primary campaign axis (PRNG controls are enforced, but dropout-path replay remains follow-up).
- Non-deterministic A/B negative-control is included for the three setup classes, but replicate count and depth are still uneven (Dense control uses a short common prefix in this report).
- Scope is single-node MI355X within the tested setup/batch envelope.

### Prioritized artifacts that upgrade evidence quality

1. **Single-flag ablation grid (1000-step replay pass/fail):**
   - Method: start from deterministic baseline, toggle one control at a time (for example `NVTE_ALLOW_NONDETERMINISTIC_ALGO`, `xla_gpu_autotune_level`, PRNG override), and record checksum pass/fail plus runtime notes.
   - Claim upgrade: moves Section 4 from assessed Required/Defensive roles to a measured Required/Defensive partition for this stack.

2. **Attention backward attribution annex (rocprof kernel timeline):**
   - Method: matched-batch det vs non-det profiling with kernel-level decomposition, isolating fused-attention backward and related workspace/reduction kernels.
   - Claim upgrade: moves Sections 5/7 from "mechanism exists and is consistent with observed overhead" to causal attribution of step-time overhead to attention backward (for example, reported as a measured share of total delta).

3. **Matched-batch replicates (`n>=3`) for overhead bands:**
   - Method: rerun matched det/non-det pairs per setup at least three times and report run-to-run spread (for example median + p5/p95).
   - Claim upgrade: moves decision-facing overhead numbers (`~7%`, `~12%`, sparse no-measurable-delta) from single-pair estimates to bounded operational expectations.

4. **Negative-control depth and replicate strengthening (`n>=3` where practical):**
   - Method: extend non-deterministic A/B controls with deeper matched horizons and/or repeated pairs per setup, then report divergence-step distribution rather than single-point divergence.
   - Claim upgrade: moves current "control exists and diverges" evidence to quantitatively stable control behavior bands for operational sign-off.

These prioritized artifacts are intentionally scoped to close the largest remaining gaps between narrative strength, causal attribution, correctness controls, and decision confidence.

---

## Scope and Limitations

- scope is single-node MI355X and the tested setup/batch envelope
- matched-batch overhead estimates are based on single run-pairs (`N=1` per mode); no replicate confidence interval
- checkpoint->resume deterministic replay was not tested in this campaign
- explicit dropout-on replay coverage was not a primary campaign axis; PRNG controls are enforced but dropout-path replay should be validated in a dedicated follow-up
- findings are not universal across all model families, software versions, or multi-node fabrics
- multi-node deterministic behavior should be validated in a dedicated campaign; collective pinning flags were carried but not stress-tested as divergence triggers here

---

## Final Takeaway

The core finding is not a single deterministic tax number. It is that deterministic behavior is architecture-shaped.

In these MaxText setup classes:

- replay stability is strong with proper control chain enforcement,
- cost profile differs materially by setup/path bundle,
- and source-level risk mapping explains the observed throughput/headroom differences.

For production engineering, the durable workflow is:

**trace -> enforce -> replay-verify -> measure-for-attribution (next loop)**

That is how deterministic training becomes a reliable platform capability instead of an ad hoc debugging mode.

