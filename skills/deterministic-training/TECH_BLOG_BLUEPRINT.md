# Deterministic Training on ROCm — Tech Blog Blueprint

**Purpose:** Elevate the deterministic training work to company tech blog quality at software architect level. This document is a planning blueprint, not the blog itself.

---

## 1. Strategic Positioning

### Why This Matters (Business/Engineering Value)

| Audience | Value Proposition |
|----------|-------------------|
| **ML Platform teams** | Reproducible training enables debugging, A/B testing, and compliance (model audit trails). |
| **Research / Science** | Bit-exact reproducibility is table stakes for paper claims and regulatory submissions. |
| **Partners & Customers** | "We can reproduce your run" = trust, supportability, SLA guarantees. |
| **Recruiting** | Demonstrates deep systems thinking and full-stack debugging at scale. |

### Differentiation

- **Not** a generic "how to set env vars" post — we did a **full-stack audit** and **root-cause isolation** across 7+ layers.
- **Not** vendor-specific hand-waving — we traced source code, identified exact line numbers, and proposed minimal fixes.
- **Architect-level insight:** The bug is a 3-line wiring error in TE, not a fundamental limitation of CK or ROCm. CK already ships deterministic kernels; they were never enabled.

---

## 2. Blog Structure (Architect-Level Narrative)

### Recommended Outline

```
1. Hook & Problem Statement (2–3 paragraphs)
   - "Why can't I reproduce my training run?" — the pain every ML engineer has felt
   - Non-determinism hides in GPU kernels, collectives, PRNG — invisible until you need it
   - Our goal: bit-exact reproducibility for MaxText on ROCm, from 7B to 70B+ MoE

2. The Stack: Where Non-Determinism Can Hide (1 page)
   - Diagram: MaxText → JAX → XLA → TE → CK / hipBLASLt / rocBLAS → RCCL
   - Each layer: what could go wrong, and how we audited it
   - Key insight: 7 potential sources, 1 actual blocker — systematic elimination

3. Root Cause: A 3-Line Bug With a 9.7x Cost (1 page)
   - The TE hardcode: deterministic=false at 3 call sites
   - Why CK isn't the culprit: _deterministic kernels exist, never dispatched
   - Evidence: CK log, workspace sizing, dQ vs dK checksums
   - The workaround: NVTE_FUSED_ATTN=0 → unfused attention → OOM → batch 8→1

4. Architecture: Fused vs Unfused Attention (1 page)
   - Memory: O(seq²) vs O(seq) — why unfused explodes
   - Compute: atomicAdd vs per-split buffers + fixed-order reduce
   - Trade-off diagram: throughput vs determinism (current state vs post-fix)

5. Verification Methodology (0.5 page)
   - Ablation: remove one flag at a time, prove sole required setting
   - Multi-model: dense (7B, 70B), MoE (256 experts, 8 nodes)
   - Checksum-based validation: loss_checksum in logs

6. Results & Takeaways (0.5 page)
   - What works today: bit-exact on all tested configs
   - What's blocked: production throughput until TE fix lands
   - Upstream contribution: bug report + 3-line patch ready

7. Lessons for Architects (0.5 page)
   - Determinism is a cross-cutting concern — no single knob
   - Defensive flags vs required flags: ablation separates them
   - Upstream hygiene: trace the flag, don't assume it propagates
```

---

## 3. Content Transformation Map

### From Internal Docs → Blog-Ready

| Internal Doc | Blog Treatment |
|--------------|----------------|
| `deterministic_training_status_2026-03-15.md` | Condense to "What works / What doesn't" in Results section. Lead with the 3-line fix story. |
| `technical-reference.md` | Extract 2–3 key diagrams (stack, fused vs unfused). Do NOT dump the full table. |
| `SKILL.md` | Use architecture diagram and ablation table. Simplify "Known Gaps" to "Future Work." |
| `upstream-bugs/te-hardcodes-deterministic-false-for-ck-backend.md` | Core of "Root Cause" section. Keep evidence (CK log, dQ/dK), drop reproduction script. |
| `cheatsheet.md` | One-liner + env var table as "Quick Reference" sidebar or appendix. |

### What to Cut

- **Per-fix deep dives** (Fix 1–7) — blog readers don't need `fused_attn_ck.cpp:867` line references.
- **RCCL prefix parser limitation** — internal implementation detail.
- **hipBLASLt TensileLite predicate logic** — "we verified GEMMs are deterministic" is enough.
- **Docker image names, host names** — anonymize or genericize.

### What to Add

- **Problem framing** — "reproducibility crisis" in ML, regulatory/compliance angle.
- **Visual diagrams** — stack diagram, fused vs unfused memory/compute flow.
- **Before/after** — throughput table (968 vs 100 TFLOP/s) with clear "why."
- **Call to action** — link to upstream bug report, invite community to validate.

---

## 4. Architecture Diagrams (Must-Have)

### Diagram A: Full-Stack Non-Determinism Audit

```
┌─────────────────────────────────────────────────────────────────────────┐
│ MaxText train_step                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Attention ──► TE ──► CK fused_attn_bwd ──► atomicAdd (dQ)  ◄── BLOCKER │
│  Linear    ──► XLA ─► hipBLASLt ─────────► GEMM (deterministic)        │
│  RMSNorm   ──► JAX ─► jnp.mean ───────────► race-free                   │
│  Dropout   ──► JAX ─► threefry2x32 ───────► deterministic               │
│  Grad sync ──► RCCL ─► Ring ───────────────► fixed order                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Diagram B: Fused vs Unfused Attention (Memory & Throughput)

```
CK Fused (NVTE_FUSED_ATTN=1)          JAX Native (NVTE_FUSED_ATTN=0)
────────────────────────────          ─────────────────────────────
Single kernel: Q×K→softmax→×V         Three ops: GEMM + softmax + GEMM
Memory: O(seq_len) per head           Memory: O(seq_len²) per head
  → never materializes full matrix      → 137 GB @ batch=8, seq=4096
~968 TFLOP/s/device                   ~100 TFLOP/s/device
  (atomicAdd in bwd → non-det)          (deterministic, but OOM → batch=1)
```

### Diagram C: The Bug — Flag Propagation

```
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
  → TE Python: deterministic=True     ✓
  → JAX FFI: deterministic=true       ✓
  → C++ attention_hip.cpp             ✓
  → fused_attn.cpp:866                ✗ hardcodes false
  → CK backend: always _ndeterministic
```

---

## 5. Tone & Voice Guidelines

| Do | Don't |
|----|-------|
| Lead with the problem and the payoff | Lead with env var lists |
| Use "we audited," "we traced," "we proved" | Use passive voice |
| Explain *why* each layer matters | Assume reader knows XLA/TE/CK |
| Acknowledge the 9.7x cost honestly | Oversell the workaround |
| Invite upstream collaboration | Blame TE or CK teams |
| Keep technical depth where it matters (root cause) | Dumb down the architecture |

---

## 6. Quality Checklist (Company Tech Blog Bar)

- [ ] **Headline** — Clear, specific, searchable (e.g., "Achieving Bit-Exact Deterministic Training on AMD ROCm: A Full-Stack Audit")
- [ ] **Abstract/Summary** — 2–3 sentences for social/LinkedIn preview
- [ ] **Diagrams** — At least 2 (stack + fused vs unfused)
- [ ] **Code snippets** — Only the 3-line fix and the one-liner usage
- [ ] **Metrics** — Throughput table, checksum examples
- [ ] **Attribution** — Team, links to upstream bugs
- [ ] **SEO** — Keywords: deterministic training, ROCm, MaxText, reproducibility, AMD
- [ ] **Legal/Compliance** — Ensure no confidential data, host names sanitized

---

## 7. Deliverables Roadmap

| Phase | Deliverable | Owner |
|-------|-------------|-------|
| 1 | Draft blog post (Markdown/HTML) following outline | Author |
| 2 | Diagrams (Mermaid or exported SVG) | Author / Design |
| 3 | Internal review (technical accuracy) | SME |
| 4 | Copy edit (tone, clarity) | Editor |
| 5 | Publish + social promo | Comms |

---

## 8. Appendix: Key Facts for Fact-Checking

- **Models verified:** llama2-7b, llama2-70b, ds-proxy-se0-e256-h4096 (256 experts, 8 nodes)
- **Throughput:** 968 TFLOP/s (CK fused) vs 100 TFLOP/s (unfused workaround)
- **Root cause:** `fused_attn.cpp` lines 491, 680, 866 — change `false` to `deterministic`
- **Sole required flag:** `NVTE_FUSED_ATTN=0` (others are defensive)
- **CK status:** `_deterministic` kernels exist in `libmha_bwd.so`, never dispatched due to TE bug
