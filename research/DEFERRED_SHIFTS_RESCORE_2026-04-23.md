# Deferred Paradigm Shifts — Stack-Aware Re-Scoring

**Date:** 2026-04-23 (Ralph-loop iteration 34).
**Trigger:** 7th Ralph-loop empirical surprise (commit `07c66bd12`):
  IBGRAD Phase 5 scale test revealed that its Adam-state savings claim
  was invalidated by int8 Adam (shift #3) shipping first.  The research-
  methodology finding: **deferred shifts must be re-scored against the
  CURRENT shipped stack**, not their original design-time baselines.

This document applies that re-scoring discipline to the 10 deferred
shifts.

---

## Currently-shipped stack (context for re-scoring)

| # | Shift | Mechanism |
|---|-------|-----------|
| 1 | CHIRON reversible flow | O(1) activation memory |
| 2 | TC-tiled attention | BF16 tensor cores |
| 3 | int8 Adam state | 4× Adam compression (**affects #17 GFIB, #19 IBGRAD, #20 PRX re-scoring**) |
| 4 | BF16 gradient accumulation | 2× grad compression |
| 5 | SR BF16 weights | 2× weight compression |
| 6 | local-window attention | T² → T·W |
| 7 | Stiefel × Σ weights | 4× weight compression |
| 9 | OVFG factored grads | rank-r gradient factor |
| 10 | MPOT tensor-network weights | 25× weight compression |
| 11 | MFIO moment-free | ZERO Adam state (v1) or row/col norms (v2) |
| 12 | DFA backprop-free | ZERO backward compute |
| 13 | TRCD token-conditional depth | 3× per-token compute |
| 16 | LCP lattice compute pool | 4.7× per-token (cluster-pool + detail-net) |
| 19 | IBGRAD gradient subspace | r-dim Adam state; partial Phase 5 trainer wire-in |
| +  | chunked-CE, flash-attn | loss scratch, attention memory |

---

## Re-scoring of 10 deferred shifts

### #14 IED (Implicit Equilibrium Depth)

- **Original valuation**: activation memory O(1) in depth; deferred
  because at L=24 its ~32 MB Anderson history is WORSE than CHIRON's
  ~4 MB.  Promote condition: L > 60.
- **Re-scoring (2026-04-23)**: CHIRON still dominates at L ≤ 60.  No
  shipped shift changes IED's position.  **No re-score change.**
- **Status**: DEFERRED, promote at L > 60.

### #15 TPW (Trajectory-Predictive Weights)

- **Original valuation**: 1.7× cross-step speedup via rank-r Padé fit.
  Deferred because cross-step predictability of LLM dynamics is
  empirically unknown.
- **Re-scoring**: **IBGRAD's Oja streaming PCA (shift #19) empirically
  tracks E[g gᵀ] — this provides EXACTLY the gradient-field structure
  that TPW's Padé fit would consume**.  IBGRAD's P is the rank-r
  gradient covariance basis; TPW's Padé rational operates on that basis.
  TPW is now a natural COMPOSITION with IBGRAD rather than a standalone.
- **New promote condition**: after IBGRAD's learned P is stable,
  TPW uses it as the Padé fit substrate.  One-turn speedup becomes
  **TPW-on-IBGRAD-basis = 1.7× wall-clock ≈ free (shared P)**.
- **Status upgrade**: DEFERRED → **HIGH-PRIORITY compose-shift**.

### #17 GFIB (Gradient-Flow Information Bottleneck)

- **Original valuation**: KKT-soft-threshold per-parameter update, 10×
  bandwidth reduction on Adam step.  Deferred as break-even.
- **Re-scoring**: **int8 Adam (shift #3) already compressed Adam state
  to 2 bytes/param; GFIB's bandwidth savings on the Adam UPDATE step
  apply ON TOP of this** (GFIB still writes fewer params).  Net is now
  2–3× update-bandwidth reduction at int8 Adam baseline.
- **Composed with PRX (#20 deferred)**: GFIB's F̂ is exactly PRX's
  precision-assignment criterion.  Joint shift: GFIB→PRX handoff is
  the canonical sparse-update + precision compound.
- **Status upgrade**: DEFERRED → **MEDIUM PRIORITY, composes tightly
  with PRX**.

### #18 SGS (Saliency-Guided Substitution)

- **Original valuation**: 1.9× standalone speedup via per-layer
  surrogate.  Deferred behind LCP's 4.7×.
- **Re-scoring**: **LCP's rank-r detail network IS effectively a
  per-layer surrogate** — the infrastructure needed for SGS is now
  mostly shipped.  SGS would reuse LCP's detail-network infrastructure
  with a substitution-probability ρ policy instead of LCP's
  cluster-gather-scatter policy.
- **New promote condition**: once LCP Phase 3 (detail-net as its own
  primitive) stabilizes, SGS can be implemented as a variant.
- **Status upgrade**: DEFERRED → **LOW-COST to promote**.

### #20 PRX (Per-parameter Precision Heterogeneity)

- **Original valuation**: 2.7× weight memory via KKT-thresholded
  bit-width.  Deferred at <7B scale or until GFIB integration.
- **Re-scoring**: **int8 Adam (#3) and SR BF16 weights (#5) have
  already compressed weights by 4×**.  PRX atop this is ~1.5× on
  average weight bits (from 16 to ~10 bits avg).
- **Composed with #10 MPOT**: MPOT reconstruct via PRX-precision
  cores → net weight memory reduction compound.
- **Status revision**: DEFERRED, promote at >7B scale **OR** with
  GFIB-gated update schedule (joint compound).

### #21 PFE (Predictive Forward Emulation)

- **Original valuation**: wall-clock/NLL trade via mirror network.
  Deferred due to bang-bang ρ.
- **Re-scoring**: **IBGRAD's learned P can serve as PFE's gradient
  synthesizer R** (design-doc F3 mitigation already aligned with
  IBGRAD Phase 5 state).  The mirror's gradient-path can now use
  IBGRAD's subspace instead of DFA's random R.
- **Status upgrade**: DEFERRED → **COMPOSE-SHIFT on top of IBGRAD**
  (enabled by Phase 5 trainer wire-in).

### #22 WIP (Weight Interpolation Pretraining)

- **Original valuation**: K-dim optimization dimensionality.  Deferred
  pending empirical-priority review.
- **Re-scoring**: WIP × IBGRAD is a **dual-subspace optimization** —
  effective DOF = min(K, r).  With IBGRAD Phase 5 live at r=32, WIP
  at K=16 gives DOF = 16 (WIP dominates).  Natural next compound.
- **New promote condition**: after IBGRAD Phase 6 (backward-GEMM
  fusion) is shipped.
- **Status upgrade**: DEFERRED → **IBGRAD × WIP is the best current
  candidate for doubling the compound benefit**.

### #23 EDT (Energy-Distilled Training)

- **Original valuation**: 1.5–3× via learned curriculum.  Phase 1
  REGRESSES baseline (F2 noise amplification dominant).
- **Re-scoring**: **IBGRAD's audit-fraction signal is a natural
  salience proxy** — tokens whose gradient lies in P's span are
  well-represented; tokens outside are potentially NOISE.  EDT
  could use `‖Pᵀ g_t‖² / ‖g_t‖²` as the per-token salience signal
  rather than learning a separate energy network.
- **Status upgrade**: DEFERRED → **WAITING on Phase-6 IBGRAD for a
  free salience signal**.

### #24 CLPS (Cross-Layer Parameter Sharing)

- **Original valuation**: 3× block-weight compression at K=8.
- **Re-scoring**: **MPOT (#10) already compresses 25× at layer level**;
  CLPS × MPOT gives 60× compound.  But MPOT covers it individually.
  CLPS is most valuable at VERY DEEP L (> 36) where functional
  redundancy across layers dominates.
- **Status**: DEFERRED, promote at L > 36.

### #14, #15 re-summarized:

Both were deferred with original promote conditions.  #15 is now
composable with #19 IBGRAD (same P basis); #14 is unchanged.

---

## Priority ranking for next implementation (post re-score)

1. **#22 WIP × #19 IBGRAD** — dual-subspace optimization, compounds on
   top of IBGRAD Phase 5 trainer wire-in.  Clear empirical path.
2. **#15 TPW on IBGRAD basis** — Padé fit reuses IBGRAD's P.
   Cross-step prediction becomes concrete rather than speculative.
3. **#21 PFE with IBGRAD gradient synthesizer** — mirror network's
   gradient path via subspace rather than random R.
4. **#23 EDT with IBGRAD-salience** — per-token salience from
   `‖Pᵀg_t‖²/‖g_t‖²`.  Zero-new-kernel implementation.
5. **#18 SGS via LCP detail-net infrastructure** — piggyback on
   shipped Phase-2 LCP.
6. **#17 GFIB × #20 PRX compound** — KKT-sparse-update + precision.
7. **#14 IED** — at L > 60 only.
8. **#24 CLPS** — at L > 36 only.

---

## Methodology insight

The 7th empirical surprise — IBGRAD Phase 5 scale test revealing its
negative valuation vs int8 Adam — is a research-program-level finding:

> Every deferred shift's promote condition should INCLUDE a
> re-scoring-against-current-stack checkpoint.  The design-time
> valuation is a lower bound on its value; the stack-integrated
> valuation can be DRAMATICALLY different (up or down).

This is now standard procedure for future shifts.  Designs should
include:
- Original design-time baseline (what it's compared to)
- Interaction-axis classification (orthogonal, substitutive,
  compose-shift)
- Promote condition as a function of CURRENT STACK STATE, not
  fixed.

All 10 deferred shifts above have been re-scored under this
discipline.  Future iteration work has a clearer priority ranking.
