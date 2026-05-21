# Paradigm shift #11 — design brief

**Date**: 2026-04-22.
**Status**: design phase.  Implementation deferred to subsequent iterations.

---

## Context — where the 16 GB ceiling is after shifts #1-#10

After all ten paradigm shifts + chunked CE land in the trainer, the
projected per-parameter memory cost on a 16 GB consumer GPU is:

| category             | per-param storage                          | attacked by     |
|----------------------|--------------------------------------------|-----------------|
| weights (compound)   | ~2 bytes·(2ρ)·(2D√(mn)/mn) ≈ 0.02–0.1 B/p  | #5 × #7 × #10   |
| gradients            | ~2 bytes · r/min(m,n) ≈ 0.1–0.3 B/p         | #9 (OVFG)       |
| optimizer moments    | ~2 B/p (int8 pair) or ~0 with OVFG          | #3, #9          |
| activations          | O(1) in depth                               | #1 (CHIRON)     |
| scratch/logits       | O(T·V_chunk)                                | chunked CE      |

At the ceiling, **Adam state is now the single largest persistent
VRAM line item** (int8 m + u8 v = 2 bytes/param, plus FP32 master = 4
bytes/param = 6 bytes/param).  At 30 B params this is 180 GB — far
over ceiling.

## Target for shift #11

**Eliminate the per-parameter optimizer state entirely.**  Instead of
storing (m_t, v_t) per parameter for Adam, replace the per-parameter
Adam update rule with a **meta-learned tiny optimizer network** that
outputs the update from (gradient, position encoding, time step)
without requiring any per-parameter memory.

If this works at LLM scale it unlocks 30 B → 100 B parameter training
on the same 16 GB hardware by collapsing the 6 B/p optimizer-state
burden to ~0.

## Three candidate formulations

### Candidate A — Meta-learned optimizer (LMO)

A tiny MLP `f_θ : (grad, prev_update, position_emb, step) → new_update`.
θ is meta-trained once on a distribution of tasks; at LLM training time
θ is frozen and used per-parameter.  Prior art: Andrychowicz et al.
"Learning to learn by gradient descent by gradient descent" (2016);
Metz et al. "Tasks, Stability, Architecture, and Compute" (2020).
These have shown meta-optimizers work for small nets but have NOT
been scaled to LLM.

**State eliminated**: m, v (the two moments).  Replaced by: one forward
pass through a ~10k-parameter MLP per Adam step.
**Added state**: θ (shared across all parameters) ≈ 40 KB.
**Per-param memory**: 0 (from 6 B/p → 0 B/p).

Risks: meta-learning convergence at LLM scale unproven; meta-overfitting
to the meta-training task distribution.

### Candidate B — Moment-free implicit optimizer (MFIO)

Frame Adam as a noiseless overdamped Langevin step with an
activation-derived preconditioner (ŝ_ℓ = activation norm ×
gradient norm, as in the earlier OVFG candidate SIU-FPF).  The key
realization: if the preconditioner only uses quantities already
produced by the backward pass (|z|², |δ|²), then neither m nor v
need persistent state.

    θ_{n+1} = θ_n − η · σ_ℓ(z, δ, t) · ∇L(θ_n)
    σ_ℓ(z, δ, t) = 1 / (ŝ_ℓ · β_schedule(t) + ε)

Prior art: SIU-FPF in `research/PARADIGM_SHIFT_9_CANDIDATE_B_SIU_FPF.md`
(rejected as #9 candidate but viable as standalone #11).

**State eliminated**: m, v.
**Added state**: L floats (one σ per layer, not per parameter).
**Per-param memory**: 0.

Risks: no explicit momentum; convergence depends on the Langevin-noise
schedule being well-tuned.

### Candidate C — Predictive residual Adam (PRA)

Keep the Adam update rule but SHARE (m, v) across a BLOCK of k
parameters.  Each parameter has its own deviation ε_i from the block's
shared moment, stored in 2 bits.  Gradient update uses block-shared
(m̄, v̄) + per-param ε_i adjustment.

**State**: shared (m, v) per block + 2 bits/param residual.
**Per-param memory**: 2 bits / param ≈ 0.25 B/p (24× less than int8 Adam).

This is a conservative compression — not a full elimination — but
with clearer convergence guarantees than A or B.

Prior art: Adafactor (row/col-only 2nd moment) but pushed to arbitrary
block structure.

## Selection: **Candidate B (MFIO)**

Rationale:
1. **Biggest upside** that still has tractable math: explicit 0
   optimizer state, NOT just compression.
2. **Reuses existing infrastructure**: CHIRON reversible flow already
   materializes z and δ on the backward; the activation norms are free.
3. **Decoupled risk**: if MFIO doesn't converge, falling back to int8
   Adam is one flag.  A & C both require deeper trainer refactors.
4. **Validated partial**: SIU-FPF (shift #9 candidate B) laid the
   math; just specializing to deterministic (noise=0) case is MFIO.

## MFIO formulation

### Primitive objects

- Weight θ_ℓ ∈ R^{m_ℓ × n_ℓ} per layer ℓ.
- Gradient g_ℓ = ∇_{θ_ℓ} L.
- Per-layer scalar σ_ℓ ∈ R_+ (NOT a buffer — recomputed each step).
- Per-layer β_schedule(t) ∈ R (cosine, linear, or step — tuned once).

### Update rule

On each optimizer step n:

1. During backward, CHIRON materializes per-layer pre-activation z_ℓ
   and gradient δ_ℓ for the GEMM that produced ∇θ_ℓ.  Compute on
   the same tile:

       ŝ_ℓ = (1/B) Σ_i ‖z_ℓ^(i)‖² · ‖δ_ℓ^(i)‖²         (1 FP32 per layer)

2. σ_ℓ = 1 / (ŝ_ℓ · β_schedule(n) + ε)                 (1 FP32 per layer)

3. θ_ℓ ← θ_ℓ − η · σ_ℓ · g_ℓ                          (dense update, no m, v)

### Why this converges

Cauchy-Schwarz gives ŝ_ℓ ≥ ‖∇θ_ℓ‖_F² / d_ℓ, so σ_ℓ is a lower bound on
the inverse Fisher diagonal averaged over layer ℓ — a coarse-grained
Adam preconditioner.  The β_schedule(n) provides the effective
warmup/decay that Adam gets from 1/(1-β_1^t).

### Composability

- **× CHIRON (#1)**: native — CHIRON reconstructs z, δ for free.
- **× TC attn (#2)**: orthogonal.
- **× int8 Adam (#3)**: *replaces*.
- **× BF16 grads (#4)**: composes — g_ℓ stays BF16.
- **× SR BF16 weights (#5)**: composes — θ_ℓ stays BF16, update
  uses stochastic rounding.
- **× local-attn (#6)**: orthogonal.
- **× Stiefel (#7)**: requires Riemannian variant — σ_ℓ · g_ℓ
  replaces the tangent grad; retraction via QR/Cayley unchanged.
- **× HRTC (#8)**: orthogonal.
- **× OVFG (#9)**: *subsumes OVFG's factored moment storage* — OVFG's
  whole purpose is compressing m, v, which MFIO eliminates.  In the
  composition, OVFG handles gradient representation; MFIO handles
  update.  Both ship; user picks.
- **× MPOT (#10)**: orthogonal — MFIO update applies to whatever
  weight parameterization is active.

### Memory accounting at 30 B params on 16 GB

Without MFIO (current best stack):
- Weights (MPOT × Stiefel × BF16): ~0.03 B/p × 30 B = 900 MB
- Gradients (OVFG factored): ~0.1 B/p × 30 B = 3 GB
- Adam int8 (m+v): ~2 B/p × 30 B = 60 GB  ← **blowing through ceiling**
- Activations (CHIRON): negligible
- Scratch: ~1.5 GB

MFIO eliminates the 60 GB line entirely, adding only L·8 bytes
(scalar σ + ŝ per layer) ≈ 400 bytes total.  New total: **~5 GB
at 30 B params**.  Unlocks headroom for an additional 40-80 B.

## Minimal prototype (≤ 2 weeks)

1. **Week 1 day 1–3**: `gpu_mfio.{h,cu}` with:
   - `mfio_activation_norms(z, δ, T, d, ŝ_out)` — two reductions per
     layer, fused into existing backward kernel where possible.
   - `mfio_update(θ, g, η, σ, n)` — Adam-less weight update.
2. **Week 1 day 4–5**: parity test
   `CHIRONMfioDescentTest` — run MFIO on a toy regression target and
   verify loss decrease matches Adam within 2× on small problems.
3. **Week 2**: trainer wire-in behind `--mfio` flag.  Replace
   Adam path; reuse β_schedule infrastructure.
4. **Week 2**: validate on pile_large smoke test (small ~80M model)
   for 500 steps; compare loss trajectory to int8 Adam baseline.

## Open research questions

1. **Convergence at LLM scale**: Langevin-without-noise training for
   2 B+ models is unproven.  The closest analogs (gradient descent
   with per-layer preconditioner) work but have weaker convergence
   than Adam.
2. **Stability with stochastic-rounded BF16 weights**: the rounding
   noise adds effective diffusion; the MFIO σ schedule must account
   for this.
3. **ŝ_ℓ collapse**: on dead layers where δ → 0, σ_ℓ → 1/ε explodes.
   Mitigation: clip σ_ℓ ≤ σ_max; monitor per-layer activation entropy.

## Failure modes and mitigations

1. **Slow convergence**: MFIO may need 2-3× more steps than Adam.
   Tracked metric: steps-to-target-loss vs Adam baseline.  If >3×,
   revert to Candidate C (block-shared moments with residuals).
2. **Loss spikes from activation-norm noise**: smooth ŝ_ℓ via EMA
   with β=0.99 (still only 2 floats/layer of state).
3. **Layer-specific tuning required**: some layers (LayerNorm γ,β,
   embeddings) may need distinct σ schedules.  Budget: L scalar
   state, well within headroom.

## Tracked as task #35 (to be created on implementation start)
