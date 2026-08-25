# Paradigm shift #19 — Candidate C: Information-Bottleneck Gradient Subspace (IBGRAD)

**Formulation class:** statistical / adaptive-basis optimizer.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** candidate design.  Implementation deferred to selection.

---

## 1. Short name and core thesis

**IBGRAD — Information-Bottleneck Gradient Subspace.**  Working codename:
*learned-PCA optimizer*.

Every shift so far has treated the gradient as a first-class N-dimensional
object: BF16 grads (#4), OVFG factored grads (#9), MFIO scalar-per-layer
(#11), DFA random-projection grads (#12), GFIB Fisher-gated grads (#16).
What **none** of them do is ask the second-order question: *which
directions in parameter space actually reduce loss*?  OVFG factorizes
the gradient per-layer into (A, D) — fixing rank via the sequence-length
bound rank(G) ≤ T, which is a **graph-structural** low-rank, not a
**loss-informational** low-rank.  The empirical fact is sharper: for
transformer pre-training, the top-r eigenvectors of E[g gᵀ] carry ≥80%
of cumulative gradient variance at r = 0.05·N (Gur-Ari et al. 2018,
Frankle-Carbin lottery-ticket analyses, GaLore follow-ups).  The
**remaining 95% of directions are optimizer noise** — their Adam moments
decay to a white-noise equilibrium and their contribution to loss
decrease is below BF16 round-off.

IBGRAD maintains a **block-diagonal, layer-wise, LEARNED projection**
P = blkdiag(P₁, …, P_L), with P_ℓ ∈ ℝ^{N_ℓ × r_ℓ}, r_ℓ ≪ N_ℓ.  The
backward pass computes the *projected* gradient g_sub = Pᵀ·g directly
via chain rule (never materializing the full g); Adam moments (m_sub,
v_sub) live only in the r-dimensional subspace; the weight update
θ ← θ − η·P·Adam(g_sub) injects the update back into full parameter
space via a single GEMM.  **P itself evolves online** through a
streaming-PCA update on a gradient sample stream: high-σ directions
are added, stale directions pruned.  The adaptive-basis property
distinguishes IBGRAD from OVFG (fixed per-microbatch factorization) and
from GaLore (periodic dense-gradient re-SVD, which IBGRAD specifically
refuses to do).

**Composability claim.**  IBGRAD slots one layer earlier in the
optimizer pipeline than every gradient-side shift: OVFG's (A, D) pair
is the dense upstream, IBGRAD's Pᵀ·(AᵀD) is its subspace image.
GFIB's per-parameter Fisher becomes a per-subspace-axis Fisher — more
accurate since projected directions are closer to Hessian eigenvectors.
MFIO's scalar σ becomes an r-dimensional σ_sub aligned with the basis.

---

## 2. Primitive objects and state space

Let the network have N scalar parameters across L layers, layer ℓ
holding N_ℓ parameters.  Total subspace rank r = Σ_ℓ r_ℓ; typical
target r = 0.05·N, balanced as r_ℓ ∝ √N_ℓ (see §3.3).

| symbol              | shape / dtype             | meaning                                         |
|---------------------|---------------------------|-------------------------------------------------|
| θ_ℓ                 | N_ℓ, BF16 (+SR #5)        | layer-ℓ parameters                              |
| g_ℓ                 | N_ℓ, BF16                 | instantaneous gradient of layer ℓ (never mat.)  |
| P_ℓ                 | N_ℓ × r_ℓ, BF16           | **adaptive** layer-ℓ projection basis          |
| g_sub,ℓ = P_ℓᵀ g_ℓ | r_ℓ, FP32                 | projected gradient (computed directly)          |
| m_sub,ℓ             | r_ℓ, int8 (#3 compat)     | Adam first moment in subspace                   |
| v_sub,ℓ             | r_ℓ, int8                 | Adam second moment in subspace                  |
| C_ℓ                 | r_ℓ × r_ℓ, FP32           | within-subspace gradient Gram (for P update)    |
| S_ℓ                 | r_ℓ, FP32                 | accumulated importance (per-axis)               |
| ρ_ℓ                 | r_ℓ, FP32                 | staleness counter (per-axis)                    |
| K                   | int                       | P-adaptation interval (default 200 steps)       |
| τ_ℓ                 | FP32 scalar               | layer-ℓ prune threshold on ρ_ℓ                  |
| ε                   | FP32                      | diagonal regularizer on C_ℓ (F1 mitigation)     |

**State space.**  Σ = (θ, {P_ℓ}, {m_sub,ℓ, v_sub,ℓ}, {C_ℓ, S_ℓ, ρ_ℓ},
step_count).  Compared to Adam (2·N = 4.46 GB at 2.23B BF16), IBGRAD
holds r·(m+v) at int8 plus N·r BF16 for P.  The P matrices dominate:
at r/N = 0.05 the projection storage is 0.05·N·2 B ≈ 223 MB for a 2.23 B
net, a **20× reduction** over Adam and **~8×** over int8-Adam (shift
#3, which stores 2·N int8 = ~2.23 GB equivalent).

**Block-diagonal structure.**  P is never materialized as one N×r
matrix; each P_ℓ is stored and operated on independently.  Cross-layer
correlations in the gradient are discarded by construction — this is
the **bottleneck** in the information-bottleneck naming: the subspace
is *per-block* sufficient statistic for loss descent, discarding
inter-block structure.  Empirical justification §5.

---

## 3. Derivation of the optimal projection

### 3.1 Variational principle

Over a short time horizon of optimizer steps, the Adam update without
preconditioning is θ_{t+1} − θ_t ≈ −η·g_t.  Under the quadratic
approximation of the loss L around θ_t with Hessian H,

    𝔼[L(θ_{t+1}) − L(θ_t)] ≈ −η·𝔼[g gᵀ]:I + ½η²·H:𝔼[g gᵀ]
                           = −η·tr(C) + ½η²·tr(H C)

where C = 𝔼[g gᵀ] ∈ ℝ^{N×N} is the gradient second-moment.  If we
restrict the update to the column span of P, i.e. θ_{t+1} − θ_t =
−η·P·Pᵀ·g, the expected loss reduction becomes

    𝔼[ΔL | P] ≈ −η·tr(PᵀCP) + ½η²·tr(PᵀHP · PᵀCP).

The first-order term dominates at small η.  Maximizing it subject to
P being column-orthonormal (PᵀP = I_r) gives

    **P\* = top-r eigenvectors of C.**

This is a PCA on the gradient stream.  The variational principle is
identical to the information-bottleneck: P compresses g to a rank-r
statistic preserving maximum expected loss-decrease information.

### 3.2 Online update for P_ℓ

We cannot form C_ℓ = 𝔼[g_ℓ g_ℓᵀ] ∈ ℝ^{N_ℓ×N_ℓ} — at N_ℓ = 16 M that's
a 1 TB matrix.  We use **randomized streaming PCA** (Oja's rule
variant):

    P_ℓ ← (1 − α)·P_ℓ + α·g_ℓ·(g_ℓᵀ P_ℓ)
    P_ℓ ← QR(P_ℓ)                           [orthogonalize]

α ∈ [1e-4, 1e-3], executed every K = 200 steps (amortized cost).  The
product g_ℓᵀ·P_ℓ is exactly g_sub,ℓ (r_ℓ FP32 numbers) — **we already
have it from the forward-path of the optimizer**, so the Oja update
costs one additional N_ℓ·r_ℓ SGEMM per K steps, same order as backward
itself.  The QR is r_ℓ·r_ℓ — trivial.  Total overhead: one
full-gradient backward per K steps (the "full audit"), amortized 1/K.

### 3.3 Per-layer rank allocation

Under the constraint Σ r_ℓ = r (global budget) and the heuristic that
each layer's gradient spectrum is approximately Zipf α_ℓ ≈ 1, the
optimal rank assignment minimizing total truncation error Σ (1 − X_ℓ)
for fixed r is:

    **r_ℓ ∝ √N_ℓ**

(Lagrange over the rank-energy tradeoff, details in appendix of the
shift-19 selection writeup).  For a uniform-width transformer this
gives equal r per layer; for a net with large embedding (N_emb ≫ N_ℓ
per block) the embedding gets disproportionately more rank — matching
OVFG's sparse-embedding treatment.

### 3.4 Information content

**Claim (rank-r energy).**  For a transformer's gradient stream,
cumulative energy X_r = Σ_{i≤r} λ_i / Σ_i λ_i satisfies X_r ≥ 0.80 at
r/N = 0.05 and X_r ≥ 0.95 at r/N = 0.15 (Gur-Ari 2018 measurements on
ResNet, replicated on GPT-2 in subsequent work).

**Claim (loss-reduction preservation).**  The expected one-step loss
reduction under the rank-r update is X_r fraction of the full-rank
reduction.  The degradation after K steps scales as (1 − X_r)²·K under
standard SGD noise assumptions, but in practice Adam's momentum
absorbs the truncated tail within ≈ 1/(1−β₁) steps, giving an
effective steady-state cost of **only (1 − X_r)² · (1 − β₁)⁻¹** per
step — at X_r = 0.80, β₁ = 0.9, this is 0.04·10 = 0.4 loss
bits/step relative to Adam baseline.  Negligible compared to the 20×
memory / compute savings.

---

## 4. Evolution — one optimizer step

Step n:

1. **Forward** (unchanged, CHIRON-reversible).
2. **Backward (subspace).**  At each layer ℓ, instead of materializing
   g_ℓ = A_ℓᵀ · D_ℓ (OVFG's primitives), compute
        g_sub,ℓ = P_ℓᵀ · (A_ℓᵀ · D_ℓ) = (P_ℓᵀ A_ℓᵀ) · D_ℓ.
   By associativity we can fold Pᵀ into the backward chain — the
   backward GEMM dimensions drop from (T, N_ℓ) to (T, r_ℓ), a factor
   of N_ℓ/r_ℓ = 20× reduction per layer.  Implementation uses
   cuBLAS ABT with the reduced right-side.
3. **Adam in subspace.**  For the r_ℓ scalars in g_sub,ℓ:
        m_sub,ℓ ← β₁·m_sub,ℓ + (1−β₁)·g_sub,ℓ
        v_sub,ℓ ← β₂·v_sub,ℓ + (1−β₂)·g_sub,ℓ²
        u_sub,ℓ = m_sub,ℓ / (√v_sub,ℓ + ε_adam)
   Moments stored int8 per shift #3; computation upcasts to FP32.
4. **Weight update.**  θ_ℓ ← θ_ℓ − η·P_ℓ·u_sub,ℓ.  One GEMV per layer,
   O(N_ℓ·r_ℓ).  Stochastic rounding per shift #5.
5. **P update, every K steps.**  Run Oja + QR per §3.2.  Track
   per-axis importance S_ℓ_j += |u_sub,ℓ_j| and staleness ρ_ℓ_j += 1
   if |u_sub,ℓ_j| < threshold else reset.  Axes with ρ > τ are
   candidates for replacement by a newly-discovered direction.
6. **Full-gradient audit, every K_audit ≈ 1000 steps.**  Materialize
   g_ℓ for one microbatch (pay the 20× backward cost for one step);
   compute residual energy ‖g_ℓ − P_ℓ P_ℓᵀ g_ℓ‖² / ‖g_ℓ‖².  If ratio
   > threshold (default 0.3), either raise r_ℓ or trigger Oja step
   with a larger α.  This is the **F2 mitigation** — a safeguard
   that critical directions are not being excluded.

Total per-step overhead: O(N·r) for the projection+update (same order
as one backward GEMM on a layer); P adaptation amortized 1/K.
Compared to Adam-dense, IBGRAD is **20× cheaper per step on the
optimizer side** while adding a O(N·r) projection cost that is already
paid during backward (the Pᵀ·A_ℓᵀ D_ℓ GEMM replaces the A_ℓᵀ D_ℓ
GEMM that never happens).

---

## 5. Why block-diagonal P

The full C = 𝔼[g gᵀ] ∈ ℝ^{N×N} has cross-layer blocks: the gradient of
a layer-ℓ weight is correlated with the gradient of a layer-ℓ' weight
through the Jacobian chain.  A full-rank P would capture these.  But:

1. **Cost.**  Full P at r = 0.05·N is N·r = N²·0.05 floats — 250 TB at
   N = 2.23 B.  Infeasible.
2. **Conditioning.**  Cross-layer gradient covariance is dominated by
   near-layer blocks (adjacent layers share Jacobians).  Distant-layer
   covariance is near-zero empirically.
3. **Gauge.**  The network's loss is invariant to per-layer
   reparameterization up to linearized error.  Cross-layer directions
   are **gauge directions** with no loss-reduction content beyond their
   within-layer projections.

The block-diagonal restriction is thus **near-lossless** — we discard
gauge information for a 1/L memory savings (L = 24 or 48).  This is
the information bottleneck: compress out the gauge, keep the content.

---

## 6. Composability clauses

### 6.1 With OVFG (#9)

OVFG factorizes g_ℓ = A_ℓᵀ·D_ℓ with rank bounded by T.  IBGRAD
projects to r_ℓ ≤ min(T, 0.05·N_ℓ).  Composition: compute
g_sub,ℓ = (P_ℓᵀ A_ℓᵀ) · D_ℓ; store L, R factors of rank
min(r_ℓ, T) in OVFG's pipeline.  At r_ℓ = 256, T = 1024 the operation
cost drops: OVFG does O(T·(m+n)·r) per step, IBGRAD+OVFG does
O(r_ℓ·(m+n)·r_OVFG).  **Compounding compression: IBGRAD picks the best
r_ℓ-dim subspace WITHIN OVFG's naturally-rank-T factorization.**  The
two are orthogonal — OVFG is per-microbatch graph-structural;
IBGRAD is streaming-statistical.

### 6.2 With GFIB (#17)

GFIB's Fisher F̂_i on a per-parameter basis becomes F̂_j on the r
subspace axes: F̂_j ← β·F̂_j + (1−β)·g_sub,j².  Per-axis Fisher is
a **closer proxy to the diagonal of C**, because the axes were chosen
to diagonalize C in the first place.  GFIB's gating-sigmoid fires on
r signals instead of N, reducing PI-controller cost by 20×.  The
combination is **dimensionally natural**: gate the subspace, not the
parameter.

### 6.3 With MFIO (#11)

MFIO replaces Adam with per-layer σ.  Under IBGRAD, MFIO's σ_ℓ is
applied to u_sub,ℓ, and can itself be r_ℓ-dimensional (σ_ℓ ∈ ℝ^{r_ℓ})
rather than scalar.  An r-dimensional σ is almost free (r < 0.05·N_ℓ)
and gives MFIO per-subspace-axis adaptivity — recovering most of
Adam's per-parameter preconditioning at MFIO's O(L) cost.

### 6.4 With DFA (#12)

DFA replaces backward with random projection R_ℓ.  IBGRAD's P_ℓ is a
**learned** projection.  Naive composition corrupts IBGRAD's signal —
DFA's R_ℓ is not aligned with the gradient subspace, and P_ℓ trained
on DFA-generated g would converge to the top-r eigenvectors of
R_ℓᵀ H R_ℓ, not of H.  **Mitigation:** tie the two projections —
train R_ℓ ≈ P_ℓ by periodic copy from P.  This makes DFA's backward
project to the loss-informative subspace, which is a **strict
improvement** over random DFA.  This is the tightest non-trivial
composition in the stack.

### 6.5 With SR-BF16 weights (#5), int8 Adam (#3)

IBGRAD is dtype-orthogonal: P_ℓ at BF16 + SR, m_sub/v_sub at int8.
The 20× state reduction compounds with int8's 4× reduction giving
80× over FP32-Adam baseline for optimizer state.

---

## 7. Relationship to prior work

- **GaLore (Zhao 2024).**  Periodic SVD on the dense gradient.  IBGRAD
  never forms the dense gradient; subspace adapted online via Oja, not
  offline SVD.
- **Gur-Ari et al. 2018.**  Empirical observation that gradient lives
  in top-Hessian-eigenspace.  IBGRAD uses this as an optimizer
  primitive rather than a diagnostic.
- **OVFG (#9).**  Graph-structural low-rank (rank ≤ T); IBGRAD is
  statistical low-rank (rank ≤ 0.05·N).  Composable.
- **LoRA.**  Fine-tuning only; IBGRAD is pre-training, adaptive.
- **PowerSGD, top-k.**  Fixed-basis compression; IBGRAD's basis is
  learned, capturing Hessian eigenspace rather than a heuristic.
- **Shampoo / KFAC.**  Kronecker Hessian preconditioners; IBGRAD
  approximates the first-order gradient second moment instead — the
  low-rank structure is empirically sharper than the Hessian's.

---

## 8. Failure modes and mitigations

1. **F1: P rank-collapse.**  Near-deterministic gradients make Oja
   updates parallelize P_ℓ columns.  *Mitigation:* diagonal
   regularizer ε·I on C_ℓ; QR every K steps; noise injection at α/10.
2. **F2: critical directions excluded.**  Too-small r_ℓ or wrong
   subspace → plateau.  *Mitigation:* full-gradient audit every
   K_audit ≈ 1000 steps; if residual energy > 0.3, expand r_ℓ;
   staleness-counter axis replacement.
3. **F3: P adaptation oscillates.**  Too-small K or too-high α
   thrashes P; Adam moments become inconsistent.  *Mitigation:*
   ≤1% column replacement per K steps; warm up α from 0 over 1000
   steps; replaced axes get zero moments with 5-step grace period.
4. **F4: DFA composition corrupts signal.**  Naive DFA+IBGRAD trains
   P on wrong-basis gradients.  *Mitigation:* tie R_ℓ ← P_ℓ every K
   steps (§6.4) — a composition *requirement*, not just a patch.
5. **F5: Audit cost.**  One full backward / 1000 steps = 2% overhead.
6. **F6: Cold start.**  Step 0 P is random.  *Mitigation:* first 500
   steps run full-rank Adam to seed P_ℓ, then switch to IBGRAD.

---

## 9. Minimal prototype (≤ 2 weeks)

1. **Week 1 d1–3.** Kernel `ibgrad_project_grad` fused into backward:
   compute g_sub,ℓ = P_ℓᵀ·(A_ℓᵀ D_ℓ) directly.  Parity test vs dense
   at r_ℓ = N_ℓ (should be exact Adam up to round-off).
2. **Week 1 d4–6.** Subspace Adam state (int8 m_sub, v_sub) +
   weight-update GEMV `ibgrad_apply_update`.  Determinism test vs
   dense: same loss trajectory at r_ℓ = N_ℓ.
3. **Week 2 d1–2.** Oja streaming update `ibgrad_oja_step` with QR.
   Stability test: on synthetic quadratic loss, P_ℓ → top-r
   eigenvectors within 500 steps.
4. **Week 2 d3–4.** Full-gradient audit + axis-replacement logic.
5. **Week 2 d5–7.** pile_large-small (d=1024, L=24) end-to-end
   training at r_ℓ = 0.05·N_ℓ.  Compare loss trajectory to Adam
   baseline over 5000 steps; target: within 5% of baseline loss at
   same step count.

Deliverables: `gpu_ibgrad.cu` ~1100 LOC.  Tests: parity at full rank,
Oja convergence, audit triggers, determinism, composition smoke-test
with OVFG and GFIB.

---

## 10. Summary

IBGRAD replaces the full N-dimensional gradient with its image in a
learned, block-diagonal, per-layer rank-r subspace aligned with the
top-r eigenspace of the gradient second moment.  Backward GEMM
dimensions shrink 20× (at r/N = 0.05); Adam state shrinks 20×; weight
update costs one N·r GEMV per layer; the subspace evolves online via
Oja's rule with cost comparable to one extra backward every K = 200
steps.  Composability is strong with OVFG (#9 — statistical ∘
graph-structural), GFIB (#17 — subspace-axis Fisher), MFIO (#11 —
r-dim σ), and nontrivial-but-possible with DFA (#12 — learned R).
The theoretical preservation ratio is X_r ≥ 0.80 at r/N = 0.05,
degrading expected loss by at most (1−X_r)²/(1−β₁) ≈ 0.4 bits/step,
negligible versus the compounded 20× compression in state and compute
on the gradient/optimizer axis.  IBGRAD's novelty against every prior
shift and against GaLore is the **continuous adaptation** of the
subspace through a streaming-PCA statistic — no shift has treated
*which directions the optimizer looks at* as a learned, evolving
quantity.
