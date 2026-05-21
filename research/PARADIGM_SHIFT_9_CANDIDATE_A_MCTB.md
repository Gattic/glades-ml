# Paradigm shift #9 — Candidate A: Manifold-Coherent Tangent Bundle (MCTB)

**Formulation class:** manifold-constrained / Riemannian.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** candidate — not selected for shift #9 implementation (see `PARADIGM_SHIFT_9_SELECTION.md`).

---

## Short name

**MCTB** — Manifold-Coherent Tangent Bundle.  Gradients and Adam moments
inhabit the tangent bundle TM of a learned Riemannian sub-manifold
M ⊂ R^D, never the ambient D-space.

## Primitive objects

Let D ≈ 2.23·10⁹ (total scalar parameters across L transformer blocks).
Shift 7 already places per-layer weights on a Stiefel-like quotient; we
build strictly above it.

- **Base manifold** M = ∏_ℓ M_ℓ where M_ℓ = St(n_ℓ, r_ℓ) × S^{r_ℓ}_{++}
  is Stiefel × positive-diagonals (the shift-7 factorization
  W_ℓ = U_ℓ Σ_ℓ V_ℓ^⊤, r_ℓ = ρ·min(n_ℓ)).
- **Horizontal coherence constraint C**: additionally demand that the
  *column span* of U_ℓ and the *row span* of V_ℓ align across depth via
  a block-cyclic relation U_{ℓ+1} = U_ℓ · Q_ℓ with Q_ℓ ∈ SO(r_ℓ).  This
  cuts M down to the "coherent Stiefel flag" M_coh.
- **Tangent bundle** TM_coh = {(x, ξ) : x ∈ M_coh, ξ ∈ T_x M_coh}.
  dim TM_coh = 2·dim M_coh.
- **Moving frame** E(x) = {e_1(x), …, e_d(x)}: a geodesically-transported
  orthonormal basis of T_x M_coh, with d = dim M_coh.
- **Reduced frame** Ê_k(x) ⊂ E(x): the first k ≪ d basis vectors,
  selected by *curvature-weighted PCA of the empirical gradient
  covariance* (learned, updated every τ steps).  This is the MCTB
  compression chart.
- **Metric** g_x: the Fisher information metric pulled back through the
  softmax output, further regularized by a Tikhonov εI in the frame
  coordinates — g̃(ξ,η) = ξ^⊤ (F̂_k + εI) η where F̂_k is the diagonal
  k-by-k Fisher estimate maintained online.
- **Connection ∇**: Levi–Civita connection of g̃ restricted to the
  reduced frame, approximated to first order by the Stiefel canonical
  connection (exact projection onto tangent, no second-fundamental-form
  term).

## State space

At step n the optimizer state is (x_n, m_n, v_n, E_n) with
- x_n ∈ M_coh  (the Stiefel × Σ factors + block rotations Q_ℓ),
- m_n, v_n ∈ R^k  (first/second Adam moments in reduced-frame coords,
  GLOBAL k, pooled across layers via the coherence constraint),
- E_n a lazy representation of the moving frame (stored as k
  Stiefel-tangent vectors per layer, re-orthonormalized every τ steps).

Crucially, **moments are not per-layer**; they are expressed in the
global reduced frame, which is why gradient + moment storage collapses
to O(k) rather than O(D).

## Evolution law

**Continuous-time:**
    ẋ(t) = −grad_g̃ L(x(t)) = −Π_{T_x M_coh} g̃⁻¹ dL(x(t))
where Π is the orthogonal projector onto T_x M_coh.

**Discrete-time** (Adam-on-manifold, rank-k):
1. Compute ambient gradient ∇L(x_n) *but never materialize it densely*
   — it is computed per-block and immediately contracted against Ê_k.
2. ĝ_n = Ê_k(x_n)^⊤ ∇L(x_n) ∈ R^k
3. m_{n+1} = β₁ m_n + (1−β₁) ĝ_n
4. v_{n+1} = β₂ v_n + (1−β₂) ĝ_n⊙ĝ_n
5. m̂ = m_{n+1}/(1−β₁^{n+1}), v̂ = v_{n+1}/(1−β₂^{n+1})
6. ξ_n = − η · m̂ / (√v̂ + ε) ∈ R^k  (reduced frame step)
7. Δx_n = Ê_k(x_n) · ξ_n (expand back to TM; implicit — we act per-block)
8. x_{n+1} = R_{x_n}(Δx_n) (retraction: Stiefel polar on U, V;
   exp on Σ; SO(r) geodesic on Q_ℓ)
9. **Frame transport**: Ê_k(x_{n+1}) ← parallel-transport(Ê_k(x_n), Δx_n)
   using τ(e) = e − x_{n+1} sym(x_{n+1}^⊤ e).

## Mechanism mapping

- **Gradient storage** — never stored as D floats.  Each transformer
  block produces its local backward activation; we contract it with the
  k tangent vectors of block ℓ *streaming*, yielding k scalars per
  block.  Global reduced gradient ĝ_n is the concatenation (or pooled
  mean under coherence Q_ℓ) — size k floats, not D.
- **Optimizer state** — m_n, v_n are length-k vectors.  Kept in FP32
  (k is tiny, no need to int8-quantize).
- **Update computation** — k divisions + per-block rank-r Stiefel
  arithmetic.  No dense D-sized intermediate ever appears.

## Objective / variational principle

MCTB solves min_{x∈M_coh} L(x) subject to the *gradient parsimony prior*:
the empirical distribution of ∇L over a mini-batch window must be
k-rank-explainable under g̃.  Formally:

    𝓛(x, λ) = L(x) + λ·‖(I − Ê_k Ê_k^⊤)·Cov̂(∇L)·(I − Ê_k Ê_k^⊤)‖_F²

The prior is *self-fulfilling*: Ê_k is re-fit to the top-k eigenvectors
of the gradient covariance, so the penalty stays small by construction.

## Memory accounting at D = 2.23 B

Baseline (paradigm-8): gradient 4.5 GB + moments 4.5 GB = **9.0 GB**.
Choose k = 2²² ≈ 4.2 M (≈ 0.19 % of D).
Per-layer rank r = ρ · min(n) with ρ = 0.25.

| Component                          | Count                                 | dtype | Size    |
|------------------------------------|---------------------------------------|-------|---------|
| Reduced gradient ĝ_n               | k                                     | FP32  | 16 MB   |
| m_n                                | k                                     | FP32  | 16 MB   |
| v_n                                | k                                     | FP32  | 16 MB   |
| Moving frame Ê_k (k vectors in TM) | k · Σ_ℓ r_ℓ(n_ℓ−r_ℓ) ≈ k·(D^{1/2})    | BF16  | ≈ 1.6 GB |
| Fisher diagonal F̂_k                | k                                     | BF16  | 8 MB    |
| Coherence rotations Q_ℓ             | L · r²                                | BF16  | ≈ 20 MB  |
| **Total**                          |                                       |       | **≈ 1.7 GB** |

**Compression on {grad, optimizer}: 9.0 → 1.7 GB ≈ 5.3×.**

If frame storage bites, int8-pack with per-row scales (shift-3 trick)
halves it to ≈ 0.8 GB, pushing compression to ≈ 9×.

## Composability with shifts 1–8

- **CHIRON (#1)**: fully compatible.  Per-block backward produces A_ℓ →
  immediate rank-k contraction → discard.  Activation memory unchanged.
- **TC-tiled attention (#2)**: no interaction.
- **Int8 Adam (#3)**: *replaced*.  MCTB's m, v are small enough
  (48 MB total) to keep in FP32.
- **BF16 grad accumulation (#4)**: subsumed.
- **SR BF16 weights (#5)**: orthogonal.
- **Local-window attention (#6)**: orthogonal.
- **Stiefel × Σ (#7)**: MCTB is a *strict super-structure* of Stiefel.
  If k = d (no frame reduction) and Q_ℓ = I (no coherence), MCTB
  degenerates to Stiefel × Σ with full tangent storage — recovering
  shift-7 exactly.
- **HRTC (#8)**: orthogonal — HRTC compresses the T axis of activations;
  MCTB compresses the parameter/gradient axis.

## Stability / convergence

**Claim (sketch).**  Under (A1) L is g̃-geodesically L-smooth on M_coh,
(A2) Ê_k captures ≥ 1−δ of the g̃-gradient energy on average, (A3)
stochastic noise is σ²-bounded, Adam-on-MCTB with step η ≤ 1/(L(1+δ))
satisfies
    E[‖grad_g̃ L(x_n)‖²] ≤ (L(x_0) − L*)/(η N) + O(η σ² + δ G²)

Under the Stiefel canonical connection the retraction is second-order,
so BF16 drift is O(u² + η²) per step.

## Minimal prototype (≤ 2 weeks)

1. Week 1: global Ê_k as concatenated per-layer Stiefel-tangent vectors;
   fit via randomized SVD of the gradient matrix over τ=64 accumulated
   mini-batch gradients (re-fit every 512 steps).
2. Week 1: global (k)-vectors m, v; rewrite optimizer step to do
   ĝ = Ê_k^⊤ ∇L per-block during backward, Adam update in k, then Ê_k ξ
   retraction per-block.
3. Week 2: parallel transport of Ê_k (O(k·r²) per step).
4. Week 2: coherence constraint Q_ℓ as SO(r) per-boundary rotation.
5. Validate on a 350 M model.

## Relationship to existing methods

- **Natural gradient / Fisher**: MCTB *restricts* natural gradient to a
  rank-k Fisher on a manifold sub-space.
- **KFAC**: MCTB *departs* from explicit Kronecker factoring (that is
  candidate C's lane).
- **Shampoo**: MCTB keeps only k diagonal entries (F̂_k).
- **GaLore**: MCTB *generalizes* GaLore: (i) weights live on M, not R^D;
  (ii) optimizer state lives in the same k-frame; (iii) adds parallel
  transport.
- **Muon**: MCTB's projection onto Stiefel tangent is structurally the
  k=d case of Muon's orthogonalization.

## Failure modes

1. **Tangent bundle rank collapse** — if δ grows during training
   (loss curvature rotates out of Ê_k), optimization stalls.
   *Mitigation*: monitor residual ‖(I − Ê_k Ê_k^⊤) ĝ‖²; trigger eager
   re-fit of Ê_k.
2. **Chart-transition inconsistency** — parallel-transport is only
   first-order accurate; after many frame re-fits, m, v may become
   stale.  *Mitigation*: project old (m, v) through the change-of-frame
   matrix Ê_k^{new⊤} Ê_k^{old}; reset only if cosine < 0.3.
3. **BF16 retraction drift** — polar decomposition in BF16 accumulates
   O(u) error per step.  *Mitigation*: periodic (every 10³ steps) full-
   FP32 re-Stiefelization.
4. **Coherence over-constraint** — Q_ℓ ties depth-wise, could limit
   expressive capacity at very large depth.  *Mitigation*: allow Q_ℓ to
   learn with a separate low LR, or free the constraint every 4-th block.
5. **Fisher estimator bias** — F̂_k diagonal can underestimate curvature
   when off-diagonal correlations are strong.  *Mitigation*: fall back
   to Shampoo-style per-block 2×2 preconditioning inside the k-frame.
