# Paradigm Shift #22 — WIP: Weight Interpolation Pretraining

**Status:** design complete; single-candidate formulation (inline design to
preserve context budget; no 3-subagent parallel dispatch this iteration).
Axis and mechanism chosen from the unattacked-axis analysis in
`PARADIGM_SHIFT_19_AXIS_NOTES.md`.

**Date:** 2026-04-22 (Ralph-loop iteration 19).

---

## 1. Target axis

**Effective optimization dimensionality.**

Every paradigm shift 1-21 (shipped + deferred) accepts the assumption that
the optimizer is working in the FULL |θ|-dimensional parameter space.
Stiefel (#7), MPOT (#10) reduce |θ| by low-rank STORAGE but the OPTIMIZER
still explores the full space when computing updates (it sees the gradient
via backward and the optimizer state is per-parameter).

IBGRAD (#19) is the closest — it shrinks the SUBSPACE IN WHICH GRADIENTS
ARE COMPUTED via a learned projection.  Still, at each step the update
moves θ anywhere in ℝ^{|θ|}.

WIP challenges this at the root: **the optimization variable is not θ
but a K-dimensional interpolation coefficient.**

## 2. Core thesis

Maintain a pool of K "weight snapshots" `{W_k}_{k=1..K}` taken from past
training trajectories.  At each step, construct the active weights as:

$$
\theta(t) = \sum_{k=1}^{K} \alpha_k(t) \cdot W_k
$$

where $\alpha \in \mathbb{R}^K$ satisfies $\sum_k \alpha_k = 1$ and
$\alpha_k \ge 0$ (simplex constraint).  Train only $\alpha$ (K floats)
with SGD / Adam.  Keep all W_k frozen once recorded.

Every ~N_refresh steps, promote the current best θ(t) into the pool as a
new W_{K+1}, and discard an old (least-useful) snapshot to keep |pool| = K.

### Why this is a paradigm shift

1. **Optimization dimensionality from |θ| to K.** At K = 16, the
   effective DOF is 16 regardless of model size.  Adam state is 16
   floats per group (instead of |θ| floats).
2. **Forward compute unchanged.**  The product θ(t) = Σα·W_k materializes
   once per step (Σ over K matmuls = K×|θ| work — amortizable across
   many forward passes within a step).
3. **Backward compute proportional to K, not |θ|.**  The gradient with
   respect to α_k is `g_α_k = ⟨W_k, ∇_θ L⟩` — a single inner product
   per k.  No per-parameter gradient is ever STORED.  |θ|-dim vector
   ∇_θ L is computed during backward but immediately reduced to K
   scalars via inner products.

## 3. State space and evolution law

**Primitive objects:**
- $W_k \in \mathbb{R}^{|\theta|}$ for $k = 1..K$ (frozen snapshots)
- $\alpha \in \Delta^{K-1}$ (simplex)
- $(m_\alpha, v_\alpha) \in \mathbb{R}^K$ (Adam state on α)
- Refresh counter $N_\text{step}$

**State:** $(\{W_k\}, \alpha, m_\alpha, v_\alpha)$.

**Evolution:**

For each step $t$:

1. Compute $\theta(t) = \sum_k \alpha_k W_k$.  (K × |θ| memory reads
   producing one |θ| vector.)
2. Standard forward + backward to get $\nabla_\theta L$.
3. Compute $g_\alpha = [\langle W_k, \nabla_\theta L \rangle]_{k=1..K}$
   via K inner products.  (This is O(K·|θ|) but reduces to a K-vector.)
4. Adam on α with $g_\alpha$, then project back onto the simplex
   (softmax or Bregman projection).
5. Every $N_\text{refresh}$ steps: promote current $\theta(t)$ as a new
   $W_{K+1}$, drop one (e.g., the lowest-|α| snapshot), reset α to the
   new uniform.

## 4. Mechanism mapping

| Axis | Mechanism | Factor |
|------|-----------|--------|
| Optimizer state | m_α, v_α ∈ ℝ^K instead of ℝ^{\|θ\|} | \|θ\|/K (= 10⁸/16 ≈ 10⁷×) |
| Backward compute | K inner products vs |θ|-dim Adam update | |θ|/(K·|θ|) = 1/K (but |θ|-dim backward still happens) |
| Gradient memory | K scalars (g_α) vs |θ| floats (∇_θ L) | |θ|/K after reduction |
| Forward compute | K×|θ| matmul to materialize θ(t) | **1× overhead per step** |
| Snapshot memory | K × |θ| floats (stored in FP16 or int8) | K× (K=16, 16× more weight memory) |

**Net:** dramatic reduction in optimizer state and gradient storage; at
the cost of K× weight storage.  Compounds favorably with weight-
compression shifts (#7 Stiefel, #10 MPOT): if W_k are each MPOT-compressed,
snapshot memory is K / MPOT_factor × original → could be net-negative
if MPOT gives 20× and K=16.

## 5. Derivation of the K-dimensional Adam gradient

Let $L(\theta)$ be the loss.  Define $\theta(\alpha) = \sum_k \alpha_k W_k$.
Then

$$
\frac{\partial L(\theta(\alpha))}{\partial \alpha_k}
= \nabla_\theta L \cdot W_k
= \langle W_k, \nabla_\theta L \rangle_{\mathbb{R}^{|\theta|}}
$$

This is a single dot product per k.  On GPU, K dot products of |θ|-dim
vectors = one $|θ| \times K$ GEMM.  Cost: O(|θ|·K) which is **K× more
work than a single Adam step** that would otherwise have required
|θ|-many scalar Adam updates.  So per-step cost of the WIP update
arithmetic is O(K·|θ|) for gradient computation plus O(K) for Adam
update, vs O(|θ|) for standard Adam.  K×|θ| > |θ| — net K× slowdown
on the Adam step itself.

**But:** the Adam step is typically 5-10% of a training step
(forward+backward dominate).  K × 5% = 50% slowdown on arithmetic is
overshadowed by the memory + bandwidth savings.

## 6. Stability, conditioning, expressivity

**Expressivity.**  θ(α) lies in the convex hull of the pool.  If the
pool spans a useful direction in parameter space, the optimization can
reach it.  Over training, N_refresh snapshots are promoted, gradually
expanding the explored region.  **Conjecture:** the span of K ≈
log₂(T_steps) snapshots covers the high-curvature subspace of the loss
landscape (similar to Nesterov momentum memory).

**Conditioning.**  The Hessian over α is
$H_\alpha = [\langle W_j, H_\theta W_k \rangle]_{j,k}$, a K×K matrix.
Adam's preconditioner handles K-dim conditioning easily (only 16 eigenvalues
to track).

**Simplex constraint.**  Bregman projection (softmax-exp / log-sum-exp
trick) preserves the constraint exactly and is differentiable.  No
barrier or penalty needed.

## 7. Failure modes and mitigations

**F1 — Snapshot pool staleness.**  Very old W_k becomes irrelevant as
training advances.  Mitigation: weighted-pool refresh with age-decay
eviction policy.

**F2 — Early training has no past snapshots.**  For the first
N_refresh steps, WIP can't work.  Mitigation: bootstrap with K
different initializations (random seeds), treat these as the initial
pool.

**F3 — Low α on a snapshot causes gradient staleness.**  If α_k ≈ 0
for a snapshot, its gradient-with-respect-to-α is uninformative.
Mitigation: Adam's m_α / v_α handles this (low-α snapshots naturally
get down-weighted).  Also: promote a snapshot only if its ⟨W_k, g⟩ is
consistently non-zero.

**F4 — Pool diversity collapses.**  If all snapshots are close to each
other, their span is low-dimensional.  Mitigation: diversity-aware
eviction (when evicting, prefer the snapshot closest to the centroid,
preserving the hull).

## 8. Composition with shipped stack

- **CHIRON (#1):** unchanged; W_k has no activation-level role.
- **MPOT (#10):** each W_k can be MPOT-compressed independently; pool
  memory goes from K·|θ| to K·|θ|/MPOT_factor.
- **MFIO (#11):** the K-dim α-update can itself be MFIO-compressed;
  synergistic.
- **TRCD / LCP (#13/#16):** orthogonal; W_k's are used wherever θ
  would have been.
- **IBGRAD (#19):** the GRADIENT subspace (IBGRAD) and the PARAMETER
  subspace (WIP) are duals — IBGRAD constrains where updates come
  from, WIP constrains where θ can go.  **Their composition is a
  DOUBLE-SUBSPACE optimization: K-dim α updated via r-dim gradient
  subspace.**  Effective DOF = min(K, r).

## 9. Comparison to prior art

- **Stochastic Weight Averaging (SWA):** averages past W's with FIXED
  weights.  WIP trains the weights.  Strict generalization.
- **Model soups:** post-hoc averaging of multiple independently trained
  models.  Not a pretraining paradigm.
- **NNCF (Dellibovi et al.):** K-dim interpolation but on low-rank
  adapters only (fine-tuning).  WIP applies during full pretraining.
- **Population-based training (DeepMind):** similar snapshot-pool idea
  but uses evolutionary selection, not gradient descent on α.
- **Hyperparameter nested loops:** PBT explores hyperparameters;
  WIP explores the weight interpolation coefficient itself.

## 10. Minimal prototype

**GPU primitives:**

```cpp
ibgrad_wip_materialize(W_pool, alpha, K, N, theta_out)  // θ = Σ α_k W_k
ibgrad_wip_grad_alpha(W_pool, grad_theta, K, N, g_alpha_out)  // g_α = ⟨W_k, g_θ⟩
ibgrad_wip_simplex_project(alpha, K)  // Bregman projection onto Δ^{K-1}
ibgrad_wip_promote_snapshot(W_pool, theta_new, evict_idx, K, N)  // in-place pool update
```

**Adam step: existing adam_update called on a K-dim vector.**

**First E2E test:** 2-layer MLP, K=8, |θ|=128, N_refresh=50 steps.
Target: match dense-Adam loss to within 20% after 300 steps.

## 11. Full research program

Phase 1: primitives + parity tests
Phase 2: E2E convergence test on MLP
Phase 3: composition with IBGRAD (WIP-on-α + IBGRAD-on-θ dual-subspace)
Phase 4: snapshot-pool management (eviction policy empirical study)
Phase 5: chiron_train wire-in — train 100M-param model via WIP with
K=16 snapshots stored in MPOT-compressed format (net ~1× weight
memory vs dense, but O(K) optimizer state).

## 12. Open conjectures / validation criteria

1. **Log-coverage conjecture:** K = log₂(T_steps) snapshots suffice to
   span the loss-reducing subspace.  Test at T=10k steps with K=14.
2. **Dual-subspace compound:** WIP × IBGRAD effective DOF is min(K, r).
   Test with K=16, r=16 on a K≠r model: at K=16, r=8, DOF should
   be 8 (IBGRAD dominates); at K=8, r=16, DOF should be 8 (WIP
   dominates).
3. **MPOT compound at memory neutrality:** snapshot pool at K=20,
   MPOT factor 20× → net weight memory equal to dense.  Training
   should work despite K× snapshot overhead.

---

**Status:** design complete.  Decision on Phase 1 implementation
pending empirical-priority review vs other deferred shifts (#17 GFIB,
#18 SGS, #20 PRX, #21 PFE).

**Promote condition:** when a strong compound with MPOT is needed
(snapshot memory becomes tractable) OR when optimizer state bandwidth
becomes the dominant bottleneck (at even larger than 2.23B-param
models).
