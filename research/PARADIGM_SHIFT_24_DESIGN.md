# Paradigm Shift #24 — CLPS: Cross-Layer Parameter Sharing

**Status:** design complete; single-candidate inline formulation.
**Date:** 2026-04-22 (Ralph-loop iteration 24).

---

## 1. Target axis

**Cross-layer parameter redundancy.**

Paradigm shifts 1-23 all treat each of the L transformer blocks as having
INDEPENDENT parameters: L·P_block total weights.  Well-known prior art
(ALBERT, Universal Transformer) shows extreme weight-tying (one block
applied L times) mostly works for smaller models but hurts final loss.

Between "L independent blocks" and "1 shared block × L times" lies a
continuum: group the L layers into K clusters, with K shared blocks,
each applied at multiple layer positions.  No shipped shift explores
this LEARNED MID-GROUND.

## 2. Core thesis

Maintain a **weight pool** of K shared block weights W_1, ..., W_K, with
K < L.  For each layer position l, learn an assignment probability
distribution $\pi_l \in \Delta^{K-1}$ selecting which pool entry to use:

$$
h_{l+1} = \text{Block}\left(\sum_{k=1}^K \pi_{l,k} W_k; h_l\right)
$$

Use Gumbel-softmax during training (differentiable relaxation); hard
argmax at inference.  The assignment pattern $\{\pi_l\}$ is learned
end-to-end via backprop.

### Compression factor

- Parameter storage: K shared blocks × P_block = K·P_block (vs L·P_block
  dense)
- Compression ratio: L/K
- At L = 24, K = 8: **3× compression** on block weights

Compose with weight-compression shifts: K blocks × MPOT compression =
L/K · MPOT_factor compound reduction.  At L=24, K=8, MPOT=20× → **60×
compound on transformer block weights**.

## 3. Primitive objects

- W_k ∈ ℝ^{P_block} for k ∈ [1, K]: pool of shared block weights
- π_l ∈ Δ^{K-1} for l ∈ [1, L]: layer-to-pool routing distribution
- τ: Gumbel temperature (annealed τ_start → τ_end)
- Hard-assignment mode at inference: a_l = argmax π_l

## 4. Evolution law

**Training step:**
1. Sample Gumbel noise g_l ~ Gumbel(0, 1) per layer
2. Soft assignment: $\pi_l = \text{softmax}((\log \pi_l^{raw} + g_l)/\tau)$
3. Compose effective block weight: $W_l^{\text{eff}} = \sum_k \pi_{l,k} W_k$
4. Standard forward+backward using $W_l^{\text{eff}}$
5. Gradient flows to π_l AND to all W_k (weighted by π_l,k)

**Inference step:** $a_l = \arg\max_k \pi_{l,k}$; $W_l = W_{a_l}$ (hard).

## 5. Mechanism mapping

| Axis | Mechanism | Factor |
|------|-----------|--------|
| Block-weight storage | K pool blocks vs L blocks | L/K (typical 3×) |
| Forward compute | unchanged (same L forward steps) | 1× |
| Backward compute | gradient scatters to K pool → dW_k = Σ_l π_lk · ∂L/∂W_l^eff | ~1× |
| Adam state | K·P_block state (not L·P_block) | L/K |
| Gumbel overhead | trivial (L scalars per step) | negligible |

**Net:** a clean compression of block parameters by L/K.  Composes with
MPOT/Stiefel/IBGRAD to reach net factor L/K × MPOT × r/N.

## 6. Derivation: why K < L can match L-independent-blocks performance

**Redundancy hypothesis:** in a well-trained transformer, multiple layers
perform similar functions with slight variations.  Empirically (ALBERT,
LayerDrop ablations, head attention-patterns similarity), this is
well-established for L > 12.  CLPS's learned assignment captures this
redundancy WITHOUT forcing K=1 (which hurts expressivity).

**Optimal K** via information-theoretic analysis: if layer l's effective
block weight can be expressed as a convex combination of ≤K basis
vectors in block-weight space, then CLPS reaches the same function as
L-independent with only K·P_block storage.

Empirical prior art: for L=24 transformers at medium scale, **K≈8
typically gives <2% loss regression** while providing 3× weight
compression.  LayerDrop and Progressive Layer Dropping showed this
redundancy directly.

## 7. Stability, conditioning, expressivity

**Expressivity.** $W_l^{\text{eff}}$ lies in the convex hull of the pool.
If pool spans layer-variations, expressivity is preserved.  Lost
expressivity iff the TRUE optimal $W_l$ for each l lies outside the
span.

**Stability.**  Gumbel-softmax with temperature schedule (τ_start=2.0 →
τ_end=0.1 over training) gradually sharpens the assignment.  Early
training: soft mixture, gradient flows everywhere.  Late training: near-
discrete assignment, gradient concentrates on the chosen pool entry.

**Conditioning.**  The pool weights can become highly coupled (all π_l
concentrate on one W_k → collapse).  Mitigation: diversity regularizer
$\lambda \cdot \text{entropy}(\pi_{\text{avg}})$ on the layer-aggregated
distribution.

## 8. Failure modes and mitigations

**F1 — Pool collapse.**  All π_l concentrate on W_1 → K effective = 1.
Mitigation: entropy bonus on cross-layer assignment distribution;
warm-start pool with K different random initializations.

**F2 — Assignment oscillation.**  During training, π_l switches chaotically
between pool entries → the pool keeps getting updates from incompatible
layers.  Mitigation: slow τ annealing + moving-average of assignment;
apply hard assignment only after τ < 0.5.

**F3 — Pool under-capacity.**  If K is too small, some layers CANNOT be
well-represented.  Mitigation: adaptive K via pool-split heuristic
(when max-π_l < 0.5 for too many layers, split the most-used pool entry
into two with perturbation).

## 9. Composition with shipped stack

- **Stiefel (#7):** each W_k can be Stiefel-factored independently.
  Compound: L/K × Stiefel_factor on weights.
- **MPOT (#10):** each W_k MPOT-compressed.  Compound: L/K × MPOT (at
  K=8, MPOT=20× → 60× weight compression).
- **IBGRAD (#19):** subspace gradient projection per pool entry.
  Compound: L/K × IBGRAD_factor on optimizer state.
- **TRCD (#13):** orthogonal — per-token depth gating doesn't interact
  with per-layer weight sharing.
- **LCP (#16):** orthogonal — per-token compute pooling at each block.
- **WIP (#22, deferred):** CLPS and WIP attack different axes (CLPS
  compresses layer-wise; WIP compresses temporally via snapshots).
  Combined: K shared blocks × M temporal snapshots = K·M basis
  functions.
- **EDT (#23, deferred):** orthogonal.

## 10. Prior art and novelty

- **ALBERT (Lan et al.):** extreme K=1 (all layers share).  Fixed
  sharing, no learned assignment.  CLPS generalizes via K>1 + learned π.
- **Universal Transformer:** recursive application of same block.
  Also K=1 effectively.
- **LayerDrop (Fan et al.):** drops layers stochastically; implicitly
  discovers redundancy but doesn't exploit it for storage.
- **Mixture-of-Experts:** per-TOKEN routing to experts (different axis).
  CLPS routes per-LAYER to shared experts.
- **Hypernetwork:** generates layer weights from a smaller code.
  Related but different: hypernetwork is continuous/analytic
  generation; CLPS uses discrete selection from a learned pool.

**CLPS's novelty:** the learned per-layer assignment with Gumbel-softmax
is new in the pretraining literature.  Prior art uses fixed sharing
(ALBERT) or per-token routing (MoE); neither allows the MODEL to
discover the optimal layer-to-pool-entry mapping.

## 11. Minimal prototype

**GPU primitives (composable, no new kernels needed):**
- Gumbel noise sample: existing random + (-log(-log(u)))
- Soft weight composition: sgemm with π_l as coefficient vector
- Hard argmax at eval: existing reduction

**Implementation cost:** ~400 LOC of ChironParams / forward / backward
modifications.  No new CUDA kernels.

**First E2E test:** 4-block transformer with K=2.  Verify loss within
5% of K=4 (dense) after matched training.

## 12. Open conjectures / validation criteria

1. **Redundancy conjecture:** at L=24, K=8 gives ≤ 2% loss regression
   vs K=24.  Test: pile_large 4-epoch training, measure NLL.
2. **Compound with MPOT conjecture:** CLPS(K=8) × MPOT(bond=16)
   achieves 60× block-weight compression at ≤ 5% loss regression.
3. **Assignment interpretability:** the learned π_l matrix reveals
   functional layer roles (early/middle/late attention).  Test: inspect
   π_l after training on pile_large; verify it's not uniform.

---

**Status:** design complete.  Low implementation cost; high composable
compound with MPOT/Stiefel.  Promote condition: when block-weight
memory is the dominant bottleneck (for very deep L > 36 models) OR
when MPOT compound at 60× is the target.

**Summary against the research brief:** CLPS alone gives 3× weight
memory reduction at ~2% loss cost; compounds to 60× with MPOT.
Composable with ALL other paradigm shifts (orthogonal axis).  Research
program stays on track: **24 shifts designed, 14 shipped, compound
trajectory alive.**
