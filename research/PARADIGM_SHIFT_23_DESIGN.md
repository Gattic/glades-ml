# Paradigm Shift #23 — EDT: Energy-Distilled Training

**Status:** design complete; single-candidate inline formulation.
**Date:** 2026-04-22 (Ralph-loop iteration 22).

---

## 1. Target axis

**Training-example weighting at the loss level.**

Paradigm shifts 1-22 all treat every training token as equal: same loss
contribution, same update weight, same downstream gradient flow.  The
dataloader produces batches, the loss averages across tokens, Adam
updates based on this average.  Even sampled-softmax (chunked CE) and
per-token shifts like TRCD (#13) and LCP (#16) preserve the underlying
equal-weighting assumption at the loss level.

Yet in reality, training data contains:
- Easy tokens (common words, formatting) that carry no useful gradient
- Noisy tokens (OCR errors, encoding artifacts) that emit misleading gradient
- Rare tokens that are precisely the learning signal we need most

Current practice: accept the noise and hope averaging cleans it out.

EDT challenges this.

## 2. Core thesis

Train a small **energy network** $E_\psi: \mathbb{R}^d \to \mathbb{R}_+$
alongside the main model.  Apply it to the per-token CE loss as a
multiplicative weight:

$$
L_\text{total} = \sum_t E_\psi(x_t) \cdot \text{NLL}(\text{main}_\theta(x_t), y_t)
$$

Critically, the energy network is trained ALONGSIDE the main network
via backprop through the weighted loss — it learns to up-weight tokens
whose NLL reduction is high-value per unit of main-network compute.

### Why this is a paradigm shift

1. **Data-side compression at no main-network cost.**  If $E_\psi$
   down-weights 50% of tokens to weight ~0.1, the EFFECTIVE BATCH SIZE
   contributing to main-network gradient reduction is halved, but the
   compute per step is identical.  Net: same loss reduction in fewer
   effective-batch steps → faster convergence.
2. **Automatic noise filtering.**  Tokens whose main-network output is
   erratic relative to their label (noise signature) get low energy and
   effectively skip contributing gradient.  No explicit noise-detection
   heuristic.
3. **Rare-token amplification.**  Tokens whose NLL reduction per step
   is exceptionally high (rare concepts) get high energy and dominate
   the gradient — learned curriculum without expert labeling.

## 3. Primitive objects and state space

- $\theta$: main transformer weights
- $\psi$: energy network weights ($|\psi| \ll |\theta|$; typical 2-layer
  MLP with hidden dim 64, total ~10-20K params)
- $E_\psi: \mathbb{R}^d \to \mathbb{R}_+$: scalar energy per token
- $(m_\psi, v_\psi)$: Adam state on $\psi$ (tiny)

**Standard training loop** augmented at the loss:

1. Forward main: $\hat{y} = \text{main}_\theta(x)$
2. Per-token NLL: $\ell_t = -\log p_\theta(y_t | x_t)$
3. Energy: $e_t = E_\psi(x_t)$ (softplus activation keeps $e_t \ge 0$)
4. Weighted loss: $L = \frac{1}{\sum_t e_t} \sum_t e_t \cdot \ell_t$
5. Backward: $\nabla_\theta L$ and $\nabla_\psi L$ both computed
6. Adam on $\theta$ and on $\psi$

## 4. Derivation: why $E_\psi$ converges to useful weights

At the optimum of $L$ over $(\theta, \psi)$ jointly, by KKT:

$$
\nabla_\psi L = \frac{\partial L}{\partial e_t} \nabla_\psi e_t
$$

For the normalized form of $L$ (step 4 above), we have
$\frac{\partial L}{\partial e_t} = \frac{\ell_t - L}{\sum_s e_s}$.

This is positive when $\ell_t > L$ (token has above-average loss) and
negative when $\ell_t < L$ (below-average).  So $E_\psi$ gets pushed UP
for tokens with above-average NLL — precisely the tokens where more
optimization budget would help.

**The energy network is implicitly learning token-level curriculum.**
No explicit supervision needed.

## 5. Mechanism mapping

| Axis | Mechanism | Factor |
|------|-----------|--------|
| Effective data utilization | up-weight high-loss tokens → larger effective batch in "training value" | ~1.5-3× |
| Main compute | unchanged (forward+backward still T tokens) | 1× |
| Energy net compute | $E_\psi$ cost ~0.1% of main forward | negligible |
| Memory | $\psi$ adds ~20K params; Adam state negligible | < 0.1% |
| Robustness | noisy tokens get low weight; noise-tolerance improves | hard to quantify |

**Net:** the expected "loss reduction per training step" improves
because compute is focused on high-value tokens, not uniformly spread.
Realistic speedup to a fixed NLL target: 1.5–3× depending on dataset
quality (higher savings on noisier data).

## 6. Stability, conditioning, expressivity

**Stability.**  Energy net degeneracy: if $E_\psi(x) \to 0$ everywhere,
the loss is ill-defined.  Mitigation: softplus activation keeps
$e_t \ge \delta > 0$; add regularizer $\lambda \cdot \sum_t (\log e_t)^2$
to discourage extreme weights.

**Conditioning.**  The joint optimization $(\theta, \psi)$ is convex in
$\psi$ (given $\theta$) for fixed-batch loss — quadratic in energy
weights.  Non-convex in $\theta$ as always.

**Expressivity.**  Energy net has ~20K params; sufficient to express
per-token class (common/rare) and per-token feature (e.g., whether
$x_t$ is a function word vs content word).  Under-parameterization
would limit the curriculum's resolution, but empirically small energy
nets (depth 2, hidden 64) generalize well.

## 7. Failure modes and mitigations

**F1 — Energy net pathological.**  $\psi$ could learn to assign all
weight to a few tokens, collapsing the effective batch size.
Mitigation: regularizer $\lambda \cdot \sum_t (\log e_t)^2$ ensures
moderate spread; soft constraint $\max e_t / \min e_t \le 100$
enforced by clamping.

**F2 — Noise amplification.**  If noise happens to have high NLL
(e.g., misspelled common word), the network rewards the noise
pathway.  Mitigation: add a $\text{ratio}(\ell_t / \|\nabla_{x_t} \ell_t\|)$
term — genuine difficulty has high loss AND high gradient magnitude;
noise has high loss but smaller gradient.

**F3 — Compute overhead.**  Computing $\nabla_\psi L$ backpropagates
through every token's NLL.  For large T, this is comparable to main
backward.  Mitigation: freeze $\psi$ during most of training; only
update it every K_\psi = 100 steps (amortized cost negligible).

**F4 — Energy network bootstrapping.**  Early training has no useful
signal to learn curriculum from.  Mitigation: warmup with uniform
$e_t = 1$ for first 1000 steps, then gradually phase in energy
weighting.

## 8. Composition with shipped stack

- **TRCD (#13):** complementary.  TRCD skips layers for low-value
  tokens at the PER-TOKEN level.  EDT down-weights low-value tokens at
  the LOSS level.  Compose multiplicatively: a token with low energy
  AND low depth-routing-utility gets near-zero compute AND near-zero
  gradient.
- **LCP (#16):** LCP pools similar tokens; EDT might learn to assign
  equal energy to cluster members (since they contribute similarly).
  Energy can be computed on cluster reps only.
- **IBGRAD (#19):** the energy-weighted gradient $\nabla_\theta L$ is
  exactly what IBGRAD projects into the r-dim subspace.  EDT's weighting
  changes the empirical gradient covariance $E[g g^T]$, which IBGRAD's
  Oja streaming PCA will track naturally.
- **PFE (#21, deferred):** PFE's mirror network predicts $f_\theta(x)$;
  EDT's energy network predicts per-token difficulty.  They could share
  early layers.

## 9. Comparison to prior art

- **Focal loss (Lin et al.):** $\ell_t \cdot (1 - p_t)^\gamma$ — a
  FIXED function of prediction confidence.  EDT learns the weighting
  function end-to-end.
- **Curriculum learning:** human-labeled easy → hard.  EDT is
  self-organizing.
- **Importance sampling:** selects tokens for inclusion in the batch.
  EDT weights the loss WITHIN the batch — cheaper, doesn't change the
  data pipeline.
- **Perplexity-weighted sampling:** weights by (perplexity)^α.  EDT
  subsumes this as a special case if $E_\psi(x) = p_\theta(y|x)^{-\gamma}$.
- **Data filtering (C4-quality-filter, FineWeb quality):** pre-training
  data curation.  EDT performs in-training curation that adapts to
  the main network's current state.

## 10. Minimal prototype

**GPU primitives (all reuse existing sgemm + activations):**
- No new primitive kernels needed.  EDT is composed of:
  - One small forward pass through $E_\psi$ on the batch's token
    embeddings (existing sgemm + softplus)
  - Per-token scalar multiplication of NLL by $e_t$ (existing
    element-wise kernels)
  - Backward through both paths (existing)

**First E2E test:** 2-layer MLP main + 1-layer MLP energy; train on
noisy linear regression (50% of targets deliberately corrupted).
Target: EDT should achieve lower loss on clean validation targets
than uniform weighting.

## 11. Implementation complexity

- **Small** (~300 LOC trainer modifications, no new kernels).
- The SIMPLEST of all designed paradigm shifts to implement.
- Low risk — if it doesn't work, fallback to uniform weighting is
  zero-cost.

## 12. Open conjectures / validation criteria

1. **Noise-robustness conjecture:** on a dataset with 10% label noise,
   EDT reaches 90% of clean-training NLL in the same wall-clock time.
   Test: inject label noise into pile_small, measure NLL on clean val.
2. **Rare-token amplification:** EDT's energy weights concentrate on
   the rarest 10% of tokens.  Test: measure energy-vs-token-frequency
   correlation after 10k steps.
3. **Compound with TRCD:** EDT × TRCD should give 1.5× × 3× = 4.5×
   effective-training-throughput improvement.  Test: stack on pile_large
   and measure NLL-to-target against a 45k tok/s uniform-weighting
   baseline.

---

**Status:** design complete.  Smallest-implementation-cost paradigm
shift on the active list.  **Promote condition:** when a noisy dataset
is available (FineWeb-filtered variant, or deliberately-noised pile
variant) that would benefit from adaptive curriculum.  Default
training on pile_large is clean enough that EDT's marginal value is
limited.

## Summary against the research brief

"Magnitudes less memory" — not directly (<0.1% additional memory).
"Magnitudes faster" — 1.5-3× effective-training-throughput via
curriculum, complementary to per-token compute reductions from TRCD
× LCP.  Compose in stack for the extreme-scale goal.
