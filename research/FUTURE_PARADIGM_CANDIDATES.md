# Future Paradigm-Shift Candidate Space

**Date:** 2026-04-23 (post-FACE validation, Ralph-loop iteration 83).
**Purpose:** Identify promising unattacked axes for paradigm shifts #29+.
**Scoring axes:** memory impact × convergence impact × implementation risk × novelty.

---

## Review: what's been attacked

Taxonomy of 28 paradigm shifts designed to date:

| Category | Shipped shifts | Representative |
|----------|---------------|----------------|
| Activation memory | #1 CHIRON, #8 HRTC | reversible flow, token compression |
| Attention compute | #2, #6 local-window, flash | TC-tiled, fixed windows |
| Optimizer state (memory) | #3, #4, #5, #11 MFIO, #19 IBGRAD, #22 WIP | low-precision, low-rank |
| Optimizer state (convergence) | **#28 FACE** | Zipfian-frequency regularizer |
| Weight shape | #7 Stiefel, #10 MPOT | manifold / tensor-net factoring |
| Gradient shape | #9 OVFG, #25 GEC | low-rank factoring, T-dim compression |
| Per-token depth | #13 TRCD | Gumbel routing |
| Per-token compute | #16 LCP | LSH clustering |
| Cross-step forward | #26 ATC-Δ | Taylor-expansion cache |
| FFN intermediate | #27 CSP | JL-sketch hidden |
| Backward alternative | #12 DFA | fixed random feedback |
| Curriculum | #23 EDT (deferred) | energy weighting |

Key finding: **only FACE is convergence-improving**.  Memory shifts
compose; but for the Ralph-loop brief ("magnitudes faster"), new
convergence-axis shifts are the highest-value target.

---

## Candidate axes (prioritized by estimated research payoff)

### #29 candidate: VOCAB — learned vocabulary pruning (convergence × memory)

**Observation**: under Zipfian token distributions, bottom-50% of the
vocabulary contributes <5% of gradient signal but 50% of the embedding
matrix's rows.  These rare rows get almost no updates yet occupy full
dense Adam state (or FACE EMAs).

**Mechanism**: identify "cold" vocab entries (|f̂_i| below a threshold)
and freeze them.  At inference, frozen rows serve their random-init
embedding.  At training, skip the row entirely — no forward lookup, no
gradient contribution, no optimizer state.

**Claim**: 2× vocab (V=32k → 16k active) would halve embedding forward
compute and halve FACE state, with negligible convergence cost (since
rare tokens contribute little).

**Implementation horizon**: Phase 1 single primitive
`vocab_prune_step(f_hat, threshold, active_mask)` — 30-line CUDA kernel.

**Novelty**: similar to "dead neuron" detection in sparse nets, but
specifically for language-modeling vocabularies under Zipf.  Distinct
from vocabulary-factorization methods which keep dense state.

**Risk**: false-positive pruning of legitimate rare tokens.  Mitigation:
un-freeze threshold has hysteresis.

**Score**: memory 2×, convergence +0.1 nat (secondary), risk low, novelty high.

---

### #30 candidate: TRAJ — trajectory-predictive Adam (convergence)

**Observation**: Adam's m, v are EMAs over steps.  Given observed m_t,
v_t at steps 1..t, the values at step t+1 are PREDICTABLE via a short
autoregressive model.  Why compute m, v freshly at each step when they
barely change?

**Mechanism**: train a tiny AR(2) predictor per parameter group that
predicts m_t, v_t from m_{t-k..t-1}, v_{t-k..t-1}.  Every N steps,
refresh the predictor.  Between refreshes, use predicted m̂, v̂ in the
Adam update.  Saves compute on the m, v state maintenance (factor ~2x)
AND provides noise-reduction via implicit smoothing.

**Claim**: 2× backward-pass state-maintenance speedup, 0.1-0.3 nat
convergence advantage from the smoothing.

**Implementation horizon**: gpu_traj.{h,cu} with 2 primitives
(predictor_fit, predictor_apply) + trainer wire-in.  Moderate.

**Novelty**: TPW (#15, deferred) predicts WEIGHTS.  TRAJ predicts ADAM
STATE.  Related but distinct.

**Risk**: predictor accuracy at long horizons; compound with periodic
refresh.

**Score**: convergence +0.2 nat, implementation medium, novelty medium-high.

---

### #31 candidate: BSHIFT — batched stochastic shuffle preconditioner

**Observation**: batch-composition noise (surprise #13, #14) causes
training loss to oscillate within a 0.2-1.1 nat band.  This is the
per-step irreducible variance.  But if we compute gradients on K
different shuffles of the SAME batch and average, the variance drops
by √K.

**Mechanism**: instead of gradient accumulation over K different batches,
accumulate over K random shuffles of the SAME batch's tokens.  Reduces
positional-distribution noise without increasing data requirements.

**Claim**: loss variance drops √K at the cost of K× forward/backward
compute.  For K=4: 2× variance reduction → faster convergence, 4×
slower per step.  Net: wall-clock neutral, loss trajectory much
smoother → better early-training dynamics.

**Implementation horizon**: token-shuffle kernel + K-round gradient accum.
Low complexity.

**Novelty**: gradient accumulation with token-shuffle rather than
different-batch.  Not in standard literature.

**Risk**: may overfit to single-batch content; not a true variance
reducer but a sampling-space smoother.

**Score**: memory neutral, wall-clock neutral, convergence +0.1 nat
(conjectured), low novelty.

---

### #32 — NESR (Langevin-Adam): EMPIRICALLY REJECTED (2026-04-23)

Shipped as Phase 1 in iteration 85; long-horizon tested in iteration 86
at 66M × 5000 steps.  RESULT: NESR hurts convergence when composed
with FACE at every horizon checkpoint:

  Step   FACE only    FACE + NESR    Penalty
  500     9.263        9.336         +0.07
  2000    8.631        8.925         +0.29
  3000    8.421        8.888         +0.47
  5000    7.755        8.127         +0.37 nat

The Langevin "escape shallow minima" hypothesis does NOT validate at
these scales.  FACE's implicit Zipfian regularization appears
sufficient; additional Gaussian noise adds pure variance.

**Status**: deferred candidate.  DO NOT promote without substantial
reformulation.  Potential reformulations (not yet explored):
- Noise ONLY on parameters with low |gradient| (stuck params)
- Warmup-only noise (first 100-500 steps) to escape poor init
- Noise on ATTENTION only (where FACE doesn't reach)

### #32 candidate: NESR — noise-equilibrium stochastic resonance (ORIGINAL DESIGN — superseded by the REJECTED finding above)

**Observation**: small amounts of noise in the weight update can help
escape shallow local minima.  Adam's implicit noise (via stochastic
mini-batches) does this.  But the noise magnitude is uncontrolled —
sometimes too much, sometimes too little.

**Mechanism**: inject controlled Gaussian noise to the weight update:
θ ← θ − lr · Adam(g) + √(2·lr·T·σ²) · N(0, I)
with T (temperature) scheduled to decay across training.  This is
Langevin-style stochastic resonance.

**Claim**: early-training: high T helps escape minima; late-training:
low T gives fine-tuning.  Estimated: +0.2-0.5 nat late-training loss
improvement.

**Implementation horizon**: one-line modification to adam_update kernel
(add noise sample).  Trivial.

**Novelty**: Langevin + Adam hybrid.  Known in literature (SGLD, etc.)
but not widely deployed in LLM training.

**Risk**: LR schedule tuning required; may hurt final loss if T too high.

**Score**: memory neutral, compute +1% (noise generation), convergence
+0.2-0.5 nat (conjectured), medium novelty.

---

### #33 candidate: RAND — random-feature attention

**Observation**: attention is O(T²).  Linear attention (Performer, etc.)
replaces softmax with random features but loses expressivity.

**Mechanism**: use FIXED random Fourier features for keys/queries +
Gumbel mask for sparsity.  Combined KV cache is small; attention
becomes O(T·r) with r ≈ 64 features.

**Claim**: 10× forward-pass attention speedup, 0.1-0.3 nat loss
degradation (manageable at T ≤ 2048).

**Implementation horizon**: rewrite of attention forward + backward.
High complexity.

**Novelty**: combines Performer with TRCD-style gating.  Original.

**Risk**: attention quality degrades sharply at long context; trade-off
curve unclear.

**Score**: memory 10×, compute 10×, convergence −0.1-0.3 nat, high complexity.

---

### #34 candidate: ZEN — zero-overhead embedding normalization

**Observation**: FACE achieves 1008× memory compression and 0.81 nat
convergence gain.  Can we push further with MORE complex preconditioners
at still-negligible cost?

**Mechanism**: extend FACE's dn̄ (column) EMA to rank-r (not just 1).
Maintain r EMAs per column, updating each conditionally on active-row
frequency quantile.  Per-column "multi-timescale" preconditioner.

**Claim**: 0.2-0.5 nat additional convergence gain over FACE; state grows
to 2V + r·m + 2 ≈ 260 KB (at r=4, m=512).  No meaningful throughput cost.

**Implementation horizon**: extend gpu_face.cu with multi-rank EMAs.  Low
complexity (mostly extends existing kernels).

**Novelty**: builds directly on FACE; natural extension.

**Risk**: r > 1 may not capture more structure than FACE's single EMA.
Empirical test required.

**Score**: memory neutral, compute neutral, convergence +0.2-0.5 nat
(conjectured), high implementability.

---

## Prioritization

For the next iteration's paradigm shift design:

1. **#32 NESR** (Langevin-Adam): 1-line kernel change, highest ratio of
   impact to effort.  Can be validated same-iteration as the design.

2. **#29 VOCAB** (vocabulary pruning): delivers on both axes (memory +
   convergence), low risk, high novelty.  Natural continuation of FACE.

3. **#34 ZEN** (multi-rank FACE): extends the validated FACE mechanism.
   High probability of success; modest gain.

4. **#30 TRAJ** (trajectory-predictive Adam): novel, but requires careful
   predictor design.  Higher effort.

5. **#31 BSHIFT** (batched-shuffle preconditioner): sampling-space
   smoother; novelty moderate but risk of overfitting.

6. **#33 RAND** (random-feature attention): known tradeoff; defer until
   flash attention memory becomes a bottleneck.

### Recommended next iteration (2026-04-23 update, post-NESR)
With #32 NESR empirically rejected, prioritize one of:
- **#29 VOCAB** — vocabulary pruning (bounded implementation, unique axis)
- **#34 ZEN** — multi-rank FACE (directly extends the validated winner)
- **#30 TRAJ** — trajectory-predictive Adam (requires more design work)

ZEN likely has the highest impact-per-iteration ratio — it extends an
already-validated mechanism, so the prior probability of success is
high.  TRAJ is the most novel but riskiest.

### Recommended next iteration (ORIGINAL 2026-04-23, pre-NESR)
Design and ship **paradigm shift #32 NESR** (Langevin-Adam):
- 1-line kernel modification
- Empirically validatable in a single iteration (500-step run)
- Minimal risk of divergence
- Tests whether controlled noise helps beyond FACE's implicit regularization
- **RESULT: rejected (see §32 entry above)**

---

## Methodology note

Each candidate above should pass the three-gate validation from
`RALPH_LOOP_METHODOLOGY_LESSONS.md`:
1. Primitive parity test (≤ 1e-4 error vs host)
2. Trainer integration smoke (≥ 100 steps, no divergence)
3. Long-horizon multi-scale validation (≥ 2500 steps, ≥ 2 scales)

Especially important for convergence-claim shifts (FACE's experience):
short-horizon benchmarks mislead.  Expect initial 500-step results to
be NOISY — trust 2500+ step EMAs.

---

## Summary

Five promising paradigm-shift candidates (#29 VOCAB, #30 TRAJ, #31 BSHIFT,
#32 NESR, #34 ZEN) have been identified for the post-FACE research
direction.  Each targets either memory, convergence, or both; each
fits within one Ralph-loop iteration of design + Phase 1 implementation.

The research program's current strength is on the optimizer axis (4
shifts in trainer, 1 with convergence improvement).  The next axis to
mature is **forward-pass compute** (ATC-Δ Phase 3, CSP Phase 3 both
pending trainer wire-in).  Paradigm shifts #32 and #34 keep momentum on
the optimizer axis; #33 would target the forward axis once ATC-Δ is
shipped.

Paradigm-design count after listing these: **34** candidates (14
shipped/wired, 14 designed, 6 candidate-space-only).
