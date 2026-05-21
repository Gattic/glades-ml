# Paradigm Shift #40 — SAS: Stochastic Attention Skipping

**Date:** 2026-04-24 (Ralph-loop iter 164, post user redirect to novel algorithms)
**Status:** Fresh design on a genuinely unattacked mechanism.
**Target magnitude:** 3-5× end-to-end compute reduction (not hyperparameter tweak).

---

## 1. Motivation & prior-art distinction

Previous paradigms (FACE, SLC, RLG) saved compute via dimensional reduction:
- FACE: reduces Adam state size
- SLC: reduces T (sequence) dimension
- RLG: reduces L (depth) dimension during warmup

**Unattacked axis: STOCHASTIC COMPUTE SCHEDULING within a training step.**
At each step, deterministically compute only a random subset of layers' attention
(the expensive 70% of step compute). Layers not selected get zero gradient that
step. Over many steps, all layers get updated on average.

**Distinction from prior art:**
- Different from stochastic depth (Huang 2016): that's a REGULARIZER for inference;
  SAS targets training compute.
- Different from TRCD (#13 designed): TRCD is PER-TOKEN learned gating; SAS is
  PER-STEP uniform-random layer selection (no routing network).
- Different from TRCD/LCP conditional-compute paradigms: SAS has zero state,
  zero overhead, purely stochastic.
- Different from MoE: SAS operates within a single dense model, no experts.

## 2. Mechanism

Per training step:

1. Sample a binary mask `active ∈ {0, 1}^L` where `P(active[l] = 1) = α`.
2. For each layer l: if `active[l] = 0`, skip attention entirely (`p += 0`, i.e.,
   no shear update). Still run `reln` on q (cheap, maintains flow).
3. If `active[l] = 1`, run full attention as usual.
4. Backward: gradients for Wq/Wk/Wv/Wo are zero on skipped layers (no backward pass).
5. Adam step: weights on skipped layers receive no update.

Per-step attention compute: expected `α · L × attention_cost`. At α=0.5, expected
half the attention compute per step.

## 3. Convergence hypothesis

**Hypothesis H40:** For α ∈ [0.3, 0.7], the model trained with SAS matches the
target-loss of α=1.0 (full attention) in **1/α × steps** but each step is 1/α
cheaper, so total wall-clock is COMPARABLE. The win is that in the α<1 regime,
ATTENTION SCALE reduces per step → can run at higher effective batch size or
lower VRAM.

**Stronger hypothesis H40a:** SAS is NOT just compute-time-equivalent. The
stochastic skipping acts as implicit regularization (analogous to dropout),
producing better generalization at equivalent compute. This would give a TRUE
speedup, not just a tradeoff.

H40a's validity is the Gate-0 question.

## 4. Expected speedup (projected)

At 1.84B with α=0.5:
- Attention cost reduces to 0.5 × 70% of step = 35%
- Other cost: 30% of step
- New step cost: 65% of old
- **Per-step speedup: 1.54×**

Stacked with existing flagship (SLC + RLG + FACE):
- Current ceiling: 3.36× wall-clock
- With SAS α=0.5: 3.36 × 1.54 = **5.18× projected**

If H40a holds (convergence benefit from regularization), effective speedup
to target loss could exceed 5×. Reaches "magnitudes faster" territory.

## 5. Implementation (minimal)

1. Add `--sas-alpha X` CLI flag (default 0.0 = disabled).
2. In forward loop, for each layer l:
   ```
   if cfg.sasAlpha > 0 and rng.uniform() > cfg.sasAlpha:
       continue  // skip attention for this layer this step
   ```
3. Track which layers were active this step (for backward to skip them).
4. No new CUDA kernels — pure scheduling logic.

Implementation size: ~50 lines of trainer code. Trivial vs FACE/SLC.

## 6. Gate-0 probe

**Experiment:** 66M × 2500 with α ∈ {1.0, 0.7, 0.5, 0.3}, fixed seed.
Compare final EMA at equal wall-clock (not equal step count).

**Accept:**
- α=0.5 reaches EMA within 0.1 nat of α=1.0 at 0.65× wall-clock → PASS, 1.54×
- α=0.3 reaches EMA within 0.2 nat of α=1.0 at 0.4× wall-clock → PASS, 2.5×
- Better yet: α=0.5 BEATS α=1.0 on final EMA → PASS (H40a confirmed)

**Reject:** α=0.5 loses >0.3 nat at equal wall-clock → skipping disrupts
gradient flow too severely.

Probe cost: ~3 minutes (4 × 60s runs at 66M).

## 7. Composition with shipped stack

Orthogonal to FACE (embedding), SLC (T), RLG (L, initial):
- FACE unchanged: embedding Adam state compression
- SLC: T curriculum, can still run
- RLG: L curriculum, can still run
- **SAS: within-step per-layer skipping**

RLG's "identity layer insertion" and SAS's "per-step skip" are compatible.
At early SLC phase with L=8 initial, SAS α=0.5 means 4 layers active per step.
After RLG grows to L=53, SAS α=0.5 means 26 layers active per step.

## 8. Failure modes

1. **All layers skipped probability = (1-α)^L.** At L=53, α=0.5: (0.5)^53 ≈ 10^-16.
   Essentially never.
2. **Gradient staleness** if some layers receive gradient only every 1/α steps.
   At α=0.3, each layer averages 750 updates in 2500 steps. Reasonable.
3. **Adam state divergence** — m_t, v_t for inactive layers age. For FACE-adjacent
   state, this should be OK since FACE's β=0.99 covers ~70-step lookback.
4. **Layer correlation breaks** — if layer l depends on layer l-1's recent update,
   skipping breaks that. Mitigation: ensure at least K=3 adjacent layers active.

## 9. Research novelty

SAS is genuinely NEW in the pretraining context:
- Stochastic depth exists but was studied as regularization, not compute optimizer
- MoE has the "expert selection" but is a full architectural change
- Mixture-of-depth (Google Deepmind 2024) routes per-token but keeps all compute available

**SAS specifically targets training-compute efficiency via zero-overhead layer
stochasticity**, which is an unattacked design point.

## 10. Phase 1 implementation plan

1. Add `--sas-alpha X` flag to chiron_train (~50 lines)
2. Modify forward/backward loops to check active mask
3. Gate-0 probe at 66M × 2500 (4 runs: α=1.0, 0.7, 0.5, 0.3)
4. If Gate-0 passes: 500M × 2500 validation
5. If 500M passes: 1.84B × 2500 flagship integration
6. Stack with FACE + SLC + RLG for full measurement

## 11. Follow-on paradigms (if SAS succeeds)

- **#41 Adaptive SAS**: learn per-layer α based on gradient saliency
- **#42 SAS + knowledge distillation**: use previous-step full-attention output
  as teacher signal for current step's partial attention
- **#43 Block-stochastic SAS**: skip blocks of 2-3 adjacent layers together
  (preserves local attention correlations)

## 12. Summary

Paradigm #40 SAS is a fundamentally new algorithm, not a hyperparameter tweak.
Its premise (per-step random layer skipping as training optimization) hasn't
been tested in the pretraining compute-reduction context. If H40a holds (SAS
as implicit regularizer), true magnitude-level speedup is achievable.

**Implementation cost:** ~1 hour engineering.
**Validation cost:** ~1 hour compute across 4 scales.
**Potential payoff:** 1.5-2× additional speedup stacked with existing flagship,
pushing the total compound to 5-6× wall-clock at ceiling.

If SAS fails, research program shifts to more speculative directions
(linear attention, state-space hybrids) which require deeper architectural changes.
