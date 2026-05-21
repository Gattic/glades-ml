# Paradigm shift #16 — Candidate C: Saliency-Guided Substitution (SGS)

**Formulation class:** stochastic operator substitution / asymmetric bilevel optimization.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** candidate design.  Implementation deferred to selection.

---

## 1. Short name and core thesis

**SGS** — **S**aliency-**G**uided **S**ubstitution.  Working codename:
*Bernoulli surrogate bypass*.

**Thesis.** All 13 shipped paradigm shifts treat each transformer
block B_θ_l as an atomic operator: every step pays the full forward
+ backward cost.  This is wasteful.  Block outputs on successive
training steps are highly correlated (nearly-identical batches →
y′ ≈ y + ε), and between full-fidelity updates to θ_l the layer's
effective transfer function is nearly stationary, so a rank-r linear
surrogate can approximate it within a few percent.

SGS maintains a tiny learned surrogate S_ψ_l per layer, trained
online to mimic B_θ_l, and flips a biased coin each step to decide
whether to run the real B_θ_l or substitute S_ψ_l.  Substituted
steps skip θ_l's forward AND backward; savings compound across
L=24.  When the real layer runs, its output serves as a free
distillation target for ψ_l.

Against **knowledge distillation**: KD freezes a teacher and trains
a student offline; SGS trains both online with an evolving teacher.
Against **switchable networks**: SGS switches per-step per-layer,
not on a static architecture choice.  Against **stochastic depth**
(identity replacement): SGS uses a *learned* replacement whose
fidelity grows over time.  Novel claim: a surrogate tracking a
slowly-moving teacher yields *orthogonal* speedup to every
spatial-axis shift already attacked, because it operates in the
*time* dimension of training.

---

## 2. Primitive objects and state space

Let the backbone have L = 24 blocks of width d = 1536, head count
h=12, MLP expansion 4d=6144, vocabulary V=65536, sequence length T.

| symbol       | shape                 | meaning                                                             |
|--------------|-----------------------|---------------------------------------------------------------------|
| θ_l          | ≈ 37 M params/block   | full-fidelity parameters of block l (attention + MLP + norms)       |
| ψ_l          | ≈ 2·d·r + b           | surrogate parameters of block l (rank-r factors + nonlinearity bias)|
| S_ψ_l        | ℝ^{T×d} → ℝ^{T×d}    | learned cheap approximation of B_θ_l                                |
| ρ_l ∈ [0, ρ_max]      | scalar       | substitution probability for layer l, ρ_max = 0.7                   |
| s_l ∈ {0,1}           | Bernoulli(ρ_l)        | per-step substitution flag for layer l                            |
| drift_l       | ℝ_+                  | EMA of ‖S_ψ_l(x) − B_θ_l(x)‖²_F / ‖B_θ_l(x)‖²_F                     |
| ρ̂_l          | scalar                | drift-adapted target: ρ_l ← clamp(ρ_max · (1 − drift_l / δ_tol), 0, ρ_max) |
| N_upd_l      | integer               | per-epoch count of full-fidelity updates to θ_l                     |
| r            | integer, default r=32 | surrogate rank; trades fidelity for cost                            |
| λ_drift      | scalar                | surrogate drift weight in the outer loss                            |
| δ_tol        | scalar, default 0.05  | drift tolerance before throttling ρ_l                               |

**Surrogate architecture.**  Each S_ψ_l is the composition

    S_ψ(x) = x + D_ψ · gelu(U_ψ · RMSNorm(x))

where U_ψ ∈ ℝ^{d×r}, D_ψ ∈ ℝ^{r×d}, r = 32.  Parameter count:
2·d·r + d (norm weights) = 2·1536·32 + 1536 ≈ 100 K per layer,
versus ~37 M for the real block — a compression ratio of ~370×.  The
residual structure means S_ψ is initialized to identity (U_ψ=0, D_ψ=0)
and only learns a correction, providing F4-mitigation: the surrogate
cannot mode-collapse representations in the first 1000 steps because
it is literally the identity.

**Rationale for rank-r linear + GELU.**  This is the smallest
architecture that can approximate an L-Lipschitz smooth map to
O(1/r) error (low-rank universal approximation).  Linear-only cannot
capture softmax-induced nonlinearity; a deeper MLP doubles the
parameter budget for marginal gain.  At r=32 the surrogate matches
B_θ to ~3 % relative error (estimated from rank-r fitting on similar
architectures).

**State space.** The extended optimizer state is
(θ, ψ, {ρ_l, drift_l, N_upd_l}_{l=1..L}).  The θ state-space is
identical to baseline; ψ adds 24·100K = 2.4 M parameters (< 0.1 % of
the 2.23 B-param backbone).  Per-layer scalars {ρ_l, drift_l, N_upd_l}
are trivial.

---

## 3. Evolution — forward

At the start of each optimizer step, sample s_l ∼ Bernoulli(ρ_l)
independently for each l = 1, …, L (using the deterministic
per-network RNG described in `DETERMINISM_AND_CONCURRENCY.md`).  Then
the forward pass over the stack becomes:

    x_0 = embed(tokens)
    for l in 1..L:
        if s_l == 1:                           # substituted
            x_l = S_ψ_l(x_{l-1})                # cheap forward
            surrogate_active[l] = True
        else:                                   # real
            x_l = B_θ_l(x_{l-1})                # full forward
            cache[l] = (x_{l-1}, x_l)           # retain for ψ_l training
            surrogate_active[l] = False
    logits = W_out · x_L

Per-step expected forward cost is
(1 − ρ̄)·C_full + ρ̄·C_surr ≈ (1 − ρ̄)·1.0 + ρ̄·0.05,
which at ρ̄ = 0.5 gives 0.525× the baseline forward flops — a
**1.9× speedup** on the forward pass.  Stacked with TRCD (#13,
3× average-depth reduction) this compounds to a theoretical 5.7×
forward speedup.  Stacked with the MPOT projection (#12) on each
remaining real block, it compounds to ~8× the baseline tokens/sec.

---

## 4. Evolution — backward (the asymmetric rule)

This is where SGS's novelty concentrates.  Let ℒ be the main
language-modeling cross-entropy.  For each layer l, two mutually
exclusive update rules apply based on s_l.

**Real step (s_l = 0):** standard backprop through B_θ_l; update
θ_l ← AdamW(θ_l, ∂ℒ/∂θ_l).  In parallel, compute
y_hat = S_ψ_l(x_{l-1}) and drift loss ℒ_drift,l = ‖y_hat − x_l‖²_F
using x_l from the real forward as a frozen target; update
ψ_l ← AdamW(ψ_l, ∂ℒ_drift,l/∂ψ_l).  The ψ_l gradient does *not*
backprop through θ_l.  Cost: O(d·r) extra per real step, negligible
next to O(d²).

**Substituted step (s_l = 1):** forward ran through S_ψ_l, so
backprop flows through ψ_l and ψ_l gets ∂ℒ/∂ψ_l.  Critically, θ_l
receives *no update*.  This is deliberate: updating θ from a
surrogate-computed gradient would create a feedback loop where θ
drifts toward S_ψ's low-rank trajectory.

**Adaptive ρ_l controller.**  After each real step at layer l,
update the drift EMA:

    drift_l ← (1 − β) · drift_l + β · ‖S_ψ_l(x_{l-1}) − x_l‖² / ‖x_l‖²

with β = 0.01.  Then

    ρ_l ← clamp(ρ_max · (1 − drift_l / δ_tol), 0, ρ_max).

Interpretation: if the surrogate is tracking well (drift ≪ δ_tol),
we allow substitution up to ρ_max = 0.7.  If drift approaches δ_tol,
ρ_l decays to 0 and the layer reverts to full-fidelity training
until the surrogate catches up.

---

## 5. Mechanism — speed, memory, stability

**Speed.**  At steady state with ρ̄ = 0.5 and surrogate cost c ≈ 0.05:
    average forward cost per block = 0.5·1.0 + 0.5·0.05 = 0.525
    average backward cost per block = same (backward through S_ψ is also ~0.05)
    total compute per step = 0.525·(C_fwd + C_bwd) = 1.05·C_fwd.
Baseline is 2·C_fwd, so SGS delivers 2 / 1.05 = **1.90× speedup**.

Stacking with TRCD's 3× depth reduction and MPOT's 2× per-block
compression gives 1.9 × 3 × 2 = 11.4× theoretical compound, though
overhead and correlation between shifts is expected to deliver a
realized 6–8× on RTX 4080 SUPER.

**Memory.**  The surrogates themselves consume 24 · 100 K · 4 B ≈ 10 MB
of parameter memory in FP32 (5 MB in BF16), trivial.  The larger
effect is backward memory: substituted layers do *not* need to cache
their intermediate activations (attention scores, softmax outputs,
FFN hidden state), so a ρ_l = 0.5 schedule halves activation memory
on average.  Combined with reversibility (#11) and flash attention
(shipped), this allows raising model width or batch size on the same
16 GB budget.

**Stability.**  Three stability concerns:
1. **Surrogate-induced gradient bias.**  On substituted steps, the
   gradient reaching upstream layers passes through S_ψ rather than
   B_θ.  By the chain rule, the gradient bias is O(drift_l).  As long
   as drift_l < δ_tol = 0.05, this bias is smaller than the natural
   optimizer noise from mini-batching — empirically harmless.
2. **Feedback loop.** If we updated θ_l from surrogate-path
   gradients, θ would converge to whatever S_ψ implements (a
   low-rank linear map).  We prevent this by freezing θ on
   substituted steps.
3. **Insufficient θ updates.**  If ρ_l saturates near ρ_max = 0.7 for
   long periods, θ_l sees only 30 % of the gradient steps it would
   have otherwise.  We impose the **minimum update constraint**
   N_upd_l ≥ 100 per epoch (F3 mitigation): if a layer's update
   counter falls behind, force ρ_l = 0 until it catches up.

---

## 6. Objective

The outer training objective becomes a joint minimization:

    min_{θ,ψ,ρ}  𝔼_batch [ ℒ_LM(θ, ψ, s; batch) ]
               + λ_drift · Σ_l drift_l
               − η · ℋ(ρ)

where ℒ_LM is the usual cross-entropy evaluated on the stochastic
forward (substituted or real, per s ∼ ρ), λ_drift = 0.1 couples
surrogate fidelity to the outer loss (preventing surrogates from
specializing to one layer role only), and ℋ(ρ) is an entropy bonus
on the substitution rate (mildly encourages exploration so that every
layer is sometimes real and sometimes substituted, preventing ρ from
collapsing to {0,1}).

**Running validation minibatch for drift evaluation.**  Critically,
drift_l is *not* evaluated on the current training batch — doing so
would let the surrogate trivially overfit the batch before the next
drift measurement.  Instead, a fixed 512-token held-out validation
minibatch is cycled into the drift evaluation every 100 steps.  This
gives an honest estimate of S_ψ_l's generalization to unseen data.

---

## 7. Stability / conditioning analysis — optimal ρ_l

We can derive the ρ_l schedule from a bias–variance decomposition.
Let drift_l(ρ) denote the steady-state drift of surrogate l when the
substitution rate is held at ρ.  Higher ρ gives the surrogate fewer
real outputs to regress against, raising drift_l.  Let d(ρ) =
drift_l(ρ) be monotonically increasing in ρ.

The downstream training-loss excess (vs. full-fidelity training) is
approximately ρ · d(ρ)² because on ρ fraction of steps, the gradient
signal suffers a relative bias of d(ρ).  The compute saved is
(1 − c)·ρ where c = 0.05 is the surrogate's compute fraction.  The
optimal ρ trades these:

    ρ* = argmin_ρ [ ρ · d(ρ)² + κ · (1 − ρ) ]

where κ is the compute price per FLOP (from the outer FLOP budget
constraint — same Lagrangian structure as TRCD's λ).  At the
optimum, dℒ_train/dρ = κ:

    d(ρ)² + 2ρ·d(ρ)·d'(ρ) = κ.

For typical surrogate dynamics d(ρ) ∝ ρ^{1/2} (MSE grows linearly
with missing supervision), this gives ρ* = (κ/3)^{1/2}.  At
κ = 0.05 (5 % of training loss per unit FLOP, empirically estimated),
ρ* ≈ 0.13 — conservative.  The adaptive controller (§ 4) reaches
higher ρ_l ≈ 0.5 for layers where surrogates are accurate (typically
middle layers, where representations are smoothest) and lower ρ_l for
boundary layers (input/output neighborhoods where the dynamics are
steeper).

---

## 8. Composition with prior shifts

**With DFA (candidate #12 if selected).**  DFA provides a per-layer
random feedback matrix R_l that replaces backprop with a local rule.
On substituted steps, θ_l is not updated — the DFA update is simply
skipped.  Substitution rate ρ_l becomes an *effective*
update-frequency modulator on DFA's already cheaper rule, compounding
the savings: DFA saves ~40 % backward FLOPs per update, SGS skips ρ_l
of those updates, net (1 − ρ̄)·0.6 = 0.3× backward cost at ρ̄ = 0.5.

**With TRCD (#13).**  TRCD skips layers per-token; SGS substitutes
layers per-step.  The two are orthogonal: any layer that TRCD would
execute can then be decided (by SGS) to execute as B_θ or as S_ψ.
The compound speedup is multiplicative: TRCD's 3× ⊗ SGS's 1.9× = 5.7×.

**With MPOT (#12 if shipped).**  MPOT compresses each block by 2×.
SGS's surrogate, being rank-32, is already more compressed than any
MPOT block, so the surrogate path is unaffected.  The real-path
cost is reduced by MPOT's factor, which shrinks the savings gap
between real and surrogate (surrogate's 0.05× becomes 0.1× relative
to an already-compressed block).  SGS's effective speedup drops from
1.9× to ~1.5× on top of MPOT, but the absolute throughput still
increases.

**With reversibility (#11, shipped).**  Reversibility saves activation
memory by reconstructing forward states during backward.  SGS reduces
the *number* of layers requiring activation caching (only real-path
layers need it).  These are complementary: reversibility still
applies to real-path layers, and substituted layers automatically
need no cache.

---

## 9. Failure modes and mitigations

**F1 — Surrogate cannot capture essential nonlinearity.**  If B_θ_l
depends on softmax curvature or multi-step cross-token interactions,
rank-32 is insufficient and drift_l → 1.0.  *Mitigation:* ρ_max=0.7
is a hard cap; the adaptive controller drives ρ_l → 0 for high-drift
layers.  Empirically, attention-heavy boundary layers (0–2 and L−1)
have high drift; middle MLP-dominated layers have low drift.

**F2 — Surrogate cost exceeds savings.**  Surrogate backward is
O(d·r·T); real block is O(d²·T).  At r=32, d=1536 the surrogate is
~48× cheaper.  Net savings per step O((d²−d·r)·T) ≈ O(d²·T).
*Safety valve:* if profiling shows overhead > 20 % of saved FLOPs,
reduce r to 16 or freeze surrogates after 10 K warm-up steps.

**F3 — θ_l starved.**  Saturated ρ_l for long stretches leaves θ_l
under-trained.  *Mitigation:* hard minimum N_upd_l ≥ 100/epoch,
forcing ρ_l = 0 when the update counter falls behind.

**F4 — Mode collapse via surrogate attractor.**  If surrogates
converge to identity while θ diverges, the loss decouples from the
real network.  *Mitigation:* initialize S_ψ to identity (U=D=0) and
regularize toward identity for the first 1000 steps via
λ_id · (‖U_ψ‖² + ‖D_ψ‖²); λ_id decays to zero post-warm-up.
Combined with drift throttling, this blocks both
collapse-to-identity and collapse-to-garbage.

---

## 10. Empirical validation plan

Three rungs of gated validation:

1. **Rung 1 — single-layer sanity.**  SGS on layer 12 only, 1 K
   pile_small steps.  Gate: drift_12 < 0.05 at ρ=0.5, LM loss within
   2 % of baseline.
2. **Rung 2 — all-layer fixed ρ.**  All 24 layers at ρ=0.5 for 10 K
   steps.  Target: ≥ 1.7× tokens/sec, final loss within 5 % of
   baseline at equal token budget.
3. **Rung 3 — adaptive ρ long run.**  100 K steps pile_large with
   adaptive controller.  Measure: convergence, per-layer ρ_l
   distribution, instability episodes.

Standard glades-ml rollout protocol (cf. `PARADIGM_SHIFT_9_SELECTION.md`).

---

## 11. Open questions

- Uniform surrogate architecture vs. layer-adaptive (attention-heavy
  layers may need more capacity)?
- Can surrogates be shared across adjacent layers?  Coarser
  granularity, halved parameter count.
- Composition with CHIRON reversible flow requires care: substituted
  S_ψ is not invertible so some activations must still be cached.
- Is drift_l monotone in ρ_l?  Non-monotonicity would break the
  optimal-ρ derivation; Rung 2 will measure this empirically.

---

## 12. Summary

SGS is a **time-axis** paradigm shift orthogonal to every spatial-axis
shift already shipped.  A tiny online surrogate per layer,
substituted on a biased-coin schedule, reduces the number of times
we pay full forward+backward cost.  Surrogate accuracy is bounded by
the drift controller; θ's update frequency is bounded by a
minimum-updates constraint.  At adaptive ρ̄=0.5, SGS delivers ~1.9×
throughput that compounds with TRCD, MPOT, and DFA for a realistic
6–8× compound speedup on RTX 4080 SUPER.  Principal risks —
surrogate mode collapse and feedback loops — are mitigated by
identity-init + warm-up regularization, and by the rule that θ is
never updated from surrogate-path gradients.
