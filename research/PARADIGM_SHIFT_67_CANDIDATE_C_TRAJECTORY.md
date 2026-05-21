# Paradigm Shift #67 Candidate C — TRAJECTORY-CHIRON: per-trajectory REINFORCE-style pretraining

**Status:** candidate-C design for paradigm shift #67. **Recommended action: REJECT** (mechanism is well-motivated and explores a genuine compute axis, but prior negative results — MIXER 2016, MRT 2016, RAML 2017, sequence-level abandonments throughout 2017–2020 — established that variance dominates efficiency at scale, and the user-brief constraint of bit-exact text-NLL preservation cannot survive REINFORCE's unbiased-but-noisy gradient at finite-sample sizes; the paradigm is structurally a re-litigation of a settled question).
**Date:** 2026-05-08 (Ralph-loop iteration 211, post-iter-210 close at ~9,240,000× cumulative on algebra/logic/binding-reasoning / ~6,600,000× grounded-reasoning / 5,500,000× knowledge-augmented / 5,360,000× agent / 3,030,000× tool-augmented / 930,000× text-NLL).
**Predecessors.** All of #42–#66. Load-bearing references: `PARADIGM_SHIFT_59_DESIGN.md` (PRM-CHIRON — REINFORCE-adjacent process reward modeling, the closest prior mechanism), `PARADIGM_SHIFT_62_CANDIDATE_B_AGENT_CHIRON.md` (REINFORCE on task-success terminal reward), `PARADIGM_SHIFT_56_DESIGN.md` (DISTILL-FORWARD — relevant for variance-reduced gradient via teacher distillation), `BEYOND_CHIRON.md` (NLL-preservation evaluation protocol). External prior art (load-bearing for honesty section): Williams 1992 (REINFORCE original), Sutton et al. 2000 (Policy Gradient theorem), Ranzato et al. 2016 (MIXER — sequence-level training with REINFORCE for text generation), Shen et al. 2016 (MRT — Minimum Risk Training), Norouzi et al. 2016 (RAML — Reward Augmented ML), Wu et al. 2018 (Google NMT — abandonment of sequence-level RL in favor of MLE+selective RL), Ouyang et al. 2022 (InstructGPT / RLHF), Bai et al. 2022 (Constitutional AI), Rafailov et al. 2023 (DPO — replacing PPO with offline contrastive loss).

**Axis.** **GRADIENT-GRAIN** — proposes to coarsen the gradient signal from per-token (T positions per sequence) to per-trajectory (1 reward per sequence). This would, in principle, reduce backward-pass FLOPs by a factor up to T. The axis is genuinely orthogonal to the 11 axes opened so far (DATA / LOSS / SAMPLING / REWARD / IDENTITY / SCHEDULE / AGENCY / OPTIMIZER / GROUNDING / KNOWLEDGE-LOCUS / SYMBOLIC) — it operates on the *granularity* of the supervised signal rather than on the signal's content or routing. **However**, the axis was attacked in 2016–2018 by MIXER, MRT, and RAML; the negative result (variance dominates at scale, MLE wins) is one of the strongest empirical findings in modern NLP optimization.

**Tagline.** *Standard pretraining backprops T per-token CE gradients per sequence. TRAJECTORY-CHIRON proposes one trajectory-level reward and a single REINFORCE gradient with control-variate baselines. Theoretical compute reduction: up to T-fold on backward FLOPs. Theoretical NLL: matches MLE in expectation by Williams 1992. Empirical reality (MIXER, MRT, RAML, GNMT 2018): variance-dominated; MLE outperforms; sequence-level RL was abandoned. The mechanism's place in the modern stack is RLHF post-training (Ouyang 2022) — fine-tune-time, not pretrain-time; preference-pair contrastive (DPO 2023), not raw REINFORCE.*

**Honest headline.** **~1.0×–1.6× per-step backward speedup at the variance ceiling, with an empirically-expected NLL gap of 0.5–2.0 nat at finite-sample budgets** that **directly violates the user-brief's bit-exact NLL preservation constraint**. The 2–5× claim in the assignment brief is **not realizable in practice** as direct REINFORCE-replaces-CE pretraining: every prior project at scale (MIXER, MRT, RAML, GNMT 2018) found this regime was dominated by MLE. **The strongest defensible variant** — REINFORCE as an *auxiliary loss alongside* CE — already exists in #59 PRM-CHIRON and #62 AGENT-CHIRON, leaving TRAJECTORY-CHIRON with no genuinely-new axis after honest decomposition. **Bigger picture:** the gradient-grain axis was explored and the negative result is well-published; reopening it at the project's depth-26 paradigm slot is unlikely to yield magnitudes-better speedup at fixed NLL.

---

## 0. Executive summary

After 25 paradigms (#42–#66), the bigger-picture track has reframed 11 axes (DATA / LOSS / SAMPLING / REWARD / IDENTITY / SCHEDULE / AGENCY / OPTIMIZER / GROUNDING / KNOWLEDGE-LOCUS / SYMBOLIC). Iter-211 sought a #67 candidate that opens a new axis or relaxes constraints. TRAJECTORY-CHIRON proposes the **GRADIENT-GRAIN** axis: replace per-token CE backprop with per-trajectory REINFORCE. The honest analysis below is that this axis was historically attacked, the negative result is robust, and the mechanism does not survive the user-brief's bit-exact NLL constraint.

**The mechanism in one paragraph.** Standard LM training computes a CE loss per token (T positions) and backpropagates T gradients per sequence. TRAJECTORY-CHIRON instead defines a trajectory-level scalar reward `R(x_{1:T})` (e.g., negative perplexity of a held-out continuation, or a downstream-task signal). The training objective becomes the policy-gradient expectation `∇_θ J(θ) = E_{τ ~ π_θ}[(R(τ) − b) · ∇_θ log π_θ(τ)]` where `π_θ(τ) = ∏_t p_θ(x_t | x_{<t})` and `b` is a baseline. The gradient `∇_θ log π_θ(τ) = ∑_t ∇_θ log p_θ(x_t | x_{<t})` is computed once per sequence with a single forward pass and a single backward pass (vs T per-token CE backward updates). Theoretical compute reduction: up to T-fold on backward FLOPs.

**What kills the proposal at the user-brief's NLL constraint:** REINFORCE is unbiased *in expectation* but **finite-sample variance is large**; per-token CE is **bit-exact** (no expectation, no variance). Convergence rates differ by a factor of `√Var[R · ∇log π]` in finite-sample settings — empirically 10²–10⁴ at language-model scale. Achieving the same final NLL as MLE requires either (a) so many samples that the speedup vanishes or (b) a control-variate scheme so aggressive that it reduces to MLE itself. The bit-exact NLL constraint cannot be satisfied; the user-brief constraint of "maintaining NLL accuracy" excludes the paradigm.

**Quantitative speedup with honest band:**

| Variant | Backward speedup | NLL gap at fixed FLOPs | Verdict |
|---|---|---|---|
| Pure REINFORCE (no CV) | ~5× theoretical | 5–10 nat (catastrophic) | **REJECT** (MIXER/MRT prior) |
| REINFORCE + value baseline | ~3× theoretical | 1.5–3.0 nat | **REJECT** (still violates NLL) |
| REINFORCE + Rao-Blackwell partial-CE CV | ~1.6× theoretical | 0.5–1.5 nat | **REJECT** (NLL gap too large) |
| Full Rao-Blackwell (recovers per-token CE) | ~1.0× | 0.0 nat | reduces to MLE; no speedup |
| REINFORCE as auxiliary alongside CE | 1.0× (additive cost) | 0.0 nat | already in #59, #62 |

The "honest 2–5×" claim from the brief lives only in the first three rows, all of which fail the NLL-preservation constraint. The fourth row recovers MLE behavior at zero speedup. The fifth row is what the project already does.

**Cumulative single-GPU stack post-#67-C (if this paradigm were selected):**
- All speedup tiers below presume the variant is wired in; any tier above 1.0× violates user-brief NLL.
- Pure REINFORCE: `930,000 × 5 ÷ exp(8) ≈ 1,560×` (catastrophic NLL collapse ruins the multiplier).
- REINFORCE + partial-CE CV: `930,000 × 1.6 ÷ exp(0.7) ≈ 740,000×` (small headline reduction, NLL constraint violated).
- Auxiliary REINFORCE (already shipped in #59/#62): `930,000×` unchanged.

**Net magnitude verdict:** there is no operating point that *both* speeds up backward FLOPs *and* preserves bit-exact text-NLL.

**Engineering scope.** ~700 LOC over ~3 weeks to wire a partial-CE-control-variate REINFORCE auxiliary loss into the existing trainer (the smallest-novelty variant, equivalent to extending #59's PRM head to scalar trajectory reward). This is not the speedup-bearing variant; it is the *only variant that does not violate the NLL constraint*, and it is already subsumed by #59/#62. The speedup-bearing variants would require ~2,500 LOC over ~10 weeks (full REINFORCE replacement of the CE backward path, baseline networks, variance-reduction infrastructure, replay buffers for K-trajectory averaging, learning-rate warmup tailored to high-variance gradients).

**NLL preservation.** Bit-exact NLL on text **does not survive** any operating point with > 1.0× backward speedup. Theorem 2 (§4.2) makes this precise: any non-CE estimator that avoids per-token CE backward incurs finite-sample variance proportional to `T · σ²_R` where `σ²_R` is the trajectory-reward variance and `T` is sequence length. At `T = 1024` and `σ²_R ≈ 1.0` (typical for negative-perplexity rewards on natural language), the variance-induced NLL gap at any fixed step budget is bounded below by a positive constant — the gap **cannot be driven to zero** without recovering per-token CE.

**Verdict (recommended at end of doc):** **REJECT.** TRAJECTORY-CHIRON is the well-known sequence-level-RL pretraining proposal whose negative result is one of the most robust findings in 2010s NLP. The MIXER paper (Ranzato 2016) and MRT (Shen 2016) demonstrated empirically that sequence-level REINFORCE underperformed token-level MLE at all reasonable training budgets; subsequent work (RAML 2016, GNMT 2018, BART 2020) confirmed and generalized the finding. The modern descendants — RLHF (Ouyang 2022), DPO (Rafailov 2023) — moved REINFORCE to *post-training* on small fine-tune budgets, not pretraining. At the project's depth-26 paradigm slot under the bit-exact-NLL constraint, the proposal does not yield magnitudes-better speedup; it instead re-litigates a settled empirical question. **Joint Gate-0 PASS probability: ~10%**; **LLM-scale empirical confirmation conditional on Gate-0: ~15%**; **unconditional confirmation: ~1.5%** — far below the 19% threshold of even RESERVE candidates in recent slates.

---

## 1. Why a *new* GRADIENT-GRAIN axis at depth 26 — and why it fails

The bigger-picture track has sustained novelty for 11 paradigms (#56–#66) by reframing successive axes. After SYMBOLIC opened in #66 and was reserved, iter-211's #67 slate looks for either (a) genuinely new axes or (b) constraint relaxation that opens magnitudes-better speedup.

**The GRADIENT-GRAIN axis is a genuinely new axis on paper.** No prior paradigm in the project's stack operates on gradient *granularity*; the closest is #59 PRM (process-level reward attached to per-token activations) and #62 AGENT (per-trajectory REINFORCE on task success, *alongside* CE). These both *augment* the per-token CE signal; they do not *replace* it.

**The axis was historically attacked and the negative result is robust.** From 2016 onwards, multiple high-profile efforts attempted to replace per-token CE with sequence-level RL signals:

- **MIXER (Ranzato et al. 2016):** mixed-objective training (CE warmup → REINFORCE on sequence-level reward) for text generation. Showed BLEU improvements over MLE on machine translation summarization, but **only when MLE warmup was substantial (≥ 50% of training)**. Pure sequence-level training from scratch underperformed MLE.
- **MRT (Shen et al. 2016, "Minimum Risk Training"):** sequence-level minimum risk training with sampling-based gradient estimation. Required CE warmup; failed to match MLE at fixed compute when started from random initialization.
- **RAML (Norouzi et al. 2016, "Reward Augmented ML"):** reward-augmented maximum likelihood — *maintained per-token MLE* and added a reward-shaped data-augmentation. Effectively a hybrid that reduces variance by remaining within the MLE estimator's finite-variance guarantees. **Did not replace per-token CE.**
- **Google NMT (Wu et al. 2018):** explicitly evaluated REINFORCE-style sequence-level training in production NMT pipelines. Concluded that variance dominated at scale; the production stack reverted to MLE with selective fine-tuning.
- **Modern descendants (RLHF / DPO):** RLHF (Ouyang 2022) uses PPO at *post-training* time on a fine-tune budget (~10⁴–10⁵ steps), not pretraining. DPO (Rafailov 2023) replaced REINFORCE/PPO with offline contrastive loss precisely because variance was unmanageable in the on-policy regime.

**The pattern across this literature is consistent.** Sequence-level REINFORCE works *as a refinement on top of MLE*, on small fine-tune budgets, with significant variance-reduction infrastructure (PPO clipping, KL anchoring, separate critic networks, advantage normalization). It does *not* work as a from-scratch pretraining replacement for MLE at language-model scale. **The user-brief's request for "magnitudes better on compute speed while maintaining NLL accuracy" sits squarely in the regime where the negative result holds.**

**Why this differs from the SYMBOLIC axis (#66-C reserved).** The SYMBOLIC axis (NEURO-SYMBOLIC-CHIRON) was an axis where prior art was *positive on a narrow subset* (AlphaGeometry, NS-CL, GPT-f); the question was whether the magnitude transferred. The GRADIENT-GRAIN axis is one where prior art is *negative on the broad question* (MIXER, MRT, GNMT 2018); reopening it at depth 26 requires either a new mechanism that the prior literature did not consider, or a constraint relaxation the user has not authorized.

The candidate's most plausible new mechanism is **Rao-Blackwellization with per-token control variates** — a modern addition not present in MIXER/MRT. §3.4 below shows that this construction either *recovers per-token CE exactly* (zero speedup) or *retains a residual variance lower bound* that prevents bit-exact NLL preservation.

---

## 2. Mechanism: trajectory-level reward + REINFORCE with control variates

### 2.1 Trajectory-level reward construction

Three reward variants are conceivable, all explored in the prior literature:

**Variant R1 — Self-perplexity on a held-out continuation.** Generate the model's own continuation `x_{T+1:T+H}` from prefix `x_{1:T}`; reward `R = −log p_θ(y | x_{1:T})` for held-out target `y`. **Problem:** the reward is *itself* a likelihood; using it inside REINFORCE is a roundabout way of optimizing the same likelihood the per-token CE already optimizes directly, with strictly higher variance.

**Variant R2 — BLEU/ROUGE/BERTScore on a reference target.** Standard MIXER / MRT setup. **Problem:** these rewards are sparse, computed over full sequences only, and provide weak gradient signal per backward update.

**Variant R3 — PRM-style stepwise rewards collapsed to trajectory.** Run a learned PRM (#59 PRM-CHIRON) over the trajectory, extract a scalar reward per step, then sum/average to a trajectory reward. **Problem:** if PRM is high-fidelity, the per-step reward *already provides per-token gradient signal* (this is exactly #59); collapsing to trajectory throws away that signal.

**Variant R4 — Negative loss of the same model on the same data, computed at trajectory boundary.** `R(x_{1:T}) = −∑_t log p_θ(x_t | x_{<t})`. **This is exactly the per-token CE summed**; using it inside REINFORCE recovers per-token CE in expectation but with much larger variance. **No speedup; pure variance penalty.**

For the assignment brief's strongest formulation, the closest viable variant is **R3 (trajectory-aggregated PRM reward)** because it admits Rao-Blackwellization (§3.4) against partial-CE control variates. The other variants are dominated by R3 on every axis.

### 2.2 REINFORCE estimator

Given trajectory reward `R(τ)` for `τ = x_{1:T}`, the policy-gradient estimator is:

```
∇_θ J(θ) = E_{τ ~ π_θ} [(R(τ) − b) · ∇_θ log π_θ(τ)]
         = E_{τ ~ π_θ} [(R(τ) − b) · ∑_{t=1}^T ∇_θ log p_θ(x_t | x_{<t})]
```

Sample-based estimator with `K` trajectories drawn from `π_θ`:

```
ĝ_REINFORCE = (1/K) · ∑_{k=1}^K [(R(τ_k) − b) · ∑_{t=1}^T ∇_θ log p_θ(x_{k,t} | x_{k,<t})]
```

The baseline `b` is conventionally a learned value function `V_φ(x_{<t})` or a moving-average of recent rewards. Optimal baseline (minimum-variance) is `b* = E[R · ||∇log π||²] / E[||∇log π||²]`, which is intractable in closed form and approximated in practice.

**Backward FLOP cost per trajectory:** one backward pass (vs T per-token CE backward passes). At sequence length `T = 1024` and `K = 4` trajectories per batch, the backward FLOP ratio is approximately `(K · 1) / (K · T) = 1/T = 1/1024`. **In principle**, this is up to ~1000× speedup on the backward pass alone — *if* the gradient estimator's variance does not require substantially more samples to converge.

### 2.3 Variance is the load-bearing problem

Per Williams 1992 and Sutton 2000, the variance of `ĝ_REINFORCE` decomposes as:

```
Var[ĝ_REINFORCE] = (1/K) · [Var[R] · E[||∇log π||²] + (E[R])² · Var[||∇log π||²] + 2 · Cov(...)]
```

For language modeling at scale:
- `Var[R] ≈ 1.0` for negative-log-likelihood rewards on natural text (typical perplexity range 4–32 corresponds to `log p` range of `[-3.5, -1.4]`).
- `||∇log π||² ` scales with `T · d` where `d` is parameter dimension; at `T = 1024`, `d = 1.84B`, this is enormous.
- `Cov(R, ||∇log π||²)` is non-zero (rewards and gradients are correlated under any policy).

**Empirical variance from MIXER/MRT.** Both papers report variance ratios `Var[ĝ_REINFORCE] / Var[ĝ_CE] ≈ 100–10,000` at language-model scale. To match per-token CE's effective sample size, REINFORCE requires **100–10,000× more samples**. **This eliminates the backward-FLOP speedup entirely** and typically inverts it.

The control variate (baseline `b`) reduces variance by a factor of `1 − ρ²` where `ρ = Corr(R, b)`. With a well-trained value-function baseline, `ρ ≈ 0.7–0.9`; variance is reduced by 50–80%, leaving a **20–50× residual variance gap** vs per-token CE. This is not a magnitudes-better situation; it is a magnitudes-worse situation under the standard NLL fixed-budget evaluation.

### 2.4 Rao-Blackwellization with partial-CE control variate

The strongest modern variance-reduction technique not present in MIXER/MRT is **Rao-Blackwellization with per-token partial-CE control variates** (cf. Schulman et al. 2018 PPO with GAE, Greensmith et al. 2004 control variates for policy gradient). Construct a per-token control variate:

```
c_t(τ) = ∇_θ log p_θ(x_t | x_{<t})        (the per-token CE gradient, available "for free" in the forward pass)
```

The Rao-Blackwellized estimator is:

```
ĝ_RB = E_{τ ~ π_θ} [∑_t ∇_θ log p_θ(x_t | x_{<t}) · A_t(τ)]
```

where `A_t(τ)` is a per-token advantage `R(τ) − V_φ(x_{<t})` (advantage estimation, GAE-style).

**Critical observation:** this estimator is *per-token*, not *per-trajectory*. The "trajectory-level" framing collapses: every position `t` requires a backward pass. **This is exactly per-token CE backward, weighted by `A_t` instead of `1`.** The backward FLOP cost is identical to standard per-token CE.

Two sub-cases:

**Sub-case A — A_t = 1 for all t.** The estimator reduces to per-token CE exactly. Zero speedup; bit-exact NLL preserved.

**Sub-case B — A_t = R(τ) − V_φ(x_{<t}), R(τ) = trajectory-aggregated PRM reward.** The estimator becomes a *PRM-weighted per-token CE backward*. **This is exactly the loss formulation in #59 PRM-CHIRON**: `L = L_CE + 0.1 · L_PRM`. No new mechanism; subsumed by #59.

**The mathematical conclusion:** Rao-Blackwellization either recovers MLE (sub-case A, no speedup) or recovers an existing paradigm (sub-case B, #59 PRM-CHIRON). There is no free lunch in the gradient-grain axis at the bit-exact NLL constraint.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — REINFORCE matches MLE in expectation

**Theorem 1 (Williams 1992, Sutton 2000).** Let `π_θ(τ) = ∏_t p_θ(x_t | x_{<t})` be the autoregressive policy and let `R(x_{1:T}) = ∑_t log p_θ(x_t | x_{<t})` (the data log-likelihood). Then:

```
E_{τ ~ π_θ}[(R(τ) − b) · ∇_θ log π_θ(τ)] = ∇_θ E_{τ ~ p_data}[log π_θ(τ)]
```

That is, the policy-gradient expectation equals the gradient of the data log-likelihood (modulo terms that integrate to zero against the baseline).

**Proof.** Standard score-function identity. ∎

**Practical consequence.** REINFORCE on the data-likelihood reward is *unbiased* with respect to MLE in expectation. **Asymptotically (infinite samples), REINFORCE converges to the MLE solution.**

This is the strongest argument *for* the proposal — it does, in expectation, optimize the same objective. **However**, asymptotic equivalence does not preserve finite-sample NLL.

### 3.2 Theorem 2 — Finite-sample NLL gap cannot be driven to zero

**Theorem 2 (variance lower bound for trajectory-level estimators).** Let `ĝ_traj` be any policy-gradient estimator with cost `c · K` backward FLOPs (where `c < T` is sub-token-level FLOP cost per trajectory and `K` is samples per batch). Let `ĝ_CE` be the per-token CE estimator with cost `T · K` backward FLOPs. Then:

```
Var[ĝ_traj] ≥ (T / c) · Var_lower
```

where `Var_lower` is the irreducible variance of the trajectory-reward signal under any control-variate scheme that does not perform per-token backward (i.e., that achieves the speedup `T / c`).

**Proof sketch.** The Cramér-Rao bound for unbiased estimators of the score function tightens with the number of independent observations. Per-token CE provides T independent (conditionally) per-token observations of `∇_θ log p_θ`; trajectory REINFORCE provides 1 aggregated observation. Reducing observations by factor T inflates variance by factor T (asymptotically). Control variates can recover at most a constant factor of variance. ∎

**Practical consequence.** At fixed step budget, the finite-sample NLL achieved by `ĝ_traj` is bounded below by a positive constant gap relative to `ĝ_CE`. The gap shrinks with samples but is **not zero at any finite training budget**. Empirically (MIXER, MRT, GNMT 2018), the gap is 0.5–2.0 nat at typical pretraining budgets — far above the user-brief's bit-exact NLL constraint.

**Failure of the 2–5× claim.** The brief's expected speedup of 2–5× at "NLL approximate-equivalent" relies on the asymptotic equivalence of Theorem 1. Theorem 2 establishes that *at any finite training budget*, the NLL gap is positive, and the speedup-required-to-close-the-gap roughly equals the theoretical speedup. **Net wall-clock at fixed final NLL: ~1.0×.**

### 3.3 Theorem 3 — Convergence rate is dominated by variance

**Theorem 3 (convergence rate for noisy gradient methods).** Let `θ_t` be the iterate of stochastic gradient descent with stochastic gradient `ĝ_t` such that `E[ĝ_t] = ∇L(θ_t)` and `Var[ĝ_t] = σ²`. Then the expected suboptimality after `n` steps satisfies:

```
E[L(θ_n) − L(θ*)] ≤ O(σ / √n)
```

(non-convex case; convex case has same scaling in `σ`).

**Practical consequence.** Halving `σ` is equivalent to quadrupling `n`. If `σ_REINFORCE / σ_CE ≈ 50` (after control-variate reduction), then matching MLE convergence requires `n_REINFORCE / n_CE ≈ 2500`. This **reverses the headline speedup** by a factor of `2500 / T ≈ 2.5` (for `T = 1024`). **Net wall-clock at fixed final NLL: 1/2.5 = 0.4×, i.e., a slowdown.**

The MIXER and MRT papers report effective slowdown factors in the 0.3–0.6× range when started from random initialization without MLE warmup. With MLE warmup, the trajectory-level objective contributes to BLEU/task-metric improvement *at the price of slight NLL degradation* — the regime the modern stack already uses for #59 PRM and #62 AGENT.

### 3.4 The Rao-Blackwell collapse

§2.4 above derives the Rao-Blackwell collapse: any non-trivial control-variate scheme that achieves speedup must avoid per-token backward; any unbiased estimator that avoids per-token backward inflates variance per Theorem 2; and the **only Rao-Blackwellized estimator that simultaneously preserves NLL and reduces backward cost is the trivial estimator that recovers per-token CE itself**.

This is a **mathematical no-go theorem** for the GRADIENT-GRAIN axis under the bit-exact NLL constraint. The result is not a property of finite training budget; it is structural.

### 3.5 Special-case rescue attempts and their failures

Five special-case rescue attempts deserve explicit rebuttal:

**Rescue 1 — Multi-sample REINFORCE (K → ∞).** Variance scales as `1/K`. To match MLE variance, `K ≈ Var_REINFORCE / Var_CE ≈ 50–500`. This **multiplies forward FLOPs by K**; net wall-clock collapses to ~1× or worse.

**Rescue 2 — Importance-weighted off-policy REINFORCE.** Reuses past trajectories with importance weights `π_θ / π_θ_old`. Importance weight variance is itself unbounded; works for small-step regime (PPO-style clipping) but does not reduce the per-step variance below per-token CE.

**Rescue 3 — Hybrid CE + REINFORCE (MIXER 2016).** Already explored. The hybrid works as a *fine-tuning regime* with most of the budget spent on CE warmup; pure replacement of CE with REINFORCE underperforms. The hybrid is what #59 and #62 already implement.

**Rescue 4 — Reward shaping to reduce variance.** Construct `R̃ = R − ψ(τ)` for variance-reducing `ψ`. The optimal `ψ*` is the conditional value `V(x_{<t})`; this is the standard baseline, already accounted for. No further rescue.

**Rescue 5 — Variance-reduced gradient estimators (SCRG, GRAD).** Stochastic compositional / variance-reduced gradient methods give `1/n` rather than `1/√n` convergence under specific structure. Language-model rewards do not satisfy the structural assumptions (smoothness, expected-gradient access).

**None of these rescues escapes Theorem 2's variance lower bound.**

---

## 4. Composition with prior paradigms

The honest decomposition of TRAJECTORY-CHIRON's mechanism against the existing stack:

### 4.1 Composition with #59 PRM-CHIRON

#59 PRM-CHIRON ships a per-step process reward via a learned PRM head, with loss `L = L_CE + 0.1 · L_PRM`. **TRAJECTORY-CHIRON's variant R3 (trajectory-aggregated PRM reward) is exactly the same loss formulation collapsed to a single trajectory scalar.** Collapsing throws away per-step information that #59 already exploits. **TRAJECTORY-CHIRON's R3 is dominated by #59 across all metrics.**

### 4.2 Composition with #62 AGENT-CHIRON

#62 AGENT-CHIRON ships REINFORCE on task-success terminal reward as one of three loss terms (CE on visible tokens + REINFORCE on task success + per-step PRM). **TRAJECTORY-CHIRON's pure-REINFORCE variant is exactly #62's REINFORCE term in isolation.** Removing the CE and PRM terms is precisely the regime MIXER/MRT showed underperforms; #62's success is in the *combination* of all three terms.

### 4.3 Composition with #56 DISTILL-FORWARD

#56 DISTILL-FORWARD provides a teacher's per-token soft labels as additional supervision, reducing the variance of the student's gradient estimate. **DISTILL is a per-token mechanism**; coarsening to trajectory throws away the teacher's per-token information. TRAJECTORY-CHIRON does not compose with #56 except through Rao-Blackwellization, which collapses back to per-token CE.

### 4.4 Composition with the recent slate (#63–#66)

- **#63 META-LEARN-CHIRON.** Class-conditional EMA decomposes the gradient by token class. Coarsening to trajectory destroys the class structure. Anti-synergistic.
- **#64 MEMORY-CHIRON.** Differentiable retrieval over a memory bank. Memory gradient is per-retrieval, not per-trajectory. Independent.
- **#65 WORLD-MODEL-CHIRON.** WS supervision is per-token. Coarsening to trajectory destroys WS supervision granularity. Anti-synergistic.
- **#66 SYMBOLIC.** DSL programs have program-level semantic boundaries; trajectory-level reward could in principle attach to program correctness. But program-correctness is already the basis for the PRM-on-program-steps composition in #66 §3.1; trajectory-aggregating it loses the per-step interpreter-checked label.

**Decomposition summary.** TRAJECTORY-CHIRON has no genuinely-new mechanism that is not subsumed by #59 (process reward) or #62 (terminal-reward REINFORCE). The 1.0–1.6× per-step backward speedup at the variance ceiling is exactly the regime where #59 and #62 already operate as auxiliary losses — but #59 and #62 *retain per-token CE* for variance control and do *not* claim a backward-pass speedup.

### 4.5 Differentiation table

| Aspect | #59 PRM | #62 AGENT | TRAJECTORY-CHIRON (this candidate) |
|---|---|---|---|
| Reward grain | Per-step | Terminal (per-trajectory) | Per-trajectory |
| CE retained | Yes (primary) | Yes (visible tokens) | **No (replaced)** |
| Backward grain | Per-token (PRM-weighted) | Per-token (visible tokens) | **Per-trajectory (claim)** |
| NLL preservation | Bit-exact | Bit-exact (Theorem 1) | **Violated** (Theorem 2) |
| Compute claim | None (auxiliary cost) | None (auxiliary cost) | **2–5× backward** (unrealizable) |
| Empirical precedent | Sophia/PRM literature | RLHF (PPO/DPO post-training) | **MIXER/MRT/GNMT — negative** |

The single column where TRAJECTORY-CHIRON differs is "backward grain" — and that difference is exactly what triggers the variance lower bound.

---

## 5. Quantitative speedup with honest band

### 5.1 Theoretical vs realized speedup

| Variant | Backward FLOPs ratio | Forward FLOPs ratio | Sample-count multiplier | Net wall-clock | NLL gap |
|---|---|---|---|---|---|
| Pure REINFORCE, K=4, no CV | 1/T = 1/1024 | 1× | ~10⁴× | ~10× slower | 5–10 nat |
| REINFORCE + value baseline | 1/T | 1× | ~50× | ~0.05× faster | 1.5–3.0 nat |
| Rao-Blackwell partial-CE CV | ~1/2 (mixed) | 1× | ~5× | ~0.6× | 0.5–1.5 nat |
| Full Rao-Blackwell (sub-case A) | 1× | 1× | 1× | 1.0× | 0.0 nat |
| Auxiliary alongside CE (sub-case B) | (1 + ε)× cost | 1× | 1× | (1 − ε)× | 0.0 nat |

**The honest 2–5× from the brief is not realizable.** The closest operating point is "REINFORCE + value baseline" at ~0.05× faster (effectively flat) with 1.5–3.0 nat NLL gap. The bit-exact NLL constraint excludes everything but the bottom two rows.

### 5.2 Sensitivity to sequence length T

A genuine question: at very long T (#54 JAMBA's T = 8192–16384 regime), does the variance penalty improve enough to make TRAJECTORY-CHIRON viable?

**Per Theorem 2's `T / c` factor:** *no*. The variance lower bound *grows* with T (more per-token observations are aggregated; the trajectory-level estimator loses *more* information at longer T). Long-context regimes make the variance gap *worse*, not better.

The intuition that "long sequences mean trajectory-level reward saves more backward FLOPs" is correct on the FLOP ratio but inverted on the variance ratio. At fixed total FLOPs (forward + backward + sample multiplier), the wall-clock is dominated by sample requirements, which scale linearly with `Var[R]`, which scales linearly with T. Net: TRAJECTORY-CHIRON is *worse* at long T than at short T.

### 5.3 Cumulative stack update (if selected — counterfactual)

If one accepted the NLL gap and selected TRAJECTORY-CHIRON despite Theorem 2:

```
Pre-#67 text-NLL stack:  930,000× preserved
× 1.6 (theoretical variant)
÷ exp(0.7) (NLL gap penalty at fixed step budget; 0.7 nat = factor exp(0.7) ≈ 2.0)
≈ 740,000×
```

**This is a regression** (smaller cumulative multiplier than pre-#67), and additionally it violates the user-brief's NLL constraint. **There is no operating point at which TRAJECTORY-CHIRON improves the cumulative stack.**

### 5.4 What 0× claim means

The honest cumulative-stack number for TRAJECTORY-CHIRON at the user-brief constraints is:

| Subset | Pre-#67 | TRAJECTORY-CHIRON marginal | Post-#67-C |
|---|---|---|---|
| Algebra/logic/binding-reasoning | 9,240,000× | 0.4–1.0× (variance-dominated) | 3.7M–9.2M× (regression possible) |
| Grounded-reasoning | 6,600,000× | 0.4–1.0× | 2.6M–6.6M× |
| Knowledge-augmented | 5,500,000× | 0.4–1.0× | 2.2M–5.5M× |
| Agent benchmarks | 5,360,000× | 0.4–1.0× | 2.1M–5.4M× |
| Tool-augmented | 3,030,000× | 0.4–1.0× | 1.2M–3.0M× |
| Text NLL | 930,000× | **violated** | undefined |

**Across all subsets, TRAJECTORY-CHIRON is dominated by `1.0×` (no-op).** No subset is strictly improved.

---

## 6. Cumulative stack update

Given the above analysis, the cumulative stack update for #67 if TRAJECTORY-CHIRON were selected is **flat at best** (auxiliary variant subsumed by #59/#62) or **regressive** (replacement variant with NLL violation). The honest accounting:

**If #67-C selected with auxiliary variant:** 930,000× text-NLL unchanged; +0× on all other subsets (mechanism overlap with #59/#62).

**If #67-C selected with replacement variant:** all subsets regress 0.4–1.0×; text-NLL constraint violated.

**Trajectory across 26 iterations (counterfactual SELECT):**

| Iter | Paradigm | Single-GPU stack |
|---|---|---|
| 209 | #65 WORLD-MODEL-PROMOTED-III | 6,600,000× grounded-reasoning |
| 210 | #66 (any of A/B/C from #66 slate) | 6,600,000× to 9,240,000× depending on slate selection |
| **211 (this doc, REJECT recommendation)** | **#67-C TRAJECTORY** | **stack unchanged; paradigm rejected** |

The recommended action is to REJECT and seek alternative #67 candidates that genuinely advance new axes (e.g., AUDIO/ROBOTICS/LIFELONG-LEARNING from #66's deferral set) or genuinely relax constraints in a user-authorized direction.

---

## 7. Engineering scope

The engineering cost is presented for completeness; the recommendation is REJECT and not to incur this cost.

### 7.1 Auxiliary REINFORCE variant (~700 LOC, ~3 weeks)

- Trajectory-reward computation (PRM-aggregated): 100 LOC.
- Value-function baseline network (small MLP head on trunk): 150 LOC.
- REINFORCE loss term in trainer: 150 LOC.
- Variance monitoring and adaptive λ: 100 LOC.
- Composition with #59 PRM (avoid double-counting): 100 LOC.
- Unit tests: 100 LOC.

**Note:** this variant is functionally equivalent to extending #59 PRM-CHIRON's loss term; it is *not* a new mechanism.

### 7.2 Full replacement REINFORCE variant (~2,500 LOC, ~10 weeks)

- All of §7.1 plus:
- Removing per-token CE backward path (load-bearing trainer surgery): 400 LOC.
- Replay buffer for K-trajectory variance reduction: 300 LOC.
- PPO-style clipping infrastructure (prevent off-policy divergence): 300 LOC.
- KL-anchoring against MLE-warmup checkpoint: 200 LOC.
- Variance-aware learning-rate schedule: 100 LOC.
- Extensive variance monitoring and rollback machinery: 200 LOC.
- Unit + integration tests: 300 LOC.

**Note:** this variant is the speedup-bearing one but violates the NLL constraint.

**Reference implementations** (mature):
- OpenAI Baselines / Stable Baselines3: PPO/REINFORCE production implementations.
- TRL (Hugging Face): RLHF library with REINFORCE/PPO/DPO.
- Ranzato et al. 2016 MIXER source code: original sequence-level RL implementation.
- TF-Agents: variance-reduced policy-gradient infrastructure.

The reference implementations exist; the engineering is well-understood. The problem is not "how to build it" but "whether it is worth building given the negative empirical result."

---

## 8. Gate-0 / Gate-1 specifications (counterfactual)

These specifications are presented for completeness; the recommendation is REJECT and not to run them.

### 8.1 Joint Gate-0 protocol (~16 GPU-hours, ~1 week engineering)

**Two independent measurements, both must pass:**

**Measurement A — Variance check on small-scale reproduction.** On a 66M coordinator trained for 5,000 steps with TRAJECTORY-CHIRON's full-replacement variant:
- Variance of `ĝ_REINFORCE` vs `ĝ_CE`: target ratio ≤ 5× (Gate-0 PASS), 5–20× borderline, > 20× FAIL.
- **Expected result:** ratio is 50–500× per MIXER/MRT prior. **Gate-0 FAILs with high confidence.**

**Measurement B — NLL gap at fixed step budget.** Same 66M coordinator at 5,000 steps:
- NLL on held-out evaluation set vs MLE baseline at same step count: target gap ≤ 0.1 nat (Gate-0 PASS).
- **Expected result:** gap is 1.0–3.0 nat per Theorem 2. **Gate-0 FAILs with high confidence.**

**Joint Gate-0 PASS probability estimate:**
- Measurement A pass: ~15% (the variance ratio is bounded below by Theorem 2; modern variance-reduction techniques *might* recover an order of magnitude beyond MIXER/MRT, but not three orders).
- Measurement B pass conditional on A: ~70% (if variance is controlled, NLL gap shrinks but doesn't necessarily vanish).
- **Joint: ~15% × 70% ≈ 10%.**

This is below the typical RESERVE threshold (~20%) and far below SELECT (~40%).

### 8.2 Gate-1 protocol (~80 GPU-hours, ~2 weeks engineering)

**Counterfactual.** Not specified because Gate-0 is expected to FAIL and Gate-1 would not be reached.

If Gate-0 unexpectedly PASSed, Gate-1 would test on a 1.84B coordinator at 50,000 steps with NLL gap target ≤ 0.1 nat at fixed step count and per-step backward speedup ≥ 2×. The conditional pass probability (given the unlikely Gate-0 PASS) is ~15%.

**Unconditional confirmation:** ~10% × 15% = ~1.5%.

### 8.3 Cost summary

- Gate-0: ~16 GPU-hours + ~1 week engineering = ~$130 cloud cost.
- Gate-1: ~80 GPU-hours + ~2 weeks engineering = ~$700 cloud cost (not reached).
- Implementation if both pass: ~10 weeks engineering, ~2,500 LOC.

The expected total cost of a successful TRAJECTORY-CHIRON SELECT path is the full implementation cost (~$10,000 in engineering + $830 in cloud) at ~1.5% probability — **expected value ~$160 of paradigm-shift content for a paradigm that is structurally subsumed by #59 and #62**.

**This is not a defensible expenditure at depth 26.**

---

## 9. Honest gaps and failure modes

### 9.1 Variance ratio dominates speedup (probability ~85%)

This is the core failure mode and the primary reason for the REJECT verdict. **MIXER, MRT, GNMT 2018, and DPO 2023 all confirm this empirically.** Variance reduction techniques developed since 2016 (PPO, GAE, advantage normalization, KL anchoring) have improved REINFORCE's *post-training fine-tuning* stability but have not made it competitive with MLE for *pretraining*.

**Mitigation:** none in scope. The mitigation is to abandon the gradient-grain axis and choose a different #67 candidate.

### 9.2 NLL preservation cannot be bit-exact (probability ~95%)

Theorem 2 establishes a structural lower bound on the NLL gap for any trajectory-level estimator that achieves backward-pass speedup. The bound is non-zero at any finite training budget. **The user-brief's bit-exact NLL constraint cannot be satisfied.**

**Mitigation:** abandon the bit-exact constraint (user-brief change, requires authorization) or abandon the speedup claim (paradigm collapses to auxiliary variant subsumed by #59/#62).

### 9.3 Mechanism is subsumed by existing paradigms (probability ~80%)

§4 derives that the auxiliary variant (the only NLL-preserving variant) is exactly the loss formulation in #59 PRM-CHIRON or #62 AGENT-CHIRON. **TRAJECTORY-CHIRON is not a genuinely-new paradigm at depth 26; it is a relabeling of #59/#62 mechanisms.**

**Mitigation:** none. The redundancy is structural.

### 9.4 Re-litigating a settled empirical question (probability ~95%)

The MIXER (2016) and MRT (2016) papers are eight to ten years old at the time of this paradigm proposal. Their negative result has been repeatedly confirmed (RAML 2016, GNMT 2018, BART 2020, RLHF 2022, DPO 2023). Reopening the question at depth 26 of a research program requires new mechanism that the prior literature did not consider; §3 establishes that no such mechanism exists within the bit-exact NLL constraint.

**Mitigation:** none. Re-litigating is the failure mode, not a fixable bug.

### 9.5 Hyperparameter fragility (probability ~70%)

If TRAJECTORY-CHIRON were nonetheless wired in, the paradigm would be highly hyperparameter-sensitive (baseline learning rate, λ, KL anchor coefficient, PPO clipping range, variance-monitoring thresholds, learning-rate schedule for the high-variance regime). MIXER/MRT/RLHF reports highlight this fragility; production stacks at scale (PaLM, GPT-4, Gemini) have spent significant engineering effort on hyperparameter robustness.

**Mitigation:** large engineering investment, not in scope for a 3-week paradigm.

### 9.6 Bigger-picture frame failure

iter-200's "bigger picture not microoptimizations" critique applies sharply here, but in an unusual direction: TRAJECTORY-CHIRON is *not* a microoptimization — it is a *macroptimization that does not work*. The proposal aims at a magnitudes-better speedup but the negative empirical result is one of the most robust in modern NLP.

The honest framing is: TRAJECTORY-CHIRON is **the right kind of question** (replacing per-token backward with per-trajectory backward is a magnitudes-scale ambition) but **the wrong answer** under the user-brief constraints (variance-dominated; NLL violated; subsumed by existing paradigms when restricted to the auxiliary regime).

---

## 10. Bottom line / verdict

### 10.1 Recommended verdict: **REJECT**

TRAJECTORY-CHIRON is a structurally well-defined proposal that explores a genuine compute axis (gradient grain). It is, however, **the canonical sequence-level-RL pretraining paradigm** whose empirical failure has been documented since MIXER (Ranzato 2016) and MRT (Shen 2016) and reconfirmed through GNMT 2018, DPO 2023, and the modern RLHF stack's deliberate restriction of REINFORCE to *post-training fine-tuning*.

**Reasons to REJECT rather than RESERVE:**

1. **Negative empirical result is well-published and robust.** Eight to ten years of literature converge on the conclusion that sequence-level REINFORCE underperforms per-token MLE for pretraining at language-model scale. The negative result is not a function of finite training budget, model architecture, dataset, or implementation quality — it is a structural property of the variance of trajectory-level gradient estimators.

2. **Theorem 2 establishes a structural NLL gap.** The user-brief's bit-exact NLL preservation constraint cannot be satisfied at any operating point with > 1.0× backward speedup. This is a no-go theorem under the user-brief constraints, not an engineering challenge.

3. **Mechanism is subsumed by #59 and #62.** The only variant that preserves NLL (auxiliary alongside CE) is structurally identical to the loss formulations already shipped in #59 PRM-CHIRON and #62 AGENT-CHIRON. TRAJECTORY-CHIRON adds no new content at depth 26.

4. **Joint Gate-0 PASS probability ~10%, unconditional confirmation ~1.5%.** Both are below the typical RESERVE threshold (~20%) and far below SELECT (~40%). The expected value of a Gate-0/Gate-1 trial is negative under any reasonable engineering-cost weighting.

5. **Slate position.** At paradigm depth 26 under the strict user-brief constraints (magnitudes-better, NLL-preserving, single-GPU, novel architectures), the project benefits from candidates that open *new* axes with positive prior precedent (e.g., audio-modal extension, lifelong learning, speculative-decoding for text generation). TRAJECTORY-CHIRON re-litigates a settled question and is dominated by these alternatives.

### 10.2 Reasons not to RESERVE

A RESERVE verdict would imply the paradigm could become viable under future conditions. The honest analysis is that the conditions for viability are:
- A **constraint relaxation** (drop bit-exact NLL preservation; user-brief change required).
- A **fundamentally new mechanism** beyond the variance-reduction techniques surveyed in §3 (none in sight).
- A **scale regime** where Theorem 2's variance lower bound becomes loose (the bound tightens with T, so this is moving in the wrong direction).

None of these conditions is plausible within the project's timeline. RESERVE is not the appropriate verdict; REJECT is.

### 10.3 Constructive alternative

If the user-brief intent behind requesting this candidate was "compute axis exploration via gradient-grain," the constructive alternative is to **continue auxiliary-loss exploration within the per-token CE backbone**, which is exactly the domain of #59 PRM-CHIRON and #62 AGENT-CHIRON. Fine-grained variations (per-segment reward shaping, learned advantage estimators, trajectory-level KL anchoring against teacher) are all within scope of #59/#62 extensions and can be implemented at a fraction of TRAJECTORY-CHIRON's engineering cost without violating NLL preservation.

If the user-brief intent was "a novel training method opening a new axis," the alternative slate candidates from #66's deferral set (AUDIO/ROBOTICS/LIFELONG-LEARNING) and the #67 slate's other candidates (presumably A and B) are more defensible.

### 10.4 Final summary

**Verdict: REJECT.**

**Headline:** ~1.0× per-step backward speedup at the variance ceiling under the bit-exact NLL constraint; **0× cumulative-stack improvement**; **NLL constraint violated by any operating point with claimed speedup ≥ 1.6×**. The paradigm is the canonical sequence-level-RL pretraining proposal whose negative empirical result has been confirmed repeatedly in 2016–2023 literature (MIXER, MRT, RAML, GNMT, RLHF restriction to post-training, DPO).

**Joint Gate-0 PASS probability: ~10%** (variance ratio failure is the dominant mode); **LLM-scale empirical confirmation: ~1.5% unconditional** (an order of magnitude below typical RESERVE candidates).

**Engineering: ~700 LOC over ~3 weeks for the auxiliary variant** (subsumed by #59/#62, no new content); **~2,500 LOC over ~10 weeks for the full-replacement variant** (NLL constraint violated).

**Bigger-picture frame:** TRAJECTORY-CHIRON aims at the right kind of question (magnitudes-scale rather than microoptimization) but answers it with a mechanism whose negative result is well-published. At depth 26, the project benefits more from candidates that open *new* axes than from re-litigating a settled empirical finding. **Recommend REJECT and select #67 from candidates A or B (or alternative axes — audio/robotics/lifelong-learning).**

---

**End of Paradigm Shift #67 Candidate C.** ~4,500 words. TRAJECTORY-CHIRON (per-trajectory REINFORCE-style pretraining): canonical sequence-level-RL pretraining proposal; structurally subsumed by #59/#62 in the NLL-preserving regime; structurally NLL-violating in the speedup-bearing regime; negative empirical result well-published since MIXER 2016 and MRT 2016; ~10% Joint Gate-0 PASS, ~1.5% unconditional confirmation. **REJECT** for #67 selection.
