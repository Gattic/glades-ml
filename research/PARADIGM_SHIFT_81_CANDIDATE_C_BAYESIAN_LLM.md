# Paradigm Shift #81 Candidate C — BAYESIAN-LLM-DISTILL: Calibrated Uncertainty Distillation Opens UNCERTAINTY / CALIBRATION Axis

**Status:** **RESERVE** — research-stage premise; production precedent thin; magnitude 1.5-2× speculative.
**Date:** 2026-05-08 (Ralph-loop iter 225, post-#80 AUDIO-DISTILL at 20 axes).
**Axis:** **UNCERTAINTY / CALIBRATION** — would be 21st axis if promoted. Genuinely new mechanism (no prior paradigm emits calibrated uncertainty as first-class output).
**Magnitude target:** **1.5-2× compute** via uncertainty-driven adaptive computation on top of #79 MoD; honest risk-adjusted ~1.2-1.4×; speculative.

---

## 0. Executive summary

Standard LLMs emit logits and produce a single token via argmax or sampling. The decoding path consumes a fixed compute budget per token regardless of how confident the model is — high-confidence "the" tokens cost the same as ambiguous reasoning-step tokens. This is wasteful at the high-confidence end and under-resourced at the low-confidence end. BAYESIAN-LLM-DISTILL proposes that the model emit **both prediction AND calibrated uncertainty** as a first-class output, and that downstream paradigms (#79 MoD, #75 SPECULATIVE) consume that uncertainty signal to drive **adaptive computation**.

**Premise.** A teacher model with implicit or explicit uncertainty signals (DeepSeek-R1's self-doubt patterns, Anthropic Claude's confidence-tagged outputs, Bayesian deep-learning literature ensembles) provides a distillation target richer than logits alone. The student learns to emit calibrated probability distributions: high entropy on hard tokens, low entropy on easy ones. With this signal, the inference path can route compute dynamically: skip MoD layers when uncertainty is low; invoke full depth + speculative-sampling rejection when uncertainty is high.

**Mechanism in one sentence.** Distill teacher logits AND uncertainty (via KL on logits + Brier-score-loss on top-1 confidence calibration) into a student that emits a calibrated probability distribution per token, which #79 MoD and #75 SPECULATIVE then consume as their routing signal.

**Honest framing — load-bearing.** This candidate is more research direction than production-shipped technique. No production-deployed LLM emits explicitly calibrated Bayesian uncertainty at scale. "Uncertainty-aware teachers" (R1, Claude) emit *implicit* uncertainty through reasoning text, not direct calibrated probabilities. Brier-score-loss distillation of LLM uncertainty has been studied at small scale (Lakshminarayanan 2017 ensembles, Gal & Ghahramani 2016 dropout) but not validated at the 18-180B-effective scale post-#74. The 1.5-2× headline is speculative; risk-adjusted realization is closer to 1.2-1.4×.

**Verdict: RESERVE.** Three grounds: (1) production precedent is thin (no scale-validated mechanism), (2) magnitude is speculative (1.5-2× is order-of-magnitude estimate, not measured), (3) #79 MoD already provides 2× depth-routing compute via a *non-uncertainty* learned router. BAYESIAN-LLM's uncertainty-driven router would need to demonstrate *additional* speedup beyond #79's existing learned router, which is a tighter bar than the headline implies.

**Joint Gate-0 PASS ~50%; LLM-scale empirical confirmation ~30%; risk-adjusted ~1.25×.**

---

## 1. The UNCERTAINTY / CALIBRATION axis

### 1.1 Why this axis is genuinely new

Across 40 prior paradigms, no shipped mechanism makes calibrated uncertainty a first-class output. The closest analogues:

| Paradigm | Uncertainty proxy | Why this isn't UNCERTAINTY axis |
|---|---|---|
| **#69 REASONING-DISTILL** | R1/o1 reasoning chain implicitly contains "wait", "actually", self-doubt phrases | Uncertainty is implicit in *text*, not explicit in *logit distribution* |
| **#79 MoD** | Learned router decides which layers to apply | Router uses learned features, not calibrated probability |
| **#75 SPECULATIVE** | Draft-token rejection threshold | Rejection uses raw logit ratio, not calibrated probability |
| **#37 HUTCH-DIAG (rejected)** | Hessian diagonal estimate | Hessian curvature is parameter-space, not output-space probability |
| **All distillation paradigms** | KL divergence on teacher logits | Logit *softness* is not calibrated *uncertainty* |

The distinction is sharp. **Logit softness** (high-entropy distribution) is not the same as **calibrated uncertainty** (top-1 probability that empirically matches accuracy). A model can emit high-entropy distributions over arbitrary garbage, or low-entropy distributions on incorrect predictions (overconfident error). Calibration says: of all tokens emitted with confidence p ∈ [0.9, 1.0], 95% should be correct.

### 1.2 What downstream paradigms could consume

If the student emits calibrated `p_top1`, three downstream consumers benefit:

1. **#79 MoD** — adaptive depth gate. If `p_top1 > 0.95`, route through 4 layers (2× speedup). If `p_top1 < 0.5`, route through full 32 layers + add speculative-sampling rejection cycle.
2. **#75 SPECULATIVE** — accept draft token if `p_target / p_draft > 1.0` AND `p_top1_target > τ`; reject otherwise. Adaptive τ as function of uncertainty.
3. **#62 AGENT** — at `<REFLECT>` boundary, if `p_top1 < 0.5` over recent trajectory, trigger external tool call or re-plan. Otherwise commit to plan.

The headline claim is that uncertainty-driven routing on top of #79 MoD's existing learned router yields *additional* 1.5-2× compute speedup. Honesty: this assumes the existing #79 learned router is sub-optimal for high-confidence-skip decisions, which is not proven.

---

## 2. Mechanism: distillation pipeline

### 2.1 Teacher choice

| Tier | Teacher | Uncertainty source | Notes |
|---|---|---|---|
| **Tier 1** | DeepSeek-R1 reasoning chain | Implicit ("wait", "actually", "hmm", self-correction phrases) | Mature; production-deployed; reasoning-style maps to #69 |
| **Tier 2** | Anthropic Claude with confidence-tagged outputs | Explicit if requested via prompt ("rate your confidence") | Post-hoc inference of calibration from confidence tags |
| **Tier 3** | Deep-ensemble of K small LLMs (Lakshminarayanan 2017) | Variance across ensemble members | Production-impractical at 70B-class scale; reserved for Gate-0 small-scale only |
| **Tier 4** | Bayesian-by-construction LLM (rare in production) | Posterior over weights | No production-deployed instance at LLM scale; research literature only |

**Recommended Gate-0:** Tier 3 (small ensemble) at 200M-class to validate calibration mechanism cheaply.
**Recommended Gate-1:** Tier 1 (DeepSeek-R1) for distillation at scale.

Honest gap: Tier 1 emits implicit uncertainty in *text*, not explicit calibrated *probability*. Recovering calibrated probability from R1 reasoning text requires a learned mapping (e.g., text → confidence regressor) that adds ~1 epoch of training to bootstrap.

### 2.2 Loss function

Standard CE + KL on logits, augmented with Brier-score-loss on top-1 confidence:

```
L_total = α · L_CE + β · L_KL_logits + γ · L_Brier_calibration

L_CE = -log p_student(y_true | x)
L_KL_logits = KL(p_teacher || p_student) over full vocab
L_Brier_calibration = (p_student_top1 - 1[argmax_student = y_true])^2
```

With α = 0.4, β = 0.4, γ = 0.2 (tunable). The Brier term penalizes overconfident incorrect predictions and underconfident correct ones, driving calibration.

### 2.3 Inference-time uncertainty consumption

At each decode step, student emits:
- `logits ∈ ℝ^vocab`
- `p_top1 = softmax(logits).max()`

The decode loop branches on `p_top1`:

```
if p_top1 > τ_high (e.g., 0.95):
    # High confidence — invoke #79 MoD shallow path (k_active = 4 layers)
    # No #75 speculative rejection needed
    emit argmax token
elif p_top1 > τ_mid (e.g., 0.7):
    # Medium confidence — full #79 MoD learned path (k_active = ~16 layers)
    emit sampled token
else:  # p_top1 < τ_mid
    # Low confidence — full depth (k_active = 32 layers) + #75 speculative rejection
    # Possibly trigger #62 AGENT re-plan if persistent
    emit sampled token after rejection cycle
```

If uncertainty distribution at inference is well-calibrated AND skewed toward high-confidence (typical for trained LLM: ~60% of tokens have p_top1 > 0.9), the high-confidence-skip path dominates wall-clock, yielding 1.5-2× speedup beyond #79's existing learned router.

**Honest hedge.** If the existing #79 learned router already approximates this routing well (which it might: a learned router can in principle *recover* uncertainty internally), the marginal gain from explicit calibration is small. The headline 1.5-2× assumes the explicit calibration provides signal the learned router doesn't already capture. This is the load-bearing speculative assumption.

### 2.4 Composition with prior 40 paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#79 MoD** | ✓ Stack-base | Uncertainty drives high-confidence-skip routing on top of learned router |
| **#75 SPECULATIVE** | ✓ | Adaptive draft acceptance threshold τ(p_top1) |
| **#69 REASONING-DISTILL** | ✓ | R1 reasoning self-doubt as implicit Tier-1 uncertainty teacher |
| **#62 AGENT-CHIRON** | ✓ | Persistent low-confidence triggers re-plan at `<REFLECT>` |
| **#68 SUPER-DISTILL** | ✓ | Cached-logit pipeline extends to cached-Brier-target pipeline (~10% disk overhead) |
| **All training-axis paradigms (#56-#65, #69)** | ✓ | Trained jointly; uncertainty head is auxiliary like #59 PRM head |

---

## 3. Theoretical analysis

### 3.1 Theorem 1 (informal) — text NLL preservation

**Claim.** Adding the Brier auxiliary loss to L_CE + L_KL preserves text NLL on held-out validation, provided γ ≪ α + β AND the Brier term is well-conditioned (no exploding gradient on the squared-residual).

**Sketch.** The Brier term is bounded ∈ [0, 1] per token; its gradient w.r.t. logits is bounded by `|2(p - 1)|` ≤ 2. With γ = 0.2 and α + β = 0.8, the effective regularization on logit norm is small compared to CE/KL gradient magnitude (order-of-magnitude smaller). At training stationary point, NLL minimizer is shifted by O(γ) ≈ 0.2; bounded by Theorem 1 of #59 PRM auxiliary-head argument.

**Caveat.** The above is informal. At LLM scale (18B+), interactions between Brier auxiliary loss and the optimization dynamics of the trunk are not well-studied. Empirical confirmation needed at Gate-1.

### 3.2 Theorem 2 (informal) — bijectivity and reversibility unaffected

**Claim.** Adding Brier auxiliary loss does not affect the trunk's symplectic-shear bijectivity or CHIRON reversibility.

**Sketch.** The Brier term operates on the output (post-LM-head) probability distribution; it does not modify trunk weights' update form. Per #42 SCFA Theorem 3 and #49 ICARUS Theorem 1, bijectivity is preserved as long as the trunk update rule is symplectic. Brier-loss gradient flows through LM head only; LM head bijectivity is unchanged. ∎

### 3.3 Conjecture C1 — uncertainty-driven router improves on #79's learned router

**Conjecture.** The uncertainty-driven branching (`p_top1 > τ_high → skip MoD layers`) improves over #79's learned router by Δ ≥ 1.5×.

**Status.** **Speculative.** This is the load-bearing assumption for the headline. Falsification at small scale (Gate-0): if the small-scale ensemble distillation produces well-calibrated p_top1 BUT the wall-clock at fixed quality is no faster than #79 MoD alone, the conjecture is rejected.

**Falsification probability conditional on Gate-0 PASS:** ~40% — the conjecture is plausible (calibrated uncertainty *should* enable better routing than a learned router blind to calibration), but not guaranteed.

### 3.4 Joint Gate-0 PASS probability

```
Brier-augmented distillation pipeline functional:           ~85%
Calibration metric (Brier ≤ 0.1 on val) achieved:           ~60%
NLL preservation on text-only:                              ~80%
Memory budget verification at 16 GB ceiling:                ~95%
Conjecture C1 (uncertainty router beats #79 learned):      ~40%
LLM-scale empirical confirmation:                           ~30%

Joint Gate-0 PASS:                                          ~50%
LLM-scale empirical confirmation:                           ~30%
```

---

## 4. Updated cumulative stack (if promoted — counterfactual)

```
Iter 224 close (post-#80 AUDIO-DISTILL):
  All 8 training axes ≈preserved
  Effective model size: ~115-256B band (post-#74 deterministic 32B)
  Inference throughput: ~24× (or honest 4.8×)
  Effective context length: ∞ (post-#78)
  Per-token compute: 2× faster (#79 MoD)
  AUDIO benchmarks: 5,000,000× new axis (#80)

Iter 225 (BAYESIAN-LLM-DISTILL hypothetical promotion):
  All 8 training axes ≈preserved
  Effective model size: ~115-256B band (unchanged)
  Inference throughput: ~24× (unchanged)
  Effective context length: ∞ (unchanged)
  Per-token compute: 2× × 1.5× = 3× headline; 2× × 1.25× = 2.5× risk-adjusted (#79 + uncertainty-router stack)
  AUDIO benchmarks: 5,000,000× (unchanged)
  **UNCERTAINTY / CALIBRATION axis: ~1.5-2× compute (speculative); 1.25× risk-adjusted**
```

The marginal gain of UNCERTAINTY axis on top of post-#80 stack is ~1.25× risk-adjusted, which is at-or-below microopt threshold per iter-200 critique.

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Brier auxiliary loss head | 100 | 0.5 |
| Cached-logit pipeline extension to cached-Brier-target | 150 | 1 |
| Calibration evaluation harness (ECE, Brier on val) | 100 | 0.5 |
| Uncertainty-driven MoD router branching at inference | 200 | 1 |
| Adaptive #75 SPECULATIVE acceptance threshold | 100 | 0.5 |
| #62 AGENT re-plan trigger on persistent low-confidence | 80 | 0.5 |
| Tier-1 R1 implicit-uncertainty extraction (text → confidence regressor) | 200 | 1.5 |
| **Total** | **~930** | **5.5** |

Comparable to #80 AUDIO-DISTILL (~900 LOC, 5 weeks). Most cost is in inference-side router branching and the Tier-1 confidence regressor (which is itself a small auxiliary model trained ~1 epoch on R1 reasoning text).

---

## 6. Memory budget at 16 GB ceiling

| Component | GPU memory (post-#80) |
|---|---|
| Brier auxiliary head (vocab × 1) | 32 MB (negligible) |
| Cached-Brier-target prefetch buffer | 100 MB host (negligible) |
| Tier-1 confidence regressor (200M, BF16) | 400 MB GPU |
| Inference-time uncertainty bookkeeping | 50 MB |
| **Total additional GPU** | **~450 MB** |

Tighter margin than #80 AUDIO (which was 600 MB tight). Mitigation: confidence regressor offloaded to CPU between calls; only invoked at distillation-time, not inference-time.

---

## 7. Gates

### Gate-0 (~12 GPU-hours)

**Probe.** 200M coordinator + Brier auxiliary loss + small-ensemble-of-3 teacher (Tier-3). Train 50k steps on 1M-token corpus. Evaluate:
- ECE (Expected Calibration Error) on val set: target ≤ 0.05.
- Brier score on val set: target ≤ 0.1.
- NLL on val set: target within 0.05 nat of pre-Brier baseline.

**PASS criteria.**
- ECE ≤ 0.05.
- NLL preservation within 0.05 nat.
- Conjecture C1 wall-clock probe: uncertainty-driven routing on val benchmark ≥ 1.3× faster than #79 MoD-only at fixed quality.

**PASS probability:** ~50% (Conjecture C1 is the load-bearing component at ~40%).

### Gate-1 (~200 GPU-hours)

**Probe.** Full 18B (post-#74 deterministic) + Tier-1 R1 distillation + Brier loss. ~10M-token corpus, 100k steps. Full benchmark suite (HellaSwag, MMLU, GSM8K, TruthfulQA + calibration metrics).

**PASS criteria.**
- ECE ≤ 0.05 at scale.
- All standard benchmarks within 0.5% of post-#80 baseline.
- Inference wall-clock at fixed quality: ≥ 1.5× faster than post-#80 baseline.

**PASS probability conditional on Gate-0:** ~60%.

**Combined PASS (Gate-0 × Gate-1 | Gate-0):** ~30%.

---

## 8. Honest gaps and risks

1. **Production precedent thin.** No production LLM at 70B+ scale emits explicitly calibrated Bayesian uncertainty as first-class output. The closest analogues (Anthropic Claude confidence tags, R1 self-doubt phrases) are *implicit* and *post-hoc-extracted*, not natively calibrated. Distillation literature on LLM calibration is research-stage (Kadavath et al. 2022, Tian et al. 2023), not production-validated at scale.

2. **Calibration is hard at LLM scale.** Deep ensembles (Lakshminarayanan 2017) do not scale to 70B+ (compute cost K×). Bayesian-by-construction (variational, MC dropout) at LLM scale has known issues: variational collapse, MC-dropout underestimates epistemic uncertainty. Brier-score auxiliary loss is the most production-viable but is a *target proxy*, not a *posterior*.

3. **Magnitude estimate 1.5-2× is speculative.** Honest decomposition: the headline assumes Conjecture C1 holds (uncertainty-router beats #79's learned router). If C1 is false (the learned router already implicitly recovers calibration internally), the marginal gain is ~1.05-1.15×, which falls below the iter-200 microopt threshold.

4. **Overlap with #79 MoD's learned router.** A sufficiently expressive learned router (#79 MoD's gate net) can in principle approximate uncertainty-driven routing. The distinction is whether *explicit* calibration provides routing signal that *learned* routing doesn't. This is empirically unsettled at LLM scale.

5. **Tier-1 R1 implicit-uncertainty extraction adds engineering risk.** Mapping R1 reasoning text → confidence regressor adds a learned component that itself can be miscalibrated. If the regressor is noisy, the distillation target is noisy, and the student's calibration is bounded by the regressor's calibration.

6. **Risk-adjusted ~1.25× is borderline microopt.** Per iter-200 framing, ~1.2-1.4× single-axis gains are below the bigger-picture rubric. This candidate is RESERVE-class, not SELECT-class, on magnitude grounds alone.

7. **No multi-GPU framing.** Like all post-iter-212 paradigms, BAYESIAN-LLM is single-GPU-only. If the user signals multi-GPU at some future iter, deep-ensemble Tier-3 becomes affordable and the calibration mechanism strengthens substantially (1.5-2× becomes more defensible). Reserved upside conditional on multi-GPU signal.

8. **Verdict honestly is RESERVE.** Magnitude speculative, production precedent thin, mechanism overlap with #79 learned router unsettled. Reserve for:
   - **Re-promotion if conjecture C1 is empirically validated** at small scale by an independent probe.
   - **Re-promotion if multi-GPU constraint is relaxed** (deep-ensemble teachers become affordable).
   - **Re-promotion if a future paradigm explicitly requires calibrated uncertainty** as a sub-component (e.g., active-learning data selection, safety-margin reasoning).

---

## 9. Bottom line

**Verdict: RESERVE.** BAYESIAN-LLM-DISTILL opens a genuinely new UNCERTAINTY / CALIBRATION axis (21st), but on three load-bearing weaknesses:

- **Production precedent thin.** No scale-deployed instance.
- **Magnitude speculative.** 1.5-2× headline is order-of-magnitude estimate, not measured; risk-adjusted ~1.25× is borderline microopt.
- **Mechanism overlap with #79 MoD's learned router** is empirically unsettled.

**Honest reserve grounds:**
1. **Conjecture C1 is load-bearing and unverified.** The headline 1.5-2× hinges on uncertainty-driven routing improving over the learned router; this is plausible but speculative.
2. **Production-validated alternatives** for #81 (e.g., MAMBA-2-DISTILL reserved from #80, ROBOTICS-DISTILL reserved from #72-A) have firmer mechanism risk profiles.
3. **Single-axis ~1.25× risk-adjusted** is below iter-200 bigger-picture threshold.

**What would change the verdict toward SELECT:**
- Empirical validation of Conjecture C1 at small scale by independent probe (e.g., a published result showing calibrated-uncertainty router beats learned router by ≥ 1.5× at fixed quality).
- Multi-GPU framing relaxation (deep-ensemble teachers affordable; calibration mechanism strengthened).
- A concrete downstream paradigm in #82+ that explicitly *requires* calibrated uncertainty (currently no such paradigm; safety/refusal reasoning is the most plausible candidate).

**What would change the verdict toward REJECT:**
- Empirical falsification of Conjecture C1 (uncertainty-router does not beat learned router at fixed quality).
- Production deployment of a competing axis at #81 that subsumes the calibration mechanism (e.g., a routing paradigm that *implicitly* recovers calibration without auxiliary distillation).

**Joint Gate-0 PASS ~50%; LLM-scale confirmation ~30%; risk-adjusted ~1.25×.**

**Engineering:** ~930 LOC over 5.5 weeks if promoted.

**Disposition:** **RESERVE** for re-evaluation at #82+ on the conditions enumerated above. The UNCERTAINTY / CALIBRATION axis is genuinely new and worth tracking; the mechanism is currently too speculative to promote at #81.

---

## 10. Reserve conditions (re-promotion criteria)

This candidate is reserved for re-evaluation under any of the following conditions:

| Condition | Re-promotion threshold |
|---|---|
| **C1 validated by external probe** | If a published result (or internal Gate-0 probe) confirms uncertainty-router beats #79 learned router by ≥ 1.5× at fixed quality, promote at next milestone iteration |
| **Multi-GPU constraint relaxed** | Deep-ensemble teachers become affordable; teacher-side uncertainty signal is direct (variance) not extracted; promote at next milestone |
| **Downstream paradigm requires calibration** | If a #82+ candidate explicitly depends on calibrated uncertainty (e.g., safety-margin reasoning, active-learning data selection, refusal calibration), promote BAYESIAN-LLM as enabling sub-paradigm |
| **NLL-only constraint sharpening** | If user re-tightens iter-220 broadening back toward strict NLL preservation, BAYESIAN-LLM's NLL-preserving aux-head property gains relative weight |
| **Iter-225+ axis-novelty pressure** | If 5+ consecutive iterations produce only architecture-version-upgrade or borderline-microopt candidates, BAYESIAN-LLM's axis-novelty becomes a higher-ranked tiebreaker |

---

## 11. Composition matrix update (if hypothetically promoted)

```
Cumulative axis count post-#81 (hypothetical):
  Training axes:                 8 (text NLL, reasoning, tool, agent, ...)
  Modality axes:                 2 (vision #71, audio #80)
  Inference / arch axes:         5 (INFERENCE_SPEED, KV-COMPRESSION,
                                    MODEL-SIZE, CONTEXT-LENGTH, DEPTH-ROUTING)
  Teacher-provenance axes:       5 (text, reasoning, tool, multimodal, language)
  **NEW: UNCERTAINTY axis:       1 (calibrated probability as first-class output)**
  Total:                         21 axes (was 20 post-#80)
```

After 41 paradigms across 21 axes, the bigger-picture stack would have reframed UNCERTAINTY as the 21st training-time-and-inference-time signal. **This framing is genuinely new** — no prior axis treats calibrated probability as first-class output. The reserved status reflects mechanism risk, not axis-novelty risk.

**Iter-225 verdict on BAYESIAN-LLM-DISTILL: RESERVE pending C1 validation, multi-GPU signal, or downstream paradigm dependency.**
