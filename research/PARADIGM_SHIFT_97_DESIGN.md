# Paradigm Shift #97 — DRAFT-VERIFIER-CO-LEARN-DISTILL: Continuous Speculative Refinement

**Status:** SELECTED. **Sixth paradigm under iter-236 brief change; alternation pattern continues** (3 operational + 3 novel-with-test).
**Date:** 2026-05-08 (Ralph-loop iter 241).
**Axis:** INFERENCE_SPEED (depth-extension of #75 SPECULATIVE-DECODING). No new axis.
**Magnitude target:** **4-6× joint inference throughput** (lift #75's 3-5× by +1.2-1.7×). Modest magnitude acknowledged.

---

## 0. Executive summary

Iter-241 alternation pattern: novel-with-built-in-test paradigm following operational #96. Mechanism extends shipped #75 SPECULATIVE-DECODING with continuous co-learning from rejection patterns.

**Mechanism:** During inference, rejected draft tokens are logged with (context, draft-prediction, main-prediction) tuples. After accumulating rejection log, draft model is fine-tuned on rejection patterns to improve acceptance rate. Continuous loop: deployment → rejection log → draft refresh → improved acceptance → repeat.

**Bold testable claim:** Continuous co-learning improves acceptance rate from baseline ~0.7 to ~0.85 over training; lifts #75's 3-5× to 4-6× joint inference throughput.

**Built-in 1-day Gate-0 (~6 GPU-hours; 4-phase orchestration):**
1. **Phase 1 (1h):** baseline α and standalone NLL on frozen draft.
2. **Phase 2 (2h):** rejection log accumulation (~60,000 sites, ~2 GB BF16 logits).
3. **Phase 3 (2h):** 5,000-step draft fine-tune on rejection-only data; KL-CE at τ=2.0; AdamW lr=1e-5.
4. **Phase 4 (1h):** re-evaluate α_post and NLL drift.

**PASS criterion:** α_post ≥ α_baseline + 0.05 absolute AND draft NLL ≤ baseline + 0.02 nat.
**Strong PASS:** ≥ +0.075 absolute.
**Hard FAIL signals (4):**
- No improvement (full 6h).
- NLL degradation > 0.05 nat (3h abort).
- Acceptance-rate regression (5h abort).
- Loss divergence (3h abort with 1 retry).

**Honest framing:**
- **Modest magnitude** 1.2-1.7× over #75 (borderline iter-200 microopt).
- **Deployment-time complexity** — continuous-learning loop, checkpoint-swap, rollback.
- **No exact production precedent** for rejection-driven continuous refresh (vs static Eagle/Medusa/Eagle-2).
- **Joint Gate-0 PASS ~50-60%; production-viable ~25-35%.**
- **Justification per iter-236 brief**: novel mechanism with built-in 1-day Gate-0 + hard-FAIL aborts.

**Engineering:** ~900 LOC over 4 weeks.

---

## 1. Candidate selection

| Candidate | Verdict |
|---|---|
| **A — DRAFT-VERIFIER-CO-LEARN-DISTILL** | **SELECTED (novel; alternation pattern; testable)** |
| B — HUTCH-DIAG-V-PROJECTION-SUNSET | RESERVE-AS-RECOMMENDATION (housekeeping; could sunset at iter-242) |
| C — KV-FACE-MLA-LIVE-PROBE | RESERVE (housekeeping; execute existing #90 plan) |

A selected on three grounds:
1. **Novel mechanism** (continuous rejection-driven co-learning).
2. **Built-in 1-day Gate-0** with 4 hard-FAIL aborts.
3. **Continues alternation pattern** (operational #96 → novel #97).

---

## 2. Mechanism

### 2.1 Standard #75 SPECULATIVE pipeline

For each inference query:
1. Draft generates K candidate tokens.
2. Main verifies in parallel; rejects via rejection sampling.
3. Output token comes from main's distribution (Theorem 3.5 of Leviathan 2023).

### 2.2 Co-learning extension (#97)

At each rejected token:
```
log_entry = (context_t, draft_prob_dist_t, main_prob_dist_t)
```

After accumulating ~60K rejections:
```
For 5K steps:
  Sample (context, draft_dist, main_dist) from rejection log
  L_t = α · CE(draft_proposed, main_argmax) + (1-α) · τ² · KL(draft_softmax(τ) || main_softmax(τ))
  Update draft via AdamW lr=1e-5
```

**Result:** draft converges toward main's distribution at rejection sites; acceptance rate increases.

### 2.3 Composition

| Paradigm | Composes? |
|---|---|
| **#75 SPECULATIVE-DECODING** | ✓ Stack-base |
| **#68 SUPER-DISTILL** | ✓ Cached-logit pipeline reused for rejection-log distillation |
| **#74 PHOENIX-1BIT** | ✓ PHOENIX-quantized main+draft unaffected by rejection sampling |

---

## 3. Theoretical analysis

### 3.1 Acceptance rate convergence

**Claim.** Under continuous co-learning, acceptance rate α converges to fixed point α* > α_initial.

**Argument.** Each co-learn round reduces TVD(p_draft, p_main); per Theorem 3 of #75, α ≥ 1 - TVD; therefore α increases. Steady state: α* = 1 - ε where ε is irreducible draft-capacity gap.

**Empirical estimate:** α_initial ≈ 0.7; α* ≈ 0.85 (limited by 200M draft capacity vs 1.84B main).

### 3.2 NLL bit-exact preservation at inference

Per #75 Theorem 1: rejection sampling preserves p_main exactly. **NLL bit-exact at inference; co-learn only modifies draft distribution.**

### 3.3 Joint Gate-0 PASS probability

```
Phase 1 baseline α measurement:                      ~99%
Phase 2 rejection log accumulation:                  ~95%
Phase 3 draft fine-tune convergence:                 ~85%
Phase 4 acceptance rate improvement ≥ +0.05:         ~65%

Joint Gate-0 PASS:                                   ~50-60%
LLM-scale empirical confirmation:                    ~50-65% conditional
Production-viable:                                   ~25-35%
```

---

## 4. Built-in 1-day Gate-0

(See §0 4-phase orchestration.)

**Total budget: ~6 GPU-hours.** Hard FAIL aborts at 3-5 GPU-hours.

---

## 5. Updated cumulative stack

```
Iter 240 close (post-#96):
  All 27 axes ≈preserved
  Validation: top-15 paradigms scheduled

Iter 241 (DRAFT-VERIFIER-CO-LEARN-DISTILL):
  All 27 axes ≈preserved (no new axis)
  Conditional on Gate-0 PASS: inference 3-5× (#75) → 4-6× (joint #75 + #97)
```

---

## 6. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Rejection log accumulator (during inference) | 200 | 1 |
| Co-learn pipeline (training on rejection log) | 250 | 1 |
| Phase orchestration + checkpoint management | 200 | 0.75 |
| Gate-0 probe runner | 150 | 0.5 |
| Evaluation harness (acceptance rate; NLL drift) | 100 | 0.75 |
| **Total** | **~900** | **4** |

---

## 7. Memory advantage preservation

| Component | Memory |
|---|---|
| Rejection log buffer (~60K entries × ~32 KB BF16 logits) | ~2 GB host (during accumulation) |
| Draft fine-tune Adam state | ~50 MB additional GPU (during co-learn phase only) |
| **Total** | **~2 GB host transient; ~50 MB GPU during co-learn** |

**Single-GPU 16 GB ceiling preserved.**

---

## 8. Honest gaps

1. **Modest magnitude** 1.2-1.7× over #75; iter-200 microopt threshold.
2. **Deployment-time complexity** — continuous-learning loop, rollback policy.
3. **No exact production precedent** for rejection-driven continuous refresh.
4. **Production-viable ~25-35%** moderate uncertainty.
5. **Iter-241 microopt continues iter-236+ pattern** of below-the-bar paradigms.

---

## 9. Bottom line

**DRAFT-VERIFIER-CO-LEARN-DISTILL extends #75 SPECULATIVE with continuous co-learning.** Modest magnitude lift 1.2-1.7× joint with #75; built-in 1-day Gate-0 with 4 hard-FAIL aborts.

**Cumulative single-GPU stack at iter-241 close:**
- All 27 prior axes ≈preserved
- Conditional on Gate-0 PASS: inference 4-6× joint with #75 (vs 3-5× #75 alone)

**Engineering:** ~900 LOC over 4 weeks. **Joint Gate-0 PASS ~50-60%; LLM-scale conditional ~50-65%; production-viable ~25-35%.**

**B and C reserved.** B HUTCH-DIAG-V-PROJECTION-SUNSET would formally close quadruple-reservation pattern; deferred. C KV-FACE-MLA-LIVE-PROBE is housekeeping for #90 reservation.

After 57 paradigms, **27 axes** unchanged.

**Iter-236+ pattern (6 paradigms):**
- Operational: #92, #94, #96 (tier-1, tier-2, tier-3 campaigns).
- Novel-with-test: #93, #95, #97 (ASTRA-KAHAN, MULTI-TEACHER-ROUTING, DRAFT-VERIFIER-CO-LEARN).

Pattern alternates predictably; iter-242+ will likely be operational (#96-style campaign) or another novel-with-test.
