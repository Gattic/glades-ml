# Paradigm Shift #97 — Candidate A: DRAFT-VERIFIER-CO-LEARN-DISTILL

**Status:** SELECTED. **Extends shipped #75 SPECULATIVE-DECODING via continuous draft-from-rejection co-learning.**
**Date:** 2026-05-08 (Ralph-loop iter 241, sixth iteration under iter-236 brief change).
**Axis:** **INFERENCE_SPEED** (extends #75; same axis, deeper mechanism).
**Magnitude target:** **1.2-1.7× incremental over #75 standalone**, lifting joint inference throughput from 3-5× (post-#75) to **4-6× (post-#97)** at greedy/temperature-1 sampling. NLL bit-exact at inference (rejection sampling preserves main model's distribution by construction; #75 Theorem 3.5 inheritance).

---

## 0. Executive summary

Iter-241 maintains the iter-236+ alternation pattern (#92 operational → #93 novel → #94 operational → #95 novel → #96 operational → #97 novel). The candidate extends #75 SPECULATIVE-DECODING-DISTILL — the only INFERENCE_SPEED paradigm shipped so far (iter-219) — by adding a continuous-learning loop that turns the draft model's rejection trace into a perpetually-refreshing distillation signal.

**Mechanism in one sentence:** Every time the main verifier rejects a draft token, the (context, draft-prediction, main-prediction) triple is logged; periodically, the draft is fine-tuned on these rejection examples so it gradually stops repeating its rejected mistakes and the acceptance rate climbs from ~0.7 toward an empirical asymptote near 0.85.

**Why this fits iter-236 brief:**
- **Novel mechanism**: rejection-pattern distillation as a training signal for the draft. No production reference at this exact framing — Eagle, Medusa, and Eagle-2 all use *static* draft heads; continuous refresh from rejection-only signal is new.
- **Built-in 1-day Gate-0 (~6 GPU-hours)**: short enough to fit a single workday; tests the central premise (acceptance rate measurably improves from rejection-only data) at the smallest scale that still validates the mechanism.
- **Bold testable claim**: ≥0.05 absolute acceptance-rate lift in 5k draft-fine-tune steps over 50k-token rejection log.

**Honest framing (acknowledged up-front):**
- **Modest magnitude (1.2-1.7× incremental)** — borderline microopt by program's 10-30× bar; acceptance is on iter-236 brief alignment grounds (testable bold claim with 1-day Gate-0), not magnitude.
- **Speculative production status**: rejection-pattern distillation as continuous draft-training has no published precedent at production scale; only the *static-draft-distillation* substrate (#75) has overwhelming production validation.
- **Deployment-time complexity**: the co-learning loop runs on a live inference path. Periodic draft updates require checkpoint-swap protocol; rollback path is mandatory if a draft update *degrades* acceptance.
- **Joint Gate-0 PASS ~50-60%**; LLM-scale empirical confirmation **~40-55% conditional**; production-viable **~25-35%**.

**Engineering:** ~900 LOC over 4 weeks (rejection logger + co-learn trainer + checkpoint-swap protocol + standalone-NLL guardrail). Reuses #68 SUPER-DISTILL cached-logit pipeline and #75 draft+verify infrastructure as stack-base.

---

## 1. Candidate selection

| Candidate | Verdict |
|---|---|
| **A — DRAFT-VERIFIER-CO-LEARN-DISTILL** | **SELECTED (novel mechanism + built-in 1-day Gate-0; modest magnitude acknowledged)** |
| B — HUTCH-DIAG-V-PROJECTION-DISTILL | RESERVE (continues; microopt; reservation now at 5 iterations) |
| C — INFERENCE-CACHE-WARMING | RESERVE (overlaps #69; deferred until #97 settles) |

**A selected on three grounds:**

1. **Novel mechanism fitting iter-236 brief.** Rejection-pattern distillation as continuous draft training is genuinely new framing within the SPECULATIVE-DECODING line — distinguishes itself from Eagle / Medusa / Eagle-2 by treating the rejection log as a *first-class training signal* rather than a debugging artifact.

2. **Built-in 1-day Gate-0 (~6 GPU-hours)** with hard-FAIL signals. The probe tests the *central premise* (acceptance rate lift from rejection-only data) at the smallest possible scale that preserves mechanism. If the premise is wrong, the candidate is rejected within 6 hours.

3. **Honest about modest magnitude.** 1.2-1.7× incremental over #75 standalone is borderline microopt — but the iter-236 brief explicitly admits this trade in exchange for testable bold claims. The candidate doc surfaces the trade-off at the top rather than burying it.

**B continues at 5 iterations of reservation.** HUTCH-DIAG-V-PROJECTION-DISTILL has been deferred at iter-237/238/239/240/241; pure microopt with no built-in 1-day Gate-0 differentiator. Closure recommendation deferred to #98.

**C INFERENCE-CACHE-WARMING** overlaps #69 KV-FACE-MERGED at the cache-management mechanism. Reserved until #97 settles to avoid axis-collision.

---

## 2. Mechanism

### 2.1 Standard #75 baseline (recap)

```
For each generation step:
  1. Draft (200M): autoregressively propose K=4-8 tokens
  2. Main (1.84B → 32B-effective post-#74): single forward pass over K-token context, parallel
  3. Rejection sampling: accept tokens up to first disagreement; resample at first disagree
  4. Net throughput: 3-5× over main-only at greedy/temp=1 (Leviathan 2023; vLLM precedent)
```

Acceptance rate `α ≈ 0.7` at #75 ship (post-iter-219). Higher α → more tokens accepted per main forward → higher speedup; theoretical ceiling at α=1 is K× speedup.

### 2.2 #97 extension: rejection-pattern co-learn loop

```
At each generation step (in addition to #75):
  for k in 0..K-1:
    if reject(draft_logits[k], main_logits[k]):
      log(context_k, draft_pred=draft_argmax[k], main_pred=main_argmax[k],
          draft_logits[k], main_logits[k])
      break  # standard #75 rejection break

Periodically (every N rejection samples):
  1. Pull rejection log (last N samples; default N=50,000).
  2. Train draft: L = α · CE(draft, main_argmax) + (1-α) · τ² · KL(draft || main_logits).
     Same loss family as #68 SUPER-DISTILL but restricted to rejection-only training set.
  3. Standalone-NLL guardrail: evaluate draft on held-out NLL benchmark;
     if NLL increases > 0.02 nat absolute, ROLLBACK draft to previous checkpoint.
  4. Checkpoint-swap protocol: atomic swap of draft weights at next inference batch boundary.
```

### 2.3 Why rejection-only data is the right signal

**Hypothesis (testable in Gate-0):** rejection sites are exactly where draft and main *disagree*. Training the draft on the main's argmax at these sites is the highest-information-density signal for closing the gap. Random non-rejection contexts are uninformative — the draft already matches main there.

**Mathematical framing.** Let `A(α)` = expected tokens accepted per main forward at draft acceptance rate α. By Leviathan 2023 Eq. 1:

```
A(α) = (1 - α^(K+1)) / (1 - α)
```

At K=4: A(0.70) ≈ 2.83; A(0.85) ≈ 3.71. Net throughput lift from α=0.70 → α=0.85: **3.71 / 2.83 ≈ 1.31×** at K=4. At K=8 (more aggressive draft): A(0.70) ≈ 3.32; A(0.85) ≈ 5.21; **lift ≈ 1.57×**. This bounds the magnitude target at 1.2-1.7× incremental.

### 2.4 Differentiation from prior speculative-decoding refinements

| Refinement | Source | Mechanism | Differentiation from #97 |
|---|---|---|---|
| **Eagle (Li 2024)** | Self-speculative draft head | Draft = lightweight head on main | Static after training; no continuous refresh from rejection log |
| **Eagle-2 (Li 2024b)** | Refined Eagle with dynamic tree | Tree expansion at inference | Tree-search axis; orthogonal to #97's training axis |
| **Medusa (Cai 2024)** | Multi-head decoding | Multiple draft heads on main | Static heads; no rejection-driven refresh |
| **#75 baseline (iter-219)** | Co-distilled 200M draft | Train draft from teacher KL | One-shot training; no continuous-learning loop |
| **#97 (iter-241)** | Rejection-driven co-learn | Continuous draft fine-tune from rejection log | **NEW**: rejection log as primary training signal |

### 2.5 Composition

| Paradigm | Composes? |
|---|---|
| **#75 SPECULATIVE-DECODING** | ✓ Stack-base (draft+verify substrate; rejection log is a free byproduct of #75's rejection sampling) |
| **#68 SUPER-DISTILL** | ✓ Cached-logit pipeline + KL-CE loss family directly reused |
| **#74 PHOENIX-1BIT-DISTILL-COMBO** | ✓ Quantized main + quantized draft both unaffected; rejection sampling on quantized logits is identical |
| **#73-#83 multi-modal teacher portfolio** | ✓ Per-modality draft refinement is a natural follow-on (deferred to #98+) |
| **#84-#91 model-size paradigms** | ✓ Independent of model-size axis |

---

## 3. Theoretical analysis

### 3.1 Acceptance rate improvement bound (Theorem 1)

**Claim.** Under continuous co-learning on rejection-only samples, draft acceptance rate `α_t` is monotonically non-decreasing in expectation, with asymptote bounded by the *information-theoretic gap* between draft and main capacities.

**Proof sketch.** Each fine-tune step minimizes `D_KL(draft || main)` restricted to rejection sites; by gradient-descent convergence theorems for convex distillation losses, expected `D_KL` is non-increasing. Rejection rate at site `x` is monotone in `D_TV(draft || main)(x)` (total-variation distance bounds rejection probability; Leviathan Lemma 4.1). Hence expected acceptance rate is non-decreasing. Asymptote bounded by Pinsker inequality: `D_TV² ≤ ½ · D_KL`, where `D_KL` floor is set by the 200M-vs-32B-effective capacity gap. ∎

**Caveat.** The proof assumes rejection log is *representative* of inference distribution. If the deployment-time prompt mix shifts (distribution drift), the floor on acceptance rate can rise rather than fall. **Continuous-learning loop addresses this naturally** — drift in prompt mix → drift in rejection patterns → drift in draft fine-tune signal.

### 3.2 NLL preservation at inference (Theorem 2)

**Claim.** Inference output distribution is bit-exact main model distribution, regardless of draft state.

**Proof.** Inherits directly from #75 / Leviathan 2023 Theorem 3.5 (rejection sampling preserves target distribution). Draft fine-tuning changes which tokens are *accepted*, not which tokens *can be output* — the resampling-from-main step preserves main's distribution at every rejection site. ∎

This is a *strong* property: the draft can be arbitrarily bad (in the limit, α=0 reduces to main-only autoregressive decoding), and inference output remains bit-exact.

### 3.3 Standalone-NLL degradation risk (Conjecture 1)

**Hazard.** Continuous fine-tuning on rejection-only data can drift the draft *away* from a generally-useful distribution. If the user re-purposes the draft (e.g., as a lightweight fallback when verifier is offline), the draft's standalone NLL may degrade.

**Mitigation:** standalone-NLL guardrail in the co-learn loop. Every fine-tune cycle evaluates draft on a held-out NLL benchmark; if NLL increases >0.02 nat absolute, rollback to previous checkpoint.

**Open question:** what fraction of co-learn cycles will trip the guardrail? Gate-0 measures this directly.

### 3.4 Joint Gate-0 PASS probability decomposition

```
Rejection logging implementation (low-risk; ~95%):           ~95%
Cached-logit fine-tune pipeline reuse (#68 substrate):       ~92%
Acceptance rate ≥ +0.05 absolute over baseline:              ~65%
Draft standalone NLL not degraded > 0.02 nat:                ~75%
Checkpoint-swap protocol stable across 5+ swaps:             ~80%
Asymptote behavior consistent with Theorem 1 over 5k steps:  ~70%

Joint Gate-0 PASS:                                           ~50-60%
LLM-scale empirical confirmation:                            ~40-55% conditional
Production-viable (joint, with guardrails operational):      ~25-35%
```

---

## 4. Built-in 1-day Gate-0

### 4.1 Probe spec (~6 GPU-hours)

**Setup.**
- 200M draft + 1.84B main (post-#75 baseline; not the larger 32B-effective composite, to keep probe ≤ 6 GPU-hours).
- Main and draft both in BF16 (no PHOENIX quantization; isolate the co-learn signal).
- Inference dataset: 50,000 token-generations on standard prompt mix (Pile-CC subset + ShareGPT subset; uniform 50/50).

**Phase 1 — Baseline measurement (1 GPU-hour).**
- Run 50,000-token inference with #75 draft+verify, draft frozen.
- Log baseline acceptance rate `α_baseline`. Expected ~0.70.
- Log standalone draft NLL on held-out benchmark.

**Phase 2 — Rejection log accumulation (2 GPU-hours).**
- Run 50,000-token inference, draft frozen, log every rejection site (context, draft-pred, main-pred, draft-logits, main-logits).
- Expected rejection count: 50,000 × (1 - α_baseline) × K ≈ 50,000 × 0.3 × 4 ≈ 60,000 rejection sites.
- Storage: ~2 GB rejection log (uncompressed BF16 logits).

**Phase 3 — Co-learn fine-tune (2 GPU-hours).**
- Train draft for 5,000 steps on rejection-only dataset.
- Loss: L = 0.5 · CE(draft, main_argmax) + 0.5 · τ² · KL(draft || main_logits) at τ=2.0.
- Optimizer: AdamW, lr=1e-5 (1/30 of pretrain lr; conservative for fine-tune).

**Phase 4 — Re-evaluate (1 GPU-hour).**
- Run 50,000-token inference with co-learned draft.
- Log new acceptance rate `α_post`.
- Log standalone draft NLL on held-out benchmark; verify ≤ baseline + 0.02 nat.

### 4.2 PASS criterion

**Strong PASS:** α_post ≥ α_baseline + 0.075 absolute (e.g., 0.70 → 0.775+) AND draft standalone NLL ≤ baseline + 0.02 nat → proceed to Gate-1 full-tier (32B-effective main + extended co-learn cycles).

**PASS:** α_post ≥ α_baseline + 0.05 absolute (e.g., 0.70 → 0.75+) AND draft standalone NLL ≤ baseline + 0.02 nat → proceed to Gate-1 conditional.

### 4.3 Hard-FAIL signals (with abort protocols)

| Signal | Abort time | Action |
|---|---|---|
| **No improvement** (α_post < α_baseline + 0.02 absolute) | After Phase 4 (full 6h) | Reject candidate; close rejection-pattern-distillation variant |
| **Draft NLL degrades > 0.05 nat** | Mid-Phase 3 (3h) | Abort fine-tune; reject candidate; standalone NLL too sensitive to rejection-only training |
| **Acceptance rate REGRESSES** (α_post < α_baseline) | Mid-Phase 4 (5h) | Hard reject; rejection-only data actively harmful |
| **Loss diverges** in fine-tune | Mid-Phase 3 (3h) | Tune lr down 5×; one retry; if still diverges, reject |

### 4.4 Decision tree

```
Strong PASS  → Gate-1 full-tier; build #97; 4-week engineering plan
PASS         → Gate-1 conditional; assess deployment-time complexity before commit
No-improve   → Reject; document rejection log as null result
NLL-degrade  → Reject; rejection-only training too narrow; revisit with mixed-data co-learn (deferred #98)
Regression   → Hard reject; close variant
```

---

## 5. Updated cumulative stack

```
Iter 240 close (post-#96):
  All 27 axes ≈preserved
  INFERENCE_SPEED axis at 3-5× via #75 (shipped iter-219)
  Operational paradigms #92, #94, #96 in place

Iter 241 candidate (DRAFT-VERIFIER-CO-LEARN-DISTILL):
  All 27 axes ≈preserved (no new axis; deepens INFERENCE_SPEED)
  Conditional on Gate-0 PASS:
    INFERENCE_SPEED axis lifted from 3-5× (#75) to 4-6× (#75+#97)
    Joint cumulative throughput: prior × 1.3-1.6× (mid-band)
```

**No new axis.** #97 is a depth-extension of #75's INFERENCE_SPEED axis, not a new orthogonal lever. This is consistent with the iter-209+ pattern in which post-axis-saturation paradigms deepen existing axes rather than open new ones.

---

## 6. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Rejection logger (free byproduct of #75 rejection sampling; minor wiring) | 100 | 0.5 |
| Cached-logit pipeline reuse (#68 substrate) | 50 | 0.25 |
| Co-learn trainer (rejection-only fine-tune; KL-CE loss; AdamW) | 250 | 1 |
| Standalone-NLL guardrail (held-out eval + rollback trigger) | 150 | 0.75 |
| Checkpoint-swap protocol (atomic draft swap at batch boundary) | 200 | 1 |
| Gate-0 probe runner + 4-phase orchestration | 100 | 0.5 |
| Evaluation harness (acceptance-rate tracking + NLL drift + asymptote curves) | 50 | 0.25 |
| **Total** | **~900** | **4** |

Reuse from #68 (cached-logit pipeline) and #75 (draft+verify substrate) keeps LOC low. New surface area concentrated in **standalone-NLL guardrail** and **checkpoint-swap protocol** — these are the deployment-time complexity surfaces.

---

## 7. Memory advantage preservation

| Component | Memory delta |
|---|---|
| Rejection log buffer (in-memory ring, 50,000 sites × ~40 KB BF16 logits) | +2 GB host RAM |
| Cached-logit pipeline (per-class teacher; reuse #68) | +0 GB (already accounted) |
| Draft fine-tune Adam state (200M × 8 bytes BF16 + Adam) | +1.6 GB host (offloaded; iter-208 CPU-OFFLOAD-ADAM substrate) |
| Checkpoint-swap (2× draft slots for atomic swap) | +0.8 GB GPU during swap window only |
| **Total** | **2 GB host RAM steady; 0.8 GB GPU swap-window-only** |

**Single-GPU 16 GB ceiling preserved.** Swap window is ~100 ms; well within batch boundary tolerance.

---

## 8. Honest gaps

1. **Modest magnitude (1.2-1.7× incremental over #75)** — borderline microopt by program's 10-30× bar. Acceptance is on iter-236 brief alignment grounds (testable bold claim with 1-day Gate-0), explicitly trading magnitude for novel-mechanism testability.

2. **No production reference at this exact framing.** Eagle, Medusa, Eagle-2 use *static* draft heads. Continuous rejection-driven refresh is genuinely new — production-viability uncertain.

3. **Deployment-time complexity (continuous learning loop on inference path).** Live model updates are operationally heavier than static draft. Checkpoint-swap protocol must be airtight; rollback path mandatory.

4. **Standalone-NLL degradation hazard.** Rejection-only training data is narrow. Guardrail addresses it but adds cost (held-out NLL eval per fine-tune cycle).

5. **Distribution-drift sensitivity.** If deployment prompt mix shifts faster than co-learn cycle, draft chases a moving target; acceptance rate can oscillate. Not measured in 1-day Gate-0.

6. **Asymptote uncertainty.** Theorem 1 bounds asymptote but does not predict it. Empirical asymptote on real workloads is unknown until LLM-scale Gate-1.

7. **Joint Gate-0 PASS ~50-60%** — moderate uncertainty. Production-viable ~25-35%; below #75's ~80% confirmation but justified by built-in 1-day Gate-0 cheap test.

8. **Iter-200 microopt critique applies.** 1.2-1.7× incremental falls below the program's 10-30× ceiling. Mitigated by built-in 1-day Gate-0 enabling fast iterative refinement and rejection.

---

## 9. Bottom line

**DRAFT-VERIFIER-CO-LEARN-DISTILL extends shipped #75 SPECULATIVE-DECODING via continuous draft fine-tuning on the rejection-only signal.** Mechanism turns the rejection log — currently discarded after each generation step — into a perpetually-refreshing distillation signal. The bold claim (≥0.05 absolute acceptance-rate lift in 5,000 fine-tune steps over 50,000-token rejection log) is directly testable in a built-in 1-day Gate-0 (~6 GPU-hours) with hard-FAIL signals.

**Cumulative single-GPU stack at iter-241 close (conditional on Gate-0 PASS):**
- All 27 prior axes ≈preserved
- INFERENCE_SPEED axis lifted from 3-5× (#75 alone) to **4-6× joint (#75+#97)**
- Magnitude lift is modest (1.2-1.7× incremental); acknowledged up-front

**Engineering:** ~900 LOC over 4 weeks. **Joint Gate-0 PASS ~50-60%; LLM-scale empirical confirmation ~40-55% conditional; production-viable ~25-35%.**

**Built-in 1-day Gate-0 (~6 GPU-hours)** with four hard-FAIL signals (no-improvement, NLL-degradation, acceptance-rate-regression, loss-divergence). Decision tree provides clean continuation/abort path within a single workday.

**B (HUTCH-DIAG-V-PROJECTION-DISTILL) reservation now at 5 iterations** — closure recommendation deferred to #98. **C (INFERENCE-CACHE-WARMING) reserved on overlap with #69.**

After 56 paradigms, **27 axes** unchanged (INFERENCE_SPEED axis depth-extension only, no new axis). Iter-236-241 pattern: 6 paradigms under brief change; alternation between operational (#92, #94, #96) and novel-with-built-in-test (#93, #95, #97) holds.

**Verdict: SELECT.** Modest magnitude is acknowledged; iter-236 brief alignment (testable bold claim with 1-day Gate-0) drives selection. If Gate-0 fails within 6 GPU-hours, the candidate is rejected at low total cost — exactly the iteration-economy property the brief change was designed to encourage.
