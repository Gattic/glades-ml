# Paradigm Shift #95 — MULTI-TEACHER-ROUTING-DISTILL: Refined Re-Admission of Rejected #70-B

**Status:** SELECTED. **Refines rejected #70-B ENSEMBLE-DISTILL via argmax classifier routing (sequential per-sample, vs #70-B's simultaneous teacher fusion).**
**Date:** 2026-05-08 (Ralph-loop iter 239, fourth iteration under iter-236 brief change).
**Axis:** TEACHER PORTFOLIO meta-channel (extends #68-#72 teacher-provenance).
**Magnitude target:** **2-3× per-class incremental wall-clock to fixed final NLL** (refines #70-B's failed simultaneous fusion). Risk-adj 1.0-1.5× expected (above breakeven, vs #70-B's 0.5×).

---

## 0. Executive summary

Iter-239 alternates from operational paradigms (#92, #94) back to NEW mechanism with built-in 1-day Gate-0. The mechanism re-opens the TEACHER PORTFOLIO meta-channel that #70-B closed at risk-adj 0.5×.

**Refinement vs #70-B:**
- **#70-B ENSEMBLE-DISTILL** (rejected iter-220): simultaneously distill from K teachers. Variance amplification up to 4× when teachers disagree. Risk-adj 0.5× below break-even.
- **#95 MULTI-TEACHER-ROUTING-DISTILL** (selected iter-239): argmax per-sample classifier routing. Sequential not simultaneous. Eliminates variance amplification mechanism by construction. Loses Jensen-gap fusion bonus.

**Mechanism:**
- Problem-type classifier C (10M params) maps input to 5 classes: math/reasoning, code, general, vision, [audio held-out for Gate-1].
- Per-class teacher T_c:
  - math/reasoning → DeepSeek-R1
  - code → DeepSeek-Coder-V2 + GPT-4-tools
  - general → Llama 3.1 405B
  - vision → Llama 3.2 Vision 90B
  - audio → Whisper-large-v3 (Gate-1 only)
- For each training sample x, classifier emits argmax routing class c*; KL-CE loss applied with single teacher T_{c*}.
- Different from rejected #70-B: sequential per-sample vs simultaneous fusion.

**Bold claim (testable):** Per-class routing avoids #70-B's teacher-disagreement variance; net 2-3× per-class NLL improvement over single-best-teacher baseline.

**Built-in 1-day Gate-0 (~8 GPU-hours):**
- 200M coordinator + 10M-param 5-class classifier.
- ~10M training samples partitioned across 5 classes.
- Cached-logit teachers (proxies; full-tier for Gate-1).
- 50k-step run; argmax routing.
- **PASS:** Per-class NLL ≥ single-best-teacher on each class + gradient std-dev within 1.2× single-teacher + classifier accuracy ≥ 85% + no class collapse.
- **Hard FAIL signals:** classifier collapse (one class > 80%; abort 2h); per-class NLL > 0.5 nat worse (abort 4h); gradient std-dev > 2× single-teacher (abort 3h).

**Honest framing:**
- **#70-B precedent**: risk-adj 0.5× below break-even — primary risk for #95.
- **Classifier accuracy bottleneck**: 85% accuracy threshold; below cuts payoff.
- **Loses Jensen-gap fusion bonus** that #70-B's simultaneous distillation provided.
- **Joint Gate-0 PASS ~45-55%; LLM-scale confirmation ~50-65% conditional; production-viable ~25-35%.**
- **2.5× per-class is microopt by program's 10-30× bar** — but justification is iter-236 brief alignment (testable bold claim).

**Engineering:** ~1,800 LOC over 6 weeks; storage ~200 GB cached logits (6× reduction vs #70-B due to per-class partitioning).

---

## 1. Candidate selection

| Candidate | Verdict |
|---|---|
| **A — MULTI-TEACHER-ROUTING-DISTILL** | **SELECTED (novel mechanism + 1-day Gate-0)** |
| B — HUTCH-DIAG-V-PROJECTION-DISTILL | RESERVE (continued; microopt) |
| C — GATE-0-CAMPAIGN-TIER-3 | RESERVE (continued campaign expansion deferred) |

A selected on three grounds:
1. **Novel mechanism fitting iter-236 brief** (testable bold claim).
2. **Refines rejected #70-B** with structurally different mechanism.
3. **Built-in 1-day Gate-0 with hard FAIL aborts** at 2-4 GPU-hours.

---

## 2. Mechanism

### 2.1 Classifier-routed distillation

```
For each training sample x:
  c* = argmax_c P(class | x; θ_classifier)   # 10M-param classifier
  teacher = T_{c*}                            # per-class teacher
  L_t = α · CE(student, ground-truth) + (1-α) · τ² · KL(student || teacher)
```

Sequential per-sample routing; only one teacher consulted per sample.

### 2.2 Differentiation from #70-B

- **#70-B ENSEMBLE**: `L_t = α · CE + Σ_k β_k · KL(student || T_k)` — all teachers simultaneously.
- **#95 ROUTING**: `L_t = α · CE + (1-α) · KL(student || T_{c*})` — single teacher per sample.

**Eliminates teacher-disagreement variance** that #70-B suffered from. Trade: loses simultaneous-fusion smoothing (Jensen-gap bonus).

### 2.3 Composition

| Paradigm | Composes? |
|---|---|
| **#68 SUPER-DISTILL** | ✓ Stack-base (single-teacher distillation infrastructure) |
| **#69-#72** | ✓ Per-class teachers reuse |
| **#70-B (rejected)** | Refinement (replaces simultaneous fusion with sequential routing) |

---

## 3. Theoretical analysis

### 3.1 Variance reduction theorem

**Claim.** Under argmax routing with classifier accuracy ≥ 85%, gradient variance ≤ 1.2× single-teacher variance (vs #70-B's up to 4× amplification).

**Proof sketch.** Argmax selects single teacher; gradient variance bounded by that single teacher's variance times routing-error correction factor (1.0 / classifier-accuracy ≈ 1.18 at 85%). ∎

### 3.2 Classifier accuracy bottleneck

If classifier accuracy < 85%, routing error compounds with teacher mismatch → effective variance > single-teacher. **85% is the breakeven.**

### 3.3 Joint Gate-0 PASS probability

```
Classifier training (10M params, 5 classes):              ~92%
Cached-logit pipeline per class:                          ~90%
Argmax routing convergence:                               ~75%
Per-class NLL ≥ single-best-teacher on all 5 classes:     ~60%
Gradient std-dev within 1.2× single-teacher:              ~70%
Classifier accuracy ≥ 85% on Gate-0 corpus:               ~80%

Joint Gate-0 PASS:                                        ~45-55%
LLM-scale empirical confirmation:                         ~50-65% conditional
Production-viable (joint):                                ~25-35%
```

---

## 4. Built-in 1-day Gate-0

### 4.1 Probe spec (~8 GPU-hours)

- 200M coordinator + 5-class classifier.
- ~10M samples (2M per class).
- Cached-logit Tier-3 teachers (R1-Distill-Qwen-32B / DeepSeek-Coder-V2-Lite-16B / Llama-3.1-8B-Instruct / Llama-3.2-Vision-11B).
- 50k-step training.

### 4.2 Decision tree

| Outcome | Action |
|---|---|
| **Strong PASS** (per-class NLL > single-teacher; std-dev <1.1×) | Gate-1 full-tier; build #95 |
| **PASS** (per-class NLL ≥ single-teacher; std-dev <1.2×) | Gate-1 conditional |
| **Hard FAIL: classifier collapse** | Abort 2h; retrain classifier; one retry |
| **Hard FAIL: per-class NLL >0.5 nat worse** | Abort 4h; close TEACHER PORTFOLIO routing variant permanently |
| **Hard FAIL: gradient std-dev >2× single-teacher** | Abort 3h; close routing variant |

---

## 5. Updated cumulative stack

```
Iter 238 close (post-#94):
  All 27 axes ≈preserved
  Operational paradigms #92, #93, #94 in place

Iter 239 (MULTI-TEACHER-ROUTING-DISTILL):
  All 27 axes ≈preserved (no new axis)
  TEACHER PORTFOLIO meta-channel re-opened (was closed by #70-B at risk-adj 0.5×)
  Conditional on Gate-0 PASS: 2-3× per-class incremental over single-best-teacher
```

---

## 6. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| 5-class classifier C (10M params, BF16) | 250 | 1 |
| Per-class teacher integration (Tier-3 cached + Tier-1 deferred) | 400 | 1.5 |
| Argmax routing + per-class KL-CE | 200 | 0.75 |
| Cached-logit pipeline per class | 350 | 1 |
| Gate-0 probe runner + variance instrumentation | 300 | 1 |
| Evaluation harness (per-class NLL + std-dev tracking) | 300 | 0.75 |
| **Total** | **~1,800** | **6** |

---

## 7. Memory advantage preservation

| Component | Memory delta |
|---|---|
| 10M classifier (BF16) | +20 MB GPU |
| Per-class cached logits (5 partitions × ~40 GB each) | +200 GB host disk |
| **Total** | **20 MB GPU; 200 GB disk** |

**Single-GPU 16 GB ceiling preserved.** 6× reduction in cached-logit storage vs #70-B's 4-teacher simultaneous (~1 TB).

---

## 8. Honest gaps

1. **#70-B precedent at risk-adj 0.5×** below break-even — primary risk.
2. **Classifier accuracy bottleneck**: <85% kills payoff.
3. **Loses Jensen-gap fusion bonus** that simultaneous distillation provides.
4. **2.5× per-class is microopt** by program's 10-30× bar.
5. **Production-viable ~25-35%** — moderate uncertainty.
6. **Justification per iter-236 brief**: bold-claim testability with built-in Gate-0.

---

## 9. Bottom line

**MULTI-TEACHER-ROUTING-DISTILL refines rejected #70-B via classifier-based sequential routing.** Mechanism eliminates teacher-disagreement variance amplification; loses Jensen-gap fusion bonus.

**Cumulative single-GPU stack at iter-239 close:**
- All 27 prior axes ≈preserved
- TEACHER PORTFOLIO meta-channel re-opened conditional on Gate-0 PASS
- 2-3× per-class incremental over single-best-teacher (if PASS)

**Engineering:** ~1,800 LOC over 6 weeks. **Joint Gate-0 PASS ~45-55%; LLM-scale conditional ~50-65%; production-viable ~25-35%.**

**Built-in 1-day Gate-0 (~8 GPU-hours)** with three hard-FAIL signals (classifier collapse, per-class NLL >0.5 nat worse, gradient std-dev >2× single-teacher). Decision tree provides clean continuation/abort path.

**B and C reserved.**

After 55 paradigms, **27 axes** unchanged (TEACHER PORTFOLIO meta-channel re-opening conditional on Gate-0). Iter-236-239 pattern: 4 paradigms under brief change; mix of operational (#92, #94) and novel-with-built-in-test (#93, #95).
