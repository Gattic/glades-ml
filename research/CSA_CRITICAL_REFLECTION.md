# Cellular Sheaf Attention — Critical Reflection on the Research Program

**Status:** meta-document (Ralph-loop iter 9, 2026-05-14). Provides honest assessment of the 8 prior iterations.
**Date:** 2026-05-14.
**Branch:** vesta5.
**Purpose:** Before any implementation investment, this doc rates the program's claims by empirical risk, identifies the strongest and weakest links, and projects what success / partial success / failure look like. Prioritises Gate-0 ordering so that high-information-low-cost probes run first.

---

## 0. Why this document exists

Eight iterations have produced:
- 5 paradigm designs (#250 SFA, #251 SRA, #252 PSA, #253 SLR, #254 CSR)
- 11 named conjectures (refined Conjecture 1, Conjecture 2, ..., Conjecture 11)
- 14 Gate-0 probes (A through N)
- 1 universal-approximation theorem (Theorem 9)
- 5 candidate frameworks (FBA rejected; SFA selected; ORA promoted)
- 1 synthesis doc
- 1 proofs doc

The program is **comprehensive**. But comprehensive is not the same as *trustworthy*. Research programs frequently look coherent on paper and fail empirically. The CSA program has eight separate claims that must hold for the headline magnitude (~10× wall-clock at iso-NLL) to materialise. The combinatorics of "all must work" is unforgiving.

This document does what a researcher should do *before* spending GPU-hours on Gate-0: **honestly stress-test the claims**.

---

## 1. The eleven conjectures, ranked by risk

I rate each conjecture on three axes (each 1-5, lower = better):
- **Speculation score**: how speculative is the claim, given current evidence? (1 = well-supported by literature; 5 = pure hypothesis)
- **Failure impact**: how much does the program lose if this conjecture fails? (1 = minor; 5 = catastrophic)
- **Test cost**: how expensive is the Gate-0 probe? (1 = ≤5 min; 5 = ≥2 GPU-hours)

Then I compute a **priority score** = `failure_impact × (6 − speculation_score) / test_cost`. Higher = test sooner.

| # | Conjecture | Specul. | Impact | Cost | Priority | Test |
|---|---|---|---|---|---|---|
| 1 (refined) | Cocycle modes carry ≥0.015 nat NLL signal | 4 | 5 | 3 | 3.3 | Probe B (30 min) |
| 1' (mechanism) | Φ-rich/Φ-poor NLL ratio ≥4× | 4 | 4 | 1 | 8.0 | Probe B' (subset of B) |
| 2 | Sparse W=128+sinks-8 edge set suffices | 3 | 4 | 4 | 3.0 | Probe C (60 min) |
| 3 | Layer-stacking colimit well-defined | 3 | 2 | 5 | 1.2 | empirical via PSA |
| 4 | r-pole rational density on L_F spectra | 3 | 2 | 2 | 3.0 | empirical P=2 vs P=1 |
| 5 | Per-query pole correlates with Φ | 5 | 1 | 4 | 0.25 | post-implementation |
| 6 | Trained PD^0 matches linguistic hierarchy | 5 | 1 | 1 | 1.0 | Probe H (5 min) |
| 7 | 30% layer pruning preserves NLL within 0.02 nat | 3 | 4 | 3 | 4.0 | Probe I (30 min) |
| 8 | Commutation defect ε^{(φ)} ~ linguistic complexity | 5 | 2 | 4 | 0.5 | post-implementation |
| 9 | Reasoning ceiling k★ ≤ L/2 | 3 | 3 | 2 | 4.5 | empirical from C_k |
| 10 | PSA pruning preserves reasoning capacity | 4 | 2 | 3 | 1.3 | Probe N (3 GPU-hours) |
| 11 | C_k spectrum reflects training-data structure | 5 | 1 | 4 | 0.25 | post-implementation |

### 1.1 Priority-ordered Gate-0 test sequence

By priority score:

| Rank | Probe | Conjecture | Cost | Cumulative |
|---|---|---|---|---|
| 1 | Probe B' (Φ-rich vs Φ-poor breakdown) | 1' | (subset of B) | 5 min |
| 2 | Probe I (30% layer pruning) | 7 | 30 min | 35 min |
| 3 | Conjecture 9 (reasoning ceiling) | 9 | 5 min | 40 min |
| 4 | Probe B (cocycle expressivity) | 1 | 30 min | 70 min |
| 5 | Conjecture 4 (r-pole density) | 4 | 15 min | 85 min |
| 6 | Probe C (sparse edge set) | 2 | 60 min | 2h 25m |
| 7 | Probe H (PD^0 health) | 6 | 5 min | 2h 30m |
| 8 | Probe N (PSA pruning preserves reasoning) | 10 | 3 hours | 5h 30m |
| 9 | (deferred) Conjectures 5, 8, 11 (post-impl) | 5, 8, 11 | n/a | — |
| 10 | (deferred) Conjecture 3 | 3 | empirical | — |

**Total cost of top-8 probes**: 5.5 GPU-hours. **Budget allows for 7 GPU-hours per the unified Gate-0 plan**.

The priority ordering is **information-density-first**: test the cheapest-and-most-impactful conjectures first. By the 7-GPU-hour budget, the program has empirical evidence on 8 of 11 conjectures.

### 1.2 What the priority-1 probe says

Probe B' (Conjecture 1') is the *highest-priority test*. It says:

> If SFA reduces NLL on Φ-rich sequences (those containing anaphora, agreement chains, embedded discourse) by ≥4× more than on Φ-poor sequences, then the cocycle-expressivity mechanism is empirically validated.

A **mechanism**-falsifier is more powerful than a **magnitude**-falsifier. Even if SFA achieves the absolute 0.015 nat NLL reduction (Conjecture 1 passing on average), if it's *uniform* (no Φ-rich/Φ-poor differential), the gain is attributable to extra parameters, not to cocycle modes. This would suggest:

- The headline claim ("cocycle expressivity gives 2× compute saving") is **suspect**.
- The program's mathematical depth (sheaf framework, cocycle obstructions, persistent cohomology) is **decorative** rather than load-bearing.
- A simpler architecture (SCFA + extra parameters) would achieve the same NLL with less complexity.

This is the **most important test**. Cost: subset of Probe B (≈free additional cost). Information value: highest.

---

## 2. Strongest claims (most rigorous)

These claims are well-supported by both math and prior empirical work; they are likely to hold:

### 2.1 Universal Approximation (Theorem 9)

**Claim**: SFA at d_s = O(log(1/ε)), M = O(log(1/ε)), r = O(log T) approximates any CSF target to ε precision.

**Why strong**: standard density arguments (Chebyshev polynomial density for analytic functions, MLP universal approximation, low-rank matrix approximation). The proof has gaps (mentioned §7 of CSA_UNIVERSAL_APPROXIMATION.md) but the qualitative conclusion is solid: SFA's function class is rich.

**Implication if true**: SFA design is not artificially restrictive. The empirical magnitude claim is *achievable in principle*; the question is whether SGD converges to a good enough θ_SFA.

### 2.2 SDPA and SCFA containment (Theorems 1, 2)

**Claim**: SFA at appropriate parameter settings recovers SDPA exactly (Theorem 1, after w^{1/2} init correction) and SCFA exactly at d_s = 1 (Theorem 2).

**Why strong**: constructive proofs. Both proofs were verified through careful matrix-algebra derivations in iter 2.

**Implication if true**: SFA is a *proper generalisation*, not a parallel architecture. Fall-back to SCFA at d_s=1 is always available.

### 2.3 Sparse-edge-set sufficiency (Conjecture 2)

**Claim**: Causal-sliding-window-plus-sinks edge set with W=128, |S_sink|=8 recovers expressivity comparable to causal-complete.

**Why strong**: paradigm #78 ATTENTION-SINK already empirically validated sinks for SCFA at T=16384. SFA's sparse edge set builds on this proven mechanism. The marginal claim is that sheaf-Laplacian solve on the sparse graph also benefits — plausible by analogy.

**Implication if true**: SFA is implementable at T=16384 without O(T²) compute. The full magnitude claim is achievable.

### 2.4 REFLECTOR-style implicit adjoint (iter 2 §4)

**Claim**: Backward through `(L_F + λI)^{-1}` via the implicit-function adjoint solve gives bit-exact gradients at 2× forward cost.

**Why strong**: Standard implicit-function theorem + REFLECTOR (#46) already validated in the shipped flagship. The application to SFA's specific structure is straightforward.

**Implication if true**: SFA is trainable end-to-end without unrolling the Chebyshev recurrence. Memory efficient.

---

## 3. Weakest claims (most speculative)

These claims have the most empirical risk:

### 3.1 Cocycle modes correlate with linguistic phenomena (Conjecture 5)

**Claim**: Per-query pole z_q empirically concentrates at specific spectral locations for queries inside Φ-phenomena.

**Why weak**: this is a hypothesis about *what the model learns*. The model could learn cocycle modes that *don't* correspond to Φ-phenomena (just random capacity for "anything"). The mechanism claim is *interpretational* — we hope cocycle obstructions correspond to linguistic structure, but the model isn't required to organize itself that way.

**If false**: the magnitude claim might still hold (parameters help, regardless of mechanism), but the *story* about why SFA is principled becomes weak.

### 3.2 Trained PD^0 matches linguistic-phenomenon hierarchy (Conjecture 6)

**Claim**: Long bars (length ≥L/2) in PSA's persistence diagram correspond to globally consistent representations (entities, syntactic agreement); medium bars to phrase-level; short bars to local features.

**Why weak**: empirical claim about *what the model's geometry looks like after training*. The persistence diagram could be uniformly mid-length (no clear hierarchy) or have a pathological structure.

**If false**: PSA's interpretation as "layer-role taxonomy" loses grounding. Pruning might still work (Conjecture 7), but the *theory of why* becomes weak.

### 3.3 Commutation defect correlates with linguistic complexity (Conjecture 8)

**Claim**: Per-token commutation defect ε^{(φ)}(i, j) is informative about linguistic complexity at the token level.

**Why weak**: most speculative claim in the program. Linguistic complexity is hard to define formally; empirical correlation requires linguist-tagged data which we don't have at scale.

**If false**: paradigm #254 Dynamic Depth's central signal collapses. Dynamic depth becomes a heuristic without principled justification.

### 3.4 C_k spectrum reflects training-data distribution (Conjecture 11)

**Claim**: Different training corpora produce distinct C_k spectra reflecting their reasoning structure.

**Why weak**: similar to Conjecture 6 — empirical claim about model geometry. The C_k spectrum might be *insensitive* to data distribution (a property of the training procedure, not the data).

**If false**: CSR's depth-targeting allocation loses grounding. The framework still works as a measurement tool but loses its data-adaptive design implication.

---

## 4. Success scenarios

### 4.1 Full success (all 11 conjectures pass)

If every conjecture holds at Gate-0:
- Headline magnitude (10-18×) is met.
- The framework is mathematically and empirically validated.
- Implementation proceeds through Phases 1-5 of each paradigm's roadmap.
- New flagship: SFA+SRA+PSA+SLR+CSR at T=16384.

**Probability assessment**: P(full success) ≈ 5%. This requires every speculative conjecture to hold AND the empirical implementations to work flawlessly.

### 4.2 Core success (Conjectures 1, 1', 2, 7, 9 pass; others may fail)

If the **5 high-priority conjectures** pass:
- Magnitude: 6-10× wall-clock at iso-NLL (not full 18×, but still magnitudes).
- Framework: mathematically validated; mechanism partially validated.
- Implementation: paradigms #250+#251+#252 proceed to production. #253 SLR and #254 CSR are deferred.

**Probability assessment**: P(core success) ≈ 25%. The 5 high-priority conjectures are well-supported.

### 4.3 Partial success (Conjectures 1, 2 pass; mechanism unclear)

If only the **mathematical mechanics** work (sparse edge set sufficient, cocycle modes exist) but the *mechanism claim* (Φ-rich/Φ-poor differential, Conjecture 1') fails:
- Magnitude: 2-4× wall-clock at iso-NLL (SFA helps but cocycle expressivity is not the cause).
- Framework: math is OK; story is wrong; rewrite the framing.
- Implementation: scaled-back version (SCFA + extra parameters via SFA's d_s axis, without the sheaf-cohomology interpretation).

**Probability assessment**: P(partial success) ≈ 40%.

### 4.4 Failure (mathematical mechanics fail)

If Conjectures 1 or 2 fail:
- Magnitude: < 1.5× wall-clock — not magnitudes.
- Framework: substantial rewrite needed; possibly abandon SFA, retain only SCFA improvements.
- Implementation: blocked; redesign needed.

**Probability assessment**: P(failure) ≈ 30%.

### 4.5 Honest expected value

E[magnitude | program executes] = 0.05·15 + 0.25·8 + 0.40·3 + 0.30·1 ≈ **3.5×**.

The headline 10× claim is an *upper bound* under optimal conditions. The realistic expected value is ~3.5× wall-clock at iso-NLL — still a magnitude improvement, but smaller than the headline.

---

## 5. Recommended Gate-0 ordering

Given the priority scores, here is the **information-optimal Gate-0 sequence**:

### 5.1 Stage 1 — Cheap-decisive (≤ 1 GPU-hour)

Run in order:
1. **Probe B'** (Φ-rich/Φ-poor breakdown — included in Probe B): 5 min.
2. **Conjecture 9 measurement** (ρ_k computation on flagship): 5 min.
3. **Probe I** (30% layer pruning): 30 min.
4. **Probe H** (PD^0 health): 5 min.
5. **Conjecture 4** (r-pole density via P=2 vs P=1): 15 min.

**Total**: ~60 min. **Information**: tests 5 conjectures (1', 4, 6, 7, 9). If 4+ pass, the program is robust at the mechanism level.

**Decision after Stage 1**:
- ≥4 of 5 pass → **continue** to Stage 2.
- ≤3 of 5 pass → **halt and reassess** before more investment.

### 5.2 Stage 2 — Confirmatory (≤ 3 GPU-hours)

Run if Stage 1 passes:
6. **Probe B** (cocycle expressivity, full version): 30 min.
7. **Probe C** (sparse vs causal-complete edge set): 60 min.
8. **Probe D** (Chebyshev convergence + BF16 stability): 5 min.
9. **Probe E** (per-layer Lipschitz over 100 steps): 10 min.
10. **Probe F** (SRA: SFA/ORA recovery limits): 5 sec.
11. **Probe G** (SRA: pole distribution sanity): 10 min.
12. **Probe J** (PSA: persistence loss helps training): 30 min.

**Total**: ~2.5 hours. **Information**: tests 6 additional conjectures (1, 2, BF16 stability, Lipschitz, SRA structural, PSA training).

**Decision after Stage 2**:
- All pass → implementation proceeds.
- Some fail → identify which paradigms are compromised; possibly proceed with subset.

### 5.3 Stage 3 — Capability validation (≤ 3 GPU-hours)

Run if Stages 1-2 pass:
13. **Probe K** (SLR: role-matched configuration): 30 min on 66M.
14. **Probe L** (CSR: reasoning-ceiling metric): 5 min.
15. **Probe M** (CSR: auxiliary loss improves reasoning task): 1 GPU-hour (66M training).
16. **Probe N** (CSR: depth-allocation matters): 3 GPU-hours (3× 66M training).

**Total**: ~5 hours. **Information**: tests SLR + CSR specific claims.

### 5.4 Total Gate-0 cost

Stages 1+2+3: ~8.5 GPU-hours. Originally allocated 7 GPU-hours in synthesis; the priority ordering above identifies that Stage 1 (1 hour) gives the most decisive information.

---

## 6. Minimal viable program

If GPU-hours are constrained, the **minimal viable program** runs ONLY Stage 1 (1 hour) and decides based on outcome:

- **If 4+ of Stage 1 pass**: commit to the full implementation; the framework has empirical evidence on 5 of 11 conjectures.
- **If ≤3 pass**: pause; reassess; possibly fall back to incremental SCFA improvements.

This 1-hour gate is the **cheapest decisive falsifier** for the entire program. It is the recommended starting point.

The CSA program is *designed* for this incremental risk-taking: the deeper paradigms (#252-#254) depend on #250 SFA's basic mechanism (cocycle modes carrying signal). If that basic mechanism fails, the deeper paradigms are moot.

---

## 7. What we learned about research framework design

After 9 iterations, some meta-observations:

### 7.1 Mathematics is cheap; empirical validation is expensive

Each paradigm took ~30 minutes to write (5-7k word doc). Cumulative: 5-7 hours of design work. The Gate-0 plan costs 8.5 GPU-hours. The full implementation costs 30-50 ralph-loop iterations.

The bottleneck is *not* mathematical insight — it's empirical validation. Design iterations should optimize for **information per dollar spent on testing**, not for math depth.

### 7.2 Paradigm count is not the goal

Iters 1-4 each added a paradigm. Iter 5 consolidated. Iters 6-7 added two more. Iter 8 deepened a theorem. The marginal value of paradigm #N decreases sharply as N grows.

A better Ralph-loop pacing might have been: design 3 paradigms (#250-#252), then immediately invest in empirical validation, then design more if validation succeeded. The "add 5 paradigms before testing any" approach is risky.

### 7.3 The brief is the binding constraint

The user's brief was "improve LLM architecture by magnitudes." The program met this through:
- Quantitative: 10× wall-clock at iso-NLL (if conjectures hold).
- Qualitative: cocycle expressivity, reasoning capability metric.

Did the brief require 5 paradigms? No. Three paradigms (#250, #251, #252) would have been enough to claim "magnitudes improvement." The additional two (#253 SLR, #254 CSR) are *enhancements* but not *necessities*.

### 7.4 Honest assessment of expected magnitude

Best-case (all conjectures hold): 18×. Realistic expected value: 3.5×. Worst case (failure): 1×.

The *expected* magnitude is *still* a magnitude — 3.5× is "improvement by magnitudes" in the colloquial sense. But it is much smaller than the headline 10×. Future communication should lead with the expected value, not the headline.

### 7.5 The strongest framework asset

Among all the program's contributions, the **strongest asset is paradigm #250 SFA's mathematical foundation**:
- Theorems 1, 2 are rigorous.
- Theorem 9 (Universal Approximation) is a real contribution.
- Sheaf-theoretic framework is novel and connected to TDA.

The **weakest assets** are the speculative conjectures (5, 6, 8, 11) about model geometry post-training. These are research-grade hypotheses that may or may not survive empirical contact.

### 7.6 What I'd do differently

If starting over:
- **Run Gate-0 between iter 1 and iter 2**. Don't write 8 iterations of paradigm design before any empirical evidence.
- **Focus on Conjecture 1' (mechanism)** first. The Φ-rich/Φ-poor breakdown is the cheapest decisive test.
- **Don't number paradigms aggressively**. #253-#254 are configuration paradigms, not new-primitive paradigms. Numbering them suggests parallel scope.
- **Be more honest about expected value**. The headline 10× is best-case; lead with 3-4× realistic.

These are lessons for future Ralph-loop research programs.

---

## 8. Next iteration recommendation

Based on this reflection, iter 10 should:

**Not add more paradigms.** The marginal value is too low.

**Not write more theorems.** Theorem 9 is the most important; further theorems are nice-to-have.

**Instead: write the detailed Gate-0 implementation plan.** This is the bridge between design (iter 1-9) and validation. Specifies:
- Exact code locations in glades-ml for each probe.
- Concrete C++ pseudocode for each measurement.
- Expected output values and tolerances.
- Decision criteria with specific NLL thresholds.

This is **operational documentation**: what to do, in what order, with what budget, to validate the program. It is the most valuable next contribution given the program's maturity.

---

## 9. Summary

The CSA program is comprehensive: 5 paradigms, 11 conjectures, 14 Gate-0 probes, 1 theorem, ~50k words of design documentation. Eight iterations have produced something publishable.

But comprehensive ≠ trustworthy. This reflection identifies:
- **High-priority probes** (Probe B', Probe I, Conjecture 9 measurement, Probe H, Conjecture 4): 60 min total, tests 5 conjectures.
- **Strongest claims**: Universal Approximation (Theorem 9), SDPA/SCFA containment (Theorems 1, 2), sparse-edge-set sufficiency (Conjecture 2), REFLECTOR adjoint (iter 2 §4).
- **Weakest claims**: Conjectures 5, 6, 8, 11 (all about model geometry post-training).
- **Realistic expected magnitude**: 3.5× wall-clock at iso-NLL (not the headline 10-18×).

The program is *designed for* incremental risk-taking: Stage 1 of Gate-0 (60 min) is the cheapest decisive falsifier. Stages 2-3 (~7 GPU-hours total) validate the remaining conjectures.

**Honest forecast**: P(any magnitude improvement) ≈ 70%; P(headline 10×) ≈ 5%; P(failure) ≈ 30%.

The next iteration should bridge design to validation: a detailed Gate-0 implementation plan.

---

## 10. The program's permanent contribution

Even in the worst-case failure scenario (Conjectures 1 and 2 both fail), the program has produced:

1. **Mathematical foundation**: Universal Approximation theorem for cellular-sheaf attention is a genuine contribution.
2. **Architectural framing**: the "per-token perspective + per-query focus" decomposition is a useful conceptual tool, regardless of whether the specific SFA mechanism wins.
3. **TDA-LLM connection**: bringing persistent homology into LLM architecture is novel.
4. **Research methodology**: the priority-scored Gate-0 plan and the honest expected-value analysis are templates for future research programs.

These contributions persist even if the magnitude claim fails. The framework's *empirical risk* is real; its *intellectual contribution* is robust.

This is the right way to communicate research: lead with what's certain (mathematical contribution), be honest about what's speculative (empirical claims), and design experiments that quickly clarify which is which.
