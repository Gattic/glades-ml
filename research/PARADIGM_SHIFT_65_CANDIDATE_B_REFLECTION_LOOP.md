# Paradigm Shift #65 Candidate B — REFLECTION-LOOP (recursive self-improvement via reflection chains; the model generates an output, reflects on its quality, produces an improved revision, and trains on the revision)

**Status:** candidate-B design for paradigm shift #65. **Recommended action: REJECT.** The mechanism overlaps substantially with the already-reserved #63-C SAGE candidate (covering the same closed-loop self-curation axis at the same time-grain with the same auxiliary-loss formulation). The principal differentiator REFLECTION-LOOP offers — explicit revision rather than filtration — has been studied at LLM scale (Madaan 2023 *Self-Refine*, Saunders 2022 *Self-Critique*, Huang 2024 *LLMs Cannot Self-Correct Reasoning Yet*) with mixed-to-negative empirical results and no compelling refinement vs SAGE that would justify rebuilding the same scaffolding. This document develops the candidate honestly enough to defend the rejection.
**Date:** 2026-05-08 (Ralph-loop iter 209+, post-#64 MEMORY-CHIRON selection, paradigm depth 23 in the bigger-picture track #56–#64).
**Predecessors.** Load-bearing references: `PARADIGM_SHIFT_63_CANDIDATE_C_SAGE.md` (the closest neighbor; whole-output self-grading + corpus filter; **the central comparison this document must justify against**), `PARADIGM_SHIFT_56_CANDIDATE_B_DISTILL_FORWARD.md` (training on model-generated tokens as a target distribution), `PARADIGM_SHIFT_57_CANDIDATE_B_METAGEN.md` (synthetic-corpus generation + multi-criterion quality filter), `PARADIGM_SHIFT_59_CANDIDATE_B_PRM_CHIRON.md` (auxiliary classifier head pattern at per-step granularity).
**Axis.** **Recursive self-improvement via explicit revision chain.** The model emits an initial output `y₀`, then conditioned on `(prompt, y₀)` emits a critique `c`, then conditioned on `(prompt, y₀, c)` emits a revised output `y₁`. The training signal is the (assumed-better) revision `y₁`, treated as a target for the original `(prompt → output)` mapping.

**Tagline.** *#56 DISTILL trains on a frozen teacher's output. #58 METAGEN trains on filtered self-generations. #59 PRM trains a per-step head. #63-C SAGE trains a whole-output self-evaluator as a filter. REFLECTION-LOOP moves differently on the same closed-loop axis: rather than filter or score, it **revises** — model writes answer, then critique, then corrected answer, trained to imitate the correction. Literature on intrinsic self-correction is mixed-to-negative. Overlap with SAGE is structural. Recommendation: reject.*

**Honest headline.** **Conjectural 1.2–1.5× wall-clock speedup at fixed final NLL — same band as #63-C SAGE — but the mechanism rebuilds 70-80% of SAGE's scaffolding for marginal additional value, and the empirical literature on self-correction at LLM scale is mixed-to-negative. Recommend rejection at design time; do not advance to Gate-0.**

---

## 0. Executive summary (HONEST claim — overlap is the dominant constraint)

The pre-#65 stack assumes #56–#64 ship: #56 DISTILL-FORWARD, #57 SCROLL, #58 METAGEN, #59 PRM-CHIRON, #60 COSMIC, #61 COSMIC-PROMOTED, #62 AGENT-CHIRON, #63 META-LEARN-PROMOTED (with #63-C SAGE *reserved* as deployment-feature with auxiliary-loss training-time tail), and #64 MEMORY-CHIRON-promoted. The closed-loop self-curation axis is densely covered.

**REFLECTION-LOOP's claim.** The missing primitive is **explicit revision**: rather than score, filter, or distill, *generate a better text and train against it*. Three sub-mechanisms: (1) **three-pass generation** — for each prompt emit `y₀`, then `c = critique(p, y₀)`, then `y₁ = revise(p, y₀, c)`, all standard KV-cached forwards; (2) **reflection-chain CE auxiliary loss** — `(p, y₁)` added to corpus with target `y₁` and standard CE at `λ_REFL = 0.05` (primary CE unchanged); (3) **optional recursive iteration** — `y₁ → y₂ → …`; literature reports monotone plateaus by `K = 1` or `K = 2` with regression beyond.

**Per-step compute.** Three forward passes per audited prompt vs SAGE's one. Audit at `N_audit = 10000`, `n_prompts = 50000`: ~21 GPU-min per cycle = ~0.18% wall-clock vs SAGE's ~0.06%. Trunk forward/backward unchanged at `3F`. **Total per-step overhead: ~0.18%** — 3× SAGE's amortized cost for the same closed-loop primitive.

**Per-effective-step speedup (conjectural).** `1.10–1.30×` conservative; `1.30–1.50×` aggressive contingent on revision-quality monotonicity holding at 1.84B+ scale. **Headline 1.2×** — same as SAGE by construction.

**Honest gaps (these dominate the recommendation).**

1. **SAGE overlap is structural.** §3 develops this. REFLECTION-LOOP's audit-cycle pipeline is SAGE's plus two extra forward passes; corpus-mixer integration is identical; joint loss is identical in form (`L = L_CE + λ · L_aux`); curriculum is identical; collapse-detection harness is identical. **Filter-vs-revise is a single-bit architectural choice; the rest is shared scaffolding.**
2. **Literature on self-correction at LLM scale is mixed-to-negative.** Huang 2024 *LLMs Cannot Self-Correct Reasoning Yet* (arXiv:2310.01798) reports intrinsic self-correction **degrades** math reasoning at 7B–70B. Madaan 2023 *Self-Refine* (arXiv:2303.17651) reports ~10–20% improvement on some open-ended tasks and **degradation** on others. Saunders 2022 *Self-Critique* reports ~50–60% self-error-identification on summarization — barely above chance. The premise (revision improves output) is empirically contested.
3. **Compute-cost asymmetry.** Three forward passes vs SAGE's one. Permanent 3× overhead for the same headline speedup band.
4. **Recursive iteration provides no further headroom.** `K ≥ 2` is precisely the regime Huang 2024 documents the largest regressions in.

**Disposition.** At paradigm depth 23, with SAGE already reserved at depth 22 covering the same axis, with literature mixed on the load-bearing premise, and with compute cost 3× higher for the same speedup band, **REFLECTION-LOOP fails the marginal-contribution test by design**.

**Engineering scope (if developed).** ~750 LOC over ~3.5 weeks (vs SAGE's ~600 / 3 weeks).

---

## 1. Mechanism: three-pass generation as a training-signal source

### 1.1 The reflection chain

For a prompt `p`, the reflection chain is:

```
y₀ = generate(p)                                  [initial answer]
c  = generate(p ⊕ y₀ ⊕ "[critique:]")             [self-critique]
y₁ = generate(p ⊕ y₀ ⊕ c ⊕ "[revised answer:]")   [revision]
```

`[critique:]` and `[revised answer:]` are special tokens added to the BPE vocabulary with frozen embeddings (mirrors #58-C REASONING-CHAIN's `[reason]`/`[/reason]` markers).

The training signal is `(p, y₁)`: the trunk is trained to map `p → y₁` via standard CE on the revision tokens. The intermediate critique `c` is *not* in the training target; it is consumed only as scaffolding to produce `y₁`. **The student learns from its own polished output, not from its critique.**

### 1.2 Auxiliary-loss formulation

```
L(θ) = L_CE(θ) + λ_REFL · L_REVISE(θ),

L_CE(θ)        = −∑_t log P_θ(x_t | x_<t)              [primary, real corpus]
L_REVISE(θ)    = −∑_{(p, y₁) ∈ R} ∑_t log P_θ(y₁,t | p, y₁,<t)   [auxiliary, revision corpus]
```

`R` is the rolling revision-corpus buffer. `λ_REFL = 0.05` default; tuning range `[0.02, 0.15]`. The form is **exactly** `L = L_CE + λ · L_aux` with the auxiliary being a CE on a self-generated corpus subset — identical in form to SAGE's loss formulation; only the inner signal differs.

### 1.3 Why revision rather than filter — the load-bearing claim

The candidate argues that **SAGE's filter is binary** (corpus inclusion or not; ≤ 1 bit per prompt), while **REFLECTION-LOOP's revision is dense** (the entire `y₁` token sequence; `O(|y₁|)` tokens per prompt). *In principle*, denser signal → more efficient closed-loop convergence.

The honest counter-claim: **the dense signal is only useful if `y₁` is reliably better than `y₀`**. If revision is not monotone-improving (the empirical situation per Huang 2024), the dense signal is a *high-bandwidth channel for noise*. SAGE's binary filter, by contrast, is an explicit threshold check; even with imperfect calibration, the 1-bit decision is robust to revision-quality drift.

In the regime where self-correction works (some open-ended tasks per Madaan 2023), REFLECTION-LOOP's denser signal *might* outperform SAGE's filter. In the regime where self-correction fails (math reasoning per Huang 2024), REFLECTION-LOOP injects systematically wrong-direction gradient. **The asymmetry of failure modes favors SAGE: SAGE's worst case is no improvement; REFLECTION-LOOP's worst case is anti-improvement.**

### 1.4 Curriculum and warmup

If developed, the curriculum mirrors SAGE's three-phase structure: Phase 0 pure-CE warmup; Phase 1 ramp-and-gate (audit cycle generates reflection chains and adds to `R` only if revision-quality probe passes); Phase 2 full-`λ` operation if probe passed, else `λ_REFL → 0` and REFLECTION-LOOP becomes a no-op. Curriculum is identical to SAGE's; only the gate metric differs (AUROC for SAGE; monotone-improvement rate for REFLECTION-LOOP).

---

## 2. Audit cycle: three-pass pipeline and corpus integration

Every `N_audit = 10000` training steps the trunk runs three KV-cached inference passes (batch=8) on `n_prompts = 50000` validation queries: initial generation `{y₀,i}` (~7 GPU-min), critique generation `{c_i}` from `(p_i, y₀,i, "[critique:]")` (~7 GPU-min), revision generation `{y₁,i}` from `(p_i, y₀,i, c_i, "[revised answer:]")` (~7 GPU-min). A revision-quality probe on a ~5000-prompt subset with ground-truth labels (MC-rollout final-answer-check; same primitive as SAGE §1.4) scores `monotone_rate = #{revision_correct ∧ initial_wrong} / #{quality_decisions}`.

Filtration: if `monotone_rate ≥ 0.55`, all unlabeled `(p_i, y₁,i)` pairs are added to `R` with weight `w = 0.5` (down-weighted; mirrors METAGEN §4.3 and SAGE §2.2). If `monotone_rate ∈ [0.45, 0.55)`, only labeled-and-improved pairs are added. If `monotone_rate < 0.45`, no pairs are added (revision is anti-monotone; closed loop unsafe this cycle).

**Total audit cycle wall-clock: ~21 minutes vs SAGE's ~10 minutes** (factor of 2 due to three forward passes vs one). Amortized over 10k steps: ~0.18% wall-clock fraction. **Negligible in absolute terms; 3× SAGE's per-step overhead for the same closed-loop primitive.** Down-weighting and revision-quality gating are direct ports of SAGE's collapse mitigations; the Renyi-entropy collapse monitor (METAGEN §3.5; SAGE §2.2) carries over unchanged.

---

## 3. The SAGE overlap problem: the load-bearing rejection rationale

This is the most important section of this document. The decision to recommend rejection rests on the structural overlap with #63-C SAGE.

### 3.1 What is shared

| Component | SAGE | REFLECTION-LOOP | Status |
|---|---|---|---|
| Audit cycle interval (`N_audit`) | 10000 steps | 10000 steps | identical |
| Audit prompt count (`n_prompts`) | 50000 | 50000 | identical |
| Joint loss form | `L_CE + λ · L_aux` | `L_CE + λ · L_aux` | identical |
| `λ` default | 0.10 | 0.05 | comparable |
| Curriculum phases | 0 → warmup → full | 0 → warmup → full | identical structure |
| Phase-2 gating metric | AUROC ≥ 0.75 | monotone-rate ≥ 0.55 | analogous |
| Down-weight on synth | `w = 0.6` | `w = 0.5` | comparable |
| MixCorpus integration | rolling pos buffer | rolling pos buffer | identical |
| Renyi-entropy collapse monitor | yes | yes | identical |
| Cold-start fail-safe | `λ → 0` if AUROC fails | `λ → 0` if rate fails | identical |
| Engineering scope | ~600 LOC, 3 weeks | ~750 LOC, 3.5 weeks | comparable |

**Eleven of eleven scaffolding components are shared.** The two candidates are the same closed-loop primitive at the same time-grain on the same axis with the same auxiliary-loss formulation, the same curriculum, the same collapse mitigations, and the same engineering footprint up to a 25% scope inflation in REFLECTION-LOOP's case.

### 3.2 What differs

The architectural difference is one bit:

- **SAGE:** auxiliary head is a **classifier** (~2M-param `f_φ : h_pool → [0,1]`) producing a binary signal used to **filter** generations.
- **REFLECTION-LOOP:** the trunk itself produces **two extra generations** (critique + revision) used to **replace** generations.

That is the entire mechanistic difference. Everything else — when audit fires, how often, how the corpus is mixed, how collapse is detected, how warmup is gated, how `λ` is bounded, how the failure mode is handled — is identical.

### 3.3 Does the architectural difference justify the duplication?

Honest assessment: **information-theoretically yes** (`O(|y₁|)` tokens vs 1 bit is a real difference); **empirically contested** (Huang 2024, Madaan 2023, Saunders 2022 do not establish denser signal beats filter signal at LLM scale, and on several benchmarks establish the opposite); **architecturally the denser signal is also a denser noise channel** (SAGE's worst case is a no-op; REFLECTION-LOOP's worst case is anti-monotone gradient injected at `λ_REFL = 0.05` for cycles before the gating metric catches up). **The architectural difference is a real claim, but the empirical literature does not support its premise at the relevant scale, and the engineering cost of duplicating the SAGE scaffolding is not justified by the modest expected delta.**

### 3.4 Joint composition: would running both help?

Geometric-mean composition heuristic for same-axis paradigms is `0.3–0.4×` of standalone surviving. SAGE ~1.2× standalone, REFLECTION-LOOP ~1.2× standalone, joint ~`1.27×` — ~5% additional over SAGE-alone for ~25% additional engineering scope and ~3× additional audit-cycle compute. **Joint composition does not justify development.**

### 3.5 What would change the recommendation?

The candidate would become defensible if any of: (1) empirical evidence intrinsic self-revision improves at 1.84B+ scale, required *before* engineering investment; (2) a meaningful refinement (tree-search-over-revisions, multi-trajectory revision aggregation, critique-via-different-distillation-checkpoint); (3) a scope reduction making REFLECTION-LOOP cheaper than SAGE; (4) a formal information-theoretic lower bound on filter-regime information loss. **None hold for the current candidate.**

---

## 4. Differentiation from neighboring shipped paradigms

**vs #56 DISTILL-FORWARD.** #56 trains against a frozen *previous-generation teacher*'s output distribution. REFLECTION-LOOP trains against its *own current-generation* revision (at most as good as the trunk is). #56 imports knowledge; REFLECTION-LOOP cannot import knowledge it does not have. Joint over #56 alone: ~1.05–1.10×.

**vs #58 METAGEN.** #58 generates from a *previous-generation teacher* at high volume (70–90% synth-share). REFLECTION-LOOP generates from the *current trunk* at low volume (≤ 10%). Shared scaffolding is the synthetic-corpus mixer. Joint: ~1.05–1.15×.

**vs #59 PRM-CHIRON.** #59 adds a per-step classifier head; REFLECTION-LOOP adds a whole-output revision pipeline. Different granularity, different mechanism. Joint: ~1.10–1.15×.

**vs #63-C SAGE — the dominant overlap.** §3 develops this. Geometric-mean composition with SAGE alone caps marginal joint at ~1.05× — below the paradigm-shift threshold.

**Net independent contribution.** vs **post-#56 + #58 + #59 + reserved-#63-C-SAGE** baseline (production stack at depth 23), REFLECTION-LOOP's marginal contribution is the geometric mean of partial-stack overlaps: **~1.05× best case, ~1.0× expected, possibly < 1.0× if revision quality regresses**. Below the paradigm-shift threshold.

---

## 5. If developed: Gate-0 protocol and predicted outcome

This section is included for completeness. The recommendation is to *not* run Gate-0.

**Question.** *On 66M CHIRON × 30k steps, does REFLECTION-LOOP reach `monotone_rate ≥ 0.55` and show ≥ 1.10× wall-clock speedup vs pre-#65 baseline + reserved-SAGE?*

**Three arms × 30k steps at 66M.** **A (control)** = post-#64 stack with reserved-SAGE auxiliary-loss-only. **B (REFLECTION auxiliary-only)** = stack + REFLECTION + `λ = 0.05`, filtration *disabled*. **C (REFLECTION full)** = B + audit-cycle filtration with monotone-rate gate.

**Pass criteria.** (1) `monotone_rate ≥ 0.55` by step 20k; (2) Arm B NLL within 0.05 nat of Arm A; (3) Arm C wall-clock to Arm-A-NLL ≤ 0.90×; (4) no collapse signature; (5) **Arm C marginal speedup over reserved-SAGE-equivalent ≥ 1.05× — the load-bearing criterion: if REFLECTION-LOOP cannot beat reserved-SAGE by at least 5%, it fails the paradigm-shift threshold.** Fail-fast: `monotone_rate ≤ 0.45` → REJECT; Arm C speedup over SAGE ≤ 1.02× → REJECT (overlap is total).

**Cost.** ~40 GPU-hours.

**Realistic Gate-0 outcome (predicted from literature).** `monotone_rate ≈ 0.48–0.52` (near chance per Huang 2024); Arm B NLL parity but no positive contribution; Arm C marginal over SAGE in `0.95–1.05×` band — overwhelmingly likely outcome is rejection at Gate-0 even if engineering proceeds.

---

## 6. Engineering: ~750 LOC over ~3.5 weeks (if developed)

| Component | Files | LOC | Week |
|---|---|---|---|
| Three-pass audit pipeline (KV-cached) | `Networks/refl_audit.cpp`, `.h` | 250 | 1–2 |
| Critique-token format spec + BPE addition | `DataObjects/refl_tokens.cpp` | 60 | 1 |
| Revision-corpus mixer (80% shared with SAGE) | `DataObjects/refl_corpus.cpp` | 80 | 2 |
| Revision-quality probe (100% shared with SAGE/PRM) | `Networks/refl_quality.cpp` | 50 | 2 |
| Auxiliary-loss CUDA integration | `cuda/refl_aux_kernel.cu` | 100 | 2–3 |
| CLI flags | run.sh, argparse | 70 | 3 |
| Gate-0 harness + monotone-rate eval + Renyi-collapse asserts | `unit-tests/.../refl_test.cpp` | 140 | 3–4 |
| **Total** | | **~750** | **~3.5 weeks** |

~25% scope inflation over SAGE for ≤ 5% additional speedup over SAGE; engineering economics do not justify development.

---

## 7. Recommended action: REJECT (do not advance to Gate-0)

**Decision rationale.** (1) Structural overlap with reserved #63-C SAGE is total at the scaffolding level (§3.1 — eleven of eleven components shared); the revise-vs-filter difference is one bit and does not justify duplicated infrastructure. (2) The empirical literature on intrinsic self-revision at LLM scale is mixed-to-negative (Huang 2024 anti-monotone on math reasoning; Madaan 2023 mixed across tasks; Saunders 2022 ~50% self-error-ID at summarization). (3) Compute cost is 3× SAGE's for the same headline speedup band — an unforced economic loss. (4) Joint composition with reserved SAGE caps marginal at ~5% (§3.4) — below the paradigm-shift threshold. (5) None of the four conditions that would change the recommendation hold (§3.5).

**Disposition.** REJECT at design time. Keep on file as reasoned negative for the closed-loop self-curation axis at depth 23. Reusable primitives (three-pass generation pipeline, revision-quality probe, monotone-rate gate) preserved for future paradigms on adjacent axes (e.g., deployment-time inference-augmentation akin to chain-of-thought self-consistency).

---

## 8. Summary

REFLECTION-LOOP is **the recursive-self-revision primitive at training time**: a three-pass generation pipeline (initial → critique → revision) producing revision tokens added to a rolling corpus subset and trained against via a `λ = 0.05` auxiliary CE loss. Mechanism differs from neighbors only in the substitution of explicit revision for binary filtration (#63-C SAGE) or for soft-distribution KL distillation (#56). All other scaffolding — audit cycle, curriculum, collapse mitigation, MixCorpus integration, fail-safe — is shared with SAGE.

**Conjectural training-side speedup at fixed final NLL: 1.2× conservative; 1.5× aggressive** — same band as SAGE by construction. **Marginal contribution over SAGE: ≤ 5% best case; possibly negative if revision quality regresses (Huang 2024 result).** Engineering scope ~750 LOC vs SAGE's ~600; audit-cycle compute 3× SAGE's per cycle.

**Honest empirical risk.** The load-bearing premise — that intrinsic self-revision improves outputs at LLM scale — is empirically contested. Huang 2024, Madaan 2023, and Saunders 2022 do not establish that the denser revision signal beats SAGE's filter signal at the relevant scale, and on several benchmarks establish the opposite.

**Honest mechanistic-overlap accounting (the paradigm's largest weakness).** §3 develops this. Reserved #63-C SAGE covers the same axis with eleven-of-eleven shared scaffolding components; REFLECTION-LOOP differs in one architectural bit (revise vs filter). Joint composition with SAGE caps marginal at ~5%.

**Recommended action: REJECT.** Do not advance to Gate-0. The Ralph-loop iter-200+ methodology requires a meaningful refinement over a reserved candidate before engineering investment; REFLECTION-LOOP offers none. The candidate document is preserved as a reasoned negative for the closed-loop self-curation axis at depth 23 and as a source of reusable scaffolding (three-pass generation pipeline, revision-quality probe, monotone-rate gate) for future paradigms on adjacent axes.

**End of Paradigm Shift #65 Candidate B design document.** Honest framing: SAGE overlap is structural and dominant; literature does not support the load-bearing premise; recommend rejection at design time.
