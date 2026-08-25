# Paradigm Shift #63 Candidate C — SAGE (Self-Audit Generation Evaluation; model self-grades its own outputs as a training-signal filter)

**Status:** candidate-C design for paradigm shift #63. **Recommended action: candidate, but honestly framed as a modest-gain entry in the #63 slate.** Significant mechanistic overlap with #56 DISTILL-FORWARD (training-on-model-output) and #58 METAGEN (synthetic-data filtering). The Self-Refine / Self-Critique literature shows mixed, often-context-sensitive results at LLM scale, and the marginal contribution at paradigm depth 22 is small.
**Date:** 2026-05-08 (Ralph-loop iter 207, post-#62 AGENT-CHIRON selection, paradigm depth 22 in the bigger-picture track #56–#62).
**Predecessors.** All of #42–#62. Load-bearing references: `PARADIGM_SHIFT_56_CANDIDATE_B_DISTILL_FORWARD.md` (training on teacher output distribution; the *target-side* analog of SAGE), `PARADIGM_SHIFT_57_CANDIDATE_B_METAGEN.md` (synthetic-corpus generation + multi-criterion quality filter; the *input-side* analog of SAGE), `PARADIGM_SHIFT_58_CANDIDATE_C_REASONING_CHAIN.md` (reasoning-segment loss-weighting), `PARADIGM_SHIFT_59_CANDIDATE_B_PRM_CHIRON.md` (auxiliary-classifier-head pattern; PRM trained jointly with trunk).
**Axis.** **Self-evaluation auxiliary head + filtered self-improvement loop.** A small classifier head co-trained with the trunk predicts whether the trunk's *entire* generated output is correct. High-confidence-correct outputs are added to the training corpus (positive examples); low-confidence outputs are flagged for further training emphasis. The mechanism is **whole-output grading**, not per-step (#59 PRM) or per-token (#58-C REASONING-CHAIN).

**Tagline.** *#56 DISTILL-FORWARD trains on the teacher's output distribution at the loss target. #57 METAGEN trains on the teacher's generated tokens at the loss input. SAGE is the small-but-orthogonal third move on this axis: train the model to grade its own outputs, then use the grade to filter what it trains on next. It is the same closed-loop self-curation idea METAGEN has — just with the model itself as the filter and the granularity of the audit at the whole-output level.*

**Honest headline.** **~1.2× wall-clock speedup at fixed final NLL (conservative); ~1.5× aggressive if self-grading is reliable above ~85% AUROC.** NLL preservation: yes (the auxiliary self-eval head fires at `λ_SAGE ≤ 0.1`; the primary CE loss is unchanged). The 1.2× claim is **the smallest training-side multiplier in the entire #63 slate by design** and reflects the substantial overlap with #56 DISTILL-FORWARD and #58 METAGEN already in the production stack.

---

## 0. Executive summary (HONEST claim — modest gain; substantial overlap)

Pre-#63 cumulative stack assumes the bigger-picture track has shipped #56 DISTILL-FORWARD, #57 SCROLL, #58 METAGEN, #59 PRM-CHIRON, #60 COSMIC, #61 COSMIC-PROMOTED, and #62 AGENT-CHIRON. The dominant per-trajectory paradigms in this stack are precisely the ones SAGE most overlaps with: **#56** trains on the teacher output distribution at the loss target; **#58** generates synthetic data via teacher inference and filters via a multi-criterion classifier; **#59** trains a small auxiliary classifier head jointly with the trunk on step-level correctness labels. SAGE's "self-grading" is structurally a degenerate same-generation case of #56's teacher-grading; its "filtered self-improvement loop" is mechanistically identical to METAGEN's filter pipeline with the filter replaced by a self-eval head; its "self-evaluation head" uses the #59 PRM head pattern at whole-output rather than per-step granularity.

SAGE's three sub-mechanisms compose as follows:

1. **Auxiliary self-evaluation head.** ~10M-param binary classifier `f_φ : pooled_hidden → [0, 1]` predicting "is this generated output correct?" Architecture identical to the #59 PRM head; granularity differs (whole-output vs per-step). Joint loss `L = L_CE + λ_SAGE · L_eval` with `λ_SAGE = 0.1` default. Per-step trunk overhead `≤ 0.05%` (head fires once per generated output, not once per token).
2. **Self-grading filtration loop.** Every `N_audit ≈ 10000` steps the trunk generates `~50000` outputs on validation prompts; the self-eval head scores each; outputs with `f_φ ≥ 0.85` are added to the training corpus as positives; outputs with `f_φ ≤ 0.30` flag their *prompts* (not the outputs) for oversample in subsequent batches.
3. **Adversarial self-eval retraining.** The self-eval head is periodically retrained on `(output, ground-truth-correctness)` pairs from three sources: MC-rollout-with-final-answer-check (math/code; ~85% accuracy), auxiliary verifier model (open-ended; ~75%), and human-curated labels (~95%, ~50k available).

**Per-step compute.** Trunk forward + backward unchanged at `3F`. Audit cycle wall-clock ~1% amortized over 10k-step intervals. **Total per-step overhead: ~0.05% on average.**

**Per-effective-step speedup (conjectured):** `1.10–1.30×` conservative; `1.30–1.60×` aggressive contingent on self-eval head reaching ≥ 85% AUROC on the held-out correctness benchmark. **Headline conservative: 1.2× over post-#62 stack.** Smallest training-side multiplier in the #63 slate by design. Orthogonal deployment-time value: the trunk-coupled self-eval head ships with the model as a confidence-score artifact at zero-marginal cost.

**Honest gaps (foregrounded).** Two largest:

1. **Mechanistic overlap with #56, #58, #59.** The "filtered self-improvement loop" *is* METAGEN's filter pipeline with the filter replaced by a self-eval head. The "high-confidence-correct outputs added to training corpus" *is* DISTILL-FORWARD with a binary correctness target instead of a soft-distribution KL target. Independent contribution at paradigm depth 22 is small.
2. **Self-grading reliability is the empirical risk.** Recent work (Madaan 2023 Self-Refine, Saunders 2022 Self-Critique, Huang 2024 *LLMs Cannot Self-Correct Reasoning Yet*) reports mixed-to-negative results on naive self-grading at 7B–70B scale. SAGE's promise is contingent on whether a 1.84B+ trunk's self-eval head can reliably distinguish correct from incorrect outputs on its own generations — a non-trivial empirical question the literature has not resolved.

**Engineering scope.** ~600 LOC over ~3 weeks (self-eval head ~150, audit-cycle pipeline ~200, MixCorpus filter integration ~80, CLI/monitoring ~60, Gate-0 harness ~110).

---

## 1. Self-evaluation head: architecture and joint loss

### 1.1 Head architecture

```
SAGE head:  h_pool ∈ ℝ^d  →  W_1 ∈ ℝ^{d × d_SAGE}  →  GeLU  →  W_2 ∈ ℝ^{d_SAGE × 1}  →  σ  →  ŝ ∈ [0, 1]
```

`h_pool` is the mean-pooled last-hidden-state vector across the generated output tokens (prompt-excluded). Parameters at 1.84B (`d = 2048, d_SAGE = 1024`): ~2.1M (0.11% of trunk); at 18B: ~4.2M (0.023%); at 144B-effective MOSAIC: shared head ~5M aggregate (0.003%). All within the same 10M-parameter budget #59 PRM-CHIRON operates inside.

The head fires **once per generated output**, not once per token: 50k forward passes per audit vs ~12.8M token forward passes from the trunk. Head wall-clock cost ≤ 0.4% of audit wall-clock.

### 1.2 Joint loss

```
L(θ, φ) = L_CE(θ) + λ_SAGE · L_eval(θ, φ),

L_CE(θ)         = −∑_t log P_θ(x_t | x_<t),                                    [primary]
L_eval(θ, φ)    = −∑_o ∈ audit_outputs [ y_o · log ŝ_o + (1 − y_o) · log(1 − ŝ_o) ],   [auxiliary]
```

where `y_o ∈ {0, 1}` is the ground-truth correctness label for output `o` (sourced from MC-rollout / verifier / human, §1.4) and `ŝ_o = f_φ(h_pool,o)` is the SAGE prediction.

`λ_SAGE = 0.1` default; tuning range `[0.05, 0.3]`. Per-output gradient norm bound (analogous to #59 PRM-CHIRON §1.5): `||∂L_eval/∂h_pool|| ≤ 0.025 · λ_SAGE`. At `λ = 0.1` this is `0.0025`, well below the CE gradient norm; no LR retuning required.

### 1.3 Why whole-output grading and not per-step

Granularity is the architectural distinction from #59 PRM-CHIRON. PRM uses per-step labels on reasoning chains for intermediate-step credit assignment (Lightman 2023: 5.8pp lift on MATH-500 over ORM); SAGE uses whole-output labels for *training-corpus-inclusion* decisions — we want to ask "is this output overall good enough to train on?" not "which token within is wrong?" Same granularity as #58 METAGEN's filter classifier. The two are not mutually exclusive; composed, PRM provides per-step gradient on reasoning chains while SAGE provides per-output gradient on full generations, constrained only by the ~0.5 cap on `||∇L_aux|| / ||∇L_CE||` (in practice `λ_PRM = 0.10` + `λ_SAGE = 0.05`).

### 1.4 Ground-truth correctness labels and warmup

The self-eval head requires labels to train against. SAGE uses three sources in priority order: (a) **MC-rollout-with-final-answer-check** for math/code/QA prompts — `K = 8` rollouts; correct iff the final answer matches the majority vote (Math-Shepherd primitive shared with #59; ~85% label accuracy; ~30% prompt coverage); (b) **auxiliary verifier model (~7B)** for open-ended prompts — RewardBench-class scoring (~75% accuracy; ~50% coverage); (c) **human-curated label set** of ~50k pairs as gold-standard hold-out and disagreement arbiter (~95% accuracy; ~5% coverage). **Total label coverage ~85%**; uncovered 15% appears in CE but not in the SAGE auxiliary loss.

The head cannot grade meaningfully at step 0; SAGE uses the same three-phase curriculum #59 PRM-CHIRON uses. **Phase 0 (steps 0 → 0.05·N):** `λ_SAGE = 0` (pure CE; head receives no gradient). **Phase 1 (0.05·N → 0.3·N):** `λ_SAGE` ramps from 0 to 0.1; audit cycle collects labels and trains the head, but does *not* add outputs to the corpus. **Phase 2 (0.3·N → N):** `λ_SAGE = 0.1`; audit cycle adds high-confidence-correct outputs to the corpus and oversamples low-confidence prompts. Phase-2 entry is gated on the head reaching ≥ 0.75 AUROC on the held-out correctness set. If the threshold is not met by step `0.3·N`, the corpus-filtration step is disabled and SAGE reduces to a pure auxiliary loss.

---

## 2. Filtered self-improvement loop: the closed-loop mechanism

### 2.1 Audit cycle

Every `N_audit = 10000` training steps:

1. **Generate.** Trunk runs inference (KV-cached, batch=8) on a held-out prompt set of `n_prompts = 50000` validation queries. Produces `n_outputs = 50000` outputs at average length 256 tokens. Cost: ~12.8M tokens at ~3500 tok/s × 8 batch = ~7 GPU-minutes.
2. **Self-grade.** SAGE head forward on each output. Output `(prompt_o, output_o, ŝ_o)` triples. Cost: negligible (SAGE head is ~2M parameters; 50k forwards = ~6 GPU-seconds).
3. **Filter.** Three buckets:
   - `ŝ ≥ 0.85` (estimated ~25% of outputs at maturity): high-confidence correct. Added to training corpus as `(prompt_o, output_o)` pairs with weight `w = 0.6` (down-weighted vs real tokens to prevent confidence-runaway, mirroring METAGEN §4.3).
   - `ŝ ∈ [0.30, 0.85]` (estimated ~50%): ambiguous. Discarded.
   - `ŝ ≤ 0.30` (estimated ~25%): low-confidence. The *prompts* (not the outputs) are added to a oversample buffer; the next 1000 training steps draw 10% of their prompts from this buffer to give the trunk more practice on hard cases.
4. **Update SAGE.** A subset of audit outputs (those with available ground-truth labels via §1.4) are added to a rolling SAGE training buffer. The SAGE head is updated for ~500 SGD steps on this buffer to keep its judgments calibrated against ground truth.

Total audit cycle wall-clock at 10k-step interval: ~10 minutes / 10000 steps = ~0.06% wall-clock fraction at 1.84B / 4080S throughput. **Negligible per-step amortized cost.**

### 2.2 Down-weighted-positive + prompt-oversample mitigation

Adding model-generated outputs to the training corpus risks the same model-collapse failure mode #58 METAGEN flags (Shumailov 2024): trunk over-fits to its own confident outputs; output distribution narrows; tail dies. Two mitigations: **(1) down-weight to `w = 0.6`** — output-as-positive has 60% of a real token's loss weight (mirrors METAGEN §4.3's `w_synth = 0.7`); **(2) oversample prompts not outputs** on low-confidence cases — the trunk re-encounters the *prompt* with the original real-data target, injecting difficulty signal without adding a single self-generated token to the training distribution. Combined, SAGE's closed-loop signal adds at most ~5–10% of total tokens as self-generated (vs METAGEN's 70–90% synth-share at maturity). **SAGE is not a corpus-replacement paradigm; it is a gradient-shaping signal with a small corpus-augmentation side effect.**

### 2.3 The self-eval head as a training-time difficulty oracle

The deeper purpose of the audit cycle is *prompt-difficulty estimation*. The trunk's confidence on its own outputs is a proxy for hard vs easy prompts; the oversampling rule turns this into an active-learning signal at the prompt level. Structurally similar to **#57-A SCROLL** (gradient-norm difficulty curriculum) and **METAGEN §3.4** (adversarial filter retraining). SAGE's contribution on this axis is *cheaper than #57-A* (no per-step gradient-norm logging required) but *less informative* (whole-output confidence is noisier than per-step gradient norm). Marginal value: small.

---

## 3. Distinction from #56 DISTILL-FORWARD, #58 METAGEN, #59 PRM-CHIRON

This section is the most important honest-accounting section of the document. SAGE has substantial mechanistic overlap with three already-shipped paradigms; the marginal contribution must be defended carefully.

### 3.1 vs #56 DISTILL-FORWARD

**#56:** trains the student against a *frozen previous-generation teacher's output distribution* via KL at the loss target. **SAGE:** trains the trunk to predict its *own* output correctness via binary cross-entropy at an auxiliary head; uses high-confidence-correct outputs as additional CE positives.

**Distinction.** #56's teacher signal is *external* (previous-generation knowledge shaping the current trunk's gradient); SAGE's signal is *self-generated* (the trunk audits its own behavior with no external teacher). **#56 imports knowledge; SAGE introspects on existing knowledge.** **Overlap.** Both operate on model-generated tokens; both add per-output auxiliary signal biasing the trunk toward "correct-looking" outputs. Joint composition is sub-multiplicative — what the student can self-assess is by construction less than what the teacher provides via #56's KL. Realistic joint multiplier: `1.05–1.10×` over post-#56 alone.

### 3.2 vs #58 METAGEN

**#58:** generates synthetic data from a previous-generation teacher; filters via a separately-trained multi-criterion discriminator (quality + perplexity + n-gram novelty); trains on `D_real ∪ D_synth_filtered`. **SAGE:** generates outputs from the *current-generation* trunk during training; filters via the trunk's own self-eval head; trains on `D_real ∪ {audit-outputs ŝ ≥ 0.85}`.

**Distinction.** #58 uses an *external* teacher + *separately-trained* filter; SAGE uses the *current-training* trunk as both generator and filter. Loop tightness differs (#58: per-generation; SAGE: per-audit-cycle, every 10k steps within a single run). **Overlap.** **The mechanism is largely the same: model-generated outputs filtered for quality and added to the training corpus.** Differences are (a) granularity of the loop (per-generation vs intra-run), (b) filter source (separate discriminator vs auxiliary self-eval head), (c) synth-share volume (METAGEN: 70–90%; SAGE: 5–10%). Realistic joint multiplier: `1.05–1.15×` over post-#58 alone.

### 3.3 vs #59 PRM-CHIRON

**#59:** auxiliary classifier head trained jointly with the trunk on *step-level* correctness labels from Math-Shepherd MC-rollouts. Per-step gradient on reasoning chains; granularity ~30 tokens/step. **SAGE:** auxiliary classifier head trained jointly with the trunk on *whole-output* correctness labels (MC-rollout + verifier + human curation); per-output gradient; granularity ~256 tokens/output.

**Distinction.** PRM operates on reasoning *trajectories* (intermediate-step credit assignment); SAGE operates on full *generations* (output-level filtration). **Granularity is the architectural distinction.** PRM cannot drive a corpus-augmentation loop because its labels are step-internal — high-confidence steps don't compose into a high-confidence trajectory without further integration. SAGE's whole-output signal naturally drives the audit-cycle filter. **Overlap.** Both use the same ~10M-param auxiliary-classifier-head pattern; both train against MC-rollout-derived labels (SAGE source 1 = PRM's Math-Shepherd primitive); both provide auxiliary gradient at `λ ≤ 0.1`. When composed, `λ_PRM = 0.10` + `λ_SAGE = 0.05` keeps `||∇L_aux|| / ||∇L_CE||` below the ~0.5 ceiling. Realistic joint multiplier: `1.10–1.20×` over post-#59 alone (different granularity preserves more independence).

### 3.4 Net independent contribution

vs **post-#56 + #58 + #59** combined baseline (the actual production stack), SAGE's marginal contribution is the geometric mean of the partial-stack overlaps: **`1.10–1.20×`**. **This is the headline `1.2×` claim.** The conservative `1.2×` reflects the depth-22 reality: the bigger-picture track has already shipped most of the closed-loop self-curation territory SAGE is moving on. SAGE is not a paradigm-defining shift; it is a structural-optimization bolt-on at the per-output granularity that the existing per-token (DISTILL), per-step (PRM), and per-corpus (METAGEN) paradigms leave underexplored.

---

## 4. Composition with #1–#62 — multiplicative on uncovered axes; sub-multiplicative on overlap

SAGE composes **multiplicatively** with the per-step kernel paradigms (#42 SCFA, #44 MELT, #50 HELIUM FA-3, #52 PHOENIX-NF4), the architecture paradigms (#53 MOSAIC-MOE, #54 NEXUS-SSM), the optimizer paradigm (#55 SOPHIA-CHIRON), the schedule paradigms (#60 COSMIC, #61 COSMIC-PROMOTED), and the agency paradigm (#62 AGENT-CHIRON) — none of these touch the model-output-as-training-signal axis SAGE moves on.

SAGE composes **sub-multiplicatively** with #56 DISTILL-FORWARD (~0.4× of standalone surviving), #57 SCROLL (~0.6×), #58 METAGEN (~0.3×), and #59 PRM-CHIRON (~0.7×). Geometric-mean composition with all four overlap paradigms simultaneously drops SAGE's ~1.5× aggressive standalone claim to the ~1.2× headline.

---

## 5. Honest gap and Gate-0 protocol

### 5.1 The two big honest risks

**Risk 1 — Mechanistic overlap is the dominant constraint.** §3.1–§3.4 detail this. The depth-22 reality is that the bigger-picture track has saturated most of the closed-loop self-curation axis. SAGE's marginal contribution is bounded above by ~1.5× and conservatively ~1.2×. **If joint speedup measured at Gate-1 falls below 1.10×, SAGE should be rejected as a paradigm shift and reduced to a deployment-time confidence-scoring feature** (which is its real distinguishing value).

**Risk 2 — Self-grading reliability is the empirical risk.** Literature is mixed: *Self-Refine* (Madaan 2023) reports self-correction improves outputs ~10–20% of the time on some tasks and degrades on others; *Self-Critique* (Saunders 2022) reports ~50–60% self-error-identification accuracy on summarization; *LLMs Cannot Self-Correct Reasoning Yet* (Huang 2024) reports self-grading on math reasoning fails to improve over no-grading at 7B–70B scale. SAGE's promise is contingent on the self-eval head reaching ≥ 85% AUROC. If it plateaus at ≤ 75%, the audit-cycle filter is too noisy to drive a reliable closed loop, and the paradigm reduces to a pure auxiliary-loss term contributing ~1.05× — barely worth the engineering. The Gate-0 protocol below is structured around this empirical question.

### 5.2 Gate-0 protocol (~30 GPU-hours)

**Question.** *On 66M CHIRON × 30k steps, does SAGE's self-eval head reach ≥ 0.80 AUROC and does the joint stack show ≥ 1.10× wall-clock speedup at fixed validation NLL vs the pre-#63 baseline?*

Three arms × 30k steps at 66M: **A (control)** = pre-#63 stack, no SAGE, target NLL ≈ 3.95 nat; **B (SAGE auxiliary-only)** = stack + SAGE head + `λ = 0.1` + labels, audit-cycle filtration *disabled* (tests whether auxiliary loss alone provides gradient-shaping benefit); **C (SAGE full)** = B + audit-cycle filtration enabled at step 9000 (tests full closed-loop mechanism).

**Pass criteria:** (1) SAGE head AUROC ≥ 0.80 by step 20k; (2) Arm B NLL within 0.05 nat of Arm A; (3) Arm C wall-clock to NLL(Arm A at 30k) ≤ 0.85 × Arm A; (4) no collapse signature in Arm C (Renyi entropy drop ≤ 10% across audit cycles, mirroring METAGEN §3.5).

**Fail-fast:** AUROC ≤ 0.65 at step 20k → REJECT (paradigm dead). Arm C NLL ≥ Arm A + 0.10 nat → REJECT (closed-loop introduces NLL drift). Renyi drop > 15% → REJECT (collapse). Arm C speedup ≤ 1.05× → MARGINAL, downgrade to 1.05× headline. Arm C speedup ≥ 1.30× → STRONG PASS; proceed to Gate-1 at 1.84B.

**Cost.** ~30 GPU-hours = 1.25 GPU-day (18h training × 3 arms + 4h generation + 8h labelling). Slightly above #59 PRM-CHIRON's 24h Gate-0, commensurate with the additional closed-loop mechanism complexity.

**Gate-1 (post-pass):** 1.84B / 100k steps with full label coverage (~85%). Target ≥ 1.15× speedup at fixed NLL, no collapse over 5 audit cycles. ~6 GPU-days. **Gate-2 (post-Gate-1):** joint composition with #56 + #58 + #59. Target marginal over full stack ≥ 1.05×; below this, downgrade to deployment-feature classification.

---

## 6. Engineering: ~600 LOC over ~3 weeks

| Component | Files | LOC | Week |
|---|---|---|---|
| SAGE head architecture (forward + backward, mirrors PRM head) | `Networks/sage_head.cpp`, `.h` | 100 | 1 |
| Joint-loss CUDA kernel (CE + λ·SAGE) | `cuda/sage_joint_loss_kernel.cu` | 50 | 1 |
| Audit-cycle generation pipeline (KV-cached batched generation) | `Networks/sage_audit.cpp`, `.h` | 200 | 1–2 |
| Filter integration with MixCorpus from #58 | `DataObjects/sage_corpus_extension.cpp` | 80 | 2 |
| Ground-truth label sources (MC-rollout + verifier + human-set loaders) | `Networks/sage_labels.cpp`, `.h` | 100 | 2–3 |
| CLI flags (`--sage 1`, `--sage-lambda 0.1`, `--sage-audit-interval 10000`, `--sage-threshold-pos 0.85`) | run.sh, argparse | 40 | 3 |
| Gate-0 harness + AUROC eval + collapse asserts | `unit-tests/.../sage_test.cpp` | 30 | 3 |
| **Total** | | **~600** | **~3 weeks** |

**Public API.** Mirrors PRM-CHIRON pattern: `glades::sage::SAGEHead` (forward / backward / `auroc_eval` / save / load) + `glades::sage::AuditCycle` (constructor takes trunk + head + MixCorpus; `run(current_step)` performs generate → grade → filter → update-SAGE).

**Risks and mitigations** (engineering). SAGE-head gradient bounded by `λ · σ'(z) · ||W_2||_F ≤ 0.025 · λ`; auto-reduce `λ` if `||∇L_SAGE|| / ||∇L_CE|| > 0.5`. Cold-start handled by Phase-0 warmup. Collapse-detection via Renyi-entropy monitor on audit outputs (reused from METAGEN). Ground-truth label drift handled by rolling SAGE-buffer retraining every audit cycle. Missing SAGE checkpoint falls back to CE-only (`λ_SAGE = 0`).

---

## 7. Summary

SAGE is **the auxiliary self-evaluation primitive at training time**: a small (~10M-parameter) classifier head co-trained with the trunk that predicts whether the trunk's *whole* outputs are correct, plus an audit-cycle pipeline that uses the head's predictions to filter self-generated outputs into the training corpus.

**Training-side speedup at fixed final NLL: 1.2× conservative; 1.5× aggressive contingent on self-eval AUROC ≥ 0.85.** Smallest training-FLOP multiplier in the #63 slate by design. NLL preservation: yes, via `λ_SAGE ≤ 0.1` auxiliary-loss formulation. Per-step overhead ~0.05%; engineering scope ~600 LOC over ~3 weeks (re-uses ~40% of #59 PRM-CHIRON code and ~25% of #58 METAGEN's filter pipeline).

**Honest mechanistic-overlap accounting (the paradigm's largest weakness).** §3 details substantial overlap with three already-shipped paradigms: #56 DISTILL-FORWARD (training-on-model-output at the target side; SAGE replaces the external teacher with same-generation self-grading; joint multiplier ~1.05–1.10× over #56-only), #58 METAGEN (SAGE's audit-cycle filtration *is* METAGEN's pipeline at finer time-grain with the filter replaced by an auxiliary self-eval head; ~1.05–1.15× over #58-only), and #59 PRM-CHIRON (same head pattern at per-output rather than per-step granularity; ~1.10–1.20× over #59-only). Geometric-mean composition with all four overlap paradigms simultaneously yields the ~1.2× headline. **SAGE is not a paradigm-defining shift at depth 22; it is a structural-optimization bolt-on at the per-output granularity that existing paradigms leave underexplored.**

**Honest empirical risk: self-grading reliability.** Recent literature (Madaan 2023 Self-Refine, Saunders 2022 Self-Critique, Huang 2024 *LLMs Cannot Self-Correct Reasoning Yet*) reports mixed-to-negative results on naive self-grading at 7B–70B scale. SAGE's promise is contingent on the self-eval head reaching ≥ 85% AUROC. If AUROC plateaus at ≤ 75%, the audit-cycle filter is too noisy to drive a reliable closed loop, and SAGE reduces to a pure auxiliary-loss term contributing ~1.05× — borderline-rejection territory.

**Recommended action.** Develop SAGE through Gate-0 to obtain the empirical AUROC and joint-speedup measurement. **Strong-pass** (AUROC ≥ 0.85, marginal speedup ≥ 1.30×) → promote to Gate-1. **Marginal-pass** (1.05–1.20× marginal speedup) → downgrade SAGE from paradigm-shift status to a deployment-feature contribution (the self-eval head ships with the model as a confidence-score artifact at zero-marginal training cost). **Fail** (AUROC ≤ 0.75 or marginal speedup ≤ 1.05×) → reject; reserve the auxiliary-self-eval-head primitive for a later paradigm where its granularity matches a less-saturated mechanism.

The 1.2× headline is modest. The paradigm depth is 22. The bigger-picture track has saturated most of the closed-loop self-curation axis. **SAGE is the candidate that most clearly admits this, and offers the smallest claim that is most honestly defensible.**
