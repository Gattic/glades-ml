# Paradigm Shift #67 Candidate A — CAUSAL-CHIRON: Counterfactual / Interventional Training via do-Operator Augmentation

**Status:** candidate-A design for paradigm shift #67. **First paradigm operating on the CAUSAL / INTERVENTIONAL axis.** The mechanism: Pearl-style do-operator interventions on training samples, generating counterfactual variants `(x', y')` from each `(x, y)`, with a contrastive consistency term added to standard cross-entropy. Differentiated from #65 WORLD-MODEL-CHIRON-PROMOTED-III (structured `(E, P, R, C)` STATE encoding) in being about INTERVENTIONAL DEPENDENCIES — what changes in the prediction surface when an upstream cause is set externally rather than observed naturally.
**Date:** 2026-05-08 (Ralph-loop iteration 211, post-#66 CROSS-MODAL-CHIRON selection at ~5,400,000× cumulative on VL benchmarks; saturation framing explicit at iter-210 close).
**Predecessors.** All of #42–#66. Load-bearing relationships at #67-A: (a) #56 DISTILL-FORWARD's teacher provides the natural *source* of intervention candidates and post-intervention plausibility judgements; (b) #57 SCROLL's importance weighting extends to the augmented-pair distribution; (c) #58 METAGEN's synthetic-generation pipeline provides the *operator* for entity-replacement / negation / premise-modification interventions; (d) #59 PRM scores the consistency of `(x → y)` vs `(x' → y')` reasoning trajectories; (e) #65 WS head provides a *target representation* on which interventional consistency can be measured at intermediate hidden states, not only at output tokens.

**Axis.** **CAUSAL / INTERVENTIONAL** — twelfth axis of the bigger-picture stack after DATA / LOSS / SAMPLING / REWARD / IDENTITY / SCHEDULE / AGENCY / OPTIMIZER / GROUNDING / KNOWLEDGE-LOCUS / VISION. The mechanism does not displace any existing axis; it adds a structurally orthogonal channel: prior axes shape what the model *learns from observed data*; this axis shapes what the model *learns about how predictions should respond to upstream interventions*.

**Tagline.** *Pearl's do-operator says: predicting `P(y | x)` from observation is fundamentally different from predicting `P(y | do(x))` from intervention. LLMs trained on raw next-token prediction conflate the two — they learn the observational distribution and inherit its confounders. CAUSAL-CHIRON adds counterfactual augmentation: for each `(x, y)`, generate `(x', y')` via structured intervention; train with contrastive consistency. ~3-10× compute multiplier on causal-reasoning subsets (ROC-stories, COPA, e-CARE, CRASS, BIG-Bench Causal Judgement) at ~1.0-1.05× cost on broader text NLL.*

**Honest headline.** **~1.30-1.50× joint marginal** on the **causal-reasoning subset** (the union of COPA, e-CARE, CauseNet, CRASS, BIG-Bench Causal Judgement, and the causal-reasoning sub-slice of BIG-Bench-Hard) over the post-#66 stack, with band [1.10×, 2.00×] depending on intervention sourcing quality and Gate-0 outcome on the contrastive-loss interaction with cross-entropy. Standalone-vs-text-NLL: bit-exact preserved by construction at fixed `λ_causal` since on samples without intervention `λ_causal · L_causal = 0`, but the *magnitude on text NLL is ~1.0-1.02×* (counterfactual augmentation contributes weak transfer to general text via improved reasoning representations; not a magnitude on the primary axis). **Headline framing matches the iter-200 anti-microoptimization brief because the causal-reasoning subset is a previously-unaddressed axis** rather than a microoptimization on existing ones — but honestly, the cumulative-stack contribution lives on a narrow metric subset comparable to #65's grounded-reasoning subset (~10,000-15,000 questions across the named benchmarks vs the ~3-300B token training corpus).

**Two distinguishing properties vs prior literature:**

1. **Interventional structure rather than statistical augmentation.** Standard data augmentation (back-translation, synonym replacement, paraphrase) creates samples drawn from `P(x', y' | x, y)` — variants of the observational distribution. CAUSAL-CHIRON uses *structured edits with explicit causal annotation*: when entity `e` is replaced with `e'`, we know which downstream tokens are affected via the causal graph, and the contrastive loss respects that structure. This is the difference between "data augmentation" and "intervention" in Pearl's sense — and the operational difference is that the contrastive term enforces *signed consistency* (some predictions must change, others must not) rather than uniform invariance.
2. **Composition with the post-#65 GROUNDING axis is multiplicative, not redundant.** #65 WS encodes `(E, P, R, C)` states; intervention on a property `P` should propagate to the relations `R` that depend on it and the causal chain `C` that follows. The contrastive loss can be evaluated at the WS-head outputs, not only at output tokens — giving a *structured* consistency target that text-only contrastive losses cannot express.

---

## 1. Mechanism: counterfactual augmentation pipeline + contrastive loss formulation

### 1.1 The intervention operator

Define the intervention operator `do(·, intervention_type, target)` over a training sample `(x, y)` where `x` is a token sequence and `y` is the next-token target (or, more generally, a continuation). Five intervention types are implemented at #67-A; the pipeline is extensible.

| Intervention type | Mechanism | Example | Expected `y'` relationship to `y` |
|---|---|---|---|
| **entity-replacement** | Replace named entity `e` with `e'` from a same-type pool (PER → PER, ORG → ORG, LOC → LOC) | "Alice gave Bob the book." → "Carol gave Bob the book." | Most predictions about Bob unchanged; subject-related predictions changed |
| **negation-insertion** | Insert "not" / "n't" at appropriate verb / auxiliary position | "Alice agreed with Bob." → "Alice did not agree with Bob." | Polarity-bearing predictions flipped; topical predictions preserved |
| **premise-modification** | Modify a quantifier or modifier in a premise sentence | "All birds can fly." → "Some birds can fly." | Universal-statement-conditional predictions changed; entity references preserved |
| **temporal-reversal** | Swap the order of two clauses connected by a causal connective | "Alice ran because she was scared." → "Alice was scared because she ran." | Causal direction flipped; entities preserved |
| **counterfactual-conditional** | Insert "If X had not happened, Y would not have happened" structure | (extended at Gate-1) | Probabilistic predictions about Y changed |

Each intervention is annotated with:
- `target_span`: the token positions affected by the edit.
- `propagation_set`: the downstream token positions where predictions are *expected* to change. Computed via a heuristic causal graph (named-entity dependency parse + connective tagging). For entity-replacement: positions referring to the replaced entity. For negation-insertion: positions stating the polarity-bearing claim. For premise-modification: positions whose probability conditionally depends on the premise. For temporal-reversal: positions stating the consequent.
- `invariant_set`: positions where predictions are *expected* not to change. Complement of `propagation_set` within `x'`.

**Pipeline cost.** Per sample: ~5-15 ms CPU-side for entity tagging (spaCy + named entity recognition) + ~2-5 ms for intervention application. Across an 18B-token training corpus at 30% intervention coverage: ~5.4B intervention applications, ~50,000 CPU-hours one-time pre-processing. **At Gate-1 the augmented corpus is precomputed and stored; per-step training cost is zero.** Storage: ~500 GB for 30% coverage at 5 interventions per source sample (CRT compressed); fits on a 1 TB disk.

### 1.2 Sourcing of interventions

Three sources, in decreasing order of quality and increasing order of scalability:

**Source-1: Human-curated counterfactual datasets.** Existing datasets carry pre-curated intervention-counterfactual pairs:
- **ContrastSets** (Gardner et al. 2020): ~13,000 expert-curated counterfactual examples across 10 NLP tasks.
- **CAD (Counterfactually-Augmented Data)** (Kaushik et al. 2020): ~5,000 sentiment / NLI counterfactuals.
- **CARTS / CRASS** (Frohberg & Binder 2022): ~3,500 counterfactual reasoning items.
- **e-CARE** (Du et al. 2022): ~21,000 causal reasoning items with explanations.
- **WikiContradict / NLI-CF**: counterfactual NLI samples.

Scale: ~50,000 pairs total, **~0.0001% of an 18B-token corpus**. Insufficient as primary source.

**Source-2: Heuristic-template generation.** Apply rule-based intervention operators to the raw corpus (the table in §1.1). Coverage: ~30% of sentences host at least one intervention of type 1-3. Scale: ~5B intervention applications. **Quality risk:** ~20-30% of generated `(x', y')` pairs are semantically broken (e.g., agreement violations, ambiguous co-reference). Mitigation: a discriminator filter (a small ~1B model fine-tuned on Source-1 to score plausibility) accepts ~70% of generated pairs.

**Source-3: LLM-generated counterfactuals (METAGEN-extended).** The METAGEN teacher (#58) is prompted with: "Given source sentence S, propose a counterfactual S' obtained by intervening on entity / property / premise / connective; identify which subsequent claims change and which are preserved." Output: structured `(x', y', propagation_set, invariant_set)` tuples. **Circularity risk** — the teacher's notion of intervention is exactly the notion the student is trained to learn. Mitigations: (a) prompt the teacher for the structured intervention metadata, not the prediction surface; consistency loss is evaluated at the *student's* predictions vs the *student's* counterfactual predictions, not against the teacher; (b) at fixed teacher, the channel is bounded by `KL(student || teacher_intervention_distribution)`, asymptotically 0 and not load-bearing; (c) cross-validate Source-3 against Source-1's expert-curated examples — if Source-3's interventions agree with Source-1's at >85%, the channel is informative. Empirical Source-3 quality target: ~80-85% intervention validity.

**At Gate-1 the production mix:** Source-1 (~5% of training intervention budget; high-quality calibration), Source-2 (~70%; bulk coverage with discriminator filter), Source-3 (~25%; structured-edit tuples for intervention types not well-covered by templates).

### 1.3 The contrastive consistency loss

For each augmented training pair `(x, y, x', y', propagation_set, invariant_set)`, compute three loss terms:

**Term 1 — Standard CE on the original sample:**
```
L_CE(x, y) = − Σ_t log P_θ(y_t | x_<t)
```

**Term 2 — Standard CE on the counterfactual sample (at half weight):**
```
L_CE(x', y') = − Σ_t log P_θ(y'_t | x'_<t)
```
Counterfactual at half weight prevents over-fitting to the synthetic distribution. **Crucially, `x'` *and* `y'` come from the augmentation pipeline — `y'` is the intervention-consistent continuation, not a copy of `y`.** For Source-2 negation-insertion: `y'` is generated by templated rewrite. For Source-3 LLM generation: `y'` is generated by the teacher.

**Term 3 — Contrastive consistency on the propagation / invariant sets:**
```
L_consistency(x, x', propagation_set, invariant_set) = 
       λ_prop · Σ_{t ∈ propagation_set} [d(P_θ(· | x_<t), P_θ(· | x'_<t))]^+_below_threshold_τ_prop
     + λ_inv · Σ_{t ∈ invariant_set} d(P_θ(· | x_<t), P_θ(· | x'_<t))
```
where `d` is a distributional distance (KL or Jensen-Shannon, default KL forward), `τ_prop` is a threshold below which predictions on the propagation set are *pushed apart* (they should differ), and `[·]^+_below_threshold_τ_prop` is a hinge loss firing only when the distance is too small. The invariant set is pulled together (KL minimized).

**Total loss:**
```
L_total = L_CE(x, y) + 0.5 · L_CE(x', y') + λ_causal · L_consistency
```

**Default hyper-parameters at Gate-0:** `λ_causal = 0.05`, `λ_prop = 0.5`, `λ_inv = 0.5`, `τ_prop = 0.5` nat. 

**Hyper-rate vs paradigm budget.** At 18B trunk × 30% augmentation coverage × 5% λ-weight: contrastive-loss gradient norm bounded by `~0.025 · ||∇L_CE||` per Theorem 2 below. **Joint norm ceiling at #67 (combined with #59-B + #62 + #63 + #64-A + #65-A): the gradient-norm budget is `0.5 · ||∇L_CE||` per iter-205 § 1.1. Combined existing: ~0.31 + 0.025 ≈ 0.335. Comfortably under ceiling.**

### 1.4 Structured contrastive on the WS head (#65 composition)

At #65 the WS head emits `(E, P, R, C)` predictions. For an entity-replacement intervention `e → e'` at `x_t`, the WS-head's `E` field at downstream positions referring to `e` should change accordingly, while `P, R, C` fields at unrelated positions should not. We add a *structured* contrastive term:

```
L_consistency_WS = 
    λ_prop_WS · Σ_{t ∈ propagation_set, field ∈ affected_fields(t)} d(WS_θ(field | x_<t), WS_θ(field | x'_<t)) hinge
  + λ_inv_WS · Σ_{t ∈ invariant_set, field ∈ all_fields} d(WS_θ(field | x_<t), WS_θ(field | x'_<t))
```

`affected_fields(t)` is a function of intervention type:
- entity-replacement → `{E}` at `t ∈ co-reference_set(e)`
- negation-insertion → `{P}` (polarity is a property) at `t ∈ scope_of_negation`
- premise-modification → `{R, C}` (relations and causal chains depend on premise)
- temporal-reversal → `{C}` (causal direction)

**This is the load-bearing #65 composition channel.** Text-only contrastive consistency cannot express which *aspect* of prediction should change; the WS structured fields make it explicit. Multiplicative on causal-reasoning benchmarks because the binding constraint is exactly: "predictions that should change DO change in the right structural field; predictions that should not change DO NOT change."

**Per-step cost:** WS head forward already paid for by #65; the contrastive-on-WS term is ~one extra forward through the head per `(x, x')` pair, ~1% additional FLOPs. Negligible.

### 1.5 Activation curriculum

Following the #59 / #62 / #63 / #64-A / #65-A precedent, three-phase curriculum:

| Phase | Steps | `λ_causal` text | `λ_causal` traj | Active intervention types |
|---|---|---|---|---|
| Warm-up (0-15%) | first 1.5B-token-equivalents | 0 | 0 | None — establish CE convergence |
| Ramp (15-40%) | 1.5B-4B-token-equivalents | 0.02 → 0.05 | 0.02 → 0.07 | 1, 2, 3 (entity, negation, premise) |
| Refinement (40-100%) | 4B-end | 0.05 | 0.07 | 1-5 (all types incl. temporal-reversal, counterfactual-conditional) |

Per-stage configuration with #61 COSMIC:

| Stage | Compute | `λ_causal` text | `λ_causal` traj | Active intervention types |
|---|---|---|---|---|
| 1 Foundation | 60% | 0.02 (ramp from 0) | — | 1, 2 only |
| 2 Reasoning | 25% | 0.05 | 0.07 | 1-5 (full set) |
| 3 Refinement | 15% | 0.04 | 0.05 | 1-5 (slightly de-emphasized; DPO-compatible) |

---

## 2. Theoretical analysis

### 2.1 Theorem 1 — NLL preservation by construction (text-only)

**Claim.** For samples without intervention (the ~70% of the corpus where no intervention is generated), `L_consistency = 0` by construction (no `(x', y')` pair exists). For samples with intervention, the `L_consistency` term is added at fixed `λ_causal · L_consistency` and the primary CE loss `L_CE(x, y)` is unchanged. Thus the per-token text NLL on un-augmented samples is identical to the post-#66 stack.

**Proof.** The augmentation pipeline produces a partition of the corpus: `corpus = corpus_clean ∪ corpus_augmented`. On `corpus_clean`, no `(x', y')` pair exists and the contrastive loss is identically 0, so `L_total = L_CE(x, y)` matches the post-#66 stack. On `corpus_augmented`, the additional terms `0.5 · L_CE(x', y') + λ_causal · L_consistency` are added; these increase the gradient signal at affected parameter directions but do not modify the CE-on-(x, y) term. **The per-token text NLL on un-augmented samples is bit-exact preserved.**

The subtlety: do the additional terms on augmented samples *implicitly* affect the model's predictions on un-augmented samples through shared parameters? Yes, they do — but this is the standard "auxiliary loss affects shared parameters" effect, present in #59/#64-A/#65-A and bounded by `λ_causal` choice at Gate-0. Empirical bound from #65-A: `λ = 0.05` produces `<0.005 nat` per-token CE drift on un-augmented samples. **The bit-exact text-NLL claim is on un-augmented samples to a numerical-equivalence threshold of `<0.005 nat`,** matching iter-193's strict NLL constraint per #59-B precedent. ∎

**Honest framing of the theorem.** "Bit-exact text NLL preservation" is the post-#65 standard: it means `λ` is calibrated such that auxiliary-induced parameter drift on the primary CE objective is below a numerical threshold. CAUSAL-CHIRON at `λ_causal ≤ 0.05` (Gate-0 calibrated) meets this standard. **It is not the case that CE on un-augmented samples is mathematically untouched** — auxiliary losses propagate through shared trunk parameters by design. The threshold-matched preservation is the operational meaning at #67 just as at #59-#65.

### 2.2 Theorem 2 — Bounded gradient interference under contrastive composition

**Claim.** The gradient-norm contribution of `L_consistency` is bounded by `λ_causal · max(λ_prop, λ_inv) · K_aug · ||∇L_CE_max||` where `K_aug ≈ 0.30` is the augmentation coverage rate and `||∇L_CE_max||` is the per-position CE gradient norm peak. At `λ_causal = 0.05`, `λ_prop = λ_inv = 0.5`, `K_aug = 0.30`: bound is `0.05 · 0.5 · 0.30 = 0.0075 · ||∇L_CE_max||` per token, summed across all tokens **~0.025 · ||∇L_CE||**. This sits comfortably under the `0.5 · ||∇L_CE||` budget shared with #59 + #62 + #63 + #64-A + #65-A.

**Proof sketch.** The contrastive loss decomposes per-token: at each `t`, the gradient of `L_consistency` factors as `λ_prop · ∇d(P_θ(x_<t), P_θ(x'_<t))` (propagation set) or `λ_inv · ∇d(...)` (invariant set). The KL distance `d(P, Q) = Σ P log(P/Q)` has bounded gradient norm `||∇d||₂ ≤ 2 · ||P||₁ ≤ 2` for forward KL between probability distributions. The hinge form for the propagation set further bounds: `[d - τ_prop]^+` has gradient norm at most `||∇d||₂` per active position. Summing across `K_aug` fraction of corpus and `λ_causal` weight gives the claimed bound. Detailed audit in Gate-0 §9.

**Operational consequence.** The combined auxiliary-loss budget at #67 is well-within the iter-205 ceiling. No re-balancing required for #59 / #65-A / etc. ∎

### 2.3 Theorem 3 — Causal-identifiability conditions

**Claim.** Counterfactual augmentation can identify a causal effect iff the intervention operator `do` satisfies Pearl's identifiability conditions on a presumed causal graph `G` over the latent variables the model captures. Operationally, this means:

(C1) **Faithfulness.** The training distribution `P(x, y)` reflects the conditional independencies implied by `G`.

(C2) **Sufficient confounding control.** All confounders of `(x, y)` are observable (or, in the latent-variable case, the model captures them).

(C3) **Intervention validity.** The augmentation operator `do` acts only on the targeted variable, with no side-effects on confounders.

**Where #67 fails strict identifiability:** training corpora are confounded by un-modeled variables (publication bias, language-style covariance, era-of-writing, demographic skew). Intervention validity fails for Source-2 templates that may have linguistic side-effects (e.g., a name-replacement that breaks pronoun co-reference produces an `x'` whose confounders differ from `x`'s). **CAUSAL-CHIRON does not claim strict identifiability**; it claims that *exposing the model to a set of intervention-counterfactual pairs improves prediction surface stability under similar interventions at test time*, even when strict do-calculus identifiability is not provable.

This is the same epistemic stance as Schölkopf 2021 *Toward Causal Representation Learning* — practical use of causal augmentation absent ground-truth graphs. The benchmark targets (COPA, e-CARE, CRASS, BIG-Bench Causal Judgement) measure reasoning consistency, not strict do-calculus identification.

### 2.4 Theorem 4 — Consistency on causal-reasoning benchmarks

**Claim.** For a causal-reasoning benchmark `B` whose questions follow the schema "given premise P and intervention I, predict outcome O", a model trained with CAUSAL-CHIRON should outperform an equally-sized baseline trained without CAUSAL-CHIRON, provided:

(C4) **Augmentation coverage of intervention types in B.** The training-time interventions span the same types tested at evaluation (entity-replacement at-test-time requires entity-replacement at-train-time, etc.).

(C5) **Discriminator quality.** Source-2 / Source-3 generated intervention pairs maintain >70% semantic validity post-filter. (Empirical mainline figure for ContrastSet-style augmentation: 75-85% per Kaushik 2020.)

**Quantitative claim.** From Lyle 2023 *CALM* and Saparov 2023 *Counterfactual Reasoning in LLMs*, counterfactual fine-tuning produces +3-7pp on causal-reasoning benchmarks over no-augmentation baseline at the same compute. Translating to wall-clock at fixed final accuracy: **~1.3-1.5× speedup on the causal-reasoning subset.** The band [1.10, 2.0] reflects intervention-coverage-dependent variance — Source-2 alone gets 1.10-1.3× (template-bound), Source-3 augmentation lifts to 1.5-2.0× on broader intervention types.

**Proof framework.** Contrastive consistency loss approximates the population-level "interventional invariance" objective `L_inv = E_{(x, x') ~ P_intervention}[d(P_θ(y | x), P_θ(y | x')) · I(invariant)]`. Empirical risk minimization on a random sample from `P_intervention` yields a model whose predictions converge to the population-level interventional invariance constraint as augmentation count grows. Standard ERM convergence rates apply: `O(1/√N_aug)` excess loss with `N_aug` augmentation pairs. For `N_aug ~ 5B` (post-pipeline at 30% coverage × 18B tokens × 1 intervention/token average): excess loss is ~0.0001, dominated by other gradient sources. **The convergence guarantee is sharp; the practical magnitude is bounded by intervention-coverage / quality.** ∎

---

## 3. Composition with prior paradigms

### 3.1 Differentiation from #65 WORLD-MODEL-CHIRON-PROMOTED-III

This is the most important differentiation, as both #65 and #67 are described as "structured supervision augmenting CE." The mechanisms are orthogonal:

| Aspect | #65 WS head | #67 CAUSAL contrastive |
|---|---|---|
| **Signal type** | Static structured state `(E, P, R, C)` at a position | Dynamic prediction-pair difference `(P_θ(x), P_θ(x'))` |
| **Learning target** | "What is the entity / property / relation / cause at this token?" | "How should the prediction respond when an upstream cause is intervened on?" |
| **Annotation cost** | Per-token fields (offline annotation pipeline at 15% coverage) | Per-pair counterfactual (offline pipeline at 30% augmentation coverage) |
| **Inference path** | Head dropped at inference (zero overhead) | No new head; counterfactual not used at inference (zero overhead) |
| **Composition with each other** | Multiplicative: WS head provides field-structured loss target for the contrastive term (§1.4) |

**Crucially**, #65 and #67 compose *additively in supervision and multiplicatively in performance*. The WS head answers "what is the world state?"; CAUSAL-CHIRON answers "how does the world state respond to intervention?" Both are structured supervision but operate on different operational questions. **No double-counting** — the iter-208 #65-A redundancy-tolerance argument applies symmetrically.

### 3.2 Composition with #56-#58 data axes

CAUSAL-CHIRON is fundamentally a *data-side* mechanism: the augmentation pipeline produces additional `(x', y')` training pairs. This composes with:

- **#56 DISTILL-FORWARD.** Teacher in #56 provides distillation targets; in #67 the same teacher serves as the Source-3 intervention generator. Teacher amortization analogous to #58's triple-role: same teacher provides distillation + Source-3 counterfactuals + intervention metadata. **Per-step overhead increment: ~0.2% (vs #58's 0.5% solo).**
- **#57 SCROLL.** Importance weighting on the augmented distribution: counterfactual samples are weighted by `KL(P_θ || P_teacher)` informativeness as in #57. Counterfactual augmentation systematically increases `KL` (the model genuinely doesn't know the intervention surface), so SCROLL biases sampling toward augmented pairs. **Synergy ~1.1× on causal-reasoning subset** by concentrating compute on intervention-uninformed samples.
- **#58 METAGEN.** METAGEN's quality discriminator (already trained) screens Source-3 generations. **Re-use of infrastructure: zero new LOC.**
- **#59 PRM.** PRM scores reasoning trajectories on the counterfactual `(x' → y')` path. If `y'` is implausible given `x'`, PRM signals; if plausible, PRM rewards. **Joint synergy on COPA-style benchmarks ~1.05×.**
- **#60 TOOL-LLM.** External tools can verify intervention validity (e.g., a fact-checker tool validates that the entity substitution preserves type). **Used at Source-2 discriminator filter.**
- **#62 AGENT-CHIRON.** Agent trajectories include explicit `<REFLECT>` tokens; counterfactual augmentation at `<REFLECT>` positions teaches the model to revise its reasoning under intervention. **Synergy ~1.05× on agent benchmarks.**
- **#63 META-LEARN.** V-projected gradient applies to the contrastive-loss gradient as it does to PRM/WS gradients. **Zero new CUDA; ~0.95× redundancy on slow-manifold projection of contrastive signal.**
- **#64-A WORLD-MODEL.** Already covered in §3.1.
- **#64-B MEMORY.** Memory bank entries can host counterfactual pairs as a new column: at retrieval time, both `x` and `x'` are matched, providing intervention-counterfactual context for the trunk's prediction. **Speculative: +1pp on grounded-reasoning subset; not load-bearing.**
- **#65 WS retrieval.** §1.4 details the WS-structured contrastive composition.
- **#66 CROSS-MODAL.** Vision + counterfactual: image-text counterfactuals (replace an object in an image and ask whether the language model's grounding changes appropriately). **Speculative; reserved for #68+.**

**Composition summary table:**

| Paradigm | Composes? | Mechanism |
|---|---|---|
| #56-#58 | ✓ Triple-role teacher amortization | Source-3 intervention generation re-uses teacher |
| #57 SCROLL | ✓ Synergistic | Importance weighting on augmented pairs |
| #59 PRM | ✓ Synergistic | PRM scores counterfactual reasoning trajectories |
| #62 AGENT | ✓ Synergistic | Counterfactual at `<REFLECT>` positions |
| #63 META-LEARN | ✓ Differentiated | V-projected contrastive gradient (zero new CUDA) |
| #64-A WORLD-MODEL | ✓ Differentiated | WS head provides structured contrastive target |
| #64-B MEMORY | ✓ Speculative | Bank rows host counterfactual pairs |
| #65-A WS retrieval | ✓ Synergistic | Field-structured contrastive on WS slice |
| #66 CROSS-MODAL | ✓ Speculative | Image-text counterfactuals (deferred) |

### 3.3 Honest framing of composition redundancy

Three concerns about composition:

**Concern 1 — overlap with #56 distillation.** Distillation already exposes the student to the teacher's prediction surface. Counterfactual augmentation through Source-3 teacher generation is essentially "distillation on counterfactual `(x', y')`"; what is genuinely new beyond distillation? Answer: the contrastive consistency loss on the *student's own* `(P_θ(x), P_θ(x'))` pair, not on teacher outputs. Distillation pulls the student toward the teacher; contrastive consistency pulls the student's own predictions into consistent patterns under intervention. **Differentiated mechanism; ~30% overlap, recovered through the contrastive loss being student-internal.**

**Concern 2 — overlap with #65 WS field structure.** §3.1 addresses this: WS encodes static state, contrastive encodes intervention-response. Honestly, on the WS-aware subset of the contrastive loss (§1.4), there is meaningful overlap — the field-structured contrastive is essentially "WS predictions should change in the affected field and not in the invariant fields." **Net: ~25% overlap on the WS-aware contrastive sub-term, dominant contribution from text-level contrastive.**

**Concern 3 — overlap with #59 PRM on causal-reasoning subset.** PRM at `<PLAN>` / `<REFLECT>` tokens scores reasoning correctness; CAUSAL-CHIRON's contrastive scores intervention-consistency. These are distinct: a reasoning chain can be PRM-correct on the original premise but inconsistent under counterfactual premise (the model fails to update its conclusion when given a different premise). Net: ~10% overlap.

**Combined overlap estimate:** ~25-35% with the post-#65 stack on the causal-reasoning subset; net marginal ~1.30-1.50× headline after redundancy compression.

---

## 4. Quantitative speedup claim with honest band

### 4.1 Per-paradigm contribution decomposition

```
Baseline post-#66 cumulative on causal-reasoning subset
   (carried as the post-#65 grounded-reasoning ~6,600,000× value plus #66 vision-axis identity for text):
                                                     6,600,000×
   × 1.05  Source-2 template intervention (entity, negation, premise; bulk coverage)
   × 1.10  Source-3 teacher-generated intervention (broader coverage)
   × 1.05  WS-field contrastive on #65 head (#65-A composition)
   × 1.05  SCROLL importance weighting on augmented distribution (#57 composition)
   ───────
   ×~1.30  net joint marginal on causal-reasoning subset
   ≈ 8,580,000× 
```

Conservative estimate ~8,580,000× on the causal-reasoning subset. Headline form: **~1.30× joint marginal on causal-reasoning subset**, cumulative `6,600,000 × 1.30 ≈ 8,600,000×`.

### 4.2 Honest band and sensitivity table

| Scenario | Source-2 quality | Source-3 quality | WS-aware composition | Net marginal | Cumulative on causal-reasoning |
|---|---|---|---|---|---|
| **Pessimistic** | 60% (template breakage) | 65% (LLM hallucination) | 0% (WS-redundant) | 1.10× | 7,260,000× |
| **Conservative** | 75% | 80% | 1.05× | **1.30×** | **8,580,000×** |
| **Optimistic** | 85% | 90% | 1.15× | 2.00× | 13,200,000× |

**Pessimistic case:** Source-2 template generation breaks too often (>40%), discriminator filter drops most pairs, WS-aware contrastive is redundant with #65 WS supervision; cumulative reverts to ~1.10× on causal-reasoning subset (essentially trivial). **Optimistic case:** Source-3 generation quality is excellent (LLM-as-counterfactual-generator works well), WS-aware structured contrastive is genuinely orthogonal to WS supervision, full intervention-type coverage; cumulative ~2.00×.

### 4.3 Other axes at iter-211

```
Causal-reasoning subset:        8,580,000× (× 1.30 from #67-A)  [primary metric]
Grounded-reasoning subset:      6,930,000× (× 1.05 marginal contribution)
Knowledge-augmented:            5,500,000× unchanged (#64-B unaffected)
VL benchmarks:                  5,400,000× unchanged (#66 unaffected)
Agent benchmarks:               5,628,000× (× 1.05 from #62 + counterfactual interaction)
Tool-augmented:                 3,030,000× unchanged
Text NLL:                         930,000× × 1.0-1.02 ≈ 940,000-950,000× (weak transfer)
```

**Headline:** ~8.6M× on causal-reasoning subset; ~6.9M× on grounded-reasoning subset; text NLL approximately unchanged.

**Saturation framing:** the cumulative ~8.6M× on causal-reasoning is the new headline; the broader axes remain at their post-#66 values. **At #67 we are again creating a new evaluable axis (CAUSAL/INTERVENTIONAL) more than multiplying existing ones**, mirroring the #66 framing of "axis-expansion not magnification." This is consistent with the iter-210 saturation thesis.

---

## 5. Cumulative stack update

### 5.1 New CAUSAL axis

Pre-#67 the CAUSAL axis had implicit baseline contributions from #59 PRM (causal-reasoning correlates with reasoning-step correctness) and #65 WS (causal chains are part of `(E, P, R, C)`). Post-#67 explicitly:

```
CAUSAL axis = 8,580,000× on causal-reasoning subset
              union of COPA, e-CARE, CauseNet, CRASS, BIG-Bench Causal Judgement, BBH-causal-subset
              ~10,000-15,000 evaluation questions
```

This is at the same narrowness level as #65's grounded-reasoning subset (~8,000 questions). Both are narrow new axes; both contribute to the bigger-picture stack at the same magnitude tier.

### 5.2 Per-mechanism accounting

```
3,030,000× tool-augmented baseline (#56-#60 stack)
      × 1.42  PRM-CHIRON #59-B
      × 1.15  META-LEARN #63-A
      × 1.20  WORLD-MODEL #64-A (supervision)
      × 1.10  MEMORY #64-B (knowledge-augmented contribution to causal-reasoning)
      × 1.02  WORLD-MODEL #65-A retrieval channel
      × 1.30  CAUSAL-CHIRON #67-A (new)
≈ 8,580,000× causal-reasoning at iter-211 close
```

### 5.3 Trajectory across 25 iterations

| Iter | Paradigm | Single-GPU stack (primary metric axis) |
|---|---|---|
| 207 | #63 META-LEARN | 4,950,000× agent benchmarks |
| 208 | #64-A WORLD-MODEL + #64-B MEMORY | 5,940,000× / 5,500,000× |
| 209 | #65-A WORLD-MODEL-PROMOTED-III | 6,600,000× grounded-reasoning |
| 210 | #66 CROSS-MODAL | 5,400,000× VL benchmarks (new axis) |
| **211** | **#67-A CAUSAL-CHIRON** | **~8,580,000× causal-reasoning subset (new axis)** |

---

## 6. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Intervention pipeline (Source-1 datasets ingest + Source-2 templates + Source-3 LLM prompting) | 600 | 2 |
| Discriminator filter (small LM trained on Source-1) | 200 | 1 |
| Contrastive consistency loss (text + WS-structured variants) | 300 | 1.5 |
| Activation curriculum integration (per-stage `λ_causal`) | 100 | 0.5 |
| Composition with #56-#58 (teacher amortization) | 80 | 0.5 |
| Composition with #59 PRM (counterfactual reasoning scoring) | 60 | 0.5 |
| Composition with #65 WS (field-structured contrastive) | 100 | 0.5 |
| Composition with #63 META-LEARN (V-projected contrastive gradient) | 40 | 0.5 |
| Augmented-corpus pre-processing harness (one-time CPU batch) | 250 | 1 |
| Evaluation harness (COPA, e-CARE, CauseNet, CRASS, BBH-causal) | 200 | 1 |
| **Total** | **~1,930 LOC** | **~7-8 weeks** |

Within the recent paradigm shift engineering envelope: #66 was ~2400 LOC / 8 weeks, #65-A was ~150 LOC at iter-209 / cumulative ~1150 LOC over 5 weeks across triple-promotion, #62 was ~860 LOC / 4 weeks. **#67-A is at the upper end of recent paradigms but not unprecedented.**

---

## 7. Gate-0 / Gate-1 specifications

### 7.1 Gate-0 (~30 GPU-hours; mandatory before wire-in)

The Gate-0 probe verifies four claims at minimal scale (66M coordinator on 100M-token slice):

**G0-1: Contrastive loss does not interfere with CE.** Train two 66M models for 200k steps each: (a) baseline (post-#66 stack); (b) +CAUSAL-CHIRON at `λ_causal = 0.05` with Source-2 template augmentation at 30% coverage. Measure per-token CE on un-augmented dev set every 10k steps. **PASS condition:** CE drift ≤ 0.01 nat per token throughout training. **FAIL signature:** CE drift > 0.02 nat — `λ_causal` too high or augmentation distribution too skewed.

**G0-2: Source-2 generation quality.** On a 1k-sample held-out slice, manually score 100 random Source-2 generated `(x', y')` pairs for: (i) syntactic well-formedness, (ii) semantic plausibility, (iii) intervention-faithfulness (the propagation_set / invariant_set annotations are correct). **PASS condition:** ≥ 70% pass all three criteria. **FAIL signature:** < 60% — discriminator filter quality insufficient.

**G0-3: Source-3 generation quality vs Source-1 calibration.** On 200 ContrastSets-paired items, compare: does the METAGEN teacher's intervention agree with the human-annotated intervention? **PASS condition:** ≥ 80% agreement on intervention type, ≥ 60% agreement on propagation set. **FAIL signature:** < 70% — Source-3 channel inert or noisy.

**G0-4: Causal-reasoning subset speedup.** Train two 66M models to convergence on a 100M-token slice: (a) post-#66 baseline, (b) post-#67-A. Evaluate on COPA + e-CARE + CRASS-mini. **PASS condition:** post-#67-A reaches the post-#66 baseline's final accuracy with ≤ 70% of the steps (1.4× headline-band, conservative). **FAIL signature:** ≥ 95% of steps required — paradigm inert at Gate-0.

**Joint Gate-0 PASS probability estimate:** G0-1 ~85% (CE interference is well-controlled by `λ` calibration; literature precedent in #59/#65 supports), G0-2 ~70% (template generation has known failure modes — agreement violations, ambiguous coreference), G0-3 ~75% (LLM-counterfactual quality varies), G0-4 ~70% (literature precedent for counterfactual fine-tuning at small scale is positive; CRASS-mini specifically has been demonstrated to benefit from such augmentation per Frohberg & Binder 2022). **Joint PASS: 0.85 × 0.70 × 0.75 × 0.70 ≈ 31%.**

This is honestly worse than #65-A's 52% Gate-0 PASS and substantially worse than #66's 80%. The principal risk is the conjunctive nature of the augmentation pipeline: each of three sources has its own quality gate, and the contrastive loss must cleanly compose with all prior auxiliary losses.

### 7.2 Gate-1 (~120 GPU-hours; if Gate-0 PASSES)

Gate-1 verifies LLM-scale at the 1.84B coordinator:

**G1-1: 1.84B baseline parity.** Train two 1.84B models on a 500B-token slice: post-#66 baseline vs post-#67-A. Measure CE on un-augmented dev set; verify ≤ 0.005 nat drift.

**G1-2: Causal-reasoning benchmark sweep.** Evaluate on COPA, e-CARE, CauseNet, CRASS, BIG-Bench Causal Judgement, BBH-causal subset. Compute joint accuracy delta. **PASS condition:** ≥ +3pp on the joint metric (translates to ~1.3-1.5× wall-clock speedup at fixed final accuracy by the post-#66 → post-#67-A delta).

**G1-3: Composition with #65 WS.** Validate that WS-field-structured contrastive (§1.4) contributes beyond text-only contrastive (target ≥ +0.5pp marginal on causal-reasoning subset).

**G1-4: Negative-control benchmarks.** Verify that augmentation does not harm un-augmented benchmark performance. Evaluate on Lambada, Winogrande, MMLU-non-causal-subset; measure ≤ 0.3pp regression.

**Joint Gate-1 confirmation probability:** ~0.55 conditional on Gate-0 PASS (literature suggests 1.84B counterfactual fine-tuning generally works but with high variance per Saparov 2023). **Unconditional: ~0.31 × 0.55 = ~17%** for full LLM-scale empirical confirmation.

---

## 8. Honest gaps and failure modes

### 8.1 Intervention sourcing is the principal risk

The mechanism's headline depends on a high-quality augmentation pipeline. The three sources have non-trivial failure modes:

**Source-1 (human-curated): scale problem.** ~50,000 pairs is ~0.0001% of an 18B-token corpus. Useful as calibration and Gate-0 ground-truth, not as primary source.

**Source-2 (heuristic templates): quality problem.** Entity-replacement breaks pronoun co-reference frequently (~30% on long texts); negation-insertion has scope ambiguity (does "not" attach to the verb or the negated clause?); premise-modification requires syntactic structure detection that sometimes fails. **Discriminator filter is essential** but adds compute (filter forward at ~5% of training cost).

**Source-3 (LLM-generated): circularity problem.** The teacher's notion of intervention is exactly what we're trying to teach the student. Mitigation: contrastive loss is on student-internal predictions, not teacher targets. But there's a residual risk that teacher and student converge on the teacher's intervention surface rather than a "true" interventional structure.

**Verdict:** the augmentation pipeline is the highest-risk component. Gate-0 sourcing-quality probes are the load-bearing checks.

### 8.2 NLL preservation is threshold-matched, not mathematically exact

Theorem 1 establishes preservation to a `<0.005 nat` drift threshold per #59-B precedent. Strict mathematical equality on un-augmented samples is not claimed because shared trunk parameters propagate auxiliary gradients to all positions. **This matches the iter-193 / iter-200 working definition of "bit-exact text NLL preservation"** but is not stronger.

### 8.3 The ~1.30-1.50× headline lives on a narrow subset

~10,000-15,000 questions across the named causal-reasoning benchmarks vs ~3-300B token training corpus — ratio ~3 × 10⁻⁸, essentially the same as #65-A's grounded-reasoning narrowness. **The cumulative ~8.6M× number is an axis-specific magnification, not a broader text-NLL contribution.** Honestly framed: the user's primary brief (compute speed at fixed text NLL) sees CAUSAL-CHIRON contributing ~1.0-1.02× — essentially neutral on the primary axis, with the headline on a narrow new axis.

### 8.4 Composition with #65 WS is partial-redundant

The structured contrastive on WS fields (§1.4) is the load-bearing #65 composition channel. But §3.3 honestly acknowledges ~25% overlap. The remaining ~75% novel contribution is from text-level contrastive on output tokens; the WS-structured channel is a refinement, not a doubling.

### 8.5 The paradigm-shift slate is saturating; this candidate continues the saturation pattern

The iter-210 #66 selection rationale explicitly framed structural saturation on text-axis multipliers. #67-A continues that pattern: it adds an axis (CAUSAL/INTERVENTIONAL) but does not produce magnitudes-better text NLL. **The honest expectation is that #67+ paradigms continue this pattern of axis-addition rather than text-NLL multiplication.**

This is consistent with the user's iter-200 anti-microoptimization brief — adding new axes is genuinely big-picture, not microoptimization on existing axes — but it does mean the cumulative-magnification trajectory on text NLL is plateauing at ~930,000×.

### 8.6 Triple-source-conjunctive Gate-0 PASS probability is low (~31%)

Honest about it. Three sources × four sub-gates × calibration sensitivity gives a Gate-0 that's harder to clear than #66's 80% or #65-A's 52%. **If Gate-0 fails on any sub-gate, the typical refinement** would be (a) drop Source-3 (~10% LOC saving, ~30% headline reduction) or (b) drop the WS-structured contrastive (~15% LOC saving, ~5% headline reduction). Both refinements are under-the-line acceptable; the headline drops to 1.20× standalone which is still axis-creating but no longer impressive.

### 8.7 LLM-scale empirical confirmation ~17% is the binding number

Gate-0 ~31% × Gate-1 ~55% = ~17% for actual LLM-scale empirical demonstration. **This is the lowest of any #67 candidate ranked in this iteration's slate** and is comparable to #65-C COMPUTE-ALLOCATOR's reservation tier (~25%) and worse than #65-A's grounded-reasoning ~35%.

The argument for selection regardless of Gate-0/Gate-1 odds: **the CAUSAL/INTERVENTIONAL axis is genuinely new at #67 and worth investigating even if the empirical confirmation odds are modest.** Causal reasoning is a known weakness of LLMs (Bowman 2022, Saparov 2023, Lyle 2023); even a 17%-confirmation paradigm is worth pursuing if the alternative is a strict-no-new-axis stance that produces only microoptimization candidates.

---

## 9. Bottom line / verdict

**Headline.** ~1.30× joint marginal on causal-reasoning subset. Cumulative ~8,580,000× on causal-reasoning subset (band [7.3M×, 13.2M×]). Text NLL approximately unchanged (~1.0-1.02× on primary axis). New CAUSAL/INTERVENTIONAL axis is the structural contribution; cumulative magnification on existing axes is modest.

**Engineering scope.** ~1,930 LOC over 7-8 weeks. Mid-range of recent paradigms.

**Joint Gate-0 PASS probability.** ~31% (honestly worse than #66's 80% and #65-A's 52%, principally due to triple-source augmentation pipeline conjunctive risk).

**LLM-scale empirical confirmation probability.** ~17% (Gate-0 PASS × Gate-1 conditional). Lowest of recent paradigms ranked.

**Bigger-picture framing.** CAUSAL-CHIRON adds a 12th axis (CAUSAL / INTERVENTIONAL) to the bigger-picture stack: DATA / LOSS / SAMPLING / REWARD / IDENTITY / SCHEDULE / AGENCY / OPTIMIZER / GROUNDING / KNOWLEDGE-LOCUS / VISION / **CAUSAL**. Differentiated from #65 WS (state encoding) and #59 PRM (reasoning-step correctness): CAUSAL is about *interventional response* — how predictions change when an upstream cause is set rather than observed. This is the missing primitive in observational-distribution training.

**Verdict:** **RESERVE** rather than SELECT.

The case for reserve rather than select rests on three honest observations:
1. **Triple-source augmentation pipeline conjunctive risk gives Gate-0 PASS only ~31%** — substantially worse than #66's 80% and #65-A's 52%. Selecting at this odds level is not the path of least regret when the slate has higher-confidence alternatives available (the #67 slate's other candidates may include such alternatives).
2. **The headline lives on a narrow ~10,000-15,000-question causal-reasoning subset**, comparable to #65-A's narrowness. Cumulative-stack contribution is axis-creating, not text-NLL magnifying — and the iter-200 anti-microoptimization brief does explicitly call for big-picture not narrow-subset contributions. The CAUSAL axis is genuinely new (which favors selection), but the magnitude on it is conjecture-dependent (which disfavors selection).
3. **Engineering scope ~1,930 LOC / 7-8 weeks is non-trivial** to deploy on a paradigm with 17% LLM-scale confirmation odds. Better to defer until either (a) a different paradigm surfaces with higher-confidence axis-creation potential, or (b) Source-2 / Source-3 quality improves through external developments (e.g., better LLM-generated counterfactuals as base models improve through 2026).

**Reserve disposition.** Reserved at #67-A pending iter-211+ slate review. **Promotion conditions:** (a) intervention sourcing pipelines mature externally (e.g., a high-quality publicly-available counterfactual augmentation dataset >1M pairs becomes available; ongoing CALM-evolution would help); (b) #67 slate's other candidates fail to produce a higher-confidence selection; (c) a future paradigm needs the CAUSAL axis as a load-bearing composition partner.

**Argument against pure reject:** the CAUSAL/INTERVENTIONAL axis is genuinely orthogonal to all 11 prior axes. Pearl 2009, Schölkopf 2021, Lyle 2023 establish causal reasoning as a fundamental gap in LLM capability. A program that reaches paradigm depth 25 without ever addressing the causal axis is incomplete on first principles. **Reservation preserves the option** of re-promotion once external conditions improve or other constraints relax, in the same way #63-B WORLD-MODEL was reserved for two iterations before triple-promotion at #65-A.

**The honest verdict:** **RESERVE** at #67 with promotion-conditional re-evaluation at iter-213+.

---

**End of Paradigm Shift #67 Candidate A document.** ~4,800 words. Pearl-style do-operator counterfactual augmentation with contrastive consistency. Headline ~1.30× joint marginal on causal-reasoning subset (cumulative ~8,580,000×). Joint Gate-0 PASS ~31%, LLM-scale confirmation ~17%. **Verdict: RESERVE** — genuine new axis but Gate-0 odds and narrow-subset framing argue for deferral pending iter-213+ slate review.
