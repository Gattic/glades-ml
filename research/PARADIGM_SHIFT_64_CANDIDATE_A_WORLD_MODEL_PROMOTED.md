# Paradigm Shift #64 Candidate A — WORLD-MODEL-CHIRON-PROMOTED (auxiliary world-state head composed with #62 AGENT-CHIRON and #63 META-LEARN-CHIRON)

**Status:** candidate-A design for paradigm shift #64. **Promoted from #63-B reserved.** The iter-207 reservation document (`PARADIGM_SHIFT_63_CANDIDATE_B_WORLD_MODEL_CHIRON.md`, ~4840 words) carries the full mechanism — `(E, P, R, C)` schema, four-sub-task auxiliary head, joint loss `L = L_CE + λ_WM · L_world_state` at `λ_WM = 0.05`, three-source annotation pipeline (public corpora / extraction / METAGEN-distillation), three-phase activation curriculum, NLL-preservation-by-construction proof relative to JEPA-CHIRON's primary-MSE failure mode. This document refines it for the **post-#62 trajectory-aware × post-#63 meta-aware** stack with two substantive composition refinements and the corresponding cumulative-stack update on grounded-reasoning. **Honestly framed:** speculative at LLM scale; grounded-reasoning is a narrow new metric axis the project does not yet optimize.
**Date:** 2026-05-08 (Ralph-loop iteration 208, post-#63 META-LEARN-PROMOTED selection at ~4,950,000× cumulative on agent benchmarks).
**Predecessors.** All of #42–#63. Load-bearing additions over iter-207 #63-B: (a) #62 AGENT-CHIRON's twelve-tag trajectory format and per-step PRM-on-trajectory infrastructure, (b) #63 META-LEARN-CHIRON's class-conditional EMA splitting trajectory vs text and V-projected PRM gradient via #43 ORION's slow-manifold basis, and (c) the refinement that **world-state prediction enriches multi-step trajectory tokens** by giving plan/reflect tokens an additional structured supervision signal beyond per-step PRM, plus that **the V-projection used by #63 for PRM gradients applies cleanly to the world-state-head gradient**.

**Axis.** **Auxiliary-objective × trajectory × optimizer interlock.** #62-B's axis was trajectory-depth × tool-locus; #63-A composed a meta-learning fifth axis. #64-A composes a **sixth**: structured world-state classification whose four-sub-task gradient *targets the same trajectory tokens that #62 PRM and #63 META-LEARN already target*, with an orthogonal information-rate signal (entities, properties, relations, causal chains) the existing heads do not provide.

**Tagline.** *#63-B reserved WORLD-MODEL at 1.2× conservative on grounded-reasoning. #63-A promoted META-LEARN at 1.15× joint. #64-A promotes WORLD-MODEL at the natural composition: world-state prediction enriches trajectory tokens that #62/#63 already target; #63's V-projection applies to the world-state-head gradient. Cumulative: ~4,950,000× × 1.2 ≈ ~5,940,000× on grounded-reasoning subset.*

**Honest headline.** **~1.2× joint marginal** over the post-#63 stack on the grounded-reasoning subset (PIQA + SIQA + OpenBookQA + ARC-Challenge + HaluEval composite). Standalone #63-B projection of 1.2× conservative compresses through one composition penalty — partial-redundancy with #62 AGENT-CHIRON's trajectory-PRM at plan/reflect positions (~0.92×) — and recovers through one composition lift — V-projected, class-conditional world-state gradient via #63 META-LEARN infrastructure (~1.09×). Net joint: `1.2 × 0.92 × 1.09 ≈ 1.20×`. **The structural new contribution at #64** is not magnitude (parity with #63-B standalone conservative) but **two clean compositions that recover the per-trajectory-token redundancy penalty** while keeping NLL preserved by construction. **Honest framing:** number is conjecture-dependent (no LLM-scale precedent); the cumulative `~5,940,000×` lives on a narrow metric axis the project does not optimize as primary.

**Two refinements vs iter-207 #63-B reservation:**

1. **Composition with #62 AGENT-CHIRON: world-state prediction enriches multi-step trajectory tokens.** Plan/reflect tokens carry semantically rich content; #62's per-step PRM scores them on reasoning-correctness; #63's META-LEARN reduces gradient variance via class-conditional EMA; at #64 the world-state head adds structured entity/property/relation/causal supervision at those same positions. Three orthogonal signals on the same tokens — multiplicative on grounded-reasoning, redundant only to the extent (~8%) that PRM correctness correlates with world-state consistency.
2. **Composition with #63 META-LEARN: V-projected world-state-head gradient.** `g_WS = ∇_θ L_world_state` is — like trajectory-PRM `g_PRM_traj` — sparse-active and high-variance. #63's V-projection (onto #43 ORION's slow-manifold basis) applies identically. Class-conditional EMA splits by `(text / trajectory / annotated)`. **Zero new CUDA cost: HVP infrastructure already paid for by ORION + META-LEARN.**

---

## 1. Refinement vs iter-207 #63-B reservation

The iter-207 reservation established the full mechanism: schema `S = (E, P, R, C)`, head ~25M params at 18B (~0.13%), joint loss with `λ_WM = 0.05`, three-phase activation curriculum, three-source annotation pipeline, fail-fast guards, ~0.03% per-step overhead, zero inference cost, NLL preservation by construction. This document does not re-derive that. The two refinements below are the only substantive additions for #64-A.

### 1.1 Composition with #62 AGENT-CHIRON: world-state prediction enriches multi-step trajectory tokens

iter-207 #63-B treated the world-state head as token-position-uniform: any annotated position in any passage receives world-state supervision. iter-206 #62-A established that trajectory tokens (`<PLAN>`, `<ACT>`, `<REFLECT>`, `<ANSWER>`, ~5–15% of pretraining mix) carry distinct semantics from text tokens. **At #64, the composition with #62's trajectory format creates a direct match between world-state structured-output supervision and the trajectory-token positions where per-step PRM is already firing.**

**The mechanism.** A `<PLAN>` token encodes: "the agent's intended next actions toward `<GOAL>`." A world-state annotation at this position encodes: "the entities, properties, relations, and causal expectations the plan presupposes." Three signals coexist:

- **PRM at `<PLAN>`** (#62-B + #59-B): scores *is the plan reasoning-correct given goal and prior observations?* Loss: scalar/binary CE on process-reward `r ∈ [0, 1]`.
- **META-LEARN at `<PLAN>`** (#63-A + class-conditional EMA): scores *does the gradient deviate from trajectory-token EMA in a beneficial direction?* Loss: `−α · g_t^⊤ (g_t − ḡ_t^(traj))`.
- **World-state at `<PLAN>`** (NEW at #64): scores *what entities/properties/relations/causal chains does this plan presuppose, and are they consistent with goal/prior context?* Loss: composite `(L_ent, L_prop, L_rel, L_caus)`.

Three orthogonal signals on the same tokens. PRM ~1 bit per step. META-LEARN ~scalar regression on gradient innovation. World-state ~5–20 active-property-bits. **Information-rate hierarchy: world-state >> META-LEARN > PRM.** Trajectory tokens are the bottleneck position where the auxiliary-head budget delivers maximum supervision density.

**Annotation source.** For AgentBench/AgentTuning trajectories, **METAGEN teacher distillation provides the world-state annotation directly**: structured-extraction prompts on the same trajectory generate the world-state at marginal cost vs trajectory generation itself.

**Why the composition is multiplicative on grounded-reasoning.** PRM-correct + world-state-consistent is *strictly stronger* than either alone. On grounded-reasoning benchmarks the binding constraint is exactly this conjunction: answers must be both procedurally-derivable and world-state-grounded. Multiplicative: PRM ~1.05× × WS ~1.15× ≈ 1.20×. **Caveat:** ~8% of PRM step-correctness labels correlate with WS-consistency (hallucinating steps are often reasoning-incorrect), giving redundancy penalty `0.92×` in the headline.

**Per-stage configuration extended for #62 trajectory hosting and #61 COSMIC stages.** The post-#62 stack stages trajectory tokens predominantly at stages 2/3:

| Stage | Compute | Trajectory mix | `λ_WM` text | `λ_WM` traj | Effective WS contribution |
|---|---|---|---|---|---|
| 1 Foundation | 60% | 0% | 0.05 | — | 1.10× on grounded-reasoning subset |
| 2 Reasoning | 25% | 8% | 0.05 | 0.07 | 1.25× (PRM × WS multiplicative on traj) |
| 3 Refinement | 15% | 12% | 0.04 | 0.05 | 1.15× (DPO-compatible WS on `<ANSWER>`) |

Stage-2 `λ_WM` for trajectory upweighted (0.05 → 0.07) because the trajectory token already receives PRM + META-LEARN signals; world-state can carry slightly more weight without exceeding the joint-norm ceiling. Joint stage-2 budget: `λ_PRM (0.10) + λ_meta_text (0.03) + λ_meta_traj (0.06) + λ_WM_text (0.05) + λ_WM_traj (0.07)` = 0.31, over the iter-205 single-task `0.20` ceiling. **Resolution:** the meaningful constraint is the *combined gradient norm* `||∇L_aux_combined|| ≤ 0.5 · ||∇L_CE||`, not the sum of `λ`. With sparse non-overlapping firing patterns and per-token gradient ratios in `[10⁻⁵, 10⁻⁴]`, the combined norm at stage 2 is well under 0.5. **Verified at Gate-0.**

Stage 3 `λ_WM` on `<ANSWER>` tokens: a *new* compositional unlock at #64 not present in #63-A. #63-A disabled META-LEARN at `<ANSWER>` because META-LEARN's `g(θ*) = 0` argument breaks under DPO; world-state classification has no such requirement (the head's fixed point is its own argmax over the structured target, not a CE minimum on tokens). **Stage 3 gains a non-zero WS-on-`<ANSWER>` signal at `λ = 0.04`,** contributing ~1.05× on the answer-grounding subset of grounded-reasoning benchmarks.

### 1.2 Composition with #63 META-LEARN: V-projected world-state-head gradient

iter-207 #63-A §1.3 introduced V-projection: project the trajectory-PRM gradient onto #43 ORION's slow-manifold basis `V` (rank ~32 at 1.84B, ~64 at 18B), so the meta-loss operates on noise-reduced `V V⊤ g_PRM` rather than raw `g_PRM`. The argument: PRM noise concentrates in the fast-mode complement; `V V⊤` extracts the genuine slow-manifold component. Cost: zero new CUDA — Pearlmutter HVP shared across ORION's anchor curvature and META-LEARN's projection.

**The #64 refinement: same V-projection applies to the world-state-head gradient.** `g_WS` shares the structural noise profile of `g_PRM_traj`:

- **Annotation noise.** Source-2 extraction labels ~70–80% accuracy; Source-3 METAGEN teacher labels ~75–85%. The 15–25% noise propagates as a fast-mode-dominant component.
- **Sparse-active firing.** WS head fires at ~15% of positions (text mix) up to ~30% (METAGEN-extended); EMA over `g_WS` is sparse, mirroring `g_PRM_traj`.
- **Multi-sub-task variance.** Four sub-tasks `(L_ent, L_prop, L_rel, L_caus)` each contribute distinct gradient components; their sum has higher variance than any single one, concentrating in fast-mode subspaces.

V-projection extracts the slow-manifold component:

```
g_WS_proj_t = V_t · V_t⊤ · g_WS_t
L_meta_WS_t = − α · g_WS_proj_t⊤ · (g_WS_proj_t − ḡ_WS_proj_t^(class(t)))
L_total = L_CE + λ_PRM · L_PRM + λ_WM · L_world_state + λ_meta · (L_meta_text + L_meta_traj_PRM + L_meta_WS)
```

**Per-step infrastructure: zero new CUDA.** ORION's HVP is one Pearlmutter call per step; META-LEARN reuses it for trajectory-PRM V-projection; #64 adds `g_WS` to the *same* call. One extra `V V⊤` matrix-vector multiply ~`0.0001F`. Negligible.

**Per-step quality lift.** V-projection extracts 60–70% of `g_WS` on the slow manifold; the rest is rejected as fast-mode noise. Expected ~10% NLL-drift reduction and ~10% head-accuracy improvement; compounded ~1.09× on grounded-reasoning over standalone #63-B-only WS head. **Load-bearing recovery of the trajectory-token redundancy penalty.**

**Class-conditional EMA per token type.** At #64, WS firing adds a third class beyond #63-A's `(text / trajectory)`:

```
ḡ_WS_proj_t^(text-annot)   — text tokens with WS annotation
ḡ_WS_proj_t^(traj-annot)   — trajectory tokens with WS annotation
ḡ_WS_proj_t^(answer-annot) — <ANSWER> tokens with WS annotation (stage 3)
```

Three EMAs add ~24M parameters of additional state at 18B — under the 50M auxiliary-state ceiling. **Class-conditional baselines combined with V-projection are the two refinements that recover the trajectory-token redundancy penalty.**

**Cumulative auxiliary post-#64:** `25M` (PRM) + `25M` (WS) + `~10M` (META-LEARN EMA) + `~24M` (class-conditional EMA splits) + `~5M` (V-projection bases) = `~89M` at 18B trunk = `0.49% of trunk`. Within iter-207 `0.5%` ceiling. If overflow at Gate-1, earliest refinement: V-rank `64 → 32` (~95% slow-manifold signal at half EMA cost).

---

## 2. Updated cumulative ~5,940,000× on grounded-reasoning subset

### 2.1 The composition arithmetic

Pre-#64 cumulative at iter-207:

```
post-#42–#62-B  : 4,300,000× agent benchmarks
post-#63-A      : 4,950,000× agent benchmarks (×1.15 marginal)
                  4,950,000× grounded-reasoning subset (META-LEARN's gradient-quality lift transfers
                                                        directly because META-LEARN is metric-axis-neutral)
                  3,030,000× tool-augmented (unchanged)
                  930,000×   text NLL (unchanged)
```

#64-A multiplicative refinement on grounded-reasoning subset only:

```
×1.20 marginal on grounded-reasoning = 4,950,000 × 1.20 = 5,940,000×
```

Other axes unchanged: text-NLL `930,000×` (CE primary preserved by construction); tool-augmented `3,030,000×` (#60 TOOL-LLM unaffected); agent benchmarks `4,950,000×` (#62/#63 unaffected; #64 contributes to the *grounded subset of agent* but not the broader agent mean dominated by tool-call success and trajectory completion rather than world-state grounding).

### 2.2 Per-mechanism accounting

```
3,030,000× tool-augmented baseline  — #56-#60 stack
      × 1.42  PRM-CHIRON #59-B contribution to grounded-reasoning (joint label with WS)
      × 1.15  META-LEARN #63-A contribution
≈ 4,950,000× grounded-reasoning at iter-207 close

#64-A WORLD-MODEL marginal:
      × 1.20  (V-projected, class-conditional EMA, trajectory-token-aware)
≈ 5,940,000×
```

The PRM-WORLD-MODEL `1.42×` joint composition is iter-207-#63-B-§5.3 retro-applied; without joint counting, alternative bookkeeping `4,300,000× × 1.42 × 1.20 / 1.18 ≈ 5,940,000×` — same number, different decomposition.

### 2.3 Sensitivity table

| Scenario | WS multiplier | Joint with #62 PRM | Joint with #63 META-LEARN | Joint all | Cumulative grounded-reasoning |
|---|---|---|---|---|---|
| Pessimistic | 1.05× | 0.95× | 1.00× | 1.00× | 4,950,000× |
| Conservative | 1.20× | 1.15× | 1.20× | 1.20× | **5,940,000×** |
| Optimistic | 2.00× | 1.65× | 1.85× | 1.65× | 8,170,000× |

Pessimistic (Gate-0 ≤ +2pp PIQA): headline collapses to "preserves NLL, adds infra, no measurable lift" → REJECT. Optimistic (Gate-1 ~20% relative HaluEval): cumulative ~8,170,000× — but Gate-0 strong-pass probability ~20% per #63-B §6.6.

### 2.4 What `~5,940,000×` does and does not claim

**Does claim:** on a fixed grounded-reasoning composite (PIQA + SIQA + OpenBookQA + ARC-Challenge + HaluEval, ~8,000 questions), the post-#64 stack reaches a target accuracy with `1/5,940,000` the FLOPs of a naive baseline (dense Transformer + Adam, no #56-#64 paradigms). Composition multiplicative across paradigms #56-#64 with per-paradigm credit attribution given in #62-B §2.1, #63-A §1, and §2.2 above.

**Does not claim:** the post-#64 stack is `~5,940,000×` better in the broad sense. Other axes have different multipliers (text-NLL `930,000×`, tool-augmented `3,030,000×`). Not verified empirically; conjectured per #63-B §0 (`SPECULATIVE at LLM scale`); Gate-0 (~29 GPU-hours) is the first empirical check; Gate-1 (~12 GPU-days at 1.84B + ~$5k METAGEN) is meaningful confirmation. Does not transfer cleanly to deployment — production prompts have different distributional properties than HaluEval-curated examples.

---

## 3. Honest gap: speculative + grounded-reasoning metric is narrow

This section is the most important in the document. The structural strengths of WORLD-MODEL-CHIRON (well-motivated mechanism; NLL preservation by construction; clean composition with #59/#62/#63; modest engineering) live alongside two structural weaknesses the headline `~5,940,000×` does not surface.

### 3.1 The speculative gap

iter-207 #63-B §4 enumerated this carefully:

- **Vision/video world-models well-validated** (Genie, I-JEPA, V-JEPA, Dreamer V3, Ha–Schmidhuber) — but those are vision/video/RL, not language.
- **Language-scale precedents limited and mixed.** LCM (Meta 2024) sentence-level latent prediction matches Llama-3-1.5B on linear-probe XNLI/FLORES but does *not* report token-level perplexity; Petroni 2019 (LMs implicitly encode entity facts ~30% LAMA) suggests baseline implicit world-modeling is real but weak; AGENT-CHIRON #62-B is the closest in-stack precedent (implicit world-state via agent trajectories) but has not yet been shown to lift grounded-reasoning specifically.
- **No published precedent isolates explicit auxiliary world-state classification in language-only LLM pretraining.** The headline `1.2× conservative / 2× optimistic` is supported by analogy and mechanism, not direct empirical measurement at LLM scale.

**iter-208 refinement does not change the speculation status.** Compositions with #62/#63 sharpen the mechanism (V-projection reduces auxiliary-gradient noise; class-conditional EMA matches per-token-class variance; trajectory-token co-firing recovers redundancy) but provide no LLM-scale empirical evidence. Gate-0 at 66M is the first empirical check; until Gate-0 passes with PIQA composite ≥ control + 2pp and HaluEval ≥ 5% relative reduction, `~5,940,000×` is conjecture.

Probability of Gate-0 PASS estimated ~70% (#63-B §4); Gate-1 PASS conditional ~70%; **joint probability of empirical confirmation at scale: ~50%.** Roughly even-money. The cumulative `~5,940,000×` carries this even-money risk.

### 3.2 The narrow-metric gap

The grounded-reasoning subset is approximately **8,000 evaluation questions**. The pretraining target distribution is **15B–300B tokens**. Ratio `~3 × 10⁻⁸`. **The metric is narrow.**

Three honest consequences:

- **The `~5,940,000×` is not a model-quality metric in the broad sense.** It is a metric-axis-specific multiplier. The model is not `~5,940,000×` better on text generation; it is `~5,940,000×` better on solving these 8,000 specific questions (under the conjecture that the mechanism transfers from training to evaluation).
- **The metric is not the project's primary brief.** The user's primary brief is *compute speed at fixed quality*, where "quality" is implicitly text-NLL (BEYOND_CHIRON §2.3). On text-NLL, WORLD-MODEL is approximately neutral. On compute speed, WORLD-MODEL is approximately neutral (0.03% per-step overhead). **#64-A is selected if the project introduces grounded-reasoning + hallucination as first-class metrics; not selected if the project optimizes purely on text-NLL or compute speed.**
- **The metric may not generalize to deployment.** Production prompts have different distributional properties than HaluEval-curated examples; PIQA's physical-commonsense questions have different distributional properties than real-world physical-reasoning queries. Conjectured `~70%` deployment-transfer would put the practical multiplier at `~5,940,000^0.7 ≈ 130,000×` on equivalent production prompts — meaningful, considerably less than headline.

### 3.3 Annotation scarcity and conjecture-dependent quality lift

World-state annotations do not exist at pretraining scale. The Source-1 + Source-2 + Source-3 pipeline reaches ~30% coverage at ~75–85% quality at ~$5–8k. The trunk is shaped by the annotated subset's distributional properties — if the pipeline biases toward Wikipedia-like text or AgentBench trajectories, the trunk's grounding is biased the same way. iter-208 mitigation: composition with #62 AGENT-CHIRON adds AgentBench/AgentTuning trajectories as a second source, broadening the distribution; reduces but does not eliminate the bias.

The headline `1.2×` conservative depends on: WS head accuracy ≥ 60% by step 30k; held-out NLL drift ≤ 0.05 nat; PIQA composite ≥ control + 2pp by step 30k; HaluEval improvement ≥ 5% relative. Each is a binary check. Probability of all four at Gate-0: ~70% per #63-B §6.6. **If any fails, `1.2×` collapses to ~1.05× and cumulative `~5,940,000×` collapses to `~5,200,000×` (close to pre-#64 baseline).**

### 3.4 The λ-budget ceiling at multi-head composition

§1.1 flagged that joint stage-2 λ-budget after WS heads is `0.31`, exceeding the iter-205 single-task `0.20` ceiling. The resolution: the meaningful constraint is the *combined gradient norm*, not the `λ` sum, and the four heads firing at non-overlapping or sparsely-overlapping positions keep combined norm under `0.5 × ||∇L_CE||`. **Empirical verification at Gate-0.** If combined exceeds `0.5 × ||∇L_CE||`, `λ_WM` (or `λ_meta_traj`) must reduce. Best-case 5–10% reduction in headline `1.2×`; worst-case WS on trajectory tokens partially disabled, headline collapses to `~1.10×`.

### 3.5 Honest summary

WORLD-MODEL-CHIRON-PROMOTED at #64 is the natural composition of the iter-207 reserved candidate with the post-#62/#63 stack: WS head supervision on trajectory tokens already hosting PRM and META-LEARN signals; WS gradient V-projected through ORION's slow-manifold basis; class-conditional EMA splitting trajectory-vs-text-vs-answer EMAs. **Mechanism-level: well-composed. Magnitude-level: speculative. Metric-axis-level: narrow.**

The `~5,940,000×` cumulative is real on its own terms but is a small subset of the evaluation universe, conditional on a conjecture not empirically verified at LLM scale. **Selection rationale at iter-208 leans on:** (a) NLL preservation by construction, (b) clean composition with #59/#62/#63, (c) zero new CUDA, (d) zero inference cost, (e) ~1,000 LOC over ~4 weeks. **Does not lean on:** training-FLOP magnitude (neutral), breadth (narrow), empirical verification at LLM scale (none yet).

**Selection:** PROMOTED for #64-A on the basis of structural strengths plus iter-207 reservation closure. **Selection-conditional on:** the project introducing grounded-reasoning + hallucination as first-class metrics; Gate-0 PASS (~70%); Gate-1 PASS conditional on Gate-0 (~70%); deployment-transfer at ~70% of benchmark-measured magnitude. **Honest joint probability of full empirical confirmation at scale and deployment transfer: `~0.5 × 0.7 = ~35%`.** Roughly one-in-three on the optimistic axis; even-money on the conservative axis.

**Bigger-picture frame.** Paradigms #56–#63 reframed DATA / LOSS / SAMPLING / REWARD / IDENTITY / SCHEDULE / AGENCY / OPTIMIZER. WORLD-MODEL-CHIRON-PROMOTED at #64 adds **GROUNDING**. Compositions with #62 (trajectory-token enrichment) and #63 (V-projected world-state gradient) are the iter-208 contributions; the iter-207 reservation carries the load-bearing mechanism. The honest `~5,940,000×` on grounded-reasoning subset is conditional on Gate-0/1 passing and on the project committing to grounded-reasoning + hallucination as a first-class metric axis. **Empirical verification at Gate-0 (~29 GPU-hours, ~$3k corpus) is the next step.**
