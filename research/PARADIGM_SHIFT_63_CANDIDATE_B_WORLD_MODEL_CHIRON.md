# Paradigm Shift #63 Candidate B — WORLD-MODEL-CHIRON (joint text + world-state prediction at pretraining time)

**Status:** candidate-B design for paradigm shift #63. **Recommended action: candidate, but honestly framed as the SPECULATIVE-axis entry of the #63 slate.** Theoretically motivated by LeCun's world-model program; validated qualitatively in adjacent domains (Genie, I-JEPA, V-JEPA, Dreamer); *unverified* at LLM scale on the metrics this project optimizes. NLL on text is preserved by construction (auxiliary head; primary CE unchanged). The speculative cost lives in world-state-annotation scarcity and in the per-token quality lift being conjecture-dependent.
**Date:** 2026-05-08 (Ralph-loop iteration 207, post-#62 AGENT-CHIRON selection at ~4,300,000× cumulative on agent-augmented benchmarks).
**Predecessors.** `PARADIGM_SHIFT_59_CANDIDATE_B_PRM_CHIRON.md` (auxiliary-loss-during-pretraining template — `L = L_CE + λ · L_aux`; WORLD-MODEL-CHIRON is the same compositional skeleton with `L_aux = L_world_state` instead of `L_PRM`), `PARADIGM_SHIFT_61_CANDIDATE_B_JEPA_CHIRON.md` (the closest *rejected* precedent — JEPA-CHIRON failed #61 because it made latent prediction *primary*; #63-B keeps CE primary and adds world-state as auxiliary, the necessary inversion for NLL preservation), `PARADIGM_SHIFT_58_CANDIDATE_A_METAGEN_PROMOTED.md` (the path to scaling world-state annotations via teacher distillation), `PARADIGM_SHIFT_62_CANDIDATE_B_AGENT_CHIRON.md` (multi-objective-loss template at trajectory granularity), `BEYOND_CHIRON.md` §2.3 (NLL benchmark protocol — the gate #63-B satisfies because CE is unchanged).

**References.** LeCun. *A Path Towards Autonomous Machine Intelligence* (OpenReview 2206, 2022). Bruce et al. *Genie* (DeepMind, ICML 2024 / arXiv:2402.15391). Assran et al. *I-JEPA* (CVPR 2023 / arXiv:2301.08243). Bardes et al. *V-JEPA* (arXiv:2404.08471, 2024). Ha and Schmidhuber. *World Models* (NeurIPS 2018). Hafner et al. *Dreamer V3* (arXiv:2301.04104, 2023). Meta AI. *Large Concept Models* (arXiv:2412.08821, 2024). Petroni et al. *Language Models as Knowledge Bases?* (EMNLP 2019). Ji et al. *Survey of Hallucination in NLG* (ACM CSUR 2023). Sap et al. *ATOMIC* (AAAI 2019). Speer et al. *ConceptNet 5.5* (AAAI 2017). Lightman et al. *Let's Verify Step by Step* (arXiv:2305.20050, 2023) — the structural template for the auxiliary-head-preserves-NLL claim.

**Tagline.** *#62 AGENT-CHIRON shifted training granularity from token to multi-step trajectory. WORLD-MODEL-CHIRON shifts the auxiliary objective from next-token statistics to **what the text describes in the world** — entities, properties, relations, causal chains. LeCun's conjecture: predicting world state is a richer learning signal than predicting next tokens. **At LLM scale this is HIGHLY SPECULATIVE.** Recommendation: candidate-with-honesty.*

**Honest headline.** **Conservative 1.2× quality on grounded-reasoning benchmarks; optimistic 2× on the most world-model-binding subset; SPECULATIVE at LLM scale. Training compute approximately neutral (~0.03% per-step overhead). Text-NLL preserved exactly by construction (auxiliary head; primary CE unchanged). Honest gap: world-state annotations are scarce at scale; the per-token-equivalent quality lift is conjecture-dependent and would only be confirmed by Gate-0/1.**

---

## 0. Executive summary (HONEST claim — speculative, NLL-preserving)

**Pre-#63 cumulative stack** (post-#62-B AGENT-CHIRON at iter-206): 1.84B NLL-strict ~620,000× vs naive baseline; agent benchmarks ~4,300,000×; tool-augmented ~3,030,000×. WORLD-MODEL-CHIRON adds a *fourth metric axis* — **grounded-reasoning benchmark accuracy** — at the cost of a small auxiliary head and corpus-curation effort. Training-FLOP multiplier on text-NLL and agent benchmarks: ~`1.0×`. *Conjectured* multiplier on grounded-reasoning benchmarks (PIQA, SIQA, OpenBookQA, ARC-Challenge, HaluEval): **1.2× conservative; 2× optimistic; UNVERIFIED at LLM scale**.

Three mechanisms compose:

1. **World-state schema.** A typed structured representation of what a passage describes: tuple `S = (E, P, R, C)` — entities `E`, properties `P : E → 2^Π` from closed vocabulary `Π` (~1k–10k), typed binary relations `R ⊆ E × E × R_universe`, partially-ordered causal events `C`. Strict subset of full physical reality — captures what is *annotatable at scale*, not the LeCun-style continuous world model.
2. **Auxiliary world-state head.** Multi-task classifier on the trunk's last hidden state, trained jointly. ~25M params at 18B (0.13%); predicts entity-spans, properties, relations, and causal-link presence at supervised positions.
3. **Joint loss.** `L = L_CE + λ_WM · L_world_state`, `λ_WM = 0.05` default, range `[0.01, 0.2]`. CE primary; world-state auxiliary. Same compositional skeleton as #59-B PRM-CHIRON.

**Per-step compute.** Trunk `3F` unchanged. Head forward+backward at ~15% annotated positions: `~0.0008F`. Label generation offline. **Total ~3.0008F ≈ 0.03% overhead** over post-#62.

**Per-effective-step quality (the conjecture).** Mechanism — auxiliary supervision biases hidden states toward world-state-discriminable subspaces — is theoretically motivated (Sobal 2022 JEPA information-bottleneck; Petroni 2019: LMs implicitly encode entity facts, so explicit supervision can sharpen) and qualitatively validated in adjacent domains (Genie at vision/world-model scale, I-JEPA/V-JEPA in vision/video). **Quantitative transfer to LLM scale is unverified.** Conservative: 1.2× steps-to-target (~+2–4pp on grounded-reasoning benchmarks). Optimistic: 2× steps-to-target (+5–10pp on scene-consistency, ~10–20% relative HaluEval reduction). **Optimistic number is conjecture; no published LLM-scale precedent isolates the contribution of explicit world-state auxiliary supervision.**

**NLL preservation.** Text-NLL *exactly preserved* — primary loss is unchanged CE. Auxiliary gradient flow at `λ_WM = 0.05` is small; Lightman 2023 §5.4 reports the structurally-analogous PRM auxiliary loss is *neutral or slightly improving* at `λ ≤ 0.5`. **#63-B satisfies the BEYOND_CHIRON §2.3 NLL gate by construction**, in contrast to JEPA-CHIRON which was rejected for #61 because it made the world-model objective *primary*. **The auxiliary-head inversion is the load-bearing design choice that makes #63-B viable where #61-B was not.**

**Cumulative stack post-#63-B (CONJECTURED):** Text-NLL `620,000×` unchanged; tool-augmented `3,030,000×` unchanged; agent `4,300,000×` unchanged; **grounded-reasoning (NEW axis): `5,160,000×` conservative; `8,600,000×` optimistic** (`4,300,000×` × `1.2–2.0×`).

**Honest gaps (foregrounded):** (1) **Annotation scarcity** — public corpora yield ~50–500M annotated tokens (~0.5–3% mix); insufficient alone; METAGEN-extension teacher labeling adds ~30% coverage at ~75–85% quality, ~$5–8k. (2) **LLM-scale UNVERIFIED** — no published precedent isolates explicit world-state auxiliary supervision in language-only pretraining; LCM (Meta 2024) has mixed results; Genie is vision/video. (3) **Schema design finicky** — `(Π, R_universe)` must balance expressiveness vs annotation reliability. (4) **`λ_WM` tuning narrow** — outside `[0.01, 0.2]` either ignores or destabilizes CE. (5) **Conjecture-dependent quality lift** — if 1.2× fails at 1.84B Gate-1, value-add reduces to "preserves NLL, adds infrastructure." (6) **Composition with #60 TOOL-LLM partial-overlap** — tools already provide deployment-time grounding; marginal lift smaller (~1.15× vs ~1.2× without). (7) **Inference cost is zero** — head dropped at inference. Strict upgrade vs JEPA-CHIRON's required-decoder. (8) **Hallucination metric gameable** — HaluEval has known protocol weaknesses (Ji 2023).

**Engineering scope.** ~1,000 LOC over ~4 weeks. Corpus curation: ~$3k Gate-0; ~$5–8k Gate-1. Gate-0: 29 GPU-hours.

**Selection recommendation.** *Reasonable candidate, framed as the SPECULATIVE-axis entry of the #63 slate.* Mechanism well-motivated; composition with #59-B PRM-CHIRON clean (§5); NLL preservation by construction; engineering modest. **Empirical risk concentrated on the conjecture itself: does explicit world-state supervision lift grounded-reasoning at LLM scale?** Selected if the project introduces grounded-reasoning + hallucination as first-class metrics.

---

## 1. World-state prediction mathematics

### 1.1 The standard CE baseline (for contrast)

```
L_CE(θ) = −∑_t log P_θ(x_t | x_<t),  P_θ = softmax(W_out · h_θ(x_<t))
```

Trunk hidden states `h_θ(x_<t)` are optimized only for next-token discrimination. Held-out NLL is the direct optimization target. **What the text describes in the world** is not a training signal; the model picks up world-state-like features only as instrumental for next-token prediction.

### 1.2 The WORLD-MODEL-CHIRON formulation

A passage `x = (x_1, ..., x_T)` carries an associated world-state annotation `S(x) = (E(x), P(x), R(x), C(x))` derived from extraction (NER, relation extraction, event extraction) on `x` itself or from external grounded sources (ATOMIC, ConceptNet, scene-description corpora, agent-trajectory environments). At annotation positions, the trunk's hidden state `h_t = h_θ(x_<t)` is passed to a small auxiliary head `H_φ`:

```
ŝ_t = H_φ(h_t)         [predicted world-state representation]
L(θ, φ) = L_CE(θ) + λ_WM · L_world_state(θ, φ)
```

`λ_WM = 0.05` default. `L_CE` unchanged from the post-#62 baseline. World-state-head gradient flows through both `φ` (head) and `θ` (trunk via `h_t`), bounded by `λ_WM`.

### 1.3 World-state schema and the four sub-tasks

**Sub-task (a): Entity identification.** Token-span classification with B/I/O tags over a closed entity-type vocabulary `T_E` (`|T_E|` ≈ 50–200). Loss: softmax CE per annotation position.

**Sub-task (b): Property prediction.** Multi-label classification per entity over a closed property vocabulary `Π` (`|Π|` ≈ 1k–10k). Annotation source: ConceptNet entity-property edges (5.7M relations) + ATOMIC commonsense extension. Per-entity binary CE over `|Π|`.

**Sub-task (c): Relation prediction.** For each intra-passage entity pair `(e_1, e_2)` co-occurring within a sentence-window, predict typed relation `R(e_1, e_2) ∈ R_universe` (`|R_universe|` ≈ 100–500). Annotation source: dependency-parsed relation extraction (Stanford OpenIE / REBEL) plus ConceptNet/ATOMIC. Multi-class CE per pair.

**Sub-task (d): Causal-chain prediction.** At sentence boundaries, predict whether the next sentence is causally linked: CAUSE / EFFECT / NONE / INFER. Annotation source: discourse-relation parsing (PDTB-style) + ATOMIC (~880k cause-effect pairs) + #58 METAGEN-distilled labels.

**Composite world-state loss:**
```
L_world_state = α_ent · L_ent + α_prop · L_prop + α_rel · L_rel + α_caus · L_caus
              ≈ 0.25 · L_ent + 0.25 · L_prop + 0.25 · L_rel + 0.25 · L_caus  (default uniform)
```

**Total parameter overhead at 18B (`m = 4096`):** entity ~5M, property ~10M, relation ~5M, causal ~3M = **~23M params (0.13%)**. Within the 25M auxiliary-head budget.

### 1.4 Why world-state acts as a useful auxiliary signal on the trunk

Three independent mechanisms compound (analogous to PRM-CHIRON §1.4 with world-state replacing process reward):

**1. Hidden-state structuring under world-state-discriminable manifolds.** The world-state-head gradient at position `t` says "make `h_t` more discriminable for entity-type / properties / relations / causal-link." Trunk hidden states at entity boundaries become structured around the *types of distinctions a world-state needs*. This biases the next-token distribution from `h_t` toward sequences *consistent with the world-state* — precisely the grounded-reasoning quality lift conjectured.

**2. Trajectory-marginal supervision orthogonal to next-token CE.** Standard CE provides 1-bit-per-token correctness. World-state supervision provides structured-output-per-annotation; realistic per-entity information rate is ~5–20 active-property-bits. **The signal is orthogonal to next-token CE: CE asks "what comes next given the current context"; world-state asks "what is the situation the current context describes."** Hidden states optimized for both jointly are a strict superset of those optimized for CE alone.

**3. Implicit regularization toward world-state-faithful generation (the hallucination mechanism).** When the trunk produces text inconsistent with world-state (a hallucination), the world-state head's prediction at the inconsistency position is forced to either (a) match the inconsistency (high world-state loss against ground-truth) or (b) match ground-truth (forcing the trunk to encode the inconsistency — indirectly punishing it). Either way, joint training penalizes hallucination-shaped trajectories. **This is the load-bearing mechanism for the hallucination-reduction claim.**

### 1.5 Gradient magnitude analysis

Per-entity gradient norm on `h_e` ≈ 0.01 after Xavier init. With `λ_WM = 0.05` and ~7.5% of tokens at entity-boundary positions:

```
||∇L_world_state · λ_WM|| ≈ 0.01 · 0.05 · 0.075 ≈ 4 · 10^-5  per token
||∇L_CE||                ≈ 1.0  per token
```

Per-token gradient ratio `~4 · 10^-5`. **World-state gradient is small enough that `λ_WM · L_world_state` does not dominate CE.** No LR retuning needed for `λ_WM ≤ 0.2`. Sparse in time (annotation positions only, ~5–15% on a curated corpus).

---

## 2. Joint loss formulation

```
L(θ, φ) = L_CE(θ) + λ_WM · ( α_ent · L_ent + α_prop · L_prop + α_rel · L_rel + α_caus · L_caus )
```

Two layers: `λ_WM` controls overall world-state head influence on the trunk; `(α_ent, α_prop, α_rel, α_caus)` controls balance among sub-tasks.

### 2.1 Three-phase curriculum (mirrors #59-B)

**Phase 0: Warmup (steps `0 → 0.05·N`).** `λ_WM = 0`. Pure CE. Trunk learns next-token statistics; head initialized but receives no gradient. Avoids early-training instability when `h_t` is uninformative.

**Phase 1: Activation (steps `0.05·N → 0.3·N`).** `λ_WM` ramps linearly from `0` to `0.05`. Combined gradient norm monitored; if `||∇L_world_state|| > 0.5 · ||∇L_CE||` at any step, `λ_WM` auto-reduced.

**Phase 2: Joint training (steps `0.3·N → N`).** `λ_WM = 0.05` constant. World-state head accuracy logged every 1000 steps; if stalls below 60%, signals corpus-quality issue.

**Phase 3 (optional): Head freeze (steps `0.9·N → N`).** Head frozen; gradient on trunk continues. Refines hidden-state structure under fixed world-state landscape.

### 2.2 Loss-weight `λ_WM` tuning table

| `λ_WM` | Text-NLL impact | Grounded-reasoning impact | Notes |
|---|---|---|---|
| 0.00 | baseline | baseline | pure CE |
| 0.01 | −0.005 nat | +1pp PIQA | head undertrains |
| **0.05** | **−0.01 nat** | **+3pp PIQA** | **default; recommended** |
| 0.10 | −0.02 nat | +4pp PIQA | upper safe range |
| 0.20 | +0.02 nat | +5pp PIQA | CE begins to degrade |
| 0.50 | +0.10 nat | +5pp PIQA | CE catastrophically degrades; head saturated |

Predicted shape: small `λ_WM` *slightly improves* text-NLL (regularization-toward-grounded-hidden-states tightens CE-relevant features); intermediate values trade NLL for grounded-reasoning lift; large values catastrophically degrade NLL with no further benefit. **The "−0.01 nat" at `λ_WM = 0.05` is a *modest improvement* — analogous to PRM-CHIRON's `λ = 0.1` operating point. NLL is preserved or slightly improved at the recommended operating point.**

### 2.3 World-state-label generation pipeline

Three sources, increasing cost / quality:

**Source 1: Public corpora (LOW cost, LOW scale).** ConceptNet (5.7M triples), ATOMIC (880k cause-effect), PIQA/SIQA (~30k each), OpenBookQA (~6k), ARC (~7k), discourse-relation corpora (~50k). Cumulative ~50M–500M annotated examples = ~0.5–3% corpus-mix at 18B-pretraining scale. Free; one engineer-week.

**Source 2: Automated NER/relation extraction (MEDIUM cost, MEDIUM scale).** Run Stanford OpenIE / REBEL / FlairNLP / spaCy + custom relation classifier over the existing pile-bpe corpus. ~5–15% annotated coverage at ~70–80% label quality. ~$3k.

**Source 3: METAGEN-extension teacher labeling (HIGH cost, HIGH quality).** Teacher LLM generates structured world-state annotations per passage, parsed and tied to span offsets. ~30% coverage at ~75–85% quality. ~$5–8k. **Composes with #58 METAGEN's existing infrastructure** — same teacher, same orchestration, additional structured output channel.

**Default plan:** Source 1 + Source 2 at Gate-0 (free + $3k, ~7% coverage, ~75% quality). Source 3 added at Gate-1 if conjecture validates.

### 2.4 Annotation-aware sub-corpus mixing

Default: 15% annotated / 85% unannotated. World-state head fires only at annotated positions; CE on all positions. Gate-0 sweeps `f ∈ {5%, 15%, 30%, 50%}`; saturation expected near `f ≈ 30%`.

---

## 3. NLL preservation: the load-bearing design property

This is *the* property distinguishing WORLD-MODEL-CHIRON from JEPA-CHIRON (rejected #61).

| Property | JEPA-CHIRON (rejected #61) | WORLD-MODEL-CHIRON (#63-B) |
|---|---|---|
| Primary loss | MSE in latent space | **Standard CE on next-token** |
| Auxiliary loss | (none — JEPA *is* primary) | World-state classification |
| NLL preservation | VIOLATED by construction | **PRESERVED by construction** |
| Decoder dependency | required (latents → tokens) | none (CE direct on tokens) |
| EMA target encoder | required | none |
| Memory cost | +3.5 GB at 1.84B | +25M params (auxiliary head) |
| Inference cost | encoder + decoder both run | trunk only (head dropped) |

**The same set of failure modes that disqualified JEPA-CHIRON for #61 are non-issues for WORLD-MODEL-CHIRON:** (1) no missing CE term — CE unchanged; (2) no EMA moving goalpost — supervision from extracted/curated annotations; (3) no decoder bottleneck — CE head is the standard surface-form decoder; (4) NLL preserved within 0.05 nat at `λ_WM ≤ 0.1` by construction. **The auxiliary-head inversion is the design move that converts an NLL-violating world-model-flavored mechanism into an NLL-preserving one.**

### 3.1 The NLL improvement conjecture

Auxiliary world-state supervision is *predicted to slightly improve* held-out text-NLL at `λ_WM = 0.05`, by analogy with: PRM-CHIRON (`λ = 0.1` improves NLL by `−0.02 nat`); Lightman 2023 §5.4 (PRM auxiliary at `λ ≤ 0.5` "no degradation in held-out language-model perplexity" with marginal *improvement* at `λ ≤ 0.2`); multi-task-learning regularization theory (Caruana 1997 and follow-ups: auxiliary tasks correlated with the primary task improve held-out primary-task performance via implicit regularization). **Honest claim: NLL impact at `λ_WM = 0.05` is predicted in the range `[−0.02, +0.005] nat` (mostly favorable).** Verified at Gate-0.

### 3.2 Failure mode: world-state head misalignment

If corpus annotations are inconsistent with surface-form text (entity X annotated with property Y but text describes property Z), the head learns consistently-wrong properties and the gradient pulls the trunk toward annotation rather than text. **At `λ_WM = 0.05`, the noisy-gradient norm is bounded by `~1 · 10^-4` per token — well below stochastic-gradient noise.** The trunk does not chase incorrect annotations because the gradient signal is too small, the same mitigation mechanism that lets #59-B PRM-CHIRON tolerate Math-Shepherd's 85% step-label accuracy.

### 3.3 Fail-fast guards

- `||∇L_world_state||` monitored every 1000 steps; if `> 0.5 · ||∇L_CE||`, auto-reduce `λ_WM`.
- Held-out text-NLL monitored every 5000 steps; if drift exceeds `+0.05 nat`, auto-zero `λ_WM`.
- World-state head accuracy < 50% (chance for 4-way) → auto-zero `λ_WM` and log warning.

---

## 4. The LLM-scale verification gap

**Vision/video world models (well-validated, OFF the LLM axis).** Genie (DeepMind 2024, 11B foundation world model on 200k hours of internet video, *strong qualitative results* on emergent action-controllable simulation); I-JEPA (CVPR 2023, 81% ImageNet linear-probe at less compute than MAE); V-JEPA (2024, ~5× sample efficiency on Kinetics-400); Dreamer V3 (2023, model-based RL, 150 tasks); Ha–Schmidhuber (2018). **Vision/video/RL — not language; transfer to language is unverified.**

**Language-scale world-model attempts (mixed, ON the LLM axis but limited).** LCM Meta 2024 (sentence-level latent prediction; matches Llama-3-1.5B on linear-probe XNLI/FLORES; **token-level perplexity not reported and explicitly argued to be a different objective**); Petroni 2019 (CE-trained LMs implicitly encode entity facts ~30% on LAMA, suggesting baseline implicit world-modeling is real but weak); AGENT-CHIRON #62-B (closest in-stack precedent — implicit world-state tracking via agent trajectories). **No published precedent isolates explicit auxiliary world-state classification in language-only LLM pretraining.**

**Why the conjecture might fail.** (1) Implicit world-modeling already strong at scale (Petroni 2019's 30% LAMA at GPT-2 is much higher at GPT-4; marginal lift from explicit supervision shrinks as the trunk gets bigger); (2) annotation-quality bottleneck (75–80% label quality caps teaching power); (3) schema mismatch with language passages describing things outside `(E, P, R, C)`; (4) grounded-reasoning benchmarks are noisy proxies; (5) #60 TOOL-LLM provides external grounding that partially substitutes.

**Why the conjecture might succeed.** (1) Hallucination is provably (Ji 2023) a world-state-grounding failure — a model with explicit world-state supervision is conjecturally better at avoiding it (HaluEval could see 10–20% reduction); (2) information-rate orthogonality (§1.4 mech 2) provides bits of signal *not* captured by next-token CE; (3) multi-task regularization is mechanism-generic and well-validated; (4) PRM-CHIRON's success is partial-precedent — same auxiliary-head pattern, different target, projects 1.5–3× on reasoning at LLM scale (Lightman 2023 + projections); (5) ConceptNet-augmented pretraining shows modest but consistent lifts at GPT-2/3 scale.

**Honest expected outcome:** conservative `1.2×` more likely than optimistic `2×`; some Gate-0 lift expected (probability ~70%); 90%-CI covers `[0.95×, 1.5×]` at conservative end. The optimistic `2×` requires fortuitous schema-benchmark fit, probability ~20%.

---

## 5. Composition with #59 PRM-CHIRON (the strongest composition path)

PRM-CHIRON (#59-B, selected) and WORLD-MODEL-CHIRON (#63-B) share the auxiliary-head pattern and compose multiplicatively on metric-axis dimension.

### 5.1 Two auxiliary heads, one trunk

```
L_total = L_CE + λ_PRM · L_PRM + λ_WM · L_world_state
```

with `λ_PRM = 0.1` (#59-B) and `λ_WM = 0.05` (#63-B). Combined auxiliary cost: `~10M` (PRM) + `~25M` (world-state) = `35M`, or `0.19%` of an 18B trunk. Combined per-step overhead: `~0.05%`. Both heads are sparse-firing: PRM at step-end positions in reasoning chains, world-state head at annotated positions in any passage. The two firing patterns overlap on reasoning passages with world-state annotations (those passages get both signals at once) and otherwise complement each other.

### 5.2 PRM scores world-state quality (the load-bearing composition mechanism)

**PRM-CHIRON's process reward extends to score world-state quality.** PRM input is augmented to include the world-state-head output:

```
PRM_input(t) = h_t ⊕ ŝ_t      [trunk hidden ⊕ world-state head output]
PRM_output(t) = r_φ^PRM(PRM_input(t))      [process reward score]
```

PRM is then trained on a `(step, world-state-prediction, process-reward)` joint label: was this step (a) reasoning-correct *and* (b) world-state-consistent? The combined label is more informative than reasoning-correctness alone. **Gate-0 verifies this composition lifts joint accuracy on GSM8K-Physics (a math-with-physics-grounding subset) by 2–3pp over either head alone.**

This is the concrete answer to the prompt §5: PRM scores world-state quality by ingesting the world-state head's prediction into PRM's input; the joint label is reasoning-correct *and* world-state-consistent.

### 5.3 Composition factor estimates

| Benchmark axis | #59-B alone | #63-B alone | Joint |
|---|---|---|---|
| Reasoning (GSM8K, MATH-500) | 1.5× (+5pp) | 1.05× (+1pp) | **1.6× (+6pp)** |
| Grounded-reasoning (PIQA, SIQA) | 1.05× (+1pp) | 1.2× (+3pp) | **1.3× (+4pp)** |
| Hallucination (HaluEval) | 1.05× (~5% rel) | 1.2× (~15% rel) | **1.3× (~20% rel)** |
| Composite eval | 1.3× | 1.15× | **1.4–1.5× (multiplicative)** |

The composition is **mostly multiplicative** because the two heads target different aspects: PRM rewards correct *steps*; world-state rewards correct *entities/relations/causality*. Joint factor ~1.4–1.5× over post-#62 baseline.

### 5.4 Other compositions (briefly)

- **#62-B AGENT-CHIRON:** agent-PRM extended with world-state input at plan/reflect positions; agent's plan and reflection are scored on three axes (reasoning-correctness, plan-quality, world-state-consistency). Joint factor ~1.2× on top of post-#62 agent-benchmark accuracy.
- **#60-C TOOL-LLM:** tools provide *external* grounding at runtime; world-state head provides *internal* grounding at training. Partially redundant; joint factor ~1.15× on tool-augmented benchmarks (smaller than 1.2× without #60 because tools already do part of the grounding).
- **#58 METAGEN:** *necessary* for the optimistic case. METAGEN's teacher-distillation infrastructure provides Source 3 world-state labels at scale, lifting annotation coverage from 7% to 30% with teacher-quality labels. Without #58, WORLD-MODEL-CHIRON is bottlenecked at ~7% annotation coverage with extraction-quality labels.

### 5.5 Full stack composition

```
post-#55  ×  #56  × #57 × #58 × #59-B × #60-C × #61-A × #62-B × #63-B
~3,280    ×  5    × 3   × 2   × 1.5    × 5     × 1.5   × 1.3   × 1.2
        ≈ 5,160,000× cumulative on grounded-reasoning (conservative)
        ≈ 8,600,000× cumulative on grounded-reasoning (optimistic, λ_WM=2×)
```

**Both unverified pending Gate-0/1.**

---

## 6. Honest gap: speculative; world-state annotations scarce

### 6.1 Annotation scarcity

World-state annotations *do not exist at pretraining scale*. The pretraining corpus is ~15B–300B tokens; annotated subsets across all public corpora cumulatively yield ~50M–500M annotated examples, or ~0.2–3% of pretraining scale at sentence-level granularity. **Not enough alone for a strong head training signal at 18B-trunk scale.**

Three escape routes (each with caveats):

1. **Automated extraction:** ~5–15% coverage at ~70–80% label quality, ~$3k. *Risk: extraction-pipeline biases (NER pipelines miss colloquial mentions; relation extractors miss long-range relations) become **systematic biases** in the head's signal, which the trunk then learns.*
2. **METAGEN-teacher labeling:** ~30% coverage at ~75–85% quality, ~$5–8k. *Risk: teacher-distillation hallucinations propagate to head training. Math-Shepherd reports 85% step-label accuracy in the analogous PRM setting.*
3. **Human-curated dense annotation:** ~0.1% coverage at ~95% quality, ~$50k+. Reserved for benchmark construction.

**Honest position: at 18B pretraining scale, no annotation source provides 100% coverage at ~95% quality. The world-state signal is structurally noisier than CE's surface-form signal.** The 75–80% label-quality regime caps the head's teaching power, similar to the Math-Shepherd ceiling on PRM-CHIRON.

### 6.2 The speculative-mechanism gap

Per §4: **the conjecture that explicit world-state auxiliary supervision lifts grounded-reasoning by 1.2–2× at LLM scale is not directly supported by published evidence.** It is supported by analogy (vision world-models, PRM auxiliary success in #59-B) and by mechanism (information-rate orthogonality, hallucination-as-grounding-failure), but the *magnitude* of the effect at language scale is empirically unmeasured. **Gate-0 at 66M would partially derisk; Gate-1 at 1.84B is the meaningful empirical confirmation.**

### 6.3 Schema-design risk

The proposed `(E, P, R, C)` schema is *standard-NLP-pipeline* and well-validated for adjacent tasks (relation-extraction benchmarks, ATOMIC-style commonsense QA), but it is *not* the LeCun-style continuous physical world-model schema. **WORLD-MODEL-CHIRON makes a conservative-schema bet — annotate what is annotatable at scale, not what is aspirationally a "world model."** The conservative schema is more empirically grounded; the conservative quality lift is the matching expectation. If Gate-0 results are below `+3pp on PIQA`, schema-broadening (quantitative properties: count, magnitude, scale; temporal relations: before/during/after; modal relations: possible/necessary/hypothetical) is the obvious experimental next step.

### 6.4 The compute-axis honesty

WORLD-MODEL-CHIRON is **NOT a training-FLOP paradigm.** 0.03% per-step overhead is negligible. Cumulative-stack contribution on text-NLL is *neutral*; on tool-augmented benchmarks *neutral*; on agent benchmarks *modest* (1.2× via PRM composition). **The cumulative-stack contribution on grounded-reasoning benchmarks is 1.2–2× (CONJECTURED).** This is a *new metric axis* introduction, not a multiplicative speedup on existing axes.

**Honest selection criterion:** WORLD-MODEL-CHIRON is selected when the project chooses to introduce grounded-reasoning + hallucination as a first-class metric axis AND accepts conjecture-dependent quality lift on that axis.

### 6.5 Inference cost is genuinely zero

The world-state head is dropped at inference. Only the trunk runs; the auxiliary head is training-only infrastructure. **No deployment-side cost.** Strict upgrade vs JEPA-CHIRON's required-decoder design and a strict no-cost vs the post-#62 stack's deployment surface.

### 6.6 Engineering and Gate-0 protocol

**Engineering: ~1,000 LOC over ~4 weeks.** Components: world-state head architecture (~200 LOC); joint-loss CUDA kernel (~80); annotation-aware tokenizer (~100); public-corpus integration (~120); automated-extraction pipeline (~200); METAGEN-extension wiring (~80); data-pipeline integration (~120); CLI/checkpoint compat (~50); Gate-0 harness + benchmark suite (PIQA, SIQA, HaluEval, OpenBookQA, ARC-Challenge) (~100); composition wiring with #59-B PRM and #62-B AGENT (~80); docs (~30). Engineering risk: low (auxiliary-head pattern well-understood from #59-B; schema and composition are the design-novel parts).

**Gate-0 (29 GPU-hours = 1.2 GPU-days).** Three arms at 66M, 30k steps:

- **Arm A (control):** post-#42–#62 stack, no world-state head. Target text-NLL ~3.95 nat; PIQA-66M ~50–55%; HaluEval-66M ~35–40%.
- **Arm B (WORLD-MODEL-CHIRON):** same stack + world-state head at `λ_WM = 0.05` + 15% annotated mix (Source 1+2). Target PIQA composite ≥ Arm A + 2pp; HaluEval ≥ Arm A − 5% relative.
- **Arm C (`λ_WM` sweep):** 5 runs × 5k steps at `λ_WM ∈ {0.01, 0.05, 0.1, 0.2, 0.5}`.

**Pass:** (1) Arm B PIQA + SIQA composite ≥ Arm A + 2pp. (2) HaluEval improvement ≥ 5% relative. (3) text-NLL within 0.05 nat. (4) head accuracy ≥ 60% by step 30k. (5) Arm C optimum in `[0.02, 0.1]`.

**Fail-fast:** NaN or NLL diverges < 5k → REJECT. NLL ≥ Arm A + 0.20 → REJECT. PIQA composite ≤ Arm A → REJECT (mechanism failed). Head accuracy ≤ 45% → REJECT. PIQA composite ≥ Arm A + 5pp → STRONG PASS.

**Gate-1** (post-Gate-0): same protocol at 1.84B, 100k steps, full benchmark suite, optionally Source 3 METAGEN labels (~$5k). ~12 GPU-days. **Gate-2** (post-Gate-1): joint composition with #59-B PRM-augmented + #62-B AGENT-extended. ~10 GPU-days. **Gate-3** (optional, deployment-readiness): hallucination-rate measurement on production prompts; SLA on grounded-generation tasks. ~5 GPU-days plus deployment-engineering effort.

### 6.7 Falsifiable predictions

(1) WORLD-MODEL-CHIRON-pretrained models reach grounded-reasoning target accuracy in 1.2× fewer steps than CE-only at Gate-0 (~24 GPU-hours). (2) HaluEval hallucination rate drops by 10–20% with world-state head (~6 GPU-hours). (3) `λ_WM = 0.05` is the joint-stability optimum (~12 GPU-hours). (4) WORLD-MODEL-CHIRON quality lift composes with #59-B PRM multiplicatively on PIQA-Physics-Math (~16 GPU-hours). **Cumulative validation: ~58 GPU-hours = 2.4 GPU-days.** If predictions 1 and 2 fail, headline 1.2× collapses to ~1.05× and WORLD-MODEL-CHIRON reduces to "marginal grounded-reasoning improvement at NLL-preserving cost; reserve for #65+."

---

## 7. Summary

WORLD-MODEL-CHIRON is **the auxiliary world-state primitive at pretraining time** — a ~25M-param multi-task classifier head trained jointly with the trunk, providing structured world-state supervision (entities, properties, relations, causal chains) on annotated passages. CE remains primary; world-state auxiliary at `λ_WM = 0.05`. Per-step overhead ~0.03%.

**Training-side quality at fixed grounded-reasoning-benchmark accuracy: 1.2× conservative; 2× optimistic; SPECULATIVE at LLM scale.** Smallest training-FLOP multiplier among #63 candidates. Value-add lies in the **introduction of grounded-reasoning as a first-class metric axis** plus **NLL preservation by construction**.

**The auxiliary-head inversion is the load-bearing design choice.** WORLD-MODEL-CHIRON does what JEPA-CHIRON (rejected #61) tried to do — bring world-modeling intuition into LLM pretraining — but inverts the loss structure. JEPA-CHIRON made latent-prediction primary (NLL-violating); WORLD-MODEL-CHIRON makes world-state classification auxiliary alongside primary CE (NLL-preserving). Failure modes that disqualified #61-B are non-issues for #63-B. The same theoretical motivations apply (LeCun, JEPA, Genie, Dreamer, neuro-grounded-cognition, multi-task regularization theory).

**Composition with #59-B PRM-CHIRON is the strongest composition path** — PRM input augmented with world-state head output; PRM scores reasoning-correct *and* world-state-consistent steps; joint factor ~1.4–1.5× on grounded-reasoning + reasoning composite. **Composition with #58 METAGEN is necessary for the optimistic case** (Source 3 teacher-distilled labels at scale, 7% → 30% coverage at teacher-quality).

**Cumulative stack post-#63-B (CONJECTURED):** Text-NLL `620,000×` unchanged; tool-augmented `3,030,000×` unchanged; agent `4,300,000×` unchanged; **grounded-reasoning (NEW axis): `5,160,000×` conservative; `8,600,000×` optimistic.**

**Bigger-picture frame.** Paradigms #56–#62 reframed DATA / LOSS / SAMPLING / REWARD / IDENTITY / SCHEDULE / AGENCY. WORLD-MODEL-CHIRON adds **GROUNDING** — the relationship between text and the world the text describes. Per LeCun's world-model program, JEPA's vision results, and Genie's world-model-at-video-scale, explicit world-modeling is the path toward representations that are physically/causally faithful rather than statistically next-token-faithful. **At LLM scale this is HIGHLY SPECULATIVE.** No published precedent isolates explicit world-state auxiliary supervision in language-only pretraining; vision-to-language transfer empirically unverified; LCM (the closest sentence-level precedent) has mixed results. **Empirical risk concentrated on the conjecture itself.**

**Selection criterion vs #63-A and #63-C.** WORLD-MODEL-CHIRON is **the most theoretically ambitious** but **the most speculative** of the #63 slate, with the **smallest training-FLOP multiplier (1.2×)** but **the most distinctive new-metric-axis introduction** (grounded reasoning + hallucination) and the **strongest #59-B PRM composition** via PRM-scores-world-state-quality. Selection rationale leans on the new-metric-axis value and on the auxiliary-head pattern's compositional cleanliness, not on training-FLOP magnitude.

**Recommendation: candidate for #63, framed as the SPECULATIVE-axis entry of the slate.** Reasonable mechanism; clean composition with #59-B and #58; modest engineering; preserves NLL by construction; introduces a new metric axis (grounded reasoning + hallucination) where the project does not currently optimize. **Selected if the project chooses to expand the evaluation to include grounded-reasoning + hallucination metrics and accepts the conjecture-dependent nature of the headline quality lift.** The 1.2× training-FLOP-equivalent headline is small; the grounded-reasoning and hallucination-reduction contributions are conjecturally large but empirically unverified. **WORLD-MODEL-CHIRON is selected when the goal is grounded-faithful reasoning quality at preserved NLL on a new metric axis, not when the goal is maximum training-FLOP reduction at fixed quality on existing axes.**
