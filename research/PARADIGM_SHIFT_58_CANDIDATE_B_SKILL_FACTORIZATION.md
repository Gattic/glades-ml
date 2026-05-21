# Paradigm Shift #58 Candidate B — SKILL-FACTORIZATION (decompose LLM knowledge into discrete reusable skill modules; library-of-skills reframe of pretraining)

**Status:** candidate-B design for paradigm shift #58. **Recommendation up front: REJECT for #58.** SKILL-FACTORIZATION is positioned as an *exploratory niche paradigm* worth re-examining once the empirical literature on LLM skill-cluster identifiability matures, but it is not selectable for the iter-202 #58 slot.
**Date:** 2026-05-08 (Ralph-loop iter 202, post-#57 SCROLL-PROMOTED selection, under the iter-200 brief: *"novel architectures, algorithms, and training methods by looking at the bigger picture instead of focusing on microoptimizations"*).
**Predecessors:** `PARADIGM_SHIFT_53_CANDIDATE_B_MOSAIC_MOE.md` (closest sibling — SKILL-FACTORIZATION extends MoE's "different sub-networks for different inputs" idea to **independent training on different data, composed at inference**); `PARADIGM_SHIFT_57_CANDIDATE_B_METAGEN.md` (the previous candidate-B; SKILL-FACTORIZATION is even more speculative); `chiron_architecture.md` (reversible shear makes per-skill `Wo=0` identity insertion cheap, paradigm #39 RLG primitive).
**Axis:** **organizational / training-time architectural decomposition.** Replace "train one large monolithic 1.84B network on the union of all data" with "train K small ~10M-parameter skill modules on K disjoint data subsets, then compose them at inference." Closest published reference points: LoRA-Hub (Huang 2023), AdapterFusion (Pfeiffer 2021), Branch-Train-Merge / Branch-Train-MiX (Sukhbaatar et al. 2024), TIES-Merging (Yadav et al. 2024). None has been demonstrated to deliver a per-step or per-epoch wall-clock advantage at LLM-pretraining scale; all delivered advantages are at fine-tuning or task-specialization scale.
**Tagline.** *Reframe the model as a library, not a brain. If knowledge is a finite list of discrete skills, training the list one entry at a time on its own data is K× cheaper than training one network on the union. The unanswered question is whether LLM knowledge is a list at all.*

---

## 0. Executive summary (HONEST: this is too speculative for #58, recommend rejection)

After paradigms #1–#57 the cumulative single-GPU stack reaches **~41,300× wall-clock at 18B / `T = 1024` with same final NLL** (per `PARADIGM_SHIFT_57_CANDIDATE_A_SCROLL_PROMOTED.md` §0). Each shipped paradigm has compressed a load-bearing object, reorganized compute on a fixed loss, reframed the loss / corpus / sampling, or restructured architecture along a *measured* axis. SKILL-FACTORIZATION proposes a fifth class: **training-time decomposition of a single LLM into K independently-trained 10M-parameter modules.** The premise is that LLM knowledge is decomposable into discrete identifiable skills (arithmetic, code, French, biology, …) admitting independent training and inference-time composition. If true: K · C_skill ≈ C_monolithic, but each skill's data efficiency is ~5× better via domain-specialization, yielding a conservative **3–5× wall-clock to fixed final NLL**.

**The premise is not established at LLM scale.** The honest empirical position in May 2026:

- **Sparse-autoencoder (SAE) interpretability** (Bricken et al. 2023; Cunningham et al. 2024; Templeton et al. 2024 *Towards Monosemanticity*) reports **~10⁵–10⁶ features per residual stream**, not the ~10²–10³ "skills" this paradigm posits. The features are micro-concepts, not skills.
- **MoE expert-specialization** (Mixtral, Switch, DeepSeek-MoE): jointly-trained experts learn *overlapping, non-interpretable* token-level routing. Mixtral §3.4: "we did not observe obvious patterns in the assignment of experts to topics." The experts are not skills.
- **Model-merging** (Ilharco 2023; Yadav 2024 TIES; Yu 2024 DARE): independently-trained adapters merge by task-vector arithmetic only at *fine-tuning* scale on a frozen base, with degradation when tasks are not pre-aligned.
- **Branch-Train-MiX** (Sukhbaatar et al. 2024) — the closest published LLM-scale instantiation: branch a base, train K branches on K domains in parallel, merge into sparse-MoE. **Reports ~1.2–1.5× over iso-FLOP dense baseline on domain benchmarks; net wall-clock at fixed pretraining NLL approximately neutral.**

**The 3–5× speedup claim is a model assumption, not an empirical finding.** The published ceiling for the closest analog is ≈1.3×.

**Engineering scope.** ~2400 LOC over 8–10 weeks: discovery clustering (~600), module training infrastructure (~700), composition layer (~500), cross-model harness (~300), Gate-0 assertions (~300). This is **2–3× the engineering scope of any paradigm in #42–#57**, for an empirical advantage unverified at LLM scale.

**Recommendation.**

1. **REJECT for #58.** Combined (a) speculative empirical premise, (b) absence of LLM-pretraining precedent reporting >1.5× wall-clock at fixed NLL, (c) 2–3× engineering scope vs precedent, (d) candidate-A is stronger.
2. **Position as exploratory niche paradigm** in `DEFERRED_PARADIGMS_CLOSURE.md`. Re-examine when one of: (i) SAE feature counts drop from 10⁵ to 10² in a published study; (ii) Branch-Train-MiX-class results report >2× wall-clock at 1B+ pretraining; (iii) the project hits MoE expert-collapse failure mode where explicit cross-cluster module reuse becomes the obvious mitigation.
3. **Do not run Gate-0 in iter-202.** The Gate-0 probe (cluster identifiability at 1.84B, 32-seed k-means + AMI vs domain labels) is a 4–6 GPU-day experiment with high prior probability of failure given the SAE evidence. Per `ralph_loop_gate0_methodology.md`, preempt the cost when the literature already supplies the answer.

The remainder develops the proposed mechanism, methodology, CHIRON-stack synergy, and #53 composition fairly, so the rejection is principled.

---

## 1. The decomposability premise: where SKILL-FACTORIZATION lives or dies

### 1.1 What "discrete skills" would mean concretely

For SKILL-FACTORIZATION to deliver, four conditions must hold simultaneously:

1. **Cluster identifiability.** Clustering on activations at depth `l*` of a converged 1.84B CHIRON yields `K` clusters with mean pairwise AMI ≥ 0.6 across seeds **and** AMI ≥ 0.7 against ground-truth corpus-provenance labels. If clusters don't align with domains, discovery fails.
2. **Module separability.** A 10M-parameter network trained on cluster-`k` data alone reaches (within 0.1 nat) the per-token NLL on cluster-`k` test data that the monolithic 1.84B reaches. A 184× smaller module must carry the relevant capacity.
3. **Composition non-degradation.** Routing-time composition of `K` modules yields aggregate NLL within 0.2 nat of the monolithic baseline.
4. **Cross-model transfer.** A skill module trained against `M_A` performs usefully on `M_B` after a bridging adapter, otherwise "reusable across models" is false advertising.

**Of these four, only (4) has partial empirical support**, and only at fine-tuning scale (LoRA-Hub, model-merging). (1) and (2) are unmeasured at LLM scale; (3) is *demonstrated to fail* in MoE expert-specialization studies.

### 1.2 Information-theoretic restatement

Let `D = ⋃_k D_k` partition the corpus. SKILL-FACTORIZATION's claim is that **`Σ_k H(D_k) ≪ H(D)`** — cross-skill correlations are negligible. This is **the opposite** of the dominant deep-learning representation-learning finding: cross-domain transfer is positive and large (vision, language, multimodal — Pile-trained baselines outperform domain-specific baselines on every covered domain; math improves code and vice versa, *Minerva*, *DeepSeek-Math*).

If correlations are large, training skills independently *discards* the cross-skill signal. The 5× per-skill data-efficiency claim then becomes a 5× *loss* of cross-domain transfer signal, and net wall-clock to fixed NLL is plausibly *negative*.

**Bottom line:** as of May 2026, the dominant published evidence is that LLM knowledge is continuous and entangled, not discrete and factored. The contradiction with SAE / MoE-specialization findings is not fatal (CHIRON-1.84B may differ from analyzed models; MoE findings may be joint-training specific) but cannot be ignored.

---

## 2. Skill discovery methodology (proposed)

For completeness, the proposed mechanism. The methodology would be sound *if* §1's premise held.

### 2.1 Activation-clustering at depth `2L/3`

Choose a discovery layer `l*` at depth `2L/3` (where representations are maximally semantic but not yet output-specialized; cf. probing literature). Extract residual-stream activation `h_l*[t]` for each token in a 100M held-out sample. Run mini-batch k-means with `K ∈ {16, 64, 256}` and 32 seeds. **Pass conditions:** mean pairwise AMI ≥ 0.6 (stability); AMI ≥ 0.7 vs corpus-provenance labels (semantic meaning). Both unmeasured at LLM scale; the SAE evidence (10⁵–10⁶ features) suggests likely failure.

### 2.2 Soft assignment at inference

`r[t, k] := softmax(-‖h_l*[t] - c_k‖² / τ)`; top-`k_top` clusters activate per token. This is exactly MoE's routing with one critical difference: SKILL-FACTORIZATION's `c_k` are **frozen from offline clustering**, MoE's are **learned end-to-end**. Mixtral §3 and Switch §4.3 both show learned wins offline-clustered by 2–3% perplexity. The frozen choice is structural for SKILL-FACTORIZATION (it's what enables cross-model reuse) but costs perplexity vs MoE. **Even granting discovery succeeds, inference composition is empirically dominated by learned MoE routing.**

---

## 3. Per-skill module training

### 3.1 Module architecture

Three options: **LoRA adapter** (~1.7M params/skill, composes by addition into FFN/attention `Wo`); **small CHIRON sub-block** (1–4 reversible-shear blocks at `m_skill = 256–512`, ~10M params/skill); **adapter-fusion module** (Pfeiffer 2021, bottleneck inside each block, ~5M params/skill). Cost analysis below uses LoRA-adapter sizing as the most defensible.

### 3.2 Per-skill training and compute

Skill `k` trains on 90% `D_k` + 10% uniform `D` (uniform tail prevents over-specialization to surface lexical statistics). Step count: 5–10× fewer than monolithic for equivalent per-token NLL on `D_k`.

Naive per-skill compute: `C_k ≈ (10M / 1.84B) · (1/5) · C_monolith ≈ 0.001 · C_monolith`. Across `K = 64` skills: **~0.07 · C_monolith → naive 14× wall-clock advantage**.

Honest discount factors:

- **Base-model bootstrap.** SKILL-FACTORIZATION needs a converged base model to produce `h_l*[t]`. Base-model training is ~half the monolithic cost. **Halves the speedup.**
- **Module-composition fine-tuning.** Independent training produces interference; ~10% of monolithic compute needed to fine-tune. **Another 1.1–1.3× discount.**
- **Cross-skill correlation loss.** Independent training discards positive cross-domain transfer; multitask-vs-single-task literature reports 10–30% NLL degradation on cross-cluster tokens.

After honest discounting: **3–5× over dense-monolithic baseline**, and the **additional speedup over the post-#57 stack is closer to 1.2–1.5×** — directly aligned with Branch-Train-MiX.

### 3.3 Why prior paradigms eat most of the gain

- **#57-A SCROLL-PROMOTED** already extracts per-token informativeness gain — preferential sampling regardless of domain.
- **#57-B METAGEN** (if selected) targets synthetic data at student-distribution gaps — eats the cross-domain gap-filling benefit.
- **#56 DISTILL-FORWARD** carries teacher's full soft-target distribution — modular decomposition gains less when target is already maximally informative.
- **#53 MOSAIC-MOE** (if shipped) provides token-routing-by-expert — captures architectural-sparsity benefit without the discrete-skills premise.

**Each of #53, #56, #57-A, #57-B already takes a slice.** The marginal left for SKILL-FACTORIZATION on top is small.

---

## 4. CHIRON-stack synergy (would-be, premise-conditional)

Granting the premise, four real engineering wins:

1. **Reversibility for module insertion.** The shear `(q, p) ↦ (q, p + Y(q))` is bijective for any continuous `Y(q)`; a skill module inserts as an additional conditional shear `Y_skill_k(q)` weighted by routing `r[t, k]`. Insertion is free at the architectural level.
2. **#39 RLG warm-start.** Per-skill modules initialize with `Wo = 0`, making the un-finetuned composed model bit-exact equivalent to the base. Each skill progressively breaks identity in a controlled way.
3. **Modular checkpointing.** CHIRON's per-`K`-block anchor cache extends per-skill: memory scales with `(active skills / token) × (anchor cost / skill)`, not `K × (anchor cost / skill)`.
4. **KV-cache continuity.** Base-model attention is unchanged; skill modules see the shared KV cache without per-skill duplication.

These synergies are real **if** the premise holds. They do not validate the premise.

---

## 5. Composition with #53 MOSAIC-MOE

Two distinct modes if MOSAIC-MOE ships:

**Mode 1 — SKILL-FACTORIZATION as MoE expert factory.** Train each MOSAIC-MOE expert independently as a SKILL-FACTORIZATION module on cluster `D_k_e`; the learned MoE router still routes tokens at inference. Converts joint expert training to independent expert pretraining + joint routing finetuning. *Pro:* expert pretraining pipelines/parallelizes cheaply. *Con:* independent training produces interference; needs expensive joint-finetuning phase. **This is exactly Branch-Train-MiX**; result is approximately neutral on net wall-clock.

**Mode 2 — SKILL-FACTORIZATION layer atop MOSAIC-MOE.** Use MoE experts as base; insert per-cluster LoRA adapters on top for fine-grained specialization. *Pro:* if experts learn coarse partitions and LoRA modules learn fine within-domain skills, the two granularities may not overlap. *Con:* doubles optimizer-state surface and composition interference. Untested at LLM scale.

**Honest assessment.** Mode 1 is Branch-Train-MiX (~1.2×, *less* than MOSAIC-MOE alone). Mode 2 is novel and untested. The compositional opportunities are real but small; they do not rescue SKILL-FACTORIZATION from rejection.

---

## 6. The honest gap: skills as discrete clusters, unverified at LLM scale

Four independent strands of empirical evidence point the same direction:

**SAE evidence.** Templeton 2024 *Towards Monosemanticity*, Cunningham 2024, and Bricken 2023 have learned `d ∈ {2¹⁵–2¹⁷}` features at the Claude-3-Sonnet residual stream. Features are monosemantic at the **micro-concept level** ("Golden Gate Bridge," "increment counter," "French negation particle") — but not at the skill level. Arithmetic is distributed across thousands of features (place-value, addition, base-conversion, …). If the right factorization is `O(10⁵)` micro-concepts, `K = 64` modules are too coarse and will overlap heavily.

**MoE-specialization evidence.** Mixtral 8x7B §3.4: with `E = 8` jointly-trained experts, no expert specializes on a clearly identifiable topic — assignments correlate more with surface-token statistics (whitespace, subword frequency) than semantic content. DeepSeek-MoE forced topic specialization via auxiliary losses; improved interpretability marginally, did not improve perplexity. **Given freedom, MoE experts choose not to be skills.**

**Transfer-learning evidence.** Pile-trained baselines outperform domain-specific baselines on every covered domain (Gao 2020 §6); math improves code and vice versa (*Minerva*, *DeepSeek-Math*). SKILL-FACTORIZATION's independent training discards this signal — empirical 10–30% NLL degradation on cross-cluster tokens, with recovery requiring *more* compute.

**Branch-Train-MiX evidence.** Sukhbaatar 2024 — most directly comparable: K = 4 branches on math, code, Wikipedia, CommonCrawl, merged into sparse MoE. **Gain: 1.2–1.5× over iso-FLOP dense.** This is the empirical ceiling SKILL-FACTORIZATION must be evaluated against.

**The selection-bar gap.**

| Quantity | Value |
|---|---|
| Required minimum for #58 (consistent with #56 5×, #57 2.5× marginal) | **>2×** |
| Branch-Train-MiX published ceiling | **~1.3×** |
| Required additional speedup over Branch-Train-MiX | **~1.5×, no published precedent showing how** |

The gap is closeable only by an unverified empirical bet.

---

## 7. Cross-model transfer claim — separately speculative

The most distinctive SKILL-FACTORIZATION claim — that skills trained against base `M_A` can be reused on `M_B` — fares poorly under scrutiny. Published findings: LoRA adapters trained against Llama-2-7B don't transfer directly to Llama-2-13B or Llama-3-8B (residual-stream geometries differ); task-vector arithmetic (Ilharco 2023) requires source and target be fine-tuned from the **same base**, merging across pretrained bases fails; cross-architecture (Llama → Mistral) needs per-skill bridging adapters whose training cost typically exceeds re-training the skill from scratch.

CHIRON's reversible-flow architecture is unusual; **there is no public CHIRON skill library to draw from**. Cross-model transfer for CHIRON would either bootstrap the library inside the project (paying full training cost without the cross-model benefit) or build a Llama→CHIRON bridge (substantial engineering, architecturally unprincipled). **In the project's specific context, the cross-model claim is not actionable.** SKILL-FACTORIZATION should drop the cross-model dimension when evaluated for #58.

---

## 8. Engineering scope and Gate-0 (would-be)

**LOC breakdown:** discovery clustering (~600), per-skill module training infrastructure (~700), composition layer with routing (~500), cross-model transfer harness (~300, bracketed per §7), Gate-0 assertions (~300). **Total: ~2400 LOC, ~8–10 weeks.** Compare: #56 DISTILL-FORWARD ~750 LOC, #57-A SCROLL-PROMOTED ~980 LOC — SKILL-FACTORIZATION is 2–3× precedent.

**Gate-0 protocol (if pursued).** Extract `h_l*[t]` at depth `2L/3` on a 100M-token sample at 1.84B; mini-batch k-means with `K ∈ {16, 64, 256}`, 32 seeds; compute pairwise AMI across seeds and AMI vs pile-bpe shard provenance. **Pass:** pairwise AMI ≥ 0.6 AND AMI vs domain labels ≥ 0.5. **Cost:** 4–6 GPU-days standalone. **Expected outcome** per §6 evidence: pairwise AMI in `[0.2, 0.4]`, fails with high probability.

**Ralph-loop preemption applies.** `ralph_loop_gate0_methodology.md`: preempt Gate-0 cost when literature supplies the answer at high confidence. SAE + MoE-specialization + model-merging collectively give ≥ 80% prior Gate-0 fails. **Recommendation: do not run Gate-0. Reject by literature.**

---

## 9. Cumulative-stack accounting (honest)

For the record:

| Stage | Marginal speedup | Cumulative |
|---|---|---|
| Pre-#42 baseline | 1× | 1× |
| Post-#42…#55 (kernels, optimizer, memory, architecture) | ~3280× | 3280× |
| Post-#56 DISTILL-FORWARD | 5× | 16,400× |
| Post-#57 SCROLL-PROMOTED | 2.52× | ~41,300× |
| **Post-#58-B SKILL-FACTORIZATION (best-case empirical, Branch-Train-MiX-class)** | **~1.3×** | **~53,700×** |
| Post-#58-B SKILL-FACTORIZATION (premise-failure case, plausible) | ~0.95× (regression) | ~39,200× |

**Best-case marginal is 1.3×, well below the >2× iter-202 selection bar; failure case is a regression.** This is, separately from the speculative-premise argument, a sufficient quantitative reason to reject for #58.

---

## 10. Recommendation

### 10.1 For #58: REJECT

Six-point rejection rationale:

1. **Premise contradicts dominant published evidence.** SAE features are 10⁵–10⁶ not 10²; MoE experts don't specialize by topic; cross-domain transfer is large and positive (eaten by independent training).
2. **Closest published analog (Branch-Train-MiX) delivers ~1.3×**, below the >2× iter-202 bar.
3. **Cross-model-reuse claim is not actionable for CHIRON.** No external CHIRON skill library exists.
4. **Slices already taken.** #53, #56, #57-A, #57-B each take part of what SKILL-FACTORIZATION targets.
5. **Engineering scope is 2–3× precedent** (~2400 LOC, 8–10 weeks) for unverified empirical advantage.
6. **Gate-0 high prior failure probability** per literature; preempted under Ralph-loop methodology.

### 10.2 Position as exploratory niche paradigm

Add to `DEFERRED_PARADIGMS_CLOSURE.md`:

> **SKILL-FACTORIZATION (deferred from #58 candidate B, iter-202).** Library-of-skills reframe: K independently-trained ~10M-param modules composed at inference. Premise contradicted by 2024–2026 SAE and MoE-specialization literature; closest analog (Branch-Train-MiX) delivers ~1.3×. **Re-examine when** any of: (i) SAE feature counts drop from 10⁵ to 10² in a published study; (ii) Branch-Train-MiX-class results report >2× wall-clock at 1B+ pretraining; (iii) MOSAIC-MOE (#53 if shipped) hits expert-collapse failure where explicit cross-cluster module reuse becomes the obvious mitigation.

### 10.3 If the user pushes back

If the argument is *"the speculative ones are exactly the ones we should explore"*: the project has rejected speculative paradigms before (#36 KV-FACE, #41 ASTRA, both post-Gate-0). Their Gate-0 budgets were small relative to implementation; SKILL-FACTORIZATION's Gate-0 is also small (4–6 GPU-days) so running it as a literature-preempted check is defensible. However, implementation cost (~2400 LOC, 8–10 weeks) is not small — even a successful Gate-0 wouldn't justify the implementation budget without Branch-Train-MiX-class evidence improving.

**Middle path (optional):** run Gate-0 as a standalone 4–6 GPU-day cluster-identifiability experiment, document the (likely-failure) result, contribute to the literature record. Selection of #58 still goes to candidate-A. The default recommendation remains the simple rejection of §10.1.

---

## 11. Closing summary

SKILL-FACTORIZATION is a coherent, scientifically interesting reframe of LLM training as library curation rather than monolithic optimization. It is also a significantly more speculative bet than any paradigm in #1–#57. The premise — discrete-skills factorization at ~10² modules — is contradicted by 2024–2026 SAE-feature-density and MoE-specialization literature. The closest published analog (Branch-Train-MiX) delivers ~1.3×, below the iter-202 selection bar. Cross-model-reuse is not actionable for CHIRON. Engineering scope is 2–3× precedent. Marginal value over the post-#57 stack is plausibly negative under cross-skill correlation discounting.

**Recommendation: REJECT for #58. Position as exploratory niche paradigm in the deferred list with re-examination triggers tied to SAE / MoE-specialization / model-merging literature updates.**

This is the second time the project writes an "honest rejection" candidate document (after #41 ASTRA post-Gate-0). The same discipline applies: when a paradigm's premise contradicts dominant published evidence, reject by literature rather than by Gate-0 budget. The stack is healthy at ~41,300×; iter-202 selection should go to the candidate with the most defensible empirical foundation, which is not this one.
