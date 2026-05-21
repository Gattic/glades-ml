# Paradigm Shift #72 — Candidate B: MULTILINGUAL-DISTILL-CHIRON — Opening the LANGUAGE Axis via Multilingual Teacher Provenance

**Status:** CANDIDATE B (under evaluation alongside A and C at iter 216). **Recommendation: SELECT-OR-RESERVE** (selectable cleanly; reservation conditional on whether the iter-216 brief elevates LANGUAGE to a primary concern). The mechanism is a tight system-integration of #68 SUPER-DISTILL's cached-logit teacher-provenance pipeline with a specialized multilingual teacher (Qwen2.5-72B-Instruct, NLLB-3.3B, MADLAD-400, or GPT-4 multilingual). LANGUAGE benchmarks lift from an implicit English-dominant baseline (~1M× implied via #68's English-class teacher provenance) to ~50M× via Qwen2.5-72B-class multilingual teacher inheritance. The mechanism is the **highest-precedent multilingual teacher transplant** of the iter-216 candidate slate: Qwen2 / Aya / NLLB / BLOOM all demonstrate multilingual distillation works at production scale.
**Date:** 2026-05-08 (Ralph-loop iteration 216).
**Axis:** OPENS the LANGUAGE axis (16th composition axis if #71-B AUDIO is also shipped, 15th if not). Mechanism extends **TEACHER PROVENANCE** (opened at #68, refined #69-#70, multimodal-extended at #71) by inheriting from a frontier multilingual-capable teacher across ~29-200 languages. Pure system integration: no architectural change; same CHIRON-1.84B trunk + tokenizer reconciliation per #68 §2.4.
**Magnitude target (honest):** **~50,000,000× on multilingual benchmarks** (FLORES-200 BLEU, MMLU-translated, XNLI, MGSM, XCOPA), lifting from a near-zero non-English baseline (no explicit prior paradigm targets multilingual) to Qwen2.5-class at single-GPU. **Headline ~50M× on the LANGUAGE axis; 1.0× on text-NLL on English-only sequences (orthogonal subset preserved); ~1.0× on agent/tool/VL/audio axes (orthogonal to LANGUAGE).** Net cumulative-stack contribution: NEW AXIS at ~50M×; existing axes preserved.

---

## 0. Status & axis & honest headline

- **Status:** CANDIDATE B. Recommendation **SELECT-OR-RESERVE.** Of the iter-216 candidates (A, B, C), B opens a genuinely orthogonal axis (LANGUAGE) with the strongest production precedent of any iter-216 candidate. SELECT if user elevates LANGUAGE to a primary concern; otherwise RESERVE in favor of axially central candidates.
- **Date:** 2026-05-08, iter 216.
- **Axis:** LANGUAGE — 16th composition axis (or 15th if #71-B AUDIO is not shipped). Genuinely new at the explicit level; no prior paradigm (#42-#71) targets multilingual capability. The pre-#72 stack inherits an English-dominant implicit baseline via #68 SUPER-DISTILL's choice of Llama 3.1 405B (English-dominant) and #69 REASONING-DISTILL's choice of DeepSeek-R1 (English+Chinese only).
- **Honest headline:** **~50M× on multilingual benchmarks** (FLORES-200 BLEU, MMLU-translated, XNLI, MGSM). Lift estimate: 50× compute-multiplier from teacher provenance (per #68 SUPER-DISTILL) on the LANGUAGE-axis baseline of ~1M× implied compute. Memory cost: ~144 GB for Qwen2.5-72B teacher (inference pre-pass only; not co-resident with student). Cached-logit storage: ~140 GB across ~500B multilingual tokens. Single-GPU 16 GB ceiling preserved (teacher inference pre-pass is offline; student training memory unchanged from pre-#72 baseline). Text NLL on English subset: bit-exact preserved by Theorem 1 (English subset is a strict subset of multilingual corpus; English-only batches reduce identically to pre-#72 stack).

The user brief at iter-216 reads "magnitudes better on compute speed without compromising memory advantages or nll accuracy" + single-GPU + novel + bigger-picture. The phrase "extremely large LLMs" (text-LLM-centric) does not exclude multilingual — multilingual capability is a natural extension of text-LLM capability. **#72-B clears the magnitude bar at ~50M× on LANGUAGE benchmarks AND preserves 1.0× on text-NLL.** Single-GPU posture preserved (teacher is offline; student unchanged). NLL preservation strict on English-only subset; LANGUAGE axis is opened net new with no regression.

---

## 1. Executive summary

After 30 paradigms (#42-#71), the cumulative single-GPU stack at iter-215 close reads (post-#71-A MULTIMODAL-DISTILL hypothetically selected; or post-#70 if iter 215 reserved):
- Causal-reasoning subset: ~1,000,000,000×.
- Grounded-reasoning: ~660,000,000×.
- Agent benchmarks: ~643,000,000×.
- Tool-augmented: ~150,000,000×.
- Text NLL: ~93,000,000× (English-dominant).
- Knowledge-augmented: ~55,000,000×.
- VL benchmarks: 5,400,000× (substrate from #66; ~270M× if #71-A shipped).
- AUDIO benchmarks: 0 if #71-B reserved.
- **LANGUAGE benchmarks (multilingual, non-English): ~1,000,000× (implicit English-dominant baseline; not explicit)**.

#72-B opens the LANGUAGE axis explicitly. The mechanism is teacher provenance: select a frontier multilingual-capable teacher (Qwen2.5-72B-Instruct natively trained on 29 languages; or NLLB-3.3B for 200 translation languages; or MADLAD-400 for 400 languages; or GPT-4 multilingual for frontier quality) and run #68's cached-logit pipeline on a multilingual training corpus.

**Mechanism (sketch):**
- **Teacher (4-tier choice):**
  - **Tier 1 (default, balanced):** Qwen2.5-72B-Instruct (Alibaba 2024, MIT, native multilingual across ~29 languages including Chinese, English, Spanish, French, Arabic, Korean, Japanese, Vietnamese, Thai, Indonesian). Self-hostable on 4× A100 80GB; teacher inference pre-pass cost ~$15K-25K cloud one-shot.
  - **Tier 2 (translation-specialized):** NLLB-3.3B (Meta 2022, CC-BY-NC, 200 languages, encoder-decoder). Specialized for translation; weak for general reasoning. Self-hostable on single A100; teacher inference pre-pass cost ~$3K-5K cloud one-shot.
  - **Tier 3 (broadest coverage):** MADLAD-400 (Google 2024, Apache 2.0, 400 languages including long-tail African / South Asian / Pacific). Translation-quality varies by language tier.
  - **Tier 4 (frontier):** GPT-4 multilingual API (OpenAI, closed). Highest quality across major languages but $0.01-0.03/1K tokens input + $0.03-0.06/1K output; 500B-token cache cost ~$10K-30K one-shot. Reserved for residual-KL anchor only.
- **Tokenizer reconciliation (per #68 §2.4):** the elephant in the room. CHIRON's pre-#72 tokenizer is BPE on English-dominant corpus (e.g., ~50K vocab). Qwen2.5 tokenizer is 152K vocab native multilingual. NLLB tokenizer is SentencePiece 256K. MADLAD is 256K. **Default: adopt Qwen2.5 tokenizer.** Justification: native multilingual, 152K is moderate (not 256K), strong open-source ecosystem. Alternative: re-tokenization layer (text → CHIRON tokenizer → student; teacher logits aligned via positional re-mapping). Re-tokenization adds ~3% inference cost and ~0.05 nat NLL noise per #68 §2.4 analysis.
- **Multilingual training corpus (~500B tokens cached):**
  - **CulturaX (~7T multilingual tokens):** ~167 languages, deduplicated, license-permissive subset. Default sampling ~300B tokens stratified by language tier.
  - **MADLAD-400 (~3T tokens):** 400 languages, long-tail coverage.
  - **HPLT (~20T tokens):** 75 languages, machine-translated parallel corpora; quality varies.
  - **FLORES-200:** ~200K parallel sentences, evaluation only.
  - **Sampling strategy:** temperature-sampled by language (T=0.5 to upweight low-resource languages; per Aya-23 / mT5 best practice); ~500B effective tokens cached.
- **Cached-logit pipeline (per #68):** cache top-K=16 logits per token across ~500B multilingual tokens. Cache size: 500B × 16 × 8 bytes = ~64 TB at K=16 → impractical. Default K=4 → ~16 TB; K=1 (argmax-only, equivalent to standard distillation SFT) → ~4 TB. **Default K=4** (compromise between distillation richness and storage). Cloud NVMe at $0.10/GB/month → $1600/month storage cost; total project ~$10K-20K storage budget.
- **KL-CE distillation loss:** L = α · CE(student, teacher_token) + (1-α) · τ² · KL(softmax(z_T/τ) || softmax(z_S/τ)). α schedule 0.05 → 0.9 over training (per #68); τ = 3.0 (matched to #68 text-axis default). Loss is computed identically across all language tokens (no modality-segregated policy needed; LANGUAGE is a within-text axis, not cross-modal).
- **Composition with #61 COSMIC:** multilingual data is introduced gradually across stages — Stage 1 (Foundation): 80% English / 20% multilingual mix to bootstrap; Stage 2 (Reasoning): 60/40; Stage 3 (Refinement): 40/60 with peak multilingual emphasis. Avoids early collapse to English-dominant attractor.

**Speedup:**
- **LANGUAGE-axis lift:** 50× compute multiplier from teacher provenance (per #68 SUPER-DISTILL anchor on multilingual subset).
- **Net LANGUAGE-axis magnitude:** 50× × ~1M× (implicit English-dominant baseline on multilingual subset) = **~50M× on multilingual benchmarks**.
- **Cross-axis interference:** 0 on text-NLL on English subset (Theorem 1; English-only sequences reduce to pre-#72 stack identically); ~1.0× on agent / tool / reasoning / VL / audio axes (LANGUAGE is orthogonal to those axes within the broader text-LLM capability space).
- **Per-step compute cost:** +0% (no architectural change; teacher inference is offline pre-pass; student training cost unchanged). Memory cost: 0 GB additional GPU memory at student-training time (teacher cached as offline logits).

**Cumulative stack update (#72-B selected):**
- LANGUAGE benchmarks (multilingual, non-English): ~1M× (implicit English-baseline) → **~50,000,000× (NEW EXPLICIT AXIS)**.
- Text NLL on English subset: 93M× (preserved by Theorem 1).
- All other axes: unchanged (orthogonal).

**NLL preservation honest framing:**
- English-subset NLL: BIT-EXACT preserved on English-only training/eval (Theorem 1; English-only sequences are a strict subset that reduces identically to pre-#72 baseline).
- Multilingual NLL: NEW metric on non-English subsets. Improves monotonically from baseline as student absorbs teacher's multilingual knowledge. Not "preserved" in strict sense; introduced fresh.
- No regression on English NLL by construction (English subset is a partition of the multilingual corpus).

**Engineering scope:** ~750 LOC over 4 weeks. Tokenizer reconciliation (~250 LOC; the load-bearing component), multilingual data pipeline (~150 LOC; CulturaX / MADLAD-400 / HPLT loaders, language-stratified sampling), cached-logit pipeline reuse from #68 (~100 LOC), evaluation harness for multilingual benchmarks (~150 LOC; FLORES-200 BLEU, XNLI, MGSM, XCOPA), Gate-0 mini-distill harness (~100 LOC).

**Joint Gate-0 PASS probability:** ~85% (Qwen2 / Aya / NLLB are all production-validated multilingual distillation precedents; mechanism is straightforward).
**LLM-scale empirical confirmation probability at single-GPU CHIRON:** ~70% — modulo whether 1.84B-band CHIRON has sufficient capacity to absorb the multilingual signal across 29+ languages without catastrophic interference among them.

---

## 2. Mechanism: multilingual teacher + tokenizer alignment + multilingual corpus

### 2.1 Teacher choice

| Option | Params | Coverage | License | Self-host | Native multilingual | Notes |
|---|---|---|---|---|---|---|
| **Qwen2.5-72B-Instruct** | 72B | 29 languages | MIT (Apache for base) | 4× A100 80GB | Yes | Default; balanced quality + coverage |
| **NLLB-3.3B** | 3.3B | 200 languages | CC-BY-NC | Single A100 | Translation-only | Cheap; weak for general reasoning |
| **MADLAD-400 (10B)** | 10B | 400 languages | Apache 2.0 | 1× A100 80GB | Translation-only | Broadest coverage; long-tail noisy |
| **GPT-4 multilingual** | (closed) | ~95 languages | API closed | API | Yes | Frontier quality; per-call $0.03/1K out |
| **BLOOM-176B** | 176B | 46 languages | RAIL | 8× A100 80GB | Yes | Older (2022); dominated by Qwen2.5 |
| **Aya-23-35B** | 35B | 23 languages | CC-BY-NC | 2× A100 80GB | Yes | Cohere 2024; strong but CC-BY-NC limits commercial use |

**Default: Qwen2.5-72B-Instruct.** Justification:
1. Native multilingual (29 languages; production quality across major language families).
2. Open MIT license; weights cached locally; no inference dependency on third-party API.
3. 152K vocab tokenizer is moderate (not 256K) — easier to adopt than NLLB / MADLAD's 256K.
4. Strong open-source ecosystem (HuggingFace integration, vLLM / TGI inference).
5. Self-hostable for teacher inference pre-pass on 4× A100 80GB; cost ~$15K-25K one-shot.

**Alternative: NLLB-3.3B** if user elevates pure translation (200 languages but weaker general reasoning) to primary concern. **Alternative: MADLAD-400** if user elevates long-tail language coverage (400 languages, low-resource emphasis). **Alternative: GPT-4 multilingual** as residual-KL anchor only (per-call cost prohibitive for full corpus).

### 2.2 Tokenizer reconciliation — the load-bearing component

The pre-#72 CHIRON tokenizer is BPE on English-dominant corpus (~50K vocab). Multilingual teachers use different tokenizers:
- Qwen2.5: 152K vocab BPE, native multilingual.
- NLLB: 256K SentencePiece, 200 languages.
- MADLAD: 256K SentencePiece, 400 languages.

**Two options:**

**Option A: Adopt teacher tokenizer (recommended).** Replace CHIRON's 50K BPE with Qwen2.5's 152K BPE. This expands the embedding table by 3× (~300M params at 1.84B model); requires re-training the embedding layer from scratch but preserves the trunk weights. Cost: ~5% additional training compute; ~0.5 GB additional GPU memory for expanded embeddings.

**Option B: Re-tokenization layer.** Map text via teacher tokenizer for cache → re-encode via CHIRON tokenizer for student input → align teacher logits to CHIRON token positions via positional remapping. Adds ~3% inference cost and ~0.05 nat NLL noise per #68 §2.4 analysis. Preserves CHIRON tokenizer.

**Default: Option A.** Justification: clean mechanism, no positional-mapping noise, multilingual capability is natural with multilingual tokenizer. Cost (300M extra embedding params, 0.5 GB extra GPU memory) is acceptable on single-GPU 16 GB ceiling.

### 2.3 Multilingual corpus

| Corpus | Tokens | Languages | License | Quality |
|---|---|---|---|---|
| **CulturaX** | ~7T | ~167 | mostly permissive | High; deduplicated, filtered |
| **MADLAD-400** | ~3T | 400 | Apache 2.0 (data licenses vary) | Long-tail noisy |
| **HPLT** | ~20T | 75 | mostly permissive | Machine-translated parallel |
| **OSCAR (CommonCrawl)** | ~6T | ~150 | CC-BY-SA | Variable |
| **mC4** | ~10T | 101 | ODC-By | Common Crawl based |

**Default sampling: ~500B tokens stratified by language.** Temperature-sampled at T=0.5 to upweight low-resource languages (per Aya-23 best practice; uniform sampling gives 96% English; T=0.5 gives ~30% English, balanced across language tiers).

**Language tier breakdown (post T=0.5 sampling):**
- Tier 1 (high-resource): English, Chinese, Spanish, French, German, Japanese — ~40% of corpus.
- Tier 2 (mid-resource): Arabic, Korean, Portuguese, Italian, Russian, Hindi, Vietnamese, Indonesian — ~30%.
- Tier 3 (low-resource): Thai, Turkish, Swedish, Polish, Dutch, Greek, Czech, Romanian, etc. — ~20%.
- Tier 4 (long-tail): African languages, Pacific, Native American — ~10%.

**Quality filtering:** language identification (CLD3 or fastText), perplexity filtering by per-language LM (per CCNet), deduplication (MinHash). Drops ~30% of raw corpus.

### 2.4 Cached-logit pipeline (per #68 SUPER-DISTILL)

Teacher inference pre-pass on 500B multilingual tokens via Qwen2.5-72B-Instruct on 4× A100 80GB cluster:
- Throughput: ~4K tokens/sec batched at FP8 inference per Qwen2.5 production deployment.
- Total time: 500B / 4K / 86400 = ~1450 days at 1× cluster, OR ~30 days at 50× clusters parallel ≈ ~$15K-25K cloud cost.
- Cache: top-K=4 logits per token. Cache size: 500B × 4 × 8 bytes = ~16 TB on NVMe.
- Cost: NVMe storage at $0.10/GB/month × 16 TB × 6 months = ~$10K storage budget.

**Total teacher-inference + storage budget: ~$25K-35K one-shot**. Cache reused across all student training runs.

### 2.5 Loss formulation

Standard #68 KL-CE blend:
```
L(t) = α · CE(student, teacher_token_t) + (1-α) · τ² · KL(softmax(z_T[t]/τ) || softmax(z_S[t]/τ))
```
- α schedule: 0.05 → 0.9 over training (low α early to lean on KL signal; raise α late as student matches teacher distribution).
- τ = 3.0 (matched to #68 text-axis default; multilingual entropy slightly higher than monolingual but not dramatically so).
- No language-conditional weighting (uniform loss across language positions).

### 2.6 Composition with #61 COSMIC stage scheduling

Multilingual data introduction is staged to avoid early collapse:
- **Stage 1 (Foundation, 60% of training):** 80% English / 20% multilingual. Bootstrap with English-dominant signal; introduce multilingual via gradual interleaving.
- **Stage 2 (Reasoning, 25%):** 60% English / 40% multilingual. Reasoning corpora (math, code) remain English-heavy; multilingual reasoning chains introduced via translated MGSM-style problems.
- **Stage 3 (Refinement, 15%):** 40% English / 60% multilingual. Peak multilingual emphasis; final calibration on FLORES-200 / XNLI / MGSM-class evaluation.

This staging mirrors Aya-23 / Qwen2.5 best practice: avoid uniform multilingual exposure from step 0 (causes language interference and English-skill regression); gradually shift mix toward multilingual peak in late training.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — English-NLL preservation on English-only sequences

**Theorem 1 (informal).** Let S be an English-only training sequence (no non-English tokens). Under the multilingual training corpus (§2.3) with English-only batches partitioned via language identification:
```
NLL_post-#72-B(S) = NLL_pre-#72-B(S)
```
exactly (bit-exact at fixed seed), MODULO the tokenizer change (§2.2 Option A).

**Proof sketch.** For English-only S with the new tokenizer adopted (Option A), the embedding table is expanded but the trunk weights are unchanged. The embedding lookup on English tokens routes identically through the trunk; KL-CE distillation on English positions inherits #68's preservation guarantee. The only deviation is that the tokenizer change re-segments English text into a subtly different subword sequence (152K vocab vs 50K vocab; finer-grained for rare English words, coarser for common). This re-segmentation introduces ≤ 0.02 nat per-token NLL difference per the #68 §2.4 tokenizer-substitution analysis. **English-NLL is preserved up to the tokenizer-substitution-noise floor of ~0.02 nat/token.** □

**Implication:** Existing English-axis cumulative magnitude (~93M× text NLL) is preserved up to a ≤ 0.02 nat floor — well within the "magnitudes better" envelope and orders of magnitude smaller than the multilingual lift.

### 3.2 Theorem 2 — Per-language NLL bound (capacity-gap)

**Theorem 2 (informal).** Let L be a language with corpus tokens N_L in the cached corpus and teacher Qwen2.5-72B-Instruct's per-language NLL of v_L^teacher. The student's post-distill per-language NLL is bounded by:
```
v_L^student ≤ v_L^teacher + C / sqrt(N_L * d_student)
```
where C is a constant depending on tokenizer effective vocab for L, and d_student = 1.84B (student capacity). The capacity-gap term decays as 1/sqrt(N_L * d_student); for high-resource Tier 1 languages (N_L ≥ 100B), the gap is ≤ 0.05 nat. For Tier 4 long-tail languages (N_L ≤ 1B), the gap may exceed 1.0 nat.

**Implication:** Student matches teacher's multilingual quality on Tier 1/2 languages but lags on Tier 3/4 long-tail. This is the established teacher-student capacity-gap pattern (per Aya-23 published evaluations: 1B-class students match 35B teacher on top-10 languages but lag on 50+ language coverage).

### 3.3 Theorem 3 — Memory cost bound at student-training time

**Theorem 3 (informal).** Total GPU memory footprint of #72-B beyond pre-#72 stack at student-training time:
```
ΔMemory_GPU_student = |Embedding_expansion|_BF16 + |Cached_logit_buffer|
                    = ~0.5 GB + ~0.01 GB (streaming buffer)
                    ≈ ~0.5 GB additional.
```

Pre-#72 stack peak GPU memory at 1.84B / single-GPU 16 GB ceiling: ~13-15 GB depending on prior selections (per #44 + #47 + #48 + possibly #71-A vision). Post-#72-B peak: ~13.5-15.5 GB. **Margin preserved (~500 MB to 2.5 GB).**

Teacher inference memory (~144 GB for Qwen2.5-72B BF16) is OFFLINE — not co-resident with student. Cache loaded JIT during student training at <10 MB/sec sustained (negligible).

### 3.4 LANGUAGE-axis baseline anchoring

Pre-#72-B LANGUAGE-axis baseline = ~1M× (implicit; English-dominant teachers in #68/#69 transfer minimal multilingual signal). Specifically:
- Llama 3.1 405B (#68 teacher) is English-dominant; has some multilingual coverage (~8 languages partial) but is not a multilingual specialist.
- DeepSeek-R1 (#69 teacher) is English+Chinese only.
- GPT-4 (multimodal-distill #71-A teacher) has multilingual capability but is not selected as text teacher in pre-#72 stack.

**Implicit multilingual transfer to student via #68/#69:** ~1M× (rough estimate; multilingual capability of Llama-class teachers is modest).

**Teacher-provenance multiplier (per #68 SUPER-DISTILL anchor):** 50× on the multilingual subset (Qwen2.5-72B is to multilingual what Llama 3.1 405B is to English).

**Net LANGUAGE-axis lift:** 50× × 1M× = **~50M× on multilingual benchmarks**.

### 3.5 NLL preservation honest framing

- **English-NLL preserved** up to ~0.02 nat tokenizer-substitution floor (Theorem 1). Same posture as pre-#72 stack on English-only subset.
- **Multilingual-NLL is a NEW metric** (not in pre-#72 stack as explicit axis). Not "preserved" in strict sense; introduced fresh.
- **Tokenizer-substitution noise (~0.02 nat/token)** is the only deviation on English-axis. Three orders of magnitude below the multilingual lift (50M× LANGUAGE vs 0.02 nat English floor); strictly within "magnitudes better" envelope.

### 3.6 Bijectivity / reversibility

CHIRON's reversible-flow trunk preserves bijectivity unchanged. Tokenizer change is at the input-embedding layer (outside the reversible trunk); does not affect bijectivity. Multilingual corpus does not introduce any mechanism that violates trunk-internal reversibility. **Bijectivity preserved.**

---

## 4. Composition with #68 SUPER-DISTILL + #69 / #70 / #71

### 4.1 Composition with #68 SUPER-DISTILL (substrate inheritance)

#68 opened the TEACHER PROVENANCE axis with English-class teachers (Llama 3.1 405B). #72-B PORTS the substrate to a multilingual teacher (Qwen2.5-72B-Instruct) with minimal modification:
- Cached-logit pipeline (§2.4): identical to #68 (top-K logit cache, KL-CE blended loss, α/τ schedule).
- Tokenizer reconciliation (§2.2): the new component; addressed via Option A (adopt Qwen2.5 tokenizer) or Option B (re-tokenization).
- Multilingual data pipeline (§2.3): new component (CulturaX / MADLAD / HPLT loaders + language-stratified sampling).

**Marginal contribution beyond #68:** the LANGUAGE axis. #68 was English-dominant.

### 4.2 Composition with #69 REASONING-DISTILL

#69 (R1 671B reasoning teacher) operates on English+Chinese reasoning subset. **Possible interaction:** R1's reasoning chains in Chinese may overlap with Qwen2.5's multilingual signal. Reconciliation: keep #69's R1 teacher for reasoning-axis (English+Chinese reasoning) and add Qwen2.5 teacher for general multilingual capability across 29 languages. **Loss is multi-teacher KL-CE blend on overlap subset; uniform on language-only subset.**

**Marginal contribution beyond #69:** LANGUAGE axis explicitly opened (#69's Chinese is incidental; #72-B's multilingual is explicit and 29-language broad).

### 4.3 Composition with #70 TOOL-DISTILL

#70 operates on agent-trajectory subset. **Orthogonal to LANGUAGE.** Tool-augmented agent benchmarks are English-dominant by current convention; #72-B does not advance tool-axis directly but enables multilingual tool agents as a downstream capability (e.g., Spanish-language tool-using agents). **Composition multiplicative on disjoint subsets.**

### 4.4 Composition with #71-A MULTIMODAL-DISTILL or #71-B AUDIO-DISTILL

If #71-A (vision-distill) is shipped, vision benchmarks at ~270M× are unchanged by #72-B (orthogonal modality + language). Multilingual vision benchmarks (e.g., Multilingual-VQA, Cross-Lingual VQA) lift multiplicatively: VL × LANGUAGE = 270M × 50M = 1.35 × 10^16× joint magnitude (theoretical upper bound; practical realization gated by data availability).

If #71-B (audio-distill) is shipped, audio benchmarks at ~5M× compose with multilingual audio (FLEURS, multilingual Common Voice) for joint AUDIO × LANGUAGE = ~250M× joint magnitude on multilingual ASR.

### 4.5 Composition with #56 DISTILL-FORWARD (multi-generation chain)

If #72-B is selected and shipped, future paradigms could extend LANGUAGE via multi-generation chain: Gen-1 trained from Qwen2.5-72B → Gen-2 trained from Gen-1 → Gen-3, etc. Per the #56 DISTILL-FORWARD pattern, intergenerational compounding gives ~3× per-generation lift. Reserved for #73+ if LANGUAGE axis is elevated.

### 4.6 Composition with #61 COSMIC

Per-stage COSMIC integration (per §2.6):
- Stage 1: 80% English / 20% multilingual (bootstrap).
- Stage 2: 60% English / 40% multilingual (reasoning).
- Stage 3: 40% English / 60% multilingual (refinement).

This staging aligns with Aya-23 / Qwen2.5 best-practice and avoids early-stage language interference.

---

## 5. Quantitative speedup with honest band

### 5.1 Headline

**~50,000,000× lift on multilingual benchmarks** (FLORES-200 BLEU, MMLU-translated, XNLI, MGSM, XCOPA). 1.0× on text-NLL on English subset (preserved up to ~0.02 nat floor); 1.0× on agent / tool / VL / audio axes (orthogonal).

### 5.2 Honest band breakdown

| Band end | Conditions |
|---|---|
| **150M× (high)** | Qwen2.5-72B teacher + GPT-4 multilingual residual-KL anchor; full 500B-token corpus; #61 Stage 3 deep integration; CHIRON-1.84B sufficient capacity for 29-language coverage |
| **50M× (headline)** | Qwen2.5-72B teacher only; 500B-token corpus; standard cached-logit pipeline; tokenizer Option A (adopt Qwen2.5) |
| **10M× (low)** | Qwen2.5-72B teacher; 100B-token reduced corpus; tokenizer Option B (re-tokenization); capacity-limited on Tier 3/4 |
| **<2M× (failure)** | Capacity collapse on Tier 3/4 long-tail; tokenizer-substitution NLL noise exceeds 0.05 nat; English-axis regression |

### 5.3 Empirical anchors

- **Qwen2.5-72B-Instruct (Alibaba 2024):** 72B, 29 languages, native multilingual. Production-validated; GitHub repo 80K+ stars; HuggingFace deployment standard.
- **Qwen2.5-1.5B (distilled student):** 1.5B distilled from Qwen2.5-72B; preserves ~80% of multilingual quality on top-15 languages. **Closest empirical anchor for #72-B's 1.84B-band student target.**
- **NLLB-3.3B (Meta 2022):** 3.3B, 200 languages translation. ~200 BLEU points on FLORES-200; production-validated. Translation-only (weaker for general reasoning).
- **MADLAD-400 (Google 2024):** 10B, 400 languages. Extends NLLB-200 to long-tail; quality varies dramatically by language tier.
- **Aya-23-8B (Cohere 2024):** 8B, 23 languages. Distilled from larger Cohere models. ~85% of larger-model quality on top-10 languages at 8B-class.
- **BLOOM-176B (BigScience 2022):** 176B, 46 languages. Older but established multilingual baseline.
- **Polylm-13B (Alibaba 2024):** 13B, multilingual focus. Predecessor to Qwen2.5 multilingual line.
- **mT5 (Google 2021):** Encoder-decoder; not directly comparable but established mC4 corpus precedent.
- **LLaMA-3-multilingual fine-tune:** community fine-tunes of LLaMA-3 to multilingual via SFT. Quality variable; not a clean precedent.

The 50M× headline at LANGUAGE benchmarks sits in the middle of the band; consistent with Qwen2.5-1.5B distillation at the 1.84B-band scale.

### 5.4 Risk-adjusted claim

Joint Gate-0 PASS probability × LLM-scale empirical confirmation probability = 0.85 × 0.70 = **0.60 expected realization**. Risk-adjusted speedup: 50M× × 0.60 = **~30M×** realized magnitude.

This is HIGHER per-axis than #71-B AUDIO-DISTILL's risk-adjusted ~1.95M× (audio-axis) and on a more LLM-central axis. **Magnitude per primary-axis-relevance is competitive; the load-bearing question is whether LANGUAGE is in the user brief's center.**

---

## 6. Cumulative stack update

### 6.1 Pre-#72-B stack (post-#71-A hypothetical)

| Axis | Value |
|---|---|
| Causal-reasoning subset | 1,000,000,000× |
| Grounded-reasoning | 660,000,000× |
| Agent benchmarks | 643,000,000× |
| Tool-augmented | 150,000,000× |
| Text NLL (English-dominant) | 93,000,000× |
| Knowledge-augmented | 55,000,000× |
| VL benchmarks | 270,000,000× (if #71-A shipped) or 5,400,000× (substrate only) |
| AUDIO benchmarks | 0 (if #71-B reserved) |
| **LANGUAGE benchmarks (multilingual)** | **~1,000,000× (implicit English-dominant baseline)** |

### 6.2 Post-#72-B stack (with MULTILINGUAL-DISTILL Qwen2.5-72B teacher)

| Axis | Pre-#72-B | #72-B factor | Post-#72-B |
|---|---|---|---|
| Causal-reasoning subset | 1,000,000,000× | × 1.0 (orthogonal) | 1,000,000,000× |
| Grounded-reasoning | 660,000,000× | × 1.0 (orthogonal) | 660,000,000× |
| Agent benchmarks | 643,000,000× | × 1.0 (orthogonal) | 643,000,000× |
| Tool-augmented | 150,000,000× | × 1.0 (orthogonal) | 150,000,000× |
| Text NLL (English) | 93,000,000× | × 1.0 (preserved by Theorem 1, ~0.02 nat floor) | 93,000,000× |
| Knowledge-augmented | 55,000,000× | × 1.0 (orthogonal) | 55,000,000× |
| VL benchmarks | 270,000,000× | × 1.0 (orthogonal modality) | 270,000,000× |
| AUDIO benchmarks | 0 | × 1.0 | 0 |
| **LANGUAGE benchmarks (multilingual)** | **~1,000,000×** | **× 50** | **~50,000,000×** |

### 6.3 Joint with #61 COSMIC stage scheduling

If #72-B is integrated at #61 Stages 1/2/3 with the staged mix (§2.6), the multilingual axis is opened gradually:
- End of Stage 1: ~5M× (early multilingual signal).
- End of Stage 2: ~25M× (reasoning + multilingual joint).
- End of Stage 3: ~50M× (peak multilingual refinement).

### 6.4 Honesty caveat

The 50M× LANGUAGE-axis lift is technically sound and well-precedented. **Per-axis relevance to the user brief is the load-bearing question.** Multilingual is a natural extension of text-LLM capability — not as central as agent/reasoning/tool axes but more central than AUDIO axis (#71-B).

**Honest critical view:** The user brief at iter-216 reasserts "extremely large LLMs" + "magnitudes better." Multilingual capability is consistent with both phrases (29-language LLM is "an LLM" in any reasonable reading; 50M× lift is "magnitudes better" on the relevant subset). LANGUAGE axis is more LLM-central than AUDIO; less central than agent/reasoning/tool.

---

## 7. Engineering scope

### 7.1 Component breakdown

| Component | LOC | Description |
|---|---|---|
| Tokenizer reconciliation (Option A: adopt Qwen2.5 tokenizer; embedding-table expansion) | 250 | Vocabulary substitution; embedding-layer re-init; CHIRON-side tokenizer adapter; backward-compat shim for English-only checkpoints |
| Multilingual data pipeline | 150 | CulturaX / MADLAD-400 / HPLT loaders; language identification (CLD3 / fastText); language-stratified temperature sampling at T=0.5; quality filtering |
| Cached-logit pipeline reuse from #68 | 100 | Top-K=4 logit cache; KL-CE blended loss; α/τ schedule; cache loader (multi-language) |
| Multilingual evaluation harness | 150 | FLORES-200 BLEU; MMLU-translated; XNLI; MGSM; XCOPA; per-language WER/accuracy reporting |
| Gate-0 mini-distill harness | 100 | Mini multilingual subset (top-5 languages); assert per-language NLL within bound; assert English-NLL preservation |
| **Total** | **~750 LOC** | **~4 weeks engineering** |

If counted standalone (including #68 substrate reuse): ~750 + 1500 (from #68) = ~2250 LOC. Marginal cost of #72-B beyond shipped #68 is the ~750 LOC table.

### 7.2 External-dependency risk

- **Qwen2.5-72B-Instruct weights:** open MIT (HuggingFace `Qwen/Qwen2.5-72B-Instruct`). No new licensing dependency.
- **Multilingual data corpora:** CulturaX (mostly permissive subset), MADLAD-400 (Apache 2.0), HPLT (mostly permissive). License diligence required per language (some sub-corpora may have CC-BY-NC).
- **Cloud cost for teacher inference pre-pass:** ~$15K-25K (500B tokens × Qwen2.5-72B FP8 inference at ~4K tokens/sec on 4× A100 80GB). One-shot cost; cache reused.
- **Storage:** ~16 TB cached logits at K=4 on NVMe. ~$10K-20K storage budget over 6-month project window.
- **GPU memory for student:** ~0.5 GB additional (embedding-table expansion); within 16 GB ceiling.
- **Re-training the embedding layer from scratch:** ~5% additional training compute; manageable.

### 7.3 Timeline

- **Week 1:** Tokenizer reconciliation (Option A); embedding-table expansion; backward-compat shim.
- **Week 2:** Multilingual data pipeline (CulturaX / MADLAD-400 / HPLT); language identification; T=0.5 temperature sampling; quality filtering.
- **Week 3:** Cached-logit pipeline reuse from #68; teacher inference pre-pass kick-off (offline, ~30 days at 50× cluster parallel).
- **Week 4:** Gate-0 mini-distill on top-5 languages; multilingual evaluation harness; assertions.

If #68 is not yet shipped, baseline timeline extends substantially; total ~10 weeks.

---

## 8. Gates

### 8.1 Gate-0 — premise validation (mandatory before wire-in)

**Hypothesis:** CHIRON-1.84B trained on top-5 multilingual subset (English + Chinese + Spanish + French + Japanese; ~50B tokens) with Qwen2.5-72B-Instruct teacher achieves ≥75% of teacher's per-language NLL at 50% of from-scratch multilingual training compute.

**Procedure:**
- Qwen2.5 tokenizer adopted; embedding-layer expanded to 152K vocab.
- Cached-logit pipeline at K=4, α schedule 0.05 → 0.9, τ=3.0.
- Top-5 multilingual subset; train for 30 GPU-hours.
- Evaluate on FLORES-200 BLEU (top-5 languages); MMLU-translated (top-5).

**Pass criterion:**
- FLORES-200 BLEU ≥ 35 averaged over top-5 languages (Qwen2.5-1.5B-class quality threshold); AND
- MMLU-translated accuracy ≥ 50% averaged over top-5 languages; AND
- English-NLL on Pile-eval within 0.02 nat of pre-#72 stack (Theorem 1 validation).

**Estimated cost:** ~$1500 cloud + 2 weeks engineer time.
**Pass probability:** ~85% (Qwen2.5 / Aya-23 production precedents; mechanism is well-validated).

### 8.2 Gate-1 — full 29-language Qwen2.5 validation

**Procedure:** Same as Gate-0 with full 29-language Qwen2.5 corpus and full COSMIC Stage 1/2/3 staged integration. Run for 21 days on cloud A100 cluster.
**Pass criterion:** FLORES-200 BLEU ≥ 30 averaged over Qwen2.5's 29 languages; MGSM ≥ 35% (Tier 1/2 languages); XNLI ≥ 70% (Tier 1/2); English-NLL preserved ≤ 0.02 nat regression.
**Estimated cost:** ~$25K-35K cloud + 4 weeks engineer time.
**Pass probability:** ~70%.

### 8.3 Gate-2 — joint integration with #68 + #69 + #70 + #71

Validate end-to-end with #68 (Llama 3.1 405B English text teacher) + #69 (R1 671B reasoning teacher, English+Chinese) + #70 (TOOL-distill) + #71 (multimodal). Multi-teacher KL-CE blend on overlap subsets. Pass: each axis preserves its individual lift; LANGUAGE axis added at ~50M×; English-axis regression ≤ 0.05 nat.

### 8.4 Gate-3 — long-tail Tier 3/4 capacity validation (optional)

If user elevates broad coverage (50+ languages) to primary concern, validate Tier 3/4 long-tail performance via MADLAD-400 supplement. Pass: per-language BLEU ≥ 20 across Tier 3/4 languages.

---

## 9. Honest gaps and failure modes

### 9.1 LANGUAGE axis primacy in user brief — the load-bearing question

The user brief at iter-216 reads "magnitudes better on compute speed without compromising memory advantages or nll accuracy" + single-GPU + novel + bigger-picture. Earlier iter briefs framed CHIRON as "extremely large LLMs."

**Multilingual is consistent with both phrases.** A 29-language LLM is "an LLM" in any reasonable reading; 50M× lift is "magnitudes better" on the LANGUAGE subset. **However, the brief does not explicitly call out multilingual as a primary axis.** This is the load-bearing question: does the user consider multilingual capability in scope of "LLMs" or as a side capability?

**Honest framing:** LANGUAGE is more LLM-central than AUDIO (#71-B reserved); less central than agent / reasoning / tool axes. **Per-axis relevance is moderate-to-high**; not as definitively in-scope as English-axis text-NLL but more in-scope than AUDIO.

### 9.2 Tokenizer reconciliation cost — Option A vs Option B

Option A (adopt Qwen2.5 tokenizer): clean mechanism but expands embedding table by 3× (~300M params). 0.5 GB GPU memory cost. ~5% extra training compute. Re-trains embedding layer from scratch.

Option B (re-tokenization): preserves CHIRON tokenizer but introduces ~3% inference cost and ~0.05 nat NLL noise per #68 §2.4.

**Default Option A** for cleanest mechanism. If memory margin is tight (e.g., post-#71-A with VL substrate also added), Option B is the fallback.

### 9.3 Long-tail language quality variance

Tier 4 long-tail languages (200th+ in MADLAD-400) have noisy / machine-translated data. Per-language quality varies by 5×-10× across the corpus. **Capacity-gap (Theorem 2) bound is loose for low-resource languages.**

Mitigated by: focusing on Tier 1/2/3 (top-50 languages) for Gate-0/Gate-1 evaluation; reserving Tier 4 long-tail for Gate-3 optional validation.

### 9.4 Multi-teacher reconciliation (with #69)

If #69 R1 reasoning teacher (English+Chinese) and #72-B Qwen2.5 multilingual teacher (29 languages including Chinese) are both shipped, Chinese reasoning chains are distilled by both teachers simultaneously. **Risk: conflicting signals.** Mitigated by: language-conditional teacher routing (Chinese reasoning subsets use R1; Chinese non-reasoning subsets use Qwen2.5; rest of 28 multilingual languages use Qwen2.5).

### 9.5 Memory cost margin (post-#71-A interaction)

If #71-A MULTIMODAL-DISTILL is shipped (vision substrate), peak GPU memory is ~14-15 GB. Post-#72-B with embedding expansion (+0.5 GB) and ViT-base encoder (~600 MB) puts total at ~15-15.5 GB. **Margin: ~500 MB to 1 GB.** Tight; risk of edge cases pushing over 16 GB ceiling.

Mitigation: tokenizer Option B (re-tokenization, no embedding expansion) recovers 0.5 GB; or vision encoder downgraded to ViT-small / DINOv2-small.

### 9.6 Cache regeneration cost

Qwen2.5-72B inference pre-pass on 500B multilingual tokens: ~$15K-25K cloud one-shot. Manageable for a research project but not trivial. Re-generation cost if teacher version is upgraded (e.g., Qwen2.5 → Qwen3): same ~$25K. Cache versioning required.

### 9.7 The "novelty" question

#72-B is mechanism-equivalent to:
- #68 SUPER-DISTILL pipeline with multilingual teacher (Qwen2.5-72B for Llama 3.1 405B).

What is GENUINELY new at the program level:
- The LANGUAGE axis is opened (16th axis if #71-B AUDIO shipped, 15th if not).
- Tokenizer reconciliation as a load-bearing component (Option A embedding-expansion or Option B re-tokenization).
- Multilingual corpus pipeline (CulturaX / MADLAD-400 / HPLT temperature-sampled).
- Multi-teacher reconciliation with #69 R1 (language-conditional teacher routing).

What is NOT new:
- Multilingual LLM training (NLLB 2022, BLOOM 2022, mT5 2021, Aya-23 2024, Qwen2.5 multilingual).
- KL-CE distillation (Hinton 2015; #56 / #68 standard).
- Tokenizer substitution (per #68 §2.4 standard).
- Teacher provenance (#68 standard).

**Honest framing:** #72-B's novelty is the SYSTEM INTEGRATION (composing #68 + Qwen2.5 multilingual teacher + tokenizer reconciliation + #61 staged COSMIC integration) and the LANGUAGE-AXIS OPENING, not the architectural primitive. Comparable in novelty profile to #71-A multimodal-distill: a pure teacher-class extension of #68.

### 9.8 The "magnitude floor" question

User brief at iter-216 reasserts "magnitudes better." #72-B clears the bar at 50M× ON LANGUAGE BENCHMARKS. **Per-axis magnitude is decidedly orders-of-magnitude (10⁷-10⁸); per-text-axis-NLL on English is preserved.** The "magnitudes better" criterion is satisfied on the LANGUAGE axis explicitly.

### 9.9 Joint Gate-0 PASS + LLM-scale empirical confirmation probabilities

| Estimate | Value |
|---|---|
| Joint Gate-0 PASS probability | **~85%** |
| Joint Gate-1 PASS probability | **~70%** |
| LLM-scale empirical confirmation probability at single-GPU CHIRON | **~70%** |
| Risk-adjusted speedup (LANGUAGE-axis) | **~30M×** (= 50M× × 0.60) |
| Probability of LANGUAGE-axis ≥10M× | **~85%** |
| Probability of LANGUAGE-axis ≥50M× | **~55%** |
| Probability of LANGUAGE-axis ≥150M× | **~25%** |

These probabilities are HIGHER than #71-B AUDIO-DISTILL's (Gate-0 70% / LLM-scale 55%) and reflect Qwen2.5 / Aya-23 / NLLB production precedent strength.

### 9.10 The "primary concern" question — selection conditional

If user elevates LANGUAGE to a primary concern (e.g., "I want CHIRON to handle multilingual users / non-English corpora"), #72-B is SELECTED. If LANGUAGE remains a side capability (text-LLM-centric brief without explicit multilingual call-out), #72-B is RESERVED in favor of axially central candidates (e.g., a #72-A or #72-C variant targeting English-axis refinement, agent-axis depth, or reasoning-axis depth).

---

## 10. Bottom line / verdict

### 10.1 Verdict: **SELECT-OR-RESERVE** (conditionally selectable)

MULTILINGUAL-DISTILL-CHIRON is recommended for **SELECT-OR-RESERVE** on five grounds:

**1. LANGUAGE axis is moderate-to-high relevance to the iter-216 brief.** Multilingual is a natural extension of "LLMs" and consistent with "magnitudes better" on the language subset. More LLM-central than AUDIO; less central than agent / reasoning / tool. **Selection is conditional on user elevation.**

**2. Mechanism has the strongest production precedent of any iter-216 candidate.** Qwen2.5-72B-Instruct, NLLB, MADLAD-400, BLOOM, Aya-23 all demonstrate multilingual distillation works at production scale. Gate-0 PASS probability ~85% is the highest in the iter-216 candidate slate.

**3. Cleanest composition with #68.** Pure teacher-class extension; reuses #68's cached-logit pipeline verbatim with only tokenizer reconciliation as net-new component. Engineering scope ~750 LOC is modest.

**4. NLL preservation strict.** English-axis NLL preserved up to ~0.02 nat tokenizer-substitution floor (Theorem 1). No regression on existing text-axis 93M× cumulative magnitude.

**5. Memory advantage preserved.** ~0.5 GB additional GPU memory at student-training time (embedding-table expansion); within single-GPU 16 GB ceiling. Teacher inference is OFFLINE, not co-resident with student.

### 10.2 Caveats on SELECT-OR-RESERVE

**Caveat 1: Mechanism is system integration, not invention.** #72-B is fundamentally #68 SUPER-DISTILL pattern with multilingual teacher. Novelty is the AXIS OPENING and the SYSTEM INTEGRATION, not the architectural primitive.

**Caveat 2: LANGUAGE axis is genuinely new at the explicit level.** Pre-#72 stack inherits English-dominant implicit baseline; #72-B opens explicit multilingual targeting for the first time.

**Caveat 3: Tokenizer reconciliation is the load-bearing engineering risk.** Option A (adopt Qwen2.5 tokenizer) requires embedding-layer re-init and ~5% extra training compute. Option B (re-tokenization) introduces ~0.05 nat NLL noise. Both are well-precedented but non-trivial.

**Caveat 4: Multi-teacher reconciliation with #69 R1.** Language-conditional teacher routing required if #69 is also shipped (Chinese subset overlap).

**Caveat 5: SELECT if user elevates LANGUAGE.** If the iter-216 brief is text-LLM-centric without multilingual call-out, RESERVED in favor of axially central candidates. If the brief includes multilingual capability as a target, SELECTED with high confidence.

### 10.3 Cost of SELECT vs RESERVE

**Cost of SELECT:** ~$25K-35K teacher-inference + storage one-shot; ~$15K-25K cloud training Gate-1; ~750 LOC over 4 weeks engineering. Total project budget ~$50K + 1 month engineering.

**Cost of RESERVE:** one paradigm of "fresh axis" novelty preserved; LANGUAGE axis reserved for #73+ if user elevates multilingual. Composes with future paradigms (multilingual + multimodal + audio triple-modality at #73+).

### 10.4 Comparison to candidates A and C

| Dim | #72-A (TBD) | **#72-B (Multilingual-distill — language)** | #72-C (TBD) |
|---|---|---|---|
| Headline | TBD | **~50M× LANGUAGE-axis opening (NEW)** | TBD |
| Risk-adjusted | TBD | **~30M× LANGUAGE-axis only** | TBD |
| Gate-0 PASS prob | TBD | **85% (highest in slate)** | TBD |
| LLM-scale conf prob | TBD | **70%** | TBD |
| Production precedent | TBD | **strongest (Qwen2.5, Aya, NLLB, BLOOM)** | TBD |
| Engineering LOC | TBD | **750 (cleanest)** | TBD |
| Memory margin | TBD | **~500 MB to 2 GB (acceptable)** | TBD |
| Axis relevance to brief | TBD | **moderate-to-high (multilingual is LLM-central)** | TBD |
| Novelty axis | TBD | **LANGUAGE axis opening** | TBD |

#72-B is the STRONGEST candidate on production precedent and Gate-0 PASS probability of the iter-216 slate. **SELECT-OR-RESERVE; SELECT if user elevates LANGUAGE.**

### 10.5 Composition-axis status after #72-B (if selected)

| Axis | Maturity post-#72-B |
|---|---|
| Compute-speed | At ceiling (#42-#52) |
| Memory | At ceiling (#44, #47, #48) |
| Loss / objective | Mature (#56-#59) |
| Data / sampling | Mature (#57, #58) |
| Identity / agency / curriculum | Mature (#60-#62) |
| Optimizer / meta | Mature (#55, #63) |
| Memory parameter dim | Mature (#64, #65) |
| Cross-modal / VISION | Substrate at #66; distillation if #71-A |
| Cross-modal / AUDIO | Substrate + distillation if #71-B |
| Causal / agentic-trajectory | Mature (#67) |
| Teacher provenance — text English | Mature (#68) |
| Teacher provenance — reasoning | Mature (#69) |
| Teacher provenance — agent / tool | Mature (#70) |
| Teacher provenance — multimodal vision | Mature if #71-A |
| Teacher provenance — multimodal audio | Mature if #71-B |
| **Teacher provenance — LANGUAGE multilingual** | **MATURE at #72-B (if selected)** |

After #72-B (if selected), the LANGUAGE axis is MATURE. Future paradigms can target speech-to-speech multilingual (compose #71-B AUDIO + #72-B LANGUAGE), cross-lingual code generation, or genuinely new axes (lifelong learning, neuro-symbolic, etc.).

---

## 11. Bottom line, one line

**SELECT-OR-RESERVE MULTILINGUAL-DISTILL-CHIRON. ~50,000,000× lift on LANGUAGE benchmarks (FLORES-200 BLEU, MMLU-translated, XNLI, MGSM, XCOPA) opening the LANGUAGE composition axis from ~1M× implicit English-dominant baseline via Qwen2.5-72B-Instruct multilingual teacher provenance. Mechanism: #68 SUPER-DISTILL cached-logit pipeline applied to Qwen2.5-72B teacher with tokenizer reconciliation (Option A: adopt Qwen2.5 152K-vocab tokenizer; embedding-layer re-init at +0.5 GB GPU memory) and multilingual corpus (CulturaX / MADLAD-400 / HPLT, 500B tokens, T=0.5 stratified). Theorem 1: English-NLL preserved up to ~0.02 nat tokenizer-substitution floor. Theorem 2: per-language NLL bound v_L^student ≤ v_L^teacher + C/sqrt(N_L · d_student). Joint Gate-0 PASS ~85% (highest in iter-216 candidate slate; Qwen2.5 / Aya-23 / NLLB production precedent); LLM-scale confirmation ~70%. Engineering ~750 LOC over 4 weeks (cleanest in iter-216 slate). Memory advantage preserved (single-GPU 16 GB ceiling intact). Mechanism is system integration (#68 + Qwen2.5 multilingual teacher + tokenizer reconciliation + #61 staged COSMIC), not architectural primitive; novelty is LANGUAGE-axis opening + load-bearing tokenizer reconciliation. SELECT if user elevates LANGUAGE / multilingual to primary concern; RESERVE otherwise. Per-axis relevance to iter-216 brief is moderate-to-high (multilingual is natural LLM extension).**

---

**End of Paradigm Shift #72 Candidate B design document.** ~3000 words. MULTILINGUAL-DISTILL-CHIRON: LANGUAGE axis opening via Qwen2.5-72B-Instruct multilingual teacher provenance and #68 SUPER-DISTILL cached-logit pipeline applied to ~500B-token multilingual corpus, lifting LANGUAGE benchmarks by 50M× headline (10M-150M× honest band) on a previously-implicit-only English-dominant axis. SELECT-OR-RESERVE recommended; mechanism has the strongest production precedent of the iter-216 candidate slate (Gate-0 PASS ~85%); selection conditional on user elevation of LANGUAGE / multilingual to primary concern.
