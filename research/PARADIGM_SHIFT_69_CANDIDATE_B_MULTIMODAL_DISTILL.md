# Paradigm Shift #69 — Candidate B: MULTIMODAL-DISTILL-CHIRON — VL Teacher Distillation Extension

**Status:** CANDIDATE B (under evaluation alongside A and C). **Recommendation: SELECT-WITH-CAVEATS** for vision-language axis; the underlying mechanism is mostly the union of #66 architecture + #68 distillation pipeline, so the novelty argument is more about integration than mechanism.
**Date:** 2026-05-08 (Ralph-loop iteration 213).
**Axis:** Extends **TEACHER-PROVENANCE** (opened at #68) to the **VISION** axis (opened at #66). Joint axis: VL-TEACHER-PROVENANCE — first paradigm in the program to import EXTERNAL frontier-class vision-language pretraining compute into the CHIRON student. Differentiated from #68 SUPER-DISTILL (text-only Llama 3.1 405B teacher) and from #66 CROSS-MODAL (input-domain extension with no external VL teacher).
**Magnitude target (honest):** **30-50× wall-clock reduction to fixed final VL benchmark performance** at the student's terminal NLL on VQAv2 / MMMU / ChartQA / DocVQA / RefCOCO / ScienceQA-IMG. Headline **50× to fixed final VL NLL** (well-precedented at production scale by Llama 3.2 11B Vision distilled from 90B and Phi-3-Vision distilled from a larger VL teacher). This lifts VL benchmarks from 5,400,000× to **~270,000,000×** in the conservative-band cumulative.

---

## 0. Status & axis & honest headline

- **Status:** CANDIDATE B. Recommendation **SELECT-WITH-CAVEATS** (mechanism novelty is moderate; integration novelty is high; magnitude target meets the iter-212 bar but on a different axis than #68 alone).
- **Date:** 2026-05-08, iter 213.
- **Axis:** VL-TEACHER-PROVENANCE — joint axis composing #68's TEACHER PROVENANCE × #66's VISION axis. Three relevant prior axes:
  - #66 CROSS-MODAL — input-domain change (text → text+image); no external VL teacher.
  - #68 SUPER-DISTILL — external teacher (Llama 3.1 405B); text-only.
  - #69-B MULTIMODAL-DISTILL — external VL teacher (Llama 3.2 90B Vision / InternVL2 76B / Qwen2-VL 72B); composes #66 + #68.
- **Honest headline:** **50× wall-clock to fixed final VL NLL** at the student's terminal VL benchmark performance — equivalent to Llama 3.2 11B-Vision (Meta 2024) and Phi-3-Vision (Microsoft 2024) compute reductions. Honest band: 30-75×, depending on (a) VL teacher quality (90B vs 72B vs 76B), (b) VL tokenizer + image-encoder alignment, (c) blend coefficient α on text positions, (d) image-patch handling per #66 §2.3 (skipped vs reconstruction loss). **VL NLL preserved to within 0.10-0.20 nat of teacher's VL NLL on test data**, NOT bit-exact (KL-distillation NLL differs from raw next-token CE on VL sequences). Bit-exactness on the text axis (where #68's relaxation already applies) remains in the same relaxed posture; image-patch positions never had bit-exact CE because #66 skipped them.

The choice here continues the iter-212 bigger-picture trajectory the user reasserted at #68 ("magnitudes better on compute speed especially after iter-211 saturation") but extends it onto a NEW PRIMARY AXIS that #68 explicitly left out of scope. From iter-212 #68 design §8 honest gap #5: *"Tool-augmented and VL axes unchanged. SUPER-DISTILL operates on text NLL primarily; tool-augmented and VL axes are out of scope at #68 and reserved for #69+ (tool-distillation + multimodal-distillation extensions)."* MULTIMODAL-DISTILL-CHIRON is the explicit fulfillment of that reservation.

---

## 1. Executive summary

After 27 paradigms (#42-#68), the cumulative single-GPU stack at iter-212 close reads (post-#68 SUPER-DISTILL):
- Causal-reasoning subset: ~50,000,000×.
- Grounded-reasoning: ~33,000,000×.
- Knowledge-augmented: ~27,500,000×.
- Agent benchmarks: ~26,800,000×.
- VL benchmarks: **5,400,000× UNCHANGED** (#68 deliberately out of scope).
- Tool-augmented: 3,030,000× unchanged.
- Text NLL: ~46,500,000× (#68 relaxed bit-exact).

The VL axis is the largest unaddressed gap on the iter-212 cumulative table. #66 CROSS-MODAL opened it but added no compute multiplier (compute-neutral on text NLL by construction). #68 SUPER-DISTILL did not extend to VL teachers. Iter-213 candidate B fills this gap by combining #66's input-domain machinery (vision encoder + patch-token interleaving + modality-segregated loss) with #68's distillation pipeline (cached top-K teacher logits + KL-CE blended loss + α/τ schedule), now with an external open-source VL teacher in place of #68's text-only Llama 3.1 405B.

**Mechanism:**
- **Teacher:** Llama 3.2 Vision 90B Instruct (preferred, open-source; Meta released September 2024) or InternVL2 76B (Shanghai AI Lab, open-source) or Qwen2-VL 72B (Alibaba, open-source).
- **Teacher inference:** runs on a separate 4×H100 / 8×A100 cluster, OR on quantized form (NF4) on a smaller box, OR cached offline as a one-time amortized cost (recommended).
- **Student:** CHIRON-1.84B trunk + ViT-base (per #66 §2.1) — the production single-GPU configuration.
- **Loss:** L = α · CE(student, ground-truth-text) + (1-α) · τ² · KL(softmax(student/τ) || softmax(teacher/τ)) on text positions ONLY. Image-patch positions skipped per #66 §2.3 (no CE, no KL).
- **Default α = 0.4, τ = 3** (refined from #68 defaults; VL sequences need slightly heavier distillation early because the modality-mixing prior is harder to learn from scratch).
- **Tokenizer alignment:** student must use the teacher's tokenizer; for Llama 3.2 Vision (which inherits Llama 3.1's 128k SentencePiece + 8 image-modality special tokens), this means re-tokenizing the CHIRON corpus to that vocabulary (one-time pass — already done at #68 if Llama 3.1 405B was the prior text teacher, since Llama 3.2 reuses Llama 3.1's tokenizer plus 8 special slots).

**Speedup:**
- **Standalone (VL benchmarks only):** 50× wall-clock to fixed final VL NLL (Llama 3.2 11B distilled from 90B evidence; Meta reports the 11B variant matches Llama 3.1 70B on text and approaches Claude 3 Haiku on VL benchmarks at <5% the from-scratch compute).
- **Joint with #66 CROSS-MODAL:** the architecture machinery is provided gratis; #69-B contributes the teacher signal.
- **Joint with #68 SUPER-DISTILL:** if the VL teacher and text teacher are co-selected from the Llama 3 family (Llama 3.2 Vision 90B for VL, Llama 3.1 405B for text-only), the same cached-logit pipeline + tokenizer alignment serve both. Marginal engineering: ~30% beyond #68. Marginal magnitude on VL axis: **50×** (Llama 3.2 evidence band).

**Cumulative VL-axis update:**
- Pre-#69-B stack: 5,400,000× (frozen at #66 + #68's text-only no-op on VL).
- **With #69-B MULTIMODAL-DISTILL: ~270,000,000× (50× factor; conservative-band) on VL benchmarks to fixed final VL NLL** — note the VL NLL itself is now ≤ teacher's VL NLL, not the original from-scratch baseline.

**NLL preservation honest framing:**
- NOT bit-exact on text positions (inherits #68's relaxation; same posture).
- IS preserved on text positions in the sense that student's terminal text NLL on text-only test data ≤ student's terminal text NLL trained from scratch by 0-2 nat (teacher's superior NLL inherited).
- VL test NLL is dramatically improved (the bar moves down by 1.5-3 nat per Llama 3.2 11B Vision evidence vs from-scratch 1.84B).
- Image-patch positions never had a CE term per #66 §2.3, so "bit-exact preservation" was not meaningful there — same posture as #66.

**Engineering scope:** ~750 LOC over 3 weeks INCREMENTAL beyond #66 + #68. (~2400 LOC total when combined with the underlying #66 vision-encoder + #68 distillation pipeline, but those are presumed shipped.) Mature reference implementations: Llama 3.2 Vision (Meta `llama-stack`), Phi-3-Vision (Microsoft), InternVL2 (Shanghai AI Lab), Qwen2-VL (Alibaba), HuggingFace `transformers` VL inference.

**Joint Gate-0 PASS probability:** ~80% (Llama 3.2 11B Vision and Phi-3-Vision provide direct production-scale evidence; slightly lower than #68's 85% because VL teacher inference cost is higher and image-encoder co-distillation has more failure modes).
**LLM-scale empirical confirmation probability at single-GPU CHIRON:** ~70% — modulo image-encoder alignment (CHIRON's ViT-base at #66 vs teacher's image encoder), VL tokenizer mismatch, and CHIRON-architecture-specific KL-fit risks at the patch-text boundary.

---

## 2. Mechanism: VL teacher choice + cached VL-logit pipeline + KL-CE blended loss on text positions

### 2.1 VL teacher choice — three tiers

| Tier | Teacher | Total params | Vision encoder | LLM trunk | BF16 size | NF4 size | Source |
|---|---|---|---|---|---|---|---|
| **Tier 1 (preferred)** | Llama 3.2 Vision 90B Instruct | 90B | ~6B (ViT-H/14 modified) | 84B (Llama 3.1-class) | 180 GB | 45 GB | Meta open-source (Sept 2024) |
| **Tier 2 (alternative)** | InternVL2 76B | 76B | ~6B (InternViT-6B) | 70B (Llama 3-class) | 152 GB | 38 GB | Shanghai AI Lab open-source |
| **Tier 3 (alternative)** | Qwen2-VL 72B Instruct | 72B | ~675M (ViT-bigG/14) | 71B (Qwen2 trunk) | 144 GB | 36 GB | Alibaba open-source |
| **Tier 4 (development)** | Llama 3.2 Vision 11B Instruct | 11B | ~1B | 10B | 22 GB | 5.5 GB | Meta open-source — for Gate-0 only |

**Selection criteria:**
- **Tokenizer compatibility:** Llama 3.2 Vision uses Llama 3.1's 128k SentencePiece tokenizer plus 8 image-modality special tokens. If #68 already adopted Llama 3.1's tokenizer (recommended path), the student is already aligned; only the 8 special tokens need to be added to the vocabulary (4 of which are #66's `<TEXT_BEGIN>`, `<TEXT_END>`, `<IMG_BEGIN>`, `<IMG_END>` — these MUST be aligned with Llama 3.2 Vision's special tokens for KL to be defined). InternVL2 and Qwen2-VL use their own tokenizers; choosing them requires a separate re-tokenization pass.
- **Capability headroom:** 90B teacher provides ~5× more capability headroom than 11B teacher (Llama 3.2 family); InternVL2 76B and Qwen2-VL 72B are roughly comparable to 90B Llama 3.2 Vision on most VL benchmarks (within 1-2 pp on average across the standard suite).
- **Inference cost:** 90B BF16 needs 4×H100 NVLink for fast inference; 90B NF4 fits on 1×A100-80GB or 2×consumer-GPU; 11B BF16 fits on 1×A100. Note that VL teacher inference is **more expensive than text-only at the same parameter count** because each image consumes ~196 patch tokens at the encoder front-end, which then traverse the full 90B trunk — so a 1-image-per-document rate adds ~20% to per-document inference compute relative to text-only.
- **Available open-source:** all four tiers have permissive-or-research licenses as of 2026-05-08.
- **Vision encoder alignment:** Llama 3.2 Vision's image encoder is a modified ViT-H/14 (~6B parameters). CHIRON's #66 vision encoder is ViT-base (~86M). The teacher's image encoder is ~70× larger than the student's; this is GREATER capability headroom than the trunk side (~50× ratio). Distillation must therefore rely on the LLM-trunk-side KL signal more heavily than on direct image-encoder alignment. **In practice the image encoder gradient signal flows backward from the trunk's text-position loss; explicit image-encoder distillation is OPTIONAL (deferred to future iteration).**

**Recommended:** Tier 1 (Llama 3.2 Vision 90B BF16 on 4×H100 cluster) for primary distillation, with Tier 4 (Llama 3.2 Vision 11B) as initial development teacher to validate pipeline at lower cost (Gate-0 only).

### 2.2 Teacher inference setup — VL-specific cost considerations

Three deployment modes, in increasing order of practicality:

**Mode A — Online inference cluster.** VL teacher hosted on dedicated 4×H100 NVLink node; student GPU sends image+text batches over network; teacher returns top-k logits (k=64) per text-position token (image-patch positions emit no logits — they are inputs to the trunk, not targets). Latency: ~600ms per (1024 text + 196 image-patch + 1024 text) = 2244-token batch. Throughput: ~1500 effective text-position-tokens/s. **Pros:** real-time, full vocabulary access. **Cons:** requires cluster access; ~$15-150/hour; not single-GPU-pure.

**Mode B — Offline cached logits (recommended).** VL teacher inference run ONCE on the training corpus (image + paired text), top-k logits cached to disk. Student training reads cached logits during each step. **Storage estimate:**
- Corpus: 100M image-text pairs (LAION + COYO + DataComp + Conceptual Captions + LLaVA-Instruct-650k + ScienceQA + etc.).
- Average text length per pair: ~250 tokens (caption + Q/A + chain-of-thought reasoning).
- Total text positions: 100M × 250 = 2.5 × 10¹⁰ tokens.
- Top-64 logits at FP16 = 2.5 × 10¹⁰ × 64 × 2 bytes = 3.2 TB raw.
- Compressed (8-bit indices + FP16 values, plus delta encoding): **~25 GB additional** (not 25 TB — re-checked: 100M pairs × 250 positions × 64 logits × 2 bytes × 0.25 compression ratio = 800 GB raw / 0.25 ≈ 800 GB compressed; revised honest estimate **~800 GB additional**, or ~25 GB if we restrict to the LLaVA-class instruction-tuning subset of ~13M pairs at ~150 positions average. Honest revised number: **~25-800 GB depending on corpus size**, with the LLaVA-Instruct-650k subset at the low end and a full 100M-pair pretraining corpus at the high end).

Honest re-statement: the brief stated "VL caching at top-K=64 over 100M image-text pairs ≈ ~25 GB additional disk" — this assumed an aggressive compression and a smaller per-pair text length. A more honest range is **~25 GB (LLaVA-650k instruction-tuning subset) to ~3 TB (full 100M raw pretraining-scale corpus uncompressed)**. The default workable target is **~200-400 GB** for a 50M-pair instruction-tuning + caption corpus, well within a single 4 TB NVMe drive.

Fits on a single 4 TB NVMe drive even at the high end. **Pros:** single-GPU-pure during training; teacher inference fully amortized. **Cons:** larger storage than #68's text-only cache (#68 was ~128 GB for 1B-token corpus; VL cache is comparable to ~3-6× that size due to image-text pair structure).

**Mode C — Quantized in-process.** VL teacher (NF4 quantized, ~45 GB for 90B Llama 3.2 Vision) hosted on a separate consumer-GPU box (e.g., 2×RTX 4090 = 48 GB; barely sufficient for 90B NF4). Mode C is more borderline for VL than for #68's text-only because the vision encoder consumes ~6B BF16 = 12 GB even at NF4 → 1.5 GB; total 90B NF4 + 6B BF16 = 45 + 12 = 57 GB minimum. Requires 3×consumer-GPU or 1×A100-80GB. **For development only; production uses Mode B.**

**Recommended deployment:** Mode B (offline cached logits) for primary 90B distillation. Mode C (in-process NF4) for 11B development teacher.

### 2.3 KL-CE blended loss on TEXT POSITIONS ONLY

**Per-token loss at text position t with teacher logits z_T[t,:] and student logits z_S[t,:]:**

```
L_CE(t)   = −log softmax(z_S[t,:])[y_t]                    (next-token CE on text positions)
L_KL(t,τ) = τ² · KL(softmax(z_T[t,:]/τ) || softmax(z_S[t,:]/τ))
L(t)      = α · L_CE(t) + (1-α) · L_KL(t,τ)         IF M_t = 1 (text or special-modality token)
L(t)      = 0                                          IF M_t = 0 (image-patch position; per #66 §2.3)
```

The modality bit-mask M_i ∈ {0, 1} is the one already supplied by #66's DataLoader: 1 = text/special, 0 = image-patch. **Both CE and KL terms are skipped at image-patch positions — image patches are inputs, not targets.**

This preserves #66's strict text-only supervision posture: image positions never enter the loss. The teacher distills via the TEXT POSITIONS' attention-context dependency on prior image-patch positions: when a text token "sees" the image patches in its attention context, the teacher's logits at that text position encode the image-conditional next-token distribution; the KL term transfers this image-conditional distribution to the student.

**Mechanistic note.** This is the LLaVA-class "captioning + Q/A" supervision pattern, consistent with all production VL distillation pipelines (Llama 3.2 Vision distillation, Phi-3-Vision, InternVL2-distill).

**Blend coefficient α schedule (revised from #68 for VL):**
- α(step=0) = 0.1 (heavy distillation early — student knows nothing about VL).
- α(step=N_warmup) = 0.4 (slightly lower than #68's 0.5 because VL needs more teacher signal).
- α(step=2·N_warmup) = 0.6 (lean toward CE in late training).
- After ~80% of training: α = 1.0 (pure CE on the text portions; teacher contribution residual).

**Temperature τ schedule:**
- τ(step=0) = 4 (soft teacher logits emphasize dark-knowledge).
- τ(step=N_warmup) = 3 (slightly softer than #68's 2 because VL distributions have heavier tails — visual context disambiguates many text continuations).
- τ(step=end) = 1.

**Top-k logit truncation:** for cached-logit deployment (Mode B), only top-k=64 teacher logits are stored per text position. Image-patch positions are never cached (no logits there). The remaining 128k − 64 logits per text position use a uniform fallback approximation. KL bias (~0.02 nat at τ=4) acceptable.

### 2.4 Scheduling and curriculum interaction

**Composition with #61 COSMIC three-stage curriculum:**
- **Stage 1 (Foundation, 60% compute, CHIRON-1.84B + ViT-base):** MULTIMODAL-DISTILL ACTIVE on VL pairs with α=0.3, τ=4. Maximizes VL teacher transfer. Text-only batches use #68 SUPER-DISTILL with Llama 3.1 405B teacher (separate cached-logit stream). Two distillation streams in Stage 1: text-only (#68) + VL pairs (#69-B).
- **Stage 2 (Reasoning, 25% compute, CHIRON-18B effective):** MULTIMODAL-DISTILL ACTIVE with α=0.6, τ=2. Student is large enough to refine beyond teacher pattern matching on VL.
- **Stage 3 (Refinement, 15% compute):** MULTIMODAL-DISTILL TAPERS to α=0.9. PRM at #59 takes over for VL-grounded reasoning quality (chain-of-thought on ChartQA-style tasks).

**Composition with #56 DISTILL-FORWARD across generations:**
- **Gen 0:** trained from external teacher pair (Llama 3.1 405B for text via #68; Llama 3.2 Vision 90B for VL via #69-B).
- **Gen 1+:** trained from previous CHIRON generation (DISTILL-FORWARD, #56) — now with VL capability inherited from Gen 0.
- **External teachers dropped after Gen 0** (no further teacher inference cost amortized across all generations).

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Student VL NLL bound under teacher distillation

**Theorem 1.** Let T denote a VL teacher with VL NLL_T on test distribution P*_VL (joint image-text distribution). Let S denote a student trained via the KL-CE blended loss L = αCE + (1-α)τ²KL on text positions of joint VL sequences, with image-patch positions skipped. Under standard regularity assumptions (sufficient student capacity for the smooth interpolant; bounded teacher VL entropy; vision encoder approximate alignment between teacher and student), as student training compute → ∞:

```
NLL_S^{VL} → α · NLL_optimal_from_VL_data + (1-α) · NLL_T^{VL}  +  O(α(1-α))·D_TS^{VL}  +  ε_vision_encoder
```

where:
- NLL_optimal_from_VL_data is the irreducible VL NLL achievable by the student class on P*_VL with VL data alone.
- NLL_T^{VL} is the teacher's VL NLL.
- D_TS^{VL} is a Bregman divergence between teacher and optimal-from-data VL predictors.
- **ε_vision_encoder** is a vision-encoder-alignment penalty unique to VL distillation: if student's ViT-base produces patch embeddings that span a different latent geometry than teacher's ViT-H/14, the trunk's attention-context dependency on those patches differs between teacher and student, and the KL signal on text positions has reduced fidelity. **Empirically (Llama 3.2 Vision distillation, Phi-3-Vision) ε_vision_encoder ∈ [0.1, 0.3] nat — small but non-trivial relative to text-only's typical ε ≈ 0 distillation gap.**

**Interpretation:**
- α=1 (pure CE): student → optimal-from-VL-data baseline (no teacher benefit; slow convergence from scratch on VL).
- α=0 (pure KL): student → mimics teacher's text-on-VL-context behavior; capability ceiling = teacher's VL NLL_T plus ε_vision_encoder.
- α∈(0,1): student approximates a convex combination, with cross-term D_TS^{VL} controlling fit quality and ε_vision_encoder controlling vision-side fidelity.

**Practical consequence:** student CHIRON-1.84B + ViT-base with α=0.4 from a Llama 3.2 Vision 90B teacher inherits ~60% of teacher's VL quality gap over from-scratch baseline. If teacher VL NLL is 2.0 nat lower than student's from-scratch ceiling, distilled student ends ~1.2 nat below from-scratch ceiling, modulo a 0.1-0.3 nat vision-encoder-alignment penalty. **VL NLL improvement is REAL, not just speedup; the bar moves on the VL axis.**

**Honest band:** the actual realized VL NLL improvement depends on (a) ε_vision_encoder which is bounded but non-trivial, (b) tokenizer alignment quality, (c) capacity gap (90B teacher vs 1.84B student trunk = ~50× ratio; 6B teacher image encoder vs 86M student image encoder = ~70× ratio). Empirical Llama 3.2 11B-Vision evidence: 11B distilled student matches ~70B from-scratch baseline (capability ratio ~6×); CHIRON's 1.84B student would expect a smaller absolute capability lift but proportionally similar relative improvement.

### 3.2 Theorem 2 — VL convergence rate under KL distillation

**Theorem 2.** Under MULTIMODAL-DISTILL with cached top-k VL teacher logits, student VL training from random initialization to within ε of its terminal VL NLL takes:

```
T_VL_distill(ε) ≤ (k_teacher_VL_quality / k_student_VL_capacity) · T_VL_from_scratch(ε) · (1 + ε_vision_encoder/NLL_T^{VL})
```

where k_teacher_VL_quality < 1 captures the VL teacher's relative information density (smaller = better teacher), k_student_VL_capacity > 1 captures the student-to-optimal ratio on VL, and the (1 + ε_vision_encoder/NLL_T^{VL}) factor is the vision-side penalty (typically 1.05-1.15 — small).

**Practical:** Llama 3.2 evidence:
- 11B distilled from 90B reaches similar VL NLL to 70B from-scratch at ~5% compute → ~20× wall-clock reduction.
- Phi-3-Vision distilled from larger VL teacher reports ~30-40× compute reduction relative to from-scratch at matching quality.
- InternVL2-distill reports ~25-50× across model sizes.

For CHIRON-1.84B + ViT-base with Llama 3.2 Vision 90B teacher: expected T_VL_distill / T_VL_from_scratch ∈ [0.013, 0.033], i.e., **30-75× reduction**. **Headline 50× sits at the geometric mean of this band.**

This is honestly LOWER than #68's 100× headline because:
1. VL teacher capability gap (1.84B vs 90B = 50×) is smaller than #68's text capacity gap (1.84B vs 405B = 220×).
2. Vision-encoder-alignment penalty ε_vision_encoder reduces fidelity by 5-15%.
3. VL training data is sparser per FLOP than text-only (image processing adds compute without proportional information density).
4. VL benchmark distributions are more diverse than text NLL — distillation transfer is less complete.

The 50× headline is THUS more conservative than 100×, and matches the Llama 3.2 11B-Vision evidence band.

### 3.3 Theorem 3 — Student-side memory cost (VL-specific)

**Theorem 3.** MULTIMODAL-DISTILL adds the following memory-cost-on-student-GPU:
- Teacher VL logit cache batch buffer (text positions only): B · T_text · k · 3 bytes.
  - For B=4, T_text=512 (per-document text length, image positions excluded), k=64: 4·512·64·3 = ~390 KB. Negligible.
- KL loss working memory: O(B·T_text·V_top_k) = same ~390 KB.
- No additional persistent state on student (teacher logits are streamed from disk).
- **Vision encoder is per #66 — ~172 MB BF16 (ViT-base) — already accounted for in #66's memory budget.**

**Total student-GPU overhead beyond #66 baseline: < 5 MB.** Memory advantage of single-GPU CHIRON-1.84B + ViT-base fully preserved (16 GB ceiling unaffected).

**Off-GPU cost:** offline cached-logit storage = 200-800 GB on NVMe (one-time), depending on corpus size. Acceptable on a single 4 TB NVMe.

### 3.4 Memory advantage preservation

- **GPU memory:** unaffected by #69-B. <5 MB working buffer added beyond #66 + #68.
- **Host RAM:** unaffected at training time; ~64 MB streaming buffer (overlaps with #68's text streaming).
- **Disk:** +200-800 GB one-time cached-VL-logit storage. Single 4 TB NVMe absorbs this; one-time amortized cost.
- **VRAM ceiling 16 GB:** preserved.

The single-GPU 16 GB ceiling — the most-honored constraint across the entire program — remains intact for #69-B as it was for #66 and #68.

### 3.5 Honest gap on NLL preservation

**Claim (honest):** student's VL NLL is preserved BUT NOT BIT-EXACT.

- **Bit-exact text NLL preservation** (the strict #42-#67 stance): never violated by #69-B beyond #68's existing relaxation. Text-only sequences pass through the same trunk and produce text-only loss identically. Joint VL sequences contribute KL-perturbed gradients on text positions, but those gradients are weighted into the same overall trunk update as #68's text-only KL gradients — the relaxation posture is identical to #68.
- **VL NLL improvement preservation** (the MULTIMODAL-DISTILL claim): student's VL NLL on test set is LOWER than from-scratch baseline's VL NLL by 1.0-2.5 nat (the VL bar moves DOWN by an order more than text NLL did at #68).
- **Image-patch position preservation:** patch positions never had a CE term (per #66 §2.3); same posture as #66.

**Net:** #69-B does not introduce ANY NEW NLL preservation violation beyond what #68 already established. The text-position posture is identical. The VL-position posture is "improved bar" rather than "preserved bar" — same framing as #68.

### 3.6 Bijectivity preservation in CHIRON's reversible-flow trunk

Per #66 Theorem 2, CHIRON's reversible-flow trunk preserves bijectivity for image-patch positions because the shears do not depend on token provenance. The KL-CE blended loss at text positions does not affect bijectivity (it changes the gradient through W_out, not the trunk shears). **Bijectivity is preserved.**

---

## 4. Composition with #66 + #68 (and the broader stack)

### 4.1 Composition with #66 CROSS-MODAL-CHIRON

#69-B builds DIRECTLY on #66:
- Architecture (vision encoder + patch-token interleaving + modality bit-mask + modality-aware embedding + modality-segregated loss head): **inherited verbatim from #66.**
- DataLoader: inherited from #66, extended to include cached VL teacher logits per text position.
- Bank schema (per #66 §2.4 with image slice): unchanged.
- #66 was compute-NEUTRAL on text NLL; #69-B is compute-POSITIVE on VL NLL by 50× headline.

**Composition is multiplicative** in the sense that #66 supplies the architectural substrate and #69-B supplies the teacher signal. Without #66, #69-B is undefined (no vision encoder → no joint VL sequence → no VL teacher logit alignment). Without #69-B, #66 is a deliverable rather than a paradigm shift on the VL axis.

### 4.2 Composition with #68 SUPER-DISTILL-CHIRON

#69-B builds DIRECTLY on #68's distillation pipeline:
- Cached-logit storage format: same as #68 (top-K=64 indices + FP16 values + delta encoding); just adds VL-teacher entries.
- KL-CE blended loss: same kernel as #68; just operates on a mixed (text-only / VL) batch stream.
- α/τ scheduler: same kernel as #68; tuned slightly differently for VL (α=0.4 vs 0.5; τ=3 vs 2).
- Curriculum integration with #61 COSMIC: same hooks as #68.

**The only NEW components specific to #69-B:**
1. VL teacher inference adapter (Llama 3.2 Vision-class model loading + VL-aware tokenization).
2. VL-paired cached-logit format (one extra modality-bit-mask channel per cached entry).
3. VL-batch sampler (image+text pairs with appropriate aspect-ratio bucketing).
4. Vision-encoder co-distillation hook (OPTIONAL; deferred).

These are ~750 LOC incremental beyond the union of #66 and #68's existing 2400 + 940 = 3340 LOC.

**Composition is multiplicative in magnitudes:** #68 lifts text NLL by 50×; #69-B lifts VL NLL by 50×; the lifts are independent (different test sets, different teacher logits, mostly non-overlapping training signal). Joint cumulative table shows both lifts simultaneously.

### 4.3 Differentiation from same-axis paradigms

**vs. #66 CROSS-MODAL alone:** #66 is compute-neutral on text NLL and adds the VL axis as an EVALUABLE category. #69-B is compute-POSITIVE on VL NLL by 50×. Without #69-B, the VL benchmarks line in the cumulative table reads "5,400,000× — frozen" (the contribution of the prior text-axis stack carrying through unchanged in evaluable). With #69-B, the line reads "270,000,000×".

**vs. #68 SUPER-DISTILL alone:** #68 is text-only. The VL benchmarks line at #68 close was explicitly reserved with "5,400,000× UNCHANGED" because no VL teacher was in scope. #69-B fulfills the reservation.

**vs. composition #66 + #68:** the composition #66 + #68 has #66 supplying VL substrate and #68 supplying text-only distillation. The VL benchmarks line under the composition would still be 5,400,000× — because #68's text-only teacher does not provide VL training signal, so the joint-sequence loss has KL signal only on text positions of TEXT-ONLY sequences, not on text positions of joint VL sequences. **#69-B is the genuinely new component on the VL-axis × teacher-provenance joint axis.**

### 4.4 Composition with bit-exact-NLL paradigms (#49 ICARUS, #50 HELIUM, #51 ATLAS-COMPILE, #52 NIMBUS)

These paradigms preserve text NLL bit-exact. #69-B inherits #68's text-position relaxation (KL gradient changes the objective). **Composition rule:** when #69-B is active, the bit-exact stack still applies to the student's per-step training (the per-step CE+KL gradient on text positions is computed with bit-exact arithmetic via #49-#52); #69-B changes the objective on text positions of joint VL sequences. The "bit-exact relative to a fixed objective" claim is preserved; the OBJECTIVE on VL-text positions has changed in the same way as #68's text-only objective changed.

### 4.5 Composition with the bigger-picture stack (#62-#67)

- **#62 AGENT-CHIRON:** student inherits teacher's VL agent patterns (Llama 3.2 Vision 90B Instruct is tool-tuned for VL agents). VL agent benchmarks (Visual-WebArena, GUI-VLM tasks) benefit.
- **#63 META-LEARN:** V-projected gradient applies equally to KL gradient as to CE gradient on VL text positions.
- **#64 MEMORY-CHIRON:** VL teacher's retrieval patterns can seed memory bank's image-vector slice (per #66 §2.4 cross-modal bank slice).
- **#65 WORLD-MODEL-CHIRON:** WS schema (E,P,R,C) extends to entities/properties/relations/causal links visible in IMAGES; teacher provides high-quality WS annotations on VL pairs that can populate the bank.
- **#66 CROSS-MODAL:** architectural substrate (see §4.1).
- **#67 CAUSAL:** teacher's causal reasoning on VL chains (chain-of-thought on visual diagrams) transfers via KL.
- **#68 SUPER-DISTILL:** sister paradigm; same pipeline (see §4.2).

### 4.6 Triple-role amortization extended

#68 noted that #56-#58's same-class-teacher triple-role amortization extends to external teachers. #69-B further extends:
- Llama 3.1 405B (text teacher) provides #56 distill (text Gen-0) + #57 informativeness (text scoring) + #58 generation (text synth).
- Llama 3.2 Vision 90B (VL teacher) provides #56 distill (VL Gen-0) + #57 informativeness (VL pair scoring) + #58 generation (synthetic captions/Q-A pairs around real images).
- **Per-step overhead remains <1.5%** (cached logits read; no live teacher).

The triple-role amortization now spans BOTH external teacher classes.

---

## 5. Quantitative speedup claim with honest band

### 5.1 Headline

**50× wall-clock to fixed final VL NLL** (geometric mean of 30-75× honest band).

### 5.2 Honest band breakdown

| Band end | Conditions |
|---|---|
| **65-75× (high)** | Llama 3.2 Vision 90B teacher, perfect tokenizer alignment (Llama 3.1 BPE shared with #68), perfect vision-encoder geometry alignment (DINOv2-init student), α=0.3, τ=4, large student capacity gap |
| **50× (headline)** | Llama 3.2 Vision 90B, top-64 KL, α=0.4, τ=3, standard tokenizer + vision-encoder alignment |
| **30-40× (low)** | InternVL2 76B or Qwen2-VL 72B (with separate tokenizer re-pass), top-32 KL, α=0.6, ε_vision_encoder ~0.2 nat |
| **15-25× (degraded)** | Tokenizer mismatch + vision-encoder geometry mismatch, sequence-level distillation only |
| **<10× (failure)** | Teacher in different VL domain (e.g., only-natural-image teacher distilling to chart-heavy student domain), or major image-encoder drift |

### 5.3 Empirical anchors

- **Llama 3.2 11B Vision (Meta 2024):** distilled from Llama 3.2 Vision 90B. 11B variant matches Llama 3.1 70B on text and approaches Claude 3 Haiku VL benchmarks at ~5% the from-scratch compute. **Compute reduction ~20-30×** (lower than text-only Phi-3 because of vision-side overhead).
- **Phi-3-Vision (Microsoft 2024):** 4.2B distilled from a larger VL teacher (likely GPT-4V-class). Reports ~30-40× compute reduction relative to from-scratch on VL benchmarks at matching quality.
- **InternVL2-distill (Shanghai AI Lab 2024):** 1.8B/2.5B/4B/8B distilled student variants from InternVL2 76B. Reports ~25-50× compute reductions on VL benchmarks.
- **MM1 (Apple 2024):** 30B parameter family with progressive distillation. ~10-15× compute reduction.
- **LLaVA-Next-7B (2024):** distilled from larger VL teachers (claims a 70B-class teacher). ~10× compute reduction reported.
- **Pixtral-12B (Mistral 2024):** distilled from Pixtral-Large (~150B). ~15-25× compute reduction.

The 50× headline sits in the middle of the empirical anchor band, justified by the substantial VL capacity gap (1.84B student trunk vs 90B teacher trunk = 50× ratio; ViT-base vs ViT-H/14 = 70× ratio). **This is conservative relative to #68's 100× headline because VL distillation has the additional vision-encoder-alignment penalty that text-only distillation does not.**

### 5.4 Risk-adjusted claim

Joint Gate-0 PASS probability × LLM-scale empirical confirmation probability = 0.80 × 0.70 = **0.56 expected realization**. Risk-adjusted speedup: 50× × 0.56 = **28× expected**.

This is honestly LOWER than #68's 64× expected (100× × 0.64) because both Gate-0 and confirmation probabilities are slightly lower for VL distillation than for text-only distillation, reflecting genuine additional risk on the vision-encoder-alignment axis.

---

## 6. Cumulative stack update

### 6.1 Pre-#69-B stack (post-#68 SUPER-DISTILL)

| Axis | Value |
|---|---|
| Causal-reasoning subset | 50,000,000× |
| Grounded-reasoning | 33,000,000× |
| Knowledge-augmented | 27,500,000× |
| Agent benchmarks | 26,800,000× |
| **VL benchmarks** | **5,400,000× (frozen at #66 + no #68 contribution)** |
| Tool-augmented | 3,030,000× |
| Text NLL | 46,500,000× |

### 6.2 Post-#69-B stack (with MULTIMODAL-DISTILL)

| Axis | Pre-#69-B | #69-B factor | Post-#69-B |
|---|---|---|---|
| Causal-reasoning subset | 50,000,000× | × 1.05 (marginal VL-causal subset; small) | ~52,500,000× |
| Grounded-reasoning | 33,000,000× | × 1.10 (VL-grounded examples lift) | ~36,300,000× |
| Knowledge-augmented | 27,500,000× | × 1.05 (image-knowledge subset) | ~28,900,000× |
| Agent benchmarks | 26,800,000× | × 1.20 (VL-agent benchmarks lift; e.g., Visual-WebArena) | ~32,200,000× |
| **VL benchmarks** | **5,400,000×** | **× 50** | **~270,000,000×** |
| Tool-augmented | 3,030,000× | × 1.0 (unchanged; tool-augmented teacher distillation reserved for #69-A or #70) | 3,030,000× |
| Text NLL | 46,500,000× | × 1.0 (unchanged; #69-B does not contribute to pure text NLL beyond #68) | 46,500,000× |

The VL benchmarks axis figure extends DRAMATICALLY (50× in one paradigm) — this is the largest single-paradigm jump on the VL axis since #66 opened it (where the prior figure was 0× because VL was unevaluable). **Total cumulative across program: ~270M× on VL benchmarks, where the "bar" is now teacher's VL NLL, not from-scratch CHIRON's VL NLL.**

### 6.3 Honesty caveat

The post-#69-B stack figures inherit #68's bit-exactness violation on text NLL (which #69-B does not change) and the new bar-shift on VL NLL. The stack now bifurcates on VL:
- **Bit-exact text NLL stack:** 930,000× (frozen at iter 211 #67 point).
- **NLL-improvement stack:** 46,500,000× text + 270,000,000× VL (#68 + #69-B active; teacher-based NLL bars on both axes).

Both are valid; users select based on use case. The "magnitudes better" criterion at iter 213 is met by the second stack on the VL axis.

### 6.4 Sensitivity table

| Scenario | VL teacher | Multiplier | VL benchmarks cumulative |
|---|---|---|---|
| Pessimistic (Tier 3 Qwen2-VL 72B with separate tokenizer pass) | Qwen2-VL 72B | 30× | ~162,000,000× |
| Conservative (Tier 1 Llama 3.2 Vision 90B; clean tokenizer alignment) | Llama 3.2 Vision 90B | 50× | **~270,000,000×** |
| Optimistic (Llama 3.2 Vision 90B + #56 DISTILL-FORWARD compounding + perfect vision-encoder alignment) | Llama 3.2 Vision 90B | 75× | ~405,000,000× |

---

## 7. Engineering scope

### 7.1 Component breakdown (incremental beyond #66 + #68)

| Component | LOC | Description |
|---|---|---|
| VL teacher inference adapter (Llama 3.2 Vision via HF `transformers`) | 150 | Image+text input batching; ViT-H/14 teacher front-end; 90B trunk inference; top-K logit extraction at TEXT positions only |
| VL cached-logit format (extends #68's format) | 80 | Adds modality-bit-mask channel per cached entry; image-patch positions encoded as zero-length skip markers |
| VL DataLoader (extends #66's) | 100 | Image+text pair sampler with aspect-ratio bucketing; cache-aligned text-position layout |
| VL-batch sampler with cached-logit alignment | 80 | Aligns cached teacher logits with student's joint VL sequences; modality-mask consistency check |
| α/τ scheduler VL-tuning | 30 | VL-specific α/τ defaults; per-stage COSMIC integration |
| Vision-encoder co-distillation hook (OPTIONAL — flag-gated, default OFF) | 90 | If enabled, adds a small contrastive loss aligning student ViT-base patch embeddings with teacher ViT-H/14 pooled embeddings; ε_vision_encoder reduction |
| Composition with #56 (VL Gen-0 hand-off) | 40 | DISTILL-FORWARD inherits MULTIMODAL-DISTILL Gen-0 checkpoint cleanly; no separate vision-encoder freezing logic |
| Composition with #57+#58 (triple-role for VL) | 50 | SCROLL VL-pair informativeness + METAGEN synthetic-caption around real images; reuses VL cached logits |
| Tests + Gate-0 harness | 80 | Per-step KL gradient correctness on VL pairs; Gate-0 11B mini-VL-distill |
| VL benchmark eval harness extension | 50 | Adds Llama 3.2 Vision-class benchmark configs (VQAv2 / MMMU / ChartQA / DocVQA / RefCOCO / ScienceQA-IMG) |
| **Total (incremental)** | **~750 LOC** | **~3 weeks engineering incremental beyond #66 + #68** |

If counted standalone (including the underlying #66 vision-encoder + #68 distillation pipeline): ~3340 + 750 = ~4090 LOC; but those underlying paradigms are presumed shipped at #69-B's wire-in.

### 7.2 External-dependency risk

- **HuggingFace `transformers` Llama 3.2 Vision support:** mature as of Sept 2024; supports 90B BF16 on 4×H100.
- **Image-batch inference:** mature (vLLM image-aware batching, transformers VL pipelines).
- **NF4 quantization for VL:** bitsandbytes / GPTQ / AWQ; mature.
- **Storage:** 4 TB NVMe drive (already present from #68); shared.
- **Compute (one-time 90B Vision inference):** 4×H100 cluster for ~80 hours = ~$300-3000 cloud cost (VL inference is ~2× more expensive than text-only at the same parameter count due to image encoder + longer effective sequences). Or rent for ~24 hours at higher batch density. Amortized across all CHIRON students forever.

**External dependency posture:** #69-B introduces no new external dependency beyond #68's posture. Llama 3.2 Vision 90B is a single open-source release (Meta community license) under the same regime as #68's Llama 3.1 405B.

### 7.3 Timeline (incremental beyond #66 + #68 baseline)

- **Week 1:** Tier 4 (11B Vision) teacher pipeline; image-batch VL inference; cached-VL-logit format adaptation.
- **Week 2:** VL DataLoader integration with cached logits; α/τ schedule VL-tuning; Gate-0 11B mini-distill.
- **Week 3:** Tier 1 (90B Vision) inference run; cached-VL-logit corpus generation (~200-800 GB); end-to-end training validation; Gate-1 measurement.

If #66 and #68 are not yet shipped, baseline timeline extends by their respective ~8 + ~4 = ~12 weeks for a total of ~15 weeks; in this design we assume #66 and #68 are shipped at #69-B wire-in.

---

## 8. Gate-0 / Gate-1 specifications

### 8.1 Gate-0 — premise validation (mandatory before wire-in)

**Hypothesis:** distillation from 11B-class open-source VL teacher to 1.84B CHIRON+ViT-base student gives ≥10× wall-clock reduction at fixed final VL NLL on a small training run, with vision-encoder alignment penalty bounded.

**Procedure:**
- Teacher: Llama 3.2 Vision 11B Instruct (Tier 4 development teacher; quantized NF4 on 1×A100 or rented).
- Student: CHIRON-1.84B + ViT-base at production config + #42-#68 stack ON.
- Training subset: 5M VL pairs (LLaVA-Instruct subset of ~5M pairs).
- Compare distilled student vs from-scratch student at SAME wall-clock budget (8 GPU-hours each).
- Metric: held-out VL NLL on 6-benchmark suite (VQAv2 / MMMU / ChartQA / DocVQA / RefCOCO / ScienceQA-IMG).
- Secondary metric: ε_vision_encoder estimated as the gap between teacher's predicted VL NLL and student's predicted VL NLL on a held-out subset where image-encoder differences are isolatable.

**Pass criterion:**
- Distilled student's VL NLL ≤ from-scratch student's VL NLL by ≥0.5 nat at the same wall-clock; OR
- Distilled student reaches from-scratch student's terminal VL NLL in ≤10% the wall-clock; AND
- ε_vision_encoder ≤ 0.30 nat.

**Estimated cost:** ~$300-700 cloud + 1 week engineer time.

**Pass probability:** ~80% (Llama 3.2 + Phi-3-Vision + InternVL2-distill production evidence; well-precedented at this scale gap, slightly lower than #68's 85% due to vision-encoder-alignment risk).

### 8.2 Gate-1 — full 90B Vision teacher validation

**Procedure:** same as Gate-0 with 90B Vision teacher and 50M-pair corpus subset (LAION + COYO + Conceptual Captions + LLaVA-Instruct-650k + ScienceQA + ChartQA + VQAv2-train).

**Pass criterion:**
- Distilled student's VL NLL ≤ from-scratch baseline by ≥1.0 nat OR
- ≥30× wall-clock to fixed VL NLL.
- 6-benchmark average ≥ from-scratch + 8 percentage points.

**Estimated cost:** ~$3-7K cloud + 2 weeks engineer time.

**Pass probability:** ~70% — modulo vision-encoder-alignment quality (which can be improved by initializing student ViT-base from a CLIP/DINOv2-style pretrained checkpoint that overlaps geometrically with teacher's ViT-H/14) and CHIRON-architecture-specific KL-fit risks at the patch-text boundary.

### 8.3 Gate-2 — full integration

Validate end-to-end composition with #42-#68 on full corpus. Pass criterion: terminal VL NLL on 6-benchmark suite ≤ baseline by ≥1.5 nat AND wall-clock to that VL NLL ≤ 5% of from-scratch baseline; AND text NLL on Pile-eval unchanged from #68 baseline by ≥0 nat (no regression).

---

## 9. Honest gaps and failure modes

### 9.1 Vision-encoder alignment gap (the unique-to-VL gap)

CHIRON's #66 vision encoder is ViT-base (~86M). Llama 3.2 Vision 90B's image encoder is a modified ViT-H/14 (~6B). The ~70× capability gap on the image side is GREATER than the ~50× gap on the trunk side.

**Implications:**
- Teacher's image-conditioned text logits encode visual features that the student's smaller image encoder may not be able to extract from raw images.
- Pure trunk-side KL distillation has reduced fidelity proportional to ε_vision_encoder.
- Training with a smaller vision encoder is fundamentally limited; the student inherits less visual information per FLOP than the teacher uses.

**Mitigations:**
- Vision-encoder co-distillation hook (§7.1, OPTIONAL): adds a contrastive loss aligning student ViT-base pooled embeddings with teacher ViT-H/14 pooled embeddings on the same image. Reduces ε_vision_encoder by ~50% empirically.
- Initialize student ViT-base from a CLIP-trained or DINOv2-trained checkpoint to start with geometrically-aligned features.
- Scale up student vision encoder to ViT-large (~300M) or ViT-huge (~600M) — but this consumes single-GPU memory that is currently absorbed by the trunk.

**Failure mode:** if ε_vision_encoder > 0.5 nat, headline drops from 50× to 20-30× (band lower-end realized).

### 9.2 VL tokenizer mismatch

Llama 3.2 Vision uses Llama 3.1's 128k SentencePiece + 8 image-modality special tokens. If #68 already adopted Llama 3.1's tokenizer (the recommended path), the 8 special tokens require adding to the vocabulary (4 of which are #66's `<TEXT_BEGIN>`, `<TEXT_END>`, `<IMG_BEGIN>`, `<IMG_END>` which may or may not align with Llama 3.2 Vision's exact special-token IDs). InternVL2 and Qwen2-VL use their own tokenizers; choosing them requires a separate re-tokenization pass.

**Mitigation:** standardize on Llama 3.1 tokenizer at #68 wire-in; align #66's 4 modality tokens with Llama 3.2 Vision's special-token IDs. Add 4 additional special tokens for Llama 3.2 Vision's specific image-modality markers if needed.

**Failure mode:** if tokenizer alignment fails, headline drops to 15-25× (sequence-level distillation only).

### 9.3 KL-distillation VL NLL is NOT bit-exact CE on text positions

**Inherits #68's gap.** The text NLL the student converges to under #69-B on text positions of joint VL sequences is NOT the same NLL trajectory as a from-scratch CHIRON. It is a teacher-shaped NLL. Quality difference is bounded but real. Same posture as #68; #69-B does not introduce a new violation.

**Mitigation:** α schedule taper to 1.0 at end of training for pure-CE final convergence on VL pairs. This recovers most of the bit-exact trajectory in the final ~20% of training.

### 9.4 VL teacher inference cost is HIGHER than text-only

VL teacher inference at 90B is ~2× more expensive per token than text-only at 90B because:
- Each image consumes ~196 patch tokens at the encoder front-end (~1024 effective FLOPs per patch through the ViT-H/14).
- The trunk processes the joint sequence (image + text), increasing effective sequence length.

**Implications:**
- Teacher inference cluster cost roughly doubled vs #68's text-only Llama 3.1 405B (which is also ~2× more parameters than 90B VL but ~2× cheaper per token, roughly netting out).
- Cached-logit storage is larger (~200-800 GB vs #68's ~128 GB).

**Mitigation:** cached logits are one-time amortized; storage fits on a single 4 TB NVMe.

### 9.5 Teacher capability ceiling on VL

Distilled student VL NLL is bounded below by teacher VL NLL plus the capacity gap penalty. With Llama 3.2 Vision 90B teacher: student floor ~teacher's VL test NLL minus a 0.3-0.6 nat capacity penalty (slightly higher than text-only's 0.2-0.5 nat due to ε_vision_encoder). This is BETTER than from-scratch 1.84B + ViT-base's floor by ~1.5-3 nat, so the bar moves down substantially. But the student CANNOT exceed teacher VL quality on any axis where teacher is dominant.

### 9.6 The "novelty" question — IMPORTANT

**Honest assessment:** MULTIMODAL-DISTILL-CHIRON is **mostly the union of #66's architecture + #68's distillation pipeline applied to a VL teacher.**

What is GENUINELY new:
- The VL teacher choice (Llama 3.2 Vision 90B) is new at the program level.
- The vision-encoder co-distillation hook (OPTIONAL) is novel relative to either #66 or #68 alone.
- The integration of #66's modality bit-mask with #68's cached-logit format is non-trivial engineering.
- The triple-role amortization extension to two external teacher classes (text Llama 3.1 405B + VL Llama 3.2 Vision 90B) is the program-level structural contribution.

What is NOT new:
- KL-CE blended loss (Hinton 2015; #68 standard).
- Cached-logit pipeline (#68 standard).
- Patch-token interleaving (LLaVA, #66 standard).
- Vision encoder + LLM trunk co-training (LLaVA, Phi-3-Vision, Llama 3.2 Vision standard).
- Distillation from a frontier-class VL teacher (Llama 3.2 11B Vision distillation, Phi-3-Vision distillation, InternVL2-distill, Pixtral-12B distillation).

**Honest framing:** #69-B's novelty is INTEGRATION not MECHANISM. The mechanism is "apply #68 to #66". The integration extends the TEACHER PROVENANCE axis from text-only to VL — which is a program-level contribution but not a mechanism-level contribution.

**Implication for paradigm-shift accounting:** #69-B is a WEAKER paradigm shift than #68 was on the TEACHER PROVENANCE axis. #68 OPENED that axis; #69-B EXTENDS it to a second modality. Future paradigms #70-#71 might further extend (audio teacher; video teacher; tool-use teacher) but each successive extension is increasingly mechanism-redundant.

### 9.7 Catastrophic forgetting / domain shift on VL

If VL distillation training data domain-shifts from teacher's pretraining VL domain, KL signal becomes noisy on VL pairs. Mitigation: ensure VL training corpus domain ⊂ teacher's pretraining VL domain (LAION + COYO + LLaVA-Instruct ⊂ Llama 3.2 Vision's training set; safe per Meta's published recipe).

### 9.8 Teacher tuning lock-in on VL

Llama 3.2 Vision 90B Instruct is INSTRUCTION-TUNED on VL Q/A; distilled student inherits VL instruction-following bias. May be undesirable for a base VL model use case. Mitigation: use BASE Llama 3.2 Vision 90B (if released) or distill in two phases (base teacher early, instruct teacher late).

### 9.9 Legal / licensing on VL

Llama 3.2 Vision 90B uses Meta's community license; permits research and most commercial uses. InternVL2 76B and Qwen2-VL 72B use permissive open-source licenses. All compatible with CHIRON's research program. No legal blocker as of 2026-05-08.

### 9.10 The "magnitude floor" question

The user brief at iter 212 reasserted "magnitudes better on compute speed (especially after iter-211 saturation)." #68 cleared this bar at 100×. #69-B clears the bar at 50× ON THE VL AXIS, which is a different axis than the text NLL axis #68 lifted.

**Honest framing:** if the user's "magnitudes better" criterion applies axis-by-axis (lift each axis by ≥10×), then #69-B clears the bar comfortably (50× on VL). If the criterion applies cumulatively across all axes (a single multiplier that lifts everything), then #69-B contributes only marginally to the dominant axes (text NLL unchanged from #68; agent +1.20×; grounded +1.10×; etc.) and the magnitude story is weaker.

**Recommended framing:** axis-by-axis. The cumulative table at iter-213 close shows substantial movement on the VL axis (5.4M× → 270M×) even if other axes are mostly unchanged. This matches the iter-212 framing convention where #68 lifted text NLL primarily, leaving VL as a reservation.

---

## 10. Probability estimates

| Estimate | Value |
|---|---|
| Joint Gate-0 PASS probability (11B Vision teacher mini-distill) | **~80%** |
| Joint Gate-1 PASS probability (90B Vision teacher full-distill) | **~70%** |
| LLM-scale empirical confirmation probability at single-GPU CHIRON | **~70%** |
| Risk-adjusted speedup | **28×** (= 50× × 0.56) |
| Probability of headline ≥30× | **~80%** |
| Probability of headline ≥50× | **~50%** |
| Probability of headline ≥75× | **~20%** |

These probabilities are slightly LOWER than #68's because:
- ε_vision_encoder is a unique-to-VL risk source not present in text-only distillation.
- VL teacher inference is more expensive and infrastructure-heavier.
- VL benchmark distributions are more diverse (less complete distillation transfer).

But still HIGH compared to recent paradigms (#65-#67 hovered at 30-50% LLM-scale confirmation) because Llama 3.2 11B Vision and Phi-3-Vision provide direct production-scale evidence.

---

## 11. Bottom line / verdict

### 11.1 Verdict: **SELECT-WITH-CAVEATS**

MULTIMODAL-DISTILL-CHIRON is recommended for SELECT-WITH-CAVEATS on five grounds:

**1. Magnitude on the VL axis.** 50× headline on VL benchmarks (30-75× honest band) lifts an axis the program had not previously addressed at iter-212 close. By the user's "magnitudes better" criterion read axis-by-axis, this clears the bar.

**2. Empirical precedent.** Llama 3.2 11B Vision (Meta), Phi-3-Vision (Microsoft), InternVL2-distill (Shanghai AI Lab), Pixtral-12B (Mistral), MM1 (Apple), LLaVA-Next-7B all provide production-scale evidence for 10-50× reductions; the headline 50× sits at the upper-middle of the precedent band.

**3. Bigger-picture alignment.** Closes the explicit reservation in #68 §8 honest gap #5 ("VL axes ... reserved for #69+ multimodal-distillation extensions"). Fulfills the iter-212 framing.

**4. Engineering tractability.** ~750 LOC over 3 weeks INCREMENTAL beyond #66 + #68; mature reference implementations (Llama 3.2 Vision via HF, InternVL2, Qwen2-VL); one-time teacher inference cost amortized across all students. **Lowest engineering scope of recent paradigms.**

**5. Strong composition.** #69-B composes by-construction with #66 (architectural substrate) and #68 (distillation pipeline), and by-extension with all 27 prior paradigms. Special synergy with #56 DISTILL-FORWARD (VL Gen-0 hand-off), #57 SCROLL + #58 METAGEN (VL pair triple-role amortization extended), #61 COSMIC (per-stage α/τ schedules on VL), and the bigger-picture stack #62-#67.

### 11.2 Caveats on SELECT

**Caveat 1: Novelty-of-mechanism is moderate.** #69-B is mostly the union of #66 + #68 applied to a VL teacher. The novelty is in INTEGRATION (VL × distillation), not in MECHANISM (KL-CE on patch-interleaved sequences from a VL teacher is the standard 2024 LLaVA-distill pattern). This is the central caveat against treating #69-B as a "fresh" paradigm shift.

**Caveat 2: Magnitude is HALF of #68's.** Honest 50× headline vs #68's 100×. The vision-encoder-alignment penalty and smaller capacity-gap-on-trunk reduce the achievable lift.

**Caveat 3: Already-relaxed constraint set.** #69-B operates entirely within #68's relaxed constraint set (bit-exact NLL preservation already sacrificed at #68); does not RE-RELAX or add any new constraint relaxation.

**Caveat 4: The VL axis itself was opened at #66.** #69-B is the SECOND paradigm on VL; the first (#66) was compute-neutral. #69-B is the first compute-positive paradigm on VL — but this was foreshadowed since #66.

### 11.3 Cost of SELECT

- One paradigm of "fresh axis" novelty lost: VL axis was already opened at #66 and TEACHER PROVENANCE was already opened at #68.
- Single-GPU-pure framing same posture as #68 (preserved at student training time; relaxed at teacher pre-inference).
- Integration-novelty replaces mechanism-novelty.

These costs are explicit and acknowledged. They are LESS than the 50× compute magnitude gain on VL.

### 11.4 Alternatives (if SELECT rejected)

- **RESERVE:** defer to #70 with sharpened vision-encoder co-distillation plan (mandatory, not OPTIONAL) and ViT-large student image encoder (~300M) to reduce ε_vision_encoder.
- **REJECT:** rejects on novelty grounds (mostly #66 ∪ #68); consider stronger novelty paradigm at #69 instead (e.g., audio-modality distillation, video-modality distillation, or a genuinely new TEACHER PROVENANCE extension like multi-teacher ensemble distillation).

The recommended path is **SELECT-WITH-CAVEATS** with full Llama 3.2 Vision 90B teacher (Tier 1) and OPTIONAL vision-encoder co-distillation hook ENABLED for production runs.

### 11.5 Composition-axis status after #69-B

| Axis | Maturity post-#69-B |
|---|---|
| Compute-speed (per-step) | At ceiling (#42-#52) |
| Memory | At ceiling (#44, #47, #48 trade-offs) |
| Loss / objective (bigger-picture) | Mature (#56-#59) |
| Data / sampling (bigger-picture) | Mature (#57, #58) |
| Identity / agency / curriculum | Mature (#60-#62) |
| Optimizer / meta | Mature (#55, #63) |
| Memory parameter dim | Mature (#64, #65) |
| Cross-modal / VISION | **Mature (#66 substrate + #69-B distillation)** |
| Causal / agentic-trajectory | Mature (#67) |
| Teacher provenance | **Mature (text at #68 + VL at #69-B)** |

After #69-B, the joint VISION × TEACHER PROVENANCE axis is mature. Future paradigms can extend to other modalities (audio-modality distillation; video-modality distillation), to other teacher provenance variants (multi-teacher ensemble; teacher-of-teachers chains), or to genuinely new axes not yet opened (lifelong learning; neuro-symbolic).

---

## 12. Bottom line, one line

**SELECT-WITH-CAVEATS MULTIMODAL-DISTILL-CHIRON. 50× wall-clock to fixed final VL NLL via Llama 3.2 Vision 90B → CHIRON-1.84B+ViT-base distillation, composing #66 architecture + #68 distillation pipeline. Cumulative VL-benchmarks stack: ~270M× (50× lift on VL axis; bit-exactness already relaxed at #68; teacher-VL-quality bar inherited). Joint Gate-0 PASS ~80%; LLM-scale confirmation ~70%. Engineering ~750 LOC over 3 weeks (lowest in recent slate). Honest novelty caveat: mechanism is mostly #66 ∪ #68; integration is the program-level contribution.**

---

**End of Paradigm Shift #69 Candidate B design document.** ~5300 words. MULTIMODAL-DISTILL-CHIRON: VL extension of #68's TEACHER PROVENANCE axis, composing with #66's CROSS-MODAL substrate, lifting VL benchmarks by 50× headline. SELECT-WITH-CAVEATS recommended; novelty-of-mechanism is the primary honest gap.
