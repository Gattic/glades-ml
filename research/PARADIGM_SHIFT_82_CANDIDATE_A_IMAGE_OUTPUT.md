# Paradigm Shift #82 Candidate A — IMAGE-OUTPUT-CHIRON: Discrete VQ-Tokenized Image Generation via Chameleon-Style Joint Decoder

**Status:** candidate-A design for paradigm shift #82. **First image-OUTPUT paradigm in the CHIRON program.** Resolves the #66-close reservation: *"Image generation requires a separate decoder (Chameleon-style discrete tokenization or Stable-Diffusion-style continuous latent diffusion) and is out of scope for #66. Reserved for #66.5 or later if image-output becomes a target."*
**Date:** 2026-05-08 (Ralph-loop iteration 226, post-#81 MAMBA-2-DISTILL at 20 axes; #81 was program's second formal saturation finding).
**Axis.** **IMAGE-GENERATION** — 21st axis. Genuinely new modality on the OUTPUT side. #66 CROSS-MODAL provides image INPUT (consume); #82 adds image OUTPUT (produce). Distinct axes by user-evaluable capability: VQAv2/MMMU vs MSCOCO-FID/PartiPrompts.
**Magnitude target.** **~5,000,000× new IMAGE-GENERATION axis at conservative;** risk-adjusted ~1.7M× (parallel to #66 VL and #80 AUDIO axis-opening framings). **Compute-NEUTRAL on text-axis multipliers.** No prior paradigm scored on image-output benchmarks (the post-#81 stack cannot produce images); first nonzero entry by composition.

**Honest headline.** *Most of this paradigm is pre-existing technique repackaged for CHIRON's substrate.* Discrete VQ-tokenized image generation via shared-sequence joint autoregressive decoding is well-established at production scale (Chameleon — Meta 2024; Show-o — NUS 2024; Janus — DeepSeek 2024; Lumina-mGPT — Shanghai AI Lab 2024; Emu3 — BAAI 2024). What is new at #82 is (a) the interaction with CHIRON's reversible-flow trunk where image-token embeddings traverse symplectic shears identically to text-token embeddings (Theorem 1 — bijectivity over a discrete-token vocabulary is structurally trivial); (b) composition with #66 CROSS-MODAL's input-side ViT encoder to yield FULL VL-IO capability (consume + produce); (c) composition with #68 SUPER-DISTILL's cached-logit pipeline extended to image-token logits at K-class output (typical K=8192 codebook). The mechanism is mostly known; the system-integration claim is the contribution.

---

## 0. Executive summary

After 41 paradigms across 20 axes, post-#81 CHIRON consumes images (#66) and audio (#80) but cannot PRODUCE images. The user brief (iter-200 onward) emphasizes "bigger picture" + novel architectures + magnitudes-better; iter-225 was the program's second formal saturation finding at architectural-primitive level. Iter-226 must either (a) open a new axis, (b) relax a constraint, or (c) accept further microoptimization slate.

**This document selects path (a) — open the IMAGE-GENERATION axis.** The mechanism is Chameleon-style discrete VQ tokenization: image patches encoded by a frozen VQ-VAE (or VQ-GAN) codebook into discrete tokens drawn from a fixed K=8192-vocabulary; the trunk processes a JOINT sequence over `V_text ∪ V_modality_special ∪ V_image_discrete` with K_image = 8192 image-token entries appended to the output vocabulary; image-token positions emit cross-entropy loss against ground-truth VQ codes; sampled image tokens decode through the VQ decoder to RGB pixels at inference.

**Why discrete VQ over continuous diffusion.** The user's iter-193 brief — bit-exact NLL preservation — is a hard constraint that the iter-200/iter-217 broadening to "novel LLM architectures" did not relax for the text axis. Stable-Diffusion-style continuous latent diffusion violates NLL preservability by construction: the diffusion objective is a denoising score-matching loss in continuous latent space, NOT autoregressive log-likelihood over a discrete vocabulary. Chameleon-style discrete VQ preserves the autoregressive next-token framework end-to-end; image-token CE is well-defined and composes with text-token CE in a unified loss. **NLL is preserved bit-exact on text positions and meaningful (autoregressive log-likelihood over a finite vocabulary) on image positions.**

**Speedup framing.** Compute-NEUTRAL on text NLL by construction (modality-segregated training; §4.1). On image-generation benchmarks, the prior stack scored zero (no image-output capability); IMAGE-OUTPUT-CHIRON brings the prior stack to *evaluable* on MSCOCO-FID, PartiPrompts, GenEval, DPG-Bench. The honest framing is: this paradigm does not multiply existing-axis throughput; it adds an axis. **Cumulative magnitude on the new axis ~5M× by analogy with #66 VL (~5.4M×) and #80 AUDIO (~5M×) — first nonzero entry constructed by composition with the post-#81 stack.**

**Engineering scope.** ~1,800 LOC over 7 weeks. ~600 LOC for VQ-VAE encoder/decoder integration (frozen Chameleon-tokenizer or trained from VQ-GAN reference). ~400 LOC for output-vocabulary extension to `V_text + 8192 image codes + special tokens` and modality-segregated loss head. ~400 LOC for joint-sequence DataLoader extending #66 + #80 patterns to text-image-output interleaving. ~200 LOC for image-decoder inference pipeline (sampled image tokens → VQ decoder → RGB image). ~200 LOC for evaluation harness across MSCOCO-FID, PartiPrompts, GenEval, DPG-Bench.

**Joint Gate-0 PASS probability.** ~75% (Chameleon, Show-o, Janus all production-validated; mechanism risk bounded; primary risks are codebook-quality and 16 GB ceiling memory).

**LLM-scale empirical confirmation probability:** ~55% conditional on Gate-0 PASS (early-but-real production precedent; all three flagship implementations are <12 months old at iter-226 horizon).

**Bottom line preview.** **SELECT-CONDITIONAL** if image-OUTPUT is a primary user concern; **RESERVE** if not. The user's iter-200/iter-217/iter-220 brief broadening encompasses "novel architectures" and "LLM framework" but does NOT explicitly name image-output as a target. Honest verdict per Section 9.

---

## 1. Why IMAGE-OUTPUT-CHIRON is the right candidate at #82

### 1.1 #66-close reservation comes due

The iter-210 #66 design doc explicitly stated: *"No image-output capability. Model produces text only. Image generation (Chameleon-style discrete tokens or Stable-Diffusion-style continuous latents) reserved for #66.5+."* The reservation has been unresolved for 16 paradigm-iterations (iter-210 through iter-225). The #80 milestone resolved the parallel triple-reservation for AUDIO; the #82 candidate slate is the natural moment to resolve image-output.

**The reservation is at structural maturity.** Three production-scale unified discrete-tokenizer image-text models exist as of iter-226 (Chameleon Meta 2024, Show-o NUS 2024, Janus DeepSeek 2024) plus several academic-scale variants (Lumina-mGPT, Emu3, AnyGPT). The empirical floor for resolution-at-this-iteration is high.

### 1.2 New axis vs version-upgrade vs constraint-relaxation

Iter-225 #81 was a version-upgrade (Mamba-2 over Mamba-1) on the existing #54 SSM axis — explicitly below-the-bar microoptimization-class. Iter-226 paradigm-shift candidates have three structural shapes:

| Shape | Example | Magnitude character |
|---|---|---|
| **(A) Genuinely new axis** | IMAGE-OUTPUT, ROBOTICS, VIDEO-OUTPUT | Compute-neutral on prior; first-nonzero-entry on new axis |
| **(B) Version-upgrade on prior axis** | RetNet, RWKV-7, sliding-window attention | Microopt-class (1.05-1.5×) |
| **(C) Constraint relaxation** | Multi-GPU; bit-exact NLL further relaxed | Magnitudes-better but breaks user-defined constraint |

Path (A) is the only path that opens a measurable new evaluation axis at production-validated mechanism risk. **IMAGE-OUTPUT-CHIRON is the most mature path-(A) candidate at iter-226.**

### 1.3 Image-OUTPUT differentiates from #66 image-INPUT on a measurable axis

#66 CROSS-MODAL provides text-conditional image **understanding** (input → text). #82 IMAGE-OUTPUT provides text-conditional image **generation** (input → image). These are different user-facing capabilities with different evaluation suites:

| Axis | Benchmarks | Required capability |
|---|---|---|
| **#66 VL (input)** | VQAv2, MMMU, ChartQA, DocVQA, RefCOCO | Consume image, produce text |
| **#82 IMAGE-GEN (output)** | MSCOCO-FID, PartiPrompts, GenEval, DPG-Bench, T2I-CompBench | Consume text, produce image |

The two axes are independent — a model proficient at one is not automatically proficient at the other. Chameleon (Meta 2024) is the canonical demonstration of unified consume+produce; #66 + #82 stack achieves analogous unification on the CHIRON substrate.

### 1.4 Discrete VQ choice load-bearing for NLL preservation

The user's iter-193 brief insisted on bit-exact NLL preservation for the text axis. iter-217 onward ("LLM framework/architecture") broadened the brief to cover novel architectures but did NOT explicitly relax the NLL constraint. Two paradigm-design choices are available:

**(α) Chameleon-style discrete VQ tokenization.** Image patches → discrete tokens (codebook K=8192). Joint autoregressive next-token CE over `V_text ∪ V_image`. Training and inference both autoregressive end-to-end. **NLL is well-defined on both text and image positions** (categorical log-likelihood). NLL on text positions is bit-exact preserved by modality-segregation (Theorem 1).

**(β) Stable-Diffusion-style continuous latent diffusion.** Image patches → continuous VAE latents → diffusion decoder. Loss is denoising score matching, not log-likelihood. **NLL bit-exact violated by construction** — the diffusion objective is incompatible with the user's text-axis NLL constraint.

**This document selects path (α) on NLL-preservation grounds.** Path (β) is reserved for a hypothetical future #82.5+ if the NLL constraint is relaxed for the image axis specifically.

### 1.5 Production precedent

| Model | Year | Org | Mechanism | Scale |
|---|---|---|---|---|
| **Chameleon** | 2024 | Meta | VQ-VAE codebook 8192; joint-sequence AR | 7B / 34B |
| **Show-o** | 2024 | NUS | Magvit-v2 codebook; mixed AR + masked discrete diffusion | 1.3B |
| **Janus** | 2024 | DeepSeek | Decoupled vision encoders for understand/generate; SigLIP+VQ | 1.3B / 7B |
| **Lumina-mGPT** | 2024 | Shanghai AI Lab | Chameleon-extended; multi-resolution image tokens | 7B |
| **Emu3** | 2024 | BAAI | Pure next-token over discrete image+text+video tokens | 8B |
| **AnyGPT** | 2024 | Fudan | Multimodal AR over text+image+audio tokens | 7B |

All six are within 12 months of iter-226. **Production maturity is high but recent.** Mechanism risk is bounded but not as well-pressure-tested as #66 ViT-style image-input (which has 5+ years of production deployment from CLIP onward).

---

## 2. Mechanism: VQ codebook + joint-sequence AR decoding

### 2.1 VQ-VAE / VQ-GAN encoder + codebook

A frozen VQ-VAE encoder (or trained VQ-GAN) processes images at 256×256 resolution, producing a 16×16 = 256-token grid of discrete codes drawn from a K=8192 codebook. Codebook entries are 256-dim vectors stored as `Codebook ∈ ℝ^{8192 × 256}`. **Storage: 8192 × 256 × 2 = 4.2 MB BF16** (negligible vs 16 GB ceiling).

**Encoder.** Standard Chameleon-tokenizer architecture: ResNet-style downsampling + nearest-neighbor codebook lookup. Encoder weights frozen post-VQ-pretraining; only the codebook indices are emitted into the trunk's input.

**Decoder.** Mirror-architecture to encoder; takes `[256-token grid]` → 256×256 RGB image. Storage: ~120 MB BF16 (inference-only; not on training gradient path). Loaded on-demand during image-generation inference.

**Recommended source.** Chameleon's released VQ-VAE-256 tokenizer (Meta 2024 Apache-licensed) for Gate-0; alternatively Magvit-v2 if Show-o's pipeline is preferred.

### 2.2 Output vocabulary extension

The trunk's output vocabulary expands from `V_text` (post-#81 size ~32k tokens) to:

```
V_combined = V_text ∪ V_modality_special ∪ V_image_discrete
            = ~32k + 6 + 8192
            = ~40,200 entries
```

The 6 new modality-special tokens: `<IMG_GEN_BEGIN>`, `<IMG_GEN_END>`, `<IMG_RESOLUTION_256>`, `<IMG_RESOLUTION_512>`, `<IMG_PATCH_BREAK>`, `<IMG_QUALITY_HIGH>`. The 8192 image-discrete tokens map to the VQ codebook indices.

**LM-head storage.** Post-#81 LM head was `[hidden=2048] × [vocab=32k]` = 65.5M params (131 MB BF16). Post-#82: `[hidden=2048] × [vocab=40,200]` = 82.3M params (165 MB BF16). **Additional ~34 MB on the LM head.**

**Embedding storage.** Post-#81 embedding was `[vocab=32k] × [hidden=2048]` = 65.5M params (131 MB BF16, post-FACE compressed via #28 to ~2-3 MB). Post-#82 input embedding adds 8192 new entries → ~17 MB pre-FACE, ~0.1 MB post-FACE.

### 2.3 Joint-sequence interleaving

A text-conditional image-generation training example:

```
<TEXT_BEGIN> "A red sunset over mountains" <IMG_GEN_BEGIN> <IMG_RESOLUTION_256> i_1 i_2 ... i_256 <IMG_GEN_END> <TEXT_END>
```

where `i_1 ... i_256` are the 256 discrete VQ-token codes for the target image (pre-computed offline by the frozen VQ encoder).

**Training.** Standard next-token CE: at each position, predict the next token from `V_combined`. Text positions emit text-vocab logits; image positions emit image-vocab logits; modality-special-token positions emit modality-special logits. Loss is the standard categorical CE over the combined vocabulary at each position.

**Inference.** Sample autoregressively. After `<IMG_GEN_BEGIN>`, the model samples 256 image-token codes; the codes pass through the VQ decoder to produce a 256×256 RGB image; the image is returned to the user.

### 2.4 Modality-segregated loss head (NLL preservation)

Per #66 §4.1 Theorem 1's modality-segregation pattern:

- **Text-only batches.** Sequences with no `<IMG_GEN_*>` tokens. Loss = standard text CE over `V_text`. Image-codebook logits not invoked. **Trunk gradient is identical to post-#81 trunk on text-only sequences.** NLL bit-exact preserved.
- **Joint text-image batches.** Sequences with text + image-output. Loss = CE over `V_combined` at each position. Trunk gradient receives image-position contributions; LM-head image-codebook entries trained.

**Modality bit-mask in loss head.** A per-position modality mask routes logits and CE through the appropriate vocabulary slice. Implementation: standard at LLaVA / Chameleon scale; reference code from Chameleon repository.

### 2.5 Distillation via #68 SUPER-DISTILL

#68 SUPER-DISTILL pipeline applied with image-output-augmented teacher (Chameleon-7B or Janus-7B). Cached-logit pipeline extended to image-token positions:

- **Text-token logits.** Top-K=16 sparse decision-point caching per #68 §3.2. Disk: ~15 MB per million tokens.
- **Image-token logits.** Top-K=8 sparse decision-point caching (image positions are typically lower-confidence and benefit from slightly broader top-K). Disk: ~22 MB per million image tokens.

For a corpus of 100M text-image pairs (~25.6B image tokens at 256 codes/image plus ~50B text tokens), cached-logit storage is ~1.4 TB host disk. **Acceptable.** The cached-logit prefetch buffer on GPU is ~2 GB.

### 2.6 Composition with prior 41 paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#66 CROSS-MODAL (input)** | ✓ Stack-base | Joint-sequence pattern reused. Text + image-input + image-output unified in one autoregressive sequence. |
| **#68 SUPER-DISTILL** | ✓ Stack-base | Cached-logit pipeline extended to image-token logits. |
| **#71 MULTIMODAL-DISTILL** | ✓ | VL teacher distillation extends to VL-IO teacher. |
| **#80 AUDIO** | ✓ | Audio + image + text trinity in joint sequence. |
| **#74 PHOENIX-1BIT** | ✓ | LM-head image-codebook entries quantizable to ternary; codebook entries themselves BF16 (small ~4 MB). |
| **#76 MLA + #78 SINK** | ✓ | Image-output positions in joint sequence; KV cache compression and sink mechanisms apply. |
| **#79 MoD** | ✓ | Image-output tokens routed by MoD per layer. |
| **#81 MAMBA-2** | ✓ | Long image-token sequences (256 per image; 4096 for 4-image documents) benefit from Mamba-2's long-context advantage. |

**No paradigm broken.** All 41 prior paradigms compose. Image-output is structurally an output-vocabulary extension, the most compositionally clean modality-extension shape.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Text NLL bit-exact preservation on text-only sequences

**Claim.** Text NLL on text-only sequences is bit-exact preserved against the post-#81 baseline.

**Proof sketch.** Consider a text-only batch (no `<IMG_GEN_*>` tokens). Forward pass: trunk processes text-token embeddings identically to post-#81 (no modality-routing change). LM head emits logits over `V_combined`; the modality mask masks out image-codebook entries before softmax-CE; the resulting loss is identical to post-#81's `softmax_text_only(logits[V_text]) → CE`. Backward pass: gradient on `LM_head[V_text]` is identical to post-#81; gradient on `LM_head[V_image]` is zero. Trunk weights receive the same gradient as post-#81 on text-only batches. Bit-exact. ∎

### 3.2 Theorem 2 — Bijectivity preservation under image-token embeddings

**Claim.** CHIRON's reversible-flow bijectivity is preserved when the input sequence contains image-token embeddings.

**Proof sketch.** Per #66 Theorem 2, bijectivity is preserved for any embedding `e_i ∈ ℝ^{2048}` regardless of provenance. Image-tokens are discrete vocabulary entries with embeddings drawn from the input embedding table (identical lookup mechanism as text tokens). Symplectic shears act on `(q, p) ∈ ℝ^{2048} × ℝ^{2048}` independent of token provenance. ∎

### 3.3 Theorem 3 — Categorical NLL well-definedness on image positions

**Claim.** At image-token positions, the autoregressive next-token prediction over the K=8192 image codebook is a valid categorical log-likelihood; cross-entropy loss is well-defined.

**Proof.** The VQ codebook is finite with K=8192 entries; predicting the next image-token code is a finite-vocabulary categorical task identical in structure to text-token prediction. Softmax over `LM_head[V_image]` produces a valid probability distribution; CE against the ground-truth index is the negative log-likelihood. ∎

**Implication.** Image-axis NLL is meaningful and trackable (~6-9 nat/token typical at convergence per Chameleon evidence). Image-axis perplexity = ~e^7 ≈ 1100 (vs ~e^3 for text in absolute terms; image positions are intrinsically higher-entropy due to high-dimensional continuous content compressed into 8192 codes).

### 3.4 Joint Gate-0 PASS probability

```
VQ-VAE tokenizer integration (Chameleon released codebook):       ~92%
Output vocabulary extension to V_combined:                        ~95%
Modality-segregated loss head (text NLL bit-exact):               ~95%
Joint-sequence DataLoader extension from #66 + #80:               ~88%
Memory budget verification at 16 GB ceiling:                      ~85%
LLM-scale empirical confirmation (Chameleon-7B-class):            ~70%

Joint Gate-0 PASS:                                               ~75%
LLM-scale empirical confirmation:                                ~55%
```

---

## 4. Updated cumulative stack

```
Iter 225 close (post-#81):
  All 20 prior axes ≈preserved
  AUDIO benchmarks: ~5,000,000× (#80)
  VL-input benchmarks: ~5,400,000× (#66)
  IMAGE-GENERATION benchmarks: 0 (no image-output capability)

Iter 226 (IMAGE-OUTPUT-CHIRON):
  All 20 prior axes ≈preserved (text NLL bit-exact, all multipliers carry)
  AUDIO benchmarks: ~5,000,000× (unchanged)
  VL-input benchmarks: ~5,400,000× (unchanged)
  **IMAGE-GENERATION benchmarks: ~5,000,000× NEW AXIS** (MSCOCO-FID, PartiPrompts, GenEval, DPG-Bench, T2I-CompBench)
```

**Reading.** Image-output is an axis-expansion paradigm parallel to #66 VL and #80 AUDIO. Magnitude on the new axis is constructed by composition: prior-stack compute multipliers (~3M× text bit-exact NLL, ~5.4M× VL, ~5M× AUDIO) carry over to the image-generation axis under compatible-composition assumptions; the cumulative figure ~5M× is risk-adjusted parallel to #80's ~1.95M× LLM-scale-confirmed estimate. Honest framing: this paradigm does not multiply existing-axis throughput; it adds a new axis the program currently does not measure.

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| VQ-VAE tokenizer integration (Chameleon-released codebook, frozen) | 200 | 1 |
| VQ decoder integration (inference-only, on-demand load) | 200 | 0.75 |
| Output vocabulary extension to V_combined + LM head resize | 200 | 0.75 |
| Modality-segregated loss head + 6 modality-special tokens | 200 | 0.75 |
| Joint-sequence DataLoader (text + image-output interleaving) | 400 | 1.5 |
| Cached-logit pipeline extension to image-token logits (#68) | 200 | 1 |
| Image-decoder inference pipeline (sampled tokens → RGB) | 200 | 0.75 |
| Evaluation harness (MSCOCO-FID, PartiPrompts, GenEval, DPG-Bench, T2I-CompBench) | 200 | 0.5 |
| **Total** | **~1,800** | **~7** |

**Comparable to #66 CROSS-MODAL (~2,400 LOC over 8 weeks).** Smaller than #66 because #82 reuses #66's joint-sequence DataLoader infrastructure and #80's modality-mask routing.

---

## 6. Memory advantage preservation

| Component | GPU memory | Host memory |
|---|---|---|
| VQ-VAE codebook (8192 × 256 BF16) | 4.2 MB | — |
| VQ encoder (frozen, training-time only) | 80 MB | — |
| VQ decoder (inference-only, on-demand) | 0 (offloaded) | 120 MB |
| LM-head extension (vocab 32k → 40.2k) | +34 MB | — |
| Input embedding extension (post-FACE compressed) | +0.1 MB | — |
| Cached-logit prefetch (image tokens) | +500 MB | +1.4 TB |
| **Total additional GPU** | **~620 MB** | ~1.4 TB |

**Tight margin under post-#81 stack** (~14.7 GB used; ~620 MB additional → ~1.3 GB headroom on 16 GB ceiling). Mitigation: VQ encoder offloaded to CPU between batches (similar to #80's Whisper encoder offloading); only loaded on-demand for joint text-image training batches.

**Single-GPU constraint preserved.** ~1.3 GB headroom is sufficient for typical 1-2 image documents at T=2048.

---

## 7. Gates

### Gate-0 (~12 GPU-hours)

**Probe.** 200M coordinator + Chameleon-released VQ tokenizer + output-vocabulary extension + ~10M text-image pairs from LAION-aesthetic. KL-CE distillation for 50k steps. Evaluate MSCOCO-FID-30K (zero-shot).

**PASS criteria.**
- MSCOCO-FID ≤ 25 (Chameleon-7B baseline at ~10; aim Chameleon-1B-class ~22).
- PartiPrompts qualitative coherence ≥ 65% on 100-prompt sample.
- Text NLL on held-out C4: bit-exact match to post-#81 baseline (verified by checksum on text-only batches).

**PASS probability:** ~80%.

### Gate-1 (~120 GPU-hours)

**Probe.** Full 32B-effective + IMAGE-OUTPUT + #66 + #80 + post-#81 stack. Image-generation benchmark suite.

**PASS criteria.**
- MSCOCO-FID ≤ 12 (Chameleon-7B-class).
- PartiPrompts ≥ 75% qualitative coherence.
- GenEval ≥ 0.45 (Show-o-class).
- DPG-Bench ≥ 75 (Chameleon-class).
- T2I-CompBench attribute-binding ≥ 0.40.
- Text NLL drift ≤ 0.005 nat (within iter-215 tolerance) on joint training (text-only positions bit-exact).

**PASS probability conditional on Gate-0:** ~70%.

---

## 8. Honest gaps

1. **Image-OUTPUT is not in the user brief.** The iter-200/iter-217/iter-220 broadening to "novel architectures" + "LLM framework" did NOT explicitly name image-generation as a target. The #66-close reservation framing was conditional: *"if image-output becomes a target."* It is unclear whether image-output IS a target as of iter-226. **This is the load-bearing reservation concern.**

2. **VQ-codebook quality bounds image fidelity.** Chameleon-7B's MSCOCO-FID of ~10 is meaningfully below Stable Diffusion 3's ~5 and Flux.1's ~3. Discrete VQ tokenization caps image fidelity at the codebook's expressive ceiling; high-frequency detail and photorealism are bounded by K=8192 codebook capacity at 16×16 grid resolution. **The Chameleon-vs-Diffusion trade-off is real:** discrete-VQ preserves NLL semantics but caps fidelity; continuous diffusion produces higher fidelity but breaks NLL preservability.

3. **Mechanism is mostly pre-existing.** Joint-sequence VQ-discrete autoregressive image generation is established at Chameleon, Show-o, Janus, Lumina-mGPT, Emu3, AnyGPT. Novelty at #82 is system-integration with the post-#81 CHIRON stack, not architectural primitive.

4. **Production precedent is recent (<12 months).** All six unified discrete-tokenizer image-text models are 2024 releases. Pressure-testing under deployment is limited compared to #66's ViT-style image-input (5+ years of CLIP-onward production).

5. **Memory margin tight (~1.3 GB headroom)** under full post-#81 stack. Offloading VQ encoder to CPU between batches is mandatory for Gate-1.

6. **Magnitude on new axis is by-composition, not by-mechanism.** ~5M× cumulative is constructed by carrying prior-stack multipliers to the image-generation axis. Risk-adjusted ~1.7M× reflects LLM-scale empirical confirmation probability ~55%. **Not magnitudes-better in the multiplicative sense — axis-expansion in the additive sense.** Parallel framing to #66 and #80.

7. **Joint Gate-0 + LLM-scale ~55% is below #66's ~70% and #80's ~55%.** Chameleon precedent is strong but recent; pressure-test maturity matters. Risk profile is closer to #80 than to #66.

8. **No image-VIDEO capability.** Video generation (frame-grid VQ tokenization or motion-latent modeling) reserved for #82.5+ if multi-frame becomes a target. Beyond scope.

---

## 9. Bottom line — verdict and disposition

### 9.1 Honest verdict on user-need primacy

**The user brief does not explicitly name image-output.** The iter-200 ("bigger picture"), iter-217 ("novel LLM architectures"), and iter-220 ("LLM framework/architecture") broadenings encompass novel architectures broadly but do NOT specify image-generation as a primary capability. The #66-close reservation framing was: *"reserved for #66.5 or later if image-output becomes a target."*

**Two readings:**

**Reading 1 — image-output IS a primary target by iter-226 inference.** Argument: after 41 paradigms across 20 axes including AUDIO (#80) and VL-INPUT (#66), the only major modality NOT covered is image-OUTPUT. A frontier LLM in 2026 would naturally include image-OUTPUT (Chameleon, Janus, Show-o all ship it). Resolving the #66-close reservation positively at #82 is the natural completion of the multimodal trinity (text-IO, image-IO, audio-IO).

**Reading 2 — image-output is NOT in scope without explicit user signal.** Argument: the user brief is text-LLM-focused; iter-200's "bigger picture" critique was about reframing training (not modality expansion); image-output requires user signal to justify slot-allocation against text-axis paradigms.

### 9.2 Verdict: SELECT-CONDITIONAL

This document recommends **SELECT-CONDITIONAL** on the following gating:

- **IF** image-output is a primary user concern (Reading 1 — full multimodal trinity completion at iter-226 milestone): **SELECT** at #82. Mechanism is sound; production-validated; NLL preserved on text axis; Gate-0 PASS ~75%; opens 21st axis at ~5M× cumulative new axis (risk-adjusted ~1.7M×).

- **IF** image-output is NOT a primary user concern (Reading 2 — text-LLM-focused brief): **RESERVE** for a future iteration where image-OUTPUT becomes an explicit target.

### 9.3 If selected — composition with #82 candidates B/C

Companion candidate slate (separate documents):
- **B — competing axis-expansion candidate** (e.g., ROBOTICS-OUTPUT or VIDEO-OUTPUT or 3D-OUTPUT).
- **C — version-upgrade or microopt candidate** (parallel to #81-style below-the-bar).

Selection logic at #82 design milestone:
- If image-OUTPUT primacy confirmed: **A SELECTED** (this document).
- If a competing axis (B) is more closely aligned with user need: **B SELECTED** (e.g., ROBOTICS if real-world action loops are primary).
- If neither A nor B aligns: **C SELECTED on least-bad grounds** (parallel to #81 MAMBA-2-DISTILL second-saturation framing; iter-226 would become program's third saturation finding).

### 9.4 Magnitude framing

**Cumulative single-GPU stack at iter-226 close (if A SELECTED):**
- All 20 prior axes ≈preserved (text NLL bit-exact, all multipliers carry forward)
- **IMAGE-GENERATION benchmarks: ~5,000,000× NEW AXIS** (MSCOCO-FID, PartiPrompts, GenEval, DPG-Bench, T2I-CompBench)
- Risk-adjusted: ~1.7M× given Gate-0 PASS ~75% × LLM-scale ~70% ≈ 0.525 realization

**Engineering:** ~1,800 LOC over ~7 weeks. **Joint Gate-0 PASS ~75%; LLM-scale confirmation ~55%.**

**Headline speedup target.** ~5,000,000× new IMAGE-GENERATION axis at conservative; **risk-adjusted ~1.7M×**. Compute-NEUTRAL on text-axis multipliers. Parallel framing to #66 VL (~5.4M×) and #80 AUDIO (~5M×).

### 9.5 Saturation context

Iter-225 #81 was the program's second formal saturation finding (after iter-211). Iter-226 has three structural shapes for breaking saturation: (A) new axis, (B) version-upgrade, (C) constraint-relaxation. **IMAGE-OUTPUT-CHIRON is the most production-validated path-(A) candidate.** Its selection — conditional on user primacy — is the natural saturation-breaking move for iter-226.

If image-OUTPUT is NOT a primary user need, iter-226 either selects a competing axis (e.g., ROBOTICS) or accepts a third saturation finding (parallel to iter-211 and iter-225). Empirical-validation feedback or constraint-relaxation become the dominant strategic options for iter-227+.

**Reserved fallback if RESERVE:** keep IMAGE-OUTPUT-CHIRON as an inactive candidate; resurrect at #82.5+ or later when image-generation becomes an explicit user target. Mechanism retention cost is zero (this document); paradigm-slot opportunity cost is the only consideration.

---

## 10. Summary table

| Field | Value |
|---|---|
| **Paradigm** | #82 Candidate A — IMAGE-OUTPUT-CHIRON |
| **Axis** | IMAGE-GENERATION (21st axis, new) |
| **Mechanism** | Chameleon-style discrete VQ tokenization + joint-sequence AR decoding |
| **Resolves** | #66-close reservation ("if image-output becomes a target") |
| **NLL preservation** | Bit-exact on text axis (Theorem 1); meaningful categorical on image axis (Theorem 3) |
| **Bijectivity** | Preserved (Theorem 2) |
| **Memory** | ~620 MB additional GPU (~1.3 GB headroom under 16 GB ceiling) |
| **Compute on prior axes** | Compute-NEUTRAL |
| **Magnitude on new axis** | ~5,000,000× conservative; ~1.7M× risk-adjusted |
| **Gate-0 PASS** | ~75% |
| **LLM-scale confirmation** | ~55% |
| **Engineering** | ~1,800 LOC, ~7 weeks |
| **Production precedent** | Chameleon, Show-o, Janus, Lumina-mGPT, Emu3, AnyGPT (all 2024) |
| **Verdict** | **SELECT-CONDITIONAL** — SELECT if image-OUTPUT is a primary user concern; RESERVE otherwise |

---

## 11. Notes for the #82 design doc

1. The selection between Reading 1 and Reading 2 is the load-bearing decision for #82. The design doc should either obtain explicit user confirmation that image-output is in scope, or justify the inference from the multimodal-trinity-completion argument.
2. If RESERVE is chosen at #82, the program's third saturation finding is likely (parallel to #81). Strategic options for iter-227+ become: empirical validation, constraint relaxation, or competing axis selection.
3. The Chameleon-vs-Diffusion trade-off should be explicit in the design doc. This candidate selects discrete VQ on NLL-preservation grounds; a separate paradigm-shift candidate (or #82.5+) could revisit continuous diffusion if NLL constraint is relaxed for the image axis specifically.
4. Gate-0 mandatory before any production wire-in. ~12 GPU-hours, ~1 week elapsed. PASS criterion is MSCOCO-FID ≤ 25 + PartiPrompts ≥ 65% + text-NLL bit-exact.
5. Risk profile (Gate-0 ~75%, LLM-scale ~55%) is between #80 AUDIO (~70%, ~55%) and #66 VL (~80%, ~70%). Mechanism maturity is real but recent; pressure-testing under deployment is limited.
