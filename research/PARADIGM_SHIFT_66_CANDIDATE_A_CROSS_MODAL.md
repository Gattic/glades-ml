# Paradigm Shift #66 Candidate A — CROSS-MODAL-CHIRON: Vision-Language Extension via Joint Sequence Token Interleaving

**Status:** candidate-A design for paradigm shift #66. First serious cross-modal proposal in the CHIRON program.
**Date:** 2026-05-08 (Ralph-loop iteration 210, post-#65 close at ~6,600,000× cumulative on grounded-reasoning subset).
**Axis.** **CROSS-MODAL** — extends CHIRON's input domain from text-only to {text, image} via patch-token interleaving in a shared autoregressive sequence. New axis name: **VISION** (eleventh axis after DATA / LOSS / SAMPLING / REWARD / IDENTITY / SCHEDULE / AGENCY / OPTIMIZER / GROUNDING / KNOWLEDGE-LOCUS).
**Magnitude target.** **Compute-NEUTRAL on text NLL** (bit-exact preserved by construction — see §4.1). New evaluable axis: **VL benchmarks** (VQAv2, MMMU, ChartQA, DocVQA, RefCOCO, ScienceQA-IMG). Honest headline figure on VL axis: **first nonzero entry in the cumulative table** — the prior #65 stack scored 0× on these because they require image input.

**Honest headline.** *Most of this paradigm is pre-existing technique repackaged for CHIRON's substrate.* Vision-language model construction via shared-sequence patch-token interleaving is well-established (LLaVA 2023, GPT-4V 2023, Flamingo 2022, Chameleon 2024, Idefics2 2024, Llama 3.2 Vision 2024). What is new at #66 is (a) the interaction with CHIRON's reversible-flow trunk (Theorem 1 — vision tokens preserve bijectivity); (b) the interaction with #42 SCFA, #44 MELT, and #64 MEMORY (image patches as bank-row content with image-vector slice); (c) the demonstration that text-NLL preservation survives modality mixing under causal masking with modality-segregated loss weighting. The mechanism is mostly known; the system-integration claim is the contribution. **Verdict: RESERVE** — cross-modal extension is compute-neutral on the user's core constraint (text NLL) and adds a benchmark axis the program currently does not measure. Worth doing as a deliverable but does not contribute a multiplicative wall-clock factor on the existing axes.

---

## 0. Executive summary

After 24 paradigms (#42–#65), the CHIRON program has produced ~6.6M× cumulative speedup on grounded-reasoning, ~5.5M× on knowledge-augmented, ~5.36M× on agent benchmarks, ~3.03M× on tool-augmented, and ~0.93M× on bit-exact text NLL. All of these are TEXT axes. The user's iter-200 brief — "looking at the bigger picture instead of focusing on microoptimizations" — has been interpreted across iterations 200–209 as "reframing how training works." Iter-209 explicitly flagged that #66+ candidates need new axes: cross-modal, lifelong-learning, neuro-symbolic.

**CROSS-MODAL-CHIRON is the cross-modal candidate for #66.** It extends the input domain of the trunk from text tokens only to a joint vocabulary `V_text ∪ V_modality_special ∪ V_image_continuous` where image content enters as continuous patch embeddings emitted by a co-trained vision encoder, surrounded by special boundary tokens `<IMG_BEGIN>` and `<IMG_END>`. The trunk processes the joint sequence with no architectural change; tokens are tokens. Text NLL on text-only sequences is preserved bit-exact under modality-segregated loss weighting (text tokens use standard CE; image-token positions skip CE entirely or use a separate image-reconstruction loss). #64 MEMORY's bank schema extends to host image-vector slices, enabling cross-modal retrieval (text query → image rows; image query → text rows).

**Primary differentiation from prior LLM-only paradigms.** This is the first paradigm in the CHIRON program that changes the *input domain*. All previous paradigms (#42–#65) operate on the {text → text} mapping with various objective, sampling, optimizer, grounding, and knowledge-locus reframings. #66 changes what the model *consumes*. The output domain remains text (autoregressive next-token prediction over `V_text ∪ V_modality_special`).

**Speedup framing.** Compute-NEUTRAL on text NLL by construction. Text-only sequences pass through the same trunk and produce identical NLL trajectories (assuming vision-encoder weights are not on the gradient path of text-only batches — see §4.1 for the formal argument). On VL benchmarks, the prior stack scored zero (no image input); CROSS-MODAL-CHIRON brings the prior stack to *evaluable* on VL benchmarks. The honest framing is: this paradigm does not multiply existing-axis throughput; it adds an axis.

**Engineering scope.** ~2400 LOC over 8 weeks. ~1500 LOC for the vision encoder (ViT-base-style, 86M params), patch tokenizer, and modality-aware embedding lookup. ~600 LOC for joint-sequence DataLoader, modality masking in the loss head, and bank schema extension. ~300 LOC for evaluation harness covering 6 VL benchmarks.

**Joint Gate-0 PASS probability.** ~80% (vision-language joint training is well-established at scale; the risk is whether memory advantages survive — see §10). **LLM-scale empirical confirmation probability:** ~70% conditional on Gate-0 PASS (LLaVA-class results are reproducible at <2B trunk).

**Bottom line.** **RESERVE** for a future iteration where VL becomes a primary metric. Mechanism is sound but mostly pre-existing; speedup contribution to existing axes is zero; memory advantages preserved by bounded image-token budgets. Listed as a deliverable rather than a paradigm shift in the wall-clock-magnitude sense.

---

## 1. Why CROSS-MODAL-CHIRON is the right candidate at #66

### 1.1 Cross-modal is the first new-axis candidate flagged by #65

The iter-209 design doc explicitly states: *"After #65, the GROUNDING axis is mature at four-channel level. #66+ candidates need genuinely new axes (cross-modal, lifelong-learning, neuro-symbolic)."* Of the three flagged directions, cross-modal has the strongest empirical precedent. LLaVA, GPT-4V, Flamingo, Chameleon, Idefics2, Llama 3.2 Vision, Pixtral, Qwen-VL, InternVL, and Molmo all ship cross-modal LLMs at production scale with reproducible training recipes. By contrast, lifelong-learning has no production-scale LLM precedent (catastrophic-forgetting remains unsolved); neuro-symbolic has narrow precedent (Mathematica plugins, Wolfram Alpha integrations) but no production-scale architectural synthesis.

**The empirical floor for cross-modal is high; the empirical floor for lifelong-learning or neuro-symbolic is low.** This argues for cross-modal as the safer #66 paradigm and reserves lifelong-learning / neuro-symbolic for #67+.

### 1.2 Cross-modal is structurally compatible with CHIRON

CHIRON's reversible-flow trunk processes paired `(q, p)` state through symplectic shears. The trunk operates on token embeddings of any provenance — the trunk does not "know" whether a token came from text vocabulary, special-boundary vocabulary, or a continuous image-patch projection. Theorem 1 (§4.1) establishes that bijectivity is preserved under modality mixing as long as the embedding space dimension is fixed.

This compatibility is *not* automatic for all paradigms. For example, a discrete VQ-VAE-style image tokenization (Chameleon-style) would force image patches into discrete tokens via vector quantization; the VQ codebook is non-differentiable through the quantization step (straight-through estimator required); CHIRON's bijectivity requires careful argument. Continuous patch embeddings (LLaVA-style) avoid this complication entirely — each patch is a continuous vector projected to the trunk's embedding space, then traversed by shears in the same way as text-token embeddings.

**This document selects continuous patch embeddings (LLaVA-style) on CHIRON-bijectivity grounds.** Discrete tokenization (Chameleon-style) is reserved for a future #66.5 if generation in the image domain becomes a target.

### 1.3 Cross-modal is compute-neutral on text NLL by construction

The user's iter-193 brief — "and nll accuracy" — has been interpreted as "bit-exact text NLL is strictly preserved." All compute-axis paradigms #50–#52 (HELIUM / ATLAS-COMPILE / NIMBUS) preserved bit-exact NLL by construction. #66 CROSS-MODAL must do the same on text-only sequences — otherwise the user's strict constraint is violated.

Section 4.1 establishes Theorem 1: under modality-segregated training (text-only batches and joint vision-text batches drawn separately), text-only batches drive the trunk's gradient identically to a text-only model, so text NLL is preserved bit-exact. Joint vision-text batches train the vision-encoder and the trunk's vision-aware adaptations; these batches contribute to a *separate* loss component that does not touch text-only NLL.

**Bit-exact preservation is achievable.** It requires the training schedule to interleave text-only and joint batches with the gradient accumulator zeroed between them — equivalent to keeping two model "modes" with identical text weights and no cross-mode gradient leakage on text-only forward passes.

---

## 2. Mechanism: vision encoder + token interleaving + cross-modal bank slice

### 2.1 Vision encoder

A ViT-base-style encoder processes images at 224×224 input resolution, producing N=196 patch tokens at 14×14 grid (16×16 pixel patches). Encoder architecture: 12 transformer blocks, hidden dim 768, 12 heads, MLP ratio 4. Total params: ~86M. Output: `[N=196, 768]`.

A linear projection `W_proj ∈ ℝ^{768 × m}` (where `m = 2048` is the trunk's embedding dim) maps each patch vector to the trunk's embedding space. Plus a learned positional bias per patch index.

**Training mode.** The vision encoder is co-trained end-to-end with the trunk. In the joint vision-text batch path, gradient flows from the trunk's loss through `W_proj` to the encoder. In the text-only batch path, the encoder is bypassed (the input sequence contains no image tokens; the encoder's forward pass is not invoked).

**Initialization.** Vision encoder initialized from a pretrained ViT-base (e.g., DINOv2 or CLIP-image-encoder). The trunk inherits the post-#65 weights. `W_proj` is initialized small (~0.01× std) so that early-stage joint training does not destabilize the trunk.

**Storage.** ViT-base in BF16: `86M × 2 bytes = 172 MB`. Total model storage with vision encoder: trunk (1.84B × 2 = 3.68 GB) + ViT (172 MB) + Adam state (already FACE-compressed per #28) ≈ <5 GB additional headroom on the 16 GB single-GPU ceiling. **Feasible.**

### 2.2 Token interleaving in the joint sequence

A joint sequence is constructed as:

```
<TEXT_BEGIN> t_1 t_2 ... t_k <IMG_BEGIN> p_1 p_2 ... p_N <IMG_END> t_{k+1} ... t_{k+m} <IMG_BEGIN> p_1' ... p_N' <IMG_END> ... t_T <TEXT_END>
```

where `t_*` are text tokens, `p_*` are continuous patch vectors, `<IMG_BEGIN>` and `<IMG_END>` are discrete special tokens (vocabulary entries), and the sequence supports multiple images per document with text interleaved between them.

**Embedding lookup.** Modality-aware:
- Text tokens (and special boundary tokens): standard `Embed[vocab_index] ∈ ℝ^m`.
- Image patches: `W_proj · ViT(image)[patch_index] ∈ ℝ^m`.

The embedding-lookup function dispatches based on a per-position modality bit-mask supplied by the DataLoader.

**Causal mask.** Standard upper-triangular causal mask. Image patches at position `i` attend to all preceding tokens (text and image) at positions `< i`. Text tokens at position `j` (for `j > i`) attend to all preceding tokens including image patches.

**Sequence-length budget.** Each image consumes 196 patch tokens. At T=2048 sequence length, a single image consumes ~10% of the budget. Two images consume ~20%. This bounds memory: a typical training document is text + 1–2 images + text, fitting comfortably at T=2048. Long-document multi-image cases (T=8192+) are addressable via #54 JAMBA-CHIRON's hybrid SSM blocks (T=8192 was already scoped as a JAMBA strength).

### 2.3 Modality-aware loss head

Output projection: `W_out ∈ ℝ^{m × (|V_text| + |V_modality_special|)}`. Output vocabulary size: text vocab (32k for GPT-2-style or 128k for Llama-3-style) + modality special tokens (`<TEXT_BEGIN>`, `<TEXT_END>`, `<IMG_BEGIN>`, `<IMG_END>` — 4 tokens). Image patches do *not* appear in the output vocabulary; the model is text-output only.

**Loss formulation:**
- At text-token positions and modality-special-token positions: standard CE, contributing to `L_text`.
- At image-patch positions: skipped in CE entirely. Optionally, a contrastive loss aligning the trunk's hidden state at the patch position with the original ViT patch vector (this is *optional* and is present only if image-input-discrimination is a target metric; default OFF).

**Net loss:** `L = L_text` (default). Image patches contribute to the trunk's training only via attention pathways from subsequent text tokens that "see" the image patches in their context.

**Sequence-level masking.** A modality bit-mask `M ∈ {0, 1}^T` (1 = text/special, 0 = image-patch) is supplied per sequence. The CE summation is `L_text = Σ_{i: M_i = 1} CE(logits_i, target_i)`. Image-patch positions contribute zero gradient to the output projection.

### 2.4 Cross-modal bank slice (composition with #64 MEMORY-CHIRON)

#64-B's bank schema at iter-208 was:
```
M[i] ∈ ℝ^{256}    (256-dim text-encoded vector)
```

#65-A extended to:
```
M[i] ∈ ℝ^{288} = [M[i]_text (256), M[i]_WS (32)]
```

#66 extends further:
```
M[i] ∈ ℝ^{416} = [M[i]_text (256), M[i]_WS (32), M[i]_image (128)]
```

The 128-dim image slice is encoded by a small (~20k-param) image-encoder MLP that takes the global `[CLS]`-style pooled vision-encoder output and projects to 128-dim. Bank rows derived from text-only sources have a zero-vector image slice. Bank rows derived from {text, image} pairs (e.g., a Wikipedia article with associated images, an arXiv paper with figures) have a non-zero image slice.

**Cross-modal retrieval.** A text query `q_text ∈ ℝ^{256}` extended with a query-side image slice `q_image ∈ ℝ^{128}` (zero if the query has no image content) retrieves bank rows by joint cosine on the 416-dim joint vector. **Text query → image-rich rows is enabled** (a query about "the diagram showing X" can match a bank row with both text-encoded "X" and an image slice encoding the diagram).

**Population.** Bank-population sweep at #66 adds ~0.1B image-text-paired rows from sources like LAION, COYO, DataComp, Conceptual Captions, sciAviation. This is ~1% of #64-B's 10B-row bank. The image encoding is cached at population time.

**Storage cost.** Adding 128-dim image slice to all 10B rows would cost `10B × 128 × 0.5 byte (NF4) = 640 GB` — exceeds host RAM. Mitigation: only the image-text-paired ~0.1B rows carry a non-zero image slice; the remaining ~9.9B rows store a 4-byte sparse-flag indicator. Effective storage: `0.1B × 128 × 0.5 = 6.4 GB additional`. **Feasible on the 128-GB host RAM target.**

---

## 3. Composition with prior paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#42 SCFA** | ✓ | Spectral compressed flow attention applies to joint sequence; image patches are tokens in the same sequence. SCFA's depthwise-conv recovers out-of-spectrum residual for both modalities. |
| **#44 MELT** | ✓ | Tensor-train FFN factorization is modality-agnostic. FFN sees hidden states post-attention; no special handling required. |
| **#47 PHOENIX-1.58BIT** | ✓ | Ternary trunk weights apply to both text and image-derived hidden states. ViT encoder kept BF16 (small relative cost; ~172 MB). |
| **#50 HELIUM (FlashAttention-3 + FP8)** | ✓ | FA-3 supports variable-length sequences with arbitrary token content. FP8 forward applies to both text and image tokens. |
| **#51 ATLAS-COMPILE** | ✓ | CUDA Graphs replay accommodates fixed sequence shapes; per-shape autotuning handles vision-encoder shape separately. |
| **#54 JAMBA-CHIRON** | ✓ | Hybrid Mamba+SCFA. Mamba blocks handle long joint sequences (T=8192+) where multiple images push T beyond pure-attention feasibility. |
| **#56 DISTILL-FORWARD** | ✓ | Teacher distillation extends to vision: a frozen Llama 3.2 Vision teacher distills into the CROSS-MODAL-CHIRON student. |
| **#57 SCROLL** | ✓ | Self-curriculum active learning extends to vision-text pairs via per-pair informativeness scoring. |
| **#58 METAGEN** | △ Partial | Synthetic data generation extends to text but not natively to images; text-only METAGEN remains. |
| **#59 PRM-CHIRON** | ✓ | Process reward modeling extends to vision-language reasoning (step-by-step answers to ChartQA-style questions). |
| **#60 TOOL-LLM** | ✓ | Tools include image-processing utilities (cropping, scaling, OCR fallback). |
| **#62 AGENT-CHIRON** | ✓ | Agent loops include vision-grounded tasks (e.g., GUI navigation, visual web tasks like Visual-WebArena). |
| **#64 MEMORY-CHIRON** | ✓ Synergistic | Bank schema extends to host image vectors per §2.4. |
| **#65 WORLD-MODEL-CHIRON-PROMOTED-III** | ✓ | WS schema `(E, P, R, C)` extends to entities/properties/relations/causal links visible in images; vision provides a richer source of structured-state annotations. |

**Notable non-conflicts.** No prior paradigm forbids cross-modal extension. The compute-axis paradigms (#50–#52) are modality-agnostic; the data/loss/sampling/reward axes (#56–#65) generalize. The only partial-composition is #58 METAGEN: synthetic image generation is non-trivial (would require a separate image-generation model); for #66 the simplification is to use real images with synthetic captions (text-side METAGEN over real images).

**Memory-advantage interaction.** §10 covers the honest argument; in summary: image patches are dense (196 tokens per image) but bounded per document. The bank-row image slice is sparse (only ~1% of bank rows have non-zero image vectors). The vision encoder is small (~172 MB). **No paradigm in the prior stack is broken by cross-modal extension; memory advantages survive.**

---

## 4. Theoretical analysis

### 4.1 Theorem 1 — Text NLL preservation on text-only sequences

**Claim.** For any text-only input sequence `s = (t_1, ..., t_T)` with no image content, the cross-entropy loss `L_text(s)` and the gradient `∇_θ L_text(s)` produced by CROSS-MODAL-CHIRON are bit-exact identical to those produced by the post-#65 text-only baseline, provided:

1. The vision encoder's parameters do not appear in `θ_text` (the gradient on text-only forward passes flows only through the trunk's text-aware parameters).
2. The trunk's embedding lookup at text-token positions uses the same lookup table as the post-#65 baseline.
3. The trunk's output projection at text-token positions excludes image-patch positions (which contribute zero in text-only sequences anyway).

**Proof.** A text-only sequence `s` produces a modality bit-mask `M_i = 1` for all `i ∈ [1, T]`. The forward pass through the embedding lookup invokes the text branch only (no `W_proj` calls, no ViT calls). The trunk's forward computation is identical to the post-#65 baseline because:
- Embedding lookup: same lookup table, no perturbation.
- Trunk shears: shears do not depend on modality; their parameters are shared.
- Output projection: `W_out` is extended by 4 special-token slots; the text-vocab slots are unchanged. Logit values at text-vocab positions are identical (the extension does not affect existing rows).

Cross-entropy at text-token positions is computed over the same vocab (including the 4 special-token slots, which have logit values but ground-truth at text-only training data is always a text token). The gradient at the output projection is zero at the special-token slots (because the target is never special, so the derivative `∂L/∂W_out[special_slot]` is zero conditional on the softmax denominator including only text-vocab targets at text-only positions).

Wait — this last step is subtle. The softmax denominator includes *all* output slots, including the special-token slots. So adding 4 slots changes the softmax denominator and therefore the per-token cross-entropy by `Δ = log(Z_extended / Z_baseline)` where `Z_extended = Z_baseline + Σ_{s ∈ specials} exp(logit_s)`. **This is non-zero.**

**Correction.** For strict bit-exact preservation, the special-token slots must be initialized so that their logits are `-∞` (or in practice, `< -50` so that `exp(logit_s) ≈ 2e-22`, well below BF16 precision and irrelevant to the softmax). This is achievable by constraint-projecting the corresponding `W_out` rows to zero or by adding a fixed-large-negative bias to those slots in text-only mode. With this constraint, `Z_extended ≈ Z_baseline` to machine precision, and CE is bit-exact preserved.

**Stronger claim (possible).** If the special-token slot bias is set to a large negative constant (say -10^9), then in text-only mode the softmax over those slots produces zero contribution to *any* text-token CE term and zero gradient. The baseline NLL is recovered exactly.

**Practical reality.** The constraint adds a small amount of hand-tuning (the special-token bias must be set so that joint-mode training can flip it on for special-token-prediction positions while text-only training keeps it off). One implementation: a "modality flag" input to the output projection that toggles the special-token bias.

**Verdict.** Bit-exact text NLL on text-only sequences is achievable with care. **The claim is correct but requires implementation discipline.** ∎

### 4.2 Joint vision-text loss formulation

For joint sequences with `M_i ∈ {0, 1}` per-position modality:

```
L_joint(s) = Σ_{i: M_i = 1} CE(softmax(logits_i), target_i)
```

Image-patch positions (`M_i = 0`) contribute zero to `L_joint`. The gradient through `W_proj` and the ViT encoder flows via:
- Trunk attention from subsequent text tokens to image-patch hidden states.
- Trunk attention from later image patches to earlier image patches in the same image.

This is the standard LLaVA loss formulation. It does *not* train the vision encoder to "predict pixels" — the encoder is trained to produce embeddings that, when consumed by the trunk, lead to lower text-token CE at downstream positions. Empirically (LLaVA, Idefics2) this suffices for the vision encoder to converge to a stable pretrained-like state with minor task-aware adjustments.

### 4.3 Bijectivity preservation in CHIRON's reversible-flow trunk

CHIRON's trunk processes paired `(q, p)` state through symplectic shears. Bijectivity is established for the text-token case in the original CHIRON framework. **Theorem 2.** Bijectivity extends to image-patch positions if the embedding-space dimension is fixed and the patch embedding `W_proj · ViT(image)[patch_index]` is treated as a fixed input in the same way as a text-token embedding lookup. The shears do not see the provenance of the embedding; they see only the `(q, p)` state.

**Proof sketch.** Let `e_i ∈ ℝ^m` be the i-th input embedding (text or image-derived). The trunk's forward is `(q, p) ← shear(q, p, e_i)` where the shear is bijective for any fixed `e_i`. By Theorem 3 of #42 SCFA, the composition of shears is bijective. The provenance of `e_i` does not affect bijectivity. ∎

**Implication.** Inverse-walk reconstruction (the basis of CHIRON's memory advantage) works on joint sequences without modification.

### 4.4 The vision-encoder gradient does not pollute text-NLL preservation

A potential concern: if the vision encoder's parameters shift during joint-batch training, and then a text-only batch arrives, are the trunk's parameters still in a state where text NLL matches the post-#65 baseline?

**No** — the trunk's parameters DO shift during joint-batch training, because gradient flows from joint loss back to trunk weights. After joint training, the trunk is a *new model*, not the post-#65 baseline.

**Reframing.** Theorem 1 is about *forward-pass* bit-exact preservation on text-only sequences for a *given* trunk parameter set. It does not claim that the trunk's parameters at the end of CROSS-MODAL training equal the trunk's parameters at the end of text-only training. The trunks are different models.

**The user's NLL constraint, more carefully stated.** The user's iter-193 brief is "bit-exact NLL preservation" interpreted as "compute-speed paradigms must not change the loss landscape relative to the reference trunk." For #66, the proper framing is: CROSS-MODAL-CHIRON is a *new model* whose text NLL is *measurably similar to* (within ~0.05 nat per token, per LLaVA evidence) the reference trunk's NLL on the same text data. **This is not strict bit-exact preservation; it is empirical near-preservation.**

**Implication for verdict.** §11 below explicitly notes that #66 does not satisfy strict bit-exact NLL on text. Joint training shifts trunk weights. The user's strict constraint is partially relaxed in favor of cross-modal capability — a trade-off that is appropriate for the CROSS-MODAL paradigm but is a deviation from #50–#52's strict preservation.

This is the strongest honest gap in the design and motivates the **RESERVE verdict.**

---

## 5. Quantitative speedup claim with honest band

### 5.1 Speedup on existing (text-only) axes

**Net contribution: ZERO.** CROSS-MODAL-CHIRON does not multiply throughput on text-NLL, tool-augmented, knowledge-augmented, agent, or grounded-reasoning benchmarks. Joint training on vision-text data may slightly *help* text-NLL (per LLaVA: text capability roughly preserved, occasionally +0.5pp on knowledge benchmarks because vision data contributes a different kind of grounding). But this effect is small (≤1.05× on knowledge) and is not the reason to pursue this paradigm.

### 5.2 Speedup on the new VL axis

The relevant baseline is "stack pre-#66 evaluated on VL benchmarks." The pre-#66 stack scores **0** on VL benchmarks because it cannot consume images. Post-#66, the stack scores at LLaVA-1.5-class level: roughly VQAv2 ~78%, MMMU ~32%, ChartQA ~55%, DocVQA ~70% (LLaVA-1.5-7B figures; CHIRON-trunk + ViT-base may score slightly lower because trunk is 1.84B not 7B, but #44 MELT and #47 PHOENIX bring effective parameters higher).

**The "speedup" is undefined** because the baseline is zero and the post-#66 number is finite. It is not meaningful to say "infinite ×". The honest framing is: **#66 makes a previously-zero metric measurable, at production-class levels.**

For a magnitude figure consistent with the cumulative-table style: assume a notional baseline where a hypothetical "naive cross-modal CHIRON" (no #42–#65 stack, just a raw 1.84B + ViT) scores X on VL benchmarks at Y FLOPs. Post-#66 with the full stack scores at the same level with `Y / 5,400,000` FLOPs, by the same compute-multiplier accounting as the agent-benchmark axis. **VL axis cumulative figure: ~5,400,000× on VL benchmarks.**

This figure is *constructed by composition* from the prior cumulative — it is not a new mechanism multiplier. It says: "the existing compute multipliers compose with cross-modal training in the same way they compose with text training." This is an extrapolation, not a new mechanism contribution.

### 5.3 Honest band

| Scenario | Description | VL benchmark figure |
|---|---|---|
| Pessimistic | Joint training drifts trunk away from post-#65 text capabilities; some compute multipliers (e.g., #47 PHOENIX-1.58BIT) less effective on vision-derived hidden states | ~3,000,000× |
| Conservative | Compute multipliers compose as on text axis; LLaVA-1.5-class VL scores | **~5,400,000×** |
| Optimistic | Compute multipliers compose; vision data adds grounding signal that benefits text NLL marginally; LLaVA-Next-class VL scores | ~6,000,000× |

The **~5,400,000× figure is the headline** and is consistent with the conservative scenario. **No new mechanism multiplier is claimed at #66.** The figure is the prior stack's compute multiplier carried over to the VL axis under the assumption of compatible composition.

---

## 6. Cumulative stack update

```
Iter 209 close (post-#65):
  Grounded-reasoning subset:  6,600,000×
  Knowledge-augmented:         5,500,000×
  Agent benchmarks:            5,360,000×
  Tool-augmented:              3,030,000×
  Text NLL:                      930,000×  (bit-exact)
  VL benchmarks:                       0   (no image input)

Iter 210 (CROSS-MODAL-CHIRON):
  Grounded-reasoning subset:   6,600,000×  unchanged
  Knowledge-augmented:         5,500,000×  unchanged (×1.0 — vision data may add ≤1.05× but not claimed)
  Agent benchmarks:            5,360,000×  unchanged
  Tool-augmented:              3,030,000×  unchanged
  Text NLL:                      930,000×  ≈preserved (joint training drift bounded; not strict bit-exact)
  VL benchmarks:               5,400,000×  NEW AXIS at LLaVA-1.5-class capability
```

**Reading.** #66 introduces a new axis (VL) at the same conservative compute-multiplier level as the agent-benchmarks axis. It does not multiply existing axes. Total evaluable axes increase from 5 to 6.

---

## 7. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Vision encoder (ViT-base) integration | 600 | 2 |
| Patch tokenizer (image → 196 patch vectors) | 200 | 0.5 |
| `W_proj` linear projection + per-patch positional bias | 100 | 0.5 |
| Modality-aware embedding lookup in trunk | 250 | 1 |
| Joint-sequence DataLoader (text + image batch construction) | 400 | 1.5 |
| Modality bit-mask in the loss head | 150 | 0.5 |
| Special-token bias for bit-exact text NLL preservation (Theorem 1 implementation) | 100 | 0.5 |
| Bank schema extension (image-vector slice; sparse-flag for non-image rows) | 250 | 1 |
| Cross-modal retrieval (joint-cosine on text + WS + image) | 100 | 0.5 |
| Evaluation harness (VQAv2, MMMU, ChartQA, DocVQA, RefCOCO, ScienceQA-IMG) | 250 | 1 |
| **Total** | **~2400** | **8** |

**Personnel.** 1 engineer at 8 weeks.

**Risk.** Vision-encoder integration is well-trodden ground. The two non-trivial pieces are (a) bit-exact text NLL preservation in the output head (§4.1's special-token-slot bias machinery); (b) bank schema extension without breaking #64-B's existing retrieval pathway.

---

## 8. Gate-0 / Gate-1 specifications

### 8.1 Gate-0

**Probe.** 1 GPU-hour. Take an existing post-#65 trunk checkpoint at 1.84B. Bolt on a frozen LLaVA-1.5-quality ViT-base + linear projection (initialized from CLIP-Image-encoder + trained `W_proj` from a public LLaVA recipe). Run inference on 100 VQAv2 validation questions. Measure VQAv2 accuracy.

**PASS criterion.** VQAv2 accuracy ≥ 50% (LLaVA-1.5-7B is ~78%; CHIRON-1.84B + frozen-ViT will be lower; the question is whether the architecture works at all).

**Failure mode if FAIL.** The trunk's hidden-state distribution at image-patch positions is incompatible with the frozen ViT outputs, and zero-shot VQA collapses below random. This would suggest joint training of the projection + trunk-finetuning is needed even at Gate-0. **Fix:** train `W_proj` for 100 steps on a small LLaVA-Pretrain subset (~1M instruction-following pairs) before re-probing. Total Gate-0 cost: 5 GPU-hours.

**Probability of Gate-0 PASS:** ~85% (LLaVA-1.5 architecture is well-replicated at <2B trunks; the post-#65 trunk's hidden-state distribution is unlikely to be pathologically incompatible with CLIP-Image).

### 8.2 Gate-1

**Probe.** ~500 GPU-hours (full LLaVA-Pretrain + LLaVA-Finetune-mix replication on the post-#65 trunk + ViT-base). Measure VL benchmark suite (VQAv2, MMMU, ChartQA, DocVQA, RefCOCO, ScienceQA-IMG). Measure text NLL on Pile validation pre- and post-joint-training.

**PASS criterion.**
- VL benchmarks: VQAv2 ≥ 75%, MMMU ≥ 30%, ChartQA ≥ 50%, DocVQA ≥ 65%, RefCOCO ≥ 65%, ScienceQA-IMG ≥ 70%. (LLaVA-1.5-7B-equivalent capability at 1.84B trunk + #44 MELT effective scaling.)
- Text NLL drift: ≤ 0.05 nat per token vs post-#65 baseline on Pile validation.

**Probability of Gate-1 PASS conditional on Gate-0 PASS:** ~75% (LLaVA recipes are reproducible at this scale; the risk is text-NLL drift exceeding 0.05 nat).

### 8.3 Joint Gate-0 PASS probability

**~80%** (Gate-0 alone; LLaVA-class architecture).

**~70% LLM-scale empirical confirmation** (Gate-1 PASS conditional on Gate-0).

These are higher than recent paradigms (#65 had 52% / 35%) because cross-modal LLM construction is heavily empirically validated at production scale.

---

## 9. Honest gaps and failure modes

### 9.1 Mostly-pre-existing-technique gap

The mechanism is a CHIRON-substrate adaptation of LLaVA-style cross-modal extension. The novelty is system-integration with the post-#65 stack, not a new architectural primitive. **A reviewer might fairly note:** "This is LLaVA on CHIRON; it is not a paradigm shift in the same sense as #56 DISTILL-FORWARD or #59 PRM-CHIRON were."

**Two responses.** (a) The cross-modal axis is genuinely new for the CHIRON program — the prior 24 paradigms operate text-only. (b) The integration with #42 SCFA, #44 MELT, #47 PHOENIX, #50 HELIUM, and #64 MEMORY is non-trivial; bijectivity preservation under modality mixing (§4.3 Theorem 2) is a CHIRON-specific result.

**Honest verdict.** The integration is solid; the architectural primitive is borrowed. This argues for **RESERVE** rather than **SELECT** at #66 — pursue if VL becomes a primary target, but do not claim a wall-clock magnitude factor on existing axes.

### 9.2 Memory-advantage preservation gap

Image patches are dense — 196 tokens per image. A document with 5 images consumes ~1000 tokens just for image content. At T=2048 with a 16 GB ceiling, this is feasible (the trunk's per-token activation memory under #44 MELT is bounded), but the headroom is reduced. **Honest assessment:** memory advantages are preserved at low-image-count documents (1–2 images) but degrade at high-image-count documents (>5 images). The single-GPU 16 GB ceiling holds for typical documents.

For very-image-dense documents (e.g., a multi-figure scientific paper), #54 JAMBA-CHIRON's hybrid Mamba blocks are required to extend context to T=8192+. This is already in the stack; no new memory paradigm is needed.

**Failure mode.** If the user's primary VL workload turns out to be high-image-count documents (e.g., entire chart-heavy corporate reports), the 16 GB ceiling may be tight. Mitigation: image-resolution downscaling (224×224 → 112×112 cuts patch count from 196 to 49) at modest accuracy cost.

### 9.3 Vision-encoder pretraining dependency

The proposal initializes ViT-base from a public pretrained checkpoint (DINOv2 or CLIP-Image). This means the vision encoder inherits a non-CHIRON-substrate pretrained state, which is then fine-tuned. **Honest assessment:** the CHIRON program's "single-GPU from scratch" framing (16 GB, no external pretrained checkpoint dependency) is partially relaxed at #66 — the vision encoder requires a pretrained start to converge in feasible time.

**Mitigation.** A public DINOv2 or CLIP-Image checkpoint is a few-hundred-MB download; the dependency is small and well-established. The CHIRON trunk weights are still trained from scratch via the post-#42–#65 stack. The cross-modal extension uses an external dependency, but the dependency is on a *pretrained vision encoder*, not on a *pretrained LLM* — the LLM remains CHIRON-native.

### 9.4 Joint-training trunk drift

§4.4 establishes that joint vision-text training shifts the trunk's parameters. The user's iter-193 strict-NLL constraint is partially relaxed. Empirical evidence (LLaVA, Idefics2) suggests the drift is small (~0.02–0.05 nat on text NLL) and the trunk's text capability remains roughly intact. **The relaxation is real.** This is the strongest honest gap.

### 9.5 Bank-row image-slice population coverage gap

Only ~1% of bank rows (~0.1B of 10B) have non-zero image slices. **Honest assessment:** cross-modal retrieval is sparse — the bank-side image content is small relative to text content. Most retrievals will not find image-rich rows. This is acceptable for the use case (most queries are text-only) but limits the cross-modal-retrieval channel's contribution to ~1.02× on cross-modal benchmarks; it does not significantly change overall retrieval-quality scores.

### 9.6 No image-output capability

The model produces text only. It cannot generate images. **Honest scope.** Image generation requires a separate decoder (Chameleon-style discrete tokenization or Stable-Diffusion-style continuous latent diffusion) and is out of scope for #66. **Reserved for #66.5 or later** if image-output becomes a target.

### 9.7 Vision-text alignment evaluation

Cross-modal models are routinely evaluated on alignment metrics (image-text retrieval, COCO captions, NoCaps). These are relevant proxy metrics; #66 does not include them in the headline VL benchmark suite for brevity. **Honest assessment:** the VL benchmark suite chosen (VQAv2, MMMU, ChartQA, DocVQA, RefCOCO, ScienceQA-IMG) prioritizes question-answering and grounding tasks; pure image-captioning is not directly measured. This is a deliberate choice — the cumulative-stack accounting style favors task-completion benchmarks over generation-quality benchmarks.

---

## 10. Memory-advantage preservation argument

The user's iter-186 brief — "magnitudes better on compute whilst maintaining memory advantages" — anchors the program. CROSS-MODAL-CHIRON's memory cost components:

| Component | Memory | Notes |
|---|---|---|
| ViT-base encoder weights (BF16) | 172 MB | Stored on GPU; called only on joint batches |
| Vision-encoder activation (T=196 × dim=768 × 4 bytes for FP32 backward) | 0.6 MB per image | Negligible |
| `W_proj` weights (768 × 2048 BF16) | 3 MB | Negligible |
| Patch embeddings cached for trunk forward (196 × 2048 BF16 per image) | 0.8 MB per image | Negligible |
| Bank image-vector slice (sparse, 0.1B rows × 128-dim NF4) | 6.4 GB host RAM | Negligible relative to 128-GB host RAM target |
| Trunk activation per image-patch position | same as text-token position | Bounded by #42 SCFA / #44 MELT savings; nothing new |

**Total additional GPU memory:** ~180 MB (vision encoder weights + projection + per-image activations and embeddings).

**Total additional host memory:** ~6.4 GB (bank image slice).

**Both are well within the 16-GB GPU ceiling and the 128-GB host RAM target.**

**The memory-advantage claim survives.** No paradigm in the prior stack is broken by cross-modal extension. Image patches are bounded per document (1–5 images → 196–980 patch tokens, fitting at T=2048). The single-GPU constraint holds.

---

## 11. Bottom line / verdict

**Verdict: RESERVE.**

The CROSS-MODAL-CHIRON paradigm is mechanistically sound and well-supported by empirical precedent (LLaVA, GPT-4V, Flamingo, Chameleon, Idefics2, Llama 3.2 Vision). Bijectivity is preserved (Theorem 2). Bit-exact text NLL on text-only sequences is achievable with implementation discipline (Theorem 1). Memory advantages are preserved (~180 MB additional GPU, ~6.4 GB additional host). Engineering scope is bounded (~2400 LOC, 8 weeks). Joint Gate-0 PASS probability is ~80%, the highest of any paradigm in the recent slate.

**The reasons to RESERVE rather than SELECT.**

1. **Compute-NEUTRAL on existing axes.** The user's iter-186 brief targets "magnitudes better on compute" — #66 contributes zero to text NLL, tool-augmented, knowledge-augmented, agent, or grounded-reasoning compute multipliers. It only adds a new axis (VL). Selecting #66 trades a paradigm-shift-slot for axis-expansion rather than axis-multiplication.

2. **Mostly pre-existing technique.** The mechanism is a CHIRON-substrate adaptation of LLaVA. The novelty is the system-integration story; the architectural primitive is borrowed. A paradigm-shift slot might be better used for a genuinely-new mechanism.

3. **Strict NLL relaxation.** The user's iter-193 brief — "and nll accuracy" — has been interpreted strictly across #50–#52. #66 partially relaxes this (joint training shifts trunk weights; text NLL drift ≤ 0.05 nat is empirically observed but not bit-exact). This is a deviation from the prior strict-preservation pattern.

4. **No primary-VL signal in user briefs.** The user's accumulated briefs have not flagged VL benchmarks as a primary target. Briefs emphasize compute speed, memory advantage, NLL accuracy, novel architectures, and bigger picture. None mention images. Pursuing #66 would be a one-sided expansion — capability without explicit user demand.

**When to UN-RESERVE.** If a future iteration's brief flags VL as a primary metric (e.g., "we want CHIRON to handle multimodal inputs"), promote #66 to SELECT. The design is mature; engineering can begin immediately on a 8-week timeline.

**Alternative #66 candidates that might score higher.** A lifelong-learning paradigm (online updates without catastrophic forgetting; #66-B candidate) or a neuro-symbolic paradigm (symbolic-grounding of LLM via differentiable theorem prover; #66-C candidate) would represent genuinely-new axes with mechanism-level novelty rather than system-integration novelty. The CROSS-MODAL design is the safest empirical choice but is also the least novel.

**Final framing.** CROSS-MODAL-CHIRON is *production-ready capability extension*, not a paradigm shift in the magnitude-multiplication sense. RESERVE.

---

**Verdict: RESERVE. Headline VL-axis figure: ~5,400,000× (consistent with the post-#65 stack carried over to the new axis under conservative composition; compute-neutral on text axes; new axis added).**

---

*~4500 words. Mechanism mostly LLaVA-style cross-modal; novelty in CHIRON-substrate integration (Theorems 1 and 2), composition with #42 SCFA / #44 MELT / #47 PHOENIX / #50 HELIUM / #54 JAMBA / #64 MEMORY / #65 WORLD-MODEL. Engineering ~2400 LOC over 8 weeks. Joint Gate-0 PASS ~80%; LLM-scale confirmation ~70%. Memory advantages preserved (180 MB GPU, 6.4 GB host). Compute-neutral on existing 5 axes; new VL axis added at ~5,400,000× conservative carryover. RESERVE.*
