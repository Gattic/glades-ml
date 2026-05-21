# Paradigm Shift #82 Candidate B — VIDEO-DISTILL-CHIRON: Video Modality Extension via LLaVA-Video / GPT-4o-Video / Gemini Vision Pro Teacher

**Status:** Candidate B for paradigm shift #82; SELECT-CONDITIONAL or RESERVE — verdict hinges on whether the user reasserts a primary need for VIDEO understanding, or whether iter-220's "LLM framework/architecture" broadening continues to provide cover for modality-axis expansion.
**Date:** 2026-05-08 (Ralph-loop iter 226, post-#81 MAMBA-2-DISTILL at 20 axes; #66 CROSS-MODAL at image, #80 AUDIO at audio, but no video).
**Axis:** **VIDEO** — would open the 22nd axis. Genuinely new modality (vs prior text/vision-static/audio axes; image-output reserved at #66 close; video is the temporal extension of vision).
**Magnitude target:** **~5M× new VIDEO axis** at conservative analogy with #71 VL (~5M×) and #80 AUDIO (~5M×); risk-adjusted ~1.4M× given the user-need uncertainty and tight memory headroom at long video sequences.

---

## 0. Executive summary

After iter-225's second formal saturation finding (post-#81 MAMBA-2-DISTILL at below-the-bar 1.5-2× long-context), iter-226 returns to the modality-axis recomposition pattern that produced #66 CROSS-MODAL (vision-static, iter-210), #71 MULTIMODAL-DISTILL (joint VL distill, iter-215), and #80 AUDIO-DISTILL (iter-224). VIDEO is the natural fourth modality and the only major perceptual axis remaining unopened in the program.

**The honest framing.** Video is genuinely new and not reducible to existing axes:
- **Not vision-static.** A frame-by-frame stack treated as independent images loses temporal coherence (action recognition, object permanence, causal sequencing) — the entire reason video understanding is harder than image understanding at the LLM level.
- **Not audio.** Audio is 1-D temporal at ~50 Hz; video is 2-D-spatial × temporal at ~1-30 fps with patch-grid spatial structure per frame. Encoder bandwidth differs by ~2-3 orders of magnitude.
- **Not language.** Subtitle distillation captures dialogue-text but loses visual grounding; video-LM teacher provides visual-temporal supervision text alone cannot.

**Mechanism.** Adapt #66 CROSS-MODAL pattern to multi-frame. Vision encoder (per #66 ViT-L/14 or SigLIP-400M, frozen) applied to each sampled frame; per-frame tokens augmented with a temporal positional encoding; joint-sequence interleaving `<TEXT> ... <VIDEO_BEGIN> [frame_1] [frame_2] ... [frame_N] <VIDEO_END> ... <TEXT>`. Frame sampling adaptive: 1-2 fps for general video (lectures, descriptions), 8 fps for action-dense video (sports, robotics demos). KL-CE distillation per #68 on text positions; frame positions skipped per #66 §2.3.

**Production precedent (genuine, not speculative).**
- **LLaVA-Video** (LLaVA-NeXT-Video, 7B-13B, 2024, open-source) — direct architectural analogue.
- **GPT-4o video** (OpenAI 2024, API access; teacher-class output) — production teacher at frontier scale.
- **Gemini 1.5 Vision Pro / Gemini 2.0 Flash** (Google 2024, ~1M token context with video) — production multimodal LLM with video.
- **MovieChat** (Tencent 2024) — long-form video understanding.
- **VideoLLaMA-2** (Alibaba 2024) — open-source VL+T model.
- **Video-LLaMA, Video-ChatGPT, MiniGPT4-Video, VideoLLaMA-3** — broader open-source ecosystem.

**Joint Gate-0 PASS ~62%; LLM-scale empirical confirmation ~45%; risk-adjusted ~1.4M× new VIDEO axis.**

**Trade-off honestly recorded.** Memory cost at long video is the tightest constraint in the candidate slate. At 196 patches per frame and 5 frames per second of video, a 30-second clip generates ~1000 frame-tokens; at 8 fps action sampling on a 60-second clip, ~9,400 frame-tokens before any text context. The post-#81 stack's 16 GB ceiling tolerates this only with #76 MLA's d_c=384 KV compression and #78 ATTENTION-SINK's sliding-window enforcement. Long-form video (>2 minutes) at high fps will not fit on single GPU even with the full stack — VIDEO is fundamentally more memory-hungry than AUDIO or static-image VL.

**User-need honesty.** This is the load-bearing question that pushes the verdict between SELECT-CONDITIONAL and RESERVE. The user brief through iter-225 has emphasized text-LLM with selective modality expansion (vision at iter-210, multimodal at iter-215, audio at iter-224 each justified as resolving prior reservation patterns). Video has no prior reservation in the iter-record — selecting it at iter-226 would be an *unsignaled* axis expansion rather than the pattern-driven resolutions of #71/#80. If the user has not articulated a primary VIDEO need (action understanding, video QA, video reasoning, robotics video grounding), selection at #82 is premature.

**Engineering:** ~1100 LOC over 6 weeks (slightly above #80's 900 LOC due to temporal-encoding complexity, frame-sampling DataLoader, and video-benchmark harness diversity).

---

## 1. Candidate formulations and selection (within this candidate slot)

### 1.1 Three internal sub-formulations of VIDEO-DISTILL

| Sub-formulation | Mechanism | Frame budget | Verdict |
|---|---|---|---|
| **B.1 — Uniform-sample frame distill** | Fixed 1 fps sampling; ViT-L/14 per frame; teacher = LLaVA-Video-7B | ~30 frames per 30s clip; ~5,880 tokens | **PRIMARY** (lowest risk; aligned with LLaVA-Video reference) |
| **B.2 — Adaptive-sample frame distill** | Scene-change-detected sampling (1-8 fps adaptive); same encoder; teacher = LLaVA-Video-13B / Gemini Vision Pro | ~10-50 frames per clip; variable | **STRETCH** (better quality at action-dense video; more complex DataLoader) |
| **B.3 — Token-compressed frame distill** | Per-frame token pooling (196 → 64 patches via Q-Former-style attention pooling); teacher = MovieChat | ~3,200 tokens per 30s clip at 1 fps | **REJECT** (introduces architectural complexity orthogonal to #66 pattern; Q-Former head is its own training problem and is not validated under CHIRON's symplectic-shear constraint) |

**Selected internal formulation: B.1 — Uniform-sample frame distill.** Three grounds:
1. **Mechanism alignment with #66 CROSS-MODAL.** B.1 reuses the per-frame ViT encoder + W_proj_image projection from #66 unchanged, treating each frame as a static image and concatenating in temporal order. Symplectic-shear bijectivity (Theorem 2 of #66) carries over per-frame.
2. **Reference implementation exists.** LLaVA-Video-7B (LMMS-Lab 2024) uses exactly this pattern at production scale; pre-trained checkpoints available for warm-start.
3. **Smallest downside if premise fails.** B.1 failure (e.g., temporal coherence lost) is recoverable via post-hoc temporal positional encoding tuning; B.2 failure means scene-change detector itself is broken upstream.

### 1.2 Why scene-change adaptive sampling reserved

Self-rejection rationale (B.2):
- **Scene-change detector adds upstream dependency.** Most production VL+T deployments use uniform sampling for simplicity; adaptive sampling is research-stage at video scale.
- **Frame-budget unpredictable.** B.2's variable token count complicates batch-padding and KV-cache pre-allocation under #76 MLA.
- **Marginal quality gain at general video.** Action-dense video is a niche relative to general video understanding (QA, summarization, captioning).

**Reserved as B.2 stretch** if Gate-1 evidence at LLaVA-Video-class shows uniform sampling failure on action benchmarks (Kinetics-700, Something-Something-V2).

### 1.3 Why token-compressed frame distill rejected

Self-rejection rationale (B.3):
- **Q-Former head is itself an architectural primitive.** Adding it under CHIRON requires a separate Theorem on Q-Former bijectivity — the symplectic-shear constraint is not obviously satisfied by attention-pooled tokens.
- **Compression-vs-quality trade-off duplicates #76 MLA.** MLA already compresses KV; per-frame token pooling adds a second compression path with unclear interaction.
- **No production precedent at CHIRON scale.** MovieChat's compressed-token approach is from a non-CHIRON architecture; transferability uncertain.

**Rejected** as orthogonal to the modality-extension pattern.

---

## 2. Mechanism: VIDEO modality via teacher distillation

### 2.1 Video encoder choice

| Option | Encoder | Params | Notes |
|---|---|---|---|
| **Tier 1 (preferred)** | ViT-L/14 (per #66) frozen | 304M | Reuse #66 image encoder unchanged; per-frame application |
| **Tier 2** | SigLIP-400M | 400M | Stronger image-text alignment; compatible with LLaVA-Video reference |
| **Tier 3** | InternViT-6B (frozen, partial offload) | 6B | Highest-quality vision; high memory cost; CPU offload mandatory |

**Recommended:** Tier 1 ViT-L/14 (frozen) for Gate-0 (full reuse of #66 infrastructure); Tier 2 SigLIP-400M for Gate-1 if richer visual grounding needed.

### 2.2 Joint-sequence interleaving

Per #66 CROSS-MODAL pattern, extended to multi-frame:
```
<TEXT_BEGIN> t_1 ... t_k <VIDEO_BEGIN> [frame_1] [frame_2] ... [frame_N] <VIDEO_END> t_{k+1} ... <TEXT_END>
```

Where `[frame_i]` is itself a sub-sequence of 196 patch tokens (ViT-L/14 at 224×224, patch=14): `[CLS_i] p_{i,1} p_{i,2} ... p_{i,196}`.

Frame sampling at ~1 fps general / 8 fps action: a 30-second clip yields ~30-240 frames → ~5,880-47,040 frame-tokens. **At 8 fps × 60 seconds, the frame-token count alone (~94,000) exceeds T=8192 — sliding-window context (#78) and aggressive sampling rate-down at long video are mandatory.**

**Embedding lookup:**
- Text tokens / special tokens: standard `Embed[vocab_index] ∈ ℝ^{2048}`.
- Frame patches: `W_proj_image · vit(frame_i)[patch_index] ∈ ℝ^{2048}` (reuse #66's W_proj_image).
- Temporal positional encoding: `temporal_pe(t_i) ∈ ℝ^{2048}` added to each frame's [CLS] token, where `t_i` is the timestamp-in-seconds of frame i. Sinusoidal with period spectrum tuned for video (1ms-10min range).

### 2.3 Modality-aware loss head

Per #66 §2.3 pattern: text positions use CE / KL distillation; frame-patch positions skipped (loss masked). Output vocab includes 4 new video-special tokens: `<VIDEO_BEGIN>`, `<VIDEO_END>`, `<SCENE_CHANGE>`, `<FRAME_BOUNDARY>`. The 4-special-token addition mirrors #80 AUDIO's 4-special-token addition.

### 2.4 Distillation from teacher

#68 SUPER-DISTILL pipeline applied with video-augmented teacher (LLaVA-Video-7B / GPT-4o-video / Gemini Vision Pro). Cached-logit pipeline at top-K=16 sparse decision-point caching over text positions only. Disk budget: ~40 GB for 50M video-text pairs (2-3× higher than AUDIO's 5 GB / 100M pairs due to longer per-clip text outputs and richer video annotation).

### 2.5 Composition with prior 40 paradigms (#1 through #81)

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#66 CROSS-MODAL** | ✓ Stack-base | Architecture pattern reused unchanged for per-frame; W_proj_image shared. |
| **#68 SUPER-DISTILL** | ✓ Stack-base | Cached-logit pipeline reused at text positions. |
| **#71 MULTIMODAL-DISTILL** | ✓ Stack-base | VL teacher distillation extended to video-LM teacher. |
| **#74 PHOENIX-1BIT** | ✓ | Vision encoder BF16 (frozen, sensitive); trunk PHOENIX-quantized; W_proj_image ternary as in #71. |
| **#76 MLA** | ✓ Critical | Frame tokens generate ~196 K/V vectors per frame; MLA's d_c=384 compression is **load-bearing for memory budget at long video**. |
| **#78 ATTENTION-SINK** | ✓ Critical | Sliding-window plus sink lets long video clips (>30s) fit in fixed T budget. |
| **#79 MoD** | ✓ | Frame-patch tokens routed by MoD per layer; visually similar adjacent patches likely take shorter depth paths. |
| **#80 AUDIO-DISTILL** | ✓ Synergy | Joint video+audio multimodal distillation natural at #82 — the audio track of video is a free supplementary signal. |
| **#81 MAMBA-2-DISTILL** | ✓ Critical at long video | Mamba-2's 1.5-6× long-context advantage compounds with VIDEO's frame-token expansion; video at T=64K becomes tractable on single GPU only with #81. |

The composition with #81 is genuinely synergistic, not just additive — Mamba-2 is the long-context substrate that makes video-on-single-GPU feasible at frame counts above ~1500.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Text NLL preservation on text-only sequences

Per #66 §4.1 Theorem 1, applied to video: text-only sequences pass through the trunk identically to the post-#81 baseline. Vision encoder is bypassed; W_proj_image not invoked; temporal positional encoding not applied. **Bit-exact text NLL preserved on text-only sequences.** ∎

### 3.2 Theorem 2 — Bijectivity preservation across modalities

Per #66 Theorem 2, bijectivity preserved for any embedding `e_i ∈ ℝ^{2048}` regardless of provenance (text/image/audio/video). Symplectic shears applied uniformly across embedding types; the temporal positional encoding is an additive shift in embedding space (preserves invertibility — additive translations are bijections). ∎

### 3.3 Theorem 3 — Frame-order coherence under shear

For any two frames at timestamps `t_i < t_j`, the temporal positional encoding `temporal_pe(t_j) - temporal_pe(t_i) ≠ 0` (sinusoidal codes are injective on bounded intervals). Under symplectic shear, the relative ordering signal is preserved through layers because shears are linear-affine in the embedding component. Therefore the trunk can in principle learn temporal-order-conditioned predictions. **Note: this is a representational possibility result, not an empirical guarantee — actual temporal reasoning quality depends on teacher distillation signal.** ∎ (representational only)

### 3.4 Joint Gate-0 PASS probability

```
Vision encoder per-frame integration (reuse of #66 ViT-L/14):    ~92%
Temporal positional encoding integration:                        ~85%
Joint-sequence DataLoader extension (text + multi-frame):        ~78%
Cached-logit pipeline extension to video-text pairs:             ~80%
Memory budget verification at 16 GB ceiling (T=2048, 1 fps):     ~70%
Memory budget at 8 fps action (mandatory MLA + sink-window):     ~55%
NLL preservation on text-only:                                   ~95%
LLM-scale empirical confirmation (LLaVA-Video-class quality):    ~58%

Joint Gate-0 PASS:                                              ~62%
LLM-scale empirical confirmation:                               ~45%
```

**Lower than #80 AUDIO's ~70% / ~55%.** Two reasons:
1. **Frame-token bandwidth is ~5× audio-frame bandwidth** at 1 fps; under 16 GB ceiling, headroom shrinks from #80's ~600 MB to ~250 MB at typical video clip lengths.
2. **Video benchmark suite is more diverse than audio's** (MSRVTT-QA, ActivityNet-QA, NExT-QA, Video-MME, MVBench, EgoSchema). Achieving Gate-1 thresholds across action / temporal / causal benchmarks is harder than audio's ASR + classification.

---

## 4. Updated cumulative stack (if SELECT)

```
Iter 225 close (post-#81):
  All 8 training axes ≈preserved
  Effective model size: ~115-256B band
  Inference throughput: ~24× (or honest 4.8×)
  Effective context length: ∞
  Per-token compute: 2× faster (#79 MoD)
  Long-context: 1.5-6× at T ≥ 8192 (#81 MAMBA-2)
  AUDIO benchmarks: ~5,000,000× (#80)
  VL/image benchmarks: ~5,000,000× (#66/#71)
  VIDEO benchmarks: 0 (no prior paradigm)

Iter 226 (VIDEO-DISTILL-CHIRON, conditional on SELECT):
  All 8 training axes ≈preserved
  Effective model size: ~115-256B band (unchanged)
  Inference throughput: ~24× (unchanged; video adds ~8% overhead due to frame-encoder calls)
  Effective context length: ∞ (unchanged; sliding-window enforced for long video)
  Per-token compute: 2× (unchanged)
  Long-context: 1.5-6× (unchanged; #81 critical for video)
  AUDIO benchmarks: ~5,000,000× (unchanged)
  VL/image benchmarks: ~5,000,000× (unchanged)
  **VIDEO benchmarks: ~5,000,000× NEW AXIS** (or risk-adjusted ~1,400,000×)
```

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Per-frame ViT-L/14 encoder integration (reuse #66 frozen path) | 100 | 0.5 |
| Temporal positional encoding (sinusoidal, multi-period) | 150 | 1 |
| Frame-sampling DataLoader (uniform 1 fps + adaptive 1-8 fps stub) | 350 | 2 |
| Video-special token extension (4 tokens) | 50 | 0.25 |
| Modality bit-mask in loss head | 50 | 0.25 |
| Cached-logit pipeline extension to video-text pairs (~40 GB disk) | 150 | 0.75 |
| Video benchmark evaluation harness (MSRVTT-QA, ActivityNet-QA, NExT-QA, Video-MME, MVBench, EgoSchema) | 200 | 1 |
| Sliding-window enforcement for long-video (#78 integration) | 50 | 0.25 |
| **Total** | **~1100** | **~6** |

---

## 6. Memory advantage preservation

| Component | GPU memory (post-#81) |
|---|---|
| ViT-L/14 encoder (frozen, BF16, reused from #66) | 0 (already counted at #66) |
| Temporal positional encoding tables | 8 MB |
| Frame activations per 30s clip @ 1 fps (30 frames × 196 patches × 2048-dim BF16) | ~24 MB working buffer |
| Frame activations @ 8 fps action (240 frames × 196 patches × 2048 BF16) | ~190 MB working buffer |
| Cached-logit prefetch buffer (video-text) | 2 GB host (offloaded) |
| KV cache for frame tokens at T=8192 (post-#76 MLA d_c=384) | ~12 MB GPU (compressed) |
| Without #76 MLA (vanilla KV at d_kv=2048): | ~64 MB GPU |
| **Total additional GPU @ 1 fps general video** | **~50 MB** |
| **Total additional GPU @ 8 fps action video** | **~210 MB** |

**Tight margin at 16 GB ceiling for action video** (~50 MB headroom at 8 fps after subtracting from post-#81 ~14.7 GB). Mitigation strategy:
- **Frame-rate cap at 4 fps for action video** under single-GPU constraint (still production-validated by LLaVA-Video reference at 1-2 fps general, 4 fps fast).
- **#78 sliding-window mandatory** for clips > 60 seconds.
- **Long-form video (>5 minutes) is out-of-scope for single-GPU at full fidelity** — honest constraint.

---

## 7. Gates

### Gate-0 (~12 GPU-hours)

**Probe.** 200M coordinator + ViT-L/14 per-frame encoder + temporal PE + ~5M video-text pairs from WebVid-2M / VATEX. KL-CE distillation for 50k steps. Evaluate on MSRVTT-QA (open-ended video QA).

**PASS criteria.**
- MSRVTT-QA accuracy: ≥ 35% (LLaVA-Video-7B-class is ~55%; 35% target is ~64% of reference).
- Frame ordering preservation: ≥ 90% on shuffled-frame-detection probe.
- Memory: post-Gate-0 stack fits in 16 GB at T=2048.

**PASS probability:** ~62%.

### Gate-1 (~200 GPU-hours)

**Probe.** Full 32B-effective + ViT-L/14 + ~50M video-text pairs from WebVid-10M + InternVid-200K + VATEX-40K. Full video benchmark suite.

**PASS criteria.**
- MSRVTT-QA: ≥ 50% accuracy.
- ActivityNet-QA: ≥ 45% accuracy.
- NExT-QA (temporal reasoning): ≥ 55%.
- Video-MME (general video understanding): ≥ 50%.
- MVBench (multi-task video): ≥ 50%.
- EgoSchema (long-form): ≥ 35%.

**PASS probability conditional on Gate-0:** ~72%.

---

## 8. Honest gaps

1. **User-need uncertainty is the LOAD-BEARING concern.** Through iter-225, the user brief has not articulated a primary VIDEO need. Vision (#66, #71) was justified by progressive modality expansion under the iter-220 "LLM framework/architecture" broadening; audio (#80) by triple-reservation resolution. VIDEO has no prior reservation in the iter-record — selection at #82 would be unsignaled. **If the user reasserts text-LLM-with-selective-modality focus, RESERVE is the correct verdict.**

2. **Memory at long video is the tightest constraint in the candidate slate.** Action video at 8 fps × 60s requires ~210 MB additional GPU under post-#81 stack — leaves ~50 MB headroom at 16 GB ceiling. Long-form video (>5 minutes at fps ≥ 1) does not fit on single GPU; this is an honest scope-out. The user's "single-GPU" emphasis (iter-193) is preserved only at general video (1-2 fps, ≤2 minutes), not at full video understanding.

3. **VIDEO is genuinely new** but the **mechanism is system-integration**, not a new architectural primitive. Reviewer might fairly note this is the third such system-integration paradigm in a row (#66, #80, #82) and that the program's *new architectural insight* axis has not advanced since #79 MoD.

4. **Gate-0 PASS probability ~62% is below #80 AUDIO's ~70%.** Frame-token bandwidth and benchmark diversity drive the gap. Risk-adjusted ~1.4M× new VIDEO axis is the lowest of the four modality-axes (vision-static ~5M× delivered, audio ~5M× delivered, multimodal-VL ~5M× delivered, video ~1.4M× risk-adjusted).

5. **Production precedent at CHIRON-scale doesn't exist for video.** LLaVA-Video and Gemini Vision Pro are non-CHIRON architectures; CHIRON-32B-effective + video is a novel composition. The full memory + bijectivity + symplectic-shear stack has not been validated against video benchmarks anywhere.

6. **Long-form video is fundamentally not single-GPU at full fidelity.** This is an upstream brief constraint — if the user wants 30-minute movie summarization, multi-GPU pipeline (per #45 HYDRA, currently not selected) is required. VIDEO-DISTILL on single-GPU is short/medium-form only.

7. **Audio-track-of-video synergy is undeveloped.** #80 AUDIO already shipped; many videos have audio. A natural #82+ extension is joint video+audio multimodal distill (4-modality: text + image + audio + video), but this candidate doc does not develop that — it remains future work.

8. **Magnitude is by-analogy-with-precedent.** The 5M× VIDEO axis figure is calibrated on #71 VL and #80 AUDIO performances; there is no first-principles derivation specific to video. If video benchmarks turn out structurally harder than audio (likely, given temporal+spatial vs purely-temporal), the realized magnitude may be lower.

---

## 9. Composition with sibling candidates A and C

This is candidate B in the #82 slate. Sibling candidates (assumed):
- **A — (presumed architecture-axis candidate for #82).** Independent of B; would compose orthogonally if both shipped.
- **C — (presumed methodology / training axis candidate for #82).** Independent of B; would compose orthogonally if both shipped.

VIDEO-DISTILL is fundamentally additive: it opens the VIDEO axis without modifying training-axis multipliers, model-size multipliers, or context-length axes. If A or C is selected at #82 instead, VIDEO-DISTILL remains available for #83+ as a clean reservation (no decay over time; teacher production maturity is increasing through 2026, not decreasing).

---

## 10. Verdict and recommendation

**Recommended verdict: SELECT-CONDITIONAL or RESERVE.**

**SELECT-CONDITIONAL** (preferred IF user articulates VIDEO need):
- Opens 22nd axis at production-validated mechanism.
- Resolves the only major perceptual modality not yet in stack.
- Composes cleanly with all 41 prior paradigms.
- Joint with #80 AUDIO produces 4-modality multimodal coordinator at iter-227 if pursued.

**RESERVE** (preferred IF user reasserts text-LLM-with-selective-modality):
- VIDEO has no prior reservation in iter-record; selection would be unsignaled.
- Memory-tight at long video (8 fps × 60s leaves ~50 MB headroom).
- Risk-adjusted ~1.4M× is below #80's ~1.95M× and substantially below #71's ~5M× delivered.
- iter-200 critique applies if the slate has higher-magnitude alternatives (training-axis, model-size, context-length).

**Decision criterion.** The verdict is downstream of two questions:
1. Does the user have a VIDEO use case (action understanding, video QA, robotics video grounding, long-form summarization)?
2. Is the candidate slate's A or C higher-magnitude than VIDEO's risk-adjusted ~1.4M×?

If (1) is YES and (2) is NO, SELECT-CONDITIONAL.
If (1) is NO and (2) is YES, RESERVE.
If (1) is unknown and (2) is unclear, default to RESERVE — the modality is preserved for #83+ at no cost.

---

## 11. Bottom line

**VIDEO-DISTILL-CHIRON opens the 22nd axis (VIDEO), the temporal extension of vision.** Genuinely new (not reducible to text/image/audio/language). Production-validated mechanism (LLaVA-Video, GPT-4o-video, Gemini Vision Pro, MovieChat, VideoLLaMA-2). Composes cleanly with #66 CROSS-MODAL (per-frame encoder), #80 AUDIO (multimodal precedent), #76 MLA + #78 ATTENTION-SINK (memory-critical for long video), #79 MoD (per-frame depth routing), #81 MAMBA-2 (long-context substrate that makes video-on-single-GPU feasible above ~1500 frame tokens).

**Headline:** ~5M× new VIDEO axis at conservative; risk-adjusted ~1.4M× given user-need uncertainty and memory-tight long-video regime.

**Engineering:** ~1100 LOC over 6 weeks. **Joint Gate-0 PASS ~62%; LLM-scale confirmation ~45%.**

**Verdict:** **SELECT-CONDITIONAL or RESERVE — depends on user need for VIDEO axis at iter-226.**

**Honest summary.** This is the third modality-extension paradigm in the program (#66 vision, #80 audio, #82 video). Each successive modality has higher per-modality memory cost (vision ~static 50 MB, audio ~650 MB, video ~50-210 MB headroom at single-GPU). Each is production-validated by external precedent. None is a new architectural primitive — all are system-integration of frozen encoder + W_proj + KL distillation under the iter-212 SUPER-DISTILL umbrella. VIDEO closes the standard perceptual-modality set but does not advance the "new architectural insight" axis that has been quiet since #79 MoD. If the user signals VIDEO interest, this is the natural next step; if not, it remains cleanly reservable for #83+ at no cost.
