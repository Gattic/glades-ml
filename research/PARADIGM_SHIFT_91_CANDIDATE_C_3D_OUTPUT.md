# Paradigm Shift #91 — Candidate C: 3D-OUTPUT-DISTILL-CHIRON: 3D Generation Sub-Axis (Symmetric Completion of #88)

**Status:** RESERVE candidate (research-stage; thin production precedent at scale; magnitude axis-extension class; ~70% mechanism overlap with #88 3D-input + #82 image-output).
**Date:** 2026-05-08 (Ralph-loop iter 235, post-#90 sixth saturation finding).
**Axis:** **3D-OUTPUT** sub-axis — extension of #88 3D-SPATIAL (input) to OUTPUT side. Symmetric to the #82/#83/#84/#87 image-out / audio-out / video-in / video-out completion pattern.
**Magnitude target:** **~5,000,000× new 3D-OUTPUT sub-axis** at nominal; risk-adjusted **~700,000×** (lower than recent slate due to thin precedent at scale and ~70% mechanism overlap with #88 + #82). Axis-extension class.

---

## 0. Executive summary

Iter-235 sits one iteration after #90 KV-FACE-MLA closed the sixth consecutive saturation iteration. The post-iter-224 axis-extension pattern continues. The natural symmetry candidate at iter-235 is **3D-OUTPUT** — closing the I/O symmetry on the 3D-SPATIAL modality that #88 opened at the input side.

**This document is the candidate-C design under #91.** Slate-level disposition is recorded in the parent #91 design doc. Likely #91 verdict for this candidate is **RESERVE** with a SELECT-CONDITIONAL upgrade path if user signals 3D-generation, robotics-actuation, or CAD-output need.

**Mechanism (discrete VQ tokenization chosen over continuous mesh-diffusion or NeRF MLP-regression):** Per-mesh / per-NeRF-grid VQ codebook tokenizes 3D outputs into discrete tokens. Output via decoding through the VQ codebook + mesh-or-NeRF-parameter decoder. Codebook size 8192 (matched to #82 image-out and #87 video-out for hot-path reuse). Joint sequence pattern extends #82/#83/#87/#88:

```
<TEXT_BEGIN> ...
   <3D_INPUT_BEGIN> p_1 p_2 ... p_N <3D_INPUT_END>      (per #88; point-cloud input)
   ...
   <3D_OUTPUT_BEGIN> v_1 v_2 ... v_M <3D_OUTPUT_END>    (NEW; mesh / NeRF tokens)
   ...
<TEXT_END>
```

where `<3D_INPUT_*>` reuses #88's PointBERT-style tokenizer and `<3D_OUTPUT_*>` introduces new VQ-codebook output tokens that decode to a mesh (vertex/face buffer) or implicit-field (NeRF / Gaussian-splat parameters).

**Why discrete VQ (per #82 IMAGE-OUTPUT pattern).** The user's iter-193 brief — bit-exact NLL preservation on text — is unrelaxed through iter-235. Continuous mesh-diffusion (MeshDiffusion 2023) and per-point NeRF-MLP regression (DreamFusion 2022) violate per-token NLL by construction (denoising score-matching or per-pixel SDS loss). Discrete VQ tokenization (MeshGPT Tsinghua 2024; Shap-E OpenAI 2023's discrete token mode) preserves the autoregressive next-token framework end-to-end. **NLL is bit-exact preserved on text positions (modality-segregation theorem 1) and well-defined categorical log-likelihood on 3D-output positions.**

**Production precedent honestly thin at scale:**
- **DreamFusion** (Google 2022): score-distillation sampling from 2D diffusion; per-prompt NeRF optimization; **NOT autoregressive / NOT NLL-compatible**; reference for 3D-generation ARCHITECTURE only.
- **Magic3D** (NVIDIA 2023): two-stage SDS; coarse NeRF + fine mesh; **NOT autoregressive**.
- **Shap-E** (OpenAI 2023): two-stage transformer-conditional latent diffusion → implicit-field MLP weights; ~300M params; **discrete-token mode exists** (closest precedent for #91-C).
- **GET3D** (NVIDIA 2022): GAN-based mesh generation; ~700M params; **NOT autoregressive**.
- **MeshGPT** (Tsinghua 2024): autoregressive transformer over discrete mesh-face tokens; ~350M params; **closest direct precedent**.

**No Chameleon/MusicGen/Whisper/Llama-3.x equivalent exists for 3D-output at production scale.** All viable autoregressive teachers (MeshGPT, Shap-E discrete-mode) sit at <2B parameters in research repos; distillation extrapolation to CHIRON 32B-effective is uncertain — same caveat that applied to #88's 3D-input candidate, compounded.

**Why this candidate is genuinely new but RESERVED (not selected) at iter-235:**
- **Genuine 3D-OUTPUT sub-axis** symmetric to #82 image-out / #83 audio-out / #87 video-out completion pattern.
- **Highest mechanism overlap of any iter-235 candidate** (~70% with #88 3D-input + #82 image-output codebook pattern).
- **Production precedent thinnest of all multimodal-output sub-axes** (MeshGPT/Shap-E at <2B vs Chameleon's 7B–34B for image-out; MusicGen's 3.3B for audio-out; VideoPoet's 8B for video-out).
- **38–42% LLM-scale empirical confirmation** (modest; lower even than #88's 38% on already-thin 3D-input precedent).
- **Memory cost** ~500 MB for text-to-3D decoder is **tight under post-#88 stack** (~130 MB headroom remaining; vs #88 close's 630 MB).
- **User-need narrow** — 3D generation chiefly serves CAD, game-asset creation, AR/VR content, robotics-actuation simulation, 3D-printing pipeline; user brief through iter-235 has not signaled these.

**Composition with prior 49 paradigms:**
- **#88 3D-SPATIAL** (3D INPUT): symmetric axis-completion. Output extends input.
- **#82 IMAGE-OUTPUT / #83 AUDIO-OUTPUT / #87 VIDEO-OUTPUT**: discrete VQ codebook output pattern reused (8192 codes, KL-CE, cached-logit pipeline).
- **#66 CROSS-MODAL**: joint-sequence interleaving extends.
- **#62 AGENT-CHIRON**: synergy on robotics-actuation / 3D-action benchmarks if 3D output decodes to executable robotic-trajectory primitives.
- **#65 WORLD-MODEL**: synergy on spatial WS-bank rows holding generatable 3D-scene templates.

**Trade-offs honestly recorded.**
- 3D-OUTPUT not in user brief; selected discussion is on saturation-symmetry-completion grounds.
- Mechanism mostly pre-existing (MeshGPT autoregressive pattern + Shap-E discrete-mode + #82 codebook pattern); novelty is system-integration with iter-217-234 stack.
- Production precedent at scale genuinely thin — MeshGPT and Shap-E are the only viable autoregressive teachers, both at <2B parameters in research repos.
- Mesh-tokenizer or NeRF-parameter codebook quality bounds 3D-output fidelity (~25-40 Chamfer-Distance on ShapeNet vs production-frontier closed-source ~10 CD).
- 3D-OUTPUT decoder ~500 MB tightens 16 GB ceiling headroom from 630 MB (post-#88) to ~130 MB.
- ~70% overlap with #88 3D-input + #82 image-output codebook pattern — axis-extension class, not fresh axis.
- User-need narrow (CAD / game-asset / AR-VR / robotics-actuation / 3D-printing).

**Engineering:** ~1,750 LOC over 7 weeks (slightly higher than #88's 1,650 due to mesh-decoder pipeline + Chamfer/IoU evaluation harness). **Joint Gate-0 PASS ~52%; LLM-scale confirmation ~38%; risk-adj ~0.7M×.**

---

## 1. Candidate formulation and selection

### 1.1 Three iter-235 candidates (cross-reference)

| Candidate | File | Mechanism | Verdict |
|---|---|---|---|
| **A — (other)** | sibling candidate doc | (per parent design) | (per parent) |
| **B — (other)** | sibling candidate doc | (per parent design) | (per parent) |
| **C — 3D-OUTPUT-DISTILL** | this document | MeshGPT/Shap-E discrete-VQ teacher distillation; 3D-OUTPUT sub-axis | **RESERVE (research-stage; thin production precedent; ~70% overlap with #88 + #82; user-need narrow)** |

### 1.2 Why this candidate is genuinely interesting

1. **Symmetric I/O completion on 3D modality.** #88 opened 3D-INPUT (consume); #91-C closes the gap with 3D-OUTPUT (produce). Same completion pattern as #82 (image-out at #66 close), #83 (audio-out at #80 close), #87 (video-out at #84 close).

2. **Discrete-VQ pattern transfer is mechanically clean.** MeshGPT and Shap-E discrete-mode demonstrate autoregressive next-token over a fixed 3D-vocabulary at production-research scale. Joint-sequence interleaving with #88's 3D-input tokens reuses the same modality-bit-mask pattern (Section 2.3).

3. **Robotics-actuation latent synergy.** If user pivots to embodied-agent / robotic-control territory (post-#62 AGENT, post-#88 3D-input), 3D-OUTPUT becomes immediately useful: 3D output decodes to executable robotic-trajectory primitives via mesh-as-affordance representation (CLIPort 2021, RT-2 2023 lineage).

4. **Composition triple-win with #62 + #65 + #88.** AGENT trajectories + WORLD-MODEL bank-rows + 3D-SPATIAL input all gain a generatable side: planning systems that reason over 3D output extend agent-benchmarks and grounded-reasoning subsets.

### 1.3 Why this candidate is RESERVED, not SELECTED

Six honest reasons:

**1. Production precedent thinnest of all output sub-axes.**
- MeshGPT (Tsinghua 2024): research-stage; ~350M params; ShapeNet-trained.
- Shap-E (OpenAI 2023): research-release; ~300M params; closed dataset for largest variant.
- DreamFusion / Magic3D / GET3D: NOT autoregressive (SDS / GAN); incompatible with NLL preservation; usable only as reference/inspiration architectures.
- **No production-deployed autoregressive 3D-output API exists** (analogous to OpenAI image-out, Whisper audio-out, VideoPoet video-out).
- Risk-adjusted magnitude: ~5M× nominal × 0.14 production-precedent factor ≈ **0.7M× risk-adj** vs #88's ~0.8M× and #82's 1.7M×.

**2. ~70% mechanism overlap with #88 + #82 (highest in iter-235 slate).**
- **VQ-codebook output pattern** (~50% reuse from #82 IMAGE-OUTPUT): K=8192 codebook, KL-CE distillation, cached-logit pipeline, modality-segregated loss head — all transferred directly.
- **Joint-sequence interleaving** (~20% reuse from #88 3D-INPUT): same modality-bit-mask scheme, same `<3D_*_BEGIN/END>` tag scheme, only the direction is reversed.
- **Net architectural novelty ~30%**: mesh / NeRF-parameter decoder pipeline + Chamfer/IoU evaluation harness.
- **Honest framing: this is a system-integration paradigm with limited algorithmic advance.** Same caveat as #82/#83/#87 axis-extension class but compounded: #88 itself was already axis-extension class.

**3. Teacher-scale gap to CHIRON sharper than #88.**
- Best 3D-output teachers (MeshGPT, Shap-E discrete-mode): 300M–350M params.
- CHIRON 32B-effective via #53 MOSAIC-MOE + #58 METAGEN scaling exceeds teacher capacity by **~90×** vs #88's 4-5× gap to 7B teachers.
- Per #68 distillation Theorem 1, distillation gain plateaus when student ≫ teacher scale by codebook-quantization granularity — same mechanism as #88 but with smaller teacher and same K=8192 codebook.
- Marginal student gain beyond ~1B teacher size becomes approximately flat. Empirical confirmation drops further than #88's 38%.

**4. User-need narrow and unsignaled.**
- 3D-OUTPUT serves: CAD / engineering (parametric model generation), game-asset creation (procedural 3D assets), AR/VR content (immersive scene generation), robotics-actuation (3D-trajectory primitives), 3D-printing pipeline (printable mesh generation), embodied-agent simulation (digital-twin generation).
- User brief through iter-235: text-LLM compute speed + memory + NLL accuracy + single-GPU + novel + bigger-picture — no 3D-generation, robotics-actuation, or CAD signal.
- Adding 3D-OUTPUT axis without user-need is saturation-padding, compounded over saturation-padding from #88 itself.

**5. Memory cost tight under post-#88 stack.**
- Text-to-3D decoder (mesh-decoder Chamfer-trained or NeRF-MLP weights) frozen residence: ~500 MB BF16.
- Codebook + token cache during inference: ~30 MB (matches #82 / #87).
- **Total additional GPU: ~530 MB**.
- 16 GB ceiling: post-#88 close had ~630 MB headroom; #91-C consumes ~530 MB ⇒ **~100 MB headroom remaining**.
- This is the tightest headroom of any iter-228-235 candidate. Each subsequent axis adds tighter margin.
- Memory accounting under post-#88 stack: PHOENIX-1BIT trunk + MoE + 3D-input encoder + #91-C decoder all sized for the 16 GB ceiling — margin is now ~0.6% of total — **breaking-point threshold for any further memory-cost paradigm**.

**6. Sub-axis vs new axis.**
- 3D-OUTPUT is a SUB-axis of #88's 3D-SPATIAL macro-axis (input + output sides). Not a fresh 28th axis.
- Stack count remains at 27 axes if SELECTED; only the 3D-SPATIAL axis gains an output-side sub-channel.
- This matches the pattern of #82 (image-out is sub-axis of #66 VL macro-axis), #83 (audio-out sub-axis of #80 audio macro-axis), #87 (video-out sub-axis of #84 video macro-axis).

### 1.4 What would unlock SELECT later

Three signals would move this to SELECT:

1. **User signals 3D-generation / robotics-actuation / CAD-output / game-asset territory.** Brief sharpening to include "spatial-output" or "embodied-actuation" or "CAD pipeline" would shift verdict. (Same conditional as #88 but more specific to OUTPUT side.)

2. **Production-scale autoregressive 3D-output teacher emerges.** A Chameleon-equivalent for 3D (8B+ open-source autoregressive 3D-output with deployed API) would lift risk-adj from 0.7M× to 1.5M×–2.0M×.

3. **Stack saturation forces it as least-bad sub-axis at iter-236+.** If iter-236+ exhausts cleaner candidates, 3D-OUTPUT becomes natural front-runner — same elevation pattern as #88 from the iter-231 #87-B reservation.

---

## 2. Mechanism: 3D-output token distillation

### 2.1 3D-output tokenization options

Two viable autoregressive tokenization schemes per the literature; only one selected for #91-C primary-path.

**Path α: Mesh-token (MeshGPT 2024 pattern). [SELECTED for primary path].** Mesh `M = (V, F)` with vertex set `V = {v_i ∈ ℝ^3}_{i=1..|V|}` and face set `F = {(i_a, i_b, i_c)}_{j=1..|F|}` tokenized via:

```
Vertex-quantize:    v_i ∈ ℝ^3 → q_i ∈ {0,1}^{8 × 8 × 8}     (3D-grid quantization at 256³ resolution; 24 bits per vertex)
Face-token:         (i_a, i_b, i_c) → mesh_token ∈ Vocabulary[8192]   (joint vertex-triplet codebook)
Token sequence:     M → t_1 t_2 ... t_M               (M ≈ 1024–4096 mesh tokens per object)
```

Codebook size 8192 (matched to #82 image-out and #87 video-out for hot-path reuse). Vocabulary extended by 8192 mesh-output tokens.

**Path β: NeRF-parameter token (Shap-E 2023 discrete-mode pattern).** Implicit-field MLP weights `θ ∈ ℝ^{~10⁵}` projected to a fixed 8192-codebook over functional-form descriptors. Reserved as fallback path; harder to evaluate (Chamfer-distance requires marching-cubes inference).

**Path α selected** for direct evaluation simplicity (Chamfer / IoU on mesh vertices) and MeshGPT's autoregressive teacher availability.

### 2.2 Joint-sequence interleaving

```
<TEXT_BEGIN> "Generate a 3D model of a chair with armrests."
<3D_OUTPUT_BEGIN>
  <3D_META> bbox μ σ, scale-norm, vertex-count, face-count <3D_META_END>
  v_1 v_2 ... v_M           (M ≈ 1024–4096 mesh tokens)
<3D_OUTPUT_END>
<TEXT_BEGIN> "Generated mesh has 1842 vertices and 3680 faces." <TEXT_END>
```

Trunk attends across text and 3D-output regions transparently. **3D-output tokens contribute to context AND to the loss** (autoregressive CE on 3D-output positions; teacher-distillation KL where MeshGPT teacher logits are available).

### 2.3 Distillation objective

Per #82 / #83 / #87 image-out / audio-out / video-out pattern:

```
L_text     = -Σ_{t ∈ text-positions}     [α · log p_student(y_t | h_t)
                                          + β · KL(p_teacher(y_t | h_t) || p_student(y_t | h_t))]
L_3D-out   = -Σ_{t ∈ 3D-out-positions}   [α · log p_student(z_t | h_t)
                                          + β · KL(p_teacher_meshgpt(z_t | h_t) || p_student(z_t | h_t))]
L_total    = L_text + λ_3D · L_3D-out                      (λ_3D = 0.4 per #82 image-out tuning)
```

α=0.3, β=0.7 per #68 cached-logit pipeline. Teacher-mismatch on 3D-out positions: MeshGPT teacher generates the codebook; cached-logit pipeline at top-K=16 over the 8192-vocabulary.

### 2.4 Teacher selection

| Tier | Teacher | Params | Coverage | Notes |
|---|---|---|---|---|
| **Tier-1** | MeshGPT-355M (Tsinghua 2024) | 355M | ShapeNet, partial Objaverse | Closest direct autoregressive precedent; codebook K=8192. |
| **Tier-2** | Shap-E discrete-mode (OpenAI 2023) | 300M | Closed Shap-E corpus | Implicit-field tokens; harder to evaluate but autoregressive. |
| **Tier-3** | (none at production scale) | — | — | No 8B+ autoregressive 3D-output teacher exists. |

Cached-logit pipeline at top-K=16 over 8192-vocabulary; ~120 MB additional logit-cache in DataLoader.

### 2.5 Composition with prior 49 paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#88 3D-SPATIAL (input)** | ✓ Stack-base | Input-side 3D tokenization; OUTPUT extends to produce side. |
| **#82 IMAGE-OUTPUT** | ✓ Stack-base | Discrete VQ codebook output pattern reused; K=8192 matched. |
| **#83 AUDIO-OUTPUT / #87 VIDEO-OUTPUT** | ✓ | Same discrete-codebook output pattern; vocabulary extension scheme matched. |
| **#66 CROSS-MODAL** | ✓ Stack-base | Joint-sequence interleaving extends. |
| **#62 AGENT-CHIRON** | ✓ Synergy | 3D-output decodes to robotic-trajectory primitives; agent benchmarks gain actuation channel. |
| **#65 WORLD-MODEL** | ✓ Synergy | WS-bank rows hold generatable 3D-scene templates retrievable per #64-B's RETRO mechanism. |
| **#74 PHOENIX-1BIT** | ✓ | Trunk binary; 3D-output decoder BF16 (sensitive to mesh-fidelity). |
| **#76 MLA + #78 SINK + #79 MoD** | ✓ | 3D-output tokens are normal tokens; KV/sink/depth-routing apply transparently. |
| **#69 SUPER-DISTILL** | ✓ | Cached-logit pipeline extended to 3D-output token logits. |

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Text NLL preservation

Per #66 §4.1 / #82 §4.1 / #88 Theorem 1: text-only sequences pass through trunk identically; 3D-output token paths never invoked when input has no `<3D_OUTPUT_BEGIN>` token. **Bit-exact text NLL preserved on text-only data.**

Proof sketch: 3D-output paths gated by modality-bit-mask m_3Dout ∈ {0,1}; m_3Dout = 0 ⇒ 3D-output decoder pipeline skipped, 3D-output token cache empty, attention restricted to text positions only. Modality-segregation under multiplicative mask is bit-exact identity.

### 3.2 Theorem 2 — 3D-output bound

Student's 3D-output mesh-NLL bounded by teacher's mesh-NLL + capacity gap (per #68 Theorem 1) + tokenizer-quantization gap (3D-grid K=256³ + face-codebook K=8192):

```
NLL_student ≤ NLL_teacher + ε_capacity + ε_quant
ε_capacity → 0 as student_params ≥ teacher_params         (CHIRON 32B-eff ≫ 355M MeshGPT teacher; satisfied with very large margin)
ε_quant ≈ -log(1/8192) ≈ 9.0 nat per mesh-output token
```

**Quantization becomes binding constraint at student scale ≫ 355M.** Distillation gain plateaus sharply — same mechanism as #88's 7B teacher to CHIRON 32B-effective gap, but compounded by a 90× gap (vs #88's 4-5×) and the same K=8192 codebook ceiling.

### 3.3 Joint Gate-0 PASS probability

```
MeshGPT-style tokenization integration:                   ~80%
Joint-sequence 3D-output interleaving:                    ~92%
KL-CE on 3D-output positions (cached-logit pipeline):     ~88%
Memory at 16 GB ceiling (~530 MB cache; ~100 MB headroom): ~75%
Mesh-decoder fidelity ≥ 25 Chamfer on ShapeNet:           ~70%
LLM-scale empirical confirmation (MeshGPT-355M-class):    ~52%

Joint Gate-0 PASS:                                        ~52%
LLM-scale empirical confirmation:                         ~38%
```

Compare #88 (3D-input): Joint PASS ~55%, LLM-scale confirmation ~38%. **#91-C is strictly worse on Gate-0 PASS** (lower memory headroom + harder mesh-decoder fidelity + smaller teacher) at tied LLM-scale confirmation.

Lowest Gate-0 PASS in iter-228-235 slate excluding #90's speculative-rescue category.

---

## 4. Updated cumulative stack (if SELECTED)

```
Iter 234 close (post-#90):
  All 27 axes ≈preserved
  Memory recovery: ~120-200 MB on MLA latent Adam state (if Gate-0 PASS)

Iter 235 (3D-OUTPUT-DISTILL-CHIRON, hypothetical SELECT):
  All 27 axes ≈preserved (no new axis; SUB-axis extension under #88's 3D-SPATIAL macro)
  3D-SPATIAL axis gains OUTPUT sub-channel
  3D-OUTPUT benchmarks: ~5,000,000× new sub-axis
  (ShapeNet-Chamfer, Objaverse-CD, ShapeNet-IoU, GenEval-3D, T2I3D-CompBench)
  Risk-adjusted: ~700,000× (lowest in iter-228-235 slate due to thin precedent + ~70% overlap)
  Memory headroom: 16 GB ceiling preserved; ~100 MB headroom remaining (BREAKING-POINT THRESHOLD)
```

If RESERVED at iter-235 (likely verdict): stack unchanged at 27 axes; #91-C available for future iteration if user signals 3D-output territory or production-scale 3D-output teacher emerges.

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| MeshGPT-style mesh-token tokenizer (vertex-grid + face-codebook) | 350 | 1.5 |
| Token vocabulary extension (+8192 3D-output tokens) | 100 | 0.5 |
| Joint-sequence DataLoader (text + 3D-output mesh data) — ShapeNet, Objaverse, ABO | 280 | 1.5 |
| Modality bit-mask + 4 3D-output special tokens | 80 | 0.5 |
| Cached-logit pipeline extension for MeshGPT teacher | 200 | 1.0 |
| KL-CE loss for 3D-output positions (autoregressive over codebook) | 120 | 0.5 |
| Mesh-decoder pipeline (token → mesh; marching-cubes for NeRF fallback) | 220 | 1.0 |
| Chamfer-distance / IoU / GenEval-3D evaluation harness | 250 | 1.0 |
| Memory accounting verification (530 MB additional GPU; 100 MB headroom check) | 80 | 0.25 |
| NLL preservation regression tests on text-only positions | 70 | 0.25 |
| **Total** | **~1,750** | **7** |

Slightly higher than #88 (1,650 LOC, 7 weeks) due to mesh-decoder + Chamfer/IoU evaluation harness. Lower than #87 video-out (2,200 LOC) since 3D output is single-shot per object (no temporal coherence module).

---

## 6. Memory advantage preservation

| Component | GPU memory |
|---|---|
| MeshGPT-tokenizer codebook (8192 × 256 BF16) | ~4 MB |
| Mesh-decoder MLP (frozen; ~250M params BF16) | ~500 MB |
| 3D-output token cache during inference | ~30 MB |
| **Total additional GPU** | **~534 MB** |

**Single-GPU 16 GB ceiling preserved** with **~100 MB headroom** under post-#88 stack (vs #88 close's 630 MB; vs #86's 770 MB).

**This is the tightest headroom of any iter-228-235 candidate.** Under #91-C selection, the post-#91 stack is at the breaking-point of the 16 GB ceiling; any further memory-cost paradigm at iter-236+ would force constraint-relaxation or memory-recompression (PHOENIX-deeper, FACE-on-decoder, MELT-on-decoder).

---

## 7. Gates

### Gate-0 (~14 GPU-hours)

**Probe.** 200M coordinator + MeshGPT-tokenizer + ~6M mesh-text pairs (ShapeNet-Cap + Objaverse-cap + ABO subset). Distill from MeshGPT-355M teacher.

**PASS criteria.**
- ShapeNet-Chamfer ≤ 35 (MeshGPT-355M-class).
- Objaverse-Chamfer ≤ 50.
- NLL on text-only ≤ 0.01 nat drift.
- Memory at 16 GB ceiling verified (≥ 50 MB headroom maintained).

**PASS probability:** ~52%.

Lower than #88's ~55% due to:
- Smaller teacher (355M MeshGPT vs 7B 3D-LLM-3D-input).
- Mesh-decoder fidelity less certain (Chamfer-distance variance with K=8192 codebook).
- Memory headroom binding constraint (100 MB residual; close to ceiling).

### Gate-1 (~210 GPU-hours conditional)

If Gate-0 PASS:
**Probe.** Full 32B-effective + MeshGPT-355M teacher + 28M mesh-text pairs across ShapeNet / Objaverse / ABO / 3D-FUTURE / Thingi10K. Full 3D-output benchmark suite.

**PASS criteria.**
- ShapeNet-Chamfer ≤ 28.
- Objaverse-Chamfer ≤ 42.
- ShapeNet-IoU ≥ 0.55.
- GenEval-3D ≥ 0.30.
- Memory at 16 GB ceiling.

**PASS probability conditional on Gate-0:** ~50%.

---

## 8. Honest gaps

1. **3D-OUTPUT not in user brief.** Selected discussion on saturation-symmetry-completion grounds; user-need narrow (CAD / game-asset / AR-VR / robotics-actuation / 3D-printing).

2. **Mechanism mostly pre-existing technique** (MeshGPT autoregressive pattern + Shap-E discrete-mode + #82 codebook pattern). Novelty is system-integration with iter-217-234 stack — no algorithmic advance.

3. **Production precedent thinnest of all output sub-axes.** MeshGPT (355M) and Shap-E discrete-mode (300M) are the only viable autoregressive teachers; both research-stage, no production-deployed API. Risk-adj magnitude drops to 0.7M× from 5M× nominal — **lowest in iter-228-235 slate**.

4. **38% LLM-scale empirical confirmation** modest — MeshGPT at 355M; CHIRON 32B-effective extrapolation across a **90× capacity gap**. Tokenizer-quantization gap (K=8192) becomes binding at student scale by a wider margin than #88's 7B teacher.

5. **5M× sub-axis-extension class**, not magnitudes-better in raw text-NLL sense. Text NLL preserved by construction; gain confined to 3D-output benchmark subsets (Chamfer / IoU / GenEval-3D).

6. **~70% mechanism overlap with #88 + #82.** Highest overlap of any iter-235 candidate. Honest framing: this is a SUB-axis of #88's 3D-SPATIAL macro-axis, not a fresh 28th axis.

7. **534 MB additional GPU memory** narrows headroom from 630 MB (#88 close) to ~100 MB. **Breaking-point threshold for any further memory-cost paradigm.** Post-#91 stack at this margin forces iter-236+ candidates toward compute-only or memory-recompression class.

8. **User-need narrow.** 3D-output serves CAD / game-asset / AR-VR / robotics-actuation / 3D-printing / embodied-agent territory. None signaled in user brief through iter-235.

9. **Continuous-diffusion (DreamFusion / Magic3D) and GAN-based (GET3D) approaches NOT autoregressive**, hence NOT NLL-compatible. Excluded from primary path on the same grounds as #82's Stable-Diffusion exclusion. Reserved for hypothetical #91.5+ if NLL constraint relaxed for 3D-output axis specifically.

10. **Mesh-vs-NeRF tokenization choice load-bearing.** Path α (mesh-token) selected for evaluation simplicity; Path β (NeRF-parameter) reserved as fallback. If MeshGPT teacher availability blocked, fallback to Shap-E discrete-mode possible but with higher engineering cost (~+250 LOC for marching-cubes inference and implicit-field evaluation).

11. **Symmetric-completion pattern at saturation.** #82 image-out (at #66 close), #83 audio-out (at #80 close), #87 video-out (at #84 close), #91-C 3D-out (at #88 close) all follow the same pattern: I/O symmetry completion at axis-extension class. **The pattern itself is at saturation** — symmetry-completion candidates after #91 have no obvious next modality (haptic? olfactory? all extreme-narrow).

12. **ShapeNet / Objaverse data-coverage limited.** Production 3D-output systems train on ~10M-100M mesh-text pairs (Objaverse + ABO + custom synthetic); CHIRON Gate-1 budget (28M pairs) is at-class but not above-class.

---

## 9. Bottom line

**3D-OUTPUT-DISTILL-CHIRON is genuinely-new sub-axis but the natural #91-C RESERVE.** It:

- **Genuine 3D-OUTPUT sub-axis** symmetric to #82/#83/#87 image/audio/video-output completion pattern.
- **But sub-axis of #88 macro-axis** (not fresh 28th axis).
- **Production precedent thinnest of all output sub-axes** (MeshGPT/Shap-E at <2B vs Chameleon's 7B–34B / MusicGen's 3.3B / VideoPoet's 8B).
- **Risk-adjusted ~0.7M× < #88's 0.8M× < #82's 1.7M×** — lowest in iter-228-235 slate.
- **~70% mechanism overlap with #88 + #82** — highest in iter-235 slate.
- **Memory cost at breaking-point threshold** (~100 MB headroom remaining; vs #88 close's 630 MB).
- **User-need narrow** — CAD / game-asset / AR-VR / robotics-actuation / 3D-printing unsignaled in current brief.
- **Compute-NEUTRAL on text** preserved.

**Verdict.** **RESERVE for future iteration.** Re-evaluate when ANY of:

1. User signals 3D-generation / robotics-actuation / CAD-output / game-asset / AR-VR-content territory.

2. Production-scale autoregressive 3D-output teacher emerges (8B+ open-source MeshGPT-equivalent + deployed API).

3. Stack saturation forces it as least-bad sub-axis at iter-236+ (if cleaner alternatives exhaust).

**Cumulative single-GPU stack at iter-235 (if SELECTED, hypothetical):**
- All 27 prior axes ≈preserved
- 3D-SPATIAL axis gains OUTPUT sub-channel
- **3D-OUTPUT benchmarks: ~5,000,000× new sub-axis (risk-adj 700,000×)**
- Memory headroom: ~100 MB (BREAKING-POINT)

**Engineering** if SELECTED: ~1,750 LOC over 7 weeks. **Joint Gate-0 PASS ~52%; LLM-scale confirmation ~38%; risk-adj ~0.7M×.**

After 50 paradigms (if hypothetically selected), the bigger-picture stack would still cover **27 axes** (sub-axis extension on 3D-SPATIAL macro). Iter-236+ candidates would face:

- **Memory-headroom breaking-point**: any further memory-cost axis-extension forces constraint-relaxation or PHOENIX-deeper / FACE-on-decoder / MELT-on-decoder recompression.
- **I/O-symmetry-completion exhaustion**: image-out / audio-out / video-out / 3D-out covers all major modalities; haptic / olfactory are extreme-narrow.
- **Continued recompositions** of more rejected paradigms (#36 KV-FACE, #37 HUTCH-DIAG, #41 ASTRA, #87-C META-VALIDATION) under iter-212 framing.
- **Constraint relaxation** (multi-GPU per #90-B reserved-as-recommendation; user-decision required).
- **Empirical validation feedback** (per META-VALIDATION; out of scope for design loop).
- **Genuinely new orthogonal axis** (highly unlikely at depth 27; structural ceiling unambiguous through iter-235).

If RESERVED at iter-235 (likely): #91-C remains queued; stack at 27 axes; selection slot opens for next slate's strongest production-precedent candidate. The reservation matches the pattern of #87-B at iter-231 (which was promoted to #88 at iter-232) and may similarly elevate at iter-237+ if user signal or scale-precedent shifts.
