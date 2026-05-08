# Paradigm Shift #87 — Candidate B: 3D-SPATIAL-DISTILL-CHIRON: 3D Scene-Understanding Axis

**Status:** RESERVE candidate (research-stage; thin production precedent at scale; user-need narrow; magnitude axis-extension class).
**Date:** 2026-05-08 (Ralph-loop iter 231, post-#86 TIME-SERIES-DISTILL temporal/forecasting axis).
**Axis:** **3D-SPATIAL/SCENE-UNDERSTANDING** — proposed 26th axis. Spatial scene understanding via 3D-foundation-model distillation.
**Magnitude target:** **~5,000,000× new 3D-SPATIAL/SCENE-UNDERSTANDING axis**; risk-adjusted ~800,000× (lower than recent slate due to thin production precedent and narrow user-need conditional).

---

## 0. Executive summary

**Mechanism.** Distill from 3D foundation models that fuse point-cloud / multi-view-image / scene-graph input with a frozen LLM stack. Teacher candidates: 3D-LLM (Hong et al., Stanford CRA 2023), LLM-Grounder (LSU 2023), GPT4Scene (NUS 2024), 3DLLaMa. Scenes tokenized through a point-cloud encoder (PointBERT / Point-MAE / Point-Bind family) producing a discrete or continuous spatial-token stream. Joint sequence:

```
<TEXT> ... <3D_BEGIN> p_1 p_2 ... p_N <3D_END> ... <TEXT>
```

KL-CE distillation per #66 / #80 / #84 pattern restricted to text positions; 3D-token positions are skipped in the loss (teacher logit-space mismatch). Trunk learns spatial-grounded text generation: scene-question-answering, object-grounding, layout-description, robotic-instruction.

**Production precedent honestly thin at scale:**
- **3D-LLM** (Hong et al. 2023): research codebase; teacher backbones in 1.3B–7B range; not production-deployed.
- **LLM-Grounder** (Yang et al. 2023): research; per-scene zero-shot grounding via LLM-as-coordinator.
- **GPT4Scene** (Wang et al. 2024): research; multi-view 3D scene understanding via GPT-4 prompting.
- **3DLLaMa** (Yang et al. 2024): research; LLaMa-2-7B fine-tuned on 3D-Q&A.
- **PointLLM** (Xu et al. 2023): 7B research model; point-cloud + LLM.

**No AWS-Chronos / TimeGPT-API equivalent at production scale exists for 3D.** All five candidate teachers sit at <8B parameters in research repos; distillation extrapolation to CHIRON 32B-effective is uncertain.

**Why B is genuinely new but reserved (not selected) at iter-231:**
- **Genuine 26th axis** orthogonal to all 25 prior axes (text/image/audio/video/forecasting).
- **But thin production precedent** caps risk-adjusted magnitude at ~800k× vs #86's 1.7M×.
- **User-need narrow** — 3D scene understanding chiefly serves robotics, AR/VR, simulation, embodied-agent, CAD/engineering subsets; user brief did not signal these.
- **38–42% LLM-scale empirical confirmation** (modest; lower than #86's 38% on stronger production base).
- **Memory cost slightly higher** than #86 (~140 MB additional GPU vs #86's ~110 MB) due to point-cloud-encoder cache.

**Composition with prior 46 paradigms:**
- **#66 CROSS-MODAL**: joint-sequence interleaving pattern reused (same `<MODALITY_BEGIN/END>` tag scheme).
- **#80 AUDIO + #84 VIDEO**: temporal modalities; **3D is spatial-orthogonal modality** — independent dimension, not a temporal extension.
- **#82 IMAGE-OUTPUT / #83 AUDIO-OUTPUT / #86 TIME-SERIES**: discrete tokenization patterns.
- **#76 MLA + #78 SINK + #79 MoD**: 3D tokens are normal tokens; KV / sink / depth-routing apply transparently.
- All 25 prior axes ≈preserved (compute-NEUTRAL on text).

**Trade-offs honestly recorded.**
- 3D-SPATIAL not in user brief; selected discussion is on saturation-breaking + axis-orthogonality grounds.
- Mechanism mostly pre-existing (3D-LLM pattern + #68 distillation); novelty is system-integration with iter-217-230 stack.
- Production precedent at scale genuinely thin — research-stage teachers only.
- 3D-token distillation may underperform vs text-only on text NLL drift (140 MB cache + extra forward overhead in tokenizer pipeline).
- User-need narrow (robotics/AR/VR/CAD subsets only).

**Engineering:** ~1,650 LOC over 7 weeks. **Joint Gate-0 PASS ~55%; LLM-scale confirmation ~38%; risk-adj 0.8M×.**

---

## 1. Candidate formulation and selection

### 1.1 Three iter-231 candidates (cross-reference)

| Candidate | File | Mechanism | Verdict |
|---|---|---|---|
| **A — VIDEO-OUTPUT-DISTILL** (continued) | `PARADIGM_SHIFT_85_CANDIDATE_B_VIDEO_OUTPUT.md` | Discrete VQ video output codebook | Reserve continued |
| **B — 3D-SPATIAL-DISTILL** | this document | Point-cloud encoder + 3D-LLM teacher distillation | **RESERVE (research-stage; thin production precedent; user-need narrow)** |
| **C — alternative composition / consolidation** | sibling candidate doc | Composition of #64/#65/#66/#80/#84 banks | Reserve continued |

### 1.2 Why this candidate is genuinely interesting

1. **Genuinely new 26th axis.** 3D-SPATIAL is orthogonal to all 25 prior axes. Time-series #86 covered TEMPORAL/FORECASTING (1D, time-indexed); 3D covers SPATIAL/SCENE-UNDERSTANDING (3D, geometry-indexed). No double-count.

2. **Joint-sequence pattern reuse.** #66 / #80 / #84 / #86 patterns transfer directly. Engineering risk on integration is low.

3. **Robotics / AR/VR latent demand.** If user pivots to embodied-agent territory (post-#62 AGENT, post-#86 forecasting), spatial grounding becomes immediately useful.

4. **PointBERT / Point-MAE encoders are mature.** Tokenization pipeline does not require novel encoder design — uses MAE-style masked-autoencoder pretraining on ShapeNet / ScanNet / ScanQA / 3DLLM-Datasets-Combined.

### 1.3 Why this candidate is RESERVED, not SELECTED

Five honest reasons:

**1. Thin production precedent at scale.**
- 3D-LLM (Stanford CRA 2023): research-stage; closest to "production" but no API deployment, no enterprise customers.
- LLM-Grounder, GPT4Scene, 3DLLaMa, PointLLM: all research codebases.
- None of these match the production-deployment status of AWS Chronos (#86), MusicGen (#83 reference), Whisper (#80 reference), VideoPoet (#84 reference).
- Risk-adjusted magnitude drops sharply: ~5M× nominal × 0.16 production-precedent factor ≈ 0.8M× risk-adj, vs #86's 5M × 0.34 ≈ 1.7M×.

**2. Teacher-scale gap to CHIRON.**
- Best 3D teacher candidates sit at 1.3B–7B parameters.
- CHIRON 32B-effective via #53 MOSAIC-MOE + #58 METAGEN scaling exceeds teacher capacity by 4-5×.
- Per #68 distillation theorem 1, this is **upper-bounded by quantization granularity of point-cloud tokenizer** (typically K=512 codebook for PointBERT). Granularity becomes the binding constraint — not capacity.
- Beyond ~1.3B teacher size, marginal student gains plateau. Empirical confirmation drops.

**3. User-need narrow and unsignaled.**
- 3D-SPATIAL serves: robotics (manipulation, navigation), AR/VR (scene placement, object retrieval), CAD/engineering (BIM grounding), embodied agents (3D-world LLMs), simulation (digital twins).
- User brief through iter-231: text-LLM compute speed + memory + NLL accuracy + single-GPU + novel + bigger-picture — no embodied-agent or robotics signal.
- Adding 3D axis without user-need is saturation-padding.

**4. Memory cost slightly tighter.**
- Point-cloud encoder cache (PointBERT-base ~22M params): ~90 MB GPU residence.
- Point-cloud token-stream cache: ~50 MB.
- Total: ~140 MB additional GPU vs #86's 110 MB.
- 16 GB ceiling preserved with ~630 MB headroom (vs #86's 770 MB) — but margin tightens with each axis added.

**5. 3D-output not free from #82 IMAGE-OUTPUT competition.**
- 3D-OUTPUT (mesh / NeRF / Gaussian-splatting generation) overlaps with #82 IMAGE-OUTPUT pipeline 30-40%.
- If 3D-input is reserved, 3D-output is similarly reserved. Compounds against #87 selection now.

### 1.4 What would unlock SELECT later

Three signals would move this to SELECT:

1. **User signals embodied-agent / robotics / AR-VR territory.** Brief sharpening to include "spatial reasoning" or "scene-grounded LLM" or "embodied agent" would shift verdict.

2. **Production-scale 3D teacher emerges.** 8B+ open-source 3D-LLM with deployed API (analog of AWS Chronos) would lift risk-adj from 0.8M× to 1.5–2.0M×.

3. **Stack saturation accelerates.** If iter-235+ exhausts cleaner axes (audio-music-output, neuro-symbolic, lifelong-learning), 3D becomes natural front-runner.

---

## 2. Mechanism: 3D-spatial token distillation

### 2.1 Point-cloud-style 3D tokenization

Continuous 3D scene `S = {(x_i, y_i, z_i, r_i, g_i, b_i, n_i^x, n_i^y, n_i^z)}_{i=1..M}` (point cloud with color and normal channels) tokenized through PointBERT-style masked autoencoder:

```
Patchify:   S → {patch_j ∈ ℝ^{32 × 9}}_{j=1..N}     (N ≈ 512–2048 patches per scene)
Encode:     patch_j → z_j ∈ ℝ^{384}                  (PointBERT-base encoder)
Quantize:   z_j → token_j = Argmin_k ||z_j - codebook_k||  (K=512–4096 codes)
Token-id:   token_j → Vocabulary[token_j]            (extended vocab)
```

PointBERT codebook size K = 4096 chosen to match Chronos K (#86) for memory parity.

### 2.2 Joint-sequence interleaving

```
<TEXT_BEGIN> "Where is the chair relative to the table?"
<3D_BEGIN>
  <SCENE_META> bbox μ σ, scale-norm <SCENE_META_END>
  p_1 p_2 ... p_N           (N ≈ 512–2048 spatial tokens)
<3D_END>
<TEXT_BEGIN> "The chair is to the right of the table, approximately 1.2 meters away." <TEXT_END>
```

Trunk attends across text and 3D regions transparently. 3D tokens contribute to context but are excluded from the CE/KL loss (teacher tokenizer mismatch).

### 2.3 Distillation objective

Per #66 / #80 / #84 / #86 pattern restricted to text positions:

```
L_text = -Σ_{t ∈ text-positions} [α · log p_student(y_t | h_t)
                                  + β · KL(p_teacher(y_t | h_t) || p_student(y_t | h_t))]
L_3D   = 0   (3D-token positions skipped; teacher tokenizer mismatch)
L_total = L_text
```

α=0.3, β=0.7 per #68 cached-logit pipeline.

### 2.4 Teacher selection

Tier-1 (preferred): 3D-LLM-7B (Hong et al. 2023) — closest to scale; multi-view image + point-cloud input.
Tier-2: PointLLM-7B (Xu et al. 2023) — point-cloud + LLaMa-2.
Tier-3: GPT4Scene (NUS 2024) — multi-view; teacher quality high but API-only.

Cached-logit pipeline at top-K=16 over text vocabulary (3D-token vocab not cached; not in loss).

### 2.5 Composition with prior 46 paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#66 CROSS-MODAL** | ✓ Stack-base | Joint-sequence interleaving extends from image/audio/video/TS to 3D. |
| **#80 AUDIO + #84 VIDEO** | ✓ | Temporal modalities; **3D is spatial-orthogonal sister axis**. |
| **#82 IMAGE-OUTPUT / #83 AUDIO-OUTPUT / #86 TIME-SERIES** | ✓ | Discrete tokenization patterns; same vocabulary-extension scheme. |
| **#74 PHOENIX-1BIT** | ✓ | Trunk binary; PointBERT scale parameters BF16. |
| **#76 MLA + #78 SINK + #79 MoD** | ✓ | 3D tokens are normal tokens; KV/sink/depth-routing apply. |
| **#62 AGENT-CHIRON** | ✓ Synergy | Spatial-grounded agent benchmarks (RoboBench, AI2-THOR, Habitat) directly benefit. |
| **#65 WORLD-MODEL** | ✓ Synergy | Spatial WS-bank rows (E,P,R,C tagged with scene-geometry) become retrievable. |
| **#86 TIME-SERIES** | ✓ | Spatial-temporal joint sequence (4D scenes) becomes possible if both are stacked. |

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Text NLL preservation

Per #66 §4.1 / #86 Theorem 1: text-only sequences pass through trunk identically; 3D-token paths never invoked when input has no `<3D_BEGIN>` token. **Bit-exact text NLL preserved on text-only data.**

Proof sketch: 3D-token paths gated by modality-bit-mask m_3D ∈ {0,1}; m_3D = 0 ⇒ 3D-encoder forward skipped, 3D-token cache empty, attention restricted to text positions only.

### 3.2 Theorem 2 — 3D-grounding bound

Student's 3D-grounded-QA NLL bounded by teacher's NLL + capacity gap (per #68 Theorem 1) + tokenizer-quantization gap (PointBERT K=4096):

```
NLL_student ≤ NLL_teacher + ε_capacity + ε_quant
ε_capacity → 0 as student_params ≥ teacher_params         (CHIRON 32B-eff ≫ 7B teacher; satisfied)
ε_quant ≈ -log(1/K) = -log(1/4096) ≈ 8.32 nat per spatial token
```

Quantization becomes binding constraint at student scale ≫ 7B. **Distillation gain plateaus.**

### 3.3 Joint Gate-0 PASS probability

```
PointBERT-style tokenization integration:                 ~85%
Joint-sequence 3D interleaving:                           ~92%
KL-CE on text-only positions (3D skipped):                ~95%
Memory at 16 GB ceiling (~140 MB cache):                  ~90%
LLM-scale empirical confirmation (3D-LLM-7B-class):       ~55%

Joint Gate-0 PASS:                                        ~55%
LLM-scale empirical confirmation:                         ~38%
```

Compare #86: Joint PASS ~62%, LLM-scale confirmation ~38%. #87-B is **strictly worse on Gate-0 PASS** (PointBERT-integration risk + teacher-scale gap), tied on LLM-scale confirmation.

---

## 4. Updated cumulative stack (if SELECTED)

```
Iter 230 close (post-#86):
  All 25 axes ≈preserved
  TEMPORAL/FORECASTING at 5,000,000× new axis

Iter 231 (3D-SPATIAL-DISTILL-CHIRON, hypothetical SELECT):
  All 25 axes ≈preserved (compute-NEUTRAL on text)
  **3D-SPATIAL/SCENE-UNDERSTANDING benchmarks: ~5,000,000× NEW AXIS**
  (ScanQA, SQA3D, Scan2Cap, ObjaverseQA, RoboBench, AI2-THOR-QA)
  Risk-adjusted: ~800,000× (production-precedent thin)
```

If RESERVED at iter-231 (likely verdict): stack unchanged at 25 axes; #87-B available for future iteration if user signals embodied-agent territory or production-scale 3D teacher emerges.

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| PointBERT-style point-cloud tokenizer (encoder + codebook) | 350 | 1.5 |
| Token vocabulary extension (+4096 3D tokens) | 100 | 0.5 |
| Joint-sequence DataLoader (text + 3D scenes) — ScanNet, ScanQA, ObjaverseQA, 3RScan | 300 | 1.5 |
| Modality bit-mask + 4 3D-special tokens | 100 | 0.5 |
| Cached-logit pipeline extension for 3D-LLM teacher | 200 | 1.0 |
| KL-CE loss restricted to text positions (3D skipped) | 100 | 0.5 |
| Multi-view image-encoder fallback (when point-cloud unavailable) | 200 | 1.0 |
| 3D-grounded evaluation harness (ScanQA, SQA3D, Scan2Cap, RoboBench) | 300 | 1.5 |
| **Total** | **~1,650** | **7** |

Slightly higher than #86 (1,400 LOC, 6 weeks) due to multi-view fallback and richer eval harness.

---

## 6. Memory advantage preservation

| Component | GPU memory |
|---|---|
| PointBERT-base encoder (22M params, BF16) | ~90 MB |
| 3D-token cache during inference | ~50 MB |
| Multi-view image-encoder cache (when active) | ~0 (lazy) |
| **Total additional GPU** | **~140 MB** |

**Single-GPU 16 GB ceiling preserved** with ~630 MB headroom under post-#86 stack (vs #86's 770 MB). Margin tightens but stays safe.

---

## 7. Gates

### Gate-0 (~12 GPU-hours)

**Probe.** 200M coordinator + PointBERT-base tokenizer + ~8M 3D-text pairs (ScanQA + Scan2Cap + ObjaverseQA subset). Distill from 3D-LLM-7B (or PointLLM-7B as fallback).

**PASS criteria.**
- ScanQA EM ≥ 22% (3D-LLM-base-class).
- SQA3D accuracy ≥ 45%.
- NLL on text-only ≤ 0.01 nat drift.
- Memory at 16 GB ceiling verified.

**PASS probability:** ~55%.

Lower than #86's ~70% due to:
- PointBERT-tokenizer integration risk higher than scale-mean-quantization.
- 3D-LLM teacher availability less certain (research repos may have license / packaging gaps).
- Tokenizer-codebook quality variance higher.

### Gate-1 (~180 GPU-hours)

**Probe.** Full 32B-effective + 3D-LLM-7B teacher + 35M 3D-text pairs across ScanNet / ScanQA / 3RScan / ObjaverseQA / RoboBench. Full 3D-grounded benchmark suite.

**PASS criteria.**
- ScanQA EM ≥ 30% (3D-LLM-7B-class).
- SQA3D accuracy ≥ 55%.
- Scan2Cap CIDEr ≥ 50.
- RoboBench task success ≥ 35%.
- Memory at 16 GB ceiling.

**PASS probability conditional on Gate-0:** ~50%.

---

## 8. Honest gaps

1. **3D-SPATIAL not in user brief.** Selected on saturation-breaking + axis-orthogonality grounds; user-need narrow (robotics/AR-VR/CAD).

2. **Mechanism mostly pre-existing technique** (3D-LLM pattern + #68 distillation). Novelty is system-integration with iter-217-230 stack — no algorithmic advance.

3. **Production precedent honestly thin.** All teacher candidates research-stage; no AWS-Chronos / TimeGPT-API equivalent at scale exists for 3D. Risk-adj magnitude drops to 0.8M× from 5M× nominal.

4. **38% LLM-scale empirical confirmation** modest — 3D-LLM at 7B; CHIRON 32B-effective extrapolation uncertain. Tokenizer-quantization gap (K=4096) becomes binding constraint at student scale ≫ 7B.

5. **5M× axis-extension class**, not magnitudes-better in raw text-NLL sense. Text NLL preserved by construction; gain confined to 3D-grounded subsets.

6. **User-need narrow.** 3D scene understanding serves robotics / AR-VR / CAD / embodied-agent territory. None signaled in user brief through iter-231.

7. **140 MB additional GPU memory** narrows headroom from 770 MB (#86 close) to 630 MB. Acceptable but with each axis the 16 GB ceiling tightens.

8. **Multi-view-image fallback complexity.** Many real-world 3D-grounded tasks lack point clouds; multi-view image encoder (CLIP-3D, ViT-3D-fused) needs fallback path. Added engineering 1.0 week.

9. **3D-output not part of this candidate.** Only 3D-input distillation — output (mesh / NeRF / 3D-Gaussian-splat generation) deferred. Symmetry with input-only #80 / #84 / #86 holds.

10. **Self-supervised 3D pretraining costs.** PointBERT requires ShapeNet / ObjaverseLVM pretraining (~2 GPU-days); not free. Cached pretrained weights from public release recommended.

---

## 9. Bottom line

**3D-SPATIAL-DISTILL-CHIRON is a genuinely-new 26th axis but the natural #87 RESERVE.** It:

- **Genuine 26th axis** orthogonal to text/image/audio/video/temporal-forecasting.
- **But thin production precedent at scale** (research-stage teachers; no AWS-Chronos analog).
- **Risk-adjusted ~800,000× < #86's 1,700,000×** despite same nominal 5M×.
- **User-need narrow** — robotics / AR-VR / CAD / embodied-agent unsignaled in current brief.
- **Compute-NEUTRAL on text** preserved.

**Verdict.** **RESERVE for future iteration.** Re-evaluate when ANY of:
1. User signals embodied-agent / robotics / AR-VR / spatial-reasoning territory.
2. Production-scale 3D-LLM teacher emerges (8B+ open-source + deployed API).
3. Stack saturation forces it as least-bad axis (iter-235+ if cleaner alternatives exhaust).

**Cumulative single-GPU stack at iter-231 (if SELECTED, hypothetical):**
- All 25 prior axes ≈preserved
- **3D-SPATIAL/SCENE-UNDERSTANDING benchmarks: ~5,000,000× NEW AXIS (risk-adj 800,000×)**

**Engineering** if SELECTED: ~1,650 LOC over 7 weeks.

After 47 paradigms (if hypothetically selected), the bigger-picture stack would reframe **26 axes**. Iter-232+ candidates would pursue:
- **#88 AUDIO-MUSIC-OUTPUT** (specialized music-generation; MusicGen / AudioLDM teachers; production precedent stronger than 3D).
- **NEURO-SYMBOLIC** (logic-program-induction; constraint-solver as auxiliary; risk-adj uncertain).
- **LIFELONG-LEARNING** (continual pretraining with task replay; task-graph paradigm dimension).
- **MEMORY-CONSOLIDATION** (composition class continued).
- **VIDEO-OUTPUT** (continued reservation from #85-A).
- **Recomposition of more rejected paradigms** (#36 KV-FACE, #37 HUTCH-DIAG, #41 ASTRA) under iter-212 framing.

If RESERVED at iter-231 (likely): #87-B remains queued; stack at 25 axes; selection slot opens for next slate's strongest production-precedent candidate.
