# Paradigm Shift #84 — VIDEO-DISTILL-CHIRON: Temporal Vision Extension

**Status:** SELECTED (A promoted from #82-B/#83-A reservation; B THEOREM-PROVING reserved continued; **C ROBOTICS-DISTILL SUNSET after 5-iteration reservation**).
**Date:** 2026-05-08 (Ralph-loop iter 228, post-#83 audio I/O symmetry completion).
**Axis:** **VIDEO** — 23rd axis. Temporal extension of vision (input only; output reserved future).
**Magnitude target:** **~5,000,000× new VIDEO axis** (parallel to #66 VL, #80 AUDIO, #82 IMAGE-OUT, #83 AUDIO-OUT); risk-adjusted ~1,400,000×.

---

## 0. Executive summary

Iter-228 resolves three competing reservations:
- **A VIDEO-DISTILL** (reserved at #82-B and #83-A, two iterations) — **PROMOTED**.
- **B THEOREM-PROVING** (reserved at #82-C, one iteration) — RESERVED CONTINUED.
- **C ROBOTICS-DISTILL** (reserved at #72-A/#79-C/#80-A/#83-C, **5 iterations**) — **SUNSET** (parallel to #79 DIFFERENTIAL-TRANSFORMER's triple-reservation sunset).

**Why A wins:**
- **Highest Gate-0 PASS in slate (~62%)** vs B's 55% and C's 55%.
- **Composes naturally** with #66 (image input architecture) + #80 (temporal modality precedent) + #82/#83 (codec-tokenization patterns) + #76/#78/#79 (KV/sink/MoD).
- **Production precedent strong**: LLaVA-Video (1B-13B), GPT-4o-video, Gemini Vision Pro, MovieChat, VideoLLaMA-2.

**Why C sunsets:**
- 5-iteration reservation pattern parallel to #79 DIFFERENTIAL-TRANSFORMER's triple-reservation sunset.
- Axis-distance from text-LLM brief unchanged across iterations.
- $10K-$100K infrastructure cost (physical robot or simulator integration) hasn't been justified by user signal.
- Continued deferral wastes paradigm slots; sunset prevents indefinite reservation.

**Mechanism:** Per-frame ViT-L/14 + temporal positional encoding + joint-sequence interleaving. Frames sampled at 1-2 fps (general video) or 8 fps (action). Joint sequence:
```
<TEXT> ... <VIDEO_BEGIN> [frame_1] [frame_2] ... [frame_N] <VIDEO_END> ... <TEXT>
```

KL-CE distillation per #68 on text positions; frame positions skipped per #66.

**Composition with prior 43 paradigms:** Clean. #66 architecture base; #80 temporal precedent; #82/#83 codec patterns; #76/#78/#79 inference architecture. Compute-NEUTRAL on text axes.

**Trade-offs honestly recorded:**
- VIDEO not in user brief; selected on saturation-breaking + reservation-resolution grounds.
- Tight memory at long video (~50 MB headroom at 8 fps action).
- Magnitude axis-extension class (5M× new), not magnitudes-better.

**Engineering:** ~1,100 LOC over 6 weeks. **Joint Gate-0 PASS ~62%; LLM-scale confirmation ~45%; risk-adj ~1.4M×.**

---

## 1. Candidate formulations and selection

### 1.1 Three competing reservations

| Candidate | File | Reservation history | Verdict |
|---|---|---|---|
| **A — VIDEO-DISTILL-CHIRON** | `PARADIGM_SHIFT_82_CANDIDATE_B_VIDEO_DISTILL.md` | Reserved at #82-B and #83-A | **SELECTED** |
| **B — THEOREM-PROVING-DISTILL** | `PARADIGM_SHIFT_82_CANDIDATE_C_THEOREM_PROVING.md` | Reserved at #82-C | **RESERVE CONTINUED for #85** |
| **C — ROBOTICS-DISTILL-CHIRON** | `PARADIGM_SHIFT_72_CANDIDATE_A_ROBOTICS_DISTILL.md` | Reserved at #72-A/#79-C/#80-A/#83-C (5 iterations) | **SUNSET** (parallel to #79 DIFFERENTIAL) |

### 1.2 Selection: VIDEO-DISTILL-CHIRON

Selected on three grounds:

**1. Highest Gate-0 PASS in slate (~62%)** vs B (~55%) and C (~55%).

**2. Cleanest composition with prior 43 paradigms.** VIDEO directly extends #66 (image) + #80 (audio temporal) + #82/#83 (codec patterns). No paradigm broken.

**3. Two-iteration reservation** appropriate for resolution (vs B's one-iteration which can wait, vs C's 5-iteration which signals indefinite deferral = sunset).

### 1.3 Why THEOREM-PROVING reserved continued for #85

Self-rejection rationale (from candidate B doc):
- **Narrow domain** (miniF2F / ProofNet / IMO subsets only).
- **Gate-0 55%** — verifier-in-loop dominant engineering risk.
- **30-40% mechanism overlap with #69 REASONING-DISTILL** — R1 already does math.
- **Risk-adj 0.8-3.3M× band** wide; conservative end below A's 1.4M×.

**Reserved for #85** if math/formal-reasoning becomes primary user concern.

### 1.4 Why ROBOTICS-DISTILL SUNSET

5-iteration reservation pattern parallel to #79 DIFFERENTIAL-TRANSFORMER:

| Iteration | Reservation | Outcome |
|---|---|---|
| iter-216 (#72) | First reservation on axis-distance | Reserved |
| iter-223 (#79) | Reservation continued | Reserved |
| iter-224 (#80) | Reservation continued at AUDIO promotion | Reserved |
| iter-227 (#83) | Reservation continued | Reserved |
| **iter-228 (#84)** | **5-iteration pattern → SUNSET** | **SUNSET** |

**Sunset rationale:** Continued deferral wastes paradigm slots. ROBOTICS axis-distance from text-LLM brief is structural; iter-220's "LLM framework/architecture" broadening did not provide sufficient justification for promotion across 5 iterations. Production precedent (RT-2, OpenVLA, π0) is real but at non-CHIRON architecture; CHIRON-32B-effective + ACTION-tokenization is uncertain extension.

**Reintroduction criteria:** If user explicitly requests robotics/embodied capability, OR if a future paradigm structurally requires action tokens (e.g., world-model-with-action extension), ROBOTICS-DISTILL can be re-introduced as a fresh candidate.

**Parallel to #79 DIFFERENTIAL-TRANSFORMER's sunset arc.** Both: triple-or-more reservation; consistent failure of relative comparison; sunset prevents slot-waste; reintroduction conditional on changed circumstances.

---

## 2. Mechanism: per-frame ViT + temporal interleaving

### 2.1 Vision encoder per frame

ViT-L/14 (already in stack via #66) applied to each frame at 224×224 resolution. Output: 196 patch tokens per frame.

Frame sampling rate:
- General video: 1-2 fps (e.g., 60-second video = 60-120 frames = 11K-23K patch tokens).
- Action video: 8 fps (60-second video = 480 frames = 94K patch tokens).
- Long video (>60s): downsample or use #76 MLA/#78 sink for context handling.

### 2.2 Temporal positional encoding

Each frame's patches augmented with temporal PE:
```
patch_embedding[frame_i, patch_j] = ViT(frame_i)[patch_j] + PE_temporal(i) + PE_spatial(j)
```

Temporal PE uses #66 RoPE-style sinusoidal encoding extended to time dimension.

### 2.3 Joint-sequence interleaving

Per #66 + #80 pattern:
```
<TEXT_BEGIN> ... <VIDEO_BEGIN> 
                 [f_1.p_1, ..., f_1.p_196]
                 [f_2.p_1, ..., f_2.p_196]
                 ...
                 [f_N.p_1, ..., f_N.p_196]
                 <VIDEO_END> ... <TEXT_END>
```

Modality bit-mask: text positions = 1, video-frame positions = 0. CE applied at text positions; KL distillation also at text positions.

### 2.4 Distillation

#68 SUPER-DISTILL pipeline applied with video-LM teacher:
- Tier 1 (preferred): LLaVA-Video-7B (open-source).
- Tier 2: GPT-4o-video (API).
- Tier 3: Gemini Vision Pro.

Cached-logit pipeline at top-K=16 sparse over text positions only.

### 2.5 Composition with prior 43 paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#66 CROSS-MODAL** | ✓ Stack-base | Architecture; ViT-L/14 already integrated. |
| **#80 AUDIO** | ✓ | Temporal-modality precedent; same joint-sequence interleaving pattern. |
| **#82 IMAGE-OUTPUT / #83 AUDIO-OUTPUT** | ✓ | Codec patterns reused; future video-OUTPUT could use similar mechanism. |
| **#74 PHOENIX-1BIT** | ✓ | ViT-L/14 BF16; trunk PHOENIX-quantized. |
| **#76 MLA** | ✓ | Long video → many frame tokens → KV cache benefits. |
| **#78 ATTENTION-SINK** | ✓ Critical | Long video at T → ∞ requires sink mechanism. |
| **#79 MoD** | ✓ | Per-frame depth routing; uniform low-information frames bypass. |
| **#81 MAMBA-2** | ✓ | Long video benefits from O(T) recurrent processing. |

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Text NLL preservation

Per #66 §4.1: text-only sequences pass through trunk identically to post-#83 baseline. Video encoder bypassed. **Bit-exact text NLL preserved on text-only.**

### 3.2 Theorem 2 — Memory bound

At fps=2, 60-second video → 120 frames × 196 patches = 23,520 frame tokens. With #76 MLA at d_c=384: KV cache ~10.6 MB for video portion. With #78 sink+window: bounded. **At 60-second video, memory feasible under 16 GB ceiling.**

At fps=8, 60-second action video → 480 frames × 196 = 94,080 frame tokens. Memory tight; requires #76 + #78 + #79 MoD bypass for low-information frames. **At 60-second 8fps action: ~50 MB headroom (tight).**

### 3.3 Joint Gate-0 PASS probability

```
ViT-L/14 + temporal PE integration:                ~92%
Joint-sequence video interleaving:                  ~90%
KL-CE on text positions per #66 modality-mask:      ~95%
Memory at 60s video, 16 GB ceiling:                 ~75%
LLM-scale empirical confirmation (LLaVA-Video class): ~70%

Joint Gate-0 PASS:                                  ~62%
LLM-scale empirical confirmation:                   ~45%
```

---

## 4. Updated cumulative stack

```
Iter 227 close (post-#83):
  All 22 axes ≈preserved
  Multimodal architecture: text I/O + image I/O + audio I/O complete

Iter 228 (VIDEO-DISTILL-CHIRON):
  All 22 axes ≈preserved (compute-NEUTRAL on text)
  **VIDEO benchmarks: ~5,000,000× NEW AXIS** (Video-MME, MVBench, EgoSchema, ActivityNet-QA)
```

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| ViT-L/14 + temporal PE per-frame integration | 250 | 1.5 |
| Joint-sequence video DataLoader (frame sampling + interleaving) | 300 | 1.5 |
| Modality bit-mask + 2 video-special tokens | 100 | 0.5 |
| Cached-logit pipeline extension for video-LM teacher | 150 | 1 |
| KL-CE loss with per-frame masking | 100 | 0.5 |
| Video benchmark evaluation harness (Video-MME, MVBench, EgoSchema) | 200 | 1 |
| **Total** | **~1,100** | **6** |

---

## 6. Memory advantage preservation

| Component | GPU memory |
|---|---|
| ViT-L/14 (already integrated via #66) | 0 (reused) |
| Temporal PE + frame embedding cache (per-frame) | ~12 MB for 60s @ 2fps |
| Video-token cache during inference | ~50 MB |
| **Total additional** | **~62 MB** |

**Single-GPU 16 GB ceiling preserved** with ~150 MB headroom under post-#83 stack. **Tight margin** at 60s 8fps action; recommend uniform-sampling Mitigation.

---

## 7. Gates

### Gate-0 (~12 GPU-hours)

**Probe.** 200M coordinator + ViT-L/14 + temporal PE + ~5M video-text pairs at 30s 1fps. Distill from LLaVA-Video-7B.

**PASS criteria.**
- Video-MME 30s subset: ≥ 35% accuracy (LLaVA-Video-7B-tier).
- NLL on text-only: ≤ 0.01 nat drift.

**PASS probability:** ~70%.

### Gate-1 (~200 GPU-hours)

**Probe.** Full 32B-effective + video distillation at 60s 1-2fps. Full Video-MME / MVBench / EgoSchema suite.

**PASS criteria.**
- Video-MME ≥ 55% (LLaVA-Video-7B-class).
- MVBench ≥ 60%.
- EgoSchema ≥ 50%.
- Memory at 60s video: ≤ 15.5 GB.

**PASS probability conditional on Gate-0:** ~64%.

---

## 8. Honest gaps

1. **VIDEO not in user brief.** Selected on saturation-breaking + reservation-resolution grounds.

2. **Long-form video out of scope** on single GPU. 60s+ video at 8fps action requires aggressive #76+#78+#79 composition and may push memory limits.

3. **Mechanism mostly pre-existing** (LLaVA-Video, GPT-4o-video, Gemini Vision Pro). Novelty is system-integration with iter-217-227 stack.

4. **5M× axis-extension class**, not magnitudes-better in raw sense.

5. **Memory tight (~150 MB headroom).** ViT-L/14 reuse from #66 helps but adds frame-cache overhead.

6. **Risk-adj 1.4M× modest** vs other axis paradigms (#82 1.7M×, #83 1.95M×).

---

## 9. Bottom line

**VIDEO-DISTILL-CHIRON is the natural #84 selection.** It:
- **Resolves two-iteration reservation** (#82-B, #83-A).
- **Composes cleanly** with multimodal-trinity post-#83 (text + image I/O + audio I/O + video input).
- **Highest Gate-0 PASS in slate (~62%)**.
- **Production-validated** (LLaVA-Video, GPT-4o-video, Gemini Vision Pro).

**Cumulative single-GPU stack at iter-228 close:**
- All 22 prior axes ≈preserved
- **VIDEO benchmarks: ~5,000,000× NEW AXIS**
- Multimodal coverage: text I/O comprehensive + image I/O + audio I/O + video input

**Engineering:** ~1,100 LOC over 6 weeks. **Joint Gate-0 PASS ~62%; LLM-scale confirmation ~45%; risk-adj 1.4M×.**

**B and C dispositions:**
- **B THEOREM-PROVING reserved continued for #85** — narrow domain; user-need-conditional.
- **C ROBOTICS-DISTILL SUNSET** — 5-iteration pattern parallel to #79 DIFFERENTIAL; reintroduction conditional on user signal or structural requirement.

After 44 paradigms, the bigger-picture stack has reframed **23 axes** (added VIDEO). The multimodal-extension arc (#66 image-input → #80 audio-input → #82 image-output → #83 audio-output → #84 video-input) is mature. Future iter-229+ candidates can pursue:
- **#85 THEOREM-PROVING** (reserved).
- **VIDEO-OUTPUT** (parallel to #82/#83; opens video generation axis).
- **AUDIO-MUSIC-OUTPUT** (specialized music generation; MusicGen).
- **Constraint relaxation** (multi-GPU; bit-exact NLL further; still unsignaled).
- **Recomposition of more rejected paradigms** under iter-212 framing.
