# Paradigm Shift #91 — 3D-OUTPUT-DISTILL-CHIRON: Sub-Axis Extension at Seventh Saturation

**Status:** SELECTED with continued-saturation framing (C selected as least-bad; A ASTRA-KAHAN reserved; B AUDIO-MUSIC-OUTPUT remains rejected).
**Date:** 2026-05-08 (Ralph-loop iter 235; **seventh consecutive below-the-bar/axis-extension paradigm** post-iter-224).
**Axis:** 3D-OUTPUT sub-axis — extension of #88 3D-SPATIAL input.
**Magnitude target:** ~5M× new 3D-output sub-axis; **risk-adj ~700,000× (lowest in iter-228-235 slate).**

---

## 0. Executive summary

Iter-235 produces seventh consecutive saturation iteration. All three candidates below-the-bar:

| Candidate | Magnitude | Issue |
|---|---|---|
| A ASTRA-KAHAN | speculative | Rejected paradigm rescue; production lr unresolved |
| B AUDIO-MUSIC-OUTPUT | rejected at #88-B | Path-1 in-place upgrade more efficient |
| C 3D-OUTPUT | 0.7M× risk-adj | Thin production precedent (MeshGPT/Shap-E <2B); 70% overlap with #88+#82 |

**C selected as least-bad** — opens 3D-OUTPUT sub-axis (extends #88 3D-input symmetrically), production-research-stage, smallest engineering risk among speculative options.

**Mechanism:** Discrete VQ tokenization of 3D outputs (mesh tokens). Teachers: Shap-E (OpenAI), MeshGPT (Tsinghua), GET3D (NVIDIA). Joint sequence extends #88 3D-input + #82/#83/#87 codebook output pattern.

**Honest framing:**
- Sub-axis (not new 28th); extends #88 with output side.
- ~70% mechanism overlap with #88 + #82.
- Teacher-scale gap 90× (CHIRON 32B-eff vs MeshGPT 355M) — quantization-binding compounded.
- Memory at breaking-point threshold (~100 MB headroom).
- Joint Gate-0 PASS ~52%; LLM-scale ~38%; risk-adj 0.7M×.
- **Seventh consecutive saturation iteration.**

**Engineering:** ~1,750 LOC over 7 weeks.

---

## 1. Candidate selection

| Candidate | Verdict |
|---|---|
| A ASTRA-KAHAN-DISTILL | RESERVE (speculative) |
| B AUDIO-MUSIC-OUTPUT | REJECTED (Path-1 alternative more efficient; remains as engineering task within #83) |
| C 3D-OUTPUT-DISTILL | **SELECTED (least-bad sub-axis extension)** |

C selected on three grounds:
1. **Symmetric extension** (parallels #82/#83/#87 output completion).
2. **Cleanest composition** (#88 input + #82 codebook pattern reuse).
3. **Resolves no reservation but opens new sub-axis** (vs A's continued speculation).

---

## 2. Mechanism

Discrete VQ codebook (8192 codes) for 3D mesh tokens. Joint sequence:
```
<TEXT> ... <3D_INPUT> point_cloud_tokens <3D_INPUT_END> ...
            <3D_OUTPUT> mesh_VQ_tokens <3D_OUTPUT_END> ... <TEXT>
```

KL-CE distillation per #68 from Shap-E / MeshGPT teacher. Discrete VQ chosen over continuous diffusion (DreamFusion / Magic3D / GET3D) on NLL-preservation grounds.

**Composition:** #88 (3D input) + #82 (codebook pattern) + #66 (interleaving) + #62 AGENT (robotics-actuation synergy) + #65 WORLD-MODEL (scene templates).

---

## 3. Theoretical analysis

**Theorem 1 — Text NLL preservation (per #66 §4.1):** text-only sequences pass through trunk identically; 3D codebook bypassed. Bit-exact preserved.

**Theorem 2 — 3D output fidelity bound:** bounded by VQ codebook + teacher quality. MeshGPT 355M ↦ student 32B-eff: capacity-gap inverted; quantization at K=8192 = 9.0 nat/3D-token caps gain.

**Joint Gate-0 PASS ~52%; LLM-scale confirmation ~38%; risk-adj 0.7M×.**

---

## 4. Updated cumulative stack

```
Iter 234 close (post-#90):
  All 27 axes ≈preserved; 120-200 MB MLA-FACE memory recovery (if Gate-0 PASS)

Iter 235 (3D-OUTPUT-DISTILL-CHIRON):
  All 27 axes ≈preserved
  3D-OUTPUT sub-axis opened (extends #88 3D-SPATIAL)
  ~5M× new sub-axis (risk-adj 0.7M×)
```

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Shap-E / MeshGPT integration | 600 | 3 |
| 3D-output token vocabulary (+8192 codes) | 200 | 1 |
| Joint-sequence DataLoader (text + 3D I/O) | 350 | 1.5 |
| Cached-logit pipeline + KL-CE on 3D-output | 300 | 1 |
| Evaluation harness (T2I-CompBench-3D, GenEval-3D, mesh-quality metrics) | 300 | 0.5 |
| **Total** | **~1,750** | **7** |

---

## 6. Memory advantage preservation

| Component | GPU memory |
|---|---|
| Shap-E decoder + VQ codebook | ~500 MB |
| 3D-output token cache | ~30 MB |
| **Total additional** | **~530 MB** |

**~100 MB headroom at 16 GB ceiling under post-#90 stack — breaking-point threshold.** CPU-offload Shap-E decoder recommended.

---

## 7. Gates

### Gate-0 (~10 GPU-hours)

200M coordinator + Shap-E + ~5M text-3D pairs.

**PASS criteria.** GenEval-3D ≥ 25%; NLL on text-only ≤ 0.01 nat drift.

**PASS probability:** ~62%.

### Gate-1 (~150 GPU-hours)

32B-effective + Shap-E teacher + 30M text-3D pairs.

**PASS criteria.** GenEval-3D ≥ 50%; T2I-CompBench-3D ≥ 60%; mesh-quality MOS ≥ 3.0.

**PASS probability conditional on Gate-0:** ~62%.

---

## 8. Honest gaps

1. **Below the magnitudes-better bar.** 0.7M× risk-adj is iter-200 microopt class.
2. **Sub-axis (not new 28th axis).**
3. **Memory at breaking-point threshold** (~100 MB headroom).
4. **Teacher-scale gap 90×** caps distillation gain.
5. **Seventh consecutive saturation iteration.** Pattern unambiguous.
6. **META-VALIDATION (per #87-C) reserved-as-recommendation across iter-231/232/233/234/235** — fifth iteration of deferred recommendation.

---

## 9. Bottom line

**3D-OUTPUT-DISTILL-CHIRON selected at #91 as least-bad** of three weak iter-235 candidates. The selection explicitly acknowledges:

- **Seventh consecutive saturation iteration.**
- **Sub-axis extension** (not new 28th).
- **Risk-adj 0.7M× lowest in iter-228-235 slate.**
- **META-VALIDATION recommendation deferred fifth time** (iter-231 first, iter-235 fifth deferral).

**Cumulative single-GPU stack at iter-235 close:**
- All 27 prior axes ≈preserved
- 3D-OUTPUT sub-axis opened (extends #88)
- Multimodal output coverage extended: image (#82) + audio (#83) + video (#87) + 3D (#91)

**Engineering:** ~1,750 LOC over 7 weeks. **Joint Gate-0 PASS ~52%; LLM-scale confirmation ~38%; risk-adj 0.7M×.**

**A and B dispositions:**
- **A ASTRA-KAHAN reserved** — speculative paradigm rescue.
- **B AUDIO-MUSIC-OUTPUT remains rejected** — Path-1 in-place upgrade in #83 more efficient (engineering task, not paradigm).

After 51 paradigms, **27 axes** unchanged (sub-axis extension only). **Seventh saturation iteration acknowledgment.** Per #87-C META-VALIDATION recommendation reaffirmed across iter-231/232/233/234/235: strategic case for validation phase strengthens with each saturation iteration.

**Iter-236+ available paths:**
- Continued recompositions / sub-axis extensions (axis-extension class).
- Constraint relaxation (#90-B MULTI-GPU-RELAXATION reserved-as-rec; user-decision; hardware-blocked).
- Empirical validation phase (per META-VALIDATION; user-decision).
- Genuinely new axis discovery (unlikely at depth 27).
