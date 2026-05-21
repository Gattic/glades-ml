# 3-paradigm arc synthesis — FP8 readout × BF16 residual-p × MoE attention-shear

**Date**: 2026-05-16
**After**: iter 61 (engineering grind exhausted, brief reframed)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)

This document synthesizes three paradigm proposals (`PARADIGM_FP8_READOUT_DESIGN.md`, `PARADIGM_BF16_RESIDUAL_P_DESIGN.md`, `PARADIGM_MOE_FFN_DESIGN.md`) into a single decision artifact.  It compares them head-to-head, identifies a critical CHIRON-architecture caveat in the MoE proposal, and recommends a dispatch order if the user elects to continue past the iter-61 reckoning.

**Bottom line**: even the optimistic stacking of all three paradigms yields ~+25-35% wall over iter61 (final flagship ~61-66k tok/s, total ~4-4.4× over the ralph-loop-iter-1 baseline of 15.2k tok/s).  **The brief's 10× target remains empirically out of reach.**

---

## 1. Cross-paradigm comparison

| axis                              | FP8 readout                       | BF16 residual-p                          | MoE attention-shear                     |
|---                                |---                                |---                                       |---                                      |
| **target slice**                  | 18% readout GEMMs + storage       | 16% SCFA element-wise on `p` stream      | 36% cuBLAS (incl. attention shear)      |
| **mechanism**                     | E4M3-fwd / E5M2-bwd, FP8 tensor cores | SR-quantized `p`, FP32 internal accum  | top-k routed K-expert attention projections |
| **expected wall win**             | +3% to +6%                        | +3% to +5% iter-bench; risky at production | +10% to +18% (cumulative across arc)  |
| **VRAM delta @ iter-bench**       | **−0.73 GB**                      | **−0.8 GB**                              | +0 (K=4 top-1 d_ff=m, neutral)          |
| **VRAM delta @ production**       | **−0.73 GB**                      | **−3.2 GB**                              | **+2.4 GB** (K=8 top-2 d_ff=2m)         |
| **fits 16 GB @ production**       | yes                               | yes                                      | borderline (16.9 GB → needs mitigation) |
| **iter budget**                   | 4 iters (62-65)                   | 5 iters (62-66)                          | **5-8 iters (62-69)**                   |
| **per-iter Gate-0 sharpness**     | sharp (NLL drift visible @ step 100-200) | sharp at iter-bench, fuzzy at L=24 | medium (load-balance + NLL trajectory)  |
| **infrastructure leverage**       | **HIGH** (paradigm #50 HELIUM scaffolding) | medium (sr_hash32 from iter 49) | LOW (~3-6 kLoC new C++/CUDA)            |
| **ship probability solo**         | ~35%                              | ~50% iter-bench / ~25% production        | ~30% by iter 66                         |
| **ship probability full arc**     | <15% (4 iters joint)              | ~25%                                     | ~20%                                    |
| **falsification clarity**         | iter 62 (single iter)             | iter 65 (production-scale kill test)     | iter 62 (forward-only feasibility)      |
| **flagship risk**                 | none (off-by-default flag)        | none (off-by-default flag)               | none (off-by-default flag)              |
| **published precedent**           | TE / MS-AMP / FP8-LM / GLM-FP8    | **NONE** for reversible-flow + SR-p      | Switch / Mixtral / DeepSeek-V3 / ST-MoE |
| **novel for CHIRON?**             | shape-specific Ada validation     | YES — fully novel theoretical territory  | port to single-GPU + Adam-int8 + reversible-flow stack |

### 1.1 Reading the comparison

- **FP8 readout** is the safest move: contained scope, existing infrastructure, sharpest falsification gate, lowest implementation cost.  Expected upside also lowest.
- **BF16 residual-p** is the highest-uncertainty: novel theoretical territory (no published reversible-flow + SR-p precedent), but production-scale viability is decided by a single 500-step test at L=24.  Worst-case deliverable is a "scale-conditional flag" — research artifact characterizing the depth-precision tradeoff.
- **MoE attention-shear** is the largest commitment with the largest plausible upside.  5-8 iters minimum.  VRAM-bound at production scale.  Subject to a CHIRON-architecture caveat (§2) that the agent design did not handle correctly.

---

## 2. Critical caveat: MoE design assumes a structure CHIRON does not have

The MoE proposal designs a top-k routed mixture replacing the dense FFN block:

> Dense FFN FLOPs (fwd): `16 × T × m²`.
> MoE FFN active FLOPs: 2/8 = 1/4 of dense.

This assumes CHIRON has a separate FFN block with `W1: 4m×m, W2: m×4m`.  **CHIRON does not have this.**  Per `chiron_main.cpp` line 9-11:

```cpp
// Architecture (per layer):
//   p += attention_shear(q ; Wq, Wk, Wv, Wo)    // symplectic shear; q unchanged
//   q  = reln(q ; gamma, beta)                  // reversible layer-norm
```

There is **no FFN block** — only the attention shear with 4 weights (`Wq, Wk, Wv, Wo`).  Each weight has shape `[m, dModel]` where `dModel = 2*m = 4096` (per chiron_main.cpp:183 `dModel = 2*m`, NOT `4*m` as the MoE agent assumed).

Per-layer parameter accounting verifies this:
- 4 weights × `m × dModel = 2048 × 4096 = 8.4M` each = 33.5M params/layer
- L=12 layers × 33.5M + embedding (V × m = 65.5M, tied to readout) = **468M params total** — exactly matches the trainer's reported `params=468.24M`.

If CHIRON had a separate 4m-expansion FFN, the param count would be ~675M, not 468M.

**Implication for the MoE arc**: there are three options for how to apply MoE to CHIRON, all of which differ from the original design:

| variant                              | what's MoE-ified                                       | refactor scope | preserves attention math? |
|---                                   |---                                                     |---             |---                        |
| **(A) Full-shear MoE**               | All 4 attention projections (`Wq_k, Wk_k, Wv_k, Wo_k` per expert), with per-expert attention | very large    | no — different attention per expert |
| **(B) Output-only MoE**              | `Wo` only (the dModel→m output projection)             | small         | yes — same QKV, different out-projection per expert |
| **(C) Add a new FFN block**          | Bolt on a 2m-expansion FFN-like block AFTER the shear, then MoE-ify it | medium       | yes (new block, no change to shear) |

The original MoE design (8 experts × full FFN) maps closest to (A).  (B) is a more conservative interpretation that fits CHIRON's architecture with the smallest change.  (C) adds parameters and changes the model shape.

**The MoE arc needs a preliminary iter (62a)** to choose between (A), (B), and (C) based on:
- Param-budget impact (16 GB ceiling)
- Math-preservation desire (whether to keep the reversible-flow attention shear semantics)
- Implementation cost (B << C < A)

This is design work that the original 5-8 iter budget did not include.  Realistic total: **6-10 iters** for MoE.

---

## 3. Stacking analysis

### 3.1 Orthogonality

The three paradigms target different slices of the GPU profile and different storage tensors:

| paradigm           | tensors changed                | kernels changed                        | conflicts with |
|---                 |---                             |---                                     |---             |
| FP8 readout        | logits, probs, dlogits, q_L (read-only for readout), W_E (read-only for readout) | readout GEMMs (3), softmax-fwd, softmax-CE-bwd, argmax | none |
| BF16 residual-p    | `p`, `dp` (residual streams)   | scfa_axpy2, scfa_sub, generic axpy on p, reln-fwd, reln-inv | none directly; both touch HBM bandwidth budget |
| MoE attention-shear | Wq/Wk/Wv/Wo (per-expert copies) | attention shear forward + backward     | none — attention is decoupled from p stream and readout |

**No fundamental conflicts.**  The three paradigms can stack mechanically.

### 3.2 Combined cumulative

Treating wall improvements as multiplicative (since they're on different slices and saturate different resources):

| scenario              | FP8 (assume) | BF16-p (assume) | MoE (assume) | combined wall multiplier |
|---                    |---:          |---:             |---:          |---:                      |
| optimistic            | 1.06×        | 1.05×           | 1.18×        | **1.31×** (+31%)        |
| realistic mean        | 1.04×        | 1.03×           | 1.10×        | **1.18×** (+18%)        |
| pessimistic (all FAIL but iter-bench wins) | 1.02× | 1.02× | 1.03× | **1.07%** |

Combined with iter61's silent +2.64%: optimistic best case is `1.026 × 1.31 = 1.345×` over iter60 baseline, or **~63,500 tok/s**.  Vs the start-of-ralph-loop baseline (15,200 tok/s): **~4.2×**.

The brief's 10× would require an additional ~2.4× from somewhere.  None of these three paradigms — alone or stacked — closes that gap.

### 3.3 VRAM accounting (production, T=16384, L=24)

Starting from iter61's ~12 GB production estimate (we don't have a fresh production-scale profile; this is extrapolated from iter60's 7.99 GB at iter-bench × ~1.5× for production T+L scaling):

| layer        | iter61 baseline | FP8 alone | BF16-p alone | MoE alone (K=8 top-2 d_ff=2m) | all three |
|---           |---:             |---:       |---:          |---:                           |---:       |
| baseline     | 12.0 GB         | 11.27 GB  | 8.8 GB       | 14.4 GB                       | **10.97 GB** |
| 16 GB headroom | 4.0 GB        | 4.73 GB   | 7.2 GB       | 1.6 GB                        | 5.03 GB   |

Stacking actually **improves** the production VRAM picture: FP8 and BF16-p free ~4 GB combined, of which MoE can spend ~2.4 GB.  Net headroom grows from 4 GB to ~5 GB — production fits comfortably.

---

## 4. Dispatch order recommendation

If the user elects to continue past iter61, the recommended order is **lowest-cost-first**, with explicit kill gates after each arc:

### Arc 1: FP8 readout (iters 62-65, ~4 iters)

- **Why first**: lowest implementation cost (extends existing `gpu_blas_fp8.h` scaffolding), sharpest single-iter falsification, no flagship risk.
- **Kill gate (after iter 62)**: forward-only FP8 readout fails C0a (≥+1.5%) OR C0b (NLL drift ≤+0.02 nat) → arc dead, pivot to Arc 2.
- **Ship gate**: iter 63 full-loop FP8 readout ≥+3% at NLL parity → ship as `--fp8-readout`.
- **Expected outcome**: 35% probability of +3-5% ship; 35% probability of NULL (HELIUM-style algo mismatch); 30% probability of NLL fail.

### Arc 2: BF16 residual-p (iters 66-70, ~5 iters)

- **Why second**: medium complexity, sharp production-scale kill test.  Could ship even if Arc 1 fails — uncorrelated risk.
- **Kill gate (after iter 62 of this arc — i.e., overall iter 66)**: RN-only NLL drift > +0.05 nat OR kernel speedup < +2% → arc dead.
- **Kill gate (after iter 65 of this arc — overall iter 69)**: production-scale L=24 NLL drift > +0.05 nat → arc dead.  Deliverable becomes a scale-conditional flag.
- **Ship gate**: iter 70 ship `--bf16-residual-p` default-on if all 6 G0.* sub-gates pass.
- **Expected outcome**: 50% probability of iter-bench ship, 25% probability of production ship.

### Arc 3: MoE attention-shear (iters 71-79, ~6-10 iters with the design-choice iter 62a)

- **Why last**: highest implementation cost, highest variance, depends on a CHIRON-specific design choice (A/B/C from §2).  Only worth committing to if Arcs 1+2 have shipped something — otherwise the marginal expected value vs cost is poor.
- **Design iter (71a)**: pick variant (A/B/C) based on iter-bench param-budget probe + design review.
- **Iter 72-74 baseline**: implement chosen variant in forward-only, K=4 top-1 d_ff=m, kernel-correctness + routing-overhead measurement.
- **Iter 75-77 scale-up**: backward, LB loss tuning, K=8 top-2.
- **Iter 78-79 production**: revalidate at L=24, ship or revert.
- **Hard falsification at iter 72**: forward-only K=4 top-1 d_ff=m gives <+1% wall at iso-NLL with healthy `f_k` → mechanism dead.
- **Expected outcome**: 20-30% probability of +10-18% cumulative ship after the full arc.

### Total commitment

| if you stop after... | iters | best-case cumulative wall vs iter61 | worst-case |
|---                   |---:   |---:                                 |---:        |
| Arc 1 ships          | 65    | +5%                                 | -3 iters effort, 0% gain |
| Arcs 1+2 ship        | 70    | +9%                                 | -5 more iters effort, 0% gain |
| Arcs 1+2+3 ship      | 79    | +24%                                | -10 more iters effort, 0% gain |

---

## 5. Decision framework — when to stop the arc

Three honest stopping conditions:

1. **Arc 1 (FP8) fails outright** + you've decided the 5-iter cost is not worth the expected 1-3% gain → stop at iter 65, write a paradigm-arc-FAIL summary, freeze the flagship.

2. **Arc 1 ships, Arc 2 fails at iter-bench (iter 66)** → consider whether the +3-5% from Arc 1 alone justifies stopping vs taking Arc 3's higher-variance bet.  Strongly defensible to stop here.

3. **Arcs 1+2 ship, Arc 3 design-iter (71a) reveals VRAM blowout at production for all variants** → ship the Arc 1+2 stack and stop.  This is the most likely "graceful exit" path.

The arc is **explicitly bounded** at iter 79 unless an Arc 3 partial ship makes the next iter clearly worth pursuing.  Open-ended grinds (the iter47-61 pattern) are not recommended.

---

## 6. What this synthesis recommends

**If continuing past iter 61 is worth ~10-20 GPU-hours of implementation + bench time**:
- Dispatch Arc 1 (FP8 readout, iters 62-65).
- Decision point at iter 65: ship or stop.
- If shipped, decide on Arc 2 based on whether the iter61 flagship + Arc 1 ship is enough deliverable.

**If continuing is NOT worth the cost** (the iter 61 reckoning was the final stop):
- Stop here.  The iter47-61 stack is the deliverable.
- These three design docs remain as research artifacts — they document what *would* be tried if the engineering budget were larger.
- The brief reframe (`research/BRIEF_REFRAME_2026_05_16.md`) plus this synthesis are the formal close-out.

**This synthesis does not advocate for one path** — both are defensible.  The deciding factor is whether the user values the marginal +5-25% cumulative win over the 10-30 GPU-hours of additional iter cost.  After 15 iters of grind, the marginal information per iter is lower than it was at iter 47 when the wins were +5-7% per iter.

---

## 7. Cross-references

- `research/PARADIGM_FP8_READOUT_DESIGN.md` — Arc 1 full design (2,712 words)
- `research/PARADIGM_BF16_RESIDUAL_P_DESIGN.md` — Arc 2 full design (2,354 words)
- `research/PARADIGM_MOE_FFN_DESIGN.md` — Arc 3 full design (2,491 words)
- `research/BRIEF_REFRAME_2026_05_16.md` — formal close-out of the iter47-61 engineering grind
- `research/ITER61_BF16_GRAD_DIRECT_FAIL.md` — final engineering iter (+2.64% silent accrual)
- `research/ITER60_RELAXED_BAR_SHIP.md` — last shipped flagship (47,271 tok/s)
