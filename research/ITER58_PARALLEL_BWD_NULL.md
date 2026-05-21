# Iter 58 — `--bf16-logits-parallel-bwd` revisit + structural-wall acknowledgment — NULL

**Date**: 2026-05-16
**Iter**: 58 (twelfth iter under "stacking-wins" brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: NULL — re-tested iter6's `--bf16-logits-parallel-bwd` on the iter49 stacked flagship.  Same NULL result as iter6 (+0.15%, bench noise).  Cross-stream cuBLAS still doesn't overlap on Ada at this scale.

---

## TL;DR

Iter 54 META cataloged that no remaining solo engineering target ≥+5% exists in the iter49 flagship.  iter55 confirmed paradigm flags (`--cuda-graphs`, `--fp8-attn`) don't move the needle.  iter56 confirmed the too-small-target pattern (warp argmax +1.62%).  iter57 punted on the fused accum/sumsq kernel due to multi-microstep semantics.

Iter 58 tests one more historically-NULL flag in case the iter49 occupancy picture changes it:

| variant                       | tok/s @ step 50 | Δ vs iter49 |
|---                            |---:             |---:         |
| iter49 baseline (no flag)     | 43,287          | —           |
| iter49 + `--bf16-logits-parallel-bwd` | 43,352  | +0.15% (bench noise) |

Same NULL as iter6 (2026-05-14 measurement on then-baseline).  Side cuBLAS handle / CUDA-stream for the dE backward GEMM doesn't overlap with the main-stream dq_L GEMM on Ada at the readout shape (T=8192, V=32000, m=2048).  Both GEMMs use full SM occupancy individually; concurrent dispatch produces serialization rather than overlap.

---

## Structural-wall acknowledgment (third META in three iters)

Engineering attempts since iter49:

| iter | result | win | notes |
|---: |---     |---: |---    |
| 50  | FAIL | +1.31% | bench noise + NLL drift |
| 51  | FAIL | +4.46% | below-bar, real win, NLL-safe |
| 52  | NULL | 0%    | cuBLAS auto-pick optimal |
| 53  | FAIL | +2.86% | LN-bwd 2-phase, NLL-borderline |
| 54  | META | n/a   | cataloged dead-zones |
| 55  | NULL | ~0%   | paradigm flags incompatible/no-effect |
| 56  | FAIL | +1.62% | warp argmax, too-small-target |
| 57  | PUNT | n/a   | accum-sumsq fusion: multi-step semantics |
| 58  | NULL | +0.15% | parallel-bwd: same NULL as iter6 |

**8 consecutive non-PASS iters.**  The brief's 1.5×-by-iter-5 target was +50% by iter 5; we have +12.5% by iter 12 (achieved entirely by iter47+49).  The 10×-by-iter-20 target is clearly out of reach absent paradigm-level change.

---

## What the data says

The dead-zone categorization (from iter54 + iter56 + iter57):

| GPU time slice | category | status                                            |
|---:           |---       |---                                                |
| 40%           | cuBLAS GEMMs | DEFAULT auto-pick optimal (iter52); FP8 doesn't bite (iter55) |
| 15%           | SCFA element-wise | mem-bound 80-98% peak (iter50 hit fp32-FMA drift) |
| 6.6%          | LN backward | too-small-target (iter48+iter53 confirmed)        |
| 8%            | Adam+cast | iter49 saved the big chunk; remainder is mem-bound |
| 3.7%          | bf16_accum + sumsq | semantically blocked for accum>1 (iter57) |
| 2.0%          | argmax (logging) | too-small-target (iter56)                    |
| ~25%          | other smaller (<2% each) kernels |                                  |

Every solo target ≥2% has been attacked.  Combined wins under a relaxed bar:

| if shipped         | win |
|---                 |---: |
| iter51 (+4.46%)    | +4.46% |
| iter53 (+2.86%)    | +2.86% |
| iter56 (+1.62%)    | +1.62% |
| Combined (stacked) | +9.6% (compounding) |

---

## What the user should do

Three options remain (no change since iter54):

**A. Relax the bar to +3% (or even +2%)**: ships iter51 + iter53 + iter56 for ~+10% additional cumulative, bringing the stack to **~+22% total**.  Empirically-aligned with what's available.  Doesn't reach the 1.5× target but is honest forward progress.

**B. Commit to a multi-iter paradigm arc**: pick one of {BF16 residual stream (q+p), FP8 readout via cuBLASLt with restructured amax caching, sparse attention, vocab subsampling}.  Each is 3-5 iters of implementation with NLL parity at risk.  Could yield +5-15% per arc but with significant failure risk.

**C. Reframe the brief itself**: same kind of empirical realignment that the iter47 reframe did to the prior "magnitudes via architectural shift" framing.  After 5 paradigm falsifications, the prior brief was lowered to "stacking 5% wins"; after 8 consecutive non-PASS engineering iters, the current bar could be lowered to "+2-3% wins" or paradigm work could be promoted to first priority.

---

## What this iter does NOT change

No code change.  No flag toggle.  No default flip.  Cumulative shipped stays at **+12.5%** (iter47 × iter49).

This iter's output is one more data point in the dead-zone map and one more explicit ack that the loop has hit a structural wall.

---

## Default action without user input

The loop will continue.  Iter 59 will pick another tiny optimization and likely produce another FAIL/NULL.  Without user direction:
- Engineering wins continue to FAIL at the strict bar.
- No paradigm arc can start (multi-iter commitment requires user buy-in).
- No bar change can happen (the brief is the brief until updated).

I will keep producing iter docs until the user redirects.

---

## Sequence status (full)

| iter | target | result | win |
|---: |---     |---     |---:  |
| 47  | SCFA dwconv-dK | PASS | +5.20% |
| 48  | LN-bwd dgamma  | FAIL | +2.4% |
| 49  | Fused Adam-int8 | PASS | +6.98% |
| 50  | SCFA sub-conv  | FAIL | +1.31% |
| 51  | --scfa-checkpoint-inner | FAIL | +4.46% |
| 52  | cuBLAS algo override | NULL | 0% |
| 53  | LN-bwd 2-phase | FAIL | +2.86% |
| 54  | (meta-analysis) | META | n/a |
| 55  | --cuda-graphs / --fp8-attn | NULL | ~0% |
| 56  | warp argmax | FAIL | +1.62% |
| 57  | accum_axpy+sumsq fused | PUNT | n/a |
| 58  | --bf16-logits-parallel-bwd revisit | NULL | +0.15% |

Cumulative shipped: **+12.5%**.  12 iters: 2 PASS / 6 FAIL / 3 NULL / 1 META / 1 PUNT.

Score 2/12 PASS (17%).  Hit rate trending down; not statistically rebounding.
