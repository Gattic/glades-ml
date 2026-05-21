# Iter 56 — Warp-parallel argmax_count_bf16 — Gate-0 FAIL (below-bar)

**Date**: 2026-05-16
**Iter**: 56 (tenth iter under "stacking-wins" brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: FAIL — +1.62% tok/s.  Clean engineering optimization (argmax kernel goes from 2.0% → ~0.1% of GPU time), train-accuracy correct, NLL untouched (kernel is purely a logging path), but kernel slice is below the +5% bar's reach.

---

## TL;DR

The legacy `argmax_count_bf16_kernel` used 1 thread per row, looping V=32000 sequentially to find the per-row argmax for train-accuracy logging.  Each call took ~3.5 ms.  At 50 instances per 50 steps (one per training step), that's 175 ms = 2.0% of GPU time.

Iter56 rewrites it as 1 warp per row with shuffle-reduce argmax.  Each warp's 32 lanes do a strided scan over V (lane handles v=lane, lane+32, lane+64, ...) then a 5-step `__shfl_down_sync` reduction to find the global max+arg.  Block is 32 warps = 1024 threads = 32 rows.

Result (3-run variance, 50-step bench, iter49 stacked flagship config):

| variant         | tok/s mean (3 runs) | Δ vs iter49 |
|---              |---:                 |---:         |
| iter49 baseline | 43,287              | —           |
| iter56 (warp argmax) | 43,989         | **+1.62%**  |

Kernel-level: argmax kernel time dropped from ~3.5 ms/call to ~0.1 ms/call (≈30× kernel speedup as expected from 32× warp parallelism).  Overall savings: 1.9% (kernel slice was 2.0%).

Below the +5% bar.  Reverted; iter49 stays default flagship.

Train-accuracy values are identical to baseline (acc=0.14 at step 50 across all runs — correct).  NLL bit-identical (kernel doesn't touch training math).

---

## Why this falls in the same pattern as iter48 / iter53

The argmax kernel was a 2.0% slice — even a perfect 100% reduction caps at 2.0% overall.  Same "too-small-target" failure mode as iter48 (4.5% LN-dgamma → +2.4%) and iter53 (same 4.5% → +2.86%).

Confirmed pattern after 10 iters: **no remaining solo kernel slice ≥+5%-reachable exists in the iter49 flagship** absent paradigm-level changes.

---

## What was attempted (reverted)

`Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`:
- Added `argmax_count_bf16_warp_kernel` next to the legacy `argmax_count_bf16_kernel`.
- Modified `argmax_count_matches_bf16` wrapper to dispatch the new kernel (block=1024, grid=(T+31)/32).
- Legacy kernel kept as dead code.

Reverted.

---

## Sequence status

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

Cumulative shipped: **+12.5%** (1.052 × 1.070).  10 iters: 2 PASS / 6 FAIL / 2 NULL / 1 META.

**6 consecutive iters now below the bar** (47 PASS, 48 FAIL, 49 PASS, 50/51/53/56 FAIL, 52/55 NULL, 54 META).

---

## What this tells us (and what iter 57+ should do)

The empirical pattern is overwhelming.  Under the strict +5% bar with the iter49 flagship's bottleneck composition:

- Every kernel slice ≥7%: cuBLAS-shape, auto-pick optimal.
- Every kernel slice 2-5%: kernel-level optimizable but the slice caps the overall gain.
- Element-wise mem-bound kernels (15% combined): individually <6%, fusing them runs into fp32-FMA precision drift (iter50).
- Existing paradigm flags: structurally incompatible (iter55) or bottleneck-shifted-not-resolved (iter55 fp8).

The remaining engineering attempts will continue producing +1-3% FAIL/NULL results unless we either:
- Relax the bar (iter54 META Option A).
- Build a multi-iter paradigm arc (BF16 residual stream, FP8 readout with amax caching restructured, sparse attention).
- Reframe the brief (iter54 META Option C).

Iter 57 will continue the current pattern unless something changes.

---

## Reproducibility

```bash
build/glades_chiron_train --data-dir pretok-data --pretokenized --vocab 32000 \
  --seq-len 8192 --m 2048 --layers 12 --heads 16 --dhead 256 \
  --lr 3e-4 --max-steps 50 --warmup 5 --grad-clip 1.0 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage \
  --seed 1337
```
