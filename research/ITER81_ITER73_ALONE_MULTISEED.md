## Iter 81 — iter 73 alone at n=3 multi-seed — parity-clean(ish), wall too small

**Date**: 2026-05-20
**Iter**: 81 (twenty-seventh iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **FAIL by wall bar** — iter 73 alone shows mean NLL drift **−0.035 nat** (BETTER direction) with std **0.0125** (TIGHTER than baseline std 0.016) at n=3, but wall gain is only **+0.42% tok/s** — below any ship bar by a wide margin.  Reverses iter 76's inference that iter 73 was the drift source in combined; iter 70 (not iter 73) drives the combined instability.

---

## Bench (iter 73 alone, n=3)

Each run: 200 steps, L=24 T=16384, --iter73-dwconv-fwd-tiled (no iter 70).

| seed | NLL @ step 200 | wall (s) |
|---:  |---:            |---:      |
| 1337 | 7.5668 | 135.7 |
| 1338 | 7.5915 | 135.9 |
| 1339 | 7.5824 | 135.9 |
| **mean (n=3)** | **7.5802** | **135.83** |
| **std** | **0.0125** | 0.12 |

## Comparison to baseline + iter 70 alone

| metric | baseline (n=4) | iter 70 alone n=5 (iter 80) | iter 73 alone n=3 (iter 81) |
|---     |---:            |---:                          |---:                          |
| NLL mean | 7.6154 ± 0.016 | **7.6437 ± 0.088** | **7.5802 ± 0.013** |
| NLL Δ vs baseline | — | +0.028 (outside strict, +) | **−0.035 (outside strict, −)** |
| NLL std vs baseline | 1× | 5.5× | **0.8×** (tighter!) |
| wall | 136.40 | 134.66 (−1.27%) | 135.83 (−0.42%) |
| wall Δ | — | **+1.27%** | **+0.42%** |

## Surprising findings

1. **iter 73 alone has TIGHTER NLL variance than baseline** at n=3.  Std 0.013 vs baseline 0.016.  This is the opposite of iter 70 (5.5× higher) and combined iter 70+73 (3.3× higher).

2. **iter 73 alone drifts NLL in the BETTER direction** (mean −0.035 nat).  Consistent across all 3 seeds (range 7.5668-7.5915).  Below baseline mean.

3. **iter 73 alone wall gain is +0.42%** — much smaller than iter 70's +1.27%.  Sub-multiplicative when combined: iter 70+73 = +1.71% combined vs 1.27% + 0.42% = +1.69% expected.  Stacking is near-additive.

4. **iter 76's combined drift was driven by iter 70, NOT iter 73**.  This reverses iter 73's earlier "FMA-emit drift source" classification.  iter 73's `#pragma unroll`-induced FMA changes do produce different math, but in a CONSISTENT (better-direction) way that doesn't introduce seed-dependent instability.

## Strategic implications

The relative stability picture is now clearer:

| mechanism | wall | NLL drift (n>=3) | NLL std vs baseline | seed-dependent? |
|---        |---:  |---:              |---:                  |---              |
| iter 70 alone | +1.27% | +0.028 (outside) | 5.5× | **YES (seed 1341 outlier)** |
| iter 73 alone | +0.42% | −0.035 (outside) | **0.8×** (tighter) | **NO** at n=3 |
| iter 70+73 combined | +1.71% | +0.067 (outside) | 3.3× | YES (seed 1338 outlier) |

**iter 73 alone is the BEST below-bar mechanism by NLL parity**, BUT wall gain (+0.42%) is too small to ship.

**iter 70 introduces seed-dependent instability**, NOT iter 73.  This is the opposite of what iter 73's `#pragma unroll` FMA-emit drift classification suggested.

## Why this still fails the per-iter bar

Wall gain: +0.42% (below +5% strict bar AND +3% iter 60-precedent bar by a wide margin).

Even though iter 73 alone has near-perfect parity (tighter than baseline std, drift in BETTER direction), the wall improvement is too small for ship.

## Default policy

`--iter73-dwconv-fwd-tiled` remains **default OFF**.  Opt-in flag preserved.

## Path forward updated

Combining iter 73 (parity-clean, +0.42%) with iter 70 (variance-introducing, +1.27%) produces the +0.067 drift seen in iter 76.  If iter 70's seed-dependent instability could be tamed (e.g., better SR-cast counter scheme), combined wall +1.69% might become parity-clean.

But that's an iter 70 refinement, not a fresh mechanism.  Not currently a high-EV avenue.

## Sequence status (27 iters)

| iter | result | wall |
|---: |---     |---:  |
| 70  | parity at n=3, instability at n=5 | +1.27% |
| 73  | parity-clean at n=3, tight std | +0.42% |
| 70+73 combined | seed-dependent drift | +1.71% |

The +0.42% iter 73 alone is the cleanest below-bar mechanism by parity metric.  Still too small to ship.

12 consecutive non-PASS iters (70-81) confirm engineering ceiling closure.

## Files

- This document.
- `research/runs/2026-05-20-iter81-bench/iter73_alone_seed{1337,1338,1339}.log`.
- No code change.
