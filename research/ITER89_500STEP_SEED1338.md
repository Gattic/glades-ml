## Iter 89 — 500-step trajectory at seed=1338 — overturns iter 88 pessimism

**Date**: 2026-05-20
**Iter**: 89 (thirty-fifth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **MEAN PARITY HOLDS AT N=2 500-step MULTI-SEED**.  iter 88's single-seed=1337 result (triple-stack +0.057 worse) is REVERSED at seed=1338 (triple-stack −0.031 BETTER).  Multi-seed mean Δ at 500 steps = **+0.013 nat** (within ±0.05 multi-seed bound, just outside ±0.02 strict).  Wall +3.43% consistent across both seeds.

---

## Bench (baseline + triple-stack at seed=1338, 500 steps each)

| step | baseline | triple-stack | Δ |
|---:  |---:      |---:          |---: |
| 0 (init) | 10.4720 | 10.4727 | +0.0007 |
| 100 | 8.6082 | **8.4314** | **−0.177** (triple AHEAD) |
| 200 | 7.7917 | **7.6370** | **−0.155** |
| 300 | 6.8285 | 6.8190 | −0.010 |
| 400 | 6.7716 | 6.8935 | +0.122 |
| **500** | **6.1887** | **6.1579** | **−0.031** (triple BETTER) |

Wall: 341.3 s baseline vs 329.9 s triple-stack = −3.34% wall (+3.46% tok/s).

## Combined picture across seeds (iter 88 + iter 89, n=2)

| seed | baseline NLL @ 500 | triple-stack NLL @ 500 | Δ |
|---:  |---:                |---:                     |---: |
| 1337 (iter 88) | 6.1184 | 6.1749 | **+0.057** (triple WORSE) |
| 1338 (iter 89) | 6.1887 | 6.1579 | **−0.031** (triple BETTER) |
| **mean (n=2)** | **6.1536** | **6.1664** | **+0.013** |

**Multi-seed mean Δ at 500 steps = +0.013 nat** — within ±0.05 multi-seed bound (just outside ±0.02 strict).

Wall improvement consistent: −3.31% wall, +3.43% tok/s across both seeds (vs iter 87's +3.46% at 200 steps).

## Seed dependence picture

Across seeds, the trajectory pattern DIFFERS significantly:

**Seed 1337** (iter 88):
- 0→100: triple ahead (−0.126)
- 100→300: baseline catches up and pulls ahead (peak +0.069 deficit @300)
- 300→500: triple stays behind (+0.057 final)
- Classic iter 41 pattern (warm-up advantage gives way to capacity deficit)

**Seed 1338** (iter 89):
- 0→200: triple consistently ahead (−0.155 to −0.177)
- 300→400: brief deficit (+0.122 @400)
- 500: triple recovers and BEATS baseline (−0.031)
- NOT iter 41 pattern — triple-stack maintains advantage

The 500-step trajectory is **highly seed-dependent**.  Some seeds favor triple-stack (1338); others don't (1337).

## Reconciling with iter 87 (200-step n=5)

At 200 steps, iter 87 showed mean Δ −0.019 (within strict).  At 500 steps, n=2 shows mean Δ +0.013.  Both are within ±0.05 multi-seed bound.

The TREND from 200 → 500 steps is +0.032 (slight worsening of mean parity).  If this trend continues to 30k steps, extrapolated final drift could be 0.05-0.15 nat — outside strict, possibly within iter 60 relaxed.

But trend extrapolation from 2 data points is highly uncertain.  Multi-seed at longer horizons (1k, 2k, 5k steps) would tighten the extrapolation.

## Updated retrain arc viability assessment

| evidence level | data | retrain arc viability |
|---             |---   |---                    |
| iter 87 (200-step n=5) | mean Δ −0.019 | high confidence |
| iter 88 (500-step n=1 seed=1337) | Δ +0.057 | low confidence |
| **iter 89 (500-step n=2)** | **mean Δ +0.013** | **medium confidence** |

The picture has stabilized at medium-confidence: multi-seed mean parity holds (within ±0.05) at 500 steps, but variance is high and the trend slightly worsens with steps.

For production retrain commitment, would want:
- Multi-seed at 1k+ steps (n=3-5)
- Confirm trend stabilizes vs continues worsening
- If multi-seed mean Δ stays within ±0.05 at 1k+ steps, commit to 30k retrain

## Default policy

No code change.  Opt-in flags preserved.  Multi-iter arc remains a candidate but requires additional longer-horizon validation before commitment.

## Sequence status (35 iters)

| iter | result | horizon | n | NLL Δ |
|---: |---     |---:     |--:|---:   |
| 87  | PASS at strict | 200 steps | 5 | −0.019 |
| 88  | FAIL single-seed | 500 steps | 1 | +0.057 |
| 89  | BORDERLINE multi-seed | 500 steps | 2 | **+0.013** |

iter 89 confirms multi-seed mean parity holds at 500 steps (within ±0.05).  iter 88's pessimism was seed-specific; broader picture is closer to iter 87's optimism but slightly worse.

## Recommended next action

The triple-stack retrain arc viability depends on the trend at 1k+ steps.  Lower-commitment next probe: **iter 90 — 1k-step trajectory at seed=1337 only** (the unfavorable seed).  If seed=1337 at 1k steps either:
- Stabilizes its deficit (parallel to baseline): arc viable
- Continues growing deficit: arc fails

Time: 2 runs × 12 min = 24 min.  Substantial but lower than full multi-seed 1k.

Or: pause autonomous loop and present cumulative state to user for retrain arc decision.

## Files

- This document.
- `research/runs/2026-05-20-iter89-bench/baseline_500step_seed1338.log`
- `research/runs/2026-05-20-iter89-bench/triple_500step_seed1338.log`
- No code change.
