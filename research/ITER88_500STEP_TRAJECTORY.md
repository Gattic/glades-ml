## Iter 88 — Triple-stack 500-step trajectory vs baseline — borderline, tempering iter 87 optimism

**Date**: 2026-05-20
**Iter**: 88 (thirty-fourth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **CAUTION** — at 500 steps (2.5× the Gate-0 horizon), the triple-stack falls **+0.057 nat behind baseline** in single-seed comparison.  Wall improvement holds at **+3.40%** (consistent with iter 87's +3.46% at 200 steps), but the NLL parity story at longer horizon extrapolates UNFAVORABLY.  iter 87's "PASS at iter 60 bar with strict NLL parity" verdict required hedging.

---

## Bench setup

Both runs: seed=1337, 500 steps fresh-init, production stack.  Triple-stack adds `--iter70-fused-axpy2-dual-p --iter73-dwconv-fwd-tiled --scfa-conv-w 4`.

## Per-step val NLL trajectory

| step | baseline | triple-stack | Δ (triple − baseline) | trend |
|---:  |---:      |---:          |---:                   |---    |
| 0 (init) | 10.4676 | 10.4683 | +0.0007 | bit-identical init |
| 100 | 8.5156 | **8.3898** | **−0.126** | triple BETTER (warm-up advantage) |
| 200 | 7.5564 | 7.5994 | +0.043 | triple falls behind |
| 300 | 6.7789 | 6.8480 | **+0.069** | peak deficit |
| 400 | 6.7537 | 6.7628 | +0.009 | stabilizing |
| **500** | **6.1184** | **6.1749** | **+0.057** | persistent gap |

## Wall comparison

| metric | baseline | triple-stack | Δ |
|---     |---:      |---:          |---: |
| Wall (500 steps) | 341.1 s | 329.9 s | −11.2 s (**−3.28%**) |
| tok/s | 24,015 | 24,830 | **+3.40%** |

Wall improvement is consistent with iter 87's multi-seed +3.46% at 200 steps.  **Wall side of the picture is robust.**

## The trajectory pattern — iter 41 echo

iter 41's "aggressive spectral truncation impairs flagship capacity" pattern:
- Early steps (0-100): the reduced-capacity model has fewer parameters fighting → can converge FASTER initially
- Mid-training (100-300): baseline catches up and pulls ahead as it leverages full capacity
- Late training (300+): gap stabilizes at the architecture-determined capacity differential

iter 88's 500-step trajectory matches this pattern:
- Step 100: triple-stack AHEAD by 0.126 nat (early warm-up advantage)
- Step 200: parity (Δ +0.043)
- Step 300: peak deficit +0.069 nat
- Step 500: persistent gap +0.057 nat

If this pattern continues to 30k steps, the final gap could be 0.05-0.15 nat — depending on whether it stabilizes or continues to grow.

## Comparison to iter 87's 200-step multi-seed result

| horizon | sample | NLL Δ vs baseline | Wall Δ |
|---      |---     |---:               |---:    |
| 200 steps, single seed (1337, this iter) | n=1 | +0.061 (baseline 7.6154 mean, triple 6.1184... wait this is 500 step value) | — |
| 200 steps, multi-seed (iter 87) | n=5 | **−0.019** (within ±0.02 strict) | +3.46% |
| 500 steps, single seed (1337, iter 88) | n=1 | **+0.057** (outside ±0.05) | +3.40% |

Wait — comparing 200-step Δ at single seed=1337:
- baseline (iter 75 rerun-1 at 200 steps): NLL 7.6002
- triple-stack (iter 86 seed=1337 at 200 steps): NLL 7.6766
- Δ @ 200: +0.0764

But iter 87's multi-seed MEAN at n=5 was −0.019 vs baseline mean.  The single-seed=1337 comparison is +0.076 — UNFAVORABLE.  iter 87's multi-seed averaging hid the seed=1337 deficit.

At 500 steps single seed=1337: Δ = +0.057 (improving from +0.076 at 200 step — triple-stack catching up a bit).

This means **at seed=1337 specifically, the triple-stack is consistently +0.05-0.08 nat behind baseline at the 200-500 step horizon**.  The iter 87 multi-seed mean−0.019 was driven by seeds 1338/1339 outperforming, not seed=1337.

## Why this fails

The brief's 5k-30k step target horizon is 10-60× longer than iter 88's 500 steps.  The persistent +0.05-0.08 nat gap at seed=1337 over 200-500 steps suggests:
- The triple-stack DOES train but converges to a slightly higher final NLL
- The gap is around 0.05-0.10 nat at the 30k horizon, likely above the strict ±0.02 bar but possibly within the iter 60 relaxed ±0.05 multi-seed bound

For a confident retrain arc commitment, would want:
- Multi-seed (n≥3) at 500 steps to confirm the seed=1337 trajectory isn't an outlier
- And/or 1k+ step trajectory at multi-seed

iter 88 is single-seed at 500 steps — not enough data to commit to a multi-iter retrain arc with confidence.

## Categorization

**CAUTION** — wall improvement validated, NLL parity AT LONGER HORIZON uncertain.  The 200-step multi-seed PASS was driven by averaging across seeds where seeds 1338/1339 outperformed but seed=1337 underperformed.  At 500 steps single seed, the picture is UNFAVORABLE.

## Default policy

No code change.  Opt-in flags preserved.  Recommend NOT committing to full 30k retrain without further multi-seed longer-horizon validation.

## Sequence status (34 iters)

| iter | result | notes |
|---: |---     |---    |
| 85 | conv-w=4 individual | +2.07% wall, NLL parity at 200 steps |
| 86 | triple-stack n=3 | +3.43% wall, NLL parity at 200 steps |
| 87 | triple-stack n=5 | +3.46% wall, mean NLL parity at 200 steps |
| **88** | **triple-stack 500-step single seed** | **+3.40% wall, NLL +0.057 worse at 500 steps** |

## Recommended next action

Per the iter 87 → iter 88 pattern shift, the multi-iter retrain arc has DOUBLED in risk:
- iter 87 suggested high-confidence retrain commitment
- iter 88 suggests longer-horizon convergence is borderline

Options:
A. **Multi-seed at 500 steps** (3 seeds × 6 min = 18 min): confirm whether seed=1337 deficit is representative
B. **1k-step single seed**: extend further to see if gap stabilizes or grows
C. **Pause autonomous loop**: iter 87 was the high-water mark; iter 88's data suggests caution before retrain commitment.  User direction needed on whether to invest training resources.

## Files

- This document.
- `research/runs/2026-05-20-iter88-bench/baseline_500step_seed1337.log`
- `research/runs/2026-05-20-iter88-bench/triple_500step_seed1337.log`
- No code change.
