## Iter 75 — Methodology calibration: single-seed baseline is unreliable at L=24 T=16384

**Date**: 2026-05-19
**Iter**: 75 (twenty-first iter under stacking-wins brief; META iter, no code change)
**Branch**: vesta5 (glades-ml)
**Verdict**: **METHODOLOGY CORRECTION** — the iter 69 baseline used for iter 70-74 NLL parity comparisons was a +0.16 nat 2σ outlier.  Today's reruns at same seed=1337 cluster at NLL 7.601 (rerun-1) and 7.602 (rerun-2), matching the iter 70-74 follow-up cluster (mean 7.629).  All iter 70-74 NLL "FAIL" verdicts must be reconsidered against the corrected baseline.  Combined iter 70 + iter 73 = +1.80% wall, +0.039 nat NLL drift vs corrected baseline (within ±0.05 multi-seed bound, just outside ±0.02 strict bound).

---

## Background — the iter 70-74 pattern

Five consecutive non-PASS iters (70-74) all showed NLL "drift" −0.10 to −0.17 nat BETTER than the iter 69 baseline saved on 2026-05-19 22:46 (NLL @ step 200 = 7.7583).  The follow-up cluster:

| iter | NLL @ step 200 | Δ vs original baseline |
|---:  |---:            |---:                    |
| 70 v1 | 7.6392 | −0.119 |
| 70 v2 | 7.6405 | −0.118 |
| 71    | 7.5900 | −0.168 |
| 72    | 7.6493 | −0.109 |
| 73    | 7.6312 | −0.127 |
| 74    | 7.6256 | −0.133 |
| **mean** | **7.6293** | **−0.129** |

Six independent code paths all drifting in the same (BETTER) direction by 0.10-0.17 nat looked suspicious.  Hypothesis: the original baseline was an outlier.

## Calibration runs (today, 2026-05-19 23:53+)

Same trainer binary, same flags, same seed=1337, same hardware:

| run | NLL @ step 200 | wall (s) | tok/s mean |
|---  |---:            |---:      |---:        |
| baseline rerun 1 | **7.6002** | 136.3 | ~24,300 |
| baseline rerun 2 | **7.6020** | 136.4 | ~24,310 |

**Today's mean baseline**: **7.6011** (variance 0.0018 across 2 reruns).

The original yesterday-baseline at 7.7583 sits +0.158 nat above today's mean — a 2σ outlier at the iter 67-documented single-run-variance estimate of ±0.06 nat.

## Re-classification of iter 70-74

Using today's calibrated baseline (7.6011):

| iter | wall Δ vs baseline | NLL @ step 200 | corrected Δ NLL | ±0.02 strict? | ±0.05 multi-seed? |
|---:  |---:                |---:            |---:             |---            |---                |
| 70 v1 | +1.38% tok/s, −1.47% wall | 7.6392 | +0.038 | FAIL | within |
| 70 v2 | +1.30% / −1.39% | 7.6405 | +0.039 | FAIL | within |
| 71    | −5.45% / +5.86% (slower) | 7.5900 | **−0.011** | **PASS** | within |
| 72    | −0.25% / +0.22% (noise) | 7.6493 | +0.048 | FAIL | just within |
| 73    | +0.56% / −0.59% | 7.6312 | +0.030 | FAIL | within |
| 74    | +0.23% / −0.29% (noise) | 7.6256 | +0.025 | FAIL (just) | within |

**Critical correction**: iter 71 (which I classified as NEGATIVE/idea-fail) is actually **within strict ±0.02 NLL parity vs the correct baseline**.  It's still NEGATIVE on wall (−5.4% tok/s), so the overall FAIL verdict stands.  But the NLL drift was illusory.

For iter 70/72/73/74: corrected drifts are +0.025 to +0.048 nat — just outside ±0.02 strict but well within ±0.05 multi-seed.  These iters likely produce real-but-microscale FP32 differences from FMA-emit / scheduling but the magnitude is at the noise floor for single-seed L=24 T=16384.

## Combined iter 70 + iter 73 retro-ship test

A combined retro-ship attempt at seed=1337 (today, fresh state):

```
build/glades_chiron_train ... --iter70-fused-axpy2-dual-p --iter73-dwconv-fwd-tiled --seed 1337
```

| metric | calibrated baseline (today, n=2 mean) | iter 70+73 combined | Δ |
|---|---:|---:|---:|
| tok/s mean | 24,310 | **24,748** | **+1.80%** |
| wall (200 steps) | 136.4 s | **134.0 s** | **−1.76%** |
| val NLL @ step 200 | 7.6011 | 7.6404 | +0.039 nat |

**Combined wall: +1.80%** (below +5% strict bar but a real cumulative gain).
**Combined NLL drift: +0.039 nat** (just outside ±0.02 strict; within ±0.05 multi-seed).

The stacking is sub-multiplicative: predicted (iter70 +1.38%) × (iter73 +0.56%) = +1.95%; actual +1.80%.  Small interference loss likely from shared SR-cast counter sequence interleaving.

## Strategic implications

1. **Single-seed parity comparison is methodologically unreliable** at L=24 T=16384 production scale.  Run-to-run variance can be as high as ±0.16 nat (today's vs yesterday's same-seed baseline difference) even with same code, same flags, same seed.  Whatever causes this (cuBLAS workspace allocation patterns? GPU thermal? cuDNN heuristic warmup? cudaMalloc fragmentation across runs?), it produces non-trivial variance.

2. **The brief's strict ±0.02 nat single-seed parity bound is essentially infeasible** at L=24 T=16384.  Any code change benched once will appear to drift outside this bound on most attempts, regardless of mechanism correctness.  Multi-seed (3+ seeds, ±0.05 mean drift bound) is the methodologically sound approach.

3. **Iters 70-74 should be reconsidered as below-bar silent-accrual candidates**, NOT as FMA-emit failures.  The FMA-emit hypothesis (iter 50/70/73 doc) may still apply, but its observable magnitude is smaller than I previously claimed (corrected ~+0.03 to +0.05 nat at single seed, not −0.12 to −0.17).

4. **Iter 71 (−scfa-parallel-branches) NLL parity is actually clean** at corrected baseline (Δ = −0.011 nat).  The Ada cuBLAS TC saturation argument for wall slowdown remains correct; that's a separate idea-fail.

5. **Combined retro-ship of iter 70 + iter 73 = +1.80% wall** at marginal parity.  Below the +5% strict bar AND below the iter 60-precedent +3% bar.  Even with relaxed bar, this is too small to retro-ship.

## Recommendation

**No code change in iter 75**.  The methodology correction is the deliverable.  Iters 70/73/74 default policies remain unchanged (opt-in flags, default OFF).  Future iter benches MUST run multi-seed (≥3 seeds) baseline + experimental for proper parity validation.

For iter 76+:
- Continue per-iter contract but with multi-seed mandate
- Accept that single-iter ≥+5% wins are not on the table at the current engineering ceiling
- Multi-iter arcs (Hadamard basis, FlashAttention-fused inner attention) remain the realistic path to magnitude-class wins

## Files

- This document (iter 75 methodology correction).
- `glades-ml/research/runs/2026-05-19-iter75-bench/baseline_rerun{1,2}.log` and `combined_70_73.log`.
- No code changes.
