## Iter 84 — Combined iter 70+73 extended to n=5 multi-seed

**Date**: 2026-05-20
**Iter**: 84 (thirtieth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **FAIL** — confirms iter 76's combined retro-ship verdict at extended n=5.  Wall +1.76% (below +3% iter 60 bar), NLL drift +0.064 nat (outside ±0.05 multi-seed bound).  Std 0.038 is better-than-iter-76 (which was 0.054 at n=3) — iter 73's tightening partially offsets iter 70's instability.

---

## Bench (combined iter 70 + iter 73 at n=5)

| seed | NLL @ step 200 | wall (s) | source |
|---:  |---:            |---:      |---     |
| 1337 | 7.6404 | 134.0 | iter 76 |
| 1338 | 7.7431 | 134.1 | iter 76 (1338 outlier) |
| 1339 | 7.6644 | 134.2 | iter 76 |
| 1340 | 7.7107 | 133.9 | iter 84 |
| 1341 | 7.6785 | 134.0 | iter 84 |
| **mean (n=5)** | **7.6794** | **134.04** | |
| **std** | **0.0381** | 0.11 | |

## Full multi-seed picture across mechanisms

| config | n | mean ± std | wall (s) | Δ wall | Δ NLL | std vs baseline |
|---     |--:|---:        |---:      |---:    |---:   |---             |
| baseline (default) | 4 | 7.6154 ± 0.016 | 136.40 | — | — | 1× |
| iter 73 alone | 3 | 7.5802 ± 0.013 | 135.83 | +0.42% | −0.035 | 0.8× (tighter) |
| iter 70 alone | 5 | 7.6437 ± 0.088 | 134.67 | +1.27% | +0.028 | 5.5× |
| **iter 70+73 combined** | **5** | **7.6794 ± 0.038** | **134.04** | **+1.76%** | **+0.064** | **2.4×** |

## Findings

1. **Combined std (0.038) is BETTER than iter 70 alone std (0.088)**.  iter 73's tightening (std 0.013, tighter than baseline) partially offsets iter 70's instability when stacked.  Combined std is between iter 73 alone and iter 70 alone, not the sum.

2. **Wall stacking is near-additive**: iter 70 (+1.27%) + iter 73 (+0.42%) ≈ +1.69%; observed combined +1.76% is slightly super-additive (the two mechanisms attack disjoint kernels — axpy2/cast vs dwconv).

3. **NLL drift comes from iter 70**: combined +0.064 ≈ iter 70 +0.028 + iter 73 −0.035 + interaction +0.071.  iter 70 contributes positive drift; iter 73 contributes negative drift; interaction term adds positive.

4. **Combined remains FAIL by both bars**:
   - Wall +1.76% < +3% iter 60-precedent bar
   - NLL drift +0.064 > ±0.05 multi-seed bound (still outside)

5. **At n=5 the picture is stable**: iter 76's n=3 conclusion (FAIL) is confirmed.  Adding 2 more seeds didn't change the verdict.

## Default policy

Both `--iter70-fused-axpy2-dual-p` and `--iter73-dwconv-fwd-tiled` remain default OFF.  Opt-in flags preserved.  No code change.

## Sequence status (30 iters)

| iter | result | notes |
|---: |---     |---    |
| 70-83 | 14× non-PASS | engineering ceiling on single-axis |
| 84  | combined n=5 confirms iter 76 FAIL | wall stacking ~additive; NLL drift from iter 70 |

15 consecutive non-PASS iters (70-84).  Engineering ceiling investigation is comprehensively closed.

## Files

- This document.
- `research/runs/2026-05-20-iter84-bench/combined_seed{1340,1341}.log`.
- No code change.
