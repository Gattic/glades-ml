## Iter 86 — Triple-stack iter 70 + iter 73 + iter 85 (conv-w=4) — FIRST +3%-bar-clearing engineering result post-iter69

**Date**: 2026-05-20
**Iter**: 86 (thirty-second iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **CLEARS +3% iter 60-precedent retro-ship bar at multi-seed AND maintains NLL parity (±0.05 multi-seed bound)**.  Wall **+3.43%**, NLL mean drift **−0.031 nat** (BETTER direction).  BUT `--scfa-conv-w 4` is a MATH change (filter size) — current CHIRON 1B flagship trained at w=8 cannot use the new filter without retraining.  Default-on ship is a multi-iter arc requiring fresh production training.

---

## Bench (triple-stack, n=3, fresh-init, 200 steps)

Stack: `--int8-adam --bf16-grads --bf16-weights --bf16-attn --no-fuse-attn --fuse-attn-reln --scfa --scfa-conv-w 4 --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt --bf16-logits --bf16-logits-storage --scfa-checkpoint-inner --scfa-checkpoint-inner-bf16 --iter70-fused-axpy2-dual-p --iter73-dwconv-fwd-tiled`.

Note: `--iter73-dwconv-fwd-tiled` is **no-op at w=4** because the tiled kernel is templated `W_FILTER=9` and falls back to the row-major kernel for other w values.

| seed | NLL @ step 200 | wall (s) |
|---:  |---:            |---:      |
| 1337 | 7.6766 | 131.8 |
| 1338 | 7.5160 | 131.9 |
| 1339 | 7.5603 | 131.9 |
| **mean (n=3)** | **7.5843** | **131.87** |
| **std** | **0.0828** | 0.06 |

## Comparison to all post-iter69 mechanisms

| config | n | NLL mean ± std | Δ wall | Δ NLL mean | std vs baseline | clears +3%? | parity multi-seed? |
|---     |--:|---:            |---:    |---:        |---             |---           |---                 |
| baseline | 4 | 7.6154 ± 0.016 | — | — | 1× | — | — |
| iter 73 alone | 3 | 7.5802 ± 0.013 | +0.42% | −0.035 | 0.8× | NO | YES |
| iter 70 alone | 5 | 7.6437 ± 0.088 | +1.27% | +0.028 | 5.5× | NO | YES |
| iter 85 (w=4) alone | 3 | 7.6201 ± 0.092 | +2.07% | +0.005 | 5.7× | NO | YES (within strict!) |
| iter 70+73 combined | 5 | 7.6794 ± 0.038 | +1.76% | +0.064 | 2.4× | NO | NO (just outside) |
| **triple-stack (iter 86)** | **3** | **7.5843 ± 0.083** | **+3.43%** | **−0.031** | **5.2×** | **YES** | **YES** |

## Findings

1. **Wall stacking is near-additive**: iter 70 (+1.27%) + iter 85 (+2.07%) + iter 73 (no-op at w=4) = predicted +3.34%; observed +3.43% (slightly super-additive).  iter 73 no-op at w=4 contributes nothing.

2. **NLL drift is in BETTER direction (mean −0.031)**: iter 73 alone contributes −0.035 (BETTER), iter 70 alone +0.028, iter 85 alone +0.005.  Sum: −0.035 + 0.028 + 0.005 = −0.002.  Observed −0.031 is more negative than additive prediction — interactions slightly favor BETTER direction.

3. **Variance is seed-dependent**: std 5.2× baseline.  Seed=1337 is the outlier at 7.6766; seeds 1338/1339 cluster at 7.5160-7.5603.  Same iter 70 pattern of seed-dependent training stability.

4. **CLEARS +3% iter 60-precedent retro-ship bar**: First post-iter69 mechanism to do so at multi-seed.

5. **Within ±0.05 multi-seed parity bound**: |−0.031| < 0.05.  Outside ±0.02 strict bound but within the practical multi-seed methodology iter 75 established.

## Caveat: conv-w=4 is a math change

`--scfa-conv-w 4` changes the depthwise causal conv filter from 9 taps to 5 taps.  The model architecture differs from production CHIRON 1B (which was trained at w=8).  The D matrix dimensions differ:
- w=8: D ∈ R^{m × 9}
- w=4: D ∈ R^{m × 5}

Direct loading of the trained checkpoint at w=4 would fail (shape mismatch).  For default-on ship:
- A new CHIRON 1B (w=4) must be retrained from scratch
- Compare final val NLL at 30k steps to current flagship's 3.77
- If within ±0.05 nat (iter 60-precedent relaxed multi-seed bound), ship as new flagship

## Why this isn't ship-eligible as-is

The brief's per-iter contract: "PASS (≥+5% tok/s win + NLL within ±0.02 nat over 5k-30k steps)".  iter 86 achieves:
- Wall +3.43% (below +5% strict but above +3% iter 60-precedent)
- NLL drift mean −0.031 at 200 steps (extrapolation to 30k steps unknown)

The 200-step fresh-init bench shows the architecture is TRAINABLE.  But final convergence at 30k steps could:
- Continue tracking baseline → ship-eligible at full validation
- Plateau higher than baseline (iter 41 pattern) → idea-fail at long horizon

Without 30k-step training validation, default-on ship is risky.

## Strategic recommendation

iter 86's result is the strongest engineering evidence since iter 69.  Recommended next action:

**Multi-iter arc A: Train CHIRON 1B (w=4) from scratch for 30k steps**.  Validate final NLL ≤ 3.77 + 0.05 = 3.82.  If parity holds, ship as new flagship with +3.43% wall over current.  Estimated training time: 6-10 hours on RTX 4080 SUPER.  This is a multi-iter arc but the FIRST clear positive case.

**Alternative**: bench at longer fresh-init horizons (1k, 3k, 5k steps) to see if NLL trajectory continues tracking or diverges from baseline.  Lower commitment than full retrain.

## Default policy (current iter)

`--scfa-conv-w` stays default 8.  `--iter70-fused-axpy2-dual-p` and `--iter73-dwconv-fwd-tiled` remain default OFF.  Triple-stack is opt-in via combined flags.  No code change.

## Sequence status (32 iters)

| iter | result | wall | parity multi-seed |
|---: |---     |---:  |---                |
| 69  | last PASS | +6.83% | — |
| 70-84 | 15× non-PASS | various | various |
| 85  | conv-w=4 mean parity within strict | +2.07% | YES (within strict) |
| **86** | **triple-stack +3.43% wall, mean -0.031 NLL** | **+3.43%** | **YES (within ±0.05)** |

17 consecutive non-PASS iters by strict bar (70-86), but iter 86 is the FIRST since iter 69 to clear +3% iter 60-precedent retro-ship bar at multi-seed parity.  Strategic situation has changed.

## Files

- This document.
- `research/runs/2026-05-20-iter86-bench/triple_seed{1337,1338,1339}.log`.
- No code change.
