# Gradient-accumulation / effective-batch pilot (2026-06-12)

**Question**: at fixed token budget, does a 4× larger effective batch
(accum=4 → 65,536 tokens/step, gradients averaged via lossNorm =
1/(accumN·T)) with appropriately scaled LR beat the production regime of
one T=16384 sequence per optimizer step?

**Design**: token-matched at 81.92M tokens (5k-equivalent), seed 1337,
full production recipe (zloss+qk-norm+SIRA+clamps; cast-elim default-on),
warmups token-scaled (LR warmup 500→125 opt steps, SIRA 1000→250), cosine
horizon = opt-step count per arm. Artifacts: glades-trainer `logs/accum_pilot_20260612_120924/`.

## Results at 81.92M tokens (single seed)

| arm | accum | LR | final val NLL | Δ vs C0 | warm tok/s | skips |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 1 | 7.5e-5 | 3.9559 | — | 28,272 | 0 |
| A4a | 4 | 7.5e-5 (×1) | 4.3641 | +0.408 FAIL | 29,398 | 0 |
| A4b | 4 | 1.5e-4 (×√4) | 3.9415 | −0.014 | 29,396 | 0 |
| **A4c** | **4** | **3e-4 (×4)** | **3.9230** | **−0.033** | **29,395** | **0** |

Token-matched trajectories (val NLL at 8.2/16.4/24.6/32.8/41/49.2/57.3/65.5/73.7/81.9M):
- C0:  7.644/6.123/5.230/4.891/4.527/4.348/4.588/4.162/4.136/3.956
- A4a: 8.754/7.444/6.435/5.868/5.504/5.074/5.133/4.730/4.517/4.364
- A4b: 8.277/6.572/5.421/4.938/4.611/4.355/4.495/4.142/4.028/3.942
- A4c: 7.704/5.678/4.720/4.418/4.210/4.103/4.327/4.019/3.974/3.923

## Findings

1. **Linear LR scaling wins, monotonically**: unscaled FAILs badly
   (step-starved); ×√4 reaches parity-plus; ×4 is best at every
   checkpoint from 16.4M onward. The trend has NOT turned over at ×4 —
   the critical batch size is at or above 65k tokens at this scale.
2. **−0.033 nat at equal tokens** is above the single-seed 5k noise scale
   (~±0.01–0.02) — a real signal, pending multi-seed.
3. **+4.0% wall for free**: accum=4 amortizes per-optimizer-step host
   work (grad-norm download, Adam dispatch, clamp counter reads) across
   4× tokens — 29.4k vs 28.3k tok/s, uniform across the accum arms.
4. **Stability improved through the data-window bump**: at the 57.3M bump
   the accum arms ride 0.09–0.26 nat better than C0 (batch averaging over
   the spiky window); zero grad-skips and zero clamp firings in all arms.
5. The bump itself reproduces at the same TOKEN position in every arm —
   further confirmation it is data-order-driven.

## Verdict and next steps

PILOT POSITIVE for accum=4 + linear LR scaling (3e-4): better NLL per
token AND better wall. Before any recipe consideration:
1. **n=3 multi-seed confirmation** (seeds 1337/1338/1339, C0 vs A4c,
   5k-equivalent, ~5h GPU) per standing methodology.
2. Optional probe: accum=8 + lr 6e-4 arm — the monotone trend says the
   critical batch may be higher still.
3. Adoption is a MATH change → ships only with a fresh blessed 30k
   checkpoint. Natural vehicle: the pending data-scale run, which could
   adopt accum=4+scaled-LR directly (with the caveat that pilot evidence
   is at 82M tokens; the 30k/long-horizon LR schedule interaction needs
   its own gate).
