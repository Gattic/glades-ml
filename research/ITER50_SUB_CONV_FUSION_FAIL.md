# Iter 50 — SCFA sub-into-conv fusion — Gate-0 FAIL (below-bar + NLL drift)

**Date**: 2026-05-16
**Iter**: 50 (fourth iter under "stacking-wins" brief; iter47 PASS, iter48 FAIL too-small-target, iter49 PASS, iter50 FAIL)
**Branch**: vesta5 (glades-ml) + glades-trainer/main
**Verdict**: FAIL — fused kernel measured +1.3% tok/s (within bench variance), and NLL drifted +0.04 at step 100 / -0.02 at step 200 (chaotic divergence from a real fp32 round-off difference that the iter47 dK fix doesn't have).  Reverted; iter51 will pivot.

---

## TL;DR

Targeted `chiron_scfa_sub` (4.3% of total GPU time) — second-largest non-cuBLAS, non-iter47/49-already-fixed kernel — by fusing `q_perp = q − q_par` directly into the depthwise causal conv that read `q_perp` immediately after.  New kernel `scfa_depthwise_causal_conv_fwd_minus(a, b, K, ...) = D(a − b)` and matching `scfa_depthwise_causal_conv_bwd_minus(a, b, K, dy, ...)`.

Only the FWD path could be fused — the BWD-RECOMPUTE writes q_perp into a scratch that's then read by the dK kernel at line 6283, and the surrounding code clobbers `q_par` (line 6281: `device_memcpy_d2d(scfa_qpar, scfa_ypar)`) to reuse it as the `dy` temp — so `q_par` isn't intact when dK runs.  To fuse the bwd path you'd need an extra T·m scratch buffer (64 MB) just to keep `q_par` alive across the conv_bwd, which is a separate change.

Result with FWD-only fusion (12 calls eliminated per step, vs 24 total sub instances/step):

| metric             | iter49 baseline | iter50 fwd-only | delta             |
|---                 |---:             |---:             |---:               |
| tok/s steady       | 43,209          | 43,775          | +1.31% (in noise) |
| val NLL @ step 100 | 8.6579          | 8.6941          | +0.036 (out of bound) |
| val NLL @ step 200 | 8.2391          | 8.2159          | −0.023 (better)   |

Tok/s gain (+1.31%) is within bench noise (typical 1-2% run-to-run variance at this scale).  NLL drift at step 100 is real (+0.036 nat outside the ±0.02 nat parity bound), recovers by step 200 (chaotic re-convergence).

Both axes fail the brief.

---

## Why the fp32 drift exists (and isn't trivially fixable)

The fused kernel computes `y[t, c] = Σ_i K[c, i] · (a[t-i, c] − b[t-i, c])` in one pass; the legacy chain computes `c[i] = a[i] − b[i]` in a kernel that writes to memory, then `y[t, c] = Σ_i K[c, i] · c[t-i, c]` in a second kernel that reads c from memory.

Mathematically equivalent.  fp32-equivalent IF:
- the fp32 subtraction result `c` is the same as the register-level `diff` (it is — single fp32 subtraction)
- the FMA `K · c + acc` is emitted identically by the compiler in both kernels (here they diverge)

Tested three variants:
1. `acc += K[i] * (av - bv)` — compiler may distribute as `K*a − K*b + acc` (two FMAs).  Drifts +0.076 nat at step 200.
2. `const float diff = av - bv; acc = __fmaf_rn(kv, diff, acc);` — forces explicit RNE FMA.  Drifts +0.031 nat at step 200.
3. `const float diff = av - bv; acc += kv * diff;` — explicit subtraction, compiler-emitted FMA.  Drifts −0.023 at step 200 (chaotic) but +0.036 at step 100 (outside bound).

None match the legacy chain bit-for-bit.  Suspect cause: the legacy `sub` kernel writes `c` to global memory (rounded to fp32 register, stored), then the conv reads it back; on Ada the load comes with a fresh FLT.32 load instruction.  In the fused kernel, the subtraction stays in a register that may have different denormal/intermediate handling than the GLD→FLT.32 path.  Forcing `__fmaf_rn` reduces but doesn't eliminate the drift.

The drift is small (single-digit milli-nats per step) but compounds chaotically.  The training trajectory diverges from the iter49 baseline outside the ±0.02 nat parity bound at step 100.  By step 200 the chaotic dynamics converge again, but that's coincidence — at scale this would not consistently stay within parity.

---

## Why tok/s gain is small

The FWD path has 12 sub calls per step (1 per layer).  Eliminating these AND the q_perp scratch round-trip should save ~2-3% based on bandwidth math:
- 12 × (q_perp materialize + L2-evict + re-read) ≈ 64 MB × 12 = 768 MB / step traffic eliminated
- At 736 GB/s, ~1.0 ms / step = 1.0% wall-clock at the iter49 base of ~100 ms/step.

Plus eliminating 12 sub kernel launches (tiny overhead).

So the realistic ceiling is ~1-2%, NOT the 4.3% kernel slice — because:
1. BWD-recompute's 12 sub calls remain (can't fuse without restructuring the qpar/dy buffer reuse).
2. The fused conv reads two T·m inputs (a, b) instead of one (c) — adds ~64 MB read traffic per call, partially offsetting the saved scratch round-trip.

The measured +1.31% (within bench noise) is consistent with this ceiling.  This is essentially a wash perf-wise even before counting the NLL drift.

---

## What was actually changed (reverted before commit; doc only ships)

- `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` — added `scfa_depthwise_causal_conv_fwd_minus_kernel` + wrapper, and `scfa_dwconv_dK_minus_kernel_par` + `scfa_depthwise_causal_conv_bwd_minus` wrapper.
- `Backend/Machine Learning/Networks/cuda/gpu_kernels.h` (+ trainer mirror) — declarations.
- `glades-trainer/trainer/chiron_main.cpp` — routed FWD SCFA path (both sequential and parallel-branches) through the fused kernel under `--scfa-fuse-streams`.

All three files reverted; the FAIL doc is the only ship.

---

## Why iter50 picked this target despite the iter48 lesson

The iter48 doc said "pick ≥7%-share targets".  sub at 4.3% alone is too small — but I bet on the FUSION lifting it because (a) sub + conv chain is 4.3 + 3.4 = 7.7% combined, and (b) eliminating the q_perp scratch saves an additional T·m memory round-trip per call.  The bet didn't pay because:

1. The BWD-RECOMPUTE blocks half of the fusion (q_perp needed for dK; q_par clobbered as dy temp).
2. Even the FWD half has compounding fp32 round-off differences that can't be patched without IDENTICAL instruction sequences.

The brief's "one kernel" rule fits the fwd-only attempt, but the realistic ceiling (≤2%) was below the bar from the start.

---

## Iter sequence

| iter | target | result |
|---: |---     |---     |
| 47  | SCFA dwconv-dK 2D-tiled par-reduction | PASS +5.20% |
| 48  | LN-bwd dgamma/dbeta par-reduction      | FAIL too-small-target (+2.4%) |
| 49  | Fused Adam-int8 BF16w/g + bf16 grad-norm | PASS +6.98% |
| 50  | SCFA sub-into-conv fusion (fwd only)   | FAIL bench-noise + NLL drift |

Cumulative shipped: 1.052 × 1.070 = **1.125** (+12.5%).  Target by iter 5 is 1.5×; we have 4 iters at +12.5%.  Need +33% more in the remaining iters to hit the 5-iter milestone.

---

## Where next (iter 51 candidates)

Top remaining ≥7%-share kernels in iter49 profile:
- cuBLAS SCFA inner GEMMs ~20% combined — cuBLAS hard to beat, but cuBLASLt heuristic exploration or tile-size override might give 2-5%.
- cuBLAS readout GEMMs ~15% combined — same caveats.

≥7%-share *if combined*:
- chiron_scfa_axpy2 (5.6%) + chiron_scfa_sub (4.3%) = 9.9% — both memory-bound at ~80-98% of peak.  Hard.
- bf16 cast residue (~3.5%) + k_bf16_accum_axpy (2.8%) = 6.3% — could be eliminated by writing fp32 grads directly as bf16 from the attention backward kernels (avoid the FP32 → BF16 accum altogether).  Risk: requires changes deep inside the attention backward.

Most promising remaining lever: **paradigm-level change** rather than kernel optimization.  E.g. enabling `--scfa-checkpoint-inner` (iter 7 PARTIAL, was rejected for +1.31 GB VRAM at +4.10% speed — but VRAM headroom is now larger after iter49's freed cast scratches).  Or revisit `--scfa-parallel-branches` (iter 8 NEGATIVE on Ada, but iter49 freed up SMs from the GPU-time leaderboard so the picture may have changed).

Iter 51: re-profile after the iter49 stack, check whether the post-iter49 VRAM and SM-occupancy picture makes any of the iter6-9 PARTIAL/NEGATIVE flags newly viable.
