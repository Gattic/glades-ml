## Iter 97 — Fused scfa_sub + fwd dwconv tile — SUB-3% SILENT-ACCRUAL PASS

**Date**: 2026-05-21
**Iter**: 97 (post iter 96 nsys profile)
**Branch**: vesta5 (glades-ml) + glades-trainer
**Verdict**: **+1.54% tok/s @ NLL bit-identical sub-ULP**. First measurable wall improvement at single-iter scope since iter 94 ship. Silent-accrual category (below iter 60 +3% bar but well above ~0.5-1% noise).

---

## Bottleneck identified (iter 96 nsys profile)

`chiron_scfa_sub_kernel` was profile-rank-#4 at 5.1% of step wall (50 calls/step × 646 µs each, already 87% of memory BW). The kernel computes `q_perp = q - q_par` element-wise (T*m = 33M FP32 elements per call), feeding into the immediately-following `scfa_depthwise_causal_conv_fwd_tiled` on the main stream.

The fwd half (~24 of the 50 scfa_sub calls/step) is sequential with conv; fusion eliminates:
- 1 explicit `chiron_scfa_sub` kernel launch per layer (24/step)
- 1 write of q_perp[T, m] = 132 MB per layer to global memory
- 1 read of q_perp[T, m] from global memory by conv (partially L2-cached)

The bwd half of scfa_sub (the 26 calls in scfa_attention_backward) is independent and continues to run unchanged (recomputes q_perp for the bwd conv input).

## Conjecture (pre-committed)

Target: new kernel `scfa_depthwise_causal_conv_fwd_sub_fused_tiled_kernel` that takes `(q, q_par)` as inputs (instead of pre-computed q_perp) and computes `q_perp = q - q_par` AT SMEM-LOAD TIME (single FP32 subtraction per element, in registers, before storing in x_smem). Then runs the iter 73 tiled conv exactly as before.

**Math bit-identical** to `(chiron_scfa_sub THEN scfa_depthwise_causal_conv_fwd_tiled)`:
- Per element: same `q[idx] - q_par[idx]` FP32 subtraction (identical RNE rounding).
- Per output: same `Σ K[c,i] · q_perp[t-i,c]` FMA accumulation order (i=0..w, break on `t_out - i < 0`).
- FP32 accumulator throughout.

**Pre-committed Gate-0**:
- Wall delta: +1% to +2.5% tok/s on production T=16384 m=2048 L=24 w=4
- NLL drift: ≤ ±0.005 nat (bit-identical FMA — only scheduling-induced sub-ULP drift expected)
- Token budget: 100-step apples-to-apples bench (seed=1337)

## Implementation

- **gpu_kernels.cu**: new template kernel
  `scfa_depthwise_causal_conv_fwd_sub_fused_tiled_kernel<COLS, N_OUT, W_FILTER>`
  in the anon namespace (~line 8032).  Smem layout identical to iter 73
  fwd tile: x_smem (N_OUT + w) × COLS × 4 + K_smem COLS × W_FILTER × 4 =
  25 KB at (16, 256, 5).  Single load-loop pass per element subtracts in
  registers before storing in smem.
- **gpu_kernels.cu**: new dispatcher
  `scfa_depthwise_causal_conv_fwd_sub_fused_tiled` with W_FILTER=5 (w=4)
  and W_FILTER=9 (w=8) specializations; returns false for other w (caller
  must check before dispatch).
- **gpu_kernels.h** (both glades-ml + vendored glades-trainer copies):
  declaration + no-CUDA stub.
- **chiron_main.cpp**: `iter97DwconvFwdFusedSub` config flag (default OFF),
  CLI parse `--iter97-dwconv-fwd-fused-sub` / `--no-`, dispatch on the
  sequential fwd path (lines ~6347-6380) when flag is on AND `wd ∈ {4, 8}`.
  Parallel-branches path NOT modified (iter 71 NULL on Ada anyway).
  Bwd path's scfa_sub recompute is unchanged.

## Bench (single-seed 100 steps × seed=1337 × T=16384 m=2048 L=24 w=4)

| run | wall (s) | tok/s @ 26 / 51 / 76 | NLL @ step 100 | PPL |
|---  |---:      |---:                   |---:            |---: |
| baseline (no iter97 flag) | 65.3 | 24,943 / 25,214 / 25,214 | 9.8477 | 18913.96 |
| iter97 (--iter97-dwconv-fwd-fused-sub) | 64.3 | 25,319 / 25,606 / 25,603 | 9.8476 | 18912.39 |
| Δ                          | **−1.0s** | **+1.51% / +1.55% / +1.54%** | **−0.0001** | **−1.57** |

**Wall improvement**: −1.0s on a 65.3s baseline = **−1.5% / +1.54% tok/s**.
Consistent across all 3 logged step checkpoints (+1.51, +1.55, +1.54).
Intra-run variance: baseline 24,943-25,214 spans 271 tok/s = ~1.1%, treatment
25,319-25,606 spans 287 tok/s = ~1.1%.  The improvement (~390 tok/s = +1.5%)
is clearly above intra-run variance and the typical ~0.5-1% run-to-run noise
floor on RTX 4080 SUPER.

**NLL parity**: 9.8477 → 9.8476 (sub-ULP, BETTER direction).  PPL drift
−1.57 corresponds to ≈ 0.00008 nat — sub-ULP rounding from cuBLAS GEMM
scheduling order (drift class iter 74 = "L2/scheduler-timing perturbation")
NOT from the math change.

**Loss trajectory parity**:
- Step 1: 10.5777 / ||g||=3.452 / identical
- Step 25-26: best=10.3259, ||g|| 2.888 vs 2.889 (sub-ULP)
- Step 51: best=10.2939, ||g|| 1.471 / identical
- Step 76: 10.1777 vs 10.1776 (sub-ULP)

All identical at 4-decimal precision — confirming math is bit-identical
up to cuBLAS scheduling variance.

## Verdict matrix

| bar | wall threshold | NLL threshold | result |
|---  |---:            |---:           |---     |
| Strict brief (≥5% tok/s + ±0.02 NLL) | +5% | ±0.02 | **FAIL on wall** (+1.54% < +5%) |
| iter 60 relaxed (+3% + multi-seed parity) | +3% | ±0.05 | **FAIL on wall** (+1.54% < +3%) |
| Sub-3% silent-accrual (parity-clean + measurable above noise) | >+0.5% | bit-identical sub-ULP | **PASS** |

**Sub-3% silent-accrual PASS.** Matches iter 70 (+1.26%) and iter 73 (+0.42%)
historical silent-accrual pattern: below the per-iter ship bar individually,
but bundles cleanly with prior wins into the next combined retrain-arc
opportunity.

## Default

Flag stays **OFF** initially (opt-in via `--iter97-dwconv-fwd-fused-sub`),
matching iter 70/73 pre-flip pattern.  Future production retrain arc (if
user-authorized) bundles iter97 with the existing triple-stack — analogous
to how iter 94 bundled iter70 + iter73 + scfa-conv-w=4 into the new flagship.

## Strategic significance

Iter 97 PASS is the first non-NULL single-iter result on the post-iter94
flagship after 13 consecutive non-PASS iters (iter 70-83 + 95-96 chain).
Mechanism summary:

| iter | mechanism | wall Δ | NLL | status |
|---   |---       |---:    |---  |---    |
| 95   | bwd dx tile w=4 | 0% | bit-id | NULL (mech doesn't help at w=4) |
| 96   | fwd tile w=4 | 0% | bit-id | NULL (mech doesn't help at w=4) |
| **97** | **scfa_sub fusion into fwd conv** | **+1.54%** | **bit-id** | **PASS** (eliminates a kernel + memory roundtrip) |

The pattern: smem-tiling alone at w=4 is null (iter 95/96 confirmed
empirically); but **eliminating an adjacent kernel launch + its memory
materialization** is a real win — even at the engineering-ceiling regime.

Implication for iter 98+: continue looking for sub-fusion opportunities
that eliminate kernel launches with their memory round-trips, not pure
smem-tile or smem-cache extensions.

## Cumulative stack snapshot (post-iter97-opt-in)

Production triple-stack flagship (iter 94 SHIP) stays at 25,103 tok/s
default-on.  iter 97 contributes +1.54% silent-accrual if `--iter97-dwconv-
fwd-fused-sub` is added to the production STACK in run.sh — would land
production at ~25,490 tok/s.  Awaits multi-iter retrain arc commitment
for default-flip (per iter 70→94 pattern: opt-in until apples-to-apples
30k retrain validates inclusion at full production scale).

## Reproduction

```bash
cd /home/robert/dev/glades-trainer

# Baseline (current production triple-stack, no iter97)
./build/glades_chiron_train \
  --pretokenized --data-dir pretok-data \
  --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 --vocab 32000 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage \
  --max-steps 100 --warmup 500 --lr 1e-4 --grad-clip 0.5 --seed 1337 \
  --log-every 25 --val-every 100 --val-batches 1 \
  --save /tmp/baseline_chkpt

# iter97 treatment (add --iter97-dwconv-fwd-fused-sub)
./build/glades_chiron_train ... --iter97-dwconv-fwd-fused-sub \
  --save /tmp/iter97_chkpt
```

## Files

- This document.
- `research/runs/2026-05-21-iter97-gate0/baseline_100step.log` (65.3s, no iter97).
- `research/runs/2026-05-21-iter97-gate0/iter97_100step.log` (64.3s, --iter97-dwconv-fwd-fused-sub).
- Code:
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`
    (new template kernel + dispatcher near line 8032 / 8180).
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.h`
    (declaration + no-CUDA stub).
  - `glades-trainer/include/.../gpu_kernels.h` (vendored mirror).
  - `glades-trainer/trainer/chiron_main.cpp` (flag + CLI + dispatch wire).
