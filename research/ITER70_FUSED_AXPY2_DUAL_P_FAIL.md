## Iter 70 — Fused dual-output axpy2 for iter65 BF16-p mirror — FAIL

**Date**: 2026-05-19
**Iter**: 70 (sixteenth iter under "stacking-wins" brief; sixth iter under iter54 META Option A relaxed-bar regime)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)
**Verdict**: **FAIL** — +1.30–1.38% tok/s (below the +5% per-iter bar) AND NLL drift outside the ±0.02 nat strict parity bound (in the BETTER direction).  Mechanism is sound but FMA-emit divergence vs the unfused (chiron_scfa_axpy2 + cast_f32_to_bf16_stochastic) chain produces a different (slightly-better) training trajectory.  Kept in tree as opt-in `--iter70-fused-axpy2-dual-p`; default OFF for silent-accrual safety.

---

## Problem statement

iter 70 nsys profile of the L=24 T=16384 production stack (with iter69 BF16-cache) revealed a new attackable cast slice (combined `k_cast_f32_to_bf16` variants + `k_cast_f32_to_bf16_stochastic` = ~7-8% GPU time on cast kernels).  The biggest single cast slice (2.4% GPU time, 2,059 instances) is the iter65 BF16-residual-p SR mirror — `cast_f32_to_bf16_stochastic` runs immediately after every `chiron_scfa_axpy2` to re-cast the just-written FP32 `s.p` into the BF16 mirror `s.p_bf16`.

Each (`chiron_scfa_axpy2`, `cast_f32_to_bf16_stochastic`) pair:
- reads `s.p` (T·m × 4 B FP32) — twice (once in axpy2 to update, once in cast to encode)
- writes `s.p` (T·m × 4 B FP32) — once in axpy2
- writes `s.p_bf16` (T·m × 2 B BF16) — once in cast

Total HBM per pair: 4·T·m bytes read + 5·T·m bytes write ÷ 2 (BF16 half-width) = 9·T·m bytes per layer per direction × L=24 × 2 dirs = 432·T·m bytes/step.  At T·m = 33.5M, T=16384, that's 14.4 GB/step of cast-related HBM.

## Hypothesis

Replace the two-kernel sequence with one fused `chiron_scfa_axpy2_dual_p_kernel` that:
- computes `new_p = p_fp32[i] + alpha · (a[i] + b[i])` in an FP32 register
- writes BOTH `p_fp32[i] = new_p` AND `p_bf16[i] = SR(new_p, i, srStepIdx, srBaseSeed)`

This eliminates:
- 1 kernel launch per layer per direction (forward axpy2 + cast pair → single dual_p call; same for backward inverse-shear)
- 1 HBM read of `s.p` per layer per direction (cast no longer re-reads what axpy2 just wrote)

Predicted gain: ~+1.0% to +1.8% wall at iso-NLL parity (HBM-bandwidth bound × cast slice + launch overhead).

## Implementation

`glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu`:

```cuda
__global__ void chiron_scfa_axpy2_dual_p_kernel(float* __restrict__ p_fp32,
                                                 unsigned short* __restrict__ p_bf16,
                                                 float alpha,
                                                 const float* __restrict__ a,
                                                 const float* __restrict__ b,
                                                 int n,
                                                 uint32_t srBaseSeed,
                                                 uint32_t srStepIdx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float p_val = p_fp32[i];
    p_val += alpha * (a[i] + b[i]);
    p_fp32[i] = p_val;
    p_bf16[i] = fp32_to_bf16_sr_dev(p_val, (uint32_t)i, srStepIdx, srBaseSeed);
}
```

Header + wrapper: `chiron_scfa_axpy2_dual_p(...)` in `gpu_chiron.h` (both glades-ml and glades-trainer mirror).

`glades-trainer/trainer/chiron_main.cpp`:

- Added `Config::iter70FusedAxpy2DualP` (default OFF after FAIL verdict).
- CLI flags: `--iter70-fused-axpy2-dual-p` / `--no-iter70-fused-axpy2-dual-p`.
- At lines 6439 (forward SCFA shear) and 7065 (backward inverse-shear): branch on `cfg.iter70FusedAxpy2DualP && cfg.bf16ResidualP && s.p_bf16.allocated()`.  Fused branch uses `chiron_scfa_axpy2_dual_p` with preserved `sr_axpy2_counter` / `sr_axpy2_bwd_counter` semantics.  Unfused else-branch preserves the iter65+iter69 (axpy2 + SR-cast) pair as before.

## Bench results (L=24 T=16384, 200 steps, seed=1337)

All three runs use the same production stack: `--int8-adam --bf16-grads --bf16-weights --bf16-attn --no-fuse-attn --fuse-attn-reln --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt --bf16-logits --bf16-logits-storage --scfa-checkpoint-inner --scfa-checkpoint-inner-bf16`.

| metric                       | iter 69 baseline | iter 70 v1 (fused, naive) | iter 70 v2 (fused, FMA-match attempt) |
|---                           |---:              |---:                       |---:                                   |
| steady-state tok/s (mean)    | 24,320           | 24,655                    | 24,636                                |
| wall (200 steps)             | 136.5 s          | 134.5 s                   | 134.6 s                               |
| Δ tok/s vs baseline          | —                | **+1.38%**                | **+1.30%**                            |
| Δ wall vs baseline           | —                | −1.47%                    | −1.39%                                |
| step-100 val NLL             | 8.5905           | 8.3989                    | 8.5068                                |
| step-200 val NLL             | 7.7583           | 7.6392                    | 7.6405                                |
| Δ NLL @ step 100 vs baseline | —                | **−0.192** (BETTER)       | **−0.084** (BETTER)                   |
| Δ NLL @ step 200 vs baseline | —                | **−0.119** (BETTER)       | **−0.118** (BETTER)                   |

Per-step training loss (selected, seed=1337):

| step | iter 69 base | iter 70 v1 | iter 70 v2 | v2 vs base |
|---:  |---:          |---:        |---:        |---:        |
|  1   | 10.5762      | 10.5762    | 10.5762    | bit-identical |
|  11  | 10.0846      | 10.0845    | **10.0846**| **bit-identical** (v2 FMA match) |
|  21  |  9.4884      |  9.4874    |  9.4872    | −0.0012     |
|  41  |  9.0502      |  9.0529    |  9.0461    | −0.0041     |
|  81  |  8.3728      |  8.3460    |  8.3788    | +0.0060     |
| 191  |  7.5118      |  7.4534    |  7.5691    | +0.0573     |

v2 (with `p_val = p_fp32[i]; p_val += alpha * (a[i] + b[i]); p_fp32[i] = p_val;` form) achieves bit-identical loss at step 11 (10.0846 matches baseline exactly) — confirming the FMA-emit match works at first reported checkpoint.  Sub-ULP rounding drift in the SR mirror (BF16 representation has ~7-bit mantissa precision) then compounds over subsequent training steps and is amplified by the gradient pathway, producing the +0.084 nat val drift by step 100.

## Why this FAILS

1. **Tok/s win below the +5% strict bar**: +1.38% (v1) and +1.30% (v2) are real, reproducible wins from kernel-launch reduction (~48 fewer launches/step at L=24) plus HBM saturation (~134 MB/layer/direction saved on the p re-read).  But far short of the strict +5% per-iter bar.

2. **NLL parity exceeds the ±0.02 nat strict bound**: The fused kernel's FP32 emit is not bit-identical to the (axpy2 + cast) chain even with the v2 FMA-emit-match attempt.  Compiler register-pressure shifts cause sub-ULP drift in `s.p`, which feeds back into next layer's reln-bwd path (T=16384's 33.5M elements compound the noise).  By step 200 the drift is 0.118 nat — well outside ±0.02 nat.  Notably the drift is in the BETTER direction (lower NLL), suggesting the alternative trajectory happens to converge faster on this seed; multi-seed eval would be required to assess if this is luck or signal.

3. **Pattern matches iter 50 (sub-into-conv fusion FAIL)**: Same FMA-emit-divergence mechanism.  iter 50 noted: "Compiler-emitted FMA from fused kernel ≠ legacy chain's FMA at fp32 even with explicit __fmaf_rn." iter 70 inherits this.

## Categorization

**impl-fail** (FMA-emit divergence vs unfused chain).  Mechanism is mathematically sound; the divergence is from compiler-internal arithmetic ordering, not a math error.  Same category as iter 50.

Difference from iter 50: iter 70's drift is in the BETTER direction (lower NLL) rather than worse — but still fails the symmetric ±0.02 nat parity bound.

## Default policy

`--iter70-fused-axpy2-dual-p` is **default OFF**.  Mechanism stays in tree as opt-in for:
- A/B testing in future iters that revisit the BF16-residual-p mirror path
- Combined retro-ship parity evaluation (multi-seed) if iter70+other below-bar wins are bundled

Production runs use the parity-validated iter65 + iter69 unfused chain.  No code path regression.

## Files changed

- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu`: new `chiron_scfa_axpy2_dual_p_kernel` + wrapper.
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.h`: prototype.
- `glades-trainer/include/Backend/Machine Learning/Networks/cuda/gpu_chiron.h`: mirror prototype.
- `glades-trainer/trainer/chiron_main.cpp`: Config field + CLI flag + branched call sites (forward at line 6439, backward at line 7065).

## Bench commands

**iter 70 (fused, opt-in)**:
```bash
build/glades_chiron_train --data-dir pretok-data --pretokenized --vocab 32000 \
  --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 \
  --lr 3e-4 --max-steps 200 --warmup 20 --grad-clip 1.0 \
  --val-every 100 --val-batches 4 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage \
  --scfa-checkpoint-inner --scfa-checkpoint-inner-bf16 \
  --iter70-fused-axpy2-dual-p \
  --seed 1337
```

**iter 69 baseline**: same command, replace `--iter70-fused-axpy2-dual-p` with `--no-iter70-fused-axpy2-dual-p` (or omit; default is OFF).

## Profile artifact

`research/runs/2026-05-19-iter70-profile/iter70_baseline.nsys-rep` — fresh L=24 T=16384 nsys profile of the iter 69 stack used to identify the cast slice.

## Sequence status (16 iters)

| iter | target | result | win |
|---: |---     |---     |---:  |
| 56  | warp argmax            | FAIL→retro-ship    | +1.62% |
| 57  | accum_axpy+sumsq       | PUNT               | n/a    |
| 58  | --bf16-logits-parallel-bwd | NULL           | +0.15% |
| 59  | cuBLAS algo (SCFA outer) | NULL             | ~0%    |
| 60  | combined retro-ship    | **PASS**           | **+9.20%** |
| 61  | BF16-grad direct cuBLAS out | FAIL          | +2.64% |
| 62  | FP8 readout (Arc 1)    | NULL (toolkit)     | 0%     |
| 63  | --bf16-residual-p framework | FRAMEWORK     | n/a    |
| 64  | SR variants (Arc 2)    | INCONCLUSIVE       | n/a    |
| 65  | BF16-p full routing    | FAIL (silent)      | +1.6%  |
| 66  | BF16-p prod kill test  | PASS               | n/a    |
| 67  | BF16-p L=24 prod       | MARGINAL PASS      | n/a    |
| 68  | BF16-p default-on ship | SHIP               | +4.28% |
| 69  | BF16 checkpoint-inner cache | **PASS**      | **+6.83%** |
| 70  | fused axpy2_dual_p     | **FAIL** (NLL drift) | +1.4% |
