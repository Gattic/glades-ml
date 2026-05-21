## Iter 73 — Shared-memory tiled scfa_depthwise_causal_conv_fwd — FAIL

**Date**: 2026-05-19
**Iter**: 73 (nineteenth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)
**Verdict**: **FAIL** — +0.56% tok/s (below +5% bar) and NLL drift −0.127 nat (BETTER direction, outside ±0.02 strict parity).  FMA-emit divergence from `#pragma unroll` of the inner conv loop produces a slightly different training trajectory.  Same iter 50/70 pattern.  Default OFF; opt-in flag `--iter73-dwconv-fwd-tiled` kept for combined retro-ship eval.

---

## Problem statement

iter 70 profile at L=24 T=16384 production stack shows `scfa_depthwise_causal_conv_fwd_kernel` at 3.9% GPU time (24 ms / step).  L2 cache hit rate ~81% per analytic: each `x[s, c]` is read by w+1=9 output positions; consecutive output rows share most of their input window.  Remaining 19% misses are HBM reads.

Hypothesis: a 32-row × 256-column shared-memory tile (later reduced to 16-row to fit 48 KB static shared mem limit) caches the input window once and serves all output rows from shared, eliminating the L2 miss pathway.  Bit-identical math (same K*x accumulation order, same break-on-negative-src).

## Implementation

`glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`:

```cuda
template<int COLS_PER_BLOCK, int N_OUT, int W_FILTER>
__global__ void scfa_depthwise_causal_conv_fwd_tiled_kernel(...) {
    const int t_base = blockIdx.y * N_OUT;
    const int c = blockIdx.x * COLS_PER_BLOCK + threadIdx.x;

    __shared__ float x_smem[N_OUT + W_FILTER - 1][COLS_PER_BLOCK];  // 24 × 256 × 4 = 24 KB
    __shared__ float K_smem[COLS_PER_BLOCK][W_FILTER];               // 256 × 9 × 4 = 9 KB

    // Each thread loads its col's W_FILTER filter weights + X_ROWS x rows.
    // ... cooperatively populate shared mem ...
    __syncthreads();

    #pragma unroll
    for (int dt = 0; dt < N_OUT; ++dt) {
        // ... compute output for t_base + dt ...
        #pragma unroll
        for (int i = 0; i < W_FILTER; ++i) {
            if (t_out - i < 0) break;
            acc += K_smem[threadIdx.x][i] * x_smem[w + dt - i][threadIdx.x];
        }
        y[(size_t)t_out * (size_t)m + (size_t)c] = acc;
    }
}
```

Dispatch at w=8: COLS=256, N_OUT=16, W_FILTER=9.  Falls back to row-major kernel for non-default w.

`glades-trainer/trainer/chiron_main.cpp`: Added `Config::iter73DwconvFwdTiled` (default OFF) + CLI flag `--iter73-dwconv-fwd-tiled`.  All 4 conv_fwd call sites (forward parallel/sequential + backward-recompute parallel/sequential) gated on the flag.

## Bench (L=24 T=16384, 200 steps, seed=1337)

| metric                       | iter 69 baseline | iter 73 (tiled)     |
|---                           |---:              |---:                  |
| steady-state tok/s (mean)    | 24,320           | 24,455               |
| wall (200 steps)             | 136.5 s          | 135.7 s              |
| **Δ tok/s vs baseline**      | —                | **+0.56%**           |
| **Δ wall vs baseline**       | —                | **−0.59%** (faster)  |
| step-200 val NLL             | 7.7583           | 7.6312               |
| Δ NLL @ step 200             | —                | **−0.127** (BETTER) |
| peak VRAM                    | 14.97 GB         | 14.97 GB             |

Per-step training loss divergence:

| step | iter 69 base | iter 73 tiled | Δ        |
|---:  |---:          |---:           |---:      |
|  1   | 10.5762      | 10.5762       | bit-identical |
| 11   | 10.0846      | 10.0846       | bit-identical |
| 21   |  9.4884      |  9.4882       | −0.0002  |
| 51   |  8.9906      |  9.0179       | +0.0273  |
| 101  |  8.4856      |  8.4276       | −0.0580  |
| 200v |  7.7583      |  7.6312       | −0.127   |

Bit-identical at step 11 confirms math equivalence at the first reported checkpoint.  Sub-ULP drift emerges by step 21 and amplifies through the gradient/Adam pathway.

## Why this fails

1. **Tok/s win below +5% bar**: +0.56% wall improvement.  Real and reproducible, but the L2 cache is already ~81% efficient — shared-memory tiling only converts the 19% miss path to zero HBM, which is a smaller win than the analytical 30-40% I'd hoped for in a less-cached kernel.

2. **NLL drift outside ±0.02 strict bound**: The tiled kernel's `#pragma unroll` on the outer N_OUT loop creates 16 copies of the inner conv-loop body.  NVCC schedules FMAs across the unrolled copies differently than in the legacy single-loop kernel — register allocation + instruction scheduling differ even though the *expression* is identical (`acc += K * x`).  Sub-ULP FP32 drift compounds over L=24 layers and many training steps.

3. **Same iter 50/70 FMA-emit pattern**: iter 50 (sub-into-conv fusion FAILED with +0.04 nat drift) and iter 70 (axpy2 fusion FAILED with +0.119 nat drift) hit the same compiler-FMA-divergence issue.  Even with `__fmaf_rn` explicit FMA intrinsics, NVCC may still re-schedule under register pressure changes (iter 50 doc).

## Categorization

**impl-fail** (FMA-emit divergence from compile-time unrolling).  Mechanism is mathematically sound; the wall win is real (+0.56%) but the FMA pathway produces a different (this seed, better-direction) trajectory.

Pattern matches:
- iter 50: sub-into-conv fusion (+1.31% wall, +0.04 nat drift)
- iter 70: fused axpy2_dual_p (+1.4% wall, −0.119 nat drift)
- iter 73: tiled dwconv_fwd (+0.56% wall, −0.127 nat drift)

All three: fused/restructured custom kernels produce ULP-scale FP32 divergence from the legacy chain.

## Default policy

`--iter73-dwconv-fwd-tiled` is **default OFF**.  Mechanism stays in tree as opt-in via the flag.  Production uses parity-validated row-major kernel.

## Sequence status (19 iters)

| iter | target | result | win |
|---: |---     |---     |---:  |
| 68  | BF16-p default-on ship | SHIP               | +4.28% |
| 69  | BF16 checkpoint-inner cache | **PASS**      | **+6.83%** |
| 70  | fused axpy2_dual_p     | FAIL (NLL drift)   | +1.4% silent |
| 71  | --scfa-parallel-branches | NEGATIVE         | −5.4% |
| 72  | batched cast kernel    | NULL               | ±0% |
| 73  | tiled dwconv_fwd       | FAIL (NLL drift)   | +0.56% silent |

Four iters post-iter69, no PASS.  Engineering ceiling pattern dominates.

## Bench command

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
  --iter73-dwconv-fwd-tiled \
  --seed 1337
```

## Files

- This document (iter 73 fail result).
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`: new
  `scfa_depthwise_causal_conv_fwd_tiled_kernel` + wrapper.
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.h`: prototype.
- `glades-trainer/include/Backend/Machine Learning/Networks/cuda/gpu_kernels.h`: mirror.
- `glades-trainer/trainer/chiron_main.cpp`: Config flag + CLI flag + 4 gated call sites.
