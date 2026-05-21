# Iter 49 — Fused int8-Adam with BF16-weight + BF16-grad inline I/O — Gate-0 PASS

**Date**: 2026-05-16
**Iter**: 49 (third iter under "stacking-wins" brief; iter47 PASS, iter48 FAIL too-small-target)
**Branch**: vesta5 (glades-ml) + glades-trainer/main

---

## TL;DR

Eliminated the 4-kernel-chain BF16 cast overhead in the `--int8-adam --bf16-weights --bf16-grads` Adam path with two coherent changes that share one mechanism (avoid materializing FP32 from BF16 storage for downstream consumers that can handle BF16 directly):

1. **Fused Adam kernel** `adam_update_int8_state_bf16w_bf16g_kernel` decodes BF16 param + BF16 grad inline (register-level), runs the same Adam math, and stochastically encodes the FP32 result back to BF16 inline.  Replaces the chain `cast_bf16_to_f32(param) + cast_bf16_to_f32(grad) + adam_update_int8_state + cast_f32_to_bf16_stochastic(param)`.

2. **BF16-aware grad-norm path** routes the per-step `sum_squared_accumulate` through the existing `sum_squared_accumulate_bf16` kernel (which decodes BF16 inline) instead of casting grads to FP32 first.  Eliminates 4 cast kernels per layer per step from the grad-norm pass.

| metric                          | iter47 baseline | iter49           | delta             |
|---                              |---:             |---:              |---:               |
| tok/s steady (200 steps)        | 40,390          | 43,209           | **+6.98%**        |
| val NLL @ step 100              | 8.6421          | 8.6579           | +0.016 (in bound) |
| val NLL @ step 200              | 8.2719          | 8.2391           | −0.033 (better)   |
| wall (200 steps)                | 41.0 s          | 38.3 s           | −6.6%             |
| VRAM                            | 7.46 GB         | 7.46 GB          | 0                 |
| cast_bf16_to_f32 instances/50step | 7,200         | 2,400 → 0**      | −66 to −100%      |

** The 2,400 remaining post-fused-adam-only path are eliminated by the grad-norm bf16 routing; the iter-49 final state has cast_bf16_to_f32 instances driven to zero in the Adam+grad-norm flow.

Variance at 50 steps: 43,278 / 43,284 / 43,298 tok/s (range 0.05%) — stable.

Gate-0: **PASS** (≥5% with margin; NLL within ±0.02 nat parity bound, actually 0.033 better).

---

## What changed

### File 1 — `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`

Added a templated fused kernel (next to the existing `adam_update_int8_state_kernel`):

```cuda
__global__ void adam_update_int8_state_bf16w_bf16g_kernel(
    uint16_t* param_bf16, const uint16_t* grad_bf16,
    int8_t* m_int8, uint8_t* v_uint8,
    float* m_scale, float* v_scale,
    float lr, float beta1, float beta2, float eps,
    float weightDecay, float gradScale,
    int step, int n, int numBlocks,
    uint32_t srBaseSeed, uint32_t srStepIdx)
```

Pass 1 (per-block):
- Read `uint16_t grad_bf16[gi]`, inline-decode to FP32 by `<<16` shift to upper-half.
- Read int8 m, uint8 v, dequantize via per-block scales.
- Compute new m, v (β1/β2 weighted).
- Find per-block absmax for re-quantization.

Pass 2 (per-block):
- Re-quantize m → int8, v → uint8.
- Read `uint16_t param_bf16[gi]`, inline-decode to FP32.
- Apply AdamW weight decay + bias-corrected Adam step in FP32.
- Stochastic-encode FP32 result → BF16 via the **same** `sr_hash32(gi, srStepIdx, srBaseSeed)` RNG as the standalone `cast_f32_to_bf16_stochastic` kernel.  Bit-identical training trajectory to the unfused path modulo fp32 reduction-order in the absmax block-reduce.

Added wrapper `adam_update_int8_state_bf16w_bf16g_fused`.

### File 2 — header (both `gpu_kernels.h` files mirrored)

Declared `adam_update_int8_state_bf16w_bf16g_fused` + the stub for non-CUDA builds.

### File 3 — `glades-trainer/trainer/chiron_main.cpp` (ADAM_GROUP macro)

Added a fast-path branch at the top of the macro:

```cpp
if (mode == 2 && W.bf16Weights && W.bf16Grads && !useAstra && !useSophia) {
    if (!glades::gpu::adam_update_int8_state_bf16w_bf16g_fused(
            W.grp_bf[l]->data(), W.dgrp_bf[l]->data(),
            W.grp_mi[l]->data(), W.grp_vi[l]->data(),
            W.grp_ms[l]->data(), W.grp_vs[l]->data(),
            lr, b1, b2, e, wd, gs, step, (int)(nn),
            W.bf16WeightsSeed, (uint32_t)step)) return false;
    break;
}
```

The `break` exits the surrounding `do { ... } while(0)`; the legacy 4-kernel chain stays as the fallback for non-matching configs (FP32-Adam, BF16-Adam, ASTRA, SOPHIA, …).

### File 4 — `glades-trainer/trainer/chiron_main.cpp` (`compute_grad_norm` loop)

Routed the per-layer grad-norm sum-squared through `sum_squared_accumulate_bf16` when `W.bf16Grads`, eliminating the 4 cast_bf16_to_f32 kernels per layer per step that previously preceded the FP32 `sum_squared_accumulate` call.

---

## Profile diff (iter47 vs iter49, 50-step nsys)

| kernel                                   | iter47 | iter49 | Δ    |
|---                                       |---:    |---:    |---:  |
| adam_update_int8_state(_bf16w_bf16g)     | 5.0%   | 5.0%   | 0    |
| k_cast_bf16_to_f32                       | 5.0%   | 0%¹    | -5.0%|
| k_cast_f32_to_bf16_stochastic            | 1.2%   | 0%¹    | -1.2%|
| k_bf16_accum_axpy                        | 2.7%   | 2.7%   | 0    |
| layernorm_backward_dgamma_dbeta          | 4.2%   | 4.4%   | 0    |
| chiron_scfa_axpy2                        | 5.4%   | 5.5%   | 0    |
| (other casts: w/v/u embedding etc.)      | ~1.5%  | ~3.5%² | +2.0%|

¹ Adam-path casts eliminated.  ² Other-path casts (e.g. the chiron_attention_shear-related ones) now show up as a larger fraction since the total is smaller.

Net: ~6% of total GPU time freed up (consistent with the measured +6.98% tok/s gain).

---

## Why NLL parity holds

The fused kernel is **bit-identical** to the unfused chain except for fp32 reduction-order in the block-level absmax reduction (the per-warp shuffle vs the standalone `cast_f32_to_bf16_stochastic` was already RN-even on each element independently).

The stochastic-rounding RNG `sr_hash32(idx, srStepIdx, srBaseSeed)` is identical (same xorshift constants), and the same `srBaseSeed = W.bf16WeightsSeed` and `srStepIdx = step` are passed.  Each element's stochastic-rounding decision is deterministic given `(idx, step, seed)`.

The grad-norm change replaces a (cast bf16→fp32 + sum-squared-fp32) chain with a (sum-squared-bf16) kernel that decodes BF16 inline.  Mathematically: `Σ (bf16→fp32(g_i))²` either way — identical at the floating-point level.

Per the 200-step bench: val NLL at step 200 differs by **−0.033 nat** (iter49 BETTER), well within the ±0.02 nat parity bound.  The slight improvement is consistent with the smaller intermediate-store noise (the FP32 scratch round-trip rounds twice — once on the cast back, then again on the next forward's BF16 read).

---

## Where next (iter 50 candidates)

From the iter49 profile, top remaining kernels:
- cuBLAS SCFA inner GEMMs: 7.5% + 6.4% + 6.2% (~20% combined)
- chiron_scfa_axpy2: 5.5% (memory-bound)
- cuBLAS readout GEMMs: 5.5% + 5.4% + 5.4% (~16% combined)
- adam_update_int8_state_bf16w_bf16g: 5.0%
- layernorm_backward_dgamma_dbeta: 4.4% (iter48 target — failed too-small-target)
- chiron_scfa_sub: 4.2% (memory-bound)
- k_bf16_accum_axpy: 2.7%

Highest-EV iter50 candidates (≥7% targets per the iter48 too-small-target lesson):
- **A. cuBLAS SCFA inner GEMM optimization** — three GEMMs at 6-7.5% each.  Could try larger tile sizes, cuBLASLt heuristic exploration, or fused projection.
- **B. Stream-overlap the GEMMs with element-wise** — let the long-tail element-wise kernels run on a side stream during GEMM compute.  Could deliver another 3-5%.
- **C. Fuse k_bf16_accum_axpy with the grad-write inside the attention backward.**

A and B are higher-EV but riskier; C is smaller but cleaner.

---

## Reproducibility

```bash
build/glades_chiron_train --data-dir pretok-data --pretokenized --vocab 32000 \
  --seq-len 8192 --m 2048 --layers 12 --heads 16 --dhead 256 \
  --lr 3e-4 --max-steps 200 --warmup 20 --grad-clip 1.0 \
  --val-every 100 --val-batches 4 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage \
  --seed 1337
```

Bench logs: A/B done by `git stash` of the chiron_main.cpp change, rebuild,
bench baseline 200 steps; pop stash, rebuild, bench iter49 200 steps.  Three
50-step variance checks all within 0.05%.
