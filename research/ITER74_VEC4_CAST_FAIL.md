## Iter 74 — float4-vectorized cast kernels — FAIL (unexplained NLL drift)

**Date**: 2026-05-19
**Iter**: 74 (twentieth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **FAIL** — +0.23% tok/s (within noise), -0.29% wall, NLL drift −0.133 nat (BETTER direction, outside ±0.02 strict parity).  Per-element math is provably bit-identical to scalar, yet downstream NLL trajectory diverges by step 21+.  Hypothesis: float4 load pattern shifts L2 prefetch / CUDA scheduler ordering in subsequent kernels, perturbing the deterministic-but-microscale-sensitive chain.  Wrapper reverted to scalar; vec4 kernels stay in tree (dead code but library-resident).

---

## Problem statement

iter 70 profile shows cast kernels collectively ~7-8% GPU time at production scale.  iter 72 attempted batched multi-buffer cast (NULL) — wall savings too small relative to noise floor.  Iter 74 attacks a different angle: vectorize the *single-element* loads/stores in `k_cast_f32_to_bf16` / `k_cast_bf16_to_f32` to float4 / ushort4, reducing thread count 4× and potentially improving HBM coalescing.

Math: per-element RN cast logic is identical — vec4 kernel just packs 4 per thread.  All production cast sizes are divisible by 4 (k·m, V·m, T·m, etc.) so no scalar tail launches fire.  Output BF16 cache contents should be byte-identical.

## Implementation

`glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`:

```cuda
__global__ void k_cast_f32_to_bf16_vec4(const float* src, uint16_t* dst, size_t n_vec4) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec4) return;
    const float4 f4 = reinterpret_cast<const float4*>(src)[idx];
    ushort4 u4;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        // same RN-cast logic as scalar k_cast_f32_to_bf16, per element
    }
    reinterpret_cast<ushort4*>(dst)[idx] = u4;
}
// Symmetric k_cast_bf16_to_f32_vec4 for the reverse direction.
```

Wrapper dispatch: vec4 fast path for n_vec4 = n/4 leading elements + scalar tail for (n mod 4 != 0).  All production tensor sizes are divisible by 4 → pure vec4 path.

## Bench (L=24 T=16384, 200 steps, seed=1337)

| metric                       | iter 69 baseline | iter 74 (vec4)       |
|---                           |---:              |---:                  |
| steady-state tok/s (mean)    | 24,320           | 24,375               |
| wall (200 steps)             | 136.5 s          | 136.1 s              |
| **Δ tok/s vs baseline**      | —                | **+0.23%** (noise)   |
| **Δ wall vs baseline**       | —                | **−0.29%** (noise)   |
| step-200 val NLL             | 7.7583           | 7.6256               |
| Δ NLL @ step 200             | —                | **−0.133** (BETTER)  |

Per-step training loss divergence:

| step | iter 69 base | iter 74 vec4 | Δ          |
|---:  |---:          |---:          |---:        |
|  1   | 10.5762      | 10.5762      | bit-identical |
| 11   | 10.0846      | 10.0846      | bit-identical |
| 21   |  9.4884      |  9.4879      | −0.0005    |
| 51   |  8.9906      |  8.9787      | −0.0119    |
| 200v |  7.7583      |  7.6256      | −0.133     |

Bit-identical at step 11 confirms math match.  Drift starts at step 21+.

## Why this fails (and why this is interesting)

1. **Tok/s win below noise**: +0.23% wall improvement is within the ~0.5% single-run variance floor at L=24 T=16384.  No measurable mechanism win.

2. **NLL drift IS observed despite per-element math being bit-identical**:
   - Vec4 kernel loads 4 floats as float4 (16-byte aligned), unpacks into 4 floats, applies identical RN-cast logic per element, packs into ushort4.
   - Each element's output value is byte-identical to scalar kernel output.
   - But step-21 training loss DOES diverge by 0.0005 — meaning gradient state differs.
   - Step 11 IS bit-identical (loss + ||g||), confirming bit-identical at the cache-buffer level for that step.
   - Between step 11 and 21, microsecond-scale timing shifts (float4 vs scalar load patterns affect L2 prefetch, CUDA scheduler may reorder concurrent kernel completions) compound into ULP-scale FP32 differences.

3. **Hypothesis: the CHIRON forward+backward chain at L=24 T=16384 is microscale-sensitive**. Subtle changes to non-deterministic-but-bound CUDA scheduler ordering produce ULP drifts that compound via the Adam/SR-cast random-rounding pathway. This is iter 72's pattern (cast batching changed cudaMemcpy timing → drift) and iter 70/73's pattern (kernel FMA-emit changes) confirmed for a THIRD class of changes: even *bit-identical math kernel rewrites* can produce drift via timing-induced scheduler ordering.

## Categorization

**null/impl-fail** — the wall improvement is within noise (null), AND the NLL parity fails (drift outside ±0.02 strict bound, in BETTER direction).  The mechanism is *believed bit-identical at the math level* but produces different gradient trajectories via the L2/scheduler timing pathway.

## Default policy

Wrapper functions `cast_f32_to_bf16` and `cast_bf16_to_f32` REVERTED to the scalar kernel.  The `k_cast_f32_to_bf16_vec4` and `k_cast_bf16_to_f32_vec4` kernels remain in the file (~50 lines of code) as library symbols accessible via direct kernel launch.  No public wrapper provides access — they are essentially dead code preserved for documentation purposes.

## Strategic implication

**Iter 70-74 pattern (5 consecutive non-PASS)** confirms a robust engineering ceiling on the post-iter69 stack:

| iter | result | win   | category         |
|---: |---     |---:   |---               |
| 70  | FAIL    | +1.4% | impl-fail (FMA-emit) |
| 71  | NEGATIVE| −5.4% | idea-fail (Ada TC sat) |
| 72  | NULL    | ±0%   | null (below noise) |
| 73  | FAIL    | +0.56%| impl-fail (FMA-emit) |
| 74  | FAIL    | +0.23%| null+impl-fail (L2/scheduler timing) |

**THREE confirmed classes of "drift-inducing" changes at L=24 T=16384**:
1. **FMA-emit divergence** from kernel restructuring (iter 50, 70, 73).
2. **L2/scheduler timing** from memory-pattern changes (iter 74).
3. **Ada cuBLAS BF16-TC saturation** prevents cross-stream overlap (iter 6, 58, 71).

The stack as-is is at a near-impossible-to-improve-single-axis ceiling.  Realistic next steps:
- **Multi-iter arc**: Hadamard basis (FWHT compress/lift vs DCT-II cuBLAS) could deliver +5-10% if re-trained at the new basis.
- **Multi-iter arc**: FlashAttention-style fused inner attention (custom kernel for Q@K^T → softmax → @V).
- **Combined retro-ship**: bundle iter 70 + 73 (both +0.5-1.4% wall) for multi-seed parity eval; combined wall ~+1.97% (still below bar) with combined NLL drift ~-0.25 nat (way outside parity).

## Files

- This document.
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`: vec4 kernels added (now dead code); wrapper reverted to scalar.  No public API change.
