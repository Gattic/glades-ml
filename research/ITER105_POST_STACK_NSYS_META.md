## Iter 105 — Post-iter-103-stack nsys re-profile — META

**Date**: 2026-05-21
**Iter**: 105 (META, no new kernel — analogous to iter 96)
**Branch**: vesta5 (glades-ml) — no commits
**Verdict**: **META** — fresh profile reveals shifted bottleneck distribution after the iter 97+99+101+103 +5.71% combined stack. Identifies the next iter target candidates and confirms the engineering ceiling is increasingly tight.

---

## Motivation

After iter 97+99+101+103 took their pieces (+5.71% n=3 multi-seed wall), the kernel breakdown has shifted. The iter 96 nsys profile is stale. Re-profile to:
1. Confirm which kernels iter 102 actually replaced (causal_mask_softmax fwd vs bwd)
2. Identify new top kernels with the iter-97-family fused kernels visible
3. Calibrate iter 106+ target selection

## Profile (25-step nsys run, iter 97+99+101+103 + iter 102 unconditional)

Top 20 kernels by % wall (post-iter-103 stack):

| % wall | kernel | calls/step | per-call avg |
|---:|---|---:|---:|
| 15.3 | cutlass_s1688bf16gemm_256x128_16x3_**nn** (SCFA outer Q/K/V/Wo fwd) | 124 | 744 µs |
| 10.2 | cutlass_s1688bf16gemm_256x128_16x3_**nt** (SCFA outer bwd) | 74 | 835 µs |
| **8.0** | chiron_scfa_axpy2_dual_p (iter 70 fused) | 50 | 964 µs |
| 5.1 | ampere_s1688gemm_bf16_128x128 (smaller bf16 GEMM) | 150 | 204 µs |
| 3.6 | cutlass_bf16_s16816gemm tn (readout fwd) | 1.08 | 20,124 µs |
| **3.6** | **scfa_dwconv_dx_tiled_kernel_dual_out<256,16,5> (iter 99)** | 24 | 896 µs |
| **3.4** | **scfa_depthwise_causal_conv_fwd_sub_fused_dual_out_tiled_kernel<256,16,5> (iter 101)** | 24 | 853 µs |
| 3.3 | cutlass_s16816gemm nn (readout dE bwd) | 1.0 | 19,652 µs |
| 3.2 | ampere_s16816gemm nt (readout dq bwd) | 1.0 | 19,558 µs |
| **3.0** | adam_update_int8_state_bf16w_bf16g_kernel | 96 | 191 µs |
| **2.7** | **scfa_depthwise_causal_conv_fwd_sub_fused_tiled_kernel<256,16,5> (iter 97)** | 27 | 641 µs |
| 2.7 | ampere_bf16_s16816gemm | 72 | 225 µs |
| 2.6 | layernorm_backward_dx | 24 | 648 µs |
| **2.6** | **causal_mask_softmax_kernel (legacy — bwd recompute path)** | 48 | 321 µs |
| 2.4 | cutlass_s1688gemm_256x64 tn | 72 | 205 µs |
| 2.1 | ampere_s1688gemm_bf16 tn | 72 | 176 µs |
| **2.0** | chiron_reln_forward_rows | 26 | 475 µs |
| 1.9 | cutlass_s1688gemm_256x64 nt | 48 | 235 µs |
| 1.8 | scfa_dwconv_dK_kernel_par | 24 | 460 µs |
| 1.8 | layernorm_backward_dgamma_dbeta_partial | 24 | 450 µs |
| 1.7 | chiron_reln_inverse_rows | 24 | 437 µs |
| 1.7 | k_cast_f32_to_bf16 (multiple call sites) | ~303 | 34 µs |
| 1.7 | chiron_scfa_scaled_copy_kernel | 24 | 425 µs |
| 1.5 | **causal_mask_softmax_bf16_out_kernel (iter 102 fused)** | 26 | 341 µs |

## Key findings

### 1. iter 97/99/101/102 fused kernels are now visible as distinct top entries

The iter-97-family kernels (the +5.71% stack contributors) collectively occupy 3.6 + 3.4 + 2.7 = **9.7% of step wall** — exactly the slot previously held by legacy scfa_sub (5.1%) + conv_fwd row-major (3.3%) + dwconv_dx row-major (2.4%) = 10.8% pre-stack. Fusion didn't shrink the total compute much, but it eliminated kernel launches and memory roundtrips between sub-ops.

### 2. iter 102's fused softmax handles only ~35% of softmax calls

| kernel | calls/step | % wall |
|---:|---:|---:|
| causal_mask_softmax (legacy, bwd recompute) | 48 | 2.6% |
| causal_mask_softmax_bf16_out (iter 102 fused, fwd inner) | 26 | 1.5% |

iter 102 only fused the fwd path inside `flash_attention_cublas_tiled_bf16`. The bwd recompute path (`flash_attention_backward_cublas_tiled` line 1664, `flash_attention_cublas_tiled` line 1539) still uses legacy `causal_mask_softmax_inplace`. The bwd-side softmax produces FP32 P (consumed by `softmax_backward_attn`) — iter 102's BF16-output variant doesn't apply directly.

Potential iter target: **bwd-recompute softmax fusion**. The bwd softmax output FP32 P is read by softmax_backward_attn (which also produces dS). Could fuse softmax+softmax_backward_attn? Per-call 321 µs softmax + ~266 µs bwd_attn = 587 µs. Fusing saves the 64 MB P round-trip between them ≈ 0.5-1% wall savings if ≥1 GB/step memory traffic (need 48 calls × 64 MB read + 64 MB write = 6 GB/step — well above iter 104 threshold).

### 3. chiron_scfa_sub kernel is no longer in top kernels

iter 97 (fwd) + iter 101 (bwd-recompute) eliminated the chiron_scfa_sub kernel from production hot path. Was 5.1% in iter 96 profile. Now sub-noise (not in top 20).

### 4. axpy_kernel is no longer in top kernels

iter 99 dual-output writes folded the line-7511 axpy into the bwd dwconv dx kernel. axpy_kernel was 2.6% in iter 96; now 0.1% (just 25 calls/step, the remaining sites outside iter 99's wire).

### 5. Largest remaining un-attacked targets

| target | % wall | tractability | notes |
|---|---:|---|---|
| adam_update_int8_state | 3.0% | LOW | 96 calls/step, complex int8 quant + state update; reduction-bound |
| causal_mask_softmax (bwd) | 2.6% | MEDIUM | Could fuse with softmax_backward_attn (~6 GB/step memory traffic above iter 104 threshold) |
| layernorm_backward_dx | 2.6% | LOW | Row-wise reductions, hard to fuse |
| chiron_reln_forward_rows | 2.0% | LOW | Already reln-opt'd, reductions block further fusion |
| chiron_reln_inverse_rows | 1.7% | LOW | Same as fwd |
| chiron_scfa_scaled_copy (bwd entry) | 1.7% | LOW | Move-only, fusion blocked by multiple consumers |
| layernorm_backward_dgamma_dbeta | 1.8% | LOW | Row-wise reductions |

### 6. cuBLAS GEMMs dominate at ~50% of step wall

Composition of cuBLAS-related kernels in top 20:
- Outer SCFA Q/K/V/Wo GEMMs: 15.3 + 10.2 = 25.5%
- Smaller BF16 GEMMs (various ampere kernels): 5.1 + 2.7 + 2.1 + 2.4 + 1.9 = 14.2%
- Readout 3 GEMMs: 3.6 + 3.3 + 3.2 = 10.1%

Total cuBLAS: ~50%. Iter 51-94 + iter 52/59/78 NULL probes already explored cuBLAS algo overrides + workspace tuning. Further cuBLAS-side optimization requires precision tier change (FP8 blocked on CUDA 12.0 toolkit) or multi-iter FlashAttention-fused SCFA inner (per iter 82 strategic).

## Iter 106+ candidate priority

**Highest EV per iter 105 profile** (sorted by expected wall delta × tractability):

1. **Bwd recompute softmax + softmax_backward_attn fusion** — single-iter, ~6 GB/step memory traffic above threshold, predicted +0.5-1.5% wall. Math bit-identical with care (fuse two passes within one row-row block).

2. **CUDA Graph capture of per-layer fwd** — single multi-day task; 30-40 launches/layer × ~20 µs overhead × 24 layers = ~14-19 ms / step = 2-3% wall headroom. iter 105 confirms launch overhead is significant given 100+ small launches in top kernels.

3. **FlashAttention-fused SCFA inner** — multi-iter (per iter 82 strategic). +5-10% wall ceiling at ratio=32 (with iter 41 NLL pattern risk). High EV but multi-iter scope.

**Sub-2% targets (less worthwhile single-iter):**
- Adam batching / kernel optimization
- Reln fwd/inverse smem caching
- Other small memcpy/cast eliminations (per iter 104 threshold, must be ≥1 GB/step)

## Cumulative state

Combined opt-in stack **iter 97+99+101+103** delivers **+5.71% n=3 multi-seed wall** at NLL bit-identical (iter 100 + iter 103 validation evidence).

Cumulative since pre-ralph-loop: ~26,738 tok/s = **1.76× over 15,200 tok/s**. 1.5× target hit at iter 10 ship; 3×/10× targets remain (multi-iter scope per iter 82 strategic).

## Files

- This document.
- `/tmp/iter105_profile.nsys-rep` (nsys profile, 25-step bench with iter 97+99+101+103 + iter 102 unconditional).
- No new code (META iter).
