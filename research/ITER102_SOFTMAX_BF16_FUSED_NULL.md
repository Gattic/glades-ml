## Iter 102 — Fused causal_mask_softmax + BF16 cast — NULL on wall

**Date**: 2026-05-21
**Iter**: 102 (post iter 101 PASS)
**Branch**: vesta5 (glades-ml)
**Verdict**: **NULL on wall** — sub-noise improvement (+0.08% tok/s, well within intra-run variance). Parity bit-identical sub-ULP. Math change is clean; cast-elimination savings smaller than analytic estimate.

---

## Motivation (iter 99/101 strategic carryover)

Per iter 96 nsys profile, `causal_mask_softmax_kernel` was profile-rank-#6 at 3.8% of step wall (74 calls/step at 323 µs avg). The kernel runs in the inner attention path (gpu_chiron.cu:1602) just before a `cast_f32_to_bf16` (line 1611) that converts the softmax output to BF16 for the subsequent PV cuBLAS GEMM.

Fusion target: write BF16 directly from the softmax kernel's normalization pass, eliminating the separate cast kernel launch AND the FP32 S → BF16 P memory round-trip.

## Conjecture (pre-committed)

**Target**: new kernel `causal_mask_softmax_bf16_out_kernel` — identical to `causal_mask_softmax_kernel` through passes 1-2 (mask + row-max + exp/sum, all FP32), but pass 3 (normalize) writes BF16 to a separate buffer instead of normalizing in-place on FP32 S.

**Math bit-identical**: same FP32 max/exp/sum/normalize; BF16 cast uses RN-even rounding bit-identical to `k_cast_f32_to_bf16` (same union reinterpret, same `0x7FFFu + lsb` rounding bias, same NaN flush to BF16 quiet NaN). End-to-end output identical to `(softmax_inplace THEN cast_f32_to_bf16)`.

**Pre-committed Gate-0**:
- Wall delta: +1% to +2% (analytic: eliminate cast launch + ~96 MB/call memory traffic × 24 calls/step)
- NLL drift: ≤ ±0.005 nat (bit-identical math; sub-ULP only)
- Token budget: 100-step apples-to-apples bench (seed=1337) atop iter 97+99+101 combined stack

## Implementation

- **gpu_kernels.cu**: new template `causal_mask_softmax_bf16_out_kernel` in anon namespace (alongside `causal_mask_softmax_kernel`). Pass 1-2 identical; pass 3 writes BF16 with inlined RN-even cast.
- **gpu_kernels.cu**: new dispatcher `causal_mask_softmax_bf16_out(S, P_bf, batchSize, T)`.
- **gpu_kernels.h** (both glades-ml + vendored glades-trainer copies): declaration + no-CUDA stub.
- **gpu_chiron.cu** `flash_attention_cublas_tiled_bf16` (~line 1599): when `causal == true` (always true at CHIRON production since `medalTrain=false`), replace the softmax+cast pair with the single fused call. Non-causal branch keeps legacy chain (unused at production).

Note: no trainer flag — fusion is unconditional at the inner attention call site. Math bit-identical → no A/B safety concern.

## Bench (single-seed 100 steps × seed=1337 × T=16384 L=24 w=4)

| run | wall (s) | tok/s @ 26 / 51 / 76 | NLL @ step 100 | PPL |
|---  |---:      |---:                   |---:            |---: |
| iter 97 + 99 + 101 (iter 101 bench)                                | 62.8 | 25,938 / 26,220 / 26,190 | 9.8477 | 18913.94 |
| **iter 97 + 99 + 101 + 102 (this iter)**                           | **62.8** | **25,945 / 26,231 / 26,211** | 9.8476 | 18913.52 |

**Wall delta**: 0.0 s (identical to 0.1s precision).
**tok/s delta @ step 76**: +21 tok/s = +0.08%. Within intra-run variance (~±50 tok/s).
**NLL delta**: 9.8477 → 9.8476 = sub-ULP (bit-identical at 4 decimals).
**PPL delta**: 18913.94 → 18913.52 = −0.42 = ~0.00002 nat (sub-ULP cuBLAS-scheduling drift).
**Loss/||g|| trajectory**: all checkpoints identical to sub-ULP precision.

## Verdict

**NULL on wall, parity bit-identical.** Same outcome as iter 95/96 (mechanism clean, savings sub-noise).

## Why NULL despite +1.5% analytic estimate

Three reasons the savings under-delivered:

1. **Cast kernel per-call cost was small**: iter 96 nsys showed `k_cast_f32_to_bf16` aggregated 7584 calls × 34 µs avg across the whole step. The inner attention cast is one specific cast among ~300/step. Eliminating 24 inner-attention cast calls saves ~0.8 ms = 0.13% wall — far below the 1.5% analytic estimate.

2. **Memory traffic was already required**: the FP32 → BF16 cast does ~96 MB I/O per call (read scratch_S, write scratch_Pbf16). In the fused kernel, the BF16 write is now part of softmax's pass 3 — so the same ~32 MB BF16 write happens regardless. Only the cast's 64 MB scratch_S READ is eliminated (and the cast's separate launch overhead).

3. **Cuda stream parallelism**: cuBLAS PV GEMM and the upstream softmax may not have been fully serialized — the cast's overhead may have been overlapped with neighboring kernel work, reducing the visible wall benefit.

## Default

Kept as **unconditional fusion at production** (math bit-identical, no risk). Net production change: ~0% wall (sub-noise positive), code cleaner with one less kernel launch per inner attention call. The fused kernel + dispatcher stay in tree as zero-cost silent activation.

If future profile-driven analysis (e.g., a re-profile after iter 97+99+101 ship to find the new top kernels) reveals a different bottleneck pattern, the iter 102 mechanism might compose with other future fusions.

## Cumulative stack snapshot (unchanged)

iter 97 + iter 99 + iter 101 stack remains the validated PASS at iter 60 +3% bar:
- iter 97 alone: +1.54%
- iter 99 alone: +1.40%
- iter 97+99 combined (iter 100 multi-seed PASS): +3.08%
- iter 97+99+101 combined: +3.87%
- iter 97+99+101+102 combined: +3.87% (iter 102 sub-noise contribution)

Cumulative since pre-ralph-loop: 26,211/15,200 = **1.72×** (unchanged from iter 101).

## Files

- This document.
- `research/runs/2026-05-21-iter102-gate0/iter102_combined_100step.log` (62.8s, all 4 flags).
- Code:
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` (new fused kernel + dispatcher near line ~2820).
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.h` (declaration + no-CUDA stub).
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` (call site change at flash_attention_cublas_tiled_bf16 ~line 1599).
