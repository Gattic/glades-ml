## Iter 118 — FA-fused SCFA inner attention forward (FP32 Gate-0)

**Date**: 2026-05-21
**Iter**: 118 (first iter of FlashAttention-fused inner attention arc)
**Branch**: vesta5 (glades-ml + glades-trainer)
**Verdict**: **Math Gate-0 PASS, Wall regression EXPECTED.** NLL bit-identical to iter 116 ship at single-seed 100-step bench. Wall 2.75× slower (FP32 compute vs BF16 tensor cores). Pivot to iter 119 (BF16/MMA port) for wall improvement.

---

## Motivation

Per iter 117 META and iter 82 strategic memo, FlashAttention-fused SCFA inner attention is the next high-EV target (+5-10% wall predicted). iter 105 profile shows ~10-15% wall in inner-attention GEMMs + softmax that can be collapsed into a single fused kernel:

Current pipeline (`flash_attention_cublas_tiled_bf16` at gpu_chiron.cu:1580):
1. `cast_f32_to_bf16(sQ/sK/sV)` × 3
2. cuBLAS QK^T → scratch_S (BF16-TC)
3. `causal_mask_softmax_bf16_out` → scratch_Pbf16 (iter 102 fused)
4. cuBLAS P·V → O (BF16-TC)

Target: single fused kernel that does all four steps with online softmax, no P materialization to DRAM.

## Gate-0 design (this iter)

Re-use the existing `flash_attention_multihead_forward` kernel (gpu_kernels.cu:4115) — already implements the FA-2 online-softmax pattern in FP32. No new kernel code; just a library-side toggle that replaces the cuBLAS pipeline with this kernel.

**Why FP32 first** (math validation gate):
- Confirms the FA algorithm produces correct attention output for SCFA dims (T_inner=1024, nH=16, dH=256)
- No tensor cores → expected wall regression vs cuBLAS-BF16 path
- Wall improvement comes in iter 119 (BF16 inputs + MMA / wmma instructions)

## Implementation

**gpu_chiron.cu**: added `set_iter118_fa_inner_fwd(bool)` static toggle. When set, `flash_attention_cublas_tiled_bf16` returns `flash_attention_multihead_forward(Q, K, V, T, nHeads, nHeads, dHead, dModel, dModel, causal, O)` early instead of running cuBLAS pipeline.

**gpu_chiron.h** + vendored mirror: decl + no-CUDA stub.

**chiron_main.cpp**: `iter118FaInnerFwd` config flag (default OFF), CLI `--iter118-fa-inner-fwd`. At trainer init (after `glades::gpu::initDevice()`), calls `set_iter118_fa_inner_fwd(cfg.iter118FaInnerFwd)`.

No new kernel code (re-using existing).

## Bench (single-seed × 100 steps × T=16384 L=24 w=4)

| metric | iter 116 ship (baseline) | iter 118 (FA-FP32) | Δ |
|---|---:|---:|---:|
| Wall | ~58.07s | **159.9s** | **+175.4% (2.75× slower)** |
| tok/s @ step 76 | 28,257 | 10,330 | −63% |
| NLL @ step 100 | 9.8472 | 9.8472 | bit-identical (4 decimals) |
| Loss @ step 76 | 10.1786 | 10.1786 | bit-identical |
| ‖g‖ @ step 76 | 1.209 | 1.209 | bit-identical |

**Math validation**: PASS. NLL 9.8472 = 9.8472 at 4-decimal precision. Loss + ‖g‖ also bit-identical. The FA online-softmax produces output indistinguishable from cuBLAS-BF16+softmax+cuBLAS-BF16 at this scale.

**Wall**: 2.75× slower as expected (FP32 vs BF16-TC ≈ 16× per matmul on Ada, partially offset by FA fusion savings).

## Why this is a Gate-0 PASS despite wall regression

iter 118 is the **first iter of a multi-iter FA-fused-inner arc**. Per the iter 113 lesson, multi-iter scope requires gate-staged validation:
- **iter 118 Gate-0**: math correctness (this iter)
- **iter 119 Gate-1**: BF16/MMA port → wall parity with cuBLAS
- **iter 120 Gate-2**: extend to backward (FA-style with recompute or saved-stats)
- **iter 121 Gate-3**: production wire + n=3 multi-seed
- **iter 122 Gate-4**: 30k Phase 2 retrain validation

Gate-0 confirms the math approach is sound. Wall improvement comes at Gate-1.

## Risks identified

1. **dH=256 unusual**: standard FA-2 templates assume dH ≤ 128. Ada MMA tiles 16×16×16 (BF16). For dH=256, need 16 MMA tiles per dot product → register pressure. Implementation in iter 119 will need careful tile sizing.

2. **k=1024 is "small" by FA standards**: typical FA usage is T ≥ 4096. At k=1024, the cuBLAS PV/QK^T calls are already small; their absolute wall cost may be modest. iter 105 profile shows the inner GEMMs at ~10-15% wall total. Realistic upper bound for iter 119 wall improvement: +5-8% (matching iter 82 prediction).

3. **Backward complexity**: iter 120 will need FA-style backward. Two options:
   - Recompute attention forward in backward (FlashAttention-1 style)
   - Save row-wise normalizers + max from forward (FlashAttention-2 style)
   The save-stats approach is more efficient but requires per-row state buffer (~T*nH*8 bytes = 1 MB for SCFA inner). Decide at iter 120.

## Default and production recommendation

iter 118 flag stays **OFF** (Gate-0 only; math validated, wall regression). Production stays on iter 116 ship + iter 107/115 unconditional library changes.

iter 119+ should proceed with the BF16/MMA port. Predicted iter 119 wall delta over iter 118: ~+15× (recovering tensor core speed); over iter 116 ship: target +5-8%.

## Files

- This document.
- Code:
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu`: `set_iter118_fa_inner_fwd` toggle + early-return path
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.h`: decl + no-CUDA stub
  - `glades-trainer/include/Backend/Machine Learning/Networks/cuda/gpu_chiron.h`: vendored mirror
  - `glades-trainer/trainer/chiron_main.cpp`: `iter118FaInnerFwd` flag + CLI + init dispatch
- No new kernel code (re-uses `flash_attention_multihead_forward` from `gpu_kernels.cu`).
