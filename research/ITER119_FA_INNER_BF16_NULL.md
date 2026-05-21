## Iter 119 — FA-fused SCFA inner attention (BF16 inputs, FP32 compute) — WALL NULL

**Date**: 2026-05-21
**Iter**: 119 (continuation of iter 118 FA-fused arc)
**Branch**: vesta5
**Verdict**: **Math PASS** (bit-identical to iter 118 and iter 116 ship). **Wall NULL** — BF16 input loading alone is insufficient; the bottleneck is the FP32 compute path (no tensor cores). To beat cuBLAS BF16-TC we MUST use BF16 MMA tensor core instructions.

---

## What iter 119 attempted

Replace iter 118's FP32-input FA kernel with the BF16-input variant (`flash_attention_multihead_forward_bf16`). The hypothesis: halving input memory bandwidth via BF16 loads might compensate for the lost tensor core speedup vs the cuBLAS pipeline.

Both kernels still do **FP32 compute** internally — only the global memory loads change (BF16 → cast-to-FP32 inline vs FP32 direct load).

## Bench (single-seed × 100 steps × seed=1337 × T=16384 L=24 w=4)

| variant | input | compute | wall | NLL | wall vs iter 116 |
|---|---|---|---:|---:|---:|
| iter 116 ship (baseline) | FP32 → cast to BF16 → cuBLAS BF16-TC | BF16 TC | ~58.07s | 9.8472 | — |
| iter 118 (FP32 FA) | FP32 direct | FP32 SIMT | 159.9s | 9.8472 | **+175% slower** |
| **iter 119 (BF16 FA)** | **BF16 direct** | **FP32 SIMT** | **158.7s** | **9.8472** | **+173% slower** |

iter 118 vs iter 119: −0.7% wall (essentially noise). BF16 input loading delivered no measurable improvement.

## Why iter 119 is NULL

Bottleneck analysis at SCFA inner attention dims (T=1024, dH=256, nH=16, batch=16 from L=24 across layers):

**Per call inner attention compute** (forward only):
- QK^T: 16 (batch) × 1024 × 1024 × 256 = 4.3 GFLOPS
- PV:   16 × 1024 × 256 × 1024 = 4.3 GFLOPS
- Total: 8.6 GFLOPS per call × ~30 calls/step (fwd+bwd recompute) = ~260 GFLOPS/step in inner attention matmul

**Memory bandwidth per call**:
- Q + K + V load: 3 × 1024 × 4096 × 2 bytes (BF16) = 24 MB
- P materialize (cuBLAS path): 16 × 1024 × 1024 × 2 = 32 MB write+read = 64 MB
- O write: 1024 × 4096 × 4 = 16 MB
- Total per call: ~100 MB

On Ada SUPER:
- BF16 TC peak: ~78 TFLOPS → 260 GFLOPS / 78 = 3.3 ms compute (TC-bound)
- FP32 SIMT: ~21 TFLOPS → 260 GFLOPS / 21 = 12.4 ms compute (SIMT-bound)
- Memory: 100 MB / 700 GB/s = 0.14 ms

**The FA kernels are compute-bound, not memory-bound**. Halving memory bandwidth via BF16 loads (saves ~0.07 ms / call) is dwarfed by the 9 ms gap between BF16-TC cuBLAS and FP32-SIMT FA.

To beat cuBLAS, the FA kernel MUST use BF16 tensor core MMA instructions (`wmma::mma_sync` or inline `mma.sync.m16n8k16` PTX).

## Implementation cost estimate for true MMA

A WMMA-based FA-2 kernel requires:
- `wmma::fragment` declarations with m16n16k16 BF16 shape
- Tiled accumulation across dHead=256 (16 wmma calls per output tile)
- Smem layout with bank-conflict-free swizzling for K/V tiles
- Online softmax with FP32 accumulators (cross-warp shuffles)
- Causal masking
- Forward + backward kernels

Realistic effort: **300-500 LOC, 1-2 days of careful CUDA**. Risk of not actually beating cuBLAS is moderate — cuBLAS is well-tuned for these dims (uses `ampere_s1688gemm_bf16_128x128`).

## Alternative paths

Given iter 119's null result and iter 120's effort estimate:

**Option A: Commit to WMMA-based FA kernel (iter 120-122)**
- High effort, moderate risk, +5-8% target wall
- Multi-iter arc: iter 120 fwd kernel, iter 121 bwd, iter 122 ship
- Best case: +8% wall over iter 116 ship → 30,000+ tok/s, ~1.97× cumulative

**Option B: Pivot to CUDA Graph capture**
- Lower effort (no math change), lower risk
- Predicted +2-3% wall from eliminating per-launch overhead
- Single arc: iter 120 graph capture, iter 121 validation
- Realistic target: ~29,000 tok/s, ~1.91× cumulative

**Option C: Stop at iter 116 ship**
- Current state is production-ready at +12.56% over iter 94 ship
- Cumulative 1.86×; 2.0× target ~93% reached
- Future improvements require multi-day investments

## Recommendation

The session's iter 113 lesson says "structurally different approaches can break past ceilings" — but iter 119 confirms BF16 inputs alone won't. The next real wall lever requires either MMA or graph capture.

My recommendation: **Option B (CUDA Graph capture)** as the iter 120 target. Reasoning:
1. **Lower-risk**: no math change, pure dispatch optimization
2. **Well-bounded**: predicted +2-3% is concrete (iter 117 estimate from launch overhead)
3. **Faster path to 2.0×**: ~1-2 iters vs WMMA's 2-3
4. **Reversible**: easy to disable per-call via flag

WMMA-based FA can come AFTER as a separate arc once Graph capture establishes the baseline.

## Default and production state

iter 118 + iter 119 flags both default OFF. Production stays on iter 116 ship + iter 107/115 unconditional library + iter 116 ship default flips.

iter 118 kernel (FP32 FA) is retained as a math-validation reference. iter 119 BF16 path adds the BF16-input alternative for testing.

## Files

- This document.
- Code:
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu`: added `g_iter119_fa_inner_bf16` toggle + BF16 cast + dispatch
  - Header decl + no-CUDA stub in glades-ml and vendored mirror
  - Trainer: `iter119FaInnerBf16` flag + CLI + init dispatch

No new kernel code (re-used existing `flash_attention_multihead_forward_bf16`).
