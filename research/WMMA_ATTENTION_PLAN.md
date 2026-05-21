# WMMA Tensor-Core Flash Attention — Plan

## Why

Grep-confirmed: `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` has
**no `wmma` / `mma` / `nvcuda::wmma` use** anywhere. The existing
`flash_attention_fwd_multihead_kernel` and `flash_attention_fwd_multiq_kernel`
use `__shfl_down_sync` warp reductions + manual FP32 FMA, which executes
on the CUDA cores rather than the tensor cores.

Measured on RTX 4080 SUPER:
- `flash_attention_multihead_forward` (FP32): **0.14 TFLOP/s**
- `flash_attention_multihead_forward_bf16`: **0.18 TFLOP/s**
- `sgemm_rowmajor` (cuBLAS TF32 tensor cores): **40 TFLOP/s**

Peak FP32 on the card is 52 TFLOP/s; peak BF16 on tensor cores is
~104 TFLOP/s. The current kernel is 2-3 orders of magnitude below peak.

**Writing a WMMA-based flash attention unlocks ~30× speedup for both the
baseline transformer and CHIRON** — the single highest-leverage GPU
optimization in the codebase.

## Approach (two-stage, recommended)

### Stage 1 (minimal win, 1-2 iterations): cuBLAS-tiled flash attention

For each tile of `QROWS=64` query rows:
1. `S_tile[64, T] = Q_tile · K^T / √dH` — `sgemm_rowmajor_abt` with
   TF32 tensor cores. This is a 64×T×dH GEMM; TF32 lights up.
2. Custom row-softmax kernel on `S_tile` with causal mask (memory-bound).
3. `O_tile[64, dH] = P_tile · V` — `sgemm_rowmajor` with TF32.

Memory: `QROWS × T × 4` bytes (64 × 4096 × 4 = 1 MB at max production T).
Fits in L2. Keeps the flash-attention memory invariant (no full T×T
materialization) while letting cuBLAS do the heavy lifting.

API shape (drop-in for existing `flash_attention_multihead_forward`):
```cpp
bool flash_attention_cublas_tiled(const float* Q, const float* K, const float* V,
                                    int T, int nHeads, int nKVHeads,
                                    int dHead, int dModel, int dModelKV,
                                    bool causal, float* O,
                                    float* scratch_Stile);  // [QROWS, T] persistent
```

Expected throughput: 15-30 TFLOP/s (cuBLAS wins for the GEMM portion; softmax
is memory-bound at ~500 GB/s ≈ 2-5 TFLOP/s).

### Stage 2 (full win, 3-5 iterations): real WMMA kernel

Write `flash_attention_wmma_kernel` using `nvcuda::wmma::fragment` for
the Q·K^T and P·V matrix multiplies with BF16 inputs and FP32 accumulators.
Warp-level 16×16×16 fragments; one block per (T_tile_of_queries, nHead).
Shared-memory staging for K/V tiles; online softmax in-kernel.

This is the full-peak path: expected 50+ TFLOP/s (approaching BF16 tensor
core peak of 104 TFLOP/s, bounded by softmax memory traffic).

## Acceptance criteria

- Drop-in replacement of `flash_attention_multihead_forward` produces
  identical (≤ 1e-4 relative) outputs on a range of shapes (nH=1..16,
  dH=64..128, T=512..2048, causal=true/false).
- GPU parity test passes at existing CHIRON tolerance (≤ 1e-3 relative).
- Measured throughput ≥ 10 TFLOP/s (Stage 1) or ≥ 40 TFLOP/s (Stage 2),
  vs current 0.15 TFLOP/s.

## Why now

CHIRON's per-block wall time is bottlenecked on `flash_attention_forward`.
Fixing this kernel:
- **Direct 30× speedup to CHIRON training wall time**
- **Same 30× speedup to the baseline transformer path** — anyone who
  uses glades-ml's attention benefits.
- Independent of CHIRON correctness — the kernel contract is unchanged.

## Deferred

- Backward pass: `flash_attention_multihead_backward` also lacks tensor
  cores. Stage-1 cuBLAS port applies there too (GEMMs in backward are
  the dominant cost).
- The BF16 flash-attention variant (`_bf16`) has the same issue; porting
  it to tensor cores is part of Stage 2.
