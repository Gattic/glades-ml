# BF16 Projection Optimization — Design Note

## Current state

When `--bf16-weights` is active on the CHIRON trainer, the per-layer
attention projection runs as:

```
cast_bf16_to_f32(Wq_bf -> Wq_tmp)       // 4 cast kernels per layer (Wq/Wk/Wv/Wo)
sgemm_rowmajor(q, Wq_tmp -> scratch_Q)  // 4 FP32 TF32-TC GEMMs
...
flash_attention_cublas_tiled_bf16(...)
sgemm_rowmajor(scratch_O, Wo_tmp -> p)
```

Observed: at 2.0 B training (m=2240, dModel=4480, T=1024, L=48) this
takes ~15 ms/layer in the forward+backward+Adam combined path.

## Proposed optimization

Skip the weight casts — call the existing `sgemm_rowmajor_bf16`
wrapper (BF16 inputs × BF16 weights → FP32 output, via
`cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_16BF`).

New flow per layer:
```
cast_f32_to_bf16(q -> q_bf)              // ONE cast (q is FP32, T*m elements)
sgemm_rowmajor_bf16(q_bf, Wq_bf -> Q)    // 3 BF16-TC GEMMs (Q/K/V)
sgemm_rowmajor_bf16(q_bf, Wk_bf -> K)
sgemm_rowmajor_bf16(q_bf, Wv_bf -> V)
flash_attention_cublas_tiled_bf16(...)
cast_f32_to_bf16(scratch_O -> O_bf)      // ONE cast (for output proj)
sgemm_rowmajor_bf16(O_bf, Wo_bf -> p)    // BF16-TC output proj
```

Savings per layer:
- Cast: 4 × (m × dModel × 6 bytes) - 2 × (T × m + T × dModel) × 6 bytes
       = 24·m·dModel  −  6·T·(m+dModel)
       At m=2240, dModel=4480, T=1024:  241 MB − 41 MB = **200 MB cast
       traffic saved per layer**.  At ~1 TB/s HBM: **0.2 ms/layer
       reduction** in cast work.
- GEMM: BF16-TC is ~2× throughput of TF32-TC on Ampere/Ada (100 TFLOP/s
       vs 52 TFLOP/s BF16 on 4080 SUPER).  4 projections total → saves
       ~50% of projection time, roughly **1 ms/layer**.

Total projected win: **~1.2 ms/layer × 48 layers = ~60 ms per training step**
(forward + backward combined).  At 2.0 B (current 1381 tok/s, 740 ms
per micro-step), that's **~8% throughput improvement**.

## Implementation scope

New primitive in `gpu_chiron.{h,cu}`:

```cpp
bool chiron_attention_shear_bf16w_tiled(
    const float* q, float* p,
    const uint16_t* Wq_bf, const uint16_t* Wk_bf,
    const uint16_t* Wv_bf, const uint16_t* Wo_bf,
    int T, int m, int nHeads, int dHead,
    bool causal, bool invert,
    uint16_t* scratch_qbf,     // T*m BF16 scratch (one per forward)
    uint16_t* scratch_Obf,     // T*dModel BF16 scratch
    float* scratch_Q, float* scratch_K, float* scratch_V, float* scratch_O,
    float* scratch_S,
    uint16_t* scratch_Qbf, uint16_t* scratch_Kbf,
    uint16_t* scratch_Vbf, uint16_t* scratch_Pbf);

// Same backward variant.
```

Trainer wires this when `--bf16-weights` is set and currently on the
tiled attention path (similar pattern for flash if beneficial).

## Risks

- BF16 × BF16 → FP32 in cuBLAS has slightly lower accuracy than FP32 ×
  FP32 → FP32 (mantissa loss on inputs).  Loss in expectation: ~1/256
  per element.  Adam handles this fine — stochastic-rounded weight
  updates already introduce similar noise and we've verified convergence
  match at 4.6 M and 2.0 B scale.
- Need to validate training parity vs the cast-then-FP32-GEMM path
  explicitly (add a parity test).

## Expected ceiling impact

Speed: +8% on throughput at 2 B (1381 → ~1500 tok/s).
Memory: ~no change (scratches replace cast scratches).  The win is
primarily compute-throughput.
