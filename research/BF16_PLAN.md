# BF16 Mixed-Precision GPU Training — Rollout Plan

## Current state (2026-04-20)

**Shipped (committed)**:
- Host-side cast primitives (`float_to_bf16_rn`, `bf16_to_float`) and
  host BF16 mirror buffers (`tokELowp`, `W*Lowp`, etc. in
  `TensorTransformerState`). Pre-existing from Phase A.
- CPU forward path uses host BF16 mirrors for embeddings + I/O projections
  and re-quantizes after every minibatch step (`sgd_transformer.cpp:6763`).
- Device cast kernels `gpu::cast_f32_to_bf16` / `cast_bf16_to_f32` with
  round-to-nearest-even, tested round-trip.
- Device BF16 weight mirrors in `GpuTransformerWeights` (one
  `GpuBuffer<uint16_t>` per major weight matrix: `tokELowp`, `WInLowp`,
  `WOutLowp`, per-block `Wq/Wk/Wv/Wo/W1/W2 Lowp`).
- `GpuTransformerWeights::ensureLowpMirrors()` allocates every mirror on
  first call and populates each from its FP32 master via the cast kernel.
- Post-optimizer-step re-quant hook in `transformerGpuTrainEpoch` (end of
  every minibatch: `ensureLowpMirrors` runs when `cfg.mpEnable`).
- cuBLAS BF16 GEMM wrappers: `sgemm_rowmajor_bf16`,
  `sgemm_rowmajor_atb_bf16`, `sgemm_rowmajor_abt_bf16` — backed by
  `cublasGemmEx` with `CUDA_R_16BF` inputs, `CUBLAS_COMPUTE_32F` accumulate,
  `CUBLAS_GEMM_DEFAULT_TENSOR_OP` algo. Correctness verified:
  Frobenius relative error 0.23% vs FP32 for plain/ATB/ABT variants.
- BF16 activation scratch buffers (`activationLowp`, `activationLowp2`) in
  `GpuTransformerScratch`, sized to T*max(dModel, ff1Width, vocab, input).
- Dispatch helpers `gpu_gemm_mp`, `gpu_gemm_atb_mp`, `gpu_gemm_abt_mp` at
  top of the GPU training path. When `useBf16` is true they cast the FP32
  activation into the provided BF16 scratch, then call the BF16 GEMM. When
  false they fall through to the FP32 cuBLAS variant.
- All 8 forward-path GEMM call sites in `transformerGpuTrainEpoch` rewritten
  to use the dispatch helpers (WIn, Wq, Wk, Wv, Wo, W1, W2, tied head).
- Parity smoke test `VESTATransformerBf16ParityTest`: trains a tiny
  transformer twice (FP32 vs `mp.enable=true`), compares final NLL.

**Per-site BF16 debug harness**:
- `useBf16` now reads `cfg.mpEnable && weightDType==BF16`.
- Site gating via env var `GLADES_BF16_SITES` (bitmask, default 0):
  - `0x01` WIn, `0x02` Wq, `0x04` Wk, `0x08` Wv
  - `0x10` Wo, `0x20` W1, `0x40` W2, `0x80` tied LM head
- All sites default to the FP32 fallback even when `mp.enable=true` until
  the NaN debug below lands — flipping bits in the mask opts individual
  sites into the BF16 path for isolation tests.
- Reproducing the NaN:
  `GLADES_BF16_SITES=0x80 ./glades-unit-tests vesta` triggers
  "Trainer::run: non-finite training aggregates detected" on
  `VESTATransformerBf16ParityTest` within the first minibatch. Every
  single-site mask (`0x01`..`0x80`) reproduces the same failure
  equally, which rules out a site-specific bug and points at a common
  code path (cast kernel, scratch aliasing, or cuBLAS BF16 tensor-op
  interaction at small shapes).

**Outstanding diagnostic** (see Remaining Work #1):
- The unit test `VESTAGpuBf16GemmTest` proves the BF16 GEMM wrapper is
  numerically correct on Gaussian-random matrices (0.23% Frobenius error).
- The forward-path integration introduces NaN somewhere that the unit
  test doesn't exercise — likely a stream-order, LN/softmax denormal
  interaction, or cuBLAS tensor-op behavior at dModel=64/vocab=31 (the
  parity test's tiny shape).

## Remaining work

### 1. Debug BF16 forward NaN (~4-6h)

Starting diagnostics to run:

- Download the first BF16 Lowp mirror immediately after `ensureLowpMirrors`,
  inspect first few elements, verify finite and within expected range.
- At each BF16 GEMM site in the first step, download a few output elements
  and verify finite. Identify the first site that goes bad.
- Compare FP32 vs BF16 logits at step 0 (before backward); measure
  element-wise max absolute difference. Should be O(1e-2 relative).
- Examine the GQA attention (`flash_attention_multihead_forward`) input:
  if Q/K/V computed via BF16 have tiny-value outliers, the scaled dot
  products could underflow or produce Inf when divided by `sqrt(dHead)` at
  very small dHead.
- Check whether the cast kernel truncates vs rounds on denormals (could
  turn tiny FP32 into BF16 zero, causing downstream 0/0).

Likely fixes depending on root cause:
- Guarantee bias application happens in FP32 (should already — biases are
  not quantized; double-check `gpu::add_bias` isn't fed BF16).
- Insert epsilon clamps in softmax/LN if denormal underflow is the issue.
- Verify `activationLowp` buffer is strictly larger than any single
  activation tile used (for SwiGLU, ff1Width = 2*dFF).

### 2. Wire the BF16 backward weight-grad path (~3h)

The forward uses weight BF16 mirrors. The backward weight-grad pattern is
`gW = dY^T * X` — both dY and X are FP32 activations. With BF16 we'd:
- Cast dY into `activationLowp`.
- Cast X into `activationLowp2`.
- Call `gpu_gemm_atb_mp(true, ...)` which fires `sgemm_rowmajor_atb_bf16`.

Call sites: roughly 10 in `transformerGpuTrainEpoch` (one per weight matrix
whose gradient is accumulated). See lines ~9613, ~9659, ~9733, ~9761,
~9798, ~9848, ~9909, ~9929, ~9949, ~10005 (post-commit line numbers —
rely on `grep sgemm_rowmajor_atb` in `sgd_transformer.cpp` to re-locate).

### 3. Wire the BF16 input-grad path (~2h)

Input-grad pattern `dX = dY * W` — dY FP32, W has BF16 mirror. Use
`gpu_gemm_mp(useBf16, ...)` which casts dY and uses `W*Lowp.data()`.

Call sites: roughly 10 pairs with the weight-grad sites above, using
`sgemm_rowmajor` (plain). Grep `gpu::sgemm_rowmajor(` in the backward
section.

### 4. Loss-scaling wiring (~2h)

`MixedPrecisionConfig::useLossScaling` + `mpLossScale` are plumbed through
`TransformerEpochCfg` but not currently applied in the GPU backward.
BF16 doesn't strictly need loss scaling (FP32 exponent range) but the
code is scaffolded for FP16 compat. Multiply `dLogits` by `mpLossScale`
before backward; divide gradients by `mpLossScale` before the optimizer
step. Check `grads_all_finite` to back off on overflow (already
implemented for the CPU path).

### 5. glades-trainer CLI plumbing (~1h)

Add to `trainer/main.cpp`:
- `--mp` bool flag -> `cfg.mixedPrecision.enable = true`
- `--mp-dtype bf16|fp16|fp32` -> `cfg.mixedPrecision.weightDType`
- `--mp-loss-scaling` -> `cfg.mixedPrecision.useLossScaling`

Add to `run.sh`:
- `--mp` switch that sets `MP=1`
- Env vars `MP_DTYPE`, `MP_LOSS_SCALING`

Sync `training_config.h` into glades-trainer's vendored header if the
struct layout changed (should not have — these fields are pre-existing).

### 6. Parity tests + throughput benchmark (~3h)

Tighten `VESTATransformerBf16ParityTest` to (a) exercise the live BF16
path (after item 1 is fixed), (b) assert relative NLL within ~5% after
a short training run, (c) compare across AdamW, VESTA, HELIOS.

Add `VESTATransformerBf16ThroughputBench` that measures tok/sec for
FP32 vs BF16 on a pile_small-shape transformer. Expected speedup on
RTX 4080 SUPER / Ada SM 8.9: 1.5-2x on GEMM-heavy paths, ~1.3-1.5x
end-to-end given non-GEMM overhead (attention softmax, LN, etc.).

## Total remaining effort

- Debug forward NaN: 4-6h
- Backward weight-grad: 3h
- Backward input-grad: 2h
- Loss scaling: 2h
- CLI + trainer: 1h
- Tests + bench: 3h

**Total: 15-17 hours** of focused work across multiple sessions.

Critical path: items 1-3 unlock the real speedup; items 4-6 are
production polish.
