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

**Forward-path NaN bug — fixed (2026-04-20)**:

Root cause: `ensureLowpMirrors()` used `master.allocated()` (which returns
a **bool**, not an element count) in place of `master.size()`. This
allocated every BF16 mirror with `allocate(1)` (from the bool→size_t
conversion), populated a single element, and left the rest of the mirror
reading whatever `cudaMalloc` happened to hand back (often zeros). The
first BF16 forward GEMM then multiplied activations by a near-all-zeros
weight tensor, producing garbage logits. Softmax + cross-entropy on the
garbage logits fed the backward with huge gradients that overflowed
within one step.

Fix: single character — `.allocated()` → `.size()` in the
`GLADES_LOWP_ENSURE` macro (`gpu_transformer_state.cu`).

Result: `VESTATransformerBf16ParityTest` with all 8 forward GEMMs in
BF16 now matches FP32 within 5.1e-5 relative error on a 1-epoch tiny
training run. Default gate is flipped back on (BF16 forward active
whenever `mp.enable=true && weightDType==BF16`); `GLADES_BF16_SITES`
env var is kept as a per-site override for debugging.

## Measured throughput (RTX 4080 SUPER, 2026-04-20)

All 18 GEMM sites (8 forward + 10 backward) through cuBLAS BF16
tensor cores, FP32 accumulate, per-step re-quant:

| Config | FP32 tok/s | BF16 tok/s | Speedup |
|---|---|---|---|
| pile_small (dModel=512, L=8, seq=1024) | 9,660 | 9,906 | **1.025x** |
| pile_medium (dModel=1024, L=8, seq=1024) | 3,116 | 3,186 | **1.022x** |

Only ~2-3% on this workload, not the 1.5-2x BF16-on-tensor-cores
advertises. Reasons and next-step levers:

- **Cast overhead dominates the GEMM savings.** With 18 casts per
  minibatch (one per activation input), the extra kernel launches
  offset the tensor-core wins at modest GEMM sizes. Fix: keep
  activations in BF16 between kernels — fuse LN+cast, activation+cast,
  attention-in-BF16 — so casts only happen at the FP32 master-weight
  boundaries (optimizer / grads / loss).
- **Attention is not GEMM-bound.** Flash-attention softmax / T² reads
  are compute-and-bandwidth-heavy at seq=1024 but stay FP32 today.
  Moving Q·K^T and attention·V through BF16 (separate kernel rewrite)
  would help proportionally.
- **Small batch saturation.** At minibatch=16, dModel<2048, tensor
  cores are not fully saturated so the FP32 TF32 path is already
  near-optimal for these shapes.

## Remaining work

### 1. Debug BF16 forward NaN (RESOLVED 2026-04-20)

Root cause: `GLADES_LOWP_ENSURE` macro used
`(master).allocated()` (which returns a `bool`) in place of
`.size()` (element count). Mirrors were therefore allocated with
element count 1, one element populated from master, rest of the
buffer reading whatever `cudaMalloc` returned (often zeros). First
BF16 GEMM multiplied activations against near-all-zero weight
tensors → garbage logits → softmax+CE feeds backward with huge
one-hot gradients → optimizer step overflows → NaN in next minibatch.

Fix: single-character edit in `GpuTransformerWeights::ensureLowpMirrors`
in `gpu_transformer_state.cu`. Parity immediately dropped from
training-blowup to 5.1e-5 relative NLL on 1-epoch tiny train.

### 2. Next perf lever: activation BF16 persistence (~6-8h)

To make BF16 actually bite on this workload, activations must stay in
BF16 between kernels instead of roundtripping through FP32 each time.
Concretely:

- Fuse cast-into-BF16 into the tail of `rmsnorm_forward` /
  `layernorm_forward` so the output of LN is BF16-native.
- Fuse cast-into-BF16 into the tail of activation kernels (`relu_forward`,
  `gelu_forward`, `swiglu_forward`).
- Rewrite flash attention to accept BF16 Q/K/V (keeping softmax in FP32
  reductions) and emit BF16 attnConcat — this alone is the biggest single
  win since attention dominates at long seq.
- Keep master weights, gradients, Adam moments, and optimizer step in
  FP32.

Estimated effort: 6-8h across the 3 fused-cast kernels plus a BF16 flash
attention variant.

### 3. BF16 backward weight-grad path (DONE 2026-04-20)

All 10 backward weight-grad sites plus the lone `gWIn` site are wired
through `gpu_gemm_atb_mp`. Weight gradients remain FP32; activation
inputs are cast into `activationLowp` / `activationLowp2` on the fly.
Regression: parity unchanged at 5.1e-5 after adding backward wiring.

### 4. BF16 input-grad path (DONE 2026-04-20)

All 10 backward input-grad sites are wired through `gpu_gemm_mp`. Weight
mirrors supply the BF16 operand.

### 5. Loss-scaling wiring (~2h)

`MixedPrecisionConfig::useLossScaling` + `mpLossScale` are plumbed through
`TransformerEpochCfg` but not currently applied in the GPU backward.
BF16 doesn't strictly need loss scaling (FP32 exponent range) but the
code is scaffolded for FP16 compat. Multiply `dLogits` by `mpLossScale`
before backward; divide gradients by `mpLossScale` before the optimizer
step. Check `grads_all_finite` to back off on overflow (already
implemented for the CPU path).

### 6. glades-trainer CLI plumbing (DONE 2026-04-20)

`--mp`, `--mp-dtype bf16|fp16|fp32`, `--mp-loss-scaling` all plumbed
through trainer/main.cpp and run.sh env vars. Smoke-tested on pile_small.

### 7. Parity tests (DONE 2026-04-20) + throughput benchmark (DONE 2026-04-20)

`VESTATransformerBf16ParityTest` exercises the live BF16 path with
tolerance tightened to 0.2% relative (observed 5.1e-5 on 1-epoch tiny
run).

Throughput measured — see "Measured throughput" section above.

## Remaining effort

- Activation BF16 persistence (next perf lever): 6-8h
- Loss scaling: 2h (optional for BF16; needed for FP16 path)
- BF16 flash attention rewrite: 4-6h (largest single speedup)

Current state is production-ready for correctness; remaining items
unlock additional throughput.
