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

All GEMMs (8 forward + 10 backward) through cuBLAS BF16 tensor cores
(CUBLAS_COMPUTE_32F_FAST_16BF, FP32 accumulate), per-step re-quant,
plus BF16 flash attention forward (Q/K/V read as BF16, softmax in FP32):

| Config | FP32 tok/s | BF16 tok/s | Speedup |
|---|---|---|---|
| pile_small (dModel=512, L=8, seq=1024) | 9,697 | 9,868 | **1.018x** |
| pile_medium (dModel=1024, L=8, seq=1024) | 3,113 | 3,164 | **1.016x** |

Variants that were tested and empirically ruled out as speedup levers
for this shape/hardware combo:

- Cast hoist (x1 shared across Q/K/V): saves 2 casts/layer, 0% end-to-end gain
- BF16 flash attention (halves Q/K/V memory reads): 0% additional gain
- CUBLAS_COMPUTE_32F_FAST_16BF vs CUBLAS_COMPUTE_32F: 0% additional gain

Only ~2% on this workload, not the 1.5-2x BF16-on-tensor-cores suggests.
We systematically eliminated the usual suspects:

- **Cast overhead is NOT the bottleneck**. Hoisting the Q/K/V-shared
  x1 cast (saving 2 casts/layer) produced no measurable gain.
- **Attention memory bandwidth is NOT the bottleneck** at these seq
  lengths. BF16 flash attention (Q/K/V loaded as BF16, halving memory
  traffic through the attention kernel) produced no measurable gain.
- **cuBLAS BF16 compute semantics are not the bottleneck**. Switching
  from CUBLAS_COMPUTE_32F to CUBLAS_COMPUTE_32F_FAST_16BF (the
  explicit BF16-tensor-core path) produced no measurable gain.

The genuine ceiling appears to be that at minibatch=16, seq_len=1024,
dModel <= 1024 on RTX 4080 SUPER (Ada SM 8.9):

- TF32 tensor cores are already fast enough that BF16's 2x advertised
  speedup isn't realized. TF32 peak is 97.5 TFLOPS vs BF16 peak
  195 TFLOPS, but reaching BF16 peak requires larger GEMM shapes.
- Non-matmul kernels (LN, activation functions, elementwise residual,
  softmax) account for a significant fraction of step time. Nothing
  we've done speeds those up.
- Step-time is not memory-bandwidth limited (memory utilization was
  ~6% in nvidia-smi), so reducing memory traffic doesn't help either.
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

### 5. Loss-scaling wiring — NOT NEEDED for BF16

Loss scaling is a remedy for **FP16's 5-bit exponent** underflowing
small gradient magnitudes. BF16 has an 8-bit exponent (identical to
FP32's range ~1e-38..1e+38), so no gradient representable in FP32 can
underflow when cast to BF16. In our implementation all gradients stay
in FP32 anyway (cast happens only on forward/backward activation
inputs into GEMM/attention, not on gradient outputs), so even the
narrow mantissa of BF16 doesn't bite.

The scaffolding is present (`MixedPrecisionConfig::useLossScaling`,
`TensorTransformerState::mpLossScale`, growth / backoff functions on
the CPU path) should an FP16 variant be wired later. For BF16
deployments, this task is intentionally a no-op — the config defaults
`useLossScaling=false` when `weightDType==BF16`.

### BF16 backward flash attention (DONE 2026-04-20)

Parallel to forward BF16 flash attention: Q/K/V are loaded as BF16 in
both pass-1 (runMax/runSum) and pass-2 (dQ/dK/dV). O, dO, dQ, dK, dV
stay FP32 since each is only loaded/written once. Softmax, atomic
adds, accumulations all in FP32. Falls back to FP32 if the
multi-query shmem budget exceeds device opt-in.

Wired into `transformerGpuTrainEpoch` alongside forward BF16
flash attention. Parity at 2.98e-05 relative NLL drift unchanged.

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

## BF16 optimizer state (AdamW, 2026-04-21)

Separately from the BF16 weight/activation mixed-precision path above:
we now support storing the AdamW optimizer state (m, v) in BF16 for the
9 large weight matrices (tokE, WIn, WOut, per-layer Wq/Wk/Wv/Wo/W1/W2),
halving their optimizer-state VRAM. Biases and LN params stay FP32.

**Kernel**: `gpu::adam_update_bf16_state` in `gpu_kernels.cu`. Loads m, v
as uint16_t BF16 bit patterns, upcasts to FP32 for EMA compute, stores
back BF16 with round-to-nearest-even. Weights + grads remain FP32.
Identical math to `adam_update` up to 7-bit mantissa quantization on
the EMAs.

**Test**: `GpuTrainingUnitTest` Test 3 (gpu-training-test.cpp) runs both
kernels side-by-side on a 512-element synthetic problem for 100 steps
with identical gradients. Observed deviation: **L2-relative 0.27%**
between FP32-Adam and BF16-state-Adam weight trajectories (well under
the 2% bound). Per-coordinate max-rel drift 1.9% (filtered to |W|>1e-3
to avoid divisor blow-up at zero crossings).

**Buffers**: BF16 shadow buffers `vTokE_bf16`, `v2TokE_bf16`, etc. added
to `GpuTransformerWeights`. Allocation is mutually exclusive with FP32
m/v — `allocate(..., adamStateBf16=true)` allocates BF16 only.

**Dispatch**: in `transformerGpuTrainEpoch`'s AdamW branch, when
`trainingConfig.mixedPrecision.adamStateBf16` is true, the 9 large
matrices are skipped from the batched Adam and processed per-matrix
via `adam_update_bf16_state`. Biases/LN continue through the batched
FP32 Adam. Zero measurable throughput impact at dModel=1024.

**Measured VRAM savings at dModel=1024, 8 layers**:

| Config | total GPU VRAM used (nvidia-smi) |
|---|---|
| AdamW FP32 state | 3,955 MB |
| AdamW BF16 state | 3,415 MB |
| **savings** | **540 MB** (~4.5% of card; matches theoretical ~467 MB for 9 big matrices × 2 moments × 2 bytes saved per param) |

CLI: `--adam-state-bf16` in glades-trainer.

**Next step for even bigger savings**: 8-bit quantized Adam state
(bitsandbytes-style block-wise scaling) would halve memory again
(~2 GB savings at this scale), at the cost of another quantization
stage. Out of scope for this session.

## Cross-optimizer coverage (2026-04-20)

The BF16 mirror refresh (`ensureLowpMirrors`) runs at the tail of
`transformerGpuTrainEpoch` AFTER all optimizer branches (AdamW, ATLAS,
VESTA, HELIOS), so any optimizer that updates the FP32 master weights
picks up a fresh BF16 mirror for the next forward pass automatically.

| Optimizer | GPU path | BF16-compatible | Parity tests |
|---|---|---|---|
| AdamW | yes | yes | existing |
| ATLAS (+ECHO/BiMAP/PACT/etc.) | yes | yes | existing |
| VESTA | yes | yes | VESTATransformerBf16ParityTest (5.1e-5) |
| HELIOS | yes (2026-04-20) | yes (via master-weight update) | HELIOSGpuParityTest + HELIOSGpuStochasticParityTest (bit-exact / 5.96e-8) |

No new BF16 work was needed to integrate HELIOS — HELIOS mutates the FP32
master weights in-place, identical to AdamW/VESTA, and the existing
post-step refresh rebuilds the BF16 mirrors. See
`research/HELIOS_framework.md` §14a for the HELIOS implementation status.
