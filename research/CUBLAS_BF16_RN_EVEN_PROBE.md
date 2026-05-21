# cuBLAS BF16-output rounding probe — iter 61 regression investigation

**Date:** 2026-05-17
**Hardware:** RTX 4080 SUPER (Ada, sm_89)
**Software:** CUDA 12.0 (toolkit) / driver 13.0.0 / cuBLAS 12.0.2
**Probe source:** `glades-trainer/research/cublas_bf16_rounding_probe.cu`
**Raw output:** `glades-trainer/research/cublas_bf16_rounding_probe.out`

---

## TL;DR

**cuBLAS BF16-output IS round-to-nearest-even, indistinguishable from the legacy
explicit cast kernel — across all 18 algo enum values, at the production shapes,
in beta=0 and beta=1 modes, with bit-deterministic reps.**

The iter 61 path produces output that **bit-matches** the legacy path (cuBLAS
FP32-out + `k_bf16_accum_axpy` RN-EVEN cast) on every one of 8.4 M output
elements, at both `(M=2048, N=4096, K=8192)` (dWq/dWk/dWv shape) and
`(M=4096, N=2048, K=8192)` (dWo shape). Therefore **cuBLAS BF16-out rounding
is NOT the cause of the +0.50 nat training regression** between the
2026-05-14 flagship and the post-iter61 binary.

The iter 61 mechanism (BF16-grad direct cuBLAS output) is **mathematically
bit-equivalent** to its predecessor, and the search for the regression must
move to other suspects.

---

## 1. cuBLAS rounding semantics — what NVIDIA documents

### What is documented

The cuBLAS Library reference (13.2, April 2026, also 12.x docs) describes the
compute-type matrix for `cublasGemmEx`:

* `CUBLAS_COMPUTE_32F_FAST_16BF` — "Allows the library to use Tensor Cores
  with automatic down-conversion and bfloat16 compute for 32-bit input and
  output matrices."

The cuBLAS docs explicitly state that **input conversions for `FAST_TF32`
round to nearest even** — but the analogous statement for `FAST_16BF` is
not made in the introduction prose. The store-side rounding mode when D is
`CUDA_R_16BF` is left **implementation-defined** in writing.

### What the PTX ISA specifies (the underlying instruction)

NVIDIA's PTX ISA manual documents the `cvt.bf16.f32` instruction family.
The `.rn` modifier corresponds to IEEE 754 **round-to-nearest-even**
(ties-to-even, the unbiased default). On sm_80+ hardware, the CUDA runtime
intrinsic `__float2bfloat16(float)` lowers to `cvt.rn.bf16.f32` per the
NVIDIA CUDA math API documentation.

Since `cublasGemmEx`'s tensor-core epilogue is implemented on top of the
same PTX instruction family, **the well-formed expectation is that
BF16 store rounding is RN-EVEN** — but it is not contractually documented.

### Reported determinism caveats from the cuBLAS docs

cuBLAS documents that GEMM algorithms can split along K and that the
intermediate splits "are summed deterministically into the resulting matrix"
when this happens. Determinism across `cublasGemmAlgo_t` *selection* is not
guaranteed — different algos have different reduction orders, but each fixed
algo is deterministic across calls (with the same handle and stream).

The docs also warn that for `cublasGemmEx`, when the compute type is greater
than the output type, "the sum of split chunks can potentially lead to
intermediate overflows" — this is a different concern (range, not rounding
bias).

### Literature on BF16 training instability

Multiple recent papers (Defeating Training-Inference Mismatch via FP16
[arXiv:2510.26788], Why Low-Precision Transformer Training Fails via Flash
Attention [arXiv:2510.04212]) attribute BF16 training instability to
**accumulation of biased rounding error from the small mantissa** combined
with **distribution-specific patterns** (e.g. identical softmax maxima
forcing pathological RZ-like behavior in Flash Attention BF16 kernels).
These reports are about **how many operations** the BF16 path goes through,
not about the cuBLAS BF16-store cast direction itself.

The probe below tests the latter directly.

---

## 2. Probe code + execution

`cublas_bf16_rounding_probe.cu` (~340 lines, single file, nvcc-only).

Build:

```
nvcc -O2 -std=c++14 cublas_bf16_rounding_probe.cu \
     -lcublas -lcudart -gencode arch=compute_89,code=sm_89 \
     -o cublas_bf16_rounding_probe
```

Method per shape (`(M=2048, N=4096, K=8192)` and `(M=4096, N=2048, K=8192)`):

1. Generate deterministic uniform-random A (row-major `[K, M]`) and B
   (row-major `[K, N]`) FP32 with scale `sqrt(3/K)` so each dot-product
   has unit variance (no overflow, finite throughout the output).
2. RN-EVEN cast A, B once to BF16. These are the fixed inputs.
3. **Legacy path:** `cublasGemmEx` with D=`CUDA_R_32F`,
   `CUBLAS_COMPUTE_32F_FAST_16BF`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`,
   followed by explicit `k_cast_f32_to_bf16_rneven` kernel implementing
   `bias = 0x7FFF + lsb` round-to-nearest-even. (This is exactly
   `k_bf16_accum_axpy` from `gpu_kernels.cu` with α=1, β=0.)
4. **Iter 61 path:** `cublasGemmEx` with D=`CUDA_R_16BF`,
   `CUBLAS_COMPUTE_32F_FAST_16BF`, under each of:
   `DEFAULT`, `DEFAULT_TENSOR_OP`, `ALGO0_TENSOR_OP` … `ALGO15_TENSOR_OP`.
5. **Beta=1 accumulation test:** initialize a BF16 dst, then compare
   `iter61(GEMM β=1 D=BF16)` to `legacy(GEMM β=0 D=FP32, then
   bf16_accum_axpy α=β=1)`. This is the production-call shape.
6. **Determinism test:** 5 reps of the same algo, bit-compare.
7. Diff histogram: exact / 1-ULP / 2..4-ULP / >4-ULP buckets, mean signed
   diff (the rounding bias), mean absolute diff. Non-finite outputs are
   counted separately and excluded from the diff sums.

The probe also includes reference casts of the FP32 GEMM output through:
* explicit RN-EVEN bias-LSB kernel (same as legacy)
* `__float2bfloat16` PTX intrinsic
* pure truncation (RZ) — control

---

## 3. Results table

### Shape A: `(M=2048, N=4096, K=8192)` (8.39 M elements, dWq/dWk/dWv)

```
[reference cast variants vs legacy RN-EVEN cast of FP32 GEMM out]
  cast: PTX __float2bfloat16       exact=100.0000%  (bit-identical)
  cast: pure truncation (RZ)       exact= 50.0813%  1ulp=49.9187%  mean_diff=-1.13e-08  mean_abs=2.48e-05

[iter 61 D=BF16 direct vs legacy D=FP32 + RN-EVEN cast]
  DEFAULT                          exact=100.0000%  (bit-identical)
  DEFAULT_TENSOR_OP                exact=100.0000%
  ALGO0_TENSOR_OP                  exact=100.0000%
  ALGO1_TENSOR_OP                  exact=100.0000%
  ALGO2_TENSOR_OP                  exact=100.0000%
  ALGO3_TENSOR_OP                  exact=100.0000%
  ALGO4_TENSOR_OP                  exact=100.0000%
  ALGO5_TENSOR_OP                  exact=100.0000%
  ALGO6_TENSOR_OP                  exact=100.0000%
  ALGO7_TENSOR_OP                  exact=100.0000%
  ALGO8_TENSOR_OP                  exact=100.0000%
  ALGO9_TENSOR_OP                  exact=100.0000%
  ALGO10_TENSOR_OP                 exact=100.0000%
  ALGO11_TENSOR_OP                 exact=100.0000%
  ALGO12_TENSOR_OP                 exact=100.0000%
  ALGO13_TENSOR_OP                 exact=100.0000%
  ALGO14_TENSOR_OP                 exact=100.0000%
  ALGO15_TENSOR_OP                 exact=100.0000%

[iter 61 beta=1 accumulation vs legacy GEMM+bf16_accum_axpy(1,1)]
  beta=1 ACCUM                     exact=100.0000%

[iter 61 determinism: 4 reps of DEFAULT_TENSOR_OP vs first run]
  rep 1..4                         exact=100.0000%
```

### Shape B: `(M=4096, N=2048, K=8192)` (8.39 M elements, dWo)

Identical pattern — every algo, beta=0 and beta=1, all 4 reps:
`exact=100.0000%`. (Full output in `cublas_bf16_rounding_probe.out`.)

### Pure-truncation control

RZ truncation differs from RN-EVEN in ~50% of elements (correct, since half
of the 17 bits being discarded round to even up and half round to even
down; among non-tie cases, RZ truncates while RN-EVEN rounds toward the
nearest representable value). The mean signed diff is **near zero (~1e-8)**
because the 1-ULP perturbations cancel — but the mean **absolute** diff is
2.5e-5 (one BF16 ULP at this magnitude scale). This confirms the diff
machinery works.

---

## 4. Interpretation

### What we now know definitively

1. **cuBLAS BF16-out uses RN-EVEN** on cuBLAS 12.0.2 / sm_89 / `CUBLAS_COMPUTE_32F_FAST_16BF`. Not RZ, not RN-away-from-zero, not stochastic. The output bits are identical to an explicit hand-coded RN-EVEN cast on every single element, across all 18 algo enum values.

2. **The choice of `cublasGemmAlgo_t` does not change rounding semantics** on this GPU/cuBLAS combo. All algo values that are accepted produce bit-identical output (and `DEFAULT` accepts identically). So no "fast vs slow algo gives different rounding" effect exists here.

3. **The iter 61 path is bit-equivalent to the legacy path with beta=1** (the actual production use). The "single fused cast at the end" of cuBLAS GEMM-with-D=BF16 produces identically the same bits as cuBLAS-GEMM-with-D=FP32 followed by `k_bf16_accum_axpy`.

4. **Iter 61 GEMMs are bit-deterministic across repeated calls** with the same handle/stream on the same device, at these shapes.

### What this rules out

* The iter 61 mechanism is **mathematically equivalent** to its predecessor at the cuBLAS-output level. It cannot, on its own, produce a NLL trajectory difference.
* "cuBLAS uses biased rounding (RZ-like)" — empirically false on this stack.
* "Algo selection drifted under cuBLAS auto-pick and a different algo with different rounding got picked" — empirically false; all algos match each other and match the explicit RN-EVEN cast.

### What this means for the regression hunt

The iter 61 path is **not** the source of the +0.50 nat regression. The
remaining iter-61-attributable suspect is therefore not the cuBLAS rounding;
it must be in **upstream/downstream code** that interacts with the BF16
grad commit:

* **Caller pre-zeroing protocol.** The iter 61 design comment explicitly
  requires the caller to "pre-zero the BF16 dW buffers at the start of each
  gradient-accumulation window" (gpu_chiron.cu:1262). If the legacy path
  zeroed FP32 scratch (always, before each call) and iter 61 relies on the
  caller zeroing the BF16 buffer at a different timing, a missed zero would
  produce stale grad re-add. Check `chiron_main.cpp:6515-6517`:
  `W.dWq_scratch.zero()` runs only on the `useFastBf16Grad == false` branch.
  Inspect the iter 61 branch's zero placement (which trainer loop sets
  dWq_bf to zero, and when).

* **bf16_accum_axpy elimination changed gradient-accumulation order.** In
  the legacy path, the per-layer dW commit was a separate FP32→BF16 step
  that happened *after* all four direction GEMMs (Q/K/V/O) had written into
  FP32 scratches. With iter 61, each direction's GEMM commits directly into
  its own BF16 dW buffer. This shouldn't change math (each direction is
  independent), but **if any code path was implicitly relying on dWq_bf
  containing stale data between calls** (e.g., a debug print, an Adam
  reader running in parallel, a sumsq check), the change-in-timing could
  matter.

* **Sumsq / grad-norm computation timing.** The iter 49 fused int8-Adam
  path computes grad-norm from BF16 grads. If iter 61 changed the moment
  at which dW_bf becomes "fully written" (the bf16_accum_axpy used to be
  the synchronization point), a stale read could occur. Check if any
  global-norm or clip step reads dW_bf without an explicit barrier after
  the iter 61 GEMM.

* **scratch_qbf/scratch_sdbf aliasing.** gpu_chiron.cu:1278+ shows the
  bf16 grad path reuses scratch buffers laid over `scfa_qpar`. The cast
  on line 1325 (`cast_f32_to_bf16(q, scratch_qbf, T*m)`) happens *after*
  flash-attention backward — if iter 61 changed the order such that
  scratch_qbf is consumed before the cast completes, you'd get garbage
  inputs to the dWq GEMM. (Probably already synced; verify.)

---

## 5. Recommended fix

**There is no fix at the cuBLAS-rounding layer to apply.** The cuBLAS path
is bit-identical to the explicit RN-EVEN kernel. Reverting iter 61 to the
legacy path would not change any output bits — it would only re-introduce
the bf16_accum_axpy launch overhead (the −2.64% throughput it gained).

The recommended next steps are:

1. **Diagnostic for the regression hunt:** add a parity test in
   `unit-tests/` that runs one training step on a tiny network with both
   the iter 61 path and the legacy path enabled, dumps `dW*_bf` for one
   layer, and bit-compares. If the test passes (it should, per this probe),
   the regression is not in the iter 61 GEMM itself but in an interaction
   with surrounding state (pre-zeroing timing, scratch aliasing, sumsq
   barrier). If the test fails, there is a missing scratch zero or a stale
   read somewhere in `chiron_main.cpp:6500-6620`.

2. **Audit `chiron_main.cpp:6500-6620`** for these specific issues:
   - Is `W.dWq_bf[l]` reliably zero at the *first* micro-step of every
     accumulation window when `useFastBf16Grad` is on? In the legacy
     branch, `W.dWq_scratch.zero()` runs unconditionally on line 6515. The
     fast branch skips that, so the BF16 zero must be done somewhere else.
     Find that "somewhere else" and confirm it always fires.
   - Does any reader of `dWq_bf[l]` between layer-completion and
     end-of-accumulation see a partially-written buffer? (Adam, clip,
     sumsq, etc.)
   - Are `scratch_qbf`/`scratch_sdbf` (aliased onto `scfa_qpar`) freed by
     the time the next GEMM reuses them?

3. **If the parity test passes** (most likely): the regression source is
   not in iter 61's GEMM mechanism but in a side-effect of the protocol
   change. Bisect by:
   - Re-running with `--bf16-grads off` (forces the legacy slow path on the
     iter 61 binary).
   - Re-running with both branches present but `useFastBf16Grad = false`
     forced via a debug flag (preserves bf16Grads = true but routes through
     the FP32-scratch+axpy path).
   - Compare NLL trajectories. If "fast off" matches the historical
     flagship, the protocol change is the culprit.

4. **If the parity test fails** (less likely, but probe doesn't simulate
   the full trainer interplay): the failure mode is identifiable from the
   parity dump (which buffer, which element, what magnitude of diff).

### Expected impact

* **Speed:** unchanged. No code change recommended at the cuBLAS layer.
* **NLL:** likely **0** from anything in the BF16 store rounding. The +0.50
  nat regression is sourced elsewhere — most likely in the iter 61 *protocol*
  changes around the GEMM (pre-zero timing, accum boundary, sumsq barrier)
  rather than in the GEMM itself.

---

## 6. Probe artifacts

* Source: `/home/robert/dev/glades-trainer/research/cublas_bf16_rounding_probe.cu`
* Binary: `/home/robert/dev/glades-trainer/research/cublas_bf16_rounding_probe`
* Raw output: `/home/robert/dev/glades-trainer/research/cublas_bf16_rounding_probe.out`
* Wall time per probe run: ~3 sec (both shapes, all algos, 4 determinism reps).

References:
* cuBLAS Library docs (NVIDIA, 13.2, Apr 2026): https://docs.nvidia.com/cuda/cublas/
* PTX ISA 9.2 (NVIDIA, 2026): https://docs.nvidia.com/cuda/parallel-thread-execution/
* "Defeating Training-Inference Mismatch via FP16" (arXiv:2510.26788)
* "Why Low-Precision Transformer Training Fails" (arXiv:2510.04212)
