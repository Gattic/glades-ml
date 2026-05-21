## Iter 108 — dW_bf cuBLAS beta=0 + skip pre-zero — FAIL on NLL drift

**Date**: 2026-05-21
**Iter**: 108 (post iter 107 PASS)
**Branch**: vesta5 (glades-ml + glades-trainer)
**Verdict**: **FAIL** — math not bit-identical despite analytically appearing equivalent. NLL drift +0.5 nat at production over 100 steps. Mechanism disabled (changes preserved in code as no-ops for documentation/future revisit).

---

## Motivation

After iter 107's STRICT PASS (cuBLAS beta=1→0 in `flash_attention_backward_cublas_tiled`, eliminating 1.15 GB/step caller memsets), apply the same pattern to dW_bf gradient buffer pre-zeros:

- Line 8985 chiron_main.cpp: `W.dWq_bf[l]->zero(); W.dWk_bf[l]->zero(); W.dWv_bf[l]->zero(); W.dWo_bf[l]->zero();` — 4 × 16 MB = 64 MB per layer × 24 layers = **1.5 GB/step memset**.
- Inside `chiron_attention_shear_backward_bf16w_bf16g_tiled`: 4 `sgemm_rowmajor_atb_bf16_dst_bf16` calls write dWq_bf/dWk_bf/dWv_bf/dWo_bf with `beta=1.0f` (accumulate into pre-zeroed buffer).

Plan: change these 4 cuBLAS beta=1→0 (overwrite). Skip the trainer pre-zero. Math analytically bit-identical when caller starts with zero buffer.

## Implementation

- gpu_chiron.cu `chiron_attention_shear_backward_bf16w_bf16g_tiled`: added `bool dw_beta_zero = false` param. When true, `dw_beta = 0.0f`; all 4 dW_bf cuBLAS writes use `dw_beta`.
- gpu_chiron.h: added default param to declaration + no-CUDA stub.
- glades-trainer/include vendored header: mirror.
- chiron_main.cpp:
  - `iter108DwBfBetaZero` flag (default OFF), CLI `--iter108-dw-bf-beta-zero`
  - When flag on AND accumSteps==1: pass `dw_beta_zero=true` to shear_backward
  - When flag on AND accumSteps==1 AND bf16Grads: skip the per-step dW_bf pre-zero

## Bench (single-seed × 100 steps × seed=1337 × T=16384 L=24 w=4)

| run | wall | tok/s @ 76 | loss @ 76 | NLL @ 100 | verdict |
|---  |---:  |---:        |---:       |---:       |---      |
| iter 97+99+101+103+106 (iter 106 PASS baseline)              | 61.1s | 26,958 | 10.1776 | 9.8476 | BASELINE |
| iter 97+99+101+103+106+107 (iter 107 PASS)                   | 60.7s | 27,106 | 10.1776 | 9.8476 | PASS |
| **iter 97+99+101+103+106+107+108**                           | **60.4s** | **27,266** | **10.5113** | **10.3517** | **FAIL** |

**Wall**: -0.3s (+0.5% additional). Wall improvement seen as predicted.

**NLL drift**: 9.8476 → 10.3517 = **+0.504 nat** at step 100. Loss trajectory diverges from step 25:
- iter 108 OFF: step 25 best=10.2939, step 76=10.1776
- iter 108 ON:  step 25 best=10.3709, step 76=10.5113 (increasing — training divergent!)

The math is NOT bit-identical. Gradient computation is broken.

## Verdict matrix

| bar | wall threshold | NLL threshold | result |
|---  |---:            |---:           |---     |
| Strict brief (≥5% tok/s + ±0.02 NLL) | +5% | ±0.02 | **FAIL** (NLL drift +0.504 nat) |
| iter 60 relaxed (+3% + multi-seed parity) | +3% | ±0.05 | **FAIL** |
| Any bar | — | bit-identical | **FAIL** (mechanism math-broken) |

**FAIL** — first parity failure in this ralph-loop session. Different from prior NULLs (iter 95/96/98/102/104) which were sub-noise but parity-clean. iter 108 is a genuine bug.

## Failure analysis

The analytic argument said math should be bit-identical:
- `dW = 1*A*B + 1*0_initial` (legacy: beta=1 accumulate, caller pre-zeroes) = `A*B`
- `dW = 1*A*B + 0*garbage_initial` (iter 108: beta=0 overwrite, no pre-zero) = `A*B`

cuBLAS GemmEx documentation states beta=0 makes C-read unnecessary ("C does not need to be a valid input when beta is zero"). So initial garbage shouldn't matter.

But empirically, the gradient IS wrong. Hypotheses (untested, future investigation):

1. **cuBLAS GemmEx BF16-dst beta=0 quirk**: the BF16 destination path (gemmex_bf16_impl_dst_bf16) may have a different beta=0 handling than the FP32-dst path. Possibly the BF16 quantization step reads the C buffer for some accumulation logic.

2. **Multi-call accumulation across stages**: although each shear_backward call writes its layer's dW_bf once, perhaps the SCFA stack has another consumer/producer that depends on the prior-step's dW_bf content (e.g., a momentum update implicit in some sub-kernel). Unlikely but possible.

3. **NaN/Inf in initial buffer**: if the un-pre-zeroed dW_bf contains NaN from prior step's gradient explosion or initial allocation garbage, cuBLAS beta=0 might propagate it. iter 108 might be exposing latent NaN issues that pre-zero was masking.

4. **Stream synchronization**: the pre-zero may have been providing an implicit synchronization barrier that downstream computation depends on.

None of these have been verified. Further investigation would require deeper kernel/cuBLAS instrumentation.

## Resolution

- iter 108 flag retained in code as **documented FAIL** (no-op when set; legacy behavior preserved).
- Library function `chiron_attention_shear_backward_bf16w_bf16g_tiled` keeps the `dw_beta_zero` parameter for API stability, but internally forces `dw_beta = 1.0f` regardless.
- Trainer's `iter108SkipDwBfZero` forced to `false`.
- All production behavior unchanged from iter 107 PASS baseline.

## Default and production recommendation

iter 108 flag is **non-functional** (no-op). Production stack remains:
- iter 97+99+101+103+106 (5 trainer flags) + iter 107 unconditional library change
- = **+7.52% n=3 multi-seed strict-bar PASS** (per iter 107 evidence)

No regression from iter 108 attempt. The mechanism remains an open opportunity for future investigation if cuBLAS BF16-dst beta=0 semantics are clarified.

## Lessons learned

1. **Analytic bit-identity ≠ empirical bit-identity** in CUDA. The cuBLAS beta=0 specification is theoretically safe but may interact unexpectedly with BF16 quantization, NaN handling, or implementation details.

2. **First-step bench is essential**. iter 108's failure was visible at step 25 (loss best=10.3709 vs baseline 10.2939). Earlier validation would have caught it before n=3 multi-seed.

3. **Pre-zero memsets may have hidden value** beyond just zero initialization — possibly stream sync, NaN scrubbing, or implicit barriers. Eliminating them requires more careful validation than other "obvious" optimizations.

4. **The "eliminate redundant memory ops" mechanism class has a boundary**: iter 103/106/107 worked, iter 108 didn't. The difference: iter 103/106/107 acted on element-wise pre-zeros / memcpys with no math interaction. iter 108 changed cuBLAS GEMM beta semantics — a deeper kernel-level interaction.

## Files

- This document.
- `research/runs/2026-05-21-iter108-gate0/iter108_combined_100step.log` (failure evidence: NLL 10.35).
- Code:
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu`: `chiron_attention_shear_backward_bf16w_bf16g_tiled` signature has `bool dw_beta_zero` param (forced to no-op via `dw_beta=1.0f`).
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.h` + vendored: declaration with default param.
  - `glades-trainer/trainer/chiron_main.cpp`: flag + CLI + conditional dispatch — all forced to no-op.
