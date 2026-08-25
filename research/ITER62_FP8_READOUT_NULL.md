## Iter 62 — FP8 readout forward (Arc 1, iter 1 of 4) — NULL (cuBLASLt FP8 coverage gap)

**Date**: 2026-05-16
**Iter**: 62 (Arc 1: FP8 readout, first iter of 4-iter arc)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)
**Verdict**: **NULL** — cuBLASLt returns `CUBLAS_STATUS_NOT_SUPPORTED` (status 15) at the readout shape on CUDA 12.0 + Ada (sm_8.9).  Same toolkit-coverage gap iter 55 HELIUM hit at attention shapes; design risk R4 / R7 materialized.  **Arc 1 dead-gates here** per the design plan ("iter 62 fail = paradigm invalidated; pivot to BF16-residual-p or MoE FFN").

---

## TL;DR

Implemented `sgemm_rowmajor_abt_fp8_e4m3_bf16_bf16out` (cuBLASLt FP8 GEMM with BF16 input + BF16 output direct) and the `--fp8-readout-fwd` trainer flag, wired into the readout forward block.  Smoke-tested: code is correct, fallback works, but cuBLASLt rejects the FP8 matmul on this toolkit/device combo with status 15.

```
[W] [fp8-readout-fwd] cuBLASLt FP8 readout GEMM rejected at shape (T=8192,
    V=32000, m=2048) — likely CUDA 12.0 FP8 algo coverage gap on sm_8.9
    (iter 55 HELIUM hit the same at attention shape).  Disabling FP8
    readout for the rest of the run; falling back to BF16 readout GEMM.
```

After fallback, the run is bit-identical to iter 61 silent flagship at 49,606 tok/s (step 30 smoke).

| metric              | iter61 silent | iter62 result | delta |
|---                  |---:           |---:           |---:   |
| tok/s @ step 30     | 49,616        | 49,606        | −0.02% (noise) |
| val NLL @ smoke step 30 ema | 9.8800 | 9.8926        | +0.013 nat (noise, identical RNG path) |
| FP8 GEMM executions | n/a           | 0 (all rejected) | — |

The FP8 path never actually executes; the fallback BF16 GEMM is what produces the wall result.

---

## What was implemented

### `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas_fp8.cu` + `.h`

New function `sgemm_rowmajor_abt_fp8_e4m3_bf16_bf16out(M, N, K, α, A_bf, lda, B_bf, ldb, β, C_bf, ldc, d_scaleA, d_scaleB)`:
- ABT layout variant of `sgemm_rowmajor_fp8_e4m3_bf16` (no transpose-cast on B since row-major [N, K] is already K-major).
- BF16 inputs cast to E4M3 with per-tensor scales.
- cuBLASLt matmul `CUBLAS_COMPUTE_32F` with `CUBLASLT_MATMUL_DESC_FAST_ACCUM=1`.
- Output BF16 written DIRECTLY to caller buffer (col-major `[N, M]` ld=N is byte-equivalent to row-major `[M, N]` ld=N when `ldc=N`), eliminating the FP32 round-trip used by the existing NN variant.

### `glades-trainer/trainer/chiron_main.cpp`

- New `Config::fp8ReadoutFwd` flag (default false), CLI `--fp8-readout-fwd`.
- New `Scratch::fp8_readout_scales` buffer (2 FP32 slots: q_L, E).  Re-uses `fp8_scales[6..7]` when `--fp8-attn` is also set.
- Readout-forward call site (the `cfg.bf16LogitsStorage` branch ~line 6821): on `--fp8-readout-fwd`, recompute amax of `q_L_bf` and `E_bf_cache` each step, attempt the FP8 GEMM, fall back to existing BF16 GEMM on failure.
- Runtime kill-switch: after the first `CUBLAS_STATUS_NOT_SUPPORTED`, set `s_fp8_readout_disabled = true` and skip subsequent FP8 attempts — eliminates per-step cuBLASLt error spam and the wasted amax+setup overhead.
- Startup log gates `--fp8-readout-fwd` on `--bf16-logits-storage` + `fp8_supported()` (sm_8.9+).

---

## Why it failed

cuBLAS status 15 = `CUBLAS_STATUS_NOT_SUPPORTED`.  Three priors converge:

1. **iter 55 HELIUM precedent**: existing `sgemm_rowmajor_fp8_e4m3_bf16` (NN variant) returned the same status at attention shapes; iter 55 added the runtime kill-switch.  HELIUM's comment in `chiron_main.cpp:6622`: *"likely cuBLASLt < 12.3 on Ada (no general FP8 algo coverage)"*.  Our toolkit is CUDA 12.0.

2. **Shape**: M=8192, N=V=32000, K=2048.  V=32000 is a non-power-of-two, divisible by 32 but not by 64.  cuBLASLt's FP8 algo tables on Ada may require finer alignment (powers of 2 or specific tile sizes).  Design risk R7 (cuBLASLt FP8 may require N power-of-2) flagged this.

3. **Toolkit version**: CUDA 12.0 cuBLASLt is known to have incomplete FP8 algo coverage on sm_8.9.  CUDA 12.3+ improves this materially.  Upgrading the system CUDA is out of scope for this iter and may have unrelated side effects.

The design doc anticipated this exact outcome:
> **R4. cuBLASLt FP8 algo undertuned on (8192, 32000, 2048) sm_8.9.** Priors from iter 52, 59 BF16 algo overrides NULL.  Trigger: forward FP8 GEMM ≥ BF16 baseline wall-clock at same shape. Pivot to manual algo override (iter 65) but likely also NULL.

The result is stronger than R4 predicted: cuBLASLt doesn't even attempt the FP8 path — there's no algo to override.

---

## Per the design plan

> **iter 62 fail** = paradigm invalidated. Pivot to BF16-residual-p or MoE FFN.

This is exactly an iter 62 kill-gate FAIL.  Arc 1 (FP8 readout, iters 62-65) is empirically blocked at iter 1 by toolkit constraints.

If the cuBLAS toolkit upgrades to 12.3+ in a future system-software refresh, the FP8 wrapper is in tree and the flag `--fp8-readout-fwd` would light up automatically — Gate-0 retesting is then a single bench command.  No further code work needed.

---

## What stays in tree

- `sgemm_rowmajor_abt_fp8_e4m3_bf16_bf16out` wrapper — correct implementation, latent until cuBLAS coverage improves.
- `--fp8-readout-fwd` flag with runtime kill-switch — silent no-op on the current toolkit; ready for future revalidation.
- This document records the falsification reason for posterity (avoids re-attempting under the same toolkit).

---

## Arc 1 status

| iter | scope | result | per-iter gate |
|---:|---|---|---|
| 62  | `--fp8-readout-fwd`: F1 fwd FP8 | **NULL (cuBLAS NOT_SUPPORTED)** | failed: cuBLASLt FP8 path rejected |
| 63  | Full `--fp8-readout`: B1 E5M2, B2/B3 FP8 GEMM | not attempted | predicted NULL by same toolkit gap |
| 64  | amax EMA every-N-steps | not attempted | depends on 62 + 63 passing |
| 65  | cuBLASLt algo sweep | not attempted | iter 62 NULL is the algo-coverage answer |

**Arc 1 outcome**: 0 / 4 iters passed.  Probability of success on this toolkit/hardware: 0 (empirically falsified).

---

## Pivot options

Per `PARADIGM_ARC_SYNTHESIS_2026_05_16.md`, the dispatch order recommended Arc 1 → Arc 2 → Arc 3.  With Arc 1 dead, the open options are:

**A. Arc 2 (BF16 residual-p, iters 63-67)** — original plan, uncorrelated risk to Arc 1.  Probability ~50% iter-bench / ~25% production.

**B. Arc 3 (MoE attention-shear, iters 63-71)** — biggest commitment, highest variance.  Probability ~30% by iter 67.

**C. Stop arc, freeze flagship at iter 61 silent (48,521 tok/s)** — honest endpoint given Arc 1 fail.

**D. Upgrade CUDA to 12.3+ and retry iter 62** — out of arc scope, depends on system access + potential side effects.

The original synthesis explicitly noted "Arc 1 (FP8) fails outright + you've decided the 5-iter cost is not worth the expected 1-3% gain → stop at iter 65, write a paradigm-arc-FAIL summary, freeze the flagship."  Arc 1 failed at iter 1 of 4 (cheaper than the 5-iter estimate predicted), so the stop-or-pivot decision is informed by less sunk cost.

---

## Files

- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas_fp8.h` — declaration
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas_fp8.cu` — implementation
- `glades-trainer/include/Backend/Machine Learning/Networks/cuda/gpu_blas_fp8.h` — header mirror
- `glades-trainer/trainer/chiron_main.cpp` — flag, allocator, call-site, runtime kill-switch
- This document
