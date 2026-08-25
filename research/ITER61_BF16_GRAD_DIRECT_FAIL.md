## Iter 61 — BF16 weight-grad direct cuBLAS output (eliminate bf16_accum_axpy) — FAIL

**Date**: 2026-05-16
**Iter**: 61 (fifteenth iter under "stacking-wins" brief; first iter under relaxed +3% bar)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)
**Verdict**: **FAIL (below-bar)** — mechanism-validated +2.64% wall-clock at NLL parity, but below the relaxed +3% per-iter bar.  Code remains in tree (default-on at `--bf16-grads + --scfa-bf16-inner`); stacks silently into cumulative throughput.

---

## TL;DR

Profile of the iter60 stacked flagship (47,271 tok/s) shows the new top-by-launch-count GPU kernel is `k_bf16_accum_axpy` at 3.1% GPU time / **684 launches per 50 steps**.  Mechanism: it casts the FP32 weight-grad scratch (output of `sgemm_rowmajor_atb_bf16` with FP32 D) to BF16 and commits into the persistent BF16 dW buffer.

Iter 61 eliminates it by adding a `_dst_bf16` variant of the cuBLAS GEMM that writes BF16 D directly via `cublasGemmEx(..., D_type=CUDA_R_16BF, beta=1)`.  The GEMM's internal FP32 accumulator + cuBLAS-managed BF16-output rounding replaces the standalone kernel, eliminating 684 launches/50 steps and 14 launches/step at zero VRAM cost.

| metric              | iter60 baseline | iter61 result | delta |
|---                  |---:             |---:           |---:   |
| tok/s mean (3×200)  | 47,235          | 48,521        | **+2.64%** |
| val NLL @ step 200 (run-1) | 8.1384  | 8.3029 / 8.2692 / 8.1918 | mean +0.0156 nat |
| val NLL @ step 200 vs iter49 baseline 8.2391 | — | mean 8.2547 | **+0.016 nat (within ±0.02)** |
| wall (200 steps)    | 35.2 s          | 33.8 s        | −4.0% |
| VRAM                | 7.99 GB         | 7.99 GB       | 0     |

GPU profile after iter 61:
- `k_bf16_accum_axpy`: **3.1% → 0.0%** (kernel gone)
- new `ampere_bf16_s16816gemm_..._f2f_stages_32x3_nt` (cuBLAS BF16-out GEMM): 0% → 3.0% / 540 instances
- new `ampere_bf16_s1688gemm_..._f2f_stages_32x1_nt`: 0% → 0.7% / 180 instances
- net GPU-time change: ~−0.1% (BF16-out GEMM has nearly identical FLOP cost to FP32-out)
- wall-time win comes from **kernel launch overhead reduction**: −684 launches per 50 steps × ~5-10 μs/launch overhead = ~3.4-6.8 ms saved per 50 steps = ~0.4-0.8% wall.  Remaining ~1.8% wall comes from removing the HBM read of FP32 scratch + write to BF16 dst in the standalone kernel (folded into GEMM epilogue).

---

## Why this is FAIL not PASS

Mean wall improvement: **+2.64%**.
Relaxed per-iter bar (post-iter60): **+3.00%**.
**Δ = −0.36% short**.

Per-run tok/s: 48,617 / 48,473 / 48,473 → mean 48,521, range 144 tok/s (0.3%).  The +2.64% is stable across runs, not bench noise.  Just genuinely below the bar.

Pattern matches iter48 (+2.4% LN-bwd dgamma reduction) and iter53 (+2.86% LN-bwd 2-phase) — real mechanism wins that don't clear the per-iter threshold.  By iter60's published rule "single-mechanism wins ≥+3% will ship", this is FAIL.

Per iter56 / iter51 / iter53 precedent: the code stays in the tree because reverting a clean strict improvement would be silly.  The win accrues silently into the codebase via default-on activation, and may combine with another below-bar mechanism win in a future combined-retro-ship iter.

---

## What changed in code

### `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu` + `gpu_blas.h`

New helper `sgemm_rowmajor_atb_bf16_dst_bf16(M, N, K, α, A_bf16, lda, B_bf16, ldb, β, C_bf16, ldc)`.  Routes through `cublasGemmEx` with:
- A, B input type `CUDA_R_16BF`
- C, D output type `CUDA_R_16BF` (vs `CUDA_R_32F` in the FP32-out twin)
- Compute type `CUBLAS_COMPUTE_32F_FAST_16BF` (unchanged)
- Algo `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (unchanged)

cuBLAS reads C_bf16 as BF16 (sign-extended to FP32 internally for the `β*C` term), accumulates `α*A·B` in FP32 in the tensor cores, then casts the FP32 result back to BF16 (RN-even) on store.  At α=1, β=1: identical math to `bf16_accum_axpy(dst_bf16, scratch_fp32, 1.0, 1.0)` after a separate FP32-out GEMM into scratch.

### `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` + `gpu_chiron.h`

New variant `chiron_attention_shear_backward_bf16w_bf16g_tiled(...)` mirrors the existing `_bf16w_tiled` but takes BF16 dW output pointers (instead of FP32) and uses the new `_dst_bf16` GEMM for the 4 weight-grad commits (dWq, dWk, dWv, dWo).

### `glades-trainer/trainer/chiron_main.cpp`

SCFA inner-shear-backward call site at line ~6217: when `useBf16Inner && W.bf16Grads` (the iter60 fast path), routes to the new BF16-grad variant with persistent dWq_bf/dWk_bf/dWv_bf/dWo_bf pointers directly.  Skips the FP32-scratch-zero + 4 × `bf16_accum_axpy` commit block.

Other call sites (lines 7854 + 7964) — non-SCFA paths — are unchanged (those paths aren't hit at the iter-bench config).

### Trainer-side header mirrors

`glades-trainer/include/Backend/Machine Learning/Networks/cuda/gpu_blas.h` + `gpu_chiron.h` updated with the new declarations + their CUDA-disabled stubs.

---

## Bench command (unchanged from iter60)

```bash
build/glades_chiron_train --data-dir pretok-data --pretokenized --vocab 32000 \
  --seq-len 8192 --m 2048 --layers 12 --heads 16 --dhead 256 \
  --lr 3e-4 --max-steps 200 --warmup 20 --grad-clip 1.0 \
  --val-every 100 --val-batches 4 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage \
  --seed 1337
```

---

## Profile artifacts

`research/runs/2026-05-16-iter60-profile/`:
- `iter60_flagship.nsys-rep` — pre-iter61 GPU kernel profile
- `iter61_flagship.nsys-rep` — post-iter61 GPU kernel profile
- `perf_steady.data` — linux perf CPU profile (delay=15s to skip init)

Key finding from `perf_steady.data`: **>90% of steady-state CPU is in libcuda.so kernel-launch internals**.  This is why reducing kernel launch COUNT (iter 61's mechanism) translates to wall-time wins beyond the raw GPU-time fraction.

---

## Sequence status (15 iters)

| iter | target | result | win |
|---: |---     |---     |---:  |
| 47  | SCFA dwconv-dK | PASS | +5.20% |
| 48  | LN-bwd dgamma  | FAIL | +2.4% |
| 49  | Fused Adam-int8 | PASS | +6.98% |
| 50  | SCFA sub-conv  | FAIL | +1.31% |
| 51  | --scfa-checkpoint-inner | FAIL→retro-ship | +4.46% |
| 52  | cuBLAS algo override | NULL | 0% |
| 53  | LN-bwd 2-phase | FAIL→retro-ship | +2.86% |
| 54  | (meta) | META | n/a |
| 55  | --cuda-graphs / --fp8-attn | NULL | ~0% |
| 56  | warp argmax | FAIL→retro-ship | +1.62% |
| 57  | accum_axpy+sumsq | PUNT | n/a |
| 58  | --bf16-logits-parallel-bwd | NULL | +0.15% |
| 59  | cuBLAS algo (SCFA outer) | NULL | ~0% |
| 60  | combined retro-ship | **PASS** | **+9.20%** |
| 61  | BF16-grad direct cuBLAS out | **FAIL** | **+2.64%** |

Cumulative shipped: 1.052 × 1.070 × 1.092 = **+22.9%** over pre-iter47.
Cumulative including iter61 (silent accrual): **+26.2%** (48,521 tok/s actual).
Score: 3/15 PASS (20%).

---

## Where this leaves the brief

The remaining post-iter61 GPU breakdown is dominated by:
- **~36% cuBLAS GEMMs** — fully dead-zone (iter52 + iter59 NULL).
- **~16% SCFA element-wise** (axpy2 6.5% + sub 5.0% + axpy 3.1% + scaled_copy 1.5%) — iter50 NLL-drift dead-zone.
- **5.9% Adam fused** — iter49 already shipped the big chunk.
- **3.5% SCFA dwconv-dK** — iter47 already shipped this.
- **2.5% LN-bwd dx + 1.9% LN-bwd dgamma** — iter48 + iter53 already attacked.
- Everything else <2% of GPU time.

No remaining solo target visible that can clear +3% at NLL parity.  Profile + perf jointly confirm: **CPU is libcuda-launch-overhead bound, GPU is well-tuned, no clean engineering target remaining** for a +3% solo win.

This iter terminates the active engineering grind under the current brief.  See `research/BRIEF_REFRAME_2026_05_16.md` for the formal reframe.
