# Iter 59 — cuBLAS algo override on SCFA outer GEMMs — NULL

**Date**: 2026-05-16
**Iter**: 59 (thirteenth iter under "stacking-wins" brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: NULL — confirms cuBLAS `DEFAULT_TENSOR_OP` is also optimal for the SCFA outer FAST_16BF GEMM shape (T=8192, m=2048, k=512), mirroring iter52's NULL on the readout shape.

---

## TL;DR

iter52 tested `CUBLAS_GEMM_ALGO0..15_TENSOR_OP` on the readout-shape GEMMs (T=8192, V=32000, m=2048) via the `gemmex_bf16_impl` shared helper — all algorithms within 0.3% of `DEFAULT_TENSOR_OP`, NULL result.

iter59 closes the loop on the OTHER major cuBLAS path that iter52 didn't reach: `sgemm_rowmajor_fast16bf_impl` (used by the SCFA outer projection GEMMs at shape T=8192, m=2048, k=512).  Added env override `GLADES_SCFA_OUTER_ALGO=0..15`, swept all 16 explicit tensor-op algos:

| algo                 | tok/s @ step 30 | Δ vs DEFAULT |
|---                   |---:             |---:          |
| `DEFAULT_TENSOR_OP`  | 43,383          | baseline     |
| `ALGO0_TENSOR_OP`    | 43,372          | −0.03%       |
| `ALGO1_TENSOR_OP`    | 43,351          | −0.07%       |
| `ALGO2_TENSOR_OP`    | 43,380          | −0.01%       |
| `ALGO3_TENSOR_OP`    | 43,321          | −0.14%       |
| ...                  | ...             | ...          |
| `ALGO15_TENSOR_OP`   | 43,316          | −0.15%       |

Full sweep range: 43,207 – 43,383.  None beat DEFAULT.

NULL.  Confirms cuBLAS DEFAULT is at the local optimum for BOTH readout (iter52) AND SCFA outer (iter59) GEMM shapes on RTX 4080 SUPER.

---

## What this closes

The cuBLAS-algo-override hypothesis is now empirically exhausted.  Both major paths through `cublasGemmEx` (gemmex_bf16_impl + sgemm_rowmajor_fast16bf_impl) have been swept; cuBLAS auto-pick is optimal everywhere.

This was the last remaining "easy cuBLAS optimization" hypothesis.  No cuBLAS-level lever beats DEFAULT.

---

## Sequence status (now 13 iters)

| iter | target | result | win |
|---: |---     |---     |---:  |
| 47  | SCFA dwconv-dK | PASS | +5.20% |
| 48  | LN-bwd dgamma  | FAIL | +2.4% |
| 49  | Fused Adam-int8 | PASS | +6.98% |
| 50  | SCFA sub-conv  | FAIL | +1.31% |
| 51  | --scfa-checkpoint-inner | FAIL | +4.46% |
| 52  | cuBLAS algo (readout/inner) | NULL | 0% |
| 53  | LN-bwd 2-phase | FAIL | +2.86% |
| 54  | (meta-analysis) | META | n/a |
| 55  | --cuda-graphs / --fp8-attn | NULL | ~0% |
| 56  | warp argmax | FAIL | +1.62% |
| 57  | accum_axpy+sumsq fused | PUNT | n/a |
| 58  | --bf16-logits-parallel-bwd | NULL | +0.15% |
| 59  | cuBLAS algo (SCFA outer) | NULL | ~0% |

Cumulative shipped: **+12.5%**.  Score 2/13 PASS (15%).  Hit rate continues to fall.

**9 consecutive non-PASS iters.**

---

## What remains

The dead-zone map (now fully cataloged):

- **40% cuBLAS GEMMs** — all paths tested with algo override.  DEFAULT optimal.
- **15% SCFA element-wise** — mem-bound + fp32-FMA precision drift on fusion (iter50).
- **6.6% LN backward** — too-small-target (iter48 + iter53).
- **8% Adam+cast** — iter49 saved the big chunk.  Remainder semantically conflicts with accum>1 (iter57).
- **8.6% depthwise conv** — iter47 saved 5.2%.  Remaining 3% well-parallelized.
- **2-3% small kernels** — argmax (iter56), softmax, casts — all too-small-target.

Every kernel slice ≥2% has been attacked.  All solo-iter wins exhausted at the strict +5% bar.

---

## Same META asking continues

The user has been silent through 3 META iters (54, 56, 57 tail, 58, now 59).  Same options remain:

A. Relax bar to +3% — ships iter51 + iter53 + iter56 for ~+10% additional cumulative.
B. Commit to multi-iter paradigm arc.
C. Reframe the brief.

Default action (continued grinding): another FAIL or NULL next iter.

---

## What this iter ships

No code change.  Doc only.
