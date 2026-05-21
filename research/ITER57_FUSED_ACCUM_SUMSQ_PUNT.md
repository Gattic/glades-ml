# Iter 57 — Fused bf16_accum_axpy + sum-squared — PUNT (semantic incompatibility with accum>1)

**Date**: 2026-05-16
**Iter**: 57 (eleventh iter under "stacking-wins" brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: PUNT — kernel was written and compiled, then reverted before benching.  The semantic of the fused sum-squared accumulation is **incompatible with gradient-accumulation across multiple micro-steps (accum > 1)**: sum-of-squares of intermediate partial-grad ≠ sum-of-squares of final-grad.  For accum=1 (our default config) the fusion would work, but the trainer-side refactor to detect accum=1 and route accordingly is heavier than 1 iter of work for the expected ~1.5% upside.

---

## TL;DR

Approach: combine the per-layer `bf16_accum_axpy` (2.8%) with the per-layer `sum_squared_accumulate_bf16` (0.9%) that `compute_grad_norm_sq` does AFTER the backward.  Both touch the same BF16 grad tensor; fusing them saves one full pass over the T·m-class BF16 grad buffer.

The fused kernel was written and built:

```cuda
__global__ void k_bf16_accum_axpy_with_sumsq(
    uint16_t* dst_bf16, const float* src_f32,
    float alpha, float beta, size_t n,
    float* sumsq_acc)
{
    // 1. dst_bf16 = bf16(alpha*src + beta*decode(dst_bf16))  [original semantic]
    // 2. atomicAdd(sumsq_acc, blockReduce(dst_bf16_value²))  [new: sum-of-squares of result]
}
```

Build clean.  Then the issue surfaced during the trainer-routing design.

---

## Why the semantic blocks general accum>1

The gradient-accumulation pattern in chiron_main.cpp:
```
for each micro-step in window:
  backward(accumulate=true)
    → bf16_accum_axpy(dW_bf, dW_scratch, 1, 1)  # adds to dW_bf
# end of window
compute_grad_norm_sq:
  sum_squared_accumulate_bf16(dW_bf, ...)
  → sum_sq of FINAL accumulated grad (after all micro-steps)
```

For accum=N micro-steps, the "correct" grad norm uses the sum-of-squares of the FINAL accumulated grad (computed once after all micro-steps).  My fused kernel would have summed squares of the per-micro-step partial accumulated grads instead — `Σᵢ ‖dW_after_step_i‖²` ≠ `‖Σᵢ ΔdW_i‖²` (the latter is what's needed for global norm).

For accum=1, both expressions are equal (single step → final == only step).  Our config uses accum=1, so the fusion would have been correct.  But routing the fused kernel ONLY for accum=1 requires:
- Detection of "is this the final (also first) micro-step in the window" inside scfa_attention_backward
- Either a runtime flag plumbed through OR a static check at trainer-init time

Plus moving `s.gradNormSq.zero()` from inside `compute_grad_norm_sq` to before the backward call, and removing the per-Wq/Wk/Wv/Wo sum_sq calls from compute_grad_norm_sq.

That's a multi-touch trainer refactor for +1.5% expected.  Not single-iter scope.

---

## Why this iter still produces an artifact

The library-side kernel (`bf16_accum_axpy_with_sumsq` in `gpu_kernels.cu` + header) compiles cleanly and is correct for accum=1.  It's reverted in this iter but the design is recorded here so a future iter can pick it up if accum=1 is locked in (or if a multi-iter refactor is accepted).

---

## Sequence status

| iter | target | result | win |
|---: |---     |---     |---:  |
| 47  | SCFA dwconv-dK | PASS | +5.20% |
| 48  | LN-bwd dgamma  | FAIL | +2.4% |
| 49  | Fused Adam-int8 | PASS | +6.98% |
| 50  | SCFA sub-conv  | FAIL | +1.31% |
| 51  | --scfa-checkpoint-inner | FAIL | +4.46% |
| 52  | cuBLAS algo override | NULL | 0% |
| 53  | LN-bwd 2-phase | FAIL | +2.86% |
| 54  | (meta-analysis) | META | n/a |
| 55  | --cuda-graphs / --fp8-attn | NULL | ~0% |
| 56  | warp argmax | FAIL | +1.62% |
| 57  | accum_axpy+sumsq fused | PUNT | n/a (semantic conflict) |

Cumulative shipped: **+12.5%**.  Score 2/11 PASS.  **Seven consecutive non-PASS iters.**

---

## Iter 58 candidate

Engineering wins are exhausted at the strict +5% bar.  Options for iter 58:

1. **Multi-iter paradigm arc**: pick one of {BF16 residual stream, FP8 readout with restructured amax, sparse attention} and treat the next 3-5 iters as a single implementation arc.

2. **Continue grinding** small engineering attempts that will all FAIL at the strict bar; ship one of them retroactively if the user signals a bar relaxation.

3. **Trust the META** (iter 54, iter 56's tail discussion): the strict +5% bar is empirically unattainable on this stack.  Wait for user direction.

Without user direction, iter 58 will default to (2).
