## Iter 83 — Online (single-pass) softmax merge — NEGATIVE (gradient explosion)

**Date**: 2026-05-20
**Iter**: 83 (twenty-ninth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **NEGATIVE** — zero wall improvement (+0.15% slower vs baseline) AND mid-train gradient explosion (||g||=11.275 at step 91 vs baseline ~2-4) AND NLL drift +0.147 nat at step 200.  Kernel reverted.

---

## Hypothesis

Combine the SEPARATE mask-write pass (write -FLT_MAX to upper triangle) with the row-max-find pass into a single loop over the row.  Saves one full read of the row (one of three passes).  Estimated +0.5-1.5% wall on the 3.6% causal_mask_softmax slice.  Math appears identical: for j > row, write -FLT_MAX AND use -FLT_MAX in max-find; for j ≤ row, read sRow[j] AND use it in max-find.

## Bench (production stack, seed=1337, 200 steps)

| metric | baseline (n=4 mean) | iter 83 (online softmax) |
|---     |---:                 |---:                       |
| tok/s (final) | 24,023 | ~24,000 (essentially same) |
| wall (200 steps) | 136.40 s | **136.6 s (+0.15% SLOWER)** |
| val NLL @ step 200 | 7.6154 | **7.7624 (+0.147 drift)** |
| ||g|| @ step 91 | ~2-4 (baseline typical) | **11.275 (GRADIENT EXPLOSION)** |

Per-step training loss trajectory:

| step | baseline | iter 83 | Δ |
|---:  |---:      |---:     |---: |
| 1    | 10.5762  | 10.5762 | bit-identical |
| 11   | 10.0846  | 10.0846 | bit-identical |
| 21   |  9.4884  |  9.4890 | +0.0006 |
| 81   |  8.3728  |  8.3491 | −0.024  |
| **91** |  ~8.5 normal | **9.0601 (||g||=11.275)** | **gradient explosion** |

## Why this fails

Despite the math appearing identical (for each j: write OR read sRow[j], then use that value in localMax), the merged kernel produces a DIFFERENT scheduling pattern.  Per the iter 74 hypothesis: changing kernel structure changes L2 prefetch / CUDA scheduler ordering which can shift sub-ULP values in subsequent kernels.  Amplified by the Adam/SR-cast pathway, the drift compounds and triggers a gradient instability at step 91.

The original 2-pass structure (mask-write + sync + max-find) has a well-defined memory barrier between writes and reads.  The merged kernel has writes and reads interleaved within the same loop body (different threads read different j values while other threads write -FLT_MAX to their j values).  Within-warp this is safe (no race on individual addresses), but the GLOBAL memory consistency model may produce different observed ordering across threads in the same block.

## Why this is unrecoverable in iter scope

Reverting the kernel restores the original 2-pass structure.  Achieving online softmax with parity would require:
- More careful memory barrier insertion (likely killing the merge benefit)
- Or, FlashAttention-style ENTIRE attention fusion (multi-day implementation, not iter scope)

## Categorization

**impl-fail / NEGATIVE** — fourth class of drift mechanism observed in iters 70-83:
1. FMA-emit divergence from kernel restructure (iter 50, 70, 73)
2. Ada cuBLAS BF16-TC stream saturation (iter 6, 58, 71)
3. L2/scheduler timing perturbation (iter 74)
4. **Memory consistency / scheduling at merged-pass kernels** (iter 83) — NEW

## Default policy

`causal_mask_softmax_kernel` reverted to original 2-pass + 1-sync structure.  No code change in tree.

## Sequence status (29 iters)

| iter | result | notes |
|---: |---     |---    |
| 70-82 | 13× non-PASS | engineering ceiling on single-axis |
| 82  | ratio sweep FAIL parity, wall ceiling at +37.6% | strategic finding |
| 83  | online softmax NEGATIVE | drift class #4 |

14 consecutive non-PASS iters (70-83).

## Files

- This document.
- `research/runs/2026-05-20-iter83-bench/iter83_online_softmax.log`.
- Code reverted to original kernel.
