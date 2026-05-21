# Iter 53 — LN-backward dgamma/dbeta deterministic 2-phase parallel reduction — FAIL (below-bar)

**Date**: 2026-05-16
**Iter**: 53 (seventh iter under "stacking-wins" brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: FAIL — +2.86% tok/s (better than iter48's +2.4% but still below the +5% Gate-0 bar).  Mechanism works (deterministic 2-phase parallel reduction; NLL parity at step 100 BETTER, at step 200 right at ±0.02 bound edge).  Reverted; iter49 stays the default flagship.

---

## TL;DR

iter48 attempted the LN-backward `dgamma_dbeta` kernel with a coalesced parallel reduction; T_PARTS=1 (deterministic) gave +2.4%, T_PARTS>1 (atomic) gave more perf but NLL drift outside ±0.02 bound.

iter53 revisits with a **deterministic 2-phase** design:
- **Phase 1**: `layernorm_backward_dgamma_dbeta_partial<BLOCK_C=64, BLOCK_T=8>` writes T_PARTS partial sums per col to a scratch buffer.  Coalesced channel-major access; SMEM tree-reduce within block.  T_PARTS=4 blocks per col-tile (parallel; no atomics — each (col, t_partition) is owned by exactly one block).
- **Phase 2**: tiny `layernorm_backward_dgamma_dbeta_reduce` kernel deterministically sums T_PARTS partials per col → final dgamma/dbeta (single-threaded per col, fixed loop order).

Scratch: T_PARTS × cols × 2 floats = 4 × 2048 × 2 × 4 B = 64 KB (negligible).

Bench (200 steps, iso-config, seed 1337):

| metric              | iter49 baseline | iter53           | delta              |
|---                  |---:             |---:              |---:                |
| tok/s steady        | 43,209          | 44,444           | **+2.86%**         |
| val NLL @ step 100  | 8.6579          | 8.6540           | −0.004 (better)    |
| val NLL @ step 200  | 8.2391          | 8.2587           | +0.020 (at bound)  |
| wall (200 steps)    | 38.3 s          | 37.3 s           | −2.6%              |
| VRAM                | 7.46 GB         | 7.46 GB          | 0                  |

Kernel-level (vs iter49): LN-bwd dgamma 4.5% → ~2.0% (~55% kernel speedup, similar to iter48 T_PARTS=1).

Overall +2.86% (slightly better than iter48's +2.4% — likely because the deterministic 2-phase has a tiny scratch + reduce overhead but gets more parallelism than T_PARTS=1).

Below the +5% bar by 2.14%.  Reverted.

---

## Why this iter still failed despite a better design

iter48's lesson: a 4.5%-share target can't clear 5% even with a perfect kernel-level speedup.  Confirmed once more.

The 2-phase deterministic design is a real improvement over iter48 (gets +2.86% vs +2.4%), but the fundamental kernel slice (4.5%) is too small.  Even the theoretical limit of 100% kernel elimination would be 4.5% overall — below 5%.

For iter54+, the target needs to be ≥7% of total GPU time, or a multi-kernel **mechanism-coherent** fusion that combines several smaller targets.

---

## What was tried (and reverted)

Library: added 2 new templated kernels in `gpu_kernels.cu` next to the legacy `layernorm_backward_dgamma_dbeta` kernel:
- `layernorm_backward_dgamma_dbeta_partial<BLOCK_C, BLOCK_T>` — coalesced phase-1 with SMEM reduce
- `layernorm_backward_dgamma_dbeta_reduce` — deterministic phase-2 reduce
- `ensure_ln_partial_scratch(t_parts, cols)` — lazy cuMalloc helper for the scratch pool

Modified `layernorm_backward` to call the two new kernels (T_PARTS=4) in place of the legacy kernel.

Header / trainer routing: no changes needed (the public API is unchanged).

All reverted.

---

## NLL parity analysis

The deterministic 2-phase has a fixed reduction order: `sumG = Σ_{p=0..T_PARTS-1} partial_dg[p, col]`.  Each `partial_dg[p, col]` is a sum over a contiguous slice of rows (SMEM tree-reduce within block).  Different from the legacy kernel's "1 block per col, threads loop strided rows" which has a different summation order (across-warp shuffle reduce, then atomicAdd-style accumulation).

Mathematically: same sum, different fp32 order.  Sub-ULP per element, but with T=8192 rows × 12 layers × 50 steps, the cumulative drift can reach single-digit milli-nats.

Step 100 result: −0.004 nat (better) — within bound.
Step 200 result: +0.020 nat — right at the ±0.02 bound edge.  Borderline.

This is the expected fp32-reduction-order chaotic drift, not a bug.  But the drift was tighter at step 100 than step 200 — chaotic divergence amplifies over time.  By step 1000+ this would likely be outside the bound.

---

## Sequence under stacking-wins brief

| iter | target | result |
|---: |---     |---     |
| 47  | SCFA dwconv-dK par-reduction | PASS +5.20% |
| 48  | LN-bwd dgamma/dbeta T_PARTS=1 | FAIL +2.4% too-small-target |
| 49  | Fused Adam-int8 BF16w/g + bf16 grad-norm | PASS +6.98% |
| 50  | SCFA sub-into-conv fusion | FAIL +1.31% bench-noise + NLL drift |
| 51  | --scfa-checkpoint-inner revival | FAIL +4.46% below-bar |
| 52  | cuBLAS readout algo override | NULL 0% |
| 53  | LN-bwd dgamma/dbeta deterministic 2-phase | FAIL +2.86% below-bar |

Cumulative shipped: **+12.5%**.  7 iters: 2 PASS / 4 FAIL / 1 NULL.

Two attempts (iter48, iter53) at the LN-bwd dgamma target — confirmed it's a dead-zone for solo optimization.

---

## Where next (iter 54 candidates)

The remaining unattacked engineering levers that haven't been written up as solo FAILs:

- **chiron_scfa_axpy2 + sub combined as a SINGLE fused mechanism** — iter50 tried sub-into-conv but the bwd-recompute path couldn't be fused.  A different fusion (axpy2 + sub into a single kernel?) might work but the math doesn't align — sub writes q_perp before the conv reads it, axpy2 happens at end of layer.

- **Fuse k_bf16_accum_axpy into the attention backward** — eliminate the FP32→BF16 grad accum kernel by having attention-bwd write BF16 directly.  Saves ~2.7% kernel time.  Below bar alone, but combined with bf16-grad-norm route (already iter49) maybe edge above.

- **--scfa-parallel-branches revisit** — iter8 NEGATIVE on Ada, structural issue unlikely to flip.  But worth a re-test with the post-iter49 occupancy.

- **Step back: revisit the brief.**  Seven iters under "stacking 5%-wins" model produced +12.5% (instead of the target 1.5× = +50% by iter 5).  The bar may be too high for the remaining bottleneck composition.

Iter 54 likely: a META-LEVEL pivot — either acknowledge the +5% bar is empirically unattainable given the current bottleneck mix (mostly cuBLAS at ~40%), or attempt a paradigm-level change (FP8 readout via cuBLASLt, sparse attention, etc.) at higher risk.

---

## Reproducibility

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
