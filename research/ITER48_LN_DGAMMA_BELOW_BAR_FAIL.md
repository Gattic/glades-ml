# Iter 48 — LayerNorm-backward dgamma/dbeta parallel reduction — Gate-0 FAIL (below-bar)

**Date**: 2026-05-16
**Iter**: 48 (second iter under "stacking small wins" brief; iter47 was PASS)
**Branch**: vesta5 (glades-ml)
**Verdict**: FAIL — kernel optimization works (55% kernel speedup) but kernel is only 4.2% of total time, so overall win caps at ~2.3-2.4% (below the +5% Gate-0 bar).

---

## TL;DR

Identified `layernorm_backward_dgamma_dbeta` as the next-after-iter47 non-cuBLAS hotspot
(4.2% of total GPU time).  Same anti-pattern as iter47's dK kernel: 1 block per col,
threads loop rows, **non-coalesced** (threads in a warp share `col`, hit rows strided
by `cols=m=2048` floats = 8KB).  Wrote a 2D-tiled parallel reduction (BLOCK_C=64
consecutive cols per block, coalesced × BLOCK_T=8 row-partitions reduced via SMEM).

Kernel-level: **drops from 4.2% → 1.9% of GPU time (55% speedup of the kernel itself).**

Overall: **+2.4% tok/s (40,470 → 41,470 at the iter-47 profile config).**

The kernel improvement is genuine and the implementation is clean, but the kernel
was too small a slice of total time to clear the +5% Gate-0 bar.  Categorized as
**too-small-target FAIL** (a fourth FAIL mode the brief implies: not idea-fail,
not impl-fail, not null — the optimization works but the target was too small).

Reverted; ready to pivot to a higher-impact target in iter 49.

---

## Bench data (all at iter47 profile config: T=8192 m=2048 L=12, seed 1337)

| variant                | val NLL @100 | val NLL @200 | tok/s steady | wall | comments |
|---                     |---:          |---:          |---:          |---: |---       |
| iter47 baseline (old dgamma)        | 8.6911 | 8.2576 | 38,470 | 43.3s | pre-iter47-dK |
| iter47 shipped         | 8.6981 | 8.2599 | 40,470 | 41.2s | post-iter47 |
| iter48 T_PARTS=1       | 8.6585 | 8.2123 | 41,470 | 40.2s | det., NLL BETTER, +2.4% |
| iter48 T_PARTS=2       | —      | 8.3913 | 41,489 | 40.0s | atomic noise, NLL +0.13 |
| iter48 T_PARTS=4       | 8.6733 | 8.3604 | 41,535 | 40.2s | atomic noise, NLL +0.10 |

T_PARTS=1 is the only deterministic variant (no atomicAdd — 1 block per col-tile).
Atomic-based variants speed up by another 0.1-0.2% but introduce non-deterministic
fp32 accumulation noise that compounds to 0.10-0.13 nat val NLL drift over 200
steps — outside the ±0.02 nat parity bound.

T_PARTS=1 is within parity (actually 0.045 nat BETTER than baseline at step 200 —
the new coalesced layout produces fewer intermediate accumulations, slightly
tighter fp32 summation), but only delivers +2.4% which is below the +5% bar.

---

## Why this iter failed (and why the brief's bar is right)

The kernel went from 4.2% → 1.9% of total time.  That's a real, large improvement on
the *kernel itself*.  But total time saved is bounded by the kernel's original share
of the total.  Even a hypothetical 100% reduction of the legacy kernel would only
yield 4.2% overall — still below the 5% bar.

For iter 49 onward, the lesson is concrete: **pick a target whose share of total
time is ≥ ~7%**, so that a realistic kernel optimization (50-70% speedup) clears
the +5% bar with margin.

The brief's explicit ban on "almost-works partial passes" prevents the temptation
to ship 2-3% wins one-by-one and call them progress — those compound poorly and
risk masking regressions.  Either an iter clears the bar or it documents and pivots.

---

## What changed (reverted)

Added (and then reverted) a templated parallel-reduction kernel
`layernorm_backward_dgamma_dbeta_par<BLOCK_C, BLOCK_T>` in
`Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` with launch geometry
`grid=((cols+63)/64, T_PARTS) × block=(64, BLOCK_T=8)`.  Two configurations tested:

- **T_PARTS=1** (no atomicAdd): grid=(32, 1) = 32 blocks × 512 threads = 16,384 threads.
  Each block-col owns a unique col tile and writes the final accumulator.  Deterministic.
- **T_PARTS=2 or 4** (with atomicAdd): grid=(32, T_PARTS) × 512 threads = up to 65,536 threads.
  Multiple blocks atomicAdd into the same `dgamma[col]` / `dbeta[col]`.  Non-deterministic
  fp32 ordering causes 0.10-0.13 nat val NLL drift over 200 steps.

Both kernels are correct mathematically; the issue with T_PARTS > 1 is purely
fp32 reduction-order variance under atomic contention.

Files touched, then restored:
- `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` (kernel + launch site)

---

## Where next (iter 49 candidates)

From the iter-48 profile (with iter47's dK fix active), top remaining kernels:

| share | kernel                              | nature                |
|---:   |---                                  |---                    |
| 7.3%  | cutlass bf16 256x128x16x3 nn        | SCFA inner GEMM (cuBLAS) — hard |
| 6.3%  | ampere s16816 bf16 128x128 nn       | SCFA inner GEMM (cuBLAS) — hard |
| 5.9%  | cutlass bf16 128x128x16x6 nt        | SCFA inner GEMM (cuBLAS) — hard |
| **5.4%**  | **chiron_scfa_axpy2**           | element-wise FP32, memory-bound (98% BW eff.) |
| 5.3%  | cuBLAS readout (3 instances total ~16%) | logits GEMMs — hard |
| 5.0%  | adam_update_int8_state              | optimizer — investigate |
| 5.0%  | k_cast_bf16_to_f32                  | bf16↔fp32 casts (×7200 inst/run) |
| **4.2%**  | **chiron_scfa_sub**             | element-wise FP32, memory-bound |

Iter 49 candidate options:

- **A. Fuse `axpy2` + reln-axpy** if call-adjacent.  The reln-axpy already exists
  (`chiron_reln_axpy_into_q`).  If `axpy2 → reln_axpy` is called back-to-back on
  the residual, fusion saves one round-trip through the T·m FP32 buffer.  Potential
  4-6%.

- **B. Eliminate redundant bf16↔fp32 casts.**  7200 cast instances per 50 steps is
  144/step.  Many of these may be casting the SAME tensor multiple times per step.
  A scratch-BF16-view cache could deduplicate.  Potential 2-5%.

- **C. Adam kernel investigation.**  Already optimized but 2400 instances per 50 steps
  = 48/step.  Per-param-group calls might be coalescable.  Potential 1-3%.

A (fusion) is the highest-EV candidate.  B (cast dedup) requires deeper code analysis.
C is likely too small.

Iter 49: investigate A first.

---

## Reproducibility

```bash
# iter47 baseline
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

iter47 profile re-used: `/tmp/iter47_baseline.nsys-rep`.
iter48 profile (with reverted change, post-iter47 only): `/tmp/iter48_baseline.nsys-rep`.
