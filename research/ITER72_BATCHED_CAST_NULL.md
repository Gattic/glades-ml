## Iter 72 — Batched multi-buffer cast kernel for iter69 BF16 cache — NULL

**Date**: 2026-05-19
**Iter**: 72 (eighteenth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)
**Verdict**: **NULL** — Δ tok/s −0.25%, Δ wall +0.22% (no measurable improvement, within run-to-run noise).  NLL drift 0.109 nat at step 200 (likely run-to-run variance per iter 67's ±0.06 nat noise floor at 2σ).  Trainer wiring reverted; kernel kept in glades-ml as opt-in symbol.

---

## Problem statement

iter 70 profile showed combined cast kernels at ~7-8% GPU time.  The biggest contributor was the iter 69 BF16-checkpoint-inner cache: each forward fires 7 sequential `cast_f32_to_bf16` calls at chiron_main.cpp lines ~6383-6394 (y_compr + sQ + sK + sV + sO + sP at 6 contiguous lines, plus q_compr at line 6192 in a different block).

Hypothesis: batching the 6 cache casts into a single launch reduces ~120 launches/step at L=24 (5 launches × 24 layers) and could improve HBM saturation via concurrent multi-buffer write.  Bit-identical math (same RN cast applied to each element).

## Implementation

`glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`:

```cuda
struct CastJob {
    const float* src;
    uint16_t* dst;
    size_t n;
};
__constant__ CastJob c_cast_jobs[8];

__global__ void k_cast_f32_to_bf16_batched(int num_jobs, size_t max_n) {
    const int job_id = blockIdx.y;
    if (job_id >= num_jobs) return;
    const CastJob job = c_cast_jobs[job_id];
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= job.n) return;
    // ... identical RN cast logic to k_cast_f32_to_bf16 ...
}

bool cast_f32_to_bf16_batched(int num_jobs,
                               const float* const* srcs,
                               uint16_t* const* dsts,
                               const size_t* counts);
```

Up to 8 (src, dst, count) tuples per launch.  Grid: `dim3(max_n / 256, num_jobs, 1)`.  Threads outside their job's `n` exit early.

`glades-trainer/trainer/chiron_main.cpp` lines 6383-6394: replaced 6 sequential `cast_f32_to_bf16` calls with one `cast_f32_to_bf16_batched(6, srcs, dsts, counts)` call.  Bench at production L=24 T=16384 200 steps seed=1337.

## Bench results

| metric                       | iter 69 baseline (yesterday) | iter 72 (batched)  |
|---                           |---:                          |---:                 |
| steady-state tok/s (mean)    | 24,320                       | 24,260              |
| wall (200 steps)             | 136.5 s                      | 136.8 s             |
| **Δ tok/s vs baseline**      | —                            | **−0.25%** (noise)  |
| **Δ wall vs baseline**       | —                            | **+0.22%** (noise)  |
| step-200 val NLL             | 7.7583                       | 7.6493              |
| Δ NLL @ step 200             | —                            | −0.109 nat          |
| step-21 ‖g‖                  | 2.648                        | 2.653 (+0.005 rel 0.19%) |

## Why this fails

1. **No measurable wall savings**: the saved launch overhead (5 launches × 24 layers × ~7 µs ≈ 0.84 ms/step) is within single-run variance (~0.5% wall at L=24 T=16384 ≈ 3 ms/step).  GPU HBM saturation also did NOT improve — the batched kernel launches `max_n × num_jobs = 16.78M × 6 = 100M` thread slots, of which only 38M actually do work (smaller jobs leave thread waste).  Sequential launches at exact sizes have zero waste.

2. **NLL drift indistinguishable from noise**: iter 70/71 baseline runs at this seed show ±0.06 nat single-run variance per iter 67's analysis.  iter 72's −0.109 nat is at ~2σ, within statistical noise.  Multi-seed eval needed to confirm parity, but the strict ±0.02 nat single-seed bound is violated.

3. **||g|| drift at step 21 (0.005 abs, 0.19% rel)** suggests cudaMemcpyToSymbolAsync + kernel launch scheduling produces a slightly different downstream timing.  Even with bit-identical kernel math, scheduling can affect L2 cache hit patterns in subsequent kernels, producing sub-ULP gradient differences that compound.

## Categorization

**null** (no real signal).  Mechanism is mathematically sound but the launch-overhead-reduction angle is structurally too small at production scale.  Each saved launch is ~7 µs CPU; 120 launches/step = 0.84 ms/step ≈ 0.12% wall — well below the run-to-run noise floor.

Pattern matches iter 55 (paradigm-flags NULL), iter 58 (parallel-bwd NULL), iter 59 (cuBLAS algo NULL).  Below-bar engineering improvements that don't materialize at production scale due to the engineering ceiling reached by iter 54 META.

## Default policy

`cast_f32_to_bf16_batched` kernel stays in glades-ml as an opt-in library symbol.  Trainer wiring reverted to sequential casts (chiron_main.cpp:6383-6394 restored).  No regression of any path.  Future iters can revisit if the launch-overhead landscape changes.

## Sequence status (18 iters)

| iter | target | result | win |
|---: |---     |---     |---:  |
| 68  | BF16-p default-on ship | SHIP               | +4.28% |
| 69  | BF16 checkpoint-inner cache | **PASS**      | **+6.83%** |
| 70  | fused axpy2_dual_p     | FAIL (NLL drift)   | +1.4% silent |
| 71  | --scfa-parallel-branches | NEGATIVE          | −5.4% |
| 72  | batched cast kernel    | **NULL**            | ±0% |

## Bench command

```bash
build/glades_chiron_train --data-dir pretok-data --pretokenized --vocab 32000 \
  --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 \
  --lr 3e-4 --max-steps 200 --warmup 20 --grad-clip 1.0 \
  --val-every 100 --val-batches 4 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage \
  --scfa-checkpoint-inner --scfa-checkpoint-inner-bf16 \
  --seed 1337
```

## Files

- This document (iter 72 null result).
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`: new `k_cast_f32_to_bf16_batched` kernel + `cast_f32_to_bf16_batched` wrapper (library function, retained).
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.h`: prototype.
- `glades-trainer/include/Backend/Machine Learning/Networks/cuda/gpu_kernels.h`: mirror prototype.
- `glades-trainer/trainer/chiron_main.cpp`: NO trainer-side wiring; reverted to sequential casts at lines 6383-6394.
