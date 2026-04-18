# VESTA Realistic-Scale Profile: dModel=4096 nLayers=4

**Run date:** 2026-04-18

**Tool:** nsys 2022.4.2

**Harness:** `unit-tests/glades-unit-tests vesta-profile-bench` — 3 epochs of full transformer training at dModel=4096, nLayers=4, corpusLen=512, VESTA r=8. Realistic scale (matches v13 ultra sweep config) but short horizon for profiling.

## The big finding

**At dModel=4096 with nLayers=4, flash attention dominates 98% of GPU kernel time.** All of VESTA's optimizer kernels together contribute <1%. Any VESTA step-code optimization has imperceptible end-to-end impact at this scale.

### GPU kernel breakdown

| kernel | % of GPU time | notes |
|--------|---------------|-------|
| flash_attention_bwd | 62.3% | 12.6s for 12 calls = 4 layers × 3 steps |
| flash_attention_fwd | 36.5% | 7.4s |
| k_vesta_fused_update (3 grid-shape variants) | 0.3% | 51ms across 48 calls |
| cutlass sgemm (transformer + VESTA) | ~0.6% | many small + 3 large variants |
| k_gram_schmidt | 0.08% | 16ms (refresh + CholQR fallbacks) |
| k_scale (gradient scaling) | 0.05% | |
| k_cholesky_small_upper | 0.004% | 0.8ms across 150 calls |
| other custom | <0.1% each | |

### Host API breakdown (much more revealing)

| API | total time | calls | median | reason |
|-----|------------|-------|--------|--------|
| cudaDeviceSynchronize (old code) | 8.5s (38%) | 75 | 543 μs | end-of-step sync |
| cudaMemcpy (synchronous) | 6s (27%) | 792 | 22 μs | mostly ell downloads + init outlier |
| cudaMemcpyAsync | 5s (22%) | 237 | 5.7 μs | |
| cudaLaunchKernel | 2.5s (11%) | 2859 | 3.8 μs | GPU submission queue saturated |

The VESTA-vs-AdamW wall-clock delta at this scale is dominated by these **host-side API calls**, not by kernel execution.

## Fixes applied after profile analysis

### 1. Regularized Cholesky (45% → negligible failure rate)

Original `k_cholesky_small_upper` failed on non-positive pivots and the caller fell back to MGS. Observed 45% failure rate on real training (URaw becomes poorly conditioned when invExpEll gets large).

Added running-diagonal regularization: `diag = max(diag, 1e-6 * trace(M) / r)` before sqrt. Cholesky now always produces usable output; for rank-deficient cases it degrades smoothly to an MGS-equivalent result. **Eliminates 67 fallback-to-MGS calls per 3-epoch run.**

### 2. All VESTA kernels on `computeStream` + removed end-of-step `cudaDeviceSynchronize`

Previously custom VESTA kernels used the default stream while cuBLAS used `computeStream`. The default stream is `cudaStreamNonBlocking` w.r.t. compute stream — they race. We papered over with an end-of-step `cudaDeviceSynchronize()` (comment in code noted "this produced NaN without the sync at dModel=2048").

Moved **all** VESTA custom kernel launches + `cudaMemcpyAsync`s to `computeStream`, matching cuBLAS. Now everything within one `vesta_gpu_step` call, everything across consecutive step calls, and the transformer's forward/backward, all share `computeStream`. Naturally serialized — no explicit sync needed.

Removed `cudaDeviceSynchronize()` at end of every step. Previously 1 call per weight matrix per step (16 × 50 = 800 calls per 50-epoch sweep). Replaced with the implicit ordering of same-stream kernel launches.

### 3. Dropped CholQR host status check

Regularized Cholesky always succeeds, so the synchronous `cudaMemcpy(d_status → host_status)` check and MGS fallback branch are unreachable in practice. Removed. Saves 2 synchronous D2H syncs per weight matrix per step.

## Verification

- VESTA unit tests: **1999/0** (parity preserved)
- GpuParityTest W maxAbs: unchanged (~9e-7)
- GpuSingleRefreshTest U/V: unchanged (~7e-7)
- Step microbenchmark at dModel=2048 r=8: **0.196 ms/step** (was 0.200 ms — tiny improvement, consistent with the microbenchmark not being sync-bound)

## What's left on the table

At dModel=4096 nLayers=4, **flash attention is the actual limit**. Further VESTA step optimization is essentially free of end-to-end impact. To push further, we'd need:

1. **Optimize flash attention itself** — shared with AdamW, not a VESTA-specific win.
2. **Overlap VESTA step with next-batch forward pass** — potentially 2× if we can run VESTA's per-matrix updates concurrently with the next iteration's forward. Requires pipelined scheduling and changes in `sgd_transformer.cpp` to move optimizer work off the critical path. Major refactor.
3. **Batch per-matrix VESTA updates across weight matrices in one step** — the original "matrix batching" idea from task #30. Would consolidate ~16 sequential per-matrix steps into 1-4 batched calls. Potential 30-50% reduction in VESTA kernel launch overhead, but launch overhead is already <12% of total at this scale.
4. **Use persistent CUDA graphs** — capture the per-step kernel sequence once and replay, eliminating per-launch overhead. Could reduce the 2.5s `cudaLaunchKernel` cost at dModel=4096 by 5-10×.

Option 4 is the most promising for a focused VESTA-only improvement at scale. But the rank-ordering of wins is now:
1. Flash attention optimization: 1.91× (full system win)
2. CUDA graphs: maybe 1.1-1.2× (VESTA step + launch overhead)
3. Matrix batching: maybe 1.05× (launch overhead portion)
4. Step-code micro-optimization: <1.01× (already squeezed)

## Artifacts

- `real.nsys-rep` — pre-fix profile (baseline)
- `real_reg.nsys-rep` — after Cholesky regularization
- `real_streamfix.nsys-rep` — after all three stream/sync fixes

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
nsys profile --trace=cuda --sample=none \
  --output=real ./build/glades-unit-tests vesta-profile-bench
```

Expected wall: 3-6 minutes for 3 epochs at dModel=4096 on 16 GiB GPU (flash attention-dominated).
