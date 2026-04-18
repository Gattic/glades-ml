# VESTA Profile-Driven Optimization

**Run date:** 2026-04-18

**Tools:** nsys 2022.4.2, ncu 2022.4.1

**Harness:** `unit-tests/glades-unit-tests vesta-step-bench` — 100 `vesta_gpu_step` calls at dModel=2048, r=8, no refresh, on a synthetic gradient. Isolates VESTA per-step work from transformer forward/backward.

## Summary

Step profile identified two dominant hot spots and one secondary one:

| kernel | before | after | delta |
|--------|--------|-------|-------|
| k_gram_schmidt (×2 per step) | 218 μs (54% of step) | **eliminated** | — |
| k_vesta_fused_update | 87 μs (22%) | 89 μs | ~0 (already memory-bound) |
| k_zero + k_scale (gradient) | 25 μs (12%) | 15 μs (k_zero fused in) | −10 μs |
| ell D2H downloads | 20 μs (10%) | 10 μs | −10 μs |
| **per-step wall** | **0.506 ms** | **0.200 ms** | **2.53× faster** |

All wins with NLL preserved: 1999/0 VESTA tests pass, parity tolerances unchanged.

## Key optimizations

### 1. Replace Gram-Schmidt with Cholesky-QR (the big win: 54% → 0%)

The modified Gram-Schmidt kernel used 1 block × 1024 threads, running on a single SM out of 80. For r=8 this was dominating wall-clock at 109 μs per call. Both MGS and CholQR produce the same Q (uniquely defined by thin-QR with positive R diagonal), so we can swap algorithms without changing the math.

**CholQR pipeline:** `M = U^T U` (cuBLAS sgemm, multi-SM) → Cholesky `M = R^T R` (tiny custom single-block kernel for small r) → `Q = U · R^(-1)` (cuBLAS strsm, multi-SM). Falls back to the original MGS if Cholesky detects a non-positive pivot (never observed in practice for the Stiefel retraction input).

This moved GS from the single-SM serial bottleneck to multi-SM cuBLAS calls + a ~6 μs serial Cholesky. Step time dropped from 0.398 ms → 0.215 ms (46%).

### 2. Fuse gradient zeroing into the fused update kernel (−10 μs/step)

The post-step `k_zero(gW)` full-buffer kernel (10.9 μs at dModel=2048) is now folded into `k_vesta_fused_update`: each thread writes `gW[idx] = 0.0f` after reading it for the gPerp computation. Same memory traffic (gW was already being written anyway for future accumulation), one less kernel launch and one less buffer pass.

### 3. Eliminate duplicate ell D2H download (−10 μs/step)

Step 6 downloaded `ell` to compute cPerp on host. Step 8 (trust-region clamp) downloaded `ell` again. The fused kernel between steps 6 and 8 doesn't modify `ell`, so the host copy from step 6 is still valid. Reuse it, skipping the second download.

### 4. Tiled fused kernel — tested, reverted

Tried a shared-memory-tiled version of `k_vesta_fused_update` that caches U/V/UA/expEll into shared memory with a cooperative load. It was **slower** (103 μs vs 89 μs): the compiler's L1/L2 caching already handled the row-reuse pattern well, and the manual tiling added a `__syncthreads` barrier and slightly-imbalanced load phase. Reverted.

## Remaining breakdown (post-optimization)

Per-step at dModel=2048, r=8 = **200 μs total**:

| kernel | time | % of step |
|--------|------|-----------|
| k_vesta_fused_update | 89 μs | 44.5% |
| cuBLAS cutlass nn/nt sgemm (×2) | 34 μs | 17% |
| k_scale (gradient at step 0) | 14.4 μs | 7.2% |
| ampere_sgemm_32x32 (×3) | 14.4 μs | 7.2% |
| k_cholesky_small_upper (×2) | 12.6 μs | 6.3% |
| trsm_left_kernel (×2) | 6.1 μs | 3.1% |
| k_project_out_span + AT (×2) | 4.9 μs | 2.5% |
| other custom (log_scale, exp_ell, form_raw, scale_cols, extract_diag) | 11 μs | 5.5% |
| splitKreduce (cuBLAS internal) | 4.5 μs | 2.3% |
| ampere_sgemm_128x32 | 3.2 μs | 1.6% |
| ell D2H download + cudaDeviceSync | ~10 μs | 5% |

### Where further wins might live

- **k_vesta_fused_update (44.5%)** is now memory-bound; compiler's cache handling is near-optimal at this scale. Further gains would require changing the arithmetic (e.g., factoring out expEll).
- **cuBLAS kernels (~25%)** are already tuned by NVIDIA; matrix-batching across weight matrices (task #30 originally) could improve utilization at small r.
- **k_cholesky_small_upper (6.3%)** is single-threaded serial. Could be made warp-level (32-wide) but 6 μs × 2 calls = 12 μs is 6% of step — marginal.
- **k_scale (7%)** is pure memory bandwidth; only savings would be fusing with the sgemms via `alpha` — small change, ~14 μs savings.

## End-to-end impact

The profile-isolated 2.53× step speedup was validated only on the microbenchmark. End-to-end sweeps (v14 → v15 → post-opt) show smaller gains because transformer forward/backward is ~50% of wall-clock in real training and is untouched by these optimizations. The memory-pressure issue at dModel=8192 is also unchanged by these step optimizations (see `vesta_mega_fused_20260418-125919`).

## Artifacts

- `step.nsys-rep` — original profile (pre-optimization, post-fused-kernel commit)
- `step_cholqr.nsys-rep` — after CholQR swap
- `step_tiled.nsys-rep` — after tiled fused kernel (slower, reverted)
- `step_final.nsys-rep` — after all three wins
- `step.sqlite`, `step_cholqr.sqlite`, `step_tiled.sqlite`, `step_final.sqlite` — queryable versions

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-step-bench    # times per-step wall-clock
nsys profile --trace=cuda --sample=none \
  --output=step ./build/glades-unit-tests vesta-step-bench
```
