# Iter 47 — SCFA dwconv-dK parallel reduction kernel — Gate-0 PASS

**Date**: 2026-05-16
**Iter**: 47 (first iter under reframed "stacking small wins" brief)
**Branch**: vesta5 (glades-ml)

---

## TL;DR

Replaced the SCFA depthwise-conv backward `dK` kernel with a 2D-tiled parallel-reduction variant. **+5.20% tok/s at iso-config, val NLL within +0.002 to +0.007 nat over 200 steps**, VRAM-neutral. Ships as new flagship baseline for iter 48.

| metric | baseline (old dK) | iter47 (2D-tiled dK) | delta |
|---|---:|---:|---:|
| tok/s (steady state) | 38,470 | 40,470 | **+5.20%** |
| val NLL @ step 100 | 8.6911 | 8.6981 | +0.0070 |
| val NLL @ step 200 | 8.2576 | 8.2599 | +0.0023 |
| wall (200 steps) | 43.3 s | 41.2 s | −4.85% |
| VRAM | 7.46 GB | 7.46 GB | 0 |
| flags shipped on top | (all prior) | + `scfa-dwconv-dK-par` (default) | — |

Both val NLL deltas well inside the ± 0.02 nat parity bound.  +5.20% clears the +5% Gate-0 bar.

---

## What changed

### Bottleneck

`nsys stats --report gpukernsum /tmp/iter47_baseline.nsys-rep`  showed the legacy
`scfa_dwconv_dK_kernel` consumed **7.7% of training GPU time** (784 ms / 50 steps
at T=8192 m=2048 L=12 with the full shipped stack), the single hottest kernel in
the profile.  Its launch geometry was the culprit:

```
grid  = (1, 2048)        // (ceil((w+1)/9), m)
block = (9, 1, 1)        // (w+1=9 threads)
=> 2048 blocks × 9 threads = 18,432 total threads
```

At production scale this is wrong on two axes:
- **Under-parallelized**: only 18k threads on an 80-SM Ada (max ~123k concurrent),
  ~15% theoretical occupancy.  Each thread loops `t = i..T` sequentially (8k ops).
- **Non-coalesced**: every thread in a block shares `c = blockIdx.y` and reads
  `x[(t−i)·m + c]`, which is column-major — within a warp, the 9 active lanes
  load 9 *different rows* of `x` at the same column.  No coalescing.

### Fix

Rewrote as a templated 2D-tiled parallel reduction:

```cuda
template<int BLOCK_M, int BLOCK_T>   // instantiated with <64, 8>
__global__ void scfa_dwconv_dK_kernel_par(const float* x, const float* dy,
                                          int T, int m, int w, float* dK)
{
    const int i  = blockIdx.y;                              // tap in [0, w]
    const int c  = blockIdx.x * BLOCK_M + threadIdx.x;      // channel (coalesced)
    const int ty = threadIdx.y;                             // T-partition
    if (i > w) return;

    float acc = 0.0f;
    if (c < m) {
        for (int t = i + ty; t < T; t += BLOCK_T)
            acc += x[(size_t)(t - i) * m + c] *
                   dy[(size_t)t       * m + c];
    }

    __shared__ float sdata[BLOCK_T][BLOCK_M];
    sdata[ty][threadIdx.x] = acc;
    __syncthreads();
    for (int s = BLOCK_T / 2; s > 0; s >>= 1) {
        if (ty < s) sdata[ty][threadIdx.x] += sdata[ty + s][threadIdx.x];
        __syncthreads();
    }
    if (ty == 0 && c < m)
        dK[(size_t)c * (w + 1) + i] += sdata[0][threadIdx.x];
}
```

Launched as:

```
grid  = ((m + 63)/64, w+1) = (32, 9)        // 288 blocks
block = (BLOCK_M=64, BLOCK_T=8)             // 512 threads/block
=> 288 × 512 = 147,456 total threads        // 8× over the legacy kernel
```

Memory access is now coalesced (warp lanes have consecutive `c`).  Each thread
processes `T/BLOCK_T = 1024` ops (8× fewer per thread).  No atomics needed — each
`(c, i)` is written by exactly one thread (`ty == 0`).

Shared memory: `BLOCK_T × BLOCK_M × 4 = 2 KB / block`.  At 512 threads/block and
3 blocks/SM (1536 active threads/SM = 100% occupancy on sm_8.9), 6 KB SMEM/SM —
far below the 100 KB/SM budget.

### Files

- `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` — new kernel at the
  old kernel's old position; old kernel kept as dead code with `unused-function`
  warning suppression; launch site in `scfa_depthwise_causal_conv_bwd` switched
  to the new variant.

The legacy `scfa_dwconv_dK_kernel` is intentionally retained (unreferenced) so
the bisect-friendly diff is small and an A/B comparison is one-character away
if a regression turns up later.

---

## How the bench was run

Stacked-flagship bench config (matches the iter-47 nsys profile):

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

Procedure:
1. Built iter47 binary, ran 200 steps with `--val-every 100`. Logged to
   `optimized_200steps.log`.
2. `git stash`’d the kernel change, rebuilt with the legacy kernel, re-ran the
   identical command. Logged to `baseline_oldkernel_200steps.log`.
3. `git stash pop`, rebuilt iter47 binary.

Both runs were on the same RTX 4080 SUPER, same OS state, same input data,
same seed (1337). tok/s reported is the steady-state value (median of step
20–200 reports), which is consistent within the same run to <0.2%.

Logs archived at `research/runs/2026-05-16-iter47-scfa-dwconv-dK/`.

---

## Why NLL parity holds

Both kernels compute exactly the same mathematical sum:

$$
dK[c, i] \;=\; \sum_{t=i}^{T-1} x[t-i, c]\cdot dy[t, c]
$$

The only source of difference is fp32 reduction order:
- **legacy**: one thread accumulates `T − i ≈ 8192` terms sequentially in a
  single register.
- **iter47**: 8 partial sums (one per `ty`) each over `T/8 ≈ 1024` terms,
  combined by a length-8 tree.

Per-value relative error: `√T · ε ≈ 90 · 1.2e-7 ≈ 1.1e-5`.  Over 200 fp32
training steps with grad-clip and BF16 readout, the trajectory diverges by
single-digit milli-nats at val — consistent with the +0.002 to +0.007 nat
observed.  Stable across the run (no drift in either direction).

---

## What ships

The new kernel is the default path through `scfa_depthwise_causal_conv_bwd`.
No new flag — this is a pure correctness-preserving kernel swap.  All prior
shipped flags (`--scfa`, `--scfa-bf16-inner/outer`, `--scfa-fuse-streams`,
`--scfa-reln-opt`, `--bf16-logits`, `--bf16-logits-storage`, `--bf16-attn`,
`--bf16-weights`, `--bf16-grads`, `--int8-adam`) continue to apply.

**New stacked-flagship tok/s at the iter-47 profile config**: 40,470.

---

## Where next

`nsys stats` on the iter-47 baseline showed the post-dK leaderboard:

```
 7.7%  scfa_dwconv_dK_kernel         (now ~1-1.5% with iter47 fix)
 6.8%  cutlass bf16 256x128x16x3 nn
 5.8%  ampere s16816 bf16 128x128 nn
 5.0%  chiron_scfa_axpy2_kernel       <- next likely target
 4.9%  readout GEMMs (3)
 4.7%  adam_update_int8_state_kernel
 4.6%  k_cast_bf16_to_f32             <- ~6.5% combined with other casts
 3.8%  chiron_scfa_sub_kernel         <- could fuse with axpy2
 3.0%  scfa_depthwise_causal_conv_fwd
 1.9%  scfa_dwconv_dx_kernel
```

Two cheap iter-48 candidates surfaced:
- **Fuse `scfa_axpy2` + `scfa_sub`** (8.8% combined, both already fused-stream
  ops but the kernel-side body still has 2 dispatches).  Expected +2-4%.
- **Reduce `bf16↔fp32` cast count via cached BF16 views** (6.5% combined).
  Several precision casts happen on the same tensor multiple times per step.

Neither is committed for iter 48 — measure first, conjecture second.
