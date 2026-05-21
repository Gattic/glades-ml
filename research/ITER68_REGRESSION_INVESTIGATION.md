# ITER 68 — +0.50 nat NLL Regression Investigation

**Date**: 2026-05-17. Code review only — diagnostic owns GPU.
**Context**: Same seed/config/step-1 NLL (10.4765) yields ema 4.29 (2026-05-14
flagship) vs ema 4.79-4.82 (post-iter47 binary). Diagnostic tentatively cleared
iter 65 (BF16-residual-p) and iter 69 (BF16 checkpoint-inner cache).

---

## 1. Git archaeology

### Historical-flagship binary state (2026-05-14)
| repo | commit | date |
|---|---|---|
| glades-ml | `75ba7fddc` | 2026-05-14 16:51 |
| glades-trainer | `9793c4e` | 2026-05-14 16:51 |

### Ship-to-commit table (post-flagship default-on changes)
| ship | repo | commit | date | one-line |
|---|---|---|---|---|
| iter 47 | glades-ml | `4948bdc17` | 05-16 07:05 | SCFA dwconv-dK parallel reduction (atomicAdd → deterministic) |
| iter 49 | glades-ml | `41259f144` | 05-16 07:45 | fused int8-Adam BF16-w + BF16-g inline I/O |
| iter 49 | trainer | `f5f235b` | 05-16 07:45 | route int8-Adam → fused; bf16 grad-norm |
| iter 53 | glades-ml | (in `7556eb454`) | 05-16 09:18 | LN-bwd dgamma/dbeta 2-phase deterministic |
| iter 56 | glades-ml | (in `7556eb454`) | 05-16 09:18 | warp-parallel argmax_count (logging-only) |
| iter 60 | trainer | `1388f46` | 05-16 09:18 | scfaCheckpointInner default `false → true` |
| iter 61 | glades-ml | `cf0ab5e3d` | 05-16 09:50 | new `sgemm_rowmajor_atb_bf16_dst_bf16` + bf16w_bf16g shear-bwd |
| iter 61 | trainer | `13a9a3f` | 05-16 09:50 | silent default-on at `bf16Inner && bf16Grads` |
| iter 65 | trainer | `745f113` | 05-16 12:55 | bf16-residual-p full routing (CLEARED) |
| iter 69 | trainer | `432202d` | 05-16 15:15 | bf16 checkpoint-inner cache (CLEARED) |

---

## 2. Per-ship mechanism review

### iter 47 — SCFA dwconv-dK parallel reduction
**Suspicion: ★★☆☆☆**

Legacy `scfa_dwconv_dK_kernel` used `atomicAdd(&dK[c*(w+1)+i], acc)` — run-to-run
non-deterministic FP32 reduction order. New `scfa_dwconv_dK_kernel_par` uses
`dK[c*(w+1)+i] += sdata[0][tx]` (no atomic; exactly one block-row per (c,i)) —
fully deterministic. Same set of FMA terms; only reduction order differs.

The legacy was ALREADY non-deterministic, so the historical flagship trained
with run-to-run noise. The change is "noisy order" → "fixed order"; both
unbiased. Unlikely to systematically bias gradients.

### iter 49 — Fused int8-Adam BF16-w + BF16-g
**Suspicion: ★★★☆☆**

Replaces 4-kernel chain (cast bf16→fp32 param + grad + adam + SR cast back)
with one fused kernel. Inline BF16 decode is bit-exact (`<<16`). Same
`sr_hash32(idx, srStepIdx, srBaseSeed)` SR-RNG for param encode. m/v moment
math identical. Code-level inspection: Pass 1 writes (mNew, vNew) to SMEM;
Pass 2 reads from SMEM and reloads same `i` per thread. No data race, no
register leakage between passes.

`compute_grad_norm` change: legacy `cast_bf16_to_f32 + sum_squared_accumulate`
vs new `sum_squared_accumulate_bf16`. Both use `atomicAdd(acc, blockReduceSum())`
in FP32 — both non-deterministic; mathematically equivalent.

Math should be bit-equivalent. If this is the bug, it's a subtle ordering
quirk not visible in source. Lower priority than iter 60/61.

### iter 53 — LN-bwd dgamma/dbeta 2-phase
**Suspicion: ★★☆☆☆**

Legacy: 1 block per col, `blockReduceSum` + single thread `dgamma[col] += localDg`
(deterministic). New: T_PARTS=4 partitions of rows, per-partition tile-reduce
to scratch (no atomics), then fixed-order phase-2 sum across 4 partials.
Production confirmed T_PARTS=4 — does NOT use the atomicAdd pattern that
caused iter 48's 0.10-0.13 nat drift.

Both deterministic, different reduction order, sub-ULP per element. iter 53's
own bench showed NLL drift +0.020 at step 200 (parity-bound edge); could
contribute to compounded regression but not dominant.

### iter 56 — warp-parallel argmax_count_bf16
**Suspicion: ★☆☆☆☆**

Pure logging path (train-accuracy top-1). No write to weights, no impact on
backward. NLL bit-identical guaranteed by construction.

### iter 60 — `scfaCheckpointInner` default `false → true`
**Suspicion: ★★★★☆**

The ONLY ship in the 47-61 window that changes the gradient APPLIED to
weights. Historical flagship trained with checkpoint OFF.

**Without checkpoint** (historical):
- Forward computes sQ/sK/sV/sO/sP via cuBLAS BF16-TC GEMMs.
- Backward RE-RUNS the same `chiron_attention_shear_*_tiled` to produce fresh
  sQ/sK/sV/sO/sP.
- Tensor-core GEMM is run-to-run non-deterministic; the recomputed forward
  differs sub-ULP from the original forward used for the loss.
- Gradient computed against this slightly-different recomputed forward → small
  forward-loss / backward-grad math inconsistency.

**With checkpoint** (current default):
- Forward writes sQ/sK/sV via `device_memcpy_d2d` to `*_save[l]` (bit-exact).
- Backward reads `*_save[l]` directly → gradient computed against EXACT
  forward used for loss.
- More mathematically consistent — but trajectory diverges from historical
  run which had been optimizing on the "noisy" recompute path.

Both paths produce valid gradients; the OPTIMIZER explores different regions
of weight space. 30k steps suffices for the trajectories to diverge by O(0.5 nat).

### iter 61 — BF16-grad direct cuBLAS output
**Suspicion: ★★★★★ — TOP SUSPECT**

Replaces the legacy weight-grad commit chain:
1. `sgemm_rowmajor_atb_bf16(D=FP32)` → dW into FP32 scratch.
2. `k_bf16_accum_axpy`: read BF16 accumulator, widen FP32, compute
   `bf16(α·src + β·dst)`, write BF16 with EXPLICIT round-to-nearest-EVEN:
```cpp
const uint32_t lsb = (v.u >> 16) & 1u;
const uint32_t roundingBias = 0x7FFFu + lsb;
dst_bf16[idx] = (uint16_t)((v.u + roundingBias) >> 16);
```

with:
1. `sgemm_rowmajor_atb_bf16_dst_bf16(D=BF16, β=1)` — cuBLAS gemmEx with
   `D_type=CUDA_R_16BF`, `CUBLAS_COMPUTE_32F_FAST_16BF`,
   `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.

Silent default-on at `useFastBf16Grad = useBf16Inner && bf16Grads` since
2026-05-16 09:50 — fires on every shipped stack since iter 60.

**The critical question**: cuBLAS's BF16-store rounding mode. The legacy
kernel uses provably IEEE-correct round-half-to-EVEN (ties to even, zero
expected bias). cuBLAS documents BF16 store as "round-to-nearest" but does
**not guarantee round-half-to-EVEN** — implementations may use round-half-
away-from-zero, or apply rounding inside the tensor-core accumulator with
a different regime than the explicit kernel. The rounding rule is
shape/algo/version-dependent.

Even if both are RN-even, the tensor-core MMA internally has FP32 mul →
BF16-add → FP32-accumulate; the rounding happens DURING accumulation, not
just once at store. The legacy path accumulates fully in FP32 then rounds
once at store. The accumulated rounding error structure is different.

**Why this is the top suspect**:
1. Silent default-on since iter 61 — every post-iter61 training run uses it.
2. Only ship that touches the weight-gradient ROUNDING (vs reduction-order
   changes elsewhere). Bias compounds directly into Adam's m/v then weights.
3. iter 61's own bench (NLL +0.0156 nat @ step 200) was at the parity-bound
   edge. If the bias is SYSTEMATIC (constant sign) rather than random walk,
   it grows linearly in steps: linear 0.0156 × 150 = +2.3 nat by step 30k
   (worst case). Random walk: √150 × 0.0156 = +0.19 nat. The observed +0.5
   nat sits between these, consistent with a partially-correlated bias.
4. Applies on every dWq/dWk/dWv/dWo across all 24 layers, every step.

---

## 3. Bisection plan

**FIRST experiment**: build binary with iter 61 reverted. Single-line override:

In `trainer/chiron_main.cpp` line 6508:
```cpp
const bool useFastBf16Grad = false;  // OVERRIDE: was useBf16Inner && W.bf16Grads
```

This routes SCFA inner-shear-backward through the legacy
`chiron_attention_shear_backward_bf16w_tiled` + `bf16_accum_axpy` chain. No
glades-ml change needed; the new BF16-out wrapper just goes unused.

Run flagship config + seed for 5000 steps, val-every=500. Compare NLL
trajectory to post-iter61 binary at same checkpoints.

**Expected**:
- If iter 61 is the culprit: NLL matches historical within ±0.1 nat at 5000.
- If not: regression persists → escalate to iter 60 (set
  `cfg.scfaCheckpointInner = false`).

**Cost**: ~50 min wall at 25k tok/s, T=16384 L=24.

**Why iter 61 first**: highest a-priori suspicion, cheapest single-line
override, no kernel rebuild.

---

## 4. Hypothesis ranking

1. **iter 61 BF16-grad direct cuBLAS-out (★★★★★)** — cuBLAS BF16-store
   rounding mode not guaranteed RN-even; may differ from legacy's explicit
   RN-even kernel. Bias compounds directly into weight updates across 24
   layers × 4 weights × 30k steps. Only ship that touches gradient rounding.
   200-step Gate-0 was at parity bound; long-horizon extrapolation consistent
   with observed +0.5 nat.

2. **iter 60 checkpoint-inner default-on (★★★★☆)** — backward now uses
   bit-exact cached forward instead of fresh recompute. Historical trained
   on recompute path. Two valid but slightly different optimization objectives;
   30k steps is plenty to diverge by 0.5 nat. Cheap to disable (CLI flag).

3. **iter 49 fused Adam (★★★☆☆)** — should be bit-equivalent to chained
   path but the kernel does m/v scale + param encode in one launch.
   Subtle ordering quirks possible.

4. **iter 53 LN-bwd 2-phase (★★☆☆☆)** — at parity-bound edge in 200-step
   Gate-0; minor contribution likely.

5. **iter 47 SCFA dK parallel (★★☆☆☆)** — atomicAdd → deterministic;
   different order, but unbiased.

6. **iter 56 argmax warp (★☆☆☆☆)** — logging path only.

**Top-1**: iter 61 BF16-grad direct cuBLAS-out is most likely culprit. Test
first by forcing `useFastBf16Grad=false`.
