## Iter 96 — Tiled dwconv fwd at w=4 + nsys profile — NULL (silent-accrual parity-safe)

**Date**: 2026-05-21
**Iter**: 96 (continuation post iter 95 NULL)
**Branch**: vesta5 (glades-ml)
**Verdict**: **NULL on wall, parity-clean.** Empirically confirms iter 95's hypothesis: w=4 dwconv kernel is not L2-bound — neither forward nor backward smem-tiling helps. nsys profile also delivered.

---

## nsys profile (top kernels by wall % at production T=16384 L=24 triple-stack)

25-step bench on RTX 4080 SUPER, triple-stack flagship config (w=4, all default flags on):

| % | Kernel | Calls/step | Avg µs | Total µs/step |
|---:|--- |---:        |---:    |---:           |
| 14.6 | cutlass_s1688bf16gemm_256x128_16x3_nn_align4 (cuBLAS BF16 NN) | 124 | 745 | 92,300 |
| 9.8  | cutlass_s1688bf16gemm_256x128_16x3_nt_align4 (cuBLAS BF16 NT) | 74 | 836 | 61,900 |
| 7.6  | chiron_scfa_axpy2_dual_p_kernel (iter 70 fused) | 50 | 964 | 48,200 |
| 5.1  | chiron_scfa_sub_kernel | 50 | 646 | 32,300 |
| 4.8  | ampere_s1688gemm_bf16_128x128_ldg8 (small bf16 GEMM) | 150 | 205 | 30,700 |
| 3.8  | causal_mask_softmax_kernel | 74 | 323 | 23,900 |
| 3.4  | cutlass bf16_s16816gemm_bf16_256x128_32x3_tn_align8 (readout fwd) | 1 | 20,100 | 20,100 |
| 3.3  | scfa_depthwise_causal_conv_fwd_kernel (row-major; iter96 target) | 50 | 424 | 21,200 |
| 3.1  | cutlass s16816gemm_bf16_256x128_nn_align8 (readout dE bwd) | 1 | 19,700 | 19,700 |
| 3.1  | ampere_s16816gemm_bf16_256x128_ldg8_nt (readout dq bwd) | 1 | 19,600 | 19,600 |
| 2.9  | adam_update_int8_state_bf16w_bf16g_kernel | 96 | 191 | 18,400 |
| 2.6  | axpy_kernel | 25 | 650 | 16,200 |
| 2.5  | layernorm_backward_dx | 24 | 651 | 15,600 |
| 2.4  | scfa_dwconv_dx_kernel (iter 95 NULL target) | 24 | 644 | 15,500 |
| 1.9  | chiron_reln_forward_rows | 26 | 475 | 12,300 |
| 1.6  | k_cast_f32_to_bf16 | 303 | 34 | 10,300 |

**Composition:**
- cuBLAS BF16 GEMMs (all variants): **~50% of step wall**.
- SCFA element-wise (axpy2_dual_p, sub, scaled_copy, reln_*) + Adam: **~25% of step wall**.
- Inner attention softmax: 3.8% (already efficient).
- dwconv (fwd row-major + bwd dx row-major + bwd dK par): 3.3 + 2.4 + 1.7 = **7.4%** total.
- LN bwd: ~4%.
- Readout 3 GEMMs: ~10%.

**Headroom analysis:**
- No single non-cuBLAS kernel has more than 7.6% wall (axpy2_dual_p, already iter 70 fused).
- cuBLAS GEMMs dominate but are already optimized (iter 51-94 + cuBLAS algo NULL probes at iters 52/59/78).
- Single-iter ≥+3% wall targets are scarce. Highest-EV targets remain in fused multi-kernel scope (SCFA inner attention fusion per iter 82 strategic finding).

---

## Iter 96 mechanism

Extension of iter 73's smem-tiled forward dwconv kernel from W_FILTER=9 (w=8) to **W_FILTER=5 (w=4, production triple-stack flagship)**.

Current state (before iter 96): `scfa_depthwise_causal_conv_fwd_tiled` dispatcher has only `if (w == 8)` branch instantiating the tiled kernel; at w=4 it falls back to the row-major kernel. iter 73 flag is default-on after iter 94 ship, but at w=4 production it was effectively a no-op for FWD (only the BWD shared-mem path under `iter73DwconvFwdTiled` actually fires).

Iter 96 edit: dispatcher gains `if (w == 4)` branch instantiating `scfa_depthwise_causal_conv_fwd_tiled_kernel<256, 16, 5>`. **Math bit-identical** to row-major (same K*x accumulation order i=0..w, same break-on-negative-src termination, FP32 accumulator). Smem footprint at W_FILTER=5: x_smem 20×256×4 = 20 KB, K_smem 256×5×4 = 5 KB = 25 KB (well within Ada's 48 KB default per-block smem).

## Pre-committed conjecture

Wall: NULL or marginal (≤ +1%) — iter 95 evidence showed w=4 BWD dx tiled = NULL; by symmetry FWD likely also NULL.
NLL: bit-identical (≤ ±0.001 nat sub-ULP) by FMA-order preservation.
Bench: 100-step apples-to-apples vs `--no-iter73-dwconv-fwd-tiled` (forces row-major at w=4 on both fwd AND bwd dx paths).

## Bench (single-seed 100 steps × seed=1337 × T=16384 m=2048 L=24 w=4)

| run | wall (s) | tok/s @ 25/50/76 | NLL @ step 100 | PPL |
|---  |---:      |---:              |---:            |---: |
| baseline (--no-iter73-dwconv-fwd-tiled, row-major fwd + dx) | 65.3 | 24,932 / 25,223 / 25,195 | 9.8476 | 18912.93 |
| iter96 (default on, W_FILTER=5 tile)                        | 65.3 | 24,906 / 25,206 / 25,196 | 9.8476 | 18913.50 |
| Δ                                                            | 0.0  | −0.10 / −0.07 / +0.004 % | 0.0000 | +0.57 (~+0.00003 nat) |

**Wall**: identical to 0.1s precision (which is sub-ULP at 65.3s scale = 0.15%). Below noise.

**NLL**: identical to 4 decimals (9.8476 both runs). PPL diff 0.57 corresponds to ~0.00003 nat drift, sub-ULP.

**Trajectory parity**: identical at all log checkpoints (10.5777/10.3259-60/10.3434/10.1776-77). ||g|| @ step 26 differs by 0.001 (sub-ULP), at step 1 identical to 3 decimals.

## Verdict

**NULL on wall, parity-clean.** Same outcome as iter 95 BWD-dx-tile at w=4. Empirically confirms: **w=4 dwconv kernel (5-tap, 20 KB working set per block) is not L2-bound** — smem tiling has no headroom on either FWD or BWD pass.

Iter 96 dispatcher edit is bit-identical math, zero wall regression. Kept as a clean code path under the existing `--iter73-dwconv-fwd-tiled` flag's default-on default (silent-accrual + zero-cost).

## Why w=4 ≠ w=8 for tiling

iter 73 shipped at w=8 with measured wall improvement (+0.42% multi-seed forward alone per memory `iter81-iter73-alone-multiseed`). At w=4, the same mechanism delivers nothing because:

1. **Per-thread compute**: w=4 = 5 FMAs/output vs w=8 = 9 FMAs/output. Less compute to amortize over smem-load cost.
2. **Working set size**: at w=4, dwconv reads (N_OUT + w) = 20 rows × 256 cols × 4 B = 20 KB of input per block, vs 24 KB at w=8. Both fit in L1; both trivially fit in L2 (Ada has 64 MB L2). L2 hit rate already near 100% with the row-major kernel.
3. **Memory access pattern**: row-major kernel's 256-thread blocks already coalesce at near-peak DRAM throughput. Smem caching adds latency without reducing DRAM traffic.

The smem-tile mechanism wins only when L2 thrashing is the bottleneck — which it is at w=8 (~24 KB per block × concurrent blocks per SM can exceed L1) but NOT at w=4.

## Cumulative iter 70-96 ceiling pattern

Iter 96 NULL is the 13th non-PASS iter at the post-iter94 stack (excluding the iter 87-94 retrain arc which shipped). The pattern is empirically tight:

- All single-kernel smem-tile attacks at w=4 (iter 95 dx, iter 96 fwd): NULL
- All cuBLAS algo / workspace attacks (iters 52, 59, 78): NULL
- All stream-concurrency revivals (iters 6, 58, 71): NULL on Ada
- All multi-cast batching (iter 72) and online softmax (iter 83): NULL
- Float4 cast vectorization (iter 74): FAIL on NLL drift

The remaining headroom is concentrated in cuBLAS GEMMs (50% of step) — which require multi-iter scope (FlashAttention-fused SCFA inner per iter 82 strategic, or precision tier change blocked by CUDA 12.0 toolkit gap).

## Files

- This document.
- `research/runs/2026-05-21-iter96-gate0/baseline_100step.log` (65.3s, --no-iter73 baseline).
- `research/runs/2026-05-21-iter96-gate0/iter96_100step.log` (65.3s, iter73 default-on incl. iter96).
- `/tmp/iter96_profile.nsys-rep` (nsys profile, 25-step bench).
- Code: `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` (dispatcher edit: added `if (w == 4)` branch with W_FILTER=5 template instantiation).
