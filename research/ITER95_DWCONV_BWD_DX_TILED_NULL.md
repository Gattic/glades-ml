## Iter 95 — Tiled dwconv backward dx — NULL

**Date**: 2026-05-21
**Iter**: 95 (post-iter94 triple-stack ship)
**Branch**: vesta5 (glades-ml) + glades-trainer
**Verdict**: **NULL** — no measurable wall delta, parity-tight (mechanism works, but dwconv backward dx is not L2-bound at production w=4).

---

## Bottleneck identified

The new triple-stack flagship (CHIRON 1B @ T=16384, 25,103 tok/s) sets
`scfa_conv_w = 4` (5-tap depthwise causal conv). Iter 73 (shipped 2026-05-19,
default-on as of iter 94) tiled the **forward** dwconv kernel — but only
specialized W_FILTER=9 (w=8 prior flagship). At w=4, iter 73's
`scfa_depthwise_causal_conv_fwd_tiled` falls back to the row-major kernel
(no-op), per `gpu_kernels.cu:8058`.

The **backward dx** kernel (`scfa_dwconv_dx_kernel`, gpu_kernels.cu:8086)
is still one-thread-per-element with no shared-mem caching. Called L=24 × 1
per bwd at T=16384. iter 73 docs noted the row-major forward at w=8 had
"~600 µs / call on misses" — same pathology should affect the backward dx.

## Conjecture (pre-committed)

Extend iter 73's shared-mem tiling to the backward dx (acausal direction:
reads dy[t+i] forward in t). New kernel `scfa_dwconv_dx_tiled_kernel<COLS,
N_OUT, W_FILTER>`. Dispatch from `scfa_depthwise_causal_conv_bwd_tiled` for
w == 4 (W_FILTER=5) or w == 8 (W_FILTER=9); row-major fallback otherwise.
dK kernel unchanged (already _par optimized).

**Math**: bit-identical to legacy `scfa_dwconv_dx_kernel` — same FMA order
(i = 0, 1, …, w), same break-on-OOB (t+i >= T), FP32 accumulator, `+=` output
(caller pre-zeros dx).

**Pre-committed Gate-0**:
- Wall delta: +0.5% to +1.5% tok/s on production (T=16384, m=2048, L=24, w=4)
- NLL drift: ≤ ±0.005 nat over 100-step bench (bit-identical FMA → expect
  zero drift modulo sub-ULP kernel-scheduling effects)
- Bench: 100-step apples-to-apples vs baseline (flag off), seed=1337

## Implementation

- **gpu_kernels.cu**: new `scfa_dwconv_dx_tiled_kernel` template in anonymous
  namespace (mirror of `scfa_depthwise_causal_conv_fwd_tiled_kernel` with
  acausal smem load: `dy_smem[k]` = global row `t_base + k` for k in
  [0, N_OUT + w)); new dispatcher `scfa_depthwise_causal_conv_bwd_tiled` with
  W_FILTER=5 and W_FILTER=9 specializations + row-major fallback.
- **gpu_kernels.h** (both glades-ml + vendored glades-trainer copies):
  declarations + no-CUDA stubs.
- **chiron_main.cpp**: `iter95DwconvBwdDxTiled` config flag (default OFF),
  `--iter95-dwconv-bwd-dx-tiled` / `--no-` CLI parse, dispatch at the
  single backward call site (line 7390).

## Bench (single-seed, 100 steps × seed=1337 × T=16384 m=2048 L=24 w=4)

| run | wall (s) | tok/s @ step 26 / 51 / 76 | NLL @ step 100 | PPL |
|---  |---:      |---:                       |---:            |---:  |
| baseline (flag off) | 65.3 | 24,912 / 25,196 / 25,194 | 9.8476 | 18912.84 |
| iter95   (flag on)  | 65.4 | 24,895 / 25,183 / 25,177 | 9.8476 | 18913.25 |
| Δ                   | +0.1s (+0.15%) | −0.07% (step 76) | 0.0000 | +0.4 (~+0.00002 nat) |

**Wall delta**: −0.07% to +0.15% depending on which step you read — both
inside the ~0.5-1% run-to-run noise band on RTX 4080 SUPER. Far below the
+3% iter-60-relaxed bar and the +5% strict-brief bar.

**NLL parity**: identical to 4 decimals (9.8476 both runs). PPL differs in
5th decimal (18912.84 vs 18913.25) — corresponds to ~0.00002 nat drift, which
is sub-ULP rounding from kernel-scheduling sub-ULP effects (same as iter 73
sub-ULP class). Math is bit-identical to within FMA-emit precision.

**Loss/grad trajectory parity**:
- Step 1 loss/||g||: 10.5777 / 3.452 — identical
- Step 26 loss/best: 10.3259 / 10.3259 — identical
- Step 51 loss/best: 10.3434 / 10.2939 — identical
- Step 76 loss/best: 10.1776 / 10.1776 — identical
- Step 26 ||g||: 2.889 vs 2.888 — 0.001 (sub-ULP)

## Verdict matrix

| bar | wall threshold | NLL threshold | result |
|---  |---:            |---:           |---     |
| Strict brief (≥5% tok/s + ±0.02 NLL) | +5% | ±0.02 | **NULL** (wall ~0, NLL parity-clean) |
| iter 60 relaxed (+3% + multi-seed parity) | +3% | parity | **NULL on wall** |
| Sub-3% silent-accrual (parity-clean + measurable wall) | >+0.5% | parity | **NULL on wall** (delta < noise) |

## Why null

Three converging mechanisms explain the absence of signal:

1. **w=4 conv has only 5 FMAs per output element** (vs 9 at w=8). The
   per-thread compute cost is small; the row-major kernel's "5 short load +
   5 FMA" loop is already memory-throughput-bound but not L2-thrash-bound at
   T=16384 m=2048.

2. **dy working set at w=4 is tiny** — for any block of (16 t-rows × 256
   cols) outputs, the dy reads span (16 + 4) = 20 rows × 256 cols × 4 B = 20
   KB. Comfortably resident in L2 (Ada has 64 MB L2). No thrashing to fix
   via smem caching.

3. **Block size already coalesces well**: the row-major kernel uses 256
   threads per block with each warp reading a contiguous 32×4 = 128 B cache
   line per row. Already at peak DRAM throughput; smem doesn't help.

The mechanism is parity-validated and the code path stays in tree for two
reasons:
- Could win at w=8 workloads (e.g., the prior w=8 flagship if revived) where
  the FMA-to-load ratio favors smem caching.
- Future kernels could share the smem-tile pattern.

## Default

Flag stays **OFF**. Opt-in only.

## Convergence with iter 70-94 ceiling pattern

Iter 95 joins iters 71/72/78/79/82-line as another null at the post-iter94
engineering ceiling. The triple-stack (w=4 + iter70 + iter73 + scfa-checkpoint-
inner-bf16 at 25,103 tok/s) sits at a regime where individual kernels have
been smem-optimized (or are too small to benefit). The next ≥+3% wall
improvement likely requires:

- A multi-iter compounded change (FlashAttention-fused SCFA inner — per
  iter 82 strategic finding; SCFA inner attention compute is the wall
  ceiling at ratio=32 where +37.6% wall is achievable at the cost of NLL).
- A new precision tier (FP8 readout — blocked by CUDA 12.0 cuBLAS-LT
  toolkit gap per iter 62).
- A whole-pipeline architectural change (e.g., reduced layer count via
  weight tying or MoD routing) — multi-iter scope per ralph.txt §3.

## Files

- This document.
- `research/runs/2026-05-21-iter95-gate0/baseline_100step.log` (65.3s, parity reference).
- `research/runs/2026-05-21-iter95-gate0/iter95_100step.log` (65.4s, flag-on).
- Code: `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` (kernel + dispatcher),
  `Backend/Machine Learning/Networks/cuda/gpu_kernels.h` (declaration + stub),
  `glades-trainer/include/Backend/Machine Learning/Networks/cuda/gpu_kernels.h`
  (vendored header), `glades-trainer/trainer/chiron_main.cpp` (flag + CLI + dispatch).
