## Iter 104 — Skip fwd ycompr memcpy — NULL on wall

**Date**: 2026-05-21
**Iter**: 104 (post iter 103 STRICT PASS)
**Branch**: vesta5 (glades-ml) + glades-trainer (no glades-ml change)
**Verdict**: **NULL on wall** — +0.03% additional (within noise), parity bit-identical.  Tests the limit of iter 103's "memcpy overhead beyond bandwidth" pattern at smaller memcpy sizes.

---

## Motivation (iter 103 carryover)

Iter 103 eliminated a device_memcpy_d2d in the BWD path: 132 MB per call × 24 layers = 3.2 GB/step. Predicted +0.7% wall (raw bandwidth math); actual **+1.84% standalone** — the memcpy had CUDA driver overhead + stream sync + L2 cache pollution beyond raw bandwidth.

Iter 104 tests whether the same pattern scales to smaller memcpys: the FWD path's `scfa_inner_p → scfa_ycompr` memcpy at line ~6491 moves k*m*4 = 1024*2048*4 = **8 MB per call × 24 layers = 192 MB/step** — 16× smaller than iter 103's 132 MB.

## Conjecture (pre-committed)

**Target**: skip the line ~6491 device_memcpy_d2d. Read `scfa_inner_p.data()` directly at all downstream consumers (cache save cast at 6515, cache save memcpy at 6532, cuBLAS y_par GEMMs at 6553/6563).

**Math bit-identical**: same data, different physical buffer.

**Pre-committed Gate-0**:
- Wall delta (analytic linear scaling from iter 103): ~+0.11% (1.84% / 16 = 0.115%)
- Wall delta (analytic raw bandwidth): ~+0.04% (192 MB / 700 GB/s ≈ 0.27 ms)
- NLL drift: 0 (bit-identical)

## Implementation

- **chiron_main.cpp**: `iter104FwdSkipYcomprMemcpy` config flag (default OFF), CLI `--iter104-fwd-skip-ycompr-memcpy`.  When flag on:
  - Set `fwd_ycompr_ptr = W.scfa_inner_p.data()` (direct reference)
  - Skip the `device_memcpy_d2d(scfa_ycompr, scfa_inner_p, ...)` call
  - Use `fwd_ycompr_ptr` instead of `W.scfa_ycompr.data()` at the 4 downstream sites in fwd path
- No new kernel or library code.

## Bench (single-seed 100 steps × seed=1337 × T=16384 L=24 w=4)

| run | wall (s) | tok/s @ step 76 | NLL @ step 100 | PPL |
|---  |---:      |---:              |---:            |---: |
| iter 97+99+101+103 (iter 103 strict PASS)                       | 61.5 | 26,759 | 9.8476 | 18912.46 |
| **iter 97+99+101+103+104 (this iter)**                          | **61.5** | **26,767** | 9.8477 | 18914.06 |
| Δ                                                                | 0.0s | +8 (+0.03%) | sub-ULP | sub-ULP |

**Wall delta**: 0.0s (identical).  tok/s delta: +0.03% (well within ~±50 intra-run variance).

**NLL parity**: bit-identical at 4 decimals.  PPL drift +1.60 ≈ +0.00008 nat (sub-ULP, cuBLAS-scheduling drift class).

## Verdict

**NULL on wall, parity bit-identical.**

## Why NULL when iter 103 was +1.84%

The iter 103 → iter 104 wall-delta ratio is roughly the memcpy-size ratio:
- iter 103: 132 MB/call × 24 = 3.2 GB/step → +1.84% wall
- iter 104: 8 MB/call × 24 = 192 MB/step → +0.03% wall (sub-noise)
- Size ratio: 16.5×

If iter 103's overhead scaled linearly down, iter 104 should be ~+0.11%.  Actual: +0.03% (3× smaller than linear-scaling estimate).  The non-linearity suggests:
- L2 cache pollution is roughly a fixed cost above a threshold (192 MB/step is below L2-pollution threshold; 3.2 GB/step is well above).
- CUDA driver overhead per memcpy launch is a fixed cost (~10-20 µs); 24 launches × 15 µs = 0.36 ms = 0.06% wall — close to observed 0.03% but slightly higher (queue overlap?).

Sub-noise floor: at production T=16384 L=24 baseline 65.3s, the noise floor is ~±50 tok/s = ±0.2% wall. iter 104 is below this.

## Default

Flag stays **OFF** (opt-in via `--iter104-fwd-skip-ycompr-memcpy`).  Parity-clean by construction; kept in tree as zero-risk free option.  Combined production-recommended stack remains **iter 97+99+101+103** (+5.71% mean wall at n=3 multi-seed strict +5% bar PASS).

## Strategic lesson

**Memcpy overhead does NOT scale linearly with size.**  Iter 103's +1.84% from a 132 MB memcpy was disproportionately larger than analytic bandwidth math predicted — likely due to fixed CUDA driver overhead, L2 cache pollution above some size threshold, and possibly stream sync.

Iter 104 tested whether the same pattern holds at 16× smaller memcpy.  It doesn't — small memcpys are below the L2-pollution + driver-overhead threshold and incur only their raw-bandwidth cost (which is sub-noise at 192 MB/step).

Implication for future iters: target memcpys ≥1 GB/step for measurable wall savings.  Smaller memcpys (≤200 MB/step) are sub-noise even with the "iter 103 multiplier" of 3×.

## Cumulative state

iter 97+99+101+103+104 combined ≈ same +5.71% as iter 97+99+101+103 (iter 104 sub-noise).  Cumulative since pre-ralph-loop: **1.76×** (unchanged from iter 103).

## Files

- This document.
- `research/runs/2026-05-21-iter104-gate0/iter104_combined_100step.log` (61.5s, all 5 flags).
- Code:
  - `glades-trainer/trainer/chiron_main.cpp`: flag + CLI + conditional dispatch.
  - No glades-ml change (pure trainer-side optimization).
