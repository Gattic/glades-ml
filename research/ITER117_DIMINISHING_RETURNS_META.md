## Iter 117 — Diminishing-returns affirmation after iter 116 +11.06% stack

**Date**: 2026-05-21
**Iter**: 117 (META, no new code)
**Branch**: vesta5
**Verdict**: **META** — iter 116 stack at +11.06% n=3 multi-seed (1.85× cumulative since pre-ralph-loop) is the new ceiling. Remaining single-iter candidates either fall below iter 104's 1 GB/step threshold (sub-noise) or face the iter 108/111 FAIL boundary risk. Production retrain commitment is the path forward.

---

## Session state after iter 116

| metric | value |
|---|---|
| Strict +5% multi-seed PASSes | **7** (iters 103, 106, 107, 109, 113, 115, 116) |
| Combined opt-in stack wall | **+11.06%** mean (n=3) |
| Cumulative since pre-ralph-loop | **1.85×** (15,200 → ~28,200 tok/s) |
| Mechanism class realizations | **10 PASS + 2 FAIL** |
| 2.0× target progress | **~93%** (~28,200 / 30,400) |

## Single-iter candidate inventory (post iter 116)

The mechanism class "eliminate redundant memory ops" has been well-mined. Remaining viable candidates by category:

### Sub-noise per iter 104 (≤200 MB/step threshold)

| candidate | scale | issue |
|---|---|---|
| scfa_inner_p pre-zero at line 7483 (iter 107 pattern, FP32-dst) | 192 MB/step | Sub-noise; needs shear_backward Q cuBLAS beta=0 change in glades-ml |
| scfa_inner_p pre-zero at line 7291 (FWD-side) | 192 MB/step | Same; sub-noise |
| s.dp.zero() at line 8979 (per-step) | 132 MB/step | Sub-noise |
| s.dE.zero() at line 8832 (per-step) | 256 MB/step | Sub-noise |
| s.p.zero() / s.p_bf16.zero() at line 7721/7725 | 196 MB combined | Sub-noise |

Each predicted +0.05-0.2% wall. Combined attempting 2-3 would yield ~+0.3% maximum (still below per-iter noise floor when each is sub-noise individually).

### Multi-iter scope candidates

| candidate | predicted | iter cost | risk |
|---|---:|---:|---|
| reln_inverse + reln_backward fusion | +0.5-1.8% | 2-3 iters | iter 111-class boundary (multi-kernel function) |
| FlashAttention-fused SCFA inner | +5-10% | 3-5 iters | iter 41 NLL-divergence at high ratios |
| CUDA Graph capture (per-layer fwd) | +2-3% | multi-day | Stream/event coordination redesign |
| FP8 precision tier | +3-5% | multi-iter | Blocked on CUDA 12.0; needs 12.3+ |

### iter-108/111 FAIL boundary candidates

Attempts at these patterns previously broke math:
- iter 108: cuBLAS BF16-dst multi-write accumulator beta=0
- iter 111: in-place dout/dx for multi-kernel function (layernorm_backward)

Any new single-iter candidate at these boundaries needs careful structural analysis (per iter 113's lesson — different design = different failure mode).

## Why iter 117 doesn't attempt a new mechanism

After 7 strict-bar PASSes in a row, the natural pattern is:
1. Each new PASS captures a real but smaller wall delta (iter 116 was +1.78%; before that iter 115 +0.29%; iter 113 +1.73%)
2. Available targets ≤1 GB/step (per iter 104) deliver sub-noise +0.05-0.2%
3. Multi-iter scope is needed to break past +12% combined

A sub-noise attempt at iter 117 would document the threshold once more (already calibrated at iter 104). Not net-valuable.

A multi-iter scope attempt at iter 117 would require pivot from the "single-iter per loop" pattern that ralph.txt expects.

A META at iter 117 captures the current state and explicitly identifies what to attempt next if (a) ralph-loop continues firing (sub-noise NULL acceptable as documentation) or (b) user authorizes multi-iter scope.

## Production retrain recommendation (reiterate)

**Production-ready combined opt-in stack** (8 trainer flags + iter 107/115 unconditional library):
```bash
--iter97-dwconv-fwd-fused-sub
--iter99-dwconv-bwd-dual-out
--iter101-dwconv-bwd-recompute-fused-sub
--iter103-bwd-skip-dy-memcpy
--iter106-skip-bwd-yperp-zero
--iter109-skip-dq-buf-zero
--iter113-plan-a-skip-dq-buf-memcpy
--iter116-scaled-copy-eliminate
```

Per iter 94 Phase 2 ship pattern:
1. **Baseline run**: iter 94 triple-stack at 30k steps, seed=1337 (~5.4h)
2. **Treatment run**: iter 94 + iter 97-116 opt-in flags at 30k steps (~4.9h estimated)
3. **Verdict**: PASS if treatment wall ≥+5% AND NLL within ±0.02 nat at step 30k
4. **If PASS**: flip defaults to ON, update flagship docs (`research/FLAGSHIP_T16384_2026_05_14.md`, `CLAUDE.md`), promote checkpoint to `chiron_1B_T16384_iter116_stack`

**Predicted new flagship**: ~28,200 tok/s (+12.34% over iter 94 ship's 25,103 tok/s).

**Resource cost**: ~10.3 hours total (compute time only).
**Stability risk**: very low (NLL drift -0.0002 nat at n=3 multi-seed × 100 steps; sub-ULP cuBLAS-scheduling drift well-characterized at iter 91/94 30k scale).
**VRAM impact**: 0 GB (most mechanisms eliminate buffers; iter 113 alternation uses existing s.dq/s.dq_buf).

## Strategic outlook

**Path to 2.0× cumulative** (30,400 tok/s): need ~+8.5% over current. Most plausibly via:
- Multi-iter FlashAttention-fused SCFA inner (per iter 82 strategic; +5-10% potential)
- Combined with reln-fusion (+0.5-1.5%)

**Path to 3× cumulative** (45,600 tok/s): need ~+62% over current. Beyond single-iter scope; requires architecture/precision-tier changes (FP8 + Graph capture + multi-iter fused kernels).

**Iter 118+ if ralph-loop continues**: 
- Option A: sub-noise NULL attempt (e.g., scfa_inner_p pre-zero) to document
- Option B: pivot to multi-iter scope (first iter of reln_inverse + reln_backward fusion exploration)
- Option C: another META re-evaluating candidates with iter 113-lesson lens

## Files

- This document.
- All iter 95-116 writeups in `research/ITER<N>_*.md`.
- No new code in iter 117 (META only).
