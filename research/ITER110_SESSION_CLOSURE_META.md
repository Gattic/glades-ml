## Iter 110 — Session closure META + production retrain recommendation

**Date**: 2026-05-21
**Iter**: 110 (META, no new code)
**Branch**: vesta5
**Verdict**: **META** — engineering ceiling reached at single-iter scope.  Combined iter 97+99+101+103+106+107+109 stack delivers **+8.35% n=3 multi-seed strict-bar PASS** at NLL within ±0.02 nat.  Strongest production retrain arc candidate of this ralph-loop session.

---

## Session arc summary (iters 95-109)

Starting state (iter 94 ship): CHIRON 1B triple-stack at 25,103 tok/s @ T=16384, NLL 4.20 @ step 30k after multi-seed 30k Phase 2 retrain.

**Mechanism class introduced this session**: "eliminate redundant memory ops" — 7 realizations validated, 1 boundary case identified (FAIL).

### Per-iter results

| iter | mechanism | wall Δ | NLL | verdict | cumulative |
|---:|---|---:|---|---|---:|
| 95 | bwd dx tile w=4 (smem) | 0% | bit-id | NULL | 25,103 |
| 96 | fwd tile w=4 + nsys profile | 0% | bit-id | NULL (informative) | 25,103 |
| 97 | smem-load arith fwd sub fold | +1.54% | bit-id | PASS | 25,603 |
| 98 | iter 95+97 stacking test | 0% | bit-id | NULL on stacking | 25,603 |
| 99 | dual-output writes (bwd dx-axpy fold) | +1.40% | bit-id | PASS | 25,990 (+ iter 97) |
| 100 | n=3 multi-seed validation (iter 97+99) | +3.08% | bit-id | **MULTI-SEED PASS +3%** | (validation) |
| 101 | dual-output side-write (bwd recompute) | +0.77% | bit-id | PASS | 26,190 |
| 102 | softmax + BF16 fuse | +0.08% | bit-id | NULL (sub-noise) | 26,211 |
| 103 | bwd dy memcpy skip (3.2 GB/step) | +1.84% | bit-id | **STRICT +5% MULTI-SEED PASS** | 26,738 |
| 104 | small fwd memcpy skip (192 MB/step) | +0.03% | bit-id | NULL (calibrates threshold) | 26,738 |
| 105 | nsys re-profile | — | — | META (info) | 26,738 |
| 106 | bwd yperp memset skip (3.2 GB/step) | +0.74% | bit-id | **STRICT +5% MULTI-SEED PASS** | 26,959 |
| 107 | cuBLAS beta=0 + 3 caller memsets (1.15 GB/step) | +0.56% | bit-id | **STRICT +5% MULTI-SEED PASS** | 27,093 |
| 108 | dW_bf BF16-dst cuBLAS beta=0 (1.5 GB/step) | +0.5% wall | **+0.5 nat drift** | **FAIL** (math boundary) | 27,093 |
| 109 | bwd dq_buf memset skip (3.2 GB/step) | +0.77% | sub-ULP | **STRICT +5% MULTI-SEED PASS** | 27,302 |
| **110** | **META (this doc)** | — | — | **session closure** | **27,302** |

### Aggregate

- **Iters**: 16 (95-110)
- **PASS**: 8 (97, 99, 100 [multi-seed], 101, 103, 106, 107, 109)
- **NULL**: 5 (95, 96, 98, 102, 104) — all parity-clean, sub-noise
- **FAIL**: 1 (108) — first parity break of session; mechanism-class boundary
- **META**: 3 (96 nsys, 105 nsys, 110 closure)
- **Strict +5% bar multi-seed PASSes**: 4 (103, 106, 107, 109)

### Cumulative wall improvement

| stage | tok/s | over baseline (25,103) | cumulative over 15,200 |
|---|---:|---:|---:|
| iter 94 ship (baseline) | 25,103 | — | 1.65× |
| iter 97 (silent-accrual) | 25,603 | +1.99% | 1.68× |
| iter 97+99 (iter 100 multi-seed) | 25,990 | +3.53% | 1.71× |
| iter 97+99+101 | 26,190 | +4.33% | 1.72× |
| iter 97+99+101+103 (iter 103 strict PASS) | 26,738 | +6.51% | 1.76× |
| iter 97+99+101+103+106 | 26,959 | +7.39% | 1.77× |
| iter 97+99+101+103+106+107 | 27,093 | +7.93% | 1.78× |
| **iter 97+99+101+103+106+107+109 (current)** | **~27,302** | **+8.76%** | **1.80×** |

### Per ralph.txt cumulative targets

| target | tok/s required | status |
|---|---:|---|
| 1.5× (iter 5) | 22,800 | ✓ HIT (iter 10 ship was 20,108, iter 94 ship 25,103) |
| 3× (iter 10) | 45,600 | NOT MET — would require multi-iter scope |
| 10× (iter 20) | 152,000 | NOT MET — multi-iter only |

Cumulative target at this point in the ralph-loop is 1.80×.  Linear extrapolation to ~iter 30 with sustained +0.5% per iter gives ~1.95× — well short of the 3×/10× cumulative goals.

The single-iter "eliminate redundant memory ops" mechanism class delivered roughly the expected ceiling: +0.5-1.8% per iter, totaling ~+8.35% combined.  Further single-iter gains face diminishing returns — most remaining memory ops are sub-noise (per iter 104 ≥1 GB/step threshold) or face the iter 108-class FAIL boundary.

## Mechanism class: "eliminate redundant memory ops"

Six realizations PASS, one FAIL (boundary identified):

| iter | realization | typical scale | Δ wall |
|---|---|---|---:|
| 97 | smem-load arithmetic (fold producer→consumer) | per-element | +1.54% |
| 99 | dual-output writes (fold consumer→producer) | per-element | +1.40% |
| 101 | dual-output side-write (fold both, side output) | per-element | +0.77% |
| 103 | pure memcpy skip (buffer rename) | 3.2 GB/step | +1.84% |
| 106 | pure memset skip (pre-zero under single-assign) | 3.2 GB/step | +0.74% |
| 107 | cuBLAS beta=0 + caller memsets skip | 1.15 GB/step | +0.56% |
| **108 FAIL** | cuBLAS BF16-dst beta=0 (multi-write accumulator) | 1.5 GB/step | **broken** |
| 109 | pure memset skip (different buffer, single-assign) | 3.2 GB/step | +0.77% |

**Class boundary** (iter 108 FAIL): cuBLAS beta=0 + skip caller pre-zero is SAFE for FP32-destination single-write cuBLAS (iter 107 PASS at sdQ), but UNSAFE for BF16-destination multi-write accumulator buffers (iter 108 FAIL at dW_bf).  Hypotheses for the BF16 quirk (untested):
1. cuBLAS GemmEx with BF16 destination may have different beta semantics from FP32-dst
2. The BF16 destination read may be needed for quantization reference
3. Pre-zero may provide an implicit stream sync barrier

**Sub-noise threshold** (iter 104 calibration): memcpys/memsets ≤200 MB/step deliver sub-noise wall savings (~0.03%-0.1%).  The "high-cost zone" starts around 1 GB/step.

## Production-ready combined stack

**Trainer flags** (all default OFF, ship by opt-in):
```bash
--iter97-dwconv-fwd-fused-sub
--iter99-dwconv-bwd-dual-out
--iter101-dwconv-bwd-recompute-fused-sub
--iter103-bwd-skip-dy-memcpy
--iter106-skip-bwd-yperp-zero
--iter109-skip-dq-buf-zero
```

**Library changes** (unconditional, applied at iter 107):
- `flash_attention_backward_cublas_tiled` uses `beta=0` for dV/dK GEMMs (was beta=1)
- 3 caller-side `cudaMemsetAsync(sdQ/sdK/sdV)` calls removed in `chiron_attention_shear_backward_{tiled,bf16w_tiled,bf16w_bf16g_tiled}`

**Library no-op** (iter 108 FAIL):
- `chiron_attention_shear_backward_bf16w_bf16g_tiled` retains `bool dw_beta_zero` parameter for API stability but ignores it (forces beta=1 internally)

**Validated**: n=3 multi-seed (seeds 1337/1338/1339) × 100-step bench × T=16384 L=24 w=4.
**Result**: +8.35% mean wall (std 0.10%), mean NLL Δ -0.0002 nat (within ±0.02 strict).
**Expected production tok/s** (after defaults flip): ~27,302 tok/s (vs iter 94 ship 25,103 = +8.76%).

## Recommendation for user: production retrain arc

Per iter 94 Phase 2 ship pattern, the combined iter 97-109 opt-in stack is ready for **30k apples-to-apples retrain validation**:

### Phase 2 protocol (matching iter 94 ship)

1. **Baseline run**: current iter 94 triple-stack at 30k steps, seed=1337.  Expected: ~25,103 tok/s, NLL ~4.20 at step 30k.
2. **Treatment run**: iter 94 triple-stack + iter 97-109 opt-in flags at 30k steps, seed=1337.  Expected: ~27,302 tok/s (+8.76%), NLL within ±0.02 nat of baseline.
3. **Verdict**: PASS if treatment wall ≥+5% AND NLL within ±0.02 nat of baseline AND no late-divergence pattern.

### Phase 3 ship (if Phase 2 PASS)

1. Update `run.sh flagship` to add the 6 opt-in flags to the STACK
2. Flip `chiron_main.cpp` defaults: `iter97DwconvFwdFusedSub(true)`, etc. for all 6 flags
3. Update `CLAUDE.md` and `research/FLAGSHIP_T16384_2026_05_14.md` to reflect new flagship at ~27,302 tok/s
4. Promote checkpoint to new `chiron_1B_T16384_iter109_stack` location

### Resource cost

- 30k retrain at iter 94 wall (25,103 tok/s): ~5.4 hours
- 30k retrain at predicted iter 109 stack wall: ~5.0 hours
- Total Phase 2 effort: ~10.5 hours of compute time

### Risk assessment

- Math correctness: NLL drift -0.0002 nat at n=3 multi-seed × 100 steps (very tight).  iter 91/94 30k retrain methodology validated sub-ULP drift to NOT compound at 30k scale.
- Stability: VRAM impact 0 GB; no new kernels add memory footprint
- Rollback: each iter flag is opt-in; can disable individually if regressions emerge

## Strategic outlook (iters beyond 110)

The single-iter "eliminate redundant memory ops" class is now well-mined.  Remaining single-iter targets at the post-iter109 stack:

**Sub-3% individual potential**:
- bwd softmax + softmax_backward_attn fusion (~6 GB/step traffic, but requires kernel-level fusion + cuBLAS reorder; complex)
- s.dq_buf → s.dq memcpy skip at line 9201 (3.2 GB/step, but requires buffer-swap pointer logic across many call sites)
- Adam batching (96 → 1 launch; sub-noise savings)
- LN bwd dgamma_dbeta partial buffer optimizations (1.8% wall but reductions block fusion)

**Multi-iter scope (per iter 82 strategic memo)**:
- FlashAttention-fused SCFA inner (predicted +5-10% but requires multi-iter custom kernel work; iter 41 NLL-divergence risk at ratio>16)
- FP8 precision tier (blocked on CUDA 12.0 toolkit; needs 12.3+ for cuBLAS LT FP8 readout)
- CUDA Graph capture of per-layer (multi-day work; ~2-3% wall headroom)
- MoE / conditional computation (ralph.txt §3 priority 3, scope > single iter)

**Path to 3× cumulative**: would require ~+50% over current state.  Realistically requires FlashAttention-fused inner attention (multi-iter) + production retrain integration.

## Files

- This document (META).
- All iter 95-109 writeups in `research/ITER<N>_*.md`.
- Combined opt-in stack code lives in:
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` (iter 97/99/101/107 library changes)
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` (iter 97/99/101/102 kernel additions)
  - `glades-trainer/trainer/chiron_main.cpp` (6 trainer flags + dispatch logic)
- No new code in iter 110 (META only).
