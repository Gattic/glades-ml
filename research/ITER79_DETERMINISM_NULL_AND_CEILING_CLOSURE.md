## Iter 79 — Determinism diagnostic NULL + engineering ceiling closure

**Date**: 2026-05-20
**Iter**: 79 (twenty-fifth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **NULL** on the determinism diagnostic AND **META** closure of the engineering-ceiling investigation phase.  10 consecutive non-PASS iters (70-79) conclusively document that single-axis ≥+5% wall wins are unavailable on the post-iter69 stack.

---

## Part 1 — Determinism diagnostic (CUBLAS_WORKSPACE_CONFIG=:16:8)

Hypothesis: the ±0.16 nat run-to-run variance documented in iter 75 might be sourced from non-deterministic cuBLAS algos picked by heuristic.  Setting workspace to minimum (`:16:8`) forces single-algo deterministic mode per CUDA docs.

| run | NLL @ step 200 | wall (s) |
|---  |---:            |---:      |
| baseline default (iter 76 mean, n=4) | 7.6154 ± 0.016 | 136.40 |
| det rerun 1 (seed=1337, :16:8) | 7.6661 | 137.1 |
| det rerun 2 (seed=1337, :16:8) | 7.6684 | 137.4 |
| Δ between det reruns | 0.0023 nat | 0.3 s |

**Findings**:
1. Determinism mode reruns differ by 0.0023 nat — comparable to default-mode within-day variance (0.0018).  Determinism does NOT achieve bit-identical reruns.
2. Determinism mode shifts absolute NLL by +0.05 nat vs default mode (different algo selected).
3. Determinism mode is ~0.5% SLOWER than default mode (137.25 s vs 136.40 s mean).

**Conclusion**: `CUBLAS_WORKSPACE_CONFIG=:16:8` doesn't solve the run-to-run variance problem.  The variance source is NOT cuBLAS algo selection — more likely the `atomicAdd` in the embedding gradient scatter (gpu_kernels.cu:1330) interacting with CUDA scheduler timing.

A deterministic embedding scatter (sort-then-segment-reduce instead of atomicAdd) would be a 2-4 hour CUDA implementation; net wall impact unknown.  Deferred as a future-iter methodology candidate, not a ship target.

---

## Part 2 — Engineering ceiling closure

After 10 consecutive non-PASS iters (70-79), the post-iter69 engineering ceiling is conclusively documented across ALL major angles:

### Confirmed dead-zones (no engineering lever available)

| angle | iters | finding |
|---    |---    |---      |
| SCFA element-wise kernel fusion/restructure | 50, 70, 73 | FMA-emit drift (3 confirmed instances) — wins +0.56% to +1.4% but NLL drift compounds |
| Cross-stream cuBLAS BF16-TC | 6, 58, 71 | Ada TC saturation NULL — stream concurrency theoretical only |
| Cast batching / vectorization | 72, 74 | Below-noise wall savings; L2/scheduler timing drift |
| cuBLAS algo override | 52 | DEFAULT optimal at readout shape |
| cuBLAS workspace tuning | 78 | DEFAULT optimal at all shapes; larger workspace = NLL drift, no speed |
| Combined retro-ship (iter 70+73) | 76 | +1.71% combined wall below +3% iter 60 bar; multi-seed NLL drift |
| Determinism mode | 79 | Doesn't achieve bit-identical reruns; slower; absolute NLL shift |

### Validated mechanism (parity-clean, below ship bar)

| iter | mechanism | wall | multi-seed NLL drift | status |
|---: |---        |---:  |---:                  |---     |
| 70  | fused chiron_scfa_axpy2 + iter65 SR-cast | **+1.27%** | **−0.008 nat (within ±0.02 strict)** | parity-clean silent-accrual; below +3% ship bar |

iter 70 is the ONLY parity-clean engineering win in the tree.  Cumulative wall improvement potential: ~+1.27% over iter 69 if shipped as default-on.

### Path-forward options requiring user direction

| option | scope | wall potential | NLL risk |
|---     |---    |---:           |---       |
| **A. Hadamard FWHT kernel** replacing SCFA outer cuBLAS GEMMs | multi-day kernel impl + production retrain | +3-7% | requires fresh-init training; cannot reuse current flagship weights |
| **B. FlashAttention-fused SCFA inner attention** | multi-day kernel impl | +5-15% | bit-identical possible with careful FMA emit matching |
| **C. Mixture-of-depth routing** | multi-day architectural change | 1.5-2× inference (not training) | requires training routing weights |
| **D. Accept iter 69 + iter 70 silent-accrual** as production ceiling | no further work | +1.27% (silent) | none |
| **E. Other (user direction)** | varies | — | — |

The brief's 10× target appears infeasible without committing to Option A, B, or C as a multi-iter arc.  Single-iter ≥+5% engineering wins on the post-iter69 stack have been conclusively ruled out across 10 iters.

## Cumulative state since pre-ralph-loop

| epoch | wall improvement |
|---    |---:              |
| pre-ralph-loop (15.2k tok/s baseline) | — |
| iter 5 ship (20.9k tok/s after bf16/fuse-streams) | +37.8% |
| iter 60 combined retro-ship (cumulative iter 51+53+56) | +22.9% over pre-iter47 |
| iter 68 SHIP (BF16-residual-p default-on) | +4.28% |
| iter 69 PASS (BF16 checkpoint-inner cache) | +6.83% |
| iter 70-78 (none shipped) | 0% net (iter 70 +1.27% silent if opted-in) |
| **Current production**: 24,023 tok/s | **1.58× over pre-ralph-loop baseline** |

Current cumulative: 1.58× wall improvement over pre-ralph-loop.  Brief target: 10× (cumulative within 20 iters).  Empirically observed ceiling: ~1.6×.

## Recommendation

The post-iter69 engineering investigation is COMPLETE.  Further single-iter attacks at the strict +5% bar produce only FAIL/NULL/NEGATIVE outcomes per iter 70-79 evidence.

**Recommended next action**: pause the ralph loop and let the user choose Option A/B/C/D/E from the table above.  Continuing the per-iter contract will produce more null/below-bar data without strategic progress.

If the loop continues autonomously, the next iter (80) should either:
- Begin Option A (Hadamard FWHT) as a multi-iter arc, with explicit scope acknowledgment
- Begin Option B (FlashAttention SCFA inner) similarly
- Ship iter 70 as default-on (Option D, +1.27% silent-accrual) and document loop completion

## Files

- This document (iter 79 + ceiling closure META).
- `research/runs/2026-05-20-iter79-bench/det_rerun{1,2}.log`.
- No code change.
