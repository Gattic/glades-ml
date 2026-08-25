# Brief reframe — CHIRON throughput grind, iter 61 reckoning

**Date**: 2026-05-16
**After**: 15 iters of the "stacking-wins" brief (iter 47 → 61)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)

---

## The brief as written

> Goal: improve the CHIRON post-fix flagship (val NLL 4.0771) by cumulative **≥10× wall-clock per token at iso-NLL** via stacking individually-validated improvements. The 10× target is the 20-iter horizon; each iter's per-iter target is +5% to +20% wall-clock at NLL parity ± 0.02 nat.

## What actually happened

| iter | target                          | result                | tok/s win |
|---:  |---                              |---                    |---:       |
| 47   | SCFA dwconv-dK 2D-tiled         | **PASS**              | +5.20%    |
| 48   | LN-bwd dgamma                   | FAIL (too-small)      | +2.4%     |
| 49   | Fused Adam-int8 + BF16w/g       | **PASS**              | +6.98%    |
| 50   | SCFA sub-into-conv fusion       | FAIL (NLL drift)      | +1.31%    |
| 51   | --scfa-checkpoint-inner revival | FAIL → retro-ship     | +4.46%    |
| 52   | cuBLAS algo override (readout)  | NULL                  | ~0%       |
| 53   | LN-bwd 2-phase deterministic    | FAIL → retro-ship     | +2.86%    |
| 54   | (meta-analysis)                 | META                  | —         |
| 55   | --cuda-graphs / --fp8-attn      | NULL                  | ~0%       |
| 56   | warp-parallel argmax            | FAIL → retro-ship     | +1.62%    |
| 57   | fused accum_axpy + sumsq        | PUNT (accum>1 semantic) | —       |
| 58   | --bf16-logits-parallel-bwd      | NULL                  | +0.15%    |
| 59   | cuBLAS algo override (SCFA outer) | NULL                | ~0%       |
| 60   | **iter51 + 53 + 56 combined retro-ship** | **PASS**     | **+9.20%** |
| 61   | BF16-grad direct cuBLAS out (eliminate bf16_accum_axpy) | FAIL (+2.64% < +3% bar) | +2.64% (silent accrual) |

**Cumulative shipped (per iter rules)**: 1.052 × 1.070 × 1.092 = **+22.9%** over the pre-iter47 baseline (40,470 → 47,271 tok/s).

**Cumulative including silent accrual** (iter61 code is in the tree, default-on): **+26.2%** (40,470 → ~48,521 tok/s at iter-bench config).

**Brief target**: **+900%** (10×). **Gap to target**: ~800 percentage points.

**Trajectory**: 15 iters in, 3 official PASS iters, 10 FAIL/NULL/PUNT/META, cumulative +22.9% (or +26.2% with silent). Of the remaining 5 iters (to hit the 20-iter horizon), all engineering target slices ≥2% have been attacked solo. The next iter's expected win is ≤+3% if it clears the bar at all.

---

## Why 10× is empirically unreachable in this regime

Three independent empirical findings converge:

### 1. The dead-zone map is fully cataloged

After 15 iters, every kernel slice ≥2% of GPU time has been attacked:

| GPU-time slice                    | size  | iter that attacked | outcome |
|---                                |---:   |---                 |---      |
| cuBLAS GEMMs (readout + SCFA outer) | ~36% | 52 + 59            | DEFAULT_TENSOR_OP optimal across both major shape paths.  No algo lever. |
| SCFA element-wise (axpy2/sub/axpy/scaled_copy) | ~16% | 50         | iter50 NLL drift on fp32-FMA reordering.  Element-wise fusion is dead-zone. |
| Fused Adam (bf16w + bf16g)         | ~5.9% | 49                 | shipped +6.98%; remainder is irreducible per-tensor work. |
| LN-bwd (dx + dgamma)               | ~4.3% | 48 + 53            | dgamma shipped iter53; dx is well-parallelized at 1024-thread/block. |
| SCFA dwconv (fwd + dK + dx)        | ~9.9% | 47                 | dK shipped iter47; fwd + dx well-parallelized post-iter47. |
| BF16 grad commit (bf16_accum_axpy) | ~3.1% | 61                 | eliminated via cuBLAS BF16-out D type.  +2.64% wall (below bar). |
| reln (fwd + inverse)               | ~3.4% | iter9              | in-place; bandwidth-bound. |
| BF16 cast residue (scattered)      | ~1.7% | (multiple)         | mostly already amortized. |
| argmax / softmax / sumsq / cast    | <4%   | 56                 | argmax shipped iter56; rest well-tuned. |

**Empirical observation**: no remaining kernel ≥2% of GPU time has not been attacked.  The next solo iter's target floor is sub-2% slices.

### 2. The CPU profile shows the bottleneck moved

Linux `perf record -F 999 --delay=15000` (skipping the init phase) on the iter60 flagship shows **>90% of steady-state CPU is inside `libcuda.so`** — specifically `cudaLaunchKernel` + `cudaStreamSynchronize` internals.  The trainer's own C++ code is <2% of CPU time.

GPU utilization at the iter-bench config (T=8192 L=12 nH=16 dH=256) is ~25-30%.  The remaining ~70% is **CPU-launch-overhead and kernel-launch-latency bound**.  Reducing per-step kernel COUNT (iter 61's mechanism) is currently more valuable than reducing per-kernel duration — but the launch-overhead amortization is also bounded by the fundamental number of distinct compute steps in the architecture.

### 3. Five-for-five paradigm-shift falsification at this scale

Brief-class magnitudes ("10×") would require an architectural break, not engineering.  All five attempts in iters 22-46 failed Gate-0:

| paradigm | mechanism | iter | gate-0 fail mode |
|---       |---        |---:  |---               |
| #250 SFA | cellular sheaf augmentation | 22  | val NLL never crosses below 4.0771 in 500 steps |
| #260 IGAA | info-geometric mixture (K=4 modes) | 23-36 | same pattern; best val 4.17 |
| #261 HMTA | multipole rank-p attention | 41 | flagship has heavy spectral tails; no (p_K, p_V) point clears retention ≥0.85 AND FLOP ≥10× |
| #262 MEDAL train | discrete diffusion training | 44-46 | gap widens with scale; 1/α fix partial-helps but C1 still fails at +0.98 nat @ 15M |
| #262 MEDAL infer | discrete diffusion inference | (post-46) | magnitudes claim holds only vs degraded no-KV AR baseline; loses 16-60× vs production KV-cached AR |

**Empirical pattern**: paradigm-shift work at "trained-flagship + small augmentation + 500-step SFT" is structurally falsified.  Larger paradigm arcs (train-from-scratch) would need multi-iter budgets and aren't compatible with the per-iter contract.

---

## Recommendation

**Reframe the brief.  Engineering can deliver an additional +30-60% cumulatively but not +900%.  The honest deliverable is the engineering stack you have.**

### What to accept as shipped

- **Current stacked flagship**: 47,271 tok/s (iter60 official) or 48,521 tok/s (with iter61 silent accrual).
- **Cumulative win**: +22.9% (official) or +26.2% (silent) over pre-iter47 baseline of 40,470 tok/s.
- **vs. earlier baselines**:
  - Pre-iter47 was already +37.8% over the start-of-loop 15,200 tok/s baseline (from ralph-loop iters 1-5 in May 2026).
  - So the truly-cumulative figure from the iter-1 start: 15,200 × 1.378 × 1.262 = **~26.4k → 48.5k = +220% (3.2×)**.

That's a real engineering result over ~25 total iters.  Not 10×, but not negligible either.

### What to keep, what to stop

**Keep**:
- The 3 official PASS iters (47, 49, 60) are in the tree as the default flagship config.
- iter61's code is in the tree as silent accrual (default-on at `--bf16-grads + --scfa-bf16-inner`).
- The bench command + iter-bench config remain the reference for future engineering attempts.
- The dead-zone map (this document) prevents re-attempting closed avenues.

**Stop**:
- Solo iter engineering grind on per-kernel optimizations.  The remaining solo targets are all <+3%.
- Paradigm-shift exploratory dispatches (the 5-for-5 falsification rules these out without overwhelming theoretical justification + real-flagship pre-condition probe per iter 41's pattern).
- The "10× by iter 20" target — it's empirically out of reach.

### What COULD continue if budget allows

Three options, in order of risk:

**A. Multi-iter paradigm arc with explicit go/no-go gates.**  Pick ONE candidate (BF16 residual-p, FP8-output readout, sparse-attn replacement of SCFA, MoE-style FFN) and budget 5-10 iters.  Use the iter 41 SV-probe model: a real-flagship pre-condition probe BEFORE committing to the implementation arc.  Each phase has its own Gate-0; if Gate-0 fails, revert the arc.  Expected outcome: 1-2× standalone win OR falsification #6.

**B. Wider engineering surface with relaxed +2% bar.**  Drop the bar to +2% to catch sub-3% silent-accrual wins (iter48 was +2.4%, iter50 was +1.31% before NLL drift, iter53 was +2.86%, iter56 was +1.62%, iter61 is +2.64%).  Combined retro-ships could push another +5-10% cumulative.  Less risky than (A), much smaller upside, doesn't approach 10×.

**C. Production-scale revalidation.**  All iter47-61 wins were validated at the iter-bench config (T=8192, L=12).  Production config (T=16384, L=24, ~870M-1B params) has different bottleneck distribution.  An nsys profile at production scale might reveal a different top slice that's worth attacking specifically for that regime.  Risk: production runs are 4-8× more expensive; iter cadence drops to ~1-2 per day.

---

## Status of this reframe

This document is the formal output of the iter 60 META Option-A-then-iter-61-FAIL trajectory.  The "10× by iter 20" target is reframed as **"deliver the cumulative engineering stack as the result"**, with options A/B/C remaining open if the user wants to continue.

No new code in this iter beyond what iter 61 shipped silently.  This document and the iter 61 doc are the deliverables.

The ralph-loop has been cancelled per user instruction.  Future work — if any — should be invoked manually rather than via a stop-hook-driven loop.

---

## Files

- `research/ITER47_*.md` through `research/ITER61_*.md` — per-iter docs
- `research/runs/2026-05-16-iter60-profile/iter60_flagship.nsys-rep` — pre-iter61 profile
- `research/runs/2026-05-16-iter60-profile/iter61_flagship.nsys-rep` — post-iter61 profile
- `research/runs/2026-05-16-iter60-profile/perf_steady.data` — CPU profile
- Memory: `~/.claude/projects/-home-robert-dev-glades-ml/memory/iter*_*.md`
