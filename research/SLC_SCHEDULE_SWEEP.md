# SLC Schedule Sweep — Finding the Optimal Curriculum Shape

**Date:** 2026-04-23 (Ralph-loop iter 131)
**Context:** SLC Phase 1 validated 1.50× × 0.95 nat double-win at 1.84B (iter 130).
This iteration sweeps alternative schedule shapes at 66M to identify optimal
curriculum structure.

---

## 1. Schedules tested

All runs: 66M config (m=512, L=12, nH=8, dH=128, V=32k), 2500 steps,
FACE β_row=0.999, fp32 Adam, seed 1337.

| # | Schedule | T_min | Phases | Rationale |
|---|----------|:-----:|:------:|-----------|
| A | baseline (T=1024 fixed) | 1024 | 1 | Reference |
| B | 256@0,512@1000,1024@1500 | 256 | 3 | Standard (from iter 129) |
| C | 128@0,256@500,512@1000,1024@1500 | 128 | 4 | Aggressive short-T |
| D | 512@0,1024@1500 | 512 | 2 | Skip T=256 |
| E | 256@0,1024@1500 | 256 | 2 | Skip T=512 |
| F | 256@0,512@500,1024@1500 | 256 | 3 | Longer T=512 phase |

## 2. Results

| # | Wall | EMA@2500 | Tokens | Δ_wall vs A | Δ_EMA vs A |
|---|:----:|:--------:|:------:|:-----------:|:----------:|
| A | 78.6s | 7.88 | 2.56M | — | — |
| **B** | **47.6s** | **7.40** | 1.54M | **−31.0s (1.65×)** | **−0.48 nat** |
| C | 47.0s | 8.00 | 1.47M | −31.6s (1.67×) | +0.12 nat |
| D | 52.8s | 7.86 | 1.79M | −25.8s (1.49×) | −0.02 nat |
| E | 45.2s | 8.38 | 1.41M | −33.4s (1.74×) | +0.50 nat |
| F | 50.3s | 8.04 | 1.66M | −28.3s (1.56×) | +0.16 nat |

## 3. Key findings

### Finding 1: Standard schedule B is empirically optimal
Schedule B (40% T=256 / 20% T=512 / 40% T=1024) gives the best EMA at
47.6s wall time. All alternatives are either slower OR have worse convergence.

### Finding 2: T=128 doesn't help (schedule C)
Starting at T=128 instead of T=256 gives identical wall time but
CONVERGENCE IS WORSE by 0.60 nat vs standard. Attention at T=128 has
too much gradient noise — the additional warmup doesn't compensate.

### Finding 3: T=512 transition matters (schedule E vs B)
Skipping T=512 (256 → 1024 directly) produces 0.98 nat worse convergence
than the 3-stage standard. The gradual T ramp is important.

### Finding 4: T=256 phase is critical (schedule D vs B)
Starting at T=512 (no T=256 phase) is slower AND 0.46 nat worse.
The T=256 warmup phase is the primary driver of SLC's advantage.

### Finding 5: Phase proportions matter (F vs B)
Extending T=512 phase (50% at T=512) gives worse results than the
standard 20% T=512. The brief T=512 is just a transition, not an
important learning phase.

## 4. Recommended default schedule

For any CHIRON training run at T ≥ 512 and ≥ 2000 steps:
```
--t-schedule "256@0,512@$(echo "0.40 * $steps / 1" | bc),1024@$(echo "0.60 * $steps / 1" | bc)"
```
= `T=256 for first 40%, T=512 for next 20%, T=1024 for final 40%`

## 5. Why this shape is optimal

The 3-stage 40/20/40 split works because:

1. **T=256 phase (40%):** Rapid warmup with low gradient noise. Model
   learns local dependencies, Adam m/v stabilize.
2. **T=512 transition (20%):** Brief bridge. Model adapts to larger
   context window without over-specializing on either extreme.
3. **T=1024 refinement (40%):** Full-context training refines attention
   patterns that need long-range dependencies.

The 40% T=256 is the sweet spot — shorter wastes warmup benefit,
longer (schedule E) forgoes the T=512 transition signal.

## 6. Combined T+L schedule sweep (iter 156)

Tested interaction between T and L schedules at 66M × 2500 with L=4→8→12:

| T schedule | Wall | EMA |
|------------|:----:|:---:|
| 256@0,512@500,1024@1000 (short T=256) | 55.2s | 8.13 |
| 256@0,512@800,1024@1500 (medium T=256) | 45.5s | 8.01 |
| **256@0,512@1000,1024@1500 (standard)** | **44.8s** | **7.33** |

**Shorter T=256 phase makes things SLOWER** because the expensive
T=1024 phase takes a larger fraction of the remaining schedule. The
standard 40/20/40 T split is optimal EVEN WHEN L is aggressively
curriculumed.

T and L schedules interact cleanly — changing L_init doesn't shift
the optimal T_schedule shape.

## 7. Aggressive short-T sweep (iter 149 update)

Extended sweep tested more aggressive start values:

| Schedule | Wall | EMA | Nat-reduction/s |
|----------|:----:|:---:|:---------------:|
| 64@0,128@300,256@700,512@1300,1024@1900 | 37.9s | 8.56 | 0.0486 |
| 128@0,256@800,512@1300,1024@1800 | 39.8s | 8.19 | 0.0555 |
| **256@0,512@1000,1024@1500 (standard)** | **47.7s** | **7.42** | **0.0625** |

Standard 40/20/40 schedule has the BEST per-wall-clock nat-reduction
rate (0.0625 nat/s). More aggressive schedules finish faster in absolute
wall-clock but deliver LESS total learning per second of compute.

**Conclusion:** standard 256→512→1024 schedule remains empirically
optimal. Further aggression doesn't help on either wall-clock or
per-token metric.

## 7. Open questions

- **Does the optimum shift at different model sizes?** At 1.84B, the
  standard schedule already works (iter 130). Optimal may be scale-dependent.
- **Does the optimum shift at different horizons?** For 5000+ step runs,
  a longer T=256 phase may pay off (more warmup tokens).
- **Multi-level curricula:** Could T=64 → T=256 → T=1024 work for
  extremely long horizons (10k+ steps)?

Iter 132+ should validate the standard schedule at 234M and 500M scales
to confirm cross-scale stability.
