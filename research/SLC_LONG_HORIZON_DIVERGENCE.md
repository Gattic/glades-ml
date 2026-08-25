# SLC Long-Horizon Divergence — 10,000-step Failure Mode

**Date:** 2026-04-24 (Ralph-loop iter 138)
**Finding:** At 10,000-step horizon, SLC diverges to NaN after the
T=512→T=1024 transition. This is a previously-unobserved failure mode.

---

## 1. Experimental setup

Config: 66M (m=512, L=12, nH=8, dH=128), fp32 Adam, FACE β=0.999, seed 1337.
Schedule: `256@0,512@4000,1024@6000` (40% T=256, 20% T=512, 40% T=1024).

## 2. Empirical trajectory

| Step | Loss | EMA | ||g|| | scale | wall |
|------|:----:|:---:|:-----:|:-----:|:----:|
| 1 | 10.39 | 10.39 | 0.50 | 1.000 | 0.0s |
| 2000 | 8.57 | 8.65 | 0.53 | 1.000 | 18.2s |
| 4000 | 8.30 | 8.45 | 0.63 | 1.000 | 36.3s |
| 6000 (T=512→1024) | 8.22 | 8.47 | **4.47** | **0.22** | 64.7s |
| 8000 | **NaN** | NaN | NaN | 0.00 | 127.6s |
| 10000 | NaN | NaN | NaN | 0.00 | 190.7s |

**Divergence symptom:** at step 6000 (T=1024 transition), gradient norm
spikes from 0.63 to 4.47 (7× increase), gradient scale drops to 0.22
(clipping kicks in). By step 8000, loss and gradient are NaN.

## 3. Diagnosis

The likely root cause: when T doubles from 512 to 1024, the effective
batch size (tokens per Adam step) doubles. With fixed LR:
- LR was tuned for T=512 effective batch (smaller)
- At T=1024, same LR gives 2× larger updates
- With FACE's high β=0.999 preconditioner memory, this creates a
  large update that blows past the gradient-clip threshold

This wasn't observed at 2500 steps because:
- SLC's 2500-step schedule has T=1024 from step 1500 onwards
- The 1000-step T=1024 tail is too short for the instability to
  manifest fully
- At 10,000 steps, the 4000-step T=1024 tail gives instability time
  to compound

## 4. Implication for SLC

SLC as currently implemented has an UNSAFE long-horizon regime. The
paradigm needs an LR-compensation mechanism at T transitions:

**Proposed fix:** scale LR inversely with T_current:
- `lr_effective = base_lr · (T_reference / T_current)`

This preserves effective update magnitude across T changes.

**Alternative:** apply a mini-warmup at each T transition (e.g., linearly
ramp LR from 10% to 100% over 50 steps after each T increase).

## 5. Revised SLC recommendation

For short horizons (≤2500 steps), the standard 256→512→1024 schedule
is SAFE and delivers 1.5-1.68× speedup.

For longer horizons (5000+ steps), implement LR-T coupling before
deploying SLC. Otherwise divergence risk is high.

## 6. Research program implication

This is a BOUNDARY condition for the SLC paradigm. Does not invalidate
the 4-scale validation (all at 2500 steps). But long-horizon safety
requires Phase 2 implementation of LR-T coupling.

SLC's characterization should be updated:
- **Safe regime:** 2500-step horizon, schedule with ≥ 40% T=max tail
- **Risky regime:** 10000+ step horizon WITHOUT LR coupling
- **Mitigation:** planned Phase 2 (LR-T coupling flag)

## 7. Honest research accounting

The iter 133 "pure throughput paradigm" characterization for SLC is
correct at tested horizons. This iter 138 finding adds:
- Divergence hazard at long horizons
- Need for Phase 2 LR-coupling before extended deployment
- Boundary condition in the paradigm's validity envelope

Strengthens the scientific framing — SLC is a well-characterized
throughput curriculum with identified safe and risky regimes, not a
magic bullet.
