# ECHO Regularizer No-Go Decision

**Date:** 2026-07-16

**Status:** Final no-go for the current hard/Huber ECHO objective

**Scope:** Production training recipes; the default-off research implementation remains available

## Decision

Do not ship or enable the current ECHO regularizer in production recipes. The implementation is correct and efficient, but the regularizer failed its quality and mechanism gates. Further tuning of the current hard/Huber hinge is not justified by the evidence.

## Evidence

- **lambda=0.1 E3:** final validation NLL was `6.7239` versus baseline `6.7077` (`+0.0162`), with a `1.47x` maximum-gradient ratio.
- **750-step low-lambda sweep:**
  - lambda=.01: `Delta NLL +0.0335`
  - lambda=.03: `Delta NLL -0.0192`
  - lambda=.05: `Delta NLL +0.0197`
- **Non-replication:** the apparent lambda=.03 gain reversed in the corrected matched rerun: baseline `7.4435`, ECHO `7.4741` (`Delta NLL +0.0306`).
- **Mechanism failure:** ECHO activation and copy-mass/PA increased in every sweep arm rather than weakening copy pressure.
- **Gradient attribution:** at the reproduced step-748 spike, full norm was `1.040997`, CE+Z was `1.039519`, SIRA-only was `0.004936`, and ECHO-only was `0.003666`. The excess was an ordinary CE+Z trajectory transient localized to tied `E` and ReLN beta, not a large direct ECHO field. Huber smoothing or ECHO-only clipping would therefore target the wrong proximal mechanism.

Primary evidence:

- [`research/CHIRON_ECHO_IMPLEMENTATION_2026_07_15.md`](../../research/CHIRON_ECHO_IMPLEMENTATION_2026_07_15.md)
- [`research/CHIRON_ECHO_GRADIENT_ATTRIBUTION_2026_07_16.md`](../../research/CHIRON_ECHO_GRADIENT_ATTRIBUTION_2026_07_16.md)
- CUDA/replay support: `7b9fa2d9c`
- Trainer replay fix: `glades-trainer@7ad3b728`

## What to Keep

- The default-off ECHO implementation and CLI flags for reproducibility.
- Owner-slot bitmap/GPU-summary optimization and CUDA parity coverage.
- Standalone component backward, exact gradient snapshot/restore, and attribution logs.
- Research reports and failed-gate evidence.

Do not add ECHO to standard recipes, serving state, or checkpoint state.

## Criteria for Any Future Redesign

A materially redesigned objective must pass all of these preregistered bars before reconsideration:

1. **Quality:** at least 3 matched seeds for at least 2,500 steps; mean `Delta NLL <= -0.01`, with no seed worse than its baseline.
2. **Mechanism:** reduce both ECHO activation rate and mean copy mass/PA by at least 10% versus baseline without worsening validation CE/NLL.
3. **Stability:** no NaNs or skipped updates; p99 gradient norm no more than `1.15x` baseline and no additional clipping dependence.
4. **Attribution:** component probes must show that the new regularizer directly creates the intended anti-copy pressure rather than only shifting later CE transients.
5. **Efficiency and isolation:** no more than 1% median training overhead, coefficient-zero bit parity, default-off behavior, and no checkpoint/serving state.

Until a different objective passes these bars, ECHO remains experimental infrastructure only.
