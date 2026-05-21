# Ralph-Loop Surprise #15 — SAS + SLC Compound Shock

**Date:** 2026-04-24 (Ralph-loop iter 169, live 1.84B × 650k run)
**Type:** Integration failure (taxonomy D from `RALPH_LOOP_METHODOLOGY_LESSONS.md`)
**Severity:** Run-terminating divergence.

---

## What happened

The first long-horizon invocation of the iter-168 SAS α-curriculum (`--sas-schedule "0.3@0,0.5@260000,1.0@390000"` on a 650k-step 1.84B CHIRON run) diverged hard between step 260k and 325k:

| Step | Loss | EMA | Wall |
|---:|:-:|:-:|:-:|
| 195,000 | 9.26 | 9.04 | 1h 41m |
| 260,000 | 9.34 | 9.49 | 3h 01m |
| **325,000** | **15.65** | **15.97** | **5h 12m** |

EMA went from 9.49 → 15.97 in 65k steps — ~665× perplexity regression. Run terminated manually.

## Design-time claim that failed

Iter 168's `--sas-schedule` flag was validated at 5k–10k steps as "long-horizon safe" via an example in the help text that set α transitions AT the same steps as the SLC T transitions (both at 40% / 60% of total). The implicit assumption: LR warmup attached to T transitions would also cover SAS α jumps.

## Actual mechanism

1. **Co-located transitions**: SLC T (256→512) and SAS α (0.3→0.5) were scheduled at step 260,000 on a percentage basis that tracks SLC exactly. The iter 138 v2 500-step post-transition LR mini-warmup (`cfg.slcLastTransitionStep` in `chiron_main.cpp`) fires **only for T/L transitions**; SAS α jumps do not set `slcLastTransitionStep`, so they get no warmup.
2. **Compound update-magnitude change**: T:256→512 doubles attention update magnitude; SAS α:0.3→0.5 further increases it ~1.67×. Combined ~3.3× sudden jump in attention-path gradient scale, only partially compensated by the T-triggered warmup.
3. **FACE preconditioner lag**: FACE β_row=0.98 has a ~50-step effective lookback. At a compound transition, FACE's Zipf statistics are still baked for the old regime and briefly apply a mis-calibrated σ.

Net result: the first few hundred post-transition updates are ~3× larger than Adam/FACE expects, destabilizing all attention-weight matrices. Once the weights leave the basin of attraction, FACE's Zipf regularization can't recover the trajectory.

## Why the 5k-step validation missed it

All previous SAS runs validating `--sas-schedule` were at ≤10,000 steps. At that scale:
- T and SAS transitions happened at step 2000/3000 — only 65-95s apart in wall clock
- 100-step LR warmup covered a larger *fraction* of the inter-transition window
- Loss oscillation band at early training is wide (0.2–1.1 nat) — a compound shock of +2–3 nat at step 2000 looks like normal noise and gets corrected within 500 steps
- **The 650k run spent 3h at T=256 α=0.3 before the shock, and the basin was much deeper → recovery not possible**

This is surprise taxonomy B ("measurement artifacts from short horizons") plus D ("integration failure not caught by primitive parity").

## Fix applied (iter 169)

### Code fix (`glades-trainer/trainer/chiron_main.cpp`)

SAS α transitions now update `slcLastTransitionStep` and inherit the 500-step LR mini-warmup. Log message also fixed from "100-step" to "500-step" (`miniWarmup = 500` since iter 138 v2).

```cpp
// iter 169: also trigger the SLC mini-warmup (reuse slcLastTransitionStep)
// because an α jump is an update-magnitude change of the same character
// as a T or L jump.
cfg.sasAlpha = newA;
cfg.slcLastTransitionStep = step;
```

### Schedule fix (`glades-trainer/run.sh`)

For `--steps >= 10000` with no user-supplied `--sas-schedule`, auto-generate a staggered schedule: SAS α transitions placed **+6% of total steps after** each SLC T transition. This guarantees the T-triggered 500-step warmup stabilizes before the α jump lands.

```bash
SAS_STAGGER=$((STEPS * 6 / 100))
CHIRON_SAS_SCHED="0.3@0,0.5@$((SLC_T1_STEP+SAS_STAGGER)),1.0@$((SLC_T2_STEP+SAS_STAGGER))"
```

The help text example is also staggered now: `"0.3@0,0.5@2300,1.0@3300"` rather than the co-located `"0.3@0,0.5@2000,1.0@3000"`.

### Checkpoint quarantine

The post-divergence `chiron_1.84B.ckpt.final` (saved at user's Ctrl-C at step 367,429 with EMA 15.97) was renamed to `chiron_1.84B.ckpt.divergent_step367429` so `run.sh`'s auto-resume logic cannot accidentally load it. The pre-divergence `chiron_1.84B.ckpt.step260000` (EMA 9.49) remains available for warm-start experiments.

## Methodology lessons

1. **Every schedule that changes update magnitude must inherit `slcLastTransitionStep`.** The existing LR mini-warmup is the right abstraction; extend it whenever a new curriculum axis is added (future: adaptive SAS, block-stochastic SAS, etc.).
2. **Schedules on different axes must not co-locate transitions.** iter 147 established this for T vs L; iter 169 extends it to T vs SAS. Generalize: any two axes with mini-warmup-dependent stability must stagger ≥ warmup-steps apart.
3. **"Long-horizon safe" claims made from ≤10k-step validations are premature.** The iter 168 label was earned at 5–10k but not at 650k. Update the 3-gate validation to require a ≥ 50k-step horizon point for curriculum-schedule claims when the flagship runs are ≥ 100k steps.
4. **Log message discrepancies are a code-rot signal.** "100-step LR warmup" while `miniWarmup = 500` went unnoticed because the 100 in the log looked plausible at 10k horizons. At 650k, each log line gets audited by a human — this is when rot becomes visible.

## Artifacts

- Code: `glades-trainer/trainer/chiron_main.cpp` (SAS warmup hook)
- Recipe: `glades-trainer/run.sh` (auto-stagger + help text)
- Checkpoint state: `database/checkpoints/chiron_1.84B/{step130000, step260000, divergent_step367429}`
- Pre-divergence loss trajectory preserved in the 14:11 progress-report terminal transcript
