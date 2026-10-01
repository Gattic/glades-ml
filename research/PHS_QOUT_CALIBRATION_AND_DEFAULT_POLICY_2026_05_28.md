# PHS qout Calibration and Default Policy (2026-05-28)

## Decision after short-test program

Use **PHS shadow diagnostics by default in research/diagnostic templates**, but keep PHS **off by default for production/flagship training**.

Do **not** run more 30k jobs just to decide this logging-only feature. Current evidence is sufficient for the policy split:

- PHS is detached shadow logging, not a training intervention.
- 2k multi-seed and 10k multi-seed tests show no stability or quality regression and negligible overhead.
- One regstack-compatible 30k PHS shadow run is clean and non-regressing.
- qout grows substantially in long clean runs, so it is useful as a diagnostic but not yet calibrated enough to justify production default-on alerts.
- PHS host downloads/logging remain incompatible with CUDA graph capture when active.

## New parser/report tool

Added in `glades-trainer`:

```text
scripts/phs_log_report.py
```

Usage:

```bash
cd ~/dev/glades-trainer
scripts/phs_log_report.py 'logs/phs*.log' > /tmp/phs_log_report_all_20260528.md
```

The parser extracts:

- bad/stability lines: NaN/Inf/bad-grad/grad-skip/overflow;
- training loss/EMA/best/grad/tok-s;
- validation NLL and position buckets;
- PHS group/bucket counts, `logpq`, `qout`, and `ema_qout`;
- qout threshold crossings;
- descriptive qout-vs-validation correlations when validation/PHS checkpoints align.

The script is intentionally dependency-free Python 3.

## Logs parsed

The all-log report parsed `22` `logs/phs*.log` files from `glades-trainer`, including:

- 2k ID-bin vs batch-quantile grouping ablations;
- Stage 1 2k disabled/PHS runs;
- Stage 2 10k disabled/PHS runs;
- non-regstack 30k PHS run;
- regstack-compatible 30k PHS run;
- 30-step smoke.

Summary from the parser:

```text
Parsed logs: 22 (16 with PHS checkpoints)
Clean logs: 22/22
bad/stability lines total: 0
qout max distribution across logs:
  min=5.48, median=5.74, p90=23.84, max=39.94
```

Top qout events:

| log | step | qout max | corr(qout,val) |
|---|---:|---:|---:|
| `phs_default_stage2_regstack_T16384_30000step_seed1337_phs_20260528_060122.log` | 30000 | 39.94 | -0.616 |
| `phs_default_stage2_T16384_30000step_seed1337_phs_20260528_011615.log` | 29000 | 30.11 | -0.659 |
| `phs_default_stage2_T16384_10000step_seed1337_phs_20260527_154455.log` | 9000 | 17.57 | -0.624 |
| `phs_default_stage2_T16384_10000step_seed777_phs_20260527_154455.log` | 9500 | 13.63 | -0.602 |
| `phs_default_stage2_T16384_10000step_seed2024_phs_20260527_154455.log` | 8500 | 12.65 | -0.609 |
| `phs_default_stage1_T16384_2000step_seed4242_phs_20260527_132817.log` | 2000 | 6.53 | n/a |

The negative qout-vs-val correlations above are descriptive only and mostly reflect training progress: qout rises while validation NLL generally falls. They should not be interpreted as causal or as proof that high qout is good.

## qout interpretation

Current `qout` is:

```text
max(|q|) / rms(q)
```

computed per PHS group × position bucket over terminal `q_L`.

For a typical cell with approximately `500 tokens × 2048 dims ≈ 1M coordinates`, a Gaussian-ish max/RMS scale is about:

```text
sqrt(2 log 1e6) ≈ 5.2
```

That explains why early/short clean runs cluster around `~5–6`. Larger values mean the terminal `q` distribution has heavier/sparser tails than a simple iid Gaussian model.

## qout bands

These are **operational diagnostic bands**, not failure thresholds:

| qout range | interpretation | policy |
|---:|---|---|
| `<=6` | expected high-dimensional max/RMS regime | normal |
| `6..10` | elevated tail; observed in clean longer runs | monitor |
| `10..20` | high tail; observed in clean 10k runs | summarize in reports |
| `20..40` | extreme but observed in clean 30k runs | investigate trend/context, do not auto-fail |
| `>40` | outside current clean-run envelope | alert until calibrated |

Important empirical point: the clean regstack-compatible 30k run reached `qout=39.94` at step 30000 and still finished with final val NLL `3.5598`, better than the disabled baseline record `3.5734`. Therefore high qout is not currently a failure criterion.

## Why research default-on is OK

The short-test evidence answers the operational cost question:

- 2k multi-seed: no stability failures; mean overhead around `-17 tok/s`, `+0.7s` wall.
- 10k multi-seed: no stability failures; mean final val NLL delta `-0.0023`; mean overhead around `-14 tok/s`, `+2.8s` wall.
- 30k regstack-compatible check: no stability failures; final val NLL `3.5598`; mean tok/s `28065.0` vs baseline `28063.2`.
- Batch-quantile grouping gives stable balanced group shares and low count CV.

For research/debugging, this is enough: PHS provides phase-state observability at negligible overhead.

## Why production default-on is still no

Production default-on has a higher bar than research default-on. PHS should remain opt-in for production because:

1. **CUDA graph policy is unresolved.** PHS logging uses host downloads and disables graph capture when active.
2. **qout alerts are not calibrated to action.** Values up to `~40` occur in clean runs, so a naive alert would create noise.
3. **PHS is not causal.** Since it is detached logging, it does not improve the model by being enabled; it only improves observability.
4. **Operator noise matters.** Default production logs should avoid extra diagnostic volume unless there is a clear runbook.

## Final policy

### Code defaults

Keep:

```text
phsShadowDiagnostics = false
```

Keep batch-quantile as the default **only when PHS is explicitly enabled**:

```text
--phs-group-mode batch-quantile
```

### Research templates

Use PHS shadow diagnostics by default for diagnostic/research runs:

```bash
--phs-shadow-diagnostics \
--phs-data-groups 4 \
--phs-position-buckets 8 \
--phs-log-every 500 \
--phs-ema-decay 0.95
```

### Production/flagship recipes

Keep PHS opt-in until both are true:

- CUDA graph interaction is explicitly resolved;
- qout/logpq alert bands have a runbook with low false-positive rate.

## Next useful work

1. Add a lightweight PHS report call to diagnostic run wrappers, not production training.
2. Collect more clean-run qout distributions opportunistically from future PHS-enabled research runs.
3. Test whether qout/logpq spikes predict real failures by including known unstable runs once PHS was enabled for them.
4. If active PHS is revisited, start with alert-only or bounded detached token weighting; do not inject hidden-state gradients.
