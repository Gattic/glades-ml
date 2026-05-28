# PHS Default-Enablement Result (2026-05-28)

## Decision

**Do not make PHS globally default-on for production/flagship training yet.**

Adopt the narrower policy:

1. **Research/diagnostic templates:** PHS shadow diagnostics are cleared for default inclusion when the purpose of the run is observability/debugging.
2. **Production/flagship default:** keep PHS explicitly opt-in.
3. **Active PHS weighting/controller:** do not implement or enable by default from the current evidence.

Recommended opt-in diagnostic flags remain:

```bash
--phs-shadow-diagnostics \
--phs-data-groups 4 \
--phs-position-buckets 8 \
--phs-log-every 500 \
--phs-ema-decay 0.95 \
--phs-group-mode batch-quantile
```

Rationale: shadow PHS is stable and near-zero overhead, but it is detached logging only, does not improve the model by construction, disables CUDA graph capture when active, and revealed large late-run `qout` excursions whose operational thresholds are not yet understood. This is enough for research observability, not enough for production default-on behavior or active control.

## Related plan

Execution followed `research/PHS_DEFAULT_ENABLEMENT_PLAN_2026_05_27.md` through Stage 2. Stage 3/4 active-PHS work was **not entered** because the shadow metrics did not establish a validated intervention policy and active PHS is not implemented.

## Stage 0 — default-off contract

Commands run:

```bash
cd /home/robert/dev/glades-ml
cmake --build unit-tests/build -j2
./unit-tests/test.sh chiron-phs
git diff --check

cd /home/robert/dev/glades-trainer
bash build.sh
scripts/phs_shadow_smoke.sh
sh run.sh flagship --phs-config-smoke \
  --phs-shadow-diagnostics --phs-data-groups 2 --phs-position-buckets 4 \
  --phs-log-every 3 --phs-ema-decay 0.8 --phs-group-mode id
git diff --check
```

Results:

- `glades-ml` unit build passed.
- `./unit-tests/test.sh chiron-phs` passed (`GLADES_ML_CHIRON_PHS_RC=0`).
- `glades-trainer` build passed.
- `scripts/phs_shadow_smoke.sh` passed (`TRAINER_PHS_SMOKE_RC=0`).
- `--phs-group-mode id` override smoke passed (`OVERRIDE_RC=0`, `OVERRIDE_GREP_RC=0`).
- `git diff --check` passed in both repos.

## Stage 1 — 2000-step research diagnostic gate

### Logs

Timestamp: `20260527_132817`.

Disabled runs:

- `logs/phs_default_stage1_T16384_2000step_seed1337_disabled_20260527_132817.log`
- `logs/phs_default_stage1_T16384_2000step_seed2024_disabled_20260527_132817.log`
- `logs/phs_default_stage1_T16384_2000step_seed777_disabled_20260527_132817.log`

PHS runs:

- `logs/phs_default_stage1_T16384_2000step_seed1337_phs_20260527_132817.log`
- `logs/phs_default_stage1_T16384_2000step_seed2024_phs_20260527_132817.log`
- `logs/phs_default_stage1_T16384_2000step_seed777_phs_20260527_132817.log`
- `logs/phs_default_stage1_T16384_2000step_seed4242_phs_20260527_132817.log`

All seven runs completed with no NaN/Inf/bad-grad/grad-skip/overflow lines.

### Disabled vs PHS, 2000 steps

PHS minus disabled, final training row at step 1901.

| seed | EMA Δ | loss Δ | best Δ | mean tok/s Δ | wall Δs | log KB disabled/PHS |
|---:|---:|---:|---:|---:|---:|---:|
| 1337 | +0.0054 | +0.0170 | -0.0569 | -39.5 | +1.6 | 7.3 / 14.3 |
| 2024 | -0.0092 | -0.0121 | +0.0399 | -1.1 | +0.1 | 7.3 / 14.3 |
| 777 | +0.0017 | +0.0057 | -0.0261 | -10.7 | +0.5 | 7.3 / 14.3 |
| mean | -0.0007 | +0.0035 | -0.0144 | -17.1 | +0.7 | +7.0 KB |

Interpretation: no meaningful learning difference; logging overhead was below 0.1%.

### PHS final metrics, 2000 steps

| seed | count CV | shares | logpq mean/std/range | qout mean/std/range | pos-NLL mean/spread |
|---:|---:|---|---|---|---|
| 1337 | 0.094 | `[.250,.250,.250,.250]` | 3.944 / 0.046 / 3.877..4.085 | 4.901 / 0.195 / 4.59..5.35 | 5.271 / 0.364 |
| 2024 | 0.094 | `[.250,.250,.250,.250]` | 3.902 / 0.046 / 3.840..4.024 | 4.895 / 0.231 / 4.59..5.48 | 5.258 / 0.421 |
| 777 | 0.094 | `[.250,.250,.250,.250]` | 3.946 / 0.057 / 3.881..4.149 | 4.952 / 0.157 / 4.66..5.18 | 5.284 / 0.393 |
| 4242 | 0.094 | `[.250,.250,.250,.250]` | 3.936 / 0.048 / 3.871..4.061 | 5.070 / 0.566 / 4.49..6.53 | 5.278 / 0.445 |

Stage 1 decision: **pass for research-default diagnostics**.

## Stage 2 — 10000-step production diagnostic gate

### Logs

Timestamp: `20260527_154455`.

Disabled runs:

- `logs/phs_default_stage2_T16384_10000step_seed1337_disabled_20260527_154455.log`
- `logs/phs_default_stage2_T16384_10000step_seed2024_disabled_20260527_154455.log`
- `logs/phs_default_stage2_T16384_10000step_seed777_disabled_20260527_154455.log`

PHS runs:

- `logs/phs_default_stage2_T16384_10000step_seed1337_phs_20260527_154455.log`
- `logs/phs_default_stage2_T16384_10000step_seed2024_phs_20260527_154455.log`
- `logs/phs_default_stage2_T16384_10000step_seed777_phs_20260527_154455.log`

All six runs completed with no NaN/Inf/bad-grad/grad-skip/overflow lines.

### Disabled vs PHS, 10000 steps

PHS minus disabled. Final validation used `--val-every 1000 --val-batches 4`.

| seed | train EMA Δ | train loss Δ | best Δ | final val disabled | final val PHS | val Δ | max abs val-bucket Δ | mean tok/s Δ | wall Δs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1337 | +0.0046 | +0.0185 | -0.0168 | 4.2403 | 4.2349 | -0.0054 | 0.0300 | -17.5 | +3.4 |
| 2024 | -0.0063 | +0.0314 | -0.0077 | 4.2333 | 4.2286 | -0.0047 | 0.0700 | -14.4 | +2.8 |
| 777 | -0.0039 | +0.0224 | -0.0172 | 4.2269 | 4.2301 | +0.0032 | 0.0600 | -10.9 | +2.2 |
| mean | -0.0019 | +0.0241 | -0.0139 | — | — | -0.0023 | 0.0533 | -14.3 | +2.8 |

Interpretation: no meaningful loss/NLL difference; throughput loss was ~0.05%; wall overhead was seconds over ~5680s.

### PHS final metrics, 10000 steps

| seed | count CV | shares | logpq mean/std/range | qout mean/std/range | pos-NLL mean/spread |
|---:|---:|---|---|---|---|
| 1337 | 0.038 | `[.250,.250,.250,.250]` | 4.086 / 0.232 / 3.971..4.928 | 11.293 / 1.601 / 9.61..14.59 | 3.965 / 0.244 |
| 2024 | 0.038 | `[.250,.250,.250,.250]` | 4.042 / 0.207 / 3.937..4.791 | 8.289 / 1.144 / 6.53..10.71 | 3.932 / 0.207 |
| 777 | 0.038 | `[.250,.250,.250,.250]` | 4.244 / 0.262 / 4.125..5.195 | 8.324 / 0.898 / 6.57..9.97 | 4.000 / 0.158 |

Late-run qout maxima by seed reached:

- seed 1337: `17.57`
- seed 2024: `12.65`
- seed 777: `13.63`

Stage 2 10k decision: **quality/stability/throughput pass**, but qout excursions are large enough that production default-on should remain a policy decision, not an automatic code default.

## Stage 2 — 30000-step flagship check

Two 30k PHS runs were made:

1. Non-regstack diagnostic run:
   - `logs/phs_default_stage2_T16384_30000step_seed1337_phs_20260528_011615.log`
   - This omitted `--zloss-coef 1e-4 --qk-norm`, so it is not comparable to the registered regstack baseline.
   - It completed cleanly with final val NLL `4.0315`.
2. Regstack flagship-comparable run:
   - `logs/phs_default_stage2_regstack_T16384_30000step_seed1337_phs_20260528_060122.log`
   - Included `--zloss-coef 1e-4 --qk-norm`.
   - Completed cleanly with final val NLL `3.5598`.

### Regstack 30k comparison

Baseline record: `logs/regstack_b5_30k.log`.

| metric | baseline disabled | PHS shadow | Δ PHS-baseline |
|---|---:|---:|---:|
| final val NLL | 3.5734 | 3.5598 | -0.0136 |
| final bpb | 1.2888 | 1.2839 | -0.0049 |
| final ppl | 35.64 | 35.16 | -0.48 |
| acc1 | 0.1374 | 0.1385 | +0.0011 |
| acc5 | 0.5494 | 0.5518 | +0.0024 |
| acc10 | 0.8365 | 0.8364 | -0.0001 |
| mean tok/s | 28063.2 | 28065.0 | +1.8 |
| wall | 17514.8s | 17513.5s | -1.3s |

Validation position-bucket deltas, PHS minus baseline:

```text
[+0.00, +0.02, -0.03, -0.02, -0.02, -0.01, -0.02, -0.02]
```

No bucket regressed by more than `0.03` nat.

Important interpretation: PHS is detached diagnostics only, so this run should not be read as proof that PHS improves NLL. It proves that enabling PHS logging did not harm this 30k regstack run and stayed within throughput/wall budgets.

### Regstack 30k PHS health

| step | count CV | shares | logpq mean/std/range | qout mean/std/range | pos-NLL mean/spread |
|---:|---:|---|---|---|---|
| 10000 | 0.038 | `[.250,.250,.250,.250]` | 3.428 / 0.023 / 3.389..3.470 | 12.514 / 0.712 / 11.58..14.25 | 3.474 / 0.338 |
| 20000 | 0.144 | `[.250,.250,.250,.250]` | 3.826 / 0.050 / 3.743..3.937 | 18.195 / 1.524 / 15.22..20.63 | 3.730 / 0.448 |
| 30000 | 0.125 | `[.250,.250,.250,.250]` | 4.041 / 0.043 / 3.941..4.118 | 36.601 / 2.616 / 30.53..39.94 | 3.594 / 0.293 |

Maximum observed qout over PHS checkpoints: `39.94` at step 30000.

This rising qout is finite and did not coincide with observed training failure, but it is not yet tied to an operational threshold or intervention. This is the main reason not to promote PHS to production default-on despite the clean 30k run.

## Final gate assessment

| Gate | Result |
|---|---|
| Disabled/default-off contract | PASS |
| Batch-quantile group balance | PASS |
| 2000-step stability | PASS |
| 2000-step overhead | PASS |
| 10000-step multi-seed stability | PASS |
| 10000-step NLL/loss non-regression | PASS |
| 10000-step overhead | PASS |
| 30000-step regstack non-regression | PASS |
| Production CUDA-graph compatibility | FAIL/UNRESOLVED: PHS logging disables graph capture |
| PHS active-controller evidence | FAIL/NOT ENTERED: no validated intervention policy |
| qout operational threshold | UNRESOLVED: late qout rose to ~40 in clean 30k |

## Final recommendation

Follow-up qout calibration/reporting is documented in
`research/PHS_QOUT_CALIBRATION_AND_DEFAULT_POLICY_2026_05_28.md`. That pass
uses the new `glades-trainer/scripts/phs_log_report.py` parser and confirms the
policy split: research diagnostic templates may default-enable PHS shadow
logging, but production/flagship training should keep PHS opt-in.

- **Code/config default:** keep `phsShadowDiagnostics=false` and keep trainer PHS opt-in.
- **When PHS is enabled:** keep `batch-quantile` as the default group mode.
- **Research templates:** it is reasonable to include PHS shadow diagnostics by default in diagnostic/research run wrappers with `--phs-log-every 500`.
- **Production/flagship recipe:** do not make PHS globally default-on until CUDA-graph policy and qout thresholds are resolved.
- **Active PHS:** do not implement or enable by default yet. First build alert-only analyses that test whether qout/logpq excursions predict validation buckets, gradient spikes, or instability across more 30k traces.

## Suggested next work

1. Add a parser/report script for PHS logs so qout/logpq trends are summarized consistently.
2. Define qout/logpq alert thresholds from clean long-run distributions.
3. Compare batch-quantile grouping against a static frequency-remapped grouping table if semantic token group identity matters.
4. Revisit production default-on only after deciding whether PHS should automatically disable under `--cuda-graphs` or whether production accepts losing graph capture for observability.
