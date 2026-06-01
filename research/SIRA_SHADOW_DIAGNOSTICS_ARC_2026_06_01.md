# SIRA Shadow Trajectory Diagnostics Arc (2026-06-01)

## Status

Default-off SIRA Phase-0 shadow trajectory diagnostics are implemented in
`glades-ml` config and `glades-trainer` runtime.  This is detached logging only:
`loss=none gradients=none`; it does not activate the full SIRA trajectory loss.

Runtime flags:

```text
--sira-shadow-diagnostics
--sira-log-every N
--sira-probe-layers 0,4,8,12,16,20,23
--sira-position-buckets 8
--sira-shadow-smoke
```

Report helper:

```text
scripts/sira_log_report.py logs/fp8_sira_shadow_2000step_20260601_091430/sira_shadow.log \
  > logs/fp8_sira_shadow_2000step_20260601_091430/sira_report.md
```

The trainer logs selected layer/position-bucket `rms(p)`, `rms(q)`,
`rms(shear)`, normalized energy, p/q balance, and action proxy.  When BF16
logit storage is active, it also emits a step-level position-bucket NLL proxy
from the current probabilities.  CUDA graph capture is disabled when the shadow
path is active because it performs detached host reductions at log cadence.

## Verification

Commands run:

```text
cd /home/robert/dev/glades-ml
cmake --build build -j $(nproc)
cmake --install build
./unit-tests/test.sh chiron-sira

cd /home/robert/dev/glades-trainer
bash build.sh
scripts/sira_shadow_smoke.sh
# includes BF16-storage pos-NLL proxy assertion: nll_proxy=bf16-position-bucket
scripts/sira_config_smoke.sh
scripts/sira_training_loss_smoke.sh
scripts/phs_shadow_smoke.sh
scripts/ptoc_shadow_smoke.sh
scripts/sira_log_report.py logs/fp8_sira_shadow_2000step_20260601_091430/sira_shadow.log \
  > logs/fp8_sira_shadow_2000step_20260601_091430/sira_report.md
```

All completed successfully.

## FP8 100-step smoke

Artifact:

```text
/home/robert/dev/glades-trainer/logs/fp8_sira_shadow_100step_20260601_091034/
```

Summary:

| mode | VRAM | val NLL | warm tok/s | bad lines |
|---|---:|---:|---:|---:|
| baseline | 14.49 / 15.56 GB | 9.7655 | 28119.0 | 0 |
| SIRA shadow | 14.49 / 15.56 GB | 9.7654 | 27706.8 | 0 |

The final step coincides with SIRA shadow logging and reports a low instantaneous
`tok/s=9291`; median warm throughput remains near baseline.

## FP8 500-step pos-NLL proxy gate

After the initial SIRA shadow implementation, the shadow path was extended to
emit a BF16-storage step-level position-bucket NLL proxy alongside the per-layer
phase metrics.

Artifact:

```text
/home/robert/dev/glades-trainer/logs/fp8_sira_shadow_nll_500step_20260601_134311/
```

Summary:

| mode | VRAM | step500 loss | val NLL | warm tok/s | non-diagnostic warm tok/s | bad lines |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 14.49 / 15.56 GB | 7.7079 | 7.6851 | 28077.9 | 28077.7 | 0 |
| SIRA shadow + pos-NLL | 14.49 / 15.56 GB | 7.7075 | 7.6849 | 27843.7 | 28052.8 | 0 |

Delta SIRA-shadow minus baseline:

```text
VRAM:                      +0.00 GB
train loss:                -0.0004
val NLL:                   -0.0002
warm tok/s mean:           -234.2 tok/s (-0.83%)
warm non-diagnostic tok/s:  -24.9 tok/s (-0.09%)
```

SIRA emitted 35 layer rows plus 5 position-NLL proxy rows.  Final proxy:

```text
step500 pos_nll=[7.716/7.589/7.722/7.669/7.745/7.630/7.727/7.786]
mean=7.6980 max=7.7860 source=bf16-position-bucket
```

Regression flags: none.  The additional NLL proxy is log-cadenced, detached,
and has no measurable VRAM effect.

## FP8 2k gate

Artifact:

```text
/home/robert/dev/glades-trainer/logs/fp8_sira_shadow_2000step_20260601_091430/
```

Summary:

| mode | VRAM | step2000 loss | val NLL | warm tok/s | non-diagnostic warm tok/s | bad lines |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 14.49 / 15.56 GB | 4.6015 | 4.9461 | 28045.5 | 28045.3 | 0 |
| SIRA shadow | 14.49 / 15.56 GB | 4.6007 | 4.9502 | 27834.4 | 28027.1 | 0 |

Delta SIRA-shadow minus baseline:

```text
VRAM:                      +0.00 GB
train loss:                -0.0008
val NLL:                   +0.0041
warm tok/s mean:           -211.1 tok/s (-0.75%)
warm non-diagnostic tok/s:  -18.2 tok/s (-0.06%)
wall time:                 +24.6 s over 2k steps
```

Regression flags: none.  The total warm mean includes the 20 deliberately slow
SIRA logging steps.  Non-diagnostic steps are essentially unchanged.

## Diagnostic trend

SIRA emitted 140 layer snapshots: 7 probe layers × 20 logging checkpoints.
Probe layers were `[0,4,8,12,16,20,23]`.

Global observed maxima through 2k:

```text
energy_max:             3.0546
abs(balance) max:       0.9259
abs(action) max:        0.8605
```

Layer/bucket energy and balance are high early and become more uniform by 2k:

```text
layer 0  energy_max: 2.7185 -> 1.5317, |balance|max: 0.7437 -> 0.3758
layer 4  energy_max: 2.9916 -> 1.0484, |balance|max: 0.8030 -> 0.0460
layer 8  energy_max: 3.0377 -> 1.0669, |balance|max: 0.8122 -> 0.0627
layer 12 energy_max: 3.0399 -> 1.0743, |balance|max: 0.8126 -> 0.0692
layer 16 energy_max: 3.0522 -> 1.0889, |balance|max: 0.8151 -> 0.0819
layer 20 energy_max: 3.0546 -> 1.0947, |balance|max: 0.8155 -> 0.0869
layer 23 energy_max: 3.0519 -> 1.0960, |balance|max: 0.8149 -> 0.0889
```

The terminal layer action proxy remains high but stable:

```text
layer 23 |action|max: 0.8605 at step100 -> 0.7245 at step2000
```

## Interpretation

The implemented SIRA shadow path is safe enough to keep as default-off telemetry:
no VRAM regression, no stability failures, and no meaningful non-diagnostic
throughput hit.  The diagnostics are also informative: early position-bucket
phase nonuniformity decays sharply during training.

This does not justify active full SIRA yet.  The next evidence gate should be a
5k SIRA-shadow run or an active energy+balance SIRA pilot only after deciding
which observed bucket/layer signals should be targeted.
