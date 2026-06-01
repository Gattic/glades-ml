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

## FP8 2k pos-NLL follow-up

Artifact:

```text
/home/robert/dev/glades-trainer/logs/fp8_sira_shadow_nll_2000step_20260601_135541/
```

SIRA-shadow-only follow-up after the pos-NLL proxy addition:

```text
steps:                       2000
VRAM:                        14.55 / 15.56 GB
train loss / EMA:            4.5998 / 4.8125
val NLL / BPB / PPL:         4.9614 / 1.7895 / 142.80
warm tok/s mean:             27851.3
warm non-diagnostic tok/s:   28043.3
bad/stability lines:         0
SIRA layer rows:             140
SIRA pos-NLL rows:           20
regression flags:            none
```

Final SIRA position-NLL proxy:

```text
step2000 pos_nll=[4.455/4.567/4.556/4.314/4.666/4.747/4.754/4.645]
mean=4.5880 max=4.7540 source=bf16-position-bucket
```

The 2k run was not paired with a fresh same-binary baseline, so treat the final
validation value as a stability/checkpoint observation rather than an attribution
claim.  It remains inside the prior 2k FP8 clean-run noise envelope and produced
no OOM, fallback, NaN/Inf, grad-skip, forward/backward, or stability failures.

## FP8 5k pos-NLL shadow gate

Artifact:

```text
/home/robert/dev/glades-trainer/logs/fp8_sira_shadow_nll_5000step_20260601_141803/
```

SIRA-shadow-only follow-up:

```text
steps:                       5000
VRAM:                        14.55 / 15.56 GB
train loss / EMA:            3.8666 / 4.0078
val NLL / BPB / PPL:         3.9591 / 1.4279 / 52.41
warm tok/s mean:             27834.6
warm non-diagnostic tok/s:   28024.6
bad/stability lines:         0
SIRA layer rows:             350
SIRA pos-NLL rows:           50
regression flags:            none
```

Position-NLL proxy trend:

```text
step100  mean=9.9011 max=9.966 pos_nll=[9.872/9.834/9.862/9.923/9.923/9.966/9.965/9.864]
step2600 mean=4.3900 max=5.014 pos_nll=[5.014/4.375/4.296/4.257/4.327/4.199/4.270/4.382]
step5000 mean=3.8536 max=4.037 pos_nll=[3.794/4.014/3.677/3.809/3.889/4.037/3.742/3.867]
```

Layer/bucket energy and balance remain bounded by 5k.  Non-terminal probe layers
settle near normalized energy `~1.05–1.08` and `|balance| <= 0.081`; the
terminal layer keeps a high but stable action proxy (`|action|max 0.8605 ->
0.8251`).  This supports keeping SIRA Phase-0 as safe default-off telemetry, but
still does not justify an active SIRA objective without a paired active-loss
pilot and same-seed baseline comparison.

## FP8 paired 5k baseline attribution

Artifact:

```text
/home/robert/dev/glades-trainer/logs/fp8_sira_shadow_nll_baseline_compare_5000step_20260601_161016/
```

This run pairs the previous same-binary 5k SIRA-shadow log against a fresh 5k
baseline.

| mode | VRAM | step5000 loss | val NLL | warm tok/s | non-diagnostic warm tok/s | bad lines |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 14.55 / 15.56 GB | 3.8686 | 3.9619 | 28029.6 | 28029.9 | 0 |
| SIRA shadow + pos-NLL | 14.55 / 15.56 GB | 3.8666 | 3.9591 | 27834.6 | 28024.6 | 0 |

Delta SIRA-shadow minus baseline:

```text
VRAM:                      +0.00 GB
train loss:                -0.0020
val NLL:                   -0.0028
warm tok/s mean:           -195.0 tok/s (-0.70%)
warm non-diagnostic tok/s:   -5.2 tok/s (-0.02%)
wall time:                 +60.2 s over 5k steps
```

Final validation position-bucket deltas were `[0/0/0/0/0/0/0/-0.01]` at log
precision.  Regression flags: none.  The shadow path's visible cost is confined
to the 50 diagnostic logging steps; ordinary training steps remain effectively
unchanged.

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
