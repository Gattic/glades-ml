# CHIRON ECHO Gradient Attribution — 2026-07-16

## Question

Which backward component caused the anomalous gradient norms previously observed for hard ECHO at `lambda=0.03`, and which parameter blocks carried the excess?

## Validity and protocol

- Matched flagship baseline and hard-ECHO arms used one trainer binary:
  `98e226c196f3ca14fadc4a006d002b7deb5c9110d843e82ab85c93c5b611e9c0`.
- Both arms used the same data, seed, 24-layer recipe, accumulation of 4, and probe steps 582/665/748.
- Each probe logged the original full gradient and detached CE+Z-loss, SIRA-only, and ECHO-only backward passes.
- Before detached replay, every optimizer-consumed gradient buffer was copied to host. The exact bytes were restored before clipping/Adam.
- Restoration relative errors were `6.78e-8` to `4.22e-7`, all below the preregistered `1e-5` bar.
- Both arms reached step 750 with no NaN, divergence, explicit gradient skip, replay failure, or parity failure.

The initial production implementation failed before this run for two independent reasons:

1. SIRA-only attempted to zero the unallocated `dlogits_bf` owner when FP8 readout had aliased the live gradient storage to `logits_bf`. It now zeros `dlogits_bf_data()`.
2. A repeated production SCFA/PIED full backward had norm relative error `2.99e-5`, above the restoration bar. Detached probes therefore use exact gradient snapshot/restore rather than a second full backward as the optimizer input.

## Global component norms

| Step | Baseline full | lambda=.03 full | Ratio | lambda=.03 CE+Z | SIRA-only | ECHO-only | `(SIRA+ECHO)/full` upper bound |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 582 | 0.531416 | 0.654879 | 1.232x | 0.655917 | 0.005768 | 0.007212 | 1.98% |
| 665 | 0.590041 | 0.428918 | 0.727x | 0.428811 | 0.004876 | 0.006090 | 2.56% |
| 748 | 0.662852 | 1.040997 | 1.570x | 1.039519 | 0.004936 | 0.003666 | 0.83% |

The treatment full norm differs from its CE+Z norm by only `-0.16%`, `+0.03%`, and `+0.14%` at the three steps. Thus the instantaneous full gradient is overwhelmingly CE+Z-driven. At the reproduced step-748 spike, ECHO-only is only `0.35%` of the full norm and SIRA-only is `0.47%`.

The old step-665 treatment spike did not reproduce in this matched rerun: treatment was below baseline (`0.429` versus `0.590`). This limits the conclusion about the exact old event, but not the component attribution of the newly observed step-748 event.

## Full-gradient module localization

| Step | Module | Baseline norm | lambda=.03 norm | Share of positive treatment sumsq excess |
|---:|---|---:|---:|---:|
| 582 | Embedding/readout `E` | 0.478844 | 0.560013 | 57.6% |
| 582 | ReLN beta | 0.212439 | 0.323226 | 40.5% |
| 748 | Embedding/readout `E` | 0.630943 | 0.850636 | 50.5% |
| 748 | ReLN beta | 0.166979 | 0.573570 | 46.7% |

`E` and ReLN beta jointly explain 98.1% of the positive sumsq excess at step 582 and 97.2% at step 748. At step 665, all major treatment modules were below baseline, consistent with the absent spike.

The same localization appears in CE+Z alone. For example, at step 748:

- Full: `E=0.850636`, ReLN beta `=0.573570`
- CE+Z: `E=0.849589`, ReLN beta `=0.572546`
- SIRA: `E=0.003457`, ReLN beta `=0.003389`
- ECHO: `E=0.002741`, ReLN beta `=0.002271`

## Top tensors

| Step | Arm | `dE` norm | `L00.dbeta` norm | Next dominant tensors |
|---:|---|---:|---:|---|
| 582 | Baseline | 0.478844 | 0.115273 | `L01..L03.dbeta` |
| 582 | lambda=.03 | 0.560013 | 0.174818 | `L01..L03.dbeta` |
| 665 | Baseline | 0.498372 | 0.150449 | `L01..L03.dbeta` |
| 665 | lambda=.03 | 0.404664 | 0.064339 | `L01..L03.dbeta` |
| 748 | Baseline | 0.630943 | 0.091198 | `L01..L03.dbeta` |
| 748 | lambda=.03 | 0.850636 | 0.287092 | `L01..L03.dbeta` |

## Interpretation

### Supported

- The reproduced step-748 excess is a **CE+Z-loss transient localized almost entirely to the tied embedding/readout matrix and ReLN beta**, especially `L00.dbeta`.
- The direct hard-ECHO and SIRA backward fields are far too small to be the proximal source of the spike.
- There is no evidence for a large same-step constructive interaction: full and CE+Z norms agree to within 0.16% in the treatment arm.

### Not established

- The probes do not rule out an **indirect trajectory effect**: hundreds of earlier ECHO updates can move the model to a state where the ordinary CE gradient is larger on a later batch.
- The old step-665 event did not reproduce, so its exact tensor-level cause remains unresolved rather than retroactively attributed.
- CE and Z-loss were probed together; the instrumentation does not separate their fields. With `zloss-coef=1e-4`, CE is the likely dominant term, but that was not independently measured.

## Decision

Do not promote Huber smoothing, an ECHO-only clip, or a SIRA mitigation as a response to these spikes. They do not target the measured proximal field. The existing global clip already handled the step-748 CE transient (`||g||=1.041`, scale `0.480`) without NaN or a skipped update.

Hard ECHO `lambda=0.03` also finished worse on validation in this rerun:

- Baseline final NLL: `7.4435`
- ECHO final NLL: `7.4741`
- Delta: `+0.0306`

Together with the earlier sweep no-go, this does not justify more mitigation tuning. If trajectory causality becomes a separate research goal, use multiple seeds and checkpointed branches that disable ECHO before divergence; do not infer it from same-step component magnitudes alone.

## Evidence

- `/home/robert/dev/glades-trainer/logs/echo_grad_probe_baseline_exact_20260716.log`
- `/home/robert/dev/glades-trainer/logs/echo_grad_probe_lambda003_exact_20260716.log`
- `/home/robert/dev/glades-trainer/logs/echo_grad_probe_pair_exact_20260716.log`
- Run deck: `run-echo-component-gradient-exact-restore-0448`
