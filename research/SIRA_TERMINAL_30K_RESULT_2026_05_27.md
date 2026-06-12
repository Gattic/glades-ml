# CHIRON-native terminal SIRA 30k result and flagship decision (2026-05-27)

**Status:** CANDIDATE-DEFAULT INVESTIGATION / default-off. Do **not** include
SIRA by default in the flagship yet.  The seed-4242 recovery is strong, but the
second same-recipe 30k validation seed exposed late bad-gradient guard skips.

**Candidate evaluated:** terminal-only CHIRON-native SIRA on final
`(q_L, p_L)`:

```bash
--sira-coef 1e-2 \
--sira-energy-weight 1.0 \
--sira-balance-weight 0.25 \
--sira-action-weight 0.0 \
--sira-warmup 1000
```

This is **not** the full trajectory/bucket SIRA objective. It regularizes only
the terminal phase state and is implemented as an opt-in trainer loss.

---

## Decision

SIRA should remain **default-off / experimental** for the flagship.

The quality signal is real: three successful 30k SIRA runs beat the B5
baseline, with mean final NLL `3.5491 ± 0.0113` versus the B5 record
`3.5734` (`-0.0243` mean delta). Bucket evidence was mostly favorable.

However, seed `4242` exposed a mixed-precision stability failure near step
15k. A guard now prevents Adam from corrupting weights on non-finite or
overflowed gradients, but the underlying first-cause instability is unresolved
and was not reproducibly attributable to either tied-readout or embedding
scatter in follow-up tracing. Therefore SIRA does not clear the flagship
stability bar.

---

## Baseline record

Baseline B5 record: `logs/regstack_b5_30k.log`

| Metric | B5 baseline |
|---|---:|
| Final val NLL | `3.5734` |
| Position buckets | `[3.45/3.47/3.65/3.53/3.52/3.69/3.62/3.65]` |
| Throughput | ~`28063–28072 tok/s` |
| NaNs / grad skips | none |

---

## SIRA 30k successful runs

| Run log | Final NLL | Δ vs B5 | Buckets | Throughput |
|---|---:|---:|---|---:|
| `logs/sira_30k_sira1e2_20260525_080603.log` | `3.5591` | `-0.0143` | `[3.44/3.48/3.63/3.52/3.50/3.68/3.59/3.63]` | ~`27998 tok/s` |
| `logs/sira_30k_sira1e2_seed2024_20260525_130725.log` | `3.5514` | `-0.0220` | `[3.42/3.47/3.60/3.50/3.51/3.69/3.61/3.63]` | ~`27997 tok/s` |
| `logs/sira_30k_sira1e2_seed777_20260525_203235.log` | `3.5369` | `-0.0365` | `[3.43/3.49/3.59/3.47/3.49/3.68/3.57/3.58]` | ~`27994 tok/s` |

Aggregate over successful SIRA seeds:

- Final NLL: `3.5491 ± 0.0113`
- Mean delta vs B5: `-0.0243`
- Position buckets: `20` better, `2` worse, `2` equal versus B5 across the
  three runs.
- Throughput impact: negligible (`~27.99k tok/s`, above the `26668 tok/s`
  gate and close to B5).
- Late-training SIRA loss contribution: about `0.014–0.0155`, typically
  `0.38–0.43%` of the base loss.

Single-run gate note: the first 30k SIRA run improved NLL but missed the
pre-registered single-run ship gate (`3.5591` vs required `<= 3.5534`). The
multi-seed average is strong, but stability blocks acceptance.

---

## Stability failure: seed 4242

Primary failed 30k log: `logs/sira_30k_sira1e2_seed4242_20260525_180028.log`

- Failed with NaN at step `15503`.
- Last good validation before failure: NLL `3.6220`, buckets
  `[3.72/3.77/3.63/3.55/3.57/3.58/3.63/3.53]`.
- Last logged SIRA diagnostics at step `15001` were normal:
  - SIRA loss `0.0148430569`
  - ratio `0.00384`
  - raw `1.48431`
- No logged SIRA scalar blow-up preceded the NaN.

Focused failure log: `logs/sira_stability_4242_default_lr1e4_clip05_20260526_015658.log`

- NaN at step `15028`.
- Old pattern before guard:
  - step `15010`: `||g||=2.579`
  - step `15011`: `||g||=286217`
  - step `15012`: `||g||=3.62e8`
  - step `15013`: `||g||=2.80e14`
  - step `15017+`: `||g||=inf`
  - step `15028`: CE/Z-loss and SIRA stats became NaN.
- SIRA diagnostics stayed in-family until the model gradients were already bad.

Mitigation sweeps:

| Log | Change | Outcome |
|---|---|---|
| `logs/sira_stability_4242_lr7p5e5_clip05_20260526_042353.log` | LR `7.5e-5`, clip `0.5` | completed 15600, final NLL `3.6659` |
| `logs/sira_stability_4242_lr1e4_clip04_20260526_065625.log` | LR `1e-4`, clip `0.4` | completed 15600, final NLL `3.6614` |
| `logs/sira_stability_4242_guard_default_lr1e4_clip05_20260526_100848.log` | bad-gradient guard | completed 15600, final NLL `3.6650`, `175` Adam updates skipped |

Guarded run first bad event:

- Step `15408`
- `global_sumsq=1.12646008e+24`
- `bad_groups=2`
- `dE`: `sumsq=1.12338884e+24`, norm `1.05990039e+12`
- `L00.dgamma`: `sumsq=3.07120331e+21`, norm `5.54184383e+10`
- First actual non-finite group later appeared at step `15431`.

The guard is required because calling Adam with `gradScale=0` is not safe in
the presence of NaN gradients (`0 * NaN` can still poison moments/weights).
The trainer now skips Adam entirely on non-finite/overflowed global gradients
and logs per-group breakdowns.

---

## dE contributor tracing

Trainer commit: `/home/robert/dev/glades-trainer` commit `e8a1cf4`
(`Add SIRA stability diagnostics and bad-grad guard`).

New opt-in diagnostics:

- `--sira-debug-start`, `--sira-debug-end`, `--sira-debug-every`
- `--sira-de-trace`
- `[loss-detail ...]`
- `[sira-detail ...]`
- `[grad-detail ...]`
- `[grad-skip ...]`
- `[dE-trace ...]` after tied-readout, before embedding scatter, and after
  embedding scatter.

Follow-up dE trace runs around the previous bad window did **not** reproduce
the step-15408 spike:

| Log | Step window | Result |
|---|---|---|
| `logs/sira_stability_4242_dEtrace_15400_15412_20260526_133823.log` | 15400–15412 | no bad grad; max tied/readout norm `1.19`, max post-embed norm `1.70` |
| `logs/sira_stability_4242_dEtrace_exactval_15001_15412_20260526_161053.log` | 15001–15412 | no bad grad; max tied/readout norm `1.14`, max post-embed norm `1.68` |
| `logs/sira_stability_4242_dEtrace_async_exactval_15001_15412_20260526_184426.log` | 15001–15412 | no bad grad; max tied/readout norm `1.18`, max post-embed norm `1.70` |
| `logs/sira_stability_4242_guard_repro_noDEtrace_15001_15412_20260526_211717.log` | 15001–15412 | no bad grad; step 15408 `||g||=1.198`; final val NLL `3.6532` |

At step `15408` in the traced runs:

- tied-readout contribution was normal (`~0.66` norm),
- post-embedding combined `dE` was normal (`~1.18–1.19` norm),
- no trace implicated either tied-readout or embedding scatter.

Conclusion: the original guarded-run spike is nondeterministic or build/run
sensitive. The current instrumentation can catch the split if it recurs, but
it did not isolate a stable root cause.

---

## Staged FP8 retry after Phase-0 shadow diagnostics (2026-06-01)

After adding default-off SIRA Phase-0 trajectory diagnostics and restoring FP8
readout headroom, the original terminal-only energy+balance SIRA recipe was
retried as a **staged active-loss pilot**, still default-off and not a flagship
promotion:

```text
--sira-coef 1e-2
--sira-energy-weight 1.0
--sira-balance-weight 0.25
--sira-action-weight 0.0
--sira-warmup 1000
--lr 7.5e-5
--grad-clip 0.5
--fp8-readout-fwd
```

Artifacts in `/home/robert/dev/glades-trainer`:

- 2k paired baseline vs active SIRA:
  `logs/fp8_sira_active_energy_balance_2000step_20260601_170226/`
- 5k active SIRA compared against the same-binary 5k baseline from
  `logs/fp8_sira_shadow_nll_baseline_compare_5000step_20260601_161016/baseline.log`:
  `logs/fp8_sira_active_energy_balance_5000step_20260601_174332/`

### 2k gate

The 2k active run completed with no OOM, FP8 fallback/rejection, NaN/Inf,
grad-skip, forward/backward, or stability failures.  VRAM was unchanged at
`14.55 / 15.56 GB`.

| mode | val NLL | bpb | ppl | bad lines |
|---|---:|---:|---:|---:|
| baseline | 4.9532 | 1.7865 | 141.62 | 0 |
| active SIRA E+B | 4.9478 | 1.7845 | 140.87 | 0 |

Delta active minus baseline: `NLL -0.0054`, `BPB -0.0020`, `PPL -0.75`.
The baseline run used `--log-every 100` and its last train log was step 1901,
so train-loss deltas from this artifact should not be used for attribution.

Active SIRA telemetry at step 2000:

```text
sira_loss=0.013030678
ratio=0.00283
raw=1.30307
terms(E/B/A)=1.15212/0.150947/0
term_grad_rms(q/p)=1.313e-11/4.817e-12
```

### 5k gate

The 5k active run completed cleanly and compared to the fresh same-binary 5k
baseline used for the SIRA-shadow attribution gate.

| mode | VRAM | step5000 loss | val NLL | warm tok/s | bad lines |
|---|---:|---:|---:|---:|---:|
| baseline | 14.55 / 15.56 GB | 3.8686 | 3.9619 | 28028.8 | 0 |
| active SIRA E+B | 14.55 / 15.56 GB | 3.8807 | 3.9590 | 27957.8 | 0 |

Delta active minus baseline:

```text
VRAM:            +0.00 GB
train loss:      +0.0121  (includes active SIRA loss)
val NLL:         -0.0029
warm tok/s mean: -71.0 tok/s (-0.25%)
wall time:       +7.0 s over 5k steps
```

Final validation position-bucket deltas were `[0/0/0/-0.01/0/0/0/0]` at log
precision.  Active SIRA diagnostics remained bounded:

```text
sira_loss max:      0.0153599344
sira_ratio max:     0.00377
step5000 loss:      0.0138957696
step5000 ratio:     0.00359
step5000 raw:       1.38958
terms(E/B/A):       1.23415/0.155427/0
term_grad_rms(q/p): 1.172e-11/3.924e-12
```

### 10k gate

The next staged gate was run as a matched 10k baseline plus active SIRA pair.
30k was not attempted in this pass because the matched 10k pair already consumed
multi-hour GPU time and the prior failure window is still closer to a 15.6k gate
than to default enablement.

Artifact:

```text
logs/fp8_sira_active_energy_balance_10000step_20260601_200056/
```

Both runs completed 10k with no OOM, FP8 fallback/rejection, NaN/Inf, grad-skip,
forward/backward, or stability failures.  VRAM stayed `14.55 / 15.56 GB`.

| mode | final val NLL | 5k val NLL | warm tok/s | bad lines |
|---|---:|---:|---:|---:|
| baseline | 3.7220 | 3.9590 | 28024.4 | 0 |
| active SIRA E+B | 3.7163 | 3.9541 | 27932.8 | 0 |

Delta active minus baseline:

```text
VRAM:            +0.00 GB
5k val NLL:      -0.0049
10k val NLL:     -0.0057
warm tok/s mean: -91.6 tok/s (-0.33%)
wall time:       +18.9 s over 10k steps
```

Final validation position-bucket deltas were
`[-0.01/-0.01/0/-0.01/-0.01/-0.01/0/-0.01]` at log precision.  The baseline's
last train log was step 9901 while active SIRA's debug cadence logged step
10000, so direct final train-loss deltas from this artifact are not used for
quality attribution.

Active SIRA diagnostics remained bounded:

```text
sira_loss max:       0.0170832649
sira_ratio max:      0.00452
step10000 sira_loss: 0.0131118475
step10000 ratio:     0.00380
step10000 raw:       1.31118
terms(E/B/A):        1.16425/0.146939/0
term_grad_rms(q/p):  1.173e-11/4.672e-12
loss-detail step10000: ce=3.43945336 zloss=0.0131696695 sira=0.0131118475 total=3.46573496
```

Regression flags: none.

### 15.6k instability-window gate

The next staged gate directly covered the prior seed-4242 instability window
(`~15.0k–15.5k`) with a matched baseline plus active SIRA pair.  30k was not run
in this pass because the 15.6k gate is the more targeted risk check before
another full 30k attempt.

Artifact:

```text
logs/fp8_sira_active_energy_balance_15600step_20260602_062926/
```

Both runs completed 15.6k with no OOM, FP8 fallback/rejection, NaN/Inf,
grad-skip, forward/backward, or stability failures.  VRAM stayed
`14.55 / 15.56 GB`.

| mode | final val NLL | 10.4k val NLL | 5.2k val NLL | warm tok/s | bad lines |
|---|---:|---:|---:|---:|---:|
| baseline | 3.6641 | 3.7234 | 3.9308 | 28017.3 | 0 |
| active SIRA E+B | 3.6615 | 3.7202 | 3.9277 | 27935.2 | 0 |

Delta active minus baseline:

```text
VRAM:              +0.00 GB
5.2k val NLL:      -0.0031
10.4k val NLL:     -0.0032
15.6k val NLL:     -0.0026
warm tok/s mean:   -82.2 tok/s (-0.29%)
wall time:         +27.5 s over 15.6k steps
```

Final validation position-bucket deltas were `[0/0/0/-0.01/-0.02/0/0/0]` at
log precision.  The baseline's last train log was step 15501 while active SIRA's
debug cadence logged step 15600, so final train-loss deltas are not used for
quality attribution.

Active SIRA diagnostics in the prior failure window remained bounded:

```text
sira_loss max:       0.0170937255
sira_ratio max:      0.00457
step15600 sira_loss: 0.0142372129
step15600 ratio:     0.00390
step15600 raw:       1.42372
terms(E/B/A):        1.26354/0.160182/0
term_grad_rms(q/p):  1.200e-11/3.646e-12
loss-detail step15600: ce=3.642277 zloss=0.0124758212 sira=0.0142372129 total=3.6689899
```

Regression flags: none.

### 30k seed-4242 candidate-default decision run

Because SIRA is now under serious consideration for candidate default status,
the most important single expensive run was the former failing seed `4242` at
30k with the stabilized LR/clip/FP8 recipe.  This was run active-only and
compared against historical 30k baselines and the earlier successful SIRA seeds;
a fresh same-recipe 30k baseline remains a follow-up if exact attribution is
needed.

Artifact:

```text
logs/fp8_sira_active_energy_balance_30000step_seed4242_20260602_121325/
```

Run recipe:

```text
--lr 7.5e-5 --grad-clip 0.5 --seed 4242 --fp8-readout-fwd
--sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25
--sira-action-weight 0.0 --sira-warmup 1000
```

The run completed 30k with no OOM, FP8 fallback/rejection, NaN/Inf, grad-skip,
forward/backward, or stability failures.  VRAM stayed `14.55 / 15.56 GB`.

| run | final NLL | bpb | ppl | acc1 | acc5 | acc10 | warm tok/s | bad lines |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active SIRA E+B seed4242 LR `7.5e-5` | 3.5235 | 1.2708 | 33.90 | 0.1408 | 0.5610 | 0.8442 | 27856.4 | 0 |

Final position buckets:

```text
[3.41/3.48/3.61/3.46/3.46/3.63/3.57/3.56]
```

Historical comparisons, noting they are not all same-binary/same-LR matched:

| Reference | final NLL | delta from new seed4242 SIRA |
|---|---:|---:|
| B5 30k baseline record | 3.5734 | -0.0499 |
| prior seed4242 baseline 30k LR `1e-4` | 3.5929 | -0.0694 |
| previous successful SIRA 30k mean | 3.5491 | -0.0256 |
| previous successful SIRA 30k best | 3.5369 | -0.0134 |
| previous successful SIRA 30k worst | 3.5591 | -0.0356 |

Active SIRA diagnostics remained bounded globally and through the former failure
window:

```text
sira_loss max:        0.0147738708
sira_ratio max:       0.00404
failure-window ratio: max 0.00369 over steps 14500..16000
step30000 sira_loss:  0.0140414126
step30000 ratio:      0.00390
step30000 raw:        1.40414
terms(E/B/A):         1.24409/0.160052/0
term_grad_rms(q/p):   1.257e-11/3.828e-12
loss-detail step30000: ce=3.58824182 zloss=0.0117069958 sira=0.0140414126 total=3.61399031
```

Regression flags: none.

Interpretation: the LR-stabilized FP8 2k/5k/10k/15.6k/30k active SIRA gates are
clean for seed `4242`, including the previous instability window and a full 30k
completion.  The seed4242 30k result is stronger than the previous successful
SIRA-seed mean and best historical run.  This justified moving SIRA from
“experimental hold” to a candidate-default investigation track, but not flipping
the shipped default yet.  A second same-recipe 30k validation seed was required
before promotion.

### Second 30k validation seed: seed 2024

A second same-recipe 30k active SIRA E+B run was performed with seed `2024`.
This is the most important follow-up for candidate-default promotion because it
checks whether the seed4242 recovery generalizes beyond one seed.

Artifact:

```text
logs/fp8_sira_active_energy_balance_30000step_seed2024_20260602_204556/
```

Run recipe matched the seed4242 candidate-default run except for `--seed 2024`:

```text
--lr 7.5e-5 --grad-clip 0.5 --seed 2024 --fp8-readout-fwd
--sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25
--sira-action-weight 0.0 --sira-warmup 1000
```

The run reached 30k and produced a finite final validation, but it did **not**
clear the candidate-default stability gate.  The bad-gradient guard started
skipping updates at step `24127` and skipped `5809` Adam updates through step
`30000`.  Early/instability-window SIRA diagnostics were normal, but late global
gradients repeatedly overflowed or became non-finite.

| run | final NLL | bpb | ppl | acc1 | acc5 | acc10 | warm tok/s | bad/skip lines |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active SIRA E+B seed2024 LR `7.5e-5` | 3.5675 | 1.2867 | 35.43 | 0.1385 | 0.5506 | 0.8361 | 27862.3 | 98040 |

Final position buckets:

```text
[3.42/3.48/3.61/3.55/3.53/3.73/3.61/3.61]
```

Seed-to-seed comparison against the clean seed4242 30k run:

| metric | seed4242 | seed2024 | seed2024 − seed4242 |
|---|---:|---:|---:|
| final NLL | 3.5235 | 3.5675 | +0.0440 |
| bpb | 1.2708 | 1.2867 | +0.0159 |
| ppl | 33.90 | 35.43 | +1.53 |
| acc1 | 0.1408 | 0.1385 | -0.0023 |
| acc5 | 0.5610 | 0.5506 | -0.0104 |
| acc10 | 0.8442 | 0.8361 | -0.0081 |
| warm tok/s | 27856.4 | 27862.3 | +5.9 |
| max SIRA loss | 0.0147739 | 0.0148582 | +0.0000844 |
| max SIRA ratio | 0.00404 | 0.00403 | -0.00001 |

Position-bucket delta seed2024 minus seed4242:

```text
[+0.01/+0.00/+0.00/+0.09/+0.07/+0.10/+0.04/+0.05]
```

Late bad-gradient details:

```text
first grad-skip: step 24127, global_sumsq=1.06774971e+21, norm=3.26764401e+10, bad group=dE
skip count:      5809 updates skipped from 24127 through 30000
max bad_groups:  30
last step:       global_sumsq=inf, norm=inf, scale=0, bad groups include dE and early-layer dgamma/dbeta
```

SIRA scalar diagnostics did not identify the late instability as an SIRA-scalar
blow-up:

```text
sira_loss max:       0.0148582356
sira_ratio max:      0.00403
failure-window ratio max over 14500..16000: 0.00371
step30000 sira_loss: 0.0139291044
step30000 ratio:     0.00384
terms(E/B/A):        1.23569/0.157224/0
term_grad_rms(q/p):  1.211e-11/3.909e-12
```

Historical comparisons, noting that this seed2024 run has many skipped updates
and is not an acceptable clean candidate-default pass:

| Reference | final NLL | delta from seed2024 SIRA |
|---|---:|---:|
| B5 30k baseline record | 3.5734 | -0.0059 |
| prior seed2024 SIRA LR `1e-4` | 3.5514 | +0.0161 |
| previous successful SIRA 30k mean | 3.5491 | +0.0184 |
| previous successful SIRA 30k best | 3.5369 | +0.0306 |
| previous successful SIRA 30k worst | 3.5591 | +0.0084 |

Regression flags:

```text
- bad/stability/fallback lines present
- VRAM >14.70GB (14.71 / 15.56 GB observed; likely external-memory sensitive but over the current alert threshold)
```

Interpretation: seed2024 blocks SIRA default promotion.  The recipe still has a
quality signal and the explicit guard prevented weight corruption, but default
promotion requires clean multi-seed 30k behavior without thousands of late
skipped Adam updates.  SIRA should remain default-off while the late dE/early
layer gradient-overflow path is investigated or a lower-risk candidate recipe is
defined.

---

## Late dE / early-layer gradient path investigation (2026-06-03)

Additional default-off trainer diagnostics were added for the investigation:

- `--sira-de-trace` now records `dq` around readout, SIRA terminal backward,
  and embedding scatter in addition to `dE`.
- `--sira-layer-grad-trace` records probed-layer `dq`/`dp` norms at
  `after-reln`, `after-fuse`, and `after-layer` points inside the debug window.
- These traces are opt-in, disable CUDA graph capture when requested, and remain
  no-ops in default/disabled runs.

Artifacts in `/home/robert/dev/glades-trainer`:

- `logs/sira_de_path_seed2024_trace_24260_20260603_062225/`
- `logs/sira_layer_path_seed2024_trace_24110_20260603_103735/`
- `logs/sira_intralayer_path_seed2024_trace_24110_20260603_143746/`
- Summary: `logs/sira_gradient_path_investigation_summary_20260603.json`

Key finding: the huge `dE` is not produced by the tied-readout GEMM itself.
In the 24260-step reproduction, the first skip moved to step `24101` and the
readout/pre-scatter `dE` norms stayed normal while post-scatter `dE` exploded:

```text
step 24101: dE after-readout norm=0.936, before-scatter=0.936, after-scatter=3.42e10
step 24102: dE after-readout norm=0.877, before-scatter=0.877, after-scatter=6.42e13
step 24108: dE after-readout norm=2.84,  before-scatter=2.84,  after-scatter=inf
```

Therefore the `dE` guard trips because embedding scatter accumulates an already
huge `s.dq` into the tied embedding rows.  The scatter kernel is an atomic add
of `dq_0[t, :]`; it is not the first creator of the large values.

The layerwise trace reproduced the same qualitative path with a large finite
spike and localized it to the early q-side reverse pass.  At step `24108`:

```text
dq after readout/SIRA:       ~6.7e-3
dq before embedding scatter: 7.53e9
dE after embedding scatter:  7.53e9

dq after-layer-06: 3.94e-2
dq after-layer-05: 7.01e-1
dq after-layer-04: 1.85e1
dq after-layer-03: 6.83e2
dq after-layer-02: 3.44e4
dq after-layer-01: 2.65e6
dq after-layer-00: 7.53e9
```

`dp` remained tiny in that trace (max layer `dp` around `3e-4`), so the observed
explosion is q-side and concentrated in the early layers.  This also explains
why `dE`, `L00.dgamma`, `L00.dbeta`, then early-layer `dgamma/dbeta` dominate
`grad-detail`: the same exploding `dq` enters the layer-0 ReLN backward and then
embedding scatter.

A more intrusive intra-layer trace did **not** reproduce the spike in the same
window (max pre-embed `dq` about `1.28`, no skips).  That result should not be
read as a fix; long CHIRON runs are run/build/synchronization sensitive.  It does
show that the failure is not a deterministic SIRA scalar blow-up.  Around the bad
window, SIRA terminal scalars and terminal-gradient RMS stayed bounded while the
reverse-path `dq` sometimes entered a high-gain early-layer regime.

A follow-up q-branch component trace added `--sira-qbranch-trace` to log
ReLN gain bounds and SCFA q-branch component norms (`scfa.entry`,
`dy_compr`, `dq_compr_attn`, `dq_perp`, `dq_compr_total`,
`after_B_dqcompr`, and `scfa.exit`).  The targeted reproduction artifact is:

- `logs/sira_qbranch_path_seed2024_trace_24110_20260603_195159/`
- Summary: `qbranch_summary.json` inside that directory.

That q-branch run did **not** reproduce the 7.5e9 spike over steps
`24100..24110` and produced no guard skips.  It still isolated the normal
clean-path early q-gain anatomy:

```text
L06..L02 ReLN amp: ~1.0..1.16
L01 ReLN amp:      ~1.23..1.32
L00 ReLN amp:      ~41.6..47.9
L00 sigma_min:     ~0.0178
L00 invstd_max:    ~56.0
L00 gamma_max:     ~1.23
L00 bound_max:     ~68.8
SCFA q branch:     entry/after_B/exit norms nearly unchanged
SCFA dp entry:     ~2e-4..3.6e-4
```

Thus, in a clean trajectory, the dominant early q-side amplification is the
layer-0 ReLN Jacobian from low embedding-state variance, not the SCFA q-branch.
The prior 7.5e9 runaway would require an abnormal early ReLN gain/stat state
(or an already-corrupted incoming `dq`) that this trace did not reproduce.  If
it recurs, `--sira-qbranch-trace --sira-probe-layers 0,1,2,3,4,5,6,23` should
show whether the first abnormal multiplier appears in ReLN stats/bounds or in a
specific SCFA component.

### L00 ReLN row/q-state consistency trace (2026-06-05)

The L00 row trace was extended under the same default-off
`--sira-qbranch-trace` flag to download the reconstructed L00 ReLN input row
and compare it with the saved forward ReLN `(mu, sigma)` stats.  It logs
`mu`, reconstructed `qmean`, `mu_delta`, `qsigma/sigma`, `xhat_rms`, and
`xhat_max` for the top row-amplification rows.

Artifacts:

- Clean smoke: `logs/sira_qbranch_qstate_smoke2_8Rhnac/smoke.log`
- Clean late window: `logs/sira_qbranch_qstate_seed2024_trace_24110_20260604_113713/`
- Bad long window: `logs/sira_qbranch_qstate_long_seed2024_trace_24150_20260604_153514/`
- Clean first-window rerun: `logs/sira_qbranch_qstate_firstbad_seed2024_trace_24060_20260604_193636/`

Clean trajectories show the diagnostic behaving as expected: `qsig_ratio` is
about `1.0`, `mu_delta` is tiny, and L00 rowamp stays around the ordinary
`gamma/sigma` scale (`~40..65x`).  For example, the 24045..24060 first-window
rerun had no guard skips, `qsig_ratio_p50 ~= 1.00001`, `qsig_ratio_max <= 1.45`,
and `rowamp_max <= 63.6`.

The bad long-window reproduction showed a qualitatively different state before
embedding scatter: many guard skips occurred and the reconstructed L00 ReLN
input no longer matched the saved forward stats.  Over steps `24090..24150`,
`rowamp_max` reached `3.77e5`, while `sigma_min` remained only about `0.018`.
The corresponding top rows had `qsig_ratio` and `xhat_rms` tens to hundreds of
times larger than the saved-normalized scale; e.g. step `24090` top token `265`
had `sigma=0.01813`, `qsigma=0.50597`, `qsig_ratio=27.9`, `xhat_rms=27.9`,
`xhat_max=1262`, and `rowamp=3.51e4`.  Step `24104` reached
`qsig_ratio=53.4` and `rowamp=1.29e5`.

A follow-up all-layer sampled q-state trace was added under the same flag and
run on seed2024 with all layers probed:

- Failing-window artifact:
  `logs/sira_qbranch_layer_qstate_seed2024_trace_24080_24110_20260605_070728/`
- Full clean smoke:
  `logs/sira_qbranch_layer_qstate_full_smoke_LOEsbg/full_smoke.log`
- Prebreak rerun that did not reproduce a skip:
  `logs/sira_qbranch_layer_qstate_seed2024_prebreak_24070_24080_20260605_111040/`

The full clean smoke had `qsig_ratio_max ~= 1` and `xhat_rms_max ~= 1` for
L23..L01, and L00 exact `qsig_ratio_max ~= 1`, `xhat_rms_max ~= 0.47`,
`rowamp_max ~= 90.5`.  In the failing-window run, the first captured debug step
was also the first guard skip (`24080`).  At that step, L23..L21 remained
consistent, L20 showed only a small mean-delta drift, and the first severe
`qsig/xhat` consistency break in reverse traversal was L06:

```text
step 24080 L06: qsig_ratio_max=2.03, xhat_rms_max=2.03, mu_delta_max=0.043
step 24080 L05: qsig_ratio_max=2.51, xhat_rms_max=2.51
step 24080 L04: qsig_ratio_max=3.10, xhat_rms_max=3.10
step 24080 L03: qsig_ratio_max=3.82, xhat_rms_max=3.82
step 24080 L02: qsig_ratio_max=4.70, xhat_rms_max=4.70
step 24080 L01: qsig_ratio_max=5.83, xhat_rms_max=5.83
step 24080 L00 sampled: qsig_ratio_max=5.99, xhat_rms_max=5.96
step 24080 L00 exact:   qsig_ratio_max=21.19, xhat_rms_max=21.19, rowamp_max=1.97e4
```

Using a looser `qsig/xhat > 1.1` drift threshold, the first reverse-layer drift
at the captured bad step starts around L13 and grows monotonically toward the
early layers; using `> 1.5`, it starts at L07; using the severe `> 2.0`
threshold, it starts at L06.  The corresponding ReLN gradient trace at step
24080 first shows abnormal amplification around L08/L07 and then explodes
through L06..L00.  A separate prebreak run with debug active from
`24070..24080` did not reproduce any guard skip or severe all-layer q-state
break, again confirming run/timing sensitivity.

A focused L00-only prebreak trace reduced the diagnostic perturbation and
captured exact full-row L00 stats over `24065..24085`:

- `logs/sira_l00_qstate_prebreak_seed2024_trace_24065_24085_20260605_155452/`

This run did not produce guard skips, but L00 q-state mismatch was already
visible well before step 24080 when compared to the full clean smoke
(`qsig_ratio_max ~= 1`, `xhat_rms_max ~= 0.47`, `rowamp_max ~= 90.5`):

```text
step 24065: qsig_ratio_max=1.13, xhat_rms_max=1.04, rowamp_max=60.4
step 24066: qsig_ratio_max=2.34, xhat_rms_max=2.27, rowamp_max=79.3  (top token 265)
step 24068: qsig_ratio_max=2.85, xhat_rms_max=2.80, rowamp_max=165.7 (top token 265)
step 24069: qsig_ratio_max=4.04, xhat_rms_max=4.00, rowamp_max=686.4 (top token 265)
step 24070: qsig_ratio_max=4.99, xhat_rms_max=4.96, rowamp_max=1080.0 (top token 265)
```

Thus the first L00 divergence in the focused prebreak trace is a mild
`qsig/xhat` deviation at step `24065`; the first severe L00 q-state mismatch is
step `24066`; the first large rowamp response is step `24069`; and rowamp first
exceeds `1000x` at step `24070`.  `sigma_min` stayed near `0.01778`, so this is
again a reconstructed-q/stat mismatch rather than an unusually small-sigma
state.  Token `265` dominates the earliest severe rows.

A token-focused L00 trace hook (`--sira-qbranch-trace-token 265`) was added to
compare the exact token-265 rows against a clean-ish all-layer trace.  Artifacts:

- Failing/focused token trace:
  `logs/sira_l00_token265_focused_seed2024_trace_24065_24070_20260606_080007/`
- Clean-ish all-layer token trace:
  `logs/sira_l00_token265_cleanish_alllayer_seed2024_trace_24065_24070_20260606_115602/`
- Comparison summary:
  `logs/sira_token265_compare_24065_24070_summary.md`

The token trace logs the L00 inverse input normalized as
`yhat=(q_out-beta)/gamma`, saved stats, `gamma/beta`, reconstructed `q_in`,
upstream `dqin`, and rowamp.  In the clean-ish trace, token-265 `gamma/beta` and
stats were stable, `yhat_rms` matched reconstructed `xhat_rms`, `dqin` stayed
around `3e-5..9e-5`, and rowamp stayed around `58..60` until a mild step-24070
outlier (`rowamp=239`).  In the failing/focused token trace, `gamma/beta` were
still effectively unchanged (`gamma_rms` ratio focused/clean `~1.00019`,
`beta_rms` ratio `~0.9992`), so parameters/stats are not the first culprit.

Within the requested `24065..24070` window:

```text
step 24065 token 265: yhat_rms_max 1.03 vs clean 0.836; dqin normal; rowamp normal
step 24066 token 265: dqin_max 1.8e19 vs clean 5.4e-5; rowamp 8.6e4 vs clean 59
step 24066 token 265: qsig/xhat 43 vs clean 1.6, while gamma/beta remain unchanged
```

So the first token-265 input drift visible before the q-state mismatch is the
L00 ReLN inverse input `q_out`/`yhat` at step `24065`, but the first catastrophic
value feeding the rowamp explosion is the upstream `dqin` tensor at step
`24066`.  The failing token-focused run had already begun guard skips before
the debug window, so this is evidence for the failing trajectory rather than a
clean no-skip trajectory.

A synchronized rerun of the focused token trace confirmed the `q_out` capture is
not an async download artifact:

- Focused synced trace:
  `logs/sira_l00_token265_sync_focused_seed2024_trace_24065_24070_20260606_170448/`
- Updated comparison:
  `logs/sira_token265_sync_focus_vs_cleanish_24065_24070_summary.md`

In the synced focused trace, `yhat_rms` and reconstructed `xhat_rms` agree to
roundoff.  At step `24065`, the top token-265 position is the same as the
clean-ish comparator (`pos=795`) and the token count is the same (`3/3`), so the
data/RoPE-position side is not the first divergence.  L00 `gamma_rms` and
`beta_rms` are essentially unchanged (`1.009` and `0.0381` in both traces), and
upstream `dqin` remains normal (`3.83e-5` vs `3.41e-5`).  The first in-window
value that has actually moved is the reconstructed L00 activation itself:
`yhat_rms_max=2.210` vs `0.836`, `qsig_ratio_max=2.276` vs `1.005`, with normal
rowamp (`58.8` vs `58.3`).  This rules out L00 gamma/beta, token identity,
position identity, and L00 ReLN download/normalization as the first cause seen
inside the window.  The focused synced run had already skipped `20` Adam updates
before the window (first at step `23939`), whereas the clean-ish comparator had
none; therefore the earliest upstream event visible in the available traces is
the prior overflow/guard-skip trajectory split, not a new L00-local weight or
RoPE/position difference at step `24065`.

Interpretation: rowamp far above the ordinary `gamma/sigma` bound is explained
by the ReLN backward using a reconstructed `q_in` whose normalized residuals are
no longer close to the saved forward normalization.  Once `xhat` is huge, the
LayerNorm/ReLN backward projection term can amplify far beyond the clean
`gamma/sigma` intuition.  The open question is why the reverse reconstruction
enters this mismatched state in bad trajectories (upper-layer reverse-state
corruption, aliasing/state overwrite, or a run-sensitive numerical path), not
whether low `sigma` alone explains the spikes.

### Cost-control decision: no more blind SIRA reruns (2026-06-07)

Because full seed2024 24k+ runs are expensive and another clean run would not
unblock default promotion, the next diagnostic step is a one-shot trigger rather
than another broad debug window.  Trainer commit `53f88a5` adds default-off flags:

```text
--sira-grad-trigger-dump
--sira-grad-trigger-stop
--sira-grad-trigger-sumsq F
--sira-qbranch-trace-token 265   # optional token focus for the L00 snapshot
```

When enabled, the trainer waits until the global gradient sumsq crosses the
threshold (default `1e20`, the bad-gradient guard scale), logs the normal
per-group gradient breakdown, downloads a single post-backward L00 q-state
snapshot (`s.q` versus saved L00 ReLN stats), optionally reports token-265 rows,
and can stop before Adam.  This has zero default overhead and avoids paying for
per-step q-branch host downloads before the first bad event.  It does not solve
the root cause; it makes the next expensive run useful by capturing the first
trajectory split instead of a later corrupted window.

Recommended next run, only if we decide the remaining uncertainty justifies the
cost:

```bash
--sira-grad-trigger-dump \
--sira-grad-trigger-stop \
--sira-grad-trigger-sumsq 1e20 \
--sira-qbranch-trace-token 265
```

Verification for the trigger hook:

- `git diff --check`
- `cmake --build build --target glades_chiron_train -j2`
- Smoke with forced threshold `0`:
  `logs/sira_grad_trigger_smoke_EaCgyx/smoke.log`

The first trigger run completed and stopped at the first bad global gradient:

- Artifact: `logs/sira_grad_trigger_seed2024_20260606_214654/`
- Summary: `trigger_summary.md` and `trigger_summary.json`
- Runtime: `rc=0`, warm logged throughput excluding step 1: `27954.0 tok/s`
- Trigger: step `24070`, `global_sumsq=2.49679264e25`, norm `4.9967918e12`
- Top gradient groups: `dE` (`2.47766636e25`, BAD) and `L00.dgamma`
  (`1.91250114e23`, BAD), then the early-layer `dgamma/dbeta` ladder.
- One-shot L00 q-state snapshot at the trigger showed q/stat mismatch already
  present at the first bad global grad: `qsig_ratio_max=13.03`,
  `xhat_rms_max=13.02`, and `mean_abs_delta_max=0.00608`.
- Token `265` was again a major outlier but not the top row: top row was
  `token=50 pos=9` (`qsig=13.03`); token `265` count was `209`, with top rows
  at `pos=1761/2978/1317` around `qsig=9.57`, `xhat_rms=9.55`.

This is now a clean first-trigger capture rather than a later window after many
prior skips.  It confirms that by the first bad global gradient the L00
reconstructed state is already inconsistent with the saved L00 ReLN stats, and
that the immediate gradient signature remains `dE` plus early ReLN parameter
grads.  It still does not identify the first upstream forward/reverse tensor
that caused the L00 q-state mismatch; a future run should only be considered if
it adds a similarly triggered all-layer or pre-L00 forward-state snapshot.

Working diagnosis: SIRA is at most an indirect trajectory nudge.  The immediate
overflow path is:

```text
normal readout dq  → early-layer q-side reverse amplification
                   → huge dq_0
                   → embedding_scatter_add writes huge dE
                   → dE + L00/L01 dgamma/dbeta dominate grad-detail
                   → guard skips Adam on overflow/non-finite global norm
```

This keeps the candidate recipe in investigation/default-off status.  Next
mitigation candidates should target the early q-side gain path or reduce the
chance of entering it (for example lower LR, smaller SIRA coefficient, an
explicit q-side gradient clamp before embedding scatter, or ReLN/early-layer
stability diagnostics), then re-run 30k multi-seed gates before any promotion.

---

## Flagship recommendation

Keep shipped flagship defaults unchanged for now:

- `siraCoef = 0.0`
- SIRA remains opt-in in production defaults.
- Disabled mode must remain parity/bit-identical.

The following recipe is the current candidate-default investigation recipe, but
seed2024's late guard skips block default promotion:

```text
--sira-coef 1e-2
--sira-energy-weight 1.0
--sira-balance-weight 0.25
--sira-action-weight 0.0
--sira-warmup 1000
--lr 7.5e-5
--grad-clip 0.5
```

Do not flip SIRA on by default until all of the following are true:

1. A same-recipe multi-seed 30k gate includes seed `4242` or an equally
   adversarial seed without NaNs, non-finite gradients, or guard skips.
2. At least one same-recipe matched 30k baseline comparison confirms the NLL
   benefit is not solely the LR/clip recipe.
3. Final NLL clears either the single-run gate (`<= 3.5534`) or a
   pre-registered multi-seed criterion.
4. Throughput remains `>= 26668 tok/s`.
5. Position bucket 7 / long-context metrics do not regress.
6. SIRA flags remain default-off and disabled parity remains verified.

Short version: **SIRA has a serious quality signal, but seed2024's late
bad-gradient skip wave blocks candidate-default promotion until the overflow path
is understood or mitigated.**

---

## Verification evidence

- `cd /home/robert/dev/glades-trainer && bash build.sh` — PASS.
- `./build/glades_chiron_train --sira-config-smoke --sira-de-trace --sira-coef 1e-2 --sira-energy-weight 1 --sira-balance-weight 0.25 --sira-action-weight 0 --sira-warmup 1000` — PASS.
- Prior SIRA suite remained green before long-run evaluation:
  - `./unit-tests/build/glades-unit-tests chiron-sira` — PASS.
  - `./unit-tests/build/glades-unit-tests chiron` — PASS.
  - full unit suite — PASS.
  - `scripts/sira_config_smoke.sh` — PASS.
  - `scripts/sira_training_loss_smoke.sh` — PASS.

---

## q-side gradient clamp implemented (2026-06-11)

The first-listed mitigation candidate — "an explicit q-side gradient clamp
before embedding scatter" — is now implemented, default-off:

- **Library** (`glades-ml`): new `glades::gpu::row_rms_clamp(x, rows, cols,
  tauRms, d_clampedCount, d_nonfiniteCount)` in `gpu_kernels.cu`.  Per row of
  `dq_0`: non-finite row → zeroed (counted); row RMS > τ → rescaled by
  τ/rms (counted); healthy row → untouched (no write; **bit-identical**).
  Row sum-of-squares accumulates in double so huge-but-finite rows (whose
  FP32 square overflows to inf — the observed failure scale) still rescale
  correctly.  Deterministic (fixed-order tree reduction).
- **Trainer**: `--dq-embed-clamp F` (`cfg.dqEmbedClampTau`, 0 = off) applies
  the clamp to `s.dq` immediately before `embedding_scatter_add`, after the
  pre-embed dq/dE norm traces — so the traces still record the RAW explosion
  while dE receives the clamped rows.  Per-step counter download logs
  `[dq-clamp step N] clamped=X zeroed=Y tau=F` only on steps where the clamp
  fired.  `--dq-embed-clamp-smoke` verifies CLI propagation + kernel
  behavior.  Wired through `run.sh` (flagship/chiron/legacy arg paths).
- **Scope honesty:** this bounds the dE contribution only.  If L00/L01 ReLN
  dgamma/dbeta alone can still overflow the global norm, the guard will
  still skip — the seed-4242 re-run decides whether dE-side clamping
  suffices.  Per-layer dq clamping is the documented escalation.
- **Design doc:** glades-ml
  `docs/superpowers/specs/2026-06-11-dq-embed-clamp-design.md`.

Verification evidence (all PASS, 2026-06-11):

- glades-ml: `cmake --build build` + `cmake --build unit-tests/build`.
- `./unit-tests/build/glades-unit-tests chiron-qclamp` — new
  `CHIRONQClampMathTest` (healthy rows bit-identical, rescaled rows match
  CPU double reference at rtol 1e-6, non-finite rows zeroed, counters
  exact, huge-finite 1e20 row exercises the double accumulator) and
  `CHIRONQClampEdgeTest` (invalid args rejected incl. τ<=0, NULL counters,
  cols==1 sign preservation).  Also added to the `chiron-sira` selector
  group; `chiron` and `chiron-sira` suites remain 0-failure.
- trainer: `bash build.sh`; `./build/glades_chiron_train
  --dq-embed-clamp-smoke --dq-embed-clamp 2.5` → PASS (RC 0); missing τ →
  RC 2; `sh run.sh flagship --dq-embed-clamp-smoke --dq-embed-clamp 1.0`
  → PASS (flag forwarding verified end-to-end).
- `git diff --check` clean in both repos.

Disabled-path parity is structural: the call is gated on
`dqEmbedClampTau > 0` and the counter buffer is not even allocated when
off — no new kernel launches, no code on the hot path.

### Next step for the arc

Re-run the candidate recipe at seed 4242 with the clamp enabled.  Choose τ
from healthy-run dq pre-embed trace stats with generous headroom (the
exploded rows sit at overflow scale, so a loose τ catches them while
remaining identity on healthy steps), and document the τ choice in the run
log.  Note promotion criterion 2 (matched no-SIRA 30k baseline at
`--lr 7.5e-5 --grad-clip 0.5`) remains outstanding and is independent of
this mitigation.

---

## q-side clamp seed-2024 30k re-run: FAIL as standalone mitigation (2026-06-11)

First gate run of the dq-embed clamp (`--dq-embed-clamp 1.0`, τ derived from
healthy ‖dq₀‖ ≈ 0.8–3.0 in the seed-2024 layer traces; ≥15× above the
worst-case healthy single-row RMS).  Exact candidate-default recipe, seed
`2024`.  Note: seed 4242 was NOT re-run — it already passed clean at this
recipe on 2026-06-02; seed 2024 is the failing seed the clamp had to fix.

Artifact: `logs/sira_clamp_tau1_seed2024_30k_20260611_010146/` (glades-trainer).
Checkpoint: `database/checkpoints/sira_clamp_seed2024/chiron_sira_clamp_tau1_seed2024_20260611_010146.final`.

**Verdict: the clamp mechanism worked exactly as designed and the stability
gate still FAILS.**  The overflow has a second, independent path the clamp
cannot reach.

| metric | original seed2024 (2026-06-02) | clamp run (2026-06-11) |
|---|---:|---:|
| first guard skip | step 24127 | step 24337 |
| total skips | 5809 | 5659 |
| final val NLL | 3.5675 | 3.5618 |
| step-27k val NLL (mid-wave) | 3.6471 | 3.6238 |
| warm tok/s | 27862 | ~28080 |

Decisive evidence — grad-detail at the clamp run's first skip (step 24337):

```text
#01 L00.dgamma  sumsq=2.34e25  BAD   (99.99% of global overflow)
#02 L01.dgamma  sumsq=5.73e20  BAD
#11 dE          sumsq=1.49e5  norm=386   (clean — 15 orders below guard)
```

In the original run, dE was the sole bad group at onset.  With dE fully
bounded by the clamp (first firing step 24321, 1–9 rows/step pre-wave —
surgical, zero rows zeroed pre-wave), **L00/L01 ReLN dgamma/dbeta overflow
independently**: the early-layer q-side reverse amplification produces huge
intermediate dq *inside* the layer backward, and `reln_backward` converts it
to huge dgamma/dbeta before the terminal dq ever reaches the embedding
scatter.  The clamp seals the terminal endpoint only.

Wave interior (weights frozen during skips, so the amplification re-fires
each step): clamp escalated from ~1.2k to ~6.9k rows/step clamped with up to
585 non-finite rows/step zeroed; totals 20.1M rows clamped, 1.14M zeroed
across 5674 firing steps.  The wave did not self-terminate in either run.

Trajectory notes: pre-wave vals tracked the original within ±0.0023 nat
(3k/6k/9k/12k/15k/18k/21k/24k), confirming the clamp is identity on healthy
steps at production shape.  Small drift vs 2026-06-02 is attributed to the
2026-06-10 BF16G replay-parity library change (d9b8e3249).  The clamp run's
better mid-wave/final NLL (−0.023 / −0.0057) comes from ~210 extra clean
steps before its later wave onset — incidental, not gate-relevant.

### Disposition

- `--dq-embed-clamp` stays default-off; it is correct, cheap, and surgical,
  but **insufficient alone**.  Keep it in the recipe for future stability
  runs (it removes dE from the failure surface and its `[dq-clamp]` log is a
  free early-warning signal — it fired 16 steps before the first skip).
- The mitigation that matches the evidence is **bounding the per-layer
  backward amplification itself**: clamp dq at each layer boundary (e.g.
  `row_rms_clamp` on `s.dq_buf` after each `chiron_attention_shear_backward`,
  before `reln_backward`), which bounds both the cascade and the dgamma/dbeta
  it produces.  Library kernel already exists; trainer needs a per-layer flag
  + call sites (~24 extra T×m reads/step ≈ 1% wall when enabled).
- Alternative: per-group selective skip/clip at the Adam boundary (grad-detail
  already computes per-group norms) — applies the finite groups and skips only
  the bad ones.  Broader safety net but masks rather than bounds.
- Promotion criterion 2 (matched no-SIRA 30k baseline at `--lr 7.5e-5
  --grad-clip 0.5`) remains outstanding and independent — still the
  highest-EV next 5h of GPU for the arc.

---

## Per-layer dq clamp seed-2024 30k: STABILITY GATE PASS (2026-06-11)

Escalation run after the dE-only clamp FAIL above.  New trainer flag
`--dq-layer-clamp F` clamps the incoming dq (`dq_in_ptr`,
iter-113-alternation-aware) at the top of EVERY backward layer iteration —
the tensor `chiron_reln_backward` consumes — bounding each layer's
dgamma/dbeta at the source.  Run = exact candidate recipe, seed `2024`,
`--dq-layer-clamp 1.0 --dq-embed-clamp 1.0`.

Artifact: `logs/sira_layerclamp_tau1_seed2024_30k_20260611_081639/` (glades-trainer).
Checkpoint: `database/checkpoints/sira_clamp_seed2024/chiron_sira_layerclamp_tau1_seed2024_20260611_081639.final`.

**Result: zero guard skips in 30k.  Best SIRA NLL in the arc.**

| run (seed 2024 unless noted) | final NLL | skips | warm tok/s |
|---|---:|---:|---:|
| B5 flagship baseline record | 3.5734 | 0 | ~28,070 |
| original (2026-06-02) | 3.5675 | 5,809 | 27,862 |
| dE-only clamp (2026-06-11) | 3.5618 | 5,659 | ~28,080 |
| clean seed-4242 (2026-06-02) | 3.5235 | 0 | 27,856 |
| **per-layer clamp (this run)** | **3.5062** | **0** | 27,184 |

Final position buckets `[3.38/3.42/3.57/3.46/3.46/3.65/3.53/3.57]` —
every bucket better than B5 `[3.45/3.47/3.65/3.53/3.52/3.69/3.62/3.65]`,
including bucket 7 (3.57 vs 3.65).

Clamp activity: confined to steps `23766–24276` (~510 steps), 443 layer-
clamp firing steps (434,304 rows total, peak 127,013 rows/step ≈ 32% of
the 24×T row-slots, **0 zeroed** — bounding the cascade prevented any
non-finite value from ever forming) + 467 embed-clamp firing steps
(175,370 rows).  **No firings after 24,276**: with updates continuing
(bounded + globally clipped) instead of 5,700 frozen-weight skip steps,
the optimizer trained through the burst region and exited it.  This
confirms the skip wave's self-perpetuation mechanism (frozen weights →
identical amplification next step) and shows bounded-update continuation
breaks the loop.

Identity evidence (300-step gate, `logs/dq_layer_clamp_gate_20260611/`):
zero firings in the healthy phase; no-clamp vs both-clamps trajectory
drift (13/30 last-digit step lines) is within same-seed rerun noise
(12/30 for a no-clamp rerun); all three gates end at identical val NLL.
Side-finding: same-seed full-recipe runs are NOT bit-reproducible
run-to-run at production shape (atomic-ordering noise in
embedding_scatter_add and friends) — "trajectory parity" claims at this
shape should always be benchmarked against a rerun control.

### Promotion scorecard (per the 2026-05-27 criteria)

1. **PASS** — adversarial-seed 30k without NaNs/non-finite/guard skips
   (seed 2024 was the blocking seed; 4242 passed clean 2026-06-02).
2. **OUTSTANDING** — matched no-SIRA 30k baseline at `--lr 7.5e-5
   --grad-clip 0.5` (now ideally + clamps) for recipe attribution.
3. **PASS** — final NLL 3.5062 ≤ 3.5534 single-run gate.
4. **PASS** — 27,184 tok/s ≥ 26,668 (clamp overhead ~2.4% vs no-clamp).
5. **PASS** — bucket 7 / long-context improved, no regression.
6. **PASS** — flags default-off; disabled parity verified by gate A/C.

**SIRA candidate-default promotion is now blocked ONLY on criterion 2.**

Caveats for the eventual promotion decision: (a) the clamp intervened in
~510 steps of the trajectory, so part of the 3.5062-vs-3.5235 margin over
the clean 4242 run may be seed/noise rather than clamp benefit; (b) all
2026-06-11 runs are on the post-BF16G-replay-parity binary (d9b8e3249),
which shifts trajectories at the ±0.002-nat scale vs 2026-06-02 runs.
The criterion-2 baseline resolves both at once if run on the current
binary with clamps in the recipe.

---

## Criterion-2 matched no-SIRA baseline: SIRA BENEFIT CONFIRMED (2026-06-11)

The last outstanding promotion criterion.  Exact twin of the per-layer-clamp
PASS run (seed 2024, `--lr 7.5e-5 --grad-clip 0.5`, zloss+qk-norm,
`--dq-layer-clamp 1.0 --dq-embed-clamp 1.0`, same binary) with **SIRA
omitted**.

Artifact: `logs/nosira_baseline_lr7p5e5_clamps_seed2024_30k_20260611_133201/`
(glades-trainer).  Checkpoint:
`database/checkpoints/nosira_baseline_seed2024/chiron_nosira_baseline_lr7p5e5_clamps_seed2024_20260611_133201.final`.

| run (seed 2024, current binary, clamps on) | final NLL | skips | warm tok/s |
|---|---:|---:|---:|
| no-SIRA matched baseline | 3.5373 | 0 | 27,326 |
| SIRA E+B candidate | **3.5062** | 0 | 27,184 |

**SIRA's attributed contribution: −0.0311 nat** at an otherwise identical
recipe — criterion 2 PASSES (the benefit is not solely the LR/clip change).
Decomposition of the total −0.0672 vs the B5 flagship record (3.5734):
≈ −0.036 from the LR 7.5e-5/clip 0.5 recipe, ≈ −0.031 from SIRA.
The baseline trailed the SIRA run at 7 of 10 val checkpoints; the gap was
concentrated in the late phase, consistent with the earlier seed-pair
evidence.

Instability attribution bonus: the burst fired in the no-SIRA run too —
window `25898–25956` (~58 steps, 40,317 rows clamped, 0 non-finite,
0 skips) vs SIRA's `23766–24276` (~510 steps).  **The q-side burst is
recipe/data-intrinsic at seed 2024; SIRA amplifies and extends it but does
not cause it.**  The dq clamps contained it in both regimes, and the
21k val bump + bucket-3 spike also reproduced without SIRA (data-driven).

### Promotion status after this run

All six pre-registered criteria now have a PASS at the candidate recipe
**as amended with the clamps** (`--sira-coef 1e-2 --sira-energy-weight 1.0
--sira-balance-weight 0.25 --sira-action-weight 0.0 --sira-warmup 1000
--lr 7.5e-5 --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0`):

1. PASS — seed 2024 (the adversarial seed) clean 30k; seed 4242 clean
   2026-06-02 (older binary, no clamps).
2. PASS — this run: SIRA −0.0311 nat vs matched baseline.
3. PASS — 3.5062 ≤ 3.5534.
4. PASS — 27,184 tok/s ≥ 26,668.
5. PASS — all 8 buckets better than B5, incl. bucket 7.
6. PASS — all flags default-off; disabled parity (300-step gate A/C).

Caveat before flipping any production default: the criteria were
pre-registered for a clamp-less recipe; the clamps are now part of the
candidate.  The clean closing move is a **same-binary multi-seed
confirmation of the full amended recipe** (e.g. seeds 4242 + 1337 at
SIRA+clamps, expecting no skips and NLL ≤ ~3.55) before promotion, plus a
flagship-doc/CLAUDE.md update if promoted.  Default flip remains a
user/owner decision.

---

## Multi-seed confirmation of the amended recipe: GATE PASS (2026-06-12)

Final closing runs for the promotion package: seeds `4242` and `1337` at the
full amended candidate recipe (SIRA E+B, `--lr 7.5e-5 --grad-clip 0.5`,
zloss+qk-norm, `--dq-layer-clamp 1.0 --dq-embed-clamp 1.0`), same binary as
the seed-2024 PASS and the criterion-2 baseline.

Artifacts (glades-trainer): `logs/sira_layerclamp_confirm_seed4242_30k_20260611_192427/`,
`logs/sira_layerclamp_confirm_seed1337_30k_20260611_192427/`.
Checkpoints: `database/checkpoints/sira_clamp_confirm/chiron_sira_layerclamp_tau1_seed{4242,1337}_20260611_192427.final`.

### Complete multi-seed gate (current binary, amended recipe)

| seed | final NLL | Δ vs B5 (3.5734) | skips | clamp firing steps | warm tok/s |
|---|---:|---:|---:|---:|---:|
| 2024 | 3.5062 | −0.0672 | 0 | 443 layer + 467 embed (23.8–24.3k) | 27,184 |
| 4242 | 3.5194 | −0.0540 | 0 | **0** | 27,325 |
| 1337 | 3.5414 | −0.0320 | 0 | **0** | 27,304 |

Mean final NLL **3.5223 ± 0.0146** (−0.0511 vs B5).  Every seed clears the
single-run NLL gate (≤ 3.5534) and the throughput bar (≥ 26,668).  Matched
no-SIRA baseline (seed 2024, same recipe/binary): 3.5373 → SIRA contributes
−0.0311 nat.

Notable: seeds 4242 and 1337 had **zero clamp firings in 30k** — at this
recipe their trajectories never approach the τ=1.0 boundary, so those runs
are mathematically identical to unclamped runs.  The clamps acted only
where needed (seed 2024's data-driven burst) and were inert insurance
elsewhere.  The 21k val bump + bucket-3 spike reproduced on all three
seeds (data-window feature, recovers by 24k in all cases).

### Promotion package — COMPLETE

All six pre-registered criteria PASS at the amended recipe, now with a
same-binary 3-seed 30k gate (2024/4242/1337), an attribution baseline, and
disabled-parity evidence.  Remaining steps are owner decisions:

1. Flip the flagship recipe (run.sh + CLAUDE.md) to include
   `--sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25
   --sira-action-weight 0.0 --sira-warmup 1000 --lr 7.5e-5 --grad-clip 0.5
   --dq-layer-clamp 1.0 --dq-embed-clamp 1.0`, OR keep SIRA opt-in.
2. Designate a new flagship checkpoint (best candidate:
   `chiron_sira_layerclamp_tau1_seed2024_20260611_081639.final`,
   val NLL 3.5062) or re-train at a blessed seed.
3. Update CLAUDE.md (also still missing the post-closure SIRA/PHS/PTOC
   history) and MEMORY.md (over size limit).
