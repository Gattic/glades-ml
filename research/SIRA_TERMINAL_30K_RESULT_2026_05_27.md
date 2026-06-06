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

Interpretation: rowamp far above the ordinary `gamma/sigma` bound is explained
by the ReLN backward using a reconstructed `q_in` whose normalized residuals are
no longer close to the saved forward normalization.  Once `xhat` is huge, the
LayerNorm/ReLN backward projection term can amplify far beyond the clean
`gamma/sigma` intuition.  The open question is why the reverse reconstruction
enters this mismatched state in bad trajectories (upper-layer reverse-state
corruption, aliasing/state overwrite, or a run-sensitive numerical path), not
whether low `sigma` alone explains the spikes.

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
