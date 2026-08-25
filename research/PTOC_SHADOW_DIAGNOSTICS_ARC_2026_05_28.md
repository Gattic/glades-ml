# PTOC Shadow Diagnostics Arc (2026-05-28)

## Decision

PHS remains opt-in/research diagnostics, not a production default and not an
active controller.  SIRA remains default-off; the current port regularizes only
terminal `(q_L,p_L)` and is not promoted because seed `4242` exposed instability.

The next CHIRON-native arc is **PTOC shadow diagnostics first**.  PTOC is now a
default-off, detached diagnostic path for sampled finite-difference tangent gain
and curvature.  No PTOC loss, PHS weighting, SIRA promotion, sample weighting, or
hidden-state gradient injection is enabled by this change.

## Implemented surface

`glades-ml`:

- `TransformerRunConfig` fields:
  - `ptocShadowDiagnostics=false`
  - `ptocLogEverySteps=0`
  - `ptocSampleLayers=2`
  - `ptocSampleTokens=64`
  - `ptocEps=1e-3`
  - `ptocEta=1e-12`
- Validation rejects invalid PTOC cadence/sample/epsilon settings and requires a
  positive cadence when PTOC is enabled.
- `glades::chiron::ptoc_should_log(...)` and
  `ptoc_detached_diagnostics_from_triplets(...)` summarize supplied
  `(F(x-eps u), F(x), F(x+eps u))` triplets into gain, curvature, and cycle
  metrics without writing anything in disabled mode.
- Targeted unit selector: `chiron-ptoc` / `ptoc`.

`glades-trainer`:

- CLI flags:
  - `--ptoc-shadow-diagnostics`
  - `--ptoc-log-every N`
  - `--ptoc-sample-layers N`
  - `--ptoc-sample-tokens N`
  - `--ptoc-eps F`
  - `--ptoc-eta F`
  - `--ptoc-config-smoke`
  - `--ptoc-shadow-smoke`
- Runtime hook logs detached ReLN finite-difference diagnostics at sampled layer
  parameters around terminal `q_L` anchors.  This is the first low-overhead PTOC
  probe; it is not a full attention-shear PTOC loss.
- PTOC logging disables CUDA graph capture because it performs host downloads and
  CPU finite-difference reductions.
- `scripts/ptoc_shadow_smoke.sh` validates config propagation, detached math,
  and a tiny 2-step runtime PTOC log.
- `scripts/ptoc_short_gate.sh` runs the pre-registered 2k multi-seed PTOC gate
  (optionally with PHS aligned in the same log via `ENABLE_PHS=1`).
- `scripts/ptoc_log_report.py` parses PTOC logs and aligns PTOC curvature/gain
  with PHS qout, SIRA terminal-gradient RMS, and train gradient norm when those
  diagnostics are present in the same log.

## Log interpretation

Trainer line format:

```text
[ptoc step      N] shadow=reln-detached layers=[...] tokens=... rows=... eps=...
  gain(mean/max)=.../... curvature(mean/max)=.../... cycle(mean/max)=.../...
  qout_sample=... loss=none gradients=none
```

Definitions:

- `gain = ||F(x+eps u)-F(x-eps u)|| / (2 eps ||u||)`.
- `curvature = ||F(x+eps u)-2F(x)+F(x-eps u)|| / (eps^2 ||u||)`.
- `cycle = ||second difference|| / ||central difference||`.
- `qout_sample` is a sampled terminal-`q_L` max/RMS proxy over the same anchor
  rows.  It is included to connect PTOC with the PHS qout outlier question.

## How PTOC is used to explain existing findings

1. **PHS qout late-run outliers**
   - Run PTOC with PHS in the same log at the same cadence.
   - Use `scripts/ptoc_log_report.py` to measure descriptive correlations:
     `curvature_max ↔ qout_sample`, `cycle_max ↔ qout_sample`, and, when PHS is
     active, `curvature_max ↔ phs_qout_max`.
   - A qout spike is considered PTOC-explained only if PTOC curvature/gain rises
     before or at the same checkpoint, not merely afterward.

2. **SIRA seed `4242` instability**
   - Re-run the known debug window with PTOC logging aligned to SIRA debug
     cadence.
   - Use the PTOC report to align `curvature_max` with SIRA terminal-gradient RMS
     and train grad norm.
   - If PTOC curvature spikes before SIRA grad spikes, PTOC becomes a candidate
     guard/diagnostic.  If not, keep PTOC diagnostic-only and continue looking at
     SIRA-specific terminal-loss gradient paths.

## Verification performed on implementation

- `cd /home/robert/dev/glades-ml && cmake --build build -j2`
- `cd /home/robert/dev/glades-ml && cmake --build unit-tests/build -j2`
- `./unit-tests/test.sh chiron-ptoc`
- `./unit-tests/test.sh chiron-phs`
- `./unit-tests/test.sh chiron-sira`
- `cd /home/robert/dev/glades-trainer && bash build.sh`
- `scripts/ptoc_shadow_smoke.sh` (config smoke, detached math smoke, and tiny
  2-step runtime PTOC log)
- `bash -n scripts/ptoc_short_gate.sh`
- `scripts/phs_shadow_smoke.sh`
- `scripts/sira_config_smoke.sh`
- `scripts/sira_training_loss_smoke.sh`
- `python3 -m py_compile scripts/ptoc_log_report.py scripts/phs_log_report.py`
- `git diff --check` in both repositories

## Short gate commands

```bash
# glades-ml unit tests
cd /home/robert/dev/glades-ml
cmake --build unit-tests/build -j2
./unit-tests/test.sh chiron-ptoc

# trainer build/smoke
cd /home/robert/dev/glades-trainer
bash build.sh
scripts/ptoc_shadow_smoke.sh

# 2k diagnostic gate, scripted multi-seed version
SEEDS="1337 2024 777 4242" STEPS=2000 scripts/ptoc_short_gate.sh
ENABLE_PHS=1 SEEDS="1337 2024 777 4242" STEPS=2000 scripts/ptoc_short_gate.sh

# 2k diagnostic gate, direct same-seed research wrapper example
./run.sh flagship --steps 2000 --seed 1337 \
  --zloss-coef 1e-4 --qk-norm \
  --ptoc-shadow-diagnostics --ptoc-log-every 500 \
  --ptoc-sample-layers 2 --ptoc-sample-tokens 64 --ptoc-eps 1e-3

# PTOC + PHS qout alignment
./run.sh flagship --steps 2000 --seed 1337 \
  --zloss-coef 1e-4 --qk-norm \
  --phs-shadow-diagnostics --phs-log-every 500 \
  --ptoc-shadow-diagnostics --ptoc-log-every 500

scripts/ptoc_log_report.py 'logs/*ptoc*.log'
```

## Promotion policy

Do not add a PTOC loss yet.  A bounded default-off PTOC objective can be designed
only after PTOC shadow logs show that local gain/curvature predicts one of:

- PHS qout spikes outside the clean-run envelope,
- SIRA terminal-gradient spikes / seed instability,
- validation or position-bucket regressions.

Even then, the active loss must remain default-off until separate stability,
throughput, and same-seed NLL gates pass.

## Evaluation addendum: 20k seed-4242 LR-stabilized no-FP8 match (2026-05-31)

This addendum follows the late seed-`4242` instability investigation.  Reducing
LR to `7.5e-5` stabilized the clean FP8 flagship through 20k, but the first
PTOC+PHS attempt with FP8 readout exhausted VRAM before step 1:

```text
[fp8] cudaMalloc(33554432) failed: out of memory
forward failed at step 0
```

The PTOC+PHS shadow run was therefore repeated with the same flagship/regstack
recipe **minus `--fp8-readout-fwd`**.  A matched no-FP8 baseline was then run to
separate the effect of removing FP8 readout from the effect of enabling detached
PTOC/PHS diagnostics.

Artifacts in `/home/robert/dev/glades-trainer`:

- FP8 baseline reference:
  `logs/baseline_20k_seed4242_lr75e5_20260530_211425/baseline_T16384_20000step_seed4242_lr75e5_20260530_211425.log`
- matched no-FP8 baseline:
  `logs/baseline_20k_seed4242_lr75e5_nofp8_20260531_101820/baseline_T16384_20000step_seed4242_lr75e5_nofp8_20260531_101820.log`
- PTOC+PHS no-FP8:
  `logs/ptoc_phs_20k_seed4242_lr75e5_nofp8_20260531_062132/ptoc_phs_T16384_20000step_seed4242_lr75e5_nofp8_20260531_062132.log`
- PTOC report:
  `logs/ptoc_phs_20k_seed4242_lr75e5_nofp8_20260531_062132/ptoc_report_20260531_062132.md`
- parsed comparison JSON:
  `logs/ptoc_phs_20k_seed4242_lr75e5_nofp8_20260531_062132/matched_baseline_compare_20260531_101820.json`

All three completed 20k with zero bad/stability lines and no grad-skip.

| Run | FP8 readout | PTOC | PHS | final NLL | bpb | ppl | acc1 | acc5 | acc10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline LR `7.5e-5` | yes | no | no | 3.5698 | 1.2875 | 35.51 | 0.1370 | 0.5495 | 0.8338 |
| matched baseline LR `7.5e-5` no-FP8 | no | no | no | 3.5786 | 1.2907 | 35.82 | 0.1365 | 0.5508 | 0.8321 |
| PTOC+PHS LR `7.5e-5` no-FP8 | no | yes | yes | 3.5834 | 1.2924 | 36.00 | 0.1365 | 0.5488 | 0.8311 |

Matched deltas:

| Delta | NLL | bpb | ppl | acc1 | acc5 | acc10 | warm tok/s mean | warm grad mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no-FP8 baseline − FP8 baseline | +0.0088 | +0.0032 | +0.31 | -0.0005 | +0.0013 | -0.0017 | -378.3 | -0.0089 |
| PTOC+PHS no-FP8 − no-FP8 baseline | +0.0048 | +0.0017 | +0.18 | +0.0000 | -0.0020 | -0.0010 | -18.2 | -0.0039 |
| PTOC+PHS no-FP8 − FP8 baseline | +0.0136 | +0.0049 | +0.49 | -0.0005 | -0.0007 | -0.0027 | -396.5 | -0.0128 |

Final position-bucket NLL deltas:

- no-FP8 baseline − FP8 baseline:
  `[+0.000, +0.000, +0.010, +0.010, +0.010, +0.020, +0.010, +0.010]`
- PTOC+PHS no-FP8 − no-FP8 baseline:
  `[+0.000, +0.000, +0.000, +0.000, +0.000, +0.000, +0.010, +0.020]`

PTOC/PHS no-FP8 telemetry remained in the clean envelope and retained strong
alignment signal:

- PTOC checkpoints: `80`
- `curvature_max`: min `2.9317`, median `3.8886`, p90 `4.2350`, max `4.4576`
- `gain_max`: min `0.9989`, median `1.0117`, p90 `1.0175`, max `1.0245`
- `qout_sample`: min `3.726`, median `13.9795`, p90 `18.0344`, max `21.380`
- `phs_qout_max`: min `4.44`, median `15.755`, p90 `20.10`, max `24.84`
- `corr(curvature_max, qout_sample)=0.880`
- `corr(cycle_max, qout_sample)=0.889`
- `corr(curvature_max, phs_qout_max)=0.874`

Interpretation: the +0.0136 NLL gap versus the clean FP8 baseline decomposes
into +0.0088 from removing FP8 readout and +0.0048 residual between matched
no-FP8 baseline and PTOC+PHS no-FP8.  Thus most of the observed gap is FP8
removal, not PTOC/PHS.  The residual is small, has no stability signature, and
is concentrated only in the last two final-validation position buckets; treat it
as PTOC/PHS-or-noise until another matched 20k ablation lands.

Recommended next 20k ablation: run **PHS-only no-FP8** with the same
seed/LR/recipe (`--phs-shadow-diagnostics --phs-log-every 250`, no PTOC).  This
isolates whether the residual comes from PHS host diagnostics versus PTOC
finite-difference diagnostics or the PTOC×PHS interaction.  If PHS-only matches
the no-FP8 baseline, run PTOC-only next; if PHS-only matches the combined run,
inspect PHS synchronization/data-path effects before considering any active loss.

## Evaluation addendum: FP8 readout restored and 2k default-off ablation (2026-06-01)

The FP8 readout OOM was traced to insufficient headroom before first-step lazy
CUDA/cuBLAS allocations, not to a new PTOC/PHS allocation.  `glades-trainer`
commit `232c9ec78b3180f5661640f793609036e39cb2d0` reuses the BF16 logits buffer
as BF16 `dlogits` on `--fp8-readout-fwd`, restoring FP8 headroom while leaving
the no-FP8 path unchanged.

Artifacts in `/home/robert/dev/glades-trainer`:

- 10-step FP8 matrix:
  `logs/fp8_alias_diag_matrix_10step_20260531_162938/`
- 100-step FP8 matrix:
  `logs/fp8_alias_diag_matrix_100step_20260531_175403/`
- 500-step FP8 baseline vs PTOC+PHS:
  `logs/fp8_alias_baseline_vs_ptoc_phs_500step_20260531_181746/`
- 1k-step FP8 baseline vs PTOC+PHS:
  `logs/fp8_alias_baseline_vs_ptoc_phs_1000step_20260531_183127/`
- 2k-step FP8 baseline vs PTOC+PHS:
  `logs/fp8_alias_baseline_vs_ptoc_phs_2000step_20260601_003004/`
- 2k four-arm FP8 ablation matrix:
  `logs/fp8_ablation_matrix_2000step_20260601_070850/`

All FP8 gates completed with `VRAM usage after allocation: 14.49 / 15.56 GB`,
zero OOM/fallback/rejection/NaN/Inf/grad-skip/forward/backward/stability lines,
and no persistent warm-throughput regression.  When a diagnostic fires on the
final step, the instantaneous final-step `tok/s` is expectedly depressed by host
logging; use warm mean or non-diagnostic-step throughput for comparisons.

Same-seed 2k four-arm ablation:

| Run | val NLL | ΔNLL vs baseline | warm tok/s | Δ warm tok/s | bad lines |
|---|---:|---:|---:|---:|---:|
| baseline | 4.9538 | +0.0000 | 28044.6 | +0.0 | 0 |
| PTOC-only | 4.9514 | -0.0024 | 28017.4 | -27.3 (-0.10%) | 0 |
| PHS-only | 4.9574 | +0.0036 | 27983.3 | -61.3 (-0.22%) | 0 |
| PTOC+PHS | 4.9515 | -0.0023 | 27965.5 | -79.1 (-0.28%) | 0 |

Telemetry at 2k stayed bounded:

- PHS-only final `qout_mean=4.879`, `qout_max=5.20`; maximum observed qout
  across PHS arms was `5.68`.
- PTOC-only final `gain_mean=0.94524`, `curvature_max=3.3155`,
  `cycle_max=0.0016591`, `qout_sample=5.179`.
- PTOC+PHS final `gain_mean=0.94525`, `curvature_max=3.3348`,
  `cycle_max=0.0016722`, `qout_sample=5.144`.
- Maximum observed PTOC curvature was `3.5358`; maximum observed cycle was
  `0.0017706`.

Attribution: the combined small NLL improvement tracks PTOC-only rather than
PHS-only (`PTOC-only -0.0024`, `PHS-only +0.0036`, `PTOC+PHS -0.0023`, combo
minus PTOC-only `+0.0001`).  Because PTOC/PHS are detached shadow diagnostics,
these tiny deltas are not causal evidence for an active loss or default enable.

Decision: freeze PTOC and PHS as CHIRON-native, default-off shadow diagnostics.
Do not promote PTOC/PHS to default-on, sample weighting, token weighting, or
active loss from this evidence.  Active work should proceed through SIRA Phase-0
trajectory diagnostics first, then only consider active SIRA/PTOC/PHS objectives
if detached telemetry predicts a clear stability or validation target.
