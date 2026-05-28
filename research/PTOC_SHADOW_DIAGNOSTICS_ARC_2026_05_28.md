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
