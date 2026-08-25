# PHS Default-Enablement Decision Plan (2026-05-27)

**Execution status (2026-05-28):** completed through Stage 2. Result: keep
PHS globally default-off for production/flagship training; allow default-on
shadow diagnostics in research/diagnostic templates. See
`research/PHS_DEFAULT_ENABLEMENT_RESULT_2026_05_28.md`.

## Goal

Determine whether PHS should become part of the default CHIRON 1B model/training recipe, and if so in which form:

1. **PHS shadow diagnostics default-on**: logging/monitoring only.
2. **PHS active controller default-on**: any token/sample weighting, curriculum, or loss/controller path.

Current recommendation before this plan is complete: **do not enable PHS by default**. PHS is stable and useful as opt-in shadow diagnostics, but default-on diagnostics add host downloads and disable CUDA graph capture, while active PHS weighting is not implemented or validated.

## Current evidence baseline

- PHS is implemented as **default-off, detached shadow diagnostics** in `glades-ml` and `glades-trainer`.
- Trainer PHS logs terminal `(q_L,p_L)` only and uses `p_L` as a detached shear proxy.
- PHS does **not** currently change loss, sampling, token weights, hidden states, gradients, optimizer state, or checkpoint semantics.
- Batch-quantile target-token grouping is now the PHS diagnostic default when PHS is explicitly enabled.
- 3-seed ID-bin vs batch-quantile 2000-step ablation showed:
  - no NaN/Inf/bad-grad/grad-skip;
  - no meaningful loss/EMA/best/throughput effect;
  - final count CV improved from `1.083` to `0.094` across all seeds;
  - batch-quantile is the correct diagnostic grouping default.
- Key evidence doc: `research/PHS_GROUPING_3SEED_ABLATION_2026_05_27.md`.

## Decision definitions

### D0 — Keep PHS default-off

Default if any required gate below fails. This means production/flagship training commands do not include PHS unless explicitly requested.

### D1 — Enable PHS shadow diagnostics by default for research recipes only

Allowed only if PHS diagnostics provide actionable monitoring value with negligible overhead and no ecosystem incompatibility. This does not affect model math.

### D2 — Enable PHS shadow diagnostics by default for flagship/production training

Higher bar than D1 because PHS host downloads disable CUDA graph capture and add log/D2H overhead.

### D3 — Enable active PHS by default

Highest bar. Requires a separate implementation and proof that active PHS improves generalization/stability without adding unacceptable throughput, variance, or hidden failure modes.

## Non-negotiable constraints

- PHS must remain CHIRON-native and phase-state based.
- Disabled mode must remain bit-identical and default-safe until a decision explicitly changes it.
- Shadow diagnostics must remain detached: no hidden-state gradient path.
- Do not use PHS to revive naive MTP, LayerDrop, or UL2-style ports.
- Do not enable active PHS weighting without a separate gated implementation plan.
- Use same-seed comparisons for flagship claims.
- Use existing baseline records for the 30k disabled/B5 baseline unless explicitly deciding to rerun.
- Report throughput/VRAM/stability honestly; default-on diagnostics must account for CUDA graph incompatibility.

## Metrics to collect

### Training quality

- Train loss curve and EMA.
- Best train-loss tracker.
- Validation NLL at preregistered checkpoints and final 30k.
- Validation position-bucket NLLs.
- Optional downstream evals already used for CHIRON if available.

### PHS health

- Group shares and per-cell count CV.
- `logpq` mean/std/range by group and position bucket.
- `qout` mean/std/range by group and position bucket.
- Position-NLL proxy mean/spread.
- EMA behavior and missing/empty bucket behavior.
- Alerts for large PHS excursions, e.g. sustained `qout` spikes or large `logpq` swings.

### Stability

- NaN/Inf/bad-gradient/grad-skip/overflow scans.
- Gradient norm and clipping scale distributions.
- Seed sensitivity, especially prior unstable seeds such as `4242` when active mechanisms are introduced.

### Systems cost

- tok/s mean/range excluding step 1.
- wall time.
- VRAM.
- CUDA graph compatibility loss or recapture behavior.
- Log volume and host-download overhead.

## Stage 0 — Freeze and verify the current default-off contract

**Purpose:** prove that PHS remains safe when disabled before evaluating default enablement.

Commands:

```bash
cd ~/dev/glades-ml
cmake --build unit-tests/build -j2
./unit-tests/test.sh chiron-phs
git diff --check

cd ~/dev/glades-trainer
bash build.sh
scripts/phs_shadow_smoke.sh
sh run.sh flagship --phs-config-smoke \
  --phs-shadow-diagnostics --phs-data-groups 2 --phs-position-buckets 4 \
  --phs-log-every 3 --phs-ema-decay 0.8 --phs-group-mode id
git diff --check
```

Pass criteria:

- All commands pass.
- Disabled PHS remains default-off in config and trainer startup.
- Explicit `--phs-group-mode id` override still works.

## Stage 1 — Decide whether shadow diagnostics should be default-on for research recipes

**Purpose:** determine if passive PHS logging is useful enough to include by default in research runs.

Run matrix:

| Run | steps | seeds | PHS config | save |
|---|---:|---|---|---|
| S1-a | 2000 | `1337,2024,777` | disabled | logs only |
| S1-b | 2000 | `1337,2024,777` | `--phs-shadow-diagnostics --phs-data-groups 4 --phs-position-buckets 8 --phs-log-every 500 --phs-ema-decay 0.95 --phs-group-mode batch-quantile` | logs only |
| S1-c | 2000 | `4242` | same PHS config | logs only |

Notes:

- S1-b partly overlaps with prior evidence, but use `--phs-log-every 500` to evaluate the recommended low-overhead monitoring cadence.
- S1-a disabled reruns are optional if recent same-seed disabled traces exist with identical config; otherwise run them to estimate pure logging overhead.

Pass criteria for research-default diagnostics:

- No NaN/Inf/bad-grad/grad-skip in all PHS runs.
- PHS group coverage remains full or explainably sparse only for tiny smoke configs.
- Count CV remains low under batch-quantile grouping.
- Mean tok/s regression vs disabled is `<0.5%` at `--phs-log-every 500`.
- PHS metrics produce actionable summaries: finite `logpq`, bounded `qout`, and interpretable bucket trends.

Decision:

- If pass, PHS shadow diagnostics may be included in **research run templates**.
- If fail, keep PHS opt-in only.
- This stage is not sufficient to make PHS default-on for flagship/production training.

## Stage 2 — Decide whether shadow diagnostics should be default-on for flagship/production training

**Purpose:** evaluate whether passive PHS should be in the standard flagship recipe despite host-download and CUDA-graph incompatibility.

Run matrix:

| Run | steps | seeds | PHS config | compare against |
|---|---:|---|---|---|
| S2-a | 10000 | `1337,2024,777` | disabled or existing disabled logs | same-seed PHS |
| S2-b | 10000 | `1337,2024,777` | recommended PHS diagnostics, log every 500 | S2-a |
| S2-c | 30000 | `1337` | recommended PHS diagnostics, log every 500 | baseline record |

Recommended PHS diagnostics:

```bash
--phs-shadow-diagnostics \
--phs-data-groups 4 \
--phs-position-buckets 8 \
--phs-log-every 500 \
--phs-ema-decay 0.95 \
--phs-group-mode batch-quantile
```

Pass criteria for production-default diagnostics:

- No stability regressions across 10k multi-seed and 30k seed `1337`.
- Final/train loss and validation NLL are statistically indistinguishable from disabled same-seed records.
- Throughput regression is `<0.25%` and wall/log overhead is acceptable.
- CUDA graph tradeoff is explicitly acceptable. If production intends to use CUDA graphs, default-on PHS diagnostics should fail this gate unless PHS logging is redesigned to be graph-compatible or inactive under graph capture.
- Log volume and operational noise are acceptable.

Expected likely outcome:

- **Do not enable PHS shadow diagnostics by default for production** unless the team prioritizes observability over CUDA-graph compatibility and small throughput overhead. Prefer explicit opt-in for PHS monitoring.

## Stage 3 — Build an active PHS candidate only if shadow metrics predict useful interventions

**Purpose:** determine if PHS should ever affect training by default. This stage requires new code and separate review.

Prerequisite evidence:

- Shadow PHS metrics correlate with later NLL, instability, gradient spikes, or bucket regressions across multiple seeds.
- There is a clear intervention target, e.g. specific group/position cells with persistent `logpq` or `qout` pathologies that precede worse validation buckets.

Allowed first active candidates, in order of safety:

1. **Alert-only / early-warning policy**: no training math changes; fails or annotates runs when PHS metrics exceed thresholds.
2. **Detached token-loss reweighting**: bounded, mean-normalized weights computed from PHS EMAs; no hidden-state gradients; warmup and clamps required.
3. **Detached sampling/curriculum adjustment**: changes future data mixture only if the trainer/data pipeline supports stable, reproducible remapping.

Disallowed for first active arc:

- Direct hidden-state gradient injection from PHS metrics.
- Unbounded weights.
- Per-step branchiness incompatible with baseline reproducibility.
- Making active PHS default-on before a default-off flag has passed all gates.

Implementation gates for any active candidate:

- New config flags with disabled defaults.
- Disabled parity test: no reads/writes/RNG/FP work when disabled.
- Config validation and smoke tests.
- Unit/reference tests for weight computation or policy decisions.
- Trainer logs showing controller state, effective weights, clamps, and mean-normalization.

## Stage 4 — Active PHS ablation ladder

Only run after Stage 3 implementation exists.

### Gate A — tiny/smoke

- 2-step tiny pretokenized smoke.
- Verify disabled parity, enabled finite metrics, and no NaN.

### Gate B — 2000-step pilot

| Run | steps | seeds | config |
|---|---:|---|---|
| B0 | 2000 | `1337,2024,777,4242` | disabled or shadow-only |
| B1 | 2000 | `1337,2024,777,4242` | active PHS low gain |
| B2 | 2000 | `1337,2024,777,4242` | active PHS medium gain |

Pass criteria:

- No NaN/Inf/bad-grad/grad-skip.
- No new gradient spike pattern.
- Mean train EMA not worse by more than `0.02` nat at 2000.
- Throughput regression `<1%`.
- PHS metrics improve in the intended direction without bucket collapse.

### Gate C — 10000-step multi-seed

Seeds: `1337,2024,777` plus `4242` if Gate B found any instability.

Pass criteria:

- Same stability requirements.
- Same-seed loss/NLL trajectory not worse.
- PHS metric variance improvements persist.
- No position-bucket or group-specific regression.

### Gate D — 30000-step flagship

Use existing disabled/B5 baseline record for primary comparison unless a same-code disabled rerun is explicitly needed.

Minimum ship/default gate:

- Mean 30k validation NLL improves by at least `0.01` nat over baseline across successful seeds, or demonstrates a clear stability/default-value win with no NLL loss.
- No single seed has worse final NLL by `>0.02` nat unless explained and reproduced.
- No bucket NLL regression `>0.03` nat in any persistent bucket.
- Throughput regression `<1%` for active PHS.
- VRAM remains within the current 4080 SUPER budget.
- Seed `4242` does not reintroduce unresolved instability.

## Final default decision matrix

| Evidence outcome | Decision |
|---|---|
| Shadow PHS only useful for analysis, overhead/graph tradeoff nonzero | keep default-off; document opt-in recipe |
| Shadow PHS has negligible overhead and catches actionable issues in research | default-on only in research templates |
| Shadow PHS negligible overhead and production accepts graph tradeoff | consider production default-on diagnostics |
| Active PHS improves 30k NLL/stability across seeds and passes all gates | consider active default-on behind a final ship review |
| Active PHS mixed, noisy, or seed-sensitive | keep opt-in/research-only |
| Any disabled parity or stability regression | block default enablement |

## Recommended immediate next step

Run Stage 1 with the low-overhead recommended diagnostic cadence:

```bash
cd ~/dev/glades-trainer
# For seeds 1337, 2024, 777, 4242:
./build/glades_chiron_train \
  --pretokenized --data-dir pretok-data/ \
  --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 \
  --vocab 32000 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage --fp8-readout-fwd \
  --max-steps 2000 --warmup 500 --lr 1e-4 --grad-clip 0.5 \
  --seed <seed> --log-every 100 --val-every 0 \
  --phs-shadow-diagnostics --phs-data-groups 4 --phs-position-buckets 8 \
  --phs-log-every 500 --phs-ema-decay 0.95 --phs-group-mode batch-quantile
```

Then parse:

- stability scan: NaN/Inf/bad-grad/grad-skip/overflow;
- tok/s mean/range excluding step 1;
- loss and EMA at logged steps;
- PHS group shares/count CV;
- `logpq` and `qout` mean/std/range;
- log size and cadence overhead.

## Current recommendation until this plan passes

- Keep PHS **default-off** in production/flagship recipes.
- Keep batch-quantile as the default **when PHS is explicitly enabled**.
- Use PHS by explicit opt-in for research monitoring:

```bash
--phs-shadow-diagnostics \
--phs-data-groups 4 \
--phs-position-buckets 8 \
--phs-log-every 500 \
--phs-ema-decay 0.95
```

- Do not implement or enable active PHS weighting until shadow metrics show predictive value and a bounded detached controller passes the active ablation ladder.
