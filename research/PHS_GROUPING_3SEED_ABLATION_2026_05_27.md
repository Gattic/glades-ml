# PHS Grouping 3-Seed Ablation — Batch-Quantile vs ID Bins (2026-05-27)

## Question

PHS shadow diagnostics initially grouped tokens by raw target-token ID bins. A 2000-step T=16384 run showed stable structural skew (~73/15/8/5%), making group-level PHS statistics hard to interpret. This ablation compares that legacy `id` grouping against `batch-quantile`, which sorts the current batch targets and assigns equal-count token-rank groups.

PHS remained diagnostics-only in all runs: no sample weighting, token-loss weighting, hidden-state gradient path, or optimizer change.

## Configuration

Common recipe, run from `glades-trainer`:

```bash
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
  --phs-log-every 200 --phs-ema-decay 0.95 \
  --phs-group-mode <id|batch-quantile>
```

Seeds: `1337`, `2024`, `777`.

Logs:

- `logs/phs_flagship_T16384_2000step_20260527_100011.log` — seed 1337, `id`
- `logs/phs_flagship_T16384_2000step_batchquant_20260527_103741.log` — seed 1337, `batch-quantile`
- `logs/phs_3seed_T16384_2000step_seed2024_id_20260527_114521.log`
- `logs/phs_3seed_T16384_2000step_seed2024_batch-quantile_20260527_114521.log`
- `logs/phs_3seed_T16384_2000step_seed777_id_20260527_114521.log`
- `logs/phs_3seed_T16384_2000step_seed777_batch-quantile_20260527_114521.log`

## Stability

All six runs completed 2000 steps with no NaN/Inf/bad-gradient/grad-skip lines. VRAM was unchanged at `15.00 / 15.56 GB`.

## Per-seed result

Batch-quantile minus ID. Final train row is step 1901; final PHS row is step 2000.

| seed | bad lines | EMA Δ | loss Δ | best Δ | tok/s Δ | wall Δs | countCV ID→BQ | ID final shares | BQ final shares | logpq std ID→BQ | qout std ID→BQ |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| 1337 | 0/0 | -0.0013 | +0.0039 | +0.0086 | -10.5 | +0.4 | 1.083→0.094 | `[.714,.157,.079,.050]` | `[.250,.250,.250,.250]` | .062→.049 | .220→.147 |
| 2024 | 0/0 | -0.0102 | +0.0230 | -0.0352 | -31.1 | +1.2 | 1.083→0.094 | `[.714,.157,.079,.050]` | `[.250,.250,.250,.250]` | .070→.058 | .209→.205 |
| 777 | 0/0 | +0.0105 | +0.0141 | +0.0436 | -2.0 | +0.1 | 1.083→0.094 | `[.714,.157,.079,.050]` | `[.250,.250,.250,.250]` | .056→.048 | .283→.337 |

## Across-seed delta summary

| metric, batch-quantile minus ID | mean Δ | sample sd | interpretation |
|---|---:|---:|---|
| final logged EMA | -0.00033 | 0.01038 | no learning effect |
| final logged loss | +0.01367 | 0.00956 | tiny/noisy |
| best tracker | +0.00567 | 0.03948 | no consistent effect |
| mean tok/s excluding step 1 | -14.54 | 14.96 | negligible, <0.1% |
| final count CV | -0.98924 | 0.00000 | large deterministic improvement |
| final logpq cell std | -0.01082 | 0.00275 | consistently slightly cleaner |
| final qout cell std | -0.00777 | 0.06351 | mixed/noisy |
| final position-NLL mean | +0.00096 | 0.01772 | no effect |

Mean loss curves did not separate materially. At step 1901, across-seed means were:

| mode | loss | EMA | mean tok/s excluding step 1 |
|---|---:|---:|---:|
| ID bins | 5.5707 | 5.6562 | 28,870.8 |
| batch-quantile | 5.5844 | 5.6558 | 28,856.2 |
| Δ BQ-ID | +0.0137 | -0.0003 | -14.5 |

## PHS checkpoint means across seeds

| PHS step | countCV ID | countCV BQ | logpq std ID | logpq std BQ | qout std ID | qout std BQ |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 1.089 | 0.067 | 0.730 | 0.794 | 0.246 | 0.257 |
| 400 | 1.103 | 0.044 | 0.528 | 0.523 | 0.287 | 0.249 |
| 800 | 0.991 | 0.322 | 0.236 | 0.182 | 0.240 | 0.230 |
| 1200 | 1.161 | 0.130 | 0.077 | 0.073 | 0.229 | 0.244 |
| 1600 | 1.078 | 0.116 | 0.034 | 0.032 | 0.236 | 0.221 |
| 2000 | 1.083 | 0.094 | 0.062 | 0.052 | 0.237 | 0.230 |

## Decision

Make `batch-quantile` the default PHS shadow-diagnostic grouping mode when `--phs-shadow-diagnostics` is enabled and `phsDataGroups > 1`. Keep `--phs-group-mode id` as an explicit compatibility override.

Rationale:

- ID bins are persistently imbalanced at approximately 71/16/8/5%, dominated by token-frequency/ID structure rather than PHS behavior.
- Batch-quantile gives exactly balanced group totals and much lower per-cell count CV, making group-axis PHS diagnostics statistically interpretable.
- Loss, EMA, best tracker, VRAM, stability, and throughput deltas are noise-level because the path is logging-only.
- PHS should remain shadow-only. This result just changes monitoring defaults; it does not justify active PHS weighting or hidden-state gradient injection.

Next useful ablation, if more grouping work is needed: compare `batch-quantile` against a static frequency-remapped token bucket table, not against raw ID bins again.
