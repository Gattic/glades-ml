# CHIRON-native terminal SIRA 30k result and flagship decision (2026-05-27)

**Status:** HOLD / default-off. Do **not** include SIRA by default in the
flagship yet.

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

## Flagship recommendation

Keep flagship defaults unchanged:

- `siraCoef = 0.0`
- SIRA remains opt-in.
- Disabled mode must remain parity/bit-identical.

Do not flip SIRA on by default until all of the following are true:

1. Multi-seed 30k includes seed `4242` or an equally adversarial seed without
   NaNs, non-finite gradients, or guard skips.
2. The first-cause of the 4242 spike is understood or the mitigation is
   principled and quality-neutral.
3. Final NLL clears either the single-run gate (`<= 3.5534`) or a
   pre-registered multi-seed criterion.
4. Throughput remains `>= 26668 tok/s`.
5. Position bucket 7 / long-context metrics do not regress.
6. SIRA flags remain default-off and disabled parity remains verified.

Short version: **SIRA is promising, but not flagship-default shippable yet.**

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
