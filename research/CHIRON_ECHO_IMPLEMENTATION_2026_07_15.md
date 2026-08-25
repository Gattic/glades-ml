# CHIRON ECHO implementation record — 2026-07-15

## Scope and verdict

ECHO (Excess-Copy Hinged Objective) from
`docs/superpowers/specs/2026-07-09-chiron-loss-regularizers-design.md` is implemented
through engineering gates **E0–E2** and its production memory/overhead gate. The
implementation is default-off, training-only, parameter-free, and adds no checkpoint or
serving state.

**Engineering verdict: PASS.** The optimized hard path uses `0.50 MiB` scratch at the
production `T=16384,w=128` shape (down from `8.25 MiB`) and measured `+0.33%` independent-
median / `+0.09%` paired-median elapsed overhead versus control.

**E3 quality verdict: FAIL / no-go for the tested λ=0.1 recipe.** Final treatment NLL
was `6.7239` versus baseline `6.7077` (`+0.0162`): bounded, but not beneficial. The
pre-registered gradient-ratio and non-increasing-trajectory bars failed. E3.5/E4 are
not promoted. BERM and LOFT remain portfolio designs and were not selected.

## Implementation

### glades-ml

- `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu/.h`
  - Hard-hinge `echo_repeat_stats` and sparse scatter.
  - Optional Huberized hinge with
    `H_delta(x)=x^2/(2 delta)` inside the knee and `x-delta/2` above it.
  - Exact weighted field `pi_u (b_u - sum_v pi_v b_v)` via dense row scalar plus
    weighted sparse scatter.
  - Active owner slots use `T*ceil(w/32)` bit words; scatter recovers vocabulary ids
    from the original token window, eliminating the old `T*w` int-id buffer.
  - `echo_summarize_stats` reduces loss/activation/copy telemetry to 11 GPU scalars,
    replacing four O(T) device-to-host copies and their host scan each step.
  - Per-row maximum active-copy probability remains available for telemetry.
- `Backend/Machine Learning/Networks/transformer_chiron_ops.h`
  - Matching CPU references for hard and Huber stats/backward/scatter.
- `unit-tests/Backend/Machine Learning/chiron-test.cpp`
  - Hard and Huber finite differences, shift invariance, truth exclusion, prefix edges,
    dedup/margins, init inactivity, row-zero-sum, delta-to-zero hard consistency,
    lambda-linear RMS calibration, CPU/GPU parity, and coefficient-zero z-loss parity.

### glades-trainer (sibling repository)

- `trainer/chiron_main.cpp`
  - All public flags: `--echo-coef`, `--echo-window`, `--echo-margin`,
    `--echo-kappa`, `--echo-huber`, and `--echo-warmup`.
  - ECHO can run with `--zloss-coef 0`; the combined kernel accepts a zero Z-loss term.
  - Warmup-gated training forwards only; validation never computes or reports ECHO.
  - No serialized state. Hard mode allocates no Huber-weight scratch.
  - Logs activation rate, active rows/ids, mean dense mass, max-copy mean/max,
    a five-bin max-copy histogram, and ECHO loss from the reduced GPU summary.
  - Production scratch: hard `0.50 MiB`; Huber `8.50 MiB` (FP32 weights retained for
    exact regularizer semantics).
- `run.sh`
  - Routes all ECHO flags in flagship and CHIRON research modes.
- `scripts/echo_training_loss_smoke.sh`
  - Wrapper/CLI propagation, invalid-config interlocks, coefficient-zero parity,
    pre-warmup parity, hard/Huber loss ordering, Z-loss independence, and telemetry.

## Verification evidence

1. Main glades CUDA build: PASS (`build_project`; 65 s).
2. CUDA unit-test build: PASS (`cmake --build unit-tests/build`).
3. `bash unit-tests/test.sh chiron-echo`: PASS.
   - hard FD worst relative error: `7.05e-05` (bar `2e-3`)
   - Huber FD worst absolute error: `3.44e-05` (bar `2e-3`)
   - lambda `0.1` field-RMS / CE-RMS: `0.01777`
   - hard stats CPU/GPU: bit-exact over 64 rows
   - Huber stats/weighted-scatter/composed backward CPU/GPU: PASS
   - hard scatter: `0/32768` differences
   - maximum public window: `w=1024`, shared-hash launch PASS
   - coefficient-zero combined-vs-shipped z-loss kernel: `0/32768` differences
4. Required static-link rebuild order: PASS.
   - `make -C build install`
   - sibling `bash build.sh` (final job `job-mrm37i6q-cb0e`, exit 0)
5. Full `chiron` unit suite: PASS (`test_project`, 298 s).
6. Trainer checkpoint self-test: PASS (job `job-mrm3hm81-5c9d`, exit 0;
   CHRF/CHRN/bad-magic matrix), confirming no ECHO checkpoint-format delta.
7. `bash scripts/echo_training_loss_smoke.sh`: PASS
   (final job `job-mrm3jvye-ed09`, exit 0).
   - disabled `3.5246`
   - coefficient zero `3.5246`; one-step checkpoints byte-identical (`cmp`)
   - pre-warmup `3.5246`
   - Huber `3.5323`
   - hard `3.5614`
   - config smoke confirms `zloss=0`, warmup transition, training-only contract,
     and `checkpoint_state=none`.
8. 2026-07-15 memory/performance pass: PASS.
   - Updated CUDA unit test and trainer smoke: PASS after install/relink.
   - Hard scratch `8.25 -> 0.50 MiB` (`-93.9%`, bar at least `-92%`).
   - Huber scratch `16.25 -> 8.50 MiB` (`-47.7%`, bar at least `-47%`).
   - Pre-optimization n=3 production elapsed medians: control `87.71 s`, ECHO
     `87.64 s` (`-0.08%`, noise), run `run-echo-preopt-production-overhead-n3-ad24`.
   - Optimized n=3 production elapsed medians: control `87.43 s`, ECHO `87.72 s`
     (`+0.33%`; paired-run median `+0.09%`), trainer wall `70.4 s` for both, run
     `run-echo-optimized-production-overhead-n3-b4ff`.
   - Optimized ECHO median was `+0.09%` versus pre-optimization, inside the
     pre-registered `0.3%` no-regression/noise band. All 12 arms exited finite.

## E3 matched quality result

Run `run-echo-e3-matched-2500-pair-061d` completed both 2500-step arms:

| Step | Baseline NLL | ECHO NLL | Δ ECHO−baseline |
|---:|---:|---:|---:|
| 1000 | 7.2505 | 7.2760 | +0.0255 |
| 1500 | 6.9652 | 6.9635 | −0.0017 |
| 2000 | 6.9074 | 6.9178 | +0.0104 |
| 2500 | 6.7077 | 6.7239 | +0.0162 |

- **PASS:** no NaN/divergence/explicit gradient skip; final ΔNLL is inside both the
  `+0.10` kill bound and `+0.04` final-trajectory bound.
- **PASS:** ECHO is non-inert (`~7–12%` activation after warmup) with finite mass/loss.
- **PASS:** wall `5884.5 s` versus `5882.1 s` (`+0.041%`).
- **FAIL:** logged maximum gradient `2.785` versus `1.894` (`1.47×`, bar `≤1.1×`).
- **FAIL:** key-step gaps `+0.0255,−0.0017,+0.0104,+0.0162` are not non-increasing.
- **UNVERIFIED:** step-1 validation NLL matches at four logged decimals (`10.8034`),
  but the logs do not prove bit identity.

Both final saves failed because their requested parent directories did not exist, so no
E3.5 generation checkpoints were produced. This does not invalidate the completed NLL
measurements, but independently blocks E3.5. Since E3 already failed its pre-registered
bars, neither E3.5 nor E4 is promoted. The implementation remains default-off.

## Low-λ 750-step sweep

Run `run-echo-low-lambda-750-sweep-db9a` reused the identical E3 binary, seed/data
order, and baseline through step 750 (`NLL=7.4570`). ECHO activated at step 250; early
means cover steps 333/416/499 and late means cover 582/665/748.

| λ | NLL@750 | ΔNLL | logged max-‖g‖ ratio | activation early→late | PA early→late | mean pmax early→late |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 7.4905 | +0.0335 | 1.253× | .1635→.2157 | .03287→.04086 | .1908→.1880 |
| 0.03 | 7.4378 | −0.0192 | 1.190× | .1470→.2212 | .02723→.04310 | .1814→.1939 |
| 0.05 | 7.4767 | +0.0197 | 1.810× | .1610→.2187 | .03071→.04282 | .1860→.1942 |

**Sweep verdict: FAIL / no-go.** λ=0.03 improved short-run NLL and λ=0.05 narrowly
passed the `+0.02` no-harm bound, but every arm exceeded the `1.10×` gradient bar and
none met the pre-registered copy-pressure weakening criterion: activation and PA rose
in every arm. Thus no tested hard-hinge coefficient weakened measured copy pressure
without a gradient/perplexity tradeoff. All arms completed without numerical anomalies,
and all three final checkpoints saved successfully under
`/media/robert/AI1/chiron_sweeps/echo_lambda_750_20260715/`.
