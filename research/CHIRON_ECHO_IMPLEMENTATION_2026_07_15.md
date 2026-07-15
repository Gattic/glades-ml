# CHIRON ECHO implementation record — 2026-07-15

## Scope and verdict

ECHO (Excess-Copy Hinged Objective) from
`docs/superpowers/specs/2026-07-09-chiron-loss-regularizers-design.md` is implemented
through engineering gates **E0–E2**. The implementation is default-off, training-only,
parameter-free, and adds no checkpoint or serving state.

**Engineering verdict: PASS.** Empirical E3 (2500-step), E3.5 generation, and E4 (30k)
gates were not run; this record makes no quality, generation, wall-overhead, or ship claim.
BERM and LOFT remain portfolio designs and were not selected for implementation.

## Implementation

### glades-ml

- `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu/.h`
  - Hard-hinge `echo_repeat_stats` and sparse scatter.
  - Optional Huberized hinge with
    `H_delta(x)=x^2/(2 delta)` inside the knee and `x-delta/2` above it.
  - Exact weighted field `pi_u (b_u - sum_v pi_v b_v)` via dense row scalar plus
    weighted sparse scatter.
  - Per-row maximum active-copy probability for telemetry.
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
    a five-bin max-copy histogram, and ECHO loss.
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

## Remaining empirical gates

- E3 matched 2500-step baseline/treatment, including production wall overhead.
- E3.5 paired free-generation metrics.
- E4 matched 30k run and wide-32 ship metric.

Those require a separately pre-registered experiment arc and are not prerequisites for
calling the code implementation complete.
