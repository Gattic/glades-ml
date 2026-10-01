# Design: q-side gradient clamp before embedding scatter (SIRA stability mitigation)

**Date:** 2026-06-11
**Status:** approved to implement (user-directed 2026-06-11)
**Baseline:** CHIRON 1B regstack Phase 2 flagship (val NLL 3.5734 @ 30k, ~28,072 tok/s)
**Motivating evidence:** `research/SIRA_TERMINAL_30K_RESULT_2026_05_27.md` —
seed-4242 / seed-2024 late bad-gradient guard skips. The traced overflow path:

```text
normal readout dq → early-layer q-side reverse amplification
                  → huge dq_0
                  → embedding_scatter_add writes huge dE
                  → dE + L00/L01 dgamma/dbeta dominate grad-detail
                  → guard skips Adam on overflow/non-finite global norm
```

This design implements the first-listed mitigation candidate from that doc:
"an explicit q-side gradient clamp before embedding scatter".

## 1. Goal and non-goals

**Goal:** bound the per-token gradient rows of `dq_0` immediately before
`embedding_scatter_add` (`trainer/chiron_main.cpp:14098` in glades-trainer:
`dE[token_t] += dq_0[t,:]`), so a small number of exploded rows cannot
contaminate `dE` and overflow the global gradient norm, while leaving healthy
rows **bit-identical**.

**Non-goals:**
- Not a fix for the upstream q-side amplification itself. L00/L01 ReLN
  dgamma/dbeta can still overflow the global norm on their own; whether
  dE-side clamping suffices is exactly what the seed-4242 re-run will decide.
  Per-layer dq clamping is the documented escalation if this is insufficient.
- No change to the production objective or defaults. Flag default-off;
  disabled mode is bit-identical (no kernel launched).
- No policy decision on the threshold τ here — τ is a run-time flag chosen
  per experiment (mechanism/policy separation, matching SIRA conventions).

## 2. Approaches considered

1. **Per-row RMS clamp (chosen).** For each row `t` of `dq_0 ∈ R^{T×m}`:
   if the row contains a non-finite value, zero the row; else if
   `rms(row) > τ`, scale the row by `τ/rms`; else leave the row untouched
   (no write). Targets exactly the observed failure (specific outlier
   tokens/positions, e.g. token 50 / token 265 at qsig ≈ 9–13 vs healthy
   ~1), preserves gradient direction, and is identity on healthy steps.
2. Element-wise value clamp. Simpler kernel, but distorts row direction and
   the threshold has no scale-free interpretation across runs.
3. Global-norm clamp on dq_0. Redundant with the existing `--grad-clip`
   global scale, and does nothing when the norm has already overflowed to
   non-finite — which is the observed failure mode.

Per-row zeroing of non-finite rows is strictly less destructive than the
current behavior (the guard discards the entire step); it converts
"skip everything" into "drop the poisoned rows' dE contribution, keep the
rest of the step" — when finite-but-huge, even that is avoided via rescale.

## 3. Library change (glades-ml)

New generic kernel in `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`
(+ declaration and no-CUDA stub in `gpu_kernels.h`, next to
`embedding_scatter_add`):

```cpp
// Per-row RMS clamp with non-finite sanitization, in place on x [rows×cols].
//  - any non-finite element  → entire row zeroed, ++*d_nonfiniteCount
//  - else row RMS > tauRms   → row scaled by tauRms/rms, ++*d_clampedCount
//  - else                    → row untouched (no write; bit-identical)
// Row sum-of-squares accumulates in double so huge-but-finite rows
// (|x| ~ 1e20+) rescale correctly instead of overflowing FP32 to inf.
// Deterministic: one block per row, fixed-order tree reduction.
// Count pointers may be NULL. Returns false if tauRms <= 0 or args invalid.
bool row_rms_clamp(float* x, int rows, int cols, float tauRms,
                   int* d_clampedCount, int* d_nonfiniteCount);
```

Launch config: one 256-thread block per row, grid = rows (T=16384 at
production shape), on `computeStream()`. Counters via `atomicAdd` (count
values only — order-independent, deterministic).

## 4. Trainer change (glades-trainer)

- `Config`: `float dqEmbedClampTau;` default `0.0f` (off).
- Flags: `--dq-embed-clamp F` (parse_f32), `--dq-embed-clamp-smoke`
  (prints parsed value + kernel availability, exits; mirrors
  `--sira-config-smoke` minimalism).
- `Scratch`: `GpuBuffer<int> dqClampCounts` (2 ints), allocated only when
  the flag is on.
- Call site (`chiron_backward`, immediately before `embedding_scatter_add`,
  after the iter-111 final dq sync): zero counters → `row_rms_clamp(s.dq,
  T, m, τ, …)` → download 2 ints → when either is nonzero, log
  `[dq-clamp step N] clamped=X zeroed=Y tau=F` (warn level, matching
  `[grad-skip]` style). The pre/post norm traces and mp-drift traces around
  the scatter are left untouched, so existing tooling shows before/after.
- `run.sh` usage text: one line under the SIRA flag block.

Overhead when enabled: one T×m read (~134 MB ≈ 0.2–0.3% of a ~580 ms step)
plus an 8-byte D2H per step. Zero when disabled.

## 5. Testing

New unit tests in `unit-tests/Backend/Machine Learning/chiron-test.cpp`
(CPU reference computed inline; CUDA-skip pattern as in
`CHIRONGpuParityTest`):

- `CHIRONQClampMathTest` — mixed rows: healthy (bit-exact untouched —
  compared as exact float equality), over-τ (scaled, rtol 1e-6 vs CPU
  reference), NaN/Inf rows (zeroed), huge-finite row `~1e20` (scaled, not
  zeroed — exercises the double accumulator), counter values exact.
- `CHIRONQClampEdgeTest` — `tauRms <= 0` rejected, NULL counters accepted,
  all-rows-clamped case, single-row/single-col shapes.

Selectors: new `chiron-qclamp` / `qclamp` in `unit-tests/main.cpp`; both
tests also appended to the `chiron-sira` umbrella group (this is part of the
SIRA stability arc).

Trainer verification: build + `--dq-embed-clamp-smoke`, plus
`git diff --check` in both repos.

## 6. Promotion path (out of scope here)

Re-run the SIRA candidate recipe at seed 4242 with the clamp enabled (τ
chosen from healthy-run dq trace stats; generous headroom, e.g. 10–100×
healthy row RMS, documented in the run doc). Success = no guard skips and
NLL within the pre-registered SIRA gates; the clamp then becomes part of the
candidate recipe (still default-off in production until the full six-point
SIRA promotion bar is met).

## 7. Per-layer escalation (2026-06-11, post-gate amendment)

The seed-2024 30k gate (run 2026-06-11) showed the embed-site clamp working
as designed (dE clean at norm 386) while **L00/L01 ReLN dgamma/dbeta
overflowed independently** — the q-side reverse amplification produces huge
intermediate dq inside the layer backward, upstream of the embedding
scatter.  Escalation per §2's design intent:

- **`--dq-layer-clamp F`** (`cfg.dqLayerClampTau`, 0 = off): apply
  `row_rms_clamp` to the incoming dq (`dq_in_ptr`, alternation-aware) at the
  **top of every backward layer iteration**, immediately after the
  iter-113 dq_in/dq_out pointer selection and before the LayerDrop
  pass-through.  This is the tensor `chiron_reln_backward` consumes, so it
  bounds every layer's dgamma/dbeta as well as the cascade itself.
- Counters: separate `dqLayerClampCounts` [2] buffer, zeroed once per step
  before the loop, accumulated across all L clamp calls (atomicAdd), one
  `[dq-layer-clamp]` log line per firing step (aggregate, not per-layer;
  per-layer detail remains available via `--sira-layer-grad-trace`).
- Same τ for all layers: healthy incoming dq *grows* toward layer 0
  (after-layer-23 ≈ 0.011 → after-layer-00 ≈ 0.87 global norm), so a τ
  sized for dq_0 has even more headroom at depth.  τ = 1.0 recommended.
- Both clamp flags added to the CUDA-graphs auto-disable list (per-step
  counter downloads are capture-incompatible).
- Overhead when enabled: L extra T×m reads/step (~3.2 GB ≈ ~1% wall at the
  flagship shape) + one 8-byte D2H per step.  Zero when disabled.
- Gate: 300-step A/B at production recipe (no-clamp vs both clamps) must be
  print-identical in the healthy phase with zero firings, then a seed-2024
  30k re-run with both clamps decides the stability gate.
