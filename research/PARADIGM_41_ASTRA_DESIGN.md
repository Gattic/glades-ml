# Paradigm shift candidate #41 — ASTRA: Stateless Adaptive Optimizer

**Status:** Design + Gate-0 in progress
**Date:** 2026-04-29 (post-iter-180)

## Premise

Adam's persistent EMA state `(m, v)` is the structural cause of long-horizon bf16
training failure (surprises #15-17), not a numerical accident. Per-step bf16
truncation in `v` integrates into linear-in-T trajectory bias. Kahan-v (iter 171)
patches this at +50% optimizer-VRAM cost.

**ASTRA replaces persistent EMA with within-step microbatch variance**: at step
*t* with *m* microbatches producing `g^{(1)}, …, g^{(m)}`, define

  `m_t := (1/m) Σ g^{(i)}`        (already computed by gradient accumulation)
  `v_t := (1/m) Σ (g^{(i)})²`     (one extra accumulator pass)
  `x_{t+1} := x_t − γ_t · m_t / (√v_t + ε)`

`v_t` is unbiased for `E[g²] = ḡ² + σ²` — same target as Adam's EMA — but uses
*m* in-step samples rather than ~1/(1−β₂) cross-step samples. Persistent
optimizer state: zero. Bf16 truncation does not propagate beyond step *t*.

## Memory at 1.84B (vs current Adam-bf16+Kahan)

|                          | Adam-bf16+Kahan | ASTRA |
|--------------------------|-----------------|-------|
| `m, v, c` persistent     | ~8.4 GB         | 0 GB  |
| `m_t, v_t` ephemeral     | 0               | folded into existing accumulation |
| **Optimizer subtotal**   | **~8.4 GB**     | **0 GB** |

Optimizer state goes to literal zero. Same 16 GB ceiling now hosts ~3B params.

## Statistical comparison (corrected for actual β₂=0.95)

For Gaussian gradients: `Var[v^{ASTRA}] / Var[v^{Adam}] = 2/[m(1−β₂)]`.
- β₂=0.999, m=8 → 250×
- β₂=0.95, m=8 → 5×       ← actual flagship setting
- β₂=0.95, m=16 → 2.5×

Multiplicative update-magnitude noise: ~`1/(2√m)` for ASTRA. At m=8: ~18 %.
Adam-bf16 update bias from EMA truncation compounds *linearly* in *T*; ASTRA's
update noise is bounded and *independent* of *T*. Crossover step
`T* := 2/[m·ε_bf16²·(1−β₂)] ≈ 800` at β₂=0.95, m=8 — i.e., for any run longer
than 800 steps, ASTRA is theoretically preferred.

## Composition

- **FACE (#28)** — ASTRA's `v_t` is computed in the FACE-preconditioned subspace.
  FACE's own row/col EMAs remain (separate β, separate object).
- **MFIO** — ASTRA `m_t, v_t` computed in the projected low-rank subspace.
- **WIP (`--wip-K 4`)** — supplanted on Wq/Wk/Wv (was extending the same v ASTRA replaces).
- **SLC / RLG / SAS** — orthogonal curricula, unaffected.

## Gate-0 (premise test)

Does Adam-without-v-state converge at all? Test at small scale before
investing in true *m*≥4 ASTRA refactor:

1. Add `--astra` flag → bypass Adam, use `v_t = g_t²` (single-microbatch limit).
2. Keep momentum `m_t = β₁ m_{t-1} + (1−β₁) g_t` (matches Lion / RMSprop-with-mom).
3. Run flagship `66M × 2500` baseline and `66M × 2500 --astra`.
4. **Pass** if ASTRA final EMA within 0.10 nat of baseline.

Pass → engineer true *m*≥4 ASTRA via backward-refactor (writes per-microbatch grads
to a tmp scratch, then a separate kernel updates `gradAccum += tmp; gradSqAccum += tmp²`).

Fail → multi-microbatch is essential to the framework; refactor unavoidable.

## Gate-0 result (2026-05-08)

**FAIL at production lr; kernel form sound at low lr.**  Both runs at 66M × 2500
(default flagship recipe: SLC 256→512→1024, RLG 4→6→12, FACE on, SAS α=0.1):

| Run | lr | Final EMA | Best loss | step-250 ‖g‖ | step-500 ‖g‖ | wall |
|-----|----|---------:|----------:|--------------:|--------------:|-----:|
| Adam baseline | 3e-4 | **9.2242** | 6.6509@1287 | 0.566 | 1.568 | 17.8 s |
| ASTRA m=1     | 3e-4 | **27.6310** | 10.0197@36 | 24.500 | 988,556 | 19.3 s |
| ASTRA m=1     | 3e-5 | 9.9095 | 8.3683@1287 | 0.567 | 1.567 | 19.3 s |

ΔEMA(baseline → ASTRA m=1, lr=3e-4) = **+18.41 nat** — 184× over the 0.10 nat pass bar.

**Diagnosis.**  At m=1, the kernel computes `Δ = -lr · m̂ / (|g| + ε)` per element.
For parameters where `|g[i]| ≈ 1e-4` (common in transformer matrices, especially
post-warmup), the per-element update magnitude becomes ~10⁴× the intended `lr`.
The global gradient-clip can't catch this — it operates on `‖g‖₂` after the
update, not before per-element scaling.  By step 250 the explosion has already
pinned `gradScale ≈ 0` and the model freezes at the FP32 cross-entropy ceiling
(`ln(32000) ≈ 10.37` shifted by token-occurrence floor).

The lr=3e-5 control confirms the kernel form is mathematically sound: at 10×
lower lr the per-element explosion is gated and ASTRA m=1 trains stably,
finishing 0.69 nat behind baseline.  This places the failure mechanism on
*update magnitude scaling*, not on the kernel logic.

**Decision (per design doc):** "Fail → multi-microbatch is essential to the
framework; refactor unavoidable."  The proper test is `(1/m)Σg²` at m≥4, which
reduces v-estimator variance by m× (per design-doc Var-ratio table:
`2/[m(1-β₂)]` → at m=4, β₂=0.95, ASTRA's v variance is 10× Adam's, vs 40× at m=1).
Pursuing the backward refactor is the only way to test the actual paradigm-#41
premise.

**Open: deployment-config caveat.**  `run.sh` never passes `--accum`; all
flagship runs (66M through 1.84B) use `accumSteps=1`.  Even after the refactor,
ASTRA can only run at m≥4 if `--accum 4` is set, which 4× the effective batch
(currently 1024 tokens → 4096).  Whether to change the deployed configuration
is a separate decision from whether ASTRA's kernel works.

**Open: VRAM accounting revision.**  The original "0 GB optimizer subtotal"
column under-counts the new persistent state ASTRA actually needs.  Honest
accounting at 1.84B (assuming bf16 grad accumulators):

| Buffer | Adam-bf16+Kahan | ASTRA m=1 | ASTRA m≥4 (refactor) |
|--------|----------------:|----------:|---------------------:|
| grad accumulator (bf16) | 3.7 GB    | 3.7 GB    | 3.7 GB (unchanged) |
| Adam m (bf16)           | 3.7 GB    | —         | —                  |
| Adam v (bf16)           | 3.7 GB    | —         | —                  |
| Kahan c (bf16)          | 3.7 GB    | —         | —                  |
| ASTRA m (FP32, no Kahan needed) | — | 7.4 GB    | 7.4 GB             |
| **gradSq accumulator (NEW, FP32)** | — | 0 (fused in kernel) | **7.4 GB** |
| **Optimizer subtotal**  | **14.8 GB** | **7.4 GB** | **14.8 GB** |

ASTRA m=1 wins ~7 GB *if* it converged.  But ASTRA m≥4 (the variant the
refactor enables) needs a persistent gradSq buffer the same shape as gradAccum
— the design doc's "folded into existing accumulation" wording elides that the
fold target doesn't exist as a free buffer.  Net VRAM at m≥4 ≈ Adam-bf16+Kahan.

This re-prices the refactor's payoff.  ASTRA m≥4 wins on (a) eliminating the
long-horizon EMA bias mechanism (surprise #17 prevention by construction) and
(b) simpler optimizer state, but **does not free the ~8 GB the design doc
projected**.  Surprise #17 is already addressed empirically by Kahan-v
(iter 171), validated by run-9 successfully completing 650k steps at
EMA 9.23 — the first 1.84B run that did not hit the mid-phase bf16 drift
that took down run-3.  So the remaining ASTRA payoff is structural simplicity
+ the chance of a tighter stability margin — not a memory unlock, and not
fixing an open problem.

(For reference: the 18.33-nat run-10 vs run-11 delta is the validation of
surprise #18's fix — iter-182 + iter-184, post-resume LR mini-warmup +
cosine LR decay — not surprise #17's.)

## Recommendation post-Gate-0 (2026-05-08)

1. **Do not pursue the m≥4 backward refactor yet.**  The empirical fail at m=1
   plus the revised VRAM accounting reduce the ROI below the 4-8 h engineering
   cost.  Surprise #17 is already solved by Kahan-v.
2. **Keep the m=1 implementation in place** as exploratory infrastructure for
   future Lion-style or Adafactor-class research.  The `--astra` flag, kernel,
   and dispatch path cost essentially zero VRAM and zero baseline impact.
3. **If a future paradigm needs the within-step variance estimator** (e.g., a
   stateless preconditioner stacking on FACE), revisit the refactor with the
   refreshed accounting.

## Validation ladder

- Gate-0: 66M × 2500 (~30s)
- Gate-1: 100M × 5000 (~5 min) convergence parity
- Gate-2: 500M × 5000 (~1 h) memory accounting
- Gate-3: 1.84B × 50k (~3 h) first long-horizon
- Gate-4 (long-horizon stability per surprise #17): 1.84B × 100k no-transitions, EMA ≤ min+1 nat

## Files

- Design: this document
- Implementation: `glades-trainer/trainer/chiron_main.cpp` `--astra` flag + `astra_update_state` kernel in `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`
- GPU parity test: `unit-tests/Backend/Machine Learning/cuda/test_astra_update.cpp` (CPU reference vs GPU)
