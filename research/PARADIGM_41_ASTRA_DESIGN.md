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
