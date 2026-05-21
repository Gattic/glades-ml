# Binary-FFN gradient stability at flagship 165M+ — investigation note

**Date:** 2026-05-09
**Status:** observed, not yet root-caused

## Observation

The first flagship long-run (commit `cd084bb69`) at d=1024 / L=16 / heads=8 /
dff=2816 (~165M params) with `--binary-ffn --mla-dc 128 --attn-sinks 4
--local-attn 256 --lr 1e-3` (no warmup, no grad-clip) **diverged**:

| Step | NLL | grad_norm |
|------|-----|-----------|
| 1    | 10.5829 (init) | n/a |
| 100  | 10.62          | (no log) |
| 350  | 10.6323        | 5.6e+9 (last non-NaN) |
| 365  | nan            | NaN |

NLL was monotonically *rising* from step 1 onward — i.e., the model wasn't
just converging slowly, it was actively diverging from random init. Grad
norms in the 10⁸-10¹⁰ range, then NaN.

The retry with `--lr 3e-4 --grad-clip 1.0 --warmup-steps 200` did not
NaN, but landed in a degenerate regime: grad norms stayed at 10⁸-10¹⁰
and grad-clip continuously scaled them to ~1e-9, making effective
parameter updates ≈ 0. NLL trajectory: 10.58 → 10.50 → plateau at 10.50
for the remaining 30 minutes of training.

A separate smoke test on the GPU-init binary (curand-based init, same
seed) at the same hyperparameters showed grad norms in the healthy
0.4-0.8 range and NLL stable at 10.57 over 25 sequences. Same
hyperparameters, different RNG path.

## Hypothesis

Most likely culprit: **the binary FFN STE backward at deep layers**.
The flagship's `--binary-ffn` keeps a float master copy of W1, W2 and
applies a straight-through estimator on the backward pass (gradient is
copied through unchanged from the binary forward). At init, the master
weights are Glorot-uniform (~ ±0.05 at d=1024), and the binary forward
maps them to {-1, +1} unconditionally. The activation magnitudes
through the FFN are O(±1) instead of O(±0.05), so the backward gradient
into W1, W2 is O(20×) larger than the conventional dense-FFN backward
would be at the same init.

This 20× per-layer gradient amplification compounds across L=16 layers
in the residual backward path. Empirically, the rare seeds that don't
diverge are ones where the initial binary forward happens to produce
near-cancelling activation patterns (giving sub-O(1) per-layer
magnitudes); the more common seeds produce the runaway grads observed.

Three pieces of evidence pointing the same way:

1. **No divergence at d=768 / L=8** (the validated flagship config from
   `MLA_PERMANENT_FIX_VALIDATION.md`). At smaller L, the layer-product
   amplification is bounded.
2. **Divergence is seed-dependent at d=1024 / L=16** (CPU-init seed
   diverged; GPU-init seed didn't, same hyperparameters).
3. **The grad-clip-to-ε regime in the v2 retry** is exactly what would
   happen if the gradient direction is a near-NaN spike: the clip
   normalization preserves direction but kills magnitude, so the model
   takes a tiny step in a noisy direction every step and never
   accumulates progress.

## Quick path-of-least-resistance fixes (if revisited)

1. **Scale binary FFN forward by sqrt(2/d_model)** in the alpha factor
   to compensate for the unit-magnitude activation amplification. Same
   correction Bitnet uses for QAT. Effectively: `forward = α · sign(W) ·
   x` with `α = abs(W).mean()` per output channel. Dampens activation
   magnitudes back to dense-FFN-equivalent scale.
2. **Apply STE backward through `tanh(W * τ)` with τ → ∞** instead of
   raw straight-through. Bounds the gradient magnitude at a level
   matching the dense FFN at small W. (Standard QAT trick.)
3. **Layer-wise gradient clipping**: clip per-tensor instead of global,
   with the per-tensor cap = 1.0. Decouples the FFN gradient explosion
   from the rest. Doesn't fix the root cause but lets training proceed
   while the actual fix is built.
4. **Slow curriculum into binary mode**: train dense FP32 FFN for the
   first ~5% of training, then switch to binary. Lets the master
   weights settle into a regime where binary→{-1,+1} is a small
   perturbation. Already a known stabilization technique for
   BitNet-style training.

The fixes compose — try (1)+(3) first as the lowest-risk pairing.

## How to reproduce

```bash
# the diverging case (seed 2026, CPU init):
./build/glades_pile_train --gpu --mp \
    --dmodel 1024 --layers 16 --heads 8 --dff 2816 \
    --seq-len 4096 --tbptt 4096 --max-tokens 1000000 \
    --attn-sinks 4 --local-attn 256 \
    --mla-dc 128 --binary-ffn \
    --lr 1e-3 --weight-decay 0.0 \
    --model-name binffn_diverge --no-auto-resume
# expect: NLL rises monotonically from 10.58, NaN around step 350-400
```

Drop `--binary-ffn` and the same config converges normally — confirming
the binary FFN is the unstable component, not MLA + sinks + local-attn.

## Why this is a small-blocker

The `flagship_x500m` probe (d=1536/L=24/dff=4096) hit a separate
**training-step scratch buffer OOM** at 3.2 GB before any gradient
explosion could show up. So at the current 16 GB GPU + this paradigm
stack, the OOM ceiling binds before the binary-FFN stability ceiling
does. Once the scratch issue is resolved (reuse, gradient
checkpointing, or reduced T), this stability investigation becomes
the next blocker.

For now: any flagship long run targeting the binary-FFN stack should
include `--grad-clip 1.0 --warmup-steps 200 --lr 3e-4` and accept the
slow convergence, OR drop `--binary-ffn` and pay the FFN GPU memory
cost.
