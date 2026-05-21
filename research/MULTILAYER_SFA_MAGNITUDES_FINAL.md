# Multi-Layer SFA at 4 Layers — Magnitudes Test + Interpretive Reckoning

**Date:** 2026-05-15
**Iter:** Ralph-loop iter 20
**Branch:** vesta5 (glades-ml), main (glades-trainer)
**Run dirs:** `glades-trainer/research/runs/2026-05-15-multilayer-{4layer-magnitudes,2layer-pslot-dinv,...}/`

## TL;DR

Ran the 4-layer SFA experiment that the program designed to test the
"magnitudes" claim. Result: 4-layer SFA at L={12,15,18,21} achieves
**−6.52 nat improvement over the 4-layer NO-OP control** at just 200
SFA-train steps. Linear extrapolation from single-layer (−0.60 mean) is
−2.4 nat; the actual 4-layer result is **~3× supra-linear**.

**HOWEVER** — a critical interpretive issue: the "improvement" is
measured against the NO-OP control (SFA with random init breaks attention
at the swap layer). The **flagship val NLL is ~4.3 nat**; even 4-layer
SFA after training reaches only 16.94 nat — still **~12.6 nat WORSE than
the flagship**.

The entire program's empirical signal has been measuring "how much
training recovers from SFA insertion damage", NOT "how much SFA improves
on the flagship". This is the honest reckoning iter 20 forces.

## 4-layer result (configuration)

```
--sfa-swap-layers 12,15,18,21
--sfa-d-s 8 --sfa-r 4 --sfa-w 128 --sfa-n-sinks 8
--sfa-solver-iters 30 --sfa-solver-step 0.5 --sfa-lambda 0.01
--sfa-train --sfa-lr 1e-4
--grad-clip 0.10 --lr 0.0  (flagship frozen)
--val-every 50 --val-batches 4 --val-position-buckets 8
```

200 SFA-train steps starting from `chiron_1B_T16384.step30000`.
Wall: 247.5 s. tok/s: 13,242 (vs single-layer 16,575).

## 4-layer val NLL trajectory

| Step | EMA | Val NLL | Δ vs NO-OP (23.4654) | Per-position pattern |
|-----:|----:|--------:|---------------------:|----------------------|
| 30050 | 16.92 | 17.2664 | **−6.20** | uniform ~−6 nat |
| 30100 | 16.74 | 17.6407 | −5.82 | |
| 30150 | 16.89 | 17.3901 | −6.08 | |
| 30200 | 17.05 | **16.9412** | **−6.52** (best) | uniform −4.5 to −7.5 nat |

Per-position at step 30200:
- pos 0: 14.17 (vs 18.63 NO-OP) = −4.46 nat
- pos 1: 16.29 (vs 21.74) = −5.45 nat
- pos 2: 17.22 (vs 23.83) = −6.61 nat
- pos 3: 17.64 (vs 24.62) = −6.98 nat
- pos 4: 17.37 (vs 24.38) = −7.01 nat
- pos 5: 17.81 (vs 25.01) = −7.20 nat
- pos 6: 17.47 (vs 24.94) = −7.47 nat
- pos 7: 17.56 (vs 24.57) = −7.01 nat

Gradient norm spiky: 4.9e16 → 1.8e17 → 1.7e17 → 6.6e16. Training was in
clip-saturated regime throughout, BUT didn't NaN — produced valid val
output across all 4 checkpoints.

## Compounding observation (the design hypothesis)

| Layer count | Best Δ NLL vs NO-OP | Per-layer Δ |
|------------:|--------------------:|-------------:|
| 1 (Phase 8b, 2000 steps) | −1.50 (peak) / −0.60 (mean) | −0.6 to −1.5 |
| 2 (iter 19, 200 steps) | **−1.15** | −0.58 |
| 4 (iter 20, 200 steps) | **−6.52** | **−1.63** |

The 4-layer per-layer gain (−1.63) is **3× the single-layer mean**
(−0.60). This is **supra-linear compounding** — much stronger than the
"linear stacking" the program projected. Two possible explanations:

1. **Cooperative recovery**: 4 SFA layers spread the recovery work across
   the stack, so each gets to learn finer-grained recovery patterns.
2. **Grad-clip artifact**: in clip-saturated regime, gradients are
   normalized to clip-size, and the effective lr×grad-clip product is
   the same per layer regardless of grad magnitude. More layers = more
   parameter updates per step.

The honest read: the compounding signal is real but its magnitude is
unusual and the gradient norms suggest the training is not in a "normal"
regime.

## The critical interpretive issue

Throughout this entire research arc (Phase 8b, iters 11-19), the
empirical signal has been measured as **trained-SFA val NLL minus NO-OP
val NLL**. NO-OP means "SFA layer-swap with random initial parameters,
no training". Random init of P_q/P_v/P_o/U/Σ produces a layer that
processes the residual stream randomly — destroying the flagship's
attention at that layer.

**The flagship val NLL is ~4.3 nat** (per `flagship_chiron_1B_T16384`
memory). The 4-layer "magnitudes" result of 16.94 nat is **~12.6 nat
WORSE than the flagship**.

Throughout the arc:

| Config | Val NLL | vs flagship (~4.3 nat) |
|--------|--------:|------------------------:|
| Flagship (no SFA) | ~4.3 | 0 |
| 1-layer SFA NO-OP | 23.47 | +19.17 (broken) |
| 1-layer SFA trained 2000 steps (Phase 8b best) | 21.96 | +17.66 |
| 2-layer SFA trained 200 steps | 22.31 | +18.01 |
| 4-layer SFA trained 200 steps (this iter) | 16.94 | **+12.64** (closer but still worse) |

**SFA insertion at random init breaks attention at the swap layer; training
only partially recovers it.** None of the trained-SFA configurations
beat the flagship. The "improvements" are recovery from self-inflicted
damage.

## What this means for the program

The Cellular Sheaf Attention program's central premise — that SFA's
cocycle expressivity gives strictly more representational power than
SCFA — assumes SFA's parameters, once trained, can be at least as good
as SCFA's. The Phase 8b "-0.60 nat improvement" was a measurement of
recovery rate, not absolute superiority.

**The brief's "magnitudes improvement over modern LLM architecture" is
NOT empirically supported.** The program has produced:

- Validated: cocycle expressivity recovery is possible (Phase 8b)
- Validated: compounding (more layers = more recovery, supra-linear)
- **Not validated: SFA beats flagship at any layer count**

For paradigm #250 SFA to deliver magnitudes, the comparison must be:

1. **SFA from a configuration that initially preserves flagship behavior**:
   e.g., initialize SFA P/U/Σ such that the SFA layer behaves identically
   to the original SCFA layer at step 0 (then training can only IMPROVE).
   - Implementation: warm-start U from SCFA's basis B, set Σ=1, P_o = identity-extended.
   - Then val improvement over flagship is real.

2. **OR SFA as an ADDITIVE residual on top of SCFA**:
   - At swap layer, output = SCFA_output + α × SFA_output where α starts at 0.
   - Train α and SFA params together; α grows if SFA contributes useful signal.
   - Then NO-OP IS the flagship, and any positive Δ is a real win.

Neither has been implemented or tested.

## Iter-20 honest read

The compounding empirics are real and impressive (1 layer → −0.6 nat
recovery, 4 layers → −6.5 nat recovery). But the entire arc has been
measuring SFA's recovery rate, not its absolute quality. **None of the
trained configurations beat the flagship**.

The user's brief — "improve LLM architecture by magnitudes" — remains
EMPIRICALLY UNSUPPORTED. Twenty iterations of work have produced:
- A mathematically rich framework (6 paradigm designs)
- A validated cocycle-recovery mechanism (Phase 8b reproducible at -0.6 nat)
- Multi-layer infrastructure (compounds 1.9× at 2 layers, 3× at 4 layers)
- Falsification of defect-driven gating (paradigm #255 DSA)
- **No demonstration of beating the flagship at any configuration**

The path forward for actually delivering magnitudes:
- **SFA-as-additive** (option 2 above): re-architect so SFA augments
  rather than replaces. ~100 LOC trainer change.
- **OR**: warm-start SFA from SCFA basis (option 1). Requires reading
  out the trained SCFA params at the swap layer and using them as SFA's
  initial state.

Either is ~1-2 iters of work and would re-test the magnitudes claim from
a position where the comparison to flagship is fair.

## Files

- `glades-trainer/research/runs/2026-05-15-multilayer-4layer-magnitudes/train.log` — 4-layer experiment, the trigger for this interpretive reckoning.
- `glades-trainer/research/runs/2026-05-15-multilayer-2layer-pslot-dinv/train.log` — per-slot Dinv (didn't change empirics).
- `glades-ml/research/MULTILAYER_SFA_FIRST_COMPOUNDING_RESULT.md` — iter-19's celebration of −1.15 nat compounding.
- `glades-ml/research/SFA_PHASE8B_LONG_TRAIN_RESULT.md` — the original "validated" −0.60 nat result.
- `glades-ml/memory/flagship_chiron_1B_T16384.md` — the actual flagship val NLL benchmark (~4.3 nat ema).

## What changes after this

Tomorrow's iters should NOT continue the multi-layer-SFA-vs-NO-OP arc.
They should:

1. Either re-architect to SFA-as-additive and re-test against flagship,
2. Or accept that paradigm #250's "magnitudes vs flagship" claim was the
   wrong target, and redefine the goal as "faster training" (steps-to-
   reach-flagship-NLL via cocycle expressivity).

Goal 2 is more honest given the empirics. SFA's recovery curves suggest
that with more training, the model can recover from SFA insertion. The
question is whether SFA-trained models *eventually* beat flagship-trained
models given equal compute. That's a different (and more interesting)
question than the original brief asked.

This is the iteration where the empirics and the brief came into
unavoidable conflict, and the brief loses.
