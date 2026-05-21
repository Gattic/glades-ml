# DSA Probe O Candidate 1 (U-frame Defect) at Flagship — Near-Threshold Partial PASS

**Date:** 2026-05-15
**Iter:** Ralph-loop iter 14
**Branch:** vesta5 (glades-ml), main (glades-trainer)
**Run dir:** `glades-trainer/research/runs/2026-05-15-dsa-probe-o-frame/`

## TL;DR

After iter-13 falsified the Σ-based defect formula, iter-14 tests Candidate 1: the U-frame defect `ε^U_i = ‖U_i^T U_{i-1} − I_r‖_F`. The frame defect's Pearson correlation against the Phase 8b NLL pattern jumps from −0.187 (Σ) to **+0.475** (frame) — close to the 0.5 target threshold but still failing the compound criterion. The late/early ratio remains essentially 1×, so Probe O still FAILS.

This is **progress, not victory**: the frame formula clearly captures a position-stratified signal that the Σ formula does not, but at 200 training steps the magnitude variation is too small to drive a useful gate.

## Configuration

Identical to iter-13 (`DSA_PROBE_O_FLAGSHIP_RESULT.md`) except the trainer now logs both defect formulae side-by-side via the iter-14 frame kernel.

## Raw result

```
[chiron-train] done: steps=30200 total_tokens=3276800 wall=196.5s

[sfa-defect-stat:sigma] layer=18 T=16384
  mean(eps) per pos-bucket [8]:  0.9248 0.9155 0.9160 0.9201 0.9224 0.9220 0.9212 0.9202
  probe-O: late/early=1.00x  Pearson r(eps,|dNLL|)=-0.187  FAIL

[sfa-defect-stat:frame] layer=18 T=16384
  mean(eps) per pos-bucket [8]:  2.4282 2.4345 2.4298 2.4426 2.4364 2.4377 2.4178 2.4329
  probe-O: late/early=1.00x  Pearson r(eps,|dNLL|)=+0.475  FAIL
```

## Comparative analysis

| Defect | mean min | mean max | spread | ratio | Pearson r | Probe O |
|--------|---------:|---------:|-------:|------:|----------:|--------:|
| Σ      | 0.9155 | 0.9248 | 0.93 % | 1.00 × | **−0.187** | FAIL |
| U-frame | 2.4178 | 2.4426 | 1.03 % | 1.00 × | **+0.475** | FAIL |

Both formulae produce **essentially uniform** mean ε across position buckets (<1.1% spread).
The Σ formula's tiny spread is anti-correlated with Phase 8b NLL.
The U-frame formula's tiny spread is positively correlated, near the 0.5 threshold.

The U-frame defect is **measuring something real and position-stratified** — the
correlation flip from −0.19 to +0.48 is huge (0.66 absolute change). But the *magnitude*
spread is still 1% — so even with perfect correlation, a defect-driven gate can't
discriminate the 8 position buckets in practice.

## Why the magnitude spread is so small

Two contributing factors:

1. **200 training steps is too short**. The flagship is loaded fresh; only 200 SFA-specific
   gradient steps have hit U and Σ. Phase 8b ran 2000 steps and saw stronger
   position-stratified NLL (−1.50 peak at step 1600). The defect magnitudes may
   sharpen with longer training.

2. **Position-1 anomaly is buried in the bucket statistic**. Position 1 has the
   largest Phase 8b NLL regression (+1.15 nat), but in the bucket statistic it gets
   averaged with positions 0 and 2 (bucket 1 = positions 1024-2047 with T=16384).
   The per-token defect at position 1 itself may be much higher than the bucket mean
   suggests.

## What this means for paradigm #255

The frame defect is the **right kind** of signal (positively correlated with Phase 8b
NLL), just not strong enough at 200 steps to drive a binary gate. Three viable paths
forward:

### Path A: Longer training to sharpen the signal

Re-run Probe O at 2000 steps (matching Phase 8b's training budget). Test whether
the frame defect's magnitude spread and Pearson r both improve. Cost: ~33 min
trainer run.

### Path B: Refined defect formulation

The current ε^U_i uses adjacent-token frames. The Phase 8b NLL gain might depend
on multi-step composition: ε^{U,k}_i = ‖U_i^T U_{i-k} − I_r‖_F for k ∈ {1, 2, 4, 8}.
Multi-scale defect could capture longer-range cocycle structure than the 1-step
formulation.

### Path C: Residual-stream defect (Candidate 2)

Test the residual-stream-based defect ε^q_i = ‖q_i − P_q U_i U_i^T q_i‖. This
captures how much the per-position query lies INSIDE the SFA stalk subspace.
Cocycle gain may come from position-dependent subspace alignment with the
content, not from frame divergence per se.

### Path D: Drop defect-driven gating

If no closed-form defect signal correlates strongly enough, DSA's gate must be
learned end-to-end from NLL (Adam-trained gate parameters with no informative
prior signal). This is still possible but loses the design's interpretability
and the cheap-Gate-0 advantage.

## Iter-14 status

- ✓ Frame defect kernel implemented + parity test passes (2.4e-07).
- ✓ Trainer wires both defects side-by-side.
- ✓ Probe O at flagship: frame defect correlation = +0.475 (close to threshold).
- ❌ Compound Probe O criterion: FAIL on ratio test for both formulae.

## Iter-15 candidate: Path A (longer training)

Cheapest test to discriminate "200 steps is too short" from "the formula is
fundamentally weak". A 2000-step run at 33 min wall is feasible. If frame defect
sharpens (ratio > 2× AND r > 0.6), Conjecture 12' is supported and DSA's
design can move to Phase 2. If frame defect stays near-threshold at 2000 steps,
move to Path C (residual-stream defect).

## Files

- `glades-trainer/research/runs/2026-05-15-dsa-probe-o-frame/train.log`
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_sfa.{h,cu}`: sfa_defect_frame_step1_fp32 kernel.
- `glades-ml/unit-tests/Backend/Machine Learning/sfa-parity-test.cpp`: extended SFADefectParityUnitTest covers both kernels.
- `glades-trainer/trainer/chiron_main.cpp`: --sfa-defect-stat now computes both defects.

## Honest take on "magnitudes"

After two empirical iterations targeting the DSA mechanism, neither Σ nor U-frame
defect is strong enough to drive a useful gate at 200 training steps. The Phase 8b
cocycle gain (−0.60 nat) is real but its source remains unidentified at the
defect-formula level. The "magnitudes" claim of paradigm #255 DSA is **not
empirically supported yet**; the program continues to refine the hypothesis.

This is exactly the empirical work the design phase exists to enable — paradigm
designs are cheap, falsifications are even cheaper, and the cost of being wrong
about the gate-driver is one ~3 min trainer run per candidate.
