# CHIRON 1B LayerDrop Arc — 5k Pilot FAIL (2026-05-23)

**Status:** PILOT FAIL. Arc closed per spec §3.2 OUTRIGHT FAIL rule.
**Spec:** `docs/superpowers/specs/2026-05-23-chiron-1b-layerdrop-design.md`
**Plan:** `docs/superpowers/plans/2026-05-23-chiron-1b-layerdrop.md`
**Disposition:** LayerDrop code stays in library + trainer as opt-in flag
(`--layer-drop-pmax`, default 0.0 = bit-identical to regstack Phase 2 ship).
Production flagship remains **regstack Phase 2** (`chiron_1B_T16384_regstack_phase2.final`,
val NLL 3.5734 @ 30k, 28,072 tok/s). No CLAUDE.md update.

---

## Headline result

| Run | Config | Val NLL @ 5k | tok/s | Wall | Peak VRAM |
|---|---|---:|---:|---:|---:|
| L0 | regstack Phase 2 (zloss + qknorm) | 3.9400 | 28,079 | 48.7 min | 15.01 GB |
| L1-buggy | L0 + `--layer-drop-pmax 0.1` (val-mode bug) | 4.1197 | 29,278 | 46.7 min | ~15.01 GB |
| **L1-fixed** | **L0 + `--layer-drop-pmax 0.1` (val-mode fix in)** | **4.0131** | **29,230** | **46.7 min** | **~15.01 GB** |

**ΔNLL (L1-fixed − L0) = +0.0731 nat.** Spec §3.2 OUTRIGHT FAIL bar = +0.05 nat. Decision: **CLOSE ARC.** No 30k retrain.

**Δtok/s = +4.10% wall improvement** — real and consistent with the hard-drop semantics' compute saving (skipped layers contribute zero kernel time). But the wall win does not offset the NLL regression at the pilot bar.

---

## Trajectory comparison

Val NLL at every val checkpoint (L0 = baseline; L1-fixed = LayerDrop):

| Step | L0 NLL | L1-fixed NLL | Δ |
|---:|---:|---:|---:|
| 1 (warmup) | 10.7430 | 10.7430 | 0.0000 (bit-identical at p_max=0 path; val now skips LayerDrop) |
| ~500 | 7.3554 | 7.5804 | +0.225 |
| ~1000 | 5.8635 | 6.1363 | +0.273 |
| ~1500 | 5.0652 | 5.2583 | +0.193 |
| ~2000 | 4.7574 | 4.9140 | +0.157 |
| ~2500 | 4.4370 | 4.6105 | +0.174 |
| ~3000 | 4.3355 | 4.4111 | +0.076 |
| ~3500 | 4.5671 | 4.6005 | +0.033 |
| ~4000 | 4.1126 | 4.2017 | +0.089 |
| ~4500 | 4.0844 | 4.1251 | +0.041 |
| **5000 (final)** | **3.9400** | **4.0131** | **+0.0731** |

The gap narrows over training (from +0.27 at step 1k to +0.07 at 5k), consistent with the
hypothesis that LayerDrop's regularization signal is data-efficiency–driven and may
only fully emerge at longer horizons. But at the 5k pilot bar, the gap is still
+0.073 — above the +0.05 OUTRIGHT FAIL threshold.

## Position-stratified eval @ step 5000

| Bucket | L0 NLL | L1-fixed NLL | Δ |
|---|---:|---:|---:|
| [0, 2k) | 3.81 | 3.91 | +0.10 |
| [2k, 4k) | 3.84 | 3.96 | +0.12 |
| [4k, 6k) | 3.92 | 4.02 | +0.10 |
| [6k, 8k) | 3.93 | 3.98 | +0.05 |
| [8k, 10k) | 3.94 | 4.01 | +0.07 |
| [10k, 12k) | 4.00 | 4.05 | +0.05 |
| [12k, 14k) | 4.07 | 4.11 | +0.04 |
| [14k, 16k] | 4.02 | 4.07 | +0.05 |

Regression is **relatively uniform** across position buckets (+0.04 to +0.12 nat),
mildly worse at shallow positions. This rules out a "LayerDrop only hurts at deep
positions" hypothesis — the issue is on the whole-stack distribution.

## Training-loss stability

L0 worst transient: step 3155 loss=5.21, ||g||=2.597 (then recovered).

L1-fixed had **larger and more frequent** transients:
- step 831: loss=13.30, ||g||=2.917 (model briefly diverged then recovered)
- step 2989: loss=11.40, ||g||=1.479
- step 3487: loss=9.64, ||g||=1.195
- step 3653: loss=9.55, ||g||=2.773

The model recovered after each spike, but the trajectory is visibly more volatile than
L0. Consistent with hard-drop semantics making the optimizer state less well-conditioned
(EMA on Adam moments has to handle layer-magnitude shifts when a deep layer drops in/out).

---

## The val-mode bug (R-LD-7), discovered mid-arc

The initial L1 run produced ΔNLL = +0.1797 nat with extreme val-NLL bouncing (e.g.,
step 2500 val NLL = 6.4835 vs L0 4.4370 — a +2.05 nat spike that partially recovered).
Investigation found:

**Bug** (`trainer/chiron_main.cpp:8704`, commit `ef9d540`): the LayerDrop gate
`if (cfg.layerDropPMax > 0.0f && !s.layerDropKept.empty())` had no train/eval check.
The same `forward()` function is called by:
1. The training loop (LayerDrop should be active).
2. `run_validation()` for held-out val NLL (LayerDrop should be SKIPPED).

The bug caused val passes to drop random layers — each val NLL was measuring a random
subnetwork's output, not the model's. Also, `s.layerDropRng` state was advanced by val
draws, shifting the training mask sequence after every val checkpoint.

**Fix** (commit `5d1aa1c`): added `bool isTraining = true` parameter to `forward()`;
7 call sites updated. `run_validation`, `run_nita_eval`, LAMBADA scoring all pass
`isTraining=false`. Training loop's call and ORION HVP's coupled fwd+bwd stay
`isTraining=true`. Smoke-verified: step-1 val NLL is now bit-identical between
`--layer-drop-pmax 0.0` and `0.1` (both 10.7430).

After the fix, the L1 re-run gave ΔNLL = +0.0731 nat — **still above the +0.05 FAIL bar**,
but materially smaller than the buggy run's +0.18 nat. The bug's noise contribution was
real but not the dominant cause; the underlying mechanism is genuinely
regressive at this scale.

---

## Three plausible explanations for the FAIL

1. **Hard-drop semantics too aggressive for CHIRON's symplectic update.** The trainer
   uses `(p, q) → (p + shear(q), reln(q))`. At `mask_l = 0`, both `shear` and `reln`
   are skipped — `p` and `q` pass through unchanged. The standard Fan 2019 / Huang 2016
   inverted-dropout `1/(1-p_l)` compensation is intentionally absent (it has no clean
   interpretation on `q = reln(q)`, a non-linear transformation). This introduces a
   train/inference distribution mismatch (~5% on the layer-stack contribution at mean
   `p̄ = 0.05`) that the model has to overcome — at 5k single-seed, it doesn't.

2. **5k pilot too early for the regularization benefit.** LayerDrop's win is data
   efficiency, which typically materializes at longer training horizons (30k+ steps).
   The narrowing gap from +0.27 nat at step 1k to +0.073 nat at step 5k is consistent
   with this hypothesis. Spec §3.2 still requires the +0.02 nat 5k pilot bar
   (or +0.05 max regression) and we did not clear it — per Phase-3 P6, we don't
   silently commit 4.5 hours of L2 GPU time on the "maybe at 30k" rationalization.

3. **CHIRON's symplectic structure intolerant of layer skips.** The
   `p`-stream is an additive momentum accumulator over depth; the `q`-stream is
   transformed by `reln` at each layer. Dropping a (shear, reln) pair shears the
   trajectory in a way that's hard to recover from in the optimizer state. The
   visibly more volatile training trajectory (4 spikes of magnitude >9 vs L0's 1
   spike of ~5) supports this.

These are not mutually exclusive. All three contribute. (1) is the main mechanical
cause; (3) explains the increased gradient instability; (2) is the most charitable
interpretation but can't be tested without paying the 4.5h L2 compute.

---

## What stays in the codebase

**Library (`glades-ml`, branch `chiron2`):**
- 9 LayerDrop wiring commits (`b75d21f69` through `64ca836da`).
- 1 spec addendum (`473ad561f`).
- All math verified bit-identical at `layerDropPMax = 0.0f`.
- 3 unit tests pass (`CHIRONLayerDropScheduleMathTest`, `CHIRONLayerDropDisabledParityTest`, `CHIRONLayerDropDeterministicMasksTest`).
- Library uses inverted-dropout per the original spec. Available to any future
  standard-residual transformer code path.

**Trainer (`glades-trainer`):**
- `224f3bf` adds `--layer-drop-pmax` flag.
- `ef9d540` wires hard-drop LayerDrop into `chiron_main.cpp` production training path.
- `5d1aa1c` fixes the val-mode bug (forward `isTraining` parameter).

The trainer can still run `--layer-drop-pmax X` at any p_max in [0, 1) for future
research (e.g., a smaller-p sweep, or a different mechanism that benefits from depth
noise). Default `0.0f` = bit-identical to the regstack Phase 2 ship.

**Production flagship: unchanged.** Still `chiron_1B_T16384_regstack_phase2.final`
(val NLL 3.5734 @ 30k, 28,072 tok/s, ~14.97 GB peak VRAM).

---

## Reproduce commands

```bash
# L0 baseline (regstack Phase 2 recipe, 5k single-seed):
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm \
    --steps 5000 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_l0_5k

# L1-fixed (regstack + LayerDrop p_max=0.1, with val-mode fix in):
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1 \
    --steps 5000 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_l1_layerdrop_5k_fixed
```

Logs from the actual runs are in `~/dev/glades-trainer/logs/l0_5k_*.log` and
`logs/l1_5k_fixed_*.log`. Checkpoints at the `--save` paths.

---

## Honest publication per Phase-3 P6

Negative results are first-class outputs of the Phase-3 program. This doc closes
the LayerDrop investigation cleanly: the mechanism was specced, implemented, gate-tested,
and rejected by the pilot bar. The spec's pre-registered §3.2 OUTRIGHT FAIL threshold
(+0.05 nat) was the decision rule; we honor it.

Next mechanism: **UL2 mixture-of-denoisers** (the second arc from the original
brainstorming session). Brainstorm fresh in its own session per the scope-decomposition
rule. Or another candidate the user prefers.
