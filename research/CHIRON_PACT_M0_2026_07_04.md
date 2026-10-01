# CHIRON PACT — M0 Measurement Gate: GO (2026-07-04)

**Status:** M0 **PASS / GO** — the pre-registered measurement gate for the PACT arc
(design: `docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md`, §13).
**Kill bar was: gated cancellation-energy share < 3%. Measured: 72.0% — 24× over the bar.**
The off-diagonal cancellation lever is decisively live on the shipped flagship. Two additional
in-vivo findings materially inform the E-phases (see §4): the WhiSC q→p leak is ~0.5 relative
(not "small"), and opposing-mass occupancy rises with depth.

## 1. Setup

- **Checkpoint:** `database/checkpoints/chiron_1B_pied_e4/chiron_1B_pied_e4.final` (the PIED
  production flagship; untouched — mtime verified before/after).
- **Probe:** env-gated (`CHIRON_PACT_PROBE=1`, `CHIRON_PACT_PROBE_MAX=8`), eval-only
  (`isTraining=false`, PIED inactive ⇒ clean increments), host-side; glades-trainer commit
  `80a269e` (branch `chiron4`); zero lib changes; reads device buffers only. Startup toy
  self-test (hand-computed χ/share values) PASS.
- **Protocol:** the wideval eval-only resume (lr=0 + `--whisc-ema 1.0 --no-resume-warmup`,
  `--steps 30002`), full flagship recipe flags, `--val-every 1 --val-batches 8`, explicit
  `--load` of the flagship final, `--save` to the `wideval_scratch` throwaway. Probe captures
  the first 8 val windows (8 × 16384 tokens; ≈ 6.4B increment samples over 24 layers).
  Log: `glades-trainer/logs/pact_m0_full8_20260704_0435.log` (plus `pact_m0_smoke_20260704_0429.log`,
  `pact_m0_full_20260704_0432.log`).
- **Measured quantities** (per (t,i) depth profile, u_l = ypar_l + yperp_l downloaded at the
  commit seam before the layer's WhiSC): damped sum `A = Σ_l D_l u_l`, damped mass
  `M = Σ_l D_l|u_l|`, energy `SS = Σ_l u_l²`, `D_{l,i} = Π_{l'≥l} cos(θ_max·tanh φ_{l',i})`
  (own layer included — matches post-loop `s.p`), gate
  `χ = (M²−A²)/(M²+‖D‖²σ̂²)`, `excess = max(0, SS − A²/‖D‖²)`,
  **gated share = Σ χ·excess / Σ SS**.

## 2. Result

| seq (val window) | gated share | ungated share | A-vs-p rel err |
|---|---:|---:|---:|
| 1 | 71.99% | 93.36% | 0.504 |
| 2 | 71.41% | 93.38% | 0.481 |
| 3 | 72.25% | 93.62% | 0.490 |
| 4 | 73.51% | 94.02% | 0.533 |
| 5 | 72.40% | 93.68% | 0.522 |
| 6 | 71.28% | 93.21% | 0.496 |
| 7 | 71.91% | 93.16% | 0.494 |
| 8 | 71.45% | 93.40% | 0.518 |
| **AGGREGATE** | **72.01%** | **93.48%** | ~0.50 |

- **χ histogram** (deciles 0.0→1.0, counts over 8×T×m profiles):
  `2.04M / 4.86M / 10.0M / 17.8M / 28.5M / 42.0M / 55.7M / 61.6M / 40.6M / 5.2M` — the mass
  peaks at χ ∈ [0.6, 0.9]: sign-opposition is pervasive across (token, channel) profiles, not
  localized to a small subset.
- **Per-layer opposing-mass occupancy** (D-weighted mass opposing sign(A)):
  layer 0: 38.8%, l1: 24.2%, l2: 16.4% (minimum), l3–l7: 27–32%, l8–l15: 31–39%,
  l16–l23: **38.7–46.8%** (max at l19). Occupancy *rises with depth* — late layers spend
  nearly half their damped mass opposing the final sum.
- **Determinism:** seq 1 reproduced bit-identically across three runs
  (71.9938 / 93.3588 / 0.504) — deterministic val stream + deterministic probe.
- **Forward untouched:** val NLL with the probe active (8-batch 1.0891; 1-batch 1.0840,
  final-val 1.0075) is in the banked family for this checkpoint (4-batch final-val 1.0259,
  wide-32 windows 1.18–1.30); no NaN, no skips; probe cost ≈ 7 s/window, ~0 when env unset.

**Verdict vs the pre-registered bar: GO.** (Pre-registered prior was ~70% pass; the measured
share is far beyond any "lever not live" reading. Failure mode #2 of the design — "PIED already
mopped it up" — is refuted: even after PIED's implicit diagonal tax, the trained assembly is
massively super-minimal, with only ~6.5% of increment energy being the minimum needed to
deliver the realized damped sums.)

## 3. Interpretation for the design (honest)

1. **The lever is live at a scale that demands a conservative λ.** With χ·excess at 72% of
   total increment energy, the penalty engages nearly everywhere. This *strengthens* the case
   for the design's E2 calibration rule (field RMS 3–5% of task-dy) as the load-bearing safety
   control, and for the Huber clamp: the model must never see a large fraction of this mass as
   simultaneous gradient pressure. It also means the *observable* (occupancy) has enormous
   headroom to move at E3 (C3's ≥30% engagement bar is easily testable).
2. **The θ=0 minimal-energy reading overestimates taxable waste.** The A-vs-p residual (~0.50
   RMS) measures the WhiSC q→p route in vivo: p_L is only ~half described by the damped linear
   accumulator; the rest enters via the per-layer rotations (sinθ·q/a² terms with the a-clamp
   binding at 0.125). Some of the measured opposing mass is therefore plausibly *functional*
   (intermediate p_l values read by WhiSC before later cancellation) — exactly the design's
   ranked-#1 failure mode. Consequence: expect the *effective* safely-taxable share to be far
   below 72%; the E3 val gap (not M0's magnitude) remains the arbiter, as pre-registered.
3. **Depth profile of opposition matches the "late corrections" picture.** Rising occupancy
   with depth (l16–23 at 39–47%) is what late-layer corrective increments look like — the
   pattern the rejected Form B gradient would have deleted and PACT's A-preserving field
   deliberately spares. It also suggests a depth-graded λ as a *documented fallback* (not the
   prototype) if E3 shows early-layer capacity drain.
4. **F2 correction for the record.** The design's F2 statement ("p_L linear in increments +
   O(sinθ) leak") is correct but the leak's in-vivo magnitude is ~0.5 relative, not ≪1. This
   does not touch PACT's safety derivations (R1 is leak-independent) or the exact
   orthogonality Σ D_l g_l = 0 (which concerns the direct p-bus delivery), but it weakens the
   *strength* of the "function-preserving" claim at θ_max=0.07: the field is exactly neutral on
   the damped-sum channel and only approximately neutral on the total loss channel. Recorded as
   a sharpened caveat for E3's trajectory rule.

## 4. Consequences for the ladder

- **M0 GO ⇒ proceed to E0–E2** (lib kernels `chiron_pact_*`, trainer flags `--pact-coef` /
  `--pact-gate` / `--pact-clamp`, unit suite `test.sh chiron-pact`, λ calibration) per the
  design §13. E4 remains **paired same-binary 30k arms** (~44 GPU-hr) and an owner spend
  decision before launch.
- The probe stays committed (env-gated, zero-cost when off) — it is the C3 occupancy
  observable for E3/E4, and the A-vs-p residual is now a standing WhiSC-leak monitor.
- λ grid for E2 stays {3e-3, 1e-2} with the field-RMS rule binding; given M0's magnitude, start
  at the smaller value first.

## 5. Artifacts

- Probe: glades-trainer `80a269e` (`trainer/chiron_main.cpp`, `[pact-m0]` lines, env-gated).
- Logs: `logs/pact_m0_smoke_20260704_0429.log`, `logs/pact_m0_full_20260704_0432.log` (3-seq,
  protocol lesson: one probed window per val event at `--val-batches 1`),
  `logs/pact_m0_full8_20260704_0435.log` (the 8-window measurement of record).
- Plan: `docs/superpowers/plans/2026-07-04-chiron-pact-m0.md` (glades-ml).
- Design: `docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md` (§13
  updated with this result).
