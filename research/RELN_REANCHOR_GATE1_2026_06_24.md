# ReLN Reverse-Consistency — Gate-1 Treatment Result (2026-06-24)

**Verdict: PASS (trigger-prevention).** `--reln-reanchor` contained the q-side
reverse-amplification instability at its source, on the production data-scale
recipe **with gg-clamp OFF**, holding 0 grad-skips through the danger zone where
the historical control died — to 1.70B tokens.

Spec: `docs/superpowers/specs/2026-06-23-reln-reverse-consistency-design.md`.
Plan: `docs/superpowers/plans/2026-06-23-reln-reverse-consistency.md` (Task 5,
adapted — see "Phase-0 deviation" below).

## Configuration

Treatment-only Gate-1 (owner-chosen, leveraging the historical control):
`run.sh flagship --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250
--zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0
--sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5
--dq-layer-clamp 1.0 --dq-embed-clamp 1.0 **--reln-reanchor** --steps 26000
--seed 1337 --save-every 6000` + trigger armed (`--sira-grad-trigger-dump
--sira-grad-trigger-stop --sira-grad-trigger-sumsq 1e20 --sira-qbranch-trace-token
265`). **NO `--grad-group-clamp`** — the cure-alone test.

- Step-1 loss 10.7816 / ‖g‖ 0.831 — bit-matches the documented data-scale gold
  step-1, confirming re-anchor is near-identity at the start (as designed).
- 16.3h wall, 28–29k tok/s, log `logs/reanchor_g1_treatment.log` (glades-trainer),
  checkpoint `database/checkpoints/_reanchor_g1_treatment`.

## Control (historical)

The production data-scale recipe **without gg-clamp and without re-anchor**
diverged: accum=4 run #1 **died at ~1.49B tokens** (the instability the gg-clamp
was introduced to contain — see CLAUDE.md / `datascale_5b_run_inflight`). That is
the comparison baseline for this treatment-only gate.

## Result — danger-zone trajectory

| step | tokens | ‖g‖ | loss-scale | val NLL |
|---:|---:|---:|---:|---:|
| 20,785 | 1.36B | 0.485 | 1.000 | — |
| 21,651 | 1.42B | 0.588 | 0.850 | — |
| 22,517 | 1.48B | 2.298 | 0.218 | — |
| **23,383** | **1.53B** | **7.011** | **0.071** | 3.31 (ppl 27.4) |
| 24,249 | 1.59B | 0.718 | 0.696 | — |
| 25,115 | 1.65B | 1.895 | 0.264 | — |
| 25,981 | 1.70B | 1.591 | 0.314 | — |
| 26,000 (final-val) | 1.70B | — | — | **3.0422 (ppl 20.95, acc10 0.90)** |

- **0 grad-skips, 0 trigger fires** across the entire run. `global_sumsq` never
  approached the 1e20 threshold (peak ‖g‖ 7.0 → sumsq ~49; the original
  catastrophic event hit 2.5e25).
- **Peak ‖g‖ = 7.0 (transient, step 23,383 / 1.53B)** — the q-side stress IS
  real and manifests right at the control's death point, but re-anchor **bounds
  it at O(10), not O(10¹²)**, and the run **self-recovers** (‖g‖ back to 0.72 by
  1.59B), staying calm (0.7–1.9) to the end.
- Reached 1.70B — past the 1.49B control death AND the ~1.6B re-emergence.
- Final val 3.04 / ppl 20.95, position-stratified flat (2.93–3.14), no
  degeneration.

## Interpretation

This is **source-cure behavior, not downstream containment.** The transient
bump-and-recover signature (‖g‖ 0.6 → 7.0 → 0.7) is qualitatively different from
gg-clamp's chronic per-step firing past 1.6B: re-anchor makes `xhat` unit-RMS by
construction, so even when the q-side recompute drifts (the bump), `dgamma`
cannot overflow → the loss-scaler absorbs it and the system self-corrects, rather
than requiring a clamp to fire every step. Where the uncontained control diverged
at 1.49B, re-anchor sailed past it.

**The Phase-0 verdict (R/V) was not captured** — the trigger never fired (because
re-anchor prevented the overflow), so no L00 snapshot was dumped. This is the
*desired* failure mode of the dump+stop arming (it only fires on a FAIL). The
mechanism is inferred R (recompute-drift) from the prior artifact + physics
(mean barely moves, recompute σ ≈ 13× saved σ under reversibility); re-anchor
cures R and V alike, so the verdict was confirmatory rather than gating.

### Phase-0 deviation (why no fresh diagnostic)

The cheap accum=1 SIRA proxy (the documented step-24070 trigger config) was
**re-run first and did NOT reproduce the instability** on the current codebase
(trainer `f6027ed`, glades-ml `chiron3` vs the Jun-6 `53f88a5`/`e09789f`):
trained clean to step 30000, final val 4.016, no trigger. The instability is a
stochastic spike, fragile to the exact numerics + same-seed non-reproducibility.
So the diagnostic pivoted to testing the cure directly on the recipe that
reliably diverges (the data-scale recipe) — this Gate-1 run.

## Matched gg-clamp control (on disk — apples-to-apples, 2026-06-24)

The production data-scale base run `logs/datascale5B_20260618_221547/run.log`
(`chiron_1B_T16384_datascale5B`) is the IDENTICAL recipe — accum=4, lr 3e-4,
seed 1337, warmup 750 — **with gg-clamp instead of re-anchor**. Same seed → same
data order → a near-perfect head-to-head control at matched tokens. Read off disk
(no new GPU):

| danger-zone (step / tokens) | gg-clamp control ‖g‖ (scale) | re-anchor ‖g‖ (scale) |
|---|---|---|
| 22,001 / 1.44B | 0.238 (1.000) | 0.485 (1.000) |
| 23,001 / 1.51B | **1712.6 (0.000)** | 7.011 (0.071) |
| 24,001 / 1.57B | 0.293 (1.000) | 0.718 (0.696) |
| 25,001 / 1.64B | **1148.6 (0.000)** | 1.895 (0.264) |
| 26,001 / 1.70B | **2361.6 (0.000)** | 1.591 (0.314) |
| 27,001 / 1.76B | **5420.1 (0.000)** | — (run ended 26k) |

| val NLL (matched tokens) | gg-clamp control | re-anchor |
|---|---|---|
| ~1.64–1.70B | **3.357 @ 1.64B** | **3.042 @ 1.70B** |
| best 4-batch window | 2.5067 @ 22.8k | 2.2036 @ 5.2k |

**Two findings:**

1. **Gradient regime: re-anchor is ~250–750× cleaner.** gg-clamp's ‖g‖ explodes
   to **1700–5400 with the loss-scale floored to 0.000, every step past 1.6B**
   (the "fires chronically past 1.6B" behavior, made quantitative) — it holds 0
   skips only by violently clamping a *raging* instability. Re-anchor keeps ‖g‖
   at **O(1–7)** because it removes the xhat inflation at the source, so nothing
   downstream blows up. This is the source-cure-vs-symptom-clamp distinction as a
   number: gg-clamp clamps the symptom (dgamma/dbeta norm) while the q-side drift
   still inflates the rest; re-anchor fixes the cause so the symptom never forms.

2. **Val: re-anchor is ~0.3 nat BETTER at matched tokens** (3.04 @ 1.70B vs
   gg-clamp 3.36 @ 1.64B), not worse. **This REVERSES the earlier "yellow flag."**
   That flag compared re-anchor's 3.04 against the **accum=1** run's "2.797 @
   1.3B" — a different batch recipe, not a valid control. Against the correct
   accum=4 gg-clamp control, re-anchor is more token-efficient on val, plausibly
   because gg-clamp's chronic hard clamping (‖g‖→thousands, scale→0) distorts the
   optimizer's signal while re-anchor keeps it clean.

**No-harm parity: PASS, and then some** — re-anchor does not harm val; at matched
tokens it improves both the gradient regime and val vs the production gg-clamp
containment. (Caveats: single seed, 4-batch window noise ±0.1–0.2 on val — though
the ‖g‖ gap of 7 vs 1700–5400 is structural, not noise; one comparison point.)

## Caveats / scope

- **Single seed (1337), treatment-only.** No fresh same-codebase control (relies
  on the historical run #1 death at 1.49B). No matched no-harm parity run yet
  (that was the separate Gate-1.2 the owner deferred).
- **To 1.70B, not 5B.** Whether re-anchor holds to 5B (the prize) is the Gate-2
  question. The transient bump shows the instability is bounded-not-absent — it
  could re-stress further out.
- **Val is constant-LR base (no finish anneal).** 3.04 @ 1.70B is the base
  trajectory; best-val (vs the flagship's 2.5) is a Gate-2 metric and needs the
  late flat-3e-5 finish. Not a no-harm parity measurement.

## Next

Gate-2 — full 5B cure-alone (`--reln-reanchor`, gg-clamp OFF, 76k steps, ~47h):
PASS = 0 grad-skips through 5B AND new best-val (< 2.5). Optionally a matched
no-harm parity / fresh control first if a controlled best-val comparison is
wanted before the 47h commit.
