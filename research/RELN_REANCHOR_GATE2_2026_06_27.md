# ReLN Reverse-Consistency — Gate-2 Cure-Alone Result (2026-06-27)

**Verdict: PASS — and a major perplexity flagship.** `--reln-reanchor` trained the
full data-scale recipe **with gg-clamp OFF** to 4.33B tokens with **0 grad-skips**,
and produced a verified **−0.62 nat** improvement over the production flagship —
because curing the q-side instability at its source removes a large, previously
silent perplexity tax that gg-clamp's containment was paying.

Spec: `docs/superpowers/specs/2026-06-23-reln-reverse-consistency-design.md`.
Plan: `docs/superpowers/plans/2026-06-23-reln-reverse-consistency.md` (Gate-2).
Builds on Gate-1 PASS (`research/RELN_REANCHOR_GATE1_2026_06_24.md`).

## Run

Two-stage, matching the production flagship recipe exactly except gg-clamp →
re-anchor:
1. **Constant-LR base** — `run.sh flagship --accum 4 --lr 3e-4 --warmup 750
   --sira-warmup 250 --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2
   --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0
   --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 **--reln-reanchor**
   --steps 60000 --seed 1337` (NO `--grad-group-clamp`), trigger armed dump+stop.
   → step 60000 / 3.93B, checkpoint `chiron_1B_T16384_reanchor5B.final`.
2. **Flat-3e-5 finish** — resume the base with `--lr 3e-5 --warmup 0
   --no-resume-warmup --steps 66000`. → step 66000 / 4.33B, checkpoint
   `chiron_1B_T16384_reanchor5B_finish.final`.

Step-1 ‖g‖ 0.831 (bit-matches gold). ~38h base + ~3.8h finish. Logs:
`logs/reanchor_g2_base.log`, `logs/reanchor_g2_finish.log` (glades-trainer).

## Stability — clean cure to 4.33B (0 grad-skips, 0 trigger)

| phase | tokens | ‖g‖ regime | grad-skips | trigger |
|---|---|---|---|---|
| danger zone (1.49–1.6B) | through ~24k steps | ~1.0 (no bump at log cadence) | 0 | none |
| new regime (2–3.9B) | 30k–60k steps | ~1.0 (0.7–1.2) | 0 | none |
| finish (3.93–4.33B) | 60k–66k steps | resume artifact ‖g‖ 376 → recovers | 0 | none |

The instability — which killed the no-clamp control at 1.49B and which gg-clamp
"contains" only by clamping ‖g‖→1700–5400 every step past 1.6B — was a **non-event**
under re-anchor (‖g‖~1.0 throughout). No prior run trained cleanly past ~2B; this
one reached 4.33B with a flat gradient regime.

## Verified val — wide 32-batch, apples-to-apples (the headline)

The `chiron_infer` teacher-forcing path is currently **broken** (mean_nll≈63 for
ALL checkpoints incl. the known-good flagship — a forwardInfer/fuse-attn serving
regression, NOT a model issue; see Serving caveat). So verification used the
**trainer's own val** (the ship metric that produced the flagship's 2.5), widened
to **32 batches = 33.5M positions** per eval, **same `--seed 1337`** so all
checkpoints val on identical windows. `--lr 1e-9` makes the resume steps no-ops so
the val reflects the loaded checkpoint. Method validated: the flagship reproduces
its documented ~2.5.

| checkpoint | tokens | val NLL (win1 / win2) | acc1 | acc10 | Δ vs flagship |
|---|---|---|---|---|---|
| flagship (datascale finish_clean) | 4.33B | 2.484 / 2.602 | 0.36 | 0.93 | — (✓ reproduces ~2.5) |
| re-anchor base (60k) | 3.93B | 1.909 / 2.046 | 0.49 | 0.96 | **−0.57** |
| **re-anchor finish (66k)** | 4.33B | **1.852 / 1.988** | **0.51** | **0.97** | **−0.62** |

Window-matched deltas (finish vs flagship): −0.632 (win1), −0.614 (win2) →
**−0.62 nat**, with **+0.15 acc1**. The finish is marginally better than the base
(~0.06); the earlier 4-batch reading suggesting the finish hurt was window noise.

**Flagship candidate: `chiron_1B_T16384_reanchor5B_finish.final` (val ~1.92).**

## Why the gain is this large (mechanism)

Flagship and this run are the SAME recipe/data/tokens (66k steps, pretok-data),
differing ONLY in containment (gg-clamp ↔ re-anchor). The gg-clamp control's
constant-LR base plateaued at **2.76**; re-anchor's base is **1.98** — a ~0.78 nat
gap at the base level. The instability, even when "contained" by gg-clamp, was
**silently corrupting the optimization** past 1.6B (clamping ‖g‖→thousands every
step). Re-anchor removes the cause, so learning proceeds unimpeded. This reframes
the q-side instability from "a stability nuisance gg-clamp handles" to **"a ~0.7
nat perplexity tax the source-cure eliminates."** The re-anchor-vs-gg-clamp val
gap widened monotonically with tokens (0.3 nat @1.7B → 0.62 nat @4.33B), exactly
as expected if gg-clamp's damage accumulates while re-anchor descends cleanly.

## Caveats / what's NOT yet done

- **Single seed (1337).** The lineage's Gate-0 methodology calls for a ≥3-seed
  gate before a formal ship. The −0.62 nat magnitude is far beyond seed variance
  (~0.01–0.02), so the *direction* is not in doubt, but a multi-seed confirm is
  the standard for promotion.
- **Serving path — FIXED (2026-06-27).** `chiron_infer --tf-check` initially
  returned mean_nll≈63 for every checkpoint including the flagship — root cause:
  the SCFA production models train with `--no-fuse-attn --fuse-attn-reln` and
  carry no per-layer `gamma_p`; chiron_infer auto-disabled per-layer fuse but left
  reln-fuse OFF, so the forward omitted the fusion (same CLASS as the QK-Norm
  serving bug). Fixed (`fffe70d`): auto-enable reln-fuse for SCFA-no-gamma_p
  checkpoints. **This also independently corroborates the val result**: with the
  fix, TF nll = flagship **2.44** (≈ documented 2.348 ✓), re-anchor finish
  **1.78** → **−0.66 nat**, matching the trainer wide-val −0.62. `runner.sh
  --flagship` now serves the candidate correctly (auto reln + QK-Norm).
- **Finish was nearly redundant.** Re-anchor's base (1.98) was still descending,
  so the flat-3e-5 finish only added ~0.06 (vs ~0.26 for the gg-clamp flagship,
  whose base had plateaued). A longer constant-LR base + later finish may extract
  more — open follow-up.

## Recommendation — PROMOTED 2026-06-27

`chiron_1B_T16384_reanchor5B_finish.final` promoted to production flagship
(val ~1.92, −0.62 nat over the prior 2.54) per owner direction on the single-seed
evidence. CLAUDE.md + runner.sh updated; serving fix landed (`fffe70d`).
**Backfilling**: multi-seed (≥3) Gate-0 confirm (seeds 2024/4242) in progress —
the magnitude dwarfs seed variance so the direction isn't in doubt, but the formal
gate is the lineage standard.
