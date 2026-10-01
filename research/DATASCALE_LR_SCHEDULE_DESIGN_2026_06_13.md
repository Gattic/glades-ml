# Data-scale retrain — LR-schedule design (2026-06-13, draft)

**Purpose**: design the LR schedule for the data-scale + accum retrain (the
merged dominant arc). Pre-registration seed; the accum=8 probe and the
short schedule-validation run below close the open parameters before the
expensive committed run.

**Inputs established**:
- Corpus: ~108B tokens int32 in `glades-trainer/pretok-data/train/`;
  flagship used 0.49B (0.45%). Data scale is fresh-token, not epoching.
- accum=4 + lr 3e-4 (linear-scaled): n=3 confirmed −0.0320 ± 0.0060 nat
  per token + 3.78% wall (`research/ACCUM_PILOT_2026_06_12.md`).
- accum=8 + lr 6e-4 probe: IN FLIGHT (decides batch size; see §4).
- Schedule mechanics (`trainer/chiron_main.cpp` ~17226): linear warmup 0→lr
  over `warmupSteps`; if `--lr-decay`, cosine 1.0→`lrDecayMin` (0.1) over
  `[warmupSteps, maxSteps]`. Peak-LR is held only at the warmup boundary;
  decay begins immediately after.

## 1. The core schedule-shape risk

The pilot measured accum=4 over a **1,250-step** cosine horizon: the model
spent a large token fraction in the decayed-LR tail. A 5B-token run is a
**~76k-step** horizon — the same cosine now holds the model near peak LR
for most of training, decaying only late. **The pilot's per-token NLL
advantage was measured under a fast-decay shape and does not automatically
transfer to a slow-decay shape.** This is the one parameter that must be
validated, not assumed.

Corollary on stability: the flagship's late q-side burst hit ~80% through
the 30k run (step ~24k). Under a slow cosine, the high-LR regime is
*extended*, so the burst-prone window is longer — exactly where accum's
batch-averaging stability margin (pilot §4: accum arms rode the bump
0.09–0.26 better) is most load-bearing. Expect the clamps to matter, and
to fire at a different (later, token-wise) point than in the flagship.

## 2. Step-count / wall table (per token budget)

accum=4 → 65,536 tok/step; accum=8 → 131,072 tok/step. Wall at ~29,300
tok/s (accum throughput).

| budget | accum=4 steps | accum=8 steps | wall @29.3k |
|---|---:|---:|---:|
| 2.5B (≈5×) | 38,147 | 19,073 | ~24 h |
| **5B (≈10×)** | **76,294** | **38,147** | **~47 h** |
| 17B (Chinchilla) | 259,400 | 129,700 | ~6.7 d |

Recommended budget: **5B tokens (~10×, ~2 days)** — decisive new baseline
without the Chinchilla week. Final call is the owner's (compute).

## 3. Schedule parameters (recommended)

- **Peak LR**: accum=4 → **3e-4** (fixed; accum=8/6e-4 regressed — §4.1).
- **Warmup**: ~1% of total steps (5B/accum=4 → ~760 → round **750**). The
  pilot's token-matched warmup (~0.65% equiv) was stable; 1% is a small
  conservative margin for the longer high-LR hold. NOT token-matched-tiny.
- **Cosine**: `--lr-decay`, `lrDecayMin` 0.1, `maxSteps` = full step count
  for the chosen budget (so decay completes exactly at the end). Unchanged
  mechanism.
- **SIRA warmup**: keep ~the flagship's 1000-step-at-accum-1 token
  equivalent → at accum=4, 250 steps. SIRA is a terminal-state regularizer;
  its warmup should track early-training token count, not the full horizon.
- **Recipe carried forward** (all default-on or in-recipe): regstack
  (zloss+qk-norm), SIRA E/B, dq clamps, cast-elim. The run also implicitly
  RE-VALIDATES these at 10× scale — watch whether each still pulls weight;
  do not assume the 0.49B-token recipe is optimal at 5B.

## 4. Open parameters the probe + validation close

1. **Batch size** — RESOLVED 2026-06-13: accum=8 + lr 6e-4 REGRESSED at
   82M (final 4.0641, behind both A4c 3.9230 and C0 3.9559; zero
   instability — pure efficiency loss). **Retrain uses accum=4 + lr 3e-4.**
   Caveat: the 82M probe confounds batch size with step-starvation (625
   steps); a clean larger-batch test would need a 500M+ token budget. Not
   pursued — accum=4 is the n=3-confirmed choice.
2. **Schedule-shape validation** — RESOLVED 2026-06-13: PASS. Ran the exact
   5B config (`max_steps=76294`, warmup 750, lr 3e-4, accum 4) to step 4001
   / 262M tokens at full peak LR, then stopped. Artifact: glades-trainer
   `logs/schedule_validation_5B_20260613_090658/val.log`. Results:
   - **Warmup clean**: step 751 post-ramp loss 4.64, ‖g‖ 0.72, no full-LR-
     onset spike.
   - **Extended peak-LR stability — the headline PASS**: 262M tokens held at
     full 3e-4 (cosine only 5% in), **zero grad-skips, zero clamp fires**,
     ‖g‖ bounded 0.38–2.39. The flagship's q-side burst (5,809 skips at
     seed 2024) occurred at *decayed* LR late in a 30k run; holding PEAK LR
     ~9× longer than the pilot is completely stable. The §1 concern (extended
     high-LR window is burst-prone) is **disproven** for this recipe.
   - **Per-token trajectory healthy**: cadence vals 4.213 (65.5M) → 3.795
     (131M) → 3.683 (197M) → 3.726 (262M). Ahead of the flagship's original
     per-token curve through 65–197M (by 0.06–0.16) while running at up to
     ~5.7× the flagship's decayed LR. At 262M the run (peak LR) sits ~0.09
     behind the flagship's *annealed* loss (3.635) at matched tokens —
     EXPECTED: validation is 5% into its cosine (pure exploration), the
     flagship was ~50% annealed there. The 262M val's +0.043 uptick is
     single-batch val noise (elevated pos[0]); EMA is monotone.
   - **Caveat (inherent)**: validation proves stability + a healthy early/mid
     path, NOT the 5B endpoint. The payoff is in the annealed tail (17× more
     tokens + final cosine), which only the full run realizes.
   - **Verdict: GO.** Schedule shape is sound; committed run gated only on the
     owner's token-budget decision.

## 5. Sequence to commit

1. accum=8 probe completes → fix batch size. (in flight)
2. Short schedule-validation run at the chosen budget's config. 
3. Owner confirms token budget (compute commitment).
4. Launch committed run; checkpoint = new blessed flagship candidate;
   re-anchor future work there (Phase-3 prereg intent).

---

## COMMITTED RUN #1 FAILED — instability at step 22737 (2026-06-14)

The 5B run (lr 3e-4, accum 4) destabilized at **step 22737 / 1.49B tokens
(30% through), at effective LR 2.47e-4** (peak 3e-4 × cosine 0.825). Two
skip clusters: 22737–41 (5, recovered), then 23198–23237+ (sustained
wave, killed at step 23238). 45 skips total. Process stopped manually.

**Mechanism**: `bad_groups=0` — no single group went non-finite. Every
layer's dgamma sat at ~1–4e19 simultaneously; the *global* sumsq summed to
1e22. The per-layer dq clamp fired on up to **300,809 rows** (~76% of all
24×T layer-row slots) but cannot bound an AGGREGATE when most rows are at
the τ ceiling at once. This is NOT the seed-2024 single-layer 1e25 spike —
it's a broad, distributed amplification driven by the 4× LR.

**Root cause**: lr 3e-4 (4× the flagship's 7.5e-5) is too aggressive at the
long horizon. The clamps (τ=1, tuned at 7.5e-5) reduce per-row magnitude
but the breadth × count of the amplification at 2.47e-4 effective overflows
the global guard.

**THE VALIDATION-GAP LESSON (load-bearing for future arcs)**: the 4k-step
schedule-validation PASSED — but the instability is a **22.7k-step / 1.5B-
token phenomenon**. A short validation structurally CANNOT catch an
instability that emerges at 30% of a long run. Likewise the 82M-token pilot
(1250 steps) never reached the danger zone, so neither A4b nor A4c's pilot
"stability" was evidence at the long horizon. **Long-horizon stability can
only be evidenced by running through the danger zone (≥~25k steps).**

**Clean checkpoints intact**: step 10000/15000/20000 (all pre-instability;
last clean val 3.41 @ 15k). Resume point = step 20000 (1.31B tokens).

### Recovery options
| option | cost | provenance | stability margin |
|---|---|---|---|
| A. Resume@20k, lr 1.5e-4, test danger-zone 20–26k (~3h) then continue | +3h to decision, saves 14h | discontinuity (3e-4→1.5e-4 @20k; standard LR-drop-on-spike) | √-scale; halves the 2.47e-4 failure LR |
| B. Restart clean, lr 1.5e-4 | +14h vs A | clean | same; but still unproven >25k until run |
| C. Restart clean, lr 2e-4 | +14h | clean | less margin (effective ~1.65e-4 at danger zone) |

**Recommendation: A** — resume from step-20000 at lr 1.5e-4 (the √-scaling
rule, pilot-stable A4b LR, half the failure LR) and run THROUGH the danger
zone as the real test. If it clears 20k–26k clean, continue to 76293 (and
decide then whether the discontinuity is acceptable for the blessed
checkpoint or to restart clean at the now-proven 1.5e-4). If it ALSO
destabilizes, the cause is deeper than LR and needs investigation before
any further long run. Data-scale gain (10× tokens) dominates final NLL; the
accum LR being 1.5e-4 vs 3e-4 costs little (pilot: −0.014 vs −0.033, both
beat baseline) and stability is the gating constraint.

---

## RECOVERY A FAILED — instability is in the weight state, not just LR (2026-06-14)

Resumed from step-20000 at lr 1.5e-4 (half the failure LR). Result:
**‖g‖ = 90,824 at the FIRST resumed step (20001)** and 15 sustained skips
from step 20410 — *earlier* than the original run's first skip (22737) and
at half the LR. Killed. Recovery A as designed does not work.

Two findings:
1. **The fragility is in the weights by ~step 20000, not the LR.** Original-
   run ‖g‖ was clean (~0.2–0.5) through ~step 21000 then spiked suddenly
   (241k→0.5→1.5M→127k) — a sharp onset near 1.4B tokens, not a gradual
   build. Lowering LR from a fragile checkpoint doesn't help.
2. **Resume ≠ original trajectory.** The original at step 20000 had ‖g‖~0.3;
   the resume from that checkpoint had ‖g‖ 90k immediately. The data-loader
   position is NOT restored on --load (weights/Adam/step are), so the
   resumed run sees different data at the step-20000 weights. Resume-based
   recovery is therefore unreliable for this failure.

**Conclusion**: the accum=4 / high-LR regime develops a fragile (sharp-
minimum) state around 1.3–1.5B tokens that the current clamp machinery
cannot contain in aggregate, and it cannot be cheaply recovered by
resume+lower-LR. The data-scale + accum arc has hit a real stability wall.

### Strategic options (needs owner direction — 2 failed expensive attempts)
1. **accum=1 data-scale (PROVEN recipe, recommended)**: drop accum, run 5B
   at the exact stable flagship recipe (accum=1, lr 7.5e-5 + clamps). ~305k
   steps, ~49h. Forgoes the accum +3.8% wall and −0.03 per-token bump, but
   banks the dominant data-scale win at the highest confidence (4× lower LR
   than the failing regime; clamps validated at this exact LR). Caveat: 305k
   steps is ~10× the longest prior run; burst could still recur but with far
   more margin.
2. **Investigate accum=4 instability** (research): why large-batch develops
   the sharp minimum at ~1.4B tokens; possible global-norm-aware clamp or
   large-batch-specific stability work. High effort, uncertain payoff.
3. **Restart clean at conservative accum=4 LR (e.g. 1e-4)**: gamble another
   ~47h that a never-fragile trajectory avoids it; weakly supported (1.5e-4-
   from-checkpoint already failed).
4. **De-risk first**: a 2B accum=1 run before committing 5B.

**Recommendation: Option 1.** The accum speedup was a +3.8% optimization;
it's not worth gating the entire (dominant) data-scale program on its
long-horizon instability. Bank the data win with the proven recipe; treat
accum-at-scale stability as a separate research thread.
