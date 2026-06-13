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
2. **Schedule-shape validation** (the §1 risk): before committing the full
   run, launch the EXACT chosen config (correct `maxSteps` for 5B) but kill
   after ~3–5k steps. Verify: warmup completes clean; the extended peak-LR
   phase is stable (skip/clamp behavior bounded); early NLL trajectory
   tracks or beats the flagship's per-token curve. ~2–4 h, cheap vs the
   2-day run. This is the gate that converts "pilot says accum wins" into
   "the long-horizon schedule is sound."

## 5. Sequence to commit

1. accum=8 probe completes → fix batch size. (in flight)
2. Short schedule-validation run at the chosen budget's config. 
3. Owner confirms token budget (compute commitment).
4. Launch committed run; checkpoint = new blessed flagship candidate;
   re-anchor future work there (Phase-3 prereg intent).
