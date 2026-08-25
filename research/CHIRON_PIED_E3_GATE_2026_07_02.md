# CHIRON PIED — E3 Gate Result: PASS (2026-07-02)

**Status:** E3 PASS on all pre-registered bars (stability gate + bounded-regression;
single-seed, 4-batch val windows). PIED additionally shows a **negative net gap**
(treatment better than matched baseline at every val point from step 1000 on,
val@2500 −0.021 nat) — promising but NOT a perplexity claim; E4 (matched 30k) is
the decisive gate per the ladder.
**Design:** `docs/superpowers/specs/2026-07-01-chiron-pied-increment-dropout-design.md` (§13 pre-registration)
**Implementation:** `research/CHIRON_PIED_IMPLEMENTATION_2026_07_01.md`
**Commits:** glades-ml `9c84806eb` (prototype) + `6e8730173` (perf pass);
glades-trainer `5a113bb` + `68ae99e`.

## Setup

Matched single-seed pair, production flagship WhiSC recipe, both arms on the
identical post-perf-pass binaries, same data order, same val batches, verified
matched at step 1 (val bit-identical 10.7428):

```
sh run.sh flagship --steps 2500 --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 \
  --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 \
  --sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5 \
  --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
  --whisc-coupling --rot-theta-max 0.07 --seed 1337 [--inc-dropout 0.1]
```

Checkpoints: `database/checkpoints/chiron_1B_pied_e3/chiron_1B_pied_e3.final`
(treatment), `database/checkpoints/chiron_1B_pied_e3_base/chiron_1B_pied_e3_base.final`
(baseline). Logs: `logs/pied_e3_treatment_20260702_0304.log`,
`logs/pied_e3_base_20260702_0854.log`.

A fresh matched baseline was run (rather than reusing the 2026-06-30 WhiSC gate
value 2.78) because the expected effect size (±0.01–0.05 nat) is far below the
documented 4-batch window noise and the binary drifted (WhiSC perf passes 3+4 +
PIED-era trainer); the paired arms cancel window noise and drift in the gap.

## Val NLL trajectory (matched 4-batch windows)

| step | baseline | PIED π=0.1 | Δ (treat−base) |
|---:|---:|---:|---:|
| 1 | 10.7428 | 10.7428 | 0 (matched) |
| 250 | 7.7146 | 7.7068 | −0.008 |
| 500 | 5.3300 | 5.3424 | +0.012 |
| 750 | 4.2765 | 4.3071 | +0.031 |
| 1000 | 4.0251 | 3.9903 | **−0.035** |
| 1250 | 3.7291 | 3.6489 | **−0.080** |
| 1500 | 3.6982 | 3.5440 | **−0.154** |
| 1750 | 3.8298 | 3.5835 | **−0.246** |
| 2000 | 3.2663 | 3.1608 | **−0.106** |
| 2250 | 3.1283 | 3.0272 | **−0.101** |
| 2500 | 2.8252 | 2.8045 | **−0.021** |

The early points (500/750) show the predicted small tax (+0.01–0.03, inside the
pre-registered Δ_tax band); from step 1000 onward the treatment is better at
every matched window — 7 consecutive checkpoints, which a single-window
fluctuation does not produce (the arms share val batches, so window noise is
common-mode in the gap). The 1750 spike is a shared window transient (both arms
bounce; the baseline bounces harder). The gap at 2500 (−0.021) is smaller than
mid-run — whether the mid-run advantage compounds or washes out at 30k is
exactly the E4 question.

## Pre-registered bars → verdicts

| Bar (design §13) | Measured | Verdict |
|---|---|---|
| No divergence / NaN / loss-scale collapse | none in either arm | **PASS** |
| 0 grad-skips | 0 (treatment), 0 (baseline) | **PASS** |
| ‖g‖ max ≤ 1.1× baseline max | 2.958 vs 3.211 (**0.92×** — treatment smoother) | **PASS** |
| val@2500 ≤ baseline + 0.10 | 2.8045 vs 2.8252 (−0.021) | **PASS** |
| Trajectory rule: gap ≤ +0.04 @2500 AND shrinking across {1k,1.5k,2k,2.5k} | gap negative at all four points | **PASS** (rule designed for a positive tax; trivially satisfied) |
| Δ_tax ∈ [0.01, 0.05] (else retry π=0.05; kill > 0.10) | net gap ≤ 0 — tax absent or fully offset by step 1000 | **PASS** (better than predicted band) |
| Wall ≤ +2% | 6533.7s vs 6410.5s = **+1.92%** (tok/s window read: −2.2%) | **PASS** (borderline; post-perf-pass kernels — was −2.75% pre-pass) |

The `[whisc]` monitor behaved identically to the flagship in both arms
(a-clamp binding at 0.125, ρ_eff → ~4.1e3): PIED does not perturb the detached
whitening stats regime (design failure mode 5 not firing).

## Notable secondary observation

The treatment's **worst gradient transient is smaller than the baseline's**
(max ‖g‖ 2.958 vs 3.211) — consistent with the design's conditioning story
(the Fisher-weighted increment-energy penalty damping p-pathway spikes), and
the opposite of LayerDrop's signature (4 large spikes vs baseline's 1).

## Honest caveats

- **Single seed (1337), 4-batch val windows.** The −0.021@2500 point estimate is
  within window-noise scale on its own; the 7-point run of negative gaps is the
  meaningful signal. No cross-seed claim.
- **E3 is the stability gate, not the perplexity gate.** The pre-registered
  decisive test is E4 (matched 30k vs the whisc30k flagship run; ship bar
  Δ ≤ −0.02 nat, kill ≥ +0.02).
- The early tax (+0.01–0.03 through step 750) is real and matches the
  prediction; the crossover at ~step 1000 is earlier than LayerDrop's
  never-crossing trajectory — mechanism-consistent but horizon-dependence
  unknown until E4.

## Disposition

E3 **PASS** → E4 is authorized per the ladder (matched 30k, ~22 GPU-hr for the
treatment; the whisc30k flagship run is the natural baseline at that horizon
since Δ-precision requirements relax at the 30k effect scale, or a fresh matched
30k baseline if the E3 discipline is preferred). E4 launch is an owner decision
(GPU-day class spend). Treatment E4 command = the E3 treatment command with
`--steps 30000 --save-every 6000`.
