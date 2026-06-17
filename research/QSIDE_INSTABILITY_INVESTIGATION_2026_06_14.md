# Q-side reverse-amplification instability — investigation (2026-06-14)

**Status**: investigation OPEN. This is the long-term central thread — the
structural ceiling on scaling the CHIRON symplectic architecture. GPU
validation is queued behind the in-flight accum=1 5B run (~49h); this doc
is the mechanism analysis + prototype plan.

## 1. The recurring failure (3 occurrences, escalating evidence)

| occurrence | regime | what happened | clamp outcome |
|---|---|---|---|
| seed-2024 30k (pre-clamp) | accum=1, lr 7.5e-5 | L00/L01 dgamma → 1e25, 5,809 skips | **dq clamps FIXED it** (0 skips) |
| 5B run #1 | accum=4, lr 3e-4 | broad dgamma (all 24 layers ~1e19), global 1e22, step 22737 | clamps fired on 300k rows but **could NOT contain** — 45 skips, killed |
| 5B recovery A | accum=4, lr 1.5e-4, resume@20k | ‖g‖ 90k at first resumed step | **fragility already in weights**; not LR-fixable |

## 1b. NEW DATA (2026-06-14): the instability is HORIZON-driven, not batch-specific

The accum=1 5B run (proven flagship recipe, lr 7.5e-5) ALSO hit the
instability — a 14-skip cluster at **step 76138 / 1.25B tokens**, then
**recovered** (contained, weights protected; the run continued at val 2.99).
This is decisive for the diagnosis:

- **It is NOT an accum=4 artifact.** The flagship never saw it only because
  it stopped at 491M tokens; past ~1B tokens it appears regardless of batch.
  The q-side instability is a **horizon / token-count phenomenon** that
  emerges past ~1–1.5B tokens — the structural ceiling on scaling the data
  budget, which is exactly the program's dominant lever.
- **Severity scales with LR (supports H1).** Same instability, different
  outcome: at accum=1 / lr 7.5e-5 the clamps+guard CONTAIN it (14 skips,
  recover); at accum=4 / lr 3e-4 it ESCALATES to a killing wave (45 skips →
  1e22, divergent). Lower LR keeps it containable; higher LR makes it lethal.
- Practical corollary: the proven low-LR recipe survives the instability at
  least to ~1.25B tokens (2.5× the flagship horizon). But as the data budget
  grows further (toward Chinchilla 17B), these contained bursts will recur
  more often and the margin shrinks — so the root fix matters more, not less,
  as scaling continues. The investigation is the right long-term priority.

## 2. Mechanism

The q-side backward (readout dq → early-layer reverse amplification → dgamma)
produces, in the unstable regime, a **broad** elevation: every layer's
dgamma sits at ~1e19 *simultaneously*, all FINITE (`bad_groups=0`), summing
to a global norm of ~1e22 that trips the >1e20 guard.

**Why the dq clamps don't contain it at scale**: `row_rms_clamp` bounds each
dq ROW's RMS to τ. But dgamma = Σ over T tokens of (per-token contribution).
When a large fraction of rows are simultaneously at the τ ceiling (observed:
up to 300,809 / 393,216 layer-row slots = 76%), the per-row bound holds but
the SUM is unbounded. Per-row clamping cannot bound an aggregate. This is the
core limitation: **the clamps are a per-row patch; the overflow is aggregate.**

**Why large batch makes it worse** (hypotheses, to be discriminated):
- (H1) Higher LR (3e-4, enabled by the larger batch) drives the weights into
  a sharp/fragile minimum faster. Supported by: the failure scales with LR
  (3e-4 failed @22737; the lower-LR flagship regime was containable).
- (H2) Large batch reduces gradient noise → less implicit regularization →
  sharper minima (standard large-batch generalization-gap effect). Supported
  by: ‖g‖ stayed ~0.3 then spiked SUDDENLY at ~1.4B tokens (sharp-minimum
  signature), and the step-20000 state had great loss (3.36) but ‖g‖ 90k on
  perturbation (textbook fragile/sharp minimum).
- (H3) Intrinsic to the symplectic backward: the (q,p) reverse reconstruction
  amplifies q-side perturbations in a way standard residual backward does
  not. This is the architecture-specific root and the highest-value target.

## 3. Candidate mitigations (ranked; prototype cheapest-first)

1. **Per-group gradient clamp (primary prototype)**. Clamp each layer's
   dgamma/dbeta (and dE) to a max L2 norm BEFORE the global-norm sum —
   per-group clipping, which bounds the aggregate the global clip can't
   (the global clip rescales AFTER summing; if the sum is already ~1e22 it
   skips). Directly targets the observed overflow groups (L*.dgamma).
   Cheap: a per-vector-norm clamp kernel on the small (size-m) dgamma
   buffers in `compute_grad_norm_sq`'s neighborhood. New flag, default-off.
2. **Finite-large global rescale, not skip (complementary)**. The failures
   were all FINITE (sumsq 1e20–1e22, never nan/inf until late). Change the
   guard so a finite-but-huge global norm is CLIPPED (scaled to grad-clip)
   rather than skipped; reserve skip for genuine nan/inf. Keeps the optimizer
   moving through rough patches instead of freezing. Risk: a clipped (tiny)
   update may not escape the fragile minimum — test whether it prevents the
   wave or just defers it.
3. **Lower effective LR / sharpness control at large batch**. If H1/H2, the
   real fix is not entering the sharp minimum: gentler LR-vs-batch scaling
   (sub-linear), longer warmup, or a sharpness-aware term. Bigger change.
4. **Symplectic-backward root fix (H3, highest payoff, hardest)**. Audit the
   q-side reverse reconstruction (ReLN backward recompute, the dq cascade
   across layers) for the amplification source; a normalization or a
   structurally-bounded reconstruction would cure it at the root and unlock
   aggressive scaling permanently.

## 4. Validation plan (GPU, queued behind the accum=1 run)

The honest constraint from recovery A: **resume ≠ original trajectory** (the
data-loader position isn't restored on `--load`), so resume-from-checkpoint
is NOT a valid test harness. Validation must be a FRESH run.

- **Cheap discriminating test**: a fresh accum=4 / lr 3e-4 run to ~25k steps
  (through the danger zone where #1 failed at 22737) WITH mitigation 1
  enabled. If per-group clamping carries it through the danger zone with
  bounded ‖g‖, mechanism 1 is the fix and accum scaling is reclaimed. ~3.5h
  of GPU. If it still bursts, escalate to 2→3→4.
- This also serves as the real long-horizon stability test the short
  validation/pilot structurally couldn't provide (lesson from
  `DATASCALE_LR_SCHEDULE_DESIGN_2026_06_13.md`).

## 5. Sequencing

The accum=1 5B run (in flight) banks data scale on the proven recipe,
independent of this thread. This investigation runs in parallel as dev work
(design now; implement mitigation 1 + unit tests next), with GPU validation
when the accum=1 run completes (or a deliberate interrupt if the
investigation is prioritized). Success here is what makes future
large-batch / higher-LR / larger-token runs viable — i.e., it removes the
ceiling on the whole scaling program.

---

## accum=1 run OUTCOME: huge win banked, then degraded at ~1.6B tokens (2026-06-15)

The accum=1 5B run did NOT complete — it hit the instability ceiling and was
stopped at step ~100k / 1.64B tokens. Trajectory (val every 10k steps):
3.673 → 3.585 → 3.560 → 3.612 → **3.088 → 2.984 → 2.994 → 2.740 → 2.797**
→ 3.388 (degraded). 

- **Peak: val 2.7403 @ step 80k (1.31B tokens)**; best SAVED checkpoint
  **step-90000, val 2.7969** (preserved as `...datascale5B_BEST_val2p797.final`).
- **This is −0.71 nat below the current flagship (3.5062)** — by far the
  largest single improvement in the project's history, and it validates the
  data-scale thesis dramatically: the flagship at 491M tokens was severely
  starved; at 1.3–1.5B tokens (still pre-anneal) the model is a categorically
  better LM (acc1 0.31 vs the flagship's ~0.14).
- **Degradation mechanism** (steps ~93k–100k): NO new grad-skips, but ‖g‖
  climbed to a SUSTAINED 4–5M (vs normal ~0.4) and train EMA rose 3.04→3.43.
  The gradients were huge-but-under-the-1e20-guard, so clipped-but-nonzero
  updates slowly corrupted the model — a slow-motion version of the accum=4
  divergence. The lr-7.5e-5 recipe survived the contained 14-skip burst at
  1.25B but entered sustained fragility by ~1.6B.

## REVISED CONCLUSION — the instability is the absolute ceiling on data scale

Both recipes now confirm it: NO current recipe reaches even 2B tokens
cleanly. accum=4/lr3e-4 diverged at 1.49B; accum=1/lr7.5e-5 degraded at
~1.6B. The dominant capability lever (data scale, 108B tokens available) is
**hard-blocked past ~1.3–1.5B tokens by the q-side instability.** The
investigation is no longer a parallel nice-to-have — it is THE gate on all
further capability progress. The −0.71 win we DID bank (a 3× data increase,
stopped before degradation) is a preview of how large the full prize is if
the ceiling is removed.

**Immediate priority shift**: implement + validate mitigation 1 (per-group
gradient clamp) — now with GPU free. The validation harness is a fresh run
through the danger zone (resume is invalid per recovery A). If it bounds the
aggregate dgamma and carries a run past ~2B tokens, data scale is unlocked.

---

## MITIGATION 1 VALIDATED — per-group clamp lifts the ceiling (2026-06-15)

Fresh accum=4 / lr 3e-4 run to step 25000 (1.64B tokens) WITH
`--grad-group-clamp 1.0` — through the step-22737 danger zone where run #1
diverged. Artifact: `glades-trainer/logs/gradgroupclamp_validation_*`.

**Result: PASS — 0 grad-skips in 25,000 steps, run completed.**

| signal | run #1 (no gg-clamp) | this run (gg-clamp 1.0) |
|---|---|---|
| step 22737 region | diverged: ‖g‖→1e10, sumsq 1e22, 45 skips, killed | ‖g‖ peak **2410** (step 23001), then 0.47 — **0 skips** |
| broad dgamma elevation | yes → overflowed global guard | yes (peak **43/48** vectors clamped @21352) → bounded |
| trajectory | died at 1.49B | healthy: val 3.2257 @ 1.64B, descending |
| clamp activity | n/a | 997 fires / 25k steps (~4%), intermittent |

**Mechanism confirmed**: the instability still OCCURS (dgamma spikes broadly,
43/48 groups), but per-group clamping bounds each vector's L2 norm so the
global sum stays ~2410 (under the 1e20 guard) instead of 1e22. grad-clip then
rescales the bounded norm and training continues — converting a lethal
overflow into a routine clipped bump. Crucially the trajectory is UNHARMED
(val tracks the unclamped run; the clamp fires on only ~4% of steps and
doesn't perturb healthy learning — validated by on-trend vals).

**This lifts the structural ceiling.** Both prior recipes failed past ~1.5B
tokens; this run cleared it and reached 1.64B clean. The dominant lever —
data scale — is unblocked.

### Next: full data-scale run WITH the fix
The validation used accum=4/lr3e-4 — so **accum=4 + gg-clamp is now viable
(fast AND stable)**, reclaiming the +3.8% wall + per-token bump the
instability had forced us to abandon. Recommended: a full 5B accum=4 run
(76k steps, ~47h) with `--grad-group-clamp 1.0` + the full recipe, to bank
the transformational flagship the −0.71 win previewed (now able to COMPLETE
+ anneal). Residual risk: validated to 1.64B; a full 5B run is 3× further and
may reveal modes beyond the dgamma-aggregate one (escalate to mitigation 2–4
if so), but the fundamental mechanism is now addressed.

---

## Phase 1 (AGC) — implemented + calibrated; validation running (2026-06-16)

Per docs/superpowers/plans/2026-06-16-chiron-stability-techniques.md. The 5B
gg-clamp run was killed at step 30000 (val 3.3132 @ 1.97B, banked as
`...ggclamp_BASELINE_step30k_val3p313.final`) to free the GPU for the
investigation — its core questions were answered (fix validated past the
danger zone); the long run is deferrable and would be improved by a cure.

- **`agc_clamp_vector` kernel + `CHIRONAgcClampTest`** (glades-ml 0fb433f49):
  clip grad to lambda*max(‖w‖,eps); unit-validated (scales to lambda*‖w‖,
  bit-identical below, eps floor) — 0 failures on GPU.
- **Trainer `--agc-lambda`/`--agc-eps`** on dgamma/dbeta, mutually exclusive
  with --grad-group-clamp (trainer commit).
- **CALIBRATION FINDING**: NFNet's default lambda=0.01 OVER-clamps here — fires
  every step on exactly 24/48 vectors (one full half: gamma OR beta), because
  NFNets tunes AGC for weight matrices and EXPLICITLY EXEMPTS LayerNorm
  gains/biases — precisely our dgamma/dbeta. Re-calibrated by sweep: fire rate
  on healthy early steps 40/40 (λ=0.1) → 8/40 (2.0) → 4/40 (10) → **1/40
  (λ=100)**. λ=100 is near-inert on healthy steps while the burst (ratio ~1e8)
  triggers with huge margin. Validation launched at λ=100.
- **Validation gate** (running, ~16h): fresh accum=4/lr3e-4 to step 25000.
  PASS = 0 skips through the danger zone + on-trend val (vs the gg-clamp
  baseline) + FEWER fires than gg-clamp's chronic ~22%+ (the selectivity win).

---

## Phase 2 (GC) — kernel validated; integration finding (2026-06-16, parallel to AGC validation)

`gradient_centralize` kernel + `CHIRONGradCentralizeTest` GPU-validated
(glades-ml; ran time-sliced alongside the AGC validation without disrupting
it — 0 failures, AGC run unaffected at 29,393 tok/s).

**Integration design finding (mirrors the AGC calibration finding):** GC, like
AGC, is defined for 2D WEIGHT matrices (subtract the per-output-row mean over
the fan-in), and is conventionally NOT applied to 1D LayerNorm gains/biases.
So GC's correct target is the attention projection grads dWq/dWk/dWv/dWo
([m, dModel]), NOT the dgamma/dbeta vectors. In the production recipe those
weight grads are BF16 (`W.dWq_bf[l]`), so the trainer integration needs a
BF16-in/BF16-out GC variant (read BF16 → center in FP32 → write BF16), and the
matrix orientation ([out,in] vs [in,out]) must be confirmed at the call site.

**Decision:** the FP32 GC kernel is banked (validated, reusable). Trainer
integration (BF16 variant + apply to dWq/k/v/o + danger-zone validation) is
sequenced AFTER the AGC verdict — because the verdict reshapes the strategy:
if AGC-on-dgamma/dbeta (the actual overflow source) wins, the weight-grad
techniques (GC, AGC-on-weights) are secondary hardening; if it doesn't, the
ReLN-bound cure (Phase 3) jumps ahead. Avoids building blind.

---

## Phase 1 (AGC) VERDICT: NEGATIVE — worse than the gg-clamp (2026-06-17)

AGC λ=100 danger-zone validation completed (0 skips through step 25000) but is
WORSE than the validated fixed-norm gg-clamp on both selectivity and quality:

| metric | AGC λ=100 | gg-clamp (apples-to-apples validation) |
|---|---:|---:|
| grad-skips | 0 | 0 |
| total clamp fires / 25k steps | **4,371** | 997 |
| final train EMA (step 24501) | 3.457 | **3.318** |
| final val (step 25000) | 3.4371 | **3.2257** |

Trajectories matched through step 15k (3.408 vs 3.411), then AGC diverged worse
through the unstable regime (20k+): it fires on ~17% of steps (vs gg-clamp's
4%) and trains ~0.14 nat worse on EMA. My "selectivity win" hypothesis is
FALSIFIED — AGC fires 4.4× MORE often (fewer vectors per event, but far more
events) and distorts the recovery the absolute-norm gg-clamp handles cleanly.

**Root cause (consistent with the 3 prior signals):** AGC's RELATIVE threshold
(λ·‖param‖) is a poor fit for LayerNorm gains/biases — exactly why NFNets
exempts them. The absolute-norm gg-clamp cleanly separates healthy (<1) from
burst (>>1) dgamma; the relative threshold does not, so it over-fires in the
unstable regime and degrades learning. Not a λ-tuning issue (lower λ over-
clamps more; higher λ stops bounding the burst) — the mechanism is wrong for
this target.

**VERDICT: gg-clamp (fixed maxNorm 1.0) remains the best CONTAINMENT mechanism.
AGC retired for dgamma/dbeta.** Decisive redirect: clamping-VARIANT tuning does
not beat the gg-clamp → the path to actual improvement is the SOURCE CURE
(Phase 3, bounded ReLN backward), which would eliminate the dgamma overflow —
and the quality cost of ANY clamping — at its origin. GC (Phase 2) targets
weight grads (orthogonal to the overflow source), so it is deprioritized below
Phase 3. **Next: Phase 3.**
