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
