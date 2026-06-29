# CHIRON OBSD (Operator-Budgeted Symplectic Drift) — Implementation & Validation Result

**Date:** 2026-06-27
**Status:** Implemented + unit/integration-tested + small/moderate-scale validated. **Production
quality gate (E3/E4 at T=16384) NOT yet run** (the FM-1 perplexity bet is unsettled).
**Design:** `docs/superpowers/specs/2026-06-27-chiron-richer-symplectic-block-design.md`
**Plan:** `docs/superpowers/plans/2026-06-27-chiron-richer-symplectic-block.md`

## What OBSD is

A per-layer, reversible, ReZero-gated, mass-preconditioned nonlinear **drift** added to the CHIRON
block so attention composes across depth through `q`:

```
forward (per layer):  shear (p += Y_l(q)) → drift (q += driftScale·a_l ⊙ tanh(M_l⁻¹ ⊙ N(p) + b_l)) → reln(q)
```

`N(p)=(p−μ)/σ` (parameter-free row-norm), `M_l⁻¹` = `gamma_p[l]` (init 1, diagonal inverse mass
INSIDE the nonlinearity), `b_l` = `beta_p[l]` (init 0), `a_l` = new ReZero gate (init 0). At a=0 the
drift is exactly 0 → bit-identical to the flagship. The bottleneck it targets: the production
flagship folds `p` into `q` only once (final layer), so interior layers attend over
attention-blind `q`.

## Implementation (committed)

**glades-ml (branch `chiron3`):** CPU references (`drift_into_q`/`drift_backward` in
`transformer_chiron_ops.h`) + GPU kernels (`chiron_drift_into_q` fwd/inv, `chiron_drift_backward`
in `gpu_chiron.cu`/`.h`; the backward reuses the shipped `chiron_reln_backward_reanchor` for
dp/dgamma/dbeta and a deterministic column-reduce for da). Five unit tests, all PASS:

| Test | Result |
|---|---|
| FD grad-check (CPU backward vs forward) | maxRelErr 0.0012 (bar 2e-2) |
| CPU/GPU forward parity | 1.19e-07 (bar 1e-4) |
| forward∘inverse reversibility | 5.96e-08 (bar 1e-5) |
| CPU/GPU backward parity (dp,da,dγ,dβ) | 4.47e-08 (bar 2e-4) |
| backward accumulation regression (2× on double-call) | PASS |

The accumulation test caught a real `dp`-overwrite bug (the reanchor's `dq_in` path *assigns*; fixed
by routing through scratch + `axpy` so the drift's dp ADDS onto the downstream adjoint).

**glades-trainer (branch `reln-reanchor`):** `a_drift[l]` param (init 0, Adam, checkpoint bit 512);
`--per-layer-drift`/`--drift-warmup` flags; allocation gating broadened so the params allocate on
the flagship's `--no-fuse-attn --fuse-attn-reln` path; forward+inverse+backward wired in all
attention branches; val-mode full-strength drift (`isTraining` gate on the warmup); `[obsd]` budget
log; and a `--per-layer-drift`/`--fuse-attn-per-layer` mutual-exclusion guard (they share
`gamma_p`/`beta_p`).

## SCFA-small-shape segfault — root-caused & fixed (a pre-existing bug, unrelated to OBSD)

Small-scale iteration via SCFA was blocked by a SIGSEGV. Systematic diagnosis:
- Reproduced: `--scfa` at small shape → SIGSEGV in `scfa_attention_backward` (gdb); no-`--scfa`
  control trains fine.
- Characterized: crashes at **all** k (compression ratio), **all** T (incl. production T=16384),
  **all** m, **all** L — so NOT a shape bug.
- Root cause: the repros omitted `--bf16-weights`. SCFA's backward consumes bf16 weight scratch
  (`scratch_qbf_w`/`sQbf16`/…) allocated only under `--bf16-weights`; without it the backward
  dereferences unallocated memory. Confirmed: adding **only** `--bf16-weights` fixes it at every
  shape; removing it (even without `--scfa-bf16-inner`) re-crashes. The production flagship always
  pairs `--scfa` with the bf16 stack, so it never hit this.
- Fix: a parse-time guard — `--scfa requires --bf16-weights` — fails fast with a clear message
  instead of segfaulting (glades-trainer `chiron_main.cpp`, committed). Verified: bad combo →
  clean exit 1; good combo → trains.

## Validation results (small / moderate scale)

**E0 — a=0 bit-identity (no forward regression).** With `--per-layer-drift` at a=0, the loss is
bit-identical to the flagship path, on both the non-SCFA path and the production SCFA path. (Once
the backward is wired, the reported global `‖g‖` differs by ~1e-3 because `a_drift` legitimately
has a nonzero gradient at a=0 — `da = Σ scale·tanh(x̂)·dq` — i.e. the ReZero gate starts learning;
the *model function* is unchanged.)

**E2 — reconstruction / reversibility in training.** Short run with the drift active: 0 grad-skips,
bounded ‖g‖, finite decreasing loss, and `a_drift` provably learns from 0 (steps before the warmup
kicks in are bit-identical to a no-drift baseline, then diverge monotonically). A broken inverse
would have corrupted reconstruction → divergent grads; instead grads stayed bounded.

**E1 — stability at the production geometry L=24, m=2048 (T=2048, full bf16+SCFA+reanchor stack):**

| | OBSD `--per-layer-drift` (40 steps) | Naive linear `--fuse-attn-per-layer` (12 steps) |
|---|---|---|
| grad-skips | 0 | 0 |
| ‖g‖ | bounded 2–5 | 6.9 → 10.1 (rising) |
| loss start→end | 10.83 → 9.25 (best 8.97) | 10.89 → 10.82 (stuck) |
| clip scale | 0.10–0.33 | 0.05–0.07 (heavy clip) |
| budget B (`[obsd]`) | smooth ramp 0.012 → 1.36 | — |

OBSD is the clearly healthier mechanism: bounded gradients, fast loss descent, controlled budget,
gate learning. The naive linear coupling has rising ‖g‖, a stuck loss, and heavy clip-crushing.

**Caveats (honest):**
- The *catastrophic* naive step-1 explosion (‖g‖ 1e6–1e18) documented historically is a **T=16384
  phenomenon** — it scales with the attention-Jacobian norm ‖A‖, which grows with T. At T=2048 the
  naive path is unhealthy-but-not-exploding, so this E1 shows the *trend*, not the full blow-up. The
  definitive explosion contrast needs production T (E3/E4 territory).
- Reanchor vs no-reanchor was **bit-identical** for the naive coupling at this scale, confirming the
  design's claim that the coupling-feedback instability is **separate** from the q-side reln-backward
  overflow that reanchor cures (reanchor is necessary-but-not-sufficient for stable coupling).
- These are **stability/sanity** results, NOT perplexity results. Loss numbers at 40 steps / T=2048
  are not a val-NLL claim.

## Production-scale stability (de-risk probe) — PASS

Short full-T=16384 run (50 steps, full flagship recipe + `--per-layer-drift --drift-warmup 40`):
**0 grad-skips, ‖g‖ bounded 0.87–3.4**, loss descending 10.77→8.73, `[obsd]` B ramping to 1.31,
tok/s ~24.3k, VRAM ~15.3 GB (fits). **This is the regime where the naive per-layer coupling
catastrophically exploded (‖g‖ 1e6–1e18 at step 1); OBSD is stable.** Stability half of FM-1
confirmed at production geometry.

## E3 — budget sweep (PASS: OBSD lowers val NLL)

Production shape (T=16384, L=24, m=2048), full flagship recipe, 2500 steps, matched seed 1337,
wide val (8 batches, 8.39M tok):

| config | val NLL @ 2500 | ppl | acc1 | grad-skips | gate maxA / B |
|---|---|---|---|---|---|
| baseline (no drift) | 3.7409 | 42.14 | 0.1318 | 0 | — |
| **OBSD `--drift-warmup 250`** | **3.7038** | **40.60** | 0.1324 | 0 | 0.456 / 12.4 |
| OBSD `--drift-warmup 1000` | 3.7088 | 40.80 | 0.1332 | 0 | 0.541 / 12.8 |

**Both OBSD configs beat the baseline at matched steps; best `--drift-warmup 250` = −0.0371 nat**
(w1000 = −0.0321). 0 grad-skips throughout; the gate learns to a_drift ~0.46 (not stuck at 0), so
the coupling is genuinely active. **B\* ≈ warmup 250.** Caveats: this is a 2500-step *sign*, not the
66k ship number — whether −0.037 nat holds/grows/shrinks to full training is the E4 question; the
margin is modest so far (cf. reanchor −0.62, data-scale −0.9, both at full training).

**Wall cost:** OBSD ~+12% (tok/s 24.3k vs 27.3k) to ~+19% (wall-clock incl. fixed overhead) — above
the design's +8–10% estimate. The uncoalesced `chiron_col_accumulate` reduction (flagged in review)
is a likely contributor and is optimizable (2-phase tiled reduction) if wall matters at ship time.

## E3 extension (de-risk to 7500) — PASS: gap robust & non-shrinking

Resumed base + w250 from their 2500 checkpoints to 7500 (val every 625, 8 batches), **0 grad-skips
in both**. OBSD beats baseline at **all 9 checkpoints**:

| step | base | w250 | gap | step | base | w250 | gap |
|---|---|---|---|---|---|---|---|
| 2500 | 3.706 | 3.654 | +0.052 | 5625 | 3.585 | 3.536 | +0.049 |
| 3125 | 3.673 | 3.628 | +0.045 | 6250 | 3.717 | 3.677 | +0.040 |
| 3750 | 3.719 | 3.663 | +0.057 | 6875 | 3.835 | 3.776 | +0.059 |
| 4375 | 3.799 | 3.759 | +0.040 | 7500 | 4.116 | 4.045 | **+0.071** |
| 5000 | 3.628 | 3.574 | +0.055 | | | | |

Mean gap ~+0.052 nat, largest at the end (+0.071) — **robust, not a fluke, holding/slightly
growing.** Both trajectories share correlated val-window noise (the 7500 spike hits both, cancels in
the gap). **Gate strengthening (watch-point):** OBSD `maxA` 0.46→**1.11**, ‖g‖ →~0.8 (moderate clip,
scale ~0.6), B proxy 12→18 — heavily used, still 0 skips. Monitor for grad-skips as the gate grows
through 30k.

## E4 — RESULT: NEGATIVE (OBSD regresses at scale; stable but not a perplexity win)

Both runs: random-init → 30k, T=16384, same local fineweb data, seed 1337, **identical recipe
except `--per-layer-drift`**. Wide val (16 batches).

| metric @ 30k | OBSD w250 | baseline |
|---|---|---|
| val NLL | 3.27 | **2.61** |
| acc1 | 0.14 | **0.32** |
| grad-skips | 0 | 0 |

**Matched val trajectory — a clear crossover, not noise.** OBSD ahead **+0.05–0.07 nat through
~step 16.5k**, gap closes ~19.5k, then the baseline pulls ahead monotonically:
−0.03 (21k) → −0.08 (24k) → −0.12 (27k) → **−0.66 (30k)**. The late divergence is real, not a val
artifact: the **baseline's train loss also dropped** 3.0→2.6 (train acc 0.22→0.31) in the 27–30k
window — a genuine late-training acceleration — while **OBSD stalled** at train loss ~3.2–3.5.

**Verdict — two findings, kept distinct:**
- **STABILITY ✅** — 0 grad-skips through the 1.6–2B instability regime (24.4k–30k); ‖g‖ bounded.
  OBSD's coupling does **not** destabilize where the flagship died at 1.49B without the cure
  (bounded-φ + reanchor held). The instability-regime question is answered: yes, stable.
- **PERPLEXITY ✗** — OBSD helps *early* but **regresses −0.66 nat by 30k**. **Does NOT ship as
  configured.**

**Root cause (well-supported):** the gate grew **unbounded** (maxA 0.46 → 1.8). OBSD won while
maxA ≲ 1.1 and lost as it grew large — the growth inflated ‖g‖ → heavier grad-clipping (OBSD clip
scale 0.5–0.7 vs baseline 0.7–1.0) → throttled *effective* LR → OBSD **missed the baseline's
late-training acceleration**. The reversal tracks the gate growth precisely. (A softer cousin of the
design's FM-1: not an explosion, but the budget growth quietly taxing convergence.)

## Salvage (in progress 2026-06-29): bounded gate

The design's deferred "optional budget projection" now looks **necessary, not optional**. Testing a
hard gate cap (`--drift-gate-cap V`, elementwise `|a_drift| ≤ V`): resume OBSD from the step-20000
checkpoint (just past the reversal onset, maxA ~1.65) with `--drift-gate-cap 0.5`, run to 30k —
does bounding the gate let OBSD recover the late acceleration? Target: baseline@30k = 2.61; un-capped
OBSD@30k = 3.27. Resume saves the early steps; the reversal window (must be run) is ~20k→30k (~7.5 hr).

## What remains: E4 — the quality gate (the unsettled ship question)

- **E4 — quality gate** (production shape, **30k steps** — owner decision 2026-06-28: 30k is
  sufficient, no 60k — single seed 1337): full flagship recipe + `--per-layer-drift
  --drift-warmup <best>`, vs a **matched flagship-recipe baseline at 30k** (resume both from the
  7500 extension checkpoints `ext_{base,w250}.final`).
  - **Criterion (note: the absolute 1.92 is a FULL-training endpoint, NOT a 30k number — at 30k
    both runs are mid-training):** validate iff OBSD@30k beats the matched baseline@30k by a clear,
    non-shrinking margin with **0 grad-skips** and reconstruction within BF16 ULP. A confirmed +Δ at
    30k means the mechanism is a real improvement; the production checkpoint (to reach the lineage's
    ~1.92-class endpoint) would then be a separate full-deployment training run.
  - If the margin shrinks toward 0 by 30k or stability degrades → record the negative (FM-1
    realized: stable but not a perplexity win); leave `--per-layer-drift` default-off, documented.
- **Multi-seed confirmation** capped at 5k–15k steps/seed (`{2024,4242}`) — sign + cross-seed
  stability only (owner cost-control decision).

**Reproduce (E4 primary, single seed):** flagship recipe (CLAUDE.md) +
`--no-fuse-attn --fuse-attn-reln --per-layer-drift --drift-warmup <best>`.

## Files / commits
- Spec/plan: `docs/superpowers/specs|plans/2026-06-27-chiron-richer-symplectic-block*.md`.
- glades-ml kernels+tests: branch `chiron3` (CPU ref → forward → backward → review fixes).
- glades-trainer integration + SCFA guard: branch `reln-reanchor`.
