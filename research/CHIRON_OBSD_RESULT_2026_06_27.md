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

## What remains: E4 — the full quality gate (the unsettled ship question)

- **E4 — quality gate** (production shape, ~60k steps ≈ 1–1.5 days, single seed 1337): full flagship
  recipe + `--per-layer-drift --drift-warmup <best>`. **Ship iff val NLL < 1.92, 0 grad-skips,
  wall ≤ +10%, reconstruction within BF16 ULP.** If it does not beat 1.92 → record the negative
  (FM-1 realized: stable but at flagship perplexity); leave `--per-layer-drift` as a default-off,
  validated, documented flag.
- **Multi-seed confirmation** capped at 5k–15k steps/seed (`{2024,4242}`) — sign + cross-seed
  stability only (owner cost-control decision).

**Reproduce (E4 primary, single seed):** flagship recipe (CLAUDE.md) +
`--no-fuse-attn --fuse-attn-reln --per-layer-drift --drift-warmup <best>`.

## Files / commits
- Spec/plan: `docs/superpowers/specs|plans/2026-06-27-chiron-richer-symplectic-block*.md`.
- glades-ml kernels+tests: branch `chiron3` (CPU ref → forward → backward → review fixes).
- glades-trainer integration + SCFA guard: branch `reln-reanchor`.
