# Program Strategic Pivot — Iter 15 (2026-05-15)

After 4 ralph-loop iterations on paradigm #255 DSA's gate-driver formula
(iter 11 design, 12 prototype, 13 Σ-defect FAIL, 14 frame-defect partial),
the honest empirical state of the Cellular Sheaf Attention program is:

| Paradigm | Mathematical design | Empirical state |
|----------|---------------------|-----------------|
| #250 SFA | Full math, proofs (iter 1-2) | **VALIDATED at one layer** (Phase 8b: −0.60 nat) |
| #251 SRA | Full design (iter 3) | NOT IMPLEMENTED |
| #252 PSA | Full design (iter 4) | NOT IMPLEMENTED |
| #253 SLR | Full design (iter 6) | NOT IMPLEMENTED (depends on #252) |
| #254 CSR | Full design (iter 7) | NOT IMPLEMENTED |
| #255 DSA | Full design (iter 11), refined post-iter-13 | **GATE-DRIVER FORMULA PARTIALLY FALSIFIED** |

**The brief's "magnitudes" target is not reachable on the current trajectory.**
Validated improvement is one paradigm × one layer × −0.60 nat. The 10–20×
wall-clock projection assumed validation of all 5 designed paradigms plus
stacking; we have validation of one slice of one paradigm.

## Why DSA iteration is hitting a wall

The DSA paradigm posits that the cocycle gain in Phase 8b is *position-
stratified* in some closed-form signal computable from SFA parameters
(Σ, U, or the residual stream). Empirically:

- **Σ defect (iter 13)**: flat across positions (spread 1%, Pearson r = −0.19).
- **U-frame defect (iter 14)**: still flat magnitude (spread 1%) but
  correlation jumps to +0.475 — *almost* at the 0.5 threshold but the
  ratio test still fails decisively.

Even if iter-15's 2000-step run sharpens the frame defect to ratio ≥ 2× and
r ≥ 0.5, the magnitude claim from a working DSA gate would be **1.3-1.6×
wall-clock at iso-NLL over uniform SFA** (per `PARADIGM_SHIFT_255_DESIGN.md`
§4.2). Compounded with the rest of the unvalidated stack, this is in the
"few-x" not "magnitudes" range.

## Three honest paths forward

### Path 1: Test SFA's compounding claim directly (multi-layer SFA)

The most direct lever toward magnitudes is whether SFA's single-layer
−0.60 nat gain **compounds** when stacked. If 4 layers gives ≈ −2.4 nat,
that's a real magnitude (10× perplexity improvement). If it saturates at
1.0-1.5 nat, the magnitudes claim is wrong and the program needs a
fundamentally different attack.

Cost: ~200-400 lines of trainer refactor for multi-layer SFA support
(per-layer SFA params + grad/Adam state buffers, multi-pass fwd/bwd).
Two empirical runs: 4-layer SFA (L=12,15,18,21) and 4-layer NO-OP control,
2000 steps each, ~70 min wall total.

Information value: HIGHEST — directly addresses the magnitudes claim with
a single experiment.

### Path 2: Validate or falsify the cocycle premise (trained-SCFA control)

Phase 8b's −0.60 nat could come from cocycle expressivity (paradigm #250's
claim) OR simply from adding ~50 K extra trainable parameters at one layer.
A trained-SCFA control replaces SFA with a same-FLOP SCFA-variant block
that's also trained for 2000 steps at the same lr. If THAT gives −0.6 nat,
SFA's cocycle claim is falsified empirically.

Cost: one trainer flag (`--sfa-control-train` that uses SCFA params at the
swap layer with the same training schedule). ~50 lines of code. One 2000-step
run, 33 min wall.

Information value: VERY HIGH — could collapse the entire CSA program if
the control matches SFA's gain.

### Path 3: Implement SRA (#251) and validate its claim

SRA is mathematically cheapest of the unvalidated paradigms (closed-form
per-query resolvent). Its projected 4.3× wall-clock factor is essential
to the stack's magnitudes claim.

Cost: SRA kernels (resolvent, multi-pole combiner) + per-query pole MLP +
trainer wiring. ~500-800 lines new code, 4-6 iterations.

Information value: HIGH but ENGINEERING-HEAVY — defers magnitude validation
by 4-6 iters of implementation work before the empirical question is even
asked.

## Recommended pivot

**Order: Path 2 → Path 1 → Path 3.**

1. **Path 2 first (cheapest, kill-criterion test)**. If trained-SCFA control
   matches SFA's gain, the program's cocycle premise is falsified, and
   chasing #251-#255 is wasted effort.
2. **Path 1 if Path 2 confirms cocycle reality**. Multi-layer SFA tests the
   compounding claim that drives the magnitudes projection.
3. **Path 3 only after Path 1 shows compounding is real**. Implementing SRA
   without first knowing SFA compounds is premature.

## What to do with the in-flight DSA work

- **Keep the infrastructure**: defect kernels, parity tests, --sfa-defect-stat
  flag are all working and useful for future analysis.
- **Mark paradigm #255 status as "deferred pending Path 1-3"**: the gate-
  driver question is interesting but not on the critical path to magnitudes.
- **Re-engage DSA later** if multi-layer SFA shows benefits but suffers from
  position-1-like regressions that gating could fix.

## The honest read on iter 11-14

The design depth was *over-allocated* relative to empirical validation
budget. Five paradigm designs (#250-#255) totalling >2000 lines of research
notes were produced before any was empirically validated at the *stacking*
level. The result is a richly documented framework whose central claim
(magnitudes via stacking) remains untested.

This iter recommends spending the next ~4-6 iterations on **empirical
validation of compounding**, not on extending the design space further.

## Decision matrix for ralph-loop continuation

If the user wants:
- **Quick falsification or validation of the program** → run Path 2.
- **Direct test of the magnitudes claim** → run Path 1.
- **Continued DSA refinement** → wait for iter-15 2000-step result; if it
  fails to sharpen U-frame signal, move to Path 1 or 2 anyway.
- **Stop the loop until results land** → /cancel-ralph and re-engage when
  ready.

## Files

- `research/SFA_PHASE8B_LONG_TRAIN_RESULT.md` — the one validated data point.
- `research/DSA_PROBE_O_FLAGSHIP_RESULT.md` — iter-13 Σ falsification.
- `research/DSA_PROBE_O_FRAME_RESULT.md` — iter-14 frame near-threshold.
- `research/PARADIGM_SHIFT_25{0..5}_DESIGN.md` — the five paradigm designs.
- `research/CELLULAR_SHEAF_ATTENTION_PROGRAM.md` — program overview, magnitude projection.

## Reproducer for Path 2 (cheapest first experiment)

Requires adding `--sfa-control-train` flag (one bool) and a conditional in
the swap-layer training path that, when set, leaves the layer as plain
SCFA and adds gradient/Adam-state buffers for the per-layer SCFA params
(SCFA B, D buffers if any layer-specific) — trained for the same 2000
steps at sfa-lr=1e-3, lr=0.0 (flagship frozen).

Cost: ~50 lines of trainer code + 33 min of wall time.
