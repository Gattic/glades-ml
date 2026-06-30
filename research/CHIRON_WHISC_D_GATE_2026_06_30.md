# CHIRON WhiSC-D — E3 Production Gate: PASS

**Date:** 2026-06-30
**Status:** E3 PASS (divergence falsifier satisfied). E4 (30k decisive perplexity gate) NOT yet run.
**Mechanism:** WhiSC-D (Whitened Symplectic Coupling, diagonal). Design spec
`docs/superpowers/specs/2026-06-30-chiron-whitened-frame-coupling-design.md`; plan
`docs/superpowers/plans/2026-06-30-chiron-whisc-d.md`. Builds on (fixes) **SORC**
(`research/CHIRON_SORC_RESULT_2026_06_30.md` — CLOSED NO-GO, diverged at the same gate).

---

## TL;DR

WhiSC-D conjugates the SORC per-channel rotation coupling by a **detached per-channel
whitening** `Φ = W⁻¹R(θ)W`, `W=diag(1/a,a)`, `a=(E[q²]/E[p²])^{1/4}` (per-channel EMA,
out of the autodiff graph). On the **same matched single-seed 2500-step T=16384 gate that
SORC diverged**, WhiSC-D **completed cleanly**: ‖g‖ O(1) (max 3.54), **0 grad-skips, 0 NaN**,
val@2500 **2.78** — while the un-whitened SORC hit val 14.45 / ‖g‖ 4.7e10 / 1619 skips and
exploded at step ~831. The `[whisc]` monitor confirms the phase ratio `ρ_eff` grew to **~3000**
(past the SORC-killing regime) and the whitening absorbed it. The R1 design claim
(backward q-gradient bounded independent of ‖p‖/‖q‖) holds in vivo.

**Two open items before any ship:** (1) a **−37% wall-throughput** regression from the
unoptimized stats kernel (perf follow-up below); (2) the perplexity win needs a
matched-2500 / E4 confirmation.

---

## E3 result (matched single-seed, T=16384, seed 1337, reanchor recipe + `--whisc-coupling --rot-theta-max 0.07`)

| metric | **WhiSC-D** | SORC (un-whitened) | baseline (no coupling) |
|---|---|---|---|
| outcome @ 2500 | **completed** | diverged @ step ~831 | completed |
| ‖g‖ (min / max / last) | **0.50 / 3.54 / 1.06** | → **4.7e10** | ~O(1) |
| steps with ‖g‖>10 | **0** | many | 0 |
| grad-skips | **0** | 1619 | 0 |
| NaN / loss-scale→0 | **0** | yes | 0 |
| **val nll @ 2500** | **2.7849** (ppl 16.20, acc1 0.325, acc5 0.745) | 14.45 | 3.5803 |
| wall (2500 steps) | ~2.53 GPU-hr (9122 s) | — | ~1.6 GPU-hr |

**Stability is the gate criterion and it is unambiguous: PASS.** WhiSC-D fixes the SORC
divergence categorically.

### Mechanism confirmation (the `[whisc]` / `[sorc]` monitors)
- `ρ_eff` (per-channel `E[p²]/E[q²]`) grew **~1 (step 1) → ~3000 (step 2491)** — i.e. training
  drove the phase asymmetry into and past the regime that exploded SORC.
- The whitening tracked it: `a` (= `ρ^{-1/4}`) ranged **[0.135, 2.18]** at step 2491 — a channel
  at ρ~3000 is whitened by a≈0.135.
- The learnable coupling angle `maxTheta` grew **0 → 0.0415** (under the 0.07 cap) — gentle and
  bounded, **not** the runaway SORC's joint-norm rotation produced.
- ‖g‖ stayed O(1) throughout despite ρ_eff~3000 — exactly the R1 prediction (the differentiated
  signal is O(‖q‖), the scale gap lives in the detached frame, the backward is non-compounding).

## Perplexity (secondary — real, corroborated, but wants a matched-2500/E4 confirm)
- **This-session matched baseline** (no-whisc, identical recipe/seed, killed ~step 1000):
  val@1000 = **4.21** vs WhiSC val@1000 = **4.04** → **−0.17 nat** (clean same-session control).
- **Prior SORC-arc baseline**: val@2500 = **3.58** vs WhiSC val@2500 = **2.78** → **−0.80 nat**;
  the gap **widens** with training (consistent with cross-depth attention composition — the
  mechanism's intended benefit — helping increasingly over depth/time).
- Calibration: the −0.17 (matched) is solid; the −0.80 (vs a prior run) is strongly suggestive
  but not yet banked. A matched no-whisc 2500-step run this session, and/or the E4 30k gate,
  would establish it rigorously.

## Wall-throughput — PERF PASS DONE (folding recovered the bulk; −37%→−22%)

**Resolved 2026-06-30 (post-E3).** An nsys profile **corrected the earlier hypothesis**: the
dominant cost was NOT the stats kernel (3.7%) but the **explicit `whiten/unwhiten` scale passes
(`whisc_scale_kernel` 19.1% of GPU time, ~768 passes/step)** — a consequence of the plan's
explicit `whiten∘rot∘unwhiten` composition. The fix was **coefficient folding** (the spec's
original §5.2 design): fold `a²` into the rot coeffs (`A=a²·sorc_a`, `C=sorc_c/a²`) so the
whitening rides the existing rot kernel, eliminating all 768 scale passes. The folded `dθ` chain
(`a²`-aware) was FD-validated at ρ=45/300/3000 (rel-err ≤2e-4). **Result: throughput 17,930 →
22,200 tok/s (+24%); slowdown −37% (×1.62) → −22% (×1.28).** E0 identity preserved (folded
φ=0 ⇒ A=C=0 ⇒ identity, loss 10.7612 exact). Committed: lib `c83303e9c`, trainer `6e429d6`.

A second pass **coalesced the stats kernel** (32 transactions/warp → 1; committed `18be3228d`,
parity worst ~1e-5) — correct, but it gave **~0 end-to-end gain** (22,170 ≈ 22,200): the stats
kernel's GPU time is **overlap-hidden** at this workload (CPU dispatch / other-GPU-work overlap),
not on the wall-clock critical path. The remaining −22% is the **inherent rotation kernels**
(~12.5%, shared with SORC) plus overlap-limited overhead; further micro-opt would also be
overlap-hidden (measured). The folding pass extracted the available room.

### (historical) Original wall finding (now addressed by folding):
- WhiSC-D **~17,930 tok/s** vs matched baseline **~28,500 tok/s** → **−37% (×0.62), +1.4 s/step
  (×1.62)**. (The design spec predicted ≈parity — that claim under-accounted for the new stats
  kernel; correcting the record here.)
- **Suspect (unprofiled hypothesis): `chiron_whisc_update_stats`** does a per-channel column
  reduction over the token axis, reading `q[t*m+i]` with **stride m (8 KB) → warp-uncoalesced**
  (~32× memory-traffic waste), launched 24 layers × 4 accum = 96×/step. The `whisc_scale`
  whiten/unwhiten/adjoint passes are coalesced and cheap; the column reduction is the trap.
- **Fix (contained, before any ship):** fuse the q²/p² accumulation into the **whiten pass**
  (which already reads `(q,p)` coalesced), or a row-major two-pass / transposed reduction.
  Orthogonal to correctness — does not affect the E3 verdict.

## Engineering record
- Committed **default-off**, fully reviewed (per-task + final whole-branch = merge-ready, 0
  Critical/0 Important). glades-ml: `whisc_scale`/`whisc_update_stats` CPU refs + GPU kernels +
  unit tests incl. the ρ=45 finite-difference backward-parity guard (0.033%). glades-trainer:
  `--whisc-coupling` (reuses SORC `rot_phi`/Adam/checkpoint-bit-1024/`[sorc]` monitor), the
  `whiten→rot→unwhiten` forward + composite-adjoint backward + `[whisc]` monitor + run.sh
  passthrough.
- Validation ladder: E0 bit-identity (loss 10.7612 exact at φ=0) ✓; E1 unit tests ✓; E2 small-
  shape ✓; **E3 PASS (this doc)**; E4 (30k) not run.
- Accum `a`-consistency confirmed by the final review (interleaved forward_k→backward_k; holds
  under CUDA graphs).

## Next steps
1. **Perf:** coalesce/fuse `chiron_whisc_update_stats` (the −37% regression) — required before ship.
2. **Confirm perplexity:** matched no-whisc 2500-step baseline this session (≈2 GPU-hr) and/or the
   **E4 30k decisive gate** vs matched baseline.
3. If both hold: WhiSC-D becomes a ship candidate (the first cross-depth coupling that both
   survives the instability AND improves perplexity — OBSD regressed, SORC diverged).

## Verdict
**E3 PASS.** WhiSC-D is the first per-layer cross-depth coupling to clear the production
divergence gate for CHIRON. The whitening fix works: forward norm-preservation was never the
issue; per-subspace scale control of the perturbation, with the statistic kept out of the
autodiff graph, is. The −0.17/−0.80 nat perplexity signal is a promising bonus pending E4. Two
open items (stats-kernel perf, perplexity confirmation) gate a ship; neither affects the
stability verdict.
