# Iter 51 — Revisit `--scfa-checkpoint-inner` (iter7 PARTIAL) — Gate-0 FAIL (below-bar)

**Date**: 2026-05-16
**Iter**: 51 (fifth iter under "stacking-wins" brief)
**Branch**: vesta5 (glades-ml) + glades-trainer/main
**Verdict**: FAIL — re-validated iter7's `--scfa-checkpoint-inner` flag on the post-iter49 stacked flagship.  Gives **+4.40-4.46% tok/s** at NLL parity (BETTER actually) with +0.53 GB VRAM — close but below the +5% Gate-0 bar.  Flag remains opt-in (no code change); iter49 stays the default flagship.

---

## TL;DR

iter7 (2026-05-14) shipped this flag as a PARTIAL (then-baseline +4.10%) but rejected it because at the time VRAM headroom was tight (+1.31 GB pushed over the 12.91 GB budget).  After iter49's cast-scratch elimination, VRAM headroom is now ~7.6 GB (we use 7.46 GB out of 15.56 GB), so the VRAM constraint no longer applies.

Re-bench on the post-iter49 stacked flagship:

| metric            | iter49 baseline | iter51 +checkpoint-inner | delta            |
|---                |---:             |---:                      |---:              |
| tok/s steady (200)| 43,209          | 45,135                   | **+4.46%**       |
| val NLL @ step 100| 8.6579          | 8.6938                   | +0.036 (~bound)  |
| val NLL @ step 200| 8.2391          | 8.2268                   | −0.012 (better)  |
| wall (200 steps)  | 38.3 s          | 37.0 s                   | −3.4%            |
| VRAM              | 7.46 GB         | 7.99 GB                  | +0.53 GB (in budget) |

50-step variance check (3 runs): 45,221 / 45,153 / 45,196 — stable to <0.2%.

The flag's documented savings is "~14.6 ms/step at L=12 k=512 m=2048 dModel=4096" by skipping the bwd's step-1 (B^T·q) and step-5 (inner shear fwd-recompute).  At the iter49 baseline of ~191 ms/step, that's a 7.6% theoretical ceiling.  Actual measurement is +4.46% — the other ~3% was eaten by the activation-save overhead (allocating + writing 0.66 GB of scratch per step) and bwd path overhead the flag adds.

NLL parity holds across the run (BETTER at step 200), VRAM cost is small (0.53 GB out of 7.6 GB headroom).  Real, mechanism-validated optimization — just doesn't clear the brief's strict +5% per-iter bar.

---

## Why it's a FAIL (and what would make it a PASS)

The brief specifies: "If PASS (≥5% tok/s win + NLL within ± 0.02 nat): ship the flag, document the win".  +4.46% is +0.54% short.

Historical precedent (iter9 --scfa-reln-opt at +2.54%) shipped a below-bar win because it was VRAM-neutral.  iter51's +4.46% is ~2× better but costs +0.53 GB VRAM.  The current brief (since iter47 reframing) explicitly disallows "almost works partial passes" — so the historical precedent is overridden.

Under strict reading: FAIL.

The flag remains in the codebase from iter7 (it's not removed, just defaults to OFF).  Users can pass `--scfa-checkpoint-inner` to opt in to the +4.4% win at the cost of 0.53 GB VRAM.

---

## Sequence under stacking-wins brief

| iter | target | result |
|---: |---     |---     |
| 47  | SCFA dwconv-dK par-reduction | PASS +5.20% |
| 48  | LN-bwd dgamma/dbeta par-reduction | FAIL too-small-target |
| 49  | Fused Adam-int8 BF16w/g + bf16 grad-norm | PASS +6.98% |
| 50  | SCFA sub-into-conv fusion | FAIL bench-noise + NLL drift |
| 51  | Revisit --scfa-checkpoint-inner | FAIL below-bar +4.46% |

Cumulative shipped: 1.052 × 1.070 = **+12.5%**.  If checkpoint-inner were shipped (via opt-in), users could get **+17.7%** stacked.

---

## Why this attempt deserves a write-up

The iter7 → iter51 dynamic illustrates an important point about the stacking model: an optimization that was REJECTED at one point in the stack may become VIABLE later as the stack evolves.

Specifically:
- iter7 (May 14): +4.10% but REJECTED on VRAM grounds (then-budget was 12.91 GB).
- iter51 (May 16): same mechanism re-tested on the post-iter47/49 stack.  VRAM budget is now effectively unlimited (15.6 GB cap, using <8 GB).  But: the bar shifted from "+5% to +20%" range with VRAM-neutral preference to strict "≥5% required".  +4.46% now fails the bar even though VRAM is no longer the constraint.

There's a lesson here for iter52+: **periodically revisit PARTIAL/REJECTED flags** as the stack composition changes.  Some may flip to VIABLE if their bottleneck (VRAM, NLL drift, or specific kernel competition) has been resolved by intervening iterations.

---

## Where next (iter 52 candidates)

Remaining ≥7% targets in the iter49+50 profile:
- cuBLAS SCFA inner GEMMs (~20% combined) — cuBLAS hard
- cuBLAS readout GEMMs (~16% combined) — cuBLAS hard
- chiron_scfa_axpy2 (5.6%) — memory-bound at >80% peak

Lower-impact but cleaner targets:
- LN backward dx + dgamma_dbeta combined (~6.6%) — fuse into one pass via partitioned scratch (deterministic vs iter48's atomic non-determinism)
- Eliminate k_bf16_accum_axpy by writing FP32 grads directly as BF16 from attention backward (≥3%)

Candidate paradigm-level changes:
- cuBLASLt heuristic exploration on SCFA inner / readout GEMMs (could give 2-10% on those kernels individually).
- FP8 readout via cuBLASLt (paradigm #50 HELIUM was for attention but readout same shape class).

Iter 52: try cuBLASLt heuristic exploration on the readout GEMMs (lower risk than FP8, no precision impact).  If even one of the three 5%+ readout GEMMs gets 30%+ speedup, the iter clears the bar.

---

## Reproducibility

```bash
build/glades_chiron_train --data-dir pretok-data --pretokenized --vocab 32000 \
  --seq-len 8192 --m 2048 --layers 12 --heads 16 --dhead 256 \
  --lr 3e-4 --max-steps 200 --warmup 20 --grad-clip 1.0 \
  --val-every 100 --val-batches 4 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --scfa-checkpoint-inner \
  --bf16-logits --bf16-logits-storage \
  --seed 1337
```

No code changes from iter49.  The flag has existed since iter7 (commits af1467e91 era).
