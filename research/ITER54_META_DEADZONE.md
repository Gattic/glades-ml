# Iter 54 — Meta-analysis: the +5% bar is empirically unattainable on the current bottleneck composition

**Date**: 2026-05-16
**Iter**: 54 (eighth iter under "stacking-wins" brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: META — after 5 consecutive non-PASS iters (48-53), the empirical evidence is clear: **no single-optimization ≥+5% target remains** in the iter49 stacked flagship.  Documenting the dead-zone map and proposing three strategic pivots.

---

## TL;DR

After iter49's PASS at +6.98%, the next 5 iters (48 was before iter49 but failed; 50-53 followed) all failed to clear the +5% bar:

| iter | target                                   | result | best metric |
|---: |---                                        |---     |---:         |
| 48  | LN-bwd dgamma/dbeta T_PARTS=1            | FAIL   | +2.4%       |
| 50  | SCFA sub-into-conv (fwd only)            | FAIL   | +1.31% (in noise) |
| 51  | --scfa-checkpoint-inner revival          | FAIL   | +4.46% (close!) |
| 52  | cuBLAS readout algo override             | NULL   | 0%          |
| 53  | LN-bwd dgamma/dbeta deterministic 2-phase | FAIL  | +2.86%      |

The non-PASS iters cover EVERY remaining ≥4% kernel slice and several combinations.  Pattern: **the remaining bottleneck composition consists entirely of either (a) cuBLAS-tuned GEMMs that auto-pick is already optimal for, (b) memory-bandwidth-bound element-wise kernels already at 80-98% of peak bandwidth, or (c) kernel slices too small (<5% individually) for any solo optimization to clear the bar**.

This is a structural property of the current iter49-flagship state.  No engineering-style iter can flip this without a paradigm change.

---

## Dead-zone map (current iter49 flagship)

```
TOTAL GPU TIME ALLOCATION (post-iter49):

  cuBLAS GEMMs (40%)              ← iter52 confirmed: DEFAULT auto-pick optimal
  ├ SCFA inner Q*K, S*V, projections (3 kernels: 7.7+6.6+6.3 = 20.6%)
  └ Readout fwd/bwd-dq/bwd-dE (3 kernels: 5.6+5.5+5.5 = 16.6%)
  
  SCFA element-wise FP32 (15%)    ← iter50 tried fusion, hit fp32-FMA precision drift
  ├ axpy2 (5.6%) — 98% mem-bandwidth peak
  ├ sub (4.3%) — 83% mem-bandwidth peak
  ├ scaled_copy (1.3%)
  └ axpy_kernel (2.6%)
  
  LayerNorm backward (6.6%)       ← iter48 + iter53 tried both atomicAdd and 2-phase
  ├ dgamma_dbeta (4.5%)
  └ dx (2.1%, already coalesced)
  
  Adam-int8 + bf16 chain (8%)     ← iter49 already saved ~6.5% from this category
  ├ adam_update_int8_state_bf16w_bf16g (5.1%, iter49 result)
  └ k_bf16_accum_axpy (2.8%) — could save 0.5-1% by special-casing accum=1
  
  Depthwise causal conv (8.6%)    ← iter47 already saved 5.2% (dK kernel rewrite)
  ├ fwd (3.4%) — well-parallelized
  ├ dK (3.0%, iter47 result)
  └ dx (2.2%) — well-parallelized
  
  Reln fwd+inverse (3.1%)         ← Already in-place, --scfa-reln-opt shipped iter9
  
  Other (~20%)                    ← Many small kernels, none >2%
    softmax, causal mask, argmax (val), cast-residue, ...
```

Of these slices:
- **40% (cuBLAS)** — empirically dead-zone (iter52 NULL).
- **15% (SCFA element-wise)** — memory-bandwidth-bound at ~90% of peak; iter50 tried fusion, ran into fp32-FMA precision drift.
- **6.6% (LN-bwd)** — too small for solo (iter48 + iter53 confirmed).
- **8% (Adam+cast)** — iter49 already saved the big chunk.
- **~30%** other small kernels, none individually ≥5%.

**There is no remaining single-target slice ≥5% that hasn't been tried, except via paradigm changes** (FP8 GEMMs, sparse attention, vocab subsampling, BF16 residual stream).

---

## Why "stack multiple sub-5% wins" doesn't naturally work under the brief

The brief specifically forbids "almost-works partial passes" — even +4.46% (iter51) is not shipped under strict reading.

But empirically, the remaining wins are all in the 2-5% range:
- iter51 checkpoint-inner: +4.46%
- iter53 LN-bwd 2-phase: +2.86%
- iter48 LN-bwd dgamma: +2.4%
- iter50 sub-conv: +1.31% (noise)
- bf16-accum special-case: ~0.7% (estimated)
- float4 vectorize element-wise: ~1.5-3% (estimated)

If we relaxed the per-iter bar to "ship any non-regressive engineering win that's mechanism-validated and NLL-parity", the cumulative could be:
- (1.046)(1.029)(1.015) = 1.094 → +9.4% from three small wins
- Stacked on iter47×iter49: 1.052 × 1.070 × 1.094 = 1.231 → **+23%**

That's a meaningful improvement over the current +12.5% but well below the 1.5×/5-iter target.

---

## Three strategic pivots for the user to choose from

### Option A: Relax the per-iter bar to "+2-5% engineering ships, NLL-parity required"

Pros: Empirically-aligned with what's available.  Stack 3-5 small wins to ~+25% total.
Cons: Departs from the brief's strict +5% framing.  Risks shipping noise as wins.

Conservative version: ship +3% threshold (instead of +5%).  This would have shipped iter51 (+4.46%) and iter53 (+2.86%).  Skip iter50/48 (in-noise).

### Option B: Pivot to paradigm-level changes (higher risk, higher upside)

Candidates:
1. **FP8 readout GEMMs** via cuBLASLt (paradigm #50 HELIUM extended to readout).  Potential 10-30% if precision survives.  Risk: NLL might regress.
2. **BF16 residual stream** (q, p, dp buffers in BF16).  Halves memory bandwidth for ~15% of GPU time.  Potential 5-8% but training stability is a big unknown.
3. **Sparse attention** (route a subset of tokens through full attn, rest through cheaper attn).  Architectural change.
4. **Vocab subsampling** during training (use 10K vocab subset per step instead of full 32K).  Reduces readout GEMM by 3×.  Algorithmic change.

These each have +5-30% potential but with NLL parity at risk.

### Option C: Step-back and update the brief

After 7 iters under the reframed "stacking-wins" model, +12.5% cumulative.  The 1.5×/5-iter target (+50%) was unmet; the 10×/20-iter target (+900%) is clearly out of reach with the current bottleneck composition.

The brief could be updated to:
- Lower the cumulative target (e.g., 2× over 20 iters instead of 10×).
- Change the "no partial passes" rule.
- Introduce a paradigm-change category alongside engineering wins.

This is the same kind of empirical-realignment that the iter47 reframing did to the earlier "magnitudes via architectural shift" target.  Five paradigm falsifications (#250 SFA, #260 IGAA, #261 HMTA, #262 MEDAL training, #262 MEDAL inference) led to that reframing; 5 below-bar engineering iters now suggest a second realignment.

---

## Recommendation

**Option A (relax to +3%)** is the most pragmatic given the empirical data.  Engineering wins in the 2-5% range exist; the +5% bar arbitrarily rejects them.  Shipping at +3% threshold would have already added 1.07 × 1.03 ≈ +10% more on top of iter49 (via checkpoint-inner + LN-bwd 2-phase).

**Option B (paradigm)** is higher risk but the brief's spirit favors high-upside attempts; this would mean a 5th paradigm-arc iteration.

**Option C (brief update)** is honest about the empirical limits but heavy meta-work; better deferred to a user check-in.

If the loop continues unchanged: iter 55 should attempt a small engineering win and accept the FAIL, OR commit to Option B with explicit user buy-in (the next iter's "candidate" comment would propose the paradigm change for user approval before implementation).

---

## What this iter does NOT change

- No code changes in this iter (no kernels added, no flags toggled, no defaults flipped).
- Default flagship stays at iter49 (post-PASS state).
- All previous PASSed iters' code remains shipping.
- All previous FAILed iters' reverts remain in place.

This iter's only output is the meta-analysis itself, recorded as a permanent artifact in `research/ITER54_META_DEADZONE.md` and committed.

---

## Sequence status

| iter | target | result | win |
|---: |---     |---     |---:  |
| 47  | SCFA dwconv-dK | PASS | +5.20% |
| 48  | LN-bwd dgamma  | FAIL | +2.4% |
| 49  | Fused Adam-int8 | PASS | +6.98% |
| 50  | SCFA sub-conv  | FAIL | +1.31% |
| 51  | --scfa-checkpoint-inner | FAIL | +4.46% |
| 52  | cuBLAS algo override | NULL | 0% |
| 53  | LN-bwd 2-phase | FAIL | +2.86% |
| 54  | (this) meta-analysis | META | n/a |

Cumulative shipped: **+12.5%** (1.052 × 1.070).  Score 2/8 PASS.

If iter51 + iter53 were retroactively shipped under Option A (3% bar): cumulative would be ~+23%.
