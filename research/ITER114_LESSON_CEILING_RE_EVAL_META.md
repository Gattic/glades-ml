## Iter 114 — Lesson from iter 113 + re-evaluation of dismissed candidates

**Date**: 2026-05-21
**Iter**: 114 (META, no new code)
**Branch**: vesta5
**Verdict**: **META** — extracts the iter 113 design lesson, re-evaluates single-iter candidates previously dismissed at iter 110/112 META, and identifies the next 2 targets worth attempting in iter 115-116.

---

## The iter 113 lesson

**iter 110 META (2026-05-21)** declared the single-iter ceiling reached. iter 111 then FAILed on in-place reln_backward. iter 112 META re-affirmed the ceiling.

**iter 113 (2026-05-21) overturned this**: the line-9201 memcpy elimination — declared too complex and post-FAIL — turned out to be tractable via **Plan A's alternation design** (different buffers each iter via `dq_accum_override` parameter in `scfa_attention_backward`, not in-place).

### Generalizable lesson

**"Ceiling" claims are valid only when ALL implementation approaches for a target have been exhausted.** When one approach FAILs, this constrains the design space but doesn't always close it. The iter 108 FAIL → iter 109 PASS pattern already hinted at this (BF16-dst beta=0 unsafe, but pure memset skip safe at same buffer scale). iter 111 FAIL → iter 113 PASS confirms it more strongly: in-place broke math, but alternation worked.

### When to retry a "failed" mechanism

Worth re-attempting a mechanism class with a NEW implementation approach when:
1. The previous approach FAILed for an identifiable, well-understood reason (e.g., multi-kernel function-boundary in iter 111)
2. The new approach is **structurally different** (e.g., different buffers each iter vs in-place)
3. The target wall-clock savings are still ≥1% (worth the implementation cost)
4. The implementation cost (LOC + risk) is bounded

### Anti-pattern: "ceiling" → premature META

The iter 110 META was technically correct (no easy single-iter remaining at THAT moment), but it discouraged the iter 113 search. **A better META should explicitly list the previously-attempted approaches AND identify orthogonal alternatives that haven't been tried** rather than declaring closure.

## Re-evaluation of "previously dismissed" candidates

iter 110/112 META listed several "not worth attempting" single-iter targets. Applying the iter 113 lesson, here's a re-evaluation:

### Candidate 1: bwd recompute softmax + softmax_backward_attn fusion

**iter 110 estimate**: +0.5-1.5% wall, blocked by "kernel-level fusion + cuBLAS reorder; complex".

**Re-evaluation**: doable as a SIMPLE concatenation (not requiring math reorder beyond the cuBLAS call order). Plan:
- New kernel `causal_softmax_with_bwd_attn_kernel` in gpu_kernels.cu: literal concatenation of `causal_mask_softmax_kernel` passes 0-3 + `softmax_backward_attn_kernel` passes A-B (no math change)
- In `flash_attention_backward_cublas_tiled`: reorder cuBLAS so `scratch_dP = dO·V^T` is issued BEFORE softmax+bwd_attn (cuBLAS is sequential on compute stream, just code-order reorder)
- Replace softmax_inplace + softmax_backward_attn calls with single fused kernel call
- Savings: 1 kernel launch saved per call (24/step) + L2 cache benefit on P read between softmax and bwd_attn (was separate kernels, now one)

**Predicted savings**: +0.3-0.7% wall (modest but real).

**Risk**: medium. iter 83 NEGATIVE warned about FMA-emit issues with merged-pass softmax, BUT iter 83 specifically merged the MASK pass + MAX pass. My concatenation keeps the original 4 softmax passes intact and just adds the bwd_attn passes at the end. Should preserve FMA-emit order.

**Implementation cost**: ~100 LOC kernel + ~30 LOC dispatcher + ~20 LOC caller updates = ~150 LOC. Library change.

**Status**: viable iter 115 candidate.

### Candidate 2: Adam batching (4 weights × 24 layers = 96 → fewer launches)

**iter 110 estimate**: "sub-noise (96 → 1 launch; sub-noise savings)".

**Re-evaluation**: was based on the assumption that each Adam call has minimal kernel launch overhead. Actually at production scale, 96 launches × 20 µs = 1.9 ms per step = ~0.3% wall. Could batch 4 weights per layer into 1 launch (4× reduction): 96 → 24 launches = save 72 × 20 µs = 1.4 ms = +0.2% wall.

**Predicted savings**: +0.2-0.4%. Still sub-noise-to-low. Not worth iter cost.

**Status**: low priority.

### Candidate 3: LN bwd dgamma_dbeta partial buffer optimizations

**iter 110 estimate**: "<0.5%, structural limit due to reductions".

**Re-evaluation**: still seems structural. Skip.

### Candidate 4: reln_inverse + reln_backward fusion

**iter 110 estimate**: "<0.5%, custom kernel with reductions; high complexity".

**Re-evaluation**: reln_inverse (1.7% wall) + reln_backward dx kernel (2.6%) + reln_backward dgamma/dbeta (1.8%) total = 6.1% wall. Even a 30% reduction (from fusion eliminating mean/invStd recomputation) would be +1.8% wall.

But the FUSION is genuinely complex (two-pass reduction + math redesign). Implementation cost ~200-300 LOC. Risk: high (iter 111-class function-boundary issues).

**Status**: high reward but high risk. Multi-iter scope likely.

### Candidate 5: s.dp pre-zero skip + first-axpy = memcpy

**iter 110 estimate**: "sub-noise per iter 104 calibration (132 MB once per step)".

**Re-evaluation**: still sub-noise. Skip.

### Candidate 6: chiron_scfa_scaled_copy elimination

**iter 110 estimate**: "1.7% wall, fusion blocked by multiple consumers".

**Re-evaluation**: chiron_scfa_scaled_copy at line 7415 writes `W.scfa_ypar = sign * s.dp`. Consumers:
- cuBLAS at line 7440-ish (B^T · scfa_ypar)
- bwd_dwconv at line ~7500 (passes scfa_ypar as dy)

If both consumers can read `s.dp` directly with the sign baked into their alpha:
- cuBLAS alpha = sign instead of 1.0 (and reads s.dp directly): savings ~1.7% wall (eliminates the scaled_copy kernel)
- bwd_dwconv: would need to handle sign internally OR receive a "sign" parameter

**Predicted savings**: +0.3-0.7% wall (the kernel eliminated, but minor cuBLAS alpha overhead).

**Implementation cost**: ~50 LOC trainer changes + signature param for bwd_dwconv. Library change.

**Status**: viable iter 116 candidate. Lower priority than bwd softmax fusion.

## Iter 115-116 plan

If ralph-loop continues firing:

**iter 115**: bwd softmax + softmax_backward_attn fusion (Candidate 1).
- Predicted +0.3-0.7% wall.
- Combined with iter 113 stack: +9.5-9.9% multi-seed.

**iter 116**: chiron_scfa_scaled_copy elimination (Candidate 6).
- Predicted +0.3-0.7% wall.
- Combined: +9.8-10.6% multi-seed.

Both target the iter 105 profile's remaining un-attacked candidates. Both have similar risk profile to iter 113 (clear implementation path, math reasoning, n=3 multi-seed validation).

## Cumulative state (iter 113)

Pre-ralph-loop: 15,200 tok/s. Current iter 113 stack: 27,775 tok/s (n=3 mean). **Cumulative 1.83×**.

| target | tok/s | status |
|---|---:|---|
| 1.5× (iter 5) | 22,800 | ✓ HIT |
| 2.0× (extrapolation) | 30,400 | ~85% of way there |
| 3× (iter 10) | 45,600 | not met |
| 10× (iter 20) | 152,000 | not met |

To reach 2.0×, would need ~+9.5% over current iter 113 stack. iter 115 + iter 116 + reln-fusion + multi-iter FlashAttention combined could plausibly hit this.

## Files

- This document.
- All iter 95-113 writeups in `research/ITER<N>_*.md`.
- No new code in iter 114 (META only).
