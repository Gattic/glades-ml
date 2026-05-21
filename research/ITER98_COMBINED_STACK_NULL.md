## Iter 98 — Combined stacking test: iter 95 + iter 97 — NULL on stacking

**Date**: 2026-05-21
**Iter**: 98 (post iter 97 PASS)
**Branch**: vesta5 (glades-ml) + glades-trainer
**Verdict**: **NULL on stacking** — iter 95 (bwd dx smem tile) contributes zero measurable wall delta when combined with iter 97 (fwd fused sub). iter 97's +1.54% stands alone.

---

## Motivation

iter 95 alone benched as NULL (single-seed wall delta within noise, parity-clean). The mechanism (smem tile of bwd dx kernel at w=4) is bit-identical math but didn't deliver measurable wall improvement standalone.

Hypothesis tested in iter 98: maybe iter 95's mechanism contributes only when combined with iter 97. Stacking context could amplify the bwd dx tile's effect if iter 97's faster fwd shifts the cuBLAS GEMM ordering or L2 state.

## Bench (single-seed 100 steps × seed=1337 × T=16384 m=2048 L=24 w=4)

| run | wall (s) | tok/s @ 26 / 51 / 76 | NLL @ 100 | PPL |
|---  |---:      |---:                  |---:       |---:  |
| baseline (no flags)                                            | 65.3 | 24,943 / 25,214 / 25,214 | 9.8477 | 18913.96 |
| iter97 alone (--iter97-dwconv-fwd-fused-sub)                   | 64.3 | 25,319 / 25,606 / 25,603 | 9.8476 | 18912.39 |
| **iter95 + iter97** (--iter95-dwconv-bwd-dx-tiled --iter97-dwconv-fwd-fused-sub) | **64.3** | **25,313 / 25,589 / 25,579** | **9.8476** | **18913.85** |

**Δ (combined − iter97 alone)**: 
- Wall: 0.0s (identical)
- tok/s @ step 26: -6 (-0.02%)
- tok/s @ step 51: -17 (-0.07%)
- tok/s @ step 76: -24 (-0.09%)
- NLL: 0.0000

All within ~0.1% intra-run variance. iter 95 contributes nothing on top of iter 97.

**Δ (combined − baseline)**:
- Wall: -1.0s (+1.44%) — same as iter 97 alone (+1.54%, within noise)

## Verdict

**NULL on stacking**. The +1.44% combined improvement matches iter 97's standalone +1.54% (within noise) — iter 95 adds zero. This confirms:

1. **iter 95 is mechanism-genuinely-NULL at production w=4** (confirmed empirically in both standalone and stacked contexts).
2. **iter 97 is the only productive single-iter contribution** at the post-iter94 stack — the +1.54% is the full stacked win.
3. Smem tiling has no headroom at w=4 dwconv kernels — neither fwd (iter 96 NULL) nor bwd dx (iter 95 + iter 98 confirmation).

## Implication for next iters

- **Don't pursue smem-tile extensions at w=4** — fwd and bwd both empirically null.
- **Do pursue adjacent-kernel fusion via smem-load arithmetic** (iter 97 mechanism) — first PASS class identified at the post-iter94 ceiling.
- **Bwd-side scfa_sub fusion blocked** by buffer-reuse complexity: scfa_qpar gets overlaid as BF16 inner-attention scratch between forward-recompute (line ~7056) and bwd_dwconv (line ~7430), so q_par is destroyed before the bwd needs it. Fusion requires algorithmic restructuring of the bwd flow OR a separate q_par-preserving scratch buffer (132 MB extra VRAM).
- **Other fusion candidates** to explore: `chiron_scfa_scaled_copy` (1.6% wall, follows by cuBLAS GEMM — possibly absorbable via cuBLAS alpha), `axpy_kernel` (2.6% wall, 25 calls/step), `causal_mask_softmax` + cast pair (3.8% softmax + small cast = fused softmax-with-BF16-output).

## Files

- This document.
- `research/runs/2026-05-21-iter97-gate0/iter98_combined_100step.log` (64.3s, iter95+iter97 combined).
- No new code (bench-only validation iter).
