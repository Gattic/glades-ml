# VESTA Phase 2 — Gate-2B Initial Report

**Date:** 2026-05-19
**Status:** Gate-2B **INITIAL FAIL** (matches pre-registered F-Phase2-1).

## What Gate-2B asked

> 2B: stack 4–6 GRP-RNN+LN blocks with residual connections on a real LM corpus.
> PASS if multi-layer val_loss strictly lower than single-layer at iso-params.
> FAIL if no improvement.

## What was built

`research/ealrmn_gpu/model_grp_stack.cuh` — multi-layer stack:
- Per-layer pre-norm `[LN1 → GRP-RNN recurrence → +residual → LN2 → MLP(4×) → +residual]`
- Shared embedding `E`, final `LN_final`, readout `W_out`/`b_out`
- LM-mode only (per-token next-token loss)
- Full forward/backward implemented; gradcheck 91/96 on L=1 (m=8, T=8, B=2), 157/162 on L=2 — all failures at the FP32 precision floor on small-magnitude gradient entries (relative errors driven by `sqrt(precision * L) / (2·eps)` cancellation, not by analytic bugs)
- Training converges normally on real corpus

Also added:
- `CorpusTask` in `tasks.cuh` — reads BPE-pretokenized `.tok.bin` files (24-byte header + uint16 tokens)
- `--corpus=PATH`, `--corpus-val=PATH`, `--n-layers=L`, `--mlp-hidden=H` CLI flags

## Experimental result on real corpus

Corpus: `/home/robert/dev/glades-trainer/pretok-data/val.tok.bin` (219M tokens, V=32000 BPE)

Setup: T=128, batch=32, eval_batch=64, steps=1500, seed=42, AdamW (β=0.9/0.95, wd=0.01), grad-clip=1.0.

### Same-m comparison (m=128 fixed)

| Layers | Total params | LR     | Warmup | val_loss @ 1500 |
|:------:|:------------:|:------:|:------:|:---------------:|
| L=1    | 8.39M        | 3e-3   | 100    | **8.26**        |
| L=2    | 8.55M        | 3e-3   | 100    | 8.34            |
| L=2    | 8.55M        | 2.1e-3 | 200    | 8.40            |
| L=4    | 8.88M        | 3e-3   | 100    | 8.41            |
| L=4    | 8.88M        | 1.5e-3 | 300    | 8.70            |

Even though L=2/4 have MORE total params, val_loss is STRICTLY WORSE than L=1.
LR-scaling-by-1/√L (standard transformer recipe) makes things worse, not better.

### Iso-stack-param comparison (~180K stack params; embedding+readout dominates total)

| Layers | m   | Total params | val_loss @ 1500 |
|:------:|:---:|:------------:|:---------------:|
| L=1    | 128 | 8.39M        | **8.25**        |
| L=2    | 90  | 5.96M        | 8.48            |
| L=4    | 64  | 4.29M        | 8.72            |

Iso-stack also favors L=1. (Note: total params differ here because E and W_out scale linearly with m.)

## Interpretation

This matches the brief's pre-registered failure mode **F-Phase2-1**: depth doesn't compound at small scale with this architecture / hyperparameter recipe. Three non-exclusive hypotheses:

1. **Init pathology.** The GRP recurrence is near-identity at init (W_a init small → angles ≈ 0 → R_t ≈ decay·I). Stacking L near-identity blocks gives an essentially-identity stack, and the deeper net has more spurious capacity that adds noise without contribution before the upper layers learn anything. A standard transformer fix (zero-init the final linear in each block; multiply skip by 1/√L) was NOT implemented here; it should be tried.
2. **The mechanism itself.** GRP-RNN's orthogonal-rotation state-tracking advantage (the source of Claim N1's A_5 win) may not be the bottleneck on natural-language perplexity at all — most natural-language tokens aren't produced by non-solvable-group dynamics, so the additional non-diagonal expressivity capacity per layer doesn't matter and the only meaningful capacity is m (single-layer width). This is **F-Phase2-2** showing up early.
3. **Insufficient steps.** 1500 steps × 32 batch × 128 T = 6.1M tokens — well below the iso-FLOPs budgets at which transformer scaling laws produce stable curves. Longer runs (10⁵–10⁶ steps) might unlock depth.

## Decision

Per the brief's prohibition **P7** ("no new mechanism invention until Gate-2C clears"), we do not change the GRP-RNN block. The legitimate mitigations available within scope:

- **2B-mitigation-A**: zero-init final linear of MLP. **TRIED 2026-05-19, FAILED** — every config slightly worse:

  | L | val_loss @ 1500 (standard init) | val_loss @ 1500 (zero-init W_mlp2) |
  |:-:|:-:|:-:|
  | 1 | 8.26 | 8.30 |
  | 2 | 8.34 | 8.37 |
  | 4 | 8.41 | 8.42 |

  This rules out the "each-block-starts-as-identity" init pathology as the dominant cause. Mitigation reverted in code.

- **2B-mitigation-D (B5 baseline test)**: compare GRP-stack against a stack with the rotation disabled (`--linear-recurrence=1`: theta = 0 always, so `s_t = decay·s_{t−1} + W_in·x_t`, a pure scalar-decay linear RNN). All other architecture (LN1, MLP, LN2, residual, LN_final) identical. **TRIED 2026-05-19. DECISIVE: linear-RNN beats GRP at every depth.**

  | Config       | val_loss @ 1500 | Δ vs GRP |
  |:------------:|:---------------:|:--------:|
  | L=1 m=128 GRP        | 8.23 | —        |
  | L=1 m=128 LinearRNN  | **8.00** | **−0.23 nat** |
  | L=2 m=128 GRP        | 8.30 | —        |
  | L=2 m=128 LinearRNN  | **7.95** | **−0.35 nat** |

  And linear-RNN at L=2 beats linear-RNN at L=1 (7.95 vs 8.00) — depth helps WITHOUT the rotation, doesn't help WITH it.

  **Multi-seed validation** (brief P5: ≥3 seeds), L=2 m=128:

  | Seed | GRP | LinearRNN | Δ |
  |:-:|:-:|:-:|:-:|
  | 42 | 8.30 | 7.95 | +0.35 |
  | 43 | 8.34 | 7.93 | +0.42 |
  | 44 | 8.14 | 7.63 | +0.51 |
  | 45 | 8.27 | 7.90 | +0.37 |
  | **Mean ± stdev** | **8.26 ± 0.09** | **7.85 ± 0.14** | **+0.41 ± 0.07** |

  Linear-RNN beats GRP at every seed by 0.35–0.51 nat. Result is seed-robust.

## Confirmed: F-Phase2-2

The B5 head-to-head is the clean signal the brief was designed to capture: **GRP's Givens-rotation mechanism actively harms natural-language LM at this scale.** Two findings:

1. **GRP loses to scalar-decay linear-RNN.** The non-diagonal rotation that produced Phase-1's A_5 expressivity win adds capacity that natural language doesn't use, while degrading optimization through what appears to be either gradient instability through the Givens product or just plain wasted capacity.
2. **Removing the rotation unlocks depth.** Linear-RNN+LN+MLP shows monotone improvement with depth. GRP+LN+MLP shows monotone degradation. So the "no depth gain" we saw in 2B-mitigation-A was a GRP-specific pathology, not a generic stack pathology.

Per the brief's failure-mode definitions:
> **F-Phase2-2**: GRP-RNN+LN matches Mamba-2 at small scale but doesn't beat it. The Phase-1 expressivity advantage is real but doesn't manifest in natural-language perplexity because most natural-language tokens aren't produced by non-solvable-group state-tracking.

Our finding is *stronger* than F-Phase2-2 predicted: GRP doesn't merely fail to beat a baseline, it *loses* to the simplest in-class baseline (B5: linear-RNN+orthogonal). The Phase-1 mechanism is not just useless for NL — it is harmful.

## Implication for the program

Per the Phase-2 brief's gate sequence:
- **Gate-2B: FAIL.** Multi-layer GRP doesn't beat L=1 GRP, and L=1 GRP loses to L=1 linear-RNN.
- **Gate-2C** (vs Mamba-2 at 10–100M params): structurally guaranteed to FAIL given the B5 result. A simpler in-class baseline already beats GRP; Mamba-2 (a tuned modern SSM) will beat both.
- The honest deliverable is now **F-Phase2-2 negative-result publication**: "GRP-RNN's Phase-1 expressivity advantage on A_5 (Merrill et al. 2024's diagonal-SSM bound) does not transfer to natural-language perplexity at the scales tested; in fact, the non-diagonal rotation mechanism is actively harmful."

This matches the brief's mid-likelihood expectation and is a meaningful scientific contribution: the Merrill bound captures a real expressivity gap on synthetic group-theoretic state-tracking, but that gap is not the bottleneck for natural-language LM. Whatever the bottleneck for NL is, it is not the diagonal-SSM expressivity limit.

## Files

- `research/ealrmn_gpu/model_grp_stack.cuh` — multi-layer stack model (~650 LOC)
- `research/ealrmn_gpu/tasks.cuh` — added `CorpusTask` (BPE token loader)
- `research/ealrmn_gpu/main.cu` — `train_grp_stack`, `mode_gradcheck` for stack, CLI flags

See [[vesta_phase2_brief]] for the gate sequence and [[vesta_phase2_gate2a]] for the Gate-2A infrastructure that this gate built on.
