## Iter 82 — SCFA compression ratio sweep (ratio=20, ratio=32) — FAIL parity, demonstrates wall ceiling

**Date**: 2026-05-20
**Iter**: 82 (twenty-eighth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **FAIL by NLL parity at both ratios**, but the **+37.6% tok/s gain at ratio=32 demonstrates the wall ceiling is much higher than the current production stack**.  Motivates multi-iter arc commitment (Hadamard FWHT, FlashAttention) that can deliver similar wall savings WITHOUT shrinking k.

---

## Background — Priority #2 brief option

The brief lists "SCFA variants (different basis, different compression ratio, different inner attention)" as Priority #2 "Proven-family extensions".  Compression ratio sweep is a zero-code-change probe via `--scfa-compression-ratio N` CLI flag.

iter 41 falsified "per-position low-rank K/V compression" at brief-class magnitudes (HMTA paradigm).  SCFA's basis projection is on the time dimension, not on K/V heads — different mechanism class.  Whether SCFA tolerates ratio > 16 (smaller k) at flagship scale was untested.

## Bench (fresh-init, seed=1337, 200 steps)

| ratio | k (= T/ratio) | tok/s mean | wall (s) | val NLL @ step 200 |
|---:   |---:           |---:        |---:      |---:                |
| 16 (baseline n=4) | 1024 | 24,023 | 136.40 | 7.6154 ± 0.016 |
| **20** | **819** | **25,440** | **130.4** | **8.0347** |
| **32** | **512** | **33,050** | **100.4** | **8.5846** |

### Δ analysis

| ratio | Δ tok/s | Δ wall | Δ NLL | parity verdict |
|---:   |---:     |---:    |---:   |---             |
| 20    | **+5.9%** | −4.4% | **+0.42 nat** | FAIL (outside ±0.02 strict, outside ±0.10 lenient) |
| 32    | **+37.6%** | −26.4% | **+0.97 nat** | FAIL (catastrophic drift) |

**Both ratios cleanly clear the +5% wall bar.  Both cleanly FAIL NLL parity.**

The wall vs NLL tradeoff at 200 steps:
- ratio=20: +14% wall-per-nat drift (5.9% / 0.42)
- ratio=32: +39% wall-per-nat drift (37.6% / 0.97)

ratio=32 is more efficient on wall-per-NLL-cost.  But both exceed the parity bar.

## Why NLL fails — iter 41 pattern at fresh init

The CHIRON flagship was trained at ratio=16 (k=1024 at T=16384).  The DCT-II basis projection retains the lowest-k frequency components and discards the (T-k)/T fraction of the spectrum.  

At ratio=16: k=1024 captures 6.25% of frequency bins. The model relies on the remaining spectral tails reaching downstream layers via the depthwise conv path (y_perp).

At ratio=20 (k=819): 5.0% retention; spectrum is 20% narrower.  
At ratio=32 (k=512): 3.1% retention; spectrum is 50% narrower.

Per iter 41's SV-probe analysis at flagship scale: heavy spectral tails contain essential signal.  Truncating the projection bandwidth removes capacity in a way that cannot be recovered by the depthwise conv residual path alone.

The 200-step bench captures the EARLY-TRAINING convergence rate.  ratio=20 is 0.42 nat behind at step 200 — extrapolating to 30k steps (full flagship training), the gap would likely persist or grow.  The CHIRON flagship achieves val NLL 3.77 @ 30k at ratio=16; at ratio=20 the final NLL would likely be in the 3.85-4.00 range; at ratio=32 in the 4.30-4.50 range (rough extrapolation).

**The ratio sweep is idea-FAIL** — same root cause as iter 41 (aggressive spectral truncation at flagship class impairs model capacity).

## Strategic value of the wall data

Despite the parity FAIL, iter 82 produces **highly informative wall data**:

- The current production stack at 24,023 tok/s is heavily SCFA-inner-attention-compute-bound.  At k=512 (ratio=32), the inner attention area drops 4× and we observe **+37.6% wall savings**.
- This means **wall improvements of +30%+ are physically achievable on this hardware/stack** — they're not limited by HBM bandwidth, kernel launch overhead, or fundamental Ada saturation.
- The barrier is **model capacity at smaller k**.  Mechanisms that reduce inner-attention compute WITHOUT reducing k could potentially capture similar wall savings without NLL cost.

### Motivated multi-iter arc candidates

| approach | mechanism | wall potential | NLL story |
|---       |---        |---:           |---        |
| **Hadamard FWHT** at ratio=16 | FWHT replaces DCT-II cuBLAS for outer compress/lift; k unchanged | +3-7% (limited by outer GEMM slice) | likely parity-clean — different basis, same dimensionality |
| **FlashAttention-fused SCFA inner** at ratio=16 | Custom kernel fuses Q@K^T + softmax + @V; eliminates inner score matrix HBM | +5-15% (covers most of inner attention slice) | bit-identical possible with careful FMA matching |
| **Inner attention quantization (FP8 / INT8)** at ratio=16 | Lower-precision K/V/score matrices | +10-30% if feasible | requires careful per-token scaling |

Iter 82's evidence: the +37.6% gain from k-shrinking shows the inner attention is the dominant cost AND is amenable to wall reduction.  Mechanisms B/C above target the same slice without dimensionality loss.

## Sequence status (28 iters)

| iter | result | wall | notes |
|---: |---     |---:  |---    |
| 69  | last PASS | +6.83% | BF16-checkpoint-inner |
| 70-81 | 12× non-PASS | various | engineering ceiling on single-axis |
| 82  | ratio sweep idea-FAIL parity | +5.9% / +37.6% but +0.42/+0.97 NLL | wall ceiling demonstrated |

## Default policy

`--scfa-compression-ratio` remains at default 16.  No code change.  Both ratio=20 and ratio=32 are documented as idea-FAIL via fresh-init convergence (iter 41 mechanism).

## Path forward

iter 82's wall data **strongly motivates multi-iter arc B (FlashAttention-fused SCFA inner)** as the highest-EV path:
- Targets the same compute slice that ratio=32 reduces
- Maintains k=1024 dimensionality (no capacity loss)
- Estimated +5-15% wall, bit-identical possible
- Multi-day implementation but clear strategic case

Hadamard FWHT (arc A) targets the outer GEMM slice, which iter 82 shows is a smaller fraction of wall.  Lower-EV than FlashAttention SCFA inner.

Mixture-of-depth (arc C) targets inference, not training wall.  Different goal.

## Files

- This document.
- `research/runs/2026-05-20-iter82-bench/ratio{20,32}_seed1337.log`.
- No code change.
