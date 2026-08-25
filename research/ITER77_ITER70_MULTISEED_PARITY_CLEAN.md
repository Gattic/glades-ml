## Iter 77 — iter 70 alone at multi-seed — PARITY CLEAN, below bar

**Date**: 2026-05-20
**Iter**: 77 (twenty-third iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **FAIL by bar, PASS parity** — iter 70 alone at multi-seed (3 seeds) shows wall +1.27% with NLL drift **−0.008 nat ± 0.035** (within ±0.02 strict AND ±0.05 multi-seed).  iter 70 is the parity-clean partner in the combined iter 70+73 path; iter 73 contributes the bulk of the +0.067 nat combined drift.  Default policy unchanged (opt-in flag default OFF) because +1.27% is below both strict (+5%) and iter 60-precedent (+3%) bars.  But iter 70 is now VALIDATED as a silent-accrual building block for future combined retro-ship eligibility.

---

## Background — iter 76 left a question unanswered

iter 76 benched the combined iter 70 + iter 73 path at multi-seed: combined wall +1.71%, NLL drift +0.067 nat ± 0.054 (outside ±0.05 multi-seed bound, std 3.3× baseline).  The +0.067 nat drift could come from either mechanism alone or interact via the combined path.  iter 77 isolates iter 70 alone.

## Multi-seed bench (iter 70 alone)

Each run: 200 steps, L=24 T=16384 m=2048, lr=3e-4, warmup=20, grad-clip=1.0.  Stack: `--int8-adam --bf16-grads --bf16-weights --bf16-attn --no-fuse-attn --fuse-attn-reln --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt --bf16-logits --bf16-logits-storage --scfa-checkpoint-inner --scfa-checkpoint-inner-bf16 --iter70-fused-axpy2-dual-p`.

| seed | NLL @ step 200 | wall (s) |
|---:  |---:            |---:      |
| 1337 (iter 70 v2, from iter 70 doc) | 7.6405 | 134.6 |
| 1338 | 7.5701 | 134.6 |
| 1339 | 7.6119 | 134.8 |
| **mean (n=3)** | **7.6075** | **134.67** |
| **std** | **0.0354** | 0.10 |

## Comparison

| metric | iter 76 baseline (n=4) | iter 70 alone (n=3) | iter 76 combined 70+73 (n=3) |
|---     |---:                    |---:                  |---:                          |
| NLL mean | 7.6154 ± 0.016 | **7.6075 ± 0.035** | 7.6826 ± 0.054 |
| NLL Δ vs baseline | — | **−0.008 nat** | +0.067 nat |
| NLL std | 0.016 | 0.035 (2.2× baseline) | 0.054 (3.3× baseline) |
| wall (200 steps) | 136.40 s | 134.67 s | 134.10 s |
| Δ wall vs baseline | — | **−1.27%** | −1.69% |
| tok/s | 24,023 | 24,332 | 24,434 |
| Δ tok/s vs baseline | — | **+1.29%** | +1.71% |

## Key findings

1. **iter 70 alone is parity-clean at multi-seed**.  Mean drift −0.008 nat is **within ±0.02 nat strict bound** and certainly within ±0.05 nat multi-seed bound.  Std 0.035 is 2.2× baseline (some seed-dependence) but the mean shift is essentially zero.

2. **iter 73 is the drift source in combined**.  Combined (70+73) shifts NLL by +0.067 nat with std 0.054; iter 70 alone shifts by only −0.008 nat with std 0.035.  Subtracting: iter 73's contribution is +0.075 nat mean shift and additional variance.  iter 73's `#pragma unroll` FMA-emit divergence is the source.

3. **iter 70 wall gain (+1.27%) is real and validated** at multi-seed.  iter 73's marginal contribution (+0.42% wall, 1.71% combined − 1.29% iter 70 alone) comes at the cost of significant NLL drift — not a favorable trade.

4. **iter 70 alone fits the silent-accrual pattern cleanly**: real wall improvement, parity-clean NLL, no architecture change.  But +1.27% is below both the strict +5% bar AND the iter 60-precedent +3% combined retro-ship bar.

## Why this remains FAIL

Strict per-iter bar: +5% wall.  iter 70 alone at +1.27% is well below.
iter 60-precedent retro-ship bar: +3% wall.  iter 70 alone at +1.27% is also below.

iter 70 alone is the cleanest below-bar silent-accrual candidate evaluated to date, but a single +1.27% mechanism doesn't clear any precedent bar for default-on ship.  Need 2+ orthogonal +1-2% wins to bundle for +3% retro-ship — and the obvious candidate (iter 73) ruins parity.

## Default policy

`--iter70-fused-axpy2-dual-p` remains **default OFF**.  Opt-in flag preserved.  Future iters that find a +1.5-2% NLL-clean mechanism could bundle with iter 70 for combined retro-ship at +3% bar.

## Sequence status (23 iters)

| iter | result | wall | parity (multi-seed) |
|---: |---     |---:  |---                  |
| 69  | **PASS** | **+6.83%** | — (single-seed era) |
| 70  | FAIL→VALIDATED | +1.27% | **clean at multi-seed (Δ=−0.008)** |
| 71  | NEGATIVE | −5.4% | clean (iter 75 reclassification) |
| 72  | NULL | ±0% | — |
| 73  | FAIL | +0.56% | drift source in combined |
| 74  | FAIL | +0.23% | noise |
| 75  | META | n/a | methodology correction |
| 76  | multi-seed retro FAIL | +1.71% combined | +0.067 nat drift |
| 77  | parity-clean below bar | +1.27% iter 70 alone | **−0.008 nat** |

## Strategic implication

After iter 77, the **realistic engineering picture** is:
- The ONLY parity-clean engineering win remaining in the custom-kernel space is iter 70 at +1.27% wall.
- Combining with any of iter 71-74 either degrades wall (71) or breaks parity (73 most strongly).
- Single-iter ≥+5% wins are not on the table.

**Realistic paths forward** (multi-iter arcs only):
1. **Hadamard basis FWHT** kernel (multi-day implementation; would replace SCFA outer cuBLAS GEMMs).  Potential +3-7% wall but requires retraining flagship.
2. **FlashAttention-fused SCFA inner attention** (multi-day kernel; would collapse the 23.5% inner-attention cuBLAS slice).  Potential +5-15% wall.
3. **Accept current flagship + iter 70 silent-accrual** at +1.27% wall.  No further single-axis wins.

The brief's 10× target appears infeasible without multi-iter arc commitment.

## Files

- This document.
- `research/runs/2026-05-20-iter77-bench/iter70_alone_seed{1338,1339}.log`.
- No code change.
