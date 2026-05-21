## Iter 78 — cuBLAS workspace size tuning — NULL

**Date**: 2026-05-20
**Iter**: 78 (twenty-fourth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **NULL** — Δ wall ±0% at both 16 MB and 64 MB workspace sizes vs default (4 MB).  Larger workspaces select different cuBLAS algorithms (causing +0.036 to +0.067 nat NLL drift) but none of them are faster.  Extends iter 52's "cuBLAS DEFAULT is optimal" finding from algo-override to workspace-size.

---

## Problem statement

After iter 77 validated iter 70 alone as parity-clean at +1.27% wall, the strategic goal was to find an ORTHOGONAL +1.5-2% NLL-clean partner mechanism to bundle for +3% iter 60-precedent retro-ship eligibility.

cuBLAS workspace size tuning is a fresh angle: iter 52 tested algo override but not workspace SIZE.  Larger workspace can enable split-K and stream-K algorithms which parallelize the K-dimension across SMs.  For SCFA inner BF16-TC GEMMs (k=1024, m=2048, T_inner=1024) at the 14.1% + 9.4% = 23.5% slice, split-K could deliver substantial speedup.  Pure env-var change (CUBLAS_WORKSPACE_CONFIG), no code touch, no NLL drift risk from kernel restructuring.

## Bench (production stack, seed=1337, 200 steps)

| config | NLL @ step 200 | wall (s) | Δ wall vs baseline |
|---     |---:            |---:      |---                 |
| baseline (default workspace ~4 MB, iter 76 mean n=4) | 7.6154 ± 0.016 | 136.40 | — |
| `CUBLAS_WORKSPACE_CONFIG=:16384:8` (16 MB) | 7.6515 | 136.4 | **±0%** |
| `CUBLAS_WORKSPACE_CONFIG=:65536:8` (64 MB) | 7.6822 | 136.5 | **±0%** |

## Why this fails

1. **Zero wall improvement at either size**: cuBLAS's default heuristic at 4 MB workspace already picks the OPTIMAL algorithm for the GEMM shapes in CHIRON's production stack.  Allowing larger workspaces doesn't enable a FASTER algo — it just enables more algos to be CONSIDERED.

2. **NLL drift from different algo choices**: With larger workspace, cuBLAS heuristic picks a different algorithm (likely a split-K variant with different reduction order).  Different reduction order produces different FP32 accumulator round-off → ULP-scale drift in the GEMM output → compounds through the training chain.  16 MB → +0.036 nat drift; 64 MB → +0.067 nat (drifts grow with workspace size, suggesting progressively different algos are being picked).

3. **iter 52 conclusion extends to workspace size**: iter 52 tested cuBLAS algo override on the LM head readout GEMM and found all 16 ALGOn_TENSOR_OP variants within 0.3% of DEFAULT_TENSOR_OP.  The default heuristic is optimal at READOUT shape.  iter 78 extends this finding: the default heuristic is also optimal for the FULL set of CHIRON GEMM shapes, regardless of workspace size.

## Categorization

**null** (no real signal on wall).  cuBLAS auto-pick at default workspace is already optimal.  No engineering lever available.

## Default policy

**Do not set CUBLAS_WORKSPACE_CONFIG**.  Default workspace (~4 MB) gives optimal wall.  Setting it explicitly only introduces NLL drift.  No code change.

## Sequence status (24 iters)

| iter | result | wall | notes |
|---: |---     |---:  |---    |
| 70  | parity-clean below bar | +1.27% (validated iter 77) | only NLL-clean engineering win remaining |
| 71  | NEGATIVE | −5.4% | Ada TC sat |
| 72-74 | NULL / FAIL | various | iter 75 reclassified |
| 75  | META | n/a | methodology correction |
| 76  | multi-seed retro FAIL | +1.71% combined | iter 73 drift source |
| 77  | iter 70 alone PARITY CLEAN | +1.27% | below bar |
| 78  | cuBLAS workspace NULL | ±0% | no engineering lever |

## Strategic implication

The "hunt for orthogonal +1.5-2% NLL-clean partner" for iter 70 produces NULL at the most obvious unattacked angle (cuBLAS workspace).  The remaining candidates require either:
- Multi-iter Hadamard FWHT kernel (replaces SCFA outer cuBLAS GEMMs but requires production retrain)
- Multi-iter FlashAttention-fused inner attention (custom kernel collapsing the 23.5% slice)
- Multi-iter mixture-of-depth routing (architectural change for inference acceleration)

All require multi-iter commitment.  Single-iter ≥+5% wins (or even +1.5% NLL-clean wins) are not on the table.

## Files

- This document.
- `research/runs/2026-05-20-iter78-bench/ws_{16M,64M}_seed1337.log`.
- No code change.
