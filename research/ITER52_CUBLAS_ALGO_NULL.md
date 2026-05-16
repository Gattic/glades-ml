# Iter 52 — cuBLAS algo override on readout GEMMs — NULL result

**Date**: 2026-05-16
**Iter**: 52 (sixth iter under "stacking-wins" brief)
**Branch**: vesta5 (glades-ml) + glades-trainer/main
**Verdict**: NULL — all 16 explicit `CUBLAS_GEMM_ALGO0_TENSOR_OP` through `CUBLAS_GEMM_ALGO15_TENSOR_OP` overrides perform within 0.3% of the cuBLAS `DEFAULT_TENSOR_OP` auto-pick on the readout shape (T=8192, V=32000, m=2048).  cuBLAS's auto-pick is already optimal for this shape; no algo-level lever available.  No code change ships.

---

## TL;DR

Hypothesis (from iter51 doc's "iter 52 candidate" section): cuBLAS's auto-pick (`CUBLAS_GEMM_DEFAULT_TENSOR_OP`) might miss a faster algorithm for the readout-shape GEMMs (3 instances per step, 5.5% each = 16.5% combined).  If even one algorithm yielded 30%+ on one of the three GEMMs, the iter would clear the +5% bar.

Implementation: added an env-driven override (`GLADES_READOUT_ALGO=0..15`) that picks `CUBLAS_GEMM_ALGOn_TENSOR_OP` and threads it through the shared `gemmex_bf16_impl` plus the standalone `sgemm_rowmajor_abt_bf16_bf16out`.

Result: full sweep at 30 steps, T=8192 m=2048 L=12, all shipped iter49 flags:

| algo                       | tok/s @ step 30 | Δ vs DEFAULT |
|---                         |---:             |---:          |
| `DEFAULT_TENSOR_OP`        | 43,393          | baseline     |
| `ALGO0_TENSOR_OP`          | 43,354          | −0.09%       |
| `ALGO1_TENSOR_OP`          | 43,347          | −0.11%       |
| `ALGO2_TENSOR_OP`          | 43,343          | −0.12%       |
| `ALGO3_TENSOR_OP`          | 43,337          | −0.13%       |
| `ALGO4_TENSOR_OP`          | 43,343          | −0.12%       |
| `ALGO5_TENSOR_OP`          | 43,278          | −0.26%       |
| `ALGO6_TENSOR_OP`          | 43,333          | −0.14%       |
| `ALGO7_TENSOR_OP`          | 43,298          | −0.22%       |
| `ALGO8_TENSOR_OP`          | 43,277          | −0.27%       |
| `ALGO9_TENSOR_OP`          | 43,349          | −0.10%       |
| `ALGO10_TENSOR_OP`         | 43,346          | −0.11%       |
| `ALGO11_TENSOR_OP`         | 43,287          | −0.24%       |
| `ALGO12_TENSOR_OP`         | 43,284          | −0.25%       |
| `ALGO13_TENSOR_OP`         | 43,271          | −0.28%       |
| `ALGO14_TENSOR_OP`         | 43,313          | −0.18%       |
| `ALGO15_TENSOR_OP`         | 43,260          | −0.31%       |

All 16 manual algorithms perform within ±0.3% of DEFAULT; none beat it.

cuBLAS's internal heuristic is already picking the best algorithm for this shape (T=8192, V=32000, m=2048 with BF16-TC compute).

---

## Why this matters / lesson for iter 53+

cuBLAS auto-pick was a strong baseline for the readout-shape GEMMs.  This is the **classical "you can't beat cuBLAS at the GEMM it was tuned for" finding** — confirmed empirically.

The iter51 doc's hypothesis (10-30% speedup possible from a non-default algo) was wrong.  cuBLAS DEFAULT was already at the local optimum.

For iter 53+ targeting the readout GEMMs: only paths left are
- Smaller-dtype compute (e.g., FP8 via cuBLASLt) — different paradigm, precision risk.
- Reduce the V dimension via partial-vocab readout — algorithmic change, not engineering.

Neither is a stacking-win-style optimization.  The readout GEMMs are likely a **dead zone** for further engineering optimization at this dtype/shape.

For iter 53, pivot to a different bottleneck.

---

## Sequence under stacking-wins brief

| iter | target | result |
|---: |---     |---     |
| 47  | SCFA dwconv-dK par-reduction | PASS +5.20% |
| 48  | LN-bwd dgamma/dbeta          | FAIL too-small-target |
| 49  | Fused Adam-int8 BF16w/g + bf16 grad-norm | PASS +6.98% |
| 50  | SCFA sub-into-conv           | FAIL bench-noise + NLL drift |
| 51  | --scfa-checkpoint-inner revival | FAIL below-bar +4.46% |
| 52  | cuBLAS readout algo override | NULL 0% |

Cumulative shipped: **+12.5%**.  Score after 6 iters: 2 PASS / 3 FAIL / 1 NULL.

---

## Reproducibility

The env-driven override was added then reverted before commit.  Sweep was done with:

```bash
for algo in {0..15}; do
  GLADES_READOUT_ALGO=$algo build/glades_chiron_train \
    --data-dir pretok-data --pretokenized --vocab 32000 \
    --seq-len 8192 --m 2048 --layers 12 --heads 16 --dhead 256 \
    --lr 3e-4 --max-steps 30 --warmup 5 --grad-clip 1.0 \
    --int8-adam --bf16-grads --bf16-weights --bf16-attn \
    --no-fuse-attn --fuse-attn-reln \
    --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
    --bf16-logits --bf16-logits-storage \
    --seed 1337 2>&1 | grep "step     30"
done
```

---

## Where next (iter 53 candidates)

Pivot away from the readout GEMMs.  Top remaining ≥7%-share targets (or combinables):
- LN backward dx + dgamma_dbeta combined (~6.6%) — fused pass with deterministic partitioned scratch
- chiron_scfa_axpy2 + sub combined (~10%) — already shown to be hard (iter50 NLL drift)
- k_bf16_accum_axpy elimination via direct BF16 grad writes from attention backward (~3%)
- A truly different paradigm: revisit FLASH-attention path (iter6 NULL on its own) maybe re-test on iter49 stack
