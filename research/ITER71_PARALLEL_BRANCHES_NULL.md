## Iter 71 — --scfa-parallel-branches revival — NEGATIVE/NULL

**Date**: 2026-05-19
**Iter**: 71 (seventeenth iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)
**Verdict**: **NEGATIVE** — −5.4% tok/s (24,320 → 22,995), +5.9% wall (136.5 → 144.5 s).  NLL drift −0.168 nat (BETTER direction, same FMA/scheduling-noise pattern as iter 70).  Same Ada cuBLAS-TC-saturation NULL mechanism as iter 6 (`--bf16-logits-parallel-bwd`) and iter 58.

---

## Problem statement

iter 8 (2026-05-14) wired `--scfa-parallel-branches` to split SCFA forward into two streams after step 1 (`q_compr = B^T·q`):
- Branch A on `sideComputeStream`: step 2 (cuBLAS `q_par = B·q_compr`), step 3 (`chiron_scfa_sub` → q_perp), step 4 (`scfa_depthwise_causal_conv_fwd` → y_perp).
- Branch B on `computeStream`: step 5 (SCFA inner attention → y_compr), step 6 (cuBLAS `y_par = B·y_compr`).

Iter 8 was SMOKE only (verified flag activates).  Backward parallel branches auto-disable when `--scfa-checkpoint-inner` is on (line 6893) which is the production default.

Conjecture: post-iter69 the production stack might let parallel-branches deliver +5-8% wall via stream overlap.  Brief category: Proven-family extension (SCFA variant).

## Bench (L=24 T=16384, 200 steps, seed=1337)

Stack: production iter69 + iter65 (`--scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt --bf16-residual-p --scfa-checkpoint-inner --scfa-checkpoint-inner-bf16` + iter49 + iter61 + flags).

| metric                       | iter 69 baseline | iter 71 (parallel-branches) |
|---                           |---:              |---:                          |
| steady-state tok/s (mean)    | 24,320           | 22,995                       |
| wall (200 steps)             | 136.5 s          | 144.5 s                      |
| **Δ tok/s vs baseline**      | —                | **−5.45%**                   |
| **Δ wall vs baseline**       | —                | **+5.86%** (SLOWER)          |
| step-200 val NLL             | 7.7583           | 7.5900                       |
| Δ NLL @ step 200             | —                | **−0.168** (BETTER, iter 70 pattern) |
| peak VRAM                    | 14.97 GB         | 14.97 GB                     |

## Why this fails

**cuBLAS BF16-TC saturation on Ada**: branch A's cuBLAS GEMMs (FAST_16BF) and branch B's SCFA inner BF16-TC GEMMs both saturate the GPU's tensor cores.  Stream concurrency is theoretical only — the GPU's hardware scheduler serializes TC ops across streams.  Net: stream synchronization overhead (cudaEventRecord + cudaStreamWaitEvent + side cuBLAS context switch) IS paid, but the parallelism benefit doesn't materialize.

Same NULL pattern as iter 6 (`--bf16-logits-parallel-bwd` for readout backward GEMMs across streams, +0.15%) and iter 58 (revisit of same flag, +0.15%).  Both are Ada cuBLAS-TC-bound NULL.

**NLL drift in better direction**: training loss at step 1 differs by 0.001 in ||g|| (3.419 vs 3.420), reproducing the iter 70 sub-ULP scheduling-divergence pattern.  By step 200 the val NLL is 0.168 nat below baseline.  Same risk profile as iter 70 — different code path, slightly different (BETTER, this seed) trajectory.

## Categorization

**idea-fail at this hardware**.  Mechanism is correct in principle (stream parallelism for data-independent kernel chains), but Ada's tensor core contention prevents real overlap for the SCFA-class kernel mix.  No code path issue.  Would require sm_90+ (Hopper) async tensor core scheduling or non-TC kernel pairing to demonstrate benefit.

## Default policy

`--scfa-parallel-branches` remains **opt-in** (default OFF, unchanged from iter 8).  Mechanism stays in tree for documentation and as a Hopper-class candidate path.

## Sequence status (17 iters)

| iter | target | result | win |
|---: |---     |---     |---:  |
| 67  | BF16-p L=24 prod       | MARGINAL PASS      | n/a    |
| 68  | BF16-p default-on ship | SHIP               | +4.28% |
| 69  | BF16 checkpoint-inner cache | **PASS**      | **+6.83%** |
| 70  | fused axpy2_dual_p     | FAIL (NLL drift)   | +1.4% silent |
| 71  | --scfa-parallel-branches | **NEGATIVE**     | **−5.4%** |

## Bench command

```bash
build/glades_chiron_train --data-dir pretok-data --pretokenized --vocab 32000 \
  --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 \
  --lr 3e-4 --max-steps 200 --warmup 20 --grad-clip 1.0 \
  --val-every 100 --val-batches 4 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage \
  --scfa-checkpoint-inner --scfa-checkpoint-inner-bf16 \
  --scfa-parallel-branches \
  --seed 1337
```

## Files

- This document (iter 71 null result).
- No code changes — `--scfa-parallel-branches` was already wired in by iter 8.
