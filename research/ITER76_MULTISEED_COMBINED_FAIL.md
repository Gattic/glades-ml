## Iter 76 — Multi-seed combined retro-ship eval (iter 70 + iter 73) — FAIL

**Date**: 2026-05-20
**Iter**: 76 (twenty-second iter under stacking-wins brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: **FAIL** combined retro-ship — multi-seed wall gain +1.71% (below +3% iter 60-precedent bar) AND multi-seed NLL drift +0.067 ± 0.054 nat (outside ±0.05 multi-seed parity bound).  Combined path also exhibits 3× higher NLL variance vs baseline (seed-dependent instability).

---

## Background — iter 75 established multi-seed methodology

iter 75 documented that single-seed parity comparison at L=24 T=16384 production scale is unreliable (run-to-run variance ±0.16 nat from same seed across program starts).  Future iter benches should use ≥3 seeds with ±0.05 nat multi-seed mean-drift parity bound.

iter 75 also showed combined iter 70 + iter 73 at single seed=1337 = +1.80% wall, drift +0.039 nat (just outside ±0.02 strict).  Iter 76 re-bench at proper multi-seed.

## Multi-seed bench

Each run: 200 steps, L=24 T=16384 m=2048, lr=3e-4, warmup=20, grad-clip=1.0.

### Baseline (default flags — iter 69 stack)

| seed | NLL @ step 200 | wall (s) |
|---:  |---:            |---:      |
| 1337 (rerun 1, from iter 75) | 7.6002 | 136.3 |
| 1337 (rerun 2, from iter 75) | 7.6020 | 136.4 |
| 1338 | 7.6290 | 136.4 |
| 1339 | 7.6305 | 136.5 |
| **mean (n=4)** | **7.6154** | **136.40** |
| **std** | **0.0162** | **0.07** |

### Combined iter 70 + iter 73

| seed | NLL @ step 200 | wall (s) |
|---:  |---:            |---:      |
| 1337 (from iter 75) | 7.6404 | 134.0 |
| 1338 | 7.7431 | 134.1 |
| 1339 | 7.6644 | 134.2 |
| **mean (n=3)** | **7.6826** | **134.10** |
| **std** | **0.0537** | **0.10** |

### Δ analysis

| metric | baseline | combined | Δ | multi-seed verdict |
|---     |---:      |---:      |---: |---     |
| tok/s mean (200 steps × 16384 / wall) | 24,023 | 24,434 | **+1.71%** | below +3% iter 60 bar |
| wall (200 steps) | 136.40 s | 134.10 s | **−1.69%** | real (means tight) |
| NLL @ step 200 mean | 7.6154 | 7.6826 | **+0.067 nat** | outside ±0.05 multi-seed |
| NLL std | 0.0162 | 0.0537 | 3.3× higher | seed-dependent instability |

The seed=1338 combined result (7.7431) is a +0.078 nat outlier vs the combined mean, while seed=1338 baseline (7.6290) is well within the baseline cluster.  This suggests the combined iter 70+73 paths introduce seed-dependent training instability — at some seeds the combined trajectory diverges meaningfully more than baseline.

## Why this fails

1. **Wall gain (+1.71%) is below the iter 60-precedent +3% combined retro-ship bar**.  The brief's strict per-iter +5% bar is also not met.  Combined retro-ship is not eligible for default-on ship.

2. **NLL drift (+0.067 nat) exceeds the ±0.05 nat multi-seed parity bound**.  Even accepting the relaxed multi-seed methodology from iter 75, the combined path drifts more than acceptable.

3. **Variance amplification (3.3×) on combined**.  Both iter 70 and iter 73 individually were assessed as "below-bar but plausibly parity-clean" at single seed; combined they introduce statistically significant seed-dependent instability.  Likely mechanism: iter 70's fused axpy2 changes SR-cast counter sequence (per-element rounding noise re-distributed); iter 73's tiled dwconv changes FMA scheduling; both compound through Adam pathway.

## Categorization

**impl-fail at multi-seed**.  Mechanisms are individually plausibly-parity-clean but combined produce drift + instability.

## Default policy

`--iter70-fused-axpy2-dual-p` and `--iter73-dwconv-fwd-tiled` remain **default OFF**.  Opt-in flags preserved for documentation.  Production uses the parity-validated iter 65 + iter 69 baseline path.

## Sequence status (22 iters)

| iter | result | win wall | category |
|---: |---     |---:      |---       |
| 68  | SHIP | +4.28%  | iter 65 BF16-p default-on |
| 69  | **PASS** | **+6.83%** | iter 69 BF16-cache |
| 70  | FAIL | +1.4% (single-seed) | impl-fail |
| 71  | NEGATIVE | −5.4% | Ada TC saturation |
| 72  | NULL | ±0% | below noise |
| 73  | FAIL | +0.56% | impl-fail |
| 74  | FAIL | +0.23% | noise |
| 75  | META | n/a | methodology correction |
| 76  | FAIL | +1.71% combined | multi-seed parity fail |

## Strategic implication

After 6 consecutive single-axis non-PASS iters + 1 META + 1 combined-retro FAIL, the **post-iter69 engineering ceiling is comprehensively documented**:
- Single-axis ≥+5% wins: **none available** in custom-kernel space (FMA-emit drift; cuBLAS dead-zone; Ada TC saturation; below-noise launch overhead).
- Combined retro-ship: **not viable** at +1.7% wall with seed-dependent instability.
- The remaining paths are **multi-iter architecture/training arcs**:
  - **Hadamard basis** (FWHT compress/lift vs DCT-II cuBLAS): potential +3-7% wall but requires retraining the production model
  - **FlashAttention-fused inner attention** (custom kernel for SCFA inner Q@K^T → softmax → @V): potential +5-15% wall but multi-day implementation
  - **Mixture-of-depth** (per-layer skip routing for inference): potential 1.5-2× inference but architectural change

## Bench commands

Baseline:
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
  --seed {1337,1338,1339}
```

Combined: same command + `--iter70-fused-axpy2-dual-p --iter73-dwconv-fwd-tiled`.

## Files

- This document.
- `research/runs/2026-05-19-iter76-bench/baseline_seed{1338,1339}.log` and `combined_seed{1338,1339}.log`.
- No code change.
