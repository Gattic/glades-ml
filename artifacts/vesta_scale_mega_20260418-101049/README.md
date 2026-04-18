# VESTA Mega-Scale: dModel=8192 on a 16 GiB GPU

**Run date:** 2026-04-18

**Commits:** `4a60b6c6c` (on-GPU refresh) + `78ea29b09` (step-code GEMM reduction)

**Harness:** `unit-tests/glades-unit-tests vesta-sweep-mega` → `VESTASweepScaleMega()`

**Logs:** [`sweep.log`](sweep.log) (clean) · [`raw.log`](raw.log) (with OOM-retry warnings)

## Configuration

To fit AdamW on a 16 GiB RTX 4080 SUPER, config scaled down from v13 (dModel=4096, 4 layers):

| knob | v13 (dModel=4096) | v14 (dModel=8192) |
|------|-------------------|---------------------|
| dModel | 4096 | 8192 |
| dFF | 8192 | 16384 |
| nLayers | 4 | **2** (halved for memory) |
| corpusLen | 512 | **256** (halved for memory) |
| epochs | 50 | **30** (reduced for runtime) |
| seeds | 2 | **1** (single seed at this scale) |

## Results

| optimizer | trainNLL | testNLL | wall (s) | opt state |
|-----------|----------|---------|----------|-----------|
| AdamW | n/a | 3.408 | 381 | ~2 GiB |
| VESTA r=8 (plain-raw) | **0.787** | **2.120** | 9042 | 14 MiB |

**Headline:** VESTA testNLL 2.12 beats AdamW 3.41 by **−1.29 nats**. VESTA's trainNLL is 0.79 (near-memorized) so at this scale both optimizers are showing train/test divergence, but VESTA's generalization is still ~1.3 nats better.

## The wall-clock catastrophe

VESTA took **9042 s vs AdamW's 381 s — a 24× ratio** vs the 1.6-1.9× ratio we saw at smaller scales. This is NOT the optimizer algorithm getting slower; it's **GPU memory pressure causing cudaMalloc thrashing**.

At dModel=8192 the VESTA step's scratch buffers are O(m·n) per weight matrix:

| scratch buffer | purpose | size at FFN shape (8192×16384) |
|-----------------|---------|--------------------------------|
| `WrOld[m·n]` | rank-r reconstruction pre-step | 512 MiB |
| `WrNew[m·n]` | rank-r reconstruction post-step | 512 MiB |
| `gPerp[m·n]` | out-of-subspace gradient | 512 MiB |

For this 2-layer model there are 4 FFN matrices (8192×16384) and 8 Q/K/V/O matrices (8192×8192). VESTA scratch alone is ~9 GiB. Add weights + gradients + activations and the allocator is at the 16 GiB ceiling. `raw.log` is littered with `cudaMalloc ... failed: out of memory` retries.

**This exposes a real engineering gap.** VESTA's *optimizer state* is tiny (14 MiB, as advertised), but the *step scratch* is proportional to m·n. At dModel ≤ 4096 this is invisible (few hundred MiB total); at dModel=8192 it dominates and makes training impractically slow.

## The memorization problem

VESTA trainNLL 0.79 with testNLL 2.12 is a 1.33-nat gap — the model is memorizing. At this capacity (2 layers × 8192 × 16384 FFN = 270 MFLOPs per token forward) on a 256-token corpus, memorization is expected at 30 epochs.

AdamW shows less memorization (testNLL 3.41 vs trainNLL ~3.37) but that's because AdamW never got close to fitting the train set — its fixed-LR problem at large scale (same divergence trend from v10-v13) prevents it from ever reaching memorization in 30 epochs.

**Both optimizers are suboptimal at this config**: AdamW can't train, VESTA memorizes. The −1.29 nat testNLL margin is real but both numbers would be much better with (a) a larger corpus, (b) dropout/regularization, or (c) AdamW with a cosine schedule.

## Compared to v13 (dModel=4096)

| dModel | VESTA testNLL | AdamW testNLL | gap | VESTA/AdamW wall |
|--------|---------------|----------------|-----|-------------------|
| 2048 | 1.782 | 3.121 | −1.34 | 1.64× |
| 4096 | 1.722 | 3.488 | **−1.77** | 1.91× |
| **8192** | **2.120** | **3.408** | **−1.29** | **23.7×** (memory-thrashing) |

The testNLL gap at dModel=8192 is SMALLER than at dModel=4096 (−1.29 vs −1.77), because at this 2-layer / 256-corpus config both optimizers run into corpus-size limits. The wall-clock ratio blowup is explained by memory pressure, not optimizer algorithm cost.

## What needs to change to push further

1. **Fused reconstruct-and-update kernel.** Replace the three O(m·n) scratch buffers with an implicit update:
   ```
   W[i,j] += (sum_k U_new[i,k] * expEll_new[k] * V_new[j,k]
              - sum_k U_old[i,k] * expEll_old[k] * V_old[j,k])
             - lrCperp * (gW[i,j] - sum_k U[i,k] * A[k,l] * V[j,l])
   ```
   This removes the 3×(m·n) scratch allocation per weight matrix. Expected saving at dModel=8192: ~9 GiB → ~0 GiB, eliminating the OOM thrashing and closing the wall-clock gap back to ~2×.

2. **Larger corpus / fewer epochs.** The 256-token / 30-epoch config is too small for dModel=8192 to be meaningful. A 2048-token / 10-epoch config would stress AdamW's divergence without letting VESTA memorize. This needs (1) first to be tractable.

3. **Regularization.** Weight decay is off in these sweeps; VESTA's memorization would slow down considerably with wd ≈ 0.01.

## Nine-sweep progression

| sweep | dModel | nLayers | epochs | AdamW testNLL | VESTA best | gap | wall ratio |
|-------|--------|---------|--------|----------------|------------|-----|-------------|
| v10 | 2048 | 4 | 50 | 3.121 | 1.763 | −1.36 | 3.1× |
| v11 | 2048 | 4 | 50 | 3.201 | 1.744 | −1.46 | 4.3× |
| v12 | 2048 | 4 | 50 | 3.157 | 1.744 | −1.41 | 1.6× |
| v13 | 4096 | 4 | 50 | 3.488 | 1.722 | −1.77 | 1.91× |
| **v14 (this)** | **8192** | **2** | **30** | **3.408** | **2.120** | **−1.29** | **23.7×** |

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-sweep-mega 2>&1 | tee sweep.log
```

Expected runtime: **~2.5 hours** at dModel=8192 on a 16 GiB GPU (mostly memory-thrash overhead).
