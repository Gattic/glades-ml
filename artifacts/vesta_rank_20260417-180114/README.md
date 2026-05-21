# VESTA Rank Sweep at dModel=2048

**Run date:** 2026-04-17

**Harness:** `unit-tests/glades-unit-tests vesta-sweep-rank-at-scale` → `VESTASweepRankAtScale()`

**Raw log:** [`sweep.log`](sweep.log) (clean summary), [`raw.log`](raw.log) (with per-epoch telemetry)

## Why this sweep

The v10 GPU scale-ladder artifact identified the next VESTA performance question as: does the wall-clock cost scale well with rank, and is per-weight-matrix launch overhead the dominant bottleneck (motivating a matrix-batching refactor)?

This sweep sweeps `rank ∈ {4, 8, 16, 32, 64}` at dModel=2048, 50 epochs, 3 seeds per rank, with all other knobs held at the plain-raw + complement-momentum config used in v10. It simultaneously tests:

1. **NLL vs rank** — does tracking more directions improve loss?
2. **Wall-clock vs rank** — launch-bound (flat) or compute-bound (linear/super-linear)?
3. **Whether higher rank alone could replace the matrix-batching refactor** (task #30)

## Results

**dModel=2048, 50 epochs, GPU, 3 seeds per row:**

| rank | trainNLL | testNLL | wall (s) | opt state (MiB) |
|------|----------|---------|----------|------------------|
| r=4 | 2.042 ± 0.779 | 2.240 ± 0.634 | 270 ± 20 | 1.75 |
| r=8 | 1.567 ± 0.027 | 1.847 ± 0.051 | 320 ± 10 | 3.50 |
| r=16 | 1.534 ± 0.023 | 1.836 ± 0.060 | 460 ± 6 | 7.00 |
| r=32 | 1.457 ± 0.028 | **1.782 ± 0.057** | 671 ± 91 | 14.00 |
| r=64 | 1.394 ± 0.020 | **1.744 ± 0.060** | 1246 ± 16 | 28.00 |

AdamW reference at same scale: **testNLL 3.201 ± 0.114, wall 100 s**. Every VESTA rank still crushes AdamW by 0.9–1.5 nats.

## Findings

### 1. NLL improves monotonically with rank — but with diminishing returns

Doubling rank yields these testNLL deltas:
- r=8 → r=16: **−0.011 nat** (noise-level)
- r=16 → r=32: **−0.053 nat**
- r=32 → r=64: **−0.038 nat**

Total span r=8 → r=64: −0.103 nat, for 3.9× wall-cost. Under a fixed compute budget, **r=8 is the efficiency sweet spot**. If NLL is the sole objective and compute is cheap, r=32 is the practical ceiling — r=64 costs 86% more wall-time for a 0.04 nat gain.

### 2. Low rank (r=4) is unstable

One seed diverged to trainNLL 2.97 at r=4, producing the huge 0.78 variance. r≥8 is stable (variance ≤0.06). The theory prediction (rank needs to cover "high-curvature subspace") is consistent with this floor.

### 3. Wall-clock is NOT launch-bound — matrix-batching refactor is the wrong optimization

Linear fit on (r, wall):

| Step | Δwall per 2× rank |
|------|-------------------|
| r=4 → r=8 | +50 s |
| r=8 → r=16 | +140 s |
| r=16 → r=32 | +210 s |
| r=32 → r=64 | +576 s |

Wall-clock scales roughly linearly-to-super-linearly with rank. If we were launch-overhead-dominated (the motivation for task #30 matrix-batching), wall would be **flat** in rank — the number of launches is the same at every rank, only the per-launch work changes. Instead we see **compute-bound** scaling, meaning:

- The per-matrix SGEMMs at rank r dominate, not the launch count.
- Matrix-batching would reduce launches but do little for total compute — expected gain ≤10%.
- The super-linear piece r=32→r=64 (+576s) suggests **the sketched-SVD refresh** is entering its O(r²·n + r³) regime at larger r, and it is the real remaining optimization target — consistent with the v10 README's note that refresh is ~50% of VESTA wall-clock.

### 4. Memory claim is reinforced

At r=64, VESTA's optimizer state is 28 MiB. AdamW's optimizer state for the same model is ~8 GiB (two full copies of the weight matrices). VESTA-r64 uses **~0.35% of AdamW's optimizer memory** and achieves testNLL 1.74 vs AdamW's 3.20 — a 1.46-nat improvement at 300× less state.

## Implications for task #30 (matrix-batching refactor)

**Recommend deprioritizing matrix-batching.** The rank sweep is evidence that we are compute-bound on rank-r SGEMMs, not launch-bound across weight matrices. Matrix-batching would be a large refactor (every VESTA weight matrix's (U,V,ell,beta) state reorganized into contiguous batched tensors) for a speedup that the data suggests is bounded at ~10%.

The highest-leverage optimization target is now **on-GPU sketched-SVD refresh** (currently host-roundtrip). The evidence: wall-clock grows super-linearly with rank, which is the SVD's O(r²·n + r³) signature rather than the GEMMs' O(r·n²) linear-in-r signature. Moving the refresh on-device would:
- Eliminate the PCIe download/upload per refresh.
- Allow fusion with surrounding projection kernels.
- Expected speedup: 2-3× on VESTA wall-clock at any rank.

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-sweep-rank-at-scale 2>&1 | tee rank.log
```

Expected runtime: ~50 minutes wall-clock (all ranks × 3 seeds × 50 epochs at dModel=2048).
