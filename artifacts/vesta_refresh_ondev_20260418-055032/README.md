# VESTA On-GPU Refresh: End-to-End Speedup

**Run date:** 2026-04-18

**Commit:** `4a60b6c6c` — `perf(vesta): on-GPU sketched-SVD refresh (240x over host roundtrip)`

**Harness:** `unit-tests/glades-unit-tests vesta-sweep-rank` → `VESTASweepRankAtScale()`

**Raw log:** [`sweep.log`](sweep.log) (with per-epoch telemetry filtered), [`raw.log`](raw.log) (with telemetry)

## What changed

The v11 rank sweep (`vesta_rank_20260417-180114`) identified on-GPU sketched-SVD refresh as the highest-leverage remaining VESTA optimization — wall-clock scaled super-linearly with rank (320 → 1246s as r goes 8 → 64), consistent with the O(r²·n + r³) signature of the host-roundtrip refresh.

This run re-executes the same rank sweep after committing the on-device refresh (`4a60b6c6c`). The config knob `VestaConfig::gpuRefreshOnDevice` defaults to `true`, so any training run that doesn't explicitly disable it now uses the on-device path. Parity tests set it to `false` to preserve bit-level CPU/GPU agreement.

## Results

**dModel=2048, 50 epochs, GPU, 3 seeds per row.** AdamW reference: **testNLL 3.157 ± 0.124, wall 100.1 s.**

| rank | testNLL | wall (s) | opt MiB |
|------|---------|----------|---------|
| r=4 | 1.884 ± 0.053 | **146** | 1.75 |
| r=8 | 1.831 ± 0.065 | **164** | 3.50 |
| r=16 | 1.830 ± 0.057 | **196** | 7.00 |
| r=32 | **1.782 ± 0.057** | **249** | 14.00 |
| r=64 | **1.744 ± 0.059** | **428** | 28.00 |

## Speedup vs v11 (host roundtrip refresh)

Same config, same seeds, same scale — only the refresh path changed:

| rank | v11 testNLL | v12 testNLL | Δ NLL | v11 wall | v12 wall | **speedup** |
|------|-------------|-------------|-------|----------|----------|-------------|
| r=4 | 2.240 ± 0.634 | 1.884 ± 0.053 | **−0.36** | 270 s | 146 s | **1.85×** |
| r=8 | 1.847 ± 0.051 | 1.831 ± 0.065 | −0.02 | 320 s | 164 s | **1.95×** |
| r=16 | 1.836 ± 0.060 | 1.830 ± 0.057 | −0.01 | 460 s | 196 s | **2.35×** |
| r=32 | 1.782 ± 0.057 | 1.782 ± 0.057 | 0.00 | 671 s | 249 s | **2.70×** |
| r=64 | 1.744 ± 0.060 | 1.744 ± 0.059 | 0.00 | 1246 s | 428 s | **2.91×** |

## Findings

### 1. Speedup scales with rank (as predicted)

The v11 analysis hypothesized the super-linear wall-clock scaling in `r` was due to the sketched-SVD refresh (O(r²·n + r³) cost). The on-device refresh keeps the expensive GEMMs on GPU — only the tiny `B[rp × n]` matrix (rp ≈ r+8, n = dModel) is downloaded for the Jacobi-based SVD.

Per-rank speedup grows from **1.85× at r=4** to **2.91× at r=64**, exactly the pattern expected if SVD refresh was the dominant bottleneck at higher rank. At r=64, v11 spent ~800s in refresh and ~400s in the actual optimizer step; v12 spends ~30s in refresh and ~400s in step, so the non-refresh portion is unchanged and the speedup ceiling converges to the inverse of the refresh fraction.

### 2. NLL is unchanged (within noise)

For r ≥ 8, testNLL agrees between v11 and v12 to within 0.02 nats — comfortably inside the 0.06-nat std. across seeds. The on-device refresh produces essentially the same trajectory as the host roundtrip; the only numerical difference is cuBLAS vs. scalar-SGEMM rounding, which microbenchmarks showed to be ~7e-7 per U/V element after a single refresh. The implicit 2 power-iterations keep the range-finder estimate well-conditioned regardless.

### 3. r=4 stability actually improved

v11's r=4 had a diverging seed (trainNLL 2.97 on one of three seeds, driving the 0.78 variance). v12's r=4 is tight at 1.88 ± 0.05 — no divergence. This is surprising; the most likely explanation is that the v11 host-roundtrip refresh occasionally produced worse rank-deficiency fills in the low-rank regime, and the on-device refresh's CPU-matched RNG consumption is slightly more stable for that corner case. Not a claim worth pushing on, since r=4 isn't a recommended config, but it's a welcome side-effect.

### 4. VESTA is now competitive with AdamW on wall-clock

| rank | v11 VESTA/AdamW wall | v12 VESTA/AdamW wall |
|------|----------------------|----------------------|
| r=8 | 3.2× | **1.6×** |
| r=16 | 4.6× | 2.0× |
| r=32 | 6.7× | 2.5× |
| r=64 | 12.4× | 4.3× |

At r=8 (the NLL sweet spot) VESTA now runs at **1.6× AdamW wall-clock** while delivering **−1.33 nats** on testNLL (1.83 vs 3.16) and using **0.04% of AdamW's optimizer memory**. This is the first time VESTA has been within 2× wall-clock of AdamW at this scale.

## Microbenchmark: raw refresh timing

`vesta-refresh-bench` isolates a single `vesta_gpu_refresh` call at dModel=2048, rank=8:

| path | per-refresh wall | |
|------|------------------|---|
| host-roundtrip | 994 ms | download 16 MiB W + CPU SVD + upload 16 MiB |
| **on-device** | **4.1 ms** | cuBLAS GEMMs + tiny CPU SVD on 32 KiB B |
| speedup | **241×** | |

The microbenchmark overshoots the end-to-end speedup (241× vs 2-3×) because `vesta_gpu_step` spends most of its time in the per-step GEMMs (WrOld/WrNew reconstruction, UA = gW·V, etc.), not in the refresh. With refresh now near-free, those step GEMMs become the next bottleneck.

## Nine-sweep progression (updated with v12)

| sweep | config | AdamW wall | VESTA best wall (vs AdamW) |
|-------|--------|------------|-----------------------------|
| v10 | dModel=2048, 50ep, GPU, host refresh | 100s | 308s (3.1×) |
| v11 | dModel=2048, 50ep, GPU rank sweep, host refresh | 100s | r=8: 320s (3.2×); r=64: 1246s (12×) |
| **v12 (this)** | **dModel=2048, 50ep, GPU rank sweep, on-device refresh** | **100s** | **r=8: 164s (1.6×); r=64: 428s (4.3×)** |

## What remains

1. **Per-step GEMM consolidation**: with refresh no longer dominant, the next bottleneck is the sequence of small GEMMs per weight matrix in `vesta_gpu_step` (UA, A, UB, UtOmU, VtOmV, Omega_U, Omega_V). At r=8 with many weight matrices, these are individually small and leave the GPU underutilized. A batched-pointer variant (analogous to the transformer attention path) could close another 1.5-2× gap.

2. **Remove host-path fallback in production**: the `gpuRefreshOnDevice = false` path stays for parity tests, but the default of `true` now gets the full speedup automatically.

3. **Test dModel=4096**: memory ratio drops to 0.18%. With refresh tractable on GPU (previously host-bottlenecked at large m×n), this is the next scale test.

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-sweep-rank 2>&1 | tee sweep.log
```

Expected runtime: ~25 minutes (vs ~50 minutes before the refresh fix).
