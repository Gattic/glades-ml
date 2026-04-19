# VESTA Ultra-Scale Rerun: Post-Perf-Session Wall-Clock

**Run date:** 2026-04-18

**Relevant commits (this session):**
- `2305934db` — flash attention warp-parallel dot product
- `f8cd50a59` — flash attention multi-query per block
- `019c870e1` — VESTA init via on-device sketched SVD (THE big one)

**Harness:** `unit-tests/glades-unit-tests vesta-sweep-ultra` → `VESTASweepScaleUltra()`. Same config as v13/v14 (dModel=4096, 50 epochs, 2 seeds, nLayers=4).

**Log:** [`sweep.log`](sweep.log) (clean), [`raw.log`](raw.log)

## The headline

| optimizer | v14 wall | post-fix wall | speedup | testNLL |
|-----------|----------|---------------|---------|---------|
| AdamW | 355 s | **29.7 s** | **12×** | 3.510 ± 0.355 |
| VESTA r=8 | 681 s | **30.5 s** | **22×** | **1.722 ± 0.022** |
| VESTA r=16 | 843 s | **32.6 s** | **26×** | **1.723 ± 0.022** |

### VESTA/AdamW wall ratio

|              | ratio | attribution |
|--------------|-------|-------------|
| v10 (pre-session baseline) | 3.1× | host-roundtrip refresh, CPU init, unoptimized flash attn |
| v12 (on-device refresh done) | 1.64× | — |
| v14 (step-code optimizations) | 1.92× | slight regression due to other work |
| **post-fix (this run)** | **1.03×** | **VESTA is essentially as fast as AdamW** |

VESTA r=8 now runs at **1.03× AdamW wall-clock** while beating AdamW by **1.79 nats** on testNLL and using 0.04% of AdamW's optimizer memory. At dModel=4096, 4 layers, 512-token contexts.

## Where each speedup came from

### Flash attention (shared benefit, affects AdamW too)

Two independent wins to the fused-attention kernels:

1. **Warp-parallel dot product + decoupled block size** (`2305934db`): inner Q·K inner-product loop had been running sequentially while redundantly executed by every thread in a block of 8 threads (shared-memory-coupled). Fixed to warp-parallel with `__shfl_down_sync`. Kernel times: fwd 617→143 ms (4.3×), bwd 1054→239 ms (4.4×).

2. **Multi-query per block** (`f8cd50a59`): one block now processes QROWS=4 query rows with shared K/V tile loads. Additional 2.3×/1.8× on top of #1. Final kernel times: fwd 143→11.1 ms, bwd 239→31.9 ms.

Cumulative flash attention speedup: **fwd 56×, bwd 33×**. At 50 epochs with 4 layers × 2 passes × 50 steps = 400 attention calls, the 20 s of attention work per 3-epoch run becomes 0.5 s. For the full 50-epoch run, attention savings alone are ~330 s — the bulk of the 325 s AdamW improvement.

### VESTA init via on-device sketched SVD (`019c870e1`)

`perf record` on `vesta-profile-bench` showed 83.5% of wall was in `glades::vesta::sketched_svd` called from `vesta_gpu_init`. At dModel=4096 with ~24 weight matrices including FFN (4096×8192 and 8192×4096), the CPU sketched_svd does ~125 GFLOPs of work at ~0.5 GFLOP/s of effective throughput = **~270 seconds per run**.

We already had the GPU equivalent (`vesta_gpu_refresh_device`, committed in `4a60b6c6c` earlier this session) — it just wasn't wired into init. Fix: init now uses the same on-device sketched SVD. The 270 s of CPU work becomes ~1 s of GPU work.

## Cumulative session impact

At dModel=4096, 50-epoch, 2-seed ultra sweep:

| generation | total runtime (all 3 configs) |
|------------|-------------------------------|
| v14 (start of session) | 1879 s = 31 min |
| post-fix (now) | **93 s = 1 min 33 s** |
| speedup | **20× for the whole sweep** |

This is a **purely wall-clock** improvement — identical NLL, identical optimizer trajectory within float rounding (trainNLL and testNLL match to 3+ decimal places).

## What it means for large-LLM training

VESTA was pitched as "beats AdamW on testNLL but pays a wall-clock premium." That premium is now gone. At dModel=4096 you can train with VESTA instead of AdamW at essentially the same cost, receive −1.79 nats on testNLL, and use 1/2500th of the optimizer state memory (7 MiB vs ~4 GiB).

For very long contexts (T=8K, 32K, 128K) the flash attention work becomes critical again — we've set up the architecture (multi-query) to support a future WMMA tensor-core upgrade when we actually run those workloads.

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-sweep-ultra 2>&1 | tee sweep.log
```

Expected runtime: **~1.5 minutes** (was ~31 minutes before this session).
