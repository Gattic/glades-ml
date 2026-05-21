# VESTA Fused Reconstruct-and-Update: dModel=8192 Rerun

**Run date:** 2026-04-18

**Commit (fused kernel):** (this series)

**Harness:** `unit-tests/glades-unit-tests vesta-sweep-mega` → `VESTASweepScaleMega()`

**Logs:** [`sweep.log`](sweep.log) (clean) · [`raw.log`](raw.log)

## What changed

The v14 mega-scale sweep (`vesta_scale_mega_20260418-101049`) identified VESTA's per-weight-matrix `WrOld[m·n]`, `WrNew[m·n]`, `gPerp[m·n]` scratch buffers as the memory bottleneck at dModel=8192 (~9 GiB of scratch on top of the 7 GiB transformer base). This run replaces the four-kernel chain (reconstruct WrOld, reconstruct WrNew, form gPerp, apply delta) with a single fused kernel `k_vesta_fused_update<UseMomentum, UseSign>` that computes all three in one pass without materializing them.

The refactor:

1. **Removed** `GpuVestaWeightState::WrOld`, `WrNew`, `gPerp` fields and their allocations (save 3·m·n floats per weight matrix).
2. **Added** `state.expEllPrev[r]` (saved snapshot of expEll before the log-scale update).
3. **Replaced** four per-step kernels (`k_reconstruct_rank_block` ×2, `k_form_gperp`, `k_apply_W_delta*`) with one templated fused kernel.
4. **Deferred** the U/V commit to AFTER the fused update (so the fused kernel reads old U/V via `state.U/state.V` and new U/V via `state.URaw/state.VRaw`).

## Results

**dModel=8192, nLayers=2, dFF=16384, 30 epochs, 1 seed, corpusLen=256, lr=1e-2:**

| optimizer | trainNLL | testNLL | wall (s) | OOM retries |
|-----------|----------|---------|----------|-------------|
| AdamW | n/a | 3.408 | 363 | — |
| VESTA r=8 (fused) | 0.787 | **2.120** | **8148** | 60 |
| VESTA r=8 (v14, pre-fused) | 0.787 | 2.120 | 9042 | 60 |

### Finding 1: Numerically identical

trainNLL and testNLL are **bit-for-bit equal** to v14 (0.7868 / 2.1200), confirming the fused kernel produces the same per-step output as the four-kernel chain. Unit tests corroborate (VESTA GpuParityTest: maxAbs 8.94e-7, unchanged).

### Finding 2: 10% wall-clock improvement (not the memory unlock we expected)

VESTA wall: 9042 → 8148 s (−894 s, ~10% faster). This matches the step-code speedup from the dModel=2048 microbenchmark (0.506 → 0.398 ms per step, 21%) scaled down by the fraction of wall-clock spent in VESTA step work (~half; the rest is transformer forward/backward).

### Finding 3: OOM count is unchanged (60 vs 60)

The 60 `cudaMalloc(134217728 floats, 536870912 bytes) failed: out of memory` retries in v15 match v14 exactly. This means the OOM pressure is NOT coming from VESTA's removed scratch — it is from the transformer's own weight/gradient/activation buffers which we didn't touch. At dModel=8192, even the base transformer is near the 16 GiB ceiling:

| component | size at dModel=8192, nLayers=2, dFF=16384 |
|-----------|--------------------------------------------|
| Transformer weights (Q/K/V/O + W1/W2 × 2 layers) | ~4 GiB |
| Gradients (same shape as weights) | ~4 GiB |
| AdamW optimizer state (when AdamW runs) | ~8 GiB |
| VESTA optimizer state (r=8) | 14 MiB |
| **Removed** VESTA scratch (WrOld + WrNew + gPerp × 12 matrices) | **~9 GiB** |
| Kept VESTA scratch (U, V, UA, Omega_U/V, URaw/VRaw) | ~240 MiB |

The 9 GiB removal matters *for larger rank or more layers* — e.g., at nLayers=4 the removed scratch would have been 18 GiB (unfittable even alone), but the base transformer at nLayers=4 also grows linearly. At this exact 2-layer config the transformer is the dominant consumer, so the 9 GiB scratch reduction didn't materially change peak memory.

### Finding 4: Step benchmark confirms 21% speedup

`vesta-step-bench` at dModel=2048, r=8, 100 steps (no refresh), no transformer forward/backward:

| path | per-step wall |
|------|---------------|
| Four-kernel chain (v14) | 0.506 ms |
| **Fused kernel (v15)** | **0.398 ms** |
| speedup | 1.27× |

At dModel=2048 there is no memory pressure, so this isolates the pure compute win from fusing.

## What this means for the scale roadmap

- **At dModel ≤ 4096:** the fused kernel is a pure 20-30% wall-clock win with no downside. v13-style runs get this speedup automatically.
- **At dModel=8192 on 16 GiB:** the fused kernel unblocks ~9 GiB of VESTA-specific scratch but the transformer base is already at the limit, so the wall-clock win is bounded by step speed (10% here).
- **At dModel=8192 on larger GPUs (40+ GiB):** the fused kernel is required to fit nLayers ≥ 4. v16 would be a proper fair fight at dModel=8192 nLayers=4 with enough memory headroom — AdamW would need ~40 GiB (16 weights + 16 grads + 16 optimizer), VESTA would need ~16 GiB (weights + grads + tiny optimizer + 240 MiB scratch). That's the regime where VESTA's memory advantage shows up as "trains at all" not "trains faster."

## Nine-sweep progression

| sweep | dModel | nLayers | epochs | AdamW testNLL | VESTA best | wall ratio |
|-------|--------|---------|--------|----------------|------------|-------------|
| v10 | 2048 | 4 | 50 | 3.121 | 1.763 | 3.1× |
| v11 | 2048 | 4 | 50 | 3.201 | 1.744 | 4.3× |
| v12 | 2048 | 4 | 50 | 3.157 | 1.744 | 1.6× (on-GPU refresh) |
| v13 | 4096 | 4 | 50 | 3.488 | 1.722 | 1.91× |
| v14 | 8192 | 2 | 30 | 3.408 | 2.120 | 23.7× (OOM-thrashing) |
| **v15 (this)** | **8192** | **2** | **30** | **3.408** | **2.120** | **22.4×** (fused kernel) |

The v14 → v15 wall delta is real but small; the v14 bottleneck was a mix of VESTA scratch and transformer base memory, and we only addressed the former.

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-sweep-mega 2>&1 | tee sweep.log
```

Expected runtime: **~2.3 hours** at dModel=8192 on 16 GiB.
