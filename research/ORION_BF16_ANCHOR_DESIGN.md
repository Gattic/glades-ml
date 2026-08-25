# ORION BF16 theta_anchor — staged design

**Date**: 2026-05-17 (during Gate-0 probe wait)
**Scope**: Halve theta_anchor VRAM from FP32 to BF16. Direct port of iter 69's BF16-checkpoint-inner pattern. ~1.7 GB saved at 1B model.

---

## Mechanism

In ORION's current code (chiron_main.cpp `OrionTensor`):

```cpp
struct OrionTensor {
    ...
    glades::gpu::GpuBuffer<float> theta_anchor;   // FP32, n elements
    glades::gpu::GpuBuffer<float> g_anchor;       // FP32, n elements
    ...
};
```

`theta_anchor` is used at exactly two points per anchor cycle:
1. **Snapshot** (start of anchor step): `device_memcpy_d2d(theta_anchor, t->theta, n*4)`
2. **Lift-back** (end of anchor): `theta = theta_anchor + V·(α − α_anchor)`

Between these, `theta_anchor` is read-only.  No accumulation, single round-trip per anchor — identical to iter 69's checkpoint-inner reload pattern, which we already validated is safe at BF16 with deterministic round-to-nearest-even.

## Code change (minimal diff sketch)

`glades-trainer/trainer/chiron_main.cpp`:

```cpp
struct OrionTensor {
    ...
    // ORION v4 quantized anchor storage (opt-in via --orion-bf16-anchor):
    glades::gpu::GpuBuffer<uint16_t> theta_anchor_bf16;   // [n], 2 bytes/elem
    glades::gpu::GpuBuffer<float>    theta_anchor;        // [n], 4 bytes/elem (kept for FP32 path)
    bool                              anchor_is_bf16;
    ...
};
```

`OrionState::addTensor()` chooses one based on `cfg.orionBf16Anchor`:

```cpp
if (cfg.orionBf16Anchor) {
    if (!t->theta_anchor_bf16.allocate(n)) { delete t; return false; }
    t->anchor_is_bf16 = true;
} else {
    if (!t->theta_anchor.allocate(n)) { delete t; return false; }
    t->anchor_is_bf16 = false;
}
```

At snapshot (replace `device_memcpy_d2d` with `cast_f32_to_bf16`):

```cpp
if (t->anchor_is_bf16) {
    glades::gpu::cast_f32_to_bf16(t->theta, t->theta_anchor_bf16.data(), t->n);
} else {
    glades::gpu::device_memcpy_d2d(t->theta_anchor.data(), t->theta, sizeof(float)*t->n);
}
```

At lift-back (replace `theta := theta_anchor + V·Δα` with: decode anchor into scratch, add the V·Δα part):

```cpp
float* anchor_fp32_ptr;
if (t->anchor_is_bf16) {
    // decode into the existing FP32 working buffer (e.g. t->theta itself is overwritten anyway)
    glades::gpu::cast_bf16_to_f32(t->theta_anchor_bf16.data(), t->theta, t->n);
    anchor_fp32_ptr = t->theta;   // theta now holds anchor
} else {
    anchor_fp32_ptr = t->theta_anchor.data();
}
// existing path: theta += V · (α − α_anchor)  (using anchor as base)
for (int k = 0; k < r; ++k)
    orion_axpy_int8_v_column(t->theta, V_q, V_scales, n, k, delta_alpha[k], stream);
```

Subtle invariant: when `anchor_is_bf16=true`, the lift-back's decode writes into `t->theta` (the same buffer that gets the V·Δα accumulation). So the order is: (1) decode anchor into theta, (2) accumulate V·Δα into theta. Same end-state as the FP32 path.

## CLI

```cpp
else if (streq(a, "--orion-bf16-anchor"))    cfg.orionBf16Anchor = true;
else if (streq(a, "--no-orion-bf16-anchor")) cfg.orionBf16Anchor = false;
```

Default off; opt-in for the staged-down-VRAM variant.

## VRAM impact at 1B model

| state | naive ORION | + BF16 anchor | savings |
|---|---:|---:|---:|
| theta_anchor (E + 24·4 paramsets) | 3.49 GB | 1.74 GB | **−1.74 GB** |

## Numerical impact

BF16 has 7 mantissa bits → relative error ≈ 2⁻⁸ = 0.39 %.

The anchor's role at lift-back is as the additive base: `θ_new = θ_anchor + δ`. The δ comes from V·Δα — its magnitude across an anchor window is roughly `lr · ‖g_∥‖ · K ≈ 1e-4 · 1.0 · 20 = 2e-3` (small).  BF16 anchor introduces ≤ 0.4 % error in the anchor's magnitude, while the δ being added has magnitude ~2e-3 relative to anchor's ~1.0. The compounding error on θ_new is therefore dominated by BF16's own weight-storage precision (the master weight is already BF16), not by the anchor cast.

**Verdict**: BF16 anchor is a clean win — same precision regime as the existing BF16 weights, no new error compounding.

## Implementation cost

- ~30 lines of code change to `OrionTensor` + `addTensor()` + 2 use sites
- 0 new kernels (uses existing `cast_f32_to_bf16` / `cast_bf16_to_f32` from gpu_kernels.cu)
- Runtime cost: 1 extra cast launch at snapshot + 1 at lift-back per anchor. With K=20 that's ≈ 0.1 % overhead.

## Combined with INT8 V

| ORION VRAM at 1B r=4 | naive | + INT8 V | + BF16 anchor | + both |
|---|---:|---:|---:|---:|
| V                | 6.97 GB | 3.49 GB | 6.97 GB | **3.49 GB** |
| theta_anchor     | 3.49 GB | 3.49 GB | 1.74 GB | **1.74 GB** |
| g_anchor (transient — see below) | 3.49 GB | 3.49 GB | 3.49 GB | 3.49 GB |
| **persistent total** | **13.95 GB** | **10.47 GB** | **12.20 GB** | **8.72 GB** |

If we additionally make `g_anchor` transient (free after V^T·g_anchor projection at start of anchor), persistent state at r=4 drops to **5.23 GB** — fits at T=8192 base of ~10 GB total (15 GB), close at T=16384.
