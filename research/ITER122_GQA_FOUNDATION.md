## Iter 122 — GQA r=4 foundation: config + validation

**Date**: 2026-05-21
**Iter**: 122 (Phase 1 of GQA r=4 multi-iter arc, per iter 121 scoping)
**Branch**: vesta5 (glades-trainer only)
**Verdict**: **Foundation in place; ratio=1 sanity PASS; ratio=4 not yet functional.** Compile verified, ratio=1 default unchanged. The actual GQA enablement (weight resizing, broadcast/reduce kernels, fwd/bwd path changes) requires iter 123-126 to complete.

---

## What iter 122 delivers

**1. Config + CLI**:
- `Config::gqaRatio` field (default 1)
- `--gqa-ratio N` CLI flag
- Validation: `nH % gqaRatio == 0` required

**2. ChironParams shape fields**:
- `gqaRatio`, `nKVHeads`, `dModelKV` added
- Computed in `allocate(const Config&)`:
  - `nKVHeads = nH / gqaRatio`
  - `dModelKV = nKVHeads * dH = dModel / gqaRatio`
- Log message when gqaRatio > 1 to confirm dimensions

**3. Sanity bench (gqaRatio=1 default)**:
- 25-step training: NLL trajectory matches current production
- Step 1 init NLL: 10.5189 (matches iter 116 ship)
- Step 25 final-val NLL: 10.4414
- Wall: 14.7s (matches iter 116 ship baseline)
- **Bit-identical to pre-iter-122 production at gqaRatio=1**

## What iter 122 does NOT yet deliver

The following are required for GQA r=4 to actually function (no crash, correct math):

**Weight allocation** (iter 123 Phase 2):
- Wk_bf, Wv_bf at `m * dModelKV` instead of `m * dModel`
- dWk_bf, dWv_bf same
- Adam state for Wk, Wv: `Wk_mb/vb/mi/vi/ms/vs`, `Wv_*` — 12 buffers at compressed size
- FP32 fallback Wk, Wv if `--no-bf16-weights` (less common path)

**Forward broadcast kernel** (iter 123 Phase 2):
- `gqa_broadcast_kv_fp32(src_compressed, dst_full, T, nKVHeads, nHeads, dH)`
- Replaces cuBLAS direct write to nHeads-wide sK/sV with: cuBLAS into compressed scratch + broadcast to full

**Backward reduce kernel** (iter 124 Phase 3):
- `gqa_reduce_dkv_fp32(src_full, dst_compressed, T, nKVHeads, nHeads, dH)`
- Sum sdK/sdV across query heads in each KV group before computing dWk, dWv gradients

**Forward path** (iter 123): wire broadcast into `chiron_attention_shear_bf16w_bf16g_tiled` fwd

**Backward path** (iter 124): wire reduce into bwd, use compressed Wk_bf, Wv_bf

**Checkpoint format** (iter 125): encode nKVHeads in CHRN/CHRF header

**30k Phase 2 retrain** (iter 125): validate NLL stays within ±0.25 nat of iter 116 ship baseline

**Default-flip + ship** (iter 126): if Phase 2 PASS, gqaRatio=4 becomes default

## Realistic remaining effort

| iter | scope | LOC | hours |
|---:|---|---:|---:|
| 123 | weight alloc + broadcast kernel + fwd path | ~200 | 4-6 |
| 124 | reduce kernel + bwd path + Adam state alignment | ~250 | 6-10 |
| 125 | checkpoint format + 30k retrain (+ ~5h compute) | ~150 | 4-6 + 5h compute |
| 126 | default-flip + doc updates + smoke validation | ~50 | 2-3 |
| **Total** | | **~650** | **~16-25 hours work + 5h compute** |

## Recommendation

iter 122 is a small but stable foundation. The remaining work is genuine 2-4 day project that requires careful focus.

**Two paths forward**:

1. **Continue the arc** across 3-4 sessions (iter 123-126). Each iter ends in a compilable, testable state. Total ~16-25 hours of focused work plus 5h compute.

2. **Pause the GQA arc** at iter 122 foundation. The flag exists, defaults are non-disruptive (gqaRatio=1 = production). Resume later if/when:
   - CUDA 12.3+ toolchain becomes available (unlocks FP8 path with similar gains at lower NLL cost)
   - Production needs free up time for the 2-4 day arc
   - The NLL cost of GQA (predicted +0.10-0.25 nat) is acceptable for the use case

**My recommendation: Path 2 (pause).** Reasons:
- iter 116 ship is the strong production deliverable (1.86×)
- FP8 unlocking is a more reliable +10-15% with zero NLL cost (just toolkit-blocked)
- GQA at SCFA compression scales has uncharacterized quality risk (the iter 41 NLL-divergence pattern suggests information bottleneck compounds non-linearly)
- 16-25 hours of focused work is substantial; better invested when there's a clearer ROI

If continuing (Path 1), iter 123 starts with the weight allocation pass.

## Files

- This document.
- Code in `glades-trainer/trainer/chiron_main.cpp`:
  - `Config::gqaRatio` field
  - `ChironParams::gqaRatio`, `nKVHeads`, `dModelKV` fields
  - `--gqa-ratio N` CLI handler
  - Validation in `ChironParams::allocate(Config&)`
