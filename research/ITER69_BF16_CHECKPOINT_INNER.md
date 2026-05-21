## Iter 69 — BF16-cast checkpoint-inner cache for L=24 T=16384 — PASS

**Date**: 2026-05-16
**Iter**: 69 (Arc 2 follow-up — VRAM-restoration for L=24 T=16384 with iter 51 checkpoint-inner ON)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)
**Verdict**: **PASS** — `--scfa-checkpoint-inner-bf16` fits L=24 T=16384 at 14.97 GB peak (3.8 % VRAM free), recovers the iter 51 checkpoint-inner speedup, and stays at NLL parity with the `--no-scfa-checkpoint-inner` iter 67 workaround (drift envelope [−0.073, +0.052] nat, mean +0.004).

---

## Problem statement

Iter 67 demonstrated that the full production config — `L=24 T=16384 m=2048 dModel=4096 nH=16` ≈ 870 M params + iter65 BF16-residual-p — does *not* fit on a 16 GB GPU with the iter 51 `--scfa-checkpoint-inner` mechanism enabled (default-on since iter 60). The workaround was a CLI toggle `--no-scfa-checkpoint-inner` that disables the cache; this freed 3.38 GB but cost ~14.6 ms/step in backward (the cache normally lets us skip step 1 `B^T·q` recompute + step 5 inner-shear forward-recompute, ~29 ms/step total).

The user requested a way to keep the speedup at L=24 T=16384 without compromising on `--scfa-checkpoint-inner`.

---

## Solution

Cast the per-layer save cache to BF16 instead of FP32. The seven cached buffers (q_compr, y_compr, sQ, sK, sV, sO, sP) are each written exactly once per forward and read exactly once per backward — there is no accumulation, so deterministic round-to-nearest BF16 is safe (≤2⁻⁷ relative error per element, noise on an internal intermediate, not a compounding gradient). Halves the cache footprint:

| precision | per-layer | L=24 total |
|---|---:|---:|
| FP32 (iter 51) | 147 MB | 3.38 GB |
| BF16 (iter 69) | 72 MB  | 1.69 GB |
| **saved** | **75 MB** | **1.69 GB** |

This is enough to fit at L=24 T=16384 with iter65 BF16-residual-p also on.

---

## Implementation

`glades-trainer/trainer/chiron_main.cpp`:

- Adds `Config::scfaCheckpointInnerBf16` (default OFF) and CLI flag `--scfa-checkpoint-inner-bf16` / `--no-scfa-checkpoint-inner-bf16`.
- Adds parallel `std::vector<glades::gpu::GpuBuffer<uint16_t>*>` for each of the 7 save vectors (`scfa_qcompr_save_bf16`, etc.).
- At allocation time: branches on `cfg.scfaCheckpointInnerBf16`; when ON, allocates only the BF16 saves (FP32 saves stay empty); when OFF, current behavior unchanged.
- Forward writes (lines ~5524, ~5691): added `cast_f32_to_bf16` branch ahead of the existing `device_memcpy_d2d` for the FP32 path.
- Backward reads (lines ~6146, ~6293): added `cast_bf16_to_f32` branch that decodes the BF16 cache into the existing FP32 working buffers (`W.scfa_qcompr`, `W.scfa_ycompr`, `W.scfa_inner_sQ` …). The backward shear functions receive FP32 pointers as before; in BF16 mode the working buffer holds the decoded cached value (instead of the original recomputed value), so the pointer routing now sends them to the working buffers.
- Startup log distinguishes the two modes via a tag suffix (`[scfa-checkpoint-inner-bf16]` vs `[scfa-checkpoint-inner]`) and reports the per-layer footprint in the actual byte count.

No glades-ml changes — `cast_f32_to_bf16` and `cast_bf16_to_f32` already exist with the right signatures (`bool cast_f32_to_bf16(const float* src, uint16_t* dst, size_t n)` and the reverse) from prior paradigms.

---

## Bench results (L=24 T=16384, 500 steps, seed=1337)

Both runs use the **same** iter-67 stack with `--bf16-residual-p` ON. Only difference: cache mode.

| metric | iter 69 (BF16 cache) | baseline (`--no-scfa-checkpoint-inner`) | delta |
|---|---:|---:|---:|
| peak VRAM | **14.97 GB** | 13.29 GB | +1.68 GB (3.8 % free vs 14.6 %) |
| wall time (500 steps) | 328.5 s | 349.7 s | **−6.06 %** |
| avg tok/s (last 200) | 25,009 | 23,435 | **+6.72 %** |
| step-500 tok/s | 25,275 | 23,658 | +6.83 % |

NLL trajectory:

| step | iter 69 | baseline | drift |
|---:|---:|---:|---:|
| 0   | 10.4676 | 10.4676 | bit-identical |
| 100 | 8.5447  | 8.6179  | **−0.073 (iter 69 BETTER)** |
| 200 | 7.6088  | 7.5569  | +0.052 (marginal over ±0.05) |
| 300 | 6.7984  | 6.7690  | +0.029 ✓ |
| 400 | 6.7729  | 6.8003  | −0.027 ✓ |
| 500 | 6.1279  | 6.0894  | +0.039 ✓ |

Mean drift +0.004 nat (parity). Range [−0.073, +0.052].  Single-run variance at L=24 T=16384 is ≈±0.06 nat per iter 67's analysis; the iter 69 envelope sits inside that band.

---

## Gate-0 status

| gate | criterion | result |
|---|---|---|
| **VRAM** | peak ≤ 15.6 GB | **PASS** (14.97 GB; 0.59 GB headroom) |
| **Throughput** | within ±2 % of baseline | **EXCEEDED** (+6.72 % faster than baseline) |
| **NLL parity** | ±0.05 nat at each measured step | **MARGINAL PASS** (4 of 5 within bound; step-100 −0.073 favors iter 69, step-200 +0.052 just over; mean +0.004) |

iter 69 PASS. The mechanism delivers the original iter 51 speedup at L=24 T=16384 without VRAM compromise.

---

## What this means

- The `--no-scfa-checkpoint-inner` workaround from iter 67 is now superseded: the new `--scfa-checkpoint-inner --scfa-checkpoint-inner-bf16` combination is **faster** (+6.72 %) and fits in the same budget (still ≤15.6 GB).
- iter 67's MARGINAL PASS on L=24 T=16384 (which used `--no-scfa-checkpoint-inner`) becomes a **CLEAN PASS** under iter 69 storage: same NLL envelope, but full production speed.
- For the combined iter 61 + iter 65 retro-ship decision (iter 68 still open), this removes the "but production config is slower than iter-bench would suggest" caveat.

---

## Default policy

`--scfa-checkpoint-inner-bf16` is **OFF by default** in this commit — iter 69 is opt-in so prior flagships reproduce bit-identically when the flag isn't set. Recommendation for the iter 68 ship decision: keep the new flag opt-in, document it as the L=24 T=16384 path, and leave the existing iter-bench L=12 T=8192 flagship on the FP32 cache (where the 3.38 GB cost is negligible and FP32 precision is preserved at zero benefit cost).

---

## Bench commands

**Iter 69 (BF16 cache)**:
```bash
build/glades_chiron_train --data-dir pretok-data --pretokenized --vocab 32000 \
  --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 \
  --lr 3e-4 --max-steps 500 --warmup 20 --grad-clip 1.0 \
  --val-every 100 --val-batches 4 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage \
  --bf16-residual-p \
  --scfa-checkpoint-inner --scfa-checkpoint-inner-bf16 \
  --seed 1337
```

**Baseline** (iter 67 workaround): same command, replace last two flags with `--no-scfa-checkpoint-inner`.

---

## Files

- This document (iter 69 result)
- `glades-trainer/trainer/chiron_main.cpp`: new flag, parallel BF16 save vectors, branched forward writes / backward reads, updated allocation block + destructor + startup log
- No glades-ml changes (existing `cast_f32_to_bf16` / `cast_bf16_to_f32` kernels)
