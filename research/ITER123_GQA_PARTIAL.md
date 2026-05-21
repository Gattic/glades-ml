## Iter 123 — GQA r=4 forward path working, backward broken

**Date**: 2026-05-21
**Iter**: 123 (Phase 2/3 of GQA arc)
**Branch**: vesta5
**Verdict**: **Forward path working at GQA r=4. Backward produces NaN gradients — needs debugging.** Library + trainer infrastructure substantially complete. Math validated at gqaRatio=1 (bit-identical). At gqaRatio=4: step 1 fwd loss=10.5815 (reasonable magnitude), but bwd produces NaN ‖g‖ that corrupts step 2.

---

## What iter 123 delivers

### Library (glades-ml)

**Broadcast + reduce kernels** in `gpu_kernels.cu`:
- `chiron_gqa_broadcast_kv(src, dst, T, nKVHeads, nHeads, dH)` — replicates K/V across query heads in each KV group (fwd)
- `chiron_gqa_reduce_dkv(src, dst, T, nKVHeads, nHeads, dH)` — sums gradient contributions across query heads in each group (bwd)

**chiron_attention_shear_bf16w_tiled** (fwd-only function): GQA path added.
- New optional params: `nKVHeads`, `scratch_sK_c`, `scratch_sV_c`
- When `nKVHeads < nHeads`: Wk/Wv cuBLAS produces compressed K/V [T, dModelKV], then broadcast to full [T, dModel] before inner attention.

**chiron_attention_shear_backward_bf16w_bf16g_tiled** (fused fwd+bwd function): GQA path added.
- New optional params: `nKVHeads`, `scratch_sK_c`, `scratch_sV_c`, `scratch_sdK_c`, `scratch_sdV_c`
- Fwd: compressed K/V cuBLAS + broadcast (same as fwd-only)
- Bwd: full-size flash_attention_backward output reduced to compressed sdK/sdV; dq accumulation + dW gradient use compressed sdK/sdV + compressed Wk_bf/dWk_bf

### Trainer (glades-trainer)

**Weight allocation** at compressed size when `gqaRatio > 1`:
- Wk_bf, Wv_bf: `m * dModelKV`
- dWk_bf, dWv_bf: `m * dModelKV`
- Wk_tmp, Wv_tmp: `m * dModelKV`
- dWk_scratch, dWv_scratch: `m * dModelKV`
- FP32 fallback Wk, Wv, dWk, dWv: `m * dModelKV`
- Adam state (via `addAdam`): `m * dModelKV`

**New compressed K/V scratches** in ChironParams (allocated only when gqaRatio > 1):
- `scfa_inner_sK_c`, `scfa_inner_sV_c`
- `scfa_inner_sdK_c`, `scfa_inner_sdV_c`

Each `[scfa_k, dModelKV]` = 4 MB at production (gqaRatio=4).

**Init vectors**: separate `Wkv` host vector at `m * dModelKV` for Wk/Wv upload.

**Save path**: `Wk_host.back()->master` download at `m * dModelKV` size.

**Call sites updated** to pass `nKVHeads` + compressed scratches.

## What's broken

At gqaRatio=4:
- Step 1 fwd: ✅ loss=10.5815 (reasonable magnitude — model output is sensible)
- Step 1 bwd: ❌ ‖g‖=nan (gradient NaN somewhere)
- Step 2 fwd: ❌ illegal memory access (corrupted weights from NaN grad in step 1 Adam update)

The NaN gradient suggests a bug in:
- `chiron_gqa_reduce_dkv` kernel (cross-head sum producing NaN?)
- Or cuBLAS gradient call with compressed dims
- Or compressed Wk_bf/dWk_bf shape coordination at some unexpected call site

## Bench results

**gqaRatio=1 (sanity, default)**:
- NLL @ step 25: **10.4414** (matches iter 122 baseline = bit-identical)
- Wall: 14.7s
- ✅ Backward compatibility preserved

**gqaRatio=4 (target)**:
- Step 1 fwd loss: 10.5815 (vs gqa=1 baseline 10.5777 — within expected variance for different K/V capacity)
- Step 1 bwd: NaN ‖g‖
- Run aborts at step 2 with illegal memory access

## Remaining work (iter 124)

**Debug the bwd NaN**:
1. Verify reduce kernel correctness with a unit test
2. Check that flash_attention_backward_cublas_tiled returns sane sdK, sdV at full size when given broadcast K/V
3. Verify cuBLAS gradient GEMMs with compressed dims don't have a layout bug

Likely culprit: the iter 69 BF16 cache path may need GQA awareness. When the cache decodes back into scfa_inner_sK, it expects the broadcast layout — but during the bwd recompute, the function re-broadcasts. Could be a subtle interaction.

**Phase 4 (after bwd debug)**:
- Checkpoint format encoding (nKVHeads field)
- 30k Phase 2 retrain at gqaRatio=4
- Default-flip if PASS

## Honest assessment

The infrastructure is ~75% complete. The remaining work is:
1. Debug NaN gradient in bwd (~2-4 hours focused)
2. Checkpoint format (~1-2 hours)
3. 30k retrain (~5h compute)
4. Default-flip + docs (~1 hour)

Once the bwd bug is fixed, the path to a tested + benched GQA r=4 implementation is clear.

## Files

- This document.
- Code:
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`: broadcast + reduce kernels
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu`: GQA path in chiron_attention_shear_bf16w_tiled (fwd) + chiron_attention_shear_backward_bf16w_bf16g_tiled (fused fwd+bwd)
  - `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.h` + `gpu_chiron.h`: decls + no-CUDA stubs
  - `glades-trainer/include/Backend/Machine Learning/Networks/cuda/gpu_kernels.h` + `gpu_chiron.h`: vendored mirrors
  - `glades-trainer/trainer/chiron_main.cpp`: cfg.gqaRatio CLI, ChironParams nKVHeads/dModelKV, allocation changes (Wk/Wv/dWk/dWv/Adam-state/scratches), call site wiring
