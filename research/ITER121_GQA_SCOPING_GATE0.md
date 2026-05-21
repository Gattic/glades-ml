## Iter 121 — GQA r=4 scoping + Gate-0 design

**Date**: 2026-05-21
**Iter**: 121 (first iter of GQA-r=4 multi-iter arc)
**Branch**: vesta5
**Verdict**: **Design doc + multi-iter plan.** Maps touch points, proposes 5-iter phased implementation. No code changes yet (next iter starts Phase 1).

---

## Why a scoping iter first

Initial audit found **305 references** to `Wq*/Wk*/Wv*/Wo*` (and grad/Adam variants) in `chiron_main.cpp`, plus 46 in `chiron_attention_shear_bf16w_bf16g_tiled` alone. Plus per-checkpoint format changes (saved Wk/Wv have different dims) and Adam state allocation per-tensor.

This is genuinely a multi-iter (1-2 week) arc. A scoping iter avoids the common pitfall of starting the refactor before understanding all touch points.

## What GQA r=4 actually changes

**Current production stack** (iter 116 ship):
- `nHeads = 16`, `dHead = 256`, `dModel = 16 × 256 = 4096`
- Wq, Wk, Wv, Wo all allocated as `[m, dModel]` = [2048, 4096] BF16

**With GQA r=4**:
- `nHeads = 16` (unchanged — Q heads)
- `nKVHeads = 4` (K/V heads)
- `dModelKV = 4 × 256 = 1024`
- **Wq, Wo unchanged**: still `[m, dModel]` = [2048, 4096]
- **Wk, Wv shrink 4×**: `[m, dModelKV]` = [2048, 1024]

Inner attention math:
- `sQ = q_compr · Wq` → [k, dModel] (unchanged)
- `sK = q_compr · Wk` → [k, **dModelKV**] (4× smaller)
- `sV = q_compr · Wv` → [k, **dModelKV**] (4× smaller)
- QK^T: 16 Q heads share K across groups of 4. Each Q head h uses K head `h/4`.
- PV: 16 Q heads use V head `h/4`.

## Touch points by category

### A. Weight allocation (chiron_main.cpp)
1. `Wk_bf[l]->allocate((size_t)m * dModel)` → `m * dModelKV` ← line 2443
2. `Wv_bf[l]->allocate((size_t)m * dModel)` → `m * dModelKV` ← line 2444
3. `Wk[l]->allocate(...)` FP32 fallback ← line 2459
4. `Wv[l]->allocate(...)` ← line 2460
5. `dWk_bf[l]->allocate(...)` gradient ← line 2505 (and similar for FP32)
6. `dWv_bf[l]->allocate(...)` ← similar

### B. Adam state (chiron_main.cpp)
For Wk: m, v at full + int8 + scales. Plus Kahan v if --kahan-v. 6 sites per tensor × Wk, Wv. ~12 sites.

For BF16 Adam mode (production):
- Wk_mb, Wk_vb (BF16 m, v)
- Wk_mi, Wk_vi (int8 quant)
- Wk_ms, Wk_vs (FP32 scales, per-256 elements)

All must use the new dimensions when GQA r > 1.

### C. Checkpoint format (chiron_main.cpp save/load)
- CHRN/CHRF writer: writes `m * dModel` Wq, Wk, Wv, Wo blocks. Need to write `m * dModelKV` for Wk, Wv.
- Loader: must read the correct sizes. Add a `nKVHeads` field to the checkpoint header for forward compatibility.

CHRN v=3 / CHRF v=4: bump versions to support nKVHeads encoding. Or add a flag bit. Layout:
```
magic(4) version(4) flags(4)  ← bit 256: nKVHeads-encoded
hdr[6]: T, m, L, nH, dH, V    ← unchanged
[if flags & 256] nKVHeads(4)  ← new field
... weights ...
```

Backward compat: when bit 256 is unset, default nKVHeads=nHeads (legacy non-GQA).

### D. Forward path (gpu_chiron.cu)

**chiron_attention_shear_bf16w_bf16g_tiled** (production fwd path):
- Wk, Wv cuBLAS projections (lines ~1337-1339): use smaller Wk_bf, Wv_bf
- `flash_attention_cublas_tiled_bf16` inner attention call: needs GQA awareness

**flash_attention_cublas_tiled_bf16**:
- Inputs Q, K, V have different shapes when GQA (K, V are dModelKV-wide)
- QK^T cuBLAS: currently single batched gemm with batchCount=nHeads. With GQA, need either:
  - **Option A**: broadcast K to nHeads (defeats purpose of saving K compute)
  - **Option B**: 4 grouped batched cuBLAS calls (batchCount=4 each, strideB=0 within group)
  - **Option C**: custom GQA-aware kernel
- PV cuBLAS: same options
- Softmax (iter 102 fused): operates on scratch_S [nHeads, k, k] — unchanged

Choosing **Option B** (4 grouped calls): minimal kernel work, leverages existing cuBLAS, ~15 µs extra launch overhead negligible.

### E. Backward path (gpu_chiron.cu)

**chiron_attention_shear_bf16w_bf16g_tiled** (also handles bwd):
- dWk_bf, dWv_bf: smaller output (cuBLAS GEMM dims change)
- dq projection backward: 3 cuBLAS calls (Q, K, V each contribute to dq) — K, V are GQA-shaped

**flash_attention_backward_cublas_tiled**:
- Computes sdQ, sdK, sdV from sQ, sK, sV, dO
- With GQA: sdK, sdV are nKVHeads-wide. sdQ stays nHeads-wide.
- Gradient flow: each Q head's dS sums into its kvHead's dK (and dV). Cross-head sum within K group.
- cuBLAS GEMMs need re-orchestration: dK = sum over heads-in-group of (dS^T · Q). Or: route via grouped batched gemm with reduction.

This is the **trickiest part of the implementation**. dK has reduction across Q heads in a group.

### F. Inference (chiron_infer.cpp)
- Checkpoint loader reads nKVHeads from header
- Inference attention kernel needs GQA awareness (flash_attention_multihead_forward already takes nKVHeads parameter; just wire it through)

### G. CLI / config
- `--gqa-ratio N` flag (default 1 = no GQA). N must divide nHeads.
- `cfg.nKVHeads = cfg.nH / cfg.gqaRatio`
- Default OFF (gqaRatio=1) for backwards compatibility

## Proposed 5-iter arc

### Iter 122 — Phase 1: weight allocation + fwd projections only
- Add `cfg.gqaRatio` field + CLI
- Allocate Wk_bf, Wv_bf at reduced size when ratio > 1
- Allocate Adam state (mb/vb/mi/vi/ms/vs) at reduced size
- Modify fwd Wk, Wv cuBLAS projection calls
- Output sK, sV are dModelKV-wide
- **Don't change inner attention yet** — assume sK, sV are broadcast/expanded back to dModel-wide before flash_attention_cublas_tiled_bf16
- Gate-0: ratio=1 must be bit-identical to current production
- Gate-1: ratio=4 fwd runs (math wrong but no crash)
- 4-5 hours implementation

### Iter 123 — Phase 2: grouped cuBLAS dispatch for inner attention
- Modify flash_attention_cublas_tiled_bf16 to handle nKVHeads parameter
- 4 grouped batched cuBLAS calls (Option B) for QK^T and PV
- At ratio=1: bit-identical (4 calls of batchCount=1 instead of 1 call of batchCount=16 — same math but more launch overhead)
- At ratio=4: 4 calls of batchCount=4, smaller K/V matrices, real wall improvement on the projections (already done in Phase 1) PLUS inner attention K/V memory savings
- Gate: wall improves at ratio=4
- 1 day implementation

### Iter 124 — Phase 3: backward path GQA
- Modify chiron_attention_shear_bf16w_bf16g_tiled bwd to handle GQA-shaped dK, dV
- Modify flash_attention_backward_cublas_tiled for grouped cuBLAS in bwd
- dK, dV gradient: cross-head reduction within K group
- Gate: ratio=1 produces bit-identical gradients
- Gate-1: ratio=4 produces reasonable-magnitude gradients (no NaN, no explosion)
- 1-2 days implementation

### Iter 125 — Phase 4: checkpoint format + retrain
- Bump CHRN/CHRF versions to encode nKVHeads
- Update saver and loader
- Update chiron_infer
- Kick off 30k Phase 2 retrain at ratio=4 (~4-5h compute)
- Gate: NLL drift acceptable (target ≤ +0.25 nat over iter 116 ship)
- 1 day implementation + 5h compute

### Iter 126 — Phase 5: ship (if PASS)
- Default-flip to gqaRatio=4
- Update CLAUDE.md, FLAGSHIP doc
- Promote new checkpoint
- ~2 hours

**Total estimated effort: 5-7 days of focused work + ~5h compute.**

## Predicted outcomes

Per earlier analysis:
- Wall: +5-10% over iter 116 ship → 28,257 → ~30,500-31,000 tok/s (~2.0× cumulative)
- NLL: +0.10-0.25 nat over iter 116 ship's 4.2039 → 4.30-4.45 (within strict ±0.5 retrain bound but worse than current)
- VRAM: −200 MB

## Risks

1. **NLL hit larger than expected**: SCFA's compression to k=1024 + GQA on top might compound the information bottleneck. If NLL increase > 0.25 nat, may not be acceptable.
2. **Backward pass complexity**: dK/dV cross-head reduction is the trickiest part. Easy to get wrong.
3. **Checkpoint compatibility break**: existing iter 116 ship checkpoint not loadable at gqaRatio≠1.
4. **Saving format proliferation**: trainer needs to handle both legacy and GQA-encoded checkpoints. Adds complexity.

## Decision points

After each phase iter, validate before continuing:
- **After iter 122**: ratio=1 wall regression check (more cuBLAS calls = slower; bound to ≤ 1% regression)
- **After iter 123**: ratio=4 wall improvement check (must show ≥ 3% improvement to justify continuing)
- **After iter 124**: ratio=4 NLL trajectory check at 100 steps (within ±0.5 nat of baseline projection)
- **After iter 125**: 30k retrain NLL @ step 30k (target ≤ +0.25 nat over iter 116 ship)

If any phase fails its gate, abort and revert.

## Iter 122 (next iter) specific plan

Start small: just weight allocation + fwd projection.

1. Add `gqaRatio` field to Config struct (default 1)
2. Add `--gqa-ratio N` CLI flag
3. Compute `cfg.nKVHeads = cfg.nH / cfg.gqaRatio` after parse
4. Validate: `nH % gqaRatio == 0`, else error
5. Modify weight allocation loop in `init_chiron_params` to use `m * dModelKV` for Wk_bf, Wv_bf (and dWk_bf, dWv_bf, and Adam state)
6. Modify the 2-3 Wk/Wv-related cuBLAS calls in chiron_attention_shear_bf16w_bf16g_tiled fwd to use compressed Wk_bf
7. After cuBLAS produces compressed sK, sV: broadcast/expand to full nHeads-wide before flash_attention_cublas_tiled_bf16 (kernel still operates on nHeads-wide K, V)
8. Bench: ratio=1 (must be bit-identical), ratio=4 (math wrong but should run without crash)

This is iter 122. Iter 121 (this doc) ends here.

## Files

- This document.
- No code changes in iter 121.
