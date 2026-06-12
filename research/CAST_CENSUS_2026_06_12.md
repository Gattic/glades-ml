# Cast-elimination arc — census + port plan (2026-06-12)

**Arc:** cast-pipeline elimination (redirect from FA-inner Gate-1 META,
`research/FA_INNER_GATE1_META_2026_06_12.md`).
**Status:** census COMPLETE; ports specified, none implemented yet.
**Profile:** `glades-trainer/logs/nsys_flagship_sira_clamp_20260612.{nsys-rep,sqlite}`
(30 steps, SIRA+clamp flagship recipe, seed 1337; step ≈ 602 ms).

## 1. Census

`k_cast_f32_to_bf16` total: 8.2% of GPU kernel time, ~1,185 launches/step,
~69 ms/step (≈11.5% of step wall incl. gaps). By grid size (block = 256,
1 elt/thread → n = gridX·256):

| gridX | n (elements) | tensor | per step | ms/step | attribution |
|---:|---:|---|---:|---:|---|
| 131,072 | 33.5M | **T×m** | ~110 | **34.8** | activations (q, dy, dq_perp) as the non-B operand of outer `B^T·X` GEMMs, fwd+bwd, ~4.6/layer |
| 16,384 | 4.2M | **k×dModel** | ~573 | **17.2** | inner-attention sQ/sK/sV casts in `flash_attention_cublas_tiled_bf16` (~24/layer; the operands are themselves GEMM outputs) |
| 65,536 | 16.7M | T×k | ~59 | 9.7 | compressed-length activations (q_compr-family at full T? verify at port time) |
| 8,192 | 2.1M | k×m | ~441 | 6.1 | q_compr/y_compr/dq_compr operands of `B·X` GEMMs (GEMM outputs re-cast) |
| 256,000 | 65.5M | V×m | ~2.5 | 1.6 | E_bf cache refresh + readout path |

Mechanism (gpu_blas.cu `sgemm_rowmajor_fast16bf_impl`): every FAST_16BF
GEMM pre-casts BOTH operands into shared scratch unless
`lookupFast16bfConstant` hits (currently only scfa_B is registered).
The casts are of distinct tensors per call — no duplicate-cast dedup win;
the wins are producer-side.

## 2. Ports (priority order, one flag each, gate-staged)

**Mechanism class:** producer dual-output side-writes (iter 97/99/101 PASS
pattern; `chiron_scfa_axpy2_dual_p` already dual-writes p_bf16 in
production) + scoped extension of `register_fast16bf_constant` so the GEMM
wrapper consumes the mirror (register q→q_bf16 for the layer, unregister
after; invalidation stays with the code that owns the dataflow).

1. **Port A — reln fwd q mirror** (~8–15 ms/step, ≈1.3–2.5% wall):
   `chiron_reln_forward_rows` side-writes q_bf16; trainer registers the
   mirror around the layer's outer GEMMs. New kernel variant ~30 LOC.
2. **Port B — bwd T×m mirrors** (remainder of the 34.8 ms bucket):
   side-writes in `layernorm_backward_dx` / fused bwd stream ops for
   dy/dq_perp operands.
3. **Port C — GEMM BF16-C output** (17.2 + 6.1 ms buckets): the
   sQ/sK/sV and X_compr operands are GEMM *outputs*; cublasGemmEx can
   write C as CUDA_R_16BF directly (FP32 compute). CAUTION: a different
   C-type can change cuBLAS algorithm selection → different accumulation
   order → trajectory drift of the iter-74 class. Must gate against the
   rerun-noise control, and check the s16816 fast path is retained (the
   CUDA-13 regression mechanism — verify with nsys per-kernel names).
4. **Port D — E_bf/readout** (1.6 ms): low value, take only if trivial.

**Ceiling if A–C land:** ~58 ms/step ≈ **+9% wall** at the current 602 ms
step; realistic with partial landings +4–7%. Ship bar +3% (iter 60
precedent) at n=3 multi-seed, NLL parity per the 2026-06-11 rerun-control
methodology (same-seed runs are not bit-reproducible; compare drift vs a
no-change rerun).

## 3. Gates per port

1. Unit/bench smoke: mirror bit-equality vs `cast_f32_to_bf16` of the
   FP32 output (the side-write must produce identical RNE bits).
2. 300-step trainer A/B/C (off / on / off-rerun) — trajectory drift(on)
   within drift(rerun); zero behavior change when flag off.
3. Stacked n=3 multi-seed 100-step bench for wall; ship decision at +3%.

## 4. State

No code changes yet. Production flagship unchanged. Next session: Port A.
