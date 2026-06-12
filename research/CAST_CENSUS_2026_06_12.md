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

---

## Port A result: PASS (2026-06-12)

Implemented as `chiron_reln_forward_dual` (glades-ml c7f1d0d5e; mirror
bit-equality unit test `chiron-castelim`) + `--cast-elim-reln-q` (trainer
59383c4). Gate (300-step A/B/C at the flagship recipe, seed 1337,
`glades-trainer/logs/cast_elim_portA_gate_20260612/`):

- **Parity PASS**: drift(B vs A) = 14/30 last-digit step lines vs
  rerun-noise control 12/30; final val 8.8292 matches the rerun control
  exactly.
- **Wall +1.3%** (24,872 vs 24,523/24,568 tok/s at 300-step scale) —
  matches the ~1.2% per-port prediction.

**Bug found and fixed by the gate** (recorded as a lifecycle lesson for
Ports B–D): the first gate run FAILED (29/30 lines, +0.077 loss delta at
step 2) because freshness was only cleared inside `scfa_attention_forward`
— but the CHIRON backward never routes through it (scfa_attention_backward
reconstructs internally), so the last layer's mirror survived into the
next step and step N+1's layer-0 GEMM consumed step N's q_L bits against
the freshly embedded s.q. Step 1 being bit-identical while step 2 diverged
localized it in two bisect runs. Fix: invalidate at forward entry, where
s.q is rewritten from embeddings. **Any future mirror port must enumerate
ALL writers of the mirrored tensor across fwd/bwd/val, not just the
producing kernel's function.**

Port A stays default-off pending the stacked multi-port +3% ship decision
(Ports B/C remain to be implemented per §2).

---

## Ports B–D continuation spec (2026-06-12, analysis complete, surgery deferred)

### Port D: SKIPPED (decided)
E_bf/readout bucket is 1.6 ms/step ≈ 0.27% — below the iter-104 sub-noise
threshold, and the producer is the shared Adam kernel (optimizer surgery
for sub-noise value). Closed.

### Port C analysis (priority next — biggest bucket, de-risked)
**De-risking discovery:** the BF16-C/D GEMM mechanism already SHIPPED
parity-clean as iter 61's `sgemm_rowmajor_atb_bf16_dst_bf16` (D=BF16,
FP32 internal accumulator, final rounding folded into the GEMM) — used in
the production bf16-grads path. Port C generalizes this to beta=0
overwrite sites.

Target sites (k×dModel bucket, 17.2 ms/step, all in `gpu_chiron.cu`):
1. **bwd sdQ/sdK/sdV** (casts at lines ~1585/1593/1601 in the bf16w path
   and their bf16g-variant twins): `flash_attention_backward_cublas_tiled`
   writes them FP32 (beta=0 since iter 107); sole consumers are the
   per-X casts into the shared `scratch_sdbf`. Change: final dQ/dK/dV
   GEMMs write BF16-D into three NEW caller-provided buffers
   (3 × T×dModel × 2B ≈ 25 MB at k=1024), delete the 3 casts, point the
   6 downstream GEMMs at them. Signature change ripples: lib header +
   trainer snapshot mirror + call sites (the Port A forward-declaration
   convention applies).
2. **fwd sQ/sK/sV inner-attention input casts** (iter-118 doc's pipeline
   step 1): AUDIT REQUIRED — FP32 sQ/sK/sV are also consumed by the
   scfa-checkpoint-inner save path and `flash_attention_backward` inputs;
   BF16-C here changes what the checkpoint stores. Only convert if the
   checkpoint already stores BF16 (scfa-checkpoint-inner-bf16 suggests
   yes — verify) and the bwd consumer accepts the mirror.

Gate: same A/B/C rerun-noise methodology. KNOWN RISK: C-type changes can
shift cuBLAS algorithm selection (iter-78 measured algo changes at
+0.036 nat — outside noise). If the gate fails parity, document FAIL and
keep the flag off (iter-102-style outcome); the iter-61 precedent says
beta=1 dst-bf16 held parity, so beta=0 likely holds too.

### Port B analysis (second)
Production skips the bwd `B^T·q` recompute (scfa-checkpoint-inner), so
Port B's real targets are:
1. **dy operand** of `B^T·dy` (line ~9387): under iter 116 (default ON)
   this operand IS `s.dp` directly. Producer: the fused bwd stream ops
   (`p±=sign·(ypar+yperp)` family). CAUTION: the existing p_bf16 mirror
   from those kernels is SR-cast (stochastic rounding) — a GEMM mirror
   must be a SEPARATE RNE side-write, not a reuse of p_bf16.
2. **dq_perp operand** of `B^T·dq_perp` (line ~9955): producer is
   `scfa_dwconv_dx_tiled_kernel_dual_out` (iter 99) — add an RNE BF16
   third output.
Both follow the Port A pattern (side-write + scoped registration +
one-shot freshness). **Lifecycle rule from Port A's bug: enumerate ALL
writers of the mirrored tensor across fwd/bwd/val before wiring the
freshness flag — the producing kernel's neighborhood is not enough.**
Estimated combined: +2–2.5% wall.

### Stacked ship decision
After B+C land: n=3 multi-seed 100-step bench of A+B+C stacked; ship at
+3% (iter 60 bar). Port A alone (+1.3%) stays flag-gated silent-accrual.
