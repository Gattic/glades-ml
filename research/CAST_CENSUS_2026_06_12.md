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

---

## Port C result: ANALYSIS-NULL (2026-06-12, no code)

Legality audit kills the specced slice and shrinks the rest:

1. **cuBLAS type-combo wall**: gemmEx supports BF16-C/D only with BF16
   A/B inputs; there is NO FP32-in → BF16-out combo. The bwd sdQ/sdK/sdV
   producers take FP32 inputs — and dQ/dK are **deliberately strict-FP32**
   since the 2026-06-10 BF16G replay-parity commit (d9b8e3249). BF16-D
   there would require casting the attention-backward inputs to BF16 — a
   precision regression of the exact path just hardened. Slice 1 is
   ILLEGAL as specced.
2. **Legal subset** = the fwd sQ/sK/sV projection GEMMs
   (`sgemm_rowmajor_bf16`, BF16 in / FP32 out, gpu_chiron.cu
   `chiron_attention_shear_bf16w_tiled` ~lines 1380-1385). Checkpoint
   audit CONFIRMS viability: production saves sQ/sK/sV to a BF16
   checkpoint (trainer ~8566) and the backward restores by decoding it
   (~9256), so the bwd already consumes BF16-rounded values; BF16-D
   projections + cast-skips in `flash_attention_cublas_tiled_bf16` +
   checkpoint saves switched to d2d copies would be value-preserving
   modulo GEMM algorithm selection.
3. **But the value shrank**: 3 inner casts × 48 shear calls/step
   (~4.3 ms) + checkpoint-save casts→memcpys (~0.8 ms) ≈ **~0.85% wall**
   — most of the census's 17.2 ms k×dModel bucket is required
   precision-boundary conversion next to strict-FP32 paths, not
   eliminable.

**Disposition**: Port C closed as analysis-NULL at current priorities
(~0.85% for multi-file GEMM-wrapper + shear + inner-attention + trainer
surgery, with algo-selection parity risk). Revisit only after Port B; the
fwd-projection slice remains specified above if the stacked total needs
the last fraction of a percent.

---

## Port B (dq_perp slice) result: PASS (2026-06-12)

Implemented as `scfa_depthwise_causal_conv_bwd_dual_out_bf16mirror`
(glades-ml b18dd8ae5; duplicated kernel so legacy codegen is untouched;
`chiron-castelim` suite extended — dx_primary/dx_secondary/dK bit-identical
to dual_out, mirror bit-identical to the cast) + `--cast-elim-dqperp`
(trainer f4c9895). Producer and consumer are adjacent in
`scfa_attention_backward` with only read-only traces between, so the
mirror uses a tight register/GEMM/unregister scope — NO freshness flag,
structurally immune to the Port A lifecycle bug class.

Stacked gate (A/B/C, B = `--cast-elim-reln-q --cast-elim-dqperp`,
`glades-trainer/logs/cast_elim_portB_gate_20260612/`):

- **Parity PASS**: drift 15/30 step lines vs 16/30 rerun-noise control
  (inside the noise band); final val matches the rerun control exactly.
- **Wall +1.9% stacked** (25,007 vs 24,593/24,505 tok/s). The dq_perp
  slice contributed ~+0.6% measured (vs ~+1.25% predicted — short-run
  tok/s noise is ±0.5%; the n=3 bench will tighten this).

## Arc status after Ports A+B

| port | status | measured wall |
|---|---|---:|
| A (reln q mirror) | PASS, default-off | +1.3% alone |
| B dq_perp slice | PASS, default-off | +1.9% stacked w/ A |
| B dy slice | OPEN (next) | ~+1.2% predicted |
| C (BF16-D GEMM) | analysis-NULL | (~0.85% legal subset, shelved) |
| D (E_bf) | closed sub-noise | — |

The stack sits at **+1.9% measured** vs the **+3% ship bar** — the dy
slice decides ship-vs-silent-accrual. dy = `bwd_dy_ptr` (= `s.dp` under
iter 116 default-ON) consumed by the `B^T·dy` GEMM; producers are the
fused bwd stream ops (`chiron_scfa_axpy2_dual_p` family). CAUTION
(repeated from §Port B analysis): those kernels' existing p_bf16 mirror
is SR-cast (stochastic rounding) — the GEMM mirror must be a separate RNE
side-write. Same tight-scope consume pattern applies if producer/consumer
adjacency holds (verify the dataflow between the last s.dp writer and the
B^T·dy GEMM).

Before any default-flip: n=3 multi-seed 100-step stacked bench per the
standing methodology.

---

## dy slice PASS + n=3 ship bench: BELOW BAR, silent-accrual (2026-06-12)

**dy slice** (`--cast-elim-dy`, trainer 5cd4923): s.dp is loop-invariant
across the backward layer loop in the production config (additive shear;
writers — zero, SIRA terminal grad, L-1 fuse axpy — all precede the first
`scfa_attention_backward`, which is read-only on s.dp per the iter 116
precondition). One hoisted `cast_f32_to_bf16` + a loop-spanning
registration replaces the 24 per-layer B^T·dy GEMM casts. No kernel
changes; RNE bits by construction. Gate
(`logs/cast_elim_dy_gate_20260612`, B = all three flags): drift 12/30 vs
13/30 rerun noise, final vals identical, wall +3.0% single-seed.

**n=3 multi-seed 100-step paired bench**
(`logs/cast_elim_n3_bench_20260612`, seeds 1337/1338/1339):

| seed | base tok/s | stack tok/s | Δ | final loss |
|---:|---:|---:|---:|---|
| 1337 | 27,542 | 28,238 | +2.53% | identical (9.9251) |
| 1338 | 27,514 | 28,254 | +2.69% | identical (9.9143) |
| 1339 | 27,469 | 28,285 | +2.97% | identical (9.9150) |

**Mean +2.73% ± 0.18% — BELOW the +3% iter-60 multi-seed ship bar.**
NLL parity is perfect (4-decimal-identical per seed at 100 steps).

**Verdict: the three-flag stack (`--cast-elim-reln-q --cast-elim-dqperp
--cast-elim-dy`) is parity-clean, kernel-level bit-identical, and stays
default-off silent-accrual** (iter 97–101 class). Not added to the
flagship recipe.

**Documented path to the bar**: the shelved Port C fwd-projection slice
(~+0.85%, §Port C result) would put the stack at ~+3.6%. It requires the
multi-file BF16-D surgery (GEMM wrapper variant + fwd shear + inner
attention cast-skip + trainer checkpoint-save switch) with the
algo-selection parity risk gate-decided. Next session item; all analysis
recorded above.

## Final arc state (2026-06-12)

| port | verdict | wall contribution |
|---|---|---:|
| A reln-q mirror | PASS | +1.3% alone |
| B dq_perp mirror | PASS | +1.9% w/ A |
| B dy hoisted cast | PASS | **+2.73% ± 0.18% full stack (n=3)** |
| C BF16-D | analysis-NULL, fwd slice shelved | ~+0.85% if revisited |
| D | closed sub-noise | — |

Flags all default-off; production flagship recipe unchanged.

---

## Port C fwd slice: QK-NORM CONFLICT — slice as specced is production-inert (2026-06-12)

Implementation began per spec (committed as opt-in infrastructure:
`sgemm_rowmajor_bf16_dst_bf16` NN wrapper, `set_cast_elim_inner_fwd`
toggle + BF16-D projections + `flash_attention_cublas_tiled_bf16_precast`
in `chiron_attention_shear_bf16w_tiled`). During trainer wiring, reading
the production dispatch revealed: **with `--qk-norm` (the production
flagship recipe), the SCFA inner forward takes the Task-4B decomposed
split path** (trainer `chiron_main.cpp`, "Task 4B BF16-inner extension"),
NOT `chiron_attention_shear_bf16w_tiled`. In that path:

- Q and K projections MUST stay FP32: `qknorm_forward_gpu` +
  `scale_q_per_head` consume and rewrite FP32 sQ/sK in place between
  projection and attention. Their inner-attention casts are required
  precision boundaries — NOT eliminable.
- The toggle therefore only accelerates the non-QK-Norm branch, which
  production does not take. **Slice as specced: production-inert.**

**Revised legal remainder at the production recipe** (~0.6% total,
sites now precisely known):
1. **V projection** (Step A, 3rd GEMM): BF16-D into p_Vbf16 + an inner
   variant that skips only the V cast (per-operand granularity).
2. **O output** (Step E output + Step F cast): `sgemm_batched_strided_bf16`
   has BF16 inputs → a batched-strided dst-BF16 wrapper writes O directly
   as BF16; Step F's cast becomes unnecessary (output projection reads the
   BF16); O is UNUSED in `flash_attention_backward_cublas_tiled` (explicit
   `/*O unused*/`); bwd's sO comes from the BF16 checkpoint restore.
3. Checkpoint saves for sV/sO switch from cast to d2d copy.

+2.73% (current stack) + ~0.6% ≈ +3.3% — would clear the bar, but needs
a new wrapper family + per-operand inner variant + two save switches +
gates. Deferred to a fresh session per scope discipline; the shipped
infrastructure (NN dst-BF16 wrapper, precast inner variant) is reusable
for it.

**Also noted for the census ledger**: the k×m bucket's dominant member is
the Step-A `q_compr → BF16` cast (~96/step). q_compr is a FAST_16BF GEMM
output (FP32 by type-combo law), but the impl's gemmEx call has BF16 A/B
inputs post-pre-cast — a FAST_16BF dst-BF16 variant is legal and could
chain with Port A's mirror. Same diminishing-returns caveat: q_compr FP32
is also consumed (q_par GEMM, dwconv path) — full audit required.
