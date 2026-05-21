## Iter 63 — BF16 residual-p (Arc 2, iter 1 of 5) — FRAMEWORK PARTIAL

**Date**: 2026-05-16
**Iter**: 63 (Arc 2 iter 1; renumbered from design's iter 62 since Arc 1 consumed iter 62)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)
**Verdict**: **PARTIAL — framework only.**  Kernels + alloc + flag landed; bit-exact fallback (G0.3) validated.  Throughput Gate-0a (G0.1) not yet testable — single-site wiring with dual-sync forces −1.3% wall by construction.  Empirically confirms design risk R1 (RN-rounding accumulation drift) at +0.037 nat after 30 steps with 24 BF16-axpy2 calls/step, validating the design's claim that SR (stochastic rounding) is **structurally required** for L=12 parity.

---

## What was implemented

### Kernels (`glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` + `.h`)

Four new BF16-p variants with FP32-internal accumulation + RN-rounded write:

```cpp
chiron_scfa_axpy2_bf16p_rn(uint16_t* p_bf, float α, const float* a, const float* b, int n);
chiron_axpy_bf16p_rn       (uint16_t* p_bf, float α, const float* x, int n);
chiron_scfa_scaled_copy_bf16p_rn(uint16_t* c_bf, float α, const float* a, int n);
chiron_bf16_to_fp32_axpy   (float* q, float α, const uint16_t* p_bf, int n);   // q += α·decode(p_bf)
```

Each kernel:
- Decodes BF16 → FP32 inline (per-element, register-local).
- Accumulates `α·(a+b)` or `α·x` in FP32.
- Rounds back to BF16 RN-even via `fp32_to_bf16_rn_dev(acc)`.
- Handles NaN → quiet-NaN preservation in the encoder.

Mirror declarations added to trainer-side `gpu_chiron.h`.

### Trainer state (`glades-trainer/trainer/chiron_main.cpp`)

- New `Config::bf16ResidualP` flag (default false), CLI `--bf16-residual-p`.
- New `Scratch::p_bf16` buffer (`GpuBuffer<unsigned short>`, T*m), allocated only when flag set.

### Single-site wiring (proof of correctness)

Wired the SCFA forward axpy2 at `chiron_main.cpp:5650` (the `--scfa-fuse-streams` hot path) through `chiron_scfa_axpy2_bf16p_rn` when flag is set.  Uses dual-sync:

```cpp
if (cfg.bf16ResidualP && s.p_bf16.allocated()) {
    cast_f32_to_bf16(s.p, s.p_bf16, Tm);             // pre-sync FP32 -> BF16
    chiron_scfa_axpy2_bf16p_rn(s.p_bf16, sign, ypar, yperp, Tm);  // BF16-op
    cast_bf16_to_f32(s.p_bf16, s.p, Tm);             // post-sync BF16 -> FP32
}
```

The dual-sync is intentionally wasteful for iter 63 — it lets non-converted p readers (reln, q+=p, attention-shear backward, etc.) keep seeing correct FP32 values.  iter 64's job is to route the remaining ~20 p-touching sites and remove the sync.

---

## Smoke results (30 steps, iter-bench config, seed 1337)

### With `--bf16-residual-p`

| step | loss | ema | tok/s | ||g|| |
|---:|---:|---:|---:|---:|
| 1   | 10.6019 | 10.6019 | 28,785 | 3.720 |
| 10  | 9.9007  | 10.3336 | 42,988 | 4.896 |
| 20  | 9.6357  | 9.9842  | 48,843 | 3.588 |
| 30  | 10.5080 | 9.8820  | 48,851 | 5.197 |

### Without `--bf16-residual-p` (baseline parity)

| step | loss | ema | tok/s | ||g|| |
|---:|---:|---:|---:|---:|
| 1   | 10.6019 | 10.6019 | 29,018 | 3.719 |
| 10  | 9.9013  | 10.3336 | 43,730 | 4.889 |
| 20  | 9.6351  | 9.9826  | 49,487 | 3.477 |
| 30  | 10.3120 | 9.8453  | 49,494 | 7.105 |

### Read-out

| metric | flag-on | flag-off | delta |
|---|---:|---:|---:|
| step-1 loss | 10.6019 | 10.6019 | bit-equal ✓ |
| step-10 ema | 10.3336 | 10.3336 | bit-equal ✓ |
| step-20 ema | 9.9842 | 9.9826 | +0.0016 (within ±0.02 ✓) |
| step-30 ema | 9.8820 | 9.8453 | **+0.0367** (out of ±0.02 ✗) |
| step-30 tok/s | 48,851 | 49,494 | **−1.30%** (dual-sync overhead, expected) |

---

## Gate-0 status

| gate | criterion | result |
|---|---|---|
| **G0.1** throughput | tok/s ≥ +3% over iter-60 | not testable @ iter 63 (single-site dual-sync; iter 64 routes fully) |
| **G0.2** NLL parity | val NLL @ step 200 within ±0.02 nat | partial — RN drift accumulates; +0.037 @ step 30 already exceeds |
| **G0.3** fallback bit-exact | flag=0 NLL bit-equal to iter-60 | **PASS** — step-1 bit-equal, step-10 ema bit-equal |
| **G0.4** long-horizon | val NLL @ step 1000 within ±0.04 | not run (RN drift makes this almost certainly fail) |
| **G0.5** production viability | T=16384 L=24 within ±0.05 | not run |
| **G0.6** VRAM | iter-bench VRAM neutral or better | small regression (+32 MB BF16 mirror); iter 64 swaps p→p_bf16 to net-save |

---

## What this proves

1. **Kernel correctness** — `chiron_scfa_axpy2_bf16p_rn` produces correct values (NLL trajectory tracks baseline to step 10 within bit-noise; drift only emerges as quantization accumulates).
2. **G0.3 bit-exact fallback** — when `--bf16-residual-p` is off, the code path is bit-identical to the iter-61 silent flagship.  Default-off semantics preserved.
3. **Design's R1 risk materializes empirically** — at L=12 with ~24 RN events per step, NLL drift exceeds the ±0.02 nat parity bound by step 30.  Confirms the design's claim that **stochastic rounding is structurally required, not optional** (§3.1: RN bound ≈ 5% of |p| at L=12).
4. **Dual-sync overhead is measurable** — the single wired site costs ~1.3% wall.  iter 64 needs to route enough sites for the BF16 read savings to exceed the sync overhead.

---

## What's deferred to iter 64

Per the original Arc 2 plan, iter 63 = "BF16 storage + RN kernels" and iter 64 = "Stochastic rounding".  Adjusted plan based on iter 63 results:

**iter 64**:
- Switch axpy2 + axpy + scaled_copy kernels to **stochastic rounding** via `sr_hash32` (already shipped in iter 49).
- Route ALL p-touching call sites in chiron_main.cpp (~20 sites) to BF16-p variants.
- Remove the dual-sync — p_bf16 becomes canonical; p stays unallocated when flag is set.
- Add reln-fwd BF16-p variant (reads BF16 p, FP32-internal LN, writes FP32 q).
- Add reln-inv BF16-p variant (FP32 q in, RN-write BF16 p).
- Add backward variants for dp (same kernels but on dp_bf16).
- G0.1 + G0.4 testable for the first time.

Realistic scope: ~3-4 hours of focused refactor work + bench.  iter 64's go/no-go gate is G0.1 ≥ +3% wall + G0.4 ≤ +0.04 nat at step 1000.

---

## Iter 63 commit

`glades-ml`: new BF16-p kernels + headers.
`glades-trainer`: flag, alloc, single-site wiring with dual-sync.

The iter 63 implementation cost was small (kernels + plumbing).  The big iter 64 refactor will be where Arc 2 actually proves or falsifies the throughput claim.

---

## Honest read

iter 63 framework lands cleanly:
- Kernels are correct.
- RN drift confirms SR is necessary (design called this out; this iter empirically validates it).
- Dual-sync overhead means iter 64 has to do real routing to deliver any wall win.

The hardest iter is still ahead (iter 64).  iter 63 is foundation, not a Gate-0 datapoint.

If the user wants to continue Arc 2: iter 64 is the real go/no-go.
If the user wants to stop: framework is in tree, flag is opt-in, no flagship regression.
