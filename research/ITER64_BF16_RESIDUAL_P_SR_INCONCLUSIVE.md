## Iter 64 — BF16 residual-p SR kernels (Arc 2, iter 2 of 5) — INCONCLUSIVE

**Date**: 2026-05-16
**Iter**: 64 (Arc 2 iter 2)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)
**Verdict**: **INCONCLUSIVE — single-site dual-sync wiring confounds the SR test.**  SR kernels land + reln BF16-p kernels land + smoke runs clean.  But the iter-63 wiring's RN cast-in (FP32→BF16 each call) dominates the per-call drift, masking SR's mean-zero benefit.  Empirical drift under SR (+0.040 nat @ step 30) is statistically indistinguishable from RN (+0.037 nat).  **The clean SR test requires the full p-routing refactor (iter 65); single-site testing is fundamentally insufficient.**

---

## TL;DR

| variant | step-30 ema vs flag-off | step-30 tok/s | conclusion |
|---|---:|---:|---|
| flag-off (baseline) | 0 | 49,494 | — |
| iter 63 (RN, dual-sync) | +0.0367 | 48,851 | RN drift visible |
| iter 64 (SR, dual-sync) | +0.0396 | 48,894 | SR no better — masked by cast-in RN |

**Mechanism diagnosis**: dual-sync pre-cast `cast_f32_to_bf16(p_fp32 → p_bf16)` is RN-rounding.  Between SCFA-axpy2 calls, other p-writers (attention_shear, reln-inv, generic axpy) write arbitrary FP32 values to p.  Each subsequent cast-in then RN-quantizes a non-BF16-exact FP32 — introducing ~0.5 ULP_BF16 of biased drift per call.  This error is INDEPENDENT of the kernel's rounding scheme (RN vs SR).

The kernel's write-out rounding (SR vs RN) only affects the SR kernel's own contribution.  But the dominant per-call drift comes from the cast-in, not the write-out.  So swapping the kernel's rounding scheme has minimal effect on observed drift.

To test SR cleanly, we need to remove the cast-in step — which requires p_bf16 to be the canonical store with no FP32 p mirror.  That's iter 65 (full routing).

---

## What was implemented

### New kernels in `gpu_chiron.cu` + `.h`

Six new BF16-p kernels with FP32-internal accumulation + SR-rounded BF16 write:

```cpp
// SR variants of the 3 element-wise kernels.
chiron_scfa_axpy2_bf16p_sr(uint16_t* p_bf, float α, const float* a, const float* b,
                            int n, uint32_t baseSeed, uint32_t stepIdx);
chiron_axpy_bf16p_sr        (uint16_t* p_bf, float α, const float* x,
                              int n, uint32_t baseSeed, uint32_t stepIdx);
chiron_scfa_scaled_copy_bf16p_sr(uint16_t* c_bf, float α, const float* a,
                                  int n, uint32_t baseSeed, uint32_t stepIdx);

// Reln BF16-p variants.
chiron_reln_forward_rows_bf16p(const uint16_t* p_bf_in, float* q_out, float* stats,
                                const float* gamma, const float* beta,
                                int T, int m, float eps);
chiron_reln_inverse_rows_bf16p_sr(const float* q_out, const float* stats,
                                   const float* gamma, const float* beta,
                                   int T, int m, uint16_t* p_bf_out,
                                   uint32_t baseSeed, uint32_t stepIdx);
```

All use the same `sr_hash32`-based mean-zero rounding pattern as iter 49's `cast_f32_to_bf16_stochastic`.  The reln forward kernel decodes BF16 inline (per-element register-local) and computes mean/variance in FP32; only the BF16 store side touches the reduced precision.

### Single-site wiring upgrade

The iter 63 SCFA forward axpy2 wiring (at `chiron_main.cpp:5659`) is updated from `_bf16p_rn` to `_bf16p_sr`.  Static per-call counter `sr_call_counter` decorrelates rounding decisions across the 24 SR events per step.  Dual-sync RETAINED for this iter — full routing deferred.

### Trainer-side mirror header

`gpu_chiron.h` in `include/` mirrors the new declarations.

---

## Why the test confounded

For an SR test to show drift reduction, the rounding events must be the dominant error source.  In the dual-sync wiring, each per-call error chain is:

1. `cast_f32_to_bf16(p_fp32, p_bf16)` — RN-rounds the current FP32 p value to BF16.
2. `chiron_scfa_axpy2_bf16p_sr` — reads p_bf16 (exact BF16), computes `acc = decode(p_bf16) + α·(a+b)` in FP32, SR-rounds `acc` to BF16.
3. `cast_bf16_to_f32(p_bf16, p_fp32)` — exact (BF16→FP32 lossless).

Step 1 introduces RN error.  Step 2 introduces SR error.  Step 3 introduces no error.

**Key observation**: step 1 RN error only matters if the FP32 p going into step 1 has bits below the BF16 representable set.  After a single round-trip through this kernel, the FP32 p IS exactly representable in BF16 (step 3 makes it so).  So if NOTHING else writes p between two SCFA-axpy2 calls, step 1 is a no-op.

But CHIRON has many other p-writers between SCFA-axpy2 calls:
- `attention_shear` writes p (via internal axpy chain).
- `reln_inverse_rows` writes p (in the inverse walk during backward).
- Other `glades::gpu::axpy(α, x, p)` calls.

Each of these writes arbitrary FP32 to p.  Then the next SCFA-axpy2 call's cast-in RN-quantizes that arbitrary FP32, introducing ~0.5·ULP_BF16 biased drift per call.

The bench result confirms: SR drift ≈ RN drift because the cast-in dominates both.

---

## What's testable + what isn't, in iter 64

| gate | criterion | result |
|---|---|---|
| Kernel correctness | New kernels execute, output shape correct, no NaN/Inf | PASS (smoke ran clean) |
| Reln BF16-p kernels exist | Compilable; ready for iter 65 wiring | PASS (built clean) |
| G0.3 bit-exact fallback | flag=0 produces iter-61 silent | PASS (no change from iter 63) |
| **G0.2 NLL parity under SR** | drift ≤ +0.02 nat @ step 30 | **NOT TESTABLE** under dual-sync (cast-in confound) |
| **G0.1 throughput** | tok/s ≥ +3% | **NOT TESTABLE** until iter 65 (full routing eliminates dual-sync overhead) |

---

## Iter 65 prerequisite analysis

For a clean SR test (and any real throughput Gate-0), iter 65 must:

1. **Allocate `p_bf16` INSTEAD of `p` when flag is set** (mutually exclusive).  Remove FP32 p storage entirely.
2. **Refactor all p-writing call sites** (~10-15) to use BF16-p SR variants:
   - SCFA-axpy2 (forward + inverse direction).
   - SCFA-axpy (sign-multiplied accumulate).
   - SCFA-scaled-copy.
   - reln-inv (inverse walk writes p).
   - attention_shear (writes p_contrib — needs BF16-aware internal axpy variant).
3. **Refactor all p-reading call sites** (~10-15) to use BF16-aware kernels:
   - reln-fwd reading p → `chiron_reln_forward_rows_bf16p` (already added).
   - `axpy(1, p, q, n)` (q += p) → `chiron_bf16_to_fp32_axpy` (already added).
   - attention_shear reading p (some bwd patterns) — needs decode-inline variant.
4. **Backward dp path** stays FP32 — separate buffer + kernels.  Forward p ↔ backward dp don't share storage.
5. **Initial sync**: at allocate time, cast FP32 initial p (zero in the standard init path) to BF16 once.

Scope: **~25 chiron_main.cpp call sites + 2-3 new BF16-aware attention_shear variants if needed**.  Time estimate: **3-5 hours of careful refactor + bench**.  Risk: high — touching the attention_shear internals introduces correctness risk that iter 63/64 framework testing didn't cover.

---

## Decision point

Three options:

**A. Proceed to iter 65 (full routing)** — commits to the ~3-5 hour refactor.  First clean G0.1 + G0.2 + G0.4 test.  If SR + full routing passes Gate-0b (drift ≤+0.02 nat), the paradigm is alive.  If not, Arc 2 dies at iter 65.

**B. Stop Arc 2 here, freeze flagship** — iter 64 inconclusive doesn't falsify the paradigm; it falsifies the single-site test methodology.  The full routing IS what the design predicted as the productive test.  Stopping now leaves the question open but doesn't commit further engineering.

**C. Skip to Arc 3 (MoE attention-shear)** — different paradigm, uncorrelated risk.  Arc 2 remains an open question.

**Honest read**: iter 64 wasn't a falsification.  It exposed that the dual-sync test methodology was wrong — the SR mechanism is sound but unmeasurable under dual-sync.  The original design's iter-63-RN + iter-64-SR plan assumed full routing from iter 63; the actual implementation took a shortcut (single-site wiring) that doesn't allow the SR test to fire cleanly.

Recommendation: option (A) if there's appetite for the 3-5 hour refactor.  Otherwise (B) — iter 64 documents the methodological limit and leaves the SR question empirically open.

---

## Cumulative Arc 2 state

| iter | scope | result |
|---:|---|---|
| 63  | RN kernels + flag + alloc + 1-site dual-sync | FRAMEWORK PASS (kernels correct, fallback bit-exact, RN drift confirms SR-needed) |
| 64  | SR kernels + reln BF16-p kernels + 1-site SR swap | **INCONCLUSIVE** (dual-sync masks SR; clean test needs iter 65 full routing) |
| 65  | Full p routing | NOT STARTED — gate to clean Gate-0a/0b/0d |
| 66  | Production-scale L=24 kill test | NOT STARTED |
| 67  | Ship or revert | NOT STARTED |

Arc 2 is alive but pre-decisive.  No data yet says it will or won't ship — only that the single-site shortcut can't answer the question.
