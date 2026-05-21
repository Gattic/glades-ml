## Iter 111 — Skip per-iter dq_buf memcpy via in-place reln_backward — FAIL on math

**Date**: 2026-05-21
**Iter**: 111 (post iter 110 META)
**Branch**: vesta5 (glades-trainer)
**Verdict**: **FAIL** — in-place reln_backward broke math at the whole-function level even though kernel 1 alone (layernorm_backward_dx) is safe in-place. NLL drift +0.12 nat at step 100, ||g|| explosion to 23.5 (vs baseline 1.2). Reverted to no-op flag.

---

## Motivation

iter 110 META identified the line 9201 memcpy (s.dq_buf → s.dq, 3.2 GB/step) as the largest remaining iter-103-class target. Approach: alternate dq_buf as INPUT for ll > 0 iterations of the bwd loop, so reln_backward reads from dq_buf and (potentially in-place) overwrites it with the new dq_pre_reln.

## Conjecture (pre-committed)

Skip the per-iter memcpy. For ll == 0, reln_backward(s.dq → s.dq_buf) — separate buffers, safe. For ll > 0, reln_backward(s.dq_buf → s.dq_buf) — in-place. Add final post-loop memcpy(s.dq, s.dq_buf) once for downstream consumers.

Analysis (pre-bench): layernorm_backward_dx kernel single-assigns dxRow[i] = ... for every (row, i). Within each thread, read of dRow[i] happens before write of dxRow[i] for the SAME i. Cross-thread: each thread handles distinct i values. Conclusion: in-place safe at the kernel level.

## Bench (single-seed × 100 steps × seed=1337 × T=16384 L=24 w=4)

| run | wall | tok/s @ 76 | loss @ 76 | NLL @ 100 | ||g|| @ 76 | verdict |
|---  |---:  |---:        |---:       |---:       |---:         |---      |
| iter 107+109 stack baseline | 60.27s mean | 27,302 | 10.18 | 9.847 | 1.20 | PASS |
| **iter 111 active (seed 1337)** | **59.3s** | **27,750** | **10.2584** | **9.9639** | **23.561** | **FAIL** |

**Wall improved as predicted** (-1.0s = +1.6% additional). **But math is BROKEN**:
- NLL drift +0.117 nat (vs ±0.02 strict bound)
- ||g|| 23.561 at step 76 (vs ~1.2 baseline) — gradient explosion
- Training trajectory diverging: best loss 10.2287@56, drifting from there

## Root cause (post-bench analysis)

`chiron_reln_backward` delegates to `layernorm_backward(dout, x, gamma, mean, invStd, T, m, dx, dgamma, dbeta)` which launches **TWO sequential kernels** on the same stream:

1. **Kernel 1**: `layernorm_backward_dx` (gpu_kernels.cu:201) — reads dout, x; **writes dx**
2. **Kernel 2**: `layernorm_backward_dgamma_dbeta_partial` (called inside same dispatcher) — reads **dout**, x; writes dgamma_partial, dbeta_partial

When dout == dx (in-place):
- After Kernel 1 completes, dx contains the new dx values (DIFFERENT from original dout).
- Kernel 2 starts and reads dout — but dout now holds the new dx values, **not the original dout**.
- dgamma/dbeta computed from corrupted "dout" → wrong gradient.

The math break is at the **function-boundary** level: kernel 1 alone is safe in-place, but kernel 2 reads dout AFTER kernel 1 has overwritten it.

## Lesson (extending iter 108 boundary)

The "eliminate redundant memory ops" mechanism class has a new boundary marker:

| pattern | safe in-place? | example |
|---|---|---|
| Single kernel, dout==dx, single-assign | YES | layernorm_backward_dx alone |
| Function with multiple kernels reading dout | NO | layernorm_backward (kernel 1 writes dx, kernel 2 reads dout) |
| Multi-write cuBLAS accumulator (BF16-dst) | NO (iter 108 FAIL) | dW_bf in shear_backward |

The analytic argument must consider ALL kernels in the function chain that read the input buffer — not just the kernel that writes the output.

## Resolution

iter 111 flag retained as documented no-op:
- `iter111Active = false` always (hard-coded)
- `(void)cfg.iter111SkipDqBufMemcpy` to silence unused warning
- `dq_in_ptr = s.dq.data()`, `dq_out_ptr = s.dq_buf.data()` (legacy behavior)
- All 3 per-iter memcpys and the final post-loop memcpy still gated on `iter111Active` (which is now false), so legacy memcpy fires.

Production behavior unchanged from iter 109 stack. iter 110 META's documented +8.35% multi-seed strict-bar PASS remains the production-ready stack.

## Mechanism class boundary update

After iter 111 FAIL, the established boundary for "skip redundant pre-zero / memcpy":

| safe | example |
|---|---|
| Pre-zero of buffer where downstream single-write kernel covers all positions | iter 106 (yperp), iter 109 (dq_buf) |
| Caller-side memcpy that's pure buffer rename (different buffers) | iter 103 (bwd dy), iter 99 (dual_out kernel) |
| cuBLAS beta=0 with FP32-destination single-write | iter 107 (sdQ via flash_attention_backward) |

| unsafe | example |
|---|---|
| In-place dout/dx for multi-kernel function | iter 111 FAIL (layernorm_backward) |
| cuBLAS beta=0 with BF16-destination accumulator | iter 108 FAIL (dW_bf) |
| Memcpy/memset <200 MB/step | iter 104 sub-noise |

## Files

- This document.
- `research/runs/2026-05-21-iter111-gate0/iter111_combined_100step.log` (failure evidence).
- Code:
  - `glades-trainer/trainer/chiron_main.cpp`: flag retained as no-op (iter111Active = false hard-coded).
  - No glades-ml change.
