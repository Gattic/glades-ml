# FA-fused SCFA inner — Gate-1 decision META (2026-06-12)

**Arc:** FlashAttention-fused SCFA inner attention (iter 118 Gate-0 PASS,
iter 119 BF16-input NULL, iter 120 ceiling META).
**Verdict:** **WMMA full-fusion NO-GO at current EV; thread redirected** to
cast-pipeline elimination (measured 9.1% of GPU kernel time), which the
fresh profile shows is the larger and lower-risk wall target.

---

## 1. Fresh measurement (SIRA+clamp flagship recipe)

First kernel-level profile since the regstack/SIRA/clamp ships (all prior
arc numbers were iter-116-era). 30 steps, seed 1337, full production
recipe, nsys 2026.1.3:
`glades-trainer/logs/nsys_flagship_sira_clamp_20260612.nsys-rep`.

Top relevant entries (% of GPU kernel time):

| component | % | notes |
|---|---:|---|
| `k_cast_f32_to_bf16` | **8.2** | 35,549 instances ≈ **1,185/step**, avg 59 µs |
| `k_cast_bf16_to_f32` | 0.9 | 5,040 instances |
| `causal_mask_softmax_bf16_out` (fwd, iter 102) | 2.5 | 1,806 inst |
| `causal_softmax_with_bwd_attn` (iter 115 fused) | 1.5 | 720 inst |
| `causal_mask_softmax` (legacy bwd recompute) | 0.9 | 720 inst |
| QK-Norm kernels (fwd/bwd/gamma_grad) | 2.2 | regstack addition |
| `chiron_scfa_axpy2_dual_p` | 9.4 | #2 kernel overall |
| bf16 TC GEMMs (all shapes pooled) | ~30 | inner + outer + readout |

## 2. Why WMMA full-fusion is a NO-GO at current EV

The fusable inner pipeline = inner QK^T/PV GEMMs + softmax + P/QKV casts.

- **The matmul share is small.** Per iter 119's own arithmetic (~260
  GFLOP/step of inner-attention matmul), cuBLAS BF16-TC executes this in
  ~3–5 ms ≈ **<1% of the ~600 ms step**. The "10–15%" iter-105 figure
  bundled softmax + casts; the matmuls themselves are cheap.
- **What fusion can actually recover** = softmax family (4.9%) + the
  inner-attention slice of the casts + S/P DRAM traffic ≈ **6–8% ceiling
  IF the hand-written WMMA matmuls match cuBLAS efficiency**.
- **They won't match.** Hand-rolled WMMA typically reaches 60–80% of
  cuBLAS's tuned `ampere_s1688gemm_bf16` at these shapes, and dH=256
  doubles the standard FA-2 register budget (iter 118 risk #1). Matmul
  losses of 1–3% eat the fusion gains: **realistic net +2–4%**, below the
  +5% strict bar, at 300–500 LOC and moderate-high risk (the iter
  70/73/83 drift classes live exactly in fused-attention-math territory).
- This **confirms iter 120's META conclusion** with current-binary data.

Gate-0/iter-118's FP32 math validation remains on file; the arc can be
reopened if hardware changes (Hopper+ TMA/WGMMA) or if SCFA shapes change.

## 3. The redirect: cast-pipeline elimination (new wall arc)

The profile's actual headline: **9.1% of GPU kernel time is pure dtype
conversion**, ~1,185 f32→bf16 launches/step — not in the iter-105 top
entries. Prime suspects for the growth: the regstack QK-Norm
backward-recompute design (recomputes qNorm/kNorm via BF16-TC projection
each step — the documented −2.82% regstack wall cost), plus scfa-bf16-outer
per-call A-side casts (216 GEMMs/step) and Phase-2 BF16-grad scratch
mirrors.

**Mechanism class (proven SAFE in this codebase):** producer-side
dual-output — the producing kernel side-writes the BF16 mirror alongside
its FP32 output, eliminating the standalone cast launch. This is the iter
97/99/101 PASS pattern (side-write of an extra output), NOT the iter 70/73
FAIL pattern (re-fusing math chains → FMA-emit drift) and NOT iter 74's
vectorized-cast FAIL (kernel-internal restructuring).

**Plan (gate-staged):**
1. **Cast census** — re-profile with shape attribution (grid-size buckets)
   to rank the 1,185 launches/step by source (QK-Norm recompute, outer
   GEMM A-side, grad mirrors, logits path).
2. **Top-source dual-output ports**, one flag per source, parity-gated at
   the 300-step rerun-noise methodology (2026-06-11 finding: compare
   against a no-change rerun control, not print-identity).
3. **n=3 multi-seed bench** per the standing Gate-0 methodology; ship bar
   +3% (iter 60 precedent) for the stacked result.

**Predicted:** +3–6% wall at LOW risk (known-safe mechanism class,
flag-gated, math-identical side-writes). Cheaper than WMMA by an order of
magnitude in effort and risk.

## 4. State

- iter 118/119 flags remain default-OFF (math-validation reference).
- No code change in this META. Tasks for the cast arc are step 1 above.
- Production flagship unchanged (SIRA+clamp ship 2026-06-12).
