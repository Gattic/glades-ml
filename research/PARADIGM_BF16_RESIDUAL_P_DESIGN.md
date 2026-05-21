# PARADIGM: BF16 Residual-p for CHIRON

**Status**: design proposal, 5-iter arc (iters 62–66)
**Date**: 2026-05-16
**Context**: option (A) of the iter-60 META Option-A trajectory after the post-iter-61 reckoning; alternative to FP8-output readout and sparse-attn replacement of SCFA.

---

## 1. TL;DR

Quantize the CHIRON secondary residual stream `p ∈ R^{T×m}` from FP32 to BF16 storage, with FP32 internal arithmetic, **stochastic-rounded writes**, and bit-exact fallback when disabled.

| axis | mechanism | expected |
|---|---|---|
| wall-clock @ iter-bench (T=8192, L=12) | halve HBM traffic on the ~16% SCFA element-wise slice | +3% to +5% |
| VRAM @ iter-bench | drop `p` + `dp` to BF16 | ~0.8 GB saved incl. scratch |
| VRAM @ production (T=16384, L=24) | 4× larger savings | ~3.2 GB saved |
| NLL parity | bit-exact w/ flag=0; conj. ≤+0.02 nat at L=12 w/ SR; **at-risk at L=24** | risk-bound |

Risk is dominated by reversible-flow inverse drift: the inverse path subtracts `δ_l` from a quantized `p`, and `δ_l` recomputed at inverse-time differs from forward-time `δ_l` by quantization noise that propagates through L layers. Stochastic rounding (SR) is the principled remedy: it makes per-layer drift a mean-zero random walk (variance `O(√L · ε_BF16)`) instead of a biased accumulation (`O(L · ε_BF16)`).

**This is one of the more dangerous paradigms in the queue**: the math is *not* obviously well-conditioned at BF16 across L=24 and only iter-65's production-scale test can resolve it. Honest forecast: ~50% ships at iter-bench, ~25% ships at production.

---

## 2. Motivation

The iter-60 nsys profile shows the dominant remaining slice on the iter-bench flagship is the SCFA element-wise operator family over the FP32 residual streams:

| kernel | %GPU | inst./50 step | op |
|---|---|---|---|
| `chiron_scfa_axpy2` | 6.2–6.5% | 339–357 | `p += α·(a + b)` |
| `chiron_scfa_sub` | 4.8–5.0% | 338–356 | `c = a − b` |
| `axpy_kernel` generic | 2.9–3.1% | 213–225 | `y += α·x` |
| `chiron_reln_forward_rows` | 1.7% | per-layer | `q += LN(p)` |
| `chiron_reln_inverse_rows` | 1.7% | per-layer | `p ⊖ LN_inv(q)` |
| **total p-touching** | **~16%** | | |

All five are memory-bandwidth-bound (1–2 FMAs/elem; `dram__bytes` > 85% peak on iter-60 nsys). Per-call traffic at T·m·4 = 64 MB per FP32 buffer. Halving the `p` half of that — but keeping `a`, `b` at FP32 since they're freshly computed scratch — drops per-call traffic by ~25%. The Amdahl bound on end-to-end is `0.16 × 0.30 ≈ 4.8%`. The +3–5% target sits at the upper edge of that bound and is achievable only because the slice is genuinely bandwidth-bound.

Secondary benefit: ~3.2 GB freed at production unblocks (a) extending T to 24576, (b) growing L to ~28, or (c) larger batch — paradigm-arc fuel, not iter-62 deliverables.

---

## 3. Numerical analysis

### 3.1 Accumulation across L

Each CHIRON reversible step writes a residual `p_{l+1} = p_l + δ_l(q_l)`. Per-step rounding error scales with the ULP of the storage format:
- FP32: ≤ 0.5 · ULP_{FP32} · |p_{l+1}| ≈ 6×10⁻⁸ · |p_{l+1}|
- BF16 (FP32-internal accum): ≤ 0.5 · ULP_{BF16} · |p_{l+1}| ≈ 4×10⁻³ · |p_{l+1}|

A factor of ~65,000 larger per step.

For L sequential writes with round-to-nearest-even (RN), worst-case drift is `O(L · ε_BF16 · sup|p|)` — *systematic*, because RN errors can sign-align. With SR, the rounding error has E=0 by construction, so drift is a random walk with `Var ≤ L · ULP²_{BF16} · sup|p|² / 12`. Expected magnitude scales `O(√L · ε_BF16 · sup|p|)`.

At L=12: RN bound ≈ 5% of |p| (marginal); SR ≈ 1.4% (in-bound).
At L=24: RN bound ≈ 10% (out-of-bound); SR ≈ 2% (borderline).

**Conclusion: SR is structurally required, not optional.** This is the decisive theoretical reason.

### 3.2 Why BF16-weights ships but BF16-residual-p might not

CHIRON ships BF16 weights (iter 49) with no NLL drift. Two structural differences:

| axis | BF16-weights | BF16-residual-p |
|---|---|---|
| update / state ratio | `|η·∇W|/|W| ≈ 10⁻⁴` (tiny) | `|δ_l|/|p_l| ≈ O(1)` (residual IS the update) |
| adds before readout | 1 per step | L = 12–24 per forward |
| accumulation regime | per-step noise on a slow drift | per-step noise compounds *within* one forward |

The number of compounding adds before a stability-critical readout is **L× larger here**.

### 3.3 The reversible-flow inverse path

Forward (eq. *):
```
p̂_{l+1} = round_φ( bf16_to_fp32(p̂_l) + δ_l )                                  (*)
```
Inverse (used by backward to reconstruct `p_l`):
```
p̃_l = bf16_to_fp32(p̂_{l+1}) − δ̃_l(q̃_l)
p̂_l = round_φ(p̃_l)
```
where `δ̃_l` is recomputed and `q̃_l` is the reconstructed q. Two error sources per round-trip layer:

1. **Forward quantization**: `bf16(p+δ) = p+δ + ε_fwd`, with `|ε_fwd| ≤ 0.5·ULP_{BF16}·|p+δ|`.
2. **δ mismatch**: even with exact weights, `δ̃ = δ + J_δ · ε_q` where `ε_q` is q-side drift.

Holding `q` in FP32 eliminates `ε_q` to first order, isolating the analysis to (1). The total inverse-path error after L back-steps is `Σ_l ε_fwd_l`, identical in bound to §3.1 but *with one more L worth of accumulation* (forward and inverse both contribute), so the effective accumulation depth is `2L`. SR converts this to `O(√(2L)·ε_BF16)`, only ~40% larger than the forward-only bound.

### 3.4 Comparison to existing residual-quantization codebases

| codebase | residual | stable depth | inverse reads quantized? |
|---|---|---|---|
| Hugging Face autocast (BF16) | BF16 + FP32 master | L≤96 | N/A (no inverse) |
| FlashAttention 2 | BF16 between blocks | L≤80 | N/A |
| Megatron-LM mixed | BF16 + FP32 master | L≤96 | N/A |
| RevNet / Reformer | FP32 reversible | L≤48 | yes (FP32 — exact) |
| **BF16-residual-p (this)** | BF16 p, FP32 q | conj. L≤24 | **yes (BF16) — novel** |

The novel risk surface: SR-quantized residuals on a reversible-flow inverse path has no published precedent. RevNet variants all assume FP32 exact reconstruction.

---

## 4. Mathematical formulation

### 4.1 State space

```
p̂_l ∈ BF16^{T×m},   p̂_l = round_φ(p_l)
q_l ∈ FP32^{T×m}    (unchanged)
```

`round_φ`:
- `round_RN`: round-to-nearest, ties-to-even (deterministic; biased accumulation)
- `round_SR(u)`: stochastic rounding with `u ∼ U[0,1)` per element; `E[round_SR(x)] = x` exactly

Per-element implementation: cast FP32 → uint32; SR is `out = (in_u32 + (rnd & 0xFFFF)) >> 16`, where `rnd = sr_hash32(elem_idx, step_idx, seed)`. The hash is the same `sr_hash32` infrastructure already shipped in `gpu_kernels.cu` (iter 49), zero new RNG infrastructure.

### 4.2 Operator dataflow

```
SCFA-sub:        c_fp32 := a_fp32 − b_fp32                    (unchanged; output FP32)
SCFA-axpy2_bf:   p̂ := round_φ( bf16_to_fp32(p̂) + α·(a + b) )
axpy_bf:         p̂ := round_φ( bf16_to_fp32(p̂) + α·x )
reln-fwd_bf:     reads p̂ (decode→fp32 for row mean/var), writes q_fp32
reln-inv_bf:     reads q_fp32, writes p̂ via round_φ
```

Only `p` is BF16; `q`, `a`, `b`, reln stats stay FP32. Reln reductions (mean, variance) compute in FP32 even when input is BF16 — only storage changes, not arithmetic.

### 4.3 Recovery as a limit

```
lim_{round_φ → identity}  CHIRON_{bf16-p}  ≡  CHIRON_{fp32-p}     (current flagship)
```

With `--bf16-residual-p=0`: no BF16 buffer allocation; all kernels dispatch to existing FP32 paths; output bit-equal to iter-60.

---

## 5. Implementation outline

### 5.1 New kernels (in `gpu_chiron.cu`)

```cuda
__global__ void chiron_bf16_axpy_sr_kernel(
    uint16_t* p_bf16, float alpha, const float* x_fp32,
    uint32_t srBaseSeed, uint32_t srStepIdx, int n);

__global__ void chiron_scfa_axpy2_bf16_sr_kernel(
    uint16_t* p_bf16, float alpha,
    const float* a, const float* b,
    uint32_t srBaseSeed, uint32_t srStepIdx, int n);

__global__ void chiron_reln_forward_rows_bf16p_kernel(
    const uint16_t* p_bf16,
    const float* gamma, const float* beta, float eps, int cols,
    float* q_out, float* stats);

__global__ void chiron_reln_inverse_rows_bf16p_kernel(
    const float* q_out_fp32, const float* stats,
    const float* gamma, const float* beta, int cols,
    uint16_t* p_in_bf16,
    uint32_t srBaseSeed, uint32_t srStepIdx);
```

Five mirror wrappers in `gpu_chiron.h` (one per kernel + a `chiron_scfa_scaled_copy_bf16` variant). Each is a structural mirror of its FP32 counterpart — same launch geometry, same grid/block sizing, only the storage type and SR step changed.

### 5.2 Storage refactor

In `GpuTransformerScratch`:
```cpp
GpuBuffer<float>     s_p;          // existing
GpuBuffer<uint16_t>  s_p_bf16;     // new, gated on cfg.bf16ResidualP
GpuBuffer<float>     s_dp;         // existing dual
GpuBuffer<uint16_t>  s_dp_bf16;    // new
```

`s_p` and `s_p_bf16` are mutually exclusive — one or the other is allocated per session, based on the config flag. The corresponding `dp` (backward dual) follows the same gating since CHIRON backward reads p̂.

### 5.3 Trainer call-site changes

Five SCFA call sites in `chiron_main.cpp` need branching (lines ~5432, 5472, 5595, 6001, 6155 — found via existing grep). Each branch is a 5-line dispatch to the BF16 variant when `cfg.bf16ResidualP && cfg.scfaInner` is set. The reln call sites (in `chiron_main.cpp` and `gpu_chiron.h`) need similar dispatch.

Suggested abstraction: a small inline template `dispatch_p_axpy2<bool BF16>(...)` that gets monomorphized once per build, avoiding runtime branches in the inner-loop kernels.

### 5.4 Trainer flag

```
--bf16-residual-p[=0|1]
```

Default `0` (off). When `1`:
- Allocates `s_p_bf16` instead of `s_p`.
- Routes SCFA + reln calls to BF16 variants.
- Sets `cfg.bf16ResidualPSeed` deterministically from training seed.

### 5.5 Save/load compatibility

Checkpoint format adds a one-bit `bf16ResidualP` flag. On load:
- flag match: load directly.
- flag mismatch: on-the-fly round (FP32→BF16 via RN) or upcast (BF16→FP32). One-time cost, no training-loop impact.

A `chiron-resume-bf16p-parity` unit test exercises both directions for a 4-layer d=64 T=16 model.

---

## 6. Gate-0 criteria

| test | config | bar |
|---|---|---|
| G0.1 throughput | iter-bench, 200 steps × 3 runs | tok/s ≥ +3% over iter-60 (47,271); bench-noise ≤0.5% |
| G0.2 NLL parity | iter-bench | val NLL @ step 200 within ±0.02 nat of iter-60 (8.2391) |
| G0.3 fallback bit-exact | `--bf16-residual-p=0`, 100 steps | NLL bit-equal (|Δ| ≤ 1e-6) to iter-60 |
| G0.4 long-horizon drift | iter-bench, 1000 steps | val NLL @ step 1000 within ±0.04 nat |
| G0.5 production viability | T=16384, L=24, 500 steps | val NLL within ±0.05 nat of production flagship |
| G0.6 VRAM | iter-bench | ≤7.99 GB (iter-60 neutral or better) |

G0.4 is the drift detector; G0.5 is the kill criterion.

---

## 7. Risks and falsification triggers

| risk | mechanism | mitigation | falsifier |
|---|---|---|---|
| R1 NLL drift at L=24 | RN accumulation exceeds bound, possibly even SR | SR + reln re-normalization caps `sup|p|` | G0.5 → abort arc |
| R2 SR RNG cost overshadows BW saving | hash adds ~5% to memory-bound kernel | hash is ~2 cycles vs ~hundreds for HBM | G0.1 fails → fall back to RN-with-anchor |
| R3 Inverse path divergence | δ̃ mismatch dominates ε_fwd | q stays FP32 isolates ε_q to zero | G0.4 fails at L=12 (isolated test) |
| R4 Checkpoint break | one-bit tag mishandled | dedicated unit test for bidir resume | resume parity test fails |
| R5 Iter-50-style FMA reordering drift | this paradigm changes storage not arithmetic | audited; only round_φ at writes | iter-50-style drift in G0.4 |
| R6 Downstream GEMM reads p as input | double quantization through cuBLAS | audit confirms only element-wise/reln readers | found readers → fallback path |

R1 and R3 are dominant. R6 is implementation diligence — verify no GEMM reads `p` directly (current dataflow: only reln and SCFA element-wise kernels do).

---

## 8. Iter budget

### iter 62 — BF16 storage + RN kernels
Goal: validate kernel throughput; expect NLL drift.
- Implement all kernels with deterministic `round_RN` (simpler, no RNG yet).
- Run G0.1, G0.2 (expect partial), G0.3.
- **Go**: G0.1 passes + G0.3 bit-exact works.
- **No-go**: kernel speedup absent → abort arc; option A falsified at Amdahl level.

### iter 63 — Stochastic rounding
Goal: control drift via SR.
- Switch kernels to `round_SR` via existing `sr_hash32`.
- Re-run G0.1, G0.2, G0.4.
- **Go**: G0.4 ≤ 0.04 nat over 1000 steps.
- **No-go**: SR insufficient → try per-token local scale (mini-anchor) or abort.

### iter 64 — reln integration
Goal: extend BF16-p to the reln kernels (trickiest — row reductions must keep FP32 accum).
- Implement reln-fwd-bf16p and reln-inv-bf16p with FP32-internal reductions.
- Re-run G0.4.
- **Go**: drift still bounded.
- **No-go**: fall back — keep reln on a transient FP32 ghost-p (~64 MB cost at iter-bench).

### iter 65 — Production-scale kill test
Goal: the L=24 falsifier.
- Run T=16384, L=24, 500 steps; track NLL drift vs production flagship.
- This iter empirically tests the L=24 worst-case drift bound (§3.3).
- **Go**: G0.5 passes → iter 66 ships.
- **No-go**: drift > 0.05 nat → abort, document falsification at production scale.

### iter 66 — Ship or revert
- Pass: replace flagship; update memory; cumulative throughput report.
- Fail: revert default-on; keep code in tree behind `--bf16-residual-p=0` as research artifact.

---

## 9. Honest expectation

Highest variance of the three iter-60 META options:

| paradigm | expected upside | downside | budget |
|---|---|---|---|
| BF16 residual-p (this) | +3–5% wall, +0.8 GB iter-bench, +3.2 GB prod | NLL drift at L=24 (R1) | 5 iters |
| FP8-output readout | +2–4% wall on readout GEMM | low (FP8 BF16-like exponent) | 3 iters |
| Sparse-attn replacement | +10–50% wall (if it works) | very high (architecture change) | 8+ iters |

Probabilistic forecast: ~50% ships at iter-bench, ~25% ships at production. The 25% reflects the L=24 drift uncertainty no pre-implementation theory can resolve.

Most likely failure mode: ships at iter-bench, fails production. Deliverable becomes a **scale-conditional flag**: `--bf16-residual-p` on at T=8192 L=12, off at T=16384 L=24 — a sharp empirical characterization of the depth-precision tradeoff in reversible-flow models, which itself is a publishable research artifact.

---

## 10. Open conjectures (for follow-up iters)

- **C1**: SR-quantized residual-p preserves NLL within ±0.02 nat for L ≤ 24 at the iter-bench training distribution. *(Tested by G0.4 + G0.5.)*
- **C2**: The dominant drift contribution comes from the inverse path, not the forward (each contributes ~L/2 of total 2L accumulation). *(Test: forward-only NLL without inverse reconstruction.)*
- **C3**: With SR at L=24, the depth ceiling for stable training is `L_max ≈ 50`. *(Test: L sweep 24/32/40/48.)*
- **C4**: A k=8 FP32 ghost-p anchor caps drift to k-step variance and unlocks L=96 (mirrors CHIRON_framework §11a's sketch-anchor remedy). *(Recovery move if C3 indicates an L-ceiling.)*

C2 and C3 are natural follow-ups if iter 65 passes; C4 is the recovery move if iter 65 fails.
