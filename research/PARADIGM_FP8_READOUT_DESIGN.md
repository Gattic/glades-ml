# Paradigm — FP8 Readout for CHIRON

**Date**: 2026-05-16
**Status**: design proposal (Gate-0 not yet attempted)
**Branch**: vesta5 (glades-ml)
**Slot**: 1 of 3 paradigm candidates (others: BF16 residual-p, MoE FFN sparsity)

---

## 1. TL;DR

Push the three readout GEMMs (`logits = q_L · W_E^T`, `dW_E += dlogits^T · q_L`, `dq_L = dlogits · W_E`) and the (T × V) logits / probs / dlogits storage from BF16 to FP8 (E4M3 for activations, E5M2 for backward errors). Keep softmax-in-registers FP32; FP8 lives only on cuBLASLt GEMM I/O and HBM-resident readout tensors.

| metric | target | revert trigger |
|---|---|---|
| tok/s win at iso-NLL | +3% to +6% over iter61 flagship | <+2% |
| NLL drift, val @ step 200 | ≤ +0.02 nat | > +0.05 nat |
| VRAM delta | −0.73 GB (3 × T×V tensors) | any net increase |
| readout-GEMM kernel time | 0.6× of BF16 (Ada FP8 = 2× peak vs BF16) | >0.85× |
| softmax row-mass error | <1e-5 per row | >1e-4 |

Readout is ~18% of GPU time at iter61, so even a 2× GEMM speedup caps overall gain near +9%. Realistic envelope after softmax/scale overhead is +3% to +6%. Viability rests on (a) softmax stability at V=32000 with FP8 inputs and (b) cuBLASLt FP8 actually beating BF16 on the readout shape on Ada — HELIUM (iter 55) showed it does **not** at attention shapes (N≤2048); the readout shape is 16× larger in N and may behave differently.

This paradigm is incremental. It does not break the 10× ceiling.

---

## 2. Motivation

### 2.1 iter61 dead-zone map

| slice | % GPU | attacked iter | status |
|---|---:|---|---|
| readout (3 GEMMs + softmax + xent + argmax) | ~18% | 3, 10, 52, 56, 58, 60, 61 | BF16-saturated |
| SCFA inner | ~15% | 1, 5, 47, 51 | BF16+ shipped |
| SCFA outer | ~12% | 2, 59 | cuBLAS dead-zone |
| LN-bwd | ~6.6% | 48, 53 | NLL-parity dead-zone |
| Adam fused | ~5% | 49 | int8 shipped |
| rest (<3% each) | ~43% | various | dead-zone |

Readout is the largest attackable slice and BF16 is at the floor:
- Inputs: BF16 via `--bf16-logits` (iter 3)
- Storage: BF16 via `--bf16-logits-storage` (iter 10, `sgemm_rowmajor_abt_bf16_bf16out` in `gpu_blas.h:210`)
- Softmax/CE/argmax: `softmax_stable_rows_bf16`, `softmax_cross_entropy_backward_bf16`, `argmax_count_bf16_warp_kernel` in `gpu_kernels.cu:3039,3075,3183`
- Parallel-bwd: NULL on Ada (iter 6, 58)
- cuBLAS algo override: NULL (iter 52)

Within the BF16 envelope, there are no more wins. The remaining lever is dtype.

### 2.2 Why FP8 might bite here when HELIUM didn't

Ada (sm_8.9) FP8 tensor cores deliver 2× BF16 throughput (~660 vs ~330 TFLOPS dense on RTX 4080 SUPER). HELIUM (`--fp8-attn`, iter 55) returned ~0% because attention shapes (k=512, m=2048, ~0.07 TFLOP) are too small for FP8 to amortize amax-cast overhead.

Readout shapes:
- Forward: (M=T=8192, N=V=32000, K=m=2048) ≈ 1.07 TFLOP
- Weight-grad: (M=V=32000, N=m=2048, K=T=8192) ≈ 1.07 TFLOP
- Input-grad: (M=T=8192, N=m=2048, K=V=32000) ≈ 1.07 TFLOP

Combined ~3.2 TFLOP per step. Per-call cost is ~15× HELIUM's attention GEMMs — firmly in the regime where FP8 tensor cores out-amortize the amax kernel.

| dim | HELIUM (`--fp8-attn`) | this (`--fp8-readout`) |
|---|---|---|
| target GEMM | ~0.07 TFLOP × 4 (QKVO) × L | ~1.07 TFLOP × 3 |
| call frequency | per-layer × L=24 | once per step |
| amax recompute | per-call | once per step (or EMA every-N steps) |
| amax overhead | ~5–15% | <1% |

---

## 3. Mathematical formulation

### 3.1 Tensor-level FP8 substitution

Let `q_L ∈ R^{T × m}` be post-final-LN hidden state, `W_E ∈ R^{V × m}` tied embedding, `b ∈ R^V` `lmBias`. The readout chain:

```
(F1) logits  = q_L · W_E^T + b         ∈ R^{T × V}
(F2) probs   = softmax_row(logits)      ∈ R^{T × V}
(F3) loss    = -mean_t log probs[t, y_t]
(B1) dlogits = probs - one_hot(y)       ∈ R^{T × V}
(B2) dW_E   += dlogits^T · q_L          ∈ R^{V × m}
(B3) dq_L    = dlogits · W_E            ∈ R^{T × m}
```

| tensor | shape | current | proposed | scale |
|---|---|---|---|---|
| `q_L` (F1, B2 input) | T×m | BF16 | **E4M3** | 448 / amax_qL |
| `W_E^T` (F1, B3 input) | m×V | BF16 | **E4M3** | 448 / amax_WE; recompute post-Adam |
| `logits` (F1 out) | T×V | BF16 storage | **E4M3 storage** | per-tensor scale |
| `probs` (F2 out) | T×V | BF16 storage | **E4M3 storage** | scale 448 (probs ≤ 1) |
| `dlogits` (B1 out) | T×V | BF16 storage | **E5M2 storage** | wider exponent for tail |
| `dq_L` (B3 out) | T×m | FP32 | **FP32** (accum from FP8 inputs) | — |
| `dW_E` accum | V×m | FP32 | **FP32** (accum from FP8 inputs) | — |

Untouched: softmax row-max subtraction (FP32 in registers), sum-of-exps (FP32 accum), cross-entropy reduction (FP32).

### 3.2 E4M3 forward / E5M2 backward rationale

E4M3 (±448, 3-bit mantissa, ~12% precision per binade) covers post-LN `q_L` (unit-variance), `W_E` (similar magnitude), and pre-softmax logits (typical span ±20). After per-row max subtraction the relevant range is [−15, 0]; E4M3's 3-bit mantissa is sufficient when softmax is dominated by the top tokens.

E5M2 (±57344, 2-bit mantissa) on `dlogits`: the natural range is [−1, +1] (since dlogits = probs − one_hot), so range alone doesn't demand E5M2. The reason is **the tail**: `dq_L = dlogits · W_E` is a sum over V=32000 entries; most are |dlogits| < 1e−3. E4M3's smallest normal is 2^−6 ≈ 0.016 — would flush ~80% of vocab entries to zero and bias `dq_L`. E5M2's smallest normal is 2^−14 ≈ 6.1e−5, preserving the small entries.

This is hypothesis Q1 (§10).

### 3.3 Softmax stability at V=32000

Standard `softmax(x) = exp(x − m) / Σ exp(x − m)`. With FP8 storage:
1. Load FP8 → dequantize to FP32 in registers → row-max in FP32 → store `m` to FP32 scratch.
2. Re-load FP8 → dequantize → `exp(x_i − m)` in FP32 → sum `S` in FP32.
3. Compute `p_i = exp(x_i − m) / S` in FP32 → store to E4M3.

FP8 quantization noise on `(x_i − m)` ≤ 2^−3 translates to a multiplicative `exp` error of ~e^0.125 ≈ 1.13 per entry. The sum `S` is dominated by the top ~10 tokens, so mid-rank contribution is bounded — **for peaky distributions**. Early in training (steps <500) the distribution is flat and every entry near `m` matters.

This is hypothesis Q2 (§10).

---

## 4. Implementation outline

### 4.1 New cuBLASLt wrappers (extend `gpu_blas_fp8.h`)

Existing scaffolding has `sgemm_rowmajor_fp8_e4m3` and `sgemm_rowmajor_fp8_e4m3_bf16` (FP32-out). Need three new wrappers:

```cpp
// F1: BF16 q_L · BF16 W_E^T → E4M3 logits
bool sgemm_rowmajor_abt_bf16_fp8_e4m3_out(int M, int N, int K, float alpha,
    const unsigned short* A_bf, int lda, const unsigned short* B_bf, int ldb,
    float beta, unsigned char* C_e4m3, int ldc, const float* d_scaleC);

// B2: E5M2 dlogits^T · BF16 q_L → FP32 dW_E
bool sgemm_rowmajor_atb_fp8_e5m2_bf16_f32(int M, int N, int K, float alpha,
    const unsigned char* A_e5m2, int lda, const unsigned short* B_bf, int ldb,
    float beta, float* C, int ldc, const float* d_scaleA);

// B3: E5M2 dlogits · BF16 W_E → FP32 dq_L
bool sgemm_rowmajor_fp8_e5m2_bf16_f32(int M, int N, int K, float alpha,
    const unsigned char* A_e5m2, int lda, const unsigned short* B_bf, int ldb,
    float beta, float* C, int ldc, const float* d_scaleA);
```

All use `CUBLAS_COMPUTE_32F`. Output scale applied internally by cuBLASLt.

### 4.2 New CUDA kernels (extend `gpu_kernels.{h,cu}`)

```cpp
bool softmax_forward_fp8_e4m3(const unsigned char* logits, const float* d_logits_scale,
    int rows, int cols, unsigned char* probs, const float* d_probs_scale,
    float* inv_sums);

bool softmax_cross_entropy_bwd_fp8(const unsigned char* probs, const float* d_probs_scale,
    const int* targets, int rows, int cols,
    unsigned char* dlogits, const float* d_dlogits_scale);

bool cross_entropy_nll_loss_fp8(const unsigned char* probs, const float* d_probs_scale,
    const int* targets, int T, int V, int padToken,
    float* loss_sum, int* valid_count);

bool argmax_count_matches_fp8(const unsigned char* probs, const float* d_probs_scale,
    const int* targets, int T, int V, int padToken,
    int* correct_count, int* valid_count);

bool fp8_calibrate_amax_e4m3_bf16(const unsigned short* d_x, size_t n, float* d_scale);
bool fp8_calibrate_amax_e5m2_f32(const float* d_x, size_t n, float* d_scale);
```

The softmax-fwd kernel re-normalizes in FP32 register before quantizing to E4M3; output scale fixed at 448 (since probs ≤ 1).

### 4.3 Trainer flag

`--fp8-readout` (mutually exclusive with `--bf16-logits-storage`; implies `--bf16-logits`). New enum slot in `training_config.h`:

```cpp
enum LogitsStorageDType {
    LOGITS_STORE_FP32 = 0,
    LOGITS_STORE_BF16 = 1,   // --bf16-logits-storage (current default)
    LOGITS_STORE_FP8  = 2    // --fp8-readout
};
```

Call sites: `sgd_transformer.cpp:7904-7924` (forward readout) and the backward block ~step 8200. Inference path (`transformer_infer.cpp:1714-1732`) stays BF16 (T=1 doesn't benefit).

### 4.4 VRAM accounting at T=8192, V=32000

| tensor (T×V) | FP32 | BF16 | FP8 |
|---|---:|---:|---:|
| logits | 1.05 GB | 0.49 GB | **0.25 GB** |
| probs | 1.05 GB | 0.49 GB | **0.25 GB** |
| dlogits | 1.05 GB | 0.49 GB | **0.25 GB** |
| total | 3.15 GB | 1.47 GB | **0.74 GB** |

**Net saving vs iter61 BF16**: −0.73 GB.

### 4.5 cuBLASLt API

```cpp
cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, M, K, lda);
cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, K, N, ldb);
cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_8F_E4M3, M, N, ldc);
cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &d_scaleC, ...);
```

Compute type FP32; cuBLASLt internally maps BF16 → FP8 tensor cores. Ada (sm_8.9) supports FP8-out. Whether the algo table is tuned for (8192, 32000, 2048) is an open question — priors from iter 52, 59 (BF16 algo overrides NULL on similar shapes) suggest default may be optimal or there may be no good algo at all.

---

## 5. Gate-0 criteria

Run against iter61 flagship (47,271 tok/s, val NLL 4.0771):

| id | criterion | bar |
|---|---|---|
| C0a | tok/s win at iso-config | ≥ +3% (relaxed-bar policy, iter 60) |
| C0b | val NLL @ step 200 | ≤ 4.0871 (≤ +0.020 nat) |
| C0c | val NLL @ step 1000 | ≤ 4.0871 (slow-drift trap) |
| C0d | VRAM at iter-bench | ≤ iter61 (prefer −0.5 GB) |
| C0e | per-row \|Σp − 1\| | < 1e-5 |
| C0f | argmax top-1 accuracy @ step 200 | within ±0.5% of iter61 |

C0a fail + others pass = FAIL-below-bar; retroactive ship eligible per iter 60.
C0b or C0c fail = paradigm invalidated; revert.

---

## 6. Risks and falsification triggers

**R1. Early-training softmax collapse.** Pre-softmax logits not yet peaky; FP8 quantizes ~256 distinct values; effective rank of `exp(logits − m)` collapses. **Trigger**: val NLL @ step 100 > 4.5. Revert.

**R2. dlogits E5M2 underflow biases `dq_L`.** 70% of vocab entries have |dlogits| < 1e−4; E5M2 smallest normal 6.1e−5; near-flush bias. **Trigger**: grad-norm @ step 50 outside [0.8, 1.2]× BF16 baseline at same step.

**R3. Microbatch accumulation rounding.** `--accum 8`: FP32 accumulator with FP8 input per microbatch. Rounding compounds. **Trigger**: val-NLL gap grows monotonically over steps 50 → 100 → 200.

**R4. cuBLASLt FP8 algo undertuned on (8192, 32000, 2048) sm_8.9.** Priors from iter 52, 59 BF16 algo overrides NULL. **Trigger**: forward FP8 GEMM ≥ BF16 baseline wall-clock at same shape. Pivot to manual algo override (iter 65) but likely also NULL.

**R5. amax recompute dominates.** 3 amax reductions over T·m = 16.8M entries × per-step cost ~0.1 ms. **Mitigation**: every-10-step EMA refresh. **Trigger**: amax-kernel time ≥ 5% of step time in nsys.

**R6. Slice fragmentation caps win.** Of 18% readout share, GEMMs ~12% and softmax/CE/argmax ~5%. FP8 only helps GEMM portion; realistic ceiling ~+9%.

**R7. cuBLASLt FP8 may require N power-of-2.** V=32000 even but not 2^k. Padding to 32768 costs 6 MB + softmax masking. Manageable.

---

## 7. Iter budget — 4-iter plan

| iter | scope | per-iter gate | revert |
|---:|---|---|---|
| 62 | `--fp8-readout-fwd`: F1 + F2 + F3 only; backward stays BF16 (FP8 probs → BF16 cast for legacy CE-bwd) | C0a ≥ +1.5% (loose, fwd-only); C0b ≤ +0.02; C0e < 1e-5 | NLL drift ≥ +0.05 @ step 200 |
| 63 | Full `--fp8-readout`: B1 writes E5M2; B2/B3 FP8 GEMM wrappers | C0a ≥ +3%; C0b ≤ +0.02; C0d ≤ iter61 VRAM | grad-norm out of [0.8, 1.2]× @ step 50 |
| 64 | amax EMA every-N-steps (tune N ∈ {5, 10, 20}) | +1% tok/s vs iter63 at NLL parity | NLL drift from staleness ≥ +0.01 |
| 65 | cuBLASLt algo sweep on FP8 readout shape (8-16 ALGOn_TENSOR_OP candidates) | +1% over iter64 | NULL (priors say likely) |

**iter 62 fail** = paradigm invalidated. Pivot to BF16-residual-p or MoE FFN.
**iter 62 pass, iter 63 fail** = retain fwd-only for inference/eval; pivot for training.
**iter 64 / 65 NULL** = ship iter63 cumulative if ≥ +3%; else FAIL-below-bar.

---

## 8. Recoverable as a limiting case

`LOGITS_STORE_BF16` (existing default) dispatches the iter61 path bit-exact. `LOGITS_STORE_FP32` dispatches the pre-iter10 path. FP8 readout is a **strict superset** — off-by-default flag, no replacement of existing infrastructure, no risk to flagship. Falsification doesn't disturb the current production stack.

---

## 9. Comparison to existing methods

| method | what they do | what we do differently |
|---|---|---|
| NVIDIA Transformer Engine | Per-layer amax + delayed scaling; full FP8 attention + MLP + embedding | Readout-only. Amax once-per-step or EMA. No FP8 in attention (HELIUM NULL) or FFN. |
| MS-AMP O3 | E4M3 weights, E5M2 grads, FP8 master | Retain BF16 weights, FP32 master. Only readout activations / errors are FP8. |
| FP8-LM (Peng et al, 2023) | Train 175B FP8 throughout; per-tensor scale tracker | One-shot amax per step. No global tracker, no non-readout FP8. |
| GLM-FP8 (2024) | FP8 KV cache + FP8 forward MLP, BF16 backward | Different slice — KV cache and MLP untouched. |

**Novel**: applying FP8 to only the LM-readout slice on **Ada (sm_8.9) consumer GPU**, where HELIUM was NULL on attention. The shape-specific test (1.1 TFLOP, N=32000) is the contribution. **Not novel**: E4M3 / E5M2 split, amax scaling.

---

## 10. Open empirical questions

**Q1. Does E5M2 underflow on dlogits bias `dq_L`?**
At step 50 of `--fp8-readout`, compute `||dq_L_fp8 − dq_L_bf16|| / ||dq_L_bf16||` on a held-out batch. Target <1%. If >5%, E5M2 in backward is unviable — fall back to fwd-only mode.

**Q2. Does softmax row-mass drift under FP8 storage?**
At step 200, sample 100 random probs rows; dequantize, compute `|Σp − 1|`. Target <1e-5 (C0e). If ~1e-3, the row normalizer is precision-limited and the softmax kernel must FP32-renormalize on read.

**Q3. Does cuBLASLt FP8 dispatch to a tuned algo on (8192, 32000, 2048) sm_8.9?**
nsys diff FP8 vs BF16 forward GEMM time. Target ≤ 0.6×. If ≥ 0.85×, algo table is incomplete — try manual override (iter 65), but iter 52/59 priors say algo overrides are NULL on similar BF16 shapes.

**Q4. How much do non-GEMM kernels (softmax-fwd/bwd, argmax, scale) gain from FP8 storage?**
Hypothesis: memory-bandwidth-bound; halving in-HBM tensor sizes should give ~1.7× kernel speedup. Would push realistic envelope to +4% to +7% overall. Method: per-kernel nsys time, BF16 baseline vs FP8.

---

## 11. Honest assessment

Coming after 5-of-5 paradigm falsifications (#260 cellular sheaves, IGAA, HMTA, MEDAL — all NEGATIVE at C1), this is the first non-architectural paradigm of the arc, which modestly raises the prior.

**Strengths**:
- Shape-specific theoretical edge: 1.1 TFLOP per readout GEMM is in Ada's FP8 sweet spot (HELIUM was below it).
- Existing `gpu_blas_fp8.h` scaffolding reusable; new wrappers extend rather than replace.
- Off-by-default flag; no flagship risk; retroactive ship eligible per iter 60.
- Sharp falsification: NLL drift in 200 steps either survives or doesn't.

**Weaknesses worth naming**:
- Headline win modest by construction. Even perfect 2× GEMM gives ~+6% overall — does not reach 10×.
- Softmax over V=32000 with FP8 row-storage has no public LLM precedent; most FP8 work keeps readout in BF16+.
- cuBLASLt FP8 algo table on sm_8.9 readout shape is unverified; could be NULL like HELIUM at attention.
- All ≥2% kernel slices already attacked; FP8 is the only remaining dtype lever for readout.

**Probability estimates** (priors-weighted, single-iter): +3% ship ≈ 35%; NULL ≈ 35%; NLL fail ≈ 30%. Joint probability of all 4 iters landing ≈ <15%.

Offered as the best available among readout-attacking options, not the path to 10×. The brief's 10× headline is empirically out of reach; this paradigm contributes one more incremental step.
