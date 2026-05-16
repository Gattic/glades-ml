// CHIRON reversible-flow transformer GPU primitives.
//
// Host wrappers for CUDA kernels that implement the CHIRON forward / inverse
// block components.  Mirrors the CPU header `transformer_chiron_ops.h` (which
// remains the reference implementation).
//
// When GLADES_HAVE_CUDA is not defined, the wrappers degrade to inline no-ops
// that return false — callers can unconditionally compile against this header.
//
// See research/CHIRON_framework.md and research/CHIRON_PROGRESS.md for the
// design and empirical findings.

#pragma once

#include <cstddef>
#include <stdint.h>
#include "gpu_device.h"  // cudaStream_t (iter 8)

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// ---------------------------------------------------------------------------
// Symplectic shears — element-wise in-place update of p (or q) by u.
// ---------------------------------------------------------------------------

// p[i] += u[i] for i in [0, n). Used by Shear^p forward.
bool chiron_shear_add(float* p, const float* u, int n);

// p[i] -= u[i] for i in [0, n). Used by Shear^p inverse.
bool chiron_shear_sub(float* p, const float* u, int n);

// ---------------------------------------------------------------------------
// SCFA stream-op fused kernels (ralph-loop iter 5, 2026-05-14).
//
// Replace memcpy_d2d + axpy pairs in the SCFA forward/backward chain with
// single-pass element-wise kernels.  Math is bit-identical to the unfused
// FP32 ops; the only gain is the saved memory round-trip for the
// eliminated intermediate buffer write.  Used only when the trainer's
// --scfa-fuse-streams flag is on (default off).
// ---------------------------------------------------------------------------

// c[i] = a[i] - b[i] for i in [0, n).  Replaces:
//   memcpy_d2d(c, a, n*sizeof(float));
//   axpy(-1.0f, b, c, n);
// iter 8 (2026-05-14): optional `stream` parameter for multi-stream branch
// parallelism (--scfa-parallel-branches).  Default = 0 means use
// computeStream() (backwards compatible).
bool chiron_scfa_sub(float* c, const float* a, const float* b, int n,
                     cudaStream_t stream = 0);

// p[i] += alpha * (a[i] + b[i]) for i in [0, n).  Replaces:
//   axpy(1.0f, b, a, n);          // a += b (mutates a)
//   axpy(alpha, a, p, n);         // p += alpha * a
// Note: this kernel does NOT mutate a, so the caller's a buffer is
// preserved.  The original chain mutated a; if the caller relied on the
// mutated value of a afterwards, the fused variant must not be used.
bool chiron_scfa_axpy2(float* p, float alpha,
                       const float* a, const float* b, int n,
                       cudaStream_t stream = 0);

// c[i] = alpha * a[i] for i in [0, n).  Replaces:
//   memcpy_d2d(c, a, n*sizeof(float));
//   scale_array(c, alpha, n);
bool chiron_scfa_scaled_copy(float* c, float alpha, const float* a, int n,
                              cudaStream_t stream = 0);

// iter 63 (Arc 2, BF16 residual-p — 2026-05-16): BF16-p storage variants.
// Read BF16 p, accumulate FP32 in registers, RN-round on write.  Used when
// --bf16-residual-p routes the residual stream to BF16 storage to halve
// HBM traffic on the SCFA element-wise stack (Amdahl ceiling +4.8% wall at
// iter-bench T=8192 L=12).  Stochastic-rounding variant lands in iter 64
// once the framework is validated.
bool chiron_scfa_axpy2_bf16p_rn(unsigned short* p_bf, float alpha,
                                 const float* a, const float* b, int n,
                                 cudaStream_t stream = 0);
bool chiron_axpy_bf16p_rn(unsigned short* p_bf, float alpha,
                           const float* x, int n,
                           cudaStream_t stream = 0);
bool chiron_scfa_scaled_copy_bf16p_rn(unsigned short* c_bf, float alpha,
                                       const float* a, int n,
                                       cudaStream_t stream = 0);
// q[i] += alpha * bf16_to_fp32(p_bf[i]).  Used at "q += p" sites when p is
// BF16 storage but q stays FP32.  FP32 result write.
bool chiron_bf16_to_fp32_axpy(float* q, float alpha,
                               const unsigned short* p_bf, int n,
                               cudaStream_t stream = 0);

// iter 64 (Arc 2): Stochastic-rounding variants.  Same FP32-internal accum
// as RN; final BF16 encode uses xorshift-mixed hash of (idx, step, seed)
// for mean-zero rounding.  Caller passes srBaseSeed (W.bf16WeightsSeed) +
// srStepIdx (training step counter).
bool chiron_scfa_axpy2_bf16p_sr(unsigned short* p_bf, float alpha,
                                 const float* a, const float* b, int n,
                                 unsigned int srBaseSeed,
                                 unsigned int srStepIdx,
                                 cudaStream_t stream = 0);
bool chiron_axpy_bf16p_sr(unsigned short* p_bf, float alpha,
                           const float* x, int n,
                           unsigned int srBaseSeed,
                           unsigned int srStepIdx,
                           cudaStream_t stream = 0);
bool chiron_scfa_scaled_copy_bf16p_sr(unsigned short* c_bf, float alpha,
                                       const float* a, int n,
                                       unsigned int srBaseSeed,
                                       unsigned int srStepIdx,
                                       cudaStream_t stream = 0);
// Reln forward reading BF16 p (decode inline), writes FP32 q_out + stats.
bool chiron_reln_forward_rows_bf16p(const unsigned short* p_bf_in,
                                     float* q_out, float* stats,
                                     const float* gamma, const float* beta,
                                     int T, int m, float eps);
// Reln inverse reading FP32 q + stats, SR-writes BF16 p.
bool chiron_reln_inverse_rows_bf16p_sr(const float* q_out, const float* stats,
                                        const float* gamma, const float* beta,
                                        int T, int m,
                                        unsigned short* p_bf_out,
                                        unsigned int srBaseSeed,
                                        unsigned int srStepIdx);

// ---------------------------------------------------------------------------
// Reversible LayerNorm (ReLN) with external stats buffer.
// ---------------------------------------------------------------------------

// Forward: given q_in[T, m] and affine parameters gamma[m], beta[m], writes:
//   - q_out[T, m]  = gamma * (q_in - mu) / sqrt(var + eps) + beta  per row
//   - stats[T, 2]  = { mu, log(sqrt(var + eps)) }                   per row
// Each row has mean mu and std sigma; stats records both so the inverse is
// deterministic.
bool chiron_reln_forward(const float* q_in, float* q_out, float* stats,
                          const float* gamma, const float* beta,
                          int T, int m, float eps);

// ralph-loop iter 9 (2026-05-14): fused reln-forward + axpy-into-q for
// CHIRON's per-layer-fuse path.  Replaces:
//   chiron_reln_forward(p, p_norm, stats, gamma, beta, T, m, eps);
//   axpy(alpha, p_norm, q, T*m);
// with a single kernel that computes normalized p AND accumulates it into
// q[i] += alpha · (gamma[i]·(p[i]-mu)/sigma + beta[i]).  Eliminates the
// p_norm scratch round-trip (~128 MB per call at T=8192 m=2048).  Math is
// bit-identical FP32 modulo sub-ULP FMA-ordering.  stats[T, 2] is written
// in the same { mu, log(sigma) } format as chiron_reln_forward.
bool chiron_reln_axpy_into_q(const float* p, float* q, float* stats,
                              const float* gamma, const float* beta,
                              float alpha, int T, int m, float eps);

// Inverse: given q_out and the stats produced by the forward, recovers q_in.
//   q_in[i] = sigma * (q_out[i] - beta[i]) / gamma[i] + mu
// where sigma = exp(stats[t, 1]) and mu = stats[t, 0].
bool chiron_reln_inverse(const float* q_out, float* q_in, const float* stats,
                          const float* gamma, const float* beta,
                          int T, int m);

// Reversible LayerNorm backward.  Given the output-space gradient dq_out
// [T, m], the pre-ReLN input q_in [T, m] (typically RECONSTRUCTED via
// reln_inverse during the CHIRON backward pass), the affine parameters
// gamma/beta [m], and the stats [T, 2] stored at forward time, this
// computes:
//   dq_in [T, m]   : gradient of the loss with respect to q_in.
//   dgamma [m]     : ACCUMULATED gradient w.r.t. gamma (pre-initialize).
//   dbeta  [m]     : ACCUMULATED gradient w.r.t. beta.
//
// Math: identical to the standard LayerNorm backward (our ReLN forward
// is numerically identical to LayerNorm forward — the only novelty is
// that stats are stored externally and the map is framed as a reversible
// shear in p-coordinates).  This is a thin wrapper that converts stats
// from (mu, log_sigma) to (mean, invStd) format and calls the existing
// layernorm_backward kernel.
//
// scratch_stats_split: caller-owned buffer of size 2*T floats, used as
// scratch for the (mean, invStd) tensors.
bool chiron_reln_backward(const float* dq_out, const float* q_in,
                           const float* gamma, const float* stats,
                           int T, int m,
                           float* dq_in, float* dgamma, float* dbeta,
                           float* scratch_stats_split);

// ---------------------------------------------------------------------------
// Sketch primitives — per-token local sketch (framework amendment §11a,
// mitigation 1).
//
// State is conceptually per-token x_t ∈ R^Ntok (with Ntok = 2m for CHIRON's
// paired q/p state).  S ∈ R^{r × Ntok} is the per-layer sketch matrix,
// SHARED across the T tokens of a single layer.
//
// The operations are expressed as batched matvecs and use cuBLAS GEMM
// internally.
// ---------------------------------------------------------------------------

// Batched sketch projection: Z[t, k] = Σ_i S[k, i] * X[t, i].
// Equivalent to Z = X · S^T where X is [T, Ntok] and S is [r, Ntok],
// producing Z [T, r].
bool chiron_sketch_project(const float* X, const float* S,
                            int T, int Ntok, int r, float* Z);

// Batched sketch lift-add: X[t, i] += (1/r) · Σ_k S[k, i] * R[t, k].
// Equivalent to X += (1/r) · R · S where R is [T, r] and S is [r, Ntok].
bool chiron_sketch_lift_add(float* X, const float* R, const float* S,
                             int T, int Ntok, int r);

// ---------------------------------------------------------------------------
// Symplectic attention shear (framework §3.2) — composition wrapper.
//
// Computes  p += Wo^T · Attention(Q=q·Wq, K=q·Wk, V=q·Wv)  on GPU, reusing
// the existing BF16/FP32 flash-attention kernels and cuBLAS GEMMs. q is
// unchanged; only p is modified.  This is the symplectic shear that makes
// the forward map (q, p) → (q, p + Y(q)) a unit lower-triangular, exactly
// invertible bijection.  The inverse is simply p -= Y(q) computed with
// the same kernel sequence.
//
// Weight layouts (row-major):
//   Wq, Wk, Wv : [m, dH]   — input proj (dH = nHeads · dHead for multihead)
//   Wo         : [dH, m]   — output proj back to the p branch
//
// Scratch buffers (caller-owned) live here so the caller controls memory
// re-use across layers:
//   scratch_Q, scratch_K, scratch_V : each [T, dH]
//   scratch_O                       : [T, dH]
//
// nHeads, dHead: single-head is nHeads=1, dHead=dH. Multihead support
// follows existing transformer conventions (dModel = nHeads · dHead in
// the attention path).
//
// `invert`: when false, compute p += Y(q). When true, p -= Y(q).
// The inverse is bit-equivalent because the same Y(q) is recomputed from
// q (which is unchanged by the shear).
bool chiron_attention_shear(const float* q, float* p,
                             const float* Wq, const float* Wk,
                             const float* Wv, const float* Wo,
                             int T, int m, int nHeads, int nKVHeads, int dHead,
                             bool causal, bool invert,
                             float* scratch_Q, float* scratch_K,
                             float* scratch_V, float* scratch_O);

// Backward through the symplectic attention shear.  Given the upstream
// p-gradient dp_new and the reconstructed q (via the block inverse),
// computes:
//   dq        += dL/dq contribution from the three Q/K/V projections
//                and the two p·Wo-like paths.  Shear-additive in p
//                means dL/dp = dL/dp_new (no change).
//   dWq, dWk, dWv, dWo: accumulated weight gradients (+=).
//
// Internally: recomputes Q/K/V/O from q, then calls
// flash_attention_multihead_backward for dQ/dK/dV, plus the output
// projection backward via cuBLAS.
//
// Scratch (caller-owned):
//   sQ, sK, sV, sO             : each [T, dModel] (dModelKV for K/V)
//   sdO, sdQ, sdK, sdV         : each same size, for the backward
//                                intermediates.
bool chiron_attention_shear_backward(
    const float* q, const float* dp_new,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    int T, int m, int nHeads, int nKVHeads, int dHead,
    bool causal,
    float* dq,
    float* dWq, float* dWk, float* dWv, float* dWo,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV);

// Tensor-core-backed shear forward.  Identical math to chiron_attention_shear
// but routes the attention core through flash_attention_cublas_tiled (TF32
// tensor cores).  Typical 5-10× wall-clock improvement at T≥512 on Ampere/Ada.
// Extra scratch: scratch_S [nHeads, T, T], caller-owned.
// Constraint: nHeads == nKVHeads (no GQA — tiled kernel doesn't broadcast).
bool chiron_attention_shear_tiled(const float* q, float* p,
                                    const float* Wq, const float* Wk,
                                    const float* Wv, const float* Wo,
                                    int T, int m, int nHeads, int dHead,
                                    bool causal, bool invert,
                                    float* scratch_Q, float* scratch_K,
                                    float* scratch_V, float* scratch_O,
                                    float* scratch_S);

// BF16-tensor-core variant of chiron_attention_shear_tiled.  Q/K/V/O
// projections stay FP32 (only 4 GEMMs per layer — modest cost); the attention
// core (QK^T, softmax, P·V) runs with BF16 inputs via
// flash_attention_cublas_tiled_bf16.  ~2× over TF32-tiled on Ampere/Ada.
// Extra scratch: BF16 staging for Q, K, V (each [T, dModel]) and P
// ([nHeads, T, T]).
bool chiron_attention_shear_bf16_tiled(const float* q, float* p,
                                         const float* Wq, const float* Wk,
                                         const float* Wv, const float* Wo,
                                         int T, int m, int nHeads, int dHead,
                                         bool causal, bool invert,
                                         float* scratch_Q, float* scratch_K,
                                         float* scratch_V, float* scratch_O,
                                         float* scratch_S,
                                         unsigned short* scratch_Qbf16,
                                         unsigned short* scratch_Kbf16,
                                         unsigned short* scratch_Vbf16,
                                         unsigned short* scratch_Pbf16);

// BF16-weight variant of chiron_attention_shear_bf16_tiled.  Takes BF16
// weight pointers directly — no per-layer weight-cast scratch needed.
// Q/K/V/O projections run through sgemm_rowmajor_bf16 (BF16×BF16→FP32 via
// BF16 tensor cores, ~2× TF32-TC throughput on Ampere/Ada).  Intended for
// use with --bf16-weights on the trainer; eliminates 4 weight-cast kernels
// per layer (~200 MB of HBM traffic saved per layer at 2 B scale) and
// doubles projection GEMM throughput.
//
// Extra scratch (caller-owned):
//   scratch_qbf   [T, m]          BF16 cast of q (one cast per layer)
//   scratch_Obf   [T, dModel]     BF16 cast of attention output for Wo proj
//   (plus the same scratch_Qbf16/Kbf16/Vbf16/Pbf16 as _bf16_tiled)
bool chiron_attention_shear_bf16w_tiled(const float* q, float* p,
                                          const unsigned short* Wq_bf,
                                          const unsigned short* Wk_bf,
                                          const unsigned short* Wv_bf,
                                          const unsigned short* Wo_bf,
                                          int T, int m, int nHeads, int dHead,
                                          bool causal, bool invert,
                                          unsigned short* scratch_qbf,
                                          unsigned short* scratch_Obf,
                                          float* scratch_Q, float* scratch_K,
                                          float* scratch_V, float* scratch_O,
                                          float* scratch_S,
                                          unsigned short* scratch_Qbf16,
                                          unsigned short* scratch_Kbf16,
                                          unsigned short* scratch_Vbf16,
                                          unsigned short* scratch_Pbf16);

// FP8 (E4M3) projection variant of `chiron_attention_shear_bf16w_tiled`
// (paradigm #50 HELIUM).  Projections Q/K/V/O run via
// `sgemm_rowmajor_fp8_e4m3_bf16` (BF16 weights cast to E4M3 internally,
// FP8 tensor-core matmul, FP32 accumulate); the attention core stays in
// BF16-TC.  Backward is unchanged — use the existing bf16w backward.
//
// Per-tensor scales are computed inside the function from amax of the
// inputs (q, Wq_bf, Wk_bf, Wv_bf, scratch_O, Wo_bf).  Caller provides
// 6 FP32 device scalars as scratch (scale_q, scale_Wq, scale_Wk,
// scale_Wv, scale_O, scale_Wo).
//
// At L=24 m=2048 T=512 we expect ~1.5-2× over BF16-TC for the projection
// GEMMs since Ada has 2× FP8 throughput vs BF16; net step speedup likely
// 10-20% since the inner attention (still BF16) doesn't change.
bool chiron_attention_shear_fp8w_tiled(const float* q, float* p,
                                         const unsigned short* Wq_bf,
                                         const unsigned short* Wk_bf,
                                         const unsigned short* Wv_bf,
                                         const unsigned short* Wo_bf,
                                         int T, int m, int nHeads, int dHead,
                                         bool causal, bool invert,
                                         unsigned short* scratch_qbf,
                                         unsigned short* scratch_Obf,
                                         float* scratch_Q, float* scratch_K,
                                         float* scratch_V, float* scratch_O,
                                         float* scratch_S,
                                         unsigned short* scratch_Qbf16,
                                         unsigned short* scratch_Kbf16,
                                         unsigned short* scratch_Vbf16,
                                         unsigned short* scratch_Pbf16,
                                         float* d_scale_q,
                                         float* d_scale_Wq, float* d_scale_Wk,
                                         float* d_scale_Wv, float* d_scale_Wo,
                                         float* d_scale_O);

// BF16-weight backward counterpart.  Takes BF16 weight pointers directly
// and uses BF16-TC GEMMs throughout the projection, dO, and dq-projection
// paths.  Weight-grad GEMMs (dWq += q^T · sdQ etc.) stay FP32 — caller may
// combine with --bf16-grads to accumulate into BF16 storage externally.
//
// Extra scratch: scratch_qbf [T, m] (reused for q cast and dp_new cast)
// and scratch_sdbf [T, dModel] (rotates through sdQ/sdK/sdV BF16 casts).
bool chiron_attention_shear_backward_bf16w_tiled(
    const float* q, const float* dp_new,
    const unsigned short* Wq_bf, const unsigned short* Wk_bf,
    const unsigned short* Wv_bf, const unsigned short* Wo_bf,
    int T, int m, int nHeads, int dHead,
    bool causal,
    float* dq,
    float* dWq, float* dWk, float* dWv, float* dWo,
    unsigned short* scratch_qbf,
    unsigned short* scratch_sdbf,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV,
    float* scratch_P, float* scratch_dP);

// iter 61 (2026-05-16): BF16-grad variant of _bf16w_tiled.  Identical math
// + identical scratch layout, but the 4 dW weight-grad GEMMs use
// sgemm_rowmajor_atb_bf16_dst_bf16 (cuBLAS gemmEx with D=BF16) to write
// directly into BF16 persistent grad buffers with beta=1.  Eliminates the
// downstream bf16_accum_axpy commit kernel (3.1% of GPU time at iter60).
// Caller must pre-zero the BF16 dW buffers at the start of the
// accumulation window (same protocol as bf16_accum_axpy path).
bool chiron_attention_shear_backward_bf16w_bf16g_tiled(
    const float* q, const float* dp_new,
    const unsigned short* Wq_bf, const unsigned short* Wk_bf,
    const unsigned short* Wv_bf, const unsigned short* Wo_bf,
    int T, int m, int nHeads, int dHead,
    bool causal,
    float* dq,
    unsigned short* dWq_bf, unsigned short* dWk_bf,
    unsigned short* dWv_bf, unsigned short* dWo_bf,
    unsigned short* scratch_qbf,
    unsigned short* scratch_sdbf,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV,
    float* scratch_P, float* scratch_dP);

// Tensor-core-backed shear backward.  Replaces flash_attention_multihead_backward
// with flash_attention_backward_cublas_tiled.  Extra scratch: scratch_P and
// scratch_dP, each [nHeads, T, T], caller-owned.
bool chiron_attention_shear_backward_tiled(
    const float* q, const float* dp_new,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    int T, int m, int nHeads, int dHead,
    bool causal,
    float* dq,
    float* dWq, float* dWk, float* dWv, float* dWo,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV,
    float* scratch_P, float* scratch_dP);

// ---------------------------------------------------------------------------
// cuBLAS-tiled flash attention — tensor-core-backed drop-in alternative.
// ---------------------------------------------------------------------------
//
// Decomposes attention into three cuBLAS / custom kernel calls:
//   S[nH, T, T] = Q · K^T   (cuBLAS sgemm_batched_strided_abt, TF32 TC)
//   P[nH, T, T] = softmax_row(S, causal mask, 1/sqrt(dHead) scaled)
//   O[nH, T, dHead] = P · V  (cuBLAS sgemm_batched_strided, TF32 TC)
//
// The existing `flash_attention_multihead_forward` avoids materialising
// S/P via an online-softmax streaming kernel; that keeps HBM usage down
// to O(T·dHead) but runs at 0.15 TFLOP/s (100x below cuBLAS peak)
// because the kernel uses no tensor cores.  This variant trades
// O(nH·T²) scratch memory for cuBLAS-backed tensor-core throughput —
// the best option for T ≤ 2048 where nH·T²·4 stays under a few hundred
// MB.
//
// Scratch: scratch_S [nH, T, T], FP32, caller-owned.
//
// Constraints: nHeads must equal nKVHeads (no GQA in this version).
bool flash_attention_cublas_tiled(const float* Q, const float* K, const float* V,
                                    int T, int nHeads, int dHead, int dModel,
                                    bool causal,
                                    float* O, float* scratch_S);

// BF16-tensor-core variant.  Casts Q/K/V to BF16 once, uses BF16 batched
// GEMMs (CUBLAS_COMPUTE_32F_FAST_16BF).  Throughput ~2x over the FP32
// variant on Ampere/Ada/Hopper (TF32 ~25 TFLOP/s vs BF16 ~52 TFLOP/s on
// RTX 4080 SUPER).
//
// Scratch:
//   scratch_S       [nH, T, T]   FP32 attention scores
//   scratch_Qbf16   [T, dModel]  BF16 Q cast
//   scratch_Kbf16   [T, dModel]  BF16 K cast
//   scratch_Vbf16   [T, dModel]  BF16 V cast
//   scratch_Pbf16   [nH, T, T]   BF16 P cast (for the PV GEMM)
bool flash_attention_cublas_tiled_bf16(
    const float* Q, const float* K, const float* V,
    int T, int nHeads, int dHead, int dModel,
    bool causal,
    float* O,
    float* scratch_S,
    unsigned short* scratch_Qbf16, unsigned short* scratch_Kbf16,
    unsigned short* scratch_Vbf16, unsigned short* scratch_Pbf16);

// cuBLAS-tiled backward counterpart.  Given Q, K, V, O (unused, kept for
// API symmetry), and the upstream gradient dO, produces dQ, dK, dV.
//
// Math:
//   P     = softmax_row(causal_mask((1/sqrt(dH)) Q K^T))   (recomputed)
//   dV   += P^T · dO                                        (per head, batched)
//   dP    = dO · V^T                                        (per head, batched)
//   dS    = softmax_backward(P, dP)                         (with causal mask)
//   dQ    = (1/sqrt(dH)) dS · K                             (batched)
//   dK   += (1/sqrt(dH)) dS^T · Q                           (batched)
//
// Constraints: nHeads == nKVHeads (no GQA).
//
// Scratch:
//   scratch_P  [nH, T, T] — recomputed attention probs
//   scratch_dP [nH, T, T] — backward intermediate dP
//
// dV, dK are ACCUMULATED (+=). dQ is WRITTEN. The attention output O is
// not needed by the cuBLAS-tiled path (we recompute P internally) but is
// kept in the signature for drop-in compatibility with
// flash_attention_multihead_backward.
bool flash_attention_backward_cublas_tiled(
    const float* Q, const float* K, const float* V,
    const float* O, const float* dO,
    int T, int nHeads, int dHead, int dModel,
    bool causal,
    float* dQ, float* dK, float* dV,
    float* scratch_P, float* scratch_dP);

// BF16-input attention shear.  Computes Q/K/V in FP32 via cuBLAS, casts
// down to BF16 for the flash-attention core, then casts the output back.
// The existing flash_attention_multihead_forward_bf16 kernel is heavily
// optimized (see research/BF16_PLAN.md) and is 50-200x faster than the
// FP32 path at training-scale T/dHead.
//
// Extra scratch (caller-owned): BF16 staging for Q, K, V [T, dModel/Kv].
bool chiron_attention_shear_bf16(const float* q, float* p,
                                  const float* Wq, const float* Wk,
                                  const float* Wv, const float* Wo,
                                  int T, int m, int nHeads, int nKVHeads, int dHead,
                                  bool causal, bool invert,
                                  float* scratch_Q, float* scratch_K,
                                  float* scratch_V, float* scratch_O,
                                  uint16_t* scratch_Qbf, uint16_t* scratch_Kbf,
                                  uint16_t* scratch_Vbf);

// Local-window BF16 shear forward.  Uses the windowed flash kernel — each
// query attends only to ±windowSize tokens (O(T·W) compute vs O(T²)).
// Preserves CHIRON reversibility (the shear's algebraic form is unchanged;
// only Y(q)'s internal complexity changes).  Falls back to full attention
// when windowSize <= 0 or >= T.  Same extra scratches as chiron_attention_shear_bf16.
bool chiron_attention_shear_local_bf16(const float* q, float* p,
                                         const float* Wq, const float* Wk,
                                         const float* Wv, const float* Wo,
                                         int T, int m, int nHeads, int nKVHeads, int dHead,
                                         bool causal, bool invert, int windowSize,
                                         float* scratch_Q, float* scratch_K,
                                         float* scratch_V, float* scratch_O,
                                         unsigned short* scratch_Qbf,
                                         unsigned short* scratch_Kbf,
                                         unsigned short* scratch_Vbf);

// Local-window BF16 shear backward.  Mirrors chiron_attention_shear_backward_bf16
// with windowed attention bwd.
bool chiron_attention_shear_backward_local_bf16(
    const float* q, const float* dp_new,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    int T, int m, int nHeads, int nKVHeads, int dHead,
    bool causal, int windowSize,
    float* dq,
    float* dWq, float* dWk, float* dWv, float* dWo,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV,
    unsigned short* scratch_Qbf, unsigned short* scratch_Kbf, unsigned short* scratch_Vbf);

// Backward counterpart to chiron_attention_shear_bf16.  Uses flash attention
// (non-materialized) for the attention backward — no O(nH*T^2) scratch for
// the softmax probabilities.  Intended for long-context training where the
// cuBLAS-tiled variant's scratch_P + scratch_dP exceeds GPU VRAM.
//
// dV, dK are ACCUMULATED (+=).  dQ is WRITTEN by the flash backward, then
// projected back into q-space with +=.
bool chiron_attention_shear_backward_bf16(
    const float* q, const float* dp_new,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    int T, int m, int nHeads, int nKVHeads, int dHead,
    bool causal,
    float* dq,
    float* dWq, float* dWk, float* dWv, float* dWo,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV,
    uint16_t* scratch_Qbf, uint16_t* scratch_Kbf, uint16_t* scratch_Vbf);

} // namespace gpu
} // namespace glades

#else  // !GLADES_HAVE_CUDA — inline no-op stubs

namespace glades {
namespace gpu {

inline bool chiron_shear_add(float*, const float*, int) { return false; }
inline bool chiron_shear_sub(float*, const float*, int) { return false; }
inline bool chiron_reln_forward(const float*, float*, float*,
                                 const float*, const float*,
                                 int, int, float) { return false; }
inline bool chiron_reln_axpy_into_q(const float*, float*, float*,
                                     const float*, const float*,
                                     float, int, int, float) { return false; }
inline bool chiron_reln_inverse(const float*, float*, const float*,
                                 const float*, const float*,
                                 int, int) { return false; }
inline bool chiron_reln_backward(const float*, const float*,
                                  const float*, const float*,
                                  int, int,
                                  float*, float*, float*, float*) { return false; }
inline bool chiron_sketch_project(const float*, const float*, int, int, int,
                                   float*) { return false; }
inline bool chiron_sketch_lift_add(float*, const float*, const float*,
                                    int, int, int) { return false; }
inline bool chiron_attention_shear(const float*, float*,
                                    const float*, const float*, const float*,
                                    const float*,
                                    int, int, int, int, int,
                                    bool, bool,
                                    float*, float*, float*, float*) { return false; }
inline bool chiron_attention_shear_tiled(const float*, float*,
                                          const float*, const float*, const float*, const float*,
                                          int, int, int, int, bool, bool,
                                          float*, float*, float*, float*, float*) { return false; }
inline bool chiron_attention_shear_bf16_tiled(const float*, float*,
                                                const float*, const float*, const float*, const float*,
                                                int, int, int, int, bool, bool,
                                                float*, float*, float*, float*, float*,
                                                unsigned short*, unsigned short*,
                                                unsigned short*, unsigned short*) { return false; }
inline bool chiron_attention_shear_backward_tiled(
    const float*, const float*,
    const float*, const float*, const float*, const float*,
    int, int, int, int, bool,
    float*, float*, float*, float*, float*,
    float*, float*, float*, float*,
    float*, float*, float*, float*,
    float*, float*) { return false; }
inline bool flash_attention_cublas_tiled(const float*, const float*, const float*,
                                          int, int, int, int, bool,
                                          float*, float*) { return false; }
inline bool flash_attention_cublas_tiled_bf16(
    const float*, const float*, const float*,
    int, int, int, int, bool,
    float*, float*,
    unsigned short*, unsigned short*, unsigned short*, unsigned short*) { return false; }
inline bool chiron_attention_shear_bf16w_tiled(const float*, float*,
                                                 const unsigned short*, const unsigned short*,
                                                 const unsigned short*, const unsigned short*,
                                                 int, int, int, int, bool, bool,
                                                 unsigned short*, unsigned short*,
                                                 float*, float*, float*, float*, float*,
                                                 unsigned short*, unsigned short*,
                                                 unsigned short*, unsigned short*) { return false; }
inline bool chiron_attention_shear_fp8w_tiled(const float*, float*,
                                                const unsigned short*, const unsigned short*,
                                                const unsigned short*, const unsigned short*,
                                                int, int, int, int, bool, bool,
                                                unsigned short*, unsigned short*,
                                                float*, float*, float*, float*, float*,
                                                unsigned short*, unsigned short*,
                                                unsigned short*, unsigned short*,
                                                float*, float*, float*, float*, float*, float*) { return false; }
inline bool chiron_attention_shear_backward_bf16w_tiled(
    const float*, const float*,
    const unsigned short*, const unsigned short*,
    const unsigned short*, const unsigned short*,
    int, int, int, int, bool,
    float*,
    float*, float*, float*, float*,
    unsigned short*, unsigned short*,
    float*, float*, float*, float*,
    float*, float*, float*, float*,
    float*, float*) { return false; }
inline bool chiron_attention_shear_backward_bf16w_bf16g_tiled(
    const float*, const float*,
    const unsigned short*, const unsigned short*,
    const unsigned short*, const unsigned short*,
    int, int, int, int, bool,
    float*,
    unsigned short*, unsigned short*, unsigned short*, unsigned short*,
    unsigned short*, unsigned short*,
    float*, float*, float*, float*,
    float*, float*, float*, float*,
    float*, float*) { return false; }
inline bool flash_attention_backward_cublas_tiled(
    const float*, const float*, const float*,
    const float*, const float*,
    int, int, int, int, bool,
    float*, float*, float*, float*, float*) { return false; }
inline bool chiron_attention_shear_bf16(const float*, float*,
                                         const float*, const float*, const float*,
                                         const float*,
                                         int, int, int, int, int,
                                         bool, bool,
                                         float*, float*, float*, float*,
                                         uint16_t*, uint16_t*, uint16_t*) { return false; }
inline bool chiron_attention_shear_backward_bf16(const float*, const float*,
                                                  const float*, const float*, const float*, const float*,
                                                  int, int, int, int, int, bool,
                                                  float*, float*, float*, float*, float*,
                                                  float*, float*, float*, float*,
                                                  float*, float*, float*, float*,
                                                  uint16_t*, uint16_t*, uint16_t*) { return false; }
inline bool chiron_attention_shear_local_bf16(const float*, float*,
                                                const float*, const float*, const float*, const float*,
                                                int, int, int, int, int, bool, bool, int,
                                                float*, float*, float*, float*,
                                                unsigned short*, unsigned short*, unsigned short*) { return false; }
inline bool chiron_attention_shear_backward_local_bf16(const float*, const float*,
                                                        const float*, const float*, const float*, const float*,
                                                        int, int, int, int, int, bool, int,
                                                        float*, float*, float*, float*, float*,
                                                        float*, float*, float*, float*,
                                                        float*, float*, float*, float*,
                                                        unsigned short*, unsigned short*, unsigned short*) { return false; }
inline bool chiron_attention_shear_backward(const float*, const float*,
                                             const float*, const float*, const float*, const float*,
                                             int, int, int, int, int, bool,
                                             float*, float*, float*, float*, float*,
                                             float*, float*, float*, float*,
                                             float*, float*, float*, float*) { return false; }

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
