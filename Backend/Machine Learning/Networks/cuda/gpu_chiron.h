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
inline bool flash_attention_cublas_tiled(const float*, const float*, const float*,
                                          int, int, int, int, bool,
                                          float*, float*) { return false; }
inline bool flash_attention_cublas_tiled_bf16(
    const float*, const float*, const float*,
    int, int, int, int, bool,
    float*, float*,
    unsigned short*, unsigned short*, unsigned short*, unsigned short*) { return false; }
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
inline bool chiron_attention_shear_backward(const float*, const float*,
                                             const float*, const float*, const float*, const float*,
                                             int, int, int, int, int, bool,
                                             float*, float*, float*, float*, float*,
                                             float*, float*, float*, float*,
                                             float*, float*, float*, float*) { return false; }

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
