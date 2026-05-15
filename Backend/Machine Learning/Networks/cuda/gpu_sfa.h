// SFA (Sheaf-Focal Attention) GPU primitives — paradigm #250.
//
// Host wrappers for CUDA kernels that implement the SFA forward components.
// Mirrors the CPU header `transformer_sfa_ops.h` (which remains the reference
// implementation, and is used in CPU-vs-GPU parity tests).
//
// When GLADES_HAVE_CUDA is not defined, the wrappers degrade to inline no-ops
// returning false — callers can unconditionally compile against this header.
//
// See research/PARADIGM_SHIFT_250_DESIGN.md for the design and
// research/CSA_GATE0_IMPLEMENTATION_PLAN.md for integration.
//
// Conventions (match gpu_chiron.h / gpu_kernels.h):
// - Row-major contiguous buffers.
// - U[T][d_s][r] flattened as i*d_s*r + a*r + beta.
// - Section s[T][d_s] flattened as i*d_s + a.
// - Edges stored as parallel arrays edge_src[|E|], edge_tgt[|E|].
// - Sigma[|E|][r] flattened as e*r + beta.
// - FP32 in CPU prototype + initial GPU port; BF16 mixed-precision via
//   cublasGemmEx in a future iter.
// - Optional cudaStream_t parameter (default = 0 → computeStream()).
// - bool return: true on success, false on launch error.

#pragma once

#include <cstddef>
#include <stdint.h>
#include "gpu_device.h"  // cudaStream_t

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// ---------------------------------------------------------------------------
// Sheaf-Laplacian matvec: out = L_F · s.
//
// L_F is encoded by per-token stalk frames U[T*d_s*r], per-edge diagonal
// modulators Sigma[|E|*r], and edge lists edge_src[|E|], edge_tgt[|E|].
// Both inputs s[T*d_s] and output out[T*d_s] are dense FP32 row-major.
//
// Mathematical content (matches transformer_sfa_ops.h::laplacianMatvec):
//   For each edge e = (i, j):
//     R_{j<-i} s_i = U_j (Sigma_e ⊙ (U_i^T s_i))
//     delta_e = s_j - R_{j<-i} s_i
//     out_j += delta_e
//     out_i -= R_{j<-i}^T delta_e
//
// Cost: O(|E| · d_s · r) FLOPs.
//
// IMPLEMENTATION NOTE (iter 21): uses atomicAdd reductions per edge for
// simplicity. This introduces non-determinism at the FP32-epsilon level.
// Production version (CSR variant below, iter 22+) uses per-vertex
// reduction to restore bit-exact determinism.
// ---------------------------------------------------------------------------
bool sfa_laplacian_matvec_fp32(const float* U,         // [T * d_s * r]
                                const float* Sigma,     // [|E| * r]
                                const int* edge_src,    // [|E|]
                                const int* edge_tgt,    // [|E|]
                                const float* s,         // [T * d_s] input
                                float* out,             // [T * d_s] output
                                int T, int E,
                                int d_s, int r,
                                cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// CSR-format L_F matvec — deterministic and faster than atomicAdd variant.
//
// CSR layout:
//   out_csr_off[i]   = start index in out_csr_edges of OUTGOING edges from i
//                       (where i is src and there's an edge i->j).
//   out_csr_off[T]   = total count.
//   out_csr_edges[k] = global edge index of the k-th outgoing edge.
//
//   in_csr_off[i]   = start index in in_csr_edges of INCOMING edges to i.
//   in_csr_off[T]   = total count.
//   in_csr_edges[k] = global edge index of the k-th incoming edge.
//
// Build CSR from edge_src/edge_tgt arrays via sfa_build_csr_host (host-side).
//
// Kernel: one block per vertex. Threads in the block cooperate to:
//   (a) sum contributions FROM outgoing edges (vertex is src in edge e):
//       contributes -R_{j<-i}^T delta_e to out_i
//   (b) sum contributions FROM incoming edges (vertex is tgt in edge e):
//       contributes +delta_e to out_j
//
// No atomicAdds. Bit-exact deterministic given fixed CSR ordering.
// ---------------------------------------------------------------------------
bool sfa_laplacian_matvec_csr_fp32(const float* U,           // [T * d_s * r]
                                    const float* Sigma,       // [|E| * r]
                                    const int* edge_src,      // [|E|]
                                    const int* edge_tgt,      // [|E|]
                                    const int* out_csr_off,   // [T + 1]
                                    const int* out_csr_edges, // [|E|]
                                    const int* in_csr_off,    // [T + 1]
                                    const int* in_csr_edges,  // [|E|]
                                    const float* s,           // [T * d_s]
                                    float* out,               // [T * d_s]
                                    int T, int E,
                                    int d_s, int r,
                                    cudaStream_t stream = 0);

// Build CSR offsets/indices arrays on host from edge_src/edge_tgt.
// Caller passes pre-allocated output vectors which are filled with size
// [T+1] for offsets and [E] for edge indices.
void sfa_build_csr_host(const int* edge_src, const int* edge_tgt, int E, int T,
                         int* out_csr_off, int* out_csr_edges,
                         int* in_csr_off, int* in_csr_edges);

// ---------------------------------------------------------------------------
// Phase 4b (2026-05-15): Jacobi-preconditioned Richardson primitives.
//
// Tikhonov solve (L_F + λI) σ = b has condition number κ = (‖L_F‖ + λ)/λ ≈
// 14000 at W=128 + n_sinks=8, σ ∈ [0.5,1], λ=0.01.  Plain Richardson needs
// O(κ) iters → impractical.  Jacobi preconditioning with D = diag(L_F + λI)
// reduces κ' ≈ 14 → ~50 Richardson iters for 1e-3 residual.
//
// Usage at SFA init (once):
//   1. sfa_laplacian_diagonal_fp32  → diag = diag(L_F + λI)
//   2. sfa_jacobi_inverse_diagonal_fp32 → Dinv = 1/diag (with clamp)
// Usage per Richardson iter:
//   3. sfa_laplacian_matvec_fp32 → Ls = L_F · s
//   4. sfa_jacobi_step_fp32       → s += α · Dinv · (b - Ls - λ·s)  (fused)
// ---------------------------------------------------------------------------

// Compute the per-element diagonal of (L_F + λI) into `diag` [T · d_s].
// Output is FP32 row-major, same indexing as the s/b buffers.
bool sfa_laplacian_diagonal_fp32(const float* U,         // [T · d_s · r]
                                  const float* Sigma,     // [|E| · r]
                                  const int* edge_src,    // [|E|]
                                  const int* edge_tgt,    // [|E|]
                                  float lambda,
                                  float* diag,            // [T · d_s] output
                                  int T, int E,
                                  int d_s, int r,
                                  cudaStream_t stream = 0);

// Element-wise reciprocal with a floor at `eps` (avoids div-by-zero on
// near-zero diagonal entries that could occur if Σ is small or λ is tiny).
bool sfa_jacobi_inverse_diagonal_fp32(const float* diag,
                                       float* Dinv,
                                       float eps,
                                       int T, int d_s,
                                       cudaStream_t stream = 0);

// One fused Jacobi-preconditioned Richardson step:
//   s += α · Dinv · (b - Ls - λ · s)
// where Ls = L_F · s has been precomputed by the caller.  Replaces a
// 4-axpy chain (memcpy(res,b), axpy(-1,Ls,res), axpy(-λ,s,res), axpy(α,res,s))
// with one element-wise kernel — both faster and easier to extend.
bool sfa_jacobi_step_fp32(float* s,                  // [T · d_s] in/out
                           const float* b,            // [T · d_s]
                           const float* Ls,           // [T · d_s]
                           const float* Dinv,         // [T · d_s]
                           float lambda, float alpha,
                           int T, int d_s,
                           cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Source assembly: b_i = U_i U_i^T P_q q_i + gamma * P_v v_i
//
// Per-token GEMV-like computation. Cost: O(T · (d_h·d_s + d_s·r)).
// ---------------------------------------------------------------------------
bool sfa_source_assembly_fp32(const float* U,         // [T * d_s * r]
                               const float* P_q,       // [d_s * d_h]
                               const float* P_v,       // [d_s * d_h]
                               const float* q,         // [T * d_h]
                               const float* v,         // [T * d_h]
                               float gamma,
                               float* b,               // [T * d_s]
                               int T, int d_s, int d_h, int r,
                               cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Readout: y_i = P_o^T s_i
// ---------------------------------------------------------------------------
bool sfa_readout_fp32(const float* P_o,                // [d_h * d_s]
                       const float* s,                  // [T * d_s]
                       float* y,                        // [T * d_h]
                       int T, int d_s, int d_h,
                       cudaStream_t stream = 0);

}  // namespace gpu
}  // namespace glades

#else  // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

inline bool sfa_laplacian_matvec_fp32(const float*, const float*, const int*, const int*,
                                       const float*, float*, int, int, int, int,
                                       cudaStream_t = 0) { return false; }
inline bool sfa_laplacian_matvec_csr_fp32(const float*, const float*, const int*, const int*,
                                           const int*, const int*, const int*, const int*,
                                           const float*, float*, int, int, int, int,
                                           cudaStream_t = 0) { return false; }
inline void sfa_build_csr_host(const int*, const int*, int, int, int*, int*, int*, int*) {}
inline bool sfa_laplacian_diagonal_fp32(const float*, const float*, const int*, const int*,
                                         float, float*, int, int, int, int,
                                         cudaStream_t = 0) { return false; }
inline bool sfa_jacobi_inverse_diagonal_fp32(const float*, float*, float, int, int,
                                              cudaStream_t = 0) { return false; }
inline bool sfa_jacobi_step_fp32(float*, const float*, const float*, const float*,
                                  float, float, int, int,
                                  cudaStream_t = 0) { return false; }
inline bool sfa_source_assembly_fp32(const float*, const float*, const float*,
                                      const float*, const float*, float, float*,
                                      int, int, int, int, cudaStream_t = 0) { return false; }
inline bool sfa_readout_fp32(const float*, const float*, float*, int, int, int,
                              cudaStream_t = 0) { return false; }

}  // namespace gpu
}  // namespace glades

#endif  // GLADES_HAVE_CUDA
