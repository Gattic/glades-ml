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
// Production version (iter 22+) will use CSR-style per-vertex reduction
// to restore bit-exact determinism.
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
inline bool sfa_source_assembly_fp32(const float*, const float*, const float*,
                                      const float*, const float*, float, float*,
                                      int, int, int, int, cudaStream_t = 0) { return false; }
inline bool sfa_readout_fp32(const float*, const float*, float*, int, int, int,
                              cudaStream_t = 0) { return false; }

}  // namespace gpu
}  // namespace glades

#endif  // GLADES_HAVE_CUDA
