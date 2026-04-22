// GPU primitives for IBGRAD — Information-Bottleneck Gradient Subspace
// (paradigm shift #19).  See research/PARADIGM_SHIFT_19_SELECTION.md
// and research/PARADIGM_SHIFT_19_CANDIDATE_C_IBGRAD.md.
//
// Block-diagonal learned projection P_l ∈ ℝ^{N_l × r_l} for each layer l.
// The subspace is the top-r eigenvectors of E[g gᵀ] — a PCA on the
// streaming gradient, updated via Oja's rule.  Optimizer state lives
// in the r-dim subspace (Adam's m, v are r-dim per block rather than
// N_l-dim).  The backward GEMM is factored as g_sub = Pᵀ · g (r-dim)
// and the update θ_l += P_l · adam(g_sub, ...) (N_l-dim).
//
// Typical r_l ≈ √N_l with global budget r/N = 0.05 → 20× state and
// backward-GEMM compression.
//
// Phase 1 (this file): init, project, Oja rank-1 update, unproject.
// Phase 2: periodic QR re-orthogonalization (requires cuSOLVER).
// Phase 3: CHIRON trainer wire-in + pile_large benchmark.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>
#include <stdint.h>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// ibgrad_init_projection — initialize P ∈ ℝ^{N × r} with random Gaussian
// entries of variance 1/N (so that Pᵀ·P ≈ I_r in expectation by CLT).
//
// Caller is responsible for a one-time QR to exactly orthonormalize; this
// init matches Xavier scaling closely enough that the first several
// Oja updates produce a usable subspace even before the first QR.
//
//   P_out      [N × r]         output buffer (row-major)
//   N, r       dimensions      r ≤ N
//   seed       uint64_t        RNG seed (deterministic)
//
// Uses Box-Muller on splitmix64 counter.
// ========================================================================
bool ibgrad_init_projection(float* P_out,
                            unsigned int N, unsigned int r,
                            uint64_t seed);

// ========================================================================
// ibgrad_project — g_sub = Pᵀ · g  (r-dim projection of the full gradient).
//
//   P          [N × r]         projection matrix (row-major)
//   g          [N]             full gradient (flattened)
//   N, r       dimensions
//   g_sub_out  [r]             subspace gradient
//
// Uses existing sgemv (P^T as column-major "P with op_T" against g).
// ========================================================================
bool ibgrad_project(const float* P, const float* g,
                    unsigned int N, unsigned int r,
                    float* g_sub_out);

// ========================================================================
// ibgrad_unproject — update_full = P · update_sub (N-dim update from r-dim).
//
//   P            [N × r]
//   update_sub   [r]
//   N, r         dimensions
//   update_full_out  [N]       overwritten by P · update_sub
// ========================================================================
bool ibgrad_unproject(const float* P, const float* update_sub,
                      unsigned int N, unsigned int r,
                      float* update_full_out);

// ========================================================================
// ibgrad_oja_rank1_update — Oja streaming PCA step (no correction term;
// caller must periodically call the QR re-orthogonalization).
//
//     P ← P + eta · g · yᵀ
//
// where y = Pᵀ · g has already been computed by ibgrad_project.  This is
// a rank-1 update via cuBLAS sger.
//
//   P_inout    [N × r]         updated in-place
//   g          [N]             current gradient
//   y          [r]             Pᵀ · g from the project call
//   N, r       dimensions
//   eta        scalar          learning rate for the subspace update
// ========================================================================
bool ibgrad_oja_rank1_update(float* P_inout,
                             const float* g, const float* y,
                             unsigned int N, unsigned int r,
                             float eta);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool ibgrad_init_projection(float*, unsigned int, unsigned int,
                                   uint64_t) { return false; }
inline bool ibgrad_project(const float*, const float*,
                           unsigned int, unsigned int, float*) { return false; }
inline bool ibgrad_unproject(const float*, const float*,
                             unsigned int, unsigned int, float*) { return false; }
inline bool ibgrad_oja_rank1_update(float*, const float*, const float*,
                                    unsigned int, unsigned int, float) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
