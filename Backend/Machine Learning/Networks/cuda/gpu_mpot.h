// GPU primitives for MPOT — Matrix Product Operator weight decomposition
// (paradigm shift #10).  See research/PARADIGM_SHIFT_10_DESIGN.md for
// the full framework.
//
// A dense weight matrix W ∈ R^{m × n} with m = m_1 · m_2, n = n_1 · n_2
// is represented by two factor tensors
//     A ∈ R^{m_1 × n_1 × D}      (row-major)
//     B ∈ R^{D × m_2 × n_2}      (row-major)
// connected by a bond of dimension D, via the identity
//     W[i_1·m_2 + i_2,  j_1·n_2 + j_2]
//         = Σ_{α=0}^{D-1}  A[i_1, j_1, α]  ·  B[α, i_2, j_2]
//
// Storage: m_1·n_1·D + D·m_2·n_2 = D · (m_1 n_1 + m_2 n_2) ≈ 2 D √(mn)
// for balanced factoring (m_1 ≈ √m, n_1 ≈ √n).  At m = n = 2048, D = 16:
// 64 K entries vs m·n = 4.2 M — **65× compression**.
//
// Composes multiplicatively with Stiefel (shift #7) via nested
// factorization: Stiefel's U, V themselves become MPOs.  Projected
// combined compression: 500×.
//
// Phase 1a (this file): `mpot_reconstruct_dense` — parity helper that
// materializes W from (A, B).  Reference path only; production MPOT
// paths NEVER reconstruct the dense matrix.  Phase 1b will add
// `mpot_init_from_dense` (SVD-based bootstrap from a pretrained W);
// Phase 2 will add the forward/backward GEMM pair.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// mpot_reconstruct_dense — materialize W = A · B (MPO contraction) for
// parity testing.  O(m · n · D) compute, O(m · n) storage.  Never used
// in the production forward path; that's what MPOT is designed to avoid.
//
// Inputs:
//   A    [m_1 · n_1 · D]      row-major factor tensor
//   B    [D · m_2 · n_2]      row-major factor tensor
//   m_1, m_2, n_1, n_2, D     — factorization dimensions
//
// Output:
//   W_out [m · n] where m = m_1·m_2, n = n_1·n_2, row-major.
//
// Returns false on invalid args or kernel launch failure.
// ========================================================================
bool mpot_reconstruct_dense(const float* A, const float* B,
                            unsigned int m_1, unsigned int m_2,
                            unsigned int n_1, unsigned int n_2,
                            unsigned int D,
                            float* W_out);

// ========================================================================
// mpot_init_from_dense — bootstrap (A, B) from a pre-trained dense W
// via truncated SVD.  For r_full = min(m_1·n_1, m_2·n_2), the SVD of
// the index-permuted matrix M[i_1·n_1 + j_1, i_2·n_2 + j_2] = W[i, j]
// yields singular triples (U, Σ, V^T).  Truncate to top D components
// and split √Σ equally:
//     A[i_1, j_1, α] = U[i_1·n_1 + j_1, α] · √Σ[α]
//     B[α, i_2, j_2] = √Σ[α] · V[i_2·n_2 + j_2, α]
//
// Reconstruction error: ‖W − reconstruct(A, B)‖_F² = Σ_{k > D} σ_k²(M)
// (Eckart-Young).  Use D large enough to capture the desired energy.
//
// Requires D ≤ min(m_1·n_1, m_2·n_2).
//
// Inputs:
//   W       [m × n]              row-major, m = m_1·m_2, n = n_1·n_2
//   m_1, m_2, n_1, n_2, D        factoring dims + bond cap
// Outputs:
//   A       [m_1 · n_1 · D]      row-major
//   B       [D · m_2 · n_2]      row-major
//
// Caller must supply scratch of size:
//   m·n                         M (permuted W)
// + max(m·n, (m_1·n_1)² + min(m_1·n_1, m_2·n_2) + (m_2·n_2)²)
//   for the cuSOLVER SGESVD working set
// Use (m·n + 2·max(m_1·n_1, m_2·n_2)² + min(m_1·n_1, m_2·n_2)) as a
// generous upper bound.
// ========================================================================
bool mpot_init_from_dense(const float* W,
                          unsigned int m_1, unsigned int m_2,
                          unsigned int n_1, unsigned int n_2,
                          unsigned int D,
                          float* A, float* B,
                          float* scratch);

// ========================================================================
// mpot_forward — compute  Y = X · W^T  where W is stored in factored
// MPO form (A, B) — WITHOUT materializing the dense W.
//
// Inputs:
//   X   [T × m]              row-major, m = m_1·m_2
//   A   [m_1 · n_1 · D]      row-major MPO factor
//   B   [D · m_2 · n_2]      row-major MPO factor
//   T                         batch / token count
//   m_1, m_2, n_1, n_2, D    factorization dims + bond
// Output:
//   Y   [T × n]              row-major, n = n_1·n_2
//
// Algorithm (two-GEMM chain; see research/PARADIGM_SHIFT_10_DESIGN.md):
//
//   (1) Permute B from (D, m_2, n_2) row-major to (m_2, D, n_2) so that
//       X · B_perm is a standard matrix multiply.
//   (2) Permute A from (m_1, n_1, D) row-major to (m_1, D, n_1).
//   (3) GEMM: X [T·m_1, m_2] · B_perm [m_2, D·n_2]  →  T1 [T·m_1, D·n_2]
//       equivalent to  T1[t, i_1, α, j_2] = Σ_{i_2} X[t, i_1, i_2] · B[α, i_2, j_2].
//   (4) Permute T1 from (T, m_1, D, n_2) to (T, n_2, m_1, D).
//   (5) GEMM: T1_perm [T·n_2, m_1·D] · A_perm [m_1·D, n_1]  →  Y_pre [T·n_2, n_1]
//       equivalent to  Y_pre[t, j_2, j_1] = Σ_{i_1, α} T1[t, i_1, α, j_2] · A[i_1, j_1, α].
//   (6) Permute Y_pre (T, n_2, n_1) → Y (T, n_1, n_2).
//
// Cost: 2 GEMMs of shape (T · (m+n)^{1/2}) × D × (m·n)^{1/2} + 3 light
// permutations.  For square factoring (m_1=m_2=√m, n_1=n_2=√n), total
// FLOPs ≈ T · D · (m · n)^{1/2} · (√m + √n) vs T · m · n for dense —
// a 2.8× theoretical speedup at m=n=2048, D=16 (see design doc).
//
// Scratch requirement (caller-allocated):
//   m_1·n_1·D               (A_perm — same size as A)
// + D·m_2·n_2               (B_perm — same size as B)
// + T·m_1·D·n_2             (T1)
// + T·n_2·m_1·D             (T1_perm — same size as T1)
// + T·n_1·n_2               (Y_pre — same size as Y)
// ========================================================================
bool mpot_forward(const float* X,
                  const float* A, const float* B,
                  unsigned int T,
                  unsigned int m_1, unsigned int m_2,
                  unsigned int n_1, unsigned int n_2,
                  unsigned int D,
                  float* Y,
                  float* scratch);

// ========================================================================
// mpot_backward — gradients through the factored forward  Y = X · W^T
// back to (X, A, B) without materializing the dense W.
//
// Inputs:
//   X    [T × m]              forward input (m = m_1·m_2)
//   A    [m_1 · n_1 · D]      forward MPO factor
//   B    [D · m_2 · n_2]      forward MPO factor
//   dY   [T × n]              upstream gradient (n = n_1·n_2)
//   T, m_1, m_2, n_1, n_2, D  dims
// Outputs:
//   dX   [T × m]
//   dA   [m_1 · n_1 · D]
//   dB   [D · m_2 · n_2]
//
// Chain rule through the forward's two GEMMs (GEMM2: Y_pre = T1_perm ·
// A_perm; GEMM1: T1 = X · B_perm) yields:
//
//   dT1_perm = dY_pre · A_perm^T
//   dA_perm  = T1_perm^T · dY_pre
//   dT1      = permute⁻¹(dT1_perm)
//   dX       = dT1 · B_perm^T
//   dB_perm  = X^T · dT1
//   dA       = permute⁻¹(dA_perm)
//   dB       = permute⁻¹(dB_perm)
//
// The four permutation inverses reuse the same kernels as the forward
// with the source/destination layouts swapped.
//
// Self-contained: recomputes A_perm, B_perm, T1, T1_perm, dY_pre from
// (X, A, B, dY) rather than taking forward-path state.  Slightly
// redundant but keeps the API clean; a future phase can expose a
// shared-state variant for inner training loops.
//
// Scratch requirement (caller-allocated):
//   |A_perm| + |B_perm| + |T1| + |T1_perm| + |dY_pre|
// + |dT1_perm| + |dT1| + |dA_perm| + |dB_perm|
// = 3·m_1·D·n_1 + 3·m_2·D·n_2 + 2·T·m_1·D·n_2 + 2·T·n_1·n_2·0  — roughly
// = 3·|A| + 3·|B| + 2·|T1| + |Y|.
// ========================================================================
bool mpot_backward(const float* X,
                   const float* A, const float* B,
                   const float* dY,
                   unsigned int T,
                   unsigned int m_1, unsigned int m_2,
                   unsigned int n_1, unsigned int n_2,
                   unsigned int D,
                   float* dX, float* dA, float* dB,
                   float* scratch);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool mpot_reconstruct_dense(const float*, const float*,
                                   unsigned int, unsigned int,
                                   unsigned int, unsigned int,
                                   unsigned int, float*) { return false; }
inline bool mpot_init_from_dense(const float*,
                                 unsigned int, unsigned int,
                                 unsigned int, unsigned int,
                                 unsigned int,
                                 float*, float*, float*) { return false; }
inline bool mpot_forward(const float*, const float*, const float*,
                         unsigned int,
                         unsigned int, unsigned int,
                         unsigned int, unsigned int,
                         unsigned int, float*, float*) { return false; }
inline bool mpot_backward(const float*, const float*, const float*, const float*,
                          unsigned int,
                          unsigned int, unsigned int,
                          unsigned int, unsigned int,
                          unsigned int,
                          float*, float*, float*, float*) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
