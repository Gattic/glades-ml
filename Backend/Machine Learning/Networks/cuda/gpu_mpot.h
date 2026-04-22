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

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool mpot_reconstruct_dense(const float*, const float*,
                                   unsigned int, unsigned int,
                                   unsigned int, unsigned int,
                                   unsigned int, float*) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
