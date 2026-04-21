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
inline bool chiron_sketch_project(const float*, const float*, int, int, int,
                                   float*) { return false; }
inline bool chiron_sketch_lift_add(float*, const float*, const float*,
                                    int, int, int) { return false; }

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
