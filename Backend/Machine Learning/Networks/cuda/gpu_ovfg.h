// GPU primitives for Operator-Valued Factored Gradient (OVFG) — paradigm
// shift #9.  See research/PARADIGM_SHIFT_9_CANDIDATE_C_OVFG.md for the
// framework and research/PARADIGM_SHIFT_9_SELECTION.md for the rationale.
//
// Core observation: for every weight matrix W ∈ R^{m×n} in a transformer,
// the gradient produced by one microbatch is exactly
//     G = A^T · D,    A ∈ R^{T×m}   (input activations)
//                     D ∈ R^{T×n}   (upstream gradient)
// so rank(G) ≤ T.  OVFG never materializes G as a dense m×n tensor; it
// stores the pair (A, D) and pushes that pair all the way through Adam
// and into the weight update.
//
// Composes with shift #7 (Stiefel × Σ): the Stiefel tangent-space grads
// dU, dΣ, dV can be computed directly from (A, D, U, Σ, V) factors via
// two r×ρ GEMMs, never forming the dense dW.  Projected memory
// compression when composed: 17× on {grad, optimizer state} at pile_large
// 2.23 B settings (ρ=0.25, r=256).
//
// Composes with shift #8 (HRTC): the T axis of A and D is exactly the
// axis HRTC compresses, so post-HRTC factor sizes are (T/k)·(m+n).
//
// BF16 interop: L, R factors stored BF16; reduction accumulators stay
// FP32 (RᵀR, LᵀL, etc.) to match BF16-Adam noise floor.
//
// Phase 1 (this file): low-level primitive kernels — store factors,
// append to factored moments, apply dense update as fallback.
// Phase 2: RSVD-based rank truncation, Adafactor row/col 2nd moment.
// Phase 3: Stiefel coupling (closed-form dU/dΣ/dV from factor pairs).
// Phase 4: HRTC composition + end-to-end validation.
#pragma once

#include "gpu_buffer.h"
#include "gpu_stiefel.h"
#include <cstddef>
#include <stdint.h>

namespace shmea { class GLogger; }

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// OvfgFactoredGrad
//
// Per-weight-matrix OVFG state on GPU.  The "dense" gradient dW ∈ R^{m×n}
// is represented as the factored pair (L, R) with L ∈ R^{m×r}, R ∈ R^{n×r}
// such that dW ≈ L · R^T.  Accumulation, Adam first moment, and the
// dense-W update path all operate on (L, R) directly.
// ========================================================================
struct OvfgFactoredGrad
{
	unsigned int m;       // weight-matrix rows
	unsigned int n;       // weight-matrix cols
	unsigned int T;       // sequence length that produced the factors
	unsigned int r;       // current rank of the factorization (≤ min(T,m,n))
	unsigned int r_max;   // hard cap on rank (truncation target)

	// Factored moment storage (BF16 in production; FP32 here for Phase 1
	// parity validation — will shrink in Phase 2 after tests pass).
	GpuBuffer<float> L;   // [m * r_max]  first-moment left factor
	GpuBuffer<float> R;   // [n * r_max]  first-moment right factor

	// Adafactor row/col for second moment — stored FP32.  Populated in
	// Phase 2.
	GpuBuffer<float> c;   // [m]
	GpuBuffer<float> d;   // [n]

	OvfgFactoredGrad()
	    : m(0), n(0), T(0), r(0), r_max(0)
	{}
};

// ========================================================================
// ovfg_compute_dense_from_factors
//
// Reference / parity helper.  Reconstructs the dense gradient
//     G = L · R^T         shape [m × n]
// from its factored representation using cuBLAS SGEMM.
//
// Use for unit tests only — the whole point of OVFG is to AVOID this
// materialization in production.
// ========================================================================
bool ovfg_compute_dense_from_factors(const float* L, const float* R,
                                     unsigned int m, unsigned int n,
                                     unsigned int r,
                                     float* G_out);

// ========================================================================
// ovfg_factored_grad_from_activation
//
// Given the pre-weight activation A [T × m] and upstream gradient
// D [T × n], produce the rank-T factored representation of the gradient
//     G = A^T · D ∈ R^{m × n}
// as the pair (L, R) with L = A^T ∈ R^{m × T}, R = D ∈ R^{n × T} (after
// transpose swap).  Storage is a trivial copy + transpose — the "factoring"
// is simply recognizing that the outer-product structure already exists
// in the backward-pass inputs.
//
// The caller must have allocated L [m × T] and R [n × T].  T becomes the
// rank of the resulting factored grad (pre-truncation).
//
// Row-major throughout.  Cost: two D2D copy kernels, no SGEMM.
// ========================================================================
bool ovfg_factored_grad_from_activation(const float* A, const float* D,
                                        unsigned int T,
                                        unsigned int m, unsigned int n,
                                        float* L_out, float* R_out);

// ========================================================================
// ovfg_apply_update_dense
//
// Phase 1 fallback path: apply the factored Adam update to a dense weight
// matrix W by materializing the rank-r update via one SGEMM.
//
//     W ← W − η · L · R^T
//
// For use when the weight matrix is *not* itself factored (LayerNorm γ,
// β, residual linears).  Production path for Stiefel-factored weights
// will skip this and go through ovfg_stiefel_tangent_grad (Phase 3).
// ========================================================================
bool ovfg_apply_update_dense(const float* L, const float* R,
                             unsigned int m, unsigned int n,
                             unsigned int r,
                             float eta,
                             float* W);

// ========================================================================
// ovfg_adafactor_moments
//
// Computes the Adafactor-style row/col second-moment updates from the
// factored gradient G = L · R^T WITHOUT ever materializing G.
//
// Naive reference:
//     c[i] ← β2 · c[i] + (1-β2) · Σ_j G[i,j]²
//     d[j] ← β2 · d[j] + (1-β2) · Σ_i G[i,j]²
//
// Factored closed form (the OVFG payoff clause):
//     Σ_j G[i,j]² = diag(G G^T)[i] = diag(L · (R^T R) · L^T)[i]
//                 = Σ_k (L · S)[i,k] · L[i,k]     with S = R^T R
//     Σ_i G[i,j]² = diag(G^T G)[j] = diag(R · (L^T L) · R^T)[j]
//                 = Σ_k (R · P)[j,k] · R[j,k]     with P = L^T L
//
// Cost: 2 r×r Grams + 2 (m or n) × r SGEMMs + 2 row-dot kernels.
//       O((m + n) r² + r³) — no m×n materialization.
//
// Scratch requirement: the caller must supply a scratch buffer of size
//       max(m*r, n*r) + r*r  floats
// to hold the intermediate L·S, R·P, and S, P tensors.  Reusing a single
// buffer across calls is fine (kernels synchronize on computeStream()).
// ========================================================================
bool ovfg_adafactor_moments(const float* L, const float* R,
                            unsigned int m, unsigned int n,
                            unsigned int r,
                            float beta2,
                            float* c, float* d,
                            float* scratch);

// ========================================================================
// ovfg_first_moment_append
//
// Scaled-concatenation implementation of the factored Adam first-moment
// update:
//     M_new = β1 · M + (1-β1) · G_acc
// With M = L R^T and G_acc = L_acc R_acc^T, we have
//     M_new = L_new R_new^T
// where
//     L_new = [√β1 · L  |  √(1-β1) · L_acc]    shape [m × (r + r_acc)]
//     R_new = [√β1 · R  |  √(1-β1) · R_acc]    shape [n × (r + r_acc)]
//
// This is pure copy + scale: no SGEMM, no truncation.  Rank grows by
// r_acc per call; Phase 2b will add an ovfg_rsvd_truncate pass to
// re-cap the rank after append.
//
// The caller owns L_new and R_new storage.  r + r_acc must not exceed
// the L_new / R_new column capacity.
// ========================================================================
bool ovfg_first_moment_append(const float* L, const float* R,
                              unsigned int r,
                              const float* L_acc, const float* R_acc,
                              unsigned int r_acc,
                              unsigned int m, unsigned int n,
                              float beta1,
                              float* L_new, float* R_new);

// ========================================================================
// ovfg_stiefel_tangent_grad — Phase 3 payoff clause.
//
// Produces the SAME Stiefel tangent-projected gradient triple
// (dU, dΣ, dV) that stiefel_dense_grad_to_tangent produces for a given
// dense dW, but taking the OVFG factored form (L, R, r) as input so
// that dW = L · R^T is NEVER materialized.
//
// Closed-form derivation.  With U ∈ St(m,ρ), V ∈ St(n,ρ), Σ ∈ R^ρ and
// dW = L · R^T:
//
//   Let A = U^T · L   ∈ R^{ρ × r}
//       B = V^T · R   ∈ R^{ρ × r}
//
//   dU_raw = (dW · V) · diag(Σ)     = (L · B^T) · diag(Σ)
//   dV_raw = (dW^T · U) · diag(Σ)   = (R · A^T) · diag(Σ)
//   dΣ[i]  = diag(U^T · dW · V)[i]  = Σ_k A[i,k] · B[i,k]
//
// Then dU_raw and dV_raw are tangent-projected by
// stiefel_tangent_project_grad to produce the final dU, dV in place.
//
// Cost: two ρ×r GEMMs (A, B) + two m×ρ and n×ρ GEMMs (dU_raw, dV_raw) +
//   one diag kernel + two Σ column-scales + the tangent-projection step.
// Total O(ρ r (m + n) + ρ² (m + n) + ρ r ρ)  — no m×n tensor ever formed.
//
// Composes multiplicatively with Stiefel (shift #7): when ρ < min(m,n)
// and r < min(m,n), both dimensions compress, so gradient state scales
// like (m+n)·max(ρ, r) instead of m·n.  At ρ=0.25·min(m,n) and r=T=1024
// on pile_large this is ~17× compression vs. dense-dW Adam.
//
// Scratch requirement:
//   2·ρ·r  floats (for A, B) +
//   2·ρ·ρ  floats (scratch_rr, scratch_rr2 for tangent projection)
// = 2·ρ·(r + ρ)  floats total, supplied in ONE packed buffer.
// ========================================================================
bool ovfg_stiefel_tangent_grad(const GpuStiefelWeight& s,
                               const float* L, const float* R,
                               unsigned int r,
                               float* dU, float* dSigma, float* dV,
                               float* scratch);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

struct OvfgFactoredGrad {
	unsigned int m; unsigned int n; unsigned int T;
	unsigned int r; unsigned int r_max;
};

inline bool ovfg_compute_dense_from_factors(const float*, const float*,
                                            unsigned int, unsigned int,
                                            unsigned int, float*) { return false; }
inline bool ovfg_factored_grad_from_activation(const float*, const float*,
                                               unsigned int, unsigned int, unsigned int,
                                               float*, float*) { return false; }
inline bool ovfg_apply_update_dense(const float*, const float*,
                                    unsigned int, unsigned int, unsigned int,
                                    float, float*) { return false; }
inline bool ovfg_adafactor_moments(const float*, const float*,
                                   unsigned int, unsigned int, unsigned int,
                                   float, float*, float*, float*) { return false; }
inline bool ovfg_first_moment_append(const float*, const float*, unsigned int,
                                     const float*, const float*, unsigned int,
                                     unsigned int, unsigned int,
                                     float, float*, float*) { return false; }
struct GpuStiefelWeight;
inline bool ovfg_stiefel_tangent_grad(const GpuStiefelWeight&,
                                      const float*, const float*, unsigned int,
                                      float*, float*, float*, float*) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
