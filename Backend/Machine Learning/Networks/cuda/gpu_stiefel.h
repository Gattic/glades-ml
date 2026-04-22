// GPU-accelerated Stiefel × Σ manifold-factored weights (paradigm shift #7).
//
// Represents each transformer weight matrix W = U · diag(Σ) · V^T with
//   U ∈ St(m,r)  (orthonormal columns: U^T U = I_r)
//   V ∈ St(n,r)
//   Σ ∈ R_+^r   (positive singular values)
//
// Intrinsic DOF k(m,n,r) = (m+n)·r − r². Forward FLOPs per GEMM scale as
// 2·r·(m+n)/(m·n), a 2·(r/d) = 2·ρ reduction over unconstrained at m=n=d.
//
// See research/WEIGHT_MANIFOLD_DESIGN.md for the full derivation and
// selection rationale.
//
// Forward path (X·W^T via 3 chained SGEMMs, inner dim r):
//     Y = ((X · V) * Σ) · U^T
//
// Backward / optimization:
//   - Tangent projection of the unconstrained gradient G ∈ R^{m×n} onto
//     T_W M yields (G_U, g_Σ, G_V) via:
//       G_U = (I − UU^T) · G · V · Σ + U · skew(U^T G V Σ)
//       G_V = (I − VV^T) · G^T · U · Σ + V · skew(V^T G^T U Σ)
//       g_Σ = diag(U^T G V)
//   - Riemannian Adam maintains first/second moments per sub-tensor in
//     int8-packed form (reuses existing asymmetric int8 Adam codec).
//   - QR retraction via cuSOLVER sgeqrf + sorgqr projects `U + η_U` back
//     onto St(m,r). A Cayley retraction fast-path is available for every
//     intermediate step to amortize QR cost.
//
// CHIRON compatibility: the symplectic shear on (q,p) is unchanged; the
// Stiefel structure is a constraint on the *weight parameters* Y uses,
// orthogonal to the phase-space flow. Inverse walk is preserved exactly
// (W is a deterministic function of (U, Σ, V) with no randomness).
//
// BF16 interop: U and V are stored in BF16 but regain exact orthonormality
// after each QR retraction — Stiefel is a contraction basin for BF16
// stochastic-rounding noise (‖ε‖ < 2^{-7} projects back to St with drift
// O(‖ε‖²), absorbed by the next retraction).
#pragma once

#include "gpu_buffer.h"
#include <cstddef>
#include <stdint.h>

namespace shmea { class GLogger; }

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// Per-weight-matrix Stiefel state on GPU.
//
// U (m × r BF16), Σ (r FP32), V (n × r BF16) live here as the canonical
// parameter representation. Adam moments are int8-packed triples
// (m_U, m_Σ, m_V) and (v_U, v_Σ, v_V); the scale factors are stored in
// FP32 per sub-tensor. Vector transport on m_U, m_V is performed before
// the Stiefel accumulation to keep momenta in the current tangent space.
struct GpuStiefelWeight
{
	unsigned int m;
	unsigned int n;
	unsigned int r;   // rank of the factorization; 1 ≤ r ≤ min(m,n)

	// Canonical parameter storage.
	GpuBuffer<uint16_t> U;      // [m * r] BF16
	GpuBuffer<float>    sigma;  // [r]     FP32 (positive orthant)
	GpuBuffer<uint16_t> V;      // [n * r] BF16

	// FP32 cache of U, V.  Populated lazily by stiefel_forward/backward and
	// invalidated by retraction.  Phase 2f: cast once per Adam step instead
	// of once per call, saving ~4× cast overhead in the fwd+bwd+Adam path.
	GpuBuffer<float>    U_f32_cache;
	GpuBuffer<float>    V_f32_cache;
	unsigned long long  param_version;     // bumped on each retraction
	unsigned long long  cache_version;     // last version the cache reflects

	// Riemannian Adam moments. Phase 2e ships these as FP32 for ease of
	// validation; Phase 2f will compress to int8/uint8 packed.
	GpuBuffer<float>   m_U;     // [m * r]
	GpuBuffer<float>   m_sigma; // [r]
	GpuBuffer<float>   m_V;     // [n * r]
	GpuBuffer<float>   v_U;     // [m * r]
	GpuBuffer<float>   v_sigma; // [r]
	GpuBuffer<float>   v_V;     // [n * r]

	// QR retraction scratch (tau vectors + workspace).
	GpuBuffer<float>   qr_tau_U;  // [r]
	GpuBuffer<float>   qr_tau_V;  // [r]
	GpuBuffer<float>   qr_work;   // cuSOLVER workspace, sized on first use

	// Re-projection health metric (‖U^T U − I‖_F).
	GpuBuffer<float>   orthogonality_drift; // [1] checked every N steps

	GpuStiefelWeight();
	void allocate(unsigned int m_, unsigned int n_, unsigned int r_);
	void release();  // wraps GpuBuffer<T>::free() across all buffers
	bool allocated() const;
};

// ========================================================================
// Forward: Y = X · W^T where W = U · diag(Σ) · V^T, via 3 chained SGEMMs.
//
// Buffers:
//   X        [B × n] input, either FP32 or BF16 (pass dtype flag)
//   stiefel  parameter state (U, Σ, V)
//   Y        [B × m] output
//   scratch1 [B × r] scratch for (X · V)
//   scratch2 [B × r] scratch for ((X·V) * Σ) (pre-Σ multiply skipped;
//            we fuse the Σ scale into the 2nd SGEMM via diag multiply)
//
// Input dtype is controlled by `x_bf16` (true → BF16 cuBLAS-tiled path).
// Output is always FP32.
// ========================================================================
void stiefel_forward(
    const void* X,
    bool x_bf16,
    const GpuStiefelWeight& stiefel,
    float* Y,
    float* scratch1,   // [B × r]
    unsigned int B);

// ========================================================================
// Unconstrained backward: raw chain-rule gradient through the 3-GEMM
// factored forward Y = X · V · diag(Σ) · U^T. Computes:
//   dX[B,n]        = dY · U · diag(Σ) · V^T
//   dU[m,r]        = dY^T · (X · V · diag(Σ))      ← raw, NOT tangent-projected
//   dΣ[r]          = diag(V^T · X^T · dY · U)
//   dV[n,r]        = X^T · (dY · U · diag(Σ))      ← raw
//
// This matches finite differences of (X, U, Σ, V) → Y exactly and serves
// as the parity-test oracle before tangent projection is layered on top.
// All dtypes FP32; BF16 variant comes later.
// ========================================================================
void stiefel_backward_unconstrained(
    const float* dY,                  // [B × m] upstream grad
    const void* X,                    // [B × n] forward input
    bool x_bf16,
    const GpuStiefelWeight& stiefel,
    float* dX,                        // [B × n] output, may be nullptr
    float* dU,                        // [m × r] output (raw, NOT tangent)
    float* dsigma,                    // [r] output
    float* dV,                        // [n × r] output (raw, NOT tangent)
    float* scratch_Br,                // [B × r] scratch
    unsigned int B);

// ========================================================================
// Tangent-space projection of raw grad_U, grad_V onto the tangent space
// of the Stiefel factors. Canonical metric:
//   proj_U(G_U) = (I − U U^T) G_U + U · skew(U^T G_U)
//   proj_V(G_V) = (I − V V^T) G_V + V · skew(V^T G_V)
// where skew(A) = (A − A^T) / 2. Done in place on grad_U, grad_V.
// ========================================================================
void stiefel_tangent_project_grad(
    const GpuStiefelWeight& stiefel,
    float* grad_U,                    // [m × r] in/out
    float* grad_V,                    // [n × r] in/out
    float* scratch_UtGU,              // [r × r]
    float* scratch_VtGV);             // [r × r]

// Legacy symbol retained — combines the unconstrained and project steps.
// (TODO: remove once callers migrate.)
void stiefel_backward_project(
    const float* G,
    const void* X,
    bool x_bf16,
    const GpuStiefelWeight& stiefel,
    float* grad_U, float* grad_sigma, float* grad_V,
    float* scratch_UtGV,
    unsigned int B);

// ========================================================================
// Riemannian Adam step. Composes the Phase 2 building blocks:
//   (1) Tangent-project dU, dV onto T_U Stiefel, T_V Stiefel
//   (2) Update FP32 first/second moments on U, Σ, V
//   (3) Compute bias-corrected Adam step in tangent space
//   (4) QR-retract U ← qf(U + η_U), V ← qf(V + η_V)
//   (5) Fisher–Rao exp update Σ ← Σ ⊙ exp(η_Σ / Σ)
//
// `step_1based` is the current optimizer step (≥ 1), used for Adam bias
// correction. Scratches:
//   scratch_rr_U [r × r], scratch_rr_V [r × r] — for tangent projection
//   scratch_etaU [m × r], scratch_etaV [n × r] — for Adam tangent step η
//   scratch_etaS [r]                            — for Adam Σ step
// The raw gradient buffers dU, dV are overwritten with their tangent
// projections during this call.
// ========================================================================
void stiefel_adam_step(
    GpuStiefelWeight& stiefel,
    float* dU,
    float* dsigma,
    float* dV,
    float lr, float beta1, float beta2, float eps,
    int step_1based,
    float* scratch_rr_U, float* scratch_rr_V,
    float* scratch_etaU, float* scratch_etaV,
    float* scratch_etaS);

// Same as stiefel_adam_step but retracts via Cayley fast-path (3 SGEMMs +
// 1 axpy instead of cuSOLVER QR).  4–6× cheaper per step; drift is
// O(‖η‖³) per call, so the caller should invoke stiefel_retract_qr every
// ~50 steps (with zero η) to clamp accumulated drift.
void stiefel_adam_step_cayley(
    GpuStiefelWeight& stiefel,
    float* dU,
    float* dsigma,
    float* dV,
    float lr, float beta1, float beta2, float eps,
    int step_1based,
    float* scratch_rr_U, float* scratch_rr_V,
    float* scratch_etaU, float* scratch_etaV,
    float* scratch_etaS);

// ========================================================================
// QR retraction: U ← qf(U + η_U) (and same for V). Uses cuSOLVER sgeqrf
// + sorgqr. η_U, η_V must be provided in FP32 (typically the Adam step).
// Σ is updated via Σ ← Σ ⊙ exp(η_Σ / Σ) (Fisher–Rao exp on R_+^r).
// ========================================================================
void stiefel_retract_qr(
    GpuStiefelWeight& stiefel,
    const float* eta_U,      // [m × r]
    const float* eta_sigma,  // [r]
    const float* eta_V);     // [n × r]

// ========================================================================
// Cayley retraction fast-path. Used for intermediate Adam steps between
// full QR refreshes; preserves Stiefel to O(‖η‖³) via
//     U_new = (I + ½A)(I − ½A)^{-1} U
// where A = U η^T − η U^T. Reduces retraction cost from O(mr²) to two
// GEMMs (plus one small r×r system solve).
// ========================================================================
void stiefel_retract_cayley(
    GpuStiefelWeight& stiefel,
    const float* eta_U,
    const float* eta_sigma,
    const float* eta_V,
    float* scratch_A_U,   // [m × r]
    float* scratch_A_V);  // [n × r]

// ========================================================================
// Vector transport for Riemannian Adam: given new U, project existing
// momenta m_U into the new tangent space via modified Gram-Schmidt
// against U. (Standard Absil–Mahony–Sepulchre §8.1.)
// ========================================================================
void stiefel_vector_transport(
    GpuStiefelWeight& stiefel,
    float* scratch_r_r);  // [r × r]

// ========================================================================
// Periodic orthogonality check. Computes ‖U^T U − I_r‖_F and writes to
// stiefel.orthogonality_drift. Caller decides to re-QR if drift > τ.
// ========================================================================
void stiefel_check_orthogonality(GpuStiefelWeight& stiefel);

// ========================================================================
// Debug / parity helper: reconstruct the dense W = U · diag(Σ) · V^T into
// `W_dense` [m × n] FP32. Used by parity tests to compare against the
// unconstrained path. Not called on the hot training path.
// ========================================================================
void stiefel_reconstruct_dense(const GpuStiefelWeight& stiefel, float* W_dense);

// ========================================================================
// Initialize a Stiefel weight from an existing dense matrix via truncated
// SVD.  W_dense [m × n] FP32 → (U [m × r], Σ [r], V [n × r]).  The top-r
// singular values/vectors are kept; the tail is discarded.  For r ≥ min(m,n)
// this is an exact factorization (up to numerical precision).
//
// Used by trainer wire-in (Phase 2h) to convert a random-initialized dense
// weight tensor into Stiefel form on the fly, preserving initialization
// statistics of the underlying architecture's random init.
//
// Computes the SVD on the GPU via cuSOLVER `cusolverDnSgesvdj` (Jacobi SVD,
// best for small m, n) or the standard `gesvd` depending on size.
// Returns true on success, false on cuSOLVER error.
// ========================================================================
bool stiefel_init_from_dense(GpuStiefelWeight& stiefel,
                             const float* W_dense);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

// Non-CUDA stubs: the Stiefel path is GPU-only.
struct GpuStiefelWeight {
	unsigned int m, n, r;
	GpuStiefelWeight() : m(0), n(0), r(0) {}
	void allocate(unsigned int, unsigned int, unsigned int) {}
	void release() {}
	bool allocated() const { return false; }
};

#endif // GLADES_HAVE_CUDA

} // namespace glades
