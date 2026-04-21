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

	// Riemannian Adam first moments (int8 asymmetric packed).
	GpuBuffer<int8_t>  m_U;     // [m * r]
	GpuBuffer<float>   m_U_scale;  // per-tensor scale
	GpuBuffer<float>   m_sigma; // [r] FP32 (small, kept dense)
	GpuBuffer<int8_t>  m_V;     // [n * r]
	GpuBuffer<float>   m_V_scale;

	// Second moments (uint8 unsigned for v; asymmetric).
	GpuBuffer<uint8_t> v_U;     // [m * r]
	GpuBuffer<float>   v_U_scale;
	GpuBuffer<float>   v_sigma; // [r] FP32
	GpuBuffer<uint8_t> v_V;     // [n * r]
	GpuBuffer<float>   v_V_scale;

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
