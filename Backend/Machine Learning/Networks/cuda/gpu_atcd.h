// GPU primitives for ATC-Δ — Activation Temporal Cache with Delta updates
// (paradigm shift #26).  See research/PARADIGM_SHIFT_26_DESIGN.md.
//
// ATC-Δ exploits cross-step forward continuity: since Adam updates satisfy
// ‖ΔW‖_F / ‖W‖_F ~ 1e-4, activations at step t+1 are first-order predictable
// from activations at step t.  Instead of recomputing the full matmul
// h_ℓ^t = σ(W_ℓ^t · h_{ℓ-1}^t) every step, ATC-Δ caches h_ℓ^{t_0} at a
// refresh step and uses a first-order Taylor expansion for K-1 subsequent
// steps:
//
//     h_ℓ^t ≈ h_ℓ^{t_0} + σ'(z_ℓ^{t_0}) ⊙ ( (U_ℓ V_ℓ^T) · h_{ℓ-1}^{t_0}
//                                           + W_ℓ^{t_0} · Δh_{ℓ-1}^t )
//
// where (U_ℓ, V_ℓ) is the rank-r factor of accumulated ΔW across K steps.
//
// Per-matmul FLOP cost drops from O(T·d²) to O(T·d·r) via thin GEMMs.
// At d=1024, r=8: 64× per-matmul speedup.  Amortized over K=8: 7.2× total
// forward speedup.  Memory: cache shared across K steps gives ~6×
// activation-memory reduction.
//
// Phase 1 (this file): core primitives for drift detection and Taylor-
// reconstructed forward.  Phase 2 will add streaming rSVD for the (U, V)
// factor and Phase 3 wires into chiron_train behind --atc-delta.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// atcd_drift_norm — fused reduction for Taylor-drift measurement.
//
//     ratio = ‖Δh‖_F / (‖h_cache‖_F + ε_denom)
//
// Used at end of each Taylor forward to decide whether to force refresh.
// Single scalar output; two parallel sum-of-squares reductions fused
// into one block-scan kernel.
//
// Inputs:
//   delta_h      [T × d]     device row-major FP32
//   h_cache      [T × d]     device row-major FP32
//   T, d                     dims
//   eps_denom                numerical floor on ‖h_cache‖_F (default 1e-12f)
// Output:
//   ratio_out                device scalar (float); writes the ratio once
//
// Cost: O(T·d).  Grid = 1 block since the output is a scalar.
// ========================================================================
bool atcd_drift_norm(const float* delta_h, const float* h_cache,
                     unsigned int T, unsigned int d,
                     float eps_denom,
                     float* ratio_out);

// ========================================================================
// atcd_taylor_weight_delta — the weight-delta-only Taylor term.
//
//     Δz = (U · Vᵀ) · h_cache        (thin GEMMs, not explicit U Vᵀ)
//     Δh = σ'_cache ⊙ Δz
//
// This computes the primary Taylor term WITHOUT the propagated-delta
// contribution (W_cache · Δh_input).  Full Taylor is:
//     Δh_out = Δh_weight (this kernel) + Δh_propagated (separate)
//
// We split because (a) the propagated term reduces to a thin matmul
// when Δh_input is low-rank, which is the expected fast path, and
// (b) at refresh-step-0 when Δh_input = 0, only this term is needed.
//
// Inputs:
//   U           [d_out × r]      row-major FP32
//   V           [d_in × r]       row-major FP32
//   h_cache     [T × d_in]       row-major FP32 (cached input activations)
//   sigma_prime [T × d_out]      row-major FP32 (cached activation derivative)
//   T, d_in, d_out, r            dims
// Outputs:
//   delta_z_scratch  [T × d_out] row-major FP32 — Δz = U·(Vᵀ·h_cache)
//                                Scratch: caller allocates.
//   delta_h_out      [T × d_out] row-major FP32 — Δh = σ' ⊙ Δz
//
// Implementation:
//   Step 1: tmp_r  [r × T]   = Vᵀ · h_cacheᵀ     (cost 2·r·d_in·T)
//   Step 2: Δz     [d_out×T] = U · tmp_r          (cost 2·d_out·r·T)
//   Step 3: Δh     [T×d_out] = σ'_cache ⊙ Δz
// ========================================================================
bool atcd_taylor_weight_delta(const float* U, const float* V,
                              const float* h_cache,
                              const float* sigma_prime,
                              unsigned int T, unsigned int d_in,
                              unsigned int d_out, unsigned int r,
                              float* delta_z_scratch,
                              float* delta_h_out);

// ========================================================================
// atcd_cache_refresh — D2D copies to snapshot h, z, σ' at a refresh step.
//
// Called at step t_0 after the exact forward pass has been computed.
// Copies are explicitly fused to a single stream to avoid sync overhead.
//
// Inputs:
//   h_full        [T × d]   row-major FP32 — exact h at refresh
//   z_full        [T × d]   row-major FP32 — exact pre-activation z
// Outputs:
//   h_cache       [T × d]   destination — h_cache[l] buffer
//   z_cache       [T × d]   destination — z_cache[l] buffer
//   sigma_prime   [T × d]   destination — σ'(z_full) evaluated elementwise
//   T, d                    dims
//   activation_kind         0 = GELU, 1 = SiLU, 2 = ReLU (unsupported),
//                           3 = identity
//
// Cost: 3·T·d elementwise (D2D + σ').
// ========================================================================
bool atcd_cache_refresh(const float* h_full, const float* z_full,
                        float* h_cache, float* z_cache, float* sigma_prime,
                        unsigned int T, unsigned int d,
                        int activation_kind);

// ========================================================================
// atcd_extract_rank1_power — top-1 SVD of ΔW via power iteration.
//
// Given a dense weight-update matrix ΔW ∈ ℝ^{d_in × d_out} (row-major),
// extract the dominant singular triple (σ, u, v) such that
//     ΔW ≈ σ · v · u^T           (outer product, as used by ATC-Δ factor)
// where u ∈ ℝ^{d_out}, v ∈ ℝ^{d_in}, both unit-norm; σ ∈ ℝ_{≥0}.
//
// Power iteration: init v randomly (or with prior v); iterate
//   u ← ΔW^T v / ‖·‖;  v ← ΔW u / ‖·‖
// for n_iters steps.  Converges to the top singular pair at rate
// σ_2/σ_1 per iter; typically 2-3 iters suffice for dominant extraction.
//
// Inputs:
//   dW            [d_in × d_out]  row-major FP32
//   d_in, d_out                   dims
//   n_iters                       number of power iterations (typical 3)
//   v_init        [d_in]          initial vector (may be NULL → use ones)
// Outputs:
//   u_out         [d_out]          dominant right singular vector (unit norm)
//   v_out         [d_in]           dominant left singular vector (unit norm)
//   sigma_out                      device scalar: σ = ‖ΔW·v‖ at convergence
//
// The factor for ATC-Δ is then V[:, slot] = v_out, U[:, slot] = σ · u_out.
//
// Cost: (2 · n_iters + 1) · d_in · d_out ≈ 7·d² at n_iters=3.  At d=1024
// that's 7 MFLOPs per layer per step — 0.15% of forward baseline 4 GFLOPs.
// ========================================================================
bool atcd_extract_rank1_power(const float* dW,
                              unsigned int d_in, unsigned int d_out,
                              int n_iters,
                              const float* v_init,
                              float* u_out, float* v_out, float* sigma_out);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool atcd_drift_norm(const float*, const float*,
                            unsigned int, unsigned int,
                            float, float*) { return false; }
inline bool atcd_taylor_weight_delta(const float*, const float*,
                                     const float*, const float*,
                                     unsigned int, unsigned int,
                                     unsigned int, unsigned int,
                                     float*, float*) { return false; }
inline bool atcd_cache_refresh(const float*, const float*,
                               float*, float*, float*,
                               unsigned int, unsigned int, int) { return false; }
inline bool atcd_extract_rank1_power(const float*,
                                     unsigned int, unsigned int,
                                     int, const float*,
                                     float*, float*, float*) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
