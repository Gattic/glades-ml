// GPU primitives for SPAREC — Sparse Post-Activation-derivative REweighted
// Coordinate gradient (paradigm shift #35).  See
// research/PARADIGM_SHIFT_35_DESIGN.md and candidate A doc.
//
// SPAREC skips rows of ∂L/∂W_up and cols of ∂L/∂h_in in the FFN backward
// pass where |σ'(x[t,i])| ≤ τ.  An adaptive threshold τ is tuned by an
// integral controller to maintain target backward sparsity ρ_target.
//
// Phase 1 primitives (this header):
//   1. sparec_compute_active_mask — σ'(x) → {M, active_idx, k_t, ρ_observed}
//   2. sparec_update_threshold    — τ controller step
//   3. sparec_backward_gathered   — gathered-sparse SGEMM for dL/dW_up and dL/dh_in
//
// Parity tests:
//   CHIRONSparecMaskParityTest   — kernel vs host reference at τ=1e-6 (≈dense)
//   CHIRONSparecBackwardParity   — gathered backward vs dense at τ=1e-6 (bit-exact)
//   CHIRONSparecControllerTest    — τ tracks ρ_target under fixed distribution

#pragma once

#include "gpu_buffer.h"
#include <cstddef>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// sparec_compute_active_mask — build (M, active_idx, k_t, ρ_observed) from
// a forward σ'(x) cache tensor.
//
// Given σ'_cache ∈ ℝ^{T × d_ff} row-major FP32 (per-token, per-neuron
// derivative values from the FFN forward), compute:
//
//   M[t, i]        = 1 if |σ'_cache[t, i]| > τ, else 0    (packed bit mask)
//   active_idx[t]  = compact list of active neuron indices per token
//   k_per_tok[t]   = active count per token
//   rho_observed   = 1 − (Σ_t k_per_tok[t]) / (T · d_ff)  (scalar)
//
// Implementation: one CUDA block per token, block-scan prefix-sum to
// build active_idx[t] and k_per_tok[t] in a single kernel.  Shared
// memory: O(d_ff) — fits d_ff ≤ 16384 at 64 KB.
//
// Inputs:
//   sigma_prime_cache  [T × d_ff]   row-major FP32
//   T, d_ff                         dims
//   tau                             scalar threshold
// Outputs:
//   mask_packed        [T × ⌈d_ff/32⌉]   packed bit mask (uint32, one bit per (t,i))
//   active_idx         [T × d_ff]   per-token compact list (padded to d_ff; use k_per_tok[t])
//   k_per_tok          [T]          active count per token
//   rho_observed_out               device scalar (FP32) — fraction inactive
//
// Cost: O(T · d_ff).  One kernel launch.
// ========================================================================
bool sparec_compute_active_mask(const float* sigma_prime_cache,
                                unsigned int T, unsigned int d_ff,
                                float tau,
                                unsigned int* mask_packed,
                                unsigned int* active_idx,
                                unsigned int* k_per_tok,
                                float* rho_observed_out);

// ========================================================================
// sparec_update_threshold — integral controller step on τ.
//
// Update:
//   τ^{t+1} = clip(τ^t · (1 + η_τ · (ρ_target − ρ_observed^{t+1})),
//                  τ_min, τ_max)
// where ρ_observed^{t+1} = β_rho · ρ_observed^t + (1 − β_rho) · ρ_new.
//
// The ρ_ramp schedule is applied on the host side by passing a time-
// dependent ρ_target to this kernel each step.
//
// Inputs:
//   rho_new        device scalar — fresh sparsity from sparec_compute_active_mask
//   rho_target     float — target sparsity (ramped on host)
//   eta_tau        float — controller gain (default 0.05)
//   tau_min, tau_max   float — clip bounds (default 1e-4, 0.5)
//   beta_rho       float — EMA decay on ρ_observed (default 0.9)
// In/out:
//   tau            device scalar (FP32) — updated in place
//   rho_observed_ema  device scalar (FP32) — updated in place
//
// Cost: O(1).  One block, one launch.
// ========================================================================
bool sparec_update_threshold(float* tau,
                             float* rho_observed_ema,
                             const float* rho_new,
                             float rho_target,
                             float eta_tau,
                             float tau_min, float tau_max,
                             float beta_rho);

// ========================================================================
// sparec_backward_gathered — compute dL/dW_up and dL/dh_in using the
// gathered active-index list from sparec_compute_active_mask.
//
// Given:
//   grad_sigma ∈ ℝ^{T × d_ff} — ∂L/∂σ(x), computed via dense
//     W_down^T · ∂L/∂h_out
//   sigma_prime_cache ∈ ℝ^{T × d_ff} — σ'(x) cache
//   h_in ∈ ℝ^{T × d_model} — FFN input
//   W_up ∈ ℝ^{d_ff × d_model} — up-projection weights (row-major)
//   active_idx, k_per_tok — from sparec_compute_active_mask
//
// Output:
//   grad_W_up ∈ ℝ^{d_ff × d_model} — accumulated ∂L/∂W_up (row-major)
//   grad_h_in ∈ ℝ^{T × d_model}    — ∂L/∂h_in (row-major)
//
// Algorithm:
//   For each t, for each k ∈ [0, k_per_tok[t]):
//     i = active_idx[t, k]
//     scalar = σ'(x[t,i]) · ∂L/∂σ(x)[t, i]
//     grad_W_up[i, :] += scalar · h_in[t, :]
//     grad_h_in[t, :] += scalar · W_up[i, :]
//
// Implementation: gather the active-scalar values into a compact
// workspace of shape [Σ_t k_t × d_model], then dispatch two dense
// cuBLAS SGEMMs (one for W_up grad, one for h_in grad).  Active-scalar
// build fused with gather in one CUDA kernel.
//
// Cost: O(Σ_t k_t · d_model) — for ρ=0.80, ~5× reduction vs dense.
// ========================================================================
bool sparec_backward_gathered(const float* grad_sigma,
                              const float* sigma_prime_cache,
                              const float* h_in,
                              const float* W_up,
                              const unsigned int* active_idx,
                              const unsigned int* k_per_tok,
                              unsigned int T, unsigned int d_ff,
                              unsigned int d_model,
                              float* grad_W_up,
                              float* grad_h_in);

// ========================================================================
// sparec_backward_dense_reference — host-side reference implementation
// of the dense backward (τ=0 equivalent) for parity tests.  Not called
// during training; exists only so the unit test can compare
// sparec_backward_gathered at τ=1e-6 against an exact dense backward.
// Defined in the .cu file (host function).
// ========================================================================
void sparec_backward_dense_reference_cpu(const float* grad_sigma,
                                         const float* sigma_prime_cache,
                                         const float* h_in,
                                         const float* W_up,
                                         unsigned int T, unsigned int d_ff,
                                         unsigned int d_model,
                                         float* grad_W_up,
                                         float* grad_h_in);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool sparec_compute_active_mask(const float*, unsigned int, unsigned int,
                                       float,
                                       unsigned int*, unsigned int*, unsigned int*,
                                       float*) { return false; }
inline bool sparec_update_threshold(float*, float*, const float*,
                                    float, float, float, float, float) { return false; }
inline bool sparec_backward_gathered(const float*, const float*, const float*,
                                     const float*, const unsigned int*,
                                     const unsigned int*,
                                     unsigned int, unsigned int, unsigned int,
                                     float*, float*) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
