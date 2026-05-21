// IGAA (Information-Geometric Attention Augmentation) GPU primitives —
// paradigm #260.  Iter 25-26 (2026-05-16) Gate-0 prototype.
//
// See research/PARADIGM_SHIFT_260_IGAA_DESIGN.md for the full design.
//
// Mechanism (per-token, per-IGAA-layer):
//   z_i      = W_pi · x_i + b_pi                 [K]
//   pi_i     = softmax(z_i / tau)                [K]
//   Δy_i     = Σ_k (pi_{i,k} − 1/K) · θ^(k)_i    [m]
//   y'_i     = y_SCFA_i + α · Δy_i
//
// At init (W_pi ≈ 0, b_pi = 0, τ = 1, α = 0): π_i = uniform → Δy ≡ 0.
// Bit-exact baseline preservation.
//
// Gate-0 prototype: K modes with mode IDs ∈ {1, 2, 3, 4}:
//   1 = SCFA-full      (θ^(k) = s.p, the existing SCFA attention output)
//   2 = SCFA-tight     (placeholder = SCFA-full in iter 26)
//   3 = banded-short   (placeholder = SCFA-full in iter 26)
//   4 = identity       (θ^(k) = 0; no attention contribution)
//
// For K modes where modes m_scfa of them have mode_id ∈ {1, 2, 3} and the
// rest are identity, the centered Δy reduces to:
//   Δy_i = (Σ_{k : scfa-mode} pi_{i,k} − m_scfa/K) · s.p_i
// because all SCFA-mode outputs are identical and identity contributes 0.
//
// igaa_apply_gate computes this in-place on s.p, after the SCFA forward.

#pragma once

#include <cstdint>

#ifdef GLADES_HAVE_CUDA
namespace glades {
namespace gpu {

// Compute IGAA forward correction in-place on the SCFA output buffer p.
//   x      : [T, m]      — layer input residual stream (read-only)
//   p      : [T, m]      — SCFA attention output (read AND written in-place)
//   W_pi   : [K, m]      — mixture logit projection (read-only)
//   b_pi   : [K]         — mixture logit bias (read-only)
//   tau    : [1]         — softmax temperature (read-only scalar)
//   alpha  : [1]         — gate scalar (read-only scalar)
//   pi_buf : [T, K]      — scratch buffer for π (written then read; size T*K floats)
//   T      : sequence length
//   m      : residual width
//   K      : number of mixture modes
//   m_scfa : number of modes with output == p (currently: K - count(mode_id==4))
//
// Steps performed:
//   1) Z = X · W_pi^T + b_pi          (T × K via sgemm + bias-add)
//   2) Pi = softmax(Z / τ) row-wise   (T × K)
//   3) gate_i = Σ_{k ∈ scfa-mode-indices} (Pi[i,k] − 1/K)
//   4) p[i, :] += α · gate_i · p[i, :]
//
// Returns true on success; false on launch failure or invalid args.
//
// Because Pi has only K columns (K ≤ 16 in design), step 2-4 fit comfortably
// in shared memory and a single block per token.  Cost: O(TmK) for the GEMM
// (negligible at K ≤ 16, m=2048) + O(Tm) for the per-token scale.
bool igaa_apply_gate(const float* x,
                     float*       p,
                     const float* W_pi,
                     const float* b_pi,
                     const float* tau,
                     const float* alpha,
                     float*       pi_buf,
                     int          T,
                     int          m,
                     int          K,
                     int          m_scfa);

// Backward pass for IGAA.  Inputs/outputs:
//   IN  dp_out   [T, m] — gradient of loss wrt the IGAA-modified p (= s.p
//                          after forward).  This is OVERWRITTEN in place
//                          with dL/dp_in by the formula
//                              dp_in[i,d] = dp_out[i,d] · (1 + scale[i])
//                          where scale[i] is recovered from pi_buf's tail.
//   IN  p_save   [T, m] — the pre-scale p (saved by the forward at the same
//                          layer when igaaTrain is on).
//   IN  pi_buf   [T*K + T] — written by the forward.  Layout:
//                              pi_buf[0..T*K]      = π (softmax output)
//                              pi_buf[T*K..T*K+T]  = scale (= α · gate_pure)
//                          read-only here.
//   IN  x        [T, m] — original layer input (= s.q at forward time).
//   IN  W_pi     [K, m] — mixture logit weights.
//   IN  alpha    [1]    — current scalar gate value.
//   IN  tau      [1]    — softmax temperature (we don't compute dτ in Gate-0;
//                          τ stays fixed).
//   OUT dW_pi    [K, m] — ACCUMULATES (+=) into gradient buffer.
//   OUT db_pi    [K]    — ACCUMULATES.
//   OUT dalpha   [1]    — ACCUMULATES.
//   OUT dx       [T, m] — ACCUMULATES (+= dz · W_pi for backprop through input).
//
//   T, m, K, m_scfa: same as forward.
//
// Computes (per-token, per-mode):
//   dG[i]      = Σ_d dp_out[i,d] · p_save[i,d]                          [T]
//   dalpha    += Σ_i dG[i] · gate_pure[i]                                [1]
//                where gate_pure[i] = Σ_{k<m_scfa} (π[i,k] − 1/K) = scale[i]/α
//   dπ[i,k]    = α · dG[i]   for k < m_scfa, else 0                     [T, K]
//   dz[i,k]    = (π[i,k]/τ) · (dπ[i,k] − Σ_j π[i,j] · dπ[i,j])           [T, K]
//   db_pi[k]  += Σ_i dz[i,k]                                              [K]
//   dW_pi[k,d]+= Σ_i dz[i,k] · x[i,d]                                     [K, m]
//   dx[i,d]   += Σ_k dz[i,k] · W_pi[k,d]                                  [T, m]
//   dp_in[i,d] = dp_out[i,d] · (1 + scale[i])                             [T, m] (in place)
bool igaa_backward(float*       dp,          // [T, m] in/out (dp_out → dp_in)
                   const float* p_save,      // [T, m]
                   const float* pi_buf,      // [T*K + T]
                   const float* x,           // [T, m]
                   const float* W_pi,        // [K, m]
                   const float* alpha,       // [1]
                   const float* tau,         // [1]
                   float*       dW_pi,       // [K, m] in/out (accumulate)
                   float*       db_pi,       // [K]    in/out (accumulate)
                   float*       dalpha,      // [1]    in/out (accumulate)
                   float*       dx,          // [T, m] in/out (accumulate)
                   float*       scratch,     // [T*K + T] (dπ buffer + dG buffer)
                   int          T,
                   int          m,
                   int          K,
                   int          m_scfa);

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades { namespace gpu {
inline bool igaa_apply_gate(const float*, float*, const float*, const float*,
                            const float*, const float*, float*,
                            int, int, int, int)
{ return false; }
inline bool igaa_backward(float*, const float*, const float*, const float*,
                          const float*, const float*, const float*,
                          float*, float*, float*, float*, float*,
                          int, int, int, int)
{ return false; }
}}

#endif
