// GPU primitives for TRCD — Token-Routed Conditional Depth (paradigm shift #13).
// See research/PARADIGM_SHIFT_13_CANDIDATE_C_TRCD.md and research/PARADIGM_SHIFT_13_SELECTION.md.
//
// Per-token dynamic depth: each token traverses a data-dependent number
// of layers d(t) ∈ {1, ..., L}, chosen at each layer by a routing policy
// trained against a bilevel KKT-tuned FLOP budget.  When token t exits at
// layer l, its representation is fed directly to a shared early-exit head.
//
// Per-layer router (linear):
//
//     u_{l, t} = a_l · h_{l, t} + b_l        (scalar, per-token, per-layer)
//
// Continue-vs-exit Gumbel-softmax gate:
//
//     α_{l, t} = σ( (u_{l, t} − λ + g) / τ )  training  (soft, differentiable)
//     d(t) = l  iff  α_{l, t} < 0.5 eval       (hard threshold, deterministic)
//
// where g ~ Gumbel(0, 1) is the standard Gumbel sample and τ is the
// temperature (annealed τ_start → τ_end over training).
//
// The Lagrangian λ is updated by a PI controller to enforce
// E_t[d(t)] ≤ d̄_target.
//
// Phase 1 (this file): core primitives.  Phase 2: prefix-sum bucketing
// for dynamic batching.  Phase 3: CHIRON trainer + pile_train wire-in.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>
#include <stdint.h>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// trcd_route_logits — per-token routing logit u_{l, t} = a_l · h_{l, t} + b_l
//
//   h_in       [T × d]   hidden states at layer l entry (row-major)
//   a_l        [d]       learned row vector
//   b_l        scalar    learned bias
//   u_out      [T]       per-token continue-utility
//
// This is a fused dot-product-plus-bias reduction across d; implemented
// as a block-per-token kernel with warp reductions (d ≤ 8192 typical).
// ========================================================================
bool trcd_route_logits(const float* h_in, const float* a_l, float b_l,
                       unsigned int T, unsigned int d,
                       float* u_out);

// ========================================================================
// trcd_route_logits_backward — gradients for (a_l, b_l) and dh_in given
// dU = dL/dU ∈ ℝ^T.
//
//   dU         [T]       upstream gradient on the utility vector
//   h_in       [T × d]   forward-pass activations
//   a_l        [d]       forward-pass a_l (needed for dh_in)
//   gA_out     [d]       accumulated into  (gA += dU^T · h_in)
//   gB_out     scalar    accumulated into  (gB += sum(dU))   (pass device ptr)
//   dh_out     [T × d]   written  (dh[t, :] = dU[t] · a_l)  — NULL to skip
//
// Any of gA_out/gB_out/dh_out may be NULL to skip that grad.
// ========================================================================
bool trcd_route_logits_backward(const float* dU, const float* h_in,
                                const float* a_l,
                                unsigned int T, unsigned int d,
                                float* gA_out, float* gB_out,
                                float* dh_out);

// ========================================================================
// trcd_gumbel_gate — compute per-token continue-vs-exit gate.
//
//   u          [T]       routing logits (from trcd_route_logits)
//   lambda     scalar    current Lagrangian (CPU value)
//   tau        scalar    Gumbel-softmax temperature
//   seed       uint64    RNG seed for this batch/step (Gumbel samples)
//   training   bool      if true: α = σ((u − λ + g) / τ) with g ~ Gumbel(0,1)
//                         if false: α = 1 iff u > λ else 0 (deterministic)
//   alpha_out  [T]       per-token continue probability (∈ [0, 1])
//
// Straight-through gradient: forward uses soft α; backward uses soft α's
// gradient w.r.t. u (via the logistic derivative) — this lives in the
// caller's backward graph.
// ========================================================================
bool trcd_gumbel_gate(const float* u, float lambda, float tau,
                      uint64_t seed, bool training,
                      unsigned int T,
                      float* alpha_out);

// ========================================================================
// trcd_apply_gate — mask a hidden-state tensor by a per-token gate.
//
//   h_in       [T × d]   incoming hidden states (row-major)
//   alpha      [T]       continue probability (from trcd_gumbel_gate)
//   h_out      [T × d]   h_out[t, :] = alpha[t] * h_in[t, :]
//
// Used as the "soft" path: the continuation of a token through layer l
// is its post-layer representation scaled by α_{l, t}.  Tokens that exit
// contribute (1 − α_{l, t}) of their post-norm hidden state to the
// shared early-exit head.
// ========================================================================
bool trcd_apply_gate(const float* h_in, const float* alpha,
                     unsigned int T, unsigned int d,
                     float* h_out);

// ========================================================================
// trcd_apply_gate_backward — gradients for (h_in, alpha) given dh_out.
//
//   dh_out     [T × d]   upstream grad on h_out
//   h_in       [T × d]   forward activations
//   alpha      [T]       forward gate
//   dh_in_out  [T × d]   written (dh_in[t, :] = alpha[t] * dh_out[t, :])
//   dalpha_out [T]       written (dalpha[t]   = sum_j h_in[t, j] * dh_out[t, j])
//
// Both output grads may be NULL to skip.
// ========================================================================
bool trcd_apply_gate_backward(const float* dh_out, const float* h_in,
                              const float* alpha,
                              unsigned int T, unsigned int d,
                              float* dh_in_out, float* dalpha_out);

// ========================================================================
// trcd_lambda_pi_update — host-side PI controller on the FLOP budget.
//
// Given observed mean depth d̄_obs over the recent window and target
// d̄_target, update the Lagrangian λ:
//
//   error_t  = d̄_obs − d̄_target
//   integral += error_t                     (controller state — carry across calls)
//   λ_new    = λ_old + kp · error_t + ki · integral
//   λ_new    = clamp(λ_new, 0, λ_max)       (λ ≥ 0; positive cost of depth)
//
// Inputs:
//   d_bar_obs, d_bar_target   scalar, dimensionless
//   kp, ki                    PI gains (typical: kp = 0.1, ki = 0.01)
//   integral_inout            running integral (read + write)
//   lambda_inout              current lambda (read + write)
//   lambda_max                hard upper bound on λ (typical: L = depth)
// ========================================================================
void trcd_lambda_pi_update(float d_bar_obs, float d_bar_target,
                           float kp, float ki,
                           float& integral_inout, float& lambda_inout,
                           float lambda_max);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool trcd_route_logits(const float*, const float*, float,
                              unsigned int, unsigned int,
                              float*) { return false; }
inline bool trcd_route_logits_backward(const float*, const float*,
                                       const float*,
                                       unsigned int, unsigned int,
                                       float*, float*, float*) { return false; }
inline bool trcd_gumbel_gate(const float*, float, float, uint64_t, bool,
                             unsigned int, float*) { return false; }
inline bool trcd_apply_gate(const float*, const float*,
                            unsigned int, unsigned int,
                            float*) { return false; }
inline bool trcd_apply_gate_backward(const float*, const float*,
                                     const float*,
                                     unsigned int, unsigned int,
                                     float*, float*) { return false; }
inline void trcd_lambda_pi_update(float, float, float, float,
                                  float&, float&, float) { }

#endif // GLADES_HAVE_CUDA

} // namespace glades
