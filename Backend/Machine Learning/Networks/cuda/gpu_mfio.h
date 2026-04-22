// GPU primitives for MFIO — Moment-Free Implicit Optimizer (paradigm
// shift #11).  See research/PARADIGM_SHIFT_11_DESIGN.md.
//
// Replaces Adam's per-parameter (m, v) state with a deterministic step
// using a per-layer preconditioner σ_ℓ derived from activation norms
// that CHIRON already materializes on the backward pass:
//
//     ŝ_ℓ = (1/B) Σ_i ‖z_ℓ^(i)‖² · ‖δ_ℓ^(i)‖²      (one FP32 scalar/layer)
//     σ_ℓ = 1 / (√(ŝ_ℓ · β_schedule(t)) + ε)
//     θ_ℓ ← θ_ℓ − η · σ_ℓ · g_ℓ
//
// The √ in σ matches Adam's 1/√v preconditioner: 1/ŝ alone would
// scale like 1/|g|² (catastrophic overshoot near the optimum), while
// 1/√ŝ scales like 1/|g| (Adam-compatible normalization).
//
// **ZERO per-parameter optimizer state.**  The only extra memory is
// one scalar per layer (ŝ_ℓ).  At 30 B params this replaces 60 GB of
// int8-packed Adam state with ~L bytes = 400 bytes — a paradigm-shift-
// level reduction (>10^8× on the optimizer-state axis).
//
// Phase 1 (this file): two standalone primitives — compute_sigma from
// (z, δ) reduction, and the moment-free weight update.  Phase 2 will
// add the end-to-end trainer wire-in behind a --mfio flag.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// mfio_compute_sigma — fused reduction computing the per-layer
// preconditioner from activation statistics.
//
//     ŝ = (1/T) · Σ_t ‖z[t, :]‖² · ‖δ[t, :]‖²
//     σ = 1 / (√(ŝ · beta_schedule) + eps)
//
// Inputs:
//   z     [T × d_in]     row-major pre-activations
//   delta [T × d_out]    row-major gradient-of-output
//   T, d_in, d_out       dims
//   beta_schedule        scalar — learning-rate warmup/decay factor
//                        (replaces Adam's 1/(1-β_1^t))
//   eps                  numerical floor on ŝ (default 1e-8)
// Output:
//   sigma_out            device scalar (float), zeroed internally
//
// Cost: O(T · (d_in + d_out)).  Two block-reduced passes fused into
// a single kernel; grid = 1 block since σ is a single scalar.
// ========================================================================
bool mfio_compute_sigma(const float* z, const float* delta,
                        unsigned int T, unsigned int d_in, unsigned int d_out,
                        float beta_schedule,
                        float eps,
                        float* sigma_out);

// ========================================================================
// mfio_update — moment-free weight step:
//     θ[i] ← θ[i] − η · σ · g[i]            for i in [0, n_params)
// where σ is the device scalar produced by mfio_compute_sigma.
//
// Optional weight decay (AdamW-style): θ[i] ← θ[i] · (1 − η·wd) − η·σ·g[i].
// Pass wd = 0.0f to disable.
//
// Inputs:
//   theta      [n_params]    weight tensor, row-major/flat
//   g          [n_params]    gradient tensor, same shape
//   sigma      device scalar (from mfio_compute_sigma)
//   lr         learning rate (η)
//   wd         weight decay
//   n_params   parameter count
// ========================================================================
bool mfio_update(float* theta, const float* g,
                 const float* sigma,
                 float lr, float wd,
                 int n_params);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool mfio_compute_sigma(const float*, const float*,
                               unsigned int, unsigned int, unsigned int,
                               float, float, float*) { return false; }
inline bool mfio_update(float*, const float*, const float*,
                        float, float, int) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
