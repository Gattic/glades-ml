// GPU primitives for CSP — Compressed-Sensing Proxy FFN (paradigm shift
// #27).  See research/PARADIGM_SHIFT_27_DESIGN.md.
//
// Replaces a dense FFN block
//
//     h_in  → W_up  → σ(·) → W_down → h_out     (d_ff intermediate)
//
// with a JL-sketched three-GEMM path of inner dimension m ≪ d_ff:
//
//     h_in → W'_up → σ̂(·) → W'_down → h_out    (m intermediate)
//
// where σ̂ : ℝ^m → ℝ^m is a learned residual nonlinearity:
//
//     σ̂(y) = σ_base(y) + U_σ · σ_hidden(V_σ^T · y)
//
// with σ_base = GELU / SiLU (elementwise) and a rank-r_σ learned residual
// term via U_σ ∈ ℝ^{m × r_σ}, V_σ ∈ ℝ^{m × r_σ}.  At r_σ = 0 the residual
// vanishes and σ̂ ≡ σ_base — recovers a plain sketched FFN with NO learned
// surrogate (used for the parity baseline).
//
// Phase 1 (this file): csp_forward only.  Phase 2 will add the σ̂ probe
// + target generation kernels.  Phase 3 wires into chiron_train behind
// --csp flag.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// csp_forward — full CSP-sketched FFN forward pass.
//
//     z_sketch = h_in · W_up                       (T × m)
//     y_base   = σ_base(z_sketch)                   (T × m)
//     y_res    = σ_hidden(z_sketch · V_σ)          (T × r_σ)  [if r_σ > 0]
//     y_res    = y_res · U_σ^T                     (T × m)
//     y        = y_base + y_res                    (T × m)
//     h_out    = y · W_down                         (T × d_model)
//
// Uses cuBLAS sgemm_rowmajor for all GEMMs.  Elementwise σ_base via
// the existing gelu_forward / silu_forward kernels.
//
// Inputs:
//   h_in         [T × d_model]   row-major FP32
//   W_up         [d_model × m]   row-major FP32 (compressed up-proj)
//   W_down       [m × d_model]   row-major FP32 (compressed down-proj)
//   U_sigma      [m × r_sigma]   row-major FP32 (may be NULL if r_sigma = 0)
//   V_sigma      [m × r_sigma]   row-major FP32 (may be NULL if r_sigma = 0)
//   sigma_kind                    0=GELU, 1=SiLU, 3=identity
//   T, d_model, m, r_sigma        dims
// Scratches (caller-allocated):
//   z_sketch_scratch  [T × m]
//   y_scratch         [T × m]
//   y_res_scratch     [T × max(m, r_sigma)]  (only needed if r_sigma > 0)
// Output:
//   h_out        [T × d_model]
//
// Cost: 2 · T · m · d_model + 2 · T · d_model · m + (2·T·m·r_σ + 2·T·r_σ·m
//       when r_sigma > 0).  At m = d_ff/4, saves ~4× FLOPs vs dense FFN.
// ========================================================================
bool csp_forward(const float* h_in,
                 const float* W_up, const float* W_down,
                 const float* U_sigma, const float* V_sigma,
                 unsigned int T, unsigned int d_model,
                 unsigned int m, unsigned int r_sigma,
                 int sigma_kind,
                 float* z_sketch_scratch,
                 float* y_scratch,
                 float* y_res_scratch,
                 float* h_out);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool csp_forward(const float*, const float*, const float*,
                        const float*, const float*,
                        unsigned int, unsigned int, unsigned int, unsigned int,
                        int, float*, float*, float*, float*) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
