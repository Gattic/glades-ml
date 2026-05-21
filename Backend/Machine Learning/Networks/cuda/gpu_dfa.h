// GPU primitives for DFA — Direct Feedback Alignment (paradigm shift #12).
// See research/PARADIGM_SHIFT_12_DESIGN.md.
//
// Replaces backpropagation's exact transposed-weight gradient path with
// a FIXED RANDOM projection of the OUTPUT ERROR to each layer's hidden
// space.  Per-layer updates are local:
//
//     e_global    = dL/dY_final        [T × d_out_final]  (known at output)
//     e_layer_l   = e_global · R_l     [T × d_l]          (R_l fixed random)
//     dW_l        = X_l^T · e_layer_l  [d_{l-1} × d_l]
//     θ_l        ← θ_l − η · dW_l      (+ MFIO / Adam normalization)
//
// No true-gradient flow through earlier layers.  Compute per step:
// 1 forward pass + 1 random-projection broadcast of e_global per layer
// = ~1.1× forward-only cost (vs the ~3× cost of forward + full backward).
//
// The fixed random R_l matrices are ~O(d_out_final · d_l) per layer —
// small compared to the weights themselves, and never updated.  Storage
// is one-time, read-only across all of training.
//
// Phase 1 (this file): primitive kernels — random-matrix init and
// error-projection.  Phase 2 will add the content-aware variant for
// attention blocks.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>
#include <stdint.h>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// dfa_init_random_matrix — fills R ∈ R^{rows × cols} with uniform-random
// values in [−scale, +scale] from a simple counter-based RNG.
//
// Called ONCE at trainer init per layer.  The matrix is FIXED for the
// remainder of training — do NOT regenerate between steps.  Seed must
// be unique per (layer, training run) to avoid aliasing.
//
// Scale recommendation: 1/√cols (Xavier-like) keeps projected-error
// variance bounded at O(1) regardless of layer width.
// ========================================================================
bool dfa_init_random_matrix(float* R_out,
                            unsigned int rows, unsigned int cols,
                            uint64_t seed,
                            float scale);

// ========================================================================
// dfa_project_error — broadcasts the output-error signal to a layer's
// feature space using the layer's fixed backward matrix R_l.
//
//     e_proj[t, :]  = e[t, :] · R         [T × d_hid]
//                                         |R is [d_out × d_hid]|
//
// Equivalent to sgemm_rowmajor(T, d_hid, d_out, e, R, e_proj) — this
// wrapper is a thin call to cuBLAS through the existing sgemm helpers,
// named for clarity in DFA-specific code paths.
//
// Output e_proj is the layer's "pseudo-gradient-of-output" signal,
// drop-in for dY in the standard dW = X^T · dY computation.
// ========================================================================
bool dfa_project_error(const float* e, const float* R,
                       unsigned int T,
                       unsigned int d_out, unsigned int d_hid,
                       float* e_proj_out);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool dfa_init_random_matrix(float*, unsigned int, unsigned int,
                                   uint64_t, float) { return false; }
inline bool dfa_project_error(const float*, const float*,
                              unsigned int, unsigned int, unsigned int,
                              float*) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
