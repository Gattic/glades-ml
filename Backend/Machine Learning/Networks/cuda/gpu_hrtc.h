// GPU primitives for Hierarchical Reversible Token Compression (HRTC) —
// paradigm shift #8.  See research/HRTC_DESIGN.md for the framework.
//
// Partitions a sequence X [T × m] into non-overlapping blocks of k tokens
// and applies a bit-exact orthogonal transform to produce:
//   - T/k super-tokens (the "compressed" sequence the transformer stack runs on)
//   - T·(k-1)/k residuals (stored but not processed through the middle layers)
//
// At k=2 the transform is a single-level Haar wavelet:
//   super[i]    = (X[2i] + X[2i+1]) / sqrt(2)
//   residual[i] = (X[2i] - X[2i+1]) / sqrt(2)
//
// Round-trip (pool → unpool) is exact to FP32 precision (~1e-7).  This is
// sufficient for CHIRON's reversible inverse-walk — the residual accumulates
// over L layers but stays well below BF16 ULP.
//
// The primitives here are GEMM-free, launch-bound, O(T · m) work per pass.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// Haar pool at k=2.
//
// Input:  X        [T × m] FP32   (T must be even)
// Output: super    [T/2 × m] FP32 — the "coarse" stream for middle layers
//         residual [T/2 × m] FP32 — the "detail" stream retained for unpool
//
// Block mapping: super[i] combines X[2i] and X[2i+1].  Causal ordering is
// preserved (super[i] depends only on X[0..2i+1]), so downstream causal
// attention over super-tokens remains valid over the original sequence.
// ========================================================================
bool hrtc_pool_haar_k2(const float* X, float* super, float* residual,
                       unsigned int T, unsigned int m);

// ========================================================================
// Inverse Haar pool at k=2 — reconstruct X from super + residual.
// ========================================================================
bool hrtc_unpool_haar_k2(const float* super, const float* residual,
                        float* X_out, unsigned int T, unsigned int m);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool hrtc_pool_haar_k2(const float*, float*, float*, unsigned int, unsigned int) { return false; }
inline bool hrtc_unpool_haar_k2(const float*, const float*, float*, unsigned int, unsigned int) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
