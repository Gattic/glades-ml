// GPU primitive for KV-FACE Gate-0 probe (paradigm shift #36).
//
// Measures whether attention popularity `p[t] = (1/T) * Σ_q P[q, t]` is
// Zipfian-like at trained state. Premise for KV-FACE to succeed: the
// per-position popularity distribution must be non-uniform (Gini ≥ 0.4).
//
// Probe outputs: per-head Gini coefficient of the popularity vector.
// A single call reduces [nHeads, T, T] softmax-probabilities into
// {mean_gini, min_gini, max_gini} scalars.
//
// See research/PARADIGM_SHIFT_36_DESIGN.md §14.

#pragma once

#include "gpu_buffer.h"
#include <cstddef>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// kvface_probe_compute_popularity — compute per-head, per-position
// popularity from an attention probability tensor.
//
// Inputs:
//   P           [nHeads, T, T]  row-major per-head causal-masked softmax
//   nHeads, T                   dims
// Outputs:
//   popularity  [nHeads, T]     p[h, t] = (1/T) * Σ_q P[h, q, t]
//
// Launch: grid(T, nHeads), block(256 threads) with shared-memory reduction.
// Complexity: O(nHeads * T^2) loads, O(nHeads * T) stores.
// ========================================================================
bool kvface_probe_compute_popularity(const float* P,
                                     float* popularity,
                                     int nHeads,
                                     int T);

// ========================================================================
// kvface_probe_gini_device — compute per-head Gini coefficient of a
// popularity [nHeads, T] tensor on device.
//
// Gini formula (using sorted Lorenz curve):
//   G = 1 - (2 / (T-1)) * Σ_{i=1}^{T-1} L_i
// where L_i = (Σ_{j=1}^{i} x_j_sorted) / (Σ_{j=1}^{T} x_j)
//
// Simpler equivalent formula (no sort):
//   G = (Σ_i Σ_j |x_i - x_j|) / (2 * T * Σ_i x_i)
//
// We use the pairwise-difference formula which is O(T^2) per head but
// parallelizable and avoids a GPU sort.
//
// Inputs:
//   popularity  [nHeads, T]  non-negative values (row-stochastic → Σ=1)
//   nHeads, T                dims
// Outputs:
//   gini_per_head [nHeads]   Gini coefficient per head, in [0, 1]
//
// Launch: one block per head, block-reduce for pairwise sum and total sum.
// ========================================================================
bool kvface_probe_gini_device(const float* popularity,
                              float* gini_per_head,
                              int nHeads,
                              int T);

// ========================================================================
// kvface_probe_reduce_stats — reduce per-head Gini to (mean, min, max)
// scalars on device.
//
// Inputs:
//   gini_per_head [nHeads]   Gini per head
//   nHeads                   count
// Outputs:
//   stats [3]    [0]=mean, [1]=min, [2]=max
// ========================================================================
bool kvface_probe_reduce_stats(const float* gini_per_head,
                               float* stats3,
                               int nHeads);

// ========================================================================
// Host reference: CPU Gini computation for parity-testing the device
// primitive. Uses the sorted-Lorenz formula for numerical stability.
// ========================================================================
float kvface_probe_gini_host(const float* popularity_host, int T);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

namespace gpu {
inline bool kvface_probe_compute_popularity(const float*, float*, int, int) { return false; }
inline bool kvface_probe_gini_device(const float*, float*, int, int) { return false; }
inline bool kvface_probe_reduce_stats(const float*, float*, int) { return false; }
inline float kvface_probe_gini_host(const float*, int) { return 0.0f; }
} // namespace gpu

#endif // GLADES_HAVE_CUDA

} // namespace glades
