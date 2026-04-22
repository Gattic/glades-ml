// GPU primitives for LCP — Lattice Compute Pooling (paradigm shift #16).
// See research/PARADIGM_SHIFT_16_CANDIDATE_B_LCP.md and
// research/PARADIGM_SHIFT_16_SELECTION.md.
//
// At each layer entry, cluster the T-token batch into M centroids via
// Locality-Sensitive Hashing (sign bits of a fixed Gaussian projection).
// Run the main block on the M cluster representatives only.  Reconstruct
// per-token output via:
//
//     h_t^out = h_{rep(t)}^main  +  α · β_l · D_φ(h_t − h_{rep(t)})
//
// where D_φ is a shared rank-r SwiGLU detail network approximating the
// block's local Jacobian and β_l is a per-layer learnable gain.
//
// Phase 1 (this file): core primitives — LSH projection, bucket assignment,
// representative selection, gather/scatter.  Phase 2: detail-network
// forward/backward primitives.  Phase 3: trainer wire-in.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>
#include <stdint.h>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// lcp_lsh_project — Locality-Sensitive Hashing.
//
// Computes b_t = pack_bits( sign(h_t · R_j) )_{j=1..K} for each token t,
// where R ∈ ℝ^{d × K} is a fixed Gaussian projection matrix (allocated
// ONCE at trainer init) and K is the number of hash bits (≤ 32).
//
//   h_in       [T × d]         row-major activations
//   R          [d × K]         fixed random Gaussian (col-major)
//   T, d, K    dimensions      (K ≤ 32 for uint32 packing)
//   bucket_out [T]             uint32 bucket id per token
//
// Two tokens with similar `h_t` end up in the same bucket with high
// probability (angular LSH).  The resulting bucket distribution over
// 2^K buckets approximates a uniform spread.
// ========================================================================
bool lcp_lsh_project(const float* h_in, const float* R,
                     unsigned int T, unsigned int d, unsigned int K,
                     unsigned int* bucket_out);

// ========================================================================
// lcp_lsh_init_matrix — initialize a Gaussian LSH projection matrix.
//
//   R_out      [d × K]         output buffer
//   seed       uint64_t        RNG seed (one-time per trainer run)
//
// Uses a Box-Muller transform on a splitmix64 counter RNG to produce
// independent N(0, 1/√d) entries.  Scale is chosen so that ‖h‖ · ‖R_j‖
// is O(1), giving good discrimination on the sign bit.
// ========================================================================
bool lcp_lsh_init_matrix(float* R_out, unsigned int d, unsigned int K,
                         uint64_t seed);

// ========================================================================
// lcp_bucket_first_index — given bucket ids for T tokens, produce the
// permutation that selects the FIRST occurrence of each unique bucket.
//
//   bucket_in  [T]             uint32 bucket ids
//   T          int             number of tokens
//   M_max      int             max representatives (≤ T; typical T/4)
//
//   rep_idx_out [M_max]        index of the first token in each cluster
//   n_reps_out  int*           number of unique buckets found
//   cluster_of_token_out [T]   cluster id ∈ [0, n_reps) for each token
//
// Implementation: sort tokens by bucket id (stable), scan for bucket
// boundaries, emit rep_idx at each boundary.  Host synchronizes n_reps.
// ========================================================================
bool lcp_bucket_first_index(const unsigned int* bucket_in,
                            unsigned int T, unsigned int M_max,
                            unsigned int* rep_idx_out,
                            int* n_reps_out,
                            unsigned int* cluster_of_token_out);

// ========================================================================
// lcp_gather — select rows h[rep_idx[m], :] for m ∈ [0, n_reps).
//
//   h_in       [T × d]
//   rep_idx    [n_reps]
//   n_reps, d  dimensions
//   h_reps_out [n_reps × d]
// ========================================================================
bool lcp_gather(const float* h_in, const unsigned int* rep_idx,
                unsigned int n_reps, unsigned int d,
                float* h_reps_out);

// ========================================================================
// lcp_scatter — broadcast per-rep outputs back to per-token outputs:
//
//     h_out[t, :] = h_reps[cluster_of_token[t], :]
//
//   h_reps_in         [n_reps × d]
//   cluster_of_token  [T]
//   T, d              dimensions
//   h_out             [T × d]
// ========================================================================
bool lcp_scatter(const float* h_reps_in,
                 const unsigned int* cluster_of_token,
                 unsigned int T, unsigned int d,
                 float* h_out);

// ========================================================================
// lcp_compute_delta — compute within-cluster deviations:
//
//     delta[t, :] = h_in[t, :] − h_reps_in[cluster_of_token[t], :]
//
// This is the input to the detail network D_φ.  It measures how much
// each token differs from its cluster representative.  By construction,
// the rep token itself has delta = 0.
//
//   h_in             [T × d]
//   h_reps_in        [n_reps × d]
//   cluster_of_token [T]
//   T, d             dimensions
//   delta_out        [T × d]
// ========================================================================
bool lcp_compute_delta(const float* h_in, const float* h_reps_in,
                       const unsigned int* cluster_of_token,
                       unsigned int T, unsigned int d,
                       float* delta_out);

// ========================================================================
// lcp_compute_delta_backward — split upstream d_delta into dh_in and dh_reps.
//
//     dh_in[t, :]    += d_delta[t, :]
//     dh_reps[c, :]  -= Σ_{t : cluster_of_token[t]==c} d_delta[t, :]
//
// dh_in_out and dh_reps_out may be NULL to skip.  Both accumulate into
// existing buffers (caller must pre-zero if desired).
// ========================================================================
bool lcp_compute_delta_backward(const float* d_delta,
                                const unsigned int* cluster_of_token,
                                unsigned int T, unsigned int d,
                                unsigned int n_reps,
                                float* dh_in_out, float* dh_reps_out);

// ========================================================================
// lcp_scatter_backward — inverse of lcp_scatter.  Accumulate per-token
// gradients back to per-rep slots:
//
//     dh_reps[m, :] += Σ_t : cluster_of_token[t]==m   dh_out[t, :]
//
//   dh_out           [T × d]
//   cluster_of_token [T]
//   dh_reps_out      [n_reps × d]  (overwritten OR accumulated)
//   accumulate       bool
// ========================================================================
bool lcp_scatter_backward(const float* dh_out,
                          const unsigned int* cluster_of_token,
                          unsigned int T, unsigned int d,
                          unsigned int n_reps,
                          bool accumulate,
                          float* dh_reps_out);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool lcp_lsh_project(const float*, const float*,
                            unsigned int, unsigned int, unsigned int,
                            unsigned int*) { return false; }
inline bool lcp_lsh_init_matrix(float*, unsigned int, unsigned int,
                                uint64_t) { return false; }
inline bool lcp_bucket_first_index(const unsigned int*,
                                   unsigned int, unsigned int,
                                   unsigned int*, int*,
                                   unsigned int*) { return false; }
inline bool lcp_gather(const float*, const unsigned int*,
                       unsigned int, unsigned int,
                       float*) { return false; }
inline bool lcp_scatter(const float*, const unsigned int*,
                        unsigned int, unsigned int,
                        float*) { return false; }
inline bool lcp_scatter_backward(const float*, const unsigned int*,
                                 unsigned int, unsigned int, unsigned int,
                                 bool, float*) { return false; }
inline bool lcp_compute_delta(const float*, const float*, const unsigned int*,
                              unsigned int, unsigned int, float*) { return false; }
inline bool lcp_compute_delta_backward(const float*, const unsigned int*,
                                       unsigned int, unsigned int, unsigned int,
                                       float*, float*) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
