// GPU primitives for FACE — Frequency-Aware Column-normalized Embedding
// optimizer (paradigm shift #28).  See research/PARADIGM_SHIFT_28_DESIGN.md.
//
// FACE repairs MFIO v2 on matrices with sparse-per-row gradients (embedding,
// output projection).  The key fix: replace MFIO's column norm
//   dn[j] = Σ_i g[i,j]²                       (dominated by frequent rows)
// with the frequency-debiased arithmetic-mean column norm
//   d̃n[j] = (1/q) · Σ_i∈active g[i,j]²       (correct under Zipfian rows)
// where q is the count of active rows in the current micro-batch.
//
// The preconditioner becomes:
//   σ_{ij} = 1 / √(zn̄[i] · dn̄[j] / (q̂ · gF̄) + ε²)
// with (zn̄, dn̄, q̂, gF̄) maintained as EMAs across steps.
//
// Phase 1 (this file): stats reduction primitive — compute zn, dn_raw,
// dn_deb, gF, q from a sparse dense-backed gradient.  Phase 2 will add
// the EMA update + preconditioner application kernels.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// face_compute_sparse_stats — reduce a sparse-per-row gradient tensor to
// the statistics FACE needs.
//
// Given g ∈ ℝ^{V × m} row-major FP32 where ROWS are sparsely nonzero
// (only "active" rows — the tokens present in the current micro-batch —
// have nonzero g[i, :]; inactive rows are exactly zero), compute:
//
//   zn[i]    = ‖g[i, :]‖²            (V values; zero for inactive rows)
//   q        = count of active rows   (scalar; "active" = any nonzero entry)
//   dn_raw[j]= Σ_i g[i, j]²           (m values; summed over active rows)
//   dn_deb[j]= dn_raw[j] / max(q, 1)  (m values; frequency-debiased)
//   gF       = Σ_{i,j} g[i,j]²        (scalar; Frobenius² of g)
//
// Implementation: fused reduction kernel that does one pass over g.
// Per-row reduction + elementwise-max activity flag → count q.
// Per-column reduction via block-per-column kernel.
// The dn_deb and gF are computed by host-side divisions after the reduction.
//
// Inputs:
//   g            [V × m]   row-major FP32
//   V, m                   dims
// Outputs:
//   zn_out       [V]        FP32; zn[i] = ‖g[i, :]‖²
//   dn_raw_out   [m]        FP32; dn_raw[j] = Σ_i g[i, j]²
//   q_out                   FP32 scalar; count of active rows
//   gF_out                  FP32 scalar; ‖g‖_F²
//
// The caller can compute dn_deb[j] = dn_raw[j] / q on-host (cheap, m values).
//
// Cost: O(V · m).  Dominated by the two pass-over-g kernels; consistent
// with existing MFIO row/col reductions.
// ========================================================================
bool face_compute_sparse_stats(const float* g,
                               unsigned int V, unsigned int m,
                               float* zn_out,
                               float* dn_raw_out,
                               float* q_out,
                               float* gF_out);

// ========================================================================
// face_apply_preconditioned_update — σ-scaled weight update.
//
// Given the EMA-smoothed state (zn̄, dn̄, q̂, gF̄), apply the FACE update
// to theta:
//
//     σ_{ij} = 1 / √( zn̄[i] · dn̄[j] / (q̂ · gF̄) + ε² )
//     θ[i,j] ← θ[i,j] − η · σ_{ij} · g[i,j]
//
// Because g is zero on inactive rows (the sparse-per-row invariant of
// embedding gradients), the update is naturally masked — no explicit
// active-row list is needed.  This is the core design property that makes
// FACE sparsity-invariant.
//
// Inputs:
//   g           [V × m]   row-major FP32 gradient (sparse per row)
//   zn_bar      [V]        EMA of row squared norms
//   dn_bar      [m]        EMA of frequency-debiased column norms
//   q_hat                  scalar EMA of active-row count (device)
//   gF_hat                 scalar EMA of Frobenius² (device)
//   V, m                   dims
//   lr                     learning rate
//   eps                    numerical floor on σ denominator
// In/out:
//   theta       [V × m]   row-major FP32 weights (in-place update)
//
// Cost: O(V · m) elementwise; one kernel launch.
// ========================================================================
bool face_apply_preconditioned_update(float* theta,
                                      const float* g,
                                      const float* zn_bar,
                                      const float* dn_bar,
                                      const float* q_hat,
                                      const float* gF_hat,
                                      unsigned int V, unsigned int m,
                                      float lr, float eps);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool face_compute_sparse_stats(const float*,
                                      unsigned int, unsigned int,
                                      float*, float*, float*, float*) { return false; }
inline bool face_apply_preconditioned_update(float*, const float*,
                                             const float*, const float*,
                                             const float*, const float*,
                                             unsigned int, unsigned int,
                                             float, float) { return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
