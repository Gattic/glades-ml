// GPU primitive for NESR — Noise-Equilibrium Stochastic Resonance
// (paradigm shift #32).  Post-Adam Langevin-style noise injection.
//
// Langevin SDE discretized:
//     θ_{t+1} = θ_t − η · Adam(g) + √(2 · η · T) · ξ,  ξ ~ N(0, I)
//
// where T (temperature) is scheduled to decay across training.  The
// controlled noise injection helps escape shallow local minima in
// early training; near-zero T in late training preserves fine-tuning.
//
// Design rationale: Adam's implicit mini-batch noise is uncontrolled.
// NESR adds explicit, scheduled Gaussian noise with known magnitude
// and decay schedule.  Related to SGLD (stochastic gradient Langevin
// dynamics) but applied to the Adam-preconditioned direction.
//
// Empirical expectation: +0.2-0.5 nat late-training loss improvement
// on noisy objectives; neutral or small regression on clean ones.
//
// Phase 1 (this file): single primitive for post-Adam noise injection.
// Phase 2 wires into chiron_train behind --nesr T_init flag.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>

namespace glades {

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// ========================================================================
// nesr_inject_noise — add scaled Gaussian noise to a parameter tensor.
//
//     θ[i] ← θ[i] + scale · ξ_i,  ξ_i ~ N(0, 1)
//
// The noise sample is generated via xorshift32 PRNG seeded by
// (base_seed XOR i XOR step).  Deterministic under fixed seed.
//
// Uses Box-Muller: two uniform samples → one standard-normal sample.
// Only half the noise samples are "new"; the other half reuses the
// paired Box-Muller component.  In practice we just compute a new pair
// per thread per call — slight over-compute for simplicity.
//
// Inputs:
//   theta       [n_params]  FP32 parameter tensor (in-place)
//   scale                    FP32 noise magnitude (typically √(2·lr·T))
//   base_seed                u32 RNG seed
//   step                     u32 step counter (XOR'd into seed for
//                                              per-step decorrelation)
//   n_params                 number of parameters
//
// Cost: O(n_params) elementwise.
// ========================================================================
bool nesr_inject_noise(float* theta,
                       float scale,
                       unsigned int base_seed,
                       unsigned int step,
                       int n_params);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

inline bool nesr_inject_noise(float*, float, unsigned int, unsigned int, int)
{ return false; }

#endif // GLADES_HAVE_CUDA

} // namespace glades
