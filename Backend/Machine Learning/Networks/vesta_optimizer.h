// VESTA optimizer: Variational Entropy-Spectral Trust-region Adaptation.
//
// Per-weight-matrix Bregman mirror descent on a von Neumann spectral entropy
// potential Phi(W) = -1/2 tr(W^T W log(W^T W)) + mu/2 |W|_F^2. Tracks top-r
// SVD (U, V, ell=log sigma) of each weight matrix via sketched randomized
// range-finder. The update rule solves a trust-region-constrained mirror-step
// variational problem in closed form and applies:
//   - Log-scale diagonal update for tracked singular directions (no second-moment
//     EMA of squared gradients; curvature comes from phi''(sigma)).
//   - Log-scale momentum (EMA of ell).
//   - Stiefel QR retraction for U, V (via off-diagonal of U^T g V).
//   - Signed complement step on the gradient component outside the tracked subspace.
//   - Operator-norm trust-region clamp on the leading singular value.
//   - Slow homeostatic adaptation of the spectral target ell_star toward ell.
//
// Reference: research/VESTA_framework.md (spectral entropy mirror descent).
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "../rng.h"

namespace shmea { class GLogger; }

namespace glades {

struct VestaConfig; // forward; defined in training_config.h.

namespace vesta {

// Per-weight-matrix VESTA optimizer state.
//
// For W in R^{m x n} with tracked rank r <= min(m, n):
//   U       [m * r] row-major orthonormal left basis (U^T U = I_r).
//   V       [n * r] row-major orthonormal right basis (V^T V = I_r).
//   ell     [r]     log-singular-values of the tracked component.
//   beta    [r]     log-scale momentum EMA of ell.
//   ellStar [r]     target log-singular profile (slow homeostasis toward ell).
struct WeightState
{
	unsigned int m;
	unsigned int n;
	unsigned int r;

	std::vector<float> U;        // [m * r] row-major
	std::vector<float> V;        // [n * r] row-major
	std::vector<float> ell;      // [r]
	std::vector<float> beta;     // [r]
	std::vector<float> ellStar;  // [r]

	// Persistent scratch buffers, allocated once at init.
	std::vector<float> scratch_A;           // [r * r] U^T g V
	std::vector<float> scratch_UA;          // [m * r] U A
	std::vector<float> scratch_WrOld;       // [m * n] U diag(exp(ell_old)) V^T
	std::vector<float> scratch_WrNew;       // [m * n] U_new diag(exp(ell_new)) V_new^T
	std::vector<float> scratch_gPerp;       // [m * n] g - U A V^T
	std::vector<float> scratch_Omega_U;     // [m * r] Stiefel tangent on U side
	std::vector<float> scratch_Omega_V;     // [n * r] Stiefel tangent on V side
	std::vector<float> scratch_URaw;        // [m * r] pre-QR U
	std::vector<float> scratch_VRaw;        // [n * r] pre-QR V
	std::vector<float> scratch_sketchOmega; // [n * (r + oversample)]
	std::vector<float> scratch_sketchY;     // [m * (r + oversample)]
	std::vector<float> scratch_sketchB;     // [(r + oversample) * n]
	std::vector<float> scratch_sketchVr;    // [n * r]
	std::vector<float> scratch_sketchS;     // [(r + oversample)]

	// Optional Lion-style complement momentum. Allocated [m * n] when
	// VestaConfig::complementMomentumEnabled is true; empty otherwise.
	std::vector<float> complementMomentum;

	unsigned long long step;
	float maxExpEllPrev;
	bool initialized;

	WeightState()
	    : m(0u), n(0u), r(0u),
	      step(0ULL),
	      maxExpEllPrev(1.0f),
	      initialized(false)
	{
	}

	void reset()
	{
		m = n = r = 0u;
		U.clear(); V.clear(); ell.clear(); beta.clear(); ellStar.clear();
		scratch_A.clear(); scratch_UA.clear();
		scratch_WrOld.clear(); scratch_WrNew.clear();
		scratch_gPerp.clear();
		scratch_Omega_U.clear(); scratch_Omega_V.clear();
		scratch_URaw.clear(); scratch_VRaw.clear();
		scratch_sketchOmega.clear(); scratch_sketchY.clear();
		scratch_sketchB.clear(); scratch_sketchVr.clear();
		scratch_sketchS.clear();
		complementMomentum.clear();
		step = 0ULL;
		maxExpEllPrev = 1.0f;
		initialized = false;
	}
};

// Modified Gram-Schmidt orthonormalization of Q[m x r] stored row-major.
// Q[i * r + j] is element (row i, col j). Drops near-zero columns.
void gramSchmidt(float* Q, unsigned int m, unsigned int r);

// Small dense SVD of B[mB x nB] (row-major) via Jacobi eigendecomposition of B^T B.
// Produces Vout[nB * r] (right singular vectors) and sOut[r] (singular values,
// sorted descending). Used only for small matrices.
// Returns true on success.
bool denseSVD_rightV(const float* B, unsigned int mB, unsigned int nB,
                     float* Vout, float* sOut, unsigned int r);

// Thin QR (row-major): factor Q[m * r] in place so its columns are orthonormal.
// Returns false on rank deficiency.
bool thinQR(float* Q, unsigned int m, unsigned int r);

// Initialize VESTA state for a weight matrix of dimensions [m x n].
// Performs a sketched SVD of W and sets U, V, ell to the top-r singular
// triplets; initializes beta = ell, ellStar = ell, step = 0.
void initWeightState(WeightState& state,
                     const float* W,
                     unsigned int m, unsigned int n,
                     const VestaConfig& vc,
                     glades::rng::Engine& rng,
                     shmea::GLogger* logger = 0);

// Refresh (U, V, ell) via sketched SVD of the current W.
// Overwrites U, V, ell with the new top-r triplets. Does not touch beta / ellStar.
// Returns false on non-finite detection.
bool refreshSubspace(WeightState& state,
                     const float* W,
                     unsigned int m, unsigned int n,
                     const VestaConfig& vc,
                     glades::rng::Engine& rng,
                     shmea::GLogger* logger = 0);

// Apply one VESTA optimizer step to the weight matrix W using gradient gW.
// Signature matches the ATLAS update convention.
//
// W   : [m * n] row-major weight matrix (modified in place).
// gW  : [m * n] row-major raw gradient (modified: cleared to zero after use).
// Returns false if a non-finite value is detected during the step.
bool applyStep(WeightState& state,
               float* W, float* gW,
               unsigned int m, unsigned int n,
               float invBatch, float lr,
               float wd1, float wd2, float gradScale,
               const VestaConfig& vc,
               glades::rng::Engine& rng,
               shmea::GLogger* logger = 0,
               const char* tag = 0);

// Convenience: initializes state if needed, then calls applyStep.
bool update(WeightState& state,
            float* W, float* gW,
            unsigned int m, unsigned int n,
            float invBatch, float lr,
            float wd1, float wd2, float gradScale,
            const VestaConfig& vc,
            glades::rng::Engine& rng,
            shmea::GLogger* logger = 0,
            const char* tag = 0);

} // namespace vesta
} // namespace glades
