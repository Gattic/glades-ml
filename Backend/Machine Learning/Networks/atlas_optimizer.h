// ATLAS optimizer: Adaptive Temporally-Predictive Learning in Active Subspaces.
//
// Per-weight-matrix subspace-based optimization with:
// - Baseline-Regularized Subspace Preconditioning (BRSP): Fisher-diagonal
//   preconditioning in the subspace, RMSprop-like baseline in the complement.
//   No gradient information is discarded.
// - Optional Predictive Natural Gradient (PNG) via temporal extrapolation
// - Online subspace tracking via randomized power iteration with EMA blending
//
// Reference: ATLAS framework (research/ATLAS_framework.md)
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "../rng.h"

namespace shmea { class GLogger; }

namespace glades {
namespace atlas {

// Per-weight-matrix ATLAS optimizer state.
//
// For a weight matrix W in R^{m x n}, ATLAS maintains:
// - U in R^{m x r}: orthonormal subspace basis (top-r Fisher eigenvectors)
// - fisherDiag in R^r: EMA of Fisher eigenvalues per subspace dimension
// - sigma2: global second moment EMA (scalar baseline preconditioner)
// - prevGz in R^{r x n}: previous step's compressed gradient
// - mu: adaptive temporal prediction coefficient
struct WeightState
{
	unsigned int m;     // rows of weight matrix
	unsigned int n;     // cols of weight matrix
	unsigned int r;     // subspace rank (r <= min(m, n))

	std::vector<float> U;           // [m * r] orthonormal subspace basis (row-major)
	std::vector<float> fisherDiag;  // [r] EMA of Fisher eigenvalues
	std::vector<float> prevGz;      // [r * n] previous compressed gradient

	float sigma2;                   // global second moment EMA (BRSP baseline)
	float mu;                       // adaptive prediction coefficient
	unsigned long long step;        // optimizer step counter
	bool initialized;

	WeightState()
	    : m(0u), n(0u), r(0u),
	      sigma2(1.0f), mu(0.01f), step(0ULL), initialized(false)
	{
	}

	void reset()
	{
		m = n = r = 0u;
		U.clear();
		fisherDiag.clear();
		prevGz.clear();
		sigma2 = 1.0f;
		mu = 0.01f;
		step = 0ULL;
		initialized = false;
	}
};

// Modified Gram-Schmidt orthonormalization of Q[m x r] stored row-major.
// Q[i * r + j] is element (row i, col j).
// logger: optional GLogger for degenerate-column warnings.
void gramSchmidt(float* Q, unsigned int m, unsigned int r,
                 shmea::GLogger* logger = 0);

// Initialize ATLAS state for a weight matrix of dimensions [m x n].
// rank: desired subspace dimension (clamped to min(m, n))
// muInit: initial prediction coefficient
// logger: optional GLogger for initialization diagnostics.
void initWeightState(WeightState& state, unsigned int m, unsigned int n,
                     unsigned int rank, float muInit, glades::rng::Engine& rng,
                     shmea::GLogger* logger = 0);

// Refresh subspace basis U via randomized power iteration with EMA blending.
// grad: [m * n] gradient (row-major), used as the signal for SVD.
// powerIters: number of power iteration steps (typically 3-5).
// betaRefresh: EMA blending coefficient for basis rotation (0=keep old, 1=full replace).
// Uses warm-start from current U. Transforms Fisher diagonal and prevGz
// into the new basis instead of resetting them.
// logger: optional GLogger for refresh diagnostics.
void refreshSubspace(WeightState& state, const float* grad,
                     unsigned int m, unsigned int n,
                     unsigned int powerIters, float betaRefresh,
                     glades::rng::Engine& rng,
                     shmea::GLogger* logger = 0);

// Apply one ATLAS optimizer step (BRSP variant).
//
// Baseline-Regularized Subspace Preconditioning (BRSP):
// 1. Update global second moment sigma2 (scalar EMA of mean(G^2))
// 2. Periodic subspace refresh with EMA blending (every tSub steps)
// 3. Apply decoupled weight decay to W (full-space)
// 4. Project gradient to subspace: gz = U^T * G
// 5. Update Fisher diagonal (EMA)
// 6. Full-space baseline update: W -= (lr/(sigma2+eps)) * G
// 7. Subspace correction: W += U * diag(lr/(sigma2+eps) - lr/(f+eps)) * gPred
// 8. Adapt prediction coefficient mu
//
// The net effect is:
//   subspace direction c:  step = -lr/(f_c+eps) * gPred_c  (Fisher-preconditioned)
//   complement direction:  step = -lr/(sigma2+eps) * G_perp (baseline-preconditioned)
//
// W: [m * n] weight matrix (modified in place)
// gW: [m * n] accumulated gradient (cleared to zero after use)
// logger: optional GLogger for step diagnostics (logged every tSub steps).
void applyStep(WeightState& state,
               float* W, float* gW,
               unsigned int m, unsigned int n,
               float invBatch, float lr,
               float wd1, float wd2, float gradScale,
               float beta, float muMin, float muMax,
               float eps, unsigned int tSub,
               unsigned int powerIters, float betaRefresh,
               glades::rng::Engine& rng,
               shmea::GLogger* logger = 0);

} // namespace atlas
} // namespace glades
