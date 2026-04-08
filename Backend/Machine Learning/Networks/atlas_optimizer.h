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

// Forward declaration (defined in training_config.h).
struct ATLASConfig;

namespace atlas {

// Per-weight-matrix ATLAS optimizer state.
//
// For a weight matrix W in R^{m x n}, ATLAS maintains:
// - U in R^{m x r}: orthonormal subspace basis (top-r Fisher eigenvectors)
// - fisherDiag in R^r: EMA of Fisher eigenvalues per subspace dimension
// - totalTrace: EMA of the normalized covariance trace on the ATLAS update scale
// - sigma2: complement closure scalar derived from totalTrace and fisherDiag
// - prevGz in R^{r x n}: previous step's compressed gradient
// - mu: adaptive temporal prediction coefficient
struct WeightState
{
	unsigned int m;     // rows of weight matrix
	unsigned int n;     // cols of weight matrix
	unsigned int r;     // subspace rank (r <= min(m, n))
	unsigned int activeRank; // currently active leading rank (activeRank <= r)

	std::vector<float> U;           // [m * r] orthonormal subspace basis (row-major)
	std::vector<float> fisherDiag;  // [r] EMA of Fisher eigenvalues
	std::vector<float> prevGz;      // [r * n] previous compressed gradient

	// Persistent scratch buffers (allocated once in initWeightState, reused every step).
	// applyStep scratch:
	std::vector<float> scratch_gz;        // [r * n]
	std::vector<float> scratch_corrected; // [r * n]
	// refreshSubspace scratch:
	std::vector<float> scratch_U_old;     // [m * r]
	std::vector<float> scratch_f_old;     // [r]
	std::vector<float> scratch_B;         // [r * n]
	std::vector<float> scratch_Z;         // [m * r]
	std::vector<float> scratch_overlap;   // [r * r]
	std::vector<float> scratch_prevGzOld; // [r * n]
	std::vector<float> scratch_basisPacked; // [m * r] packed leading basis for GEMM fast path

	float totalTrace;               // EMA trace of the normalized covariance operator
	float sigma2;                   // complement closure scalar (BRSP baseline)
	float mu;                       // adaptive prediction coefficient
	float lastBaselineRate;         // diagnostics for the most recent baseline step
	unsigned long long step;        // optimizer step counter
	bool initialized;

	WeightState()
	    : m(0u), n(0u), r(0u), activeRank(0u),
	      totalTrace(0.0f), sigma2(0.0f), mu(0.01f), lastBaselineRate(0.0f),
	      step(0ULL), initialized(false)
	{
	}

	void reset()
	{
		m = n = r = activeRank = 0u;
		U.clear();
		fisherDiag.clear();
		prevGz.clear();
		scratch_gz.clear();
		scratch_corrected.clear();
		scratch_U_old.clear();
		scratch_f_old.clear();
		scratch_B.clear();
		scratch_Z.clear();
		scratch_overlap.clear();
		scratch_prevGzOld.clear();
		scratch_basisPacked.clear();
		totalTrace = 0.0f;
		sigma2 = 0.0f;
		mu = 0.01f;
		lastBaselineRate = 0.0f;
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
// Returns true on success, false if non-finite values detected during refresh.
bool refreshSubspace(WeightState& state, const float* grad,
                     unsigned int m, unsigned int n,
                     unsigned int powerIters, float betaRefresh,
                     bool fisherWeightedRefresh,
                     glades::rng::Engine& rng,
                     shmea::GLogger* logger = 0);

// Apply one ATLAS optimizer step (BRSP variant).
//
// Baseline-Regularized Subspace Preconditioning (BRSP):
// 1. Update the normalized covariance trace on the ATLAS update scale
// 2. Periodic subspace refresh with EMA blending (every tSub steps)
// 3. Apply decoupled weight decay to W (full-space)
// 4. Project gradient to subspace: gz = U^T * G
// 5. Update Fisher diagonal (EMA)
// 6. Recompute sigma2 by trace-closing the complement covariance
// 7. Full-space baseline update: W -= (lr/(sigma2+eps)) * G
// 8. Subspace correction: W += U * diag(lr/(sigma2+eps) - lr/(f+eps)) * gPred
// 9. Adapt prediction coefficient mu
//
// The net effect is:
//   subspace direction c:  step = -lr/(f_c+eps) * gPred_c  (Fisher-preconditioned)
//   complement direction:  step = -lr/(sigma2+eps) * G_perp (baseline-preconditioned)
//
// W: [m * n] weight matrix (modified in place)
// gW: [m * n] accumulated gradient (cleared to zero after use)
// logger: optional GLogger for step diagnostics (logged every tSub steps).
// tag: optional per-matrix identifier included in log messages (e.g. "block3.Wq").
// Returns false if the optimizer entered a NaN/Inf recovery path during this step.
bool applyStep(WeightState& state,
               float* W, float* gW,
               unsigned int m, unsigned int n,
               float invBatch, float lr,
               float wd1, float wd2, float gradScale,
               const ATLASConfig& ac,
               glades::rng::Engine& rng,
               shmea::GLogger* logger = 0,
               const char* tag = 0);

// Convenience wrapper: initializes state if needed, then calls applyStep.
// Replaces the repeated init-if-needed + applyStep boilerplate in SGD files.
// Returns false if applyStep entered a recovery path.
bool update(WeightState& state, float* W, float* gW,
            unsigned int m, unsigned int n,
            float invBatch, float lr,
            float wd1, float wd2, float gradScale,
            const ATLASConfig& ac,
            glades::rng::Engine& rng,
            shmea::GLogger* logger = 0,
            const char* tag = 0);

// Apply vanilla SGD to a 1D bias vector and zero the gradient.
// Returns false if any bias element becomes non-finite (NaN/Inf).
// This centralizes the repeated bias-update-with-NaN-check pattern
// used in all SGD files alongside ATLAS weight updates.
bool updateBias(float* bias, float* gBias, unsigned int size,
                float invBatch, float lr, float gradScale);

} // namespace atlas
} // namespace glades
