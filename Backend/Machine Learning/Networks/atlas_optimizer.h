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
// - V in R^{m x b}: residual complement block basis (row-major, packed by row)
// - complementBlock in R^{b x b}: EMA covariance block captured by V
// - scoutBasis in R^{m x b}: orthonormal scout basis used only for residual proposal quality
// - scoutCov / scoutNoise in R^{b x b}: transported scout covariance and innovation penalties
// - activeComplementRank: online active prefix inside the retained residual block
// - trialComplement*: probationary Kelly controller state for the next residual mode
// - complementFisher: trace(complementBlock) for diagnostics / compatibility
// - totalTrace: EMA of the normalized covariance trace on the ATLAS update scale
// - sigma2: isotropic tail closure derived from totalTrace, fisherDiag, and V
// - prevGz in R^{r x n}: previous step's compressed gradient
// - prevGv in R^{b x n}: previous step's complement-block gradient
// - mu: adaptive temporal prediction coefficient
struct WeightState
{
	unsigned int m;     // rows of weight matrix
	unsigned int n;     // cols of weight matrix
	unsigned int r;     // subspace rank (r <= min(m, n))
	unsigned int activeRank; // currently active leading rank (activeRank <= r)
	unsigned int complementRank; // allocated complement-block rank (>= 1 for storage)
	unsigned int sparrowModeRank; // allocated SPARROW transfer mode rank (>= 1 for storage)
	unsigned int activeComplementRank; // runtime active residual rank (<= complementRank)
	unsigned int trialComplementRank; // probationary target rank awaiting promotion
	unsigned int trialComplementWins; // consecutive accepted control boundaries

	std::vector<float> U;           // [m * r] orthonormal subspace basis (row-major)
	std::vector<float> fisherDiag;  // [r] EMA of Fisher eigenvalues
	std::vector<float> V;           // [m * complementRank] residual complement basis
	std::vector<float> complementBlock; // [complementRank * complementRank] dense residual covariance EMA
	std::vector<float> scoutBasis;  // [m * complementRank] scout basis for quotient residual proposals
	std::vector<float> scoutCov;    // [complementRank * complementRank] transported scout covariance EMA
	std::vector<float> scoutNoise;  // [complementRank * complementRank] scout innovation penalty EMA
	std::vector<float> prevGz;      // [r * n] previous compressed gradient
	std::vector<float> prevPrevGz;  // [r * n] two-step history for PRISM memory correction
	std::vector<float> prevGv;      // [complementRank * n] previous complement-block gradient
	std::vector<float> resolveGzHistory; // [4 * r * n] lagged active compressed-gradient history for RESOLVE
	std::vector<float> heroGwHistory; // [4 * complementRank * n] lagged scout compressed-gradient history for HERO/COBALT
	std::vector<float> sparrowPrevActive; // [r * n] previous quotient-horizontal active sketch
	std::vector<float> sparrowPrevScout;  // [complementRank * n] previous quotient-horizontal scout sketch
	std::vector<float> sparrowFutureCov;  // [r * r] EMA covariance of current horizontal active sketch
	std::vector<float> sparrowPastCov;    // [(r + complementRank)^2] EMA covariance of past active+scout sketch
	std::vector<float> sparrowCrossCov;   // [r * (r + complementRank)] EMA cross-covariance current vs past
	std::vector<float> sparrowLeftMode;   // [sparrowModeRank * r] retained SPARROW left/output modes
	std::vector<float> sparrowRightMode;  // [sparrowModeRank * (r + complementRank)] retained SPARROW right/input modes
	std::vector<float> sparrowLatent;     // [sparrowModeRank * n] SPARROW latent memory state across columns
	std::vector<float> qbrtLeftMode;      // [r] retained QBRT balanced left/output mode
	std::vector<float> qbrtLatent;        // [n] QBRT latent memory state across columns
	std::vector<float> qrcLeftMode;       // [r] retained QRC closed-loop output mode
	std::vector<float> qrcLatent;         // [n] QRC control signal across columns
	std::vector<float> riftLeftMode;      // [r] retained RIFT output mode
	std::vector<float> riftLatent;        // [n] RIFT latent memory state across columns
	std::vector<float> orbitPrevSignal;   // [n] previous ORBIT-Lite output-mode signal
	std::vector<float> orbitLeftMode;     // [r] active-space lift of the retained ORBIT mode
	std::vector<float> orbitLatent;       // [n] ORBIT-Lite latent memory state across columns

	// Persistent scratch buffers (allocated once in initWeightState, reused every step).
	// applyStep scratch:
	std::vector<float> scratch_gz;        // [r * n]
	std::vector<float> scratch_corrected; // [r * n]
	std::vector<float> scratch_gv;        // [complementRank * n]
	std::vector<float> scratch_correctedV; // [complementRank * n]
	std::vector<float> scratch_gwScout;   // [complementRank * n]
	// refreshSubspace scratch:
	std::vector<float> scratch_U_old;     // [m * r]
	std::vector<float> scratch_f_old;     // [r]
	std::vector<float> scratch_B;         // [r * n]
	std::vector<float> scratch_Z;         // [m * r]
	std::vector<float> scratch_overlap;   // [r * r]
	std::vector<float> scratch_prevGzOld; // [r * n]
	std::vector<float> scratch_prevPrevGzOld; // [r * n]
	std::vector<float> scratch_resolveGzHistoryOld; // [4 * r * n]
	std::vector<float> scratch_heroGwHistoryOld; // [4 * complementRank * n]
	std::vector<float> scratch_basisPacked; // [m * r] packed leading basis for GEMM fast path
	std::vector<float> scratch_V_old;     // [m * complementRank]
	std::vector<float> scratch_Bv;        // [complementRank * n]
	std::vector<float> scratch_Zv;        // [m * complementRank]
	std::vector<float> scratch_W_old;     // [m * complementRank]
	std::vector<float> scratch_Bw;        // [complementRank * n]
	std::vector<float> scratch_Zw;        // [m * complementRank]
	std::vector<float> scratch_complementMat; // [complementRank * complementRank]
	std::vector<float> scratch_complementEigVec; // [complementRank * complementRank]
	std::vector<float> scratch_complementEigVal; // [complementRank]
	std::vector<float> scratch_scoutMat;  // [complementRank * complementRank]
	std::vector<float> scratch_scoutEigVec; // [complementRank * complementRank]
	std::vector<float> scratch_scoutEigVal; // [complementRank]
	std::vector<float> scratch_sparrowActive; // [r * n]
	std::vector<float> scratch_sparrowScout;  // [complementRank * n]
	std::vector<float> scratch_sparrowPastSignal; // [n]
	std::vector<float> scratch_geodeRhsCol;   // [m]
	std::vector<float> scratch_geodeInvDiagCol; // [m]
	std::vector<float> scratch_geodeActiveCurrent; // [r]
	std::vector<float> scratch_geodeActiveDelta; // [r]
	std::vector<float> scratch_geodeSystemMat; // [r * r]
	std::vector<float> scratch_geodeRhs;      // [r]
	std::vector<float> scratch_geodeSolution; // [r]

	float complementFisher;         // trace(complementBlock)
	float totalTrace;               // EMA trace of the normalized covariance operator
	float sigma2;                   // isotropic complement-tail closure scalar
	float trialComplementMean;      // EMA excess utility for probationary residual mode
	float trialComplementVar;       // EMA squared innovation / uncertainty proxy
	float lastPredictiveEdge;       // latest PRISM predictive-edge score
	float lastMemoryGain;           // latest PRISM active-memory gain
	float lastResolveEdge;          // latest RESOLVE transfer-edge score
	float lastResolveKernelRho;     // latest RESOLVE stable memory pole estimate
	float lastResolveMemoryGain;    // latest RESOLVE active-memory gain
	float lastHeroEdge;             // latest HERO Hankel-edge score
	float lastHeroSigma;            // latest HERO top whitened Hankel singular proxy
	float lastHeroMemoryGain;       // latest HERO active-memory gain
	float lastCobaltEdge;           // latest COBALT transfer-edge score
	float lastCobaltSigma;          // latest COBALT top transfer singular proxy
	float lastCobaltMemoryGain;     // latest COBALT active-memory gain
	float lastBirchEdge;            // latest BIRCH Hankel transfer-edge score
	float lastBirchSigma;           // latest BIRCH top Hankel singular proxy
	float lastBirchMemoryGain;      // latest BIRCH active-memory gain
	float lastGhostEdge;            // latest GHOST quotient-transfer edge score
	float lastGhostSigma;           // latest GHOST top balanced transfer singular proxy
	float lastGhostHorizontalRatio; // latest retained horizontal energy fraction
	float lastGhostMemoryGain;      // latest GHOST active-memory gain
	float sparrowPoleNumer;         // EMA numerator for SPARROW pole fit
	float sparrowPoleDenom;         // EMA denominator for SPARROW pole fit
	float sparrowPole;              // latest retained SPARROW stable pole
	float lastSparrowEdge;          // latest SPARROW canonical-edge score
	float lastSparrowSigma;         // latest SPARROW top whitened transfer singular proxy
	float lastSparrowSecondEdge;    // latest raw mode-2 SPARROW edge before gating
	float lastSparrowSecondSigma;   // latest raw mode-2 SPARROW singular proxy before gating
	float lastSparrowHorizontalRatio; // latest retained SPARROW horizontal energy fraction
	float lastSparrowMemoryGain;    // latest SPARROW active-memory gain
	unsigned int lastSparrowActiveModes; // latest retained SPARROW mode count after gating
	float qbrtPoleNumer;            // EMA numerator for QBRT pole fit
	float qbrtPoleDenom;            // EMA denominator for QBRT pole fit
	float qbrtPole;                 // latest retained QBRT stable pole
	float lastQbrtEdge;             // latest QBRT transfer-edge score
	float lastQbrtSigma;            // latest QBRT balanced transfer singular proxy
	float lastQbrtHorizontalRatio;  // latest retained QBRT horizontal energy fraction
	float lastQbrtMemoryGain;       // latest QBRT active-memory gain
	float qrcPoleNumer;             // EMA numerator for QRC plant pole fit
	float qrcPoleDenom;             // EMA denominator for QRC plant pole fit
	float qrcPole;                  // latest retained QRC plant pole
	float lastQrcEdge;              // latest QRC closed-loop edge score
	float lastQrcSigma;             // latest QRC balanced transfer singular proxy
	float lastQrcHorizontalRatio;   // latest retained QRC horizontal energy fraction
	float lastQrcControlGain;       // latest QRC scalar feedback gain
	float lastQrcMemoryGain;        // latest QRC active-memory gain
	float riftPoleNumer;            // EMA numerator for RIFT pole fit
	float riftPoleDenom;            // EMA denominator for RIFT pole fit
	float riftPole;                 // latest retained RIFT stable pole
	float lastRiftEdge;             // latest RIFT path-edge score
	float lastRiftSigma;            // latest RIFT canonical signature correlation
	float lastRiftHorizontalRatio;  // latest retained RIFT horizontal energy fraction
	float lastRiftAreaEnergy;       // latest normalized level-2 area-energy share
	float lastRiftPredR2;           // latest RIFT predictive explained-variance proxy
	float lastRiftMemoryGain;       // latest RIFT active-memory gain
	float orbitPoleNumer;           // EMA numerator for ORBIT-Lite pole fit
	float orbitPoleDenom;           // EMA denominator for ORBIT-Lite pole fit
	float orbitPole;                // latest retained ORBIT-Lite stable pole
	float lastOrbitEdge;            // latest ORBIT-Lite functional-edge score
	float lastOrbitSigma;           // latest ORBIT-Lite top generalized output-mode score
	float lastOrbitHorizontalRatio; // latest ORBIT-Lite output-horizontal energy fraction
	float lastOrbitMemoryGain;      // latest ORBIT-Lite active-memory gain
	float externalSparrowTrust;     // outer-loop trust multiplier applied to SPARROW memory gain
	float mu;                       // adaptive prediction coefficient
	float lastBaselineRate;         // diagnostics for the most recent baseline step
	unsigned long long step;        // optimizer step counter
	bool initialized;

	WeightState()
	    : m(0u), n(0u), r(0u), activeRank(0u), complementRank(0u),
	      sparrowModeRank(1u),
	      activeComplementRank(0u), trialComplementRank(0u),
	      trialComplementWins(0u),
	      complementFisher(0.0f), totalTrace(0.0f), sigma2(0.0f),
	      trialComplementMean(0.0f), trialComplementVar(0.0f),
	      lastPredictiveEdge(0.0f), lastMemoryGain(0.0f),
	      lastResolveEdge(0.0f), lastResolveKernelRho(0.0f),
	      lastResolveMemoryGain(0.0f),
	      lastHeroEdge(0.0f), lastHeroSigma(0.0f), lastHeroMemoryGain(0.0f),
	      lastCobaltEdge(0.0f), lastCobaltSigma(0.0f), lastCobaltMemoryGain(0.0f),
	      lastBirchEdge(0.0f), lastBirchSigma(0.0f), lastBirchMemoryGain(0.0f),
	      lastGhostEdge(0.0f), lastGhostSigma(0.0f),
	      lastGhostHorizontalRatio(1.0f), lastGhostMemoryGain(0.0f),
	      sparrowPoleNumer(0.0f), sparrowPoleDenom(0.0f), sparrowPole(0.0f),
	      lastSparrowEdge(0.0f), lastSparrowSigma(0.0f),
	      lastSparrowSecondEdge(0.0f), lastSparrowSecondSigma(0.0f),
	      lastSparrowHorizontalRatio(1.0f), lastSparrowMemoryGain(0.0f),
	      lastSparrowActiveModes(0u),
	      qbrtPoleNumer(0.0f), qbrtPoleDenom(0.0f), qbrtPole(0.0f),
	      lastQbrtEdge(0.0f), lastQbrtSigma(0.0f),
	      lastQbrtHorizontalRatio(1.0f), lastQbrtMemoryGain(0.0f),
	      qrcPoleNumer(0.0f), qrcPoleDenom(0.0f), qrcPole(0.0f),
	      lastQrcEdge(0.0f), lastQrcSigma(0.0f),
	      lastQrcHorizontalRatio(1.0f), lastQrcControlGain(0.0f),
	      lastQrcMemoryGain(0.0f),
	      riftPoleNumer(0.0f), riftPoleDenom(0.0f), riftPole(0.0f),
	      lastRiftEdge(0.0f), lastRiftSigma(0.0f),
	      lastRiftHorizontalRatio(1.0f), lastRiftAreaEnergy(0.0f),
	      lastRiftPredR2(0.0f), lastRiftMemoryGain(0.0f),
	      orbitPoleNumer(0.0f), orbitPoleDenom(0.0f), orbitPole(0.0f),
	      lastOrbitEdge(0.0f), lastOrbitSigma(0.0f),
	      lastOrbitHorizontalRatio(1.0f), lastOrbitMemoryGain(0.0f),
	      externalSparrowTrust(1.0f),
	      mu(0.01f), lastBaselineRate(0.0f),
	      step(0ULL), initialized(false)
	{
	}

	void reset()
	{
		m = n = r = activeRank = complementRank = activeComplementRank = 0u;
		sparrowModeRank = 1u;
		trialComplementRank = 0u;
		trialComplementWins = 0u;
		U.clear();
		fisherDiag.clear();
		V.clear();
		complementBlock.clear();
		scoutBasis.clear();
		scoutCov.clear();
		scoutNoise.clear();
		prevGz.clear();
		prevPrevGz.clear();
		prevGv.clear();
		resolveGzHistory.clear();
		heroGwHistory.clear();
		sparrowPrevActive.clear();
		sparrowPrevScout.clear();
		sparrowFutureCov.clear();
		sparrowPastCov.clear();
		sparrowCrossCov.clear();
		sparrowLeftMode.clear();
		sparrowRightMode.clear();
		sparrowLatent.clear();
		qbrtLeftMode.clear();
		qbrtLatent.clear();
		qrcLeftMode.clear();
		qrcLatent.clear();
		riftLeftMode.clear();
		riftLatent.clear();
		orbitPrevSignal.clear();
		orbitLeftMode.clear();
		orbitLatent.clear();
		scratch_gz.clear();
		scratch_corrected.clear();
		scratch_gv.clear();
		scratch_correctedV.clear();
		scratch_gwScout.clear();
		scratch_U_old.clear();
		scratch_f_old.clear();
		scratch_B.clear();
		scratch_Z.clear();
		scratch_overlap.clear();
		scratch_prevGzOld.clear();
		scratch_prevPrevGzOld.clear();
		scratch_resolveGzHistoryOld.clear();
		scratch_heroGwHistoryOld.clear();
		scratch_basisPacked.clear();
		scratch_V_old.clear();
		scratch_Bv.clear();
		scratch_Zv.clear();
		scratch_W_old.clear();
		scratch_Bw.clear();
		scratch_Zw.clear();
		scratch_complementMat.clear();
		scratch_complementEigVec.clear();
		scratch_complementEigVal.clear();
		scratch_scoutMat.clear();
		scratch_scoutEigVec.clear();
		scratch_scoutEigVal.clear();
		scratch_sparrowActive.clear();
		scratch_sparrowScout.clear();
		scratch_sparrowPastSignal.clear();
		scratch_geodeRhsCol.clear();
		scratch_geodeInvDiagCol.clear();
		scratch_geodeActiveCurrent.clear();
		scratch_geodeActiveDelta.clear();
		scratch_geodeSystemMat.clear();
		scratch_geodeRhs.clear();
		scratch_geodeSolution.clear();
		complementFisher = 0.0f;
		totalTrace = 0.0f;
		sigma2 = 0.0f;
		trialComplementMean = 0.0f;
		trialComplementVar = 0.0f;
		lastPredictiveEdge = 0.0f;
		lastMemoryGain = 0.0f;
		lastResolveEdge = 0.0f;
		lastResolveKernelRho = 0.0f;
		lastResolveMemoryGain = 0.0f;
		lastHeroEdge = 0.0f;
		lastHeroSigma = 0.0f;
		lastHeroMemoryGain = 0.0f;
		lastCobaltEdge = 0.0f;
		lastCobaltSigma = 0.0f;
		lastCobaltMemoryGain = 0.0f;
		lastBirchEdge = 0.0f;
		lastBirchSigma = 0.0f;
		lastBirchMemoryGain = 0.0f;
		lastGhostEdge = 0.0f;
		lastGhostSigma = 0.0f;
		lastGhostHorizontalRatio = 1.0f;
		lastGhostMemoryGain = 0.0f;
		sparrowPoleNumer = 0.0f;
		sparrowPoleDenom = 0.0f;
		sparrowPole = 0.0f;
		lastSparrowEdge = 0.0f;
		lastSparrowSigma = 0.0f;
		lastSparrowSecondEdge = 0.0f;
		lastSparrowSecondSigma = 0.0f;
		lastSparrowHorizontalRatio = 1.0f;
		lastSparrowMemoryGain = 0.0f;
		lastSparrowActiveModes = 0u;
		qbrtPoleNumer = 0.0f;
		qbrtPoleDenom = 0.0f;
		qbrtPole = 0.0f;
		lastQbrtEdge = 0.0f;
		lastQbrtSigma = 0.0f;
		lastQbrtHorizontalRatio = 1.0f;
		lastQbrtMemoryGain = 0.0f;
		qrcPoleNumer = 0.0f;
		qrcPoleDenom = 0.0f;
		qrcPole = 0.0f;
		lastQrcEdge = 0.0f;
		lastQrcSigma = 0.0f;
		lastQrcHorizontalRatio = 1.0f;
		lastQrcControlGain = 0.0f;
		lastQrcMemoryGain = 0.0f;
		riftPoleNumer = 0.0f;
		riftPoleDenom = 0.0f;
		riftPole = 0.0f;
		lastRiftEdge = 0.0f;
		lastRiftSigma = 0.0f;
		lastRiftHorizontalRatio = 1.0f;
		lastRiftAreaEnergy = 0.0f;
		lastRiftPredR2 = 0.0f;
		lastRiftMemoryGain = 0.0f;
		orbitPoleNumer = 0.0f;
		orbitPoleDenom = 0.0f;
		orbitPole = 0.0f;
		lastOrbitEdge = 0.0f;
		lastOrbitSigma = 0.0f;
		lastOrbitHorizontalRatio = 1.0f;
		lastOrbitMemoryGain = 0.0f;
		externalSparrowTrust = 1.0f;
		mu = 0.01f;
		lastBaselineRate = 0.0f;
		step = 0ULL;
		initialized = false;
	}
};

// Blockwise matrix-preconditioner state.
//
// BiMAP-lite keeps only row/column second-moment scaling. BiMAP-v2 extends that
// with low-rank row/column factors extracted by subspace iteration on the
// current gradient matrix, then applies a two-sided Woodbury inverse around the
// Adam-style per-element backbone.
struct BiMAPWeightState
{
	unsigned int m;
	unsigned int n;
	std::vector<float> rowSecond;   // [m] EMA row second moments
	std::vector<float> colSecond;   // [n] EMA column second moments
	std::vector<float> prevMhat;    // [m * n] previous bias-corrected first moment
	std::vector<float> scratchRow;  // [m]
	std::vector<float> scratchCol;  // [n]
	std::vector<float> rowBasis;    // [m * rowRank] low-rank row factors
	std::vector<float> colBasis;    // [n * colRank] low-rank column factors
	std::vector<float> rowEigVal;   // [rowRank] retained normalized row energies
	std::vector<float> colEigVal;   // [colRank] retained normalized column energies
	unsigned int rowRank;
	unsigned int colRank;
	float lastPredictiveTrust;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	float lastRowCapture;
	float lastColCapture;
	unsigned long long step;
	bool initialized;

	BiMAPWeightState()
	    : m(0u), n(0u),
	      rowRank(0u), colRank(0u),
	      lastPredictiveTrust(0.0f),
	      lastRowAnisotropy(1.0f),
	      lastColAnisotropy(1.0f),
	      lastRowCapture(0.0f),
	      lastColCapture(0.0f),
	      step(0ULL),
	      initialized(false)
	{
	}

	void reset()
	{
		m = 0u;
		n = 0u;
		rowSecond.clear();
		colSecond.clear();
		prevMhat.clear();
		scratchRow.clear();
		scratchCol.clear();
		rowBasis.clear();
		colBasis.clear();
		rowEigVal.clear();
		colEigVal.clear();
		rowRank = 0u;
		colRank = 0u;
		lastPredictiveTrust = 0.0f;
		lastRowAnisotropy = 1.0f;
		lastColAnisotropy = 1.0f;
		lastRowCapture = 0.0f;
		lastColCapture = 0.0f;
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

// Initialize BiMAP-lite state for a weight matrix of dimensions [m x n].
void initBiMAPWeightState(BiMAPWeightState& state, unsigned int m, unsigned int n);

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

// Apply one BiMAP-lite step to a matrix block.
//
// W/m/v2/gW are [m * n] row-major buffers. The first/second moments are
// Adam-style; BiMAP augments them with row/column block factors.
bool bimapUpdate(BiMAPWeightState& state,
                 float* W, float* m1, float* v2, float* gW,
                 unsigned int m, unsigned int n,
                 float lr,
                 float beta1, float beta2,
                 float inv1mB1t, float inv1mB2t,
                 float eps,
                 float invBatch, float gradScale,
                 float wd1, float wd2,
                 const ATLASConfig& ac,
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
