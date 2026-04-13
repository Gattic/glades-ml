// GPU-accelerated ATLAS optimizer (BRSP variant).
//
// Mirrors the CPU atlas_optimizer.h interface but operates on device memory,
// using cuBLAS for GEMMs and custom kernels for elementwise/reduction ops.
//
// Determinism: GPU ATLAS uses deterministic two-pass reductions (no atomicAdd)
// and synchronous D2H transfers, so sigma2 and mu are updated in the same step
// as on CPU. Given the same RNG seed and inputs, GPU and CPU produce the same
// training trajectory (within float32 rounding from cuBLAS vs CPU GEMM).
#pragma once

#include "gpu_buffer.h"
#include "../training_config.h"
#include "../../rng.h"
#include <cstddef>
#include <vector>

namespace shmea { class GLogger; }

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// Per-weight-matrix ATLAS optimizer state on GPU.
// Mirrors atlas::WeightState but uses GpuBuffer for device allocations.
//
// Dual-space subspace selection:
//   When m <= n: left subspace U ∈ R^{m×r}, gz ∈ R^{r×n} (row-major, direction = row)
//   When m >  n: right subspace V ∈ R^{n×r}, gz ∈ R^{m×r} (row-major, direction = col)
// This ensures the subspace always operates in the smaller dimension for
// better coverage with fixed rank r.
struct GpuAtlasWeightState
{
	unsigned int m;     // rows of weight matrix
	unsigned int n;     // cols of weight matrix
	unsigned int r;     // subspace rank
	unsigned int complementRank; // allocated complement-block rank (>= 1 for storage)
	unsigned int activeComplementRank; // runtime active residual rank (<= complementRank)
	unsigned int trialComplementRank; // probationary target rank awaiting promotion
	unsigned int trialComplementWins; // consecutive accepted control boundaries
	bool rightSubspace; // true when m > n (use right singular vectors)

	// subDim = min(m,n): dimension the subspace basis lives in
	// outerDim = max(m,n): the other dimension
	// U/V: [subDim * r] orthonormal subspace basis (row-major)
	// gz/gPred/prevGz: [outerDim * r] projected gradient
	//   Left subspace (rightSubspace=false): gz[r, n] — direction c is row c
	//   Right subspace (rightSubspace=true):  gz[m, r] — direction c is col c
	// activeComplementRank: online active prefix inside the retained residual block

	GpuBuffer<float> U;          // [subDim * r] orthonormal subspace basis
	GpuBuffer<float> fisherDiag; // [r] EMA of Fisher eigenvalues
	GpuBuffer<float> V;          // [subDim * complementRank] residual complement basis
	GpuBuffer<float> complementBlock; // [complementRank * complementRank] dense residual covariance EMA
	GpuBuffer<float> complementFisher; // [1] trace(complementBlock)
	GpuBuffer<float> prevGz;     // [outerDim * r] previous compressed gradient
	GpuBuffer<float> prevGv;     // [outerDim * complementRank] previous complement-block gradient

	// Scratch buffers (persistent to avoid per-step allocation)
	GpuBuffer<float> gz;         // [outerDim * r] current projected gradient
	GpuBuffer<float> gPred;      // [outerDim * r] scaled prediction for correction SGEMM
	GpuBuffer<float> gv;         // [outerDim * complementRank] current complement-block gradient
	GpuBuffer<float> gPredV;     // [outerDim * complementRank] scaled complement correction
	GpuBuffer<float> complementMat; // [complementRank * complementRank] dense complement correction matrix
	GpuBuffer<float> d_reduce;   // [2] reduction output (errNormSq, gzNormSq)
	GpuBuffer<float> d_partials; // [512] per-block partial sums for deterministic reductions

	// Subspace refresh scratch
	GpuBuffer<float> U_old;      // [subDim * r]
	GpuBuffer<float> f_old;      // [r]
	GpuBuffer<float> B;          // [outerDim * r] power iteration intermediate
	GpuBuffer<float> overlap;    // [r * r * 2]
	GpuBuffer<float> prevGzOld;  // [outerDim * r]
	GpuBuffer<float> qrTemp;     // [subDim * r] scratch for Cholesky QR SGEMM output
	GpuBuffer<float> V_old;      // [subDim * complementRank]
	GpuBuffer<float> Bv;         // [outerDim * complementRank]
	GpuBuffer<float> Zv;         // [subDim * complementRank]
	bool refreshAllocated;       // true after refresh scratch buffers allocated

	// Host-side scalars (passed to kernels as parameters, updated on host)
	float totalTrace;
	float sigma2;
	float trialComplementMean;
	float trialComplementVar;
	float mu;
	unsigned long long step;
	bool initialized;
	float lastBaselineRate;

	GpuAtlasWeightState()
	    : m(0u), n(0u), r(0u), complementRank(0u), activeComplementRank(0u),
	      trialComplementRank(0u), trialComplementWins(0u), rightSubspace(false),
	      refreshAllocated(false),
	      totalTrace(0.0f), sigma2(0.0f), trialComplementMean(0.0f), trialComplementVar(0.0f),
	      mu(0.01f), step(0ULL), initialized(false),
	      lastBaselineRate(0.0f)
	{
	}

private:
	// Non-copyable (GpuBuffer resources are owned exclusively).
	GpuAtlasWeightState(const GpuAtlasWeightState&);
	GpuAtlasWeightState& operator=(const GpuAtlasWeightState&);
};

// Initialize ATLAS state for a weight matrix [m x n] with subspace rank r.
// Allocates all device buffers and initializes U with random orthonormal basis.
// rng: network RNG engine (generates U on CPU for determinism, then uploads to GPU).
bool atlas_gpu_init(GpuAtlasWeightState& state,
                    unsigned int m, unsigned int n,
                    unsigned int rank, float muInit,
                    glades::rng::Engine& rng);

// Apply one ATLAS optimizer step on GPU (BRSP variant).
// d_W: device pointer to weight matrix [m*n]
// d_gW: device pointer to gradient matrix [m*n] (cleared to zero after use)
// Returns true on success.
bool atlas_gpu_step(GpuAtlasWeightState& state,
                    float* d_W, float* d_gW,
                    unsigned int m, unsigned int n,
                    float invBatch, float lr,
                    float wd1, float wd2, float gradScale,
                    const glades::ATLASConfig& ac,
                    shmea::GLogger* logger = 0,
                    const char* tag = 0);

// Retrieve diagnostic info from the last step (for logging).
// Only meaningful when step % tSub == 0.
struct AtlasGpuDiag
{
	float sigma2;
	float mu;
	float baselineRate;
	float gzNorm;
	float updateNorm;
	float fisherMin, fisherMax, fisherMean;
	float fisherMedian;
	float corrScaleMin, corrScaleMax;

	// Spectral efficiency metrics (adaptive rank diagnostics)
	float effectiveRank;      // exp(entropy of Fisher distribution), in [1, r]
	float spectralEfficiency; // effectiveRank / r, in [0, 1]
	float top1Concentration;  // fraction of total Fisher in strongest direction
	float top10Concentration; // fraction of total Fisher in top-10 directions

	unsigned long long step;
	bool rightSubspace;
	bool valid;
	AtlasGpuDiag() : sigma2(0), mu(0), baselineRate(0), gzNorm(0), updateNorm(0),
	                  fisherMin(0), fisherMax(0), fisherMean(0), fisherMedian(0),
	                  corrScaleMin(0), corrScaleMax(0),
	                  effectiveRank(0), spectralEfficiency(0),
	                  top1Concentration(0), top10Concentration(0),
	                  step(0), rightSubspace(false), valid(false) {}
};

// Lightweight GPU PACT state.
//
// This intentionally implements only the PACT-lite path:
// - AdamW backbone lives in the shared batched Adam kernel
// - row/column anisotropy is tracked with diagonal second moments
// - promotion is cadence-gated and host-scored from tiny downloaded summaries
// - no low-rank factor extraction and no heavy sidecar controller
struct GpuPactWeightState
{
	unsigned int m;
	unsigned int n;

	GpuBuffer<float> rowSecond;   // [m] EMA row second moments
	GpuBuffer<float> colSecond;   // [n] EMA col second moments
	GpuBuffer<float> colScratch;  // [n] raw column g^2 sums for the current refresh
	GpuBuffer<float> gainScratch; // [2] adamGainSum, precondGainSum

	std::vector<float> hostRowSecond;
	std::vector<float> hostColSecond;
	std::vector<float> hostColScratch;

	float rowMean;
	float colMean;
	float promotionScore;
	float lastAdamGain;
	float lastPrecondGain;
	float lastCostPenalty;
	float lastPromotionMargin;
	float lastPredictiveTrust;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	bool promoted;
	unsigned long long promotedSteps;
	unsigned long long step;
	bool initialized;

	GpuPactWeightState()
	    : m(0u), n(0u),
	      rowMean(1.0e-12f), colMean(1.0e-12f),
	      promotionScore(0.0f),
	      lastAdamGain(0.0f), lastPrecondGain(0.0f),
	      lastCostPenalty(0.0f), lastPromotionMargin(0.0f),
	      lastPredictiveTrust(0.0f),
	      lastRowAnisotropy(1.0f), lastColAnisotropy(1.0f),
	      promoted(false), promotedSteps(0ULL), step(0ULL),
	      initialized(false)
	{
	}
};

// Lightweight GPU BiMAP state.
//
// This intentionally implements only the BiMAP-lite path:
// - exact AdamW backbone stays in the shared batched Adam kernel
// - row/column anisotropy is tracked with diagonal second moments
// - the BiMAP residual is formed fully on device with no per-element host work
// - low-rank BiMAP-v2 remains out of scope for the GPU fast path
struct GpuBiMAPWeightState
{
	unsigned int m;
	unsigned int n;
	unsigned int storageRank;
	unsigned int rowRank;
	unsigned int colRank;

	GpuBuffer<float> rowSecond;   // [m] EMA row second moments
	GpuBuffer<float> colSecond;   // [n] EMA col second moments
	GpuBuffer<float> colScratch;  // [n] raw column g^2 sums for the current refresh
	GpuBuffer<float> prevMhat;    // [m * n] previous bias-corrected first moment
	GpuBuffer<float> stepMatrix;  // [m * n] BiMAP-v2 preconditioned step matrix
	GpuBuffer<float> rowMetric;   // [m] diagonal row metric
	GpuBuffer<float> colMetric;   // [n] diagonal column metric
	GpuBuffer<float> rowBasis;    // [m * r] low-rank row basis
	GpuBuffer<float> colBasis;    // [n * r] low-rank column basis
	GpuBuffer<float> rowWork;     // [m * r] QR temp / scaled row basis
	GpuBuffer<float> colWork;     // [n * r] QR temp / scaled column basis
	GpuBuffer<float> rowProj;     // [r * n] row basis projection / solve scratch
	GpuBuffer<float> colProj;     // [m * r] column basis projection / solve scratch
	GpuBuffer<float> factorScratch; // [max(m, n) * r] factor solve scratch
	GpuBuffer<float> rowEigVal;   // [r] retained normalized row energies
	GpuBuffer<float> colEigVal;   // [r] retained normalized column energies
	GpuBuffer<float> coreScratch; // [2 * r * r] small SPD core + Cholesky scratch
	GpuBuffer<float> scalarScratch; // [8] stats / trust / capture scratch

	std::vector<float> hostRowSecond;
	std::vector<float> hostColSecond;
	std::vector<float> hostColScratch;

	float rowMean;
	float colMean;
	float lastPredictiveTrust;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	float lastRowCapture;
	float lastColCapture;
	unsigned long long step;
	bool initialized;

	GpuBiMAPWeightState()
	    : m(0u), n(0u), storageRank(0u), rowRank(0u), colRank(0u),
	      rowMean(1.0e-12f), colMean(1.0e-12f),
	      lastPredictiveTrust(0.0f),
	      lastRowAnisotropy(1.0f), lastColAnisotropy(1.0f),
	      lastRowCapture(0.0f), lastColCapture(0.0f),
	      step(0ULL), initialized(false)
	{
	}
};

// Lightweight GPU ECHO state.
//
// ECHO tracks separable row/column geometry directly from backward operands
// and applies a two-sided diagonal residual on top of the AdamW backbone.
struct GpuEchoWeightState
{
	unsigned int m;
	unsigned int n;

	GpuBuffer<float> rowSecond;     // [m] EMA row geometry from adjoint operands
	GpuBuffer<float> colSecond;     // [n] EMA col geometry from input operands
	GpuBuffer<float> rowInvMetric;  // [m] cached inverse row metric for the base ECHO anchor
	GpuBuffer<float> colInvMetric;  // [n] cached inverse col metric for the base ECHO anchor
	GpuBuffer<float> rowStructInvMetric; // [m] cached inverse row metric for the structural anchor
	GpuBuffer<float> colStructInvMetric; // [n] cached inverse col metric for the structural anchor
	GpuBuffer<float> prevMhat;      // [m * n] previous bias-corrected first moment
	GpuBuffer<float> scalarScratch; // row/col stats plus simplex trust weights and maturity

	float rowMean;
	float colMean;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	float lastGeometryScale;
	float lastGeometryTrust;
	float lastPredictiveTrust;
	float lastStructuralTrust;
	float lastMaturity;
	unsigned long long step;
	bool initialized;

	GpuEchoWeightState()
	    : m(0u), n(0u),
	      rowMean(1.0f), colMean(1.0f),
	      lastRowAnisotropy(1.0f), lastColAnisotropy(1.0f),
	      lastGeometryScale(0.0f),
	      lastGeometryTrust(1.0f),
	      lastPredictiveTrust(0.0f),
	      lastStructuralTrust(0.0f),
	      lastMaturity(0.0f),
	      step(0ULL), initialized(false)
	{
	}
};

struct GpuEchoObserveEntry
{
	const float* rowObs;
	const float* colObs;
	float* rowSecond;
	float* colSecond;
	int samples;
	int rows;
	int cols;
	int featureBase;

	GpuEchoObserveEntry()
	    : rowObs(0),
	      colObs(0),
	      rowSecond(0),
	      colSecond(0),
	      samples(0),
	      rows(0),
	      cols(0),
	      featureBase(0)
	{
	}
};

// Lightweight GPU RACER state.
//
// This is the minimal GPU RACER-lite path:
// - exact AdamW backbone stays in the shared batched Adam kernel
// - row/column anisotropy is tracked with cadence-gated diagonal statistics
// - stable-signal and previous-momentum EMAs stay resident on device
// - promotion is scored from tiny scalar summaries rather than host-side full
//   matrix reconstruction
// - all non-promoted blocks degenerate exactly to the AdamW backbone
struct GpuRacerWeightState
{
	unsigned int m;
	unsigned int n;

	GpuBuffer<float> rowSecond;   // [m] EMA row second moments
	GpuBuffer<float> colSecond;   // [n] EMA column second moments
	GpuBuffer<float> colScratch;  // [n] raw column g^2 sums for the current refresh
	GpuBuffer<float> gainScratch; // [2] adamRewardSum, racerRewardSum
	GpuBuffer<float> corrScratch; // [4] dot, curNorm, prevNorm, reserved
	GpuBuffer<float> prevMhat;    // [m * n] previous bias-corrected first moment
	GpuBuffer<float> stableMhat;  // [m * n] delayed stable-signal EMA
	GpuBuffer<float> adamStep;    // [m * n] exact AdamW backbone step
	GpuBuffer<float> racerStep;   // [m * n] promoted RACER step

	std::vector<float> hostRowSecond;
	std::vector<float> hostColSecond;
	std::vector<float> hostColScratch;

	float rowMean;
	float colMean;
	float promotionScore;
	float lastAdamGain;
	float lastPrecondGain;
	float lastCostPenalty;
	float lastPromotionMargin;
	float lastPredictiveTrust;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	bool promoted;
	unsigned long long promotedSteps;
	unsigned long long step;
	bool initialized;

	GpuRacerWeightState()
	    : m(0u), n(0u),
	      rowMean(1.0e-12f), colMean(1.0e-12f),
	      promotionScore(0.0f),
	      lastAdamGain(0.0f), lastPrecondGain(0.0f),
	      lastCostPenalty(0.0f), lastPromotionMargin(0.0f),
	      lastPredictiveTrust(0.0f),
	      lastRowAnisotropy(1.0f), lastColAnisotropy(1.0f),
	      promoted(false), promotedSteps(0ULL), step(0ULL),
	      initialized(false)
	{
	}
};

// Lightweight GPU MATRA state.
//
// MATRA keeps AdamW as the exact backbone, tracks cheap row/column anisotropy
// statistics on device, builds a two-sided geometry candidate from those
// statistics, and optionally forms an exact MUON-style orthogonal candidate on
// trusted matrix blocks.
struct GpuMatraWeightState
{
	unsigned int m;
	unsigned int n;

	GpuBuffer<float> rowSecond;   // [m] EMA row second moments
	GpuBuffer<float> colSecond;   // [n] EMA column second moments
	GpuBuffer<float> colScratch;  // [n] raw column g^2 sums for the current refresh
	GpuBuffer<float> prevMhat;    // [m * n] previous bias-corrected first moment
	GpuBuffer<float> backboneStep; // [m * n] exact pre-applied Adam step on GPU
	GpuBuffer<float> adamStep;    // [m * n] MATRA Adam prior step after predictive transport
	GpuBuffer<float> geomStep;    // [m * n] two-sided geometry candidate
	GpuBuffer<float> orthStep;    // [m * n] orthogonal candidate
	GpuBuffer<float> coreScratch; // [2 * coreDim * coreDim] Gram + factor scratch
	GpuBuffer<float> scalarScratch; // [20] means / trust / eligibility / scale / corr stats

	float rowMean;
	float colMean;
	float lastPredictiveTrust;
	float lastGeometryTrust;
	float lastOrthTrust;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	float lastAspect;
	float lastSignalScale;
	float lastOrthError;
	bool lastEligible;
	unsigned long long step;
	bool initialized;

	GpuMatraWeightState()
	    : m(0u), n(0u),
	      rowMean(1.0e-12f), colMean(1.0e-12f),
	      lastPredictiveTrust(0.0f),
	      lastGeometryTrust(0.0f),
	      lastOrthTrust(0.0f),
	      lastRowAnisotropy(1.0f), lastColAnisotropy(1.0f),
	      lastAspect(1.0f),
	      lastSignalScale(0.0f),
	      lastOrthError(0.0f),
	      lastEligible(false),
	      step(0ULL), initialized(false)
	{
	}
};

struct GpuMatraBatchItem
{
	GpuMatraWeightState* state;
	float* d_W;
	float* d_gW;
	float* d_m;
	float* d_v;
	float* d_rowSecond;
	float* d_colSecond;
	float* d_colScratch;
	float* d_prevMhat;
	float* d_backboneStep;
	float* d_adamStep;
	float* d_geomStep;
	float* d_orthStep;
	float* d_coreScratch;
	float* d_scalarScratch;
	unsigned int m;
	unsigned int n;
	unsigned int predictiveEnabled;
	unsigned int refreshMetrics;
	float lr;
	const char* tag;

	GpuMatraBatchItem()
	    : state(0),
	      d_W(0), d_gW(0), d_m(0), d_v(0),
	      d_rowSecond(0), d_colSecond(0), d_colScratch(0),
	      d_prevMhat(0), d_backboneStep(0), d_adamStep(0),
	      d_geomStep(0), d_orthStep(0), d_coreScratch(0), d_scalarScratch(0),
	      m(0u), n(0u), predictiveEnabled(0u), refreshMetrics(0u), lr(0.0f), tag(0)
	{
	}
};

// Lightweight GPU MUON state.
//
// This is the device-native MUON-lite path:
// - AdamW backbone stays in the shared batched Adam kernel
// - eligible blocks keep all momentum history and orthogonalization scratch
//   resident on device
// - the MUON residual is formed on the compute stream with no per-step host
//   downloads or uploads
// - all non-eligible blocks degenerate exactly to the AdamW backbone
struct GpuMuonWeightState
{
	unsigned int m;
	unsigned int n;

	GpuBuffer<float> prevMhat;      // [m * n] previous bias-corrected first moment
	GpuBuffer<float> adamStep;      // [m * n] current Adam-style step matrix
	GpuBuffer<float> muonStep;      // [m * n] orthogonalized MUON direction
	GpuBuffer<float> coreScratch;   // [2 * coreDim * coreDim] Gram + Cholesky scratch
	GpuBuffer<float> scalarScratch; // [8] dot, curNorm, prevNorm, trust, froSq, signalScale, trace, spare

	float lastPredictiveTrust;
	float lastAspect;
	float lastSignalScale;
	float lastOrthError;
	bool lastEligible;
	unsigned long long step;
	bool initialized;

	GpuMuonWeightState()
	    : m(0u), n(0u),
	      lastPredictiveTrust(0.0f),
	      lastAspect(1.0f),
	      lastSignalScale(0.0f),
	      lastOrthError(0.0f),
	      lastEligible(false),
	      step(0ULL),
	      initialized(false)
	{
	}
};

struct GpuMuonBatchItem
{
	GpuMuonWeightState* state;
	float* d_W;
	float* d_gW;
	float* d_m;
	float* d_v;
	float* d_prevMhat;
	float* d_adamStep;
	float* d_muonStep;
	float* d_coreScratch;
	float* d_scalarScratch;
	unsigned int m;
	unsigned int n;
	unsigned int predictiveEnabled;
	float lr;
	const char* tag;

	GpuMuonBatchItem()
	    : state(0),
	      d_W(0), d_gW(0), d_m(0), d_v(0),
	      d_prevMhat(0), d_adamStep(0), d_muonStep(0), d_coreScratch(0), d_scalarScratch(0),
	      m(0u), n(0u), predictiveEnabled(0u), lr(0.0f), tag(0)
	{
	}
};

// Convenience wrapper: initializes state if needed, then calls atlas_gpu_step.
// Mirrors the CPU atlas::update() function. Returns true on success.
bool atlas_gpu_update(GpuAtlasWeightState& state,
                      float* d_W, float* d_gW,
                      unsigned int m, unsigned int n,
                      float invBatch, float lr,
                      float wd1, float wd2, float gradScale,
                      const glades::ATLASConfig& ac,
                      glades::rng::Engine& rng,
                      shmea::GLogger* logger = 0,
                      const char* tag = 0);

// Residual-only ATLAS update.
// Uses the same subspace tracking and Fisher adaptation as atlas_gpu_update,
// but skips decoupled weight decay and the full-space baseline step so callers
// can layer ATLAS geometry on top of another backbone optimizer.
bool atlas_gpu_residual_update(GpuAtlasWeightState& state,
                               float* d_W, float* d_gW,
                               unsigned int m, unsigned int n,
                               float invBatch, float lr,
                               float gradScale,
                               const glades::ATLASConfig& ac,
                               glades::rng::Engine& rng,
                               shmea::GLogger* logger = 0,
                               const char* tag = 0);

// GPU PACT-lite residual update on top of an AdamW backbone that has already
// updated W/m/v for the current step. The function:
// - refreshes row/column anisotropy statistics on cadence boundaries
// - scores promotion using predicted gain minus analytical cost
// - applies only the residual difference from AdamW to the promoted blocks
// - degenerates exactly to AdamW when promotion is off
bool pact_gpu_update_lite(GpuPactWeightState& state,
                          float* d_W, float* d_gW,
                          float* d_m, float* d_v,
                          unsigned int m, unsigned int n,
                          float lr,
                          float invBatch, float gradScale,
                          float inv1mB1t, float inv1mB2t,
                          float wd1, float eps,
                          const glades::ATLASConfig& ac,
                          shmea::GLogger* logger = 0,
                          const char* tag = 0);

// GPU BiMAP-lite residual update on top of an AdamW backbone that has already
// updated W/m/v for the current step. This path:
// - refreshes row/column anisotropy statistics on cadence boundaries,
// - applies the diagonal row/column BiMAP residual fully on device,
// - degenerates exactly to AdamW when geometry scale is zero.
// GPU BiMAP-v2 extends this with device-native low-rank row/column factors,
// predictive transport, and two-sided Woodbury-style residual application.
// The dispatcher picks lite vs v2 from `ac.bimapLowRankEnabled`.
bool bimap_gpu_update(GpuBiMAPWeightState& state,
                      float* d_W, float* d_gW,
                      float* d_m, float* d_v,
                      unsigned int m, unsigned int n,
                      float lr,
                      float invBatch, float gradScale,
                      float inv1mB1t, float inv1mB2t,
                      float eps,
                      const glades::ATLASConfig& ac,
                      shmea::GLogger* logger = 0,
                      const char* tag = 0);

bool bimap_gpu_update_lite(GpuBiMAPWeightState& state,
                           float* d_W, float* d_gW,
                           float* d_m, float* d_v,
                           unsigned int m, unsigned int n,
                           float lr,
                           float invBatch, float gradScale,
                           float inv1mB1t, float inv1mB2t,
                           float eps,
                           const glades::ATLASConfig& ac,
                           shmea::GLogger* logger = 0,
                           const char* tag = 0);

bool echo_gpu_observe(GpuEchoWeightState& state,
                      const float* d_rowObs,
                      const float* d_colObs,
                      unsigned int samples,
                      unsigned int m,
                      unsigned int n,
                      const glades::ATLASConfig& ac);

bool echo_gpu_ensure_state(GpuEchoWeightState& state,
                           unsigned int m,
                           unsigned int n);

bool echo_gpu_observe_batch(GpuEchoObserveEntry* d_entries,
                            int entryCapacity,
                            const GpuEchoObserveEntry* h_entries,
                            int entryCount,
                            int totalFeatures,
                            unsigned long long optimizerStep,
                            const glades::ATLASConfig& ac);

bool echo_gpu_launch_observe_batch(const GpuEchoObserveEntry* d_entries,
                                   int entryCount,
                                   int totalFeatures,
                                   unsigned long long optimizerStep,
                                   const glades::ATLASConfig& ac);

bool echo_gpu_prepare_metrics(GpuEchoWeightState& state,
                              unsigned int m,
                              unsigned int n,
                              unsigned long long optimizerStep,
                              float eps,
                              const glades::ATLASConfig& ac);

bool echo_gpu_prepare_metrics_batch(float** d_rowSecond,
                                    float** d_colSecond,
                                    float** d_rowInvMetric,
                                    float** d_colInvMetric,
                                    float** d_rowStructInvMetric,
                                    float** d_colStructInvMetric,
                                    float** d_scalarScratch,
                                    const int* d_rows,
                                    const int* d_cols,
                                    int groupCount,
                                    unsigned long long optimizerStep,
                                    float eps,
                                    const glades::ATLASConfig& ac);

bool echo_gpu_post_update(GpuEchoWeightState& state,
                          const glades::ATLASConfig& ac,
                          shmea::GLogger* logger = 0,
                          const char* tag = 0);

bool echo_gpu_update(GpuEchoWeightState& state,
                     float* d_W, float* d_gW,
                     float* d_m, float* d_v,
                     unsigned int m, unsigned int n,
                     float lr,
                     float gradScale,
                     unsigned long long optimizerStep,
                     float inv1mB1t, float inv1mB2t,
                     float eps,
                     const glades::ATLASConfig& ac,
                     shmea::GLogger* logger = 0,
                     const char* tag = 0);

// GPU RACER-lite residual update on top of an AdamW backbone that has already
// updated W/m/v for the current step. The function:
// - refreshes row/column anisotropy statistics on cadence boundaries
// - maintains stable and previous momentum estimates resident on device
// - scores promotion using stable reward minus curvature/noise/cost penalties
// - applies only the residual difference from AdamW to promoted blocks
// - degenerates exactly to AdamW when promotion is off
bool racer_gpu_update_lite(GpuRacerWeightState& state,
                           float* d_W, float* d_gW,
                           float* d_m, float* d_v,
                           unsigned int m, unsigned int n,
                           float lr,
                           float invBatch, float gradScale,
                           float inv1mB1t, float inv1mB2t,
                           float wd1, float eps,
                           const glades::ATLASConfig& ac,
                           shmea::GLogger* logger = 0,
                           const char* tag = 0);

// GPU MATRA residual update on top of an AdamW backbone that has already
// updated W/m/v for the current step. This path:
// - tracks row/column anisotropy with cheap diagonal EMAs,
// - blends trusted geometry and orthogonal candidates under a fixed residual budget,
// - degenerates exactly to AdamW when MATRA trust weights collapse to zero.
bool matra_gpu_update(GpuMatraWeightState& state,
                      float* d_W, float* d_gW,
                      float* d_m, float* d_v,
                      unsigned int m, unsigned int n,
                      float lr,
                      float invBatch, float gradScale,
                      float inv1mB1t, float inv1mB2t,
                      float eps,
                      const glades::ATLASConfig& ac,
                      shmea::GLogger* logger = 0,
                      const char* tag = 0);

// Batched MATRA update for small tall matrices with the same shape.
// This path batches the exact small-core orthogonal solve across eligible
// matrices while preserving the existing single-matrix fallback for all other
// shapes.
bool matra_gpu_update_small_batches(const GpuMatraBatchItem* items,
                                    int itemCount,
                                    const int* groupOffsets,
                                    const int* groupCounts,
                                    int groupCount,
                                    GpuMatraBatchItem* d_batchItems,
                                    float* d_statsBatch,
                                    float** d_corePtrScratch,
                                    float** d_stepPtrScratch,
                                    int* d_infoScratch,
                                    int corePtrCapacity,
                                    float invBatch, float gradScale,
                                    float inv1mB1t, float inv1mB2t,
                                    float eps,
                                    const glades::ATLASConfig& ac,
                                    shmea::GLogger* logger = 0);

// GPU MUON-lite residual update on top of an AdamW backbone that has already
// updated W/m/v for the current step. This path:
// - keeps previous momentum and scratch resident on device,
// - computes predictive trust and orthogonalized momentum on the compute stream,
// - applies only the residual difference from AdamW to eligible blocks,
// - degenerates exactly to AdamW for ineligible blocks.
bool muon_gpu_update_lite(GpuMuonWeightState& state,
                          float* d_W, float* d_gW,
                          float* d_m, float* d_v,
                          unsigned int m, unsigned int n,
                          float lr,
                          float inv1mB1t, float inv1mB2t,
                          float eps,
                          const glades::ATLASConfig& ac,
                          shmea::GLogger* logger = 0,
                          const char* tag = 0);

// Batched MUON-lite update for small tall matrices with the same shape.
// This path prepares each eligible matrix independently, batches the small
// Cholesky factorization across all cores to improve GPU occupancy, then
// completes the per-matrix triangular solve and residual application.
bool muon_gpu_update_lite_small_batch(const GpuMuonBatchItem* items,
                                      int count,
                                      GpuMuonBatchItem* d_batchItems,
                                      float** d_corePtrScratch,
                                      float** d_stepPtrScratch,
                                      int* d_infoScratch,
                                      int corePtrCapacity,
                                      float inv1mB1t, float inv1mB2t,
                                      float eps,
                                      const glades::ATLASConfig& ac,
                                      shmea::GLogger* logger = 0);

// Bulk batched MUON-lite update for multiple same-shape groups. The caller
// provides a flat item array plus per-group offsets/counts so the device batch
// descriptors can be uploaded once and reused across each grouped launch.
bool muon_gpu_update_lite_small_batches(const GpuMuonBatchItem* items,
                                        int itemCount,
                                        const int* groupOffsets,
                                        const int* groupCounts,
                                        int groupCount,
                                        GpuMuonBatchItem* d_batchItems,
                                        float** d_corePtrScratch,
                                        float** d_stepPtrScratch,
                                        int* d_infoScratch,
                                        int corePtrCapacity,
                                        float inv1mB1t, float inv1mB2t,
                                        float eps,
                                        const glades::ATLASConfig& ac,
                                        shmea::GLogger* logger = 0);

// Retrieve diagnostic info from the current state.
// Downloads Fisher diagonal from GPU — call sparingly (e.g. every tSub steps).
AtlasGpuDiag atlas_gpu_get_diag(const GpuAtlasWeightState& state);

// --- Individual CUDA kernels (host wrappers) ---

// Gram-Schmidt orthonormalization of Q[m, r] on GPU (row-major).
bool atlas_gpu_gram_schmidt(float* d_Q, int m, int r);

// Elementwise weight decay: W[i] -= lr * (wd1*sign(W[i]) + wd2*W[i])
bool atlas_gpu_weight_decay(float* d_W, size_t mn, float lr, float wd1, float wd2);

// Baseline update: W[i] -= baseScaled * gW[i]
bool atlas_gpu_baseline_update(float* d_W, const float* d_gW, size_t mn, float baseScaled);

// Fisher diagonal update: EMA of mean-squared projected gradient per subspace direction.
// Left subspace:  gz[r, n], direction c = row c (contiguous, stride=n)
// Right subspace: gz[m, r], direction c = col c (strided, stride=r)
// outerDim: n (left) or m (right) — number of elements per direction
bool atlas_gpu_fisher_update(const float* d_gz, float* d_fisherDiag,
                              int r, int outerDim, float beta, float sampleScale,
                              bool rightSubspace, bool bootstrap);

// Prepare scaled prediction for subspace correction.
// Left subspace:  gz/out are [r, outerDim], direction c = row c
// Right subspace: gz/out are [outerDim, r], direction c = col c
// corrScale[c] = baselineRate - min(lr/(fisherDiag[c]*bcFactor+eps), kappaLr)
bool atlas_gpu_prepare_correction(const float* d_gz, const float* d_prevGz,
                                   const float* d_fisherDiag,
                                   float* d_out, int r, int outerDim,
                                   float onePlusMu, float negMu,
                                   float baselineRate, float lr, float eps,
                                   float kappaLr, float bcFactor,
                                   bool rightSubspace);

// Compute norms for mu adaptation (deterministic two-pass reduction):
// d_out[0] = sum((gz[i]-prevGz[i])^2), d_out[1] = sum(gz[i]^2)
// d_partials: device scratch buffer of at least 2*min(256,ceil(rn/256)) floats.
bool atlas_gpu_mu_norms(const float* d_gz, const float* d_prevGz,
                         size_t rn, float* d_out, float* d_partials);

// EMA blend: dst[i] = (1-beta)*a[i] + beta*b[i]
bool atlas_gpu_ema_blend(float* d_dst, const float* d_a, const float* d_b,
                          size_t count, float beta);

// Transform Fisher diagonal into new basis:
// f_new[c] = sum_j overlap[c*r+j]^2 * f_old[j]
bool atlas_gpu_transform_fisher(const float* d_overlap, const float* d_f_old,
                                  float* d_f_new, int r);

// Scale gradient for refresh: out[i] = gW[i] * gScale
bool atlas_gpu_scale_grad(const float* d_gW, float* d_out, size_t mn, float gScale);

// Guard: replace NaN/Inf values with 0.0f. Returns true on success.
bool atlas_gpu_guard(float* d_W, size_t mn);

// Transform prevGz into new basis: prevGz_new = overlap * prevGz_old
// overlap[r,r] * prevGzOld[r,n] -> prevGz[r,n]
// Uses cuBLAS sgemm_rowmajor.

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

struct GpuAtlasWeightState
{
	float totalTrace;
	float sigma2;
	float mu;
	unsigned long long step;
	bool initialized;
	GpuAtlasWeightState() : totalTrace(0.0f), sigma2(0.0f), mu(0.0f), step(0ULL), initialized(false) {}
};

struct AtlasGpuDiag
{
	bool valid;
	AtlasGpuDiag() : valid(false) {}
};

struct GpuPactWeightState
{
	float promotionScore;
	float lastAdamGain;
	float lastPrecondGain;
	float lastCostPenalty;
	float lastPromotionMargin;
	float lastPredictiveTrust;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	bool promoted;
	unsigned long long promotedSteps;
	unsigned long long step;
	bool initialized;
	GpuPactWeightState()
	    : promotionScore(0.0f), lastAdamGain(0.0f), lastPrecondGain(0.0f),
	      lastCostPenalty(0.0f), lastPromotionMargin(0.0f), lastPredictiveTrust(0.0f),
	      lastRowAnisotropy(1.0f), lastColAnisotropy(1.0f),
	      promoted(false), promotedSteps(0ULL), step(0ULL), initialized(false) {}
};

struct GpuBiMAPWeightState
{
	float rowMean;
	float colMean;
	float lastPredictiveTrust;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	float lastRowCapture;
	float lastColCapture;
	unsigned int rowRank;
	unsigned int colRank;
	unsigned long long step;
	bool initialized;
	GpuBiMAPWeightState()
	    : rowMean(1.0f), colMean(1.0f),
	      lastPredictiveTrust(0.0f),
	      lastRowAnisotropy(1.0f), lastColAnisotropy(1.0f),
	      lastRowCapture(0.0f), lastColCapture(0.0f),
	      rowRank(0u), colRank(0u),
	      step(0ULL), initialized(false) {}
};

struct GpuEchoWeightState
{
	float rowMean;
	float colMean;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	float lastGeometryScale;
	float lastGeometryTrust;
	float lastPredictiveTrust;
	float lastStructuralTrust;
	float lastMaturity;
	unsigned long long step;
	bool initialized;
	GpuEchoWeightState()
	    : rowMean(1.0f), colMean(1.0f),
	      lastRowAnisotropy(1.0f), lastColAnisotropy(1.0f),
	      lastGeometryScale(0.0f),
	      lastGeometryTrust(1.0f),
	      lastPredictiveTrust(0.0f),
	      lastStructuralTrust(0.0f),
	      lastMaturity(0.0f),
	      step(0ULL), initialized(false) {}
};

struct GpuEchoObserveEntry
{
	const float* rowObs;
	const float* colObs;
	float* rowSecond;
	float* colSecond;
	int samples;
	int rows;
	int cols;
	int featureBase;

	GpuEchoObserveEntry()
	    : rowObs(0),
	      colObs(0),
	      rowSecond(0),
	      colSecond(0),
	      samples(0),
	      rows(0),
	      cols(0),
	      featureBase(0)
	{
	}
};

struct GpuRacerWeightState
{
	float promotionScore;
	float lastAdamGain;
	float lastPrecondGain;
	float lastCostPenalty;
	float lastPromotionMargin;
	float lastPredictiveTrust;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	bool promoted;
	unsigned long long promotedSteps;
	unsigned long long step;
	bool initialized;
	GpuRacerWeightState()
	    : promotionScore(0.0f), lastAdamGain(0.0f), lastPrecondGain(0.0f),
	      lastCostPenalty(0.0f), lastPromotionMargin(0.0f), lastPredictiveTrust(0.0f),
	      lastRowAnisotropy(1.0f), lastColAnisotropy(1.0f),
	      promoted(false), promotedSteps(0ULL), step(0ULL), initialized(false) {}
};

struct GpuMatraWeightState
{
	float rowMean;
	float colMean;
	float lastPredictiveTrust;
	float lastGeometryTrust;
	float lastOrthTrust;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	float lastAspect;
	float lastSignalScale;
	float lastOrthError;
	bool lastEligible;
	unsigned long long step;
	bool initialized;
	GpuMatraWeightState()
	    : rowMean(1.0e-12f), colMean(1.0e-12f),
	      lastPredictiveTrust(0.0f), lastGeometryTrust(0.0f), lastOrthTrust(0.0f),
	      lastRowAnisotropy(1.0f), lastColAnisotropy(1.0f), lastAspect(1.0f),
	      lastSignalScale(0.0f), lastOrthError(0.0f), lastEligible(false),
	      step(0ULL), initialized(false) {}
};

struct GpuMuonWeightState
{
	float lastPredictiveTrust;
	float lastAspect;
	float lastSignalScale;
	float lastOrthError;
	bool lastEligible;
	unsigned long long step;
	bool initialized;
	GpuMuonWeightState()
	    : lastPredictiveTrust(0.0f), lastAspect(1.0f), lastSignalScale(0.0f),
	      lastOrthError(0.0f), lastEligible(false), step(0ULL), initialized(false) {}
};

inline bool atlas_gpu_init(GpuAtlasWeightState&, unsigned int, unsigned int,
                           unsigned int, float, glades::rng::Engine&) { return false; }
inline bool atlas_gpu_step(GpuAtlasWeightState&, float*, float*,
                           unsigned int, unsigned int,
                           float, float, float, float, float,
                           const glades::ATLASConfig&,
                           shmea::GLogger* = 0,
                           const char* = 0) { return false; }
inline AtlasGpuDiag atlas_gpu_get_diag(const GpuAtlasWeightState&)
{ return AtlasGpuDiag(); }
inline bool atlas_gpu_update(GpuAtlasWeightState&, float*, float*,
                             unsigned int, unsigned int,
                             float, float, float, float, float,
                             const glades::ATLASConfig&,
                             glades::rng::Engine&,
                             shmea::GLogger* = 0,
                             const char* = 0) { return false; }
inline bool atlas_gpu_residual_update(GpuAtlasWeightState&, float*, float*,
                                      unsigned int, unsigned int,
                                      float, float, float,
                                      const glades::ATLASConfig&,
                                      glades::rng::Engine&,
                                      shmea::GLogger* = 0,
                                      const char* = 0) { return false; }
inline bool pact_gpu_update_lite(GpuPactWeightState&, float*, float*, float*, float*,
                                 unsigned int, unsigned int,
                                 float, float, float,
                                 float, float, float, float,
                                 const glades::ATLASConfig&,
                                 shmea::GLogger* = 0,
                                 const char* = 0) { return false; }
inline bool bimap_gpu_update(GpuBiMAPWeightState&, float*, float*, float*, float*,
                             unsigned int, unsigned int,
                             float, float, float,
                             float, float, float,
                             const glades::ATLASConfig&,
                             shmea::GLogger* = 0,
                             const char* = 0) { return false; }
inline bool bimap_gpu_update_lite(GpuBiMAPWeightState&, float*, float*, float*, float*,
                                  unsigned int, unsigned int,
                                  float, float, float,
                                  float, float, float,
                                  const glades::ATLASConfig&,
                                  shmea::GLogger* = 0,
                                  const char* = 0) { return false; }
inline bool echo_gpu_observe(GpuEchoWeightState&, const float*, const float*,
                             unsigned int, unsigned int, unsigned int,
                             const glades::ATLASConfig&) { return false; }
inline bool echo_gpu_ensure_state(GpuEchoWeightState&, unsigned int, unsigned int) { return false; }
inline bool echo_gpu_observe_batch(GpuEchoObserveEntry*, int,
                                   const GpuEchoObserveEntry*, int, int,
                                   unsigned long long,
                                   const glades::ATLASConfig&) { return false; }
inline bool echo_gpu_launch_observe_batch(const GpuEchoObserveEntry*, int, int,
                                          unsigned long long,
                                          const glades::ATLASConfig&) { return false; }
inline bool echo_gpu_prepare_metrics(GpuEchoWeightState&, unsigned int, unsigned int,
                                     unsigned long long, float,
                                     const glades::ATLASConfig&) { return false; }
inline bool echo_gpu_prepare_metrics_batch(float**, float**, float**, float**, float**, float**, float**,
                                           const int*, const int*, int,
                                           unsigned long long, float,
                                           const glades::ATLASConfig&) { return false; }
inline bool echo_gpu_post_update(GpuEchoWeightState&,
                                 const glades::ATLASConfig&,
                                 shmea::GLogger* = 0,
                                 const char* = 0) { return false; }
inline bool echo_gpu_update(GpuEchoWeightState&, float*, float*, float*, float*,
                            unsigned int, unsigned int,
                            float, float, unsigned long long,
                            float, float, float,
                            const glades::ATLASConfig&,
                            shmea::GLogger* = 0,
                            const char* = 0) { return false; }
inline bool racer_gpu_update_lite(GpuRacerWeightState&, float*, float*, float*, float*,
                                  unsigned int, unsigned int,
                                  float, float, float,
                                  float, float, float, float,
                                  const glades::ATLASConfig&,
                                  shmea::GLogger* = 0,
                                  const char* = 0) { return false; }
inline bool matra_gpu_update(GpuMatraWeightState&, float*, float*, float*, float*,
                             unsigned int, unsigned int,
                             float, float, float,
                             float, float, float,
                             const glades::ATLASConfig&,
                             shmea::GLogger* = 0,
                             const char* = 0) { return false; }
inline bool matra_gpu_update_small_batches(const GpuMatraBatchItem*, int,
                                           const int*, const int*, int,
                                           GpuMatraBatchItem*, float*, float**, float**, int*, int,
                                           float, float,
                                           float, float,
                                           float,
                                           const glades::ATLASConfig&,
                                           shmea::GLogger* = 0) { return false; }
inline bool muon_gpu_update_lite(GpuMuonWeightState&, float*, float*, float*, float*,
                                 unsigned int, unsigned int,
                                 float, float, float,
                                 float,
                                 const glades::ATLASConfig&,
                                 shmea::GLogger* = 0,
                                 const char* = 0) { return false; }
inline bool muon_gpu_update_lite_small_batch(const GpuMuonBatchItem*, int,
                                             GpuMuonBatchItem*, float**, float**, int*, int,
                                             float, float,
                                             float,
                                             const glades::ATLASConfig&,
                                             shmea::GLogger* = 0) { return false; }
inline bool muon_gpu_update_lite_small_batches(const GpuMuonBatchItem*, int,
                                               const int*, const int*, int,
                                               GpuMuonBatchItem*, float**, float**, int*, int,
                                               float, float,
                                               float,
                                               const glades::ATLASConfig&,
                                               shmea::GLogger* = 0) { return false; }
inline bool atlas_gpu_guard(float*, size_t) { return false; }

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
