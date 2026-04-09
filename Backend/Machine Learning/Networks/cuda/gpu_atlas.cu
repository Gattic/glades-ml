// GPU-accelerated ATLAS optimizer (BRSP variant).
//
// Mirrors atlas_optimizer.cpp but operates entirely on device memory.
// Uses cuBLAS SGEMM for matrix products and custom CUDA kernels for
// elementwise / reduction operations.
//
// Requirements: CUDA 11+, SM 6.0+, cuBLAS.

#include "gpu_atlas.h"

#ifdef GLADES_HAVE_CUDA

#include "gpu_blas.h"
#include "gpu_device.h"
#include "Backend/Database/GLogger.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <vector>

namespace glades {
namespace gpu {

namespace {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define ATLAS_CUDA_CHECK(call)                                                \
	do {                                                                      \
		cudaError_t err_ = (call);                                            \
		if (err_ != cudaSuccess) {                                            \
			fprintf(stderr, "[atlas-gpu] %s:%d  %s  -> %s\n",                \
			        __FILE__, __LINE__, #call, cudaGetErrorString(err_));     \
			return false;                                                     \
		}                                                                     \
	} while (0)

static constexpr int kBlock = 256;

// Note: GPU memory tracking is done per-init via stderr logging.
// No global state is maintained — callers can aggregate if needed.

static inline float atlas_bootstrap_or_ema(float prev,
                                           float sample,
                                           float beta,
                                           unsigned long long step)
{
	if (step <= 1ULL)
		return sample;
	return beta * prev + (1.0f - beta) * sample;
}

static unsigned int atlas_requested_complement_rank(unsigned int enabledRank,
                                                    unsigned int subDim,
                                                    unsigned int activeRank);

static bool atlas_tag_contains(const char* tag, const char* needle)
{
	return tag && needle && std::strstr(tag, needle) != 0;
}

static bool atlas_hidden_fc_eligible(const char* tag,
                                     unsigned int m,
                                     unsigned int n)
{
	if (!tag || !tag[0])
		return true;
	if (m <= 16u || n <= 16u)
		return false;
	return atlas_tag_contains(tag, "cnn.fc")
	    || atlas_tag_contains(tag, "dff.W");
}

static unsigned int atlas_enabled_complement_rank(unsigned int configuredRank,
                                                  const char* tag,
                                                  unsigned int m,
                                                  unsigned int n,
                                                  unsigned int activeRank)
{
	if (configuredRank == 0u)
		return 0u;
	if (!atlas_hidden_fc_eligible(tag, m, n))
		return 0u;
	return atlas_requested_complement_rank(configuredRank, m, activeRank);
}

static double sum_leading_spectrum(const std::vector<float>& eigVal,
                                   unsigned int count)
{
	double sum = 0.0;
	const unsigned int limit =
	    (count < static_cast<unsigned int>(eigVal.size()))
	        ? count
	        : static_cast<unsigned int>(eigVal.size());
	for (unsigned int i = 0; i < limit; ++i)
	{
		double v = static_cast<double>(eigVal[i]);
		if (!std::isfinite(v) || v < 0.0)
			v = 0.0;
		sum += v;
	}
	return sum;
}

static double complement_tail_mean(double closedTrace,
                                   double activeTrace,
                                   double selectedTrace,
                                   unsigned int subDim,
                                   unsigned int activeRank,
                                   unsigned int selectedRank,
                                   float eps)
{
	const unsigned int tailDim =
	    (subDim > activeRank + selectedRank)
	        ? (subDim - activeRank - selectedRank)
	        : 0u;
	if (tailDim == 0u)
		return std::max<double>(static_cast<double>(eps), 0.0);

	double tailMean = (closedTrace - activeTrace - selectedTrace)
	                / static_cast<double>(tailDim);
	if (!std::isfinite(tailMean) || tailMean < static_cast<double>(eps))
		tailMean = static_cast<double>(eps);
	return tailMean;
}

static double complement_kelly_fraction(double scoutEig,
                                        double tailMean)
{
	if (!std::isfinite(scoutEig) || scoutEig <= 0.0)
		return 0.0;
	if (!std::isfinite(tailMean) || tailMean <= 0.0)
		tailMean = 0.0;
	const double edge = scoutEig - tailMean;
	if (!(edge > 0.0))
		return 0.0;
	double fraction = edge / scoutEig;
	if (!std::isfinite(fraction) || fraction < 0.0)
		return 0.0;
	if (fraction > 1.0)
		fraction = 1.0;
	return fraction;
}

static double atlas_clamp_unit(double x)
{
	if (!std::isfinite(x) || x <= 0.0)
		return 0.0;
	if (x >= 1.0)
		return 1.0;
	return x;
}

struct ResidualScoutQuality
{
	double rawTopEig;
	double projectedEig;
	double birthScout;
	double birthKelly;
	double alignment;
	double contamination;
	double uncertainty;
	double quality;

	ResidualScoutQuality()
	    : rawTopEig(0.0), projectedEig(0.0), birthScout(0.0),
	      birthKelly(0.0), alignment(1.0), contamination(0.0),
	      uncertainty(0.0), quality(1.0)
	{
	}
};

static void reset_complement_trial(GpuAtlasWeightState& state)
{
	state.trialComplementRank = 0u;
	state.trialComplementWins = 0u;
	state.trialComplementMean = 0.0f;
	state.trialComplementVar = 0.0f;
}

static double complement_direction_overlap_sq(const std::vector<float>& eigVec,
                                              const std::vector<float>* scoutEigVec,
                                              unsigned int dim,
                                              unsigned int refMode,
                                              unsigned int scoutMode)
{
	if (!scoutEigVec || refMode >= dim || scoutMode >= dim
	    || eigVec.size() < static_cast<size_t>(dim) * dim
	    || scoutEigVec->size() < static_cast<size_t>(dim) * dim)
		return 0.0;

	double dot = 0.0;
	double refNorm = 0.0;
	double scoutNorm = 0.0;
	for (unsigned int k = 0; k < dim; ++k)
	{
		const double a =
		    static_cast<double>(eigVec[static_cast<size_t>(k) * dim + refMode]);
		const double b =
		    static_cast<double>((*scoutEigVec)[static_cast<size_t>(k) * dim + scoutMode]);
		dot += a * b;
		refNorm += a * a;
		scoutNorm += b * b;
	}
	if (!(refNorm > 1e-18) || !(scoutNorm > 1e-18))
		return 0.0;
	return atlas_clamp_unit((dot * dot) / (refNorm * scoutNorm));
}

static unsigned int complement_informative_scout_count(const std::vector<float>* scoutEigVal,
                                                       unsigned int dim,
                                                       unsigned int scoutCount,
                                                       float eps)
{
	const unsigned int limit = (scoutCount < dim) ? scoutCount : dim;
	if (!scoutEigVal || scoutEigVal->empty())
		return limit;

	double topEig = 0.0;
	for (unsigned int scoutMode = 0; scoutMode < limit && scoutMode < scoutEigVal->size(); ++scoutMode)
	{
		double eig = static_cast<double>((*scoutEigVal)[scoutMode]);
		if (std::isfinite(eig) && eig > topEig)
			topEig = eig;
	}
	if (!(topEig > static_cast<double>(eps)))
		return 0u;

	const double minEig = std::max<double>(static_cast<double>(eps), topEig * 1e-3);
	unsigned int informative = 0u;
	for (unsigned int scoutMode = 0; scoutMode < limit && scoutMode < scoutEigVal->size(); ++scoutMode)
	{
		const double eig = static_cast<double>((*scoutEigVal)[scoutMode]);
		if (std::isfinite(eig) && eig >= minEig)
			++informative;
	}
	return informative;
}

static double complement_direction_subspace_alignment(const std::vector<float>& eigVec,
                                                      const std::vector<float>* scoutEigVal,
                                                      const std::vector<float>* scoutEigVec,
                                                      unsigned int dim,
                                                      unsigned int refMode,
                                                      unsigned int scoutCount,
                                                      float eps)
{
	if (refMode >= dim)
		return 0.0;
	if (!scoutEigVec)
		return 1.0;
	const unsigned int limit =
	    complement_informative_scout_count(scoutEigVal, dim, scoutCount, eps);
	if (limit == 0u)
		return 0.0;
	double overlap = 0.0;
	double totalWeight = 0.0;
	for (unsigned int scoutMode = 0; scoutMode < limit; ++scoutMode)
	{
		double weight = 1.0;
		if (scoutEigVal && scoutMode < scoutEigVal->size())
		{
			weight = static_cast<double>((*scoutEigVal)[scoutMode]);
			if (!std::isfinite(weight) || weight <= static_cast<double>(eps))
				continue;
		}
		overlap += weight * complement_direction_overlap_sq(eigVec, scoutEigVec, dim,
		                                                    refMode, scoutMode);
		totalWeight += weight;
	}
	if (!(totalWeight > 1e-18))
		return 0.0;
	return atlas_clamp_unit(overlap / totalWeight);
}

static double complement_projected_scout_eigenvalue(const std::vector<float>* scoutEigVal,
                                                    const std::vector<float>& eigVec,
                                                    const std::vector<float>* scoutEigVec,
                                                    unsigned int dim,
                                                    unsigned int refMode,
                                                    unsigned int scoutCount)
{
	if (!scoutEigVal || !scoutEigVec || refMode >= dim)
		return 0.0;
	const unsigned int limit =
	    std::min<unsigned int>(scoutCount,
	                           static_cast<unsigned int>(scoutEigVal->size()));
	double projectedEig = 0.0;
	for (unsigned int scoutMode = 0; scoutMode < limit; ++scoutMode)
	{
		double eig = static_cast<double>((*scoutEigVal)[scoutMode]);
		if (!std::isfinite(eig) || eig < 0.0)
			eig = 0.0;
		projectedEig += complement_direction_overlap_sq(eigVec, scoutEigVec, dim,
		                                                refMode, scoutMode) * eig;
	}
	return projectedEig;
}

static double complement_scout_contamination(const std::vector<float>* scoutEigVal,
                                             const std::vector<float>& eigVec,
                                             const std::vector<float>* scoutEigVec,
                                             unsigned int dim,
                                             unsigned int retainedCount,
                                             unsigned int scoutCount)
{
	if (!scoutEigVec || retainedCount == 0u)
		return 0.0;
	const unsigned int scoutLimit =
	    std::min<unsigned int>(scoutCount, dim);
	const unsigned int retainedLimit =
	    std::min<unsigned int>(retainedCount, dim);
	double contam = 0.0;
	double totalWeight = 0.0;
	for (unsigned int scoutMode = 0; scoutMode < scoutLimit; ++scoutMode)
	{
		double weight = 1.0;
		if (scoutEigVal && scoutMode < scoutEigVal->size())
		{
			weight = static_cast<double>((*scoutEigVal)[scoutMode]);
			if (!std::isfinite(weight) || weight <= 0.0)
				weight = 0.0;
		}
		double overlap = 0.0;
		for (unsigned int refMode = 0; refMode < retainedLimit; ++refMode)
			overlap += complement_direction_overlap_sq(eigVec, scoutEigVec, dim,
			                                           refMode, scoutMode);
		contam += weight * atlas_clamp_unit(overlap);
		totalWeight += weight;
	}
	if (!(totalWeight > 1e-18))
		return 0.0;
	return atlas_clamp_unit(contam / totalWeight);
}

static ResidualScoutQuality evaluate_residual_scout(const std::vector<float>& eigVal,
                                                    const std::vector<float>& eigVec,
                                                    const std::vector<float>* scoutEigVal,
                                                    const std::vector<float>* scoutEigVec,
                                                    unsigned int nextMode,
                                                    unsigned int retainedCount,
                                                    double tailMean,
                                                    float eps)
{
	ResidualScoutQuality quality;
	if (nextMode >= eigVal.size())
		return quality;

	double nextEmaEig = static_cast<double>(eigVal[nextMode]);
	if (!std::isfinite(nextEmaEig) || nextEmaEig < 0.0)
		nextEmaEig = 0.0;
	quality.rawTopEig = nextEmaEig;
	quality.projectedEig = nextEmaEig;
	quality.birthScout = nextEmaEig;

	if (scoutEigVal && !scoutEigVal->empty() && scoutEigVec)
	{
		const unsigned int dim = static_cast<unsigned int>(eigVal.size());
		const unsigned int scoutCount =
		    complement_informative_scout_count(scoutEigVal, dim, 2u, eps);
		double rawTopEig = static_cast<double>((*scoutEigVal)[0u]);
		if (!std::isfinite(rawTopEig) || rawTopEig < 0.0)
			rawTopEig = 0.0;
		quality.rawTopEig = rawTopEig;
		quality.alignment =
		    complement_direction_subspace_alignment(eigVec, scoutEigVal, scoutEigVec,
		                                            dim, nextMode, scoutCount, eps);
		quality.projectedEig =
		    complement_projected_scout_eigenvalue(scoutEigVal, eigVec, scoutEigVec,
		                                          dim, nextMode, scoutCount);
		quality.contamination =
		    complement_scout_contamination(scoutEigVal, eigVec, scoutEigVec,
		                                   dim, retainedCount, scoutCount);
		const double emaSupport = quality.alignment * nextEmaEig;
		quality.birthScout = std::max(quality.projectedEig, emaSupport);
		const double uncertaintyScale =
		    std::max<double>(std::max<double>(quality.birthScout, tailMean),
		                     static_cast<double>(eps));
		quality.uncertainty =
		    atlas_clamp_unit(std::fabs(quality.projectedEig - emaSupport)
		                     / uncertaintyScale);
	}

	quality.birthKelly = complement_kelly_fraction(quality.birthScout, tailMean);
	quality.quality = quality.alignment * (1.0 - quality.contamination);
	if (!std::isfinite(quality.quality) || quality.quality < 0.0)
		quality.quality = 0.0;
	return quality;
}

static unsigned int choose_active_complement_rank(GpuAtlasWeightState& state,
                                                  const std::vector<float>& eigVal,
                                                  const std::vector<float>& eigVec,
                                                  const std::vector<float>* scoutEigVal,
                                                  const std::vector<float>* scoutEigVec,
                                                  unsigned int informativeRank,
                                                  unsigned int prevRank,
                                                  double activeTrace,
                                                  float totalTrace,
                                                  unsigned int subDim,
                                                  unsigned int activeRank,
                                                  float eps,
                                                  double* birthKellyOut = 0,
                                                  double* birthScoutOut = 0,
                                                  double* scoutProjectedOut = 0,
                                                  double* trialKellyOut = 0,
                                                  double* trialAlignmentOut = 0,
                                                  double* trialContaminationOut = 0,
                                                  double* trialReturnOut = 0,
                                                  double* trialScoreOut = 0)
{
	if (informativeRank == 0u)
	{
		reset_complement_trial(state);
		return 0u;
	}
	if (prevRank > informativeRank)
		prevRank = informativeRank;
	if (state.trialComplementRank > informativeRank
	    || state.trialComplementRank <= prevRank)
		reset_complement_trial(state);

	const double fullTrace = sum_leading_spectrum(eigVal, informativeRank);
	const double closedTrace = std::max<double>(static_cast<double>(totalTrace),
	                                            activeTrace + fullTrace);
	const double deathRatio = 1.10;
	const double birthKellyThreshold = 0.10;
	const double birthMinShare = 0.005;
	const double deathMinShare = 0.03;
	const double trialBeta = 0.8;
	const double trialUncertaintyWeight = 0.25;
	const double trialContaminationWeight = 0.10;
	const double trialKellyThreshold = 0.10;
	const double trialScoreThreshold = 0.15;
	const double trialDropScore = 0.02;
	const double trialAlignmentFloor = 0.25;
	const double trialRiskFloor = 0.05;
	const unsigned int trialWinsRequired = 2u;
	if (birthKellyOut)
		*birthKellyOut = 0.0;
	if (birthScoutOut)
		*birthScoutOut = 0.0;
	if (scoutProjectedOut)
		*scoutProjectedOut = 0.0;
	if (trialKellyOut)
		*trialKellyOut = 0.0;
	if (trialAlignmentOut)
		*trialAlignmentOut = 0.0;
	if (trialContaminationOut)
		*trialContaminationOut = 0.0;
	if (trialReturnOut)
		*trialReturnOut = 0.0;
	if (trialScoreOut)
		*trialScoreOut = 0.0;

	if (prevRank < informativeRank)
	{
		const unsigned int candidateRank = prevRank + 1u;
		const double selectedTrace = sum_leading_spectrum(eigVal, prevRank);
		const double tailMean =
		    complement_tail_mean(closedTrace, activeTrace, selectedTrace,
		                         subDim, activeRank, prevRank, eps);
		const ResidualScoutQuality scoutQuality =
		    evaluate_residual_scout(eigVal, eigVec, scoutEigVal, scoutEigVec,
		                            prevRank, prevRank, tailMean, eps);
		const double birthScout = scoutQuality.birthScout;
		const double birthKelly = scoutQuality.birthKelly;
		const double alignment = scoutQuality.alignment;
		const double contamination = scoutQuality.contamination;
		const double uncertainty = scoutQuality.uncertainty;
		const double edgeShare = (closedTrace > static_cast<double>(eps))
		    ? std::max<double>(0.0, birthScout - tailMean) / closedTrace
		    : 0.0;
		const double sampleReturn =
		    edgeShare
		    - trialUncertaintyWeight * uncertainty
		    - trialContaminationWeight * contamination;
		if (birthKellyOut)
			*birthKellyOut = birthKelly;
		if (birthScoutOut)
			*birthScoutOut = scoutQuality.rawTopEig;
		if (scoutProjectedOut)
			*scoutProjectedOut = scoutQuality.projectedEig;
		if (trialAlignmentOut)
			*trialAlignmentOut = alignment;
		if (trialContaminationOut)
			*trialContaminationOut = contamination;
		if (trialReturnOut)
			*trialReturnOut = sampleReturn;
		if (birthKelly >= birthKellyThreshold
		    && birthScout > birthMinShare * closedTrace)
		{
			if (state.trialComplementRank != candidateRank)
				reset_complement_trial(state);
			state.trialComplementRank = candidateRank;
			const double prevMean = static_cast<double>(state.trialComplementMean);
			const double prevVar = static_cast<double>(state.trialComplementVar);
			const double nextMean =
			    (state.trialComplementWins == 0u && prevMean == 0.0 && prevVar == 0.0)
			        ? sampleReturn
			        : (trialBeta * prevMean + (1.0 - trialBeta) * sampleReturn);
			const double innovation = sampleReturn - prevMean;
			const double nextVar =
			    (state.trialComplementWins == 0u && prevMean == 0.0 && prevVar == 0.0)
			        ? (innovation * innovation)
			        : (trialBeta * prevVar + (1.0 - trialBeta) * innovation * innovation);
			state.trialComplementMean = static_cast<float>(nextMean);
			state.trialComplementVar = static_cast<float>(nextVar);
			const double riskPenalty =
			    std::max<double>(trialRiskFloor,
			                     nextVar + uncertainty + (1.0 - alignment));
			const double trialKelly =
			    atlas_clamp_unit((nextMean > 0.0) ? (nextMean / riskPenalty) : 0.0);
			const double trialScore =
			    trialKelly * scoutQuality.quality * birthKelly;
			if (trialKellyOut)
				*trialKellyOut = trialKelly;
			if (trialScoreOut)
				*trialScoreOut = trialScore;
			if (nextMean > 0.0
			    && trialKelly >= trialKellyThreshold
			    && trialScore >= trialScoreThreshold
			    && alignment >= trialAlignmentFloor)
			{
				++state.trialComplementWins;
				if (state.trialComplementWins >= trialWinsRequired)
				{
					reset_complement_trial(state);
					return candidateRank;
				}
			}
			else if (nextMean <= 0.0
			         || trialScore <= trialDropScore
			         || alignment < 0.05)
			{
				reset_complement_trial(state);
			}
			else
			{
				state.trialComplementWins = 0u;
			}
		}
		else if (state.trialComplementRank == candidateRank)
		{
			reset_complement_trial(state);
		}
	}
	else
	{
		reset_complement_trial(state);
	}

	if (prevRank > 0u)
	{
		const double selectedTrace = sum_leading_spectrum(eigVal, prevRank - 1u);
		const double tailMean =
		    complement_tail_mean(closedTrace, activeTrace, selectedTrace,
		                         subDim, activeRank, prevRank - 1u, eps);
		double weakestEig = static_cast<double>(eigVal[prevRank - 1u]);
		if (!std::isfinite(weakestEig) || weakestEig < 0.0)
			weakestEig = 0.0;
		if (weakestEig <= deathRatio * tailMean
		    || weakestEig <= deathMinShare * closedTrace)
		{
			reset_complement_trial(state);
			return prevRank - 1u;
		}
	}

	return prevRank;
}

static unsigned int atlas_storage_complement_rank(unsigned int enabledRank)
{
	return (enabledRank > 0u) ? enabledRank : 1u;
}

static unsigned int atlas_requested_complement_rank(unsigned int enabledRank,
                                                    unsigned int subDim,
                                                    unsigned int activeRank)
{
	if (enabledRank == 0u)
		return 0u;
	if (subDim <= activeRank)
		return 0u;
	const unsigned int available = subDim - activeRank;
	return (enabledRank < available) ? enabledRank : available;
}

static float compute_complement_sigma2(float totalTrace,
                                       const std::vector<float>& fisherDiag,
                                       unsigned int activeRank,
                                       double activeSectorTrace,
                                       unsigned int effectiveComplementRank,
                                       unsigned int subDim,
                                       float eps,
                                       double* activeTraceOut = 0,
                                       double* sectorTraceOut = 0,
                                       double* closureGapOut = 0)
{
	double activeTrace = 0.0;
	for (unsigned int c = 0; c < activeRank; ++c)
		activeTrace += static_cast<double>(fisherDiag[c]);
	const double sectorTrace =
	    (effectiveComplementRank > 0u) ? activeSectorTrace : 0.0;
	const double modeledTrace = activeTrace + sectorTrace;
	const double closureGap = static_cast<double>(totalTrace) - modeledTrace;
	const double closedTrace = (closureGap >= 0.0)
	    ? static_cast<double>(totalTrace)
	    : modeledTrace;
	const unsigned int complementDim =
	    (subDim > activeRank + effectiveComplementRank)
	        ? (subDim - activeRank - effectiveComplementRank)
	        : 0u;
	double sigma2 = 0.0;
	if (complementDim > 0u)
		sigma2 = (closedTrace - modeledTrace) / static_cast<double>(complementDim);
	else if (activeRank + effectiveComplementRank > 0u)
		sigma2 = closedTrace / static_cast<double>(activeRank + effectiveComplementRank);
	if (activeTraceOut) *activeTraceOut = activeTrace;
	if (sectorTraceOut) *sectorTraceOut = sectorTrace;
	if (closureGapOut) *closureGapOut = closureGap;
	if (!std::isfinite(sigma2) || sigma2 < static_cast<double>(eps))
		sigma2 = static_cast<double>(eps);
	return static_cast<float>(sigma2);
}

static void project_out_active_basis(std::vector<float>& v,
                                     const std::vector<float>& U,
                                     unsigned int fullRank,
                                     unsigned int activeRank,
                                     unsigned int subDim)
{
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		double dot = 0.0;
		for (unsigned int i = 0; i < subDim; ++i)
			dot += static_cast<double>(v[i]) * static_cast<double>(U[static_cast<size_t>(i) * fullRank + c]);
		const float dotf = static_cast<float>(dot);
		for (unsigned int i = 0; i < subDim; ++i)
			v[i] -= dotf * U[static_cast<size_t>(i) * fullRank + c];
	}
}

static void project_out_basis_block(std::vector<float>& v,
                                    const std::vector<float>& basis,
                                    unsigned int stride,
                                    unsigned int rank,
                                    unsigned int subDim)
{
	for (unsigned int c = 0; c < rank; ++c)
	{
		double dot = 0.0;
		for (unsigned int i = 0; i < subDim; ++i)
			dot += static_cast<double>(v[i]) * static_cast<double>(basis[static_cast<size_t>(i) * stride + c]);
		const float dotf = static_cast<float>(dot);
		for (unsigned int i = 0; i < subDim; ++i)
			v[i] -= dotf * basis[static_cast<size_t>(i) * stride + c];
	}
}

static bool normalize_vector(std::vector<float>& v)
{
	double normSq = 0.0;
	for (size_t i = 0; i < v.size(); ++i)
	{
		const double x = static_cast<double>(v[i]);
		normSq += x * x;
	}
	if (normSq <= 1e-12)
		return false;
	const float invNorm = static_cast<float>(1.0 / std::sqrt(normSq));
	for (size_t i = 0; i < v.size(); ++i)
		v[i] *= invNorm;
	return true;
}

static bool build_coordinate_complement_seed(std::vector<float>& v,
                                             const std::vector<float>& U,
                                             unsigned int fullRank,
                                             unsigned int activeRank,
                                             const std::vector<float>& extraBasis,
                                             unsigned int extraStride,
                                             unsigned int extraRank,
                                             unsigned int subDim)
{
	for (unsigned int basisRow = 0; basisRow < subDim; ++basisRow)
	{
		std::fill(v.begin(), v.end(), 0.0f);
		v[basisRow] = 1.0f;
		project_out_active_basis(v, U, fullRank, activeRank, subDim);
		if (!extraBasis.empty() && extraRank > 0u)
			project_out_basis_block(v, extraBasis, extraStride, extraRank, subDim);
		if (normalize_vector(v))
			return true;
	}
	std::fill(v.begin(), v.end(), 0.0f);
	return false;
}

static unsigned int orthonormalize_complement_block(std::vector<float>& V,
                                                    unsigned int stride,
                                                    const std::vector<float>& U,
                                                    unsigned int fullRank,
                                                    unsigned int activeRank,
                                                    unsigned int subDim,
                                                    unsigned int targetRank)
{
	if (targetRank == 0u)
	{
		std::fill(V.begin(), V.end(), 0.0f);
		return 0u;
	}

	unsigned int informative = 0u;
	for (unsigned int c = 0; c < targetRank; ++c)
	{
		std::vector<float> col(subDim, 0.0f);
		for (unsigned int i = 0; i < subDim; ++i)
			col[i] = V[static_cast<size_t>(i) * stride + c];
		project_out_active_basis(col, U, fullRank, activeRank, subDim);
		if (c > 0u)
			project_out_basis_block(col, V, stride, c, subDim);
		if (!normalize_vector(col)
		    && !build_coordinate_complement_seed(col, U, fullRank, activeRank, V, stride, c, subDim))
		{
			for (unsigned int i = 0; i < subDim; ++i)
				V[static_cast<size_t>(i) * stride + c] = 0.0f;
			continue;
		}
		for (unsigned int i = 0; i < subDim; ++i)
			V[static_cast<size_t>(i) * stride + c] = col[i];
		informative = c + 1u;
	}
	for (unsigned int c = targetRank; c < stride; ++c)
		for (unsigned int i = 0; i < subDim; ++i)
			V[static_cast<size_t>(i) * stride + c] = 0.0f;
	return informative;
}

static unsigned int effective_complement_rank(const std::vector<float>& V,
                                              unsigned int stride,
                                              unsigned int enabledRank,
                                              unsigned int subDim,
                                              unsigned int activeRank)
{
	const unsigned int requested =
	    atlas_requested_complement_rank(enabledRank, subDim, activeRank);
	const unsigned int inspect = (requested < stride) ? requested : stride;
	unsigned int informative = 0u;
	for (unsigned int c = 0; c < inspect; ++c)
	{
		double normSq = 0.0;
		for (unsigned int i = 0; i < subDim; ++i)
		{
			const double v = static_cast<double>(V[static_cast<size_t>(i) * stride + c]);
			normSq += v * v;
		}
		if (normSq <= 1e-12)
			break;
		informative = c + 1u;
	}
	return informative;
}

static float trace_complement_block(const std::vector<float>& block,
                                    unsigned int dim)
{
	double trace = 0.0;
	for (unsigned int i = 0; i < dim; ++i)
		trace += static_cast<double>(block[static_cast<size_t>(i) * dim + i]);
	return static_cast<float>(trace);
}

static void symmetrize_block(std::vector<float>& block, unsigned int dim)
{
	for (unsigned int i = 0; i < dim; ++i)
	{
		for (unsigned int j = i + 1u; j < dim; ++j)
		{
			const float v = 0.5f * (block[static_cast<size_t>(i) * dim + j]
			                      + block[static_cast<size_t>(j) * dim + i]);
			block[static_cast<size_t>(i) * dim + j] = v;
			block[static_cast<size_t>(j) * dim + i] = v;
		}
	}
}

static void jacobi_eigendecompose(const std::vector<float>& symBlock,
                                  unsigned int dim,
                                  std::vector<float>& eigVec,
                                  std::vector<float>& eigVal)
{
	eigVec.assign(static_cast<size_t>(dim) * dim, 0.0f);
	eigVal.assign(static_cast<size_t>(dim), 0.0f);
	if (dim == 0u)
		return;
	std::vector<float> a(symBlock);
	for (unsigned int i = 0; i < dim; ++i)
		eigVec[static_cast<size_t>(i) * dim + i] = 1.0f;
	const unsigned int maxSweeps = 32u + dim * 8u;
	for (unsigned int sweep = 0; sweep < maxSweeps; ++sweep)
	{
		unsigned int p = 0u, q = 0u;
		float maxOff = 0.0f;
		for (unsigned int i = 0; i < dim; ++i)
		{
			for (unsigned int j = i + 1u; j < dim; ++j)
			{
				const float off = std::fabs(a[static_cast<size_t>(i) * dim + j]);
				if (off > maxOff)
				{
					maxOff = off;
					p = i;
					q = j;
				}
			}
		}
		if (maxOff < 1e-6f)
			break;
		const float app = a[static_cast<size_t>(p) * dim + p];
		const float aqq = a[static_cast<size_t>(q) * dim + q];
		const float apq = a[static_cast<size_t>(p) * dim + q];
		const float tau = (aqq - app) / (2.0f * apq);
		const float t = (tau >= 0.0f)
		    ? (1.0f / (tau + std::sqrt(1.0f + tau * tau)))
		    : (-1.0f / (-tau + std::sqrt(1.0f + tau * tau)));
		const float c = 1.0f / std::sqrt(1.0f + t * t);
		const float s = t * c;
		for (unsigned int k = 0; k < dim; ++k)
		{
			if (k == p || k == q)
				continue;
			const float aik = a[static_cast<size_t>(p) * dim + k];
			const float aqk = a[static_cast<size_t>(q) * dim + k];
			const float newAik = c * aik - s * aqk;
			const float newAqk = s * aik + c * aqk;
			a[static_cast<size_t>(p) * dim + k] = newAik;
			a[static_cast<size_t>(k) * dim + p] = newAik;
			a[static_cast<size_t>(q) * dim + k] = newAqk;
			a[static_cast<size_t>(k) * dim + q] = newAqk;
		}
		const float newApp = c * c * app - 2.0f * s * c * apq + s * s * aqq;
		const float newAqq = s * s * app + 2.0f * s * c * apq + c * c * aqq;
		a[static_cast<size_t>(p) * dim + p] = newApp;
		a[static_cast<size_t>(q) * dim + q] = newAqq;
		a[static_cast<size_t>(p) * dim + q] = 0.0f;
		a[static_cast<size_t>(q) * dim + p] = 0.0f;
		for (unsigned int k = 0; k < dim; ++k)
		{
			const float vip = eigVec[static_cast<size_t>(k) * dim + p];
			const float viq = eigVec[static_cast<size_t>(k) * dim + q];
			eigVec[static_cast<size_t>(k) * dim + p] = c * vip - s * viq;
			eigVec[static_cast<size_t>(k) * dim + q] = s * vip + c * viq;
		}
	}
	for (unsigned int i = 0; i < dim; ++i)
		eigVal[i] = a[static_cast<size_t>(i) * dim + i];
	for (unsigned int i = 0; i + 1u < dim; ++i)
	{
		unsigned int best = i;
		for (unsigned int j = i + 1u; j < dim; ++j)
			if (eigVal[j] > eigVal[best]) best = j;
		if (best == i)
			continue;
		std::swap(eigVal[i], eigVal[best]);
		for (unsigned int k = 0; k < dim; ++k)
			std::swap(eigVec[static_cast<size_t>(k) * dim + i],
			          eigVec[static_cast<size_t>(k) * dim + best]);
	}
}

static void transform_symmetric_block(std::vector<float>& dst,
                                      const std::vector<float>& overlap,
                                      const std::vector<float>& src,
                                      unsigned int dim)
{
	std::vector<float> tmp(static_cast<size_t>(dim) * dim, 0.0f);
	for (unsigned int i = 0; i < dim; ++i)
	{
		for (unsigned int j = 0; j < dim; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < dim; ++k)
				sum += static_cast<double>(overlap[static_cast<size_t>(i) * dim + k])
				     * static_cast<double>(src[static_cast<size_t>(k) * dim + j]);
			tmp[static_cast<size_t>(i) * dim + j] = static_cast<float>(sum);
		}
	}
	dst.assign(static_cast<size_t>(dim) * dim, 0.0f);
	for (unsigned int i = 0; i < dim; ++i)
	{
		for (unsigned int j = 0; j < dim; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < dim; ++k)
				sum += static_cast<double>(tmp[static_cast<size_t>(i) * dim + k])
				     * static_cast<double>(overlap[static_cast<size_t>(j) * dim + k]);
			dst[static_cast<size_t>(i) * dim + j] = static_cast<float>(sum);
		}
	}
}

// ---------------------------------------------------------------------------
// Warp / block reduction primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warpReduceSum(float val)
{
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}

__device__ float blockReduceSum(float val, float* smem)
{
	int lane = threadIdx.x & 31;
	int wid  = threadIdx.x >> 5;

	val = warpReduceSum(val);
	if (lane == 0) smem[wid] = val;
	__syncthreads();

	int numWarps = (blockDim.x + 31) / 32;
	val = (threadIdx.x < (unsigned)numWarps) ? smem[threadIdx.x] : 0.0f;
	if (wid == 0) val = warpReduceSum(val);
	return val;
}

__device__ __forceinline__ double warpReduceSumD(double val)
{
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}

__device__ double blockReduceSumD(double val, double* smem)
{
	int lane = threadIdx.x & 31;
	int wid  = threadIdx.x >> 5;

	val = warpReduceSumD(val);
	if (lane == 0) smem[wid] = val;
	__syncthreads();

	int numWarps = (blockDim.x + 31) / 32;
	val = (threadIdx.x < (unsigned)numWarps) ? smem[threadIdx.x] : 0.0;
	if (wid == 0) val = warpReduceSumD(val);
	return val;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// --- Gram-Schmidt orthonormalization ---
// Q is [m, r] row-major.  We process columns sequentially (j = 0..r-1).
// One block is launched; threads cooperate on dot products and normalization.
// Uses double accumulators for dot products and norms to match CPU path.
__global__ void atlas_gs_kernel(float* __restrict__ Q, int m, int r)
{
	extern __shared__ char smem_gs_bytes[];
	double* smem = reinterpret_cast<double*>(smem_gs_bytes);
	__shared__ double sDot;
	__shared__ float sInvNorm;

	for (int j = 0; j < r; ++j)
	{
		// Subtract projections onto previous columns (modified GS)
		for (int p = 0; p < j; ++p)
		{
			double localDot = 0.0;
				for (int k = threadIdx.x; k < m; k += blockDim.x)
					localDot += (double)Q[k * r + j] * (double)Q[k * r + p];
			double dot = blockReduceSumD(localDot, smem);
			if (threadIdx.x == 0)
				sDot = dot;
			__syncthreads();

			for (int k = threadIdx.x; k < m; k += blockDim.x)
				Q[k * r + j] -= (float)(sDot * (double)Q[k * r + p]);
			__syncthreads();
		}

		// Normalize column j
		double localNorm = 0.0;
		for (int k = threadIdx.x; k < m; k += blockDim.x)
		{
			double v = (double)Q[k * r + j];
			localNorm += v * v;
		}
		double norm = blockReduceSumD(localNorm, smem);
		__syncthreads();

		if (threadIdx.x == 0)
		{
			norm = sqrt(norm);
			sInvNorm = (norm > 1e-12) ? (float)(1.0 / norm) : 0.0f;
		}
		__syncthreads();

		float inv = sInvNorm;
		if (inv > 0.0f)
		{
			for (int k = threadIdx.x; k < m; k += blockDim.x)
				Q[k * r + j] *= inv;
		}
		else
		{
			for (int k = threadIdx.x; k < m; k += blockDim.x)
				Q[k * r + j] = (k == j && j < m) ? 1.0f : 0.0f;
		}
		__syncthreads();
	}
}

// --- Weight decay: W[i] -= lr * (wd1*sign(W[i]) + wd2*W[i]) ---
__global__ void atlas_wd_kernel(float* __restrict__ W, size_t mn,
                                float lr, float wd1, float wd2)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= mn) return;
	float w = W[idx];
	float decay = 0.0f;
	if (wd1 != 0.0f)
	{
		float s = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
		decay += wd1 * s;
	}
	if (wd2 != 0.0f)
		decay += wd2 * w;
	W[idx] = w - lr * decay;
}

// --- Baseline update: W[i] -= baseScaled * gW[i] ---
__global__ void atlas_baseline_kernel(float* __restrict__ W,
                                      const float* __restrict__ gW,
                                      size_t mn, float baseScaled)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= mn) return;
	W[idx] -= baseScaled * gW[idx];
}

// --- Fisher diagonal update: EMA of mean-squared projected gradient ---
// Left subspace: gz[r, outerDim], direction c = contiguous row c.
// One block per subspace component.
__global__ void atlas_fisher_kernel(const float* __restrict__ gz,
                                    float* __restrict__ fisherDiag,
                                    int r, int outerDim,
                                    float beta, float oneMinusBeta,
                                    float sampleScale,
                                    int bootstrap)
{
	int c = blockIdx.x;
	if (c >= r) return;

	extern __shared__ char smem_fisher_bytes[];
	double* smem = reinterpret_cast<double*>(smem_fisher_bytes);
	const float* row = gz + (size_t)c * outerDim;

	double localSum = 0.0;
	for (int j = threadIdx.x; j < outerDim; j += blockDim.x)
	{
		double v = (double)row[j];
		localSum += v * v;
	}
	double sumsq = blockReduceSumD(localSum, smem);

	if (threadIdx.x == 0)
	{
		float meansq = (float)(sumsq / (double)outerDim) * sampleScale;
		fisherDiag[c] = bootstrap ? meansq : (beta * fisherDiag[c] + oneMinusBeta * meansq);
	}
}

// Right subspace: gz[outerDim, r], direction c = column c (stride r).
// One block per subspace component.
// Each thread reads a contiguous chunk of the row (all r cols) but only
// accumulates column c — this gives coalesced reads when threads in a warp
// process consecutive rows.
__global__ void atlas_fisher_col_kernel(const float* __restrict__ gz,
                                         float* __restrict__ fisherDiag,
                                         int r, int outerDim,
                                         float beta, float oneMinusBeta,
                                         float sampleScale,
                                         int bootstrap)
{
	int c = blockIdx.x;
	if (c >= r) return;

	extern __shared__ char smem_fisher_bytes[];
	double* smem = reinterpret_cast<double*>(smem_fisher_bytes);

	// Threads stride by blockDim.x over rows — consecutive threads access
	// consecutive rows, so gz[i*r + c] and gz[(i+1)*r + c] differ by r floats.
	// With r=256, this is 1KB stride — not ideal for coalescing but acceptable
	// for the reduction pattern (compute-bound, not memory-bound).
	double localSum = 0.0;
	for (int i = threadIdx.x; i < outerDim; i += blockDim.x)
	{
		double v = (double)gz[(size_t)i * r + c];
		localSum += v * v;
	}
	double sumsq = blockReduceSumD(localSum, smem);

	if (threadIdx.x == 0)
	{
		float meansq = (float)(sumsq / (double)outerDim) * sampleScale;
		fisherDiag[c] = bootstrap ? meansq : (beta * fisherDiag[c] + oneMinusBeta * meansq);
	}
}

// --- Prepare correction: fused PNG prediction + per-component scaling ---
// Left subspace: gz[r, outerDim], direction c = row c.
// Grid: (ceil(outerDim/block), r)
__global__ void atlas_correction_kernel(const float* __restrict__ gz,
                                        const float* __restrict__ prevGz,
                                        const float* __restrict__ fisherDiag,
                                        float* __restrict__ out,
                                        int r, int outerDim,
                                        float onePlusMu, float negMu,
                                        float baselineRate, float lr, float eps,
                                        float kappaLr, float bcFactor)
{
	int c = blockIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (c >= r || j >= outerDim) return;

	const float effFisher = fisherDiag[c] * bcFactor;
	float fisherLR = fminf(lr / (effFisher + eps), kappaLr);
	float corrScale = baselineRate - fisherLR;
	size_t idx = (size_t)c * outerDim + j;
	float gPred = onePlusMu * gz[idx] + negMu * prevGz[idx];
	out[idx] = corrScale * gPred;
}

// Right subspace: gz[outerDim, r] — each row has r elements (all directions).
// Process entire rows: each thread handles one element of a row, giving
// coalesced access when adjacent threads process adjacent columns.
// Grid: (ceil(r/block), outerDim) — each block row = one outerDim row.
__global__ void atlas_correction_col_kernel(const float* __restrict__ gz,
                                             const float* __restrict__ prevGz,
                                             const float* __restrict__ fisherDiag,
                                             float* __restrict__ out,
                                             int r, int outerDim,
                                             float onePlusMu, float negMu,
                                             float baselineRate, float lr, float eps,
                                             float kappaLr, float bcFactor)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;  // subspace direction
	int i = blockIdx.y;                               // row in outerDim
	if (c >= r || i >= outerDim) return;

	const float effFisher = fisherDiag[c] * bcFactor;
	float fisherLR = fminf(lr / (effFisher + eps), kappaLr);
	float corrScale = baselineRate - fisherLR;
	size_t idx = (size_t)i * r + c;
	float gPred = onePlusMu * gz[idx] + negMu * prevGz[idx];
	out[idx] = corrScale * gPred;
}

// --- Mu adaptation norms: dual reduction ---
// d_out[0] = sum((gz[i]-prevGz[i])^2)  (errNormSq)
// d_out[1] = sum(gz[i]^2)              (gzNormSq)
// Phase 1: each block writes partial sums to d_partials.
// d_partials layout: [errPartials(nBlocks), gzPartials(nBlocks)]
// where errPartials[blockIdx.x] = partial errNormSq,
//       gzPartials[blockIdx.x]  = partial gzNormSq.
__global__ void atlas_mu_norms_kernel(const float* __restrict__ gz,
                                      const float* __restrict__ prevGz,
                                      size_t rn, int nBlocks,
                                      float* __restrict__ d_partials)
{
	extern __shared__ float smem[];
	float* sErr = smem;
	float* sGz  = smem + (blockDim.x / 32 + 1);

	float localErr = 0.0f;
	float localGz  = 0.0f;
	for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rn;
	     i += (size_t)gridDim.x * blockDim.x)
	{
		float g = gz[i];
		float e = g - prevGz[i];
		localErr += e * e;
		localGz  += g * g;
	}

	float errSum = blockReduceSum(localErr, sErr);
	__syncthreads();
	float gzSum = blockReduceSum(localGz, sGz);

	if (threadIdx.x == 0)
	{
		d_partials[blockIdx.x] = errSum;
		d_partials[nBlocks + blockIdx.x] = gzSum;
	}
}

// Phase 2: single-block kernel reduces two channels of partials to d_out[0..1].
__global__ void atlas_reduce_partials2_kernel(const float* __restrict__ d_partials,
                                               int nBlocks,
                                               float* __restrict__ d_out)
{
	extern __shared__ float smem[];
	float* s0 = smem;
	float* s1 = smem + (blockDim.x / 32 + 1);

	float v0 = 0.0f;
	float v1 = 0.0f;
	for (int i = threadIdx.x; i < nBlocks; i += blockDim.x)
	{
		v0 += d_partials[i];
		v1 += d_partials[nBlocks + i];
	}
	float sum0 = blockReduceSum(v0, s0);
	__syncthreads();
	float sum1 = blockReduceSum(v1, s1);
	if (threadIdx.x == 0)
	{
		d_out[0] = sum0;
		d_out[1] = sum1;
	}
}

// --- EMA blend: dst[i] = (1-beta)*a[i] + beta*b[i] ---
// NOTE: dst may alias a or b. Each thread reads/writes only its own index
// so no cross-thread race exists, but we avoid __restrict__ to be safe.
__global__ void atlas_ema_kernel(float* dst,
                                 const float* a,
                                 const float* b,
                                 size_t count, float oneMinusBeta, float beta)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count) return;
	dst[idx] = oneMinusBeta * a[idx] + beta * b[idx];
}

// --- Transform Fisher diagonal into new basis ---
// f_new[c] = max(sum_j overlap[c*r+j]^2 * f_old[j], 1e-12)
// Small kernel: r threads, one block.
__global__ void atlas_transform_fisher_kernel(const float* __restrict__ overlap,
                                              const float* __restrict__ f_old,
                                              float* __restrict__ f_new,
                                              int r)
{
	int c = threadIdx.x;
	if (c >= r) return;

	float fNew = 0.0f;
	for (int j = 0; j < r; ++j)
	{
		float o = overlap[c * r + j];
		fNew += o * o * f_old[j];
	}
	f_new[c] = (fNew > 1e-12f) ? fNew : 1e-12f;
}

// --- Scale kernel: dst[i] = src[i] * scale ---
__global__ void atlas_scale_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    size_t n, float scale)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	dst[idx] = src[idx] * scale;
}

// --- NaN/Inf guard: clamp non-finite weights to zero ---
// Uses CUDA's isfinite() intrinsic which is safe under --use_fast_math,
// unlike the manual (x == x) && (x - x == 0) pattern which can be
// optimized away (same reasoning as the CPU atlas_isfinite comment).
__global__ void atlas_guard_kernel(float* __restrict__ W, size_t mn)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= mn) return;
	if (!isfinite(W[idx]))
		W[idx] = 0.0f;
}

// --- Sigma2 reduction: compute sum(gW[i]^2 * gScale^2) ---
// Phase 1: each block writes its partial sum to d_partials[blockIdx.x].
__global__ void atlas_sigma2_kernel(const float* __restrict__ gW,
                                    size_t mn, float gScale,
                                    float* __restrict__ d_partials)
{
	extern __shared__ float smem[];

	float localSum = 0.0f;
	for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < mn;
	     i += (size_t)gridDim.x * blockDim.x)
	{
		float v = gW[i] * gScale;
		localSum += v * v;
	}

	float sum = blockReduceSum(localSum, smem);
	if (threadIdx.x == 0)
		d_partials[blockIdx.x] = sum;
}

// Phase 2: single-block kernel reduces d_partials[0..nBlocks-1] to d_out[0].
// Deterministic: single block, fixed thread order, no atomics.
__global__ void atlas_reduce_partials_kernel(const float* __restrict__ d_partials,
                                              int nBlocks,
                                              float* __restrict__ d_out)
{
	extern __shared__ float smem[];
	float val = 0.0f;
	for (int i = threadIdx.x; i < nBlocks; i += blockDim.x)
		val += d_partials[i];
	float sum = blockReduceSum(val, smem);
	if (threadIdx.x == 0)
		d_out[0] = sum;
}

// --- On-GPU Cholesky factorization + inversion of a small [r x r] matrix ---
// Single-block kernel: thread 0 does the serial Cholesky/backsolve; all threads
// participate in loading/storing. For r <= 128 this is cheaper than two PCIe
// round-trips (D2H + H2D) that the old CPU path required.
// d_G: [r*r] symmetric positive-definite Gram matrix (overwritten with R^{-1})
// regularize: if true, adds 1e-4 diagonal shift for ill-conditioned inputs.
__global__ void atlas_cholesky_inv_kernel(float* __restrict__ d_G, int r,
                                           bool regularize)
{
	// Only thread 0 does the serial work (r is small — O(r^3) ≈ few ms for r=128).
	if (threadIdx.x != 0) return;

	float* G = d_G;  // in-place on global memory

	// 1. Column-norm pre-scaling: G_s[i,j] = G[i,j] / (sqrt(G[i,i]) * sqrt(G[j,j]))
	float maxDiag = 0.0f;
	for (int j = 0; j < r; ++j)
		if (G[j * r + j] > maxDiag) maxDiag = G[j * r + j];
	float diagFloor = 1e-12f * (maxDiag > 0.0f ? maxDiag : 1.0f);

	// Use d_G + r*r as scratch for colScale (caller ensures buffer is large enough).
	float* colScale = G + r * r;
	for (int j = 0; j < r; ++j)
	{
		float d = G[j * r + j] > diagFloor ? G[j * r + j] : diagFloor;
		colScale[j] = sqrtf(d);
	}
	for (int i = 0; i < r; ++i)
		for (int j = 0; j < r; ++j)
			G[i * r + j] /= (colScale[i] * colScale[j]);

	// 2. Regularize
	if (regularize)
		for (int j = 0; j < r; ++j)
			G[j * r + j] += 1e-4f;

	// 3. Cholesky: G_s = R^T R (R upper triangular, row-major)
	for (int j = 0; j < r; ++j)
	{
		float d = G[j * r + j];
		for (int k = 0; k < j; ++k)
			d -= G[k * r + j] * G[k * r + j];
		if (d < 1e-8f) d = 1e-8f;
		G[j * r + j] = sqrtf(d);
		float invD = 1.0f / G[j * r + j];
		for (int i = j + 1; i < r; ++i)
		{
			float v = G[j * r + i];
			for (int k = 0; k < j; ++k)
				v -= G[k * r + j] * G[k * r + i];
			G[j * r + i] = v * invD;
		}
	}

	// Zero lower triangle
	for (int i = 0; i < r; ++i)
		for (int j = 0; j < i; ++j)
			G[i * r + j] = 0.0f;

	// 4. Un-scale: R_true = R * diag(colScale)
	for (int i = 0; i < r; ++i)
		for (int j = i; j < r; ++j)
			G[i * r + j] *= colScale[j];

	// 5. In-place inversion of upper-triangular R via back-substitution.
	// Result overwrites G with R^{-1}.
	// First, copy R to colScale area as scratch (we need original R during inversion).
	float* R = colScale;  // reuse colScale area — we're done with it
	for (int i = 0; i < r * r; ++i)
		R[i] = G[i];

	// Zero G for accumulation
	for (int i = 0; i < r * r; ++i)
		G[i] = 0.0f;

	for (int j = r - 1; j >= 0; --j)
	{
		G[j * r + j] = 1.0f / R[j * r + j];
		for (int i = j - 1; i >= 0; --i)
		{
			float s = 0.0f;
			for (int k = i + 1; k <= j; ++k)
				s += R[i * r + k] * G[k * r + j];
			G[i * r + j] = -s / R[i * r + i];
		}
	}
}

} // anonymous namespace

// ===========================================================================
//  Host-side kernel wrappers
// ===========================================================================

bool atlas_gpu_gram_schmidt(float* d_Q, int m, int r)
{
	if (r <= 0 || m <= 0) return true;
	int blockSize = 256;
	if (blockSize > m) blockSize = ((m + 31) / 32) * 32;
	if (blockSize < 32) blockSize = 32;
	int smemBytes = ((blockSize / 32) + 1) * sizeof(double);
	atlas_gs_kernel<<<1, blockSize, smemBytes, computeStream()>>>(d_Q, m, r);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Single-pass Cholesky QR helper (fully on-GPU, no host round-trips).
// If `regularize` is true, adds diagonal shift for float32 stability.
// d_scratch_rr: device buffer of at least scratchBufElems floats (Gram matrix + scratch).
// d_qrTemp: device buffer [m*r].
static bool cholesky_qr_pass(float* d_Q, int m, int r,
                              float* d_scratch_rr, size_t scratchBufElems,
                              float* d_qrTemp,
                              bool regularize)
{
	// The Cholesky kernel uses d_scratch_rr[0..r*r-1] for the Gram matrix and
	// d_scratch_rr[r*r..2*r*r-1] as working scratch. Ensure the caller provided enough.
	const size_t requiredElems = (size_t)r * r * 2;
	if (scratchBufElems < requiredElems)
	{
		fprintf(stderr, "[atlas-gpu] cholesky_qr_pass: scratch buffer too small "
		        "(%zu < %zu for r=%d)\n", scratchBufElems, requiredElems, r);
		return false;
	}

	// 1. Gram matrix: G[r,r] = Q^T Q (on GPU via cuBLAS)
	if (!sgemm_rowmajor_atb(r, r, m, 1.0f, d_Q, r, d_Q, r,
	                         0.0f, d_scratch_rr, r))
		return false;

	// 2. Cholesky factorization + inversion entirely on GPU.
	// The kernel reads G from d_scratch_rr, writes R^{-1} back to d_scratch_rr.
	// It uses d_scratch_rr[r*r .. 2*r*r-1] as scratch for colScale and R copy.
	atlas_cholesky_inv_kernel<<<1, 1, 0, computeStream()>>>(d_scratch_rr, r, regularize);
	ATLAS_CUDA_CHECK(cudaGetLastError());

	// 3. Q_new = Q * R_inv via cuBLAS SGEMM
	if (!sgemm_rowmajor(m, r, r, 1.0f, d_Q, r, d_scratch_rr, r,
	                     0.0f, d_qrTemp, r))
		return false;

	// 4. Copy result back to Q
	ATLAS_CUDA_CHECK(cudaMemcpyAsync(d_Q, d_qrTemp,
	                                 (size_t)m * r * sizeof(float),
	                                 cudaMemcpyDeviceToDevice,
	                                 computeStream()));

	return true;
}

// CholeskyQR² — two-pass orthonormalization for machine-precision results.
// Pass 1 (regularized): handles ill-conditioned input from power iteration,
//   gets columns to approximately unit norm.
// Pass 2 (exact): Gram matrix is now well-conditioned (diag ≈ 1), so Cholesky
//   succeeds without regularization, producing exact orthonormality.
// d_scratch_rr must be a device buffer of at least 2*r*r floats
// (the Cholesky kernel uses d_scratch_rr[r*r .. 2*r*r-1] as scratch).
static bool cholesky_qr(float* d_Q, int m, int r,
                         float* d_scratch_rr, size_t scratchBufElems,
                         float* d_qrTemp)
{
	if (r <= 0 || m <= 0) return true;

	// Pass 1: regularized — handles ill-conditioned input
	if (!cholesky_qr_pass(d_Q, m, r, d_scratch_rr, scratchBufElems, d_qrTemp, true))
		return false;

	// Pass 2: exact — cleans up regularization artifacts
	if (!cholesky_qr_pass(d_Q, m, r, d_scratch_rr, scratchBufElems, d_qrTemp, false))
		return false;

	return true;
}

bool atlas_gpu_weight_decay(float* d_W, size_t mn, float lr, float wd1, float wd2)
{
	if (mn == 0) return true;
	if (wd1 == 0.0f && wd2 == 0.0f) return true;
	int grid = (int)((mn + kBlock - 1) / kBlock);
	atlas_wd_kernel<<<grid, kBlock, 0, computeStream()>>>(d_W, mn, lr, wd1, wd2);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_baseline_update(float* d_W, const float* d_gW, size_t mn, float baseScaled)
{
	if (mn == 0) return true;
	int grid = (int)((mn + kBlock - 1) / kBlock);
	atlas_baseline_kernel<<<grid, kBlock, 0, computeStream()>>>(d_W, d_gW, mn, baseScaled);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_fisher_update(const float* d_gz, float* d_fisherDiag,
                             int r, int outerDim, float beta, float sampleScale,
                             bool rightSubspace, bool bootstrap)
{
	if (r <= 0 || outerDim <= 0) return true;
	int blockSize = 256;
	if (blockSize > outerDim) blockSize = ((outerDim + 31) / 32) * 32;
	if (blockSize < 32) blockSize = 32;
	int smemBytes = ((blockSize / 32) + 1) * sizeof(double);
	if (rightSubspace)
		atlas_fisher_col_kernel<<<r, blockSize, smemBytes, computeStream()>>>(
			d_gz, d_fisherDiag, r, outerDim, beta, 1.0f - beta, sampleScale,
			bootstrap ? 1 : 0);
	else
		atlas_fisher_kernel<<<r, blockSize, smemBytes, computeStream()>>>(
			d_gz, d_fisherDiag, r, outerDim, beta, 1.0f - beta, sampleScale,
			bootstrap ? 1 : 0);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_prepare_correction(const float* d_gz, const float* d_prevGz,
                                   const float* d_fisherDiag,
                                   float* d_out, int r, int outerDim,
                                   float onePlusMu, float negMu,
                                   float baselineRate, float lr, float eps,
                                   float kappaLr, float bcFactor,
                                   bool rightSubspace)
{
	if (r <= 0 || outerDim <= 0) return true;
	dim3 block(kBlock);
	if (rightSubspace)
	{
		// Col kernel: x=direction (r), y=row (outerDim) — coalesced reads
		dim3 grid((r + kBlock - 1) / kBlock, outerDim);
		atlas_correction_col_kernel<<<grid, block, 0, computeStream()>>>(
			d_gz, d_prevGz, d_fisherDiag, d_out, r, outerDim,
			onePlusMu, negMu, baselineRate, lr, eps, kappaLr, bcFactor);
	}
	else
	{
		// Row kernel: x=element (outerDim), y=direction (r)
		dim3 grid((outerDim + kBlock - 1) / kBlock, r);
		atlas_correction_kernel<<<grid, block, 0, computeStream()>>>(
			d_gz, d_prevGz, d_fisherDiag, d_out, r, outerDim,
			onePlusMu, negMu, baselineRate, lr, eps, kappaLr, bcFactor);
	}
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

static float atlas_nonnegative_finite(float v, float fallback)
{
	return (std::isfinite(v) && v >= 0.0f) ? v : fallback;
}

static float atlas_clamped_rate(float nominalLr,
                                float fisher,
                                float eps,
                                float kappaMax)
{
	if (!(nominalLr > 0.0f))
		return 0.0f;
	const float cappedKappa = atlas_nonnegative_finite(kappaMax, 0.0f);
	const float cap = cappedKappa * nominalLr;
	float rate = nominalLr / (fisher + eps);
	if (!std::isfinite(rate) || rate < 0.0f)
		rate = 0.0f;
	if (rate > cap)
		rate = cap;
	return rate;
}

static void compute_complement_block_sample(const std::vector<float>& gv,
                                            unsigned int rank,
                                            unsigned int outerDim,
                                            bool isRight,
                                            float statScaleSq,
                                            std::vector<float>& block)
{
	block.assign(static_cast<size_t>(rank) * rank, 0.0f);
	if (outerDim == 0u)
		return;
	for (unsigned int i = 0; i < rank; ++i)
	{
		for (unsigned int j = i; j < rank; ++j)
		{
			double dot = 0.0;
			if (isRight)
			{
				for (unsigned int row = 0; row < outerDim; ++row)
				{
					dot += static_cast<double>(gv[static_cast<size_t>(row) * rank + i])
					     * static_cast<double>(gv[static_cast<size_t>(row) * rank + j]);
				}
			}
			else
			{
				for (unsigned int col = 0; col < outerDim; ++col)
				{
					dot += static_cast<double>(gv[static_cast<size_t>(i) * outerDim + col])
					     * static_cast<double>(gv[static_cast<size_t>(j) * outerDim + col]);
				}
			}
			const float meansq = static_cast<float>(dot / static_cast<double>(outerDim)) * statScaleSq;
			block[static_cast<size_t>(i) * rank + j] = meansq;
			block[static_cast<size_t>(j) * rank + i] = meansq;
		}
	}
}

static float build_complement_correction_matrix(const std::vector<float>& block,
                                                unsigned int rank,
                                                unsigned int activeRank,
                                                float baselineRate,
                                                float nominalLr,
                                                float eps,
                                                float kappaMax,
                                                float bcFactor,
                                                std::vector<float>& corrMat)
{
	std::vector<float> scaledBlock(block);
	for (size_t i = 0; i < scaledBlock.size(); ++i)
		scaledBlock[i] *= bcFactor;
	symmetrize_block(scaledBlock, rank);

	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(scaledBlock, rank, eigVec, eigVal);
	corrMat.assign(static_cast<size_t>(rank) * rank, 0.0f);
	float maxRate = 0.0f;
	for (unsigned int mode = 0; mode < rank; ++mode)
	{
		if (mode >= activeRank)
			continue;
		float fisher = eigVal[mode];
		if (!std::isfinite(fisher) || fisher < 0.0f)
			fisher = 0.0f;
		const float rate = atlas_clamped_rate(nominalLr, fisher, eps, kappaMax);
		if (rate > maxRate)
			maxRate = rate;
		const float scale = baselineRate - rate;
		for (unsigned int i = 0; i < rank; ++i)
		{
			const float qi = eigVec[static_cast<size_t>(i) * rank + mode];
			for (unsigned int j = 0; j < rank; ++j)
				corrMat[static_cast<size_t>(i) * rank + j] +=
				    scale * qi * eigVec[static_cast<size_t>(j) * rank + mode];
		}
	}
	symmetrize_block(corrMat, rank);
	return maxRate;
}

bool atlas_gpu_mu_norms(const float* d_gz, const float* d_prevGz,
                         size_t rn, float* d_out, float* d_partials)
{
	if (rn == 0) return true;
	int grid = (int)((rn + kBlock - 1) / kBlock);
	if (grid > 256) grid = 256;
	// Phase 1: per-block partial sums (layout: [errPartials(grid), gzPartials(grid)])
	int smemBytes = 2 * ((kBlock / 32) + 1) * sizeof(float);
	atlas_mu_norms_kernel<<<grid, kBlock, smemBytes, computeStream()>>>(d_gz, d_prevGz, rn, grid, d_partials);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	// Phase 2: single-block deterministic reduction
	int smemBytes2 = 2 * ((kBlock / 32) + 1) * sizeof(float);
	atlas_reduce_partials2_kernel<<<1, kBlock, smemBytes2, computeStream()>>>(d_partials, grid, d_out);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_ema_blend(float* d_dst, const float* d_a, const float* d_b,
                          size_t count, float beta)
{
	if (count == 0) return true;
	int grid = (int)((count + kBlock - 1) / kBlock);
	atlas_ema_kernel<<<grid, kBlock, 0, computeStream()>>>(d_dst, d_a, d_b, count, 1.0f - beta, beta);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_transform_fisher(const float* d_overlap, const float* d_f_old,
                                  float* d_f_new, int r)
{
	if (r <= 0) return true;
	int blockSize = ((r + 31) / 32) * 32;
	if (blockSize < 32) blockSize = 32;
	if (blockSize > 1024) blockSize = 1024;
	atlas_transform_fisher_kernel<<<1, blockSize, 0, computeStream()>>>(d_overlap, d_f_old, d_f_new, r);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_scale_grad(const float* d_gW, float* d_out, size_t mn, float gScale)
{
	if (mn == 0) return true;
	int grid = (int)((mn + kBlock - 1) / kBlock);
	atlas_scale_kernel<<<grid, kBlock, 0, computeStream()>>>(d_gW, d_out, mn, gScale);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_guard(float* d_W, size_t mn)
{
	if (mn == 0) return true;
	const int block = 256;
	int grid = (int)((mn + block - 1) / block);
	atlas_guard_kernel<<<grid, block, 0, computeStream()>>>(d_W, mn);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

static bool ensureComplementStorage(GpuAtlasWeightState& state,
                                    unsigned int requestedRank)
{
	const unsigned int storageRank = atlas_storage_complement_rank(requestedRank);
	const unsigned int activeRank = state.r;
	const unsigned int subDim = state.rightSubspace ? state.n : state.m;
	const unsigned int outerDim = state.rightSubspace ? state.m : state.n;
	if (state.complementRank == storageRank
	    && state.V.size() == static_cast<size_t>(subDim) * storageRank
	    && state.complementBlock.size() == static_cast<size_t>(storageRank) * storageRank
	    && state.prevGv.size() == static_cast<size_t>(outerDim) * storageRank)
		return true;

	std::vector<float> h_U((size_t)subDim * activeRank, 0.0f);
	ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
	ATLAS_CUDA_CHECK(cudaMemcpy(h_U.data(), state.U.data(),
	                              h_U.size() * sizeof(float), cudaMemcpyDeviceToHost));

	const unsigned int oldRank = state.complementRank;
	std::vector<float> h_V_old((size_t)subDim * oldRank, 0.0f);
	std::vector<float> h_prev_old((size_t)outerDim * oldRank, 0.0f);
	std::vector<float> h_block_old((size_t)oldRank * oldRank, 0.0f);
	if (oldRank > 0u)
	{
		if (state.V.size() > 0u)
			ATLAS_CUDA_CHECK(cudaMemcpy(h_V_old.data(), state.V.data(),
			                              h_V_old.size() * sizeof(float), cudaMemcpyDeviceToHost));
		if (state.prevGv.size() > 0u)
			ATLAS_CUDA_CHECK(cudaMemcpy(h_prev_old.data(), state.prevGv.data(),
			                              h_prev_old.size() * sizeof(float), cudaMemcpyDeviceToHost));
		if (state.complementBlock.size() > 0u)
			ATLAS_CUDA_CHECK(cudaMemcpy(h_block_old.data(), state.complementBlock.data(),
			                              h_block_old.size() * sizeof(float), cudaMemcpyDeviceToHost));
	}

	std::vector<float> h_V((size_t)subDim * storageRank, 0.0f);
	std::vector<float> h_prev((size_t)outerDim * storageRank, 0.0f);
	std::vector<float> h_block((size_t)storageRank * storageRank, 0.0f);
	const unsigned int copyRank = (oldRank < storageRank) ? oldRank : storageRank;
	for (unsigned int i = 0; i < subDim; ++i)
		for (unsigned int c = 0; c < copyRank; ++c)
			h_V[static_cast<size_t>(i) * storageRank + c] =
			    h_V_old[static_cast<size_t>(i) * oldRank + c];
	if (state.rightSubspace)
	{
		for (unsigned int row = 0; row < outerDim; ++row)
			for (unsigned int c = 0; c < copyRank; ++c)
				h_prev[static_cast<size_t>(row) * storageRank + c] =
				    h_prev_old[static_cast<size_t>(row) * oldRank + c];
	}
	else
	{
		for (unsigned int c = 0; c < copyRank; ++c)
			for (unsigned int col = 0; col < outerDim; ++col)
				h_prev[static_cast<size_t>(c) * outerDim + col] =
				    h_prev_old[static_cast<size_t>(c) * outerDim + col];
	}
	for (unsigned int i = 0; i < copyRank; ++i)
		for (unsigned int j = 0; j < copyRank; ++j)
			h_block[static_cast<size_t>(i) * storageRank + j] =
			    h_block_old[static_cast<size_t>(i) * oldRank + j];

	for (unsigned int c = copyRank; c < storageRank; ++c)
		for (unsigned int i = 0; i < subDim; ++i)
			h_V[static_cast<size_t>(i) * storageRank + c] =
			    0.1f * static_cast<float>(((i + 1u) * (c + 3u)) % 17u + 1u);
	const unsigned int targetRank =
	    atlas_requested_complement_rank(storageRank, subDim, activeRank);
	orthonormalize_complement_block(h_V, storageRank, h_U, activeRank, activeRank, subDim, targetRank);

	state.complementRank = storageRank;
	if (state.activeComplementRank > storageRank)
		state.activeComplementRank = storageRank;
	if (state.trialComplementRank > storageRank
	    || state.trialComplementRank <= state.activeComplementRank)
		reset_complement_trial(state);
	if (!state.V.allocate(static_cast<size_t>(subDim) * storageRank)) return false;
	if (!state.complementBlock.allocate(static_cast<size_t>(storageRank) * storageRank)) return false;
	if (!state.prevGv.allocate(static_cast<size_t>(outerDim) * storageRank)) return false;
	if (!state.gv.allocate(static_cast<size_t>(outerDim) * storageRank)) return false;
	if (!state.gPredV.allocate(static_cast<size_t>(outerDim) * storageRank)) return false;
	if (!state.complementMat.allocate(static_cast<size_t>(storageRank) * storageRank)) return false;
	if (!state.V_old.allocate(static_cast<size_t>(subDim) * storageRank)) return false;
	if (!state.Bv.allocate(static_cast<size_t>(outerDim) * storageRank)) return false;
	if (!state.Zv.allocate(static_cast<size_t>(subDim) * storageRank)) return false;

	if (!state.V.upload(h_V.data(), h_V.size())) return false;
	if (!state.complementBlock.upload(h_block.data(), h_block.size())) return false;
	if (!state.prevGv.upload(h_prev.data(), h_prev.size())) return false;
	const float blockTrace = trace_complement_block(h_block, storageRank);
	if (!state.complementFisher.upload(&blockTrace, 1)) return false;
	return true;
}

static bool initComplementSector(GpuAtlasWeightState& state,
                                 glades::rng::Engine& rng)
{
	(void)rng;
	return ensureComplementStorage(state, 0u);
}

static bool refreshComplementSector(GpuAtlasWeightState& state,
                                    const float* d_grad,
                                    unsigned int powerIters,
                                    float betaRefresh)
{
	const bool isRight = state.rightSubspace;
	const unsigned int subDim = isRight ? state.n : state.m;
	const unsigned int outerDim = isRight ? state.m : state.n;
	const unsigned int activeRank = state.r;
	const unsigned int storageRank = state.complementRank;
	const unsigned int targetRank =
	    atlas_requested_complement_rank(storageRank, subDim, activeRank);
	if (targetRank == 0u)
	{
		ATLAS_CUDA_CHECK(cudaMemset(state.V.data(), 0, state.V.size() * sizeof(float)));
		ATLAS_CUDA_CHECK(cudaMemset(state.prevGv.data(), 0, state.prevGv.size() * sizeof(float)));
		ATLAS_CUDA_CHECK(cudaMemset(state.complementBlock.data(), 0, state.complementBlock.size() * sizeof(float)));
		const float zero = 0.0f;
		ATLAS_CUDA_CHECK(cudaMemcpy(state.complementFisher.data(), &zero,
		                              sizeof(float), cudaMemcpyHostToDevice));
		return true;
	}

	std::vector<float> h_U((size_t)subDim * activeRank);
	std::vector<float> h_V_old((size_t)subDim * storageRank, 0.0f);
	std::vector<float> h_block_old((size_t)storageRank * storageRank, 0.0f);
	ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
	ATLAS_CUDA_CHECK(cudaMemcpy(h_U.data(), state.U.data(),
	                              h_U.size() * sizeof(float), cudaMemcpyDeviceToHost));
	ATLAS_CUDA_CHECK(cudaMemcpy(h_V_old.data(), state.V.data(),
	                              h_V_old.size() * sizeof(float), cudaMemcpyDeviceToHost));
	ATLAS_CUDA_CHECK(cudaMemcpy(h_block_old.data(), state.complementBlock.data(),
	                              h_block_old.size() * sizeof(float), cudaMemcpyDeviceToHost));

	std::vector<float> q(h_V_old);
	orthonormalize_complement_block(q, storageRank, h_U, activeRank, activeRank, subDim, targetRank);

	std::vector<float> h_Z((size_t)subDim * storageRank, 0.0f);
	for (unsigned int p = 0; p < powerIters; ++p)
	{
		ATLAS_CUDA_CHECK(cudaMemcpy(state.V.data(), q.data(),
		                              q.size() * sizeof(float), cudaMemcpyHostToDevice));
		if (isRight)
		{
			if (!sgemm_rowmajor((int)outerDim, (int)storageRank, (int)subDim,
			                     1.0f,
			                     d_grad, (int)subDim,
			                     state.V.data(), (int)storageRank,
			                     0.0f,
			                     state.Bv.data(), (int)storageRank))
				return false;
			if (!sgemm_rowmajor_atb((int)subDim, (int)storageRank, (int)outerDim,
			                         1.0f,
			                         d_grad, (int)subDim,
			                         state.Bv.data(), (int)storageRank,
			                         0.0f,
			                         state.Zv.data(), (int)storageRank))
				return false;
		}
		else
		{
			if (!sgemm_rowmajor_atb((int)storageRank, (int)outerDim, (int)subDim,
			                         1.0f,
			                         state.V.data(), (int)storageRank,
			                         d_grad, (int)outerDim,
			                         0.0f,
			                         state.Bv.data(), (int)outerDim))
				return false;
			if (!sgemm_rowmajor_abt((int)subDim, (int)storageRank, (int)outerDim,
			                         1.0f,
			                         d_grad, (int)outerDim,
			                         state.Bv.data(), (int)outerDim,
			                         0.0f,
			                         state.Zv.data(), (int)storageRank))
				return false;
		}

		ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
		ATLAS_CUDA_CHECK(cudaMemcpy(h_Z.data(), state.Zv.data(),
		                              h_Z.size() * sizeof(float), cudaMemcpyDeviceToHost));
		q.swap(h_Z);
		orthonormalize_complement_block(q, storageRank, h_U, activeRank, activeRank, subDim, targetRank);
	}

	for (unsigned int i = 0; i < subDim; ++i)
	{
		for (unsigned int c = 0; c < storageRank; ++c)
		{
			q[static_cast<size_t>(i) * storageRank + c] =
			    (1.0f - betaRefresh) * h_V_old[static_cast<size_t>(i) * storageRank + c]
			  + betaRefresh * q[static_cast<size_t>(i) * storageRank + c];
		}
	}
	const unsigned int informativeRank =
	    orthonormalize_complement_block(q, storageRank, h_U, activeRank, activeRank, subDim, targetRank);

	std::vector<float> overlap((size_t)storageRank * storageRank, 0.0f);
	for (unsigned int c = 0; c < storageRank; ++c)
	{
		for (unsigned int k = 0; k < storageRank; ++k)
		{
			double dot = 0.0;
			for (unsigned int i = 0; i < subDim; ++i)
				dot += static_cast<double>(q[static_cast<size_t>(i) * storageRank + c])
				     * static_cast<double>(h_V_old[static_cast<size_t>(i) * storageRank + k]);
			overlap[static_cast<size_t>(c) * storageRank + k] = static_cast<float>(dot);
		}
	}
	std::vector<float> h_block_new;
	transform_symmetric_block(h_block_new, overlap, h_block_old, storageRank);
	symmetrize_block(h_block_new, storageRank);
	const float h_compFisher = trace_complement_block(h_block_new, storageRank);

	ATLAS_CUDA_CHECK(cudaMemcpy(state.V.data(), q.data(),
	                              q.size() * sizeof(float), cudaMemcpyHostToDevice));
	ATLAS_CUDA_CHECK(cudaMemset(state.prevGv.data(), 0, state.prevGv.size() * sizeof(float)));
	ATLAS_CUDA_CHECK(cudaMemcpy(state.complementBlock.data(), h_block_new.data(),
	                              h_block_new.size() * sizeof(float), cudaMemcpyHostToDevice));
	ATLAS_CUDA_CHECK(cudaMemcpy(state.complementFisher.data(), &h_compFisher,
	                              sizeof(float), cudaMemcpyHostToDevice));
	(void)outerDim;
	(void)informativeRank;
	return true;
}

// ===========================================================================
//  atlas_gpu_init — allocate and initialize core state buffers
// ===========================================================================

bool atlas_gpu_init(GpuAtlasWeightState& state,
                    unsigned int m, unsigned int n,
                    unsigned int rank, float muInit,
                    glades::rng::Engine& rng)
{
	state.m = m;
	state.n = n;
	state.r = rank;
	if (state.r > m) state.r = m;
	if (state.r > n) state.r = n;
	if (state.r == 0u) state.r = 1u;

	// Dual-space selection: use whichever dimension is smaller for the subspace.
	state.rightSubspace = (m > n);
	state.activeComplementRank = 0u;
	reset_complement_trial(state);

	// Dimension-proportional rank cap: subspace rank should not exceed 25% of
	// the subspace dimension. This prevents over-provisioning rank relative to
	// the available directions (e.g., for small layers where subDim < 4*rank).
	{
		const unsigned int subDim = state.rightSubspace ? n : m;
		const unsigned int maxRank = subDim / 4u;
		if (maxRank > 0u && state.r > maxRank)
			state.r = maxRank;
	}

	if (state.r > 256u)
	{
		fprintf(stderr, "[atlas-gpu] WARNING: rank %u > 256 may use significant GPU memory "
		        "(m=%u n=%u)\n", state.r, m, n);
	}

	const unsigned int r = state.r;
	const bool isRight = state.rightSubspace;
	const unsigned int subDim = isRight ? n : m;    // dimension the basis lives in
	const unsigned int outerDim = isRight ? m : n;  // the other dimension
	state.complementRank = atlas_storage_complement_rank(0u);
	const unsigned int cr = state.complementRank;
	const size_t sr = (size_t)subDim * r;    // basis size: U/V [subDim, r]
	const size_t or_ = (size_t)outerDim * r; // projected gradient size: gz [outerDim, r] or [r, outerDim]
	const size_t compBasis = (size_t)subDim * cr;
	const size_t compOuter = (size_t)outerDim * cr;
	const size_t compBlock = (size_t)cr * cr;

	// For left subspace: gz is [r, n] so or_ = r*n (but laid out as r*outerDim)
	// For right subspace: gz is [m, r] so or_ = m*r
	// Both cases: or_ = outerDim * r

	// Allocate persistent buffers
	if (!state.U.allocate(sr)) return false;
	if (!state.fisherDiag.allocate(r)) return false;
	if (!state.V.allocate(compBasis)) return false;
	if (!state.complementBlock.allocate(compBlock)) return false;
	if (!state.complementFisher.allocate(1)) return false;
	if (!state.prevGz.allocate(or_)) return false;
	if (!state.prevGv.allocate(compOuter)) return false;

	// Allocate per-step scratch buffers
	if (!state.gz.allocate(or_)) return false;
	if (!state.gPred.allocate(or_)) return false;
	if (!state.gv.allocate(compOuter)) return false;
	if (!state.gPredV.allocate(compOuter)) return false;
	if (!state.complementMat.allocate(compBlock)) return false;
	if (!state.d_reduce.allocate(2)) return false;
	if (!state.d_partials.allocate(512)) return false;

	// Pre-allocate refresh scratch buffers
	if (!state.U_old.allocate(sr)) return false;
	if (!state.f_old.allocate(r)) return false;
	if (!state.B.allocate(or_)) return false;
	if (!state.overlap.allocate((size_t)r * r * 2)) return false;
	if (!state.prevGzOld.allocate(or_)) return false;
	if (!state.V_old.allocate(compBasis)) return false;
	if (!state.Bv.allocate(compOuter)) return false;
	if (!state.Zv.allocate(compBasis)) return false;
	state.refreshAllocated = true;

	// Pre-allocate Cholesky QR temp buffer
	if (state.qrTemp.size() < sr)
	{
		state.qrTemp.free();
		if (!state.qrTemp.allocate(sr)) return false;
	}

	// Initialize subspace basis with random Gaussian values on CPU, then upload.
	{
		std::vector<float> h_U(sr);
		const float scale = 1.0f / sqrtf((float)subDim);
		for (size_t i = 0; i < sr; ++i)
			h_U[i] = glades::rng::standard_normal(rng) * scale;

		if (!state.U.upload(h_U.data(), sr)) return false;
	}

	// Orthogonalize basis on device (Cholesky QR² for machine-precision results)
	if (!cholesky_qr(state.U.data(), (int)subDim, (int)r,
	                 state.overlap.data(), state.overlap.size(),
	                 state.qrTemp.data()))
		return false;

	// Initialize Fisher diagonal to zero; the first real gradient bootstraps
	// the curvature statistics before they are used for preconditioning.
	{
		std::vector<float> initFisher(r, 0.0f);
		if (!state.fisherDiag.upload(initFisher.data(), r)) return false;
	}

	// Zero prevGz
	if (!state.prevGz.zero()) return false;
	if (!state.prevGv.zero()) return false;

	state.totalTrace = 0.0f;
	state.sigma2 = 0.0f;
	state.mu = muInit;
	state.step = 0ULL;
	state.initialized = true;
	if (!initComplementSector(state, rng)) return false;

	{
		const size_t totalElems = sr + r + or_
		    + compBasis + compBlock + 1 + compOuter
		    + or_ + or_ + compOuter + compOuter + compBlock + 2 + 512
		    + sr + r + or_ + (size_t)r*r*2 + or_ + compBasis + compOuter + compBasis
		    + sr;
		fprintf(stderr, "[atlas-gpu] init m=%u n=%u r=%u %s total_gpu_bytes=%zu\n",
		        m, n, r, isRight ? "RIGHT" : "LEFT", totalElems * sizeof(float));
	}

	ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
	return true;
}

// ===========================================================================
//  refreshSubspace — periodic subspace update via randomized power iteration
// ===========================================================================

// d_grad is the raw gradient [m*n], gScale folds into SGEMM alpha.
// Eigenvectors are scale-invariant, so we pass alpha=1 (no scaling).
//
// Left subspace (m <= n): power iteration finds top-r left singular vectors of G.
//   Q ∈ R^{m×r}: B = G^T Q, Z = G B, Q = orth(Z)
// Right subspace (m > n): power iteration finds top-r right singular vectors of G.
//   Q ∈ R^{n×r}: B = G Q, Z = G^T B, Q = orth(Z)
static bool refreshSubspace(GpuAtlasWeightState& state,
                            const float* d_grad,
                            unsigned int m, unsigned int n,
                            unsigned int powerIters, float betaRefresh)
{
	const unsigned int r = state.r;
	if (r == 0u || m == 0u || n == 0u) return true;

	const bool isRight = state.rightSubspace;
	const unsigned int subDim = isRight ? n : m;
	const unsigned int outerDim = isRight ? m : n;

	if (state.overlap.size() < (size_t)r * r * 2)
	{
		fprintf(stderr, "[atlas-gpu] FATAL: overlap buffer too small (%zu < %zu)\n",
		        state.overlap.size(), (size_t)r * r * 2);
		return false;
	}

	const size_t sr = (size_t)subDim * r;
	const size_t or_ = (size_t)outerDim * r;

	// Save old basis and Fisher
	ATLAS_CUDA_CHECK(cudaMemcpyAsync(state.U_old.data(), state.U.data(),
	                                 sr * sizeof(float), cudaMemcpyDeviceToDevice,
	                                 computeStream()));
	ATLAS_CUDA_CHECK(cudaMemcpyAsync(state.f_old.data(), state.fisherDiag.data(),
	                                 r * sizeof(float), cudaMemcpyDeviceToDevice,
	                                 computeStream()));

	float* d_Q = state.U.data();

	for (unsigned int p = 0; p < powerIters; ++p)
	{
		if (isRight)
		{
			// Right subspace: Q ∈ R^{n×r}
			// B[m,r] = G[m,n] * Q[n,r]
			if (!sgemm_rowmajor((int)m, (int)r, (int)n,
			                     1.0f,
			                     d_grad, (int)n,
			                     d_Q, (int)r,
			                     0.0f,
			                     state.B.data(), (int)r))
				return false;

			// Z[n,r] = G^T[n,m] * B[m,r]
			if (!sgemm_rowmajor_atb((int)n, (int)r, (int)m,
			                         1.0f,
			                         d_grad, (int)n,
			                         state.B.data(), (int)r,
			                         0.0f,
			                         d_Q, (int)r))
				return false;
		}
		else
		{
			// Left subspace: Q ∈ R^{m×r}
			// B[n,r] = G^T[n,m] * Q[m,r]
			if (!sgemm_rowmajor_atb((int)n, (int)r, (int)m,
			                         1.0f,
			                         d_grad, (int)n,
			                         d_Q, (int)r,
			                         0.0f,
			                         state.B.data(), (int)r))
				return false;

			// Z[m,r] = G[m,n] * B[n,r]
			if (!sgemm_rowmajor((int)m, (int)r, (int)n,
			                     1.0f,
			                     d_grad, (int)n,
			                     state.B.data(), (int)r,
			                     0.0f,
			                     d_Q, (int)r))
				return false;
		}

		if (!cholesky_qr(d_Q, (int)subDim, (int)r,
		                 state.overlap.data(), state.overlap.size(),
		                 state.qrTemp.data()))
			return false;
	}

	// --- EMA blend ---
	if (!atlas_gpu_ema_blend(d_Q, state.U_old.data(), d_Q, sr, betaRefresh))
		return false;

	// --- NaN guard on blended basis ---
	// The power iteration + CholeskyQR can produce NaN/Inf when the gradient
	// is nearly rank-deficient (e.g. layer 0 Wq with very small gradients).
	// CholeskyQR's R_inv amplifies tiny values by 1e10+, and cross-terms in
	// the back-substitution can overflow to Inf → NaN in the SGEMM output.
	// The EMA blend then propagates: 0.5*U_old + 0.5*NaN = NaN.
	//
	// Detection: sample a few elements of d_Q. If any are non-finite, the
	// entire matrix is likely corrupted (NaN propagates through SGEMM).
	// Recovery: restore U_old. This makes the refresh a no-op for this weight:
	// overlap = I, Fisher and prevGz are unchanged. The weight simply keeps
	// its current subspace until the next refresh with better-conditioned gradients.
	{
		ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
		float qCheck[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		ATLAS_CUDA_CHECK(cudaMemcpy(&qCheck[0], d_Q, sizeof(float), cudaMemcpyDeviceToHost));
		ATLAS_CUDA_CHECK(cudaMemcpy(&qCheck[1], d_Q + sr / 4, sizeof(float), cudaMemcpyDeviceToHost));
		ATLAS_CUDA_CHECK(cudaMemcpy(&qCheck[2], d_Q + sr / 2, sizeof(float), cudaMemcpyDeviceToHost));
		ATLAS_CUDA_CHECK(cudaMemcpy(&qCheck[3], d_Q + sr - 1, sizeof(float), cudaMemcpyDeviceToHost));
		if (!std::isfinite(qCheck[0]) || !std::isfinite(qCheck[1])
		    || !std::isfinite(qCheck[2]) || !std::isfinite(qCheck[3]))
		{
			// Restore old basis — refresh becomes no-op.
			ATLAS_CUDA_CHECK(cudaMemcpy(d_Q, state.U_old.data(),
			                            sr * sizeof(float), cudaMemcpyDeviceToDevice));
			// Restore old Fisher (f_old was saved earlier).
			ATLAS_CUDA_CHECK(cudaMemcpy(state.fisherDiag.data(), state.f_old.data(),
			                            r * sizeof(float), cudaMemcpyDeviceToDevice));
			// prevGz stays unchanged (we haven't modified it yet).
			return true; // skip overlap/transform — basis unchanged
		}
	}

	// Re-orthogonalize after blending
	if (!cholesky_qr(d_Q, (int)subDim, (int)r,
	                 state.overlap.data(), state.overlap.size(),
	                 state.qrTemp.data()))
		return false;

	// --- Overlap matrix O[r,r] = Q_new^T * Q_old ---
	if (!sgemm_rowmajor_atb((int)r, (int)r, (int)subDim,
	                         1.0f,
	                         d_Q, (int)r,
	                         state.U_old.data(), (int)r,
	                         0.0f,
	                         state.overlap.data(), (int)r))
		return false;

	// --- Transform Fisher diagonal into new basis ---
	if (!atlas_gpu_transform_fisher(state.overlap.data(), state.f_old.data(),
	                                  state.fisherDiag.data(), (int)r))
		return false;

	// --- Transform prevGz into new basis ---
	ATLAS_CUDA_CHECK(cudaMemcpyAsync(state.prevGzOld.data(), state.prevGz.data(),
	                                 or_ * sizeof(float), cudaMemcpyDeviceToDevice,
	                                 computeStream()));

	if (isRight)
	{
		// prevGz is [m, r]: prevGz_new[i,c] = sum_j prevGzOld[i,j] * overlap[j,c]
		// This is: prevGz_new[m,r] = prevGzOld[m,r] * overlap[r,r]
		if (!sgemm_rowmajor((int)outerDim, (int)r, (int)r,
		                     1.0f,
		                     state.prevGzOld.data(), (int)r,
		                     state.overlap.data(), (int)r,
		                     0.0f,
		                     state.prevGz.data(), (int)r))
			return false;
	}
	else
	{
		// prevGz is [r, n]: prevGz_new[c,j] = sum_k overlap[c,k] * prevGzOld[k,j]
		// This is: prevGz_new[r,n] = overlap[r,r] * prevGzOld[r,n]
		if (!sgemm_rowmajor((int)r, (int)outerDim, (int)r,
		                     1.0f,
		                     state.overlap.data(), (int)r,
		                     state.prevGzOld.data(), (int)outerDim,
		                     0.0f,
		                     state.prevGz.data(), (int)outerDim))
			return false;
	}

	return true;
}

// ===========================================================================
//  atlas_gpu_step — one full BRSP optimizer step on GPU
// ===========================================================================

bool atlas_gpu_step(GpuAtlasWeightState& state,
                    float* d_W, float* d_gW,
                    unsigned int m, unsigned int n,
                    float invBatch, float lr,
                    float wd1, float wd2, float gradScale,
                    const glades::ATLASConfig& ac,
                    shmea::GLogger* logger,
                    const char* tag)
{
	const float beta = ac.beta;
	const float muMin = ac.muMin;
	const float muMax = ac.muMax;
	const float eps = ac.eps;
	const unsigned int tSub = ac.tSub;
	const unsigned int powerIters = ac.powerIters;
	const float betaRefresh = ac.betaRefresh;
	const float kappaMax = ac.kappaMax;
	const float muGrowthRate = ac.muGrowthRate;
	if (!state.initialized) return false;

	const unsigned int r = state.r;
	const bool isRight = state.rightSubspace;
	const unsigned int subDim = isRight ? n : m;
	const unsigned int outerDim = isRight ? m : n;
	const unsigned int enabledComplementRank =
	    atlas_enabled_complement_rank(ac.complementRank, tag, m, n, r);
	const bool adaptiveComplementController = (tag && tag[0]);
	if (!ensureComplementStorage(state, enabledComplementRank))
		return false;
	const unsigned int complementRank = state.complementRank;
	unsigned int effectiveComplementRank = 0u;
	const size_t mn = (size_t)m * n;
	const size_t or_ = (size_t)outerDim * r;  // gz/gPred/prevGz size
	const size_t compOuter = (size_t)outerDim * complementRank;
	state.step += 1ULL;
	if (state.activeComplementRank > enabledComplementRank)
		state.activeComplementRank = enabledComplementRank;
	if (state.trialComplementRank > enabledComplementRank
	    || state.trialComplementRank <= state.activeComplementRank)
		reset_complement_trial(state);
	const bool complementControlStep =
	    adaptiveComplementController
	    && (enabledComplementRank > 0u)
	    && ((tSub == 0u) || (state.step % (unsigned long long)tSub) == 0ULL);

	// Guard incoming gradient against NaN/Inf before any computation.
	// A single NaN in d_gW would corrupt sigma2 (via reduction), the subspace
	// basis U (via power iteration at refresh), gz (via projection), and the
	// baseline update. Replacing NaN with 0 is safe: zero gradient entries
	// contribute nothing, and the weight matrix is unaffected by them.
	if (!atlas_gpu_guard(d_gW, mn))
		return false;

	const float gScale = invBatch * gradScale;
	const float statScaleSq = (invBatch > 0.0f) ? (1.0f / (invBatch * invBatch)) : 1.0f;

	// NaN diagnostic: verify guard worked (only at diagnostic steps)
	if (logger && tag && tSub > 0u && ((state.step % (unsigned long long)tSub) == 0ULL))
	{
		ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
		float gwSamples[3] = {0.0f, 0.0f, 0.0f};
		cudaMemcpy(&gwSamples[0], d_gW, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gwSamples[1], d_gW + mn / 2, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gwSamples[2], d_gW + mn - 1, sizeof(float), cudaMemcpyDeviceToHost);
		if (!std::isfinite(gwSamples[0]) || !std::isfinite(gwSamples[1]) || !std::isfinite(gwSamples[2]))
		{
			std::ostringstream oss;
			oss << "event=atlas_nan_diag tag=" << tag << " step=" << state.step
			    << " location=gW_after_guard"
			    << " gW[0]=" << gwSamples[0] << " gW[mid]=" << gwSamples[1]
			    << " gW[end]=" << gwSamples[2];
			logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
		}
	}

	// === Bias correction factor (matches CPU path) ===
	// Compensates for zero-initialization bias in EMA quantities.
	float bcFactor = 1.0f;
	if (ac.biasCorrection)
	{
		const double betaPow = pow((double)beta, (double)state.step);
		const double denom = 1.0 - betaPow;
		if (denom > 1e-15)
			bcFactor = (float)(1.0 / denom);
	}

	float traceCapture = 0.0f;
	float activeTraceCapture = 0.0f;
	float sectorTraceCapture = 0.0f;
	float closureGap = 0.0f;
	float sectorFisherDiag = 0.0f;
	float sectorRate = 0.0f;

	// === Step 1: Compute the normalized covariance trace ===
	// Two-pass deterministic reduction: block partials → single-block sum.
	// totalTrace tracks tr(C) where C is the minibatch-normalized covariance on
	// the accumulated-gradient gauge used historically by ATLAS.
	{
		int grid = (int)((mn + kBlock - 1) / kBlock);
		if (grid > 256) grid = 256;
		int smemBytes = ((kBlock / 32) + 1) * sizeof(float);
		// Phase 1: per-block partial sums
		atlas_sigma2_kernel<<<grid, kBlock, smemBytes>>>(d_gW, mn, gradScale,
		                                                  state.d_partials.data());
		ATLAS_CUDA_CHECK(cudaGetLastError());
		// Phase 2: single-block deterministic final reduction
		atlas_reduce_partials_kernel<<<1, kBlock, smemBytes>>>(
		    state.d_partials.data(), grid, state.d_reduce.data());
		ATLAS_CUDA_CHECK(cudaGetLastError());
		// Synchronous D2H — consume in the same step (matches CPU path).
		float h_traceSum = 0.0f;
		ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
		ATLAS_CUDA_CHECK(cudaMemcpy(&h_traceSum, state.d_reduce.data(),
		                              sizeof(float), cudaMemcpyDeviceToHost));
		const float traceSample = h_traceSum / (float)outerDim;
		state.totalTrace = atlas_bootstrap_or_ema(state.totalTrace, traceSample, beta, state.step);
		if (state.totalTrace < eps) state.totalTrace = eps;
		if (!std::isfinite(state.totalTrace))
		{
			state.totalTrace = eps;
			if (logger)
			{
				std::ostringstream oss;
				oss << "event=atlas_gpu_total_trace_recovery step=" << state.step;
				if (tag) oss << " tag=" << tag;
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
		}
	}

	// === Step 2: Periodic subspace refresh ===
	// Pass raw d_gW directly — eigenvectors are scale-invariant.
	if (tSub > 0u && (state.step % (unsigned long long)tSub) == 0ULL)
	{
		// NaN diagnostic: check U BEFORE refresh
		if (logger && tag)
		{
			const unsigned int subDim = isRight ? n : m;
			const size_t sr = (size_t)subDim * r;
			ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
			float uPre[2] = {0.0f, 0.0f};
			cudaMemcpy(&uPre[0], state.U.data(), sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(&uPre[1], state.U.data() + sr - 1, sizeof(float), cudaMemcpyDeviceToHost);
			if (!std::isfinite(uPre[0]) || !std::isfinite(uPre[1]))
			{
				std::ostringstream oss;
				oss << "event=atlas_nan_diag tag=" << tag << " step=" << state.step
				    << " location=U_BEFORE_refresh"
				    << " U[0]=" << uPre[0] << " U[end]=" << uPre[1];
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
		}

		if (!refreshSubspace(state, d_gW, m, n, powerIters, betaRefresh))
		{
			if (logger)
			{
				std::ostringstream oss;
				oss << "event=atlas_gpu_refresh_failure step=" << state.step
				    << " m=" << m << " n=" << n << " rank=" << r;
				if (tag) oss << " tag=" << tag;
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
			return false;
		}
		if (enabledComplementRank > 0u
		    && !refreshComplementSector(state, d_gW, powerIters, betaRefresh))
			return false;

		// NaN diagnostic: check U after refresh
		if (logger && tag)
		{
			ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
			float uSample[2] = {0.0f, 0.0f};
			const unsigned int subDim = isRight ? n : m;
			const size_t sr = (size_t)subDim * r;
			cudaMemcpy(&uSample[0], state.U.data(), sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(&uSample[1], state.U.data() + sr / 2, sizeof(float), cudaMemcpyDeviceToHost);
			if (!std::isfinite(uSample[0]) || !std::isfinite(uSample[1]))
			{
				std::ostringstream oss;
				oss << "event=atlas_nan_diag tag=" << tag
				    << " step=" << state.step
				    << " location=U_after_refresh"
				    << " U[0]=" << uSample[0] << " U[mid]=" << uSample[1];
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
		}
	}

	// === Step 3: Decoupled weight decay ===
	if (!atlas_gpu_weight_decay(d_W, mn, lr, wd1, wd2))
		return false;

	// === Step 4: Project gradient to subspace ===
	if (isRight)
	{
		// Right subspace: gz[m,r] = gScale * G[m,n] * V[n,r]
		if (!sgemm_rowmajor((int)m, (int)r, (int)n,
		                     gScale,
		                     d_gW, (int)n,
		                     state.U.data(), (int)r,
		                     0.0f,
		                     state.gz.data(), (int)r))
			return false;
	}
	else
	{
		// Left subspace: gz[r,n] = gScale * U^T[r,m] * G[m,n]
		if (!sgemm_rowmajor_atb((int)r, (int)n, (int)m,
		                         gScale,
		                         state.U.data(), (int)r,
		                         d_gW, (int)n,
		                         0.0f,
		                         state.gz.data(), (int)n))
			return false;
	}

	if (enabledComplementRank > 0u)
	{
		if (isRight)
		{
			if (!sgemm_rowmajor((int)outerDim, (int)complementRank, (int)subDim,
			                     gScale,
			                     d_gW, (int)subDim,
			                     state.V.data(), (int)complementRank,
			                     0.0f,
			                     state.gv.data(), (int)complementRank))
				return false;
		}
		else
		{
			if (!sgemm_rowmajor_atb((int)complementRank, (int)outerDim, (int)subDim,
			                         gScale,
			                         state.V.data(), (int)complementRank,
			                         d_gW, (int)outerDim,
			                         0.0f,
			                         state.gv.data(), (int)outerDim))
				return false;
		}
	}
	else if (state.gv.size() > 0u)
	{
		ATLAS_CUDA_CHECK(cudaMemset(state.gv.data(), 0, state.gv.size() * sizeof(float)));
	}

	// NaN diagnostic: check gz, prevGz, and U after projection
	if (logger && tag && (state.step % (unsigned long long)tSub) == 0ULL)
	{
		ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
		float gzSample[2] = {0.0f, 0.0f};
		float prevGzSample[2] = {0.0f, 0.0f};
		float uSample = 0.0f;
		float gwSample = 0.0f;
		cudaMemcpy(&gzSample[0], state.gz.data(), sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gzSample[1], state.gz.data() + or_ / 2, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&prevGzSample[0], state.prevGz.data(), sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&prevGzSample[1], state.prevGz.data() + or_ / 2, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&uSample, state.U.data(), sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gwSample, d_gW, sizeof(float), cudaMemcpyDeviceToHost);
		if (!std::isfinite(gzSample[0]) || !std::isfinite(gzSample[1])
		    || !std::isfinite(prevGzSample[0]) || !std::isfinite(prevGzSample[1]))
		{
			std::ostringstream oss;
			oss << "event=atlas_nan_diag tag=" << tag
			    << " step=" << state.step
			    << " location=gz_after_projection"
			    << " gz[0]=" << gzSample[0] << " gz[mid]=" << gzSample[1]
			    << " prevGz[0]=" << prevGzSample[0] << " prevGz[mid]=" << prevGzSample[1]
			    << " U[0]=" << uSample << " gW[0]=" << gwSample
			    << " gScale=" << gScale;
			logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
		}
	}

	// === Step 5: Update Fisher diagonal ===
	if (!atlas_gpu_fisher_update(state.gz.data(), state.fisherDiag.data(),
	                             (int)r, (int)outerDim, beta, statScaleSq, isRight,
	                             state.step == 1ULL))
		return false;
	// === Step 6: Recompute sigma2 from the shared covariance model ===
	std::vector<float> h_fisher(r);
	std::vector<float> h_block((size_t)complementRank * complementRank, 0.0f);
	std::vector<float> h_sample((size_t)complementRank * complementRank, 0.0f);
	std::vector<float> h_gv(compOuter, 0.0f);
	float h_blockTrace = 0.0f;
	double activeSectorTrace = 0.0;
	float complementTailMean = eps;
	float complementScoutTop = 0.0f;
	double complementScoutProjected = 0.0;
	float complementBirthKelly = 0.0f;
	double complementTrialKelly = 0.0;
	double complementTrialAlignment = 0.0;
	double complementTrialContamination = 0.0;
	double complementTrialReturn = 0.0;
	double complementTrialScore = 0.0;
	unsigned int informativeComplementRank = 0u;
	ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
	ATLAS_CUDA_CHECK(cudaMemcpy(h_fisher.data(), state.fisherDiag.data(),
	                              r * sizeof(float), cudaMemcpyDeviceToHost));
	if (enabledComplementRank > 0u)
	{
		ATLAS_CUDA_CHECK(cudaMemcpy(h_gv.data(), state.gv.data(),
		                              h_gv.size() * sizeof(float), cudaMemcpyDeviceToHost));
		ATLAS_CUDA_CHECK(cudaMemcpy(h_block.data(), state.complementBlock.data(),
		                              h_block.size() * sizeof(float), cudaMemcpyDeviceToHost));
		compute_complement_block_sample(h_gv, complementRank, outerDim, isRight, statScaleSq, h_sample);
		for (size_t i = 0; i < h_block.size(); ++i)
			h_block[i] = atlas_bootstrap_or_ema(h_block[i], h_sample[i], beta, state.step);
		symmetrize_block(h_block, complementRank);
		h_blockTrace = trace_complement_block(h_block, complementRank);
		ATLAS_CUDA_CHECK(cudaMemcpy(state.complementBlock.data(), h_block.data(),
		                              h_block.size() * sizeof(float), cudaMemcpyHostToDevice));
		ATLAS_CUDA_CHECK(cudaMemcpy(state.complementFisher.data(), &h_blockTrace,
		                              sizeof(float), cudaMemcpyHostToDevice));
	}
	else
	{
		h_blockTrace = 0.0f;
		state.activeComplementRank = 0u;
		reset_complement_trial(state);
		ATLAS_CUDA_CHECK(cudaMemset(state.complementBlock.data(), 0, state.complementBlock.size() * sizeof(float)));
		ATLAS_CUDA_CHECK(cudaMemcpy(state.complementFisher.data(), &h_blockTrace,
		                              sizeof(float), cudaMemcpyHostToDevice));
	}
	std::vector<float> h_V((size_t)subDim * complementRank, 0.0f);
	ATLAS_CUDA_CHECK(cudaMemcpy(h_V.data(), state.V.data(),
	                              h_V.size() * sizeof(float), cudaMemcpyDeviceToHost));
	informativeComplementRank =
	    effective_complement_rank(h_V, complementRank, enabledComplementRank, subDim, r);
	std::vector<float> h_blockEigVec;
	std::vector<float> h_blockEigVal;
	if (enabledComplementRank > 0u && informativeComplementRank > 0u)
	{
		jacobi_eigendecompose(h_block, complementRank, h_blockEigVec, h_blockEigVal);
		for (unsigned int i = 0; i < complementRank; ++i)
		{
			if (!std::isfinite(h_blockEigVal[i]) || h_blockEigVal[i] < 0.0f)
				h_blockEigVal[i] = 0.0f;
		}
		std::vector<float> h_scoutEigVec;
		std::vector<float> h_scoutEigVal;
		if (adaptiveComplementController)
		{
			symmetrize_block(h_sample, complementRank);
			jacobi_eigendecompose(h_sample, complementRank, h_scoutEigVec, h_scoutEigVal);
			for (unsigned int i = 0; i < complementRank; ++i)
			{
				if (!std::isfinite(h_scoutEigVal[i]) || h_scoutEigVal[i] < 0.0f)
					h_scoutEigVal[i] = 0.0f;
			}
			if (!h_scoutEigVal.empty())
				complementScoutTop = h_scoutEigVal[0];
		}
		if (!adaptiveComplementController)
		{
			state.activeComplementRank = informativeComplementRank;
			reset_complement_trial(state);
		}
		else if (complementControlStep)
		{
			const unsigned int prevActiveComplementRank = state.activeComplementRank;
			double activeTraceNow = 0.0;
			for (unsigned int c = 0; c < r; ++c)
				activeTraceNow += static_cast<double>(h_fisher[c]);
			double birthKelly = 0.0;
			double birthScout = 0.0;
			state.activeComplementRank =
			    choose_active_complement_rank(state,
			                                  h_blockEigVal,
			                                  h_blockEigVec,
			                                  h_scoutEigVal.empty() ? 0 : &h_scoutEigVal,
			                                  h_scoutEigVec.empty() ? 0 : &h_scoutEigVec,
			                                  informativeComplementRank,
			                                  state.activeComplementRank,
			                                  activeTraceNow,
			                                  state.totalTrace,
			                                  subDim,
			                                  r,
			                                  eps,
			                                  &birthKelly,
			                                  &birthScout,
			                                  &complementScoutProjected,
			                                  &complementTrialKelly,
			                                  &complementTrialAlignment,
			                                  &complementTrialContamination,
			                                  &complementTrialReturn,
			                                  &complementTrialScore);
			complementBirthKelly = static_cast<float>(birthKelly);
			complementScoutTop = static_cast<float>(birthScout);
			if (state.activeComplementRank > informativeComplementRank)
				state.activeComplementRank = informativeComplementRank;
			if (logger && state.activeComplementRank != prevActiveComplementRank)
			{
				const double closedTrace = std::max<double>(
				    static_cast<double>(state.totalTrace),
				    activeTraceNow + sum_leading_spectrum(h_blockEigVal, informativeComplementRank));
				const double prevTrace =
				    sum_leading_spectrum(h_blockEigVal, prevActiveComplementRank);
				const double tailMean =
				    complement_tail_mean(closedTrace, activeTraceNow, prevTrace,
				                         subDim, r, prevActiveComplementRank, eps);
				std::ostringstream oss;
				oss << "event=atlas_gpu_complement_rank_change";
				if (tag) oss << " tag=" << tag;
				oss << " step=" << state.step
				    << " m=" << m
				    << " n=" << n
				    << " complement_rank_cap=" << enabledComplementRank
				    << " complement_rank_prev=" << prevActiveComplementRank
				    << " complement_rank_new=" << state.activeComplementRank
				    << " tail_mean=" << static_cast<float>(tailMean)
				    << " birth_kelly=" << complementBirthKelly
				    << " birth_scout=" << complementScoutTop
				    << " birth_projected=" << complementScoutProjected
				    << " trial_kelly=" << complementTrialKelly
				    << " trial_alignment=" << complementTrialAlignment
				    << " trial_contam=" << complementTrialContamination
				    << " trial_return=" << complementTrialReturn
				    << " trial_score=" << complementTrialScore
				    << " next_mode="
				    << ((prevActiveComplementRank < informativeComplementRank)
				            ? h_blockEigVal[prevActiveComplementRank]
				            : 0.0f)
				    << " weakest_active="
				    << ((prevActiveComplementRank > 0u)
				            ? h_blockEigVal[prevActiveComplementRank - 1u]
				            : 0.0f);
				logger->info("ATLAS", shmea::GString(oss.str().c_str()));
			}
		}
	}
	else
	{
		state.activeComplementRank = 0u;
		reset_complement_trial(state);
	}
	if (state.activeComplementRank > informativeComplementRank)
		state.activeComplementRank = informativeComplementRank;
	if (state.trialComplementRank > informativeComplementRank
	    || state.trialComplementRank <= state.activeComplementRank)
		reset_complement_trial(state);
	effectiveComplementRank = state.activeComplementRank;
	activeSectorTrace = sum_leading_spectrum(h_blockEigVal, effectiveComplementRank);
	{
		double activeTrace = 0.0;
		double sectorTrace = 0.0;
		double gap = 0.0;
		for (unsigned int c = 0; c < r; ++c)
			activeTrace += static_cast<double>(h_fisher[c]);
		const double fullBlockTrace =
		    sum_leading_spectrum(h_blockEigVal, informativeComplementRank);
		const double closedTrace = std::max<double>(static_cast<double>(state.totalTrace),
		                                            activeTrace + fullBlockTrace);
		complementTailMean = static_cast<float>(
		    complement_tail_mean(closedTrace, activeTrace, activeSectorTrace,
		                         subDim, r, effectiveComplementRank, eps));
		state.sigma2 = compute_complement_sigma2(state.totalTrace, h_fisher, r,
		                                         activeSectorTrace, effectiveComplementRank, subDim, eps,
		                                         &activeTrace, &sectorTrace, &gap);
		activeTraceCapture = (state.totalTrace > 1e-30f)
		    ? (float)(activeTrace / (double)state.totalTrace)
		    : 0.0f;
		sectorTraceCapture = (state.totalTrace > 1e-30f)
		    ? (float)(sectorTrace / (double)state.totalTrace)
		    : 0.0f;
		traceCapture = (state.totalTrace > 1e-30f)
		    ? (float)((activeTrace + sectorTrace) / (double)state.totalTrace)
		    : 0.0f;
		closureGap = (float)gap;
	}
	if (!std::isfinite(state.sigma2))
		state.sigma2 = eps;
	sectorFisherDiag = static_cast<float>(activeSectorTrace);

	// === Step 7: Full-space baseline update ===
	{
		// Apply bias correction to sigma2 (matches CPU path).
		const float effSigma2 = state.sigma2 * bcFactor;
		const float baselineRate = atlas_clamped_rate(lr, effSigma2, eps, kappaMax);
		state.lastBaselineRate = baselineRate;
		const float baseScaled = baselineRate * gScale;
		if (!atlas_gpu_baseline_update(d_W, d_gW, mn, baseScaled))
			return false;

		// === Step 8: Subspace correction with PNG ===
		const float onePlusMu = 1.0f + state.mu;
		const float negMu = -state.mu;

		if (!atlas_gpu_prepare_correction(state.gz.data(), state.prevGz.data(),
		                                   state.fisherDiag.data(),
		                                   state.gPred.data(), (int)r, (int)outerDim,
		                                   onePlusMu, negMu,
		                                   baselineRate, lr, eps, kappaMax * lr,
		                                   bcFactor, isRight))
			return false;

		if (isRight)
		{
			// Right subspace: W[m,n] += gPred[m,r] * V^T[r,n]
			// sgemm_rowmajor_abt: C[M,N] = A[M,K] * B^T[K,N] → B is [N,K]
			if (!sgemm_rowmajor_abt((int)m, (int)n, (int)r,
			                         1.0f,
			                         state.gPred.data(), (int)r,
			                         state.U.data(), (int)r,
			                         1.0f,
			                         d_W, (int)n))
				return false;
		}
		else
		{
			// Left subspace: W[m,n] += U[m,r] * gPred[r,n]
			if (!sgemm_rowmajor((int)m, (int)n, (int)r,
			                     1.0f,
			                     state.U.data(), (int)r,
			                     state.gPred.data(), (int)n,
			                     1.0f,
			                     d_W, (int)n))
				return false;
		}

		if (enabledComplementRank > 0u && effectiveComplementRank > 0u)
		{
			const float sectorNominalLr = lr * atlas_nonnegative_finite(ac.complementLrScale, 0.0f);
			std::vector<float> h_corrMat;
			sectorRate = build_complement_correction_matrix(h_block, complementRank,
			                                                effectiveComplementRank,
			                                                baselineRate,
			                                                sectorNominalLr,
			                                                eps,
			                                                ac.complementKappaMax,
			                                                bcFactor,
			                                                h_corrMat);
			ATLAS_CUDA_CHECK(cudaMemcpy(state.complementMat.data(), h_corrMat.data(),
			                              h_corrMat.size() * sizeof(float), cudaMemcpyHostToDevice));
			if (isRight)
			{
				if (!sgemm_rowmajor((int)outerDim, (int)complementRank, (int)complementRank,
				                     1.0f,
				                     state.gv.data(), (int)complementRank,
				                     state.complementMat.data(), (int)complementRank,
				                     0.0f,
				                     state.gPredV.data(), (int)complementRank))
					return false;
			}
			else
			{
				if (!sgemm_rowmajor((int)complementRank, (int)outerDim, (int)complementRank,
				                     1.0f,
				                     state.complementMat.data(), (int)complementRank,
				                     state.gv.data(), (int)outerDim,
				                     0.0f,
				                     state.gPredV.data(), (int)outerDim))
					return false;
			}
			if (isRight)
			{
				if (!sgemm_rowmajor_abt((int)outerDim, (int)subDim, (int)complementRank,
				                         1.0f,
				                         state.gPredV.data(), (int)complementRank,
				                         state.V.data(), (int)complementRank,
				                         1.0f,
				                         d_W, (int)subDim))
					return false;
			}
			else
			{
				if (!sgemm_rowmajor((int)subDim, (int)outerDim, (int)complementRank,
				                     1.0f,
				                     state.V.data(), (int)complementRank,
				                     state.gPredV.data(), (int)outerDim,
				                     1.0f,
			                     d_W, (int)outerDim))
					return false;
			}
		}
		else if (state.gPredV.size() > 0u)
		{
			ATLAS_CUDA_CHECK(cudaMemset(state.gPredV.data(), 0,
			                            state.gPredV.size() * sizeof(float)));
		}

		// Guard against NaN/Inf propagation (matches CPU path)
		if (!atlas_gpu_guard(d_W, mn))
			return false;
	}

	// === Step 9: Compute mu adaptation (synchronous, matches CPU path) ===
	if (state.step > 1ULL && muMax > 0.0f)
	{
		if (!atlas_gpu_mu_norms(state.gz.data(), state.prevGz.data(),
		                         or_, state.d_reduce.data(), state.d_partials.data()))
			return false;
		// Synchronous D2H — consume in the same step (matches CPU path).
		float h_muNorms[2] = {0.0f, 0.0f};
		ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
		ATLAS_CUDA_CHECK(cudaMemcpy(h_muNorms, state.d_reduce.data(),
		                              2 * sizeof(float), cudaMemcpyDeviceToHost));
		float errNormSq = h_muNorms[0];
		float gzNormSq  = h_muNorms[1];
		float gzNorm = sqrtf(gzNormSq);
		if (gzNorm > 1e-12f)
		{
			float ratio = sqrtf(errNormSq) / (gzNorm + 1e-12f);
			float newMu = state.mu * (1.0f - ratio)
			            + muGrowthRate * (muMax - state.mu);
			if (newMu < muMin) newMu = muMin;
			if (newMu > muMax) newMu = muMax;
			state.mu = newMu;
		}
		if (!std::isfinite(state.mu))
		{
			state.mu = muMin;
			if (logger)
			{
				std::ostringstream oss;
				oss << "event=atlas_gpu_mu_recovery step=" << state.step;
				if (tag) oss << " tag=" << tag;
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
		}
	}

	// === Periodic diagnostics (mirrors CPU atlas_optimizer.cpp) ===
	const bool diagStep = logger && tSub > 0u
	    && (state.step % (unsigned long long)tSub) == 0ULL;
	if (diagStep)
	{
		AtlasGpuDiag diag = atlas_gpu_get_diag(state);
		std::ostringstream oss;
		oss << "event=atlas_gpu_step";
		if (tag) oss << " tag=" << tag;
		oss << " step=" << state.step
		    << " m=" << m << " n=" << n << " rank=" << r
		    << " complement_rank=" << enabledComplementRank
		    << " complement_active_rank=" << state.activeComplementRank
		    << " complement_effective_rank=" << effectiveComplementRank
		    << " complement_trial_rank=" << state.trialComplementRank
		    << " complement_trial_wins=" << state.trialComplementWins
		    << " subspace=" << (isRight ? "right" : "left")
		    << " lr=" << lr
		    << " mu=" << state.mu
		    << " total_trace=" << state.totalTrace
		    << " sigma2=" << state.sigma2
		    << " bc_factor=" << bcFactor;
		if (diag.valid)
		{
			oss << " baseline_rate=" << diag.baselineRate
			    << " fisher_min=" << diag.fisherMin
			    << " fisher_max=" << diag.fisherMax
			    << " fisher_mean=" << diag.fisherMean
			    << " fisher_median=" << diag.fisherMedian
			    << " fisher_ratio=" << (diag.fisherMin > 1e-15f ? diag.fisherMax / diag.fisherMin : 0.0f)
			    << " sigma2_fisher_ratio=" << (diag.fisherMean > 1e-15f ? diag.sigma2 / diag.fisherMean : 0.0f)
			    << " active_trace_capture=" << activeTraceCapture
			    << " trace_capture=" << traceCapture
			    << " sector_trace_capture=" << sectorTraceCapture
			    << " sector_fisher=" << sectorFisherDiag
			    << " complement_block_trace=" << h_blockTrace
			    << " complement_tail_mean=" << complementTailMean
			    << " complement_scout_top=" << complementScoutTop
			    << " complement_scout_projected=" << complementScoutProjected
			    << " complement_birth_kelly=" << complementBirthKelly
			    << " complement_trial_kelly=" << complementTrialKelly
			    << " complement_trial_alignment=" << complementTrialAlignment
			    << " complement_trial_contam=" << complementTrialContamination
			    << " complement_trial_return=" << complementTrialReturn
			    << " complement_trial_score=" << complementTrialScore
			    << " sector_rate=" << sectorRate
			    << " closure_gap=" << closureGap
			    << " effective_rank=" << diag.effectiveRank
			    << " spectral_efficiency=" << diag.spectralEfficiency
			    << " top1_concentration=" << diag.top1Concentration
			    << " top10_concentration=" << diag.top10Concentration
			    << " gz_norm=" << diag.gzNorm
			    << " pred_err_norm=" << diag.updateNorm;
		}
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}

	// === Step 9: Store compressed gradient for next step ===
	ATLAS_CUDA_CHECK(cudaMemcpyAsync(state.prevGz.data(), state.gz.data(),
	                                 or_ * sizeof(float), cudaMemcpyDeviceToDevice,
	                                 computeStream()));
	if (enabledComplementRank > 0u)
	{
		ATLAS_CUDA_CHECK(cudaMemcpyAsync(state.prevGv.data(), state.gv.data(),
		                                 compOuter * sizeof(float), cudaMemcpyDeviceToDevice,
		                                 computeStream()));
	}
	else
	{
		ATLAS_CUDA_CHECK(cudaMemsetAsync(state.prevGv.data(), 0,
		                                 compOuter * sizeof(float), computeStream()));
	}

	// === Step 10: Clear accumulated gradients ===
	ATLAS_CUDA_CHECK(cudaMemsetAsync(d_gW, 0, (size_t)mn * sizeof(float), computeStream()));

	return true;
}

bool atlas_gpu_update(GpuAtlasWeightState& state,
                      float* d_W, float* d_gW,
                      unsigned int m, unsigned int n,
                      float invBatch, float lr,
                      float wd1, float wd2, float gradScale,
                      const glades::ATLASConfig& ac,
                      glades::rng::Engine& rng,
                      shmea::GLogger* logger,
                      const char* tag)
{
	if (!state.initialized && m > 0u && n > 0u)
	{
		if (!atlas_gpu_init(state, m, n, ac.rank, ac.muMin, rng))
			return false;
	}
	return atlas_gpu_step(state, d_W, d_gW, m, n, invBatch, lr, wd1, wd2, gradScale,
	                      ac, logger, tag);
}

AtlasGpuDiag atlas_gpu_get_diag(const GpuAtlasWeightState& state)
{
	AtlasGpuDiag d;
	if (!state.initialized)
		return d;

	const unsigned int r = state.r;
	d.sigma2 = state.sigma2;
	d.mu = state.mu;
	d.step = state.step;
	d.baselineRate = state.lastBaselineRate;
	d.rightSubspace = state.rightSubspace;

	if (cudaStreamSynchronize(computeStream()) != cudaSuccess)
		return d;

	// Download Fisher diagonal to compute min/max/mean/median
	if (r > 0u)
	{
		std::vector<float> h_fisher(r);
		cudaError_t err = cudaMemcpy(h_fisher.data(), state.fisherDiag.data(),
		                              r * sizeof(float), cudaMemcpyDeviceToHost);
		if (err == cudaSuccess)
		{
			float fMin = h_fisher[0], fMax = h_fisher[0], fSum = 0.0f;
			for (unsigned int c = 0; c < r; ++c)
			{
				float f = h_fisher[c];
				if (f < fMin) fMin = f;
				if (f > fMax) fMax = f;
				fSum += f;
			}
			d.fisherMin = fMin;
			d.fisherMax = fMax;
			d.fisherMean = fSum / (float)r;

			// Median (partial sort)
			std::vector<float> sorted(h_fisher);
			std::nth_element(sorted.begin(), sorted.begin() + (int)(r/2), sorted.end());
			d.fisherMedian = sorted[r/2];

			// Amplification ratio range (sigma2/fisher gives relative LR multiplier)
			d.corrScaleMin = fMin > 1e-12f ? d.sigma2 / fMin : 0.0f;
			d.corrScaleMax = fMax > 1e-12f ? d.sigma2 / fMax : 0.0f;

			// --- Spectral efficiency: how well the rank is utilized ---
			if (fSum > 1e-30f)
			{
				// Fisher entropy: H = -Σ p_c ln(p_c)
				double entropy = 0.0;
				for (unsigned int c = 0; c < r; ++c)
				{
					double p = (double)h_fisher[c] / (double)fSum;
					if (p > 1e-30)
						entropy -= p * log(p);
				}
				d.effectiveRank = (float)exp(entropy);
				d.spectralEfficiency = d.effectiveRank / (float)r;

				// Top-1 concentration
				d.top1Concentration = fMax / fSum;

				// Top-10 concentration (or top-r if r < 10)
				const unsigned int topK = (r < 10u) ? r : 10u;
				std::nth_element(sorted.begin(), sorted.begin() + (int)(r - topK), sorted.end());
				double topKSum = 0.0;
				for (unsigned int c = r - topK; c < r; ++c)
					topKSum += (double)sorted[c];
				d.top10Concentration = (float)(topKSum / (double)fSum);
			}
		}
	}

	// Download gz norm (from last step's d_reduce[1] = gzNormSq)
	if (state.d_reduce.size() >= 2)
	{
		float h_norms[2] = {0.0f, 0.0f};
		cudaError_t err = cudaMemcpy(h_norms, state.d_reduce.data(),
		                              2 * sizeof(float), cudaMemcpyDeviceToHost);
		if (err == cudaSuccess)
		{
			d.gzNorm = sqrtf(h_norms[1]);
			d.updateNorm = sqrtf(h_norms[0]);
		}
	}

	d.valid = true;
	return d;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
