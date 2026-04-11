// ATLAS optimizer implementation (BRSP variant).
// See atlas_optimizer.h for API documentation.
#include "atlas_optimizer.h"
#include "gemm_helpers.h"
#include "training_config.h"
#include "transformer_kernels.h"
#include "Backend/Database/GLogger.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <sstream>

namespace glades {
namespace atlas {

using glades::transformer_kernels::axpy_f32;

// logfmt helpers (local to this TU)
static void append_kv(std::ostringstream& oss, const char* k, unsigned int v)
{
	oss << ' ' << k << '=' << v;
}
static void append_kv(std::ostringstream& oss, const char* k, unsigned long long v)
{
	oss << ' ' << k << '=' << v;
}
static void append_kv(std::ostringstream& oss, const char* k, float v)
{
	oss << ' ' << k << '=' << v;
}
static void append_kv(std::ostringstream& oss, const char* k, const char* v)
{
	oss << ' ' << k << '=' << (v ? v : "");
}

// NaN/Inf check for internal state protection.
// Uses std::isfinite which is safe under all compiler optimization levels,
// unlike the manual (x == x) && (x - x == 0.0f) pattern which can be
// optimized away under -ffast-math.
static inline bool atlas_isfinite(float x)
{
	return std::isfinite(x);
}

static inline float atlas_sign(float x)
{
	return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

static inline float atlas_bootstrap_or_ema(float prev,
                                           float sample,
                                           float beta,
                                           unsigned long long step)
{
	if (step <= 1ULL)
	{
		if (std::isfinite(prev) && prev != 0.0f)
			return beta * prev + (1.0f - beta) * sample;
		return sample;
	}
	return beta * prev + (1.0f - beta) * sample;
}

static double compute_active_trace(const WeightState& state, unsigned int activeRank)
{
	double activeTrace = 0.0;
	for (unsigned int c = 0; c < activeRank; ++c)
		activeTrace += static_cast<double>(state.fisherDiag[c]);
	return activeTrace;
}

static unsigned int atlas_requested_complement_rank(unsigned int enabledRank,
                                                    unsigned int subDim,
                                                    unsigned int activeRank);
static unsigned int atlas_requested_scout_rank(unsigned int enabledRank,
                                               unsigned int subDim,
                                               unsigned int activeRank,
                                               unsigned int retainedRank);
static void multiply_left_block(float* dst,
                                const float* lhs,
                                const float* rhs,
                                unsigned int rows,
                                unsigned int inner,
                                unsigned int cols);
static void symmetrize_block(float* block, unsigned int dim);
static void jacobi_eigendecompose(const float* symBlock,
                                  unsigned int dim,
                                  std::vector<float>& eigVec,
                                  std::vector<float>& eigVal);
static bool build_inv_sqrt_psd(const float* block,
                               unsigned int dim,
                               float eps,
                               std::vector<float>& invSqrt);
static double symmetric_quadratic_form(const float* block,
                                       unsigned int dim,
                                       const std::vector<float>& vec);
static bool top_generalized_eigenpair(const float* numer,
                                      const float* denom,
                                      unsigned int dim,
                                      float eps,
                                      double* eigValOut,
                                      std::vector<float>& eigVecOut);
static unsigned int atlas_sparrow_mode_rank(unsigned int configured);
static void build_scout_contamination_matrix(const WeightState& state,
                                             unsigned int scoutRank,
                                             unsigned int complementRankUsed,
                                             unsigned int n,
                                             float statScaleSq,
                                             float eps,
                                             std::vector<float>& out);

static const unsigned int kATLASResolveMaxLagHorizon = 4u;
static const unsigned int kATLASHeroMaxLagHorizon = 4u;
static const unsigned int kATLASCobaltMaxLagHorizon = 4u;
static const unsigned int kATLASBirchMaxPastHorizon = 3u;
static const unsigned int kATLASBirchMaxFutureHorizon = 2u;
static const unsigned int kATLASGhostMaxLagHorizon = 4u;
static const unsigned int kATLASQBRTMaxLagHorizon = 4u;
static const unsigned int kATLASQRCMaxLagHorizon = 4u;
static const unsigned int kATLASRiftMaxLagHorizon = 4u;
static const unsigned int kATLASSparrowMaxModeRank = 2u;

static bool atlas_tag_contains(const char* tag, const char* needle)
{
	return tag && needle && std::strstr(tag, needle) != 0;
}

static bool atlas_hidden_fc_eligible(const char* tag,
                                     unsigned int m,
                                     unsigned int n)
{
	// Direct applyStep tests often omit tags; keep the optimizer unrestricted in
	// that path so the numerical unit tests still exercise the complement math.
	if (!tag || !tag[0])
		return true;

	// The current adaptive residual controller is intentionally conservative:
	// enable it only on larger hidden FC-style matrices, not small classifier
	// heads or conv kernels. This matches the benchmark failure mode it targets.
	if (m <= 16u || n <= 16u)
		return false;
	return atlas_tag_contains(tag, "cnn.fc")
	    || atlas_tag_contains(tag, "dff.W");
}

static bool atlas_output_head_eligible(const char* tag,
                                       unsigned int m,
                                       unsigned int n)
{
	// Direct applyStep tests often omit tags; keep ORBIT-Lite reachable there so
	// the controller/math unit tests can exercise the path.
	if (!tag || !tag[0])
		return (m > 1u && n > 0u);

	// ORBIT-Lite is intentionally conservative in the minimal prototype: it is
	// only meant for small output heads where parameter rows correspond closely
	// to logits/classes, not hidden FC blocks or large recurrent gate packs.
	if (m <= 1u || m > 32u || n == 0u)
		return false;
	return atlas_tag_contains(tag, "cnn.fc")
	    || atlas_tag_contains(tag, "dff.W")
	    || atlas_tag_contains(tag, "rnn.Why")
	    || atlas_tag_contains(tag, "gru.Why")
	    || atlas_tag_contains(tag, "lstm.Why");
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
	double generalizedEig;
	double birthKelly;
	double alignment;
	double contamination;
	double uncertainty;
	double quality;

	ResidualScoutQuality()
	    : rawTopEig(0.0), projectedEig(0.0), birthScout(0.0),
	      generalizedEig(0.0), birthKelly(0.0), alignment(1.0), contamination(0.0),
	      uncertainty(0.0), quality(1.0)
	{
	}
};

static void reset_complement_trial(WeightState& state)
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

static bool evaluate_generalized_residual_scout(const WeightState& state,
                                                unsigned int activeRank,
                                                unsigned int retainedCount,
                                                unsigned int informativeRank,
                                                float statScaleSq,
                                                double tailMean,
                                                float eps,
                                                ResidualScoutQuality& quality)
{
	const unsigned int scoutRank =
	    atlas_requested_scout_rank(state.complementRank, state.m, activeRank, informativeRank);
	if (scoutRank == 0u
	    || state.scoutCov.size() < static_cast<size_t>(state.complementRank) * state.complementRank
	    || !(tailMean > static_cast<double>(eps)))
		return false;

	std::vector<float> scoutCovTop(static_cast<size_t>(scoutRank) * scoutRank, 0.0f);
	std::vector<float> scoutNoiseTop(static_cast<size_t>(scoutRank) * scoutRank, 0.0f);
	for (unsigned int i = 0; i < scoutRank; ++i)
	{
		for (unsigned int j = 0; j < scoutRank; ++j)
		{
			scoutCovTop[static_cast<size_t>(i) * scoutRank + j] =
			    state.scoutCov[static_cast<size_t>(i) * state.complementRank + j];
			scoutNoiseTop[static_cast<size_t>(i) * scoutRank + j] =
			    state.scoutNoise[static_cast<size_t>(i) * state.complementRank + j];
		}
	}
	symmetrize_block(&scoutCovTop[0], scoutRank);
	symmetrize_block(&scoutNoiseTop[0], scoutRank);

	std::vector<float> contamTop;
	build_scout_contamination_matrix(state, scoutRank, informativeRank, state.n,
	                                 statScaleSq, eps, contamTop);

	const double noiseWeight = 0.25;
	const double contamWeight = 1.0;
	std::vector<float> denom(static_cast<size_t>(scoutRank) * scoutRank, 0.0f);
	for (unsigned int i = 0; i < scoutRank; ++i)
	{
		for (unsigned int j = 0; j < scoutRank; ++j)
		{
			double v = noiseWeight * static_cast<double>(scoutNoiseTop[static_cast<size_t>(i) * scoutRank + j])
			         + contamWeight * static_cast<double>(contamTop[static_cast<size_t>(i) * scoutRank + j]);
			if (i == j)
				v += tailMean;
			denom[static_cast<size_t>(i) * scoutRank + j] = static_cast<float>(v);
		}
	}
	symmetrize_block(&denom[0], scoutRank);

	double generalizedEig = 0.0;
	std::vector<float> generalizedVec;
	if (!top_generalized_eigenpair(&scoutCovTop[0], &denom[0], scoutRank, eps,
	                               &generalizedEig, generalizedVec))
		return false;

	std::vector<float> scoutEigVec;
	std::vector<float> scoutEigVal;
	jacobi_eigendecompose(&scoutCovTop[0], scoutRank, scoutEigVec, scoutEigVal);
	const double topRawEig =
	    (!scoutEigVal.empty() && std::isfinite(scoutEigVal[0]) && scoutEigVal[0] > 0.0f)
	        ? static_cast<double>(scoutEigVal[0])
	        : static_cast<double>(eps);
	const double rawTopEig =
	    symmetric_quadratic_form(&scoutCovTop[0], scoutRank, generalizedVec);
	const double denomEnergy =
	    std::max<double>(static_cast<double>(eps),
	                     symmetric_quadratic_form(&denom[0], scoutRank, generalizedVec));
	const double noiseEnergy =
	    noiseWeight * symmetric_quadratic_form(&scoutNoiseTop[0], scoutRank, generalizedVec);
	const double contamEnergy =
	    contamWeight * symmetric_quadratic_form(&contamTop[0], scoutRank, generalizedVec);

	quality.rawTopEig = rawTopEig;
	quality.alignment = atlas_clamp_unit(rawTopEig / topRawEig);
	quality.projectedEig = rawTopEig * quality.alignment;
	quality.birthScout = quality.projectedEig;
	quality.generalizedEig = generalizedEig;
	quality.birthKelly = (generalizedEig > 1.0)
	    ? atlas_clamp_unit((generalizedEig - 1.0) / generalizedEig)
	    : 0.0;
	quality.uncertainty = atlas_clamp_unit(noiseEnergy / denomEnergy);
	quality.contamination = atlas_clamp_unit(contamEnergy / denomEnergy);
	quality.quality = quality.alignment * (1.0 - quality.contamination);
	if (!std::isfinite(quality.quality) || quality.quality < 0.0)
		quality.quality = 0.0;
	return true;
}

static ResidualScoutQuality evaluate_residual_scout(const WeightState& state,
                                                    const std::vector<float>& eigVal,
                                                    const std::vector<float>& eigVec,
                                                    const std::vector<float>* scoutEigVal,
                                                    const std::vector<float>* scoutEigVec,
                                                    unsigned int nextMode,
                                                    unsigned int retainedCount,
                                                    unsigned int informativeRank,
                                                    unsigned int activeRank,
                                                    bool useGeneralizedScout,
                                                    float statScaleSq,
                                                    double tailMean,
                                                    float eps)
{
	ResidualScoutQuality quality;
	if (useGeneralizedScout
	    && evaluate_generalized_residual_scout(state, activeRank, retainedCount,
	                                        informativeRank, statScaleSq,
	                                        tailMean, eps, quality))
		return quality;
	if (nextMode >= eigVal.size())
		return quality;

	double nextEmaEig = static_cast<double>(eigVal[nextMode]);
	if (!std::isfinite(nextEmaEig) || nextEmaEig < 0.0)
		nextEmaEig = 0.0;
	quality.rawTopEig = nextEmaEig;
	quality.projectedEig = nextEmaEig;
	quality.birthScout = nextEmaEig;
	quality.generalizedEig = (tailMean > static_cast<double>(eps))
	    ? (nextEmaEig / tailMean)
	    : 0.0;

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

static unsigned int choose_active_complement_rank(WeightState& state,
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
                                                  bool useGeneralizedScout,
                                                  float statScaleSq,
                                                  float eps,
                                                  double* scoutLambdaOut = 0,
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
	if (scoutLambdaOut)
		*scoutLambdaOut = 0.0;
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
		    evaluate_residual_scout(state, eigVal, eigVec, scoutEigVal, scoutEigVec,
		                            prevRank, prevRank, informativeRank, activeRank,
		                            useGeneralizedScout,
		                            statScaleSq,
		                            tailMean, eps);
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
		if (scoutLambdaOut)
			*scoutLambdaOut = scoutQuality.generalizedEig;
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

static unsigned int atlas_requested_scout_rank(unsigned int enabledRank,
                                               unsigned int subDim,
                                               unsigned int activeRank,
                                               unsigned int retainedRank)
{
	if (enabledRank == 0u)
		return 0u;
	if (subDim <= activeRank + retainedRank)
		return 0u;
	const unsigned int available = subDim - activeRank - retainedRank;
	return (enabledRank < available) ? enabledRank : available;
}

static void project_out_active_basis(float* v,
                                     const std::vector<float>& U,
                                     unsigned int fullRank,
                                     unsigned int activeRank,
                                     unsigned int m)
{
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		double dot = 0.0;
		for (unsigned int i = 0; i < m; ++i)
			dot += static_cast<double>(v[i]) * static_cast<double>(U[static_cast<size_t>(i) * fullRank + c]);
		const float dotf = static_cast<float>(dot);
		for (unsigned int i = 0; i < m; ++i)
			v[i] -= dotf * U[static_cast<size_t>(i) * fullRank + c];
	}
}

static void project_out_basis_block(float* v,
                                    const float* basis,
                                    unsigned int stride,
                                    unsigned int rank,
                                    unsigned int m)
{
	for (unsigned int c = 0; c < rank; ++c)
	{
		double dot = 0.0;
		for (unsigned int i = 0; i < m; ++i)
			dot += static_cast<double>(v[i]) * static_cast<double>(basis[static_cast<size_t>(i) * stride + c]);
		const float dotf = static_cast<float>(dot);
		for (unsigned int i = 0; i < m; ++i)
			v[i] -= dotf * basis[static_cast<size_t>(i) * stride + c];
	}
}

static double vector_norm_sq(const float* v, unsigned int m)
{
	double normSq = 0.0;
	for (unsigned int i = 0; i < m; ++i)
	{
		const double x = static_cast<double>(v[i]);
		normSq += x * x;
	}
	return normSq;
}

static bool normalize_vector(float* v, unsigned int m)
{
	const double normSq = vector_norm_sq(v, m);
	if (normSq <= 1e-12)
		return false;
	const float invNorm = static_cast<float>(1.0 / sqrt(normSq));
	for (unsigned int i = 0; i < m; ++i)
		v[i] *= invNorm;
	return true;
}

static bool build_coordinate_complement_seed(float* v,
                                             const std::vector<float>& U,
                                             unsigned int fullRank,
                                             unsigned int activeRank,
                                             const float* extraBasis,
                                             unsigned int extraStride,
                                             unsigned int extraRank,
                                             unsigned int m)
{
	if (m == 0u)
		return false;
	for (unsigned int basisRow = 0; basisRow < m; ++basisRow)
	{
		for (unsigned int i = 0; i < m; ++i)
			v[i] = 0.0f;
		v[basisRow] = 1.0f;
		project_out_active_basis(v, U, fullRank, activeRank, m);
		if (extraBasis && extraRank > 0u)
			project_out_basis_block(v, extraBasis, extraStride, extraRank, m);
		if (normalize_vector(v, m))
			return true;
	}
	return false;
}

static bool build_coordinate_scout_seed(float* v,
                                        const std::vector<float>& U,
                                        unsigned int fullRank,
                                        unsigned int activeRank,
                                        const std::vector<float>& V,
                                        unsigned int vStride,
                                        unsigned int vRank,
                                        const std::vector<float>& W,
                                        unsigned int wStride,
                                        unsigned int wRank,
                                        unsigned int m)
{
	if (m == 0u)
		return false;
	for (unsigned int basisRow = 0; basisRow < m; ++basisRow)
	{
		for (unsigned int i = 0; i < m; ++i)
			v[i] = 0.0f;
		v[basisRow] = 1.0f;
		project_out_active_basis(v, U, fullRank, activeRank, m);
		if (vRank > 0u)
			project_out_basis_block(v, &V[0], vStride, vRank, m);
		if (wRank > 0u)
			project_out_basis_block(v, &W[0], wStride, wRank, m);
		if (normalize_vector(v, m))
			return true;
	}
	return false;
}

static void zero_complement_columns(WeightState& state,
                                    unsigned int beginCol)
{
	for (unsigned int c = beginCol; c < state.complementRank; ++c)
	{
		for (unsigned int i = 0; i < state.m; ++i)
			state.V[static_cast<size_t>(i) * state.complementRank + c] = 0.0f;
		for (unsigned int j = 0; j < state.n; ++j)
			state.prevGv[static_cast<size_t>(c) * state.n + j] = 0.0f;
		for (unsigned int k = 0; k < state.complementRank; ++k)
		{
			state.complementBlock[static_cast<size_t>(c) * state.complementRank + k] = 0.0f;
			state.complementBlock[static_cast<size_t>(k) * state.complementRank + c] = 0.0f;
		}
	}
}

static void zero_scout_columns(WeightState& state,
                               unsigned int beginCol)
{
	for (unsigned int c = beginCol; c < state.complementRank; ++c)
	{
		for (unsigned int i = 0; i < state.m; ++i)
			state.scoutBasis[static_cast<size_t>(i) * state.complementRank + c] = 0.0f;
		for (unsigned int k = 0; k < state.complementRank; ++k)
		{
			state.scoutCov[static_cast<size_t>(c) * state.complementRank + k] = 0.0f;
			state.scoutCov[static_cast<size_t>(k) * state.complementRank + c] = 0.0f;
			state.scoutNoise[static_cast<size_t>(c) * state.complementRank + k] = 0.0f;
			state.scoutNoise[static_cast<size_t>(k) * state.complementRank + c] = 0.0f;
		}
	}
}

static unsigned int orthonormalize_complement_block(WeightState& state,
                                                    unsigned int activeRank,
                                                    unsigned int targetRank,
                                                    const float* fallback,
                                                    shmea::GLogger* logger)
{
	if (state.complementRank == 0u || targetRank == 0u)
	{
		if (state.complementRank > 0u)
			zero_complement_columns(state, 0u);
		return 0u;
	}

	std::vector<float> col(static_cast<size_t>(state.m), 0.0f);
	unsigned int informativeRank = 0u;
	for (unsigned int c = 0; c < targetRank; ++c)
	{
		for (unsigned int i = 0; i < state.m; ++i)
			col[i] = state.V[static_cast<size_t>(i) * state.complementRank + c];

		project_out_active_basis(&col[0], state.U, state.r, activeRank, state.m);
		if (c > 0u)
			project_out_basis_block(&col[0], &state.V[0], state.complementRank, c, state.m);

		if (!normalize_vector(&col[0], state.m) && fallback)
		{
			for (unsigned int i = 0; i < state.m; ++i)
				col[i] = fallback[static_cast<size_t>(i) * state.complementRank + c];
			project_out_active_basis(&col[0], state.U, state.r, activeRank, state.m);
			if (c > 0u)
				project_out_basis_block(&col[0], &state.V[0], state.complementRank, c, state.m);
			normalize_vector(&col[0], state.m);
		}
		if (!normalize_vector(&col[0], state.m)
		    && !build_coordinate_complement_seed(&col[0], state.U, state.r, activeRank,
		                                        &state.V[0], state.complementRank, c, state.m))
		{
			for (unsigned int i = 0; i < state.m; ++i)
				state.V[static_cast<size_t>(i) * state.complementRank + c] = 0.0f;
			continue;
		}

		for (unsigned int i = 0; i < state.m; ++i)
			state.V[static_cast<size_t>(i) * state.complementRank + c] = col[i];
		informativeRank = c + 1u;
	}

	if (targetRank < state.complementRank)
		zero_complement_columns(state, targetRank);

	if (informativeRank < targetRank && logger)
	{
		std::ostringstream oss;
		oss << "event=atlas_complement_rank_drop";
		append_kv(oss, "m", state.m);
		append_kv(oss, "rank", state.r);
		append_kv(oss, "active_rank", activeRank);
		append_kv(oss, "requested_rank", targetRank);
		append_kv(oss, "informative_rank", informativeRank);
		logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
	}

	return informativeRank;
}

static unsigned int orthonormalize_scout_block(WeightState& state,
                                               unsigned int activeRank,
                                               unsigned int retainedRank,
                                               unsigned int targetRank,
                                               const float* fallback,
                                               shmea::GLogger* logger)
{
	if (state.complementRank == 0u || targetRank == 0u)
	{
		if (state.complementRank > 0u)
			zero_scout_columns(state, 0u);
		return 0u;
	}

	std::vector<float> col(static_cast<size_t>(state.m), 0.0f);
	unsigned int informativeRank = 0u;
	for (unsigned int c = 0; c < targetRank; ++c)
	{
		for (unsigned int i = 0; i < state.m; ++i)
			col[i] = state.scoutBasis[static_cast<size_t>(i) * state.complementRank + c];

		project_out_active_basis(&col[0], state.U, state.r, activeRank, state.m);
		if (retainedRank > 0u)
			project_out_basis_block(&col[0], &state.V[0], state.complementRank, retainedRank, state.m);
		if (c > 0u)
			project_out_basis_block(&col[0], &state.scoutBasis[0], state.complementRank, c, state.m);

		if (!normalize_vector(&col[0], state.m) && fallback)
		{
			for (unsigned int i = 0; i < state.m; ++i)
				col[i] = fallback[static_cast<size_t>(i) * state.complementRank + c];
			project_out_active_basis(&col[0], state.U, state.r, activeRank, state.m);
			if (retainedRank > 0u)
				project_out_basis_block(&col[0], &state.V[0], state.complementRank, retainedRank, state.m);
			if (c > 0u)
				project_out_basis_block(&col[0], &state.scoutBasis[0], state.complementRank, c, state.m);
			normalize_vector(&col[0], state.m);
		}
		if (!normalize_vector(&col[0], state.m)
		    && !build_coordinate_scout_seed(&col[0], state.U, state.r, activeRank,
		                                    state.V, state.complementRank, retainedRank,
		                                    state.scoutBasis, state.complementRank, c,
		                                    state.m))
		{
			for (unsigned int i = 0; i < state.m; ++i)
				state.scoutBasis[static_cast<size_t>(i) * state.complementRank + c] = 0.0f;
			continue;
		}

		for (unsigned int i = 0; i < state.m; ++i)
			state.scoutBasis[static_cast<size_t>(i) * state.complementRank + c] = col[i];
		informativeRank = c + 1u;
	}

	if (targetRank < state.complementRank)
		zero_scout_columns(state, targetRank);

	if (informativeRank < targetRank && logger)
	{
		std::ostringstream oss;
		oss << "event=atlas_scout_rank_drop";
		append_kv(oss, "m", state.m);
		append_kv(oss, "rank", state.r);
		append_kv(oss, "active_rank", activeRank);
		append_kv(oss, "retained_rank", retainedRank);
		append_kv(oss, "requested_rank", targetRank);
		append_kv(oss, "informative_rank", informativeRank);
		logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
	}

	return informativeRank;
}

static void resize_complement_storage(WeightState& state,
                                      unsigned int storageRank,
                                      unsigned int activeRank,
                                      glades::rng::Engine& rng,
                                      shmea::GLogger* logger)
{
	if (storageRank == 0u)
		storageRank = 1u;
	if (state.complementRank == storageRank
	    && state.V.size() == static_cast<size_t>(state.m) * storageRank
	    && state.complementBlock.size() == static_cast<size_t>(storageRank) * storageRank
	    && state.prevGv.size() == static_cast<size_t>(storageRank) * state.n
	    && state.heroGwHistory.size() == static_cast<size_t>(kATLASHeroMaxLagHorizon) * static_cast<size_t>(storageRank) * state.n
	    && state.sparrowPrevScout.size() == static_cast<size_t>(storageRank) * state.n)
		return;

	const unsigned int oldRank = state.complementRank;
	const unsigned int copyRank = (oldRank < storageRank) ? oldRank : storageRank;
	const unsigned int fullPastDimOld = state.r + oldRank;
	std::vector<float> oldV(state.V);
	std::vector<float> oldPrevGv(state.prevGv);
	std::vector<float> oldBlock(state.complementBlock);
	std::vector<float> oldScoutBasis(state.scoutBasis);
	std::vector<float> oldScoutCov(state.scoutCov);
	std::vector<float> oldScoutNoise(state.scoutNoise);
	std::vector<float> oldHeroGwHistory(state.heroGwHistory);
	std::vector<float> oldSparrowPrevScout(state.sparrowPrevScout);
	std::vector<float> oldSparrowPastCov(state.sparrowPastCov);
	std::vector<float> oldSparrowCrossCov(state.sparrowCrossCov);
	std::vector<float> oldSparrowRightMode(state.sparrowRightMode);
	unsigned int oldSparrowModeRank = 0u;
	if (fullPastDimOld > 0u)
		oldSparrowModeRank = static_cast<unsigned int>(oldSparrowRightMode.size() / fullPastDimOld);
	if (oldSparrowModeRank == 0u)
		oldSparrowModeRank = atlas_sparrow_mode_rank(state.sparrowModeRank);
	const unsigned int sparrowModeRank =
	    atlas_sparrow_mode_rank(state.sparrowModeRank);
	const unsigned int sparrowCopyModes =
	    (oldSparrowModeRank < sparrowModeRank)
	        ? oldSparrowModeRank
	        : sparrowModeRank;

	state.complementRank = storageRank;
	if (state.activeComplementRank > storageRank)
		state.activeComplementRank = storageRank;
	if (state.trialComplementRank > storageRank
	    || state.trialComplementRank <= state.activeComplementRank)
		reset_complement_trial(state);
	state.V.assign(static_cast<size_t>(state.m) * storageRank, 0.0f);
	state.complementBlock.assign(static_cast<size_t>(storageRank) * storageRank, 0.0f);
	state.scoutBasis.assign(static_cast<size_t>(state.m) * storageRank, 0.0f);
	state.scoutCov.assign(static_cast<size_t>(storageRank) * storageRank, 0.0f);
	state.scoutNoise.assign(static_cast<size_t>(storageRank) * storageRank, 0.0f);
	state.prevGv.assign(static_cast<size_t>(storageRank) * state.n, 0.0f);
	state.heroGwHistory.assign(static_cast<size_t>(kATLASHeroMaxLagHorizon) * static_cast<size_t>(storageRank) * state.n, 0.0f);
	state.sparrowPrevScout.assign(static_cast<size_t>(storageRank) * state.n, 0.0f);

	for (unsigned int i = 0; i < state.m; ++i)
		for (unsigned int c = 0; c < copyRank; ++c)
			state.V[static_cast<size_t>(i) * storageRank + c] =
			    oldV[static_cast<size_t>(i) * oldRank + c];
	for (unsigned int i = 0; i < state.m; ++i)
		for (unsigned int c = 0; c < copyRank; ++c)
			state.scoutBasis[static_cast<size_t>(i) * storageRank + c] =
			    oldScoutBasis.empty()
			        ? 0.0f
			        : oldScoutBasis[static_cast<size_t>(i) * oldRank + c];
	for (unsigned int c = 0; c < copyRank; ++c)
		for (unsigned int j = 0; j < state.n; ++j)
			state.prevGv[static_cast<size_t>(c) * state.n + j] =
			    oldPrevGv[static_cast<size_t>(c) * state.n + j];
	for (unsigned int lag = 0; lag < kATLASHeroMaxLagHorizon; ++lag)
	{
		for (unsigned int c = 0; c < copyRank; ++c)
		{
			for (unsigned int j = 0; j < state.n; ++j)
			{
				state.heroGwHistory[static_cast<size_t>(lag) * static_cast<size_t>(storageRank) * state.n
				                    + static_cast<size_t>(c) * state.n + j] =
				    oldHeroGwHistory.empty()
				        ? 0.0f
				        : oldHeroGwHistory[static_cast<size_t>(lag) * static_cast<size_t>(oldRank) * state.n
				                           + static_cast<size_t>(c) * state.n + j];
			}
		}
	}
	for (unsigned int c = 0; c < copyRank; ++c)
		for (unsigned int j = 0; j < state.n; ++j)
			state.sparrowPrevScout[static_cast<size_t>(c) * state.n + j] =
			    oldSparrowPrevScout.empty()
			        ? 0.0f
			        : oldSparrowPrevScout[static_cast<size_t>(c) * state.n + j];
	for (unsigned int i = 0; i < copyRank; ++i)
		for (unsigned int j = 0; j < copyRank; ++j)
		{
			state.complementBlock[static_cast<size_t>(i) * storageRank + j] =
			    oldBlock[static_cast<size_t>(i) * oldRank + j];
			state.scoutCov[static_cast<size_t>(i) * storageRank + j] =
			    oldScoutCov.empty()
			        ? 0.0f
			        : oldScoutCov[static_cast<size_t>(i) * oldRank + j];
			state.scoutNoise[static_cast<size_t>(i) * storageRank + j] =
			    oldScoutNoise.empty()
			        ? 0.0f
			        : oldScoutNoise[static_cast<size_t>(i) * oldRank + j];
		}

	const unsigned int fullPastDimNew = state.r + storageRank;
	state.sparrowPastCov.assign(static_cast<size_t>(fullPastDimNew) * fullPastDimNew, 0.0f);
	state.sparrowCrossCov.assign(static_cast<size_t>(state.r) * fullPastDimNew, 0.0f);
	state.sparrowRightMode.assign(static_cast<size_t>(sparrowModeRank) * fullPastDimNew, 0.0f);
	for (unsigned int i = 0; i < state.r; ++i)
	{
		for (unsigned int j = 0; j < state.r; ++j)
		{
			const size_t oldIdx = static_cast<size_t>(i) * fullPastDimOld + j;
			const size_t newIdx = static_cast<size_t>(i) * fullPastDimNew + j;
			state.sparrowPastCov[newIdx] =
			    oldSparrowPastCov.empty() ? 0.0f : oldSparrowPastCov[oldIdx];
			state.sparrowCrossCov[newIdx] =
			    oldSparrowCrossCov.empty() ? 0.0f : oldSparrowCrossCov[oldIdx];
		}
	}
	for (unsigned int i = 0; i < copyRank; ++i)
	{
		for (unsigned int j = 0; j < state.r; ++j)
		{
			const size_t oldIdx0 = static_cast<size_t>(state.r + i) * fullPastDimOld + j;
			const size_t newIdx0 = static_cast<size_t>(state.r + i) * fullPastDimNew + j;
			const size_t oldIdx1 = static_cast<size_t>(j) * fullPastDimOld + (state.r + i);
			const size_t newIdx1 = static_cast<size_t>(j) * fullPastDimNew + (state.r + i);
			const float v0 = oldSparrowPastCov.empty() ? 0.0f : oldSparrowPastCov[oldIdx0];
			const float v1 = oldSparrowPastCov.empty() ? 0.0f : oldSparrowPastCov[oldIdx1];
			state.sparrowPastCov[newIdx0] = v0;
			state.sparrowPastCov[newIdx1] = v1;
		}
		for (unsigned int j = 0; j < copyRank; ++j)
		{
			const size_t oldIdx = static_cast<size_t>(state.r + i) * fullPastDimOld + (state.r + j);
			const size_t newIdx = static_cast<size_t>(state.r + i) * fullPastDimNew + (state.r + j);
			state.sparrowPastCov[newIdx] =
			    oldSparrowPastCov.empty() ? 0.0f : oldSparrowPastCov[oldIdx];
		}
	}
	for (unsigned int i = 0; i < state.r; ++i)
		for (unsigned int j = 0; j < copyRank; ++j)
		{
			const size_t oldIdx = static_cast<size_t>(i) * fullPastDimOld + (state.r + j);
			const size_t newIdx = static_cast<size_t>(i) * fullPastDimNew + (state.r + j);
			state.sparrowCrossCov[newIdx] =
			    oldSparrowCrossCov.empty() ? 0.0f : oldSparrowCrossCov[oldIdx];
		}
	for (unsigned int mode = 0; mode < sparrowModeRank; ++mode)
	{
		const size_t oldBase = static_cast<size_t>(mode) * fullPastDimOld;
		const size_t newBase = static_cast<size_t>(mode) * fullPastDimNew;
		for (unsigned int i = 0; i < state.r; ++i)
		{
			state.sparrowRightMode[newBase + i] =
			    (oldSparrowRightMode.empty() || mode >= sparrowCopyModes)
			        ? 0.0f
			        : oldSparrowRightMode[oldBase + i];
		}
		for (unsigned int i = 0; i < copyRank; ++i)
		{
			state.sparrowRightMode[newBase + state.r + i] =
			    (oldSparrowRightMode.empty() || mode >= sparrowCopyModes)
			        ? 0.0f
			        : oldSparrowRightMode[oldBase + state.r + i];
		}
	}

	for (unsigned int c = copyRank; c < storageRank; ++c)
	{
		for (unsigned int i = 0; i < state.m; ++i)
			state.V[static_cast<size_t>(i) * storageRank + c] =
			    glades::rng::standard_normal(rng);
		for (unsigned int i = 0; i < state.m; ++i)
			state.scoutBasis[static_cast<size_t>(i) * storageRank + c] =
			    glades::rng::standard_normal(rng);
	}
	const unsigned int targetRank =
	    atlas_requested_complement_rank(storageRank, state.m, activeRank);
	orthonormalize_complement_block(state, activeRank, targetRank, 0, logger);
	const unsigned int scoutRank =
	    atlas_requested_scout_rank(storageRank, state.m, activeRank, targetRank);
	orthonormalize_scout_block(state, activeRank, targetRank, scoutRank, 0, logger);

	const size_t rnComp = static_cast<size_t>(storageRank) * state.n;
	const size_t mrComp = static_cast<size_t>(state.m) * storageRank;
	state.scratch_gv.resize(rnComp);
	state.scratch_correctedV.resize(rnComp);
	state.scratch_gwScout.resize(rnComp);
	state.scratch_V_old.resize(mrComp);
	state.scratch_Bv.resize(rnComp);
	state.scratch_Zv.resize(mrComp);
	state.scratch_W_old.resize(mrComp);
	state.scratch_Bw.resize(rnComp);
	state.scratch_Zw.resize(mrComp);
	state.scratch_complementMat.resize(static_cast<size_t>(storageRank) * storageRank);
	state.scratch_complementEigVec.resize(static_cast<size_t>(storageRank) * storageRank);
	state.scratch_complementEigVal.resize(static_cast<size_t>(storageRank));
	state.scratch_scoutMat.resize(static_cast<size_t>(storageRank) * storageRank);
	state.scratch_scoutEigVec.resize(static_cast<size_t>(storageRank) * storageRank);
	state.scratch_scoutEigVal.resize(static_cast<size_t>(storageRank));
	state.scratch_heroGwHistoryOld.resize(static_cast<size_t>(kATLASHeroMaxLagHorizon) * rnComp);
	state.scratch_sparrowScout.resize(rnComp);
}

static void ensure_sparrow_storage(WeightState& state, unsigned int modeRank)
{
	modeRank = atlas_sparrow_mode_rank(modeRank);
	if (state.sparrowModeRank == modeRank
	    && state.sparrowLeftMode.size() == static_cast<size_t>(modeRank) * state.r
	    && state.sparrowRightMode.size() == static_cast<size_t>(modeRank) * (state.r + state.complementRank)
	    && state.sparrowLatent.size() == static_cast<size_t>(modeRank) * state.n)
		return;

	const unsigned int fullPastDim = state.r + state.complementRank;
	std::vector<float> oldLeftMode(state.sparrowLeftMode);
	std::vector<float> oldRightMode(state.sparrowRightMode);
	std::vector<float> oldLatent(state.sparrowLatent);
	unsigned int oldModeRank = 0u;
	if (state.r > 0u && fullPastDim > 0u && state.n > 0u)
	{
		oldModeRank = static_cast<unsigned int>(
		    std::min(oldLeftMode.size() / state.r,
		             std::min(oldRightMode.size() / fullPastDim,
		                      oldLatent.size() / state.n)));
	}
	if (oldModeRank == 0u)
		oldModeRank = atlas_sparrow_mode_rank(state.sparrowModeRank);
	const unsigned int copyModes = (oldModeRank < modeRank) ? oldModeRank : modeRank;

	state.sparrowModeRank = modeRank;
	state.sparrowLeftMode.assign(static_cast<size_t>(modeRank) * state.r, 0.0f);
	state.sparrowRightMode.assign(static_cast<size_t>(modeRank) * fullPastDim, 0.0f);
	state.sparrowLatent.assign(static_cast<size_t>(modeRank) * state.n, 0.0f);

	for (unsigned int mode = 0; mode < copyModes; ++mode)
	{
		if (!oldLeftMode.empty())
		{
			std::copy(oldLeftMode.begin() + static_cast<size_t>(mode) * state.r,
			          oldLeftMode.begin() + static_cast<size_t>(mode + 1u) * state.r,
			          state.sparrowLeftMode.begin() + static_cast<size_t>(mode) * state.r);
		}
		if (!oldRightMode.empty())
		{
			std::copy(oldRightMode.begin() + static_cast<size_t>(mode) * fullPastDim,
			          oldRightMode.begin() + static_cast<size_t>(mode + 1u) * fullPastDim,
			          state.sparrowRightMode.begin() + static_cast<size_t>(mode) * fullPastDim);
		}
		if (!oldLatent.empty())
		{
			std::copy(oldLatent.begin() + static_cast<size_t>(mode) * state.n,
			          oldLatent.begin() + static_cast<size_t>(mode + 1u) * state.n,
			          state.sparrowLatent.begin() + static_cast<size_t>(mode) * state.n);
		}
	}
}

static void ensure_complement_basis(WeightState& state,
                                    unsigned int activeRank,
                                    glades::rng::Engine& rng,
                                    shmea::GLogger* logger)
{
	const unsigned int storageRank =
	    (state.complementRank > 0u) ? state.complementRank : 1u;
	resize_complement_storage(state, storageRank, activeRank, rng, logger);
	const unsigned int targetRank =
	    atlas_requested_complement_rank(storageRank, state.m, activeRank);

	for (unsigned int c = 0; c < targetRank; ++c)
	{
		double normSq = 0.0;
		for (unsigned int i = 0; i < state.m; ++i)
		{
			const double v =
			    static_cast<double>(state.V[static_cast<size_t>(i) * storageRank + c]);
			normSq += v * v;
		}
		if (normSq <= 1e-12)
		{
			for (unsigned int i = 0; i < state.m; ++i)
				state.V[static_cast<size_t>(i) * storageRank + c] =
				    glades::rng::standard_normal(rng);
		}
	}

	if (orthonormalize_complement_block(state, activeRank, targetRank, 0, logger) == 0u
	    && targetRank > 0u && logger)
	{
		std::ostringstream oss;
		oss << "event=atlas_complement_seed_failure";
		append_kv(oss, "m", state.m);
		append_kv(oss, "rank", state.r);
		append_kv(oss, "active_rank", activeRank);
		append_kv(oss, "complement_rank", storageRank);
		logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
	}
}

static void ensure_scout_basis(WeightState& state,
                               unsigned int activeRank,
                               unsigned int retainedRank,
                               glades::rng::Engine& rng,
                               shmea::GLogger* logger)
{
	const unsigned int storageRank =
	    (state.complementRank > 0u) ? state.complementRank : 1u;
	const unsigned int targetRank =
	    atlas_requested_scout_rank(storageRank, state.m, activeRank, retainedRank);
	if (targetRank == 0u)
	{
		zero_scout_columns(state, 0u);
		return;
	}

	for (unsigned int c = 0; c < targetRank; ++c)
	{
		double normSq = 0.0;
		for (unsigned int i = 0; i < state.m; ++i)
		{
			const double v =
			    static_cast<double>(state.scoutBasis[static_cast<size_t>(i) * storageRank + c]);
			normSq += v * v;
		}
		if (normSq <= 1e-12)
		{
			for (unsigned int i = 0; i < state.m; ++i)
				state.scoutBasis[static_cast<size_t>(i) * storageRank + c] =
				    glades::rng::standard_normal(rng);
		}
	}

	if (orthonormalize_scout_block(state, activeRank, retainedRank, targetRank, 0, logger) == 0u
	    && targetRank > 0u && logger)
	{
		std::ostringstream oss;
		oss << "event=atlas_scout_seed_failure";
		append_kv(oss, "m", state.m);
		append_kv(oss, "rank", state.r);
		append_kv(oss, "active_rank", activeRank);
		append_kv(oss, "retained_rank", retainedRank);
		append_kv(oss, "scout_rank", targetRank);
		logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
	}
}

static float trace_complement_block(const WeightState& state)
{
	double trace = 0.0;
	for (unsigned int c = 0; c < state.complementRank; ++c)
		trace += static_cast<double>(state.complementBlock[static_cast<size_t>(c) * state.complementRank + c]);
	return static_cast<float>(trace);
}

static unsigned int effective_complement_rank(const WeightState& state,
                                              unsigned int enabledRank,
                                              unsigned int subDim,
                                              unsigned int activeRank)
{
	const unsigned int requested =
	    atlas_requested_complement_rank(enabledRank, subDim, activeRank);
	const unsigned int inspectRank =
	    (requested < state.complementRank) ? requested : state.complementRank;
	unsigned int informative = 0u;
	for (unsigned int c = 0; c < inspectRank; ++c)
	{
		double normSq = 0.0;
		for (unsigned int i = 0; i < subDim; ++i)
		{
			const double v =
			    static_cast<double>(state.V[static_cast<size_t>(i) * state.complementRank + c]);
			normSq += v * v;
		}
		if (normSq <= 1e-12)
			break;
		informative = c + 1u;
	}
	return informative;
}

static float compute_complement_sigma2(const WeightState& state,
                                       unsigned int activeRank,
                                       double activeSectorTrace,
                                       unsigned int effectiveComplementRank,
                                       unsigned int subDim,
                                       float eps,
                                       double* activeTraceOut = 0,
                                       double* sectorTraceOut = 0,
                                       double* closureGapOut = 0)
{
	const double activeTrace = compute_active_trace(state, activeRank);
	const double sectorTrace =
	    (effectiveComplementRank > 0u) ? activeSectorTrace : 0.0;
	const double modeledTrace = activeTrace + sectorTrace;
	const double closureGap = static_cast<double>(state.totalTrace) - modeledTrace;
	const double closedTrace = (closureGap >= 0.0)
	    ? static_cast<double>(state.totalTrace)
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
	if (!atlas_isfinite(rate) || rate < 0.0f)
		rate = 0.0f;
	if (rate > cap)
		rate = cap;
	return rate;
}

static void multiply_left_block(float* dst,
                                const float* lhs,
                                const float* rhs,
                                unsigned int rows,
                                unsigned int inner,
                                unsigned int cols)
{
	for (unsigned int i = 0; i < rows; ++i)
	{
		for (unsigned int j = 0; j < cols; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < inner; ++k)
				sum += static_cast<double>(lhs[static_cast<size_t>(i) * inner + k])
				     * static_cast<double>(rhs[static_cast<size_t>(k) * cols + j]);
			dst[static_cast<size_t>(i) * cols + j] = static_cast<float>(sum);
		}
	}
}

static void multiply_symmetric_transform(float* dst,
                                         const float* overlap,
                                         const float* src,
                                         unsigned int dim,
                                         std::vector<float>& scratch)
{
	scratch.resize(static_cast<size_t>(dim) * dim);
	multiply_left_block(&scratch[0], overlap, src, dim, dim, dim);
	for (unsigned int i = 0; i < dim; ++i)
	{
		for (unsigned int j = 0; j < dim; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < dim; ++k)
				sum += static_cast<double>(scratch[static_cast<size_t>(i) * dim + k])
				     * static_cast<double>(overlap[static_cast<size_t>(j) * dim + k]);
			dst[static_cast<size_t>(i) * dim + j] = static_cast<float>(sum);
		}
	}
}

static void symmetrize_block(float* block, unsigned int dim)
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

static void jacobi_eigendecompose(const float* symBlock,
                                  unsigned int dim,
                                  std::vector<float>& eigVec,
                                  std::vector<float>& eigVal)
{
	eigVec.assign(static_cast<size_t>(dim) * dim, 0.0f);
	eigVal.assign(static_cast<size_t>(dim), 0.0f);
	if (dim == 0u)
		return;

	std::vector<float> a(static_cast<size_t>(dim) * dim, 0.0f);
	for (unsigned int i = 0; i < dim * dim; ++i)
		a[i] = symBlock[i];
	for (unsigned int i = 0; i < dim; ++i)
		eigVec[static_cast<size_t>(i) * dim + i] = 1.0f;

	const unsigned int maxSweeps = 32u + dim * 8u;
	for (unsigned int sweep = 0; sweep < maxSweeps; ++sweep)
	{
		unsigned int p = 0u;
		unsigned int q = 0u;
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
		{
			if (eigVal[j] > eigVal[best])
				best = j;
		}
		if (best == i)
			continue;
		std::swap(eigVal[i], eigVal[best]);
		for (unsigned int k = 0; k < dim; ++k)
			std::swap(eigVec[static_cast<size_t>(k) * dim + i],
			          eigVec[static_cast<size_t>(k) * dim + best]);
	}
}

static void update_complement_block_ema(WeightState& state,
                                        float beta,
                                        unsigned long long step,
                                        float statScaleSq,
                                        unsigned int n)
{
	const unsigned int b = state.complementRank;
	std::vector<float>& gv = state.scratch_gv;
	std::vector<float>& sample = state.scratch_complementMat;
	sample.assign(static_cast<size_t>(b) * b, 0.0f);
	if (n == 0u)
	{
		std::fill(state.complementBlock.begin(), state.complementBlock.end(), 0.0f);
		state.complementFisher = 0.0f;
		return;
	}
	for (unsigned int i = 0; i < b; ++i)
	{
		for (unsigned int j = i; j < b; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(gv[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(gv[static_cast<size_t>(j) * n + col]);
			}
			const float meansq =
			    static_cast<float>(dot / static_cast<double>(n)) * statScaleSq;
			sample[static_cast<size_t>(i) * b + j] = meansq;
			sample[static_cast<size_t>(j) * b + i] = meansq;
		}
	}
	for (unsigned int i = 0; i < b * b; ++i)
		state.complementBlock[i] = atlas_bootstrap_or_ema(state.complementBlock[i], sample[i], beta, step);
	symmetrize_block(&state.complementBlock[0], b);
	state.complementFisher = trace_complement_block(state);
}

static float build_complement_correction_matrix(WeightState& state,
                                                float baselineRate,
                                                float nominalLr,
                                                float eps,
                                                float kappaMax,
                                                float bcFactor,
                                                unsigned int activeComplementRank)
{
	const unsigned int b = state.complementRank;
	std::vector<float>& eigVec = state.scratch_complementEigVec;
	std::vector<float>& eigVal = state.scratch_complementEigVal;
	std::vector<float>& corrMat = state.scratch_complementMat;

	if (b == 0u)
		return 0.0f;

	std::vector<float> scaledBlock(state.complementBlock);
	for (unsigned int i = 0; i < b * b; ++i)
		scaledBlock[i] *= bcFactor;
	symmetrize_block(&scaledBlock[0], b);
	jacobi_eigendecompose(&scaledBlock[0], b, eigVec, eigVal);

	corrMat.assign(static_cast<size_t>(b) * b, 0.0f);
	float maxRate = 0.0f;
	for (unsigned int mode = 0; mode < b; ++mode)
	{
		if (mode >= activeComplementRank)
			continue;
		float fisher = eigVal[mode];
		if (!atlas_isfinite(fisher) || fisher < 0.0f)
			fisher = 0.0f;
		const float modeRate = atlas_clamped_rate(nominalLr, fisher, eps, kappaMax);
		if (modeRate > maxRate)
			maxRate = modeRate;
		const float scale = baselineRate - modeRate;
		for (unsigned int i = 0; i < b; ++i)
		{
			const float qi = eigVec[static_cast<size_t>(i) * b + mode];
			for (unsigned int j = 0; j < b; ++j)
			{
				corrMat[static_cast<size_t>(i) * b + j] +=
				    scale * qi * eigVec[static_cast<size_t>(j) * b + mode];
			}
		}
	}
	symmetrize_block(&corrMat[0], b);
	return maxRate;
}

static unsigned int atlas_clamp_active_rank(unsigned int activeRank,
                                            unsigned int maxRank,
                                            unsigned int minActiveRank)
{
	if (maxRank == 0u)
		return 0u;
	if (minActiveRank == 0u)
		minActiveRank = 1u;
	if (minActiveRank > maxRank)
		minActiveRank = maxRank;
	if (activeRank < minActiveRank)
		activeRank = minActiveRank;
	if (activeRank > maxRank)
		activeRank = maxRank;
	return activeRank;
}

static void pack_active_basis(const std::vector<float>& U,
                              unsigned int fullRank,
                              unsigned int m,
                              unsigned int activeRank,
                              std::vector<float>& packed)
{
	if (activeRank == 0u)
		return;
	for (unsigned int i = 0; i < m; ++i)
	{
		const float* src = &U[static_cast<size_t>(i) * fullRank];
		float* dst = &packed[static_cast<size_t>(i) * activeRank];
		for (unsigned int c = 0; c < activeRank; ++c)
			dst[c] = src[c];
	}
}

static unsigned int orthonormalize_active(float* Q,
                                          const float* fallbackFull,
                                          unsigned int fallbackStride,
                                          unsigned int m,
                                          unsigned int r,
                                          shmea::GLogger* logger)
{
	unsigned int informativeRank = 0u;
	const double tol = 1e-12;

	for (unsigned int j = 0; j < r; ++j)
	{
		for (unsigned int p = 0; p < j; ++p)
		{
			double dot = 0.0;
			for (unsigned int k = 0; k < m; ++k)
				dot += static_cast<double>(Q[k * r + j]) * static_cast<double>(Q[k * r + p]);
			const float dotf = static_cast<float>(dot);
			for (unsigned int k = 0; k < m; ++k)
				Q[k * r + j] -= dotf * Q[k * r + p];
		}

		double norm = 0.0;
		for (unsigned int k = 0; k < m; ++k)
		{
			const double v = static_cast<double>(Q[k * r + j]);
			norm += v * v;
		}

		const bool informative = (norm > tol);
		if (!informative && fallbackFull)
		{
			for (unsigned int k = 0; k < m; ++k)
				Q[k * r + j] = fallbackFull[k * fallbackStride + j];
			for (unsigned int p = 0; p < j; ++p)
			{
				double dot = 0.0;
				for (unsigned int k = 0; k < m; ++k)
					dot += static_cast<double>(Q[k * r + j]) * static_cast<double>(Q[k * r + p]);
				const float dotf = static_cast<float>(dot);
				for (unsigned int k = 0; k < m; ++k)
					Q[k * r + j] -= dotf * Q[k * r + p];
			}
			norm = 0.0;
			for (unsigned int k = 0; k < m; ++k)
			{
				const double v = static_cast<double>(Q[k * r + j]);
				norm += v * v;
			}
		}

		if (norm <= tol)
		{
			if (logger)
			{
				std::ostringstream oss;
				oss << "event=atlas_gram_schmidt_degenerate";
				append_kv(oss, "col", j);
				append_kv(oss, "m", m);
				append_kv(oss, "r", r);
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
			for (unsigned int k = 0; k < m; ++k)
				Q[k * r + j] = 0.0f;
			if (m > 0u)
			{
				const unsigned int basisRow = (j < m) ? j : (m - 1u);
				Q[basisRow * r + j] = 1.0f;
			}
			for (unsigned int p = 0; p < j; ++p)
			{
				double dot = 0.0;
				for (unsigned int k = 0; k < m; ++k)
					dot += static_cast<double>(Q[k * r + j]) * static_cast<double>(Q[k * r + p]);
				const float dotf = static_cast<float>(dot);
				for (unsigned int k = 0; k < m; ++k)
					Q[k * r + j] -= dotf * Q[k * r + p];
			}
			norm = 0.0;
			for (unsigned int k = 0; k < m; ++k)
			{
				const double v = static_cast<double>(Q[k * r + j]);
				norm += v * v;
			}
		}

		if (norm > tol)
		{
			const float inv = static_cast<float>(1.0 / sqrt(norm));
			for (unsigned int k = 0; k < m; ++k)
				Q[k * r + j] *= inv;
		}
		else
		{
			for (unsigned int k = 0; k < m; ++k)
				Q[k * r + j] = 0.0f;
		}

		if (informative)
			informativeRank = j + 1u;
	}

	return (informativeRank > 0u) ? informativeRank : 1u;
}

static void repair_inactive_basis(std::vector<float>& U,
                                  const float* fallbackFull,
                                  unsigned int stride,
                                  unsigned int m,
                                  unsigned int activeRank,
                                  unsigned int fullRank,
                                  shmea::GLogger* logger)
{
	if (activeRank >= fullRank)
		return;

	const double tol = 1e-12;
	for (unsigned int j = activeRank; j < fullRank; ++j)
	{
		for (unsigned int i = 0; i < m; ++i)
		{
			const float seed = fallbackFull
			    ? fallbackFull[static_cast<size_t>(i) * stride + j]
			    : 0.0f;
			U[static_cast<size_t>(i) * fullRank + j] = seed;
		}

		for (unsigned int p = 0; p < j; ++p)
		{
			double dot = 0.0;
			for (unsigned int k = 0; k < m; ++k)
			{
				dot += static_cast<double>(U[static_cast<size_t>(k) * fullRank + j])
				    * static_cast<double>(U[static_cast<size_t>(k) * fullRank + p]);
			}
			const float dotf = static_cast<float>(dot);
			for (unsigned int k = 0; k < m; ++k)
			{
				U[static_cast<size_t>(k) * fullRank + j] -=
				    dotf * U[static_cast<size_t>(k) * fullRank + p];
			}
		}

		double norm = 0.0;
		for (unsigned int k = 0; k < m; ++k)
		{
			const double v = static_cast<double>(U[static_cast<size_t>(k) * fullRank + j]);
			norm += v * v;
		}

		if (norm <= tol)
		{
			for (unsigned int k = 0; k < m; ++k)
				U[static_cast<size_t>(k) * fullRank + j] = 0.0f;
			if (m > 0u)
			{
				const unsigned int basisRow = (j < m) ? j : (m - 1u);
				U[static_cast<size_t>(basisRow) * fullRank + j] = 1.0f;
			}
			for (unsigned int p = 0; p < j; ++p)
			{
				double dot = 0.0;
				for (unsigned int k = 0; k < m; ++k)
				{
					dot += static_cast<double>(U[static_cast<size_t>(k) * fullRank + j])
					    * static_cast<double>(U[static_cast<size_t>(k) * fullRank + p]);
				}
				const float dotf = static_cast<float>(dot);
				for (unsigned int k = 0; k < m; ++k)
				{
					U[static_cast<size_t>(k) * fullRank + j] -=
					    dotf * U[static_cast<size_t>(k) * fullRank + p];
				}
			}
			norm = 0.0;
			for (unsigned int k = 0; k < m; ++k)
			{
				const double v = static_cast<double>(U[static_cast<size_t>(k) * fullRank + j]);
				norm += v * v;
			}
		}

		if (norm > tol)
		{
			const float inv = static_cast<float>(1.0 / sqrt(norm));
			for (unsigned int k = 0; k < m; ++k)
				U[static_cast<size_t>(k) * fullRank + j] *= inv;
		}
		else
		{
			if (logger)
			{
				std::ostringstream oss;
				oss << "event=atlas_inactive_basis_degenerate";
				append_kv(oss, "col", j);
				append_kv(oss, "m", m);
				append_kv(oss, "r", fullRank);
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
			for (unsigned int k = 0; k < m; ++k)
				U[static_cast<size_t>(k) * fullRank + j] = 0.0f;
		}
	}
}

static void sort_directions_by_fisher(WeightState& state,
                                      unsigned int limit,
                                      std::vector<float>* gzBuffer,
                                      unsigned int nCols)
{
	if (limit <= 1u || limit > state.r)
		return;

	for (unsigned int i = 0; i + 1u < limit; ++i)
	{
		unsigned int best = i;
		float bestVal = state.fisherDiag[i];
		for (unsigned int j = i + 1u; j < limit; ++j)
		{
			if (state.fisherDiag[j] > bestVal)
			{
				bestVal = state.fisherDiag[j];
				best = j;
			}
		}
		if (best == i)
			continue;

		std::swap(state.fisherDiag[i], state.fisherDiag[best]);
		for (unsigned int row = 0; row < state.m; ++row)
			std::swap(state.U[static_cast<size_t>(row) * state.r + i],
			          state.U[static_cast<size_t>(row) * state.r + best]);
		for (unsigned int col = 0; col < state.n; ++col)
			std::swap(state.prevGz[static_cast<size_t>(i) * state.n + col],
			          state.prevGz[static_cast<size_t>(best) * state.n + col]);
		for (unsigned int col = 0; col < state.n; ++col)
			std::swap(state.prevPrevGz[static_cast<size_t>(i) * state.n + col],
			          state.prevPrevGz[static_cast<size_t>(best) * state.n + col]);
		const size_t historyStride =
		    static_cast<size_t>(state.r) * static_cast<size_t>(state.n);
		for (unsigned int lag = 0; lag < kATLASResolveMaxLagHorizon; ++lag)
		{
			const size_t lagOffset = static_cast<size_t>(lag) * historyStride;
			for (unsigned int col = 0; col < state.n; ++col)
				std::swap(state.resolveGzHistory[lagOffset + static_cast<size_t>(i) * state.n + col],
				          state.resolveGzHistory[lagOffset + static_cast<size_t>(best) * state.n + col]);
		}
		if (gzBuffer)
		{
			for (unsigned int col = 0; col < nCols; ++col)
				std::swap((*gzBuffer)[static_cast<size_t>(i) * nCols + col],
				          (*gzBuffer)[static_cast<size_t>(best) * nCols + col]);
		}
	}
}

static unsigned int choose_active_rank(const WeightState& state,
                                       unsigned int limit,
                                       float capture,
                                       unsigned int minActiveRank)
{
	if (limit == 0u)
		return 0u;

	limit = atlas_clamp_active_rank(limit, limit, minActiveRank);
	if (capture <= 0.0f)
		return atlas_clamp_active_rank(1u, limit, minActiveRank);
	if (capture >= 1.0f)
		capture = 1.0f;

	double total = 0.0;
	for (unsigned int i = 0; i < limit; ++i)
		total += static_cast<double>(state.fisherDiag[i]);
	if (total <= 1e-30)
		return atlas_clamp_active_rank(1u, limit, minActiveRank);

	double accum = 0.0;
	const double target = static_cast<double>(capture) * total;
	for (unsigned int i = 0; i < limit; ++i)
	{
		accum += static_cast<double>(state.fisherDiag[i]);
		if (accum >= target)
			return atlas_clamp_active_rank(i + 1u, limit, minActiveRank);
	}

	return atlas_clamp_active_rank(limit, limit, minActiveRank);
}

static unsigned int choose_flat_spectrum_rank(unsigned int activeRank,
                                              unsigned int minActiveRank)
{
	if (activeRank == 0u)
		return 0u;
	if (activeRank <= minActiveRank)
		return activeRank;

	// A flat Fisher spectrum means the tracked directions have similar curvature,
	// not that the layer is intrinsically rank-1. Shrink the sketch budget
	// conservatively instead of collapsing to the minimum rank in one step.
	const unsigned int halfRank = (activeRank + 1u) / 2u;
	return atlas_clamp_active_rank(halfRank, activeRank, minActiveRank);
}

static float compute_effective_rank(const WeightState& state, unsigned int limit)
{
	if (limit == 0u)
		return 0.0f;

	double total = 0.0;
	for (unsigned int i = 0; i < limit; ++i)
		total += static_cast<double>(state.fisherDiag[i]);
	if (total <= 1e-30)
		return 0.0f;

	double entropy = 0.0;
	for (unsigned int i = 0; i < limit; ++i)
	{
		const double p = static_cast<double>(state.fisherDiag[i]) / total;
		if (p > 1e-30)
			entropy -= p * log(p);
	}
	return static_cast<float>(exp(entropy));
}

static float compute_topk_concentration(const WeightState& state,
                                        unsigned int limit,
                                        unsigned int k)
{
	if (limit == 0u || k == 0u)
		return 0.0f;
	if (k > limit)
		k = limit;

	double total = 0.0;
	double top = 0.0;
	for (unsigned int i = 0; i < limit; ++i)
	{
		const double v = static_cast<double>(state.fisherDiag[i]);
		total += v;
		if (i < k)
			top += v;
	}
	if (total <= 1e-30)
		return 0.0f;
	return static_cast<float>(top / total);
}

void gramSchmidt(float* Q, unsigned int m, unsigned int r,
                 shmea::GLogger* logger)
{
	for (unsigned int j = 0; j < r; ++j)
	{
		// Subtract projections onto previous columns (modified Gram-Schmidt)
		for (unsigned int p = 0; p < j; ++p)
		{
			double dot = 0.0;
			for (unsigned int k = 0; k < m; ++k)
				dot += static_cast<double>(Q[k * r + j]) * static_cast<double>(Q[k * r + p]);
			const float dotf = static_cast<float>(dot);
			for (unsigned int k = 0; k < m; ++k)
				Q[k * r + j] -= dotf * Q[k * r + p];
		}
		// Normalize
		double norm = 0.0;
		for (unsigned int k = 0; k < m; ++k)
		{
			const double v = static_cast<double>(Q[k * r + j]);
			norm += v * v;
		}
		norm = sqrt(norm);
		if (norm > 1e-12)
		{
			const float inv = static_cast<float>(1.0 / norm);
			for (unsigned int k = 0; k < m; ++k)
				Q[k * r + j] *= inv;
		}
		else
		{
			if (logger)
			{
				std::ostringstream oss;
				oss << "event=atlas_gram_schmidt_degenerate";
				append_kv(oss, "col", j);
				append_kv(oss, "m", m);
				append_kv(oss, "r", r);
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
			for (unsigned int k = 0; k < m; ++k)
				Q[k * r + j] = 0.0f;
			if (j < m)
				Q[j * r + j] = 1.0f;
		}
	}
}

void initWeightState(WeightState& state, unsigned int m, unsigned int n,
                     unsigned int rank, float muInit, glades::rng::Engine& rng,
                     shmea::GLogger* logger)
{
	state.m = m;
	state.n = n;
	state.r = rank;
	if (state.r > m) state.r = m;
	if (state.r > n) state.r = n;
	if (state.r == 0u) state.r = 1u;
	state.activeRank = state.r;

	if (state.r > 256u && logger)
	{
		std::ostringstream oss;
		oss << "event=atlas_rank_warning";
		append_kv(oss, "m", m);
		append_kv(oss, "n", n);
		append_kv(oss, "rank", state.r);
		oss << " msg=rank>256_may_use_significant_memory";
		logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
	}

	const unsigned int r = state.r;

	// Initialize U with random Gaussian values, then orthogonalize
	state.U.resize(static_cast<size_t>(m) * static_cast<size_t>(r));
	const float scale = 1.0f / static_cast<float>(sqrt(static_cast<double>(m)));
	for (size_t i = 0; i < state.U.size(); ++i)
		state.U[i] = glades::rng::standard_normal(rng) * scale;

	gramSchmidt(&state.U[0], m, r, logger);

	// Initialize Fisher statistics to zero; the first real gradient bootstraps
	// them to data-dependent values before they are used for preconditioning.
	state.fisherDiag.assign(static_cast<size_t>(r), 0.0f);
	state.complementFisher = 0.0f;
	state.complementRank = atlas_storage_complement_rank(0u);
	state.activeComplementRank = 0u;
	reset_complement_trial(state);
	state.complementBlock.assign(static_cast<size_t>(state.complementRank) * state.complementRank, 0.0f);
	state.scoutBasis.assign(static_cast<size_t>(m) * static_cast<size_t>(state.complementRank), 0.0f);
	state.scoutCov.assign(static_cast<size_t>(state.complementRank) * state.complementRank, 0.0f);
	state.scoutNoise.assign(static_cast<size_t>(state.complementRank) * state.complementRank, 0.0f);

	// Initialize previous compressed gradients to zero
	state.prevGz.assign(static_cast<size_t>(r) * static_cast<size_t>(n), 0.0f);
	state.prevPrevGz.assign(static_cast<size_t>(r) * static_cast<size_t>(n), 0.0f);
	state.prevGv.assign(static_cast<size_t>(state.complementRank) * static_cast<size_t>(n), 0.0f);
	state.resolveGzHistory.assign(static_cast<size_t>(kATLASResolveMaxLagHorizon) * static_cast<size_t>(r) * static_cast<size_t>(n), 0.0f);
	state.heroGwHistory.assign(static_cast<size_t>(kATLASHeroMaxLagHorizon) * static_cast<size_t>(state.complementRank) * static_cast<size_t>(n), 0.0f);
	state.sparrowModeRank = 1u;
	state.sparrowPrevActive.assign(static_cast<size_t>(r) * static_cast<size_t>(n), 0.0f);
	state.sparrowPrevScout.assign(static_cast<size_t>(state.complementRank) * static_cast<size_t>(n), 0.0f);
	state.sparrowFutureCov.assign(static_cast<size_t>(r) * static_cast<size_t>(r), 0.0f);
	state.sparrowPastCov.assign(static_cast<size_t>(r + state.complementRank) * static_cast<size_t>(r + state.complementRank), 0.0f);
	state.sparrowCrossCov.assign(static_cast<size_t>(r) * static_cast<size_t>(r + state.complementRank), 0.0f);
	state.sparrowLeftMode.assign(static_cast<size_t>(state.sparrowModeRank) * static_cast<size_t>(r), 0.0f);
	state.sparrowRightMode.assign(static_cast<size_t>(state.sparrowModeRank) * static_cast<size_t>(r + state.complementRank), 0.0f);
	state.sparrowLatent.assign(static_cast<size_t>(state.sparrowModeRank) * static_cast<size_t>(n), 0.0f);
	state.qbrtLeftMode.assign(static_cast<size_t>(r), 0.0f);
	state.qbrtLatent.assign(static_cast<size_t>(n), 0.0f);
	state.qrcLeftMode.assign(static_cast<size_t>(r), 0.0f);
	state.qrcLatent.assign(static_cast<size_t>(n), 0.0f);
	state.riftLeftMode.assign(static_cast<size_t>(r), 0.0f);
	state.riftLatent.assign(static_cast<size_t>(n), 0.0f);
	state.orbitPrevSignal.assign(static_cast<size_t>(n), 0.0f);
	state.orbitLeftMode.assign(static_cast<size_t>(r), 0.0f);
	state.orbitLatent.assign(static_cast<size_t>(n), 0.0f);
	state.V.assign(static_cast<size_t>(m) * static_cast<size_t>(state.complementRank), 0.0f);
	if (atlas_requested_complement_rank(state.complementRank, m, state.activeRank) > 0u)
		ensure_complement_basis(state, state.activeRank, rng, logger);
	if (atlas_requested_scout_rank(state.complementRank, m, state.activeRank, state.complementRank) > 0u)
		ensure_scout_basis(state, state.activeRank, state.complementRank, rng, logger);

	// Allocate persistent scratch buffers (reused every step, avoids per-step heap churn).
	const size_t mr = static_cast<size_t>(m) * static_cast<size_t>(r);
	const size_t rn = static_cast<size_t>(r) * static_cast<size_t>(n);
	const size_t cr = static_cast<size_t>(state.complementRank);
	const size_t cn = cr * static_cast<size_t>(n);
	const size_t mc = static_cast<size_t>(m) * cr;
	state.scratch_gz.resize(rn);
	state.scratch_corrected.resize(rn);
	state.scratch_gv.resize(cn);
	state.scratch_correctedV.resize(cn);
	state.scratch_gwScout.resize(cn);
	state.scratch_U_old.resize(mr);
	state.scratch_f_old.resize(static_cast<size_t>(r));
	state.scratch_B.resize(rn);
	state.scratch_Z.resize(mr);
	state.scratch_overlap.resize(static_cast<size_t>(r) * static_cast<size_t>(r));
	state.scratch_prevGzOld.resize(rn);
	state.scratch_prevPrevGzOld.resize(rn);
	state.scratch_resolveGzHistoryOld.resize(static_cast<size_t>(kATLASResolveMaxLagHorizon) * rn);
	state.scratch_heroGwHistoryOld.resize(static_cast<size_t>(kATLASHeroMaxLagHorizon) * cn);
	state.scratch_basisPacked.resize(mr);
	state.scratch_V_old.resize(mc);
	state.scratch_Bv.resize(cn);
	state.scratch_Zv.resize(mc);
	state.scratch_W_old.resize(mc);
	state.scratch_Bw.resize(cn);
	state.scratch_Zw.resize(mc);
	state.scratch_complementMat.resize(cr * cr);
	state.scratch_complementEigVec.resize(cr * cr);
	state.scratch_complementEigVal.resize(cr);
	state.scratch_scoutMat.resize(cr * cr);
	state.scratch_scoutEigVec.resize(cr * cr);
	state.scratch_scoutEigVal.resize(cr);
	state.scratch_sparrowActive.resize(rn);
	state.scratch_sparrowScout.resize(cn);
	state.scratch_sparrowPastSignal.resize(static_cast<size_t>(n));
	state.scratch_geodeRhsCol.resize(static_cast<size_t>(m));
	state.scratch_geodeInvDiagCol.resize(static_cast<size_t>(m));
	state.scratch_geodeActiveCurrent.resize(static_cast<size_t>(r));
	state.scratch_geodeActiveDelta.resize(static_cast<size_t>(r));
	state.scratch_geodeSystemMat.resize(static_cast<size_t>(r) * static_cast<size_t>(r));
	state.scratch_geodeRhs.resize(static_cast<size_t>(r));
	state.scratch_geodeSolution.resize(static_cast<size_t>(r));

	state.totalTrace = 0.0f;
	state.sigma2 = 0.0f;
	state.lastPredictiveEdge = 0.0f;
	state.lastMemoryGain = 0.0f;
	state.lastResolveEdge = 0.0f;
	state.lastResolveKernelRho = 0.0f;
	state.lastResolveMemoryGain = 0.0f;
	state.lastHeroEdge = 0.0f;
	state.lastHeroSigma = 0.0f;
	state.lastHeroMemoryGain = 0.0f;
	state.lastCobaltEdge = 0.0f;
	state.lastCobaltSigma = 0.0f;
	state.lastCobaltMemoryGain = 0.0f;
	state.lastBirchEdge = 0.0f;
	state.lastBirchSigma = 0.0f;
	state.lastBirchMemoryGain = 0.0f;
	state.lastGhostEdge = 0.0f;
	state.lastGhostSigma = 0.0f;
	state.lastGhostHorizontalRatio = 1.0f;
	state.lastGhostMemoryGain = 0.0f;
	state.sparrowPoleNumer = 0.0f;
	state.sparrowPoleDenom = 0.0f;
	state.sparrowPole = 0.0f;
	state.lastSparrowEdge = 0.0f;
	state.lastSparrowSigma = 0.0f;
	state.lastSparrowSecondEdge = 0.0f;
	state.lastSparrowSecondSigma = 0.0f;
	state.lastSparrowHorizontalRatio = 1.0f;
	state.lastSparrowMemoryGain = 0.0f;
	state.lastSparrowActiveModes = 0u;
	state.qbrtPoleNumer = 0.0f;
	state.qbrtPoleDenom = 0.0f;
	state.qbrtPole = 0.0f;
	state.lastQbrtEdge = 0.0f;
	state.lastQbrtSigma = 0.0f;
	state.lastQbrtHorizontalRatio = 1.0f;
	state.lastQbrtMemoryGain = 0.0f;
	state.qrcPoleNumer = 0.0f;
	state.qrcPoleDenom = 0.0f;
	state.qrcPole = 0.0f;
	state.lastQrcEdge = 0.0f;
	state.lastQrcSigma = 0.0f;
	state.lastQrcHorizontalRatio = 1.0f;
	state.lastQrcControlGain = 0.0f;
	state.lastQrcMemoryGain = 0.0f;
	state.riftPoleNumer = 0.0f;
	state.riftPoleDenom = 0.0f;
	state.riftPole = 0.0f;
	state.lastRiftEdge = 0.0f;
	state.lastRiftSigma = 0.0f;
	state.lastRiftHorizontalRatio = 1.0f;
	state.lastRiftAreaEnergy = 0.0f;
	state.lastRiftPredR2 = 0.0f;
	state.lastRiftMemoryGain = 0.0f;
	state.orbitPoleNumer = 0.0f;
	state.orbitPoleDenom = 0.0f;
	state.orbitPole = 0.0f;
	state.lastOrbitEdge = 0.0f;
	state.lastOrbitSigma = 0.0f;
	state.lastOrbitHorizontalRatio = 1.0f;
	state.lastOrbitMemoryGain = 0.0f;
	state.mu = muInit;
	state.lastBaselineRate = 0.0f;
	state.step = 0ULL;
	state.initialized = true;

	if (logger)
	{
		std::ostringstream oss;
		oss << "event=atlas_init";
		append_kv(oss, "m", m);
		append_kv(oss, "n", n);
		append_kv(oss, "rank_requested", rank);
		append_kv(oss, "rank_actual", r);
		append_kv(oss, "active_rank", state.activeRank);
		append_kv(oss, "complement_rank", state.complementRank);
		append_kv(oss, "complement_active_rank", state.activeComplementRank);
		append_kv(oss, "mu_init", muInit);
		append_kv(oss, "U_size", static_cast<unsigned long long>(state.U.size()));
		append_kv(oss, "V_size", static_cast<unsigned long long>(state.V.size()));
		append_kv(oss, "W_size", static_cast<unsigned long long>(state.scoutBasis.size()));
		append_kv(oss, "prevGz_size", static_cast<unsigned long long>(state.prevGz.size()));
		append_kv(oss, "prevPrevGz_size", static_cast<unsigned long long>(state.prevPrevGz.size()));
		append_kv(oss, "prevGv_size", static_cast<unsigned long long>(state.prevGv.size()));
		append_kv(oss, "resolveGzHistory_size", static_cast<unsigned long long>(state.resolveGzHistory.size()));
		append_kv(oss, "heroGwHistory_size", static_cast<unsigned long long>(state.heroGwHistory.size()));
		append_kv(oss, "sparrowPrevActive_size", static_cast<unsigned long long>(state.sparrowPrevActive.size()));
		append_kv(oss, "sparrowPrevScout_size", static_cast<unsigned long long>(state.sparrowPrevScout.size()));
		append_kv(oss, "sparrowLatent_size", static_cast<unsigned long long>(state.sparrowLatent.size()));
		append_kv(oss, "qbrtLatent_size", static_cast<unsigned long long>(state.qbrtLatent.size()));
		append_kv(oss, "qrcLatent_size", static_cast<unsigned long long>(state.qrcLatent.size()));
		append_kv(oss, "riftLatent_size", static_cast<unsigned long long>(state.riftLatent.size()));
		append_kv(oss, "orbitPrevSignal_size", static_cast<unsigned long long>(state.orbitPrevSignal.size()));
		append_kv(oss, "orbitLatent_size", static_cast<unsigned long long>(state.orbitLatent.size()));
		const unsigned long long totalBytes =
		    static_cast<unsigned long long>(state.U.size() + state.fisherDiag.size()
		        + state.V.size() + state.scoutBasis.size()
		        + state.scoutCov.size() + state.scoutNoise.size()
		        + state.prevGz.size() + state.prevPrevGz.size() + state.prevGv.size()
		        + state.resolveGzHistory.size() + state.heroGwHistory.size()
		        + state.sparrowPrevActive.size() + state.sparrowPrevScout.size()
		        + state.sparrowFutureCov.size() + state.sparrowPastCov.size()
		        + state.sparrowCrossCov.size() + state.sparrowLeftMode.size()
		        + state.sparrowRightMode.size() + state.sparrowLatent.size()
		        + state.qbrtLeftMode.size() + state.qbrtLatent.size()
		        + state.qrcLeftMode.size() + state.qrcLatent.size()
		        + state.riftLeftMode.size() + state.riftLatent.size()
		        + state.orbitPrevSignal.size() + state.orbitLeftMode.size()
		        + state.orbitLatent.size()
		        + state.scratch_gz.size() + state.scratch_corrected.size()
		        + state.scratch_gv.size() + state.scratch_correctedV.size()
		        + state.scratch_gwScout.size()
		        + state.scratch_U_old.size() + state.scratch_f_old.size()
		        + state.scratch_B.size() + state.scratch_Z.size()
		        + state.scratch_overlap.size() + state.scratch_prevGzOld.size()
		        + state.scratch_prevPrevGzOld.size()
		        + state.scratch_resolveGzHistoryOld.size()
		        + state.scratch_heroGwHistoryOld.size()
		        + state.scratch_V_old.size() + state.scratch_Bv.size()
		        + state.scratch_Zv.size()
		        + state.scratch_W_old.size() + state.scratch_Bw.size()
		        + state.scratch_Zw.size()
		        + state.complementBlock.size()
		        + state.scratch_complementMat.size()
		        + state.scratch_complementEigVec.size()
		        + state.scratch_complementEigVal.size()
		        + state.scratch_scoutMat.size()
		        + state.scratch_scoutEigVec.size()
		        + state.scratch_scoutEigVal.size()
		        + state.scratch_sparrowActive.size()
		        + state.scratch_sparrowScout.size()
		        + state.scratch_sparrowPastSignal.size()
		        + state.scratch_basisPacked.size())
		    * static_cast<unsigned long long>(sizeof(float));
		append_kv(oss, "total_bytes", totalBytes);
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}
}

void initBiMAPWeightState(BiMAPWeightState& state, unsigned int m, unsigned int n)
{
	state.reset();
	state.m = m;
	state.n = n;
	state.rowSecond.assign(static_cast<size_t>(m), 1.0f);
	state.colSecond.assign(static_cast<size_t>(n), 1.0f);
	state.prevMhat.assign(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);
	state.scratchRow.assign(static_cast<size_t>(m), 0.0f);
	state.scratchCol.assign(static_cast<size_t>(n), 0.0f);
	state.rowBasis.clear();
	state.colBasis.clear();
	state.rowEigVal.clear();
	state.colEigVal.clear();
	state.rowRank = 0u;
	state.colRank = 0u;
	state.lastPredictiveTrust = 0.0f;
	state.lastRowAnisotropy = 1.0f;
	state.lastColAnisotropy = 1.0f;
	state.lastRowCapture = 0.0f;
	state.lastColCapture = 0.0f;
	state.step = 0ULL;
	state.initialized = true;
}

bool refreshSubspace(WeightState& state, const float* grad,
                     unsigned int m, unsigned int n,
                     unsigned int powerIters, float betaRefresh,
                     bool fisherWeightedRefresh,
                     glades::rng::Engine& rng,
                     shmea::GLogger* logger)
{
	const unsigned int r = state.r;
	if (r == 0u || m == 0u || n == 0u) return true;

	const unsigned int activeRank = atlas_clamp_active_rank(state.activeRank, r, 1u);
	const size_t mr = static_cast<size_t>(m) * static_cast<size_t>(r);

	// Save old basis and Fisher for EMA blending and Fisher transform.
	std::copy(state.U.begin(), state.U.end(), state.scratch_U_old.begin());
	std::copy(state.fisherDiag.begin(), state.fisherDiag.end(), state.scratch_f_old.begin());
	std::vector<float>& U_old = state.scratch_U_old;
	std::vector<float>& f_old = state.scratch_f_old;

	// Q_active is stored packed as [m, activeRank] so GEMM uses the reduced rank.
	std::vector<float>& Q_active = state.scratch_basisPacked;
	pack_active_basis(state.U, state.r, m, activeRank, Q_active);

	if (fisherWeightedRefresh && activeRank > 0u && !f_old.empty())
	{
		double fisherMean = 0.0;
		for (unsigned int c = 0; c < activeRank; ++c)
			fisherMean += static_cast<double>(f_old[c]);
		fisherMean /= static_cast<double>(activeRank);
		if (fisherMean < 1e-12)
			fisherMean = 1.0;
		for (unsigned int c = 0; c < activeRank; ++c)
		{
			const double ratio = static_cast<double>(f_old[c]) / fisherMean;
			const float scale = static_cast<float>(sqrt(ratio > 1e-12 ? ratio : 1e-12));
			for (unsigned int i = 0; i < m; ++i)
				Q_active[static_cast<size_t>(i) * activeRank + c] *= scale;
		}
	}
	orthonormalize_active(&Q_active[0], &U_old[0], state.r, m, activeRank, logger);

	std::vector<float>& B = state.scratch_B;
	std::vector<float>& Z = state.scratch_Z;
	for (unsigned int p = 0; p < powerIters; ++p)
	{
		glades::gemm::atb(&B[0], &Q_active[0], grad, activeRank, m, n, 1.0f);
		glades::gemm::abt(&Z[0], grad, &B[0], m, n, activeRank, 1.0f);
		std::copy(Z.begin(), Z.begin() + static_cast<size_t>(m) * activeRank, Q_active.begin());
		orthonormalize_active(&Q_active[0], &U_old[0], state.r, m, activeRank, logger);
	}

	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int c = 0; c < activeRank; ++c)
		{
			Q_active[static_cast<size_t>(i) * activeRank + c] =
			    (1.0f - betaRefresh) * U_old[static_cast<size_t>(i) * state.r + c]
			  + betaRefresh * Q_active[static_cast<size_t>(i) * activeRank + c];
		}
	}
	const unsigned int informativeRank =
	    orthonormalize_active(&Q_active[0], &U_old[0], state.r, m, activeRank, logger);

	// Copy packed basis back into the leading active columns.
	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int c = 0; c < activeRank; ++c)
			state.U[static_cast<size_t>(i) * state.r + c] = Q_active[static_cast<size_t>(i) * activeRank + c];
	}
	repair_inactive_basis(state.U, &U_old[0], state.r, m, activeRank, state.r, logger);

	// overlap = U_new_active^T * U_old_active
	pack_active_basis(U_old, state.r, m, activeRank, Z);
	std::vector<float>& overlap = state.scratch_overlap;
	glades::gemm::atb(&overlap[0], &Q_active[0], &Z[0], activeRank, m, activeRank, 1.0f);

	for (unsigned int c = 0; c < activeRank; ++c)
	{
		double fNew = 0.0;
		for (unsigned int j = 0; j < activeRank; ++j)
		{
			const double o = static_cast<double>(overlap[c * activeRank + j]);
			fNew += o * o * static_cast<double>(f_old[j]);
		}
		if (fNew < 1e-12) fNew = 1e-12;
		state.fisherDiag[c] = static_cast<float>(fNew);
	}

	std::vector<float>& prevGzOld = state.scratch_prevGzOld;
	std::copy(state.prevGz.begin(), state.prevGz.end(), prevGzOld.begin());
	std::fill(state.prevGz.begin(), state.prevGz.end(), 0.0f);
	std::vector<float>& prevPrevGzOld = state.scratch_prevPrevGzOld;
	std::copy(state.prevPrevGz.begin(), state.prevPrevGz.end(), prevPrevGzOld.begin());
	std::fill(state.prevPrevGz.begin(), state.prevPrevGz.end(), 0.0f);
	std::vector<float>& resolveGzHistoryOld = state.scratch_resolveGzHistoryOld;
	std::copy(state.resolveGzHistory.begin(), state.resolveGzHistory.end(), resolveGzHistoryOld.begin());
	std::fill(state.resolveGzHistory.begin(), state.resolveGzHistory.end(), 0.0f);
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		for (unsigned int k = 0; k < activeRank; ++k)
		{
			const float o_ck = overlap[c * activeRank + k];
			if (o_ck == 0.0f) continue;
			axpy_f32(&state.prevGz[c * n], &prevGzOld[k * n], o_ck, n);
			axpy_f32(&state.prevPrevGz[c * n], &prevPrevGzOld[k * n], o_ck, n);
			for (unsigned int lag = 0; lag < kATLASResolveMaxLagHorizon; ++lag)
			{
				const size_t lagOffset = static_cast<size_t>(lag) * static_cast<size_t>(state.r) * static_cast<size_t>(n);
				axpy_f32(&state.resolveGzHistory[lagOffset + static_cast<size_t>(c) * n],
				         &resolveGzHistoryOld[lagOffset + static_cast<size_t>(k) * n],
				         o_ck, n);
			}
		}
	}

	bool refreshOk = true;
	for (unsigned int i = 0; i < m && refreshOk; ++i)
	{
		for (unsigned int c = 0; c < activeRank; ++c)
		{
			if (!atlas_isfinite(state.U[static_cast<size_t>(i) * state.r + c]))
			{
				refreshOk = false;
				break;
			}
		}
	}
	if (refreshOk)
	{
		for (unsigned int c = 0; c < activeRank; ++c)
		{
			if (!atlas_isfinite(state.fisherDiag[c]))
			{
				refreshOk = false;
				break;
			}
		}
	}

	if (!refreshOk && logger)
	{
		std::ostringstream oss;
		oss << "event=atlas_refresh_nonfinite";
		append_kv(oss, "step", state.step);
		append_kv(oss, "m", m);
		append_kv(oss, "n", n);
		append_kv(oss, "rank", r);
		append_kv(oss, "active_rank", activeRank);
		logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
	}

	if (logger && refreshOk)
	{
		float fMin = state.fisherDiag[0];
		float fMax = state.fisherDiag[0];
		double fSum = 0.0;
		for (unsigned int c = 0; c < activeRank; ++c)
		{
			const float f = state.fisherDiag[c];
			if (f < fMin) fMin = f;
			if (f > fMax) fMax = f;
			fSum += static_cast<double>(f);
		}
		double overlapDiagSum = 0.0;
		for (unsigned int c = 0; c < activeRank; ++c)
		{
			const double od = static_cast<double>(overlap[c * activeRank + c]);
			overlapDiagSum += (od > 0.0 ? od : -od);
		}

		std::ostringstream oss;
		oss << "event=atlas_subspace_refresh";
		append_kv(oss, "step", state.step);
		append_kv(oss, "m", m);
		append_kv(oss, "n", n);
		append_kv(oss, "rank", r);
		append_kv(oss, "active_rank", state.activeRank);
		append_kv(oss, "informative_rank", informativeRank);
		append_kv(oss, "power_iters", powerIters);
		append_kv(oss, "beta_refresh", betaRefresh);
		append_kv(oss, "mean_overlap",
		          static_cast<float>(overlapDiagSum / static_cast<double>(activeRank)));
		append_kv(oss, "fisher_min", fMin);
		append_kv(oss, "fisher_max", fMax);
		append_kv(oss, "fisher_mean", static_cast<float>(fSum / static_cast<double>(activeRank)));
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}

	return refreshOk;
}

static bool refreshComplementSector(WeightState& state,
                                    const float* grad,
                                    unsigned int m,
                                    unsigned int n,
                                    unsigned int activeRank,
                                    unsigned int powerIters,
                                    float betaRefresh,
                                    glades::rng::Engine& rng,
                                    shmea::GLogger* logger)
{
	const unsigned int storageRank = state.complementRank;
	const unsigned int targetRank =
	    atlas_requested_complement_rank(storageRank, m, activeRank);
	if (targetRank == 0u)
	{
		std::fill(state.complementBlock.begin(), state.complementBlock.end(), 0.0f);
		state.complementFisher = 0.0f;
		if (!state.prevGv.empty())
			std::fill(state.prevGv.begin(), state.prevGv.end(), 0.0f);
		if (!state.V.empty())
			std::fill(state.V.begin(), state.V.end(), 0.0f);
		return true;
	}

	ensure_complement_basis(state, activeRank, rng, logger);
	std::copy(state.V.begin(), state.V.end(), state.scratch_V_old.begin());
	std::vector<float>& V_old = state.scratch_V_old;
	std::vector<float> oldBlock(state.complementBlock);
	std::vector<float> oldPrevGv(state.prevGv);

	std::vector<float>& Bv = state.scratch_Bv;
	std::vector<float>& Zv = state.scratch_Zv;
	for (unsigned int p = 0; p < powerIters; ++p)
	{
		glades::gemm::atb(&Bv[0], &state.V[0], grad, storageRank, m, n, 1.0f);
		glades::gemm::abt(&Zv[0], grad, &Bv[0], m, n, storageRank, 1.0f);
		std::copy(Zv.begin(), Zv.end(), state.V.begin());
		orthonormalize_complement_block(state, activeRank, targetRank, &V_old[0], logger);
	}

	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int c = 0; c < storageRank; ++c)
		{
			state.V[static_cast<size_t>(i) * storageRank + c] =
			    (1.0f - betaRefresh) * V_old[static_cast<size_t>(i) * storageRank + c]
			  + betaRefresh * state.V[static_cast<size_t>(i) * storageRank + c];
		}
	}
	const unsigned int informativeRank =
	    orthonormalize_complement_block(state, activeRank, targetRank, &V_old[0], logger);

	std::vector<float>& overlap = state.scratch_complementMat;
	overlap.assign(static_cast<size_t>(storageRank) * storageRank, 0.0f);
	for (unsigned int c = 0; c < storageRank; ++c)
	{
		for (unsigned int k = 0; k < storageRank; ++k)
		{
			double dot = 0.0;
			for (unsigned int i = 0; i < m; ++i)
			{
				dot += static_cast<double>(state.V[static_cast<size_t>(i) * storageRank + c])
				     * static_cast<double>(V_old[static_cast<size_t>(i) * storageRank + k]);
			}
			overlap[static_cast<size_t>(c) * storageRank + k] = static_cast<float>(dot);
		}
	}

	multiply_symmetric_transform(&state.complementBlock[0], &overlap[0], &oldBlock[0],
	                             storageRank, state.scratch_complementEigVec);
	symmetrize_block(&state.complementBlock[0], storageRank);
	state.complementFisher = trace_complement_block(state);
	multiply_left_block(&state.prevGv[0], &overlap[0], &oldPrevGv[0], storageRank, storageRank, n);

	if (logger)
	{
		double overlapDiagSum = 0.0;
		for (unsigned int c = 0; c < informativeRank; ++c)
		{
			const double od = static_cast<double>(overlap[static_cast<size_t>(c) * storageRank + c]);
			overlapDiagSum += (od > 0.0 ? od : -od);
		}
		std::ostringstream oss;
		oss << "event=atlas_complement_refresh";
		append_kv(oss, "step", state.step);
		append_kv(oss, "m", m);
		append_kv(oss, "n", n);
		append_kv(oss, "active_rank", activeRank);
		append_kv(oss, "complement_rank", storageRank);
		append_kv(oss, "informative_rank", informativeRank);
		append_kv(oss, "power_iters", powerIters);
		append_kv(oss, "beta_refresh", betaRefresh);
		append_kv(oss, "mean_overlap",
		          (informativeRank > 0u)
		              ? static_cast<float>(overlapDiagSum / static_cast<double>(informativeRank))
		              : 0.0f);
		append_kv(oss, "sector_fisher", state.complementFisher);
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}

	return true;
}

static bool refreshComplementScout(WeightState& state,
                                   const float* grad,
                                   unsigned int m,
                                   unsigned int n,
                                   unsigned int activeRank,
                                   unsigned int retainedRank,
                                   unsigned int powerIters,
                                   float betaRefresh,
                                   glades::rng::Engine& rng,
                                   shmea::GLogger* logger)
{
	const unsigned int storageRank = state.complementRank;
	const unsigned int scoutRank =
	    atlas_requested_scout_rank(storageRank, m, activeRank, retainedRank);
	if (scoutRank == 0u)
	{
		zero_scout_columns(state, 0u);
		return true;
	}

	ensure_scout_basis(state, activeRank, retainedRank, rng, logger);
	std::copy(state.scoutBasis.begin(), state.scoutBasis.end(), state.scratch_W_old.begin());
	std::vector<float>& W_old = state.scratch_W_old;
	std::vector<float> oldScoutCov(state.scoutCov);
	std::vector<float> oldScoutNoise(state.scoutNoise);
	std::vector<float>& heroGwHistoryOld = state.scratch_heroGwHistoryOld;
	if (heroGwHistoryOld.size() >= state.heroGwHistory.size())
		std::copy(state.heroGwHistory.begin(), state.heroGwHistory.end(), heroGwHistoryOld.begin());
	std::fill(state.heroGwHistory.begin(), state.heroGwHistory.end(), 0.0f);

	std::vector<float>& Bw = state.scratch_Bw;
	std::vector<float>& Zw = state.scratch_Zw;
	for (unsigned int p = 0; p < powerIters; ++p)
	{
		glades::gemm::atb(&Bw[0], &state.scoutBasis[0], grad, storageRank, m, n, 1.0f);
		glades::gemm::abt(&Zw[0], grad, &Bw[0], m, n, storageRank, 1.0f);
		std::copy(Zw.begin(), Zw.end(), state.scoutBasis.begin());
		orthonormalize_scout_block(state, activeRank, retainedRank, scoutRank, &W_old[0], logger);
	}

	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int c = 0; c < storageRank; ++c)
		{
			state.scoutBasis[static_cast<size_t>(i) * storageRank + c] =
			    (1.0f - betaRefresh) * W_old[static_cast<size_t>(i) * storageRank + c]
			  + betaRefresh * state.scoutBasis[static_cast<size_t>(i) * storageRank + c];
		}
	}
	const unsigned int informativeRank =
	    orthonormalize_scout_block(state, activeRank, retainedRank, scoutRank, &W_old[0], logger);

	std::vector<float>& overlap = state.scratch_scoutMat;
	overlap.assign(static_cast<size_t>(storageRank) * storageRank, 0.0f);
	for (unsigned int c = 0; c < storageRank; ++c)
	{
		for (unsigned int k = 0; k < storageRank; ++k)
		{
			double dot = 0.0;
			for (unsigned int i = 0; i < m; ++i)
			{
				dot += static_cast<double>(state.scoutBasis[static_cast<size_t>(i) * storageRank + c])
				     * static_cast<double>(W_old[static_cast<size_t>(i) * storageRank + k]);
			}
			overlap[static_cast<size_t>(c) * storageRank + k] = static_cast<float>(dot);
		}
	}

	multiply_symmetric_transform(&state.scoutCov[0], &overlap[0], &oldScoutCov[0],
	                             storageRank, state.scratch_scoutEigVec);
	multiply_symmetric_transform(&state.scoutNoise[0], &overlap[0], &oldScoutNoise[0],
	                             storageRank, state.scratch_complementEigVec);
	symmetrize_block(&state.scoutCov[0], storageRank);
	symmetrize_block(&state.scoutNoise[0], storageRank);
	if (!state.heroGwHistory.empty()
	    && heroGwHistoryOld.size() >= state.heroGwHistory.size()
	    && n > 0u)
	{
		const size_t lagSlice = static_cast<size_t>(storageRank) * static_cast<size_t>(n);
		for (unsigned int lag = 0; lag < kATLASHeroMaxLagHorizon; ++lag)
		{
			const size_t lagOffset = static_cast<size_t>(lag) * lagSlice;
			for (unsigned int c = 0; c < storageRank; ++c)
			{
				for (unsigned int k = 0; k < storageRank; ++k)
				{
					const float o_ck = overlap[static_cast<size_t>(c) * storageRank + k];
					if (o_ck == 0.0f)
						continue;
					axpy_f32(&state.heroGwHistory[lagOffset + static_cast<size_t>(c) * n],
					         &heroGwHistoryOld[lagOffset + static_cast<size_t>(k) * n],
					         o_ck, n);
				}
			}
		}
	}
	if (logger)
	{
		double overlapDiagSum = 0.0;
		for (unsigned int c = 0; c < informativeRank; ++c)
		{
			const double od = static_cast<double>(overlap[static_cast<size_t>(c) * storageRank + c]);
			overlapDiagSum += (od > 0.0 ? od : -od);
		}
		std::ostringstream oss;
		oss << "event=atlas_scout_refresh";
		append_kv(oss, "step", state.step);
		append_kv(oss, "m", m);
		append_kv(oss, "n", n);
		append_kv(oss, "active_rank", activeRank);
		append_kv(oss, "retained_rank", retainedRank);
		append_kv(oss, "scout_rank", storageRank);
		append_kv(oss, "informative_rank", informativeRank);
		append_kv(oss, "power_iters", powerIters);
		append_kv(oss, "beta_refresh", betaRefresh);
		append_kv(oss, "mean_overlap",
		          (informativeRank > 0u)
		              ? static_cast<float>(overlapDiagSum / static_cast<double>(informativeRank))
		              : 0.0f);
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}

	return true;
}

static void update_scout_statistics(WeightState& state,
                                    float beta,
                                    unsigned long long step,
                                    float statScaleSq,
                                    unsigned int n,
                                    unsigned int scoutRank)
{
	const unsigned int b = state.complementRank;
	if (b == 0u || scoutRank == 0u)
	{
		if (!state.scoutCov.empty())
			std::fill(state.scoutCov.begin(), state.scoutCov.end(), 0.0f);
		if (!state.scoutNoise.empty())
			std::fill(state.scoutNoise.begin(), state.scoutNoise.end(), 0.0f);
		return;
	}

	std::vector<float> sample(static_cast<size_t>(b) * b, 0.0f);
	if (n == 0u)
	{
		std::fill(state.scoutCov.begin(), state.scoutCov.end(), 0.0f);
		std::fill(state.scoutNoise.begin(), state.scoutNoise.end(), 0.0f);
		return;
	}

	const std::vector<float>& gw = state.scratch_gwScout;
	for (unsigned int i = 0; i < scoutRank; ++i)
	{
		for (unsigned int j = i; j < scoutRank; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(gw[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(gw[static_cast<size_t>(j) * n + col]);
			}
			const float meansq =
			    static_cast<float>(dot / static_cast<double>(n)) * statScaleSq;
			sample[static_cast<size_t>(i) * b + j] = meansq;
			sample[static_cast<size_t>(j) * b + i] = meansq;
		}
	}

	std::vector<float> prevCov(state.scoutCov);
	for (unsigned int i = 0; i < b * b; ++i)
		state.scoutCov[i] = atlas_bootstrap_or_ema(state.scoutCov[i], sample[i], beta, step);
	symmetrize_block(&state.scoutCov[0], b);

	std::vector<float> diff(static_cast<size_t>(b) * b, 0.0f);
	for (unsigned int i = 0; i < scoutRank; ++i)
		for (unsigned int j = 0; j < scoutRank; ++j)
			diff[static_cast<size_t>(i) * b + j] =
			    sample[static_cast<size_t>(i) * b + j] - prevCov[static_cast<size_t>(i) * b + j];

	std::vector<float> noiseSample(static_cast<size_t>(b) * b, 0.0f);
	for (unsigned int i = 0; i < scoutRank; ++i)
	{
		for (unsigned int j = i; j < scoutRank; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < scoutRank; ++k)
				sum += static_cast<double>(diff[static_cast<size_t>(i) * b + k])
				     * static_cast<double>(diff[static_cast<size_t>(j) * b + k]);
			noiseSample[static_cast<size_t>(i) * b + j] = static_cast<float>(sum);
			noiseSample[static_cast<size_t>(j) * b + i] = static_cast<float>(sum);
		}
	}
	for (unsigned int i = 0; i < b * b; ++i)
		state.scoutNoise[i] = atlas_bootstrap_or_ema(state.scoutNoise[i], noiseSample[i], beta, step);
	symmetrize_block(&state.scoutNoise[0], b);
}

static double prism_row_cosine(const float* a,
                               const float* b,
                               unsigned int n)
{
	double dot = 0.0;
	double normA = 0.0;
	double normB = 0.0;
	for (unsigned int j = 0; j < n; ++j)
	{
		const double av = static_cast<double>(a[j]);
		const double bv = static_cast<double>(b[j]);
		dot += av * bv;
		normA += av * av;
		normB += bv * bv;
	}
	if (!(normA > 1e-18) || !(normB > 1e-18))
		return 0.0;
	double rho = dot / std::sqrt(normA * normB);
	if (!std::isfinite(rho))
		return 0.0;
	if (rho > 1.0)
		rho = 1.0;
	else if (rho < -1.0)
		rho = -1.0;
	return rho;
}

static float compute_prism_predictive_edge(const std::vector<float>& current,
                                           const std::vector<float>& previous,
                                           unsigned int rank,
                                           unsigned int n,
                                           float statScaleSq,
                                           float tailMean,
                                           float eps)
{
	if (rank == 0u || n == 0u
	    || current.size() < static_cast<size_t>(rank) * n
	    || previous.size() < static_cast<size_t>(rank) * n)
		return 0.0f;

	std::vector<float> cross(static_cast<size_t>(rank) * rank, 0.0f);
	for (unsigned int i = 0; i < rank; ++i)
	{
		for (unsigned int j = i; j < rank; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(current[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(previous[static_cast<size_t>(j) * n + col]);
				if (i != j)
					dot += static_cast<double>(current[static_cast<size_t>(j) * n + col])
					     * static_cast<double>(previous[static_cast<size_t>(i) * n + col]);
			}
			const float meansq =
			    static_cast<float>(0.5 * dot / static_cast<double>(n)) * statScaleSq;
			cross[static_cast<size_t>(i) * rank + j] = meansq;
			cross[static_cast<size_t>(j) * rank + i] = meansq;
		}
	}
	symmetrize_block(&cross[0], rank);

	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(&cross[0], rank, eigVec, eigVal);
	double topEig = (!eigVal.empty()) ? static_cast<double>(eigVal[0]) : 0.0;
	if (!std::isfinite(topEig) || topEig < 0.0)
		topEig = 0.0;

	double trace = 0.0;
	for (unsigned int i = 0; i < rank; ++i)
	{
		const double d = static_cast<double>(cross[static_cast<size_t>(i) * rank + i]);
		if (std::isfinite(d) && d > 0.0)
			trace += d;
	}
	const double bulk = std::max<double>(static_cast<double>(eps),
	                                     static_cast<double>(tailMean)
	                                     + trace / static_cast<double>(rank));
	const double edge = topEig / bulk - 1.0;
	return static_cast<float>(std::isfinite(edge) ? edge : 0.0);
}

static unsigned int atlas_resolve_lag_horizon(unsigned int configured)
{
	return (configured > kATLASResolveMaxLagHorizon)
	    ? kATLASResolveMaxLagHorizon
	    : configured;
}

static unsigned int atlas_hero_lag_horizon(unsigned int configured)
{
	return (configured > kATLASHeroMaxLagHorizon)
	    ? kATLASHeroMaxLagHorizon
	    : configured;
}

static unsigned int atlas_cobalt_lag_horizon(unsigned int configured)
{
	return (configured > kATLASCobaltMaxLagHorizon)
	    ? kATLASCobaltMaxLagHorizon
	    : configured;
}

static unsigned int atlas_birch_past_horizon(unsigned int configured)
{
	return (configured > kATLASBirchMaxPastHorizon)
	    ? kATLASBirchMaxPastHorizon
	    : configured;
}

static unsigned int atlas_birch_future_horizon(unsigned int configured)
{
	return (configured > kATLASBirchMaxFutureHorizon)
	    ? kATLASBirchMaxFutureHorizon
	    : configured;
}

static unsigned int atlas_ghost_lag_horizon(unsigned int configured)
{
	return (configured > kATLASGhostMaxLagHorizon)
	    ? kATLASGhostMaxLagHorizon
	    : configured;
}

static unsigned int atlas_qbrt_lag_horizon(unsigned int configured)
{
	return (configured > kATLASQBRTMaxLagHorizon)
	    ? kATLASQBRTMaxLagHorizon
	    : configured;
}

static unsigned int atlas_qrc_lag_horizon(unsigned int configured)
{
	return (configured > kATLASQRCMaxLagHorizon)
	    ? kATLASQRCMaxLagHorizon
	    : configured;
}

static unsigned int atlas_rift_lag_horizon(unsigned int configured)
{
	return (configured > kATLASRiftMaxLagHorizon)
	    ? kATLASRiftMaxLagHorizon
	    : configured;
}

static unsigned int atlas_sparrow_mode_rank(unsigned int configured)
{
	if (configured == 0u)
		return 1u;
	return (configured > kATLASSparrowMaxModeRank)
	    ? kATLASSparrowMaxModeRank
	    : configured;
}

static float remove_vectorized_projection(float* block,
                                          const float* gauge,
                                          size_t count,
                                          float eps)
{
	if (!block || !gauge || count == 0u)
		return 1.0f;

	double blockNormSq = 0.0;
	double gaugeNormSq = 0.0;
	double dot = 0.0;
	for (size_t i = 0; i < count; ++i)
	{
		const double b = static_cast<double>(block[i]);
		const double g = static_cast<double>(gauge[i]);
		blockNormSq += b * b;
		gaugeNormSq += g * g;
		dot += b * g;
	}
	if (!(blockNormSq > static_cast<double>(eps)))
		return 1.0f;
	if (gaugeNormSq > static_cast<double>(eps))
	{
		const double alpha = dot / gaugeNormSq;
		if (std::isfinite(alpha))
		{
			for (size_t i = 0; i < count; ++i)
				block[i] = static_cast<float>(static_cast<double>(block[i])
				                            - alpha * static_cast<double>(gauge[i]));
		}
	}

	double horizNormSq = 0.0;
	for (size_t i = 0; i < count; ++i)
	{
		const double b = static_cast<double>(block[i]);
		horizNormSq += b * b;
	}
	if (!(horizNormSq >= 0.0) || !std::isfinite(horizNormSq))
		return 0.0f;
	return static_cast<float>(std::sqrt(std::max<double>(0.0, horizNormSq / blockNormSq)));
}

static bool top_singular_triplet(const std::vector<float>& matrix,
                                 unsigned int rows,
                                 unsigned int cols,
                                 float* sigmaOut,
                                 std::vector<float>& leftOut,
                                 std::vector<float>& rightOut)
{
	leftOut.assign(static_cast<size_t>(rows), 0.0f);
	rightOut.assign(static_cast<size_t>(cols), 0.0f);
	if (sigmaOut)
		*sigmaOut = 0.0f;
	if (rows == 0u || cols == 0u
	    || matrix.size() < static_cast<size_t>(rows) * static_cast<size_t>(cols))
		return false;

	std::vector<float> gram(static_cast<size_t>(rows) * rows, 0.0f);
	for (unsigned int i = 0; i < rows; ++i)
	{
		for (unsigned int j = i; j < rows; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < cols; ++k)
			{
				sum += static_cast<double>(matrix[static_cast<size_t>(i) * cols + k])
				     * static_cast<double>(matrix[static_cast<size_t>(j) * cols + k]);
			}
			gram[static_cast<size_t>(i) * rows + j] = static_cast<float>(sum);
			gram[static_cast<size_t>(j) * rows + i] = static_cast<float>(sum);
		}
	}
	symmetrize_block(&gram[0], rows);

	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(&gram[0], rows, eigVec, eigVal);
	if (eigVal.empty())
		return false;

	double topEig = static_cast<double>(eigVal[0]);
	if (!std::isfinite(topEig) || topEig < 0.0)
		topEig = 0.0;
	const double sigma = std::sqrt(topEig);
	if (!(sigma > 0.0))
		return false;
	if (sigmaOut)
		*sigmaOut = static_cast<float>(sigma);

	for (unsigned int i = 0; i < rows; ++i)
		leftOut[i] = eigVec[static_cast<size_t>(i) * rows + 0u];
	if (!normalize_vector(&leftOut[0], rows))
		return false;

	for (unsigned int j = 0; j < cols; ++j)
	{
		double sum = 0.0;
		for (unsigned int i = 0; i < rows; ++i)
		{
			sum += static_cast<double>(matrix[static_cast<size_t>(i) * cols + j])
			     * static_cast<double>(leftOut[i]);
		}
		rightOut[j] = static_cast<float>(sum / sigma);
	}
	return normalize_vector(&rightOut[0], cols);
}

static unsigned int top_singular_modes(const std::vector<float>& matrix,
                                       unsigned int rows,
                                       unsigned int cols,
                                       unsigned int maxModes,
                                       std::vector<float>& sigmaOut,
                                       std::vector<float>& leftOut,
                                       std::vector<float>& rightOut)
{
	sigmaOut.clear();
	leftOut.clear();
	rightOut.clear();
	if (rows == 0u || cols == 0u || maxModes == 0u
	    || matrix.size() < static_cast<size_t>(rows) * static_cast<size_t>(cols))
		return 0u;

	std::vector<float> gram(static_cast<size_t>(rows) * rows, 0.0f);
	for (unsigned int i = 0; i < rows; ++i)
	{
		for (unsigned int j = i; j < rows; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < cols; ++k)
			{
				sum += static_cast<double>(matrix[static_cast<size_t>(i) * cols + k])
				     * static_cast<double>(matrix[static_cast<size_t>(j) * cols + k]);
			}
			gram[static_cast<size_t>(i) * rows + j] = static_cast<float>(sum);
			gram[static_cast<size_t>(j) * rows + i] = static_cast<float>(sum);
		}
	}
	symmetrize_block(&gram[0], rows);

	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(&gram[0], rows, eigVec, eigVal);
	if (eigVal.empty())
		return 0u;

	const unsigned int requestedModes = (maxModes > rows) ? rows : maxModes;
	sigmaOut.assign(static_cast<size_t>(requestedModes), 0.0f);
	leftOut.assign(static_cast<size_t>(requestedModes) * rows, 0.0f);
	rightOut.assign(static_cast<size_t>(requestedModes) * cols, 0.0f);

	unsigned int found = 0u;
	for (unsigned int mode = 0; mode < requestedModes; ++mode)
	{
		double topEig = static_cast<double>(eigVal[mode]);
		if (!std::isfinite(topEig) || topEig <= 0.0)
			break;
		const double sigma = std::sqrt(topEig);
		if (!(sigma > 0.0))
			break;

		std::vector<float> left(static_cast<size_t>(rows), 0.0f);
		for (unsigned int i = 0; i < rows; ++i)
			left[i] = eigVec[static_cast<size_t>(i) * rows + mode];
		if (!normalize_vector(&left[0], rows))
			break;

		std::vector<float> right(static_cast<size_t>(cols), 0.0f);
		for (unsigned int j = 0; j < cols; ++j)
		{
			double sum = 0.0;
			for (unsigned int i = 0; i < rows; ++i)
			{
				sum += static_cast<double>(matrix[static_cast<size_t>(i) * cols + j])
				     * static_cast<double>(left[i]);
			}
			right[j] = static_cast<float>(sum / sigma);
		}
		if (!normalize_vector(&right[0], cols))
			break;

		sigmaOut[found] = static_cast<float>(sigma);
		std::copy(left.begin(), left.end(),
		          leftOut.begin() + static_cast<size_t>(found) * rows);
		std::copy(right.begin(), right.end(),
		          rightOut.begin() + static_cast<size_t>(found) * cols);
		++found;
	}

	sigmaOut.resize(found);
	leftOut.resize(static_cast<size_t>(found) * rows);
	rightOut.resize(static_cast<size_t>(found) * cols);
	return found;
}

static float compute_ghost_balanced_mode(const WeightState& state,
                                         const float* W,
                                         const float* activeBasisPacked,
                                         const std::vector<float>& gz,
                                         unsigned int scoutRank,
                                         unsigned int activeRank,
                                         unsigned int m,
                                         unsigned int n,
                                         unsigned int lagHorizon,
                                         float statScaleSq,
                                         float eps,
                                         float* sigmaOut,
                                         float* horizontalRatioOut,
                                         std::vector<float>& leftModeOut,
                                         std::vector<float>& rightModeOut,
                                         std::vector<float>& pastStackOut)
{
	leftModeOut.clear();
	rightModeOut.clear();
	pastStackOut.clear();
	if (sigmaOut)
		*sigmaOut = 0.0f;
	if (horizontalRatioOut)
		*horizontalRatioOut = 1.0f;
	if (activeRank == 0u || n == 0u || lagHorizon == 0u
	    || !W || !activeBasisPacked
	    || gz.size() < static_cast<size_t>(activeRank) * static_cast<size_t>(n))
		return 0.0f;

	const size_t activeSlice = static_cast<size_t>(state.r) * static_cast<size_t>(n);
	if (state.resolveGzHistory.size() < static_cast<size_t>(lagHorizon) * activeSlice)
		return 0.0f;

	unsigned int effectiveScoutRank = scoutRank;
	const unsigned int storageRank = state.complementRank;
	const size_t scoutSlice = static_cast<size_t>(storageRank) * static_cast<size_t>(n);
	if (effectiveScoutRank > 0u)
	{
		if (storageRank == 0u
		    || state.heroGwHistory.size() < static_cast<size_t>(lagHorizon) * scoutSlice)
			effectiveScoutRank = 0u;
	}

	std::vector<float> activeGauge(static_cast<size_t>(activeRank) * n, 0.0f);
	glades::gemm::atb(&activeGauge[0], activeBasisPacked, W, activeRank, m, n, 1.0f);

	std::vector<float> scoutGauge;
	if (effectiveScoutRank > 0u)
	{
		std::vector<float> scoutBasisPacked(static_cast<size_t>(m) * effectiveScoutRank, 0.0f);
		pack_active_basis(state.scoutBasis, state.complementRank, m, effectiveScoutRank, scoutBasisPacked);
		scoutGauge.assign(static_cast<size_t>(effectiveScoutRank) * n, 0.0f);
		glades::gemm::atb(&scoutGauge[0], &scoutBasisPacked[0], W, effectiveScoutRank, m, n, 1.0f);
	}

	std::vector<float> futureBlock(gz.begin(), gz.begin() + static_cast<size_t>(activeRank) * n);
	const float horizRatio =
	    remove_vectorized_projection(&futureBlock[0], &activeGauge[0], futureBlock.size(), eps);
	if (horizontalRatioOut)
		*horizontalRatioOut = horizRatio;

	const unsigned int pastRows = lagHorizon * (activeRank + effectiveScoutRank);
	if (pastRows == 0u)
		return 0.0f;
	pastStackOut.assign(static_cast<size_t>(pastRows) * n, 0.0f);
	for (unsigned int lag = 0u; lag < lagHorizon; ++lag)
	{
		const float lagScale =
		    static_cast<float>(1.0 / std::sqrt(static_cast<double>(lag + 1u)));
		const size_t activeBase = static_cast<size_t>(lag) * activeSlice;
		const unsigned int blockOffset = lag * (activeRank + effectiveScoutRank);

		std::vector<float> activeChunk(static_cast<size_t>(activeRank) * n, 0.0f);
		for (unsigned int row = 0; row < activeRank; ++row)
		{
			std::copy(state.resolveGzHistory.begin() + activeBase + static_cast<size_t>(row) * n,
			          state.resolveGzHistory.begin() + activeBase + static_cast<size_t>(row + 1u) * n,
			          activeChunk.begin() + static_cast<size_t>(row) * n);
		}
		remove_vectorized_projection(&activeChunk[0], &activeGauge[0], activeChunk.size(), eps);
		for (size_t idx = 0; idx < activeChunk.size(); ++idx)
			activeChunk[idx] *= lagScale;
		for (unsigned int row = 0; row < activeRank; ++row)
		{
			std::copy(activeChunk.begin() + static_cast<size_t>(row) * n,
			          activeChunk.begin() + static_cast<size_t>(row + 1u) * n,
			          pastStackOut.begin() + static_cast<size_t>(blockOffset + row) * n);
		}

		if (effectiveScoutRank > 0u)
		{
			const size_t scoutBase = static_cast<size_t>(lag) * scoutSlice;
			std::vector<float> scoutChunk(static_cast<size_t>(effectiveScoutRank) * n, 0.0f);
			for (unsigned int row = 0; row < effectiveScoutRank; ++row)
			{
				std::copy(state.heroGwHistory.begin() + scoutBase + static_cast<size_t>(row) * n,
				          state.heroGwHistory.begin() + scoutBase + static_cast<size_t>(row + 1u) * n,
				          scoutChunk.begin() + static_cast<size_t>(row) * n);
			}
			remove_vectorized_projection(&scoutChunk[0], &scoutGauge[0], scoutChunk.size(), eps);
			for (size_t idx = 0; idx < scoutChunk.size(); ++idx)
				scoutChunk[idx] *= lagScale;
			for (unsigned int row = 0; row < effectiveScoutRank; ++row)
			{
				std::copy(scoutChunk.begin() + static_cast<size_t>(row) * n,
				          scoutChunk.begin() + static_cast<size_t>(row + 1u) * n,
				          pastStackOut.begin()
				              + static_cast<size_t>(blockOffset + activeRank + row) * n);
			}
		}
	}

	std::vector<float> futureCov(static_cast<size_t>(activeRank) * activeRank, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = i; j < activeRank; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(futureBlock[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(futureBlock[static_cast<size_t>(j) * n + col]);
			}
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq));
			futureCov[static_cast<size_t>(i) * activeRank + j] = v;
			futureCov[static_cast<size_t>(j) * activeRank + i] = v;
		}
	}

	std::vector<float> pastCov(static_cast<size_t>(pastRows) * pastRows, 0.0f);
	for (unsigned int i = 0; i < pastRows; ++i)
	{
		for (unsigned int j = i; j < pastRows; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(pastStackOut[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(pastStackOut[static_cast<size_t>(j) * n + col]);
			}
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq));
			pastCov[static_cast<size_t>(i) * pastRows + j] = v;
			pastCov[static_cast<size_t>(j) * pastRows + i] = v;
		}
	}

	std::vector<float> cross(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int p = 0; p < pastRows; ++p)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(futureBlock[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(pastStackOut[static_cast<size_t>(p) * n + col]);
			}
			cross[static_cast<size_t>(i) * pastRows + p] =
			    static_cast<float>((dot / static_cast<double>(n))
			                      * static_cast<double>(statScaleSq));
		}
	}

	std::vector<float> invSqrtFuture;
	std::vector<float> invSqrtPast;
	if (!build_inv_sqrt_psd(&futureCov[0], activeRank, eps, invSqrtFuture)
	    || !build_inv_sqrt_psd(&pastCov[0], pastRows, eps, invSqrtPast))
		return 0.0f;

	std::vector<float> temp(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	multiply_left_block(&temp[0], &invSqrtFuture[0], &cross[0], activeRank, activeRank, pastRows);

	std::vector<float> whitened(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = 0; j < pastRows; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < pastRows; ++k)
			{
				sum += static_cast<double>(temp[static_cast<size_t>(i) * pastRows + k])
				     * static_cast<double>(invSqrtPast[static_cast<size_t>(k) * pastRows + j]);
			}
			whitened[static_cast<size_t>(i) * pastRows + j] = static_cast<float>(sum);
		}
	}

	std::vector<float> leftWhite;
	std::vector<float> rightWhite;
	float sigma = 0.0f;
	if (!top_singular_triplet(whitened, activeRank, pastRows, &sigma, leftWhite, rightWhite))
		return 0.0f;
	if (!std::isfinite(sigma))
		return 0.0f;
	if (sigmaOut)
		*sigmaOut = sigma;

	leftModeOut.assign(static_cast<size_t>(activeRank), 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		double sum = 0.0;
		for (unsigned int k = 0; k < activeRank; ++k)
			sum += static_cast<double>(invSqrtFuture[static_cast<size_t>(i) * activeRank + k])
			     * static_cast<double>(leftWhite[k]);
		leftModeOut[i] = static_cast<float>(sum);
	}
	if (!normalize_vector(&leftModeOut[0], activeRank))
		return 0.0f;

	rightModeOut.assign(static_cast<size_t>(pastRows), 0.0f);
	for (unsigned int i = 0; i < pastRows; ++i)
	{
		double sum = 0.0;
		for (unsigned int k = 0; k < pastRows; ++k)
			sum += static_cast<double>(invSqrtPast[static_cast<size_t>(i) * pastRows + k])
			     * static_cast<double>(rightWhite[k]);
		rightModeOut[i] = static_cast<float>(sum);
	}
	if (!normalize_vector(&rightModeOut[0], pastRows))
		return 0.0f;

	const float edge = sigma - 1.0f;
	return std::isfinite(edge) ? edge : 0.0f;
}

static float compute_qbrt_balanced_mode(WeightState& state,
                                        const float* W,
                                        const float* activeBasisPacked,
                                        const std::vector<float>& gz,
                                        unsigned int scoutRank,
                                        unsigned int activeRank,
                                        unsigned int m,
                                        unsigned int n,
                                        unsigned int lagHorizon,
                                        float beta,
                                        float statScaleSq,
                                        float eps,
                                        float poleMax,
                                        float* sigmaOut,
                                        float* poleOut,
                                        float* horizontalRatioOut,
                                        std::vector<float>& leftModeOut,
                                        std::vector<float>& latentOut)
{
	leftModeOut.clear();
	latentOut.clear();
	if (sigmaOut)
		*sigmaOut = 0.0f;
	if (poleOut)
		*poleOut = 0.0f;
	if (horizontalRatioOut)
		*horizontalRatioOut = 1.0f;
	if (activeRank == 0u || n == 0u || lagHorizon == 0u
	    || !W || !activeBasisPacked
	    || gz.size() < static_cast<size_t>(activeRank) * static_cast<size_t>(n))
		return 0.0f;
	if (state.qbrtLeftMode.size() < static_cast<size_t>(state.r)
	    || state.qbrtLatent.size() < static_cast<size_t>(n))
		return 0.0f;

	const size_t activeSlice = static_cast<size_t>(state.r) * static_cast<size_t>(n);
	if (state.resolveGzHistory.size() < static_cast<size_t>(lagHorizon) * activeSlice)
		return 0.0f;

	unsigned int effectiveScoutRank = scoutRank;
	const unsigned int storageRank = state.complementRank;
	const size_t scoutSlice = static_cast<size_t>(storageRank) * static_cast<size_t>(n);
	if (effectiveScoutRank > 0u)
	{
		if (storageRank == 0u
		    || state.heroGwHistory.size() < static_cast<size_t>(lagHorizon) * scoutSlice)
			effectiveScoutRank = 0u;
	}

	std::vector<float> activeGauge(static_cast<size_t>(activeRank) * n, 0.0f);
	glades::gemm::atb(&activeGauge[0], activeBasisPacked, W, activeRank, m, n, 1.0f);

	std::vector<float> scoutGauge;
	if (effectiveScoutRank > 0u)
	{
		std::vector<float> scoutBasisPacked(static_cast<size_t>(m) * effectiveScoutRank, 0.0f);
		pack_active_basis(state.scoutBasis, state.complementRank, m, effectiveScoutRank, scoutBasisPacked);
		scoutGauge.assign(static_cast<size_t>(effectiveScoutRank) * n, 0.0f);
		glades::gemm::atb(&scoutGauge[0], &scoutBasisPacked[0], W, effectiveScoutRank, m, n, 1.0f);
	}

	std::vector<float> futureBlock(gz.begin(), gz.begin() + static_cast<size_t>(activeRank) * n);
	const float horizRatio =
	    remove_vectorized_projection(&futureBlock[0], &activeGauge[0], futureBlock.size(), eps);
	if (horizontalRatioOut)
		*horizontalRatioOut = horizRatio;

	const unsigned int pastRows = lagHorizon * (activeRank + effectiveScoutRank);
	if (pastRows == 0u)
		return 0.0f;
	std::vector<float> pastStack(static_cast<size_t>(pastRows) * n, 0.0f);
	for (unsigned int lag = 0u; lag < lagHorizon; ++lag)
	{
		const float lagScale =
		    static_cast<float>(1.0 / std::sqrt(static_cast<double>(lag + 1u)));
		const size_t activeBase = static_cast<size_t>(lag) * activeSlice;
		const unsigned int blockOffset = lag * (activeRank + effectiveScoutRank);

		std::vector<float> activeChunk(static_cast<size_t>(activeRank) * n, 0.0f);
		for (unsigned int row = 0; row < activeRank; ++row)
		{
			std::copy(state.resolveGzHistory.begin() + activeBase + static_cast<size_t>(row) * n,
			          state.resolveGzHistory.begin() + activeBase + static_cast<size_t>(row + 1u) * n,
			          activeChunk.begin() + static_cast<size_t>(row) * n);
		}
		remove_vectorized_projection(&activeChunk[0], &activeGauge[0], activeChunk.size(), eps);
		for (size_t idx = 0; idx < activeChunk.size(); ++idx)
			activeChunk[idx] *= lagScale;
		for (unsigned int row = 0; row < activeRank; ++row)
		{
			std::copy(activeChunk.begin() + static_cast<size_t>(row) * n,
			          activeChunk.begin() + static_cast<size_t>(row + 1u) * n,
			          pastStack.begin() + static_cast<size_t>(blockOffset + row) * n);
		}

		if (effectiveScoutRank > 0u)
		{
			const size_t scoutBase = static_cast<size_t>(lag) * scoutSlice;
			std::vector<float> scoutChunk(static_cast<size_t>(effectiveScoutRank) * n, 0.0f);
			for (unsigned int row = 0; row < effectiveScoutRank; ++row)
			{
				std::copy(state.heroGwHistory.begin() + scoutBase + static_cast<size_t>(row) * n,
				          state.heroGwHistory.begin() + scoutBase + static_cast<size_t>(row + 1u) * n,
				          scoutChunk.begin() + static_cast<size_t>(row) * n);
			}
			remove_vectorized_projection(&scoutChunk[0], &scoutGauge[0], scoutChunk.size(), eps);
			for (size_t idx = 0; idx < scoutChunk.size(); ++idx)
				scoutChunk[idx] *= lagScale;
			for (unsigned int row = 0; row < effectiveScoutRank; ++row)
			{
				std::copy(scoutChunk.begin() + static_cast<size_t>(row) * n,
				          scoutChunk.begin() + static_cast<size_t>(row + 1u) * n,
				          pastStack.begin()
				              + static_cast<size_t>(blockOffset + activeRank + row) * n);
			}
		}
	}

	std::vector<float> futureCov(static_cast<size_t>(activeRank) * activeRank, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = i; j < activeRank; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(futureBlock[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(futureBlock[static_cast<size_t>(j) * n + col]);
			}
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq));
			futureCov[static_cast<size_t>(i) * activeRank + j] = v;
			futureCov[static_cast<size_t>(j) * activeRank + i] = v;
		}
	}

	std::vector<float> pastCov(static_cast<size_t>(pastRows) * pastRows, 0.0f);
	for (unsigned int i = 0; i < pastRows; ++i)
	{
		for (unsigned int j = i; j < pastRows; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(pastStack[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(pastStack[static_cast<size_t>(j) * n + col]);
			}
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq));
			pastCov[static_cast<size_t>(i) * pastRows + j] = v;
			pastCov[static_cast<size_t>(j) * pastRows + i] = v;
		}
	}

	std::vector<float> cross(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int p = 0; p < pastRows; ++p)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(futureBlock[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(pastStack[static_cast<size_t>(p) * n + col]);
			}
			cross[static_cast<size_t>(i) * pastRows + p] =
			    static_cast<float>((dot / static_cast<double>(n))
			                      * static_cast<double>(statScaleSq));
		}
	}

	std::vector<float> invSqrtFuture;
	std::vector<float> invSqrtPast;
	if (!build_inv_sqrt_psd(&futureCov[0], activeRank, eps, invSqrtFuture)
	    || !build_inv_sqrt_psd(&pastCov[0], pastRows, eps, invSqrtPast))
		return 0.0f;

	std::vector<float> temp(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	multiply_left_block(&temp[0], &invSqrtFuture[0], &cross[0], activeRank, activeRank, pastRows);

	std::vector<float> whitened(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = 0; j < pastRows; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < pastRows; ++k)
			{
				sum += static_cast<double>(temp[static_cast<size_t>(i) * pastRows + k])
				     * static_cast<double>(invSqrtPast[static_cast<size_t>(k) * pastRows + j]);
			}
			whitened[static_cast<size_t>(i) * pastRows + j] = static_cast<float>(sum);
		}
	}

	std::vector<float> leftWhite;
	std::vector<float> rightWhite;
	float sigma = 0.0f;
	if (!top_singular_triplet(whitened, activeRank, pastRows, &sigma, leftWhite, rightWhite)
	    || !std::isfinite(sigma))
		return 0.0f;
	if (sigmaOut)
		*sigmaOut = sigma;

	leftModeOut.assign(static_cast<size_t>(activeRank), 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		double sum = 0.0;
		for (unsigned int k = 0; k < activeRank; ++k)
			sum += static_cast<double>(invSqrtFuture[static_cast<size_t>(i) * activeRank + k])
			     * static_cast<double>(leftWhite[k]);
		leftModeOut[i] = static_cast<float>(sum);
	}
	if (!normalize_vector(&leftModeOut[0], activeRank))
		return 0.0f;

	std::vector<float> rightMode(static_cast<size_t>(pastRows), 0.0f);
	for (unsigned int i = 0; i < pastRows; ++i)
	{
		double sum = 0.0;
		for (unsigned int k = 0; k < pastRows; ++k)
			sum += static_cast<double>(invSqrtPast[static_cast<size_t>(i) * pastRows + k])
			     * static_cast<double>(rightWhite[k]);
		rightMode[i] = static_cast<float>(sum);
	}
	if (!normalize_vector(&rightMode[0], pastRows))
		return 0.0f;

	const double emaKeep =
	    std::min<double>(1.0, std::max<double>(0.0, static_cast<double>(beta)));
	const double emaAdd = 1.0 - emaKeep;
	const float priorPole = atlas_isfinite(state.qbrtPole) ? state.qbrtPole : 0.0f;
	latentOut.assign(static_cast<size_t>(n), 0.0f);
	double poleNumerSample = 0.0;
	double poleDenomSample = 0.0;
	for (unsigned int j = 0; j < n; ++j)
	{
		double input = 0.0;
		for (unsigned int row = 0; row < pastRows; ++row)
			input += static_cast<double>(rightMode[row])
			      * static_cast<double>(pastStack[static_cast<size_t>(row) * n + j]);
		const double oldLatent = static_cast<double>(state.qbrtLatent[j]);
		const double newLatent =
		    static_cast<double>(priorPole) * oldLatent + input;
		latentOut[j] = static_cast<float>(newLatent);
		poleNumerSample += newLatent * oldLatent;
		poleDenomSample += oldLatent * oldLatent;
	}

	if (n > 0u)
	{
		poleNumerSample /= static_cast<double>(n);
		poleDenomSample /= static_cast<double>(n);
	}
	state.qbrtPoleNumer = static_cast<float>(
	    emaKeep * static_cast<double>(state.qbrtPoleNumer) + emaAdd * poleNumerSample);
	state.qbrtPoleDenom = static_cast<float>(
	    emaKeep * static_cast<double>(state.qbrtPoleDenom) + emaAdd * poleDenomSample);
	if (state.qbrtPoleDenom > eps && atlas_isfinite(state.qbrtPoleDenom))
	{
		double rho = static_cast<double>(state.qbrtPoleNumer)
		           / static_cast<double>(state.qbrtPoleDenom);
		const double rhoMax =
		    std::min<double>(0.999, std::max<double>(0.0, static_cast<double>(poleMax)));
		if (!std::isfinite(rho))
			rho = 0.0;
		rho = std::max(-rhoMax, std::min(rhoMax, rho));
		state.qbrtPole = static_cast<float>(rho);
	}
	else if (!atlas_isfinite(state.qbrtPole))
	{
		state.qbrtPole = 0.0f;
	}
	if (poleOut)
		*poleOut = state.qbrtPole;
	std::copy(leftModeOut.begin(), leftModeOut.end(), state.qbrtLeftMode.begin());
	std::copy(latentOut.begin(), latentOut.end(), state.qbrtLatent.begin());

	double edge = std::max<double>(0.0, static_cast<double>(sigma) - 1.0);
	edge *= std::max<double>(0.0, static_cast<double>(horizRatio));
	edge = std::min(1.0, std::max(0.0, edge));
	return std::isfinite(edge) ? static_cast<float>(edge) : 0.0f;
}

static float compute_qrc_control_mode(WeightState& state,
                                      const float* W,
                                      const float* activeBasisPacked,
                                      const std::vector<float>& gz,
                                      unsigned int scoutRank,
                                      unsigned int activeRank,
                                      unsigned int m,
                                      unsigned int n,
                                      unsigned int lagHorizon,
                                      float beta,
                                      float statScaleSq,
                                      float eps,
                                      float poleMax,
                                      float* sigmaOut,
                                      float* poleOut,
                                      float* horizontalRatioOut,
                                      float* controlGainOut,
                                      std::vector<float>& leftModeOut,
                                      std::vector<float>& latentOut)
{
	leftModeOut.clear();
	latentOut.clear();
	if (sigmaOut)
		*sigmaOut = 0.0f;
	if (poleOut)
		*poleOut = 0.0f;
	if (horizontalRatioOut)
		*horizontalRatioOut = 1.0f;
	if (controlGainOut)
		*controlGainOut = 0.0f;
	if (activeRank == 0u || n == 0u || lagHorizon == 0u
	    || !W || !activeBasisPacked
	    || gz.size() < static_cast<size_t>(activeRank) * static_cast<size_t>(n))
		return 0.0f;
	if (state.qrcLeftMode.size() < static_cast<size_t>(state.r)
	    || state.qrcLatent.size() < static_cast<size_t>(n))
		return 0.0f;

	const size_t activeSlice = static_cast<size_t>(state.r) * static_cast<size_t>(n);
	if (state.resolveGzHistory.size() < static_cast<size_t>(lagHorizon) * activeSlice)
		return 0.0f;

	unsigned int effectiveScoutRank = scoutRank;
	const unsigned int storageRank = state.complementRank;
	const size_t scoutSlice = static_cast<size_t>(storageRank) * static_cast<size_t>(n);
	if (effectiveScoutRank > 0u)
	{
		if (storageRank == 0u
		    || state.heroGwHistory.size() < static_cast<size_t>(lagHorizon) * scoutSlice)
			effectiveScoutRank = 0u;
	}

	std::vector<float> activeGauge(static_cast<size_t>(activeRank) * n, 0.0f);
	glades::gemm::atb(&activeGauge[0], activeBasisPacked, W, activeRank, m, n, 1.0f);

	std::vector<float> scoutGauge;
	if (effectiveScoutRank > 0u)
	{
		std::vector<float> scoutBasisPacked(static_cast<size_t>(m) * effectiveScoutRank, 0.0f);
		pack_active_basis(state.scoutBasis, state.complementRank, m, effectiveScoutRank, scoutBasisPacked);
		scoutGauge.assign(static_cast<size_t>(effectiveScoutRank) * n, 0.0f);
		glades::gemm::atb(&scoutGauge[0], &scoutBasisPacked[0], W, effectiveScoutRank, m, n, 1.0f);
	}

	std::vector<float> futureBlock(gz.begin(), gz.begin() + static_cast<size_t>(activeRank) * n);
	const float horizRatio =
	    remove_vectorized_projection(&futureBlock[0], &activeGauge[0], futureBlock.size(), eps);
	if (horizontalRatioOut)
		*horizontalRatioOut = horizRatio;

	const unsigned int pastRows = lagHorizon * (activeRank + effectiveScoutRank);
	if (pastRows == 0u)
		return 0.0f;
	std::vector<float> pastStack(static_cast<size_t>(pastRows) * n, 0.0f);
	for (unsigned int lag = 0u; lag < lagHorizon; ++lag)
	{
		const float lagScale =
		    static_cast<float>(1.0 / std::sqrt(static_cast<double>(lag + 1u)));
		const size_t activeBase = static_cast<size_t>(lag) * activeSlice;
		const unsigned int blockOffset = lag * (activeRank + effectiveScoutRank);

		std::vector<float> activeChunk(static_cast<size_t>(activeRank) * n, 0.0f);
		for (unsigned int row = 0; row < activeRank; ++row)
		{
			std::copy(state.resolveGzHistory.begin() + activeBase + static_cast<size_t>(row) * n,
			          state.resolveGzHistory.begin() + activeBase + static_cast<size_t>(row + 1u) * n,
			          activeChunk.begin() + static_cast<size_t>(row) * n);
		}
		remove_vectorized_projection(&activeChunk[0], &activeGauge[0], activeChunk.size(), eps);
		for (size_t idx = 0; idx < activeChunk.size(); ++idx)
			activeChunk[idx] *= lagScale;
		for (unsigned int row = 0; row < activeRank; ++row)
		{
			std::copy(activeChunk.begin() + static_cast<size_t>(row) * n,
			          activeChunk.begin() + static_cast<size_t>(row + 1u) * n,
			          pastStack.begin() + static_cast<size_t>(blockOffset + row) * n);
		}

		if (effectiveScoutRank > 0u)
		{
			const size_t scoutBase = static_cast<size_t>(lag) * scoutSlice;
			std::vector<float> scoutChunk(static_cast<size_t>(effectiveScoutRank) * n, 0.0f);
			for (unsigned int row = 0; row < effectiveScoutRank; ++row)
			{
				std::copy(state.heroGwHistory.begin() + scoutBase + static_cast<size_t>(row) * n,
				          state.heroGwHistory.begin() + scoutBase + static_cast<size_t>(row + 1u) * n,
				          scoutChunk.begin() + static_cast<size_t>(row) * n);
			}
			remove_vectorized_projection(&scoutChunk[0], &scoutGauge[0], scoutChunk.size(), eps);
			for (size_t idx = 0; idx < scoutChunk.size(); ++idx)
				scoutChunk[idx] *= lagScale;
			for (unsigned int row = 0; row < effectiveScoutRank; ++row)
			{
				std::copy(scoutChunk.begin() + static_cast<size_t>(row) * n,
				          scoutChunk.begin() + static_cast<size_t>(row + 1u) * n,
				          pastStack.begin()
				              + static_cast<size_t>(blockOffset + activeRank + row) * n);
			}
		}
	}

	std::vector<float> futureCov(static_cast<size_t>(activeRank) * activeRank, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = i; j < activeRank; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(futureBlock[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(futureBlock[static_cast<size_t>(j) * n + col]);
			}
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq));
			futureCov[static_cast<size_t>(i) * activeRank + j] = v;
			futureCov[static_cast<size_t>(j) * activeRank + i] = v;
		}
	}

	std::vector<float> pastCov(static_cast<size_t>(pastRows) * pastRows, 0.0f);
	for (unsigned int i = 0; i < pastRows; ++i)
	{
		for (unsigned int j = i; j < pastRows; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(pastStack[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(pastStack[static_cast<size_t>(j) * n + col]);
			}
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq));
			pastCov[static_cast<size_t>(i) * pastRows + j] = v;
			pastCov[static_cast<size_t>(j) * pastRows + i] = v;
		}
	}

	std::vector<float> cross(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int p = 0; p < pastRows; ++p)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(futureBlock[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(pastStack[static_cast<size_t>(p) * n + col]);
			}
			cross[static_cast<size_t>(i) * pastRows + p] =
			    static_cast<float>((dot / static_cast<double>(n))
			                      * static_cast<double>(statScaleSq));
		}
	}

	std::vector<float> invSqrtFuture;
	std::vector<float> invSqrtPast;
	if (!build_inv_sqrt_psd(&futureCov[0], activeRank, eps, invSqrtFuture)
	    || !build_inv_sqrt_psd(&pastCov[0], pastRows, eps, invSqrtPast))
		return 0.0f;

	std::vector<float> temp(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	multiply_left_block(&temp[0], &invSqrtFuture[0], &cross[0], activeRank, activeRank, pastRows);

	std::vector<float> whitened(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = 0; j < pastRows; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < pastRows; ++k)
			{
				sum += static_cast<double>(temp[static_cast<size_t>(i) * pastRows + k])
				     * static_cast<double>(invSqrtPast[static_cast<size_t>(k) * pastRows + j]);
			}
			whitened[static_cast<size_t>(i) * pastRows + j] = static_cast<float>(sum);
		}
	}

	std::vector<float> leftWhite;
	std::vector<float> rightWhite;
	float sigma = 0.0f;
	if (!top_singular_triplet(whitened, activeRank, pastRows, &sigma, leftWhite, rightWhite)
	    || !std::isfinite(sigma))
		return 0.0f;
	if (sigmaOut)
		*sigmaOut = sigma;

	leftModeOut.assign(static_cast<size_t>(activeRank), 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		double sum = 0.0;
		for (unsigned int k = 0; k < activeRank; ++k)
			sum += static_cast<double>(invSqrtFuture[static_cast<size_t>(i) * activeRank + k])
			     * static_cast<double>(leftWhite[k]);
		leftModeOut[i] = static_cast<float>(sum);
	}
	if (!normalize_vector(&leftModeOut[0], activeRank))
		return 0.0f;

	std::vector<float> rightMode(static_cast<size_t>(pastRows), 0.0f);
	for (unsigned int i = 0; i < pastRows; ++i)
	{
		double sum = 0.0;
		for (unsigned int k = 0; k < pastRows; ++k)
			sum += static_cast<double>(invSqrtPast[static_cast<size_t>(i) * pastRows + k])
			     * static_cast<double>(rightWhite[k]);
		rightMode[i] = static_cast<float>(sum);
	}
	if (!normalize_vector(&rightMode[0], pastRows))
		return 0.0f;

	const double emaKeep =
	    std::min<double>(1.0, std::max<double>(0.0, static_cast<double>(beta)));
	const double emaAdd = 1.0 - emaKeep;
	const float priorPole = atlas_isfinite(state.qrcPole) ? state.qrcPole : 0.0f;
	latentOut.assign(static_cast<size_t>(n), 0.0f);
	double poleNumerSample = 0.0;
	double poleDenomSample = 0.0;
	for (unsigned int j = 0; j < n; ++j)
	{
		double input = 0.0;
		for (unsigned int row = 0; row < pastRows; ++row)
			input += static_cast<double>(rightMode[row])
			      * static_cast<double>(pastStack[static_cast<size_t>(row) * n + j]);
		const double oldLatent = static_cast<double>(state.qrcLatent[j]);
		const double newLatent =
		    static_cast<double>(priorPole) * oldLatent + input;
		latentOut[j] = static_cast<float>(newLatent);
		poleNumerSample += newLatent * oldLatent;
		poleDenomSample += oldLatent * oldLatent;
	}

	if (n > 0u)
	{
		poleNumerSample /= static_cast<double>(n);
		poleDenomSample /= static_cast<double>(n);
	}
	state.qrcPoleNumer = static_cast<float>(
	    emaKeep * static_cast<double>(state.qrcPoleNumer) + emaAdd * poleNumerSample);
	state.qrcPoleDenom = static_cast<float>(
	    emaKeep * static_cast<double>(state.qrcPoleDenom) + emaAdd * poleDenomSample);
	if (state.qrcPoleDenom > eps && atlas_isfinite(state.qrcPoleDenom))
	{
		double rho = static_cast<double>(state.qrcPoleNumer)
		           / static_cast<double>(state.qrcPoleDenom);
		const double rhoMax =
		    std::min<double>(0.999, std::max<double>(0.0, static_cast<double>(poleMax)));
		if (!std::isfinite(rho))
			rho = 0.0;
		rho = std::max(-rhoMax, std::min(rhoMax, rho));
		state.qrcPole = static_cast<float>(rho);
	}
	else if (!atlas_isfinite(state.qrcPole))
	{
		state.qrcPole = 0.0f;
	}
	if (poleOut)
		*poleOut = state.qrcPole;
	std::copy(leftModeOut.begin(), leftModeOut.end(), state.qrcLeftMode.begin());
	std::copy(latentOut.begin(), latentOut.end(), state.qrcLatent.begin());

	const double poleAbs = std::fabs(static_cast<double>(state.qrcPole));
	double controlGain = poleAbs / (1.0 + poleAbs);
	if (!std::isfinite(controlGain))
		controlGain = 0.0;
	controlGain = std::min(1.0, std::max(0.0, controlGain));
	if (controlGainOut)
		*controlGainOut = static_cast<float>(controlGain);

	const double closedLoopPole = std::max<double>(0.0, poleAbs - controlGain);
	double edge = std::max<double>(0.0, static_cast<double>(sigma) - 1.0);
	edge *= std::max<double>(0.0, static_cast<double>(horizRatio));
	edge *= std::max<double>(0.0, 1.0 - closedLoopPole);
	edge = std::min(1.0, std::max(0.0, edge));
	return std::isfinite(edge) ? static_cast<float>(edge) : 0.0f;
}

static float compute_rift_signature_mode(WeightState& state,
                                         const float* W,
                                         const float* activeBasisPacked,
                                         const std::vector<float>& gz,
                                         unsigned int scoutRank,
                                         unsigned int activeRank,
                                         unsigned int m,
                                         unsigned int n,
                                         unsigned int lagHorizon,
                                         float beta,
                                         float statScaleSq,
                                         float eps,
                                         float poleMax,
                                         float* sigmaOut,
                                         float* poleOut,
                                         float* horizontalRatioOut,
                                         float* areaEnergyOut,
                                         float* predR2Out,
                                         std::vector<float>& leftModeOut,
                                         std::vector<float>& latentOut)
{
	leftModeOut.clear();
	latentOut.clear();
	if (sigmaOut)
		*sigmaOut = 0.0f;
	if (poleOut)
		*poleOut = 0.0f;
	if (horizontalRatioOut)
		*horizontalRatioOut = 1.0f;
	if (areaEnergyOut)
		*areaEnergyOut = 0.0f;
	if (predR2Out)
		*predR2Out = 0.0f;
	if (activeRank == 0u || n == 0u || lagHorizon < 2u
	    || !W || !activeBasisPacked
	    || gz.size() < static_cast<size_t>(activeRank) * static_cast<size_t>(n))
		return 0.0f;
	if (state.riftLeftMode.size() < static_cast<size_t>(state.r)
	    || state.riftLatent.size() < static_cast<size_t>(n))
		return 0.0f;

	const size_t activeSlice = static_cast<size_t>(state.r) * static_cast<size_t>(n);
	if (state.resolveGzHistory.size() < static_cast<size_t>(lagHorizon) * activeSlice)
		return 0.0f;

	unsigned int effectiveScoutRank = scoutRank;
	const unsigned int storageRank = state.complementRank;
	const size_t scoutSlice = static_cast<size_t>(storageRank) * static_cast<size_t>(n);
	if (effectiveScoutRank > 0u)
	{
		if (storageRank == 0u
		    || state.heroGwHistory.size() < static_cast<size_t>(lagHorizon) * scoutSlice)
			effectiveScoutRank = 0u;
	}

	std::vector<float> activeGauge(static_cast<size_t>(activeRank) * n, 0.0f);
	glades::gemm::atb(&activeGauge[0], activeBasisPacked, W, activeRank, m, n, 1.0f);

	std::vector<float> scoutGauge;
	if (effectiveScoutRank > 0u)
	{
		std::vector<float> scoutBasisPacked(static_cast<size_t>(m) * effectiveScoutRank, 0.0f);
		pack_active_basis(state.scoutBasis, state.complementRank, m, effectiveScoutRank, scoutBasisPacked);
		scoutGauge.assign(static_cast<size_t>(effectiveScoutRank) * n, 0.0f);
		glades::gemm::atb(&scoutGauge[0], &scoutBasisPacked[0], W, effectiveScoutRank, m, n, 1.0f);
	}

	std::vector<float> futureBlock(gz.begin(), gz.begin() + static_cast<size_t>(activeRank) * n);
	const float horizRatio =
	    remove_vectorized_projection(&futureBlock[0], &activeGauge[0], futureBlock.size(), eps);
	if (horizontalRatioOut)
		*horizontalRatioOut = horizRatio;

	const unsigned int stateDim = activeRank + effectiveScoutRank;
	if (stateDim == 0u)
		return 0.0f;
	const unsigned int areaRows = (stateDim * (stateDim - 1u)) / 2u;
	const unsigned int featureRows = stateDim + areaRows;
	if (featureRows == 0u)
		return 0.0f;

	std::vector<float> histStates(static_cast<size_t>(lagHorizon) * stateDim * n, 0.0f);
	for (unsigned int slot = 0u; slot < lagHorizon; ++slot)
	{
		const unsigned int histLag = lagHorizon - 1u - slot;
		const size_t activeBase = static_cast<size_t>(histLag) * activeSlice;
		for (unsigned int row = 0; row < activeRank; ++row)
		{
			std::copy(state.resolveGzHistory.begin() + activeBase + static_cast<size_t>(row) * n,
			          state.resolveGzHistory.begin() + activeBase + static_cast<size_t>(row + 1u) * n,
			          histStates.begin() + (static_cast<size_t>(slot) * stateDim + row) * n);
		}
		remove_vectorized_projection(&histStates[(static_cast<size_t>(slot) * stateDim) * n],
		                             &activeGauge[0],
		                             static_cast<size_t>(activeRank) * n,
		                             eps);
		if (effectiveScoutRank > 0u)
		{
			const size_t scoutBase = static_cast<size_t>(histLag) * scoutSlice;
			for (unsigned int row = 0; row < effectiveScoutRank; ++row)
			{
				std::copy(state.heroGwHistory.begin() + scoutBase + static_cast<size_t>(row) * n,
				          state.heroGwHistory.begin() + scoutBase + static_cast<size_t>(row + 1u) * n,
				          histStates.begin()
				              + (static_cast<size_t>(slot) * stateDim + activeRank + row) * n);
			}
			remove_vectorized_projection(
			    &histStates[(static_cast<size_t>(slot) * stateDim + activeRank) * n],
			    &scoutGauge[0],
			    static_cast<size_t>(effectiveScoutRank) * n,
			    eps);
		}
	}

	std::vector<float> featureBlock(static_cast<size_t>(featureRows) * n, 0.0f);
	double firstEnergy = 0.0;
	double areaEnergy = 0.0;
	std::vector<double> totalInc(static_cast<size_t>(stateDim), 0.0);
	std::vector<double> prefix(static_cast<size_t>(stateDim), 0.0);
	std::vector<double> inc(static_cast<size_t>(stateDim), 0.0);
	std::vector<double> area(static_cast<size_t>(areaRows), 0.0);
	for (unsigned int col = 0; col < n; ++col)
	{
		std::fill(totalInc.begin(), totalInc.end(), 0.0);
		std::fill(prefix.begin(), prefix.end(), 0.0);
		std::fill(area.begin(), area.end(), 0.0);
		for (unsigned int step = 1u; step < lagHorizon; ++step)
		{
			for (unsigned int dim = 0u; dim < stateDim; ++dim)
			{
				const double curr =
				    static_cast<double>(histStates[(static_cast<size_t>(step) * stateDim + dim) * n + col]);
				const double prev =
				    static_cast<double>(histStates[(static_cast<size_t>(step - 1u) * stateDim + dim) * n + col]);
				inc[dim] = curr - prev;
			}
			unsigned int pair = 0u;
			for (unsigned int i = 0u; i < stateDim; ++i)
			{
				for (unsigned int j = i + 1u; j < stateDim; ++j)
				{
					area[pair] += 0.5 * (prefix[i] * inc[j] - prefix[j] * inc[i]);
					++pair;
				}
			}
			for (unsigned int dim = 0u; dim < stateDim; ++dim)
			{
				totalInc[dim] += inc[dim];
				prefix[dim] += inc[dim];
			}
		}
		for (unsigned int dim = 0u; dim < stateDim; ++dim)
		{
			const float v = static_cast<float>(totalInc[dim]);
			featureBlock[static_cast<size_t>(dim) * n + col] = v;
			firstEnergy += static_cast<double>(v) * static_cast<double>(v);
		}
		for (unsigned int pair = 0u; pair < areaRows; ++pair)
		{
			const float v = static_cast<float>(area[pair]);
			featureBlock[(static_cast<size_t>(stateDim) + pair) * n + col] = v;
			areaEnergy += static_cast<double>(v) * static_cast<double>(v);
		}
	}
	const double totalEnergy = firstEnergy + areaEnergy;
	const double areaShare =
	    (totalEnergy > static_cast<double>(eps)) ? (areaEnergy / totalEnergy) : 0.0;
	if (areaEnergyOut)
		*areaEnergyOut = static_cast<float>(std::max(0.0, std::min(1.0, areaShare)));

	std::vector<float> futureCov(static_cast<size_t>(activeRank) * activeRank, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = i; j < activeRank; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(futureBlock[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(futureBlock[static_cast<size_t>(j) * n + col]);
			}
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq));
			futureCov[static_cast<size_t>(i) * activeRank + j] = v;
			futureCov[static_cast<size_t>(j) * activeRank + i] = v;
		}
	}

	std::vector<float> featureCov(static_cast<size_t>(featureRows) * featureRows, 0.0f);
	for (unsigned int i = 0; i < featureRows; ++i)
	{
		for (unsigned int j = i; j < featureRows; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(featureBlock[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(featureBlock[static_cast<size_t>(j) * n + col]);
			}
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq));
			featureCov[static_cast<size_t>(i) * featureRows + j] = v;
			featureCov[static_cast<size_t>(j) * featureRows + i] = v;
		}
	}

	std::vector<float> cross(static_cast<size_t>(activeRank) * featureRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int f = 0; f < featureRows; ++f)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(futureBlock[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(featureBlock[static_cast<size_t>(f) * n + col]);
			}
			cross[static_cast<size_t>(i) * featureRows + f] =
			    static_cast<float>((dot / static_cast<double>(n))
			                      * static_cast<double>(statScaleSq));
		}
	}

	std::vector<float> invSqrtFuture;
	std::vector<float> invSqrtFeature;
	if (!build_inv_sqrt_psd(&futureCov[0], activeRank, eps, invSqrtFuture)
	    || !build_inv_sqrt_psd(&featureCov[0], featureRows, eps, invSqrtFeature))
		return 0.0f;

	std::vector<float> temp(static_cast<size_t>(activeRank) * featureRows, 0.0f);
	multiply_left_block(&temp[0], &invSqrtFuture[0], &cross[0], activeRank, activeRank, featureRows);

	std::vector<float> whitened(static_cast<size_t>(activeRank) * featureRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = 0; j < featureRows; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < featureRows; ++k)
			{
				sum += static_cast<double>(temp[static_cast<size_t>(i) * featureRows + k])
				     * static_cast<double>(invSqrtFeature[static_cast<size_t>(k) * featureRows + j]);
			}
			whitened[static_cast<size_t>(i) * featureRows + j] = static_cast<float>(sum);
		}
	}

	std::vector<float> leftWhite;
	std::vector<float> rightWhite;
	float sigma = 0.0f;
	if (!top_singular_triplet(whitened, activeRank, featureRows, &sigma, leftWhite, rightWhite)
	    || !std::isfinite(sigma))
		return 0.0f;
	sigma = std::max(0.0f, std::min(1.0f, sigma));
	if (sigmaOut)
		*sigmaOut = sigma;

	leftModeOut.assign(static_cast<size_t>(activeRank), 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		double sum = 0.0;
		for (unsigned int k = 0; k < activeRank; ++k)
			sum += static_cast<double>(invSqrtFuture[static_cast<size_t>(i) * activeRank + k])
			     * static_cast<double>(leftWhite[k]);
		leftModeOut[i] = static_cast<float>(sum);
	}
	if (!normalize_vector(&leftModeOut[0], activeRank))
		return 0.0f;

	std::vector<float> rightMode(static_cast<size_t>(featureRows), 0.0f);
	for (unsigned int i = 0; i < featureRows; ++i)
	{
		double sum = 0.0;
		for (unsigned int k = 0; k < featureRows; ++k)
			sum += static_cast<double>(invSqrtFeature[static_cast<size_t>(i) * featureRows + k])
			     * static_cast<double>(rightWhite[k]);
		rightMode[i] = static_cast<float>(sum);
	}
	if (!normalize_vector(&rightMode[0], featureRows))
		return 0.0f;

	const double emaKeep =
	    std::min<double>(1.0, std::max<double>(0.0, static_cast<double>(beta)));
	const double emaAdd = 1.0 - emaKeep;
	const float priorPole = atlas_isfinite(state.riftPole) ? state.riftPole : 0.0f;
	latentOut.assign(static_cast<size_t>(n), 0.0f);
	double poleNumerSample = 0.0;
	double poleDenomSample = 0.0;
	for (unsigned int col = 0; col < n; ++col)
	{
		double input = 0.0;
		for (unsigned int row = 0; row < featureRows; ++row)
			input += static_cast<double>(rightMode[row])
			      * static_cast<double>(featureBlock[static_cast<size_t>(row) * n + col]);
		const double oldLatent = static_cast<double>(state.riftLatent[col]);
		const double newLatent = static_cast<double>(priorPole) * oldLatent + input;
		latentOut[col] = static_cast<float>(newLatent);
		poleNumerSample += newLatent * oldLatent;
		poleDenomSample += oldLatent * oldLatent;
	}

	if (n > 0u)
	{
		poleNumerSample /= static_cast<double>(n);
		poleDenomSample /= static_cast<double>(n);
	}
	state.riftPoleNumer = static_cast<float>(
	    emaKeep * static_cast<double>(state.riftPoleNumer) + emaAdd * poleNumerSample);
	state.riftPoleDenom = static_cast<float>(
	    emaKeep * static_cast<double>(state.riftPoleDenom) + emaAdd * poleDenomSample);
	if (state.riftPoleDenom > eps && atlas_isfinite(state.riftPoleDenom))
	{
		double rho = static_cast<double>(state.riftPoleNumer)
		           / static_cast<double>(state.riftPoleDenom);
		const double rhoMax =
		    std::min<double>(0.999, std::max<double>(0.0, static_cast<double>(poleMax)));
		if (!std::isfinite(rho))
			rho = 0.0;
		rho = std::max(-rhoMax, std::min(rhoMax, rho));
		state.riftPole = static_cast<float>(rho);
	}
	else if (!atlas_isfinite(state.riftPole))
	{
		state.riftPole = 0.0f;
	}
	if (poleOut)
		*poleOut = state.riftPole;
	std::copy(leftModeOut.begin(), leftModeOut.end(), state.riftLeftMode.begin());
	std::copy(latentOut.begin(), latentOut.end(), state.riftLatent.begin());

	const double predR2 = std::max(0.0, std::min(1.0, static_cast<double>(sigma) * static_cast<double>(sigma)));
	if (predR2Out)
		*predR2Out = static_cast<float>(predR2);
	double edge = predR2
	            * std::max<double>(0.0, static_cast<double>(horizRatio))
	            * std::sqrt(std::max(0.0, areaShare));
	edge = std::max(0.0, std::min(1.0, edge));
	return std::isfinite(edge) ? static_cast<float>(edge) : 0.0f;
}

static float remove_row_mean_projection(float* mat,
                                        unsigned int rows,
                                        unsigned int cols,
                                        float eps)
{
	if (!mat || rows == 0u || cols == 0u)
		return 1.0f;
	double beforeNormSq = 0.0;
	double afterNormSq = 0.0;
	for (unsigned int col = 0; col < cols; ++col)
	{
		double mean = 0.0;
		for (unsigned int row = 0; row < rows; ++row)
		{
			const double v = static_cast<double>(mat[static_cast<size_t>(row) * cols + col]);
			beforeNormSq += v * v;
			mean += v;
		}
		mean /= static_cast<double>(rows);
		for (unsigned int row = 0; row < rows; ++row)
		{
			const size_t idx = static_cast<size_t>(row) * cols + col;
			mat[idx] = static_cast<float>(static_cast<double>(mat[idx]) - mean);
			const double centered = static_cast<double>(mat[idx]);
			afterNormSq += centered * centered;
		}
	}
	if (!(beforeNormSq > static_cast<double>(eps)))
		return 1.0f;
	const double ratio = afterNormSq / beforeNormSq;
	if (!std::isfinite(ratio))
		return 1.0f;
	return static_cast<float>(std::min<double>(1.0, std::max<double>(0.0, ratio)));
}

static float compute_orbit_lite_mode(WeightState& state,
                                     const float* W,
                                     const float* activeBasisPacked,
                                     const float* gW,
                                     unsigned int activeRank,
                                     unsigned int m,
                                     unsigned int n,
                                     float beta,
                                     float statScaleSq,
                                     float eps,
                                     float poleMax,
                                     float* sigmaOut,
                                     float* poleOut,
                                     float* horizontalRatioOut,
                                     std::vector<float>& leftModeOut,
                                     std::vector<float>& latentOut)
{
	leftModeOut.clear();
	latentOut.clear();
	if (sigmaOut)
		*sigmaOut = 0.0f;
	if (poleOut)
		*poleOut = 0.0f;
	if (horizontalRatioOut)
		*horizontalRatioOut = 1.0f;
	if (!W || !gW || !activeBasisPacked || activeRank == 0u || m <= 1u || n == 0u)
		return 0.0f;
	if (state.orbitPrevSignal.size() < static_cast<size_t>(n)
	    || state.orbitLeftMode.size() < static_cast<size_t>(activeRank)
	    || state.orbitLatent.size() < static_cast<size_t>(n))
		return 0.0f;

	std::vector<float> horizGrad(static_cast<size_t>(m) * n, 0.0f);
	std::copy(gW, gW + static_cast<size_t>(m) * n, horizGrad.begin());
	const float horizontalRatio =
	    remove_row_mean_projection(&horizGrad[0], m, n, eps);
	if (horizontalRatioOut)
		*horizontalRatioOut = horizontalRatio;

	std::vector<float> horizWeight(static_cast<size_t>(m) * n, 0.0f);
	std::copy(W, W + static_cast<size_t>(m) * n, horizWeight.begin());
	remove_row_mean_projection(&horizWeight[0], m, n, eps);

	std::vector<float> numer(static_cast<size_t>(m) * m, 0.0f);
	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int j = i; j < m; ++j)
		{
			double dotG = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				const size_t idxI = static_cast<size_t>(i) * n + col;
				const size_t idxJ = static_cast<size_t>(j) * n + col;
				dotG += static_cast<double>(horizGrad[idxI])
				     * static_cast<double>(horizGrad[idxJ]);
			}
			const float gVal = static_cast<float>(
			    (dotG / static_cast<double>(n)) * static_cast<double>(statScaleSq));
			numer[static_cast<size_t>(i) * m + j] = gVal;
			numer[static_cast<size_t>(j) * m + i] = gVal;
		}
	}

	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(&numer[0], m, eigVec, eigVal);
	if (eigVal.empty())
		return 0.0f;

	double topEig = static_cast<double>(eigVal[0]);
	if (!std::isfinite(topEig) || topEig <= static_cast<double>(eps))
		return 0.0f;

	double trace = 0.0;
	for (unsigned int i = 0; i < m && i < eigVal.size(); ++i)
	{
		const double lambda = static_cast<double>(eigVal[i]);
		if (std::isfinite(lambda) && lambda > 0.0)
			trace += lambda;
	}
	if (!(trace > static_cast<double>(eps)))
		trace = topEig;

	std::vector<float> rowMode(static_cast<size_t>(m), 0.0f);
	for (unsigned int i = 0; i < m; ++i)
		rowMode[i] = eigVec[static_cast<size_t>(i) * m];

	double mean = 0.0;
	for (unsigned int i = 0; i < m; ++i)
		mean += static_cast<double>(rowMode[i]);
	mean /= static_cast<double>(m);
	for (unsigned int i = 0; i < m; ++i)
		rowMode[i] = static_cast<float>(static_cast<double>(rowMode[i]) - mean);
	if (!normalize_vector(&rowMode[0], m))
		return 0.0f;

	leftModeOut.assign(static_cast<size_t>(activeRank), 0.0f);
	double liftNormSq = 0.0;
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		double sum = 0.0;
		for (unsigned int i = 0; i < m; ++i)
		{
			sum += static_cast<double>(activeBasisPacked[static_cast<size_t>(i) * activeRank + c])
			     * static_cast<double>(rowMode[i]);
		}
		leftModeOut[c] = static_cast<float>(sum);
		liftNormSq += sum * sum;
	}
	if (!(liftNormSq > 1e-12) || !normalize_vector(&leftModeOut[0], activeRank))
		return 0.0f;

	std::vector<float> modeSignal(static_cast<size_t>(n), 0.0f);
	for (unsigned int col = 0; col < n; ++col)
	{
		double sum = 0.0;
		for (unsigned int row = 0; row < m; ++row)
		{
			sum += static_cast<double>(rowMode[row])
			     * static_cast<double>(horizGrad[static_cast<size_t>(row) * n + col]);
		}
		modeSignal[col] = static_cast<float>(sum);
	}

	double numerPole = 0.0;
	double denomPole = 0.0;
	for (unsigned int col = 0; col < n; ++col)
	{
		numerPole += static_cast<double>(modeSignal[col])
		           * static_cast<double>(state.orbitPrevSignal[col]);
		denomPole += static_cast<double>(state.orbitPrevSignal[col])
		           * static_cast<double>(state.orbitPrevSignal[col]);
	}
	numerPole /= static_cast<double>(n);
	denomPole /= static_cast<double>(n);
	state.orbitPoleNumer =
	    beta * state.orbitPoleNumer + (1.0f - beta) * static_cast<float>(numerPole);
	state.orbitPoleDenom =
	    beta * state.orbitPoleDenom + (1.0f - beta) * static_cast<float>(denomPole);
	float pole = 0.0f;
	if (state.orbitPoleDenom > eps)
		pole = state.orbitPoleNumer / (state.orbitPoleDenom + eps);
	if (pole > poleMax) pole = poleMax;
	if (pole < -poleMax) pole = -poleMax;
	if (!atlas_isfinite(pole))
		pole = 0.0f;
	state.orbitPole = pole;
	if (poleOut)
		*poleOut = pole;

	latentOut.assign(static_cast<size_t>(n), 0.0f);
	for (unsigned int col = 0; col < n; ++col)
	{
		// ORBIT-Lite is memory-only: use the prior quotient signal to drive the
		// latent observer, then shift in the current signal for the next step.
		latentOut[col] =
		    pole * state.orbitLatent[col] + state.orbitPrevSignal[col];
	}
	std::copy(modeSignal.begin(), modeSignal.end(), state.orbitPrevSignal.begin());
	std::copy(leftModeOut.begin(), leftModeOut.end(), state.orbitLeftMode.begin());
	std::copy(latentOut.begin(), latentOut.end(), state.orbitLatent.begin());

	const double topShare = topEig / std::max<double>(static_cast<double>(eps), trace);
	const double isotropicShare = 1.0 / static_cast<double>(m);
	double boundedSigma = 0.0;
	if (topShare > isotropicShare)
	{
		boundedSigma =
		    (topShare - isotropicShare)
		    / std::max<double>(1e-12, 1.0 - isotropicShare);
	}
	boundedSigma = std::max(0.0, std::min(1.0, boundedSigma));
	if (sigmaOut)
		*sigmaOut = static_cast<float>(boundedSigma);
	const double liftShare = std::min<double>(1.0, std::max<double>(0.0, liftNormSq));
	const double edge = boundedSigma
	                  * static_cast<double>(horizontalRatio)
	                  * liftShare;
	return std::isfinite(edge) ? static_cast<float>(edge) : 0.0f;
}

static float compute_sparrow_streaming_mode(WeightState& state,
                                            const float* W,
                                            const float* activeBasisPacked,
                                            const std::vector<float>& gz,
                                            const std::vector<float>& gwScout,
                                            unsigned int probeRank,
                                            unsigned int activeRank,
                                            unsigned int m,
                                            unsigned int n,
                                            float beta,
                                            float statScaleSq,
                                            float eps,
                                            float poleMax,
                                            unsigned int modeRankRequested,
                                            bool autoModeGate,
                                            float secondEdgeThreshold,
                                            float secondEdgeFraction,
                                            float* sigmaOut,
                                            float* secondSigmaOut,
                                            float* secondEdgeOut,
                                            float* poleOut,
                                            float* horizontalRatioOut,
                                            unsigned int* activeModesOut,
                                            std::vector<float>& leftModeOut,
                                            std::vector<float>& latentOut)
{
	leftModeOut.clear();
	latentOut.clear();
	if (sigmaOut)
		*sigmaOut = 0.0f;
	if (secondSigmaOut)
		*secondSigmaOut = 0.0f;
	if (secondEdgeOut)
		*secondEdgeOut = 0.0f;
	if (poleOut)
		*poleOut = 0.0f;
	if (horizontalRatioOut)
		*horizontalRatioOut = 1.0f;
	if (activeModesOut)
		*activeModesOut = 0u;
	if (activeRank == 0u || n == 0u || !W || !activeBasisPacked
	    || gz.size() < static_cast<size_t>(activeRank) * static_cast<size_t>(n))
		return 0.0f;

	const unsigned int modeRank = atlas_sparrow_mode_rank(modeRankRequested);
	const unsigned int storageRank = state.complementRank;
	if (probeRank > storageRank)
		probeRank = storageRank;

	const size_t rn = static_cast<size_t>(state.r) * static_cast<size_t>(n);
	const size_t cn = static_cast<size_t>(storageRank) * static_cast<size_t>(n);
	const unsigned int fullPastDim = state.r + storageRank;
	if (state.sparrowPrevActive.size() < rn
	    || state.sparrowPrevScout.size() < cn
	    || state.sparrowFutureCov.size() < static_cast<size_t>(state.r) * state.r
	    || state.sparrowPastCov.size() < static_cast<size_t>(fullPastDim) * fullPastDim
	    || state.sparrowCrossCov.size() < static_cast<size_t>(state.r) * fullPastDim
	    || state.sparrowLeftMode.size() < static_cast<size_t>(modeRank) * state.r
	    || state.sparrowRightMode.size() < static_cast<size_t>(modeRank) * fullPastDim
	    || state.sparrowLatent.size() < static_cast<size_t>(modeRank) * n)
		return 0.0f;

	std::vector<float>& horizActive = state.scratch_sparrowActive;
	std::vector<float>& horizScout = state.scratch_sparrowScout;
	std::fill(horizActive.begin(), horizActive.end(), 0.0f);
	std::fill(horizScout.begin(), horizScout.end(), 0.0f);
	std::copy(gz.begin(), gz.begin() + static_cast<size_t>(activeRank) * n, horizActive.begin());

	std::vector<float> activeGauge(static_cast<size_t>(activeRank) * n, 0.0f);
	glades::gemm::atb(&activeGauge[0], activeBasisPacked, W, activeRank, m, n, 1.0f);
	const float horizRatio =
	    remove_vectorized_projection(&horizActive[0], &activeGauge[0],
	                                 static_cast<size_t>(activeRank) * n, eps);
	if (horizontalRatioOut)
		*horizontalRatioOut = horizRatio;

	if (probeRank > 0u && gwScout.size() >= static_cast<size_t>(probeRank) * n)
	{
		std::copy(gwScout.begin(), gwScout.begin() + static_cast<size_t>(probeRank) * n,
		          horizScout.begin());
		std::vector<float> scoutBasisPacked(static_cast<size_t>(m) * probeRank, 0.0f);
		pack_active_basis(state.scoutBasis, storageRank, m, probeRank, scoutBasisPacked);
		std::vector<float> scoutGauge(static_cast<size_t>(probeRank) * n, 0.0f);
		glades::gemm::atb(&scoutGauge[0], &scoutBasisPacked[0], W, probeRank, m, n, 1.0f);
		remove_vectorized_projection(&horizScout[0], &scoutGauge[0],
		                             static_cast<size_t>(probeRank) * n, eps);
	}
	else
	{
		probeRank = 0u;
	}

	const unsigned int pastRows = activeRank + probeRank;
	if (pastRows == 0u)
		return 0.0f;

	std::vector<float> pastBlock(static_cast<size_t>(pastRows) * n, 0.0f);
	for (unsigned int row = 0; row < activeRank; ++row)
	{
		std::copy(state.sparrowPrevActive.begin() + static_cast<size_t>(row) * n,
		          state.sparrowPrevActive.begin() + static_cast<size_t>(row + 1u) * n,
		          pastBlock.begin() + static_cast<size_t>(row) * n);
	}
	for (unsigned int row = 0; row < probeRank; ++row)
	{
		std::copy(state.sparrowPrevScout.begin() + static_cast<size_t>(row) * n,
		          state.sparrowPrevScout.begin() + static_cast<size_t>(row + 1u) * n,
		          pastBlock.begin() + static_cast<size_t>(activeRank + row) * n);
	}

	const double emaKeep =
	    std::min<double>(1.0, std::max<double>(0.0, static_cast<double>(beta)));
	const double emaAdd = 1.0 - emaKeep;
	const double normScale = (n > 0u)
	    ? (static_cast<double>(statScaleSq) / static_cast<double>(n))
	    : 0.0;

	std::vector<float> futureCovSmall(static_cast<size_t>(activeRank) * activeRank, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = i; j < activeRank; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(horizActive[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(horizActive[static_cast<size_t>(j) * n + col]);
			}
			const float sample = static_cast<float>(dot * normScale);
			const size_t fullIdx0 = static_cast<size_t>(i) * state.r + j;
			const size_t fullIdx1 = static_cast<size_t>(j) * state.r + i;
			const float updated = static_cast<float>(
			    emaKeep * static_cast<double>(state.sparrowFutureCov[fullIdx0])
			    + emaAdd * static_cast<double>(sample));
			state.sparrowFutureCov[fullIdx0] = updated;
			state.sparrowFutureCov[fullIdx1] = updated;
			futureCovSmall[static_cast<size_t>(i) * activeRank + j] = updated;
			futureCovSmall[static_cast<size_t>(j) * activeRank + i] = updated;
		}
	}

	std::vector<float> pastCovSmall(static_cast<size_t>(pastRows) * pastRows, 0.0f);
	for (unsigned int i = 0; i < pastRows; ++i)
	{
		const unsigned int fullI = (i < activeRank) ? i : (state.r + (i - activeRank));
		for (unsigned int j = i; j < pastRows; ++j)
		{
			const unsigned int fullJ = (j < activeRank) ? j : (state.r + (j - activeRank));
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(pastBlock[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(pastBlock[static_cast<size_t>(j) * n + col]);
			}
			const float sample = static_cast<float>(dot * normScale);
			const size_t fullIdx0 = static_cast<size_t>(fullI) * fullPastDim + fullJ;
			const size_t fullIdx1 = static_cast<size_t>(fullJ) * fullPastDim + fullI;
			const float updated = static_cast<float>(
			    emaKeep * static_cast<double>(state.sparrowPastCov[fullIdx0])
			    + emaAdd * static_cast<double>(sample));
			state.sparrowPastCov[fullIdx0] = updated;
			state.sparrowPastCov[fullIdx1] = updated;
			pastCovSmall[static_cast<size_t>(i) * pastRows + j] = updated;
			pastCovSmall[static_cast<size_t>(j) * pastRows + i] = updated;
		}
	}

	std::vector<float> crossSmall(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = 0; j < pastRows; ++j)
		{
			const unsigned int fullJ = (j < activeRank) ? j : (state.r + (j - activeRank));
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(horizActive[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(pastBlock[static_cast<size_t>(j) * n + col]);
			}
			const float sample = static_cast<float>(dot * normScale);
			const size_t fullIdx = static_cast<size_t>(i) * fullPastDim + fullJ;
			const float updated = static_cast<float>(
			    emaKeep * static_cast<double>(state.sparrowCrossCov[fullIdx])
			    + emaAdd * static_cast<double>(sample));
			state.sparrowCrossCov[fullIdx] = updated;
			crossSmall[static_cast<size_t>(i) * pastRows + j] = updated;
		}
	}

	std::vector<float> invSqrtFuture;
	std::vector<float> invSqrtPast;
	if (!build_inv_sqrt_psd(&futureCovSmall[0], activeRank, eps, invSqrtFuture)
	    || !build_inv_sqrt_psd(&pastCovSmall[0], pastRows, eps, invSqrtPast))
	{
		std::fill(state.sparrowPrevActive.begin(), state.sparrowPrevActive.end(), 0.0f);
		std::copy(horizActive.begin(), horizActive.begin() + static_cast<size_t>(activeRank) * n,
		          state.sparrowPrevActive.begin());
		std::fill(state.sparrowPrevScout.begin(), state.sparrowPrevScout.end(), 0.0f);
		if (probeRank > 0u)
		{
			std::copy(horizScout.begin(), horizScout.begin() + static_cast<size_t>(probeRank) * n,
			          state.sparrowPrevScout.begin());
		}
		return 0.0f;
	}

	std::vector<float> temp(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	multiply_left_block(&temp[0], &invSqrtFuture[0], &crossSmall[0],
	                    activeRank, activeRank, pastRows);

	std::vector<float> whitened(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = 0; j < pastRows; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < pastRows; ++k)
			{
				sum += static_cast<double>(temp[static_cast<size_t>(i) * pastRows + k])
				     * static_cast<double>(invSqrtPast[static_cast<size_t>(k) * pastRows + j]);
			}
			whitened[static_cast<size_t>(i) * pastRows + j] = static_cast<float>(sum);
		}
	}

	std::fill(state.sparrowLeftMode.begin(), state.sparrowLeftMode.end(), 0.0f);
	std::fill(state.sparrowRightMode.begin(), state.sparrowRightMode.end(), 0.0f);
	leftModeOut.assign(static_cast<size_t>(modeRank) * state.r, 0.0f);
	latentOut.assign(static_cast<size_t>(modeRank) * n, 0.0f);
	std::vector<float> sigmaModes;
	std::vector<float> leftWhiteModes;
	std::vector<float> rightWhiteModes;
	const unsigned int foundModes =
	    top_singular_modes(whitened, activeRank, pastRows, modeRank,
	                       sigmaModes, leftWhiteModes, rightWhiteModes);
	const double horizEdgeScale =
	    std::max<double>(0.0, static_cast<double>(horizRatio));
	float sigma = 0.0f;
	float secondSigma = 0.0f;
	float secondEdge = 0.0f;
	bool modeEnabled[kATLASSparrowMaxModeRank] = { false, false };
	if (foundModes > 0u && sigmaModes[0] > 0.0f && std::isfinite(sigmaModes[0]))
	{
		modeEnabled[0] = true;
		sigma = sigmaModes[0];
	}
	if (foundModes > 1u && modeRank > 1u
	    && sigmaModes[1] > 0.0f && std::isfinite(sigmaModes[1]))
	{
		secondSigma = sigmaModes[1];
		secondEdge = static_cast<float>(
		    std::max<double>(0.0, static_cast<double>(secondSigma)) * horizEdgeScale);
		bool allowSecond = true;
		if (autoModeGate)
		{
			const double threshold =
			    std::max<double>(0.0, static_cast<double>(secondEdgeThreshold));
			const double fraction =
			    std::min<double>(1.0, std::max<double>(0.0, static_cast<double>(secondEdgeFraction)));
			const double mode1Edge =
			    std::max<double>(0.0, static_cast<double>(sigma)) * horizEdgeScale;
			allowSecond =
			    (static_cast<double>(secondEdge) >= threshold)
			    && (static_cast<double>(secondEdge)
			        >= fraction * std::max<double>(1e-12, mode1Edge));
		}
		modeEnabled[1] = allowSecond;
	}
	double sigmaAccumSq = 0.0;
	unsigned int retainedModes = 0u;
	for (unsigned int mode = 0; mode < foundModes; ++mode)
	{
		if (mode >= kATLASSparrowMaxModeRank || !modeEnabled[mode])
			continue;
		if (!(sigmaModes[mode] > 0.0f) || !std::isfinite(sigmaModes[mode]))
			continue;

		std::vector<float> leftModeSmall(static_cast<size_t>(activeRank), 0.0f);
		for (unsigned int i = 0; i < activeRank; ++i)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < activeRank; ++k)
			{
				sum += static_cast<double>(invSqrtFuture[static_cast<size_t>(i) * activeRank + k])
				     * static_cast<double>(leftWhiteModes[static_cast<size_t>(mode) * activeRank + k]);
			}
			leftModeSmall[i] = static_cast<float>(sum);
		}
		if (!normalize_vector(&leftModeSmall[0], activeRank))
			continue;
		for (unsigned int i = 0; i < activeRank; ++i)
		{
			leftModeOut[static_cast<size_t>(mode) * state.r + i] = leftModeSmall[i];
			state.sparrowLeftMode[static_cast<size_t>(mode) * state.r + i] = leftModeSmall[i];
		}

		std::vector<float> rightModeSmall(static_cast<size_t>(pastRows), 0.0f);
		for (unsigned int i = 0; i < pastRows; ++i)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < pastRows; ++k)
			{
				sum += static_cast<double>(invSqrtPast[static_cast<size_t>(i) * pastRows + k])
				     * static_cast<double>(rightWhiteModes[static_cast<size_t>(mode) * pastRows + k]);
			}
			rightModeSmall[i] = static_cast<float>(sum);
		}
		if (!normalize_vector(&rightModeSmall[0], pastRows))
		{
			for (unsigned int i = 0; i < activeRank; ++i)
			{
				leftModeOut[static_cast<size_t>(mode) * state.r + i] = 0.0f;
				state.sparrowLeftMode[static_cast<size_t>(mode) * state.r + i] = 0.0f;
			}
			continue;
		}
		const size_t rightBase = static_cast<size_t>(mode) * fullPastDim;
		for (unsigned int i = 0; i < activeRank; ++i)
			state.sparrowRightMode[rightBase + i] = rightModeSmall[i];
		for (unsigned int i = 0; i < probeRank; ++i)
			state.sparrowRightMode[rightBase + state.r + i] = rightModeSmall[activeRank + i];
		if (!(sigma > 0.0f))
			sigma = sigmaModes[mode];
		sigmaAccumSq += static_cast<double>(sigmaModes[mode]) * static_cast<double>(sigmaModes[mode]);
		retainedModes += 1u;
	}
	if (sigmaOut)
		*sigmaOut = sigma;
	if (secondSigmaOut)
		*secondSigmaOut = secondSigma;
	if (secondEdgeOut)
		*secondEdgeOut = secondEdge;
	if (activeModesOut)
		*activeModesOut = retainedModes;

	const float priorPole =
	    atlas_isfinite(state.sparrowPole) ? state.sparrowPole : 0.0f;
	double poleNumerSample = 0.0;
	double poleDenomSample = 0.0;
	for (unsigned int mode = 0; mode < modeRank; ++mode)
	{
		if (mode >= kATLASSparrowMaxModeRank || !modeEnabled[mode])
			continue;
		const size_t rightBase = static_cast<size_t>(mode) * fullPastDim;
		const size_t latentBase = static_cast<size_t>(mode) * n;
		for (unsigned int j = 0; j < n; ++j)
		{
			double input = 0.0;
			for (unsigned int i = 0; i < activeRank; ++i)
			{
				input += static_cast<double>(state.sparrowRightMode[rightBase + i])
				     * static_cast<double>(pastBlock[static_cast<size_t>(i) * n + j]);
			}
			for (unsigned int i = 0; i < probeRank; ++i)
			{
				input += static_cast<double>(state.sparrowRightMode[rightBase + state.r + i])
				     * static_cast<double>(pastBlock[static_cast<size_t>(activeRank + i) * n + j]);
			}
			const double oldLatent =
			    static_cast<double>(state.sparrowLatent[latentBase + j]);
			const double newLatent =
			    static_cast<double>(priorPole) * oldLatent + input;
			latentOut[latentBase + j] = static_cast<float>(newLatent);
			poleNumerSample += newLatent * oldLatent;
			poleDenomSample += oldLatent * oldLatent;
		}
	}

	if (n > 0u && retainedModes > 0u)
	{
		const double denom = static_cast<double>(n) * static_cast<double>(retainedModes);
		poleNumerSample /= denom;
		poleDenomSample /= denom;
	}
	state.sparrowPoleNumer = static_cast<float>(
	    emaKeep * static_cast<double>(state.sparrowPoleNumer) + emaAdd * poleNumerSample);
	state.sparrowPoleDenom = static_cast<float>(
	    emaKeep * static_cast<double>(state.sparrowPoleDenom) + emaAdd * poleDenomSample);
	if (state.sparrowPoleDenom > eps && atlas_isfinite(state.sparrowPoleDenom))
	{
		double rho = static_cast<double>(state.sparrowPoleNumer)
		           / static_cast<double>(state.sparrowPoleDenom);
		const double rhoMax =
		    std::min<double>(0.999, std::max<double>(0.0, static_cast<double>(poleMax)));
		if (!std::isfinite(rho))
			rho = 0.0;
		rho = std::max(-rhoMax, std::min(rhoMax, rho));
		state.sparrowPole = static_cast<float>(rho);
	}
	else if (!atlas_isfinite(state.sparrowPole))
	{
		state.sparrowPole = 0.0f;
	}
	if (poleOut)
		*poleOut = state.sparrowPole;
	std::copy(latentOut.begin(), latentOut.end(), state.sparrowLatent.begin());

	std::fill(state.sparrowPrevActive.begin(), state.sparrowPrevActive.end(), 0.0f);
	std::copy(horizActive.begin(), horizActive.begin() + static_cast<size_t>(activeRank) * n,
	          state.sparrowPrevActive.begin());
	std::fill(state.sparrowPrevScout.begin(), state.sparrowPrevScout.end(), 0.0f);
	if (probeRank > 0u)
	{
		std::copy(horizScout.begin(), horizScout.begin() + static_cast<size_t>(probeRank) * n,
		          state.sparrowPrevScout.begin());
	}

	double edge = std::sqrt(std::max<double>(0.0, sigmaAccumSq))
	            * horizEdgeScale;
	if (!std::isfinite(edge))
		edge = 0.0;
	edge = std::max(0.0, std::min(1.0, edge));
	return static_cast<float>(edge);
}

static float compute_cobalt_transfer_edge(const WeightState& state,
                                          const std::vector<float>& gz,
                                          unsigned int scoutRank,
                                          unsigned int activeRank,
                                          unsigned int n,
                                          unsigned int lagHorizon,
                                          float statScaleSq,
                                          float eps,
                                          float* sigmaOut)
{
	if (sigmaOut)
		*sigmaOut = 0.0f;
	if (activeRank == 0u || n == 0u || lagHorizon == 0u
	    || gz.size() < static_cast<size_t>(activeRank) * static_cast<size_t>(n))
		return 0.0f;

	const size_t activeSlice = static_cast<size_t>(state.r) * static_cast<size_t>(n);
	if (state.resolveGzHistory.size() < static_cast<size_t>(lagHorizon) * activeSlice)
		return 0.0f;

	unsigned int effectiveScoutRank = scoutRank;
	const unsigned int storageRank = state.complementRank;
	const size_t scoutSlice = static_cast<size_t>(storageRank) * static_cast<size_t>(n);
	if (effectiveScoutRank > 0u
	    && state.heroGwHistory.size() < static_cast<size_t>(lagHorizon) * scoutSlice)
		effectiveScoutRank = 0u;

	const unsigned int blockRows = activeRank + effectiveScoutRank;
	if (blockRows == 0u)
		return 0.0f;
	const unsigned int pastRows = lagHorizon * blockRows;

	std::vector<const float*> pastPtrs(static_cast<size_t>(pastRows), 0);
	std::vector<float> pastScales(static_cast<size_t>(pastRows), 0.0f);
	for (unsigned int lag = 0; lag < lagHorizon; ++lag)
	{
		const float lagScale =
		    static_cast<float>(1.0 / std::sqrt(static_cast<double>(lag + 1u)));
		const size_t activeBase = static_cast<size_t>(lag) * activeSlice;
		const size_t scoutBase = static_cast<size_t>(lag) * scoutSlice;
		const unsigned int lagOffset = lag * blockRows;
		for (unsigned int row = 0; row < activeRank; ++row)
		{
			const unsigned int idx = lagOffset + row;
			pastPtrs[idx] = &state.resolveGzHistory[activeBase + static_cast<size_t>(row) * n];
			pastScales[idx] = lagScale;
		}
		for (unsigned int row = 0; row < effectiveScoutRank; ++row)
		{
			const unsigned int idx = lagOffset + activeRank + row;
			pastPtrs[idx] = &state.heroGwHistory[scoutBase + static_cast<size_t>(row) * n];
			pastScales[idx] = lagScale;
		}
	}

	std::vector<float> activeCov(static_cast<size_t>(activeRank) * activeRank, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = i; j < activeRank; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(gz[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(gz[static_cast<size_t>(j) * n + col]);
			}
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq));
			activeCov[static_cast<size_t>(i) * activeRank + j] = v;
			activeCov[static_cast<size_t>(j) * activeRank + i] = v;
		}
	}

	std::vector<float> cross(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int p = 0; p < pastRows; ++p)
		{
			const float* hist = pastPtrs[p];
			if (!hist)
				continue;
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(gz[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(hist[col]);
			}
			cross[static_cast<size_t>(i) * pastRows + p] =
			    static_cast<float>((dot / static_cast<double>(n))
			                      * static_cast<double>(statScaleSq)
			                      * static_cast<double>(pastScales[p]));
		}
	}

	std::vector<float> pastCov(static_cast<size_t>(pastRows) * pastRows, 0.0f);
	for (unsigned int i = 0; i < pastRows; ++i)
	{
		const float* histI = pastPtrs[i];
		if (!histI)
			continue;
		for (unsigned int j = i; j < pastRows; ++j)
		{
			const float* histJ = pastPtrs[j];
			if (!histJ)
				continue;
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
				dot += static_cast<double>(histI[col]) * static_cast<double>(histJ[col]);
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq)
			                                  * static_cast<double>(pastScales[i])
			                                  * static_cast<double>(pastScales[j]));
			pastCov[static_cast<size_t>(i) * pastRows + j] = v;
			pastCov[static_cast<size_t>(j) * pastRows + i] = v;
		}
	}

	std::vector<float> invSqrtActive;
	std::vector<float> invSqrtPast;
	if (!build_inv_sqrt_psd(&activeCov[0], activeRank, eps, invSqrtActive)
	    || !build_inv_sqrt_psd(&pastCov[0], pastRows, eps, invSqrtPast))
		return 0.0f;

	std::vector<float> temp(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	multiply_left_block(&temp[0], &invSqrtActive[0], &cross[0], activeRank, activeRank, pastRows);

	std::vector<float> whitened(static_cast<size_t>(activeRank) * pastRows, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = 0; j < pastRows; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < pastRows; ++k)
			{
				sum += static_cast<double>(temp[static_cast<size_t>(i) * pastRows + k])
				     * static_cast<double>(invSqrtPast[static_cast<size_t>(k) * pastRows + j]);
			}
			whitened[static_cast<size_t>(i) * pastRows + j] = static_cast<float>(sum);
		}
	}

	std::vector<float> gram(static_cast<size_t>(activeRank) * activeRank, 0.0f);
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		for (unsigned int j = i; j < activeRank; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < pastRows; ++k)
			{
				sum += static_cast<double>(whitened[static_cast<size_t>(i) * pastRows + k])
				     * static_cast<double>(whitened[static_cast<size_t>(j) * pastRows + k]);
			}
			gram[static_cast<size_t>(i) * activeRank + j] = static_cast<float>(sum);
			gram[static_cast<size_t>(j) * activeRank + i] = static_cast<float>(sum);
		}
	}
	symmetrize_block(&gram[0], activeRank);

	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(&gram[0], activeRank, eigVec, eigVal);
	double topEig = (!eigVal.empty()) ? static_cast<double>(eigVal[0]) : 0.0;
	if (!std::isfinite(topEig) || topEig < 0.0)
		topEig = 0.0;
	if (sigmaOut)
		*sigmaOut = static_cast<float>(std::sqrt(topEig));

	double trace = 0.0;
	for (unsigned int i = 0; i < activeRank; ++i)
	{
		const double d = static_cast<double>(gram[static_cast<size_t>(i) * activeRank + i]);
		if (std::isfinite(d) && d > 0.0)
			trace += d;
	}
	const double bulk = std::max<double>(static_cast<double>(eps),
	                                     trace / static_cast<double>(activeRank));
	const double edge = topEig / bulk - 1.0;
	return static_cast<float>(std::isfinite(edge) ? edge : 0.0);
}

static float compute_birch_hankel_edge(const WeightState& state,
                                       const std::vector<float>& gz,
                                       unsigned int scoutRank,
                                       unsigned int activeRank,
                                       unsigned int n,
                                       unsigned int pastHorizon,
                                       unsigned int futureHorizon,
                                       float statScaleSq,
                                       float eps,
                                       float* sigmaOut)
{
	if (sigmaOut)
		*sigmaOut = 0.0f;
	if (activeRank == 0u || n == 0u || pastHorizon == 0u || futureHorizon == 0u
	    || gz.size() < static_cast<size_t>(activeRank) * static_cast<size_t>(n))
		return 0.0f;

	const size_t activeSlice = static_cast<size_t>(state.r) * static_cast<size_t>(n);
	const unsigned int requiredHistorySlices = futureHorizon + pastHorizon - 1u;
	if (state.resolveGzHistory.size() < static_cast<size_t>(requiredHistorySlices) * activeSlice)
		return 0.0f;

	unsigned int effectiveScoutRank = scoutRank;
	const unsigned int storageRank = state.complementRank;
	const size_t scoutSlice = static_cast<size_t>(storageRank) * static_cast<size_t>(n);
	if (effectiveScoutRank > 0u
	    && state.heroGwHistory.size() < static_cast<size_t>(requiredHistorySlices) * scoutSlice)
		effectiveScoutRank = 0u;

	const unsigned int futureRows = futureHorizon * activeRank;
	const unsigned int pastRows = pastHorizon * (activeRank + effectiveScoutRank);
	if (futureRows == 0u || pastRows == 0u)
		return 0.0f;

	std::vector<const float*> futurePtrs(static_cast<size_t>(futureRows), 0);
	std::vector<float> futureScales(static_cast<size_t>(futureRows), 0.0f);
	for (unsigned int fh = 0; fh < futureHorizon; ++fh)
	{
		const float lagScale =
		    static_cast<float>(1.0 / std::sqrt(static_cast<double>(fh + 1u)));
		for (unsigned int row = 0; row < activeRank; ++row)
		{
			const unsigned int idx = fh * activeRank + row;
			if (fh == 0u)
				futurePtrs[idx] = &gz[static_cast<size_t>(row) * n];
			else
				futurePtrs[idx] =
				    &state.resolveGzHistory[(static_cast<size_t>(fh) - 1u) * activeSlice
				                            + static_cast<size_t>(row) * n];
			futureScales[idx] = lagScale;
		}
	}

	std::vector<const float*> pastPtrs(static_cast<size_t>(pastRows), 0);
	std::vector<float> pastScales(static_cast<size_t>(pastRows), 0.0f);
	for (unsigned int ph = 0; ph < pastHorizon; ++ph)
	{
		const unsigned int histIdx = futureHorizon - 1u + ph;
		const float lagScale =
		    static_cast<float>(1.0 / std::sqrt(static_cast<double>(histIdx + 1u)));
		const size_t activeBase = static_cast<size_t>(histIdx) * activeSlice;
		const size_t scoutBase = static_cast<size_t>(histIdx) * scoutSlice;
		const unsigned int blockOffset = ph * (activeRank + effectiveScoutRank);
		for (unsigned int row = 0; row < activeRank; ++row)
		{
			const unsigned int idx = blockOffset + row;
			pastPtrs[idx] = &state.resolveGzHistory[activeBase + static_cast<size_t>(row) * n];
			pastScales[idx] = lagScale;
		}
		for (unsigned int row = 0; row < effectiveScoutRank; ++row)
		{
			const unsigned int idx = blockOffset + activeRank + row;
			pastPtrs[idx] = &state.heroGwHistory[scoutBase + static_cast<size_t>(row) * n];
			pastScales[idx] = lagScale;
		}
	}

	std::vector<float> futureCov(static_cast<size_t>(futureRows) * futureRows, 0.0f);
	for (unsigned int i = 0; i < futureRows; ++i)
	{
		const float* rowI = futurePtrs[i];
		if (!rowI)
			continue;
		for (unsigned int j = i; j < futureRows; ++j)
		{
			const float* rowJ = futurePtrs[j];
			if (!rowJ)
				continue;
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
				dot += static_cast<double>(rowI[col]) * static_cast<double>(rowJ[col]);
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq)
			                                  * static_cast<double>(futureScales[i])
			                                  * static_cast<double>(futureScales[j]));
			futureCov[static_cast<size_t>(i) * futureRows + j] = v;
			futureCov[static_cast<size_t>(j) * futureRows + i] = v;
		}
	}

	std::vector<float> cross(static_cast<size_t>(futureRows) * pastRows, 0.0f);
	for (unsigned int i = 0; i < futureRows; ++i)
	{
		const float* rowI = futurePtrs[i];
		if (!rowI)
			continue;
		for (unsigned int p = 0; p < pastRows; ++p)
		{
			const float* rowP = pastPtrs[p];
			if (!rowP)
				continue;
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
				dot += static_cast<double>(rowI[col]) * static_cast<double>(rowP[col]);
			cross[static_cast<size_t>(i) * pastRows + p] =
			    static_cast<float>((dot / static_cast<double>(n))
			                      * static_cast<double>(statScaleSq)
			                      * static_cast<double>(futureScales[i])
			                      * static_cast<double>(pastScales[p]));
		}
	}

	std::vector<float> pastCov(static_cast<size_t>(pastRows) * pastRows, 0.0f);
	for (unsigned int i = 0; i < pastRows; ++i)
	{
		const float* rowI = pastPtrs[i];
		if (!rowI)
			continue;
		for (unsigned int j = i; j < pastRows; ++j)
		{
			const float* rowJ = pastPtrs[j];
			if (!rowJ)
				continue;
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
				dot += static_cast<double>(rowI[col]) * static_cast<double>(rowJ[col]);
			const float v = static_cast<float>((dot / static_cast<double>(n))
			                                  * static_cast<double>(statScaleSq)
			                                  * static_cast<double>(pastScales[i])
			                                  * static_cast<double>(pastScales[j]));
			pastCov[static_cast<size_t>(i) * pastRows + j] = v;
			pastCov[static_cast<size_t>(j) * pastRows + i] = v;
		}
	}

	std::vector<float> invSqrtFuture;
	std::vector<float> invSqrtPast;
	if (!build_inv_sqrt_psd(&futureCov[0], futureRows, eps, invSqrtFuture)
	    || !build_inv_sqrt_psd(&pastCov[0], pastRows, eps, invSqrtPast))
		return 0.0f;

	std::vector<float> temp(static_cast<size_t>(futureRows) * pastRows, 0.0f);
	multiply_left_block(&temp[0], &invSqrtFuture[0], &cross[0], futureRows, futureRows, pastRows);

	std::vector<float> whitened(static_cast<size_t>(futureRows) * pastRows, 0.0f);
	for (unsigned int i = 0; i < futureRows; ++i)
	{
		for (unsigned int j = 0; j < pastRows; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < pastRows; ++k)
			{
				sum += static_cast<double>(temp[static_cast<size_t>(i) * pastRows + k])
				     * static_cast<double>(invSqrtPast[static_cast<size_t>(k) * pastRows + j]);
			}
			whitened[static_cast<size_t>(i) * pastRows + j] = static_cast<float>(sum);
		}
	}

	std::vector<float> gram(static_cast<size_t>(futureRows) * futureRows, 0.0f);
	for (unsigned int i = 0; i < futureRows; ++i)
	{
		for (unsigned int j = i; j < futureRows; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < pastRows; ++k)
			{
				sum += static_cast<double>(whitened[static_cast<size_t>(i) * pastRows + k])
				     * static_cast<double>(whitened[static_cast<size_t>(j) * pastRows + k]);
			}
			gram[static_cast<size_t>(i) * futureRows + j] = static_cast<float>(sum);
			gram[static_cast<size_t>(j) * futureRows + i] = static_cast<float>(sum);
		}
	}
	symmetrize_block(&gram[0], futureRows);

	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(&gram[0], futureRows, eigVec, eigVal);
	double topEig = (!eigVal.empty()) ? static_cast<double>(eigVal[0]) : 0.0;
	if (!std::isfinite(topEig) || topEig < 0.0)
		topEig = 0.0;
	if (sigmaOut)
		*sigmaOut = static_cast<float>(std::sqrt(topEig));

	double trace = 0.0;
	for (unsigned int i = 0; i < futureRows; ++i)
	{
		const double d = static_cast<double>(gram[static_cast<size_t>(i) * futureRows + i]);
		if (std::isfinite(d) && d > 0.0)
			trace += d;
	}
	const double bulk =
	    std::max<double>(static_cast<double>(eps),
	                     trace / static_cast<double>(futureRows));
	const double edge = topEig / bulk - 1.0;
	return static_cast<float>(std::isfinite(edge) ? edge : 0.0);
}

static float compute_hero_hankel_edge(const WeightState& state,
                                      const std::vector<float>& gz,
                                      unsigned int scoutRank,
                                      unsigned int activeRank,
                                      unsigned int n,
                                      unsigned int lagHorizon,
                                      float statScaleSq,
                                      float tailMean,
                                      float eps,
                                      float* sigmaOut)
{
	if (sigmaOut)
		*sigmaOut = 0.0f;
	const unsigned int storageRank = state.complementRank;
	if (scoutRank == 0u || activeRank == 0u || storageRank == 0u || n == 0u
	    || lagHorizon == 0u
	    || gz.size() < static_cast<size_t>(activeRank) * static_cast<size_t>(n))
		return 0.0f;

	const size_t scoutSlice = static_cast<size_t>(storageRank) * static_cast<size_t>(n);
	if (state.heroGwHistory.size() < static_cast<size_t>(lagHorizon) * scoutSlice)
		return 0.0f;
	if (state.scoutCov.size() < static_cast<size_t>(storageRank) * storageRank
	    || state.scoutNoise.size() < static_cast<size_t>(storageRank) * storageRank)
		return 0.0f;

	std::vector<float> gram(static_cast<size_t>(activeRank) * activeRank, 0.0f);
	for (unsigned int c1 = 0; c1 < activeRank; ++c1)
	{
		const double fisher1 =
		    std::max<double>(static_cast<double>(state.fisherDiag[c1]),
		                     static_cast<double>(eps));
		for (unsigned int c2 = c1; c2 < activeRank; ++c2)
		{
			const double fisher2 =
			    std::max<double>(static_cast<double>(state.fisherDiag[c2]),
			                     static_cast<double>(eps));
			double sum = 0.0;
			for (unsigned int lag = 0; lag < lagHorizon; ++lag)
			{
				const double lagWeight = 1.0 / std::sqrt(static_cast<double>(lag + 1u));
				const size_t lagBase = static_cast<size_t>(lag) * scoutSlice;
				for (unsigned int i = 0; i < scoutRank; ++i)
				{
					const double scoutDiag =
					    std::max<double>(static_cast<double>(tailMean)
					                     + static_cast<double>(state.scoutCov[static_cast<size_t>(i) * storageRank + i])
					                     + static_cast<double>(state.scoutNoise[static_cast<size_t>(i) * storageRank + i]),
					                     static_cast<double>(eps));
					const float* histRow =
					    &state.heroGwHistory[lagBase + static_cast<size_t>(i) * n];
					double dot1 = 0.0;
					double dot2 = 0.0;
					for (unsigned int col = 0; col < n; ++col)
					{
						const double hist = static_cast<double>(histRow[col]);
						dot1 += static_cast<double>(gz[static_cast<size_t>(c1) * n + col]) * hist;
						dot2 += static_cast<double>(gz[static_cast<size_t>(c2) * n + col]) * hist;
					}
					const double denom1 = std::sqrt(fisher1 * scoutDiag);
					const double denom2 = std::sqrt(fisher2 * scoutDiag);
					double h1 = (dot1 / static_cast<double>(n))
					          * static_cast<double>(statScaleSq)
					          * lagWeight / denom1;
					double h2 = (dot2 / static_cast<double>(n))
					          * static_cast<double>(statScaleSq)
					          * lagWeight / denom2;
					if (!std::isfinite(h1)) h1 = 0.0;
					if (!std::isfinite(h2)) h2 = 0.0;
					sum += h1 * h2;
				}
			}
			gram[static_cast<size_t>(c1) * activeRank + c2] = static_cast<float>(sum);
			gram[static_cast<size_t>(c2) * activeRank + c1] = static_cast<float>(sum);
		}
	}
	symmetrize_block(&gram[0], activeRank);

	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(&gram[0], activeRank, eigVec, eigVal);
	double topEig = (!eigVal.empty()) ? static_cast<double>(eigVal[0]) : 0.0;
	if (!std::isfinite(topEig) || topEig < 0.0)
		topEig = 0.0;
	double trace = 0.0;
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		const double d = static_cast<double>(gram[static_cast<size_t>(c) * activeRank + c]);
		if (std::isfinite(d) && d > 0.0)
			trace += d;
	}
	const double bulk =
	    std::max<double>(static_cast<double>(eps),
	                     trace / static_cast<double>(activeRank));
	if (sigmaOut)
		*sigmaOut = static_cast<float>(std::sqrt(topEig));
	const double edge = topEig / bulk - 1.0;
	return static_cast<float>(std::isfinite(edge) ? edge : 0.0);
}

static float compute_resolve_predictive_edge(const WeightState& state,
                                             unsigned int probeRank,
                                             bool useScoutProbe,
                                             unsigned int activeRank,
                                             unsigned int n,
                                             unsigned int lagHorizon,
                                             float statScaleSq,
                                             float tailMean,
                                             float eps)
{
	const unsigned int storageRank = state.complementRank;
	if (probeRank == 0u || storageRank == 0u || activeRank == 0u || n == 0u
	    || lagHorizon == 0u)
		return 0.0f;

	const size_t slice = static_cast<size_t>(state.r) * static_cast<size_t>(n);
	if (state.resolveGzHistory.size() < static_cast<size_t>(lagHorizon) * slice)
		return 0.0f;

	const std::vector<float>& probe =
	    useScoutProbe ? state.scratch_gwScout : state.scratch_gv;
	if (probe.size() < static_cast<size_t>(storageRank) * static_cast<size_t>(n))
		return 0.0f;

	const std::vector<float>& denomMain =
	    useScoutProbe ? state.scoutCov : state.complementBlock;
	const std::vector<float>* denomExtra =
	    useScoutProbe ? &state.scoutNoise : 0;
	if (denomMain.size() < static_cast<size_t>(storageRank) * storageRank)
		return 0.0f;

	std::vector<float> numer(static_cast<size_t>(probeRank) * probeRank, 0.0f);
	std::vector<float> cross(static_cast<size_t>(probeRank) * activeRank, 0.0f);
	std::vector<float> denom(static_cast<size_t>(probeRank) * probeRank, 0.0f);

	for (unsigned int i = 0; i < probeRank; ++i)
	{
		for (unsigned int j = 0; j < probeRank; ++j)
		{
			float v = denomMain[static_cast<size_t>(i) * storageRank + j];
			if (denomExtra
			    && denomExtra->size() >= static_cast<size_t>(storageRank) * storageRank)
			{
				v += (*denomExtra)[static_cast<size_t>(i) * storageRank + j];
			}
			if (i == j)
				v += std::max(eps, tailMean);
			denom[static_cast<size_t>(i) * probeRank + j] = v;
		}
	}
	symmetrize_block(&denom[0], probeRank);

	for (unsigned int lag = 0; lag < lagHorizon; ++lag)
	{
		const double wLag = 1.0 / static_cast<double>(lag + 1u);
		const size_t lagBase = static_cast<size_t>(lag) * slice;
		for (unsigned int i = 0; i < probeRank; ++i)
		{
			const float* probeRow = &probe[static_cast<size_t>(i) * n];
			for (unsigned int c = 0; c < activeRank; ++c)
			{
				const float* histRow =
				    &state.resolveGzHistory[lagBase + static_cast<size_t>(c) * n];
				double dot = 0.0;
				for (unsigned int col = 0; col < n; ++col)
				{
					dot += static_cast<double>(probeRow[col])
					     * static_cast<double>(histRow[col]);
				}
				double crossVal = (dot / static_cast<double>(n))
				                * static_cast<double>(statScaleSq);
				const double fisher =
				    std::max<double>(static_cast<double>(state.fisherDiag[c]),
				                     static_cast<double>(eps));
				crossVal /= std::sqrt(fisher);
				if (!std::isfinite(crossVal))
					crossVal = 0.0;
				cross[static_cast<size_t>(i) * activeRank + c] =
				    static_cast<float>(crossVal);
			}
		}

		for (unsigned int i = 0; i < probeRank; ++i)
		{
			for (unsigned int j = i; j < probeRank; ++j)
			{
				double sum = 0.0;
				for (unsigned int c = 0; c < activeRank; ++c)
				{
					sum += static_cast<double>(cross[static_cast<size_t>(i) * activeRank + c])
					     * static_cast<double>(cross[static_cast<size_t>(j) * activeRank + c]);
				}
				numer[static_cast<size_t>(i) * probeRank + j] +=
				    static_cast<float>(wLag * sum);
				if (i != j)
				{
					numer[static_cast<size_t>(j) * probeRank + i] =
					    numer[static_cast<size_t>(i) * probeRank + j];
				}
			}
		}
	}
	symmetrize_block(&numer[0], probeRank);

	double eigVal = 0.0;
	std::vector<float> eigVec;
	if (!top_generalized_eigenpair(&numer[0], &denom[0], probeRank, eps, &eigVal, eigVec))
		return 0.0f;
	if (!std::isfinite(eigVal))
		return 0.0f;
	return static_cast<float>(eigVal - 1.0);
}

static double fit_resolve_memory_pole(const WeightState& state,
                                      const std::vector<float>& gz,
                                      unsigned int row,
                                      unsigned int n,
                                      unsigned int lagHorizon,
                                      float statScaleSq,
                                      float eps)
{
	const size_t slice = static_cast<size_t>(state.r) * static_cast<size_t>(n);
	if (lagHorizon == 0u || row >= state.r || n == 0u
	    || gz.size() < static_cast<size_t>(state.r) * static_cast<size_t>(n)
	    || state.resolveGzHistory.size() < static_cast<size_t>(lagHorizon) * slice)
		return 0.0;

	const double gamma0 =
	    std::max<double>(static_cast<double>(state.fisherDiag[row]),
	                     static_cast<double>(eps));
	double rhoSum = 0.0;
	double wSum = 0.0;
	for (unsigned int lag = 1u; lag <= lagHorizon; ++lag)
	{
		const float* histRow =
		    &state.resolveGzHistory[(static_cast<size_t>(lag) - 1u) * slice
		                            + static_cast<size_t>(row) * n];
		double dot = 0.0;
		for (unsigned int col = 0; col < n; ++col)
		{
			dot += static_cast<double>(gz[static_cast<size_t>(row) * n + col])
			     * static_cast<double>(histRow[col]);
		}
		double gammaLag = (dot / static_cast<double>(n))
		                * static_cast<double>(statScaleSq);
		if (!(gammaLag > 0.0))
			continue;
		double ratio = gammaLag / gamma0;
		if (!std::isfinite(ratio) || !(ratio > 0.0))
			continue;
		if (ratio > 1.0)
			ratio = 1.0;
		double rho = std::pow(ratio, 1.0 / static_cast<double>(lag));
		if (!std::isfinite(rho) || rho <= 0.0)
			continue;
		if (rho > 0.95)
			rho = 0.95;
		const double wLag = 1.0 / static_cast<double>(lag);
		rhoSum += wLag * rho;
		wSum += wLag;
	}
	if (!(wSum > 0.0))
		return 0.0;
	return rhoSum / wSum;
}

static double symmetric_quadratic_form(const float* block,
                                       unsigned int dim,
                                       const std::vector<float>& vec)
{
	if (dim == 0u || vec.size() < dim)
		return 0.0;
	double sum = 0.0;
	for (unsigned int i = 0; i < dim; ++i)
	{
		double row = 0.0;
		for (unsigned int j = 0; j < dim; ++j)
			row += static_cast<double>(block[static_cast<size_t>(i) * dim + j])
			     * static_cast<double>(vec[j]);
		sum += static_cast<double>(vec[i]) * row;
	}
	return sum;
}

static bool top_generalized_eigenpair(const float* numer,
                                      const float* denom,
                                      unsigned int dim,
                                      float eps,
                                      double* eigValOut,
                                      std::vector<float>& eigVecOut)
{
	eigVecOut.assign(static_cast<size_t>(dim), 0.0f);
	if (eigValOut)
		*eigValOut = 0.0;
	if (dim == 0u)
		return false;

	std::vector<float> denomEigVec;
	std::vector<float> denomEigVal;
	std::vector<float> denomBlock(static_cast<size_t>(dim) * dim, 0.0f);
	for (unsigned int i = 0; i < dim * dim; ++i)
		denomBlock[i] = denom[i];
	symmetrize_block(&denomBlock[0], dim);
	jacobi_eigendecompose(&denomBlock[0], dim, denomEigVec, denomEigVal);

	std::vector<float> invSqrt(static_cast<size_t>(dim) * dim, 0.0f);
	for (unsigned int i = 0; i < dim; ++i)
	{
		const double lambda = std::max<double>(static_cast<double>(denomEigVal[i]),
		                                       static_cast<double>(eps));
		const double scale = 1.0 / std::sqrt(lambda);
		for (unsigned int r = 0; r < dim; ++r)
			for (unsigned int c = 0; c < dim; ++c)
				invSqrt[static_cast<size_t>(r) * dim + c] +=
				    static_cast<float>(scale)
				    * denomEigVec[static_cast<size_t>(r) * dim + i]
				    * denomEigVec[static_cast<size_t>(c) * dim + i];
	}

	std::vector<float> temp(static_cast<size_t>(dim) * dim, 0.0f);
	multiply_left_block(&temp[0], &invSqrt[0], numer, dim, dim, dim);
	std::vector<float> whitened(static_cast<size_t>(dim) * dim, 0.0f);
	for (unsigned int i = 0; i < dim; ++i)
	{
		for (unsigned int j = 0; j < dim; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < dim; ++k)
				sum += static_cast<double>(temp[static_cast<size_t>(i) * dim + k])
				     * static_cast<double>(invSqrt[static_cast<size_t>(j) * dim + k]);
			whitened[static_cast<size_t>(i) * dim + j] = static_cast<float>(sum);
		}
	}
	symmetrize_block(&whitened[0], dim);
	std::vector<float> whiteEigVec;
	std::vector<float> whiteEigVal;
	jacobi_eigendecompose(&whitened[0], dim, whiteEigVec, whiteEigVal);
	if (whiteEigVal.empty())
		return false;

	if (eigValOut)
		*eigValOut = std::max<double>(0.0, static_cast<double>(whiteEigVal[0]));
	for (unsigned int i = 0; i < dim; ++i)
	{
		double sum = 0.0;
		for (unsigned int k = 0; k < dim; ++k)
			sum += static_cast<double>(invSqrt[static_cast<size_t>(i) * dim + k])
			     * static_cast<double>(whiteEigVec[static_cast<size_t>(k) * dim + 0u]);
		eigVecOut[i] = static_cast<float>(sum);
	}
	return normalize_vector(&eigVecOut[0], dim);
}

static bool build_inv_sqrt_psd(const float* block,
                               unsigned int dim,
                               float eps,
                               std::vector<float>& invSqrt)
{
	invSqrt.assign(static_cast<size_t>(dim) * dim, 0.0f);
	if (dim == 0u)
		return false;
	std::vector<float> sym(static_cast<size_t>(dim) * dim, 0.0f);
	for (unsigned int i = 0; i < dim * dim; ++i)
		sym[i] = block[i];
	symmetrize_block(&sym[0], dim);
	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(&sym[0], dim, eigVec, eigVal);
	if (eigVal.empty())
		return false;
	for (unsigned int mode = 0; mode < dim; ++mode)
	{
		const double lambda = std::max<double>(static_cast<double>(eigVal[mode]),
		                                       static_cast<double>(eps));
		const double scale = 1.0 / std::sqrt(lambda);
		for (unsigned int r = 0; r < dim; ++r)
		{
			for (unsigned int c = 0; c < dim; ++c)
			{
				invSqrt[static_cast<size_t>(r) * dim + c] +=
				    static_cast<float>(scale)
				    * eigVec[static_cast<size_t>(r) * dim + mode]
				    * eigVec[static_cast<size_t>(c) * dim + mode];
			}
		}
	}
	return true;
}

static void build_scout_contamination_matrix(const WeightState& state,
                                             unsigned int scoutRank,
                                             unsigned int complementRankUsed,
                                             unsigned int n,
                                             float statScaleSq,
                                             float eps,
                                             std::vector<float>& out)
{
	out.assign(static_cast<size_t>(scoutRank) * scoutRank, 0.0f);
	if (scoutRank == 0u || complementRankUsed == 0u || n == 0u)
		return;

	std::vector<float> cross(static_cast<size_t>(scoutRank) * complementRankUsed, 0.0f);
	for (unsigned int i = 0; i < scoutRank; ++i)
	{
		for (unsigned int j = 0; j < complementRankUsed; ++j)
		{
			double dot = 0.0;
			for (unsigned int col = 0; col < n; ++col)
			{
				dot += static_cast<double>(state.scratch_gwScout[static_cast<size_t>(i) * n + col])
				     * static_cast<double>(state.scratch_gv[static_cast<size_t>(j) * n + col]);
			}
			cross[static_cast<size_t>(i) * complementRankUsed + j] =
			    static_cast<float>(dot / static_cast<double>(n)) * statScaleSq;
		}
	}

	std::vector<float> activeBlock(static_cast<size_t>(complementRankUsed) * complementRankUsed, 0.0f);
	for (unsigned int i = 0; i < complementRankUsed; ++i)
		for (unsigned int j = 0; j < complementRankUsed; ++j)
			activeBlock[static_cast<size_t>(i) * complementRankUsed + j] =
			    state.complementBlock[static_cast<size_t>(i) * state.complementRank + j];
	symmetrize_block(&activeBlock[0], complementRankUsed);

	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(&activeBlock[0], complementRankUsed, eigVec, eigVal);
	std::vector<float> inv(static_cast<size_t>(complementRankUsed) * complementRankUsed, 0.0f);
	for (unsigned int mode = 0; mode < complementRankUsed; ++mode)
	{
		const double lambda =
		    std::max<double>(static_cast<double>(eigVal[mode]), static_cast<double>(eps));
		const double scale = 1.0 / lambda;
		for (unsigned int r = 0; r < complementRankUsed; ++r)
			for (unsigned int c = 0; c < complementRankUsed; ++c)
				inv[static_cast<size_t>(r) * complementRankUsed + c] +=
				    static_cast<float>(scale)
				    * eigVec[static_cast<size_t>(r) * complementRankUsed + mode]
				    * eigVec[static_cast<size_t>(c) * complementRankUsed + mode];
	}

	std::vector<float> temp(static_cast<size_t>(scoutRank) * complementRankUsed, 0.0f);
	multiply_left_block(&temp[0], &cross[0], &inv[0], scoutRank, complementRankUsed, complementRankUsed);
	for (unsigned int i = 0; i < scoutRank; ++i)
	{
		for (unsigned int j = i; j < scoutRank; ++j)
		{
			double sum = 0.0;
			for (unsigned int k = 0; k < complementRankUsed; ++k)
				sum += static_cast<double>(temp[static_cast<size_t>(i) * complementRankUsed + k])
				     * static_cast<double>(cross[static_cast<size_t>(j) * complementRankUsed + k]);
			out[static_cast<size_t>(i) * scoutRank + j] = static_cast<float>(sum);
			out[static_cast<size_t>(j) * scoutRank + i] = static_cast<float>(sum);
		}
	}
}

bool applyStep(WeightState& state,
               float* W, float* gW,
               unsigned int m, unsigned int n,
               float invBatch, float lr,
               float wd1, float wd2, float gradScale,
               const ATLASConfig& ac,
               glades::rng::Engine& rng,
               shmea::GLogger* logger,
               const char* tag)
{
	if (!state.initialized) return false;
	bool recovered = false;

	const float beta = ac.beta;
	const float muMin = ac.muMin;
	const float muMax = ac.muMax;
	const float eps = ac.eps;
	const float kappaMax = ac.kappaMax;
	const unsigned int tSub = ac.tSub;
	const float muGrowthRate = ac.muGrowthRate;

	const unsigned int r = state.r;
	const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
	state.step += 1ULL;
	state.activeRank = atlas_clamp_active_rank(state.activeRank, r, ac.minActiveRank);
	unsigned int activeRank = state.activeRank;
	const unsigned int enabledComplementRank =
	    atlas_enabled_complement_rank(ac.complementRank, tag, m, n, activeRank);
	const unsigned int sparrowModeRank =
	    atlas_sparrow_mode_rank(ac.sparrowModeRank);
	const bool orbitEligible = atlas_output_head_eligible(tag, m, n);
	const bool adaptiveComplementController = (tag && tag[0]);
	const unsigned int storageComplementRank =
	    atlas_storage_complement_rank(enabledComplementRank);
	if (state.complementRank != storageComplementRank)
		resize_complement_storage(state, storageComplementRank, activeRank, rng, logger);
	if (state.sparrowModeRank != sparrowModeRank)
		ensure_sparrow_storage(state, sparrowModeRank);
	const unsigned int complementRank = state.complementRank;
	size_t rn = static_cast<size_t>(activeRank) * static_cast<size_t>(n);
	const size_t cn = static_cast<size_t>(complementRank) * static_cast<size_t>(n);
	if (state.activeComplementRank > enabledComplementRank)
		state.activeComplementRank = enabledComplementRank;
	if (state.trialComplementRank > enabledComplementRank
	    || state.trialComplementRank <= state.activeComplementRank)
		reset_complement_trial(state);

	const bool diagStep = logger && tSub > 0u
		&& (state.step % static_cast<unsigned long long>(tSub)) == 0ULL;
	const bool complementControlStep =
	    adaptiveComplementController
	    && (enabledComplementRank > 0u)
	    && ((tSub == 0u) || (state.step % static_cast<unsigned long long>(tSub)) == 0ULL);
	const unsigned int complementTargetRank =
	    atlas_requested_complement_rank(enabledComplementRank, m, activeRank);
	const unsigned int cobaltProbeRank =
	    (ac.cobaltEnabled && enabledComplementRank > 0u)
	        ? atlas_requested_scout_rank(complementRank, m, activeRank, complementTargetRank)
	        : 0u;
	const unsigned int birchProbeRank =
	    (ac.birchEnabled && enabledComplementRank > 0u)
	        ? atlas_requested_scout_rank(complementRank, m, activeRank, complementTargetRank)
	        : 0u;
	const unsigned int sparrowProbeRank =
	    ac.sparrowEnabled
	        ? atlas_requested_scout_rank(complementRank, m, activeRank,
	                                     (sparrowModeRank < enabledComplementRank)
	                                         ? sparrowModeRank
	                                         : enabledComplementRank)
	        : 0u;
	const unsigned int ghostProbeRank =
	    (ac.ghostEnabled && enabledComplementRank > 0u)
	        ? atlas_requested_scout_rank(complementRank, m, activeRank, complementTargetRank)
	        : 0u;
	const bool generalizedScoutEnabled =
	    complementControlStep && tSub > 0u && complementTargetRank > 0u;
	const unsigned int scoutControlRank =
	    generalizedScoutEnabled
	        ? atlas_requested_scout_rank(complementRank, m, activeRank, complementTargetRank)
	        : 0u;
	const unsigned int scoutProjectionRank =
	    std::max(std::max(std::max(std::max(cobaltProbeRank, birchProbeRank), sparrowProbeRank),
	                      ghostProbeRank),
	             scoutControlRank);

	const float gScale = invBatch * gradScale;
	const float statScaleSq = (invBatch > 0.0f) ? (1.0f / (invBatch * invBatch)) : 1.0f;

	// === Bias correction factor ===
	// Compensates for zero-initialization bias in EMA quantities.
	// Applied to effective sigma2 and fisherDiag when computing learning rates,
	// NOT to the stored raw EMA values (which remain uncorrected for stability).
	float bcFactor = 1.0f;
	if (ac.biasCorrection)
	{
		const double betaPow = pow(static_cast<double>(beta),
		                           static_cast<double>(state.step));
		const double denom = 1.0 - betaPow;
		if (denom > 1e-15)
			bcFactor = static_cast<float>(1.0 / denom);
	}

	// === Step 1: Update the normalized covariance trace ===
	// totalTrace tracks tr(C) where C = (1/n) * H * H^T and H = gradScale * G.
	// The active Fisher statistics are reduced from the same operator, so sigma2
	// can be recovered by trace closure without changing the historical gauge.
	{
		double traceSample = 0.0;
		for (size_t idx = 0; idx < mn; ++idx)
		{
			const double v = static_cast<double>(gW[idx]) * static_cast<double>(gradScale);
			traceSample += v * v;
		}
		traceSample /= static_cast<double>(n);
		state.totalTrace = atlas_bootstrap_or_ema(state.totalTrace,
		                                          static_cast<float>(traceSample),
		                                          beta,
		                                          state.step);
		if (state.totalTrace < eps)
			state.totalTrace = eps;
		if (!atlas_isfinite(state.totalTrace))
		{
			state.totalTrace = eps;
			recovered = true;
		}
	}

	// === Step 2: Periodic subspace refresh (EMA-blended) ===
	// Pass raw gW directly — eigenvectors of G*G^T are scale-invariant.
	if (tSub > 0u && (state.step % static_cast<unsigned long long>(tSub)) == 0ULL)
	{
			if (!refreshSubspace(state, gW, m, n, ac.powerIters, ac.betaRefresh,
			                     ac.fisherWeightedRefresh, rng, logger))
			{
				recovered = true;
			}
		if (enabledComplementRank > 0u
		    && !refreshComplementSector(state, gW, m, n, activeRank,
		                                ac.powerIters, ac.betaRefresh, rng, logger))
			{
				recovered = true;
			}
		if (scoutControlRank > 0u
		    && !refreshComplementScout(state, gW, m, n, activeRank,
		                               complementTargetRank, ac.powerIters,
		                               ac.betaRefresh, rng, logger))
			{
				recovered = true;
			}
	}

	// === Step 3: Decoupled weight decay (applied in full parameter space) ===
	if (wd1 != 0.0f || wd2 != 0.0f)
	{
		for (size_t idx = 0; idx < mn; ++idx)
		{
			const float w = W[idx];
			if (wd1 != 0.0f)
			{
				const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
				W[idx] -= lr * wd1 * wSign;
			}
			if (wd2 != 0.0f)
				W[idx] -= lr * wd2 * w;
		}
	}

	// === Step 4: Project gradient to subspace ===
	// gz[r,n] = gScale * U^T[r,m] * gW[m,n]
	std::vector<float>& gz = state.scratch_gz;
	std::vector<float>& basisPacked = state.scratch_basisPacked;
	pack_active_basis(state.U, state.r, m, activeRank, basisPacked);
	glades::gemm::atb(&gz[0], &basisPacked[0], gW, activeRank, m, n, gScale);

	std::vector<float>& gv = state.scratch_gv;
	if (enabledComplementRank > 0u)
		glades::gemm::atb(&gv[0], &state.V[0], gW, complementRank, m, n, gScale);
	else if (!gv.empty())
		std::fill(gv.begin(), gv.end(), 0.0f);

	if (scoutProjectionRank > 0u)
	{
		glades::gemm::atb(&state.scratch_gwScout[0], &state.scoutBasis[0], gW,
		                  complementRank, m, n, gScale);
	}
	else if (!state.scratch_gwScout.empty())
	{
		std::fill(state.scratch_gwScout.begin(), state.scratch_gwScout.end(), 0.0f);
	}

	// === Step 5: Update Fisher diagonal (EMA of mean squared projected gradient) ===
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		double sumsq = 0.0;
		for (unsigned int j = 0; j < n; ++j)
		{
			const double v = static_cast<double>(gz[c * n + j]);
			sumsq += v * v;
		}
		const float meansq = static_cast<float>(sumsq / static_cast<double>(n)) * statScaleSq;
		state.fisherDiag[c] = atlas_bootstrap_or_ema(state.fisherDiag[c],
		                                             meansq,
		                                             beta,
		                                             state.step);
		if (!atlas_isfinite(state.fisherDiag[c]))
		{
				state.fisherDiag[c] = eps;
				recovered = true;
			}
	}
	if (enabledComplementRank > 0u)
	{
		update_complement_block_ema(state, beta, state.step, statScaleSq, n);
		if (!atlas_isfinite(state.complementFisher))
		{
			state.complementFisher = eps;
				std::fill(state.complementBlock.begin(), state.complementBlock.end(), 0.0f);
				state.complementBlock[0] = eps;
				recovered = true;
			}
	}
	else
	{
		std::fill(state.complementBlock.begin(), state.complementBlock.end(), 0.0f);
		state.complementFisher = 0.0f;
		state.activeComplementRank = 0u;
		reset_complement_trial(state);
	}
	if (scoutControlRank > 0u)
		update_scout_statistics(state, beta, state.step, statScaleSq, n, scoutControlRank);
	else if (!state.scoutCov.empty())
	{
		std::fill(state.scoutCov.begin(), state.scoutCov.end(), 0.0f);
		std::fill(state.scoutNoise.begin(), state.scoutNoise.end(), 0.0f);
	}

	// === Step 6: Recompute sigma2 from the shared covariance model ===
	unsigned int informativeComplementRank =
	    effective_complement_rank(state, enabledComplementRank, m, activeRank);
	double activeSectorTrace = 0.0;
	float complementTailMean = eps;
	float complementScoutTop = 0.0f;
	double complementScoutProjected = 0.0;
	float complementScoutLambda = 0.0f;
	float complementBirthKelly = 0.0f;
	double complementTrialKelly = 0.0;
	double complementTrialAlignment = 0.0;
	double complementTrialContamination = 0.0;
	double complementTrialReturn = 0.0;
	double complementTrialScore = 0.0;
	float prismPredictiveEdge = 0.0f;
	float prismLag1 = 0.0f;
	float prismLag2 = 0.0f;
	float prismMemoryGain = 0.0f;
	float resolvePredictiveEdge = 0.0f;
	float resolveKernelRho = 0.0f;
	float resolveMemoryGain = 0.0f;
	float heroEdge = 0.0f;
	float heroSigma = 0.0f;
	float heroMemoryGain = 0.0f;
	float cobaltEdge = 0.0f;
	float cobaltSigma = 0.0f;
	float cobaltMemoryGain = 0.0f;
	float birchEdge = 0.0f;
	float birchSigma = 0.0f;
	float birchMemoryGain = 0.0f;
	float orbitEdge = 0.0f;
	float orbitSigma = 0.0f;
	float orbitPole = 0.0f;
	float orbitHorizontalRatio = 1.0f;
	float orbitMemoryGain = 0.0f;
	float sparrowEdge = 0.0f;
	float sparrowSigma = 0.0f;
	float sparrowSecondEdge = 0.0f;
	float sparrowSecondSigma = 0.0f;
	float sparrowPole = 0.0f;
	float sparrowHorizontalRatio = 1.0f;
	float sparrowMemoryGain = 0.0f;
	unsigned int sparrowActiveModes = 0u;
	float qbrtEdge = 0.0f;
	float qbrtSigma = 0.0f;
	float qbrtPole = 0.0f;
	float qbrtHorizontalRatio = 1.0f;
	float qbrtMemoryGain = 0.0f;
	float qrcEdge = 0.0f;
	float qrcSigma = 0.0f;
	float qrcPole = 0.0f;
	float qrcHorizontalRatio = 1.0f;
	float qrcControlGain = 0.0f;
	float qrcMemoryGain = 0.0f;
	float riftEdge = 0.0f;
	float riftSigma = 0.0f;
	float riftPole = 0.0f;
	float riftHorizontalRatio = 1.0f;
	float riftAreaEnergy = 0.0f;
	float riftPredR2 = 0.0f;
	float riftMemoryGain = 0.0f;
	float ghostEdge = 0.0f;
	float ghostSigma = 0.0f;
	float ghostHorizontalRatio = 1.0f;
	float ghostMemoryGain = 0.0f;
	std::vector<float> orbitLeftMode;
	std::vector<float> orbitLatent;
	std::vector<float> sparrowLeftMode;
	std::vector<float> sparrowLatent;
	std::vector<float> qbrtLeftMode;
	std::vector<float> qbrtLatent;
	std::vector<float> qrcLeftMode;
	std::vector<float> qrcLatent;
	std::vector<float> riftLeftMode;
	std::vector<float> riftLatent;
	std::vector<float> ghostLeftMode;
	std::vector<float> ghostRightMode;
	std::vector<float> ghostPastStack;
	state.lastPredictiveEdge = 0.0f;
	state.lastMemoryGain = 0.0f;
	state.lastResolveEdge = 0.0f;
	state.lastResolveKernelRho = 0.0f;
	state.lastResolveMemoryGain = 0.0f;
	state.lastHeroEdge = 0.0f;
	state.lastHeroSigma = 0.0f;
	state.lastHeroMemoryGain = 0.0f;
	state.lastCobaltEdge = 0.0f;
	state.lastCobaltSigma = 0.0f;
	state.lastCobaltMemoryGain = 0.0f;
	state.lastBirchEdge = 0.0f;
	state.lastBirchSigma = 0.0f;
	state.lastBirchMemoryGain = 0.0f;
	state.lastOrbitEdge = 0.0f;
	state.lastOrbitSigma = 0.0f;
	state.lastOrbitHorizontalRatio = 1.0f;
	state.lastOrbitMemoryGain = 0.0f;
	state.lastSparrowEdge = 0.0f;
	state.lastSparrowSigma = 0.0f;
	state.lastSparrowSecondEdge = 0.0f;
	state.lastSparrowSecondSigma = 0.0f;
	state.lastSparrowHorizontalRatio = 1.0f;
	state.lastSparrowMemoryGain = 0.0f;
	state.lastSparrowActiveModes = 0u;
	state.lastQbrtEdge = 0.0f;
	state.lastQbrtSigma = 0.0f;
	state.lastQbrtHorizontalRatio = 1.0f;
	state.lastQbrtMemoryGain = 0.0f;
	state.lastQrcEdge = 0.0f;
	state.lastQrcSigma = 0.0f;
	state.lastQrcHorizontalRatio = 1.0f;
	state.lastQrcControlGain = 0.0f;
	state.lastQrcMemoryGain = 0.0f;
	state.lastRiftEdge = 0.0f;
	state.lastRiftSigma = 0.0f;
	state.lastRiftHorizontalRatio = 1.0f;
	state.lastRiftAreaEnergy = 0.0f;
	state.lastRiftPredR2 = 0.0f;
	state.lastRiftMemoryGain = 0.0f;
	state.lastGhostEdge = 0.0f;
	state.lastGhostSigma = 0.0f;
	state.lastGhostHorizontalRatio = 1.0f;
	state.lastGhostMemoryGain = 0.0f;
	const unsigned int cobaltLagHorizon = atlas_cobalt_lag_horizon(ac.cobaltLagHorizon);
	const unsigned int birchPastHorizon = atlas_birch_past_horizon(ac.birchPastHorizon);
	const unsigned int birchFutureHorizon = atlas_birch_future_horizon(ac.birchFutureHorizon);
	const unsigned int ghostLagHorizon = atlas_ghost_lag_horizon(ac.ghostLagHorizon);
	const unsigned int qbrtLagHorizon = atlas_qbrt_lag_horizon(ac.qbrtLagHorizon);
	const unsigned int qrcLagHorizon = atlas_qrc_lag_horizon(ac.qrcLagHorizon);
	const unsigned int riftLagHorizon = atlas_rift_lag_horizon(ac.riftLagHorizon);
	const unsigned int resolveLagHorizon = atlas_resolve_lag_horizon(ac.resolveLagHorizon);
	const unsigned int heroLagHorizon = atlas_hero_lag_horizon(ac.heroLagHorizon);
	if (ac.orbitEnabled && orbitEligible)
	{
		orbitEdge =
		    compute_orbit_lite_mode(state,
		                            W,
		                            basisPacked.empty() ? 0 : &basisPacked[0],
		                            gW,
		                            activeRank,
		                            m,
		                            n,
		                            beta,
		                            statScaleSq,
		                            eps,
		                            ac.orbitPoleMax,
		                            &orbitSigma,
		                            &orbitPole,
		                            &orbitHorizontalRatio,
		                            orbitLeftMode,
		                            orbitLatent);
		if (!atlas_isfinite(orbitEdge))
			orbitEdge = 0.0f;
		if (!atlas_isfinite(orbitSigma))
			orbitSigma = 0.0f;
		if (!atlas_isfinite(orbitPole))
			orbitPole = 0.0f;
		if (!atlas_isfinite(orbitHorizontalRatio))
			orbitHorizontalRatio = 1.0f;
		state.lastOrbitEdge = orbitEdge;
		state.lastOrbitSigma = orbitSigma;
		state.lastOrbitHorizontalRatio = orbitHorizontalRatio;
	}
	if (ac.sparrowEnabled)
	{
		sparrowEdge =
		    compute_sparrow_streaming_mode(state,
		                                   W,
		                                   basisPacked.empty() ? 0 : &basisPacked[0],
		                                   gz,
		                                   state.scratch_gwScout,
		                                   sparrowProbeRank,
		                                   activeRank,
		                                   m,
		                                   n,
		                                   beta,
		                                   statScaleSq,
		                                   eps,
		                                   ac.sparrowPoleMax,
		                                   sparrowModeRank,
		                                   ac.sparrowAutoModeGate,
		                                   ac.sparrowSecondEdgeThreshold,
		                                   ac.sparrowSecondEdgeFraction,
		                                   &sparrowSigma,
		                                   &sparrowSecondSigma,
		                                   &sparrowSecondEdge,
		                                   &sparrowPole,
		                                   &sparrowHorizontalRatio,
		                                   &sparrowActiveModes,
		                                   sparrowLeftMode,
		                                   sparrowLatent);
		if (!atlas_isfinite(sparrowEdge))
			sparrowEdge = 0.0f;
		if (!atlas_isfinite(sparrowSigma))
			sparrowSigma = 0.0f;
		if (!atlas_isfinite(sparrowPole))
			sparrowPole = 0.0f;
		if (!atlas_isfinite(sparrowHorizontalRatio))
			sparrowHorizontalRatio = 1.0f;
		state.lastSparrowEdge = sparrowEdge;
		state.lastSparrowSigma = sparrowSigma;
		state.lastSparrowSecondEdge = sparrowSecondEdge;
		state.lastSparrowSecondSigma = sparrowSecondSigma;
		state.lastSparrowHorizontalRatio = sparrowHorizontalRatio;
		state.lastSparrowActiveModes = sparrowActiveModes;
	}
	if (ac.qbrtEnabled)
	{
		qbrtEdge =
		    compute_qbrt_balanced_mode(state,
		                               W,
		                               basisPacked.empty() ? 0 : &basisPacked[0],
		                               gz,
		                               ghostProbeRank,
		                               activeRank,
		                               m,
		                               n,
		                               qbrtLagHorizon,
		                               beta,
		                               statScaleSq,
		                               eps,
		                               ac.qbrtPoleMax,
		                               &qbrtSigma,
		                               &qbrtPole,
		                               &qbrtHorizontalRatio,
		                               qbrtLeftMode,
		                               qbrtLatent);
		if (!atlas_isfinite(qbrtEdge))
			qbrtEdge = 0.0f;
		if (!atlas_isfinite(qbrtSigma))
			qbrtSigma = 0.0f;
		if (!atlas_isfinite(qbrtPole))
			qbrtPole = 0.0f;
		if (!atlas_isfinite(qbrtHorizontalRatio))
			qbrtHorizontalRatio = 1.0f;
		state.lastQbrtEdge = qbrtEdge;
		state.lastQbrtSigma = qbrtSigma;
		state.lastQbrtHorizontalRatio = qbrtHorizontalRatio;
	}
	if (ac.qrcEnabled)
	{
		qrcEdge =
		    compute_qrc_control_mode(state,
		                             W,
		                             basisPacked.empty() ? 0 : &basisPacked[0],
		                             gz,
		                             ghostProbeRank,
		                             activeRank,
		                             m,
		                             n,
		                             qrcLagHorizon,
		                             beta,
		                             statScaleSq,
		                             eps,
		                             ac.qrcPoleMax,
		                             &qrcSigma,
		                             &qrcPole,
		                             &qrcHorizontalRatio,
		                             &qrcControlGain,
		                             qrcLeftMode,
		                             qrcLatent);
		if (!atlas_isfinite(qrcEdge))
			qrcEdge = 0.0f;
		if (!atlas_isfinite(qrcSigma))
			qrcSigma = 0.0f;
		if (!atlas_isfinite(qrcPole))
			qrcPole = 0.0f;
		if (!atlas_isfinite(qrcHorizontalRatio))
			qrcHorizontalRatio = 1.0f;
		if (!atlas_isfinite(qrcControlGain))
			qrcControlGain = 0.0f;
		state.lastQrcEdge = qrcEdge;
		state.lastQrcSigma = qrcSigma;
		state.lastQrcHorizontalRatio = qrcHorizontalRatio;
		state.lastQrcControlGain = qrcControlGain;
	}
	if (ac.riftEnabled)
	{
		riftEdge =
		    compute_rift_signature_mode(state,
		                                W,
		                                basisPacked.empty() ? 0 : &basisPacked[0],
		                                gz,
		                                ghostProbeRank,
		                                activeRank,
		                                m,
		                                n,
		                                riftLagHorizon,
		                                beta,
		                                statScaleSq,
		                                eps,
		                                ac.riftPoleMax,
		                                &riftSigma,
		                                &riftPole,
		                                &riftHorizontalRatio,
		                                &riftAreaEnergy,
		                                &riftPredR2,
		                                riftLeftMode,
		                                riftLatent);
		if (!atlas_isfinite(riftEdge))
			riftEdge = 0.0f;
		if (!atlas_isfinite(riftSigma))
			riftSigma = 0.0f;
		if (!atlas_isfinite(riftPole))
			riftPole = 0.0f;
		if (!atlas_isfinite(riftHorizontalRatio))
			riftHorizontalRatio = 1.0f;
		if (!atlas_isfinite(riftAreaEnergy))
			riftAreaEnergy = 0.0f;
		if (!atlas_isfinite(riftPredR2))
			riftPredR2 = 0.0f;
		state.lastRiftEdge = riftEdge;
		state.lastRiftSigma = riftSigma;
		state.lastRiftHorizontalRatio = riftHorizontalRatio;
		state.lastRiftAreaEnergy = riftAreaEnergy;
		state.lastRiftPredR2 = riftPredR2;
	}
	if (ac.ghostEnabled)
	{
		ghostEdge =
		    compute_ghost_balanced_mode(state,
		                                W,
		                                basisPacked.empty() ? 0 : &basisPacked[0],
		                                gz,
		                                ghostProbeRank,
		                                activeRank,
		                                m,
		                                n,
		                                ghostLagHorizon,
		                                statScaleSq,
		                                eps,
		                                &ghostSigma,
		                                &ghostHorizontalRatio,
		                                ghostLeftMode,
		                                ghostRightMode,
		                                ghostPastStack);
		if (!atlas_isfinite(ghostEdge))
			ghostEdge = 0.0f;
		if (!atlas_isfinite(ghostSigma))
			ghostSigma = 0.0f;
		if (!atlas_isfinite(ghostHorizontalRatio))
			ghostHorizontalRatio = 1.0f;
		state.lastGhostEdge = ghostEdge;
		state.lastGhostSigma = ghostSigma;
		state.lastGhostHorizontalRatio = ghostHorizontalRatio;
	}
	if (ac.birchEnabled)
	{
		birchEdge =
		    compute_birch_hankel_edge(state,
		                              gz,
		                              birchProbeRank,
		                              activeRank,
		                              n,
		                              birchPastHorizon,
		                              birchFutureHorizon,
		                              statScaleSq,
		                              eps,
		                              &birchSigma);
		if (!atlas_isfinite(birchEdge))
			birchEdge = 0.0f;
		if (!atlas_isfinite(birchSigma))
			birchSigma = 0.0f;
		state.lastBirchEdge = birchEdge;
		state.lastBirchSigma = birchSigma;
	}
	if (ac.cobaltEnabled)
	{
		cobaltEdge =
		    compute_cobalt_transfer_edge(state,
		                                 gz,
		                                 cobaltProbeRank,
		                                 activeRank,
		                                 n,
		                                 cobaltLagHorizon,
		                                 statScaleSq,
		                                 eps,
		                                 &cobaltSigma);
		if (!atlas_isfinite(cobaltEdge))
			cobaltEdge = 0.0f;
		if (!atlas_isfinite(cobaltSigma))
			cobaltSigma = 0.0f;
		state.lastCobaltEdge = cobaltEdge;
		state.lastCobaltSigma = cobaltSigma;
	}
	if (enabledComplementRank > 0u && informativeComplementRank > 0u)
	{
		jacobi_eigendecompose(&state.complementBlock[0], complementRank,
		                      state.scratch_complementEigVec,
		                      state.scratch_complementEigVal);
		for (unsigned int i = 0; i < complementRank; ++i)
		{
			if (!atlas_isfinite(state.scratch_complementEigVal[i])
			    || state.scratch_complementEigVal[i] < 0.0f)
				state.scratch_complementEigVal[i] = 0.0f;
		}
		std::vector<float> scoutEigVal;
		std::vector<float> scoutEigVec;
		if (adaptiveComplementController)
		{
			std::vector<float> scoutBlock(state.scratch_complementMat);
			symmetrize_block(&scoutBlock[0], complementRank);
			jacobi_eigendecompose(&scoutBlock[0], complementRank, scoutEigVec, scoutEigVal);
			for (unsigned int i = 0; i < complementRank; ++i)
			{
				if (!atlas_isfinite(scoutEigVal[i]) || scoutEigVal[i] < 0.0f)
					scoutEigVal[i] = 0.0f;
			}
			if (!scoutEigVal.empty())
				complementScoutTop = scoutEigVal[0];
		}
		const double activeTraceForGate = compute_active_trace(state, activeRank);
		const double fullBlockTraceForGate =
		    sum_leading_spectrum(state.scratch_complementEigVal, informativeComplementRank);
		const double closedTraceForGate = std::max<double>(
		    static_cast<double>(state.totalTrace),
		    activeTraceForGate + fullBlockTraceForGate);
		const float prismTailMean = static_cast<float>(
		    complement_tail_mean(closedTraceForGate, activeTraceForGate, 0.0,
		                         m, activeRank, 0u, eps));
		if (ac.heroEnabled && complementControlStep)
		{
			heroEdge =
			    compute_hero_hankel_edge(state,
			                             gz,
			                             scoutControlRank,
			                             activeRank,
			                             n,
			                             heroLagHorizon,
			                             statScaleSq,
			                             prismTailMean,
			                             eps,
			                             &heroSigma);
			if (!atlas_isfinite(heroEdge))
				heroEdge = 0.0f;
			if (!atlas_isfinite(heroSigma))
				heroSigma = 0.0f;
			state.lastHeroEdge = heroEdge;
			state.lastHeroSigma = heroSigma;
		}
		if (!ac.heroEnabled && ac.resolveEnabled && complementControlStep)
		{
			const bool useScoutProbe = (scoutControlRank > 0u);
			const unsigned int probeRank =
			    useScoutProbe ? scoutControlRank : informativeComplementRank;
			resolvePredictiveEdge =
			    compute_resolve_predictive_edge(state,
			                                    probeRank,
			                                    useScoutProbe,
			                                    activeRank,
			                                    n,
			                                    resolveLagHorizon,
			                                    statScaleSq,
			                                    prismTailMean,
			                                    eps);
			if (!atlas_isfinite(resolvePredictiveEdge))
				resolvePredictiveEdge = 0.0f;
			state.lastResolveEdge = resolvePredictiveEdge;
		}
		if (!ac.heroEnabled && ac.prismEnabled)
		{
			prismPredictiveEdge =
			    compute_prism_predictive_edge(gv,
			                                  state.prevGv,
			                                  informativeComplementRank,
			                                  n,
			                                  statScaleSq,
			                                  prismTailMean,
			                                  eps);
			if (!atlas_isfinite(prismPredictiveEdge))
				prismPredictiveEdge = 0.0f;
			state.lastPredictiveEdge = prismPredictiveEdge;
		}
		if (!adaptiveComplementController)
		{
			state.activeComplementRank = informativeComplementRank;
			reset_complement_trial(state);
		}
		else if (complementControlStep && ac.qrcEnabled)
		{
			if (state.activeComplementRank != 0u || state.trialComplementRank != 0u)
			{
				const unsigned int prevActiveComplementRank = state.activeComplementRank;
				reset_complement_trial(state);
				state.activeComplementRank = 0u;
				if (logger)
				{
					std::ostringstream oss;
					oss << "event=atlas_complement_rank_change reason=qrc_memory_only";
					if (tag) oss << " tag=" << tag;
					append_kv(oss, "step", state.step);
					append_kv(oss, "m", m);
					append_kv(oss, "n", n);
					append_kv(oss, "active_rank", activeRank);
					append_kv(oss, "complement_rank_cap", enabledComplementRank);
					append_kv(oss, "complement_rank_prev", prevActiveComplementRank);
					append_kv(oss, "complement_rank_new", 0u);
					append_kv(oss, "qrc_edge", qrcEdge);
					append_kv(oss, "qrc_sigma", qrcSigma);
					append_kv(oss, "qrc_pole", qrcPole);
					append_kv(oss, "qrc_horizontal_ratio", qrcHorizontalRatio);
					append_kv(oss, "qrc_control_gain", qrcControlGain);
					append_kv(oss, "qrc_edge_threshold", ac.qrcEdgeThreshold);
					logger->info("ATLAS", shmea::GString(oss.str().c_str()));
				}
			}
			else
			{
				reset_complement_trial(state);
			}
		}
		else if (complementControlStep && ac.riftEnabled)
		{
			if (state.activeComplementRank != 0u || state.trialComplementRank != 0u)
			{
				const unsigned int prevActiveComplementRank = state.activeComplementRank;
				reset_complement_trial(state);
				state.activeComplementRank = 0u;
				if (logger)
				{
					std::ostringstream oss;
					oss << "event=atlas_complement_rank_change reason=rift_memory_only";
					if (tag) oss << " tag=" << tag;
					append_kv(oss, "step", state.step);
					append_kv(oss, "m", m);
					append_kv(oss, "n", n);
					append_kv(oss, "active_rank", activeRank);
					append_kv(oss, "complement_rank_cap", enabledComplementRank);
					append_kv(oss, "complement_rank_prev", prevActiveComplementRank);
					append_kv(oss, "complement_rank_new", 0u);
					append_kv(oss, "rift_edge", riftEdge);
					append_kv(oss, "rift_sigma", riftSigma);
					append_kv(oss, "rift_pole", riftPole);
					append_kv(oss, "rift_horizontal_ratio", riftHorizontalRatio);
					append_kv(oss, "rift_area_energy", riftAreaEnergy);
					append_kv(oss, "rift_pred_r2", riftPredR2);
					append_kv(oss, "rift_edge_threshold", ac.riftEdgeThreshold);
					logger->info("ATLAS", shmea::GString(oss.str().c_str()));
				}
			}
			else
			{
				reset_complement_trial(state);
			}
		}
		else if (complementControlStep && ac.sparrowEnabled)
		{
			if (state.activeComplementRank != 0u || state.trialComplementRank != 0u)
			{
				const unsigned int prevActiveComplementRank = state.activeComplementRank;
				reset_complement_trial(state);
				state.activeComplementRank = 0u;
				if (logger)
				{
					std::ostringstream oss;
					oss << "event=atlas_complement_rank_change reason=sparrow_memory_only";
					if (tag) oss << " tag=" << tag;
					append_kv(oss, "step", state.step);
					append_kv(oss, "m", m);
					append_kv(oss, "n", n);
					append_kv(oss, "active_rank", activeRank);
					append_kv(oss, "complement_rank_cap", enabledComplementRank);
					append_kv(oss, "complement_rank_prev", prevActiveComplementRank);
					append_kv(oss, "complement_rank_new", 0u);
					append_kv(oss, "sparrow_edge", sparrowEdge);
					append_kv(oss, "sparrow_sigma", sparrowSigma);
					append_kv(oss, "sparrow_second_edge", sparrowSecondEdge);
					append_kv(oss, "sparrow_second_sigma", sparrowSecondSigma);
					append_kv(oss, "sparrow_active_modes", sparrowActiveModes);
					append_kv(oss, "sparrow_pole", sparrowPole);
					append_kv(oss, "sparrow_horizontal_ratio", sparrowHorizontalRatio);
					append_kv(oss, "sparrow_edge_threshold", ac.sparrowEdgeThreshold);
					append_kv(oss, "sparrow_auto_mode_gate", ac.sparrowAutoModeGate ? 1u : 0u);
					append_kv(oss, "sparrow_second_edge_threshold", ac.sparrowSecondEdgeThreshold);
					append_kv(oss, "sparrow_second_edge_fraction", ac.sparrowSecondEdgeFraction);
					logger->info("ATLAS", shmea::GString(oss.str().c_str()));
				}
			}
			else
			{
				reset_complement_trial(state);
			}
		}
		else if (complementControlStep && ac.ghostEnabled)
		{
			if (state.activeComplementRank != 0u || state.trialComplementRank != 0u)
			{
				const unsigned int prevActiveComplementRank = state.activeComplementRank;
				reset_complement_trial(state);
				state.activeComplementRank = 0u;
				if (logger)
				{
					std::ostringstream oss;
					oss << "event=atlas_complement_rank_change reason=ghost_memory_only";
					if (tag) oss << " tag=" << tag;
					append_kv(oss, "step", state.step);
					append_kv(oss, "m", m);
					append_kv(oss, "n", n);
					append_kv(oss, "active_rank", activeRank);
					append_kv(oss, "complement_rank_cap", enabledComplementRank);
					append_kv(oss, "complement_rank_prev", prevActiveComplementRank);
					append_kv(oss, "complement_rank_new", 0u);
					append_kv(oss, "ghost_edge", ghostEdge);
					append_kv(oss, "ghost_sigma", ghostSigma);
					append_kv(oss, "ghost_horizontal_ratio", ghostHorizontalRatio);
					append_kv(oss, "ghost_edge_threshold", ac.ghostEdgeThreshold);
					logger->info("ATLAS", shmea::GString(oss.str().c_str()));
				}
			}
			else
			{
				reset_complement_trial(state);
			}
		}
		else if (complementControlStep && ac.birchEnabled)
		{
			if (state.activeComplementRank != 0u || state.trialComplementRank != 0u)
			{
				const unsigned int prevActiveComplementRank = state.activeComplementRank;
				reset_complement_trial(state);
				state.activeComplementRank = 0u;
				if (logger)
				{
					std::ostringstream oss;
					oss << "event=atlas_complement_rank_change reason=birch_memory_only";
					if (tag) oss << " tag=" << tag;
					append_kv(oss, "step", state.step);
					append_kv(oss, "m", m);
					append_kv(oss, "n", n);
					append_kv(oss, "active_rank", activeRank);
					append_kv(oss, "complement_rank_cap", enabledComplementRank);
					append_kv(oss, "complement_rank_prev", prevActiveComplementRank);
					append_kv(oss, "complement_rank_new", 0u);
					append_kv(oss, "birch_edge", birchEdge);
					append_kv(oss, "birch_sigma", birchSigma);
					append_kv(oss, "birch_edge_threshold", ac.birchEdgeThreshold);
					logger->info("ATLAS", shmea::GString(oss.str().c_str()));
				}
			}
			else
			{
				reset_complement_trial(state);
			}
		}
		else if (complementControlStep
		         && ac.cobaltEnabled
		         && cobaltEdge < ac.cobaltEdgeThreshold)
		{
			if (state.activeComplementRank != 0u || state.trialComplementRank != 0u)
			{
				const unsigned int prevActiveComplementRank = state.activeComplementRank;
				reset_complement_trial(state);
				state.activeComplementRank = 0u;
				if (logger)
				{
					std::ostringstream oss;
					oss << "event=atlas_complement_rank_change reason=cobalt_edge";
					if (tag) oss << " tag=" << tag;
					append_kv(oss, "step", state.step);
					append_kv(oss, "m", m);
					append_kv(oss, "n", n);
					append_kv(oss, "active_rank", activeRank);
					append_kv(oss, "complement_rank_cap", enabledComplementRank);
					append_kv(oss, "complement_rank_prev", prevActiveComplementRank);
					append_kv(oss, "complement_rank_new", 0u);
					append_kv(oss, "cobalt_edge", cobaltEdge);
					append_kv(oss, "cobalt_sigma", cobaltSigma);
					append_kv(oss, "cobalt_edge_threshold", ac.cobaltEdgeThreshold);
					logger->info("ATLAS", shmea::GString(oss.str().c_str()));
				}
			}
			else
			{
				reset_complement_trial(state);
			}
		}
		else if (complementControlStep
		         && !ac.cobaltEnabled
		         && ac.heroEnabled
		         && heroEdge < ac.heroEdgeThreshold)
		{
			if (state.activeComplementRank != 0u || state.trialComplementRank != 0u)
			{
				const unsigned int prevActiveComplementRank = state.activeComplementRank;
				reset_complement_trial(state);
				state.activeComplementRank = 0u;
				if (logger)
				{
					std::ostringstream oss;
					oss << "event=atlas_complement_rank_change reason=hero_edge";
					if (tag) oss << " tag=" << tag;
					append_kv(oss, "step", state.step);
					append_kv(oss, "m", m);
					append_kv(oss, "n", n);
					append_kv(oss, "active_rank", activeRank);
					append_kv(oss, "complement_rank_cap", enabledComplementRank);
					append_kv(oss, "complement_rank_prev", prevActiveComplementRank);
					append_kv(oss, "complement_rank_new", 0u);
					append_kv(oss, "hero_edge", heroEdge);
					append_kv(oss, "hero_sigma", heroSigma);
					append_kv(oss, "hero_edge_threshold", ac.heroEdgeThreshold);
					logger->info("ATLAS", shmea::GString(oss.str().c_str()));
				}
			}
			else
			{
				reset_complement_trial(state);
			}
		}
		else if (complementControlStep
		         && !ac.cobaltEnabled
		         && !ac.heroEnabled
		         && ac.resolveEnabled
		         && resolvePredictiveEdge < ac.resolvePredictiveEdgeThreshold)
		{
			if (state.activeComplementRank != 0u || state.trialComplementRank != 0u)
			{
				const unsigned int prevActiveComplementRank = state.activeComplementRank;
				reset_complement_trial(state);
				state.activeComplementRank = 0u;
				if (logger)
				{
					std::ostringstream oss;
					oss << "event=atlas_complement_rank_change reason=resolve_predictive_edge";
					if (tag) oss << " tag=" << tag;
					append_kv(oss, "step", state.step);
					append_kv(oss, "m", m);
					append_kv(oss, "n", n);
					append_kv(oss, "active_rank", activeRank);
					append_kv(oss, "complement_rank_cap", enabledComplementRank);
					append_kv(oss, "complement_rank_prev", prevActiveComplementRank);
					append_kv(oss, "complement_rank_new", 0u);
					append_kv(oss, "resolve_predictive_edge", resolvePredictiveEdge);
					append_kv(oss, "resolve_edge_threshold", ac.resolvePredictiveEdgeThreshold);
					logger->info("ATLAS", shmea::GString(oss.str().c_str()));
				}
			}
			else
			{
				reset_complement_trial(state);
			}
		}
		else if (complementControlStep
		         && !ac.cobaltEnabled
		         && !ac.heroEnabled
		         && !ac.resolveEnabled
		         && ac.prismEnabled
		         && prismPredictiveEdge < ac.prismPredictiveEdgeThreshold)
		{
			if (state.activeComplementRank != 0u || state.trialComplementRank != 0u)
			{
				const unsigned int prevActiveComplementRank = state.activeComplementRank;
				reset_complement_trial(state);
				state.activeComplementRank = 0u;
				if (logger)
				{
					std::ostringstream oss;
					oss << "event=atlas_complement_rank_change reason=prism_predictive_edge";
					if (tag) oss << " tag=" << tag;
					append_kv(oss, "step", state.step);
					append_kv(oss, "m", m);
					append_kv(oss, "n", n);
					append_kv(oss, "active_rank", activeRank);
					append_kv(oss, "complement_rank_cap", enabledComplementRank);
					append_kv(oss, "complement_rank_prev", prevActiveComplementRank);
					append_kv(oss, "complement_rank_new", 0u);
					append_kv(oss, "prism_predictive_edge", prismPredictiveEdge);
					append_kv(oss, "prism_edge_threshold", ac.prismPredictiveEdgeThreshold);
					logger->info("ATLAS", shmea::GString(oss.str().c_str()));
				}
			}
			else
			{
				reset_complement_trial(state);
			}
		}
		else if (complementControlStep)
		{
			const unsigned int prevActiveComplementRank = state.activeComplementRank;
			double scoutLambda = 0.0;
			double birthKelly = 0.0;
			double birthScout = 0.0;
			state.activeComplementRank =
			    choose_active_complement_rank(state,
			                                  state.scratch_complementEigVal,
			                                  state.scratch_complementEigVec,
			                                  scoutEigVal.empty() ? 0 : &scoutEigVal,
			                                  scoutEigVec.empty() ? 0 : &scoutEigVec,
			                                  informativeComplementRank,
			                                  state.activeComplementRank,
			                                  compute_active_trace(state, activeRank),
			                                  state.totalTrace,
			                                  m,
			                                  activeRank,
			                                  generalizedScoutEnabled,
			                                  statScaleSq,
			                                  eps,
			                                  &scoutLambda,
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
			complementScoutLambda = static_cast<float>(scoutLambda);
			if (state.activeComplementRank > informativeComplementRank)
				state.activeComplementRank = informativeComplementRank;
			if (logger && state.activeComplementRank != prevActiveComplementRank)
			{
				const double closedTrace = std::max<double>(
				    static_cast<double>(state.totalTrace),
				    compute_active_trace(state, activeRank)
				        + sum_leading_spectrum(state.scratch_complementEigVal,
				                               informativeComplementRank));
				const double prevTrace =
				    sum_leading_spectrum(state.scratch_complementEigVal, prevActiveComplementRank);
				const double tailMean =
				    complement_tail_mean(closedTrace,
				                         compute_active_trace(state, activeRank),
				                         prevTrace,
				                         m,
				                         activeRank,
				                         prevActiveComplementRank,
				                         eps);
				std::ostringstream oss;
				oss << "event=atlas_complement_rank_change";
				if (tag) oss << " tag=" << tag;
				append_kv(oss, "step", state.step);
				append_kv(oss, "m", m);
				append_kv(oss, "n", n);
				append_kv(oss, "active_rank", activeRank);
				append_kv(oss, "complement_rank_cap", enabledComplementRank);
				append_kv(oss, "complement_rank_prev", prevActiveComplementRank);
				append_kv(oss, "complement_rank_new", state.activeComplementRank);
				append_kv(oss, "tail_mean", static_cast<float>(tailMean));
				append_kv(oss, "scout_lambda", complementScoutLambda);
				append_kv(oss, "birth_kelly", complementBirthKelly);
				append_kv(oss, "birth_scout", complementScoutTop);
				append_kv(oss, "birth_projected", static_cast<float>(complementScoutProjected));
				append_kv(oss, "trial_kelly", static_cast<float>(complementTrialKelly));
				append_kv(oss, "trial_alignment", static_cast<float>(complementTrialAlignment));
				append_kv(oss, "trial_contam", static_cast<float>(complementTrialContamination));
				append_kv(oss, "trial_return", static_cast<float>(complementTrialReturn));
				append_kv(oss, "trial_score", static_cast<float>(complementTrialScore));
				append_kv(oss, "next_mode",
				          (prevActiveComplementRank < informativeComplementRank)
				              ? state.scratch_complementEigVal[prevActiveComplementRank]
				              : 0.0f);
				append_kv(oss, "weakest_active",
				          (prevActiveComplementRank > 0u)
				              ? state.scratch_complementEigVal[prevActiveComplementRank - 1u]
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
	unsigned int effectiveComplementRank = state.activeComplementRank;
	activeSectorTrace =
	    sum_leading_spectrum(state.scratch_complementEigVal, effectiveComplementRank);
	{
		const double activeTrace = compute_active_trace(state, activeRank);
		const double fullBlockTrace =
		    sum_leading_spectrum(state.scratch_complementEigVal, informativeComplementRank);
		const double closedTrace = std::max<double>(static_cast<double>(state.totalTrace),
		                                            activeTrace + fullBlockTrace);
		complementTailMean = static_cast<float>(
		    complement_tail_mean(closedTrace, activeTrace, activeSectorTrace,
		                         m, activeRank, effectiveComplementRank, eps));
	}
	state.sigma2 = compute_complement_sigma2(state, activeRank, activeSectorTrace,
	                                         effectiveComplementRank, m, eps);
	if (!atlas_isfinite(state.sigma2))
	{
		state.sigma2 = eps;
		recovered = true;
	}

	// === Step 7: Full-space baseline update ===
	// W -= min(lr / (effSigma2 + eps), kappaMax * lr) * G
	// Baseline rate is capped at kappaMax*lr to prevent divergence
	// when sigma2 converges to small complement variance.
	// effSigma2 includes bias correction so early steps get meaningful preconditioning.
	const float effSigma2 = state.sigma2 * bcFactor;
	const float baselineRate = atlas_clamped_rate(lr, effSigma2, eps, kappaMax);
	state.lastBaselineRate = baselineRate;
	{
		const float baseScaled = -baselineRate * gScale;
		for (size_t idx = 0; idx < mn; ++idx)
			W[idx] += baseScaled * gW[idx];
	}

	// === Step 8: Subspace correction with optional PNG ===
	//
	// corrScale_c = baselineRate - min(lr/(effFisher_c+eps), kappaLr)
	//
	// This ADDS BACK the baseline step in subspace directions and REPLACES
	// it with Fisher-preconditioned step. The net update per direction:
	//   subspace c: -min(lr/(effFisher_c+eps), kappaLr) * gPred_c  (Fisher-preconditioned)
	//   complement: -baselineRate * G_perp (baseline-preconditioned)
	//
	// gPred = (1+mu)*gz - mu*prevGz  (PNG temporal extrapolation)
	const float onePlusMu = 1.0f + state.mu;
	const float negMu = -state.mu;
	float sectorRate = 0.0f;
	const bool qrcMemoryEnabled =
	    ac.qrcEnabled
	    && atlas_nonnegative_finite(ac.qrcMemoryScale, 0.0f) > 0.0f
	    && !qrcLeftMode.empty()
	    && !qrcLatent.empty();
	const float qrcMemoryScale =
	    qrcMemoryEnabled ? atlas_nonnegative_finite(ac.qrcMemoryScale, 0.0f) : 0.0f;
	const bool riftMemoryEnabled =
	    !qrcMemoryEnabled
	    && ac.riftEnabled
	    && atlas_nonnegative_finite(ac.riftMemoryScale, 0.0f) > 0.0f
	    && !riftLeftMode.empty()
	    && !riftLatent.empty();
	const float riftMemoryScale =
	    riftMemoryEnabled ? atlas_nonnegative_finite(ac.riftMemoryScale, 0.0f) : 0.0f;
	const bool qbrtMemoryEnabled =
	    !qrcMemoryEnabled
	    && !riftMemoryEnabled
	    && ac.qbrtEnabled
	    && atlas_nonnegative_finite(ac.qbrtMemoryScale, 0.0f) > 0.0f
	    && !qbrtLeftMode.empty()
	    && !qbrtLatent.empty();
	const float qbrtMemoryScale =
	    qbrtMemoryEnabled ? atlas_nonnegative_finite(ac.qbrtMemoryScale, 0.0f) : 0.0f;
	const bool orbitMemoryEnabled =
	    !qrcMemoryEnabled
	    && !riftMemoryEnabled
	    && !qbrtMemoryEnabled
	    &&
	    orbitEligible
	    && ac.orbitEnabled
	    && atlas_nonnegative_finite(ac.orbitMemoryScale, 0.0f) > 0.0f
	    && !orbitLeftMode.empty()
	    && !orbitLatent.empty();
	const float orbitMemoryScale =
	    orbitMemoryEnabled ? atlas_nonnegative_finite(ac.orbitMemoryScale, 0.0f) : 0.0f;
	const bool sparrowMemoryEnabled =
	    !qrcMemoryEnabled
	    && !riftMemoryEnabled
	    &&
	    !orbitMemoryEnabled
	    && ac.sparrowEnabled
	    && atlas_nonnegative_finite(ac.sparrowMemoryScale, 0.0f) > 0.0f
	    && !sparrowLeftMode.empty()
	    && !sparrowLatent.empty();
	const float sparrowMemoryScale =
	    sparrowMemoryEnabled ? atlas_nonnegative_finite(ac.sparrowMemoryScale, 0.0f) : 0.0f;
	const float sparrowTrust =
	    std::max(0.0f,
	             std::min(1.0f,
	                      atlas_nonnegative_finite(state.externalSparrowTrust, 1.0f)));
	const bool ghostMemoryEnabled =
	    !qrcMemoryEnabled
	    && !riftMemoryEnabled
	    &&
	    !ac.sparrowEnabled
	    && ac.ghostEnabled
	    && atlas_nonnegative_finite(ac.ghostMemoryScale, 0.0f) > 0.0f
	    && ghostLagHorizon > 0u
	    && !ghostLeftMode.empty()
	    && !ghostRightMode.empty()
	    && !ghostPastStack.empty();
	const float ghostMemoryScale =
	    ghostMemoryEnabled ? atlas_nonnegative_finite(ac.ghostMemoryScale, 0.0f) : 0.0f;
	const bool birchMemoryEnabled =
	    !qrcMemoryEnabled
	    && !riftMemoryEnabled
	    &&
	    !ac.sparrowEnabled
	    && !ac.ghostEnabled
	    && ac.birchEnabled
	    && atlas_nonnegative_finite(ac.birchMemoryScale, 0.0f) > 0.0f
	    && birchPastHorizon > 0u
	    && birchFutureHorizon > 0u;
	const float birchMemoryScale =
	    birchMemoryEnabled ? atlas_nonnegative_finite(ac.birchMemoryScale, 0.0f) : 0.0f;
	const bool cobaltMemoryEnabled =
	    !qrcMemoryEnabled
	    && !riftMemoryEnabled
	    &&
	    !ac.sparrowEnabled
	    && !ac.ghostEnabled
	    && !ac.birchEnabled
	    && ac.cobaltEnabled
	    && atlas_nonnegative_finite(ac.cobaltMemoryScale, 0.0f) > 0.0f
	    && cobaltLagHorizon > 0u;
	const float cobaltMemoryScale =
	    cobaltMemoryEnabled ? atlas_nonnegative_finite(ac.cobaltMemoryScale, 0.0f) : 0.0f;
	const bool heroMemoryEnabled =
	    !qrcMemoryEnabled
	    && !riftMemoryEnabled
	    &&
	    !ac.sparrowEnabled
	    && !ac.ghostEnabled
	    && !ac.birchEnabled
	    && !ac.cobaltEnabled
	    && ac.heroEnabled
	    && atlas_nonnegative_finite(ac.heroMemoryScale, 0.0f) > 0.0f
	    && heroLagHorizon > 0u;
	const float heroMemoryScale =
	    heroMemoryEnabled ? atlas_nonnegative_finite(ac.heroMemoryScale, 0.0f) : 0.0f;
	const bool resolveMemoryEnabled =
	    !qrcMemoryEnabled
	    && !riftMemoryEnabled
	    &&
	    !ac.sparrowEnabled
	    && !ac.ghostEnabled
	    && !ac.birchEnabled
	    && !ac.cobaltEnabled
	    && !ac.heroEnabled
	    && ac.resolveEnabled
	    && atlas_nonnegative_finite(ac.resolveMemoryScale, 0.0f) > 0.0f
	    && resolveLagHorizon > 0u;
	const float resolveMemoryScale =
	    resolveMemoryEnabled ? atlas_nonnegative_finite(ac.resolveMemoryScale, 0.0f) : 0.0f;
	const bool prismMemoryEnabled =
	    !qrcMemoryEnabled
	    && !riftMemoryEnabled
	    &&
	    !ac.sparrowEnabled
	    && !ac.ghostEnabled
	    && !ac.birchEnabled
	    && !ac.cobaltEnabled
	    && !ac.heroEnabled
	    && !ac.resolveEnabled
	    && ac.prismEnabled
	    && atlas_nonnegative_finite(ac.prismMemoryScale, 0.0f) > 0.0f;
	const unsigned int prismLagHorizon = (ac.prismLagHorizon > 2u) ? 2u : ac.prismLagHorizon;
	const float prismMemoryScale =
	    prismMemoryEnabled ? atlas_nonnegative_finite(ac.prismMemoryScale, 0.0f) : 0.0f;
	const double riftTransferScale =
	    (riftMemoryEnabled && riftEdge > ac.riftEdgeThreshold)
	        ? std::min<double>(1.0,
	                           std::max<double>(
	                               0.0,
	                               (static_cast<double>(riftEdge) - static_cast<double>(ac.riftEdgeThreshold))
	                                   / std::max<double>(1e-12,
	                                                      1.0 - static_cast<double>(ac.riftEdgeThreshold))))
	        : 0.0;
	const double qbrtTransferScale =
	    (qbrtMemoryEnabled && qbrtEdge > ac.qbrtEdgeThreshold)
	        ? std::min<double>(1.0,
	                           std::max<double>(0.0, static_cast<double>(qbrtEdge)))
	        : 0.0;
	const double qrcTransferScale =
	    (qrcMemoryEnabled && qrcEdge > ac.qrcEdgeThreshold)
	        ? std::min<double>(1.0,
	                           std::max<double>(0.0, static_cast<double>(qrcEdge)))
	        : 0.0;
	const double orbitTransferScale =
	    (orbitMemoryEnabled && orbitEdge > ac.orbitEdgeThreshold)
	        ? std::min<double>(1.0,
	                           std::max<double>(
	                               0.0,
	                               (static_cast<double>(orbitEdge) - static_cast<double>(ac.orbitEdgeThreshold))
	                                   / std::max<double>(1e-12,
	                                                      1.0 - static_cast<double>(ac.orbitEdgeThreshold))))
	        : 0.0;
	const double sparrowTransferScale =
	    (sparrowMemoryEnabled && sparrowEdge > ac.sparrowEdgeThreshold)
	        ? std::min<double>(1.0,
	                           std::max<double>(
	                               0.0,
	                               (static_cast<double>(sparrowEdge) - static_cast<double>(ac.sparrowEdgeThreshold))
	                                   / std::max<double>(1e-12,
	                                                      1.0 - static_cast<double>(ac.sparrowEdgeThreshold))))
	        : 0.0;
	const double ghostTransferScale =
	    (ghostMemoryEnabled && ghostEdge > ac.ghostEdgeThreshold)
	        ? std::min<double>(1.0,
	                           std::max<double>(0.0, static_cast<double>(ghostEdge)))
	        : 0.0;
	if (riftMemoryEnabled && riftTransferScale > 0.0)
	{
		riftMemoryGain = static_cast<float>(
		    static_cast<double>(riftMemoryScale) * riftTransferScale);
	}
	if (qbrtMemoryEnabled && qbrtTransferScale > 0.0)
	{
		qbrtMemoryGain = static_cast<float>(
		    static_cast<double>(qbrtMemoryScale)
		    * qbrtTransferScale
		    * std::max<double>(0.0, static_cast<double>(qbrtHorizontalRatio)));
	}
	if (qrcMemoryEnabled && qrcTransferScale > 0.0)
	{
		qrcMemoryGain = static_cast<float>(
		    static_cast<double>(qrcMemoryScale)
		    * qrcTransferScale
		    * std::max<double>(0.0, static_cast<double>(qrcControlGain)));
	}
	if (orbitMemoryEnabled && orbitTransferScale > 0.0)
	{
		orbitMemoryGain = static_cast<float>(
		    static_cast<double>(orbitMemoryScale)
		    * orbitTransferScale
		    * std::max<double>(0.0, static_cast<double>(orbitHorizontalRatio)));
	}
	if (sparrowMemoryEnabled && sparrowTransferScale > 0.0)
	{
		sparrowMemoryGain = static_cast<float>(
		    static_cast<double>(sparrowMemoryScale)
		    * sparrowTransferScale
		    * std::max<double>(0.0, static_cast<double>(sparrowHorizontalRatio))
		    * static_cast<double>(sparrowTrust));
	}
	std::vector<float> ghostPastSignal;
	if (ghostMemoryEnabled && ghostTransferScale > 0.0)
	{
		ghostPastSignal.assign(static_cast<size_t>(n), 0.0f);
		const unsigned int pastRows =
		    static_cast<unsigned int>(ghostRightMode.size());
		for (unsigned int j = 0; j < n; ++j)
		{
			double sum = 0.0;
			for (unsigned int row = 0; row < pastRows; ++row)
			{
				sum += static_cast<double>(ghostRightMode[row])
				     * static_cast<double>(ghostPastStack[static_cast<size_t>(row) * n + j]);
			}
			ghostPastSignal[j] = static_cast<float>(sum);
		}
		ghostMemoryGain = static_cast<float>(
		    static_cast<double>(ghostMemoryScale)
		    * ghostTransferScale
		    * std::max<double>(0.0, static_cast<double>(ghostHorizontalRatio)));
	}

	// Precompute corrected[r,n] = corrScale[c] * gPred[c,n]
	std::vector<float>& corrected = state.scratch_corrected;
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		const float effFisher = state.fisherDiag[c] * bcFactor;
		const float fisherLR = atlas_clamped_rate(lr, effFisher, eps, kappaMax);
		const float corrScale = baselineRate - fisherLR;
		double heroRho = 0.0;
		double heroGain = 0.0;
		double cobaltRho = 0.0;
		double cobaltGain = 0.0;
		double birchRho = 0.0;
		double birchGain = 0.0;
		double resolveRho = 0.0;
		double resolveGain = 0.0;
		double lag1 = 0.0;
		double lag2 = 0.0;
		double mem1 = 0.0;
		double mem2 = 0.0;
		if (sparrowMemoryEnabled)
		{
			// SPARROW uses a rank-1 streaming Petrov mode and a stable one-pole
			// latent state, so the scalar gain is logged once per step and the
			// row-specific shaping comes from the retained left mode.
		}
		else if (ghostMemoryEnabled)
		{
			// GHOST uses a rank-1 biorthogonal transfer mode rather than a
			// per-row autoregressive pole fit. The scalar gain is logged once
			// per step and the row-specific shaping comes from the left mode.
		}
		else if (birchMemoryEnabled)
		{
			birchRho =
			    fit_resolve_memory_pole(state, gz, c, n, birchPastHorizon, statScaleSq, eps);
			if (!(birchRho > 0.0) || !std::isfinite(birchRho))
				birchRho = 0.0;
			const double transferScale =
			    (birchEdge > ac.birchEdgeThreshold)
			        ? std::min<double>(1.0,
			                           std::max<double>(0.0,
			                                            static_cast<double>(birchSigma) - 1.0))
			        : 0.0;
			double gainSum = 0.0;
			for (unsigned int lag = 1u; lag <= birchPastHorizon; ++lag)
				gainSum += static_cast<double>(birchMemoryScale)
				        * transferScale
				        * std::pow(birchRho, static_cast<double>(lag));
			birchGain = gainSum;
			birchMemoryGain += static_cast<float>(birchGain);
		}
		else if (cobaltMemoryEnabled)
		{
			cobaltRho =
			    fit_resolve_memory_pole(state, gz, c, n, cobaltLagHorizon, statScaleSq, eps);
			if (!(cobaltRho > 0.0) || !std::isfinite(cobaltRho))
				cobaltRho = 0.0;
			double gainSum = 0.0;
			const double transferScale =
			    std::min<double>(1.0, std::max<double>(0.0, static_cast<double>(cobaltSigma)));
			for (unsigned int lag = 1u; lag <= cobaltLagHorizon; ++lag)
				gainSum += static_cast<double>(cobaltMemoryScale)
				        * transferScale
				        * std::pow(cobaltRho, static_cast<double>(lag));
			cobaltGain = gainSum;
			cobaltMemoryGain += static_cast<float>(cobaltGain);
		}
		else if (heroMemoryEnabled)
		{
			heroRho =
			    fit_resolve_memory_pole(state, gz, c, n, heroLagHorizon, statScaleSq, eps);
			if (!(heroRho > 0.0) || !std::isfinite(heroRho))
				heroRho = 0.0;
			double gainSum = 0.0;
			for (unsigned int lag = 1u; lag <= heroLagHorizon; ++lag)
				gainSum += static_cast<double>(heroMemoryScale)
				        * std::pow(heroRho, static_cast<double>(lag));
			heroGain = gainSum;
			heroMemoryGain += static_cast<float>(heroGain);
		}
		else if (resolveMemoryEnabled)
		{
			resolveRho =
			    fit_resolve_memory_pole(state, gz, c, n, resolveLagHorizon, statScaleSq, eps);
			if (!(resolveRho > 0.0) || !std::isfinite(resolveRho))
				resolveRho = 0.0;
			double gainSum = 0.0;
			for (unsigned int lag = 1u; lag <= resolveLagHorizon; ++lag)
				gainSum += static_cast<double>(resolveMemoryScale)
				        * std::pow(resolveRho, static_cast<double>(lag));
			resolveGain = gainSum;
			resolveKernelRho += static_cast<float>(resolveRho);
			resolveMemoryGain += static_cast<float>(resolveGain);
		}
		else if (prismMemoryEnabled)
		{
			lag1 = std::max<double>(0.0,
			                        prism_row_cosine(&gz[static_cast<size_t>(c) * n],
			                                         &state.prevGz[static_cast<size_t>(c) * n],
			                                         n));
			mem1 = static_cast<double>(prismMemoryScale) * lag1;
			if (prismLagHorizon > 1u)
			{
				lag2 = std::max<double>(0.0,
				                        prism_row_cosine(&gz[static_cast<size_t>(c) * n],
				                                         &state.prevPrevGz[static_cast<size_t>(c) * n],
				                                         n));
				mem2 = 0.5 * static_cast<double>(prismMemoryScale) * lag2;
			}
			prismLag1 += static_cast<float>(lag1);
			prismLag2 += static_cast<float>(lag2);
			prismMemoryGain += static_cast<float>(mem1 + mem2);
		}
		for (unsigned int j = 0; j < n; ++j)
		{
			const size_t cj = static_cast<size_t>(c) * n + j;
			double gPred =
			    onePlusMu * static_cast<double>(gz[cj])
			    + negMu * static_cast<double>(state.prevGz[cj])
			    - mem1 * static_cast<double>(state.prevGz[cj])
			    - mem2 * static_cast<double>(state.prevPrevGz[cj]);
			if (riftMemoryEnabled && riftTransferScale > 0.0)
			{
				gPred -= static_cast<double>(riftMemoryGain)
				      * static_cast<double>(riftLeftMode[c])
				      * static_cast<double>(riftLatent[j]);
			}
			else if (qrcMemoryEnabled && qrcTransferScale > 0.0)
			{
				gPred -= static_cast<double>(qrcMemoryGain)
				      * static_cast<double>(qrcLeftMode[c])
				      * static_cast<double>(qrcLatent[j]);
			}
			else if (qbrtMemoryEnabled && qbrtTransferScale > 0.0)
			{
				gPred -= static_cast<double>(qbrtMemoryGain)
				      * static_cast<double>(qbrtLeftMode[c])
				      * static_cast<double>(qbrtLatent[j]);
			}
			else if (orbitMemoryEnabled && orbitTransferScale > 0.0)
			{
				gPred -= static_cast<double>(orbitMemoryGain)
				      * static_cast<double>(orbitLeftMode[c])
				      * static_cast<double>(orbitLatent[j]);
			}
			else if (sparrowMemoryEnabled && sparrowTransferScale > 0.0)
			{
				double sparrowCorrection = 0.0;
				for (unsigned int mode = 0; mode < sparrowActiveModes; ++mode)
				{
					const size_t leftIdx = static_cast<size_t>(mode) * state.r + c;
					const size_t latentIdx = static_cast<size_t>(mode) * n + j;
					if (leftIdx < sparrowLeftMode.size() && latentIdx < sparrowLatent.size())
					{
						sparrowCorrection += static_cast<double>(sparrowLeftMode[leftIdx])
						                  * static_cast<double>(sparrowLatent[latentIdx]);
					}
				}
				gPred -= static_cast<double>(sparrowMemoryGain) * sparrowCorrection;
			}
			else if (ghostMemoryEnabled && ghostTransferScale > 0.0)
			{
				gPred -= static_cast<double>(ghostMemoryGain)
				      * static_cast<double>(ghostLeftMode[c])
				      * static_cast<double>(ghostPastSignal[j]);
			}
			else if (birchMemoryEnabled)
			{
				const double transferScale =
				    (birchEdge > ac.birchEdgeThreshold)
				        ? std::min<double>(1.0,
				                           std::max<double>(0.0,
				                                            static_cast<double>(birchSigma) - 1.0))
				        : 0.0;
				for (unsigned int lag = 1u; lag <= birchPastHorizon; ++lag)
				{
					const size_t histBase =
					    (static_cast<size_t>(lag) - 1u) * static_cast<size_t>(state.r) * n;
					const double kLag =
					    static_cast<double>(birchMemoryScale)
					    * transferScale
					    * std::pow(birchRho, static_cast<double>(lag));
					gPred -= kLag * static_cast<double>(
					    state.resolveGzHistory[histBase + cj]);
				}
			}
			else if (cobaltMemoryEnabled)
			{
				for (unsigned int lag = 1u; lag <= cobaltLagHorizon; ++lag)
				{
					const size_t histBase =
					    (static_cast<size_t>(lag) - 1u) * static_cast<size_t>(state.r) * n;
					const double kLag =
					    static_cast<double>(cobaltMemoryScale)
					    * std::min<double>(1.0, std::max<double>(0.0, static_cast<double>(cobaltSigma)))
					    * std::pow(cobaltRho, static_cast<double>(lag));
					gPred -= kLag * static_cast<double>(
					    state.resolveGzHistory[histBase + cj]);
				}
			}
			else if (heroMemoryEnabled)
			{
				for (unsigned int lag = 1u; lag <= heroLagHorizon; ++lag)
				{
					const size_t histBase =
					    (static_cast<size_t>(lag) - 1u) * static_cast<size_t>(state.r) * n;
					const double kLag =
					    static_cast<double>(heroMemoryScale)
					    * std::pow(heroRho, static_cast<double>(lag));
					gPred -= kLag * static_cast<double>(
					    state.resolveGzHistory[histBase + cj]);
				}
			}
			else if (resolveMemoryEnabled)
			{
				for (unsigned int lag = 1u; lag <= resolveLagHorizon; ++lag)
				{
					const size_t histBase =
					    (static_cast<size_t>(lag) - 1u) * static_cast<size_t>(state.r) * n;
					const double kLag =
					    static_cast<double>(resolveMemoryScale)
					    * std::pow(resolveRho, static_cast<double>(lag));
					gPred -= kLag * static_cast<double>(
					    state.resolveGzHistory[histBase + cj]);
				}
			}
			corrected[cj] = static_cast<float>(static_cast<double>(corrScale) * gPred);
		}
	}
	if (activeRank > 0u)
	{
		const float invActiveRank = 1.0f / static_cast<float>(activeRank);
		prismLag1 *= invActiveRank;
		prismLag2 *= invActiveRank;
		prismMemoryGain *= invActiveRank;
		resolveKernelRho *= invActiveRank;
		resolveMemoryGain *= invActiveRank;
		heroMemoryGain *= invActiveRank;
		cobaltMemoryGain *= invActiveRank;
		birchMemoryGain *= invActiveRank;
	}
	state.lastMemoryGain = prismMemoryGain;
	state.lastResolveKernelRho = resolveKernelRho;
	state.lastResolveMemoryGain = resolveMemoryGain;
	state.lastHeroMemoryGain = heroMemoryGain;
	state.lastCobaltMemoryGain = cobaltMemoryGain;
	state.lastBirchMemoryGain = birchMemoryGain;
	state.lastQbrtMemoryGain = qbrtMemoryGain;
	state.lastQrcMemoryGain = qrcMemoryGain;
	state.lastRiftMemoryGain = riftMemoryGain;
	state.lastOrbitMemoryGain = orbitMemoryGain;
	state.lastSparrowMemoryGain = sparrowMemoryGain;
	state.lastGhostMemoryGain = ghostMemoryGain;

	// W[m,n] += U[m,r] * corrected[r,n]
	glades::gemm::ab_accum(W, &basisPacked[0], &corrected[0], m, activeRank, n, 1.0f);

	if (enabledComplementRank > 0u && effectiveComplementRank > 0u)
	{
		std::vector<float>& correctedV = state.scratch_correctedV;
		const float sectorNominalLr = lr * atlas_nonnegative_finite(ac.complementLrScale, 0.0f);
		sectorRate = build_complement_correction_matrix(state,
		                                                baselineRate,
		                                                sectorNominalLr,
		                                                eps,
		                                                ac.complementKappaMax,
		                                                bcFactor,
		                                                effectiveComplementRank);
		multiply_left_block(&correctedV[0], &state.scratch_complementMat[0], &gv[0],
		                    complementRank, complementRank, n);
		glades::gemm::ab_accum(W, &state.V[0], &correctedV[0], m, complementRank, n, 1.0f);
	}
	else if (!state.scratch_correctedV.empty())
	{
		std::fill(state.scratch_correctedV.begin(), state.scratch_correctedV.end(), 0.0f);
	}

	// Guard against NaN/Inf propagation from corrupted U or corrected buffers.
	for (size_t idx = 0; idx < mn; ++idx)
	{
		if (!atlas_isfinite(W[idx]))
		{
			W[idx] = 0.0f;
			recovered = true;
			if (logger)
			{
				std::ostringstream oss;
				oss << "event=atlas_nonfinite_weight";
				append_kv(oss, "step", state.step);
				append_kv(oss, "idx", static_cast<unsigned long long>(idx));
				if (tag) oss << " tag=" << tag;
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
		}
	}

	// Compute update norm for diagnostics (only on diag steps)
	double updateNormSq = 0.0;
	if (diagStep)
	{
		for (size_t idx = 0; idx < rn; ++idx)
			updateNormSq += static_cast<double>(corrected[idx]) * static_cast<double>(corrected[idx]);
		if (enabledComplementRank > 0u && effectiveComplementRank > 0u)
		{
			const std::vector<float>& correctedV = state.scratch_correctedV;
			for (size_t idx = 0; idx < cn; ++idx)
				updateNormSq += static_cast<double>(correctedV[idx]) * static_cast<double>(correctedV[idx]);
		}
	}

	// === Step 9: Adapt prediction coefficient ===
	// Bidirectional adaptation: mu decreases when gradients oscillate (ratio > 0)
	// and slowly recovers via muGrowthRate when gradients are smooth.
	// newMu = mu * (1 - ratio) + muGrowthRate * (muMax - mu)
	double errNormSq = 0.0;
	double gzNormSq = 0.0;
	if (state.step > 1ULL)
	{
		for (size_t idx = 0; idx < rn; ++idx)
		{
			const double e = static_cast<double>(gz[idx] - state.prevGz[idx]);
			errNormSq += e * e;
			const double g = static_cast<double>(gz[idx]);
			gzNormSq += g * g;
		}
		const double gzNorm = sqrt(gzNormSq);
		if (gzNorm > 1e-12)
		{
			const float ratio = static_cast<float>(sqrt(errNormSq) / (gzNorm + 1e-12));
			float newMu = state.mu * (1.0f - ratio)
			            + muGrowthRate * (muMax - state.mu);
			if (newMu < muMin) newMu = muMin;
			if (newMu > muMax) newMu = muMax;
			state.mu = newMu;
			if (!atlas_isfinite(state.mu))
			{
				state.mu = muMin;
				recovered = true;
			}
		}
	}

	// Keep the leading directions ordered by Fisher mass so active-rank truncation
	// uses the most informative prefix.
	sort_directions_by_fisher(state, activeRank, &gz, n);
	float shrinkFisherMin = state.fisherDiag[0];
	float shrinkFisherMax = state.fisherDiag[0];
	for (unsigned int c = 1u; c < activeRank; ++c)
	{
		const float f = state.fisherDiag[c];
		if (f < shrinkFisherMin) shrinkFisherMin = f;
		if (f > shrinkFisherMax) shrinkFisherMax = f;
	}
	if (ac.adaptiveRank && tSub > 0u
	    && (state.step % static_cast<unsigned long long>(tSub)) == 0ULL)
	{
		unsigned int targetRank =
		    choose_active_rank(state, activeRank, ac.rankCapture, ac.minActiveRank);
		if (activeRank > ac.minActiveRank
		    && ac.flatSpectrumThreshold > 1.0f
		    && shrinkFisherMin > 1e-12f
		    && (shrinkFisherMax / shrinkFisherMin) <= ac.flatSpectrumThreshold)
		{
			const unsigned int flatRank =
			    choose_flat_spectrum_rank(activeRank, ac.minActiveRank);
			if (flatRank < targetRank)
				targetRank = flatRank;
		}
		if (targetRank < state.activeRank)
			state.activeRank = targetRank;
		activeRank = state.activeRank;
		rn = static_cast<size_t>(activeRank) * static_cast<size_t>(n);
		informativeComplementRank =
		    effective_complement_rank(state, enabledComplementRank, m, activeRank);
		if (state.activeComplementRank > informativeComplementRank)
			state.activeComplementRank = informativeComplementRank;
		if (state.trialComplementRank > informativeComplementRank
		    || state.trialComplementRank <= state.activeComplementRank)
			reset_complement_trial(state);
		effectiveComplementRank = state.activeComplementRank;
		activeSectorTrace =
		    sum_leading_spectrum(state.scratch_complementEigVal, state.activeComplementRank);
		{
			const double activeTrace = compute_active_trace(state, activeRank);
			const double fullBlockTrace =
			    sum_leading_spectrum(state.scratch_complementEigVal, informativeComplementRank);
			const double closedTrace = std::max<double>(static_cast<double>(state.totalTrace),
			                                            activeTrace + fullBlockTrace);
			complementTailMean = static_cast<float>(
			    complement_tail_mean(closedTrace, activeTrace, activeSectorTrace,
			                         m, activeRank, effectiveComplementRank, eps));
		}
		state.sigma2 = compute_complement_sigma2(state, activeRank, activeSectorTrace,
		                                         state.activeComplementRank, m, eps);
	}

	// === Periodic diagnostics ===
	if (diagStep)
	{
		if (state.step <= 1ULL)
		{
			for (size_t idx = 0; idx < rn; ++idx)
			{
				const double g = static_cast<double>(gz[idx]);
				gzNormSq += g * g;
			}
		}

		float fMin = state.fisherDiag[0];
		float fMax = state.fisherDiag[0];
		double fSum = 0.0;
		for (unsigned int c = 0; c < activeRank; ++c)
		{
			const float f = state.fisherDiag[c];
			if (f < fMin) fMin = f;
			if (f > fMax) fMax = f;
			fSum += static_cast<double>(f);
		}
		const float effectiveRank = compute_effective_rank(state, activeRank);
		const float spectralEfficiency = (activeRank > 0u)
		    ? (effectiveRank / static_cast<float>(activeRank))
		    : 0.0f;
		const float top1Concentration = compute_topk_concentration(state, activeRank, 1u);
		const float top10Concentration = compute_topk_concentration(state, activeRank, 10u);
		const float fisherRatio = (fMin > 1e-12f) ? (fMax / fMin) : 0.0f;
		double activeTrace = 0.0;
		double sectorTrace = 0.0;
		double closureGap = 0.0;
		const float sigma2Closed =
		    compute_complement_sigma2(state, activeRank, activeSectorTrace,
		                              effectiveComplementRank, m, eps,
		                              &activeTrace, &sectorTrace, &closureGap);
		const float sigma2FisherRatio = (fSum > 1e-30)
		    ? static_cast<float>(sigma2Closed / (fSum / static_cast<double>(activeRank)))
		    : 0.0f;
		const float activeTraceCapture = (state.totalTrace > 1e-30f)
		    ? static_cast<float>(activeTrace / static_cast<double>(state.totalTrace))
		    : 0.0f;
		const float traceCapture = (state.totalTrace > 1e-30f)
		    ? static_cast<float>((activeTrace + sectorTrace) / static_cast<double>(state.totalTrace))
		    : 0.0f;
		const float sectorTraceCapture = (state.totalTrace > 1e-30f)
		    ? static_cast<float>(sectorTrace / static_cast<double>(state.totalTrace))
		    : 0.0f;
		std::ostringstream oss;
		oss << "event=atlas_step";
		if (tag) oss << " tag=" << tag;
		append_kv(oss, "step", state.step);
		append_kv(oss, "m", m);
		append_kv(oss, "n", n);
		append_kv(oss, "rank", r);
		append_kv(oss, "active_rank", activeRank);
		append_kv(oss, "complement_rank", enabledComplementRank);
		append_kv(oss, "complement_active_rank", state.activeComplementRank);
		append_kv(oss, "complement_effective_rank", effectiveComplementRank);
		append_kv(oss, "complement_trial_rank", state.trialComplementRank);
		append_kv(oss, "complement_trial_wins", state.trialComplementWins);
		append_kv(oss, "lr", lr);
		append_kv(oss, "mu", state.mu);
		append_kv(oss, "total_trace", state.totalTrace);
		append_kv(oss, "sigma2", state.sigma2);
		append_kv(oss, "baseline_rate", baselineRate);
		append_kv(oss, "gz_norm", static_cast<float>(sqrt(gzNormSq)));
		append_kv(oss, "update_norm", static_cast<float>(sqrt(updateNormSq)));
		append_kv(oss, "fisher_min", fMin);
		append_kv(oss, "fisher_max", fMax);
		append_kv(oss, "fisher_mean", static_cast<float>(fSum / static_cast<double>(activeRank)));
		append_kv(oss, "fisher_ratio", fisherRatio);
		append_kv(oss, "sigma2_fisher_ratio", sigma2FisherRatio);
		append_kv(oss, "active_trace_capture", activeTraceCapture);
		append_kv(oss, "trace_capture", traceCapture);
		append_kv(oss, "sector_trace_capture", sectorTraceCapture);
		append_kv(oss, "sector_fisher", static_cast<float>(activeSectorTrace));
		append_kv(oss, "complement_block_trace", state.complementFisher);
		append_kv(oss, "complement_tail_mean", complementTailMean);
		append_kv(oss, "complement_scout_top", complementScoutTop);
		append_kv(oss, "complement_scout_projected", static_cast<float>(complementScoutProjected));
		append_kv(oss, "complement_scout_lambda", complementScoutLambda);
		append_kv(oss, "complement_birth_kelly", complementBirthKelly);
		append_kv(oss, "complement_trial_kelly", static_cast<float>(complementTrialKelly));
		append_kv(oss, "complement_trial_alignment", static_cast<float>(complementTrialAlignment));
		append_kv(oss, "complement_trial_contam", static_cast<float>(complementTrialContamination));
		append_kv(oss, "complement_trial_return", static_cast<float>(complementTrialReturn));
		append_kv(oss, "complement_trial_score", static_cast<float>(complementTrialScore));
		append_kv(oss, "prism_predictive_edge", prismPredictiveEdge);
		append_kv(oss, "prism_lag1", prismLag1);
		append_kv(oss, "prism_lag2", prismLag2);
		append_kv(oss, "prism_memory_gain", prismMemoryGain);
		append_kv(oss, "resolve_predictive_edge", resolvePredictiveEdge);
		append_kv(oss, "resolve_kernel_rho", resolveKernelRho);
		append_kv(oss, "resolve_memory_gain", resolveMemoryGain);
		append_kv(oss, "hero_edge", heroEdge);
		append_kv(oss, "hero_sigma", heroSigma);
		append_kv(oss, "hero_memory_gain", heroMemoryGain);
		append_kv(oss, "cobalt_edge", cobaltEdge);
		append_kv(oss, "cobalt_sigma", cobaltSigma);
		append_kv(oss, "cobalt_memory_gain", cobaltMemoryGain);
		append_kv(oss, "birch_edge", birchEdge);
		append_kv(oss, "birch_sigma", birchSigma);
		append_kv(oss, "birch_memory_gain", birchMemoryGain);
		append_kv(oss, "orbit_edge", orbitEdge);
		append_kv(oss, "orbit_sigma", orbitSigma);
		append_kv(oss, "orbit_pole", orbitPole);
		append_kv(oss, "orbit_horizontal_ratio", orbitHorizontalRatio);
		append_kv(oss, "orbit_memory_gain", orbitMemoryGain);
		append_kv(oss, "sparrow_edge", sparrowEdge);
		append_kv(oss, "sparrow_sigma", sparrowSigma);
		append_kv(oss, "sparrow_second_edge", sparrowSecondEdge);
		append_kv(oss, "sparrow_second_sigma", sparrowSecondSigma);
		append_kv(oss, "sparrow_active_modes", sparrowActiveModes);
		append_kv(oss, "sparrow_pole", sparrowPole);
		append_kv(oss, "sparrow_horizontal_ratio", sparrowHorizontalRatio);
		append_kv(oss, "sparrow_trust", sparrowTrust);
		append_kv(oss, "sparrow_memory_gain", sparrowMemoryGain);
		append_kv(oss, "qbrt_edge", qbrtEdge);
		append_kv(oss, "qbrt_sigma", qbrtSigma);
		append_kv(oss, "qbrt_pole", qbrtPole);
		append_kv(oss, "qbrt_horizontal_ratio", qbrtHorizontalRatio);
		append_kv(oss, "qbrt_memory_gain", qbrtMemoryGain);
		append_kv(oss, "qrc_edge", qrcEdge);
		append_kv(oss, "qrc_sigma", qrcSigma);
		append_kv(oss, "qrc_pole", qrcPole);
		append_kv(oss, "qrc_horizontal_ratio", qrcHorizontalRatio);
		append_kv(oss, "qrc_control_gain", qrcControlGain);
		append_kv(oss, "qrc_memory_gain", qrcMemoryGain);
		append_kv(oss, "rift_edge", riftEdge);
		append_kv(oss, "rift_sigma", riftSigma);
		append_kv(oss, "rift_pole", riftPole);
		append_kv(oss, "rift_horizontal_ratio", riftHorizontalRatio);
		append_kv(oss, "rift_area_energy", riftAreaEnergy);
		append_kv(oss, "rift_pred_r2", riftPredR2);
		append_kv(oss, "rift_memory_gain", riftMemoryGain);
		append_kv(oss, "ghost_edge", ghostEdge);
		append_kv(oss, "ghost_sigma", ghostSigma);
		append_kv(oss, "ghost_horizontal_ratio", ghostHorizontalRatio);
		append_kv(oss, "ghost_memory_gain", ghostMemoryGain);
		append_kv(oss, "sector_rate", sectorRate);
		append_kv(oss, "closure_gap", static_cast<float>(closureGap));
		append_kv(oss, "effective_rank", effectiveRank);
		append_kv(oss, "spectral_efficiency", spectralEfficiency);
		append_kv(oss, "top1_concentration", top1Concentration);
		append_kv(oss, "top10_concentration", top10Concentration);
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}

	// === Step 10: Store compressed gradient for next step ===
	if (!state.resolveGzHistory.empty())
	{
		const size_t lagSlice = static_cast<size_t>(state.r) * static_cast<size_t>(n);
		for (unsigned int lag = kATLASResolveMaxLagHorizon - 1u; lag > 0u; --lag)
		{
			const size_t dstBase = static_cast<size_t>(lag) * lagSlice;
			const size_t srcBase = static_cast<size_t>(lag - 1u) * lagSlice;
			std::copy(state.resolveGzHistory.begin() + srcBase,
			          state.resolveGzHistory.begin() + srcBase + lagSlice,
			          state.resolveGzHistory.begin() + dstBase);
		}
		std::fill(state.resolveGzHistory.begin(),
		          state.resolveGzHistory.begin() + lagSlice,
		          0.0f);
		if (rn > 0u)
		{
			std::copy(gz.begin(), gz.begin() + rn, state.resolveGzHistory.begin());
		}
		for (unsigned int lag = 0u; lag < kATLASResolveMaxLagHorizon; ++lag)
		{
			const size_t base = static_cast<size_t>(lag) * lagSlice;
			if (lagSlice > rn)
			{
				std::fill(state.resolveGzHistory.begin() + base + rn,
				          state.resolveGzHistory.begin() + base + lagSlice,
				          0.0f);
			}
		}
	}
	if ((ac.heroEnabled || ac.cobaltEnabled || ac.birchEnabled || ac.ghostEnabled || ac.qbrtEnabled
	     || ac.qrcEnabled || ac.riftEnabled)
	    && !state.heroGwHistory.empty())
	{
		const size_t lagSlice = static_cast<size_t>(state.complementRank) * static_cast<size_t>(n);
		for (unsigned int lag = kATLASHeroMaxLagHorizon - 1u; lag > 0u; --lag)
		{
			const size_t dstBase = static_cast<size_t>(lag) * lagSlice;
			const size_t srcBase = static_cast<size_t>(lag - 1u) * lagSlice;
			std::copy(state.heroGwHistory.begin() + srcBase,
			          state.heroGwHistory.begin() + srcBase + lagSlice,
			          state.heroGwHistory.begin() + dstBase);
		}
		std::fill(state.heroGwHistory.begin(),
		          state.heroGwHistory.begin() + lagSlice,
		          0.0f);
		if (cn > 0u)
		{
			std::copy(state.scratch_gwScout.begin(),
			          state.scratch_gwScout.begin() + cn,
			          state.heroGwHistory.begin());
		}
	}
	if (rn > 0u)
		std::copy(gz.begin(), gz.begin() + rn, state.prevGz.begin());
	if (state.prevGz.size() > rn)
		std::fill(state.prevGz.begin() + rn, state.prevGz.end(), 0.0f);
	if (!state.resolveGzHistory.empty())
	{
		const size_t lagSlice = static_cast<size_t>(state.r) * static_cast<size_t>(n);
		const size_t prevBase = lagSlice;
		if (rn > 0u && state.resolveGzHistory.size() >= prevBase + rn)
		{
			std::copy(state.resolveGzHistory.begin() + prevBase,
			          state.resolveGzHistory.begin() + prevBase + rn,
			          state.prevPrevGz.begin());
		}
	}
	if (state.prevPrevGz.size() > rn)
		std::fill(state.prevPrevGz.begin() + rn, state.prevPrevGz.end(), 0.0f);
	if (enabledComplementRank > 0u)
		std::copy(gv.begin(), gv.begin() + cn, state.prevGv.begin());
	else if (!state.prevGv.empty())
		std::fill(state.prevGv.begin(), state.prevGv.end(), 0.0f);

	// === Step 11: Clear accumulated gradients ===
	std::memset(gW, 0, mn * sizeof(float));

	return !recovered;
}

bool update(WeightState& state, float* W, float* gW,
            unsigned int m, unsigned int n,
            float invBatch, float lr,
            float wd1, float wd2, float gradScale,
            const ATLASConfig& ac,
            glades::rng::Engine& rng,
            shmea::GLogger* logger,
            const char* tag)
{
	if (!state.initialized && m > 0u && n > 0u)
		initWeightState(state, m, n, ac.rank, ac.muMin, rng, logger);

	return applyStep(state, W, gW, m, n, invBatch, lr, wd1, wd2, gradScale,
	                 ac, rng, logger, tag);
}

static void bimap_init_identity_basis(std::vector<float>& basis,
                                      unsigned int dim,
                                      unsigned int rank)
{
	basis.assign(static_cast<size_t>(dim) * rank, 0.0f);
	for (unsigned int c = 0u; c < rank; ++c)
	{
		const unsigned int row = (dim > 0u) ? (c % dim) : 0u;
		basis[static_cast<size_t>(row) * rank + c] = 1.0f;
	}
	if (dim > 0u && rank > 0u)
		orthonormalize_active(&basis[0], 0, rank, dim, rank, 0);
}

static void bimap_invert_spd(const std::vector<float>& block,
                             unsigned int dim,
                             float eps,
                             std::vector<float>& invOut)
{
	invOut.assign(static_cast<size_t>(dim) * dim, 0.0f);
	if (dim == 0u)
		return;

	std::vector<float> sym(block);
	symmetrize_block(&sym[0], dim);
	std::vector<float> eigVec;
	std::vector<float> eigVal;
	jacobi_eigendecompose(&sym[0], dim, eigVec, eigVal);
	for (unsigned int k = 0u; k < dim; ++k)
	{
		const double lambda = std::max<double>(static_cast<double>(eigVal[k]),
		                                       static_cast<double>(eps));
		const double scale = 1.0 / lambda;
		for (unsigned int r = 0u; r < dim; ++r)
		{
			for (unsigned int c = 0u; c < dim; ++c)
			{
				invOut[static_cast<size_t>(r) * dim + c] +=
				    static_cast<float>(scale)
				    * eigVec[static_cast<size_t>(r) * dim + k]
				    * eigVec[static_cast<size_t>(c) * dim + k];
			}
		}
	}
}

static void bimap_refresh_low_rank_factors(BiMAPWeightState& state,
                                           const std::vector<float>& grad,
                                           unsigned int m,
                                           unsigned int n,
                                           unsigned int rankCap,
                                           unsigned int powerIters,
                                           float betaGeom,
                                           float rowMeanF,
                                           float colMeanF)
{
	const unsigned int rank = std::min(rankCap, std::min(m, n));
	if (rank == 0u || grad.size() != static_cast<size_t>(m) * n)
	{
		state.rowRank = 0u;
		state.colRank = 0u;
		state.lastRowCapture = 0.0f;
		state.lastColCapture = 0.0f;
		return;
	}

	if (state.rowBasis.size() != static_cast<size_t>(m) * rank)
		bimap_init_identity_basis(state.rowBasis, m, rank);
	if (state.colBasis.size() != static_cast<size_t>(n) * rank)
		bimap_init_identity_basis(state.colBasis, n, rank);
	if (state.rowEigVal.size() != rank)
		state.rowEigVal.assign(rank, 0.0f);
	if (state.colEigVal.size() != rank)
		state.colEigVal.assign(rank, 0.0f);

	std::vector<float> rowNext(static_cast<size_t>(m) * rank, 0.0f);
	std::vector<float> colNext(static_cast<size_t>(n) * rank, 0.0f);
	for (unsigned int iter = 0u; iter < std::max(1u, powerIters); ++iter)
	{
		for (unsigned int i = 0u; i < m; ++i)
		{
			for (unsigned int c = 0u; c < rank; ++c)
			{
				double sum = 0.0;
				for (unsigned int j = 0u; j < n; ++j)
					sum += static_cast<double>(grad[static_cast<size_t>(i) * n + j])
					     * static_cast<double>(state.colBasis[static_cast<size_t>(j) * rank + c]);
				rowNext[static_cast<size_t>(i) * rank + c] = static_cast<float>(sum);
			}
		}
		orthonormalize_active(&rowNext[0], &state.rowBasis[0], rank, m, rank, 0);
		state.rowBasis.swap(rowNext);

		for (unsigned int j = 0u; j < n; ++j)
		{
			for (unsigned int c = 0u; c < rank; ++c)
			{
				double sum = 0.0;
				for (unsigned int i = 0u; i < m; ++i)
					sum += static_cast<double>(grad[static_cast<size_t>(i) * n + j])
					     * static_cast<double>(state.rowBasis[static_cast<size_t>(i) * rank + c]);
				colNext[static_cast<size_t>(j) * rank + c] = static_cast<float>(sum);
			}
		}
		orthonormalize_active(&colNext[0], &state.colBasis[0], rank, n, rank, 0);
		state.colBasis.swap(colNext);
	}

	std::vector<float> rowProj(static_cast<size_t>(rank) * n, 0.0f);
	std::vector<float> colProj(static_cast<size_t>(m) * rank, 0.0f);
	for (unsigned int c = 0u; c < rank; ++c)
	{
		for (unsigned int j = 0u; j < n; ++j)
		{
			double sum = 0.0;
			for (unsigned int i = 0u; i < m; ++i)
				sum += static_cast<double>(state.rowBasis[static_cast<size_t>(i) * rank + c])
				     * static_cast<double>(grad[static_cast<size_t>(i) * n + j]);
			rowProj[static_cast<size_t>(c) * n + j] = static_cast<float>(sum);
		}
	}
	for (unsigned int i = 0u; i < m; ++i)
	{
		for (unsigned int c = 0u; c < rank; ++c)
		{
			double sum = 0.0;
			for (unsigned int j = 0u; j < n; ++j)
				sum += static_cast<double>(grad[static_cast<size_t>(i) * n + j])
				     * static_cast<double>(state.colBasis[static_cast<size_t>(j) * rank + c]);
			colProj[static_cast<size_t>(i) * rank + c] = static_cast<float>(sum);
		}
	}

	unsigned int rowActive = 0u;
	unsigned int colActive = 0u;
	for (unsigned int c = 0u; c < rank; ++c)
	{
		double rowEnergy = 0.0;
		for (unsigned int j = 0u; j < n; ++j)
		{
			const double v = static_cast<double>(rowProj[static_cast<size_t>(c) * n + j]);
			rowEnergy += v * v;
		}
		rowEnergy /= static_cast<double>(std::max(1u, n));
		const float rowSample =
		    static_cast<float>(rowEnergy / static_cast<double>(std::max(rowMeanF, 1.0e-6f)));
		const float rowExcess = std::max(0.0f, rowSample - 1.0f);
		state.rowEigVal[c] =
		    betaGeom * state.rowEigVal[c] + (1.0f - betaGeom) * rowExcess;
		if (state.rowEigVal[c] > 1.0e-3f)
			rowActive = c + 1u;

		double colEnergy = 0.0;
		for (unsigned int i = 0u; i < m; ++i)
		{
			const double v = static_cast<double>(colProj[static_cast<size_t>(i) * rank + c]);
			colEnergy += v * v;
		}
		colEnergy /= static_cast<double>(std::max(1u, m));
		const float colSample =
		    static_cast<float>(colEnergy / static_cast<double>(std::max(colMeanF, 1.0e-6f)));
		const float colExcess = std::max(0.0f, colSample - 1.0f);
		state.colEigVal[c] =
		    betaGeom * state.colEigVal[c] + (1.0f - betaGeom) * colExcess;
		if (state.colEigVal[c] > 1.0e-3f)
			colActive = c + 1u;
	}

	state.rowRank = rowActive;
	state.colRank = colActive;
	state.lastRowCapture = (rank > 0u)
	    ? (static_cast<float>(rowActive) / static_cast<float>(rank))
	    : 0.0f;
	state.lastColCapture = (rank > 0u)
	    ? (static_cast<float>(colActive) / static_cast<float>(rank))
	    : 0.0f;
}

static void bimap_apply_left_inverse(std::vector<float>& matrix,
                                     unsigned int m,
                                     unsigned int n,
                                     const std::vector<float>& diagMetric,
                                     const std::vector<float>& basis,
                                     const std::vector<float>& eigVal,
                                     unsigned int rank,
                                     float geomScale,
                                     float eps)
{
	if (matrix.size() != static_cast<size_t>(m) * n || diagMetric.size() != m)
		return;
	for (unsigned int i = 0u; i < m; ++i)
	{
		const float invDiag = 1.0f / std::max(diagMetric[i], eps);
		for (unsigned int j = 0u; j < n; ++j)
			matrix[static_cast<size_t>(i) * n + j] *= invDiag;
	}
	if (rank == 0u || geomScale <= 0.0f || basis.size() < static_cast<size_t>(m) * rank)
		return;

	std::vector<float> core(static_cast<size_t>(rank) * rank, 0.0f);
	for (unsigned int a = 0u; a < rank; ++a)
	{
		for (unsigned int b = 0u; b < rank; ++b)
		{
			double sum = 0.0;
			for (unsigned int i = 0u; i < m; ++i)
			{
				const double invDiag = 1.0 / std::max(static_cast<double>(diagMetric[i]),
				                                      static_cast<double>(eps));
				sum += invDiag
				     * static_cast<double>(basis[static_cast<size_t>(i) * rank + a])
				     * static_cast<double>(basis[static_cast<size_t>(i) * rank + b]);
			}
			core[static_cast<size_t>(a) * rank + b] = static_cast<float>(sum);
		}
		const float lambdaInv =
		    1.0f / std::max(geomScale * std::max(eigVal[a], 0.0f), eps);
		core[static_cast<size_t>(a) * rank + a] += lambdaInv;
	}

	std::vector<float> coreInv;
	bimap_invert_spd(core, rank, eps, coreInv);
	std::vector<float> proj(static_cast<size_t>(rank) * n, 0.0f);
	for (unsigned int a = 0u; a < rank; ++a)
	{
		for (unsigned int j = 0u; j < n; ++j)
		{
			double sum = 0.0;
			for (unsigned int i = 0u; i < m; ++i)
				sum += static_cast<double>(basis[static_cast<size_t>(i) * rank + a])
				     * static_cast<double>(matrix[static_cast<size_t>(i) * n + j]);
			proj[static_cast<size_t>(a) * n + j] = static_cast<float>(sum);
		}
	}
	std::vector<float> solved(static_cast<size_t>(rank) * n, 0.0f);
	for (unsigned int a = 0u; a < rank; ++a)
	{
		for (unsigned int j = 0u; j < n; ++j)
		{
			double sum = 0.0;
			for (unsigned int b = 0u; b < rank; ++b)
				sum += static_cast<double>(coreInv[static_cast<size_t>(a) * rank + b])
				     * static_cast<double>(proj[static_cast<size_t>(b) * n + j]);
			solved[static_cast<size_t>(a) * n + j] = static_cast<float>(sum);
		}
	}
	for (unsigned int i = 0u; i < m; ++i)
	{
		const float invDiag = 1.0f / std::max(diagMetric[i], eps);
		for (unsigned int j = 0u; j < n; ++j)
		{
			double correction = 0.0;
			for (unsigned int a = 0u; a < rank; ++a)
				correction += static_cast<double>(basis[static_cast<size_t>(i) * rank + a])
				            * static_cast<double>(solved[static_cast<size_t>(a) * n + j]);
			matrix[static_cast<size_t>(i) * n + j] -=
			    invDiag * static_cast<float>(correction);
		}
	}
}

static void bimap_apply_right_inverse(std::vector<float>& matrix,
                                      unsigned int m,
                                      unsigned int n,
                                      const std::vector<float>& diagMetric,
                                      const std::vector<float>& basis,
                                      const std::vector<float>& eigVal,
                                      unsigned int rank,
                                      float geomScale,
                                      float eps)
{
	if (matrix.size() != static_cast<size_t>(m) * n || diagMetric.size() != n)
		return;
	for (unsigned int i = 0u; i < m; ++i)
	{
		for (unsigned int j = 0u; j < n; ++j)
			matrix[static_cast<size_t>(i) * n + j] *=
			    1.0f / std::max(diagMetric[j], eps);
	}
	if (rank == 0u || geomScale <= 0.0f || basis.size() < static_cast<size_t>(n) * rank)
		return;

	std::vector<float> core(static_cast<size_t>(rank) * rank, 0.0f);
	for (unsigned int a = 0u; a < rank; ++a)
	{
		for (unsigned int b = 0u; b < rank; ++b)
		{
			double sum = 0.0;
			for (unsigned int j = 0u; j < n; ++j)
			{
				const double invDiag = 1.0 / std::max(static_cast<double>(diagMetric[j]),
				                                      static_cast<double>(eps));
				sum += invDiag
				     * static_cast<double>(basis[static_cast<size_t>(j) * rank + a])
				     * static_cast<double>(basis[static_cast<size_t>(j) * rank + b]);
			}
			core[static_cast<size_t>(a) * rank + b] = static_cast<float>(sum);
		}
		const float lambdaInv =
		    1.0f / std::max(geomScale * std::max(eigVal[a], 0.0f), eps);
		core[static_cast<size_t>(a) * rank + a] += lambdaInv;
	}

	std::vector<float> coreInv;
	bimap_invert_spd(core, rank, eps, coreInv);
	std::vector<float> proj(static_cast<size_t>(m) * rank, 0.0f);
	for (unsigned int i = 0u; i < m; ++i)
	{
		for (unsigned int a = 0u; a < rank; ++a)
		{
			double sum = 0.0;
			for (unsigned int j = 0u; j < n; ++j)
				sum += static_cast<double>(matrix[static_cast<size_t>(i) * n + j])
				     * static_cast<double>(basis[static_cast<size_t>(j) * rank + a]);
			proj[static_cast<size_t>(i) * rank + a] = static_cast<float>(sum);
		}
	}
	std::vector<float> solved(static_cast<size_t>(m) * rank, 0.0f);
	for (unsigned int i = 0u; i < m; ++i)
	{
		for (unsigned int a = 0u; a < rank; ++a)
		{
			double sum = 0.0;
			for (unsigned int b = 0u; b < rank; ++b)
				sum += static_cast<double>(proj[static_cast<size_t>(i) * rank + b])
				     * static_cast<double>(coreInv[static_cast<size_t>(b) * rank + a]);
			solved[static_cast<size_t>(i) * rank + a] = static_cast<float>(sum);
		}
	}
	for (unsigned int i = 0u; i < m; ++i)
	{
		for (unsigned int j = 0u; j < n; ++j)
		{
			double correction = 0.0;
			for (unsigned int a = 0u; a < rank; ++a)
				correction += static_cast<double>(solved[static_cast<size_t>(i) * rank + a])
				            * static_cast<double>(basis[static_cast<size_t>(j) * rank + a]);
			matrix[static_cast<size_t>(i) * n + j] -=
			    static_cast<float>(correction)
			    * (1.0f / std::max(diagMetric[j], eps));
		}
	}
}

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
                 shmea::GLogger* logger,
                 const char* tag)
{
	if (!W || !m1 || !v2 || !gW || m == 0u || n == 0u)
		return true;
	if (!state.initialized || state.m != m || state.n != n)
		initBiMAPWeightState(state, m, n);
	if (!state.initialized || state.m != m || state.n != n)
		return false;

	const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
	const float oneMinusB1 = 1.0f - beta1;
	const float oneMinusB2 = 1.0f - beta2;
	const float betaGeom = std::min<float>(std::max<float>(ac.beta, 0.0f), 1.0f);
	const float geomScale = std::max(0.0f, ac.bimapGeometryScale);
	const bool useLowRank = ac.bimapLowRankEnabled && geomScale > 0.0f;
	const float predictiveScale =
	    std::max(0.0f, std::min(1.0f, ac.bimapPredictiveScale));
	const unsigned int cadence = std::max(1u, ac.bimapFactorCadence);
	const unsigned int rankCap =
	    useLowRank ? std::min(ac.rank, std::min(m, n)) : 0u;
	const unsigned int powerIters = std::max(1u, std::min(ac.powerIters, 2u));
	const bool refreshLowRank = useLowRank && ((state.step % cadence) == 0ULL);
	std::vector<float> gradSnapshot;
	if (refreshLowRank)
		gradSnapshot.assign(mn, 0.0f);

	std::fill(state.scratchRow.begin(), state.scratchRow.end(), 0.0f);
	std::fill(state.scratchCol.begin(), state.scratchCol.end(), 0.0f);

	for (unsigned int i = 0u; i < m; ++i)
	{
		double rowSq = 0.0;
		for (unsigned int j = 0u; j < n; ++j)
		{
			const size_t idx = static_cast<size_t>(i) * n + j;
			const float gScaledRaw = gW[idx] * invBatch * gradScale;
			if (refreshLowRank)
				gradSnapshot[idx] = gScaledRaw;
			float g = gScaledRaw;
			if (wd1 != 0.0f)
				g += wd1 * atlas_sign(W[idx]) * gradScale;
			m1[idx] = beta1 * m1[idx] + oneMinusB1 * g;
			v2[idx] = beta2 * v2[idx] + oneMinusB2 * (g * g);
			const double g2 = static_cast<double>(gScaledRaw) * static_cast<double>(gScaledRaw);
			rowSq += g2;
			state.scratchCol[j] += static_cast<float>(g2);
		}
		state.scratchRow[i] = static_cast<float>(rowSq / static_cast<double>(std::max(1u, n)));
	}
	for (unsigned int j = 0u; j < n; ++j)
		state.scratchCol[j] /= static_cast<float>(std::max(1u, m));

	if ((state.step % cadence) == 0ULL)
	{
		for (unsigned int i = 0u; i < m; ++i)
			state.rowSecond[i] =
			    betaGeom * state.rowSecond[i]
			    + (1.0f - betaGeom) * std::max(state.scratchRow[i], 1.0e-12f);
		for (unsigned int j = 0u; j < n; ++j)
			state.colSecond[j] =
			    betaGeom * state.colSecond[j]
			    + (1.0f - betaGeom) * std::max(state.scratchCol[j], 1.0e-12f);
	}

	double rowMean = 0.0;
	double colMean = 0.0;
	float rowMin = FLT_MAX;
	float rowMax = 0.0f;
	float colMin = FLT_MAX;
	float colMax = 0.0f;
	for (unsigned int i = 0u; i < m; ++i)
	{
		const float v = std::max(state.rowSecond[i], 1.0e-12f);
		rowMean += static_cast<double>(v);
		rowMin = std::min(rowMin, v);
		rowMax = std::max(rowMax, v);
	}
	for (unsigned int j = 0u; j < n; ++j)
	{
		const float v = std::max(state.colSecond[j], 1.0e-12f);
		colMean += static_cast<double>(v);
		colMin = std::min(colMin, v);
		colMax = std::max(colMax, v);
	}
	rowMean /= static_cast<double>(std::max(1u, m));
	colMean /= static_cast<double>(std::max(1u, n));
	const float rowMeanF = static_cast<float>(std::max(rowMean, 1.0e-12));
	const float colMeanF = static_cast<float>(std::max(colMean, 1.0e-12));
	if (refreshLowRank)
	{
		bimap_refresh_low_rank_factors(state,
		                               gradSnapshot,
		                               m,
		                               n,
		                               rankCap,
		                               powerIters,
		                               betaGeom,
		                               rowMeanF,
		                               colMeanF);
	}
	else if (!useLowRank)
	{
		state.rowRank = 0u;
		state.colRank = 0u;
		state.lastRowCapture = 0.0f;
		state.lastColCapture = 0.0f;
	}

	double dot = 0.0;
	double curNorm = 0.0;
	double prevNorm = 0.0;
	if (predictiveScale > 0.0f && state.step > 0ULL && state.prevMhat.size() == mn)
	{
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			const double cur = static_cast<double>(m1[idx] * inv1mB1t);
			const double prev = static_cast<double>(state.prevMhat[idx]);
			dot += cur * prev;
			curNorm += cur * cur;
			prevNorm += prev * prev;
		}
	}
	float predictiveTrust = 0.0f;
	if (curNorm > 1.0e-18 && prevNorm > 1.0e-18)
	{
		const double cosine = dot / (std::sqrt(curNorm * prevNorm) + 1.0e-18);
		predictiveTrust =
		    predictiveScale * std::max(0.0f, std::min(1.0f, static_cast<float>(cosine)));
	}

	if (!useLowRank)
	{
		for (unsigned int i = 0u; i < m; ++i)
		{
			const float rowScaleRaw =
			    std::sqrt((std::max(state.rowSecond[i], 1.0e-12f) + eps) / (rowMeanF + eps));
			for (unsigned int j = 0u; j < n; ++j)
			{
				const size_t idx = static_cast<size_t>(i) * n + j;
				const float colScaleRaw =
				    std::sqrt((std::max(state.colSecond[j], 1.0e-12f) + eps) / (colMeanF + eps));
				float matrixScale = rowScaleRaw * colScaleRaw;
				if (matrixScale < 0.25f)
					matrixScale = 0.25f;
				else if (matrixScale > 4.0f)
					matrixScale = 4.0f;
				const float mixedScale = std::max(0.25f, 1.0f + geomScale * (matrixScale - 1.0f));

				const float vhat = v2[idx] * inv1mB2t;
				const float diagDen =
				    static_cast<float>(std::sqrt(static_cast<double>(std::max(vhat, 0.0f)))) + eps;
				const float currentMhat = m1[idx] * inv1mB1t;
				float effectiveMhat = currentMhat;
				if (predictiveTrust > 0.0f && state.prevMhat.size() == mn)
				{
					float delta = currentMhat - state.prevMhat[idx];
					const float deltaCap = 0.5f * (fabsf(currentMhat) + eps);
					if (delta > deltaCap)
						delta = deltaCap;
					else if (delta < -deltaCap)
						delta = -deltaCap;
					effectiveMhat += predictiveTrust * delta;
				}

				if (wd2 != 0.0f)
					W[idx] -= lr * wd2 * W[idx];
				W[idx] -= lr * (effectiveMhat / (diagDen * mixedScale));
				gW[idx] = 0.0f;
				state.prevMhat[idx] = currentMhat;
				if (!atlas_isfinite(W[idx]))
					return false;
			}
		}
	}
	else
	{
		std::vector<float> rowMetric(static_cast<size_t>(m), 1.0f);
		std::vector<float> colMetric(static_cast<size_t>(n), 1.0f);
		std::vector<float> stepMatrix(mn, 0.0f);
		for (unsigned int i = 0u; i < m; ++i)
		{
			const float rowScaleRaw =
			    std::sqrt((std::max(state.rowSecond[i], 1.0e-12f) + eps) / (rowMeanF + eps));
			rowMetric[i] = std::max(0.25f, std::min(4.0f, 1.0f + geomScale * (rowScaleRaw - 1.0f)));
		}
		for (unsigned int j = 0u; j < n; ++j)
		{
			const float colScaleRaw =
			    std::sqrt((std::max(state.colSecond[j], 1.0e-12f) + eps) / (colMeanF + eps));
			colMetric[j] = std::max(0.25f, std::min(4.0f, 1.0f + geomScale * (colScaleRaw - 1.0f)));
		}
		for (unsigned int i = 0u; i < m; ++i)
		{
			for (unsigned int j = 0u; j < n; ++j)
			{
				const size_t idx = static_cast<size_t>(i) * n + j;
				const float vhat = v2[idx] * inv1mB2t;
				const float diagDen =
				    static_cast<float>(std::sqrt(static_cast<double>(std::max(vhat, 0.0f)))) + eps;
				const float currentMhat = m1[idx] * inv1mB1t;
				float effectiveMhat = currentMhat;
				if (predictiveTrust > 0.0f && state.prevMhat.size() == mn)
				{
					float delta = currentMhat - state.prevMhat[idx];
					const float deltaCap = 0.5f * (fabsf(currentMhat) + eps);
					if (delta > deltaCap)
						delta = deltaCap;
					else if (delta < -deltaCap)
						delta = -deltaCap;
					effectiveMhat += predictiveTrust * delta;
				}
				stepMatrix[idx] = effectiveMhat / diagDen;
				state.prevMhat[idx] = currentMhat;
			}
		}
		bimap_apply_left_inverse(stepMatrix,
		                         m,
		                         n,
		                         rowMetric,
		                         state.rowBasis,
		                         state.rowEigVal,
		                         state.rowRank,
		                         geomScale,
		                         eps);
		bimap_apply_right_inverse(stepMatrix,
		                          m,
		                          n,
		                          colMetric,
		                          state.colBasis,
		                          state.colEigVal,
		                          state.colRank,
		                          geomScale,
		                          eps);
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			if (wd2 != 0.0f)
				W[idx] -= lr * wd2 * W[idx];
			W[idx] -= lr * stepMatrix[idx];
			gW[idx] = 0.0f;
			if (!atlas_isfinite(W[idx]))
				return false;
		}
	}

	state.lastPredictiveTrust = predictiveTrust;
	state.lastRowAnisotropy = (rowMin > 1.0e-12f) ? (rowMax / rowMin) : 1.0f;
	state.lastColAnisotropy = (colMin > 1.0e-12f) ? (colMax / colMin) : 1.0f;
	state.step += 1ULL;

	if (logger && ac.tSub > 0u
	    && ((state.step % static_cast<unsigned long long>(std::max(1u, ac.tSub))) == 0ULL))
	{
		std::ostringstream oss;
		oss << "event=bimap_step";
		if (tag && tag[0])
			append_kv(oss, "tag", tag);
		append_kv(oss, "step", state.step);
		append_kv(oss, "predTrust", state.lastPredictiveTrust);
		append_kv(oss, "rowAniso", state.lastRowAnisotropy);
		append_kv(oss, "colAniso", state.lastColAnisotropy);
		append_kv(oss, "rowRank", state.rowRank);
		append_kv(oss, "colRank", state.colRank);
		append_kv(oss, "rowCapture", state.lastRowCapture);
		append_kv(oss, "colCapture", state.lastColCapture);
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}

	return true;
}

bool updateBias(float* bias, float* gBias, unsigned int size,
                float invBatch, float lr, float gradScale)
{
	for (unsigned int i = 0; i < size; ++i)
	{
		float gB = gBias[i] * invBatch;
		gB *= gradScale;
		bias[i] -= lr * gB;
		gBias[i] = 0.0f;
		if (!atlas_isfinite(bias[i]))
			return false;
	}
	return true;
}

} // namespace atlas
} // namespace glades
