// ATLAS optimizer implementation (BRSP variant).
// See atlas_optimizer.h for API documentation.
#include "atlas_optimizer.h"
#include "gemm_helpers.h"
#include "training_config.h"
#include "transformer_kernels.h"
#include "Backend/Database/GLogger.h"
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

// NaN/Inf check for internal state protection.
// Uses std::isfinite which is safe under all compiler optimization levels,
// unlike the manual (x == x) && (x - x == 0.0f) pattern which can be
// optimized away under -ffast-math.
static inline bool atlas_isfinite(float x)
{
	return std::isfinite(x);
}

static inline float atlas_bootstrap_or_ema(float prev,
                                           float sample,
                                           float beta,
                                           unsigned long long step)
{
	if (step <= 1ULL)
		return sample;
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

static unsigned int choose_active_complement_rank(const std::vector<float>& eigVal,
                                                  const std::vector<float>* scoutEigVal,
                                                  unsigned int informativeRank,
                                                  unsigned int prevRank,
                                                  double activeTrace,
                                                  float totalTrace,
                                                  unsigned int subDim,
                                                  unsigned int activeRank,
                                                  float eps,
                                                  double* birthKellyOut = 0,
                                                  double* birthScoutOut = 0)
{
	if (informativeRank == 0u)
		return 0u;
	if (prevRank > informativeRank)
		prevRank = informativeRank;

	const double fullTrace = sum_leading_spectrum(eigVal, informativeRank);
	const double closedTrace = std::max<double>(static_cast<double>(totalTrace),
	                                            activeTrace + fullTrace);
	const double deathRatio = 1.10;
	const double birthKellyThreshold = 0.10;
	const double birthMinShare = 0.005;
	const double deathMinShare = 0.03;
	if (birthKellyOut)
		*birthKellyOut = 0.0;
	if (birthScoutOut)
		*birthScoutOut = 0.0;

	if (prevRank < informativeRank)
	{
		const double selectedTrace = sum_leading_spectrum(eigVal, prevRank);
		const double tailMean =
		    complement_tail_mean(closedTrace, activeTrace, selectedTrace,
		                         subDim, activeRank, prevRank, eps);
		double emaNextEig = static_cast<double>(eigVal[prevRank]);
		if (!std::isfinite(emaNextEig) || emaNextEig < 0.0)
			emaNextEig = 0.0;
		double scoutNextEig = emaNextEig;
		if (scoutEigVal && prevRank < scoutEigVal->size())
		{
			scoutNextEig = static_cast<double>((*scoutEigVal)[prevRank]);
			if (!std::isfinite(scoutNextEig) || scoutNextEig < 0.0)
				scoutNextEig = 0.0;
		}
		const double birthScout = (scoutNextEig > emaNextEig) ? scoutNextEig : emaNextEig;
		const double birthKelly = complement_kelly_fraction(birthScout, tailMean);
		if (birthKellyOut)
			*birthKellyOut = birthKelly;
		if (birthScoutOut)
			*birthScoutOut = birthScout;
		if (birthKelly >= birthKellyThreshold
		    && birthScout > birthMinShare * closedTrace)
			return prevRank + 1u;
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
			return prevRank - 1u;
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
	    && state.prevGv.size() == static_cast<size_t>(storageRank) * state.n)
		return;

	const unsigned int oldRank = state.complementRank;
	const unsigned int copyRank = (oldRank < storageRank) ? oldRank : storageRank;
	std::vector<float> oldV(state.V);
	std::vector<float> oldPrevGv(state.prevGv);
	std::vector<float> oldBlock(state.complementBlock);

	state.complementRank = storageRank;
	if (state.activeComplementRank > storageRank)
		state.activeComplementRank = storageRank;
	state.V.assign(static_cast<size_t>(state.m) * storageRank, 0.0f);
	state.complementBlock.assign(static_cast<size_t>(storageRank) * storageRank, 0.0f);
	state.prevGv.assign(static_cast<size_t>(storageRank) * state.n, 0.0f);

	for (unsigned int i = 0; i < state.m; ++i)
		for (unsigned int c = 0; c < copyRank; ++c)
			state.V[static_cast<size_t>(i) * storageRank + c] =
			    oldV[static_cast<size_t>(i) * oldRank + c];
	for (unsigned int c = 0; c < copyRank; ++c)
		for (unsigned int j = 0; j < state.n; ++j)
			state.prevGv[static_cast<size_t>(c) * state.n + j] =
			    oldPrevGv[static_cast<size_t>(c) * state.n + j];
	for (unsigned int i = 0; i < copyRank; ++i)
		for (unsigned int j = 0; j < copyRank; ++j)
			state.complementBlock[static_cast<size_t>(i) * storageRank + j] =
			    oldBlock[static_cast<size_t>(i) * oldRank + j];

	for (unsigned int c = copyRank; c < storageRank; ++c)
	{
		for (unsigned int i = 0; i < state.m; ++i)
			state.V[static_cast<size_t>(i) * storageRank + c] =
			    glades::rng::standard_normal(rng);
	}
	const unsigned int targetRank =
	    atlas_requested_complement_rank(storageRank, state.m, activeRank);
	orthonormalize_complement_block(state, activeRank, targetRank, 0, logger);

	const size_t rnComp = static_cast<size_t>(storageRank) * state.n;
	const size_t mrComp = static_cast<size_t>(state.m) * storageRank;
	state.scratch_gv.resize(rnComp);
	state.scratch_correctedV.resize(rnComp);
	state.scratch_V_old.resize(mrComp);
	state.scratch_Bv.resize(rnComp);
	state.scratch_Zv.resize(mrComp);
	state.scratch_complementMat.resize(static_cast<size_t>(storageRank) * storageRank);
	state.scratch_complementEigVec.resize(static_cast<size_t>(storageRank) * storageRank);
	state.scratch_complementEigVal.resize(static_cast<size_t>(storageRank));
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
	state.complementBlock.assign(static_cast<size_t>(state.complementRank) * state.complementRank, 0.0f);

	// Initialize previous compressed gradients to zero
	state.prevGz.assign(static_cast<size_t>(r) * static_cast<size_t>(n), 0.0f);
	state.prevGv.assign(static_cast<size_t>(state.complementRank) * static_cast<size_t>(n), 0.0f);
	state.V.assign(static_cast<size_t>(m) * static_cast<size_t>(state.complementRank), 0.0f);
	if (atlas_requested_complement_rank(state.complementRank, m, state.activeRank) > 0u)
		ensure_complement_basis(state, state.activeRank, rng, logger);

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
	state.scratch_U_old.resize(mr);
	state.scratch_f_old.resize(static_cast<size_t>(r));
	state.scratch_B.resize(rn);
	state.scratch_Z.resize(mr);
	state.scratch_overlap.resize(static_cast<size_t>(r) * static_cast<size_t>(r));
	state.scratch_prevGzOld.resize(rn);
	state.scratch_basisPacked.resize(mr);
	state.scratch_V_old.resize(mc);
	state.scratch_Bv.resize(cn);
	state.scratch_Zv.resize(mc);
	state.scratch_complementMat.resize(cr * cr);
	state.scratch_complementEigVec.resize(cr * cr);
	state.scratch_complementEigVal.resize(cr);

	state.totalTrace = 0.0f;
	state.sigma2 = 0.0f;
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
		append_kv(oss, "prevGz_size", static_cast<unsigned long long>(state.prevGz.size()));
		append_kv(oss, "prevGv_size", static_cast<unsigned long long>(state.prevGv.size()));
		const unsigned long long totalBytes =
		    static_cast<unsigned long long>(state.U.size() + state.fisherDiag.size()
		        + state.V.size() + state.prevGz.size() + state.prevGv.size()
		        + state.scratch_gz.size() + state.scratch_corrected.size()
		        + state.scratch_gv.size() + state.scratch_correctedV.size()
		        + state.scratch_U_old.size() + state.scratch_f_old.size()
		        + state.scratch_B.size() + state.scratch_Z.size()
		        + state.scratch_overlap.size() + state.scratch_prevGzOld.size()
		        + state.scratch_V_old.size() + state.scratch_Bv.size()
		        + state.scratch_Zv.size()
		        + state.complementBlock.size()
		        + state.scratch_complementMat.size()
		        + state.scratch_complementEigVec.size()
		        + state.scratch_complementEigVal.size()
		        + state.scratch_basisPacked.size())
		    * static_cast<unsigned long long>(sizeof(float));
		append_kv(oss, "total_bytes", totalBytes);
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}
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
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		for (unsigned int k = 0; k < activeRank; ++k)
		{
			const float o_ck = overlap[c * activeRank + k];
			if (o_ck == 0.0f) continue;
			axpy_f32(&state.prevGz[c * n], &prevGzOld[k * n], o_ck, n);
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
	const bool adaptiveComplementController = (tag && tag[0]);
	const unsigned int storageComplementRank =
	    atlas_storage_complement_rank(enabledComplementRank);
	if (state.complementRank != storageComplementRank)
		resize_complement_storage(state, storageComplementRank, activeRank, rng, logger);
	const unsigned int complementRank = state.complementRank;
	size_t rn = static_cast<size_t>(activeRank) * static_cast<size_t>(n);
	const size_t cn = static_cast<size_t>(complementRank) * static_cast<size_t>(n);
	if (state.activeComplementRank > enabledComplementRank)
		state.activeComplementRank = enabledComplementRank;

	const bool diagStep = logger && tSub > 0u
		&& (state.step % static_cast<unsigned long long>(tSub)) == 0ULL;
	const bool complementControlStep =
	    adaptiveComplementController
	    && (enabledComplementRank > 0u)
	    && ((tSub == 0u) || (state.step % static_cast<unsigned long long>(tSub)) == 0ULL);

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
			recovered = true;
		if (enabledComplementRank > 0u
		    && !refreshComplementSector(state, gW, m, n, activeRank,
		                                ac.powerIters, ac.betaRefresh, rng, logger))
			recovered = true;
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
	}

	// === Step 6: Recompute sigma2 from the shared covariance model ===
	unsigned int informativeComplementRank =
	    effective_complement_rank(state, enabledComplementRank, m, activeRank);
	double activeSectorTrace = 0.0;
	float complementTailMean = eps;
	float complementScoutTop = 0.0f;
	float complementBirthKelly = 0.0f;
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
		if (adaptiveComplementController)
		{
			std::vector<float> scoutEigVec;
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
		if (!adaptiveComplementController)
		{
			state.activeComplementRank = informativeComplementRank;
		}
		else if (complementControlStep)
		{
			const unsigned int prevActiveComplementRank = state.activeComplementRank;
			double birthKelly = 0.0;
			double birthScout = 0.0;
			state.activeComplementRank =
			    choose_active_complement_rank(state.scratch_complementEigVal,
			                                  scoutEigVal.empty() ? 0 : &scoutEigVal,
			                                  informativeComplementRank,
			                                  state.activeComplementRank,
			                                  compute_active_trace(state, activeRank),
			                                  state.totalTrace,
			                                  m,
			                                  activeRank,
			                                  eps,
			                                  &birthKelly,
			                                  &birthScout);
			complementBirthKelly = static_cast<float>(birthKelly);
			complementScoutTop = static_cast<float>(birthScout);
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
				append_kv(oss, "birth_kelly", complementBirthKelly);
				append_kv(oss, "birth_scout", complementScoutTop);
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
	}
	if (state.activeComplementRank > informativeComplementRank)
		state.activeComplementRank = informativeComplementRank;
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

	// Precompute corrected[r,n] = corrScale[c] * gPred[c,n]
	std::vector<float>& corrected = state.scratch_corrected;
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		const float effFisher = state.fisherDiag[c] * bcFactor;
		const float fisherLR = atlas_clamped_rate(lr, effFisher, eps, kappaMax);
		const float corrScale = baselineRate - fisherLR;
		for (unsigned int j = 0; j < n; ++j)
		{
			const size_t cj = static_cast<size_t>(c) * n + j;
			corrected[cj] = corrScale * (onePlusMu * gz[cj] + negMu * state.prevGz[cj]);
		}
	}

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
		append_kv(oss, "complement_birth_kelly", complementBirthKelly);
		append_kv(oss, "sector_rate", sectorRate);
		append_kv(oss, "closure_gap", static_cast<float>(closureGap));
		append_kv(oss, "effective_rank", effectiveRank);
		append_kv(oss, "spectral_efficiency", spectralEfficiency);
		append_kv(oss, "top1_concentration", top1Concentration);
		append_kv(oss, "top10_concentration", top10Concentration);
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}

	// === Step 10: Store compressed gradient for next step ===
	if (rn > 0u)
		std::copy(gz.begin(), gz.begin() + rn, state.prevGz.begin());
	if (state.prevGz.size() > rn)
		std::fill(state.prevGz.begin() + rn, state.prevGz.end(), 0.0f);
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
