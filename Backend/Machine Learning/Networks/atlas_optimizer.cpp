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

	// Initialize Fisher diagonal to 1.0 (neutral preconditioning)
	state.fisherDiag.assign(static_cast<size_t>(r), 1.0f);

	// Initialize previous compressed gradient to zero
	state.prevGz.assign(static_cast<size_t>(r) * static_cast<size_t>(n), 0.0f);

	// Allocate persistent scratch buffers (reused every step, avoids per-step heap churn).
	const size_t mr = static_cast<size_t>(m) * static_cast<size_t>(r);
	const size_t rn = static_cast<size_t>(r) * static_cast<size_t>(n);
	state.scratch_gz.resize(rn);
	state.scratch_corrected.resize(rn);
	state.scratch_U_old.resize(mr);
	state.scratch_f_old.resize(static_cast<size_t>(r));
	state.scratch_B.resize(rn);
	state.scratch_Z.resize(mr);
	state.scratch_overlap.resize(static_cast<size_t>(r) * static_cast<size_t>(r));
	state.scratch_prevGzOld.resize(rn);
	state.scratch_basisPacked.resize(mr);

	state.sigma2 = 1.0f;
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
		append_kv(oss, "mu_init", muInit);
		append_kv(oss, "U_size", static_cast<unsigned long long>(state.U.size()));
		append_kv(oss, "prevGz_size", static_cast<unsigned long long>(state.prevGz.size()));
		const unsigned long long totalBytes =
		    static_cast<unsigned long long>(state.U.size() + state.fisherDiag.size()
		        + state.prevGz.size()
		        + state.scratch_gz.size() + state.scratch_corrected.size()
		        + state.scratch_U_old.size() + state.scratch_f_old.size()
		        + state.scratch_B.size() + state.scratch_Z.size()
		        + state.scratch_overlap.size() + state.scratch_prevGzOld.size()
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
	size_t rn = static_cast<size_t>(activeRank) * static_cast<size_t>(n);

	const bool diagStep = logger && tSub > 0u
		&& (state.step % static_cast<unsigned long long>(tSub)) == 0ULL;

	const float gScale = invBatch * gradScale;

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

	// === Step 1: Update global second moment sigma2 ===
	// sigma2 tracks mean(G_accum^2) on the accumulated-gradient scale.
	// Using gScale here would inject an invBatch^2 factor, driving sigma2
	// toward zero for large batches and collapsing baseline-rate adaptation.
	// The invBatch normalization is applied later in the actual update via
	// baseScaled = baselineRate * gScale.
	{
		double gMeanSq = 0.0;
		for (size_t idx = 0; idx < mn; ++idx)
		{
			const double v = static_cast<double>(gW[idx]) * static_cast<double>(gradScale);
			gMeanSq += v * v;
		}
		gMeanSq /= static_cast<double>(mn);
		if (state.step == 1ULL)
		{
			// Initialize from the actual first-step gradient statistics rather than
			// decaying from the arbitrary reset value.
			state.sigma2 = (gMeanSq > static_cast<double>(eps))
			             ? static_cast<float>(gMeanSq)
			             : eps;
		}
		else
		{
			state.sigma2 = beta * state.sigma2 + (1.0f - beta) * static_cast<float>(gMeanSq);
		}
		if (state.sigma2 < eps)
			state.sigma2 = eps;
		if (!atlas_isfinite(state.sigma2))
		{
			state.sigma2 = eps;
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

	// === Step 5: Update Fisher diagonal (EMA of mean squared projected gradient) ===
	const float oneMinusBeta = 1.0f - beta;
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		double sumsq = 0.0;
		for (unsigned int j = 0; j < n; ++j)
		{
			const double v = static_cast<double>(gz[c * n + j]);
			sumsq += v * v;
		}
		const float meansq = static_cast<float>(sumsq / static_cast<double>(n));
		state.fisherDiag[c] = beta * state.fisherDiag[c] + oneMinusBeta * meansq;
		if (!atlas_isfinite(state.fisherDiag[c]))
		{
			state.fisherDiag[c] = 1.0f;
			recovered = true;
		}
	}

	// === Step 6: Full-space baseline update ===
	// W -= min(lr / (effSigma2 + eps), kappaMax * lr) * G
	// Baseline rate is capped at kappaMax*lr to prevent divergence
	// when sigma2 converges to small gradient variance.
	// effSigma2 includes bias correction so early steps get meaningful preconditioning.
	const float kappaLr = kappaMax * lr;
	const float effSigma2 = state.sigma2 * bcFactor;
	float rawBaselineRate = lr / (effSigma2 + eps);
	if (rawBaselineRate > kappaLr) rawBaselineRate = kappaLr;
	const float baselineRate = rawBaselineRate;
	state.lastBaselineRate = baselineRate;
	{
		const float baseScaled = -baselineRate * gScale;
		for (size_t idx = 0; idx < mn; ++idx)
			W[idx] += baseScaled * gW[idx];
	}

	// === Step 7: Subspace correction with optional PNG ===
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

	// Precompute corrected[r,n] = corrScale[c] * gPred[c,n]
	std::vector<float>& corrected = state.scratch_corrected;
	for (unsigned int c = 0; c < activeRank; ++c)
	{
		const float effFisher = state.fisherDiag[c] * bcFactor;
		float fisherLR = lr / (effFisher + eps);
		if (fisherLR > kappaLr) fisherLR = kappaLr;
		const float corrScale = baselineRate - fisherLR;
		for (unsigned int j = 0; j < n; ++j)
		{
			const size_t cj = static_cast<size_t>(c) * n + j;
			corrected[cj] = corrScale * (onePlusMu * gz[cj] + negMu * state.prevGz[cj]);
		}
	}

	// W[m,n] += U[m,r] * corrected[r,n]
	glades::gemm::ab_accum(W, &basisPacked[0], &corrected[0], m, activeRank, n, 1.0f);

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
	}

	// === Step 8: Adapt prediction coefficient ===
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
		const float sigma2FisherRatio = (fSum > 1e-30)
		    ? static_cast<float>(state.sigma2 / (fSum / static_cast<double>(activeRank)))
		    : 0.0f;

		std::ostringstream oss;
		oss << "event=atlas_step";
		if (tag) oss << " tag=" << tag;
		append_kv(oss, "step", state.step);
		append_kv(oss, "m", m);
		append_kv(oss, "n", n);
		append_kv(oss, "rank", r);
		append_kv(oss, "active_rank", activeRank);
		append_kv(oss, "lr", lr);
		append_kv(oss, "mu", state.mu);
		append_kv(oss, "sigma2", state.sigma2);
		append_kv(oss, "baseline_rate", baselineRate);
		append_kv(oss, "gz_norm", static_cast<float>(sqrt(gzNormSq)));
		append_kv(oss, "update_norm", static_cast<float>(sqrt(updateNormSq)));
		append_kv(oss, "fisher_min", fMin);
		append_kv(oss, "fisher_max", fMax);
		append_kv(oss, "fisher_mean", static_cast<float>(fSum / static_cast<double>(activeRank)));
		append_kv(oss, "fisher_ratio", fisherRatio);
		append_kv(oss, "sigma2_fisher_ratio", sigma2FisherRatio);
		append_kv(oss, "effective_rank", effectiveRank);
		append_kv(oss, "spectral_efficiency", spectralEfficiency);
		append_kv(oss, "top1_concentration", top1Concentration);
		append_kv(oss, "top10_concentration", top10Concentration);
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}

	// === Step 9: Store compressed gradient for next step ===
	if (rn > 0u)
		std::copy(gz.begin(), gz.begin() + rn, state.prevGz.begin());
	if (state.prevGz.size() > rn)
		std::fill(state.prevGz.begin() + rn, state.prevGz.end(), 0.0f);

	// === Step 10: Clear accumulated gradients ===
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
