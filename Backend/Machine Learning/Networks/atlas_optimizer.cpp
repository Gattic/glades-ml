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

	state.sigma2 = 1.0f;
	state.mu = muInit;
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
		append_kv(oss, "mu_init", muInit);
		append_kv(oss, "U_size", static_cast<unsigned long long>(state.U.size()));
		append_kv(oss, "prevGz_size", static_cast<unsigned long long>(state.prevGz.size()));
		const unsigned long long totalBytes =
		    static_cast<unsigned long long>(state.U.size() + state.fisherDiag.size()
		        + state.prevGz.size()
		        + state.scratch_gz.size() + state.scratch_corrected.size()
		        + state.scratch_U_old.size() + state.scratch_f_old.size()
		        + state.scratch_B.size() + state.scratch_Z.size()
		        + state.scratch_overlap.size() + state.scratch_prevGzOld.size())
		    * static_cast<unsigned long long>(sizeof(float));
		append_kv(oss, "total_bytes", totalBytes);
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}
}

bool refreshSubspace(WeightState& state, const float* grad,
                     unsigned int m, unsigned int n,
                     unsigned int powerIters, float betaRefresh,
                     glades::rng::Engine& rng,
                     shmea::GLogger* logger)
{
	const unsigned int r = state.r;
	if (r == 0u || m == 0u || n == 0u) return true;

	const size_t mr = static_cast<size_t>(m) * static_cast<size_t>(r);
	const size_t rn = static_cast<size_t>(r) * static_cast<size_t>(n);

	// Save old basis and Fisher for EMA blending and Fisher transform.
	std::copy(state.U.begin(), state.U.end(), state.scratch_U_old.begin());
	std::copy(state.fisherDiag.begin(), state.fisherDiag.end(), state.scratch_f_old.begin());
	std::vector<float>& U_old = state.scratch_U_old;
	std::vector<float>& f_old = state.scratch_f_old;

	// --- Randomized power iteration (warm-started from current U) ---
	std::vector<float>& Q = state.U;
	// B stored as [r, n] (transposed from original [n, r]) for SIMD-friendly access.
	std::vector<float>& B = state.scratch_B;
	std::vector<float>& Z = state.scratch_Z;

	for (unsigned int p = 0; p < powerIters; ++p)
	{
		// B[r,n] = Q^T[r,m] * grad[m,n]  (Q is [m,r], so Q^T is [r,m])
		glades::gemm::atb(&B[0], &Q[0], grad, r, m, n, 1.0f);

		// Z[m,r] = grad[m,n] * B[r,n]^T
		// B is [r,n] row-major; B^T is [n,r]; Z[i,c] = dot(grad[i,:], B[c,:])
		glades::gemm::abt(&Z[0], grad, &B[0], m, n, r, 1.0f);

		std::copy(Z.begin(), Z.end(), Q.begin());
		gramSchmidt(&Q[0], m, r, logger);
	}

	// Q now contains U_new (raw power iteration result).

	// --- EMA blend: U = (1-betaRefresh)*U_old + betaRefresh*U_new ---
	for (size_t idx = 0; idx < mr; ++idx)
		Q[idx] = (1.0f - betaRefresh) * U_old[idx] + betaRefresh * Q[idx];
	gramSchmidt(&Q[0], m, r, logger);

	// --- Compute overlap matrix O = U_final^T * U_old [r x r] ---
	// O[c*r+j] = sum_i U_final[i*r+c] * U_old[i*r+j]
	// This is O[r,r] = Q^T[r,m] * U_old[m,r]
	std::vector<float>& overlap = state.scratch_overlap;
	glades::gemm::atb(&overlap[0], &Q[0], &U_old[0], r, m, r, 1.0f);

	// --- Transform Fisher diagonal into new basis ---
	// f_new[c] = sum_j O[c,j]^2 * f_old[j]
	// This transfers curvature information from old directions to new directions
	// based on their overlap, preserving accumulated preconditioning knowledge.
	for (unsigned int c = 0; c < r; ++c)
	{
		double fNew = 0.0;
		for (unsigned int j = 0; j < r; ++j)
		{
			const double o = static_cast<double>(overlap[c * r + j]);
			fNew += o * o * static_cast<double>(f_old[j]);
		}
		if (fNew < 1e-12) fNew = 1e-12;
		state.fisherDiag[c] = static_cast<float>(fNew);
	}

	// --- Transform prevGz into new basis ---
	// prevGz_new[c*n+j] = sum_k O[c,k] * prevGz_old[k*n+j]
	std::vector<float>& prevGzOld = state.scratch_prevGzOld;
	std::copy(state.prevGz.begin(), state.prevGz.end(), prevGzOld.begin());
	std::fill(state.prevGz.begin(), state.prevGz.end(), 0.0f);
	for (unsigned int c = 0; c < r; ++c)
	{
		for (unsigned int k = 0; k < r; ++k)
		{
			const float o_ck = overlap[c * r + k];
			if (o_ck == 0.0f) continue;
			axpy_f32(&state.prevGz[c * n], &prevGzOld[k * n], o_ck, n);
		}
	}

	// Verify U and Fisher are finite after refresh.
	bool refreshOk = true;
	for (size_t idx = 0; idx < mr; ++idx)
	{
		if (!atlas_isfinite(Q[idx]))
		{
			refreshOk = false;
			break;
		}
	}
	if (refreshOk)
	{
		for (unsigned int c = 0; c < r; ++c)
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
		logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
	}

	if (logger && refreshOk)
	{
		float fMin = state.fisherDiag[0];
		float fMax = state.fisherDiag[0];
		double fSum = 0.0;
		for (unsigned int c = 0; c < r; ++c)
		{
			const float f = state.fisherDiag[c];
			if (f < fMin) fMin = f;
			if (f > fMax) fMax = f;
			fSum += static_cast<double>(f);
		}
		// Mean diagonal overlap measures basis stability (1.0 = no change)
		double overlapDiagSum = 0.0;
		for (unsigned int c = 0; c < r; ++c)
		{
			const double od = static_cast<double>(overlap[c * r + c]);
			overlapDiagSum += (od > 0.0 ? od : -od);
		}

		std::ostringstream oss;
		oss << "event=atlas_subspace_refresh";
		append_kv(oss, "step", state.step);
		append_kv(oss, "m", m);
		append_kv(oss, "n", n);
		append_kv(oss, "rank", r);
		append_kv(oss, "power_iters", powerIters);
		append_kv(oss, "beta_refresh", betaRefresh);
		append_kv(oss, "mean_overlap", static_cast<float>(overlapDiagSum / static_cast<double>(r)));
		append_kv(oss, "fisher_min", fMin);
		append_kv(oss, "fisher_max", fMax);
		append_kv(oss, "fisher_mean", static_cast<float>(fSum / static_cast<double>(r)));
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
	const size_t rn = static_cast<size_t>(r) * static_cast<size_t>(n);
	state.step += 1ULL;

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
	// sigma2 is the EMA of mean(G^2), providing a data-driven baseline
	// preconditioner for the complement space.
	{
		double gMeanSq = 0.0;
		for (size_t idx = 0; idx < mn; ++idx)
		{
			const double v = static_cast<double>(gW[idx]) * static_cast<double>(gScale);
			gMeanSq += v * v;
		}
		gMeanSq /= static_cast<double>(mn);
		state.sigma2 = beta * state.sigma2 + (1.0f - beta) * static_cast<float>(gMeanSq);
		if (!atlas_isfinite(state.sigma2))
		{
			state.sigma2 = 1.0f;
			recovered = true;
		}
	}

	// === Step 2: Periodic subspace refresh (EMA-blended) ===
	// Pass raw gW directly — eigenvectors of G*G^T are scale-invariant.
	if (tSub > 0u && (state.step % static_cast<unsigned long long>(tSub)) == 0ULL)
	{
		if (!refreshSubspace(state, gW, m, n, ac.powerIters, ac.betaRefresh, rng, logger))
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
	glades::gemm::atb(&gz[0], &state.U[0], gW, r, m, n, gScale);

	// === Step 5: Update Fisher diagonal (EMA of mean squared projected gradient) ===
	const float oneMinusBeta = 1.0f - beta;
	for (unsigned int c = 0; c < r; ++c)
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
	for (unsigned int c = 0; c < r; ++c)
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
	glades::gemm::ab_accum(W, &state.U[0], &corrected[0], m, r, n, 1.0f);

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
		for (unsigned int c = 0; c < r; ++c)
		{
			const float f = state.fisherDiag[c];
			if (f < fMin) fMin = f;
			if (f > fMax) fMax = f;
			fSum += static_cast<double>(f);
		}

		std::ostringstream oss;
		oss << "event=atlas_step";
		if (tag) oss << " tag=" << tag;
		append_kv(oss, "step", state.step);
		append_kv(oss, "m", m);
		append_kv(oss, "n", n);
		append_kv(oss, "rank", r);
		append_kv(oss, "lr", lr);
		append_kv(oss, "mu", state.mu);
		append_kv(oss, "sigma2", state.sigma2);
		append_kv(oss, "baseline_rate", baselineRate);
		append_kv(oss, "gz_norm", static_cast<float>(sqrt(gzNormSq)));
		append_kv(oss, "update_norm", static_cast<float>(sqrt(updateNormSq)));
		append_kv(oss, "fisher_min", fMin);
		append_kv(oss, "fisher_max", fMax);
		append_kv(oss, "fisher_mean", static_cast<float>(fSum / static_cast<double>(r)));
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}

	// === Step 9: Store compressed gradient for next step ===
	std::copy(gz.begin(), gz.end(), state.prevGz.begin());

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
