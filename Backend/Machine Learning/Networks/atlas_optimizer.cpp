// ATLAS optimizer implementation (BRSP variant).
// See atlas_optimizer.h for API documentation.
#include "atlas_optimizer.h"
#include "Backend/Database/GLogger.h"
#include <sstream>

namespace glades {
namespace atlas {

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
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}
}

void refreshSubspace(WeightState& state, const float* grad,
                     unsigned int m, unsigned int n,
                     unsigned int powerIters, float betaRefresh,
                     glades::rng::Engine& rng,
                     shmea::GLogger* logger)
{
	const unsigned int r = state.r;
	if (r == 0u || m == 0u || n == 0u) return;

	const size_t mr = static_cast<size_t>(m) * static_cast<size_t>(r);
	const size_t nr = static_cast<size_t>(n) * static_cast<size_t>(r);

	// Save old basis and Fisher for EMA blending and Fisher transform.
	std::vector<float> U_old(state.U.begin(), state.U.end());
	std::vector<float> f_old(state.fisherDiag.begin(), state.fisherDiag.end());

	// --- Randomized power iteration (warm-started from current U) ---
	std::vector<float>& Q = state.U;
	std::vector<float> B(nr, 0.0f);
	std::vector<float> Z(mr, 0.0f);

	for (unsigned int p = 0; p < powerIters; ++p)
	{
		// B = grad^T * Q
		std::fill(B.begin(), B.end(), 0.0f);
		for (unsigned int i = 0; i < m; ++i)
		{
			for (unsigned int c = 0; c < r; ++c)
			{
				const float q_ic = Q[i * r + c];
				for (unsigned int j = 0; j < n; ++j)
					B[j * r + c] += grad[i * n + j] * q_ic;
			}
		}

		// Z = grad * B
		std::fill(Z.begin(), Z.end(), 0.0f);
		for (unsigned int i = 0; i < m; ++i)
		{
			for (unsigned int j = 0; j < n; ++j)
			{
				const float g_ij = grad[i * n + j];
				for (unsigned int c = 0; c < r; ++c)
					Z[i * r + c] += g_ij * B[j * r + c];
			}
		}

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
	std::vector<float> overlap(static_cast<size_t>(r) * static_cast<size_t>(r), 0.0f);
	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int c = 0; c < r; ++c)
		{
			const float u_ic = Q[i * r + c];
			for (unsigned int j = 0; j < r; ++j)
				overlap[c * r + j] += u_ic * U_old[i * r + j];
		}
	}

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
	std::vector<float> prevGzOld(state.prevGz.begin(), state.prevGz.end());
	std::fill(state.prevGz.begin(), state.prevGz.end(), 0.0f);
	for (unsigned int c = 0; c < r; ++c)
	{
		for (unsigned int k = 0; k < r; ++k)
		{
			const float o_ck = overlap[c * r + k];
			if (o_ck == 0.0f) continue;
			for (unsigned int j = 0; j < n; ++j)
				state.prevGz[c * n + j] += o_ck * prevGzOld[k * n + j];
		}
	}

	if (logger)
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
}

void applyStep(WeightState& state,
               float* W, float* gW,
               unsigned int m, unsigned int n,
               float invBatch, float lr,
               float wd1, float wd2, float gradScale,
               float beta, float muMin, float muMax,
               float eps, unsigned int tSub,
               unsigned int powerIters, float betaRefresh,
               glades::rng::Engine& rng,
               shmea::GLogger* logger)
{
	if (!state.initialized) return;

	const unsigned int r = state.r;
	const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
	const size_t rn = static_cast<size_t>(r) * static_cast<size_t>(n);
	state.step += 1ULL;

	const bool diagStep = logger && tSub > 0u
		&& (state.step % static_cast<unsigned long long>(tSub)) == 0ULL;

	const float gScale = invBatch * gradScale;

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
	}

	// === Step 2: Periodic subspace refresh (EMA-blended) ===
	if (tSub > 0u && (state.step % static_cast<unsigned long long>(tSub)) == 0ULL)
	{
		std::vector<float> avgGrad(mn);
		for (size_t idx = 0; idx < mn; ++idx)
			avgGrad[idx] = gW[idx] * gScale;
		refreshSubspace(state, &avgGrad[0], m, n, powerIters, betaRefresh, rng, logger);
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
	// gz[c*n+j] = sum_i U[i*r+c] * G[i*n+j]
	std::vector<float> gz(rn, 0.0f);
	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int c = 0; c < r; ++c)
		{
			const float u_ic = state.U[i * r + c];
			for (unsigned int j = 0; j < n; ++j)
				gz[c * n + j] += u_ic * (gW[i * n + j] * gScale);
		}
	}

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
	}

	// === Step 6: Full-space baseline update ===
	// W -= (lr / (sigma2 + eps)) * G
	// This provides RMSprop-like preconditioning to ALL directions,
	// ensuring no gradient information is ever discarded.
	const float baselineRate = lr / (state.sigma2 + eps);
	{
		const float baseScaled = baselineRate * gScale;
		for (size_t idx = 0; idx < mn; ++idx)
			W[idx] -= baseScaled * gW[idx];
	}

	// === Step 7: Subspace correction with optional PNG ===
	//
	// corrScale_c = lr/(sigma2+eps) - lr/(f_c+eps)
	//
	// This ADDS BACK the baseline step in subspace directions and REPLACES
	// it with Fisher-preconditioned step. The net update per direction:
	//   subspace c: -lr/(f_c+eps) * gPred_c  (Fisher-preconditioned)
	//   complement: -lr/(sigma2+eps) * G_perp (baseline-preconditioned)
	//
	// gPred = (1+mu)*gz - mu*prevGz  (PNG temporal extrapolation)
	std::vector<float> corrScale(static_cast<size_t>(r));
	for (unsigned int c = 0; c < r; ++c)
		corrScale[c] = baselineRate - lr / (state.fisherDiag[c] + eps);

	const float onePlusMu = 1.0f + state.mu;
	const float negMu = -state.mu;

	double updateNormSq = 0.0;

	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int c = 0; c < r; ++c)
		{
			const float u_cs = state.U[i * r + c] * corrScale[c];
			for (unsigned int j = 0; j < n; ++j)
			{
				const size_t cj = static_cast<size_t>(c) * static_cast<size_t>(n) + static_cast<size_t>(j);
				const float gPred = onePlusMu * gz[cj] + negMu * state.prevGz[cj];
				const float delta = u_cs * gPred;
				W[i * n + j] += delta;
				if (diagStep)
					updateNormSq += static_cast<double>(delta) * static_cast<double>(delta);
			}
		}
	}

	// === Step 8: Adapt prediction coefficient ===
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
			float newMu = state.mu * (1.0f - ratio);
			if (newMu < muMin) newMu = muMin;
			if (newMu > muMax) newMu = muMax;
			state.mu = newMu;
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
}

} // namespace atlas
} // namespace glades
