// VESTA optimizer CPU reference implementation.
// See vesta_optimizer.h for interface documentation.

#include "vesta_optimizer.h"
#include "training_config.h"
#include "Backend/Database/GLogger.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <utility>

namespace glades {
namespace vesta {

// ---------------- Small linear-algebra helpers ----------------

// Modified Gram-Schmidt on Q[m x r] row-major.
// Zeros out any columns whose post-projection residual norm drops below
// a relative threshold of the pre-projection norm — this is the correct
// rank-deficiency detector for floating-point MGS.
void gramSchmidt(float* Q, unsigned int m, unsigned int r)
{
	const float kTiny = 1e-12f;
	const float kRelThresh = 1e-5f;
	for (unsigned int j = 0; j < r; ++j)
	{
		float origNorm2 = 0.0f;
		for (unsigned int i = 0; i < m; ++i)
			origNorm2 += Q[i * r + j] * Q[i * r + j];
		const float origNorm = sqrtf(origNorm2);

		for (unsigned int k = 0; k < j; ++k)
		{
			float dot = 0.0f;
			for (unsigned int i = 0; i < m; ++i)
				dot += Q[i * r + k] * Q[i * r + j];
			for (unsigned int i = 0; i < m; ++i)
				Q[i * r + j] -= dot * Q[i * r + k];
		}
		float norm2 = 0.0f;
		for (unsigned int i = 0; i < m; ++i)
			norm2 += Q[i * r + j] * Q[i * r + j];
		const float norm = sqrtf(norm2);
		if (norm <= kTiny || (origNorm > 0.0f && norm <= kRelThresh * origNorm))
		{
			for (unsigned int i = 0; i < m; ++i)
				Q[i * r + j] = 0.0f;
		}
		else
		{
			const float inv = 1.0f / norm;
			for (unsigned int i = 0; i < m; ++i)
				Q[i * r + j] *= inv;
		}
	}
}

// Thin QR via MGS; returns false if any column is linearly dependent on
// prior columns (residual norm below 1e-5 of the original column norm).
bool thinQR(float* Q, unsigned int m, unsigned int r)
{
	const float kTiny = 1e-12f;
	const float kRelThresh = 1e-5f;
	for (unsigned int j = 0; j < r; ++j)
	{
		float origNorm2 = 0.0f;
		for (unsigned int i = 0; i < m; ++i)
			origNorm2 += Q[i * r + j] * Q[i * r + j];
		const float origNorm = sqrtf(origNorm2);

		for (unsigned int k = 0; k < j; ++k)
		{
			float dot = 0.0f;
			for (unsigned int i = 0; i < m; ++i)
				dot += Q[i * r + k] * Q[i * r + j];
			for (unsigned int i = 0; i < m; ++i)
				Q[i * r + j] -= dot * Q[i * r + k];
		}
		float norm2 = 0.0f;
		for (unsigned int i = 0; i < m; ++i)
			norm2 += Q[i * r + j] * Q[i * r + j];
		const float norm = sqrtf(norm2);
		if (norm <= kTiny || norm <= kRelThresh * origNorm)
			return false;
		const float inv = 1.0f / norm;
		for (unsigned int i = 0; i < m; ++i)
			Q[i * r + j] *= inv;
	}
	return true;
}

// ---------------- Jacobi eigendecomposition for small SVD ----------------

// One Jacobi sweep on symmetric matrix S[n x n] row-major, accumulating
// rotations into Vacc[n x n]. Returns the total absolute off-diagonal mass
// after the sweep through *off.
static void jacobi_sweep(float* S, float* Vacc, unsigned int n, float* off)
{
	float offSum = 0.0f;
	for (unsigned int p = 0; p < n; ++p)
	{
		for (unsigned int q = p + 1; q < n; ++q)
		{
			const float spq = S[p * n + q];
			const float spp = S[p * n + p];
			const float sqq = S[q * n + q];
			const float absSpq = fabsf(spq);
			offSum += absSpq;
			if (absSpq < 1e-14f)
				continue;

			const float theta = (sqq - spp) / (2.0f * spq);
			float t;
			if (theta >= 0.0f)
				t = 1.0f / (theta + sqrtf(1.0f + theta * theta));
			else
				t = 1.0f / (theta - sqrtf(1.0f + theta * theta));
			const float c = 1.0f / sqrtf(1.0f + t * t);
			const float s = t * c;

			S[p * n + p] = spp - t * spq;
			S[q * n + q] = sqq + t * spq;
			S[p * n + q] = 0.0f;
			S[q * n + p] = 0.0f;
			for (unsigned int k = 0; k < n; ++k)
			{
				if (k != p && k != q)
				{
					const float skp = S[k * n + p];
					const float skq = S[k * n + q];
					S[k * n + p] = c * skp - s * skq;
					S[k * n + q] = s * skp + c * skq;
					S[p * n + k] = S[k * n + p];
					S[q * n + k] = S[k * n + q];
				}
			}
			for (unsigned int k = 0; k < n; ++k)
			{
				const float vkp = Vacc[k * n + p];
				const float vkq = Vacc[k * n + q];
				Vacc[k * n + p] = c * vkp - s * vkq;
				Vacc[k * n + q] = s * vkp + c * vkq;
			}
		}
	}
	*off = offSum;
}

bool denseSVD_rightV(const float* B, unsigned int mB, unsigned int nB,
                     float* Vout, float* sOut, unsigned int r)
{
	if (r == 0u || r > nB)
		return false;

	// S = B^T B, symmetric PSD [nB x nB].
	std::vector<float> S(static_cast<size_t>(nB) * nB, 0.0f);
	for (unsigned int i = 0; i < nB; ++i)
	{
		for (unsigned int j = i; j < nB; ++j)
		{
			float dot = 0.0f;
			for (unsigned int k = 0; k < mB; ++k)
				dot += B[k * nB + i] * B[k * nB + j];
			S[i * nB + j] = dot;
			S[j * nB + i] = dot;
		}
	}

	// V = I_{nB}.
	std::vector<float> V(static_cast<size_t>(nB) * nB, 0.0f);
	for (unsigned int i = 0; i < nB; ++i)
		V[i * nB + i] = 1.0f;

	const unsigned int maxSweeps = 80u;
	for (unsigned int sweep = 0; sweep < maxSweeps; ++sweep)
	{
		float off = 0.0f;
		jacobi_sweep(&S[0], &V[0], nB, &off);
		if (off < 1e-12f)
			break;
	}

	// Singular values = sqrt(max(eig, 0)).
	std::vector<std::pair<float, unsigned int> > eigs(nB);
	for (unsigned int i = 0; i < nB; ++i)
	{
		const float ev = S[i * nB + i];
		const float sv = (ev > 0.0f) ? sqrtf(ev) : 0.0f;
		eigs[i] = std::make_pair(sv, i);
	}
	for (unsigned int i = 0; i < nB; ++i)
	{
		unsigned int maxIdx = i;
		for (unsigned int j = i + 1; j < nB; ++j)
			if (eigs[j].first > eigs[maxIdx].first)
				maxIdx = j;
		if (maxIdx != i)
			std::swap(eigs[i], eigs[maxIdx]);
	}

	for (unsigned int i = 0; i < r; ++i)
	{
		sOut[i] = eigs[i].first;
		const unsigned int col = eigs[i].second;
		for (unsigned int k = 0; k < nB; ++k)
			Vout[k * r + i] = V[k * nB + col];
	}
	return true;
}

// ---------------- State initialization and sketched SVD refresh ----------------

static void allocate_scratch(WeightState& s)
{
	const unsigned int m = s.m, n = s.n, r = s.r;
	const unsigned int over = 8u;
	const unsigned int rp = r + over;
	s.scratch_A.assign(static_cast<size_t>(r) * r, 0.0f);
	s.scratch_UA.assign(static_cast<size_t>(m) * r, 0.0f);
	s.scratch_WrOld.assign(static_cast<size_t>(m) * n, 0.0f);
	s.scratch_WrNew.assign(static_cast<size_t>(m) * n, 0.0f);
	s.scratch_gPerp.assign(static_cast<size_t>(m) * n, 0.0f);
	s.scratch_Omega_U.assign(static_cast<size_t>(m) * r, 0.0f);
	s.scratch_Omega_V.assign(static_cast<size_t>(n) * r, 0.0f);
	s.scratch_URaw.assign(static_cast<size_t>(m) * r, 0.0f);
	s.scratch_VRaw.assign(static_cast<size_t>(n) * r, 0.0f);
	s.scratch_sketchOmega.assign(static_cast<size_t>(n) * rp, 0.0f);
	s.scratch_sketchY.assign(static_cast<size_t>(m) * rp, 0.0f);
	s.scratch_sketchB.assign(static_cast<size_t>(rp) * n, 0.0f);
	s.scratch_sketchVr.assign(static_cast<size_t>(n) * r, 0.0f);
	s.scratch_sketchS.assign(rp, 0.0f);
}

// Randomized range-finder sketched SVD:
//   1. Omega ~ N(0,1) [n, rp]
//   2. Y = W Omega [m, rp]
//   3. Power iterations: Y = W (W^T Y), thin-QR after each
//   4. U_over = QR(Y) [m, rp]
//   5. B = U_over^T W [rp, n]
//   6. Dense SVD of B: right singular vectors V [n, r], singular values s [r]
//   7. Reassemble U = U_over * U_B[:, :r]
static bool sketched_svd(const float* W, unsigned int m, unsigned int n,
                         unsigned int r,
                         WeightState& s,
                         const VestaConfig& vc,
                         glades::rng::Engine& rng)
{
	const unsigned int over = 8u;
	unsigned int rp = r + over;
	if (rp > m) rp = m;
	if (rp > n) rp = n;
	if (rp < r) return false;

	for (size_t i = 0; i < static_cast<size_t>(n) * rp; ++i)
		s.scratch_sketchOmega[i] = glades::rng::standard_normal(rng);

	// Y = W Omega
	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int c = 0; c < rp; ++c)
		{
			float acc = 0.0f;
			for (unsigned int k = 0; k < n; ++k)
				acc += W[i * n + k] * s.scratch_sketchOmega[k * rp + c];
			s.scratch_sketchY[i * rp + c] = acc;
		}
	}

	std::vector<float> WtY(static_cast<size_t>(n) * rp, 0.0f);
	for (unsigned int p = 0; p < vc.powerIters; ++p)
	{
		// gramSchmidt tolerates rank-deficient input (zeros out degenerate columns),
		// which is important when W has low effective rank: power iteration drives
		// the oversample columns into the same subspace.
		gramSchmidt(&s.scratch_sketchY[0], m, rp);
		for (unsigned int j = 0; j < n; ++j)
		{
			for (unsigned int c = 0; c < rp; ++c)
			{
				float acc = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					acc += W[i * n + j] * s.scratch_sketchY[i * rp + c];
				WtY[j * rp + c] = acc;
			}
		}
		for (unsigned int i = 0; i < m; ++i)
		{
			for (unsigned int c = 0; c < rp; ++c)
			{
				float acc = 0.0f;
				for (unsigned int k = 0; k < n; ++k)
					acc += W[i * n + k] * WtY[k * rp + c];
				s.scratch_sketchY[i * rp + c] = acc;
			}
		}
	}

	gramSchmidt(&s.scratch_sketchY[0], m, rp);

	// B = Y^T W  [rp, n]
	for (unsigned int c = 0; c < rp; ++c)
	{
		for (unsigned int j = 0; j < n; ++j)
		{
			float acc = 0.0f;
			for (unsigned int k = 0; k < m; ++k)
				acc += s.scratch_sketchY[k * rp + c] * W[k * n + j];
			s.scratch_sketchB[c * n + j] = acc;
		}
	}

	std::vector<float> Vrp(static_cast<size_t>(n) * rp, 0.0f);
	std::vector<float> srp(rp, 0.0f);
	if (!denseSVD_rightV(&s.scratch_sketchB[0], rp, n, &Vrp[0], &srp[0], rp))
		return false;

	// Singular-value threshold: below this we declare rank deficiency and
	// fill U[:, i], V[:, i] with random orthonormal extensions. We use a
	// generous relative threshold because when the true rank is < r, the
	// "tail" singular value comes from numerical noise in B which can easily
	// be 1e-3 to 1e-1 of the leading singular value in float precision.
	const float kRankThresh = std::max(1e-6f, srp[0] * 1e-3f);

	// U_B[:, i] = B V[:, i] / s[i] for valid directions; zeros for deficient ones.
	std::vector<float> UB(static_cast<size_t>(rp) * r, 0.0f);
	std::vector<bool> validCol(r, false);
	for (unsigned int i = 0; i < r; ++i)
	{
		const float si = srp[i];
		if (si > kRankThresh)
		{
			validCol[i] = true;
			for (unsigned int a = 0; a < rp; ++a)
			{
				float acc = 0.0f;
				for (unsigned int j = 0; j < n; ++j)
					acc += s.scratch_sketchB[a * n + j] * Vrp[j * rp + i];
				UB[a * r + i] = acc / si;
			}
		}
	}

	std::vector<float> Ufinal(static_cast<size_t>(m) * r, 0.0f);
	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int c = 0; c < r; ++c)
		{
			if (!validCol[c]) continue;
			float acc = 0.0f;
			for (unsigned int a = 0; a < rp; ++a)
				acc += s.scratch_sketchY[i * rp + a] * UB[a * r + c];
			Ufinal[i * r + c] = acc;
		}
	}

	std::vector<float> Vfinal(static_cast<size_t>(n) * r, 0.0f);
	for (unsigned int j = 0; j < n; ++j)
		for (unsigned int c = 0; c < r; ++c)
			if (validCol[c])
				Vfinal[j * r + c] = Vrp[j * rp + c];

	// Fill deficient columns with random vectors then orthonormalize U, V.
	for (unsigned int c = 0; c < r; ++c)
	{
		if (!validCol[c])
		{
			for (unsigned int i = 0; i < m; ++i)
				Ufinal[i * r + c] = glades::rng::standard_normal(rng);
			for (unsigned int j = 0; j < n; ++j)
				Vfinal[j * r + c] = glades::rng::standard_normal(rng);
		}
	}
	gramSchmidt(&Ufinal[0], m, r);
	gramSchmidt(&Vfinal[0], n, r);

	s.U = Ufinal;
	s.V = Vfinal;
	s.ell.assign(r, 0.0f);
	for (unsigned int i = 0; i < r; ++i)
	{
		float l;
		if (validCol[i])
			l = logf(std::max(srp[i], 1e-20f));
		else
			l = vc.ellMin;
		if (l < vc.ellMin) l = vc.ellMin;
		if (l > vc.ellMax) l = vc.ellMax;
		s.ell[i] = l;
	}
	return true;
}

void initWeightState(WeightState& state,
                     const float* W,
                     unsigned int m, unsigned int n,
                     const VestaConfig& vc,
                     glades::rng::Engine& rng,
                     shmea::GLogger* /*logger*/)
{
	unsigned int r = vc.rank;
	const unsigned int dMin = std::min(m, n);
	if (r > dMin) r = dMin;
	if (r == 0u) r = 1u;

	state.reset();
	state.m = m;
	state.n = n;
	state.r = r;
	state.U.assign(static_cast<size_t>(m) * r, 0.0f);
	state.V.assign(static_cast<size_t>(n) * r, 0.0f);
	state.ell.assign(r, 0.0f);
	state.beta.assign(r, 0.0f);
	state.ellStar.assign(r, 0.0f);
	allocate_scratch(state);

	if (!sketched_svd(W, m, n, r, state, vc, rng))
	{
		// Fallback: identity-like basis, unit singular values.
		for (unsigned int i = 0; i < r; ++i)
		{
			state.U[i * r + i] = 1.0f;
			state.V[i * r + i] = 1.0f;
			state.ell[i] = 0.0f;
		}
	}
	state.beta = state.ell;
	state.ellStar = state.ell;
	state.maxExpEllPrev = expf(state.ell[0]);
	state.step = 0ULL;
	state.initialized = true;
}

bool refreshSubspace(WeightState& state,
                     const float* W,
                     unsigned int m, unsigned int n,
                     const VestaConfig& vc,
                     glades::rng::Engine& rng,
                     shmea::GLogger* /*logger*/)
{
	if (!state.initialized || state.m != m || state.n != n)
		return false;
	return sketched_svd(W, m, n, state.r, state, vc, rng);
}

bool applyStep(WeightState& /*state*/,
               float* /*W*/, float* /*gW*/,
               unsigned int /*m*/, unsigned int /*n*/,
               float /*invBatch*/, float /*lr*/,
               float /*wd1*/, float /*wd2*/, float /*gradScale*/,
               const VestaConfig& /*vc*/,
               glades::rng::Engine& /*rng*/,
               shmea::GLogger* /*logger*/,
               const char* /*tag*/)
{
	return false;
}

bool update(WeightState& state,
            float* W, float* gW,
            unsigned int m, unsigned int n,
            float invBatch, float lr,
            float wd1, float wd2, float gradScale,
            const VestaConfig& vc,
            glades::rng::Engine& rng,
            shmea::GLogger* logger,
            const char* tag)
{
	if (!state.initialized)
		initWeightState(state, W, m, n, vc, rng, logger);
	return applyStep(state, W, gW, m, n, invBatch, lr, wd1, wd2, gradScale, vc, rng, logger, tag);
}

} // namespace vesta
} // namespace glades
