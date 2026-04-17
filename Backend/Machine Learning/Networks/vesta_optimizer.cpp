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

	// Which side is smaller? For B shape (mB x nB) with mB << nB (typical in
	// the sketched-SVD path where mB = r+8 and nB = dModel), we Jacobi on
	// B B^T (size mB x mB = small) and derive right singular vectors via
	//   V[:, i] = B^T U_B[:, i] / sigma_i
	// This avoids the O(nB^3) cost of Jacobi on B^T B. For mB >= nB we keep
	// the original B^T B path (nB is small).
	if (mB < nB)
	{
		// Small side Jacobi: S = B B^T, symmetric PSD [mB x mB].
		std::vector<float> S(static_cast<size_t>(mB) * mB, 0.0f);
		for (unsigned int i = 0; i < mB; ++i)
		{
			for (unsigned int j = i; j < mB; ++j)
			{
				float dot = 0.0f;
				for (unsigned int k = 0; k < nB; ++k)
					dot += B[i * nB + k] * B[j * nB + k];
				S[i * mB + j] = dot;
				S[j * mB + i] = dot;
			}
		}

		std::vector<float> U_B(static_cast<size_t>(mB) * mB, 0.0f);
		for (unsigned int i = 0; i < mB; ++i)
			U_B[i * mB + i] = 1.0f;

		const unsigned int maxSweeps = 80u;
		for (unsigned int sweep = 0; sweep < maxSweeps; ++sweep)
		{
			float off = 0.0f;
			jacobi_sweep(&S[0], &U_B[0], mB, &off);
			if (off < 1e-12f)
				break;
		}

		// Extract (eigenvalue, index), sort descending by eigenvalue.
		std::vector<std::pair<float, unsigned int> > eigs(mB);
		for (unsigned int i = 0; i < mB; ++i)
		{
			const float ev = S[i * mB + i];
			const float sv = (ev > 0.0f) ? sqrtf(ev) : 0.0f;
			eigs[i] = std::make_pair(sv, i);
		}
		for (unsigned int i = 0; i < mB; ++i)
		{
			unsigned int maxIdx = i;
			for (unsigned int j = i + 1; j < mB; ++j)
				if (eigs[j].first > eigs[maxIdx].first)
					maxIdx = j;
			if (maxIdx != i)
				std::swap(eigs[i], eigs[maxIdx]);
		}

		// r could exceed mB — VESTA callers ask for r = min(m_orig, n_orig)
		// top singular values; any beyond mB have singular value 0 and any
		// orthonormal vectors in the n-space complement work. Limit to
		// min(r, mB) for the well-defined part; pad the remainder with zeros
		// and orthogonal extension via gramSchmidt on random fill.
		const unsigned int rReal = (r < mB) ? r : mB;
		for (unsigned int i = 0; i < rReal; ++i)
		{
			const float si = eigs[i].first;
			sOut[i] = si;
			const unsigned int col = eigs[i].second;
			// V[:, i] = B^T U_B[:, col] / si
			if (si > 1e-12f)
			{
				const float invSi = 1.0f / si;
				for (unsigned int k = 0; k < nB; ++k)
				{
					float acc = 0.0f;
					for (unsigned int a = 0; a < mB; ++a)
						acc += B[a * nB + k] * U_B[a * mB + col];
					Vout[k * r + i] = acc * invSi;
				}
			}
			else
			{
				for (unsigned int k = 0; k < nB; ++k)
					Vout[k * r + i] = 0.0f;
			}
		}
		for (unsigned int i = rReal; i < r; ++i)
		{
			sOut[i] = 0.0f;
			for (unsigned int k = 0; k < nB; ++k)
				Vout[k * r + i] = 0.0f;
		}
		return true;
	}

	// Fallback (mB >= nB): original B^T B path. nB is small here so the
	// O(nB^3) cost is cheap.
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

// ---------------- Step helpers ----------------

// W_r = U diag(exp(ell)) V^T  [m * n]
static void reconstruct_rank_block(const float* U, const float* V, const float* ell,
                                   unsigned int m, unsigned int n, unsigned int r,
                                   float* out)
{
	for (unsigned int i = 0; i < m; ++i)
	{
		for (unsigned int j = 0; j < n; ++j)
		{
			float acc = 0.0f;
			for (unsigned int k = 0; k < r; ++k)
				acc += U[i * r + k] * expf(ell[k]) * V[j * r + k];
			out[i * n + j] = acc;
		}
	}
}

// A = U^T g V [r * r]; also stores (g V) in scratchUA[m * r].
static void compute_A_and_gV(const float* U, const float* g, const float* V,
                             unsigned int m, unsigned int n, unsigned int r,
                             float* scratchUA, float* A)
{
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int k = 0; k < r; ++k)
		{
			float acc = 0.0f;
			for (unsigned int j = 0; j < n; ++j)
				acc += g[i * n + j] * V[j * r + k];
			scratchUA[i * r + k] = acc;
		}
	for (unsigned int a = 0; a < r; ++a)
		for (unsigned int b = 0; b < r; ++b)
		{
			float acc = 0.0f;
			for (unsigned int i = 0; i < m; ++i)
				acc += U[i * r + a] * scratchUA[i * r + b];
			A[a * r + b] = acc;
		}
}

// g_perp = g - U A V^T [m * n]; uses scratchUA[m * r] as temp.
static void form_g_perp(const float* g,
                        const float* U, const float* A, const float* V,
                        unsigned int m, unsigned int n, unsigned int r,
                        float* scratchUA, float* g_perp)
{
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int b = 0; b < r; ++b)
		{
			float acc = 0.0f;
			for (unsigned int a = 0; a < r; ++a)
				acc += U[i * r + a] * A[a * r + b];
			scratchUA[i * r + b] = acc;
		}
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int j = 0; j < n; ++j)
		{
			float acc = 0.0f;
			for (unsigned int b = 0; b < r; ++b)
				acc += scratchUA[i * r + b] * V[j * r + b];
			g_perp[i * n + j] = g[i * n + j] - acc;
		}
}

bool applyStep(WeightState& state,
               float* W, float* gW,
               unsigned int m, unsigned int n,
               float invBatch, float lr,
               float /*wd1*/, float /*wd2*/, float gradScale,
               const VestaConfig& vc,
               glades::rng::Engine& rng,
               shmea::GLogger* /*logger*/,
               const char* /*tag*/)
{
	if (!state.initialized || state.m != m || state.n != n) return false;
	const unsigned int r = state.r;
	const size_t mn = static_cast<size_t>(m) * n;

	// Step 0: apply invBatch and gradScale to gW in place.
	const float gFactor = gradScale * invBatch;
	for (size_t i = 0; i < mn; ++i)
		gW[i] *= gFactor;

	// Update gradient EMA (used as basis source when vc.basisSource == 1).
	if (vc.basisSource == 1u)
	{
		if (state.gradientEma.size() != mn)
		{
			state.gradientEma.assign(mn, 0.0f);
			state.gradientEmaWarm = false;
		}
		const float b = vc.basisEmaBeta;
		const float ombeta = 1.0f - b;
		for (size_t i = 0; i < mn; ++i)
			state.gradientEma[i] = b * state.gradientEma[i] + ombeta * gW[i];
		state.gradientEmaWarm = true;
	}

	// Step 1: maybe refresh subspace.
	if (state.step != 0ULL && vc.tSk > 0u && (state.step % vc.tSk) == 0ULL)
	{
		const float* basisSrc = W;
		if (vc.basisSource == 1u && state.gradientEmaWarm)
			basisSrc = &state.gradientEma[0];
		if (!sketched_svd(basisSrc, m, n, r, state, vc, rng))
			return false;
		// Note: aDiagEma is deliberately NOT reset on refresh. In practice
		// consecutive refreshes produce small (U,V) rotations and the EMA's
		// drift across them is empirically smaller than the variance reduction
		// it provides per-step. Resetting would kill the EMA's benefit when
		// tSk is small.
	}

	// Step 2: compute WrOld, A, g_perp.
	reconstruct_rank_block(&state.U[0], &state.V[0], &state.ell[0], m, n, r,
	                       &state.scratch_WrOld[0]);
	compute_A_and_gV(&state.U[0], gW, &state.V[0], m, n, r,
	                 &state.scratch_UA[0], &state.scratch_A[0]);
	form_g_perp(gW, &state.U[0], &state.scratch_A[0], &state.V[0], m, n, r,
	            &state.scratch_UA[0], &state.scratch_gPerp[0]);

	// Step 3: log-scale update using OLD ell (save a copy).
	std::vector<float> ellOld = state.ell;
	std::vector<float> ellNew(r, 0.0f);

	// Optional EMA of tracked-subspace diagonal (sign-stabilized update).
	if (vc.trackedEmaEnabled)
	{
		if (state.aDiagEma.size() != r)
			state.aDiagEma.assign(r, 0.0f);
		const float b = vc.trackedEmaBeta;
		const float ombeta = 1.0f - b;
		for (unsigned int i = 0; i < r; ++i)
			state.aDiagEma[i] = b * state.aDiagEma[i] + ombeta * state.scratch_A[i * r + i];
	}

	for (unsigned int i = 0; i < r; ++i)
	{
		const float l = ellOld[i];
		float phi_dd = -2.0f * l - 3.0f + vc.mu;
		if (phi_dd < vc.phiDdFloor) phi_dd = vc.phiDdFloor;
		const float sigma = expf(l);
		const float denom = phi_dd * sigma * sigma;
		const float Aii = vc.trackedEmaEnabled
		                      ? state.aDiagEma[i]
		                      : state.scratch_A[i * r + i];
		float lNext = l - lr * Aii / denom - lr * vc.tau * (l - state.ellStar[i]);
		if (lNext < vc.ellMin) lNext = vc.ellMin;
		if (lNext > vc.ellMax) lNext = vc.ellMax;
		ellNew[i] = lNext;
	}

	// Step 4: log-scale momentum.
	std::vector<float> betaNew(r, 0.0f);
	for (unsigned int i = 0; i < r; ++i)
	{
		betaNew[i] = (1.0f - vc.gamma) * state.beta[i] + vc.gamma * ellNew[i];
		ellNew[i] = (1.0f - vc.kappa) * ellNew[i] + vc.kappa * betaNew[i];
		if (ellNew[i] < vc.ellMin) ellNew[i] = vc.ellMin;
		if (ellNew[i] > vc.ellMax) ellNew[i] = vc.ellMax;
	}

	// Step 5: Stiefel QR retraction on U.
	// Omega_U = (I - U U^T) gW V diag(exp(-ell_old))
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int k = 0; k < r; ++k)
		{
			float acc = 0.0f;
			for (unsigned int j = 0; j < n; ++j)
				acc += gW[i * n + j] * state.V[j * r + k];
			state.scratch_Omega_U[i * r + k] = acc;
		}
	std::vector<float> UtOmU(static_cast<size_t>(r) * r, 0.0f);
	for (unsigned int a = 0; a < r; ++a)
		for (unsigned int b = 0; b < r; ++b)
		{
			float acc = 0.0f;
			for (unsigned int i = 0; i < m; ++i)
				acc += state.U[i * r + a] * state.scratch_Omega_U[i * r + b];
			UtOmU[a * r + b] = acc;
		}
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int b = 0; b < r; ++b)
		{
			float acc = 0.0f;
			for (unsigned int a = 0; a < r; ++a)
				acc += state.U[i * r + a] * UtOmU[a * r + b];
			state.scratch_Omega_U[i * r + b] -= acc;
		}
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int k = 0; k < r; ++k)
			state.scratch_Omega_U[i * r + k] *= expf(-ellOld[k]);
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int k = 0; k < r; ++k)
			state.scratch_URaw[i * r + k] = state.U[i * r + k] - lr * state.scratch_Omega_U[i * r + k];
	gramSchmidt(&state.scratch_URaw[0], m, r);

	// Stiefel QR retraction on V. Omega_V = (I - V V^T) gW^T U diag(exp(-ell_old))
	for (unsigned int j = 0; j < n; ++j)
		for (unsigned int k = 0; k < r; ++k)
		{
			float acc = 0.0f;
			for (unsigned int i = 0; i < m; ++i)
				acc += gW[i * n + j] * state.U[i * r + k];
			state.scratch_Omega_V[j * r + k] = acc;
		}
	std::vector<float> VtOmV(static_cast<size_t>(r) * r, 0.0f);
	for (unsigned int a = 0; a < r; ++a)
		for (unsigned int b = 0; b < r; ++b)
		{
			float acc = 0.0f;
			for (unsigned int j = 0; j < n; ++j)
				acc += state.V[j * r + a] * state.scratch_Omega_V[j * r + b];
			VtOmV[a * r + b] = acc;
		}
	for (unsigned int j = 0; j < n; ++j)
		for (unsigned int b = 0; b < r; ++b)
		{
			float acc = 0.0f;
			for (unsigned int a = 0; a < r; ++a)
				acc += state.V[j * r + a] * VtOmV[a * r + b];
			state.scratch_Omega_V[j * r + b] -= acc;
		}
	for (unsigned int j = 0; j < n; ++j)
		for (unsigned int k = 0; k < r; ++k)
			state.scratch_Omega_V[j * r + k] *= expf(-ellOld[k]);
	for (unsigned int j = 0; j < n; ++j)
		for (unsigned int k = 0; k < r; ++k)
			state.scratch_VRaw[j * r + k] = state.V[j * r + k] - lr * state.scratch_Omega_V[j * r + k];
	gramSchmidt(&state.scratch_VRaw[0], n, r);

	// Step 6: write new state.
	state.U = state.scratch_URaw;
	state.V = state.scratch_VRaw;
	state.ell = ellNew;
	state.beta = betaNew;

	// Step 7: reconstruct new rank block and apply delta to W.
	reconstruct_rank_block(&state.U[0], &state.V[0], &state.ell[0], m, n, r,
	                       &state.scratch_WrNew[0]);
	for (size_t i = 0; i < mn; ++i)
		W[i] += (state.scratch_WrNew[i] - state.scratch_WrOld[i]);

	// Step 8: signed complement step.
	// Classical form:          W -= lr * c_perp * sign(g_perp)
	// With Lion-style momentum: m = beta * m + (1-beta) * g_perp
	//                           W -= lr * c_perp * sign(m)
	float meanInvSigma = 0.0f;
	for (unsigned int i = 0; i < r; ++i)
		meanInvSigma += expf(-state.ell[i]);
	meanInvSigma /= static_cast<float>(r);
	const float c_perp = vc.lambdaPerp / (meanInvSigma > 1e-12f ? meanInvSigma : 1e-12f);

	if (vc.complementMomentumEnabled)
	{
		if (state.complementMomentum.size() != mn)
			state.complementMomentum.assign(mn, 0.0f);
		const float b = vc.complementBeta;
		const float ombeta = 1.0f - b;
		if (vc.complementUseSign)
		{
			// Lion-style: fixed-magnitude signed step via sign of EMA.
			for (size_t i = 0; i < mn; ++i)
			{
				float m = state.complementMomentum[i];
				m = b * m + ombeta * state.scratch_gPerp[i];
				state.complementMomentum[i] = m;
				const float sgn = (m > 0.0f) ? 1.0f : ((m < 0.0f) ? -1.0f : 0.0f);
				W[i] -= lr * c_perp * sgn;
			}
		}
		else
		{
			// Raw heavy-ball: step shrinks with gradient magnitude, enabling
			// fine-tuning at long horizons. lambdaPerp tuning scale changes
			// (typically 3-10x larger than sign-mode).
			for (size_t i = 0; i < mn; ++i)
			{
				float m = state.complementMomentum[i];
				m = b * m + ombeta * state.scratch_gPerp[i];
				state.complementMomentum[i] = m;
				W[i] -= lr * vc.lambdaPerp * m;
			}
		}
	}
	else
	{
		for (size_t i = 0; i < mn; ++i)
		{
			const float gp = state.scratch_gPerp[i];
			const float sgn = (gp > 0.0f) ? 1.0f : ((gp < 0.0f) ? -1.0f : 0.0f);
			W[i] -= lr * c_perp * sgn;
		}
	}

	// Step 9: trust-region clamp on max exp(ell).
	float curMaxExpEll = expf(state.ell[0]);
	for (unsigned int i = 1; i < r; ++i)
	{
		const float c = expf(state.ell[i]);
		if (c > curMaxExpEll) curMaxExpEll = c;
	}
	if (curMaxExpEll > (1.0f + vc.rho) * state.maxExpEllPrev)
	{
		const float allowed = (1.0f + vc.rho) * state.maxExpEllPrev;
		unsigned int iMax = 0;
		for (unsigned int i = 1; i < r; ++i)
			if (state.ell[i] > state.ell[iMax]) iMax = i;
		state.ell[iMax] = logf(allowed);
		if (state.beta[iMax] > state.ell[iMax])
			state.beta[iMax] = state.ell[iMax];
		curMaxExpEll = allowed;
	}
	state.maxExpEllPrev = curMaxExpEll;

	// Step 10: homeostatic update of ellStar.
	if (vc.tHom > 0u && ((state.step + 1ULL) % vc.tHom) == 0ULL)
	{
		for (unsigned int i = 0; i < r; ++i)
			state.ellStar[i] = (1.0f - vc.nu) * state.ellStar[i] + vc.nu * state.ell[i];
	}

	// Step 11: finiteness check.
	bool allFinite = true;
	for (unsigned int i = 0; i < r && allFinite; ++i)
		if (!(state.ell[i] == state.ell[i])) allFinite = false;
	for (size_t i = 0; i < state.U.size() && allFinite; ++i)
		if (!(state.U[i] == state.U[i])) allFinite = false;
	for (size_t i = 0; i < state.V.size() && allFinite; ++i)
		if (!(state.V[i] == state.V[i])) allFinite = false;

	std::memset(gW, 0, mn * sizeof(float));
	state.step += 1ULL;
	return allFinite;
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
