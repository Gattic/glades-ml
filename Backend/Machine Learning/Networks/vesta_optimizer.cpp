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
// Zeros out any columns that become numerically zero.
void gramSchmidt(float* Q, unsigned int m, unsigned int r)
{
	const float kTiny = 1e-12f;
	for (unsigned int j = 0; j < r; ++j)
	{
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
		if (norm <= kTiny)
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

// ---------------- Stubs (implemented in later tasks) ----------------


void initWeightState(WeightState& /*state*/,
                     const float* /*W*/,
                     unsigned int /*m*/, unsigned int /*n*/,
                     const VestaConfig& /*vc*/,
                     glades::rng::Engine& /*rng*/,
                     shmea::GLogger* /*logger*/)
{
}

bool refreshSubspace(WeightState& /*state*/,
                     const float* /*W*/,
                     unsigned int /*m*/, unsigned int /*n*/,
                     const VestaConfig& /*vc*/,
                     glades::rng::Engine& /*rng*/,
                     shmea::GLogger* /*logger*/)
{
	return false;
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
