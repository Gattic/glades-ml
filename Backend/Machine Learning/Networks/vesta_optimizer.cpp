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

// Thin QR via MGS; returns false if any column drops to near-zero.
bool thinQR(float* Q, unsigned int m, unsigned int r)
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
			return false;
		const float inv = 1.0f / norm;
		for (unsigned int i = 0; i < m; ++i)
			Q[i * r + j] *= inv;
	}
	return true;
}

// ---------------- Stubs (implemented in later tasks) ----------------

bool denseSVD_rightV(const float* /*B*/, unsigned int /*mB*/, unsigned int /*nB*/,
                     float* /*Vout*/, float* /*sOut*/, unsigned int /*r*/)
{
	return false;
}

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
