// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "vesta-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/vesta_optimizer.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/rng.h"

#ifdef GLADES_HAVE_CUDA
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_vesta.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_device.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_buffer.h"
#endif

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

static bool close_abs(float a, float b, float tol)
{
	return fabsf(a - b) <= tol;
}

static void assert_close(const char* label, float got, float expected, float tol)
{
	char msg[256];
	sprintf(msg, "%s: got %.6g expected %.6g tol %.6g", label, got, expected, tol);
	ASSERT(msg, close_abs(got, expected, tol));
}

static void fill_random(std::vector<float>& out, unsigned int m, unsigned int r, uint64_t seed)
{
	out.assign(static_cast<size_t>(m) * r, 0.0f);
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, seed);
	for (size_t i = 0; i < out.size(); ++i)
		out[i] = glades::rng::standard_normal(eng);
}

// Check that Q[m x r] row-major has orthonormal columns.
static float orth_error(const float* Q, unsigned int m, unsigned int r)
{
	float maxErr = 0.0f;
	for (unsigned int i = 0; i < r; ++i)
	{
		for (unsigned int j = i; j < r; ++j)
		{
			float dot = 0.0f;
			for (unsigned int k = 0; k < m; ++k)
				dot += Q[k * r + i] * Q[k * r + j];
			const float expected = (i == j) ? 1.0f : 0.0f;
			const float err = fabsf(dot - expected);
			if (err > maxErr) maxErr = err;
		}
	}
	return maxErr;
}

} // namespace

void VESTAGramSchmidtTest()
{
	printf("[vesta] GramSchmidtTest\n");
	const unsigned int m = 16;
	const unsigned int r = 4;
	std::vector<float> Q;
	fill_random(Q, m, r, 0xC0FFEEULL);

	glades::vesta::gramSchmidt(&Q[0], m, r);
	const float err = orth_error(&Q[0], m, r);
	assert_close("gramSchmidt orth err", err, 0.0f, 1e-5f);
}

void VESTAThinQRTest()
{
	printf("[vesta] ThinQRTest\n");
	const unsigned int m = 32;
	const unsigned int r = 8;
	std::vector<float> Q;
	fill_random(Q, m, r, 0xDECAFULL);

	const bool ok = glades::vesta::thinQR(&Q[0], m, r);
	ASSERT("thinQR returned false on full-rank input", ok);
	const float err = orth_error(&Q[0], m, r);
	assert_close("thinQR orth err", err, 0.0f, 1e-5f);

	// Rank-deficient case: third column is a copy of the first.
	std::vector<float> QDef;
	fill_random(QDef, m, r, 0xBADF00DULL);
	for (unsigned int i = 0; i < m; ++i)
		QDef[i * r + 2] = QDef[i * r + 0];
	const bool okDef = glades::vesta::thinQR(&QDef[0], m, r);
	ASSERT("thinQR returned true on rank-deficient input", !okDef);
}

void VESTASketchedSVDTest()
{
	printf("[vesta] SketchedSVDTest\n");
	// Diagonal B[4 x 8] = diag(4,3,2,1) padded.
	const unsigned int mB = 4, nB = 8;
	std::vector<float> B(static_cast<size_t>(mB) * nB, 0.0f);
	for (unsigned int i = 0; i < 4; ++i)
		B[i * nB + i] = static_cast<float>(4 - i);

	const unsigned int r = 3;
	std::vector<float> V(static_cast<size_t>(nB) * r, 0.0f);
	std::vector<float> s(r, 0.0f);
	const bool ok = glades::vesta::denseSVD_rightV(&B[0], mB, nB, &V[0], &s[0], r);
	ASSERT("denseSVD returned false", ok);

	assert_close("s[0]", s[0], 4.0f, 1e-4f);
	assert_close("s[1]", s[1], 3.0f, 1e-4f);
	assert_close("s[2]", s[2], 2.0f, 1e-4f);

	const float err = orth_error(&V[0], nB, r);
	assert_close("denseSVD V orth err", err, 0.0f, 1e-4f);
}

void VESTAInitStateTest()
{
	printf("[vesta] InitStateTest (stub)\n");
}

void VESTALogScaleUpdateTest()
{
	printf("[vesta] LogScaleUpdateTest (stub)\n");
}

void VESTATrustRegionClampTest()
{
	printf("[vesta] TrustRegionClampTest (stub)\n");
}

void VESTAStepDescentTest()
{
	printf("[vesta] StepDescentTest (stub)\n");
}

void VESTAOrthogonalInvarianceTest()
{
	printf("[vesta] OrthogonalInvarianceTest (stub)\n");
}

void VESTAGpuParityTest()
{
	printf("[vesta] GpuParityTest (stub)\n");
}

void VESTAUnitTest()
{
	VESTAGramSchmidtTest();
	VESTAThinQRTest();
	VESTASketchedSVDTest();
}
