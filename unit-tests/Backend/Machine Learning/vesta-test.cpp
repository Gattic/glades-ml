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
#include "test_token_id_input_fixture.h"

#include "../../../Backend/Machine Learning/Networks/vesta_optimizer.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/rng.h"

#include <sys/time.h>

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
	printf("[vesta] InitStateTest\n");
	const unsigned int m = 16, n = 12;
	// W = sum_k svs[k] * u_k v_k^T with orthonormal u_k and orthonormal v_k,
	// so the true singular values are exactly svs = (5, 3, 2).
	std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, 0x12345ULL);

	// Build U[m, 3] and V[n, 3] as orthonormal matrices via Gram-Schmidt.
	std::vector<float> Uorth(static_cast<size_t>(m) * 3u, 0.0f);
	std::vector<float> Vorth(static_cast<size_t>(n) * 3u, 0.0f);
	for (size_t i = 0; i < Uorth.size(); ++i) Uorth[i] = glades::rng::standard_normal(eng);
	for (size_t i = 0; i < Vorth.size(); ++i) Vorth[i] = glades::rng::standard_normal(eng);
	glades::vesta::gramSchmidt(&Uorth[0], m, 3u);
	glades::vesta::gramSchmidt(&Vorth[0], n, 3u);

	const float svs[3] = { 5.0f, 3.0f, 2.0f };
	for (unsigned int k = 0; k < 3u; ++k)
		for (unsigned int i = 0; i < m; ++i)
			for (unsigned int j = 0; j < n; ++j)
				W[i * n + j] += svs[k] * Uorth[i * 3u + k] * Vorth[j * 3u + k];

	glades::VestaConfig vc;
	vc.rank = 4u;
	glades::vesta::WeightState st;
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 0xABCULL);
	glades::vesta::initWeightState(st, &W[0], m, n, vc, rng, 0);

	ASSERT("init: initialized flag", st.initialized);
	ASSERT("init: m", st.m == m);
	ASSERT("init: n", st.n == n);
	ASSERT("init: r", st.r == 4u);
	ASSERT("init: step", st.step == 0ULL);
	ASSERT("init: ell size", st.ell.size() == 4u);
	ASSERT("init: U size", st.U.size() == static_cast<size_t>(m) * 4u);

	for (int k = 0; k < 3; ++k)
	{
		const float got = expf(st.ell[k]);
		const float expected = svs[k];
		const float rel = fabsf(got - expected) / expected;
		char msg[128];
		sprintf(msg, "init: exp(ell[%d])=%.4f expected %.2f rel %.4f", k, got, expected, rel);
		ASSERT(msg, rel < 0.05f);
	}

	const float uErr = orth_error(&st.U[0], m, st.r);
	assert_close("init: U orth err", uErr, 0.0f, 1e-3f);
	const float vErr = orth_error(&st.V[0], n, st.r);
	assert_close("init: V orth err", vErr, 0.0f, 1e-3f);

	for (unsigned int i = 0; i < st.r; ++i)
	{
		assert_close("init: beta[i]==ell[i]", st.beta[i], st.ell[i], 1e-7f);
		assert_close("init: ellStar[i]==ell[i]", st.ellStar[i], st.ell[i], 1e-7f);
	}
}

// Verify applyStep reduces ||W - W_target||_F^2 over a few steps on a
// simple quadratic loss: L = 0.5 * ||W - W_target||_F^2, grad = W - W_target.
void VESTALogScaleUpdateTest()
{
	printf("[vesta] LogScaleUpdateTest\n");
	const unsigned int m = 16, n = 12;
	std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
	std::vector<float> Wtarget(static_cast<size_t>(m) * n, 0.0f);
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, 0x33ULL);
	for (size_t i = 0; i < W.size(); ++i)
	{
		W[i] = glades::rng::standard_normal(eng);
		Wtarget[i] = glades::rng::standard_normal(eng);
	}

	glades::VestaConfig vc;
	vc.rank = 4u;
	vc.tau = 0.0f;
	vc.lambdaPerp = 0.2f;
	vc.rho = 0.5f;
	vc.tSk = 1u;

	glades::vesta::WeightState st;
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 0xBULL);

	float prevLoss = 0.0f;
	for (size_t i = 0; i < W.size(); ++i)
		prevLoss += (W[i] - Wtarget[i]) * (W[i] - Wtarget[i]);
	prevLoss *= 0.5f;

	const unsigned int steps = 5u;
	float lastLoss = prevLoss;
	for (unsigned int s = 0; s < steps; ++s)
	{
		std::vector<float> g(W.size(), 0.0f);
		for (size_t i = 0; i < W.size(); ++i)
			g[i] = W[i] - Wtarget[i];

		const bool ok = glades::vesta::update(st, &W[0], &g[0], m, n,
		                                      1.0f, 0.05f, 0.0f, 0.0f, 1.0f,
		                                      vc, rng, 0, 0);
		ASSERT("applyStep non-finite", ok);

		float loss = 0.0f;
		for (size_t i = 0; i < W.size(); ++i)
			loss += (W[i] - Wtarget[i]) * (W[i] - Wtarget[i]);
		loss *= 0.5f;
		lastLoss = loss;
	}

	printf("  init loss=%.4f final loss=%.4f ratio=%.4f\n",
	       prevLoss, lastLoss, lastLoss / prevLoss);
	ASSERT("loss did not decrease", lastLoss < 0.9f * prevLoss);
}

void VESTATrustRegionClampTest()
{
	printf("[vesta] TrustRegionClampTest\n");
	const unsigned int m = 12, n = 10;
	std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
	for (unsigned int i = 0; i < m && i < n; ++i)
		W[i * n + i] = 1.0f - 0.1f * static_cast<float>(i);

	glades::VestaConfig vc;
	vc.rank = 3u;
	vc.tau = 0.0f;
	vc.rho = 0.05f;       // tight trust region
	vc.lambdaPerp = 0.0f;
	vc.tSk = 1000u;        // no refresh during this test
	vc.tHom = 1000000u;    // no homeostasis
	vc.gamma = 0.0f;       // disable momentum for predictable behavior
	vc.kappa = 0.0f;

	glades::vesta::WeightState st;
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 0xBEEFULL);
	glades::vesta::initWeightState(st, &W[0], m, n, vc, rng, 0);

	const float initMaxSigma = expf(st.ell[0]);

	std::vector<float> g(W.size(), 0.0f);
	for (unsigned int s = 0; s < 50u; ++s)
	{
		// Gradient that drives ell[0] upward: g = -U[:,0] V[:,0]^T → A[0,0] = -1
		for (unsigned int i = 0; i < m; ++i)
			for (unsigned int j = 0; j < n; ++j)
				g[i * n + j] = -st.U[i * st.r + 0] * st.V[j * st.r + 0];

		const bool ok = glades::vesta::applyStep(st, &W[0], &g[0], m, n,
		                                         1.0f, 0.1f, 0.0f, 0.0f, 1.0f,
		                                         vc, rng, 0, 0);
		ASSERT("applyStep non-finite", ok);
	}

	const float finalMaxSigma = expf(st.ell[0]);
	// With rho=0.05 per step, after 50 steps max growth is (1.05)^50 ≈ 11.47x.
	ASSERT("trust region allowed unbounded growth",
	       finalMaxSigma < 15.0f * initMaxSigma);
}

void VESTAStepDescentTest()
{
	printf("[vesta] StepDescentTest (longer horizon)\n");
	const unsigned int m = 24, n = 20;
	std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
	std::vector<float> Wtarget(static_cast<size_t>(m) * n, 0.0f);
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, 0x424242ULL);
	for (size_t i = 0; i < W.size(); ++i)
	{
		W[i] = 0.3f * glades::rng::standard_normal(eng);
		Wtarget[i] = 0.5f * glades::rng::standard_normal(eng);
	}

	glades::VestaConfig vc;
	vc.rank = 6u;
	vc.tau = 0.0f;
	vc.lambdaPerp = 0.2f;
	vc.rho = 0.1f;
	vc.tSk = 4u;

	glades::vesta::WeightState st;
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 0xABCULL);

	float prevLoss = 0.0f;
	for (size_t i = 0; i < W.size(); ++i)
		prevLoss += (W[i] - Wtarget[i]) * (W[i] - Wtarget[i]);
	prevLoss *= 0.5f;

	const unsigned int steps = 50u;
	float lastLoss = prevLoss;
	for (unsigned int s = 0; s < steps; ++s)
	{
		std::vector<float> g(W.size(), 0.0f);
		for (size_t i = 0; i < W.size(); ++i)
			g[i] = W[i] - Wtarget[i];
		const bool ok = glades::vesta::update(st, &W[0], &g[0], m, n,
		                                      1.0f, 0.1f, 0.0f, 0.0f, 1.0f,
		                                      vc, rng, 0, 0);
		ASSERT("non-finite during long horizon", ok);
		float loss = 0.0f;
		for (size_t i = 0; i < W.size(); ++i)
			loss += (W[i] - Wtarget[i]) * (W[i] - Wtarget[i]);
		loss *= 0.5f;
		lastLoss = loss;
	}

	printf("  init loss=%.4f final loss=%.4f ratio=%.4f\n",
	       prevLoss, lastLoss, lastLoss / prevLoss);
	ASSERT("50-step ratio", lastLoss < 0.5f * prevLoss);
}

// Verify that the singular-value spectrum is invariant under left/right
// orthogonal transforms of W and g (same rng seed used on both sides).
// Exact weight-level equivariance requires a rotationally equivariant
// sketch Omega, which we do not enforce; we therefore check the spectrum,
// which is intrinsic.
void VESTAOrthogonalInvarianceTest()
{
	printf("[vesta] OrthogonalInvarianceTest\n");
	const unsigned int m = 8, n = 6, r = 3;

	std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
	std::vector<float> g(static_cast<size_t>(m) * n, 0.0f);
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, 0x99ULL);
	for (size_t i = 0; i < W.size(); ++i)
	{
		W[i] = glades::rng::standard_normal(eng);
		g[i] = glades::rng::standard_normal(eng);
	}

	std::vector<float> P(static_cast<size_t>(m) * m, 0.0f);
	std::vector<float> Q(static_cast<size_t>(n) * n, 0.0f);
	for (size_t i = 0; i < P.size(); ++i) P[i] = glades::rng::standard_normal(eng);
	for (size_t i = 0; i < Q.size(); ++i) Q[i] = glades::rng::standard_normal(eng);
	glades::vesta::gramSchmidt(&P[0], m, m);
	glades::vesta::gramSchmidt(&Q[0], n, n);

	// W2 = P W Q^T, g2 = P g Q^T.
	std::vector<float> W2(W.size(), 0.0f), g2(g.size(), 0.0f);
	std::vector<float> tmp(W.size(), 0.0f);
	// tmp = P * W
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int j = 0; j < n; ++j)
		{
			float acc = 0.0f;
			for (unsigned int k = 0; k < m; ++k)
				acc += P[i * m + k] * W[k * n + j];
			tmp[i * n + j] = acc;
		}
	// W2 = tmp * Q^T: (tmp Q^T)[i,j] = sum_k tmp[i,k] * Q[j,k]
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int j = 0; j < n; ++j)
		{
			float acc = 0.0f;
			for (unsigned int k = 0; k < n; ++k)
				acc += tmp[i * n + k] * Q[j * n + k];
			W2[i * n + j] = acc;
		}
	// g2 similarly.
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int j = 0; j < n; ++j)
		{
			float acc = 0.0f;
			for (unsigned int k = 0; k < m; ++k)
				acc += P[i * m + k] * g[k * n + j];
			tmp[i * n + j] = acc;
		}
	for (unsigned int i = 0; i < m; ++i)
		for (unsigned int j = 0; j < n; ++j)
		{
			float acc = 0.0f;
			for (unsigned int k = 0; k < n; ++k)
				acc += tmp[i * n + k] * Q[j * n + k];
			g2[i * n + j] = acc;
		}

	glades::VestaConfig vc;
	vc.rank = r;
	vc.tau = 0.0f;
	vc.rho = 0.5f;
	vc.lambdaPerp = 0.0f;
	vc.tSk = 1u;

	glades::vesta::WeightState st1, st2;
	glades::rng::Engine rng1, rng2;
	glades::rng::seed_engine(rng1, 0xFEEDULL);
	glades::rng::seed_engine(rng2, 0xFEEDULL);
	glades::vesta::initWeightState(st1, &W[0], m, n, vc, rng1, 0);
	glades::vesta::initWeightState(st2, &W2[0], m, n, vc, rng2, 0);

	std::vector<float> g1 = g, g2b = g2;
	(void)glades::vesta::applyStep(st1, &W[0], &g1[0], m, n, 1.0f, 0.05f,
	                               0.0f, 0.0f, 1.0f, vc, rng1, 0, 0);
	(void)glades::vesta::applyStep(st2, &W2[0], &g2b[0], m, n, 1.0f, 0.05f,
	                               0.0f, 0.0f, 1.0f, vc, rng2, 0, 0);

	for (unsigned int i = 0; i < r; ++i)
	{
		const float delta = fabsf(st1.ell[i] - st2.ell[i]);
		char msg[128];
		sprintf(msg, "orth invariance ell[%u] delta %.6g", i, delta);
		ASSERT(msg, delta < 5e-3f);
	}
}

#ifdef GLADES_HAVE_CUDA

void VESTAGpuParityTest()
{
	printf("[vesta] GpuParityTest\n");

	if (!glades::gpu::isAvailable())
	{
		if (!glades::gpu::initDevice(0))
		{
			printf("  GPU unavailable; skipping parity test\n");
			return;
		}
	}

	const unsigned int m = 32, n = 24;
	std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, 0xC0DEULL);
	for (size_t i = 0; i < W.size(); ++i)
		W[i] = 0.5f * glades::rng::standard_normal(eng);

	glades::VestaConfig vc;
	vc.rank = 5u;
	vc.tau = 0.1f;
	vc.rho = 0.05f;
	vc.lambdaPerp = 0.2f;
	vc.tSk = 5u;

	std::vector<float> Wcpu = W;
	glades::vesta::WeightState stCpu;
	glades::rng::Engine rngCpu;
	glades::rng::seed_engine(rngCpu, 0x555ULL);
	glades::vesta::initWeightState(stCpu, &Wcpu[0], m, n, vc, rngCpu, 0);

	glades::gpu::GpuBuffer<float> dW, dG;
	ASSERT("alloc dW", dW.allocate(static_cast<size_t>(m) * n));
	ASSERT("alloc dG", dG.allocate(static_cast<size_t>(m) * n));
	ASSERT("upload W", dW.upload(&W[0], static_cast<size_t>(m) * n));

	glades::gpu::GpuVestaWeightState stGpu;
	glades::rng::Engine rngGpu;
	glades::rng::seed_engine(rngGpu, 0x555ULL);
	ASSERT("gpu init",
	       glades::gpu::vesta_gpu_init(stGpu, dW.data(), m, n, vc, rngGpu, 0));

	const unsigned int steps = 10u;
	for (unsigned int s = 0; s < steps; ++s)
	{
		glades::rng::Engine gradEng;
		glades::rng::seed_engine(gradEng, 0xAA00ULL + s);
		std::vector<float> gStep(static_cast<size_t>(m) * n, 0.0f);
		for (size_t i = 0; i < gStep.size(); ++i)
			gStep[i] = 0.05f * glades::rng::standard_normal(gradEng);

		std::vector<float> gCpuStep = gStep;
		std::vector<float> gGpuStep = gStep;

		const bool okCpu = glades::vesta::applyStep(stCpu, &Wcpu[0], &gCpuStep[0], m, n,
		                                            1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                            vc, rngCpu, 0, 0);
		ASSERT("cpu step", okCpu);

		ASSERT("upload g", dG.upload(&gGpuStep[0], static_cast<size_t>(m) * n));
		const bool okGpu = glades::gpu::vesta_gpu_step(stGpu, dW.data(), dG.data(), m, n,
		                                               1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                               vc, rngGpu, 0, 0);
		ASSERT("gpu step", okGpu);
	}

	std::vector<float> Wgpu(static_cast<size_t>(m) * n, 0.0f);
	ASSERT("download Wgpu", dW.download(&Wgpu[0], static_cast<size_t>(m) * n));
	float maxAbs = 0.0f, meanAbs = 0.0f;
	for (size_t i = 0; i < Wcpu.size(); ++i)
	{
		const float d = fabsf(Wcpu[i] - Wgpu[i]);
		if (d > maxAbs) maxAbs = d;
		meanAbs += d;
	}
	meanAbs /= static_cast<float>(Wcpu.size());
	printf("  parity W maxAbs=%.6g meanAbs=%.6g\n", maxAbs, meanAbs);
	ASSERT("parity W maxAbs", maxAbs < 5e-3f);
	ASSERT("parity W meanAbs", meanAbs < 5e-4f);

	std::vector<float> ellGpu(stGpu.r, 0.0f);
	ASSERT("download ell", stGpu.ell.download(&ellGpu[0], stGpu.r));
	for (unsigned int i = 0; i < stGpu.r; ++i)
	{
		char msg[128];
		sprintf(msg, "parity ell[%u] cpu=%.6g gpu=%.6g", i, stCpu.ell[i], ellGpu[i]);
		ASSERT(msg, fabsf(stCpu.ell[i] - ellGpu[i]) < 1e-3f);
	}
}

#else

void VESTAGpuParityTest()
{
	printf("[vesta] GpuParityTest: CUDA not compiled; skipping\n");
}

#endif

// End-to-end: configure a tiny token-LM transformer with optimizer=VESTA and
// train it for a couple of epochs. Verifies the sgd_transformer.cpp dispatch
// reaches the VESTA branch and that a full train step produces finite weights.
void VESTATransformerIntegrationTest()
{
	printf("[vesta] TransformerIntegrationTest\n");
	const unsigned int vocab = 7u;

	InMemoryTokenIdInput di;
	{
		std::vector<unsigned int> toks;
		for (unsigned int i = 0; i < 32u; ++i)
			toks.push_back((i * 3u + 1u) % vocab);
		di.setTrainTokens(toks, -1);
		di.mirrorTrainToTest();
	}

	glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
	std::vector<glades::HiddenLayerInfo*> hidden;
	hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
	glades::NNInfo* info = new glades::NNInfo("vesta_transformer_integration", in, hidden, out);

	glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
	net.getTerminatorMutable().setEpoch(1);
	net.getTerminatorMutable().setAccuracy(0);
	{
		glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
		cfg.transformer.enableTokenEmbedding = true;
		cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
		cfg.transformer.tieEmbeddings = true;
		cfg.transformer.nHeadsOverride = 2;
		cfg.transformer.dFFOverride = 32;
		cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_NONE;
		cfg.optimizer.type = glades::OptimizerConfig::VESTA;
		cfg.vesta.rank = 4u;
		cfg.vesta.tau = 0.0f;
		cfg.vesta.rho = 0.1f;
		cfg.vesta.tSk = 100u; // avoid sketched-SVD refresh in a single-epoch test
	}

	ASSERT("VESTA transformer: initial test", net.test(&di).ok());
	const glades::NNetworkStatus st = net.train(&di);
	ASSERT("VESTA transformer: train status", st.ok());
	ASSERT("VESTA transformer: post-train test", net.test(&di).ok());

	delete info;
}

// =================================================================
// Sweep benchmark: VESTA vs AdamW vs ATLAS on a tiny token-LM.
// =================================================================

namespace {

struct SweepResult
{
	float finalTrainNll;
	float finalTrainPpl;
	float finalTestNll;
	float finalTestPpl;
	double wallSec;
	bool ok;
	SweepResult() : finalTrainNll(0.0f), finalTrainPpl(0.0f),
	                finalTestNll(0.0f), finalTestPpl(0.0f),
	                wallSec(0.0), ok(false) {}
};

class MetricCapture : public glades::ITrainingCallbacks
{
public:
	MetricCapture() : saw(false), last() {}
	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}
	bool saw;
	glades::NNetworkEpochMetrics last;
};

static double wall_ms()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return static_cast<double>(tv.tv_sec) * 1000.0 + static_cast<double>(tv.tv_usec) / 1000.0;
}

static void build_token_corpus(std::vector<unsigned int>& toks, unsigned int vocab, unsigned int length, uint64_t seed)
{
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, seed);
	toks.clear();
	toks.reserve(length);
	// Structured pattern + noise — non-trivial to memorize but learnable.
	for (unsigned int i = 0; i < length; ++i)
	{
		unsigned int t;
		if ((i % 7u) == 0u)
			t = glades::rng::uniform_uint(eng, 0u, vocab - 1u);
		else
			t = static_cast<unsigned int>((i * 3u + (i / 7u)) % vocab);
		toks.push_back(t);
	}
}

static SweepResult run_one(glades::OptimizerConfig::Type optType,
                           unsigned int seed,
                           unsigned int vocab,
                           unsigned int dModel,
                           unsigned int dFF,
                           unsigned int nLayers,
                           unsigned int nHeads,
                           unsigned int epochs,
                           unsigned int corpusLen,
                           const char* label)
{
	SweepResult res;

	std::vector<unsigned int> trainToks;
	build_token_corpus(trainToks, vocab, corpusLen, 0x5EEDULL + seed);
	std::vector<unsigned int> testToks;
	build_token_corpus(testToks, vocab, corpusLen, 0x7357ULL + seed);

	InMemoryTokenIdInput di;
	di.setTrainTokens(trainToks, -1);
	di.setTestTokens(testToks, -1);

	glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
	std::vector<glades::HiddenLayerInfo*> hidden;
	for (unsigned int i = 0; i < nLayers; ++i)
		hidden.push_back(new glades::HiddenLayerInfo(
		    static_cast<int>(dModel), 0.001f, 0.0f, 0.0f, 0.0f, 0.0f,
		    glades::GMath::LINEAR, 1.0f));
	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
	    static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
	glades::NNInfo* info = new glades::NNInfo("vesta_sweep", in, hidden, out);

	glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
	net.setSeed(seed);
	net.getTerminatorMutable().setEpoch(static_cast<int>(epochs));
	net.getTerminatorMutable().setAccuracy(0.0f);
	{
		glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
		cfg.transformer.enableTokenEmbedding = true;
		cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
		cfg.transformer.tieEmbeddings = true;
		cfg.transformer.nHeadsOverride = static_cast<int>(nHeads);
		cfg.transformer.dFFOverride = static_cast<int>(dFF);
		cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_NONE;
		cfg.optimizer.type = optType;
		cfg.optimizer.adamBeta1 = 0.9f;
		cfg.optimizer.adamBeta2 = 0.999f;
		cfg.optimizer.adamEps = 1e-8f;
		cfg.optimizer.adamBiasCorrection = true;
		cfg.atlas.rank = 4u;
		cfg.vesta.rank = 4u;
		cfg.vesta.tau = 0.1f;
		cfg.vesta.rho = 0.1f;
		cfg.vesta.tSk = 8u;
		cfg.vesta.lambdaPerp = 0.2f;
	}

	MetricCapture trainCb;
	const double t0 = wall_ms();
	const glades::NNetworkStatus stTrain = net.train(&di, &trainCb);
	const double t1 = wall_ms();
	res.wallSec = (t1 - t0) / 1000.0;
	if (!stTrain.ok() || !trainCb.saw)
	{
		printf("  [sweep:%s seed=%u] TRAIN FAILED: %s\n",
		       label, seed, stTrain.message.c_str());
		delete info;
		return res;
	}
	res.finalTrainNll = trainCb.last.totalError;
	res.finalTrainPpl = trainCb.last.perplexity;

	MetricCapture testCb;
	const glades::NNetworkStatus stTest = net.test(&di, &testCb);
	if (!stTest.ok() || !testCb.saw)
	{
		printf("  [sweep:%s seed=%u] TEST FAILED\n", label, seed);
		delete info;
		return res;
	}
	res.finalTestNll = testCb.last.totalError;
	res.finalTestPpl = testCb.last.perplexity;
	res.ok = true;

	delete info;
	return res;
}

struct AggStats
{
	float mean;
	float stddev;
	AggStats() : mean(0.0f), stddev(0.0f) {}
};

static AggStats aggregate(const std::vector<float>& xs)
{
	AggStats a;
	if (xs.empty()) return a;
	float sum = 0.0f;
	for (size_t i = 0; i < xs.size(); ++i) sum += xs[i];
	a.mean = sum / static_cast<float>(xs.size());
	float sq = 0.0f;
	for (size_t i = 0; i < xs.size(); ++i) sq += (xs[i] - a.mean) * (xs[i] - a.mean);
	a.stddev = (xs.size() > 1u) ? sqrtf(sq / static_cast<float>(xs.size() - 1u)) : 0.0f;
	return a;
}

} // namespace

void VESTASweepBenchmark()
{
	printf("\n============================================================\n");
	printf("VESTA sweep: VESTA vs AdamW vs ATLAS on tiny token-LM\n");
	printf("============================================================\n");

	const unsigned int vocab = 29u;
	const unsigned int dModel = 64u;
	const unsigned int dFF = 128u;
	const unsigned int nLayers = 3u;
	const unsigned int nHeads = 4u;
	const unsigned int epochs = 30u;
	const unsigned int corpusLen = 256u;
	const unsigned int seeds[] = { 101u, 202u, 303u, 404u, 505u };
	const unsigned int nSeeds = sizeof(seeds) / sizeof(seeds[0]);

	printf("Config: vocab=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u seq=%u seeds=%u\n",
	       vocab, dModel, dFF, nLayers, nHeads, epochs, corpusLen, nSeeds);

	struct OptSpec
	{
		glades::OptimizerConfig::Type type;
		const char* label;
	};
	OptSpec specs[3];
	specs[0].type = glades::OptimizerConfig::ADAMW; specs[0].label = "AdamW";
	specs[1].type = glades::OptimizerConfig::ATLAS; specs[1].label = "ATLAS";
	specs[2].type = glades::OptimizerConfig::VESTA; specs[2].label = "VESTA";

	printf("\n%-8s  %-6s  %-10s  %-10s  %-10s  %-10s  %-8s\n",
	       "opt", "seed", "trainNLL", "trainPPL", "testNLL", "testPPL", "wall(s)");
	printf("%-8s  %-6s  %-10s  %-10s  %-10s  %-10s  %-8s\n",
	       "---", "---", "----------", "----------", "----------", "----------", "-------");

	std::vector<std::vector<float> > testNlls(3);
	std::vector<std::vector<float> > testPpls(3);
	std::vector<std::vector<float> > trainNlls(3);
	std::vector<std::vector<float> > walls(3);
	std::vector<unsigned int> okCounts(3, 0u);

	for (unsigned int o = 0; o < 3u; ++o)
	{
		for (unsigned int s = 0; s < nSeeds; ++s)
		{
			const SweepResult r = run_one(specs[o].type, seeds[s],
			                              vocab, dModel, dFF, nLayers, nHeads,
			                              epochs, corpusLen, specs[o].label);
			printf("%-8s  %-6u  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-8.2f%s\n",
			       specs[o].label, seeds[s],
			       r.finalTrainNll, r.finalTrainPpl,
			       r.finalTestNll, r.finalTestPpl, r.wallSec,
			       r.ok ? "" : "  [FAIL]");
			if (r.ok)
			{
				testNlls[o].push_back(r.finalTestNll);
				testPpls[o].push_back(r.finalTestPpl);
				trainNlls[o].push_back(r.finalTrainNll);
				walls[o].push_back(static_cast<float>(r.wallSec));
				okCounts[o]++;
			}
		}
	}

	printf("\nSummary (mean +/- stddev across seeds):\n");
	printf("%-8s  %-4s  %-18s  %-18s  %-18s  %-12s\n",
	       "opt", "n", "trainNLL", "testNLL", "testPPL", "wall(s)");
	printf("%-8s  %-4s  %-18s  %-18s  %-18s  %-12s\n",
	       "---", "---", "------------------", "------------------",
	       "------------------", "------------");
	for (unsigned int o = 0; o < 3u; ++o)
	{
		const AggStats tNll = aggregate(trainNlls[o]);
		const AggStats vNll = aggregate(testNlls[o]);
		const AggStats vPpl = aggregate(testPpls[o]);
		const AggStats wAgg = aggregate(walls[o]);
		printf("%-8s  %-4u  %6.4f +/- %-7.4f  %6.4f +/- %-7.4f  %6.4f +/- %-7.4f  %5.2f +/- %-5.2f\n",
		       specs[o].label, okCounts[o],
		       tNll.mean, tNll.stddev,
		       vNll.mean, vNll.stddev,
		       vPpl.mean, vPpl.stddev,
		       wAgg.mean, wAgg.stddev);
	}
	printf("\n");
}

void VESTAUnitTest()
{
	VESTAGramSchmidtTest();
	VESTAThinQRTest();
	VESTASketchedSVDTest();
	VESTAInitStateTest();
	VESTALogScaleUpdateTest();
	VESTATrustRegionClampTest();
	VESTAOrthogonalInvarianceTest();
	VESTAStepDescentTest();
	VESTAGpuParityTest();
	VESTATransformerIntegrationTest();
}
