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

struct RunSpec
{
	glades::OptimizerConfig::Type optType;
	const char* label;
	unsigned int vocab;
	unsigned int dModel;
	unsigned int dFF;
	unsigned int nLayers;
	unsigned int nHeads;
	unsigned int epochs;
	unsigned int corpusLen;
	float learningRate;
	unsigned int atlasRank;
	unsigned int vestaRank;
	float vestaTau;
	float vestaRho;
	unsigned int vestaTSk;
	float vestaLambdaPerp;
	bool vestaComplementMomentum;
	float vestaComplementBeta;
	bool vestaTrackedEma;
	float vestaTrackedEmaBeta;
	unsigned int vestaBasisSource; // 0=weights, 1=gradient
	float vestaBasisEmaBeta;

	RunSpec()
	    : optType(glades::OptimizerConfig::ADAMW), label("AdamW"),
	      vocab(29u), dModel(64u), dFF(128u), nLayers(3u), nHeads(4u),
	      epochs(30u), corpusLen(256u),
	      learningRate(0.001f),
	      atlasRank(4u),
	      vestaRank(8u), vestaTau(0.1f), vestaRho(0.1f),
	      vestaTSk(16u), vestaLambdaPerp(0.2f),
	      vestaComplementMomentum(false), vestaComplementBeta(0.9f),
	      vestaTrackedEma(false), vestaTrackedEmaBeta(0.9f),
	      vestaBasisSource(0u), vestaBasisEmaBeta(0.99f)
	{
	}
};

static SweepResult run_one(const RunSpec& spec, unsigned int seed)
{
	SweepResult res;

	std::vector<unsigned int> trainToks;
	build_token_corpus(trainToks, spec.vocab, spec.corpusLen, 0x5EEDULL + seed);
	std::vector<unsigned int> testToks;
	build_token_corpus(testToks, spec.vocab, spec.corpusLen, 0x7357ULL + seed);

	InMemoryTokenIdInput di;
	di.setTrainTokens(trainToks, -1);
	di.setTestTokens(testToks, -1);

	glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
	std::vector<glades::HiddenLayerInfo*> hidden;
	for (unsigned int i = 0; i < spec.nLayers; ++i)
		hidden.push_back(new glades::HiddenLayerInfo(
		    static_cast<int>(spec.dModel), spec.learningRate, 0.0f, 0.0f, 0.0f, 0.0f,
		    glades::GMath::LINEAR, 1.0f));
	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
	    static_cast<int>(spec.vocab), glades::OutputLayerInfo::CLASSIFICATION);
	glades::NNInfo* info = new glades::NNInfo("vesta_sweep", in, hidden, out);

	glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
	net.setSeed(seed);
	net.getTerminatorMutable().setEpoch(static_cast<int>(spec.epochs));
	net.getTerminatorMutable().setAccuracy(0.0f);
	{
		glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
		cfg.transformer.enableTokenEmbedding = true;
		cfg.transformer.vocabSizeOverride = static_cast<int>(spec.vocab);
		cfg.transformer.tieEmbeddings = true;
		cfg.transformer.nHeadsOverride = static_cast<int>(spec.nHeads);
		cfg.transformer.dFFOverride = static_cast<int>(spec.dFF);
		cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_NONE;
		cfg.optimizer.type = spec.optType;
		cfg.optimizer.adamBeta1 = 0.9f;
		cfg.optimizer.adamBeta2 = 0.999f;
		cfg.optimizer.adamEps = 1e-8f;
		cfg.optimizer.adamBiasCorrection = true;
		cfg.atlas.rank = spec.atlasRank;
		cfg.vesta.rank = spec.vestaRank;
		cfg.vesta.tau = spec.vestaTau;
		cfg.vesta.rho = spec.vestaRho;
		cfg.vesta.tSk = spec.vestaTSk;
		cfg.vesta.lambdaPerp = spec.vestaLambdaPerp;
		cfg.vesta.complementMomentumEnabled = spec.vestaComplementMomentum;
		cfg.vesta.complementBeta = spec.vestaComplementBeta;
		cfg.vesta.trackedEmaEnabled = spec.vestaTrackedEma;
		cfg.vesta.trackedEmaBeta = spec.vestaTrackedEmaBeta;
		cfg.vesta.basisSource = spec.vestaBasisSource;
		cfg.vesta.basisEmaBeta = spec.vestaBasisEmaBeta;
	}

	MetricCapture trainCb;
	const double t0 = wall_ms();
	const glades::NNetworkStatus stTrain = net.train(&di, &trainCb);
	const double t1 = wall_ms();
	res.wallSec = (t1 - t0) / 1000.0;
	if (!stTrain.ok() || !trainCb.saw)
	{
		printf("  [sweep:%s seed=%u] TRAIN FAILED: %s\n",
		       spec.label, seed, stTrain.message.c_str());
		delete info;
		return res;
	}
	res.finalTrainNll = trainCb.last.totalError;
	res.finalTrainPpl = trainCb.last.perplexity;

	MetricCapture testCb;
	const glades::NNetworkStatus stTest = net.test(&di, &testCb);
	if (!stTest.ok() || !testCb.saw)
	{
		printf("  [sweep:%s seed=%u] TEST FAILED\n", spec.label, seed);
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
			RunSpec spec;
			spec.optType = specs[o].type;
			spec.label = specs[o].label;
			spec.vocab = vocab; spec.dModel = dModel; spec.dFF = dFF;
			spec.nLayers = nLayers; spec.nHeads = nHeads;
			spec.epochs = epochs; spec.corpusLen = corpusLen;
			spec.vestaRank = 4u; spec.atlasRank = 4u; spec.vestaTSk = 8u;
			const SweepResult r = run_one(spec, seeds[s]);
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

// =================================================================
// V2 sweep: scale-up config + per-optimizer LR sweep + VESTA HP sweep.
// Covers recommendations 2, 3, 4 from the first sweep artifact.
// =================================================================

namespace {

static void run_axis(const char* axisName,
                     const RunSpec& baseSpec,
                     const char* valueLabelFmt,
                     const float* values,
                     unsigned int nValues,
                     const unsigned int* seeds, unsigned int nSeeds,
                     std::vector<AggStats>& outTrain,
                     std::vector<AggStats>& outTest,
                     std::vector<AggStats>& outWall,
                     // optional mutator: applies value v to a copy of baseSpec
                     RunSpec (*mutate)(const RunSpec&, float))
{
	printf("\n%-10s  %-10s  %4s  %-18s  %-18s  %-12s\n",
	       "axis", axisName, "n", "trainNLL", "testNLL", "wall(s)");
	printf("%-10s  %-10s  %4s  %-18s  %-18s  %-12s\n",
	       "----", "---", "--", "------------------", "------------------", "------------");
	for (unsigned int v = 0; v < nValues; ++v)
	{
		RunSpec spec = mutate(baseSpec, values[v]);
		std::vector<float> trains, tests, walls;
		for (unsigned int s = 0; s < nSeeds; ++s)
		{
			const SweepResult r = run_one(spec, seeds[s]);
			if (r.ok)
			{
				trains.push_back(r.finalTrainNll);
				tests.push_back(r.finalTestNll);
				walls.push_back(static_cast<float>(r.wallSec));
			}
		}
		AggStats a = aggregate(trains);
		AggStats b = aggregate(tests);
		AggStats c = aggregate(walls);
		outTrain.push_back(a);
		outTest.push_back(b);
		outWall.push_back(c);
		char buf[32];
		sprintf(buf, valueLabelFmt, values[v]);
		printf("%-10s  %-10s  %4u  %6.4f +/- %-7.4f  %6.4f +/- %-7.4f  %5.2f +/- %-5.2f\n",
		       baseSpec.label, buf, (unsigned int)trains.size(),
		       a.mean, a.stddev, b.mean, b.stddev, c.mean, c.stddev);
	}
}

static RunSpec mutate_lr(const RunSpec& base, float v)          { RunSpec s = base; s.learningRate = v; return s; }
static RunSpec mutate_vrank(const RunSpec& base, float v)       { RunSpec s = base; s.vestaRank = static_cast<unsigned int>(v); return s; }
static RunSpec mutate_vtau(const RunSpec& base, float v)        { RunSpec s = base; s.vestaTau = v; return s; }
static RunSpec mutate_vtsk(const RunSpec& base, float v)        { RunSpec s = base; s.vestaTSk = static_cast<unsigned int>(v); return s; }
static RunSpec mutate_vlambda(const RunSpec& base, float v)     { RunSpec s = base; s.vestaLambdaPerp = v; return s; }

static unsigned int best_index(const std::vector<AggStats>& stats)
{
	if (stats.empty()) return 0u;
	unsigned int bi = 0u;
	for (unsigned int i = 1; i < stats.size(); ++i)
		if (stats[i].mean < stats[bi].mean) bi = i;
	return bi;
}

} // namespace

void VESTASweepV2Benchmark()
{
	printf("\n============================================================\n");
	printf("VESTA sweep v2: scale-up + per-opt LR sweep + VESTA HP sweep\n");
	printf("============================================================\n");

	// Scale-up config: ~4x cost of v1 sweep. Stays under ~15 min total
	// for the full LR x HP sweep.
	RunSpec base;
	base.vocab = 29u;
	base.dModel = 128u;
	base.dFF = 256u;
	base.nLayers = 4u;
	base.nHeads = 4u;
	base.epochs = 15u;
	base.corpusLen = 384u;
	base.atlasRank = 8u;
	base.vestaRank = 8u;
	base.vestaTau = 0.1f;
	base.vestaRho = 0.1f;
	base.vestaTSk = 16u;
	base.vestaLambdaPerp = 0.2f;

	const unsigned int seeds3[] = { 101u, 202u, 303u };
	const unsigned int nSeeds = sizeof(seeds3) / sizeof(seeds3[0]);

	printf("Scale-up base: vocab=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u seq=%u seeds=%u\n",
	       base.vocab, base.dModel, base.dFF, base.nLayers, base.nHeads,
	       base.epochs, base.corpusLen, nSeeds);

	// ============== Phase 1: LR sweep per optimizer ==============
	printf("\n================ Phase 1: learning-rate sweep ================\n");
	const float lrs[] = { 3e-4f, 1e-3f, 3e-3f, 1e-2f };
	const unsigned int nLrs = sizeof(lrs) / sizeof(lrs[0]);

	struct OptSpec { glades::OptimizerConfig::Type type; const char* label; };
	OptSpec specs[3];
	specs[0].type = glades::OptimizerConfig::ADAMW; specs[0].label = "AdamW";
	specs[1].type = glades::OptimizerConfig::ATLAS; specs[1].label = "ATLAS";
	specs[2].type = glades::OptimizerConfig::VESTA; specs[2].label = "VESTA";

	float bestLr[3] = { 1e-3f, 1e-3f, 1e-3f };
	AggStats bestTest[3];

	for (unsigned int o = 0; o < 3u; ++o)
	{
		RunSpec optBase = base;
		optBase.optType = specs[o].type;
		optBase.label = specs[o].label;
		std::vector<AggStats> trains, tests, walls;
		run_axis("lr", optBase, "%.1e", lrs, nLrs, seeds3, nSeeds,
		         trains, tests, walls, mutate_lr);
		const unsigned int bi = best_index(tests);
		bestLr[o] = lrs[bi];
		bestTest[o] = tests[bi];
	}

	printf("\nPhase 1 summary (best testNLL per optimizer):\n");
	printf("%-8s  %-10s  %-18s\n", "opt", "bestLR", "testNLL@bestLR");
	printf("%-8s  %-10s  %-18s\n", "---", "-------", "------------------");
	for (unsigned int o = 0; o < 3u; ++o)
		printf("%-8s  %.1e    %6.4f +/- %-7.4f\n",
		       specs[o].label, bestLr[o], bestTest[o].mean, bestTest[o].stddev);

	// ============== Phase 2: VESTA hyperparameter sweeps ==============
	printf("\n================ Phase 2: VESTA hyperparameter sweep ================\n");
	printf("Using VESTA best LR = %.1e from Phase 1\n", bestLr[2]);

	RunSpec vBase = base;
	vBase.optType = glades::OptimizerConfig::VESTA;
	vBase.label = "VESTA";
	vBase.learningRate = bestLr[2];

	std::vector<AggStats> vRankTrain, vRankTest, vRankWall;
	const float vRanks[] = { 4.0f, 8.0f, 16.0f };
	run_axis("rank", vBase, "r=%.0f", vRanks, 3u, seeds3, nSeeds,
	         vRankTrain, vRankTest, vRankWall, mutate_vrank);

	std::vector<AggStats> vTauTrain, vTauTest, vTauWall;
	const float vTaus[] = { 0.0f, 0.05f, 0.1f, 0.2f };
	run_axis("tau", vBase, "tau=%.2f", vTaus, 4u, seeds3, nSeeds,
	         vTauTrain, vTauTest, vTauWall, mutate_vtau);

	std::vector<AggStats> vTskTrain, vTskTest, vTskWall;
	const float vTsks[] = { 4.0f, 16.0f, 64.0f };
	run_axis("tSk", vBase, "tSk=%.0f", vTsks, 3u, seeds3, nSeeds,
	         vTskTrain, vTskTest, vTskWall, mutate_vtsk);

	std::vector<AggStats> vLamTrain, vLamTest, vLamWall;
	const float vLams[] = { 0.0f, 0.1f, 0.2f, 0.4f };
	run_axis("lambdaPerp", vBase, "lp=%.2f", vLams, 4u, seeds3, nSeeds,
	         vLamTrain, vLamTest, vLamWall, mutate_vlambda);

	const unsigned int biR = best_index(vRankTest);
	const unsigned int biT = best_index(vTauTest);
	const unsigned int biS = best_index(vTskTest);
	const unsigned int biL = best_index(vLamTest);

	printf("\nPhase 2 summary (VESTA best single-axis settings):\n");
	printf("  best rank       = %-4.0f  (testNLL %6.4f +/- %6.4f)\n", vRanks[biR], vRankTest[biR].mean, vRankTest[biR].stddev);
	printf("  best tau        = %-4.2f  (testNLL %6.4f +/- %6.4f)\n", vTaus[biT], vTauTest[biT].mean, vTauTest[biT].stddev);
	printf("  best tSk        = %-4.0f  (testNLL %6.4f +/- %6.4f)\n", vTsks[biS], vTskTest[biS].mean, vTskTest[biS].stddev);
	printf("  best lambdaPerp = %-4.2f  (testNLL %6.4f +/- %6.4f)\n", vLams[biL], vLamTest[biL].mean, vLamTest[biL].stddev);

	// ============== Phase 3: head-to-head at best configs ==============
	printf("\n================ Phase 3: head-to-head (best per optimizer) ================\n");

	const unsigned int seeds5[] = { 101u, 202u, 303u, 404u, 505u };
	const unsigned int nSeeds5 = sizeof(seeds5) / sizeof(seeds5[0]);

	printf("\n%-8s  %-10s  %-4s  %-18s  %-18s  %-12s\n",
	       "opt", "config", "n", "trainNLL", "testNLL", "wall(s)");
	printf("%-8s  %-10s  %-4s  %-18s  %-18s  %-12s\n",
	       "---", "------", "--", "------------------", "------------------", "------------");
	AggStats finalTest[3];
	for (unsigned int o = 0; o < 3u; ++o)
	{
		RunSpec spec = base;
		spec.optType = specs[o].type;
		spec.label = specs[o].label;
		spec.learningRate = bestLr[o];
		if (o == 2u)
		{
			spec.vestaRank = static_cast<unsigned int>(vRanks[biR]);
			spec.vestaTau = vTaus[biT];
			spec.vestaTSk = static_cast<unsigned int>(vTsks[biS]);
			spec.vestaLambdaPerp = vLams[biL];
		}
		std::vector<float> trains, tests, walls;
		for (unsigned int s = 0; s < nSeeds5; ++s)
		{
			const SweepResult r = run_one(spec, seeds5[s]);
			if (r.ok)
			{
				trains.push_back(r.finalTrainNll);
				tests.push_back(r.finalTestNll);
				walls.push_back(static_cast<float>(r.wallSec));
			}
		}
		const AggStats tA = aggregate(trains);
		const AggStats te = aggregate(tests);
		const AggStats wA = aggregate(walls);
		finalTest[o] = te;
		char cfgLabel[32];
		if (o == 2u)
			sprintf(cfgLabel, "r%u/t%.2f", spec.vestaRank, spec.vestaTau);
		else if (o == 1u)
			sprintf(cfgLabel, "atlas-r%u", spec.atlasRank);
		else
			sprintf(cfgLabel, "adamw");
		printf("%-8s  %-10s  %-4u  %6.4f +/- %-7.4f  %6.4f +/- %-7.4f  %5.2f +/- %-5.2f\n",
		       specs[o].label, cfgLabel, (unsigned int)trains.size(),
		       tA.mean, tA.stddev, te.mean, te.stddev, wA.mean, wA.stddev);
	}

	printf("\nFinal head-to-head deltas (testNLL, lower is better):\n");
	for (unsigned int o = 0; o < 3u; ++o)
	{
		if (o == 0u) continue;
		const float d = finalTest[o].mean - finalTest[0].mean;
		printf("  %-8s vs AdamW: %+.4f nats  (%s)\n",
		       specs[o].label, d, d < 0.0f ? "wins" : "loses");
	}
	printf("\n");
}

// =================================================================
// Extended lambdaPerp sweep: investigate whether the monotone trend
// observed in v2 (lp 0.2 → 0.4) continues.
// =================================================================

namespace {

// Runs a fixed NOISY quadratic descent with the given VESTA config and returns
// the final 0.5*||W - Wstar||_F^2. Noise is injected into the gradient at each
// step to simulate mini-batch stochasticity — the regime where momentum
// actually helps. Seed controls both VESTA internals and the noise stream.
static float run_noisy_quadratic_vesta(const std::vector<float>& Wstar,
                                       const std::vector<float>& W0,
                                       unsigned int m, unsigned int n,
                                       const glades::VestaConfig& vc,
                                       unsigned int steps,
                                       float lr,
                                       float noiseSigma,
                                       uint64_t rngSeed,
                                       uint64_t noiseSeed)
{
	std::vector<float> W = W0;
	glades::vesta::WeightState st;
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, rngSeed);
	glades::rng::Engine noise;
	glades::rng::seed_engine(noise, noiseSeed);
	for (unsigned int s = 0; s < steps; ++s)
	{
		std::vector<float> g(W.size(), 0.0f);
		for (size_t i = 0; i < W.size(); ++i)
			g[i] = (W[i] - Wstar[i]) + noiseSigma * glades::rng::standard_normal(noise);
		const bool ok = glades::vesta::update(st, &W[0], &g[0], m, n,
		                                      1.0f, lr, 0.0f, 0.0f, 1.0f,
		                                      vc, rng, 0, 0);
		ASSERT("run_noisy_quadratic_vesta: non-finite", ok);
	}
	float loss = 0.0f;
	for (size_t i = 0; i < W.size(); ++i)
		loss += (W[i] - Wstar[i]) * (W[i] - Wstar[i]);
	return 0.5f * loss;
}

} // namespace

// TDD: EMA of tracked-subspace diagonal should reduce final loss on a noisy
// low-rank target where the useful signal lives in the tracked subspace.
void VESTATrackedEmaTest()
{
	printf("[vesta] TrackedEmaTest\n");
	const unsigned int m = 24, n = 18;
	const unsigned int rank = 4;
	// Build a rank-rank target so all signal lives inside the tracked subspace.
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, 0xFACEFEEDULL);
	std::vector<float> U0(static_cast<size_t>(m) * rank, 0.0f);
	std::vector<float> V0(static_cast<size_t>(n) * rank, 0.0f);
	for (size_t i = 0; i < U0.size(); ++i) U0[i] = glades::rng::standard_normal(eng);
	for (size_t i = 0; i < V0.size(); ++i) V0[i] = glades::rng::standard_normal(eng);
	glades::vesta::gramSchmidt(&U0[0], m, rank);
	glades::vesta::gramSchmidt(&V0[0], n, rank);
	const float svs[4] = { 3.0f, 2.0f, 1.0f, 0.5f };
	std::vector<float> Wstar(static_cast<size_t>(m) * n, 0.0f);
	for (unsigned int k = 0; k < rank; ++k)
		for (unsigned int i = 0; i < m; ++i)
			for (unsigned int j = 0; j < n; ++j)
				Wstar[i * n + j] += svs[k] * U0[i * rank + k] * V0[j * rank + k];

	std::vector<float> W0(Wstar.size(), 0.0f);
	for (size_t i = 0; i < W0.size(); ++i)
		W0[i] = 0.2f * glades::rng::standard_normal(eng);

	glades::VestaConfig vcNo;
	vcNo.rank = rank;
	vcNo.tau = 0.0f;
	vcNo.lambdaPerp = 0.0f; // isolate the tracked update entirely
	vcNo.rho = 0.5f;
	vcNo.tSk = 2u;          // aggressive refresh so U,V are near W's SVD
	vcNo.trackedEmaEnabled = false;

	glades::VestaConfig vcYes = vcNo;
	vcYes.trackedEmaEnabled = true;
	vcYes.trackedEmaBeta = 0.9f;

	// Average over noise draws: high per-step noise is where EMA helps.
	const unsigned int trials = 6u;
	const unsigned int steps = 80u;
	const float lr = 0.05f;
	const float noise = 0.8f;
	float sumNo = 0.0f, sumYes = 0.0f;
	for (unsigned int k = 0; k < trials; ++k)
	{
		sumNo  += run_noisy_quadratic_vesta(Wstar, W0, m, n, vcNo,  steps, lr, noise,
		                                    0xCAFE00ULL, 0xABCDEF00ULL + k);
		sumYes += run_noisy_quadratic_vesta(Wstar, W0, m, n, vcYes, steps, lr, noise,
		                                    0xCAFE00ULL, 0xABCDEF00ULL + k);
	}
	const float lossNo  = sumNo  / static_cast<float>(trials);
	const float lossYes = sumYes / static_cast<float>(trials);
	printf("  [no ema] mean final loss = %.4f\n", lossNo);
	printf("  [+ema]   mean final loss = %.4f\n", lossYes);
	ASSERT("tracked EMA should reduce loss on noisy low-rank target",
	       lossYes < lossNo);
}

// TDD: When the loss's natural descent direction is orthogonal to W's own
// SVD directions (small random W, big rank-r target Wstar), the gradient-driven
// basis should align U,V with the target and learn faster than the weight-driven
// basis which is tracking essentially noise directions.
//
// With complement disabled (lambdaPerp = 0) the tracked-subspace update is the
// ONLY learning path, so the comparison isolates which basis source wins.
void VESTAGradientBasisTest()
{
	printf("[vesta] GradientBasisTest\n");
	const unsigned int m = 24, n = 18;
	const unsigned int rank = 4;
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, 0xA11CE);

	// Wstar: clean rank-4 with strong singular values.
	std::vector<float> U0(static_cast<size_t>(m) * rank, 0.0f);
	std::vector<float> V0(static_cast<size_t>(n) * rank, 0.0f);
	for (size_t i = 0; i < U0.size(); ++i) U0[i] = glades::rng::standard_normal(eng);
	for (size_t i = 0; i < V0.size(); ++i) V0[i] = glades::rng::standard_normal(eng);
	glades::vesta::gramSchmidt(&U0[0], m, rank);
	glades::vesta::gramSchmidt(&V0[0], n, rank);
	const float svs[4] = { 4.0f, 3.0f, 2.0f, 1.0f };
	std::vector<float> Wstar(static_cast<size_t>(m) * n, 0.0f);
	for (unsigned int k = 0; k < rank; ++k)
		for (unsigned int i = 0; i < m; ++i)
			for (unsigned int j = 0; j < n; ++j)
				Wstar[i * n + j] += svs[k] * U0[i * rank + k] * V0[j * rank + k];

	// W0: tiny random, unrelated to Wstar.
	std::vector<float> W0(Wstar.size(), 0.0f);
	for (size_t i = 0; i < W0.size(); ++i)
		W0[i] = 0.05f * glades::rng::standard_normal(eng);

	glades::VestaConfig vcWt;
	vcWt.rank = rank;
	vcWt.tau = 0.0f;
	vcWt.lambdaPerp = 0.0f; // disable complement; isolate tracked update
	vcWt.rho = 0.5f;
	vcWt.tSk = 4u;
	vcWt.trackedEmaEnabled = false;
	vcWt.basisSource = 0u; // weight-driven

	glades::VestaConfig vcGd = vcWt;
	vcGd.basisSource = 1u; // gradient-driven
	vcGd.basisEmaBeta = 0.95f;

	const unsigned int steps = 60u;
	const float lr = 0.05f;
	const float noise = 0.0f;
	const float lossWt = run_noisy_quadratic_vesta(Wstar, W0, m, n, vcWt, steps, lr,
	                                               noise, 0xFEDC0DEULL, 0xDEAD01ULL);
	const float lossGd = run_noisy_quadratic_vesta(Wstar, W0, m, n, vcGd, steps, lr,
	                                               noise, 0xFEDC0DEULL, 0xDEAD01ULL);
	printf("  [weight-basis]   final loss = %.4f\n", lossWt);
	printf("  [gradient-basis] final loss = %.4f\n", lossGd);
	ASSERT("gradient-basis should beat weight-basis when W is unrelated to Wstar",
	       lossGd < lossWt);
}

// TDD: Lion-style complement momentum must actually accelerate descent.
void VESTAComplementMomentumTest()
{
	printf("[vesta] ComplementMomentumTest\n");
	const unsigned int m = 20, n = 16;
	std::vector<float> Wstar(static_cast<size_t>(m) * n, 0.0f);
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, 0xBADC0FFEEULL);
	for (size_t i = 0; i < Wstar.size(); ++i)
		Wstar[i] = glades::rng::standard_normal(eng);

	std::vector<float> W0(Wstar.size(), 0.0f);
	for (size_t i = 0; i < W0.size(); ++i)
		W0[i] = 0.2f * glades::rng::standard_normal(eng);

	glades::VestaConfig vcNo;
	vcNo.rank = 4u;
	vcNo.tau = 0.0f;
	vcNo.lambdaPerp = 0.4f;
	vcNo.rho = 0.5f;
	vcNo.tSk = 4u;
	vcNo.complementMomentumEnabled = false;

	glades::VestaConfig vcYes = vcNo;
	vcYes.complementMomentumEnabled = true;
	vcYes.complementBeta = 0.9f;

	// Average over several noise draws so the comparison isn't seed-dependent.
	const unsigned int trials = 5u;
	float sumNo = 0.0f, sumYes = 0.0f;
	for (unsigned int k = 0; k < trials; ++k)
	{
		sumNo  += run_noisy_quadratic_vesta(Wstar, W0, m, n, vcNo,  50u, 0.05f, 0.3f,
		                                    0xCAFEULL, 0xBEEFULL + k);
		sumYes += run_noisy_quadratic_vesta(Wstar, W0, m, n, vcYes, 50u, 0.05f, 0.3f,
		                                    0xCAFEULL, 0xBEEFULL + k);
	}
	const float lossNo  = sumNo  / static_cast<float>(trials);
	const float lossYes = sumYes / static_cast<float>(trials);
	printf("  [no mom] mean final loss over %u trials = %.4f\n", trials, lossNo);
	printf("  [+mom]   mean final loss over %u trials = %.4f\n", trials, lossYes);
	ASSERT("momentum should reduce loss on noisy quadratic", lossYes < lossNo);
}

void VESTASweepLambdaPerpExtended()
{
	printf("\n============================================================\n");
	printf("VESTA: extended lambdaPerp sweep (5 seeds)\n");
	printf("============================================================\n");

	RunSpec base;
	base.optType = glades::OptimizerConfig::VESTA;
	base.label = "VESTA";
	base.vocab = 29u;
	base.dModel = 128u;
	base.dFF = 256u;
	base.nLayers = 4u;
	base.nHeads = 4u;
	base.epochs = 15u;
	base.corpusLen = 384u;
	base.learningRate = 1e-2f;
	base.vestaRank = 8u;
	base.vestaTau = 0.1f;
	base.vestaRho = 0.1f;
	base.vestaTSk = 16u;

	const unsigned int seeds[] = { 101u, 202u, 303u, 404u, 505u };
	const unsigned int nSeeds = sizeof(seeds) / sizeof(seeds[0]);

	const float lps[] = { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.5f, 2.0f, 3.0f };
	const unsigned int nLps = sizeof(lps) / sizeof(lps[0]);

	printf("Base: dModel=%u layers=%u epochs=%u seq=%u LR=%.1e rank=%u tSk=%u\n\n",
	       base.dModel, base.nLayers, base.epochs, base.corpusLen,
	       base.learningRate, base.vestaRank, base.vestaTSk);

	printf("%-12s  %-4s  %-18s  %-18s  %-10s\n",
	       "lambdaPerp", "n", "trainNLL", "testNLL", "wall(s)");
	printf("%-12s  %-4s  %-18s  %-18s  %-10s\n",
	       "----------", "---", "------------------", "------------------", "----------");

	AggStats bestTest;
	bestTest.mean = 1e30f;
	float bestLp = 0.0f;

	for (unsigned int i = 0; i < nLps; ++i)
	{
		RunSpec s = base;
		s.vestaLambdaPerp = lps[i];
		std::vector<float> trains, tests, walls;
		for (unsigned int k = 0; k < nSeeds; ++k)
		{
			const SweepResult r = run_one(s, seeds[k]);
			if (r.ok)
			{
				trains.push_back(r.finalTrainNll);
				tests.push_back(r.finalTestNll);
				walls.push_back(static_cast<float>(r.wallSec));
			}
		}
		const AggStats tA = aggregate(trains);
		const AggStats te = aggregate(tests);
		const AggStats wA = aggregate(walls);
		printf("lp=%-10.2f  %-4u  %6.4f +/- %-7.4f  %6.4f +/- %-7.4f  %5.2f +/- %-5.2f\n",
		       lps[i], (unsigned int)trains.size(),
		       tA.mean, tA.stddev, te.mean, te.stddev, wA.mean, wA.stddev);
		if (te.mean < bestTest.mean)
		{
			bestTest = te;
			bestLp = lps[i];
		}
	}

	printf("\nBest lambdaPerp = %.2f  (testNLL %.4f +/- %.4f)\n",
	       bestLp, bestTest.mean, bestTest.stddev);
	printf("\n");
}

// =================================================================
// Momentum compare: VESTA-plain vs VESTA+Lion-style complement momentum
// swept over lambdaPerp. Final row: head-to-head vs AdamW at best VESTA.
// =================================================================

void VESTASweepMomentumCompare()
{
	printf("\n============================================================\n");
	printf("VESTA momentum comparison (5 seeds, lp x {no-mom, +mom})\n");
	printf("============================================================\n");

	RunSpec base;
	base.optType = glades::OptimizerConfig::VESTA;
	base.label = "VESTA";
	base.vocab = 29u;
	base.dModel = 128u;
	base.dFF = 256u;
	base.nLayers = 4u;
	base.nHeads = 4u;
	base.epochs = 15u;
	base.corpusLen = 384u;
	base.learningRate = 1e-2f;
	base.vestaRank = 8u;
	base.vestaTau = 0.1f;
	base.vestaRho = 0.1f;
	base.vestaTSk = 16u;

	const unsigned int seeds[] = { 101u, 202u, 303u, 404u, 505u };
	const unsigned int nSeeds = sizeof(seeds) / sizeof(seeds[0]);
	const float lps[] = { 0.2f, 0.4f, 0.6f, 0.8f, 1.0f };
	const unsigned int nLps = sizeof(lps) / sizeof(lps[0]);

	printf("Base: dModel=%u layers=%u epochs=%u seq=%u LR=%.1e\n\n",
	       base.dModel, base.nLayers, base.epochs, base.corpusLen,
	       base.learningRate);

	printf("%-8s  %-10s  %-4s  %-18s  %-18s  %-10s\n",
	       "mom", "lambdaPerp", "n", "trainNLL", "testNLL", "wall(s)");
	printf("%-8s  %-10s  %-4s  %-18s  %-18s  %-10s\n",
	       "---", "----------", "---", "------------------", "------------------", "----------");

	AggStats bestTestNo, bestTestYes;
	bestTestNo.mean = 1e30f; bestTestYes.mean = 1e30f;
	float bestLpNo = 0.0f, bestLpYes = 0.0f;

	for (unsigned int momOn = 0; momOn < 2u; ++momOn)
	{
		for (unsigned int i = 0; i < nLps; ++i)
		{
			RunSpec s = base;
			s.vestaLambdaPerp = lps[i];
			s.vestaComplementMomentum = (momOn == 1u);
			s.vestaComplementBeta = 0.9f;
			std::vector<float> trains, tests, walls;
			for (unsigned int k = 0; k < nSeeds; ++k)
			{
				const SweepResult r = run_one(s, seeds[k]);
				if (r.ok)
				{
					trains.push_back(r.finalTrainNll);
					tests.push_back(r.finalTestNll);
					walls.push_back(static_cast<float>(r.wallSec));
				}
			}
			const AggStats tA = aggregate(trains);
			const AggStats te = aggregate(tests);
			const AggStats wA = aggregate(walls);
			const char* tag = (momOn == 1u) ? "+mom" : "plain";
			printf("%-8s  lp=%-7.2f  %-4u  %6.4f +/- %-7.4f  %6.4f +/- %-7.4f  %5.2f +/- %-5.2f\n",
			       tag, lps[i], (unsigned int)trains.size(),
			       tA.mean, tA.stddev, te.mean, te.stddev, wA.mean, wA.stddev);

			if (momOn == 0u && te.mean < bestTestNo.mean)
			{
				bestTestNo = te; bestLpNo = lps[i];
			}
			if (momOn == 1u && te.mean < bestTestYes.mean)
			{
				bestTestYes = te; bestLpYes = lps[i];
			}
		}
	}

	printf("\nBest per branch:\n");
	printf("  plain  lambdaPerp = %.2f  testNLL = %.4f +/- %.4f\n",
	       bestLpNo, bestTestNo.mean, bestTestNo.stddev);
	printf("  +mom   lambdaPerp = %.2f  testNLL = %.4f +/- %.4f\n",
	       bestLpYes, bestTestYes.mean, bestTestYes.stddev);
	printf("  Delta (+mom - plain) = %+.4f nats\n",
	       bestTestYes.mean - bestTestNo.mean);

	// Final head-to-head: AdamW @ best-known vs VESTA+mom @ best lambdaPerp.
	printf("\nAdamW reference (LR=1e-2, same seeds, 5 runs):\n");
	RunSpec adam;
	adam.optType = glades::OptimizerConfig::ADAMW;
	adam.label = "AdamW";
	adam.vocab = base.vocab; adam.dModel = base.dModel; adam.dFF = base.dFF;
	adam.nLayers = base.nLayers; adam.nHeads = base.nHeads;
	adam.epochs = base.epochs; adam.corpusLen = base.corpusLen;
	adam.learningRate = 1e-2f;
	std::vector<float> adamTrain, adamTest, adamWall;
	for (unsigned int k = 0; k < nSeeds; ++k)
	{
		const SweepResult r = run_one(adam, seeds[k]);
		if (r.ok)
		{
			adamTrain.push_back(r.finalTrainNll);
			adamTest.push_back(r.finalTestNll);
			adamWall.push_back(static_cast<float>(r.wallSec));
		}
	}
	const AggStats adamT = aggregate(adamTest);
	const AggStats adamW = aggregate(adamWall);
	printf("  AdamW   testNLL = %.4f +/- %.4f   wall = %.2f s\n",
	       adamT.mean, adamT.stddev, adamW.mean);

	printf("\nFinal deltas vs AdamW (testNLL, lower is better):\n");
	printf("  VESTA plain  (lp=%.2f) : %+.4f nats\n",
	       bestLpNo,  bestTestNo.mean  - adamT.mean);
	printf("  VESTA +mom   (lp=%.2f) : %+.4f nats\n",
	       bestLpYes, bestTestYes.mean - adamT.mean);
	printf("\n");
}

// =================================================================
// Ablation: measure marginal contribution of each VESTA feature by
// toggling {momentum, tracked-EMA, gradient-basis} independently at
// the current best config.
// =================================================================

namespace {

struct AblationRow
{
	const char* label;
	bool mom;
	bool ema;
	unsigned int basis;  // 0 = weights, 1 = gradient
};

} // namespace

void VESTASweepAblationCompare()
{
	printf("\n============================================================\n");
	printf("VESTA feature ablation (5 seeds)\n");
	printf("============================================================\n");

	RunSpec base;
	base.optType = glades::OptimizerConfig::VESTA;
	base.label = "VESTA";
	base.vocab = 29u;
	base.dModel = 128u;
	base.dFF = 256u;
	base.nLayers = 4u;
	base.nHeads = 4u;
	base.epochs = 15u;
	base.corpusLen = 384u;
	base.learningRate = 1e-2f;
	base.vestaRank = 8u;
	base.vestaTau = 0.1f;
	base.vestaRho = 0.1f;
	base.vestaTSk = 4u; // exercise refresh path within ~15 optimizer steps
	base.vestaLambdaPerp = 0.2f; // best for +mom; neutral starting point
	// Fast EMA so gradientEma carries signal within a 15-step training horizon.
	// (basisEmaBeta is consulted only when basisSource == 1.)

	const unsigned int seeds[] = { 101u, 202u, 303u, 404u, 505u };
	const unsigned int nSeeds = sizeof(seeds) / sizeof(seeds[0]);

	// Always include plain and +mom for continuity with prior sweeps, plus
	// single-feature and combined ablations.
	AblationRow rows[] = {
	    { "plain",             false, false, 0u },
	    { "+mom",              true,  false, 0u },
	    { "+ema",              false, true,  0u },
	    { "+gradbasis",        false, false, 1u },
	    { "+mom+ema",          true,  true,  0u },
	    { "+mom+gradbasis",    true,  false, 1u },
	    { "+ema+gradbasis",    false, true,  1u },
	    { "all3",              true,  true,  1u },
	};
	const unsigned int nRows = sizeof(rows) / sizeof(rows[0]);

	printf("Base: dModel=%u layers=%u epochs=%u seq=%u LR=%.1e rank=%u lp=%.2f tSk=%u\n\n",
	       base.dModel, base.nLayers, base.epochs, base.corpusLen,
	       base.learningRate, base.vestaRank, base.vestaLambdaPerp, base.vestaTSk);

	printf("%-18s  %-4s  %-18s  %-18s  %-10s\n",
	       "config", "n", "trainNLL", "testNLL", "wall(s)");
	printf("%-18s  %-4s  %-18s  %-18s  %-10s\n",
	       "------", "---", "------------------", "------------------", "----------");

	AggStats bestTest;
	bestTest.mean = 1e30f;
	const char* bestLabel = "(none)";

	for (unsigned int i = 0; i < nRows; ++i)
	{
		RunSpec s = base;
		s.vestaComplementMomentum = rows[i].mom;
		s.vestaTrackedEma = rows[i].ema;
		s.vestaBasisSource = rows[i].basis;
		s.vestaBasisEmaBeta = 0.7f; // fast warmup for short training horizon
		std::vector<float> trains, tests, walls;
		for (unsigned int k = 0; k < nSeeds; ++k)
		{
			const SweepResult r = run_one(s, seeds[k]);
			if (r.ok)
			{
				trains.push_back(r.finalTrainNll);
				tests.push_back(r.finalTestNll);
				walls.push_back(static_cast<float>(r.wallSec));
			}
		}
		const AggStats tA = aggregate(trains);
		const AggStats te = aggregate(tests);
		const AggStats wA = aggregate(walls);
		printf("%-18s  %-4u  %6.4f +/- %-7.4f  %6.4f +/- %-7.4f  %5.2f +/- %-5.2f\n",
		       rows[i].label, (unsigned int)trains.size(),
		       tA.mean, tA.stddev, te.mean, te.stddev, wA.mean, wA.stddev);
		if (te.mean < bestTest.mean)
		{
			bestTest = te;
			bestLabel = rows[i].label;
		}
	}

	// AdamW reference.
	RunSpec adam;
	adam.optType = glades::OptimizerConfig::ADAMW;
	adam.label = "AdamW";
	adam.vocab = base.vocab; adam.dModel = base.dModel; adam.dFF = base.dFF;
	adam.nLayers = base.nLayers; adam.nHeads = base.nHeads;
	adam.epochs = base.epochs; adam.corpusLen = base.corpusLen;
	adam.learningRate = 1e-2f;
	std::vector<float> adamTest;
	for (unsigned int k = 0; k < nSeeds; ++k)
	{
		const SweepResult r = run_one(adam, seeds[k]);
		if (r.ok) adamTest.push_back(r.finalTestNll);
	}
	const AggStats adamT = aggregate(adamTest);

	printf("\nAdamW reference: testNLL = %.4f +/- %.4f\n", adamT.mean, adamT.stddev);
	printf("\nBest VESTA variant: %s\n", bestLabel);
	printf("  testNLL = %.4f +/- %.4f\n", bestTest.mean, bestTest.stddev);
	printf("  Delta vs AdamW  = %+.4f nats\n", bestTest.mean - adamT.mean);
	printf("\n");
}

// =================================================================
// Scale ladder: does VESTA's gap to AdamW close as dModel grows, and
// does the memory advantage become real? Test at dModel in {64..512}.
// =================================================================

// Analytical optimizer-state bytes per weight matrix (fp32).
struct StateByteBreakdown
{
	size_t adamw;
	size_t vestaPlain;
	size_t vestaMom;
};

// File-scope (not namespace-anonymous, not local) so it can be used with
// std::vector under C++98.
struct AccShape { unsigned int m; unsigned int n; };

static StateByteBreakdown compute_state_bytes(unsigned int vocab,
                                              unsigned int dModel,
                                              unsigned int dFF,
                                              unsigned int nLayers,
                                              unsigned int vestaRank)
{
	StateByteBreakdown out;
	out.adamw = 0u; out.vestaPlain = 0u; out.vestaMom = 0u;

	std::vector<AccShape> shapes;
	// tokE [V, d]
	AccShape s; s.m = vocab; s.n = dModel; shapes.push_back(s);
	// per-block: Wq/Wk/Wv/Wo [d, d], W1 [dFF, d], W2 [d, dFF]
	for (unsigned int l = 0; l < nLayers; ++l)
	{
		AccShape a; a.m = dModel; a.n = dModel;
		shapes.push_back(a);
		shapes.push_back(a);
		shapes.push_back(a);
		shapes.push_back(a);
		AccShape w1; w1.m = dFF; w1.n = dModel; shapes.push_back(w1);
		AccShape w2; w2.m = dModel; w2.n = dFF; shapes.push_back(w2);
	}

	for (size_t i = 0; i < shapes.size(); ++i)
	{
		const size_t m = shapes[i].m, n = shapes[i].n;
		size_t rEff = vestaRank;
		if (rEff > m) rEff = m;
		if (rEff > n) rEff = n;
		const size_t adamw = 2u * m * n;
		const size_t vestaPlain = (m + n) * rEff + 2u * rEff;
		const size_t vestaMom = vestaPlain + m * n;
		out.adamw += adamw;
		out.vestaPlain += vestaPlain;
		out.vestaMom += vestaMom;
	}
	out.adamw *= sizeof(float);
	out.vestaPlain *= sizeof(float);
	out.vestaMom *= sizeof(float);
	return out;
}

void VESTASweepScaleLadder()
{
	printf("\n============================================================\n");
	printf("VESTA scale ladder: AdamW vs VESTA-plain vs VESTA+mom\n");
	printf("dModel in {64, 128, 256, 512}, 3 seeds each\n");
	printf("============================================================\n");

	const unsigned int vocab = 29u;
	const unsigned int nLayers = 4u;
	const unsigned int nHeads = 4u;
	const unsigned int epochs = 15u;
	const unsigned int corpusLen = 384u;
	const float lr = 1e-2f;
	const unsigned int vestaRank = 8u;

	// dModel=512 is unreasonably slow on CPU due to O(n^3) Jacobi SVD at
	// each subspace refresh. Stopping at 256 for the feasibility-bound
	// ladder; a follow-up GPU-training-loop run is needed for >=512.
	const unsigned int scales[] = { 64u, 128u, 256u };
	const unsigned int nScales = sizeof(scales) / sizeof(scales[0]);
	const unsigned int seeds[] = { 101u, 202u, 303u };
	const unsigned int nSeeds = sizeof(seeds) / sizeof(seeds[0]);

	printf("Fixed: vocab=%u layers=%u heads=%u epochs=%u corpus=%u LR=%.1e rank=%u\n\n",
	       vocab, nLayers, nHeads, epochs, corpusLen, lr, vestaRank);

	// State-size comparison (analytical, no training needed).
	printf("Optimizer state (KiB, fp32, summed across all transformer weights):\n");
	printf("%-8s  %-10s  %-12s  %-11s  %-12s  %-11s\n",
	       "dModel", "AdamW(KiB)", "VESTA_plain", "plain/Adam", "VESTA+mom", "+mom/Adam");
	printf("%-8s  %-10s  %-12s  %-11s  %-12s  %-11s\n",
	       "------", "----------", "------------", "-----------", "------------", "-----------");
	for (unsigned int si = 0; si < nScales; ++si)
	{
		const unsigned int d = scales[si];
		const unsigned int dFF = 2u * d;
		const StateByteBreakdown sb = compute_state_bytes(vocab, d, dFF, nLayers, vestaRank);
		printf("%-8u  %-10.1f  %-12.1f  %-11.3f  %-12.1f  %-11.3f\n",
		       d,
		       sb.adamw / 1024.0,
		       sb.vestaPlain / 1024.0,
		       (double)sb.vestaPlain / sb.adamw,
		       sb.vestaMom / 1024.0,
		       (double)sb.vestaMom / sb.adamw);
	}

	printf("\nTrain/test NLL at each scale:\n\n");

	struct Variant { const char* label; glades::OptimizerConfig::Type type; bool mom; };
	Variant variants[3];
	variants[0].label = "AdamW";       variants[0].type = glades::OptimizerConfig::ADAMW; variants[0].mom = false;
	variants[1].label = "VESTA-plain"; variants[1].type = glades::OptimizerConfig::VESTA; variants[1].mom = false;
	variants[2].label = "VESTA+mom";   variants[2].type = glades::OptimizerConfig::VESTA; variants[2].mom = true;

	printf("%-8s  %-12s  %-4s  %-18s  %-18s  %-10s\n",
	       "dModel", "optimizer", "n", "trainNLL", "testNLL", "wall(s)");
	printf("%-8s  %-12s  %-4s  %-18s  %-18s  %-10s\n",
	       "------", "----------", "---", "------------------", "------------------", "----------");

	for (unsigned int si = 0; si < nScales; ++si)
	{
		const unsigned int d = scales[si];
		const unsigned int dFF = 2u * d;
		std::vector<AggStats> testPerVariant(3);
		std::vector<AggStats> wallPerVariant(3);
		for (unsigned int vi = 0; vi < 3u; ++vi)
		{
			RunSpec s;
			s.optType = variants[vi].type;
			s.label = variants[vi].label;
			s.vocab = vocab;
			s.dModel = d;
			s.dFF = dFF;
			s.nLayers = nLayers;
			s.nHeads = nHeads;
			s.epochs = epochs;
			s.corpusLen = corpusLen;
			s.learningRate = lr;
			s.vestaRank = vestaRank;
			s.vestaTSk = 16u;
			s.vestaLambdaPerp = variants[vi].mom ? 0.2f : 0.4f;
			s.vestaComplementMomentum = variants[vi].mom;
			s.vestaComplementBeta = 0.9f;
			std::vector<float> trains, tests, walls;
			for (unsigned int k = 0; k < nSeeds; ++k)
			{
				const SweepResult r = run_one(s, seeds[k]);
				if (r.ok)
				{
					trains.push_back(r.finalTrainNll);
					tests.push_back(r.finalTestNll);
					walls.push_back(static_cast<float>(r.wallSec));
				}
			}
			const AggStats tA = aggregate(trains);
			const AggStats te = aggregate(tests);
			const AggStats wA = aggregate(walls);
			testPerVariant[vi] = te;
			wallPerVariant[vi] = wA;
			printf("%-8u  %-12s  %-4u  %6.4f +/- %-7.4f  %6.4f +/- %-7.4f  %5.2f +/- %-5.2f\n",
			       d, variants[vi].label, (unsigned int)trains.size(),
			       tA.mean, tA.stddev, te.mean, te.stddev, wA.mean, wA.stddev);
		}
		// Row summary: deltas and wall-clock ratios.
		const float gapPlain = testPerVariant[1].mean - testPerVariant[0].mean;
		const float gapMom = testPerVariant[2].mean - testPerVariant[0].mean;
		const float wRatioPlain = wallPerVariant[1].mean / std::max(wallPerVariant[0].mean, 1e-6f);
		const float wRatioMom = wallPerVariant[2].mean / std::max(wallPerVariant[0].mean, 1e-6f);
		printf("%-8s  plain:  testNLL delta = %+.4f nats   wall ratio = %.2fx\n",
		       "", gapPlain, wRatioPlain);
		printf("%-8s  +mom :  testNLL delta = %+.4f nats   wall ratio = %.2fx\n\n",
		       "", gapMom, wRatioMom);
	}

	printf("\nEnd of ladder. Look for monotone shrinking gap in +mom delta\n");
	printf("and an asymptotically small wall-clock ratio as scale grows.\n");
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
	VESTAComplementMomentumTest();
	VESTATrackedEmaTest();
	VESTAGradientBasisTest();
	VESTATransformerIntegrationTest();
}
