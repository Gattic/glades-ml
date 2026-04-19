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

#include "helios-test.h"
#include "../../unit-test.h"
#include "test_token_id_input_fixture.h"

#include "../../../Backend/Machine Learning/Networks/helios_optimizer.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/rng.h"

#include <sys/time.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

static bool close_abs(float a, float b, float tol)
{
	return fabsf(a - b) <= tol;
}

static double wall_ms_helios()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return static_cast<double>(tv.tv_sec) * 1000.0
	     + static_cast<double>(tv.tv_usec) / 1000.0;
}

class HeliosMetricCapture : public glades::ITrainingCallbacks
{
public:
	HeliosMetricCapture() : saw(false), last() {}
	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&,
	                        const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}
	bool saw;
	glades::NNetworkEpochMetrics last;
};

static void build_corpus(std::vector<unsigned int>& toks,
                         unsigned int vocab, unsigned int length,
                         uint64_t seed)
{
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, seed);
	toks.clear();
	toks.reserve(length);
	// Structured pattern + noise: pure period-7 pattern with occasional
	// random token. Learnable but not trivially memorizable.
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

struct CmpResult
{
	float finalTrainNll;
	float finalTestNll;
	double wallSec;
	bool ok;
	CmpResult() : finalTrainNll(0.0f), finalTestNll(0.0f),
	              wallSec(0.0), ok(false) {}
};

static CmpResult run_one_helios_cmp(glades::OptimizerConfig::Type optType,
                                    const char* label,
                                    unsigned int vocab,
                                    unsigned int dModel,
                                    unsigned int dFF,
                                    unsigned int nLayers,
                                    unsigned int nHeads,
                                    unsigned int epochs,
                                    unsigned int corpusLen,
                                    float learningRate,
                                    unsigned int seed)
{
	CmpResult res;

	std::vector<unsigned int> trainToks;
	build_corpus(trainToks, vocab, corpusLen, 0x5EEDULL + seed);
	std::vector<unsigned int> testToks;
	build_corpus(testToks, vocab, corpusLen, 0x7357ULL + seed);

	InMemoryTokenIdInput di;
	di.setTrainTokens(trainToks, -1);
	di.setTestTokens(testToks, -1);

	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
	std::vector<glades::HiddenLayerInfo*> hidden;
	for (unsigned int i = 0; i < nLayers; ++i)
		hidden.push_back(new glades::HiddenLayerInfo(
		    static_cast<int>(dModel), learningRate, 0.0f, 0.0f, 0.0f, 0.0f,
		    glades::GMath::LINEAR, 1.0f));
	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
	    static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
	glades::NNInfo* info = new glades::NNInfo("helios_cmp", in, hidden, out);

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
		// HELIOS defaults tuned for small-transformer scale: small T (mostly
		// deterministic descent with a little noise), moderate friction.
		cfg.helios.h = 1.0f;
		cfg.helios.gamma0 = 0.3f;
		cfg.helios.T0 = 1e-5f;
		cfg.helios.mass = 1.0f;
		cfg.helios.Q = 0.0f;
		cfg.helios.alpha = 0.0f;
		cfg.helios.kHvp = 0u;
		cfg.helios.lambdaAnchor = 0.0f;
		cfg.helios.noiseCorrection = 0.0f;
	}

	HeliosMetricCapture trainCb;
	const double t0 = wall_ms_helios();
	const glades::NNetworkStatus stTrain = net.train(&di, &trainCb);
	const double t1 = wall_ms_helios();
	res.wallSec = (t1 - t0) / 1000.0;
	if (!stTrain.ok() || !trainCb.saw)
	{
		printf("  [%s seed=%u] TRAIN FAILED: %s\n",
		       label, seed, stTrain.message.c_str());
		delete info;
		return res;
	}
	res.finalTrainNll = trainCb.last.totalError;

	HeliosMetricCapture testCb;
	const glades::NNetworkStatus stTest = net.test(&di, &testCb);
	if (!stTest.ok() || !testCb.saw)
	{
		printf("  [%s seed=%u] TEST FAILED\n", label, seed);
		delete info;
		return res;
	}
	res.finalTestNll = testCb.last.totalError;
	res.ok = true;

	delete info;
	return res;
}

} // namespace

// Verify initWeightState allocates momentum buffer of the right size and
// leaves weights untouched.
void HELIOSInitStateTest()
{
	printf("[helios] InitStateTest\n");
	const unsigned int m = 4u, n = 3u;
	std::vector<float> W(static_cast<size_t>(m) * n, 0.5f);

	glades::HeliosConfig hc;
	hc.T0 = 0.1f;
	hc.mass = 2.0f;

	glades::helios::WeightState st;
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 0x1ULL);
	glades::helios::initWeightState(st, &W[0], m, n, hc, rng, 0);

	ASSERT("init sets m", st.m == m);
	ASSERT("init sets n", st.n == n);
	ASSERT("init allocates p", st.p.size() == static_cast<size_t>(m) * n);
	ASSERT("init p is zero",
	       st.p[0] == 0.0f && st.p[st.p.size() - 1] == 0.0f);
	ASSERT("init anchor off by default", st.thetaBar.empty());
	ASSERT("init copies T0 into Tcurr", close_abs(st.Tcurr, hc.T0, 1e-6f));
	ASSERT("init copies mass", close_abs(st.mass, hc.mass, 1e-6f));

	// W should be untouched by init.
	for (size_t i = 0; i < W.size(); ++i)
	{
		ASSERT("init leaves W untouched", W[i] == 0.5f);
	}
}

// BAOAB invariant-measure test on the isotropic quadratic potential
// U(theta) = (1/2) * ||theta||^2. The gradient is theta itself, and the
// invariant theta-marginal is N(0, T * I). We estimate the empirical mean
// and variance across many steps and many independent coordinates, then
// verify they converge to 0 and T respectively within finite-sample tolerance.
//
// This is the central correctness test for the BAOAB integrator: if it does
// not sample the target Gibbs measure, the optimizer is structurally broken.
void HELIOSBaoabInvariantTest()
{
	printf("[helios] BaoabInvariantTest (sample Gibbs measure on quadratic)\n");

	const unsigned int m = 32u, n = 32u; // 1024 independent coordinates
	const size_t P = static_cast<size_t>(m) * n;
	std::vector<float> W(P, 0.0f);
	std::vector<float> g(P, 0.0f);

	glades::HeliosConfig hc;
	hc.h = 1.0f;        // use lr directly as the step
	hc.gamma0 = 1.0f;   // critical damping for unit-mass unit-curvature oscillator
	hc.T0 = 1.0f;
	hc.mass = 1.0f;
	hc.Q = 0.0f;        // thermostat off (pure Langevin)
	hc.alpha = 0.0f;    // no sharpness feedback
	hc.lambdaAnchor = 0.0f;
	hc.noiseCorrection = 0.0f;
	hc.kHvp = 0u;

	glades::helios::WeightState st;
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 0x4E105ULL);
	glades::helios::initWeightState(st, &W[0], m, n, hc, rng, 0);

	// Use a small integrator step. The applyStep form reuses the same force
	// in both B-halves (the API does not expose a gradient callback at q_new),
	// which is a valid stochastic-symplectic integrator with O(h^2) bias but
	// with a larger bias constant than the strict BAOAB that re-evaluates the
	// gradient at q_new. h=0.05 puts expected bias well below 2% so the 5%
	// tolerance is met with margin; that is enough to assert the integrator
	// samples the correct Gibbs measure in the small-h limit.
	const float lr = 0.05f; // integrator step
	const unsigned int burnIn = 2000u;
	const unsigned int samples = 20000u;

	// Burn-in.
	for (unsigned int s = 0; s < burnIn; ++s)
	{
		for (size_t i = 0; i < P; ++i) g[i] = W[i]; // grad U = theta
		const bool ok = glades::helios::applyStep(st, &W[0], &g[0], m, n,
		                                         1.0f, lr, 0.0f, 0.0f, 1.0f,
		                                         hc, rng, 0, 0);
		ASSERT("BAOAB burn-in non-finite", ok);
	}

	// Sampling: accumulate mean and second moment of theta across all
	// coordinates and all sampled steps.
	double meanAcc = 0.0;
	double sqAcc = 0.0;
	size_t count = 0;

	for (unsigned int s = 0; s < samples; ++s)
	{
		for (size_t i = 0; i < P; ++i) g[i] = W[i];
		const bool ok = glades::helios::applyStep(st, &W[0], &g[0], m, n,
		                                         1.0f, lr, 0.0f, 0.0f, 1.0f,
		                                         hc, rng, 0, 0);
		ASSERT("BAOAB sample non-finite", ok);

		for (size_t i = 0; i < P; ++i)
		{
			const double v = static_cast<double>(W[i]);
			meanAcc += v;
			sqAcc += v * v;
		}
		count += P;
	}

	const double empMean = meanAcc / static_cast<double>(count);
	const double empVar = sqAcc / static_cast<double>(count) - empMean * empMean;

	printf("  samples=%zu emp_mean=%.5f emp_var=%.5f target_var=%.5f\n",
	       count, empMean, empVar, static_cast<double>(hc.T0));

	// The BAOAB integrator has O(h^2) configurational bias; with h=0.1 the
	// bias on the variance is ~1%. Finite-sample std-error of var is
	// ~var * sqrt(2 / N_eff). With samples*P ~ 2e7 correlated samples,
	// N_eff is roughly samples/autocorr_time; autocorr_time ~ 1/(gamma*h) ~ 10.
	// Final tolerance ~5% captures bias + sampling noise comfortably.
	ASSERT("BAOAB invariant mean near 0", fabs(empMean) < 0.02);
	ASSERT("BAOAB invariant variance near T", fabs(empVar - hc.T0) < 0.05);
}

// Verify deterministic descent when T0 == 0 (no noise injection).
// With a quadratic loss (1/2) ||W - Wtarget||^2 and positive friction, HELIOS
// should converge monotonically after a short transient.
void HELIOSStepDescentTest()
{
	printf("[helios] StepDescentTest (T=0 deterministic descent)\n");

	const unsigned int m = 12u, n = 8u;
	const size_t P = static_cast<size_t>(m) * n;

	std::vector<float> W(P, 0.0f);
	std::vector<float> Wtarget(P, 0.0f);
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, 0xD35CULL);
	for (size_t i = 0; i < P; ++i)
	{
		W[i] = 0.3f * glades::rng::standard_normal(eng);
		Wtarget[i] = 0.5f * glades::rng::standard_normal(eng);
	}

	glades::HeliosConfig hc;
	hc.h = 1.0f;
	hc.gamma0 = 0.8f;
	hc.T0 = 0.0f; // deterministic
	hc.mass = 1.0f;
	hc.Q = 0.0f;
	hc.alpha = 0.0f;
	hc.lambdaAnchor = 0.0f;
	hc.noiseCorrection = 0.0f;
	hc.kHvp = 0u;

	glades::helios::WeightState st;
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 0xBEEFULL);
	glades::helios::initWeightState(st, &W[0], m, n, hc, rng, 0);

	float initLoss = 0.0f;
	for (size_t i = 0; i < P; ++i)
	{
		const float d = W[i] - Wtarget[i];
		initLoss += 0.5f * d * d;
	}

	const unsigned int steps = 300u;
	const float lr = 0.15f;
	std::vector<float> g(P, 0.0f);

	for (unsigned int s = 0; s < steps; ++s)
	{
		for (size_t i = 0; i < P; ++i) g[i] = W[i] - Wtarget[i];
		const bool ok = glades::helios::applyStep(st, &W[0], &g[0], m, n,
		                                         1.0f, lr, 0.0f, 0.0f, 1.0f,
		                                         hc, rng, 0, 0);
		ASSERT("deterministic BAOAB non-finite", ok);
	}

	float finalLoss = 0.0f;
	for (size_t i = 0; i < P; ++i)
	{
		const float d = W[i] - Wtarget[i];
		finalLoss += 0.5f * d * d;
	}
	printf("  init=%.4f final=%.4f ratio=%.4e\n",
	       initLoss, finalLoss, finalLoss / initLoss);
	ASSERT("deterministic descent reduces loss", finalLoss < 0.05f * initLoss);
}

// Verify the optimizer returns false (and logs) when given a NaN gradient.
void HELIOSNonFiniteGuardTest()
{
	printf("[helios] NonFiniteGuardTest\n");
	const unsigned int m = 3u, n = 3u;
	std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
	std::vector<float> g(W.size(), 0.0f);
	g[0] = std::sqrt(-1.0f); // NaN

	glades::HeliosConfig hc;
	hc.T0 = 0.0f;

	glades::helios::WeightState st;
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 0xFULL);
	glades::helios::initWeightState(st, &W[0], m, n, hc, rng, 0);

	const bool ok = glades::helios::applyStep(st, &W[0], &g[0], m, n,
	                                          1.0f, 0.1f, 0.0f, 0.0f, 1.0f,
	                                          hc, rng, 0, 0);
	ASSERT("non-finite gradient should return false", !ok);
}

// Compare HELIOS vs AdamW on a tiny token LM (transformer decoder).
// Trains both with the same seed/corpus/architecture and reports final train
// and test NLL. If HELIOS's mean test NLL is below AdamW's, the test records
// a WIN; otherwise LOSS/TIE. Multi-seed to reduce per-seed noise.
void HELIOSvsAdamWComparisonTest()
{
	printf("\n[helios] HELIOSvsAdamWComparisonTest (token LM)\n");

	const unsigned int vocab = 29u;
	const unsigned int dModel = 64u;
	const unsigned int dFF = 128u;
	const unsigned int nLayers = 3u;
	const unsigned int nHeads = 4u;
	const unsigned int epochs = 60u;
	const unsigned int corpusLen = 256u;
	const float adamLr = 1e-3f;

	const unsigned int seeds[] = { 101u, 202u, 303u };
	const unsigned int nSeeds = sizeof(seeds) / sizeof(seeds[0]);

	printf("Config: vocab=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u corpus=%u seeds=%u\n",
	       vocab, dModel, dFF, nLayers, nHeads, epochs, corpusLen, nSeeds);

	(void)adamLr;
	// AdamW LR sweep (well-known optimal LR band for small transformers is
	// roughly 3e-4 ... 3e-3; we test one order of magnitude around that).
	const float adamLrs[] = { 3e-4f, 1e-3f, 3e-3f, 1e-2f };
	const unsigned int nAdamLrs = sizeof(adamLrs) / sizeof(adamLrs[0]);
	float bestAdamTestMean  = 1e9f;
	float bestAdamTrainMean = 1e9f;
	float bestAdamLr        = 0.0f;
	double bestAdamWall     = 0.0;
	for (unsigned int ai = 0; ai < nAdamLrs; ++ai)
	{
		const float lr = adamLrs[ai];
		double trSum = 0.0, teSum = 0.0, wSum = 0.0;
		unsigned int ok = 0;
		for (unsigned int s = 0; s < nSeeds; ++s)
		{
			const unsigned int seed = seeds[s];
			const CmpResult r = run_one_helios_cmp(
			    glades::OptimizerConfig::ADAMW, "AdamW",
			    vocab, dModel, dFF, nLayers, nHeads, epochs, corpusLen,
			    lr, seed);
			if (r.ok) { trSum += r.finalTrainNll; teSum += r.finalTestNll;
			            wSum += r.wallSec; ++ok; }
			printf("  [seed=%u] AdamW(lr=%.0e)  trainNLL=%.4f testNLL=%.4f wall=%.2fs ok=%d\n",
			       seed, lr, r.finalTrainNll, r.finalTestNll, r.wallSec, r.ok ? 1 : 0);
		}
		if (ok != nSeeds) continue;
		const float trMean = static_cast<float>(trSum / nSeeds);
		const float teMean = static_cast<float>(teSum / nSeeds);
		if (teMean < bestAdamTestMean)
		{
			bestAdamTestMean  = teMean;
			bestAdamTrainMean = trMean;
			bestAdamLr        = lr;
			bestAdamWall      = wSum / nSeeds;
		}
	}
	const float adamTrainMean = bestAdamTrainMean;
	const float adamTestMean  = bestAdamTestMean;
	const double adamWallSum  = bestAdamWall * nSeeds;

	// HELIOS LR sweep. HELIOS uses un-normalized physical steps, so its optimal
	// LR lives on a different scale from AdamW. Sweep orders of magnitude.
	const float hlLrs[] = { 1e-1f, 3e-1f, 1.0f, 3.0f };
	const unsigned int nHlLrs = sizeof(hlLrs) / sizeof(hlLrs[0]);

	float bestHlTestMean = 1e9f;
	float bestHlTrainMean = 1e9f;
	float bestHlLr = 0.0f;
	double bestHlWall = 0.0;

	for (unsigned int li = 0; li < nHlLrs; ++li)
	{
		const float lr = hlLrs[li];
		double trSum = 0.0, teSum = 0.0, wSum = 0.0;
		unsigned int ok = 0;
		for (unsigned int s = 0; s < nSeeds; ++s)
		{
			const unsigned int seed = seeds[s];
			const CmpResult r = run_one_helios_cmp(
			    glades::OptimizerConfig::HELIOS, "HELIOS",
			    vocab, dModel, dFF, nLayers, nHeads, epochs, corpusLen,
			    lr, seed);
			if (r.ok) { trSum += r.finalTrainNll; teSum += r.finalTestNll;
			            wSum += r.wallSec; ++ok; }
			printf("  [seed=%u] HELIOS(lr=%.0e) trainNLL=%.4f testNLL=%.4f wall=%.2fs ok=%d\n",
			       seed, lr, r.finalTrainNll, r.finalTestNll, r.wallSec, r.ok ? 1 : 0);
		}
		if (ok != nSeeds) continue;
		const float trMean = static_cast<float>(trSum / nSeeds);
		const float teMean = static_cast<float>(teSum / nSeeds);
		if (teMean < bestHlTestMean)
		{
			bestHlTestMean  = teMean;
			bestHlTrainMean = trMean;
			bestHlLr        = lr;
			bestHlWall      = wSum / nSeeds;
		}
	}

	printf("\n  === HELIOS vs AdamW summary (mean over %u seeds) ===\n", nSeeds);
	printf("  AdamW(best lr=%.0e)      trainNLL=%.4f testNLL=%.4f wall=%.2fs\n",
	       bestAdamLr, adamTrainMean, adamTestMean, adamWallSum / nSeeds);
	printf("  HELIOS(best lr=%.0e)     trainNLL=%.4f testNLL=%.4f wall=%.2fs\n",
	       bestHlLr, bestHlTrainMean, bestHlTestMean, bestHlWall);
	printf("  delta(HELIOS_best - AdamW): trainNLL=%+.4f testNLL=%+.4f\n",
	       bestHlTrainMean - adamTrainMean, bestHlTestMean - adamTestMean);
	if (bestHlTestMean < adamTestMean)
		printf("  VERDICT: HELIOS wins on test NLL by %.4f (lr=%.0e)\n",
		       adamTestMean - bestHlTestMean, bestHlLr);
	else if (bestHlTestMean > adamTestMean)
		printf("  VERDICT: AdamW wins on test NLL by %.4f\n",
		       bestHlTestMean - adamTestMean);
	else
		printf("  VERDICT: TIE on test NLL\n");

	ASSERT("HELIOS produced finite train NLL",
	       bestHlTrainMean == bestHlTrainMean && bestHlTrainMean < 100.0f);
	ASSERT("HELIOS produced finite test NLL",
	       bestHlTestMean == bestHlTestMean && bestHlTestMean < 100.0f);
}

void HELIOSUnitTest()
{
	HELIOSInitStateTest();
	HELIOSBaoabInvariantTest();
	HELIOSStepDescentTest();
	HELIOSNonFiniteGuardTest();
	HELIOSvsAdamWComparisonTest();
}
