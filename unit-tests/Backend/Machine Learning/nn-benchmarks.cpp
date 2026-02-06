// Benchmark: compare DFF vs RNN vs GRU vs LSTM performance on a small dataset.
//
// This follows the unit-tests runner dispatch pattern (see unit-tests/main.cpp).
// Run:
//   ./build/glades-unit-tests nn-bench --dataset datasets/rnn.csv --epochs 200 --hidden 8 --repeats 3
//
#include "nn-benchmarks.h"
#include "../../../Backend/Machine Learning/main.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Database/GTable.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cmath>
#include <string>
#include <vector>

namespace {
static int64_t now_ms()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return static_cast<int64_t>(tv.tv_sec) * 1000LL + static_cast<int64_t>(tv.tv_usec) / 1000LL;
}

static bool streq(const char* a, const char* b) { return (a && b && strcmp(a, b) == 0); }

class CaptureMetricsCallbacks : public glades::ITrainingCallbacks
{
public:
	CaptureMetricsCallbacks() : last(), saw(false) { last = glades::NNetworkEpochMetrics(); }
	virtual void onRunStart(const glades::NNetwork& /*net*/, int /*runType*/) {}
	virtual bool onEpochEnd(const glades::NNetwork& /*net*/, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork& /*net*/, int /*runType*/) {}

	glades::NNetworkEpochMetrics last;
	bool saw;
};

static const char* netTypeName(int t)
{
	switch (t)
	{
	case glades::NNetwork::TYPE_DFF: return "DFF";
	case glades::NNetwork::TYPE_RNN: return "RNN";
	case glades::NNetwork::TYPE_GRU: return "GRU";
	case glades::NNetwork::TYPE_LSTM: return "LSTM";
	default: return "UNKNOWN";
	}
}

struct BenchResult
{
	const char* name;
	const char* variant;
	long long trainMs;
	long long testMs;
	glades::NNetworkEpochMetrics trainLast;
	glades::NNetworkEpochMetrics testLast;
	bool ok;
	std::string err;

	BenchResult()
	    : name(""),
	      variant("baseline"),
	      trainMs(0),
	      testMs(0),
	      trainLast(),
	      testLast(),
	      ok(true),
	      err()
	{
	}
};

struct BenchVariantConfig
{
	BenchVariantConfig()
	    : schedule("none"),
	      stepSize(0),
	      gamma(1.0f),
	      tMax(0),
	      minMult(0.0f),
	      clipNorm(0.0f)
	{
	}

	const char* schedule; // none|step|exp|cosine
	int stepSize;
	float gamma;
	int tMax;
	float minMult;
	float clipNorm; // DFF only; 0 disables
};

static bool is_finite(float x)
{
	return std::isfinite(x);
}

static bool validate_metrics(const glades::NNetworkEpochMetrics& m, std::string& outErr)
{
	outErr.clear();
	if (!is_finite(m.totalError) || !is_finite(m.totalAccuracy) ||
	    !is_finite(m.learningRate) || !is_finite(m.lrMultiplier) ||
	    !is_finite(m.gradNorm) || !is_finite(m.gradNormScale))
	{
		outErr = "non-finite metrics detected (NaN/Inf)";
		return false;
	}
	// For regression: regMAE/regRMSE should be finite.
	if (!is_finite(m.regMAE) || !is_finite(m.regRMSE))
	{
		outErr = "non-finite regression metrics detected (NaN/Inf)";
		return false;
	}
	// Grad norm scale should be in [0,1] for train passes.
	if (m.gradNormScale < 0.0f || m.gradNormScale > 1.0f + 1e-6f)
	{
		outErr = "gradNormScale out of expected range";
		return false;
	}
	return true;
}

static BenchResult run_one(const glades::NumberInput& data,
                           int netType,
                           int epochs,
                           int hiddenSize,
                           float lr,
                           unsigned int seed,
                           const BenchVariantConfig& cfg)
{
	BenchResult r;
	r.name = netTypeName(netType);
	r.variant = cfg.schedule;

	// Model:
	// - input features: 2 (x,y)
	// - hidden: hiddenSize
	// - output: 1 regression (z)
	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    /*batchSize*/ 1,
	    /*learningRate*/ lr,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::LINEAR,
	    /*activationParam*/ 1.0f);

	std::vector<glades::HiddenLayerInfo*> hidden;
	hidden.push_back(new glades::HiddenLayerInfo(
	    /*size*/ hiddenSize,
	    /*learningRate*/ lr,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::LINEAR,
	    /*activationParam*/ 1.0f));

	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
	glades::NNInfo* info = new glades::NNInfo((std::string("bench_") + r.name).c_str(), in, hidden, out);

	glades::NNetwork net(info, netType);
	net.setSeed(static_cast<uint64_t>(seed));
	net.getTerminatorMutable().setEpoch(epochs);
	net.getTerminatorMutable().setAccuracy(0.0f);

	// Apply modern training-loop variants (optional).
	if (cfg.schedule)
	{
		if (streq(cfg.schedule, "none"))
			net.setLearningRateScheduleNone();
		else if (streq(cfg.schedule, "exp"))
			net.setLearningRateScheduleExp(cfg.gamma);
		else if (streq(cfg.schedule, "step"))
			net.setLearningRateScheduleStep(cfg.stepSize, cfg.gamma);
		else if (streq(cfg.schedule, "cosine"))
			net.setLearningRateScheduleCosine(cfg.tMax, cfg.minMult);
		else
			net.setLearningRateScheduleNone();
	}
	if (cfg.clipNorm > 0.0f)
		net.setGlobalGradClipNorm(cfg.clipNorm);

	// Train
	CaptureMetricsCallbacks trainCb;
	const int64_t t0 = now_ms();
	const glades::NNetworkStatus stTrain = net.train(&data, &trainCb);
	const int64_t t1 = now_ms();
	r.trainMs = (long long)(t1 - t0);
	if (!stTrain.ok())
	{
		r.ok = false;
		r.err = stTrain.message;
	}
	if (trainCb.saw)
		r.trainLast = trainCb.last;
	if (r.ok && trainCb.saw)
	{
		std::string err;
		if (!validate_metrics(r.trainLast, err))
		{
			r.ok = false;
			r.err = std::string("train: ") + err;
		}
		// Schedule sanity: "none" should keep multiplier at 1.
		if (r.ok && cfg.schedule && streq(cfg.schedule, "none") && fabs(r.trainLast.lrMultiplier - 1.0f) > 1e-6f)
		{
			r.ok = false;
			r.err = "train: lrMultiplier should be 1.0 for schedule=none";
		}
		// Clip sanity: when enabled, scale should not exceed 1.
		if (r.ok && cfg.clipNorm > 0.0f && (r.trainLast.gradNormScale > 1.0f + 1e-6f))
		{
			r.ok = false;
			r.err = "train: gradNormScale > 1 with clipping enabled";
		}
	}

	// Test: evaluates the test split (bench harness may mirror train->test when absent).
	CaptureMetricsCallbacks testCb;
	const int64_t t2 = now_ms();
	const glades::NNetworkStatus stTest = net.test(&data, &testCb);
	const int64_t t3 = now_ms();
	r.testMs = (long long)(t3 - t2);
	if (!stTest.ok())
	{
		r.ok = false;
		if (!r.err.size())
			r.err = stTest.message;
	}
	if (testCb.saw)
		r.testLast = testCb.last;
	if (r.ok && testCb.saw)
	{
		std::string err;
		if (!validate_metrics(r.testLast, err))
		{
			r.ok = false;
			if (!r.err.size())
				r.err = std::string("test: ") + err;
		}
		// Evaluation should report neutral schedule/grad fields (set by Trainer).
		if (r.ok && (fabs(r.testLast.lrMultiplier - 1.0f) > 1e-6f || fabs(r.testLast.learningRate - 0.0f) > 1e-6f))
		{
			r.ok = false;
			if (!r.err.size())
				r.err = "test: schedule fields not neutral (expected lrMultiplier=1, learningRate=0)";
		}
	}

	delete info; // owns in/hidden/out
	return r;
}

static void print_header()
{
	printf("Type\tVariant\tTrain(ms)\tTest(ms)\tTrain R2(%%)\tTrain MSE\tTrain MAE\tTrain RMSE\tLR(mult)\tGradNorm(scale)\tTest R2(%%)\tTest MSE\tStatus\n");
}

static void print_row(const BenchResult& r)
{
	printf("%s\t%s\t%lld\t\t%lld\t\t%.3f\t\t%.6f\t%.6f\t%.6f\t%g(%g)\t%g(%g)\t%.3f\t\t%.6f\t%s\n",
	       r.name,
	       (r.variant ? r.variant : "baseline"),
	       r.trainMs,
	       r.testMs,
	       r.trainLast.totalAccuracy,
	       r.trainLast.totalError,
	       r.trainLast.regMAE,
	       r.trainLast.regRMSE,
	       r.trainLast.learningRate,
	       r.trainLast.lrMultiplier,
	       r.trainLast.gradNorm,
	       r.trainLast.gradNormScale,
	       r.testLast.totalAccuracy,
	       r.testLast.totalError,
	       (r.ok ? "OK" : "ERR"));
	if (!r.ok && r.err.size())
		printf("  error: %s\n", r.err.c_str());
}
} // namespace

void NNBenchmarks(int argc, char* argv[])
{
	const char* dataset = "datasets/rnn.csv";
	int epochs = 200;
	int hidden = 8;
	int repeats = 3;
	float lr = 0.05f;
	int standardize = glades::GMath::MINMAX;
	bool doAll = true;
	bool wantDff = true, wantRnn = true, wantGru = true, wantLstm = true;
	BenchVariantConfig variant;

	// argv here is the full process argv from main; we expect:
	//   argv[1] == "nn-bench"
	// and options starting at argv[2].
	for (int i = 2; i < argc; ++i)
	{
		if (streq(argv[i], "--dataset") && i + 1 < argc) dataset = argv[++i];
		else if (streq(argv[i], "--epochs") && i + 1 < argc) epochs = atoi(argv[++i]);
		else if (streq(argv[i], "--hidden") && i + 1 < argc) hidden = atoi(argv[++i]);
		else if (streq(argv[i], "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
		else if (streq(argv[i], "--lr") && i + 1 < argc) lr = (float)atof(argv[++i]);
		else if (streq(argv[i], "--lr-schedule") && i + 1 < argc) variant.schedule = argv[++i]; // none|step|exp|cosine
		else if (streq(argv[i], "--gamma") && i + 1 < argc) variant.gamma = (float)atof(argv[++i]);
		else if (streq(argv[i], "--step-size") && i + 1 < argc) variant.stepSize = atoi(argv[++i]);
		else if (streq(argv[i], "--tmax") && i + 1 < argc) variant.tMax = atoi(argv[++i]);
		else if (streq(argv[i], "--min-mult") && i + 1 < argc) variant.minMult = (float)atof(argv[++i]);
		else if (streq(argv[i], "--clip-norm") && i + 1 < argc) variant.clipNorm = (float)atof(argv[++i]);
		else if (streq(argv[i], "--standardize") && i + 1 < argc)
		{
			const char* v = argv[++i];
			if (streq(v, "none"))
				standardize = glades::GMath::NONE;
			else
				standardize = glades::GMath::MINMAX;
		}
		else if (streq(argv[i], "--net") && i + 1 < argc)
		{
			const char* v = argv[++i];
			doAll = false;
			wantDff = wantRnn = wantGru = wantLstm = false;
			if (streq(v, "dff")) wantDff = true;
			else if (streq(v, "rnn")) wantRnn = true;
			else if (streq(v, "gru")) wantGru = true;
			else if (streq(v, "lstm")) wantLstm = true;
			else if (streq(v, "all")) { doAll = true; wantDff = wantRnn = wantGru = wantLstm = true; }
		}
		else if (streq(argv[i], "--help"))
		{
			printf("Usage: glades-unit-tests nn-bench [--dataset PATH] [--epochs N] [--hidden N] [--repeats N] [--lr F] [--standardize none|minmax] [--net all|dff|rnn|gru|lstm]\n");
			printf("                             [--lr-schedule none|step|exp|cosine] [--gamma F] [--step-size N] [--tmax N] [--min-mult F]\n");
			printf("                             [--clip-norm F]\n");
			return;
		}
	}

	if (epochs <= 0) epochs = 1;
	if (hidden <= 0) hidden = 1;
	if (repeats <= 0) repeats = 1;

	// Load dataset and mark output column.
	shmea::GTable raw(dataset, ',', shmea::GTable::TYPE_FILE);
	raw.clearOutputs();
	// rnn.csv: columns x,y,z (z is output)
	raw.toggleOutput(2);

	glades::NumberInput di;
	di.import(raw, standardize);
	if (di.getTrainSize() == 0 || di.getFeatureCount() == 0)
	{
		printf("[bench] failed to load dataset or empty data: %s\n", dataset);
		return;
	}

	// Production semantics: RUN_TEST evaluates the test split.
	// Many benchmark datasets are single-split (train only). For benchmarking only,
	// mirror train->test so the eval pass can run.
	if (di.getTestSize() == 0)
	{
		di.testMatrix = di.trainMatrix;
		di.testExpectedMatrix = di.trainExpectedMatrix;
		printf("[bench] note: dataset has no test split; mirroring train->test for benchmarking\n");
	}

	printf("[bench] dataset=%s trainSize=%u features=%u epochs=%d hidden=%d repeats=%d lr=%f standardize=%s\n",
	       dataset, di.getTrainSize(), di.getFeatureCount(), epochs, hidden, repeats, lr,
	       (standardize == glades::GMath::NONE ? "none" : "minmax"));
	printf("[bench] variant: lrSchedule=%s gamma=%f stepSize=%d tMax=%d minMult=%f clipNorm=%f\n",
	       (variant.schedule ? variant.schedule : "none"),
	       variant.gamma, variant.stepSize, variant.tMax, variant.minMult, variant.clipNorm);

	print_header();

	std::vector<int> types;
	if (doAll || wantDff) types.push_back(glades::NNetwork::TYPE_DFF);
	if (doAll || wantRnn) types.push_back(glades::NNetwork::TYPE_RNN);
	if (doAll || wantGru) types.push_back(glades::NNetwork::TYPE_GRU);
	if (doAll || wantLstm) types.push_back(glades::NNetwork::TYPE_LSTM);

	for (size_t ti = 0; ti < types.size(); ++ti)
	{
		const int t = types[ti];
		long long trainSum = 0;
		long long testSum = 0;
		BenchResult last;
		bool okAll = true;

		for (int r = 0; r < repeats; ++r)
		{
			const BenchResult one = run_one(di, t, epochs, hidden, lr, 1234u + (unsigned int)r, variant);
			trainSum += one.trainMs;
			testSum += one.testMs;
			last = one;
			okAll = okAll && one.ok;
		}

		BenchResult avg = last;
		avg.trainMs = (repeats > 0) ? (trainSum / repeats) : trainSum;
		avg.testMs = (repeats > 0) ? (testSum / repeats) : testSum;
		avg.ok = okAll;
		print_row(avg);
	}
}

