#include "atlas-bench.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
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

class CaptureMetricsCallbacks : public glades::ITrainingCallbacks
{
public:
	CaptureMetricsCallbacks() : last(), saw(false) { last = glades::NNetworkEpochMetrics(); }
	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}

	glades::NNetworkEpochMetrics last;
	bool saw;
};

struct BenchResult
{
	const char* netType;
	const char* optimizer;
	long long trainMs;
	float finalLoss;
	bool ok;
	std::string err;
	BenchResult() : netType(""), optimizer(""), trainMs(0), finalLoss(0.0f), ok(true) {}
};

static BenchResult runBench(int netType, bool useAtlas,
                            int epochs, int hiddenSize, unsigned int atlasRank,
                            unsigned int seed)
{
	BenchResult r;
	r.netType = (netType == glades::NNetwork::TYPE_DFF) ? "DFF" : "RNN";
	r.optimizer = useAtlas ? "ATLAS" : "SGD";

	// Dataset: simple regression (sum of inputs)
	glades::NumberInput di;
	di.trainMatrix = shmea::GMatrix(8, shmea::GVector<float>(2, 0.0f));
	di.trainExpectedMatrix = shmea::GMatrix(8, shmea::GVector<float>(1, 0.0f));
	for (int i = 0; i < 8; ++i)
	{
		float x = static_cast<float>(i) / 8.0f;
		float y = static_cast<float>(i + 1) / 8.0f;
		di.trainMatrix[i][0] = x;
		di.trainMatrix[i][1] = y;
		di.trainExpectedMatrix[i][0] = (x + y) * 0.5f;
	}
	di.testMatrix = di.trainMatrix;
	di.testExpectedMatrix = di.trainExpectedMatrix;

	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);

	std::vector<glades::HiddenLayerInfo*> hidden;
	hidden.push_back(new glades::HiddenLayerInfo(
	    hiddenSize, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));

	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
	std::string name = std::string("bench_atlas_") + r.netType + "_" + r.optimizer;
	glades::NNInfo* info = new glades::NNInfo(name.c_str(), in, hidden, out);

	glades::NNetwork net(info, netType);
	net.setSeed(static_cast<uint64_t>(seed));
	net.getTerminatorMutable().setEpoch(epochs);
	net.getTerminatorMutable().setAccuracy(0);

	if (useAtlas)
	{
		glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
		cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
		cfg.atlas.rank = atlasRank;
		cfg.atlas.tSub = 50;
	}

	CaptureMetricsCallbacks cb;
	const int64_t t0 = now_ms();
	const glades::NNetworkStatus st = net.train(&di, &cb);
	const int64_t t1 = now_ms();

	r.trainMs = static_cast<long long>(t1 - t0);
	if (!st.ok())
	{
		r.ok = false;
		r.err = st.message;
	}
	if (cb.saw)
		r.finalLoss = cb.last.totalError;

	delete info;
	return r;
}
} // anonymous namespace

void ATLASBenchmark(int argc, char* argv[])
{
	printf("============================================================\n");
	printf("ATLAS vs SGD Benchmark\n");
	printf("============================================================\n");

	// Parse args
	int epochs = 200;
	int hiddenSize = 8;
	unsigned int atlasRank = 4;
	int repeats = 3;

	for (int i = 2; i < argc; ++i)
	{
		if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) epochs = atoi(argv[++i]);
		else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hiddenSize = atoi(argv[++i]);
		else if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) atlasRank = static_cast<unsigned int>(atoi(argv[++i]));
		else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) repeats = atoi(argv[++i]);
	}

	printf("Config: epochs=%d hidden=%d rank=%u repeats=%d\n\n", epochs, hiddenSize, atlasRank, repeats);

	// Header
	printf("Type\tOptimizer\tTrain(ms)\tFinalLoss\tStatus\n");
	printf("----\t---------\t---------\t---------\t------\n");

	int netTypes[] = { glades::NNetwork::TYPE_DFF, glades::NNetwork::TYPE_RNN };
	const char* typeNames[] = { "DFF", "RNN" };
	const int nTypes = 2;

	for (int t = 0; t < nTypes; ++t)
	{
		for (int useAtlas = 0; useAtlas <= 1; ++useAtlas)
		{
			long long totalMs = 0;
			float totalLoss = 0.0f;
			bool allOk = true;
			std::string lastErr;

			for (int rep = 0; rep < repeats; ++rep)
			{
				const unsigned int seed = static_cast<unsigned int>(42 + rep);
				BenchResult r = runBench(netTypes[t], (useAtlas != 0),
				                         epochs, hiddenSize, atlasRank, seed);
				totalMs += r.trainMs;
				totalLoss += r.finalLoss;
				if (!r.ok) { allOk = false; lastErr = r.err; }
			}

			const long long avgMs = totalMs / repeats;
			const float avgLoss = totalLoss / static_cast<float>(repeats);
			printf("%s\t%s\t\t%lld\t\t%.6f\t%s\n",
				typeNames[t],
				useAtlas ? "ATLAS" : "SGD",
				avgMs, avgLoss,
				allOk ? "OK" : lastErr.c_str());
		}
	}

	printf("\n");
}
