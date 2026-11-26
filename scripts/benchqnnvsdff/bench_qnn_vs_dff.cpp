// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Isolated QNN (TYPE_QNN) vs. DFNN (TYPE_DFF) benchmark worker.
// Designed to be invoked once per (task, model) in a fresh OS process by
// scripts/benchqnnvsdff.sh so peak RSS (ru_maxrss) and live heap delta
// (mallinfo2) reflect each model configuration in isolation.

#include "Backend/Machine Learning/main.h"
#include "Backend/Machine Learning/DataObjects/NumberInput.h"
#include "Backend/Machine Learning/DataObjects/QuaternionInput.h"
#include "Backend/Machine Learning/GMath/gmath.h"
#include "Backend/Machine Learning/GMath/quaternion.h"
#include "Backend/Machine Learning/Networks/network.h"
#include "Backend/Machine Learning/Networks/training_callbacks.h"
#include "Backend/Machine Learning/State/Terminator.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Database/GLogger.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

struct ModelSpec
{
	const char* id;
	const char* label;
	const char* task; // "xor", "iris80", "iris40"
	int netType;      // NNetwork::TYPE_QNN or NNetwork::TYPE_DFF
	unsigned int hiddenSize;
	glades::QuaternionConfig::OutputReadoutMode qnnReadout;
	glades::OptimizerConfig::Type optimizer;
	float lr;
	float momentum;
	float wd2;
	unsigned int epochs;
};

static const ModelSpec kSpecs[] = {
	// === XOR Gate (4 samples, batch=4, TANH -> SIGMOID, 20 seeds) ===
	{ "qnn_1_4_1_sgdm",       "QNN 1->4->1 (SGD+M)",             "xor",    glades::NNetwork::TYPE_QNN,  4u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.20f, 0.9f, 0.0f,  800u },
	{ "dff_2_16_1_sgdm",      "DFF 2->16->1 [width=] (SGD+M)",   "xor",    glades::NNetwork::TYPE_DFF, 16u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.20f, 0.9f, 0.0f,  800u },
	{ "dff_2_12_1_sgdm",      "DFF 2->12->1 [param=] (SGD+M)",   "xor",    glades::NNetwork::TYPE_DFF, 12u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.20f, 0.9f, 0.0f,  800u },
	{ "qnn_1_4_1_adamw",      "QNN 1->4->1 (AdamW)",             "xor",    glades::NNetwork::TYPE_QNN,  4u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::ADAMW,        0.05f, 0.0f, 0.0f,  800u },
	{ "qnn_1_1_1_sgdm",       "QNN 1->1->1 (SGD+M)",             "xor",    glades::NNetwork::TYPE_QNN,  1u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.20f, 0.9f, 0.0f, 1500u },
	{ "dff_2_4_1_sgdm",       "DFF 2->4->1 [width=] (SGD+M)",    "xor",    glades::NNetwork::TYPE_DFF,  4u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.20f, 0.9f, 0.0f, 1500u },
	{ "dff_2_3_1_sgdm",       "DFF 2->3->1 [param=] (SGD+M)",    "xor",    glades::NNetwork::TYPE_DFF,  3u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.20f, 0.9f, 0.0f, 1500u },
	{ "dff_2_2_1_sgdm",       "DFF 2->2->1 [min] (SGD+M)",       "xor",    glades::NNetwork::TYPE_DFF,  2u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.20f, 0.9f, 0.0f, 1500u },
	{ "qnn_1_1_1_adamw",      "QNN 1->1->1 (AdamW)",             "xor",    glades::NNetwork::TYPE_QNN,  1u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::ADAMW,        0.05f, 0.0f, 0.0f, 1500u },

	// === Iris 80/20 (120 train / 30 test, batch=16, 5 splits x 5 seeds) ===
	{ "qnn_1_6_3_real_sgdm",  "QNN 1->6->3 Real (SGD+M)",        "iris80", glades::NNetwork::TYPE_QNN,  6u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "qnn_1_6_1_split_sgdm", "QNN 1->6->1 Split (SGD+M)",       "iris80", glades::NNetwork::TYPE_QNN,  6u, glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS, glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "dff_4_24_3_sgdm",      "DFF 4->24->3 [width=] (SGD+M)",   "iris80", glades::NNetwork::TYPE_DFF, 24u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "dff_4_15_3_sgdm",      "DFF 4->15->3 [param=R] (SGD+M)",  "iris80", glades::NNetwork::TYPE_DFF, 15u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "dff_4_9_3_sgdm",       "DFF 4->9->3 [param=S] (SGD+M)",   "iris80", glades::NNetwork::TYPE_DFF,  9u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "dff_4_6_3_sgdm",       "DFF 4->6->3 [nodes=] (SGD+M)",    "iris80", glades::NNetwork::TYPE_DFF,  6u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "qnn_1_6_3_real_adamw", "QNN 1->6->3 Real (AdamW)",        "iris80", glades::NNetwork::TYPE_QNN,  6u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::ADAMW,        0.02f, 0.0f, 1e-4f, 300u },
	{ "qnn_1_6_1_split_adamw","QNN 1->6->1 Split (AdamW)",       "iris80", glades::NNetwork::TYPE_QNN,  6u, glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS, glades::OptimizerConfig::ADAMW,        0.02f, 0.0f, 1e-4f, 300u },
	{ "dff_4_15_3_sgdm_lr003","DFF 4->15->3 (SGD+M lr=0.03)",    "iris80", glades::NNetwork::TYPE_DFF, 15u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.03f, 0.9f, 1e-4f, 300u },

	// === Iris 40/60 (60 train / 90 test, batch=16, 5 splits x 5 seeds) ===
	{ "qnn_1_6_3_real_sgdm",  "QNN 1->6->3 Real (SGD+M)",        "iris40", glades::NNetwork::TYPE_QNN,  6u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "qnn_1_6_1_split_sgdm", "QNN 1->6->1 Split (SGD+M)",       "iris40", glades::NNetwork::TYPE_QNN,  6u, glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS, glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "dff_4_24_3_sgdm",      "DFF 4->24->3 [width=] (SGD+M)",   "iris40", glades::NNetwork::TYPE_DFF, 24u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "dff_4_15_3_sgdm",      "DFF 4->15->3 [param=R] (SGD+M)",  "iris40", glades::NNetwork::TYPE_DFF, 15u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "dff_4_9_3_sgdm",       "DFF 4->9->3 [param=S] (SGD+M)",   "iris40", glades::NNetwork::TYPE_DFF,  9u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "dff_4_6_3_sgdm",       "DFF 4->6->3 [nodes=] (SGD+M)",    "iris40", glades::NNetwork::TYPE_DFF,  6u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.10f, 0.9f, 1e-4f, 300u },
	{ "qnn_1_6_3_real_adamw", "QNN 1->6->3 Real (AdamW)",        "iris40", glades::NNetwork::TYPE_QNN,  6u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::ADAMW,        0.02f, 0.0f, 1e-4f, 300u },
	{ "qnn_1_6_1_split_adamw","QNN 1->6->1 Split (AdamW)",       "iris40", glades::NNetwork::TYPE_QNN,  6u, glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS, glades::OptimizerConfig::ADAMW,        0.02f, 0.0f, 1e-4f, 300u },
	{ "dff_4_15_3_sgdm_lr003","DFF 4->15->3 (SGD+M lr=0.03)",    "iris40", glades::NNetwork::TYPE_DFF, 15u, glades::QuaternionConfig::READOUT_REAL_PART,        glades::OptimizerConfig::SGD_MOMENTUM, 0.03f, 0.9f, 1e-4f, 300u }
};

static const size_t kNumSpecs = sizeof(kSpecs) / sizeof(kSpecs[0]);

static double now_seconds()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

static size_t current_heap_bytes()
{
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
	struct mallinfo2 mi = mallinfo2();
	return mi.uordblks;
#else
	struct mallinfo mi = mallinfo();
	return (mi.uordblks > 0) ? static_cast<size_t>(mi.uordblks) : 0u;
#endif
}

static long peak_rss_kib()
{
	struct rusage ru;
	if (getrusage(RUSAGE_SELF, &ru) == 0)
		return ru.ru_maxrss;
	return 0L;
}

static void update_heap_delta(size_t baseline, size_t& maxDelta)
{
	const size_t cur = current_heap_bytes();
	if (cur > baseline)
	{
		const size_t d = cur - baseline;
		if (d > maxDelta)
			maxDelta = d;
	}
}

static double vec_mean(const std::vector<double>& v)
{
	if (v.empty())
		return 0.0;
	double s = 0.0;
	for (size_t i = 0u; i < v.size(); ++i)
		s += v[i];
	return s / static_cast<double>(v.size());
}

static double vec_std(const std::vector<double>& v)
{
	if (v.empty())
		return 0.0;
	const double m = vec_mean(v);
	double s2 = 0.0;
	for (size_t i = 0u; i < v.size(); ++i)
	{
		const double d = v[i] - m;
		s2 += d * d;
	}
	return std::sqrt(s2 / static_cast<double>(v.size()));
}

static double vec_median(std::vector<double> v)
{
	if (v.empty())
		return 0.0;
	std::sort(v.begin(), v.end());
	const size_t n = v.size();
	if ((n & 1u) == 1u)
		return v[n / 2u];
	return 0.5 * (v[n / 2u - 1u] + v[n / 2u]);
}

static glades::NNInfo* make_net_info(const char* name,
                                     int batchSize,
                                     float lr,
                                     float momentum,
                                     float wd2,
                                     unsigned int hiddenSize,
                                     int hiddenAct,
                                     int outSize,
                                     int outAct)
{
	shmea::GPointer<glades::InputLayerInfo> in(new glades::InputLayerInfo(
	    batchSize, lr, momentum, 0.0f, wd2, 0.0f, hiddenAct, 1.0f));
	std::vector< shmea::GPointer<glades::HiddenLayerInfo> > hidden;
	hidden.push_back(shmea::GPointer<glades::HiddenLayerInfo>(new glades::HiddenLayerInfo(
	    static_cast<int>(hiddenSize), lr, momentum, 0.0f, wd2, 0.0f, outAct, 1.0f)));
	shmea::GPointer<glades::OutputLayerInfo> out(new glades::OutputLayerInfo(
	    outSize, glades::OutputLayerInfo::CLASSIFICATION));
	return new glades::NNInfo(shmea::GString(name), in, hidden, out);
}

static void compute_param_and_state_sizes(const ModelSpec& spec,
                                          unsigned int realIn,
                                          unsigned int realOut,
                                          unsigned int& outActiveParams,
                                          unsigned int& outStoredParams,
                                          unsigned int& outCoreStateBytes)
{
	if (spec.netType == glades::NNetwork::TYPE_QNN)
	{
		const unsigned int qIn = glades::quaternionInputCount(realIn, glades::QUAT_PACK_GROUPS_OF_4);
		const unsigned int qH = spec.hiddenSize;
		const glades::QuaternionOutputReadoutMode rMode =
		    (spec.qnnReadout == glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS)
		        ? glades::QUAT_READOUT_SPLIT_COMPONENTS
		        : glades::QUAT_READOUT_REAL_PART;
		const unsigned int qOut = glades::quaternionOutputCount(realOut, rMode);

		const unsigned int w0 = qH * qIn * 4u;
		const unsigned int b0 = qH * 4u;
		const unsigned int w1 = qOut * qH * 4u;
		const unsigned int b1Stored = qOut * 4u;
		const unsigned int b1Active = realOut;

		outStoredParams = w0 + b0 + w1 + b1Stored;
		outActiveParams = w0 + b0 + w1 + b1Active;

		// TensorQNNState: per transition W,vW,v2W,gW + bias,vBias,v2Bias,gBias (16 B/quat)
		// plus scratch z (layers 1..2), a (layers 0..2), delta (layers 0..2), outReadout (4 B/float).
		const unsigned int transBytes =
		    4u * (qH * qIn + qH) * 16u +
		    4u * (qOut * qH + qOut) * 16u;
		const unsigned int scratchBytes =
		    (qH + qOut) * 16u +
		    2u * (qIn + qH + qOut) * 16u +
		    realOut * 4u;
		outCoreStateBytes = transBytes + scratchBytes;
	}
	else
	{
		const unsigned int h = spec.hiddenSize;
		const unsigned int w0 = h * realIn;
		const unsigned int b0 = h;
		const unsigned int w1 = realOut * h;
		const unsigned int b1 = realOut;

		outStoredParams = w0 + b0 + w1 + b1;
		outActiveParams = outStoredParams;

		// TensorDFFState core (excluding helm/aster): per transition W,vW,gW + bias,gBias (4 B/float)
		// plus scratch a (layers 0..2) and delta (layers 1..2).
		const unsigned int transBytes =
		    (3u * w0 + 2u * b0 + 3u * w1 + 2u * b1) * 4u;
		const unsigned int scratchBytes =
		    (realIn + h + realOut) * 4u +
		    (h + realOut) * 4u;
		outCoreStateBytes = transBytes + scratchBytes;
	}
}

static size_t measure_weights_bin_bytes(glades::NNetwork& net, glades::DataInput* di)
{
	char nameBuf[128];
	std::sprintf(nameBuf, "bench_qnn_tmp_%ld", static_cast<long>(getpid()));
	const std::string modelName(nameBuf);
	const std::string dir = "database/models/" + modelName;

	// Ensure database/models exists in the isolated working directory.
	(void)::mkdir("database", 0755);
	(void)::mkdir("database/models", 0755);

	glades::NNetworkStatus st = net.saveModel(modelName, di);
	size_t bytes = 0u;
	if (st.ok())
	{
		struct stat sb;
		const std::string wpath = dir + "/weights.bin";
		if (::stat(wpath.c_str(), &sb) == 0 && sb.st_size > 0)
			bytes = static_cast<size_t>(sb.st_size);
	}

	(void)::remove((dir + "/manifest.txt").c_str());
	(void)::remove((dir + "/nninfo.csv").c_str());
	(void)::remove((dir + "/weights.bin").c_str());
	(void)::rmdir(dir.c_str());
	return bytes;
}

struct EpochTracker : public glades::ITrainingCallbacks
{
	unsigned int numClasses;
	bool isXor;
	size_t heapBaseline;
	size_t* maxHeapDelta;

	unsigned int epBce017;
	unsigned int epTrain95;
	unsigned int epTrain98;
	double lastError;

	EpochTracker(unsigned int nCls, bool xorTask, size_t base, size_t* maxDelta)
	    : numClasses(nCls),
	      isXor(xorTask),
	      heapBaseline(base),
	      maxHeapDelta(maxDelta),
	      epBce017(0u),
	      epTrain95(0u),
	      epTrain98(0u),
	      lastError(0.0)
	{
	}

	virtual void onRunStart(const glades::NNetwork&, int)
	{
		if (maxHeapDelta)
			update_heap_delta(heapBaseline, *maxHeapDelta);
	}

	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		lastError = static_cast<double>(m.totalError);
		const unsigned int ep = static_cast<unsigned int>(m.epoch);
		if (isXor)
		{
			if (epBce017 == 0u && m.totalError <= 0.17f)
				epBce017 = ep;
		}
		else
		{
			// Exact Top-1 accuracy from Glades one-vs-rest confusion-matrix accuracy:
			// ovr = 100 - (2/C)*(100 - top1) => top1 = 100 - (C/2)*(100 - ovr).
			const double top1 = 100.0 - 0.5 * static_cast<double>(numClasses) * (100.0 - static_cast<double>(m.totalAccuracy));
			if (epTrain95 == 0u && top1 >= 95.0)
				epTrain95 = ep;
			if (epTrain98 == 0u && top1 >= 98.0)
				epTrain98 = ep;
		}
		if (maxHeapDelta && (ep == 1u || ep == 10u || (ep % 100u) == 0u))
			update_heap_delta(heapBaseline, *maxHeapDelta);
		return false;
	}

	virtual void onRunEnd(const glades::NNetwork&, int)
	{
		if (maxHeapDelta)
			update_heap_delta(heapBaseline, *maxHeapDelta);
	}
};

struct EvalStats
{
	double accPct;
	double loss;
	double brier;
	double f1[3];
	double macroF1;
	unsigned int correct;
	unsigned int total;

	EvalStats() : accPct(0.0), loss(0.0), brier(0.0), macroF1(0.0), correct(0u), total(0u)
	{
		f1[0] = f1[1] = f1[2] = 0.0;
	}
};

static bool eval_split_exact(glades::NNetwork& net,
                             const glades::NumberInput& di,
                             bool useTrainSplit,
                             const unsigned int* classOrderMap, // canonical index -> raw class index (or NULL for identity)
                             EvalStats& out)
{
	const unsigned int nRows = useTrainSplit ? di.getTrainSize() : di.getTestSize();
	const unsigned int outSize = net.getNNInfo()->getOutputLayerSize();
	const unsigned int nClasses = (outSize > 1u) ? outSize : 2u;
	if (nRows == 0u || nClasses > 3u)
		return false;

	unsigned int cm[3][3];
	for (unsigned int i = 0u; i < 3u; ++i)
		for (unsigned int j = 0u; j < 3u; ++j)
			cm[i][j] = 0u;

	unsigned int correct = 0u;
	double lossSum = 0.0;
	double brierSum = 0.0;
	const double eps = 1e-7;

	glades::NumberInput single;
	single.testMatrix = shmea::GMatrix(1, shmea::GVector<float>(di.getFeatureCount(), 0.0f));
	single.testExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(outSize, 0.0f));
	single.trainMatrix = single.testMatrix;
	single.trainExpectedMatrix = single.testExpectedMatrix;
	single.loaded = true;

	for (unsigned int r = 0u; r < nRows; ++r)
	{
		const float* x = NULL;
		unsigned int xN = 0u;
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
		{
			di.getTrainRowView(r, x, xN);
			di.getTrainExpectedRowView(r, y, yN);
		}
		else
		{
			di.getTestRowView(r, x, xN);
			di.getTestExpectedRowView(r, y, yN);
		}
		for (unsigned int c = 0u; c < xN; ++c)
		{
			single.testMatrix[0][c] = x[c];
			single.trainMatrix[0][c] = x[c];
		}
		for (unsigned int c = 0u; c < yN; ++c)
		{
			single.testExpectedMatrix[0][c] = y[c];
			single.trainExpectedMatrix[0][c] = y[c];
		}

		glades::NNetworkStatus st = net.test(&single);
		if (!st.ok())
			return false;
		shmea::GList res = net.getResults();
		if (res.size() != outSize * 2u)
			return false;

		double probs[3];
		for (unsigned int k = 0u; k < outSize; ++k)
		{
			const double p = static_cast<double>(res.getFloat(k * 2u + 1u));
			if (!std::isfinite(p))
				return false;
			probs[k] = p;
		}

		unsigned int expRaw = 0u;
		unsigned int predRaw = 0u;
		if (outSize == 1u)
		{
			const double p1 = std::min(1.0 - eps, std::max(eps, probs[0]));
			const double yt = (y[0] >= 0.5f) ? 1.0 : 0.0;
			expRaw = (yt >= 0.5) ? 1u : 0u;
			predRaw = (probs[0] >= 0.5) ? 1u : 0u;
			lossSum += -(yt * std::log(p1) + (1.0 - yt) * std::log(1.0 - p1));
			const double diff = probs[0] - yt;
			brierSum += diff * diff;
		}
		else
		{
			expRaw = static_cast<unsigned int>(glades::GMath::argmax(y, outSize));
			predRaw = 0u;
			for (unsigned int k = 1u; k < outSize; ++k)
			{
				if (probs[k] > probs[predRaw])
					predRaw = k;
			}
			const double pTrue = std::min(1.0 - eps, std::max(eps, probs[expRaw]));
			lossSum += -std::log(pTrue);
			for (unsigned int k = 0u; k < outSize; ++k)
			{
				const double yt = (k == expRaw) ? 1.0 : 0.0;
				const double diff = probs[k] - yt;
				brierSum += diff * diff;
			}
		}

		if (expRaw == predRaw)
			++correct;
		if (expRaw < nClasses && predRaw < nClasses)
			++cm[expRaw][predRaw];
	}

	out.correct = correct;
	out.total = nRows;
	out.accPct = (100.0 * static_cast<double>(correct)) / static_cast<double>(nRows);
	out.loss = lossSum / static_cast<double>(nRows);
	out.brier = brierSum / static_cast<double>(nRows);

	double f1Sum = 0.0;
	for (unsigned int cCanon = 0u; cCanon < nClasses; ++cCanon)
	{
		const unsigned int cRaw = classOrderMap ? classOrderMap[cCanon] : cCanon;
		const unsigned int tp = cm[cRaw][cRaw];
		unsigned int fp = 0u;
		unsigned int fn = 0u;
		for (unsigned int j = 0u; j < nClasses; ++j)
		{
			if (j != cRaw)
			{
				fp += cm[j][cRaw];
				fn += cm[cRaw][j];
			}
		}
		const unsigned int denom = 2u * tp + fp + fn;
		const double f1Val = (denom > 0u) ? (100.0 * 2.0 * static_cast<double>(tp) / static_cast<double>(denom)) : 0.0;
		out.f1[cCanon] = f1Val;
		f1Sum += f1Val;
	}
	out.macroF1 = f1Sum / static_cast<double>(nClasses);
	return true;
}

static void resolve_iris_class_map(const glades::QuaternionInput& di, unsigned int outMap[3])
{
	outMap[0] = 0u;
	outMap[1] = 1u;
	outMap[2] = 2u;
	const std::vector<std::string>& labels = di.getClassLabels();
	const char* canonical[3] = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
	for (unsigned int c = 0u; c < 3u; ++c)
	{
		for (size_t i = 0u; i < labels.size(); ++i)
		{
			if (labels[i] == canonical[c])
			{
				outMap[c] = static_cast<unsigned int>(i);
				break;
			}
		}
	}
}

static void print_tsv_header()
{
	std::printf(
	    "task\tmodel_id\tlabel\tactive_params\tstored_params\tbudget_epochs\t"
	    "runs\tconverged_runs\ttrain_acc_mean\ttrain_acc_std\ttest_acc_mean\ttest_acc_std\tcanon_test_acc\t"
	    "ep_metric1_mean\tep_metric1_med\tep_metric2_mean\t"
	    "train_loss_mean\ttest_loss_mean\ttest_loss_std\ttest_brier_mean\t"
	    "f1_c0\tf1_c1\tf1_c2\tmacro_f1_mean\tmacro_f1_std\t"
	    "train_ms_mean\ttrain_ms_std\tpredict_split_us\tpredict_sample_us\t"
	    "peak_heap_delta_kib\tcore_state_bytes\tweights_bin_bytes\tpeak_rss_kib\n");
}

static void print_list()
{
	for (size_t i = 0u; i < kNumSpecs; ++i)
		std::printf("%s\t%s\t%s\n", kSpecs[i].task, kSpecs[i].id, kSpecs[i].label);
}

static int run_worker(const ModelSpec& spec, const std::string& dataDir, unsigned int predictRepeats)
{
	shmea::GLogger quietLogger(shmea::GLogger::LOG_FATAL);
	quietLogger.setPrintToConsole(false);

	const bool isXor = (std::strcmp(spec.task, "xor") == 0);
	const bool isIris80 = (std::strcmp(spec.task, "iris80") == 0);
	const unsigned int realIn = isXor ? 2u : 4u;
	const unsigned int realOut = isXor ? 1u : 3u;
	const int batchSize = isXor ? 4 : 16;
	const int outAct = isXor ? glades::GMath::SIGMOID : glades::GMath::LINEAR;

	unsigned int activeParams = 0u;
	unsigned int storedParams = 0u;
	unsigned int coreStateBytes = 0u;
	compute_param_and_state_sizes(spec, realIn, realOut, activeParams, storedParams, coreStateBytes);

	std::vector<double> trainAccs;
	std::vector<double> testAccs;
	std::vector<double> canonTestAccs;
	std::vector<double> epM1;
	std::vector<double> epM2;
	std::vector<double> trainLosses;
	std::vector<double> testLosses;
	std::vector<double> testBriers;
	std::vector<double> f1C0;
	std::vector<double> f1C1;
	std::vector<double> f1C2;
	std::vector<double> macroF1s;
	std::vector<double> trainMs;
	std::vector<double> predictSplitUs;

	unsigned int convergedRuns = 0u;
	size_t maxHeapDeltaBytes = 0u;
	size_t weightsBinBytes = 0u;
	unsigned int splitTestRows = 0u;

	if (isXor)
	{
		glades::QuaternionInput xorDi;
		const std::string path = dataDir + "/xorgate.csv";
		if (!xorDi.loadXOR(path, glades::QUAT_PACK_GROUPS_OF_4, glades::QUAT_READOUT_REAL_PART))
		{
			std::fprintf(stderr, "ERROR: failed to load %s\n", path.c_str());
			return 2;
		}
		splitTestRows = xorDi.getTestSize();

		for (unsigned int s = 0u; s < 20u; ++s)
		{
			const uint64_t seed = 100ULL + static_cast<uint64_t>(s);
			const size_t heapBaseline = current_heap_bytes();
			glades::NNInfo* info = make_net_info(
			    spec.id, batchSize, spec.lr, spec.momentum, spec.wd2,
			    spec.hiddenSize, glades::GMath::TANH, static_cast<int>(realOut), outAct);
			glades::NNetwork net(info, spec.netType);
			delete info;
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			net.setLogger(&quietLogger);
			net.setSeed(seed);
			glades::TrainingConfig cfg = net.getTrainingConfig();
			cfg.optimizer.type = spec.optimizer;
			cfg.qnn.inputPacking = glades::QuaternionConfig::PACK_GROUPS_OF_4;
			cfg.qnn.outputReadout = spec.qnnReadout;
			if (!net.setTrainingConfig(cfg).ok())
			{
				std::fprintf(stderr, "ERROR: setTrainingConfig failed for %s\n", spec.id);
				return 3;
			}

			glades::Terminator term;
			term.setEpoch(static_cast<int>(spec.epochs));
			if (!net.setTerminator(term).ok())
				return 3;

			EpochTracker cb(2u, true, heapBaseline, &maxHeapDeltaBytes);
			const double t0 = now_seconds();
			glades::NNetworkStatus st = net.train(&xorDi, &cb);
			const double t1 = now_seconds();
			if (!st.ok())
			{
				std::fprintf(stderr, "ERROR: train failed for %s seed=%llu: %s\n",
				             spec.id, static_cast<unsigned long long>(seed), st.message.c_str());
				return 4;
			}
			trainMs.push_back((t1 - t0) * 1000.0);
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			// Full-split prediction latency timing.
			const double p0 = now_seconds();
			for (unsigned int rep = 0u; rep < predictRepeats; ++rep)
			{
				st = net.test(&xorDi);
				if (!st.ok())
					return 5;
			}
			const double p1 = now_seconds();
			predictSplitUs.push_back(((p1 - p0) * 1e6) / static_cast<double>(predictRepeats));
			update_heap_delta(heapBaseline, maxHeapDeltaBytes);

			EvalStats ev;
			if (!eval_split_exact(net, xorDi, false, NULL, ev))
				return 6;

			trainAccs.push_back(ev.accPct);
			testAccs.push_back(ev.accPct);
			canonTestAccs.push_back(ev.accPct);
			trainLosses.push_back(ev.loss);
			testLosses.push_back(ev.loss);
			testBriers.push_back(ev.brier);
			f1C0.push_back(ev.f1[0]);
			f1C1.push_back(ev.f1[1]);
			f1C2.push_back(0.0);
			macroF1s.push_back(ev.macroF1);
			if (ev.correct == ev.total)
				++convergedRuns;
			if (cb.epBce017 > 0u)
				epM1.push_back(static_cast<double>(cb.epBce017));

			if (s == 19u)
				weightsBinBytes = measure_weights_bin_bytes(net, &xorDi);
		}
	}
	else
	{
		const uint64_t splitSeeds[5] = { 42ULL, 1337ULL, 2024ULL, 7777ULL, 9001ULL };
		const uint64_t trainSeeds[5] = { 42ULL, 1337ULL, 2024ULL, 7777ULL, 9001ULL };
		const float testFrac = isIris80 ? 0.2f : 0.6f;
		const std::string path = dataDir + "/iris.data";
		const glades::QuaternionOutputReadoutMode rMode =
		    (spec.qnnReadout == glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS)
		        ? glades::QUAT_READOUT_SPLIT_COMPONENTS
		        : glades::QUAT_READOUT_REAL_PART;

		for (unsigned int sp = 0u; sp < 5u; ++sp)
		{
			glades::NumberInput::TrainTestSplitConfig splitCfg;
			splitCfg.testFraction = testFrac;
			splitCfg.shuffle = true;
			splitCfg.stratify = true;
			splitCfg.seed = splitSeeds[sp];

			glades::QuaternionInput irisDi;
			if (!irisDi.loadIris(path, splitCfg, glades::GMath::ZSCORE, glades::QUAT_PACK_GROUPS_OF_4, rMode))
			{
				std::fprintf(stderr, "ERROR: failed to load %s\n", path.c_str());
				return 2;
			}
			splitTestRows = irisDi.getTestSize();
			unsigned int classMap[3];
			resolve_iris_class_map(irisDi, classMap);

			for (unsigned int ts = 0u; ts < 5u; ++ts)
			{
				const size_t heapBaseline = current_heap_bytes();
				glades::NNInfo* info = make_net_info(
				    spec.id, batchSize, spec.lr, spec.momentum, spec.wd2,
				    spec.hiddenSize, glades::GMath::TANH, static_cast<int>(realOut), outAct);
				glades::NNetwork net(info, spec.netType);
				delete info;
				update_heap_delta(heapBaseline, maxHeapDeltaBytes);

				net.setLogger(&quietLogger);
				net.setSeed(trainSeeds[ts]);
				glades::TrainingConfig cfg = net.getTrainingConfig();
				cfg.optimizer.type = spec.optimizer;
				cfg.qnn.inputPacking = glades::QuaternionConfig::PACK_GROUPS_OF_4;
				cfg.qnn.outputReadout = spec.qnnReadout;
				if (!net.setTrainingConfig(cfg).ok())
				{
					std::fprintf(stderr, "ERROR: setTrainingConfig failed for %s\n", spec.id);
					return 3;
				}

				glades::Terminator term;
				term.setEpoch(static_cast<int>(spec.epochs));
				if (!net.setTerminator(term).ok())
					return 3;

				EpochTracker cb(3u, false, heapBaseline, &maxHeapDeltaBytes);
				const double t0 = now_seconds();
				glades::NNetworkStatus st = net.train(&irisDi, &cb);
				const double t1 = now_seconds();
				if (!st.ok())
				{
					std::fprintf(stderr, "ERROR: train failed for %s split=%u seed=%u: %s\n",
					             spec.id, sp, ts, st.message.c_str());
					return 4;
				}
				trainMs.push_back((t1 - t0) * 1000.0);
				update_heap_delta(heapBaseline, maxHeapDeltaBytes);

				// Full-split prediction latency timing.
				const double p0 = now_seconds();
				for (unsigned int rep = 0u; rep < predictRepeats; ++rep)
				{
					st = net.test(&irisDi);
					if (!st.ok())
						return 5;
				}
				const double p1 = now_seconds();
				predictSplitUs.push_back(((p1 - p0) * 1e6) / static_cast<double>(predictRepeats));
				update_heap_delta(heapBaseline, maxHeapDeltaBytes);

				EvalStats trainEv;
				EvalStats testEv;
				if (!eval_split_exact(net, irisDi, true, classMap, trainEv))
					return 6;
				if (!eval_split_exact(net, irisDi, false, classMap, testEv))
					return 6;

				trainAccs.push_back(trainEv.accPct);
				testAccs.push_back(testEv.accPct);
				if (sp == 0u)
					canonTestAccs.push_back(testEv.accPct);
				trainLosses.push_back(trainEv.loss);
				testLosses.push_back(testEv.loss);
				testBriers.push_back(testEv.brier);
				f1C0.push_back(testEv.f1[0]);
				f1C1.push_back(testEv.f1[1]);
				f1C2.push_back(testEv.f1[2]);
				macroF1s.push_back(testEv.macroF1);
				if (testEv.accPct >= 93.0)
					++convergedRuns;

				epM1.push_back(cb.epTrain95 > 0u ? static_cast<double>(cb.epTrain95) : static_cast<double>(spec.epochs));
				epM2.push_back(cb.epTrain98 > 0u ? static_cast<double>(cb.epTrain98) : static_cast<double>(spec.epochs));

				if (sp == 4u && ts == 4u)
					weightsBinBytes = measure_weights_bin_bytes(net, &irisDi);
			}
		}
	}

	const double meanPredSplitUs = vec_mean(predictSplitUs);
	const double meanPredSampleUs = (splitTestRows > 0u) ? (meanPredSplitUs / static_cast<double>(splitTestRows)) : 0.0;
	const double peakHeapKib = static_cast<double>(maxHeapDeltaBytes) / 1024.0;
	const long rssKib = peak_rss_kib();

	std::printf(
	    "%s\t%s\t%s\t%u\t%u\t%u\t"
	    "%u\t%u\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t"
	    "%.2f\t%.2f\t%.2f\t"
	    "%.6f\t%.6f\t%.6f\t%.6f\t"
	    "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t"
	    "%.4f\t%.4f\t%.4f\t%.4f\t"
	    "%.2f\t%u\t%lu\t%ld\n",
	    spec.task,
	    spec.id,
	    spec.label,
	    activeParams,
	    storedParams,
	    spec.epochs,
	    static_cast<unsigned int>(testAccs.size()),
	    convergedRuns,
	    vec_mean(trainAccs),
	    vec_std(trainAccs),
	    vec_mean(testAccs),
	    vec_std(testAccs),
	    vec_mean(canonTestAccs),
	    vec_mean(epM1),
	    vec_median(epM1),
	    vec_mean(epM2),
	    vec_mean(trainLosses),
	    vec_mean(testLosses),
	    vec_std(testLosses),
	    vec_mean(testBriers),
	    vec_mean(f1C0),
	    vec_mean(f1C1),
	    vec_mean(f1C2),
	    vec_mean(macroF1s),
	    vec_std(macroF1s),
	    vec_mean(trainMs),
	    vec_std(trainMs),
	    meanPredSplitUs,
	    meanPredSampleUs,
	    peakHeapKib,
	    coreStateBytes,
	    static_cast<unsigned long>(weightsBinBytes),
	    rssKib);

	return 0;
}

} // namespace

int main(int argc, char** argv)
{
	glades::init();

	std::string task;
	std::string modelId;
	std::string dataDir = "unit-tests/datasets";
	unsigned int predictRepeats = 0u;
	bool doList = false;
	bool doHeader = false;

	for (int i = 1; i < argc; ++i)
	{
		if (std::strcmp(argv[i], "--list") == 0)
		{
			doList = true;
		}
		else if (std::strcmp(argv[i], "--tsv-header") == 0)
		{
			doHeader = true;
		}
		else if (std::strcmp(argv[i], "--task") == 0 && i + 1 < argc)
		{
			task = argv[++i];
		}
		else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
		{
			modelId = argv[++i];
		}
		else if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc)
		{
			dataDir = argv[++i];
		}
		else if (std::strcmp(argv[i], "--predict-repeats") == 0 && i + 1 < argc)
		{
			predictRepeats = static_cast<unsigned int>(std::atoi(argv[++i]));
		}
		else
		{
			std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
			return 1;
		}
	}

	if (doList)
	{
		print_list();
		return 0;
	}
	if (doHeader)
	{
		print_tsv_header();
		return 0;
	}

	if (task.empty() || modelId.empty())
	{
		std::fprintf(stderr, "Usage: %s --task <xor|iris80|iris40> --model <id> [--data-dir PATH] [--predict-repeats N]\n", argv[0]);
		return 1;
	}

	for (size_t i = 0u; i < kNumSpecs; ++i)
	{
		if (task == kSpecs[i].task && modelId == kSpecs[i].id)
		{
			if (predictRepeats == 0u)
				predictRepeats = (task == "xor") ? 100u : 50u;
			return run_worker(kSpecs[i], dataDir, predictRepeats);
		}
	}

	std::fprintf(stderr, "ERROR: unknown (task, model) pair: (%s, %s)\n", task.c_str(), modelId.c_str());
	return 1;
}
