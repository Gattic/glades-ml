#include "atlas-bench.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace {

static int64_t now_ms()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return static_cast<int64_t>(tv.tv_sec) * 1000LL + static_cast<int64_t>(tv.tv_usec) / 1000LL;
}

static bool streq(const char* a, const char* b)
{
	return (a && b && strcmp(a, b) == 0);
}

static bool is_finite(float v)
{
	return (v == v) && ((v - v) == 0.0f);
}

static bool path_exists(const std::string& path)
{
	struct stat st;
	return (::stat(path.c_str(), &st) == 0);
}

static bool is_directory(const std::string& path)
{
	struct stat st;
	if (::stat(path.c_str(), &st) != 0)
		return false;
	return S_ISDIR(st.st_mode) != 0;
}

static std::string join_path(const std::string& a, const std::string& b)
{
	if (b.empty())
		return a;
	if (!b.empty() && b[0] == '/')
		return b;
	if (a.empty())
		return b;
	if (a[a.size() - 1u] == '/')
		return a + b;
	return a + "/" + b;
}

static std::string to_lower_copy(const std::string& s)
{
	std::string out = s;
	for (size_t i = 0; i < out.size(); ++i)
	{
		if (out[i] >= 'A' && out[i] <= 'Z')
			out[i] = static_cast<char>(out[i] - 'A' + 'a');
	}
	return out;
}

static bool has_image_manifests(const std::string& dir)
{
	return is_directory(dir) &&
	       path_exists(join_path(dir, "train.csv")) &&
	       path_exists(join_path(dir, "test.csv"));
}

static bool parse_uint_arg(const char* text, unsigned int& outValue)
{
	if (!text || !*text)
		return false;
	char* end = NULL;
	const unsigned long v = strtoul(text, &end, 10);
	if (!end || *end != '\0')
		return false;
	outValue = static_cast<unsigned int>(v);
	return true;
}

static bool parse_float_arg(const char* text, float& outValue)
{
	if (!text || !*text)
		return false;
	char* end = NULL;
	const double v = strtod(text, &end);
	if (!end || *end != '\0')
		return false;
	outValue = static_cast<float>(v);
	return true;
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

struct BenchConfig
{
	std::string datasetRequest;
	unsigned int trainLimit;
	unsigned int testLimit;
	unsigned int epochs;
	unsigned int repeats;
	unsigned int batchSize;
	unsigned int atlasRank;
	unsigned int atlasTSub;
	unsigned int seed;
	float clipNorm;
	float sgdLR;
	float sgdMomentum;
	float adamLR;
	float atlasLR;

	BenchConfig()
	    : datasetRequest("auto"),
	      trainLimit(5000u),
	      testLimit(1000u),
	      epochs(5u),
	      repeats(2u),
	      batchSize(64u),
	      atlasRank(16u),
	      atlasTSub(200u),
	      seed(1337u),
	      clipNorm(5.0f),
	      sgdLR(0.01f),
	      sgdMomentum(0.9f),
	      adamLR(0.001f),
	      atlasLR(0.1f)
	{
	}
};

struct DatasetInfo
{
	std::string resolvedDir;
	std::string datasetName;
	std::string note;
	unsigned int originalTrainSize;
	unsigned int originalTestSize;

	DatasetInfo()
	    : resolvedDir(),
	      datasetName(),
	      note(),
	      originalTrainSize(0u),
	      originalTestSize(0u)
	{
	}
};

struct SingleRunResult
{
	const char* optimizerLabel;
	long long trainMs;
	long long testMs;
	float trainLoss;
	float trainAcc;
	float testLoss;
	float testAcc;
	double trainImagesPerSec;
	bool ok;
	std::string err;

	SingleRunResult()
	    : optimizerLabel(""),
	      trainMs(0LL),
	      testMs(0LL),
	      trainLoss(0.0f),
	      trainAcc(0.0f),
	      testLoss(0.0f),
	      testAcc(0.0f),
	      trainImagesPerSec(0.0),
	      ok(true),
	      err()
	{
	}
};

struct AggregateStats
{
	double mean;
	double stddev;
	AggregateStats() : mean(0.0), stddev(0.0) {}
};

struct BenchmarkSummary
{
	const char* label;
	AggregateStats trainSec;
	AggregateStats imgPerSec;
	AggregateStats trainLoss;
	AggregateStats trainAcc;
	AggregateStats testLoss;
	AggregateStats testAcc;
	bool ok;
	std::string status;

	BenchmarkSummary()
	    : label(""),
	      trainSec(),
	      imgPerSec(),
	      trainLoss(),
	      trainAcc(),
	      testLoss(),
	      testAcc(),
	      ok(false),
	      status()
	{
	}
};

enum OptimizerVariant
{
	OPT_SGD = 0,
	OPT_ADAMW = 1,
	OPT_ATLAS_BRSP = 2
};

static void print_usage()
{
	printf("Usage: glades-unit-tests atlas-bench [options]\n");
	printf("Options:\n");
	printf("  --dataset auto|mnist|mnist-small|PATH   Dataset directory with train.csv/test.csv (default: auto)\n");
	printf("  --train-limit N                         Class-balanced train subset size; 0 = full split (default: 5000)\n");
	printf("  --test-limit N                          Class-balanced test subset size; 0 = full split (default: 1000)\n");
	printf("  --epochs N                              Training epochs per run (default: 5)\n");
	printf("  --repeats N                             Repeats per optimizer (default: 2)\n");
	printf("  --batch-size N                          Mini-batch size (default: 64)\n");
	printf("  --clip-norm X                           Global grad clip norm (default: 5.0)\n");
	printf("  --sgd-lr X                              SGD learning rate (default: 0.01)\n");
	printf("  --sgd-momentum X                        SGD momentum factor (default: 0.9)\n");
	printf("  --adam-lr X                             AdamW learning rate (default: 0.001)\n");
	printf("  --atlas-lr X                            ATLAS-BSRP learning rate (default: 0.1)\n");
	printf("  --rank N                                ATLAS subspace rank (default: 16)\n");
	printf("  --tsub N                                ATLAS subspace refresh interval in steps (default: 200)\n");
	printf("  --seed N                                Base seed for repeats (default: 1337)\n");
	printf("  --help                                  Show this message\n");
}

static bool find_named_dataset(const char* leafName, std::string& outDir)
{
	std::vector<std::string> candidates;
	candidates.push_back(std::string("datasets/images/") + leafName);
	candidates.push_back(std::string("unit-tests/datasets/images/") + leafName);
	candidates.push_back(std::string("../datasets/images/") + leafName);
	candidates.push_back(std::string("../unit-tests/datasets/images/") + leafName);
	candidates.push_back(std::string("../../datasets/images/") + leafName);
	candidates.push_back(std::string("../../unit-tests/datasets/images/") + leafName);

	for (size_t i = 0; i < candidates.size(); ++i)
	{
		if (has_image_manifests(candidates[i]))
		{
			outDir = candidates[i];
			return true;
		}
	}
	return false;
}

static bool resolve_dataset_info(const std::string& request, DatasetInfo& outInfo, std::string& outErr)
{
	outInfo = DatasetInfo();
	outErr.clear();

	const std::string lower = to_lower_copy(request);
	if (lower == "auto")
	{
		if (find_named_dataset("MNIST", outInfo.resolvedDir))
		{
			outInfo.datasetName = "MNIST";
			return true;
		}
		if (find_named_dataset("MNIST_small", outInfo.resolvedDir))
		{
			outInfo.datasetName = "MNIST_small";
			outInfo.note = "Full MNIST not found; using MNIST_small fallback.";
			return true;
		}
		outErr = "Could not find MNIST or MNIST_small under the repo/unit-tests datasets.";
		return false;
	}

	if (lower == "mnist")
	{
		if (find_named_dataset("MNIST", outInfo.resolvedDir))
		{
			outInfo.datasetName = "MNIST";
			return true;
		}
		outErr = "Requested dataset 'mnist' was not found.";
		return false;
	}

	if (lower == "mnist-small" || lower == "mnist_small")
	{
		if (find_named_dataset("MNIST_small", outInfo.resolvedDir))
		{
			outInfo.datasetName = "MNIST_small";
			return true;
		}
		outErr = "Requested dataset 'mnist-small' was not found.";
		return false;
	}

	if (has_image_manifests(request))
	{
		outInfo.resolvedDir = request;
		outInfo.datasetName = request;
		return true;
	}

	if (find_named_dataset(request.c_str(), outInfo.resolvedDir))
	{
		outInfo.datasetName = request;
		return true;
	}

	outErr = std::string("Dataset path does not contain train.csv/test.csv: ") + request;
	return false;
}

static shmea::GTable table_from_indices(const shmea::GTable& src, const std::vector<unsigned int>& idx)
{
	shmea::GTable out(src.getDelimiter(), src.getHeaders());
	for (unsigned int c = 0; c < src.numberOfCols(); ++c)
	{
		if (src.isOutput(c))
			out.toggleOutput(c);
	}
	for (size_t i = 0; i < idx.size(); ++i)
		out.addRow(src.getRow(idx[i]));
	return out;
}

static std::vector<unsigned int> make_balanced_indices(const shmea::GTable& legend, unsigned int limit)
{
	const unsigned int rowCount = legend.numberOfRows();
	std::vector<unsigned int> out;
	if (limit == 0u || limit >= rowCount)
	{
		out.reserve(rowCount);
		for (unsigned int i = 0; i < rowCount; ++i)
			out.push_back(i);
		return out;
	}

	std::map<std::string, std::vector<unsigned int> > byLabel;
	for (unsigned int i = 0; i < rowCount; ++i)
	{
		const std::string label = legend.getCell(i, 1).c_str();
		byLabel[label].push_back(i);
	}

	std::vector<std::string> labels;
	for (std::map<std::string, std::vector<unsigned int> >::const_iterator it = byLabel.begin();
	     it != byLabel.end(); ++it)
	{
		labels.push_back(it->first);
	}

	std::vector<size_t> cursor(labels.size(), 0u);
	out.reserve(limit);
	while (out.size() < limit)
	{
		bool madeProgress = false;
		for (size_t i = 0; i < labels.size() && out.size() < limit; ++i)
		{
			const std::vector<unsigned int>& bucket = byLabel[labels[i]];
			if (cursor[i] >= bucket.size())
				continue;
			out.push_back(bucket[cursor[i]]);
			++cursor[i];
			madeProgress = true;
		}
		if (!madeProgress)
			break;
	}

	return out;
}

static bool load_image_dataset(glades::ImageInput& di,
                               DatasetInfo& datasetInfo,
                               unsigned int trainLimit,
                               unsigned int testLimit,
                               std::string& outErr)
{
	outErr.clear();

	const std::string trainManifest = join_path(datasetInfo.resolvedDir, "train.csv");
	const std::string testManifest = join_path(datasetInfo.resolvedDir, "test.csv");

	const shmea::GTable trainRaw(shmea::GString(trainManifest.c_str()), ',', shmea::GTable::TYPE_FILE_STRINGS_ONLY);
	const shmea::GTable testRaw(shmea::GString(testManifest.c_str()), ',', shmea::GTable::TYPE_FILE_STRINGS_ONLY);
	if (trainRaw.numberOfRows() == 0u || testRaw.numberOfRows() == 0u)
	{
		outErr = "Dataset manifests loaded, but one of the splits is empty.";
		return false;
	}

	datasetInfo.originalTrainSize = trainRaw.numberOfRows();
	datasetInfo.originalTestSize = testRaw.numberOfRows();

	const std::vector<unsigned int> trainIdx = make_balanced_indices(trainRaw, trainLimit);
	const std::vector<unsigned int> testIdx = make_balanced_indices(testRaw, testLimit);

	di.loaded = false;
	di.name = shmea::GString(datasetInfo.datasetName.c_str());
	di.trainingLegend = table_from_indices(trainRaw, trainIdx);
	di.testingLegend = table_from_indices(testRaw, testIdx);
	di.trainingOHEMaps.clear();
	di.trainingFeatureIsCategorical.clear();
	di.testingOHEMaps.clear();
	di.testingFeatureIsCategorical.clear();
	di.trainingPaths.clear();
	di.testingPaths.clear();
	di.featureCount = 0u;
	di.rowCacheOrder.clear();
	di.rowCache.clear();
	di.scratchRow.clear();
	di.scratchExpected.clear();
	di.oneHotByIndex.clear();
	di.emptyRow.clear();

	di.importHelper(di.trainingLegend, di.trainingOHEMaps, di.trainingFeatureIsCategorical);
	if (di.trainingOHEMaps.size() <= 1u || !di.trainingOHEMaps[1])
	{
		outErr = "Failed to build label encoding for the training split.";
		return false;
	}
	di.testingOHEMaps = di.trainingOHEMaps;
	di.testingFeatureIsCategorical = di.trainingFeatureIsCategorical;

	const unsigned int classCount = static_cast<unsigned int>(di.trainingOHEMaps[1]->size());
	di.oneHotByIndex.resize(classCount);
	for (unsigned int i = 0; i < classCount; ++i)
	{
		di.oneHotByIndex[i] = shmea::GVector<float>(classCount, 0.0f);
		di.oneHotByIndex[i][i] = 1.0f;
	}

	di.trainingPaths.reserve(di.trainingLegend.numberOfRows());
	for (unsigned int i = 0; i < di.trainingLegend.numberOfRows(); ++i)
		di.trainingPaths.push_back(join_path(datasetInfo.resolvedDir, di.trainingLegend.getCell(i, 0).c_str()));

	di.testingPaths.reserve(di.testingLegend.numberOfRows());
	for (unsigned int i = 0; i < di.testingLegend.numberOfRows(); ++i)
		di.testingPaths.push_back(join_path(datasetInfo.resolvedDir, di.testingLegend.getCell(i, 0).c_str()));

	if (di.trainingPaths.empty())
	{
		outErr = "Selected training split is empty after subsetting.";
		return false;
	}

	shmea::Image img;
	img.LoadPNG(shmea::GString(di.trainingPaths[0].c_str()));
	di.featureCount = img.getPixelCount();
	if (di.featureCount == 0u)
	{
		outErr = std::string("Failed to load the first training image: ") + di.trainingPaths[0];
		return false;
	}

	di.rowCacheMaxEntries = di.getTrainSize() + di.getTestSize();
	di.loaded = true;
	return true;
}

static bool warm_image_cache(glades::ImageInput& di, long long& outMs, std::string& outErr)
{
	outMs = 0LL;
	outErr.clear();

	const int64_t t0 = now_ms();
	for (unsigned int i = 0; i < di.getTrainSize(); ++i)
	{
		const float* data = NULL;
		unsigned int size = 0u;
		if (!di.getTrainRowView(i, data, size) || !data || size != di.getFeatureCount())
		{
			outErr = "Failed to materialize a training image while warming the cache.";
			return false;
		}
	}
	for (unsigned int i = 0; i < di.getTestSize(); ++i)
	{
		const float* data = NULL;
		unsigned int size = 0u;
		if (!di.getTestRowView(i, data, size) || !data || size != di.getFeatureCount())
		{
			outErr = "Failed to materialize a test image while warming the cache.";
			return false;
		}
	}
	const int64_t t1 = now_ms();
	outMs = static_cast<long long>(t1 - t0);
	return true;
}

static glades::NNetwork make_lenet_mnist(const std::string& name,
                                         float learningRate,
                                         float momentum,
                                         int batchSize,
                                         unsigned int seed)
{
	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    batchSize,
	    learningRate,
	    momentum,
	    0.0f,
	    0.0f,
	    0.0f,
	    glades::GMath::RELU,
	    1.0f);

	std::vector<glades::HiddenLayerInfo*> hidden;
	hidden.push_back(new glades::HiddenLayerInfo(
	    128,
	    learningRate,
	    momentum,
	    0.0f,
	    0.0f,
	    0.0f,
	    glades::GMath::RELU,
	    1.0f));

	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
	    10,
	    glades::OutputLayerInfo::CLASSIFICATION);

	glades::NNInfo* info = new glades::NNInfo(name.c_str(), in, hidden, out);
	glades::NNetwork net(info, glades::NNetwork::TYPE_CNN);
	net.setSeed(seed);

	glades::CNNConfig::ConvLayerSpec conv1;
	conv1.outChannels = 8u;
	conv1.kernelH = 5u; conv1.kernelW = 5u;
	conv1.strideH = 1u; conv1.strideW = 1u;
	conv1.padH = 0u; conv1.padW = 0u;
	conv1.useBatchNorm = false;
	conv1.useMaxPool = true;
	conv1.poolH = 2u; conv1.poolW = 2u;
	conv1.poolStrideH = 2u; conv1.poolStrideW = 2u;

	glades::CNNConfig::ConvLayerSpec conv2;
	conv2.outChannels = 16u;
	conv2.kernelH = 5u; conv2.kernelW = 5u;
	conv2.strideH = 1u; conv2.strideW = 1u;
	conv2.padH = 0u; conv2.padW = 0u;
	conv2.useBatchNorm = false;
	conv2.useMaxPool = true;
	conv2.poolH = 2u; conv2.poolW = 2u;
	conv2.poolStrideH = 2u; conv2.poolStrideW = 2u;

	glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
	cfg.cnn.inputC = 1u;
	cfg.cnn.inputH = 28u;
	cfg.cnn.inputW = 28u;
	cfg.cnn.convLayers.clear();
	cfg.cnn.convLayers.push_back(conv1);
	cfg.cnn.convLayers.push_back(conv2);

	delete info;
	return net;
}

static void configure_optimizer(glades::NNetwork& net,
                                OptimizerVariant optimizer,
                                const BenchConfig& cfg)
{
	glades::TrainingConfig& tc = net.getTrainingConfigMutable();
	tc.globalGradClipNorm = cfg.clipNorm;

	if (optimizer == OPT_SGD)
	{
		tc.optimizer.type = glades::OptimizerConfig::SGD_MOMENTUM;
	}
	else if (optimizer == OPT_ADAMW)
	{
		tc.optimizer.type = glades::OptimizerConfig::ADAMW;
		tc.optimizer.adamBeta1 = 0.9f;
		tc.optimizer.adamBeta2 = 0.999f;
		tc.optimizer.adamEps = 1e-8f;
		tc.optimizer.adamBiasCorrection = true;
	}
	else
	{
		tc.optimizer.type = glades::OptimizerConfig::ATLAS;
		tc.atlas.rank = cfg.atlasRank;
		tc.atlas.tSub = cfg.atlasTSub;
		tc.atlas.beta = 0.999f;
		tc.atlas.betaRefresh = 0.5f;
		tc.atlas.kappaMax = 10.0f;
	}
}

static const char* optimizer_label(OptimizerVariant optimizer)
{
	switch (optimizer)
	{
	case OPT_SGD: return "SGD";
	case OPT_ADAMW: return "AdamW";
	case OPT_ATLAS_BRSP: return "ATLAS-BSRP";
	default: return "Unknown";
	}
}

static float optimizer_learning_rate(OptimizerVariant optimizer, const BenchConfig& cfg)
{
	switch (optimizer)
	{
	case OPT_SGD: return cfg.sgdLR;
	case OPT_ADAMW: return cfg.adamLR;
	case OPT_ATLAS_BRSP: return cfg.atlasLR;
	default: return cfg.sgdLR;
	}
}

static float optimizer_momentum(OptimizerVariant optimizer, const BenchConfig& cfg)
{
	return (optimizer == OPT_SGD) ? cfg.sgdMomentum : 0.0f;
}

static SingleRunResult run_single_benchmark(glades::ImageInput& data,
                                            OptimizerVariant optimizer,
                                            const BenchConfig& cfg,
                                            unsigned int seed)
{
	SingleRunResult out;
	out.optimizerLabel = optimizer_label(optimizer);

	const float lr = optimizer_learning_rate(optimizer, cfg);
	const float momentum = optimizer_momentum(optimizer, cfg);
	const std::string netName = std::string("atlas_bench_") + out.optimizerLabel;

	glades::NNetwork net = make_lenet_mnist(netName, lr, momentum, static_cast<int>(cfg.batchSize), seed);
	configure_optimizer(net, optimizer, cfg);
	net.getTerminatorMutable().setEpoch(static_cast<int>(cfg.epochs));
	net.getTerminatorMutable().setAccuracy(0.0f);

	CaptureMetricsCallbacks trainCb;
	const int64_t t0 = now_ms();
	const glades::NNetworkStatus trainStatus = net.train(&data, &trainCb);
	const int64_t t1 = now_ms();
	out.trainMs = static_cast<long long>(t1 - t0);
	if (!trainStatus.ok())
	{
		out.ok = false;
		out.err = trainStatus.message;
		return out;
	}
	if (!trainCb.saw)
	{
		out.ok = false;
		out.err = "Training completed without reporting epoch metrics.";
		return out;
	}

	CaptureMetricsCallbacks testCb;
	const int64_t t2 = now_ms();
	const glades::NNetworkStatus testStatus = net.test(&data, &testCb);
	const int64_t t3 = now_ms();
	out.testMs = static_cast<long long>(t3 - t2);
	if (!testStatus.ok())
	{
		out.ok = false;
		out.err = testStatus.message;
		return out;
	}
	if (!testCb.saw)
	{
		out.ok = false;
		out.err = "Evaluation completed without reporting metrics.";
		return out;
	}

	out.trainLoss = trainCb.last.totalError;
	out.trainAcc = trainCb.last.classAccuracy;
	out.testLoss = testCb.last.totalError;
	out.testAcc = testCb.last.classAccuracy;
	if (!is_finite(out.trainLoss) || !is_finite(out.trainAcc) ||
	    !is_finite(out.testLoss) || !is_finite(out.testAcc))
	{
		out.ok = false;
		out.err = "Non-finite loss/accuracy detected.";
		return out;
	}

	const double totalTrainImages = static_cast<double>(cfg.epochs) * static_cast<double>(data.getTrainSize());
	const double seconds = static_cast<double>(out.trainMs) / 1000.0;
	out.trainImagesPerSec = (seconds > 0.0) ? (totalTrainImages / seconds) : 0.0;
	return out;
}

static AggregateStats compute_stats(const std::vector<double>& values)
{
	AggregateStats stats;
	if (values.empty())
		return stats;

	double sum = 0.0;
	for (size_t i = 0; i < values.size(); ++i)
		sum += values[i];
	stats.mean = sum / static_cast<double>(values.size());

	if (values.size() == 1u)
		return stats;

	double sumsq = 0.0;
	for (size_t i = 0; i < values.size(); ++i)
	{
		const double d = values[i] - stats.mean;
		sumsq += d * d;
	}
	stats.stddev = sqrt(sumsq / static_cast<double>(values.size()));
	return stats;
}

static void print_summary_row(const char* label,
                              const AggregateStats& trainSec,
                              const AggregateStats& imgPerSec,
                              const AggregateStats& trainLoss,
                              const AggregateStats& trainAcc,
                              const AggregateStats& testLoss,
                              const AggregateStats& testAcc,
                              const char* status)
{
	printf("%-11s  %7.2f +/- %-7.2f  %8.1f +/- %-8.1f  %8.4f +/- %-8.4f  %7.2f +/- %-7.2f  %8.4f +/- %-8.4f  %7.2f +/- %-7.2f  %s\n",
	       label,
	       trainSec.mean, trainSec.stddev,
	       imgPerSec.mean, imgPerSec.stddev,
	       trainLoss.mean, trainLoss.stddev,
	       trainAcc.mean, trainAcc.stddev,
	       testLoss.mean, testLoss.stddev,
	       testAcc.mean, testAcc.stddev,
	       status);
}

} // anonymous namespace

void ATLASBenchmark(int argc, char* argv[])
{
	BenchConfig cfg;
	for (int i = 2; i < argc; ++i)
	{
		if (streq(argv[i], "--help"))
		{
			print_usage();
			return;
		}
		else if (streq(argv[i], "--dataset") && i + 1 < argc)
			cfg.datasetRequest = argv[++i];
		else if (streq(argv[i], "--train-limit") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.trainLimit))
			{
				printf("Invalid value for --train-limit\n");
				return;
			}
		}
		else if (streq(argv[i], "--test-limit") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.testLimit))
			{
				printf("Invalid value for --test-limit\n");
				return;
			}
		}
		else if (streq(argv[i], "--epochs") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.epochs))
			{
				printf("Invalid value for --epochs\n");
				return;
			}
		}
		else if (streq(argv[i], "--repeats") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.repeats))
			{
				printf("Invalid value for --repeats\n");
				return;
			}
		}
		else if (streq(argv[i], "--batch-size") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.batchSize))
			{
				printf("Invalid value for --batch-size\n");
				return;
			}
		}
		else if (streq(argv[i], "--clip-norm") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.clipNorm))
			{
				printf("Invalid value for --clip-norm\n");
				return;
			}
		}
		else if (streq(argv[i], "--sgd-lr") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.sgdLR))
			{
				printf("Invalid value for --sgd-lr\n");
				return;
			}
		}
		else if (streq(argv[i], "--sgd-momentum") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.sgdMomentum))
			{
				printf("Invalid value for --sgd-momentum\n");
				return;
			}
		}
		else if (streq(argv[i], "--adam-lr") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.adamLR))
			{
				printf("Invalid value for --adam-lr\n");
				return;
			}
		}
		else if (streq(argv[i], "--atlas-lr") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasLR))
			{
				printf("Invalid value for --atlas-lr\n");
				return;
			}
		}
		else if (streq(argv[i], "--rank") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasRank))
			{
				printf("Invalid value for --rank\n");
				return;
			}
		}
		else if (streq(argv[i], "--tsub") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasTSub))
			{
				printf("Invalid value for --tsub\n");
				return;
			}
		}
		else if (streq(argv[i], "--seed") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.seed))
			{
				printf("Invalid value for --seed\n");
				return;
			}
		}
		else
		{
			printf("Unknown argument: %s\n", argv[i]);
			print_usage();
			return;
		}
	}

	printf("============================================================\n");
	printf("ATLAS-BSRP vs SGD vs AdamW CNN Benchmark\n");
	printf("============================================================\n");

	DatasetInfo datasetInfo;
	std::string err;
	if (!resolve_dataset_info(cfg.datasetRequest, datasetInfo, err))
	{
		printf("Dataset resolution failed: %s\n", err.c_str());
		return;
	}

	glades::ImageInput data;
	if (!load_image_dataset(data, datasetInfo, cfg.trainLimit, cfg.testLimit, err))
	{
		printf("Dataset load failed: %s\n", err.c_str());
		return;
	}

	long long warmMs = 0LL;
	if (!warm_image_cache(data, warmMs, err))
	{
		printf("Cache warmup failed: %s\n", err.c_str());
		return;
	}

	printf("Dataset: %s\n", datasetInfo.datasetName.c_str());
	printf("Path: %s\n", datasetInfo.resolvedDir.c_str());
	printf("Train split: %u / %u\n", data.getTrainSize(), datasetInfo.originalTrainSize);
	printf("Test split:  %u / %u\n", data.getTestSize(), datasetInfo.originalTestSize);
	if (!datasetInfo.note.empty())
		printf("Note: %s\n", datasetInfo.note.c_str());
	printf("Model: LeNet-style CNN (8x5x5 -> pool -> 16x5x5 -> pool -> FC128 -> 10)\n");
	printf("Config: epochs=%u repeats=%u batch=%u clip=%.2f\n",
	       cfg.epochs, cfg.repeats, cfg.batchSize, cfg.clipNorm);
	printf("LRs: SGD=%.4f (momentum=%.2f) AdamW=%.4f ATLAS-BSRP=%.4f rank=%u tSub=%u\n",
	       cfg.sgdLR, cfg.sgdMomentum, cfg.adamLR, cfg.atlasLR, cfg.atlasRank, cfg.atlasTSub);
	printf("Cache warmup: %lld ms for %u images (excluded from benchmark timing)\n",
	       warmMs, data.getTrainSize() + data.getTestSize());
	printf("\n");

	const OptimizerVariant optimizers[] = { OPT_SGD, OPT_ADAMW, OPT_ATLAS_BRSP };
	const size_t optimizerCount = sizeof(optimizers) / sizeof(optimizers[0]);
	std::vector<BenchmarkSummary> summaries;
	summaries.reserve(optimizerCount);

	for (size_t opt = 0; opt < optimizerCount; ++opt)
	{
		printf("Running %s\n", optimizer_label(optimizers[opt]));

		std::vector<double> trainSecVals;
		std::vector<double> imgPerSecVals;
		std::vector<double> trainLossVals;
		std::vector<double> trainAccVals;
		std::vector<double> testLossVals;
		std::vector<double> testAccVals;
		bool allOk = true;
		std::string firstErr;

		for (unsigned int rep = 0u; rep < cfg.repeats; ++rep)
		{
			const unsigned int seed = cfg.seed + rep;
			const SingleRunResult r = run_single_benchmark(data, optimizers[opt], cfg, seed);
			if (!r.ok)
			{
				allOk = false;
				if (firstErr.empty())
					firstErr = r.err;
				printf("  [%u/%u] seed=%u failed: %s\n",
				       rep + 1u, cfg.repeats, seed, r.err.c_str());
				continue;
			}

			printf("  [%u/%u] seed=%u train=%.2fs test=%.2fs trainLoss=%.4f trainAcc=%.2f%% testLoss=%.4f testAcc=%.2f%% img/s=%.1f\n",
			       rep + 1u, cfg.repeats, seed,
			       static_cast<double>(r.trainMs) / 1000.0,
			       static_cast<double>(r.testMs) / 1000.0,
			       r.trainLoss, r.trainAcc,
			       r.testLoss, r.testAcc,
			       r.trainImagesPerSec);

			trainSecVals.push_back(static_cast<double>(r.trainMs) / 1000.0);
			imgPerSecVals.push_back(r.trainImagesPerSec);
			trainLossVals.push_back(r.trainLoss);
			trainAccVals.push_back(r.trainAcc);
			testLossVals.push_back(r.testLoss);
			testAccVals.push_back(r.testAcc);
		}

		BenchmarkSummary summary;
		summary.label = optimizer_label(optimizers[opt]);
		summary.ok = allOk && !trainSecVals.empty();
		if (!summary.ok)
		{
			summary.status = firstErr.empty() ? "FAILED" : firstErr;
		}
		else
		{
			summary.trainSec = compute_stats(trainSecVals);
			summary.imgPerSec = compute_stats(imgPerSecVals);
			summary.trainLoss = compute_stats(trainLossVals);
			summary.trainAcc = compute_stats(trainAccVals);
			summary.testLoss = compute_stats(testLossVals);
			summary.testAcc = compute_stats(testAccVals);
			summary.status = "OK";
		}
		summaries.push_back(summary);

		printf("\n");
	}

	printf("Final summary\n");
	printf("Optimizer     Train(s)              Img/s                 TrainLoss             TrainAcc(%%)          TestLoss              TestAcc(%%)           Status\n");
	printf("------------  --------------------  --------------------  --------------------  --------------------  --------------------  --------------------  --------\n");
	for (size_t i = 0; i < summaries.size(); ++i)
	{
		if (!summaries[i].ok)
		{
			printf("%-11s  %-20s  %-20s  %-20s  %-20s  %-20s  %-20s  %s\n",
			       summaries[i].label,
			       "-", "-", "-", "-", "-", "-", summaries[i].status.c_str());
			continue;
		}
		print_summary_row(
		    summaries[i].label,
		    summaries[i].trainSec,
		    summaries[i].imgPerSec,
		    summaries[i].trainLoss,
		    summaries[i].trainAcc,
		    summaries[i].testLoss,
		    summaries[i].testAcc,
		    summaries[i].status.c_str());
	}
	printf("\n");
	printf("============================================================\n");
}
