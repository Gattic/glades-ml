#include "atlas-alt-bench.h"
#include "test_token_id_input_fixture.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/DataObjects/DataInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Database/GLogger.h"

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

static bool streq(const char* a, const char* b)
{
	return (a && b && strcmp(a, b) == 0);
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

static unsigned int mix_u32(unsigned int x)
{
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

static float rand_signed(unsigned int& state)
{
	state = mix_u32(state + 0x9e3779b9U);
	const unsigned int mantissa = state & 0x00ffffffU;
	return (static_cast<float>(mantissa) / 8388607.5f) - 1.0f;
}

static float safe_exp(float v)
{
	if (v > 60.0f)
		v = 60.0f;
	if (v < -60.0f)
		v = -60.0f;
	return expf(v);
}

static shmea::GLogger* quiet_logger()
{
	static shmea::GLogger logger(shmea::GLogger::LOG_ERROR);
	static bool initialized = false;
	if (!initialized)
	{
		logger.setPrintToConsole(false);
		initialized = true;
	}
	return &logger;
}

struct CaptureMetricsCallbacks : public glades::ITrainingCallbacks
{
	glades::NNetworkEpochMetrics last;
	bool saw;
	bool sawAtlas;
	double atlasSparrowActiveModesSum;
	double atlasSparrowMode2FractionSum;
	double atlasSparrowSecondEdgeRatioSum;
	double atlasSparrowEdgeSum;
	double atlasSparrowMemoryGainSum;
	double atlasSparrowHorizontalRatioSum;
	double atlasHelmEdgeSum;
	double atlasHelmSecondEdgeSum;
	double atlasHelmSecondEdgeRatioSum;
	double atlasHelmActiveModesSum;
	double atlasHelmSigmaSum;
	double atlasHelmPredR2Sum;
	double atlasHelmMemoryGainSum;
	double atlasHelmPoleSum;
	double atlasAsterActiveModesSum;
	double atlasAsterSecondEdgeSum;
	double atlasAsterSecondEdgeRatioSum;
	double atlasAsterEdgeSum;
	double atlasAsterSigmaSum;
	double atlasAsterPredR2Sum;
	double atlasAsterMemoryGainSum;
	double atlasAsterPoleSum;
	unsigned int atlasHelmMode2Epochs;
	unsigned int atlasSparrowEpochs;
	unsigned int atlasHelmEpochs;
	unsigned int atlasAsterMode2Epochs;
	unsigned int atlasAsterEpochs;

	CaptureMetricsCallbacks()
	    : last(),
	      saw(false),
	      sawAtlas(false),
	      atlasSparrowActiveModesSum(0.0),
	      atlasSparrowMode2FractionSum(0.0),
	      atlasSparrowSecondEdgeRatioSum(0.0),
	      atlasSparrowEdgeSum(0.0),
	      atlasSparrowMemoryGainSum(0.0),
	      atlasSparrowHorizontalRatioSum(0.0),
	      atlasHelmEdgeSum(0.0),
	      atlasHelmSecondEdgeSum(0.0),
	      atlasHelmSecondEdgeRatioSum(0.0),
	      atlasHelmActiveModesSum(0.0),
	      atlasHelmSigmaSum(0.0),
	      atlasHelmPredR2Sum(0.0),
	      atlasHelmMemoryGainSum(0.0),
	      atlasHelmPoleSum(0.0),
	      atlasAsterActiveModesSum(0.0),
	      atlasAsterSecondEdgeSum(0.0),
	      atlasAsterSecondEdgeRatioSum(0.0),
	      atlasAsterEdgeSum(0.0),
	      atlasAsterSigmaSum(0.0),
	      atlasAsterPredR2Sum(0.0),
	      atlasAsterMemoryGainSum(0.0),
	      atlasAsterPoleSum(0.0),
	      atlasHelmMode2Epochs(0u),
	      atlasSparrowEpochs(0u),
	      atlasHelmEpochs(0u),
	      atlasAsterMode2Epochs(0u),
	      atlasAsterEpochs(0u)
	{
	}

	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork& net, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;
		glades::NNetwork::AtlasRuntimeDiagnostics diag;
		if (net.getAtlasRuntimeDiagnostics(diag))
		{
			if (diag.sparrowMatrices > 0u || diag.helmMatrices > 0u || diag.asterMatrices > 0u)
				sawAtlas = true;
			if (diag.sparrowMatrices > 0u)
			{
				atlasSparrowActiveModesSum += diag.sparrowMeanActiveModes;
				atlasSparrowMode2FractionSum += diag.sparrowMode2Fraction;
				atlasSparrowSecondEdgeRatioSum += diag.sparrowMeanSecondEdgeRatio;
				atlasSparrowEdgeSum += diag.sparrowMeanEdge;
				atlasSparrowMemoryGainSum += diag.sparrowMeanMemoryGain;
				atlasSparrowHorizontalRatioSum += diag.sparrowMeanHorizontalRatio;
				atlasSparrowEpochs += 1u;
			}
			if (diag.helmMatrices > 0u)
			{
				atlasHelmActiveModesSum += diag.helmMeanActiveModes;
				atlasHelmSecondEdgeSum += diag.helmMeanSecondEdge;
				atlasHelmSecondEdgeRatioSum += diag.helmMeanSecondEdgeRatio;
				atlasHelmEdgeSum += diag.helmMeanEdge;
				atlasHelmSigmaSum += diag.helmMeanSigma;
				atlasHelmPredR2Sum += diag.helmMeanPredR2;
				atlasHelmMemoryGainSum += diag.helmMeanMemoryGain;
				atlasHelmPoleSum += diag.helmMeanPole;
				if (diag.helmMode2Fraction > 0.0)
					atlasHelmMode2Epochs += 1u;
				atlasHelmEpochs += 1u;
			}
			if (diag.asterMatrices > 0u)
			{
				atlasAsterActiveModesSum += diag.asterMeanActiveModes;
				atlasAsterSecondEdgeSum += diag.asterMeanSecondEdge;
				atlasAsterSecondEdgeRatioSum += diag.asterMeanSecondEdgeRatio;
				atlasAsterEdgeSum += diag.asterMeanEdge;
				atlasAsterSigmaSum += diag.asterMeanSigma;
				atlasAsterPredR2Sum += diag.asterMeanPredR2;
				atlasAsterMemoryGainSum += diag.asterMeanMemoryGain;
				atlasAsterPoleSum += diag.asterMeanPole;
				if (diag.asterMode2Fraction > 0.0)
					atlasAsterMode2Epochs += 1u;
				atlasAsterEpochs += 1u;
			}
		}
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}
};

class DenseRegressionInput : public glades::DataInput
{
public:
	DenseRegressionInput()
	    : trainRows_(0u),
	      testRows_(0u),
	      featureCount_(0u),
	      expectedCount_(0u)
	{
	}

	void setTrain(const std::vector<float>& x, unsigned int rows, unsigned int featureCount,
	              const std::vector<float>& y, unsigned int expectedCount)
	{
		trainX_ = x;
		trainY_ = y;
		trainRows_ = rows;
		featureCount_ = featureCount;
		expectedCount_ = expectedCount;
	}

	void setTest(const std::vector<float>& x, unsigned int rows, unsigned int featureCount,
	             const std::vector<float>& y, unsigned int expectedCount)
	{
		testX_ = x;
		testY_ = y;
		testRows_ = rows;
		featureCount_ = featureCount;
		expectedCount_ = expectedCount;
	}

	virtual void import(shmea::GString, int = 0) {}
	virtual void import(const shmea::GTable&, int = 0) {}

	virtual shmea::GVector<float> getTrainRow(unsigned int i) const
	{
		return copy_row_(trainX_, trainRows_, featureCount_, i);
	}

	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int i) const
	{
		return copy_row_(trainY_, trainRows_, expectedCount_, i);
	}

	virtual shmea::GVector<float> getTestRow(unsigned int i) const
	{
		return copy_row_(testX_, testRows_, featureCount_, i);
	}

	virtual shmea::GVector<float> getTestExpectedRow(unsigned int i) const
	{
		return copy_row_(testY_, testRows_, expectedCount_, i);
	}

	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		return row_view_(trainX_, trainRows_, featureCount_, index, outData, outSize);
	}

	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		return row_view_(trainY_, trainRows_, expectedCount_, index, outData, outSize);
	}

	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		return row_view_(testX_, testRows_, featureCount_, index, outData, outSize);
	}

	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		return row_view_(testY_, testRows_, expectedCount_, index, outData, outSize);
	}

	virtual unsigned int getTrainSize() const { return trainRows_; }
	virtual unsigned int getTestSize() const { return testRows_; }
	virtual unsigned int getFeatureCount() const { return featureCount_; }
	virtual bool hasFixedTrainRowSize() const { return featureCount_ > 0u; }
	virtual unsigned int getFixedTrainRowSize() const { return featureCount_; }
	virtual bool hasFixedTrainExpectedRowSize() const { return expectedCount_ > 0u; }
	virtual unsigned int getFixedTrainExpectedRowSize() const { return expectedCount_; }
	virtual bool hasFixedTestRowSize() const { return featureCount_ > 0u; }
	virtual unsigned int getFixedTestRowSize() const { return featureCount_; }
	virtual bool hasFixedTestExpectedRowSize() const { return expectedCount_ > 0u; }
	virtual unsigned int getFixedTestExpectedRowSize() const { return expectedCount_; }
	virtual int getType() const { return CSV; }

private:
	static shmea::GVector<float> copy_row_(const std::vector<float>& flat,
	                                       unsigned int rows,
	                                       unsigned int cols,
	                                       unsigned int index)
	{
		if (index >= rows || cols == 0u)
			return shmea::GVector<float>();
		shmea::GVector<float> out(cols, 0.0f);
		const size_t offset = static_cast<size_t>(index) * static_cast<size_t>(cols);
		for (unsigned int j = 0u; j < cols; ++j)
			out[j] = flat[offset + j];
		return out;
	}

	static bool row_view_(const std::vector<float>& flat,
	                      unsigned int rows,
	                      unsigned int cols,
	                      unsigned int index,
	                      const float*& outData,
	                      unsigned int& outSize)
	{
		outData = NULL;
		outSize = 0u;
		if (index >= rows || cols == 0u)
			return false;
		const size_t offset = static_cast<size_t>(index) * static_cast<size_t>(cols);
		outData = &flat[offset];
		outSize = cols;
		return true;
	}

	std::vector<float> trainX_;
	std::vector<float> trainY_;
	std::vector<float> testX_;
	std::vector<float> testY_;
	unsigned int trainRows_;
	unsigned int testRows_;
	unsigned int featureCount_;
	unsigned int expectedCount_;
};

enum BenchMode
{
	MODE_ALL = 0,
	MODE_TOKEN_LM = 1,
	MODE_TEACHER_STUDENT = 2,
	MODE_LATENT_FORECAST = 3,
	MODE_TEACHER_SWEEP = 4,
	MODE_TEACHER_CANONICAL = 5,
	MODE_NONLINEAR_FORECAST = 6
};

enum VariantKind
{
	VARIANT_ADAMW = 0,
	VARIANT_ATLAS_BASE = 1,
	VARIANT_ATLAS_SPARROW = 2,
	VARIANT_ATLAS_HELM = 3,
	VARIANT_ATLAS_ASTER = 4
};

struct TokenConfig
{
	unsigned int vocab;
	unsigned int dModel;
	unsigned int dFF;
	unsigned int layers;
	unsigned int heads;
	unsigned int kvHeads;
	unsigned int seqLen;
	unsigned int trainSeqs;
	unsigned int testSeqs;
	unsigned int epochs;
	float adamLR;
	float atlasLR;

	TokenConfig()
	    : vocab(65u),
	      dModel(48u),
	      dFF(192u),
	      layers(2u),
	      heads(4u),
	      kvHeads(4u),
	      seqLen(32u),
	      trainSeqs(128u),
	      testSeqs(32u),
	      epochs(6u),
	      adamLR(0.0010f),
	      atlasLR(0.020f)
	{
	}
};

struct TeacherConfig
{
	unsigned int inputDim;
	unsigned int teacherRank;
	unsigned int bulkRank;
	unsigned int trainSamples;
	unsigned int testSamples;
	unsigned int batchSize;
	unsigned int epochs;
	float bulkScale;
	float adamLR;
	float atlasLR;

	TeacherConfig()
	    : inputDim(32u),
	      teacherRank(4u),
	      bulkRank(12u),
	      trainSamples(4096u),
	      testSamples(1024u),
	      batchSize(64u),
	      epochs(20u),
	      bulkScale(0.20f),
	      adamLR(0.0025f),
	      atlasLR(0.050f)
	{
	}
};

struct LatentConfig
{
	unsigned int latentDim;
	unsigned int obsDim;
	unsigned int window;
	unsigned int trainSeqs;
	unsigned int testSeqs;
	unsigned int seqLen;
	unsigned int batchSize;
	unsigned int epochs;
	float processNoise;
	float obsNoise;
	float adamLR;
	float atlasLR;
	float nonlinearMix;
	float switchingScale;
	float observationMix;

	LatentConfig()
	    : latentDim(6u),
	      obsDim(4u),
	      window(8u),
	      trainSeqs(96u),
	      testSeqs(24u),
	      seqLen(40u),
	      batchSize(64u),
	      epochs(18u),
	      processNoise(0.035f),
	      obsNoise(0.020f),
	      adamLR(0.0020f),
	      atlasLR(0.035f),
	      nonlinearMix(0.12f),
	      switchingScale(0.08f),
	      observationMix(0.06f)
	{
	}
};

enum SweepProfile
{
	SWEEP_QUICK = 0,
	SWEEP_FULL = 1
};

struct BenchConfig
{
	BenchMode mode;
	unsigned int repeats;
	unsigned int seed;
	unsigned int atlasRank;
	unsigned int atlasComplementRank;
	unsigned int atlasTSub;
	float atlasKappaMax;
	unsigned int atlasSparrowModeRank;
	unsigned int atlasSparrowAutoModeGate;
	float atlasSparrowMemoryScale;
	float atlasSparrowEdgeThreshold;
	float atlasSparrowSecondEdgeThreshold;
	float atlasSparrowSecondEdgeFraction;
	float atlasSparrowPoleMax;
	float atlasHelmMemoryScale;
	float atlasHelmEdgeThreshold;
	unsigned int atlasHelmModeRank;
	unsigned int atlasHelmHiddenStackDepth;
	float atlasHelmPoleMax;
	float atlasAsterMemoryScale;
	float atlasAsterEdgeThreshold;
	unsigned int atlasAsterStateRank;
	unsigned int atlasAsterHiddenStackDepth;
	float atlasAsterPoleMax;
	TokenConfig token;
	TeacherConfig teacher;
	LatentConfig latent;
	SweepProfile teacherSweepProfile;
	unsigned int teacherSweepLimit;

	BenchConfig()
	    : mode(MODE_ALL),
	      repeats(3u),
	      seed(1337u),
	      atlasRank(16u),
	      atlasComplementRank(4u),
	      atlasTSub(64u),
	      atlasKappaMax(10.0f),
	      atlasSparrowModeRank(1u),
	      atlasSparrowAutoModeGate(0u),
	      atlasSparrowMemoryScale(0.05f),
	      atlasSparrowEdgeThreshold(0.10f),
	      atlasSparrowSecondEdgeThreshold(0.10f),
	      atlasSparrowSecondEdgeFraction(0.50f),
	      atlasSparrowPoleMax(0.95f),
	      atlasHelmMemoryScale(0.05f),
	      atlasHelmEdgeThreshold(0.10f),
	      atlasHelmModeRank(2u),
	      atlasHelmHiddenStackDepth(2u),
	      atlasHelmPoleMax(0.95f),
	      atlasAsterMemoryScale(0.05f),
	      atlasAsterEdgeThreshold(0.10f),
	      atlasAsterStateRank(2u),
	      atlasAsterHiddenStackDepth(2u),
	      atlasAsterPoleMax(0.95f),
	      token(),
	      teacher(),
	      latent(),
	      teacherSweepProfile(SWEEP_QUICK),
	      teacherSweepLimit(0u)
	{
	}
};

struct TokenDataset
{
	InMemoryTokenIdInput di;
	unsigned int padTokenId;
	unsigned long long trainTokensPerEpoch;
	unsigned long long testTokens;

	TokenDataset()
	    : di(),
	      padTokenId(0u),
	      trainTokensPerEpoch(0ULL),
	      testTokens(0ULL)
	{
	}
};

struct TeacherSpec
{
	std::vector<float> signalW;
	std::vector<float> signalOut;
	std::vector<float> bulkW;
	std::vector<float> bulkOut;
	std::vector<float> bulkPhase;
};

struct RegressionNetworkSpec
{
	const char* name;
	unsigned int batchSize;
	std::vector<unsigned int> hiddenSizes;
	unsigned int outputDim;
	float adamLR;
	float atlasLR;
	float clipNorm;

	RegressionNetworkSpec()
	    : name(""),
	      batchSize(1u),
	      hiddenSizes(),
	      outputDim(1u),
	      adamLR(0.001f),
	      atlasLR(0.010f),
	      clipNorm(1.0f)
	{
	}
};

struct RunResult
{
	const char* label;
	long long trainMs;
	long long evalMs;
	double throughput;
	float trainLoss;
	float trainMetric;
	float testLoss;
	float testMetric;
	bool sparrowDiagValid;
	double sparrowActiveModes;
	double sparrowMode2Fraction;
	double sparrowSecondEdgeRatio;
	double sparrowEdge;
	double sparrowMemoryGain;
	double sparrowHorizontalRatio;
	bool helmDiagValid;
	double helmActiveModes;
	double helmMode2Fraction;
	double helmSecondEdgeRatio;
	double helmEdge;
	double helmSigma;
	double helmPredR2;
	double helmMemoryGain;
	double helmPole;
	bool asterDiagValid;
	double asterActiveModes;
	double asterMode2Fraction;
	double asterSecondEdgeRatio;
	double asterEdge;
	double asterSigma;
	double asterPredR2;
	double asterMemoryGain;
	double asterPole;
	bool ok;
	std::string err;

	RunResult()
	    : label(""),
	      trainMs(0LL),
	      evalMs(0LL),
	      throughput(0.0),
	      trainLoss(0.0f),
	      trainMetric(0.0f),
	      testLoss(0.0f),
	      testMetric(0.0f),
	      sparrowDiagValid(false),
	      sparrowActiveModes(0.0),
	      sparrowMode2Fraction(0.0),
	      sparrowSecondEdgeRatio(0.0),
	      sparrowEdge(0.0),
	      sparrowMemoryGain(0.0),
	      sparrowHorizontalRatio(0.0),
	      helmDiagValid(false),
	      helmActiveModes(0.0),
	      helmMode2Fraction(0.0),
	      helmSecondEdgeRatio(0.0),
	      helmEdge(0.0),
	      helmSigma(0.0),
	      helmPredR2(0.0),
	      helmMemoryGain(0.0),
	      helmPole(0.0),
	      asterDiagValid(false),
	      asterActiveModes(0.0),
	      asterMode2Fraction(0.0),
	      asterSecondEdgeRatio(0.0),
	      asterEdge(0.0),
	      asterSigma(0.0),
	      asterPredR2(0.0),
	      asterMemoryGain(0.0),
	      asterPole(0.0),
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

struct Summary
{
	const char* label;
	AggregateStats trainSec;
	AggregateStats throughput;
	AggregateStats trainLoss;
	AggregateStats trainMetric;
	AggregateStats testLoss;
	AggregateStats testMetric;
	bool sparrowDiagValid;
	AggregateStats sparrowActiveModes;
	AggregateStats sparrowMode2Fraction;
	AggregateStats sparrowSecondEdgeRatio;
	AggregateStats sparrowEdge;
	AggregateStats sparrowMemoryGain;
	AggregateStats sparrowHorizontalRatio;
	bool helmDiagValid;
	AggregateStats helmActiveModes;
	AggregateStats helmMode2Fraction;
	AggregateStats helmSecondEdgeRatio;
	AggregateStats helmEdge;
	AggregateStats helmSigma;
	AggregateStats helmPredR2;
	AggregateStats helmMemoryGain;
	AggregateStats helmPole;
	bool asterDiagValid;
	AggregateStats asterActiveModes;
	AggregateStats asterMode2Fraction;
	AggregateStats asterSecondEdgeRatio;
	AggregateStats asterEdge;
	AggregateStats asterSigma;
	AggregateStats asterPredR2;
	AggregateStats asterMemoryGain;
	AggregateStats asterPole;
	bool ok;
	std::string status;

	Summary()
	    : label(""),
	      trainSec(),
	      throughput(),
	      trainLoss(),
	      trainMetric(),
	      testLoss(),
	      testMetric(),
	      sparrowDiagValid(false),
	      sparrowActiveModes(),
	      sparrowMode2Fraction(),
	      sparrowSecondEdgeRatio(),
	      sparrowEdge(),
	      sparrowMemoryGain(),
	      sparrowHorizontalRatio(),
	      helmDiagValid(false),
	      helmActiveModes(),
	      helmMode2Fraction(),
	      helmSecondEdgeRatio(),
	      helmEdge(),
	      helmSigma(),
	      helmPredR2(),
	      helmMemoryGain(),
	      helmPole(),
	      asterDiagValid(false),
	      asterActiveModes(),
	      asterMode2Fraction(),
	      asterSecondEdgeRatio(),
	      asterEdge(),
	      asterSigma(),
	      asterPredR2(),
	      asterMemoryGain(),
	      asterPole(),
	      ok(false),
	      status()
	{
	}
};

static void fill_sparrow_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);
static void fill_helm_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);
static void fill_aster_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);

class NetworkOwner
{
public:
	glades::NNInfo* info;
	glades::NNetwork* net;

	NetworkOwner() : info(NULL), net(NULL) {}
	~NetworkOwner()
	{
		delete net;
		delete info;
	}

private:
	NetworkOwner(const NetworkOwner&);
	NetworkOwner& operator=(const NetworkOwner&);
};

static const char* variant_label(VariantKind variant)
{
	switch (variant)
	{
	case VARIANT_ADAMW: return "AdamW";
	case VARIANT_ATLAS_BASE: return "ATLAS-BSRP";
	case VARIANT_ATLAS_SPARROW: return "ATLAS-SPARROW";
	case VARIANT_ATLAS_HELM: return "ATLAS-HELM";
	case VARIANT_ATLAS_ASTER: return "ATLAS-ASTER";
	default: return "Unknown";
	}
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

static void print_usage()
{
	printf("Usage: glades-unit-tests atlas-alt-bench [options]\n");
	printf("Options:\n");
	printf("  --mode all|token-lm|teacher-student|latent-forecast|nonlinear-forecast|teacher-sweep|teacher-canonical\n");
	printf("                                         Run the alternate-task benches or the teacher-student sweep (default: all)\n");
	printf("  --repeats N                           Repeats per optimizer variant (default: 3)\n");
	printf("  --seed N                              Base RNG seed (default: 1337)\n");
	printf("  --rank N                              ATLAS active rank (default: 16)\n");
	printf("  --atlas-complement-rank N             ATLAS-SPARROW scout/complement cap (default: 4)\n");
	printf("  --tsub N                              ATLAS subspace refresh interval (default: 64)\n");
	printf("  --atlas-kappa-max X                   ATLAS active kappaMax (default: 10.0)\n");
	printf("  --atlas-sparrow-memory-scale X        SPARROW memory scale (default: 0.05)\n");
	printf("  --atlas-sparrow-edge-threshold X      SPARROW edge threshold (default: 0.10)\n");
	printf("  --atlas-sparrow-auto-mode-gate 0|1    Treat SPARROW mode rank as a cap with auto mode-2 admission (default: 0)\n");
	printf("  --atlas-sparrow-second-edge-threshold X  Minimum raw mode-2 SPARROW edge for auto admission (default: 0.10)\n");
	printf("  --atlas-sparrow-second-edge-fraction X   Minimum mode-2/mode-1 edge ratio for auto admission (default: 0.50)\n");
	printf("  --atlas-sparrow-pole-max X            SPARROW pole clamp (default: 0.95)\n");
	printf("  --atlas-sparrow-mode-rank N           SPARROW retained transfer mode rank (default: 1)\n");
	printf("  --atlas-helm-memory-scale X           HELM memory scale (default: 0.05)\n");
	printf("  --atlas-helm-edge-threshold X         HELM edge threshold (default: 0.10)\n");
	printf("  --atlas-helm-mode-rank N              HELM retained transfer mode rank (default: 2)\n");
	printf("  --atlas-helm-hidden-stack-depth N     HELM trailing hidden layers to stack (default: 2)\n");
	printf("  --atlas-helm-pole-max X               HELM pole clamp (default: 0.95)\n");
	printf("  --atlas-aster-memory-scale X          ASTER memory scale (default: 0.05)\n");
	printf("  --atlas-aster-edge-threshold X        ASTER edge threshold (default: 0.10)\n");
	printf("  --atlas-aster-state-rank N            ASTER retained state rank (default: 2)\n");
	printf("  --atlas-aster-hidden-stack-depth N    ASTER trailing hidden layers to transport (default: 2)\n");
	printf("  --atlas-aster-pole-max X              ASTER pole clamp (default: 0.95)\n");
	printf("  --token-epochs N                      Token-LM epochs (default: 6)\n");
	printf("  --token-train-seqs N                  Token-LM train sequence count (default: 128)\n");
	printf("  --token-test-seqs N                   Token-LM test sequence count (default: 32)\n");
	printf("  --teacher-epochs N                    Teacher-student epochs (default: 20)\n");
	printf("  --teacher-train-size N                Teacher-student train size (default: 4096)\n");
	printf("  --teacher-test-size N                 Teacher-student test size (default: 1024)\n");
	printf("  --teacher-bulk-rank N                 Teacher-student nuisance bulk rank (default: 12)\n");
	printf("  --teacher-bulk-scale X                Teacher-student nuisance bulk scale (default: 0.20)\n");
	printf("  --latent-epochs N                     Latent-forecast epochs (default: 18)\n");
	printf("  --latent-train-seqs N                 Latent-forecast train sequence count (default: 96)\n");
	printf("  --latent-test-seqs N                  Latent-forecast test sequence count (default: 24)\n");
	printf("  --latent-seq-len N                    Latent-forecast sequence length (default: 40)\n");
	printf("  --teacher-sweep-profile quick|full    Teacher-student sweep grid size (default: quick)\n");
	printf("  --teacher-sweep-limit N               Limit sweep base configurations for smoke runs (default: 0 = no limit)\n");
	printf("  --help                                Show this message\n");
}

static bool parse_mode_arg(const char* text, BenchMode& outMode)
{
	if (!text || !*text)
		return false;
	if (streq(text, "all"))
	{
		outMode = MODE_ALL;
		return true;
	}
	if (streq(text, "token-lm") || streq(text, "token") || streq(text, "lm"))
	{
		outMode = MODE_TOKEN_LM;
		return true;
	}
	if (streq(text, "teacher-student") || streq(text, "teacher") || streq(text, "ts"))
	{
		outMode = MODE_TEACHER_STUDENT;
		return true;
	}
	if (streq(text, "latent-forecast") || streq(text, "latent") || streq(text, "forecast") || streq(text, "lds"))
	{
		outMode = MODE_LATENT_FORECAST;
		return true;
	}
	if (streq(text, "nonlinear-forecast") || streq(text, "nonlinear") || streq(text, "latent-nonlinear") || streq(text, "switching-forecast"))
	{
		outMode = MODE_NONLINEAR_FORECAST;
		return true;
	}
	if (streq(text, "teacher-sweep") || streq(text, "sweep"))
	{
		outMode = MODE_TEACHER_SWEEP;
		return true;
	}
	if (streq(text, "teacher-canonical") || streq(text, "teacher-regression") || streq(text, "canonical"))
	{
		outMode = MODE_TEACHER_CANONICAL;
		return true;
	}
	return false;
}

static bool parse_sweep_profile_arg(const char* text, SweepProfile& outProfile)
{
	if (!text || !*text)
		return false;
	if (streq(text, "quick"))
	{
		outProfile = SWEEP_QUICK;
		return true;
	}
	if (streq(text, "full"))
	{
		outProfile = SWEEP_FULL;
		return true;
	}
	return false;
}

static bool parse_args(int argc, char* argv[], BenchConfig& cfg, std::string& err)
{
	err.clear();
	for (int i = 2; i < argc; ++i)
	{
		if (streq(argv[i], "--help"))
			return false;
		else if (streq(argv[i], "--mode") && i + 1 < argc)
		{
			if (!parse_mode_arg(argv[++i], cfg.mode))
			{
				err = "invalid --mode value";
				return false;
			}
		}
		else if (streq(argv[i], "--repeats") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.repeats) || cfg.repeats == 0u)
			{
				err = "invalid --repeats";
				return false;
			}
		}
		else if (streq(argv[i], "--seed") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.seed))
			{
				err = "invalid --seed";
				return false;
			}
		}
		else if (streq(argv[i], "--rank") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasRank) || cfg.atlasRank == 0u)
			{
				err = "invalid --rank";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-complement-rank") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasComplementRank))
			{
				err = "invalid --atlas-complement-rank";
				return false;
			}
		}
		else if (streq(argv[i], "--tsub") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasTSub) || cfg.atlasTSub == 0u)
			{
				err = "invalid --tsub";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-kappa-max") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasKappaMax) || cfg.atlasKappaMax <= 0.0f)
			{
				err = "invalid --atlas-kappa-max";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-sparrow-memory-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasSparrowMemoryScale))
			{
				err = "invalid --atlas-sparrow-memory-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-sparrow-edge-threshold") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasSparrowEdgeThreshold))
			{
				err = "invalid --atlas-sparrow-edge-threshold";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-sparrow-auto-mode-gate") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasSparrowAutoModeGate) || cfg.atlasSparrowAutoModeGate > 1u)
			{
				err = "invalid --atlas-sparrow-auto-mode-gate";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-sparrow-second-edge-threshold") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasSparrowSecondEdgeThreshold))
			{
				err = "invalid --atlas-sparrow-second-edge-threshold";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-sparrow-second-edge-fraction") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasSparrowSecondEdgeFraction))
			{
				err = "invalid --atlas-sparrow-second-edge-fraction";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-sparrow-pole-max") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasSparrowPoleMax))
			{
				err = "invalid --atlas-sparrow-pole-max";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-sparrow-mode-rank") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasSparrowModeRank) || cfg.atlasSparrowModeRank == 0u)
			{
				err = "invalid --atlas-sparrow-mode-rank";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-helm-memory-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasHelmMemoryScale))
			{
				err = "invalid --atlas-helm-memory-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-helm-edge-threshold") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasHelmEdgeThreshold))
			{
				err = "invalid --atlas-helm-edge-threshold";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-helm-mode-rank") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasHelmModeRank) || cfg.atlasHelmModeRank == 0u)
			{
				err = "invalid --atlas-helm-mode-rank";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-helm-hidden-stack-depth") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasHelmHiddenStackDepth) || cfg.atlasHelmHiddenStackDepth == 0u)
			{
				err = "invalid --atlas-helm-hidden-stack-depth";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-helm-pole-max") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasHelmPoleMax))
			{
				err = "invalid --atlas-helm-pole-max";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-aster-memory-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasAsterMemoryScale))
			{
				err = "invalid --atlas-aster-memory-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-aster-edge-threshold") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasAsterEdgeThreshold))
			{
				err = "invalid --atlas-aster-edge-threshold";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-aster-state-rank") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasAsterStateRank) || cfg.atlasAsterStateRank == 0u)
			{
				err = "invalid --atlas-aster-state-rank";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-aster-hidden-stack-depth") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasAsterHiddenStackDepth) || cfg.atlasAsterHiddenStackDepth == 0u)
			{
				err = "invalid --atlas-aster-hidden-stack-depth";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-aster-pole-max") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasAsterPoleMax))
			{
				err = "invalid --atlas-aster-pole-max";
				return false;
			}
		}
		else if (streq(argv[i], "--token-epochs") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.token.epochs) || cfg.token.epochs == 0u)
			{
				err = "invalid --token-epochs";
				return false;
			}
		}
		else if (streq(argv[i], "--token-train-seqs") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.token.trainSeqs) || cfg.token.trainSeqs == 0u)
			{
				err = "invalid --token-train-seqs";
				return false;
			}
		}
		else if (streq(argv[i], "--token-test-seqs") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.token.testSeqs) || cfg.token.testSeqs == 0u)
			{
				err = "invalid --token-test-seqs";
				return false;
			}
		}
		else if (streq(argv[i], "--teacher-epochs") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.teacher.epochs) || cfg.teacher.epochs == 0u)
			{
				err = "invalid --teacher-epochs";
				return false;
			}
		}
		else if (streq(argv[i], "--teacher-train-size") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.teacher.trainSamples) || cfg.teacher.trainSamples == 0u)
			{
				err = "invalid --teacher-train-size";
				return false;
			}
		}
		else if (streq(argv[i], "--teacher-test-size") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.teacher.testSamples) || cfg.teacher.testSamples == 0u)
			{
				err = "invalid --teacher-test-size";
				return false;
			}
		}
		else if (streq(argv[i], "--teacher-bulk-rank") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.teacher.bulkRank))
			{
				err = "invalid --teacher-bulk-rank";
				return false;
			}
		}
		else if (streq(argv[i], "--teacher-bulk-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.teacher.bulkScale) || cfg.teacher.bulkScale < 0.0f)
			{
				err = "invalid --teacher-bulk-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--latent-epochs") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.latent.epochs) || cfg.latent.epochs == 0u)
			{
				err = "invalid --latent-epochs";
				return false;
			}
		}
		else if (streq(argv[i], "--latent-train-seqs") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.latent.trainSeqs) || cfg.latent.trainSeqs == 0u)
			{
				err = "invalid --latent-train-seqs";
				return false;
			}
		}
		else if (streq(argv[i], "--latent-test-seqs") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.latent.testSeqs) || cfg.latent.testSeqs == 0u)
			{
				err = "invalid --latent-test-seqs";
				return false;
			}
		}
		else if (streq(argv[i], "--latent-seq-len") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.latent.seqLen) || cfg.latent.seqLen < 4u)
			{
				err = "invalid --latent-seq-len";
				return false;
			}
		}
		else if (streq(argv[i], "--teacher-sweep-profile") && i + 1 < argc)
		{
			if (!parse_sweep_profile_arg(argv[++i], cfg.teacherSweepProfile))
			{
				err = "invalid --teacher-sweep-profile";
				return false;
			}
		}
		else if (streq(argv[i], "--teacher-sweep-limit") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.teacherSweepLimit))
			{
				err = "invalid --teacher-sweep-limit";
				return false;
			}
		}
		else
		{
			err = std::string("unknown or incomplete option: ") + argv[i];
			return false;
		}
	}

	if (cfg.token.vocab < 8u || cfg.token.heads == 0u || cfg.token.kvHeads == 0u ||
	    (cfg.token.dModel % cfg.token.heads) != 0u || (cfg.token.heads % cfg.token.kvHeads) != 0u)
	{
		err = "invalid token-LM transformer configuration";
		return false;
	}
	if (cfg.teacher.inputDim == 0u || cfg.teacher.teacherRank == 0u)
	{
		err = "invalid teacher-student configuration";
		return false;
	}
	if (cfg.latent.latentDim < cfg.latent.obsDim || cfg.latent.obsDim == 0u ||
	    cfg.latent.window == 0u || cfg.latent.seqLen <= cfg.latent.window + 1u)
	{
		err = "invalid latent-forecast configuration";
		return false;
	}
	return true;
}

static void normalize_rows(std::vector<float>& w, unsigned int rows, unsigned int cols)
{
	for (unsigned int r = 0u; r < rows; ++r)
	{
		double sumsq = 0.0;
		for (unsigned int c = 0u; c < cols; ++c)
		{
			const float v = w[r * cols + c];
			sumsq += static_cast<double>(v) * static_cast<double>(v);
		}
		const float scale = (sumsq > 1e-12) ? static_cast<float>(1.0 / sqrt(sumsq)) : 1.0f;
		for (unsigned int c = 0u; c < cols; ++c)
			w[r * cols + c] *= scale;
	}
}

static void build_teacher_spec(const TeacherConfig& cfg, unsigned int seed, TeacherSpec& out)
{
	unsigned int state = seed;
	out.signalW.assign(cfg.teacherRank * cfg.inputDim, 0.0f);
	out.signalOut.assign(cfg.teacherRank, 0.0f);
	out.bulkW.assign(cfg.bulkRank * cfg.inputDim, 0.0f);
	out.bulkOut.assign(cfg.bulkRank, 0.0f);
	out.bulkPhase.assign(cfg.bulkRank, 0.0f);

	for (size_t i = 0; i < out.signalW.size(); ++i)
		out.signalW[i] = rand_signed(state);
	for (size_t i = 0; i < out.bulkW.size(); ++i)
		out.bulkW[i] = rand_signed(state);
	normalize_rows(out.signalW, cfg.teacherRank, cfg.inputDim);
	normalize_rows(out.bulkW, cfg.bulkRank, cfg.inputDim);

	for (unsigned int i = 0u; i < cfg.teacherRank; ++i)
		out.signalOut[i] = 0.75f * rand_signed(state);
	for (unsigned int i = 0u; i < cfg.bulkRank; ++i)
	{
		out.bulkOut[i] = 0.35f * rand_signed(state);
		out.bulkPhase[i] = 3.1415926f * rand_signed(state);
	}
}

static float teacher_target(const TeacherConfig& cfg,
                            const TeacherSpec& spec,
                            const float* x)
{
	float signal = 0.0f;
	for (unsigned int r = 0u; r < cfg.teacherRank; ++r)
	{
		float dot = 0.0f;
		for (unsigned int j = 0u; j < cfg.inputDim; ++j)
			dot += spec.signalW[r * cfg.inputDim + j] * x[j];
		signal += spec.signalOut[r] * tanhf(dot);
	}

	float bulk = 0.0f;
	for (unsigned int b = 0u; b < cfg.bulkRank; ++b)
	{
		float dot = 0.0f;
		for (unsigned int j = 0u; j < cfg.inputDim; ++j)
			dot += spec.bulkW[b * cfg.inputDim + j] * x[j];
		bulk += spec.bulkOut[b] * sinf(dot + spec.bulkPhase[b]);
	}

	return tanhf(signal + cfg.bulkScale * bulk);
}

static void build_teacher_dataset(const BenchConfig& cfg, DenseRegressionInput& out)
{
	const TeacherConfig& tcfg = cfg.teacher;
	TeacherSpec spec;
	build_teacher_spec(tcfg, cfg.seed + 17u, spec);

	std::vector<float> trainX(static_cast<size_t>(tcfg.trainSamples) * static_cast<size_t>(tcfg.inputDim), 0.0f);
	std::vector<float> trainY(static_cast<size_t>(tcfg.trainSamples), 0.0f);
	std::vector<float> testX(static_cast<size_t>(tcfg.testSamples) * static_cast<size_t>(tcfg.inputDim), 0.0f);
	std::vector<float> testY(static_cast<size_t>(tcfg.testSamples), 0.0f);

	unsigned int trainState = cfg.seed + 101u;
	for (unsigned int row = 0u; row < tcfg.trainSamples; ++row)
	{
		float* x = &trainX[static_cast<size_t>(row) * static_cast<size_t>(tcfg.inputDim)];
		for (unsigned int j = 0u; j < tcfg.inputDim; ++j)
			x[j] = rand_signed(trainState);
		trainY[row] = teacher_target(tcfg, spec, x);
	}

	unsigned int testState = cfg.seed + 10001u;
	for (unsigned int row = 0u; row < tcfg.testSamples; ++row)
	{
		float* x = &testX[static_cast<size_t>(row) * static_cast<size_t>(tcfg.inputDim)];
		for (unsigned int j = 0u; j < tcfg.inputDim; ++j)
			x[j] = rand_signed(testState);
		testY[row] = teacher_target(tcfg, spec, x);
	}

	out.setTrain(trainX, tcfg.trainSamples, tcfg.inputDim, trainY, 1u);
	out.setTest(testX, tcfg.testSamples, tcfg.inputDim, testY, 1u);
}

static void normalize_projection_rows(std::vector<float>& w, unsigned int rows, unsigned int cols)
{
	for (unsigned int r = 0u; r < rows; ++r)
	{
		double sumsq = 0.0;
		for (unsigned int c = 0u; c < cols; ++c)
		{
			const float v = w[r * cols + c];
			sumsq += static_cast<double>(v) * static_cast<double>(v);
		}
		const float invNorm = (sumsq > 1e-12) ? static_cast<float>(1.0 / sqrt(sumsq)) : 1.0f;
		for (unsigned int c = 0u; c < cols; ++c)
			w[r * cols + c] *= invNorm;
	}
}

static void build_latent_system(const LatentConfig& cfg,
                                unsigned int seed,
                                std::vector<float>& transition,
                                std::vector<float>& observe)
{
	transition.assign(static_cast<size_t>(cfg.latentDim) * static_cast<size_t>(cfg.latentDim), 0.0f);
	observe.assign(static_cast<size_t>(cfg.obsDim) * static_cast<size_t>(cfg.latentDim), 0.0f);

	const float rhoSlow = 0.965f;
	const float angle = 0.31f;
	if (cfg.latentDim >= 2u)
	{
		transition[0u * cfg.latentDim + 0u] = rhoSlow * cosf(angle);
		transition[0u * cfg.latentDim + 1u] = -rhoSlow * sinf(angle);
		transition[1u * cfg.latentDim + 0u] = rhoSlow * sinf(angle);
		transition[1u * cfg.latentDim + 1u] = rhoSlow * cosf(angle);
	}
	for (unsigned int i = 2u; i < cfg.latentDim; ++i)
	{
		const float decay = 0.30f + 0.10f * static_cast<float>((i + 1u) % 4u);
		transition[i * cfg.latentDim + i] = decay;
		transition[i * cfg.latentDim + (i - 1u)] = ((i & 1u) == 0u) ? 0.10f : -0.08f;
	}

	unsigned int state = seed;
	for (size_t i = 0; i < observe.size(); ++i)
		observe[i] = 0.65f * rand_signed(state);
	normalize_projection_rows(observe, cfg.obsDim, cfg.latentDim);
}

static void step_latent_system(const LatentConfig& cfg,
                               const std::vector<float>& transition,
                               const std::vector<float>& observe,
                               unsigned int& state,
                               std::vector<float>& latent,
                               std::vector<float>& obs)
{
	std::vector<float> next(latent.size(), 0.0f);
	for (unsigned int i = 0u; i < cfg.latentDim; ++i)
	{
		float acc = 0.0f;
		for (unsigned int j = 0u; j < cfg.latentDim; ++j)
			acc += transition[i * cfg.latentDim + j] * latent[j];
		next[i] = acc + cfg.processNoise * rand_signed(state);
	}
	latent.swap(next);

	obs.assign(cfg.obsDim, 0.0f);
	for (unsigned int i = 0u; i < cfg.obsDim; ++i)
	{
		float acc = 0.0f;
		for (unsigned int j = 0u; j < cfg.latentDim; ++j)
			acc += observe[i * cfg.latentDim + j] * latent[j];
		obs[i] = acc + cfg.obsNoise * rand_signed(state);
	}
}

static void fill_latent_split(const LatentConfig& cfg,
                              const std::vector<float>& transition,
                              const std::vector<float>& observe,
                              unsigned int seqCount,
                              unsigned int seed,
                              std::vector<float>& outX,
                              std::vector<float>& outY)
{
	const unsigned int inputDim = cfg.obsDim * cfg.window;
	const unsigned int outputDim = cfg.obsDim;
	const unsigned int samplesPerSeq = cfg.seqLen - cfg.window;
	const unsigned int totalSamples = seqCount * samplesPerSeq;
	outX.assign(static_cast<size_t>(totalSamples) * static_cast<size_t>(inputDim), 0.0f);
	outY.assign(static_cast<size_t>(totalSamples) * static_cast<size_t>(outputDim), 0.0f);

	unsigned int state = seed;
	unsigned int row = 0u;
	std::vector<float> latent(cfg.latentDim, 0.0f);
	std::vector<float> obs(cfg.obsDim, 0.0f);
	std::vector<float> seqObs(static_cast<size_t>(cfg.seqLen) * static_cast<size_t>(cfg.obsDim), 0.0f);
	for (unsigned int seq = 0u; seq < seqCount; ++seq)
	{
		for (unsigned int j = 0u; j < cfg.latentDim; ++j)
			latent[j] = 0.25f * rand_signed(state);
		for (unsigned int t = 0u; t < cfg.seqLen; ++t)
		{
			step_latent_system(cfg, transition, observe, state, latent, obs);
			for (unsigned int o = 0u; o < cfg.obsDim; ++o)
				seqObs[static_cast<size_t>(t) * static_cast<size_t>(cfg.obsDim) + o] = obs[o];
		}

		for (unsigned int t = cfg.window; t < cfg.seqLen; ++t, ++row)
		{
			float* inputRow = &outX[static_cast<size_t>(row) * static_cast<size_t>(inputDim)];
			float* targetRow = &outY[static_cast<size_t>(row) * static_cast<size_t>(outputDim)];
			for (unsigned int w = 0u; w < cfg.window; ++w)
			{
				const unsigned int srcT = t - cfg.window + w;
				for (unsigned int o = 0u; o < cfg.obsDim; ++o)
					inputRow[w * cfg.obsDim + o] =
					    seqObs[static_cast<size_t>(srcT) * static_cast<size_t>(cfg.obsDim) + o];
			}
			for (unsigned int o = 0u; o < cfg.obsDim; ++o)
				targetRow[o] = seqObs[static_cast<size_t>(t) * static_cast<size_t>(cfg.obsDim) + o];
		}
	}
}

static void build_latent_dataset(const BenchConfig& cfg, DenseRegressionInput& out)
{
	std::vector<float> transition;
	std::vector<float> observe;
	build_latent_system(cfg.latent, cfg.seed + 717u, transition, observe);

	std::vector<float> trainX;
	std::vector<float> trainY;
	std::vector<float> testX;
	std::vector<float> testY;
	fill_latent_split(cfg.latent, transition, observe, cfg.latent.trainSeqs, cfg.seed + 3001u, trainX, trainY);
	fill_latent_split(cfg.latent, transition, observe, cfg.latent.testSeqs, cfg.seed + 13001u, testX, testY);

	const unsigned int inputDim = cfg.latent.obsDim * cfg.latent.window;
	const unsigned int outputDim = cfg.latent.obsDim;
	const unsigned int trainRows = cfg.latent.trainSeqs * (cfg.latent.seqLen - cfg.latent.window);
	const unsigned int testRows = cfg.latent.testSeqs * (cfg.latent.seqLen - cfg.latent.window);
	out.setTrain(trainX, trainRows, inputDim, trainY, outputDim);
	out.setTest(testX, testRows, inputDim, testY, outputDim);
}

static void step_nonlinear_latent_system(const LatentConfig& cfg,
                                         const std::vector<float>& transition,
                                         const std::vector<float>& observe,
                                         unsigned int& state,
                                         std::vector<float>& latent,
                                         std::vector<float>& obs)
{
	std::vector<float> next(latent.size(), 0.0f);
	const float regime = (latent.empty() || latent[0] >= 0.0f) ? 1.0f : -1.0f;
	for (unsigned int i = 0u; i < cfg.latentDim; ++i)
	{
		float lin = 0.0f;
		for (unsigned int j = 0u; j < cfg.latentDim; ++j)
			lin += transition[i * cfg.latentDim + j] * latent[j];
		const float neighbor = latent[(i + 1u) % cfg.latentDim];
		const float cross = latent[(i + 1u) % cfg.latentDim] * latent[(i + 2u) % cfg.latentDim];
		float driven = 0.82f * lin
		             + cfg.nonlinearMix * tanhf(1.15f * lin)
		             + cfg.switchingScale * regime * neighbor
		             + 0.05f * tanhf(cross);
		driven = 0.85f * driven + 0.15f * tanhf(driven);
		next[i] = driven + cfg.processNoise * rand_signed(state);
	}
	latent.swap(next);

	for (unsigned int i = 0u; i < cfg.obsDim; ++i)
	{
		float acc = 0.0f;
		for (unsigned int j = 0u; j < cfg.latentDim; ++j)
			acc += observe[i * cfg.latentDim + j] * latent[j];
		acc += cfg.observationMix
		     * tanhf(latent[i % cfg.latentDim] * latent[(i + 1u) % cfg.latentDim]);
		obs[i] = acc + cfg.obsNoise * rand_signed(state);
	}
}

static void fill_nonlinear_latent_split(const LatentConfig& cfg,
                                        const std::vector<float>& transition,
                                        const std::vector<float>& observe,
                                        unsigned int seqCount,
                                        unsigned int seed,
                                        std::vector<float>& outX,
                                        std::vector<float>& outY)
{
	const unsigned int inputDim = cfg.obsDim * cfg.window;
	const unsigned int outputDim = cfg.obsDim;
	const unsigned int samplesPerSeq = cfg.seqLen - cfg.window;
	const unsigned int totalSamples = seqCount * samplesPerSeq;
	outX.assign(static_cast<size_t>(totalSamples) * static_cast<size_t>(inputDim), 0.0f);
	outY.assign(static_cast<size_t>(totalSamples) * static_cast<size_t>(outputDim), 0.0f);

	unsigned int state = seed;
	unsigned int row = 0u;
	std::vector<float> latent(cfg.latentDim, 0.0f);
	std::vector<float> obs(cfg.obsDim, 0.0f);
	std::vector<float> seqObs(static_cast<size_t>(cfg.seqLen) * static_cast<size_t>(cfg.obsDim), 0.0f);
	for (unsigned int seq = 0u; seq < seqCount; ++seq)
	{
		for (unsigned int j = 0u; j < cfg.latentDim; ++j)
			latent[j] = 0.25f * rand_signed(state);
		for (unsigned int t = 0u; t < cfg.seqLen; ++t)
		{
			step_nonlinear_latent_system(cfg, transition, observe, state, latent, obs);
			for (unsigned int o = 0u; o < cfg.obsDim; ++o)
				seqObs[static_cast<size_t>(t) * static_cast<size_t>(cfg.obsDim) + o] = obs[o];
		}

		for (unsigned int t = cfg.window; t < cfg.seqLen; ++t, ++row)
		{
			float* inputRow = &outX[static_cast<size_t>(row) * static_cast<size_t>(inputDim)];
			float* targetRow = &outY[static_cast<size_t>(row) * static_cast<size_t>(outputDim)];
			for (unsigned int w = 0u; w < cfg.window; ++w)
			{
				const unsigned int srcT = t - cfg.window + w;
				for (unsigned int o = 0u; o < cfg.obsDim; ++o)
					inputRow[w * cfg.obsDim + o] =
					    seqObs[static_cast<size_t>(srcT) * static_cast<size_t>(cfg.obsDim) + o];
			}
			for (unsigned int o = 0u; o < cfg.obsDim; ++o)
				targetRow[o] = seqObs[static_cast<size_t>(t) * static_cast<size_t>(cfg.obsDim) + o];
		}
	}
}

static void build_nonlinear_latent_dataset(const BenchConfig& cfg, DenseRegressionInput& out)
{
	std::vector<float> transition;
	std::vector<float> observe;
	build_latent_system(cfg.latent, cfg.seed + 1717u, transition, observe);

	std::vector<float> trainX;
	std::vector<float> trainY;
	std::vector<float> testX;
	std::vector<float> testY;
	fill_nonlinear_latent_split(cfg.latent, transition, observe,
	                            cfg.latent.trainSeqs, cfg.seed + 9001u, trainX, trainY);
	fill_nonlinear_latent_split(cfg.latent, transition, observe,
	                            cfg.latent.testSeqs, cfg.seed + 19001u, testX, testY);

	const unsigned int inputDim = cfg.latent.obsDim * cfg.latent.window;
	const unsigned int outputDim = cfg.latent.obsDim;
	const unsigned int trainRows = cfg.latent.trainSeqs * (cfg.latent.seqLen - cfg.latent.window);
	const unsigned int testRows = cfg.latent.testSeqs * (cfg.latent.seqLen - cfg.latent.window);
	out.setTrain(trainX, trainRows, inputDim, trainY, outputDim);
	out.setTest(testX, testRows, inputDim, testY, outputDim);
}

static void build_token_split(unsigned int vocab,
                              unsigned int seqCount,
                              unsigned int seqLen,
                              unsigned int seed,
                              unsigned int padTokenId,
                              std::vector<unsigned int>& outTokens,
                              std::vector<glades::DataInput::SequenceSpan>& outSpans)
{
	outTokens.clear();
	outSpans.clear();
	outTokens.reserve(static_cast<size_t>(seqCount) * static_cast<size_t>(seqLen + 1u));
	outSpans.reserve(seqCount);

	for (unsigned int seq = 0u; seq < seqCount; ++seq)
	{
		const unsigned int start = static_cast<unsigned int>(outTokens.size());
		unsigned int a = mix_u32(seed + 17u * (seq + 1u)) % (vocab - 1u);
		unsigned int b = mix_u32(seed + 31u * (seq + 3u)) % (vocab - 1u);
		const unsigned int phase = 1u + (mix_u32(seed + 97u * (seq + 5u)) % 7u);
		for (unsigned int t = 0u; t < seqLen; ++t)
		{
			unsigned int tok = 0u;
			if (t == 0u)
				tok = a;
			else if (t == 1u)
				tok = b;
			else
			{
				tok = (a + b + phase + ((t / 4u) % 3u)) % (vocab - 1u);
				a = b;
				b = tok;
			}
			outTokens.push_back(tok);
		}
		outSpans.push_back(glades::DataInput::SequenceSpan(start, seqLen));
		outTokens.push_back(padTokenId);
	}
}

static void build_token_dataset(const BenchConfig& cfg, TokenDataset& out)
{
	out = TokenDataset();
	out.padTokenId = cfg.token.vocab - 1u;

	std::vector<unsigned int> trainTokens;
	std::vector<unsigned int> testTokens;
	std::vector<glades::DataInput::SequenceSpan> trainSpans;
	std::vector<glades::DataInput::SequenceSpan> testSpans;

	build_token_split(cfg.token.vocab, cfg.token.trainSeqs, cfg.token.seqLen, cfg.seed + 11u,
	                  out.padTokenId, trainTokens, trainSpans);
	build_token_split(cfg.token.vocab, cfg.token.testSeqs, cfg.token.seqLen, cfg.seed + 1011u,
	                  out.padTokenId, testTokens, testSpans);

	out.di.setTrainTokens(trainTokens, static_cast<int>(out.padTokenId));
	out.di.setTestTokens(testTokens, static_cast<int>(out.padTokenId));
	(void)out.di.setTrainSequences(trainSpans);
	(void)out.di.setTestSequences(testSpans);

	out.trainTokensPerEpoch = static_cast<unsigned long long>(cfg.token.trainSeqs) *
	                          static_cast<unsigned long long>(cfg.token.seqLen);
	out.testTokens = static_cast<unsigned long long>(cfg.token.testSeqs) *
	                 static_cast<unsigned long long>(cfg.token.seqLen);
}

static void reset_atlas_family(glades::TrainingConfig& tc)
{
	tc.atlas.prismEnabled = false;
	tc.atlas.resolveEnabled = false;
	tc.atlas.heroEnabled = false;
	tc.atlas.cobaltEnabled = false;
	tc.atlas.birchEnabled = false;
	tc.atlas.ghostEnabled = false;
	tc.atlas.sparrowEnabled = false;
	tc.atlas.qbrtEnabled = false;
	tc.atlas.qrcEnabled = false;
	tc.atlas.riftEnabled = false;
	tc.atlas.orbitEnabled = false;
	tc.atlas.helmEnabled = false;
	tc.atlas.asterEnabled = false;
}

static void configure_atlas(glades::TrainingConfig& tc,
                            const BenchConfig& cfg,
                            VariantKind variant)
{
	tc.optimizer.type = glades::OptimizerConfig::ATLAS;
	tc.atlas.rank = cfg.atlasRank;
	tc.atlas.complementRank = (variant == VARIANT_ATLAS_SPARROW) ? cfg.atlasComplementRank : 0u;
	tc.atlas.tSub = cfg.atlasTSub;
	tc.atlas.beta = 0.999f;
	tc.atlas.betaRefresh = 0.5f;
	tc.atlas.kappaMax = cfg.atlasKappaMax;
	tc.atlas.complementLrScale = 0.25f;
	tc.atlas.complementKappaMax = 0.5f;
	reset_atlas_family(tc);
	if (variant == VARIANT_ATLAS_SPARROW)
	{
		tc.atlas.sparrowEnabled = true;
		tc.atlas.sparrowModeRank = cfg.atlasSparrowModeRank;
		tc.atlas.sparrowAutoModeGate = (cfg.atlasSparrowAutoModeGate != 0u);
		tc.atlas.sparrowMemoryScale = cfg.atlasSparrowMemoryScale;
		tc.atlas.sparrowEdgeThreshold = cfg.atlasSparrowEdgeThreshold;
		tc.atlas.sparrowSecondEdgeThreshold = cfg.atlasSparrowSecondEdgeThreshold;
		tc.atlas.sparrowSecondEdgeFraction = cfg.atlasSparrowSecondEdgeFraction;
		tc.atlas.sparrowPoleMax = cfg.atlasSparrowPoleMax;
	}
	else if (variant == VARIANT_ATLAS_HELM)
	{
		tc.atlas.helmEnabled = true;
		tc.atlas.helmMemoryScale = cfg.atlasHelmMemoryScale;
		tc.atlas.helmEdgeThreshold = cfg.atlasHelmEdgeThreshold;
		tc.atlas.helmModeRank = cfg.atlasHelmModeRank;
		tc.atlas.helmHiddenStackDepth = cfg.atlasHelmHiddenStackDepth;
		tc.atlas.helmPoleMax = cfg.atlasHelmPoleMax;
	}
	else if (variant == VARIANT_ATLAS_ASTER)
	{
		tc.atlas.asterEnabled = true;
		tc.atlas.asterMemoryScale = cfg.atlasAsterMemoryScale;
		tc.atlas.asterEdgeThreshold = cfg.atlasAsterEdgeThreshold;
		tc.atlas.asterStateRank = cfg.atlasAsterStateRank;
		tc.atlas.asterHiddenStackDepth = cfg.atlasAsterHiddenStackDepth;
		tc.atlas.asterPoleMax = cfg.atlasAsterPoleMax;
	}
}

static bool configure_optimizer(glades::TrainingConfig& tc,
                                const BenchConfig& cfg,
                                VariantKind variant,
                                float adamLR,
                                float atlasLR,
                                float clipNorm)
{
	tc.globalGradClipNorm = clipNorm;
	if (variant == VARIANT_ADAMW)
	{
		tc.optimizer.type = glades::OptimizerConfig::ADAMW;
		tc.optimizer.adamBeta1 = 0.9f;
		tc.optimizer.adamBeta2 = 0.999f;
		tc.optimizer.adamEps = 1e-8f;
		tc.optimizer.adamBiasCorrection = true;
		return true;
	}

	configure_atlas(tc, cfg, variant);
	(void)atlasLR;
	(void)adamLR;
	return true;
}

static glades::NNInfo* build_token_info(const BenchConfig& cfg, float learningRate, const char* name)
{
	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    1,
	    learningRate,
	    0.0f,
	    0.0f,
	    0.0f,
	    0.0f,
	    glades::GMath::LINEAR,
	    1.0f);

	std::vector<glades::HiddenLayerInfo*> hidden;
	for (unsigned int i = 0u; i < cfg.token.layers; ++i)
	{
		hidden.push_back(new glades::HiddenLayerInfo(
		    static_cast<int>(cfg.token.dModel),
		    learningRate,
		    0.0f,
		    0.0f,
		    0.0f,
		    0.0f,
		    glades::GMath::LINEAR,
		    1.0f));
	}

	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(cfg.token.vocab),
	                                                           glades::OutputLayerInfo::CLASSIFICATION);
	return new glades::NNInfo(name, in, hidden, out);
}

static bool make_token_network(const BenchConfig& cfg,
                               VariantKind variant,
                               unsigned int seed,
                               unsigned int padTokenId,
                               NetworkOwner& out,
                               std::string& err)
{
	const float lr = (variant == VARIANT_ADAMW) ? cfg.token.adamLR : cfg.token.atlasLR;
	out.info = build_token_info(cfg, lr, "atlas_alt_token_lm");
	out.net = new glades::NNetwork(out.info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
	out.net->setSeed(seed);
	out.net->setLogger(quiet_logger());
	out.net->getTerminatorMutable().setEpoch(static_cast<int>(cfg.token.epochs));
	out.net->getTerminatorMutable().setAccuracy(0.0f);

	glades::TrainingConfig tc = out.net->getTrainingConfig();
	tc.transformer.enableTokenEmbedding = true;
	tc.transformer.vocabSizeOverride = static_cast<int>(cfg.token.vocab);
	tc.transformer.tieEmbeddings = true;
	tc.transformer.padTokenId = static_cast<int>(padTokenId);
	tc.transformer.nHeadsOverride = static_cast<int>(cfg.token.heads);
	tc.transformer.nKVHeadsOverride = static_cast<int>(cfg.token.kvHeads);
	tc.transformer.dFFOverride = static_cast<int>(cfg.token.dFF);
	tc.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
	tc.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
	tc.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
	tc.transformer.ropeTheta = 10000.0f;
	configure_optimizer(tc, cfg, variant, cfg.token.adamLR, cfg.token.atlasLR, 1.0f);

	const glades::NNetworkStatus stCfg = out.net->setTrainingConfig(tc);
	if (!stCfg.ok())
	{
		err = stCfg.message;
		return false;
	}
	return true;
}

static glades::NNetwork make_regression_network(const BenchConfig& cfg,
                                                const RegressionNetworkSpec& spec,
                                                unsigned int epochs,
                                                VariantKind variant,
                                                unsigned int seed)
{
	const float lr = (variant == VARIANT_ADAMW) ? spec.adamLR : spec.atlasLR;
	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    static_cast<int>(spec.batchSize),
	    lr,
	    0.0f,
	    0.0f,
	    0.0f,
	    0.0f,
	    glades::GMath::LINEAR,
	    1.0f);

	std::vector<glades::HiddenLayerInfo*> hidden;
	for (size_t i = 0; i < spec.hiddenSizes.size(); ++i)
	{
		hidden.push_back(new glades::HiddenLayerInfo(
		    static_cast<int>(spec.hiddenSizes[i]),
		    lr,
		    0.0f,
		    0.0f,
		    0.0f,
		    0.0f,
		    glades::GMath::RELU,
		    1.0f));
	}

	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(spec.outputDim),
	                                                           glades::OutputLayerInfo::REGRESSION);
	glades::NNInfo* info = new glades::NNInfo(spec.name, in, hidden, out);
	glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
	net.setSeed(seed);
	net.setLogger(quiet_logger());
	net.getTerminatorMutable().setEpoch(static_cast<int>(epochs));
	net.getTerminatorMutable().setAccuracy(0.0f);

	glades::TrainingConfig tc = net.getTrainingConfig();
	configure_optimizer(tc, cfg, variant, spec.adamLR, spec.atlasLR, spec.clipNorm);
	net.setTrainingConfig(tc);

	delete info;
	return net;
}

static glades::NNetwork make_teacher_network(const BenchConfig& cfg,
                                             VariantKind variant,
                                             unsigned int seed)
{
	RegressionNetworkSpec spec;
	spec.name = "atlas_alt_teacher_student";
	spec.batchSize = cfg.teacher.batchSize;
	spec.hiddenSizes.push_back(64u);
	spec.hiddenSizes.push_back(32u);
	spec.outputDim = 1u;
	spec.adamLR = cfg.teacher.adamLR;
	spec.atlasLR = cfg.teacher.atlasLR;
	spec.clipNorm = 5.0f;
	return make_regression_network(cfg, spec, cfg.teacher.epochs, variant, seed);
}

static glades::NNetwork make_latent_network(const BenchConfig& cfg,
                                            VariantKind variant,
                                            unsigned int seed)
{
	RegressionNetworkSpec spec;
	spec.name = "atlas_alt_latent_forecast";
	spec.batchSize = cfg.latent.batchSize;
	spec.hiddenSizes.push_back(96u);
	spec.hiddenSizes.push_back(64u);
	spec.outputDim = cfg.latent.obsDim;
	spec.adamLR = cfg.latent.adamLR;
	spec.atlasLR = cfg.latent.atlasLR;
	spec.clipNorm = 5.0f;
	return make_regression_network(cfg, spec, cfg.latent.epochs, variant, seed);
}

static glades::NNetwork make_nonlinear_latent_network(const BenchConfig& cfg,
                                                      VariantKind variant,
                                                      unsigned int seed)
{
	RegressionNetworkSpec spec;
	spec.name = "atlas_alt_nonlinear_forecast";
	spec.batchSize = cfg.latent.batchSize;
	spec.hiddenSizes.push_back(128u);
	spec.hiddenSizes.push_back(96u);
	spec.outputDim = cfg.latent.obsDim;
	spec.adamLR = cfg.latent.adamLR;
	spec.atlasLR = cfg.latent.atlasLR;
	spec.clipNorm = 5.0f;
	return make_regression_network(cfg, spec, cfg.latent.epochs, variant, seed);
}

static RunResult run_token_variant(const BenchConfig& cfg,
                                   const TokenDataset& data,
                                   VariantKind variant,
                                   unsigned int seed)
{
	RunResult out;
	out.label = variant_label(variant);
	NetworkOwner owner;
	if (!make_token_network(cfg, variant, seed, data.padTokenId, owner, out.err))
	{
		out.ok = false;
		return out;
	}

	const glades::NNetworkStatus warm = owner.net->test(const_cast<InMemoryTokenIdInput*>(&data.di));
	if (!warm.ok())
	{
		out.ok = false;
		out.err = warm.message;
		return out;
	}

	CaptureMetricsCallbacks trainCb;
	const int64_t t0 = now_ms();
	const glades::NNetworkStatus trainStatus = owner.net->train(const_cast<InMemoryTokenIdInput*>(&data.di), &trainCb);
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
		out.err = "token-LM training completed without metrics";
		return out;
	}
	fill_sparrow_run_result(trainCb, out);
	fill_helm_run_result(trainCb, out);
	fill_aster_run_result(trainCb, out);

	CaptureMetricsCallbacks testCb;
	const int64_t t2 = now_ms();
	const glades::NNetworkStatus testStatus = owner.net->test(const_cast<InMemoryTokenIdInput*>(&data.di), &testCb);
	const int64_t t3 = now_ms();
	out.evalMs = static_cast<long long>(t3 - t2);
	if (!testStatus.ok())
	{
		out.ok = false;
		out.err = testStatus.message;
		return out;
	}
	if (!testCb.saw)
	{
		out.ok = false;
		out.err = "token-LM evaluation completed without metrics";
		return out;
	}

	out.trainLoss = trainCb.last.totalError;
	out.trainMetric = (trainCb.last.perplexity > 0.0f) ? trainCb.last.perplexity : safe_exp(trainCb.last.totalError);
	out.testLoss = testCb.last.totalError;
	out.testMetric = (testCb.last.perplexity > 0.0f) ? testCb.last.perplexity : safe_exp(testCb.last.totalError);

	const double seconds = static_cast<double>(out.trainMs) / 1000.0;
	const double tokens = static_cast<double>(data.trainTokensPerEpoch) * static_cast<double>(cfg.token.epochs);
	out.throughput = (seconds > 0.0) ? (tokens / seconds) : 0.0;
	return out;
}

static RunResult run_teacher_variant(const BenchConfig& cfg,
                                     DenseRegressionInput& data,
                                     VariantKind variant,
                                     unsigned int seed)
{
	RunResult out;
	out.label = variant_label(variant);
	glades::NNetwork net = make_teacher_network(cfg, variant, seed);

	const glades::NNetworkStatus warm = net.test(&data);
	if (!warm.ok())
	{
		out.ok = false;
		out.err = warm.message;
		return out;
	}

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
		out.err = "teacher-student training completed without metrics";
		return out;
	}
	fill_sparrow_run_result(trainCb, out);
	fill_helm_run_result(trainCb, out);
	fill_aster_run_result(trainCb, out);

	CaptureMetricsCallbacks testCb;
	const int64_t t2 = now_ms();
	const glades::NNetworkStatus testStatus = net.test(&data, &testCb);
	const int64_t t3 = now_ms();
	out.evalMs = static_cast<long long>(t3 - t2);
	if (!testStatus.ok())
	{
		out.ok = false;
		out.err = testStatus.message;
		return out;
	}
	if (!testCb.saw)
	{
		out.ok = false;
		out.err = "teacher-student evaluation completed without metrics";
		return out;
	}

	out.trainLoss = trainCb.last.totalError;
	out.trainMetric = trainCb.last.totalAccuracy;
	out.testLoss = testCb.last.totalError;
	out.testMetric = testCb.last.totalAccuracy;

	const double seconds = static_cast<double>(out.trainMs) / 1000.0;
	const double samples = static_cast<double>(cfg.teacher.trainSamples) * static_cast<double>(cfg.teacher.epochs);
	out.throughput = (seconds > 0.0) ? (samples / seconds) : 0.0;
	return out;
}

static RunResult run_latent_variant(const BenchConfig& cfg,
                                    DenseRegressionInput& data,
                                    VariantKind variant,
                                    unsigned int seed)
{
	RunResult out;
	out.label = variant_label(variant);
	glades::NNetwork net = make_latent_network(cfg, variant, seed);

	const glades::NNetworkStatus warm = net.test(&data);
	if (!warm.ok())
	{
		out.ok = false;
		out.err = warm.message;
		return out;
	}

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
		out.err = "latent-forecast training completed without metrics";
		return out;
	}
	fill_sparrow_run_result(trainCb, out);
	fill_helm_run_result(trainCb, out);
	fill_aster_run_result(trainCb, out);

	CaptureMetricsCallbacks testCb;
	const int64_t t2 = now_ms();
	const glades::NNetworkStatus testStatus = net.test(&data, &testCb);
	const int64_t t3 = now_ms();
	out.evalMs = static_cast<long long>(t3 - t2);
	if (!testStatus.ok())
	{
		out.ok = false;
		out.err = testStatus.message;
		return out;
	}
	if (!testCb.saw)
	{
		out.ok = false;
		out.err = "latent-forecast evaluation completed without metrics";
		return out;
	}

	out.trainLoss = trainCb.last.totalError;
	out.trainMetric = trainCb.last.totalAccuracy;
	out.testLoss = testCb.last.totalError;
	out.testMetric = testCb.last.totalAccuracy;

	const double seconds = static_cast<double>(out.trainMs) / 1000.0;
	const double windows = static_cast<double>(cfg.latent.trainSeqs) *
	                       static_cast<double>(cfg.latent.seqLen - cfg.latent.window) *
	                       static_cast<double>(cfg.latent.epochs);
	out.throughput = (seconds > 0.0) ? (windows / seconds) : 0.0;
	return out;
}

static RunResult run_nonlinear_latent_variant(const BenchConfig& cfg,
                                              DenseRegressionInput& data,
                                              VariantKind variant,
                                              unsigned int seed)
{
	RunResult out;
	out.label = variant_label(variant);
	glades::NNetwork net = make_nonlinear_latent_network(cfg, variant, seed);

	const glades::NNetworkStatus warm = net.test(&data);
	if (!warm.ok())
	{
		out.ok = false;
		out.err = warm.message;
		return out;
	}

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
		out.err = "nonlinear-forecast training completed without metrics";
		return out;
	}
	fill_sparrow_run_result(trainCb, out);
	fill_helm_run_result(trainCb, out);
	fill_aster_run_result(trainCb, out);

	CaptureMetricsCallbacks testCb;
	const int64_t t2 = now_ms();
	const glades::NNetworkStatus testStatus = net.test(&data, &testCb);
	const int64_t t3 = now_ms();
	out.evalMs = static_cast<long long>(t3 - t2);
	if (!testStatus.ok())
	{
		out.ok = false;
		out.err = testStatus.message;
		return out;
	}
	if (!testCb.saw)
	{
		out.ok = false;
		out.err = "nonlinear-forecast evaluation completed without metrics";
		return out;
	}

	out.trainLoss = trainCb.last.totalError;
	out.trainMetric = trainCb.last.totalAccuracy;
	out.testLoss = testCb.last.totalError;
	out.testMetric = testCb.last.totalAccuracy;

	const double seconds = static_cast<double>(out.trainMs) / 1000.0;
	const double windows = static_cast<double>(cfg.latent.trainSeqs) *
	                       static_cast<double>(cfg.latent.seqLen - cfg.latent.window) *
	                       static_cast<double>(cfg.latent.epochs);
	out.throughput = (seconds > 0.0) ? (windows / seconds) : 0.0;
	return out;
}

static void print_summary_row(const Summary& s)
{
	printf("%-15s  %7.2f +/- %-7.2f  %10.1f +/- %-10.1f  %9.5f +/- %-9.5f  %9.3f +/- %-9.3f  %9.5f +/- %-9.5f  %9.3f +/- %-9.3f  %s\n",
	       s.label,
	       s.trainSec.mean, s.trainSec.stddev,
	       s.throughput.mean, s.throughput.stddev,
	       s.trainLoss.mean, s.trainLoss.stddev,
	       s.trainMetric.mean, s.trainMetric.stddev,
	       s.testLoss.mean, s.testLoss.stddev,
	       s.testMetric.mean, s.testMetric.stddev,
	       s.status.c_str());
}

static void print_sparrow_usage_row(const Summary& s)
{
	if (!s.sparrowDiagValid)
		return;
	printf("  SPARROW usage: modes=%5.2f +/- %-5.2f  mode2Frac=%5.3f +/- %-5.3f  edge2/1=%5.3f +/- %-5.3f  edge=%5.3f +/- %-5.3f  mem=%5.3f +/- %-5.3f  horiz=%5.3f +/- %-5.3f\n",
	       s.sparrowActiveModes.mean, s.sparrowActiveModes.stddev,
	       s.sparrowMode2Fraction.mean, s.sparrowMode2Fraction.stddev,
	       s.sparrowSecondEdgeRatio.mean, s.sparrowSecondEdgeRatio.stddev,
	       s.sparrowEdge.mean, s.sparrowEdge.stddev,
	       s.sparrowMemoryGain.mean, s.sparrowMemoryGain.stddev,
	       s.sparrowHorizontalRatio.mean, s.sparrowHorizontalRatio.stddev);
}

static void print_helm_usage_row(const Summary& s)
{
	if (!s.helmDiagValid)
		return;
	printf("  HELM usage:    modes=%5.2f +/- %-5.2f  mode2Frac=%5.3f +/- %-5.3f  edge2/1=%5.3f +/- %-5.3f  edge=%5.3f +/- %-5.3f  sigma=%5.3f +/- %-5.3f  predR2=%5.3f +/- %-5.3f  mem=%5.3f +/- %-5.3f  pole=%5.3f +/- %-5.3f\n",
	       s.helmActiveModes.mean, s.helmActiveModes.stddev,
	       s.helmMode2Fraction.mean, s.helmMode2Fraction.stddev,
	       s.helmSecondEdgeRatio.mean, s.helmSecondEdgeRatio.stddev,
	       s.helmEdge.mean, s.helmEdge.stddev,
	       s.helmSigma.mean, s.helmSigma.stddev,
	       s.helmPredR2.mean, s.helmPredR2.stddev,
	       s.helmMemoryGain.mean, s.helmMemoryGain.stddev,
	       s.helmPole.mean, s.helmPole.stddev);
}

static void print_aster_usage_row(const Summary& s)
{
	if (!s.asterDiagValid)
		return;
	printf("  ASTER usage:   modes=%5.2f +/- %-5.2f  mode2Frac=%5.3f +/- %-5.3f  edge2/1=%5.3f +/- %-5.3f  edge=%5.3f +/- %-5.3f  sigma=%5.3f +/- %-5.3f  predR2=%5.3f +/- %-5.3f  mem=%5.3f +/- %-5.3f  pole=%5.3f +/- %-5.3f\n",
	       s.asterActiveModes.mean, s.asterActiveModes.stddev,
	       s.asterMode2Fraction.mean, s.asterMode2Fraction.stddev,
	       s.asterSecondEdgeRatio.mean, s.asterSecondEdgeRatio.stddev,
	       s.asterEdge.mean, s.asterEdge.stddev,
	       s.asterSigma.mean, s.asterSigma.stddev,
	       s.asterPredR2.mean, s.asterPredR2.stddev,
	       s.asterMemoryGain.mean, s.asterMemoryGain.stddev,
	       s.asterPole.mean, s.asterPole.stddev);
}

static void fill_sparrow_run_result(const CaptureMetricsCallbacks& cb, RunResult& out)
{
	if (!cb.sawAtlas || cb.atlasSparrowEpochs == 0u)
		return;
	const double denom = static_cast<double>(cb.atlasSparrowEpochs);
	out.sparrowDiagValid = true;
	out.sparrowActiveModes = cb.atlasSparrowActiveModesSum / denom;
	out.sparrowMode2Fraction = cb.atlasSparrowMode2FractionSum / denom;
	out.sparrowSecondEdgeRatio = cb.atlasSparrowSecondEdgeRatioSum / denom;
	out.sparrowEdge = cb.atlasSparrowEdgeSum / denom;
	out.sparrowMemoryGain = cb.atlasSparrowMemoryGainSum / denom;
	out.sparrowHorizontalRatio = cb.atlasSparrowHorizontalRatioSum / denom;
}

static void fill_helm_run_result(const CaptureMetricsCallbacks& cb, RunResult& out)
{
	if (!cb.sawAtlas || cb.atlasHelmEpochs == 0u)
		return;
	const double denom = static_cast<double>(cb.atlasHelmEpochs);
	out.helmDiagValid = true;
	out.helmActiveModes = cb.atlasHelmActiveModesSum / denom;
	out.helmMode2Fraction = static_cast<double>(cb.atlasHelmMode2Epochs) / denom;
	out.helmSecondEdgeRatio = cb.atlasHelmSecondEdgeRatioSum / denom;
	out.helmEdge = cb.atlasHelmEdgeSum / denom;
	out.helmSigma = cb.atlasHelmSigmaSum / denom;
	out.helmPredR2 = cb.atlasHelmPredR2Sum / denom;
	out.helmMemoryGain = cb.atlasHelmMemoryGainSum / denom;
	out.helmPole = cb.atlasHelmPoleSum / denom;
}

static void fill_aster_run_result(const CaptureMetricsCallbacks& cb, RunResult& out)
{
	if (!cb.sawAtlas || cb.atlasAsterEpochs == 0u)
		return;
	const double denom = static_cast<double>(cb.atlasAsterEpochs);
	out.asterDiagValid = true;
	out.asterActiveModes = cb.atlasAsterActiveModesSum / denom;
	out.asterMode2Fraction = static_cast<double>(cb.atlasAsterMode2Epochs) / denom;
	out.asterSecondEdgeRatio = cb.atlasAsterSecondEdgeRatioSum / denom;
	out.asterEdge = cb.atlasAsterEdgeSum / denom;
	out.asterSigma = cb.atlasAsterSigmaSum / denom;
	out.asterPredR2 = cb.atlasAsterPredR2Sum / denom;
	out.asterMemoryGain = cb.atlasAsterMemoryGainSum / denom;
	out.asterPole = cb.atlasAsterPoleSum / denom;
}

static Summary summarize_runs(const char* label, const std::vector<RunResult>& runs)
{
	Summary s;
	s.label = label;
	if (runs.empty())
	{
		s.status = "no runs";
		return s;
	}

	std::vector<double> trainSecVals;
	std::vector<double> throughputVals;
	std::vector<double> trainLossVals;
	std::vector<double> trainMetricVals;
	std::vector<double> testLossVals;
	std::vector<double> testMetricVals;
	std::vector<double> sparrowActiveModesVals;
	std::vector<double> sparrowMode2FractionVals;
	std::vector<double> sparrowSecondEdgeRatioVals;
	std::vector<double> sparrowEdgeVals;
	std::vector<double> sparrowMemoryGainVals;
	std::vector<double> sparrowHorizontalRatioVals;
	std::vector<double> helmActiveModesVals;
	std::vector<double> helmMode2FractionVals;
	std::vector<double> helmSecondEdgeRatioVals;
	std::vector<double> helmEdgeVals;
	std::vector<double> helmSigmaVals;
	std::vector<double> helmPredR2Vals;
	std::vector<double> helmMemoryGainVals;
	std::vector<double> helmPoleVals;
	std::vector<double> asterActiveModesVals;
	std::vector<double> asterMode2FractionVals;
	std::vector<double> asterSecondEdgeRatioVals;
	std::vector<double> asterEdgeVals;
	std::vector<double> asterSigmaVals;
	std::vector<double> asterPredR2Vals;
	std::vector<double> asterMemoryGainVals;
	std::vector<double> asterPoleVals;
	bool allOk = true;
	std::string firstErr;
	for (size_t i = 0; i < runs.size(); ++i)
	{
		if (!runs[i].ok)
		{
			allOk = false;
			if (firstErr.empty())
				firstErr = runs[i].err;
			continue;
		}
		trainSecVals.push_back(static_cast<double>(runs[i].trainMs) / 1000.0);
		throughputVals.push_back(runs[i].throughput);
		trainLossVals.push_back(runs[i].trainLoss);
		trainMetricVals.push_back(runs[i].trainMetric);
		testLossVals.push_back(runs[i].testLoss);
		testMetricVals.push_back(runs[i].testMetric);
		if (runs[i].sparrowDiagValid)
		{
			sparrowActiveModesVals.push_back(runs[i].sparrowActiveModes);
			sparrowMode2FractionVals.push_back(runs[i].sparrowMode2Fraction);
			sparrowSecondEdgeRatioVals.push_back(runs[i].sparrowSecondEdgeRatio);
			sparrowEdgeVals.push_back(runs[i].sparrowEdge);
			sparrowMemoryGainVals.push_back(runs[i].sparrowMemoryGain);
			sparrowHorizontalRatioVals.push_back(runs[i].sparrowHorizontalRatio);
		}
		if (runs[i].helmDiagValid)
		{
			helmActiveModesVals.push_back(runs[i].helmActiveModes);
			helmMode2FractionVals.push_back(runs[i].helmMode2Fraction);
			helmSecondEdgeRatioVals.push_back(runs[i].helmSecondEdgeRatio);
			helmEdgeVals.push_back(runs[i].helmEdge);
			helmSigmaVals.push_back(runs[i].helmSigma);
			helmPredR2Vals.push_back(runs[i].helmPredR2);
			helmMemoryGainVals.push_back(runs[i].helmMemoryGain);
			helmPoleVals.push_back(runs[i].helmPole);
		}
		if (runs[i].asterDiagValid)
		{
			asterActiveModesVals.push_back(runs[i].asterActiveModes);
			asterMode2FractionVals.push_back(runs[i].asterMode2Fraction);
			asterSecondEdgeRatioVals.push_back(runs[i].asterSecondEdgeRatio);
			asterEdgeVals.push_back(runs[i].asterEdge);
			asterSigmaVals.push_back(runs[i].asterSigma);
			asterPredR2Vals.push_back(runs[i].asterPredR2);
			asterMemoryGainVals.push_back(runs[i].asterMemoryGain);
			asterPoleVals.push_back(runs[i].asterPole);
		}
	}

	s.ok = allOk && !trainSecVals.empty();
	s.status = s.ok ? "ok" : firstErr;
	s.trainSec = compute_stats(trainSecVals);
	s.throughput = compute_stats(throughputVals);
	s.trainLoss = compute_stats(trainLossVals);
	s.trainMetric = compute_stats(trainMetricVals);
	s.testLoss = compute_stats(testLossVals);
	s.testMetric = compute_stats(testMetricVals);
	s.sparrowDiagValid = !sparrowActiveModesVals.empty();
	s.sparrowActiveModes = compute_stats(sparrowActiveModesVals);
	s.sparrowMode2Fraction = compute_stats(sparrowMode2FractionVals);
	s.sparrowSecondEdgeRatio = compute_stats(sparrowSecondEdgeRatioVals);
	s.sparrowEdge = compute_stats(sparrowEdgeVals);
	s.sparrowMemoryGain = compute_stats(sparrowMemoryGainVals);
	s.sparrowHorizontalRatio = compute_stats(sparrowHorizontalRatioVals);
	s.helmDiagValid = !helmEdgeVals.empty();
	s.helmActiveModes = compute_stats(helmActiveModesVals);
	s.helmMode2Fraction = compute_stats(helmMode2FractionVals);
	s.helmSecondEdgeRatio = compute_stats(helmSecondEdgeRatioVals);
	s.helmEdge = compute_stats(helmEdgeVals);
	s.helmSigma = compute_stats(helmSigmaVals);
	s.helmPredR2 = compute_stats(helmPredR2Vals);
	s.helmMemoryGain = compute_stats(helmMemoryGainVals);
	s.helmPole = compute_stats(helmPoleVals);
	s.asterDiagValid = !asterEdgeVals.empty();
	s.asterActiveModes = compute_stats(asterActiveModesVals);
	s.asterMode2Fraction = compute_stats(asterMode2FractionVals);
	s.asterSecondEdgeRatio = compute_stats(asterSecondEdgeRatioVals);
	s.asterEdge = compute_stats(asterEdgeVals);
	s.asterSigma = compute_stats(asterSigmaVals);
	s.asterPredR2 = compute_stats(asterPredR2Vals);
	s.asterMemoryGain = compute_stats(asterMemoryGainVals);
	s.asterPole = compute_stats(asterPoleVals);
	return s;
}

static void fill_teacher_sweep_axes(SweepProfile profile,
                                    std::vector<unsigned int>& teacherRanks,
                                    std::vector<unsigned int>& bulkRanks,
                                    std::vector<float>& bulkScales,
                                    std::vector<unsigned int>& cRanks)
{
	teacherRanks.clear();
	bulkRanks.clear();
	bulkScales.clear();
	cRanks.clear();

	teacherRanks.push_back(2u);
	teacherRanks.push_back(4u);
	teacherRanks.push_back(8u);

	if (profile == SWEEP_FULL)
	{
		bulkRanks.push_back(0u);
		bulkRanks.push_back(4u);
		bulkRanks.push_back(12u);
		bulkRanks.push_back(24u);

		bulkScales.push_back(0.0f);
		bulkScales.push_back(0.1f);
		bulkScales.push_back(0.2f);
		bulkScales.push_back(0.4f);
		bulkScales.push_back(0.8f);

		cRanks.push_back(0u);
		cRanks.push_back(2u);
		cRanks.push_back(4u);
		cRanks.push_back(8u);
		return;
	}

	bulkRanks.push_back(0u);
	bulkRanks.push_back(12u);
	bulkRanks.push_back(24u);

	bulkScales.push_back(0.0f);
	bulkScales.push_back(0.2f);
	bulkScales.push_back(0.8f);

	cRanks.push_back(0u);
	cRanks.push_back(4u);
}

static bool run_token_case(const BenchConfig& cfg)
{
	TokenDataset data;
	build_token_dataset(cfg, data);

	printf("------------------------------------------------------------\n");
	printf("Case: token-lm\n");
	printf("Description: autoregressive next-token prediction with a small decoder-only transformer on a synthetic order-2 recurrence.\n");
	printf("Config: vocab=%u dModel=%u dFF=%u layers=%u heads=%u seqLen=%u trainSeqs=%u testSeqs=%u epochs=%u repeats=%u\n",
	       cfg.token.vocab, cfg.token.dModel, cfg.token.dFF, cfg.token.layers, cfg.token.heads,
	       cfg.token.seqLen, cfg.token.trainSeqs, cfg.token.testSeqs, cfg.token.epochs, cfg.repeats);
	printf("Optimizers: AdamW(lr=%.4f) ATLAS-BSRP(lr=%.4f cRank=0) ATLAS-SPARROW(lr=%.4f cRank=%u modeRankCap=%u autoGate=%u)\n",
	       cfg.token.adamLR, cfg.token.atlasLR, cfg.token.atlasLR, cfg.atlasComplementRank,
	       cfg.atlasSparrowModeRank, cfg.atlasSparrowAutoModeGate);
	printf("ATLAS: rank=%u tSub=%u kappaMax=%.3f sparrow(modeRankCap=%u autoGate=%u memoryScale=%.3f edge=%.3f secondEdge=%.3f secondFrac=%.3f poleMax=%.3f)\n",
	       cfg.atlasRank, cfg.atlasTSub, cfg.atlasKappaMax,
	       cfg.atlasSparrowModeRank,
	       cfg.atlasSparrowAutoModeGate,
	       cfg.atlasSparrowMemoryScale, cfg.atlasSparrowEdgeThreshold,
	       cfg.atlasSparrowSecondEdgeThreshold, cfg.atlasSparrowSecondEdgeFraction,
	       cfg.atlasSparrowPoleMax);
	printf("\n");
	printf("%-15s  %7s          %10s            %9s           %9s           %9s           %9s         %s\n",
	       "Optimizer", "Train(s)", "Tok/s", "TrainNLL", "TrainPPL", "TestNLL", "TestPPL", "Status");

	const VariantKind variants[] = { VARIANT_ADAMW, VARIANT_ATLAS_BASE, VARIANT_ATLAS_SPARROW };
	const size_t variantCount = sizeof(variants) / sizeof(variants[0]);
	for (size_t v = 0; v < variantCount; ++v)
	{
		std::vector<RunResult> runs;
		runs.reserve(cfg.repeats);
		for (unsigned int rep = 0u; rep < cfg.repeats; ++rep)
			runs.push_back(run_token_variant(cfg, data, variants[v], cfg.seed + static_cast<unsigned int>(100u * v) + rep));
		const Summary s = summarize_runs(variant_label(variants[v]), runs);
		print_summary_row(s);
		print_sparrow_usage_row(s);
	}
	printf("\n");
	return true;
}

static bool run_latent_case(const BenchConfig& cfg)
{
	DenseRegressionInput data;
	build_latent_dataset(cfg, data);

	printf("------------------------------------------------------------\n");
	printf("Case: latent-forecast\n");
	printf("Description: partially observed latent linear dynamical system with windowed sequence forecasting.\n");
	printf("Config: latentDim=%u obsDim=%u window=%u seqLen=%u trainSeqs=%u testSeqs=%u batch=%u epochs=%u repeats=%u\n",
	       cfg.latent.latentDim, cfg.latent.obsDim, cfg.latent.window, cfg.latent.seqLen,
	       cfg.latent.trainSeqs, cfg.latent.testSeqs, cfg.latent.batchSize, cfg.latent.epochs, cfg.repeats);
	printf("Noise: process=%.3f observation=%.3f\n", cfg.latent.processNoise, cfg.latent.obsNoise);
	printf("Optimizers: AdamW(lr=%.4f) ATLAS-BSRP(lr=%.4f cRank=0) ATLAS-SPARROW(lr=%.4f cRank=%u modeRankCap=%u autoGate=%u) ATLAS-HELM(lr=%.4f) ATLAS-ASTER(lr=%.4f)\n",
	       cfg.latent.adamLR, cfg.latent.atlasLR, cfg.latent.atlasLR, cfg.atlasComplementRank,
	       cfg.atlasSparrowModeRank, cfg.atlasSparrowAutoModeGate, cfg.latent.atlasLR, cfg.latent.atlasLR);
	printf("ATLAS: rank=%u tSub=%u kappaMax=%.3f sparrow(modeRankCap=%u autoGate=%u memoryScale=%.3f edge=%.3f secondEdge=%.3f secondFrac=%.3f poleMax=%.3f) helm(modeRank=%u hiddenStack=%u memoryScale=%.3f edge=%.3f poleMax=%.3f) aster(stateRank=%u hiddenStack=%u memoryScale=%.3f edge=%.3f poleMax=%.3f)\n",
	       cfg.atlasRank, cfg.atlasTSub, cfg.atlasKappaMax,
	       cfg.atlasSparrowModeRank,
	       cfg.atlasSparrowAutoModeGate,
	       cfg.atlasSparrowMemoryScale, cfg.atlasSparrowEdgeThreshold,
	       cfg.atlasSparrowSecondEdgeThreshold, cfg.atlasSparrowSecondEdgeFraction,
	       cfg.atlasSparrowPoleMax,
	       cfg.atlasHelmModeRank, cfg.atlasHelmHiddenStackDepth,
	       cfg.atlasHelmMemoryScale, cfg.atlasHelmEdgeThreshold, cfg.atlasHelmPoleMax,
	       cfg.atlasAsterStateRank, cfg.atlasAsterHiddenStackDepth,
	       cfg.atlasAsterMemoryScale, cfg.atlasAsterEdgeThreshold, cfg.atlasAsterPoleMax);
	printf("\n");
	printf("%-15s  %7s          %10s            %9s           %9s           %9s           %9s         %s\n",
	       "Optimizer", "Train(s)", "Windows/s", "TrainMSE", "TrainR2%", "TestMSE", "TestR2%", "Status");

	const VariantKind variants[] = { VARIANT_ADAMW, VARIANT_ATLAS_BASE, VARIANT_ATLAS_SPARROW, VARIANT_ATLAS_HELM, VARIANT_ATLAS_ASTER };
	const size_t variantCount = sizeof(variants) / sizeof(variants[0]);
	for (size_t v = 0; v < variantCount; ++v)
	{
		std::vector<RunResult> runs;
		runs.reserve(cfg.repeats);
		for (unsigned int rep = 0u; rep < cfg.repeats; ++rep)
			runs.push_back(run_latent_variant(cfg, data, variants[v], cfg.seed + static_cast<unsigned int>(2000u + 100u * v) + rep));
		const Summary s = summarize_runs(variant_label(variants[v]), runs);
		print_summary_row(s);
		print_sparrow_usage_row(s);
		print_helm_usage_row(s);
		print_aster_usage_row(s);
	}
	printf("\n");
	return true;
}

static bool run_nonlinear_case(const BenchConfig& cfg)
{
	DenseRegressionInput data;
	build_nonlinear_latent_dataset(cfg, data);

	printf("------------------------------------------------------------\n");
	printf("Case: nonlinear-forecast\n");
	printf("Description: partially observed switching/tanh latent dynamics with windowed sequence forecasting.\n");
	printf("Config: latentDim=%u obsDim=%u window=%u seqLen=%u trainSeqs=%u testSeqs=%u batch=%u epochs=%u repeats=%u\n",
	       cfg.latent.latentDim, cfg.latent.obsDim, cfg.latent.window, cfg.latent.seqLen,
	       cfg.latent.trainSeqs, cfg.latent.testSeqs, cfg.latent.batchSize, cfg.latent.epochs, cfg.repeats);
	printf("Noise: process=%.3f observation=%.3f nonlinearMix=%.3f switching=%.3f obsMix=%.3f\n",
	       cfg.latent.processNoise, cfg.latent.obsNoise,
	       cfg.latent.nonlinearMix, cfg.latent.switchingScale, cfg.latent.observationMix);
	printf("Optimizers: AdamW(lr=%.4f) ATLAS-BSRP(lr=%.4f cRank=0) ATLAS-SPARROW(lr=%.4f cRank=%u modeRankCap=%u autoGate=%u) ATLAS-HELM(lr=%.4f) ATLAS-ASTER(lr=%.4f)\n",
	       cfg.latent.adamLR, cfg.latent.atlasLR, cfg.latent.atlasLR, cfg.atlasComplementRank,
	       cfg.atlasSparrowModeRank, cfg.atlasSparrowAutoModeGate, cfg.latent.atlasLR, cfg.latent.atlasLR);
	printf("ATLAS: rank=%u tSub=%u kappaMax=%.3f sparrow(modeRankCap=%u autoGate=%u memoryScale=%.3f edge=%.3f secondEdge=%.3f secondFrac=%.3f poleMax=%.3f) helm(modeRank=%u hiddenStack=%u memoryScale=%.3f edge=%.3f poleMax=%.3f) aster(stateRank=%u hiddenStack=%u memoryScale=%.3f edge=%.3f poleMax=%.3f)\n",
	       cfg.atlasRank, cfg.atlasTSub, cfg.atlasKappaMax,
	       cfg.atlasSparrowModeRank,
	       cfg.atlasSparrowAutoModeGate,
	       cfg.atlasSparrowMemoryScale, cfg.atlasSparrowEdgeThreshold,
	       cfg.atlasSparrowSecondEdgeThreshold, cfg.atlasSparrowSecondEdgeFraction,
	       cfg.atlasSparrowPoleMax,
	       cfg.atlasHelmModeRank, cfg.atlasHelmHiddenStackDepth,
	       cfg.atlasHelmMemoryScale, cfg.atlasHelmEdgeThreshold, cfg.atlasHelmPoleMax,
	       cfg.atlasAsterStateRank, cfg.atlasAsterHiddenStackDepth,
	       cfg.atlasAsterMemoryScale, cfg.atlasAsterEdgeThreshold, cfg.atlasAsterPoleMax);
	printf("\n");
	printf("%-15s  %7s          %10s            %9s           %9s           %9s           %9s         %s\n",
	       "Optimizer", "Train(s)", "Windows/s", "TrainMSE", "TrainR2%", "TestMSE", "TestR2%", "Status");

	const VariantKind variants[] = { VARIANT_ADAMW, VARIANT_ATLAS_BASE, VARIANT_ATLAS_SPARROW, VARIANT_ATLAS_HELM, VARIANT_ATLAS_ASTER };
	const size_t variantCount = sizeof(variants) / sizeof(variants[0]);
	for (size_t v = 0; v < variantCount; ++v)
	{
		std::vector<RunResult> runs;
		runs.reserve(cfg.repeats);
		for (unsigned int rep = 0u; rep < cfg.repeats; ++rep)
			runs.push_back(run_nonlinear_latent_variant(cfg, data, variants[v],
			                                            cfg.seed + static_cast<unsigned int>(4000u + 100u * v) + rep));
		const Summary s = summarize_runs(variant_label(variants[v]), runs);
		print_summary_row(s);
		print_sparrow_usage_row(s);
		print_helm_usage_row(s);
		print_aster_usage_row(s);
	}
	printf("\n");
	return true;
}

static bool run_teacher_case(const BenchConfig& cfg)
{
	DenseRegressionInput data;
	build_teacher_dataset(cfg, data);

	printf("------------------------------------------------------------\n");
	printf("Case: teacher-student\n");
	printf("Description: planted low-rank teacher signal plus nuisance bulk, trained with a student DFF regressor.\n");
	printf("Config: inputDim=%u teacherRank=%u bulkRank=%u train=%u test=%u batch=%u epochs=%u repeats=%u bulkScale=%.3f\n",
	       cfg.teacher.inputDim, cfg.teacher.teacherRank, cfg.teacher.bulkRank,
	       cfg.teacher.trainSamples, cfg.teacher.testSamples, cfg.teacher.batchSize,
	       cfg.teacher.epochs, cfg.repeats, cfg.teacher.bulkScale);
	printf("Optimizers: AdamW(lr=%.4f) ATLAS-BSRP(lr=%.4f cRank=0) ATLAS-SPARROW(lr=%.4f cRank=%u modeRankCap=%u autoGate=%u)\n",
	       cfg.teacher.adamLR, cfg.teacher.atlasLR, cfg.teacher.atlasLR, cfg.atlasComplementRank,
	       cfg.atlasSparrowModeRank, cfg.atlasSparrowAutoModeGate);
	printf("ATLAS: rank=%u tSub=%u kappaMax=%.3f sparrow(modeRankCap=%u autoGate=%u memoryScale=%.3f edge=%.3f secondEdge=%.3f secondFrac=%.3f poleMax=%.3f)\n",
	       cfg.atlasRank, cfg.atlasTSub, cfg.atlasKappaMax,
	       cfg.atlasSparrowModeRank,
	       cfg.atlasSparrowAutoModeGate,
	       cfg.atlasSparrowMemoryScale, cfg.atlasSparrowEdgeThreshold,
	       cfg.atlasSparrowSecondEdgeThreshold, cfg.atlasSparrowSecondEdgeFraction,
	       cfg.atlasSparrowPoleMax);
	printf("\n");
	printf("%-15s  %7s          %10s            %9s           %9s           %9s           %9s         %s\n",
	       "Optimizer", "Train(s)", "Samples/s", "TrainMSE", "TrainR2%", "TestMSE", "TestR2%", "Status");

	const VariantKind variants[] = { VARIANT_ADAMW, VARIANT_ATLAS_BASE, VARIANT_ATLAS_SPARROW };
	const size_t variantCount = sizeof(variants) / sizeof(variants[0]);
	for (size_t v = 0; v < variantCount; ++v)
	{
		std::vector<RunResult> runs;
		runs.reserve(cfg.repeats);
		for (unsigned int rep = 0u; rep < cfg.repeats; ++rep)
			runs.push_back(run_teacher_variant(cfg, data, variants[v], cfg.seed + static_cast<unsigned int>(1000u + 100u * v) + rep));
		const Summary s = summarize_runs(variant_label(variants[v]), runs);
		print_summary_row(s);
		print_sparrow_usage_row(s);
	}
	printf("\n");
	return true;
}

static bool run_teacher_canonical_case(const BenchConfig& cfg)
{
	struct CanonicalCase
	{
		const char* name;
		const char* note;
		unsigned int teacherRank;
		unsigned int bulkRank;
		float bulkScale;
		unsigned int cRank;
	};

	const CanonicalCase cases[] = {
		{ "signal-win", "best positive planted regime seen in the repeated quick sweep", 8u, 24u, 0.00f, 4u },
		{ "default-anchor", "default planted regime used as the neutral anchor for future regressions", 4u, 12u, 0.20f, 4u },
		{ "failure-band", "known negative SPARROW band from the repeated quick sweep", 8u, 12u, 0.20f, 4u }
	};
	const size_t caseCount = sizeof(cases) / sizeof(cases[0]);

	printf("------------------------------------------------------------\n");
	printf("Case: teacher-canonical\n");
	printf("Description: fixed planted teacher-student regression cases for SPARROW regression testing.\n");
	printf("Repeats=%u  ATLAS rank=%u tSub=%u kappaMax=%.3f  sparrow(memoryScale=%.3f edge=%.3f secondEdge=%.3f secondFrac=%.3f poleMax=%.3f)\n",
	       cfg.repeats, cfg.atlasRank, cfg.atlasTSub, cfg.atlasKappaMax,
	       cfg.atlasSparrowMemoryScale, cfg.atlasSparrowEdgeThreshold,
	       cfg.atlasSparrowSecondEdgeThreshold, cfg.atlasSparrowSecondEdgeFraction,
	       cfg.atlasSparrowPoleMax);
	printf("%-14s  %6s  %6s  %7s  %5s  %10s  %10s  %10s  %9s  %9s  %7s\n",
	       "Case", "tRank", "bRank", "bScale", "cRk",
	       "BaseMSE", "SpR1MSE", "SpAuto", "dR1", "dAuto", "Winner");

	for (size_t idx = 0; idx < caseCount; ++idx)
	{
		BenchConfig caseCfg = cfg;
		caseCfg.teacher.teacherRank = cases[idx].teacherRank;
		caseCfg.teacher.bulkRank = cases[idx].bulkRank;
		caseCfg.teacher.bulkScale = cases[idx].bulkScale;
		caseCfg.atlasComplementRank = cases[idx].cRank;

		DenseRegressionInput data;
		build_teacher_dataset(caseCfg, data);

		std::vector<RunResult> baseRuns;
		std::vector<RunResult> rank1Runs;
		std::vector<RunResult> autoRuns;
		baseRuns.reserve(caseCfg.repeats);
		rank1Runs.reserve(caseCfg.repeats);
		autoRuns.reserve(caseCfg.repeats);
		for (unsigned int rep = 0u; rep < caseCfg.repeats; ++rep)
		{
			baseRuns.push_back(run_teacher_variant(caseCfg, data, VARIANT_ATLAS_BASE,
			                                      caseCfg.seed + static_cast<unsigned int>(9000u + 101u * idx + rep)));

			BenchConfig rank1Cfg = caseCfg;
			rank1Cfg.atlasSparrowModeRank = 1u;
			rank1Runs.push_back(run_teacher_variant(rank1Cfg, data, VARIANT_ATLAS_SPARROW,
			                                       rank1Cfg.seed + static_cast<unsigned int>(10000u + 131u * idx + rep)));

			BenchConfig autoCfg = caseCfg;
			autoCfg.atlasSparrowModeRank = 2u;
			autoCfg.atlasSparrowAutoModeGate = 1u;
			autoRuns.push_back(run_teacher_variant(autoCfg, data, VARIANT_ATLAS_SPARROW,
			                                      autoCfg.seed + static_cast<unsigned int>(11000u + 151u * idx + rep)));
		}

		const Summary baseSummary = summarize_runs("ATLAS-BSRP", baseRuns);
		const Summary rank1Summary = summarize_runs("ATLAS-SPARROW-r1", rank1Runs);
		const Summary autoSummary = summarize_runs("ATLAS-SPARROW-auto", autoRuns);

		double deltaR1 = 0.0;
		double deltaAuto = 0.0;
		if (baseSummary.ok && rank1Summary.ok)
			deltaR1 = rank1Summary.testLoss.mean - baseSummary.testLoss.mean;
		if (baseSummary.ok && autoSummary.ok)
			deltaAuto = autoSummary.testLoss.mean - baseSummary.testLoss.mean;

		const char* winner = "base";
		const double baseLoss = baseSummary.ok ? baseSummary.testLoss.mean : 1e30;
		const double rank1Loss = rank1Summary.ok ? rank1Summary.testLoss.mean : 1e30;
		const double autoLoss = autoSummary.ok ? autoSummary.testLoss.mean : 1e30;
		if (autoLoss <= baseLoss && autoLoss <= rank1Loss)
			winner = "sparrow-auto";
		else if (rank1Loss <= baseLoss)
			winner = "sparrow-r1";

		printf("%-14s  %6u  %6u  %7.2f  %5u  %10.5f  %10.5f  %10.5f  %+9.5f  %+9.5f  %s\n",
		       cases[idx].name,
		       cases[idx].teacherRank,
		       cases[idx].bulkRank,
		       cases[idx].bulkScale,
		       cases[idx].cRank,
		       baseSummary.testLoss.mean,
		       rank1Summary.testLoss.mean,
		       autoSummary.testLoss.mean,
		       deltaR1,
		       deltaAuto,
		       winner);
		printf("  note: %s\n", cases[idx].note);
	}

	printf("\n");
	return true;
}

static bool run_teacher_sweep_case(const BenchConfig& cfg)
{
	std::vector<unsigned int> teacherRanks;
	std::vector<unsigned int> bulkRanks;
	std::vector<float> bulkScales;
	std::vector<unsigned int> cRanks;
	fill_teacher_sweep_axes(cfg.teacherSweepProfile, teacherRanks, bulkRanks, bulkScales, cRanks);

	printf("------------------------------------------------------------\n");
	printf("Case: teacher-sweep\n");
	printf("Description: phase sweep over planted teacher-student signal/bulk structure, comparing isotropic ATLAS against SPARROW.\n");
	printf("Sweep profile: %s  repeats=%u  limit=%u (0 means no limit)\n",
	       (cfg.teacherSweepProfile == SWEEP_FULL) ? "full" : "quick",
	       cfg.repeats, cfg.teacherSweepLimit);
	printf("ATLAS defaults: rank=%u tSub=%u kappaMax=%.3f sparrow(modeRankCap=%u autoGate=%u memoryScale=%.3f edge=%.3f secondEdge=%.3f secondFrac=%.3f poleMax=%.3f)\n",
	       cfg.atlasRank, cfg.atlasTSub, cfg.atlasKappaMax,
	       cfg.atlasSparrowModeRank,
	       cfg.atlasSparrowAutoModeGate,
	       cfg.atlasSparrowMemoryScale, cfg.atlasSparrowEdgeThreshold,
	       cfg.atlasSparrowSecondEdgeThreshold, cfg.atlasSparrowSecondEdgeFraction,
	       cfg.atlasSparrowPoleMax);
	printf("Note: AdamW is omitted here; the sweep is isolating when SPARROW beats isotropic ATLAS.\n\n");
	printf("%6s  %6s  %7s  %5s  %10s  %10s  %9s  %8s  %8s  %7s  %7s  %s\n",
	       "tRank", "bRank", "bScale", "cRk", "BaseMSE", "SpMSE", "Delta", "BaseR2", "SpR2", "Bsec", "Ssec", "Status");

	unsigned int baseConfigs = 0u;
	unsigned int sparrowWins = 0u;
	double bestDelta = 0.0;
	bool haveBest = false;
	unsigned int bestTeacherRank = 0u;
	unsigned int bestBulkRank = 0u;
	float bestBulkScale = 0.0f;
	unsigned int bestCRank = 0u;
	bool done = false;
	for (size_t ti = 0; ti < teacherRanks.size() && !done; ++ti)
	{
		for (size_t bi = 0; bi < bulkRanks.size() && !done; ++bi)
		{
			for (size_t si = 0; si < bulkScales.size() && !done; ++si)
			{
				if (cfg.teacherSweepLimit > 0u && baseConfigs >= cfg.teacherSweepLimit)
				{
					done = true;
					break;
				}

				BenchConfig caseCfg = cfg;
				caseCfg.teacher.teacherRank = teacherRanks[ti];
				caseCfg.teacher.bulkRank = bulkRanks[bi];
				caseCfg.teacher.bulkScale = bulkScales[si];

				DenseRegressionInput data;
				build_teacher_dataset(caseCfg, data);

				std::vector<RunResult> baseRuns;
				baseRuns.reserve(caseCfg.repeats);
				for (unsigned int rep = 0u; rep < caseCfg.repeats; ++rep)
					baseRuns.push_back(run_teacher_variant(caseCfg, data, VARIANT_ATLAS_BASE,
					                                      caseCfg.seed + static_cast<unsigned int>(5000u + 31u * baseConfigs + rep)));
				const Summary baseSummary = summarize_runs("ATLAS-BSRP", baseRuns);
				++baseConfigs;

				for (size_t ci = 0; ci < cRanks.size(); ++ci)
				{
					BenchConfig sparrowCfg = caseCfg;
					sparrowCfg.atlasComplementRank = cRanks[ci];

					std::vector<RunResult> spRuns;
					spRuns.reserve(sparrowCfg.repeats);
					for (unsigned int rep = 0u; rep < sparrowCfg.repeats; ++rep)
						spRuns.push_back(run_teacher_variant(sparrowCfg, data, VARIANT_ATLAS_SPARROW,
						                                     sparrowCfg.seed + static_cast<unsigned int>(7000u + 53u * baseConfigs + 7u * ci + rep)));
					const Summary spSummary = summarize_runs("ATLAS-SPARROW", spRuns);

					double delta = 0.0;
					if (baseSummary.ok && spSummary.ok)
					{
						delta = spSummary.testLoss.mean - baseSummary.testLoss.mean;
						if (delta < 0.0)
							++sparrowWins;
						if (!haveBest || delta < bestDelta)
						{
							haveBest = true;
							bestDelta = delta;
							bestTeacherRank = caseCfg.teacher.teacherRank;
							bestBulkRank = caseCfg.teacher.bulkRank;
							bestBulkScale = caseCfg.teacher.bulkScale;
							bestCRank = sparrowCfg.atlasComplementRank;
						}
					}

					printf("%6u  %6u  %7.2f  %5u  %10.5f  %10.5f  %+9.5f  %8.3f  %8.3f  %7.2f  %7.2f  %s\n",
					       caseCfg.teacher.teacherRank,
					       caseCfg.teacher.bulkRank,
					       caseCfg.teacher.bulkScale,
					       sparrowCfg.atlasComplementRank,
					       baseSummary.testLoss.mean,
					       spSummary.testLoss.mean,
					       delta,
					       baseSummary.testMetric.mean,
					       spSummary.testMetric.mean,
					       baseSummary.trainSec.mean,
					       spSummary.trainSec.mean,
					       (baseSummary.ok && spSummary.ok) ? "ok" : "error");
				}
			}
		}
	}

	printf("\n");
	printf("Teacher sweep summary: baseConfigs=%u sparrowWins=%u\n", baseConfigs, sparrowWins);
	if (haveBest)
	{
		printf("Best delta(TestMSE): %+0.5f at teacherRank=%u bulkRank=%u bulkScale=%.2f cRank=%u\n",
		       bestDelta, bestTeacherRank, bestBulkRank, bestBulkScale, bestCRank);
	}
	printf("\n");
	return true;
}

} // namespace

void ATLASAltBenchmark(int argc, char* argv[])
{
	printf("============================================================\n");
	printf("ATLAS Alternate Benchmark Harness\n");
	printf("============================================================\n");

	BenchConfig cfg;
	std::string err;
	if (!parse_args(argc, argv, cfg, err))
	{
		if (!err.empty())
			printf("Argument error: %s\n\n", err.c_str());
		print_usage();
		printf("============================================================\n");
		return;
	}

	if (cfg.mode == MODE_ALL || cfg.mode == MODE_TOKEN_LM)
		run_token_case(cfg);
	if (cfg.mode == MODE_ALL || cfg.mode == MODE_TEACHER_STUDENT)
		run_teacher_case(cfg);
	if (cfg.mode == MODE_ALL || cfg.mode == MODE_LATENT_FORECAST)
		run_latent_case(cfg);
	if (cfg.mode == MODE_ALL || cfg.mode == MODE_NONLINEAR_FORECAST)
		run_nonlinear_case(cfg);
	if (cfg.mode == MODE_TEACHER_SWEEP)
		run_teacher_sweep_case(cfg);
	if (cfg.mode == MODE_TEACHER_CANONICAL)
		run_teacher_canonical_case(cfg);

	printf("============================================================\n");
}
