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
#include <iomanip>
#include <map>
#include <sstream>
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

static const char* bimap_scope_label(unsigned int scope)
{
	switch (scope)
	{
	case glades::ATLASConfig::BIMAP_SCOPE_ALL: return "all";
	case glades::ATLASConfig::BIMAP_SCOPE_HEAD_ONLY: return "head";
	case glades::ATLASConfig::BIMAP_SCOPE_LATE_ONLY: return "late";
	case glades::ATLASConfig::BIMAP_SCOPE_LATE_HEAD: return "late-head";
	default: return "unknown";
	}
}

static bool parse_bimap_scope_arg(const char* text, unsigned int& outScope)
{
	if (!text)
		return false;
	if (streq(text, "all") || streq(text, "0"))
	{
		outScope = glades::ATLASConfig::BIMAP_SCOPE_ALL;
		return true;
	}
	if (streq(text, "head") || streq(text, "head-only") || streq(text, "1"))
	{
		outScope = glades::ATLASConfig::BIMAP_SCOPE_HEAD_ONLY;
		return true;
	}
	if (streq(text, "late") || streq(text, "late-only") || streq(text, "2"))
	{
		outScope = glades::ATLASConfig::BIMAP_SCOPE_LATE_ONLY;
		return true;
	}
	if (streq(text, "late-head") || streq(text, "head-late") || streq(text, "3"))
	{
		outScope = glades::ATLASConfig::BIMAP_SCOPE_LATE_HEAD;
		return true;
	}
	return false;
}

static const char* echo_scope_label(unsigned int scope)
{
	switch (scope)
	{
	case glades::ATLASConfig::ECHO_SCOPE_ALL: return "all";
	case glades::ATLASConfig::ECHO_SCOPE_LARGE_ONLY: return "large-only";
	case glades::ATLASConfig::ECHO_SCOPE_LATE_HEAD: return "late-head";
	case glades::ATLASConfig::ECHO_SCOPE_LATE_HEAD_LARGE: return "late-head-large";
	default: return "unknown";
	}
}

static bool parse_echo_scope_arg(const char* text, unsigned int& outScope)
{
	if (!text)
		return false;
	if (streq(text, "all") || streq(text, "0"))
	{
		outScope = glades::ATLASConfig::ECHO_SCOPE_ALL;
		return true;
	}
	if (streq(text, "large") || streq(text, "large-only") || streq(text, "1"))
	{
		outScope = glades::ATLASConfig::ECHO_SCOPE_LARGE_ONLY;
		return true;
	}
	if (streq(text, "late-head") || streq(text, "head-late") || streq(text, "2"))
	{
		outScope = glades::ATLASConfig::ECHO_SCOPE_LATE_HEAD;
		return true;
	}
	if (streq(text, "late-head-large") || streq(text, "head-late-large") || streq(text, "3"))
	{
		outScope = glades::ATLASConfig::ECHO_SCOPE_LATE_HEAD_LARGE;
		return true;
	}
	return false;
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
	double atlasAsterBoundaryMsSum;
	double atlasAsterSetupMsSum;
	double atlasAsterTransportMsSum;
	double atlasAsterTransferFitMsSum;
	double atlasAsterStateFitMsSum;
	double atlasAsterInnovationFitMsSum;
	double atlasAsterApplyMsSum;
	double atlasAegisLambdaSpatialSum;
	double atlasAegisLambdaPredictiveSum;
	double atlasAegisLambdaOutputSum;
	double atlasAegisPredictivePredictedSum;
	double atlasAegisPredictiveRealizedSum;
	double atlasAegisOutputPredictedSum;
	double atlasAegisOutputRealizedSum;
	double atlasAegisPredictiveErrorSum;
	double atlasAegisOutputErrorSum;
	double atlasAegisChannelDisagreementSum;
	double atlasCitadelAnchorSum;
	double atlasCitadelHardRegimeMassSum;
	double atlasCitadelSparrowTrustSum;
	double atlasRampartTauSum;
	double atlasRampartBudgetSum;
	double atlasRampartCovarianceSum;
	double atlasRampartSparrowTrustSum;
	double atlasMeritTauSum;
	double atlasMeritBudgetSum;
	double atlasMeritCovarianceSum;
	double atlasMeritSparrowTrustSum;
	double atlasMeritGeometryTrustSum;
	double atlasStrataNullModeSum;
	double atlasStrataPredictiveModeSum;
	double atlasStrataOutputModeSum;
	double atlasStrataCoupledModeSum;
	double atlasStrataBudgetSum;
	double atlasStrataNullBenefitSum;
	double atlasStrataPredictiveBenefitSum;
	double atlasStrataOutputBenefitSum;
	double atlasStrataCoupledBenefitSum;
	double atlasStrataSelectedExcessSum;
	double atlasStrataSwitchRateSum;
	unsigned int atlasHelmMode2Epochs;
	unsigned int atlasSparrowEpochs;
	unsigned int atlasHelmEpochs;
	unsigned int atlasAsterMode2Epochs;
	unsigned int atlasAsterEpochs;
	unsigned int atlasAegisEpochs;
	unsigned int atlasCitadelEpochs;
	unsigned int atlasRampartEpochs;
	unsigned int atlasMeritEpochs;
	unsigned int atlasStrataEpochs;

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
	      atlasAsterBoundaryMsSum(0.0),
	      atlasAsterSetupMsSum(0.0),
	      atlasAsterTransportMsSum(0.0),
	      atlasAsterTransferFitMsSum(0.0),
	      atlasAsterStateFitMsSum(0.0),
	      atlasAsterInnovationFitMsSum(0.0),
	      atlasAsterApplyMsSum(0.0),
	      atlasAegisLambdaSpatialSum(0.0),
	      atlasAegisLambdaPredictiveSum(0.0),
	      atlasAegisLambdaOutputSum(0.0),
	      atlasAegisPredictivePredictedSum(0.0),
	      atlasAegisPredictiveRealizedSum(0.0),
	      atlasAegisOutputPredictedSum(0.0),
	      atlasAegisOutputRealizedSum(0.0),
	      atlasAegisPredictiveErrorSum(0.0),
	      atlasAegisOutputErrorSum(0.0),
	      atlasAegisChannelDisagreementSum(0.0),
	      atlasCitadelAnchorSum(0.0),
	      atlasCitadelHardRegimeMassSum(0.0),
	      atlasCitadelSparrowTrustSum(0.0),
	      atlasRampartTauSum(0.0),
	      atlasRampartBudgetSum(0.0),
	      atlasRampartCovarianceSum(0.0),
	      atlasRampartSparrowTrustSum(0.0),
	      atlasMeritTauSum(0.0),
	      atlasMeritBudgetSum(0.0),
	      atlasMeritCovarianceSum(0.0),
	      atlasMeritSparrowTrustSum(0.0),
	      atlasMeritGeometryTrustSum(0.0),
	      atlasStrataNullModeSum(0.0),
	      atlasStrataPredictiveModeSum(0.0),
	      atlasStrataOutputModeSum(0.0),
	      atlasStrataCoupledModeSum(0.0),
	      atlasStrataBudgetSum(0.0),
	      atlasStrataNullBenefitSum(0.0),
	      atlasStrataPredictiveBenefitSum(0.0),
	      atlasStrataOutputBenefitSum(0.0),
	      atlasStrataCoupledBenefitSum(0.0),
	      atlasStrataSelectedExcessSum(0.0),
	      atlasStrataSwitchRateSum(0.0),
	      atlasHelmMode2Epochs(0u),
	      atlasSparrowEpochs(0u),
	      atlasHelmEpochs(0u),
	      atlasAsterMode2Epochs(0u),
	      atlasAsterEpochs(0u),
	      atlasAegisEpochs(0u),
	      atlasCitadelEpochs(0u),
	      atlasRampartEpochs(0u),
	      atlasMeritEpochs(0u),
	      atlasStrataEpochs(0u)
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
			if (diag.sparrowMatrices > 0u || diag.helmMatrices > 0u || diag.asterMatrices > 0u
			    || diag.aegisMatrices > 0u || diag.meritMatrices > 0u || diag.strataMatrices > 0u)
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
				atlasAsterBoundaryMsSum += diag.asterMeanBoundaryMs;
				atlasAsterSetupMsSum += diag.asterMeanSetupMs;
				atlasAsterTransportMsSum += diag.asterMeanTransportMs;
				atlasAsterTransferFitMsSum += diag.asterMeanTransferFitMs;
				atlasAsterStateFitMsSum += diag.asterMeanStateFitMs;
				atlasAsterInnovationFitMsSum += diag.asterMeanInnovationFitMs;
				atlasAsterApplyMsSum += diag.asterMeanApplyMs;
				if (diag.asterMode2Fraction > 0.0)
					atlasAsterMode2Epochs += 1u;
				atlasAsterEpochs += 1u;
			}
			if (diag.aegisMatrices > 0u)
			{
				atlasAegisLambdaSpatialSum += diag.aegisMeanLambdaSpatial;
				atlasAegisLambdaPredictiveSum += diag.aegisMeanLambdaPredictive;
				atlasAegisLambdaOutputSum += diag.aegisMeanLambdaOutput;
				atlasAegisPredictivePredictedSum += diag.aegisMeanPredictivePredicted;
				atlasAegisPredictiveRealizedSum += diag.aegisMeanPredictiveRealized;
				atlasAegisOutputPredictedSum += diag.aegisMeanOutputPredicted;
				atlasAegisOutputRealizedSum += diag.aegisMeanOutputRealized;
				atlasAegisPredictiveErrorSum += diag.aegisMeanPredictiveError;
				atlasAegisOutputErrorSum += diag.aegisMeanOutputError;
				atlasAegisChannelDisagreementSum += diag.aegisMeanChannelDisagreement;
				atlasAegisEpochs += 1u;
			}
			if (diag.citadelMatrices > 0u)
			{
				atlasCitadelAnchorSum += diag.citadelMeanAnchor;
				atlasCitadelHardRegimeMassSum += diag.citadelMeanHardRegimeMass;
				atlasCitadelSparrowTrustSum += diag.citadelMeanSparrowTrust;
				atlasCitadelEpochs += 1u;
			}
			if (diag.rampartMatrices > 0u)
			{
				atlasRampartTauSum += diag.rampartMeanTau;
				atlasRampartBudgetSum += diag.rampartMeanBudget;
				atlasRampartCovarianceSum += diag.rampartMeanCovariance;
				atlasRampartSparrowTrustSum += diag.rampartMeanSparrowTrust;
				atlasRampartEpochs += 1u;
			}
			if (diag.meritMatrices > 0u)
			{
				atlasMeritTauSum += diag.meritMeanTau;
				atlasMeritBudgetSum += diag.meritMeanBudget;
				atlasMeritCovarianceSum += diag.meritMeanCovariance;
				atlasMeritSparrowTrustSum += diag.meritMeanSparrowTrust;
				atlasMeritGeometryTrustSum += diag.meritMeanGeometryTrust;
				atlasMeritEpochs += 1u;
			}
			if (diag.strataMatrices > 0u)
			{
				atlasStrataNullModeSum += diag.strataMeanNullMode;
				atlasStrataPredictiveModeSum += diag.strataMeanPredictiveMode;
				atlasStrataOutputModeSum += diag.strataMeanOutputMode;
				atlasStrataCoupledModeSum += diag.strataMeanCoupledMode;
				atlasStrataBudgetSum += diag.strataMeanBudget;
				atlasStrataNullBenefitSum += diag.strataMeanNullBenefit;
				atlasStrataPredictiveBenefitSum += diag.strataMeanPredictiveBenefit;
				atlasStrataOutputBenefitSum += diag.strataMeanOutputBenefit;
				atlasStrataCoupledBenefitSum += diag.strataMeanCoupledBenefit;
				atlasStrataSelectedExcessSum += diag.strataMeanSelectedExcess;
				atlasStrataSwitchRateSum += diag.strataMeanSwitchRate;
				atlasStrataEpochs += 1u;
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
	MODE_NONLINEAR_FORECAST = 6,
	MODE_TOKEN_LM_LARGE = 7,
	MODE_TOKEN_LM_CONTEXT = 8,
	MODE_TOKEN_LM_CONTEXT_LARGE = 9,
	MODE_TOKEN_LM_DOCUMENT = 10,
	MODE_TOKEN_LM_CORPUS = 11,
	MODE_TOKEN_LM_CORPUS_LARGE = 12
};

enum VariantKind
{
	VARIANT_ADAMW = 0,
	VARIANT_ADAMW_GROUP = 1,
	VARIANT_ATLAS_BASE = 2,
	VARIANT_ATLAS_SPARROW = 3,
	VARIANT_ATLAS_HELM = 4,
	VARIANT_ATLAS_ASTER = 5,
	VARIANT_ATLAS_AEGIS = 6,
	VARIANT_ATLAS_CITADEL = 7,
	VARIANT_ATLAS_RAMPART = 8,
	VARIANT_ATLAS_MERIT = 9,
	VARIANT_ATLAS_STRATA = 10,
	VARIANT_ATLAS_AURORA = 11,
	VARIANT_ATLAS_SEAM = 12,
	VARIANT_ATLAS_QUASAR = 13,
	VARIANT_ATLAS_GEODE = 14,
	VARIANT_ATLAS_ECHO = 15,
	VARIANT_ATLAS_BIMAP = 16,
	VARIANT_ATLAS_PACT = 17,
	VARIANT_ATLAS_RACER = 18,
	VARIANT_ATLAS_KRON = 19,
	VARIANT_ATLAS_MUON = 20,
	VARIANT_ATLAS_MATRA = 21
};

enum VariantSelection
{
	VARIANT_SELECTION_ALL = 0,
	VARIANT_SELECTION_ADAMW = 1,
	VARIANT_SELECTION_ADAMW_GROUP = 2,
	VARIANT_SELECTION_ATLAS_BASE = 3,
	VARIANT_SELECTION_ATLAS_SPARROW = 4,
	VARIANT_SELECTION_ATLAS_HELM = 5,
	VARIANT_SELECTION_ATLAS_ASTER = 6,
	VARIANT_SELECTION_ATLAS_AEGIS = 7,
	VARIANT_SELECTION_ATLAS_CITADEL = 8,
	VARIANT_SELECTION_ATLAS_RAMPART = 9,
	VARIANT_SELECTION_ATLAS_MERIT = 10,
	VARIANT_SELECTION_ATLAS_STRATA = 11,
	VARIANT_SELECTION_ATLAS_AURORA = 12,
	VARIANT_SELECTION_ATLAS_SEAM = 13,
	VARIANT_SELECTION_ATLAS_QUASAR = 14,
	VARIANT_SELECTION_ATLAS_GEODE = 15,
	VARIANT_SELECTION_ATLAS_ECHO = 16,
	VARIANT_SELECTION_ATLAS_BIMAP = 17,
	VARIANT_SELECTION_ATLAS_PACT = 18,
	VARIANT_SELECTION_ATLAS_RACER = 19,
	VARIANT_SELECTION_ATLAS_KRON = 20,
	VARIANT_SELECTION_ATLAS_MUON = 21,
	VARIANT_SELECTION_ATLAS_MATRA = 22
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
	VariantSelection variantSelection;
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
	unsigned int atlasKappaEnabled;
	unsigned int atlasKappaHeads;
	unsigned int atlasKappaLagBuckets;
	unsigned int atlasKappaRank;
	unsigned int atlasAuroraAdamwBackbone;
	float atlasAuroraHeadGain;
	float atlasAuroraBodyTrustScale;
	float atlasGeodeGeometryScale;
	float atlasGeodePredictiveScale;
	float atlasEchoGeometryScale;
	float atlasEchoGeometryScaleFinal;
	unsigned int atlasEchoGeometryDecaySteps;
	unsigned int atlasEchoMetricCadence;
	float atlasEchoTrustScale;
	float atlasEchoPredictiveScale;
	float atlasEchoStructuralScale;
	unsigned int atlasEchoStructuralGroups;
	unsigned int atlasEchoScope;
	unsigned int atlasBiMAPLowRank;
	unsigned int atlasBiMAPScope;
	float atlasBiMAPGeometryScale;
	float atlasBiMAPPredictiveScale;
	unsigned int atlasBiMAPFactorCadence;
	unsigned int atlasPACTLowRank;
	float atlasPACTGeometryScale;
	float atlasPACTPredictiveScale;
	unsigned int atlasPACTFactorCadence;
	float atlasPACTCostScale;
	float atlasPACTPromoteThreshold;
	float atlasPACTDemoteThreshold;
	float atlasRACERGeometryScale;
	float atlasRACERPredictiveScale;
	unsigned int atlasRACERFactorCadence;
	float atlasRACERRiskScale;
	float atlasRACERCostScale;
	float atlasRACERPromoteThreshold;
	float atlasRACERDemoteThreshold;
	float atlasKronGeometryScale;
	float atlasKronPredictiveScale;
	unsigned int atlasKronFactorCadence;
	float atlasKronDamping;
	float atlasMuonGeometryScale;
	float atlasMuonPredictiveScale;
	float atlasMuonMaxAspect;
	unsigned int atlasMuonMinDim;
	float atlasMuonDamping;
	float atlasMatraGeometryScale;
	float atlasMatraOrthogonalScale;
	float atlasMatraPredictiveScale;
	float atlasMatraTrustRadius;
	unsigned int atlasMatraMetricCadence;
	float atlasMatraMaxAspect;
	unsigned int atlasMatraMinDim;
	float atlasMatraDamping;
	unsigned int gpuEnable;
	int gpuDeviceId;
	TokenConfig token;
	TeacherConfig teacher;
	LatentConfig latent;
	SweepProfile teacherSweepProfile;
	unsigned int teacherSweepLimit;

	BenchConfig()
	    : mode(MODE_ALL),
	      variantSelection(VARIANT_SELECTION_ALL),
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
	      atlasKappaEnabled(0u),
	      atlasKappaHeads(1u),
	      atlasKappaLagBuckets(4u),
	      atlasKappaRank(2u),
	      atlasAuroraAdamwBackbone(1u),
	      atlasAuroraHeadGain(3.0f),
	      atlasAuroraBodyTrustScale(0.60f),
	      atlasGeodeGeometryScale(1.0f),
	      atlasGeodePredictiveScale(0.25f),
	      atlasEchoGeometryScale(1.0f),
	      atlasEchoGeometryScaleFinal(1.0f),
	      atlasEchoGeometryDecaySteps(0u),
	      atlasEchoMetricCadence(1u),
	      atlasEchoTrustScale(0.0f),
	      atlasEchoPredictiveScale(0.0f),
	      atlasEchoStructuralScale(0.0f),
	      atlasEchoStructuralGroups(1u),
	      atlasEchoScope(glades::ATLASConfig::ECHO_SCOPE_ALL),
	      atlasBiMAPLowRank(1u),
	      atlasBiMAPScope(glades::ATLASConfig::BIMAP_SCOPE_ALL),
	      atlasBiMAPGeometryScale(1.0f),
	      atlasBiMAPPredictiveScale(0.15f),
	      atlasBiMAPFactorCadence(8u),
	      atlasPACTLowRank(1u),
	      atlasPACTGeometryScale(1.0f),
	      atlasPACTPredictiveScale(0.10f),
	      atlasPACTFactorCadence(8u),
	      atlasPACTCostScale(0.0010f),
	      atlasPACTPromoteThreshold(0.0f),
	      atlasPACTDemoteThreshold(-0.0005f),
	      atlasRACERGeometryScale(1.0f),
	      atlasRACERPredictiveScale(0.05f),
	      atlasRACERFactorCadence(8u),
	      atlasRACERRiskScale(0.50f),
	      atlasRACERCostScale(0.0010f),
	      atlasRACERPromoteThreshold(0.0f),
	      atlasRACERDemoteThreshold(-0.0005f),
	      atlasKronGeometryScale(1.0f),
	      atlasKronPredictiveScale(0.05f),
	      atlasKronFactorCadence(8u),
	      atlasKronDamping(0.10f),
	      atlasMuonGeometryScale(1.0f),
	      atlasMuonPredictiveScale(0.05f),
	      atlasMuonMaxAspect(1.50f),
	      atlasMuonMinDim(8u),
	      atlasMuonDamping(0.01f),
	      atlasMatraGeometryScale(1.0f),
	      atlasMatraOrthogonalScale(0.5f),
	      atlasMatraPredictiveScale(0.05f),
	      atlasMatraTrustRadius(0.50f),
	      atlasMatraMetricCadence(1u),
	      atlasMatraMaxAspect(1.50f),
	      atlasMatraMinDim(8u),
	      atlasMatraDamping(0.01f),
	      gpuEnable(0u),
	      gpuDeviceId(0),
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
	std::vector<unsigned int> testTokenStream;
	std::vector<glades::DataInput::SequenceSpan> testSpans;
	std::vector<unsigned char> testRoleBuckets;
	std::vector<unsigned char> testRecallDistanceBuckets;
	std::vector<unsigned char> testRecallSubtypeBuckets;

	TokenDataset()
	    : di(),
	      padTokenId(0u),
	      trainTokensPerEpoch(0ULL),
	      testTokens(0ULL),
	      testTokenStream(),
	      testSpans(),
	      testRoleBuckets(),
	      testRecallDistanceBuckets(),
	      testRecallSubtypeBuckets()
	{
	}
};

enum ContextFamilyRoleBucket
{
	CTX_ROLE_OTHER = 0,
	CTX_ROLE_TOPIC = 1,
	CTX_ROLE_MARKER = 2,
	CTX_ROLE_QUERY = 3,
	CTX_ROLE_RECALL = 4,
	CTX_ROLE_STATE = 5,
	CTX_ROLE_CONTENT = 6,
	CTX_ROLE_SEPARATOR = 7,
	CTX_ROLE_COUNT = 8
};

enum ContextFamilyRecallDistanceBucket
{
	CTX_RECALL_PREV = 0,
	CTX_RECALL_OLDER = 1,
	CTX_RECALL_DISTANCE_COUNT = 2
};

enum ContextFamilyRecallSubtypeBucket
{
	CTX_SUB_SUMMARY = 0,
	CTX_SUB_ANCHOR = 1,
	CTX_SUB_OBJECT = 2,
	CTX_SUB_PLACE = 3,
	CTX_SUB_YEAR = 4,
	CTX_SUB_SPEAKER = 5,
	CTX_SUBTYPE_COUNT = 6
};

static const unsigned char kContextRecallDistanceNone = 255u;
static const unsigned char kContextRecallSubtypeNone = 255u;

static const char* context_family_role_label(unsigned int idx)
{
	switch (idx)
	{
	case CTX_ROLE_OTHER: return "other";
	case CTX_ROLE_TOPIC: return "topic";
	case CTX_ROLE_MARKER: return "marker";
	case CTX_ROLE_QUERY: return "query";
	case CTX_ROLE_RECALL: return "recall";
	case CTX_ROLE_STATE: return "state";
	case CTX_ROLE_CONTENT: return "content";
	case CTX_ROLE_SEPARATOR: return "sep";
	default: return "unknown";
	}
}

static const char* context_family_recall_distance_label(unsigned int idx)
{
	switch (idx)
	{
	case CTX_RECALL_PREV: return "prev";
	case CTX_RECALL_OLDER: return "older";
	default: return "unknown";
	}
}

static const char* context_family_recall_subtype_label(unsigned int idx)
{
	switch (idx)
	{
	case CTX_SUB_SUMMARY: return "summary";
	case CTX_SUB_ANCHOR: return "anchor";
	case CTX_SUB_OBJECT: return "object";
	case CTX_SUB_PLACE: return "place";
	case CTX_SUB_YEAR: return "year";
	case CTX_SUB_SPEAKER: return "speaker";
	default: return "unknown";
	}
}

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
	double asterBoundaryMs;
	double asterSetupMs;
	double asterTransportMs;
	double asterTransferFitMs;
	double asterStateFitMs;
	double asterInnovationFitMs;
	double asterApplyMs;
	bool aegisDiagValid;
	double aegisLambdaSpatial;
	double aegisLambdaPredictive;
	double aegisLambdaOutput;
	double aegisPredictivePredicted;
	double aegisPredictiveRealized;
	double aegisOutputPredicted;
	double aegisOutputRealized;
	double aegisPredictiveError;
	double aegisOutputError;
	double aegisChannelDisagreement;
	bool citadelDiagValid;
	double citadelAnchor;
	double citadelHardRegimeMass;
	double citadelSparrowTrust;
	bool rampartDiagValid;
	double rampartTau;
	double rampartBudget;
	double rampartCovariance;
	double rampartSparrowTrust;
	bool meritDiagValid;
	double meritTau;
	double meritBudget;
	double meritCovariance;
	double meritSparrowTrust;
	double meritGeometryTrust;
	bool strataDiagValid;
	double strataNullMode;
	double strataPredictiveMode;
	double strataOutputMode;
	double strataCoupledMode;
	double strataBudget;
	double strataNullBenefit;
	double strataPredictiveBenefit;
	double strataOutputBenefit;
	double strataCoupledBenefit;
	double strataSelectedExcess;
	double strataSwitchRate;
	bool transformerGapDiagValid;
	double transformerInputUpdateNorm;
	std::vector<double> transformerBlockUpdateNorms;
	double transformerFinalNormUpdateNorm;
	double transformerHeadUpdateNorm;
	double transformerHeadShare;
	double transformerNonHeadShare;
	double transformerApplyMs;
	bool transformerTrainMarginValid;
	double transformerTrainTargetMargin;
	double transformerTrainHardNegativeLogit;
	bool transformerTestMarginValid;
	double transformerTestTargetMargin;
	double transformerTestHardNegativeLogit;
	bool contextFamilyDiagValid;
	std::vector<double> contextRoleNll;
	std::vector<double> contextRoleShare;
	std::vector<double> contextRecallDistanceNll;
	std::vector<double> contextRecallDistanceShare;
	std::vector<double> contextRecallSubtypeNll;
	std::vector<double> contextRecallSubtypeShare;
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
	      asterBoundaryMs(0.0),
	      asterSetupMs(0.0),
	      asterTransportMs(0.0),
	      asterTransferFitMs(0.0),
	      asterStateFitMs(0.0),
	      asterInnovationFitMs(0.0),
	      asterApplyMs(0.0),
	      aegisDiagValid(false),
	      aegisLambdaSpatial(0.0),
	      aegisLambdaPredictive(0.0),
	      aegisLambdaOutput(0.0),
	      aegisPredictivePredicted(0.0),
	      aegisPredictiveRealized(0.0),
	      aegisOutputPredicted(0.0),
	      aegisOutputRealized(0.0),
	      aegisPredictiveError(0.0),
	      aegisOutputError(0.0),
	      aegisChannelDisagreement(0.0),
	      citadelDiagValid(false),
	      citadelAnchor(0.0),
	      citadelHardRegimeMass(0.0),
	      citadelSparrowTrust(0.0),
	      rampartDiagValid(false),
	      rampartTau(0.0),
	      rampartBudget(0.0),
	      rampartCovariance(0.0),
	      rampartSparrowTrust(0.0),
	      meritDiagValid(false),
	      meritTau(0.0),
	      meritBudget(0.0),
	      meritCovariance(0.0),
	      meritSparrowTrust(0.0),
	      meritGeometryTrust(0.0),
	      strataDiagValid(false),
	      strataNullMode(0.0),
	      strataPredictiveMode(0.0),
	      strataOutputMode(0.0),
	      strataCoupledMode(0.0),
	      strataBudget(0.0),
	      strataNullBenefit(0.0),
	      strataPredictiveBenefit(0.0),
	      strataOutputBenefit(0.0),
	      strataCoupledBenefit(0.0),
	      strataSelectedExcess(0.0),
	      strataSwitchRate(0.0),
	      transformerGapDiagValid(false),
	      transformerInputUpdateNorm(0.0),
	      transformerBlockUpdateNorms(),
	      transformerFinalNormUpdateNorm(0.0),
	      transformerHeadUpdateNorm(0.0),
	      transformerHeadShare(0.0),
	      transformerNonHeadShare(0.0),
	      transformerApplyMs(0.0),
	      transformerTrainMarginValid(false),
	      transformerTrainTargetMargin(0.0),
	      transformerTrainHardNegativeLogit(0.0),
	      transformerTestMarginValid(false),
	      transformerTestTargetMargin(0.0),
	      transformerTestHardNegativeLogit(0.0),
	      contextFamilyDiagValid(false),
	      contextRoleNll(),
	      contextRoleShare(),
	      contextRecallDistanceNll(),
	      contextRecallDistanceShare(),
	      contextRecallSubtypeNll(),
	      contextRecallSubtypeShare(),
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
	AggregateStats asterBoundaryMs;
	AggregateStats asterSetupMs;
	AggregateStats asterTransportMs;
	AggregateStats asterTransferFitMs;
	AggregateStats asterStateFitMs;
	AggregateStats asterInnovationFitMs;
	AggregateStats asterApplyMs;
	bool aegisDiagValid;
	AggregateStats aegisLambdaSpatial;
	AggregateStats aegisLambdaPredictive;
	AggregateStats aegisLambdaOutput;
	AggregateStats aegisPredictivePredicted;
	AggregateStats aegisPredictiveRealized;
	AggregateStats aegisOutputPredicted;
	AggregateStats aegisOutputRealized;
	AggregateStats aegisPredictiveError;
	AggregateStats aegisOutputError;
	AggregateStats aegisChannelDisagreement;
	bool citadelDiagValid;
	AggregateStats citadelAnchor;
	AggregateStats citadelHardRegimeMass;
	AggregateStats citadelSparrowTrust;
	bool rampartDiagValid;
	AggregateStats rampartTau;
	AggregateStats rampartBudget;
	AggregateStats rampartCovariance;
	AggregateStats rampartSparrowTrust;
	bool meritDiagValid;
	AggregateStats meritTau;
	AggregateStats meritBudget;
	AggregateStats meritCovariance;
	AggregateStats meritSparrowTrust;
	AggregateStats meritGeometryTrust;
	bool strataDiagValid;
	AggregateStats strataNullMode;
	AggregateStats strataPredictiveMode;
	AggregateStats strataOutputMode;
	AggregateStats strataCoupledMode;
	AggregateStats strataBudget;
	AggregateStats strataNullBenefit;
	AggregateStats strataPredictiveBenefit;
	AggregateStats strataOutputBenefit;
	AggregateStats strataCoupledBenefit;
	AggregateStats strataSelectedExcess;
	AggregateStats strataSwitchRate;
	bool transformerGapDiagValid;
	AggregateStats transformerInputUpdateNorm;
	std::vector<AggregateStats> transformerBlockUpdateNorms;
	AggregateStats transformerFinalNormUpdateNorm;
	AggregateStats transformerHeadUpdateNorm;
	AggregateStats transformerHeadShare;
	AggregateStats transformerNonHeadShare;
	AggregateStats transformerApplyMs;
	bool transformerTrainMarginValid;
	AggregateStats transformerTrainTargetMargin;
	AggregateStats transformerTrainHardNegativeLogit;
	bool transformerTestMarginValid;
	AggregateStats transformerTestTargetMargin;
	AggregateStats transformerTestHardNegativeLogit;
	bool contextFamilyDiagValid;
	std::vector<AggregateStats> contextRoleNll;
	std::vector<AggregateStats> contextRoleShare;
	std::vector<AggregateStats> contextRecallDistanceNll;
	std::vector<AggregateStats> contextRecallDistanceShare;
	std::vector<AggregateStats> contextRecallSubtypeNll;
	std::vector<AggregateStats> contextRecallSubtypeShare;
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
	      asterBoundaryMs(),
	      asterSetupMs(),
	      asterTransportMs(),
	      asterTransferFitMs(),
	      asterStateFitMs(),
	      asterInnovationFitMs(),
	      asterApplyMs(),
	      aegisDiagValid(false),
	      aegisLambdaSpatial(),
	      aegisLambdaPredictive(),
	      aegisLambdaOutput(),
	      aegisPredictivePredicted(),
	      aegisPredictiveRealized(),
	      aegisOutputPredicted(),
	      aegisOutputRealized(),
	      aegisPredictiveError(),
	      aegisOutputError(),
	      aegisChannelDisagreement(),
	      citadelDiagValid(false),
	      citadelAnchor(),
	      citadelHardRegimeMass(),
	      citadelSparrowTrust(),
	      rampartDiagValid(false),
	      rampartTau(),
	      rampartBudget(),
	      rampartCovariance(),
	      rampartSparrowTrust(),
	      meritDiagValid(false),
	      meritTau(),
	      meritBudget(),
	      meritCovariance(),
	      meritSparrowTrust(),
	      meritGeometryTrust(),
	      strataDiagValid(false),
	      strataNullMode(),
	      strataPredictiveMode(),
	      strataOutputMode(),
	      strataCoupledMode(),
	      strataBudget(),
	      strataNullBenefit(),
	      strataPredictiveBenefit(),
	      strataOutputBenefit(),
	      strataCoupledBenefit(),
	      strataSelectedExcess(),
	      strataSwitchRate(),
	      transformerGapDiagValid(false),
	      transformerInputUpdateNorm(),
	      transformerBlockUpdateNorms(),
	      transformerFinalNormUpdateNorm(),
	      transformerHeadUpdateNorm(),
	      transformerHeadShare(),
	      transformerNonHeadShare(),
	      transformerApplyMs(),
	      transformerTrainMarginValid(false),
	      transformerTrainTargetMargin(),
	      transformerTrainHardNegativeLogit(),
	      transformerTestMarginValid(false),
	      transformerTestTargetMargin(),
	      transformerTestHardNegativeLogit(),
	      contextFamilyDiagValid(false),
	      contextRoleNll(),
	      contextRoleShare(),
	      contextRecallDistanceNll(),
	      contextRecallDistanceShare(),
	      contextRecallSubtypeNll(),
	      contextRecallSubtypeShare(),
	      ok(false),
	      status()
	{
	}
};

static void fill_sparrow_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);
static void fill_helm_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);
static void fill_aster_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);
static void fill_aegis_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);
static void fill_citadel_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);
static void fill_rampart_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);
static void fill_merit_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);
static void fill_strata_run_result(const CaptureMetricsCallbacks& cb, RunResult& out);

struct ContextFamilyBucketAccum
{
	double lossSum;
	unsigned long long count;
	ContextFamilyBucketAccum() : lossSum(0.0), count(0ULL) {}
};

static double logits_target_nll(const float* logits, unsigned int vocab, unsigned int targetId)
{
	if (!logits || vocab == 0u || targetId >= vocab)
		return 0.0;
	float maxLogit = logits[0];
	for (unsigned int i = 1u; i < vocab; ++i)
	{
		if (logits[i] > maxLogit)
			maxLogit = logits[i];
	}
	double sumExp = 0.0;
	for (unsigned int i = 0u; i < vocab; ++i)
		sumExp += exp(static_cast<double>(logits[i] - maxLogit));
	const double logZ = static_cast<double>(maxLogit) + log(sumExp);
	return logZ - static_cast<double>(logits[targetId]);
}

static bool fill_context_family_test_diag(const TokenDataset& data,
                                          const glades::NNetwork& net,
                                          RunResult& out,
                                          std::string* errMsg)
{
	out.contextFamilyDiagValid = false;
	out.contextRoleNll.clear();
	out.contextRoleShare.clear();
	out.contextRecallDistanceNll.clear();
	out.contextRecallDistanceShare.clear();
	out.contextRecallSubtypeNll.clear();
	out.contextRecallSubtypeShare.clear();

	if (data.testTokenStream.empty() || data.testSpans.empty() ||
	    data.testRoleBuckets.size() != data.testTokenStream.size() ||
	    data.testRecallDistanceBuckets.size() != data.testTokenStream.size() ||
	    data.testRecallSubtypeBuckets.size() != data.testTokenStream.size())
	{
		return false;
	}

	const unsigned int batchSize = static_cast<unsigned int>(data.testSpans.size());
	unsigned int maxLen = 0u;
	for (size_t i = 0u; i < data.testSpans.size(); ++i)
	{
		if (data.testSpans[i].length > maxLen)
			maxLen = data.testSpans[i].length;
	}
	if (batchSize == 0u || maxLen < 2u)
		return false;

	glades::NNetwork::TransformerLmBatchSession session;
	const glades::NNetworkStatus stReset = net.transformerLmBatchSessionReset(session, batchSize, maxLen);
	if (!stReset.ok())
	{
		if (errMsg)
			*errMsg = stReset.message;
		return false;
	}

	std::vector<unsigned int> tokenIds(batchSize, data.padTokenId);
	std::vector<unsigned char> active(batchSize, 0u);
	std::vector<float> logitsFlat;
	std::vector<ContextFamilyBucketAccum> roleAcc(CTX_ROLE_COUNT);
	std::vector<ContextFamilyBucketAccum> distanceAcc(CTX_RECALL_DISTANCE_COUNT);
	std::vector<ContextFamilyBucketAccum> subtypeAcc(CTX_SUBTYPE_COUNT);
	unsigned long long totalCount = 0ULL;
	unsigned long long recallCount = 0ULL;

	for (unsigned int t = 0u; t + 1u < maxLen; ++t)
	{
		bool anyActive = false;
		for (unsigned int b = 0u; b < batchSize; ++b)
		{
			const glades::DataInput::SequenceSpan& span = data.testSpans[b];
			if ((t + 1u) < span.length)
			{
				active[b] = 1u;
				tokenIds[b] = data.testTokenStream[static_cast<size_t>(span.start + t)];
				anyActive = true;
			}
			else
			{
				active[b] = 0u;
				tokenIds[b] = data.padTokenId;
			}
		}
		if (!anyActive)
			break;

		const glades::NNetworkStatus stStep =
		    net.transformerLmBatchSessionAppendSelective(session, tokenIds, active, &logitsFlat);
		if (!stStep.ok())
		{
			if (errMsg)
				*errMsg = stStep.message;
			return false;
		}
		const unsigned int vocab =
		    (batchSize > 0u) ? static_cast<unsigned int>(logitsFlat.size() / static_cast<size_t>(batchSize)) : 0u;
		if (vocab == 0u)
		{
			if (errMsg)
				*errMsg = "context diagnostic produced empty logits";
			return false;
		}

		for (unsigned int b = 0u; b < batchSize; ++b)
		{
			if (active[b] == 0u)
				continue;
			const glades::DataInput::SequenceSpan& span = data.testSpans[b];
			const size_t targetIndex = static_cast<size_t>(span.start + t + 1u);
			const unsigned int targetId = data.testTokenStream[targetIndex];
			if (targetId >= vocab)
				continue;
			const float* row = &logitsFlat[static_cast<size_t>(b) * static_cast<size_t>(vocab)];
			const double nll = logits_target_nll(row, vocab, targetId);

			const unsigned char role = data.testRoleBuckets[targetIndex];
			if (role < CTX_ROLE_COUNT)
			{
				roleAcc[role].lossSum += nll;
				roleAcc[role].count += 1ULL;
			}
			totalCount += 1ULL;

			const unsigned char dist = data.testRecallDistanceBuckets[targetIndex];
			if (dist < CTX_RECALL_DISTANCE_COUNT)
			{
				distanceAcc[dist].lossSum += nll;
				distanceAcc[dist].count += 1ULL;
				recallCount += 1ULL;
			}

			const unsigned char subtype = data.testRecallSubtypeBuckets[targetIndex];
			if (subtype < CTX_SUBTYPE_COUNT)
			{
				subtypeAcc[subtype].lossSum += nll;
				subtypeAcc[subtype].count += 1ULL;
			}
		}
	}

	if (totalCount == 0ULL)
		return false;

	out.contextRoleNll.assign(CTX_ROLE_COUNT, 0.0);
	out.contextRoleShare.assign(CTX_ROLE_COUNT, 0.0);
	for (unsigned int i = 0u; i < CTX_ROLE_COUNT; ++i)
	{
		if (roleAcc[i].count > 0ULL)
			out.contextRoleNll[i] = roleAcc[i].lossSum / static_cast<double>(roleAcc[i].count);
		out.contextRoleShare[i] = static_cast<double>(roleAcc[i].count) / static_cast<double>(totalCount);
	}

	out.contextRecallDistanceNll.assign(CTX_RECALL_DISTANCE_COUNT, 0.0);
	out.contextRecallDistanceShare.assign(CTX_RECALL_DISTANCE_COUNT, 0.0);
	out.contextRecallSubtypeNll.assign(CTX_SUBTYPE_COUNT, 0.0);
	out.contextRecallSubtypeShare.assign(CTX_SUBTYPE_COUNT, 0.0);
	if (recallCount > 0ULL)
	{
		for (unsigned int i = 0u; i < CTX_RECALL_DISTANCE_COUNT; ++i)
		{
			if (distanceAcc[i].count > 0ULL)
				out.contextRecallDistanceNll[i] = distanceAcc[i].lossSum / static_cast<double>(distanceAcc[i].count);
			out.contextRecallDistanceShare[i] = static_cast<double>(distanceAcc[i].count) / static_cast<double>(recallCount);
		}
		for (unsigned int i = 0u; i < CTX_SUBTYPE_COUNT; ++i)
		{
			if (subtypeAcc[i].count > 0ULL)
				out.contextRecallSubtypeNll[i] = subtypeAcc[i].lossSum / static_cast<double>(subtypeAcc[i].count);
			out.contextRecallSubtypeShare[i] = static_cast<double>(subtypeAcc[i].count) / static_cast<double>(recallCount);
		}
	}

	out.contextFamilyDiagValid = true;
	return true;
}

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
	case VARIANT_ADAMW_GROUP: return "AdamW-Group";
	case VARIANT_ATLAS_BASE: return "ATLAS-BSRP";
	case VARIANT_ATLAS_SPARROW: return "ATLAS-SPARROW";
	case VARIANT_ATLAS_HELM: return "ATLAS-HELM";
	case VARIANT_ATLAS_ASTER: return "ATLAS-ASTER";
	case VARIANT_ATLAS_AEGIS: return "ATLAS-AEGIS";
	case VARIANT_ATLAS_CITADEL: return "ATLAS-CITADEL";
	case VARIANT_ATLAS_RAMPART: return "ATLAS-RAMPART";
	case VARIANT_ATLAS_MERIT: return "ATLAS-MERIT";
	case VARIANT_ATLAS_STRATA: return "ATLAS-STRATA";
	case VARIANT_ATLAS_AURORA: return "ATLAS-AURORA";
	case VARIANT_ATLAS_SEAM: return "ATLAS-SEAM";
	case VARIANT_ATLAS_QUASAR: return "ATLAS-QUASAR";
	case VARIANT_ATLAS_GEODE: return "ATLAS-GEODE";
	case VARIANT_ATLAS_ECHO: return "ATLAS-ECHO";
	case VARIANT_ATLAS_BIMAP: return "ATLAS-BIMAP";
	case VARIANT_ATLAS_PACT: return "ATLAS-PACT";
	case VARIANT_ATLAS_RACER: return "ATLAS-RACER";
	case VARIANT_ATLAS_KRON: return "ATLAS-KRON";
	case VARIANT_ATLAS_MUON: return "ATLAS-MUON";
	case VARIANT_ATLAS_MATRA: return "ATLAS-MATRA";
	default: return "Unknown";
	}
}

static bool variant_matches_selection(VariantSelection selection, VariantKind variant)
{
	switch (selection)
	{
	case VARIANT_SELECTION_ALL:
		return true;
	case VARIANT_SELECTION_ADAMW:
		return variant == VARIANT_ADAMW;
	case VARIANT_SELECTION_ADAMW_GROUP:
		return variant == VARIANT_ADAMW_GROUP;
	case VARIANT_SELECTION_ATLAS_BASE:
		return variant == VARIANT_ATLAS_BASE;
	case VARIANT_SELECTION_ATLAS_SPARROW:
		return variant == VARIANT_ATLAS_SPARROW;
	case VARIANT_SELECTION_ATLAS_HELM:
		return variant == VARIANT_ATLAS_HELM;
	case VARIANT_SELECTION_ATLAS_ASTER:
		return variant == VARIANT_ATLAS_ASTER;
	case VARIANT_SELECTION_ATLAS_AEGIS:
		return variant == VARIANT_ATLAS_AEGIS;
	case VARIANT_SELECTION_ATLAS_CITADEL:
		return variant == VARIANT_ATLAS_CITADEL;
	case VARIANT_SELECTION_ATLAS_RAMPART:
		return variant == VARIANT_ATLAS_RAMPART;
	case VARIANT_SELECTION_ATLAS_MERIT:
		return variant == VARIANT_ATLAS_MERIT;
	case VARIANT_SELECTION_ATLAS_STRATA:
		return variant == VARIANT_ATLAS_STRATA;
	case VARIANT_SELECTION_ATLAS_AURORA:
		return variant == VARIANT_ATLAS_AURORA;
	case VARIANT_SELECTION_ATLAS_SEAM:
		return variant == VARIANT_ATLAS_SEAM;
	case VARIANT_SELECTION_ATLAS_QUASAR:
		return variant == VARIANT_ATLAS_QUASAR;
	case VARIANT_SELECTION_ATLAS_GEODE:
		return variant == VARIANT_ATLAS_GEODE;
	case VARIANT_SELECTION_ATLAS_ECHO:
		return variant == VARIANT_ATLAS_ECHO;
	case VARIANT_SELECTION_ATLAS_BIMAP:
		return variant == VARIANT_ATLAS_BIMAP;
	case VARIANT_SELECTION_ATLAS_PACT:
		return variant == VARIANT_ATLAS_PACT;
	case VARIANT_SELECTION_ATLAS_RACER:
		return variant == VARIANT_ATLAS_RACER;
	case VARIANT_SELECTION_ATLAS_KRON:
		return variant == VARIANT_ATLAS_KRON;
	case VARIANT_SELECTION_ATLAS_MUON:
		return variant == VARIANT_ATLAS_MUON;
	case VARIANT_SELECTION_ATLAS_MATRA:
		return variant == VARIANT_ATLAS_MATRA;
	default:
		return false;
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

static std::vector<AggregateStats> compute_stats_by_index(const std::vector< std::vector<double> >& values)
{
	std::vector<AggregateStats> out;
	size_t maxSize = 0u;
	for (size_t i = 0; i < values.size(); ++i)
	{
		if (values[i].size() > maxSize)
			maxSize = values[i].size();
	}
	out.resize(maxSize);
	for (size_t idx = 0; idx < maxSize; ++idx)
	{
		std::vector<double> bucket;
		for (size_t row = 0; row < values.size(); ++row)
		{
			if (idx < values[row].size())
				bucket.push_back(values[row][idx]);
		}
		out[idx] = compute_stats(bucket);
	}
	return out;
}

static double squared_snapshot_delta(const std::vector<float>& before,
                                     const std::vector<float>& after)
{
	if (before.size() != after.size())
		return 0.0;
	double sumsq = 0.0;
	for (size_t i = 0; i < before.size(); ++i)
	{
		const double delta = static_cast<double>(after[i]) - static_cast<double>(before[i]);
		sumsq += delta * delta;
	}
	return sumsq;
}

static void fill_transformer_gap_from_snapshots(
    const glades::NNetwork::TransformerGroupedParameterSnapshot& before,
    const glades::NNetwork::TransformerGroupedParameterSnapshot& after,
    RunResult& out)
{
	if (!before.valid || !after.valid)
		return;
	if (before.blockGroups.size() != after.blockGroups.size())
		return;

	const double inputSq = squared_snapshot_delta(before.inputGroup, after.inputGroup);
	const double finalNormSq = squared_snapshot_delta(before.finalNormGroup, after.finalNormGroup);
	const double headSq = squared_snapshot_delta(before.headGroup, after.headGroup);
	double totalSq = inputSq + finalNormSq + headSq;

	out.transformerBlockUpdateNorms.assign(before.blockGroups.size(), 0.0);
	for (size_t i = 0; i < before.blockGroups.size(); ++i)
	{
		const double blockSq = squared_snapshot_delta(before.blockGroups[i], after.blockGroups[i]);
		out.transformerBlockUpdateNorms[i] = std::sqrt(std::max(0.0, blockSq));
		totalSq += blockSq;
	}

	out.transformerGapDiagValid = true;
	out.transformerInputUpdateNorm = std::sqrt(std::max(0.0, inputSq));
	out.transformerFinalNormUpdateNorm = std::sqrt(std::max(0.0, finalNormSq));
	out.transformerHeadUpdateNorm = std::sqrt(std::max(0.0, headSq));
	if (totalSq > 0.0)
	{
		out.transformerHeadShare = headSq / totalSq;
		out.transformerNonHeadShare = (totalSq - headSq) / totalSq;
	}
}

static std::vector<double> make_transformer_profile(const Summary& s)
{
	std::vector<double> profile;
	if (!s.transformerGapDiagValid)
		return profile;
	profile.push_back(s.transformerInputUpdateNorm.mean);
	for (size_t i = 0; i < s.transformerBlockUpdateNorms.size(); ++i)
		profile.push_back(s.transformerBlockUpdateNorms[i].mean);
	profile.push_back(s.transformerFinalNormUpdateNorm.mean);
	profile.push_back(s.transformerHeadUpdateNorm.mean);
	return profile;
}

static double profile_cosine(const std::vector<double>& a, const std::vector<double>& b)
{
	if (a.empty() || a.size() != b.size())
		return 0.0;
	double dot = 0.0;
	double aa = 0.0;
	double bb = 0.0;
	for (size_t i = 0; i < a.size(); ++i)
	{
		dot += a[i] * b[i];
		aa += a[i] * a[i];
		bb += b[i] * b[i];
	}
	if (!(aa > 0.0) || !(bb > 0.0))
		return 0.0;
	return dot / (sqrt(aa) * sqrt(bb));
}

static void print_usage()
{
	printf("Usage: glades-unit-tests atlas-alt-bench [options]\n");
	printf("Options:\n");
	printf("  --mode all|token-lm|token-lm-large|token-lm-context|token-lm-context-large|token-lm-document|token-lm-corpus|token-lm-corpus-large|teacher-student|latent-forecast|nonlinear-forecast|teacher-sweep|teacher-canonical\n");
	printf("                                         Run the alternate-task benches or the teacher-student sweep (default: all)\n");
	printf("  --variant all|adamw|adamw-group|base|sparrow|helm|aster|aegis|citadel|rampart|merit|strata|aurora|seam|quasar|geode|echo|bimap|pact|racer|kron|muon|matra\n");
	printf("                                         Restrict runs to one optimizer variant when the case supports it (default: all)\n");
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
	printf("  --atlas-kappa-enabled 0|1             Enable KAPPA retrieval observables inside transformer ASTER/AEGIS (default: 0)\n");
	printf("  --atlas-kappa-heads N                 Number of tracked attention heads per block for KAPPA (default: 1)\n");
	printf("  --atlas-kappa-lag-buckets N           Number of lag buckets for KAPPA retrieval summaries (default: 4)\n");
	printf("  --atlas-kappa-rank N                  Projected value channels per head/lag KAPPA observable (default: 2)\n");
	printf("  --atlas-aurora-adamw-backbone 0|1     Use AdamW as the AURORA backbone update instead of ATLAS/BSRP (default: 1)\n");
	printf("  --atlas-aurora-head-gain X            Extra AURORA head-actuation gain on transformer token heads (default: 3.0)\n");
	printf("  --atlas-aurora-body-trust-scale X     Retained non-head SPARROW trust inside AURORA (default: 0.60)\n");
	printf("  --atlas-geode-geometry-scale X        Low-rank geometry strength for GEODE (default: 1.0)\n");
	printf("  --atlas-geode-predictive-scale X      SPARROW-style active prediction blend for GEODE (default: 0.25)\n");
	printf("  --atlas-echo-geometry-scale X         Operand-harvested two-sided diagonal strength for ECHO (default: 1.0)\n");
	printf("  --atlas-echo-final-geometry-scale X   Final ECHO geometry strength after schedule decay (default: 1.0)\n");
	printf("  --atlas-echo-decay-steps N            Optimizer steps for linear ECHO geometry decay (default: 0 = off)\n");
	printf("  --atlas-echo-cadence N                Optimizer steps between ECHO metric refreshes (default: 1)\n");
	printf("  --atlas-echo-trust-scale X            Scalar trust gate for ECHO geometry, 0 disables gating (default: 0.0)\n");
	printf("  --atlas-echo-predictive-scale X       Bounded one-step predictive blend for ECHO momentum (default: 0.0)\n");
	printf("  --atlas-echo-structural-scale X       Strength of grouped structural factors inside ECHO (default: 0.0)\n");
	printf("  --atlas-echo-structural-groups N      Number of contiguous row/col groups for ECHO structure (default: 1)\n");
	printf("  --atlas-echo-scope all|large-only|late-head|late-head-large  Restrict ECHO to all matrices, large matrices only, last block + head, or that subset filtered to large matrices (default: all)\n");
	printf("  --atlas-bimap-low-rank 0|1            Enable BiMAP-v2 low-rank row/column factors (default: 1)\n");
	printf("  --atlas-bimap-scope all|head|late|late-head  Restrict BiMAP to all matrices, head only, late block only, or late block + head (default: all)\n");
	printf("  --atlas-bimap-geometry-scale X        Row/column geometry strength for BiMAP (default: 1.0)\n");
	printf("  --atlas-bimap-predictive-scale X      Momentum secant blend for BiMAP (default: 0.15)\n");
	printf("  --atlas-bimap-factor-cadence N        Steps between BiMAP factor EMA refreshes (default: 8)\n");
	printf("  --atlas-pact-low-rank 0|1             Enable low-rank row/column factors for PACT (default: 1)\n");
	printf("  --atlas-pact-geometry-scale X         Row/column geometry strength for PACT (default: 1.0)\n");
	printf("  --atlas-pact-predictive-scale X       Bounded secant transport scale for PACT (default: 0.10)\n");
	printf("  --atlas-pact-factor-cadence N         Steps between PACT factor EMA refreshes (default: 8)\n");
	printf("  --atlas-pact-cost-scale X             Analytical overhead penalty scale for PACT promotion (default: 0.0010)\n");
	printf("  --atlas-pact-promote-threshold X      Promotion threshold for PACT block EMA score (default: 0.0)\n");
	printf("  --atlas-pact-demote-threshold X       Demotion threshold for PACT block EMA score (default: -0.0005)\n");
	printf("  --atlas-racer-geometry-scale X        Two-sided geometry strength for RACER-lite (default: 1.0)\n");
	printf("  --atlas-racer-predictive-scale X      Bounded secant transport scale for RACER-lite (default: 0.05)\n");
	printf("  --atlas-racer-factor-cadence N        Steps between RACER-lite factor EMA refreshes (default: 8)\n");
	printf("  --atlas-racer-risk-scale X            Noise-shaping penalty scale for RACER-lite (default: 0.50)\n");
	printf("  --atlas-racer-cost-scale X            Compute-cost penalty scale for RACER-lite (default: 0.0010)\n");
	printf("  --atlas-racer-promote-threshold X     Promotion threshold for RACER-lite block EMA score (default: 0.0)\n");
	printf("  --atlas-racer-demote-threshold X      Demotion threshold for RACER-lite block EMA score (default: -0.0005)\n");
	printf("  --atlas-kron-geometry-scale X         Two-sided factor strength for KRON (default: 1.0)\n");
	printf("  --atlas-kron-predictive-scale X       Bounded secant blend for KRON (default: 0.05)\n");
	printf("  --atlas-kron-factor-cadence N         Steps between KRON factor refreshes (default: 8)\n");
	printf("  --atlas-kron-damping X                Normalized covariance damping for KRON (default: 0.10)\n");
	printf("  --atlas-muon-geometry-scale X         Orthogonalized-momentum blend strength for MUON-lite (default: 1.0)\n");
	printf("  --atlas-muon-predictive-scale X       Bounded secant blend for MUON-lite (default: 0.05)\n");
	printf("  --atlas-muon-max-aspect X             Maximum block aspect ratio eligible for MUON-lite (default: 1.50)\n");
	printf("  --atlas-muon-min-dim N                Minimum block side length eligible for MUON-lite (default: 8)\n");
	printf("  --atlas-muon-damping X                Gram damping inside MUON-lite polar factors (default: 0.01)\n");
	printf("  --atlas-matra-geometry-scale X        Two-sided geometry trust cap for MATRA (default: 1.0)\n");
	printf("  --atlas-matra-orthogonal-scale X      Orthogonal residual trust cap for MATRA (default: 0.5)\n");
	printf("  --atlas-matra-predictive-scale X      Bounded one-step predictive transport for MATRA (default: 0.05)\n");
	printf("  --atlas-matra-trust-radius X          Total structured trust budget for MATRA (default: 0.50)\n");
	printf("  --atlas-matra-cadence N               Steps between MATRA row/column metric refreshes (default: 1)\n");
	printf("  --atlas-matra-max-aspect X            Maximum block aspect ratio eligible for MATRA orthogonal branch (default: 1.50)\n");
	printf("  --atlas-matra-min-dim N               Minimum block side length eligible for MATRA orthogonal branch (default: 8)\n");
	printf("  --atlas-matra-damping X               Gram damping inside MATRA orthogonal factors (default: 0.01)\n");
	printf("  --gpu-enable 0|1                      Attempt GPU offload when available (default: 0)\n");
	printf("  --gpu-device N                        CUDA device id when GPU offload is enabled (default: 0)\n");
	printf("  --token-epochs N                      Token-LM epochs (default: 6)\n");
	printf("  --token-train-seqs N                  Token-LM train sequence count (default: 128)\n");
	printf("  --token-test-seqs N                   Token-LM test sequence count (default: 32)\n");
	printf("  --token-seq-len N                     Token-LM sequence length (default: 32)\n");
	printf("  --token-dmodel N                      Token-LM decoder width (default: 48)\n");
	printf("  --token-dff N                         Token-LM FFN width (default: 192)\n");
	printf("  --token-layers N                      Token-LM decoder layers (default: 2)\n");
	printf("  --token-heads N                       Token-LM attention heads (default: 4)\n");
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

static void apply_large_token_preset(BenchConfig& cfg)
{
	cfg.token.vocab = 97u;
	cfg.token.dModel = 24u;
	cfg.token.dFF = 96u;
	cfg.token.layers = 2u;
	cfg.token.heads = 4u;
	cfg.token.kvHeads = 4u;
	cfg.token.seqLen = 24u;
	cfg.token.trainSeqs = 8u;
	cfg.token.testSeqs = 4u;
	cfg.token.epochs = 1u;
	cfg.token.adamLR = 0.0010f;
	cfg.token.atlasLR = 0.020f;
}

static void apply_context_token_preset(BenchConfig& cfg)
{
	cfg.token.vocab = 129u;
	cfg.token.dModel = 32u;
	cfg.token.dFF = 128u;
	cfg.token.layers = 2u;
	cfg.token.heads = 4u;
	cfg.token.kvHeads = 4u;
	cfg.token.seqLen = 56u;
	cfg.token.trainSeqs = 24u;
	cfg.token.testSeqs = 8u;
	cfg.token.epochs = 2u;
	cfg.token.adamLR = 0.0010f;
	cfg.token.atlasLR = 0.020f;
}

static void apply_context_large_token_preset(BenchConfig& cfg)
{
	cfg.token.vocab = 193u;
	cfg.token.dModel = 40u;
	cfg.token.dFF = 160u;
	cfg.token.layers = 3u;
	cfg.token.heads = 5u;
	cfg.token.kvHeads = 5u;
	cfg.token.seqLen = 80u;
	cfg.token.trainSeqs = 32u;
	cfg.token.testSeqs = 8u;
	cfg.token.epochs = 2u;
	cfg.token.adamLR = 0.0010f;
	cfg.token.atlasLR = 0.020f;
}

static void apply_document_token_preset(BenchConfig& cfg)
{
	cfg.token.vocab = 257u;
	cfg.token.dModel = 48u;
	cfg.token.dFF = 192u;
	cfg.token.layers = 3u;
	cfg.token.heads = 6u;
	cfg.token.kvHeads = 6u;
	cfg.token.seqLen = 96u;
	cfg.token.trainSeqs = 48u;
	cfg.token.testSeqs = 12u;
	cfg.token.epochs = 2u;
	cfg.token.adamLR = 0.0010f;
	cfg.token.atlasLR = 0.020f;
}

static void apply_corpus_token_preset(BenchConfig& cfg)
{
	cfg.token.vocab = 513u;
	cfg.token.dModel = 48u;
	cfg.token.dFF = 192u;
	cfg.token.layers = 3u;
	cfg.token.heads = 6u;
	cfg.token.kvHeads = 6u;
	cfg.token.seqLen = 96u;
	cfg.token.trainSeqs = 48u;
	cfg.token.testSeqs = 12u;
	cfg.token.epochs = 2u;
	cfg.token.adamLR = 0.0010f;
	cfg.token.atlasLR = 0.020f;
}

static void apply_corpus_large_token_preset(BenchConfig& cfg)
{
	cfg.token.vocab = 769u;
	cfg.token.dModel = 56u;
	cfg.token.dFF = 224u;
	cfg.token.layers = 4u;
	cfg.token.heads = 8u;
	cfg.token.kvHeads = 8u;
	cfg.token.seqLen = 112u;
	cfg.token.trainSeqs = 64u;
	cfg.token.testSeqs = 16u;
	cfg.token.epochs = 2u;
	cfg.token.adamLR = 0.0010f;
	cfg.token.atlasLR = 0.020f;
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
	if (streq(text, "token-lm-large") || streq(text, "llm-large") || streq(text, "token-large"))
	{
		outMode = MODE_TOKEN_LM_LARGE;
		return true;
	}
	if (streq(text, "token-lm-context") || streq(text, "llm-context") || streq(text, "token-context") || streq(text, "context-lm"))
	{
		outMode = MODE_TOKEN_LM_CONTEXT;
		return true;
	}
	if (streq(text, "token-lm-context-large") || streq(text, "llm-context-large") || streq(text, "token-context-large") || streq(text, "context-lm-large"))
	{
		outMode = MODE_TOKEN_LM_CONTEXT_LARGE;
		return true;
	}
	if (streq(text, "token-lm-document") || streq(text, "llm-document") || streq(text, "token-document") || streq(text, "document-lm") || streq(text, "doc-lm"))
	{
		outMode = MODE_TOKEN_LM_DOCUMENT;
		return true;
	}
	if (streq(text, "token-lm-corpus") || streq(text, "llm-corpus") || streq(text, "token-corpus")
	    || streq(text, "corpus-lm") || streq(text, "doc-corpus"))
	{
		outMode = MODE_TOKEN_LM_CORPUS;
		return true;
	}
	if (streq(text, "token-lm-corpus-large") || streq(text, "llm-corpus-large") || streq(text, "token-corpus-large")
	    || streq(text, "corpus-lm-large") || streq(text, "doc-corpus-large"))
	{
		outMode = MODE_TOKEN_LM_CORPUS_LARGE;
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

static bool parse_variant_arg(const char* text, VariantSelection& outSelection)
{
	if (!text || !*text)
		return false;
	if (streq(text, "all"))
	{
		outSelection = VARIANT_SELECTION_ALL;
		return true;
	}
	if (streq(text, "adamw") || streq(text, "adam"))
	{
		outSelection = VARIANT_SELECTION_ADAMW;
		return true;
	}
	if (streq(text, "adamw-group") || streq(text, "groupadam") || streq(text, "group"))
	{
		outSelection = VARIANT_SELECTION_ADAMW_GROUP;
		return true;
	}
	if (streq(text, "base") || streq(text, "atlas-base") || streq(text, "bsrp"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_BASE;
		return true;
	}
	if (streq(text, "sparrow") || streq(text, "atlas-sparrow"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_SPARROW;
		return true;
	}
	if (streq(text, "helm") || streq(text, "atlas-helm"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_HELM;
		return true;
	}
	if (streq(text, "aster") || streq(text, "atlas-aster"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_ASTER;
		return true;
	}
	if (streq(text, "aegis") || streq(text, "atlas-aegis"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_AEGIS;
		return true;
	}
	if (streq(text, "citadel") || streq(text, "atlas-citadel"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_CITADEL;
		return true;
	}
	if (streq(text, "rampart") || streq(text, "atlas-rampart"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_RAMPART;
		return true;
	}
	if (streq(text, "merit") || streq(text, "atlas-merit"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_MERIT;
		return true;
	}
	if (streq(text, "strata") || streq(text, "atlas-strata"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_STRATA;
		return true;
	}
	if (streq(text, "aurora") || streq(text, "atlas-aurora"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_AURORA;
		return true;
	}
	if (streq(text, "seam") || streq(text, "atlas-seam"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_SEAM;
		return true;
	}
	if (streq(text, "quasar") || streq(text, "atlas-quasar"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_QUASAR;
		return true;
	}
	if (streq(text, "geode") || streq(text, "atlas-geode"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_GEODE;
		return true;
	}
	if (streq(text, "echo") || streq(text, "atlas-echo"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_ECHO;
		return true;
	}
	if (streq(text, "bimap") || streq(text, "atlas-bimap"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_BIMAP;
		return true;
	}
	if (streq(text, "pact") || streq(text, "atlas-pact"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_PACT;
		return true;
	}
	if (streq(text, "racer") || streq(text, "atlas-racer"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_RACER;
		return true;
	}
	if (streq(text, "kron") || streq(text, "atlas-kron"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_KRON;
		return true;
	}
	if (streq(text, "muon") || streq(text, "atlas-muon"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_MUON;
		return true;
	}
	if (streq(text, "matra") || streq(text, "atlas-matra"))
	{
		outSelection = VARIANT_SELECTION_ATLAS_MATRA;
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
			if (cfg.mode == MODE_TOKEN_LM_LARGE)
				apply_large_token_preset(cfg);
			else if (cfg.mode == MODE_TOKEN_LM_CONTEXT)
				apply_context_token_preset(cfg);
			else if (cfg.mode == MODE_TOKEN_LM_CONTEXT_LARGE)
				apply_context_large_token_preset(cfg);
			else if (cfg.mode == MODE_TOKEN_LM_DOCUMENT)
				apply_document_token_preset(cfg);
			else if (cfg.mode == MODE_TOKEN_LM_CORPUS)
				apply_corpus_token_preset(cfg);
			else if (cfg.mode == MODE_TOKEN_LM_CORPUS_LARGE)
				apply_corpus_large_token_preset(cfg);
		}
		else if (streq(argv[i], "--variant") && i + 1 < argc)
		{
			if (!parse_variant_arg(argv[++i], cfg.variantSelection))
			{
				err = "invalid --variant value";
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
		else if (streq(argv[i], "--atlas-kappa-enabled") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasKappaEnabled) || cfg.atlasKappaEnabled > 1u)
			{
				err = "invalid --atlas-kappa-enabled";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-kappa-heads") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasKappaHeads) || cfg.atlasKappaHeads == 0u)
			{
				err = "invalid --atlas-kappa-heads";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-kappa-lag-buckets") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasKappaLagBuckets) || cfg.atlasKappaLagBuckets == 0u)
			{
				err = "invalid --atlas-kappa-lag-buckets";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-kappa-rank") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasKappaRank) || cfg.atlasKappaRank == 0u)
			{
				err = "invalid --atlas-kappa-rank";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-aurora-adamw-backbone") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasAuroraAdamwBackbone) || cfg.atlasAuroraAdamwBackbone > 1u)
			{
				err = "invalid --atlas-aurora-adamw-backbone";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-aurora-head-gain") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasAuroraHeadGain) || cfg.atlasAuroraHeadGain < 0.0f)
			{
				err = "invalid --atlas-aurora-head-gain";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-aurora-body-trust-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasAuroraBodyTrustScale) || cfg.atlasAuroraBodyTrustScale < 0.0f)
			{
				err = "invalid --atlas-aurora-body-trust-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-geode-geometry-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasGeodeGeometryScale) || cfg.atlasGeodeGeometryScale < 0.0f)
			{
				err = "invalid --atlas-geode-geometry-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-geode-predictive-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasGeodePredictiveScale) || cfg.atlasGeodePredictiveScale < 0.0f)
			{
				err = "invalid --atlas-geode-predictive-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-echo-geometry-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasEchoGeometryScale) || cfg.atlasEchoGeometryScale < 0.0f)
			{
				err = "invalid --atlas-echo-geometry-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-echo-final-geometry-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasEchoGeometryScaleFinal) || cfg.atlasEchoGeometryScaleFinal < 0.0f)
			{
				err = "invalid --atlas-echo-final-geometry-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-echo-decay-steps") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasEchoGeometryDecaySteps))
			{
				err = "invalid --atlas-echo-decay-steps";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-echo-cadence") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasEchoMetricCadence))
			{
				err = "invalid --atlas-echo-cadence";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-echo-trust-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasEchoTrustScale) || cfg.atlasEchoTrustScale < 0.0f)
			{
				err = "invalid --atlas-echo-trust-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-echo-predictive-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasEchoPredictiveScale) || cfg.atlasEchoPredictiveScale < 0.0f)
			{
				err = "invalid --atlas-echo-predictive-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-echo-structural-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasEchoStructuralScale) || cfg.atlasEchoStructuralScale < 0.0f)
			{
				err = "invalid --atlas-echo-structural-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-echo-structural-groups") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasEchoStructuralGroups) || cfg.atlasEchoStructuralGroups == 0u)
			{
				err = "invalid --atlas-echo-structural-groups";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-echo-scope") && i + 1 < argc)
		{
			if (!parse_echo_scope_arg(argv[++i], cfg.atlasEchoScope))
			{
				err = "invalid --atlas-echo-scope";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-bimap-low-rank") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasBiMAPLowRank) || cfg.atlasBiMAPLowRank > 1u)
			{
				err = "invalid --atlas-bimap-low-rank";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-bimap-scope") && i + 1 < argc)
		{
			if (!parse_bimap_scope_arg(argv[++i], cfg.atlasBiMAPScope))
			{
				err = "invalid --atlas-bimap-scope";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-bimap-geometry-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasBiMAPGeometryScale) || cfg.atlasBiMAPGeometryScale < 0.0f)
			{
				err = "invalid --atlas-bimap-geometry-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-bimap-predictive-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasBiMAPPredictiveScale) || cfg.atlasBiMAPPredictiveScale < 0.0f)
			{
				err = "invalid --atlas-bimap-predictive-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-bimap-factor-cadence") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasBiMAPFactorCadence) || cfg.atlasBiMAPFactorCadence == 0u)
			{
				err = "invalid --atlas-bimap-factor-cadence";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-pact-low-rank") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasPACTLowRank) || cfg.atlasPACTLowRank > 1u)
			{
				err = "invalid --atlas-pact-low-rank";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-pact-geometry-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasPACTGeometryScale) || cfg.atlasPACTGeometryScale < 0.0f)
			{
				err = "invalid --atlas-pact-geometry-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-pact-predictive-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasPACTPredictiveScale) || cfg.atlasPACTPredictiveScale < 0.0f)
			{
				err = "invalid --atlas-pact-predictive-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-pact-factor-cadence") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasPACTFactorCadence) || cfg.atlasPACTFactorCadence == 0u)
			{
				err = "invalid --atlas-pact-factor-cadence";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-pact-cost-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasPACTCostScale) || cfg.atlasPACTCostScale < 0.0f)
			{
				err = "invalid --atlas-pact-cost-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-pact-promote-threshold") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasPACTPromoteThreshold))
			{
				err = "invalid --atlas-pact-promote-threshold";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-pact-demote-threshold") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasPACTDemoteThreshold))
			{
				err = "invalid --atlas-pact-demote-threshold";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-racer-geometry-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasRACERGeometryScale) || cfg.atlasRACERGeometryScale < 0.0f)
			{
				err = "invalid --atlas-racer-geometry-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-racer-predictive-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasRACERPredictiveScale) || cfg.atlasRACERPredictiveScale < 0.0f)
			{
				err = "invalid --atlas-racer-predictive-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-racer-factor-cadence") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasRACERFactorCadence) || cfg.atlasRACERFactorCadence == 0u)
			{
				err = "invalid --atlas-racer-factor-cadence";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-racer-risk-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasRACERRiskScale) || cfg.atlasRACERRiskScale < 0.0f)
			{
				err = "invalid --atlas-racer-risk-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-racer-cost-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasRACERCostScale) || cfg.atlasRACERCostScale < 0.0f)
			{
				err = "invalid --atlas-racer-cost-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-racer-promote-threshold") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasRACERPromoteThreshold))
			{
				err = "invalid --atlas-racer-promote-threshold";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-racer-demote-threshold") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasRACERDemoteThreshold))
			{
				err = "invalid --atlas-racer-demote-threshold";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-kron-geometry-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasKronGeometryScale) || cfg.atlasKronGeometryScale < 0.0f)
			{
				err = "invalid --atlas-kron-geometry-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-kron-predictive-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasKronPredictiveScale) || cfg.atlasKronPredictiveScale < 0.0f)
			{
				err = "invalid --atlas-kron-predictive-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-kron-factor-cadence") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasKronFactorCadence) || cfg.atlasKronFactorCadence == 0u)
			{
				err = "invalid --atlas-kron-factor-cadence";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-kron-damping") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasKronDamping) || cfg.atlasKronDamping < 0.0f)
			{
				err = "invalid --atlas-kron-damping";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-muon-geometry-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasMuonGeometryScale) || cfg.atlasMuonGeometryScale < 0.0f)
			{
				err = "invalid --atlas-muon-geometry-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-muon-predictive-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasMuonPredictiveScale) || cfg.atlasMuonPredictiveScale < 0.0f)
			{
				err = "invalid --atlas-muon-predictive-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-muon-max-aspect") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasMuonMaxAspect) || cfg.atlasMuonMaxAspect < 1.0f)
			{
				err = "invalid --atlas-muon-max-aspect";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-muon-min-dim") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasMuonMinDim) || cfg.atlasMuonMinDim == 0u)
			{
				err = "invalid --atlas-muon-min-dim";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-muon-damping") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasMuonDamping) || cfg.atlasMuonDamping < 0.0f)
			{
				err = "invalid --atlas-muon-damping";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-matra-geometry-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasMatraGeometryScale) || cfg.atlasMatraGeometryScale < 0.0f)
			{
				err = "invalid --atlas-matra-geometry-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-matra-orthogonal-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasMatraOrthogonalScale) || cfg.atlasMatraOrthogonalScale < 0.0f)
			{
				err = "invalid --atlas-matra-orthogonal-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-matra-predictive-scale") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasMatraPredictiveScale) || cfg.atlasMatraPredictiveScale < 0.0f)
			{
				err = "invalid --atlas-matra-predictive-scale";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-matra-trust-radius") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasMatraTrustRadius) || cfg.atlasMatraTrustRadius < 0.0f)
			{
				err = "invalid --atlas-matra-trust-radius";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-matra-cadence") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasMatraMetricCadence) || cfg.atlasMatraMetricCadence == 0u)
			{
				err = "invalid --atlas-matra-cadence";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-matra-max-aspect") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasMatraMaxAspect) || cfg.atlasMatraMaxAspect < 1.0f)
			{
				err = "invalid --atlas-matra-max-aspect";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-matra-min-dim") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.atlasMatraMinDim) || cfg.atlasMatraMinDim == 0u)
			{
				err = "invalid --atlas-matra-min-dim";
				return false;
			}
		}
		else if (streq(argv[i], "--atlas-matra-damping") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.atlasMatraDamping) || cfg.atlasMatraDamping < 0.0f)
			{
				err = "invalid --atlas-matra-damping";
				return false;
			}
		}
		else if (streq(argv[i], "--gpu-enable") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.gpuEnable) || cfg.gpuEnable > 1u)
			{
				err = "invalid --gpu-enable";
				return false;
			}
		}
		else if (streq(argv[i], "--gpu-device") && i + 1 < argc)
		{
			unsigned int gpuDevice = 0u;
			if (!parse_uint_arg(argv[++i], gpuDevice))
			{
				err = "invalid --gpu-device";
				return false;
			}
			cfg.gpuDeviceId = static_cast<int>(gpuDevice);
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
		else if (streq(argv[i], "--token-seq-len") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.token.seqLen) || cfg.token.seqLen < 2u)
			{
				err = "invalid --token-seq-len";
				return false;
			}
		}
		else if (streq(argv[i], "--token-dmodel") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.token.dModel) || cfg.token.dModel == 0u)
			{
				err = "invalid --token-dmodel";
				return false;
			}
		}
		else if (streq(argv[i], "--token-dff") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.token.dFF) || cfg.token.dFF == 0u)
			{
				err = "invalid --token-dff";
				return false;
			}
		}
		else if (streq(argv[i], "--token-layers") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.token.layers) || cfg.token.layers == 0u)
			{
				err = "invalid --token-layers";
				return false;
			}
		}
		else if (streq(argv[i], "--token-heads") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.token.heads) || cfg.token.heads == 0u)
			{
				err = "invalid --token-heads";
				return false;
			}
			cfg.token.kvHeads = cfg.token.heads;
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

static unsigned int token_context_global_content(unsigned int contentBase,
                                                 unsigned int contentCount,
                                                 unsigned int value)
{
	return contentBase + (value % contentCount);
}

static unsigned int token_context_topic_content(unsigned int contentBase,
                                                unsigned int contentCount,
                                                unsigned int topic,
                                                unsigned int topicCount,
                                                unsigned int value)
{
	const unsigned int start = contentBase + ((topic * contentCount) / topicCount);
	const unsigned int end = contentBase + (((topic + 1u) * contentCount) / topicCount);
	const unsigned int width = (end > start) ? (end - start) : 1u;
	return start + (value % width);
}

static unsigned int token_pick_from_range(unsigned int base,
                                          unsigned int count,
                                          unsigned int value)
{
	if (count == 0u)
		return base;
	return base + (value % count);
}

static void build_token_context_split(unsigned int vocab,
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

	static const unsigned int kTopicCount = 4u;
	static const unsigned int kTopicBase = 0u;
	static const unsigned int kQuerySummary = 4u;
	static const unsigned int kQueryAnchor = 5u;
	static const unsigned int kMix = 6u;
	static const unsigned int kLocal = 7u;
	static const unsigned int kStore = 8u;
	static const unsigned int kSep = 9u;
	static const unsigned int kReservedCount = 10u;
	static const unsigned int kSegmentLen = 14u;

	if (vocab <= (kReservedCount + 1u))
	{
		build_token_split(vocab, seqCount, seqLen, seed, padTokenId, outTokens, outSpans);
		return;
	}

	const unsigned int contentBase = kReservedCount;
	const unsigned int contentCount = vocab - kReservedCount - 1u;

	for (unsigned int seq = 0u; seq < seqCount; ++seq)
	{
		const unsigned int start = static_cast<unsigned int>(outTokens.size());
		const unsigned int topic = mix_u32(seed + 41u * (seq + 1u)) % kTopicCount;
		unsigned int prevSummary = token_context_topic_content(contentBase, contentCount, topic, kTopicCount,
		                                                      mix_u32(seed + 73u * (seq + 3u)));
		unsigned int olderSummary = token_context_topic_content(contentBase, contentCount, topic, kTopicCount,
		                                                       mix_u32(seed + 109u * (seq + 5u)));
		unsigned int prevAnchor = token_context_topic_content(contentBase, contentCount, topic, kTopicCount,
		                                                     mix_u32(seed + 149u * (seq + 7u)));
		unsigned int olderAnchor = token_context_topic_content(contentBase, contentCount, topic, kTopicCount,
		                                                      mix_u32(seed + 197u * (seq + 11u)));

		for (unsigned int t = 0u; t < seqLen; ++t)
		{
			const unsigned int segment = t / kSegmentLen;
			const unsigned int slot = t % kSegmentLen;
			const unsigned int noiseA = mix_u32(seed + 911u * (seq + 1u) + 37u * (segment + 1u));
			const unsigned int noiseB = mix_u32(seed + 1237u * (seq + 3u) + 53u * (segment + 5u));
			const unsigned int summaryQuery = ((segment % 3u) == 2u) ? olderSummary : prevSummary;
			const unsigned int anchorQuery = ((segment % 2u) == 1u) ? olderAnchor : prevAnchor;
			const unsigned int a = token_context_topic_content(contentBase, contentCount, topic, kTopicCount,
			                                                  noiseA + prevAnchor + 7u * segment);
			const unsigned int b = token_context_topic_content(contentBase, contentCount, topic, kTopicCount,
			                                                  noiseB + prevSummary + 11u * segment);
			const unsigned int newSummary =
			    token_context_global_content(contentBase, contentCount,
			                                (a * 3u) + (b * 5u) + (topic * 17u) + (segment * 19u) + prevSummary);
			const unsigned int newLocal =
			    token_context_global_content(contentBase, contentCount,
			                                (newSummary * 7u) + (b * 3u) + (anchorQuery * 5u) + (segment * 23u));
			const unsigned int newAnchor =
			    token_context_topic_content(contentBase, contentCount, topic, kTopicCount,
			                               (a * 11u) + (prevAnchor * 3u) + (segment * 5u) + noiseB);

			unsigned int tok = 0u;
			switch (slot)
			{
				case 0u: tok = kTopicBase + topic; break;
				case 1u: tok = kQuerySummary; break;
				case 2u: tok = summaryQuery; break;
				case 3u: tok = kQueryAnchor; break;
				case 4u: tok = anchorQuery; break;
				case 5u: tok = a; break;
				case 6u: tok = b; break;
				case 7u: tok = kMix; break;
				case 8u: tok = newSummary; break;
				case 9u: tok = kLocal; break;
				case 10u: tok = newLocal; break;
				case 11u: tok = kStore; break;
				case 12u: tok = newAnchor; break;
				default: tok = kSep; break;
			}
			outTokens.push_back(tok);

			if (slot == (kSegmentLen - 1u))
			{
				olderSummary = prevSummary;
				prevSummary = newSummary;
				olderAnchor = prevAnchor;
				prevAnchor = newAnchor;
			}
		}

		outSpans.push_back(glades::DataInput::SequenceSpan(start, seqLen));
		outTokens.push_back(padTokenId);
	}
}

static void build_token_document_split(unsigned int vocab,
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

	static const unsigned int kDoc = 0u;
	static const unsigned int kHead = 1u;
	static const unsigned int kBy = 2u;
	static const unsigned int kIn = 3u;
	static const unsigned int kOn = 4u;
	static const unsigned int kLead = 5u;
	static const unsigned int kQuote = 6u;
	static const unsigned int kSays = 7u;
	static const unsigned int kAbout = 8u;
	static const unsigned int kWith = 9u;
	static const unsigned int kAfter = 10u;
	static const unsigned int kBefore = 11u;
	static const unsigned int kRecall = 12u;
	static const unsigned int kSummary = 13u;
	static const unsigned int kContinue = 14u;
	static const unsigned int kSep = 15u;
	static const unsigned int kStructureCount = 16u;
	static const unsigned int kTopicCount = 8u;
	static const unsigned int kTopicBase = kStructureCount;
	static const unsigned int kReservedCount = kStructureCount + kTopicCount;
	static const unsigned int kParagraphLen = 24u;
	static const unsigned int kMinPoolWidth = 8u;

	if (vocab <= (kReservedCount + 6u * kMinPoolWidth + 1u))
	{
		build_token_context_split(vocab, seqCount, seqLen, seed, padTokenId, outTokens, outSpans);
		return;
	}

	const unsigned int contentBase = kReservedCount;
	const unsigned int contentCount = vocab - kReservedCount - 1u;
	unsigned int remaining = contentCount;
	const unsigned int entityCount =
	    std::min(remaining - 5u * kMinPoolWidth, std::max(kMinPoolWidth, contentCount / 5u));
	remaining -= entityCount;
	const unsigned int placeCount =
	    std::min(remaining - 4u * kMinPoolWidth, std::max(kMinPoolWidth, contentCount / 7u));
	remaining -= placeCount;
	const unsigned int yearCount =
	    std::min(remaining - 3u * kMinPoolWidth, std::max(kMinPoolWidth, contentCount / 10u));
	remaining -= yearCount;
	const unsigned int verbCount =
	    std::min(remaining - 2u * kMinPoolWidth, std::max(kMinPoolWidth, contentCount / 7u));
	remaining -= verbCount;
	const unsigned int objectCount =
	    std::min(remaining - 1u * kMinPoolWidth, std::max(kMinPoolWidth, contentCount / 5u));
	remaining -= objectCount;
	const unsigned int detailCount = remaining;

	const unsigned int entityBase = contentBase;
	const unsigned int placeBase = entityBase + entityCount;
	const unsigned int yearBase = placeBase + placeCount;
	const unsigned int verbBase = yearBase + yearCount;
	const unsigned int objectBase = verbBase + verbCount;
	const unsigned int detailBase = objectBase + objectCount;

	for (unsigned int seq = 0u; seq < seqCount; ++seq)
	{
		const unsigned int start = static_cast<unsigned int>(outTokens.size());
		const unsigned int topic = mix_u32(seed + 41u * (seq + 1u)) % kTopicCount;
		const unsigned int topicTok = kTopicBase + topic;
		const unsigned int leadEntity =
		    token_pick_from_range(entityBase, entityCount, mix_u32(seed + 73u * (seq + 3u)));
		const unsigned int authorEntity =
		    token_pick_from_range(entityBase, entityCount, mix_u32(seed + 109u * (seq + 5u)));
		const unsigned int openingPlace =
		    token_pick_from_range(placeBase, placeCount, mix_u32(seed + 149u * (seq + 7u)));
		const unsigned int openingYear =
		    token_pick_from_range(yearBase, yearCount, mix_u32(seed + 197u * (seq + 11u)));

		unsigned int prevSummary =
		    token_context_topic_content(detailBase, detailCount, topic, kTopicCount,
		                               mix_u32(seed + 223u * (seq + 13u)));
		unsigned int olderSummary =
		    token_context_topic_content(detailBase, detailCount, topic, kTopicCount,
		                               mix_u32(seed + 257u * (seq + 17u)));
		unsigned int prevObject =
		    token_pick_from_range(objectBase, objectCount, mix_u32(seed + 307u * (seq + 19u)));
		unsigned int olderObject =
		    token_pick_from_range(objectBase, objectCount, mix_u32(seed + 353u * (seq + 23u)));
		unsigned int prevPlace = openingPlace;
		unsigned int olderPlace =
		    token_pick_from_range(placeBase, placeCount, mix_u32(seed + 401u * (seq + 29u)));
		unsigned int prevYear = openingYear;
		unsigned int olderYear =
		    token_pick_from_range(yearBase, yearCount, mix_u32(seed + 443u * (seq + 31u)));
		unsigned int prevSpeaker = authorEntity;
		unsigned int olderSpeaker = leadEntity;

		const unsigned int paragraphCount = (seqLen + kParagraphLen - 1u) / kParagraphLen;
		for (unsigned int paragraph = 0u; paragraph < paragraphCount; ++paragraph)
		{
			unsigned int paragraphTokens[kParagraphLen];
			for (unsigned int i = 0; i < kParagraphLen; ++i)
				paragraphTokens[i] = kSep;

			const unsigned int recallSummary = ((paragraph % 3u) == 2u) ? olderSummary : prevSummary;
			const unsigned int recallObject = ((paragraph % 2u) == 1u) ? olderObject : prevObject;
			const unsigned int recallPlace = ((paragraph % 2u) == 1u) ? olderPlace : prevPlace;
			const unsigned int recallYear = ((paragraph % 3u) == 1u) ? olderYear : prevYear;
			const unsigned int recallSpeaker = ((paragraph % 2u) == 1u) ? olderSpeaker : prevSpeaker;
			const unsigned int entityA =
			    token_pick_from_range(entityBase, entityCount,
			                          mix_u32(seed + 601u * (seq + 1u) + 31u * (paragraph + 1u)));
			const unsigned int entityB =
			    token_pick_from_range(entityBase, entityCount,
			                          mix_u32(seed + 647u * (seq + 3u) + 37u * (paragraph + 5u)));
			const unsigned int verbA =
			    token_pick_from_range(verbBase, verbCount,
			                          mix_u32(seed + 691u * (seq + 7u) + 41u * (paragraph + 9u)));
			const unsigned int verbB =
			    token_pick_from_range(verbBase, verbCount,
			                          mix_u32(seed + 743u * (seq + 11u) + 43u * (paragraph + 13u)));
			const unsigned int objectA =
			    token_pick_from_range(objectBase, objectCount,
			                          mix_u32(seed + 797u * (seq + 13u) + 47u * (paragraph + 17u)));
			const unsigned int objectB =
			    token_pick_from_range(objectBase, objectCount,
			                          mix_u32(seed + 853u * (seq + 17u) + 53u * (paragraph + 19u)));
			const unsigned int placeA =
			    token_pick_from_range(placeBase, placeCount,
			                          mix_u32(seed + 911u * (seq + 19u) + 59u * (paragraph + 23u)));
			const unsigned int yearA =
			    token_pick_from_range(yearBase, yearCount,
			                          mix_u32(seed + 977u * (seq + 23u) + 61u * (paragraph + 29u)));
			const unsigned int detailA =
			    token_context_topic_content(detailBase, detailCount, topic, kTopicCount,
			                               mix_u32(seed + 1031u * (seq + 29u) + 67u * (paragraph + 31u)));
			const unsigned int detailB =
			    token_context_global_content(detailBase, detailCount,
			                                 mix_u32(seed + 1087u * (seq + 31u) + 71u * (paragraph + 37u))
			                                     + recallSummary + recallObject);
			const unsigned int detailC =
			    token_context_topic_content(detailBase, detailCount, topic, kTopicCount,
			                               mix_u32(seed + 1151u * (seq + 37u) + 73u * (paragraph + 41u)));
			const unsigned int detailD =
			    token_context_global_content(detailBase, detailCount,
			                                 mix_u32(seed + 1229u * (seq + 41u) + 79u * (paragraph + 43u))
			                                     + recallPlace + recallYear);
			const unsigned int newSummary =
			    token_context_global_content(detailBase, detailCount,
			                                 (entityA * 3u) + (objectA * 5u) + (recallSummary * 7u)
			                                     + (paragraph * 19u) + detailA + detailC);
			const unsigned int newObject =
			    token_pick_from_range(objectBase, objectCount,
			                          (objectB * 11u) + (detailB * 3u) + (paragraph * 17u) + recallObject);
			const unsigned int newPlace =
			    token_pick_from_range(placeBase, placeCount,
			                          (placeA * 7u) + (detailC * 5u) + (paragraph * 13u) + recallPlace);
			const unsigned int newYear =
			    token_pick_from_range(yearBase, yearCount,
			                          (yearA * 5u) + (objectA * 7u) + (paragraph * 11u) + recallYear);
			const unsigned int newSpeaker = ((paragraph % 2u) == 0u) ? entityB : entityA;

			switch (paragraph % 4u)
			{
				case 0u:
					paragraphTokens[0] = kDoc;
					paragraphTokens[1] = topicTok;
					paragraphTokens[2] = kHead;
					paragraphTokens[3] = leadEntity;
					paragraphTokens[4] = kBy;
					paragraphTokens[5] = authorEntity;
					paragraphTokens[6] = kIn;
					paragraphTokens[7] = recallPlace;
					paragraphTokens[8] = kOn;
					paragraphTokens[9] = recallYear;
					paragraphTokens[10] = kLead;
					paragraphTokens[11] = leadEntity;
					paragraphTokens[12] = verbA;
					paragraphTokens[13] = objectA;
					paragraphTokens[14] = kWith;
					paragraphTokens[15] = detailA;
					paragraphTokens[16] = detailB;
					paragraphTokens[17] = kSummary;
					paragraphTokens[18] = recallSummary;
					paragraphTokens[19] = kContinue;
					paragraphTokens[20] = newSummary;
					paragraphTokens[21] = newObject;
					paragraphTokens[22] = newPlace;
					paragraphTokens[23] = kSep;
					break;
				case 1u:
					paragraphTokens[0] = topicTok;
					paragraphTokens[1] = entityA;
					paragraphTokens[2] = verbA;
					paragraphTokens[3] = objectA;
					paragraphTokens[4] = kIn;
					paragraphTokens[5] = placeA;
					paragraphTokens[6] = kAfter;
					paragraphTokens[7] = yearA;
					paragraphTokens[8] = detailA;
					paragraphTokens[9] = entityB;
					paragraphTokens[10] = verbB;
					paragraphTokens[11] = objectB;
					paragraphTokens[12] = kWith;
					paragraphTokens[13] = recallObject;
					paragraphTokens[14] = kAbout;
					paragraphTokens[15] = recallSummary;
					paragraphTokens[16] = kSummary;
					paragraphTokens[17] = newSummary;
					paragraphTokens[18] = kContinue;
					paragraphTokens[19] = detailC;
					paragraphTokens[20] = newObject;
					paragraphTokens[21] = newPlace;
					paragraphTokens[22] = newYear;
					paragraphTokens[23] = kSep;
					break;
				case 2u:
					paragraphTokens[0] = kQuote;
					paragraphTokens[1] = recallSpeaker;
					paragraphTokens[2] = kSays;
					paragraphTokens[3] = leadEntity;
					paragraphTokens[4] = verbB;
					paragraphTokens[5] = objectB;
					paragraphTokens[6] = kAbout;
					paragraphTokens[7] = topicTok;
					paragraphTokens[8] = kWith;
					paragraphTokens[9] = detailA;
					paragraphTokens[10] = detailD;
					paragraphTokens[11] = kIn;
					paragraphTokens[12] = recallPlace;
					paragraphTokens[13] = kOn;
					paragraphTokens[14] = recallYear;
					paragraphTokens[15] = kSummary;
					paragraphTokens[16] = recallSummary;
					paragraphTokens[17] = kContinue;
					paragraphTokens[18] = newSummary;
					paragraphTokens[19] = newSpeaker;
					paragraphTokens[20] = newObject;
					paragraphTokens[21] = newPlace;
					paragraphTokens[22] = newYear;
					paragraphTokens[23] = kSep;
					break;
				default:
					paragraphTokens[0] = kRecall;
					paragraphTokens[1] = recallSpeaker;
					paragraphTokens[2] = recallSummary;
					paragraphTokens[3] = kIn;
					paragraphTokens[4] = recallPlace;
					paragraphTokens[5] = kOn;
					paragraphTokens[6] = recallYear;
					paragraphTokens[7] = leadEntity;
					paragraphTokens[8] = verbA;
					paragraphTokens[9] = recallObject;
					paragraphTokens[10] = kBefore;
					paragraphTokens[11] = olderYear;
					paragraphTokens[12] = kWith;
					paragraphTokens[13] = detailB;
					paragraphTokens[14] = detailC;
					paragraphTokens[15] = kSummary;
					paragraphTokens[16] = newSummary;
					paragraphTokens[17] = kContinue;
					paragraphTokens[18] = newObject;
					paragraphTokens[19] = newPlace;
					paragraphTokens[20] = newYear;
					paragraphTokens[21] = entityB;
					paragraphTokens[22] = detailD;
					paragraphTokens[23] = kSep;
					break;
			}

			const unsigned int startToken = paragraph * kParagraphLen;
			const unsigned int limit = std::min(kParagraphLen, seqLen - startToken);
			for (unsigned int i = 0u; i < limit; ++i)
				outTokens.push_back(paragraphTokens[i]);

			olderSummary = prevSummary;
			prevSummary = newSummary;
			olderObject = prevObject;
			prevObject = newObject;
			olderPlace = prevPlace;
			prevPlace = newPlace;
			olderYear = prevYear;
			prevYear = newYear;
			olderSpeaker = prevSpeaker;
			prevSpeaker = newSpeaker;
		}

		outSpans.push_back(glades::DataInput::SequenceSpan(start, seqLen));
		outTokens.push_back(padTokenId);
	}
}

static unsigned int corpus_hash_word(const std::string& word)
{
	unsigned int h = 2166136261u;
	for (size_t i = 0; i < word.size(); ++i)
	{
		h ^= static_cast<unsigned char>(word[i]);
		h *= 16777619u;
	}
	return mix_u32(h);
}

static unsigned int corpus_intern_word(const std::string& word,
                                       unsigned int padTokenId,
                                       std::map<std::string, unsigned int>& lexicon,
                                       unsigned int& nextId)
{
	const std::map<std::string, unsigned int>::const_iterator it = lexicon.find(word);
	if (it != lexicon.end())
		return it->second;

	unsigned int tokenId = 0u;
	if (nextId < padTokenId)
		tokenId = nextId++;
	else
	{
		const unsigned int base = 3u;
		const unsigned int width = (padTokenId > base) ? (padTokenId - base) : 1u;
		tokenId = base + (corpus_hash_word(word) % width);
	}
	lexicon.insert(std::make_pair(word, tokenId));
	return tokenId;
}

static void corpus_push_boundary(unsigned int tokenId, std::vector<unsigned int>& stream)
{
	if (stream.empty() || stream.back() != tokenId)
		stream.push_back(tokenId);
}

static void append_corpus_docs(const char* const* docs,
                               size_t docCount,
                               unsigned int padTokenId,
                               std::map<std::string, unsigned int>& lexicon,
                               unsigned int& nextId,
                               std::vector<unsigned int>& outStream)
{
	static const unsigned int kDocToken = 0u;
	static const unsigned int kParaToken = 1u;
	static const unsigned int kEosToken = 2u;

	for (size_t doc = 0; doc < docCount; ++doc)
	{
		corpus_push_boundary(kDocToken, outStream);
		const char* text = docs[doc];
		std::string word;
		word.reserve(24u);
		for (size_t i = 0; text[i] != '\0'; ++i)
		{
			const unsigned char uc = static_cast<unsigned char>(text[i]);
			const char c = static_cast<char>(uc);
			const bool alphaNum = ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
			                    || (c >= '0' && c <= '9') || c == '\'');
			if (alphaNum)
			{
				word.push_back((c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c);
				continue;
			}
			if (!word.empty())
			{
				outStream.push_back(corpus_intern_word(word, padTokenId, lexicon, nextId));
				word.clear();
			}
			if (c == '.' || c == '!' || c == '?' || c == ';' || c == ':')
				corpus_push_boundary(kEosToken, outStream);
			else if (c == '\n')
				corpus_push_boundary(kParaToken, outStream);
		}
		if (!word.empty())
			outStream.push_back(corpus_intern_word(word, padTokenId, lexicon, nextId));
		corpus_push_boundary(kParaToken, outStream);
	}
}

static void build_token_windows_from_stream(const std::vector<unsigned int>& stream,
                                            unsigned int seqCount,
                                            unsigned int seqLen,
                                            unsigned int seed,
                                            unsigned int padTokenId,
                                            std::vector<unsigned int>& outTokens,
                                            std::vector<glades::DataInput::SequenceSpan>& outSpans)
{
	outTokens.clear();
	outSpans.clear();
	if (seqCount == 0u || seqLen == 0u)
		return;

	std::vector<unsigned int> expanded = stream;
	if (expanded.empty())
		expanded.push_back(0u);
	while (expanded.size() < static_cast<size_t>(seqLen))
		expanded.insert(expanded.end(), stream.begin(), stream.end());

	const unsigned int maxStart =
	    (expanded.size() > static_cast<size_t>(seqLen))
	        ? static_cast<unsigned int>(expanded.size() - static_cast<size_t>(seqLen))
	        : 0u;
	const unsigned int stride = std::max(1u, (seqLen / 3u) + 7u);
	const unsigned int jitterSpan = std::max(1u, (seqLen / 4u) + 1u);
	const unsigned int baseStart =
	    (maxStart > 0u) ? (mix_u32(seed + 0x6a09e667U) % (maxStart + 1u)) : 0u;

	outTokens.reserve(static_cast<size_t>(seqCount) * static_cast<size_t>(seqLen + 1u));
	outSpans.reserve(seqCount);
	for (unsigned int seq = 0u; seq < seqCount; ++seq)
	{
		const unsigned int jitter = mix_u32(seed + 0x9e3779b9U * (seq + 1u)) % jitterSpan;
		const unsigned int start =
		    (maxStart > 0u) ? ((baseStart + seq * stride + jitter) % (maxStart + 1u)) : 0u;
		const unsigned int spanStart = static_cast<unsigned int>(outTokens.size());
		for (unsigned int i = 0; i < seqLen; ++i)
			outTokens.push_back(expanded[static_cast<size_t>(start + i)]);
		outSpans.push_back(glades::DataInput::SequenceSpan(spanStart, seqLen));
		outTokens.push_back(padTokenId);
	}
}

static void build_token_corpus_streams(bool largeCorpus,
                                       unsigned int padTokenId,
                                       std::vector<unsigned int>& trainStream,
                                       std::vector<unsigned int>& testStream)
{
	static const char* kTrainDocs[] = {
	    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it. What is the use of a book, thought Alice, without pictures or conversation? So she was considering in her own mind whether the pleasure of making a daisy chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.",
	    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.",
	    "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",
	    "The story of the house is printed in a little book. It tells of an old red brick farmhouse, standing in a rich pasture country, and of a road that ran before the door and passed away among ancient trees. The house looked across wide meadows, and behind it there were orchards, hedges, and a brook that moved slowly under the willows.",
	    "The Time Traveller had finally finished the tale of his machine, and we sat in the yellow lamplight looking from him to the fire. His model stood on the table near the lamp, a little thing of ivory and shining metal, while the larger apparatus waited in the laboratory beyond the smoking room. We asked questions, and he answered them absently, as if he were still watching some remote horizon.",
	    "To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind." };

	static const char* kTestDocs[] = {
	    "When I was a child my mother used to tell me stories of voyages by sea, and of harbours where the masts of ships stood thick as leafless trees in winter. I remembered the smell of tar and salt, the clatter of boots upon the quay, and the ringing cry of the tide among the stones. Those memories returned to me many years later when I first set foot upon a windswept pier at dusk.",
	    "No one would have believed in the last years of the nineteenth century that this world was being watched keenly and closely by intelligences greater than man's. Yet across the gulf of space minds that are to our minds as ours are to the beasts that perish regarded this earth with envious eyes, and slowly and surely drew their plans against us.",
	    "3 May. Bistritz. Left Munich at eight thirty five in the evening on first May, arriving at Vienna early next morning. Budapest seems a wonderful place, from the glimpse which I got of it from the train and the little I could walk through the streets. I feared to go very far from the station, as we had arrived late and would start as near the correct time as possible." };

	static const char* kTrainDocsExtra[] = {
	    "There was no possibility of taking a walk that day. We had been wandering, indeed, in the leafless shrubbery an hour in the morning, but since dinner the cold winter wind had brought with it clouds so sombre and a rain so penetrating that further out-door exercise was now out of the question.",
	    "I remember him well, standing in the doorway with the lamp behind him and the wind lifting the edges of his cloak. He had the air of a man who had seen many roads and trusted none of them, yet his voice when he spoke of the valley and the mills was quiet and exact, as if he carried an entire map of the district in his mind.",
	    "My father's family name being Pirrip, and my Christian name Philip, my infant tongue could make of both names nothing longer or more explicit than Pip. So I called myself Pip, and came to be called Pip. I give Pirrip as my father's family name, on the authority of his tombstone and my sister Mrs Joe Gargery, who married the blacksmith.",
	    "The blackness of darkness fell from the air and the rain beat in gusts against the panes. Far down the lane there was a lantern moving, dipping and rising as the bearer made his slow way between hedges that shone with wet. In the kitchen the clock ticked loudly, and every sound in the house seemed to wait upon the knock that had not yet come." };

	static const char* kTestDocsExtra[] = {
	    "In the centre of the room there was a table spread with papers, charts, and a small globe stained by years of handling. The windows looked over a narrow court where rainwater gathered in the ruts, and beyond the court rose a wall of warehouses whose blank brick fronts caught the last grey of evening.",
	    "The garden lay still under the first light of morning, and only the birds had begun their business. Paths of damp gravel wound between the beds, and every leaf carried a bead of water that flashed briefly and was gone. She paused at the gate because the place seemed older than memory and yet freshly made before her eyes." };

	trainStream.clear();
	testStream.clear();
	std::map<std::string, unsigned int> lexicon;
	lexicon.insert(std::make_pair(std::string("<doc>"), 0u));
	lexicon.insert(std::make_pair(std::string("<para>"), 1u));
	lexicon.insert(std::make_pair(std::string("<eos>"), 2u));
	unsigned int nextId = 3u;
	append_corpus_docs(kTrainDocs, sizeof(kTrainDocs) / sizeof(kTrainDocs[0]),
	                   padTokenId, lexicon, nextId, trainStream);
	if (largeCorpus)
	{
		append_corpus_docs(kTrainDocsExtra, sizeof(kTrainDocsExtra) / sizeof(kTrainDocsExtra[0]),
		                   padTokenId, lexicon, nextId, trainStream);
	}
	append_corpus_docs(kTestDocs, sizeof(kTestDocs) / sizeof(kTestDocs[0]),
	                   padTokenId, lexicon, nextId, testStream);
	if (largeCorpus)
	{
		append_corpus_docs(kTestDocsExtra, sizeof(kTestDocsExtra) / sizeof(kTestDocsExtra[0]),
		                   padTokenId, lexicon, nextId, testStream);
	}
}

static void clear_context_family_diag(TokenDataset& out)
{
	out.testRoleBuckets.clear();
	out.testRecallDistanceBuckets.clear();
	out.testRecallSubtypeBuckets.clear();
}

static void set_context_family_diag(std::vector<unsigned char>& roles,
                                    std::vector<unsigned char>& distances,
                                    std::vector<unsigned char>& subtypes,
                                    size_t index,
                                    unsigned char role,
                                    unsigned char distance,
                                    unsigned char subtype)
{
	if (index >= roles.size() || index >= distances.size() || index >= subtypes.size())
		return;
	roles[index] = role;
	distances[index] = distance;
	subtypes[index] = subtype;
}

static void annotate_context_family_context_split(const std::vector<glades::DataInput::SequenceSpan>& spans,
                                                  std::vector<unsigned char>& roles,
                                                  std::vector<unsigned char>& distances,
                                                  std::vector<unsigned char>& subtypes)
{
	static const unsigned int kSegmentLen = 14u;
	for (size_t s = 0u; s < spans.size(); ++s)
	{
		const glades::DataInput::SequenceSpan& span = spans[s];
		for (unsigned int t = 0u; t < span.length; ++t)
		{
			const size_t idx = static_cast<size_t>(span.start + t);
			const unsigned int segment = t / kSegmentLen;
			const unsigned int slot = t % kSegmentLen;
			switch (slot)
			{
				case 0u:
					set_context_family_diag(roles, distances, subtypes, idx,
					                       CTX_ROLE_TOPIC, kContextRecallDistanceNone, kContextRecallSubtypeNone);
					break;
				case 1u:
				case 3u:
					set_context_family_diag(roles, distances, subtypes, idx,
					                       CTX_ROLE_QUERY, kContextRecallDistanceNone, kContextRecallSubtypeNone);
					break;
				case 2u:
					set_context_family_diag(roles, distances, subtypes, idx,
					                       CTX_ROLE_RECALL,
					                       ((segment % 3u) == 2u) ? CTX_RECALL_OLDER : CTX_RECALL_PREV,
					                       CTX_SUB_SUMMARY);
					break;
				case 4u:
					set_context_family_diag(roles, distances, subtypes, idx,
					                       CTX_ROLE_RECALL,
					                       ((segment % 2u) == 1u) ? CTX_RECALL_OLDER : CTX_RECALL_PREV,
					                       CTX_SUB_ANCHOR);
					break;
				case 5u:
				case 6u:
					set_context_family_diag(roles, distances, subtypes, idx,
					                       CTX_ROLE_CONTENT, kContextRecallDistanceNone, kContextRecallSubtypeNone);
					break;
				case 8u:
				case 10u:
				case 12u:
					set_context_family_diag(roles, distances, subtypes, idx,
					                       CTX_ROLE_STATE, kContextRecallDistanceNone, kContextRecallSubtypeNone);
					break;
				case 13u:
					set_context_family_diag(roles, distances, subtypes, idx,
					                       CTX_ROLE_SEPARATOR, kContextRecallDistanceNone, kContextRecallSubtypeNone);
					break;
				default:
					set_context_family_diag(roles, distances, subtypes, idx,
					                       CTX_ROLE_MARKER, kContextRecallDistanceNone, kContextRecallSubtypeNone);
					break;
			}
		}
	}
}

static void annotate_context_family_document_split(const std::vector<glades::DataInput::SequenceSpan>& spans,
                                                   std::vector<unsigned char>& roles,
                                                   std::vector<unsigned char>& distances,
                                                   std::vector<unsigned char>& subtypes)
{
	static const unsigned int kParagraphLen = 24u;
	for (size_t s = 0u; s < spans.size(); ++s)
	{
		const glades::DataInput::SequenceSpan& span = spans[s];
		for (unsigned int t = 0u; t < span.length; ++t)
		{
			const size_t idx = static_cast<size_t>(span.start + t);
			const unsigned int paragraph = t / kParagraphLen;
			const unsigned int slot = t % kParagraphLen;
			const unsigned char prevDist = CTX_RECALL_PREV;
			const unsigned char olderDist = CTX_RECALL_OLDER;
			switch (paragraph % 4u)
			{
				case 0u:
					switch (slot)
					{
						case 1u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_TOPIC, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 7u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, prevDist, CTX_SUB_PLACE);
							break;
						case 9u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, prevDist, CTX_SUB_YEAR);
							break;
						case 18u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, prevDist, CTX_SUB_SUMMARY);
							break;
						case 20u:
						case 21u:
						case 22u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_STATE, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 23u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_SEPARATOR, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 0u:
						case 2u:
						case 4u:
						case 6u:
						case 8u:
						case 10u:
						case 14u:
						case 17u:
						case 19u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_MARKER, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						default:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_CONTENT, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
					}
					break;
				case 1u:
					switch (slot)
					{
						case 0u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_TOPIC, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 13u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, olderDist, CTX_SUB_OBJECT);
							break;
						case 15u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, prevDist, CTX_SUB_SUMMARY);
							break;
						case 17u:
						case 20u:
						case 21u:
						case 22u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_STATE, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 23u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_SEPARATOR, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 4u:
						case 6u:
						case 12u:
						case 14u:
						case 16u:
						case 18u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_MARKER, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						default:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_CONTENT, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
					}
					break;
				case 2u:
					switch (slot)
					{
						case 7u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_TOPIC, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 1u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, prevDist, CTX_SUB_SPEAKER);
							break;
						case 12u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, prevDist, CTX_SUB_PLACE);
							break;
						case 14u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, prevDist, CTX_SUB_YEAR);
							break;
						case 16u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, olderDist, CTX_SUB_SUMMARY);
							break;
						case 18u:
						case 19u:
						case 20u:
						case 21u:
						case 22u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_STATE, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 23u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_SEPARATOR, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 0u:
						case 2u:
						case 6u:
						case 8u:
						case 11u:
						case 13u:
						case 15u:
						case 17u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_MARKER, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						default:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_CONTENT, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
					}
					break;
				default:
					switch (slot)
					{
						case 1u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, olderDist, CTX_SUB_SPEAKER);
							break;
						case 2u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, prevDist, CTX_SUB_SUMMARY);
							break;
						case 4u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, olderDist, CTX_SUB_PLACE);
							break;
						case 6u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, prevDist, CTX_SUB_YEAR);
							break;
						case 9u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, olderDist, CTX_SUB_OBJECT);
							break;
						case 11u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_RECALL, olderDist, CTX_SUB_YEAR);
							break;
						case 16u:
						case 18u:
						case 19u:
						case 20u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_STATE, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 23u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_SEPARATOR, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						case 0u:
						case 3u:
						case 5u:
						case 10u:
						case 12u:
						case 15u:
						case 17u:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_MARKER, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
						default:
							set_context_family_diag(roles, distances, subtypes, idx, CTX_ROLE_CONTENT, kContextRecallDistanceNone, kContextRecallSubtypeNone);
							break;
					}
					break;
			}
		}
	}
}

static void annotate_context_family_test_split(BenchMode mode, TokenDataset& out)
{
	clear_context_family_diag(out);
	if (out.testTokenStream.empty() || out.testSpans.empty())
		return;
	out.testRoleBuckets.assign(out.testTokenStream.size(), static_cast<unsigned char>(CTX_ROLE_OTHER));
	out.testRecallDistanceBuckets.assign(out.testTokenStream.size(), kContextRecallDistanceNone);
	out.testRecallSubtypeBuckets.assign(out.testTokenStream.size(), kContextRecallSubtypeNone);
	if (mode == MODE_TOKEN_LM_CONTEXT || mode == MODE_TOKEN_LM_CONTEXT_LARGE)
	{
		annotate_context_family_context_split(out.testSpans, out.testRoleBuckets,
		                                      out.testRecallDistanceBuckets, out.testRecallSubtypeBuckets);
	}
	else if (mode == MODE_TOKEN_LM_DOCUMENT)
	{
		annotate_context_family_document_split(out.testSpans, out.testRoleBuckets,
		                                       out.testRecallDistanceBuckets, out.testRecallSubtypeBuckets);
	}
	else
	{
		clear_context_family_diag(out);
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

	if (cfg.mode == MODE_TOKEN_LM_CONTEXT || cfg.mode == MODE_TOKEN_LM_CONTEXT_LARGE)
	{
		build_token_context_split(cfg.token.vocab, cfg.token.trainSeqs, cfg.token.seqLen, cfg.seed + 11u,
		                          out.padTokenId, trainTokens, trainSpans);
		build_token_context_split(cfg.token.vocab, cfg.token.testSeqs, cfg.token.seqLen, cfg.seed + 1011u,
		                          out.padTokenId, testTokens, testSpans);
	}
	else if (cfg.mode == MODE_TOKEN_LM_DOCUMENT)
	{
		build_token_document_split(cfg.token.vocab, cfg.token.trainSeqs, cfg.token.seqLen, cfg.seed + 211u,
		                           out.padTokenId, trainTokens, trainSpans);
		build_token_document_split(cfg.token.vocab, cfg.token.testSeqs, cfg.token.seqLen, cfg.seed + 1211u,
		                           out.padTokenId, testTokens, testSpans);
	}
	else if (cfg.mode == MODE_TOKEN_LM_CORPUS || cfg.mode == MODE_TOKEN_LM_CORPUS_LARGE)
	{
		std::vector<unsigned int> trainStream;
		std::vector<unsigned int> testStream;
		const bool largeCorpus = (cfg.mode == MODE_TOKEN_LM_CORPUS_LARGE);
		build_token_corpus_streams(largeCorpus, out.padTokenId, trainStream, testStream);
		build_token_windows_from_stream(trainStream, cfg.token.trainSeqs, cfg.token.seqLen,
		                                cfg.seed + 311u, out.padTokenId, trainTokens, trainSpans);
		build_token_windows_from_stream(testStream, cfg.token.testSeqs, cfg.token.seqLen,
		                                cfg.seed + 1311u, out.padTokenId, testTokens, testSpans);
	}
	else
	{
		build_token_split(cfg.token.vocab, cfg.token.trainSeqs, cfg.token.seqLen, cfg.seed + 11u,
		                  out.padTokenId, trainTokens, trainSpans);
		build_token_split(cfg.token.vocab, cfg.token.testSeqs, cfg.token.seqLen, cfg.seed + 1011u,
		                  out.padTokenId, testTokens, testSpans);
	}

	out.di.setTrainTokens(trainTokens, static_cast<int>(out.padTokenId));
	out.di.setTestTokens(testTokens, static_cast<int>(out.padTokenId));
	(void)out.di.setTrainSequences(trainSpans);
	(void)out.di.setTestSequences(testSpans);
	out.testTokenStream = testTokens;
	out.testSpans = testSpans;
	annotate_context_family_test_split(cfg.mode, out);

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
	tc.atlas.aegisEnabled = false;
	tc.atlas.citadelEnabled = false;
	tc.atlas.rampartEnabled = false;
	tc.atlas.meritEnabled = false;
	tc.atlas.strataEnabled = false;
	tc.atlas.auroraEnabled = false;
	tc.atlas.seamEnabled = false;
	tc.atlas.quasarEnabled = false;
	tc.atlas.geodeEnabled = false;
	tc.atlas.echoEnabled = false;
	tc.atlas.bimapEnabled = false;
	tc.atlas.pactEnabled = false;
	tc.atlas.racerEnabled = false;
	tc.atlas.kronEnabled = false;
	tc.atlas.muonEnabled = false;
	tc.atlas.matraEnabled = false;
	tc.atlas.kappaEnabled = false;
}

static void configure_atlas(glades::TrainingConfig& tc,
                            const BenchConfig& cfg,
                            VariantKind variant)
{
	tc.optimizer.type = glades::OptimizerConfig::ATLAS;
	tc.atlas.rank = cfg.atlasRank;
	tc.atlas.complementRank =
	    ((variant == VARIANT_ATLAS_SPARROW) || (variant == VARIANT_ATLAS_AEGIS)
	     || (variant == VARIANT_ATLAS_CITADEL) || (variant == VARIANT_ATLAS_RAMPART)
	     || (variant == VARIANT_ATLAS_MERIT) || (variant == VARIANT_ATLAS_STRATA)
	     || (variant == VARIANT_ATLAS_AURORA) || (variant == VARIANT_ATLAS_SEAM)
	     || (variant == VARIANT_ATLAS_QUASAR) || (variant == VARIANT_ATLAS_GEODE))
	        ? cfg.atlasComplementRank
	        : 0u;
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
		tc.atlas.kappaEnabled = (cfg.atlasKappaEnabled != 0u);
		tc.atlas.kappaHeads = cfg.atlasKappaHeads;
		tc.atlas.kappaLagBuckets = cfg.atlasKappaLagBuckets;
		tc.atlas.kappaRank = cfg.atlasKappaRank;
		tc.atlas.auroraAdamwBackbone = (cfg.atlasAuroraAdamwBackbone != 0u);
		tc.atlas.auroraHeadGain = cfg.atlasAuroraHeadGain;
		tc.atlas.auroraBodyTrustScale = cfg.atlasAuroraBodyTrustScale;
	}
	else if (variant == VARIANT_ATLAS_AEGIS)
	{
		tc.atlas.aegisEnabled = true;
		tc.atlas.sparrowEnabled = true;
		tc.atlas.sparrowModeRank = cfg.atlasSparrowModeRank;
		tc.atlas.sparrowAutoModeGate = (cfg.atlasSparrowAutoModeGate != 0u);
		tc.atlas.sparrowMemoryScale = cfg.atlasSparrowMemoryScale;
		tc.atlas.sparrowEdgeThreshold = cfg.atlasSparrowEdgeThreshold;
		tc.atlas.sparrowSecondEdgeThreshold = cfg.atlasSparrowSecondEdgeThreshold;
		tc.atlas.sparrowSecondEdgeFraction = cfg.atlasSparrowSecondEdgeFraction;
		tc.atlas.sparrowPoleMax = cfg.atlasSparrowPoleMax;
		tc.atlas.asterEnabled = true;
		tc.atlas.asterMemoryScale = cfg.atlasAsterMemoryScale;
		tc.atlas.asterEdgeThreshold = cfg.atlasAsterEdgeThreshold;
		tc.atlas.asterStateRank = cfg.atlasAsterStateRank;
		tc.atlas.asterHiddenStackDepth = cfg.atlasAsterHiddenStackDepth;
		tc.atlas.asterPoleMax = cfg.atlasAsterPoleMax;
		tc.atlas.kappaEnabled = (cfg.atlasKappaEnabled != 0u);
		tc.atlas.kappaHeads = cfg.atlasKappaHeads;
		tc.atlas.kappaLagBuckets = cfg.atlasKappaLagBuckets;
		tc.atlas.kappaRank = cfg.atlasKappaRank;
	}
	else if (variant == VARIANT_ATLAS_CITADEL)
	{
		tc.atlas.aegisEnabled = true;
		tc.atlas.citadelEnabled = true;
		tc.atlas.sparrowEnabled = true;
		tc.atlas.sparrowModeRank = cfg.atlasSparrowModeRank;
		tc.atlas.sparrowAutoModeGate = (cfg.atlasSparrowAutoModeGate != 0u);
		tc.atlas.sparrowMemoryScale = cfg.atlasSparrowMemoryScale;
		tc.atlas.sparrowEdgeThreshold = cfg.atlasSparrowEdgeThreshold;
		tc.atlas.sparrowSecondEdgeThreshold = cfg.atlasSparrowSecondEdgeThreshold;
		tc.atlas.sparrowSecondEdgeFraction = cfg.atlasSparrowSecondEdgeFraction;
		tc.atlas.sparrowPoleMax = cfg.atlasSparrowPoleMax;
		tc.atlas.asterEnabled = true;
		tc.atlas.asterMemoryScale = cfg.atlasAsterMemoryScale;
		tc.atlas.asterEdgeThreshold = cfg.atlasAsterEdgeThreshold;
		tc.atlas.asterStateRank = cfg.atlasAsterStateRank;
		tc.atlas.asterHiddenStackDepth = cfg.atlasAsterHiddenStackDepth;
		tc.atlas.asterPoleMax = cfg.atlasAsterPoleMax;
		tc.atlas.kappaEnabled = (cfg.atlasKappaEnabled != 0u);
		tc.atlas.kappaHeads = cfg.atlasKappaHeads;
		tc.atlas.kappaLagBuckets = cfg.atlasKappaLagBuckets;
		tc.atlas.kappaRank = cfg.atlasKappaRank;
	}
	else if (variant == VARIANT_ATLAS_RAMPART)
	{
		tc.atlas.aegisEnabled = true;
		tc.atlas.rampartEnabled = true;
		tc.atlas.sparrowEnabled = true;
		tc.atlas.sparrowModeRank = cfg.atlasSparrowModeRank;
		tc.atlas.sparrowAutoModeGate = (cfg.atlasSparrowAutoModeGate != 0u);
		tc.atlas.sparrowMemoryScale = cfg.atlasSparrowMemoryScale;
		tc.atlas.sparrowEdgeThreshold = cfg.atlasSparrowEdgeThreshold;
		tc.atlas.sparrowSecondEdgeThreshold = cfg.atlasSparrowSecondEdgeThreshold;
		tc.atlas.sparrowSecondEdgeFraction = cfg.atlasSparrowSecondEdgeFraction;
		tc.atlas.sparrowPoleMax = cfg.atlasSparrowPoleMax;
		tc.atlas.asterEnabled = true;
		tc.atlas.asterMemoryScale = cfg.atlasAsterMemoryScale;
		tc.atlas.asterEdgeThreshold = cfg.atlasAsterEdgeThreshold;
		tc.atlas.asterStateRank = cfg.atlasAsterStateRank;
		tc.atlas.asterHiddenStackDepth = cfg.atlasAsterHiddenStackDepth;
		tc.atlas.asterPoleMax = cfg.atlasAsterPoleMax;
		tc.atlas.kappaEnabled = (cfg.atlasKappaEnabled != 0u);
		tc.atlas.kappaHeads = cfg.atlasKappaHeads;
		tc.atlas.kappaLagBuckets = cfg.atlasKappaLagBuckets;
		tc.atlas.kappaRank = cfg.atlasKappaRank;
	}
	else if (variant == VARIANT_ATLAS_MERIT)
	{
		tc.atlas.aegisEnabled = true;
		tc.atlas.meritEnabled = true;
		tc.atlas.sparrowEnabled = true;
		tc.atlas.sparrowModeRank = cfg.atlasSparrowModeRank;
		tc.atlas.sparrowAutoModeGate = (cfg.atlasSparrowAutoModeGate != 0u);
		tc.atlas.sparrowMemoryScale = cfg.atlasSparrowMemoryScale;
		tc.atlas.sparrowEdgeThreshold = cfg.atlasSparrowEdgeThreshold;
		tc.atlas.sparrowSecondEdgeThreshold = cfg.atlasSparrowSecondEdgeThreshold;
		tc.atlas.sparrowSecondEdgeFraction = cfg.atlasSparrowSecondEdgeFraction;
		tc.atlas.sparrowPoleMax = cfg.atlasSparrowPoleMax;
		tc.atlas.asterEnabled = true;
		tc.atlas.asterMemoryScale = cfg.atlasAsterMemoryScale;
		tc.atlas.asterEdgeThreshold = cfg.atlasAsterEdgeThreshold;
		tc.atlas.asterStateRank = cfg.atlasAsterStateRank;
		tc.atlas.asterHiddenStackDepth = cfg.atlasAsterHiddenStackDepth;
		tc.atlas.asterPoleMax = cfg.atlasAsterPoleMax;
		tc.atlas.kappaEnabled = (cfg.atlasKappaEnabled != 0u);
		tc.atlas.kappaHeads = cfg.atlasKappaHeads;
		tc.atlas.kappaLagBuckets = cfg.atlasKappaLagBuckets;
		tc.atlas.kappaRank = cfg.atlasKappaRank;
	}
	else if (variant == VARIANT_ATLAS_STRATA)
	{
		tc.atlas.aegisEnabled = true;
		tc.atlas.strataEnabled = true;
		tc.atlas.sparrowEnabled = true;
		tc.atlas.sparrowModeRank = cfg.atlasSparrowModeRank;
		tc.atlas.sparrowAutoModeGate = (cfg.atlasSparrowAutoModeGate != 0u);
		tc.atlas.sparrowMemoryScale = cfg.atlasSparrowMemoryScale;
		tc.atlas.sparrowEdgeThreshold = cfg.atlasSparrowEdgeThreshold;
		tc.atlas.sparrowSecondEdgeThreshold = cfg.atlasSparrowSecondEdgeThreshold;
		tc.atlas.sparrowSecondEdgeFraction = cfg.atlasSparrowSecondEdgeFraction;
		tc.atlas.sparrowPoleMax = cfg.atlasSparrowPoleMax;
		tc.atlas.asterEnabled = true;
		tc.atlas.asterMemoryScale = cfg.atlasAsterMemoryScale;
		tc.atlas.asterEdgeThreshold = cfg.atlasAsterEdgeThreshold;
		tc.atlas.asterStateRank = cfg.atlasAsterStateRank;
		tc.atlas.asterHiddenStackDepth = cfg.atlasAsterHiddenStackDepth;
		tc.atlas.asterPoleMax = cfg.atlasAsterPoleMax;
		tc.atlas.kappaEnabled = (cfg.atlasKappaEnabled != 0u);
		tc.atlas.kappaHeads = cfg.atlasKappaHeads;
		tc.atlas.kappaLagBuckets = cfg.atlasKappaLagBuckets;
		tc.atlas.kappaRank = cfg.atlasKappaRank;
	}
	else if (variant == VARIANT_ATLAS_AURORA)
	{
		tc.atlas.auroraEnabled = true;
		tc.atlas.sparrowEnabled = true;
		tc.atlas.sparrowModeRank = cfg.atlasSparrowModeRank;
		tc.atlas.sparrowAutoModeGate = (cfg.atlasSparrowAutoModeGate != 0u);
		tc.atlas.sparrowMemoryScale = cfg.atlasSparrowMemoryScale;
		tc.atlas.sparrowEdgeThreshold = cfg.atlasSparrowEdgeThreshold;
		tc.atlas.sparrowSecondEdgeThreshold = cfg.atlasSparrowSecondEdgeThreshold;
		tc.atlas.sparrowSecondEdgeFraction = cfg.atlasSparrowSecondEdgeFraction;
		tc.atlas.sparrowPoleMax = cfg.atlasSparrowPoleMax;
		tc.atlas.asterEnabled = true;
		tc.atlas.asterMemoryScale = cfg.atlasAsterMemoryScale;
		tc.atlas.asterEdgeThreshold = cfg.atlasAsterEdgeThreshold;
		tc.atlas.asterStateRank = cfg.atlasAsterStateRank;
		tc.atlas.asterHiddenStackDepth = cfg.atlasAsterHiddenStackDepth;
		tc.atlas.asterPoleMax = cfg.atlasAsterPoleMax;
		tc.atlas.kappaEnabled = (cfg.atlasKappaEnabled != 0u);
		tc.atlas.kappaHeads = cfg.atlasKappaHeads;
		tc.atlas.kappaLagBuckets = cfg.atlasKappaLagBuckets;
		tc.atlas.kappaRank = cfg.atlasKappaRank;
	}
	else if (variant == VARIANT_ATLAS_SEAM)
	{
		tc.atlas.seamEnabled = true;
		tc.atlas.sparrowEnabled = true;
		tc.atlas.sparrowModeRank = cfg.atlasSparrowModeRank;
		tc.atlas.sparrowAutoModeGate = (cfg.atlasSparrowAutoModeGate != 0u);
		tc.atlas.sparrowMemoryScale = cfg.atlasSparrowMemoryScale;
		tc.atlas.sparrowEdgeThreshold = cfg.atlasSparrowEdgeThreshold;
		tc.atlas.sparrowSecondEdgeThreshold = cfg.atlasSparrowSecondEdgeThreshold;
		tc.atlas.sparrowSecondEdgeFraction = cfg.atlasSparrowSecondEdgeFraction;
		tc.atlas.sparrowPoleMax = cfg.atlasSparrowPoleMax;
		tc.atlas.asterEnabled = true;
		tc.atlas.asterMemoryScale = cfg.atlasAsterMemoryScale;
		tc.atlas.asterEdgeThreshold = cfg.atlasAsterEdgeThreshold;
		tc.atlas.asterStateRank = cfg.atlasAsterStateRank;
		tc.atlas.asterHiddenStackDepth = cfg.atlasAsterHiddenStackDepth;
		tc.atlas.asterPoleMax = cfg.atlasAsterPoleMax;
		tc.atlas.kappaEnabled = (cfg.atlasKappaEnabled != 0u);
		tc.atlas.kappaHeads = cfg.atlasKappaHeads;
		tc.atlas.kappaLagBuckets = cfg.atlasKappaLagBuckets;
		tc.atlas.kappaRank = cfg.atlasKappaRank;
	}
	else if (variant == VARIANT_ATLAS_QUASAR)
	{
		tc.atlas.quasarEnabled = true;
		tc.atlas.sparrowEnabled = true;
		tc.atlas.sparrowModeRank = cfg.atlasSparrowModeRank;
		tc.atlas.sparrowAutoModeGate = (cfg.atlasSparrowAutoModeGate != 0u);
		tc.atlas.sparrowMemoryScale = cfg.atlasSparrowMemoryScale;
		tc.atlas.sparrowEdgeThreshold = cfg.atlasSparrowEdgeThreshold;
		tc.atlas.sparrowSecondEdgeThreshold = cfg.atlasSparrowSecondEdgeThreshold;
		tc.atlas.sparrowSecondEdgeFraction = cfg.atlasSparrowSecondEdgeFraction;
		tc.atlas.sparrowPoleMax = cfg.atlasSparrowPoleMax;
		tc.atlas.asterEnabled = true;
		tc.atlas.asterMemoryScale = cfg.atlasAsterMemoryScale;
		tc.atlas.asterEdgeThreshold = cfg.atlasAsterEdgeThreshold;
		tc.atlas.asterStateRank = cfg.atlasAsterStateRank;
		tc.atlas.asterHiddenStackDepth = cfg.atlasAsterHiddenStackDepth;
		tc.atlas.asterPoleMax = cfg.atlasAsterPoleMax;
		tc.atlas.kappaEnabled = (cfg.atlasKappaEnabled != 0u);
		tc.atlas.kappaHeads = cfg.atlasKappaHeads;
		tc.atlas.kappaLagBuckets = cfg.atlasKappaLagBuckets;
		tc.atlas.kappaRank = cfg.atlasKappaRank;
	}
	else if (variant == VARIANT_ATLAS_GEODE)
	{
		tc.atlas.geodeEnabled = true;
		tc.atlas.sparrowEnabled = (cfg.atlasGeodePredictiveScale > 0.0f);
		tc.atlas.sparrowModeRank = cfg.atlasSparrowModeRank;
		tc.atlas.sparrowAutoModeGate = (cfg.atlasSparrowAutoModeGate != 0u);
		tc.atlas.sparrowMemoryScale = cfg.atlasSparrowMemoryScale;
		tc.atlas.sparrowEdgeThreshold = cfg.atlasSparrowEdgeThreshold;
		tc.atlas.sparrowSecondEdgeThreshold = cfg.atlasSparrowSecondEdgeThreshold;
		tc.atlas.sparrowSecondEdgeFraction = cfg.atlasSparrowSecondEdgeFraction;
		tc.atlas.sparrowPoleMax = cfg.atlasSparrowPoleMax;
		tc.atlas.geodeGeometryScale = cfg.atlasGeodeGeometryScale;
		tc.atlas.geodePredictiveScale = cfg.atlasGeodePredictiveScale;
	}
	else if (variant == VARIANT_ATLAS_ECHO)
	{
		tc.atlas.echoEnabled = true;
		tc.atlas.echoGeometryScale = cfg.atlasEchoGeometryScale;
		tc.atlas.echoGeometryScaleFinal = cfg.atlasEchoGeometryScaleFinal;
		tc.atlas.echoGeometryDecaySteps = cfg.atlasEchoGeometryDecaySteps;
		tc.atlas.echoMetricCadence = cfg.atlasEchoMetricCadence;
		tc.atlas.echoTrustScale = cfg.atlasEchoTrustScale;
		tc.atlas.echoPredictiveScale = cfg.atlasEchoPredictiveScale;
		tc.atlas.echoStructuralScale = cfg.atlasEchoStructuralScale;
		tc.atlas.echoStructuralGroups = cfg.atlasEchoStructuralGroups;
		tc.atlas.echoScope = cfg.atlasEchoScope;
	}
	else if (variant == VARIANT_ATLAS_BIMAP)
	{
		tc.atlas.bimapEnabled = true;
		tc.atlas.bimapLowRankEnabled = (cfg.atlasBiMAPLowRank != 0u);
		tc.atlas.bimapScope = cfg.atlasBiMAPScope;
		tc.atlas.bimapGeometryScale = cfg.atlasBiMAPGeometryScale;
		tc.atlas.bimapPredictiveScale = cfg.atlasBiMAPPredictiveScale;
		tc.atlas.bimapFactorCadence = cfg.atlasBiMAPFactorCadence;
	}
	else if (variant == VARIANT_ATLAS_PACT)
	{
		tc.atlas.pactEnabled = true;
		tc.atlas.pactLowRankEnabled = (cfg.atlasPACTLowRank != 0u);
		tc.atlas.pactGeometryScale = cfg.atlasPACTGeometryScale;
		tc.atlas.pactPredictiveScale = cfg.atlasPACTPredictiveScale;
		tc.atlas.pactFactorCadence = cfg.atlasPACTFactorCadence;
		tc.atlas.pactCostScale = cfg.atlasPACTCostScale;
		tc.atlas.pactPromoteThreshold = cfg.atlasPACTPromoteThreshold;
		tc.atlas.pactDemoteThreshold = cfg.atlasPACTDemoteThreshold;
	}
	else if (variant == VARIANT_ATLAS_RACER)
	{
		tc.atlas.racerEnabled = true;
		tc.atlas.racerGeometryScale = cfg.atlasRACERGeometryScale;
		tc.atlas.racerPredictiveScale = cfg.atlasRACERPredictiveScale;
		tc.atlas.racerFactorCadence = cfg.atlasRACERFactorCadence;
		tc.atlas.racerRiskScale = cfg.atlasRACERRiskScale;
		tc.atlas.racerCostScale = cfg.atlasRACERCostScale;
		tc.atlas.racerPromoteThreshold = cfg.atlasRACERPromoteThreshold;
		tc.atlas.racerDemoteThreshold = cfg.atlasRACERDemoteThreshold;
	}
	else if (variant == VARIANT_ATLAS_KRON)
	{
		tc.atlas.kronEnabled = true;
		tc.atlas.kronGeometryScale = cfg.atlasKronGeometryScale;
		tc.atlas.kronPredictiveScale = cfg.atlasKronPredictiveScale;
		tc.atlas.kronFactorCadence = cfg.atlasKronFactorCadence;
		tc.atlas.kronDamping = cfg.atlasKronDamping;
	}
	else if (variant == VARIANT_ATLAS_MUON)
	{
		tc.atlas.muonEnabled = true;
		tc.atlas.muonGeometryScale = cfg.atlasMuonGeometryScale;
		tc.atlas.muonPredictiveScale = cfg.atlasMuonPredictiveScale;
		tc.atlas.muonMaxAspect = cfg.atlasMuonMaxAspect;
		tc.atlas.muonMinDim = cfg.atlasMuonMinDim;
		tc.atlas.muonDamping = cfg.atlasMuonDamping;
	}
	else if (variant == VARIANT_ATLAS_MATRA)
	{
		tc.atlas.matraEnabled = true;
		tc.atlas.matraGeometryScale = cfg.atlasMatraGeometryScale;
		tc.atlas.matraOrthogonalScale = cfg.atlasMatraOrthogonalScale;
		tc.atlas.matraPredictiveScale = cfg.atlasMatraPredictiveScale;
		tc.atlas.matraTrustRadius = cfg.atlasMatraTrustRadius;
		tc.atlas.matraMetricCadence = cfg.atlasMatraMetricCadence;
		tc.atlas.matraMaxAspect = cfg.atlasMatraMaxAspect;
		tc.atlas.matraMinDim = cfg.atlasMatraMinDim;
		tc.atlas.matraDamping = cfg.atlasMatraDamping;
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
	if (variant == VARIANT_ADAMW_GROUP)
	{
		tc.optimizer.type = glades::OptimizerConfig::ADAMW;
		tc.optimizer.adamBeta1 = 0.9f;
		tc.optimizer.adamBeta2 = 0.999f;
		tc.optimizer.adamEps = 1e-8f;
		tc.optimizer.adamBiasCorrection = true;
		tc.optimizer.adamGroupwiseEnabled = true;
		tc.optimizer.adamGroupStabilityScale = 0.05f;
		tc.optimizer.adamGroupSnrScale = 0.05f;
		tc.optimizer.adamGroupRatioScale = 0.50f;
		tc.optimizer.adamGroupMinScale = 0.90f;
		tc.optimizer.adamGroupMaxScale = 1.15f;
		tc.optimizer.adamGroupMinSize = 256u;
		return true;
	}

	configure_atlas(tc, cfg, variant);
	(void)atlasLR;
	(void)adamLR;
	return true;
}

static bool token_variant_uses_adamw_backbone(const BenchConfig& cfg, VariantKind variant)
{
	return (variant == VARIANT_ADAMW)
	    || (variant == VARIANT_ADAMW_GROUP)
	    || (variant == VARIANT_ATLAS_AURORA && cfg.atlasAuroraAdamwBackbone != 0u)
	    || (variant == VARIANT_ATLAS_GEODE)
	    || (variant == VARIANT_ATLAS_ECHO)
	    || (variant == VARIANT_ATLAS_BIMAP)
	    || (variant == VARIANT_ATLAS_PACT)
	    || (variant == VARIANT_ATLAS_RACER)
	    || (variant == VARIANT_ATLAS_KRON)
	    || (variant == VARIANT_ATLAS_MUON)
	    || (variant == VARIANT_ATLAS_MATRA);
}

static float token_variant_learning_rate(const BenchConfig& cfg, VariantKind variant)
{
	return token_variant_uses_adamw_backbone(cfg, variant) ? cfg.token.adamLR : cfg.token.atlasLR;
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
	const float lr = token_variant_learning_rate(cfg, variant);
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
	tc.transformer.ffnActivation = glades::TransformerRunConfig::FFN_GELU;
	tc.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
	tc.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
	tc.transformer.kvCacheDType = glades::TransformerRunConfig::KV_CACHE_F32;
	tc.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX;
	tc.transformer.tokenLmSampledNegatives = 64;
	tc.transformer.tokenLmAllowHugeFullSoftmax = false;
	tc.transformer.captureOptimizerGapDiagnostics = true;
	tc.transformer.layerNormEps = 1e-5f;
	tc.transformer.ropeTheta = 10000.0f;
	tc.transformer.embeddingDropoutRate = 0.0f;
	tc.transformer.residualDropoutRate = 0.0f;
	tc.gpu.enable = (cfg.gpuEnable != 0u);
	tc.gpu.deviceId = cfg.gpuDeviceId;
	tc.gpu.minProblemSize = 0u;
	configure_optimizer(tc, cfg, variant, cfg.token.adamLR, cfg.token.atlasLR, 1.0f);

	const glades::NNetworkStatus stCfg = out.net->setTrainingConfig(tc);
	if (!stCfg.ok())
	{
		err = stCfg.message;
		return false;
	}
	return true;
}

static bool make_regression_network(const BenchConfig& cfg,
                                    const RegressionNetworkSpec& spec,
                                    unsigned int epochs,
                                    VariantKind variant,
                                    unsigned int seed,
                                    NetworkOwner& out,
                                    std::string& err)
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

	glades::OutputLayerInfo* outLayer = new glades::OutputLayerInfo(static_cast<int>(spec.outputDim),
	                                                                glades::OutputLayerInfo::REGRESSION);
	out.info = new glades::NNInfo(spec.name, in, hidden, outLayer);
	out.net = new glades::NNetwork(out.info, glades::NNetwork::TYPE_DFF);
	out.net->setSeed(seed);
	out.net->setLogger(quiet_logger());
	out.net->getTerminatorMutable().setEpoch(static_cast<int>(epochs));
	out.net->getTerminatorMutable().setAccuracy(0.0f);

	glades::TrainingConfig tc = out.net->getTrainingConfig();
	tc.transformer.enableTokenEmbedding = false;
	tc.transformer.vocabSizeOverride = 0;
	tc.transformer.tieEmbeddings = true;
	tc.transformer.padTokenId = -1;
	tc.transformer.nHeadsOverride = 0;
	tc.transformer.nKVHeadsOverride = 0;
	tc.transformer.dFFOverride = 0;
	tc.transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
	tc.transformer.ffnActivation = glades::TransformerRunConfig::FFN_RELU;
	tc.transformer.normType = glades::TransformerRunConfig::NORM_LAYERNORM;
	tc.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_SINUSOIDAL;
	tc.transformer.kvCacheDType = glades::TransformerRunConfig::KV_CACHE_F32;
	tc.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX;
	tc.transformer.tokenLmSampledNegatives = 64;
	tc.transformer.tokenLmAllowHugeFullSoftmax = false;
	tc.transformer.layerNormEps = 1e-5f;
	tc.transformer.ropeTheta = 10000.0f;
	tc.transformer.embeddingDropoutRate = 0.0f;
	tc.transformer.residualDropoutRate = 0.0f;
	configure_optimizer(tc, cfg, variant, spec.adamLR, spec.atlasLR, spec.clipNorm);
	const glades::NNetworkStatus stCfg = out.net->setTrainingConfig(tc);
	if (!stCfg.ok())
	{
		err = stCfg.message;
		return false;
	}
	return true;
}

static bool make_teacher_network(const BenchConfig& cfg,
                                 VariantKind variant,
                                 unsigned int seed,
                                 NetworkOwner& out,
                                 std::string& err)
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
	return make_regression_network(cfg, spec, cfg.teacher.epochs, variant, seed, out, err);
}

static bool make_latent_network(const BenchConfig& cfg,
                                VariantKind variant,
                                unsigned int seed,
                                NetworkOwner& out,
                                std::string& err)
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
	return make_regression_network(cfg, spec, cfg.latent.epochs, variant, seed, out, err);
}

static bool make_nonlinear_latent_network(const BenchConfig& cfg,
                                          VariantKind variant,
                                          unsigned int seed,
                                          NetworkOwner& out,
                                          std::string& err)
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
	return make_regression_network(cfg, spec, cfg.latent.epochs, variant, seed, out, err);
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

	glades::NNetwork::TransformerGroupedParameterSnapshot beforeTrainSnapshot;
	owner.net->getTransformerGroupedParameterSnapshot(beforeTrainSnapshot);

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
	fill_aegis_run_result(trainCb, out);
	fill_citadel_run_result(trainCb, out);
	fill_rampart_run_result(trainCb, out);
	fill_merit_run_result(trainCb, out);
	fill_strata_run_result(trainCb, out);
	glades::NNetwork::TransformerGroupedParameterSnapshot afterTrainSnapshot;
	owner.net->getTransformerGroupedParameterSnapshot(afterTrainSnapshot);
	fill_transformer_gap_from_snapshots(beforeTrainSnapshot, afterTrainSnapshot, out);
	glades::NNetwork::AtlasRuntimeDiagnostics trainDiag;
	if (owner.net->getAtlasRuntimeDiagnostics(trainDiag))
	{
		if (trainDiag.transformerGapBatches > 0u)
			out.transformerApplyMs = trainDiag.transformerMeanApplyMs;
		if (trainDiag.transformerMarginSnapshots > 0u)
		{
			out.transformerTrainMarginValid = true;
			out.transformerTrainTargetMargin = trainDiag.transformerMeanTargetMargin;
			out.transformerTrainHardNegativeLogit = trainDiag.transformerMeanHardNegativeLogit;
		}
	}

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
	glades::NNetwork::AtlasRuntimeDiagnostics testDiag;
	if (owner.net->getAtlasRuntimeDiagnostics(testDiag) && testDiag.transformerMarginSnapshots > 0u)
	{
		out.transformerTestMarginValid = true;
		out.transformerTestTargetMargin = testDiag.transformerMeanTargetMargin;
		out.transformerTestHardNegativeLogit = testDiag.transformerMeanHardNegativeLogit;
	}
	std::string contextDiagErr;
	(void)fill_context_family_test_diag(data, *owner.net, out, &contextDiagErr);

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
	NetworkOwner owner;
	if (!make_teacher_network(cfg, variant, seed, owner, out.err))
	{
		out.ok = false;
		return out;
	}

	const glades::NNetworkStatus warm = owner.net->test(&data);
	if (!warm.ok())
	{
		out.ok = false;
		out.err = warm.message;
		return out;
	}

	CaptureMetricsCallbacks trainCb;
	const int64_t t0 = now_ms();
	const glades::NNetworkStatus trainStatus = owner.net->train(&data, &trainCb);
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
	fill_aegis_run_result(trainCb, out);
	fill_citadel_run_result(trainCb, out);
	fill_rampart_run_result(trainCb, out);
	fill_merit_run_result(trainCb, out);
	fill_strata_run_result(trainCb, out);

	CaptureMetricsCallbacks testCb;
	const int64_t t2 = now_ms();
	const glades::NNetworkStatus testStatus = owner.net->test(&data, &testCb);
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
	NetworkOwner owner;
	if (!make_latent_network(cfg, variant, seed, owner, out.err))
	{
		out.ok = false;
		return out;
	}

	const glades::NNetworkStatus warm = owner.net->test(&data);
	if (!warm.ok())
	{
		out.ok = false;
		out.err = warm.message;
		return out;
	}

	CaptureMetricsCallbacks trainCb;
	const int64_t t0 = now_ms();
	const glades::NNetworkStatus trainStatus = owner.net->train(&data, &trainCb);
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
	fill_aegis_run_result(trainCb, out);
	fill_citadel_run_result(trainCb, out);
	fill_rampart_run_result(trainCb, out);
	fill_merit_run_result(trainCb, out);
	fill_strata_run_result(trainCb, out);

	CaptureMetricsCallbacks testCb;
	const int64_t t2 = now_ms();
	const glades::NNetworkStatus testStatus = owner.net->test(&data, &testCb);
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
	NetworkOwner owner;
	if (!make_nonlinear_latent_network(cfg, variant, seed, owner, out.err))
	{
		out.ok = false;
		return out;
	}

	const glades::NNetworkStatus warm = owner.net->test(&data);
	if (!warm.ok())
	{
		out.ok = false;
		out.err = warm.message;
		return out;
	}

	CaptureMetricsCallbacks trainCb;
	const int64_t t0 = now_ms();
	const glades::NNetworkStatus trainStatus = owner.net->train(&data, &trainCb);
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
	fill_aegis_run_result(trainCb, out);
	fill_citadel_run_result(trainCb, out);
	fill_rampart_run_result(trainCb, out);
	fill_merit_run_result(trainCb, out);
	fill_strata_run_result(trainCb, out);

	CaptureMetricsCallbacks testCb;
	const int64_t t2 = now_ms();
	const glades::NNetworkStatus testStatus = owner.net->test(&data, &testCb);
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
	printf("  ASTER time:    boundaryMs=%6.3f +/- %-6.3f  setupMs=%6.3f +/- %-6.3f  transportMs=%6.3f +/- %-6.3f  transferMs=%6.3f +/- %-6.3f  stateMs=%6.3f +/- %-6.3f  innovMs=%6.3f +/- %-6.3f  applyMs=%6.3f +/- %-6.3f\n",
	       s.asterBoundaryMs.mean, s.asterBoundaryMs.stddev,
	       s.asterSetupMs.mean, s.asterSetupMs.stddev,
	       s.asterTransportMs.mean, s.asterTransportMs.stddev,
	       s.asterTransferFitMs.mean, s.asterTransferFitMs.stddev,
	       s.asterStateFitMs.mean, s.asterStateFitMs.stddev,
	       s.asterInnovationFitMs.mean, s.asterInnovationFitMs.stddev,
	       s.asterApplyMs.mean, s.asterApplyMs.stddev);
}

static void print_aegis_usage_row(const Summary& s)
{
	if (!s.aegisDiagValid)
		return;
	printf("  AEGIS usage:   lS=%5.3f +/- %-5.3f  lP=%5.3f +/- %-5.3f  lO=%5.3f +/- %-5.3f  pPred=%5.3f +/- %-5.3f  pReal=%5.3f +/- %-5.3f\n",
	       s.aegisLambdaSpatial.mean, s.aegisLambdaSpatial.stddev,
	       s.aegisLambdaPredictive.mean, s.aegisLambdaPredictive.stddev,
	       s.aegisLambdaOutput.mean, s.aegisLambdaOutput.stddev,
	       s.aegisPredictivePredicted.mean, s.aegisPredictivePredicted.stddev,
	       s.aegisPredictiveRealized.mean, s.aegisPredictiveRealized.stddev);
	printf("                oPred=%5.3f +/- %-5.3f  oReal=%5.3f +/- %-5.3f  pErr=%5.3f +/- %-5.3f  oErr=%5.3f +/- %-5.3f  disagree=%5.3f +/- %-5.3f\n",
	       s.aegisOutputPredicted.mean, s.aegisOutputPredicted.stddev,
	       s.aegisOutputRealized.mean, s.aegisOutputRealized.stddev,
	       s.aegisPredictiveError.mean, s.aegisPredictiveError.stddev,
	       s.aegisOutputError.mean, s.aegisOutputError.stddev,
	       s.aegisChannelDisagreement.mean, s.aegisChannelDisagreement.stddev);
}

static void print_citadel_usage_row(const Summary& s)
{
	if (!s.citadelDiagValid)
		return;
	printf("  CITADEL:      anchor=%5.3f +/- %-5.3f  hard=%5.3f +/- %-5.3f  sparrowTrust=%5.3f +/- %-5.3f\n",
	       s.citadelAnchor.mean, s.citadelAnchor.stddev,
	       s.citadelHardRegimeMass.mean, s.citadelHardRegimeMass.stddev,
	       s.citadelSparrowTrust.mean, s.citadelSparrowTrust.stddev);
}

static void print_rampart_usage_row(const Summary& s)
{
	if (!s.rampartDiagValid)
		return;
	printf("  RAMPART:      tau=%5.3f +/- %-5.3f  budget=%5.3f +/- %-5.3f  cov=%5.3f +/- %-5.3f  sparrowTrust=%5.3f +/- %-5.3f\n",
	       s.rampartTau.mean, s.rampartTau.stddev,
	       s.rampartBudget.mean, s.rampartBudget.stddev,
	       s.rampartCovariance.mean, s.rampartCovariance.stddev,
	       s.rampartSparrowTrust.mean, s.rampartSparrowTrust.stddev);
}

static void print_merit_usage_row(const Summary& s)
{
	if (!s.meritDiagValid)
		return;
	printf("  MERIT:        tau=%5.3f +/- %-5.3f  budget=%5.3f +/- %-5.3f  cov=%5.3f +/- %-5.3f  sparrowTrust=%5.3f +/- %-5.3f  geom=%5.3f +/- %-5.3f\n",
	       s.meritTau.mean, s.meritTau.stddev,
	       s.meritBudget.mean, s.meritBudget.stddev,
	       s.meritCovariance.mean, s.meritCovariance.stddev,
	       s.meritSparrowTrust.mean, s.meritSparrowTrust.stddev,
	       s.meritGeometryTrust.mean, s.meritGeometryTrust.stddev);
}

static void print_strata_usage_row(const Summary& s)
{
	if (!s.strataDiagValid)
		return;
	printf("  STRATA:       null=%5.3f +/- %-5.3f  pred=%5.3f +/- %-5.3f  out=%5.3f +/- %-5.3f  coupled=%5.3f +/- %-5.3f  budget=%5.3f +/- %-5.3f\n",
	       s.strataNullMode.mean, s.strataNullMode.stddev,
	       s.strataPredictiveMode.mean, s.strataPredictiveMode.stddev,
	       s.strataOutputMode.mean, s.strataOutputMode.stddev,
	       s.strataCoupledMode.mean, s.strataCoupledMode.stddev,
	       s.strataBudget.mean, s.strataBudget.stddev);
	printf("                bNull=%5.3f +/- %-5.3f  bPred=%5.3f +/- %-5.3f  bOut=%5.3f +/- %-5.3f  bCoupled=%5.3f +/- %-5.3f  excess=%5.3f +/- %-5.3f  switch=%5.3f +/- %-5.3f\n",
	       s.strataNullBenefit.mean, s.strataNullBenefit.stddev,
	       s.strataPredictiveBenefit.mean, s.strataPredictiveBenefit.stddev,
	       s.strataOutputBenefit.mean, s.strataOutputBenefit.stddev,
	       s.strataCoupledBenefit.mean, s.strataCoupledBenefit.stddev,
	       s.strataSelectedExcess.mean, s.strataSelectedExcess.stddev,
	       s.strataSwitchRate.mean, s.strataSwitchRate.stddev);
}

static void print_transformer_gap_row(const Summary& s)
{
	if (!s.transformerGapDiagValid)
		return;
	std::ostringstream oss;
	oss << "  XFORM proxy:  in=" << std::fixed << std::setprecision(3)
	    << s.transformerInputUpdateNorm.mean << " +/- " << s.transformerInputUpdateNorm.stddev
	    << "  blk=[";
	for (size_t i = 0; i < s.transformerBlockUpdateNorms.size(); ++i)
	{
		if (i > 0u)
			oss << ' ';
		oss << s.transformerBlockUpdateNorms[i].mean;
	}
	oss << "]  final=" << s.transformerFinalNormUpdateNorm.mean
	    << " +/- " << s.transformerFinalNormUpdateNorm.stddev
	    << "  head=" << s.transformerHeadUpdateNorm.mean
	    << " +/- " << s.transformerHeadUpdateNorm.stddev
	    << "  hShare=" << s.transformerHeadShare.mean
	    << " +/- " << s.transformerHeadShare.stddev
	    << "  applyMs=" << s.transformerApplyMs.mean
	    << " +/- " << s.transformerApplyMs.stddev;
	printf("%s\n", oss.str().c_str());
	if (s.transformerTrainMarginValid || s.transformerTestMarginValid)
	{
		printf("                trainMargin=%5.3f +/- %-5.3f  trainHardNeg=%5.3f +/- %-5.3f  testMargin=%5.3f +/- %-5.3f  testHardNeg=%5.3f +/- %-5.3f\n",
		       s.transformerTrainTargetMargin.mean, s.transformerTrainTargetMargin.stddev,
		       s.transformerTrainHardNegativeLogit.mean, s.transformerTrainHardNegativeLogit.stddev,
		       s.transformerTestTargetMargin.mean, s.transformerTestTargetMargin.stddev,
		       s.transformerTestHardNegativeLogit.mean, s.transformerTestHardNegativeLogit.stddev);
	}
}

static void append_context_bucket(std::ostringstream& oss,
                                  bool& first,
                                  const char* label,
                                  const std::vector<AggregateStats>& nlls,
                                  const std::vector<AggregateStats>& shares,
                                  size_t idx)
{
	if (idx >= nlls.size() || idx >= shares.size() || shares[idx].mean <= 0.0)
		return;
	if (!first)
		oss << "  ";
	first = false;
	oss << label << '=' << std::fixed << std::setprecision(3) << nlls[idx].mean
	    << '/' << shares[idx].mean;
}

static void print_context_family_diag_row(const Summary& s)
{
	if (!s.contextFamilyDiagValid)
		return;

	{
		std::ostringstream oss;
		oss << "  CTX role:     ";
		bool first = true;
		for (unsigned int i = 0u; i < CTX_ROLE_COUNT; ++i)
			append_context_bucket(oss, first, context_family_role_label(i), s.contextRoleNll, s.contextRoleShare, i);
		printf("%s\n", oss.str().c_str());
	}

	{
		std::ostringstream oss;
		oss << "  CTX recall:   ";
		bool first = true;
		for (unsigned int i = 0u; i < CTX_RECALL_DISTANCE_COUNT; ++i)
			append_context_bucket(oss, first, context_family_recall_distance_label(i),
			                     s.contextRecallDistanceNll, s.contextRecallDistanceShare, i);
		if (!first)
			printf("%s\n", oss.str().c_str());
	}

	{
		std::ostringstream oss;
		oss << "  CTX subtype:  ";
		bool first = true;
		for (unsigned int i = 0u; i < CTX_SUBTYPE_COUNT; ++i)
			append_context_bucket(oss, first, context_family_recall_subtype_label(i),
			                     s.contextRecallSubtypeNll, s.contextRecallSubtypeShare, i);
		if (!first)
			printf("%s\n", oss.str().c_str());
	}
}

static void print_transformer_gap_compare_row(const Summary& adamw, const Summary& other)
{
	if (!adamw.transformerGapDiagValid || !other.transformerGapDiagValid)
		return;
	const std::vector<double> adamwProfile = make_transformer_profile(adamw);
	const std::vector<double> otherProfile = make_transformer_profile(other);
	const double cosine = profile_cosine(adamwProfile, otherProfile);
	printf("  vs AdamW %-12s  profileCos=%+6.3f  dHeadShare=%+7.4f  dTrainMargin=%+7.4f  dTestMargin=%+7.4f  dApplyMs=%+7.4f\n",
	       other.label,
	       cosine,
	       other.transformerHeadShare.mean - adamw.transformerHeadShare.mean,
	       other.transformerTrainTargetMargin.mean - adamw.transformerTrainTargetMargin.mean,
	       other.transformerTestTargetMargin.mean - adamw.transformerTestTargetMargin.mean,
	       other.transformerApplyMs.mean - adamw.transformerApplyMs.mean);
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
	out.asterBoundaryMs = cb.atlasAsterBoundaryMsSum / denom;
	out.asterSetupMs = cb.atlasAsterSetupMsSum / denom;
	out.asterTransportMs = cb.atlasAsterTransportMsSum / denom;
	out.asterTransferFitMs = cb.atlasAsterTransferFitMsSum / denom;
	out.asterStateFitMs = cb.atlasAsterStateFitMsSum / denom;
	out.asterInnovationFitMs = cb.atlasAsterInnovationFitMsSum / denom;
	out.asterApplyMs = cb.atlasAsterApplyMsSum / denom;
}

static void fill_aegis_run_result(const CaptureMetricsCallbacks& cb, RunResult& out)
{
	if (!cb.sawAtlas || cb.atlasAegisEpochs == 0u)
		return;
	const double denom = static_cast<double>(cb.atlasAegisEpochs);
	out.aegisDiagValid = true;
	out.aegisLambdaSpatial = cb.atlasAegisLambdaSpatialSum / denom;
	out.aegisLambdaPredictive = cb.atlasAegisLambdaPredictiveSum / denom;
	out.aegisLambdaOutput = cb.atlasAegisLambdaOutputSum / denom;
	out.aegisPredictivePredicted = cb.atlasAegisPredictivePredictedSum / denom;
	out.aegisPredictiveRealized = cb.atlasAegisPredictiveRealizedSum / denom;
	out.aegisOutputPredicted = cb.atlasAegisOutputPredictedSum / denom;
	out.aegisOutputRealized = cb.atlasAegisOutputRealizedSum / denom;
	out.aegisPredictiveError = cb.atlasAegisPredictiveErrorSum / denom;
	out.aegisOutputError = cb.atlasAegisOutputErrorSum / denom;
	out.aegisChannelDisagreement = cb.atlasAegisChannelDisagreementSum / denom;
}

static void fill_citadel_run_result(const CaptureMetricsCallbacks& cb, RunResult& out)
{
	if (!cb.sawAtlas || cb.atlasCitadelEpochs == 0u)
		return;
	const double denom = static_cast<double>(cb.atlasCitadelEpochs);
	out.citadelDiagValid = true;
	out.citadelAnchor = cb.atlasCitadelAnchorSum / denom;
	out.citadelHardRegimeMass = cb.atlasCitadelHardRegimeMassSum / denom;
	out.citadelSparrowTrust = cb.atlasCitadelSparrowTrustSum / denom;
}

static void fill_rampart_run_result(const CaptureMetricsCallbacks& cb, RunResult& out)
{
	if (!cb.sawAtlas || cb.atlasRampartEpochs == 0u)
		return;
	const double denom = static_cast<double>(cb.atlasRampartEpochs);
	out.rampartDiagValid = true;
	out.rampartTau = cb.atlasRampartTauSum / denom;
	out.rampartBudget = cb.atlasRampartBudgetSum / denom;
	out.rampartCovariance = cb.atlasRampartCovarianceSum / denom;
	out.rampartSparrowTrust = cb.atlasRampartSparrowTrustSum / denom;
}

static void fill_merit_run_result(const CaptureMetricsCallbacks& cb, RunResult& out)
{
	if (!cb.sawAtlas || cb.atlasMeritEpochs == 0u)
		return;
	const double denom = static_cast<double>(cb.atlasMeritEpochs);
	out.meritDiagValid = true;
	out.meritTau = cb.atlasMeritTauSum / denom;
	out.meritBudget = cb.atlasMeritBudgetSum / denom;
	out.meritCovariance = cb.atlasMeritCovarianceSum / denom;
	out.meritSparrowTrust = cb.atlasMeritSparrowTrustSum / denom;
	out.meritGeometryTrust = cb.atlasMeritGeometryTrustSum / denom;
}

static void fill_strata_run_result(const CaptureMetricsCallbacks& cb, RunResult& out)
{
	if (!cb.sawAtlas || cb.atlasStrataEpochs == 0u)
		return;
	const double denom = static_cast<double>(cb.atlasStrataEpochs);
	out.strataDiagValid = true;
	out.strataNullMode = cb.atlasStrataNullModeSum / denom;
	out.strataPredictiveMode = cb.atlasStrataPredictiveModeSum / denom;
	out.strataOutputMode = cb.atlasStrataOutputModeSum / denom;
	out.strataCoupledMode = cb.atlasStrataCoupledModeSum / denom;
	out.strataBudget = cb.atlasStrataBudgetSum / denom;
	out.strataNullBenefit = cb.atlasStrataNullBenefitSum / denom;
	out.strataPredictiveBenefit = cb.atlasStrataPredictiveBenefitSum / denom;
	out.strataOutputBenefit = cb.atlasStrataOutputBenefitSum / denom;
	out.strataCoupledBenefit = cb.atlasStrataCoupledBenefitSum / denom;
	out.strataSelectedExcess = cb.atlasStrataSelectedExcessSum / denom;
	out.strataSwitchRate = cb.atlasStrataSwitchRateSum / denom;
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
	std::vector<double> asterBoundaryMsVals;
	std::vector<double> asterSetupMsVals;
	std::vector<double> asterTransportMsVals;
	std::vector<double> asterTransferFitMsVals;
	std::vector<double> asterStateFitMsVals;
	std::vector<double> asterInnovationFitMsVals;
	std::vector<double> asterApplyMsVals;
	std::vector<double> aegisLambdaSpatialVals;
	std::vector<double> aegisLambdaPredictiveVals;
	std::vector<double> aegisLambdaOutputVals;
	std::vector<double> aegisPredictivePredictedVals;
	std::vector<double> aegisPredictiveRealizedVals;
	std::vector<double> aegisOutputPredictedVals;
	std::vector<double> aegisOutputRealizedVals;
	std::vector<double> aegisPredictiveErrorVals;
	std::vector<double> aegisOutputErrorVals;
	std::vector<double> aegisChannelDisagreementVals;
	std::vector<double> citadelAnchorVals;
	std::vector<double> citadelHardRegimeMassVals;
	std::vector<double> citadelSparrowTrustVals;
	std::vector<double> rampartTauVals;
	std::vector<double> rampartBudgetVals;
	std::vector<double> rampartCovarianceVals;
	std::vector<double> rampartSparrowTrustVals;
	std::vector<double> meritTauVals;
	std::vector<double> meritBudgetVals;
	std::vector<double> meritCovarianceVals;
	std::vector<double> meritSparrowTrustVals;
	std::vector<double> meritGeometryTrustVals;
	std::vector<double> strataNullModeVals;
	std::vector<double> strataPredictiveModeVals;
	std::vector<double> strataOutputModeVals;
	std::vector<double> strataCoupledModeVals;
	std::vector<double> strataBudgetVals;
	std::vector<double> strataNullBenefitVals;
	std::vector<double> strataPredictiveBenefitVals;
	std::vector<double> strataOutputBenefitVals;
	std::vector<double> strataCoupledBenefitVals;
	std::vector<double> strataSelectedExcessVals;
	std::vector<double> strataSwitchRateVals;
	std::vector<double> transformerInputUpdateNormVals;
	std::vector< std::vector<double> > transformerBlockUpdateNormRows;
	std::vector<double> transformerFinalNormUpdateNormVals;
	std::vector<double> transformerHeadUpdateNormVals;
	std::vector<double> transformerHeadShareVals;
	std::vector<double> transformerNonHeadShareVals;
	std::vector<double> transformerApplyMsVals;
	std::vector<double> transformerTrainTargetMarginVals;
	std::vector<double> transformerTrainHardNegativeLogitVals;
	std::vector<double> transformerTestTargetMarginVals;
	std::vector<double> transformerTestHardNegativeLogitVals;
	std::vector< std::vector<double> > contextRoleNllRows;
	std::vector< std::vector<double> > contextRoleShareRows;
	std::vector< std::vector<double> > contextRecallDistanceNllRows;
	std::vector< std::vector<double> > contextRecallDistanceShareRows;
	std::vector< std::vector<double> > contextRecallSubtypeNllRows;
	std::vector< std::vector<double> > contextRecallSubtypeShareRows;
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
			asterBoundaryMsVals.push_back(runs[i].asterBoundaryMs);
			asterSetupMsVals.push_back(runs[i].asterSetupMs);
			asterTransportMsVals.push_back(runs[i].asterTransportMs);
			asterTransferFitMsVals.push_back(runs[i].asterTransferFitMs);
			asterStateFitMsVals.push_back(runs[i].asterStateFitMs);
			asterInnovationFitMsVals.push_back(runs[i].asterInnovationFitMs);
			asterApplyMsVals.push_back(runs[i].asterApplyMs);
		}
		if (runs[i].aegisDiagValid)
		{
			aegisLambdaSpatialVals.push_back(runs[i].aegisLambdaSpatial);
			aegisLambdaPredictiveVals.push_back(runs[i].aegisLambdaPredictive);
			aegisLambdaOutputVals.push_back(runs[i].aegisLambdaOutput);
			aegisPredictivePredictedVals.push_back(runs[i].aegisPredictivePredicted);
			aegisPredictiveRealizedVals.push_back(runs[i].aegisPredictiveRealized);
			aegisOutputPredictedVals.push_back(runs[i].aegisOutputPredicted);
			aegisOutputRealizedVals.push_back(runs[i].aegisOutputRealized);
			aegisPredictiveErrorVals.push_back(runs[i].aegisPredictiveError);
			aegisOutputErrorVals.push_back(runs[i].aegisOutputError);
			aegisChannelDisagreementVals.push_back(runs[i].aegisChannelDisagreement);
		}
		if (runs[i].citadelDiagValid)
		{
			citadelAnchorVals.push_back(runs[i].citadelAnchor);
			citadelHardRegimeMassVals.push_back(runs[i].citadelHardRegimeMass);
			citadelSparrowTrustVals.push_back(runs[i].citadelSparrowTrust);
		}
		if (runs[i].rampartDiagValid)
		{
			rampartTauVals.push_back(runs[i].rampartTau);
			rampartBudgetVals.push_back(runs[i].rampartBudget);
			rampartCovarianceVals.push_back(runs[i].rampartCovariance);
			rampartSparrowTrustVals.push_back(runs[i].rampartSparrowTrust);
		}
		if (runs[i].meritDiagValid)
		{
			meritTauVals.push_back(runs[i].meritTau);
			meritBudgetVals.push_back(runs[i].meritBudget);
			meritCovarianceVals.push_back(runs[i].meritCovariance);
			meritSparrowTrustVals.push_back(runs[i].meritSparrowTrust);
			meritGeometryTrustVals.push_back(runs[i].meritGeometryTrust);
		}
		if (runs[i].strataDiagValid)
		{
			strataNullModeVals.push_back(runs[i].strataNullMode);
			strataPredictiveModeVals.push_back(runs[i].strataPredictiveMode);
			strataOutputModeVals.push_back(runs[i].strataOutputMode);
			strataCoupledModeVals.push_back(runs[i].strataCoupledMode);
			strataBudgetVals.push_back(runs[i].strataBudget);
			strataNullBenefitVals.push_back(runs[i].strataNullBenefit);
			strataPredictiveBenefitVals.push_back(runs[i].strataPredictiveBenefit);
			strataOutputBenefitVals.push_back(runs[i].strataOutputBenefit);
			strataCoupledBenefitVals.push_back(runs[i].strataCoupledBenefit);
			strataSelectedExcessVals.push_back(runs[i].strataSelectedExcess);
			strataSwitchRateVals.push_back(runs[i].strataSwitchRate);
		}
		if (runs[i].transformerGapDiagValid)
		{
			transformerInputUpdateNormVals.push_back(runs[i].transformerInputUpdateNorm);
			transformerBlockUpdateNormRows.push_back(runs[i].transformerBlockUpdateNorms);
			transformerFinalNormUpdateNormVals.push_back(runs[i].transformerFinalNormUpdateNorm);
			transformerHeadUpdateNormVals.push_back(runs[i].transformerHeadUpdateNorm);
			transformerHeadShareVals.push_back(runs[i].transformerHeadShare);
			transformerNonHeadShareVals.push_back(runs[i].transformerNonHeadShare);
			transformerApplyMsVals.push_back(runs[i].transformerApplyMs);
		}
		if (runs[i].transformerTrainMarginValid)
		{
			transformerTrainTargetMarginVals.push_back(runs[i].transformerTrainTargetMargin);
			transformerTrainHardNegativeLogitVals.push_back(runs[i].transformerTrainHardNegativeLogit);
		}
		if (runs[i].transformerTestMarginValid)
		{
			transformerTestTargetMarginVals.push_back(runs[i].transformerTestTargetMargin);
			transformerTestHardNegativeLogitVals.push_back(runs[i].transformerTestHardNegativeLogit);
		}
		if (runs[i].contextFamilyDiagValid)
		{
			contextRoleNllRows.push_back(runs[i].contextRoleNll);
			contextRoleShareRows.push_back(runs[i].contextRoleShare);
			contextRecallDistanceNllRows.push_back(runs[i].contextRecallDistanceNll);
			contextRecallDistanceShareRows.push_back(runs[i].contextRecallDistanceShare);
			contextRecallSubtypeNllRows.push_back(runs[i].contextRecallSubtypeNll);
			contextRecallSubtypeShareRows.push_back(runs[i].contextRecallSubtypeShare);
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
	s.asterBoundaryMs = compute_stats(asterBoundaryMsVals);
	s.asterSetupMs = compute_stats(asterSetupMsVals);
	s.asterTransportMs = compute_stats(asterTransportMsVals);
	s.asterTransferFitMs = compute_stats(asterTransferFitMsVals);
	s.asterStateFitMs = compute_stats(asterStateFitMsVals);
	s.asterInnovationFitMs = compute_stats(asterInnovationFitMsVals);
	s.asterApplyMs = compute_stats(asterApplyMsVals);
	s.aegisDiagValid = !aegisLambdaSpatialVals.empty();
	s.aegisLambdaSpatial = compute_stats(aegisLambdaSpatialVals);
	s.aegisLambdaPredictive = compute_stats(aegisLambdaPredictiveVals);
	s.aegisLambdaOutput = compute_stats(aegisLambdaOutputVals);
	s.aegisPredictivePredicted = compute_stats(aegisPredictivePredictedVals);
	s.aegisPredictiveRealized = compute_stats(aegisPredictiveRealizedVals);
	s.aegisOutputPredicted = compute_stats(aegisOutputPredictedVals);
	s.aegisOutputRealized = compute_stats(aegisOutputRealizedVals);
	s.aegisPredictiveError = compute_stats(aegisPredictiveErrorVals);
	s.aegisOutputError = compute_stats(aegisOutputErrorVals);
	s.aegisChannelDisagreement = compute_stats(aegisChannelDisagreementVals);
	s.citadelDiagValid = !citadelAnchorVals.empty();
	s.citadelAnchor = compute_stats(citadelAnchorVals);
	s.citadelHardRegimeMass = compute_stats(citadelHardRegimeMassVals);
	s.citadelSparrowTrust = compute_stats(citadelSparrowTrustVals);
	s.rampartDiagValid = !rampartTauVals.empty();
	s.rampartTau = compute_stats(rampartTauVals);
	s.rampartBudget = compute_stats(rampartBudgetVals);
	s.rampartCovariance = compute_stats(rampartCovarianceVals);
	s.rampartSparrowTrust = compute_stats(rampartSparrowTrustVals);
	s.meritDiagValid = !meritTauVals.empty();
	s.meritTau = compute_stats(meritTauVals);
	s.meritBudget = compute_stats(meritBudgetVals);
	s.meritCovariance = compute_stats(meritCovarianceVals);
	s.meritSparrowTrust = compute_stats(meritSparrowTrustVals);
	s.meritGeometryTrust = compute_stats(meritGeometryTrustVals);
	s.strataDiagValid = !strataNullModeVals.empty();
	s.strataNullMode = compute_stats(strataNullModeVals);
	s.strataPredictiveMode = compute_stats(strataPredictiveModeVals);
	s.strataOutputMode = compute_stats(strataOutputModeVals);
	s.strataCoupledMode = compute_stats(strataCoupledModeVals);
	s.strataBudget = compute_stats(strataBudgetVals);
	s.strataNullBenefit = compute_stats(strataNullBenefitVals);
	s.strataPredictiveBenefit = compute_stats(strataPredictiveBenefitVals);
	s.strataOutputBenefit = compute_stats(strataOutputBenefitVals);
	s.strataCoupledBenefit = compute_stats(strataCoupledBenefitVals);
	s.strataSelectedExcess = compute_stats(strataSelectedExcessVals);
	s.strataSwitchRate = compute_stats(strataSwitchRateVals);
	s.transformerGapDiagValid = !transformerInputUpdateNormVals.empty();
	s.transformerInputUpdateNorm = compute_stats(transformerInputUpdateNormVals);
	s.transformerBlockUpdateNorms = compute_stats_by_index(transformerBlockUpdateNormRows);
	s.transformerFinalNormUpdateNorm = compute_stats(transformerFinalNormUpdateNormVals);
	s.transformerHeadUpdateNorm = compute_stats(transformerHeadUpdateNormVals);
	s.transformerHeadShare = compute_stats(transformerHeadShareVals);
	s.transformerNonHeadShare = compute_stats(transformerNonHeadShareVals);
	s.transformerApplyMs = compute_stats(transformerApplyMsVals);
	s.transformerTrainMarginValid = !transformerTrainTargetMarginVals.empty();
	s.transformerTrainTargetMargin = compute_stats(transformerTrainTargetMarginVals);
	s.transformerTrainHardNegativeLogit = compute_stats(transformerTrainHardNegativeLogitVals);
	s.transformerTestMarginValid = !transformerTestTargetMarginVals.empty();
	s.transformerTestTargetMargin = compute_stats(transformerTestTargetMarginVals);
	s.transformerTestHardNegativeLogit = compute_stats(transformerTestHardNegativeLogitVals);
	s.contextFamilyDiagValid = !contextRoleNllRows.empty();
	s.contextRoleNll = compute_stats_by_index(contextRoleNllRows);
	s.contextRoleShare = compute_stats_by_index(contextRoleShareRows);
	s.contextRecallDistanceNll = compute_stats_by_index(contextRecallDistanceNllRows);
	s.contextRecallDistanceShare = compute_stats_by_index(contextRecallDistanceShareRows);
	s.contextRecallSubtypeNll = compute_stats_by_index(contextRecallSubtypeNllRows);
	s.contextRecallSubtypeShare = compute_stats_by_index(contextRecallSubtypeShareRows);
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
	const bool largeCase = (cfg.mode == MODE_TOKEN_LM_LARGE);
	const bool contextCase = (cfg.mode == MODE_TOKEN_LM_CONTEXT);
	const bool contextLargeCase = (cfg.mode == MODE_TOKEN_LM_CONTEXT_LARGE);
	const bool documentCase = (cfg.mode == MODE_TOKEN_LM_DOCUMENT);
	const bool corpusCase = (cfg.mode == MODE_TOKEN_LM_CORPUS);
	const bool corpusLargeCase = (cfg.mode == MODE_TOKEN_LM_CORPUS_LARGE);
	const char* caseName = "token-lm";
	const char* caseDescription =
	    "autoregressive next-token prediction with a small decoder-only transformer on a synthetic order-2 recurrence.";

	if (largeCase)
	{
		caseName = "token-lm-large";
		caseDescription =
		    "larger autoregressive next-token prediction benchmark on the same synthetic order-2 recurrence, using a wider/deeper decoder preset.";
	}
	if (contextCase)
	{
		caseName = "token-lm-context";
		caseDescription =
		    "structured-context autoregressive next-token prediction with topic markers, delayed summary recall, anchor recall, and local continuation inside each sequence.";
	}
	if (contextLargeCase)
	{
		caseName = "token-lm-context-large";
		caseDescription =
		    "larger structured-context autoregressive next-token prediction with longer sequences, deeper decoder, and heavier delayed recall pressure.";
	}
	if (documentCase)
	{
		caseName = "token-lm-document";
		caseDescription =
		    "larger pseudo-document autoregressive next-token prediction with article-style sections, cross-paragraph entity recall, and mixed topical/detail token populations.";
	}
	if (corpusCase)
	{
		caseName = "token-lm-corpus";
		caseDescription =
		    "checked-in small-corpus autoregressive next-token prediction on public-domain prose excerpts with a shared train/test vocabulary and contiguous sequence windows.";
	}
	if (corpusLargeCase)
	{
		caseName = "token-lm-corpus-large";
		caseDescription =
		    "larger checked-in corpus autoregressive next-token prediction with more public-domain prose documents, longer windows, and a shared train/test vocabulary.";
	}

	printf("------------------------------------------------------------\n");
	printf("Case: %s\n", caseName);
	printf("Description: %s\n", caseDescription);
	printf("Config: vocab=%u dModel=%u dFF=%u layers=%u heads=%u seqLen=%u trainSeqs=%u testSeqs=%u epochs=%u repeats=%u\n",
	       cfg.token.vocab, cfg.token.dModel, cfg.token.dFF, cfg.token.layers, cfg.token.heads,
	       cfg.token.seqLen, cfg.token.trainSeqs, cfg.token.testSeqs, cfg.token.epochs, cfg.repeats);
	const float baseTokenLR = cfg.token.atlasLR;
	const float sparrowTokenLR = cfg.token.atlasLR;
	const float helmTokenLR = cfg.token.atlasLR;
	const float asterTokenLR = cfg.token.atlasLR;
	const float aegisTokenLR = cfg.token.atlasLR;
	const float citadelTokenLR = cfg.token.atlasLR;
	const float rampartTokenLR = cfg.token.atlasLR;
	const float meritTokenLR = cfg.token.atlasLR;
	const float strataTokenLR = cfg.token.atlasLR;
	const float auroraTokenLR = token_variant_learning_rate(cfg, VARIANT_ATLAS_AURORA);
	const float seamTokenLR = cfg.token.atlasLR;
	const float quasarTokenLR = cfg.token.atlasLR;
	const float geodeTokenLR = token_variant_learning_rate(cfg, VARIANT_ATLAS_GEODE);
	const float echoTokenLR = token_variant_learning_rate(cfg, VARIANT_ATLAS_ECHO);
	const float bimapTokenLR = token_variant_learning_rate(cfg, VARIANT_ATLAS_BIMAP);
	const float pactTokenLR = token_variant_learning_rate(cfg, VARIANT_ATLAS_PACT);
	const float racerTokenLR = token_variant_learning_rate(cfg, VARIANT_ATLAS_RACER);
	const float kronTokenLR = token_variant_learning_rate(cfg, VARIANT_ATLAS_KRON);
	const float muonTokenLR = token_variant_learning_rate(cfg, VARIANT_ATLAS_MUON);
	const float matraTokenLR = token_variant_learning_rate(cfg, VARIANT_ATLAS_MATRA);
	printf("Optimizers: AdamW(lr=%.4f) ATLAS-BSRP(lr=%.4f cRank=0) ATLAS-SPARROW(lr=%.4f cRank=%u modeRankCap=%u autoGate=%u) ATLAS-HELM(lr=%.4f modeRank=%u hiddenStack=%u) ATLAS-ASTER(lr=%.4f stateRank=%u hiddenStack=%u) ATLAS-AEGIS(lr=%.4f cRank=%u) ATLAS-CITADEL(lr=%.4f cRank=%u) ATLAS-RAMPART(lr=%.4f cRank=%u) ATLAS-MERIT(lr=%.4f cRank=%u) ATLAS-STRATA(lr=%.4f cRank=%u) ATLAS-AURORA(lr=%.4f cRank=%u) ATLAS-SEAM(lr=%.4f cRank=%u) ATLAS-QUASAR(lr=%.4f cRank=%u) ATLAS-GEODE(lr=%.4f cRank=%u) ATLAS-ECHO(lr=%.4f scope=%s cadence=%u groups=%u) ATLAS-BIMAP(lr=%.4f scope=%s lowRank=%u cadence=%u) ATLAS-PACT(lr=%.4f lowRank=%u cadence=%u) ATLAS-RACER(lr=%.4f cadence=%u) ATLAS-KRON(lr=%.4f cadence=%u) ATLAS-MUON(lr=%.4f minDim=%u maxAspect=%.2f) ATLAS-MATRA(lr=%.4f cadence=%u trust=%.2f)\n",
	       cfg.token.adamLR, baseTokenLR, sparrowTokenLR, cfg.atlasComplementRank,
	       cfg.atlasSparrowModeRank, cfg.atlasSparrowAutoModeGate,
	       helmTokenLR, cfg.atlasHelmModeRank, cfg.atlasHelmHiddenStackDepth, asterTokenLR,
	       cfg.atlasAsterStateRank, cfg.atlasAsterHiddenStackDepth, aegisTokenLR,
	       cfg.atlasComplementRank, citadelTokenLR, cfg.atlasComplementRank,
	       rampartTokenLR, cfg.atlasComplementRank, meritTokenLR, cfg.atlasComplementRank,
	       strataTokenLR, cfg.atlasComplementRank, auroraTokenLR, cfg.atlasComplementRank,
	       seamTokenLR, cfg.atlasComplementRank, quasarTokenLR, cfg.atlasComplementRank,
	       geodeTokenLR, cfg.atlasComplementRank, echoTokenLR, echo_scope_label(cfg.atlasEchoScope), cfg.atlasEchoMetricCadence, cfg.atlasEchoStructuralGroups, bimapTokenLR, bimap_scope_label(cfg.atlasBiMAPScope), cfg.atlasBiMAPLowRank, cfg.atlasBiMAPFactorCadence,
	       pactTokenLR, cfg.atlasPACTLowRank, cfg.atlasPACTFactorCadence,
	       racerTokenLR, cfg.atlasRACERFactorCadence,
	       kronTokenLR, cfg.atlasKronFactorCadence, muonTokenLR, cfg.atlasMuonMinDim, cfg.atlasMuonMaxAspect,
	       matraTokenLR, cfg.atlasMatraMetricCadence, cfg.atlasMatraTrustRadius);
	printf("ATLAS: rank=%u tSub=%u kappaMax=%.3f sparrow(modeRankCap=%u autoGate=%u memoryScale=%.3f edge=%.3f secondEdge=%.3f secondFrac=%.3f poleMax=%.3f) helm(modeRank=%u hiddenStack=%u memoryScale=%.3f edge=%.3f poleMax=%.3f) aster(stateRank=%u hiddenStack=%u memoryScale=%.3f edge=%.3f poleMax=%.3f) kappa(enabled=%u heads=%u lags=%u rank=%u) aurora(adamwBackbone=%u headGain=%.3f bodyTrust=%.3f) geode(geom=%.3f pred=%.3f) echo(scope=%s geom=%.3f final=%.3f decay=%u cadence=%u trust=%.3f pred=%.3f struct=%.3f groups=%u) bimap(scope=%s lowRank=%u geom=%.3f pred=%.3f cadence=%u) pact(lowRank=%u geom=%.3f pred=%.3f cadence=%u cost=%.4f promote=%.4f demote=%.4f) racer(geom=%.3f pred=%.3f cadence=%u risk=%.3f cost=%.4f promote=%.4f demote=%.4f) kron(geom=%.3f pred=%.3f cadence=%u damping=%.3f) muon(geom=%.3f pred=%.3f maxAspect=%.3f minDim=%u damping=%.3f) matra(geom=%.3f orth=%.3f pred=%.3f trust=%.3f cadence=%u maxAspect=%.3f minDim=%u damping=%.3f)\n",
	       cfg.atlasRank, cfg.atlasTSub, cfg.atlasKappaMax,
	       cfg.atlasSparrowModeRank,
	       cfg.atlasSparrowAutoModeGate,
	       cfg.atlasSparrowMemoryScale, cfg.atlasSparrowEdgeThreshold,
	       cfg.atlasSparrowSecondEdgeThreshold, cfg.atlasSparrowSecondEdgeFraction,
	       cfg.atlasSparrowPoleMax,
	       cfg.atlasHelmModeRank, cfg.atlasHelmHiddenStackDepth,
	       cfg.atlasHelmMemoryScale, cfg.atlasHelmEdgeThreshold, cfg.atlasHelmPoleMax,
	       cfg.atlasAsterStateRank, cfg.atlasAsterHiddenStackDepth,
	       cfg.atlasAsterMemoryScale, cfg.atlasAsterEdgeThreshold, cfg.atlasAsterPoleMax,
	       cfg.atlasKappaEnabled, cfg.atlasKappaHeads, cfg.atlasKappaLagBuckets, cfg.atlasKappaRank,
	       cfg.atlasAuroraAdamwBackbone,
	       cfg.atlasAuroraHeadGain, cfg.atlasAuroraBodyTrustScale,
	       cfg.atlasGeodeGeometryScale, cfg.atlasGeodePredictiveScale,
	       echo_scope_label(cfg.atlasEchoScope), cfg.atlasEchoGeometryScale, cfg.atlasEchoGeometryScaleFinal, cfg.atlasEchoGeometryDecaySteps, cfg.atlasEchoMetricCadence, cfg.atlasEchoTrustScale, cfg.atlasEchoPredictiveScale, cfg.atlasEchoStructuralScale, cfg.atlasEchoStructuralGroups,
	       bimap_scope_label(cfg.atlasBiMAPScope), cfg.atlasBiMAPLowRank, cfg.atlasBiMAPGeometryScale, cfg.atlasBiMAPPredictiveScale, cfg.atlasBiMAPFactorCadence,
	       cfg.atlasPACTLowRank, cfg.atlasPACTGeometryScale, cfg.atlasPACTPredictiveScale, cfg.atlasPACTFactorCadence,
	       cfg.atlasPACTCostScale, cfg.atlasPACTPromoteThreshold, cfg.atlasPACTDemoteThreshold,
	       cfg.atlasRACERGeometryScale, cfg.atlasRACERPredictiveScale, cfg.atlasRACERFactorCadence,
	       cfg.atlasRACERRiskScale, cfg.atlasRACERCostScale, cfg.atlasRACERPromoteThreshold, cfg.atlasRACERDemoteThreshold,
	       cfg.atlasKronGeometryScale, cfg.atlasKronPredictiveScale, cfg.atlasKronFactorCadence, cfg.atlasKronDamping,
	       cfg.atlasMuonGeometryScale, cfg.atlasMuonPredictiveScale, cfg.atlasMuonMaxAspect, cfg.atlasMuonMinDim, cfg.atlasMuonDamping,
	       cfg.atlasMatraGeometryScale, cfg.atlasMatraOrthogonalScale, cfg.atlasMatraPredictiveScale, cfg.atlasMatraTrustRadius,
	       cfg.atlasMatraMetricCadence, cfg.atlasMatraMaxAspect, cfg.atlasMatraMinDim, cfg.atlasMatraDamping);
	printf("\n");
	printf("%-15s  %7s          %10s            %9s           %9s           %9s           %9s         %s\n",
	       "Optimizer", "Train(s)", "Tok/s", "TrainNLL", "TrainPPL", "TestNLL", "TestPPL", "Status");

	const VariantKind variants[] = { VARIANT_ADAMW, VARIANT_ADAMW_GROUP, VARIANT_ATLAS_BASE, VARIANT_ATLAS_SPARROW, VARIANT_ATLAS_HELM, VARIANT_ATLAS_ASTER, VARIANT_ATLAS_AEGIS, VARIANT_ATLAS_CITADEL, VARIANT_ATLAS_RAMPART, VARIANT_ATLAS_MERIT, VARIANT_ATLAS_STRATA, VARIANT_ATLAS_AURORA, VARIANT_ATLAS_SEAM, VARIANT_ATLAS_QUASAR, VARIANT_ATLAS_GEODE, VARIANT_ATLAS_ECHO, VARIANT_ATLAS_BIMAP, VARIANT_ATLAS_PACT, VARIANT_ATLAS_RACER, VARIANT_ATLAS_KRON, VARIANT_ATLAS_MUON, VARIANT_ATLAS_MATRA };
	const size_t variantCount = sizeof(variants) / sizeof(variants[0]);
	bool ranAny = false;
	std::vector<VariantKind> summaryVariants;
	std::vector<Summary> summaries;
	for (size_t v = 0; v < variantCount; ++v)
	{
		if (!variant_matches_selection(cfg.variantSelection, variants[v]))
			continue;
		ranAny = true;
		std::vector<RunResult> runs;
		runs.reserve(cfg.repeats);
		for (unsigned int rep = 0u; rep < cfg.repeats; ++rep)
			runs.push_back(run_token_variant(cfg, data, variants[v], cfg.seed + static_cast<unsigned int>(100u * v) + rep));
		const Summary s = summarize_runs(variant_label(variants[v]), runs);
		print_summary_row(s);
		print_sparrow_usage_row(s);
		print_helm_usage_row(s);
		print_aster_usage_row(s);
		print_aegis_usage_row(s);
		print_citadel_usage_row(s);
		print_rampart_usage_row(s);
		print_merit_usage_row(s);
		print_strata_usage_row(s);
		print_transformer_gap_row(s);
		print_context_family_diag_row(s);
		summaryVariants.push_back(variants[v]);
		summaries.push_back(s);
	}
	if (!ranAny)
	{
		printf("No selected optimizer variants are supported for this case.\n\n");
		return false;
	}
	if ((documentCase || corpusLargeCase) && !summaries.empty())
	{
		int adamwIndex = -1;
		int baseIndex = -1;
		int auroraIndex = -1;
		int geodeIndex = -1;
		int echoIndex = -1;
		int bimapIndex = -1;
		int pactIndex = -1;
		int racerIndex = -1;
		int kronIndex = -1;
		int muonIndex = -1;
		int matraIndex = -1;
		for (size_t i = 0; i < summaryVariants.size(); ++i)
		{
			if (summaryVariants[i] == VARIANT_ADAMW)
				adamwIndex = static_cast<int>(i);
			else if (summaryVariants[i] == VARIANT_ATLAS_BASE)
				baseIndex = static_cast<int>(i);
			else if (summaryVariants[i] == VARIANT_ATLAS_AURORA)
				auroraIndex = static_cast<int>(i);
			else if (summaryVariants[i] == VARIANT_ATLAS_GEODE)
				geodeIndex = static_cast<int>(i);
			else if (summaryVariants[i] == VARIANT_ATLAS_ECHO)
				echoIndex = static_cast<int>(i);
			else if (summaryVariants[i] == VARIANT_ATLAS_BIMAP)
				bimapIndex = static_cast<int>(i);
			else if (summaryVariants[i] == VARIANT_ATLAS_PACT)
				pactIndex = static_cast<int>(i);
			else if (summaryVariants[i] == VARIANT_ATLAS_RACER)
				racerIndex = static_cast<int>(i);
			else if (summaryVariants[i] == VARIANT_ATLAS_KRON)
				kronIndex = static_cast<int>(i);
			else if (summaryVariants[i] == VARIANT_ATLAS_MUON)
				muonIndex = static_cast<int>(i);
			else if (summaryVariants[i] == VARIANT_ATLAS_MATRA)
				matraIndex = static_cast<int>(i);
		}
		if (adamwIndex >= 0 && (baseIndex >= 0 || auroraIndex >= 0 || geodeIndex >= 0 || echoIndex >= 0 || bimapIndex >= 0 || pactIndex >= 0 || racerIndex >= 0 || kronIndex >= 0 || muonIndex >= 0 || matraIndex >= 0))
		{
			printf("  AdamW gap comparison:\n");
			if (baseIndex >= 0)
				print_transformer_gap_compare_row(summaries[static_cast<size_t>(adamwIndex)],
				                                 summaries[static_cast<size_t>(baseIndex)]);
			if (auroraIndex >= 0)
				print_transformer_gap_compare_row(summaries[static_cast<size_t>(adamwIndex)],
				                                 summaries[static_cast<size_t>(auroraIndex)]);
			if (geodeIndex >= 0)
				print_transformer_gap_compare_row(summaries[static_cast<size_t>(adamwIndex)],
				                                 summaries[static_cast<size_t>(geodeIndex)]);
			if (echoIndex >= 0)
				print_transformer_gap_compare_row(summaries[static_cast<size_t>(adamwIndex)],
				                                 summaries[static_cast<size_t>(echoIndex)]);
			if (bimapIndex >= 0)
				print_transformer_gap_compare_row(summaries[static_cast<size_t>(adamwIndex)],
				                                 summaries[static_cast<size_t>(bimapIndex)]);
			if (pactIndex >= 0)
				print_transformer_gap_compare_row(summaries[static_cast<size_t>(adamwIndex)],
				                                 summaries[static_cast<size_t>(pactIndex)]);
			if (racerIndex >= 0)
				print_transformer_gap_compare_row(summaries[static_cast<size_t>(adamwIndex)],
				                                 summaries[static_cast<size_t>(racerIndex)]);
			if (kronIndex >= 0)
				print_transformer_gap_compare_row(summaries[static_cast<size_t>(adamwIndex)],
				                                 summaries[static_cast<size_t>(kronIndex)]);
			if (muonIndex >= 0)
				print_transformer_gap_compare_row(summaries[static_cast<size_t>(adamwIndex)],
				                                 summaries[static_cast<size_t>(muonIndex)]);
			if (matraIndex >= 0)
				print_transformer_gap_compare_row(summaries[static_cast<size_t>(adamwIndex)],
				                                 summaries[static_cast<size_t>(matraIndex)]);
		}
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
	printf("Optimizers: AdamW(lr=%.4f) ATLAS-BSRP(lr=%.4f cRank=0) ATLAS-SPARROW(lr=%.4f cRank=%u modeRankCap=%u autoGate=%u) ATLAS-HELM(lr=%.4f) ATLAS-ASTER(lr=%.4f) ATLAS-AEGIS(lr=%.4f cRank=%u) ATLAS-CITADEL(lr=%.4f cRank=%u) ATLAS-RAMPART(lr=%.4f cRank=%u) ATLAS-MERIT(lr=%.4f cRank=%u) ATLAS-STRATA(lr=%.4f cRank=%u) ATLAS-AURORA(lr=%.4f cRank=%u) ATLAS-SEAM(lr=%.4f cRank=%u) ATLAS-QUASAR(lr=%.4f cRank=%u)\n",
	       cfg.latent.adamLR, cfg.latent.atlasLR, cfg.latent.atlasLR, cfg.atlasComplementRank,
	       cfg.atlasSparrowModeRank, cfg.atlasSparrowAutoModeGate, cfg.latent.atlasLR, cfg.latent.atlasLR,
	       cfg.latent.atlasLR, cfg.atlasComplementRank, cfg.latent.atlasLR, cfg.atlasComplementRank,
	       cfg.latent.atlasLR, cfg.atlasComplementRank, cfg.latent.atlasLR, cfg.atlasComplementRank,
	       cfg.latent.atlasLR, cfg.atlasComplementRank, cfg.latent.atlasLR, cfg.atlasComplementRank,
	       cfg.latent.atlasLR, cfg.atlasComplementRank, cfg.latent.atlasLR, cfg.atlasComplementRank);
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

	const VariantKind variants[] = { VARIANT_ADAMW, VARIANT_ATLAS_BASE, VARIANT_ATLAS_SPARROW, VARIANT_ATLAS_HELM, VARIANT_ATLAS_ASTER, VARIANT_ATLAS_AEGIS, VARIANT_ATLAS_CITADEL, VARIANT_ATLAS_RAMPART, VARIANT_ATLAS_MERIT, VARIANT_ATLAS_STRATA, VARIANT_ATLAS_AURORA, VARIANT_ATLAS_SEAM, VARIANT_ATLAS_QUASAR };
	const size_t variantCount = sizeof(variants) / sizeof(variants[0]);
	bool ranAny = false;
	for (size_t v = 0; v < variantCount; ++v)
	{
		if (!variant_matches_selection(cfg.variantSelection, variants[v]))
			continue;
		ranAny = true;
		std::vector<RunResult> runs;
		runs.reserve(cfg.repeats);
		for (unsigned int rep = 0u; rep < cfg.repeats; ++rep)
			runs.push_back(run_latent_variant(cfg, data, variants[v], cfg.seed + static_cast<unsigned int>(2000u + 100u * v) + rep));
		const Summary s = summarize_runs(variant_label(variants[v]), runs);
		print_summary_row(s);
		print_sparrow_usage_row(s);
		print_helm_usage_row(s);
		print_aster_usage_row(s);
		print_aegis_usage_row(s);
		print_citadel_usage_row(s);
		print_rampart_usage_row(s);
		print_merit_usage_row(s);
		print_strata_usage_row(s);
	}
	if (!ranAny)
	{
		printf("No selected optimizer variants are supported for this case.\n\n");
		return false;
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
	printf("Optimizers: AdamW(lr=%.4f) ATLAS-BSRP(lr=%.4f cRank=0) ATLAS-SPARROW(lr=%.4f cRank=%u modeRankCap=%u autoGate=%u) ATLAS-HELM(lr=%.4f) ATLAS-ASTER(lr=%.4f) ATLAS-AEGIS(lr=%.4f cRank=%u) ATLAS-CITADEL(lr=%.4f cRank=%u) ATLAS-RAMPART(lr=%.4f cRank=%u) ATLAS-MERIT(lr=%.4f cRank=%u) ATLAS-STRATA(lr=%.4f cRank=%u) ATLAS-AURORA(lr=%.4f cRank=%u) ATLAS-SEAM(lr=%.4f cRank=%u) ATLAS-QUASAR(lr=%.4f cRank=%u)\n",
	       cfg.latent.adamLR, cfg.latent.atlasLR, cfg.latent.atlasLR, cfg.atlasComplementRank,
	       cfg.atlasSparrowModeRank, cfg.atlasSparrowAutoModeGate, cfg.latent.atlasLR, cfg.latent.atlasLR,
	       cfg.latent.atlasLR, cfg.atlasComplementRank, cfg.latent.atlasLR, cfg.atlasComplementRank,
	       cfg.latent.atlasLR, cfg.atlasComplementRank, cfg.latent.atlasLR, cfg.atlasComplementRank,
	       cfg.latent.atlasLR, cfg.atlasComplementRank, cfg.latent.atlasLR, cfg.atlasComplementRank,
	       cfg.latent.atlasLR, cfg.atlasComplementRank, cfg.latent.atlasLR, cfg.atlasComplementRank);
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

	const VariantKind variants[] = { VARIANT_ADAMW, VARIANT_ATLAS_BASE, VARIANT_ATLAS_SPARROW, VARIANT_ATLAS_HELM, VARIANT_ATLAS_ASTER, VARIANT_ATLAS_AEGIS, VARIANT_ATLAS_CITADEL, VARIANT_ATLAS_RAMPART, VARIANT_ATLAS_MERIT, VARIANT_ATLAS_STRATA, VARIANT_ATLAS_AURORA, VARIANT_ATLAS_SEAM, VARIANT_ATLAS_QUASAR };
	const size_t variantCount = sizeof(variants) / sizeof(variants[0]);
	bool ranAny = false;
	for (size_t v = 0; v < variantCount; ++v)
	{
		if (!variant_matches_selection(cfg.variantSelection, variants[v]))
			continue;
		ranAny = true;
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
		print_aegis_usage_row(s);
		print_citadel_usage_row(s);
		print_rampart_usage_row(s);
		print_merit_usage_row(s);
		print_strata_usage_row(s);
	}
	if (!ranAny)
	{
		printf("No selected optimizer variants are supported for this case.\n\n");
		return false;
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
	printf("Optimizers: AdamW(lr=%.4f) ATLAS-BSRP(lr=%.4f cRank=0) ATLAS-SPARROW(lr=%.4f cRank=%u modeRankCap=%u autoGate=%u) ATLAS-AEGIS(lr=%.4f cRank=%u) ATLAS-CITADEL(lr=%.4f cRank=%u) ATLAS-RAMPART(lr=%.4f cRank=%u) ATLAS-MERIT(lr=%.4f cRank=%u) ATLAS-STRATA(lr=%.4f cRank=%u) ATLAS-AURORA(lr=%.4f cRank=%u) ATLAS-SEAM(lr=%.4f cRank=%u) ATLAS-QUASAR(lr=%.4f cRank=%u)\n",
	       cfg.teacher.adamLR, cfg.teacher.atlasLR, cfg.teacher.atlasLR, cfg.atlasComplementRank,
	       cfg.atlasSparrowModeRank, cfg.atlasSparrowAutoModeGate, cfg.teacher.atlasLR,
	       cfg.atlasComplementRank, cfg.teacher.atlasLR, cfg.atlasComplementRank,
	       cfg.teacher.atlasLR, cfg.atlasComplementRank, cfg.teacher.atlasLR, cfg.atlasComplementRank,
	       cfg.teacher.atlasLR, cfg.atlasComplementRank, cfg.teacher.atlasLR, cfg.atlasComplementRank,
	       cfg.teacher.atlasLR, cfg.atlasComplementRank, cfg.teacher.atlasLR, cfg.atlasComplementRank);
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

	const VariantKind variants[] = { VARIANT_ADAMW, VARIANT_ATLAS_BASE, VARIANT_ATLAS_SPARROW, VARIANT_ATLAS_AEGIS, VARIANT_ATLAS_CITADEL, VARIANT_ATLAS_RAMPART, VARIANT_ATLAS_MERIT, VARIANT_ATLAS_STRATA, VARIANT_ATLAS_AURORA, VARIANT_ATLAS_SEAM, VARIANT_ATLAS_QUASAR };
	const size_t variantCount = sizeof(variants) / sizeof(variants[0]);
	bool ranAny = false;
	for (size_t v = 0; v < variantCount; ++v)
	{
		if (!variant_matches_selection(cfg.variantSelection, variants[v]))
			continue;
		ranAny = true;
		std::vector<RunResult> runs;
		runs.reserve(cfg.repeats);
		for (unsigned int rep = 0u; rep < cfg.repeats; ++rep)
			runs.push_back(run_teacher_variant(cfg, data, variants[v], cfg.seed + static_cast<unsigned int>(1000u + 100u * v) + rep));
		const Summary s = summarize_runs(variant_label(variants[v]), runs);
		print_summary_row(s);
		print_sparrow_usage_row(s);
		print_aster_usage_row(s);
		print_aegis_usage_row(s);
		print_citadel_usage_row(s);
		print_rampart_usage_row(s);
		print_merit_usage_row(s);
		print_strata_usage_row(s);
	}
	if (!ranAny)
	{
		printf("No selected optimizer variants are supported for this case.\n\n");
		return false;
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

	if (cfg.mode == MODE_ALL || cfg.mode == MODE_TOKEN_LM || cfg.mode == MODE_TOKEN_LM_LARGE
	    || cfg.mode == MODE_TOKEN_LM_CONTEXT || cfg.mode == MODE_TOKEN_LM_CONTEXT_LARGE
	    || cfg.mode == MODE_TOKEN_LM_DOCUMENT || cfg.mode == MODE_TOKEN_LM_CORPUS
	    || cfg.mode == MODE_TOKEN_LM_CORPUS_LARGE)
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
