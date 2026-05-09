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
#ifndef _NNETWORK
#define _NNETWORK

#include "Backend/Database/GPointer.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "../State/Terminator.h"
#include "../GMath/cmatrix.h"
#include "../Structure/nninfo.h" // ensure NNInfo is a complete type for ownedSkeleton deletion
#include "../rng.h"
#include "training_callbacks.h"
#include "training_config.h"
#include "atlas_optimizer.h"
#include "vesta_optimizer.h"
#include "helios_optimizer.h"
#include "../nnetwork_status.h"
#include "bayes.h"
#include "bayes-optimizer.h"
#include "transformer_ops.h"
#include "transformer_types.h"
#include "aligned_allocator.h"
#include "cuda/gpu_dispatch.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <stdint.h>

#ifdef GLADES_HAVE_CUDA
#include "cuda/gpu_device.h"
#include "cuda/gpu_blas.h"
#include "cuda/gpu_transformer_state.h"
#include "cuda/gpu_dff_state.h"
#include "cuda/gpu_rnn_state.h"
#include "cuda/gpu_cnn_state.h"
#include "cuda/gpu_init.h"
#include "cuda/gpu_face.h"
#endif

// Concurrency primitives:
// Prefer standard C++ atomics when available; fall back to legacy builtins otherwise.
#if __cplusplus >= 201103L
#include <atomic>
#define GLADES_HAVE_STD_ATOMICS 1
#else
#define GLADES_HAVE_STD_ATOMICS 0
#endif

class Point2;

namespace shmea {
class GTable;
class GLogger;
};

namespace GNet {
class GServer;
class Connection;
};

void GANUnitTest();

namespace glades {

class TransformerServingLayer;

class DataInput;
class CMatrix;
class MetaNetwork;
class TrainingCore;
class Trainer;
class GAN;
struct GradientBuffer;
struct DeconvScratchArena;

class NNetwork
{
private:
	friend MetaNetwork;
	friend TrainingCore;
	friend Trainer;
	friend GAN;
	friend GradientBuffer;
	friend DeconvScratchArena;
	friend void ::GANUnitTest();

public:
	struct TrainerRunDiagnostics
	{
		uint64_t totalRunAttempts;
		uint64_t totalRunSuccesses;
		uint64_t totalRunFailures;
		uint64_t totalPreflightFailures;
		uint64_t totalNullSkeletonFailures;
		uint64_t totalNullDataFailures;
		uint64_t totalUnknownRunTypeFailures;
		uint64_t totalEmptyDataFailures;
		uint64_t totalPostBuildEmptyDataFailures;
		uint64_t totalContractFailures;
		uint64_t totalTensorInitFailures;
		int lastRunType;
		int lastNetType;
		bool lastTrainRun;
		bool lastEvalRun;
		bool lastTokenLM;
		bool lastTokenLMInput;
		bool lastSequenceModel;
		bool lastFailureDuringPreflight;
		bool lastFailurePostBuildCheck;
		unsigned int lastDataSize;
		unsigned int lastFeatureCount;
		unsigned int lastOutputSize;
		unsigned int lastExpectedFeatureCount;
		unsigned int lastExpectedOutputSize;
		std::string lastFailureStage;
		NNetworkStatus lastRunStatus;
		NNetworkStatus lastFailureStatus;
		NNetworkStatus lastDataInputStatus;

		TrainerRunDiagnostics()
		    : totalRunAttempts(0ULL),
		      totalRunSuccesses(0ULL),
		      totalRunFailures(0ULL),
		      totalPreflightFailures(0ULL),
		      totalNullSkeletonFailures(0ULL),
		      totalNullDataFailures(0ULL),
		      totalUnknownRunTypeFailures(0ULL),
		      totalEmptyDataFailures(0ULL),
		      totalPostBuildEmptyDataFailures(0ULL),
		      totalContractFailures(0ULL),
		      totalTensorInitFailures(0ULL),
		      lastRunType(-1),
		      lastNetType(-1),
		      lastTrainRun(false),
		      lastEvalRun(false),
		      lastTokenLM(false),
		      lastTokenLMInput(false),
		      lastSequenceModel(false),
		      lastFailureDuringPreflight(false),
		      lastFailurePostBuildCheck(false),
		      lastDataSize(0u),
		      lastFeatureCount(0u),
		      lastOutputSize(0u),
		      lastExpectedFeatureCount(0u),
		      lastExpectedOutputSize(0u),
		      lastFailureStage(),
		      lastRunStatus(NNetworkStatus::OK, std::string()),
		      lastFailureStatus(NNetworkStatus::OK, std::string()),
		      lastDataInputStatus(NNetworkStatus::OK, std::string())
		{
		}
	};

	struct PersistenceDiagnostics
	{
		uint64_t totalPersistenceOps;
		uint64_t totalPersistenceSuccesses;
		uint64_t totalPersistenceFailures;
		uint64_t totalRejectedInputs;
		uint64_t totalPublishFailures;
		uint64_t totalModelSaveAttempts;
		uint64_t totalModelSaveSuccesses;
		uint64_t totalModelSaveFailures;
		uint64_t totalModelPublishFailures;
		uint64_t totalCheckpointSaveAttempts;
		uint64_t totalCheckpointSaveSuccesses;
		uint64_t totalCheckpointSaveFailures;
		uint64_t totalCheckpointPublishFailures;
		uint64_t totalRotateFailures;
		uint64_t totalPublishRenameFailures;
		uint64_t totalManifestWriteFailures;
		uint64_t totalNninfoWriteFailures;
		uint64_t totalWeightsWriteFailures;
		uint64_t totalCheckpointTensorCollectionFailures;
		uint64_t totalCheckpointShardWriteFailures;
		uint64_t totalIntegrityFailures;
		int lastNetType;
		bool lastOperationWasCheckpoint;
		bool lastOperationSucceeded;
		bool lastOperationRejected;
		bool lastRotatedPrevious;
		bool lastTokenizerPresent;
		bool lastIncludeOptimizerState;
		uint64_t lastShardCount;
		uint64_t lastTensorCount;
		uint64_t lastWeightsBytes;
		uint64_t lastMaxShardBytes;
		std::string lastOperation;
		std::string lastName;
		std::string lastStage;
		NNetworkStatus lastStatus;

		PersistenceDiagnostics()
		    : totalPersistenceOps(0ULL),
		      totalPersistenceSuccesses(0ULL),
		      totalPersistenceFailures(0ULL),
		      totalRejectedInputs(0ULL),
		      totalPublishFailures(0ULL),
		      totalModelSaveAttempts(0ULL),
		      totalModelSaveSuccesses(0ULL),
		      totalModelSaveFailures(0ULL),
		      totalModelPublishFailures(0ULL),
		      totalCheckpointSaveAttempts(0ULL),
		      totalCheckpointSaveSuccesses(0ULL),
		      totalCheckpointSaveFailures(0ULL),
		      totalCheckpointPublishFailures(0ULL),
		      totalRotateFailures(0ULL),
		      totalPublishRenameFailures(0ULL),
		      totalManifestWriteFailures(0ULL),
		      totalNninfoWriteFailures(0ULL),
		      totalWeightsWriteFailures(0ULL),
		      totalCheckpointTensorCollectionFailures(0ULL),
		      totalCheckpointShardWriteFailures(0ULL),
		      totalIntegrityFailures(0ULL),
		      lastNetType(-1),
		      lastOperationWasCheckpoint(false),
		      lastOperationSucceeded(false),
		      lastOperationRejected(false),
		      lastRotatedPrevious(false),
		      lastTokenizerPresent(false),
		      lastIncludeOptimizerState(false),
		      lastShardCount(0ULL),
		      lastTensorCount(0ULL),
		      lastWeightsBytes(0ULL),
		      lastMaxShardBytes(0ULL),
		      lastOperation(),
		      lastName(),
		      lastStage(),
		      lastStatus(NNetworkStatus::OK, std::string())
		{
		}
	};

	struct AtlasRuntimeDiagnostics
	{
		unsigned int atlasMatrices;
		unsigned int sparrowMatrices;
		unsigned int sparrowMode2Matrices;
		double sparrowMeanActiveModes;
		double sparrowMode2Fraction;
		double sparrowMeanEdge;
		double sparrowMeanSecondEdge;
		double sparrowMeanSecondEdgeRatio;
		double sparrowMeanMemoryGain;
		double sparrowMeanHorizontalRatio;
		unsigned int helmMatrices;
		unsigned int helmMode2Matrices;
		double helmMeanActiveModes;
		double helmMode2Fraction;
		double helmMeanEdge;
		double helmMeanSecondEdge;
		double helmMeanSecondEdgeRatio;
		double helmMeanSigma;
		double helmMeanPredR2;
		double helmMeanMemoryGain;
		double helmMeanPole;
		unsigned int asterMatrices;
		unsigned int asterMode2Matrices;
		double asterMeanActiveModes;
		double asterMode2Fraction;
		double asterMeanEdge;
		double asterMeanSecondEdge;
		double asterMeanSecondEdgeRatio;
		double asterMeanSigma;
		double asterMeanPredR2;
		double asterMeanMemoryGain;
		double asterMeanPole;
		double asterMeanBoundaryMs;
		double asterMeanSetupMs;
		double asterMeanTransportMs;
		double asterMeanTransferFitMs;
		double asterMeanStateFitMs;
		double asterMeanInnovationFitMs;
		double asterMeanApplyMs;
		unsigned int aegisMatrices;
		double aegisMeanLambdaSpatial;
		double aegisMeanLambdaPredictive;
		double aegisMeanLambdaOutput;
		double aegisMeanPredictivePredicted;
		double aegisMeanPredictiveRealized;
		double aegisMeanOutputPredicted;
		double aegisMeanOutputRealized;
		double aegisMeanPredictiveError;
		double aegisMeanOutputError;
		double aegisMeanChannelDisagreement;
		unsigned int citadelMatrices;
		double citadelMeanAnchor;
		double citadelMeanHardRegimeMass;
		double citadelMeanSparrowTrust;
		unsigned int rampartMatrices;
		double rampartMeanTau;
		double rampartMeanBudget;
		double rampartMeanCovariance;
		double rampartMeanSparrowTrust;
		unsigned int meritMatrices;
		double meritMeanTau;
		double meritMeanBudget;
		double meritMeanCovariance;
		double meritMeanSparrowTrust;
		double meritMeanGeometryTrust;
		unsigned int strataMatrices;
		double strataMeanNullMode;
		double strataMeanPredictiveMode;
		double strataMeanOutputMode;
		double strataMeanCoupledMode;
		double strataMeanBudget;
		double strataMeanNullBenefit;
		double strataMeanPredictiveBenefit;
		double strataMeanOutputBenefit;
		double strataMeanCoupledBenefit;
		double strataMeanSelectedExcess;
		double strataMeanSwitchRate;
		unsigned int transformerGapBatches;
		double transformerMeanInputUpdateNorm;
		std::vector<double> transformerMeanBlockUpdateNorms;
		double transformerMeanFinalNormUpdateNorm;
		double transformerMeanHeadUpdateNorm;
		double transformerMeanHeadShare;
		double transformerMeanNonHeadShare;
		double transformerMeanApplyMs;
		unsigned int transformerMarginSnapshots;
		double transformerMeanTargetMargin;
		double transformerMeanHardNegativeLogit;

		AtlasRuntimeDiagnostics()
		    : atlasMatrices(0u),
		      sparrowMatrices(0u),
		      sparrowMode2Matrices(0u),
		      sparrowMeanActiveModes(0.0),
		      sparrowMode2Fraction(0.0),
		      sparrowMeanEdge(0.0),
		      sparrowMeanSecondEdge(0.0),
		      sparrowMeanSecondEdgeRatio(0.0),
		      sparrowMeanMemoryGain(0.0),
		      sparrowMeanHorizontalRatio(0.0),
		      helmMatrices(0u),
		      helmMode2Matrices(0u),
		      helmMeanActiveModes(0.0),
		      helmMode2Fraction(0.0),
		      helmMeanEdge(0.0),
		      helmMeanSecondEdge(0.0),
		      helmMeanSecondEdgeRatio(0.0),
		      helmMeanSigma(0.0),
		      helmMeanPredR2(0.0),
		      helmMeanMemoryGain(0.0),
		      helmMeanPole(0.0),
		      asterMatrices(0u),
		      asterMode2Matrices(0u),
		      asterMeanActiveModes(0.0),
		      asterMode2Fraction(0.0),
		      asterMeanEdge(0.0),
		      asterMeanSecondEdge(0.0),
		      asterMeanSecondEdgeRatio(0.0),
		      asterMeanSigma(0.0),
		      asterMeanPredR2(0.0),
		      asterMeanMemoryGain(0.0),
		      asterMeanPole(0.0),
		      asterMeanBoundaryMs(0.0),
		      asterMeanSetupMs(0.0),
		      asterMeanTransportMs(0.0),
		      asterMeanTransferFitMs(0.0),
		      asterMeanStateFitMs(0.0),
		      asterMeanInnovationFitMs(0.0),
		      asterMeanApplyMs(0.0),
		      aegisMatrices(0u),
		      aegisMeanLambdaSpatial(0.0),
		      aegisMeanLambdaPredictive(0.0),
		      aegisMeanLambdaOutput(0.0),
		      aegisMeanPredictivePredicted(0.0),
		      aegisMeanPredictiveRealized(0.0),
		      aegisMeanOutputPredicted(0.0),
		      aegisMeanOutputRealized(0.0),
		      aegisMeanPredictiveError(0.0),
		      aegisMeanOutputError(0.0),
		      aegisMeanChannelDisagreement(0.0),
		      citadelMatrices(0u),
		      citadelMeanAnchor(0.0),
		      citadelMeanHardRegimeMass(0.0),
		      citadelMeanSparrowTrust(0.0),
		      rampartMatrices(0u),
		      rampartMeanTau(0.0),
		      rampartMeanBudget(0.0),
		      rampartMeanCovariance(0.0),
		      rampartMeanSparrowTrust(0.0),
		      meritMatrices(0u),
		      meritMeanTau(0.0),
		      meritMeanBudget(0.0),
		      meritMeanCovariance(0.0),
		      meritMeanSparrowTrust(0.0),
		      meritMeanGeometryTrust(0.0),
		      strataMatrices(0u),
		      strataMeanNullMode(0.0),
		      strataMeanPredictiveMode(0.0),
		      strataMeanOutputMode(0.0),
		      strataMeanCoupledMode(0.0),
		      strataMeanBudget(0.0),
		      strataMeanNullBenefit(0.0),
		      strataMeanPredictiveBenefit(0.0),
		      strataMeanOutputBenefit(0.0),
		      strataMeanCoupledBenefit(0.0),
		      strataMeanSelectedExcess(0.0),
		      strataMeanSwitchRate(0.0),
		      transformerGapBatches(0u),
		      transformerMeanInputUpdateNorm(0.0),
		      transformerMeanBlockUpdateNorms(),
		      transformerMeanFinalNormUpdateNorm(0.0),
		      transformerMeanHeadUpdateNorm(0.0),
		      transformerMeanHeadShare(0.0),
		      transformerMeanNonHeadShare(0.0),
		      transformerMeanApplyMs(0.0),
		      transformerMarginSnapshots(0u),
		      transformerMeanTargetMargin(0.0),
		      transformerMeanHardNegativeLogit(0.0)
		{
		}
	};

	struct TransformerGroupedParameterSnapshot
	{
		bool valid;
		std::vector<float> inputGroup;
		std::vector< std::vector<float> > blockGroups;
		std::vector<float> finalNormGroup;
		std::vector<float> headGroup;

		TransformerGroupedParameterSnapshot()
		    : valid(false)
		{
		}
	};

private:

	// Tensor-based DFF training state.
	//
	// This is a contiguous-buffer rewrite of the historical (graph-based) training core,
	// but implemented purely in packed vectors/matrices for cache-friendly execution.
	struct TensorDFFState
	{
		struct HelmState
		{
			bool initialized;
			unsigned int rawHiddenDim;
			unsigned int hiddenDim;
			unsigned int outputDim;
			unsigned int hiddenStackDepth;
			unsigned int modeRank;
			std::vector<unsigned int> hiddenLayerActivationIndices;
			std::vector<unsigned int> hiddenLayerOffsets;
			std::vector<unsigned int> hiddenLayerSizes;
			std::vector<float> prevHiddenMean;
			std::vector<float> prevResidualMean;
			std::vector<float> hiddenVar;
			std::vector<float> residualVar;
			std::vector<float> crossCov;
			std::vector<float> sigma;
			std::vector<float> leftMode;
			std::vector<float> rightMode;
			std::vector<float> batchHiddenSum;
			std::vector<float> batchHiddenSqSum;
			std::vector<float> batchResidualSum;
			std::vector<float> batchResidualSqSum;
			std::vector<float> latent;
			std::vector<float> poleNumer;
			std::vector<float> poleDenom;
			std::vector<float> pole;
			unsigned int lastActiveModes;
			float lastEdge;
			float lastSecondEdge;
			float lastSecondEdgeRatio;
			float lastSigma;
			float lastPredR2;
			float lastMemoryGain;

			HelmState()
			    : initialized(false),
			      rawHiddenDim(0u),
			      hiddenDim(0u),
			      outputDim(0u),
			      hiddenStackDepth(0u),
			      modeRank(0u),
			      lastActiveModes(0u),
			      lastEdge(0.0f),
			      lastSecondEdge(0.0f),
			      lastSecondEdgeRatio(0.0f),
			      lastSigma(0.0f),
			      lastPredR2(0.0f),
			      lastMemoryGain(0.0f)
			{
			}

			void reset()
			{
				initialized = false;
				rawHiddenDim = 0u;
				hiddenDim = 0u;
				outputDim = 0u;
				hiddenStackDepth = 0u;
				modeRank = 0u;
				hiddenLayerActivationIndices.clear();
				hiddenLayerOffsets.clear();
				hiddenLayerSizes.clear();
				prevHiddenMean.clear();
				prevResidualMean.clear();
				hiddenVar.clear();
				residualVar.clear();
				crossCov.clear();
				sigma.clear();
				leftMode.clear();
				rightMode.clear();
				batchHiddenSum.clear();
				batchHiddenSqSum.clear();
				batchResidualSum.clear();
				batchResidualSqSum.clear();
				latent.clear();
				poleNumer.clear();
				poleDenom.clear();
				pole.clear();
				lastActiveModes = 0u;
				lastEdge = 0.0f;
				lastSecondEdge = 0.0f;
				lastSecondEdgeRatio = 0.0f;
				lastSigma = 0.0f;
				lastPredR2 = 0.0f;
				lastMemoryGain = 0.0f;
			}
		};

		struct AsterState
		{
			bool initialized;
			unsigned int rawHiddenDim;
			unsigned int controlDim;
			unsigned int outputDim;
			unsigned int hiddenStackDepth;
			unsigned int stateRank;
			std::vector<unsigned int> hiddenLayerActivationIndices;
			std::vector<unsigned int> hiddenLayerOffsets;
			std::vector<unsigned int> hiddenLayerSizes;
			std::vector<float> prevControlMean;
			std::vector<float> prevResidualMean;
			std::vector<float> controlVar;
			std::vector<float> residualVar;
			std::vector<float> pastCov;
			std::vector<float> crossCov;
			std::vector<float> theta;
			std::vector<float> statePastCov;
			std::vector<float> stateCrossCov;
			std::vector<float> innovationCov;
			std::vector<float> innovationCross;
			std::vector<float> sigma;
			std::vector<float> leftMode;
			std::vector<float> rightMode;
			std::vector<float> batchHiddenSum;
			std::vector<float> batchHiddenSqSum;
			std::vector<float> batchResidualSum;
			std::vector<float> batchResidualSqSum;
			std::vector<float> latent;
			std::vector<float> poleNumer;
			std::vector<float> poleDenom;
			std::vector<float> pole;
			unsigned int lastActiveModes;
			float lastEdge;
			float lastSecondEdge;
			float lastSecondEdgeRatio;
			float lastSigma;
			float lastPredR2;
			float lastMemoryGain;
			float aegisPredictiveErrorEma;
			float aegisOutputErrorEma;
			float aegisPrevPredictiveScore;
			float aegisPrevOutputScore;
			float aegisLastLambdaSpatial;
			float aegisLastLambdaPredictive;
			float aegisLastLambdaOutput;
			float aegisLastPredictivePredicted;
			float aegisLastPredictiveRealized;
			float aegisLastOutputPredicted;
			float aegisLastOutputRealized;
			float aegisLastChannelDisagreement;
			float citadelPredictiveTrustEma;
			float citadelOutputTrustEma;
			float citadelLastAnchor;
			float citadelLastHardRegimeMass;
			float citadelLastSparrowTrust;
			float rampartLastTau;
			float rampartLastBudget;
			float rampartLastCovariance;
			float rampartLastSparrowTrust;
			float meritLastTau;
			float meritLastBudget;
			float meritLastCovariance;
			float meritLastSparrowTrust;
			float meritLastGeometryTrust;
			float strataLastNullMode;
			float strataLastPredictiveMode;
			float strataLastOutputMode;
			float strataLastCoupledMode;
			float strataLastBudget;
			float strataNullBenefitEma;
			float strataPredictiveBenefitEma;
			float strataOutputBenefitEma;
			float strataCoupledBenefitEma;
			float strataLastNullBenefit;
			float strataLastPredictiveBenefit;
			float strataLastOutputBenefit;
			float strataLastCoupledBenefit;
			float strataLastSelectedExcess;
			float strataLastSwitchRate;
			unsigned long long timingBoundaryCount;
			double totalBoundaryNs;
			double totalSetupNs;
			double totalTransportNs;
			double totalTransferFitNs;
			double totalStateFitNs;
			double totalInnovationFitNs;
			double totalApplyNs;

			AsterState()
			    : initialized(false),
			      rawHiddenDim(0u),
			      controlDim(0u),
			      outputDim(0u),
			      hiddenStackDepth(0u),
			      stateRank(0u),
			      lastActiveModes(0u),
			      lastEdge(0.0f),
			      lastSecondEdge(0.0f),
			      lastSecondEdgeRatio(0.0f),
			      lastSigma(0.0f),
			      lastPredR2(0.0f),
			      lastMemoryGain(0.0f),
			      aegisPredictiveErrorEma(0.0f),
			      aegisOutputErrorEma(0.0f),
			      aegisPrevPredictiveScore(0.0f),
			      aegisPrevOutputScore(0.0f),
			      aegisLastLambdaSpatial(0.0f),
			      aegisLastLambdaPredictive(0.0f),
			      aegisLastLambdaOutput(0.0f),
			      aegisLastPredictivePredicted(0.0f),
			      aegisLastPredictiveRealized(0.0f),
			      aegisLastOutputPredicted(0.0f),
			      aegisLastOutputRealized(0.0f),
			      aegisLastChannelDisagreement(0.0f),
			      citadelPredictiveTrustEma(0.0f),
			      citadelOutputTrustEma(0.0f),
			      citadelLastAnchor(0.0f),
			      citadelLastHardRegimeMass(0.0f),
			      citadelLastSparrowTrust(1.0f),
			      rampartLastTau(0.0f),
			      rampartLastBudget(0.0f),
			      rampartLastCovariance(0.0f),
			      rampartLastSparrowTrust(1.0f),
			      meritLastTau(0.0f),
			      meritLastBudget(0.0f),
			      meritLastCovariance(0.0f),
			      meritLastSparrowTrust(1.0f),
			      meritLastGeometryTrust(0.0f),
			      strataLastNullMode(1.0f),
			      strataLastPredictiveMode(0.0f),
			      strataLastOutputMode(0.0f),
			      strataLastCoupledMode(0.0f),
			      strataLastBudget(0.0f),
			      strataNullBenefitEma(0.0f),
			      strataPredictiveBenefitEma(0.0f),
			      strataOutputBenefitEma(0.0f),
			      strataCoupledBenefitEma(0.0f),
			      strataLastNullBenefit(0.0f),
			      strataLastPredictiveBenefit(0.0f),
			      strataLastOutputBenefit(0.0f),
			      strataLastCoupledBenefit(0.0f),
			      strataLastSelectedExcess(0.0f),
			      strataLastSwitchRate(0.0f),
			      timingBoundaryCount(0ULL),
			      totalBoundaryNs(0.0),
			      totalSetupNs(0.0),
			      totalTransportNs(0.0),
			      totalTransferFitNs(0.0),
			      totalStateFitNs(0.0),
			      totalInnovationFitNs(0.0),
			      totalApplyNs(0.0)
			{
			}

			void reset()
			{
				initialized = false;
				rawHiddenDim = 0u;
				controlDim = 0u;
				outputDim = 0u;
				hiddenStackDepth = 0u;
				stateRank = 0u;
				hiddenLayerActivationIndices.clear();
				hiddenLayerOffsets.clear();
				hiddenLayerSizes.clear();
				prevControlMean.clear();
				prevResidualMean.clear();
				controlVar.clear();
				residualVar.clear();
				pastCov.clear();
				crossCov.clear();
				theta.clear();
				statePastCov.clear();
				stateCrossCov.clear();
				innovationCov.clear();
				innovationCross.clear();
				sigma.clear();
				leftMode.clear();
				rightMode.clear();
				batchHiddenSum.clear();
				batchHiddenSqSum.clear();
				batchResidualSum.clear();
				batchResidualSqSum.clear();
				latent.clear();
				poleNumer.clear();
				poleDenom.clear();
				pole.clear();
				lastActiveModes = 0u;
				lastEdge = 0.0f;
				lastSecondEdge = 0.0f;
				lastSecondEdgeRatio = 0.0f;
				lastSigma = 0.0f;
				lastPredR2 = 0.0f;
				lastMemoryGain = 0.0f;
				aegisPredictiveErrorEma = 0.0f;
				aegisOutputErrorEma = 0.0f;
				aegisPrevPredictiveScore = 0.0f;
				aegisPrevOutputScore = 0.0f;
				aegisLastLambdaSpatial = 0.0f;
				aegisLastLambdaPredictive = 0.0f;
				aegisLastLambdaOutput = 0.0f;
				aegisLastPredictivePredicted = 0.0f;
				aegisLastPredictiveRealized = 0.0f;
				aegisLastOutputPredicted = 0.0f;
				aegisLastOutputRealized = 0.0f;
				aegisLastChannelDisagreement = 0.0f;
				citadelPredictiveTrustEma = 0.0f;
				citadelOutputTrustEma = 0.0f;
				citadelLastAnchor = 0.0f;
				citadelLastHardRegimeMass = 0.0f;
				citadelLastSparrowTrust = 1.0f;
				rampartLastTau = 0.0f;
				rampartLastBudget = 0.0f;
				rampartLastCovariance = 0.0f;
				rampartLastSparrowTrust = 1.0f;
				meritLastTau = 0.0f;
				meritLastBudget = 0.0f;
				meritLastCovariance = 0.0f;
				meritLastSparrowTrust = 1.0f;
				meritLastGeometryTrust = 0.0f;
				strataLastNullMode = 1.0f;
				strataLastPredictiveMode = 0.0f;
				strataLastOutputMode = 0.0f;
				strataLastCoupledMode = 0.0f;
				strataLastBudget = 0.0f;
				strataNullBenefitEma = 0.0f;
				strataPredictiveBenefitEma = 0.0f;
				strataOutputBenefitEma = 0.0f;
				strataCoupledBenefitEma = 0.0f;
				strataLastNullBenefit = 0.0f;
				strataLastPredictiveBenefit = 0.0f;
				strataLastOutputBenefit = 0.0f;
				strataLastCoupledBenefit = 0.0f;
				strataLastSelectedExcess = 0.0f;
				strataLastSwitchRate = 0.0f;
				timingBoundaryCount = 0ULL;
				totalBoundaryNs = 0.0;
				totalSetupNs = 0.0;
				totalTransportNs = 0.0;
				totalTransferFitNs = 0.0;
				totalStateFitNs = 0.0;
				totalInnovationFitNs = 0.0;
				totalApplyNs = 0.0;
			}
		};

		bool initialized;
		// Layer sizes including input and output: [in, h1, ..., hH, out]
		std::vector<unsigned int> sizes;

		// Per-transition weights (mapping sizes[t] -> sizes[t+1]) and optim state.
		// W is row-major [out][in].
		struct Transition
		{
			unsigned int in;
			unsigned int out;
			std::vector<float> W;
			std::vector<float> vW; // momentum/velocity
			// Per-neuron bias (stored in the Node graph as the final edge weight).
			// bias[j] corresponds to output unit j for this transition.
			std::vector<float> bias;

			// Accumulated gradients for current minibatch
			std::vector<float> gW;
			std::vector<float> gBias;

			Transition() : in(0), out(0) {}
		};

		std::vector<Transition> T;

		// Cached activations and deltas for a single sample (forward/backward).
		std::vector< std::vector<float> > a;      // a[layerIndex][i]
		std::vector< std::vector<float> > delta;  // delta for non-input layers; delta[li] aligns with a[li]

		// Minibatch accumulation count.
		unsigned int batchCount;

		// ATLAS optimizer state (one per Transition; used when optimizer.type==ATLAS).
		std::vector<atlas::WeightState> atlasState;
		HelmState helm;
		AsterState aster;

		TensorDFFState() : initialized(false), batchCount(0) {}

		void reset()
		{
			initialized = false;
			sizes.clear();
			T.clear();
			a.clear();
			delta.clear();
			batchCount = 0;
			helm.reset();
			aster.reset();
		}
	};

	// Tensorized recurrent training state (RNN/GRU/LSTM).
	//
	// These store parameters in packed contiguous arrays so the recurrent SGD helpers can run
	// mostly on cache-friendly kernels rather than pointer chasing through heap objects.
	// Momentum and weight decay semantics match the DFF tensor path (v = mf*v + lr*g; w -= v).
	struct TensorRNNState
	{
		bool initialized;
		unsigned int inputSize;
		unsigned int outSize;
		std::vector<unsigned int> hiddenSizes;

		struct Hidden
		{
			unsigned int in;
			unsigned int h;
			// Wxh: [h, in], Whh: [h, h] row-major
			std::vector<float> Wxh;
			std::vector<float> Whh;
			std::vector<float> vWxh;
			std::vector<float> vWhh;
			std::vector<float> gWxh;
			std::vector<float> gWhh;
			// Per-unit bias
			std::vector<float> bias;
			std::vector<float> gBias;
			atlas::WeightState atlasWxh;
			atlas::WeightState atlasWhh;
			Hidden() : in(0u), h(0u) {}
		};

		struct Out
		{
			unsigned int in;
			unsigned int out;
			// Why: [out, in]
			std::vector<float> Why;
			std::vector<float> vWhy;
			std::vector<float> gWhy;
			std::vector<float> bias;
			std::vector<float> gBias;
			atlas::WeightState atlasWhy;
			Out() : in(0u), out(0u) {}
		};

		std::vector<Hidden> H;
		Out O;

		TensorRNNState() : initialized(false), inputSize(0u), outSize(0u) {}

		void reset()
		{
			initialized = false;
			inputSize = 0u;
			outSize = 0u;
			hiddenSizes.clear();
			H.clear();
			O = Out();
		}
	};

	struct TensorGatedState
	{
		bool initialized;
		unsigned int inputSize;
		unsigned int outSize;
		unsigned int gateCount;
		std::vector<unsigned int> hiddenSizes;

		struct Hidden
		{
			unsigned int in;
			unsigned int h;
			// Packed W: [gateCount, h, in] and U: [gateCount, h, h] row-major
			// Bias: [gateCount, h]
			std::vector<float> W;
			std::vector<float> U;
			std::vector<float> vW;
			std::vector<float> vU;
			std::vector<float> gW;
			std::vector<float> gU;
			std::vector<float> bias;
			std::vector<float> gBias;
			atlas::WeightState atlasW;
			atlas::WeightState atlasU;
			Hidden() : in(0u), h(0u) {}
		};

		struct Out
		{
			unsigned int in;
			unsigned int out;
			std::vector<float> Why;
			std::vector<float> vWhy;
			std::vector<float> gWhy;
			std::vector<float> bias;
			std::vector<float> gBias;
			atlas::WeightState atlasWhy;
			Out() : in(0u), out(0u) {}
		};

		std::vector<Hidden> H;
		Out O;

		TensorGatedState(unsigned int g = 1u) : initialized(false), inputSize(0u), outSize(0u), gateCount(g) {}

		void reset()
		{
			initialized = false;
			inputSize = 0u;
			outSize = 0u;
			hiddenSizes.clear();
			H.clear();
			O = Out();
		}
	};

	TensorRNNState tensorRnn;
	// GRU: gateCount=3, LSTM: gateCount=4
	TensorGatedState tensorGru;
	TensorGatedState tensorLstm;

	// === CNN packed parameters ===
	//
	// Convolutional layers followed by fully-connected (FC) layers.
	// Data layout: channel-first (NCHW) throughout.
	struct TensorCNNState
	{
		bool initialized;

		// Image input dimensions.
		unsigned int inputH, inputW, inputC;

		// Precomputed spatial dimensions per conv layer.
		struct ConvSpatialInfo
		{
			unsigned int inH, inW, inC;
			unsigned int outH, outW, outC;
			unsigned int kH, kW;
			unsigned int strideH, strideW;
			unsigned int padH, padW;
			bool useBatchNorm;
			bool useMaxPool;
			unsigned int poolH, poolW, poolStrideH, poolStrideW;
			unsigned int poolOutH, poolOutW;
			// im2col matrix dims: rows = outH*outW, cols = inC*kH*kW
			unsigned int im2colRows, im2colCols;
		};

		std::vector<ConvSpatialInfo> spatialInfo;

		// Per-conv-layer weights.
		struct ConvLayer
		{
			unsigned int outC, inC, kH, kW;
			// W: [outC, inC*kH*kW] row-major
			std::vector<float> W;
			std::vector<float> bias; // [outC]
			std::vector<float> gW;
			std::vector<float> gBias;
			// Momentum / Adam state
			std::vector<float> vW;
			std::vector<float> v2W;  // Adam second moment
			std::vector<float> vBias;
			std::vector<float> v2Bias;
			// BatchNorm parameters (per channel)
			std::vector<float> bnGamma;   // [outC]
			std::vector<float> bnBeta;    // [outC]
			std::vector<float> bnRunMean; // [outC] running EMA
			std::vector<float> bnRunVar;  // [outC] running EMA
			std::vector<float> gBnGamma;
			std::vector<float> gBnBeta;
			std::vector<float> vBnGamma;
			std::vector<float> v2BnGamma;
			std::vector<float> vBnBeta;
			std::vector<float> v2BnBeta;

			// ATLAS optimizer state for this conv layer's weight matrix.
			atlas::WeightState atlasW;

			ConvLayer() : outC(0u), inC(0u), kH(0u), kW(0u) {}
		};

		std::vector<ConvLayer> convLayers;

		// FC transition layers (identical layout to DFF Transition).
		struct FCTransition
		{
			unsigned int in;
			unsigned int out;
			std::vector<float> W;    // [out, in]
			std::vector<float> bias; // [out]
			std::vector<float> gW;
			std::vector<float> gBias;
			std::vector<float> vW;
			std::vector<float> v2W;
			std::vector<float> vBias;
			std::vector<float> v2Bias;

			// ATLAS optimizer state for this FC layer's weight matrix.
			atlas::WeightState atlasW;

			FCTransition() : in(0u), out(0u) {}
		};

		std::vector<FCTransition> fcLayers;

		// Flattened feature size after last conv layer.
		unsigned int flattenedSize;
		// Optimizer step counter (for Adam bias correction).
		unsigned long long optimizerStep;
		// Minibatch accumulation count.
		unsigned int batchCount;

		TensorCNNState()
		    : initialized(false),
		      inputH(0u), inputW(0u), inputC(0u),
		      flattenedSize(0u),
		      optimizerStep(0ULL),
		      batchCount(0u)
		{
		}

		void reset()
		{
			initialized = false;
			inputH = inputW = inputC = 0u;
			flattenedSize = 0u;
			optimizerStep = 0ULL;
			batchCount = 0u;
			spatialInfo.clear();
			convLayers.clear();
			fcLayers.clear();
		}
	};

	// Scratch buffers for CNN forward/backward pass (reused per sample).
	struct CNNScratch
	{
		// Per conv layer: post-conv output, im2col matrix, bn/pool intermediates
		struct ConvLayerScratch
		{
			std::vector<float> im2col;    // [outH*outW, inC*kH*kW]
			std::vector<float> convOut;   // [outC, outH*outW]
			std::vector<float> bnOut;     // [outC, outH*outW] (after BN)
			std::vector<float> bnMean;    // [outC] per-channel mean
			std::vector<float> bnInvStd;  // [outC] per-channel 1/sqrt(var+eps)
			std::vector<float> bnNorm;    // [outC, outH*outW] normalized (before gamma/beta)
			std::vector<float> actOut;    // [outC, outH*outW] (after ReLU)
			std::vector<float> poolOut;   // [outC, poolOutH*poolOutW]
			std::vector<int> poolArgmax;  // [outC, poolOutH*poolOutW] argmax indices

			// Backward scratch
			std::vector<float> dPoolOut;  // gradient from upstream
			std::vector<float> dActOut;   // gradient after pool backward / before ReLU backward
			std::vector<float> dConvOut;  // gradient at conv output (after BN backward)
			std::vector<float> dIm2col;   // gradient for col2im
		};

		std::vector<ConvLayerScratch> convScratch;

		// FC layer scratch (activations and deltas, similar to DFF)
		std::vector<std::vector<float> > fcA;     // activations per FC layer
		std::vector<std::vector<float> > fcDelta; // deltas per FC layer

		// Reusable buffer for conv input gradient during backward (avoids static/thread-unsafe alloc).
		std::vector<float> convInputGrad;

		// Gradient at flatten layer input (propagated from first FC layer).
		std::vector<float> dFlatten;

		// Output
		std::vector<float> logits;
		std::vector<float> probs;
	};

	TensorCNNState tensorCnn;
	CNNScratch cnnScratch;

	// === Transposed-convolution (deconv) generator state ===
	struct TensorDeconvState
	{
		bool initialized;

		// FC projection: noise -> projectC * projectH * projectW
		unsigned int fcIn, fcOut;
		std::vector<float> fcW, fcBias, fcGW, fcGBias;
		std::vector<float> fcVW, fcV2W, fcVBias, fcV2Bias; // Adam

		unsigned int projectC, projectH, projectW;

		struct DeconvLayer
		{
			unsigned int inC, outC, kH, kW;
			unsigned int strideH, strideW, padH, padW;
			unsigned int inH, inW, outH, outW;
			bool useBatchNorm, useReLU;
			bool useUpsampleConv; // nearest-neighbor upsample + standard conv
			bool useTanh;

			std::vector<float> W, bias, gW, gBias;
			std::vector<float> vW, v2W, vBias, v2Bias; // Adam

			// Batch normalization (per outC channel)
			std::vector<float> bnGamma, bnBeta, bnRunMean, bnRunVar;
			std::vector<float> gBnGamma, gBnBeta;                    // gradients
			std::vector<float> vBnGamma, v2BnGamma, vBnBeta, v2BnBeta; // Adam

			DeconvLayer()
			    : inC(0u), outC(0u), kH(0u), kW(0u),
			      strideH(0u), strideW(0u), padH(0u), padW(0u),
			      inH(0u), inW(0u), outH(0u), outW(0u),
			      useBatchNorm(false), useReLU(true),
			      useUpsampleConv(false),
			      useTanh(false)
			{
			}
		};

		std::vector<DeconvLayer> layers;
		unsigned long long optimizerStep;

		TensorDeconvState()
		    : initialized(false),
		      fcIn(0u), fcOut(0u),
		      projectC(0u), projectH(0u), projectW(0u),
		      optimizerStep(0ULL)
		{
		}

		void reset()
		{
			initialized = false;
			fcIn = fcOut = 0u;
			projectC = projectH = projectW = 0u;
			fcW.clear(); fcBias.clear(); fcGW.clear(); fcGBias.clear();
			fcVW.clear(); fcV2W.clear(); fcVBias.clear(); fcV2Bias.clear();
			layers.clear();
			optimizerStep = 0ULL;
		}
	};

	TensorDeconvState tensorDeconv;

	// Reusable scratch buffers for recurrent (RNN/GRU/LSTM) forward/backward passes.
	// This avoids per-window nested-vector allocations in the hot path.
	struct RecurrentScratch
	{
		unsigned int winLen;
		unsigned int inputSize;
		unsigned int outSize;
		std::vector<unsigned int> hiddenSizes;

		// Common buffers
		std::vector<float> x; // [winLen, inputSize]
		std::vector<float> y; // [winLen, outSize]
		std::vector<float> outLogits; // [outSize]
		std::vector<float> outProbs;  // [outSize]

		// Per hidden layer buffers (flattened [winLen, hiddenSize])
		std::vector< std::vector<float> > h;
		std::vector< std::vector<float> > hPrevAtT;

		// GRU-only buffers
		std::vector< std::vector<float> > z;
		std::vector< std::vector<float> > r;
		std::vector< std::vector<float> > hTilde;

		// LSTM-only buffers
		std::vector< std::vector<float> > c;
		std::vector< std::vector<float> > cPrevAtT;
		std::vector< std::vector<float> > iGate;
		std::vector< std::vector<float> > fGate;
		std::vector< std::vector<float> > oGate;
		std::vector< std::vector<float> > gGate;
		std::vector< std::vector<float> > tanhC;

		RecurrentScratch() : winLen(0u), inputSize(0u), outSize(0u) {}

		static void resizeAndZero(std::vector<float>& v, size_t n)
		{
			if (v.size() != n)
				v.resize(n);
			std::fill(v.begin(), v.end(), 0.0f);
		}

		static void resizeAndZero2D(std::vector< std::vector<float> >& vv, size_t rows, const std::vector<unsigned int>& widths, unsigned int winLen)
		{
			if (vv.size() != rows)
				vv.resize(rows);
			for (size_t i = 0; i < rows; ++i)
			{
				const size_t n = static_cast<size_t>(winLen) * static_cast<size_t>(widths[i]);
				resizeAndZero(vv[i], n);
			}
		}

		void ensureCommon(unsigned int newWinLen,
		                  unsigned int newInputSize,
		                  unsigned int newOutSize,
		                  const std::vector<unsigned int>& newHiddenSizes)
		{
			winLen = newWinLen;
			inputSize = newInputSize;
			outSize = newOutSize;
			hiddenSizes = newHiddenSizes;

			resizeAndZero(x, static_cast<size_t>(winLen) * static_cast<size_t>(inputSize));
			resizeAndZero(y, static_cast<size_t>(winLen) * static_cast<size_t>(outSize));

			// Softmax temps (reused per timestep)
			if (outLogits.size() != outSize)
				outLogits.resize(outSize);
			if (outProbs.size() != outSize)
				outProbs.resize(outSize);
		}

		void ensureRNN(unsigned int newWinLen,
		               unsigned int newInputSize,
		               unsigned int newOutSize,
		               const std::vector<unsigned int>& newHiddenSizes)
		{
			ensureCommon(newWinLen, newInputSize, newOutSize, newHiddenSizes);
			resizeAndZero2D(h, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(hPrevAtT, newHiddenSizes.size(), newHiddenSizes, winLen);
		}

		void ensureGRU(unsigned int newWinLen,
		               unsigned int newInputSize,
		               unsigned int newOutSize,
		               const std::vector<unsigned int>& newHiddenSizes)
		{
			ensureRNN(newWinLen, newInputSize, newOutSize, newHiddenSizes);
			resizeAndZero2D(z, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(r, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(hTilde, newHiddenSizes.size(), newHiddenSizes, winLen);
		}

		void ensureLSTM(unsigned int newWinLen,
		                unsigned int newInputSize,
		                unsigned int newOutSize,
		                const std::vector<unsigned int>& newHiddenSizes)
		{
			ensureRNN(newWinLen, newInputSize, newOutSize, newHiddenSizes);
			resizeAndZero2D(c, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(cPrevAtT, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(iGate, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(fGate, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(oGate, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(gGate, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(tanhC, newHiddenSizes.size(), newHiddenSizes, winLen);
		}
	};

	CMatrix confusionMatrix;
	GNet::GServer* serverInstance;
	GNet::Connection* cConnection;
	// Optional logger override. If NULL, use the attached server logger (if any),
	// otherwise fall back to a default logger instance.
	shmea::GLogger* loggerOverride;
	glades::NaiveBayes bModel;

	volatile int running;
	int netType;
	int epochs;
	bool saveInstance;
	float overallTotalError;
	float overallTotalAccuracy;
	float overallClassAccuracy;
	float overallClassPrecision;
	float overallClassRecall;
	float overallClassSpecificity;
	float overallClassF1;
	int minibatchSize;
	int64_t id;
	uint64_t rngSeed;
	// Per-network RNG engine. Training installs this as the "current" RNG via a scoped guard
	// so random draws are deterministic per-network.
	glades::rng::Engine rngEngine;
	// Concurrency guard for train/test runs.
	//
	// Policy:
	// - A single NNetwork instance is NOT re-entrant: do not call train()/test() concurrently
	//   on the same object from multiple threads.
	// - Different NNetwork instances MAY be run concurrently in different threads (subject to
	//   DataInput thread-safety; see determinism/concurrency policy docs).
	//
	// This lock prevents two threads from mutating shared scratch buffers / tensor state / metrics.
	// Implementation:
	// - C++11+: std::atomic_flag spinlock (portable)
	// - pre-C++11: volatile int with GCC/Clang __sync builtins (legacy)
#if GLADES_HAVE_STD_ATOMICS
	std::atomic_flag runLock;
#else
	volatile int runLock;
#endif

	// Low-level lock primitives (implemented in network.cpp; GCC/Clang use atomic builtins).
	bool tryAcquireRunLock();
	void releaseRunLock();
	bool loadRunningFlag() const;
	void storeRunningFlag(bool value);
	static void resetPersistenceDiagnosticsAttempt(PersistenceDiagnostics& d,
	                                               const char* operation,
	                                               const std::string& name,
	                                               int netType,
	                                               bool isCheckpoint,
	                                               bool tokenizerPresent,
	                                               bool includeOptimizerState,
	                                               uint64_t maxShardBytes);
	static void notePersistenceDiagnosticsFailure(PersistenceDiagnostics& d,
	                                              const char* stage,
	                                              bool rejectedInput,
	                                              bool rotatedPrevious,
	                                              const NNetworkStatus& st,
	                                              uint64_t shardCount,
	                                              uint64_t tensorCount,
	                                              uint64_t weightsBytes);
	static void notePersistenceDiagnosticsSuccess(PersistenceDiagnostics& d,
	                                              const char* stage,
	                                              bool rotatedPrevious,
	                                              const NNetworkStatus& st,
	                                              uint64_t shardCount,
	                                              uint64_t tensorCount,
	                                              uint64_t weightsBytes);
	uint64_t loadConfiguredSeed() const;
	void storeConfiguredSeed(uint64_t seed);
	shmea::GLogger* loadLoggerOverride() const;
	void storeLoggerOverride(shmea::GLogger* logger);

	// Epoch-scoped metric accumulators (reset at the start of each epoch).
	// Regression:
	// - SSE/SAE across all samples and outputs (used to compute MSE/MAE/RMSE/R^2).
	// - SumY/SumY2 used to compute SST without a second pass.
	double regSSE;
	double regSAE;
	double regSumY;
	double regSumY2;
	unsigned long long regCount;

	// Classification/KL: top-1 accuracy across samples.
	unsigned long long clsCorrect;
	unsigned long long clsTotal;

	bool firstRunActivation;
	NNetworkStatus lastStatus;
	TrainerRunDiagnostics trainerRunDiagnostics;
	mutable PersistenceDiagnostics persistenceDiagnostics;
	TensorDFFState tensorDff;
	RecurrentScratch recScratch;

	#include "transformer_model_state.inc"

	struct TransformerScratch
	{
		unsigned int T;
		unsigned int inputSize;
		unsigned int outSize;
		unsigned int dModel;
		unsigned int dFF;
		// Width of K/V projections: nKVHeads * dHead.
		unsigned int dModelKV;
		unsigned int nHeads;
		unsigned int nLayers;
		// Width of FF1 pre-activation buffer (dFF for MLP, 2*dFF for SwiGLU).
		unsigned int ff1Width;

		// Common (aligned for SIMD-friendly kernels)
		std::vector<float, glades::AlignedAllocator<float, 64> > x; // [T, inputSize]
		std::vector<float, glades::AlignedAllocator<float, 64> > h; // [T, dModel] after input projection + pos enc

		// Per-layer caches (flattened by layer index)
		// LN1
		std::vector<float, glades::AlignedAllocator<float, 64> > ln1Mean;   // [nLayers, T]
		std::vector<float, glades::AlignedAllocator<float, 64> > ln1InvStd; // [nLayers, T]
		std::vector<float, glades::AlignedAllocator<float, 64> > x1;        // [nLayers, T, dModel] (LN1 output)
		// Q,K,V and attention
		std::vector<float, glades::AlignedAllocator<float, 64> > Q; // [nLayers, T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > K; // [nLayers, T, dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > V; // [nLayers, T, dModelKV]
		// Concatenated per-head attention outputs before Wo.
		// This is stored for backward so we don't need to cache/store full [T,T] attention probabilities.
		std::vector<float, glades::AlignedAllocator<float, 64> > attnConcat; // [nLayers, T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > attnOut;    // [nLayers, T, dModel] (after Wo)
		std::vector<float, glades::AlignedAllocator<float, 64> > hAfterAttn; // [nLayers, T, dModel] (residual)

		// LN2
		std::vector<float, glades::AlignedAllocator<float, 64> > ln2Mean;   // [nLayers, T]
		std::vector<float, glades::AlignedAllocator<float, 64> > ln2InvStd; // [nLayers, T]
		std::vector<float, glades::AlignedAllocator<float, 64> > x2;        // [nLayers, T, dModel] (LN2 output)
		// FFN caches
		std::vector<float, glades::AlignedAllocator<float, 64> > ff1;    // [nLayers, T, ff1Width] pre-activation
		std::vector<float, glades::AlignedAllocator<float, 64> > ff1Act; // [nLayers, T, dFF] post-activation
		std::vector<float, glades::AlignedAllocator<float, 64> > ffOut;  // [nLayers, T, dModel] (after W2)
		// Output of each block
		std::vector<float, glades::AlignedAllocator<float, 64> > hAfterFF; // [nLayers, T, dModel]

		// Final LayerNorm scratch
		std::vector<float, glades::AlignedAllocator<float, 64> > lnFinalMean;   // [T]
		std::vector<float, glades::AlignedAllocator<float, 64> > lnFinalInvStd; // [T]
		std::vector<float, glades::AlignedAllocator<float, 64> > hPostFinalLN;  // [T, dModel]

		// Dropout masks (unsigned char, aligned)
		std::vector<unsigned char, glades::AlignedAllocator<unsigned char, 64> > dropoutMaskEmb;     // [T*dModel]
		std::vector<unsigned char, glades::AlignedAllocator<unsigned char, 64> > dropoutMaskResAttn;  // [nLayers*T*dModel]
		std::vector<unsigned char, glades::AlignedAllocator<unsigned char, 64> > dropoutMaskResFF;    // [nLayers*T*dModel]

		// Gradient checkpointing recompute buffers (only allocated when enabled)
		std::vector<float, glades::AlignedAllocator<float, 64> > recomp_x1;        // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > recomp_Q;         // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > recomp_K;         // [T, dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > recomp_V;         // [T, dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > recomp_attnConcat; // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > recomp_x2;        // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > recomp_ff1;       // [T, ff1Width]
		std::vector<float, glades::AlignedAllocator<float, 64> > recomp_ff1Act;    // [T, dFF]

		// Output head
		std::vector<float, glades::AlignedAllocator<float, 64> > logits; // [T, outSize]
		std::vector<float, glades::AlignedAllocator<float, 64> > probs;  // [T, outSize] (softmax if needed)
		// Token LM sampled-softmax: per-timestep sampled token ids corresponding to logits/probs.
		// When sampled-softmax is enabled, logits/probs are sized [T, (1+K)] and tokenLmSampleIds
		// holds the vocabulary indices for each sampled column (col 0 is always the target id).
		std::vector<int> tokenLmSampleIds; // [T, outSize] (only used for token LM sampled-softmax)

		// === Backward scratch (reused across sequences/layers; aligned) ===
		// These buffers eliminate per-sequence/per-layer allocations in transformer backward.
		std::vector<float, glades::AlignedAllocator<float, 64> > dLogits; // [T, outSize]
		std::vector<float, glades::AlignedAllocator<float, 64> > dH;      // [T, dModel] upstream gradient
		std::vector<float, glades::AlignedAllocator<float, 64> > dH2;     // [T, dModel] secondary buffer
		// FFN backward temps
		std::vector<float, glades::AlignedAllocator<float, 64> > dFF1Act;          // [T, dFF]
		std::vector<float, glades::AlignedAllocator<float, 64> > dFF1Cat;          // [T, ff1Width]
		std::vector<float, glades::AlignedAllocator<float, 64> > dX2;              // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dHAfterAttnFromLN; // [T, dModel]
		// Attention/backprop temps
		std::vector<float, glades::AlignedAllocator<float, 64> > dAttnConcat; // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dQfull;      // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dKfull;      // [T, dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > dVfull;      // [T, dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > dX1;         // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dXtmp;       // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dHInFromLN;  // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dInput;      // [T, inputSize]

		// Attention backward chunked dK/dV scratch (reused across layers).
		// Sized lazily on first use based on nChunksPerHead; persists across layers/sequences.
		std::vector<float, glades::AlignedAllocator<float, 64> > dKVscratch;

		TransformerScratch()
		    : T(0u),
		      inputSize(0u),
		      outSize(0u),
		      dModel(0u),
		      dFF(0u),
		      dModelKV(0u),
		      nHeads(0u),
		      nLayers(0u),
		      ff1Width(0u)
		{
		}

		template <typename VecT>
		static void resize_and_zero(VecT& v, size_t n)
		{
			if (v.size() != n)
				v.resize(n);
			std::fill(v.begin(), v.end(), 0.0f);
		}

		template <typename UCharVec>
		static void resize_uchar_zero(UCharVec& v, size_t n)
		{
			if (v.size() != n)
				v.resize(n);
			std::fill(v.begin(), v.end(), static_cast<unsigned char>(0));
		}

		void ensure(unsigned int newT,
		            unsigned int newInputSize,
		            unsigned int newOutSize,
		            unsigned int newDModel,
		            unsigned int newDFF,
		            unsigned int newDModelKV,
		            unsigned int newNHeads,
		            unsigned int newNLayers,
		            unsigned int newFF1Width,
		            float embDropRate = 0.0f,
		            float resDropRate = 0.0f,
		            bool gradCheckpoint = false)
		{
			T = newT;
			inputSize = newInputSize;
			outSize = newOutSize;
			dModel = newDModel;
			dFF = newDFF;
			dModelKV = newDModelKV;
			nHeads = newNHeads;
			nLayers = newNLayers;
			ff1Width = newFF1Width;

			resize_and_zero(x, static_cast<size_t>(T) * static_cast<size_t>(inputSize));
			resize_and_zero(h, static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(ln1Mean, static_cast<size_t>(nLayers) * static_cast<size_t>(T));
			resize_and_zero(ln1InvStd, static_cast<size_t>(nLayers) * static_cast<size_t>(T));
			resize_and_zero(x1, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(Q, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(K, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
			resize_and_zero(V, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));

			resize_and_zero(attnConcat, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(attnOut, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(hAfterAttn, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(ln2Mean, static_cast<size_t>(nLayers) * static_cast<size_t>(T));
			resize_and_zero(ln2InvStd, static_cast<size_t>(nLayers) * static_cast<size_t>(T));
			resize_and_zero(x2, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(ff1, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
			resize_and_zero(ff1Act, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dFF));
			resize_and_zero(ffOut, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(hAfterFF, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

			// Final LayerNorm scratch (always allocated)
			resize_and_zero(lnFinalMean, static_cast<size_t>(T));
			resize_and_zero(lnFinalInvStd, static_cast<size_t>(T));
			resize_and_zero(hPostFinalLN, static_cast<size_t>(T) * static_cast<size_t>(dModel));

			// Dropout masks (only allocated if rates > 0)
			const size_t TD = static_cast<size_t>(T) * static_cast<size_t>(dModel);
			const size_t LTD = static_cast<size_t>(nLayers) * TD;
			if (embDropRate > 0.0f)
				resize_uchar_zero(dropoutMaskEmb, TD);
			else
				dropoutMaskEmb.clear();
			if (resDropRate > 0.0f)
			{
				resize_uchar_zero(dropoutMaskResAttn, LTD);
				resize_uchar_zero(dropoutMaskResFF, LTD);
			}
			else
			{
				dropoutMaskResAttn.clear();
				dropoutMaskResFF.clear();
			}

			// Gradient checkpointing recompute buffers
			if (gradCheckpoint)
			{
				resize_and_zero(recomp_x1, static_cast<size_t>(T) * static_cast<size_t>(dModel));
				resize_and_zero(recomp_Q, static_cast<size_t>(T) * static_cast<size_t>(dModel));
				resize_and_zero(recomp_K, static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
				resize_and_zero(recomp_V, static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
				resize_and_zero(recomp_attnConcat, static_cast<size_t>(T) * static_cast<size_t>(dModel));
				resize_and_zero(recomp_x2, static_cast<size_t>(T) * static_cast<size_t>(dModel));
				resize_and_zero(recomp_ff1, static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
				resize_and_zero(recomp_ff1Act, static_cast<size_t>(T) * static_cast<size_t>(dFF));
			}
			else
			{
				recomp_x1.clear(); recomp_Q.clear(); recomp_K.clear(); recomp_V.clear();
				recomp_attnConcat.clear(); recomp_x2.clear(); recomp_ff1.clear(); recomp_ff1Act.clear();
			}

			resize_and_zero(logits, static_cast<size_t>(T) * static_cast<size_t>(outSize));
			resize_and_zero(probs, static_cast<size_t>(T) * static_cast<size_t>(outSize));
			if (tokenLmSampleIds.size() != static_cast<size_t>(T) * static_cast<size_t>(outSize))
				tokenLmSampleIds.resize(static_cast<size_t>(T) * static_cast<size_t>(outSize));
			std::fill(tokenLmSampleIds.begin(), tokenLmSampleIds.end(), 0);

			// Backward scratch (not per-layer; reused across the backward pass)
			// Note: we do not rely on these being zeroed except where explicitly filled in the hot path.
			resize_and_zero(dLogits, static_cast<size_t>(T) * static_cast<size_t>(outSize));
			resize_and_zero(dH, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dH2, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dFF1Act, static_cast<size_t>(T) * static_cast<size_t>(dFF));
			resize_and_zero(dFF1Cat, static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
			resize_and_zero(dX2, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dHAfterAttnFromLN, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dAttnConcat, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dQfull, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dKfull, static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
			resize_and_zero(dVfull, static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
			resize_and_zero(dX1, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dXtmp, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dHInFromLN, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dInput, static_cast<size_t>(T) * static_cast<size_t>(inputSize));
		}
	};

	TensorTransformerState tensorTransformer;
	TransformerScratch transformerScratch;

	// === GPU-resident state (CUDA offloading) ===
	//
	// When GLADES_HAVE_CUDA is defined and the GPU is enabled, these hold device-side
	// mirrors of the corresponding CPU tensor state. Weights stay GPU-resident; only
	// inputs/outputs cross PCIe.
#ifdef GLADES_HAVE_CUDA
	gpu::GpuTransformerWeights* gpuTransformerWeights;
	gpu::GpuTransformerScratch* gpuTransformerScratch;
	gpu::GpuDFFWeights* gpuDffWeights;
	gpu::GpuDFFScratch* gpuDffScratch;
	gpu::GpuRNNWeights* gpuRnnWeights;
	gpu::GpuGatedWeights* gpuGruWeights;
	gpu::GpuGatedWeights* gpuLstmWeights;
	gpu::GpuCNNWeights* gpuCnnWeights;
	gpu::GpuCNNScratch* gpuCnnScratch;
#else
	void* gpuTransformerWeights;
	void* gpuTransformerScratch;
	void* gpuDffWeights;
	void* gpuDffScratch;
	void* gpuRnnWeights;
	void* gpuGruWeights;
	void* gpuLstmWeights;
	void* gpuCnnWeights;
	void* gpuCnnScratch;
#endif
	bool gpuStateReady;

	// Ensure GPU state is allocated and weights are uploaded.
	// Returns true if GPU is ready for use, false if CPU fallback should be used.
	bool ensureGpuState();
	// Free all GPU state.
	void freeGpuState();

	// === Positional encoding caches (Transformer) ===
	//
	// These caches avoid recomputing expensive pow()-derived frequency terms (sinusoidal PE and RoPE).
	// They intentionally do NOT cache full [T,dModel] sin/cos tables (which can be enormous).
	//
	// NOTE: This cache is mutable so const inference helpers (e.g. transformerLmForwardLastLogits)
	// can reuse it without allocating each call. NNetwork is not re-entrant; callers should not
	// invoke transformer inference concurrently on the same instance.
	struct TransformerPosEncCache
	{
		// Sinusoidal positional encoding cache:
		// invDenomPair[ii] = 1 / 10000^(2*ii/dModel), length ceil(dModel/2).
		unsigned int sinDModelCached;
		std::vector<double> sinInvDenomPair;

		// RoPE cache:
		// invFreq[ii] = theta^(-2*ii/ropeDim), length ropeDim/2.
		unsigned int ropeDimCached;
		float ropeThetaCached;
		std::vector<double> ropeInvFreq;

		TransformerPosEncCache()
		    : sinDModelCached(0u),
		      sinInvDenomPair(),
		      ropeDimCached(0u),
		      ropeThetaCached(0.0f),
		      ropeInvFreq()
		{
		}

		void reset()
		{
			sinDModelCached = 0u;
			sinInvDenomPair.clear();
			ropeDimCached = 0u;
			ropeThetaCached = 0.0f;
			ropeInvFreq.clear();
		}

		void ensureSinusoidal(unsigned int dModel)
		{
			if (dModel == 0u)
			{
				sinDModelCached = 0u;
				sinInvDenomPair.clear();
				return;
			}
			if (sinDModelCached == dModel && !sinInvDenomPair.empty())
				return;

			sinDModelCached = dModel;
			const unsigned int nPairs = (dModel + 1u) / 2u;
			sinInvDenomPair.assign(static_cast<size_t>(nPairs), 0.0);
			for (unsigned int ii = 0; ii < nPairs; ++ii)
			{
				// invDenom = 10000^(-2*ii/dModel)
				const double exponent = (2.0 * static_cast<double>(ii)) / static_cast<double>(dModel);
				sinInvDenomPair[static_cast<size_t>(ii)] = pow(10000.0, -exponent);
			}
		}

		void ensureRope(unsigned int ropeDimEven, float ropeTheta)
		{
			if (ropeDimEven < 2u || ropeTheta <= 0.0f)
			{
				ropeDimCached = 0u;
				ropeThetaCached = 0.0f;
				ropeInvFreq.clear();
				return;
			}
			if ((ropeDimEven % 2u) != 0u)
				ropeDimEven -= 1u;
			if (ropeDimCached == ropeDimEven && ropeThetaCached == ropeTheta && !ropeInvFreq.empty())
				return;

			ropeDimCached = ropeDimEven;
			ropeThetaCached = ropeTheta;
			ropeInvFreq.assign(static_cast<size_t>(ropeDimEven / 2u), 0.0);
			for (unsigned int ii = 0; ii < (ropeDimEven / 2u); ++ii)
			{
				const double frac = (2.0 * static_cast<double>(ii)) / static_cast<double>(ropeDimEven);
				ropeInvFreq[static_cast<size_t>(ii)] = pow(static_cast<double>(ropeTheta), -frac);
			}
		}
	};
	mutable TransformerPosEncCache transformerPosEncCache;

	struct TransformerTokenStepCore
	{
		TransformerTokenStepCore()
		    : where(NULL),
		      tokenId(0u),
		      pos(0u),
		      maxLen(0u),
		      keyValid(NULL),
		      kSeq(NULL),
		      vSeq(NULL),
		      kSeq16(NULL),
		      vSeq16(NULL),
		      outLogits(NULL),
		      dModel(0u),
		      dFF(0u),
		      nHeads(0u),
		      nKVHeads(0u),
		      nLayers(0u),
		      dHead(0u),
		      dModelKV(0u),
		      ffnKind(0u),
		      ff1Width(0u),
		      layerNormEps(0.0f),
		      normType(0u),
		      positionalEncoding(0u),
		      ropeDim(0u),
		      ffnActivation(0u),
		      lowpDType(0),
		      metricsEnabled(false),
		      metricsBreakdownEnabled(false),
		      posEncCache(NULL),
		      perf(NULL),
		      h(NULL),
		      x1(NULL),
		      x2(NULL),
		      q(NULL),
		      kvec(NULL),
		      vvec(NULL),
		      attnConcat(NULL),
		      attnOut(NULL),
		      ffPre(NULL),
		      ffAct(NULL),
		      ffOut(NULL),
		      scores(NULL)
		{
		}

		bool usesLowPrecisionKvCache() const { return kSeq16 != NULL && vSeq16 != NULL; }

		const char* where;
		unsigned int tokenId;
		unsigned int pos;
		unsigned int maxLen;
		unsigned char* keyValid;
		float* kSeq;
		float* vSeq;
		uint16_t* kSeq16;
		uint16_t* vSeq16;
		float* outLogits;

		unsigned int dModel;
		unsigned int dFF;
		unsigned int nHeads;
		unsigned int nKVHeads;
		unsigned int nLayers;
		unsigned int dHead;
		unsigned int dModelKV;
		unsigned int ffnKind;
		unsigned int ff1Width;
		float layerNormEps;
		unsigned int normType;
		unsigned int positionalEncoding;
		unsigned int ropeDim;
		unsigned int ffnActivation;
		int lowpDType;
		bool metricsEnabled;
		bool metricsBreakdownEnabled;

		TransformerPosEncCache* posEncCache;
		void* perf;
		std::vector<float, glades::AlignedAllocator<float, 64> >* h;
		std::vector<float, glades::AlignedAllocator<float, 64> >* x1;
		std::vector<float, glades::AlignedAllocator<float, 64> >* x2;
		std::vector<float, glades::AlignedAllocator<float, 64> >* q;
		std::vector<float, glades::AlignedAllocator<float, 64> >* kvec;
		std::vector<float, glades::AlignedAllocator<float, 64> >* vvec;
		std::vector<float, glades::AlignedAllocator<float, 64> >* attnConcat;
		std::vector<float, glades::AlignedAllocator<float, 64> >* attnOut;
		std::vector<float, glades::AlignedAllocator<float, 64> >* ffPre;
		std::vector<float, glades::AlignedAllocator<float, 64> >* ffAct;
		std::vector<float, glades::AlignedAllocator<float, 64> >* ffOut;
		std::vector<float, glades::AlignedAllocator<float, 64> >* scores;
	};

	NNetworkStatus transformerLmAppendCpuTokenCore(TransformerTokenStepCore& core) const;

	// Ensure packed tensor parameters are initialized from the attached DataInput shape.
	// Returns false and sets lastStatus on failure.
	bool ensureTensorParametersInitialized();

	// Tensor-first persistence for model packages (manifest version >= 2).
	// These write/read packed tensors directly.
	NNetworkStatus saveTensorWeightsToFile(const std::string& filePath) const;
	NNetworkStatus loadTensorWeightsFromFile(const std::string& filePath);

	// for tables & graphs
	std::vector<Point2*> rocCurve;
	shmea::GList results;
	shmea::GTable nbRecord;
	//Only for sending on the network
	shmea::GList cNodeActivations;

	// === Modern training loop features (minimal, backward-compatible) ===
	//
	// These features are implemented in the training core and (for DFF) in the tensor SGD step.
	// Defaults preserve historical behavior.
	TrainingConfig trainingConfig;
	float lrScheduleMultiplier; // computed each epoch by the scheduler; starts at 1
	int lrScheduleEpochOffset;  // added to epochFromStart in Trainer::run(); caller sets this
	                            // when train() is called once per epoch in a loop
	float lastGradNorm;
	float lastGradNormScale;
	int64_t lastStepLogTime;

	// Bayesian adaptive LR state (used when lrSchedule.type == BAYESIAN).
	float bayesianLRMultiplier_;       // current Bayesian LR multiplier (default 1.0)
	int bayesianLREpochCounter_;       // epochs since last LR adjustment
	GaussianProcess bayesianLRGP_;     // 1D GP for inner-loop adaptive LR

	// === Tokenizer + vocabulary artifacts (deployment metadata) ===
	//
	// Glades models operate on token IDs. To make model packages self-contained for deployment,
	// callers may attach tokenizer/vocab artifacts to the network and persist them alongside
	// the model weights/architecture.
	//
	// IMPORTANT:
	// - This is metadata only. Glades does not implement BPE/SentencePiece tokenization here.
	// - The artifact format is intentionally dependency-free and validated strictly on load.
public:
	struct TokenizerArtifacts
	{
		// Opaque tokenizer type identifier (examples: "bpe", "sentencepiece", "wordpiece", "custom").
		// This is intended for consumers to route to the appropriate tokenizer implementation.
		std::string type;
		// Vocabulary table mapping token id -> token bytes (UTF-8 recommended but not required).
		std::vector<std::string> vocab;

		// Special token ids (optional; -1 means "not set").
		// These are NOT automatically forced to match trainingConfig.transformer.padTokenId, etc.
		int padTokenId;
		int bosTokenId;
		int eosTokenId;
		int unkTokenId;

		TokenizerArtifacts()
		    : type(),
		      vocab(),
		      padTokenId(-1),
		      bosTokenId(-1),
		      eosTokenId(-1),
		      unkTokenId(-1)
		{
		}

		bool hasType() const { return !type.empty(); }
		bool hasVocab() const { return !vocab.empty(); }
		size_t vocabSize() const { return vocab.size(); }
		bool hasAnySpecialTokenId() const
		{
			return padTokenId >= 0 || bosTokenId >= 0 || eosTokenId >= 0 || unkTokenId >= 0;
		}
		static bool isSpecialTokenIdSet(int id) { return id >= 0; }

		void reset() { *this = TokenizerArtifacts(); }
	};

private:
	// Whether tokenizerArtifacts is present/meaningful.
	bool tokenizerArtifactsPresent;
	TokenizerArtifacts tokenizerArtifacts;

	// Internal helper used by TrainingCore (friend) to compute the schedule multiplier.
	float computeLearningRateMultiplier(int epochFromStart) const;

	NNetworkStatus run(const DataInput*, int, ITrainingCallbacks*);
	NNetworkStatus failStatus(NNetworkStatus::Code code, const std::string& message);
	// Reset core state (used by constructors/destructor and internal load/build paths).
	void clean();
	// Reset graph/curve outputs (train runs only).
	void resetGraphs();
	// Per-sample forward/backprop/update.
	// Returns explicit status; callers MUST stop on failure.
	NNetworkStatus SGDHelper(unsigned int, int); // Stochastic Gradient Descent
	// Net-type-specific SGD implementations (split into separate translation units).
	// These functions preserve the historical behavior of the corresponding blocks
	// that previously lived inside SGDHelper().
	void SGDHelper_DFF(unsigned int inputRowCounter, int runType);
	void SGDHelper_RNN(unsigned int inputRowCounter, int runType);
	void SGDHelper_GRU(unsigned int inputRowCounter, int runType);
	void SGDHelper_LSTM(unsigned int inputRowCounter, int runType);
	void SGDHelper_TRANSFORMER(unsigned int inputRowCounter, int runType);
	void SGDHelper_CNN(unsigned int inputRowCounter, int runType);

	// Transformer SGD sub-functions (split from SGDHelper_TRANSFORMER for readability).
	// The config struct bundles per-epoch derived values so the extracted methods
	// share state without passing dozens of individual parameters.
	struct TransformerEpochCfg
	{
		unsigned int inputSize, outSize, dModel, dFF, nHeads, nKVHeads, nLayers;
		unsigned int vocabSize, dHead, dModelKV, ff1Width;
		int padTokenId;
		bool causal, tokenLM, tieEmb, isTrain;
		float gradClip, lnEps, ropeTheta;
		int costFx, posEnc, normType, ffnKind, ffnAct, ropeDimOverride;
		int tokenLmNegK;
		bool ddpEnabled;
		bool useLowpWeights;
		int lowpDType;
		bool mpEnable, mpUseLossScaling, mpDynamicLossScaling;
		unsigned int seqBatchMax;
		glades::TransformerRunConfig::TokenLMLossKind tokenLmLossKind;
		bool tokenLmAllowHuge;
		float lrScheduleMultiplier;
	};
#ifdef GLADES_HAVE_CUDA
	bool tryRunTransformerGpuEpoch(const TransformerEpochCfg& cfg, unsigned int seqCount,
	                               int epochIdx, int64_t epochStartMs,
	                               unsigned long long& tokensProcessed,
	                               unsigned long long& targetsProcessed,
	                               double& tokenLmNllSum,
	                               unsigned long long& tokenLmTokenCount,
	                               unsigned long long& clsCorrect,
	                               unsigned long long& clsTotal,
	                               shmea::GLogger* logger);
	bool ensureTransformerGpuTrainingScratch(const TransformerEpochCfg& cfg, unsigned int T);
	bool syncTransformerGpuTrainingWeightsToCpu();
	void transformerGpuTrainEpoch(const TransformerEpochCfg& cfg, unsigned int seqCount,
	                              int epochIdx, int64_t epochStartMs,
	                              unsigned long long& tokensProcessed,
	                              unsigned long long& targetsProcessed,
	                              double& tokenLmNllSum,
	                              unsigned long long& tokenLmTokenCount,
	                              unsigned long long& clsCorrect,
	                              unsigned long long& clsTotal,
	                              shmea::GLogger* logger);
#endif
	void transformerCpuForwardPass(const TransformerEpochCfg& cfg, unsigned int T, unsigned int s,
	                               const std::vector<int>& tokenIds,
	                               const std::vector<int>& targetIds,
	                               const std::vector<unsigned char>& keyAllowed,
	                               unsigned int scratchOutSize, unsigned int sampleCount);
	void transformerCpuBackwardPass(const TransformerEpochCfg& cfg, unsigned int T, unsigned int s,
	                                const std::vector<int>& tokenIds,
	                                const std::vector<int>& targetIds,
	                                unsigned int scratchOutSize,
	                                unsigned int& seqInBatch,
	                                unsigned int& timeStepsInBatch);

#ifdef GLADES_HAVE_CUDA
	// Runs the GPU forward pass (embedding → per-layer blocks → final LN →
	// output head → softmax) for ONE sequence. Token IDs must already be
	// uploaded to gpuTransformerScratch->tokenIds (tokenLM mode) by the
	// caller. Does NOT compute loss/metrics and does NOT run backward.
	// Writes logits/probs to scratch buffers. Used for both the normal
	// training forward and the HELIOS FD-HVP probe's perturbed re-forward.
	//
	// gpuPerfOpaque: nullable pointer to TransformerGpuPerfBreakdown (cast
	// internally to avoid exposing the perf struct in this header).
	bool transformerGpuRunForwardOnly(const TransformerEpochCfg& cfg,
	                                  unsigned int T,
	                                  bool useBf16, bool useRope,
	                                  bool bf16WIn, bool bf16Wq, bool bf16Wk,
	                                  bool bf16Wv, bool bf16Wo, bool bf16W1,
	                                  bool bf16W2, bool bf16Head,
	                                  int ropeDimOverride,
	                                  void* gpuPerfOpaque);
#endif


	// Owned resources (used only in some construction paths)
	shmea::GPointer<NNInfo> ownedSkeleton;

	// === Core owned state (private; access via explicit API) ===
	//
	// Attached dataset for the active run (train/test) only.
	// This MUST NOT be used as a long-lived pointer: the caller owns DataInput and may delete
	// it immediately after train()/test() returns. `Trainer` is responsible for attaching and
	// detaching this pointer for the duration of a run.
	const DataInput* di;

	// Network architecture ("skeleton").
	//
	// The authoritative lifetime is owned by `ownedSkeleton` when present.
	// `skeleton` is a convenience alias for fast access.
	NNInfo* skeleton;

	// Train loop termination controls (epoch/accuracy/time, etc.).
	Terminator terminator;

	// Internal run-scope guard used by Trainer and internal APIs to enforce the concurrency policy.
	// If ok()==false, the caller must not touch or mutate the network (it is already running).
	class RunLockGuard
	{
	public:
		explicit RunLockGuard(NNetwork& n) : net(&n), acquired(false)
		{
			acquired = (net ? net->tryAcquireRunLock() : false);
		}
		~RunLockGuard()
		{
			if (acquired && net)
				net->releaseRunLock();
		}
		bool ok() const { return acquired; }
	private:
		NNetwork* net;
		bool acquired;
		// non-copyable
		RunLockGuard(const RunLockGuard&);
		RunLockGuard& operator=(const RunLockGuard&);
	};
public:
	enum
	{
		TYPE_DFF = 0,
		TYPE_RNN = 1,
		TYPE_GRU = 2,
		TYPE_LSTM = 3,
		// Transformer encoder: bidirectional self-attention over sequences.
		TYPE_TRANSFORMER_ENCODER = 4,
		// Transformer decoder-only: causal self-attention over sequences.
		TYPE_TRANSFORMER_DECODER = 5,
		// Convolutional neural network: im2col+SGEMM convolution, pooling, FC head.
		TYPE_CNN = 6,
		// CHIRON reversible-flow transformer: bijective symplectic blocks,
		// O(1)-in-depth activation memory (research/CHIRON_framework.md).
		// When enabled via cfg.chiron.enable, the training loop routes
		// forward/backward through CHIRON primitives (chiron_attention_shear,
		// chiron_reln_forward/inverse/backward, chiron_attention_shear_backward)
		// instead of storing activations. Phase A: dispatch enum + feature
		// flag. Phase B: full forward/backward orchestration.
		TYPE_TRANSFORMER_CHIRON = 7
	};

	enum
	{
		RUN_TRAIN = 0,
		RUN_TEST = 1,
		RUN_VALIDATE = 2
	};

	NNetwork(int=TYPE_DFF);
	// Construction from an external NNInfo is non-owning: the network clones and owns it internally.
	explicit NNetwork(const NNInfo* newNNInfo, int newNetType=TYPE_DFF);
	virtual ~NNetwork();
	void setSeed(uint64_t seed);
	uint64_t getSeed() const { return loadConfiguredSeed(); }
	// Create a fresh NNetwork from the same skeleton/type/config for hyperparameter tuning.
	// Caller owns the returned pointer and must delete it.
	NNetwork* cloneForTrial() const;
	// Returns the network's architecture type (TYPE_DFF, TYPE_RNN, etc.).
	int getNetType() const { return netType; }
	int64_t getCurrentTimeMilliseconds() const;
	bool getRunning() const;
	int getEpochs() const;
	void stop();
	// Unified, versioned persistence (architecture + weights).
	//
	// This is the production-facing API. It stores a self-contained "model package" under:
	//   database/models/<modelName>/{manifest.txt, nninfo.csv, weights.bin}
	//
	// Loading requires a DataInput instance to provide the input feature count so the
	// network tensors can be shaped before applying weights.
	NNetworkStatus saveModel(const std::string& modelName, const DataInput* externalDI = NULL) const;
	NNetworkStatus loadModel(const std::string& modelName, const DataInput* forShape, int netTypeOverride = -1);

	// === Tokenizer/vocab artifacts (optional) ===
	//
	// These APIs manage deployment metadata stored with model packages:
	//   database/models/<modelName>/tokenizer/{manifest.txt,vocab.bin}
	//
	// Thread-safety:
	// - Not safe to mutate while the network is running (same as trainingConfig/terminator).
	bool hasTokenizerArtifacts() const { return tokenizerArtifactsPresent; }
	const TokenizerArtifacts& getTokenizerArtifacts() const { return tokenizerArtifacts; }
	static NNetworkStatus validateTokenizerArtifacts(const TokenizerArtifacts& a);
	NNetworkStatus setTokenizerArtifacts(const TokenizerArtifacts& a);
	void clearTokenizerArtifacts();

	// === Scalable training checkpointing (resumable) ===
	//
	// Unlike saveModel/loadModel, checkpoints may include optimizer state and are stored in a
	// sharded format to avoid huge single files for large transformers.
	//
	// Layout:
	//   database/checkpoints/<checkpointName>/{manifest.txt, nninfo.csv, shard_000.bin, ...}
	//
	// Notes:
	// - This API is intended for resuming training. It is stricter than saveModel/loadModel:
	//   optimizer state is validated by tensor name and exact element count.
	// - Loading requires a DataInput instance to allocate tensors with the correct input feature
	//   count before applying checkpoint tensors.
	struct CheckpointConfig
	{
		// Maximum bytes per shard file. A value of 0 defaults to 1 GiB.
		size_t maxShardBytes;
		// If true, include optimizer state (momentum / Adam moments) in the checkpoint.
		bool includeOptimizerState;
		CheckpointConfig()
		    : maxShardBytes(static_cast<size_t>(1024ull * 1024ull * 1024ull)),
		      includeOptimizerState(true)
		{
		}
	};

	NNetworkStatus saveCheckpoint(const std::string& checkpointName, const CheckpointConfig& cfg = CheckpointConfig()) const;
	NNetworkStatus loadCheckpoint(const std::string& checkpointName, const DataInput* forShape, int netTypeOverride = -1);
	void setServer(GNet::GServer*, GNet::Connection*);
	// Structured logging support.
	// If no logger override is set, this will use the attached server logger (if any),
	// otherwise a default logger.
	void setLogger(shmea::GLogger* logger);
	shmea::GLogger* getLogger() const;

	// Stochastic Gradient Descent
	NNetworkStatus train(const DataInput*);
	NNetworkStatus test(const DataInput*);
	NNetworkStatus train(const DataInput*, ITrainingCallbacks*);
	NNetworkStatus test(const DataInput*, ITrainingCallbacks*);
	const NNetworkStatus& getLastStatus() const { return lastStatus; }
	bool getTrainerRunDiagnostics(TrainerRunDiagnostics& out) const;
	bool getPersistenceDiagnostics(PersistenceDiagnostics& out) const;

	// Training loop controls (optional).
	// These are intentionally simple knobs that do not require modifying NNInfo persistence.
	void setLearningRateScheduleNone();
	void setLearningRateScheduleStep(int stepSizeEpochs, float gamma);
	void setLearningRateScheduleExp(float gamma);
	void setLearningRateScheduleCosine(int tMaxEpochs, float minMultiplier);
	float getLearningRateMultiplier() const { return lrScheduleMultiplier; }
	void setLrScheduleEpochOffset(int offset) { lrScheduleEpochOffset = offset; }
	void setGlobalGradClipNorm(float clipNorm);
	float getGlobalGradClipNorm() const { return trainingConfig.globalGradClipNorm; }
	void setPerElementGradClip(float clipLimit);
	float getPerElementGradClip() const { return trainingConfig.perElementGradClip; }
	float getLastGradNorm() const { return lastGradNorm; }
	float getLastGradNormScale() const { return lastGradNormScale; }
	const TrainingConfig& getTrainingConfig() const { return trainingConfig; }
	// Mutable access to training config is supported for backwards compatibility.
	//
	// WARNING:
	// - Do not mutate this while the network is running.
	// - Prefer setTrainingConfig() for a single, validated update point.
	TrainingConfig& getTrainingConfigMutable() { return trainingConfig; }
	// Replace the training config as a single operation.
	// This fails if the network is currently running in another thread.
	NNetworkStatus setTrainingConfig(const TrainingConfig& cfg);

	int64_t getID() const;
	shmea::GString getName() const;
	// Architecture accessors.
	// - getNNInfo(): read-only view of the owned skeleton (may be NULL before load/build).
	const NNInfo* getNNInfo() const;
	// Run-scoped attached dataset (NULL when idle).
	const DataInput* getAttachedDataInput() const { return di; }
	// Terminator accessors.
	const Terminator& getTerminator() const { return terminator; }
	// Mutable access is supported for backwards compatibility.
	//
	// WARNING:
	// - Do not mutate this while the network is running.
	// - Prefer setTerminator() when possible.
	Terminator& getTerminatorMutable() { return terminator; }
	// Replace the terminator settings as a single operation.
	// This fails if the network is currently running in another thread.
	NNetworkStatus setTerminator(const Terminator& t);
	// Primary "accuracy-like" score used by Terminator and UI:
	// - Regression: R^2 expressed as percent in [0,100] (computed in Trainer).
	// - Classification/KL: top-1 accuracy expressed as percent in [0,100] (computed in Trainer).
	//
	// IMPORTANT: This must not return MCC. MCC has its own accessor.
	float getAccuracy() const;
	// Matthews correlation coefficient for classification/KL.
	// Returned in the same units produced by CMatrix (typically percent in this codebase).
	float getMCC() const;
	const CMatrix& getConfusionMatrix() const;
	const shmea::GList& getNodeActivations() const;
	bool getAtlasRuntimeDiagnostics(AtlasRuntimeDiagnostics& out) const;
	bool getTransformerGroupedParameterSnapshot(TransformerGroupedParameterSnapshot& out) const;

	// graphing
	shmea::GList getResults() const;

	// Returns weights in the same GUI serialization format used historically by the UI
	// followed by bias summaries.
	// This reads directly from tensor parameters.
	shmea::GList getWeightsForGui() const;

	// === Transformer token LM KV-cache inference sessions (re-entrant) ===
	//
	// These session APIs allow callers to own KV cache + scratch buffers per request (or per batch),
	// enabling:
	// - concurrent inference across threads using a shared, read-only model
	// - explicit memory ownership and reuse across requests
	// - allocation-free per-token append in hot loops (after Reset)
	//
	// Thread-safety:
	// - The session objects are owned by the caller and are not shared unless you share them.
	// - The NNetwork must not be mutated concurrently with session inference (i.e., do not train while serving).
	//
	#include "transformer_metrics_state.inc"

private:
	// Transformer serving/inference metrics configuration (default: disabled).
	TransformerMetricsConfig transformerMetricsCfg;
	mutable TransformerGpuPerfBreakdown lastTransformerTrainGpuPerf;
	mutable TransformerGpuPerfBreakdown lastTransformerInferGpuPerf;

public:
	// Configure structured transformer metrics/logging.
	// - When enabled, KV inference sessions will accumulate perf counters and the generation
	//   APIs will emit structured log lines through getLogger().
	void setTransformerMetricsConfig(const TransformerMetricsConfig& cfg) { transformerMetricsCfg = cfg; }
	const TransformerMetricsConfig& getTransformerMetricsConfig() const { return transformerMetricsCfg; }
	const TransformerGpuPerfBreakdown& getLastTransformerTrainGpuPerf() const { return lastTransformerTrainGpuPerf; }
	const TransformerGpuPerfBreakdown& getLastTransformerInferGpuPerf() const { return lastTransformerInferGpuPerf; }

	struct TransformerLmSession
	{
		enum KVCacheDType
		{
			KV_CACHE_F32 = 0,
			KV_CACHE_F16 = 1,
			KV_CACHE_BF16 = 2
		};

		bool isInitialized() const { return initialized; }
		unsigned int getMaxLen() const { return maxLen; }
		unsigned int getCurrentLength() const { return curLen; }

		TransformerLmSession()
		    : initialized(false),
		      maxLen(0u),
		      curLen(0u),
		      dModel(0u),
		      dFF(0u),
		      nHeads(0u),
		      nKVHeads(0u),
		      nLayers(0u),
		      dHead(0u),
		      dModelKV(0u),
		      ffnKind(0u),
		      ff1Width(0u),
		      kvCacheDType(KV_CACHE_F32),
		      k(),
		      v(),
		      k16(),
		      v16(),
		      keyValid(),
		      h(),
		      x1(),
		      x2(),
		      q(),
		      kvec(),
		      vvec(),
		      attnConcat(),
		      attnOut(),
		      ffPre(),
		      ffAct(),
		      ffOut(),
		      scores(),
		      posEncCache(),
		      metricsEnabled(false),
		      metricsBreakdownEnabled(false),
		      metricsLogPerKvAppend(false),
		      metricsGpuPerfEnabled(false),
		      perf(),
		      layerNormEps(0.0f),
		      normType(0u),
		      positionalEncoding(0u),
		      ropeDimOverride(0),
		      ropeTheta(0.0f),
		      ffnActivation(0u),
		      padTokenId(-1),
		      logger(NULL),
		      gpuInferState(0)
		{
		}

		// Destructor frees GPU inference state if allocated.
		// Implemented in transformer_infer.cpp to keep CUDA out of the header.
		~TransformerLmSession();

		void reset()
		{
			initialized = false;
			maxLen = 0u;
			curLen = 0u;
			dModel = dFF = nHeads = nKVHeads = nLayers = dHead = dModelKV = 0u;
			ffnKind = 0u;
			ff1Width = 0u;
			kvCacheDType = KV_CACHE_F32;
			k.clear();
			v.clear();
			k16.clear();
			v16.clear();
			keyValid.clear();
			h.clear();
			x1.clear();
			x2.clear();
			q.clear();
			kvec.clear();
			vvec.clear();
			attnConcat.clear();
			attnOut.clear();
			ffPre.clear();
			ffAct.clear();
			ffOut.clear();
			scores.clear();
			posEncCache.reset();
			metricsEnabled = false;
			metricsBreakdownEnabled = false;
			metricsLogPerKvAppend = false;
			metricsGpuPerfEnabled = false;
			perf.reset();
			layerNormEps = 0.0f;
			normType = 0u;
			positionalEncoding = 0u;
			ropeDimOverride = 0;
			ropeTheta = 0.0f;
			ffnActivation = 0u;
			padTokenId = -1;
			logger = NULL;
			// Note: gpuInferState is NOT freed here; the caller (transformerLmSessionReset)
			// manages GPU lifecycle to avoid pulling CUDA into the header.
		}

	private:
		friend class NNetwork;

		// Initialized-session contract:
		// - cached dimensions mirror the active transformer tensor layout
		// - exactly one KV storage pair is active (`k/v` for F32 or `k16/v16` for low-precision)
		// - scratch buffers are pre-sized so Append stays allocation-free
		// - `keyValid.size() == maxLen`
		bool usesLowPrecisionKvCache() const { return kvCacheDType != KV_CACHE_F32; }
		size_t kvElementsPerSequence() const
		{
			return static_cast<size_t>(nLayers) * static_cast<size_t>(maxLen) * static_cast<size_t>(dModelKV);
		}
		bool shapeMatches(unsigned int expectedDModel,
		                 unsigned int expectedDFF,
		                 unsigned int expectedNHeads,
		                 unsigned int expectedNKVHeads,
		                 unsigned int expectedNLayers,
		                 unsigned int expectedDHead,
		                 unsigned int expectedDModelKV,
		                 unsigned int expectedFfnKind,
		                 unsigned int expectedFf1Width) const
		{
			return dModel == expectedDModel &&
			       dFF == expectedDFF &&
			       nHeads == expectedNHeads &&
			       nKVHeads == expectedNKVHeads &&
			       nLayers == expectedNLayers &&
			       dHead == expectedDHead &&
			       dModelKV == expectedDModelKV &&
			       ffnKind == expectedFfnKind &&
			       ff1Width == expectedFf1Width;
		}
		bool storageInvariantsHold() const
		{
			if (curLen > maxLen ||
			    keyValid.size() != static_cast<size_t>(maxLen) ||
			    h.size() != static_cast<size_t>(dModel) ||
			    x1.size() != static_cast<size_t>(dModel) ||
			    x2.size() != static_cast<size_t>(dModel) ||
			    q.size() != static_cast<size_t>(dModel) ||
			    kvec.size() != static_cast<size_t>(dModelKV) ||
			    vvec.size() != static_cast<size_t>(dModelKV) ||
			    attnConcat.size() != static_cast<size_t>(dModel) ||
			    attnOut.size() != static_cast<size_t>(dModel) ||
			    ffPre.size() != static_cast<size_t>(ff1Width) ||
			    ffAct.size() != static_cast<size_t>(dFF) ||
			    ffOut.size() != static_cast<size_t>(dModel) ||
			    scores.size() != static_cast<size_t>(maxLen))
				return false;

			const size_t kvElems = kvElementsPerSequence();
			if (usesLowPrecisionKvCache())
				return k.empty() && v.empty() && k16.size() == kvElems && v16.size() == kvElems;
			return k.size() == kvElems && v.size() == kvElems && k16.empty() && v16.empty();
		}

		bool initialized;
		unsigned int maxLen;
		unsigned int curLen;
		// Cached model dims (for indexing and sanity).
		unsigned int dModel;
		unsigned int dFF;
		unsigned int nHeads;
		unsigned int nKVHeads;
		unsigned int nLayers;
		unsigned int dHead;
		unsigned int dModelKV;
		unsigned int ffnKind;
		unsigned int ff1Width;

		// KV-cache storage dtype (owned by the session).
		KVCacheDType kvCacheDType;

		// Cached K/V per layer: [nLayers, maxLen, dModelKV]
		// Exactly one storage is used based on kvCacheDType.
		std::vector<float, glades::AlignedAllocator<float, 64> > k;
		std::vector<float, glades::AlignedAllocator<float, 64> > v;
		// Low-precision KV cache storage:
		// - When kvCacheDType==KV_CACHE_F16: values are IEEE754 binary16 (FP16)
		// - When kvCacheDType==KV_CACHE_BF16: values are bfloat16 (BF16)
		std::vector<uint16_t, glades::AlignedAllocator<uint16_t, 64> > k16;
		std::vector<uint16_t, glades::AlignedAllocator<uint16_t, 64> > v16;
		// keyValid[pos] == 1 => real token, 0 => padding (masked out of attention)
		std::vector<unsigned char, glades::AlignedAllocator<unsigned char, 64> > keyValid; // [maxLen]

		// Scratch buffers sized in Reset and reused across appends.
		std::vector<float, glades::AlignedAllocator<float, 64> > h;          // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > x1;         // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > x2;         // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > q;          // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > kvec;       // [dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > vvec;       // [dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > attnConcat; // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > attnOut;    // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffPre;      // [ff1Width]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffAct;      // [dFF]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffOut;      // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > scores;     // [maxLen]

		// Positional encoding caches (owned by the session to avoid mutating NNetwork).
		// Reuses the same struct as the training-side cache to avoid divergent implementations.
		TransformerPosEncCache posEncCache;

		// Optional performance counters/timers (populated only when enabled).
		bool metricsEnabled;
		bool metricsBreakdownEnabled;
		bool metricsLogPerKvAppend;
		bool metricsGpuPerfEnabled;
		TransformerKvPerfBreakdown perf;
		float layerNormEps;
		unsigned int normType;
		unsigned int positionalEncoding;
		int ropeDimOverride;
		float ropeTheta;
		unsigned int ffnActivation;
		int padTokenId;
		shmea::GLogger* logger;

		// Opaque pointer to GPU inference state (allocated/freed by transformer_infer.cpp).
		// NULL when GPU inference is not active.
		void* gpuInferState;

	};

	struct TransformerLmBatchSession
	{
		enum KVCacheDType
		{
			KV_CACHE_F32 = 0,
			KV_CACHE_F16 = 1,
			KV_CACHE_BF16 = 2
		};

		bool isInitialized() const { return initialized; }
		unsigned int getBatchSize() const { return batchSize; }
		unsigned int getMaxLen() const { return maxLen; }
		unsigned int getCurrentLength(unsigned int index) const
		{
			return (index < curLen.size()) ? curLen[index] : 0u;
		}

		TransformerLmBatchSession()
		    : initialized(false),
		      batchSize(0u),
		      maxLen(0u),
		      curLen(),
		      dModel(0u),
		      dFF(0u),
		      nHeads(0u),
		      nKVHeads(0u),
		      nLayers(0u),
		      dHead(0u),
		      dModelKV(0u),
		      ffnKind(0u),
		      ff1Width(0u),
		      kvCacheDType(KV_CACHE_F32),
		      k(),
		      v(),
		      k16(),
		      v16(),
		      keyValid(),
		      h(),
		      x1(),
		      x2(),
		      q(),
		      kvec(),
		      vvec(),
		      attnConcat(),
		      attnOut(),
		      ffPre(),
		      ffAct(),
		      ffOut(),
		      scores(),
		      posEncCache(),
		      metricsEnabled(false),
		      metricsBreakdownEnabled(false),
		      metricsLogPerKvAppend(false),
		      metricsGpuPerfEnabled(false),
		      perf(),
		      layerNormEps(0.0f),
		      normType(0u),
		      positionalEncoding(0u),
		      ropeDimOverride(0),
		      ropeTheta(0.0f),
		      ffnActivation(0u),
		      padTokenId(-1),
		      logger(NULL)
		{
		}

		void reset()
		{
			initialized = false;
			batchSize = 0u;
			maxLen = 0u;
			curLen.clear();
			dModel = dFF = nHeads = nKVHeads = nLayers = dHead = dModelKV = 0u;
			ffnKind = 0u;
			ff1Width = 0u;
			kvCacheDType = KV_CACHE_F32;
			k.clear();
			v.clear();
			k16.clear();
			v16.clear();
			keyValid.clear();
			h.clear();
			x1.clear();
			x2.clear();
			q.clear();
			kvec.clear();
			vvec.clear();
			attnConcat.clear();
			attnOut.clear();
			ffPre.clear();
			ffAct.clear();
			ffOut.clear();
			scores.clear();
			posEncCache.reset();
			metricsEnabled = false;
			metricsBreakdownEnabled = false;
			metricsLogPerKvAppend = false;
			metricsGpuPerfEnabled = false;
			perf.reset();
			layerNormEps = 0.0f;
			normType = 0u;
			positionalEncoding = 0u;
			ropeDimOverride = 0;
			ropeTheta = 0.0f;
			ffnActivation = 0u;
			padTokenId = -1;
			logger = NULL;
		}

	private:
		friend class NNetwork;

		// Initialized-session contract:
		// - cached dimensions mirror the active transformer tensor layout
		// - `curLen.size() == batchSize` and every entry stays <= maxLen
		// - exactly one KV storage pair is active (`k/v` for F32 or `k16/v16` for low-precision)
		// - scratch buffers are shared across batch elements and pre-sized so Append stays allocation-free
		// - `keyValid.size() == batchSize * maxLen`
		bool usesLowPrecisionKvCache() const { return kvCacheDType != KV_CACHE_F32; }
		size_t kvElementsPerSequence() const
		{
			return static_cast<size_t>(nLayers) * static_cast<size_t>(maxLen) * static_cast<size_t>(dModelKV);
		}
		size_t kvElementsTotal() const
		{
			return static_cast<size_t>(batchSize) * kvElementsPerSequence();
		}
		bool shapeMatches(unsigned int expectedDModel,
		                 unsigned int expectedDFF,
		                 unsigned int expectedNHeads,
		                 unsigned int expectedNKVHeads,
		                 unsigned int expectedNLayers,
		                 unsigned int expectedDHead,
		                 unsigned int expectedDModelKV,
		                 unsigned int expectedFfnKind,
		                 unsigned int expectedFf1Width) const
		{
			return dModel == expectedDModel &&
			       dFF == expectedDFF &&
			       nHeads == expectedNHeads &&
			       nKVHeads == expectedNKVHeads &&
			       nLayers == expectedNLayers &&
			       dHead == expectedDHead &&
			       dModelKV == expectedDModelKV &&
			       ffnKind == expectedFfnKind &&
			       ff1Width == expectedFf1Width;
		}
		bool storageInvariantsHold() const
		{
			if (curLen.size() != static_cast<size_t>(batchSize) ||
			    keyValid.size() != (static_cast<size_t>(batchSize) * static_cast<size_t>(maxLen)) ||
			    h.size() != static_cast<size_t>(dModel) ||
			    x1.size() != static_cast<size_t>(dModel) ||
			    x2.size() != static_cast<size_t>(dModel) ||
			    q.size() != static_cast<size_t>(dModel) ||
			    kvec.size() != static_cast<size_t>(dModelKV) ||
			    vvec.size() != static_cast<size_t>(dModelKV) ||
			    attnConcat.size() != static_cast<size_t>(dModel) ||
			    attnOut.size() != static_cast<size_t>(dModel) ||
			    ffPre.size() != static_cast<size_t>(ff1Width) ||
			    ffAct.size() != static_cast<size_t>(dFF) ||
			    ffOut.size() != static_cast<size_t>(dModel) ||
			    scores.size() != static_cast<size_t>(maxLen))
				return false;

			for (size_t i = 0u; i < curLen.size(); ++i)
				if (curLen[i] > maxLen)
					return false;

			const size_t kvElems = kvElementsTotal();
			if (usesLowPrecisionKvCache())
				return k.empty() && v.empty() && k16.size() == kvElems && v16.size() == kvElems;
			return k.size() == kvElems && v.size() == kvElems && k16.empty() && v16.empty();
		}

		bool initialized;
		unsigned int batchSize;
		unsigned int maxLen;
		// Per-sequence current lengths.
		std::vector<unsigned int> curLen; // [batchSize]

		// Cached model dims.
		unsigned int dModel;
		unsigned int dFF;
		unsigned int nHeads;
		unsigned int nKVHeads;
		unsigned int nLayers;
		unsigned int dHead;
		unsigned int dModelKV;
		unsigned int ffnKind;
		unsigned int ff1Width;

		// KV-cache storage dtype (owned by the session).
		KVCacheDType kvCacheDType;

		// Cached K/V per sequence:
		// - k/v are laid out as [batchSize, nLayers, maxLen, dModelKV] in a contiguous buffer.
		// Exactly one storage is used based on kvCacheDType.
		std::vector<float, glades::AlignedAllocator<float, 64> > k;
		std::vector<float, glades::AlignedAllocator<float, 64> > v;
		// Low-precision KV cache storage (same encoding as TransformerLmSession::k16/v16).
		std::vector<uint16_t, glades::AlignedAllocator<uint16_t, 64> > k16;
		std::vector<uint16_t, glades::AlignedAllocator<uint16_t, 64> > v16;
		// keyValid per sequence: [batchSize, maxLen]
		std::vector<unsigned char, glades::AlignedAllocator<unsigned char, 64> > keyValid;

		// Shared scratch buffers (reused while looping over batch elements).
		std::vector<float, glades::AlignedAllocator<float, 64> > h;          // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > x1;         // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > x2;         // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > q;          // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > kvec;       // [dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > vvec;       // [dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > attnConcat; // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > attnOut;    // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffPre;      // [ff1Width]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffAct;      // [dFF]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffOut;      // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > scores;     // [maxLen]

		// Positional encoding caches (session-owned).
		// Reuses the same struct as the training-side cache to avoid divergent implementations.
		TransformerPosEncCache posEncCache;

		// Optional performance counters/timers (populated only when enabled).
		bool metricsEnabled;
		bool metricsBreakdownEnabled;
		bool metricsLogPerKvAppend;
		bool metricsGpuPerfEnabled;
		TransformerKvPerfBreakdown perf;
		float layerNormEps;
		unsigned int normType;
		unsigned int positionalEncoding;
		int ropeDimOverride;
		float ropeTheta;
		unsigned int ffnActivation;
		int padTokenId;
		shmea::GLogger* logger;

	};

	// Session APIs (const: do not mutate NNetwork inference state).
	NNetworkStatus transformerLmSessionReset(TransformerLmSession& session, unsigned int maxSeqLen) const;
	NNetworkStatus transformerLmSessionAppend(TransformerLmSession& session, unsigned int tokenId, std::vector<float>* outLogits /* optional */) const;

	NNetworkStatus transformerLmBatchSessionReset(TransformerLmBatchSession& session, unsigned int batchSize, unsigned int maxSeqLen) const;
	// Append one token for each active batch element (ragged-safe):
		// - Only active[b]!=0 advances the current length tracked for that batch element
	// - If tokenValid is provided and tokenValid[b]==0, the position is treated as padding and masked out of attention
	// - If outLogitsFlat is provided, it is resized to [batchSize * vocabSize] and filled row-major; inactive rows are zeros
	NNetworkStatus transformerLmBatchSessionAppendSelective(TransformerLmBatchSession& session,
	                                                       const std::vector<unsigned int>& tokenIds,
	                                                       const std::vector<unsigned char>* tokenValid /* optional */,
	                                                       const std::vector<unsigned char>& active,
	                                                       std::vector<float>* outLogitsFlat /* optional */) const;
	NNetworkStatus transformerLmBatchSessionAppendSelective(TransformerLmBatchSession& session,
	                                                       const std::vector<unsigned int>& tokenIds,
	                                                       const std::vector<unsigned char>& active,
	                                                       std::vector<float>* outLogitsFlat /* optional */) const
	{
		return transformerLmBatchSessionAppendSelective(session, tokenIds, NULL, active, outLogitsFlat);
	}

	// === Transformer token LM generation API (decoder-only, KV-cache) ===
	//
	// This is the production-facing "real inference" API:
	// - KV-cache prefill on a prompt
	// - iterative decode with greedy or sampling (temperature/top-k/top-p)
	// - streaming callbacks for token emission / cancellation
	//
	// IMPORTANT:
	// - This API allocates and uses a per-call KV session (no internal KV state is retained).
	// - It fails fast if the same NNetwork instance is already running training/eval/inference.
	// - This API requires token LM mode (enableTokenEmbedding==true) and decoder net type.
	// - For supported transformer-facing callers, prefer TransformerPublicAPI / TransformerPublicAPI::runtime(net).
	// Legacy compatibility aliases only.
	// New code should include transformer_types.h and use the freestanding names directly.
	typedef glades::TransformerGenerateConfig TransformerGenerateConfig;
	typedef glades::TransformerGenerateResult TransformerGenerateResult;
	typedef glades::ITransformerGenerateCallbacks ITransformerGenerateCallbacks;

	// Generate tokens given a prompt (token IDs).
	// - `promptTokens` must be non-empty (callers should include a BOS token if needed).
	// - `out` is always overwritten.
	NNetworkStatus transformerLmGenerate(const std::vector<unsigned int>& promptTokens,
	                                    const TransformerGenerateConfig& cfg,
	                                    TransformerGenerateResult& out,
	                                    ITransformerGenerateCallbacks* cb /* optional */) const;

	// === Serving-grade generation (batched, ragged prompts, continuous decode) ===
	//
	// This API is designed for "real serving" needs:
	// - Multiple requests in one call (batching)
	// - Ragged prompts without positional-encoding distortion
	// - Per-request early stop (EOS/limits/callback cancellation)
	// - Token streaming callbacks with request index
	//
	// Implementation notes:
	// - Uses the internal batched KV cache with selective appends (no fake padding positions).
	// - Still scalar (loops requests), but allocation-free per decode step.
	typedef glades::TransformerServeRequest TransformerServeRequest;
	typedef glades::TransformerServeBatchResult TransformerServeBatchResult;
	typedef glades::ITransformerServeCallbacks ITransformerServeCallbacks;

	// === Continuous batching scheduler (persistent) ===
	//
	// This is the "real batching architecture" primitive used by serving stacks:
	// - Create a batcher with a fixed capacity and max sequence length.
	// - Submit requests into slots (join) and remove them when done (leave).
	// - Call Step() repeatedly to advance all active requests by one token append:
	//   - requests still in prompt prefill append one prompt token
	//   - requests in decode sample + append one generated token
	//
	// Ownership:
	// - The batcher owns all request state (prompt tokens, stop tokens, results) per slot.
	// - The caller owns the batcher object and can reuse it across multiple batches.
	//
	// Thread-safety:
	// - A batcher is not internally synchronized; do not call Step/Submit/Remove concurrently
	//   on the same batcher from multiple threads.
	// - Multiple batchers may be used concurrently with the same NNetwork only while the network
	//   is otherwise idle; public Reset/Step/generate/forward entry points fail fast if the
	//   network is already running training/eval/inference.
	// - Prefer TransformerPublicAPI for the supported one-shot runtime surface; these low-level
	//   batcher/session entry points remain on NNetwork for compatibility and advanced callers.
	struct TransformerServeBatcherConfig
	{
		// Maximum number of concurrent requests (slots) in this batcher.
		unsigned int maxBatchSize;
		// Maximum KV cache length per request. Requests with larger maxSeqLen are rejected.
		unsigned int maxSeqLen;
		// If true, zero-out the used KV prefix when removing a slot, including
		// both FP32 and low-precision KV cache storage.
		// This is more secure but can be expensive for large models/long sequences.
		bool wipeKvOnRemove;
		// Seed for the batcher's shared RNG stream (used when a request does not provide rngSeedOverride).
		// 0 => derive from this network's seed.
		uint64_t rngSeed;

		TransformerServeBatcherConfig()
		    : maxBatchSize(0u),
		      maxSeqLen(0u),
		      wipeKvOnRemove(false),
		      rngSeed(0ULL)
		{
		}
	};

	struct TransformerServeBatcher
	{
	public:
		enum SlotLifecycle
		{
			SLOT_FREE = 0,
			SLOT_PREFILL,
			SLOT_DECODE,
			SLOT_DONE
		};

		bool isInitialized() const { return initialized; }
		unsigned int capacity() const { return maxBatchSize; }
		bool slotFree(unsigned int slot) const
		{
			return slot < inUse.size() && inUse[slot] == 0u;
		}
		bool slotInUse(unsigned int slot) const
		{
			return !slotFree(slot);
		}
		bool slotDone(unsigned int slot) const
		{
			return slot < done.size() && done[slot] != 0u;
		}
		bool slotInPrefill(unsigned int slot) const
		{
			return slot < promptPos.size() && slot < promptLen.size() && promptPos[slot] < promptLen[slot];
		}
		bool slotCanDecode(unsigned int slot) const
		{
			return slotInUse(slot) && !slotDone(slot) &&
			       !slotInPrefill(slot) &&
			       slot < generated.size() &&
			       slot < reqMaxNew.size() &&
			       generated[slot] < reqMaxNew[slot];
		}
		unsigned int slotCurrentLen(unsigned int slot) const
		{
			return slot < session.curLen.size() ? session.curLen[slot] : 0u;
		}
		bool slotReachedMaxLen(unsigned int slot) const
		{
			return slot < reqMaxLen.size() && slotCurrentLen(slot) >= reqMaxLen[slot];
		}
		SlotLifecycle slotLifecycle(unsigned int slot) const
		{
			if (slotFree(slot))
				return SLOT_FREE;
			if (slotDone(slot))
				return SLOT_DONE;
			return slotInPrefill(slot) ? SLOT_PREFILL : SLOT_DECODE;
		}
		const TransformerGenerateResult* slotResult(unsigned int slot) const
		{
			return slot < results.size() ? &results[slot] : NULL;
		}

		TransformerServeBatcher()
		    : initialized(false),
		      vocab(0u),
		      maxBatchSize(0u),
		      maxSeqLen(0u),
		      wipeKvOnRemove(false),
		      session(),
		      inUse(),
		      done(),
		      promptPos(),
		      promptLen(),
		      generated(),
		      reqMaxNew(),
		      reqMaxLen(),
		      req(),
		      results(),
		      batchEngine(),
		      overrideEngines(),
		      hasOverride(),
		      tokenIds(),
		      active(),
		      sampledTok(),
		      sampledIsValid(),
		      prevLogitsFlat(),
		      logitsFlat(),
		      idxScratch(),
		      weightScratch()
		{
		}

		void reset()
		{
			initialized = false;
			vocab = 0u;
			maxBatchSize = 0u;
			maxSeqLen = 0u;
			wipeKvOnRemove = false;
			session.reset();

			inUse.clear();
			done.clear();
			promptPos.clear();
			promptLen.clear();
			generated.clear();
			reqMaxNew.clear();
			reqMaxLen.clear();
			req.clear();
			results.clear();

			overrideEngines.clear();
			hasOverride.clear();

			tokenIds.clear();
			active.clear();
			sampledTok.clear();
			sampledIsValid.clear();
			prevLogitsFlat.clear();
			logitsFlat.clear();
			idxScratch.clear();
			weightScratch.clear();
		}

	private:
		friend class NNetwork;

		void zeroLogitsRow(unsigned int slot)
		{
			if (slot >= maxBatchSize || vocab == 0u)
				return;
			const size_t offset = static_cast<size_t>(slot) * static_cast<size_t>(vocab);
			if (!prevLogitsFlat.empty())
			{
				float* row = &prevLogitsFlat[offset];
				std::fill(row, row + vocab, 0.0f);
			}
			if (!logitsFlat.empty())
			{
				float* row = &logitsFlat[offset];
				std::fill(row, row + vocab, 0.0f);
			}
		}

		void installSlotRequest(unsigned int slot,
		                       const TransformerServeRequest& newReq,
		                       unsigned int newPromptLen,
		                       unsigned int newMaxNew,
		                       unsigned int newMaxLen)
		{
			if (slot >= maxBatchSize)
				return;
			req[slot] = newReq;
			results[slot] = TransformerGenerateResult();
			if (newReq.cfg.includePromptInOutput)
				results[slot].tokens = newReq.promptTokens;
			inUse[slot] = 1u;
			done[slot] = 0u;
			promptPos[slot] = 0u;
			promptLen[slot] = newPromptLen;
			generated[slot] = 0u;
			reqMaxNew[slot] = newMaxNew;
			reqMaxLen[slot] = newMaxLen;
			if (slot < session.curLen.size())
				session.curLen[slot] = 0u;
			hasOverride[slot] = 0u;
			if (slot < tokenIds.size())
				tokenIds[slot] = 0u;
			if (slot < active.size())
				active[slot] = 0u;
			if (slot < sampledTok.size())
				sampledTok[slot] = 0u;
			if (slot < sampledIsValid.size())
				sampledIsValid[slot] = 0u;
			zeroLogitsRow(slot);
		}

		void clearSlotState(unsigned int slot)
		{
			if (slot >= maxBatchSize)
				return;
			inUse[slot] = 0u;
			done[slot] = 0u;
			promptPos[slot] = 0u;
			promptLen[slot] = 0u;
			generated[slot] = 0u;
			reqMaxNew[slot] = 0u;
			reqMaxLen[slot] = 0u;
			if (slot < session.curLen.size())
				session.curLen[slot] = 0u;
			hasOverride[slot] = 0u;
			req[slot] = TransformerServeRequest();
			results[slot] = TransformerGenerateResult();
			if (slot < tokenIds.size())
				tokenIds[slot] = 0u;
			if (slot < active.size())
				active[slot] = 0u;
			if (slot < sampledTok.size())
				sampledTok[slot] = 0u;
			if (slot < sampledIsValid.size())
				sampledIsValid[slot] = 0u;
			zeroLogitsRow(slot);
		}

		void markSlotStoppedByLimit(unsigned int slot)
		{
			if (slot >= maxBatchSize)
				return;
			results[slot].stoppedByCallback = false;
			results[slot].stoppedOnEos = false;
			results[slot].stoppedByStopToken = false;
			results[slot].stoppedByLimit = true;
			done[slot] = 1u;
		}

		void markSlotStoppedByCallback(unsigned int slot)
		{
			if (slot >= maxBatchSize)
				return;
			results[slot].stoppedByCallback = true;
			results[slot].stoppedOnEos = false;
			results[slot].stoppedByStopToken = false;
			results[slot].stoppedByLimit = false;
			done[slot] = 1u;
		}

		bool initialized;
		unsigned int vocab;
		unsigned int maxBatchSize;
		unsigned int maxSeqLen;
		bool wipeKvOnRemove;

		// One KV-cache session sized for [maxBatchSize, maxSeqLen].
		TransformerLmBatchSession session;

		// Per-slot state (size maxBatchSize).
		std::vector<unsigned char> inUse;
		std::vector<unsigned char> done;
		std::vector<unsigned int> promptPos;
		std::vector<unsigned int> promptLen;
		std::vector<unsigned int> generated;
		std::vector<unsigned int> reqMaxNew;
		std::vector<unsigned int> reqMaxLen;

		// Request payload per slot (owned).
		std::vector<TransformerServeRequest> req;
		// Results per slot (owned).
		std::vector<TransformerGenerateResult> results;

		// RNG: one shared stream for non-overridden requests, and optional per-slot overrides.
		glades::rng::Engine batchEngine;
		std::vector<glades::rng::Engine> overrideEngines;
		std::vector<unsigned char> hasOverride;

		// Hot-loop buffers (no per-step allocations after Reset).
		std::vector<unsigned int> tokenIds;
		std::vector<unsigned char> active;
		std::vector<unsigned int> sampledTok;     // only meaningful for decode slots in the current step
		std::vector<unsigned char> sampledIsValid;
		std::vector<float> prevLogitsFlat; // [B, vocab]
		std::vector<float> logitsFlat;     // [B, vocab]
		// Sampling scratch (reused across slots; Step processes slots sequentially for sampling).
		std::vector<unsigned int> idxScratch;
		std::vector<float> weightScratch;

	};

	// Initialize/reset a persistent continuous batcher.
	// After reset, the batcher has no active requests; callers may Submit() requests into free slots.
	NNetworkStatus transformerLmServeBatcherReset(TransformerServeBatcher& batcher,
	                                             const TransformerServeBatcherConfig& cfg) const;
	// Submit a request into a free slot. Returns the slot index in outSlot.
	NNetworkStatus transformerLmServeBatcherSubmit(TransformerServeBatcher& batcher,
	                                              const TransformerServeRequest& request,
	                                              unsigned int& outSlot) const;
	// Remove (free) a slot. Safe to call on done or cancelled slots.
	NNetworkStatus transformerLmServeBatcherRemove(TransformerServeBatcher& batcher, unsigned int slot) const;
	// Advance all active slots by one append step (prompt prefill or decode).
	// - Emits callbacks for generated tokens.
	// - Does not allocate on the hot path after Reset (subject to request submission copying).
	NNetworkStatus transformerLmServeBatcherStep(TransformerServeBatcher& batcher,
	                                            ITransformerServeCallbacks* cb /* optional */) const;
	// Mark an in-use slot as callback-stopped without removing it from the batcher.
	// Intended for serving runtimes that need to defer terminalization until after user callbacks return.
	NNetworkStatus transformerLmServeBatcherCancelSlot(TransformerServeBatcher& batcher, unsigned int slot) const;

	// Batched generation entrypoint.
	// - Requests must be non-empty; each request must have a non-empty promptTokens.
	// - out.results is always overwritten and sized to requests.size().
	NNetworkStatus transformerLmServeGenerateBatch(const std::vector<TransformerServeRequest>& requests,
	                                              TransformerServeBatchResult& out,
	                                              ITransformerServeCallbacks* cb /* optional */) const;

	// === Transformer token LM full forward inference (debug/test) ===
	//
	// Computes the logits for the *last* token position of a full forward pass over `tokenIds`.
	// This is intended for unit testing (e.g., validating KV-cache parity) and small-scale debugging.
	//
	// Preconditions:
	// - netType == TYPE_TRANSFORMER_DECODER
	// - trainingConfig.transformer.enableTokenEmbedding == true
	// - tensorTransformer.initialized == true
	//
	// Output:
	// - outLogits is resized to vocabSize and filled with unnormalized logits.
	NNetworkStatus transformerLmForwardLastLogits(const std::vector<unsigned int>& tokenIds,
	                                             std::vector<float>& outLogits) const;
};
};

#endif
