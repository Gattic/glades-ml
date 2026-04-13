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
#include "network.h"
#include "transformer_config.h"
#include "transformer_public_api.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Database/GLogger.h"
#include "Backend/Database/ServiceData.h"
#include "Backend/Networking/main.h"
#include "../GMath/OHE.h"
#include "../GMath/cmatrix.h"
#include "../GMath/gmath.h"
#include "../Structure/nninfo.h"
#include "../DataObjects/NumberInput.h"
#include "../DataObjects/ImageInput.h"
#include "trainer.h"
#include "param_layout.h"
#include "ddp_comm.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <new>
#include <sstream>
#include <stdexcept>

#include "logfmt_utils.h"

using namespace glades;
using namespace glades::logfmt;

// Force-link optional DataObjects translation units that otherwise may be discarded when building
// libglades.so from static sub-libraries. These are public APIs used by production/CLI consumers.
extern "C" void glades_link_anchor_mappedmatrix();
extern "C" void glades_link_anchor_mappednumberinput();
// Force-link tokenizer/vocab artifacts translation unit (public API surface).
extern "C" void glades_link_anchor_tokenizer_artifacts();

namespace {
struct GladesLinkAnchorsOnce
{
	GladesLinkAnchorsOnce()
	{
		glades_link_anchor_mappedmatrix();
		glades_link_anchor_mappednumberinput();
		glades_link_anchor_tokenizer_artifacts();
	}
};
static GladesLinkAnchorsOnce g_glades_link_anchors_once;
static shmea::GLogger g_default_network_logger(shmea::GLogger::LOG_INFO);

bool atlas_transformer_needs_adam_moments(const glades::TrainingConfig& trainingConfig)
{
	return (trainingConfig.optimizer.type != glades::OptimizerConfig::ATLAS)
	    || (trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS
	        && ((trainingConfig.atlas.auroraEnabled
	             && trainingConfig.atlas.auroraAdamwBackbone)
	            || trainingConfig.atlas.geodeEnabled
	            || trainingConfig.atlas.echoEnabled
	            || trainingConfig.atlas.bimapEnabled
	            || trainingConfig.atlas.pactEnabled
	            || trainingConfig.atlas.racerEnabled
	            || trainingConfig.atlas.kronEnabled
	            || trainingConfig.atlas.matraEnabled
	            || trainingConfig.atlas.muonEnabled));
}

void ensure_transformer_moment_buffer(std::vector<float>& buffer, size_t wanted)
{
	if (buffer.size() != wanted)
		buffer.assign(wanted, 0.0f);
}

#ifdef GLADES_HAVE_CUDA
bool gpu_transformer_has_adam_moments(const glades::gpu::GpuTransformerWeights& gt)
{
	if (gt.tokenModel)
	{
		if (gt.vTokE.size() != gt.tokE.size() || gt.v2TokE.size() != gt.tokE.size())
			return false;
		if (gt.mLmBias.size() != gt.lmBias.size() || gt.v2LmBias.size() != gt.lmBias.size())
			return false;
	}
	else
	{
		if (gt.vWIn.size() != gt.WIn.size() || gt.v2WIn.size() != gt.WIn.size())
			return false;
		if (gt.mBIn.size() != gt.bIn.size() || gt.v2BIn.size() != gt.bIn.size())
			return false;
		if (gt.vWOut.size() != gt.WOut.size() || gt.v2WOut.size() != gt.WOut.size())
			return false;
		if (gt.mBOut.size() != gt.bOut.size() || gt.v2BOut.size() != gt.bOut.size())
			return false;
	}
	if (gt.mLnFinalGamma.size() != gt.lnFinalGamma.size() || gt.v2LnFinalGamma.size() != gt.lnFinalGamma.size())
		return false;
	if (gt.mLnFinalBeta.size() != gt.lnFinalBeta.size() || gt.v2LnFinalBeta.size() != gt.lnFinalBeta.size())
		return false;
	for (unsigned int li = 0u; li < gt.nLayers; ++li)
	{
		const glades::gpu::GpuTransformerWeights::Block& b = gt.blocks[li];
		if (b.mLn1Gamma.size() != b.ln1Gamma.size() || b.v2Ln1Gamma.size() != b.ln1Gamma.size())
			return false;
		if (b.mLn1Beta.size() != b.ln1Beta.size() || b.v2Ln1Beta.size() != b.ln1Beta.size())
			return false;
		if (b.vWq.size() != b.Wq.size() || b.v2Wq.size() != b.Wq.size())
			return false;
		if (b.vWk.size() != b.Wk.size() || b.v2Wk.size() != b.Wk.size())
			return false;
		if (b.vWv.size() != b.Wv.size() || b.v2Wv.size() != b.Wv.size())
			return false;
		if (b.vWo.size() != b.Wo.size() || b.v2Wo.size() != b.Wo.size())
			return false;
		if (b.mBq.size() != b.bq.size() || b.v2Bq.size() != b.bq.size())
			return false;
		if (b.mBk.size() != b.bk.size() || b.v2Bk.size() != b.bk.size())
			return false;
		if (b.mBv.size() != b.bv.size() || b.v2Bv.size() != b.bv.size())
			return false;
		if (b.mBo.size() != b.bo.size() || b.v2Bo.size() != b.bo.size())
			return false;
		if (b.mLn2Gamma.size() != b.ln2Gamma.size() || b.v2Ln2Gamma.size() != b.ln2Gamma.size())
			return false;
		if (b.mLn2Beta.size() != b.ln2Beta.size() || b.v2Ln2Beta.size() != b.ln2Beta.size())
			return false;
		if (b.vW1.size() != b.W1.size() || b.v2W1.size() != b.W1.size())
			return false;
		if (b.vW2.size() != b.W2.size() || b.v2W2.size() != b.W2.size())
			return false;
		if (b.mB1.size() != b.b1.size() || b.v2B1.size() != b.b1.size())
			return false;
		if (b.mB2.size() != b.b2.size() || b.v2B2.size() != b.b2.size())
			return false;
	}
	return true;
}
#endif

struct AtlasRuntimeAccumulator
{
	unsigned int atlasMatrices;
	unsigned int sparrowMatrices;
	unsigned int sparrowMode2Matrices;
	double sparrowActiveModesSum;
	double sparrowEdgeSum;
	double sparrowSecondEdgeSum;
	double sparrowSecondEdgeRatioSum;
	double sparrowMemoryGainSum;
	double sparrowHorizontalRatioSum;
	unsigned int helmMatrices;
	unsigned int helmMode2Matrices;
	double helmActiveModesSum;
	double helmEdgeSum;
	double helmSecondEdgeSum;
	double helmSecondEdgeRatioSum;
	double helmSigmaSum;
	double helmPredR2Sum;
	double helmMemoryGainSum;
	double helmPoleSum;
	unsigned int asterMatrices;
	unsigned int asterMode2Matrices;
	double asterActiveModesSum;
	double asterEdgeSum;
	double asterSecondEdgeSum;
	double asterSecondEdgeRatioSum;
	double asterSigmaSum;
	double asterPredR2Sum;
	double asterMemoryGainSum;
	double asterPoleSum;
	double asterBoundaryMsSum;
	double asterSetupMsSum;
	double asterTransportMsSum;
	double asterTransferFitMsSum;
	double asterStateFitMsSum;
	double asterInnovationFitMsSum;
	double asterApplyMsSum;
	unsigned int aegisMatrices;
	double aegisLambdaSpatialSum;
	double aegisLambdaPredictiveSum;
	double aegisLambdaOutputSum;
	double aegisPredictivePredictedSum;
	double aegisPredictiveRealizedSum;
	double aegisOutputPredictedSum;
	double aegisOutputRealizedSum;
	double aegisPredictiveErrorSum;
	double aegisOutputErrorSum;
	double aegisChannelDisagreementSum;
	unsigned int citadelMatrices;
	double citadelAnchorSum;
	double citadelHardRegimeMassSum;
	double citadelSparrowTrustSum;
	unsigned int rampartMatrices;
	double rampartTauSum;
	double rampartBudgetSum;
	double rampartCovarianceSum;
	double rampartSparrowTrustSum;
	unsigned int meritMatrices;
	double meritTauSum;
	double meritBudgetSum;
	double meritCovarianceSum;
	double meritSparrowTrustSum;
	double meritGeometryTrustSum;
	unsigned int strataMatrices;
	double strataNullModeSum;
	double strataPredictiveModeSum;
	double strataOutputModeSum;
	double strataCoupledModeSum;
	double strataBudgetSum;
	double strataNullBenefitSum;
	double strataPredictiveBenefitSum;
	double strataOutputBenefitSum;
	double strataCoupledBenefitSum;
	double strataSelectedExcessSum;
	double strataSwitchRateSum;
	unsigned int transformerGapBatches;
	double transformerInputUpdateNormSum;
	std::vector<double> transformerBlockUpdateNormSum;
	double transformerFinalNormUpdateNormSum;
	double transformerHeadUpdateNormSum;
	double transformerHeadShareSum;
	double transformerNonHeadShareSum;
	double transformerApplyMsSum;
	unsigned int transformerMarginSnapshots;
	double transformerTargetMarginSum;
	double transformerHardNegativeLogitSum;

	AtlasRuntimeAccumulator()
	    : atlasMatrices(0u),
	      sparrowMatrices(0u),
	      sparrowMode2Matrices(0u),
	      sparrowActiveModesSum(0.0),
	      sparrowEdgeSum(0.0),
	      sparrowSecondEdgeSum(0.0),
	      sparrowSecondEdgeRatioSum(0.0),
	      sparrowMemoryGainSum(0.0),
	      sparrowHorizontalRatioSum(0.0),
	      helmMatrices(0u),
	      helmMode2Matrices(0u),
	      helmActiveModesSum(0.0),
	      helmEdgeSum(0.0),
	      helmSecondEdgeSum(0.0),
	      helmSecondEdgeRatioSum(0.0),
	      helmSigmaSum(0.0),
	      helmPredR2Sum(0.0),
	      helmMemoryGainSum(0.0),
	      helmPoleSum(0.0),
	      asterMatrices(0u),
	      asterMode2Matrices(0u),
	      asterActiveModesSum(0.0),
	      asterEdgeSum(0.0),
	      asterSecondEdgeSum(0.0),
	      asterSecondEdgeRatioSum(0.0),
	      asterSigmaSum(0.0),
	      asterPredR2Sum(0.0),
	      asterMemoryGainSum(0.0),
	      asterPoleSum(0.0),
	      asterBoundaryMsSum(0.0),
	      asterSetupMsSum(0.0),
	      asterTransportMsSum(0.0),
	      asterTransferFitMsSum(0.0),
	      asterStateFitMsSum(0.0),
	      asterInnovationFitMsSum(0.0),
	      asterApplyMsSum(0.0),
	      aegisMatrices(0u),
	      aegisLambdaSpatialSum(0.0),
	      aegisLambdaPredictiveSum(0.0),
	      aegisLambdaOutputSum(0.0),
	      aegisPredictivePredictedSum(0.0),
	      aegisPredictiveRealizedSum(0.0),
	      aegisOutputPredictedSum(0.0),
	      aegisOutputRealizedSum(0.0),
	      aegisPredictiveErrorSum(0.0),
	      aegisOutputErrorSum(0.0),
	      aegisChannelDisagreementSum(0.0),
	      citadelMatrices(0u),
	      citadelAnchorSum(0.0),
	      citadelHardRegimeMassSum(0.0),
	      citadelSparrowTrustSum(0.0),
	      rampartMatrices(0u),
	      rampartTauSum(0.0),
	      rampartBudgetSum(0.0),
	      rampartCovarianceSum(0.0),
	      rampartSparrowTrustSum(0.0),
	      meritMatrices(0u),
	      meritTauSum(0.0),
	      meritBudgetSum(0.0),
	      meritCovarianceSum(0.0),
	      meritSparrowTrustSum(0.0),
	      meritGeometryTrustSum(0.0),
	      strataMatrices(0u),
	      strataNullModeSum(0.0),
	      strataPredictiveModeSum(0.0),
	      strataOutputModeSum(0.0),
	      strataCoupledModeSum(0.0),
	      strataBudgetSum(0.0),
	      strataNullBenefitSum(0.0),
	      strataPredictiveBenefitSum(0.0),
	      strataOutputBenefitSum(0.0),
	      strataCoupledBenefitSum(0.0),
	      strataSelectedExcessSum(0.0),
	      strataSwitchRateSum(0.0),
	      transformerGapBatches(0u),
	      transformerInputUpdateNormSum(0.0),
	      transformerBlockUpdateNormSum(),
	      transformerFinalNormUpdateNormSum(0.0),
	      transformerHeadUpdateNormSum(0.0),
	      transformerHeadShareSum(0.0),
	      transformerNonHeadShareSum(0.0),
	      transformerApplyMsSum(0.0),
	      transformerMarginSnapshots(0u),
	      transformerTargetMarginSum(0.0),
	      transformerHardNegativeLogitSum(0.0)
	{
	}
};

static bool atlas_runtime_finite(float v)
{
	return (v == v)
	    && (v != std::numeric_limits<float>::infinity())
	    && (v != -std::numeric_limits<float>::infinity());
}

static double atlas_runtime_nonneg(float v)
{
	if (!atlas_runtime_finite(v))
		return 0.0;
	return std::max<double>(0.0, static_cast<double>(v));
}

static double atlas_runtime_value(float v, double fallback)
{
	if (!atlas_runtime_finite(v))
		return fallback;
	return static_cast<double>(v);
}

static void accumulate_atlas_runtime(AtlasRuntimeAccumulator& acc,
                                     const glades::atlas::WeightState& st,
                                     bool sparrowEnabled)
{
	if (!st.initialized)
		return;

	acc.atlasMatrices += 1u;
	if (!sparrowEnabled)
		return;

	acc.sparrowMatrices += 1u;
	acc.sparrowActiveModesSum += static_cast<double>(st.lastSparrowActiveModes);
	if (st.lastSparrowActiveModes >= 2u)
		acc.sparrowMode2Matrices += 1u;
	acc.sparrowEdgeSum += atlas_runtime_nonneg(st.lastSparrowEdge);
	acc.sparrowSecondEdgeSum += atlas_runtime_nonneg(st.lastSparrowSecondEdge);
	acc.sparrowMemoryGainSum += atlas_runtime_nonneg(st.lastSparrowMemoryGain);
	acc.sparrowHorizontalRatioSum += atlas_runtime_value(st.lastSparrowHorizontalRatio, 1.0);
	if (atlas_runtime_finite(st.lastSparrowSigma) && st.lastSparrowSigma > 1e-12f
	    && atlas_runtime_finite(st.lastSparrowSecondSigma) && st.lastSparrowSecondSigma > 0.0f)
	{
		acc.sparrowSecondEdgeRatioSum += static_cast<double>(st.lastSparrowSecondSigma)
		                               / static_cast<double>(st.lastSparrowSigma);
	}
}

} // namespace

// for stopping ml  training instances

/*!
 * @brief NNetwork constructor
 * @details creates an empty nnetwork
 */
glades::NNetwork::NNetwork(int newNetType)
{
	storeRunningFlag(false);
#if GLADES_HAVE_STD_ATOMICS
	runLock.clear(std::memory_order_release);
#else
	runLock = 0;
#endif
	di = NULL;
	skeleton = NULL;
	ownedSkeleton.reset();
	serverInstance = NULL;
	cConnection = NULL;
	loggerOverride = NULL;
	// `trainingConfig` is default-constructed before entering the constructor body.
	lrScheduleMultiplier = 1.0f;
	lrScheduleEpochOffset = 0;
	lastGradNorm = 0.0f;
	lastGradNormScale = 1.0f;
	lastStepLogTime = 0;
	bayesianLRMultiplier_ = 1.0f;
	bayesianLREpochCounter_ = 0;
	// Initialize tensor gate packs (gateCount is fixed by architecture type).
	tensorGru = TensorGatedState(3u);
	tensorLstm = TensorGatedState(4u);
	gpuTransformerWeights = NULL;
	gpuTransformerScratch = NULL;
	gpuDffWeights = NULL;
	gpuDffScratch = NULL;
	gpuRnnWeights = NULL;
	gpuGruWeights = NULL;
	gpuLstmWeights = NULL;
	gpuCnnWeights = NULL;
	gpuCnnScratch = NULL;
	gpuStateReady = false;
	clean();
	netType = newNetType;
	minibatchSize = NNInfo::BATCH_STOCHASTIC;
	// Deterministic default: each network starts from the RNG engine's default seed
	// until explicitly overridden by the caller.
	rngSeed = rngEngine.seed;
	glades::rng::seed_engine(rngEngine, rngSeed);
}

/*!
 * @brief NNetwork destructor
 * @details destroys the NNetwork object
 */
glades::NNetwork::NNetwork(const NNInfo* newNNInfo, int newNetType)
{
	// Constructors must always fully initialize the object. Never early-return.
	storeRunningFlag(false);
#if GLADES_HAVE_STD_ATOMICS
	runLock.clear(std::memory_order_release);
#else
	runLock = 0;
#endif
	di = NULL;
	skeleton = NULL;
	ownedSkeleton.reset();
	serverInstance = NULL;
	cConnection = NULL;
	loggerOverride = NULL;
	// `trainingConfig` is default-constructed before entering the constructor body.
	lrScheduleMultiplier = 1.0f;
	lrScheduleEpochOffset = 0;
	lastGradNorm = 0.0f;
	lastGradNormScale = 1.0f;
	lastStepLogTime = 0;
	bayesianLRMultiplier_ = 1.0f;
	bayesianLREpochCounter_ = 0;
	tensorGru = TensorGatedState(3u);
	tensorLstm = TensorGatedState(4u);
	gpuTransformerWeights = NULL;
	gpuTransformerScratch = NULL;
	gpuDffWeights = NULL;
	gpuDffScratch = NULL;
	gpuRnnWeights = NULL;
	gpuGruWeights = NULL;
	gpuLstmWeights = NULL;
	gpuCnnWeights = NULL;
	gpuCnnScratch = NULL;
	gpuStateReady = false;
	clean();

	// Lifetime safety: clone and own the NNInfo rather than borrowing a raw pointer.
	// Many call sites allocate an NNInfo, pass it into NNetwork, and later delete it.
	// Borrowing would leave `skeleton` dangling.
	if (newNNInfo)
	{
		ownedSkeleton = shmea::GPointer<NNInfo>(new NNInfo(newNNInfo->getName(), newNNInfo->toGTable()));
		skeleton = ownedSkeleton.get();
	}
	netType = newNetType;
	minibatchSize = (skeleton ? skeleton->getBatchSize() : NNInfo::BATCH_STOCHASTIC);
	// Deterministic default: each network starts from the RNG engine's default seed
	// until explicitly overridden by the caller.
	rngSeed = rngEngine.seed;
	glades::rng::seed_engine(rngEngine, rngSeed);
}

glades::NNetwork::~NNetwork()
{
	freeGpuState();
	clean();
	resetGraphs();
}

bool glades::NNetwork::tryAcquireRunLock()
{
#if GLADES_HAVE_STD_ATOMICS
	// Atomic test-and-set.
	// Returns true if we observed it as clear (unlocked) and acquired it.
	return !runLock.test_and_set(std::memory_order_acquire);
#else
#if !(defined(__GNUC__) || defined(__clang__))
#error "NNetwork::tryAcquireRunLock requires either C++11 std::atomic or GCC/Clang atomic builtins"
#endif
	// Atomic compare-and-swap from 0 -> 1.
	return __sync_bool_compare_and_swap(&runLock, 0, 1);
#endif
}

glades::NNetworkStatus glades::TransformerPublicAPI::generate(const glades::NNetwork& net,
                                                              const std::vector<glades::TokenId>& promptTokens,
                                                              const glades::TransformerGenerateConfig& cfg,
                                                              glades::TransformerGenerateResult& out,
                                                              glades::ITransformerGenerateCallbacks* cb)
{
	return TransformerPublicAPI::runtime(net).generate(promptTokens, cfg, out, cb);
}

glades::NNetworkStatus glades::TransformerPublicAPI::generateBatch(const glades::NNetwork& net,
                                                                   const std::vector<glades::TransformerServeRequest>& requests,
                                                                   glades::TransformerServeBatchResult& out,
                                                                   glades::ITransformerServeCallbacks* cb)
{
	return TransformerPublicAPI::runtime(net).generateBatch(requests, out, cb);
}

glades::NNetworkStatus glades::TransformerPublicAPI::forwardLastLogits(const glades::NNetwork& net,
                                                                       const std::vector<glades::TokenId>& tokenIds,
                                                                       std::vector<float>& outLogits)
{
	return TransformerPublicAPI::runtime(net).forwardLastLogits(tokenIds, outLogits);
}

glades::TransformerPublicAPI::Runtime glades::TransformerPublicAPI::runtime(const glades::NNetwork& net)
{
	return Runtime(net);
}

glades::TransformerPublicAPI::ServingRuntime glades::TransformerPublicAPI::serving(const glades::NNetwork& net)
{
	return ServingRuntime(net);
}

glades::NNetworkStatus glades::TransformerPublicAPI::Runtime::generate(const std::vector<glades::TokenId>& promptTokens,
                                                                       const glades::TransformerGenerateConfig& cfg,
                                                                       glades::TransformerGenerateResult& out,
                                                                       glades::ITransformerGenerateCallbacks* cb) const
{
	return net.transformerLmGenerate(promptTokens, cfg, out, cb);
}

glades::NNetworkStatus glades::TransformerPublicAPI::Runtime::generateBatch(const std::vector<glades::TransformerServeRequest>& requests,
                                                                            glades::TransformerServeBatchResult& out,
                                                                            glades::ITransformerServeCallbacks* cb) const
{
	return net.transformerLmServeGenerateBatch(requests, out, cb);
}

glades::NNetworkStatus glades::TransformerPublicAPI::Runtime::forwardLastLogits(const std::vector<glades::TokenId>& tokenIds,
                                                                                std::vector<float>& outLogits) const
{
	return net.transformerLmForwardLastLogits(tokenIds, outLogits);
}

glades::NNetworkStatus glades::TransformerPublicAPI::ServingRuntime::resetBatcher(Batcher& batcher, const BatcherConfig& cfg) const
{
	return net.transformerLmServeBatcherReset(batcher, cfg);
}

glades::NNetworkStatus glades::TransformerPublicAPI::ServingRuntime::submit(Batcher& batcher,
                                                                            const glades::TransformerServeRequest& request,
                                                                            unsigned int& outSlot) const
{
	return net.transformerLmServeBatcherSubmit(batcher, request, outSlot);
}

glades::NNetworkStatus glades::TransformerPublicAPI::ServingRuntime::remove(Batcher& batcher, unsigned int slot) const
{
	return net.transformerLmServeBatcherRemove(batcher, slot);
}

glades::NNetworkStatus glades::TransformerPublicAPI::ServingRuntime::step(Batcher& batcher, glades::ITransformerServeCallbacks* cb) const
{
	return net.transformerLmServeBatcherStep(batcher, cb);
}

glades::NNetworkStatus glades::TransformerPublicAPI::ServingRuntime::cancelSlot(Batcher& batcher, unsigned int slot) const
{
	return net.transformerLmServeBatcherCancelSlot(batcher, slot);
}

void glades::NNetwork::releaseRunLock()
{
#if GLADES_HAVE_STD_ATOMICS
	runLock.clear(std::memory_order_release);
#else
#if !(defined(__GNUC__) || defined(__clang__))
#error "NNetwork::releaseRunLock requires either C++11 std::atomic or GCC/Clang atomic builtins"
#endif
	__sync_lock_release(&runLock);
#endif
}

shmea::GList glades::NNetwork::getWeightsForGui() const
{
	// Preserve historical serialization format used by GUI:
	// - Per-layer, per-node weights (excluding per-neuron bias edges), with ',' between nodes and ';' between layers
	// - Then a 'B' marker followed by per-layer bias summaries.
	//
	// IMPORTANT: For recurrent nets, this matches historical behavior of LayerBuilder::getWeights():
	// it includes only the "input" weights (Wx/W) stored on the hidden/output nodes, and does not
	// include recurrent matrices stored on context nodes (Wh/U).

	shmea::GList weights;

	const bool hasDff = (netType == TYPE_DFF) && tensorDff.initialized;
	const bool hasRnn = (netType == TYPE_RNN) && tensorRnn.initialized;
	const bool hasGru = (netType == TYPE_GRU) && tensorGru.initialized;
	const bool hasLstm = (netType == TYPE_LSTM) && tensorLstm.initialized;
	if (!hasDff && !hasRnn && !hasGru && !hasLstm)
		return weights;

	if (hasDff)
	{
		for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
		{
			const TensorDFFState::Transition& tr = tensorDff.T[t];
			for (unsigned int j = 0; j < tr.out; ++j)
			{
				const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(tr.in);
				for (unsigned int i = 0; i < tr.in; ++i)
					weights.addFloat(tr.W[rowOff + i]);
				weights.addString(',');
			}
			weights.addString(';');
		}

		weights.addString('B');
		for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
		{
			const TensorDFFState::Transition& tr = tensorDff.T[t];
			double sum = 0.0;
			for (size_t i = 0; i < tr.bias.size(); ++i)
				sum += static_cast<double>(tr.bias[i]);
			const float mean = (tr.bias.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(tr.bias.size())));
			weights.addFloat(mean);
		}
		return weights;
	}

	if (hasRnn)
	{
		// Hidden layers: Wxh only
		for (size_t l = 0; l < tensorRnn.H.size(); ++l)
		{
			const TensorRNNState::Hidden& hl = tensorRnn.H[l];
			const unsigned int prevSize = hl.in;
			const unsigned int curSize = hl.h;
			for (unsigned int i = 0; i < curSize; ++i)
			{
				const size_t rowOff = static_cast<size_t>(i) * static_cast<size_t>(prevSize);
				for (unsigned int p = 0; p < prevSize; ++p)
					weights.addFloat(hl.Wxh[rowOff + p]);
				weights.addString(',');
			}
			weights.addString(';');
		}

		// Output: Why
		{
			const unsigned int prevSize = tensorRnn.O.in;
			const unsigned int outSize = tensorRnn.O.out;
			for (unsigned int k = 0; k < outSize; ++k)
			{
				const size_t rowOff = static_cast<size_t>(k) * static_cast<size_t>(prevSize);
				for (unsigned int i = 0; i < prevSize; ++i)
					weights.addFloat(tensorRnn.O.Why[rowOff + i]);
				weights.addString(',');
			}
			weights.addString(';');
		}

		weights.addString('B');
		for (size_t l = 0; l < tensorRnn.H.size(); ++l)
		{
			const TensorRNNState::Hidden& hl = tensorRnn.H[l];
			double sum = 0.0;
			for (size_t i = 0; i < hl.bias.size(); ++i)
				sum += static_cast<double>(hl.bias[i]);
			const float mean = (hl.bias.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(hl.bias.size())));
			weights.addFloat(mean);
		}
		{
			double sum = 0.0;
			for (size_t i = 0; i < tensorRnn.O.bias.size(); ++i)
				sum += static_cast<double>(tensorRnn.O.bias[i]);
			const float mean = (tensorRnn.O.bias.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(tensorRnn.O.bias.size())));
			weights.addFloat(mean);
		}
		return weights;
	}

	const TensorGatedState& tg = hasGru ? tensorGru : tensorLstm;
	// Hidden layers: W only (input-side weights); omit U (recurrent)
	for (size_t l = 0; l < tg.H.size(); ++l)
	{
		const TensorGatedState::Hidden& hl = tg.H[l];
		const unsigned int prevSize = hl.in;
		const unsigned int curSize = hl.h;
		const unsigned int gateCount = tg.gateCount;
		for (unsigned int u = 0; u < curSize; ++u)
		{
			for (unsigned int g = 0; g < gateCount; ++g)
			{
				const size_t wBase =
				    (static_cast<size_t>(g) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize);
				for (unsigned int p = 0; p < prevSize; ++p)
					weights.addFloat(hl.W[wBase + p]);
			}
			weights.addString(',');
		}
		weights.addString(';');
	}

	// Output: Why
	{
		const unsigned int prevSize = tg.O.in;
		const unsigned int outSize = tg.O.out;
		for (unsigned int k = 0; k < outSize; ++k)
		{
			const size_t rowOff = static_cast<size_t>(k) * static_cast<size_t>(prevSize);
			for (unsigned int i = 0; i < prevSize; ++i)
				weights.addFloat(tg.O.Why[rowOff + i]);
			weights.addString(',');
		}
		weights.addString(';');
	}

	weights.addString('B');
	for (size_t l = 0; l < tg.H.size(); ++l)
	{
		const TensorGatedState::Hidden& hl = tg.H[l];
		double sum = 0.0;
		for (size_t i = 0; i < hl.bias.size(); ++i)
			sum += static_cast<double>(hl.bias[i]);
		const float mean = (hl.bias.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(hl.bias.size())));
		weights.addFloat(mean);
	}
	{
		double sum = 0.0;
		for (size_t i = 0; i < tg.O.bias.size(); ++i)
			sum += static_cast<double>(tg.O.bias[i]);
		const float mean = (tg.O.bias.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(tg.O.bias.size())));
		weights.addFloat(mean);
	}
	return weights;
}

int64_t NNetwork::getCurrentTimeMilliseconds() const
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<unsigned long long>(tv.tv_sec) * 1000ULL + tv.tv_usec / 1000ULL;
}

bool glades::NNetwork::getRunning() const
{
	return loadRunningFlag();
}

int glades::NNetwork::getEpochs() const
{
	return epochs;
}

void glades::NNetwork::stop()
{
	storeRunningFlag(false);
}

bool glades::NNetwork::loadRunningFlag() const
{
	return (__atomic_load_n(&running, __ATOMIC_SEQ_CST) != 0);
}

void glades::NNetwork::storeRunningFlag(bool value)
{
	__atomic_store_n(&running, value ? 1 : 0, __ATOMIC_SEQ_CST);
}

uint64_t glades::NNetwork::loadConfiguredSeed() const
{
	return __atomic_load_n(&rngSeed, __ATOMIC_SEQ_CST);
}

void glades::NNetwork::storeConfiguredSeed(uint64_t seed)
{
	__atomic_store_n(&rngSeed, seed, __ATOMIC_SEQ_CST);
}

shmea::GLogger* glades::NNetwork::loadLoggerOverride() const
{
	return __atomic_load_n(&loggerOverride, __ATOMIC_SEQ_CST);
}

void glades::NNetwork::storeLoggerOverride(shmea::GLogger* logger)
{
	__atomic_store_n(&loggerOverride, logger, __ATOMIC_SEQ_CST);
}

void glades::NNetwork::setSeed(uint64_t seed)
{
	storeConfiguredSeed(seed);
	RunLockGuard runGuard(*this);
	if (runGuard.ok())
		glades::rng::seed_engine(rngEngine, seed);
}

glades::NNetworkStatus glades::NNetwork::train(const DataInput* newDataInput)
{
    return train(newDataInput, NULL);
}

glades::NNetworkStatus glades::NNetwork::test(const DataInput* newDataInput)
{
    return test(newDataInput, NULL);
}

// NOTE on ownership:
// `GNet::GServer::send(shmea::ServiceData*)` is an async-style API that takes ownership of the
// heap-allocated ServiceData object. We express that contract explicitly by using a GPointer
// with a no-op deleter, and "disarming" it after handing the raw pointer off to the server.
//
// IMPORTANT: GPointer's deleter must have external linkage to be used as a template argument.
namespace glades {
namespace ownership_detail {
void service_data_noop_deleter(shmea::ServiceData* p) { (void)p; }
} // namespace ownership_detail
} // namespace glades

namespace {
typedef shmea::GPointer<shmea::ServiceData, glades::ownership_detail::service_data_noop_deleter> ServiceDataSendPtr;

class CompositeCallbacks : public glades::ITrainingCallbacks
{
public:
	CompositeCallbacks(glades::ITrainingCallbacks* a, glades::ITrainingCallbacks* b) : cbA(a), cbB(b) {}

	virtual void onRunStart(const glades::NNetwork& net, int runType)
	{
		if (cbA) cbA->onRunStart(net, runType);
		if (cbB) cbB->onRunStart(net, runType);
	}

	virtual bool onEpochEnd(const glades::NNetwork& net, const glades::NNetworkEpochMetrics& m)
	{
		bool stop = false;
		if (cbA) stop = cbA->onEpochEnd(net, m) || stop;
		if (cbB) stop = cbB->onEpochEnd(net, m) || stop;
		return stop;
	}

	virtual void onRunEnd(const glades::NNetwork& net, int runType)
	{
		if (cbA) cbA->onRunEnd(net, runType);
		if (cbB) cbB->onRunEnd(net, runType);
	}

private:
	glades::ITrainingCallbacks* cbA;
	glades::ITrainingCallbacks* cbB;
};

static const char* run_type_name(int runType)
{
	switch (runType)
	{
	case glades::NNetwork::RUN_TRAIN: return "train";
	case glades::NNetwork::RUN_TEST: return "test";
	case glades::NNetwork::RUN_VALIDATE: return "validate";
	default: return "unknown";
	}
}

static const char* status_code_name(glades::NNetworkStatus::Code code)
{
	switch (code)
	{
	case glades::NNetworkStatus::OK: return "OK";
	case glades::NNetworkStatus::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
	case glades::NNetworkStatus::INVALID_STATE: return "INVALID_STATE";
	case glades::NNetworkStatus::EMPTY_DATA: return "EMPTY_DATA";
	case glades::NNetworkStatus::BUILD_FAILED: return "BUILD_FAILED";
	case glades::NNetworkStatus::INTERNAL_ERROR: return "INTERNAL_ERROR";
	default: return "UNKNOWN";
	}
}

static const char* output_type_name(int outType)
{
	switch (outType)
	{
	case glades::GMath::REGRESSION: return "regression";
	case glades::GMath::CLASSIFICATION: return "classification";
	case glades::GMath::KL: return "kl";
	default: return "unknown";
	}
}

class LoggerCallbacks : public glades::ITrainingCallbacks
{
public:
	explicit LoggerCallbacks(shmea::GLogger* l) : logger(l), last(), lastEpochLogTime(0) {}

	virtual void onRunStart(const glades::NNetwork& net, int runType)
	{
		if (!logger)
			return;
		const glades::NNInfo* sk = net.getNNInfo();

		std::ostringstream oss;
		oss << "event=nn_run_start";
		append_logfmt_kv(oss, "run_type", std::string(run_type_name(runType)));
		append_logfmt_kv(oss, "net_type", net.getNetType());
		append_logfmt_kv(oss, "epochs", net.getEpochs());
		append_logfmt_kv(oss, "rng_seed", static_cast<unsigned long long>(net.getSeed()));
		if (sk)
		{
			append_logfmt_kv(oss, "name", std::string(sk->getName().c_str()));
			append_logfmt_kv(oss, "output_type", std::string(output_type_name(sk->getOutputType())));
			append_logfmt_kv(oss, "output_size", static_cast<int>(sk->getOutputLayerSize()));
			append_logfmt_kv(oss, "hidden_layers", sk->numHiddenLayers());
		}
		logger->info("NNetwork", shmea::GString(oss.str().c_str()));
	}

	virtual bool onEpochEnd(const glades::NNetwork& net, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		if (!logger)
			return false;
		if (m.runType != glades::NNetwork::RUN_TRAIN)
			return false;

		// Rate limit: log at most once per second to avoid flooding on fast epochs.
		const int64_t now = net.getCurrentTimeMilliseconds();
		if (now - lastEpochLogTime < 1000)
			return false;
		lastEpochLogTime = now;

		std::ostringstream oss;
		oss << "event=nn_epoch_end";
		append_logfmt_kv(oss, "run_type", std::string(run_type_name(m.runType)));
		append_logfmt_kv(oss, "epoch", m.epoch);
		append_logfmt_kv(oss, "output_type", std::string(output_type_name(m.outputType)));

		// Metrics
		append_logfmt_kv(oss, "loss", m.totalError);
		append_logfmt_kv(oss, "primary_accuracy", m.totalAccuracy);
		append_logfmt_kv(oss, "perplexity", m.perplexity);

		// Regression extras
		append_logfmt_kv(oss, "mae", m.regMAE);
		append_logfmt_kv(oss, "rmse", m.regRMSE);

		// Classification extras
		append_logfmt_kv(oss, "class_accuracy", m.classAccuracy);
		append_logfmt_kv(oss, "mcc", m.classMCC);
		append_logfmt_kv(oss, "precision", m.classPrecision);
		append_logfmt_kv(oss, "recall", m.classRecall);
		append_logfmt_kv(oss, "specificity", m.classSpecificity);
		append_logfmt_kv(oss, "f1", m.classF1);

		// Training metadata
		append_logfmt_kv(oss, "lr", m.learningRate);
		append_logfmt_kv(oss, "lr_mult", m.lrMultiplier);
		append_logfmt_kv(oss, "grad_norm", m.gradNorm);
		append_logfmt_kv(oss, "grad_norm_scale", m.gradNormScale);

		const glades::NNInfo* sk = net.getNNInfo();
		if (sk)
			append_logfmt_kv(oss, "name", std::string(sk->getName().c_str()));

		logger->info("NNetwork", shmea::GString(oss.str().c_str()));
		return false;
	}

	virtual void onRunEnd(const glades::NNetwork& net, int runType)
	{
		if (!logger)
			return;
		const glades::NNInfo* sk = net.getNNInfo();

		std::ostringstream oss;
		oss << "event=nn_run_end";
		append_logfmt_kv(oss, "run_type", std::string(run_type_name(runType)));
		append_logfmt_kv(oss, "epoch", last.epoch);
		append_logfmt_kv(oss, "loss", last.totalError);
		append_logfmt_kv(oss, "primary_accuracy", last.totalAccuracy);
		append_logfmt_kv(oss, "perplexity", last.perplexity);
		if (sk)
		{
			append_logfmt_kv(oss, "name", std::string(sk->getName().c_str()));
			append_logfmt_kv(oss, "output_type", std::string(output_type_name(sk->getOutputType())));
		}
		logger->info("NNetwork", shmea::GString(oss.str().c_str()));
	}

private:
	shmea::GLogger* logger;
	glades::NNetworkEpochMetrics last;
	int64_t lastEpochLogTime;
};

static bool emit_trainer_preflight_failure_log(const glades::NNetwork& net,
                                               int runType,
                                               const glades::NNetworkStatus& st)
{
	glades::NNetwork::TrainerRunDiagnostics diag;
	if (!net.getTrainerRunDiagnostics(diag))
		return false;
	if (!diag.lastFailureDuringPreflight)
		return false;
	if (diag.lastRunType != runType)
		return false;
	if (diag.lastFailureStatus.code != st.code || diag.lastFailureStatus.message != st.message)
		return false;

	shmea::GLogger* logger = net.getLogger();
	if (!logger)
		return false;

	std::ostringstream oss;
	oss << "event=nn_run_preflight_fail";
	append_logfmt_kv(oss, "run_type", std::string(run_type_name(runType)));
	append_logfmt_kv(oss, "net_type", net.getNetType());
	append_logfmt_kv(oss, "stage", diag.lastFailureStage);
	append_logfmt_kv(oss, "run_attempts", static_cast<unsigned long long>(diag.totalRunAttempts));
	append_logfmt_kv(oss, "run_failures", static_cast<unsigned long long>(diag.totalRunFailures));
	append_logfmt_kv(oss, "preflight_failures", static_cast<unsigned long long>(diag.totalPreflightFailures));
	append_logfmt_kv(oss, "post_build_check", diag.lastFailurePostBuildCheck);
	append_logfmt_kv(oss, "data_size", diag.lastDataSize);
	append_logfmt_kv(oss, "feature_count", diag.lastFeatureCount);
	append_logfmt_kv(oss, "output_size", diag.lastOutputSize);
	append_logfmt_kv(oss, "expected_feature_count", diag.lastExpectedFeatureCount);
	append_logfmt_kv(oss, "expected_output_size", diag.lastExpectedOutputSize);
	append_logfmt_kv(oss, "token_lm", diag.lastTokenLM);
	append_logfmt_kv(oss, "token_lm_input", diag.lastTokenLMInput);
	append_logfmt_kv(oss, "sequence_model", diag.lastSequenceModel);
	append_logfmt_kv(oss, "status_code", std::string(status_code_name(st.code)));
	append_logfmt_kv(oss, "status_ok", st.ok());
	if (!st.message.empty())
		append_logfmt_kv(oss, "error", st.message);
	if (!diag.lastDataInputStatus.ok())
	{
		append_logfmt_kv(oss, "data_status_code", std::string(status_code_name(diag.lastDataInputStatus.code)));
		if (!diag.lastDataInputStatus.message.empty())
			append_logfmt_kv(oss, "data_error", diag.lastDataInputStatus.message);
	}

	const glades::NNInfo* sk = net.getNNInfo();
	if (sk)
	{
		append_logfmt_kv(oss, "name", std::string(sk->getName().c_str()));
		append_logfmt_kv(oss, "output_type", std::string(output_type_name(sk->getOutputType())));
		append_logfmt_kv(oss, "hidden_layers", sk->numHiddenLayers());
	}

	const bool hardFailure =
	    (st.code == glades::NNetworkStatus::INTERNAL_ERROR) ||
	    (diag.lastFailureStage == "initialize_tensors");
	if (hardFailure)
		logger->error("NNetwork", shmea::GString(oss.str().c_str()));
	else
		logger->warning("NNetwork", shmea::GString(oss.str().c_str()));
	return true;
}

class GuiCallbacks : public glades::ITrainingCallbacks
{
public:
	GuiCallbacks(GNet::GServer* s, GNet::Connection* c)
	    : server(s), conn(c), lastUpdateTime(0), sentLayerSizes(false)
	{
	}

	virtual void onRunStart(const glades::NNetwork& /*net*/, int /*runType*/)
	{
		if (!server || !conn)
			return;

		shmea::GList argData;
		argData.addString("RESET");

		ServiceDataSendPtr cData(new shmea::ServiceData(conn, "GUI_Callback"));
		cData->set(argData);
		cData->setArgList(argData);
		server->send(cData);
		cData.reset();
	}

	virtual bool onEpochEnd(const glades::NNetwork& net, const glades::NNetworkEpochMetrics& m)
	{
		if (!server || !conn)
			return false;

		// Rate limit: first few epochs always, then ~60fps equivalent.
		const int64_t ms = net.getCurrentTimeMilliseconds();
		const int64_t timeDiff = ms - lastUpdateTime;
		if (!((m.epoch - m.startingEpoch < 10) || (timeDiff > 16)))
			return false;

		// First epoch is random (historical behavior: skip plotting for epoch 0).
		if (m.epoch > 0)
		{
			// Learning curve point: (epoch-1, totalError)
			shmea::GList wData;
			wData.addInt(m.epoch - 1);
			wData.addFloat(m.totalError);

			shmea::GList argData;
			argData.addString("PROGRESSIVE");

			ServiceDataSendPtr cData(new shmea::ServiceData(conn, "GUI_Callback"));
			cData->set(wData);
			cData->setArgList(argData);
			server->send(cData);
			cData.reset();

			// Activations: first message sends layer sizes, subsequent sends activations list
			argData.clear();
			argData.addString("ACTIVATIONS");
			cData = ServiceDataSendPtr(new shmea::ServiceData(conn, "GUI_Callback"));

			if (!sentLayerSizes)
			{
				sentLayerSizes = true;
				shmea::GList layerSizes;
				const glades::NNInfo* sk = net.getNNInfo();
				const glades::DataInput* di = net.getAttachedDataInput();
				if (sk && di)
				{
					layerSizes.addInt(static_cast<int>(di->getFeatureCount()));
					for (int i = 0; i < sk->numHiddenLayers(); ++i)
						layerSizes.addInt(sk->getHiddenLayerSize(i));
					layerSizes.addInt(static_cast<int>(sk->getOutputLayerSize()));
				}
				cData->set(layerSizes);
			}
			else
			{
				cData->set(net.getNodeActivations());
			}
			cData->setArgList(argData);
			server->send(cData);
			cData.reset();

			// Weights
			argData.clear();
			shmea::GList obtainedWeights = net.getWeightsForGui();
			argData.addString("WEIGHTS");
			cData = ServiceDataSendPtr(new shmea::ServiceData(conn, "GUI_Callback"));
			cData->set(obtainedWeights);
			cData->setArgList(argData);
			server->send(cData);
			cData.reset();
		}

		// Rich metrics message: all NNetworkEpochMetrics fields.
		// Format: [epoch, totalAccuracy, totalError, perplexity, outputType,
		//          regMAE, regRMSE, classAccuracy, classPrecision, classRecall,
		//          classSpecificity, classF1, classMCC, learningRate, lrMultiplier,
		//          gradNorm, gradNormScale, runType]
		{
			shmea::GList argData;
			argData.addString("ACC");

			shmea::GList wData;
			wData.addInt(m.epoch);              // [0]
			wData.addFloat(m.totalAccuracy);    // [1]
			wData.addFloat(m.totalError);       // [2]
			wData.addFloat(m.perplexity);       // [3]
			wData.addInt(m.outputType);         // [4]
			wData.addFloat(m.regMAE);           // [5]
			wData.addFloat(m.regRMSE);          // [6]
			wData.addFloat(m.classAccuracy);    // [7]
			wData.addFloat(m.classPrecision);   // [8]
			wData.addFloat(m.classRecall);      // [9]
			wData.addFloat(m.classSpecificity); // [10]
			wData.addFloat(m.classF1);          // [11]
			wData.addFloat(m.classMCC);         // [12]
			wData.addFloat(m.learningRate);     // [13]
			wData.addFloat(m.lrMultiplier);     // [14]
			wData.addFloat(m.gradNorm);         // [15]
			wData.addFloat(m.gradNormScale);    // [16]
			wData.addInt(m.runType);            // [17]

			ServiceDataSendPtr cData(new shmea::ServiceData(conn, "GUI_Callback"));
			cData->set(wData);
			cData->setArgList(argData);
			server->send(cData);
			cData.reset();
		}

		// Confusion matrix (classification/KL only)
		if ((m.outputType == glades::GMath::CLASSIFICATION) || (m.outputType == glades::GMath::KL))
		{
			const glades::CMatrix& cm = net.getConfusionMatrix();
			shmea::GList argData;
			argData.addString("CONF");
			argData.addFloat(cm.getOverallFalseAlarm());
			argData.addFloat(cm.getOverallRecall());

			ServiceDataSendPtr cData(new shmea::ServiceData(conn, "GUI_Callback"));
			cData->set(cm.getMatrix());
			cData->setArgList(argData);
			server->send(cData);
			cData.reset();
		}

		lastUpdateTime = ms;
		return false;
	}

	virtual void onRunEnd(const glades::NNetwork& /*net*/, int /*runType*/)
	{
		if (!server || !conn)
			return;

		shmea::GList argData;
		argData.addString("UPDATE-GRAPHS");

		ServiceDataSendPtr cData(new shmea::ServiceData(conn, "GUI_Callback"));
		cData->set(argData);
		cData->setArgList(argData);
		server->send(cData);
		cData.reset();
	}

private:
	GNet::GServer* server;
	GNet::Connection* conn;
	int64_t lastUpdateTime;
	bool sentLayerSizes;
};
} // namespace

glades::NNetworkStatus glades::NNetwork::train(const DataInput* newDataInput, ITrainingCallbacks* callbacks)
{
    return run(newDataInput, RUN_TRAIN, callbacks);
}

glades::NNetworkStatus glades::NNetwork::test(const DataInput* newDataInput, ITrainingCallbacks* callbacks)
{
    return run(newDataInput, RUN_TEST, callbacks);
}

glades::NNetworkStatus glades::NNetwork::run(const DataInput* newDataInput, int runType, ITrainingCallbacks* callbacks)
{
	// Default callbacks preserve historical behavior: console + optional GUI adapter.
	// NOTE: The *core* run loop is now in Trainer, which is side-effect-free and
	// does not create default callbacks.
	if (callbacks)
	{
		const glades::NNetworkStatus st = glades::Trainer::run(*this, newDataInput, runType, callbacks);
		if (!st.ok())
		{
			if (!emit_trainer_preflight_failure_log(*this, runType, st))
			{
				shmea::GLogger* logger = getLogger();
				if (logger)
					logger->error("NNetwork", st.message.c_str());
			}
		}
		return st;
	}

	LoggerCallbacks logCb(getLogger());
	GuiCallbacks guiCb(serverInstance, cConnection);
	CompositeCallbacks defaultCb(&logCb, (serverInstance && cConnection) ? static_cast<glades::ITrainingCallbacks*>(&guiCb) : NULL);
	{
		const glades::NNetworkStatus st = glades::Trainer::run(*this, newDataInput, runType, static_cast<glades::ITrainingCallbacks*>(&defaultCb));
		if (!st.ok())
		{
			if (!emit_trainer_preflight_failure_log(*this, runType, st))
			{
				shmea::GLogger* logger = getLogger();
				if (logger)
					logger->error("NNetwork", st.message.c_str());
			}
		}
		return st;
	}
}

glades::NNetworkStatus glades::NNetwork::failStatus(glades::NNetworkStatus::Code code, const std::string& message)
{
	lastStatus = glades::NNetworkStatus(code, message);
	storeRunningFlag(false);
	return lastStatus;
}

glades::NNetworkStatus glades::NNetwork::SGDHelper(unsigned int inputRowCounter, int runType)
{
	if (!skeleton)
		return failStatus(NNetworkStatus::INVALID_STATE, "NNetwork::SGDHelper: skeleton is NULL");

	// Net-type-specific SGD implementations live in separate translation units.
	// This preserves behavior while reducing the size/complexity of this file.
	lastStatus = NNetworkStatus(NNetworkStatus::OK, std::string());
	switch (netType)
	{
	case TYPE_DFF:
		SGDHelper_DFF(inputRowCounter, runType);
		return lastStatus;
	case TYPE_RNN:
		// Recurrent helpers only run once per epoch (on row 0).
		if (inputRowCounter != 0)
			return lastStatus;
		SGDHelper_RNN(inputRowCounter, runType);
		return lastStatus;
	case TYPE_GRU:
		if (inputRowCounter != 0)
			return lastStatus;
		SGDHelper_GRU(inputRowCounter, runType);
		return lastStatus;
	case TYPE_LSTM:
		if (inputRowCounter != 0)
			return lastStatus;
		SGDHelper_LSTM(inputRowCounter, runType);
		return lastStatus;
	case TYPE_TRANSFORMER_ENCODER:
	case TYPE_TRANSFORMER_DECODER:
		// Transformer helpers run once per epoch (on row 0) and iterate sequences internally.
		if (inputRowCounter != 0)
			return lastStatus;
		SGDHelper_TRANSFORMER(inputRowCounter, runType);
		return lastStatus;
	case TYPE_CNN:
		SGDHelper_CNN(inputRowCounter, runType);
		return lastStatus;
	default:
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "NNetwork::SGDHelper: unknown netType");
	}
}

/*!
 * @brief get ID
 * @details get the network's unique ID
 * @return the network's ID
 */
int64_t glades::NNetwork::getID() const
{
	return id;
}

shmea::GString glades::NNetwork::getName() const
{
	if (!skeleton)
		return "";

	return skeleton->getName();
}

const NNInfo* glades::NNetwork::getNNInfo() const
{
	if (!skeleton)
		return NULL;

	return skeleton;
}

float glades::NNetwork::getAccuracy() const
{
	if (!skeleton)
		return 0.0f;

	// NOTE:
	// This method is intentionally named "getAccuracy" for historical/UI reasons.
	// It must return the primary accuracy-like score computed by the training loop:
	// - Regression: R^2 (%) in overallTotalAccuracy
	// - Classification/KL: top-1 accuracy (%) in overallTotalAccuracy
	//
	// MCC is available separately via getMCC().
	return overallTotalAccuracy;
}

float glades::NNetwork::getMCC() const
{
	if (!skeleton)
		return 0.0f;

	if ((skeleton->getOutputType() == GMath::CLASSIFICATION) ||
	    (skeleton->getOutputType() == GMath::KL))
		return confusionMatrix.getOverallMCC();

	return 0.0f;
}

const glades::CMatrix& glades::NNetwork::getConfusionMatrix() const
{
	return confusionMatrix;
}

const shmea::GList& glades::NNetwork::getNodeActivations() const
{
	return cNodeActivations;
}

bool glades::NNetwork::getTrainerRunDiagnostics(TrainerRunDiagnostics& out) const
{
	out = trainerRunDiagnostics;
	return true;
}

bool glades::NNetwork::getPersistenceDiagnostics(PersistenceDiagnostics& out) const
{
	out = persistenceDiagnostics;
	return true;
}

bool glades::NNetwork::getAtlasRuntimeDiagnostics(AtlasRuntimeDiagnostics& out) const
{
	AtlasRuntimeAccumulator acc;
	const bool sparrowEnabled =
	    (trainingConfig.optimizer.type == OptimizerConfig::ATLAS) && trainingConfig.atlas.sparrowEnabled;
	const bool helmEnabled =
	    (trainingConfig.optimizer.type == OptimizerConfig::ATLAS) && trainingConfig.atlas.helmEnabled;
	const bool asterEnabled =
	    (trainingConfig.optimizer.type == OptimizerConfig::ATLAS) && trainingConfig.atlas.asterEnabled;

	if (tensorDff.initialized)
	{
		for (size_t i = 0; i < tensorDff.atlasState.size(); ++i)
			accumulate_atlas_runtime(acc, tensorDff.atlasState[i], sparrowEnabled);
		if (helmEnabled && tensorDff.helm.initialized)
		{
			acc.helmMatrices += 1u;
			acc.helmActiveModesSum += static_cast<double>(tensorDff.helm.lastActiveModes);
			if (tensorDff.helm.lastActiveModes >= 2u)
				acc.helmMode2Matrices += 1u;
			acc.helmEdgeSum += atlas_runtime_nonneg(tensorDff.helm.lastEdge);
			acc.helmSecondEdgeSum += atlas_runtime_nonneg(tensorDff.helm.lastSecondEdge);
			acc.helmSecondEdgeRatioSum += atlas_runtime_nonneg(tensorDff.helm.lastSecondEdgeRatio);
			acc.helmSigmaSum += atlas_runtime_nonneg(tensorDff.helm.lastSigma);
			acc.helmPredR2Sum += atlas_runtime_value(tensorDff.helm.lastPredR2, 0.0);
			acc.helmMemoryGainSum += atlas_runtime_nonneg(tensorDff.helm.lastMemoryGain);
			acc.helmPoleSum += tensorDff.helm.pole.empty()
			                   ? 0.0
			                   : atlas_runtime_value(tensorDff.helm.pole[0], 0.0);
		}
		if (asterEnabled && tensorDff.aster.initialized)
		{
			acc.asterMatrices += 1u;
			acc.asterActiveModesSum += static_cast<double>(tensorDff.aster.lastActiveModes);
			if (tensorDff.aster.lastActiveModes >= 2u)
				acc.asterMode2Matrices += 1u;
			acc.asterEdgeSum += atlas_runtime_nonneg(tensorDff.aster.lastEdge);
			acc.asterSecondEdgeSum += atlas_runtime_nonneg(tensorDff.aster.lastSecondEdge);
			acc.asterSecondEdgeRatioSum += atlas_runtime_nonneg(tensorDff.aster.lastSecondEdgeRatio);
			acc.asterSigmaSum += atlas_runtime_nonneg(tensorDff.aster.lastSigma);
			acc.asterPredR2Sum += atlas_runtime_value(tensorDff.aster.lastPredR2, 0.0);
			acc.asterMemoryGainSum += atlas_runtime_nonneg(tensorDff.aster.lastMemoryGain);
			acc.asterPoleSum += tensorDff.aster.pole.empty()
			                    ? 0.0
			                    : atlas_runtime_value(tensorDff.aster.pole[0], 0.0);
			if (tensorDff.aster.timingBoundaryCount > 0ULL)
			{
				const double invCount =
				    1.0 / static_cast<double>(tensorDff.aster.timingBoundaryCount);
				const double nsToMs = 1.0e-6;
				acc.asterBoundaryMsSum += tensorDff.aster.totalBoundaryNs * invCount * nsToMs;
				acc.asterSetupMsSum += tensorDff.aster.totalSetupNs * invCount * nsToMs;
				acc.asterTransportMsSum += tensorDff.aster.totalTransportNs * invCount * nsToMs;
				acc.asterTransferFitMsSum += tensorDff.aster.totalTransferFitNs * invCount * nsToMs;
				acc.asterStateFitMsSum += tensorDff.aster.totalStateFitNs * invCount * nsToMs;
				acc.asterInnovationFitMsSum += tensorDff.aster.totalInnovationFitNs * invCount * nsToMs;
				acc.asterApplyMsSum += tensorDff.aster.totalApplyNs * invCount * nsToMs;
			}
			if (trainingConfig.atlas.aegisEnabled)
			{
				acc.aegisMatrices += 1u;
				acc.aegisLambdaSpatialSum += atlas_runtime_nonneg(tensorDff.aster.aegisLastLambdaSpatial);
				acc.aegisLambdaPredictiveSum += atlas_runtime_nonneg(tensorDff.aster.aegisLastLambdaPredictive);
				acc.aegisLambdaOutputSum += atlas_runtime_nonneg(tensorDff.aster.aegisLastLambdaOutput);
				acc.aegisPredictivePredictedSum += atlas_runtime_nonneg(tensorDff.aster.aegisLastPredictivePredicted);
				acc.aegisPredictiveRealizedSum += atlas_runtime_nonneg(tensorDff.aster.aegisLastPredictiveRealized);
				acc.aegisOutputPredictedSum += atlas_runtime_nonneg(tensorDff.aster.aegisLastOutputPredicted);
				acc.aegisOutputRealizedSum += atlas_runtime_nonneg(tensorDff.aster.aegisLastOutputRealized);
				acc.aegisPredictiveErrorSum += atlas_runtime_nonneg(tensorDff.aster.aegisPredictiveErrorEma);
				acc.aegisOutputErrorSum += atlas_runtime_nonneg(tensorDff.aster.aegisOutputErrorEma);
				acc.aegisChannelDisagreementSum += atlas_runtime_nonneg(tensorDff.aster.aegisLastChannelDisagreement);
			}
			if (trainingConfig.atlas.citadelEnabled)
			{
				acc.citadelMatrices += 1u;
				acc.citadelAnchorSum += atlas_runtime_nonneg(tensorDff.aster.citadelLastAnchor);
				acc.citadelHardRegimeMassSum += atlas_runtime_nonneg(tensorDff.aster.citadelLastHardRegimeMass);
				acc.citadelSparrowTrustSum += atlas_runtime_nonneg(tensorDff.aster.citadelLastSparrowTrust);
			}
			if (trainingConfig.atlas.rampartEnabled)
			{
				acc.rampartMatrices += 1u;
				acc.rampartTauSum += atlas_runtime_nonneg(tensorDff.aster.rampartLastTau);
				acc.rampartBudgetSum += atlas_runtime_nonneg(tensorDff.aster.rampartLastBudget);
				acc.rampartCovarianceSum += atlas_runtime_nonneg(tensorDff.aster.rampartLastCovariance);
				acc.rampartSparrowTrustSum += atlas_runtime_nonneg(tensorDff.aster.rampartLastSparrowTrust);
			}
			if (trainingConfig.atlas.meritEnabled)
			{
				acc.meritMatrices += 1u;
				acc.meritTauSum += atlas_runtime_nonneg(tensorDff.aster.meritLastTau);
				acc.meritBudgetSum += atlas_runtime_nonneg(tensorDff.aster.meritLastBudget);
				acc.meritCovarianceSum += atlas_runtime_nonneg(tensorDff.aster.meritLastCovariance);
				acc.meritSparrowTrustSum += atlas_runtime_nonneg(tensorDff.aster.meritLastSparrowTrust);
				acc.meritGeometryTrustSum += atlas_runtime_nonneg(tensorDff.aster.meritLastGeometryTrust);
			}
			if (trainingConfig.atlas.strataEnabled)
			{
				acc.strataMatrices += 1u;
				acc.strataNullModeSum += atlas_runtime_nonneg(tensorDff.aster.strataLastNullMode);
				acc.strataPredictiveModeSum += atlas_runtime_nonneg(tensorDff.aster.strataLastPredictiveMode);
				acc.strataOutputModeSum += atlas_runtime_nonneg(tensorDff.aster.strataLastOutputMode);
				acc.strataCoupledModeSum += atlas_runtime_nonneg(tensorDff.aster.strataLastCoupledMode);
				acc.strataBudgetSum += atlas_runtime_nonneg(tensorDff.aster.strataLastBudget);
				acc.strataNullBenefitSum += atlas_runtime_value(tensorDff.aster.strataLastNullBenefit, 0.0);
				acc.strataPredictiveBenefitSum += atlas_runtime_value(tensorDff.aster.strataLastPredictiveBenefit, 0.0);
				acc.strataOutputBenefitSum += atlas_runtime_value(tensorDff.aster.strataLastOutputBenefit, 0.0);
				acc.strataCoupledBenefitSum += atlas_runtime_value(tensorDff.aster.strataLastCoupledBenefit, 0.0);
				acc.strataSelectedExcessSum += atlas_runtime_value(tensorDff.aster.strataLastSelectedExcess, 0.0);
				acc.strataSwitchRateSum += atlas_runtime_nonneg(tensorDff.aster.strataLastSwitchRate);
			}
		}
	}
	if (tensorRnn.initialized)
	{
		for (size_t i = 0; i < tensorRnn.H.size(); ++i)
		{
			accumulate_atlas_runtime(acc, tensorRnn.H[i].atlasWxh, sparrowEnabled);
			accumulate_atlas_runtime(acc, tensorRnn.H[i].atlasWhh, sparrowEnabled);
		}
		accumulate_atlas_runtime(acc, tensorRnn.O.atlasWhy, sparrowEnabled);
	}
	if (tensorGru.initialized)
	{
		for (size_t i = 0; i < tensorGru.H.size(); ++i)
		{
			accumulate_atlas_runtime(acc, tensorGru.H[i].atlasW, sparrowEnabled);
			accumulate_atlas_runtime(acc, tensorGru.H[i].atlasU, sparrowEnabled);
		}
		accumulate_atlas_runtime(acc, tensorGru.O.atlasWhy, sparrowEnabled);
	}
	if (tensorLstm.initialized)
	{
		for (size_t i = 0; i < tensorLstm.H.size(); ++i)
		{
			accumulate_atlas_runtime(acc, tensorLstm.H[i].atlasW, sparrowEnabled);
			accumulate_atlas_runtime(acc, tensorLstm.H[i].atlasU, sparrowEnabled);
		}
		accumulate_atlas_runtime(acc, tensorLstm.O.atlasWhy, sparrowEnabled);
	}
	if (tensorCnn.initialized)
	{
		for (size_t i = 0; i < tensorCnn.convLayers.size(); ++i)
			accumulate_atlas_runtime(acc, tensorCnn.convLayers[i].atlasW, sparrowEnabled);
		for (size_t i = 0; i < tensorCnn.fcLayers.size(); ++i)
			accumulate_atlas_runtime(acc, tensorCnn.fcLayers[i].atlasW, sparrowEnabled);
	}
	if (tensorTransformer.initialized)
	{
		accumulate_atlas_runtime(acc, tensorTransformer.atlasTokE, sparrowEnabled);
		accumulate_atlas_runtime(acc, tensorTransformer.atlasWIn, sparrowEnabled);
		for (size_t i = 0; i < tensorTransformer.blocks.size(); ++i)
		{
			const TensorTransformerState::Block& b = tensorTransformer.blocks[i];
			accumulate_atlas_runtime(acc, b.atlasWq, sparrowEnabled);
			accumulate_atlas_runtime(acc, b.atlasWk, sparrowEnabled);
			accumulate_atlas_runtime(acc, b.atlasWv, sparrowEnabled);
			accumulate_atlas_runtime(acc, b.atlasWo, sparrowEnabled);
			accumulate_atlas_runtime(acc, b.atlasW1, sparrowEnabled);
			accumulate_atlas_runtime(acc, b.atlasW2, sparrowEnabled);
		}
		accumulate_atlas_runtime(acc, tensorTransformer.atlasWOut, sparrowEnabled);
		if (tensorTransformer.gapApplyCount > 0ULL)
		{
			const double denom = static_cast<double>(tensorTransformer.gapApplyCount);
			acc.transformerGapBatches += 1u;
			acc.transformerInputUpdateNormSum += tensorTransformer.gapInputUpdateNormSum / denom;
			if (acc.transformerBlockUpdateNormSum.size() < tensorTransformer.gapBlockUpdateNormSums.size())
				acc.transformerBlockUpdateNormSum.resize(tensorTransformer.gapBlockUpdateNormSums.size(), 0.0);
			for (size_t i = 0; i < tensorTransformer.gapBlockUpdateNormSums.size(); ++i)
				acc.transformerBlockUpdateNormSum[i] += tensorTransformer.gapBlockUpdateNormSums[i] / denom;
			acc.transformerFinalNormUpdateNormSum += tensorTransformer.gapFinalNormUpdateNormSum / denom;
			acc.transformerHeadUpdateNormSum += tensorTransformer.gapHeadUpdateNormSum / denom;
			acc.transformerHeadShareSum += tensorTransformer.gapHeadShareSum / denom;
			acc.transformerNonHeadShareSum += tensorTransformer.gapNonHeadShareSum / denom;
			acc.transformerApplyMsSum += (tensorTransformer.gapApplyNsSum / denom) * 1.0e-6;
		}
		if (tensorTransformer.gapMarginSnapshotCount > 0ULL)
		{
			const double denom = static_cast<double>(tensorTransformer.gapMarginSnapshotCount);
			acc.transformerMarginSnapshots += 1u;
			acc.transformerTargetMarginSum += tensorTransformer.gapTargetMarginSum / denom;
			acc.transformerHardNegativeLogitSum += tensorTransformer.gapHardNegativeLogitSum / denom;
		}
		if (helmEnabled && tensorTransformer.helm.initialized)
		{
			acc.helmMatrices += 1u;
			acc.helmActiveModesSum += static_cast<double>(tensorTransformer.helm.lastActiveModes);
			if (tensorTransformer.helm.lastActiveModes >= 2u)
				acc.helmMode2Matrices += 1u;
			acc.helmEdgeSum += atlas_runtime_nonneg(tensorTransformer.helm.lastEdge);
			acc.helmSecondEdgeSum += atlas_runtime_nonneg(tensorTransformer.helm.lastSecondEdge);
			acc.helmSecondEdgeRatioSum += atlas_runtime_nonneg(tensorTransformer.helm.lastSecondEdgeRatio);
			acc.helmSigmaSum += atlas_runtime_nonneg(tensorTransformer.helm.lastSigma);
			acc.helmPredR2Sum += atlas_runtime_value(tensorTransformer.helm.lastPredR2, 0.0);
			acc.helmMemoryGainSum += atlas_runtime_nonneg(tensorTransformer.helm.lastMemoryGain);
			acc.helmPoleSum += tensorTransformer.helm.pole.empty()
			                   ? 0.0
			                   : atlas_runtime_value(tensorTransformer.helm.pole[0], 0.0);
		}
		if (asterEnabled && tensorTransformer.aster.initialized)
		{
			acc.asterMatrices += 1u;
			acc.asterActiveModesSum += static_cast<double>(tensorTransformer.aster.lastActiveModes);
			if (tensorTransformer.aster.lastActiveModes >= 2u)
				acc.asterMode2Matrices += 1u;
			acc.asterEdgeSum += atlas_runtime_nonneg(tensorTransformer.aster.lastEdge);
			acc.asterSecondEdgeSum += atlas_runtime_nonneg(tensorTransformer.aster.lastSecondEdge);
			acc.asterSecondEdgeRatioSum += atlas_runtime_nonneg(tensorTransformer.aster.lastSecondEdgeRatio);
			acc.asterSigmaSum += atlas_runtime_nonneg(tensorTransformer.aster.lastSigma);
			acc.asterPredR2Sum += atlas_runtime_value(tensorTransformer.aster.lastPredR2, 0.0);
			acc.asterMemoryGainSum += atlas_runtime_nonneg(tensorTransformer.aster.lastMemoryGain);
			acc.asterPoleSum += atlas_runtime_value(tensorTransformer.aster.lastPoleSummary, 0.0);
			if (tensorTransformer.aster.timingBoundaryCount > 0ULL)
			{
				const double nsToMs = 1.0 / 1000000.0;
				const double invCount =
				    1.0 / static_cast<double>(tensorTransformer.aster.timingBoundaryCount);
				acc.asterBoundaryMsSum += tensorTransformer.aster.totalBoundaryNs * invCount * nsToMs;
				acc.asterSetupMsSum += tensorTransformer.aster.totalSetupNs * invCount * nsToMs;
				acc.asterTransportMsSum += tensorTransformer.aster.totalTransportNs * invCount * nsToMs;
				acc.asterTransferFitMsSum += tensorTransformer.aster.totalTransferFitNs * invCount * nsToMs;
				acc.asterStateFitMsSum += tensorTransformer.aster.totalStateFitNs * invCount * nsToMs;
				acc.asterInnovationFitMsSum += tensorTransformer.aster.totalInnovationFitNs * invCount * nsToMs;
				acc.asterApplyMsSum += tensorTransformer.aster.totalApplyNs * invCount * nsToMs;
			}
			if (trainingConfig.atlas.aegisEnabled)
			{
				acc.aegisMatrices += 1u;
				acc.aegisLambdaSpatialSum += atlas_runtime_nonneg(tensorTransformer.aster.aegisLastLambdaSpatial);
				acc.aegisLambdaPredictiveSum += atlas_runtime_nonneg(tensorTransformer.aster.aegisLastLambdaPredictive);
				acc.aegisLambdaOutputSum += atlas_runtime_nonneg(tensorTransformer.aster.aegisLastLambdaOutput);
				acc.aegisPredictivePredictedSum += atlas_runtime_nonneg(tensorTransformer.aster.aegisLastPredictivePredicted);
				acc.aegisPredictiveRealizedSum += atlas_runtime_nonneg(tensorTransformer.aster.aegisLastPredictiveRealized);
				acc.aegisOutputPredictedSum += atlas_runtime_nonneg(tensorTransformer.aster.aegisLastOutputPredicted);
				acc.aegisOutputRealizedSum += atlas_runtime_nonneg(tensorTransformer.aster.aegisLastOutputRealized);
				acc.aegisPredictiveErrorSum += atlas_runtime_nonneg(tensorTransformer.aster.aegisPredictiveErrorEma);
				acc.aegisOutputErrorSum += atlas_runtime_nonneg(tensorTransformer.aster.aegisOutputErrorEma);
				acc.aegisChannelDisagreementSum += atlas_runtime_nonneg(tensorTransformer.aster.aegisLastChannelDisagreement);
			}
			if (trainingConfig.atlas.citadelEnabled)
			{
				acc.citadelMatrices += 1u;
				acc.citadelAnchorSum += atlas_runtime_nonneg(tensorTransformer.aster.citadelLastAnchor);
				acc.citadelHardRegimeMassSum += atlas_runtime_nonneg(tensorTransformer.aster.citadelLastHardRegimeMass);
				acc.citadelSparrowTrustSum += atlas_runtime_nonneg(tensorTransformer.aster.citadelLastSparrowTrust);
			}
			if (trainingConfig.atlas.rampartEnabled)
			{
				acc.rampartMatrices += 1u;
				acc.rampartTauSum += atlas_runtime_nonneg(tensorTransformer.aster.rampartLastTau);
				acc.rampartBudgetSum += atlas_runtime_nonneg(tensorTransformer.aster.rampartLastBudget);
				acc.rampartCovarianceSum += atlas_runtime_nonneg(tensorTransformer.aster.rampartLastCovariance);
				acc.rampartSparrowTrustSum += atlas_runtime_nonneg(tensorTransformer.aster.rampartLastSparrowTrust);
			}
			if (trainingConfig.atlas.meritEnabled)
			{
				acc.meritMatrices += 1u;
				acc.meritTauSum += atlas_runtime_nonneg(tensorTransformer.aster.meritLastTau);
				acc.meritBudgetSum += atlas_runtime_nonneg(tensorTransformer.aster.meritLastBudget);
				acc.meritCovarianceSum += atlas_runtime_nonneg(tensorTransformer.aster.meritLastCovariance);
				acc.meritSparrowTrustSum += atlas_runtime_nonneg(tensorTransformer.aster.meritLastSparrowTrust);
				acc.meritGeometryTrustSum += atlas_runtime_nonneg(tensorTransformer.aster.meritLastGeometryTrust);
			}
			if (trainingConfig.atlas.strataEnabled)
			{
				acc.strataMatrices += 1u;
				acc.strataNullModeSum += atlas_runtime_nonneg(tensorTransformer.aster.strataLastNullMode);
				acc.strataPredictiveModeSum += atlas_runtime_nonneg(tensorTransformer.aster.strataLastPredictiveMode);
				acc.strataOutputModeSum += atlas_runtime_nonneg(tensorTransformer.aster.strataLastOutputMode);
				acc.strataCoupledModeSum += atlas_runtime_nonneg(tensorTransformer.aster.strataLastCoupledMode);
				acc.strataBudgetSum += atlas_runtime_nonneg(tensorTransformer.aster.strataLastBudget);
				acc.strataNullBenefitSum += atlas_runtime_value(tensorTransformer.aster.strataLastNullBenefit, 0.0);
				acc.strataPredictiveBenefitSum += atlas_runtime_value(tensorTransformer.aster.strataLastPredictiveBenefit, 0.0);
				acc.strataOutputBenefitSum += atlas_runtime_value(tensorTransformer.aster.strataLastOutputBenefit, 0.0);
				acc.strataCoupledBenefitSum += atlas_runtime_value(tensorTransformer.aster.strataLastCoupledBenefit, 0.0);
				acc.strataSelectedExcessSum += atlas_runtime_value(tensorTransformer.aster.strataLastSelectedExcess, 0.0);
				acc.strataSwitchRateSum += atlas_runtime_nonneg(tensorTransformer.aster.strataLastSwitchRate);
			}
		}
	}

	out = AtlasRuntimeDiagnostics();
	out.atlasMatrices = acc.atlasMatrices;
	out.sparrowMatrices = acc.sparrowMatrices;
	out.sparrowMode2Matrices = acc.sparrowMode2Matrices;
	if (acc.sparrowMatrices > 0u)
	{
		const double denom = static_cast<double>(acc.sparrowMatrices);
		out.sparrowMeanActiveModes = acc.sparrowActiveModesSum / denom;
		out.sparrowMode2Fraction = static_cast<double>(acc.sparrowMode2Matrices) / denom;
		out.sparrowMeanEdge = acc.sparrowEdgeSum / denom;
		out.sparrowMeanSecondEdge = acc.sparrowSecondEdgeSum / denom;
		out.sparrowMeanSecondEdgeRatio = acc.sparrowSecondEdgeRatioSum / denom;
		out.sparrowMeanMemoryGain = acc.sparrowMemoryGainSum / denom;
		out.sparrowMeanHorizontalRatio = acc.sparrowHorizontalRatioSum / denom;
	}
	out.helmMatrices = acc.helmMatrices;
	out.helmMode2Matrices = acc.helmMode2Matrices;
	if (acc.helmMatrices > 0u)
	{
		const double denom = static_cast<double>(acc.helmMatrices);
		out.helmMeanActiveModes = acc.helmActiveModesSum / denom;
		out.helmMode2Fraction = static_cast<double>(acc.helmMode2Matrices) / denom;
		out.helmMeanEdge = acc.helmEdgeSum / denom;
		out.helmMeanSecondEdge = acc.helmSecondEdgeSum / denom;
		out.helmMeanSecondEdgeRatio = acc.helmSecondEdgeRatioSum / denom;
		out.helmMeanSigma = acc.helmSigmaSum / denom;
		out.helmMeanPredR2 = acc.helmPredR2Sum / denom;
		out.helmMeanMemoryGain = acc.helmMemoryGainSum / denom;
		out.helmMeanPole = acc.helmPoleSum / denom;
	}
	out.asterMatrices = acc.asterMatrices;
	out.asterMode2Matrices = acc.asterMode2Matrices;
	if (acc.asterMatrices > 0u)
	{
		const double denom = static_cast<double>(acc.asterMatrices);
		out.asterMeanActiveModes = acc.asterActiveModesSum / denom;
		out.asterMode2Fraction = static_cast<double>(acc.asterMode2Matrices) / denom;
		out.asterMeanEdge = acc.asterEdgeSum / denom;
		out.asterMeanSecondEdge = acc.asterSecondEdgeSum / denom;
		out.asterMeanSecondEdgeRatio = acc.asterSecondEdgeRatioSum / denom;
		out.asterMeanSigma = acc.asterSigmaSum / denom;
		out.asterMeanPredR2 = acc.asterPredR2Sum / denom;
		out.asterMeanMemoryGain = acc.asterMemoryGainSum / denom;
		out.asterMeanPole = acc.asterPoleSum / denom;
		out.asterMeanBoundaryMs = acc.asterBoundaryMsSum / denom;
		out.asterMeanSetupMs = acc.asterSetupMsSum / denom;
		out.asterMeanTransportMs = acc.asterTransportMsSum / denom;
		out.asterMeanTransferFitMs = acc.asterTransferFitMsSum / denom;
		out.asterMeanStateFitMs = acc.asterStateFitMsSum / denom;
		out.asterMeanInnovationFitMs = acc.asterInnovationFitMsSum / denom;
		out.asterMeanApplyMs = acc.asterApplyMsSum / denom;
	}
	out.aegisMatrices = acc.aegisMatrices;
	if (acc.aegisMatrices > 0u)
	{
		const double denom = static_cast<double>(acc.aegisMatrices);
		out.aegisMeanLambdaSpatial = acc.aegisLambdaSpatialSum / denom;
		out.aegisMeanLambdaPredictive = acc.aegisLambdaPredictiveSum / denom;
		out.aegisMeanLambdaOutput = acc.aegisLambdaOutputSum / denom;
		out.aegisMeanPredictivePredicted = acc.aegisPredictivePredictedSum / denom;
		out.aegisMeanPredictiveRealized = acc.aegisPredictiveRealizedSum / denom;
		out.aegisMeanOutputPredicted = acc.aegisOutputPredictedSum / denom;
		out.aegisMeanOutputRealized = acc.aegisOutputRealizedSum / denom;
		out.aegisMeanPredictiveError = acc.aegisPredictiveErrorSum / denom;
		out.aegisMeanOutputError = acc.aegisOutputErrorSum / denom;
		out.aegisMeanChannelDisagreement = acc.aegisChannelDisagreementSum / denom;
	}
	out.citadelMatrices = acc.citadelMatrices;
	if (acc.citadelMatrices > 0u)
	{
		const double denom = static_cast<double>(acc.citadelMatrices);
		out.citadelMeanAnchor = acc.citadelAnchorSum / denom;
		out.citadelMeanHardRegimeMass = acc.citadelHardRegimeMassSum / denom;
		out.citadelMeanSparrowTrust = acc.citadelSparrowTrustSum / denom;
	}
	out.rampartMatrices = acc.rampartMatrices;
	if (acc.rampartMatrices > 0u)
	{
		const double denom = static_cast<double>(acc.rampartMatrices);
		out.rampartMeanTau = acc.rampartTauSum / denom;
		out.rampartMeanBudget = acc.rampartBudgetSum / denom;
		out.rampartMeanCovariance = acc.rampartCovarianceSum / denom;
		out.rampartMeanSparrowTrust = acc.rampartSparrowTrustSum / denom;
	}
	out.meritMatrices = acc.meritMatrices;
	if (acc.meritMatrices > 0u)
	{
		const double denom = static_cast<double>(acc.meritMatrices);
		out.meritMeanTau = acc.meritTauSum / denom;
		out.meritMeanBudget = acc.meritBudgetSum / denom;
		out.meritMeanCovariance = acc.meritCovarianceSum / denom;
		out.meritMeanSparrowTrust = acc.meritSparrowTrustSum / denom;
		out.meritMeanGeometryTrust = acc.meritGeometryTrustSum / denom;
	}
	out.strataMatrices = acc.strataMatrices;
	if (acc.strataMatrices > 0u)
	{
		const double denom = static_cast<double>(acc.strataMatrices);
		out.strataMeanNullMode = acc.strataNullModeSum / denom;
		out.strataMeanPredictiveMode = acc.strataPredictiveModeSum / denom;
		out.strataMeanOutputMode = acc.strataOutputModeSum / denom;
		out.strataMeanCoupledMode = acc.strataCoupledModeSum / denom;
		out.strataMeanBudget = acc.strataBudgetSum / denom;
		out.strataMeanNullBenefit = acc.strataNullBenefitSum / denom;
		out.strataMeanPredictiveBenefit = acc.strataPredictiveBenefitSum / denom;
		out.strataMeanOutputBenefit = acc.strataOutputBenefitSum / denom;
		out.strataMeanCoupledBenefit = acc.strataCoupledBenefitSum / denom;
		out.strataMeanSelectedExcess = acc.strataSelectedExcessSum / denom;
		out.strataMeanSwitchRate = acc.strataSwitchRateSum / denom;
	}
	out.transformerGapBatches = acc.transformerGapBatches;
	if (acc.transformerGapBatches > 0u)
	{
		const double denom = static_cast<double>(acc.transformerGapBatches);
		out.transformerMeanInputUpdateNorm = acc.transformerInputUpdateNormSum / denom;
		out.transformerMeanBlockUpdateNorms.resize(acc.transformerBlockUpdateNormSum.size(), 0.0);
		for (size_t i = 0; i < acc.transformerBlockUpdateNormSum.size(); ++i)
			out.transformerMeanBlockUpdateNorms[i] = acc.transformerBlockUpdateNormSum[i] / denom;
		out.transformerMeanFinalNormUpdateNorm = acc.transformerFinalNormUpdateNormSum / denom;
		out.transformerMeanHeadUpdateNorm = acc.transformerHeadUpdateNormSum / denom;
		out.transformerMeanHeadShare = acc.transformerHeadShareSum / denom;
		out.transformerMeanNonHeadShare = acc.transformerNonHeadShareSum / denom;
		out.transformerMeanApplyMs = acc.transformerApplyMsSum / denom;
	}
	out.transformerMarginSnapshots = acc.transformerMarginSnapshots;
	if (acc.transformerMarginSnapshots > 0u)
	{
		const double denom = static_cast<double>(acc.transformerMarginSnapshots);
		out.transformerMeanTargetMargin = acc.transformerTargetMarginSum / denom;
		out.transformerMeanHardNegativeLogit = acc.transformerHardNegativeLogitSum / denom;
	}
	return true;
}

bool glades::NNetwork::getTransformerGroupedParameterSnapshot(TransformerGroupedParameterSnapshot& out) const
{
	out = TransformerGroupedParameterSnapshot();
	if (!tensorTransformer.initialized)
		return false;

	const TensorTransformerState& tt = tensorTransformer;
	const unsigned int nLayers = tt.nLayers;

	out.inputGroup.reserve(tt.WIn.size() + tt.bIn.size());
	out.inputGroup.insert(out.inputGroup.end(), tt.WIn.begin(), tt.WIn.end());
	out.inputGroup.insert(out.inputGroup.end(), tt.bIn.begin(), tt.bIn.end());

	out.blockGroups.resize(nLayers);
	for (unsigned int li = 0; li < nLayers; ++li)
	{
		const TensorTransformerState::Block& b = tt.blocks[li];
		std::vector<float>& group = out.blockGroups[li];
		group.reserve(b.Wq.size() + b.Wk.size() + b.Wv.size() + b.Wo.size()
		              + b.W1.size() + b.W2.size()
		              + b.bq.size() + b.bk.size() + b.bv.size() + b.bo.size()
		              + b.b1.size() + b.b2.size()
		              + b.ln1Gamma.size() + b.ln1Beta.size()
		              + b.ln2Gamma.size() + b.ln2Beta.size());
		group.insert(group.end(), b.Wq.begin(), b.Wq.end());
		group.insert(group.end(), b.Wk.begin(), b.Wk.end());
		group.insert(group.end(), b.Wv.begin(), b.Wv.end());
		group.insert(group.end(), b.Wo.begin(), b.Wo.end());
		group.insert(group.end(), b.W1.begin(), b.W1.end());
		group.insert(group.end(), b.W2.begin(), b.W2.end());
		group.insert(group.end(), b.bq.begin(), b.bq.end());
		group.insert(group.end(), b.bk.begin(), b.bk.end());
		group.insert(group.end(), b.bv.begin(), b.bv.end());
		group.insert(group.end(), b.bo.begin(), b.bo.end());
		group.insert(group.end(), b.b1.begin(), b.b1.end());
		group.insert(group.end(), b.b2.begin(), b.b2.end());
		group.insert(group.end(), b.ln1Gamma.begin(), b.ln1Gamma.end());
		group.insert(group.end(), b.ln1Beta.begin(), b.ln1Beta.end());
		group.insert(group.end(), b.ln2Gamma.begin(), b.ln2Gamma.end());
		group.insert(group.end(), b.ln2Beta.begin(), b.ln2Beta.end());
	}

	out.finalNormGroup.reserve(tt.lnFinalGamma.size() + tt.lnFinalBeta.size());
	out.finalNormGroup.insert(out.finalNormGroup.end(), tt.lnFinalGamma.begin(), tt.lnFinalGamma.end());
	out.finalNormGroup.insert(out.finalNormGroup.end(), tt.lnFinalBeta.begin(), tt.lnFinalBeta.end());

	if (tt.tokenModel)
	{
		out.headGroup.reserve(tt.tokE.size() + tt.lmBias.size());
		out.headGroup.insert(out.headGroup.end(), tt.tokE.begin(), tt.tokE.end());
		out.headGroup.insert(out.headGroup.end(), tt.lmBias.begin(), tt.lmBias.end());
	}
	else
	{
		out.headGroup.reserve(tt.WOut.size() + tt.bOut.size());
		out.headGroup.insert(out.headGroup.end(), tt.WOut.begin(), tt.WOut.end());
		out.headGroup.insert(out.headGroup.end(), tt.bOut.begin(), tt.bOut.end());
	}

	out.valid = true;
	return true;
}

void glades::NNetwork::resetPersistenceDiagnosticsAttempt(PersistenceDiagnostics& d,
                                                          const char* operation,
                                                          const std::string& name,
                                                          int netType,
                                                          bool isCheckpoint,
                                                          bool tokenizerPresent,
                                                          bool includeOptimizerState,
                                                          uint64_t maxShardBytes)
{
	d.totalPersistenceOps += 1ULL;
	if (isCheckpoint)
		d.totalCheckpointSaveAttempts += 1ULL;
	else
		d.totalModelSaveAttempts += 1ULL;

	d.lastNetType = netType;
	d.lastOperationWasCheckpoint = isCheckpoint;
	d.lastOperationSucceeded = false;
	d.lastOperationRejected = false;
	d.lastRotatedPrevious = false;
	d.lastTokenizerPresent = tokenizerPresent;
	d.lastIncludeOptimizerState = includeOptimizerState;
	d.lastShardCount = 0ULL;
	d.lastTensorCount = 0ULL;
	d.lastWeightsBytes = 0ULL;
	d.lastMaxShardBytes = maxShardBytes;
	d.lastOperation = operation ? std::string(operation) : std::string();
	d.lastName = name;
	d.lastStage = "begin";
	d.lastStatus = glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

void glades::NNetwork::notePersistenceDiagnosticsFailure(PersistenceDiagnostics& d,
                                                         const char* stage,
                                                         bool rejectedInput,
                                                         bool rotatedPrevious,
                                                         const NNetworkStatus& st,
                                                         uint64_t shardCount,
                                                         uint64_t tensorCount,
                                                         uint64_t weightsBytes)
{
	d.totalPersistenceFailures += 1ULL;
	if (d.lastOperationWasCheckpoint)
		d.totalCheckpointSaveFailures += 1ULL;
	else
		d.totalModelSaveFailures += 1ULL;

	d.lastOperationSucceeded = false;
	d.lastOperationRejected = rejectedInput;
	d.lastRotatedPrevious = rotatedPrevious;
	d.lastStage = stage ? std::string(stage) : std::string();
	d.lastStatus = st;
	d.lastShardCount = shardCount;
	d.lastTensorCount = tensorCount;
	d.lastWeightsBytes = weightsBytes;

	if (rejectedInput)
	{
		d.totalRejectedInputs += 1ULL;
		return;
	}

	d.totalPublishFailures += 1ULL;
	if (d.lastOperationWasCheckpoint)
		d.totalCheckpointPublishFailures += 1ULL;
	else
		d.totalModelPublishFailures += 1ULL;

	if (d.lastStage == "rotate_existing")
		d.totalRotateFailures += 1ULL;
	else if (d.lastStage == "publish")
		d.totalPublishRenameFailures += 1ULL;
	else if (d.lastStage == "write_manifest")
		d.totalManifestWriteFailures += 1ULL;
	else if (d.lastStage == "write_nninfo")
		d.totalNninfoWriteFailures += 1ULL;
	else if (d.lastStage == "write_weights")
		d.totalWeightsWriteFailures += 1ULL;
	else if (d.lastStage == "compute_integrity")
		d.totalIntegrityFailures += 1ULL;
	else if (d.lastStage == "collect_tensors")
		d.totalCheckpointTensorCollectionFailures += 1ULL;
	else if (d.lastStage == "open_first_shard" ||
	         d.lastStage == "write_shards" ||
	         d.lastStage == "finalize_shards")
		d.totalCheckpointShardWriteFailures += 1ULL;
}

void glades::NNetwork::notePersistenceDiagnosticsSuccess(PersistenceDiagnostics& d,
                                                         const char* stage,
                                                         bool rotatedPrevious,
                                                         const NNetworkStatus& st,
                                                         uint64_t shardCount,
                                                         uint64_t tensorCount,
                                                         uint64_t weightsBytes)
{
	d.totalPersistenceSuccesses += 1ULL;
	if (d.lastOperationWasCheckpoint)
		d.totalCheckpointSaveSuccesses += 1ULL;
	else
		d.totalModelSaveSuccesses += 1ULL;

	d.lastOperationSucceeded = true;
	d.lastOperationRejected = false;
	d.lastRotatedPrevious = rotatedPrevious;
	d.lastStage = stage ? std::string(stage) : std::string();
	d.lastStatus = st;
	d.lastShardCount = shardCount;
	d.lastTensorCount = tensorCount;
	d.lastWeightsBytes = weightsBytes;
}

void glades::NNetwork::setServer(GNet::GServer* newServer, GNet::Connection* newConnection)
{
	serverInstance = newServer;
	cConnection = newConnection;
}

void glades::NNetwork::setLogger(shmea::GLogger* logger)
{
	storeLoggerOverride(logger);
}

shmea::GLogger* glades::NNetwork::getLogger() const
{
	shmea::GLogger* logger = loadLoggerOverride();
	if (logger)
		return logger;
	if (serverInstance && serverInstance->logger)
		return serverInstance->logger.get();
	return &g_default_network_logger;
}

shmea::GList glades::NNetwork::getResults() const
{
	return results;
}

void glades::NNetwork::clean()
{
	id = -1;
	ownedSkeleton.reset();
	skeleton = NULL;
	di = NULL;
	confusionMatrix.clean();
	serverInstance = NULL;
	cConnection = NULL;
	loggerOverride = NULL;
	results.clear();
	nbRecord.clear();
	epochs = 0;
	overallTotalError = 0.0f;
	overallTotalAccuracy = 0.0f;
	overallClassAccuracy = 0.0f;
	overallClassPrecision = 0.0f;
	overallClassRecall = 0.0f;
	overallClassSpecificity = 0.0f;
	overallClassF1 = 0.0f;
	minibatchSize = NNInfo::BATCH_STOCHASTIC;
	storeRunningFlag(false);
	lastStatus = NNetworkStatus(NNetworkStatus::OK, std::string());
	// Reconstruct configs in place so reset does not depend on assignment over a live object.
	trainingConfig.~TrainingConfig();
	new (&trainingConfig) TrainingConfig();
	transformerMetricsCfg.~TransformerMetricsConfig();
	new (&transformerMetricsCfg) TransformerMetricsConfig();
	// Reset tokenizer/vocab artifacts (deployment metadata).
	tokenizerArtifactsPresent = false;
	tokenizerArtifacts.reset();
	// Schedule bookkeeping resets each run
	lrScheduleMultiplier = 1.0f;
	lrScheduleEpochOffset = 0;
	lastGradNorm = 0.0f;
	lastGradNormScale = 1.0f;
	lastStepLogTime = 0;

	// Reset tensor training state caches (they will be re-initialized on next run).
	tensorDff.reset();
	tensorRnn.reset();
	tensorGru.reset();
	tensorLstm.reset();
	tensorTransformer.reset();
	tensorCnn.reset();
	transformerPosEncCache.reset();
	freeGpuState();

	// Epoch-scoped metric accumulators
	regSSE = 0.0;
	regSAE = 0.0;
	regSumY = 0.0;
	regSumY2 = 0.0;
	regCount = 0ULL;
	clsCorrect = 0ULL;
	clsTotal = 0ULL;
	trainerRunDiagnostics = TrainerRunDiagnostics();
	persistenceDiagnostics = PersistenceDiagnostics();

	// Ensure the run lock is released when resetting the instance state.
#if GLADES_HAVE_STD_ATOMICS
	runLock.clear(std::memory_order_release);
#else
	runLock = 0;
#endif
}

bool glades::NNetwork::ensureTensorParametersInitialized()
{
	if (!skeleton)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: skeleton is NULL");
		return false;
	}
	if (!di)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: DataInput is not attached");
		return false;
	}

	// Determinism: parameter initialization must always use this network's RNG engine.

	const bool tokenModel = trainingConfig.transformer.enableTokenEmbedding;
	const bool tokenIdInput = tokenModel && di->hasTokenIdInput();
	const unsigned int inputSize = tokenIdInput ? 1u : di->getFeatureCount();
	const unsigned int outSize = skeleton->getOutputLayerSize();
	const int H = skeleton->numHiddenLayers();
	if (inputSize == 0u || outSize == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: invalid input/output size (0)");
		return false;
	}

	struct InitGlorot
	{
		static void run(glades::rng::Engine& eng, std::vector<float>& W, unsigned int fanIn, unsigned int fanOut)
		{
			if (fanIn == 0u || fanOut == 0u || W.empty())
				return;
			const double limit = sqrt(6.0 / (static_cast<double>(fanIn) + static_cast<double>(fanOut)));
			for (size_t i = 0; i < W.size(); ++i)
				W[i] = static_cast<float>(glades::rng::uniform_double(eng, -limit, limit));
		}
	};

	// === DFF ===
	if (netType == TYPE_DFF)
	{
		std::vector<unsigned int> wantSizes;
		wantSizes.reserve(static_cast<size_t>(H) + 2u);
		wantSizes.push_back(inputSize);
		for (int l = 0; l < H; ++l)
		{
			const int hs = skeleton->getHiddenLayerSize(static_cast<unsigned int>(l));
			wantSizes.push_back(hs > 0 ? static_cast<unsigned int>(hs) : 0u);
		}
		wantSizes.push_back(outSize);

		for (size_t i = 0; i < wantSizes.size(); ++i)
		{
			if (wantSizes[i] == 0u)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: invalid DFF layer size (0)");
				return false;
			}
		}

		if (tensorDff.initialized && tensorDff.sizes == wantSizes)
			return true;

		tensorDff.reset();
		tensorDff.sizes = wantSizes;

		const unsigned int numTransitions =
		    (wantSizes.size() >= 2u) ? static_cast<unsigned int>(wantSizes.size() - 1u) : 0u;
		tensorDff.T.resize(numTransitions);
		tensorDff.a.resize(wantSizes.size());
		tensorDff.delta.resize(wantSizes.size());

		for (unsigned int li = 0; li < wantSizes.size(); ++li)
		{
			tensorDff.a[li].assign(wantSizes[li], 0.0f);
			if (li == 0u)
				tensorDff.delta[li].clear();
			else
				tensorDff.delta[li].assign(wantSizes[li], 0.0f);
		}

		for (unsigned int t = 0; t < numTransitions; ++t)
		{
			const unsigned int in = wantSizes[t];
			const unsigned int out = wantSizes[t + 1u];
			TensorDFFState::Transition& tr = tensorDff.T[t];
			tr.in = in;
			tr.out = out;
			tr.W.assign(static_cast<size_t>(out) * static_cast<size_t>(in), 0.0f);
			tr.vW.assign(tr.W.size(), 0.0f);
			tr.gW.assign(tr.W.size(), 0.0f);
			tr.bias.assign(out, 0.0f);
			tr.gBias.assign(out, 0.0f);

			InitGlorot::run(rngEngine, tr.W, in, out);
		}

		tensorDff.helm.reset();
		tensorDff.aster.reset();
		if (numTransitions > 0u)
		{
			const TensorDFFState::Transition& outTr = tensorDff.T[numTransitions - 1u];
			const unsigned int hiddenLayerCount = (numTransitions > 0u) ? (numTransitions - 1u) : 0u;
			const unsigned int requestedStackDepth = std::max(1u, trainingConfig.atlas.helmHiddenStackDepth);
			const unsigned int stackDepth = std::min(requestedStackDepth, hiddenLayerCount);
			const unsigned int outputDim = outTr.out;
			unsigned int rawHiddenDim = 0u;
			tensorDff.helm.hiddenLayerActivationIndices.clear();
			tensorDff.helm.hiddenLayerOffsets.clear();
			tensorDff.helm.hiddenLayerSizes.clear();
			if (stackDepth > 0u)
			{
				const unsigned int firstActIndex = numTransitions - stackDepth;
				for (unsigned int d = 0; d < stackDepth; ++d)
				{
					const unsigned int actIndex = firstActIndex + d;
					const unsigned int layerSize = (actIndex < tensorDff.sizes.size()) ? tensorDff.sizes[actIndex] : 0u;
					tensorDff.helm.hiddenLayerActivationIndices.push_back(actIndex);
					tensorDff.helm.hiddenLayerOffsets.push_back(rawHiddenDim);
					tensorDff.helm.hiddenLayerSizes.push_back(layerSize);
					rawHiddenDim += layerSize;
				}
			}
			const unsigned int hiddenDim = stackDepth * outputDim;
			const unsigned int pastDim = hiddenDim + outputDim;
			const unsigned int modeRank =
			    std::max(1u, std::min(trainingConfig.atlas.helmModeRank, std::max(1u, outputDim)));
			tensorDff.helm.initialized = (rawHiddenDim > 0u) && (hiddenDim > 0u) && (outputDim > 0u);
			tensorDff.helm.rawHiddenDim = rawHiddenDim;
			tensorDff.helm.hiddenDim = hiddenDim;
			tensorDff.helm.outputDim = outputDim;
			tensorDff.helm.hiddenStackDepth = stackDepth;
			tensorDff.helm.modeRank = modeRank;
			tensorDff.helm.prevHiddenMean.assign(hiddenDim, 0.0f);
			tensorDff.helm.prevResidualMean.assign(outputDim, 0.0f);
			tensorDff.helm.hiddenVar.assign(hiddenDim, 1.0f);
			tensorDff.helm.residualVar.assign(outputDim, 1.0f);
			tensorDff.helm.crossCov.assign(static_cast<size_t>(outputDim) * static_cast<size_t>(pastDim), 0.0f);
			tensorDff.helm.sigma.assign(modeRank, 0.0f);
			tensorDff.helm.leftMode.assign(static_cast<size_t>(modeRank) * static_cast<size_t>(outputDim), 0.0f);
			tensorDff.helm.rightMode.assign(static_cast<size_t>(modeRank) * static_cast<size_t>(pastDim), 0.0f);
			for (unsigned int m = 0; m < modeRank; ++m)
			{
				if (m < outputDim)
					tensorDff.helm.leftMode[static_cast<size_t>(m) * static_cast<size_t>(outputDim) + m] = 1.0f;
				if (m < pastDim)
					tensorDff.helm.rightMode[static_cast<size_t>(m) * static_cast<size_t>(pastDim) + m] = 1.0f;
			}
			tensorDff.helm.batchHiddenSum.assign(rawHiddenDim, 0.0f);
			tensorDff.helm.batchHiddenSqSum.assign(rawHiddenDim, 0.0f);
			tensorDff.helm.batchResidualSum.assign(outputDim, 0.0f);
			tensorDff.helm.batchResidualSqSum.assign(outputDim, 0.0f);
			tensorDff.helm.latent.assign(modeRank, 0.0f);
			tensorDff.helm.poleNumer.assign(modeRank, 0.0f);
			tensorDff.helm.poleDenom.assign(modeRank, 0.0f);
			tensorDff.helm.pole.assign(modeRank, 0.0f);

			const unsigned int requestedAsterDepth = std::max(1u, trainingConfig.atlas.asterHiddenStackDepth);
			const unsigned int asterStackDepth = std::min(requestedAsterDepth, hiddenLayerCount);
			unsigned int asterRawHiddenDim = 0u;
			tensorDff.aster.hiddenLayerActivationIndices.clear();
			tensorDff.aster.hiddenLayerOffsets.clear();
			tensorDff.aster.hiddenLayerSizes.clear();
			if (asterStackDepth > 0u)
			{
				const unsigned int firstActIndex = numTransitions - asterStackDepth;
				for (unsigned int d = 0; d < asterStackDepth; ++d)
				{
					const unsigned int actIndex = firstActIndex + d;
					const unsigned int layerSize = (actIndex < tensorDff.sizes.size()) ? tensorDff.sizes[actIndex] : 0u;
					tensorDff.aster.hiddenLayerActivationIndices.push_back(actIndex);
					tensorDff.aster.hiddenLayerOffsets.push_back(asterRawHiddenDim);
					tensorDff.aster.hiddenLayerSizes.push_back(layerSize);
					asterRawHiddenDim += layerSize;
				}
			}
			const unsigned int asterControlDim = asterStackDepth * outputDim;
			const unsigned int asterFeatureDim = outputDim + (2u * asterControlDim);
			const unsigned int asterStateRank =
			    std::max(1u, std::min(trainingConfig.atlas.asterStateRank, std::max(1u, outputDim)));
			const unsigned int asterStateFeatureDim = asterStateRank + (2u * asterControlDim);
			tensorDff.aster.initialized = (asterRawHiddenDim > 0u) && (asterControlDim > 0u) && (outputDim > 0u);
			tensorDff.aster.rawHiddenDim = asterRawHiddenDim;
			tensorDff.aster.controlDim = asterControlDim;
			tensorDff.aster.outputDim = outputDim;
			tensorDff.aster.hiddenStackDepth = asterStackDepth;
			tensorDff.aster.stateRank = asterStateRank;
			tensorDff.aster.prevControlMean.assign(asterControlDim, 0.0f);
			tensorDff.aster.prevResidualMean.assign(outputDim, 0.0f);
			tensorDff.aster.controlVar.assign(asterControlDim, 1.0f);
			tensorDff.aster.residualVar.assign(outputDim, 1.0f);
			tensorDff.aster.pastCov.assign(static_cast<size_t>(asterFeatureDim) * static_cast<size_t>(asterFeatureDim), 0.0f);
			tensorDff.aster.crossCov.assign(static_cast<size_t>(outputDim) * static_cast<size_t>(asterFeatureDim), 0.0f);
			tensorDff.aster.theta.assign(static_cast<size_t>(outputDim) * static_cast<size_t>(asterFeatureDim), 0.0f);
			tensorDff.aster.statePastCov.assign(static_cast<size_t>(asterStateFeatureDim) * static_cast<size_t>(asterStateFeatureDim), 0.0f);
			tensorDff.aster.stateCrossCov.assign(static_cast<size_t>(asterStateRank) * static_cast<size_t>(asterStateFeatureDim), 0.0f);
			tensorDff.aster.innovationCov.assign(static_cast<size_t>(outputDim) * static_cast<size_t>(outputDim), 0.0f);
			tensorDff.aster.innovationCross.assign(static_cast<size_t>(asterStateRank) * static_cast<size_t>(outputDim), 0.0f);
			tensorDff.aster.sigma.assign(asterStateRank, 0.0f);
			tensorDff.aster.leftMode.assign(static_cast<size_t>(asterStateRank) * static_cast<size_t>(outputDim), 0.0f);
			tensorDff.aster.rightMode.assign(static_cast<size_t>(asterStateRank) * static_cast<size_t>(asterFeatureDim), 0.0f);
			for (unsigned int m = 0; m < asterStateRank; ++m)
			{
				if (m < outputDim)
					tensorDff.aster.leftMode[static_cast<size_t>(m) * static_cast<size_t>(outputDim) + m] = 1.0f;
				if (m < asterFeatureDim)
					tensorDff.aster.rightMode[static_cast<size_t>(m) * static_cast<size_t>(asterFeatureDim) + m] = 1.0f;
			}
			tensorDff.aster.batchHiddenSum.assign(asterRawHiddenDim, 0.0f);
			tensorDff.aster.batchHiddenSqSum.assign(asterRawHiddenDim, 0.0f);
			tensorDff.aster.batchResidualSum.assign(outputDim, 0.0f);
			tensorDff.aster.batchResidualSqSum.assign(outputDim, 0.0f);
			tensorDff.aster.latent.assign(asterStateRank, 0.0f);
			tensorDff.aster.poleNumer.assign(asterStateRank, 0.0f);
			tensorDff.aster.poleDenom.assign(asterStateRank, 0.0f);
			tensorDff.aster.pole.assign(asterStateRank, 0.0f);
		}

		tensorDff.batchCount = 0u;
		tensorDff.initialized = true;
		return true;
	}

	// === RNN ===
	if (netType == TYPE_RNN)
	{
		if (H <= 0)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: RNN requires >= 1 hidden layer");
			return false;
		}

		std::vector<unsigned int> hiddenSizes;
		hiddenSizes.resize(static_cast<size_t>(H));
		for (int l = 0; l < H; ++l)
		{
			const int hs = skeleton->getHiddenLayerSize(static_cast<unsigned int>(l));
			if (hs <= 0)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: invalid RNN hidden layer size (<=0)");
				return false;
			}
			hiddenSizes[static_cast<size_t>(l)] = static_cast<unsigned int>(hs);
		}

		const bool mismatch = (!tensorRnn.initialized) || (tensorRnn.inputSize != inputSize) || (tensorRnn.outSize != outSize) ||
		                      (tensorRnn.hiddenSizes != hiddenSizes) || (tensorRnn.H.size() != static_cast<size_t>(H));
		if (!mismatch)
			return true;

		tensorRnn.reset();
		tensorRnn.initialized = true;
		tensorRnn.inputSize = inputSize;
		tensorRnn.outSize = outSize;
		tensorRnn.hiddenSizes = hiddenSizes;
		tensorRnn.H.resize(static_cast<size_t>(H));

		for (int l = 0; l < H; ++l)
		{
			const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[static_cast<size_t>(l - 1)];
			const unsigned int curSize = hiddenSizes[static_cast<size_t>(l)];
			TensorRNNState::Hidden& hl = tensorRnn.H[static_cast<size_t>(l)];
			hl.in = prevSize;
			hl.h = curSize;
			hl.Wxh.assign(static_cast<size_t>(curSize) * static_cast<size_t>(prevSize), 0.0f);
			hl.Whh.assign(static_cast<size_t>(curSize) * static_cast<size_t>(curSize), 0.0f);
			hl.vWxh.assign(hl.Wxh.size(), 0.0f);
			hl.vWhh.assign(hl.Whh.size(), 0.0f);
			hl.gWxh.assign(hl.Wxh.size(), 0.0f);
			hl.gWhh.assign(hl.Whh.size(), 0.0f);
			hl.bias.assign(curSize, 0.0f);
			hl.gBias.assign(curSize, 0.0f);

			InitGlorot::run(rngEngine, hl.Wxh, prevSize, curSize);
			InitGlorot::run(rngEngine, hl.Whh, curSize, curSize);
		}

		{
			const unsigned int prevSize = hiddenSizes[static_cast<size_t>(H - 1)];
			tensorRnn.O.in = prevSize;
			tensorRnn.O.out = outSize;
			tensorRnn.O.Why.assign(static_cast<size_t>(outSize) * static_cast<size_t>(prevSize), 0.0f);
			tensorRnn.O.vWhy.assign(tensorRnn.O.Why.size(), 0.0f);
			tensorRnn.O.gWhy.assign(tensorRnn.O.Why.size(), 0.0f);
			tensorRnn.O.bias.assign(outSize, 0.0f);
			tensorRnn.O.gBias.assign(outSize, 0.0f);
			InitGlorot::run(rngEngine, tensorRnn.O.Why, prevSize, outSize);
		}

		return true;
	}

	// === GRU / LSTM (gated) ===
	if (netType == TYPE_GRU || netType == TYPE_LSTM)
	{
		if (H <= 0)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: gated nets require >= 1 hidden layer");
			return false;
		}

		TensorGatedState& tg = (netType == TYPE_GRU) ? tensorGru : tensorLstm;
		const unsigned int gateCount = (netType == TYPE_GRU) ? 3u : 4u;

		std::vector<unsigned int> hiddenSizes;
		hiddenSizes.resize(static_cast<size_t>(H));
		for (int l = 0; l < H; ++l)
		{
			const int hs = skeleton->getHiddenLayerSize(static_cast<unsigned int>(l));
			if (hs <= 0)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: invalid gated hidden layer size (<=0)");
				return false;
			}
			hiddenSizes[static_cast<size_t>(l)] = static_cast<unsigned int>(hs);
		}

		const bool mismatch = (!tg.initialized) || (tg.inputSize != inputSize) || (tg.outSize != outSize) || (tg.gateCount != gateCount) ||
		                      (tg.hiddenSizes != hiddenSizes) || (tg.H.size() != static_cast<size_t>(H));
		if (!mismatch)
			return true;

		tg.reset();
		tg.initialized = true;
		tg.inputSize = inputSize;
		tg.outSize = outSize;
		tg.gateCount = gateCount;
		tg.hiddenSizes = hiddenSizes;
		tg.H.resize(static_cast<size_t>(H));

		for (int l = 0; l < H; ++l)
		{
			const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[static_cast<size_t>(l - 1)];
			const unsigned int curSize = hiddenSizes[static_cast<size_t>(l)];
			TensorGatedState::Hidden& hl = tg.H[static_cast<size_t>(l)];
			hl.in = prevSize;
			hl.h = curSize;
			hl.W.assign(static_cast<size_t>(gateCount) * static_cast<size_t>(curSize) * static_cast<size_t>(prevSize), 0.0f);
			hl.U.assign(static_cast<size_t>(gateCount) * static_cast<size_t>(curSize) * static_cast<size_t>(curSize), 0.0f);
			hl.vW.assign(hl.W.size(), 0.0f);
			hl.vU.assign(hl.U.size(), 0.0f);
			hl.gW.assign(hl.W.size(), 0.0f);
			hl.gU.assign(hl.U.size(), 0.0f);
			hl.bias.assign(static_cast<size_t>(gateCount) * static_cast<size_t>(curSize), 0.0f);
			hl.gBias.assign(hl.bias.size(), 0.0f);

			InitGlorot::run(rngEngine, hl.W, prevSize, curSize);
			InitGlorot::run(rngEngine, hl.U, curSize, curSize);
		}

		{
			const unsigned int prevSize = hiddenSizes[static_cast<size_t>(H - 1)];
			tg.O.in = prevSize;
			tg.O.out = outSize;
			tg.O.Why.assign(static_cast<size_t>(outSize) * static_cast<size_t>(prevSize), 0.0f);
			tg.O.vWhy.assign(tg.O.Why.size(), 0.0f);
			tg.O.gWhy.assign(tg.O.Why.size(), 0.0f);
			tg.O.bias.assign(outSize, 0.0f);
			tg.O.gBias.assign(outSize, 0.0f);
			InitGlorot::run(rngEngine, tg.O.Why, prevSize, outSize);
		}

		return true;
	}

	// === Transformer (encoder/decoder) ===
	if (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER)
	{
		std::vector<unsigned int> hiddenSizes;
		hiddenSizes.reserve(static_cast<size_t>(H > 0 ? H : 0));
		for (int l = 0; l < H; ++l)
		{
			const int hs = skeleton->getHiddenLayerSize(static_cast<unsigned int>(l));
			hiddenSizes.push_back(hs > 0 ? static_cast<unsigned int>(hs) : 0u);
		}

		TransformerModelConfigSnapshot modelCfg;
		lastStatus = buildTransformerModelConfigSnapshot("ensureTensorParametersInitialized",
		                                                 trainingConfig,
		                                                 hiddenSizes,
		                                                 outSize,
		                                                 tokenIdInput,
		                                                 netType == TYPE_TRANSFORMER_DECODER,
		                                                 modelCfg);
		if (!lastStatus.ok())
			return false;

		const unsigned int dModel = modelCfg.dModel;
		const unsigned int nHeads = modelCfg.nHeads;
		const unsigned int nKVHeads = modelCfg.nKVHeads;
		const unsigned int dFF = modelCfg.dFF;
		const unsigned int vocabSize = modelCfg.vocabSize;
		const unsigned int ffnKind = modelCfg.ffnKind;
		const bool tokenModel = modelCfg.tokenModel;
		const unsigned int ff1Width = modelCfg.ff1Width;

		const bool needAdamMoments = atlas_transformer_needs_adam_moments(trainingConfig);
		const bool mismatch = (!tensorTransformer.initialized) || (tensorTransformer.inputSize != inputSize) || (tensorTransformer.outSize != outSize) ||
		                      (tensorTransformer.dModel != dModel) || (tensorTransformer.dFF != dFF) || (tensorTransformer.nHeads != nHeads) ||
		                      (tensorTransformer.nKVHeads != nKVHeads) || (tensorTransformer.ffnKind != ffnKind) ||
		                      (tensorTransformer.tokenModel != tokenModel) ||
		                      (tensorTransformer.vocabSize != vocabSize) ||
		                      (tensorTransformer.padTokenId != modelCfg.padTokenId) ||
		                      (tensorTransformer.tieEmbeddings != modelCfg.tieEmbeddings) ||
		                      (tensorTransformer.nLayers != modelCfg.nLayers) ||
		                      (tensorTransformer.causal != modelCfg.causal);
		if (!mismatch)
		{
			if (needAdamMoments)
			{
				TensorTransformerState& tt = tensorTransformer;
				if (tt.tokenModel)
				{
					ensure_transformer_moment_buffer(tt.vTokE, tt.tokE.size());
					ensure_transformer_moment_buffer(tt.v2TokE, tt.tokE.size());
					ensure_transformer_moment_buffer(tt.mLmBias, tt.lmBias.size());
					ensure_transformer_moment_buffer(tt.v2LmBias, tt.lmBias.size());
				}
				else
				{
					ensure_transformer_moment_buffer(tt.vWIn, tt.WIn.size());
					ensure_transformer_moment_buffer(tt.v2WIn, tt.WIn.size());
					ensure_transformer_moment_buffer(tt.mBIn, tt.bIn.size());
					ensure_transformer_moment_buffer(tt.v2BIn, tt.bIn.size());
					ensure_transformer_moment_buffer(tt.vWOut, tt.WOut.size());
					ensure_transformer_moment_buffer(tt.v2WOut, tt.WOut.size());
					ensure_transformer_moment_buffer(tt.mBOut, tt.bOut.size());
					ensure_transformer_moment_buffer(tt.v2BOut, tt.bOut.size());
				}
				ensure_transformer_moment_buffer(tt.mLnFinalGamma, tt.lnFinalGamma.size());
				ensure_transformer_moment_buffer(tt.v2LnFinalGamma, tt.lnFinalGamma.size());
				ensure_transformer_moment_buffer(tt.mLnFinalBeta, tt.lnFinalBeta.size());
				ensure_transformer_moment_buffer(tt.v2LnFinalBeta, tt.lnFinalBeta.size());
				for (size_t i = 0; i < tt.blocks.size(); ++i)
				{
					TensorTransformerState::Block& block = tt.blocks[i];
					ensure_transformer_moment_buffer(block.mLn1Gamma, block.ln1Gamma.size());
					ensure_transformer_moment_buffer(block.v2Ln1Gamma, block.ln1Gamma.size());
					ensure_transformer_moment_buffer(block.mLn1Beta, block.ln1Beta.size());
					ensure_transformer_moment_buffer(block.v2Ln1Beta, block.ln1Beta.size());
					ensure_transformer_moment_buffer(block.vWq, block.Wq.size());
					ensure_transformer_moment_buffer(block.v2Wq, block.Wq.size());
					ensure_transformer_moment_buffer(block.vWk, block.Wk.size());
					ensure_transformer_moment_buffer(block.v2Wk, block.Wk.size());
					ensure_transformer_moment_buffer(block.vWv, block.Wv.size());
					ensure_transformer_moment_buffer(block.v2Wv, block.Wv.size());
					ensure_transformer_moment_buffer(block.vWo, block.Wo.size());
					ensure_transformer_moment_buffer(block.v2Wo, block.Wo.size());
					ensure_transformer_moment_buffer(block.mBq, block.bq.size());
					ensure_transformer_moment_buffer(block.v2Bq, block.bq.size());
					ensure_transformer_moment_buffer(block.mBk, block.bk.size());
					ensure_transformer_moment_buffer(block.v2Bk, block.bk.size());
					ensure_transformer_moment_buffer(block.mBv, block.bv.size());
					ensure_transformer_moment_buffer(block.v2Bv, block.bv.size());
					ensure_transformer_moment_buffer(block.mBo, block.bo.size());
					ensure_transformer_moment_buffer(block.v2Bo, block.bo.size());
					ensure_transformer_moment_buffer(block.mLn2Gamma, block.ln2Gamma.size());
					ensure_transformer_moment_buffer(block.v2Ln2Gamma, block.ln2Gamma.size());
					ensure_transformer_moment_buffer(block.mLn2Beta, block.ln2Beta.size());
					ensure_transformer_moment_buffer(block.v2Ln2Beta, block.ln2Beta.size());
					ensure_transformer_moment_buffer(block.vW1, block.W1.size());
					ensure_transformer_moment_buffer(block.v2W1, block.W1.size());
					ensure_transformer_moment_buffer(block.vW2, block.W2.size());
					ensure_transformer_moment_buffer(block.v2W2, block.W2.size());
					ensure_transformer_moment_buffer(block.mB1, block.b1.size());
					ensure_transformer_moment_buffer(block.v2B1, block.b1.size());
					ensure_transformer_moment_buffer(block.mB2, block.b2.size());
					ensure_transformer_moment_buffer(block.v2B2, block.b2.size());
				}
			}
			return true;
		}

		tensorTransformer.reset();
		tensorTransformer.initialized = true;
		tensorTransformer.inputSize = inputSize;
		tensorTransformer.outSize = outSize;
		tensorTransformer.dModel = dModel;
		tensorTransformer.dFF = dFF;
		tensorTransformer.nHeads = nHeads;
		tensorTransformer.nKVHeads = nKVHeads;
		tensorTransformer.nLayers = modelCfg.nLayers;
		tensorTransformer.causal = modelCfg.causal;
		tensorTransformer.ffnKind = ffnKind;
		tensorTransformer.tokenModel = tokenModel;
		tensorTransformer.vocabSize = vocabSize;
		tensorTransformer.padTokenId = modelCfg.padTokenId;
		tensorTransformer.tieEmbeddings = modelCfg.tieEmbeddings;
		tensorTransformer.optimizerStep = 0ULL;

		// ATLAS normally uses its own per-matrix state and skips AdamW moments to
		// save memory. Some transformer-side experimental branches reuse Adam-style
		// diagonal moments as part of their backbone even under optimizer=ATLAS.
		// Token LM tensors (embedding + bias)
		if (tokenModel)
		{
			const size_t eN = static_cast<size_t>(vocabSize) * static_cast<size_t>(dModel);
			tensorTransformer.tokE.assign(eN, 0.0f);
			if (needAdamMoments) tensorTransformer.vTokE.assign(eN, 0.0f);
			if (needAdamMoments) tensorTransformer.v2TokE.assign(eN, 0.0f);
			tensorTransformer.gTokE.assign(eN, 0.0f);
			tensorTransformer.lmBias.assign(vocabSize, 0.0f);
			if (needAdamMoments) tensorTransformer.mLmBias.assign(vocabSize, 0.0f);
			if (needAdamMoments) tensorTransformer.v2LmBias.assign(vocabSize, 0.0f);
			tensorTransformer.gLmBias.assign(vocabSize, 0.0f);
		}

		const size_t inW = static_cast<size_t>(dModel) * static_cast<size_t>(inputSize);
		tensorTransformer.WIn.assign(inW, 0.0f);
		if (needAdamMoments) tensorTransformer.vWIn.assign(inW, 0.0f);
		if (needAdamMoments) tensorTransformer.v2WIn.assign(inW, 0.0f);
		tensorTransformer.gWIn.assign(inW, 0.0f);
		tensorTransformer.bIn.assign(dModel, 0.0f);
		if (needAdamMoments) tensorTransformer.mBIn.assign(dModel, 0.0f);
		if (needAdamMoments) tensorTransformer.v2BIn.assign(dModel, 0.0f);
		tensorTransformer.gBIn.assign(dModel, 0.0f);

		const size_t outW = static_cast<size_t>(outSize) * static_cast<size_t>(dModel);
		tensorTransformer.WOut.assign(outW, 0.0f);
		if (needAdamMoments) tensorTransformer.vWOut.assign(outW, 0.0f);
		if (needAdamMoments) tensorTransformer.v2WOut.assign(outW, 0.0f);
		tensorTransformer.gWOut.assign(outW, 0.0f);
		tensorTransformer.bOut.assign(outSize, 0.0f);
		if (needAdamMoments) tensorTransformer.mBOut.assign(outSize, 0.0f);
		if (needAdamMoments) tensorTransformer.v2BOut.assign(outSize, 0.0f);
		tensorTransformer.gBOut.assign(outSize, 0.0f);

		tensorTransformer.blocks.resize(static_cast<size_t>(H));
		const unsigned int dHead = dModel / nHeads;
		const unsigned int dModelKV = nKVHeads * dHead;
		for (int li = 0; li < H; ++li)
		{
			TensorTransformerState::Block& b = tensorTransformer.blocks[static_cast<size_t>(li)];
			b.ln1Gamma.assign(dModel, 1.0f);
			b.ln1Beta.assign(dModel, 0.0f);
			if (needAdamMoments) b.mLn1Gamma.assign(dModel, 0.0f);
			if (needAdamMoments) b.v2Ln1Gamma.assign(dModel, 0.0f);
			if (needAdamMoments) b.mLn1Beta.assign(dModel, 0.0f);
			if (needAdamMoments) b.v2Ln1Beta.assign(dModel, 0.0f);
			b.gLn1Gamma.assign(dModel, 0.0f);
			b.gLn1Beta.assign(dModel, 0.0f);
			b.ln2Gamma.assign(dModel, 1.0f);
			b.ln2Beta.assign(dModel, 0.0f);
			if (needAdamMoments) b.mLn2Gamma.assign(dModel, 0.0f);
			if (needAdamMoments) b.v2Ln2Gamma.assign(dModel, 0.0f);
			if (needAdamMoments) b.mLn2Beta.assign(dModel, 0.0f);
			if (needAdamMoments) b.v2Ln2Beta.assign(dModel, 0.0f);
			b.gLn2Gamma.assign(dModel, 0.0f);
			b.gLn2Beta.assign(dModel, 0.0f);

			const size_t mm = static_cast<size_t>(dModel) * static_cast<size_t>(dModel);
			const size_t mkv = static_cast<size_t>(dModelKV) * static_cast<size_t>(dModel);
			b.Wq.assign(mm, 0.0f);
			b.Wk.assign(mkv, 0.0f);
			b.Wv.assign(mkv, 0.0f);
			b.Wo.assign(mm, 0.0f);
			if (needAdamMoments) { b.vWq.assign(mm, 0.0f); b.vWk.assign(mkv, 0.0f); b.vWv.assign(mkv, 0.0f); b.vWo.assign(mm, 0.0f); }
			if (needAdamMoments) { b.v2Wq.assign(mm, 0.0f); b.v2Wk.assign(mkv, 0.0f); b.v2Wv.assign(mkv, 0.0f); b.v2Wo.assign(mm, 0.0f); }
			b.gWq.assign(mm, 0.0f);
			b.gWk.assign(mkv, 0.0f);
			b.gWv.assign(mkv, 0.0f);
			b.gWo.assign(mm, 0.0f);
			b.bq.assign(dModel, 0.0f);
			b.bk.assign(dModelKV, 0.0f);
			b.bv.assign(dModelKV, 0.0f);
			b.bo.assign(dModel, 0.0f);
			if (needAdamMoments) { b.mBq.assign(dModel, 0.0f); b.mBk.assign(dModelKV, 0.0f); b.mBv.assign(dModelKV, 0.0f); b.mBo.assign(dModel, 0.0f); }
			if (needAdamMoments) { b.v2Bq.assign(dModel, 0.0f); b.v2Bk.assign(dModelKV, 0.0f); b.v2Bv.assign(dModelKV, 0.0f); b.v2Bo.assign(dModel, 0.0f); }
			b.gBq.assign(dModel, 0.0f);
			b.gBk.assign(dModelKV, 0.0f);
			b.gBv.assign(dModelKV, 0.0f);
			b.gBo.assign(dModel, 0.0f);

			const size_t w1 = static_cast<size_t>(ff1Width) * static_cast<size_t>(dModel);
			const size_t w2 = static_cast<size_t>(dModel) * static_cast<size_t>(dFF);
			b.W1.assign(w1, 0.0f); b.W2.assign(w2, 0.0f);
			if (needAdamMoments) { b.vW1.assign(w1, 0.0f); b.vW2.assign(w2, 0.0f); }
			if (needAdamMoments) { b.v2W1.assign(w1, 0.0f); b.v2W2.assign(w2, 0.0f); }
			b.gW1.assign(w1, 0.0f); b.gW2.assign(w2, 0.0f);
			b.b1.assign(ff1Width, 0.0f); b.b2.assign(dModel, 0.0f);
			if (needAdamMoments) { b.mB1.assign(ff1Width, 0.0f); b.mB2.assign(dModel, 0.0f); }
			if (needAdamMoments) { b.v2B1.assign(ff1Width, 0.0f); b.v2B2.assign(dModel, 0.0f); }
			b.gB1.assign(ff1Width, 0.0f); b.gB2.assign(dModel, 0.0f);
		}

		// Final LayerNorm: gamma=1, beta=0, Adam/grad state=0
		tensorTransformer.lnFinalGamma.assign(dModel, 1.0f);
		tensorTransformer.lnFinalBeta.assign(dModel, 0.0f);
		if (needAdamMoments) tensorTransformer.mLnFinalGamma.assign(dModel, 0.0f);
		if (needAdamMoments) tensorTransformer.v2LnFinalGamma.assign(dModel, 0.0f);
		if (needAdamMoments) tensorTransformer.mLnFinalBeta.assign(dModel, 0.0f);
		if (needAdamMoments) tensorTransformer.v2LnFinalBeta.assign(dModel, 0.0f);
		tensorTransformer.gLnFinalGamma.assign(dModel, 0.0f);
		tensorTransformer.gLnFinalBeta.assign(dModel, 0.0f);
		tensorTransformer.adamBeta1Power = 1.0;
		tensorTransformer.adamBeta2Power = 1.0;

		InitGlorot::run(rngEngine, tensorTransformer.WIn, inputSize, dModel);
		InitGlorot::run(rngEngine, tensorTransformer.WOut, dModel, outSize);
		if (tokenModel)
		{
			// Initialize embeddings with N(0, 0.02) (standard LLM practice).
			for (size_t i = 0; i < tensorTransformer.tokE.size(); ++i)
				tensorTransformer.tokE[i] = glades::rng::normal(rngEngine, 0.0f, 0.02f);
		}
		for (int li = 0; li < H; ++li)
		{
			TensorTransformerState::Block& b = tensorTransformer.blocks[static_cast<size_t>(li)];
			InitGlorot::run(rngEngine, b.Wq, dModel, dModel);
			InitGlorot::run(rngEngine, b.Wk, dModel, dModelKV);
			InitGlorot::run(rngEngine, b.Wv, dModel, dModelKV);
			InitGlorot::run(rngEngine, b.Wo, dModel, dModel);
			InitGlorot::run(rngEngine, b.W1, dModel, ff1Width);
			InitGlorot::run(rngEngine, b.W2, dFF, dModel);
		}

		// DDP: broadcast weights from rank 0 so all workers start with identical parameters.
		if (trainingConfig.ddp.enable && glades::ddp::worldSize() > 1)
		{
			// Global tensors
			if (!tensorTransformer.tokE.empty())
				glades::ddp::broadcastFromRoot(&tensorTransformer.tokE[0], tensorTransformer.tokE.size());
			if (!tensorTransformer.lmBias.empty())
				glades::ddp::broadcastFromRoot(&tensorTransformer.lmBias[0], tensorTransformer.lmBias.size());
			glades::ddp::broadcastFromRoot(&tensorTransformer.WIn[0], tensorTransformer.WIn.size());
			glades::ddp::broadcastFromRoot(&tensorTransformer.bIn[0], tensorTransformer.bIn.size());
			glades::ddp::broadcastFromRoot(&tensorTransformer.WOut[0], tensorTransformer.WOut.size());
			glades::ddp::broadcastFromRoot(&tensorTransformer.bOut[0], tensorTransformer.bOut.size());
			if (!tensorTransformer.lnFinalGamma.empty())
				glades::ddp::broadcastFromRoot(&tensorTransformer.lnFinalGamma[0], tensorTransformer.lnFinalGamma.size());
			if (!tensorTransformer.lnFinalBeta.empty())
				glades::ddp::broadcastFromRoot(&tensorTransformer.lnFinalBeta[0], tensorTransformer.lnFinalBeta.size());

			// Per-block tensors
			for (int li = 0; li < H; ++li)
			{
				TensorTransformerState::Block& b = tensorTransformer.blocks[static_cast<size_t>(li)];
				glades::ddp::broadcastFromRoot(&b.Wq[0], b.Wq.size());
				glades::ddp::broadcastFromRoot(&b.Wk[0], b.Wk.size());
				glades::ddp::broadcastFromRoot(&b.Wv[0], b.Wv.size());
				glades::ddp::broadcastFromRoot(&b.Wo[0], b.Wo.size());
				glades::ddp::broadcastFromRoot(&b.bq[0], b.bq.size());
				glades::ddp::broadcastFromRoot(&b.bk[0], b.bk.size());
				glades::ddp::broadcastFromRoot(&b.bv[0], b.bv.size());
				glades::ddp::broadcastFromRoot(&b.bo[0], b.bo.size());
				glades::ddp::broadcastFromRoot(&b.ln1Gamma[0], b.ln1Gamma.size());
				glades::ddp::broadcastFromRoot(&b.ln1Beta[0], b.ln1Beta.size());
				glades::ddp::broadcastFromRoot(&b.ln2Gamma[0], b.ln2Gamma.size());
				glades::ddp::broadcastFromRoot(&b.ln2Beta[0], b.ln2Beta.size());
				glades::ddp::broadcastFromRoot(&b.W1[0], b.W1.size());
				glades::ddp::broadcastFromRoot(&b.W2[0], b.W2.size());
				glades::ddp::broadcastFromRoot(&b.b1[0], b.b1.size());
				glades::ddp::broadcastFromRoot(&b.b2[0], b.b2.size());
			}
		}

		if ((trainingConfig.optimizer.type == OptimizerConfig::ATLAS)
		    && trainingConfig.atlas.helmEnabled
		    && tokenModel
		    && modelCfg.nLayers > 0u
		    && dModel > 0u)
		{
			TensorTransformerState::HelmState& helm = tensorTransformer.helm;
			const unsigned int requestedDepth = std::max(1u, trainingConfig.atlas.helmHiddenStackDepth);
			const unsigned int stackDepth = std::min(requestedDepth, modelCfg.nLayers);
			const unsigned int hiddenDim = stackDepth * dModel;
			const unsigned int outputDim = dModel;
			const unsigned int pastDim = hiddenDim + outputDim;
			const unsigned int modeRank =
			    std::max(1u, std::min(trainingConfig.atlas.helmModeRank, std::max(1u, outputDim)));

			helm.reset();
			helm.initialized = (stackDepth > 0u) && (hiddenDim > 0u) && (outputDim > 0u);
			helm.hiddenDim = hiddenDim;
			helm.outputDim = outputDim;
			helm.hiddenStackDepth = stackDepth;
			helm.modeRank = modeRank;
			helm.trackedBlockIndices.reserve(stackDepth);
			for (unsigned int d = 0u; d < stackDepth; ++d)
				helm.trackedBlockIndices.push_back(modelCfg.nLayers - stackDepth + d);
			helm.prevHiddenMean.assign(hiddenDim, 0.0f);
			helm.prevResidualMean.assign(outputDim, 0.0f);
			helm.hiddenVar.assign(hiddenDim, 1.0f);
			helm.residualVar.assign(outputDim, 1.0f);
			helm.crossCov.assign(static_cast<size_t>(outputDim) * static_cast<size_t>(pastDim), 0.0f);
			helm.sigma.assign(modeRank, 0.0f);
			helm.leftMode.assign(static_cast<size_t>(modeRank) * static_cast<size_t>(outputDim), 0.0f);
			helm.rightMode.assign(static_cast<size_t>(modeRank) * static_cast<size_t>(pastDim), 0.0f);
			for (unsigned int m = 0u; m < modeRank; ++m)
			{
				if (m < outputDim)
					helm.leftMode[static_cast<size_t>(m) * static_cast<size_t>(outputDim) + m] = 1.0f;
				if (m < pastDim)
					helm.rightMode[static_cast<size_t>(m) * static_cast<size_t>(pastDim) + m] = 1.0f;
			}
			helm.batchHiddenSum.assign(hiddenDim, 0.0f);
			helm.batchHiddenSqSum.assign(hiddenDim, 0.0f);
			helm.batchResidualSum.assign(outputDim, 0.0f);
			helm.batchResidualSqSum.assign(outputDim, 0.0f);
			helm.latent.assign(modeRank, 0.0f);
			helm.poleNumer.assign(modeRank, 0.0f);
			helm.poleDenom.assign(modeRank, 0.0f);
			helm.pole.assign(modeRank, 0.0f);
			helm.forwardCorrection.assign(outputDim, 0.0f);
		}
		else
		{
			tensorTransformer.helm.reset();
		}

		if ((trainingConfig.optimizer.type == OptimizerConfig::ATLAS)
		    && trainingConfig.atlas.asterEnabled
		    && tokenModel
		    && modelCfg.nLayers > 0u
		    && dModel > 0u)
		{
			TensorTransformerState::AsterState& aster = tensorTransformer.aster;
			const unsigned int requestedDepth = std::max(1u, trainingConfig.atlas.asterHiddenStackDepth);
			const unsigned int stackDepth = std::min(requestedDepth, modelCfg.nLayers);
			const unsigned int regimeCount = 4u;
			const unsigned int sketchDim = std::max(4u, std::min(16u, vocabSize));
			const unsigned int supportDim = std::max(1u, std::min(5u, vocabSize));
			const unsigned int marginDim = (supportDim > 0u) ? (supportDim - 1u) : 0u;
			const unsigned int tokenCondDim = trainingConfig.atlas.auroraEnabled ? 12u : 8u;
			const unsigned int kappaHeads =
			    trainingConfig.atlas.kappaEnabled ? std::max(1u, std::min(trainingConfig.atlas.kappaHeads, modelCfg.nHeads)) : 0u;
			const unsigned int kappaLagBuckets =
			    trainingConfig.atlas.kappaEnabled ? std::max(1u, std::min(trainingConfig.atlas.kappaLagBuckets, 4u)) : 0u;
			const unsigned int kappaRank =
			    trainingConfig.atlas.kappaEnabled ? std::max(1u, std::min(trainingConfig.atlas.kappaRank, 4u)) : 0u;
			const unsigned int kappaObsDim =
			    (trainingConfig.atlas.kappaEnabled && kappaHeads > 0u && kappaLagBuckets > 0u && kappaRank > 0u)
			        ? (kappaHeads * kappaLagBuckets * kappaRank)
			        : 0u;
			const unsigned int obsDim = sketchDim + supportDim + marginDim + tokenCondDim + kappaObsDim;
			const unsigned int controlStreams = (kappaObsDim > 0u) ? 4u : 3u;
			const unsigned int stateRank =
			    std::max(1u, std::min(trainingConfig.atlas.asterStateRank, obsDim));
			const unsigned int controlDim = stackDepth * controlStreams * obsDim;
			const unsigned int featureDim = (2u * obsDim) + (3u * controlDim);
			const unsigned int stateFeatureDim = (2u * stateRank) + (3u * controlDim);

			aster.reset();
			aster.initialized = (stackDepth > 0u) && (controlDim > 0u) && (obsDim > 0u);
			aster.regimeCount = regimeCount;
			aster.sketchDim = sketchDim;
			aster.supportDim = supportDim;
			aster.tokenCondDim = tokenCondDim;
			aster.kappaObsDim = kappaObsDim;
			aster.kappaHeads = kappaHeads;
			aster.kappaLagBuckets = kappaLagBuckets;
			aster.kappaRank = kappaRank;
			aster.controlDim = controlDim;
			aster.hiddenStackDepth = stackDepth;
			aster.stateRank = stateRank;
			aster.trackedBlockIndices.reserve(stackDepth);
			for (unsigned int d = 0u; d < stackDepth; ++d)
				aster.trackedBlockIndices.push_back(modelCfg.nLayers - stackDepth + d);
			aster.prevPrevControlMean.assign(static_cast<size_t>(regimeCount) * controlDim, 0.0f);
			aster.prevControlMean.assign(static_cast<size_t>(regimeCount) * controlDim, 0.0f);
			aster.prevPrevResidualMean.assign(static_cast<size_t>(regimeCount) * obsDim, 0.0f);
			aster.prevResidualMean.assign(static_cast<size_t>(regimeCount) * obsDim, 0.0f);
			aster.controlVar.assign(static_cast<size_t>(regimeCount) * controlDim, 1.0f);
			aster.residualVar.assign(static_cast<size_t>(regimeCount) * obsDim, 1.0f);
			aster.transportPastCov.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stackDepth) * sketchDim * sketchDim, 0.0f);
			aster.transportCrossCov.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stackDepth) * sketchDim * sketchDim, 0.0f);
			aster.pastCov.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(featureDim) * featureDim, 0.0f);
			aster.crossCov.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(obsDim) * featureDim, 0.0f);
			aster.theta.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(obsDim) * featureDim, 0.0f);
			aster.statePastCov.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stateFeatureDim) * stateFeatureDim, 0.0f);
			aster.stateCrossCov.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stateRank) * stateFeatureDim, 0.0f);
			aster.innovationCov.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(obsDim) * obsDim, 0.0f);
			aster.innovationCross.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stateRank) * obsDim, 0.0f);
			aster.sigma.assign(static_cast<size_t>(regimeCount) * stateRank, 0.0f);
			aster.leftMode.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stateRank) * obsDim, 0.0f);
			aster.rightMode.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stateRank) * featureDim, 0.0f);
			for (unsigned int g = 0u; g < regimeCount; ++g)
			{
				const size_t leftBase = static_cast<size_t>(g) * static_cast<size_t>(stateRank) * obsDim;
				const size_t rightBase = static_cast<size_t>(g) * static_cast<size_t>(stateRank) * featureDim;
				for (unsigned int m = 0u; m < stateRank; ++m)
				{
					if (m < obsDim)
						aster.leftMode[leftBase + static_cast<size_t>(m) * obsDim + m] = 1.0f;
					if (m < featureDim)
						aster.rightMode[rightBase + static_cast<size_t>(m) * featureDim + m] = 1.0f;
				}
			}
			aster.batchFinalHiddenRawSum.assign(static_cast<size_t>(regimeCount) * dModel, 0.0f);
			aster.batchFinalHiddenSketchSum.assign(static_cast<size_t>(regimeCount) * sketchDim, 0.0f);
			aster.batchLayerHiddenRawSum.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stackDepth) * dModel, 0.0f);
			aster.batchLayerHiddenSketchSum.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stackDepth) * sketchDim, 0.0f);
			aster.batchLayerAttnRawSum.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stackDepth) * dModel, 0.0f);
			aster.batchLayerAttnSketchSum.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stackDepth) * sketchDim, 0.0f);
			aster.batchLayerPatternSum.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stackDepth) * tokenCondDim, 0.0f);
			aster.batchLayerKappaSum.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(stackDepth) * kappaObsDim, 0.0f);
			aster.batchResidualSum.assign(static_cast<size_t>(regimeCount) * sketchDim, 0.0f);
			aster.batchSupportLogitSum.assign(static_cast<size_t>(regimeCount) * supportDim, 0.0f);
			aster.batchSupportResidualSum.assign(static_cast<size_t>(regimeCount) * supportDim, 0.0f);
			aster.batchSupportCount.assign(static_cast<size_t>(regimeCount) * supportDim, 0.0f);
			aster.batchSupportHiddenRawSum.assign(static_cast<size_t>(regimeCount) * static_cast<size_t>(supportDim) * dModel, 0.0f);
			aster.batchTargetMarginSum.assign(regimeCount, 0.0f);
			aster.batchHardNegativeLogitSum.assign(regimeCount, 0.0f);
			aster.batchBaselineWorseSum.assign(regimeCount, 0.0f);
			aster.batchRegimeTokenCount.assign(regimeCount, 0.0f);
			aster.targetMarginEma.assign(regimeCount, 0.75f);
			aster.hardNegativeLogitEma.assign(regimeCount, 0.0f);
			aster.hardMarginShortfallEma.assign(regimeCount, 0.0f);
			aster.prevPrevLatent.assign(static_cast<size_t>(regimeCount) * stateRank, 0.0f);
			aster.latent.assign(static_cast<size_t>(regimeCount) * stateRank, 0.0f);
			aster.poleNumer.assign(static_cast<size_t>(regimeCount) * stateRank, 0.0f);
			aster.poleDenom.assign(static_cast<size_t>(regimeCount) * stateRank, 0.0f);
			aster.pole.assign(static_cast<size_t>(regimeCount) * stateRank, 0.0f);
		}
		else
		{
			tensorTransformer.aster.reset();
		}

		return true;
	}

	// === CNN ===
	if (netType == TYPE_CNN)
	{
		if (tensorCnn.initialized)
			return true;

		const CNNConfig& cfg = trainingConfig.cnn;
		if (cfg.inputH == 0u || cfg.inputW == 0u || cfg.inputC == 0u)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: CNN requires inputH/inputW/inputC > 0 in trainingConfig.cnn");
			return false;
		}
		if (static_cast<size_t>(cfg.inputH) * cfg.inputW * cfg.inputC != static_cast<size_t>(inputSize))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: CNN inputH*inputW*inputC != featureCount");
			return false;
		}
		if (cfg.convLayers.empty())
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: CNN requires >= 1 conv layer");
			return false;
		}

		tensorCnn.reset();
		tensorCnn.inputH = cfg.inputH;
		tensorCnn.inputW = cfg.inputW;
		tensorCnn.inputC = cfg.inputC;

		// Kaiming (He) initialization for conv layers.
		struct InitKaiming
		{
			static void run(glades::rng::Engine& eng, std::vector<float>& W, unsigned int fanIn)
			{
				if (fanIn == 0u || W.empty())
					return;
				const double stddev = sqrt(2.0 / static_cast<double>(fanIn));
				for (size_t i = 0; i < W.size(); ++i)
				{
					// Box-Muller for normal distribution.
					const double u1 = glades::rng::uniform_double(eng, 1e-7, 1.0);
					const double u2 = glades::rng::uniform_double(eng, 0.0, 6.283185307179586);
					const double z = sqrt(-2.0 * log(u1)) * cos(u2);
					W[i] = static_cast<float>(z * stddev);
				}
			}
		};

		// Build spatial info and conv layers.
		unsigned int curH = cfg.inputH;
		unsigned int curW = cfg.inputW;
		unsigned int curC = cfg.inputC;

		tensorCnn.spatialInfo.resize(cfg.convLayers.size());
		tensorCnn.convLayers.resize(cfg.convLayers.size());

		for (size_t l = 0; l < cfg.convLayers.size(); ++l)
		{
			const CNNConfig::ConvLayerSpec& spec = cfg.convLayers[l];
			TensorCNNState::ConvSpatialInfo& sp = tensorCnn.spatialInfo[l];
			TensorCNNState::ConvLayer& cl = tensorCnn.convLayers[l];

			sp.inH = curH; sp.inW = curW; sp.inC = curC;
			sp.kH = spec.kernelH; sp.kW = spec.kernelW;
			sp.strideH = spec.strideH; sp.strideW = spec.strideW;
			sp.padH = spec.padH; sp.padW = spec.padW;
			sp.useBatchNorm = spec.useBatchNorm;
			sp.useMaxPool = spec.useMaxPool;
			sp.poolH = spec.poolH; sp.poolW = spec.poolW;
			sp.poolStrideH = spec.poolStrideH; sp.poolStrideW = spec.poolStrideW;

			sp.outH = (curH + 2u * spec.padH - spec.kernelH) / spec.strideH + 1u;
			sp.outW = (curW + 2u * spec.padW - spec.kernelW) / spec.strideW + 1u;
			sp.outC = spec.outChannels;

			if (sp.outH == 0u || sp.outW == 0u)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: CNN conv layer produces 0-dim output");
				tensorCnn.reset();
				return false;
			}

			sp.im2colRows = sp.outH * sp.outW;
			sp.im2colCols = curC * spec.kernelH * spec.kernelW;

			if (spec.useMaxPool)
			{
				sp.poolOutH = (sp.outH - spec.poolH) / spec.poolStrideH + 1u;
				sp.poolOutW = (sp.outW - spec.poolW) / spec.poolStrideW + 1u;
				if (sp.poolOutH == 0u || sp.poolOutW == 0u)
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: CNN pool produces 0-dim output");
					tensorCnn.reset();
					return false;
				}
			}
			else
			{
				sp.poolOutH = sp.outH;
				sp.poolOutW = sp.outW;
			}

			// Allocate conv weights.
			cl.outC = spec.outChannels;
			cl.inC = curC;
			cl.kH = spec.kernelH;
			cl.kW = spec.kernelW;

			const size_t K = static_cast<size_t>(curC) * spec.kernelH * spec.kernelW;
			const size_t wSize = static_cast<size_t>(spec.outChannels) * K;
			cl.W.assign(wSize, 0.0f);
			cl.bias.assign(spec.outChannels, 0.0f);
			cl.gW.assign(wSize, 0.0f);
			cl.gBias.assign(spec.outChannels, 0.0f);
			cl.vW.assign(wSize, 0.0f);
			cl.v2W.assign(wSize, 0.0f);
			cl.vBias.assign(spec.outChannels, 0.0f);
			cl.v2Bias.assign(spec.outChannels, 0.0f);

			// Kaiming init.
			InitKaiming::run(rngEngine, cl.W, static_cast<unsigned int>(K));

			// BatchNorm parameters.
			if (spec.useBatchNorm)
			{
				cl.bnGamma.assign(spec.outChannels, 1.0f);
				cl.bnBeta.assign(spec.outChannels, 0.0f);
				cl.bnRunMean.assign(spec.outChannels, 0.0f);
				cl.bnRunVar.assign(spec.outChannels, 1.0f);
				cl.gBnGamma.assign(spec.outChannels, 0.0f);
				cl.gBnBeta.assign(spec.outChannels, 0.0f);
				cl.vBnGamma.assign(spec.outChannels, 0.0f);
				cl.v2BnGamma.assign(spec.outChannels, 0.0f);
				cl.vBnBeta.assign(spec.outChannels, 0.0f);
				cl.v2BnBeta.assign(spec.outChannels, 0.0f);
			}

			// Advance spatial dims for next layer.
			curH = sp.poolOutH;
			curW = sp.poolOutW;
			curC = spec.outChannels;
		}

		// Flattened size.
		tensorCnn.flattenedSize = curC * curH * curW;
		if (tensorCnn.flattenedSize == 0u)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: CNN flattened size is 0");
			tensorCnn.reset();
			return false;
		}

		// FC layers: NNInfo hidden layers define FC layers; output layer is the final.
		std::vector<unsigned int> fcSizes;
		fcSizes.push_back(tensorCnn.flattenedSize);
		for (int l = 0; l < H; ++l)
		{
			const int hs = skeleton->getHiddenLayerSize(static_cast<unsigned int>(l));
			if (hs <= 0)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitialized: CNN FC hidden layer size <= 0");
				tensorCnn.reset();
				return false;
			}
			fcSizes.push_back(static_cast<unsigned int>(hs));
		}
		fcSizes.push_back(outSize);

		const unsigned int numFC = static_cast<unsigned int>(fcSizes.size() - 1u);
		tensorCnn.fcLayers.resize(numFC);
		for (unsigned int t = 0; t < numFC; ++t)
		{
			TensorCNNState::FCTransition& fc = tensorCnn.fcLayers[t];
			fc.in = fcSizes[t];
			fc.out = fcSizes[t + 1u];
			const size_t wSize = static_cast<size_t>(fc.out) * fc.in;
			fc.W.assign(wSize, 0.0f);
			fc.bias.assign(fc.out, 0.0f);
			fc.gW.assign(wSize, 0.0f);
			fc.gBias.assign(fc.out, 0.0f);
			fc.vW.assign(wSize, 0.0f);
			fc.v2W.assign(wSize, 0.0f);
			fc.vBias.assign(fc.out, 0.0f);
			fc.v2Bias.assign(fc.out, 0.0f);

			InitGlorot::run(rngEngine, fc.W, fc.in, fc.out);
		}

		tensorCnn.batchCount = 0u;
		tensorCnn.optimizerStep = 0ULL;
		tensorCnn.initialized = true;
		return true;
	}

	lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "ensureTensorParametersInitialized: unknown netType");
	return false;
}

namespace {
static const char* kTensorWeightsMagicBin = "GLADES_TENSOR_WEIGHTS_BIN";
static const unsigned int kTensorWeightsMagicBinFixedBytes = 32u; // fixed-size header field
// Tensor weights file format version.
//
// NOTE:
// We previously had multiple historical versions; those have been removed.
// The current format (with modern transformer fields + token LM tensors) is now the canonical v1.
static const unsigned int kTensorWeightsVersionBin = 1u;

static void write_u32_le(std::ostream& out, unsigned int v)
{
	unsigned char b[4];
	b[0] = static_cast<unsigned char>((v >> 0) & 0xFFu);
	b[1] = static_cast<unsigned char>((v >> 8) & 0xFFu);
	b[2] = static_cast<unsigned char>((v >> 16) & 0xFFu);
	b[3] = static_cast<unsigned char>((v >> 24) & 0xFFu);
	out.write(reinterpret_cast<const char*>(b), 4);
}

static void write_u64_le(std::ostream& out, unsigned long long v)
{
	unsigned char b[8];
	b[0] = static_cast<unsigned char>((v >> 0) & 0xFFull);
	b[1] = static_cast<unsigned char>((v >> 8) & 0xFFull);
	b[2] = static_cast<unsigned char>((v >> 16) & 0xFFull);
	b[3] = static_cast<unsigned char>((v >> 24) & 0xFFull);
	b[4] = static_cast<unsigned char>((v >> 32) & 0xFFull);
	b[5] = static_cast<unsigned char>((v >> 40) & 0xFFull);
	b[6] = static_cast<unsigned char>((v >> 48) & 0xFFull);
	b[7] = static_cast<unsigned char>((v >> 56) & 0xFFull);
	out.write(reinterpret_cast<const char*>(b), 8);
}

static bool read_u32_le(std::istream& in, unsigned int& outV)
{
	unsigned char b[4];
	in.read(reinterpret_cast<char*>(b), 4);
	if (!in)
		return false;
	outV = (static_cast<unsigned int>(b[0]) << 0) |
	       (static_cast<unsigned int>(b[1]) << 8) |
	       (static_cast<unsigned int>(b[2]) << 16) |
	       (static_cast<unsigned int>(b[3]) << 24);
	return true;
}

static bool read_u64_le(std::istream& in, unsigned long long& outV)
{
	unsigned char b[8];
	in.read(reinterpret_cast<char*>(b), 8);
	if (!in)
		return false;
	outV =
	    (static_cast<unsigned long long>(b[0]) << 0) |
	    (static_cast<unsigned long long>(b[1]) << 8) |
	    (static_cast<unsigned long long>(b[2]) << 16) |
	    (static_cast<unsigned long long>(b[3]) << 24) |
	    (static_cast<unsigned long long>(b[4]) << 32) |
	    (static_cast<unsigned long long>(b[5]) << 40) |
	    (static_cast<unsigned long long>(b[6]) << 48) |
	    (static_cast<unsigned long long>(b[7]) << 56);
	return true;
}

static void write_f32_le(std::ostream& out, float f)
{
	unsigned int bits = 0u;
	std::memcpy(&bits, &f, sizeof(float));
	write_u32_le(out, bits);
}

static bool read_f32_le(std::istream& in, float& outF)
{
	unsigned int bits = 0u;
	if (!read_u32_le(in, bits))
		return false;
	std::memcpy(&outF, &bits, sizeof(float));
	return true;
}

static bool write_vec_f32(std::ostream& out, const std::vector<float>& v)
{
	write_u64_le(out, static_cast<unsigned long long>(v.size()));
	for (size_t i = 0; i < v.size(); ++i)
		write_f32_le(out, v[i]);
	return static_cast<bool>(out);
}

static bool read_vec_f32(std::istream& in, std::vector<float>& v)
{
	unsigned long long n = 0ull;
	if (!read_u64_le(in, n))
		return false;
	// Hard sanity cap: prevent pathological allocations on corrupted files.
	if (n > (1ull << 31))
		return false;
	v.assign(static_cast<size_t>(n), 0.0f);
	for (size_t i = 0; i < v.size(); ++i)
	{
		if (!read_f32_le(in, v[i]))
			return false;
	}
	return true;
}

static bool mul_size_checked(size_t a, size_t b, size_t& out)
{
	if (a == 0u || b == 0u)
	{
		out = 0u;
		return true;
	}
	if (a > (std::numeric_limits<size_t>::max() / b))
		return false;
	out = a * b;
	return true;
}

// Read a vector<float> where the file specifies a length, but we require an exact expected length.
// This prevents corrupted/malicious files from forcing massive allocations or shape-mismatched tensors.
static bool read_vec_f32_exact(std::istream& in, std::vector<float>& v, size_t expectedCount)
{
	unsigned long long n = 0ull;
	if (!read_u64_le(in, n))
		return false;
	if (static_cast<unsigned long long>(expectedCount) != n)
		return false;
	// expectedCount is derived from model dimensions and is trusted (bounded by architecture).
	v.assign(expectedCount, 0.0f);
	for (size_t i = 0; i < expectedCount; ++i)
	{
		if (!read_f32_le(in, v[i]))
			return false;
	}
	return true;
}

struct TransformerTensorWeightsHeader
{
	unsigned int causal;
	unsigned int nLayers;
	unsigned int inputSize;
	unsigned int dModel;
	unsigned int dFF;
	unsigned int nHeads;
	unsigned int outSize;
	unsigned int nKVHeads;
	unsigned int ffnKind;
	unsigned int tokenModel;
	unsigned int vocabSize;
	unsigned int padTokenId;
	unsigned int tieEmbeddings;

	TransformerTensorWeightsHeader()
	    : causal(0u),
	      nLayers(0u),
	      inputSize(0u),
	      dModel(0u),
	      dFF(0u),
	      nHeads(0u),
	      outSize(0u),
	      nKVHeads(0u),
	      ffnKind(0u),
	      tokenModel(0u),
	      vocabSize(0u),
	      padTokenId(0u),
	      tieEmbeddings(0u)
	{
	}
};

struct TransformerWeightWriteField
{
	const char* name;
	const std::vector<float>* values;

	TransformerWeightWriteField(const char* fieldName, const std::vector<float>& fieldValues)
	    : name(fieldName), values(&fieldValues)
	{
	}
};

struct TransformerWeightReadField
{
	const char* name;
	std::vector<float>* values;
	size_t expectedCount;

	TransformerWeightReadField(const char* fieldName, std::vector<float>& fieldValues, size_t fieldExpectedCount)
	    : name(fieldName), values(&fieldValues), expectedCount(fieldExpectedCount)
	{
	}
};

static void append_transformer_write_field(std::vector<TransformerWeightWriteField>& fields,
                                           const char* name,
                                           const std::vector<float>& values)
{
	fields.push_back(TransformerWeightWriteField(name, values));
}

static void append_transformer_read_field(std::vector<TransformerWeightReadField>& fields,
                                          const char* name,
                                          std::vector<float>& values,
                                          size_t expectedCount)
{
	fields.push_back(TransformerWeightReadField(name, values, expectedCount));
}

static glades::NNetworkStatus write_transformer_weights_header(std::ostream& out,
                                                               const TransformerTensorWeightsHeader& header)
{
	write_u32_le(out, header.causal);
	write_u32_le(out, header.nLayers);
	write_u32_le(out, header.inputSize);
	write_u32_le(out, header.dModel);
	write_u32_le(out, header.dFF);
	write_u32_le(out, header.nHeads);
	write_u32_le(out, header.outSize);
	write_u32_le(out, header.nKVHeads);
	write_u32_le(out, header.ffnKind);
	write_u32_le(out, header.tokenModel);
	write_u32_le(out, header.vocabSize);
	write_u32_le(out, header.padTokenId);
	write_u32_le(out, header.tieEmbeddings);
	if (!out)
		return glades::NNetworkStatus(glades::NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: failed to write transformer header");
	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

static glades::NNetworkStatus read_transformer_weights_header(std::istream& in,
                                                              TransformerTensorWeightsHeader& header)
{
	if (!read_u32_le(in, header.causal))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.causal");
	if (!read_u32_le(in, header.nLayers))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.nLayers");
	if (!read_u32_le(in, header.inputSize))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.inputSize");
	if (!read_u32_le(in, header.dModel))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.dModel");
	if (!read_u32_le(in, header.dFF))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.dFF");
	if (!read_u32_le(in, header.nHeads))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.nHeads");
	if (!read_u32_le(in, header.outSize))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.outSize");
	if (!read_u32_le(in, header.nKVHeads))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.nKVHeads");
	if (!read_u32_le(in, header.ffnKind))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.ffnKind");
	if (!read_u32_le(in, header.tokenModel))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.tokenModel");
	if (!read_u32_le(in, header.vocabSize))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.vocabSize");
	if (!read_u32_le(in, header.padTokenId))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.padTokenId");
	if (!read_u32_le(in, header.tieEmbeddings))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing transformer.tieEmbeddings");
	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

static glades::NNetworkStatus write_transformer_weight_fields(std::ostream& out,
                                                              const std::vector<TransformerWeightWriteField>& fields)
{
	for (size_t i = 0; i < fields.size(); ++i)
	{
		if (!write_vec_f32(out, *fields[i].values))
		{
			std::string msg("saveTensorWeightsToFile: write failed (Transformer ");
			msg += fields[i].name;
			msg += ")";
			return glades::NNetworkStatus(glades::NNetworkStatus::INTERNAL_ERROR, msg);
		}
	}
	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

static glades::NNetworkStatus read_transformer_weight_fields(std::istream& in,
                                                             const std::vector<TransformerWeightReadField>& fields)
{
	for (size_t i = 0; i < fields.size(); ++i)
	{
		if (!read_vec_f32_exact(in, *fields[i].values, fields[i].expectedCount))
		{
			std::string msg("loadTensorWeightsFromFile: failed to read Transformer ");
			msg += fields[i].name;
			msg += " (size mismatch/corrupt)";
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, msg);
		}
	}
	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

static void append_transformer_global_write_fields(std::vector<TransformerWeightWriteField>& fields,
                                                   const std::vector<float>& WIn,
                                                   const std::vector<float>& bIn,
                                                   const std::vector<float>& WOut,
                                                   const std::vector<float>& bOut,
                                                   const std::vector<float>& tokE,
                                                   const std::vector<float>& lmBias,
                                                   const std::vector<float>& lnFinalGamma,
                                                   const std::vector<float>& lnFinalBeta)
{
	fields.clear();
	fields.reserve(8u);
	append_transformer_write_field(fields, "WIn", WIn);
	append_transformer_write_field(fields, "bIn", bIn);
	append_transformer_write_field(fields, "WOut", WOut);
	append_transformer_write_field(fields, "bOut", bOut);
	append_transformer_write_field(fields, "tokE", tokE);
	append_transformer_write_field(fields, "lmBias", lmBias);
	append_transformer_write_field(fields, "lnFinalGamma", lnFinalGamma);
	append_transformer_write_field(fields, "lnFinalBeta", lnFinalBeta);
}

static void append_transformer_global_read_fields(std::vector<TransformerWeightReadField>& fields,
                                                  std::vector<float>& WIn,
                                                  size_t WInCount,
                                                  std::vector<float>& bIn,
                                                  size_t bInCount,
                                                  std::vector<float>& WOut,
                                                  size_t WOutCount,
                                                  std::vector<float>& bOut,
                                                  size_t bOutCount,
                                                  std::vector<float>& tokE,
                                                  size_t tokECount,
                                                  std::vector<float>& lmBias,
                                                  size_t lmBiasCount,
                                                  std::vector<float>& lnFinalGamma,
                                                  size_t lnFinalGammaCount,
                                                  std::vector<float>& lnFinalBeta,
                                                  size_t lnFinalBetaCount)
{
	fields.clear();
	fields.reserve(8u);
	append_transformer_read_field(fields, "WIn", WIn, WInCount);
	append_transformer_read_field(fields, "bIn", bIn, bInCount);
	append_transformer_read_field(fields, "WOut", WOut, WOutCount);
	append_transformer_read_field(fields, "bOut", bOut, bOutCount);
	append_transformer_read_field(fields, "tokE", tokE, tokECount);
	append_transformer_read_field(fields, "lmBias", lmBias, lmBiasCount);
	append_transformer_read_field(fields, "lnFinalGamma", lnFinalGamma, lnFinalGammaCount);
	append_transformer_read_field(fields, "lnFinalBeta", lnFinalBeta, lnFinalBetaCount);
}

static void append_transformer_block_write_fields(std::vector<TransformerWeightWriteField>& fields,
                                                  const std::vector<float>& ln1Gamma,
                                                  const std::vector<float>& ln1Beta,
                                                  const std::vector<float>& Wq,
                                                  const std::vector<float>& Wk,
                                                  const std::vector<float>& Wv,
                                                  const std::vector<float>& Wo,
                                                  const std::vector<float>& bq,
                                                  const std::vector<float>& bk,
                                                  const std::vector<float>& bv,
                                                  const std::vector<float>& bo,
                                                  const std::vector<float>& ln2Gamma,
                                                  const std::vector<float>& ln2Beta,
                                                  const std::vector<float>& W1,
                                                  const std::vector<float>& b1,
                                                  const std::vector<float>& W2,
                                                  const std::vector<float>& b2)
{
	fields.clear();
	fields.reserve(16u);
	append_transformer_write_field(fields, "ln1Gamma", ln1Gamma);
	append_transformer_write_field(fields, "ln1Beta", ln1Beta);
	append_transformer_write_field(fields, "Wq", Wq);
	append_transformer_write_field(fields, "Wk", Wk);
	append_transformer_write_field(fields, "Wv", Wv);
	append_transformer_write_field(fields, "Wo", Wo);
	append_transformer_write_field(fields, "bq", bq);
	append_transformer_write_field(fields, "bk", bk);
	append_transformer_write_field(fields, "bv", bv);
	append_transformer_write_field(fields, "bo", bo);
	append_transformer_write_field(fields, "ln2Gamma", ln2Gamma);
	append_transformer_write_field(fields, "ln2Beta", ln2Beta);
	append_transformer_write_field(fields, "W1", W1);
	append_transformer_write_field(fields, "b1", b1);
	append_transformer_write_field(fields, "W2", W2);
	append_transformer_write_field(fields, "b2", b2);
}

static void append_transformer_block_read_fields(std::vector<TransformerWeightReadField>& fields,
                                                 std::vector<float>& ln1Gamma,
                                                 size_t ln1GammaCount,
                                                 std::vector<float>& ln1Beta,
                                                 size_t ln1BetaCount,
                                                 std::vector<float>& Wq,
                                                 size_t WqCount,
                                                 std::vector<float>& Wk,
                                                 size_t WkCount,
                                                 std::vector<float>& Wv,
                                                 size_t WvCount,
                                                 std::vector<float>& Wo,
                                                 size_t WoCount,
                                                 std::vector<float>& bq,
                                                 size_t bqCount,
                                                 std::vector<float>& bk,
                                                 size_t bkCount,
                                                 std::vector<float>& bv,
                                                 size_t bvCount,
                                                 std::vector<float>& bo,
                                                 size_t boCount,
                                                 std::vector<float>& ln2Gamma,
                                                 size_t ln2GammaCount,
                                                 std::vector<float>& ln2Beta,
                                                 size_t ln2BetaCount,
                                                 std::vector<float>& W1,
                                                 size_t W1Count,
                                                 std::vector<float>& b1,
                                                 size_t b1Count,
                                                 std::vector<float>& W2,
                                                 size_t W2Count,
                                                 std::vector<float>& b2,
                                                 size_t b2Count)
{
	fields.clear();
	fields.reserve(16u);
	append_transformer_read_field(fields, "ln1Gamma", ln1Gamma, ln1GammaCount);
	append_transformer_read_field(fields, "ln1Beta", ln1Beta, ln1BetaCount);
	append_transformer_read_field(fields, "Wq", Wq, WqCount);
	append_transformer_read_field(fields, "Wk", Wk, WkCount);
	append_transformer_read_field(fields, "Wv", Wv, WvCount);
	append_transformer_read_field(fields, "Wo", Wo, WoCount);
	append_transformer_read_field(fields, "bq", bq, bqCount);
	append_transformer_read_field(fields, "bk", bk, bkCount);
	append_transformer_read_field(fields, "bv", bv, bvCount);
	append_transformer_read_field(fields, "bo", bo, boCount);
	append_transformer_read_field(fields, "ln2Gamma", ln2Gamma, ln2GammaCount);
	append_transformer_read_field(fields, "ln2Beta", ln2Beta, ln2BetaCount);
	append_transformer_read_field(fields, "W1", W1, W1Count);
	append_transformer_read_field(fields, "b1", b1, b1Count);
	append_transformer_read_field(fields, "W2", W2, W2Count);
	append_transformer_read_field(fields, "b2", b2, b2Count);
}
} // namespace

glades::NNetworkStatus glades::NNetwork::saveTensorWeightsToFile(const std::string& filePath) const
{
	if (filePath.empty())
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveTensorWeightsToFile: filePath is empty");
	if (!skeleton)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "saveTensorWeightsToFile: skeleton is null");

	// Ensure tensors exist (but don't require an attached DataInput if tensors are already initialized).
	{
		const bool hasDff = (netType == TYPE_DFF) && tensorDff.initialized;
		const bool hasRnn = (netType == TYPE_RNN) && tensorRnn.initialized;
		const bool hasGru = (netType == TYPE_GRU) && tensorGru.initialized;
		const bool hasLstm = (netType == TYPE_LSTM) && tensorLstm.initialized;
		const bool hasTr = (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER) && tensorTransformer.initialized;
		const bool hasCnn = (netType == TYPE_CNN) && tensorCnn.initialized;
		if (!hasDff && !hasRnn && !hasGru && !hasLstm && !hasTr && !hasCnn)
		{
			if (!const_cast<glades::NNetwork*>(this)->ensureTensorParametersInitialized())
				return lastStatus;
		}
	}

	// Atomic write: write to a temporary file then rename into place.
	const std::string tmpPath = filePath + ".tmp";
	std::ofstream out(tmpPath.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
	if (!out)
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: unable to open file for writing");
	struct TmpFileGuard
	{
		std::string path;
		bool keep;
		explicit TmpFileGuard(const std::string& p) : path(p), keep(false) {}
		~TmpFileGuard()
		{
			if (!keep && !path.empty())
				(void)::remove(path.c_str());
		}
		void dismiss() { keep = true; }
	private:
		TmpFileGuard(const TmpFileGuard&);
		TmpFileGuard& operator=(const TmpFileGuard&);
	};
	TmpFileGuard tmpGuard(tmpPath);

	// Fixed-size header:
	// magic[32], version(u32), netType(u32), reserved(u32), reserved(u32)
	{
		char magic[kTensorWeightsMagicBinFixedBytes];
		std::memset(magic, 0, sizeof(magic));
		const size_t ml = strlen(kTensorWeightsMagicBin);
		const size_t copyN = (ml < sizeof(magic) ? ml : sizeof(magic));
		std::memcpy(magic, kTensorWeightsMagicBin, copyN);
		out.write(magic, static_cast<std::streamsize>(sizeof(magic)));
		write_u32_le(out, kTensorWeightsVersionBin);
		write_u32_le(out, static_cast<unsigned int>(netType));
		write_u32_le(out, 0u);
		write_u32_le(out, 0u);
		if (!out)
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: failed to write header");
	}

	if (netType == TYPE_DFF)
	{
		write_u32_le(out, static_cast<unsigned int>(tensorDff.T.size()));
		for (size_t t = 0; t < tensorDff.T.size(); ++t)
		{
			const TensorDFFState::Transition& tr = tensorDff.T[t];
			write_u32_le(out, tr.in);
			write_u32_le(out, tr.out);
			if (!write_vec_f32(out, tr.W)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (DFF W)");
			if (!write_vec_f32(out, tr.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (DFF bias)");
		}
		out.flush();
		out.close();
		if (!out || ::rename(tmpPath.c_str(), filePath.c_str()) != 0)
		{
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: failed to publish file (rename failed)");
		}
		tmpGuard.dismiss();
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_RNN)
	{
		write_u32_le(out, static_cast<unsigned int>(tensorRnn.H.size()));
		write_u32_le(out, tensorRnn.inputSize);
		write_u32_le(out, tensorRnn.outSize);
		for (size_t l = 0; l < tensorRnn.H.size(); ++l)
		{
			const TensorRNNState::Hidden& hl = tensorRnn.H[l];
			write_u32_le(out, hl.in);
			write_u32_le(out, hl.h);
			if (!write_vec_f32(out, hl.Wxh)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (RNN Wxh)");
			if (!write_vec_f32(out, hl.Whh)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (RNN Whh)");
			if (!write_vec_f32(out, hl.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (RNN bias)");
		}
		write_u32_le(out, tensorRnn.O.in);
		write_u32_le(out, tensorRnn.O.out);
		if (!write_vec_f32(out, tensorRnn.O.Why)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (RNN Why)");
		if (!write_vec_f32(out, tensorRnn.O.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (RNN out bias)");
		out.flush();
		out.close();
		if (!out || ::rename(tmpPath.c_str(), filePath.c_str()) != 0)
		{
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: failed to publish file (rename failed)");
		}
		tmpGuard.dismiss();
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_GRU || netType == TYPE_LSTM)
	{
		const TensorGatedState& tg = (netType == TYPE_GRU) ? tensorGru : tensorLstm;
		write_u32_le(out, tg.gateCount);
		write_u32_le(out, static_cast<unsigned int>(tg.H.size()));
		write_u32_le(out, tg.inputSize);
		write_u32_le(out, tg.outSize);
		for (size_t l = 0; l < tg.H.size(); ++l)
		{
			const TensorGatedState::Hidden& hl = tg.H[l];
			write_u32_le(out, hl.in);
			write_u32_le(out, hl.h);
			if (!write_vec_f32(out, hl.W)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (gated W)");
			if (!write_vec_f32(out, hl.U)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (gated U)");
			if (!write_vec_f32(out, hl.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (gated bias)");
		}
		write_u32_le(out, tg.O.in);
		write_u32_le(out, tg.O.out);
		if (!write_vec_f32(out, tg.O.Why)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (gated Why)");
		if (!write_vec_f32(out, tg.O.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (gated out bias)");
		out.flush();
		out.close();
		if (!out || ::rename(tmpPath.c_str(), filePath.c_str()) != 0)
		{
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: failed to publish file (rename failed)");
		}
		tmpGuard.dismiss();
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER)
	{
		const TensorTransformerState& tt = tensorTransformer;
		TransformerTensorWeightsHeader header;
		header.causal = tt.causal ? 1u : 0u;
		header.nLayers = static_cast<unsigned int>(tt.blocks.size());
		header.inputSize = tt.inputSize;
		header.dModel = tt.dModel;
		header.dFF = tt.dFF;
		header.nHeads = tt.nHeads;
		header.outSize = tt.outSize;
		header.nKVHeads = tt.nKVHeads;
		header.ffnKind = tt.ffnKind;
		header.tokenModel = tt.tokenModel ? 1u : 0u;
		header.vocabSize = tt.vocabSize;
		header.padTokenId = static_cast<unsigned int>(tt.padTokenId);
		header.tieEmbeddings = tt.tieEmbeddings ? 1u : 0u;
		{
			const NNetworkStatus stHeader = write_transformer_weights_header(out, header);
			if (!stHeader.ok())
				return stHeader;
		}

		std::vector<TransformerWeightWriteField> fields;
		append_transformer_global_write_fields(fields,
		                                     tt.WIn, tt.bIn,
		                                     tt.WOut, tt.bOut,
		                                     tt.tokE, tt.lmBias,
		                                     tt.lnFinalGamma, tt.lnFinalBeta);
		{
			const NNetworkStatus stGlobals = write_transformer_weight_fields(out, fields);
			if (!stGlobals.ok())
				return stGlobals;
		}

		for (size_t l = 0; l < tt.blocks.size(); ++l)
		{
			const TensorTransformerState::Block& b = tt.blocks[l];
			append_transformer_block_write_fields(fields,
			                                      b.ln1Gamma, b.ln1Beta,
			                                      b.Wq, b.Wk, b.Wv, b.Wo,
			                                      b.bq, b.bk, b.bv, b.bo,
			                                      b.ln2Gamma, b.ln2Beta,
			                                      b.W1, b.b1, b.W2, b.b2);
			const NNetworkStatus stBlock = write_transformer_weight_fields(out, fields);
			if (!stBlock.ok())
				return stBlock;
		}

		out.flush();
		out.close();
		if (!out || ::rename(tmpPath.c_str(), filePath.c_str()) != 0)
		{
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: failed to publish file (rename failed)");
		}
		tmpGuard.dismiss();
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_CNN)
	{
		const TensorCNNState& cs = tensorCnn;
		write_u32_le(out, static_cast<unsigned int>(cs.convLayers.size()));
		write_u32_le(out, static_cast<unsigned int>(cs.fcLayers.size()));
		write_u32_le(out, cs.inputH);
		write_u32_le(out, cs.inputW);
		write_u32_le(out, cs.inputC);
		write_u32_le(out, cs.flattenedSize);
		for (size_t l = 0; l < cs.convLayers.size(); ++l)
		{
			const TensorCNNState::ConvLayer& cl = cs.convLayers[l];
			const TensorCNNState::ConvSpatialInfo& sp = cs.spatialInfo[l];
			write_u32_le(out, cl.inC);
			write_u32_le(out, cl.outC);
			write_u32_le(out, cl.kH);
			write_u32_le(out, cl.kW);
			// Spatial config needed to reconstruct spatial info on load.
			write_u32_le(out, sp.strideH);
			write_u32_le(out, sp.strideW);
			write_u32_le(out, sp.padH);
			write_u32_le(out, sp.padW);
			write_u32_le(out, sp.useBatchNorm ? 1u : 0u);
			write_u32_le(out, sp.useMaxPool ? 1u : 0u);
			write_u32_le(out, sp.poolH);
			write_u32_le(out, sp.poolW);
			write_u32_le(out, sp.poolStrideH);
			write_u32_le(out, sp.poolStrideW);
			write_u32_le(out, sp.outH);
			write_u32_le(out, sp.outW);
			write_u32_le(out, sp.poolOutH);
			write_u32_le(out, sp.poolOutW);
			write_u32_le(out, sp.im2colRows);
			write_u32_le(out, sp.im2colCols);
			if (!write_vec_f32(out, cl.W)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (CNN conv W)");
			if (!write_vec_f32(out, cl.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (CNN conv bias)");
			if (!write_vec_f32(out, cl.bnGamma)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (CNN bnGamma)");
			if (!write_vec_f32(out, cl.bnBeta)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (CNN bnBeta)");
			if (!write_vec_f32(out, cl.bnRunMean)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (CNN bnRunMean)");
			if (!write_vec_f32(out, cl.bnRunVar)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (CNN bnRunVar)");
		}
		for (size_t t = 0; t < cs.fcLayers.size(); ++t)
		{
			const TensorCNNState::FCTransition& fc = cs.fcLayers[t];
			write_u32_le(out, fc.in);
			write_u32_le(out, fc.out);
			if (!write_vec_f32(out, fc.W)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (CNN FC W)");
			if (!write_vec_f32(out, fc.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (CNN FC bias)");
		}
		out.flush();
		out.close();
		if (!out || ::rename(tmpPath.c_str(), filePath.c_str()) != 0)
		{
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: failed to publish file (rename failed)");
		}
		tmpGuard.dismiss();
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	// Unknown netType; ensure we don't leak a temp file.
	return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveTensorWeightsToFile: unknown netType");
}

glades::NNetworkStatus glades::NNetwork::loadTensorWeightsFromFile(const std::string& filePath)
{
	if (filePath.empty())
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadTensorWeightsFromFile: filePath is empty");
	if (!skeleton)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: skeleton is null");

	std::ifstream in(filePath.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return failStatus(NNetworkStatus::INTERNAL_ERROR, "loadTensorWeightsFromFile: unable to open file");

	// Header
	char magic[kTensorWeightsMagicBinFixedBytes];
	in.read(magic, static_cast<std::streamsize>(sizeof(magic)));
	if (!in)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: unable to read header magic");
	{
		const size_t ml = strlen(kTensorWeightsMagicBin);
		if (ml > sizeof(magic) || std::memcmp(magic, kTensorWeightsMagicBin, ml) != 0)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: magic mismatch");
	}
	unsigned int version = 0u;
	unsigned int fileNetTypeU = 0u;
	unsigned int r0 = 0u, r1 = 0u;
	if (!read_u32_le(in, version) || !read_u32_le(in, fileNetTypeU) || !read_u32_le(in, r0) || !read_u32_le(in, r1))
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: unable to read header fields");
	if (version != kTensorWeightsVersionBin)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: unsupported tensor weights version");
	const int fileNetType = static_cast<int>(fileNetTypeU);
	if (fileNetType != netType)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: netType mismatch vs manifest");

	if (netType == TYPE_DFF)
	{
		unsigned int transitionsU = 0u;
		if (!read_u32_le(in, transitionsU))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing dff.transitions");
		const size_t transitions = static_cast<size_t>(transitionsU);

		tensorDff.reset();
		tensorDff.T.resize(transitions);
		tensorDff.sizes.clear();
		tensorDff.sizes.reserve(transitions + 1u);

		for (size_t t = 0; t < transitions; ++t)
		{
			unsigned int inSize = 0, outSize = 0;
			if (!read_u32_le(in, inSize) || !read_u32_le(in, outSize))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed dff transition header");

			TensorDFFState::Transition& tr = tensorDff.T[t];
			tr.in = inSize;
			tr.out = outSize;

			if (t == 0u)
				tensorDff.sizes.push_back(inSize);
			tensorDff.sizes.push_back(outSize);

			{
				size_t want = 0u;
				if (!mul_size_checked(static_cast<size_t>(outSize), static_cast<size_t>(inSize), want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: DFF W size overflow");
				if (!read_vec_f32_exact(in, tr.W, want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read DFF W (size mismatch/corrupt)");
			}
			if (!read_vec_f32_exact(in, tr.bias, static_cast<size_t>(outSize)))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read DFF bias (size mismatch/corrupt)");

			// Reset optimizer/grads.
			tr.vW.assign(tr.W.size(), 0.0f);
			tr.gW.assign(tr.W.size(), 0.0f);
			tr.gBias.assign(tr.bias.size(), 0.0f);
		}

		// Allocate activations/deltas for shape (training will overwrite).
		tensorDff.a.resize(tensorDff.sizes.size());
		tensorDff.delta.resize(tensorDff.sizes.size());
		for (size_t li = 0; li < tensorDff.sizes.size(); ++li)
		{
			tensorDff.a[li].assign(tensorDff.sizes[li], 0.0f);
			if (li == 0u) tensorDff.delta[li].clear();
			else tensorDff.delta[li].assign(tensorDff.sizes[li], 0.0f);
		}
		tensorDff.batchCount = 0u;
		tensorDff.initialized = true;
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_RNN)
	{
		unsigned int hiddenLayersU = 0u;
		unsigned int inputSize = 0, outSize = 0;
		if (!read_u32_le(in, hiddenLayersU))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing rnn.hiddenLayers");
		if (!read_u32_le(in, inputSize))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing rnn.inputSize");
		if (!read_u32_le(in, outSize))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing rnn.outSize");
		const size_t hiddenLayers = static_cast<size_t>(hiddenLayersU);

		tensorRnn.reset();
		tensorRnn.initialized = true;
		tensorRnn.inputSize = inputSize;
		tensorRnn.outSize = outSize;
		tensorRnn.hiddenSizes.assign(hiddenLayers, 0u);
		tensorRnn.H.resize(hiddenLayers);

		for (size_t l = 0; l < hiddenLayers; ++l)
		{
			unsigned int inSize = 0, hSize = 0;
			if (!read_u32_le(in, inSize) || !read_u32_le(in, hSize))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed rnn.h header");

			TensorRNNState::Hidden& hl = tensorRnn.H[l];
			hl.in = inSize;
			hl.h = hSize;
			tensorRnn.hiddenSizes[l] = hSize;

			{
				size_t want = 0u;
				if (!mul_size_checked(static_cast<size_t>(hSize), static_cast<size_t>(inSize), want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: RNN Wxh size overflow");
				if (!read_vec_f32_exact(in, hl.Wxh, want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read RNN Wxh (size mismatch/corrupt)");
			}
			{
				size_t want = 0u;
				if (!mul_size_checked(static_cast<size_t>(hSize), static_cast<size_t>(hSize), want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: RNN Whh size overflow");
				if (!read_vec_f32_exact(in, hl.Whh, want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read RNN Whh (size mismatch/corrupt)");
			}
			if (!read_vec_f32_exact(in, hl.bias, static_cast<size_t>(hSize)))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read RNN bias (size mismatch/corrupt)");

			hl.vWxh.assign(hl.Wxh.size(), 0.0f);
			hl.vWhh.assign(hl.Whh.size(), 0.0f);
			hl.gWxh.assign(hl.Wxh.size(), 0.0f);
			hl.gWhh.assign(hl.Whh.size(), 0.0f);
			hl.gBias.assign(hl.bias.size(), 0.0f);
		}

		unsigned int oIn = 0, oOut = 0;
		if (!read_u32_le(in, oIn) || !read_u32_le(in, oOut))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing rnn.o header");
		tensorRnn.O.in = oIn;
		tensorRnn.O.out = oOut;
		{
			size_t want = 0u;
			if (!mul_size_checked(static_cast<size_t>(oOut), static_cast<size_t>(oIn), want))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: RNN Why size overflow");
			if (!read_vec_f32_exact(in, tensorRnn.O.Why, want))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read RNN Why (size mismatch/corrupt)");
		}
		if (!read_vec_f32_exact(in, tensorRnn.O.bias, static_cast<size_t>(oOut)))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read RNN out bias (size mismatch/corrupt)");
		tensorRnn.O.vWhy.assign(tensorRnn.O.Why.size(), 0.0f);
		tensorRnn.O.gWhy.assign(tensorRnn.O.Why.size(), 0.0f);
		tensorRnn.O.gBias.assign(tensorRnn.O.bias.size(), 0.0f);

		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_GRU || netType == TYPE_LSTM)
	{
		unsigned int gateCount = 0;
		unsigned int hiddenLayersU = 0u;
		unsigned int inputSize = 0, outSize = 0;
		if (!read_u32_le(in, gateCount))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing gated.gateCount");
		if (!read_u32_le(in, hiddenLayersU))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing gated.hiddenLayers");
		if (!read_u32_le(in, inputSize))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing gated.inputSize");
		if (!read_u32_le(in, outSize))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing gated.outSize");
		const size_t hiddenLayers = static_cast<size_t>(hiddenLayersU);

		TensorGatedState& tg = (netType == TYPE_GRU) ? tensorGru : tensorLstm;
		tg.reset();
		tg.initialized = true;
		tg.inputSize = inputSize;
		tg.outSize = outSize;
		tg.gateCount = gateCount;
		tg.hiddenSizes.assign(hiddenLayers, 0u);
		tg.H.resize(hiddenLayers);

		for (size_t l = 0; l < hiddenLayers; ++l)
		{
			unsigned int inSize = 0, hSize = 0;
			if (!read_u32_le(in, inSize) || !read_u32_le(in, hSize))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed gated.h header");

			TensorGatedState::Hidden& hl = tg.H[l];
			hl.in = inSize;
			hl.h = hSize;
			tg.hiddenSizes[l] = hSize;

			{
				size_t want = 0u;
				if (!mul_size_checked(static_cast<size_t>(gateCount) * static_cast<size_t>(hSize), static_cast<size_t>(inSize), want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: gated W size overflow");
				if (!read_vec_f32_exact(in, hl.W, want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read gated W (size mismatch/corrupt)");
			}
			{
				size_t want = 0u;
				if (!mul_size_checked(static_cast<size_t>(gateCount) * static_cast<size_t>(hSize), static_cast<size_t>(hSize), want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: gated U size overflow");
				if (!read_vec_f32_exact(in, hl.U, want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read gated U (size mismatch/corrupt)");
			}
			if (!read_vec_f32_exact(in, hl.bias, static_cast<size_t>(gateCount) * static_cast<size_t>(hSize)))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read gated bias (size mismatch/corrupt)");

			hl.vW.assign(hl.W.size(), 0.0f);
			hl.vU.assign(hl.U.size(), 0.0f);
			hl.gW.assign(hl.W.size(), 0.0f);
			hl.gU.assign(hl.U.size(), 0.0f);
			hl.gBias.assign(hl.bias.size(), 0.0f);
		}

		unsigned int oIn = 0, oOut = 0;
		if (!read_u32_le(in, oIn) || !read_u32_le(in, oOut))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing gated.o header");
		tg.O.in = oIn;
		tg.O.out = oOut;
		{
			size_t want = 0u;
			if (!mul_size_checked(static_cast<size_t>(oOut), static_cast<size_t>(oIn), want))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: gated Why size overflow");
			if (!read_vec_f32_exact(in, tg.O.Why, want))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read gated Why (size mismatch/corrupt)");
		}
		if (!read_vec_f32_exact(in, tg.O.bias, static_cast<size_t>(oOut)))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read gated out bias (size mismatch/corrupt)");
		tg.O.vWhy.assign(tg.O.Why.size(), 0.0f);
		tg.O.gWhy.assign(tg.O.Why.size(), 0.0f);
		tg.O.gBias.assign(tg.O.bias.size(), 0.0f);

		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER)
	{
		TransformerTensorWeightsHeader header;
		{
			const NNetworkStatus stHeader = read_transformer_weights_header(in, header);
			if (!stHeader.ok())
				return failStatus(stHeader.code, stHeader.message);
		}

		const unsigned int causalIntU = header.causal;
		const unsigned int nLayersU = header.nLayers;
		const unsigned int inputSize = header.inputSize;
		const unsigned int dModel = header.dModel;
		const unsigned int dFF = header.dFF;
		const unsigned int nHeads = header.nHeads;
		const unsigned int outSize = header.outSize;
		const unsigned int nKVHeads = header.nKVHeads;
		const unsigned int ffnKind = header.ffnKind;
		const unsigned int tokenModelU = header.tokenModel;
		const unsigned int vocabSizeU = header.vocabSize;
		const unsigned int padTokenU = header.padTokenId;
		const unsigned int tieEmbU = header.tieEmbeddings;
		const size_t nLayers = static_cast<size_t>(nLayersU);

		tensorTransformer.reset();
		tensorTransformer.initialized = true;
		tensorTransformer.causal = (causalIntU != 0u);
		tensorTransformer.inputSize = inputSize;
		tensorTransformer.dModel = dModel;
		tensorTransformer.dFF = dFF;
		tensorTransformer.nHeads = nHeads;
		tensorTransformer.nKVHeads = nKVHeads;
		tensorTransformer.outSize = outSize;
		tensorTransformer.nLayers = static_cast<unsigned int>(nLayers);
		tensorTransformer.ffnKind = ffnKind;
		tensorTransformer.tokenModel = (tokenModelU != 0u);
		tensorTransformer.vocabSize = vocabSizeU;
		tensorTransformer.padTokenId = static_cast<int>(padTokenU);
		tensorTransformer.tieEmbeddings = (tieEmbU != 0u);
		tensorTransformer.optimizerStep = 0ULL;

		// Validate transformer dimensions before any large allocations.
		if (dModel == 0u || nHeads == 0u)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: invalid transformer dimensions");
		if ((dModel % nHeads) != 0u)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: transformer dModel must be divisible by nHeads");
		if (nKVHeads == 0u || (nHeads % nKVHeads) != 0u)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: transformer nKVHeads must divide nHeads");
		const unsigned int dHead = dModel / nHeads;
		const unsigned int dModelKV = nKVHeads * dHead;
		const unsigned int ff1Width =
		    (ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFF) : dFF;
		if (ff1Width == 0u)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: invalid transformer FFN width");
		if (tensorTransformer.tokenModel && !tensorTransformer.tieEmbeddings)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: tokenModel requires tieEmbeddings");

		size_t WInCount = 0u;
		if (!mul_size_checked(static_cast<size_t>(dModel), static_cast<size_t>(inputSize), WInCount))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: Transformer WIn size overflow");
		size_t WOutCount = 0u;
		if (!mul_size_checked(static_cast<size_t>(outSize), static_cast<size_t>(dModel), WOutCount))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: Transformer WOut size overflow");
		size_t tokECount = 0u;
		if (tensorTransformer.tokenModel &&
		    !mul_size_checked(static_cast<size_t>(tensorTransformer.vocabSize), static_cast<size_t>(dModel), tokECount))
		{
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: Transformer tokE size overflow");
		}
		const size_t dModelCount = static_cast<size_t>(dModel);
		const size_t outSizeCount = static_cast<size_t>(outSize);
		const size_t vocabCount = tensorTransformer.tokenModel ? static_cast<size_t>(tensorTransformer.vocabSize) : 0u;
		std::vector<TransformerWeightReadField> fields;
		append_transformer_global_read_fields(fields,
		                                     tensorTransformer.WIn, WInCount,
		                                     tensorTransformer.bIn, dModelCount,
		                                     tensorTransformer.WOut, WOutCount,
		                                     tensorTransformer.bOut, outSizeCount,
		                                     tensorTransformer.tokE, tokECount,
		                                     tensorTransformer.lmBias, vocabCount,
		                                     tensorTransformer.lnFinalGamma, dModelCount,
		                                     tensorTransformer.lnFinalBeta, dModelCount);
		{
			const NNetworkStatus stGlobals = read_transformer_weight_fields(in, fields);
			if (!stGlobals.ok())
				return failStatus(stGlobals.code, stGlobals.message);
		}

		tensorTransformer.vWIn.assign(tensorTransformer.WIn.size(), 0.0f);
		tensorTransformer.v2WIn.assign(tensorTransformer.WIn.size(), 0.0f);
		tensorTransformer.gWIn.assign(tensorTransformer.WIn.size(), 0.0f);
		tensorTransformer.mBIn.assign(tensorTransformer.bIn.size(), 0.0f);
		tensorTransformer.v2BIn.assign(tensorTransformer.bIn.size(), 0.0f);
		tensorTransformer.gBIn.assign(tensorTransformer.bIn.size(), 0.0f);
		tensorTransformer.vWOut.assign(tensorTransformer.WOut.size(), 0.0f);
		tensorTransformer.v2WOut.assign(tensorTransformer.WOut.size(), 0.0f);
		tensorTransformer.gWOut.assign(tensorTransformer.WOut.size(), 0.0f);
		tensorTransformer.mBOut.assign(tensorTransformer.bOut.size(), 0.0f);
		tensorTransformer.v2BOut.assign(tensorTransformer.bOut.size(), 0.0f);
		tensorTransformer.gBOut.assign(tensorTransformer.bOut.size(), 0.0f);
		tensorTransformer.vTokE.assign(tensorTransformer.tokE.size(), 0.0f);
		tensorTransformer.v2TokE.assign(tensorTransformer.tokE.size(), 0.0f);
		tensorTransformer.gTokE.assign(tensorTransformer.tokE.size(), 0.0f);
		tensorTransformer.mLmBias.assign(tensorTransformer.lmBias.size(), 0.0f);
		tensorTransformer.v2LmBias.assign(tensorTransformer.lmBias.size(), 0.0f);
		tensorTransformer.gLmBias.assign(tensorTransformer.lmBias.size(), 0.0f);
		tensorTransformer.mLnFinalGamma.assign(tensorTransformer.lnFinalGamma.size(), 0.0f);
		tensorTransformer.v2LnFinalGamma.assign(tensorTransformer.lnFinalGamma.size(), 0.0f);
		tensorTransformer.gLnFinalGamma.assign(tensorTransformer.lnFinalGamma.size(), 0.0f);
		tensorTransformer.mLnFinalBeta.assign(tensorTransformer.lnFinalBeta.size(), 0.0f);
		tensorTransformer.v2LnFinalBeta.assign(tensorTransformer.lnFinalBeta.size(), 0.0f);
		tensorTransformer.gLnFinalBeta.assign(tensorTransformer.lnFinalBeta.size(), 0.0f);

		size_t WqCount = 0u;
		if (!mul_size_checked(static_cast<size_t>(dModel), static_cast<size_t>(dModel), WqCount))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: Transformer Wq size overflow");
		size_t WkCount = 0u;
		if (!mul_size_checked(static_cast<size_t>(dModelKV), static_cast<size_t>(dModel), WkCount))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: Transformer Wk size overflow");
		size_t WvCount = 0u;
		if (!mul_size_checked(static_cast<size_t>(dModelKV), static_cast<size_t>(dModel), WvCount))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: Transformer Wv size overflow");
		size_t WoCount = 0u;
		if (!mul_size_checked(static_cast<size_t>(dModel), static_cast<size_t>(dModel), WoCount))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: Transformer Wo size overflow");
		size_t W1Count = 0u;
		if (!mul_size_checked(static_cast<size_t>(ff1Width), static_cast<size_t>(dModel), W1Count))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: Transformer W1 size overflow");
		size_t W2Count = 0u;
		if (!mul_size_checked(static_cast<size_t>(dModel), static_cast<size_t>(dFF), W2Count))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: Transformer W2 size overflow");
		const size_t dModelKVCount = static_cast<size_t>(dModelKV);
		const size_t ff1WidthCount = static_cast<size_t>(ff1Width);

		tensorTransformer.blocks.resize(nLayers);
		for (size_t l = 0; l < nLayers; ++l)
		{
			TensorTransformerState::Block& b = tensorTransformer.blocks[l];
			append_transformer_block_read_fields(fields,
			                                     b.ln1Gamma, dModelCount,
			                                     b.ln1Beta, dModelCount,
			                                     b.Wq, WqCount,
			                                     b.Wk, WkCount,
			                                     b.Wv, WvCount,
			                                     b.Wo, WoCount,
			                                     b.bq, dModelCount,
			                                     b.bk, dModelKVCount,
			                                     b.bv, dModelKVCount,
			                                     b.bo, dModelCount,
			                                     b.ln2Gamma, dModelCount,
			                                     b.ln2Beta, dModelCount,
			                                     b.W1, W1Count,
			                                     b.b1, ff1WidthCount,
			                                     b.W2, W2Count,
			                                     b.b2, dModelCount);
			const NNetworkStatus stBlock = read_transformer_weight_fields(in, fields);
			if (!stBlock.ok())
				return failStatus(stBlock.code, stBlock.message);

			b.mLn1Gamma.assign(b.ln1Gamma.size(), 0.0f);
			b.v2Ln1Gamma.assign(b.ln1Gamma.size(), 0.0f);
			b.mLn1Beta.assign(b.ln1Beta.size(), 0.0f);
			b.v2Ln1Beta.assign(b.ln1Beta.size(), 0.0f);
			b.gLn1Gamma.assign(b.ln1Gamma.size(), 0.0f);
			b.gLn1Beta.assign(b.ln1Beta.size(), 0.0f);
			b.vWq.assign(b.Wq.size(), 0.0f); b.vWk.assign(b.Wk.size(), 0.0f); b.vWv.assign(b.Wv.size(), 0.0f); b.vWo.assign(b.Wo.size(), 0.0f);
			b.v2Wq.assign(b.Wq.size(), 0.0f); b.v2Wk.assign(b.Wk.size(), 0.0f); b.v2Wv.assign(b.Wv.size(), 0.0f); b.v2Wo.assign(b.Wo.size(), 0.0f);
			b.gWq.assign(b.Wq.size(), 0.0f); b.gWk.assign(b.Wk.size(), 0.0f); b.gWv.assign(b.Wv.size(), 0.0f); b.gWo.assign(b.Wo.size(), 0.0f);
			b.mBq.assign(b.bq.size(), 0.0f); b.mBk.assign(b.bk.size(), 0.0f); b.mBv.assign(b.bv.size(), 0.0f); b.mBo.assign(b.bo.size(), 0.0f);
			b.v2Bq.assign(b.bq.size(), 0.0f); b.v2Bk.assign(b.bk.size(), 0.0f); b.v2Bv.assign(b.bv.size(), 0.0f); b.v2Bo.assign(b.bo.size(), 0.0f);
			b.gBq.assign(b.bq.size(), 0.0f); b.gBk.assign(b.bk.size(), 0.0f); b.gBv.assign(b.bv.size(), 0.0f); b.gBo.assign(b.bo.size(), 0.0f);
			b.mLn2Gamma.assign(b.ln2Gamma.size(), 0.0f);
			b.v2Ln2Gamma.assign(b.ln2Gamma.size(), 0.0f);
			b.mLn2Beta.assign(b.ln2Beta.size(), 0.0f);
			b.v2Ln2Beta.assign(b.ln2Beta.size(), 0.0f);
			b.gLn2Gamma.assign(b.ln2Gamma.size(), 0.0f);
			b.gLn2Beta.assign(b.ln2Beta.size(), 0.0f);
			b.vW1.assign(b.W1.size(), 0.0f); b.vW2.assign(b.W2.size(), 0.0f);
			b.v2W1.assign(b.W1.size(), 0.0f); b.v2W2.assign(b.W2.size(), 0.0f);
			b.gW1.assign(b.W1.size(), 0.0f); b.gW2.assign(b.W2.size(), 0.0f);
			b.mB1.assign(b.b1.size(), 0.0f); b.mB2.assign(b.b2.size(), 0.0f);
			b.v2B1.assign(b.b1.size(), 0.0f); b.v2B2.assign(b.b2.size(), 0.0f);
			b.gB1.assign(b.b1.size(), 0.0f); b.gB2.assign(b.b2.size(), 0.0f);
		}

		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_CNN)
	{
		unsigned int numConvLayers = 0u, numFCLayers = 0u;
		unsigned int loadInputH = 0u, loadInputW = 0u, loadInputC = 0u, loadFlattenedSize = 0u;
		if (!read_u32_le(in, numConvLayers))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing cnn.numConvLayers");
		if (!read_u32_le(in, numFCLayers))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing cnn.numFCLayers");
		if (!read_u32_le(in, loadInputH))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing cnn.inputH");
		if (!read_u32_le(in, loadInputW))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing cnn.inputW");
		if (!read_u32_le(in, loadInputC))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing cnn.inputC");
		if (!read_u32_le(in, loadFlattenedSize))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing cnn.flattenedSize");

		tensorCnn.reset();
		tensorCnn.inputH = loadInputH;
		tensorCnn.inputW = loadInputW;
		tensorCnn.inputC = loadInputC;
		tensorCnn.flattenedSize = loadFlattenedSize;
		tensorCnn.convLayers.resize(numConvLayers);
		tensorCnn.spatialInfo.resize(numConvLayers);

		for (size_t l = 0; l < numConvLayers; ++l)
		{
			unsigned int inC = 0u, outC = 0u, kH = 0u, kW = 0u;
			if (!read_u32_le(in, inC) || !read_u32_le(in, outC) || !read_u32_le(in, kH) || !read_u32_le(in, kW))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn conv header");

			TensorCNNState::ConvLayer& cl = tensorCnn.convLayers[l];
			cl.inC = inC; cl.outC = outC; cl.kH = kH; cl.kW = kW;

			// Read spatial config.
			TensorCNNState::ConvSpatialInfo& sp = tensorCnn.spatialInfo[l];
			sp.inC = inC; sp.outC = outC; sp.kH = kH; sp.kW = kW;
			// Reconstruct inH/inW from previous layer or from global input.
			if (l == 0u)
			{
				sp.inH = loadInputH;
				sp.inW = loadInputW;
			}
			else
			{
				const TensorCNNState::ConvSpatialInfo& prev = tensorCnn.spatialInfo[l - 1u];
				sp.inH = prev.poolOutH;
				sp.inW = prev.poolOutW;
			}
			unsigned int tmp = 0u;
			if (!read_u32_le(in, sp.strideH) || !read_u32_le(in, sp.strideW))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn spatial stride");
			if (!read_u32_le(in, sp.padH) || !read_u32_le(in, sp.padW))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn spatial pad");
			if (!read_u32_le(in, tmp)) return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn spatial useBN");
			sp.useBatchNorm = (tmp != 0u);
			if (!read_u32_le(in, tmp)) return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn spatial usePool");
			sp.useMaxPool = (tmp != 0u);
			if (!read_u32_le(in, sp.poolH) || !read_u32_le(in, sp.poolW))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn spatial pool size");
			if (!read_u32_le(in, sp.poolStrideH) || !read_u32_le(in, sp.poolStrideW))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn spatial pool stride");
			if (!read_u32_le(in, sp.outH) || !read_u32_le(in, sp.outW))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn spatial outH/W");
			if (!read_u32_le(in, sp.poolOutH) || !read_u32_le(in, sp.poolOutW))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn spatial poolOutH/W");
			if (!read_u32_le(in, sp.im2colRows) || !read_u32_le(in, sp.im2colCols))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn spatial im2col");

			{
				size_t want = 0u;
				if (!mul_size_checked(static_cast<size_t>(outC), static_cast<size_t>(inC) * kH * kW, want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: CNN conv W size overflow");
				if (!read_vec_f32_exact(in, cl.W, want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read CNN conv W");
			}
			if (!read_vec_f32_exact(in, cl.bias, static_cast<size_t>(outC)))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read CNN conv bias");

			// BN params: read_vec_f32 reads the u64 count + data (may be 0-length if no BN).
			if (!read_vec_f32(in, cl.bnGamma))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read CNN bnGamma");
			if (!read_vec_f32(in, cl.bnBeta))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read CNN bnBeta");
			if (!read_vec_f32(in, cl.bnRunMean))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read CNN bnRunMean");
			if (!read_vec_f32(in, cl.bnRunVar))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read CNN bnRunVar");

			// Reset optimizer/grads.
			cl.gW.assign(cl.W.size(), 0.0f);
			cl.gBias.assign(cl.bias.size(), 0.0f);
			cl.vW.assign(cl.W.size(), 0.0f);
			cl.v2W.assign(cl.W.size(), 0.0f);
			cl.vBias.assign(cl.bias.size(), 0.0f);
			cl.v2Bias.assign(cl.bias.size(), 0.0f);
			cl.gBnGamma.assign(cl.bnGamma.size(), 0.0f);
			cl.gBnBeta.assign(cl.bnBeta.size(), 0.0f);
			cl.vBnGamma.assign(cl.bnGamma.size(), 0.0f);
			cl.v2BnGamma.assign(cl.bnGamma.size(), 0.0f);
			cl.vBnBeta.assign(cl.bnBeta.size(), 0.0f);
			cl.v2BnBeta.assign(cl.bnBeta.size(), 0.0f);
		}

		tensorCnn.fcLayers.resize(numFCLayers);
		for (size_t t = 0; t < numFCLayers; ++t)
		{
			unsigned int fcIn = 0u, fcOut = 0u;
			if (!read_u32_le(in, fcIn) || !read_u32_le(in, fcOut))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed cnn FC header");

			TensorCNNState::FCTransition& fc = tensorCnn.fcLayers[t];
			fc.in = fcIn; fc.out = fcOut;
			{
				size_t want = 0u;
				if (!mul_size_checked(static_cast<size_t>(fcOut), static_cast<size_t>(fcIn), want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: CNN FC W size overflow");
				if (!read_vec_f32_exact(in, fc.W, want))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read CNN FC W");
			}
			if (!read_vec_f32_exact(in, fc.bias, static_cast<size_t>(fcOut)))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read CNN FC bias");

			fc.gW.assign(fc.W.size(), 0.0f);
			fc.gBias.assign(fc.bias.size(), 0.0f);
			fc.vW.assign(fc.W.size(), 0.0f);
			fc.v2W.assign(fc.W.size(), 0.0f);
			fc.vBias.assign(fc.bias.size(), 0.0f);
			fc.v2Bias.assign(fc.bias.size(), 0.0f);
		}

		tensorCnn.batchCount = 0u;
		tensorCnn.optimizerStep = 0ULL;
		tensorCnn.initialized = true;
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadTensorWeightsFromFile: unknown netType");
}

void glades::NNetwork::setLearningRateScheduleNone()
{
	trainingConfig.lrSchedule.setNone();
	lrScheduleMultiplier = 1.0f;
	lrScheduleEpochOffset = 0;
}

void glades::NNetwork::setLearningRateScheduleStep(int stepSizeEpochs, float gamma)
{
	trainingConfig.lrSchedule.setStep(stepSizeEpochs, gamma);
	lrScheduleMultiplier = 1.0f;
	lrScheduleEpochOffset = 0;
}

void glades::NNetwork::setLearningRateScheduleExp(float gamma)
{
	trainingConfig.lrSchedule.setExp(gamma);
	lrScheduleMultiplier = 1.0f;
	lrScheduleEpochOffset = 0;
}

void glades::NNetwork::setLearningRateScheduleCosine(int tMaxEpochs, float minMultiplier)
{
	trainingConfig.lrSchedule.setCosine(tMaxEpochs, minMultiplier);
	lrScheduleMultiplier = 1.0f;
	lrScheduleEpochOffset = 0;
}

void glades::NNetwork::setGlobalGradClipNorm(float clipNorm)
{
	trainingConfig.globalGradClipNorm = clipNorm;
}

void glades::NNetwork::setPerElementGradClip(float clipLimit)
{
	trainingConfig.perElementGradClip = clipLimit;
}

glades::NNetworkStatus glades::NNetwork::setTrainingConfig(const glades::TrainingConfig& cfg)
{
	// Prevent concurrent mutation while a run is active.
	RunLockGuard runGuard(*this);
	if (!runGuard.ok())
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "setTrainingConfig: network is running (not thread-safe/re-entrant)");
	NNetworkStatus st = validateTransformerTrainingConfig("setTrainingConfig", cfg);
	if (!st.ok())
		return st;
	if ((netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER) && skeleton && di)
	{
		std::vector<unsigned int> hiddenSizes;
		const int hiddenCount = skeleton->numHiddenLayers();
		hiddenSizes.reserve(static_cast<size_t>(hiddenCount > 0 ? hiddenCount : 0));
		for (int i = 0; i < hiddenCount; ++i)
		{
			const int hs = skeleton->getHiddenLayerSize(static_cast<unsigned int>(i));
			hiddenSizes.push_back(hs > 0 ? static_cast<unsigned int>(hs) : 0u);
		}
		TransformerModelConfigSnapshot modelCfg;
		st = buildTransformerModelConfigSnapshot("setTrainingConfig",
		                                         cfg,
		                                         hiddenSizes,
		                                         skeleton->getOutputLayerSize(),
		                                         cfg.transformer.enableTokenEmbedding && di->hasTokenIdInput(),
		                                         netType == TYPE_TRANSFORMER_DECODER,
		                                         modelCfg);
		if (!st.ok())
			return st;
	}
	trainingConfig = cfg;
	// Reset schedule bookkeeping to avoid leaking stale multipliers into the next run.
	lrScheduleMultiplier = 1.0f;
	lrScheduleEpochOffset = 0;
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::NNetwork::setTerminator(const glades::Terminator& t)
{
	RunLockGuard runGuard(*this);
	if (!runGuard.ok())
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "setTerminator: network is running (not thread-safe/re-entrant)");
	terminator = t;
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetwork* glades::NNetwork::cloneForTrial() const
{
	NNetwork* net = new NNetwork(skeleton, netType);
	net->trainingConfig = trainingConfig;
	net->terminator = terminator;
	net->setSeed(rngSeed + 1);
	return net;
}

float glades::NNetwork::computeLearningRateMultiplier(int epochFromStart) const
{
	if (trainingConfig.lrSchedule.type == LearningRateScheduleConfig::BAYESIAN)
		return bayesianLRMultiplier_;
	return trainingConfig.lrSchedule.multiplier(epochFromStart);
}

void glades::NNetwork::resetGraphs()
{
	// create the results again
	results.clear();
}

// === GPU state lifecycle ===

bool glades::NNetwork::ensureGpuState()
{
#ifdef GLADES_HAVE_CUDA
	if (!trainingConfig.gpu.enable)
		return false;

	if (gpuStateReady)
		return true;

	// Initialize the CUDA device (idempotent).
	if (!gpu::initDevice(trainingConfig.gpu.deviceId))
		return false;

	// Initialize cuBLAS.
	if (!gpu::blasInit())
		return false;

	// Allocate GPU state based on network type.
	const bool isTransformer = (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER);

	if (isTransformer && tensorTransformer.initialized)
	{
		const TensorTransformerState& ts = tensorTransformer;
		const unsigned int dHead = ts.dModel / ts.nHeads;
		const unsigned int dModelKV = ts.nKVHeads * dHead;
		const unsigned int ff1Width = (ts.ffnKind == 1) ? (2u * ts.dFF) : ts.dFF;


		// Allocate weights
		if (!gpuTransformerWeights)
			gpuTransformerWeights = new gpu::GpuTransformerWeights();

		const bool needAdamMoments = atlas_transformer_needs_adam_moments(trainingConfig);
		const bool skipAdam = !needAdamMoments;
		const bool needGpuReallocate =
		    (!gpuTransformerWeights->initialized)
		    || (needAdamMoments && !gpu_transformer_has_adam_moments(*gpuTransformerWeights));
		if (needGpuReallocate)
		{
			if (gpuTransformerWeights->initialized)
				gpuTransformerWeights->free();
			if (!gpuTransformerWeights->allocate(ts.dModel, ts.dFF, ts.nHeads, ts.nKVHeads,
			                                      ts.nLayers, ts.vocabSize, ts.inputSize,
			                                      ts.outSize, ts.ffnKind, ts.tokenModel,
			                                      ts.tieEmbeddings, skipAdam))
			{
				return false;
			}
		}

		// Upload weights
		gpu::uploadTransformerWeights(*gpuTransformerWeights,
		                              ts.tokE.empty() ? NULL : &ts.tokE[0], ts.tokE.size(),
		                              ts.WIn.empty() ? NULL : &ts.WIn[0], ts.WIn.size(),
		                              ts.bIn.empty() ? NULL : &ts.bIn[0], ts.bIn.size(),
		                              ts.WOut.empty() ? NULL : &ts.WOut[0], ts.WOut.size(),
		                              ts.bOut.empty() ? NULL : &ts.bOut[0], ts.bOut.size(),
		                              ts.lmBias.empty() ? NULL : &ts.lmBias[0], ts.lmBias.size(),
		                              ts.lnFinalGamma.empty() ? NULL : &ts.lnFinalGamma[0], ts.lnFinalGamma.size(),
		                              ts.lnFinalBeta.empty() ? NULL : &ts.lnFinalBeta[0], ts.lnFinalBeta.size());

		// Upload per-block weights
		for (unsigned int l = 0; l < ts.nLayers; ++l)
		{
			const TensorTransformerState::Block& cb = ts.blocks[l];
			gpu::uploadTransformerBlockWeights(gpuTransformerWeights->blocks[l],
			                                   ts.dModel, dModelKV, ff1Width, ts.dFF,
			                                   cb.ln1Gamma.empty() ? NULL : &cb.ln1Gamma[0],
			                                   cb.ln1Beta.empty() ? NULL : &cb.ln1Beta[0],
			                                   cb.Wq.empty() ? NULL : &cb.Wq[0],
			                                   cb.Wk.empty() ? NULL : &cb.Wk[0],
			                                   cb.Wv.empty() ? NULL : &cb.Wv[0],
			                                   cb.Wo.empty() ? NULL : &cb.Wo[0],
			                                   cb.bq.empty() ? NULL : &cb.bq[0],
			                                   cb.bk.empty() ? NULL : &cb.bk[0],
			                                   cb.bv.empty() ? NULL : &cb.bv[0],
			                                   cb.bo.empty() ? NULL : &cb.bo[0],
			                                   cb.ln2Gamma.empty() ? NULL : &cb.ln2Gamma[0],
			                                   cb.ln2Beta.empty() ? NULL : &cb.ln2Beta[0],
			                                   cb.W1.empty() ? NULL : &cb.W1[0],
			                                   cb.W2.empty() ? NULL : &cb.W2[0],
			                                   cb.b1.empty() ? NULL : &cb.b1[0],
			                                   cb.b2.empty() ? NULL : &cb.b2[0]);
		}

		// Upload optimizer state (Adam m1/m2) for each weight tensor
		for (unsigned int l = 0; l < ts.nLayers; ++l)
		{
			const TensorTransformerState::Block& cb = ts.blocks[l];
			gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[l];

			// Upload optimizer state for LN, QKV, FFN
			if (!cb.mLn1Gamma.empty()) gb.mLn1Gamma.upload(&cb.mLn1Gamma[0], cb.mLn1Gamma.size());
			if (!cb.v2Ln1Gamma.empty()) gb.v2Ln1Gamma.upload(&cb.v2Ln1Gamma[0], cb.v2Ln1Gamma.size());
			if (!cb.mLn1Beta.empty()) gb.mLn1Beta.upload(&cb.mLn1Beta[0], cb.mLn1Beta.size());
			if (!cb.v2Ln1Beta.empty()) gb.v2Ln1Beta.upload(&cb.v2Ln1Beta[0], cb.v2Ln1Beta.size());

			if (!cb.vWq.empty()) gb.vWq.upload(&cb.vWq[0], cb.vWq.size());
			if (!cb.v2Wq.empty()) gb.v2Wq.upload(&cb.v2Wq[0], cb.v2Wq.size());
			if (!cb.vWk.empty()) gb.vWk.upload(&cb.vWk[0], cb.vWk.size());
			if (!cb.v2Wk.empty()) gb.v2Wk.upload(&cb.v2Wk[0], cb.v2Wk.size());
			if (!cb.vWv.empty()) gb.vWv.upload(&cb.vWv[0], cb.vWv.size());
			if (!cb.v2Wv.empty()) gb.v2Wv.upload(&cb.v2Wv[0], cb.v2Wv.size());
			if (!cb.vWo.empty()) gb.vWo.upload(&cb.vWo[0], cb.vWo.size());
			if (!cb.v2Wo.empty()) gb.v2Wo.upload(&cb.v2Wo[0], cb.v2Wo.size());

			if (!cb.mBq.empty()) gb.mBq.upload(&cb.mBq[0], cb.mBq.size());
			if (!cb.v2Bq.empty()) gb.v2Bq.upload(&cb.v2Bq[0], cb.v2Bq.size());
			if (!cb.mBk.empty()) gb.mBk.upload(&cb.mBk[0], cb.mBk.size());
			if (!cb.v2Bk.empty()) gb.v2Bk.upload(&cb.v2Bk[0], cb.v2Bk.size());
			if (!cb.mBv.empty()) gb.mBv.upload(&cb.mBv[0], cb.mBv.size());
			if (!cb.v2Bv.empty()) gb.v2Bv.upload(&cb.v2Bv[0], cb.v2Bv.size());
			if (!cb.mBo.empty()) gb.mBo.upload(&cb.mBo[0], cb.mBo.size());
			if (!cb.v2Bo.empty()) gb.v2Bo.upload(&cb.v2Bo[0], cb.v2Bo.size());

			if (!cb.mLn2Gamma.empty()) gb.mLn2Gamma.upload(&cb.mLn2Gamma[0], cb.mLn2Gamma.size());
			if (!cb.v2Ln2Gamma.empty()) gb.v2Ln2Gamma.upload(&cb.v2Ln2Gamma[0], cb.v2Ln2Gamma.size());
			if (!cb.mLn2Beta.empty()) gb.mLn2Beta.upload(&cb.mLn2Beta[0], cb.mLn2Beta.size());
			if (!cb.v2Ln2Beta.empty()) gb.v2Ln2Beta.upload(&cb.v2Ln2Beta[0], cb.v2Ln2Beta.size());

			if (!cb.vW1.empty()) gb.vW1.upload(&cb.vW1[0], cb.vW1.size());
			if (!cb.v2W1.empty()) gb.v2W1.upload(&cb.v2W1[0], cb.v2W1.size());
			if (!cb.vW2.empty()) gb.vW2.upload(&cb.vW2[0], cb.vW2.size());
			if (!cb.v2W2.empty()) gb.v2W2.upload(&cb.v2W2[0], cb.v2W2.size());
			if (!cb.mB1.empty()) gb.mB1.upload(&cb.mB1[0], cb.mB1.size());
			if (!cb.v2B1.empty()) gb.v2B1.upload(&cb.v2B1[0], cb.v2B1.size());
			if (!cb.mB2.empty()) gb.mB2.upload(&cb.mB2[0], cb.mB2.size());
			if (!cb.v2B2.empty()) gb.v2B2.upload(&cb.v2B2[0], cb.v2B2.size());
		}

		// Upload global optimizer state
		if (ts.tokenModel)
		{
			if (!ts.vTokE.empty()) gpuTransformerWeights->vTokE.upload(&ts.vTokE[0], ts.vTokE.size());
			if (!ts.v2TokE.empty()) gpuTransformerWeights->v2TokE.upload(&ts.v2TokE[0], ts.v2TokE.size());
			if (!ts.mLmBias.empty()) gpuTransformerWeights->mLmBias.upload(&ts.mLmBias[0], ts.mLmBias.size());
			if (!ts.v2LmBias.empty()) gpuTransformerWeights->v2LmBias.upload(&ts.v2LmBias[0], ts.v2LmBias.size());
		}
		else
		{
			if (!ts.vWIn.empty()) gpuTransformerWeights->vWIn.upload(&ts.vWIn[0], ts.vWIn.size());
			if (!ts.v2WIn.empty()) gpuTransformerWeights->v2WIn.upload(&ts.v2WIn[0], ts.v2WIn.size());
			if (!ts.mBIn.empty()) gpuTransformerWeights->mBIn.upload(&ts.mBIn[0], ts.mBIn.size());
			if (!ts.v2BIn.empty()) gpuTransformerWeights->v2BIn.upload(&ts.v2BIn[0], ts.v2BIn.size());
			if (!ts.vWOut.empty()) gpuTransformerWeights->vWOut.upload(&ts.vWOut[0], ts.vWOut.size());
			if (!ts.v2WOut.empty()) gpuTransformerWeights->v2WOut.upload(&ts.v2WOut[0], ts.v2WOut.size());
			if (!ts.mBOut.empty()) gpuTransformerWeights->mBOut.upload(&ts.mBOut[0], ts.mBOut.size());
			if (!ts.v2BOut.empty()) gpuTransformerWeights->v2BOut.upload(&ts.v2BOut[0], ts.v2BOut.size());
		}

		// Final LayerNorm optimizer state (always present)
		if (!ts.mLnFinalGamma.empty()) gpuTransformerWeights->mLnFinalGamma.upload(&ts.mLnFinalGamma[0], ts.mLnFinalGamma.size());
		if (!ts.v2LnFinalGamma.empty()) gpuTransformerWeights->v2LnFinalGamma.upload(&ts.v2LnFinalGamma[0], ts.v2LnFinalGamma.size());
		if (!ts.mLnFinalBeta.empty()) gpuTransformerWeights->mLnFinalBeta.upload(&ts.mLnFinalBeta[0], ts.mLnFinalBeta.size());
		if (!ts.v2LnFinalBeta.empty()) gpuTransformerWeights->v2LnFinalBeta.upload(&ts.v2LnFinalBeta[0], ts.v2LnFinalBeta.size());
	}

	gpuStateReady = true;
	return true;
#else
	(void)0;
	return false;
#endif
}

void glades::NNetwork::freeGpuState()
{
#ifdef GLADES_HAVE_CUDA
	if (gpuTransformerWeights)
	{
		delete gpuTransformerWeights;
		gpuTransformerWeights = NULL;
	}
	if (gpuTransformerScratch)
	{
		delete gpuTransformerScratch;
		gpuTransformerScratch = NULL;
	}
	if (gpuDffWeights)
	{
		delete gpuDffWeights;
		gpuDffWeights = NULL;
	}
	if (gpuDffScratch)
	{
		delete gpuDffScratch;
		gpuDffScratch = NULL;
	}
	if (gpuRnnWeights)
	{
		delete gpuRnnWeights;
		gpuRnnWeights = NULL;
	}
	if (gpuGruWeights)
	{
		delete gpuGruWeights;
		gpuGruWeights = NULL;
	}
	if (gpuLstmWeights)
	{
		delete gpuLstmWeights;
		gpuLstmWeights = NULL;
	}
	if (gpuCnnWeights)
	{
		delete gpuCnnWeights;
		gpuCnnWeights = NULL;
	}
	if (gpuCnnScratch)
	{
		delete gpuCnnScratch;
		gpuCnnScratch = NULL;
	}
#endif
	gpuStateReady = false;
}
