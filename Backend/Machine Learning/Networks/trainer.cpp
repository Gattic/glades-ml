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
//
#include "trainer.h"

#include "network.h"
#include "ddp_comm.h"
#include "../GMath/gmath.h"
#include "../DataObjects/DataInput.h"

#include <cmath>
#include <string>
#include <time.h>

using namespace glades;

namespace {

static bool is_transformer_token_lm_type(int netType)
{
	return (netType == glades::NNetwork::TYPE_TRANSFORMER_DECODER) ||
	       (netType == glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
}

static bool is_sequence_model_type(int netType)
{
	return (netType == glades::NNetwork::TYPE_RNN) ||
	       (netType == glades::NNetwork::TYPE_GRU) ||
	       (netType == glades::NNetwork::TYPE_LSTM) ||
	       (netType == glades::NNetwork::TYPE_TRANSFORMER_ENCODER) ||
	       (netType == glades::NNetwork::TYPE_TRANSFORMER_DECODER);
}

struct TrainerRunPreflight
{
	TrainerRunPreflight()
	    : dataSize(0u),
	      featureCount(0u),
	      outputSize(0u),
	      expectedFeatureCount(0u),
	      expectedOutputSize(0u),
	      tokenLM(false),
	      tokenLMInput(false),
	      useConfusionMatrix(false),
	      isSequenceModel(false)
	{
	}

	unsigned int dataSize;
	unsigned int featureCount;
	unsigned int outputSize;
	unsigned int expectedFeatureCount;
	unsigned int expectedOutputSize;
	bool tokenLM;
	bool tokenLMInput;
	bool useConfusionMatrix;
	bool isSequenceModel;
};

static glades::NNetworkStatus build_empty_data_status(const glades::DataInput& di,
                                                      bool isTrainRun,
                                                      bool postBuildCheck)
{
	std::string extra;
	{
		const glades::NNetworkStatus diSt = di.getLastStatus();
		if (!diSt.ok() && !diSt.message.empty())
			extra = std::string(" (DataInput: ") + diSt.message + ")";
	}

	glades::NNetworkStatus st(
	    glades::NNetworkStatus::EMPTY_DATA,
	    isTrainRun
	        ? (postBuildCheck ? "Trainer::run: empty training data or feature count is zero (post-build check)"
	                          : "Trainer::run: empty training data or feature count is zero")
	        : (postBuildCheck ? "Trainer::run: empty test data or feature count is zero (post-build check)"
	                          : "Trainer::run: empty test data or feature count is zero"));
	if (!extra.empty())
		st.message += extra;
	return st;
}

static glades::NNetworkStatus validate_active_split_data(const glades::DataInput& di,
                                                         bool isTrainRun,
                                                         unsigned int dataSize,
                                                         bool tokenLMInput,
                                                         bool postBuildCheck)
{
	if ((dataSize > 0u) && (tokenLMInput || (di.getFeatureCount() > 0u)))
		return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
	return build_empty_data_status(di, isTrainRun, postBuildCheck);
}

static glades::NNetworkStatus validate_sequence_contracts(const glades::DataInput& di,
                                                          const glades::NNInfo& skeleton,
                                                          bool isTrainRun,
                                                          bool isSequenceModel)
{
	if (!isSequenceModel)
		return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());

	// Recurrent paths require at least one hidden layer (state lives there).
	if (skeleton.numHiddenLayers() <= 0)
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT,
		                              "Trainer::run: sequence net type requires at least one hidden layer");
	}

	// Recurrent paths require a valid sequence model (even if it is just the default single sequence).
	std::string seqErr;
	if (isTrainRun)
	{
		if (!di.validateTrainSequences(&seqErr))
		{
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT,
			                              std::string("Trainer::run: invalid train sequences: ") + seqErr);
		}
		if (di.getTrainSequenceCount() == 0)
		{
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT,
			                              "Trainer::run: sequence net type requires at least one non-empty train sequence");
		}
	}
	else
	{
		if (!di.validateTestSequences(&seqErr))
		{
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT,
			                              std::string("Trainer::run: invalid test sequences: ") + seqErr);
		}
		if (di.getTestSequenceCount() == 0)
		{
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT,
			                              "Trainer::run: sequence net type requires at least one non-empty test sequence");
		}
	}

	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

static glades::NNetworkStatus validate_row_shape_contracts(const glades::DataInput& di,
                                                           bool isTrainRun,
                                                           unsigned int expectedFeatureCount,
                                                           unsigned int expectedOutputSize)
{
	// Validate that every row has the expected dimensionality.
	// IMPORTANT:
	// Do NOT materialize every row to validate shapes. For streaming inputs (e.g. ImageInput),
	// getTrainRow() can decode images from disk; scanning the whole dataset is prohibitive.
	//
	// Instead, use DataInput's shape contract. Implementations with fixed shapes validate
	// in O(1); others do a bounded spot-check.
	std::string shapeErr;
	if (isTrainRun)
	{
		if (!di.validateTrainRowShapes(expectedFeatureCount, expectedOutputSize, &shapeErr))
		{
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT,
			                              std::string("Trainer::run: invalid training row shapes: ") + shapeErr);
		}
	}
	else
	{
		if (!di.validateTestRowShapes(expectedFeatureCount, expectedOutputSize, &shapeErr))
		{
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT,
			                              std::string("Trainer::run: invalid test row shapes: ") + shapeErr);
		}
	}

	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

static glades::NNetworkStatus build_run_preflight(const glades::DataInput& di,
                                                  const glades::NNInfo& skeleton,
                                                  int netType,
                                                  const glades::TrainingConfig& trainingConfig,
                                                  bool isTrainRun,
                                                  TrainerRunPreflight& out)
{
	out.dataSize = isTrainRun ? di.getTrainSize() : di.getTestSize();
	out.featureCount = di.getFeatureCount();
	out.outputSize = skeleton.getOutputLayerSize();
	out.tokenLM = is_transformer_token_lm_type(netType) && trainingConfig.transformer.enableTokenEmbedding;
	out.tokenLMInput = out.tokenLM && di.hasTokenIdInput();
	out.useConfusionMatrix =
	    !out.tokenLM && ((skeleton.getOutputType() == glades::GMath::CLASSIFICATION) ||
	                     (skeleton.getOutputType() == glades::GMath::KL));
	out.isSequenceModel = is_sequence_model_type(netType);

	glades::NNetworkStatus st = validate_active_split_data(di, isTrainRun, out.dataSize, out.tokenLMInput, false);
	if (!st.ok())
		return st;

	if (out.outputSize == 0u)
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "Trainer::run: output layer size is zero");

	st = validate_sequence_contracts(di, skeleton, isTrainRun, out.isSequenceModel);
	if (!st.ok())
		return st;

	out.expectedFeatureCount = out.tokenLMInput ? 0u : (out.tokenLM ? 1u : out.featureCount);
	// Token LM expected rows are a single token id, not a dense one-hot of size outputSize.
	out.expectedOutputSize = out.tokenLM ? 1u : out.outputSize;
	return validate_row_shape_contracts(di, isTrainRun, out.expectedFeatureCount, out.expectedOutputSize);
}

} // namespace

glades::NNetworkStatus glades::Trainer::run(glades::NNetwork& net,
                                           const glades::DataInput* newDataInput,
                                           int runType,
                                           glades::ITrainingCallbacks* cb)
{
	// NOTE: This is the extracted training/inference driver from the ML engine.
	// It MUST NOT instantiate any default callbacks or emit side effects.

	// === Concurrency policy enforcement ===
	//
	// NNetwork is not re-entrant: do not run train/test concurrently on the same instance.
	// This guard prevents concurrent mutation of shared tensor state, scratch buffers, metrics,
	// and the per-network RNG engine.
	glades::NNetwork::RunLockGuard runGuard(net);
	if (!runGuard.ok())
	{
		// Do not mutate `net` here: if another thread is running it, touching fields would be a data race.
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
		                             "Trainer::run: NNetwork is already running (not re-entrant/thread-safe). "
		                             "Create separate NNetwork instances per thread.");
	}

	// Lifetime safety:
	// DataInput is owned by the caller and may be deleted immediately after this function returns.
	// The model must not retain a pointer to it beyond the scope of this call.
	struct RunDataAttachmentGuard
	{
		glades::NNetwork& net;
		explicit RunDataAttachmentGuard(glades::NNetwork& n, const glades::DataInput* di) : net(n)
		{
			net.di = di;
		}
		~RunDataAttachmentGuard()
		{
			net.di = NULL;
		}
	private:
		RunDataAttachmentGuard(const RunDataAttachmentGuard&);
		RunDataAttachmentGuard& operator=(const RunDataAttachmentGuard&);
	};

	if (!net.skeleton)
	{
		net.running = false;
		net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "Trainer::run: skeleton is NULL (load/build a network first)");
		return net.lastStatus;
	}

	if (!newDataInput)
	{
		net.running = false;
		net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "Trainer::run: DataInput is NULL");
		return net.lastStatus;
	}

	// RNG determinism:
	// Training/inference code should draw randomness explicitly from `net.rngEngine`
	// (no implicit global/TLS "current engine").

	RunDataAttachmentGuard dataGuard(net, newDataInput);
	const bool isTrainRun = (runType == glades::NNetwork::RUN_TRAIN);
	const bool isEvalRun = (runType == glades::NNetwork::RUN_TEST) || (runType == glades::NNetwork::RUN_VALIDATE);
	if (!isTrainRun && !isEvalRun)
	{
		net.running = false;
		net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "Trainer::run: unknown runType");
		return net.lastStatus;
	}

	TrainerRunPreflight preflight;
	net.lastStatus = build_run_preflight(*net.di, *net.skeleton, net.netType, net.trainingConfig, isTrainRun, preflight);
	if (!net.lastStatus.ok())
	{
		net.running = false;
		return net.lastStatus;
	}

	// For learning-rate schedules, treat this run's start epoch as the baseline.
	// This makes schedules work sensibly for resumed training.
	int starting_epochs = isTrainRun ? net.epochs : 0;

	// Ensure weights/parameters exist for this shape before any SGD steps run.
	if (!net.ensureTensorParametersInitialized())
	{
		net.running = false;
		if (net.lastStatus.ok())
			net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "Trainer::run: failed to initialize tensor parameters");
		return net.lastStatus;
	}

	// Clean confusion matrix
	const bool tokenLM = preflight.tokenLM;
	const bool useConfusionMatrix = preflight.useConfusionMatrix;
	if (useConfusionMatrix)
		net.confusionMatrix.clean();

	// Set the mini batch size
	{
		const int cfg = net.trainingConfig.minibatchSizeOverride;
		net.minibatchSize = (cfg > 0) ? cfg : net.skeleton->getBatchSize();
	}

	// Re-check the active split is non-empty post-build (build can succeed for featureCount-only cases).
	net.lastStatus = validate_active_split_data(*net.di, isTrainRun, preflight.dataSize, preflight.tokenLMInput, true);
	if (!net.lastStatus.ok())
	{
		net.running = false;
		return net.lastStatus;
	}

	// Build empty confusion matrix
	if (useConfusionMatrix)
		net.confusionMatrix.build(net.skeleton->getOutputLayerSize());

	// Reset graphs (e.g. learning curve) for TRAIN runs only.
	// Evaluation should be side-effect-free w.r.t. training graph history.
	if (isTrainRun)
		net.resetGraphs();

	// arbitrary independent var (time dimension)
	net.running = true;
	net.firstRunActivation = false;

	if (cb)
		cb->onRunStart(net, runType);

	while (net.running)
	{
		// === Learning rate schedule (train only) ===
		// Compute schedule multiplier based on "epoch-from-start" for this run.
		float lrMult = 1.0f;
		if (runType == glades::NNetwork::RUN_TRAIN && net.skeleton)
		{
			lrMult = net.computeLearningRateMultiplier(net.epochs - starting_epochs + net.lrScheduleEpochOffset);
			net.lrScheduleMultiplier = lrMult;
		}

		// Global network statistics
		net.overallTotalError = 0.0f;
		net.overallTotalAccuracy = 0.0f;
		// Always reset classification-derived metrics too. These are validated for finiteness
		// even on regression runs; leaving them uninitialized/stale can abort training.
		net.overallClassAccuracy = 0.0f;
		net.overallClassPrecision = 0.0f;
		net.overallClassRecall = 0.0f;
		net.overallClassSpecificity = 0.0f;
		net.overallClassF1 = 0.0f;
		net.regSSE = 0.0;
		net.regSAE = 0.0;
		net.regSumY = 0.0;
		net.regSumY2 = 0.0;
		net.regCount = 0ULL;
		net.clsCorrect = 0ULL;
		net.clsTotal = 0ULL;

		// Reset confusion matrix
		if (useConfusionMatrix)
			net.confusionMatrix.reset();

		// Forward/backprop/update
		// DFF paths step over individual rows. Recurrent paths run as a single "step" per epoch
		// (the helper iterates sequences/timesteps internally).
		const unsigned int steps = preflight.isSequenceModel ? 1u : preflight.dataSize;
		for (unsigned int step = 0; step < steps; ++step)
		{
			const unsigned int r = preflight.isSequenceModel ? 0u : step;
			const glades::NNetworkStatus stStep = net.SGDHelper(r, runType);
			if (!stStep.ok())
			{
				// Stop immediately on internal failures: continuing would produce silent corruption.
				net.running = false;
				if (cb)
					cb->onRunEnd(net, runType);
				return stStep;
			}
		}

		// === Hard safety checks (fail fast on NaN/Inf) ===
		//
		// If these aggregates go non-finite, continuing would silently corrupt metrics and
		// (in train runs) push non-finite updates into weights.
		if (!std::isfinite(net.overallTotalError) || !std::isfinite(net.overallTotalAccuracy) ||
		    !std::isfinite(net.regSSE) || !std::isfinite(net.regSAE) ||
		    !std::isfinite(net.regSumY) || !std::isfinite(net.regSumY2))
		{
			net.running = false;
			net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                                "Trainer::run: non-finite training aggregates detected (NaN/Inf). Aborting run.");
			if (cb)
				cb->onRunEnd(net, runType);
			return net.lastStatus;
		}

		// Update the epoch counter for TRAIN only.
		// Evaluation runs must be side-effect-free w.r.t. training bookkeeping.
		if (isTrainRun)
			++net.epochs;

		// Finalize metrics that require epoch-level aggregation.
		const int outType = net.skeleton->getOutputType();
		if (outType == glades::GMath::REGRESSION)
		{
			// Loss: mean squared error over all samples and outputs.
			if (net.regCount > 0ULL)
			{
				const double mse = net.regSSE / static_cast<double>(net.regCount);
				net.overallTotalError = static_cast<float>(mse);

				// R^2: 1 - SSE/SST (computed without a second pass).
				const double n = static_cast<double>(net.regCount);
				const double sst = net.regSumY2 - (net.regSumY * net.regSumY) / n;
				double r2 = 0.0;
				if (sst > 1e-12)
					r2 = 1.0 - (net.regSSE / sst);
				else
					r2 = (net.regSSE <= 1e-12) ? 1.0 : 0.0;

				// Present as percent for consistency with existing UI expectations.
				net.overallTotalAccuracy = static_cast<float>(r2 * 100.0);
			}
			else
			{
				net.overallTotalError = 0.0f;
				net.overallTotalAccuracy = 0.0f;
			}
		}
		else if ((outType == glades::GMath::CLASSIFICATION) || (outType == glades::GMath::KL))
		{
			// Top-1 accuracy (%).
			if (net.clsTotal > 0ULL)
				net.overallTotalAccuracy = static_cast<float>((100.0 * static_cast<double>(net.clsCorrect)) / static_cast<double>(net.clsTotal));
			else
				net.overallTotalAccuracy = 0.0f;
		}

		// Update confusion-matrix derived metrics once per epoch.
		if (useConfusionMatrix && ((outType == glades::GMath::CLASSIFICATION) || (outType == glades::GMath::KL)))
		{
			net.confusionMatrix.updateResultParams();
			net.overallClassAccuracy = (net.confusionMatrix.getOverallAccuracy() * 100.0f);
			net.overallClassPrecision = (net.confusionMatrix.getOverallPrecision() * 100.0f);
			net.overallClassRecall = (net.confusionMatrix.getOverallRecall() * 100.0f);
			net.overallClassSpecificity = (net.confusionMatrix.getOverallSpecificity() * 100.0f);
			net.overallClassF1 = net.confusionMatrix.getOverallF1Score() * 100.0f;

			// Historically "overallTotalAccuracy" was used for GUI's single "ACC" label.
			// For classification-style outputs, prefer the confusion-matrix accuracy.
			net.overallTotalAccuracy = net.overallClassAccuracy;
		}
		else if ((outType == glades::GMath::CLASSIFICATION) || (outType == glades::GMath::KL))
		{
			// Token LM (or other non-confusion-matrix classification): use top-1 accuracy accumulated by SGDHelper.
			if (net.clsTotal > 0ULL)
				net.overallTotalAccuracy = static_cast<float>((100.0 * static_cast<double>(net.clsCorrect)) / static_cast<double>(net.clsTotal));
			else
				net.overallTotalAccuracy = 0.0f;
			net.overallClassAccuracy = net.overallTotalAccuracy;
		}

		// Non-finite user-facing metrics are never acceptable.
		if (!std::isfinite(net.overallTotalError) || !std::isfinite(net.overallTotalAccuracy) ||
		    !std::isfinite(net.overallClassAccuracy) || !std::isfinite(net.overallClassPrecision) ||
		    !std::isfinite(net.overallClassRecall) || !std::isfinite(net.overallClassSpecificity) ||
		    !std::isfinite(net.overallClassF1))
		{
			net.running = false;
			net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                                "Trainer::run: non-finite metrics detected (NaN/Inf). Aborting run.");
			if (cb)
				cb->onRunEnd(net, runType);
			return net.lastStatus;
		}

		// Emit epoch metrics (no side effects here).
		glades::NNetworkEpochMetrics metrics;
		metrics.runType = runType;
		metrics.outputType = net.skeleton->getOutputType();
		metrics.startingEpoch = starting_epochs;
		// For evaluation, report the current trained-epoch index (do not advance).
		metrics.epoch = net.epochs;
		metrics.totalError = net.overallTotalError;
		metrics.totalAccuracy = net.overallTotalAccuracy;
		// Token LM: perplexity = exp(mean NLL per token).
		// (Only meaningful when the transformer is running in token LM mode with FULL softmax;
		// sampled-softmax objectives are not exact NLL and must not be reported as perplexity.)
		const bool tokenLMFullSoftmax =
		    tokenLM && (net.trainingConfig.transformer.tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX);
		if (tokenLMFullSoftmax && ((outType == glades::GMath::CLASSIFICATION) || (outType == glades::GMath::KL)))
		{
			double arg = static_cast<double>(metrics.totalError);
			// Avoid overflow in exp(). float overflows around exp(88.7).
			if (arg > 80.0) arg = 80.0;
			if (arg < -80.0) arg = -80.0;
			metrics.perplexity = static_cast<float>(exp(arg));
		}
		// Regression extras
		if (outType == glades::GMath::REGRESSION && net.regCount > 0ULL)
		{
			const double mae = net.regSAE / static_cast<double>(net.regCount);
			const double rmse = sqrt(static_cast<double>(net.overallTotalError));
			metrics.regMAE = static_cast<float>(mae);
			metrics.regRMSE = static_cast<float>(rmse);
		}
		metrics.classAccuracy = net.overallClassAccuracy;
		metrics.classPrecision = net.overallClassPrecision;
		metrics.classRecall = net.overallClassRecall;
		metrics.classSpecificity = net.overallClassSpecificity;
		metrics.classF1 = net.overallClassF1;
		metrics.classMCC = useConfusionMatrix ? net.confusionMatrix.getOverallMCC() : 0.0f;
		// Schedule/gradient metadata:
		// - For TRAIN: report the effective scheduled LR and the last observed grad-norm info.
		// - For EVAL: these must be neutral values (never leak stale training metadata).
		if (isTrainRun)
		{
			metrics.lrMultiplier = net.lrScheduleMultiplier;
			// Report effective output-layer LR (base * multiplier) without mutating the skeleton.
			metrics.learningRate =
			    (net.skeleton ? (net.skeleton->getLearningRate(static_cast<unsigned int>(net.skeleton->numHiddenLayers())) * net.lrScheduleMultiplier) : 0.0f);
			metrics.gradNorm = net.lastGradNorm;
			metrics.gradNormScale = net.lastGradNormScale;
		}
		else
		{
			metrics.lrMultiplier = 1.0f;
			metrics.learningRate = 0.0f;
			metrics.gradNorm = 0.0f;
			metrics.gradNormScale = 1.0f;
		}

		if (!std::isfinite(metrics.learningRate) || !std::isfinite(metrics.lrMultiplier) ||
		    !std::isfinite(metrics.gradNorm) || !std::isfinite(metrics.gradNormScale))
		{
			net.running = false;
			net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                                "Trainer::run: non-finite schedule/gradient metadata detected (NaN/Inf). Aborting run.");
			if (cb)
				cb->onRunEnd(net, runType);
			return net.lastStatus;
		}

		// Bayesian adaptive LR update.
		// Every `windowEpochs` training epochs, record the current (logLR, loss) pair in the
		// inner-loop GP and pick the log-LR with the highest expected improvement.
		if (isTrainRun && net.trainingConfig.lrSchedule.type == LearningRateScheduleConfig::BAYESIAN)
		{
			++net.bayesianLREpochCounter_;
			if (net.bayesianLREpochCounter_ >= net.trainingConfig.bayesianLR.windowEpochs)
			{
				net.bayesianLREpochCounter_ = 0;
				float currentLoss = metrics.totalError;
				float currentLogLR = logf(net.bayesianLRMultiplier_);

				net.bayesianLRGP_.addSample(currentLogLR, currentLoss);
				net.bayesianLRGP_.fit();

				float logMin = logf(net.trainingConfig.bayesianLR.minLR);
				float logMax = logf(net.trainingConfig.bayesianLR.maxLR);
				float bestEI = -1.0f;
				float bestLogLR = currentLogLR;

				if (net.bayesianLRGP_.numSamples() >= 2)
				{
					float bestLoss = currentLoss;
					for (float logLR = logMin; logLR <= logMax; logLR += (logMax - logMin) / 100.0f)
					{
						std::pair<float, float> pred = net.bayesianLRGP_.predict(logLR);
						float mu = pred.first;
						float sigma = sqrtf(pred.second);
						if (sigma < 1e-8f) continue;
						float z = (bestLoss - mu) / sigma;
						float ei = (bestLoss - mu) * BayesianOptimizer::cdf(z) + sigma * BayesianOptimizer::pdf(z);
						if (ei > bestEI) { bestEI = ei; bestLogLR = logLR; }
					}
				}

				net.bayesianLRMultiplier_ = expf(bestLogLR);
				if (net.bayesianLRMultiplier_ < net.trainingConfig.bayesianLR.minLR)
					net.bayesianLRMultiplier_ = net.trainingConfig.bayesianLR.minLR;
				if (net.bayesianLRMultiplier_ > net.trainingConfig.bayesianLR.maxLR)
					net.bayesianLRMultiplier_ = net.trainingConfig.bayesianLR.maxLR;
			}
		}

		bool callbackStop = false;
		if (cb)
			callbackStop = cb->onEpochEnd(net, metrics);

		// DDP: consensus on early-stop so all workers stop together.
		if (net.trainingConfig.ddp.enable && glades::ddp::worldSize() > 1)
		{
			unsigned int stopFlag = callbackStop ? 1u : 0u;
			glades::ddp::allReduceSumInPlace(&stopFlag, 1);
			callbackStop = (stopFlag > 0u);
		}

		net.cNodeActivations.clear();

		// Train-only termination controls.
		if (isTrainRun)
		{
			if (callbackStop || net.terminator.triggered(time(NULL), net.epochs - starting_epochs, net.getAccuracy()))
				break;
		}

		// Evaluation runs are single-pass by design.
		if (!isTrainRun)
			break;
	}

	if (cb)
		cb->onRunEnd(net, runType);

	// So the network doesnt immediately quit next time and we can prematurely start our net
	net.running = false;

	net.lastStatus = NNetworkStatus(NNetworkStatus::OK, std::string());
	return net.lastStatus;
}
