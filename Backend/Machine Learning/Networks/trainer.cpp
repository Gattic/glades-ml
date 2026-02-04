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
#include "../GMath/gmath.h"
#include "../DataObjects/DataInput.h"

#include <cmath>
#include <string>
#include <time.h>

using namespace glades;

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
			// LayerBuilder needs access to DataInput only to materialize input rows on demand.
			net.meat.attachDataInput(di);
		}
		~RunDataAttachmentGuard()
		{
			net.di = NULL;
			net.meat.detachDataInput();
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

	// Install per-network RNG for the duration of this run.
	// All legacy glades::rng::* call sites (weight init, dropout, etc.) will now draw
	// from this network's engine, eliminating cross-network RNG interference.
	glades::rng::ScopedEngine rngGuard(&net.rngEngine);

	RunDataAttachmentGuard dataGuard(net, newDataInput);
	const bool isTrainRun = (runType == glades::NNetwork::RUN_TRAIN);
	const bool isEvalRun = (runType == glades::NNetwork::RUN_TEST) || (runType == glades::NNetwork::RUN_VALIDATE);
	if (!isTrainRun && !isEvalRun)
	{
		net.running = false;
		net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "Trainer::run: unknown runType");
		return net.lastStatus;
	}

	const unsigned int dataSize = isTrainRun ? net.di->getTrainSize() : net.di->getTestSize();
	if ((dataSize <= 0u) || (net.di->getFeatureCount() <= 0))
	{
		net.running = false;
		net.lastStatus = NNetworkStatus(
		    NNetworkStatus::EMPTY_DATA,
		    isTrainRun ? "Trainer::run: empty training data or feature count is zero"
		               : "Trainer::run: empty test data or feature count is zero");
		return net.lastStatus;
	}

	// === Core invariants (fail fast, with explicit status) ===
	{
		const unsigned int featureCount = net.di->getFeatureCount();
		const unsigned int outSize = net.skeleton->getOutputLayerSize();

		if (outSize == 0)
		{
			net.running = false;
			net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "Trainer::run: output layer size is zero");
			return net.lastStatus;
		}

		// Recurrent paths require at least one hidden layer (state lives there).
		if ((net.netType == glades::NNetwork::TYPE_RNN || net.netType == glades::NNetwork::TYPE_GRU || net.netType == glades::NNetwork::TYPE_LSTM) &&
		    net.skeleton->numHiddenLayers() <= 0)
		{
			net.running = false;
			net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "Trainer::run: recurrent net type requires at least one hidden layer");
			return net.lastStatus;
		}

		// Recurrent paths require a valid sequence model (even if it is just the default single sequence).
		if (net.netType == glades::NNetwork::TYPE_RNN || net.netType == glades::NNetwork::TYPE_GRU || net.netType == glades::NNetwork::TYPE_LSTM)
		{
			std::string seqErr;
			if (isTrainRun)
			{
				if (!net.di->validateTrainSequences(&seqErr))
				{
					net.running = false;
					net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
					                                std::string("Trainer::run: invalid train sequences: ") + seqErr);
					return net.lastStatus;
				}
				if (net.di->getTrainSequenceCount() == 0)
				{
					net.running = false;
					net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
					                                "Trainer::run: recurrent net type requires at least one non-empty train sequence");
					return net.lastStatus;
				}
			}
			else
			{
				if (!net.di->validateTestSequences(&seqErr))
				{
					net.running = false;
					net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
					                                std::string("Trainer::run: invalid test sequences: ") + seqErr);
					return net.lastStatus;
				}
				if (net.di->getTestSequenceCount() == 0)
				{
					net.running = false;
					net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
					                                "Trainer::run: recurrent net type requires at least one non-empty test sequence");
					return net.lastStatus;
				}
			}
		}

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
			if (!net.di->validateTrainRowShapes(featureCount, outSize, &shapeErr))
			{
				net.running = false;
				net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
				                                std::string("Trainer::run: invalid training row shapes: ") + shapeErr);
				return net.lastStatus;
			}
		}
		else
		{
			if (!net.di->validateTestRowShapes(featureCount, outSize, &shapeErr))
			{
				net.running = false;
				net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
				                                std::string("Trainer::run: invalid test row shapes: ") + shapeErr);
				return net.lastStatus;
			}
		}
	}

	int starting_epochs = 0;
	// Get the input, expected, and layers/nodes/edges
	if (net.epochs == 0 && net.mustBuildMeat)
	{
		const bool ok = net.meat.build(net.skeleton, net.di, net.netType);
		if (!ok)
		{
			net.running = false;
			const std::string detail = net.meat.getLastError();
			if (!detail.empty())
				net.lastStatus = NNetworkStatus(NNetworkStatus::BUILD_FAILED, std::string("Trainer::run: failed to build network layers/weights: ") + detail);
			else
				net.lastStatus = NNetworkStatus(NNetworkStatus::BUILD_FAILED, "Trainer::run: failed to build network layers/weights (LayerBuilder::build returned false)");
			return net.lastStatus;
		}
	}

	if (net.changeInputLayers)
	{
		net.meat.rebuildInputLayers(net.skeleton, net.di);
		starting_epochs = net.epochs;
	}

	// Clean confusion matrix
	if ((net.skeleton->getOutputType() == glades::GMath::CLASSIFICATION) ||
	    (net.skeleton->getOutputType() == glades::GMath::KL))
		net.confusionMatrix.clean();

	// Set the mini batch size
	{
		const int cfg = net.trainingConfig.minibatchSizeOverride;
		net.minibatchSize = (cfg > 0) ? cfg : net.skeleton->getBatchSize();
	}

	// Valid layers?
	if ((net.meat.getLayersSize() <= 0))
	{
		net.running = false;
		net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "Trainer::run: network has no valid layers (LayerBuilder not built or invalid input)");
		return net.lastStatus;
	}

	// Post-build shape sanity checks.
	if (net.meat.getLayerSize(0) != net.di->getFeatureCount())
	{
		net.running = false;
		net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "Trainer::run: built input layer size does not match DataInput feature count");
		return net.lastStatus;
	}
	if (net.meat.getLayerSize(net.meat.getLayersSize()) != net.skeleton->getOutputLayerSize())
	{
		net.running = false;
		net.lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "Trainer::run: built output layer size does not match skeleton output layer size");
		return net.lastStatus;
	}

	// Re-check the active split is non-empty post-build (build can succeed for featureCount-only cases).
	if ((dataSize <= 0u) || (net.di->getFeatureCount() <= 0))
	{
		net.running = false;
		net.lastStatus = NNetworkStatus(
		    NNetworkStatus::EMPTY_DATA,
		    isTrainRun ? "Trainer::run: empty training data or feature count is zero (post-build check)"
		               : "Trainer::run: empty test data or feature count is zero (post-build check)");
		return net.lastStatus;
	}

	// Build empty confusion matrix
	if ((net.skeleton->getOutputType() == glades::GMath::CLASSIFICATION) ||
	    (net.skeleton->getOutputType() == glades::GMath::KL))
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
			lrMult = net.computeLearningRateMultiplier(net.epochs - starting_epochs);
			net.lrScheduleMultiplier = lrMult;
		}

		// Global network statistics
		net.overallTotalError = 0.0f;
		net.overallTotalAccuracy = 0.0f;
		net.regSSE = 0.0;
		net.regSAE = 0.0;
		net.regSumY = 0.0;
		net.regSumY2 = 0.0;
		net.regCount = 0ULL;
		net.clsCorrect = 0ULL;
		net.clsTotal = 0ULL;

		// Reset confusion matrix
		if ((net.skeleton->getOutputType() == glades::GMath::CLASSIFICATION) ||
		    (net.skeleton->getOutputType() == glades::GMath::KL))
			net.confusionMatrix.reset();

		// Recurrent: reset hidden-state at the start of each epoch/run iteration.
		if (net.netType == glades::NNetwork::TYPE_RNN || net.netType == glades::NNetwork::TYPE_GRU || net.netType == glades::NNetwork::TYPE_LSTM)
			net.meat.resetContextState(0.0f);

		// Forward/backprop/update
		const bool isRecurrent =
		    (net.netType == glades::NNetwork::TYPE_RNN) ||
		    (net.netType == glades::NNetwork::TYPE_GRU) ||
		    (net.netType == glades::NNetwork::TYPE_LSTM);

		// DFF paths step over individual rows. Recurrent paths run as a single "step" per epoch
		// (the helper iterates sequences/timesteps internally).
		const unsigned int steps = isRecurrent ? 1u : (isTrainRun ? net.di->getTrainSize() : net.di->getTestSize());
		for (unsigned int step = 0; step < steps; ++step)
		{
			const unsigned int r = isRecurrent ? 0u : step;
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
		if ((outType == glades::GMath::CLASSIFICATION) || (outType == glades::GMath::KL))
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
		metrics.classMCC = net.confusionMatrix.getOverallMCC();
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

		bool callbackStop = false;
		if (cb)
			callbackStop = cb->onEpochEnd(net, metrics);

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
	net.changeInputLayers = false;

	net.lastStatus = NNetworkStatus(NNetworkStatus::OK, std::string());
	return net.lastStatus;
}

