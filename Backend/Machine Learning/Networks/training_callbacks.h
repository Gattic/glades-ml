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
// Training callbacks for the neural network engine.
//
// Goal: keep the training core side-effect free (no printf, no GUI/network sends).
// Observability and visualization is provided by pluggable callbacks.
//
#ifndef _GLADES_TRAINING_CALLBACKS_H_
#define _GLADES_TRAINING_CALLBACKS_H_

namespace glades {

class NNetwork;

struct NNetworkEpochMetrics
{
	// NOTE: This codebase compiles under C++98, so we avoid C++11 in-class initializers.
	NNetworkEpochMetrics()
	    : runType(0),
	      outputType(0),
	      startingEpoch(0),
	      epoch(0),
	      totalError(0.0f),
	      perplexity(0.0f),
	      totalAccuracy(0.0f),
	      regMAE(0.0f),
	      regRMSE(0.0f),
	      classAccuracy(0.0f),
	      classPrecision(0.0f),
	      classRecall(0.0f),
	      classSpecificity(0.0f),
	      classF1(0.0f),
	      classMCC(0.0f),
	      learningRate(0.0f),
	      lrMultiplier(1.0f),
	      gradNorm(0.0f),
	      gradNormScale(1.0f)
	{
	}

	int runType;    // NNetwork::RUN_*
	int outputType; // GMath::*

	int startingEpoch;
	int epoch;

	float totalError;
	// Token LM extra (valid only when training a token language model):
	// - totalError should be mean NLL per non-pad token (natural log)
	// - perplexity = exp(totalError)
	float perplexity;
	// Meaning depends on outputType:
	// - REGRESSION: totalError = MSE, totalAccuracy = R^2 * 100
	// - CLASSIFICATION/KL: totalError = mean cross-entropy / KL per sample, totalAccuracy = overall (macro) accuracy * 100
	float totalAccuracy;

	// Regression-only extras (valid only when outputType indicates regression)
	float regMAE;  // mean absolute error over all samples/outputs
	float regRMSE; // root mean squared error over all samples/outputs

	// Classification/KL-style metrics (valid only when outputType indicates classification/KL)
	float classAccuracy;
	float classPrecision;
	float classRecall;
	float classSpecificity;
	float classF1;
	float classMCC;

	// Training loop metadata (best-effort; may be 0 for some net types / modes)
	float learningRate;   // effective LR for the output transition this epoch
	float lrMultiplier;   // schedule multiplier applied to base LR(s)
	float gradNorm;       // global L2 norm of gradients on last applied update
	float gradNormScale;  // scaling factor applied by global grad-norm clipping
};

class ITrainingCallbacks
{
public:
	virtual ~ITrainingCallbacks() {}

	// Called before the first epoch is executed.
	virtual void onRunStart(const NNetwork& /*net*/, int /*runType*/) {}

	// Called after each epoch completes. Return true to request early stop.
	virtual bool onEpochEnd(const NNetwork& /*net*/, const NNetworkEpochMetrics& /*metrics*/) { return false; }

	// Called once after the run terminates (either naturally or early-stop).
	virtual void onRunEnd(const NNetwork& /*net*/, int /*runType*/) {}
};

} // namespace glades

#endif

