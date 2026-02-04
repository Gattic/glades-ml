// Net-type-specific SGD: DFF tensor path (split out of network.cpp)
#include "network.h"
#include "param_layout.h"
#include "sgd_utils.h"

#include "../DataObjects/DataInput.h"
#include "../GMath/gmath.h"
#include "../State/layer.h"
#include "../State/node.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace glades;

void glades::NNetwork::SGDHelper_DFF(unsigned int inputRowCounter, int runType)
{
	using namespace glades::sgd_detail;
	using namespace glades::param_layout;

	const bool isTrain = (runType == RUN_TRAIN);
	const unsigned int dataSize = isTrain ? (di ? di->getTrainSize() : 0u) : (di ? di->getTestSize() : 0u);
	const float gradClip = trainingConfig.perElementGradClip;

	// === DFF (TYPE_DFF): tensor-based forward/backward + SGD ===
	//
	// This replaces the historical Node/Edge activation bookkeeping with explicit
	// contiguous buffers. LayerBuilder ("meat") remains responsible for:
	// - constructing the network shape and initializing weights
	// - generating dropout masks per-sample
	// - providing a place to persist updated weights/biases for visualization/save
	//
	// The math here intentionally preserves the engine's "per-connection-index" hyperparam
	// semantics (activation/lr/momentum/decay indexed by the *input-side* layer index).

	// Dropout:
	// - Train: scramble per-sample (historical behavior).
	// - Eval: disable dropout (production-correct behavior).
	if (isTrain)
	{
		std::vector<float> pHiddenVec;
		for (int i = 0; i < skeleton->numHiddenLayers(); ++i)
			pHiddenVec.push_back(skeleton->getPDropout(i));
		meat.scrambleDropout(inputRowCounter, skeleton->getPInput(), pHiddenVec);
	}
	else
	{
		meat.clearDropout();
	}

	// Reset per-sample outputs
	results.clear();
	// Evaluation should not destroy any caller-visible training records.
	if (isTrain)
		nbRecord.clear();

	// (Re)initialize tensor buffers if shape changed or first use.
	{
		const unsigned int inSize = meat.getLayerSize(0);
		const int H = skeleton->numHiddenLayers();
		const unsigned int outSize = skeleton->getOutputLayerSize();

		std::vector<unsigned int> wantSizes;
		wantSizes.reserve(static_cast<size_t>(H) + 2);
		wantSizes.push_back(inSize);
		for (int l = 0; l < H; ++l)
			wantSizes.push_back(meat.getLayerSize(static_cast<unsigned int>(l) + 1));
		wantSizes.push_back(outSize);

		const bool mismatch = (!tensorDff.initialized) || (tensorDff.sizes != wantSizes);
		if (mismatch)
		{
			tensorDff.reset();
			tensorDff.sizes = wantSizes;

			const unsigned int numTransitions = (wantSizes.size() >= 2) ? static_cast<unsigned int>(wantSizes.size() - 1) : 0;
			tensorDff.T.resize(numTransitions);
			tensorDff.a.resize(wantSizes.size());
			tensorDff.delta.resize(wantSizes.size());

			for (unsigned int li = 0; li < wantSizes.size(); ++li)
			{
				tensorDff.a[li].assign(wantSizes[li], 0.0f);
				if (li == 0)
					tensorDff.delta[li].clear();
				else
					tensorDff.delta[li].assign(wantSizes[li], 0.0f);
			}

			for (unsigned int t = 0; t < numTransitions; ++t)
			{
				const unsigned int in = wantSizes[t];
				const unsigned int out = wantSizes[t + 1];

				TensorDFFState::Transition& tr = tensorDff.T[t];
				tr.in = in;
				tr.out = out;
				tr.W.assign(static_cast<size_t>(out) * static_cast<size_t>(in), 0.0f);
				tr.vW.assign(static_cast<size_t>(out) * static_cast<size_t>(in), 0.0f);
				tr.gW.assign(static_cast<size_t>(out) * static_cast<size_t>(in), 0.0f);
				tr.bias.assign(out, 0.0f);
				tr.gBias.assign(out, 0.0f);

				Layer* outLayer = meat.getOutputLayer(t + 1);
				if (!outLayer)
					continue;

				for (unsigned int j = 0; j < out; ++j)
				{
					Node* node = meat.getOutputNode(outLayer, j);
					if (!node)
						continue;
					for (unsigned int i = 0; i < in; ++i)
						tr.W[static_cast<size_t>(j) * static_cast<size_t>(in) + i] = node->getEdgeWeight(i);

					// Per-neuron bias is stored as the final edge weight.
					tr.bias[j] = node->getEdgeWeight(dense_bias_edge(in));
				}
			}

			tensorDff.batchCount = 0;
			tensorDff.initialized = true;
		}
	}

	// Forward pass for this sample
	{
		// In eval mode, we do not rely on LayerBuilder's row materialization (which is train-row based).
		Layer* inLayer = isTrain ? meat.getInputLayer(inputRowCounter, 0, glades::LayerBuilder::SPLIT_TRAIN) : NULL;
		const float* xData = NULL;
		unsigned int xSize = 0u;
		if (di)
		{
			if (isTrain)
				di->getTrainRowView(inputRowCounter, xData, xSize);
			else
				di->getTestRowView(inputRowCounter, xData, xSize);
		}
		// Hard safety check: reject non-finite inputs early (sampled for performance on huge rows).
		if (!span_all_finite_bounded(xData, xSize, /*maxChecks*/ 32u))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_DFF: non-finite input data detected (NaN/Inf)");
			running = false;
			return;
		}
		const unsigned int in = tensorDff.sizes.empty() ? 0 : tensorDff.sizes[0];
		for (unsigned int i = 0; i < in; ++i)
		{
			const bool keep = (isTrain ? (inLayer ? inLayer->possiblePath(i) : true) : true);
			const float x = (xData && (i < xSize)) ? xData[i] : 0.0f;
			tensorDff.a[0][i] = keep ? x : 0.0f;
		}

		// We record raw pre-activations for GUI visualization on the last sample only.
		const bool recordActivations = (di && (dataSize > 0) && (inputRowCounter == dataSize - 1));

		const int costFx = skeleton->getOutputType();
		const unsigned int outSize = skeleton->getOutputLayerSize();
		const unsigned int lastTransition = (tensorDff.T.size() > 0) ? static_cast<unsigned int>(tensorDff.T.size() - 1) : 0;
		const bool useSoftmax =
		    ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1);
		std::vector<float> outLogits;
		std::vector<float> outProbs;

		for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
		{
			const TensorDFFState::Transition& tr = tensorDff.T[t];
			Layer* outLayer = meat.getOutputLayer(t + 1);

			const int actFx = skeleton->getActivationType(t);
			const float actParam = skeleton->getActivationParam(t);
			const bool softmaxThisLayer = useSoftmax && (t == lastTransition) && (tr.out == outSize);
			if (softmaxThisLayer)
				outLogits.assign(tr.out, 0.0f);

			for (unsigned int j = 0; j < tr.out; ++j)
			{
				// Node-scoped dropout: dropped nodes have zero activation and do not receive bias.
				if (outLayer && !outLayer->possiblePath(j))
				{
					tensorDff.a[t + 1][j] = 0.0f;
					continue;
				}

				float z = (j < tr.bias.size()) ? tr.bias[j] : 0.0f;
				const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(tr.in);
				for (unsigned int i = 0; i < tr.in; ++i)
					z += tr.W[rowOff + i] * tensorDff.a[t][i];
				if (!is_finite(z))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite pre-activation detected (NaN/Inf)");
					running = false;
					return;
				}

				if (recordActivations)
					cNodeActivations.addFloat(z);

				if (softmaxThisLayer)
					outLogits[j] = z;
				else
				{
					const float a = GMath::squash(z, actFx, actParam);
					if (!is_finite(a))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite activation detected (NaN/Inf)");
						running = false;
						return;
					}
					tensorDff.a[t + 1][j] = a;
				}
			}

			// Apply softmax as a layer-level activation for multi-class classification/KL.
			if (softmaxThisLayer)
			{
				softmax_stable(outLogits, outProbs);
				for (unsigned int j = 0; j < tr.out; ++j)
				{
					const float p = outProbs[j];
					if (!is_finite(p))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite softmax probability detected (NaN/Inf)");
						running = false;
						return;
					}
					tensorDff.a[t + 1][j] = p;
				}
			}
		}

		if (recordActivations)
			cNodeActivations.addString(",");
	}

	// Output layer bookkeeping (results + loss; accuracy is derived later per output type)
	{
		const unsigned int outSize = skeleton->getOutputLayerSize();
		const unsigned int last = (tensorDff.sizes.size() > 0) ? static_cast<unsigned int>(tensorDff.sizes.size() - 1) : 0;
		const float* yData = NULL;
		unsigned int ySize = 0u;
		if (di)
		{
			if (isTrain)
				di->getTrainExpectedRowView(inputRowCounter, yData, ySize);
			else
				di->getTestExpectedRowView(inputRowCounter, yData, ySize);
		}
		if (!span_all_finite_bounded(yData, ySize, /*maxChecks*/ 32u))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_DFF: non-finite expected outputs detected (NaN/Inf)");
			running = false;
			return;
		}
		const int costFx = skeleton->getOutputType();
		const bool useSoftmax =
		    ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1);
		const unsigned int N = dataSize;

		double loss = 0.0;
		for (unsigned int k = 0; k < outSize; ++k)
		{
			const float pred = tensorDff.a[last][k];
			const float expv = (yData && (k < ySize)) ? yData[k] : 0.0f;
			if (!is_finite(expv))
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_DFF: non-finite expected value detected (NaN/Inf)");
				running = false;
				return;
			}
			if (!is_finite(pred))
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite prediction detected (NaN/Inf)");
				running = false;
				return;
			}

			results.addFloat(expv);
			results.addFloat(pred);

			if (costFx == GMath::REGRESSION)
			{
				const double diff = static_cast<double>(pred) - static_cast<double>(expv);
				regSSE += diff * diff;
				regSAE += fabs(diff);
				regSumY += static_cast<double>(expv);
				regSumY2 += static_cast<double>(expv) * static_cast<double>(expv);
				++regCount;
			}
			else if (useSoftmax)
			{
				const float p = clamp_prob01(pred);
				if (costFx == GMath::CLASSIFICATION)
					loss += -static_cast<double>(expv) * log(static_cast<double>(p));
				else
					loss += static_cast<double>(GMath::KLDivergence(expv, pred));
			}
			else
			{
				const double denom = (N > 0 && outSize > 0) ? static_cast<double>(N) * static_cast<double>(outSize) : 1.0;
				overallTotalError += static_cast<float>(GMath::outputNodeCost(expv, pred, static_cast<float>(denom), costFx));
			}
		}

		if (useSoftmax && N > 0)
			overallTotalError += static_cast<float>(loss / static_cast<double>(N));
		if (!std::isfinite(overallTotalError))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite loss aggregate detected (NaN/Inf)");
			running = false;
			return;
		}
	}

	// Add current results to cmatrix for accuracy vars
	if ((skeleton->getOutputType() == GMath::CLASSIFICATION) ||
		(skeleton->getOutputType() == GMath::KL))
		confusionMatrix.addResult(results);

	// Backprop + SGD update (minibatched) for train
	if (isTrain)
	{
		const unsigned int numLayers = static_cast<unsigned int>(tensorDff.sizes.size());
		if (numLayers >= 2)
		{
			const unsigned int lastLayer = numLayers - 1;
			const unsigned int lastTransition = static_cast<unsigned int>(tensorDff.T.size() - 1);
			const unsigned int outSize = tensorDff.sizes[lastLayer];

			Layer* outLayer = meat.getOutputLayer(lastLayer);
			const float* yData = NULL;
			unsigned int ySize = 0u;
			if (di)
				di->getTrainExpectedRowView(inputRowCounter, yData, ySize);
			const int costFx = skeleton->getOutputType();
			const int outActFx = skeleton->getActivationType(lastTransition);
			const float outActParam = skeleton->getActivationParam(lastTransition);

			// Output deltas
			for (unsigned int k = 0; k < outSize; ++k)
			{
				if (outLayer && !outLayer->possiblePath(k))
				{
					tensorDff.delta[lastLayer][k] = 0.0f;
					continue;
				}
				const float pred = tensorDff.a[lastLayer][k];
					const float expv = (yData && (k < ySize)) ? yData[k] : 0.0f;
				const bool useSoftmax =
				    ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) &&
				    (skeleton->getOutputLayerSize() > 1);
				if (useSoftmax)
				{
					// Softmax + (cross-entropy or KL) => dL/dz = p - y
					tensorDff.delta[lastLayer][k] = clipf_maybe(pred - expv, gradClip);
				}
				else
				{
					// Binary classification: if the output nonlinearity is sigmoid, use the stable
					// combined derivative for BCE-with-sigmoid:
					//   dL/dz = p - y
					// This avoids the numerically-unstable (p-y)/(p(1-p)) * p(1-p) path.
					if ((costFx == GMath::CLASSIFICATION) && (outActFx == GMath::SIGMOID || outActFx == GMath::SIGMOIDP))
					{
						tensorDff.delta[lastLayer][k] = clipf_maybe(pred - expv, gradClip);
					}
					else
					{
						const float dCost_dA = GMath::costErrDer(expv, pred, costFx);
						const float dA_dZ = GMath::activationErrDer(pred, outActFx, outActParam);
						// Basic gradient clipping for stability in extreme cases.
						tensorDff.delta[lastLayer][k] = clipf_maybe(dCost_dA * dA_dZ, gradClip);
					}
				}
				if (!is_finite(tensorDff.delta[lastLayer][k]))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite output delta detected (NaN/Inf)");
					running = false;
					return;
				}
			}

			// Hidden deltas (backwards)
			for (int li = static_cast<int>(lastLayer) - 1; li >= 1; --li)
			{
				const unsigned int l = static_cast<unsigned int>(li);
				const TensorDFFState::Transition& nextTr = tensorDff.T[l]; // maps layer l -> l+1
				Layer* curLayer = meat.getOutputLayer(l);

				const int actFx = skeleton->getActivationType(l - 1);
				const float actParam = skeleton->getActivationParam(l - 1);

				for (unsigned int i = 0; i < tensorDff.sizes[l]; ++i)
				{
					if (curLayer && !curLayer->possiblePath(i))
					{
						tensorDff.delta[l][i] = 0.0f;
						continue;
					}

					float sum = 0.0f;
					// sum_j delta_{l+1}[j] * W_next[j,i]
					for (unsigned int j = 0; j < nextTr.out; ++j)
						sum += tensorDff.delta[l + 1][j] * nextTr.W[static_cast<size_t>(j) * static_cast<size_t>(nextTr.in) + i];

					const float a = tensorDff.a[l][i];
					const float dA_dZ = GMath::activationErrDer(a, actFx, actParam);
					// Basic gradient clipping for stability in extreme cases.
					tensorDff.delta[l][i] = clipf_maybe(sum * dA_dZ, gradClip);
					if (!is_finite(tensorDff.delta[l][i]))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite hidden delta detected (NaN/Inf)");
						running = false;
						return;
					}
				}
			}

			// Accumulate minibatch gradients
			for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
			{
				TensorDFFState::Transition& tr = tensorDff.T[t];
				const unsigned int out = tr.out;
				const unsigned int in = tr.in;

				for (unsigned int j = 0; j < out; ++j)
				{
					const float d = tensorDff.delta[t + 1][j];
					if (!is_finite(d))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite minibatch delta detected (NaN/Inf)");
						running = false;
						return;
					}
					if (j < tr.gBias.size())
						tr.gBias[j] += d;
					const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(in);
					for (unsigned int i = 0; i < in; ++i)
						tr.gW[rowOff + i] += d * tensorDff.a[t][i];
				}
			}
			++tensorDff.batchCount;

			// End-of-batch detection (mirrors legacy minibatch semantics)
				const unsigned int trainSize = di ? di->getTrainSize() : 0;
				const int effectiveMiniBatchSize = (minibatchSize > 0) ? minibatchSize : static_cast<int>(trainSize);
				const bool isLastSample = (trainSize > 0) && (inputRowCounter + 1 >= trainSize);
			const bool isBatchEnd =
				(effectiveMiniBatchSize <= 1) ||
				(((inputRowCounter + 1) % static_cast<unsigned int>(effectiveMiniBatchSize)) == 0) ||
				isLastSample;

			if (isBatchEnd && tensorDff.batchCount > 0)
			{
				const float invBatch = 1.0f / static_cast<float>(tensorDff.batchCount);

					// Optional global grad-norm clipping (modern feature).
					// We compute the L2 norm of the (averaged) gradients including L1/L2 decay terms
					// (for parity with the actual update direction), then scale all gradients by:
					//   scale = min(1, clipNorm / (norm + eps))
					//
					// Defaults preserve behavior (globalGradClipNorm == 0 disables).
					float gradNorm = 0.0f;
					float gradScale = 1.0f;
					const float clipNorm = trainingConfig.globalGradClipNorm;
					if (clipNorm > 0.0f)
					{
						double sumsq = 0.0;
						for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
						{
							const TensorDFFState::Transition& tr = tensorDff.T[t];
							const float wd1 = skeleton->getWeightDecay1(t);
							const float wd2 = skeleton->getWeightDecay2(t);

							for (size_t idx = 0; idx < tr.W.size(); ++idx)
							{
								float g = tr.gW[idx] * invBatch;
								if ((wd1 != 0.0f) || (wd2 != 0.0f))
								{
									const float w = tr.W[idx];
									const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
									g += (wd1 * wSign) + (wd2 * w);
								}
								const double gd = static_cast<double>(g);
								sumsq += gd * gd;
							}

							// Bias grads (no decay, but included in the global norm).
							for (unsigned int j = 0; j < tr.gBias.size(); ++j)
							{
								const float gB = tr.gBias[j] * invBatch;
								const double bd = static_cast<double>(gB);
								sumsq += bd * bd;
							}
						}

						if (!is_finite_double(sumsq))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite grad-norm accumulation detected (NaN/Inf)");
							running = false;
							return;
						}
						gradNorm = static_cast<float>(sqrt(sumsq));
						const float eps = 1e-12f;
						if (gradNorm > clipNorm)
							gradScale = clipNorm / (gradNorm + eps);
						else
							gradScale = 1.0f;
					}
					lastGradNorm = gradNorm;
					lastGradNormScale = gradScale;
					if (!is_finite(lastGradNorm) || !is_finite(lastGradNormScale) || lastGradNormScale <= 0.0f)
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite grad clipping metadata detected (NaN/Inf)");
						running = false;
						return;
					}

				for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
				{
					TensorDFFState::Transition& tr = tensorDff.T[t];
						const float lr = skeleton->getLearningRate(t) * lrScheduleMultiplier;
					const float mf = skeleton->getMomentumFactor(t);
					const float wd1 = skeleton->getWeightDecay1(t);
					const float wd2 = skeleton->getWeightDecay2(t);

					for (size_t idx = 0; idx < tr.W.size(); ++idx)
					{
							float g = tr.gW[idx] * invBatch;
						// L1/L2 decay on weight (matches Node::getDelta semantics after bugfix).
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = tr.W[idx];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							g += (wd1 * wSign) + (wd2 * w);
						}
							// Apply optional global grad clipping scale.
							g *= gradScale;
						if (!is_finite(g))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite weight gradient detected (NaN/Inf)");
							running = false;
							return;
						}

						const float v = (mf * tr.vW[idx]) + (lr * g);
						tr.vW[idx] = v;
						tr.W[idx] -= v;
						tr.gW[idx] = 0.0f;
						if (!is_finite(tr.W[idx]) || !is_finite(tr.vW[idx]))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite weight update detected (NaN/Inf)");
							running = false;
							return;
						}
					}

					// Per-neuron bias (no momentum/weight decay; matches legacy engine behavior).
					{
						for (unsigned int j = 0; j < tr.out; ++j)
						{
								float gB = (j < tr.gBias.size()) ? (tr.gBias[j] * invBatch) : 0.0f;
								gB *= gradScale;
							if (!is_finite(gB))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite bias gradient detected (NaN/Inf)");
								running = false;
								return;
							}
							if (j < tr.bias.size())
								tr.bias[j] -= (lr * gB);
							if (j < tr.gBias.size())
								tr.gBias[j] = 0.0f;
							if (j < tr.bias.size() && !is_finite(tr.bias[j]))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite bias update detected (NaN/Inf)");
								running = false;
								return;
							}
						}
					}
				}

				// Defer syncing tensors -> Node/Edge graph until an epoch boundary.
				// (Trainer will sync once per epoch before callbacks.)
				graphWeightsDirty = true;
				tensorDff.batchCount = 0;
			}

			// Save the autotuning data (kept for parity with old path)
			{
				const float learningRate = skeleton->getLearningRate(lastTransition) * lrScheduleMultiplier;
				shmea::GList nbRow;
				nbRow.addFloat(overallTotalAccuracy);
				nbRow.addFloat(learningRate);
				nbRecord.addRow(nbRow);
			}
		}
	}
}

