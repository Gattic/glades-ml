// Net-type-specific SGD: DFF tensor path (split out of network.cpp)
#include "network.h"
#include "param_layout.h"
#include "sgd_utils.h"

#include "Backend/Database/GLogger.h"

#include "../DataObjects/DataInput.h"
#include "../GMath/gmath.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

using namespace glades;

namespace {
static inline void append_logfmt_kv(std::ostringstream& oss, const char* k, const std::string& v)
{
	oss << ' ' << k << '=';
	bool needQuote = false;
	for (size_t i = 0; i < v.size(); ++i)
	{
		const char c = v[i];
		if (c == ' ' || c == '=' || c == '"' || c == '\\' || c == '\n' || c == '\r' || c == '\t')
		{
			needQuote = true;
			break;
		}
	}
	if (!needQuote)
	{
		oss << v;
		return;
	}
	oss << '"';
	for (size_t i = 0; i < v.size(); ++i)
	{
		const char c = v[i];
		if (c == '\\' || c == '"')
			oss << '\\' << c;
		else if (c == '\n')
			oss << "\\n";
		else if (c == '\r')
			oss << "\\r";
		else if (c == '\t')
			oss << "\\t";
		else
			oss << c;
	}
	oss << '"';
}
static inline void append_logfmt_kv(std::ostringstream& oss, const char* k, int v) { oss << ' ' << k << '=' << v; }
static inline void append_logfmt_kv(std::ostringstream& oss, const char* k, unsigned int v) { oss << ' ' << k << '=' << v; }
static inline void append_logfmt_kv(std::ostringstream& oss, const char* k, unsigned long long v) { oss << ' ' << k << '=' << v; }
static inline void append_logfmt_kv(std::ostringstream& oss, const char* k, float v) { oss << ' ' << k << '=' << v; }
static inline void append_logfmt_kv(std::ostringstream& oss, const char* k, double v) { oss << ' ' << k << '=' << v; }
} // namespace

void glades::NNetwork::SGDHelper_DFF(unsigned int inputRowCounter, int runType)
{
	using namespace glades::sgd_detail;
	using namespace glades::param_layout;

	const bool isTrain = (runType == RUN_TRAIN);
	const unsigned int dataSize = isTrain ? (di ? di->getTrainSize() : 0u) : (di ? di->getTestSize() : 0u);
	const float gradClip = trainingConfig.perElementGradClip;

	// Ensure tensors exist (modern/tensor-only build).
	if (!ensureTensorParametersInitialized())
	{
		running = false;
		return;
	}
	if (!tensorDff.initialized)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_DFF: tensor state is not initialized");
		running = false;
		return;
	}

	// Input row: prefer sparse view when available.
	const unsigned int in = tensorDff.sizes.empty() ? 0u : tensorDff.sizes[0];
	const unsigned int* xIdx = NULL;
	const float* xVal = NULL;
	unsigned int xNNZ = 0u;
	unsigned int xFullSize = 0u;
	bool haveSparseX = false;
	if (di)
	{
		if (isTrain)
			haveSparseX = di->getTrainRowSparseView(inputRowCounter, xIdx, xVal, xNNZ, xFullSize);
		else
			haveSparseX = di->getTestRowSparseView(inputRowCounter, xIdx, xVal, xNNZ, xFullSize);
	}
	if (!(haveSparseX && xFullSize == in))
	{
		haveSparseX = false;
		xIdx = NULL;
		xVal = NULL;
		xNNZ = 0u;
		xFullSize = 0u;
	}
	// Filtered sparse features after (optional) input dropout.
	std::vector<unsigned int> xIdxF;
	std::vector<float> xValF;

	// Dropout masks (tensor-only):
	// - Layer 0 uses skeleton->getPInput()
	// - Hidden layers use skeleton->getPDropout(hiddenIdx)
	// Output layer is never dropped.
	//
	// IMPORTANT: This uses *inverted dropout* semantics:
	// - During training, kept activations are scaled by 1/(1-p) so their expectation matches inference.
	// - During evaluation, dropout is disabled (no scaling).
	const int H = skeleton->numHiddenLayers();
	std::vector<std::vector<unsigned char> > keepMasks;
	keepMasks.resize(static_cast<size_t>(H) + 1u);
	std::vector<float> keepScales;
	keepScales.assign(keepMasks.size(), 1.0f);
	{
		const float pIn = skeleton->getPInput();
		// Sparse input: never allocate a full per-feature dropout mask.
		// We will apply dropout only to the active (nnz) indices.
		if (!haveSparseX)
			keepMasks[0].assign(in, 1u);
		else
			keepMasks[0].clear();
		if (isTrain && pIn > 0.0f && !haveSparseX)
		{
			// Inverted dropout: scale kept activations by 1/(1-p).
			// Guard p>=1 to avoid division-by-zero (all units will be dropped anyway).
			if (pIn < 1.0f)
				keepScales[0] = 1.0f / (1.0f - pIn);
			for (unsigned int i = 0; i < in; ++i)
				keepMasks[0][i] = (glades::rng::uniform_double(rngEngine, 0.0, 1.0) >= static_cast<double>(pIn)) ? 1u : 0u;
		}
		// Sparse input dropout is handled after we read xIdx/xVal (see forward pass).
		if (isTrain && pIn > 0.0f && haveSparseX)
		{
			if (pIn < 1.0f)
				keepScales[0] = 1.0f / (1.0f - pIn);
		}
	}
	for (int h = 0; h < H; ++h)
	{
		const unsigned int layerIdx = static_cast<unsigned int>(h) + 1u;
		const unsigned int sz = (layerIdx < tensorDff.sizes.size()) ? tensorDff.sizes[layerIdx] : 0u;
		keepMasks[layerIdx].assign(sz, 1u);
		const float p = skeleton->getPDropout(static_cast<unsigned int>(h));
		if (isTrain && p > 0.0f)
		{
			if (p < 1.0f && layerIdx < keepScales.size())
				keepScales[layerIdx] = 1.0f / (1.0f - p);
			for (unsigned int j = 0; j < sz; ++j)
				keepMasks[layerIdx][j] = (glades::rng::uniform_double(rngEngine, 0.0, 1.0) >= static_cast<double>(p)) ? 1u : 0u;
		}
	}

	// Reset per-sample outputs
	results.clear();
	// Evaluation should not destroy any caller-visible training records.
	if (isTrain)
		nbRecord.clear();

	// Tensor buffers are initialized in ensureTensorParametersInitialized().

	// Forward pass for this sample
	{
		const float* xData = NULL;
		unsigned int xSize = 0u;
		if (!haveSparseX && di)
		{
			if (isTrain)
				di->getTrainRowView(inputRowCounter, xData, xSize);
			else
				di->getTestRowView(inputRowCounter, xData, xSize);
		}
		// Hard safety check: reject non-finite inputs early (sampled for performance on huge rows).
		if (!haveSparseX)
		{
			if (!span_all_finite_bounded(xData, xSize, /*maxChecks*/ 32u))
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_DFF: non-finite input data detected (NaN/Inf)");
				running = false;
				return;
			}
			for (unsigned int i = 0; i < in; ++i)
			{
				const bool keep = (keepMasks[0].empty() ? true : (keepMasks[0][i] != 0u));
				const float x = (xData && (i < xSize)) ? xData[i] : 0.0f;
				tensorDff.a[0][i] = keep ? (x * keepScales[0]) : 0.0f;
			}
		}
		else
		{
			// Sparse: build the post-dropout sparse view for this sample.
			// (Never materialize tensorDff.a[0] as a dense vector.)
			xIdxF.clear();
			xValF.clear();
			xIdxF.reserve(xNNZ);
			xValF.reserve(xNNZ);

			// Validate sparse input values (nnz is expected to be small).
			for (unsigned int k = 0; k < xNNZ; ++k)
			{
				const unsigned int fi = xIdx[k];
				const float xv = xVal[k];
				if (!is_finite(xv))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_DFF: non-finite sparse input value detected (NaN/Inf)");
					running = false;
					return;
				}
				// Enforce bounds defensively.
				if (fi >= in)
					continue;

				// Optional input dropout: apply only to active indices.
				const float pIn = skeleton->getPInput();
				if (isTrain && pIn > 0.0f)
				{
					const bool keep = (glades::rng::uniform_double(rngEngine, 0.0, 1.0) >= static_cast<double>(pIn));
					if (!keep)
						continue;
					xIdxF.push_back(fi);
					xValF.push_back(xv * keepScales[0]);
				}
				else
				{
					xIdxF.push_back(fi);
					xValF.push_back(xv);
				}
			}
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

			const int actFx = skeleton->getActivationType(t);
			const float actParam = skeleton->getActivationParam(t);
			const bool softmaxThisLayer = useSoftmax && (t == lastTransition) && (tr.out == outSize);
			if (softmaxThisLayer)
				outLogits.assign(tr.out, 0.0f);

			const bool dropoutThisLayer = isTrain && (t < static_cast<unsigned int>(H));
			const unsigned int outLayerIdx = t + 1u; // layer index in tensorDff.a / keepMasks (hidden only)
			const float dropoutScale =
			    (dropoutThisLayer && outLayerIdx < keepScales.size()) ? keepScales[outLayerIdx] : 1.0f;

			for (unsigned int j = 0; j < tr.out; ++j)
			{
				// Dropout: hidden layers only.
				if (dropoutThisLayer)
				{
					if (outLayerIdx < keepMasks.size() && (j < keepMasks[outLayerIdx].size()) && (keepMasks[outLayerIdx][j] == 0u))
					{
						tensorDff.a[t + 1][j] = 0.0f;
						continue;
					}
				}

				float z = (j < tr.bias.size()) ? tr.bias[j] : 0.0f;
				const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(tr.in);
				if (haveSparseX && t == 0u)
				{
					// Sparse input only applies to the first transition.
					for (unsigned int kk = 0; kk < static_cast<unsigned int>(xIdxF.size()); ++kk)
					{
						const unsigned int fi = xIdxF[kk];
						// fi < tr.in is guaranteed by our filtering above.
						z += tr.W[rowOff + fi] * xValF[kk];
					}
				}
				else
				{
					for (unsigned int i = 0; i < tr.in; ++i)
						z += tr.W[rowOff + i] * tensorDff.a[t][i];
				}
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
					// Inverted dropout scaling on hidden layers during training.
					tensorDff.a[t + 1][j] = dropoutThisLayer ? (a * dropoutScale) : a;
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

		// Add current results to cmatrix for accuracy vars
		if ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL))
		{
			const int expIdx = GMath::argmax(yData, outSize);
			const int predIdx = GMath::argmax(&tensorDff.a[last][0], outSize);
			confusionMatrix.addResultDirect(static_cast<unsigned int>(expIdx), static_cast<unsigned int>(predIdx));
		}
	}

	// Progress logs for long DFF epochs (bounded to ~20 messages per epoch,
	// and at most once per second wall-clock to avoid flooding on fast small datasets).
	{
		shmea::GLogger* logger = getLogger();
		if (logger && dataSize > 0u)
		{
			unsigned int every = 1u;
			if (dataSize > 20u)
				every = dataSize / 20u;
			if (every == 0u)
				every = 1u;
			const unsigned int done = inputRowCounter + 1u;
			if (done == dataSize || (done % every) == 0u)
			{
				const int64_t now = getCurrentTimeMilliseconds();
				if (now - lastStepLogTime >= 1000)
				{
					lastStepLogTime = now;
					std::ostringstream oss;
					oss << "event=nn_step_progress";
					append_logfmt_kv(oss, "net_type", netType);
					append_logfmt_kv(oss, "run_type", std::string(isTrain ? "train" : "eval"));
					append_logfmt_kv(oss, "epoch", epochs);
					append_logfmt_kv(oss, "step", done);
					append_logfmt_kv(oss, "steps_total", dataSize);
					append_logfmt_kv(oss, "loss_so_far", overallTotalError);
					append_logfmt_kv(oss, "lr_mult", lrScheduleMultiplier);
					if (trainingConfig.globalGradClipNorm > 0.0f)
					{
						append_logfmt_kv(oss, "grad_norm", lastGradNorm);
						append_logfmt_kv(oss, "grad_norm_scale", lastGradNormScale);
					}
					else
					{
						append_logfmt_kv(oss, "grad_norm", std::string("na"));
						append_logfmt_kv(oss, "grad_norm_scale", std::string("na"));
					}
					logger->info("NNetwork", shmea::GString(oss.str().c_str()));
				}
			}
		}
	}

	// Backprop + SGD update (minibatched) for train
	if (isTrain)
	{
		const unsigned int numLayers = static_cast<unsigned int>(tensorDff.sizes.size());
		if (numLayers >= 2)
		{
			const unsigned int lastLayer = numLayers - 1;
			const unsigned int lastTransition = static_cast<unsigned int>(tensorDff.T.size() - 1);
			const unsigned int outSize = tensorDff.sizes[lastLayer];
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

				const int actFx = skeleton->getActivationType(l - 1);
				const float actParam = skeleton->getActivationParam(l - 1);

				for (unsigned int i = 0; i < tensorDff.sizes[l]; ++i)
				{
					if (l < keepMasks.size() && (i < keepMasks[l].size()) && (keepMasks[l][i] == 0u))
					{
						tensorDff.delta[l][i] = 0.0f;
						continue;
					}

					float sum = 0.0f;
					// sum_j delta_{l+1}[j] * W_next[j,i]
					for (unsigned int j = 0; j < nextTr.out; ++j)
						sum += tensorDff.delta[l + 1][j] * nextTr.W[static_cast<size_t>(j) * static_cast<size_t>(nextTr.in) + i];

					// If dropout was applied to this hidden layer, tensorDff.a[l][i] stores the *scaled*
					// activation. activationErrDer expects the *unscaled* activation output (e.g. tanh(x), sigmoid(x)),
					// so undo the inverted-dropout scale here for correct derivatives.
					float aUnscaled = tensorDff.a[l][i];
					if (isTrain && l < keepScales.size())
					{
						const float sc = keepScales[l];
						if (sc != 1.0f)
						{
							// Keep-mask is non-zero here (dropped units continued above).
							aUnscaled = aUnscaled / sc;
						}
					}
					const float dA_dZ = GMath::activationErrDer(aUnscaled, actFx, actParam);
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
					if (haveSparseX && t == 0u)
					{
						// Sparse input: only accumulate gradients for active indices.
						for (unsigned int kk = 0; kk < static_cast<unsigned int>(xIdxF.size()); ++kk)
						{
							const unsigned int fi = xIdxF[kk];
							// fi < in by construction.
							tr.gW[rowOff + fi] += d * xValF[kk];
						}
					}
					else
					{
						for (unsigned int i = 0; i < in; ++i)
							tr.gW[rowOff + i] += d * tensorDff.a[t][i];
					}
				}
			}
			++tensorDff.batchCount;

			// End-of-batch detection
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

					// Per-neuron bias (no momentum/weight decay).
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

