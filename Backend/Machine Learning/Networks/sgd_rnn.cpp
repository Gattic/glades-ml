// Net-type-specific SGD: RNN BPTT path (split out of network.cpp)
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

void glades::NNetwork::SGDHelper_RNN(unsigned int inputRowCounter, int runType)
{
	using namespace glades::sgd_detail;
	using namespace glades::param_layout;

	const bool isTrain = (runType == RUN_TRAIN);
	const unsigned int dataSize = isTrain ? (di ? di->getTrainSize() : 0u) : (di ? di->getTestSize() : 0u);

	// Only run once per epoch (run() loops over all input layers).
	if (inputRowCounter != 0)
		return;

	// NOTE: Proper dropout-through-time requires saving/restoring dropout masks per timestep.
	// For correctness, we disable per-timestep scrambling in the BPTT path.
	// (dropout-through-time not implemented in this scalar BPTT path)

	const unsigned int seqCount = di ? (isTrain ? di->getTrainSequenceCount() : di->getTestSequenceCount()) : 0;
	if (seqCount == 0)
	{
		lastStatus = NNetworkStatus(
		    NNetworkStatus::INVALID_STATE,
		    isTrain ? "SGDHelper_RNN: no train sequences (DataInput::getTrainSequenceCount() == 0)"
		            : "SGDHelper_RNN: no test sequences (DataInput::getTestSequenceCount() == 0)");
		running = false;
		return;
	}

	const int H = skeleton->numHiddenLayers();
	const unsigned int outSize = skeleton->getOutputLayerSize();
	const unsigned int inputSize = di->getFeatureCount();
	if (H <= 0 || outSize == 0 || inputSize == 0)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_RNN: invalid layer sizes (hidden/output/input size == 0)");
		running = false;
		return;
	}

	// Optional truncated BPTT length (in timesteps).
	// IMPORTANT: This is intentionally separate from minibatch size.
	// - tbpttWindow == 0 => full-sequence BPTT
	// - tbpttWindow  > 0 => truncated BPTT windows of that length
	unsigned int tbptt = 0;
	{
		const int tbpttCfg = (trainingConfig.tbpttWindowOverride > 0) ? trainingConfig.tbpttWindowOverride
		                                                             : skeleton->getTBPTTWindow();
		if (tbpttCfg > 0)
			tbptt = static_cast<unsigned int>(tbpttCfg);
	}

	const float gradClip = trainingConfig.perElementGradClip;
	const int costFx = skeleton->getOutputType();

	// Ensure packed tensor parameters exist (modern/tensor-only).
	if (!ensureTensorParametersInitialized())
	{
		running = false;
		return;
	}
	if (!tensorRnn.initialized)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_RNN: tensor state not initialized");
		running = false;
		return;
	}

	// Sizes from tensor state (authoritative).
	std::vector<unsigned int> hiddenSizes = tensorRnn.hiddenSizes;

	// Reset per-epoch bookkeeping (RNN uses explicit per-sequence forward here).
	results.clear();
	// Evaluation should not destroy any caller-visible training records.
	if (isTrain)
		nbRecord.clear();

	// === Minibatching for recurrent nets (window batching) ===
	//
	// Historically this code applied updates after *every* TBPTT window, which is effectively
	// "batch size = 1 window" regardless of NNInfo::batchSize. That is both slow and not what
	// users expect when they configure a batch size.
	//
	// We now treat `minibatchSize` as "number of TBPTT windows to accumulate" before applying
	// an update.
	//
	// IMPORTANT (minibatch semantics):
	// For recurrent nets, a "batch element" is a TBPTT window (or a full sequence when tbptt==0).
	// We optimize for *sequence/window-level* averaging (sample averaging), not token/timestep
	// averaging:
	// - Within a window of length winLen, we accumulate gradients as the mean over timesteps
	//   (scale each timestep's gradient contribution by 1/winLen).
	// - Across a minibatch group, we apply gradients as the mean over windows
	//   (scale by 1/windowsInBatch in ApplyBatch()).
	//
	// This prevents longer sequences from dominating updates purely due to length and makes
	// "batch size" behave as users expect for sequence data.
	const unsigned int windowBatchMax =
	    (minibatchSize > 0 ? static_cast<unsigned int>(minibatchSize) : 1u);
	unsigned int windowsInBatch = 0u;
	unsigned int timeStepsInBatch = 0u;

	// Preallocate recurrent backprop buffers (avoid per-window heap churn).
	std::vector<float> deltaY(outSize, 0.0f);
	std::vector< std::vector<float> > deltaH;
	std::vector< std::vector<float> > nextTimeDeltaH;
	deltaH.resize(H);
	nextTimeDeltaH.resize(H);
	for (int l = 0; l < H; ++l)
	{
		deltaH[l].assign(hiddenSizes[l], 0.0f);
		nextTimeDeltaH[l].assign(hiddenSizes[l], 0.0f);
	}

	// Cache expected outputs for the current TBPTT window to avoid allocating/copying
	// expected rows again during backward.
	std::vector<float> expFlat;

	// === Minibatch application helpers (replaces macro-based control flow) ===
	struct ClearGrads
	{
		TensorRNNState& tensorRnn;
		int H;
		ClearGrads(TensorRNNState& tr, int h) : tensorRnn(tr), H(h) {}
		void operator()() const
		{
			for (int l = 0; l < H; ++l)
			{
				TensorRNNState::Hidden& hl = tensorRnn.H[static_cast<size_t>(l)];
				std::fill(hl.gWxh.begin(), hl.gWxh.end(), 0.0f);
				std::fill(hl.gWhh.begin(), hl.gWhh.end(), 0.0f);
				std::fill(hl.gBias.begin(), hl.gBias.end(), 0.0f);
			}
			std::fill(tensorRnn.O.gWhy.begin(), tensorRnn.O.gWhy.end(), 0.0f);
			std::fill(tensorRnn.O.gBias.begin(), tensorRnn.O.gBias.end(), 0.0f);
		}
	};
	ClearGrads clearGrads(tensorRnn, H);

	struct ApplyBatch
	{
		NNInfo* skeleton;
		float& lrScheduleMultiplier;
		TrainingConfig& trainingConfig;
		float& lastGradNorm;
		float& lastGradNormScale;
		NNetworkStatus& lastStatus;
		bool& running;
		TensorRNNState& tensorRnn;
		int H;
		unsigned int outSize;

		ApplyBatch(NNetwork& net, TensorRNNState& tr, int h, unsigned int os)
		    : skeleton(net.skeleton),
		      lrScheduleMultiplier(net.lrScheduleMultiplier),
		      trainingConfig(net.trainingConfig),
		      lastGradNorm(net.lastGradNorm),
		      lastGradNormScale(net.lastGradNormScale),
		      lastStatus(net.lastStatus),
		      running(net.running),
		      tensorRnn(tr),
		      H(h),
		      outSize(os)
		{
		}

		bool operator()(int windowsInBatch) const
		{
			using namespace glades::sgd_detail;
			if (windowsInBatch <= 0)
				return true;

			// NOTE: recurrent minibatch averaging is by windows/sequences, not timesteps.
			const float invBatch = 1.0f / static_cast<float>(windowsInBatch);

			// Output transition hyperparams live at index == H.
			const unsigned int outIdx = static_cast<unsigned int>(H);
			const float lrOut = skeleton->getLearningRate(outIdx) * lrScheduleMultiplier;
			const float mfOut = skeleton->getMomentumFactor(outIdx);
			const float wd1Out = skeleton->getWeightDecay1(outIdx);
			const float wd2Out = skeleton->getWeightDecay2(outIdx);

			// Optional global grad-norm clipping (unified with DFF path).
			float gradNorm = 0.0f;
			float gradScale = 1.0f;
			const float clipNorm = trainingConfig.globalGradClipNorm;
			if (clipNorm > 0.0f)
			{
				double sumsq = 0.0;
				// Output weights (include decay terms for parity with update direction).
				for (size_t gidx = 0; gidx < tensorRnn.O.Why.size(); ++gidx)
				{
					float gg = tensorRnn.O.gWhy[gidx] * invBatch;
					if ((wd1Out != 0.0f) || (wd2Out != 0.0f))
					{
						const float w = tensorRnn.O.Why[gidx];
						const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
						gg += (wd1Out * wSign) + (wd2Out * w);
					}
					const double gd = static_cast<double>(gg);
					sumsq += gd * gd;
				}
				for (unsigned int k2 = 0; k2 < outSize; ++k2)
				{
					const float gB2 = tensorRnn.O.gBias[k2] * invBatch;
					const double bd = static_cast<double>(gB2);
					sumsq += bd * bd;
				}
				// Hidden layers
				for (int l2 = 0; l2 < H; ++l2)
				{
					const unsigned int li2 = static_cast<unsigned int>(l2);
					const TensorRNNState::Hidden& hl2 = tensorRnn.H[static_cast<size_t>(l2)];
					const float wd1 = skeleton->getWeightDecay1(li2);
					const float wd2 = skeleton->getWeightDecay2(li2);
					for (size_t gidx = 0; gidx < hl2.Wxh.size(); ++gidx)
					{
						float gg = hl2.gWxh[gidx] * invBatch;
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = hl2.Wxh[gidx];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							gg += (wd1 * wSign) + (wd2 * w);
						}
						const double gd = static_cast<double>(gg);
						sumsq += gd * gd;
					}
					for (size_t gidx = 0; gidx < hl2.Whh.size(); ++gidx)
					{
						float gg = hl2.gWhh[gidx] * invBatch;
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = hl2.Whh[gidx];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							gg += (wd1 * wSign) + (wd2 * w);
						}
						const double gd = static_cast<double>(gg);
						sumsq += gd * gd;
					}
					for (unsigned int i2 = 0; i2 < hl2.gBias.size(); ++i2)
					{
						const float gB2 = hl2.gBias[i2] * invBatch;
						const double bd = static_cast<double>(gB2);
						sumsq += bd * bd;
					}
				}

				if (!is_finite_double(sumsq))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite grad-norm accumulation detected (NaN/Inf)");
					running = false;
					return false;
				}
				gradNorm = static_cast<float>(sqrt(sumsq));
				const float eps = 1e-12f;
				gradScale = (gradNorm > clipNorm) ? (clipNorm / (gradNorm + eps)) : 1.0f;
			}
			lastGradNorm = gradNorm;
			lastGradNormScale = gradScale;
			if (!is_finite(lastGradNorm) || !is_finite(lastGradNormScale) || lastGradNormScale <= 0.0f)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite grad clipping metadata detected (NaN/Inf)");
				running = false;
				return false;
			}

			// Update output weights.
			for (size_t idx = 0; idx < tensorRnn.O.Why.size(); ++idx)
			{
				float g = tensorRnn.O.gWhy[idx] * invBatch;
				if ((wd1Out != 0.0f) || (wd2Out != 0.0f))
				{
					const float w = tensorRnn.O.Why[idx];
					const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
					g += (wd1Out * wSign) + (wd2Out * w);
				}
				g *= gradScale;
				const float v = (mfOut * tensorRnn.O.vWhy[idx]) + (lrOut * g);
				tensorRnn.O.vWhy[idx] = v;
				tensorRnn.O.Why[idx] -= v;
				tensorRnn.O.gWhy[idx] = 0.0f;
				if (!is_finite(tensorRnn.O.Why[idx]))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite output weight after tensor update (NaN/Inf)");
					running = false;
					return false;
				}
			}
			for (unsigned int k = 0; k < outSize; ++k)
			{
				float gB = tensorRnn.O.gBias[k] * invBatch;
				gB *= gradScale;
				tensorRnn.O.bias[k] -= (lrOut * gB);
				tensorRnn.O.gBias[k] = 0.0f;
				if (!is_finite(tensorRnn.O.bias[k]))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite output bias after tensor update (NaN/Inf)");
					running = false;
					return false;
				}
			}

			// Update hidden layers.
			for (int l = 0; l < H; ++l)
			{
				const unsigned int li = static_cast<unsigned int>(l);
				TensorRNNState::Hidden& hl = tensorRnn.H[static_cast<size_t>(l)];
				const float lr = skeleton->getLearningRate(li) * lrScheduleMultiplier;
				const float mf = skeleton->getMomentumFactor(li);
				const float wd1 = skeleton->getWeightDecay1(li);
				const float wd2 = skeleton->getWeightDecay2(li);

				for (size_t idx = 0; idx < hl.Wxh.size(); ++idx)
				{
					float g = hl.gWxh[idx] * invBatch;
					if ((wd1 != 0.0f) || (wd2 != 0.0f))
					{
						const float w = hl.Wxh[idx];
						const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
						g += (wd1 * wSign) + (wd2 * w);
					}
					g *= gradScale;
					const float v = (mf * hl.vWxh[idx]) + (lr * g);
					hl.vWxh[idx] = v;
					hl.Wxh[idx] -= v;
					hl.gWxh[idx] = 0.0f;
					if (!is_finite(hl.Wxh[idx]))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite hidden Wx after tensor update (NaN/Inf)");
						running = false;
						return false;
					}
				}
				for (size_t idx = 0; idx < hl.Whh.size(); ++idx)
				{
					float g = hl.gWhh[idx] * invBatch;
					if ((wd1 != 0.0f) || (wd2 != 0.0f))
					{
						const float w = hl.Whh[idx];
						const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
						g += (wd1 * wSign) + (wd2 * w);
					}
					g *= gradScale;
					const float v = (mf * hl.vWhh[idx]) + (lr * g);
					hl.vWhh[idx] = v;
					hl.Whh[idx] -= v;
					hl.gWhh[idx] = 0.0f;
					if (!is_finite(hl.Whh[idx]))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite hidden Wh after tensor update (NaN/Inf)");
						running = false;
						return false;
					}
				}
				for (unsigned int i = 0; i < hl.h; ++i)
				{
					float gB = hl.gBias[i] * invBatch;
					gB *= gradScale;
					hl.bias[i] -= (lr * gB);
					hl.gBias[i] = 0.0f;
					if (!is_finite(hl.bias[i]))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite hidden bias after tensor update (NaN/Inf)");
						running = false;
						return false;
					}
				}
			}

			return true;
		}
	};
	ApplyBatch applyBatch(*this, tensorRnn, H, outSize);

	// NOTE: overallTotalAccuracy is reported timestep-averaged for compatibility.

	// Progress logging (heartbeat) for long sequence epochs.
	shmea::GLogger* logger = getLogger();
	const int64_t epochStartMs = getCurrentTimeMilliseconds();
	int64_t lastProgressMs = epochStartMs;
	static const int64_t kProgressIntervalMs = 5000;
	unsigned int progressEverySeq = 1u;
	if (seqCount > 20u)
		progressEverySeq = seqCount / 20u;
	if (progressEverySeq == 0u)
		progressEverySeq = 1u;
	unsigned long long tokensProcessed = 0ULL;

	// Process each sequence independently (hidden state resets at each sequence boundary).
	for (unsigned int s = 0; s < seqCount; ++s)
	{
		const unsigned int seqLen = isTrain ? di->getTrainSequenceLength(s) : di->getTestSequenceLength(s);
		if (seqLen == 0)
			continue;
		tokensProcessed += static_cast<unsigned long long>(seqLen);

		// Hidden state carried forward across TBPTT windows within a sequence.
		std::vector< std::vector<float> > hPrev;
		hPrev.resize(H);
		for (int l = 0; l < H; ++l)
			hPrev[l].assign(hiddenSizes[l], 0.0f);

		unsigned int t0 = 0;
		while (t0 < seqLen)
		{
			const unsigned int winLen = (tbptt > 0) ? std::min(tbptt, seqLen - t0) : (seqLen - t0);
			if (winLen == 0)
				break;

			// Preallocate scratch buffers for this window
			recScratch.ensureRNN(winLen, inputSize, outSize, hiddenSizes);
			std::vector<float>& xFlat = recScratch.x;
			std::vector<float>& yFlat = recScratch.y;
			std::vector< std::vector<float> >& hFlat = recScratch.h;
			std::vector< std::vector<float> >& hPrevAtTFlat = recScratch.hPrevAtT;

			// Ensure expected-output cache is sized for this window.
			{
				const size_t want = static_cast<size_t>(winLen) * static_cast<size_t>(outSize);
				if (expFlat.size() != want)
					expFlat.resize(want);
			}

			// --- forward ---
			for (unsigned int t = 0; t < winLen; ++t)
			{
				const unsigned int tt = t0 + t;
				const float* rowData = NULL;
				unsigned int rowSize = 0u;
				if (isTrain)
					di->getTrainSequenceRowView(s, tt, rowData, rowSize);
				else
					di->getTestSequenceRowView(s, tt, rowData, rowSize);
				for (unsigned int p = 0; p < inputSize; ++p)
				{
					const float v = (rowData && (p < rowSize)) ? rowData[p] : 0.0f;
					if (!is_finite(v))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_RNN: non-finite input sequence value detected (NaN/Inf)");
						running = false;
						return;
					}
					xFlat[static_cast<size_t>(t) * static_cast<size_t>(inputSize) + p] = v;
				}

				// Hidden layers
				for (int l = 0; l < H; ++l)
				{
					const unsigned int curSize = hiddenSizes[l];
					const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[l - 1];
					std::vector<float>& hL = hFlat[l];
					std::vector<float>& hPrevAtTL = hPrevAtTFlat[l];
					const size_t tOffCur = static_cast<size_t>(t) * static_cast<size_t>(curSize);

					// Snapshot h(t-1) for recurrent gradient
					for (unsigned int i = 0; i < curSize; ++i)
						hPrevAtTL[tOffCur + i] = hPrev[l][i];

					const int actFx = skeleton->getActivationType(static_cast<unsigned int>(l));
					const float actParam = skeleton->getActivationParam(static_cast<unsigned int>(l));

					for (unsigned int i = 0; i < curSize; ++i)
					{
						TensorRNNState::Hidden& hl = tensorRnn.H[static_cast<size_t>(l)];
						float net = (i < hl.bias.size()) ? hl.bias[i] : 0.0f;
						// Feedforward contribution
						for (unsigned int p = 0; p < prevSize; ++p)
						{
							const float inAct =
								(l == 0)
									? xFlat[static_cast<size_t>(t) * static_cast<size_t>(inputSize) + p]
									: hFlat[l - 1][static_cast<size_t>(t) * static_cast<size_t>(hiddenSizes[l - 1]) + p];
							net += hl.Wxh[static_cast<size_t>(i) * static_cast<size_t>(prevSize) + p] * inAct;
						}
						// Recurrent contribution
						for (unsigned int j = 0; j < curSize; ++j)
							net += hl.Whh[static_cast<size_t>(i) * static_cast<size_t>(curSize) + j] * hPrev[l][j];

						if (!is_finite(net))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite hidden pre-activation detected (NaN/Inf)");
							running = false;
							return;
						}
						const float a = GMath::squash(net, actFx, actParam);
						if (!is_finite(a))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite hidden activation detected (NaN/Inf)");
							running = false;
							return;
						}
						hL[tOffCur + i] = a;
					}

					// Update carried state for next timestep
					for (unsigned int i = 0; i < curSize; ++i)
						hPrev[l][i] = hL[tOffCur + i];
				}

				// Output layer
				{
					const unsigned int lrIdx = static_cast<unsigned int>(H);
					const int actFx = skeleton->getActivationType(lrIdx);
					const float actParam = skeleton->getActivationParam(lrIdx);
					const bool useSoftmax =
					    ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1);

					std::vector<float>& outLogits = recScratch.outLogits;
					std::vector<float>& outProbs = recScratch.outProbs;
					if (useSoftmax)
						std::fill(outLogits.begin(), outLogits.end(), 0.0f);

					const size_t tOffOut = static_cast<size_t>(t) * static_cast<size_t>(outSize);
					for (unsigned int k = 0; k < outSize; ++k)
					{
						const unsigned int prevSize = hiddenSizes[H - 1];
						float net = (k < tensorRnn.O.bias.size()) ? tensorRnn.O.bias[k] : 0.0f;
						const size_t rowOff = static_cast<size_t>(k) * static_cast<size_t>(prevSize);
						const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(prevSize);
						for (unsigned int i = 0; i < prevSize; ++i)
							net += tensorRnn.O.Why[rowOff + i] * hFlat[H - 1][hOff + i];

						if (!is_finite(net))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite output pre-activation detected (NaN/Inf)");
							running = false;
							return;
						}
						if (useSoftmax)
						{
							if (!is_finite(net))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite output logit detected (NaN/Inf)");
								running = false;
								return;
							}
							outLogits[k] = net;
						}
						else
						{
							const float a = GMath::squash(net, actFx, actParam);
							if (!is_finite(a))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite output activation detected (NaN/Inf)");
								running = false;
								return;
							}
							yFlat[tOffOut + k] = a;
						}
					}

					if (useSoftmax)
					{
						softmax_stable(outLogits, outProbs);
						for (unsigned int k = 0; k < outSize; ++k)
						{
							const float p = outProbs[k];
							if (!is_finite(p))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite softmax probability detected (NaN/Inf)");
								running = false;
								return;
							}
							yFlat[tOffOut + k] = p;
						}
					}
				}

				// Metrics + confusion matrix for this timestep
				{
					results.clear();
					const float* expData = NULL;
					unsigned int expSize = 0u;
					if (isTrain)
						di->getTrainSequenceExpectedRowView(s, tt, expData, expSize);
					else
						di->getTestSequenceExpectedRowView(s, tt, expData, expSize);

					const unsigned int N = dataSize;
					const double denom = (N > 0 && outSize > 0) ? static_cast<double>(N) * static_cast<double>(outSize) : 1.0;
					const bool useSoftmax =
					    ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1);

					double loss = 0.0;
					const size_t tOffOut = static_cast<size_t>(t) * static_cast<size_t>(outSize);
					for (unsigned int k = 0; k < outSize; ++k)
					{
						const float expectation = (expData && (k < expSize)) ? expData[k] : 0.0f;
						// Cache expected values for backward pass (avoid repeated DataInput materialization).
						expFlat[tOffOut + k] = expectation;
						const float prediction = yFlat[tOffOut + k];
						if (!is_finite(expectation))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_RNN: non-finite expected sequence value detected (NaN/Inf)");
							running = false;
							return;
						}
						if (!is_finite(prediction))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite prediction detected (NaN/Inf)");
							running = false;
							return;
						}

						results.addFloat(expectation);
						results.addFloat(prediction);

						if (costFx == GMath::REGRESSION)
						{
							const double diff = static_cast<double>(prediction) - static_cast<double>(expectation);
							regSSE += diff * diff;
							regSAE += fabs(diff);
							regSumY += static_cast<double>(expectation);
							regSumY2 += static_cast<double>(expectation) * static_cast<double>(expectation);
							++regCount;
						}
						else if (useSoftmax)
						{
							const float p = clamp_prob01(prediction);
							if (costFx == GMath::CLASSIFICATION)
								loss += -static_cast<double>(expectation) * log(static_cast<double>(p));
							else
								loss += static_cast<double>(GMath::KLDivergence(expectation, prediction));
						}
						else
						{
							overallTotalError += static_cast<float>(GMath::outputNodeCost(expectation, prediction, static_cast<float>(denom), costFx));
						}
					}

					if (useSoftmax && N > 0)
						overallTotalError += static_cast<float>(loss / static_cast<double>(N));
					if (!std::isfinite(overallTotalError))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite loss aggregate detected (NaN/Inf)");
						running = false;
						return;
					}

					if ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL))
					{
						const int expIdx = GMath::argmax(expData, expSize);
						const int predIdx = GMath::argmax(&yFlat[tOffOut], outSize);
						confusionMatrix.addResultDirect(static_cast<unsigned int>(expIdx), static_cast<unsigned int>(predIdx));
					}
				}
			} // forward timesteps

			// --- backward (BPTT) + update for this window ---
			if (runType == RUN_TRAIN)
			{
				// Clear accumulators only when starting a new minibatch group of windows.
				if (windowsInBatch == 0u)
				{
					clearGrads();
					timeStepsInBatch = 0u;
				}

				// Per-window normalization so each window contributes equally regardless of winLen.
				// This implements "mean loss per timestep within a window" semantics.
				const float invWinLen = (winLen > 0u) ? (1.0f / static_cast<float>(winLen)) : 1.0f;

				std::fill(deltaY.begin(), deltaY.end(), 0.0f);
				for (int l = 0; l < H; ++l)
				{
					std::fill(deltaH[l].begin(), deltaH[l].end(), 0.0f);
					std::fill(nextTimeDeltaH[l].begin(), nextTimeDeltaH[l].end(), 0.0f);
				}

				// Backward through time within this window only (TBPTT detaches across windows).
				for (int t = static_cast<int>(winLen) - 1; t >= 0; --t)
				{
					const size_t tOffOut = static_cast<size_t>(t) * static_cast<size_t>(outSize);

					// --- output deltas + output weight/bias deltas ---
					{
						const unsigned int outActIdx = static_cast<unsigned int>(H);
						const int actFx = skeleton->getActivationType(outActIdx);
						const float actParam = skeleton->getActivationParam(outActIdx);
						const unsigned int outIn = hiddenSizes[H - 1];

						for (unsigned int k = 0; k < outSize; ++k)
						{
							const float pred = yFlat[tOffOut + k];
							const float expv = expFlat[tOffOut + k];
							if (!is_finite(expv))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_RNN: non-finite expected value detected in BPTT (NaN/Inf)");
								running = false;
								return;
							}
							if (!is_finite(pred))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite prediction detected in BPTT (NaN/Inf)");
								running = false;
								return;
							}
							const bool useSoftmax =
							    ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1);
							if (useSoftmax)
							{
								// Softmax + (cross-entropy or KL) => dL/dz = p - y
								deltaY[k] = clipf_maybe(pred - expv, gradClip);
							}
							else
							{
								// Binary classification: if the output nonlinearity is sigmoid, use the stable
								// combined derivative for BCE-with-sigmoid:
								//   dL/dz = p - y
								if ((costFx == GMath::CLASSIFICATION) && (actFx == GMath::SIGMOID || actFx == GMath::SIGMOIDP))
								{
									deltaY[k] = clipf_maybe(pred - expv, gradClip);
								}
								else
								{
									const float dCost_dA = GMath::costErrDer(expv, pred, costFx);
									const float dA_dZ = GMath::activationErrDer(pred, actFx, actParam);
									deltaY[k] = clipf_maybe(dCost_dA * dA_dZ, gradClip);
								}
							}
							if (!is_finite(deltaY[k]))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite output delta detected (NaN/Inf)");
								running = false;
								return;
							}

							// Accumulate output gradients (Why and bias).
							const size_t rowOff = static_cast<size_t>(k) * static_cast<size_t>(outIn);
							const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(outIn);
							for (unsigned int i = 0; i < outIn; ++i)
								tensorRnn.O.gWhy[rowOff + i] += (deltaY[k] * invWinLen) * hFlat[H - 1][hOff + i];
							tensorRnn.O.gBias[k] += (deltaY[k] * invWinLen);
						}
					}

					// --- hidden deltas (each layer, backwards) ---
					for (int l = H - 1; l >= 0; --l)
					{
						const unsigned int actIdx = static_cast<unsigned int>(l);
						const int actFx = skeleton->getActivationType(actIdx);
						const float actParam = skeleton->getActivationParam(actIdx);

						const unsigned int curSize = hiddenSizes[l];
						const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[l - 1];
						const size_t tOffCur = static_cast<size_t>(t) * static_cast<size_t>(curSize);
						TensorRNNState::Hidden& hl = tensorRnn.H[static_cast<size_t>(l)];

						for (unsigned int i = 0; i < curSize; ++i)
						{
							float sumFromNextLayer = 0.0f;

							// Contribution from next layer at same time (feedforward)
							if (l == H - 1)
							{
								// from output layer
								const unsigned int outIn = hiddenSizes[H - 1];
								for (unsigned int k = 0; k < outSize; ++k)
								{
									sumFromNextLayer += (deltaY[k] * tensorRnn.O.Why[static_cast<size_t>(k) * static_cast<size_t>(outIn) + i]);
								}
							}
							else
							{
								// from hidden layer l+1
								const unsigned int nextSize = hiddenSizes[l + 1];
								TensorRNNState::Hidden& hn = tensorRnn.H[static_cast<size_t>(l + 1)];
								for (unsigned int j = 0; j < nextSize; ++j)
								{
									sumFromNextLayer += (deltaH[l + 1][j] *
									                    hn.Wxh[static_cast<size_t>(j) * static_cast<size_t>(curSize) + i]);
								}
							}

							// Recurrent contribution from next time step (full Wh^T * delta(t+1))
							float recurrent = 0.0f;
							{
								// recurrent_i = sum_k delta_{t+1,k} * Wh_{k,i}
								for (unsigned int k = 0; k < curSize; ++k)
									recurrent += nextTimeDeltaH[l][k] *
									             hl.Whh[static_cast<size_t>(k) * static_cast<size_t>(curSize) + i];
							}

							const float a = hFlat[l][tOffCur + i];
							const float dA_dZ = GMath::activationErrDer(a, actFx, actParam);
							deltaH[l][i] = clipf_maybe((sumFromNextLayer + recurrent) * dA_dZ, gradClip);
							if (!is_finite(deltaH[l][i]))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite hidden delta detected (NaN/Inf)");
								running = false;
								return;
							}

							// Accumulate feedforward weight gradients
							for (unsigned int p = 0; p < prevSize; ++p)
							{
								const float inAct =
									(l == 0)
										? xFlat[static_cast<size_t>(t) * static_cast<size_t>(inputSize) + p]
										: hFlat[l - 1][static_cast<size_t>(t) * static_cast<size_t>(hiddenSizes[l - 1]) + p];
								hl.gWxh[static_cast<size_t>(i) * static_cast<size_t>(prevSize) + p] += (deltaH[l][i] * invWinLen) * inAct;
							}

							// Bias gradient
							hl.gBias[i] += (deltaH[l][i] * invWinLen);

							// Recurrent weight deltas: dWh_ij += delta_i(t) * h_j(t-1)
							for (unsigned int j = 0; j < curSize; ++j)
							{
								const float prevH = hPrevAtTFlat[l][tOffCur + j];
								hl.gWhh[static_cast<size_t>(i) * static_cast<size_t>(curSize) + j] += (deltaH[l][i] * invWinLen) * prevH;
							}
						} // i
					} // l

					// Shift time-recursion deltas: nextTimeDeltaH = deltaH at current timestep
					for (int l = 0; l < H; ++l)
						nextTimeDeltaH[l] = deltaH[l];
				} // t

				// Batch bookkeeping: accumulate timesteps across windows in this minibatch group.
				timeStepsInBatch += winLen;
				++windowsInBatch;
				if (windowsInBatch >= windowBatchMax)
				{
					if (!applyBatch(static_cast<int>(windowsInBatch)))
						return;
					windowsInBatch = 0u;
					timeStepsInBatch = 0u;
				}
			} // RUN_TRAIN

			t0 += winLen;
		} // TBPTT windows

		// Log at sequence boundaries (time- or count-based), excluding the final sequence (final epoch metrics are logged elsewhere).
		if (logger && (s + 1u) < seqCount)
		{
			const int64_t nowMs = getCurrentTimeMilliseconds();
			const bool dueBySeq = (((s + 1u) % progressEverySeq) == 0u);
			const bool dueByTime = ((nowMs - lastProgressMs) >= kProgressIntervalMs);
			if (dueBySeq || dueByTime)
			{
				lastProgressMs = nowMs;
				const double elapsedMs = static_cast<double>(nowMs - epochStartMs);
				const double tokPerSec = (elapsedMs > 0.0) ? (static_cast<double>(tokensProcessed) / (elapsedMs / 1000.0)) : 0.0;
				std::ostringstream oss;
				oss << "event=nn_epoch_progress";
				append_logfmt_kv(oss, "net_type", netType);
				append_logfmt_kv(oss, "run_type", std::string(isTrain ? "train" : "eval"));
				append_logfmt_kv(oss, "epoch", epochs);
				append_logfmt_kv(oss, "seq_done", s + 1u);
				append_logfmt_kv(oss, "seq_total", seqCount);
				append_logfmt_kv(oss, "tokens_seen", tokensProcessed);
				append_logfmt_kv(oss, "tokens_per_sec", tokPerSec);
				append_logfmt_kv(oss, "loss_so_far", overallTotalError);
				append_logfmt_kv(oss, "lr_mult", lrScheduleMultiplier);
				// Grad-norm is only computed when global grad clipping is enabled (for performance).
				// Avoid printing misleading zeros when it is disabled.
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
	} // sequences

	// Flush any partial minibatch group.
	if (runType == RUN_TRAIN && windowsInBatch > 0u)
	{
		if (!applyBatch(static_cast<int>(windowsInBatch)))
			return;
		windowsInBatch = 0u;
		timeStepsInBatch = 0u;
	}
}

