// Net-type-specific SGD: GRU BPTT path (split out of network.cpp)
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

void glades::NNetwork::SGDHelper_GRU(unsigned int inputRowCounter, int runType)
{
	using namespace glades::sgd_detail;
	using namespace glades::param_layout;

	const bool isTrain = (runType == RUN_TRAIN);
	const unsigned int dataSize = isTrain ? (di ? di->getTrainSize() : 0u) : (di ? di->getTestSize() : 0u);

	// Only run once per epoch (run() loops over all input layers).
	if (inputRowCounter != 0)
		return;

	// (dropout-through-time not implemented in this scalar BPTT path)

	const unsigned int seqCount = di ? (isTrain ? di->getTrainSequenceCount() : di->getTestSequenceCount()) : 0;
	if (seqCount == 0)
	{
		lastStatus = NNetworkStatus(
		    NNetworkStatus::INVALID_STATE,
		    isTrain ? "SGDHelper_GRU: no train sequences (DataInput::getTrainSequenceCount() == 0)"
		            : "SGDHelper_GRU: no test sequences (DataInput::getTestSequenceCount() == 0)");
		running = false;
		return;
	}

	const int H = skeleton->numHiddenLayers();
	const unsigned int outSize = skeleton->getOutputLayerSize();
	const unsigned int inputSize = di->getFeatureCount();
	if (H <= 0 || outSize == 0 || inputSize == 0)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_GRU: invalid layer sizes (hidden/output/input size == 0)");
		running = false;
		return;
	}

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
	if (!tensorGru.initialized)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_GRU: tensor state not initialized");
		running = false;
		return;
	}

	// Sizes from tensor state (authoritative).
	std::vector<unsigned int> hiddenSizes = tensorGru.hiddenSizes;

	results.clear();
	// Evaluation should not destroy any caller-visible training records.
	if (isTrain)
		nbRecord.clear();

	// === Minibatching for recurrent nets (window batching) ===
	// See sgd_rnn.cpp for rationale. We batch TBPTT windows and apply updates once per batch.
	const unsigned int windowBatchMax =
	    (minibatchSize > 0 ? static_cast<unsigned int>(minibatchSize) : 1u);
	unsigned int windowsInBatch = 0u;
	unsigned int timeStepsInBatch = 0u;

	// Preallocate recurrent backprop buffers (avoid per-window heap churn).
	std::vector<float> deltaY(outSize, 0.0f);
	std::vector< std::vector<float> > nextTimeDh;
	nextTimeDh.resize(H);
	for (int l = 0; l < H; ++l)
		nextTimeDh[l].assign(hiddenSizes[l], 0.0f);

	// === Backward-pass scratch buffers ===
	// These were historically allocated per-timestep, per-layer (catastrophic for long sequences).
	// We preallocate once and reuse to eliminate inner-loop heap churn.
	std::vector< std::vector<float> > dhBuf;
	std::vector< std::vector<float> > daZBuf;
	std::vector< std::vector<float> > daRBuf;
	std::vector< std::vector<float> > daHBuf;
	std::vector< std::vector<float> > dhPrevBuf;
	std::vector< std::vector<float> > tmpBuf;
	std::vector< std::vector<float> > dInputBuf; // gradient w.r.t. this layer's input (prev activations)
	dhBuf.resize(H);
	daZBuf.resize(H);
	daRBuf.resize(H);
	daHBuf.resize(H);
	dhPrevBuf.resize(H);
	tmpBuf.resize(H);
	dInputBuf.resize(H);
	for (int l = 0; l < H; ++l)
	{
		const unsigned int curSize = hiddenSizes[l];
		const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[l - 1];
		dhBuf[l].resize(curSize);
		daZBuf[l].resize(curSize);
		daRBuf[l].resize(curSize);
		daHBuf[l].resize(curSize);
		dhPrevBuf[l].resize(curSize);
		tmpBuf[l].resize(curSize);
		dInputBuf[l].resize(prevSize);
	}
	const std::vector<float>* dInputFromAbove = NULL;

	// Cache expected outputs for the current TBPTT window to avoid allocating/copying
	// expected rows again during backward.
	std::vector<float> expFlat;

	// === Minibatch application helpers (replaces macro-based control flow) ===
	struct ClearGrads
	{
		TensorGatedState& tensorGru;
		int H;
		ClearGrads(TensorGatedState& tg, int h) : tensorGru(tg), H(h) {}
		void operator()() const
		{
			for (int l = 0; l < H; ++l)
			{
				TensorGatedState::Hidden& hl = tensorGru.H[static_cast<size_t>(l)];
				std::fill(hl.gW.begin(), hl.gW.end(), 0.0f);
				std::fill(hl.gU.begin(), hl.gU.end(), 0.0f);
				std::fill(hl.gBias.begin(), hl.gBias.end(), 0.0f);
			}
			std::fill(tensorGru.O.gWhy.begin(), tensorGru.O.gWhy.end(), 0.0f);
			std::fill(tensorGru.O.gBias.begin(), tensorGru.O.gBias.end(), 0.0f);
		}
	};
	ClearGrads clearGrads(tensorGru, H);

	struct ApplyBatch
	{
		NNInfo* skeleton;
		float& lrScheduleMultiplier;
		TrainingConfig& trainingConfig;
		float& lastGradNorm;
		float& lastGradNormScale;
		NNetworkStatus& lastStatus;
		bool& running;
		TensorGatedState& tensorGru;
		int H;
		unsigned int outSize;

		ApplyBatch(NNetwork& net, TensorGatedState& tg, int h, unsigned int os)
		    : skeleton(net.skeleton),
		      lrScheduleMultiplier(net.lrScheduleMultiplier),
		      trainingConfig(net.trainingConfig),
		      lastGradNorm(net.lastGradNorm),
		      lastGradNormScale(net.lastGradNormScale),
		      lastStatus(net.lastStatus),
		      running(net.running),
		      tensorGru(tg),
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

			// Output transition hyperparams live at index == H
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
				// Output weights
				for (size_t gidx = 0; gidx < tensorGru.O.Why.size(); ++gidx)
				{
					float gg = tensorGru.O.gWhy[gidx] * invBatch;
					if ((wd1Out != 0.0f) || (wd2Out != 0.0f))
					{
						const float w = tensorGru.O.Why[gidx];
						const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
						gg += (wd1Out * wSign) + (wd2Out * w);
					}
					const double gd = static_cast<double>(gg);
					sumsq += gd * gd;
				}
				for (unsigned int k2 = 0; k2 < outSize; ++k2)
				{
					const float gB2 = tensorGru.O.gBias[k2] * invBatch;
					const double bd = static_cast<double>(gB2);
					sumsq += bd * bd;
				}
				// Hidden layers
				for (int l2 = 0; l2 < H; ++l2)
				{
					const unsigned int li2 = static_cast<unsigned int>(l2);
					const TensorGatedState::Hidden& hl2 = tensorGru.H[static_cast<size_t>(l2)];
					const float wd1 = skeleton->getWeightDecay1(li2);
					const float wd2 = skeleton->getWeightDecay2(li2);
					for (size_t gidx = 0; gidx < hl2.W.size(); ++gidx)
					{
						float gg = hl2.gW[gidx] * invBatch;
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = hl2.W[gidx];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							gg += (wd1 * wSign) + (wd2 * w);
						}
						const double gd = static_cast<double>(gg);
						sumsq += gd * gd;
					}
					for (size_t gidx = 0; gidx < hl2.U.size(); ++gidx)
					{
						float gg = hl2.gU[gidx] * invBatch;
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = hl2.U[gidx];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							gg += (wd1 * wSign) + (wd2 * w);
						}
						const double gd = static_cast<double>(gg);
						sumsq += gd * gd;
					}
					for (size_t bi2 = 0; bi2 < hl2.gBias.size(); ++bi2)
					{
						const float gB2 = hl2.gBias[bi2] * invBatch;
						const double bd = static_cast<double>(gB2);
						sumsq += bd * bd;
					}
				}
				if (!is_finite_double(sumsq))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite grad-norm accumulation detected (NaN/Inf)");
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
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite grad clipping metadata detected (NaN/Inf)");
				running = false;
				return false;
			}

			// Output weights
			for (size_t idx = 0; idx < tensorGru.O.Why.size(); ++idx)
			{
				float g = tensorGru.O.gWhy[idx] * invBatch;
				if ((wd1Out != 0.0f) || (wd2Out != 0.0f))
				{
					const float w = tensorGru.O.Why[idx];
					const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
					g += (wd1Out * wSign) + (wd2Out * w);
				}
				g *= gradScale;
				const float v = (mfOut * tensorGru.O.vWhy[idx]) + (lrOut * g);
				tensorGru.O.vWhy[idx] = v;
				tensorGru.O.Why[idx] -= v;
				tensorGru.O.gWhy[idx] = 0.0f;
				if (!is_finite(tensorGru.O.Why[idx]))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite output weight after tensor update (NaN/Inf)");
					running = false;
					return false;
				}
			}
			for (unsigned int k = 0; k < outSize; ++k)
			{
				float gB = tensorGru.O.gBias[k] * invBatch;
				gB *= gradScale;
				tensorGru.O.bias[k] -= (lrOut * gB);
				tensorGru.O.gBias[k] = 0.0f;
				if (!is_finite(tensorGru.O.bias[k]))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite output bias after tensor update (NaN/Inf)");
					running = false;
					return false;
				}
			}

			// Hidden weights
			for (int l = 0; l < H; ++l)
			{
				const unsigned int li = static_cast<unsigned int>(l);
				TensorGatedState::Hidden& hl = tensorGru.H[static_cast<size_t>(l)];
				const float lr = skeleton->getLearningRate(li) * lrScheduleMultiplier;
				const float mf = skeleton->getMomentumFactor(li);
				const float wd1 = skeleton->getWeightDecay1(li);
				const float wd2 = skeleton->getWeightDecay2(li);

				for (size_t idx = 0; idx < hl.W.size(); ++idx)
				{
					float g = hl.gW[idx] * invBatch;
					if ((wd1 != 0.0f) || (wd2 != 0.0f))
					{
						const float w = hl.W[idx];
						const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
						g += (wd1 * wSign) + (wd2 * w);
					}
					g *= gradScale;
					const float v = (mf * hl.vW[idx]) + (lr * g);
					hl.vW[idx] = v;
					hl.W[idx] -= v;
					hl.gW[idx] = 0.0f;
					if (!is_finite(hl.W[idx]))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite hidden W after tensor update (NaN/Inf)");
						running = false;
						return false;
					}
				}
				for (size_t idx = 0; idx < hl.U.size(); ++idx)
				{
					float g = hl.gU[idx] * invBatch;
					if ((wd1 != 0.0f) || (wd2 != 0.0f))
					{
						const float w = hl.U[idx];
						const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
						g += (wd1 * wSign) + (wd2 * w);
					}
					g *= gradScale;
					const float v = (mf * hl.vU[idx]) + (lr * g);
					hl.vU[idx] = v;
					hl.U[idx] -= v;
					hl.gU[idx] = 0.0f;
					if (!is_finite(hl.U[idx]))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite hidden U after tensor update (NaN/Inf)");
						running = false;
						return false;
					}
				}
				for (size_t bi = 0; bi < hl.bias.size(); ++bi)
				{
					float gB = hl.gBias[bi] * invBatch;
					gB *= gradScale;
					hl.bias[bi] -= (lr * gB);
					hl.gBias[bi] = 0.0f;
					if (!is_finite(hl.bias[bi]))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite hidden bias after tensor update (NaN/Inf)");
						running = false;
						return false;
					}
				}
			}

			// Defer syncing tensors -> Node/Edge graph until an epoch boundary.
			return true;
		}
	};
	ApplyBatch applyBatch(*this, tensorGru, H, outSize);

	// Gate layout conventions:
	// - Node edges per hidden unit: gateCount*(prevSize+1)
	// - Context node edges per hidden unit: gateCount*hiddenSize
	const unsigned int gateCount = 3;
	const unsigned int GZ = 0;
	const unsigned int GR = 1;
	const unsigned int GH = 2;

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

	for (unsigned int s = 0; s < seqCount; ++s)
	{
		const unsigned int seqLen = isTrain ? di->getTrainSequenceLength(s) : di->getTestSequenceLength(s);
		if (seqLen == 0)
			continue;
		tokensProcessed += static_cast<unsigned long long>(seqLen);

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
			recScratch.ensureGRU(winLen, inputSize, outSize, hiddenSizes);
			std::vector<float>& xFlat = recScratch.x;
			std::vector<float>& yFlat = recScratch.y;
			std::vector< std::vector<float> >& hFlat = recScratch.h;
			std::vector< std::vector<float> >& hPrevAtTFlat = recScratch.hPrevAtT;
			std::vector< std::vector<float> >& zFlat = recScratch.z;
			std::vector< std::vector<float> >& rFlat = recScratch.r;
			std::vector< std::vector<float> >& hTildeFlat = recScratch.hTilde;

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
						lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_GRU: non-finite input sequence value detected (NaN/Inf)");
						running = false;
						return;
					}
					xFlat[static_cast<size_t>(t) * static_cast<size_t>(inputSize) + p] = v;
				}

				for (int l = 0; l < H; ++l)
				{
					const unsigned int curSize = hiddenSizes[l];
					const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[l - 1];
					const Gated layout = {prevSize, curSize, gateCount};
					const size_t tOffCur = static_cast<size_t>(t) * static_cast<size_t>(curSize);
					TensorGatedState::Hidden& th = tensorGru.H[static_cast<size_t>(l)];

					for (unsigned int i = 0; i < curSize; ++i)
						hPrevAtTFlat[l][tOffCur + i] = hPrev[l][i];

					// Compute z and r for all units first
					for (unsigned int i = 0; i < curSize; ++i)
					{
						// z gate
						float netZ = th.bias[static_cast<size_t>(GZ) * static_cast<size_t>(curSize) + static_cast<size_t>(i)];
						const size_t wzOff = (static_cast<size_t>(GZ) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) *
						                     static_cast<size_t>(prevSize);
						for (unsigned int p = 0; p < prevSize; ++p)
						{
							const float inAct =
								(l == 0)
									? xFlat[static_cast<size_t>(t) * static_cast<size_t>(inputSize) + p]
									: hFlat[l - 1][static_cast<size_t>(t) * static_cast<size_t>(hiddenSizes[l - 1]) + p];
							netZ += th.W[wzOff + p] * inAct;
						}
						const size_t uzOff = (static_cast<size_t>(GZ) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) *
						                     static_cast<size_t>(curSize);
						for (unsigned int j = 0; j < curSize; ++j)
							netZ += th.U[uzOff + j] * hPrev[l][j];
						if (!is_finite(netZ))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite z-gate pre-activation detected (NaN/Inf)");
							running = false;
							return;
						}
						zFlat[l][tOffCur + i] = GMath::squash(netZ, GMath::SIGMOID, 0.0f);
						if (!is_finite(zFlat[l][tOffCur + i]))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite z-gate activation detected (NaN/Inf)");
							running = false;
							return;
						}

						// r gate
						float netR = th.bias[static_cast<size_t>(GR) * static_cast<size_t>(curSize) + static_cast<size_t>(i)];
						const size_t wrOff = (static_cast<size_t>(GR) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) *
						                     static_cast<size_t>(prevSize);
						for (unsigned int p = 0; p < prevSize; ++p)
						{
							const float inAct =
								(l == 0)
									? xFlat[static_cast<size_t>(t) * static_cast<size_t>(inputSize) + p]
									: hFlat[l - 1][static_cast<size_t>(t) * static_cast<size_t>(hiddenSizes[l - 1]) + p];
							netR += th.W[wrOff + p] * inAct;
						}
						const size_t urOff = (static_cast<size_t>(GR) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) *
						                     static_cast<size_t>(curSize);
						for (unsigned int j = 0; j < curSize; ++j)
							netR += th.U[urOff + j] * hPrev[l][j];
						if (!is_finite(netR))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite r-gate pre-activation detected (NaN/Inf)");
							running = false;
							return;
						}
						rFlat[l][tOffCur + i] = GMath::squash(netR, GMath::SIGMOID, 0.0f);
						if (!is_finite(rFlat[l][tOffCur + i]))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite r-gate activation detected (NaN/Inf)");
							running = false;
							return;
						}
					}

					// Candidate and new hidden
					for (unsigned int i = 0; i < curSize; ++i)
					{
						float netH = th.bias[static_cast<size_t>(GH) * static_cast<size_t>(curSize) + static_cast<size_t>(i)];
						const size_t whOff = (static_cast<size_t>(GH) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) *
						                     static_cast<size_t>(prevSize);
						for (unsigned int p = 0; p < prevSize; ++p)
						{
							const float inAct =
								(l == 0)
									? xFlat[static_cast<size_t>(t) * static_cast<size_t>(inputSize) + p]
									: hFlat[l - 1][static_cast<size_t>(t) * static_cast<size_t>(hiddenSizes[l - 1]) + p];
							netH += th.W[whOff + p] * inAct;
						}
						const size_t uhOff = (static_cast<size_t>(GH) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) *
						                     static_cast<size_t>(curSize);
						for (unsigned int j = 0; j < curSize; ++j)
						{
							const float rh = rFlat[l][tOffCur + j] * hPrev[l][j];
							netH += th.U[uhOff + j] * rh;
						}
						if (!is_finite(netH))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite candidate pre-activation detected (NaN/Inf)");
							running = false;
							return;
						}
						hTildeFlat[l][tOffCur + i] = GMath::squash(netH, GMath::TANH, 0.0f);
						if (!is_finite(hTildeFlat[l][tOffCur + i]))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite candidate activation detected (NaN/Inf)");
							running = false;
							return;
						}

						const float zi = zFlat[l][tOffCur + i];
						const float hPrevI = hPrev[l][i];
						const float hi = (1.0f - zi) * hPrevI + zi * hTildeFlat[l][tOffCur + i];
						if (!is_finite(hi))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite hidden activation detected (NaN/Inf)");
							running = false;
							return;
						}
						hFlat[l][tOffCur + i] = hi;
					}

					for (unsigned int i = 0; i < curSize; ++i)
						hPrev[l][i] = hFlat[l][tOffCur + i];
				}

				// Output layer (same as RNN path)
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
						float net = (k < tensorGru.O.bias.size()) ? tensorGru.O.bias[k] : 0.0f;
						const size_t rowOff = static_cast<size_t>(k) * static_cast<size_t>(prevSize);
						const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(prevSize);
						for (unsigned int i = 0; i < prevSize; ++i)
							net += tensorGru.O.Why[rowOff + i] * hFlat[H - 1][hOff + i];
						if (!is_finite(net))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite output pre-activation detected (NaN/Inf)");
							running = false;
							return;
						}

						if (useSoftmax)
							outLogits[k] = net;
						else
						{
							const float a = GMath::squash(net, actFx, actParam);
							if (!is_finite(a))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite output activation detected (NaN/Inf)");
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
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite softmax probability detected (NaN/Inf)");
								running = false;
								return;
							}
							yFlat[tOffOut + k] = p;
						}
					}
				}

				// Metrics + confusion matrix for this timestep (same as RNN path)
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
							lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_GRU: non-finite expected sequence value detected (NaN/Inf)");
							running = false;
							return;
						}
						if (!is_finite(prediction))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite prediction detected (NaN/Inf)");
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
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite loss aggregate detected (NaN/Inf)");
						running = false;
						return;
					}

					if ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL))
						confusionMatrix.addResult(results);
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
				const float invWinLen = (winLen > 0u) ? (1.0f / static_cast<float>(winLen)) : 1.0f;

				std::fill(deltaY.begin(), deltaY.end(), 0.0f);
				// nextTimeDh[l][i] = dL/dh_{t+1,i} carried back through recurrence
				for (int l = 0; l < H; ++l)
					std::fill(nextTimeDh[l].begin(), nextTimeDh[l].end(), 0.0f);

				// dInputFromAbove is the gradient w.r.t. the current layer's input (x or h_{l-1})
				// coming from the layer above at the same timestep.
				dInputFromAbove = NULL;

				for (int t = static_cast<int>(winLen) - 1; t >= 0; --t)
				{
					const size_t tOffOut = static_cast<size_t>(t) * static_cast<size_t>(outSize);

					// Output layer deltas + updates
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
								lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_GRU: non-finite expected value detected in BPTT (NaN/Inf)");
								running = false;
								return;
							}
							if (!is_finite(pred))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite prediction detected in BPTT (NaN/Inf)");
								running = false;
								return;
							}
							const bool useSoftmax =
							    ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1);
							if (useSoftmax)
								deltaY[k] = clipf_maybe(pred - expv, gradClip);
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
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite output delta detected (NaN/Inf)");
								running = false;
								return;
							}

							// Accumulate output gradients (Why and bias).
							const size_t rowOff = static_cast<size_t>(k) * static_cast<size_t>(outIn);
							const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(outIn);
							for (unsigned int i = 0; i < outIn; ++i)
								tensorGru.O.gWhy[rowOff + i] += (deltaY[k] * invWinLen) * hFlat[H - 1][hOff + i];
							tensorGru.O.gBias[k] += (deltaY[k] * invWinLen);
						}
					}

					// Backprop through hidden layers (top-down) at this timestep.
					dInputFromAbove = NULL;
					for (int l = H - 1; l >= 0; --l)
					{
						const unsigned int curSize = hiddenSizes[l];
						const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[l - 1];
						const Gated layout = {prevSize, curSize, gateCount};
						(void)layout; // layout is kept for readability; indices use packed tensors directly.
						const size_t tOffCur = static_cast<size_t>(t) * static_cast<size_t>(curSize);
						TensorGatedState::Hidden& thl = tensorGru.H[static_cast<size_t>(l)];

						// dh = dL/dh_t (from above at same time + from future via recurrence)
						std::vector<float>& dh = dhBuf[l];
						std::fill(dh.begin(), dh.end(), 0.0f);

						if (l == H - 1)
						{
							// From output layer
							const unsigned int outIn = hiddenSizes[H - 1];
							for (unsigned int i = 0; i < curSize; ++i)
							{
								float sum = 0.0f;
								for (unsigned int k = 0; k < outSize; ++k)
									sum += deltaY[k] * tensorGru.O.Why[static_cast<size_t>(k) * static_cast<size_t>(outIn) + i];
								dh[i] = sum;
							}
						}
						else
						{
							// From layer above (already computed as gradient w.r.t this layer's activations)
							if (dInputFromAbove && dInputFromAbove->size() == curSize)
							{
								for (unsigned int i = 0; i < curSize; ++i)
									dh[i] = (*dInputFromAbove)[i];
							}
						}
						// Add recurrence-from-future
						for (unsigned int i = 0; i < curSize; ++i)
							dh[i] += nextTimeDh[l][i];

						// Gate derivatives
						std::vector<float>& daZ = daZBuf[l];
						std::vector<float>& daR = daRBuf[l];
						std::vector<float>& daH = daHBuf[l];
						std::fill(daZ.begin(), daZ.end(), 0.0f);
						std::fill(daR.begin(), daR.end(), 0.0f);
						std::fill(daH.begin(), daH.end(), 0.0f);

						// dL/dz and dL/dhTilde, plus direct path to hPrev
						std::vector<float>& dhPrevLocal = dhPrevBuf[l];
						std::fill(dhPrevLocal.begin(), dhPrevLocal.end(), 0.0f);
						for (unsigned int i = 0; i < curSize; ++i)
						{
							const float zi = zFlat[l][tOffCur + i];
							const float hiT = hTildeFlat[l][tOffCur + i];
							const float hPrevI = hPrevAtTFlat[l][tOffCur + i];
							const float dZ = dh[i] * (hiT - hPrevI);
							const float dHT = dh[i] * zi;
							dhPrevLocal[i] += dh[i] * (1.0f - zi);

							// da = dOut * dActivation
							daZ[i] = clipf_maybe(dZ * GMath::activationErrDer(zi, GMath::SIGMOID, 0.0f), gradClip);
							daH[i] = clipf_maybe(dHT * GMath::activationErrDer(hiT, GMath::TANH, 0.0f), gradClip);
								if (!is_finite(daZ[i]) || !is_finite(daH[i]))
								{
									lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite gate derivatives detected (NaN/Inf)");
									running = false;
									return;
								}
						}

						// Candidate recurrent backprop: tmp = Uh^T * daH
						std::vector<float>& tmp = tmpBuf[l];
						std::fill(tmp.begin(), tmp.end(), 0.0f);
						for (unsigned int j = 0; j < curSize; ++j)
						{
							float sum = 0.0f;
							for (unsigned int i = 0; i < curSize; ++i)
								sum += daH[i] * thl.U[(static_cast<size_t>(GH) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(curSize) + static_cast<size_t>(j)];
							tmp[j] = sum;
						}

						// dr and additional dhPrev from candidate path
						for (unsigned int j = 0; j < curSize; ++j)
						{
							const float rj = rFlat[l][tOffCur + j];
							const float hPrevJ = hPrevAtTFlat[l][tOffCur + j];
							dhPrevLocal[j] += tmp[j] * rj;
							const float dR = tmp[j] * hPrevJ;
							daR[j] = clipf_maybe(dR * GMath::activationErrDer(rj, GMath::SIGMOID, 0.0f), gradClip);
								if (!is_finite(daR[j]))
								{
									lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_GRU: non-finite r-gate derivative detected (NaN/Inf)");
									running = false;
									return;
								}
						}

						// Recurrent contributions from z and r gates: Uz^T*daZ + Ur^T*daR
						for (unsigned int j = 0; j < curSize; ++j)
						{
							float add = 0.0f;
							for (unsigned int i = 0; i < curSize; ++i)
							{
								add += daZ[i] * thl.U[(static_cast<size_t>(GZ) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(curSize) + static_cast<size_t>(j)];
								add += daR[i] * thl.U[(static_cast<size_t>(GR) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(curSize) + static_cast<size_t>(j)];
							}
							dhPrevLocal[j] += add;
						}

						// Compute gradient to this layer's input (prev layer at same timestep)
						std::vector<float>& dInput = dInputBuf[l];
						std::fill(dInput.begin(), dInput.end(), 0.0f);
						for (unsigned int p = 0; p < prevSize; ++p)
						{
							float sum = 0.0f;
							for (unsigned int i = 0; i < curSize; ++i)
							{
								sum += daZ[i] * thl.W[(static_cast<size_t>(GZ) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(prevSize) + static_cast<size_t>(p)];
								sum += daR[i] * thl.W[(static_cast<size_t>(GR) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(prevSize) + static_cast<size_t>(p)];
								sum += daH[i] * thl.W[(static_cast<size_t>(GH) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(prevSize) + static_cast<size_t>(p)];
							}
							dInput[p] = sum;
						}
						// Make this layer's input-gradient visible to the layer below.
						dInputFromAbove = &dInput;

						// Accumulate weight gradients (W and U, plus biases) for this layer at this timestep
						for (unsigned int i = 0; i < curSize; ++i)
						{
							const size_t wzOff = (static_cast<size_t>(GZ) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(prevSize);
							const size_t wrOff = (static_cast<size_t>(GR) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(prevSize);
							const size_t whOff = (static_cast<size_t>(GH) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(prevSize);

							for (unsigned int p = 0; p < prevSize; ++p)
							{
								const float inAct =
								    (l == 0)
								        ? xFlat[static_cast<size_t>(t) * static_cast<size_t>(inputSize) + p]
								        : hFlat[l - 1][static_cast<size_t>(t) * static_cast<size_t>(hiddenSizes[l - 1]) + p];
								thl.gW[wzOff + p] += (daZ[i] * invWinLen) * inAct;
								thl.gW[wrOff + p] += (daR[i] * invWinLen) * inAct;
								thl.gW[whOff + p] += (daH[i] * invWinLen) * inAct;
							}

							thl.gBias[static_cast<size_t>(GZ) * static_cast<size_t>(curSize) + static_cast<size_t>(i)] += (daZ[i] * invWinLen);
							thl.gBias[static_cast<size_t>(GR) * static_cast<size_t>(curSize) + static_cast<size_t>(i)] += (daR[i] * invWinLen);
							thl.gBias[static_cast<size_t>(GH) * static_cast<size_t>(curSize) + static_cast<size_t>(i)] += (daH[i] * invWinLen);

							const size_t uzOff = (static_cast<size_t>(GZ) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(curSize);
							const size_t urOff = (static_cast<size_t>(GR) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(curSize);
							const size_t uhOff = (static_cast<size_t>(GH) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(curSize);
							for (unsigned int j = 0; j < curSize; ++j)
							{
								const float prevH = hPrevAtTFlat[l][tOffCur + j];
								const float rj = rFlat[l][tOffCur + j];
								thl.gU[uzOff + j] += (daZ[i] * invWinLen) * prevH;
								thl.gU[urOff + j] += (daR[i] * invWinLen) * prevH;
								thl.gU[uhOff + j] += (daH[i] * invWinLen) * (rj * prevH);
							}
						}

						// Carry recurrence gradient to previous timestep
						nextTimeDh[l] = dhPrevLocal;
					} // l
				} // t

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
				append_logfmt_kv(oss, "grad_norm", lastGradNorm);
				append_logfmt_kv(oss, "grad_norm_scale", lastGradNormScale);
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

