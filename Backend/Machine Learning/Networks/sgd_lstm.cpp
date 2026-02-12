// Net-type-specific SGD: LSTM BPTT path (split out of network.cpp)
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

void glades::NNetwork::SGDHelper_LSTM(unsigned int inputRowCounter, int runType)
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
		    isTrain ? "SGDHelper_LSTM: no train sequences (DataInput::getTrainSequenceCount() == 0)"
		            : "SGDHelper_LSTM: no test sequences (DataInput::getTestSequenceCount() == 0)");
		running = false;
		return;
	}

	const int H = skeleton->numHiddenLayers();
	const unsigned int outSize = skeleton->getOutputLayerSize();
	const unsigned int inputSize = di->getFeatureCount();
	if (H <= 0 || outSize == 0 || inputSize == 0)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_LSTM: invalid layer sizes (hidden/output/input size == 0)");
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
	if (!tensorLstm.initialized)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_LSTM: tensor state not initialized");
		running = false;
		return;
	}

	// Sizes from tensor state (authoritative).
	std::vector<unsigned int> hiddenSizes = tensorLstm.hiddenSizes;

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
	std::vector< std::vector<float> > nextTimeDc;
	nextTimeDh.resize(H);
	nextTimeDc.resize(H);
	for (int l = 0; l < H; ++l)
	{
		nextTimeDh[l].assign(hiddenSizes[l], 0.0f);
		nextTimeDc[l].assign(hiddenSizes[l], 0.0f);
	}

	// === Backward-pass scratch buffers ===
	// Historically these were allocated per-timestep. Preallocate and reuse to avoid
	// inner-loop heap churn.
	std::vector< std::vector<float> > dhBuf;
	std::vector< std::vector<float> > daIBuf;
	std::vector< std::vector<float> > daFBuf;
	std::vector< std::vector<float> > daOBuf;
	std::vector< std::vector<float> > daGBuf;
	std::vector< std::vector<float> > dcBuf;
	std::vector< std::vector<float> > dcPrevBuf;
	std::vector< std::vector<float> > dhPrevBuf;
	std::vector< std::vector<float> > dInputBuf;
	dhBuf.resize(H);
	daIBuf.resize(H);
	daFBuf.resize(H);
	daOBuf.resize(H);
	daGBuf.resize(H);
	dcBuf.resize(H);
	dcPrevBuf.resize(H);
	dhPrevBuf.resize(H);
	dInputBuf.resize(H);
	for (int l = 0; l < H; ++l)
	{
		const unsigned int curSize = hiddenSizes[l];
		const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[l - 1];
		dhBuf[l].resize(curSize);
		daIBuf[l].resize(curSize);
		daFBuf[l].resize(curSize);
		daOBuf[l].resize(curSize);
		daGBuf[l].resize(curSize);
		dcBuf[l].resize(curSize);
		dcPrevBuf[l].resize(curSize);
		dhPrevBuf[l].resize(curSize);
		dInputBuf[l].resize(prevSize);
	}
	const std::vector<float>* dInputFromAbove = NULL;

	// Cache expected outputs for the current TBPTT window to avoid allocating/copying
	// expected rows again during backward.
	std::vector<float> expFlat;

	// === Minibatch application helpers (replaces macro-based control flow) ===
	struct ClearGrads
	{
		TensorGatedState& tensorLstm;
		int H;
		ClearGrads(TensorGatedState& tg, int h) : tensorLstm(tg), H(h) {}
		void operator()() const
		{
			for (int l = 0; l < H; ++l)
			{
				TensorGatedState::Hidden& hl = tensorLstm.H[static_cast<size_t>(l)];
				std::fill(hl.gW.begin(), hl.gW.end(), 0.0f);
				std::fill(hl.gU.begin(), hl.gU.end(), 0.0f);
				std::fill(hl.gBias.begin(), hl.gBias.end(), 0.0f);
			}
			std::fill(tensorLstm.O.gWhy.begin(), tensorLstm.O.gWhy.end(), 0.0f);
			std::fill(tensorLstm.O.gBias.begin(), tensorLstm.O.gBias.end(), 0.0f);
		}
	};
	ClearGrads clearGrads(tensorLstm, H);

	struct ApplyBatch
	{
		NNInfo* skeleton;
		float& lrScheduleMultiplier;
		TrainingConfig& trainingConfig;
		float& lastGradNorm;
		float& lastGradNormScale;
		NNetworkStatus& lastStatus;
		bool& running;
		TensorGatedState& tensorLstm;
		int H;
		unsigned int outSize;

		ApplyBatch(NNetwork& net, TensorGatedState& tl, int h, unsigned int os)
		    : skeleton(net.skeleton),
		      lrScheduleMultiplier(net.lrScheduleMultiplier),
		      trainingConfig(net.trainingConfig),
		      lastGradNorm(net.lastGradNorm),
		      lastGradNormScale(net.lastGradNormScale),
		      lastStatus(net.lastStatus),
		      running(net.running),
		      tensorLstm(tl),
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

			float gradNorm = 0.0f;
			float gradScale = 1.0f;
			const float clipNorm = trainingConfig.globalGradClipNorm;
			if (clipNorm > 0.0f)
			{
				double sumsq = 0.0;
				for (size_t gidx = 0; gidx < tensorLstm.O.Why.size(); ++gidx)
				{
					float gg = tensorLstm.O.gWhy[gidx] * invBatch;
					if ((wd1Out != 0.0f) || (wd2Out != 0.0f))
					{
						const float w = tensorLstm.O.Why[gidx];
						const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
						gg += (wd1Out * wSign) + (wd2Out * w);
					}
					const double gd = static_cast<double>(gg);
					sumsq += gd * gd;
				}
				for (unsigned int k2 = 0; k2 < outSize; ++k2)
				{
					const float gB2 = tensorLstm.O.gBias[k2] * invBatch;
					const double bd = static_cast<double>(gB2);
					sumsq += bd * bd;
				}
				for (int l2 = 0; l2 < H; ++l2)
				{
					const unsigned int li2 = static_cast<unsigned int>(l2);
					const TensorGatedState::Hidden& hl2 = tensorLstm.H[static_cast<size_t>(l2)];
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
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite grad-norm accumulation detected (NaN/Inf)");
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
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite grad clipping metadata detected (NaN/Inf)");
				running = false;
				return false;
			}

			for (size_t idx = 0; idx < tensorLstm.O.Why.size(); ++idx)
			{
				float g = tensorLstm.O.gWhy[idx] * invBatch;
				if ((wd1Out != 0.0f) || (wd2Out != 0.0f))
				{
					const float w = tensorLstm.O.Why[idx];
					const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
					g += (wd1Out * wSign) + (wd2Out * w);
				}
				g *= gradScale;
				const float v = (mfOut * tensorLstm.O.vWhy[idx]) + (lrOut * g);
				tensorLstm.O.vWhy[idx] = v;
				tensorLstm.O.Why[idx] -= v;
				tensorLstm.O.gWhy[idx] = 0.0f;
				if (!is_finite(tensorLstm.O.Why[idx]))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite output weight after tensor update (NaN/Inf)");
					running = false;
					return false;
				}
			}
			for (unsigned int k = 0; k < outSize; ++k)
			{
				float gB = tensorLstm.O.gBias[k] * invBatch;
				gB *= gradScale;
				tensorLstm.O.bias[k] -= (lrOut * gB);
				tensorLstm.O.gBias[k] = 0.0f;
				if (!is_finite(tensorLstm.O.bias[k]))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite output bias after tensor update (NaN/Inf)");
					running = false;
					return false;
				}
			}

			for (int l = 0; l < H; ++l)
			{
				const unsigned int li = static_cast<unsigned int>(l);
				TensorGatedState::Hidden& hl = tensorLstm.H[static_cast<size_t>(l)];
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
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite hidden W after tensor update (NaN/Inf)");
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
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite hidden U after tensor update (NaN/Inf)");
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
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite hidden bias after tensor update (NaN/Inf)");
						running = false;
						return false;
					}
				}
			}

			return true;
		}
	};
	ApplyBatch applyBatch(*this, tensorLstm, H, outSize);

	const unsigned int gateCount = 4;
	const unsigned int GI = 0;
	const unsigned int GF = 1;
	const unsigned int GO = 2;
	const unsigned int GG = 3;

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
		std::vector< std::vector<float> > cPrev;
		hPrev.resize(H);
		cPrev.resize(H);
		for (int l = 0; l < H; ++l)
		{
			hPrev[l].assign(hiddenSizes[l], 0.0f);
			cPrev[l].assign(hiddenSizes[l], 0.0f);
		}

		unsigned int t0 = 0;
		while (t0 < seqLen)
		{
			const unsigned int winLen = (tbptt > 0) ? std::min(tbptt, seqLen - t0) : (seqLen - t0);
			if (winLen == 0)
				break;

			// Preallocate scratch buffers for this window
			recScratch.ensureLSTM(winLen, inputSize, outSize, hiddenSizes);
			std::vector<float>& xFlat = recScratch.x;
			std::vector<float>& yFlat = recScratch.y;
			std::vector< std::vector<float> >& hFlat = recScratch.h;
			std::vector< std::vector<float> >& hPrevAtTFlat = recScratch.hPrevAtT;
			std::vector< std::vector<float> >& cFlat = recScratch.c;
			std::vector< std::vector<float> >& cPrevAtTFlat = recScratch.cPrevAtT;
			std::vector< std::vector<float> >& iGateFlat = recScratch.iGate;
			std::vector< std::vector<float> >& fGateFlat = recScratch.fGate;
			std::vector< std::vector<float> >& oGateFlat = recScratch.oGate;
			std::vector< std::vector<float> >& gGateFlat = recScratch.gGate;
			std::vector< std::vector<float> >& tanhCFlat = recScratch.tanhC;

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
						lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_LSTM: non-finite input sequence value detected (NaN/Inf)");
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
					TensorGatedState::Hidden& th = tensorLstm.H[static_cast<size_t>(l)];

					for (unsigned int i = 0; i < curSize; ++i)
					{
						hPrevAtTFlat[l][tOffCur + i] = hPrev[l][i];
						cPrevAtTFlat[l][tOffCur + i] = cPrev[l][i];
					}

					for (unsigned int u = 0; u < curSize; ++u)
					{
						float netI = th.bias[static_cast<size_t>(GI) * static_cast<size_t>(curSize) + static_cast<size_t>(u)];
						float netF = th.bias[static_cast<size_t>(GF) * static_cast<size_t>(curSize) + static_cast<size_t>(u)];
						float netO = th.bias[static_cast<size_t>(GO) * static_cast<size_t>(curSize) + static_cast<size_t>(u)];
						float netG = th.bias[static_cast<size_t>(GG) * static_cast<size_t>(curSize) + static_cast<size_t>(u)];
						const size_t wiOff = (static_cast<size_t>(GI) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize);
						const size_t wfOff = (static_cast<size_t>(GF) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize);
						const size_t woOff = (static_cast<size_t>(GO) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize);
						const size_t wgOff = (static_cast<size_t>(GG) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize);
						for (unsigned int p = 0; p < prevSize; ++p)
						{
							const float inAct =
								(l == 0)
									? xFlat[static_cast<size_t>(t) * static_cast<size_t>(inputSize) + p]
									: hFlat[l - 1][static_cast<size_t>(t) * static_cast<size_t>(hiddenSizes[l - 1]) + p];
							netI += th.W[wiOff + p] * inAct;
							netF += th.W[wfOff + p] * inAct;
							netO += th.W[woOff + p] * inAct;
							netG += th.W[wgOff + p] * inAct;
						}
						const size_t uiOff = (static_cast<size_t>(GI) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize);
						const size_t ufOff = (static_cast<size_t>(GF) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize);
						const size_t uoOff = (static_cast<size_t>(GO) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize);
						const size_t ugOff = (static_cast<size_t>(GG) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize);
						for (unsigned int j = 0; j < curSize; ++j)
						{
							const float hp = hPrev[l][j];
							netI += th.U[uiOff + j] * hp;
							netF += th.U[ufOff + j] * hp;
							netO += th.U[uoOff + j] * hp;
							netG += th.U[ugOff + j] * hp;
						}

						if (!is_finite(netI) || !is_finite(netF) || !is_finite(netO) || !is_finite(netG))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite gate pre-activation detected (NaN/Inf)");
							running = false;
							return;
						}
						iGateFlat[l][tOffCur + u] = GMath::squash(netI, GMath::SIGMOID, 0.0f);
						fGateFlat[l][tOffCur + u] = GMath::squash(netF, GMath::SIGMOID, 0.0f);
						oGateFlat[l][tOffCur + u] = GMath::squash(netO, GMath::SIGMOID, 0.0f);
						gGateFlat[l][tOffCur + u] = GMath::squash(netG, GMath::TANH, 0.0f);
						if (!is_finite(iGateFlat[l][tOffCur + u]) || !is_finite(fGateFlat[l][tOffCur + u]) ||
						    !is_finite(oGateFlat[l][tOffCur + u]) || !is_finite(gGateFlat[l][tOffCur + u]))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite gate activation detected (NaN/Inf)");
							running = false;
							return;
						}

						const float cNew = fGateFlat[l][tOffCur + u] * cPrev[l][u] + iGateFlat[l][tOffCur + u] * gGateFlat[l][tOffCur + u];
						if (!is_finite(cNew))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite cell state detected (NaN/Inf)");
							running = false;
							return;
						}
						cFlat[l][tOffCur + u] = cNew;
						tanhCFlat[l][tOffCur + u] = GMath::squash(cNew, GMath::TANH, 0.0f);
						if (!is_finite(tanhCFlat[l][tOffCur + u]))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite tanh(cell) detected (NaN/Inf)");
							running = false;
							return;
						}
						const float hNew = oGateFlat[l][tOffCur + u] * tanhCFlat[l][tOffCur + u];
						if (!is_finite(hNew))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite hidden activation detected (NaN/Inf)");
							running = false;
							return;
						}
						hFlat[l][tOffCur + u] = hNew;
					}

					for (unsigned int i = 0; i < curSize; ++i)
					{
						hPrev[l][i] = hFlat[l][tOffCur + i];
						cPrev[l][i] = cFlat[l][tOffCur + i];
					}
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
						float net = (k < tensorLstm.O.bias.size()) ? tensorLstm.O.bias[k] : 0.0f;
						const size_t rowOff = static_cast<size_t>(k) * static_cast<size_t>(prevSize);
						const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(prevSize);
						for (unsigned int i = 0; i < prevSize; ++i)
							net += tensorLstm.O.Why[rowOff + i] * hFlat[H - 1][hOff + i];
						if (!is_finite(net))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite output pre-activation detected (NaN/Inf)");
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
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite output activation detected (NaN/Inf)");
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
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite softmax probability detected (NaN/Inf)");
								running = false;
								return;
							}
							yFlat[tOffOut + k] = p;
						}
					}
				}

				// Metrics + confusion matrix
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
							lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_LSTM: non-finite expected sequence value detected (NaN/Inf)");
							running = false;
							return;
						}
						if (!is_finite(prediction))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite prediction detected (NaN/Inf)");
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
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite loss aggregate detected (NaN/Inf)");
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
				const float invWinLen = (winLen > 0u) ? (1.0f / static_cast<float>(winLen)) : 1.0f;

				std::fill(deltaY.begin(), deltaY.end(), 0.0f);
				for (int l = 0; l < H; ++l)
				{
					std::fill(nextTimeDh[l].begin(), nextTimeDh[l].end(), 0.0f);
					std::fill(nextTimeDc[l].begin(), nextTimeDc[l].end(), 0.0f);
				}

				dInputFromAbove = NULL;

				for (int t = static_cast<int>(winLen) - 1; t >= 0; --t)
				{
					const size_t tOffOut = static_cast<size_t>(t) * static_cast<size_t>(outSize);

					// Output deltas + updates
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
								lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_LSTM: non-finite expected value detected in BPTT (NaN/Inf)");
								running = false;
								return;
							}
							if (!is_finite(pred))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite prediction detected in BPTT (NaN/Inf)");
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
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite output delta detected (NaN/Inf)");
								running = false;
								return;
							}

							// Accumulate output gradients (Why and bias).
							const size_t rowOff = static_cast<size_t>(k) * static_cast<size_t>(outIn);
							const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(outIn);
							for (unsigned int i = 0; i < outIn; ++i)
								tensorLstm.O.gWhy[rowOff + i] += (deltaY[k] * invWinLen) * hFlat[H - 1][hOff + i];
							tensorLstm.O.gBias[k] += (deltaY[k] * invWinLen);
						}
					}

					dInputFromAbove = NULL;
					for (int l = H - 1; l >= 0; --l)
					{
						const unsigned int curSize = hiddenSizes[l];
						const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[l - 1];
						const Gated layout = {prevSize, curSize, gateCount};
						(void)layout; // layout is kept for readability; indices use packed tensors directly.
						const size_t tOffCur = static_cast<size_t>(t) * static_cast<size_t>(curSize);
						TensorGatedState::Hidden& thl = tensorLstm.H[static_cast<size_t>(l)];

						std::vector<float>& dh = dhBuf[l];
						std::fill(dh.begin(), dh.end(), 0.0f);
						if (l == H - 1)
						{
							for (unsigned int i = 0; i < curSize; ++i)
							{
								float sum = 0.0f;
								const unsigned int outIn = hiddenSizes[H - 1];
								for (unsigned int k = 0; k < outSize; ++k)
									sum += deltaY[k] * tensorLstm.O.Why[static_cast<size_t>(k) * static_cast<size_t>(outIn) + i];
								dh[i] = sum;
							}
						}
						else
						{
							if (dInputFromAbove && dInputFromAbove->size() == curSize)
							{
								for (unsigned int i = 0; i < curSize; ++i)
									dh[i] = (*dInputFromAbove)[i];
							}
						}
						for (unsigned int i = 0; i < curSize; ++i)
							dh[i] += nextTimeDh[l][i];

						// LSTM gate preactivation derivatives
						std::vector<float>& daI = daIBuf[l];
						std::vector<float>& daF = daFBuf[l];
						std::vector<float>& daO = daOBuf[l];
						std::vector<float>& daG = daGBuf[l];
						std::fill(daI.begin(), daI.end(), 0.0f);
						std::fill(daF.begin(), daF.end(), 0.0f);
						std::fill(daO.begin(), daO.end(), 0.0f);
						std::fill(daG.begin(), daG.end(), 0.0f);

						std::vector<float>& dc = dcBuf[l];
						std::vector<float>& dcPrevLocal = dcPrevBuf[l];
						std::fill(dc.begin(), dc.end(), 0.0f);
						std::fill(dcPrevLocal.begin(), dcPrevLocal.end(), 0.0f);

						for (unsigned int u = 0; u < curSize; ++u)
						{
							const float iT = iGateFlat[l][tOffCur + u];
							const float fT = fGateFlat[l][tOffCur + u];
							const float oT = oGateFlat[l][tOffCur + u];
							const float gT = gGateFlat[l][tOffCur + u];
							const float tanhCT = tanhCFlat[l][tOffCur + u];

							// do = dh * tanh(c)
							const float dO = dh[u] * tanhCT;
							// dc = dh * o * (1 - tanh(c)^2) + dc_next
							dc[u] = dh[u] * oT * GMath::activationErrDer(tanhCT, GMath::TANH, 0.0f) + nextTimeDc[l][u];

							const float dI = dc[u] * gT;
							const float dG = dc[u] * iT;
							const float dF = dc[u] * cPrevAtTFlat[l][tOffCur + u];
							dcPrevLocal[u] = dc[u] * fT;

							daI[u] = clipf_maybe(dI * GMath::activationErrDer(iT, GMath::SIGMOID, 0.0f), gradClip);
							daF[u] = clipf_maybe(dF * GMath::activationErrDer(fT, GMath::SIGMOID, 0.0f), gradClip);
							daO[u] = clipf_maybe(dO * GMath::activationErrDer(oT, GMath::SIGMOID, 0.0f), gradClip);
							daG[u] = clipf_maybe(dG * GMath::activationErrDer(gT, GMath::TANH, 0.0f), gradClip);
							if (!is_finite(daI[u]) || !is_finite(daF[u]) || !is_finite(daO[u]) || !is_finite(daG[u]))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_LSTM: non-finite gate derivatives detected (NaN/Inf)");
								running = false;
								return;
							}
						}

						// dhPrev = U^T * da (sum over gates)
						std::vector<float>& dhPrevLocal = dhPrevBuf[l];
						std::fill(dhPrevLocal.begin(), dhPrevLocal.end(), 0.0f);
						for (unsigned int j = 0; j < curSize; ++j)
						{
							float sum = 0.0f;
							for (unsigned int u = 0; u < curSize; ++u)
							{
								sum += daI[u] * thl.U[(static_cast<size_t>(GI) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize) + static_cast<size_t>(j)];
								sum += daF[u] * thl.U[(static_cast<size_t>(GF) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize) + static_cast<size_t>(j)];
								sum += daO[u] * thl.U[(static_cast<size_t>(GO) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize) + static_cast<size_t>(j)];
								sum += daG[u] * thl.U[(static_cast<size_t>(GG) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize) + static_cast<size_t>(j)];
							}
							dhPrevLocal[j] = sum;
						}

						// dInput for lower layer at same time
						std::vector<float>& dInput = dInputBuf[l];
						std::fill(dInput.begin(), dInput.end(), 0.0f);
						for (unsigned int p = 0; p < prevSize; ++p)
						{
							float sum = 0.0f;
							for (unsigned int u = 0; u < curSize; ++u)
							{
								sum += daI[u] * thl.W[(static_cast<size_t>(GI) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize) + static_cast<size_t>(p)];
								sum += daF[u] * thl.W[(static_cast<size_t>(GF) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize) + static_cast<size_t>(p)];
								sum += daO[u] * thl.W[(static_cast<size_t>(GO) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize) + static_cast<size_t>(p)];
								sum += daG[u] * thl.W[(static_cast<size_t>(GG) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize) + static_cast<size_t>(p)];
							}
							dInput[p] = sum;
						}
						dInputFromAbove = &dInput;

						// Accumulate weight gradients for this layer
						for (unsigned int u = 0; u < curSize; ++u)
						{
							const size_t wiOff = (static_cast<size_t>(GI) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize);
							const size_t wfOff = (static_cast<size_t>(GF) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize);
							const size_t woOff = (static_cast<size_t>(GO) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize);
							const size_t wgOff = (static_cast<size_t>(GG) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize);
							for (unsigned int p = 0; p < prevSize; ++p)
							{
								const float inAct =
								    (l == 0)
								        ? xFlat[static_cast<size_t>(t) * static_cast<size_t>(inputSize) + p]
								        : hFlat[l - 1][static_cast<size_t>(t) * static_cast<size_t>(hiddenSizes[l - 1]) + p];
								thl.gW[wiOff + p] += (daI[u] * invWinLen) * inAct;
								thl.gW[wfOff + p] += (daF[u] * invWinLen) * inAct;
								thl.gW[woOff + p] += (daO[u] * invWinLen) * inAct;
								thl.gW[wgOff + p] += (daG[u] * invWinLen) * inAct;
							}

							thl.gBias[static_cast<size_t>(GI) * static_cast<size_t>(curSize) + static_cast<size_t>(u)] += (daI[u] * invWinLen);
							thl.gBias[static_cast<size_t>(GF) * static_cast<size_t>(curSize) + static_cast<size_t>(u)] += (daF[u] * invWinLen);
							thl.gBias[static_cast<size_t>(GO) * static_cast<size_t>(curSize) + static_cast<size_t>(u)] += (daO[u] * invWinLen);
							thl.gBias[static_cast<size_t>(GG) * static_cast<size_t>(curSize) + static_cast<size_t>(u)] += (daG[u] * invWinLen);

							const size_t uiOff = (static_cast<size_t>(GI) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize);
							const size_t ufOff = (static_cast<size_t>(GF) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize);
							const size_t uoOff = (static_cast<size_t>(GO) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize);
							const size_t ugOff = (static_cast<size_t>(GG) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(curSize);
							for (unsigned int j = 0; j < curSize; ++j)
							{
								const float prevH = hPrevAtTFlat[l][tOffCur + j];
								thl.gU[uiOff + j] += (daI[u] * invWinLen) * prevH;
								thl.gU[ufOff + j] += (daF[u] * invWinLen) * prevH;
								thl.gU[uoOff + j] += (daO[u] * invWinLen) * prevH;
								thl.gU[ugOff + j] += (daG[u] * invWinLen) * prevH;
							}
						}

						nextTimeDh[l] = dhPrevLocal;
						nextTimeDc[l] = dcPrevLocal;
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

