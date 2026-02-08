// Net-type-specific SGD: Transformer encoder/decoder-only path
#include "network.h"
#include "sgd_utils.h"
#include "transformer_kernels.h"

#include "Backend/Database/GLogger.h"

#include "../DataObjects/DataInput.h"
#include "../GMath/gmath.h"
#include "../rng.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

using namespace glades;

namespace {

static inline float clip_maybe(float v, float limit)
{
	return glades::sgd_detail::clipf_maybe(v, limit);
}

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
static inline void append_logfmt_kv(std::ostringstream& oss, const char* k, bool v) { oss << ' ' << k << '=' << (v ? 1 : 0); }

static void add_positional_encoding(float* h,
                                    unsigned int T,
                                    unsigned int dModel,
                                    const std::vector<double>& invDenomPair)
{
	// Sinusoidal positional encoding (Vaswani et al.).
	// Kept as a thin wrapper so both training and inference share the same implementation.
	if (T == 0u || dModel == 0u)
		return;
	if (!h)
		return;
	glades::transformer_kernels::add_sinusoidal_positional_encoding_seq_inplace(h, T, dModel, invDenomPair);
}

static void linear_forward_maybe_lowp(const float* X,
                                      unsigned int T,
                                      unsigned int inSize,
                                      const std::vector<float>& W,
                                      const std::vector<uint16_t>& WLowp,
                                      bool useLowp,
                                      int lowpDType,
                                      const std::vector<float>& b,
                                      unsigned int outSize,
                                      float* Y)
{
	if (useLowp)
	{
		if (WLowp.size() == static_cast<size_t>(outSize) * static_cast<size_t>(inSize))
			glades::transformer_kernels::linear_forward_lowp(X, T, inSize, &WLowp[0], lowpDType, b, outSize, Y);
		else
			glades::transformer_kernels::linear_forward_opt(X, T, inSize, W, b, outSize, Y);
	}
	else
	{
		glades::transformer_kernels::linear_forward_opt(X, T, inSize, W, b, outSize, Y);
	}
}

// Backprop linear:
// - Accumulate gW += dY^T * X, gB += sum_t dY
// - dX += dY * W  (W is [out,in])
static void linear_backward_accum(const float* X,
                                  const float* dY,
                                  unsigned int T,
                                  unsigned int inSize,
                                  unsigned int outSize,
                                  std::vector<float>& gW,
                                  std::vector<float>& gB,
                                  const std::vector<float>& W,
                                  float* dXOut /* optional; size [T*inSize] */)
{
	if (gW.size() != static_cast<size_t>(outSize) * static_cast<size_t>(inSize))
		gW.assign(static_cast<size_t>(outSize) * static_cast<size_t>(inSize), 0.0f);
	if (gB.size() != outSize)
		gB.assign(outSize, 0.0f);

	if (dXOut)
		std::fill(dXOut, dXOut + (static_cast<size_t>(T) * static_cast<size_t>(inSize)), 0.0f);

	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t xOff = static_cast<size_t>(t) * static_cast<size_t>(inSize);
		const size_t dyOff = static_cast<size_t>(t) * static_cast<size_t>(outSize);
		for (unsigned int o = 0; o < outSize; ++o)
		{
			const float dy = dY[dyOff + o];
			gB[o] += dy;
			const size_t wOff = static_cast<size_t>(o) * static_cast<size_t>(inSize);
			for (unsigned int i = 0; i < inSize; ++i)
				gW[wOff + i] += dy * X[xOff + i];
		}

		if (dXOut)
		{
			const size_t dxOff = static_cast<size_t>(t) * static_cast<size_t>(inSize);
			for (unsigned int i = 0; i < inSize; ++i)
			{
				double acc = 0.0;
				for (unsigned int o = 0; o < outSize; ++o)
					acc += static_cast<double>(dY[dyOff + o]) *
					       static_cast<double>(W[static_cast<size_t>(o) * static_cast<size_t>(inSize) + i]);
				dXOut[dxOff + i] += static_cast<float>(acc);
			}
		}
	}
}

static void linear_backward_accum_maybe_lowp(const float* X,
                                             const float* dY,
                                             unsigned int T,
                                             unsigned int inSize,
                                             unsigned int outSize,
                                             std::vector<float>& gW,
                                             std::vector<float>& gB,
                                             const std::vector<float>& WMaster,
                                             const std::vector<uint16_t>& WLowp,
                                             bool useLowp,
                                             int lowpDType,
                                             float* dXOut /* optional; size [T*inSize] */)
{
	// gW/gB accumulation does not depend on W; only dX does.
	linear_backward_accum(X, dY, T, inSize, outSize, gW, gB, WMaster, NULL);

	if (!dXOut)
		return;
	std::fill(dXOut, dXOut + (static_cast<size_t>(T) * static_cast<size_t>(inSize)), 0.0f);

	const bool haveLowp = useLowp && (WLowp.size() == static_cast<size_t>(outSize) * static_cast<size_t>(inSize));
	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t dyOff = static_cast<size_t>(t) * static_cast<size_t>(outSize);
		const size_t dxOff = static_cast<size_t>(t) * static_cast<size_t>(inSize);
		for (unsigned int i = 0; i < inSize; ++i)
		{
			double acc = 0.0;
			for (unsigned int o = 0; o < outSize; ++o)
			{
				const size_t wIdx = static_cast<size_t>(o) * static_cast<size_t>(inSize) + static_cast<size_t>(i);
				const float w = haveLowp ? glades::transformer_kernels::lowp_to_float(WLowp[wIdx], lowpDType) : WMaster[wIdx];
				acc += static_cast<double>(dY[dyOff + o]) * static_cast<double>(w);
			}
			dXOut[dxOff + i] += static_cast<float>(acc);
		}
	}
}

// Normalization + softmax kernels live in `transformer_kernels.h` so training and inference
// share math conventions and future optimized kernels can be dropped in behind a stable API.

// RoPE helpers moved to transformer_kernels.h (shared by train + inference).

} // namespace

void glades::NNetwork::SGDHelper_TRANSFORMER(unsigned int inputRowCounter, int runType)
{
	using namespace glades::sgd_detail;
	(void)inputRowCounter;

	const bool isTrain = (runType == RUN_TRAIN);
	const unsigned int dataSize = isTrain ? (di ? di->getTrainSize() : 0u) : (di ? di->getTestSize() : 0u);

	// Only run once per epoch (Trainer loops over steps=1 for sequence models).
	if (inputRowCounter != 0u)
		return;

	// Epoch-local progress logging (rate-limited by a fixed number of updates per epoch).
	// This keeps long-running LLM epochs from appearing "stuck" while avoiding log spam.
	shmea::GLogger* logger = getLogger();
	const int64_t epochStartMs = getCurrentTimeMilliseconds();
	const int epochIdx = epochs; // current epoch number for this run (Trainer increments after SGDHelper returns)

	// Transformers do not use the graph's dropout masks (no node-level dropout here).
	// (legacy graph dropout removed)

	const unsigned int seqCount = di ? (isTrain ? di->getTrainSequenceCount() : di->getTestSequenceCount()) : 0u;
	if (seqCount == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE,
		                            isTrain ? "SGDHelper_TRANSFORMER: no train sequences"
		                                    : "SGDHelper_TRANSFORMER: no test sequences");
		running = false;
		return;
	}

	// Ensure transformer parameters exist.
	if (!ensureTensorParametersInitialized())
	{
		running = false;
		return;
	}

	const TensorTransformerState& ttConst = tensorTransformer;
	const unsigned int inputSize = ttConst.inputSize;
	const unsigned int outSize = ttConst.outSize;
	const unsigned int dModel = ttConst.dModel;
	const unsigned int dFF = ttConst.dFF;
	const unsigned int nHeads = ttConst.nHeads;
	const unsigned int nLayers = ttConst.nLayers;
	const bool causal = ttConst.causal;
	const bool tokenLM = ttConst.tokenModel;
	const unsigned int vocabSize = ttConst.vocabSize;
	const int padTokenId = ttConst.padTokenId;
	const bool tieEmb = ttConst.tieEmbeddings;

	if (inputSize == 0u || outSize == 0u || dModel == 0u || dFF == 0u || nHeads == 0u || nLayers == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_TRANSFORMER: invalid transformer state sizes");
		running = false;
		return;
	}
	if (tokenLM)
	{
		if (!tieEmb)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token LM mode requires tieEmbeddings=true");
			running = false;
			return;
		}
		if (inputSize != 1u)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token LM mode requires inputSize==1 (token id)");
			running = false;
			return;
		}
		if (outSize != vocabSize || vocabSize == 0u)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token LM mode requires outSize==vocabSize>0");
			running = false;
			return;
		}
		if (tieEmb && (ttConst.tokE.size() != static_cast<size_t>(vocabSize) * static_cast<size_t>(dModel)))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_TRANSFORMER: token LM embedding table is not initialized");
			running = false;
			return;
		}
	}

	const float gradClip = trainingConfig.perElementGradClip;
	const int costFx = skeleton->getOutputType();
	const float lnEps = (trainingConfig.transformer.layerNormEps > 0.0f ? trainingConfig.transformer.layerNormEps : 1e-5f);
	const int posEnc = static_cast<int>(trainingConfig.transformer.positionalEncoding);
	const int normType = static_cast<int>(trainingConfig.transformer.normType);
	const int ffnKind = static_cast<int>(trainingConfig.transformer.ffnKind);
	const int ffnAct = static_cast<int>(trainingConfig.transformer.ffnActivation);
	const float ropeTheta = (trainingConfig.transformer.ropeTheta > 0.0f ? trainingConfig.transformer.ropeTheta : 10000.0f);
	const int ropeDimOverride = trainingConfig.transformer.ropeDimOverride;
	const glades::TransformerRunConfig::TokenLMLossKind tokenLmLossKind = trainingConfig.transformer.tokenLmLossKind;
	const int tokenLmNegK = trainingConfig.transformer.tokenLmSampledNegatives;
	const bool tokenLmAllowHuge = trainingConfig.transformer.tokenLmAllowHugeFullSoftmax;

	// Trainability triage:
	// - For real LLMs, AdamW is the supported optimizer in this backend.
	// - Full softmax is guarded to avoid silently allocating/computing O(T*vocab) buffers.
	if (tokenLM && isTrain && (trainingConfig.optimizer.type != glades::OptimizerConfig::ADAMW))
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "SGDHelper_TRANSFORMER: token LM training requires optimizer=ADAMW for LLM-scale stability");
		running = false;
		return;
	}
	if (tokenLM && (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX) && (tokenLmNegK < 1))
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "SGDHelper_TRANSFORMER: token LM sampled-softmax requires tokenLmSampledNegatives >= 1");
		running = false;
		return;
	}

	// Reset per-epoch bookkeeping
	results.clear();
	if (isTrain)
		nbRecord.clear();

	// Minibatch: number of sequences to accumulate before applying an update.
	const unsigned int seqBatchMax = (minibatchSize > 0 ? static_cast<unsigned int>(minibatchSize) : 1u);
	unsigned int seqInBatch = 0u;
	// Gradient averaging divisor for the minibatch:
	// - Non-tokenLM: total timesteps across sequences in the batch (as before).
	// - Token LM: total *valid target tokens* (non-pad) across sequences in the batch.
	//   This matches the loss normalization (mean NLL over non-pad targets).
	unsigned int timeStepsInBatch = 0u;

	// Helper: clear gradient accumulators when starting a new minibatch.
	struct ClearGrads
	{
		TensorTransformerState& tt;
		ClearGrads(TensorTransformerState& t) : tt(t) {}
		void operator()() const
		{
			std::fill(tt.gTokE.begin(), tt.gTokE.end(), 0.0f);
			std::fill(tt.gLmBias.begin(), tt.gLmBias.end(), 0.0f);
			std::fill(tt.gWIn.begin(), tt.gWIn.end(), 0.0f);
			std::fill(tt.gBIn.begin(), tt.gBIn.end(), 0.0f);
			std::fill(tt.gWOut.begin(), tt.gWOut.end(), 0.0f);
			std::fill(tt.gBOut.begin(), tt.gBOut.end(), 0.0f);
			for (size_t l = 0; l < tt.blocks.size(); ++l)
			{
				TensorTransformerState::Block& b = tt.blocks[l];
				std::fill(b.gLn1Gamma.begin(), b.gLn1Gamma.end(), 0.0f);
				std::fill(b.gLn1Beta.begin(), b.gLn1Beta.end(), 0.0f);
				std::fill(b.gWq.begin(), b.gWq.end(), 0.0f);
				std::fill(b.gWk.begin(), b.gWk.end(), 0.0f);
				std::fill(b.gWv.begin(), b.gWv.end(), 0.0f);
				std::fill(b.gWo.begin(), b.gWo.end(), 0.0f);
				std::fill(b.gBq.begin(), b.gBq.end(), 0.0f);
				std::fill(b.gBk.begin(), b.gBk.end(), 0.0f);
				std::fill(b.gBv.begin(), b.gBv.end(), 0.0f);
				std::fill(b.gBo.begin(), b.gBo.end(), 0.0f);
				std::fill(b.gLn2Gamma.begin(), b.gLn2Gamma.end(), 0.0f);
				std::fill(b.gLn2Beta.begin(), b.gLn2Beta.end(), 0.0f);
				std::fill(b.gW1.begin(), b.gW1.end(), 0.0f);
				std::fill(b.gW2.begin(), b.gW2.end(), 0.0f);
				std::fill(b.gB1.begin(), b.gB1.end(), 0.0f);
				std::fill(b.gB2.begin(), b.gB2.end(), 0.0f);
			}
		}
	};

	// Helper: apply accumulated gradients (averaged by timesteps) with momentum + optional global norm clip.
	struct ApplyBatch
	{
		NNetwork& net;
		TensorTransformerState& tt;
		unsigned int nLayers;
		unsigned int dModel;
		unsigned int outSize;

		ApplyBatch(NNetwork& n, TensorTransformerState& t, unsigned int nl, unsigned int dm, unsigned int os)
		    : net(n), tt(t), nLayers(nl), dModel(dm), outSize(os)
		{
		}

		bool operator()(unsigned int batchTimeSteps) const
		{
			using namespace glades::sgd_detail;
			if (batchTimeSteps == 0u)
				return true;

			const float invBatch = 1.0f / static_cast<float>(batchTimeSteps);
			const bool useAdamW = (net.trainingConfig.optimizer.type == glades::OptimizerConfig::ADAMW);
			// Token LM mode uses a tied embedding head: logits = H * E^T + lmBias.
			// In this mode, the generic output projection (WOut/bOut) is UNUSED and must not:
			// - contribute to global grad-norm clipping (via weight decay terms), or
			// - be updated/decayed by the optimizer.
			//
			// Otherwise, the model will "silently" change unused parameters and can skew grad clipping
			// scale for the parameters that actually affect the forward pass.
			const bool tokenLMTiedHead = tt.tokenModel;

			// Optional global grad norm clip (same semantics as other tensor paths).
			float gradNorm = 0.0f;
			float gradScale = 1.0f;
			const float clipNorm = net.trainingConfig.globalGradClipNorm;
			if (clipNorm > 0.0f)
			{
				double sumsq = 0.0;

				if (useAdamW)
				{
					// For AdamW, clip is applied to raw gradients (weight decay is decoupled).
					if (tt.tokenModel)
					{
						for (size_t i = 0; i < tt.gTokE.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gTokE[i] * invBatch);
							sumsq += gd * gd;
						}
						for (size_t i = 0; i < tt.gLmBias.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gLmBias[i] * invBatch);
							sumsq += gd * gd;
						}
					}
					for (size_t i = 0; i < tt.gWIn.size(); ++i)
					{
						const double gd = static_cast<double>(tt.gWIn[i] * invBatch);
						sumsq += gd * gd;
					}
					for (size_t i = 0; i < tt.gBIn.size(); ++i)
					{
						const double gd = static_cast<double>(tt.gBIn[i] * invBatch);
						sumsq += gd * gd;
					}
					for (unsigned int li = 0; li < nLayers; ++li)
					{
						const TensorTransformerState::Block& b = tt.blocks[li];
						for (size_t j = 0; j < b.gWq.size(); ++j) { const double gd = static_cast<double>(b.gWq[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gWk.size(); ++j) { const double gd = static_cast<double>(b.gWk[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gWv.size(); ++j) { const double gd = static_cast<double>(b.gWv[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gWo.size(); ++j) { const double gd = static_cast<double>(b.gWo[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gW1.size(); ++j) { const double gd = static_cast<double>(b.gW1[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gW2.size(); ++j) { const double gd = static_cast<double>(b.gW2[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gBq.size(); ++j) { const double gd = static_cast<double>(b.gBq[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gBk.size(); ++j) { const double gd = static_cast<double>(b.gBk[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gBv.size(); ++j) { const double gd = static_cast<double>(b.gBv[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gBo.size(); ++j) { const double gd = static_cast<double>(b.gBo[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gB1.size(); ++j) { const double gd = static_cast<double>(b.gB1[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gB2.size(); ++j) { const double gd = static_cast<double>(b.gB2[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gLn1Gamma.size(); ++j) { const double gd = static_cast<double>(b.gLn1Gamma[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gLn1Beta.size(); ++j) { const double gd = static_cast<double>(b.gLn1Beta[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gLn2Gamma.size(); ++j) { const double gd = static_cast<double>(b.gLn2Gamma[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gLn2Beta.size(); ++j) { const double gd = static_cast<double>(b.gLn2Beta[j] * invBatch); sumsq += gd * gd; }
					}
					// Output projection gradients exist only for non-tokenLM paths.
					if (!tokenLMTiedHead)
					{
						for (size_t i = 0; i < tt.gWOut.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gWOut[i] * invBatch);
							sumsq += gd * gd;
						}
						for (size_t i = 0; i < tt.gBOut.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gBOut[i] * invBatch);
							sumsq += gd * gd;
						}
					}
				}
				else
				{
					// Historical semantics: include L1/L2 weight decay in clip norm.
					// Token LM embedding + bias (index 0)
					if (tt.tokenModel)
					{
						const float wd1 = net.skeleton->getWeightDecay1(0u);
						const float wd2 = net.skeleton->getWeightDecay2(0u);
						for (size_t i = 0; i < tt.tokE.size(); ++i)
						{
							float g = tt.gTokE[i] * invBatch;
							if ((wd1 != 0.0f) || (wd2 != 0.0f))
							{
								const float w = tt.tokE[i];
								const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
								g += (wd1 * wSign) + (wd2 * w);
							}
							const double gd = static_cast<double>(g);
							sumsq += gd * gd;
						}
						for (size_t i = 0; i < tt.lmBias.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gLmBias[i] * invBatch);
							sumsq += gd * gd;
						}
					}

					// Input projection (index 0)
					{
						const float wd1 = net.skeleton->getWeightDecay1(0u);
						const float wd2 = net.skeleton->getWeightDecay2(0u);
						for (size_t i = 0; i < tt.WIn.size(); ++i)
						{
							float g = tt.gWIn[i] * invBatch;
							if ((wd1 != 0.0f) || (wd2 != 0.0f))
							{
								const float w = tt.WIn[i];
								const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
								g += (wd1 * wSign) + (wd2 * w);
							}
							const double gd = static_cast<double>(g);
							sumsq += gd * gd;
						}
						for (size_t i = 0; i < tt.bIn.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gBIn[i] * invBatch);
							sumsq += gd * gd;
						}
					}

					// Blocks (index 1..nLayers)
					for (unsigned int li = 0; li < nLayers; ++li)
					{
						const unsigned int idx = li + 1u;
						const float wd1 = net.skeleton->getWeightDecay1(idx);
						const float wd2 = net.skeleton->getWeightDecay2(idx);
						const TensorTransformerState::Block& b = tt.blocks[li];
						// Wq/Wk/Wv/Wo/W1/W2
						{
							const std::vector<float>* Wv[6] = {&b.Wq, &b.Wk, &b.Wv, &b.Wo, &b.W1, &b.W2};
							const std::vector<float>* Gv[6] = {&b.gWq, &b.gWk, &b.gWv, &b.gWo, &b.gW1, &b.gW2};
							for (unsigned int wi = 0; wi < 6u; ++wi)
							{
								const std::vector<float>& W = *Wv[wi];
								const std::vector<float>& gW = *Gv[wi];
								for (size_t j = 0; j < W.size(); ++j)
								{
									float g = gW[j] * invBatch;
									if ((wd1 != 0.0f) || (wd2 != 0.0f))
									{
										const float w = W[j];
										const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
										g += (wd1 * wSign) + (wd2 * w);
									}
									const double gd = static_cast<double>(g);
									sumsq += gd * gd;
								}
							}
						}
						// Include biases, LN params
						for (size_t j = 0; j < b.bq.size(); ++j) sumsq += static_cast<double>(b.gBq[j] * invBatch) * static_cast<double>(b.gBq[j] * invBatch);
						for (size_t j = 0; j < b.bk.size(); ++j) sumsq += static_cast<double>(b.gBk[j] * invBatch) * static_cast<double>(b.gBk[j] * invBatch);
						for (size_t j = 0; j < b.bv.size(); ++j) sumsq += static_cast<double>(b.gBv[j] * invBatch) * static_cast<double>(b.gBv[j] * invBatch);
						for (size_t j = 0; j < b.bo.size(); ++j) sumsq += static_cast<double>(b.gBo[j] * invBatch) * static_cast<double>(b.gBo[j] * invBatch);
						for (size_t j = 0; j < b.b1.size(); ++j) sumsq += static_cast<double>(b.gB1[j] * invBatch) * static_cast<double>(b.gB1[j] * invBatch);
						for (size_t j = 0; j < b.b2.size(); ++j) sumsq += static_cast<double>(b.gB2[j] * invBatch) * static_cast<double>(b.gB2[j] * invBatch);
						for (size_t j = 0; j < b.ln1Gamma.size(); ++j) sumsq += static_cast<double>(b.gLn1Gamma[j] * invBatch) * static_cast<double>(b.gLn1Gamma[j] * invBatch);
						for (size_t j = 0; j < b.ln1Beta.size(); ++j) sumsq += static_cast<double>(b.gLn1Beta[j] * invBatch) * static_cast<double>(b.gLn1Beta[j] * invBatch);
						for (size_t j = 0; j < b.ln2Gamma.size(); ++j) sumsq += static_cast<double>(b.gLn2Gamma[j] * invBatch) * static_cast<double>(b.gLn2Gamma[j] * invBatch);
						for (size_t j = 0; j < b.ln2Beta.size(); ++j) sumsq += static_cast<double>(b.gLn2Beta[j] * invBatch) * static_cast<double>(b.gLn2Beta[j] * invBatch);
					}

					// Output projection (index nLayers)
					if (!tokenLMTiedHead)
					{
						const unsigned int idx = nLayers;
						const float wd1 = net.skeleton->getWeightDecay1(idx);
						const float wd2 = net.skeleton->getWeightDecay2(idx);
						for (size_t i = 0; i < tt.WOut.size(); ++i)
						{
							float g = tt.gWOut[i] * invBatch;
							if ((wd1 != 0.0f) || (wd2 != 0.0f))
							{
								const float w = tt.WOut[i];
								const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
								g += (wd1 * wSign) + (wd2 * w);
							}
							const double gd = static_cast<double>(g);
							sumsq += gd * gd;
						}
						for (size_t i = 0; i < tt.bOut.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gBOut[i] * invBatch);
							sumsq += gd * gd;
						}
					}
				}

				if (!is_finite_double(sumsq))
				{
					net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_TRANSFORMER: non-finite grad-norm accumulation detected (NaN/Inf)");
					net.running = false;
					return false;
				}
				gradNorm = static_cast<float>(sqrt(sumsq));
				const float eps = 1e-12f;
				gradScale = (gradNorm > clipNorm) ? (clipNorm / (gradNorm + eps)) : 1.0f;
			}

			net.lastGradNorm = gradNorm;
			net.lastGradNormScale = gradScale;
			if (!is_finite(net.lastGradNorm) || !is_finite(net.lastGradNormScale) || net.lastGradNormScale <= 0.0f)
			{
				net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_TRANSFORMER: non-finite grad clipping metadata detected (NaN/Inf)");
				net.running = false;
				return false;
			}

			// --- Apply updates ---
			if (!useAdamW)
			{
				// SGD + momentum (historical behavior).
				// Token embedding (index 0 in LM mode)
				if (tt.tokenModel)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier;
					const float mf = net.skeleton->getMomentumFactor(0u);
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);

					for (size_t i = 0; i < tt.tokE.size(); ++i)
					{
						float g = tt.gTokE[i] * invBatch;
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = tt.tokE[i];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							g += (wd1 * wSign) + (wd2 * w);
						}
						g *= gradScale;
						const float v = (mf * tt.vTokE[i]) + (lr * g);
						tt.vTokE[i] = v;
						tt.tokE[i] -= v;
						tt.gTokE[i] = 0.0f;
					}
					for (size_t i = 0; i < tt.lmBias.size(); ++i)
					{
						const float gB = (tt.gLmBias[i] * invBatch) * gradScale;
						tt.lmBias[i] -= lr * gB;
						tt.gLmBias[i] = 0.0f;
					}
				}

				// Input projection (index 0)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier;
					const float mf = net.skeleton->getMomentumFactor(0u);
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					for (size_t i = 0; i < tt.WIn.size(); ++i)
					{
						float g = tt.gWIn[i] * invBatch;
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = tt.WIn[i];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							g += (wd1 * wSign) + (wd2 * w);
						}
						g *= gradScale;
						const float v = (mf * tt.vWIn[i]) + (lr * g);
						tt.vWIn[i] = v;
						tt.WIn[i] -= v;
						tt.gWIn[i] = 0.0f;
					}
					for (size_t i = 0; i < tt.bIn.size(); ++i)
					{
						const float gB = (tt.gBIn[i] * invBatch) * gradScale;
						tt.bIn[i] -= lr * gB;
						tt.gBIn[i] = 0.0f;
					}
				}

				// Blocks (index 1..nLayers)
				for (unsigned int li = 0; li < nLayers; ++li)
				{
					const unsigned int idx = li + 1u;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier;
					const float mf = net.skeleton->getMomentumFactor(idx);
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					TensorTransformerState::Block& b = tt.blocks[li];

					// Attention/FFN weights with momentum/decay
					struct Upd
					{
						static void run(std::vector<float>& W, std::vector<float>& vW, std::vector<float>& gW,
						                float lr, float mf, float wd1, float wd2, float invBatch, float gradScale)
						{
							for (size_t i = 0; i < W.size(); ++i)
							{
								float g = gW[i] * invBatch;
								if ((wd1 != 0.0f) || (wd2 != 0.0f))
								{
									const float w = W[i];
									const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
									g += (wd1 * wSign) + (wd2 * w);
								}
								g *= gradScale;
								const float v = (mf * vW[i]) + (lr * g);
								vW[i] = v;
								W[i] -= v;
								gW[i] = 0.0f;
							}
						}
					};

					Upd::run(b.Wq, b.vWq, b.gWq, lr, mf, wd1, wd2, invBatch, gradScale);
					Upd::run(b.Wk, b.vWk, b.gWk, lr, mf, wd1, wd2, invBatch, gradScale);
					Upd::run(b.Wv, b.vWv, b.gWv, lr, mf, wd1, wd2, invBatch, gradScale);
					Upd::run(b.Wo, b.vWo, b.gWo, lr, mf, wd1, wd2, invBatch, gradScale);
					Upd::run(b.W1, b.vW1, b.gW1, lr, mf, wd1, wd2, invBatch, gradScale);
					Upd::run(b.W2, b.vW2, b.gW2, lr, mf, wd1, wd2, invBatch, gradScale);

					// Biases and LN params (no momentum/decay)
					for (size_t i = 0; i < b.bq.size(); ++i) { b.bq[i] -= lr * (b.gBq[i] * invBatch) * gradScale; b.gBq[i] = 0.0f; }
					for (size_t i = 0; i < b.bk.size(); ++i) { b.bk[i] -= lr * (b.gBk[i] * invBatch) * gradScale; b.gBk[i] = 0.0f; }
					for (size_t i = 0; i < b.bv.size(); ++i) { b.bv[i] -= lr * (b.gBv[i] * invBatch) * gradScale; b.gBv[i] = 0.0f; }
					for (size_t i = 0; i < b.bo.size(); ++i) { b.bo[i] -= lr * (b.gBo[i] * invBatch) * gradScale; b.gBo[i] = 0.0f; }
					for (size_t i = 0; i < b.b1.size(); ++i) { b.b1[i] -= lr * (b.gB1[i] * invBatch) * gradScale; b.gB1[i] = 0.0f; }
					for (size_t i = 0; i < b.b2.size(); ++i) { b.b2[i] -= lr * (b.gB2[i] * invBatch) * gradScale; b.gB2[i] = 0.0f; }
					for (size_t i = 0; i < b.ln1Gamma.size(); ++i) { b.ln1Gamma[i] -= lr * (b.gLn1Gamma[i] * invBatch) * gradScale; b.gLn1Gamma[i] = 0.0f; }
					for (size_t i = 0; i < b.ln1Beta.size(); ++i) { b.ln1Beta[i] -= lr * (b.gLn1Beta[i] * invBatch) * gradScale; b.gLn1Beta[i] = 0.0f; }
					for (size_t i = 0; i < b.ln2Gamma.size(); ++i) { b.ln2Gamma[i] -= lr * (b.gLn2Gamma[i] * invBatch) * gradScale; b.gLn2Gamma[i] = 0.0f; }
					for (size_t i = 0; i < b.ln2Beta.size(); ++i) { b.ln2Beta[i] -= lr * (b.gLn2Beta[i] * invBatch) * gradScale; b.gLn2Beta[i] = 0.0f; }
				}

				// Output projection (index nLayers) is unused in token LM tied-head mode.
				if (!tokenLMTiedHead)
				{
					const unsigned int idx = nLayers;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier;
					const float mf = net.skeleton->getMomentumFactor(idx);
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					for (size_t i = 0; i < tt.WOut.size(); ++i)
					{
						float g = tt.gWOut[i] * invBatch;
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = tt.WOut[i];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							g += (wd1 * wSign) + (wd2 * w);
						}
						g *= gradScale;
						const float v = (mf * tt.vWOut[i]) + (lr * g);
						tt.vWOut[i] = v;
						tt.WOut[i] -= v;
						tt.gWOut[i] = 0.0f;
					}
					for (size_t i = 0; i < tt.bOut.size(); ++i)
					{
						const float gB = (tt.gBOut[i] * invBatch) * gradScale;
						tt.bOut[i] -= lr * gB;
						tt.gBOut[i] = 0.0f;
					}
				}
				else
				{
					// Defensive: ensure gradients are cleared so stale values never leak into later non-tokenLM runs.
					std::fill(tt.gWOut.begin(), tt.gWOut.end(), 0.0f);
					std::fill(tt.gBOut.begin(), tt.gBOut.end(), 0.0f);
				}
			}
			else
			{
				// AdamW (recommended for transformers).
				const float beta1 = net.trainingConfig.optimizer.adamBeta1;
				const float beta2 = net.trainingConfig.optimizer.adamBeta2;
				const float eps = net.trainingConfig.optimizer.adamEps;
				const bool biasCorr = net.trainingConfig.optimizer.adamBiasCorrection;

				tt.optimizerStep += 1ULL;
				const double t = static_cast<double>(tt.optimizerStep);
				const double b1t = biasCorr ? pow(static_cast<double>(beta1), t) : 0.0;
				const double b2t = biasCorr ? pow(static_cast<double>(beta2), t) : 0.0;
				const float inv1mB1t = biasCorr ? static_cast<float>(1.0 / (1.0 - b1t)) : 1.0f;
				const float inv1mB2t = biasCorr ? static_cast<float>(1.0 / (1.0 - b2t)) : 1.0f;

				struct Adam
				{
					static float signf(float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); }

					static void update_weight(std::vector<float>& W,
					                          std::vector<float>& m,
					                          std::vector<float>& v2,
					                          std::vector<float>& gW,
					                          float lr,
					                          float beta1,
					                          float beta2,
					                          float inv1mB1t,
					                          float inv1mB2t,
					                          float eps,
					                          float invBatch,
					                          float gradScale,
					                          float wd1,
					                          float wd2)
					{
						const float oneMinusB1 = 1.0f - beta1;
						const float oneMinusB2 = 1.0f - beta2;
						for (size_t i = 0; i < W.size(); ++i)
						{
							float g = gW[i] * invBatch;
							if (wd1 != 0.0f)
								g += wd1 * signf(W[i]);
							g *= gradScale;

							const float mi = (beta1 * m[i]) + (oneMinusB1 * g);
							const float vi = (beta2 * v2[i]) + (oneMinusB2 * (g * g));
							m[i] = mi;
							v2[i] = vi;

							const float mhat = mi * inv1mB1t;
							const float vhat = vi * inv1mB2t;
							const float denom = static_cast<float>(sqrt(static_cast<double>(vhat))) + eps;
							const float step = mhat / denom;

							// Decoupled weight decay.
							if (wd2 != 0.0f)
								W[i] -= lr * wd2 * W[i];
							W[i] -= lr * step;
							gW[i] = 0.0f;
						}
					}

					static void update_param(std::vector<float>& P,
					                         std::vector<float>& m,
					                         std::vector<float>& v2,
					                         std::vector<float>& gP,
					                         float lr,
					                         float beta1,
					                         float beta2,
					                         float inv1mB1t,
					                         float inv1mB2t,
					                         float eps,
					                         float invBatch,
					                         float gradScale)
					{
						const float oneMinusB1 = 1.0f - beta1;
						const float oneMinusB2 = 1.0f - beta2;
						for (size_t i = 0; i < P.size(); ++i)
						{
							float g = (gP[i] * invBatch) * gradScale;
							const float mi = (beta1 * m[i]) + (oneMinusB1 * g);
							const float vi = (beta2 * v2[i]) + (oneMinusB2 * (g * g));
							m[i] = mi;
							v2[i] = vi;

							const float mhat = mi * inv1mB1t;
							const float vhat = vi * inv1mB2t;
							const float denom = static_cast<float>(sqrt(static_cast<double>(vhat))) + eps;
							const float step = mhat / denom;

							P[i] -= lr * step;
							gP[i] = 0.0f;
						}
					}
				};

				// Token embedding (index 0 in LM mode)
				if (tt.tokenModel)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					Adam::update_weight(tt.tokE, tt.vTokE, tt.v2TokE, tt.gTokE,
					                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
					                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}

				// Input projection (index 0)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					Adam::update_weight(tt.WIn, tt.vWIn, tt.v2WIn, tt.gWIn,
					                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
					                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}

				// Blocks (index 1..nLayers)
				for (unsigned int li = 0; li < nLayers; ++li)
				{
					const unsigned int idx = li + 1u;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					TensorTransformerState::Block& b = tt.blocks[li];

					Adam::update_weight(b.Wq, b.vWq, b.v2Wq, b.gWq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_weight(b.Wk, b.vWk, b.v2Wk, b.gWk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_weight(b.Wv, b.vWv, b.v2Wv, b.gWv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_weight(b.Wo, b.vWo, b.v2Wo, b.gWo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_weight(b.W1, b.vW1, b.v2W1, b.gW1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_weight(b.W2, b.vW2, b.v2W2, b.gW2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);

					// Biases + LN params (no weight decay)
					Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}

				// Output projection (index nLayers) is unused in token LM tied-head mode.
				if (!tokenLMTiedHead)
				{
					const unsigned int idx = nLayers;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					Adam::update_weight(tt.WOut, tt.vWOut, tt.v2WOut, tt.gWOut,
					                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
					                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}
				else
				{
					std::fill(tt.gWOut.begin(), tt.gWOut.end(), 0.0f);
					std::fill(tt.gBOut.begin(), tt.gBOut.end(), 0.0f);
				}
			}

			return true;
		}
	};

	TensorTransformerState& tt = tensorTransformer;
	ClearGrads clearGrads(tt);
	ApplyBatch applyBatch(*this, tt, nLayers, dModel, outSize);

	// Mixed precision helper (must live inside this member function because TensorTransformerState is private).
	struct MixedPrecisionHelper
	{
		static void quantize_lowp(const std::vector<float>& src, std::vector<uint16_t>& dst, int lowpDType)
		{
			dst.resize(src.size());
			for (size_t i = 0; i < src.size(); ++i)
				dst[i] = glades::transformer_kernels::float_to_lowp(src[i], lowpDType);
		}

		static int lowp_dtype_from_cfg(const glades::TrainingConfig& cfg)
		{
			return (cfg.mixedPrecision.weightDType == glades::MixedPrecisionConfig::WEIGHT_BF16) ? glades::transformer_kernels::LOWP_BF16
			                                                                                     : glades::transformer_kernels::LOWP_F16;
		}

		static void ensure_transformer_lowp_weights(TensorTransformerState& tt, const glades::TrainingConfig& cfg)
		{
			const bool mpEnable = cfg.mixedPrecision.enable &&
			                      (cfg.mixedPrecision.weightDType != glades::MixedPrecisionConfig::WEIGHT_F32);
			if (!mpEnable)
			{
				tt.mpLowpReady = false;
				tt.mpLowpDType = 0;
				tt.tokELowp.clear();
				tt.WInLowp.clear();
				tt.WOutLowp.clear();
				for (size_t li = 0; li < tt.blocks.size(); ++li)
				{
					TensorTransformerState::Block& b = tt.blocks[li];
					b.WqLowp.clear();
					b.WkLowp.clear();
					b.WvLowp.clear();
					b.WoLowp.clear();
					b.W1Lowp.clear();
					b.W2Lowp.clear();
				}
				return;
			}

			const int lowpDType = lowp_dtype_from_cfg(cfg);
			tt.mpLowpDType = lowpDType;

			// Initialize loss scale on first use.
			if (cfg.mixedPrecision.useLossScaling)
			{
				if (!tt.mpLowpReady)
					tt.mpLossScale = cfg.mixedPrecision.lossScaleInit;
				if (tt.mpLossScale < cfg.mixedPrecision.lossScaleMin)
					tt.mpLossScale = cfg.mixedPrecision.lossScaleMin;
				if (tt.mpLossScale > cfg.mixedPrecision.lossScaleMax)
					tt.mpLossScale = cfg.mixedPrecision.lossScaleMax;
			}
			else
			{
				tt.mpLossScale = 1.0f;
			}

			// Always resync from master when called: correctness-first baseline.
			quantize_lowp(tt.tokE, tt.tokELowp, lowpDType);
			quantize_lowp(tt.WIn, tt.WInLowp, lowpDType);
			quantize_lowp(tt.WOut, tt.WOutLowp, lowpDType);
			for (size_t li = 0; li < tt.blocks.size(); ++li)
			{
				TensorTransformerState::Block& b = tt.blocks[li];
				quantize_lowp(b.Wq, b.WqLowp, lowpDType);
				quantize_lowp(b.Wk, b.WkLowp, lowpDType);
				quantize_lowp(b.Wv, b.WvLowp, lowpDType);
				quantize_lowp(b.Wo, b.WoLowp, lowpDType);
				quantize_lowp(b.W1, b.W1Lowp, lowpDType);
				quantize_lowp(b.W2, b.W2Lowp, lowpDType);
			}
			tt.mpLowpReady = true;
		}

		static bool grads_all_finite(const TensorTransformerState& tt)
		{
			for (size_t i = 0; i < tt.gTokE.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gTokE[i]))
					return false;
			for (size_t i = 0; i < tt.gLmBias.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gLmBias[i]))
					return false;
			for (size_t i = 0; i < tt.gWIn.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gWIn[i]))
					return false;
			for (size_t i = 0; i < tt.gBIn.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gBIn[i]))
					return false;
			for (size_t i = 0; i < tt.gWOut.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gWOut[i]))
					return false;
			for (size_t i = 0; i < tt.gBOut.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gBOut[i]))
					return false;
			for (size_t li = 0; li < tt.blocks.size(); ++li)
			{
				const TensorTransformerState::Block& b = tt.blocks[li];
				const std::vector<float>* gv[] = {&b.gLn1Gamma, &b.gLn1Beta, &b.gWq, &b.gWk, &b.gWv, &b.gWo, &b.gBq, &b.gBk, &b.gBv, &b.gBo,
				                                  &b.gLn2Gamma, &b.gLn2Beta, &b.gW1, &b.gW2, &b.gB1, &b.gB2};
				for (unsigned int gi = 0; gi < (sizeof(gv) / sizeof(gv[0])); ++gi)
				{
					const std::vector<float>& g = *gv[gi];
					for (size_t j = 0; j < g.size(); ++j)
						if (!glades::transformer_kernels::is_finite(g[j]))
							return false;
				}
			}
			return true;
		}

		static void scale_all_grads(TensorTransformerState& tt, float scale)
		{
			for (size_t i = 0; i < tt.gTokE.size(); ++i)
				tt.gTokE[i] *= scale;
			for (size_t i = 0; i < tt.gLmBias.size(); ++i)
				tt.gLmBias[i] *= scale;
			for (size_t i = 0; i < tt.gWIn.size(); ++i)
				tt.gWIn[i] *= scale;
			for (size_t i = 0; i < tt.gBIn.size(); ++i)
				tt.gBIn[i] *= scale;
			for (size_t i = 0; i < tt.gWOut.size(); ++i)
				tt.gWOut[i] *= scale;
			for (size_t i = 0; i < tt.gBOut.size(); ++i)
				tt.gBOut[i] *= scale;
			for (size_t li = 0; li < tt.blocks.size(); ++li)
			{
				TensorTransformerState::Block& b = tt.blocks[li];
				std::vector<float>* gv[] = {&b.gLn1Gamma, &b.gLn1Beta, &b.gWq, &b.gWk, &b.gWv, &b.gWo, &b.gBq, &b.gBk, &b.gBv, &b.gBo,
				                            &b.gLn2Gamma, &b.gLn2Beta, &b.gW1, &b.gW2, &b.gB1, &b.gB2};
				for (unsigned int gi = 0; gi < (sizeof(gv) / sizeof(gv[0])); ++gi)
				{
					std::vector<float>& g = *gv[gi];
					for (size_t j = 0; j < g.size(); ++j)
						g[j] *= scale;
				}
			}
		}
	};

	// Mixed precision (Transformer):
	// - FP32 master weights live in tt.* vectors (single source of truth).
	// - Optional low-precision copies are kept in tt.*Lowp and used for forward/backward matmuls.
	const bool mpEnable = trainingConfig.mixedPrecision.enable &&
	                      (trainingConfig.mixedPrecision.weightDType != glades::MixedPrecisionConfig::WEIGHT_F32);
	const bool mpUseLossScaling = mpEnable && trainingConfig.mixedPrecision.useLossScaling;
	const bool mpDynamicLossScaling = mpUseLossScaling && trainingConfig.mixedPrecision.dynamicLossScaling;
	if (mpEnable)
		MixedPrecisionHelper::ensure_transformer_lowp_weights(tt, trainingConfig);
	const bool useLowpWeights = mpEnable && tt.mpLowpReady;
	const int lowpDType = tt.mpLowpDType;

	// Attention scratch buffers (reused) to avoid per-head allocations.
	const unsigned int nKVHeads = (ttConst.nKVHeads > 0u ? ttConst.nKVHeads : nHeads);
	const unsigned int dHead = dModel / nHeads;
	const unsigned int dModelKV = nKVHeads * dHead;
	const unsigned int ff1Width = (ffnKind == static_cast<int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFF) : dFF;

	// Token LM metrics:
	// Accumulate mean NLL over non-pad tokens (natural log).
	double tokenLmNllSum = 0.0;
	unsigned long long tokenLmTokenCount = 0ULL;

	unsigned long long tokensProcessed = 0ULL;
	unsigned long long targetsProcessed = 0ULL; // token LM: non-pad targets; else: timesteps

	// Emit ~20 progress updates per epoch (plus final).
	unsigned int progressEverySeq = 1u;
	if (seqCount > 20u)
		progressEverySeq = seqCount / 20u;
	if (progressEverySeq == 0u)
		progressEverySeq = 1u;
	int64_t lastProgressMs = epochStartMs;
	static const int64_t kProgressIntervalMs = 5000; // 5s heartbeat

	for (unsigned int s = 0; s < seqCount; ++s)
	{
		const unsigned int T = isTrain ? di->getTrainSequenceLength(s) : di->getTestSequenceLength(s);
		if (T == 0u)
			continue;
		tokensProcessed += static_cast<unsigned long long>(T);

		// For sampled-softmax token LM, we do NOT allocate [T,vocab] logits/probs/dLogits.
		// Instead, logits/probs are sized [T,(1+K)] for K negatives per token.
		unsigned int scratchOutSize = outSize;
		unsigned int sampleCount = 0u;
		if (tokenLM && (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX))
		{
			const unsigned int K = static_cast<unsigned int>(tokenLmNegK);
			sampleCount = 1u + K;
			scratchOutSize = sampleCount;
		}

		// Guardrail: full-softmax token LM allocates O(T*vocab) buffers (logits+probs+dLogits).
		if (tokenLM && (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX) && !tokenLmAllowHuge)
		{
			const size_t Tv = static_cast<size_t>(T) * static_cast<size_t>(vocabSize);
			// logits + probs + dLogits ~= 3 buffers of float32
			const size_t bytes = Tv * 3u * sizeof(float);
			// Hard cap to stop pretending this backend trains real LLMs via full softmax.
			static const size_t kMaxSoftmaxScratchBytes = static_cast<size_t>(256ull * 1024ull * 1024ull);
			if (Tv > 0u && bytes > kMaxSoftmaxScratchBytes)
			{
				std::ostringstream oss;
				oss << "SGDHelper_TRANSFORMER: token LM full softmax would allocate ~" << (bytes / (1024ull * 1024ull))
				    << " MiB just for logits/probs/dLogits (T=" << T << ", vocab=" << vocabSize << "). "
				    << "Use sampled-softmax (tokenLmLossKind=SAMPLED) or set tokenLmAllowHugeFullSoftmax=1 to override.";
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, oss.str());
				running = false;
				return;
			}
		}

		transformerScratch.ensure(T, inputSize, scratchOutSize, dModel, dFF, dModelKV, nHeads, nLayers, ff1Width);

		// Load x[t] for this sequence into scratch.x (non-tokenLM).
		// In tokenLM mode, inputs are token ids (ints) and scratch.x is unused.
		std::vector<int> tokenIds;
		std::vector<int> targetIds;
		// Padding mask for attention in token LM mode:
		// keyAllowed[t] == 1 => timestep t participates as a key/value
		// keyAllowed[t] == 0 => timestep t is padding and must be masked out of attention
		std::vector<unsigned char> keyAllowed;
		if (tokenLM)
		{
			tokenIds.assign(T, 0);
			targetIds.assign(T, 0);
			// Ensure the (unused) float input view is deterministic/finite.
			std::fill(transformerScratch.x.begin(),
			          transformerScratch.x.begin() + (static_cast<size_t>(T) * static_cast<size_t>(inputSize)),
			          0.0f);
		}
		for (unsigned int t = 0; t < T; ++t)
		{
			if (tokenLM)
			{
				// Token LM requires first-class token-id accessors (no float casting fallback).
				int tid = 0;
				bool okTok = false;
				if (isTrain)
					okTok = di->getTrainSequenceTokenId(s, t, tid);
				else
					okTok = di->getTestSequenceTokenId(s, t, tid);
				if (!okTok)
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
					                            "SGDHelper_TRANSFORMER: token LM requires integer token-id accessors on DataInput (get*SequenceTokenId)");
					running = false;
					return;
				}
				if (tid < 0 || static_cast<unsigned int>(tid) >= vocabSize)
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token id out of range");
					running = false;
					return;
				}
				tokenIds[t] = tid;

				// Expected token id (next token).
				int yid = padTokenId;
				bool okY = false;
				if (isTrain)
					okY = di->getTrainSequenceExpectedTokenId(s, t, yid);
				else
					okY = di->getTestSequenceExpectedTokenId(s, t, yid);
				if (!okY)
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
					                            "SGDHelper_TRANSFORMER: token LM requires integer expected-token-id accessors on DataInput (get*SequenceExpectedTokenId)");
					running = false;
					return;
				}
				targetIds[t] = yid;
			}
			else
			{
				const float* row = NULL;
				unsigned int rowSize = 0u;
				if (isTrain)
					di->getTrainSequenceRowView(s, t, row, rowSize);
				else
					di->getTestSequenceRowView(s, t, row, rowSize);
				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(inputSize);
				for (unsigned int i = 0; i < inputSize; ++i)
				{
					const float v = (row && i < rowSize) ? row[i] : 0.0f;
					if (!is_finite(v))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: non-finite input sequence value detected (NaN/Inf)");
						running = false;
						return;
					}
					transformerScratch.x[off + i] = v;
				}
			}
		}

		// Build key mask (token LM only). If padTokenId < 0, masking is disabled.
		keyAllowed.clear();
		if (tokenLM && padTokenId >= 0)
		{
			keyAllowed.assign(T, 1u);
			for (unsigned int t = 0; t < T; ++t)
				if (tokenIds[t] == padTokenId)
					keyAllowed[t] = 0u;
		}

		// Progress counters:
		// - tokenLM: count valid target tokens (non-pad) for loss normalization/throughput.
		// - non-tokenLM: count timesteps.
		if (tokenLM)
		{
			unsigned int valid = 0u;
			for (unsigned int t = 0; t < T; ++t)
			{
				const int yid = targetIds[t];
				if (padTokenId >= 0 && yid == padTokenId)
					continue;
				if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
					continue;
				++valid;
			}
			targetsProcessed += static_cast<unsigned long long>(valid);
		}
		else
		{
			targetsProcessed += static_cast<unsigned long long>(T);
		}

		// === Forward ===
		// Input to h
		if (tokenLM)
		{
			// Embedding lookup.
			const bool haveLowpE = useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty();
			for (unsigned int t = 0; t < T; ++t)
			{
				const int tid = tokenIds[t];
				// tid already validated when loaded.
				const size_t eOff = static_cast<size_t>(tid) * static_cast<size_t>(dModel);
				const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
				for (unsigned int i = 0; i < dModel; ++i)
				{
					transformerScratch.h[hOff + i] = haveLowpE ? glades::transformer_kernels::lowp_to_float(tt.tokELowp[eOff + i], lowpDType)
					                                          : tt.tokE[eOff + i];
				}
			}
		}
		else
		{
			// Input projection to h
			linear_forward_maybe_lowp(transformerScratch.x.data(), T, inputSize, tt.WIn, tt.WInLowp, useLowpWeights, lowpDType, tt.bIn, dModel,
			                          transformerScratch.h.data());
		}
		if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
		{
			// Avoid recomputing pow()-derived denominators per sequence.
			transformerPosEncCache.ensureSinusoidal(dModel);
			add_positional_encoding(transformerScratch.h.empty() ? NULL : &transformerScratch.h[0], T, dModel,
			                        transformerPosEncCache.sinInvDenomPair);
		}

		// RoPE precompute (used when positionalEncoding==POSENC_ROPE).
		const bool useRope = (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE));
		unsigned int ropeDim = dHead;
		if (ropeDimOverride > 0)
		{
			const unsigned int rd = static_cast<unsigned int>(ropeDimOverride);
			ropeDim = (rd < ropeDim) ? rd : ropeDim;
		}
		// Must be even.
		if ((ropeDim % 2u) != 0u)
			ropeDim -= 1u;
		const std::vector<double>* ropeInvFreq = NULL;
		if (useRope && ropeDim >= 2u)
		{
			// Avoid recomputing pow()-derived invFreq per sequence.
			transformerPosEncCache.ensureRope(ropeDim, ropeTheta);
			ropeInvFreq = &transformerPosEncCache.ropeInvFreq;
		}
		const unsigned int groupSize = (nKVHeads > 0u) ? (nHeads / nKVHeads) : 0u;

		// Per-layer forward
		for (unsigned int li = 0; li < nLayers; ++li)
		{
			const TensorTransformerState::Block& b = tt.blocks[li];
			const float* hIn = (li == 0u) ? transformerScratch.h.data()
			                              : (transformerScratch.hAfterFF.data() + (static_cast<size_t>(li - 1u) * static_cast<size_t>(T) * static_cast<size_t>(dModel)));

			float* x1 = transformerScratch.x1.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			float* ln1Mean = transformerScratch.ln1Mean.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
			float* ln1InvStd = transformerScratch.ln1InvStd.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				// Store invRms in ln1InvStd; ln1Mean is unused for RMSNorm.
				std::fill(ln1Mean, ln1Mean + T, 0.0f);
				glades::transformer_kernels::rmsnorm_forward_rows(hIn, T, dModel, b.ln1Gamma, b.ln1Beta, lnEps, x1, ln1InvStd);
			}
			else
			{
				glades::transformer_kernels::layernorm_forward_rows(hIn, T, dModel, b.ln1Gamma, b.ln1Beta, lnEps, x1, ln1Mean, ln1InvStd);
			}

			float* Q = transformerScratch.Q.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			float* K = transformerScratch.K.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
			float* V = transformerScratch.V.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));

			linear_forward_maybe_lowp(x1, T, dModel, b.Wq, b.WqLowp, useLowpWeights, lowpDType, b.bq, dModel, Q);
			linear_forward_maybe_lowp(x1, T, dModel, b.Wk, b.WkLowp, useLowpWeights, lowpDType, b.bk, dModelKV, K);
			linear_forward_maybe_lowp(x1, T, dModel, b.Wv, b.WvLowp, useLowpWeights, lowpDType, b.bv, dModelKV, V);

			// RoPE on Q/K (in-place) if enabled.
			// IMPORTANT: We rotate the packed Q/K buffers directly to avoid per-head gather/scatter.
			// Backprop will invert-rotate gradients back into the linear-projection space.
			if (useRope && ropeInvFreq)
			{
				for (unsigned int h = 0; h < nHeads; ++h)
					glades::transformer_kernels::rope_apply_inplace_strided(
					    Q + static_cast<size_t>(h) * static_cast<size_t>(dHead), T, dModel, dHead, ropeDim, *ropeInvFreq, /*inverse*/ false);
				for (unsigned int hk = 0; hk < nKVHeads; ++hk)
					glades::transformer_kernels::rope_apply_inplace_strided(
					    K + static_cast<size_t>(hk) * static_cast<size_t>(dHead), T, dModelKV, dHead, ropeDim, *ropeInvFreq, /*inverse*/ false);
			}

			// Multi-head attention: for each head compute attention and concatenate.
			float* attnConcat = transformerScratch.attnConcat.data() +
			                    (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			std::fill(attnConcat, attnConcat + (static_cast<size_t>(T) * static_cast<size_t>(dModel)), 0.0f);
			for (unsigned int h = 0; h < nHeads; ++h)
			{
				const unsigned int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0u ? (h / groupSize) : 0u);
				glades::transformer_ops::scaled_dot_product_attention_forward_flash_strided(
				    /*Qbase*/ Q + static_cast<size_t>(h) * static_cast<size_t>(dHead),
				    /*qStride*/ dModel,
				    /*Kbase*/ K + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead),
				    /*kStride*/ dModelKV,
				    /*Vbase*/ V + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead),
				    /*vStride*/ dModelKV,
				    T,
				    /*dK*/ dHead,
				    /*dV*/ dHead,
				    causal,
				    /*Obase*/ attnConcat + static_cast<size_t>(h) * static_cast<size_t>(dHead),
				    /*oStride*/ dModel,
				    keyAllowed.empty() ? NULL : &keyAllowed[0]);
			}

			// Apply Wo
			float* attnOut = transformerScratch.attnOut.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			linear_forward_maybe_lowp(attnConcat, T, dModel, b.Wo, b.WoLowp, useLowpWeights, lowpDType, b.bo, dModel, attnOut);

			// Residual add
			float* hAfterAttn = transformerScratch.hAfterAttn.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dModel); ++i)
				hAfterAttn[i] = hIn[i] + attnOut[i];

			// LN2
			float* x2 = transformerScratch.x2.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			float* ln2Mean = transformerScratch.ln2Mean.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
			float* ln2InvStd = transformerScratch.ln2InvStd.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				std::fill(ln2Mean, ln2Mean + T, 0.0f);
				glades::transformer_kernels::rmsnorm_forward_rows(hAfterAttn, T, dModel, b.ln2Gamma, b.ln2Beta, lnEps, x2, ln2InvStd);
			}
			else
			{
				glades::transformer_kernels::layernorm_forward_rows(hAfterAttn, T, dModel, b.ln2Gamma, b.ln2Beta, lnEps, x2, ln2Mean, ln2InvStd);
			}

			// FFN
			float* ff1 = transformerScratch.ff1.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
			float* ff1Act = transformerScratch.ff1Act.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dFF));
			linear_forward_maybe_lowp(x2, T, dModel, b.W1, b.W1Lowp, useLowpWeights, lowpDType, b.b1, ff1Width, ff1);
			if (ffnKind == static_cast<int>(glades::TransformerRunConfig::FFN_SWIGLU))
			{
				// Packed [gate, up] -> SiLU(gate) * up
				for (unsigned int t = 0; t < T; ++t)
				{
					const size_t preOff = static_cast<size_t>(t) * static_cast<size_t>(ff1Width);
					const size_t outOff = static_cast<size_t>(t) * static_cast<size_t>(dFF);
					for (unsigned int i = 0; i < dFF; ++i)
					{
						const float gatePre = ff1[preOff + i];
						const float upPre = ff1[preOff + static_cast<size_t>(dFF) + i];
						ff1Act[outOff + i] = glades::transformer_ops::silu(gatePre) * upPre;
					}
				}
			}
			else
			{
				for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dFF); ++i)
				{
					const float x = ff1[i];
					ff1Act[i] = (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
					                ? glades::transformer_ops::gelu(x)
					                : glades::transformer_ops::relu(x);
				}
			}

			float* ffOut = transformerScratch.ffOut.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			linear_forward_maybe_lowp(ff1Act, T, dFF, b.W2, b.W2Lowp, useLowpWeights, lowpDType, b.b2, dModel, ffOut);

			// Residual add
			float* hAfterFF = transformerScratch.hAfterFF.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dModel); ++i)
				hAfterFF[i] = hAfterAttn[i] + ffOut[i];
		}

		const float* hFinal = transformerScratch.hAfterFF.data() + (static_cast<size_t>(nLayers - 1u) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

		// Output head logits
		if (tokenLM)
		{
			if (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX)
			{
				// Token LM tied embedding head: logits[t, v] = dot(hFinal[t], E[v]) + lmBias[v]
				if (useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty())
				{
					glades::transformer_kernels::tied_embedding_logits_forward_rows_lowp(hFinal, T, dModel, &tt.tokELowp[0], lowpDType, tt.lmBias,
					                                                                    vocabSize, transformerScratch.logits.data());
				}
				else
				{
					glades::transformer_kernels::tied_embedding_logits_forward_rows(hFinal, T, dModel, tt.tokE, tt.lmBias, vocabSize,
					                                                               transformerScratch.logits.data());
				}
			}
			else
			{
				// Sampled-softmax: compute logits only for {target + negatives}.
				// Column 0 is always the target id; remaining columns are uniform negatives.
				const unsigned int S = scratchOutSize;
				for (unsigned int t = 0; t < T; ++t)
				{
					const int yid = targetIds[t];
					if (padTokenId >= 0 && yid == padTokenId)
						continue;
					if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
						continue;

					const size_t off = static_cast<size_t>(t) * static_cast<size_t>(S);
					transformerScratch.tokenLmSampleIds[off + 0u] = yid;
					// Negatives (allow duplicates; bias is acceptable for this reference objective).
					for (unsigned int j = 1u; j < S; ++j)
					{
						int nid = yid;
						// Avoid trivial collision with target.
						for (int tries = 0; tries < 4 && nid == yid; ++tries)
							nid = glades::rng::uniform_int(rngEngine, 0, static_cast<int>(vocabSize) - 1);
						if (nid == yid)
							nid = (yid + 1) % static_cast<int>(vocabSize);
						transformerScratch.tokenLmSampleIds[off + j] = nid;
					}

					// Logits for sampled ids
					const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
					const bool haveLowpE = useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty();
					for (unsigned int j = 0u; j < S; ++j)
					{
						const int vid = transformerScratch.tokenLmSampleIds[off + j];
						const size_t eOff = static_cast<size_t>(vid) * static_cast<size_t>(dModel);
						double acc = static_cast<double>(tt.lmBias[static_cast<size_t>(vid)]);
						for (unsigned int i = 0; i < dModel; ++i)
						{
							const float ev = haveLowpE ? glades::transformer_kernels::lowp_to_float(tt.tokELowp[eOff + i], lowpDType) : tt.tokE[eOff + i];
							acc += static_cast<double>(hFinal[hOff + i]) * static_cast<double>(ev);
						}
						transformerScratch.logits[off + j] = static_cast<float>(acc);
					}
					// Softmax over the sampled set into probs (for loss/grad).
					glades::transformer_kernels::softmax_stable_into(&transformerScratch.logits[off], static_cast<size_t>(S),
					                                                &transformerScratch.probs[off]);
				}
			}
		}
		else
		{
			linear_forward_maybe_lowp(hFinal, T, dModel, tt.WOut, tt.WOutLowp, useLowpWeights, lowpDType, tt.bOut, outSize,
			                          transformerScratch.logits.data());
		}

		// Output activation
		if (tokenLM && (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX))
		{
			// Sampled-softmax already computed probs in the tokenLM branch above.
		}
		else if (((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1u))
		{
			// Per-timestep softmax (no per-timestep allocations, no extra buffers).
			for (unsigned int t = 0; t < T; ++t)
			{
				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(outSize);
				glades::transformer_kernels::softmax_stable_into(&transformerScratch.logits[off], static_cast<size_t>(outSize),
				                                                &transformerScratch.probs[off]);
			}
		}
		else if ((costFx == GMath::CLASSIFICATION) && (outSize == 1u))
		{
			// Sigmoid
			for (unsigned int t = 0; t < T; ++t)
			{
				const float z = transformerScratch.logits[static_cast<size_t>(t) * static_cast<size_t>(outSize)];
				transformerScratch.probs[static_cast<size_t>(t) * static_cast<size_t>(outSize)] = GMath::squash(z, GMath::SIGMOID, 0.0f);
			}
		}
		else
		{
			// Linear regression
			std::copy(transformerScratch.logits.begin(), transformerScratch.logits.end(), transformerScratch.probs.begin());
		}

		// === Metrics ===
		if (tokenLM)
		{
			if (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX)
			{
				// Next-token cross entropy + top-1 accuracy.
				for (unsigned int t = 0; t < T; ++t)
				{
					const int yid = targetIds[t];
					if (padTokenId >= 0 && yid == padTokenId)
						continue;
					if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
						continue;

					const size_t off = static_cast<size_t>(t) * static_cast<size_t>(vocabSize);
					// argmax
					unsigned int argm = 0u;
					float best = transformerScratch.probs[off + 0u];
					for (unsigned int v = 1u; v < vocabSize; ++v)
					{
						const float p = transformerScratch.probs[off + v];
						if (p > best)
						{
							best = p;
							argm = v;
						}
					}
					++clsTotal;
					if (argm == static_cast<unsigned int>(yid))
						++clsCorrect;

					const float py = clamp_prob01(transformerScratch.probs[off + static_cast<unsigned int>(yid)]);
					tokenLmNllSum += -log(static_cast<double>(py));
					++tokenLmTokenCount;

					// Results: store [expectedTokenId, predictedTokenId] for last timestep processed.
					results.clear();
					results.addFloat(static_cast<float>(yid));
					results.addFloat(static_cast<float>(argm));
				}
			}
			else
			{
				// Sampled-softmax: loss is NOT exact NLL; do not report as perplexity (handled in Trainer).
				const unsigned int S = scratchOutSize;
				for (unsigned int t = 0; t < T; ++t)
				{
					const int yid = targetIds[t];
					if (padTokenId >= 0 && yid == padTokenId)
						continue;
					if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
						continue;

					const size_t off = static_cast<size_t>(t) * static_cast<size_t>(S);
					const float py = clamp_prob01(transformerScratch.probs[off + 0u]); // col 0 is target
					tokenLmNllSum += -log(static_cast<double>(py));
					++tokenLmTokenCount;

					// Results: expected token, predicted token is unknown without full vocab.
					results.clear();
					results.addFloat(static_cast<float>(yid));
					results.addFloat(-1.0f);
				}
			}
		}
		else
		{
			// Accumulate loss and confusion per timestep like recurrent paths.
			for (unsigned int t = 0; t < T; ++t)
			{
				results.clear();
				const float* expRow = NULL;
				unsigned int expSize = 0u;
				if (isTrain)
					di->getTrainSequenceExpectedRowView(s, t, expRow, expSize);
				else
					di->getTestSequenceExpectedRowView(s, t, expRow, expSize);

				const unsigned int N = dataSize;
				const double denom = (N > 0 && outSize > 0) ? static_cast<double>(N) * static_cast<double>(outSize) : 1.0;
				const bool useSoftmax = ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1u);
				double loss = 0.0;

				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(outSize);
				for (unsigned int k = 0; k < outSize; ++k)
				{
					const float expv = (expRow && k < expSize) ? expRow[k] : 0.0f;
					const float pred = transformerScratch.probs[off + k];
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
						overallTotalError += static_cast<float>(GMath::outputNodeCost(expv, pred, static_cast<float>(denom), costFx));
					}
				}

				if (useSoftmax && N > 0)
					overallTotalError += static_cast<float>(loss / static_cast<double>(N));
				if ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL))
					confusionMatrix.addResult(results);
			}
		}

		// === Backward + grad accumulation ===
		if (isTrain)
		{
			const float lossScale = (mpUseLossScaling ? tt.mpLossScale : 1.0f);
			if (seqInBatch == 0u)
			{
				clearGrads();
				timeStepsInBatch = 0u;
			}

			// dLogits (delta) per timestep (reused scratch to avoid allocations)
			std::vector<float, glades::AlignedAllocator<float, 64> >& dLogits = transformerScratch.dLogits;
			if (dLogits.size() != (static_cast<size_t>(T) * static_cast<size_t>(scratchOutSize)))
				dLogits.resize(static_cast<size_t>(T) * static_cast<size_t>(scratchOutSize));
			std::fill(dLogits.begin(), dLogits.end(), 0.0f);

			// Upstream gradient dH for current layer output (reused scratch to avoid allocations)
			std::vector<float, glades::AlignedAllocator<float, 64> >& dH = transformerScratch.dH;
			if (dH.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
				dH.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
			if (tokenLM)
			{
				unsigned int validTargetsThisSeq = 0u;
				std::fill(dH.begin(), dH.end(), 0.0f);
				if (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX)
				{
					for (unsigned int t = 0; t < T; ++t)
					{
						const int yid = targetIds[t];
						if (padTokenId >= 0 && yid == padTokenId)
							continue;
						if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
							continue;
						++validTargetsThisSeq;
						const size_t off = static_cast<size_t>(t) * static_cast<size_t>(vocabSize);
						for (unsigned int v = 0; v < vocabSize; ++v)
							dLogits[off + v] = transformerScratch.probs[off + v];
						dLogits[off + static_cast<unsigned int>(yid)] -= 1.0f;
					}

					// Backprop tied LM head (dense).
					const bool haveLowpE = useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty();
					for (unsigned int t = 0; t < T; ++t)
					{
						const size_t off = static_cast<size_t>(t) * static_cast<size_t>(vocabSize);
						const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
						for (unsigned int v = 0; v < vocabSize; ++v)
						{
							const float dz = clip_maybe(dLogits[off + v], gradClip) * lossScale;
							if (dz == 0.0f)
								continue;
							tt.gLmBias[v] += dz;
							const size_t eOff = static_cast<size_t>(v) * static_cast<size_t>(dModel);
							for (unsigned int i = 0; i < dModel; ++i)
							{
								tt.gTokE[eOff + i] += dz * hFinal[hOff + i];
								const float ev = haveLowpE ? glades::transformer_kernels::lowp_to_float(tt.tokELowp[eOff + i], lowpDType) : tt.tokE[eOff + i];
								dH[hOff + i] += dz * ev;
							}
						}
					}
				}
				else
				{
					// Sampled-softmax backprop (sparse): only update sampled vocab rows.
					const unsigned int S = scratchOutSize;
					for (unsigned int t = 0; t < T; ++t)
					{
						const int yid = targetIds[t];
						if (padTokenId >= 0 && yid == padTokenId)
							continue;
						if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
							continue;
						++validTargetsThisSeq;

						const size_t off = static_cast<size_t>(t) * static_cast<size_t>(S);
						for (unsigned int j = 0u; j < S; ++j)
							dLogits[off + j] = transformerScratch.probs[off + j];
						dLogits[off + 0u] -= 1.0f; // col 0 is target

						const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
						const bool haveLowpE = useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty();
						for (unsigned int j = 0u; j < S; ++j)
						{
							const float dz = clip_maybe(dLogits[off + j], gradClip) * lossScale;
							if (dz == 0.0f)
								continue;
							const int vid = transformerScratch.tokenLmSampleIds[off + j];
							if (vid < 0 || static_cast<unsigned int>(vid) >= vocabSize)
								continue;
							tt.gLmBias[static_cast<size_t>(vid)] += dz;
							const size_t eOff = static_cast<size_t>(vid) * static_cast<size_t>(dModel);
							for (unsigned int i = 0; i < dModel; ++i)
							{
								tt.gTokE[eOff + i] += dz * hFinal[hOff + i];
								const float ev = haveLowpE ? glades::transformer_kernels::lowp_to_float(tt.tokELowp[eOff + i], lowpDType) : tt.tokE[eOff + i];
								dH[hOff + i] += dz * ev;
							}
						}
					}
				}

				// Batch divisor must match token LM loss normalization:
				// average by the number of non-pad target tokens, not raw sequence length.
				timeStepsInBatch += validTargetsThisSeq;
			}
			else
			{
				for (unsigned int t = 0; t < T; ++t)
				{
					const float* expRow = NULL;
					unsigned int expSize = 0u;
					di->getTrainSequenceExpectedRowView(s, t, expRow, expSize);
					const size_t off = static_cast<size_t>(t) * static_cast<size_t>(outSize);
					for (unsigned int k = 0; k < outSize; ++k)
					{
						const float expv = (expRow && k < expSize) ? expRow[k] : 0.0f;
						const float pred = transformerScratch.probs[off + k];
						float d = 0.0f;
						const bool useSoftmax = ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1u);
						if (useSoftmax)
						{
							d = pred - expv;
						}
						else if ((costFx == GMath::CLASSIFICATION) && (outSize == 1u))
						{
							// sigmoid + BCE => dL/dz = p - y
							d = pred - expv;
						}
						else
						{
							// linear regression
							d = GMath::costErrDer(expv, pred, costFx);
						}
						dLogits[off + k] = clip_maybe(d, gradClip) * lossScale;
					}
				}

				// Backprop output projection: gWOut/gBOut, dHFinal
				linear_backward_accum_maybe_lowp(hFinal, dLogits.data(), T, dModel, outSize, tt.gWOut, tt.gBOut, tt.WOut, tt.WOutLowp, useLowpWeights,
				                                 lowpDType, dH.data());

				// Non-tokenLM: average by timesteps (historical behavior).
				timeStepsInBatch += T;
			}

			// Backprop through blocks (reverse)
			for (int li = static_cast<int>(nLayers) - 1; li >= 0; --li)
			{
				TensorTransformerState::Block& b = tt.blocks[static_cast<size_t>(li)];
				const float* hIn = (li == 0) ? transformerScratch.h.data()
				                             : (transformerScratch.hAfterFF.data() + (static_cast<size_t>(li - 1) * static_cast<size_t>(T) * static_cast<size_t>(dModel)));

				const float* hAfterAttn = transformerScratch.hAfterAttn.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
				const float* x1 = transformerScratch.x1.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
				const float* x2 = transformerScratch.x2.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
				const float* ff1 = transformerScratch.ff1.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
				const float* ff1Act = transformerScratch.ff1Act.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dFF));

				// dH is gradient w.r.t hAfterFF
				// Split residual: hAfterFF = hAfterAttn + ffOut
				// dFFOut and the residual path into hAfterAttn both start as dH.
				// Use scratch buffers to avoid per-layer allocations.
				std::vector<float, glades::AlignedAllocator<float, 64> >& dHAfterAttn = transformerScratch.dH2;
				if (dHAfterAttn.size() != dH.size())
					dHAfterAttn.resize(dH.size());
				std::copy(dH.begin(), dH.end(), dHAfterAttn.begin());

				// FFN backward
				// ffOut = W2 * ff1Act + b2
				// ff1Act = activation(ff1)  (MLP) or SwiGLU product (SwiGLU)
				std::vector<float, glades::AlignedAllocator<float, 64> >& dFF1Act = transformerScratch.dFF1Act;
				if (dFF1Act.size() != (static_cast<size_t>(T) * static_cast<size_t>(dFF)))
					dFF1Act.resize(static_cast<size_t>(T) * static_cast<size_t>(dFF));
				linear_backward_accum_maybe_lowp(ff1Act, dH.data(), T, dFF, dModel, b.gW2, b.gB2, b.W2, b.W2Lowp, useLowpWeights, lowpDType,
				                                 dFF1Act.data());

				std::vector<float, glades::AlignedAllocator<float, 64> >& dX2 = transformerScratch.dX2;
				if (dX2.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
					dX2.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
				if (ffnKind == static_cast<int>(glades::TransformerRunConfig::FFN_SWIGLU))
				{
					// Backprop through SwiGLU:
					// out = SiLU(gatePre) * upPre
					// ff1 packed as [gatePre (dFF), upPre (dFF)]
					std::vector<float, glades::AlignedAllocator<float, 64> >& dFF1Cat = transformerScratch.dFF1Cat;
					if (dFF1Cat.size() != (static_cast<size_t>(T) * static_cast<size_t>(ff1Width)))
						dFF1Cat.resize(static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
					std::fill(dFF1Cat.begin(), dFF1Cat.end(), 0.0f);
					for (unsigned int t = 0; t < T; ++t)
					{
						const size_t preOff = static_cast<size_t>(t) * static_cast<size_t>(ff1Width);
						const size_t outOff = static_cast<size_t>(t) * static_cast<size_t>(dFF);
						for (unsigned int i = 0; i < dFF; ++i)
						{
							const float gatePre = ff1[preOff + i];
							const float upPre = ff1[preOff + static_cast<size_t>(dFF) + i];
							const float siluVal = glades::transformer_ops::silu(gatePre);
							const float dOut = dFF1Act[outOff + i];
							// dGatePre = dOut * upPre * silu'(gatePre)
							dFF1Cat[preOff + i] = dOut * upPre * glades::transformer_ops::silu_deriv(gatePre);
							// dUpPre = dOut * SiLU(gatePre)
							dFF1Cat[preOff + static_cast<size_t>(dFF) + i] = dOut * siluVal;
						}
					}
					linear_backward_accum_maybe_lowp(x2, dFF1Cat.data(), T, dModel, ff1Width, b.gW1, b.gB1, b.W1, b.W1Lowp, useLowpWeights, lowpDType,
					                                 dX2.data());
				}
				else
				{
					// FFN activation backprop (ReLU or GELU)
					for (size_t i = 0; i < dFF1Act.size(); ++i)
					{
						const float pre = ff1[i];
						const float deriv =
						    (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
						        ? glades::transformer_ops::gelu_deriv(pre)
						        : glades::transformer_ops::relu_deriv_from_y(ff1Act[i]);
						dFF1Act[i] = dFF1Act[i] * deriv;
					}
					linear_backward_accum_maybe_lowp(x2, dFF1Act.data(), T, dModel, dFF, b.gW1, b.gB1, b.W1, b.W1Lowp, useLowpWeights, lowpDType,
					                                 dX2.data());
				}

				// LN2 backward: x2 = LN(hAfterAttn)
				std::vector<float, glades::AlignedAllocator<float, 64> >& dHAfterAttnFromLN = transformerScratch.dHAfterAttnFromLN;
				if (dHAfterAttnFromLN.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
					dHAfterAttnFromLN.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
				std::fill(dHAfterAttnFromLN.begin(), dHAfterAttnFromLN.end(), 0.0f);
				const float* ln2Mean = transformerScratch.ln2Mean.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
				const float* ln2InvStd = transformerScratch.ln2InvStd.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
				{
					glades::transformer_kernels::rmsnorm_backward_rows_accum(hAfterAttn, dX2.data(), T, dModel, b.ln2Gamma, ln2InvStd,
					                                                        dHAfterAttnFromLN.data(), b.gLn2Gamma, b.gLn2Beta);
				}
				else
				{
					glades::transformer_kernels::layernorm_backward_rows_accum(hAfterAttn, dX2.data(), T, dModel, b.ln2Gamma, ln2Mean, ln2InvStd,
					                                                          dHAfterAttnFromLN.data(), b.gLn2Gamma, b.gLn2Beta);
				}

				// Combine gradients to hAfterAttn
				for (size_t i = 0; i < dHAfterAttn.size(); ++i)
					dHAfterAttn[i] += dHAfterAttnFromLN[i];

				// Split residual at attention: hAfterAttn = hIn + attnOut
				// Use dHAfterAttn as dAttnOut and copy into dH for the residual-to-hIn path.
				std::copy(dHAfterAttn.begin(), dHAfterAttn.end(), dH.begin());

			// Need attnConcat to backprop Wo. Use the cached per-layer concatenation from forward
			// (O(nLayers*T*dModel) memory) instead of storing full attention probability matrices.
			const float* attnConcat = transformerScratch.attnConcat.data() +
			                          (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

				// Backprop Wo: attnOut = Wo*attnConcat + bo
				std::vector<float, glades::AlignedAllocator<float, 64> >& dAttnConcat = transformerScratch.dAttnConcat;
				if (dAttnConcat.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
					dAttnConcat.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
				linear_backward_accum_maybe_lowp(attnConcat, dHAfterAttn.data(), T, dModel, dModel, b.gWo, b.gBo, b.Wo, b.WoLowp, useLowpWeights,
				                                 lowpDType, dAttnConcat.data());

				// Backprop attention per head directly on packed Q/K/V buffers:
				// - no gather/scatter temporaries
				// - no per-head allocations
				// - KV grads accumulate correctly when using GQA (nKVHeads < nHeads)
			const float* Vfull = transformerScratch.V.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
			const float* Qfull = transformerScratch.Q.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			const float* Kfull = transformerScratch.K.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
				std::vector<float, glades::AlignedAllocator<float, 64> >& dQfull = transformerScratch.dQfull;
				std::vector<float, glades::AlignedAllocator<float, 64> >& dKfull = transformerScratch.dKfull;
				std::vector<float, glades::AlignedAllocator<float, 64> >& dVfull = transformerScratch.dVfull;
				if (dQfull.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
					dQfull.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
				if (dKfull.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModelKV)))
					dKfull.resize(static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
				if (dVfull.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModelKV)))
					dVfull.resize(static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
				std::fill(dQfull.begin(), dQfull.end(), 0.0f);
				std::fill(dKfull.begin(), dKfull.end(), 0.0f);
				std::fill(dVfull.begin(), dVfull.end(), 0.0f);
				for (unsigned int h = 0; h < nHeads; ++h)
				{
					const unsigned int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0u ? (h / groupSize) : 0u);
					glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_strided(
					    /*Qbase*/ Qfull + static_cast<size_t>(h) * static_cast<size_t>(dHead),
					    /*qStride*/ dModel,
					    /*Kbase*/ Kfull + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead),
					    /*kStride*/ dModelKV,
					    /*Vbase*/ Vfull + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead),
					    /*vStride*/ dModelKV,
					    /*dObase*/ dAttnConcat.data() + static_cast<size_t>(h) * static_cast<size_t>(dHead),
					    /*dOStride*/ dModel,
					    T,
					    /*dK*/ dHead,
					    /*dV*/ dHead,
					    causal,
					    /*dQbase*/ dQfull.data() + static_cast<size_t>(h) * static_cast<size_t>(dHead),
					    /*dQStride*/ dModel,
					    /*dKbase*/ dKfull.data() + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead),
					    /*dKStride*/ dModelKV,
					    /*dVbase*/ dVfull.data() + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead),
					    /*dVStride*/ dModelKV,
					    keyAllowed.empty() ? NULL : &keyAllowed[0]);
				}

				// Backprop through RoPE rotation (inverse rotation on gradients).
				// Q/K were rotated in-place before the forward attention, so attention backward produces gradients
				// in the rotated space; invert-rotate to map gradients back to the linear-projection outputs.
				if (useRope && ropeInvFreq)
				{
					for (unsigned int h = 0; h < nHeads; ++h)
						glades::transformer_kernels::rope_apply_inplace_strided(
						    dQfull.data() + static_cast<size_t>(h) * static_cast<size_t>(dHead), T, dModel, dHead, ropeDim, *ropeInvFreq, /*inverse*/ true);
					for (unsigned int hk = 0; hk < nKVHeads; ++hk)
						glades::transformer_kernels::rope_apply_inplace_strided(
						    dKfull.data() + static_cast<size_t>(hk) * static_cast<size_t>(dHead), T, dModelKV, dHead, ropeDim, *ropeInvFreq, /*inverse*/ true);
				}

				// Backprop Q/K/V linear projections into x1
				std::vector<float, glades::AlignedAllocator<float, 64> >& dX1 = transformerScratch.dX1;
				std::vector<float, glades::AlignedAllocator<float, 64> >& dXtmp = transformerScratch.dXtmp;
				if (dX1.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
					dX1.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
				if (dXtmp.size() != dX1.size())
					dXtmp.resize(dX1.size());
				std::fill(dX1.begin(), dX1.end(), 0.0f);
				// q = Wq*x1 + bq
				{
					linear_backward_accum_maybe_lowp(x1, dQfull.data(), T, dModel, dModel, b.gWq, b.gBq, b.Wq, b.WqLowp, useLowpWeights, lowpDType,
					                                 dXtmp.data());
					for (size_t i = 0; i < dX1.size(); ++i) dX1[i] += dXtmp[i];
				}
				{
					linear_backward_accum_maybe_lowp(x1, dKfull.data(), T, dModel, dModelKV, b.gWk, b.gBk, b.Wk, b.WkLowp, useLowpWeights, lowpDType,
					                                 dXtmp.data());
					for (size_t i = 0; i < dX1.size(); ++i) dX1[i] += dXtmp[i];
				}
				{
					linear_backward_accum_maybe_lowp(x1, dVfull.data(), T, dModel, dModelKV, b.gWv, b.gBv, b.Wv, b.WvLowp, useLowpWeights, lowpDType,
					                                 dXtmp.data());
					for (size_t i = 0; i < dX1.size(); ++i) dX1[i] += dXtmp[i];
				}

				// LN1 backward: x1 = LN(hIn)
				std::vector<float, glades::AlignedAllocator<float, 64> >& dHInFromLN = transformerScratch.dHInFromLN;
				if (dHInFromLN.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
					dHInFromLN.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
				std::fill(dHInFromLN.begin(), dHInFromLN.end(), 0.0f);
				const float* ln1Mean = transformerScratch.ln1Mean.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
				const float* ln1InvStd = transformerScratch.ln1InvStd.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
				{
					glades::transformer_kernels::rmsnorm_backward_rows_accum(hIn, dX1.data(), T, dModel, b.ln1Gamma, ln1InvStd,
					                                                        dHInFromLN.data(), b.gLn1Gamma, b.gLn1Beta);
				}
				else
				{
					glades::transformer_kernels::layernorm_backward_rows_accum(hIn, dX1.data(), T, dModel, b.ln1Gamma, ln1Mean, ln1InvStd,
					                                                          dHInFromLN.data(), b.gLn1Gamma, b.gLn1Beta);
				}

				// Combine into dH (currently the residual-to-hIn path from attention).
				for (size_t i = 0; i < dH.size(); ++i)
					dH[i] += dHInFromLN[i];
			} // layers

			// Backprop input projection: h = WIn*x + bIn (+ posEnc)
			// dH is gradient w.r.t h (posEnc has no params)
			if (tokenLM)
			{
				// Accumulate embedding grads for input embedding lookup: gE[token] += dH[t]
				for (unsigned int t = 0; t < T; ++t)
				{
					const int tid = tokenIds[t];
					const size_t eOff = static_cast<size_t>(tid) * static_cast<size_t>(dModel);
					const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
					for (unsigned int i = 0; i < dModel; ++i)
						tt.gTokE[eOff + i] += dH[hOff + i];
				}
			}
			else
			{
				std::vector<float, glades::AlignedAllocator<float, 64> >& dX = transformerScratch.dInput;
				if (dX.size() != (static_cast<size_t>(T) * static_cast<size_t>(inputSize)))
					dX.resize(static_cast<size_t>(T) * static_cast<size_t>(inputSize));
				linear_backward_accum_maybe_lowp(transformerScratch.x.data(), dH.data(), T, inputSize, dModel, tt.gWIn, tt.gBIn, tt.WIn, tt.WInLowp,
				                                 useLowpWeights, lowpDType, dX.data());
				(void)dX; // gradient w.r.t. raw inputs unused
			}

			++seqInBatch;
			if (seqInBatch >= seqBatchMax)
			{
				// Note: for token LM, timeStepsInBatch counts valid target tokens and may be 0
				// (e.g. all-pad targets). In that case applyBatch() is a no-op.
				if (timeStepsInBatch > 0u)
				{
					// Dynamic loss scaling: detect NaN/Inf in scaled grads, back off, and skip the step.
					if (mpUseLossScaling && !MixedPrecisionHelper::grads_all_finite(tt))
					{
						if (mpDynamicLossScaling)
						{
								const float prev = tt.mpLossScale;
							tt.mpLossScale *= trainingConfig.mixedPrecision.backoffFactor;
							if (tt.mpLossScale < trainingConfig.mixedPrecision.lossScaleMin)
								tt.mpLossScale = trainingConfig.mixedPrecision.lossScaleMin;
							tt.mpLossScaleGoodSteps = 0;
								if (logger)
								{
									std::ostringstream oss;
									oss << "event=nn_loss_scale_backoff";
									append_logfmt_kv(oss, "net_type", netType);
									append_logfmt_kv(oss, "epoch", epochIdx);
									append_logfmt_kv(oss, "optimizer_step", static_cast<unsigned long long>(tt.optimizerStep));
									append_logfmt_kv(oss, "loss_scale_prev", prev);
									append_logfmt_kv(oss, "loss_scale_new", tt.mpLossScale);
									logger->info("NNetwork", shmea::GString(oss.str().c_str()));
								}
						}
						clearGrads();
					}
					else
					{
						// Unscale gradients back to FP32 magnitude before optimizer/clipping.
						if (mpUseLossScaling && tt.mpLossScale != 1.0f)
							MixedPrecisionHelper::scale_all_grads(tt, 1.0f / tt.mpLossScale);

						if (!applyBatch(timeStepsInBatch))
							return;

						// Grow loss scale after a run of good steps.
						if (mpDynamicLossScaling)
						{
							tt.mpLossScaleGoodSteps += 1;
							if (tt.mpLossScaleGoodSteps >= trainingConfig.mixedPrecision.growthInterval)
							{
									const float prev = tt.mpLossScale;
								tt.mpLossScale *= trainingConfig.mixedPrecision.growthFactor;
								if (tt.mpLossScale > trainingConfig.mixedPrecision.lossScaleMax)
									tt.mpLossScale = trainingConfig.mixedPrecision.lossScaleMax;
								tt.mpLossScaleGoodSteps = 0;
									if (logger && tt.mpLossScale != prev)
									{
										std::ostringstream oss;
										oss << "event=nn_loss_scale_grow";
										append_logfmt_kv(oss, "net_type", netType);
										append_logfmt_kv(oss, "epoch", epochIdx);
										append_logfmt_kv(oss, "optimizer_step", static_cast<unsigned long long>(tt.optimizerStep));
										append_logfmt_kv(oss, "loss_scale_prev", prev);
										append_logfmt_kv(oss, "loss_scale_new", tt.mpLossScale);
										logger->info("NNetwork", shmea::GString(oss.str().c_str()));
									}
							}
						}

						// Sync low-precision weight copies from updated master weights.
						if (mpEnable)
							MixedPrecisionHelper::ensure_transformer_lowp_weights(tt, trainingConfig);
					}
				}
				seqInBatch = 0u;
				timeStepsInBatch = 0u;
			}

			// Save autotuning record for parity (use last layer's effective LR).
			{
				const float learningRate = skeleton->getLearningRate(nLayers) * lrScheduleMultiplier;
				shmea::GList nbRow;
				nbRow.addFloat(overallTotalAccuracy);
				nbRow.addFloat(learningRate);
				nbRecord.addRow(nbRow);
			}
		}

		// Periodic progress logs within the epoch (exclude last; a final log is emitted below).
		if (logger && (s + 1u) < seqCount)
		{
			const int64_t nowMs = getCurrentTimeMilliseconds();
			const bool dueBySeq = (((s + 1u) % progressEverySeq) == 0u);
			const bool dueByTime = ((nowMs - lastProgressMs) >= kProgressIntervalMs);
			if (!(dueBySeq || dueByTime))
				continue;
			lastProgressMs = nowMs;
			const double elapsedMs = static_cast<double>(nowMs - epochStartMs);
			const double tokPerSec = (elapsedMs > 0.0) ? (static_cast<double>(targetsProcessed) / (elapsedMs / 1000.0)) : 0.0;

			std::ostringstream oss;
			oss << "event=nn_epoch_progress";
			append_logfmt_kv(oss, "net_type", netType);
			append_logfmt_kv(oss, "run_type", std::string(isTrain ? "train" : "eval"));
			append_logfmt_kv(oss, "epoch", epochIdx);
			append_logfmt_kv(oss, "seq_done", s + 1u);
			append_logfmt_kv(oss, "seq_total", seqCount);
			append_logfmt_kv(oss, "tokens_seen", tokensProcessed);
			append_logfmt_kv(oss, "targets_seen", targetsProcessed);
			append_logfmt_kv(oss, "targets_per_sec", tokPerSec);
			if (tokenLM)
			{
				const double meanNll = (tokenLmTokenCount > 0ULL) ? (tokenLmNllSum / static_cast<double>(tokenLmTokenCount)) : 0.0;
				double ppl = 0.0;
				if (tokenLmTokenCount > 0ULL)
				{
					double arg = meanNll;
					if (arg > 80.0) arg = 80.0;
					if (arg < -80.0) arg = -80.0;
					ppl = exp(arg);
				}
				append_logfmt_kv(oss, "nll", meanNll);
				append_logfmt_kv(oss, "perplexity", ppl);
				append_logfmt_kv(oss, "acc_top1", (clsTotal > 0ULL) ? (100.0 * static_cast<double>(clsCorrect) / static_cast<double>(clsTotal)) : 0.0);
			}
			else
			{
				append_logfmt_kv(oss, "loss_so_far", overallTotalError);
			}
			append_logfmt_kv(oss, "lr_mult", lrScheduleMultiplier);
			append_logfmt_kv(oss, "grad_norm", lastGradNorm);
			append_logfmt_kv(oss, "grad_norm_scale", lastGradNormScale);
			append_logfmt_kv(oss, "optimizer_step", static_cast<unsigned long long>(tt.optimizerStep));
			if (mpUseLossScaling)
				append_logfmt_kv(oss, "loss_scale", tt.mpLossScale);

			logger->info("NNetwork", shmea::GString(oss.str().c_str()));
		}
	} // sequences

	// Progress log (final) and periodic log points.
	// NOTE: emit at end of the epoch regardless of seqCount%progressEverySeq to provide a clear heartbeat.
	if (logger)
	{
		const int64_t nowMs = getCurrentTimeMilliseconds();
		const double elapsedMs = static_cast<double>(nowMs - epochStartMs);
		const double meanNll = (tokenLmTokenCount > 0ULL) ? (tokenLmNllSum / static_cast<double>(tokenLmTokenCount)) : 0.0;
		double ppl = 0.0;
		if (tokenLM && tokenLmTokenCount > 0ULL)
		{
			double arg = meanNll;
			if (arg > 80.0) arg = 80.0;
			if (arg < -80.0) arg = -80.0;
			ppl = exp(arg);
		}
		const double tokPerSec = (elapsedMs > 0.0) ? (static_cast<double>(targetsProcessed) / (elapsedMs / 1000.0)) : 0.0;

		std::ostringstream oss;
		oss << "event=nn_epoch_progress";
		append_logfmt_kv(oss, "net_type", netType);
		append_logfmt_kv(oss, "run_type", std::string(isTrain ? "train" : "eval"));
		append_logfmt_kv(oss, "epoch", epochIdx);
		append_logfmt_kv(oss, "seq_done", seqCount);
		append_logfmt_kv(oss, "seq_total", seqCount);
		append_logfmt_kv(oss, "tokens_seen", tokensProcessed);
		append_logfmt_kv(oss, "targets_seen", targetsProcessed);
		append_logfmt_kv(oss, "targets_per_sec", tokPerSec);
		// Token LM running loss (mean NLL); for non-tokenLM use overallTotalError accumulator.
		if (tokenLM)
		{
			append_logfmt_kv(oss, "nll", meanNll);
			append_logfmt_kv(oss, "perplexity", ppl);
			append_logfmt_kv(oss, "acc_top1", (clsTotal > 0ULL) ? (100.0 * static_cast<double>(clsCorrect) / static_cast<double>(clsTotal)) : 0.0);
		}
		else
		{
			append_logfmt_kv(oss, "loss_so_far", overallTotalError);
		}
		append_logfmt_kv(oss, "lr_mult", lrScheduleMultiplier);
		append_logfmt_kv(oss, "grad_norm", lastGradNorm);
		append_logfmt_kv(oss, "grad_norm_scale", lastGradNormScale);
		append_logfmt_kv(oss, "optimizer_step", static_cast<unsigned long long>(tt.optimizerStep));
		if (mpUseLossScaling)
			append_logfmt_kv(oss, "loss_scale", tt.mpLossScale);

		logger->info("NNetwork", shmea::GString(oss.str().c_str()));
	}

	// Normalize token LM loss: mean NLL per non-pad token.
	// (Trainer expects overallTotalError to be an epoch-level mean-like quantity.)
	if (tokenLM)
	{
		if (tokenLmTokenCount > 0ULL)
			overallTotalError = static_cast<float>(tokenLmNllSum / static_cast<double>(tokenLmTokenCount));
		else
			overallTotalError = 0.0f;
	}

	// Flush partial minibatch
	if (isTrain && seqInBatch > 0u && timeStepsInBatch > 0u)
	{
		// Dynamic loss scaling: detect NaN/Inf in scaled grads, back off, and skip the step.
		if (mpUseLossScaling && !MixedPrecisionHelper::grads_all_finite(tt))
		{
			if (mpDynamicLossScaling)
			{
				tt.mpLossScale *= trainingConfig.mixedPrecision.backoffFactor;
				if (tt.mpLossScale < trainingConfig.mixedPrecision.lossScaleMin)
					tt.mpLossScale = trainingConfig.mixedPrecision.lossScaleMin;
				tt.mpLossScaleGoodSteps = 0;
			}
			clearGrads();
		}
		else
		{
			if (mpUseLossScaling && tt.mpLossScale != 1.0f)
				MixedPrecisionHelper::scale_all_grads(tt, 1.0f / tt.mpLossScale);
			if (!applyBatch(timeStepsInBatch))
				return;
			if (mpDynamicLossScaling)
			{
				tt.mpLossScaleGoodSteps += 1;
				if (tt.mpLossScaleGoodSteps >= trainingConfig.mixedPrecision.growthInterval)
				{
					tt.mpLossScale *= trainingConfig.mixedPrecision.growthFactor;
					if (tt.mpLossScale > trainingConfig.mixedPrecision.lossScaleMax)
						tt.mpLossScale = trainingConfig.mixedPrecision.lossScaleMax;
					tt.mpLossScaleGoodSteps = 0;
				}
			}
			if (mpEnable)
				MixedPrecisionHelper::ensure_transformer_lowp_weights(tt, trainingConfig);
		}
		seqInBatch = 0u;
		timeStepsInBatch = 0u;
	}
}

