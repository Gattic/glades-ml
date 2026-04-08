// Net-type-specific SGD: Transformer encoder/decoder-only path
#include "network.h"
#include "sgd_utils.h"
#include "transformer_common_utils.h"
#include "transformer_train_detail.h"
#include "transformer_kernels.h"
#include "glades_thread_pool.h"
#include "ddp_comm.h"

#ifdef GLADES_HAVE_CUDA
#include "cuda/gpu_dispatch.h"
#include "cuda/gpu_kernels.h"
#include "cuda/gpu_blas.h"
#include "cuda/gpu_atlas.h"
#include "cuda/gpu_transformer_state.h"
#endif

#include "Backend/Database/GLogger.h"

#include "../DataObjects/DataInput.h"
#include "../GMath/gmath.h"
#include "../rng.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#include "logfmt_utils.h"

using namespace glades;
using namespace glades::logfmt;
using namespace glades::transformer_train_detail;

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
		storeRunningFlag(false);
		return;
	}

	// Ensure transformer parameters exist.
	if (!ensureTensorParametersInitialized())
	{
		storeRunningFlag(false);
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
		storeRunningFlag(false);
		return;
	}
	if (tokenLM)
	{
		if (!tieEmb)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token LM mode requires tieEmbeddings=true");
			storeRunningFlag(false);
			return;
		}
		if (inputSize != 1u)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token LM mode requires inputSize==1 (token id)");
			storeRunningFlag(false);
			return;
		}
		if (outSize != vocabSize || vocabSize == 0u)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token LM mode requires outSize==vocabSize>0");
			storeRunningFlag(false);
			return;
		}
		if (tieEmb && (ttConst.tokE.size() != static_cast<size_t>(vocabSize) * static_cast<size_t>(dModel)))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_TRANSFORMER: token LM embedding table is not initialized");
			storeRunningFlag(false);
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
	const bool ddpEnabled = trainingConfig.ddp.enable && (glades::ddp::worldSize() > 1);

	// Trainability triage:
	// - For real LLMs, AdamW or ATLAS are the supported optimizers in this backend.
	// - Full softmax is guarded to avoid silently allocating/computing O(T*vocab) buffers.
	if (tokenLM && isTrain && (trainingConfig.optimizer.type != glades::OptimizerConfig::ADAMW)
	                       && (trainingConfig.optimizer.type != glades::OptimizerConfig::ATLAS))
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "SGDHelper_TRANSFORMER: token LM training requires optimizer=ADAMW or ATLAS for LLM-scale stability");
		storeRunningFlag(false);
		return;
	}
	if (tokenLM && (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX) && (tokenLmNegK < 1))
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "SGDHelper_TRANSFORMER: token LM sampled-softmax requires tokenLmSampledNegatives >= 1");
		storeRunningFlag(false);
		return;
	}

	// Reset per-epoch bookkeeping
	results.clear();
	if (isTrain)
		nbRecord.clear();

	// Minibatch: number of sequences to accumulate before applying an update.
	const unsigned int seqBatchMax = (minibatchSize > 0 ? static_cast<unsigned int>(minibatchSize) : 1u);
	const unsigned int optimizerStepsPerEpoch = (seqCount + seqBatchMax - 1u) / seqBatchMax;
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
			std::fill(tt.gLnFinalGamma.begin(), tt.gLnFinalGamma.end(), 0.0f);
			std::fill(tt.gLnFinalBeta.begin(), tt.gLnFinalBeta.end(), 0.0f);
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
			const bool useAtlas = (net.trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS);
			// Token LM mode uses a tied embedding head: logits = H * E^T + lmBias.
			// In this mode, the generic output projection (WOut/bOut) is UNUSED and must not:
			// - contribute to global grad-norm clipping (via weight decay terms), or
			// - be updated/decayed by the optimizer.
			//
			// Otherwise, the model will "silently" change unused parameters and can skew grad clipping
			// scale for the parameters that actually affect the forward pass.
			const bool tokenLMTiedHead = tt.tokenModel;

			// Warmup + DDP LR scaling multipliers.
			const float warmupMult = net.trainingConfig.warmup.multiplier(static_cast<int>(tt.optimizerStep));
			const float ddpLRScale = (net.trainingConfig.ddp.enable && net.trainingConfig.ddp.linearLRScaling)
			                       ? static_cast<float>(glades::ddp::worldSize()) : 1.0f;
			const float extraLRMult = warmupMult * ddpLRScale;

			// Optional global grad norm clip (same semantics as other tensor paths).
			float gradNorm = 0.0f;
			float gradScale = 1.0f;
			const float clipNorm = net.trainingConfig.globalGradClipNorm;
			if (clipNorm > 0.0f)
			{
				double sumsq = 0.0;

				if (useAdamW || useAtlas)
				{
					// For AdamW/ATLAS, clip is applied to raw gradients (weight decay is decoupled).
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
					// Final LayerNorm gradients
					for (size_t j = 0; j < tt.gLnFinalGamma.size(); ++j) { const double gd = static_cast<double>(tt.gLnFinalGamma[j] * invBatch); sumsq += gd * gd; }
					for (size_t j = 0; j < tt.gLnFinalBeta.size(); ++j) { const double gd = static_cast<double>(tt.gLnFinalBeta[j] * invBatch); sumsq += gd * gd; }
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

					// Final LayerNorm gradients
					for (size_t j = 0; j < tt.gLnFinalGamma.size(); ++j) sumsq += static_cast<double>(tt.gLnFinalGamma[j] * invBatch) * static_cast<double>(tt.gLnFinalGamma[j] * invBatch);
					for (size_t j = 0; j < tt.gLnFinalBeta.size(); ++j) sumsq += static_cast<double>(tt.gLnFinalBeta[j] * invBatch) * static_cast<double>(tt.gLnFinalBeta[j] * invBatch);

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
					net.storeRunningFlag(false);
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
				net.storeRunningFlag(false);
				return false;
			}

			// --- Apply updates ---
			if (!useAdamW && !useAtlas)
			{
				// SGD + momentum (historical behavior).
				// Token embedding (index 0 in LM mode)
				if (tt.tokenModel)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
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
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
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
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
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

				// Final LayerNorm (SGD, use block 0 LR; no weight decay)
				{
					const float lr = net.skeleton->getLearningRate(1u) * net.lrScheduleMultiplier * extraLRMult;
					for (size_t i = 0; i < tt.lnFinalGamma.size(); ++i) { tt.lnFinalGamma[i] -= lr * (tt.gLnFinalGamma[i] * invBatch) * gradScale; tt.gLnFinalGamma[i] = 0.0f; }
					for (size_t i = 0; i < tt.lnFinalBeta.size(); ++i) { tt.lnFinalBeta[i] -= lr * (tt.gLnFinalBeta[i] * invBatch) * gradScale; tt.gLnFinalBeta[i] = 0.0f; }
				}

				// Output projection (index nLayers) is unused in token LM tied-head mode.
				if (!tokenLMTiedHead)
				{
					const unsigned int idx = nLayers;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
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
			else if (useAtlas)
			{
				// ATLAS optimizer: subspace-projected natural gradient with temporal prediction.
				const glades::ATLASConfig& ac = net.trainingConfig.atlas;

				tt.optimizerStep += 1ULL;

				// Compute weight matrix dimensions.
				const unsigned int dmTT = tt.dModel;
				const unsigned int dFFTT = tt.dFF;
				const unsigned int nHeadsTT = tt.nHeads;
				const unsigned int nKVHeadsTT = (tt.nKVHeads > 0u ? tt.nKVHeads : nHeadsTT);
				const unsigned int dHeadTT = dmTT / nHeadsTT;
				const unsigned int dModelKVTT = nKVHeadsTT * dHeadTT;
				const unsigned int ffnKindTT = tt.ffnKind;
				const unsigned int ff1WidthTT = (ffnKindTT == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFFTT) : dFFTT;

				// Token embedding (index 0 in LM mode)
				if (tt.tokenModel)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);

					// tokE: [vocabSize, dModel]
					if (!atlas::update(tt.atlasTokE, &tt.tokE[0], &tt.gTokE[0],
					              tt.vocabSize, dmTT, invBatch, lr, wd1, wd2, gradScale,
					              ac, net.rngEngine, net.getLogger(), "tr.tokE"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: ATLAS tokE update entered NaN recovery");
						net.storeRunningFlag(false);
						return false;
					}

					// lmBias: simple SGD (no subspace projection for 1D bias)
					if (!atlas::updateBias(&tt.lmBias[0], &tt.gLmBias[0],
					                       static_cast<unsigned int>(tt.lmBias.size()),
					                       invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: ATLAS lmBias update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				// Input projection (index 0)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);

					// WIn: [dModel, inputSize]
					if (!atlas::update(tt.atlasWIn, &tt.WIn[0], &tt.gWIn[0],
					              dmTT, tt.inputSize, invBatch, lr, wd1, wd2, gradScale,
					              ac, net.rngEngine, net.getLogger(), "tr.WIn"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: ATLAS WIn update entered NaN recovery");
						net.storeRunningFlag(false);
						return false;
					}

					// bIn: simple SGD
					if (!atlas::updateBias(&tt.bIn[0], &tt.gBIn[0],
					                       static_cast<unsigned int>(tt.bIn.size()),
					                       invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: ATLAS bIn update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				// Blocks (index 1..nLayers)
				for (unsigned int li = 0; li < nLayers; ++li)
				{
					const unsigned int idx = li + 1u;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					TensorTransformerState::Block& b = tt.blocks[li];

					// Wq: [dModel, dModel]
					if (!atlas::update(b.atlasWq, &b.Wq[0], &b.gWq[0], dmTT, dmTT,
					              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.Wq")
					// Wk: [dModelKV, dModel]
					    || !atlas::update(b.atlasWk, &b.Wk[0], &b.gWk[0], dModelKVTT, dmTT,
					              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.Wk")
					// Wv: [dModelKV, dModel]
					    || !atlas::update(b.atlasWv, &b.Wv[0], &b.gWv[0], dModelKVTT, dmTT,
					              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.Wv")
					// Wo: [dModel, dModel]
					    || !atlas::update(b.atlasWo, &b.Wo[0], &b.gWo[0], dmTT, dmTT,
					              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.Wo")
					// W1: [ff1Width, dModel]
					    || !atlas::update(b.atlasW1, &b.W1[0], &b.gW1[0], ff1WidthTT, dmTT,
					              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.W1")
					// W2: [dModel, dFF]
					    || !atlas::update(b.atlasW2, &b.W2[0], &b.gW2[0], dmTT, dFFTT,
					              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.W2"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: ATLAS block weight update entered NaN recovery");
						net.storeRunningFlag(false);
						return false;
					}

					// Biases and LN params: simple SGD (no subspace projection)
					if (!atlas::updateBias(&b.bq[0], &b.gBq[0], static_cast<unsigned int>(b.bq.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.bk[0], &b.gBk[0], static_cast<unsigned int>(b.bk.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.bv[0], &b.gBv[0], static_cast<unsigned int>(b.bv.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.bo[0], &b.gBo[0], static_cast<unsigned int>(b.bo.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.b1[0], &b.gB1[0], static_cast<unsigned int>(b.b1.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.b2[0], &b.gB2[0], static_cast<unsigned int>(b.b2.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln1Gamma[0], &b.gLn1Gamma[0], static_cast<unsigned int>(b.ln1Gamma.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln1Beta[0], &b.gLn1Beta[0], static_cast<unsigned int>(b.ln1Beta.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln2Gamma[0], &b.gLn2Gamma[0], static_cast<unsigned int>(b.ln2Gamma.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln2Beta[0], &b.gLn2Beta[0], static_cast<unsigned int>(b.ln2Beta.size()), invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: ATLAS block bias/LN update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				// Final LayerNorm (SGD, use block 0 LR; no weight decay)
				{
					const float lr = net.skeleton->getLearningRate(1u) * net.lrScheduleMultiplier * extraLRMult;
					if (!atlas::updateBias(&tt.lnFinalGamma[0], &tt.gLnFinalGamma[0],
					                       static_cast<unsigned int>(tt.lnFinalGamma.size()),
					                       invBatch, lr, gradScale)
					    || !atlas::updateBias(&tt.lnFinalBeta[0], &tt.gLnFinalBeta[0],
					                          static_cast<unsigned int>(tt.lnFinalBeta.size()),
					                          invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: ATLAS final LN update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				// Output projection (index nLayers) is unused in token LM tied-head mode.
				if (!tokenLMTiedHead)
				{
					const unsigned int idx = nLayers;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);

					// WOut: [outSize, dModel]
					if (!atlas::update(tt.atlasWOut, &tt.WOut[0], &tt.gWOut[0],
					              outSize, dmTT, invBatch, lr, wd1, wd2, gradScale,
					              ac, net.rngEngine, net.getLogger(), "tr.WOut"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: ATLAS WOut update entered NaN recovery");
						net.storeRunningFlag(false);
						return false;
					}

					// bOut: simple SGD
					if (!atlas::updateBias(&tt.bOut[0], &tt.gBOut[0],
					                       static_cast<unsigned int>(tt.bOut.size()),
					                       invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: ATLAS bOut update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
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
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					Adam::update_weight(tt.tokE, tt.vTokE, tt.v2TokE, tt.gTokE,
					                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
					                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}

				// Input projection (index 0)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
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
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
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

				// Final LayerNorm (use block 0 LR; no weight decay)
				{
					const float lr = net.skeleton->getLearningRate(1u) * net.lrScheduleMultiplier * extraLRMult;
					Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}

				// Output projection (index nLayers) is unused in token LM tied-head mode.
				if (!tokenLMTiedHead)
				{
					const unsigned int idx = nLayers;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
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

	// Helper: AllReduce all gradient vectors across DDP workers (SUM).
	struct DDPReduceGrads
	{
		TensorTransformerState& tt;
		unsigned int nLayers;
		DDPReduceGrads(TensorTransformerState& t, unsigned int nl) : tt(t), nLayers(nl) {}
		void operator()(unsigned int& timeStepsInBatch) const
		{
			if (glades::ddp::worldSize() <= 1) return;

			const int maxBufs = 8 + 16 * static_cast<int>(nLayers);
			std::vector<float*> bufs;
			std::vector<size_t> sizes;
			bufs.reserve(maxBufs);
			sizes.reserve(maxBufs);

			// Global tensors
			if (!tt.gTokE.empty())        { bufs.push_back(&tt.gTokE[0]);        sizes.push_back(tt.gTokE.size()); }
			if (!tt.gLmBias.empty())       { bufs.push_back(&tt.gLmBias[0]);      sizes.push_back(tt.gLmBias.size()); }
			if (!tt.gWIn.empty())          { bufs.push_back(&tt.gWIn[0]);         sizes.push_back(tt.gWIn.size()); }
			if (!tt.gBIn.empty())          { bufs.push_back(&tt.gBIn[0]);         sizes.push_back(tt.gBIn.size()); }
			if (!tt.gWOut.empty())         { bufs.push_back(&tt.gWOut[0]);        sizes.push_back(tt.gWOut.size()); }
			if (!tt.gBOut.empty())         { bufs.push_back(&tt.gBOut[0]);        sizes.push_back(tt.gBOut.size()); }
			if (!tt.gLnFinalGamma.empty()) { bufs.push_back(&tt.gLnFinalGamma[0]); sizes.push_back(tt.gLnFinalGamma.size()); }
			if (!tt.gLnFinalBeta.empty())  { bufs.push_back(&tt.gLnFinalBeta[0]);  sizes.push_back(tt.gLnFinalBeta.size()); }

			// Per-block tensors
			for (unsigned int l = 0; l < nLayers; ++l)
			{
				TensorTransformerState::Block& b = tt.blocks[l];
				if (!b.gWq.empty())      { bufs.push_back(&b.gWq[0]);      sizes.push_back(b.gWq.size()); }
				if (!b.gWk.empty())      { bufs.push_back(&b.gWk[0]);      sizes.push_back(b.gWk.size()); }
				if (!b.gWv.empty())      { bufs.push_back(&b.gWv[0]);      sizes.push_back(b.gWv.size()); }
				if (!b.gWo.empty())      { bufs.push_back(&b.gWo[0]);      sizes.push_back(b.gWo.size()); }
				if (!b.gBq.empty())      { bufs.push_back(&b.gBq[0]);      sizes.push_back(b.gBq.size()); }
				if (!b.gBk.empty())      { bufs.push_back(&b.gBk[0]);      sizes.push_back(b.gBk.size()); }
				if (!b.gBv.empty())      { bufs.push_back(&b.gBv[0]);      sizes.push_back(b.gBv.size()); }
				if (!b.gBo.empty())      { bufs.push_back(&b.gBo[0]);      sizes.push_back(b.gBo.size()); }
				if (!b.gLn1Gamma.empty()){ bufs.push_back(&b.gLn1Gamma[0]);sizes.push_back(b.gLn1Gamma.size()); }
				if (!b.gLn1Beta.empty()) { bufs.push_back(&b.gLn1Beta[0]); sizes.push_back(b.gLn1Beta.size()); }
				if (!b.gLn2Gamma.empty()){ bufs.push_back(&b.gLn2Gamma[0]);sizes.push_back(b.gLn2Gamma.size()); }
				if (!b.gLn2Beta.empty()) { bufs.push_back(&b.gLn2Beta[0]); sizes.push_back(b.gLn2Beta.size()); }
				if (!b.gW1.empty())      { bufs.push_back(&b.gW1[0]);      sizes.push_back(b.gW1.size()); }
				if (!b.gW2.empty())      { bufs.push_back(&b.gW2[0]);      sizes.push_back(b.gW2.size()); }
				if (!b.gB1.empty())      { bufs.push_back(&b.gB1[0]);      sizes.push_back(b.gB1.size()); }
				if (!b.gB2.empty())      { bufs.push_back(&b.gB2[0]);      sizes.push_back(b.gB2.size()); }
			}

			int numBufs = static_cast<int>(bufs.size());
			glades::ddp::allReduceSumInPlaceBucketed(
				numBufs > 0 ? &bufs[0] : NULL,
				numBufs > 0 ? &sizes[0] : NULL,
				numBufs,
				&timeStepsInBatch, 1);
		}
	};
	DDPReduceGrads ddpReduceGrads(tt, nLayers);

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
	if (nHeads == 0u || (dModel % nHeads) != 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_TRANSFORMER: dModel is not divisible by nHeads");
		storeRunningFlag(false);
		return;
	}
	if ((nHeads % nKVHeads) != 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_TRANSFORMER: nHeads is not divisible by nKVHeads");
		storeRunningFlag(false);
		return;
	}
	const unsigned int dHead = dModel / nHeads;
	const unsigned int dModelKV = nKVHeads * dHead;
	const unsigned int ff1Width = (ffnKind == static_cast<int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFF) : dFF;

	// Bundle config for extracted sub-functions.
	TransformerEpochCfg epochCfg;
	epochCfg.inputSize = inputSize;
	epochCfg.outSize = outSize;
	epochCfg.dModel = dModel;
	epochCfg.dFF = dFF;
	epochCfg.nHeads = nHeads;
	epochCfg.nKVHeads = nKVHeads;
	epochCfg.nLayers = nLayers;
	epochCfg.vocabSize = vocabSize;
	epochCfg.dHead = dHead;
	epochCfg.dModelKV = dModelKV;
	epochCfg.ff1Width = ff1Width;
	epochCfg.padTokenId = padTokenId;
	epochCfg.causal = causal;
	epochCfg.tokenLM = tokenLM;
	epochCfg.tieEmb = tieEmb;
	epochCfg.isTrain = isTrain;
	epochCfg.gradClip = gradClip;
	epochCfg.lnEps = lnEps;
	epochCfg.ropeTheta = ropeTheta;
	epochCfg.costFx = costFx;
	epochCfg.posEnc = posEnc;
	epochCfg.normType = normType;
	epochCfg.ffnKind = ffnKind;
	epochCfg.ffnAct = ffnAct;
	epochCfg.ropeDimOverride = ropeDimOverride;
	epochCfg.tokenLmNegK = tokenLmNegK;
	epochCfg.ddpEnabled = ddpEnabled;
	epochCfg.useLowpWeights = useLowpWeights;
	epochCfg.lowpDType = lowpDType;
	epochCfg.mpEnable = mpEnable;
	epochCfg.mpUseLossScaling = mpUseLossScaling;
	epochCfg.mpDynamicLossScaling = mpDynamicLossScaling;
	epochCfg.seqBatchMax = seqBatchMax;
	epochCfg.tokenLmLossKind = tokenLmLossKind;
	epochCfg.tokenLmAllowHuge = tokenLmAllowHuge;
	epochCfg.lrScheduleMultiplier = lrScheduleMultiplier;

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

#ifdef GLADES_HAVE_CUDA
	// === GPU accelerated training path ===
	//
	// When GPU is enabled and available, offload the entire forward/backward/optimizer
	// loop to the GPU. Weights stay GPU-resident; only token inputs and loss/metrics
	// cross PCIe per sequence.
	//
	// If ensureGpuState() fails, fall through to the CPU path silently.
	if (trainingConfig.gpu.enable && isTrain)
	{
		const bool gpuReady = ensureGpuState();
		if (gpuReady && gpuTransformerWeights && gpuTransformerWeights->initialized)
		{
			transformerGpuTrainEpoch(epochCfg, seqCount, epochIdx, epochStartMs,
			                        tokensProcessed, targetsProcessed,
			                        tokenLmNllSum, tokenLmTokenCount,
			                        clsCorrect, clsTotal, logger);
			return; // GPU path complete; skip CPU fallback.
		}
	}
#endif // GLADES_HAVE_CUDA

	// Build shuffled sequence order. When DDP is active, each rank uses a different
	// seed so workers process sequences in different orders (reducing correlation).
	std::vector<unsigned int> seqOrder(seqCount);
	for (unsigned int si = 0; si < seqCount; ++si)
		seqOrder[si] = si;
	if (isTrain && seqCount > 1u)
	{
		unsigned int seed = static_cast<unsigned int>(epochIdx * 31 + 7);
		if (ddpEnabled)
			seed += static_cast<unsigned int>(glades::ddp::rank()) * 1000003u;
		// Fisher-Yates shuffle with a simple LCG.
		for (unsigned int i = seqCount - 1; i > 0; --i)
		{
			seed = seed * 1664525u + 1013904223u;
			const unsigned int j = seed % (i + 1u);
			const unsigned int tmp = seqOrder[i];
			seqOrder[i] = seqOrder[j];
			seqOrder[j] = tmp;
		}
	}

	for (unsigned int si = 0; si < seqCount; ++si)
	{
		if (!loadRunningFlag())
			break;

		const unsigned int s = seqOrder[si];
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
				storeRunningFlag(false);
				return;
			}
		}

		transformerScratch.ensure(T, inputSize, scratchOutSize, dModel, dFF, dModelKV, nHeads, nLayers, ff1Width,
		                         trainingConfig.transformer.embeddingDropoutRate,
		                         trainingConfig.transformer.residualDropoutRate,
		                         trainingConfig.gradientCheckpointing);

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
					storeRunningFlag(false);
					return;
				}
				if (tid < 0 || static_cast<unsigned int>(tid) >= vocabSize)
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token id out of range");
					storeRunningFlag(false);
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
					storeRunningFlag(false);
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
						storeRunningFlag(false);
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

		// === Forward + output head (delegated to extracted method) ===
		transformerCpuForwardPass(epochCfg, T, s, tokenIds, targetIds, keyAllowed, scratchOutSize, sampleCount);


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
				{
					const int expIdx = GMath::argmax(expRow, expSize);
					const int predIdx = GMath::argmax(&transformerScratch.probs[off], outSize);
					confusionMatrix.addResultDirect(static_cast<unsigned int>(expIdx), static_cast<unsigned int>(predIdx));
				}
			}
		}

		// === Backward + grad accumulation (delegated to extracted method) ===
		if (isTrain)
		{
			if (seqInBatch == 0u)
			{
				clearGrads();
				timeStepsInBatch = 0u;
			}

			transformerCpuBackwardPass(epochCfg, T, s, tokenIds, targetIds, scratchOutSize, seqInBatch, timeStepsInBatch);


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

						if (ddpEnabled)
							ddpReduceGrads(timeStepsInBatch);

						const unsigned int stepInEpoch = (s + 1u) / seqBatchMax;
						lrScheduleMultiplier = transformer_schedule_multiplier(
						    trainingConfig.lrSchedule,
						    epochIdx + lrScheduleEpochOffset,
						    stepInEpoch,
						    optimizerStepsPerEpoch);
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
				const bool tokenLmFullSoftmax = (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX);
				append_logfmt_kv(oss, "token_lm_loss_kind", std::string(tokenLmFullSoftmax ? "full_softmax" : "sampled_softmax"));
				const double meanNll = (tokenLmTokenCount > 0ULL) ? (tokenLmNllSum / static_cast<double>(tokenLmTokenCount)) : 0.0;
				append_logfmt_kv(oss, "nll", meanNll);
				if (tokenLmFullSoftmax)
				{
					double ppl = 0.0;
					if (tokenLmTokenCount > 0ULL)
					{
						double arg = meanNll;
						if (arg > 80.0) arg = 80.0;
						if (arg < -80.0) arg = -80.0;
						ppl = exp(arg);
					}
					append_logfmt_kv(oss, "perplexity", ppl);
					append_logfmt_kv(oss, "acc_top1", (clsTotal > 0ULL) ? (100.0 * static_cast<double>(clsCorrect) / static_cast<double>(clsTotal)) : 0.0);
				}
				else
				{
					// Sampled-softmax does not produce an exact perplexity, and we do not have a full-vocab argmax.
					append_logfmt_kv(oss, "perplexity", std::string("na"));
					append_logfmt_kv(oss, "acc_top1", std::string("na"));
				}
			}
			else
			{
				append_logfmt_kv(oss, "loss_so_far", overallTotalError);
			}
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
			const bool tokenLmFullSoftmax = (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX);
			append_logfmt_kv(oss, "token_lm_loss_kind", std::string(tokenLmFullSoftmax ? "full_softmax" : "sampled_softmax"));
			append_logfmt_kv(oss, "nll", meanNll);
			if (tokenLmFullSoftmax)
			{
				append_logfmt_kv(oss, "perplexity", ppl);
				append_logfmt_kv(oss, "acc_top1", (clsTotal > 0ULL) ? (100.0 * static_cast<double>(clsCorrect) / static_cast<double>(clsTotal)) : 0.0);
			}
			else
			{
				append_logfmt_kv(oss, "perplexity", std::string("na"));
				append_logfmt_kv(oss, "acc_top1", std::string("na"));
			}
		}
		else
		{
			append_logfmt_kv(oss, "loss_so_far", overallTotalError);
		}
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
		append_logfmt_kv(oss, "optimizer_step", static_cast<unsigned long long>(tt.optimizerStep));
		if (mpUseLossScaling)
			append_logfmt_kv(oss, "loss_scale", tt.mpLossScale);

		logger->info("NNetwork", shmea::GString(oss.str().c_str()));
	}

	// DDP: aggregate metrics across all workers before finalization.
	if (ddpEnabled)
	{
		glades::ddp::allReduceSumInPlace(&tokenLmNllSum, 1);
		glades::ddp::allReduceSumInPlace(&tokenLmTokenCount, 1);
		glades::ddp::allReduceSumInPlace(&clsCorrect, 1);
		glades::ddp::allReduceSumInPlace(&clsTotal, 1);
		glades::ddp::allReduceSumInPlace(&regSSE, 1);
		glades::ddp::allReduceSumInPlace(&regSAE, 1);
		glades::ddp::allReduceSumInPlace(&regSumY, 1);
		glades::ddp::allReduceSumInPlace(&regSumY2, 1);
		glades::ddp::allReduceSumInPlace(&regCount, 1);
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
			if (ddpEnabled)
				ddpReduceGrads(timeStepsInBatch);
			lrScheduleMultiplier = transformer_schedule_multiplier(
			    trainingConfig.lrSchedule,
			    epochIdx + lrScheduleEpochOffset,
			    optimizerStepsPerEpoch,
			    optimizerStepsPerEpoch);
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

// ---------------------------------------------------------------------------
// Extracted CPU forward pass (previously inlined in SGDHelper_TRANSFORMER).
// Computes forward activations through all transformer blocks and produces
// logits/probs in transformerScratch.  Called once per sequence.
// ---------------------------------------------------------------------------
void glades::NNetwork::transformerCpuForwardPass(const TransformerEpochCfg& cfg, unsigned int T, unsigned int s,
                                                 const std::vector<int>& tokenIds,
                                                 const std::vector<int>& targetIds,
                                                 const std::vector<unsigned char>& keyAllowed,
                                                 unsigned int scratchOutSize, unsigned int sampleCount)
{
	TensorTransformerState& tt = tensorTransformer;

	const unsigned int dModel = cfg.dModel;
	const unsigned int dFF = cfg.dFF;
	const unsigned int nHeads = cfg.nHeads;
	const unsigned int nKVHeads = cfg.nKVHeads;
	const unsigned int nLayers = cfg.nLayers;
	const unsigned int vocabSize = cfg.vocabSize;
	const unsigned int inputSize = cfg.inputSize;
	const unsigned int outSize = cfg.outSize;
	const unsigned int dHead = cfg.dHead;
	const unsigned int dModelKV = cfg.dModelKV;
	const unsigned int ff1Width = cfg.ff1Width;
	const bool tokenLM = cfg.tokenLM;
	const bool causal = cfg.causal;
	const int posEnc = cfg.posEnc;
	const int normType = cfg.normType;
	const int ffnKind = cfg.ffnKind;
	const int ffnAct = cfg.ffnAct;
	const float lnEps = cfg.lnEps;
	const float ropeTheta = cfg.ropeTheta;
	const int ropeDimOverride = cfg.ropeDimOverride;
	const bool useLowpWeights = cfg.useLowpWeights;
	const int lowpDType = cfg.lowpDType;
	const int costFx = cfg.costFx;
	const glades::TransformerRunConfig::TokenLMLossKind tokenLmLossKind = cfg.tokenLmLossKind;
	const int tokenLmNegK = cfg.tokenLmNegK;
	const int padTokenId = cfg.padTokenId;

	// === Forward ===
	// Input to h
	if (tokenLM)
	{
		// Embedding lookup.
		const bool haveLowpE = useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty();
		for (unsigned int t = 0; t < T; ++t)
		{
			const int tid = tokenIds[t];
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
		linear_forward_maybe_lowp(transformerScratch.x.data(), T, inputSize, tt.WIn, tt.WInLowp, useLowpWeights, lowpDType, tt.bIn, dModel,
		                          transformerScratch.h.data());
	}
	if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
	{
		transformerPosEncCache.ensureSinusoidal(dModel);
		add_positional_encoding(transformerScratch.h.empty() ? NULL : &transformerScratch.h[0], T, dModel,
		                        transformerPosEncCache.sinInvDenomPair);
	}

	// Embedding dropout
	{
		const float embDropRate = trainingConfig.transformer.embeddingDropoutRate;
		if (embDropRate > 0.0f && !transformerScratch.h.empty())
		{
			const size_t hLen = static_cast<size_t>(T) * static_cast<size_t>(dModel);
			unsigned char* mask = transformerScratch.dropoutMaskEmb.empty() ? NULL : &transformerScratch.dropoutMaskEmb[0];
			if (mask)
			{
				glades::transformer_kernels::generate_dropout_mask(rngEngine, mask, hLen, embDropRate);
				const float scale = 1.0f / (1.0f - embDropRate);
				glades::transformer_kernels::apply_dropout_mask_inplace(&transformerScratch.h[0], mask, scale, hLen);
			}
		}
	}

	// RoPE precompute
	const bool useRope = (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE));
	unsigned int ropeDim = dHead;
	if (ropeDimOverride > 0)
	{
		const unsigned int rd = static_cast<unsigned int>(ropeDimOverride);
		ropeDim = (rd < ropeDim) ? rd : ropeDim;
	}
	if ((ropeDim % 2u) != 0u)
		ropeDim -= 1u;
	const std::vector<double>* ropeInvFreq = NULL;
	if (useRope && ropeDim >= 2u)
	{
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
		{
			const bool normWorthParallel = (static_cast<unsigned long long>(T) * dModel >= 65536ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (normWorthParallel && T > 1u && pool.numThreads() > 1u)
			{
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
					std::fill(ln1Mean, ln1Mean + T, 0.0f);
				NormFwdCtx nctx;
				nctx.X = hIn;
				nctx.D = dModel;
				nctx.gamma = b.ln1Gamma.empty() ? NULL : &b.ln1Gamma[0];
				nctx.beta = b.ln1Beta.empty() ? NULL : &b.ln1Beta[0];
				nctx.gammaSize = static_cast<unsigned int>(b.ln1Gamma.size());
				nctx.betaSize = static_cast<unsigned int>(b.ln1Beta.size());
				nctx.eps = lnEps;
				nctx.Y = x1;
				nctx.meanOut = (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM)) ? NULL : ln1Mean;
				nctx.invStdOut = ln1InvStd;
				nctx.isRmsNorm = (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM));
				pool.parallel_for(T, norm_fwd_body, &nctx);
			}
			else if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				std::fill(ln1Mean, ln1Mean + T, 0.0f);
				glades::transformer_kernels::rmsnorm_forward_rows(hIn, T, dModel, b.ln1Gamma, b.ln1Beta, lnEps, x1, ln1InvStd);
			}
			else
			{
				glades::transformer_kernels::layernorm_forward_rows(hIn, T, dModel, b.ln1Gamma, b.ln1Beta, lnEps, x1, ln1Mean, ln1InvStd);
			}
		}

		float* Q = transformerScratch.Q.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		float* K = transformerScratch.K.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
		float* V = transformerScratch.V.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));

		linear_forward_maybe_lowp(x1, T, dModel, b.Wq, b.WqLowp, useLowpWeights, lowpDType, b.bq, dModel, Q);
		linear_forward_maybe_lowp(x1, T, dModel, b.Wk, b.WkLowp, useLowpWeights, lowpDType, b.bk, dModelKV, K);
		linear_forward_maybe_lowp(x1, T, dModel, b.Wv, b.WvLowp, useLowpWeights, lowpDType, b.bv, dModelKV, V);

		// RoPE
		if (useRope && ropeInvFreq)
		{
			const bool ropeWorthParallel = (static_cast<unsigned long long>(T) * ropeDim >= 4096ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (ropeWorthParallel && pool.numThreads() > 1u)
			{
				if (nHeads > 1u)
				{
					RopeFwdCtx rctx;
					rctx.buf = Q; rctx.T = T; rctx.rowStride = dModel; rctx.dHead = dHead;
					rctx.ropeDim = ropeDim; rctx.invFreq = ropeInvFreq; rctx.inverse = false;
					pool.parallel_for(nHeads, rope_body, &rctx);
				}
				else
				{
					glades::transformer_kernels::rope_apply_inplace_strided(Q, T, dModel, dHead, ropeDim, *ropeInvFreq, false);
				}
				if (nKVHeads > 1u)
				{
					RopeFwdCtx rctx;
					rctx.buf = K; rctx.T = T; rctx.rowStride = dModelKV; rctx.dHead = dHead;
					rctx.ropeDim = ropeDim; rctx.invFreq = ropeInvFreq; rctx.inverse = false;
					pool.parallel_for(nKVHeads, rope_body, &rctx);
				}
				else
				{
					for (unsigned int hk = 0; hk < nKVHeads; ++hk)
						glades::transformer_kernels::rope_apply_inplace_strided(
						    K + static_cast<size_t>(hk) * static_cast<size_t>(dHead), T, dModelKV, dHead, ropeDim, *ropeInvFreq, false);
				}
			}
			else
			{
				for (unsigned int h = 0; h < nHeads; ++h)
					glades::transformer_kernels::rope_apply_inplace_strided(
					    Q + static_cast<size_t>(h) * static_cast<size_t>(dHead), T, dModel, dHead, ropeDim, *ropeInvFreq, false);
				for (unsigned int hk = 0; hk < nKVHeads; ++hk)
					glades::transformer_kernels::rope_apply_inplace_strided(
					    K + static_cast<size_t>(hk) * static_cast<size_t>(dHead), T, dModelKV, dHead, ropeDim, *ropeInvFreq, false);
			}
		}

		// Multi-head attention forward
		float* attnConcat = transformerScratch.attnConcat.data() +
		                    (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		std::fill(attnConcat, attnConcat + (static_cast<size_t>(T) * static_cast<size_t>(dModel)), 0.0f);
		{
			const bool attnWorthParallel = (static_cast<unsigned long long>(T) * T * dHead >= 32768ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (attnWorthParallel && nHeads > 1u && pool.numThreads() > 1u)
			{
				AttnFwdCtx actx;
				actx.Q = Q; actx.K = K; actx.V = V; actx.O = attnConcat;
				actx.dModel = dModel; actx.dModelKV = dModelKV; actx.dHead = dHead;
				actx.nHeads = nHeads; actx.nKVHeads = nKVHeads; actx.T = T;
				actx.groupSize = groupSize; actx.causal = causal;
				actx.keyAllowed = keyAllowed.empty() ? NULL : &keyAllowed[0];
				pool.parallel_for(nHeads, attn_fwd_body, &actx);
			}
			else
			{
				for (unsigned int h = 0; h < nHeads; ++h)
				{
					const unsigned int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0u ? (h / groupSize) : 0u);
					glades::transformer_ops::scaled_dot_product_attention_forward_flash_strided(
					    Q + static_cast<size_t>(h) * static_cast<size_t>(dHead), dModel,
					    K + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    V + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    T, dHead, dHead, causal,
					    attnConcat + static_cast<size_t>(h) * static_cast<size_t>(dHead), dModel,
					    keyAllowed.empty() ? NULL : &keyAllowed[0]);
				}
			}
		}

		// Wo projection
		float* attnOut = transformerScratch.attnOut.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		linear_forward_maybe_lowp(attnConcat, T, dModel, b.Wo, b.WoLowp, useLowpWeights, lowpDType, b.bo, dModel, attnOut);

		// Residual attention dropout
		{
			const float resDropRate = trainingConfig.transformer.residualDropoutRate;
			if (resDropRate > 0.0f)
			{
				const size_t layerOff = static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const size_t n = static_cast<size_t>(T) * static_cast<size_t>(dModel);
				unsigned char* mask = transformerScratch.dropoutMaskResAttn.empty() ? NULL : &transformerScratch.dropoutMaskResAttn[layerOff];
				if (mask)
				{
					glades::transformer_kernels::generate_dropout_mask(rngEngine, mask, n, resDropRate);
					const float scale = 1.0f / (1.0f - resDropRate);
					glades::transformer_kernels::apply_dropout_mask_inplace(attnOut, mask, scale, n);
				}
			}
		}

		// Residual add
		float* hAfterAttn = transformerScratch.hAfterAttn.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dModel); ++i)
			hAfterAttn[i] = hIn[i] + attnOut[i];

		// LN2 forward
		float* x2 = transformerScratch.x2.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		float* ln2Mean = transformerScratch.ln2Mean.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
		float* ln2InvStd = transformerScratch.ln2InvStd.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
		{
			const bool normWorthParallel = (static_cast<unsigned long long>(T) * dModel >= 65536ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (normWorthParallel && T > 1u && pool.numThreads() > 1u)
			{
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
					std::fill(ln2Mean, ln2Mean + T, 0.0f);
				NormFwdCtx nctx;
				nctx.X = hAfterAttn; nctx.D = dModel;
				nctx.gamma = b.ln2Gamma.empty() ? NULL : &b.ln2Gamma[0];
				nctx.beta = b.ln2Beta.empty() ? NULL : &b.ln2Beta[0];
				nctx.gammaSize = static_cast<unsigned int>(b.ln2Gamma.size());
				nctx.betaSize = static_cast<unsigned int>(b.ln2Beta.size());
				nctx.eps = lnEps; nctx.Y = x2;
				nctx.meanOut = (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM)) ? NULL : ln2Mean;
				nctx.invStdOut = ln2InvStd;
				nctx.isRmsNorm = (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM));
				pool.parallel_for(T, norm_fwd_body, &nctx);
			}
			else if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				std::fill(ln2Mean, ln2Mean + T, 0.0f);
				glades::transformer_kernels::rmsnorm_forward_rows(hAfterAttn, T, dModel, b.ln2Gamma, b.ln2Beta, lnEps, x2, ln2InvStd);
			}
			else
			{
				glades::transformer_kernels::layernorm_forward_rows(hAfterAttn, T, dModel, b.ln2Gamma, b.ln2Beta, lnEps, x2, ln2Mean, ln2InvStd);
			}
		}

		// FFN
		float* ff1 = transformerScratch.ff1.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
		float* ff1Act = transformerScratch.ff1Act.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dFF));
		linear_forward_maybe_lowp(x2, T, dModel, b.W1, b.W1Lowp, useLowpWeights, lowpDType, b.b1, ff1Width, ff1);
		if (ffnKind == static_cast<int>(glades::TransformerRunConfig::FFN_SWIGLU))
		{
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
			const size_t actLen = static_cast<size_t>(T) * static_cast<size_t>(dFF);
			if (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
				glades::transformer_kernels::gelu_forward_buf(ff1, ff1Act, actLen);
			else
				for (size_t i = 0; i < actLen; ++i)
					ff1Act[i] = glades::transformer_ops::relu(ff1[i]);
		}

		float* ffOut = transformerScratch.ffOut.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		linear_forward_maybe_lowp(ff1Act, T, dFF, b.W2, b.W2Lowp, useLowpWeights, lowpDType, b.b2, dModel, ffOut);

		// Residual FFN dropout
		{
			const float resDropRate = trainingConfig.transformer.residualDropoutRate;
			if (resDropRate > 0.0f)
			{
				const size_t layerOff = static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const size_t n = static_cast<size_t>(T) * static_cast<size_t>(dModel);
				unsigned char* mask = transformerScratch.dropoutMaskResFF.empty() ? NULL : &transformerScratch.dropoutMaskResFF[layerOff];
				if (mask)
				{
					glades::transformer_kernels::generate_dropout_mask(rngEngine, mask, n, resDropRate);
					const float scale = 1.0f / (1.0f - resDropRate);
					glades::transformer_kernels::apply_dropout_mask_inplace(ffOut, mask, scale, n);
				}
			}
		}

		// Residual add
		float* hAfterFF = transformerScratch.hAfterFF.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dModel); ++i)
			hAfterFF[i] = hAfterAttn[i] + ffOut[i];

		// Per-layer NaN detection: check hidden state after each layer and abort early.
		{
			const size_t layerElems = static_cast<size_t>(T) * static_cast<size_t>(dModel);
			if (!glades::transformer_kernels::all_finite_full(hAfterFF, layerElems))
			{
				std::ostringstream oss;
				oss << "transformerCpuForwardPass: non-finite hidden state detected at layer " << li;
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, oss.str());
				storeRunningFlag(false);
				return;
			}
		}
	}

	const float* hFinal = transformerScratch.hAfterFF.data() + (static_cast<size_t>(nLayers - 1u) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

	// Final LayerNorm
	float* hPostFinalLN = transformerScratch.hPostFinalLN.data();
	if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
	{
		glades::transformer_kernels::rmsnorm_forward_rows(hFinal, T, dModel,
		    tt.lnFinalGamma, tt.lnFinalBeta, lnEps,
		    hPostFinalLN, transformerScratch.lnFinalInvStd.data());
	}
	else
	{
		glades::transformer_kernels::layernorm_forward_rows(hFinal, T, dModel,
		    tt.lnFinalGamma, tt.lnFinalBeta, lnEps,
		    hPostFinalLN, transformerScratch.lnFinalMean.data(), transformerScratch.lnFinalInvStd.data());
	}

	// Output head logits
	if (tokenLM)
	{
		if (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX)
		{
			if (useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty())
			{
				glades::transformer_kernels::tied_embedding_logits_forward_rows_lowp(hPostFinalLN, T, dModel, &tt.tokELowp[0], lowpDType, tt.lmBias,
				                                                                    vocabSize, transformerScratch.logits.data());
			}
			else
			{
				const bool embWorthParallel = (static_cast<unsigned long long>(T) * vocabSize * dModel >= 500000ULL);
				glades::ThreadPool& pool = glades::ThreadPool::instance();
				if (embWorthParallel && T > 1u && pool.numThreads() > 1u)
				{
					TiedEmbLogitsCtx ectx;
					ectx.H = hPostFinalLN; ectx.dModel = dModel;
					ectx.tokE = tt.tokE.empty() ? NULL : &tt.tokE[0];
					ectx.lmBias = tt.lmBias.empty() ? NULL : &tt.lmBias[0];
					ectx.lmBiasSize = static_cast<unsigned int>(tt.lmBias.size());
					ectx.vocab = vocabSize;
					ectx.logitsOut = transformerScratch.logits.data();
					pool.parallel_for(T, tied_emb_logits_body, &ectx);
				}
				else
				{
					glades::transformer_kernels::tied_embedding_logits_forward_rows(hPostFinalLN, T, dModel, tt.tokE, tt.lmBias, vocabSize,
					                                                               transformerScratch.logits.data());
				}
			}
		}
		else
		{
			// Sampled-softmax logits
			const unsigned int K = static_cast<unsigned int>(tokenLmNegK);
			const unsigned int S = sampleCount;
			for (unsigned int t = 0; t < T; ++t)
			{
				const int yid = targetIds[t];
				if ((padTokenId >= 0 && yid == padTokenId) || yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
				{
					const size_t off = static_cast<size_t>(t) * static_cast<size_t>(S);
					for (unsigned int j = 0u; j < S; ++j)
					{
						transformerScratch.logits[off + j] = 0.0f;
						transformerScratch.tokenLmSampleIds[off + j] = -1;
					}
					continue;
				}

				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(S);
				transformerScratch.tokenLmSampleIds[off + 0u] = yid;
				for (unsigned int k = 0; k < K; ++k)
				{
					int neg = glades::rng::uniform_int(rngEngine, 0, static_cast<int>(vocabSize) - 2);
					if (neg >= yid) ++neg;
					transformerScratch.tokenLmSampleIds[off + 1u + k] = neg;
				}

				const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
				const bool haveLowpE = useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty();
				for (unsigned int j = 0u; j < S; ++j)
				{
					const int vid = transformerScratch.tokenLmSampleIds[off + j];
					if (vid < 0 || static_cast<unsigned int>(vid) >= vocabSize)
					{
						transformerScratch.logits[off + j] = 0.0f;
						continue;
					}
					float dot = 0.0f;
					const size_t eOff = static_cast<size_t>(vid) * static_cast<size_t>(dModel);
					if (haveLowpE)
					{
						for (unsigned int d = 0; d < dModel; ++d)
							dot += hPostFinalLN[hOff + d] * glades::transformer_kernels::lowp_to_float(tt.tokELowp[eOff + d], lowpDType);
					}
					else
					{
						dot = glades::transformer_kernels::dot_f32(&hPostFinalLN[hOff], &tt.tokE[eOff], dModel);
					}
					const float bias = (static_cast<size_t>(vid) < tt.lmBias.size()) ? tt.lmBias[static_cast<size_t>(vid)] : 0.0f;
					transformerScratch.logits[off + j] = dot + bias;
				}
			}
		}
	}
	else
	{
		linear_forward_maybe_lowp(hPostFinalLN, T, dModel, tt.WOut, tt.WOutLowp, useLowpWeights, lowpDType, tt.bOut, outSize,
		                          transformerScratch.logits.data());
	}

	// Softmax / sigmoid / identity
	if (tokenLM)
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const size_t off = static_cast<size_t>(t) * static_cast<size_t>(scratchOutSize);
			glades::transformer_kernels::softmax_stable_into(&transformerScratch.logits[off], static_cast<size_t>(scratchOutSize),
			                                                &transformerScratch.probs[off]);
		}
	}
	else if (((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1u))
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const size_t off = static_cast<size_t>(t) * static_cast<size_t>(outSize);
			glades::transformer_kernels::softmax_stable_into(&transformerScratch.logits[off], static_cast<size_t>(outSize),
			                                                &transformerScratch.probs[off]);
		}
	}
	else if ((costFx == GMath::CLASSIFICATION) && (outSize == 1u))
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const float z = transformerScratch.logits[static_cast<size_t>(t) * static_cast<size_t>(outSize)];
			transformerScratch.probs[static_cast<size_t>(t) * static_cast<size_t>(outSize)] = GMath::squash(z, GMath::SIGMOID, 0.0f);
		}
	}
	else
	{
		std::copy(transformerScratch.logits.begin(), transformerScratch.logits.end(), transformerScratch.probs.begin());
	}
}

// ---------------------------------------------------------------------------
// Extracted CPU backward pass + gradient accumulation.
// Assumes the forward pass has already been run.
// ---------------------------------------------------------------------------
void glades::NNetwork::transformerCpuBackwardPass(const TransformerEpochCfg& cfg, unsigned int T, unsigned int s,
                                                  const std::vector<int>& tokenIds,
                                                  const std::vector<int>& targetIds,
                                                  unsigned int scratchOutSize,
                                                  unsigned int& seqInBatch,
                                                  unsigned int& timeStepsInBatch)
{
	using namespace glades::sgd_detail;

	TensorTransformerState& tt = tensorTransformer;

	const unsigned int dModel = cfg.dModel;
	const unsigned int dFF = cfg.dFF;
	const unsigned int nHeads = cfg.nHeads;
	const unsigned int nKVHeads = cfg.nKVHeads;
	const unsigned int nLayers = cfg.nLayers;
	const unsigned int vocabSize = cfg.vocabSize;
	const unsigned int inputSize = cfg.inputSize;
	const unsigned int outSize = cfg.outSize;
	const unsigned int dHead = cfg.dHead;
	const unsigned int dModelKV = cfg.dModelKV;
	const unsigned int ff1Width = cfg.ff1Width;
	const bool tokenLM = cfg.tokenLM;
	const bool causal = cfg.causal;
	const int posEnc = cfg.posEnc;
	const int normType = cfg.normType;
	const int ffnKind = cfg.ffnKind;
	const int ffnAct = cfg.ffnAct;
	const float lnEps = cfg.lnEps;
	const float ropeTheta = cfg.ropeTheta;
	const int ropeDimOverride = cfg.ropeDimOverride;
	const bool useLowpWeights = cfg.useLowpWeights;
	const int lowpDType = cfg.lowpDType;
	const int costFx = cfg.costFx;
	const float gradClip = cfg.gradClip;
	const glades::TransformerRunConfig::TokenLMLossKind tokenLmLossKind = cfg.tokenLmLossKind;
	const int padTokenId = cfg.padTokenId;
	const bool mpUseLossScaling = cfg.mpUseLossScaling;

	const float lossScale = (mpUseLossScaling ? tt.mpLossScale : 1.0f);

	std::vector<float, glades::AlignedAllocator<float, 64> >& dLogits = transformerScratch.dLogits;
	if (dLogits.size() != (static_cast<size_t>(T) * static_cast<size_t>(scratchOutSize)))
		dLogits.resize(static_cast<size_t>(T) * static_cast<size_t>(scratchOutSize));
	std::fill(dLogits.begin(), dLogits.end(), 0.0f);

	std::vector<float, glades::AlignedAllocator<float, 64> >& dH = transformerScratch.dH;
	if (dH.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
		dH.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));

	const float* hFinal = transformerScratch.hAfterFF.data() + (static_cast<size_t>(nLayers - 1u) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
	float* hPostFinalLN = transformerScratch.hPostFinalLN.data();

	if (tokenLM)
	{
		unsigned int validTargetsThisSeq = 0u;
		std::fill(dH.begin(), dH.end(), 0.0f);
		if (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX)
		{
			for (unsigned int t = 0; t < T; ++t)
			{
				const int yid = targetIds[t];
				if (padTokenId >= 0 && yid == padTokenId) continue;
				if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize) continue;
				++validTargetsThisSeq;
				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(vocabSize);
				for (unsigned int v = 0; v < vocabSize; ++v)
					dLogits[off + v] = transformerScratch.probs[off + v];
				dLogits[off + static_cast<unsigned int>(yid)] -= 1.0f;
			}

			// Apply gradient clipping and loss scaling to dLogits in-place
			{
				const size_t dLogitsLen = static_cast<size_t>(T) * static_cast<size_t>(vocabSize);
				for (size_t idx = 0; idx < dLogitsLen; ++idx)
					dLogits[idx] = clip_maybe(dLogits[idx], gradClip) * lossScale;
			}

			// Parallelized tied-embedding backward:
			//   gTokE  += dLogits^T * hPostFinalLN   (weight gradient)
			//   gLmBias += sum_t dLogits[t,:]         (bias gradient)
			//   dH      += dLogits * tokE             (input gradient)
			linear_backward_accum_maybe_lowp(hPostFinalLN, dLogits.data(), T, dModel, vocabSize,
			    tt.gTokE, tt.gLmBias, tt.tokE, tt.tokELowp, useLowpWeights, lowpDType, dH.data());
		}
		else
		{
			const unsigned int S = scratchOutSize;
			for (unsigned int t = 0; t < T; ++t)
			{
				const int yid = targetIds[t];
				if (padTokenId >= 0 && yid == padTokenId) continue;
				if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize) continue;
				++validTargetsThisSeq;

				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(S);
				for (unsigned int j = 0u; j < S; ++j)
					dLogits[off + j] = transformerScratch.probs[off + j];
				dLogits[off + 0u] -= 1.0f;

				const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
				const bool haveLowpE = useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty();
				for (unsigned int j = 0u; j < S; ++j)
				{
					const float dz = clip_maybe(dLogits[off + j], gradClip) * lossScale;
					if (dz == 0.0f) continue;
					const int vid = transformerScratch.tokenLmSampleIds[off + j];
					if (vid < 0 || static_cast<unsigned int>(vid) >= vocabSize) continue;
					tt.gLmBias[static_cast<size_t>(vid)] += dz;
					const size_t eOff = static_cast<size_t>(vid) * static_cast<size_t>(dModel);
					for (unsigned int i = 0; i < dModel; ++i)
					{
						tt.gTokE[eOff + i] += dz * hPostFinalLN[hOff + i];
						const float ev = haveLowpE ? glades::transformer_kernels::lowp_to_float(tt.tokELowp[eOff + i], lowpDType) : tt.tokE[eOff + i];
						dH[hOff + i] += dz * ev;
					}
				}
			}
		}
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
					d = pred - expv;
				else if ((costFx == GMath::CLASSIFICATION) && (outSize == 1u))
					d = pred - expv;
				else
					d = GMath::costErrDer(expv, pred, costFx);
				dLogits[off + k] = clip_maybe(d, gradClip) * lossScale;
			}
		}

		linear_backward_accum_maybe_lowp(hPostFinalLN, dLogits.data(), T, dModel, outSize, tt.gWOut, tt.gBOut, tt.WOut, tt.WOutLowp, useLowpWeights,
		                                 lowpDType, dH.data());
		timeStepsInBatch += T;
	}

	// Backprop Final LayerNorm
	{
		std::vector<float, glades::AlignedAllocator<float, 64> >& dHPreFinalLN = transformerScratch.dH2;
		if (dHPreFinalLN.size() != dH.size()) dHPreFinalLN.resize(dH.size());
		std::fill(dHPreFinalLN.begin(), dHPreFinalLN.end(), 0.0f);
		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
		{
			glades::transformer_kernels::rmsnorm_backward_rows_accum(hFinal, dH.data(), T, dModel, tt.lnFinalGamma,
			    transformerScratch.lnFinalInvStd.data(), dHPreFinalLN.data(), tt.gLnFinalGamma, tt.gLnFinalBeta);
		}
		else
		{
			glades::transformer_kernels::layernorm_backward_rows_accum(hFinal, dH.data(), T, dModel, tt.lnFinalGamma,
			    transformerScratch.lnFinalMean.data(), transformerScratch.lnFinalInvStd.data(),
			    dHPreFinalLN.data(), tt.gLnFinalGamma, tt.gLnFinalBeta);
		}
		std::copy(dHPreFinalLN.begin(), dHPreFinalLN.end(), dH.begin());
	}

	// RoPE precompute for backward
	const bool useRope = (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE));
	unsigned int ropeDim = dHead;
	if (ropeDimOverride > 0)
	{
		const unsigned int rd = static_cast<unsigned int>(ropeDimOverride);
		ropeDim = (rd < ropeDim) ? rd : ropeDim;
	}
	if ((ropeDim % 2u) != 0u) ropeDim -= 1u;
	const std::vector<double>* ropeInvFreq = NULL;
	if (useRope && ropeDim >= 2u)
	{
		transformerPosEncCache.ensureRope(ropeDim, ropeTheta);
		ropeInvFreq = &transformerPosEncCache.ropeInvFreq;
	}
	const unsigned int groupSize = (nKVHeads > 0u) ? (nHeads / nKVHeads) : 0u;

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

		std::vector<float, glades::AlignedAllocator<float, 64> >& dHAfterAttn = transformerScratch.dH2;
		if (dHAfterAttn.size() != dH.size()) dHAfterAttn.resize(dH.size());
		std::copy(dH.begin(), dH.end(), dHAfterAttn.begin());

		// FFN residual dropout backward
		{
			const float resDropRate = trainingConfig.transformer.residualDropoutRate;
			if (resDropRate > 0.0f)
			{
				const size_t layerOff = static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const size_t n = static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const unsigned char* mask = transformerScratch.dropoutMaskResFF.empty() ? NULL : &transformerScratch.dropoutMaskResFF[layerOff];
				if (mask)
				{
					const float scale = 1.0f / (1.0f - resDropRate);
					for (size_t i = 0; i < n; ++i)
						dH[i] *= mask[i] ? scale : 0.0f;
				}
			}
		}

		// FFN backward
		std::vector<float, glades::AlignedAllocator<float, 64> >& dFF1Act = transformerScratch.dFF1Act;
		if (dFF1Act.size() != (static_cast<size_t>(T) * static_cast<size_t>(dFF)))
			dFF1Act.resize(static_cast<size_t>(T) * static_cast<size_t>(dFF));
		linear_backward_accum_maybe_lowp(ff1Act, dH.data(), T, dFF, dModel, b.gW2, b.gB2, b.W2, b.W2Lowp, useLowpWeights, lowpDType, dFF1Act.data());

		std::vector<float, glades::AlignedAllocator<float, 64> >& dX2 = transformerScratch.dX2;
		if (dX2.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
			dX2.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
		if (ffnKind == static_cast<int>(glades::TransformerRunConfig::FFN_SWIGLU))
		{
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
					dFF1Cat[preOff + i] = dOut * upPre * glades::transformer_ops::silu_deriv(gatePre);
					dFF1Cat[preOff + static_cast<size_t>(dFF) + i] = dOut * siluVal;
				}
			}
			linear_backward_accum_maybe_lowp(x2, dFF1Cat.data(), T, dModel, ff1Width, b.gW1, b.gB1, b.W1, b.W1Lowp, useLowpWeights, lowpDType, dX2.data());
		}
		else
		{
			const size_t actLen = dFF1Act.size();
			if (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
				glades::transformer_kernels::gelu_backward_buf(ff1, dFF1Act.data(), actLen);
			else
				for (size_t i = 0; i < actLen; ++i)
					dFF1Act[i] *= glades::transformer_ops::relu_deriv_from_y(ff1Act[i]);
			linear_backward_accum_maybe_lowp(x2, dFF1Act.data(), T, dModel, dFF, b.gW1, b.gB1, b.W1, b.W1Lowp, useLowpWeights, lowpDType, dX2.data());
		}

		// LN2 backward
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
		for (size_t i = 0; i < dHAfterAttn.size(); ++i)
			dHAfterAttn[i] += dHAfterAttnFromLN[i];

		std::copy(dHAfterAttn.begin(), dHAfterAttn.end(), dH.begin());

		// Attention residual dropout backward
		{
			const float resDropRate = trainingConfig.transformer.residualDropoutRate;
			if (resDropRate > 0.0f)
			{
				const size_t layerOff = static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const size_t n = static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const unsigned char* mask = transformerScratch.dropoutMaskResAttn.empty() ? NULL : &transformerScratch.dropoutMaskResAttn[layerOff];
				if (mask)
				{
					const float scale = 1.0f / (1.0f - resDropRate);
					for (size_t i = 0; i < n; ++i)
						dHAfterAttn[i] *= mask[i] ? scale : 0.0f;
				}
			}
		}

		const float* attnConcat = transformerScratch.attnConcat.data() +
		                          (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

		// Backprop Wo
		std::vector<float, glades::AlignedAllocator<float, 64> >& dAttnConcat = transformerScratch.dAttnConcat;
		if (dAttnConcat.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
			dAttnConcat.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
		linear_backward_accum_maybe_lowp(attnConcat, dHAfterAttn.data(), T, dModel, dModel, b.gWo, b.gBo, b.Wo, b.WoLowp, useLowpWeights,
		                                 lowpDType, dAttnConcat.data());

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

		// Attention backward
		{
			const bool attnWorthParallel = (static_cast<unsigned long long>(T) * T * dHead >= 32768ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (attnWorthParallel && nHeads > 1u && pool.numThreads() > 1u)
			{
				const unsigned int nThreads = pool.numThreads();
				unsigned int nChunksPerHead = 1u;
				if (nHeads < nThreads && T >= 512u)
				{
					nChunksPerHead = (nThreads + nHeads - 1u) / nHeads;
					if (nChunksPerHead > 4u) nChunksPerHead = 4u;
				}

				if (nChunksPerHead <= 1u)
				{
					AttnBwdCtx actx;
					actx.Q = Qfull; actx.K = Kfull; actx.V = Vfull;
					actx.dO = dAttnConcat.data();
					actx.dQ = dQfull.data(); actx.dK = dKfull.data(); actx.dV = dVfull.data();
					actx.dModel = dModel; actx.dModelKV = dModelKV;
					actx.dHead = dHead; actx.nHeads = nHeads; actx.nKVHeads = nKVHeads;
					actx.T = T; actx.groupSize = groupSize;
					actx.causal = causal; actx.keyAllowed = NULL;
					actx.nChunksPerHead = 1u; actx.totalItems = nKVHeads;
					actx.dKVscratch = NULL;
					pool.parallel_for(nKVHeads, attn_bwd_body, &actx);
				}
				else
				{
					const unsigned int totalItems = nHeads * nChunksPerHead;
					const size_t scratchPerItem = static_cast<size_t>(T) * dHead * 2u;
					const size_t totalScratch = static_cast<size_t>(totalItems) * scratchPerItem;
					if (transformerScratch.dKVscratch.size() < totalScratch)
						transformerScratch.dKVscratch.resize(totalScratch);
					std::fill(transformerScratch.dKVscratch.begin(), transformerScratch.dKVscratch.begin() + totalScratch, 0.0f);

					AttnBwdCtx actx;
					actx.Q = Qfull; actx.K = Kfull; actx.V = Vfull;
					actx.dO = dAttnConcat.data();
					actx.dQ = dQfull.data(); actx.dK = dKfull.data(); actx.dV = dVfull.data();
					actx.dModel = dModel; actx.dModelKV = dModelKV;
					actx.dHead = dHead; actx.nHeads = nHeads; actx.nKVHeads = nKVHeads;
					actx.T = T; actx.groupSize = groupSize;
					actx.causal = causal; actx.keyAllowed = NULL;
					actx.nChunksPerHead = nChunksPerHead; actx.totalItems = totalItems;
					actx.dKVscratch = &transformerScratch.dKVscratch[0];
					pool.parallel_for(totalItems, attn_bwd_body, &actx);

					AttnBwdReduceCtx rctx;
					rctx.dKVscratch = &transformerScratch.dKVscratch[0];
					rctx.dK = dKfull.data(); rctx.dV = dVfull.data();
					rctx.dHead = dHead; rctx.dModelKV = dModelKV; rctx.T = T;
					rctx.nHeads = nHeads; rctx.nKVHeads = nKVHeads;
					rctx.groupSize = groupSize; rctx.nChunksPerHead = nChunksPerHead;
					pool.parallel_for(nKVHeads, attn_bwd_reduce_body, &rctx);
				}
			}
			else
			{
				for (unsigned int h = 0; h < nHeads; ++h)
				{
					const unsigned int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0u ? (h / groupSize) : 0u);
					glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_strided(
					    Qfull + static_cast<size_t>(h) * static_cast<size_t>(dHead), dModel,
					    Kfull + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    Vfull + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    dAttnConcat.data() + static_cast<size_t>(h) * static_cast<size_t>(dHead), dModel,
					    T, dHead, dHead, causal,
					    dQfull.data() + static_cast<size_t>(h) * static_cast<size_t>(dHead), dModel,
					    dKfull.data() + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    dVfull.data() + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    NULL);
				}
			}
		}

		// RoPE backward
		if (useRope && ropeInvFreq)
		{
			const bool ropeWorthParallel = (static_cast<unsigned long long>(T) * ropeDim >= 4096ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (ropeWorthParallel && pool.numThreads() > 1u)
			{
				if (nHeads > 1u)
				{
					RopeFwdCtx rctx;
					rctx.buf = dQfull.data(); rctx.T = T; rctx.rowStride = dModel; rctx.dHead = dHead;
					rctx.ropeDim = ropeDim; rctx.invFreq = ropeInvFreq; rctx.inverse = true;
					pool.parallel_for(nHeads, rope_body, &rctx);
				}
				else
				{
					glades::transformer_kernels::rope_apply_inplace_strided(dQfull.data(), T, dModel, dHead, ropeDim, *ropeInvFreq, true);
				}
				if (nKVHeads > 1u)
				{
					RopeFwdCtx rctx;
					rctx.buf = dKfull.data(); rctx.T = T; rctx.rowStride = dModelKV; rctx.dHead = dHead;
					rctx.ropeDim = ropeDim; rctx.invFreq = ropeInvFreq; rctx.inverse = true;
					pool.parallel_for(nKVHeads, rope_body, &rctx);
				}
				else
				{
					for (unsigned int hk = 0; hk < nKVHeads; ++hk)
						glades::transformer_kernels::rope_apply_inplace_strided(
						    dKfull.data() + static_cast<size_t>(hk) * static_cast<size_t>(dHead), T, dModelKV, dHead, ropeDim, *ropeInvFreq, true);
				}
			}
			else
			{
				for (unsigned int h = 0; h < nHeads; ++h)
					glades::transformer_kernels::rope_apply_inplace_strided(
					    dQfull.data() + static_cast<size_t>(h) * static_cast<size_t>(dHead), T, dModel, dHead, ropeDim, *ropeInvFreq, true);
				for (unsigned int hk = 0; hk < nKVHeads; ++hk)
					glades::transformer_kernels::rope_apply_inplace_strided(
					    dKfull.data() + static_cast<size_t>(hk) * static_cast<size_t>(dHead), T, dModelKV, dHead, ropeDim, *ropeInvFreq, true);
			}
		}

		// QKV projection backward
		std::vector<float, glades::AlignedAllocator<float, 64> >& dX1 = transformerScratch.dX1;
		std::vector<float, glades::AlignedAllocator<float, 64> >& dXtmp = transformerScratch.dXtmp;
		if (dX1.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
			dX1.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
		if (dXtmp.size() != dX1.size()) dXtmp.resize(dX1.size());
		std::fill(dX1.begin(), dX1.end(), 0.0f);
		{
			linear_backward_accum_maybe_lowp(x1, dQfull.data(), T, dModel, dModel, b.gWq, b.gBq, b.Wq, b.WqLowp, useLowpWeights, lowpDType, dXtmp.data());
			for (size_t i = 0; i < dX1.size(); ++i) dX1[i] += dXtmp[i];
		}
		{
			linear_backward_accum_maybe_lowp(x1, dKfull.data(), T, dModel, dModelKV, b.gWk, b.gBk, b.Wk, b.WkLowp, useLowpWeights, lowpDType, dXtmp.data());
			for (size_t i = 0; i < dX1.size(); ++i) dX1[i] += dXtmp[i];
		}
		{
			linear_backward_accum_maybe_lowp(x1, dVfull.data(), T, dModel, dModelKV, b.gWv, b.gBv, b.Wv, b.WvLowp, useLowpWeights, lowpDType, dXtmp.data());
			for (size_t i = 0; i < dX1.size(); ++i) dX1[i] += dXtmp[i];
		}

		// LN1 backward
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

		for (size_t i = 0; i < dH.size(); ++i)
			dH[i] += dHInFromLN[i];
	} // layers

	// Embedding dropout backward
	{
		const float embDropRate = trainingConfig.transformer.embeddingDropoutRate;
		if (embDropRate > 0.0f)
		{
			const size_t n = static_cast<size_t>(T) * static_cast<size_t>(dModel);
			const unsigned char* mask = transformerScratch.dropoutMaskEmb.empty() ? NULL : &transformerScratch.dropoutMaskEmb[0];
			if (mask)
			{
				const float scale = 1.0f / (1.0f - embDropRate);
				for (size_t i = 0; i < n; ++i)
					dH[i] *= mask[i] ? scale : 0.0f;
			}
		}
	}

	// Backprop input projection
	if (tokenLM)
	{
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
		(void)dX;
	}
}


// ---------------------------------------------------------------------------
// Extracted GPU training epoch (previously inlined in SGDHelper_TRANSFORMER).
// Runs the complete forward/backward/optimizer loop on GPU for all sequences
// in the epoch.
// ---------------------------------------------------------------------------
#ifdef GLADES_HAVE_CUDA
void glades::NNetwork::transformerGpuTrainEpoch(const TransformerEpochCfg& cfg, unsigned int seqCount,
                                                int epochIdx, int64_t epochStartMs,
                                                unsigned long long& tokensProcessed,
                                                unsigned long long& targetsProcessed,
                                                double& tokenLmNllSum,
                                                unsigned long long& tokenLmTokenCount,
                                                unsigned long long& clsCorrect,
                                                unsigned long long& clsTotal,
                                                shmea::GLogger* logger)
{
	using namespace glades::logfmt;

	TensorTransformerState& tt = tensorTransformer;

	const unsigned int dModel = cfg.dModel;
	const unsigned int dFF = cfg.dFF;
	const unsigned int nHeads = cfg.nHeads;
	const unsigned int nKVHeads = cfg.nKVHeads;
	const unsigned int nLayers = cfg.nLayers;
	const unsigned int vocabSize = cfg.vocabSize;
	const unsigned int inputSize = cfg.inputSize;
	const unsigned int outSize = cfg.outSize;
	const bool tokenLM = cfg.tokenLM;
	const bool tieEmb = cfg.tieEmb;
	const bool causal = cfg.causal;
	const int padTokenId = cfg.padTokenId;
	const int posEnc = cfg.posEnc;
	const int normType = cfg.normType;
	const int ffnKind = cfg.ffnKind;
	const int ffnAct = cfg.ffnAct;
	const float lnEps = cfg.lnEps;
	const int ropeDimOverride = cfg.ropeDimOverride;
	const unsigned int seqBatchMax = cfg.seqBatchMax;
	const unsigned int optimizerStepsPerEpoch = (seqCount + seqBatchMax - 1u) / seqBatchMax;

	unsigned int seqInBatch = 0u;
	unsigned int timeStepsInBatch = 0u;

	// Progress logging setup
	unsigned int progressEverySeq = 1u;
	if (seqCount > 20u)
		progressEverySeq = seqCount / 20u;
	if (progressEverySeq == 0u)
		progressEverySeq = 1u;
	int64_t lastProgressMs = epochStartMs;
	static const int64_t kProgressIntervalMs = 5000;

	if (nHeads == 0u || (dModel % nHeads) != 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerGpuTrainEpoch: dModel is not divisible by nHeads");
		storeRunningFlag(false);
		return;
	}
	if ((nHeads % nKVHeads) != 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerGpuTrainEpoch: nHeads is not divisible by nKVHeads");
		storeRunningFlag(false);
		return;
	}
	const unsigned int dHead = dModel / nHeads;
	const unsigned int dModelKV = nKVHeads * dHead;
	const unsigned int ff1Width = (ffnKind == 1) ? (2u * dFF) : dFF;
	const bool useRope = (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE));
	cudaEvent_t gpuTransferReadyEvent = gpu::createEvent(false);
	cudaEvent_t gpuComputeReadyEvent = gpu::createEvent(false);

	for (unsigned int s = 0; s < seqCount; ++s)
	{
		if (!loadRunningFlag())
			break;

		const unsigned int T = di->getTrainSequenceLength(s);
		if (T == 0u)
			continue;
		tokensProcessed += static_cast<unsigned long long>(T);


		// Ensure GPU scratch is big enough for this sequence.
		if (!gpuTransformerScratch)
			gpuTransformerScratch = new gpu::GpuTransformerScratch();

		if (!gpuTransformerScratch->initialized || gpuTransformerScratch->T < T)
		{
			if (!gpuTransformerScratch->allocate(T, inputSize, outSize, dModel, dFF, dModelKV, nHeads, nLayers, ff1Width))
			{
				// GPU scratch allocation failed, fall through to CPU.
				break;
			}
		}

		// Upload token IDs for this sequence.
		if (tokenLM)
		{
			std::vector<int> tokenIdsInt(T);
			for (unsigned int t = 0; t < T; ++t)
			{
				int tid = 0;
				di->getTrainSequenceTokenId(s, t, tid);
				tokenIdsInt[t] = tid;
			}
			gpuTransformerScratch->tokenIds.uploadAsync(&tokenIdsInt[0], T);
			gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
			gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);

			// Forward: embedding gather
			gpu::embedding_gather(
			    gpuTransformerWeights->tokE.data(),
			    gpuTransformerScratch->tokenIds.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize),
			    static_cast<int>(dModel),
			    gpuTransformerScratch->h.data());
		}
		else
		{
			// Upload input features and run linear projection.
			std::vector<float> xHost(static_cast<size_t>(T) * inputSize);
			for (unsigned int t = 0; t < T; ++t)
			{
				const float* row = NULL;
				unsigned int rowSize = 0u;
				di->getTrainSequenceRowView(s, t, row, rowSize);
				const size_t off = static_cast<size_t>(t) * inputSize;
				for (unsigned int f = 0; f < inputSize; ++f)
					xHost[off + f] = (row && f < rowSize) ? row[f] : 0.0f;
			}
			gpuTransformerScratch->x.uploadAsync(&xHost[0], xHost.size());
			gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
			gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);

			// Input projection: h = x * WIn^T + bIn
			gpu::sgemm_rowmajor_abt(static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(inputSize),
			                     1.0f,
			                     gpuTransformerScratch->x.data(), static_cast<int>(inputSize),
			                     gpuTransformerWeights->WIn.data(), static_cast<int>(inputSize),
			                     0.0f,
			                     gpuTransformerScratch->h.data(), static_cast<int>(dModel));
			gpu::add_bias(gpuTransformerScratch->h.data(),
			              gpuTransformerWeights->bIn.data(),
			              static_cast<int>(T), static_cast<int>(dModel));
		}

		// Upload RoPE invFreq to scratch (shared across all layers).
		unsigned int fwdRopeHalfDim = 0u;
		if (useRope && !transformerPosEncCache.ropeInvFreq.empty())
		{
			const unsigned int rd = (ropeDimOverride > 0 && static_cast<unsigned int>(ropeDimOverride) < dHead)
			                        ? static_cast<unsigned int>(ropeDimOverride) : dHead;
			fwdRopeHalfDim = rd / 2u;
			std::vector<float> invFreqF(fwdRopeHalfDim);
			for (unsigned int i = 0; i < fwdRopeHalfDim && i < transformerPosEncCache.ropeInvFreq.size(); ++i)
				invFreqF[i] = static_cast<float>(transformerPosEncCache.ropeInvFreq[i]);
			gpuTransformerScratch->gpuInvFreq.uploadAsync(&invFreqF[0], invFreqF.size());
			gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
		}

		// Per-layer transformer blocks.
		for (unsigned int li = 0; li < nLayers; ++li)
		{
			gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[li];
			const size_t layerOff = static_cast<size_t>(li) * static_cast<size_t>(T);

			// Input to this layer is h (or hAfterFF from previous layer).
			const float* layerIn = (li == 0) ? gpuTransformerScratch->h.data()
			                                 : (gpuTransformerScratch->hAfterFF.data() + static_cast<size_t>(li - 1) * T * dModel);

			float* x1_l = gpuTransformerScratch->x1.data() + static_cast<size_t>(li) * T * dModel;
			float* ln1Mean_l = gpuTransformerScratch->ln1Mean.data() + layerOff;
			float* ln1InvStd_l = gpuTransformerScratch->ln1InvStd.data() + layerOff;

			// Pre-LN 1
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				gpu::rmsnorm_forward(layerIn, gb.ln1Gamma.data(), lnEps,
				                      static_cast<int>(T), static_cast<int>(dModel),
				                      x1_l, ln1InvStd_l);
			}
			else
			{
				gpu::layernorm_forward(layerIn, gb.ln1Gamma.data(), gb.ln1Beta.data(),
				                        lnEps, static_cast<int>(T), static_cast<int>(dModel),
				                        x1_l, ln1Mean_l, ln1InvStd_l);
			}

			// QKV projections
			float* Q_l = gpuTransformerScratch->Q.data() + static_cast<size_t>(li) * T * dModel;
			float* K_l = gpuTransformerScratch->K.data() + static_cast<size_t>(li) * T * dModelKV;
			float* V_l = gpuTransformerScratch->V.data() + static_cast<size_t>(li) * T * dModelKV;

			gpu::sgemm_rowmajor_abt(static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel),
			                     1.0f, x1_l, static_cast<int>(dModel),
			                     gb.Wq.data(), static_cast<int>(dModel),
			                     0.0f, Q_l, static_cast<int>(dModel));
			gpu::add_bias(Q_l, gb.bq.data(), static_cast<int>(T), static_cast<int>(dModel));

			gpu::sgemm_rowmajor_abt(static_cast<int>(T), static_cast<int>(dModelKV), static_cast<int>(dModel),
			                     1.0f, x1_l, static_cast<int>(dModel),
			                     gb.Wk.data(), static_cast<int>(dModel),
			                     0.0f, K_l, static_cast<int>(dModelKV));
			gpu::add_bias(K_l, gb.bk.data(), static_cast<int>(T), static_cast<int>(dModelKV));

			gpu::sgemm_rowmajor_abt(static_cast<int>(T), static_cast<int>(dModelKV), static_cast<int>(dModel),
			                     1.0f, x1_l, static_cast<int>(dModel),
			                     gb.Wv.data(), static_cast<int>(dModel),
			                     0.0f, V_l, static_cast<int>(dModelKV));
			gpu::add_bias(V_l, gb.bv.data(), static_cast<int>(T), static_cast<int>(dModelKV));

			// RoPE (if enabled) — fused Q+K in single kernel launch
			if (useRope && !transformerPosEncCache.ropeInvFreq.empty())
			{
				if (li == 0u)
					gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);
				const unsigned int rd = (ropeDimOverride > 0 && static_cast<unsigned int>(ropeDimOverride) < dHead)
				                        ? static_cast<unsigned int>(ropeDimOverride) : dHead;
				// Use persistent gpuInvFreq from scratch (uploaded before layer loop).
				gpu::rope_apply_qk(Q_l, K_l, gpuTransformerScratch->gpuInvFreq.data(),
				                    static_cast<int>(T), static_cast<int>(nHeads),
				                    static_cast<int>(nKVHeads), static_cast<int>(dHead),
				                    static_cast<int>(rd / 2u));
			}

			// Flash-style packed multi-head attention without materializing T*T scores/probs.
			float* attnConcat_l = gpuTransformerScratch->attnConcat.data() + static_cast<size_t>(li) * T * dModel;
			gpu::flash_attention_multihead_forward(
			    Q_l, K_l, V_l,
			    static_cast<int>(T),
			    static_cast<int>(nHeads),
			    static_cast<int>(nKVHeads),
			    static_cast<int>(dHead),
			    static_cast<int>(dModel),
			    static_cast<int>(dModelKV),
			    causal,
			    attnConcat_l);

			// Wo projection
			float* attnOut_l = gpuTransformerScratch->attnOut.data() + static_cast<size_t>(li) * T * dModel;
			gpu::sgemm_rowmajor_abt(static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel),
			                     1.0f, attnConcat_l, static_cast<int>(dModel),
			                     gb.Wo.data(), static_cast<int>(dModel),
			                     0.0f, attnOut_l, static_cast<int>(dModel));
			gpu::add_bias(attnOut_l, gb.bo.data(), static_cast<int>(T), static_cast<int>(dModel));

			// Residual 1: hAfterAttn = layerIn + attnOut
			float* hAfterAttn_l = gpuTransformerScratch->hAfterAttn.data() + static_cast<size_t>(li) * T * dModel;
			gpu::add_two(hAfterAttn_l, layerIn, attnOut_l, static_cast<int>(T * dModel));

			// Pre-LN 2
			float* x2_l = gpuTransformerScratch->x2.data() + static_cast<size_t>(li) * T * dModel;
			float* ln2Mean_l = gpuTransformerScratch->ln2Mean.data() + layerOff;
			float* ln2InvStd_l = gpuTransformerScratch->ln2InvStd.data() + layerOff;

			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				gpu::rmsnorm_forward(hAfterAttn_l, gb.ln2Gamma.data(), lnEps,
				                      static_cast<int>(T), static_cast<int>(dModel),
				                      x2_l, ln2InvStd_l);
			}
			else
			{
				gpu::layernorm_forward(hAfterAttn_l, gb.ln2Gamma.data(), gb.ln2Beta.data(),
				                        lnEps, static_cast<int>(T), static_cast<int>(dModel),
				                        x2_l, ln2Mean_l, ln2InvStd_l);
			}

			// FFN
			float* ff1_l = gpuTransformerScratch->ff1.data() + static_cast<size_t>(li) * T * ff1Width;
			float* ff1Act_l = gpuTransformerScratch->ff1Act.data() + static_cast<size_t>(li) * T * dFF;
			float* ffOut_l = gpuTransformerScratch->ffOut.data() + static_cast<size_t>(li) * T * dModel;

			// FF1: x2 * W1^T + b1
			gpu::sgemm_rowmajor_abt(static_cast<int>(T), static_cast<int>(ff1Width), static_cast<int>(dModel),
			                     1.0f, x2_l, static_cast<int>(dModel),
			                     gb.W1.data(), static_cast<int>(dModel),
			                     0.0f, ff1_l, static_cast<int>(ff1Width));
			gpu::add_bias(ff1_l, gb.b1.data(), static_cast<int>(T), static_cast<int>(ff1Width));

			// Activation
			if (ffnKind == 1) // SwiGLU
			{
				gpu::swiglu_forward(ff1_l, static_cast<int>(T), static_cast<int>(dFF), ff1Act_l);
			}
			else if (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
			{
				gpu::gelu_forward(ff1_l, static_cast<int>(T * dFF), ff1Act_l);
			}
			else
			{
				gpu::relu_forward(ff1_l, static_cast<int>(T * dFF), ff1Act_l);
			}

			// FF2: ffAct * W2^T + b2
			gpu::sgemm_rowmajor_abt(static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dFF),
			                     1.0f, ff1Act_l, static_cast<int>(dFF),
			                     gb.W2.data(), static_cast<int>(dFF),
			                     0.0f, ffOut_l, static_cast<int>(dModel));
			gpu::add_bias(ffOut_l, gb.b2.data(), static_cast<int>(T), static_cast<int>(dModel));

			// Residual 2: hAfterFF = hAfterAttn + ffOut
			float* hAfterFF_l = gpuTransformerScratch->hAfterFF.data() + static_cast<size_t>(li) * T * dModel;
			gpu::add_two(hAfterFF_l, hAfterAttn_l, ffOut_l, static_cast<int>(T * dModel));
		}

		// Final LayerNorm
		const float* finalH = gpuTransformerScratch->hAfterFF.data() + static_cast<size_t>(nLayers - 1) * T * dModel;
		float* hPostFinalLN = gpuTransformerScratch->hPostFinalLN.data();
		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
		{
			gpu::rmsnorm_forward(finalH, gpuTransformerWeights->lnFinalGamma.data(), lnEps,
			                     static_cast<int>(T), static_cast<int>(dModel),
			                     hPostFinalLN, gpuTransformerScratch->lnFinalInvStd.data());
		}
		else
		{
			gpu::layernorm_forward(finalH, gpuTransformerWeights->lnFinalGamma.data(),
			                       gpuTransformerWeights->lnFinalBeta.data(), lnEps,
			                       static_cast<int>(T), static_cast<int>(dModel),
			                       hPostFinalLN, gpuTransformerScratch->lnFinalMean.data(),
			                       gpuTransformerScratch->lnFinalInvStd.data());
		}

		// Output logits
		if (tokenLM && tieEmb)
		{
			// logits = hPostFinalLN * E^T + lmBias
			gpu::sgemm_rowmajor_abt(static_cast<int>(T), static_cast<int>(vocabSize), static_cast<int>(dModel),
			                     1.0f, hPostFinalLN, static_cast<int>(dModel),
			                     gpuTransformerWeights->tokE.data(), static_cast<int>(dModel),
			                     0.0f, gpuTransformerScratch->logits.data(), static_cast<int>(vocabSize));
			gpu::add_bias(gpuTransformerScratch->logits.data(),
			              gpuTransformerWeights->lmBias.data(),
			              static_cast<int>(T), static_cast<int>(vocabSize));
		}

		// Softmax
		gpu::softmax_forward(gpuTransformerScratch->logits.data(),
		                      static_cast<int>(T), static_cast<int>(outSize),
		                      gpuTransformerScratch->probs.data());

		// === Loss / metrics ===
		std::vector<int> gpuTargetIds;
		unsigned int gpuValidTargets = 0u;
		if (tokenLM)
		{
			gpuTargetIds.resize(T);
			for (unsigned int t = 0; t < T; ++t)
			{
				int yid = padTokenId;
				di->getTrainSequenceExpectedTokenId(s, t, yid);
				gpuTargetIds[t] = yid;
			}
			// Upload targets to persistent scratch buffer.
			gpuTransformerScratch->gpuTargetsT.uploadAsync(&gpuTargetIds[0], T);
			gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
			gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);

			// GPU loss: cross-entropy NLL.
			gpu::cross_entropy_nll_loss(
			    gpuTransformerScratch->probs.data(),
			    gpuTransformerScratch->gpuTargetsT.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize),
			    padTokenId,
			    gpuTransformerScratch->lossSum.data(),
			    gpuTransformerScratch->lossCount.data());

			// GPU accuracy: argmax match count.
			gpu::argmax_count_matches(
			    gpuTransformerScratch->probs.data(),
			    gpuTransformerScratch->gpuTargetsT.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize),
			    padTokenId,
			    gpuTransformerScratch->correctCount.data(),
			    gpuTransformerScratch->validCount.data());

			// Pack 4 loss scalars into contiguous buffer, download once.
			gpu::pack_loss_scalars(
			    gpuTransformerScratch->lossSum.data(),
			    gpuTransformerScratch->lossCount.data(),
			    gpuTransformerScratch->correctCount.data(),
			    gpuTransformerScratch->validCount.data(),
			    gpuTransformerScratch->lossPack.data());
			int lossPacked[4];
			gpu::recordEvent(gpuComputeReadyEvent, gpu::computeStream());
			gpu::streamWaitEvent(gpu::transferStream(), gpuComputeReadyEvent);
			gpuTransformerScratch->lossPack.downloadAsync(lossPacked, 4);
			gpu::synchronizeTransferStream();
			float lossVal;
			memcpy(&lossVal, &lossPacked[0], sizeof(float));
			int lossCountVal = lossPacked[1], correctVal = lossPacked[2], validVal = lossPacked[3];

			gpuValidTargets = static_cast<unsigned int>(lossCountVal);
			tokenLmNllSum += static_cast<double>(lossVal);
			tokenLmTokenCount += static_cast<unsigned long long>(lossCountVal);
			clsCorrect += static_cast<unsigned long long>(correctVal);
			clsTotal += static_cast<unsigned long long>(validVal);

			targetsProcessed += static_cast<unsigned long long>(gpuValidTargets);
		}
		else
		{
			targetsProcessed += static_cast<unsigned long long>(T);
		}

		// Periodic progress logging (mirrors CPU path).
		if (logger && (s + 1u) < seqCount)
		{
			const int64_t nowMs = getCurrentTimeMilliseconds();
			const bool dueBySeq = (((s + 1u) % progressEverySeq) == 0u);
			const bool dueByTime = ((nowMs - lastProgressMs) >= kProgressIntervalMs);
			if (dueBySeq || dueByTime)
			{
				lastProgressMs = nowMs;
				const double elapsedMs = static_cast<double>(nowMs - epochStartMs);
				const double tokPerSec = (elapsedMs > 0.0) ? (static_cast<double>(targetsProcessed) / (elapsedMs / 1000.0)) : 0.0;
				const double meanNll = (tokenLmTokenCount > 0ULL) ? (tokenLmNllSum / static_cast<double>(tokenLmTokenCount)) : 0.0;

				std::ostringstream oss;
				oss << "event=nn_epoch_progress";
				append_logfmt_kv(oss, "net_type", netType);
				append_logfmt_kv(oss, "run_type", std::string("train"));
				append_logfmt_kv(oss, "gpu", true);
				append_logfmt_kv(oss, "epoch", epochIdx);
				append_logfmt_kv(oss, "seq_done", s + 1u);
				append_logfmt_kv(oss, "seq_total", seqCount);
				append_logfmt_kv(oss, "tokens_seen", tokensProcessed);
				append_logfmt_kv(oss, "targets_seen", targetsProcessed);
				append_logfmt_kv(oss, "targets_per_sec", tokPerSec);
				if (tokenLM)
				{
					append_logfmt_kv(oss, "token_lm_loss_kind", std::string("full_softmax"));
					append_logfmt_kv(oss, "nll", meanNll);
					double ppl = 0.0;
					if (tokenLmTokenCount > 0ULL)
					{
						double arg = meanNll;
						if (arg > 80.0) arg = 80.0;
						if (arg < -80.0) arg = -80.0;
						ppl = exp(arg);
					}
					append_logfmt_kv(oss, "perplexity", ppl);
					append_logfmt_kv(oss, "acc_top1", (clsTotal > 0ULL) ? (100.0 * static_cast<double>(clsCorrect) / static_cast<double>(clsTotal)) : 0.0);
				}
				else
				{
					append_logfmt_kv(oss, "loss_so_far", overallTotalError);
				}
				append_logfmt_kv(oss, "lr_mult", lrScheduleMultiplier);
				if (trainingConfig.globalGradClipNorm > 0.0f)
				{
					append_logfmt_kv(oss, "grad_norm", lastGradNorm);
					append_logfmt_kv(oss, "grad_norm_scale", lastGradNormScale);
				}
				append_logfmt_kv(oss, "optimizer_step", static_cast<unsigned long long>(tensorTransformer.optimizerStep));
				logger->info("NNetwork", shmea::GString(oss.str().c_str()));
			}
		}

		// === GPU Backward pass ===
		if (seqInBatch == 0u)
		{
			gpu::zeroTransformerGradients(*gpuTransformerWeights);
			timeStepsInBatch = 0u;
		}

		// Compute dLogits on GPU.
		const float* bwdFinalH = gpuTransformerScratch->hAfterFF.data() +
		    static_cast<size_t>(nLayers - 1) * T * dModel;
		const float* bwdPostFinalLN = gpuTransformerScratch->hPostFinalLN.data();

		if (tokenLM)
		{
			// dLogits = probs - one_hot(targets)
			gpu::softmax_cross_entropy_bwd(
			    gpuTransformerScratch->probs.data(),
			    gpuTransformerScratch->gpuTargetsT.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize),
			    gpuTransformerScratch->dLogits.data());

			// Backprop tied LM head: logits = hPostFinalLN * E^T + lmBias
			// dH (w.r.t. hPostFinalLN) = dLogits * E
			gpu::sgemm_rowmajor(
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(vocabSize),
			    1.0f, gpuTransformerScratch->dLogits.data(), static_cast<int>(vocabSize),
			    gpuTransformerWeights->tokE.data(), static_cast<int>(dModel),
			    0.0f, gpuTransformerScratch->dH.data(), static_cast<int>(dModel));

			// gTokE += dLogits^T * hPostFinalLN  [vocabSize, dModel]
			gpu::sgemm_rowmajor_atb(
			    static_cast<int>(vocabSize), static_cast<int>(dModel), static_cast<int>(T),
			    1.0f, gpuTransformerScratch->dLogits.data(), static_cast<int>(vocabSize),
			    bwdPostFinalLN, static_cast<int>(dModel),
			    1.0f, gpuTransformerWeights->gTokE.data(), static_cast<int>(dModel));

			// gLmBias += sum_rows(dLogits)
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dLogits.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize),
			    1.0f, gpuTransformerWeights->gLmBias.data());

			timeStepsInBatch += gpuValidTargets;
		}
		else
		{
			// Non-tokenLM: dLogits computed from probs - expected on CPU, upload.
			std::vector<float> probsHost(static_cast<size_t>(T) * outSize);
			gpuTransformerScratch->probs.download(&probsHost[0], probsHost.size());
			std::vector<float> dLogitsHost(static_cast<size_t>(T) * outSize, 0.0f);
			for (unsigned int t = 0; t < T; ++t)
			{
				const float* expRow = NULL;
				unsigned int expSize = 0u;
				di->getTrainSequenceExpectedRowView(s, t, expRow, expSize);
				const size_t off = static_cast<size_t>(t) * outSize;
				for (unsigned int k = 0; k < outSize; ++k)
				{
					const float expv = (expRow && k < expSize) ? expRow[k] : 0.0f;
					dLogitsHost[off + k] = probsHost[off + k] - expv;
				}
			}
			gpuTransformerScratch->dLogits.upload(&dLogitsHost[0], dLogitsHost.size());

			// dH (w.r.t. hPostFinalLN) = dLogits * WOut  [T, dModel]
			gpu::sgemm_rowmajor(
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(outSize),
			    1.0f, gpuTransformerScratch->dLogits.data(), static_cast<int>(outSize),
			    gpuTransformerWeights->WOut.data(), static_cast<int>(dModel),
			    0.0f, gpuTransformerScratch->dH.data(), static_cast<int>(dModel));

			// gWOut += dLogits^T * hPostFinalLN  [outSize, dModel]
			gpu::sgemm_rowmajor_atb(
			    static_cast<int>(outSize), static_cast<int>(dModel), static_cast<int>(T),
			    1.0f, gpuTransformerScratch->dLogits.data(), static_cast<int>(outSize),
			    bwdPostFinalLN, static_cast<int>(dModel),
			    1.0f, gpuTransformerWeights->gWOut.data(), static_cast<int>(dModel));

			// gBOut += sum_rows(dLogits)
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dLogits.data(),
			    static_cast<int>(T), static_cast<int>(outSize),
			    1.0f, gpuTransformerWeights->gBOut.data());

			timeStepsInBatch += T;
		}

		// Backprop Final LayerNorm: dH (w.r.t. hPostFinalLN) -> dH (w.r.t. hFinal)
		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
		{
			gpu::rmsnorm_backward(
			    gpuTransformerScratch->dH.data(), bwdFinalH,
			    gpuTransformerWeights->lnFinalGamma.data(),
			    gpuTransformerScratch->lnFinalInvStd.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    gpuTransformerScratch->dH2.data(),
			    gpuTransformerWeights->gLnFinalGamma.data());
		}
		else
		{
			gpu::layernorm_backward(
			    gpuTransformerScratch->dH.data(), bwdFinalH,
			    gpuTransformerWeights->lnFinalGamma.data(),
			    gpuTransformerScratch->lnFinalMean.data(),
			    gpuTransformerScratch->lnFinalInvStd.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    gpuTransformerScratch->dH2.data(),
			    gpuTransformerWeights->gLnFinalGamma.data(),
			    gpuTransformerWeights->gLnFinalBeta.data());
		}
		// dH2 now has gradient w.r.t. hFinal; swap into dH for block backprop.
		gpu::device_memcpy_d2d(gpuTransformerScratch->dH.data(),
		                       gpuTransformerScratch->dH2.data(),
		                       static_cast<size_t>(T) * dModel * sizeof(float));

		// RoPE invFreq already uploaded to scratch before forward layer loop.
		const unsigned int ropeHalfDim = fwdRopeHalfDim;

		// Backprop through blocks (reverse order).
		for (int li = static_cast<int>(nLayers) - 1; li >= 0; --li)
		{
			gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[li];
			const size_t layerOff = static_cast<size_t>(li) * static_cast<size_t>(T);

			const float* layerIn = (li == 0) ? gpuTransformerScratch->h.data()
			    : (gpuTransformerScratch->hAfterFF.data() + static_cast<size_t>(li - 1) * T * dModel);
			const float* x1_l = gpuTransformerScratch->x1.data() + static_cast<size_t>(li) * T * dModel;
			const float* x2_l = gpuTransformerScratch->x2.data() + static_cast<size_t>(li) * T * dModel;
			const float* ff1_l = gpuTransformerScratch->ff1.data() + static_cast<size_t>(li) * T * ff1Width;
			const float* hAfterAttn_l = gpuTransformerScratch->hAfterAttn.data() + static_cast<size_t>(li) * T * dModel;
			const float* attnConcat_l = gpuTransformerScratch->attnConcat.data() + static_cast<size_t>(li) * T * dModel;
			const float* ff1Act_l = gpuTransformerScratch->ff1Act.data() + static_cast<size_t>(li) * T * dFF;

			// dH is gradient w.r.t. hAfterFF[li].
			// Residual: hAfterFF = hAfterAttn + ffOut => dFFOut = dH, dHAfterAttn (residual) = dH.
			// --- FFN backward ---
			// ffOut = W2 * ff1Act + b2  =>  dFF1Act = dH * W2^T, gW2 += dH^T * ff1Act
			gpu::sgemm_rowmajor(
			    static_cast<int>(T), static_cast<int>(dFF), static_cast<int>(dModel),
			    1.0f, gpuTransformerScratch->dH.data(), static_cast<int>(dModel),
			    gb.W2.data(), static_cast<int>(dFF),
			    0.0f, gpuTransformerScratch->dFF1Act.data(), static_cast<int>(dFF));
			gpu::sgemm_rowmajor_atb(
			    static_cast<int>(dModel), static_cast<int>(dFF), static_cast<int>(T),
			    1.0f, gpuTransformerScratch->dH.data(), static_cast<int>(dModel),
			    ff1Act_l, static_cast<int>(dFF),
			    1.0f, gb.gW2.data(), static_cast<int>(dFF));
			// gB2 += sum_rows(dH)
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dH.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    1.0f, gb.gB2.data());

			// Activation backward.
			if (ffnKind == 1) // SwiGLU
			{
				gpu::swiglu_backward(
				    gpuTransformerScratch->dFF1Act.data(), ff1_l,
				    static_cast<int>(T), static_cast<int>(dFF),
				    gpuTransformerScratch->dFF1Cat.data());
				// ff1 = W1 * x2 + b1 => dX2 = dFF1Cat * W1^T, gW1 += dFF1Cat^T * x2
				gpu::sgemm_rowmajor(
				    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(ff1Width),
				    1.0f, gpuTransformerScratch->dFF1Cat.data(), static_cast<int>(ff1Width),
				    gb.W1.data(), static_cast<int>(dModel),
				    0.0f, gpuTransformerScratch->dX2.data(), static_cast<int>(dModel));
				gpu::sgemm_rowmajor_atb(
				    static_cast<int>(ff1Width), static_cast<int>(dModel), static_cast<int>(T),
				    1.0f, gpuTransformerScratch->dFF1Cat.data(), static_cast<int>(ff1Width),
				    x2_l, static_cast<int>(dModel),
				    1.0f, gb.gW1.data(), static_cast<int>(dModel));
				gpu::reduce_rows_sum(
				    gpuTransformerScratch->dFF1Cat.data(),
				    static_cast<int>(T), static_cast<int>(ff1Width),
				    1.0f, gb.gB1.data());
			}
			else
			{
				// GELU/ReLU backward
				if (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
				{
					gpu::gelu_backward(
					    gpuTransformerScratch->dFF1Act.data(), ff1_l,
					    static_cast<int>(T * dFF),
					    gpuTransformerScratch->dFF1Act.data());
				}
				else
				{
					gpu::relu_backward(
					    gpuTransformerScratch->dFF1Act.data(), ff1_l,
					    static_cast<int>(T * dFF),
					    gpuTransformerScratch->dFF1Act.data());
				}
				// ff1 = W1 * x2 + b1 => dX2, gW1
				gpu::sgemm_rowmajor(
				    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dFF),
				    1.0f, gpuTransformerScratch->dFF1Act.data(), static_cast<int>(dFF),
				    gb.W1.data(), static_cast<int>(dModel),
				    0.0f, gpuTransformerScratch->dX2.data(), static_cast<int>(dModel));
				gpu::sgemm_rowmajor_atb(
				    static_cast<int>(dFF), static_cast<int>(dModel), static_cast<int>(T),
				    1.0f, gpuTransformerScratch->dFF1Act.data(), static_cast<int>(dFF),
				    x2_l, static_cast<int>(dModel),
				    1.0f, gb.gW1.data(), static_cast<int>(dModel));
				gpu::reduce_rows_sum(
				    gpuTransformerScratch->dFF1Act.data(),
				    static_cast<int>(T), static_cast<int>(dFF),
				    1.0f, gb.gB1.data());
			}

			// --- LN2 backward ---
			const float* ln2Mean_l = gpuTransformerScratch->ln2Mean.data() + layerOff;
			const float* ln2InvStd_l = gpuTransformerScratch->ln2InvStd.data() + layerOff;
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				gpu::rmsnorm_backward(
				    gpuTransformerScratch->dX2.data(), hAfterAttn_l,
				    gb.ln2Gamma.data(), ln2InvStd_l,
				    static_cast<int>(T), static_cast<int>(dModel),
				    gpuTransformerScratch->dHAfterAttnFromLN.data(),
				    gb.gLn2Gamma.data());
			}
			else
			{
				gpu::layernorm_backward(
				    gpuTransformerScratch->dX2.data(), hAfterAttn_l,
				    gb.ln2Gamma.data(), ln2Mean_l, ln2InvStd_l,
				    static_cast<int>(T), static_cast<int>(dModel),
				    gpuTransformerScratch->dHAfterAttnFromLN.data(),
				    gb.gLn2Gamma.data(), gb.gLn2Beta.data());
			}

			// Combine: dHAfterAttn = dH (residual) + dHAfterAttnFromLN
			gpu::add_two(gpuTransformerScratch->dH2.data(),
			    gpuTransformerScratch->dH.data(),
			    gpuTransformerScratch->dHAfterAttnFromLN.data(),
			    static_cast<int>(T * dModel));

			// --- Wo backward ---
			// attnOut = Wo * attnConcat + bo  =>  dAttnConcat, gWo
			gpu::sgemm_rowmajor(
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel),
			    1.0f, gpuTransformerScratch->dH2.data(), static_cast<int>(dModel),
			    gb.Wo.data(), static_cast<int>(dModel),
			    0.0f, gpuTransformerScratch->dAttnConcat.data(), static_cast<int>(dModel));
			gpu::sgemm_rowmajor_atb(
			    static_cast<int>(dModel), static_cast<int>(dModel), static_cast<int>(T),
			    1.0f, gpuTransformerScratch->dH2.data(), static_cast<int>(dModel),
			    attnConcat_l, static_cast<int>(dModel),
			    1.0f, gb.gWo.data(), static_cast<int>(dModel));
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dH2.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    1.0f, gb.gBo.data());

			// --- Attention backward (flash-style recompute) ---
			float* Q_l = gpuTransformerScratch->Q.data() + static_cast<size_t>(li) * T * dModel;
			float* K_l = gpuTransformerScratch->K.data() + static_cast<size_t>(li) * T * dModelKV;
			float* V_l = gpuTransformerScratch->V.data() + static_cast<size_t>(li) * T * dModelKV;

			// Zero dK/dV (dQ is overwritten per-head, but dK/dV accumulate for GQA).
			gpu::zero_buffers_batch(gpuTransformerScratch->d_dKdVZeroPtrs,
			    gpuTransformerScratch->d_dKdVZeroSizes, 2);

			gpu::flash_attention_multihead_backward(
			    Q_l, K_l, V_l,
			    attnConcat_l,
			    gpuTransformerScratch->dAttnConcat.data(),
			    static_cast<int>(T),
			    static_cast<int>(nHeads),
			    static_cast<int>(nKVHeads),
			    static_cast<int>(dHead),
			    static_cast<int>(dModel),
			    static_cast<int>(dModelKV),
			    causal,
			    gpuTransformerScratch->dQfull.data(),
			    gpuTransformerScratch->dKfull.data(),
			    gpuTransformerScratch->dVfull.data());

			// --- RoPE backward (inverse rotation) — fused Q+K ---
			if (useRope && gpuTransformerScratch->gpuInvFreq.allocated())
			{
				gpu::rope_apply_qk(gpuTransformerScratch->dQfull.data(),
				    gpuTransformerScratch->dKfull.data(),
				    gpuTransformerScratch->gpuInvFreq.data(),
				    static_cast<int>(T), static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads), static_cast<int>(dHead),
				    static_cast<int>(ropeHalfDim), true);
			}

			// --- Q/K/V projection backward ---
			// dX1 = dQ*Wq^T + dK*Wk^T + dV*Wv^T
			// Also accumulate gWq, gBq, gWk, gBk, gWv, gBv.

			// Q: dXtmp = dQ * Wq^T
			gpu::sgemm_rowmajor(
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel),
			    1.0f, gpuTransformerScratch->dQfull.data(), static_cast<int>(dModel),
			    gb.Wq.data(), static_cast<int>(dModel),
			    0.0f, gpuTransformerScratch->dX1.data(), static_cast<int>(dModel));
			gpu::sgemm_rowmajor_atb(
			    static_cast<int>(dModel), static_cast<int>(dModel), static_cast<int>(T),
			    1.0f, gpuTransformerScratch->dQfull.data(), static_cast<int>(dModel),
			    x1_l, static_cast<int>(dModel),
			    1.0f, gb.gWq.data(), static_cast<int>(dModel));
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dQfull.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    1.0f, gb.gBq.data());

			// K: accumulate dK * Wk^T directly into dX1 (beta=1.0)
			gpu::sgemm_rowmajor(
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModelKV),
			    1.0f, gpuTransformerScratch->dKfull.data(), static_cast<int>(dModelKV),
			    gb.Wk.data(), static_cast<int>(dModel),
			    1.0f, gpuTransformerScratch->dX1.data(), static_cast<int>(dModel));
			gpu::sgemm_rowmajor_atb(
			    static_cast<int>(dModelKV), static_cast<int>(dModel), static_cast<int>(T),
			    1.0f, gpuTransformerScratch->dKfull.data(), static_cast<int>(dModelKV),
			    x1_l, static_cast<int>(dModel),
			    1.0f, gb.gWk.data(), static_cast<int>(dModel));
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dKfull.data(),
			    static_cast<int>(T), static_cast<int>(dModelKV),
			    1.0f, gb.gBk.data());

			// V: accumulate dV * Wv^T directly into dX1 (beta=1.0)
			gpu::sgemm_rowmajor(
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModelKV),
			    1.0f, gpuTransformerScratch->dVfull.data(), static_cast<int>(dModelKV),
			    gb.Wv.data(), static_cast<int>(dModel),
			    1.0f, gpuTransformerScratch->dX1.data(), static_cast<int>(dModel));
			gpu::sgemm_rowmajor_atb(
			    static_cast<int>(dModelKV), static_cast<int>(dModel), static_cast<int>(T),
			    1.0f, gpuTransformerScratch->dVfull.data(), static_cast<int>(dModelKV),
			    x1_l, static_cast<int>(dModel),
			    1.0f, gb.gWv.data(), static_cast<int>(dModel));
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dVfull.data(),
			    static_cast<int>(T), static_cast<int>(dModelKV),
			    1.0f, gb.gBv.data());

			// --- LN1 backward ---
			const float* ln1Mean_l = gpuTransformerScratch->ln1Mean.data() + layerOff;
			const float* ln1InvStd_l = gpuTransformerScratch->ln1InvStd.data() + layerOff;
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				gpu::rmsnorm_backward(
				    gpuTransformerScratch->dX1.data(), layerIn,
				    gb.ln1Gamma.data(), ln1InvStd_l,
				    static_cast<int>(T), static_cast<int>(dModel),
				    gpuTransformerScratch->dHInFromLN.data(),
				    gb.gLn1Gamma.data());
			}
			else
			{
				gpu::layernorm_backward(
				    gpuTransformerScratch->dX1.data(), layerIn,
				    gb.ln1Gamma.data(), ln1Mean_l, ln1InvStd_l,
				    static_cast<int>(T), static_cast<int>(dModel),
				    gpuTransformerScratch->dHInFromLN.data(),
				    gb.gLn1Gamma.data(), gb.gLn1Beta.data());
			}

			// Combine into dH: dH = dH2 + dHInFromLN
			gpu::add_two(gpuTransformerScratch->dH.data(),
			    gpuTransformerScratch->dH2.data(),
			    gpuTransformerScratch->dHInFromLN.data(),
			    static_cast<int>(T * dModel));
		} // layers backward

		// Backprop through input embedding/projection.
		if (tokenLM)
		{
			// gTokE[tokenId[t]] += dH[t] for each timestep.
			gpu::embedding_scatter_add(
			    gpuTransformerWeights->gTokE.data(),
			    gpuTransformerScratch->tokenIds.data(),
			    gpuTransformerScratch->dH.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize), static_cast<int>(dModel));
		}
		else
		{
			// gWIn += dH^T * x, gBIn += sum_rows(dH)
			gpu::sgemm_rowmajor_atb(
			    static_cast<int>(dModel), static_cast<int>(inputSize), static_cast<int>(T),
			    1.0f, gpuTransformerScratch->dH.data(), static_cast<int>(dModel),
			    gpuTransformerScratch->x.data(), static_cast<int>(inputSize),
			    1.0f, gpuTransformerWeights->gWIn.data(), static_cast<int>(inputSize));
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dH.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    1.0f, gpuTransformerWeights->gBIn.data());
		}

		++seqInBatch;

		// === Optimizer step (when batch is complete) ===
		if (seqInBatch >= seqBatchMax && timeStepsInBatch > 0u)
		{
			const float invBatch = 1.0f / static_cast<float>(timeStepsInBatch);
			tensorTransformer.optimizerStep += 1ULL;

			// Warmup + DDP LR scaling (matches CPU path).
			const float warmupMult = trainingConfig.warmup.multiplier(static_cast<int>(tensorTransformer.optimizerStep));
			const float ddpLRScale = (trainingConfig.ddp.enable && trainingConfig.ddp.linearLRScaling)
			                       ? static_cast<float>(glades::ddp::worldSize()) : 1.0f;
			const float gpuExtraLRMult = warmupMult * ddpLRScale;

			const unsigned int stepInEpoch = (s + 1u) / seqBatchMax;
			lrScheduleMultiplier = transformer_schedule_multiplier(
			    trainingConfig.lrSchedule,
			    epochIdx + lrScheduleEpochOffset,
			    stepInEpoch,
			    optimizerStepsPerEpoch);

			// Global gradient norm clipping on GPU.
			float gradScale = 1.0f;
			const float clipNorm = trainingConfig.globalGradClipNorm;
			if (clipNorm > 0.0f)
			{
				// Use lossSum buffer as temporary accumulator for gradient norm.
				gpu::device_memset_bytes(gpuTransformerScratch->lossSum.data(), 0, sizeof(float));

				// Accumulate sum(g^2) across all gradient buffers.
				if (tokenLM)
				{
					gpu::sum_squared_accumulate(gpuTransformerWeights->gTokE.data(),
					    static_cast<int>(gpuTransformerWeights->gTokE.size()),
					    gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gpuTransformerWeights->gLmBias.data(),
					    static_cast<int>(gpuTransformerWeights->gLmBias.size()),
					    gpuTransformerScratch->lossSum.data());
				}
				else
				{
					gpu::sum_squared_accumulate(gpuTransformerWeights->gWIn.data(),
					    static_cast<int>(gpuTransformerWeights->gWIn.size()),
					    gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gpuTransformerWeights->gBIn.data(),
					    static_cast<int>(gpuTransformerWeights->gBIn.size()),
					    gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gpuTransformerWeights->gWOut.data(),
					    static_cast<int>(gpuTransformerWeights->gWOut.size()),
					    gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gpuTransformerWeights->gBOut.data(),
					    static_cast<int>(gpuTransformerWeights->gBOut.size()),
					    gpuTransformerScratch->lossSum.data());
				}
				for (unsigned int gli = 0; gli < nLayers; ++gli)
				{
					gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[gli];
					gpu::sum_squared_accumulate(gb.gWq.data(), static_cast<int>(gb.gWq.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gWk.data(), static_cast<int>(gb.gWk.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gWv.data(), static_cast<int>(gb.gWv.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gWo.data(), static_cast<int>(gb.gWo.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gW1.data(), static_cast<int>(gb.gW1.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gW2.data(), static_cast<int>(gb.gW2.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gBq.data(), static_cast<int>(gb.gBq.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gBk.data(), static_cast<int>(gb.gBk.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gBv.data(), static_cast<int>(gb.gBv.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gBo.data(), static_cast<int>(gb.gBo.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gB1.data(), static_cast<int>(gb.gB1.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gB2.data(), static_cast<int>(gb.gB2.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gLn1Gamma.data(), static_cast<int>(gb.gLn1Gamma.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gLn1Beta.data(), static_cast<int>(gb.gLn1Beta.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gLn2Gamma.data(), static_cast<int>(gb.gLn2Gamma.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gLn2Beta.data(), static_cast<int>(gb.gLn2Beta.size()), gpuTransformerScratch->lossSum.data());
				}
				// Final LayerNorm gradients
				gpu::sum_squared_accumulate(gpuTransformerWeights->gLnFinalGamma.data(),
				    static_cast<int>(gpuTransformerWeights->gLnFinalGamma.size()),
				    gpuTransformerScratch->lossSum.data());
				gpu::sum_squared_accumulate(gpuTransformerWeights->gLnFinalBeta.data(),
				    static_cast<int>(gpuTransformerWeights->gLnFinalBeta.size()),
				    gpuTransformerScratch->lossSum.data());

				float h_sumSq = 0.0f;
				gpu::recordEvent(gpuComputeReadyEvent, gpu::computeStream());
				gpu::streamWaitEvent(gpu::transferStream(), gpuComputeReadyEvent);
				gpuTransformerScratch->lossSum.downloadAsync(&h_sumSq, 1);
				gpu::synchronizeTransferStream();
				const float gradNorm = sqrtf(h_sumSq) * invBatch;
				if (gradNorm > clipNorm)
					gradScale = clipNorm / (gradNorm + 1e-12f);
				lastGradNorm = gradNorm;
				lastGradNormScale = gradScale;
			}

			const bool gpuUseAtlas = (trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS);

			if (gpuUseAtlas)
			{
			// === GPU ATLAS optimizer ===
			// Weight matrices use atlas_gpu_step (BRSP subspace preconditioning).
			// Biases and LN params use vanilla SGD (matching CPU ATLAS path).
			const glades::ATLASConfig& ac = trainingConfig.atlas;

			bool gpuAtlasError = false;

			// Macro: vanilla SGD for 1D bias/LN param on GPU.
			// W -= lr * invBatch * gradScale * g;  then zero g.
#define GLADES_GPU_SGD_BIAS(param, grad, lr_) do { \
	const int sgd_sz_ = static_cast<int>((param).size()); \
	if (sgd_sz_ > 0) { \
gpu::atlas_gpu_baseline_update((param).data(), (grad).data(), sgd_sz_, (lr_) * invBatch * gradScale); \
gpu::atlas_gpu_guard((param).data(), sgd_sz_); \
(grad).zero(); \
	} \
} while(0)

			// Token embedding (layer index 0)
			if (tokenLM)
			{
				const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_0 = skeleton->getWeightDecay1(0u);
				const float wd2_0 = skeleton->getWeightDecay2(0u);

				if (!gpuAtlasError && !gpu::atlas_gpu_update(gpuTransformerWeights->atlasTokE,
				    gpuTransformerWeights->tokE.data(), gpuTransformerWeights->gTokE.data(),
				    vocabSize, dModel, invBatch, lr0, wd1_0, wd2_0, gradScale,
				    ac, rngEngine, getLogger(), "tr.tokE"))
				    gpuAtlasError = true;

				GLADES_GPU_SGD_BIAS(gpuTransformerWeights->lmBias, gpuTransformerWeights->gLmBias, lr0);
			}

			// Input projection (layer index 0, not used for token-LM models)
			if (!tokenLM)
			{
				const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_0 = skeleton->getWeightDecay1(0u);
				const float wd2_0 = skeleton->getWeightDecay2(0u);

				if (!gpuAtlasError && !gpu::atlas_gpu_update(gpuTransformerWeights->atlasWIn,
				    gpuTransformerWeights->WIn.data(), gpuTransformerWeights->gWIn.data(),
				    dModel, inputSize, invBatch, lr0, wd1_0, wd2_0, gradScale,
				    ac, rngEngine, getLogger(), "tr.WIn"))
				    gpuAtlasError = true;

				GLADES_GPU_SGD_BIAS(gpuTransformerWeights->bIn, gpuTransformerWeights->gBIn, lr0);
			}

			// Per-layer blocks (layer index 1..nLayers)
			for (unsigned int bli = 0; bli < nLayers; ++bli)
			{
				const float lr_l = skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_l = skeleton->getWeightDecay1(bli + 1u);
				const float wd2_l = skeleton->getWeightDecay2(bli + 1u);
				gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];

				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasWq, gb.Wq.data(), gb.gWq.data(), dModel, dModel, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.Wq"))
				    gpuAtlasError = true;
				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasWk, gb.Wk.data(), gb.gWk.data(), dModelKV, dModel, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.Wk"))
				    gpuAtlasError = true;
				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasWv, gb.Wv.data(), gb.gWv.data(), dModelKV, dModel, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.Wv"))
				    gpuAtlasError = true;
				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasWo, gb.Wo.data(), gb.gWo.data(), dModel, dModel, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.Wo"))
				    gpuAtlasError = true;
				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasW1, gb.W1.data(), gb.gW1.data(), ff1Width, dModel, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.W1"))
				    gpuAtlasError = true;
				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasW2, gb.W2.data(), gb.gW2.data(), dModel, dFF, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.W2"))
				    gpuAtlasError = true;

				// Biases and LN params: vanilla SGD (no subspace projection)
				GLADES_GPU_SGD_BIAS(gb.bq, gb.gBq, lr_l);
				GLADES_GPU_SGD_BIAS(gb.bk, gb.gBk, lr_l);
				GLADES_GPU_SGD_BIAS(gb.bv, gb.gBv, lr_l);
				GLADES_GPU_SGD_BIAS(gb.bo, gb.gBo, lr_l);
				GLADES_GPU_SGD_BIAS(gb.b1, gb.gB1, lr_l);
				GLADES_GPU_SGD_BIAS(gb.b2, gb.gB2, lr_l);
				GLADES_GPU_SGD_BIAS(gb.ln1Gamma, gb.gLn1Gamma, lr_l);
				GLADES_GPU_SGD_BIAS(gb.ln1Beta, gb.gLn1Beta, lr_l);
				GLADES_GPU_SGD_BIAS(gb.ln2Gamma, gb.gLn2Gamma, lr_l);
				GLADES_GPU_SGD_BIAS(gb.ln2Beta, gb.gLn2Beta, lr_l);
			}

			// Final LayerNorm (use block 0 LR; no weight decay)
			{
				const float lrLN = skeleton->getLearningRate(1u) * lrScheduleMultiplier * gpuExtraLRMult;
				GLADES_GPU_SGD_BIAS(gpuTransformerWeights->lnFinalGamma, gpuTransformerWeights->gLnFinalGamma, lrLN);
				GLADES_GPU_SGD_BIAS(gpuTransformerWeights->lnFinalBeta, gpuTransformerWeights->gLnFinalBeta, lrLN);
			}

			// Output projection (layer index nLayers, unused in tied-head mode)
			if (!tokenLM)
			{
				const float lrO = skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_o = skeleton->getWeightDecay1(nLayers);
				const float wd2_o = skeleton->getWeightDecay2(nLayers);

				if (!gpuAtlasError && !gpu::atlas_gpu_update(gpuTransformerWeights->atlasWOut,
				    gpuTransformerWeights->WOut.data(), gpuTransformerWeights->gWOut.data(),
				    outSize, dModel, invBatch, lrO, wd1_o, wd2_o, gradScale,
				    ac, rngEngine, getLogger(), "tr.WOut"))
				    gpuAtlasError = true;

				GLADES_GPU_SGD_BIAS(gpuTransformerWeights->bOut, gpuTransformerWeights->gBOut, lrO);
			}

			// --- GPU ATLAS periodic diagnostics ---
			if (!gpuAtlasError && logger && ac.tSub > 0u)
			{
				const unsigned long long tSubULL = static_cast<unsigned long long>(ac.tSub);

				// Helper lambda-like macro to log a single weight state
#define GLADES_GPU_ATLAS_DIAG(st, tag_str) do { \
	if ((st).initialized && ((st).step % tSubULL) == 0ULL) { \
gpu::AtlasGpuDiag ad_ = gpu::atlas_gpu_get_diag((st)); \
if (ad_.valid) { \
	std::ostringstream oss_; \
	oss_ << "event=gpu_atlas_step tag=" << (tag_str); \
	oss_ << " step=" << ad_.step; \
	oss_ << " m=" << (st).m << " n=" << (st).n << " rank=" << (st).r; \
	oss_ << " mu=" << ad_.mu; \
	oss_ << " sigma2=" << ad_.sigma2; \
	oss_ << " baseline_rate=" << ad_.baselineRate; \
	oss_ << " gz_norm=" << ad_.gzNorm; \
	oss_ << " update_norm=" << ad_.updateNorm; \
	oss_ << " fisher_min=" << ad_.fisherMin; \
	oss_ << " fisher_max=" << ad_.fisherMax; \
	oss_ << " fisher_mean=" << ad_.fisherMean; \
	logger->info("ATLAS", shmea::GString(oss_.str().c_str())); \
} \
	} \
} while(0)

				if (tokenLM)
					GLADES_GPU_ATLAS_DIAG(gpuTransformerWeights->atlasTokE, "tr.tokE");
				if (!tokenLM)
					GLADES_GPU_ATLAS_DIAG(gpuTransformerWeights->atlasWIn, "tr.WIn");

				for (unsigned int bli = 0; bli < nLayers; ++bli)
				{
					gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
					GLADES_GPU_ATLAS_DIAG(gb.atlasWq, "tr.Wq");
					GLADES_GPU_ATLAS_DIAG(gb.atlasWk, "tr.Wk");
					GLADES_GPU_ATLAS_DIAG(gb.atlasWv, "tr.Wv");
					GLADES_GPU_ATLAS_DIAG(gb.atlasWo, "tr.Wo");
					GLADES_GPU_ATLAS_DIAG(gb.atlasW1, "tr.W1");
					GLADES_GPU_ATLAS_DIAG(gb.atlasW2, "tr.W2");
				}

				if (!tokenLM)
					GLADES_GPU_ATLAS_DIAG(gpuTransformerWeights->atlasWOut, "tr.WOut");

#undef GLADES_GPU_ATLAS_DIAG
			}

			if (gpuAtlasError)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
				    "SGDHelper_TRANSFORMER: GPU ATLAS update failed");
				storeRunningFlag(false);
			}
#undef GLADES_GPU_SGD_BIAS
			}
			else
			{
			// === Batched Adam optimizer ===
			const float beta1 = trainingConfig.optimizer.adamBeta1;
			const float beta2 = trainingConfig.optimizer.adamBeta2;
			const float adamEps = trainingConfig.optimizer.adamEps;
			const int stepInt = static_cast<int>(tensorTransformer.optimizerStep);

			// Build device pointer arrays on first step (pointers are fixed after GPU alloc).
			if (!gpuTransformerWeights->adamPtrsUploaded)
			{
				float* hParams[6 + 16 * 256];
				float* hGrads[6 + 16 * 256];
				float* hMs[6 + 16 * 256];
				float* hVs[6 + 16 * 256];
				int hSizes[6 + 16 * 256];
				int gc = 0;
				int maxSz = 0;

#define GLADES_ADD_ADAM_GROUP(p, g, m, v, sz) do { \
	if ((sz) > 0) { \
hParams[gc] = (p); hGrads[gc] = (g); \
hMs[gc] = (m); hVs[gc] = (v); \
hSizes[gc] = (sz); \
if ((sz) > maxSz) maxSz = (sz); \
++gc; \
	} \
} while(0)

				if (tokenLM)
				{
					GLADES_ADD_ADAM_GROUP(gpuTransformerWeights->tokE.data(),
					    gpuTransformerWeights->gTokE.data(),
					    gpuTransformerWeights->vTokE.data(),
					    gpuTransformerWeights->v2TokE.data(),
					    static_cast<int>(static_cast<size_t>(vocabSize) * dModel));
					GLADES_ADD_ADAM_GROUP(gpuTransformerWeights->lmBias.data(),
					    gpuTransformerWeights->gLmBias.data(),
					    gpuTransformerWeights->mLmBias.data(),
					    gpuTransformerWeights->v2LmBias.data(),
					    static_cast<int>(vocabSize));
				}
				{
					GLADES_ADD_ADAM_GROUP(gpuTransformerWeights->WIn.data(),
					    gpuTransformerWeights->gWIn.data(),
					    gpuTransformerWeights->vWIn.data(),
					    gpuTransformerWeights->v2WIn.data(),
					    static_cast<int>(gpuTransformerWeights->WIn.size()));
					GLADES_ADD_ADAM_GROUP(gpuTransformerWeights->bIn.data(),
					    gpuTransformerWeights->gBIn.data(),
					    gpuTransformerWeights->mBIn.data(),
					    gpuTransformerWeights->v2BIn.data(),
					    static_cast<int>(gpuTransformerWeights->bIn.size()));
				}
				for (unsigned int bli = 0; bli < nLayers; ++bli)
				{
					gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
					GLADES_ADD_ADAM_GROUP(gb.Wq.data(), gb.gWq.data(), gb.vWq.data(), gb.v2Wq.data(), static_cast<int>(gb.Wq.size()));
					GLADES_ADD_ADAM_GROUP(gb.Wk.data(), gb.gWk.data(), gb.vWk.data(), gb.v2Wk.data(), static_cast<int>(gb.Wk.size()));
					GLADES_ADD_ADAM_GROUP(gb.Wv.data(), gb.gWv.data(), gb.vWv.data(), gb.v2Wv.data(), static_cast<int>(gb.Wv.size()));
					GLADES_ADD_ADAM_GROUP(gb.Wo.data(), gb.gWo.data(), gb.vWo.data(), gb.v2Wo.data(), static_cast<int>(gb.Wo.size()));
					GLADES_ADD_ADAM_GROUP(gb.W1.data(), gb.gW1.data(), gb.vW1.data(), gb.v2W1.data(), static_cast<int>(gb.W1.size()));
					GLADES_ADD_ADAM_GROUP(gb.W2.data(), gb.gW2.data(), gb.vW2.data(), gb.v2W2.data(), static_cast<int>(gb.W2.size()));
					GLADES_ADD_ADAM_GROUP(gb.bq.data(), gb.gBq.data(), gb.mBq.data(), gb.v2Bq.data(), static_cast<int>(gb.bq.size()));
					GLADES_ADD_ADAM_GROUP(gb.bk.data(), gb.gBk.data(), gb.mBk.data(), gb.v2Bk.data(), static_cast<int>(gb.bk.size()));
					GLADES_ADD_ADAM_GROUP(gb.bv.data(), gb.gBv.data(), gb.mBv.data(), gb.v2Bv.data(), static_cast<int>(gb.bv.size()));
					GLADES_ADD_ADAM_GROUP(gb.bo.data(), gb.gBo.data(), gb.mBo.data(), gb.v2Bo.data(), static_cast<int>(gb.bo.size()));
					GLADES_ADD_ADAM_GROUP(gb.b1.data(), gb.gB1.data(), gb.mB1.data(), gb.v2B1.data(), static_cast<int>(gb.b1.size()));
					GLADES_ADD_ADAM_GROUP(gb.b2.data(), gb.gB2.data(), gb.mB2.data(), gb.v2B2.data(), static_cast<int>(gb.b2.size()));
					GLADES_ADD_ADAM_GROUP(gb.ln1Gamma.data(), gb.gLn1Gamma.data(), gb.mLn1Gamma.data(), gb.v2Ln1Gamma.data(), static_cast<int>(gb.ln1Gamma.size()));
					GLADES_ADD_ADAM_GROUP(gb.ln1Beta.data(), gb.gLn1Beta.data(), gb.mLn1Beta.data(), gb.v2Ln1Beta.data(), static_cast<int>(gb.ln1Beta.size()));
					GLADES_ADD_ADAM_GROUP(gb.ln2Gamma.data(), gb.gLn2Gamma.data(), gb.mLn2Gamma.data(), gb.v2Ln2Gamma.data(), static_cast<int>(gb.ln2Gamma.size()));
					GLADES_ADD_ADAM_GROUP(gb.ln2Beta.data(), gb.gLn2Beta.data(), gb.mLn2Beta.data(), gb.v2Ln2Beta.data(), static_cast<int>(gb.ln2Beta.size()));
				}
				// Final LayerNorm
				GLADES_ADD_ADAM_GROUP(gpuTransformerWeights->lnFinalGamma.data(),
				    gpuTransformerWeights->gLnFinalGamma.data(),
				    gpuTransformerWeights->mLnFinalGamma.data(),
				    gpuTransformerWeights->v2LnFinalGamma.data(),
				    static_cast<int>(gpuTransformerWeights->lnFinalGamma.size()));
				GLADES_ADD_ADAM_GROUP(gpuTransformerWeights->lnFinalBeta.data(),
				    gpuTransformerWeights->gLnFinalBeta.data(),
				    gpuTransformerWeights->mLnFinalBeta.data(),
				    gpuTransformerWeights->v2LnFinalBeta.data(),
				    static_cast<int>(gpuTransformerWeights->lnFinalBeta.size()));
				if (!tokenLM)
				{
					GLADES_ADD_ADAM_GROUP(gpuTransformerWeights->WOut.data(),
					    gpuTransformerWeights->gWOut.data(),
					    gpuTransformerWeights->vWOut.data(),
					    gpuTransformerWeights->v2WOut.data(),
					    static_cast<int>(gpuTransformerWeights->WOut.size()));
					GLADES_ADD_ADAM_GROUP(gpuTransformerWeights->bOut.data(),
					    gpuTransformerWeights->gBOut.data(),
					    gpuTransformerWeights->mBOut.data(),
					    gpuTransformerWeights->v2BOut.data(),
					    static_cast<int>(gpuTransformerWeights->bOut.size()));
				}
#undef GLADES_ADD_ADAM_GROUP

				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamParams, hParams, gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamGrads, hGrads, gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamM, hMs, gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamV, hVs, gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamSizes, hSizes, gc * sizeof(int));
				gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
				gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);
				gpuTransformerWeights->adamGroupCount = gc;
				gpuTransformerWeights->adamMaxSize = maxSz;
				gpuTransformerWeights->adamPtrsUploaded = true;
			}

			// Fill lr/wd arrays each step and launch single batched kernel.
			{
				const int gc = gpuTransformerWeights->adamGroupCount;
				float hLrs[6 + 16 * 256];
				float hWds[6 + 16 * 256];
				int gi = 0;

				if (tokenLM)
				{
					const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
					const float wd0 = skeleton->getWeightDecay2(0u);
					hLrs[gi] = lr0; hWds[gi] = wd0; ++gi;
					hLrs[gi] = lr0; hWds[gi] = 0.0f; ++gi;
				}
				{
					const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
					const float wd0 = skeleton->getWeightDecay2(0u);
					if (gpuTransformerWeights->WIn.size() > 0)
					{ hLrs[gi] = lr0; hWds[gi] = wd0; ++gi; }
					if (gpuTransformerWeights->bIn.size() > 0)
					{ hLrs[gi] = lr0; hWds[gi] = 0.0f; ++gi; }
				}
				for (unsigned int bli = 0; bli < nLayers; ++bli)
				{
					const float lr_l = skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
					const float wd_l = skeleton->getWeightDecay2(bli + 1u);
					// 6 weight groups (with wd)
					for (int w = 0; w < 6; ++w)
					{ hLrs[gi] = lr_l; hWds[gi] = wd_l; ++gi; }
					// 10 bias/LN groups (no wd)
					for (int b = 0; b < 10; ++b)
					{ hLrs[gi] = lr_l; hWds[gi] = 0.0f; ++gi; }
				}
				// Final LayerNorm (use block 0 LR; no weight decay)
				{
					const float lrLN = skeleton->getLearningRate(1u) * lrScheduleMultiplier * gpuExtraLRMult;
					hLrs[gi] = lrLN; hWds[gi] = 0.0f; ++gi; // lnFinalGamma
					hLrs[gi] = lrLN; hWds[gi] = 0.0f; ++gi; // lnFinalBeta
				}
				if (!tokenLM)
				{
					const float lrO = skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
					const float wdO = skeleton->getWeightDecay2(nLayers);
					if (gpuTransformerWeights->WOut.size() > 0)
					{ hLrs[gi] = lrO; hWds[gi] = wdO; ++gi; }
					if (gpuTransformerWeights->bOut.size() > 0)
					{ hLrs[gi] = lrO; hWds[gi] = 0.0f; ++gi; }
				}

				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamLr, hLrs, gc * sizeof(float));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamWd, hWds, gc * sizeof(float));
				gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
				gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);

				gpu::adam_update_batch(
				    gpuTransformerWeights->d_adamParams,
				    gpuTransformerWeights->d_adamGrads,
				    gpuTransformerWeights->d_adamM,
				    gpuTransformerWeights->d_adamV,
				    gpuTransformerWeights->d_adamLr,
				    gpuTransformerWeights->d_adamWd,
				    gpuTransformerWeights->d_adamSizes,
				    gpuTransformerWeights->adamMaxSize,
				    beta1, beta2, adamEps,
				    invBatch * gradScale, stepInt, gc);
			}
			} // end Adam branch

			gpu::synchronizeComputeStream();
			seqInBatch = 0u;
			timeStepsInBatch = 0u;
		}
	}

	gpu::destroyEvent(gpuTransferReadyEvent);
	gpu::destroyEvent(gpuComputeReadyEvent);

	// After GPU training loop: download updated weights back to CPU.
	TensorTransformerState& ttMut = tensorTransformer;
	gpu::downloadTransformerWeights(*gpuTransformerWeights,
	                                ttMut.tokE.empty() ? NULL : &ttMut.tokE[0], ttMut.tokE.size(),
	                                ttMut.WIn.empty() ? NULL : &ttMut.WIn[0], ttMut.WIn.size(),
	                                ttMut.bIn.empty() ? NULL : &ttMut.bIn[0], ttMut.bIn.size(),
	                                ttMut.WOut.empty() ? NULL : &ttMut.WOut[0], ttMut.WOut.size(),
	                                ttMut.bOut.empty() ? NULL : &ttMut.bOut[0], ttMut.bOut.size(),
	                                ttMut.lmBias.empty() ? NULL : &ttMut.lmBias[0], ttMut.lmBias.size(),
	                                ttMut.lnFinalGamma.empty() ? NULL : &ttMut.lnFinalGamma[0], ttMut.lnFinalGamma.size(),
	                                ttMut.lnFinalBeta.empty() ? NULL : &ttMut.lnFinalBeta[0], ttMut.lnFinalBeta.size());

	// Download per-block weights.
	for (unsigned int l = 0; l < nLayers; ++l)
	{
		TensorTransformerState::Block& cb = ttMut.blocks[l];
		const gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[l];
		if (gb.Wq.allocated()) gb.Wq.download(&cb.Wq[0], cb.Wq.size());
		if (gb.Wk.allocated()) gb.Wk.download(&cb.Wk[0], cb.Wk.size());
		if (gb.Wv.allocated()) gb.Wv.download(&cb.Wv[0], cb.Wv.size());
		if (gb.Wo.allocated()) gb.Wo.download(&cb.Wo[0], cb.Wo.size());
		if (gb.W1.allocated()) gb.W1.download(&cb.W1[0], cb.W1.size());
		if (gb.W2.allocated()) gb.W2.download(&cb.W2[0], cb.W2.size());
		if (gb.bq.allocated()) gb.bq.download(&cb.bq[0], cb.bq.size());
		if (gb.bk.allocated()) gb.bk.download(&cb.bk[0], cb.bk.size());
		if (gb.bv.allocated()) gb.bv.download(&cb.bv[0], cb.bv.size());
		if (gb.bo.allocated()) gb.bo.download(&cb.bo[0], cb.bo.size());
		if (gb.b1.allocated()) gb.b1.download(&cb.b1[0], cb.b1.size());
		if (gb.b2.allocated()) gb.b2.download(&cb.b2[0], cb.b2.size());
		if (gb.ln1Gamma.allocated()) gb.ln1Gamma.download(&cb.ln1Gamma[0], cb.ln1Gamma.size());
		if (gb.ln1Beta.allocated()) gb.ln1Beta.download(&cb.ln1Beta[0], cb.ln1Beta.size());
		if (gb.ln2Gamma.allocated()) gb.ln2Gamma.download(&cb.ln2Gamma[0], cb.ln2Gamma.size());
		if (gb.ln2Beta.allocated()) gb.ln2Beta.download(&cb.ln2Beta[0], cb.ln2Beta.size());
	}

	// Finalize epoch-level loss before returning.
	// tokenLmNllSum / tokenLmTokenCount are locals; copy to the member
	// that the Trainer reads (overallTotalError).
	if (tokenLM)
	{
		if (tokenLmTokenCount > 0ULL)
			overallTotalError = static_cast<float>(tokenLmNllSum / static_cast<double>(tokenLmTokenCount));
		else
			overallTotalError = 0.0f;
	}

}
#endif // GLADES_HAVE_CUDA
