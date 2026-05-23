#include "transformer_config.h"

namespace {

static glades::NNetworkStatus invalid_argument(const char* where, const char* msg)
{
	return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT,
	                              std::string(where ? where : "transformer_config") + ": " + msg);
}

static glades::NNetworkStatus invalid_state(const char* where, const char* msg)
{
	return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
	                              std::string(where ? where : "transformer_config") + ": " + msg);
}

static glades::NNetworkStatus validateTransformerRuntimeConfig(const char* where,
                                                               const glades::TransformerRunConfig& runtimeCfg)
{
	const int posEnc = static_cast<int>(runtimeCfg.positionalEncoding);
	if (posEnc != static_cast<int>(glades::TransformerRunConfig::POSENC_NONE) &&
	    posEnc != static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL) &&
	    posEnc != static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE))
		return invalid_argument(where, "unknown positionalEncoding");

	const int normType = static_cast<int>(runtimeCfg.normType);
	if (normType != static_cast<int>(glades::TransformerRunConfig::NORM_LAYERNORM) &&
	    normType != static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
		return invalid_argument(where, "unknown normType");

	const int ffnKind = static_cast<int>(runtimeCfg.ffnKind);
	if (ffnKind != static_cast<int>(glades::TransformerRunConfig::FFN_MLP) &&
	    ffnKind != static_cast<int>(glades::TransformerRunConfig::FFN_SWIGLU))
		return invalid_argument(where, "unknown ffnKind");

	const int ffnActivation = static_cast<int>(runtimeCfg.ffnActivation);
	if (ffnActivation != static_cast<int>(glades::TransformerRunConfig::FFN_RELU) &&
	    ffnActivation != static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
		return invalid_argument(where, "unknown ffnActivation");

	const int kvCacheDType = static_cast<int>(runtimeCfg.kvCacheDType);
	if (kvCacheDType != static_cast<int>(glades::TransformerRunConfig::KV_CACHE_F32) &&
	    kvCacheDType != static_cast<int>(glades::TransformerRunConfig::KV_CACHE_F16) &&
	    kvCacheDType != static_cast<int>(glades::TransformerRunConfig::KV_CACHE_BF16))
		return invalid_argument(where, "unknown kvCacheDType");

	if (runtimeCfg.embeddingDropoutRate < 0.0f || runtimeCfg.embeddingDropoutRate >= 1.0f)
		return invalid_argument(where, "embeddingDropoutRate must be in [0,1)");
	if (runtimeCfg.residualDropoutRate < 0.0f || runtimeCfg.residualDropoutRate >= 1.0f)
		return invalid_argument(where, "residualDropoutRate must be in [0,1)");

	if (runtimeCfg.layerDropPMax < 0.0f || runtimeCfg.layerDropPMax >= 1.0f)
		return invalid_argument(where, "layerDropPMax must be in [0,1)");

	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

static glades::NNetworkStatus validateTransformerTokenLmConfig(const char* where,
                                                               const glades::TrainingConfig& cfg)
{
	if (cfg.transformer.tokenLmLossKind != glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX &&
	    cfg.transformer.tokenLmLossKind != glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX)
		return invalid_argument(where, "unknown tokenLmLossKind");

	if (cfg.transformer.tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX &&
	    cfg.transformer.tokenLmSampledNegatives < 1)
		return invalid_argument(where, "tokenLmSampledNegatives must be >= 1 for sampled softmax");

	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

} // namespace

glades::NNetworkStatus glades::buildTransformerRuntimeConfigSnapshot(const char* where,
                                                                     const TransformerRunConfig& runtimeCfg,
                                                                     TransformerRuntimeConfigSnapshot& out)
{
	NNetworkStatus st = validateTransformerRuntimeConfig(where, runtimeCfg);
	if (!st.ok())
		return st;

	out.tokenModel = runtimeCfg.enableTokenEmbedding;
	out.layerNormEps = (runtimeCfg.layerNormEps > 0.0f ? runtimeCfg.layerNormEps : 1e-5f);
	out.normType = static_cast<unsigned int>(runtimeCfg.normType);
	out.positionalEncoding = static_cast<unsigned int>(runtimeCfg.positionalEncoding);
	out.kvCacheDType = static_cast<unsigned int>(runtimeCfg.kvCacheDType);
	out.ropeDimOverride = runtimeCfg.ropeDimOverride;
	out.ropeTheta = (runtimeCfg.ropeTheta > 0.0f ? runtimeCfg.ropeTheta : 10000.0f);
	out.ffnKind = static_cast<unsigned int>(runtimeCfg.ffnKind);
	out.ffnActivation = static_cast<unsigned int>(runtimeCfg.ffnActivation);
	out.padTokenId = runtimeCfg.padTokenId;
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::validateTransformerTrainingConfig(const char* where,
                                                                 const TrainingConfig& cfg)
{
	NNetworkStatus st = validateTransformerRuntimeConfig(where, cfg.transformer);
	if (!st.ok())
		return st;
	st = validateTransformerTokenLmConfig(where, cfg);
	if (!st.ok())
		return st;

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::buildTransformerModelConfigSnapshot(const char* where,
                                                                   const TrainingConfig& cfg,
                                                                   const std::vector<unsigned int>& hiddenSizes,
                                                                   unsigned int outSize,
                                                                   bool tokenIdInput,
                                                                   bool causal,
                                                                   TransformerModelConfigSnapshot& out)
{
	NNetworkStatus st = validateTransformerTrainingConfig(where, cfg);
	if (!st.ok())
		return st;
	st = buildTransformerRuntimeConfigSnapshot(where, cfg.transformer, out);
	if (!st.ok())
		return st;

	if (hiddenSizes.empty())
		return invalid_state(where, "transformer requires >= 1 hidden layer (blocks)");

	out.dModel = hiddenSizes[0];
	out.nLayers = static_cast<unsigned int>(hiddenSizes.size());
	out.nHeads = (cfg.transformer.nHeadsOverride > 0) ? static_cast<unsigned int>(cfg.transformer.nHeadsOverride) : 4u;
	out.nKVHeads = (cfg.transformer.nKVHeadsOverride > 0) ? static_cast<unsigned int>(cfg.transformer.nKVHeadsOverride) : out.nHeads;
	out.dFF = (cfg.transformer.dFFOverride > 0) ? static_cast<unsigned int>(cfg.transformer.dFFOverride)
	                                            : (4u * out.dModel);
	out.ff1Width = (out.ffnKind == static_cast<unsigned int>(TransformerRunConfig::FFN_SWIGLU)) ? (2u * out.dFF) : out.dFF;
	out.vocabSize = (cfg.transformer.vocabSizeOverride > 0) ? static_cast<unsigned int>(cfg.transformer.vocabSizeOverride) : outSize;
	out.tieEmbeddings = cfg.transformer.tieEmbeddings;
	out.causal = causal;

	if (out.dModel == 0u || out.nHeads == 0u || out.nKVHeads == 0u || out.dFF == 0u || out.ff1Width == 0u)
		return invalid_state(where, "invalid transformer config (dModel/heads/dFF)");
	if ((out.dModel % out.nHeads) != 0u)
		return invalid_argument(where, "transformer dModel must be divisible by nHeads");
	if ((out.nHeads % out.nKVHeads) != 0u)
		return invalid_argument(where, "transformer nKVHeads must divide nHeads");

	for (size_t i = 0; i < hiddenSizes.size(); ++i)
	{
		if (hiddenSizes[i] == 0u || hiddenSizes[i] != out.dModel)
			return invalid_argument(where, "transformer requires constant hidden size (dModel) across all blocks");
	}

	if (out.tokenModel)
	{
		if (!out.tieEmbeddings)
			return invalid_argument(where, "token LM mode currently requires tieEmbeddings=true");
		if (!tokenIdInput)
			return invalid_argument(where, "token LM mode requires DataInput token-id accessors");
		if (out.vocabSize == 0u)
			return invalid_argument(where, "token LM mode requires vocabSize > 0");
		if (cfg.transformer.vocabSizeOverride > 0 && outSize != out.vocabSize)
			return invalid_argument(where, "token LM vocabSizeOverride must match NNInfo output layer size");
	}

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}
