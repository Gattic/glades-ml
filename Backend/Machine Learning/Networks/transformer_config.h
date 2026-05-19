#pragma once

#include "training_config.h"
#include "../nnetwork_status.h"

#include <vector>

namespace glades {

struct TransformerRuntimeConfigSnapshot
{
	bool tokenModel;
	float layerNormEps;
	unsigned int normType;
	unsigned int positionalEncoding;
	unsigned int kvCacheDType;
	int ropeDimOverride;
	float ropeTheta;
	unsigned int ffnKind;
	unsigned int ffnActivation;
	int padTokenId;

	TransformerRuntimeConfigSnapshot()
	    : tokenModel(false),
	      layerNormEps(1e-5f),
	      normType(0u),
	      positionalEncoding(0u),
	      kvCacheDType(0u),
	      ropeDimOverride(0),
	      ropeTheta(10000.0f),
	      ffnKind(0u),
	      ffnActivation(0u),
	      padTokenId(-1)
	{
	}
};

struct TransformerModelConfigSnapshot : public TransformerRuntimeConfigSnapshot
{
	unsigned int dModel;
	unsigned int dFF;
	unsigned int nHeads;
	unsigned int nKVHeads;
	unsigned int nLayers;
	unsigned int vocabSize;
	unsigned int ff1Width;
	bool tieEmbeddings;
	bool causal;

	TransformerModelConfigSnapshot()
	    : TransformerRuntimeConfigSnapshot(),
	      dModel(0u),
	      dFF(0u),
	      nHeads(0u),
	      nKVHeads(0u),
	      nLayers(0u),
	      vocabSize(0u),
	      ff1Width(0u),
	      tieEmbeddings(true),
	      causal(false)
	{
	}
};

NNetworkStatus buildTransformerRuntimeConfigSnapshot(const char* where,
                                                     const TransformerRunConfig& runtimeCfg,
                                                     TransformerRuntimeConfigSnapshot& out);

NNetworkStatus validateTransformerTrainingConfig(const char* where,
                                                 const TrainingConfig& cfg);

NNetworkStatus buildTransformerModelConfigSnapshot(const char* where,
                                                   const TrainingConfig& cfg,
                                                   const std::vector<unsigned int>& hiddenSizes,
                                                   unsigned int outSize,
                                                   bool tokenIdInput,
                                                   bool causal,
                                                   TransformerModelConfigSnapshot& out);

} // namespace glades
