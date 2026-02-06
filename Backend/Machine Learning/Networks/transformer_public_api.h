// Stable transformer-facing API surface (C++98-friendly).
//
// This header intentionally exposes a small, const-correct wrapper around the
// transformer inference + serving entry points, without requiring callers to
// understand internal tensor layouts.
//
#pragma once

#include "network.h"

namespace glades {

// Token identifiers are first-class integers throughout transformer APIs.
// (They must be < vocabSize; negative IDs are reserved for internal sentinel use.)
typedef unsigned int TokenId;
typedef int TokenLabelId; // may be negative for padding/ignore depending on config

struct TransformerPublicAPI
{
	// Single-request generation (KV-cache incremental decode).
	static inline NNetworkStatus generate(const NNetwork& net,
	                                     const std::vector<TokenId>& promptTokens,
	                                     const NNetwork::TransformerGenerateConfig& cfg,
	                                     NNetwork::TransformerGenerateResult& out,
	                                     NNetwork::ITransformerGenerateCallbacks* cb /* optional */)
	{
		return net.transformerLmGenerate(promptTokens, cfg, out, cb);
	}

	// Batched generation (one-shot, ragged prompts).
	static inline NNetworkStatus generateBatch(const NNetwork& net,
	                                          const std::vector<NNetwork::TransformerServeRequest>& requests,
	                                          NNetwork::TransformerServeBatchResult& out,
	                                          NNetwork::ITransformerServeCallbacks* cb /* optional */)
	{
		return net.transformerLmServeGenerateBatch(requests, out, cb);
	}

	// Full forward last-logits (debug/test parity helper).
	static inline NNetworkStatus forwardLastLogits(const NNetwork& net, const std::vector<TokenId>& tokenIds, std::vector<float>& outLogits)
	{
		return net.transformerLmForwardLastLogits(tokenIds, outLogits);
	}
};

} // namespace glades

