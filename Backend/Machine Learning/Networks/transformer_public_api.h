// Stable transformer-facing API surface (C++98-friendly).
//
// This header intentionally exposes a small, const-correct wrapper around the
// transformer inference + serving entry points, without requiring callers to
// understand internal tensor layouts.
//
#pragma once

#include "../nnetwork_status.h"
#include "transformer_types.h"

#include <vector>

namespace glades {

class NNetwork;

// Token identifiers are first-class integers throughout transformer APIs.
// (They must be < vocabSize; negative IDs are reserved for internal sentinel use.)
typedef unsigned int TokenId;
typedef int TokenLabelId; // may be negative for padding/ignore depending on config

struct TransformerPublicAPI
{
	// Bound runtime view for callers that want a transformer-specific API boundary
	// without reaching through `NNetwork` directly.
	struct Runtime
	{
		explicit Runtime(const NNetwork& network) : net(network) {}

		NNetworkStatus generate(const std::vector<TokenId>& promptTokens,
		                        const TransformerGenerateConfig& cfg,
		                        TransformerGenerateResult& out,
		                        ITransformerGenerateCallbacks* cb /* optional */) const;

		NNetworkStatus generateBatch(const std::vector<TransformerServeRequest>& requests,
		                             TransformerServeBatchResult& out,
		                             ITransformerServeCallbacks* cb /* optional */) const;

		NNetworkStatus forwardLastLogits(const std::vector<TokenId>& tokenIds,
		                                 std::vector<float>& outLogits) const;

	private:
		const NNetwork& net;
	};

	struct ServingRuntime
	{
		typedef NNetwork::TransformerServeBatcherConfig BatcherConfig;
		typedef NNetwork::TransformerServeBatcher Batcher;

		explicit ServingRuntime(const NNetwork& network) : net(network) {}

		NNetworkStatus resetBatcher(Batcher& batcher, const BatcherConfig& cfg) const;
		NNetworkStatus submit(Batcher& batcher, const TransformerServeRequest& request, unsigned int& outSlot) const;
		NNetworkStatus remove(Batcher& batcher, unsigned int slot) const;
		NNetworkStatus step(Batcher& batcher, ITransformerServeCallbacks* cb /* optional */) const;
		NNetworkStatus cancelSlot(Batcher& batcher, unsigned int slot) const;

	private:
		const NNetwork& net;
	};

	// Preferred entrypoint: bind a transformer-specific runtime facade to a network.
	static Runtime runtime(const NNetwork& net);
	static ServingRuntime serving(const NNetwork& net);

	// Single-request generation (KV-cache incremental decode).
	static NNetworkStatus generate(const NNetwork& net,
	                               const std::vector<TokenId>& promptTokens,
	                               const TransformerGenerateConfig& cfg,
	                               TransformerGenerateResult& out,
	                               ITransformerGenerateCallbacks* cb /* optional */);

	// Batched generation (one-shot, ragged prompts).
	static NNetworkStatus generateBatch(const NNetwork& net,
	                                    const std::vector<TransformerServeRequest>& requests,
	                                    TransformerServeBatchResult& out,
	                                    ITransformerServeCallbacks* cb /* optional */);

	// Full forward last-logits (debug/test parity helper).
	static NNetworkStatus forwardLastLogits(const NNetwork& net, const std::vector<TokenId>& tokenIds, std::vector<float>& outLogits);
};

} // namespace glades
