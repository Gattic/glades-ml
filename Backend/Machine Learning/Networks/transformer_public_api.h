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

		// GPU full-forward helper used by measurement/evaluation harnesses. It
		// executes the same sequence-level GPU path as training and downloads
		// only the final row of unnormalized logits.
		NNetworkStatus forwardLastLogitsGpu(const std::vector<TokenId>& tokenIds,
		                                    std::vector<float>& outLogits) const;

		// Runs one full causal sequence on GPU and reduces next-token NLL
		// and top-1 accuracy on device without downloading vocabulary logits.
		NNetworkStatus evaluateTokenMetricsGpu(const std::vector<TokenId>& tokenIds,
		                                       const std::vector<TokenLabelId>& targetIds,
		                                       TransformerTokenMetrics& outMetrics) const;

		// Diagnostic full-sequence feature/logit output from the canonical GPU
		// forward. Intended for bounded read-only measurement tooling.
		NNetworkStatus forwardFeaturesGpu(const std::vector<TokenId>& tokenIds,
		                                  TransformerFullSequenceFeatures& out) const;

		// Read-only host copy of the tied output matrix and bias.
		NNetworkStatus readoutParameters(TransformerReadoutParameters& out) const;

		// Diagnostic full-forward helper. Returns the same logits as
		// forwardLastLogits plus the last-position input/post-block hidden rows.
		NNetworkStatus forwardLastTrace(const std::vector<TokenId>& tokenIds,
		                                std::vector<float>& outLogits,
		                                TransformerForwardTrace& outTrace) const;

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

	// GPU sequence-level full forward; unavailable in non-CUDA builds.
	static NNetworkStatus forwardLastLogitsGpu(const NNetwork& net, const std::vector<TokenId>& tokenIds, std::vector<float>& outLogits);

	// Full-sequence GPU next-token metric reduction; unavailable in non-CUDA builds.
	static NNetworkStatus evaluateTokenMetricsGpu(const NNetwork& net,
	                                              const std::vector<TokenId>& tokenIds,
	                                              const std::vector<TokenLabelId>& targetIds,
	                                              TransformerTokenMetrics& outMetrics);

	// Diagnostic full-sequence final hidden rows plus logits on GPU.
	static NNetworkStatus forwardFeaturesGpu(const NNetwork& net,
	                                        const std::vector<TokenId>& tokenIds,
	                                        TransformerFullSequenceFeatures& out);

	static NNetworkStatus readoutParameters(const NNetwork& net,
	                                       TransformerReadoutParameters& out);

	// Diagnostic full-forward last-logits plus bounded last-position hidden trace.
	static NNetworkStatus forwardLastTrace(const NNetwork& net,
	                                       const std::vector<TokenId>& tokenIds,
	                                       std::vector<float>& outLogits,
	                                       TransformerForwardTrace& outTrace);
};

} // namespace glades
