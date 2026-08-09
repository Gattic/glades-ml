// Freestanding transformer API types (decoupled from NNetwork).
//
// These were previously nested inside NNetwork. They are now at namespace scope
// so consumers can include this lightweight header without pulling in all of network.h.
#pragma once

#include <vector>
#include <stdint.h>

namespace glades {

class NNetwork;

class ITransformerGenerateCallbacks
{
public:
	virtual ~ITransformerGenerateCallbacks() {}
	// Called after a token is emitted (and appended to the KV cache).
	// Return true to stop generation early.
	virtual bool onToken(const NNetwork& /*net*/, unsigned int /*tokenId*/, unsigned int /*generatedIndex*/) { return false; }
	// Polled once per step; return true to cancel generation.
	virtual bool shouldStop(const NNetwork& /*net*/) { return false; }
};

class ITransformerServeCallbacks
{
public:
	virtual ~ITransformerServeCallbacks() {}
	// Called after a token is emitted for a request.
	// Return true to stop that request early.
	virtual bool onToken(const NNetwork& /*net*/, unsigned int /*requestIndex*/, unsigned int /*tokenId*/, unsigned int /*generatedIndex*/) { return false; }
	// Polled once per global decode step; return true to cancel all requests.
	virtual bool shouldStopAll(const NNetwork& /*net*/) { return false; }
	// Polled before sampling for a request each step; return true to cancel that request.
	virtual bool shouldStopRequest(const NNetwork& /*net*/, unsigned int /*requestIndex*/) { return false; }
};

struct TransformerGenerateConfig
{
	// Maximum number of new tokens to generate (excluding the prompt).
	unsigned int maxNewTokens;
	// Total KV cache length cap. If 0, defaults to promptLen + maxNewTokens.
	// If provided, it must be >= promptLen + maxNewTokens.
	unsigned int maxSeqLen;

	// Sampling controls:
	// - temperature <= 0 => greedy (argmax)
	// - topK == 0 => disabled
	// - topP <= 0 or > 1 => disabled
	float temperature;
	unsigned int topK;
	float topP;
	// Nucleus (top-p) implementation policy:
	//
	// When topP < 1 and topK == 0, a "pure" nucleus implementation would need to:
	// - sort the full vocabulary by logit each step (O(V log V)), then
	// - take the smallest prefix whose cumulative probability >= topP.
	//
	// That can be prohibitively expensive for large vocabularies on CPU.
	//
	// Glades defaults to an explicit approximation:
	// - if topP < 1 and topK == 0, we first cap candidates to the top-K tokens where
	//   K = min(vocabSize, topPTopKCap), then apply top-p within those candidates.
	//
	// Set topPTopKCap to 0 to disable this approximation (full-vocab nucleus).
	unsigned int topPTopKCap;

	// Stop controls:
	// - eosTokenId < 0 => disabled
	// - if stopOnEos==true and eosTokenId is produced, generation stops after emitting it
	int eosTokenId;
	bool stopOnEos;

	// Output formatting:
	// - includePromptInOutput==true => out.tokens includes prompt first, then generated tokens
	// - otherwise out.tokens contains only generated tokens
	bool includePromptInOutput;

	// RNG control:
	// - rngSeedOverride!=0 => seed the per-call RNG with this value
	// - rngSeedOverride==0 => derive a deterministic seed from the network seed + prompt tokens
	//
	// IMPORTANT:
	// - Generation does not mutate or depend on the shared `NNetwork::rngEngine`.
	// - If you want stochastic variation across calls, you must supply different rngSeedOverride values.
	uint64_t rngSeedOverride;

	TransformerGenerateConfig()
	    : maxNewTokens(0u),
	      maxSeqLen(0u),
	      temperature(1.0f),
	      topK(0u),
	      topP(1.0f),
	      topPTopKCap(256u),
	      eosTokenId(-1),
	      stopOnEos(true),
	      includePromptInOutput(false),
	      rngSeedOverride(0ULL)
	{
	}
};

struct TransformerGenerateResult
{
	// Tokens returned (see includePromptInOutput).
	std::vector<unsigned int> tokens;
	// Why generation ended.
	bool stoppedOnEos;
	// Stopped because a non-EOS stop token was encountered (TransformerServeRequest::stopTokenIds).
	// This is distinct from stoppedOnEos so serving telemetry can distinguish these cases.
	bool stoppedByStopToken;
	bool stoppedByCallback;
	bool stoppedByLimit;
	// Last token emitted (undefined if no tokens were emitted).
	unsigned int lastToken;

	TransformerGenerateResult()
	    : tokens(),
	      stoppedOnEos(false),
	      stoppedByStopToken(false),
	      stoppedByCallback(false),
	      stoppedByLimit(false),
	      lastToken(0u)
	{
	}
};

// Aggregate next-token metrics from one full-sequence GPU forward.
// `nllSum` is the sum over valid targets; `tokenCount` is the denominator.
struct TransformerTokenMetrics
{
	double nllSum;
	unsigned long long tokenCount;
	unsigned long long correct;

	TransformerTokenMetrics()
	    : nllSum(0.0), tokenCount(0ULL), correct(0ULL)
	{
	}

	void clear()
	{
		nllSum = 0.0;
		tokenCount = 0ULL;
		correct = 0ULL;
	}
};

// Diagnostic-only last-position hidden trace from a full causal forward.
// `lastHidden` is stage-major: embedded/positioned input first, then one row
// after each Transformer block. It deliberately excludes attention matrices
// and full-sequence activations so callers cannot accidentally turn this into
// a high-memory serving path.
struct TransformerForwardTrace
{
	unsigned int hiddenSize;
	unsigned int layers;
	std::vector<float> lastHidden;

	TransformerForwardTrace()
	    : hiddenSize(0u), layers(0u), lastHidden()
	{
	}

	void clear()
	{
		hiddenSize = 0u;
		layers = 0u;
		lastHidden.clear();
	}
};

struct TransformerServeRequest
{
	std::vector<unsigned int> promptTokens;
	TransformerGenerateConfig cfg;
	// Optional additional stop tokens (besides eosTokenId).
	// If any token in stopTokenIds is generated, generation stops after emitting it.
	std::vector<unsigned int> stopTokenIds;
};

struct TransformerServeBatchResult
{
	// One result per request (aligned to input order).
	std::vector<TransformerGenerateResult> results;
};

} // namespace glades
