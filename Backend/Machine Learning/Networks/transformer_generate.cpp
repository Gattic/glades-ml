// Production-style transformer token LM generation API (decoder-only, KV-cache).
//
// This is intentionally lightweight and dependency-free:
// - uses session-based KV-cache inference (TransformerLmSession / TransformerLmBatchSession)
// - provides greedy + temperature/top-k/top-p sampling
// - supports streaming callbacks + cancellation polling
//
#include "network.h"
#include "sampling_utils.h"
#include "../rng.h"
#include "Backend/Database/GLogger.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "logfmt_utils.h"

using namespace glades;
using namespace glades::logfmt;

namespace {

using namespace glades::sampling;

static inline bool is_finite(float x) { return std::isfinite(x); }

// Lightweight 64-bit mixing utilities for deterministic per-call/per-request RNG seeding.
// We intentionally DO NOT use (or mutate) NNetwork::rngEngine in inference APIs.
static inline uint64_t mix64(uint64_t x)
{
	// splitmix64 finalizer (public domain-ish; widely used).
	x += 0x9E3779B97F4A7C15ULL;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
	return x ^ (x >> 31);
}

static inline uint64_t hash_u32_vec_fnv1a64(const std::vector<unsigned int>& v)
{
	// Stable across platforms (treat each element as 4 little-endian bytes).
	uint64_t h = 1469598103934665603ULL;
	const uint64_t prime = 1099511628211ULL;
	for (size_t i = 0; i < v.size(); ++i)
	{
		const uint32_t x = static_cast<uint32_t>(v[i]);
		const unsigned char b0 = static_cast<unsigned char>((x >> 0) & 0xFFu);
		const unsigned char b1 = static_cast<unsigned char>((x >> 8) & 0xFFu);
		const unsigned char b2 = static_cast<unsigned char>((x >> 16) & 0xFFu);
		const unsigned char b3 = static_cast<unsigned char>((x >> 24) & 0xFFu);
		h ^= static_cast<uint64_t>(b0); h *= prime;
		h ^= static_cast<uint64_t>(b1); h *= prime;
		h ^= static_cast<uint64_t>(b2); h *= prime;
		h ^= static_cast<uint64_t>(b3); h *= prime;
	}
	return h;
}

struct ScopedTimerMs
{
	const glades::NNetwork* net;
	double* acc;
	int64_t t0ms;
	explicit ScopedTimerMs(const glades::NNetwork* n, double* outAcc) : net(n), acc((n && outAcc) ? outAcc : NULL), t0ms(0)
	{
		if (acc)
			t0ms = net->getCurrentTimeMilliseconds();
	}
	~ScopedTimerMs()
	{
		if (!acc)
			return;
		const int64_t t1ms = net->getCurrentTimeMilliseconds();
		*acc += static_cast<double>(t1ms - t0ms);
	}
};

static const char* stop_reason_string(const glades::NNetwork::TransformerGenerateResult& r)
{
	if (r.stoppedByCallback) return "callback";
	if (r.stoppedOnEos) return "eos";
	if (r.stoppedByStopToken) return "stop_token";
	if (r.stoppedByLimit) return "limit";
	return "unknown";
}

static void log_transformer_generate_start(shmea::GLogger* logger,
                                          unsigned int promptLen,
                                          const glades::NNetwork::TransformerGenerateConfig& cfg,
                                          unsigned int vocabSize,
                                          unsigned int wantMaxLen)
{
	if (!logger)
		return;
	std::ostringstream oss;
	oss << "event=transformer_generate_start";
	append_logfmt_kv(oss, "prompt_len", promptLen);
	append_logfmt_kv(oss, "max_new", cfg.maxNewTokens);
	append_logfmt_kv(oss, "max_seq_len", wantMaxLen);
	append_logfmt_kv(oss, "vocab_size", vocabSize);
	append_logfmt_kv(oss, "temperature", cfg.temperature);
	append_logfmt_kv(oss, "top_k", cfg.topK);
	append_logfmt_kv(oss, "top_p", cfg.topP);
	append_logfmt_kv(oss, "top_p_top_k_cap", cfg.topPTopKCap);
	// For explicitness: when using top-p without an explicit top-k, we may apply an approximation cap.
	if (cfg.topP < 1.0f && cfg.topK == 0u && cfg.topPTopKCap > 0u)
		append_logfmt_kv(oss, "top_p_effective_top_k", std::min(vocabSize, cfg.topPTopKCap));
	append_logfmt_kv(oss, "eos_token_id", cfg.eosTokenId);
	append_logfmt_kv(oss, "stop_on_eos", cfg.stopOnEos);
	append_logfmt_kv(oss, "include_prompt", cfg.includePromptInOutput);
	append_logfmt_kv(oss, "rng_seed_override", static_cast<unsigned long long>(cfg.rngSeedOverride));
	logger->info("Transformer", shmea::GString(oss.str().c_str()));
}

static void log_transformer_generate_end(shmea::GLogger* logger,
                                        const glades::NNetworkStatus& st,
                                        unsigned int promptLen,
                                        const glades::NNetwork::TransformerGenerateConfig& cfg,
                                        unsigned int wantMaxLen,
                                        const glades::NNetwork::TransformerGenerateResult& out,
                                        unsigned int genTokens,
                                        double wallMs,
                                        double msPrefill,
                                        double msSample,
                                        double msDecodeAppend,
                                        const glades::NNetwork::TransformerKvPerfBreakdown* perf)
{
	if (!logger)
		return;
	const double tps = (wallMs > 0.0) ? (static_cast<double>(genTokens) / (wallMs / 1000.0)) : 0.0;

	std::ostringstream oss;
	oss << "event=transformer_generate_end";
	append_logfmt_kv(oss, "ok", st.ok());
	if (!st.ok())
		append_logfmt_kv(oss, "error", st.message);
	append_logfmt_kv(oss, "prompt_len", promptLen);
	append_logfmt_kv(oss, "max_new", cfg.maxNewTokens);
	append_logfmt_kv(oss, "max_seq_len", wantMaxLen);
	append_logfmt_kv(oss, "include_prompt", cfg.includePromptInOutput);
	append_logfmt_kv(oss, "stop_reason", std::string(stop_reason_string(out)));
	append_logfmt_kv(oss, "tokens_generated", genTokens);
	append_logfmt_kv(oss, "wall_ms", wallMs);
	append_logfmt_kv(oss, "tokens_per_sec", tps);
	append_logfmt_kv(oss, "prefill_ms", msPrefill);
	append_logfmt_kv(oss, "sample_ms", msSample);
	append_logfmt_kv(oss, "decode_append_ms", msDecodeAppend);
	if (perf)
	{
		const glades::NNetwork::TransformerKvPerfBreakdown& p = *perf;
		append_logfmt_kv(oss, "kv_appends", static_cast<unsigned long long>(p.kvAppends));
		append_logfmt_kv(oss, "kv_ms_total", p.msTotal);
		append_logfmt_kv(oss, "kv_ms_embed", p.msEmbed);
		append_logfmt_kv(oss, "kv_ms_posenc", p.msPosEnc);
		append_logfmt_kv(oss, "kv_ms_norm", p.msNorm);
		append_logfmt_kv(oss, "kv_ms_qkv", p.msProjQKV);
		append_logfmt_kv(oss, "kv_ms_rope", p.msRoPE);
		append_logfmt_kv(oss, "kv_ms_kvstore", p.msKVStore);
		append_logfmt_kv(oss, "kv_ms_attn", p.msAttention);
		append_logfmt_kv(oss, "kv_ms_wo", p.msWo);
		append_logfmt_kv(oss, "kv_ms_ffn", p.msFFN);
		append_logfmt_kv(oss, "kv_ms_logits", p.msLogits);
		append_logfmt_kv(oss, "sin_cache_hits", static_cast<unsigned long long>(p.sinCacheHits));
		append_logfmt_kv(oss, "sin_cache_misses", static_cast<unsigned long long>(p.sinCacheMisses));
		append_logfmt_kv(oss, "rope_cache_hits", static_cast<unsigned long long>(p.ropeCacheHits));
		append_logfmt_kv(oss, "rope_cache_misses", static_cast<unsigned long long>(p.ropeCacheMisses));
		append_logfmt_kv(oss, "non_finite_hidden", static_cast<unsigned long long>(p.nonFiniteHiddenState));
		append_logfmt_kv(oss, "non_finite_layer", p.lastNonFiniteLayer);
		append_logfmt_kv(oss, "non_finite_pos", p.lastNonFinitePos);
	}
	logger->info("Transformer", shmea::GString(oss.str().c_str()));
}

static void log_transformer_serve_batch_start(shmea::GLogger* logger,
                                             unsigned int batchSize,
                                             unsigned int maxPromptLen,
                                             unsigned int globalMaxLen,
                                             unsigned int globalMaxNew)
{
	if (!logger)
		return;
	std::ostringstream oss;
	oss << "event=transformer_serve_batch_start";
	append_logfmt_kv(oss, "batch_size", batchSize);
	append_logfmt_kv(oss, "max_prompt_len", maxPromptLen);
	append_logfmt_kv(oss, "global_max_len", globalMaxLen);
	append_logfmt_kv(oss, "global_max_new", globalMaxNew);
	logger->info("Transformer", shmea::GString(oss.str().c_str()));
}

static void log_transformer_serve_request_end(shmea::GLogger* logger,
                                             unsigned int requestIndex,
                                             unsigned int promptLen,
                                             const glades::NNetwork::TransformerGenerateConfig& cfg,
                                             unsigned int maxSeqLen,
                                             unsigned int vocabSize,
                                             const glades::NNetwork::TransformerGenerateResult& rr,
                                             unsigned int tokensGenerated)
{
	if (!logger)
		return;
	std::ostringstream oss;
	oss << "event=transformer_serve_request_end";
	append_logfmt_kv(oss, "request", requestIndex);
	append_logfmt_kv(oss, "prompt_len", promptLen);
	append_logfmt_kv(oss, "max_new", cfg.maxNewTokens);
	append_logfmt_kv(oss, "max_seq_len", maxSeqLen);
	append_logfmt_kv(oss, "include_prompt", cfg.includePromptInOutput);
	append_logfmt_kv(oss, "vocab_size", vocabSize);
	append_logfmt_kv(oss, "temperature", cfg.temperature);
	append_logfmt_kv(oss, "top_k", cfg.topK);
	append_logfmt_kv(oss, "top_p", cfg.topP);
	append_logfmt_kv(oss, "top_p_top_k_cap", cfg.topPTopKCap);
	if (cfg.topP < 1.0f && cfg.topK == 0u && cfg.topPTopKCap > 0u)
		append_logfmt_kv(oss, "top_p_effective_top_k", std::min(vocabSize, cfg.topPTopKCap));
	append_logfmt_kv(oss, "stop_reason", std::string(stop_reason_string(rr)));
	append_logfmt_kv(oss, "tokens_generated", tokensGenerated);
	logger->info("Transformer", shmea::GString(oss.str().c_str()));
}

static void log_transformer_serve_batch_end(shmea::GLogger* logger,
                                           const glades::NNetworkStatus& st,
                                           unsigned int batchSize,
                                           unsigned int globalMaxLen,
                                           unsigned int globalMaxNew,
                                           double wallMs,
                                           unsigned long long totalTokensGenerated,
                                           double msPrefillAppend,
                                           double msSample,
                                           double msDecodeAppend,
                                           const glades::NNetwork::TransformerKvPerfBreakdown* perf)
{
	if (!logger)
		return;
	const double tps = (wallMs > 0.0) ? (static_cast<double>(totalTokensGenerated) / (wallMs / 1000.0)) : 0.0;
	std::ostringstream oss;
	oss << "event=transformer_serve_batch_end";
	append_logfmt_kv(oss, "ok", st.ok());
	if (!st.ok())
		append_logfmt_kv(oss, "error", st.message);
	append_logfmt_kv(oss, "batch_size", batchSize);
	append_logfmt_kv(oss, "global_max_len", globalMaxLen);
	append_logfmt_kv(oss, "global_max_new", globalMaxNew);
	append_logfmt_kv(oss, "wall_ms", wallMs);
	append_logfmt_kv(oss, "tokens_generated", totalTokensGenerated);
	append_logfmt_kv(oss, "tokens_per_sec", tps);
	append_logfmt_kv(oss, "prefill_append_ms", msPrefillAppend);
	append_logfmt_kv(oss, "sample_ms", msSample);
	append_logfmt_kv(oss, "decode_append_ms", msDecodeAppend);
	if (perf)
	{
		const glades::NNetwork::TransformerKvPerfBreakdown& p = *perf;
		append_logfmt_kv(oss, "kv_appends", static_cast<unsigned long long>(p.kvAppends));
		append_logfmt_kv(oss, "kv_ms_total", p.msTotal);
		append_logfmt_kv(oss, "kv_ms_embed", p.msEmbed);
		append_logfmt_kv(oss, "kv_ms_posenc", p.msPosEnc);
		append_logfmt_kv(oss, "kv_ms_norm", p.msNorm);
		append_logfmt_kv(oss, "kv_ms_qkv", p.msProjQKV);
		append_logfmt_kv(oss, "kv_ms_rope", p.msRoPE);
		append_logfmt_kv(oss, "kv_ms_kvstore", p.msKVStore);
		append_logfmt_kv(oss, "kv_ms_attn", p.msAttention);
		append_logfmt_kv(oss, "kv_ms_wo", p.msWo);
		append_logfmt_kv(oss, "kv_ms_ffn", p.msFFN);
		append_logfmt_kv(oss, "kv_ms_logits", p.msLogits);
		append_logfmt_kv(oss, "sin_cache_hits", static_cast<unsigned long long>(p.sinCacheHits));
		append_logfmt_kv(oss, "sin_cache_misses", static_cast<unsigned long long>(p.sinCacheMisses));
		append_logfmt_kv(oss, "rope_cache_hits", static_cast<unsigned long long>(p.ropeCacheHits));
		append_logfmt_kv(oss, "rope_cache_misses", static_cast<unsigned long long>(p.ropeCacheMisses));
		append_logfmt_kv(oss, "non_finite_hidden", static_cast<unsigned long long>(p.nonFiniteHiddenState));
		append_logfmt_kv(oss, "non_finite_layer", p.lastNonFiniteLayer);
		append_logfmt_kv(oss, "non_finite_pos", p.lastNonFinitePos);
	}
	logger->info("Transformer", shmea::GString(oss.str().c_str()));
}

static bool sample_token_from_logits(const std::vector<float>& logits,
                                     glades::rng::Engine& rng,
                                     float temperature,
                                     unsigned int topK,
                                     float topP,
                                     unsigned int topPTopKCap,
                                     unsigned int& outToken,
                                     std::vector<unsigned int>& idxScratch,
                                     std::vector<float>& weightScratch)
{
	return sample_token_from_logits_ptr(logits.empty() ? NULL : &logits[0],
	                                    rng,
	                                    static_cast<unsigned int>(logits.size()),
	                                    temperature,
	                                    topK,
	                                    topP,
	                                    topPTopKCap,
	                                    outToken,
	                                    idxScratch,
	                                    weightScratch);
}

} // namespace

glades::NNetworkStatus glades::NNetwork::transformerLmGenerate(const std::vector<unsigned int>& promptTokens,
                                                              const TransformerGenerateConfig& cfg,
                                                              TransformerGenerateResult& out,
                                                              ITransformerGenerateCallbacks* cb) const
{
	out = TransformerGenerateResult();

	if (netType != TYPE_TRANSFORMER_DECODER)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmGenerate: requires TYPE_TRANSFORMER_DECODER");
	if (!trainingConfig.transformer.enableTokenEmbedding)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmGenerate: requires transformer.enableTokenEmbedding");
	if (!tensorTransformer.initialized || !tensorTransformer.tokenModel)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmGenerate: transformer tensors not initialized for token LM (loadModel or run once)");
	if (promptTokens.empty())
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmGenerate: promptTokens is empty (include a BOS token if needed)");

	const unsigned int vocab = tensorTransformer.vocabSize;
	if (vocab == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmGenerate: vocabSize is 0");

	for (size_t i = 0; i < promptTokens.size(); ++i)
	{
		if (promptTokens[i] >= vocab)
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmGenerate: prompt tokenId out of range");
	}

	// Install a per-call RNG engine (never mutate the network RNG engine).
	//
	// Determinism policy for inference:
	// - If cfg.rngSeedOverride != 0: seed exactly with that value.
	// - Else: derive a deterministic seed from this network's configured rngSeed and the prompt tokens.
	//   This makes repeated calls with the same prompt/config deterministic and avoids shared-state races.
	glades::rng::Engine callEngine;
	uint64_t seed = cfg.rngSeedOverride;
	if (seed == 0ULL)
	{
		// Base: network seed + prompt hash.
		const uint64_t h = hash_u32_vec_fnv1a64(promptTokens);
		seed = mix64(rngSeed ^ h);
	}
	glades::rng::seed_engine(callEngine, seed);

	// KV cache sizing.
	const unsigned int promptLen = static_cast<unsigned int>(promptTokens.size());
	const unsigned int wantMaxLen = (cfg.maxSeqLen > 0u) ? cfg.maxSeqLen : (promptLen + cfg.maxNewTokens);
	if (wantMaxLen == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmGenerate: maxSeqLen resolves to 0");
	if (wantMaxLen < promptLen)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmGenerate: maxSeqLen < prompt length");
	if (wantMaxLen < (promptLen + cfg.maxNewTokens))
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmGenerate: maxSeqLen < promptLen + maxNewTokens");

	const TransformerMetricsConfig& mcfg = getTransformerMetricsConfig();
	const bool metricsOn = mcfg.enable;
	shmea::GLogger* logger = metricsOn ? getLogger() : NULL;
	const int64_t wall0ms = getCurrentTimeMilliseconds();
	double msPrefill = 0.0;
	double msSample = 0.0;
	double msDecodeAppend = 0.0;
	if (metricsOn && logger)
		log_transformer_generate_start(logger, promptLen, cfg, vocab, wantMaxLen);

	// Output tokens: optionally include prompt.
	if (cfg.includePromptInOutput)
		out.tokens = promptTokens;
	else
		out.tokens.clear();

	// Initialize per-call KV session and prefill prompt.
	TransformerLmSession session;
	{
		const NNetworkStatus st = transformerLmSessionReset(session, wantMaxLen);
		if (!st.ok())
		{
			if (metricsOn && logger)
			{
				const double wallMs = static_cast<double>(getCurrentTimeMilliseconds() - wall0ms);
				const unsigned int totalOut = static_cast<unsigned int>(out.tokens.size());
				const unsigned int genTokens =
				    cfg.includePromptInOutput ? ((totalOut >= promptLen) ? (totalOut - promptLen) : 0u) : totalOut;
				log_transformer_generate_end(logger, st, promptLen, cfg, wantMaxLen, out, genTokens, wallMs, msPrefill, msSample,
				                             msDecodeAppend, NULL);
			}
			return st;
		}
	}

	std::vector<float> logits; // reused

	// Prefill: append all prompt tokens, computing logits only for the final prompt token.
	for (unsigned int i = 0; i < promptLen; ++i)
	{
		const bool last = (i + 1u == promptLen);
		ScopedTimerMs t(this, metricsOn ? &msPrefill : NULL);
		const NNetworkStatus st = transformerLmSessionAppend(session, promptTokens[i], last ? &logits : NULL);
		if (!st.ok())
		{
			if (metricsOn && logger)
			{
				const double wallMs = static_cast<double>(getCurrentTimeMilliseconds() - wall0ms);
				const unsigned int totalOut = static_cast<unsigned int>(out.tokens.size());
				const unsigned int genTokens =
				    cfg.includePromptInOutput ? ((totalOut >= promptLen) ? (totalOut - promptLen) : 0u) : totalOut;
				log_transformer_generate_end(logger, st, promptLen, cfg, wantMaxLen, out, genTokens, wallMs, msPrefill, msSample,
				                             msDecodeAppend, &session.perf);
			}
			return st;
		}
	}

	// If no generation requested, we're done.
	if (cfg.maxNewTokens == 0u)
	{
		out.stoppedByLimit = true;
		out.lastToken = cfg.includePromptInOutput ? promptTokens[promptTokens.size() - 1u] : 0u;
		const NNetworkStatus st(NNetworkStatus::OK, std::string());
		if (metricsOn && logger)
		{
			const double wallMs = static_cast<double>(getCurrentTimeMilliseconds() - wall0ms);
			const unsigned int totalOut = static_cast<unsigned int>(out.tokens.size());
			const unsigned int genTokens =
			    cfg.includePromptInOutput ? ((totalOut >= promptLen) ? (totalOut - promptLen) : 0u) : totalOut;
			log_transformer_generate_end(logger, st, promptLen, cfg, wantMaxLen, out, genTokens, wallMs, msPrefill, msSample, msDecodeAppend,
			                             &session.perf);
		}
		return st;
	}

	// Sampling scratch buffers (pre-sized to vocab to avoid per-step allocations).
	std::vector<unsigned int> idxScratch(vocab);
	for (unsigned int i = 0u; i < vocab; ++i)
		idxScratch[i] = i;
	std::vector<float> weightScratch(vocab, 0.0f);

	const int eos = cfg.eosTokenId;
	const bool stopOnEos = cfg.stopOnEos && (eos >= 0);

	for (unsigned int genIdx = 0u; genIdx < cfg.maxNewTokens; ++genIdx)
	{
		if (cb && cb->shouldStop(*this))
		{
			out.stoppedByCallback = true;
			const NNetworkStatus st(NNetworkStatus::OK, std::string());
			if (metricsOn && logger)
			{
				const double wallMs = static_cast<double>(getCurrentTimeMilliseconds() - wall0ms);
				const unsigned int totalOut = static_cast<unsigned int>(out.tokens.size());
				const unsigned int genTokens =
				    cfg.includePromptInOutput ? ((totalOut >= promptLen) ? (totalOut - promptLen) : 0u) : totalOut;
				log_transformer_generate_end(logger, st, promptLen, cfg, wantMaxLen, out, genTokens, wallMs, msPrefill, msSample, msDecodeAppend,
				                             &session.perf);
			}
			return st;
		}

		unsigned int nextTok = 0u;
		{
			ScopedTimerMs t(this, metricsOn ? &msSample : NULL);
			if (!sample_token_from_logits(logits, callEngine, cfg.temperature, cfg.topK, cfg.topP, cfg.topPTopKCap, nextTok, idxScratch, weightScratch))
			{
				const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "transformerLmGenerate: failed to sample token from logits");
				if (metricsOn && logger)
				{
					const double wallMs = static_cast<double>(getCurrentTimeMilliseconds() - wall0ms);
					const unsigned int totalOut = static_cast<unsigned int>(out.tokens.size());
					const unsigned int genTokens =
					    cfg.includePromptInOutput ? ((totalOut >= promptLen) ? (totalOut - promptLen) : 0u) : totalOut;
					log_transformer_generate_end(logger, st, promptLen, cfg, wantMaxLen, out, genTokens, wallMs, msPrefill, msSample, msDecodeAppend,
					                             &session.perf);
				}
				return st;
			}
		}
		if (nextTok >= vocab)
		{
			const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "transformerLmGenerate: sampled tokenId out of range");
			if (metricsOn && logger)
			{
				const double wallMs = static_cast<double>(getCurrentTimeMilliseconds() - wall0ms);
				const unsigned int totalOut = static_cast<unsigned int>(out.tokens.size());
				const unsigned int genTokens =
				    cfg.includePromptInOutput ? ((totalOut >= promptLen) ? (totalOut - promptLen) : 0u) : totalOut;
				log_transformer_generate_end(logger, st, promptLen, cfg, wantMaxLen, out, genTokens, wallMs, msPrefill, msSample, msDecodeAppend,
				                             &session.perf);
			}
			return st;
		}

		// Append token to KV cache and fetch logits for next step.
		{
			ScopedTimerMs t(this, metricsOn ? &msDecodeAppend : NULL);
			const NNetworkStatus st = transformerLmSessionAppend(session, nextTok, &logits);
			if (!st.ok())
			{
				if (metricsOn && logger)
				{
					const double wallMs = static_cast<double>(getCurrentTimeMilliseconds() - wall0ms);
					const unsigned int totalOut = static_cast<unsigned int>(out.tokens.size());
					const unsigned int genTokens =
					    cfg.includePromptInOutput ? ((totalOut >= promptLen) ? (totalOut - promptLen) : 0u) : totalOut;
					log_transformer_generate_end(logger, st, promptLen, cfg, wantMaxLen, out, genTokens, wallMs, msPrefill, msSample, msDecodeAppend,
					                             &session.perf);
				}
				return st;
			}
		}

		out.tokens.push_back(nextTok);
		out.lastToken = nextTok;

		if (cb)
		{
			const bool stop = cb->onToken(*this, nextTok, genIdx);
			if (stop)
			{
				out.stoppedByCallback = true;
				const NNetworkStatus st(NNetworkStatus::OK, std::string());
				if (metricsOn && logger)
				{
					const double wallMs = static_cast<double>(getCurrentTimeMilliseconds() - wall0ms);
					const unsigned int totalOut = static_cast<unsigned int>(out.tokens.size());
					const unsigned int genTokens =
					    cfg.includePromptInOutput ? ((totalOut >= promptLen) ? (totalOut - promptLen) : 0u) : totalOut;
					log_transformer_generate_end(logger, st, promptLen, cfg, wantMaxLen, out, genTokens, wallMs, msPrefill, msSample, msDecodeAppend,
					                             &session.perf);
				}
				return st;
			}
		}

		if (stopOnEos && nextTok == static_cast<unsigned int>(eos))
		{
			out.stoppedOnEos = true;
			const NNetworkStatus st(NNetworkStatus::OK, std::string());
			if (metricsOn && logger)
			{
				const double wallMs = static_cast<double>(getCurrentTimeMilliseconds() - wall0ms);
				const unsigned int totalOut = static_cast<unsigned int>(out.tokens.size());
				const unsigned int genTokens =
				    cfg.includePromptInOutput ? ((totalOut >= promptLen) ? (totalOut - promptLen) : 0u) : totalOut;
				log_transformer_generate_end(logger, st, promptLen, cfg, wantMaxLen, out, genTokens, wallMs, msPrefill, msSample, msDecodeAppend,
				                             &session.perf);
			}
			return st;
		}
	}

	out.stoppedByLimit = true;
	{
		const NNetworkStatus st(NNetworkStatus::OK, std::string());
		if (metricsOn && logger)
		{
			const double wallMs = static_cast<double>(getCurrentTimeMilliseconds() - wall0ms);
			const unsigned int totalOut = static_cast<unsigned int>(out.tokens.size());
			const unsigned int genTokens =
			    cfg.includePromptInOutput ? ((totalOut >= promptLen) ? (totalOut - promptLen) : 0u) : totalOut;
			log_transformer_generate_end(logger, st, promptLen, cfg, wantMaxLen, out, genTokens, wallMs, msPrefill, msSample, msDecodeAppend,
			                             &session.perf);
		}
		return st;
	}
}

glades::NNetworkStatus glades::NNetwork::transformerLmServeGenerateBatch(const std::vector<TransformerServeRequest>& requests,
                                                                         TransformerServeBatchResult& out,
                                                                         ITransformerServeCallbacks* cb) const
{
	out = TransformerServeBatchResult();
	out.results.clear();

	if (netType != TYPE_TRANSFORMER_DECODER)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeGenerateBatch: requires TYPE_TRANSFORMER_DECODER");
	if (!trainingConfig.transformer.enableTokenEmbedding)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeGenerateBatch: requires transformer.enableTokenEmbedding");
	if (!tensorTransformer.initialized || !tensorTransformer.tokenModel)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmServeGenerateBatch: transformer tensors not initialized for token LM (loadModel or run once)");
	if (requests.empty())
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeGenerateBatch: requests is empty");

	const unsigned int B = static_cast<unsigned int>(requests.size());
	const unsigned int vocab = tensorTransformer.vocabSize;
	if (vocab == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmServeGenerateBatch: vocabSize is 0");

	const TransformerMetricsConfig& mcfg = getTransformerMetricsConfig();
	const bool metricsOn = mcfg.enable;
	shmea::GLogger* logger = metricsOn ? getLogger() : NULL;
	const int64_t wall0ms = getCurrentTimeMilliseconds();
	double msPrefillAppend = 0.0;
	double msDecodeAppend = 0.0;
	double msSample = 0.0;

	// Determine per-request prompt length and required max sequence length.
	std::vector<unsigned int> promptLen(B, 0u);
	std::vector<unsigned int> reqMaxLen(B, 0u);
	std::vector<unsigned int> reqMaxNew(B, 0u);
	unsigned int globalMaxLen = 0u;
	unsigned int globalMaxNew = 0u;
	unsigned int maxPromptLen = 0u;
	for (unsigned int r = 0; r < B; ++r)
	{
		const TransformerServeRequest& req = requests[r];
		if (req.promptTokens.empty())
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeGenerateBatch: request has empty promptTokens");

		promptLen[r] = static_cast<unsigned int>(req.promptTokens.size());
		if (promptLen[r] > maxPromptLen)
			maxPromptLen = promptLen[r];
		reqMaxNew[r] = req.cfg.maxNewTokens;
		if (reqMaxNew[r] > globalMaxNew)
			globalMaxNew = reqMaxNew[r];

		const unsigned int want = (req.cfg.maxSeqLen > 0u) ? req.cfg.maxSeqLen : (promptLen[r] + req.cfg.maxNewTokens);
		if (want == 0u)
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeGenerateBatch: request maxSeqLen resolves to 0");
		if (want < promptLen[r])
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeGenerateBatch: request maxSeqLen < prompt length");
		if (want < (promptLen[r] + req.cfg.maxNewTokens))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeGenerateBatch: request maxSeqLen < promptLen + maxNewTokens");
		reqMaxLen[r] = want;
		if (want > globalMaxLen)
			globalMaxLen = want;

		// Validate prompt token ids.
		for (size_t i = 0; i < req.promptTokens.size(); ++i)
		{
			if (req.promptTokens[i] >= vocab)
				return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeGenerateBatch: prompt tokenId out of range");
		}
		// Validate stop tokens.
		for (size_t i = 0; i < req.stopTokenIds.size(); ++i)
		{
			if (req.stopTokenIds[i] >= vocab)
				return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeGenerateBatch: stopTokenId out of range");
		}
	}

	if (metricsOn && logger)
		log_transformer_serve_batch_start(logger, B, maxPromptLen, globalMaxLen, globalMaxNew);

	// Initialize per-call batched KV session for the maximum required length.
	TransformerLmBatchSession session;
	{
		const NNetworkStatus st = transformerLmBatchSessionReset(session, B, globalMaxLen);
		if (!st.ok())
		{
			if (metricsOn && logger)
			{
				std::ostringstream oss;
				oss << "event=transformer_serve_batch_end";
				append_logfmt_kv(oss, "ok", false);
				append_logfmt_kv(oss, "error", st.message);
				append_logfmt_kv(oss, "batch_size", B);
				logger->info("Transformer", shmea::GString(oss.str().c_str()));
			}
			return st;
		}
	}

	// Single-exit return status (so we can always emit end-of-call metrics).
	NNetworkStatus retSt(NNetworkStatus::OK, std::string());

	// Initialize outputs (optionally include prompt).
	out.results.resize(B);
	for (unsigned int r = 0; r < B; ++r)
	{
		TransformerGenerateResult& rr = out.results[r];
		rr = TransformerGenerateResult();
		if (requests[r].cfg.includePromptInOutput)
			rr.tokens = requests[r].promptTokens;
	}

	// End-of-call structured metrics emitter (runs on all returns after this point).
	struct ServeBatchMetricsGuard
	{
		const glades::NNetwork* net;
		shmea::GLogger* logger;
		const glades::NNetwork::TransformerMetricsConfig* cfg;
		const int64_t* wall0ms;
		const glades::NNetworkStatus* status;
		const std::vector<glades::NNetwork::TransformerServeRequest>* requests;
		const std::vector<unsigned int>* promptLen;
		const std::vector<unsigned int>* reqMaxLen;
		const glades::NNetwork::TransformerLmBatchSession* session;
		const glades::NNetwork::TransformerServeBatchResult* out;
		unsigned int B;
		unsigned int globalMaxLen;
		unsigned int globalMaxNew;
		const double* msPrefillAppend;
		const double* msSample;
		const double* msDecodeAppend;

		ServeBatchMetricsGuard(const glades::NNetwork* n,
		                      shmea::GLogger* l,
		                      const glades::NNetwork::TransformerMetricsConfig* c,
		                      const int64_t* w0,
		                      const glades::NNetworkStatus* st,
		                      const std::vector<glades::NNetwork::TransformerServeRequest>* rq,
		                      const std::vector<unsigned int>* pl,
		                      const std::vector<unsigned int>* rml,
		                      const glades::NNetwork::TransformerLmBatchSession* sess,
		                      const glades::NNetwork::TransformerServeBatchResult* o,
		                      unsigned int bsz,
		                      unsigned int gml,
		                      unsigned int gmn,
		                      const double* pms,
		                      const double* sms,
		                      const double* dms)
		    : net(n),
		      logger(l),
		      cfg(c),
		      wall0ms(w0),
		      status(st),
		      requests(rq),
		      promptLen(pl),
		      reqMaxLen(rml),
		      session(sess),
		      out(o),
		      B(bsz),
		      globalMaxLen(gml),
		      globalMaxNew(gmn),
		      msPrefillAppend(pms),
		      msSample(sms),
		      msDecodeAppend(dms)
		{
		}

		~ServeBatchMetricsGuard()
		{
			if (!net || !logger || !cfg || !cfg->enable || !wall0ms || !status || !requests || !promptLen || !reqMaxLen || !session || !out)
				return;

			const double wallMs = static_cast<double>(net->getCurrentTimeMilliseconds() - *wall0ms);
			unsigned long long totalGen = 0ULL;
			for (unsigned int r = 0; r < B; ++r)
			{
				const glades::NNetwork::TransformerGenerateResult& rr = out->results[r];
				const unsigned int totalOut = static_cast<unsigned int>(rr.tokens.size());
				const unsigned int gen =
				    (*requests)[r].cfg.includePromptInOutput ? ((totalOut >= (*promptLen)[r]) ? (totalOut - (*promptLen)[r]) : 0u) : totalOut;
				totalGen += static_cast<unsigned long long>(gen);
			}

			log_transformer_serve_batch_end(logger, *status, B, globalMaxLen, globalMaxNew, wallMs, totalGen,
			                                (msPrefillAppend ? *msPrefillAppend : 0.0),
			                                (msSample ? *msSample : 0.0),
			                                (msDecodeAppend ? *msDecodeAppend : 0.0),
			                                &session->perf);

			if (cfg->logPerRequest)
			{
				for (unsigned int r = 0; r < B; ++r)
				{
					const glades::NNetwork::TransformerGenerateResult& rr = out->results[r];
					const unsigned int totalOut = static_cast<unsigned int>(rr.tokens.size());
					const unsigned int gen =
					    (*requests)[r].cfg.includePromptInOutput ? ((totalOut >= (*promptLen)[r]) ? (totalOut - (*promptLen)[r]) : 0u) : totalOut;
					log_transformer_serve_request_end(logger,
					                                 r,
					                                 (*promptLen)[r],
					                                 (*requests)[r].cfg,
					                                 (*reqMaxLen)[r],
					                                 net->tensorTransformer.vocabSize,
					                                 rr,
					                                 gen);
				}
			}
		}
	};

	ServeBatchMetricsGuard metricsGuard(this,
	                                   logger,
	                                   &mcfg,
	                                   &wall0ms,
	                                   &retSt,
	                                   &requests,
	                                   &promptLen,
	                                   &reqMaxLen,
	                                   &session,
	                                   &out,
	                                   B,
	                                   globalMaxLen,
	                                   globalMaxNew,
	                                   &msPrefillAppend,
	                                   &msSample,
	                                   &msDecodeAppend);

	// Per-request RNG engines (never mutate the network RNG engine).
	//
	// Determinism policy:
	// - If request.cfg.rngSeedOverride != 0: seed exactly with that value.
	// - Else: derive a deterministic seed from the network seed + request index + prompt tokens.
	//   This avoids cross-request RNG coupling and is safe under concurrency.
	std::vector<glades::rng::Engine> reqEngines;
	reqEngines.resize(B);
	for (unsigned int r = 0; r < B; ++r)
	{
		uint64_t s = requests[r].cfg.rngSeedOverride;
		if (s == 0ULL)
		{
			const uint64_t h = hash_u32_vec_fnv1a64(requests[r].promptTokens);
			const uint64_t tag = mix64(static_cast<uint64_t>(r) + 1ULL);
			s = mix64(rngSeed ^ h ^ tag);
		}
		glades::rng::seed_engine(reqEngines[r], s);
	}

	// Buffers reused across the entire call (no per-step heap churn).
	std::vector<unsigned int> tokenIds(B, 0u);
	std::vector<unsigned char> active(B, 0u);
	std::vector<float> logitsFlat;      // output of append: [B, vocab]
	std::vector<float> prevLogitsFlat;  // "current" logits used for sampling next token: [B, vocab]
	prevLogitsFlat.assign(static_cast<size_t>(B) * static_cast<size_t>(vocab), 0.0f);

	// Prefill prompts without positional distortion (ragged):
	// We append only those requests that have a real token at this timestep.
	for (unsigned int t = 0; t < maxPromptLen; ++t)
	{
		for (unsigned int r = 0; r < B; ++r)
		{
			if (t < promptLen[r])
			{
				tokenIds[r] = requests[r].promptTokens[t];
				active[r] = 1u;
			}
			else
			{
				tokenIds[r] = 0u;
				active[r] = 0u;
			}
		}

		ScopedTimerMs tms(this, metricsOn ? &msPrefillAppend : NULL);
		const NNetworkStatus st = transformerLmBatchSessionAppendSelective(session, tokenIds, active, &logitsFlat);
		if (!st.ok())
		{
			retSt = st;
			return retSt;
		}

		// If this was the last prompt token for a request, capture its logits as the starting distribution.
		for (unsigned int r = 0; r < B; ++r)
		{
			if (active[r] == 0u)
				continue;
			if (t + 1u == promptLen[r])
			{
				const float* row = logitsFlat.empty() ? NULL : &logitsFlat[static_cast<size_t>(r) * static_cast<size_t>(vocab)];
				float* dst = prevLogitsFlat.empty() ? NULL : &prevLogitsFlat[static_cast<size_t>(r) * static_cast<size_t>(vocab)];
				if (row && dst)
					std::copy(row, row + vocab, dst);
			}
		}
	}

	// Requests that don't want new tokens are done now.
	std::vector<unsigned char> done(B, 0u);
	for (unsigned int r = 0; r < B; ++r)
	{
		if (requests[r].cfg.maxNewTokens == 0u)
		{
			out.results[r].stoppedByLimit = true;
			done[r] = 1u;
		}
	}

	// Sampling scratch buffers (pre-sized to vocab to avoid per-step allocations).
	std::vector<unsigned int> idxScratch(vocab);
	for (unsigned int i = 0u; i < vocab; ++i)
		idxScratch[i] = i;
	std::vector<float> weightScratch(vocab, 0.0f);

	// Decode: continuous batching with per-request early stopping.
	for (unsigned int genIdx = 0u; genIdx < globalMaxNew; ++genIdx)
	{
		if (cb && cb->shouldStopAll(*this))
		{
			// Mark any unfinished requests as callback-stopped.
			for (unsigned int r = 0; r < B; ++r)
				if (done[r] == 0u)
					out.results[r].stoppedByCallback = true;
			const NNetworkStatus st(NNetworkStatus::OK, std::string());
			retSt = st;
			return retSt;
		}

		unsigned int activeCount = 0u;
		for (unsigned int r = 0; r < B; ++r)
		{
			if (done[r] != 0u || genIdx >= reqMaxNew[r])
			{
				active[r] = 0u;
				continue;
			}
			if (cb && cb->shouldStopRequest(*this, r))
			{
				out.results[r].stoppedByCallback = true;
				done[r] = 1u;
				active[r] = 0u;
				continue;
			}

			const TransformerGenerateConfig& cfg = requests[r].cfg;

			// Enforce per-request max sequence length (prompt + generated) at KV-cache level.
			// If we've reached the request's configured max length, stop now without advancing position.
			const unsigned int curLen = (r < session.curLen.size()) ? session.curLen[r] : 0u;
			if (curLen >= reqMaxLen[r])
			{
				out.results[r].stoppedByLimit = true;
				done[r] = 1u;
				active[r] = 0u;
				continue;
			}

			const float* row = prevLogitsFlat.empty() ? NULL : &prevLogitsFlat[static_cast<size_t>(r) * static_cast<size_t>(vocab)];

			unsigned int nextTok = 0u;
			{
				ScopedTimerMs tms(this, metricsOn ? &msSample : NULL);
				if (!sample_token_from_logits_ptr(row, reqEngines[r], vocab, cfg.temperature, cfg.topK, cfg.topP, cfg.topPTopKCap, nextTok, idxScratch,
				                                  weightScratch))
				{
					const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "transformerLmServeGenerateBatch: failed to sample token from logits");
					retSt = st;
					return retSt;
				}
			}
			if (nextTok >= vocab)
			{
				const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "transformerLmServeGenerateBatch: sampled tokenId out of range");
				retSt = st;
				return retSt;
			}

			tokenIds[r] = nextTok;
			active[r] = 1u;
			++activeCount;
		}

		if (activeCount == 0u)
			break;

		// Append sampled tokens only for active requests and compute logits for the next step.
		{
			ScopedTimerMs tms(this, metricsOn ? &msDecodeAppend : NULL);
			const NNetworkStatus st = transformerLmBatchSessionAppendSelective(session, tokenIds, active, &logitsFlat);
			if (!st.ok())
			{
				retSt = st;
				return retSt;
			}
		}

		// Make logitsFlat the new "current" logits for sampling next step (zero-copy swap).
		prevLogitsFlat.swap(logitsFlat);

		// Emit + stop checks.
		for (unsigned int r = 0; r < B; ++r)
		{
			if (active[r] == 0u || done[r] != 0u)
				continue;

			const unsigned int tok = tokenIds[r];
			TransformerGenerateResult& rr = out.results[r];
			rr.tokens.push_back(tok);
			rr.lastToken = tok;

			if (cb)
			{
				const bool stopReq = cb->onToken(*this, r, tok, genIdx);
				if (stopReq)
				{
					rr.stoppedByCallback = true;
					done[r] = 1u;
					continue;
				}
			}

			const TransformerGenerateConfig& cfg = requests[r].cfg;
			const int eos = cfg.eosTokenId;
			if (cfg.stopOnEos && eos >= 0 && tok == static_cast<unsigned int>(eos))
			{
				rr.stoppedOnEos = true;
				done[r] = 1u;
				continue;
			}

			// Additional stop tokens.
			const std::vector<unsigned int>& stopTok = requests[r].stopTokenIds;
			for (size_t i = 0; i < stopTok.size(); ++i)
			{
				if (tok == stopTok[i])
				{
					rr.stoppedByStopToken = true;
					done[r] = 1u;
					break;
				}
			}

			// Per-request max token limit.
			if (done[r] == 0u && (genIdx + 1u) >= reqMaxNew[r])
			{
				rr.stoppedByLimit = true;
				done[r] = 1u;
			}
		}
	}

	// Any remaining unfinished requests stopped by limit (exhausted global loop or no active left).
	for (unsigned int r = 0; r < B; ++r)
	{
		if (done[r] == 0u)
		{
			out.results[r].stoppedByLimit = true;
			done[r] = 1u;
		}
	}

	retSt = NNetworkStatus(NNetworkStatus::OK, std::string());
	return retSt;
}

glades::NNetworkStatus glades::NNetwork::transformerLmServeBatcherReset(glades::NNetwork::TransformerServeBatcher& batcher,
                                                                        const glades::NNetwork::TransformerServeBatcherConfig& cfg) const
{
	batcher.reset();

	if (netType != TYPE_TRANSFORMER_DECODER)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherReset: requires TYPE_TRANSFORMER_DECODER");
	if (!trainingConfig.transformer.enableTokenEmbedding)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherReset: requires transformer.enableTokenEmbedding");
	if (!tensorTransformer.initialized || !tensorTransformer.tokenModel)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmServeBatcherReset: transformer tensors not initialized for token LM (loadModel or run once)");

	const unsigned int vocab = tensorTransformer.vocabSize;
	if (vocab == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmServeBatcherReset: vocabSize is 0");
	if (cfg.maxBatchSize == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherReset: maxBatchSize is 0");
	if (cfg.maxSeqLen == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherReset: maxSeqLen is 0");

	batcher.initialized = true;
	batcher.vocab = vocab;
	batcher.maxBatchSize = cfg.maxBatchSize;
	batcher.maxSeqLen = cfg.maxSeqLen;
	batcher.wipeKvOnRemove = cfg.wipeKvOnRemove;

	// Own a dedicated RNG stream for this batcher (does not mutate the network RNG).
	{
		const uint64_t seed = (cfg.rngSeed != 0ULL) ? cfg.rngSeed : (rngSeed != 0ULL ? rngSeed : 5489ULL);
		glades::rng::seed_engine(batcher.batchEngine, seed);
	}

	{
		const NNetworkStatus st = transformerLmBatchSessionReset(batcher.session, batcher.maxBatchSize, batcher.maxSeqLen);
		if (!st.ok())
			return st;
	}

	const unsigned int B = batcher.maxBatchSize;
	batcher.inUse.assign(B, 0u);
	batcher.done.assign(B, 0u);
	batcher.promptPos.assign(B, 0u);
	batcher.promptLen.assign(B, 0u);
	batcher.generated.assign(B, 0u);
	batcher.reqMaxNew.assign(B, 0u);
	batcher.reqMaxLen.assign(B, 0u);

	batcher.req.resize(B);
	batcher.results.resize(B);

	batcher.overrideEngines.resize(B);
	batcher.hasOverride.assign(B, 0u);

	batcher.tokenIds.assign(B, 0u);
	batcher.active.assign(B, 0u);
	batcher.sampledTok.assign(B, 0u);
	batcher.sampledIsValid.assign(B, 0u);

	batcher.prevLogitsFlat.assign(static_cast<size_t>(B) * static_cast<size_t>(vocab), 0.0f);
	batcher.logitsFlat.assign(static_cast<size_t>(B) * static_cast<size_t>(vocab), 0.0f);

	// Sampling scratch: pre-size to avoid hot-path resize.
	batcher.idxScratch.resize(vocab);
	for (unsigned int i = 0u; i < vocab; ++i)
		batcher.idxScratch[i] = i;
	batcher.weightScratch.assign(vocab, 0.0f);

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::NNetwork::transformerLmServeBatcherSubmit(glades::NNetwork::TransformerServeBatcher& batcher,
                                                                         const glades::NNetwork::TransformerServeRequest& request,
                                                                         unsigned int& outSlot) const
{
	outSlot = 0u;
	if (!batcher.initialized)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmServeBatcherSubmit: batcher not initialized (call transformerLmServeBatcherReset)");
	if (request.promptTokens.empty())
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherSubmit: request has empty promptTokens");
	if (batcher.vocab == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmServeBatcherSubmit: batcher vocabSize is 0");

	// Find a free slot.
	unsigned int slot = batcher.maxBatchSize;
	for (unsigned int i = 0u; i < batcher.maxBatchSize; ++i)
	{
		if (batcher.inUse[i] == 0u)
		{
			slot = i;
			break;
		}
	}
	if (slot >= batcher.maxBatchSize)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmServeBatcherSubmit: no free batch slots");

	const unsigned int vocab = batcher.vocab;

	// Validate prompt tokens.
	for (size_t i = 0; i < request.promptTokens.size(); ++i)
	{
		if (request.promptTokens[i] >= vocab)
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherSubmit: prompt tokenId out of range");
	}
	// Validate stop tokens.
	for (size_t i = 0; i < request.stopTokenIds.size(); ++i)
	{
		if (request.stopTokenIds[i] >= vocab)
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherSubmit: stopTokenId out of range");
	}

	const unsigned int promptLen = static_cast<unsigned int>(request.promptTokens.size());
	const unsigned int maxNew = request.cfg.maxNewTokens;
	const unsigned int wantMaxLen = (request.cfg.maxSeqLen > 0u) ? request.cfg.maxSeqLen : (promptLen + maxNew);
	if (wantMaxLen == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherSubmit: request maxSeqLen resolves to 0");
	if (wantMaxLen < promptLen)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherSubmit: request maxSeqLen < prompt length");
	if (wantMaxLen < (promptLen + maxNew))
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherSubmit: request maxSeqLen < promptLen + maxNewTokens");
	if (wantMaxLen > batcher.maxSeqLen)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherSubmit: request maxSeqLen exceeds batcher maxSeqLen");

	// Install request payload (owned copy).
	batcher.req[slot] = request;
	batcher.results[slot] = TransformerGenerateResult();
	if (request.cfg.includePromptInOutput)
		batcher.results[slot].tokens = request.promptTokens;

	batcher.inUse[slot] = 1u;
	batcher.done[slot] = 0u;
	batcher.promptPos[slot] = 0u;
	batcher.promptLen[slot] = promptLen;
	batcher.generated[slot] = 0u;
	batcher.reqMaxNew[slot] = maxNew;
	batcher.reqMaxLen[slot] = wantMaxLen;

	// Reset KV position for this slot (old KV contents are unreachable past curLen).
	if (slot < batcher.session.curLen.size())
		batcher.session.curLen[slot] = 0u;

	// Reset per-slot RNG override.
	batcher.hasOverride[slot] = 0u;
	if (request.cfg.rngSeedOverride != 0ULL)
	{
		glades::rng::seed_engine(batcher.overrideEngines[slot], request.cfg.rngSeedOverride);
		batcher.hasOverride[slot] = 1u;
	}

	// Reset logits row.
	{
		float* row = batcher.prevLogitsFlat.empty() ? NULL : &batcher.prevLogitsFlat[static_cast<size_t>(slot) * static_cast<size_t>(vocab)];
		if (row)
			std::fill(row, row + vocab, 0.0f);
	}

	outSlot = slot;
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::NNetwork::transformerLmServeBatcherRemove(glades::NNetwork::TransformerServeBatcher& batcher,
                                                                         unsigned int slot) const
{
	if (!batcher.initialized)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmServeBatcherRemove: batcher not initialized");
	if (slot >= batcher.maxBatchSize)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmServeBatcherRemove: slot out of range");
	if (batcher.inUse[slot] == 0u)
		return NNetworkStatus(NNetworkStatus::OK, std::string()); // idempotent

	// Optionally wipe KV + mask prefix used by this slot.
	if (batcher.wipeKvOnRemove)
	{
		const unsigned int usedLen = (slot < batcher.session.curLen.size()) ? batcher.session.curLen[slot] : 0u;
		const unsigned int maxLen = batcher.session.maxLen;
		const unsigned int nLayers = batcher.session.nLayers;
		const unsigned int dModelKV = batcher.session.dModelKV;

		if (usedLen > 0u && maxLen > 0u && nLayers > 0u && dModelKV > 0u)
		{
			const size_t perLayer = static_cast<size_t>(maxLen) * static_cast<size_t>(dModelKV);
			const size_t perSeq = static_cast<size_t>(nLayers) * perLayer;
			float* kSeq = batcher.session.k.empty() ? NULL : &batcher.session.k[static_cast<size_t>(slot) * perSeq];
			float* vSeq = batcher.session.v.empty() ? NULL : &batcher.session.v[static_cast<size_t>(slot) * perSeq];
			unsigned char* keyValidSeq =
			    batcher.session.keyValid.empty() ? NULL : &batcher.session.keyValid[static_cast<size_t>(slot) * static_cast<size_t>(maxLen)];

			const size_t prefix = static_cast<size_t>(usedLen) * static_cast<size_t>(dModelKV);
			for (unsigned int li = 0u; li < nLayers; ++li)
			{
				const size_t off = static_cast<size_t>(li) * perLayer;
				if (kSeq)
					std::fill(kSeq + off, kSeq + off + prefix, 0.0f);
				if (vSeq)
					std::fill(vSeq + off, vSeq + off + prefix, 0.0f);
			}
			if (keyValidSeq)
				std::fill(keyValidSeq, keyValidSeq + usedLen, 0u);
		}
	}

	// Release slot state.
	batcher.inUse[slot] = 0u;
	batcher.done[slot] = 0u;
	batcher.promptPos[slot] = 0u;
	batcher.promptLen[slot] = 0u;
	batcher.generated[slot] = 0u;
	batcher.reqMaxNew[slot] = 0u;
	batcher.reqMaxLen[slot] = 0u;
	if (slot < batcher.session.curLen.size())
		batcher.session.curLen[slot] = 0u;
	batcher.hasOverride[slot] = 0u;
	batcher.req[slot] = TransformerServeRequest();
	batcher.results[slot] = TransformerGenerateResult();

	// Clear logits row for hygiene.
	if (!batcher.prevLogitsFlat.empty() && batcher.vocab > 0u)
	{
		float* row = &batcher.prevLogitsFlat[static_cast<size_t>(slot) * static_cast<size_t>(batcher.vocab)];
		std::fill(row, row + batcher.vocab, 0.0f);
	}

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::NNetwork::transformerLmServeBatcherStep(glades::NNetwork::TransformerServeBatcher& batcher,
                                                                       glades::NNetwork::ITransformerServeCallbacks* cb) const
{
	if (!batcher.initialized)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmServeBatcherStep: batcher not initialized");
	if (batcher.maxBatchSize == 0u || batcher.vocab == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmServeBatcherStep: invalid batcher state");

	const unsigned int B = batcher.maxBatchSize;
	const unsigned int vocab = batcher.vocab;

	if (cb && cb->shouldStopAll(*this))
	{
		for (unsigned int s = 0u; s < B; ++s)
		{
			if (batcher.inUse[s] != 0u && batcher.done[s] == 0u)
			{
				batcher.results[s].stoppedByCallback = true;
				batcher.done[s] = 1u;
			}
		}
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	// Build one append step across the whole batch.
	unsigned int activeCount = 0u;
	for (unsigned int s = 0u; s < B; ++s)
	{
		batcher.active[s] = 0u;
		batcher.sampledIsValid[s] = 0u;

		if (batcher.inUse[s] == 0u || batcher.done[s] != 0u)
			continue;

		const unsigned int curLen = (s < batcher.session.curLen.size()) ? batcher.session.curLen[s] : 0u;
		if (curLen >= batcher.reqMaxLen[s])
		{
			batcher.results[s].stoppedByLimit = true;
			batcher.done[s] = 1u;
			continue;
		}

		// Prefill: append prompt tokens until promptPos == promptLen.
		if (batcher.promptPos[s] < batcher.promptLen[s])
		{
			const unsigned int p = batcher.promptPos[s];
			const std::vector<unsigned int>& pt = batcher.req[s].promptTokens;
			if (p >= pt.size())
			{
				// Defensive: promptLen/promptPos mismatch.
				batcher.results[s].stoppedByCallback = true;
				batcher.done[s] = 1u;
				continue;
			}
			batcher.tokenIds[s] = pt[p];
			batcher.active[s] = 1u;
			++activeCount;
			continue;
		}

		// Decode: stop checks and sampling.
		if (batcher.generated[s] >= batcher.reqMaxNew[s])
		{
			batcher.results[s].stoppedByLimit = true;
			batcher.done[s] = 1u;
			continue;
		}
		if (cb && cb->shouldStopRequest(*this, s))
		{
			batcher.results[s].stoppedByCallback = true;
			batcher.done[s] = 1u;
			continue;
		}

		const TransformerGenerateConfig& cfg = batcher.req[s].cfg;

		glades::rng::Engine* useEngine = (batcher.hasOverride[s] ? &batcher.overrideEngines[s] : &batcher.batchEngine);

		const float* row = batcher.prevLogitsFlat.empty() ? NULL : &batcher.prevLogitsFlat[static_cast<size_t>(s) * static_cast<size_t>(vocab)];
		unsigned int nextTok = 0u;
		if (!sample_token_from_logits_ptr(row, *useEngine, vocab, cfg.temperature, cfg.topK, cfg.topP, cfg.topPTopKCap, nextTok, batcher.idxScratch,
		                                  batcher.weightScratch))
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "transformerLmServeBatcherStep: failed to sample token from logits");
		if (nextTok >= vocab)
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "transformerLmServeBatcherStep: sampled tokenId out of range");

		batcher.tokenIds[s] = nextTok;
		batcher.sampledTok[s] = nextTok;
		batcher.sampledIsValid[s] = 1u;
		batcher.active[s] = 1u;
		++activeCount;
	}

	if (activeCount == 0u)
		return NNetworkStatus(NNetworkStatus::OK, std::string());

	{
		const NNetworkStatus st = transformerLmBatchSessionAppendSelective(batcher.session, batcher.tokenIds, batcher.active, &batcher.logitsFlat);
		if (!st.ok())
			return st;
	}

	// Make logitsFlat the new "current" logits for the next step (zero-copy swap).
	batcher.prevLogitsFlat.swap(batcher.logitsFlat);

	// Emit decode tokens and advance prompt cursors.
	for (unsigned int s = 0u; s < B; ++s)
	{
		if (batcher.active[s] == 0u || batcher.inUse[s] == 0u || batcher.done[s] != 0u)
			continue;

		// Prefill path.
		if (batcher.sampledIsValid[s] == 0u)
		{
			++batcher.promptPos[s];
			// If prompt just completed and no decode requested, stop now.
			if (batcher.promptPos[s] >= batcher.promptLen[s] && batcher.reqMaxNew[s] == 0u)
			{
				batcher.results[s].stoppedByLimit = true;
				batcher.done[s] = 1u;
			}
			continue;
		}

		// Decode path (token already appended to KV in this step).
		const unsigned int tok = batcher.sampledTok[s];
		TransformerGenerateResult& rr = batcher.results[s];
		rr.tokens.push_back(tok);
		rr.lastToken = tok;

		const unsigned int genIdx = batcher.generated[s];
		++batcher.generated[s];

		if (cb)
		{
			const bool stopReq = cb->onToken(*this, s, tok, genIdx);
			if (stopReq)
			{
				rr.stoppedByCallback = true;
				batcher.done[s] = 1u;
				continue;
			}
		}

		const TransformerGenerateConfig& cfg = batcher.req[s].cfg;
		const int eos = cfg.eosTokenId;
		if (cfg.stopOnEos && eos >= 0 && tok == static_cast<unsigned int>(eos))
		{
			rr.stoppedOnEos = true;
			batcher.done[s] = 1u;
			continue;
		}

		// Additional stop tokens.
		const std::vector<unsigned int>& stopTok = batcher.req[s].stopTokenIds;
		for (size_t i = 0; i < stopTok.size(); ++i)
		{
			if (tok == stopTok[i])
			{
				rr.stoppedByStopToken = true;
				batcher.done[s] = 1u;
				break;
			}
		}
		if (batcher.done[s] != 0u)
			continue;

		// Per-request max token limit.
		if (batcher.generated[s] >= batcher.reqMaxNew[s])
		{
			rr.stoppedByLimit = true;
			batcher.done[s] = 1u;
			continue;
		}
	}

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

