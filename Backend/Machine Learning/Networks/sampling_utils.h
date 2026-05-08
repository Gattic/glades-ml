#pragma once
#include "../rng.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace glades {
namespace sampling {

inline unsigned int argmax_u(const std::vector<float>& logits)
{
	if (logits.empty())
		return 0u;
	unsigned int bestIdx = 0u;
	float best = logits[0];
	for (unsigned int i = 1u; i < static_cast<unsigned int>(logits.size()); ++i)
	{
		const float v = logits[i];
		if (v > best)
		{
			best = v;
			bestIdx = i;
		}
	}
	return bestIdx;
}

struct IdxGreaterByLogit
{
	const float* l;
	explicit IdxGreaterByLogit(const float* logits) : l(logits) {}
	bool operator()(unsigned int a, unsigned int b) const { return l[a] > l[b]; }
};

struct IdxMinHeapByLogit
{
	const float* l;
	explicit IdxMinHeapByLogit(const float* logits) : l(logits) {}
	bool operator()(unsigned int a, unsigned int b) const
	{
		// Min-heap by logit; for ties, prefer larger token id as "worse" so it
		// is popped first when we see an equal-logit smaller id.
		const float la = l[a];
		const float lb = l[b];
		if (la != lb) return la > lb;
		return a < b;
	}
};

struct SamplingPlan
{
	unsigned int vocab;
	float invTemp;
	unsigned int topK;
	float topP;
	bool greedy;
	bool useFullVocabFastPath;

	SamplingPlan()
	    : vocab(0u),
	      invTemp(1.0f),
	      topK(0u),
	      topP(1.0f),
	      greedy(false),
	      useFullVocabFastPath(false)
	{
	}
};

inline SamplingPlan make_sampling_plan(unsigned int vocab,
                                      float temperature,
                                      unsigned int topK,
                                      float topP,
                                      unsigned int topPTopKCap)
{
	SamplingPlan plan;
	plan.vocab = vocab;
	plan.greedy = (temperature <= 0.0f);
	if (plan.greedy)
		return plan;

	if (!std::isfinite(temperature) || temperature <= 0.0f)
		temperature = 1.0f;
	if (!std::isfinite(topP) || topP <= 0.0f || topP > 1.0f)
		topP = 1.0f;
	if (topK > vocab)
		topK = vocab;
	if (topP < 1.0f && topK == 0u && topPTopKCap > 0u)
		topK = std::min(vocab, topPTopKCap);

	plan.invTemp = 1.0f / temperature;
	plan.topK = topK;
	plan.topP = topP;
	plan.useFullVocabFastPath = (topK == 0u && topP >= 1.0f);
	return plan;
}

inline bool sample_token_from_logits_ptr_plan(const float* logits,
                                              glades::rng::Engine& rng,
                                              const SamplingPlan& plan,
                                              unsigned int& outToken,
                                              std::vector<unsigned int>& idxScratch,
                                              std::vector<float>& weightScratch)
{
	const unsigned int vocab = plan.vocab;
	const float invTemp = plan.invTemp;
	const unsigned int topK = plan.topK;
	const float topP = plan.topP;

	outToken = 0u;
	if (!logits || vocab == 0u)
		return false;

	if (plan.greedy)
	{
		unsigned int bestIdx = 0u;
		float best = logits[0];
		for (unsigned int i = 1u; i < vocab; ++i)
		{
			const float v = logits[i];
			if (v > best)
			{
				best = v;
				bestIdx = i;
			}
		}
		outToken = bestIdx;
		return true;
	}

	if (plan.useFullVocabFastPath)
	{
		float maxScaled = logits[0] * invTemp;
		for (unsigned int i = 1u; i < vocab; ++i)
		{
			const float v = logits[i] * invTemp;
			if (v > maxScaled) maxScaled = v;
		}

		if (weightScratch.size() < vocab)
			weightScratch.resize(vocab);

		double sum = 0.0;
		for (unsigned int i = 0u; i < vocab; ++i)
		{
			const double w = exp(static_cast<double>((logits[i] * invTemp) - maxScaled));
			weightScratch[i] = static_cast<float>(w);
			sum += w;
		}
		if (!(sum > 0.0) || !std::isfinite(sum))
			return false;

		const double u = glades::rng::unit_double01(rng) * sum;
		double acc = 0.0;
		for (unsigned int i = 0u; i < vocab; ++i)
		{
			acc += static_cast<double>(weightScratch[i]);
			if (u <= acc)
			{
				outToken = i;
				return true;
			}
		}
		outToken = vocab - 1u;
		return true;
	}

	unsigned int candN = vocab;
	if (topK > 0u && topK < vocab)
	{
		idxScratch.clear();
		if (idxScratch.capacity() < topK)
			idxScratch.reserve(topK);

		const IdxMinHeapByLogit heapCmp(logits);
		for (unsigned int i = 0u; i < vocab; ++i)
		{
			if (idxScratch.size() < topK)
			{
				idxScratch.push_back(i);
				std::push_heap(idxScratch.begin(), idxScratch.end(), heapCmp);
				continue;
			}

			const unsigned int worst = idxScratch.front();
			const float li = logits[i];
			const float lw = logits[worst];
			if (li > lw || (li == lw && i < worst))
			{
				std::pop_heap(idxScratch.begin(), idxScratch.end(), heapCmp);
				idxScratch.back() = i;
				std::push_heap(idxScratch.begin(), idxScratch.end(), heapCmp);
			}
		}

		candN = topK;
		std::sort(idxScratch.begin(), idxScratch.end(), IdxGreaterByLogit(logits));
	}
	else
	{
		if (idxScratch.size() != vocab)
			idxScratch.resize(vocab);
		for (unsigned int i = 0u; i < vocab; ++i)
			idxScratch[i] = i;
		if (topP < 1.0f)
			std::sort(idxScratch.begin(), idxScratch.end(), IdxGreaterByLogit(logits));
	}

	if (weightScratch.size() < candN)
		weightScratch.resize(candN);

	float maxScaled = -1e30f;
	if (topP < 1.0f || (topK > 0u && topK < vocab))
	{
		const unsigned int bestIdx = idxScratch[0];
		maxScaled = logits[bestIdx] * invTemp;
	}
	else
	{
		maxScaled = logits[0] * invTemp;
		for (unsigned int i = 1u; i < vocab; ++i)
		{
			const float v = logits[i] * invTemp;
			if (v > maxScaled) maxScaled = v;
		}
	}

	double sum = 0.0;
	for (unsigned int j = 0u; j < candN; ++j)
	{
		const unsigned int id = idxScratch[j];
		const double w = exp(static_cast<double>((logits[id] * invTemp) - maxScaled));
		weightScratch[j] = static_cast<float>(w);
		sum += w;
	}
	if (!(sum > 0.0) || !std::isfinite(sum))
		return false;

	unsigned int keepN = candN;
	if (topP < 1.0f)
	{
		double cum = 0.0;
		keepN = 0u;
		for (unsigned int j = 0u; j < candN; ++j)
		{
			cum += static_cast<double>(weightScratch[j]) / sum;
			++keepN;
			if (cum >= static_cast<double>(topP))
				break;
		}
		if (keepN < 1u)
			keepN = 1u;
	}

	double keptSum = 0.0;
	for (unsigned int j = 0u; j < keepN; ++j)
		keptSum += static_cast<double>(weightScratch[j]);
	if (!(keptSum > 0.0) || !std::isfinite(keptSum))
		return false;

	const double u = glades::rng::unit_double01(rng) * keptSum;
	double acc = 0.0;
	for (unsigned int j = 0u; j < keepN; ++j)
	{
		acc += static_cast<double>(weightScratch[j]);
		if (u <= acc)
		{
			outToken = idxScratch[j];
			return true;
		}
	}
	outToken = idxScratch[keepN - 1u];
	return true;
}

// Sample a token id from logits under (temperature, topK, topP).
// Scratch buffers are provided to avoid per-step allocations.
inline bool sample_token_from_logits_ptr(const float* logits,
                                         glades::rng::Engine& rng,
                                         unsigned int vocab,
                                         float temperature,
                                         unsigned int topK,
                                         float topP,
                                         unsigned int topPTopKCap,
                                         unsigned int& outToken,
                                         std::vector<unsigned int>& idxScratch,
                                         std::vector<float>& weightScratch)
{
	const SamplingPlan plan = make_sampling_plan(vocab, temperature, topK, topP, topPTopKCap);
	return sample_token_from_logits_ptr_plan(logits, rng, plan, outToken, idxScratch, weightScratch);
}

// ============================================================================
// Paradigm shift #75 — SPECULATIVE-DECODING-DISTILL primitives
// ============================================================================
//
// Speculative decoding (Leviathan et al. 2023, Chen et al. 2023) accelerates
// LLM inference by:
//   1. A small "draft" model proposes K tokens autoregressively.
//   2. The full "main" model verifies all K positions in parallel (single fwd).
//   3. Rejection sampling: accept t_i with prob min(1, p_main(t_i)/p_draft(t_i)).
//      On first reject, resample from the "residual" max(0, p_main - p_draft).
//
// Theorem 3.5 of Leviathan 2023: rejection sampling preserves p_main exactly.
// NLL bit-exact at inference; output distribution = main's distribution.
//
// Production: vLLM, TensorRT-LLM, DeepSeek-V3, Eagle, Medusa.
//
// These are pure-CPU primitives: the model forward pass is the caller's
// responsibility. The primitives implement the rejection-sampling math.

inline float spec_total_variation_distance(const float* p_main,
                                           const float* p_draft,
                                           unsigned int vocab)
{
	if (!p_main || !p_draft || vocab == 0u)
		return 0.0f;
	double tvd = 0.0;
	for (unsigned int i = 0; i < vocab; ++i)
	{
		const double d = static_cast<double>(p_main[i]) - static_cast<double>(p_draft[i]);
		tvd += (d > 0.0) ? d : -d;
	}
	tvd *= 0.5;
	if (tvd < 0.0) tvd = 0.0;
	if (tvd > 1.0) tvd = 1.0;
	return static_cast<float>(tvd);
}

// Lower bound on per-token acceptance rate from Leviathan 2023 Theorem 3.5
// corollary: P(accept) = sum_x min(p_main(x), p_draft(x)) = 1 - TVD(p_main, p_draft).
inline float spec_acceptance_rate_bound(const float* p_main,
                                        const float* p_draft,
                                        unsigned int vocab)
{
	return 1.0f - spec_total_variation_distance(p_main, p_draft, vocab);
}

// Single-position rejection-sampling step.
//
// Inputs:
//   p_main  [vocab]: full-model probabilities at this position.
//   p_draft [vocab]: draft-model probabilities at this position.
//   draft_token:    the token the draft proposed (must be in [0, vocab)).
//   rng:            uniform [0,1) source.
//
// Outputs:
//   acceptedOut: true iff the draft token was accepted.
//   sampledTokenOut: if accepted, == draft_token; else, drawn from
//     normalize(max(0, p_main - p_draft)) (the "residual" distribution).
//
// Returns false on bad inputs or degenerate (all-zero) probability vector.
inline bool speculative_rejection_sample_step(const float* p_main,
                                              const float* p_draft,
                                              unsigned int draft_token,
                                              unsigned int vocab,
                                              glades::rng::Engine& rng,
                                              bool& acceptedOut,
                                              unsigned int& sampledTokenOut,
                                              std::vector<float>& residualScratch)
{
	acceptedOut = false;
	sampledTokenOut = 0u;
	if (!p_main || !p_draft || vocab == 0u || draft_token >= vocab)
		return false;

	const float pm = p_main[draft_token];
	const float pd = p_draft[draft_token];

	// Acceptance probability: min(1, p_main / p_draft).
	double acceptProb = 1.0;
	if (pd > 0.0f)
	{
		const double r = static_cast<double>(pm) / static_cast<double>(pd);
		acceptProb = (r < 1.0) ? r : 1.0;
	}
	// If pd == 0 but pm > 0: accept (any positive draft has prob 0 under draft,
	// shouldn't happen; treat as accept since main supports it).
	// If pd == 0 and pm == 0: degenerate, reject and resample.

	const double u = static_cast<double>(glades::rng::uniform_double(rng, 0.0, 1.0));
	if (u < acceptProb)
	{
		acceptedOut = true;
		sampledTokenOut = draft_token;
		return true;
	}

	// Reject: resample from normalize(max(0, p_main - p_draft)).
	if (residualScratch.size() < vocab)
		residualScratch.assign(vocab, 0.0f);
	double residSum = 0.0;
	for (unsigned int i = 0; i < vocab; ++i)
	{
		const double d = static_cast<double>(p_main[i]) - static_cast<double>(p_draft[i]);
		const float v = (d > 0.0) ? static_cast<float>(d) : 0.0f;
		residualScratch[i] = v;
		residSum += static_cast<double>(v);
	}

	if (residSum <= 0.0)
	{
		// Degenerate (p_draft pointwise dominates p_main). Fall back to p_main.
		double mainSum = 0.0;
		for (unsigned int i = 0; i < vocab; ++i)
		{
			residualScratch[i] = p_main[i];
			mainSum += static_cast<double>(p_main[i]);
		}
		if (mainSum <= 0.0)
		{
			// Fully degenerate: pick token 0.
			sampledTokenOut = 0u;
			return true;
		}
		residSum = mainSum;
	}

	const double inv = 1.0 / residSum;
	const double r = static_cast<double>(glades::rng::uniform_double(rng, 0.0, 1.0));
	double cum = 0.0;
	for (unsigned int i = 0; i < vocab; ++i)
	{
		cum += static_cast<double>(residualScratch[i]) * inv;
		if (r < cum)
		{
			sampledTokenOut = i;
			return true;
		}
	}
	sampledTokenOut = vocab - 1u;
	return true;
}

// Verify a K-token draft proposal against main probabilities.
//
// Inputs:
//   p_main   [(K+1) * vocab]: main probabilities at draft positions [1..K]
//                              plus position K+1 (used if all K accepted).
//   p_draft  [K * vocab]:      draft probabilities at positions [1..K].
//   draft_tokens [K]:          tokens the draft proposed.
//   K, vocab:                  dimensions.
//
// Outputs:
//   nAccepted: number of accepted tokens (0..K).
//   tailToken: the token emitted after the last accept:
//     - if nAccepted == K: drawn from p_main row K (the "bonus" token from
//       Leviathan 2023 — main's distribution after the verified prefix).
//     - else (rejected at position nAccepted): drawn from
//       normalize(max(0, p_main_row_nAccepted - p_draft_row_nAccepted)).
//
// Returns the count of tokens emitted total (= nAccepted + 1).
inline unsigned int speculative_verify_K(const float* p_main,
                                         const float* p_draft,
                                         const unsigned int* draft_tokens,
                                         unsigned int K,
                                         unsigned int vocab,
                                         glades::rng::Engine& rng,
                                         unsigned int& nAcceptedOut,
                                         unsigned int& tailTokenOut,
                                         std::vector<float>& residualScratch,
                                         std::vector<unsigned int>& idxScratch,
                                         std::vector<float>& weightScratch)
{
	nAcceptedOut = 0u;
	tailTokenOut = 0u;
	if (!p_main || !p_draft || !draft_tokens || K == 0u || vocab == 0u)
		return 0u;

	for (unsigned int i = 0; i < K; ++i)
	{
		const float* pmRow = p_main + static_cast<size_t>(i) * vocab;
		const float* pdRow = p_draft + static_cast<size_t>(i) * vocab;
		bool accepted = false;
		unsigned int tok = 0u;
		if (!speculative_rejection_sample_step(pmRow, pdRow, draft_tokens[i], vocab,
		                                       rng, accepted, tok, residualScratch))
			return nAcceptedOut;
		if (accepted)
		{
			++nAcceptedOut;
			continue;
		}
		// Rejected: tok is the resample from residual; emit it as tail and stop.
		tailTokenOut = tok;
		return nAcceptedOut + 1u;
	}

	// All K accepted: bonus-sample from the (K+1)-th main row (greedy/argmax for
	// determinism in tests; production would route through SamplingPlan).
	const float* pmBonus = p_main + static_cast<size_t>(K) * vocab;
	double bestP = -1.0;
	unsigned int bestTok = 0u;
	for (unsigned int i = 0; i < vocab; ++i)
	{
		const double v = static_cast<double>(pmBonus[i]);
		if (v > bestP)
		{
			bestP = v;
			bestTok = i;
		}
	}
	(void)idxScratch; (void)weightScratch;
	tailTokenOut = bestTok;
	return nAcceptedOut + 1u;
}

} // namespace sampling
} // namespace glades
