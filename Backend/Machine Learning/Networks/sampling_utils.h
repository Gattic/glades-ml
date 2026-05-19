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

} // namespace sampling
} // namespace glades
