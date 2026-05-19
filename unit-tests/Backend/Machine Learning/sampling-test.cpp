// Sampling function unit tests.
//
// Tests greedy, temperature, top-k, top-p, combined modes, edge cases, and
// determinism of sample_token_from_logits_ptr in sampling_utils.h.

#include "sampling-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/sampling_utils.h"
#include "../../../Backend/Machine Learning/rng.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Helper: call sample N times, return token histogram.
static std::vector<unsigned int> sample_histogram(const float* logits, unsigned int vocab,
                                                  float temperature, unsigned int topK, float topP,
                                                  unsigned int topPTopKCap, unsigned int N,
                                                  uint64_t baseSeed)
{
	std::vector<unsigned int> hist(vocab, 0u);
	std::vector<unsigned int> idxScratch;
	std::vector<float> weightScratch;
	for (unsigned int i = 0; i < N; ++i)
	{
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, baseSeed + static_cast<uint64_t>(i));
		unsigned int tok = 0u;
		bool ok = glades::sampling::sample_token_from_logits_ptr(
		    logits, rng, vocab, temperature, topK, topP, topPTopKCap,
		    tok, idxScratch, weightScratch);
		if (ok && tok < vocab)
			hist[tok]++;
	}
	return hist;
}

} // namespace

void SamplingUnitTest()
{
	printf("============================================================\n");
	printf("Sampling Test Suite\n");
	printf("============================================================\n");

	// Scratch buffers reused across tests.
	std::vector<unsigned int> idxScratch;
	std::vector<float> weightScratch;

	// --- Test 1: Greedy returns argmax ---
	{
		printf("-----------------------------------\n");
		printf("Greedy returns argmax\n");
		printf("-----------------------------------\n");
		float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 42);
		unsigned int tok = 99u;
		bool ok = glades::sampling::sample_token_from_logits_ptr(
		    logits, rng, 4, 0.0f, 0, 1.0f, 256, tok, idxScratch, weightScratch);
		ASSERT("greedy should succeed", ok);
		ASSERT("greedy should return argmax (1)", tok == 1u);
	}

	// --- Test 2: Negative temperature acts as greedy ---
	{
		printf("-----------------------------------\n");
		printf("Negative temperature acts as greedy\n");
		printf("-----------------------------------\n");
		float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 42);
		unsigned int tok = 99u;
		bool ok = glades::sampling::sample_token_from_logits_ptr(
		    logits, rng, 4, -1.0f, 0, 1.0f, 256, tok, idxScratch, weightScratch);
		ASSERT("neg temp should succeed", ok);
		ASSERT("neg temp should return argmax (1)", tok == 1u);
	}

	// --- Test 3: topK=1 effectively greedy ---
	{
		printf("-----------------------------------\n");
		printf("topK=1 effectively greedy\n");
		printf("-----------------------------------\n");
		float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
		for (unsigned int s = 0; s < 10; ++s)
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 100 + s);
			unsigned int tok = 99u;
			bool ok = glades::sampling::sample_token_from_logits_ptr(
			    logits, rng, 4, 1.0f, 1, 1.0f, 256, tok, idxScratch, weightScratch);
			ASSERT("topK=1 should succeed", ok);
			ASSERT("topK=1 should always return argmax (1)", tok == 1u);
		}
	}

	// --- Test 4: topK only selects from top-K ---
	{
		printf("-----------------------------------\n");
		printf("topK only selects from top-K\n");
		printf("-----------------------------------\n");
		float logits[] = {10.0f, 9.0f, 1.0f, 1.0f, 0.0f};
		std::vector<unsigned int> hist = sample_histogram(logits, 5, 1.0f, 2, 1.0f, 256, 1000, 200);
		unsigned int outsideTopK = hist[2] + hist[3] + hist[4];
		ASSERT("topK=2: no tokens outside top-2", outsideTopK == 0u);
		ASSERT("topK=2: token 0 selected at least once", hist[0] > 0u);
		ASSERT("topK=2: token 1 selected at least once", hist[1] > 0u);
	}

	// --- Test 5: topP narrow nucleus ---
	{
		printf("-----------------------------------\n");
		printf("topP narrow nucleus\n");
		printf("-----------------------------------\n");
		float logits[] = {10.0f, 1.0f, 1.0f, 1.0f};
		std::vector<unsigned int> hist = sample_histogram(logits, 4, 1.0f, 0, 0.01f, 256, 1000, 300);
		float frac0 = static_cast<float>(hist[0]) / 1000.0f;
		ASSERT("narrow topP: token 0 > 95%", frac0 > 0.95f);
	}

	// --- Test 6: topP=1.0 keeps all tokens ---
	{
		printf("-----------------------------------\n");
		printf("topP=1.0 keeps all tokens (uniform logits)\n");
		printf("-----------------------------------\n");
		float logits[] = {1.0f, 1.0f, 1.0f, 1.0f};
		std::vector<unsigned int> hist = sample_histogram(logits, 4, 1.0f, 0, 1.0f, 256, 10000, 400);
		for (unsigned int i = 0; i < 4; ++i)
		{
			float frac = static_cast<float>(hist[i]) / 10000.0f;
			ASSERT("uniform: each token 15-35%", frac > 0.15f && frac < 0.35f);
		}
	}

	// --- Test 7: High temperature flattens distribution ---
	{
		printf("-----------------------------------\n");
		printf("High temperature flattens distribution\n");
		printf("-----------------------------------\n");
		float logits[] = {10.0f, 0.0f, 0.0f};
		std::vector<unsigned int> hist = sample_histogram(logits, 3, 100.0f, 0, 1.0f, 256, 10000, 500);
		for (unsigned int i = 0; i < 3; ++i)
		{
			float frac = static_cast<float>(hist[i]) / 10000.0f;
			ASSERT("high temp: each token > 20%", frac > 0.20f);
		}
	}

	// --- Test 8: Low temperature sharpens distribution ---
	{
		printf("-----------------------------------\n");
		printf("Low temperature sharpens distribution\n");
		printf("-----------------------------------\n");
		float logits[] = {10.0f, 9.9f, 0.0f};
		std::vector<unsigned int> hist = sample_histogram(logits, 3, 0.01f, 0, 1.0f, 256, 1000, 600);
		float frac0 = static_cast<float>(hist[0]) / 1000.0f;
		ASSERT("low temp: token 0 > 90%", frac0 > 0.90f);
	}

	// --- Test 9: vocab=1 returns 0 ---
	{
		printf("-----------------------------------\n");
		printf("vocab=1 returns 0\n");
		printf("-----------------------------------\n");
		float logits[] = {5.0f};
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 42);
		unsigned int tok = 99u;
		bool ok = glades::sampling::sample_token_from_logits_ptr(
		    logits, rng, 1, 1.0f, 0, 1.0f, 256, tok, idxScratch, weightScratch);
		ASSERT("vocab=1 should succeed", ok);
		ASSERT("vocab=1 should return 0", tok == 0u);
	}

	// --- Test 10: All equal logits ---
	{
		printf("-----------------------------------\n");
		printf("All equal logits uniform distribution\n");
		printf("-----------------------------------\n");
		float logits[] = {0.0f, 0.0f, 0.0f, 0.0f};
		std::vector<unsigned int> hist = sample_histogram(logits, 4, 1.0f, 0, 1.0f, 256, 10000, 700);
		for (unsigned int i = 0; i < 4; ++i)
		{
			float frac = static_cast<float>(hist[i]) / 10000.0f;
			ASSERT("equal logits: each token 15-35%", frac > 0.15f && frac < 0.35f);
		}
	}

	// --- Test 11: Determinism ---
	{
		printf("-----------------------------------\n");
		printf("Determinism: same seed same output\n");
		printf("-----------------------------------\n");
		float logits[] = {3.0f, 1.0f, 2.0f, 4.0f, 0.5f};
		for (unsigned int trial = 0; trial < 20; ++trial)
		{
			const uint64_t seed = 8000 + trial;
			unsigned int tok1 = 99u, tok2 = 99u;
			{
				glades::rng::Engine rng;
				glades::rng::seed_engine(rng, seed);
				glades::sampling::sample_token_from_logits_ptr(
				    logits, rng, 5, 0.8f, 3, 0.9f, 256, tok1, idxScratch, weightScratch);
			}
			{
				glades::rng::Engine rng;
				glades::rng::seed_engine(rng, seed);
				glades::sampling::sample_token_from_logits_ptr(
				    logits, rng, 5, 0.8f, 3, 0.9f, 256, tok2, idxScratch, weightScratch);
			}
			ASSERT("determinism: same seed same token", tok1 == tok2);
		}
	}

	// --- Test 12: topK + topP combined ---
	{
		printf("-----------------------------------\n");
		printf("topK + topP combined\n");
		printf("-----------------------------------\n");
		float logits[] = {10.0f, 9.0f, 8.0f, 1.0f, 1.0f};
		std::vector<unsigned int> hist = sample_histogram(logits, 5, 1.0f, 3, 0.5f, 256, 1000, 900);
		unsigned int outsideTopK = hist[3] + hist[4];
		ASSERT("topK+topP: no tokens outside top-3", outsideTopK == 0u);
		// With topP=0.5 within top-3, mostly tokens 0 and 1
		float fracTop2 = static_cast<float>(hist[0] + hist[1]) / 1000.0f;
		ASSERT("topK+topP: tokens 0,1 dominate (>70%)", fracTop2 > 0.70f);
	}

	// --- Test 13: NULL logits returns false ---
	{
		printf("-----------------------------------\n");
		printf("NULL logits returns false\n");
		printf("-----------------------------------\n");
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 42);
		unsigned int tok = 99u;
		bool ok = glades::sampling::sample_token_from_logits_ptr(
		    NULL, rng, 5, 1.0f, 0, 1.0f, 256, tok, idxScratch, weightScratch);
		ASSERT("NULL logits should fail", !ok);
	}

	// --- Test 14: vocab=0 returns false ---
	{
		printf("-----------------------------------\n");
		printf("vocab=0 returns false\n");
		printf("-----------------------------------\n");
		float logits[] = {1.0f};
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 42);
		unsigned int tok = 99u;
		bool ok = glades::sampling::sample_token_from_logits_ptr(
		    logits, rng, 0, 1.0f, 0, 1.0f, 256, tok, idxScratch, weightScratch);
		ASSERT("vocab=0 should fail", !ok);
	}

	// --- Test 15: topPTopKCap effective capping ---
	{
		printf("-----------------------------------\n");
		printf("topPTopKCap effective capping\n");
		printf("-----------------------------------\n");
		std::vector<float> logits(100, 1.0f);
		logits[0] = 10.0f;
		std::vector<unsigned int> hist = sample_histogram(&logits[0], 100, 1.0f, 0, 0.9f, 10, 100, 1000);
		unsigned int total = 0;
		for (unsigned int i = 0; i < 100; ++i)
			total += hist[i];
		ASSERT("topPTopKCap: all samples valid", total == 100u);
	}

	printf("============================================================\n");
	printf("All Sampling Tests Passed\n");
	printf("============================================================\n");
}
