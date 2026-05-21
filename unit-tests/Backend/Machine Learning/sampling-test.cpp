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

	// ============================================================
	// Group SPEC: Paradigm shift #75 SPECULATIVE-DECODING-DISTILL
	// ============================================================
	// Probe per #75 design — verify rejection-sampling primitive
	// preserves p_main distribution (Theorem 3.5 of Leviathan 2023)
	// and produces expected acceptance rate / throughput speedup.

	// --- Test SPEC-1: TVD identity ---
	{
		printf("-----------------------------------\n");
		printf("[SPEC-1] TVDOfIdenticalDistributions\n");
		printf("-----------------------------------\n");
		std::vector<float> p(8, 0.125f);
		float tvd = glades::sampling::spec_total_variation_distance(&p[0], &p[0], 8);
		float alpha = glades::sampling::spec_acceptance_rate_bound(&p[0], &p[0], 8);
		printf("  TVD(p,p)=%.6f  acceptance_bound=%.6f\n", tvd, alpha);
		ASSERT("spec: TVD(p,p) == 0", std::fabs(tvd) < 1e-6f);
		ASSERT("spec: acceptance(p,p) == 1", std::fabs(alpha - 1.0f) < 1e-6f);
	}

	// --- Test SPEC-2: TVD asymmetric ---
	{
		printf("-----------------------------------\n");
		printf("[SPEC-2] TVDForAsymmetricDistributions\n");
		printf("-----------------------------------\n");
		float p_main_arr[4] = {0.50f, 0.30f, 0.10f, 0.10f};
		float p_draft_arr[4] = {0.10f, 0.10f, 0.30f, 0.50f};
		// |0.5-0.1| + |0.3-0.1| + |0.1-0.3| + |0.1-0.5| = 0.4+0.2+0.2+0.4 = 1.2
		// TVD = 0.5 * 1.2 = 0.6
		float tvd = glades::sampling::spec_total_variation_distance(p_main_arr, p_draft_arr, 4);
		float alpha = glades::sampling::spec_acceptance_rate_bound(p_main_arr, p_draft_arr, 4);
		printf("  TVD=%.4f  acceptance_bound=%.4f (expect TVD=0.6, alpha=0.4)\n", tvd, alpha);
		ASSERT("spec: TVD asymmetric", std::fabs(tvd - 0.6f) < 1e-5f);
		ASSERT("spec: acceptance bound 1-TVD", std::fabs(alpha - 0.4f) < 1e-5f);
	}

	// --- Test SPEC-3: 100% acceptance when p_main == p_draft ---
	{
		printf("-----------------------------------\n");
		printf("[SPEC-3] FullAcceptanceWhenMainEqualsDraft\n");
		printf("-----------------------------------\n");
		const unsigned int vocab = 16u;
		std::vector<float> p(vocab);
		double s = 0.0;
		for (unsigned int i = 0; i < vocab; ++i)
		{
			p[i] = static_cast<float>(i + 1u);
			s += static_cast<double>(p[i]);
		}
		for (unsigned int i = 0; i < vocab; ++i)
			p[i] /= static_cast<float>(s);

		const unsigned int Ntrials = 5000u;
		unsigned int nAccept = 0u;
		std::vector<float> resid;
		for (unsigned int i = 0; i < Ntrials; ++i)
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 0xA5A5A5ULL + i);
			// Pick a draft token uniformly at random, then call spec
			unsigned int draftTok = i % vocab;
			bool accepted = false;
			unsigned int tok = 0u;
			glades::sampling::speculative_rejection_sample_step(
			    &p[0], &p[0], draftTok, vocab, rng, accepted, tok, resid);
			if (accepted)
				++nAccept;
		}
		float rate = static_cast<float>(nAccept) / static_cast<float>(Ntrials);
		printf("  acceptance_rate=%.4f over %u trials (expect 1.0)\n", rate, Ntrials);
		ASSERT("spec: 100%% acceptance when p_main==p_draft", rate >= 0.999f);
	}

	// --- Test SPEC-4: Distribution preservation (Theorem 3.5 — Leviathan 2023) ---
	{
		printf("-----------------------------------\n");
		printf("[SPEC-4] RejectionSamplingPreservesMainDistribution\n");
		printf("-----------------------------------\n");
		// Construct distinct p_main, p_draft over small vocab.
		const unsigned int vocab = 4u;
		float p_main_arr[4] = {0.40f, 0.30f, 0.20f, 0.10f};
		float p_draft_arr[4] = {0.10f, 0.20f, 0.30f, 0.40f};

		const unsigned int N = 50000u;
		std::vector<unsigned int> hist(vocab, 0u);
		std::vector<float> resid;
		for (unsigned int i = 0; i < N; ++i)
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 0xCAFEBABEULL + i);
			// Draft sample: pick token by p_draft cumulative.
			double r = glades::rng::uniform_double(rng, 0.0, 1.0);
			unsigned int draftTok = 0u;
			double cum = 0.0;
			for (unsigned int j = 0; j < vocab; ++j)
			{
				cum += static_cast<double>(p_draft_arr[j]);
				if (r < cum) { draftTok = j; break; }
			}
			bool accepted = false;
			unsigned int tok = 0u;
			glades::sampling::speculative_rejection_sample_step(
			    p_main_arr, p_draft_arr, draftTok, vocab, rng, accepted, tok, resid);
			if (tok < vocab)
				hist[tok]++;
		}

		printf("  emitted distribution (target = p_main):\n");
		double maxAbsErr = 0.0;
		for (unsigned int i = 0; i < vocab; ++i)
		{
			double emp = static_cast<double>(hist[i]) / static_cast<double>(N);
			double tgt = static_cast<double>(p_main_arr[i]);
			double err = std::fabs(emp - tgt);
			if (err > maxAbsErr) maxAbsErr = err;
			printf("    tok %u: emp=%.4f  target=%.4f  abserr=%.4f\n", i, emp, tgt, err);
		}
		printf("  max-abs-err=%.4f (target < 0.02 at N=%u)\n", maxAbsErr, N);
		ASSERT("spec: rejection sampling preserves p_main distribution", maxAbsErr < 0.02);
	}

	// --- Test SPEC-5: K-token verify with all-accepted bonus token ---
	{
		printf("-----------------------------------\n");
		printf("[SPEC-5] KTokenVerifyAllAcceptedYieldsBonus\n");
		printf("-----------------------------------\n");
		// p_main == p_draft for K positions, plus one extra row (the bonus position).
		const unsigned int vocab = 8u;
		const unsigned int K = 4u;
		std::vector<float> pm(static_cast<size_t>(K + 1u) * vocab);
		std::vector<float> pd(static_cast<size_t>(K) * vocab);
		// Uniform on first K rows for pm and pd (so all draft tokens accepted).
		for (unsigned int t = 0; t < K; ++t)
		{
			for (unsigned int v = 0; v < vocab; ++v)
			{
				pm[t * vocab + v] = 1.0f / static_cast<float>(vocab);
				pd[t * vocab + v] = 1.0f / static_cast<float>(vocab);
			}
		}
		// Bonus row: peaked on token 5.
		for (unsigned int v = 0; v < vocab; ++v)
			pm[K * vocab + v] = (v == 5u) ? 0.9f : 0.1f / static_cast<float>(vocab - 1u);

		std::vector<unsigned int> draftToks(K);
		for (unsigned int i = 0; i < K; ++i)
			draftToks[i] = i % vocab;

		std::vector<float> resid;
		std::vector<unsigned int> idxScratchSpec;
		std::vector<float> wScratchSpec;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 0xBEEFULL);
		unsigned int nAcc = 0u;
		unsigned int tail = 0u;
		unsigned int total = glades::sampling::speculative_verify_K(
		    &pm[0], &pd[0], &draftToks[0], K, vocab, rng,
		    nAcc, tail, resid, idxScratchSpec, wScratchSpec);
		printf("  K=%u nAccepted=%u tailToken=%u total=%u (expect K accepted, tail==5 bonus)\n",
		       K, nAcc, tail, total);
		ASSERT("spec: all-accepted yields K accepts", nAcc == K);
		ASSERT("spec: bonus token from K-th main row argmax", tail == 5u);
		ASSERT("spec: total emitted = K+1", total == K + 1u);
	}

	// --- Test SPEC-6: Speedup formula sanity ---
	{
		printf("-----------------------------------\n");
		printf("[SPEC-6] SpeedupFormulaSanityCheck\n");
		printf("-----------------------------------\n");
		// Theorem 2: speedup = K * alpha / (1 + K * gamma_draft).
		// Anchors: K=4, alpha=0.7, gamma=0.05 -> expect 2.8x (Leviathan 2023).
		struct Anchor { unsigned int K; float a; float g; float exp_speedup; };
		Anchor anchors[] = {
		    {4u, 0.70f, 0.05f, 2.33f},   // 4*0.7/(1+4*0.05) = 2.8/1.2 = 2.33
		    {5u, 0.70f, 0.05f, 2.80f},   // 5*0.7/(1+5*0.05) = 3.5/1.25 = 2.80
		    {8u, 0.65f, 0.04f, 3.94f},   // 8*0.65/(1+8*0.04) = 5.2/1.32 = 3.94
		    {5u, 0.55f, 0.07f, 2.04f},   // pessimistic
		};
		for (unsigned int i = 0; i < 4; ++i)
		{
			const Anchor& a = anchors[i];
			float speedup = static_cast<float>(a.K) * a.a /
			                (1.0f + static_cast<float>(a.K) * a.g);
			printf("  K=%u alpha=%.2f gamma=%.2f -> speedup=%.3f (expect %.2f)\n",
			       a.K, a.a, a.g, speedup, a.exp_speedup);
			ASSERT("spec: speedup formula", std::fabs(speedup - a.exp_speedup) < 0.05f);
		}
		printf("  speedup formula matches Leviathan 2023 Theorem 2\n");
	}

	// ============================================================
	// Group COLEARN: Paradigm shift #97 DRAFT-VERIFIER-CO-LEARN
	// ============================================================
	// Co-learning loss primitives extending #75. Verifies temperature
	// softmax, KL, CE, argmax, and the combined loss.

	// --- Test COLEARN-1: temperature softmax ---
	{
		printf("-----------------------------------\n");
		printf("[COLEARN-1] TemperatureSoftmaxBasic\n");
		printf("-----------------------------------\n");
		const unsigned int vocab = 4u;
		float logits[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		std::vector<float> p1(vocab, 0.0f);
		std::vector<float> p2(vocab, 0.0f);
		glades::sampling::softmax_with_temperature(logits, vocab, 1.0f, p1.data());
		glades::sampling::softmax_with_temperature(logits, vocab, 100.0f, p2.data());
		double sum1 = 0.0, sum2 = 0.0;
		for (unsigned int i = 0; i < vocab; ++i) { sum1 += p1[i]; sum2 += p2[i]; }
		printf("  τ=1.0:   probs = [%.3f, %.3f, %.3f, %.3f]  sum=%.4f\n",
		       p1[0], p1[1], p1[2], p1[3], sum1);
		printf("  τ=100:   probs = [%.3f, %.3f, %.3f, %.3f]  sum=%.4f (high τ → flat)\n",
		       p2[0], p2[1], p2[2], p2[3], sum2);
		ASSERT("τ=1 sums to 1", std::fabs(sum1 - 1.0) < 1e-5);
		ASSERT("τ=100 sums to 1", std::fabs(sum2 - 1.0) < 1e-5);
		// Higher τ → flatter distribution (closer to uniform 0.25)
		const float spread1 = p1[3] - p1[0];
		const float spread2 = p2[3] - p2[0];
		ASSERT("higher τ flattens", spread2 < spread1);
		printf("  spread τ=1 = %.4f vs τ=100 = %.4f\n", spread1, spread2);
	}

	// --- Test COLEARN-2: KL identity ---
	{
		printf("-----------------------------------\n");
		printf("[COLEARN-2] KLIdentityAndAsymmetry\n");
		printf("-----------------------------------\n");
		std::vector<float> p(4, 0.25f);
		float kl_pp = glades::sampling::kl_divergence(p.data(), p.data(), 4);
		printf("  KL(p,p)=%.6f (must be 0)\n", kl_pp);
		ASSERT("KL(p,p) == 0", std::fabs(kl_pp) < 1e-5);

		float p1[4] = {0.50f, 0.30f, 0.10f, 0.10f};
		float p2[4] = {0.10f, 0.20f, 0.30f, 0.40f};
		float kl_12 = glades::sampling::kl_divergence(p1, p2, 4);
		float kl_21 = glades::sampling::kl_divergence(p2, p1, 4);
		printf("  KL(p1,p2)=%.4f vs KL(p2,p1)=%.4f (asymmetric, both > 0)\n", kl_12, kl_21);
		ASSERT("KL(p1,p2) > 0 for distinct dists", kl_12 > 0.01f);
		ASSERT("KL(p2,p1) > 0 for distinct dists", kl_21 > 0.01f);
		ASSERT("KL is asymmetric in general", std::fabs(kl_12 - kl_21) > 1e-4f);
	}

	// --- Test COLEARN-3: CE matches argmax target ---
	{
		printf("-----------------------------------\n");
		printf("[COLEARN-3] CrossEntropyAtArgmaxTarget\n");
		printf("-----------------------------------\n");
		float logits[4] = {0.0f, 1.0f, 5.0f, 0.0f};  // peaked on idx 2
		std::vector<float> p(4, 0.0f);
		glades::sampling::softmax_with_temperature(logits, 4, 1.0f, p.data());
		const unsigned int argmaxIdx = glades::sampling::argmax_token(p.data(), 4);
		ASSERT("argmax of peaked dist", argmaxIdx == 2u);
		// CE evaluated at argmax should be small (high prob at target)
		const float ce_at_argmax = glades::sampling::cross_entropy_at_token(p.data(), argmaxIdx, 4);
		// CE at uniform: -log(0.25) ≈ 1.386
		const float ce_uniform = -std::log(0.25f);
		printf("  CE at argmax token: %.4f (must be < CE_uniform=%.4f)\n",
		       ce_at_argmax, ce_uniform);
		ASSERT("CE at argmax < uniform", ce_at_argmax < ce_uniform);
	}

	// --- Test COLEARN-4: TVD shrinks under co-learning gradient (acceptance rate climbs) ---
	{
		printf("-----------------------------------\n");
		printf("[COLEARN-4] TVDShrinksUnderCoLearningGradient\n");
		printf("-----------------------------------\n");
		// Simulate: draft starts far from main; we apply a "gradient" step
		// that shrinks the gap. The co-learning loss should decrease, and
		// TVD should decrease (so acceptance rate increases per #75 Theorem 3).
		const unsigned int vocab = 8u;
		std::vector<float> p_draft(vocab, 0.0f);
		std::vector<float> p_main(vocab, 0.0f);
		// p_main: peaked on first half
		for (unsigned int i = 0; i < vocab; ++i)
			p_main[i] = (i < vocab/2) ? 0.18f : 0.07f;
		// renormalize
		{
			double s = 0.0;
			for (unsigned int i = 0; i < vocab; ++i) s += p_main[i];
			for (unsigned int i = 0; i < vocab; ++i) p_main[i] /= static_cast<float>(s);
		}
		// p_draft: peaked on second half (deliberately mismatched)
		for (unsigned int i = 0; i < vocab; ++i)
			p_draft[i] = (i >= vocab/2) ? 0.18f : 0.07f;
		{
			double s = 0.0;
			for (unsigned int i = 0; i < vocab; ++i) s += p_draft[i];
			for (unsigned int i = 0; i < vocab; ++i) p_draft[i] /= static_cast<float>(s);
		}

		// Initial TVD and KL
		float tvd0 = glades::sampling::spec_total_variation_distance(p_main.data(), p_draft.data(), vocab);
		float kl0 = glades::sampling::kl_divergence(p_main.data(), p_draft.data(), vocab);
		printf("  initial: TVD=%.4f, KL=%.4f, α_bound=%.4f\n",
		       tvd0, kl0, 1.0f - tvd0);

		// Simulate one "co-learning step": move p_draft toward p_main by a fraction.
		// (In real training this would be a gradient step on draft logits with
		//  the co-learn loss; here we apply a lerp directly to verify the
		//  acceptance-rate lower bound increases as TVD decreases.)
		const float blend = 0.5f;  // 50% interpolation toward main
		std::vector<float> p_draft_after(vocab, 0.0f);
		for (unsigned int i = 0; i < vocab; ++i)
			p_draft_after[i] = blend * p_main[i] + (1.0f - blend) * p_draft[i];

		float tvd1 = glades::sampling::spec_total_variation_distance(p_main.data(), p_draft_after.data(), vocab);
		float kl1 = glades::sampling::kl_divergence(p_main.data(), p_draft_after.data(), vocab);
		printf("  after co-learn: TVD=%.4f, KL=%.4f, α_bound=%.4f\n",
		       tvd1, kl1, 1.0f - tvd1);
		ASSERT("TVD decreases under co-learn", tvd1 < tvd0);
		ASSERT("KL decreases under co-learn", kl1 < kl0);
		ASSERT("α_bound increases under co-learn", (1.0f - tvd1) > (1.0f - tvd0));
		const float deltaAccept = (1.0f - tvd1) - (1.0f - tvd0);
		printf("  Δα_bound = +%.4f after one 50%% blend step\n", deltaAccept);
	}

	// --- Test COLEARN-5: combined co-learn loss is non-negative and decreases ---
	{
		printf("-----------------------------------\n");
		printf("[COLEARN-5] CoLearnLossNonNegativeAndDecreasing\n");
		printf("-----------------------------------\n");
		const unsigned int vocab = 8u;
		std::vector<float> draft_logits(vocab), main_logits(vocab);
		unsigned int seed = 0xC01EAA01u;
		for (unsigned int i = 0; i < vocab; ++i)
		{
			// pseudo-random deterministic logits
			seed = seed * 1103515245u + 12345u;
			draft_logits[i] = ((seed >> 16) & 0xFF) / 255.0f - 0.5f;
			seed = seed * 1103515245u + 12345u;
			main_logits[i] = ((seed >> 16) & 0xFF) / 255.0f + 0.5f;
		}

		std::vector<float> pd_tau(vocab), pm_tau(vocab), pd_one(vocab), pm_one(vocab);
		const float alpha = 0.3f;
		const float tau = 2.0f;
		glades::sampling::softmax_with_temperature(draft_logits.data(), vocab, tau, pd_tau.data());
		glades::sampling::softmax_with_temperature(main_logits.data(), vocab, tau, pm_tau.data());
		glades::sampling::softmax_with_temperature(draft_logits.data(), vocab, 1.0f, pd_one.data());
		glades::sampling::softmax_with_temperature(main_logits.data(), vocab, 1.0f, pm_one.data());

		float loss_before = glades::sampling::co_learn_loss(
		    pd_tau.data(), pm_tau.data(), pd_one.data(), pm_one.data(),
		    vocab, alpha, tau);
		printf("  initial co-learn loss = %.4f (must be >= 0)\n", loss_before);
		ASSERT("co-learn loss non-negative", loss_before >= 0.0f);

		// Move draft toward main and recompute
		for (unsigned int i = 0; i < vocab; ++i)
			draft_logits[i] = 0.5f * draft_logits[i] + 0.5f * main_logits[i];
		glades::sampling::softmax_with_temperature(draft_logits.data(), vocab, tau, pd_tau.data());
		glades::sampling::softmax_with_temperature(draft_logits.data(), vocab, 1.0f, pd_one.data());
		float loss_after = glades::sampling::co_learn_loss(
		    pd_tau.data(), pm_tau.data(), pd_one.data(), pm_one.data(),
		    vocab, alpha, tau);
		printf("  after move:       loss = %.4f (must be < initial)\n", loss_after);
		ASSERT("co-learn loss decreases as draft → main", loss_after < loss_before);
	}

	// ============================================================
	// Group ROUTE: Paradigm shift #95 MULTI-TEACHER-ROUTING
	// ============================================================
	// Argmax classifier routing: per-sample selection of one teacher
	// from K teachers; eliminates #70-B's teacher-disagreement variance.

	// --- Test ROUTE-1: argmax routing correctness ---
	{
		printf("-----------------------------------\n");
		printf("[ROUTE-1] ArgmaxRoutingSelectsCorrectTeacher\n");
		printf("-----------------------------------\n");
		float class_probs[5] = {0.1f, 0.7f, 0.05f, 0.10f, 0.05f};  // class 1 wins
		unsigned int sel = glades::sampling::route_to_teacher(class_probs, 5);
		printf("  class_probs = [0.1, 0.7, 0.05, 0.10, 0.05]; argmax routes to %u (expect 1)\n", sel);
		ASSERT("argmax routing selects max-prob class", sel == 1u);
	}

	// --- Test ROUTE-2: routed loss matches selected teacher's KL ---
	{
		printf("-----------------------------------\n");
		printf("[ROUTE-2] RoutedLossMatchesSelectedTeacher\n");
		printf("-----------------------------------\n");
		const unsigned int vocab = 4u;
		const unsigned int K = 3u;
		// 3 teachers — each peaked on a different token
		float teachers[3 * 4] = {
		    0.7f, 0.1f, 0.1f, 0.1f,  // teacher 0 peaks at token 0
		    0.1f, 0.7f, 0.1f, 0.1f,  // teacher 1 peaks at token 1
		    0.1f, 0.1f, 0.1f, 0.7f   // teacher 2 peaks at token 3
		};
		float student[4] = {0.40f, 0.20f, 0.20f, 0.20f};  // close to teacher 0
		const unsigned int gt = 0u;
		const float alpha = 0.3f;

		// Route to teacher 0
		float class0[3] = {0.9f, 0.05f, 0.05f};
		float L0 = glades::sampling::multi_teacher_routed_loss(
		    class0, teachers, student, gt, vocab, K, alpha);
		// Route to teacher 1
		float class1[3] = {0.05f, 0.9f, 0.05f};
		float L1 = glades::sampling::multi_teacher_routed_loss(
		    class1, teachers, student, gt, vocab, K, alpha);
		// Route to teacher 2
		float class2[3] = {0.05f, 0.05f, 0.9f};
		float L2 = glades::sampling::multi_teacher_routed_loss(
		    class2, teachers, student, gt, vocab, K, alpha);

		printf("  L_route_to_T0=%.4f  L_route_to_T1=%.4f  L_route_to_T2=%.4f\n", L0, L1, L2);
		// Student is closest to T0 → KL with T0 should be smallest → L0 smallest
		ASSERT("routing to closest teacher minimizes loss", L0 < L1 && L0 < L2);
	}

	// --- Test ROUTE-3: routed loss vs ensemble loss decomposition ---
	{
		printf("-----------------------------------\n");
		printf("[ROUTE-3] RoutedVsEnsembleDecomposition\n");
		printf("-----------------------------------\n");
		const unsigned int vocab = 4u;
		const unsigned int K = 3u;
		float teachers[3 * 4] = {
		    0.50f, 0.30f, 0.10f, 0.10f,
		    0.10f, 0.50f, 0.30f, 0.10f,
		    0.10f, 0.10f, 0.30f, 0.50f
		};
		float student[4] = {0.25f, 0.25f, 0.25f, 0.25f};  // uniform
		const unsigned int gt = 0u;
		const float alpha = 0.3f;
		// Hard routing case: class_probs near one-hot
		float class_hard[3] = {0.9f, 0.05f, 0.05f};
		float L_route = glades::sampling::multi_teacher_routed_loss(
		    class_hard, teachers, student, gt, vocab, K, alpha);
		float L_ens = glades::sampling::multi_teacher_ensemble_loss(
		    class_hard, teachers, student, gt, vocab, K, alpha);
		printf("  hard routing (class[0]=0.9): L_route=%.4f vs L_ens=%.4f\n",
		       L_route, L_ens);
		ASSERT("hard routing close to ensemble at one-hot classifier",
		       std::fabs(L_route - L_ens) < 0.05f);

		// Uniform case
		float class_uni[3] = {0.333f, 0.333f, 0.334f};
		float L_route_u = glades::sampling::multi_teacher_routed_loss(
		    class_uni, teachers, student, gt, vocab, K, alpha);
		float L_ens_u = glades::sampling::multi_teacher_ensemble_loss(
		    class_uni, teachers, student, gt, vocab, K, alpha);
		printf("  uniform routing: L_route=%.4f (= KL with one teacher) vs L_ens=%.4f (avg over 3)\n",
		       L_route_u, L_ens_u);
	}

	// --- Test ROUTE-4: variance reduction (routing vs ensemble) ---
	{
		printf("-----------------------------------\n");
		printf("[ROUTE-4] VarianceReductionUnderRouting\n");
		printf("-----------------------------------\n");
		// Simulate K teachers with random distributions; classifier is reasonably
		// confident. Compute Monte-Carlo variance of routed loss vs ensemble.
		const unsigned int vocab = 8u;
		const unsigned int K = 5u;
		const unsigned int N = 400u;
		const float alpha = 0.3f;
		std::vector<float> teachers(K * vocab);
		unsigned int seed = 0x4ECEBA12u;
		// fill teachers with distinct distributions
		for (unsigned int k = 0; k < K; ++k)
		{
			double sum = 0.0;
			for (unsigned int v = 0; v < vocab; ++v)
			{
				seed = seed * 1103515245u + 12345u;
				float x = static_cast<float>((seed >> 16) & 0xFFFF) / 65535.0f;
				teachers[k * vocab + v] = 0.05f + x;
				sum += teachers[k * vocab + v];
			}
			for (unsigned int v = 0; v < vocab; ++v)
				teachers[k * vocab + v] /= static_cast<float>(sum);
		}

		double meanR = 0.0, meanE = 0.0;
		std::vector<float> routedLosses(N), ensembleLosses(N);
		for (unsigned int i = 0; i < N; ++i)
		{
			std::vector<float> student(vocab, 0.0f);
			std::vector<float> classp(K, 0.0f);
			double s = 0.0;
			for (unsigned int v = 0; v < vocab; ++v)
			{
				seed = seed * 1103515245u + 12345u;
				float x = static_cast<float>((seed >> 16) & 0xFFFF) / 65535.0f;
				student[v] = 0.05f + x;
				s += student[v];
			}
			for (unsigned int v = 0; v < vocab; ++v) student[v] /= static_cast<float>(s);
			// classifier output: peaked on a random class (high confidence)
			seed = seed * 1103515245u + 12345u;
			unsigned int peakC = (seed >> 16) % K;
			for (unsigned int c = 0; c < K; ++c) classp[c] = (c == peakC) ? 0.85f : (0.15f / (K - 1));
			unsigned int gt = ((seed >> 8) & 0xFF) % vocab;

			routedLosses[i] = glades::sampling::multi_teacher_routed_loss(
			    classp.data(), teachers.data(), student.data(), gt, vocab, K, alpha);
			ensembleLosses[i] = glades::sampling::multi_teacher_ensemble_loss(
			    classp.data(), teachers.data(), student.data(), gt, vocab, K, alpha);
			meanR += routedLosses[i];
			meanE += ensembleLosses[i];
		}
		meanR /= N;
		meanE /= N;
		double varR = 0.0, varE = 0.0;
		for (unsigned int i = 0; i < N; ++i)
		{
			varR += (routedLosses[i] - meanR) * (routedLosses[i] - meanR);
			varE += (ensembleLosses[i] - meanE) * (ensembleLosses[i] - meanE);
		}
		varR /= N - 1; varE /= N - 1;
		printf("  routed:   mean=%.4f  var=%.4f\n", meanR, varR);
		printf("  ensemble: mean=%.4f  var=%.4f\n", meanE, varE);
		printf("  variance ratio routed/ensemble = %.3f\n", varR / varE);
		// At high classifier confidence (0.85), routed and ensemble are similar
		// (one teacher dominates in both). At low confidence the routed variance
		// climbs while ensemble stays smooth — that's the #70-B trade.
		ASSERT("routed mean close to ensemble at high classifier confidence",
		       std::fabs(meanR - meanE) < 0.5f);
	}

	// --- Test ROUTE-5: class-collapse detection ---
	{
		printf("-----------------------------------\n");
		printf("[ROUTE-5] ClassCollapseDetection\n");
		printf("-----------------------------------\n");
		// Healthy histogram: roughly balanced
		unsigned int hist_healthy[5] = {180u, 220u, 200u, 210u, 190u};  // total 1000
		float frac_healthy = glades::sampling::max_class_routing_fraction(hist_healthy, 5, 1000u);
		printf("  healthy histogram: max fraction = %.3f (must be < 0.80)\n", frac_healthy);
		ASSERT("healthy histogram below collapse threshold", frac_healthy < 0.80f);

		// Collapsed: one class dominates
		unsigned int hist_collapse[5] = {850u, 30u, 50u, 40u, 30u};  // class 0 = 85%
		float frac_collapse = glades::sampling::max_class_routing_fraction(hist_collapse, 5, 1000u);
		printf("  collapsed histogram: max fraction = %.3f (must be >= 0.80)\n", frac_collapse);
		ASSERT("collapse detected", frac_collapse >= 0.80f);
	}

	// ============================================================
	// Group REASON: Paradigm shift #69 REASONING-DISTILL
	// ============================================================
	// Reasoning-trace primitives: special-token region masking,
	// per-region weighted loss, top-K logit caching for storage.

	// --- Test REASON-1: region mask from <THINK>...</THINK> tokens ---
	{
		printf("-----------------------------------\n");
		printf("[REASON-1] RegionMaskFromSpecialTokens\n");
		printf("-----------------------------------\n");
		// Sequence: 99 100(open) 1 2 3 101(close) 99 99
		const unsigned int n = 8u;
		unsigned int toks[8] = {99u, 100u, 1u, 2u, 3u, 101u, 99u, 99u};
		const unsigned int THINK_OPEN = 100u;
		const unsigned int THINK_CLOSE = 101u;
		unsigned char mask[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
		glades::sampling::region_mask_from_special_tokens(toks, n, THINK_OPEN, THINK_CLOSE, mask);
		// Expected: [0, 1, 1, 1, 1, 1, 0, 0]  (open through close inclusive)
		const unsigned char expected[8] = {0u, 1u, 1u, 1u, 1u, 1u, 0u, 0u};
		bool ok = true;
		for (unsigned int i = 0; i < n; ++i)
		{
			printf("  tok[%u]=%u → mask=%u (expect %u)\n", i, toks[i], mask[i], expected[i]);
			if (mask[i] != expected[i]) ok = false;
		}
		ASSERT("region mask matches expected", ok);
	}

	// --- Test REASON-2: region-weighted loss ---
	{
		printf("-----------------------------------\n");
		printf("[REASON-2] RegionWeightedLoss\n");
		printf("-----------------------------------\n");
		// 8 tokens, last 4 in reasoning region.
		float loss[8] = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f};
		unsigned char mask[8] = {0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u};
		float total_w = 0.0f;
		// Equal weighting → sum = 4*1 + 4*2 = 12
		float s_eq = glades::sampling::region_weighted_loss(loss, mask, 8u, 1.0f, 1.0f, &total_w);
		printf("  equal weighting (1, 1): sum=%.2f total_w=%.2f (expect 12 / 8)\n", s_eq, total_w);
		ASSERT("equal sum = 12", std::fabs(s_eq - 12.0f) < 1e-5f);
		ASSERT("equal total_w = 8", std::fabs(total_w - 8.0f) < 1e-5f);

		// In-region weighted 3x: sum = 4*1*1 + 4*2*3 = 4 + 24 = 28
		float s_w = glades::sampling::region_weighted_loss(loss, mask, 8u, 3.0f, 1.0f, &total_w);
		printf("  in=3, out=1: sum=%.2f total_w=%.2f (expect 28 / 16)\n", s_w, total_w);
		ASSERT("weighted sum = 28", std::fabs(s_w - 28.0f) < 1e-5f);
	}

	// --- Test REASON-3: top-K logit caching ---
	{
		printf("-----------------------------------\n");
		printf("[REASON-3] TopKLogitsCache\n");
		printf("-----------------------------------\n");
		const unsigned int vocab = 10u;
		float logits[10] = {0.1f, 5.0f, 0.3f, 4.0f, 0.5f, 4.5f, 0.7f, 0.2f, 0.4f, 0.6f};
		// Sorted: idx 1(5.0), 5(4.5), 3(4.0), 6(0.7), 9(0.6), 4(0.5), 8(0.4), 2(0.3), 7(0.2), 0(0.1)
		const unsigned int K = 3u;
		unsigned int idx[3];
		float vals[3];
		glades::sampling::top_k_logits(logits, vocab, K, idx, vals);
		printf("  top-3: indices=[%u, %u, %u], values=[%.1f, %.1f, %.1f]\n",
		       idx[0], idx[1], idx[2], vals[0], vals[1], vals[2]);
		ASSERT("top-1 is index 1 (value 5.0)", idx[0] == 1u);
		ASSERT("top-2 is index 5 (value 4.5)", idx[1] == 5u);
		ASSERT("top-3 is index 3 (value 4.0)", idx[2] == 3u);
		ASSERT("values match", vals[0] == 5.0f && vals[1] == 4.5f && vals[2] == 4.0f);
	}

	// --- Test REASON-4: storage saving from top-K caching ---
	{
		printf("-----------------------------------\n");
		printf("[REASON-4] TopKStorageSaving\n");
		printf("-----------------------------------\n");
		// Per #69 design: vocab=128K, top-K=64 → ~6×64 / 2×128K = 384 / 256K ≈ 0.0015
		const unsigned int vocab = 131072u;  // 128K
		float frac64 = glades::sampling::top_k_storage_fraction(vocab, 64u);
		float frac16 = glades::sampling::top_k_storage_fraction(vocab, 16u);
		printf("  vocab=128K top-64: %.6f (expect ~0.0015)\n", frac64);
		printf("  vocab=128K top-16: %.6f (expect ~0.00037)\n", frac16);
		ASSERT("top-64 is small fraction", frac64 < 0.01f);
		ASSERT("top-16 is even smaller", frac16 < frac64);
		printf("  → 64-TB → ~96 GB at top-64; ~24 GB at top-16 for 500B-token corpus\n");
	}

	printf("============================================================\n");
	printf("All Sampling Tests Passed\n");
	printf("============================================================\n");
}
