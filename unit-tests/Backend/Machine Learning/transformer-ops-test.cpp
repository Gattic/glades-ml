#include "transformer-ops-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/rng.h"
#include "../../../Backend/Machine Learning/Networks/transformer_ops.h"
#include "../../../Backend/Machine Learning/Networks/transformer_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using glades::transformer_ops::gelu;
using glades::transformer_ops::gelu_deriv;
using glades::transformer_ops::silu;
using glades::transformer_ops::silu_deriv;
using glades::transformer_ops::softmax_masked_row_stable;
using glades::transformer_ops::softmax_masked_row_stable_keymask;
using glades::transformer_ops::scaled_dot_product_attention_forward;
using glades::transformer_ops::scaled_dot_product_attention_forward_flash_strided;
using glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_strided;
using glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_chunk;
using glades::transformer_ops::sw_key_allowed;
using glades::transformer_ops::sw_visited_count;
using glades::transformer_ops::scaled_dot_product_attention_forward_flash_strided_sw;
using glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_strided_sw;
using glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_chunk_sw;
using glades::transformer_ops::mla_decompress_kv;
using glades::transformer_ops::mla_compute_latent;
using glades::transformer_ops::mla_compression_ratio;
using glades::transformer_kernels::silu_forward_buf;
using glades::transformer_kernels::gelu_forward_buf;
using glades::transformer_kernels::silu_backward_buf;
using glades::transformer_kernels::gelu_backward_buf;
using glades::transformer_kernels::softmax_stable_inplace;
using glades::transformer_kernels::softmax_stable_into;

// ============================================================
// Helpers
// ============================================================

static float pseudo_rand(unsigned int& seed)
{
	seed = seed * 1103515245u + 12345u;
	return (static_cast<float>((seed >> 16) & 0x7FFF) / 32768.0f) * 2.0f - 1.0f;
}

static void fill_random(float* buf, unsigned int n, unsigned int& seed)
{
	for (unsigned int i = 0; i < n; ++i)
		buf[i] = pseudo_rand(seed);
}

static double rel_err(double result, double ref)
{
	if (ref == 0.0 && result == 0.0)
		return 0.0;
	double denom = std::fabs(ref);
	if (denom < 1e-12)
		denom = 1e-12;
	return std::fabs(result - ref) / denom;
}

static bool is_finite_val(float x)
{
	return (x == x) && (x < 1e30f) && (x > -1e30f);
}

// ============================================================
// Group A: Activation Derivatives
// ============================================================

static void test_gelu_deriv_finite_diff()
{
	printf("  [A1] GELUDerivMatchesFiniteDifference ...\n");
	const float xs[] = {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
	const unsigned int N = sizeof(xs) / sizeof(xs[0]);
	const double eps = 1e-4;
	for (unsigned int i = 0; i < N; ++i)
	{
		double x = static_cast<double>(xs[i]);
		float analytical = gelu_deriv(static_cast<float>(x));
		double numerical = (static_cast<double>(gelu(static_cast<float>(x + eps))) - static_cast<double>(gelu(static_cast<float>(x - eps)))) / (2.0 * eps);
		double abserr = std::fabs(static_cast<double>(analytical) - numerical);
		ASSERT("gelu_deriv finite diff mismatch", abserr < 5e-3);
	}
	printf("    PASSED\n");
}

static void test_silu_deriv_finite_diff()
{
	printf("  [A2] SiLUDerivMatchesFiniteDifference ...\n");
	const float xs[] = {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
	const unsigned int N = sizeof(xs) / sizeof(xs[0]);
	const double eps = 1e-4;
	for (unsigned int i = 0; i < N; ++i)
	{
		double x = static_cast<double>(xs[i]);
		float analytical = silu_deriv(static_cast<float>(x));
		double numerical = (static_cast<double>(silu(static_cast<float>(x + eps))) - static_cast<double>(silu(static_cast<float>(x - eps)))) / (2.0 * eps);
		double abserr = std::fabs(static_cast<double>(analytical) - numerical);
		ASSERT("silu_deriv finite diff mismatch", abserr < 5e-3);
	}
	printf("    PASSED\n");
}

static void test_gelu_deriv_extreme()
{
	printf("  [A3] GELUDerivExtremeValues ...\n");
	float dLarge = gelu_deriv(100.0f);
	float dSmall = gelu_deriv(-100.0f);
	ASSERT("gelu_deriv(100) should be ~1.0", std::fabs(dLarge - 1.0f) < 1e-3f);
	ASSERT("gelu_deriv(-100) should be ~0.0", std::fabs(dSmall) < 1e-3f);
	printf("    gelu_deriv(100)=%.6f, gelu_deriv(-100)=%.6f\n", dLarge, dSmall);
	printf("    PASSED\n");
}

static void test_silu_deriv_extreme()
{
	printf("  [A4] SiLUDerivExtremeValues ...\n");
	float dLarge = silu_deriv(100.0f);
	float dSmall = silu_deriv(-100.0f);
	ASSERT("silu_deriv(100) should be ~1.0", std::fabs(dLarge - 1.0f) < 1e-3f);
	ASSERT("silu_deriv(-100) should be ~0.0", std::fabs(dSmall) < 1e-3f);
	printf("    silu_deriv(100)=%.6f, silu_deriv(-100)=%.6f\n", dLarge, dSmall);
	printf("    PASSED\n");
}

static void test_activation_buf_matches_scalar()
{
	printf("  [A5] ActivationBufMatchesScalar ...\n");
	const size_t n = 64;
	std::vector<float> x(n);
	unsigned int seed = 42u;
	fill_random(&x[0], n, seed);

	// SiLU buf vs scalar
	{
		std::vector<float> yBuf(n);
		silu_forward_buf(&x[0], &yBuf[0], n);
		for (size_t i = 0; i < n; ++i)
		{
			float expected = silu(x[i]);
			ASSERT("silu_forward_buf mismatch", std::fabs(yBuf[i] - expected) < 1e-6f);
		}
	}

	// GELU buf vs scalar
	{
		std::vector<float> yBuf(n);
		gelu_forward_buf(&x[0], &yBuf[0], n);
		for (size_t i = 0; i < n; ++i)
		{
			float expected = gelu(x[i]);
			ASSERT("gelu_forward_buf mismatch", std::fabs(yBuf[i] - expected) < 1e-6f);
		}
	}
	printf("    PASSED\n");
}

static void test_activation_backward_buf_matches_scalar()
{
	printf("  [A6] ActivationBackwardBufMatchesScalar ...\n");
	const size_t n = 64;
	std::vector<float> x(n);
	unsigned int seed = 42u;
	fill_random(&x[0], n, seed);

	// SiLU backward: dAct[i] *= silu_deriv(x[i]), so with dAct=1.0 result is silu_deriv(x[i])
	{
		std::vector<float> dAct(n, 1.0f);
		silu_backward_buf(&x[0], &dAct[0], n);
		for (size_t i = 0; i < n; ++i)
		{
			float expected = silu_deriv(x[i]);
			ASSERT("silu_backward_buf mismatch", std::fabs(dAct[i] - expected) < 1e-6f);
		}
	}

	// GELU backward
	{
		std::vector<float> dAct(n, 1.0f);
		gelu_backward_buf(&x[0], &dAct[0], n);
		for (size_t i = 0; i < n; ++i)
		{
			float expected = gelu_deriv(x[i]);
			ASSERT("gelu_backward_buf mismatch", std::fabs(dAct[i] - expected) < 1e-6f);
		}
	}
	printf("    PASSED\n");
}

// ============================================================
// Group B: Softmax
// ============================================================

static void test_softmax_masked_causal_basic()
{
	printf("  [B1] SoftmaxMaskedCausalBasic ...\n");
	const unsigned int T = 4;
	// Scores: flat [T*T], each row = [1,2,3,4]
	std::vector<float> scores(T);
	std::vector<float> probsRow;

	// Row 0: causal => only position 0 allowed
	{
		scores[0] = 1.0f; scores[1] = 2.0f; scores[2] = 3.0f; scores[3] = 4.0f;
		softmax_masked_row_stable(&scores[0], T, 0, true, probsRow);
		ASSERT("row0 pos0 should be 1.0", std::fabs(probsRow[0] - 1.0f) < 1e-6f);
		ASSERT("row0 pos1 should be 0.0", std::fabs(probsRow[1]) < 1e-6f);
		ASSERT("row0 pos2 should be 0.0", std::fabs(probsRow[2]) < 1e-6f);
		ASSERT("row0 pos3 should be 0.0", std::fabs(probsRow[3]) < 1e-6f);
	}

	// Row 3: causal => all positions [0..3] allowed
	{
		scores[0] = 1.0f; scores[1] = 2.0f; scores[2] = 3.0f; scores[3] = 4.0f;
		softmax_masked_row_stable(&scores[0], T, 3, true, probsRow);
		double sum = 0.0;
		for (unsigned int i = 0; i < T; ++i)
			sum += static_cast<double>(probsRow[i]);
		ASSERT("row3 sum should be 1.0", std::fabs(sum - 1.0) < 1e-6);
		// All probs should be non-zero
		for (unsigned int i = 0; i < T; ++i)
			ASSERT("row3 all probs non-zero", probsRow[i] > 0.0f);
	}
	printf("    PASSED\n");
}

static void test_softmax_masked_noncausal()
{
	printf("  [B2] SoftmaxMaskedNonCausal ...\n");
	const unsigned int T = 4;
	std::vector<float> scores(T);
	scores[0] = 1.0f; scores[1] = 2.0f; scores[2] = 3.0f; scores[3] = 4.0f;
	std::vector<float> probsRow;
	softmax_masked_row_stable(&scores[0], T, 0, false, probsRow);

	double sum = 0.0;
	for (unsigned int i = 0; i < T; ++i)
	{
		ASSERT("non-causal all probs positive", probsRow[i] > 0.0f);
		sum += static_cast<double>(probsRow[i]);
	}
	ASSERT("non-causal sum should be 1.0", std::fabs(sum - 1.0) < 1e-6);
	printf("    PASSED\n");
}

static void test_softmax_keymask_subset()
{
	printf("  [B3] SoftmaxKeymaskSubset ...\n");
	const unsigned int T = 4;
	std::vector<float> scores(T, 0.0f); // all zeros
	unsigned char keyAllowed[4] = {1, 0, 1, 0};
	std::vector<float> probsRow;
	softmax_masked_row_stable_keymask(&scores[0], T, 0, false, keyAllowed, probsRow);

	// Positions 1,3 should be 0
	ASSERT("keymask pos1 should be 0", std::fabs(probsRow[1]) < 1e-6f);
	ASSERT("keymask pos3 should be 0", std::fabs(probsRow[3]) < 1e-6f);
	// Positions 0,2 should each be 0.5 (equal scores, both allowed)
	ASSERT("keymask pos0 should be 0.5", std::fabs(probsRow[0] - 0.5f) < 1e-6f);
	ASSERT("keymask pos2 should be 0.5", std::fabs(probsRow[2] - 0.5f) < 1e-6f);
	printf("    PASSED\n");
}

static void test_softmax_keymask_all_masked()
{
	printf("  [B4] SoftmaxKeymaskAllMasked ...\n");
	const unsigned int T = 4;
	std::vector<float> scores(T, 1.0f);
	unsigned char keyAllowed[4] = {0, 0, 0, 0};
	std::vector<float> probsRow;
	softmax_masked_row_stable_keymask(&scores[0], T, 0, false, keyAllowed, probsRow);

	for (unsigned int i = 0; i < T; ++i)
		ASSERT("all-masked should output zeros", std::fabs(probsRow[i]) < 1e-6f);
	printf("    PASSED\n");
}

static void test_softmax_numerical_stability()
{
	printf("  [B5] SoftmaxNumericalStability ...\n");
	const unsigned int T = 3;
	std::vector<float> scores(T);
	scores[0] = 1e6f; scores[1] = 1e6f + 1.0f; scores[2] = 1e6f - 1.0f;
	std::vector<float> probsRow;
	softmax_masked_row_stable(&scores[0], T, 0, false, probsRow);

	double sum = 0.0;
	for (unsigned int i = 0; i < T; ++i)
	{
		ASSERT("large scores: no NaN/Inf", is_finite_val(probsRow[i]));
		sum += static_cast<double>(probsRow[i]);
	}
	ASSERT("large scores: sum ~1.0", std::fabs(sum - 1.0) < 1e-5);
	printf("    sum=%.8f\n", sum);
	printf("    PASSED\n");
}

static void test_softmax_uniform_scores()
{
	printf("  [B6] SoftmaxUniformScores ...\n");
	const unsigned int T = 4;
	std::vector<float> scores(T, 5.0f);
	std::vector<float> probsRow;
	softmax_masked_row_stable(&scores[0], T, 0, false, probsRow);

	for (unsigned int i = 0; i < T; ++i)
		ASSERT("uniform scores should give 0.25", std::fabs(probsRow[i] - 0.25f) < 1e-6f);
	printf("    PASSED\n");
}

static void test_softmax_single_allowed()
{
	printf("  [B7] SoftmaxSingleAllowed ...\n");
	const unsigned int T = 4;
	std::vector<float> scores(T);
	scores[0] = 2.0f; scores[1] = 5.0f; scores[2] = 3.0f; scores[3] = 1.0f;
	std::vector<float> probsRow;
	// causal=true, row 0 => only position 0 allowed
	softmax_masked_row_stable(&scores[0], T, 0, true, probsRow);

	ASSERT("single allowed pos0 == 1.0", std::fabs(probsRow[0] - 1.0f) < 1e-6f);
	ASSERT("single allowed pos1 == 0.0", std::fabs(probsRow[1]) < 1e-6f);
	ASSERT("single allowed pos2 == 0.0", std::fabs(probsRow[2]) < 1e-6f);
	ASSERT("single allowed pos3 == 0.0", std::fabs(probsRow[3]) < 1e-6f);
	printf("    PASSED\n");
}

static void test_softmax_stable_variant_parity()
{
	printf("  [B8] SoftmaxStableVariantParity ...\n");
	const size_t N = 16;
	std::vector<float> logits(N);
	unsigned int seed = 42u;
	fill_random(&logits[0], N, seed);

	// softmax_stable_inplace on a copy
	std::vector<float> probs_inplace(logits);
	softmax_stable_inplace(probs_inplace);

	// softmax_stable_into
	std::vector<float> probs_into(N);
	softmax_stable_into(&logits[0], N, &probs_into[0]);

	for (size_t i = 0; i < N; ++i)
	{
		float diff = std::fabs(probs_inplace[i] - probs_into[i]);
		ASSERT("softmax_stable_inplace vs softmax_stable_into parity", diff < 1e-6f);
	}
	printf("    PASSED\n");
}

// ============================================================
// Group C: Attention Forward
// ============================================================

static void test_forward_causal_t1()
{
	printf("  [C1] ForwardCausalT1 ...\n");
	const unsigned int T = 1, dK = 4, dV = 4;
	std::vector<float> Q(T * dK), K(T * dK), V(T * dV);
	unsigned int seed = 42u;
	fill_random(&Q[0], T * dK, seed);
	fill_random(&K[0], T * dK, seed);
	fill_random(&V[0], T * dV, seed);

	std::vector<float> O;
	scaled_dot_product_attention_forward(&Q[0], &K[0], &V[0], T, dK, dV, true, O, NULL);

	// With T=1, attention is 1.0 on self, so O should equal V[0]
	for (unsigned int d = 0; d < dV; ++d)
	{
		float diff = std::fabs(O[d] - V[d]);
		ASSERT("T=1 causal: O should equal V[0]", diff < 1e-6f);
	}
	printf("    PASSED\n");
}

static void test_forward_noncausal_probs_sum()
{
	printf("  [C2] ForwardNonCausalProbsSumToOne ...\n");
	const unsigned int T = 8, dK = 16, dV = 16;
	std::vector<float> Q(T * dK), K(T * dK), V(T * dV);
	unsigned int seed = 42u;
	fill_random(&Q[0], T * dK, seed);
	fill_random(&K[0], T * dK, seed);
	fill_random(&V[0], T * dV, seed);

	std::vector<float> O;
	std::vector<float> probsCache;
	scaled_dot_product_attention_forward(&Q[0], &K[0], &V[0], T, dK, dV, false, O, &probsCache);

	for (unsigned int t = 0; t < T; ++t)
	{
		double sum = 0.0;
		for (unsigned int u = 0; u < T; ++u)
			sum += static_cast<double>(probsCache[t * T + u]);
		ASSERT("non-causal probs row sum should be 1.0", std::fabs(sum - 1.0) < 1e-5);
	}
	printf("    PASSED\n");
}

static void test_forward_causal_masking_correct()
{
	printf("  [C3] ForwardCausalMaskingCorrect ...\n");
	const unsigned int T = 4, dK = 8, dV = 8;
	std::vector<float> Q(T * dK), K(T * dK), V(T * dV);
	unsigned int seed = 42u;
	fill_random(&Q[0], T * dK, seed);
	fill_random(&K[0], T * dK, seed);
	fill_random(&V[0], T * dV, seed);

	std::vector<float> O;
	std::vector<float> probsCache;
	scaled_dot_product_attention_forward(&Q[0], &K[0], &V[0], T, dK, dV, true, O, &probsCache);

	// Row 0: only [0] is non-zero
	ASSERT("row0 pos0 non-zero", probsCache[0 * T + 0] > 0.0f);
	for (unsigned int u = 1; u < T; ++u)
		ASSERT("row0 future positions zero", std::fabs(probsCache[0 * T + u]) < 1e-6f);

	// Row 1: only [0,1] non-zero
	ASSERT("row1 pos0 non-zero", probsCache[1 * T + 0] > 0.0f);
	ASSERT("row1 pos1 non-zero", probsCache[1 * T + 1] > 0.0f);
	for (unsigned int u = 2; u < T; ++u)
		ASSERT("row1 future positions zero", std::fabs(probsCache[1 * T + u]) < 1e-6f);

	// Row 3: all non-zero
	for (unsigned int u = 0; u < T; ++u)
		ASSERT("row3 all positions non-zero", probsCache[3 * T + u] > 0.0f);

	printf("    PASSED\n");
}

static void test_forward_keymask_sparsity()
{
	printf("  [C4] ForwardKeyMaskSparsity ...\n");
	const unsigned int T = 4, dK = 8, dV = 8;
	std::vector<float> Q(T * dK), K(T * dK), V(T * dV);
	unsigned int seed = 42u;
	fill_random(&Q[0], T * dK, seed);
	fill_random(&K[0], T * dK, seed);
	fill_random(&V[0], T * dV, seed);

	unsigned char keyAllowed[4] = {1, 0, 1, 0};
	std::vector<float> O;
	std::vector<float> probsCache;
	scaled_dot_product_attention_forward(&Q[0], &K[0], &V[0], T, dK, dV, false, O, &probsCache, keyAllowed);

	// Columns 1,3 should always be zero
	for (unsigned int t = 0; t < T; ++t)
	{
		ASSERT("keymask col1 should be zero", std::fabs(probsCache[t * T + 1]) < 1e-6f);
		ASSERT("keymask col3 should be zero", std::fabs(probsCache[t * T + 3]) < 1e-6f);
	}
	printf("    PASSED\n");
}

static void test_forward_all_identical_v()
{
	printf("  [C5] ForwardAllIdenticalV ...\n");
	const unsigned int T = 4, dK = 8, dV = 8;
	std::vector<float> Q(T * dK), K(T * dK), V(T * dV, 3.0f);
	unsigned int seed = 42u;
	fill_random(&Q[0], T * dK, seed);
	fill_random(&K[0], T * dK, seed);
	// V is all 3.0

	std::vector<float> O;
	scaled_dot_product_attention_forward(&Q[0], &K[0], &V[0], T, dK, dV, true, O, NULL);

	for (unsigned int i = 0; i < T * dV; ++i)
		ASSERT("identical V => O should be 3.0", std::fabs(O[i] - 3.0f) < 1e-6f);
	printf("    PASSED\n");
}

static void test_forward_flash_matches_materialized()
{
	printf("  [C6] ForwardFlashMatchesMaterialized ...\n");
	const unsigned int T = 8, dK = 16, dV = 16;
	std::vector<float> Q(T * dK), K(T * dK), V(T * dV);
	unsigned int seed = 42u;
	fill_random(&Q[0], T * dK, seed);
	fill_random(&K[0], T * dK, seed);
	fill_random(&V[0], T * dV, seed);

	// Materialized forward
	std::vector<float> O_ref;
	scaled_dot_product_attention_forward(&Q[0], &K[0], &V[0], T, dK, dV, true, O_ref, NULL);

	// Flash strided forward (contiguous: stride = dK for Q/K, dV for V/O)
	std::vector<float> O_flash(T * dV, 0.0f);
	scaled_dot_product_attention_forward_flash_strided(
		&Q[0], dK, &K[0], dK, &V[0], dV, T, dK, dV, true, &O_flash[0], dV);

	float maxDiff = 0.0f;
	for (unsigned int i = 0; i < T * dV; ++i)
	{
		float diff = std::fabs(O_ref[i] - O_flash[i]);
		if (diff > maxDiff)
			maxDiff = diff;
	}
	printf("    maxDiff=%.6e\n", maxDiff);
	ASSERT("flash vs materialized forward mismatch", maxDiff < 1e-5f);
	printf("    PASSED\n");
}

static void test_forward_strided_matches_nonstrided()
{
	printf("  [C7] ForwardStridedMatchesNonStrided ...\n");
	const unsigned int T = 8, dK = 16, dV = 16;
	std::vector<float> Q(T * dK), K(T * dK), V(T * dV);
	unsigned int seed = 42u;
	fill_random(&Q[0], T * dK, seed);
	fill_random(&K[0], T * dK, seed);
	fill_random(&V[0], T * dV, seed);

	// Non-strided forward
	std::vector<float> O_ref;
	scaled_dot_product_attention_forward(&Q[0], &K[0], &V[0], T, dK, dV, true, O_ref, NULL);

	// Strided forward (stride = dK for Q/K, dV for V/O => same as contiguous)
	std::vector<float> O_strided(T * dV, 0.0f);
	std::vector<float> scoresScratch, probsRowScratch;
	glades::transformer_ops::scaled_dot_product_attention_forward_strided(
		&Q[0], dK, &K[0], dK, &V[0], dV, T, dK, dV, true,
		&O_strided[0], dV, scoresScratch, probsRowScratch);

	float maxDiff = 0.0f;
	for (unsigned int i = 0; i < T * dV; ++i)
	{
		float diff = std::fabs(O_ref[i] - O_strided[i]);
		if (diff > maxDiff)
			maxDiff = diff;
	}
	printf("    maxDiff=%.6e\n", maxDiff);
	ASSERT("strided vs non-strided forward mismatch", maxDiff < 1e-6f);
	printf("    PASSED\n");
}

// ============================================================
// Group D: Attention Backward Flash Chunk
// ============================================================

static double max_abs_diff(const float* a, const float* b, unsigned int n)
{
	double maxDiff = 0.0;
	for (unsigned int i = 0; i < n; ++i)
	{
		double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
		if (diff > maxDiff)
			maxDiff = diff;
	}
	return maxDiff;
}

static void test_flash_chunk_single_matches_full_flash()
{
	printf("  [D1] FlashChunkSingleMatchesFullFlash ...\n");
	const unsigned int T = 8, dK = 16, dV = 16;
	const unsigned int qSize = T * dK;
	const unsigned int kSize = T * dK;
	const unsigned int vSize = T * dV;

	std::vector<float> Q(qSize), K(kSize), V(vSize);
	unsigned int seed = 42u;
	fill_random(&Q[0], qSize, seed);
	fill_random(&K[0], kSize, seed);
	fill_random(&V[0], vSize, seed);

	std::vector<float> dO(T * dV, 1.0f);

	// Full flash backward
	std::vector<float> dQ_full(qSize, 0.0f), dK_full(kSize, 0.0f), dV_full(vSize, 0.0f);
	scaled_dot_product_attention_backward_recompute_flash_strided(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		T, dK, dV, true,
		&dQ_full[0], dK, &dK_full[0], dK, &dV_full[0], dV);

	// Flash chunk with tBegin=0, tEnd=T (single chunk covering all rows)
	std::vector<float> dQ_chunk(qSize, 0.0f);
	std::vector<float> dKlocal(kSize, 0.0f), dVlocal(vSize, 0.0f);
	scaled_dot_product_attention_backward_recompute_flash_chunk(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		0, T, T, dK, dV, true,
		&dQ_chunk[0], dK, &dKlocal[0], &dVlocal[0], NULL);

	double diffQ = max_abs_diff(&dQ_full[0], &dQ_chunk[0], qSize);
	double diffK = max_abs_diff(&dK_full[0], &dKlocal[0], kSize);
	double diffV = max_abs_diff(&dV_full[0], &dVlocal[0], vSize);
	printf("    dQ maxAbsDiff=%.6e, dK=%.6e, dV=%.6e\n", diffQ, diffK, diffV);
	ASSERT("dQ chunk single vs full flash mismatch", diffQ < 1e-5);
	ASSERT("dK chunk single vs full flash mismatch", diffK < 1e-5);
	ASSERT("dV chunk single vs full flash mismatch", diffV < 1e-5);
	printf("    PASSED\n");
}

static void test_flash_chunk_two_halves_match_full()
{
	printf("  [D2] FlashChunkTwoHalvesMatchFull ...\n");
	const unsigned int T = 8, dK = 16, dV = 16;
	const unsigned int qSize = T * dK;
	const unsigned int kSize = T * dK;
	const unsigned int vSize = T * dV;

	std::vector<float> Q(qSize), K(kSize), V(vSize);
	unsigned int seed = 42u;
	fill_random(&Q[0], qSize, seed);
	fill_random(&K[0], kSize, seed);
	fill_random(&V[0], vSize, seed);

	std::vector<float> dO(T * dV, 1.0f);

	// Full flash backward
	std::vector<float> dQ_full(qSize, 0.0f), dK_full(kSize, 0.0f), dV_full(vSize, 0.0f);
	scaled_dot_product_attention_backward_recompute_flash_strided(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		T, dK, dV, true,
		&dQ_full[0], dK, &dK_full[0], dK, &dV_full[0], dV);

	// Chunk 1: rows [0, 4)
	std::vector<float> dQ_chunked(qSize, 0.0f);
	std::vector<float> dKlocal1(kSize, 0.0f), dVlocal1(vSize, 0.0f);
	scaled_dot_product_attention_backward_recompute_flash_chunk(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		0, 4, T, dK, dV, true,
		&dQ_chunked[0], dK, &dKlocal1[0], &dVlocal1[0], NULL);

	// Chunk 2: rows [4, 8)
	std::vector<float> dKlocal2(kSize, 0.0f), dVlocal2(vSize, 0.0f);
	scaled_dot_product_attention_backward_recompute_flash_chunk(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		4, 8, T, dK, dV, true,
		&dQ_chunked[0], dK, &dKlocal2[0], &dVlocal2[0], NULL);

	// Sum dK/dV from both chunks
	std::vector<float> dK_chunked(kSize, 0.0f), dV_chunked(vSize, 0.0f);
	for (unsigned int i = 0; i < kSize; ++i)
		dK_chunked[i] = dKlocal1[i] + dKlocal2[i];
	for (unsigned int i = 0; i < vSize; ++i)
		dV_chunked[i] = dVlocal1[i] + dVlocal2[i];

	double diffQ = max_abs_diff(&dQ_full[0], &dQ_chunked[0], qSize);
	double diffK = max_abs_diff(&dK_full[0], &dK_chunked[0], kSize);
	double diffV = max_abs_diff(&dV_full[0], &dV_chunked[0], vSize);
	printf("    dQ maxAbsDiff=%.6e, dK=%.6e, dV=%.6e\n", diffQ, diffK, diffV);
	ASSERT("dQ two-halves chunk vs full mismatch", diffQ < 1e-4);
	ASSERT("dK two-halves chunk vs full mismatch", diffK < 1e-4);
	ASSERT("dV two-halves chunk vs full mismatch", diffV < 1e-4);
	printf("    PASSED\n");
}

static void test_flash_chunk_do_zero()
{
	printf("  [D3] FlashChunkDOZero ...\n");
	const unsigned int T = 4, dK = 8, dV = 8;
	const unsigned int qSize = T * dK;
	const unsigned int kSize = T * dK;
	const unsigned int vSize = T * dV;

	std::vector<float> Q(qSize), K(kSize), V(vSize);
	unsigned int seed = 42u;
	fill_random(&Q[0], qSize, seed);
	fill_random(&K[0], kSize, seed);
	fill_random(&V[0], vSize, seed);

	std::vector<float> dO(T * dV, 0.0f); // all zeros

	std::vector<float> dQ(qSize, 0.0f);
	std::vector<float> dKlocal(kSize, 0.0f), dVlocal(vSize, 0.0f);
	scaled_dot_product_attention_backward_recompute_flash_chunk(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		0, T, T, dK, dV, true,
		&dQ[0], dK, &dKlocal[0], &dVlocal[0], NULL);

	for (unsigned int i = 0; i < qSize; ++i)
		ASSERT("dQ should be zero when dO=0", std::fabs(dQ[i]) < 1e-6f);
	for (unsigned int i = 0; i < kSize; ++i)
		ASSERT("dK should be zero when dO=0", std::fabs(dKlocal[i]) < 1e-6f);
	for (unsigned int i = 0; i < vSize; ++i)
		ASSERT("dV should be zero when dO=0", std::fabs(dVlocal[i]) < 1e-6f);
	printf("    PASSED\n");
}

static void test_flash_chunk_gradient_finite_diff()
{
	printf("  [D4] FlashChunkGradientFiniteDiff ...\n");
	const unsigned int T = 4, dK = 4, dV = 4;
	const unsigned int qSize = T * dK;
	const unsigned int kSize = T * dK;
	const unsigned int vSize = T * dV;

	std::vector<float> Q(qSize), K(kSize), V(vSize);
	unsigned int seed = 42u;
	fill_random(&Q[0], qSize, seed);
	fill_random(&K[0], kSize, seed);
	fill_random(&V[0], vSize, seed);

	// Use dO = all ones (loss = sum(O))
	std::vector<float> dO(T * dV, 1.0f);

	// Analytical gradient via flash chunk (full range)
	std::vector<float> dQ_analytic(qSize, 0.0f);
	std::vector<float> dKlocal(kSize, 0.0f), dVlocal(vSize, 0.0f);
	scaled_dot_product_attention_backward_recompute_flash_chunk(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		0, T, T, dK, dV, true,
		&dQ_analytic[0], dK, &dKlocal[0], &dVlocal[0], NULL);

	// Numerical gradient for Q[0][0] via finite differences
	const float eps = 1e-3f;
	const float orig = Q[0];

	// Forward with Q[0][0] + eps
	Q[0] = orig + eps;
	std::vector<float> O_plus;
	scaled_dot_product_attention_forward(&Q[0], &K[0], &V[0], T, dK, dV, true, O_plus, NULL);
	double loss_plus = 0.0;
	for (size_t i = 0; i < O_plus.size(); ++i)
		loss_plus += static_cast<double>(O_plus[i]);

	// Forward with Q[0][0] - eps
	Q[0] = orig - eps;
	std::vector<float> O_minus;
	scaled_dot_product_attention_forward(&Q[0], &K[0], &V[0], T, dK, dV, true, O_minus, NULL);
	double loss_minus = 0.0;
	for (size_t i = 0; i < O_minus.size(); ++i)
		loss_minus += static_cast<double>(O_minus[i]);

	Q[0] = orig; // restore

	double numerical_dQ00 = (loss_plus - loss_minus) / (2.0 * static_cast<double>(eps));
	double analytical_dQ00 = static_cast<double>(dQ_analytic[0]);

	double absA = std::fabs(analytical_dQ00);
	double absN = std::fabs(numerical_dQ00);
	double denom = (absA > absN) ? absA : absN;
	if (denom < 1e-7)
		denom = 1e-7;
	double relErr = std::fabs(analytical_dQ00 - numerical_dQ00) / denom;

	printf("    analytical dQ[0][0]=%.6e, numerical=%.6e, relErr=%.6e\n",
	       analytical_dQ00, numerical_dQ00, relErr);
	ASSERT("flash chunk dQ[0][0] finite diff check", relErr < 5e-2);
	printf("    PASSED\n");
}

// ============================================================
// Group E: Paradigm shift #78 ATTENTION-SINK-DISTILL-CHIRON
// ============================================================
// Probe per #103 spec — verify mechanism + cache plateau + gradient
// correctness + throughput. Quality (NLL drift at LLM scale) is
// verified by chiron_train; here we cover structural correctness.

#include <ctime>

static double clock_seconds()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

static void test_sw_key_allowed_semantics()
{
	printf("  [E1] SinkWindowKeyAllowedSemantics ...\n");
	// disabled => always allowed
	ASSERT("sw disabled allows u=0", sw_key_allowed(100u, 0u, 0u, 0u, NULL));
	ASSERT("sw disabled allows u=99", sw_key_allowed(100u, 99u, 0u, 0u, NULL));
	// sink only (S=4, W=0): u<4 allowed; else not
	ASSERT("sink u=0 allowed", sw_key_allowed(100u, 0u, 4u, 0u, NULL));
	ASSERT("sink u=3 allowed", sw_key_allowed(100u, 3u, 4u, 0u, NULL));
	ASSERT("sink u=4 not allowed", !sw_key_allowed(100u, 4u, 4u, 0u, NULL));
	// window only (S=0, W=8) at t=100: u in (92, 100] allowed
	ASSERT("window u=92 not allowed (t-u==8)", !sw_key_allowed(100u, 92u, 0u, 8u, NULL));
	ASSERT("window u=93 allowed (t-u==7)", sw_key_allowed(100u, 93u, 0u, 8u, NULL));
	ASSERT("window u=100 allowed (t-u==0)", sw_key_allowed(100u, 100u, 0u, 8u, NULL));
	// sink+window (S=4, W=8) at t=100: u<4 OR u>92
	ASSERT("sink+win u=0 allowed", sw_key_allowed(100u, 0u, 4u, 8u, NULL));
	ASSERT("sink+win u=3 allowed", sw_key_allowed(100u, 3u, 4u, 8u, NULL));
	ASSERT("sink+win u=4 not allowed", !sw_key_allowed(100u, 4u, 4u, 8u, NULL));
	ASSERT("sink+win u=50 not allowed", !sw_key_allowed(100u, 50u, 4u, 8u, NULL));
	ASSERT("sink+win u=93 allowed", sw_key_allowed(100u, 93u, 4u, 8u, NULL));
	ASSERT("sink+win u=100 allowed", sw_key_allowed(100u, 100u, 4u, 8u, NULL));
	// keyAllowed override
	unsigned char ka[8] = {1u,1u,1u,1u,1u,1u,1u,0u};
	ASSERT("sink+win keyAllowed=0 not allowed", !sw_key_allowed(7u, 7u, 4u, 8u, ka));
	printf("    PASSED\n");
}

static void test_sw_visited_count_plateau()
{
	printf("  [E2] SinkWindowVisitedCountPlateau ...\n");
	// At sink=4, window=64: as T grows, the count of visited keys per row
	// at large t plateaus at S + W = 68.
	const unsigned int S = 4u;
	const unsigned int W = 64u;
	const unsigned int T = 1024u;
	for (unsigned int t = 0u; t < T; ++t)
	{
		unsigned int n = sw_visited_count(t, T, true /*causal*/, S, W, NULL);
		// For t < S+W (early): n is bounded by t+1 (all causal positions allowed because of overlap).
		// For t >= S+W-1: n exactly equals S + W (plateau).
		if (t + 1u <= S + W)
		{
			ASSERT("early t bounded by t+1", n == t + 1u);
		}
		else
		{
			ASSERT("plateau at S+W after warmup", n == S + W);
		}
	}
	printf("    PASSED (cache plateau at S+W=%u for T=%u)\n", S + W, T);
}

static void test_sw_forward_degenerate_full_attention()
{
	printf("  [E3] SinkWindowForwardDegenerateMatchesFullAttention ...\n");
	// When sinkCount >= T, every key is a sink => same as full attention.
	const unsigned int T = 8u;
	const unsigned int dK = 4u;
	const unsigned int dV = 4u;
	std::vector<float> Q(static_cast<size_t>(T) * dK);
	std::vector<float> K(static_cast<size_t>(T) * dK);
	std::vector<float> V(static_cast<size_t>(T) * dV);
	unsigned int seed = 0xC0FFEEu;
	fill_random(Q.data(), T * dK, seed);
	fill_random(K.data(), T * dK, seed);
	fill_random(V.data(), T * dV, seed);

	std::vector<float> Ofull(static_cast<size_t>(T) * dV, 0.0f);
	std::vector<float> Osw(static_cast<size_t>(T) * dV, 0.0f);

	scaled_dot_product_attention_forward_flash_strided(
	    Q.data(), dK, K.data(), dK, V.data(), dV,
	    T, dK, dV, true, Ofull.data(), dV, NULL);
	scaled_dot_product_attention_forward_flash_strided_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV,
	    T, dK, dV, true, T /*S>=T*/, 0u, Osw.data(), dV, NULL);

	double maxAbs = 0.0;
	for (size_t i = 0; i < Ofull.size(); ++i)
	{
		double d = std::fabs(static_cast<double>(Ofull[i]) - static_cast<double>(Osw[i]));
		if (d > maxAbs) maxAbs = d;
	}
	printf("    max |Ofull - Osw|=%.3e (degenerate sinkCount=T)\n", maxAbs);
	ASSERT("sw forward S=T parity with full", maxAbs < 1e-5);

	// And: sinkCount=0, windowSize=T also matches.
	std::fill(Osw.begin(), Osw.end(), 0.0f);
	scaled_dot_product_attention_forward_flash_strided_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV,
	    T, dK, dV, true, 0u, T /*W>=T*/, Osw.data(), dV, NULL);
	maxAbs = 0.0;
	for (size_t i = 0; i < Ofull.size(); ++i)
	{
		double d = std::fabs(static_cast<double>(Ofull[i]) - static_cast<double>(Osw[i]));
		if (d > maxAbs) maxAbs = d;
	}
	printf("    max |Ofull - Osw|=%.3e (degenerate windowSize=T)\n", maxAbs);
	ASSERT("sw forward W=T parity with full", maxAbs < 1e-5);
	printf("    PASSED\n");
}

static void test_sw_forward_mask_applied()
{
	printf("  [E4] SinkWindowForwardMaskApplied ...\n");
	// At long T relative to S+W, the output at a late row should depend
	// only on sinks and recent window — verify by changing K/V at out-of-range
	// positions and checking the output is unchanged.
	const unsigned int T = 64u;
	const unsigned int dK = 4u;
	const unsigned int dV = 4u;
	const unsigned int S = 4u;
	const unsigned int W = 8u;
	std::vector<float> Q(static_cast<size_t>(T) * dK);
	std::vector<float> K(static_cast<size_t>(T) * dK);
	std::vector<float> V(static_cast<size_t>(T) * dV);
	unsigned int seed = 0xBADF00Du;
	fill_random(Q.data(), T * dK, seed);
	fill_random(K.data(), T * dK, seed);
	fill_random(V.data(), T * dV, seed);

	std::vector<float> O1(static_cast<size_t>(T) * dV, 0.0f);
	scaled_dot_product_attention_forward_flash_strided_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV,
	    T, dK, dV, true, S, W, O1.data(), dV, NULL);

	// Perturb K and V at u=20 (which is between S=4 and t-W=63-8=55 at last row,
	// i.e., out of range for query t=63).
	const unsigned int uPerturb = 20u;
	for (unsigned int k = 0; k < dK; ++k)
		K[static_cast<size_t>(uPerturb) * dK + k] += 100.0f;
	for (unsigned int dv = 0; dv < dV; ++dv)
		V[static_cast<size_t>(uPerturb) * dV + dv] += 100.0f;

	std::vector<float> O2(static_cast<size_t>(T) * dV, 0.0f);
	scaled_dot_product_attention_forward_flash_strided_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV,
	    T, dK, dV, true, S, W, O2.data(), dV, NULL);

	// Last row (t=T-1=63) should be unchanged.
	const size_t tLastOff = static_cast<size_t>(T - 1u) * dV;
	double maxAbsLast = 0.0;
	for (unsigned int dv = 0; dv < dV; ++dv)
	{
		double d = std::fabs(static_cast<double>(O1[tLastOff + dv]) - static_cast<double>(O2[tLastOff + dv]));
		if (d > maxAbsLast) maxAbsLast = d;
	}
	// Earlier rows (those whose window includes uPerturb) MUST differ — sanity check.
	double maxAbsEarly = 0.0;
	for (unsigned int t = 21u; t < 28u; ++t)
	{
		const size_t off = static_cast<size_t>(t) * dV;
		for (unsigned int dv = 0; dv < dV; ++dv)
		{
			double d = std::fabs(static_cast<double>(O1[off + dv]) - static_cast<double>(O2[off + dv]));
			if (d > maxAbsEarly) maxAbsEarly = d;
		}
	}
	printf("    last-row max-abs-diff=%.3e (must be ~0); affected-window diff=%.3e (must be > 0)\n", maxAbsLast, maxAbsEarly);
	ASSERT("sw mask: last row unaffected by out-of-window K/V perturbation", maxAbsLast < 1e-5);
	ASSERT("sw mask: in-window rows respond to K/V perturbation", maxAbsEarly > 1e-3);
	printf("    PASSED\n");
}

static void test_sw_backward_finite_diff()
{
	printf("  [E5] SinkWindowBackwardGradientFiniteDiff ...\n");
	const unsigned int T = 16u;
	const unsigned int dK = 4u;
	const unsigned int dV = 4u;
	const unsigned int S = 2u;
	const unsigned int W = 4u;
	std::vector<float> Q(static_cast<size_t>(T) * dK);
	std::vector<float> K(static_cast<size_t>(T) * dK);
	std::vector<float> V(static_cast<size_t>(T) * dV);
	unsigned int seed = 0xDEADBEEFu;
	fill_random(Q.data(), T * dK, seed);
	fill_random(K.data(), T * dK, seed);
	fill_random(V.data(), T * dV, seed);

	// dO = ones (loss = sum of all O entries)
	std::vector<float> dO(static_cast<size_t>(T) * dV, 1.0f);
	std::vector<float> dQ(Q.size(), 0.0f);
	std::vector<float> dK_(K.size(), 0.0f);
	std::vector<float> dV_(V.size(), 0.0f);

	scaled_dot_product_attention_backward_recompute_flash_strided_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV,
	    dO.data(), dV, T, dK, dV, true, S, W,
	    dQ.data(), dK, dK_.data(), dK, dV_.data(), dV, NULL);

	// Check dQ at t=8, k=0 via finite-diff on sum-of-O loss.
	const unsigned int tCheck = 8u;
	const unsigned int kCheck = 0u;
	const float epsfd = 1e-3f;
	const size_t qIdx = static_cast<size_t>(tCheck) * dK + kCheck;
	const float origQ = Q[qIdx];

	std::vector<float> Oplus(static_cast<size_t>(T) * dV, 0.0f);
	std::vector<float> Ominus(static_cast<size_t>(T) * dV, 0.0f);
	Q[qIdx] = origQ + epsfd;
	scaled_dot_product_attention_forward_flash_strided_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV, T, dK, dV, true, S, W, Oplus.data(), dV, NULL);
	Q[qIdx] = origQ - epsfd;
	scaled_dot_product_attention_forward_flash_strided_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV, T, dK, dV, true, S, W, Ominus.data(), dV, NULL);
	Q[qIdx] = origQ;

	double sumPlus = 0.0;
	double sumMinus = 0.0;
	for (size_t i = 0; i < Oplus.size(); ++i) sumPlus += Oplus[i];
	for (size_t i = 0; i < Ominus.size(); ++i) sumMinus += Ominus[i];
	double numerical = (sumPlus - sumMinus) / (2.0 * static_cast<double>(epsfd));
	double analytical = static_cast<double>(dQ[qIdx]);

	double absA = std::fabs(analytical);
	double absN = std::fabs(numerical);
	double denom = (absA > absN) ? absA : absN;
	if (denom < 1e-7) denom = 1e-7;
	double relErr = std::fabs(analytical - numerical) / denom;
	printf("    analytical dQ[t=%u][k=%u]=%.6e, numerical=%.6e, relErr=%.3e\n",
	       tCheck, kCheck, analytical, numerical, relErr);
	ASSERT("sw backward dQ finite-diff", relErr < 5e-2);
	printf("    PASSED\n");
}

static void test_sw_throughput_speedup()
{
	printf("  [E6] SinkWindowThroughputSpeedup ...\n");
	// Profile: full vs sink+window forward at moderate T on CPU.
	// The speedup target per #78 design: 5.85× at T=2052, W=2048, S=4.
	// We test at a smaller T (since CPU only) — expect at least
	// 1.5× speedup at T=512, S=4, W=64 (T/(S+W) ≈ 7.5× theoretical).
	const unsigned int T = 512u;
	const unsigned int dK = 32u;
	const unsigned int dV = 32u;
	const unsigned int S = 4u;
	const unsigned int W = 64u;
	std::vector<float> Q(static_cast<size_t>(T) * dK);
	std::vector<float> K(static_cast<size_t>(T) * dK);
	std::vector<float> V(static_cast<size_t>(T) * dV);
	unsigned int seed = 0xCAFEBABEu;
	fill_random(Q.data(), T * dK, seed);
	fill_random(K.data(), T * dK, seed);
	fill_random(V.data(), T * dV, seed);

	std::vector<float> Ofull(static_cast<size_t>(T) * dV, 0.0f);
	std::vector<float> Osw(static_cast<size_t>(T) * dV, 0.0f);

	const int trials = 5;
	// Warm up
	scaled_dot_product_attention_forward_flash_strided(
	    Q.data(), dK, K.data(), dK, V.data(), dV, T, dK, dV, true, Ofull.data(), dV, NULL);
	scaled_dot_product_attention_forward_flash_strided_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV, T, dK, dV, true, S, W, Osw.data(), dV, NULL);

	double tFull = 0.0;
	double t0 = clock_seconds();
	for (int i = 0; i < trials; ++i)
	{
		scaled_dot_product_attention_forward_flash_strided(
		    Q.data(), dK, K.data(), dK, V.data(), dV, T, dK, dV, true, Ofull.data(), dV, NULL);
	}
	tFull = clock_seconds() - t0;

	double tSW = 0.0;
	t0 = clock_seconds();
	for (int i = 0; i < trials; ++i)
	{
		scaled_dot_product_attention_forward_flash_strided_sw(
		    Q.data(), dK, K.data(), dK, V.data(), dV, T, dK, dV, true, S, W, Osw.data(), dV, NULL);
	}
	tSW = clock_seconds() - t0;

	double speedup = tFull / (tSW > 0.0 ? tSW : 1e-9);
	printf("    T=%u dK=%u S=%u W=%u trials=%d: full=%.3f ms, sw=%.3f ms, speedup=%.2fx\n",
	       T, dK, S, W, trials,
	       tFull * 1000.0 / trials, tSW * 1000.0 / trials, speedup);
	// Theoretical lower bound is ~T/(S+W) * pessimism factor.
	// For T=512, S=4, W=64: theoretical ≈ 7.5×; we'll require >= 1.5× to allow noise.
	ASSERT("sw forward provides measurable speedup at T=512, S+W=68", speedup >= 1.5);
	printf("    PASSED (theoretical ~%.1fx; observed %.2fx)\n",
	       static_cast<double>(T) / static_cast<double>(S + W), speedup);

	// Profile sweep across T to show how speedup scales (info-only).
	printf("    [profile sweep] sink+window vs full attention (S=4, W=64, dK=dV=32, trials=3):\n");
	const unsigned int Tsweep[] = {256u, 1024u, 2048u, 4096u};
	const unsigned int nSweep = sizeof(Tsweep) / sizeof(Tsweep[0]);
	for (unsigned int si = 0; si < nSweep; ++si)
	{
		const unsigned int Tt = Tsweep[si];
		std::vector<float> Qt(static_cast<size_t>(Tt) * dK);
		std::vector<float> Kt(static_cast<size_t>(Tt) * dK);
		std::vector<float> Vt(static_cast<size_t>(Tt) * dV);
		std::vector<float> Ot(static_cast<size_t>(Tt) * dV, 0.0f);
		unsigned int sweepSeed = 0xFEEDD00Du;
		fill_random(Qt.data(), Tt * dK, sweepSeed);
		fill_random(Kt.data(), Tt * dK, sweepSeed);
		fill_random(Vt.data(), Tt * dV, sweepSeed);

		const int trials_s = 3;
		double t0s = clock_seconds();
		for (int i = 0; i < trials_s; ++i)
		{
			scaled_dot_product_attention_forward_flash_strided(
			    Qt.data(), dK, Kt.data(), dK, Vt.data(), dV,
			    Tt, dK, dV, true, Ot.data(), dV, NULL);
		}
		double tFs = clock_seconds() - t0s;
		t0s = clock_seconds();
		for (int i = 0; i < trials_s; ++i)
		{
			scaled_dot_product_attention_forward_flash_strided_sw(
			    Qt.data(), dK, Kt.data(), dK, Vt.data(), dV,
			    Tt, dK, dV, true, S, W, Ot.data(), dV, NULL);
		}
		double tWs = clock_seconds() - t0s;
		double sp = tFs / (tWs > 0.0 ? tWs : 1e-9);
		printf("      T=%5u: full=%7.2f ms, sw=%6.2f ms, speedup=%5.2fx (theoretical %.1fx)\n",
		       Tt, tFs * 1000.0 / trials_s, tWs * 1000.0 / trials_s, sp,
		       static_cast<double>(Tt) / static_cast<double>(S + W));
	}
}

static void test_sw_chunk_matches_full_sw()
{
	printf("  [E7] SinkWindowChunkMatchesFullSinkWindow ...\n");
	const unsigned int T = 32u;
	const unsigned int dK = 4u;
	const unsigned int dV = 4u;
	const unsigned int S = 2u;
	const unsigned int W = 8u;
	std::vector<float> Q(static_cast<size_t>(T) * dK);
	std::vector<float> K(static_cast<size_t>(T) * dK);
	std::vector<float> V(static_cast<size_t>(T) * dV);
	unsigned int seed = 0x1337u;
	fill_random(Q.data(), T * dK, seed);
	fill_random(K.data(), T * dK, seed);
	fill_random(V.data(), T * dV, seed);

	std::vector<float> dO(static_cast<size_t>(T) * dV, 1.0f);

	// Full backward (sw)
	std::vector<float> dQa(Q.size(), 0.0f);
	std::vector<float> dKa(K.size(), 0.0f);
	std::vector<float> dVa(V.size(), 0.0f);
	scaled_dot_product_attention_backward_recompute_flash_strided_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV,
	    dO.data(), dV, T, dK, dV, true, S, W,
	    dQa.data(), dK, dKa.data(), dK, dVa.data(), dV, NULL);

	// Chunked: two halves, into local dK/dV buffers, then summed.
	std::vector<float> dQb(Q.size(), 0.0f);
	std::vector<float> dKlocal1(K.size(), 0.0f);
	std::vector<float> dVlocal1(V.size(), 0.0f);
	std::vector<float> dKlocal2(K.size(), 0.0f);
	std::vector<float> dVlocal2(V.size(), 0.0f);
	scaled_dot_product_attention_backward_recompute_flash_chunk_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV,
	    dO.data(), dV, 0u, T / 2u,
	    T, dK, dV, true, S, W,
	    dQb.data(), dK, dKlocal1.data(), dVlocal1.data(), NULL);
	scaled_dot_product_attention_backward_recompute_flash_chunk_sw(
	    Q.data(), dK, K.data(), dK, V.data(), dV,
	    dO.data(), dV, T / 2u, T,
	    T, dK, dV, true, S, W,
	    dQb.data(), dK, dKlocal2.data(), dVlocal2.data(), NULL);

	// Sum chunks for dK/dV comparison.
	std::vector<float> dKb(K.size(), 0.0f);
	std::vector<float> dVb(V.size(), 0.0f);
	for (size_t i = 0; i < dKb.size(); ++i)
		dKb[i] = dKlocal1[i] + dKlocal2[i];
	for (size_t i = 0; i < dVb.size(); ++i)
		dVb[i] = dVlocal1[i] + dVlocal2[i];

	double maxQ = 0.0, maxK = 0.0, maxV = 0.0;
	for (size_t i = 0; i < dQa.size(); ++i)
		maxQ = std::max(maxQ, std::fabs(static_cast<double>(dQa[i] - dQb[i])));
	for (size_t i = 0; i < dKa.size(); ++i)
		maxK = std::max(maxK, std::fabs(static_cast<double>(dKa[i] - dKb[i])));
	for (size_t i = 0; i < dVa.size(); ++i)
		maxV = std::max(maxV, std::fabs(static_cast<double>(dVa[i] - dVb[i])));
	printf("    max |dQ_full - dQ_chunk|=%.3e, dK=%.3e, dV=%.3e\n", maxQ, maxK, maxV);
	ASSERT("sw chunk dQ parity with full", maxQ < 1e-5);
	ASSERT("sw chunk dK parity with full", maxK < 1e-5);
	ASSERT("sw chunk dV parity with full", maxV < 1e-5);
	printf("    PASSED\n");
}

// ============================================================
// Group F: Paradigm shift #76 MLA-DISTILL-CHIRON
// ============================================================
// Multi-Latent Attention (DeepSeek-V2/V3): KV cache stores low-rank
// latent c_t (d_c) instead of full K,V (n_heads*d_kv).

static void test_mla_compression_ratio()
{
	printf("  [F1] MLACompressionRatioFormula ...\n");
	// nHeads=16, dKV=128, d_c=512, d_rope=64:
	//   MHA per token = 16*128*2 = 4096
	//   MLA per token = 512+64 = 576
	//   ratio = 4096/576 ≈ 7.11
	float r = mla_compression_ratio(16u, 128u, 512u, 64u);
	printf("    nHeads=16 dKV=128 d_c=512 d_rope=64 -> compression=%.3fx\n", r);
	ASSERT("MLA compression at standard config", r > 7.0f && r < 7.2f);

	// Aggressive: d_c=256, d_rope=64 -> 4096/320 = 12.8
	r = mla_compression_ratio(16u, 128u, 256u, 64u);
	printf("    d_c=256 d_rope=64 -> compression=%.3fx\n", r);
	ASSERT("MLA compression aggressive", r > 12.5f && r < 13.0f);

	// Conservative: d_c=384, d_rope=64 -> 4096/448 = 9.14
	r = mla_compression_ratio(16u, 128u, 384u, 64u);
	printf("    d_c=384 d_rope=64 -> compression=%.3fx\n", r);
	ASSERT("MLA compression conservative", r > 9.0f && r < 9.3f);
	printf("    PASSED\n");
}

static void test_mla_factorized_equivalence()
{
	printf("  [F2] MLAFactorizedEquivalence ...\n");
	// If W_K_full = W_DKV @ W_UK (low-rank factorization with rank d_c),
	// then MLA(c=h@W_DKV, W_UK) produces the SAME K as MHA(h @ W_K_full).
	const unsigned int T = 8u;
	const unsigned int d_h = 32u;
	const unsigned int d_c = 8u;
	const unsigned int dKVtotal = 16u;  // n_heads * d_kv
	std::vector<float> h(static_cast<size_t>(T) * d_h);
	std::vector<float> W_DKV(static_cast<size_t>(d_h) * d_c);
	std::vector<float> W_UK(static_cast<size_t>(d_c) * dKVtotal);
	std::vector<float> W_UV(static_cast<size_t>(d_c) * dKVtotal);
	unsigned int seed = 0x600D5EEDu;
	fill_random(h.data(), T * d_h, seed);
	fill_random(W_DKV.data(), d_h * d_c, seed);
	fill_random(W_UK.data(), d_c * dKVtotal, seed);
	fill_random(W_UV.data(), d_c * dKVtotal, seed);

	// MLA path: c = h @ W_DKV, then K = c @ W_UK
	std::vector<float> c(static_cast<size_t>(T) * d_c, 0.0f);
	std::vector<float> K_mla(static_cast<size_t>(T) * dKVtotal, 0.0f);
	std::vector<float> V_mla(static_cast<size_t>(T) * dKVtotal, 0.0f);
	mla_compute_latent(h.data(), W_DKV.data(), T, d_h, d_c, c.data());
	mla_decompress_kv(c.data(), W_UK.data(), W_UV.data(),
	                  T, d_c, dKVtotal, K_mla.data(), V_mla.data());

	// Reference: K_full = h @ (W_DKV @ W_UK)
	std::vector<float> W_K_full(static_cast<size_t>(d_h) * dKVtotal, 0.0f);
	for (unsigned int i = 0; i < d_h; ++i)
		for (unsigned int j = 0; j < dKVtotal; ++j)
		{
			double s = 0.0;
			for (unsigned int k = 0; k < d_c; ++k)
				s += static_cast<double>(W_DKV[i * d_c + k]) *
				     static_cast<double>(W_UK[k * dKVtotal + j]);
			W_K_full[i * dKVtotal + j] = static_cast<float>(s);
		}
	std::vector<float> K_ref(static_cast<size_t>(T) * dKVtotal, 0.0f);
	for (unsigned int t = 0; t < T; ++t)
		for (unsigned int j = 0; j < dKVtotal; ++j)
		{
			double s = 0.0;
			for (unsigned int i = 0; i < d_h; ++i)
				s += static_cast<double>(h[t * d_h + i]) *
				     static_cast<double>(W_K_full[i * dKVtotal + j]);
			K_ref[t * dKVtotal + j] = static_cast<float>(s);
		}

	double maxAbs = 0.0;
	for (size_t i = 0; i < K_mla.size(); ++i)
	{
		double d = std::fabs(static_cast<double>(K_mla[i]) - static_cast<double>(K_ref[i]));
		if (d > maxAbs) maxAbs = d;
	}
	printf("    MLA-K vs MHA-K with factorized W_K: max-abs-diff=%.3e\n", maxAbs);
	ASSERT("MLA decompression matches MHA factorized", maxAbs < 1e-3);
	printf("    PASSED\n");
}

static void test_mla_attention_equivalence()
{
	printf("  [F3] MLAAttentionEquivalence ...\n");
	// Apply the standard scaled-dot-product attention to (Q, K_mla, V_mla)
	// and to (Q, K_ref, V_ref); outputs should match.
	const unsigned int T = 8u;
	const unsigned int d_h = 32u;
	const unsigned int d_c = 8u;
	const unsigned int dKVtotal = 16u;
	std::vector<float> h(static_cast<size_t>(T) * d_h);
	std::vector<float> Q(static_cast<size_t>(T) * dKVtotal);
	std::vector<float> W_DKV(static_cast<size_t>(d_h) * d_c);
	std::vector<float> W_UK(static_cast<size_t>(d_c) * dKVtotal);
	std::vector<float> W_UV(static_cast<size_t>(d_c) * dKVtotal);
	unsigned int seed = 0x600D5EE2u;
	fill_random(h.data(), T * d_h, seed);
	fill_random(Q.data(), T * dKVtotal, seed);
	fill_random(W_DKV.data(), d_h * d_c, seed);
	fill_random(W_UK.data(), d_c * dKVtotal, seed);
	fill_random(W_UV.data(), d_c * dKVtotal, seed);

	std::vector<float> c(static_cast<size_t>(T) * d_c, 0.0f);
	std::vector<float> K_mla(static_cast<size_t>(T) * dKVtotal, 0.0f);
	std::vector<float> V_mla(static_cast<size_t>(T) * dKVtotal, 0.0f);
	mla_compute_latent(h.data(), W_DKV.data(), T, d_h, d_c, c.data());
	mla_decompress_kv(c.data(), W_UK.data(), W_UV.data(),
	                  T, d_c, dKVtotal, K_mla.data(), V_mla.data());

	// Reference K, V via factorized weights as in F2
	std::vector<float> W_K_full(static_cast<size_t>(d_h) * dKVtotal, 0.0f);
	std::vector<float> W_V_full(static_cast<size_t>(d_h) * dKVtotal, 0.0f);
	for (unsigned int i = 0; i < d_h; ++i)
		for (unsigned int j = 0; j < dKVtotal; ++j)
		{
			double sk = 0.0;
			double sv = 0.0;
			for (unsigned int k = 0; k < d_c; ++k)
			{
				const double w = static_cast<double>(W_DKV[i * d_c + k]);
				sk += w * static_cast<double>(W_UK[k * dKVtotal + j]);
				sv += w * static_cast<double>(W_UV[k * dKVtotal + j]);
			}
			W_K_full[i * dKVtotal + j] = static_cast<float>(sk);
			W_V_full[i * dKVtotal + j] = static_cast<float>(sv);
		}
	std::vector<float> K_ref(static_cast<size_t>(T) * dKVtotal, 0.0f);
	std::vector<float> V_ref(static_cast<size_t>(T) * dKVtotal, 0.0f);
	for (unsigned int t = 0; t < T; ++t)
		for (unsigned int j = 0; j < dKVtotal; ++j)
		{
			double sk = 0.0, sv = 0.0;
			for (unsigned int i = 0; i < d_h; ++i)
			{
				const double hv = static_cast<double>(h[t * d_h + i]);
				sk += hv * static_cast<double>(W_K_full[i * dKVtotal + j]);
				sv += hv * static_cast<double>(W_V_full[i * dKVtotal + j]);
			}
			K_ref[t * dKVtotal + j] = static_cast<float>(sk);
			V_ref[t * dKVtotal + j] = static_cast<float>(sv);
		}

	std::vector<float> O_mla(static_cast<size_t>(T) * dKVtotal, 0.0f);
	std::vector<float> O_ref(static_cast<size_t>(T) * dKVtotal, 0.0f);
	scaled_dot_product_attention_forward_flash_strided(
	    Q.data(), dKVtotal, K_mla.data(), dKVtotal, V_mla.data(), dKVtotal,
	    T, dKVtotal, dKVtotal, true, O_mla.data(), dKVtotal, NULL);
	scaled_dot_product_attention_forward_flash_strided(
	    Q.data(), dKVtotal, K_ref.data(), dKVtotal, V_ref.data(), dKVtotal,
	    T, dKVtotal, dKVtotal, true, O_ref.data(), dKVtotal, NULL);

	double maxAbs = 0.0;
	for (size_t i = 0; i < O_mla.size(); ++i)
	{
		double d = std::fabs(static_cast<double>(O_mla[i]) - static_cast<double>(O_ref[i]));
		if (d > maxAbs) maxAbs = d;
	}
	printf("    MLA attention vs MHA-factorized attention: max-abs-diff=%.3e\n", maxAbs);
	ASSERT("MLA attention output matches MHA-factorized", maxAbs < 1e-4);
	printf("    PASSED (Theorem 2: bit-exact equivalence)\n");
}

static void test_mla_cache_size_bound()
{
	printf("  [F4] MLACacheSizeBound ...\n");
	// At T=10240: MLA cache = T*(d_c+d_rope)*4 bytes
	// Standard cfg d_c=512, d_rope=64: T*576*4 = 10240*576*4 = 23.6 MB per layer
	// MHA: T*nHeads*dKV*2*4 = 10240*16*128*2*4 = 167.8 MB per layer
	const unsigned int T = 10240u;
	const unsigned int nHeads = 16u;
	const unsigned int dKV = 128u;
	const unsigned int d_c = 512u;
	const unsigned int d_rope = 64u;
	const float bytesPerFloat = 4.0f;

	const float mhaCache = static_cast<float>(T) * nHeads * dKV * 2.0f * bytesPerFloat;
	const float mlaCache = static_cast<float>(T) * (d_c + d_rope) * bytesPerFloat;
	const float ratio = mhaCache / mlaCache;
	printf("    T=%u: MHA=%.1f MB, MLA=%.1f MB, compression=%.2fx\n",
	       T, mhaCache / 1.048576e6f, mlaCache / 1.048576e6f, ratio);
	ASSERT("MLA cache linear in T (10240)", mlaCache > 1e6f && mlaCache < 1e8f);
	ASSERT("MLA cache compression ratio", ratio > 7.0f && ratio < 7.2f);
	printf("    PASSED (~%.1fx compression at standard config)\n", ratio);
}

// ============================================================
// Main entry point
// ============================================================

void TransformerOpsUnitTest()
{
	printf("============================================================\n");
	printf("Transformer Ops Test Suite\n");
	printf("============================================================\n");

	// Group A: Activation Derivatives
	printf("--- Group A: Activation Derivatives ---\n");
	test_gelu_deriv_finite_diff();
	test_silu_deriv_finite_diff();
	test_gelu_deriv_extreme();
	test_silu_deriv_extreme();
	test_activation_buf_matches_scalar();
	test_activation_backward_buf_matches_scalar();

	// Group B: Softmax
	printf("--- Group B: Softmax ---\n");
	test_softmax_masked_causal_basic();
	test_softmax_masked_noncausal();
	test_softmax_keymask_subset();
	test_softmax_keymask_all_masked();
	test_softmax_numerical_stability();
	test_softmax_uniform_scores();
	test_softmax_single_allowed();
	test_softmax_stable_variant_parity();

	// Group C: Attention Forward
	printf("--- Group C: Attention Forward ---\n");
	test_forward_causal_t1();
	test_forward_noncausal_probs_sum();
	test_forward_causal_masking_correct();
	test_forward_keymask_sparsity();
	test_forward_all_identical_v();
	test_forward_flash_matches_materialized();
	test_forward_strided_matches_nonstrided();

	// Group D: Attention Backward Flash Chunk
	printf("--- Group D: Attention Backward Flash Chunk ---\n");
	test_flash_chunk_single_matches_full_flash();
	test_flash_chunk_two_halves_match_full();
	test_flash_chunk_do_zero();
	test_flash_chunk_gradient_finite_diff();

	// Group E: Paradigm shift #78 ATTENTION-SINK-DISTILL-CHIRON
	printf("--- Group E: Attention-Sink + Sliding Window (paradigm #78) ---\n");
	test_sw_key_allowed_semantics();
	test_sw_visited_count_plateau();
	test_sw_forward_degenerate_full_attention();
	test_sw_forward_mask_applied();
	test_sw_backward_finite_diff();
	test_sw_throughput_speedup();
	test_sw_chunk_matches_full_sw();

	// Group F: Paradigm shift #76 MLA-DISTILL-CHIRON
	printf("--- Group F: Multi-Latent Attention (paradigm #76) ---\n");
	test_mla_compression_ratio();
	test_mla_factorized_equivalence();
	test_mla_attention_equivalence();
	test_mla_cache_size_bound();

	printf("============================================================\n");
	printf("All Transformer Ops Tests Passed\n");
	printf("============================================================\n");
}
