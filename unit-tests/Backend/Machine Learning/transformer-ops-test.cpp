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

	printf("============================================================\n");
	printf("All Transformer Ops Tests Passed\n");
	printf("============================================================\n");
}
