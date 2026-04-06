#include "attention-backward-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/rng.h"
#include "../../../Backend/Machine Learning/Networks/transformer_ops.h"
#include "../../../Backend/Machine Learning/Networks/transformer_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ============================================================
// Helpers
// ============================================================

// Simple deterministic pseudo-random number generator.
static float pseudo_rand(unsigned int& seed)
{
	seed = seed * 1103515245u + 12345u;
	return static_cast<float>(static_cast<int>((seed >> 16) & 0x7FFF)) / 32768.0f - 0.5f;
}

static void fill_random(float* buf, unsigned int n, unsigned int& seed)
{
	for (unsigned int i = 0; i < n; ++i)
		buf[i] = pseudo_rand(seed);
}

// Loss = sum of all elements of the attention output O.
// This makes dO = all-ones, which is the simplest upstream gradient.
static double attention_loss(const float* Q, const float* K, const float* V,
                             unsigned int T, unsigned int dK, unsigned int dV,
                             bool causal)
{
	std::vector<float> O;
	glades::transformer_ops::scaled_dot_product_attention_forward(
		Q, K, V, T, dK, dV, causal, O, NULL);
	double loss = 0.0;
	for (size_t i = 0; i < O.size(); ++i)
		loss += static_cast<double>(O[i]);
	return loss;
}

// Loss with key mask.
static double attention_loss_masked(const float* Q, const float* K, const float* V,
                                    unsigned int T, unsigned int dK, unsigned int dV,
                                    bool causal, const unsigned char* keyAllowed)
{
	std::vector<float> O;
	glades::transformer_ops::scaled_dot_product_attention_forward(
		Q, K, V, T, dK, dV, causal, O, NULL, keyAllowed);
	double loss = 0.0;
	for (size_t i = 0; i < O.size(); ++i)
		loss += static_cast<double>(O[i]);
	return loss;
}

// Finite-difference gradient check for an unmasked loss.
// Returns max relative error across all elements.
// relErr = |analytical - numerical| / max(|analytical|, |numerical|, 1e-7)
static double finite_diff_check(float* param, unsigned int paramSize,
                                const float* Q, const float* K, const float* V,
                                unsigned int T, unsigned int dK, unsigned int dV,
                                bool causal,
                                const float* analyticalGrad,
                                float eps)
{
	double maxRelErr = 0.0;
	for (unsigned int i = 0; i < paramSize; ++i)
	{
		const float orig = param[i];
		param[i] = orig + eps;
		const double lossPlus = attention_loss(Q, K, V, T, dK, dV, causal);
		param[i] = orig - eps;
		const double lossMinus = attention_loss(Q, K, V, T, dK, dV, causal);
		param[i] = orig;

		const double numerical = (lossPlus - lossMinus) / (2.0 * static_cast<double>(eps));
		const double analytical = static_cast<double>(analyticalGrad[i]);
		const double absA = std::abs(analytical);
		const double absN = std::abs(numerical);
		double denom = absA > absN ? absA : absN;
		if (denom < 1e-7)
			denom = 1e-7;
		const double relErr = std::abs(analytical - numerical) / denom;
		if (relErr > maxRelErr)
			maxRelErr = relErr;
	}
	return maxRelErr;
}

// Finite-difference gradient check with key mask.
static double finite_diff_check_masked(float* param, unsigned int paramSize,
                                       const float* Q, const float* K, const float* V,
                                       unsigned int T, unsigned int dK, unsigned int dV,
                                       bool causal, const unsigned char* keyAllowed,
                                       const float* analyticalGrad,
                                       float eps)
{
	double maxRelErr = 0.0;
	for (unsigned int i = 0; i < paramSize; ++i)
	{
		const float orig = param[i];
		param[i] = orig + eps;
		const double lossPlus = attention_loss_masked(Q, K, V, T, dK, dV, causal, keyAllowed);
		param[i] = orig - eps;
		const double lossMinus = attention_loss_masked(Q, K, V, T, dK, dV, causal, keyAllowed);
		param[i] = orig;

		const double numerical = (lossPlus - lossMinus) / (2.0 * static_cast<double>(eps));
		const double analytical = static_cast<double>(analyticalGrad[i]);
		const double absA = std::abs(analytical);
		const double absN = std::abs(numerical);
		double denom = absA > absN ? absA : absN;
		if (denom < 1e-7)
			denom = 1e-7;
		const double relErr = std::abs(analytical - numerical) / denom;
		if (relErr > maxRelErr)
			maxRelErr = relErr;
	}
	return maxRelErr;
}

// Max absolute difference between two buffers.
static double max_abs_diff(const float* a, const float* b, unsigned int n)
{
	double maxDiff = 0.0;
	for (unsigned int i = 0; i < n; ++i)
	{
		const double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
		if (diff > maxDiff)
			maxDiff = diff;
	}
	return maxDiff;
}

// ============================================================
// Test functions
// ============================================================

// Test 1: Finite-diff recompute causal (T=4, dK=3, dV=3)
static void test_finite_diff_recompute_causal()
{
	printf("  [1] FiniteDiff Recompute Causal (T=4, dK=3, dV=3) ...\n");

	const unsigned int T = 4, dK = 3, dV = 3;
	const unsigned int qSize = T * dK;
	const unsigned int kSize = T * dK;
	const unsigned int vSize = T * dV;

	std::vector<float> Q(qSize), K(kSize), V(vSize);
	unsigned int seed = 42u;
	fill_random(&Q[0], qSize, seed);
	fill_random(&K[0], kSize, seed);
	fill_random(&V[0], vSize, seed);

	// dO = all ones (loss = sum(O))
	std::vector<float> dO(T * dV, 1.0f);

	// Analytical backward
	std::vector<float> dQ, dK_out, dV_out;
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute(
		&Q[0], &K[0], &V[0], &dO[0], T, dK, dV, true, dQ, dK_out, dV_out);

	const float eps = 1e-4f;
	const double tol = 5e-2;

	// Check Q gradients
	double errQ = finite_diff_check(&Q[0], qSize, &Q[0], &K[0], &V[0], T, dK, dV, true, &dQ[0], eps);
	printf("    dQ maxRelErr = %.6e\n", errQ);
	ASSERT("dQ gradient check failed (causal recompute)", errQ < tol);

	// Check K gradients
	double errK = finite_diff_check(&K[0], kSize, &Q[0], &K[0], &V[0], T, dK, dV, true, &dK_out[0], eps);
	printf("    dK maxRelErr = %.6e\n", errK);
	ASSERT("dK gradient check failed (causal recompute)", errK < tol);

	// Check V gradients
	double errV = finite_diff_check(&V[0], vSize, &Q[0], &K[0], &V[0], T, dK, dV, true, &dV_out[0], eps);
	printf("    dV maxRelErr = %.6e\n", errV);
	ASSERT("dV gradient check failed (causal recompute)", errV < tol);

	printf("    PASSED\n");
}

// Test 2: Finite-diff recompute non-causal (T=3, dK=4, dV=2)
static void test_finite_diff_recompute_noncausal()
{
	printf("  [2] FiniteDiff Recompute NonCausal (T=3, dK=4, dV=2) ...\n");

	const unsigned int T = 3, dK = 4, dV = 2;
	const unsigned int qSize = T * dK;
	const unsigned int kSize = T * dK;
	const unsigned int vSize = T * dV;

	std::vector<float> Q(qSize), K(kSize), V(vSize);
	unsigned int seed = 123u;
	fill_random(&Q[0], qSize, seed);
	fill_random(&K[0], kSize, seed);
	fill_random(&V[0], vSize, seed);

	std::vector<float> dO(T * dV, 1.0f);

	std::vector<float> dQ, dK_out, dV_out;
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute(
		&Q[0], &K[0], &V[0], &dO[0], T, dK, dV, false, dQ, dK_out, dV_out);

	const float eps = 1e-4f;
	const double tol = 5e-2;

	double errQ = finite_diff_check(&Q[0], qSize, &Q[0], &K[0], &V[0], T, dK, dV, false, &dQ[0], eps);
	printf("    dQ maxRelErr = %.6e\n", errQ);
	ASSERT("dQ gradient check failed (noncausal recompute)", errQ < tol);

	double errK = finite_diff_check(&K[0], kSize, &Q[0], &K[0], &V[0], T, dK, dV, false, &dK_out[0], eps);
	printf("    dK maxRelErr = %.6e\n", errK);
	ASSERT("dK gradient check failed (noncausal recompute)", errK < tol);

	double errV = finite_diff_check(&V[0], vSize, &Q[0], &K[0], &V[0], T, dK, dV, false, &dV_out[0], eps);
	printf("    dV maxRelErr = %.6e\n", errV);
	ASSERT("dV gradient check failed (noncausal recompute)", errV < tol);

	printf("    PASSED\n");
}

// Test 3: Finite-diff flash-strided causal (T=4, dK=3, dV=3)
static void test_finite_diff_flash_strided_causal()
{
	printf("  [3] FiniteDiff Flash-Strided Causal (T=4, dK=3, dV=3) ...\n");

	const unsigned int T = 4, dK = 3, dV = 3;
	const unsigned int qSize = T * dK;
	const unsigned int kSize = T * dK;
	const unsigned int vSize = T * dV;

	std::vector<float> Q(qSize), K(kSize), V(vSize);
	unsigned int seed = 7u;
	fill_random(&Q[0], qSize, seed);
	fill_random(&K[0], kSize, seed);
	fill_random(&V[0], vSize, seed);

	std::vector<float> dO(T * dV, 1.0f);

	// Flash-strided backward (contiguous layout: stride = dK for Q/K, dV for V/dO)
	std::vector<float> dQ(qSize, 0.0f), dK_out(kSize, 0.0f), dV_out(vSize, 0.0f);
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_strided(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		T, dK, dV, true,
		&dQ[0], dK, &dK_out[0], dK, &dV_out[0], dV);

	const float eps = 1e-4f;
	const double tol = 5e-2;

	double errQ = finite_diff_check(&Q[0], qSize, &Q[0], &K[0], &V[0], T, dK, dV, true, &dQ[0], eps);
	printf("    dQ maxRelErr = %.6e\n", errQ);
	ASSERT("dQ gradient check failed (flash-strided causal)", errQ < tol);

	double errK = finite_diff_check(&K[0], kSize, &Q[0], &K[0], &V[0], T, dK, dV, true, &dK_out[0], eps);
	printf("    dK maxRelErr = %.6e\n", errK);
	ASSERT("dK gradient check failed (flash-strided causal)", errK < tol);

	double errV = finite_diff_check(&V[0], vSize, &Q[0], &K[0], &V[0], T, dK, dV, true, &dV_out[0], eps);
	printf("    dV maxRelErr = %.6e\n", errV);
	ASSERT("dV gradient check failed (flash-strided causal)", errV < tol);

	printf("    PASSED\n");
}

// Test 4: Flash vs Reference Parity (T=8, dK=8, dV=8)
static void test_flash_vs_reference_parity()
{
	printf("  [4] Flash vs Reference Parity (T=8, dK=8, dV=8) ...\n");

	const unsigned int T = 8, dK = 8, dV = 8;
	const unsigned int qSize = T * dK;
	const unsigned int kSize = T * dK;
	const unsigned int vSize = T * dV;

	std::vector<float> Q(qSize), K(kSize), V(vSize);
	unsigned int seed = 999u;
	fill_random(&Q[0], qSize, seed);
	fill_random(&K[0], kSize, seed);
	fill_random(&V[0], vSize, seed);

	std::vector<float> dO(T * dV, 1.0f);

	// Reference backward
	std::vector<float> dQ_ref, dK_ref, dV_ref;
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute(
		&Q[0], &K[0], &V[0], &dO[0], T, dK, dV, true, dQ_ref, dK_ref, dV_ref);

	// Flash-strided backward (contiguous)
	std::vector<float> dQ_flash(qSize, 0.0f), dK_flash(kSize, 0.0f), dV_flash(vSize, 0.0f);
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_strided(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		T, dK, dV, true,
		&dQ_flash[0], dK, &dK_flash[0], dK, &dV_flash[0], dV);

	const double tol = 1e-4;

	double diffQ = max_abs_diff(&dQ_ref[0], &dQ_flash[0], qSize);
	printf("    dQ maxAbsDiff = %.6e\n", diffQ);
	ASSERT("dQ flash vs reference mismatch", diffQ < tol);

	double diffK = max_abs_diff(&dK_ref[0], &dK_flash[0], kSize);
	printf("    dK maxAbsDiff = %.6e\n", diffK);
	ASSERT("dK flash vs reference mismatch", diffK < tol);

	double diffV = max_abs_diff(&dV_ref[0], &dV_flash[0], vSize);
	printf("    dV maxAbsDiff = %.6e\n", diffV);
	ASSERT("dV flash vs reference mismatch", diffV < tol);

	printf("    PASSED\n");
}

// Test 5: Flash Chunk Correctness (T=6, dK=4, dV=4)
static void test_flash_chunk_correctness()
{
	printf("  [5] Flash Chunk Correctness (T=6, dK=4, dV=4) ...\n");

	const unsigned int T = 6, dK = 4, dV = 4;
	const unsigned int qSize = T * dK;
	const unsigned int kSize = T * dK;
	const unsigned int vSize = T * dV;

	std::vector<float> Q(qSize), K(kSize), V(vSize);
	unsigned int seed = 314u;
	fill_random(&Q[0], qSize, seed);
	fill_random(&K[0], kSize, seed);
	fill_random(&V[0], vSize, seed);

	std::vector<float> dO(T * dV, 1.0f);

	// Full flash backward
	std::vector<float> dQ_full(qSize, 0.0f), dK_full(kSize, 0.0f), dV_full(vSize, 0.0f);
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_strided(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		T, dK, dV, true,
		&dQ_full[0], dK, &dK_full[0], dK, &dV_full[0], dV);

	// Chunked backward: chunk1 [0,3), chunk2 [3,6)
	std::vector<float> dQ_chunked(qSize, 0.0f);
	std::vector<float> dK_local1(kSize, 0.0f), dV_local1(vSize, 0.0f);
	std::vector<float> dK_local2(kSize, 0.0f), dV_local2(vSize, 0.0f);

	// Chunk 1: rows [0, 3)
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_chunk(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		0, 3, T, dK, dV, true,
		&dQ_chunked[0], dK, &dK_local1[0], &dV_local1[0], NULL);

	// Chunk 2: rows [3, 6)
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_chunk(
		&Q[0], dK, &K[0], dK, &V[0], dV, &dO[0], dV,
		3, 6, T, dK, dV, true,
		&dQ_chunked[0], dK, &dK_local2[0], &dV_local2[0], NULL);

	// Accumulate dK and dV from both chunks
	std::vector<float> dK_chunked(kSize, 0.0f), dV_chunked(vSize, 0.0f);
	for (unsigned int i = 0; i < kSize; ++i)
		dK_chunked[i] = dK_local1[i] + dK_local2[i];
	for (unsigned int i = 0; i < vSize; ++i)
		dV_chunked[i] = dV_local1[i] + dV_local2[i];

	const double tol = 1e-6;

	double diffQ = max_abs_diff(&dQ_chunked[0], &dQ_full[0], qSize);
	printf("    dQ maxAbsDiff = %.6e\n", diffQ);
	ASSERT("dQ chunk vs full mismatch", diffQ < tol);

	double diffK = max_abs_diff(&dK_chunked[0], &dK_full[0], kSize);
	printf("    dK maxAbsDiff = %.6e\n", diffK);
	ASSERT("dK chunk vs full mismatch", diffK < tol);

	double diffV = max_abs_diff(&dV_chunked[0], &dV_full[0], vSize);
	printf("    dV maxAbsDiff = %.6e\n", diffV);
	ASSERT("dV chunk vs full mismatch", diffV < tol);

	printf("    PASSED\n");
}

// Test 6: T=1 Edge Case (dK=4, dV=4, causal=true)
static void test_t1_edge_case()
{
	printf("  [6] T=1 Edge Case (dK=4, dV=4, causal) ...\n");

	const unsigned int T = 1, dK = 4, dV = 4;

	float Q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
	float K[4] = {0.0f, 1.0f, 0.0f, 0.0f};
	float V[4] = {0.0f, 0.0f, 1.0f, 0.0f};
	float dO[4] = {1.0f, 1.0f, 1.0f, 1.0f};

	std::vector<float> dQ, dK_out, dV_out;
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute(
		Q, K, V, dO, T, dK, dV, true, dQ, dK_out, dV_out);

	const double tol = 1e-6;

	// With T=1 causal, softmax of a single element -> probs = [1.0]
	// O = V, so dV = dO
	// Softmax gradient is zero with 1 element, so dQ and dK should be ~0
	bool dQ_zero = true;
	for (unsigned int i = 0; i < dK; ++i)
	{
		if (std::abs(static_cast<double>(dQ[i])) > tol)
		{
			dQ_zero = false;
			break;
		}
	}
	printf("    dQ all near zero: %s\n", dQ_zero ? "yes" : "no");
	ASSERT("dQ should be ~0 for T=1", dQ_zero);

	bool dK_zero = true;
	for (unsigned int i = 0; i < dK; ++i)
	{
		if (std::abs(static_cast<double>(dK_out[i])) > tol)
		{
			dK_zero = false;
			break;
		}
	}
	printf("    dK all near zero: %s\n", dK_zero ? "yes" : "no");
	ASSERT("dK should be ~0 for T=1", dK_zero);

	// dV should equal dO
	double maxDiffDV = 0.0;
	for (unsigned int i = 0; i < dV; ++i)
	{
		const double diff = std::abs(static_cast<double>(dV_out[i]) - static_cast<double>(dO[i]));
		if (diff > maxDiffDV)
			maxDiffDV = diff;
	}
	printf("    dV vs dO maxAbsDiff = %.6e\n", maxDiffDV);
	ASSERT("dV should equal dO for T=1", maxDiffDV < tol);

	printf("    PASSED\n");
}

// Test 7: T=0 Empty Case
static void test_t0_empty_case()
{
	printf("  [7] T=0 Empty Case (dK=2, dV=2) ...\n");

	const unsigned int T = 0, dK = 2, dV = 2;

	// Passing NULL pointers since T=0 should early-exit
	std::vector<float> dQ, dK_out, dV_out;
	float dummy = 0.0f;
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute(
		&dummy, &dummy, &dummy, &dummy, T, dK, dV, true, dQ, dK_out, dV_out);

	ASSERT("dQ should be empty for T=0", dQ.empty());
	ASSERT("dK should be empty for T=0", dK_out.empty());
	ASSERT("dV should be empty for T=0", dV_out.empty());

	printf("    dQ.size()=%u dK.size()=%u dV.size()=%u\n",
	       (unsigned int)dQ.size(), (unsigned int)dK_out.size(), (unsigned int)dV_out.size());
	printf("    PASSED\n");
}

// Test 8: Key Mask Partial (T=4, dK=3, dV=3, causal=false)
static void test_key_mask_partial()
{
	printf("  [8] Key Mask Partial (T=4, dK=3, dV=3, causal=false) ...\n");

	const unsigned int T = 4, dK = 3, dV = 3;
	const unsigned int qSize = T * dK;
	const unsigned int kSize = T * dK;
	const unsigned int vSize = T * dV;

	std::vector<float> Q(qSize), K(kSize), V(vSize);
	unsigned int seed = 77u;
	fill_random(&Q[0], qSize, seed);
	fill_random(&K[0], kSize, seed);
	fill_random(&V[0], vSize, seed);

	// keyAllowed = [1, 0, 1, 0] -- mask out keys 1 and 3
	unsigned char keyAllowed[4] = {1, 0, 1, 0};

	std::vector<float> dO(T * dV, 1.0f);

	// Analytical backward with key mask
	std::vector<float> dQ_a, dK_a, dV_a;
	glades::transformer_ops::scaled_dot_product_attention_backward_recompute(
		&Q[0], &K[0], &V[0], &dO[0], T, dK, dV, false, dQ_a, dK_a, dV_a, keyAllowed);

	// Check that dK for masked keys (rows 1 and 3) is zero
	const double tol_zero = 1e-6;
	bool masked_k_zero = true;
	for (unsigned int k = 0; k < dK; ++k)
	{
		// Key row 1
		if (std::abs(static_cast<double>(dK_a[1 * dK + k])) > tol_zero)
			masked_k_zero = false;
		// Key row 3
		if (std::abs(static_cast<double>(dK_a[3 * dK + k])) > tol_zero)
			masked_k_zero = false;
	}
	printf("    dK masked rows (1,3) near zero: %s\n", masked_k_zero ? "yes" : "no");
	ASSERT("dK for masked keys should be ~0", masked_k_zero);

	// Check that dK for allowed keys (rows 0 and 2) is non-zero
	double maxAllowed = 0.0;
	for (unsigned int k = 0; k < dK; ++k)
	{
		double v0 = std::abs(static_cast<double>(dK_a[0 * dK + k]));
		double v2 = std::abs(static_cast<double>(dK_a[2 * dK + k]));
		if (v0 > maxAllowed) maxAllowed = v0;
		if (v2 > maxAllowed) maxAllowed = v2;
	}
	printf("    dK allowed rows (0,2) max abs = %.6e\n", maxAllowed);
	ASSERT("dK for allowed keys should be non-zero", maxAllowed > 1e-6);

	// Also check dV for masked keys (rows 1 and 3) is zero
	bool masked_v_zero = true;
	for (unsigned int d = 0; d < dV; ++d)
	{
		if (std::abs(static_cast<double>(dV_a[1 * dV + d])) > tol_zero)
			masked_v_zero = false;
		if (std::abs(static_cast<double>(dV_a[3 * dV + d])) > tol_zero)
			masked_v_zero = false;
	}
	printf("    dV masked rows (1,3) near zero: %s\n", masked_v_zero ? "yes" : "no");
	ASSERT("dV for masked keys should be ~0", masked_v_zero);

	// Finite-diff check for Q and K with mask
	const float eps = 1e-3f;
	const double tol_grad = 5e-2;

	double errQ = finite_diff_check_masked(&Q[0], qSize, &Q[0], &K[0], &V[0],
	                                       T, dK, dV, false, keyAllowed, &dQ_a[0], eps);
	printf("    dQ maxRelErr (masked) = %.6e\n", errQ);
	ASSERT("dQ gradient check failed (key mask)", errQ < tol_grad);

	double errK = finite_diff_check_masked(&K[0], kSize, &Q[0], &K[0], &V[0],
	                                       T, dK, dV, false, keyAllowed, &dK_a[0], eps);
	printf("    dK maxRelErr (masked) = %.6e\n", errK);
	ASSERT("dK gradient check failed (key mask)", errK < tol_grad);

	double errV = finite_diff_check_masked(&V[0], vSize, &Q[0], &K[0], &V[0],
	                                       T, dK, dV, false, keyAllowed, &dV_a[0], eps);
	printf("    dV maxRelErr (masked) = %.6e\n", errV);
	ASSERT("dV gradient check failed (key mask)", errV < tol_grad);

	printf("    PASSED\n");
}

// ============================================================
// Main test entry point
// ============================================================

void AttentionBackwardUnitTest()
{
	printf("============================================================\n");
	printf("Attention Backward Test Suite\n");
	printf("============================================================\n");

	test_finite_diff_recompute_causal();
	test_finite_diff_recompute_noncausal();
	test_finite_diff_flash_strided_causal();
	test_flash_vs_reference_parity();
	test_flash_chunk_correctness();
	test_t1_edge_case();
	test_t0_empty_case();
	test_key_mask_partial();

	printf("============================================================\n");
	printf("All Attention Backward Tests Passed\n");
	printf("============================================================\n");
}
