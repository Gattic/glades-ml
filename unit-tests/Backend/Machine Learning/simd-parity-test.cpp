#include "simd-parity-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/rng.h"
#include "../../../Backend/Machine Learning/Networks/transformer_kernels.h"
#include "../../../Backend/Machine Learning/Networks/transformer_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using glades::transformer_kernels::dot_f32;
using glades::transformer_kernels::axpy_f32;
using glades::transformer_kernels::linear_into;
using glades::transformer_kernels::linear_into_opt;
using glades::transformer_kernels::softmax_stable_inplace;
using glades::transformer_kernels::all_finite_full;

// ============================================================
// Double-precision reference implementations
// ============================================================

static double double_dot(const float* a, const float* b, unsigned int n)
{
	double acc = 0.0;
	for (unsigned int i = 0; i < n; ++i)
		acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
	return acc;
}

// ============================================================
// Pseudo-random data helpers
// ============================================================

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

// ============================================================
// Relative error helper
// ============================================================

static double rel_err(double result, double ref)
{
	if (ref == 0.0 && result == 0.0)
		return 0.0;
	double denom = std::fabs(ref);
	if (denom < 1e-12)
		denom = 1e-12;
	return std::fabs(result - ref) / denom;
}

// ============================================================
// SIMDParityUnitTest
// ============================================================

void SIMDParityUnitTest()
{
	printf("============================================================\n");
	printf("SIMD Parity Test Suite\n");
	printf("============================================================\n");

	unsigned int seed = 42u;

	// ----------------------------------------------------------
	// dot_f32 Tests
	// ----------------------------------------------------------

	// Test 1: N=0 returns 0
	{
		float a[1] = {1.0f};
		float b[1] = {1.0f};
		float result = dot_f32(a, b, 0);
		ASSERT("dot_f32: N=0 should return 0", result == 0.0f);
		printf("[PASS] dot_f32 Test 1: N=0 returns 0\n");
	}

	// Test 2: NULL pointers return 0
	{
		float a[10];
		float b[10];
		fill_random(a, 10, seed);
		fill_random(b, 10, seed);
		float r1 = dot_f32(NULL, b, 10);
		float r2 = dot_f32(a, NULL, 10);
		ASSERT("dot_f32: NULL first arg should return 0", r1 == 0.0f);
		ASSERT("dot_f32: NULL second arg should return 0", r2 == 0.0f);
		printf("[PASS] dot_f32 Test 2: NULL pointers return 0\n");
	}

	// Test 3: N=1
	{
		float a[1] = {3.14f};
		float b[1] = {2.71f};
		float result = dot_f32(a, b, 1);
		double ref = double_dot(a, b, 1);
		double err = rel_err(static_cast<double>(result), ref);
		ASSERT("dot_f32: N=1 parity", err < 1e-6);
		printf("[PASS] dot_f32 Test 3: N=1 (relErr=%.2e)\n", err);
	}

	// Test 4: N=7 (not multiple of 8)
	{
		const unsigned int N = 7;
		float a[7], b[7];
		fill_random(a, N, seed);
		fill_random(b, N, seed);
		float result = dot_f32(a, b, N);
		double ref = double_dot(a, b, N);
		double err = rel_err(static_cast<double>(result), ref);
		ASSERT("dot_f32: N=7 parity", err < 1e-5);
		printf("[PASS] dot_f32 Test 4: N=7 (relErr=%.2e)\n", err);
	}

	// Test 5: N=8 (exactly one SIMD iteration)
	{
		const unsigned int N = 8;
		float a[8], b[8];
		fill_random(a, N, seed);
		fill_random(b, N, seed);
		float result = dot_f32(a, b, N);
		double ref = double_dot(a, b, N);
		double err = rel_err(static_cast<double>(result), ref);
		ASSERT("dot_f32: N=8 parity", err < 1e-5);
		printf("[PASS] dot_f32 Test 5: N=8 (relErr=%.2e)\n", err);
	}

	// Test 6: N=31 (boundary of 32-element unrolling)
	{
		const unsigned int N = 31;
		std::vector<float> a(N), b(N);
		fill_random(&a[0], N, seed);
		fill_random(&b[0], N, seed);
		float result = dot_f32(&a[0], &b[0], N);
		double ref = double_dot(&a[0], &b[0], N);
		double err = rel_err(static_cast<double>(result), ref);
		ASSERT("dot_f32: N=31 parity", err < 1e-4);
		printf("[PASS] dot_f32 Test 6: N=31 (relErr=%.2e)\n", err);
	}

	// Test 7: N=32 (exactly one 32-wide iteration)
	{
		const unsigned int N = 32;
		std::vector<float> a(N), b(N);
		fill_random(&a[0], N, seed);
		fill_random(&b[0], N, seed);
		float result = dot_f32(&a[0], &b[0], N);
		double ref = double_dot(&a[0], &b[0], N);
		double err = rel_err(static_cast<double>(result), ref);
		ASSERT("dot_f32: N=32 parity", err < 1e-4);
		printf("[PASS] dot_f32 Test 7: N=32 (relErr=%.2e)\n", err);
	}

	// Test 8: N=33 (32-wide + tail)
	{
		const unsigned int N = 33;
		std::vector<float> a(N), b(N);
		fill_random(&a[0], N, seed);
		fill_random(&b[0], N, seed);
		float result = dot_f32(&a[0], &b[0], N);
		double ref = double_dot(&a[0], &b[0], N);
		double err = rel_err(static_cast<double>(result), ref);
		ASSERT("dot_f32: N=33 parity", err < 1e-4);
		printf("[PASS] dot_f32 Test 8: N=33 (relErr=%.2e)\n", err);
	}

	// Test 9: N=1024 (large)
	{
		const unsigned int N = 1024;
		std::vector<float> a(N), b(N);
		fill_random(&a[0], N, seed);
		fill_random(&b[0], N, seed);
		float result = dot_f32(&a[0], &b[0], N);
		double ref = double_dot(&a[0], &b[0], N);
		double err = rel_err(static_cast<double>(result), ref);
		ASSERT("dot_f32: N=1024 parity", err < 1e-3);
		printf("[PASS] dot_f32 Test 9: N=1024 (relErr=%.2e)\n", err);
	}

	// Test 10: N=10000 precision bound
	{
		const unsigned int N = 10000;
		std::vector<float> a(N), b(N);
		for (unsigned int i = 0; i < N; ++i)
		{
			a[i] = 1.0f;
			b[i] = 1e-4f;
		}
		float result = dot_f32(&a[0], &b[0], N);
		double expected = 1.0;
		double err = rel_err(static_cast<double>(result), expected);
		ASSERT("dot_f32: N=10000 precision bound", err < 1e-2);
		printf("[PASS] dot_f32 Test 10: N=10000 precision (relErr=%.2e)\n", err);
	}

	// ----------------------------------------------------------
	// axpy_f32 Tests
	// ----------------------------------------------------------

	// Test 11: N=0 does nothing
	{
		float y[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		float x[4] = {10.0f, 20.0f, 30.0f, 40.0f};
		float y_orig[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		axpy_f32(y, x, 2.0f, 0);
		bool unchanged = true;
		for (int i = 0; i < 4; ++i)
		{
			if (y[i] != y_orig[i])
				unchanged = false;
		}
		ASSERT("axpy_f32: N=0 should leave y unchanged", unchanged);
		printf("[PASS] axpy_f32 Test 11: N=0 does nothing\n");
	}

	// Test 12: NULL pointers do nothing (no crash)
	{
		float x[10], y[10];
		fill_random(x, 10, seed);
		fill_random(y, 10, seed);
		axpy_f32(NULL, x, 2.0f, 10);
		axpy_f32(y, NULL, 2.0f, 10);
		// If we get here without crashing, it passed
		printf("[PASS] axpy_f32 Test 12: NULL pointers no crash\n");
	}

	// Test 13: N=7
	{
		const unsigned int N = 7;
		float x[7], y[7], y_orig[7];
		fill_random(x, N, seed);
		fill_random(y, N, seed);
		std::memcpy(y_orig, y, N * sizeof(float));
		axpy_f32(y, x, 2.5f, N);
		bool ok = true;
		for (unsigned int i = 0; i < N; ++i)
		{
			float expected = y_orig[i] + 2.5f * x[i];
			if (std::fabs(y[i] - expected) > 1e-6f)
				ok = false;
		}
		ASSERT("axpy_f32: N=7 element-wise parity", ok);
		printf("[PASS] axpy_f32 Test 13: N=7\n");
	}

	// Test 14: N=32
	{
		const unsigned int N = 32;
		std::vector<float> x(N), y(N), y_orig(N);
		fill_random(&x[0], N, seed);
		fill_random(&y[0], N, seed);
		std::memcpy(&y_orig[0], &y[0], N * sizeof(float));
		axpy_f32(&y[0], &x[0], 2.5f, N);
		bool ok = true;
		for (unsigned int i = 0; i < N; ++i)
		{
			float expected = y_orig[i] + 2.5f * x[i];
			if (std::fabs(y[i] - expected) > 1e-6f)
				ok = false;
		}
		ASSERT("axpy_f32: N=32 element-wise parity", ok);
		printf("[PASS] axpy_f32 Test 14: N=32\n");
	}

	// Test 15: N=1024
	{
		const unsigned int N = 1024;
		std::vector<float> x(N), y(N), y_orig(N);
		fill_random(&x[0], N, seed);
		fill_random(&y[0], N, seed);
		std::memcpy(&y_orig[0], &y[0], N * sizeof(float));
		axpy_f32(&y[0], &x[0], -3.7f, N);
		float maxErr = 0.0f;
		for (unsigned int i = 0; i < N; ++i)
		{
			float expected = y_orig[i] + (-3.7f) * x[i];
			float err = std::fabs(y[i] - expected);
			if (err > maxErr)
				maxErr = err;
		}
		ASSERT("axpy_f32: N=1024 max element-wise error", maxErr < 1e-5f);
		printf("[PASS] axpy_f32 Test 15: N=1024 (maxErr=%.2e)\n", maxErr);
	}

	// Test 16: alpha=0
	{
		const unsigned int N = 100;
		std::vector<float> x(N), y(N), y_orig(N);
		fill_random(&x[0], N, seed);
		fill_random(&y[0], N, seed);
		std::memcpy(&y_orig[0], &y[0], N * sizeof(float));
		axpy_f32(&y[0], &x[0], 0.0f, N);
		bool unchanged = true;
		for (unsigned int i = 0; i < N; ++i)
		{
			if (y[i] != y_orig[i])
				unchanged = false;
		}
		ASSERT("axpy_f32: alpha=0 should leave y unchanged", unchanged);
		printf("[PASS] axpy_f32 Test 16: alpha=0\n");
	}

	// Test 17: alpha=-1 (subtraction)
	{
		const unsigned int N = 100;
		std::vector<float> x(N), y(N);
		fill_random(&x[0], N, seed);
		for (unsigned int i = 0; i < N; ++i)
			y[i] = x[i];
		axpy_f32(&y[0], &x[0], -1.0f, N);
		bool allZero = true;
		for (unsigned int i = 0; i < N; ++i)
		{
			if (std::fabs(y[i]) > 1e-7f)
				allZero = false;
		}
		ASSERT("axpy_f32: alpha=-1 should zero out y", allZero);
		printf("[PASS] axpy_f32 Test 17: alpha=-1 (subtraction)\n");
	}

	// ----------------------------------------------------------
	// linear_into_opt vs linear_into Tests
	// ----------------------------------------------------------

	// Test 18: Small (inSize=4, outSize=3)
	{
		const unsigned int inSize = 4;
		const unsigned int outSize = 3;
		std::vector<float> W(outSize * inSize), b(outSize), x(inSize);
		fill_random(&W[0], outSize * inSize, seed);
		fill_random(&b[0], outSize, seed);
		fill_random(&x[0], inSize, seed);

		std::vector<float> y_ref(outSize, 0.0f);
		std::vector<float> y_opt(outSize, 0.0f);
		linear_into(&x[0], inSize, W, b, outSize, &y_ref[0]);
		linear_into_opt(&x[0], inSize, W, b, outSize, &y_opt[0]);

		float maxDiff = 0.0f;
		for (unsigned int j = 0; j < outSize; ++j)
		{
			float diff = std::fabs(y_ref[j] - y_opt[j]);
			if (diff > maxDiff)
				maxDiff = diff;
		}
		ASSERT("linear_into vs linear_into_opt: small (4x3)", maxDiff < 1e-4f);
		printf("[PASS] linear Test 18: inSize=4, outSize=3 (maxDiff=%.2e)\n", maxDiff);
	}

	// Test 19: Medium (inSize=64, outSize=16)
	{
		const unsigned int inSize = 64;
		const unsigned int outSize = 16;
		std::vector<float> W(outSize * inSize), b(outSize), x(inSize);
		fill_random(&W[0], outSize * inSize, seed);
		fill_random(&b[0], outSize, seed);
		fill_random(&x[0], inSize, seed);

		std::vector<float> y_ref(outSize, 0.0f);
		std::vector<float> y_opt(outSize, 0.0f);
		linear_into(&x[0], inSize, W, b, outSize, &y_ref[0]);
		linear_into_opt(&x[0], inSize, W, b, outSize, &y_opt[0]);

		float maxDiff = 0.0f;
		for (unsigned int j = 0; j < outSize; ++j)
		{
			float diff = std::fabs(y_ref[j] - y_opt[j]);
			if (diff > maxDiff)
				maxDiff = diff;
		}
		ASSERT("linear_into vs linear_into_opt: medium (64x16)", maxDiff < 1e-3f);
		printf("[PASS] linear Test 19: inSize=64, outSize=16 (maxDiff=%.2e)\n", maxDiff);
	}

	// ----------------------------------------------------------
	// softmax_stable_inplace Tests
	// ----------------------------------------------------------

	// Test 20: Sum to one
	{
		const unsigned int N = 100;
		std::vector<float> logits(N);
		fill_random(&logits[0], N, seed);
		softmax_stable_inplace(logits);
		double sum = 0.0;
		bool allNonNeg = true;
		for (unsigned int i = 0; i < N; ++i)
		{
			sum += static_cast<double>(logits[i]);
			if (logits[i] < 0.0f)
				allNonNeg = false;
		}
		ASSERT("softmax: sum to one", std::fabs(sum - 1.0) < 1e-5);
		ASSERT("softmax: all non-negative", allNonNeg);
		printf("[PASS] softmax Test 20: sum=%.8f, all non-negative\n", sum);
	}

	// Test 21: Large values no overflow
	{
		std::vector<float> logits(3);
		logits[0] = 1000.0f;
		logits[1] = 999.0f;
		logits[2] = 998.0f;
		softmax_stable_inplace(logits);
		double sum = 0.0;
		bool finite = true;
		for (unsigned int i = 0; i < 3; ++i)
		{
			sum += static_cast<double>(logits[i]);
			if (logits[i] != logits[i])
				finite = false; // NaN check
			// Inf check
			if (logits[i] > 1e30f || logits[i] < -1e30f)
				finite = false;
		}
		ASSERT("softmax: large values sum to one", std::fabs(sum - 1.0) < 1e-5);
		ASSERT("softmax: large values remain finite", finite);
		printf("[PASS] softmax Test 21: large values, sum=%.8f\n", sum);
	}

	// Test 22: N=1
	{
		std::vector<float> logits(1);
		logits[0] = 42.0f;
		softmax_stable_inplace(logits);
		ASSERT("softmax: N=1 should be 1.0", std::fabs(logits[0] - 1.0f) < 1e-7f);
		printf("[PASS] softmax Test 22: N=1, prob=%.8f\n", logits[0]);
	}

	// Test 23: All equal
	{
		std::vector<float> logits(4, 5.0f);
		softmax_stable_inplace(logits);
		bool allEqual = true;
		for (unsigned int i = 0; i < 4; ++i)
		{
			if (std::fabs(logits[i] - 0.25f) > 1e-6f)
				allEqual = false;
		}
		ASSERT("softmax: all equal should give uniform", allEqual);
		printf("[PASS] softmax Test 23: all equal -> uniform (p[0]=%.8f)\n", logits[0]);
	}

	// ----------------------------------------------------------
	// all_finite_full Tests
	// ----------------------------------------------------------

	// Test 24: Denormalized floats are finite
	{
		float denorm = 1e-45f;
		bool result = all_finite_full(&denorm, 1);
		ASSERT("all_finite_full: denormalized float is finite", result == true);
		printf("[PASS] all_finite_full Test 24: denormalized float\n");
	}

	// Test 25: -0.0f is finite
	{
		float negzero = -0.0f;
		bool result = all_finite_full(&negzero, 1);
		ASSERT("all_finite_full: -0.0f is finite", result == true);
		printf("[PASS] all_finite_full Test 25: -0.0f\n");
	}

	printf("============================================================\n");
	printf("All SIMD Parity Tests Passed\n");
	printf("============================================================\n");
}
