#include "transformer-kernels-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/rng.h"
#include "../../../Backend/Machine Learning/Networks/transformer_kernels.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace glades::transformer_kernels;

// ============================================================
// Pseudo-random data helpers
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

// ============================================================
// TransformerKernelsUnitTest
// ============================================================

void TransformerKernelsUnitTest()
{
	printf("============================================================\n");
	printf("Transformer Kernels Test Suite\n");
	printf("============================================================\n");

	unsigned int seed = 42u;

	// ----------------------------------------------------------
	// Group E: Normalization Forward/Backward
	// ----------------------------------------------------------
	printf("\n--- Group E: Normalization Forward/Backward ---\n");

	// E1: LayerNormForwardZeroMeanUnitVar
	{
		const unsigned int D = 16;
		float x[16], y[16];
		fill_random(x, D, seed);
		std::vector<float> gamma(D, 1.0f);
		std::vector<float> beta(D, 0.0f);
		layernorm_into(x, D, gamma, beta, 1e-5f, y);

		double sum = 0.0;
		for (unsigned int i = 0; i < D; ++i)
			sum += static_cast<double>(y[i]);
		double mean = sum / static_cast<double>(D);

		double var = 0.0;
		for (unsigned int i = 0; i < D; ++i)
		{
			double d = static_cast<double>(y[i]) - mean;
			var += d * d;
		}
		var /= static_cast<double>(D);

		ASSERT("LayerNorm forward: mean should be ~0", std::fabs(mean) < 1e-5);
		ASSERT("LayerNorm forward: variance should be ~1", std::fabs(var - 1.0) < 1e-4);
		printf("[PASS] E1: LayerNormForwardZeroMeanUnitVar (mean=%.2e, var=%.6f)\n", mean, var);
	}

	// E2: LayerNormBackwardFiniteDiff
	{
		const unsigned int D = 8;
		const unsigned int T = 1;
		float x[8];
		fill_random(x, D, seed);
		std::vector<float> gamma(D, 1.0f);
		std::vector<float> beta(D, 0.0f);
		const float eps = 1e-5f;
		const float delta = 1e-3f;

		// Forward pass at x
		float Y0[8], meanBuf[1], invStdBuf[1];
		layernorm_forward_rows(x, T, D, gamma, beta, eps, Y0, meanBuf, invStdBuf);

		// Perturb x[0] by +delta
		float xp[8];
		std::memcpy(xp, x, D * sizeof(float));
		xp[0] += delta;
		float Y1[8], meanBuf2[1], invStdBuf2[1];
		layernorm_forward_rows(xp, T, D, gamma, beta, eps, Y1, meanBuf2, invStdBuf2);

		// dY = ones
		float dY[8];
		for (unsigned int i = 0; i < D; ++i)
			dY[i] = 1.0f;

		// Numerical dX[0] = sum_j (Y1[j] - Y0[j]) * dY[j] / delta
		double numGrad = 0.0;
		for (unsigned int i = 0; i < D; ++i)
			numGrad += static_cast<double>(Y1[i] - Y0[i]) * static_cast<double>(dY[i]);
		numGrad /= static_cast<double>(delta);

		// Analytic backward
		float dX[8];
		std::memset(dX, 0, D * sizeof(float));
		std::vector<float> gGamma(D, 0.0f);
		std::vector<float> gBeta(D, 0.0f);
		layernorm_backward_rows_accum(x, dY, T, D, gamma, meanBuf, invStdBuf, dX, gGamma, gBeta);

		double err = std::fabs(static_cast<double>(dX[0]) - numGrad);
		ASSERT("LayerNorm backward finite diff dX[0]", err < 5e-2);
		printf("[PASS] E2: LayerNormBackwardFiniteDiff (analytic=%.6f, numerical=%.6f, err=%.2e)\n",
		       dX[0], numGrad, err);
	}

	// E3: LayerNormBackwardGammaAccumulates
	{
		const unsigned int D = 8;
		const unsigned int T = 1;
		float x[8];
		unsigned int seed3 = 123u;
		fill_random(x, D, seed3);
		std::vector<float> gamma(D, 1.0f);
		std::vector<float> beta(D, 0.0f);
		const float eps = 1e-5f;

		float Y[8], meanBuf[1], invStdBuf[1];
		layernorm_forward_rows(x, T, D, gamma, beta, eps, Y, meanBuf, invStdBuf);

		float dY[8];
		for (unsigned int i = 0; i < D; ++i)
			dY[i] = 1.0f;

		// Single call
		float dX1[8];
		std::memset(dX1, 0, D * sizeof(float));
		std::vector<float> gGamma1(D, 0.0f);
		std::vector<float> gBeta1(D, 0.0f);
		layernorm_backward_rows_accum(x, dY, T, D, gamma, meanBuf, invStdBuf, dX1, gGamma1, gBeta1);

		// Double call (accumulates)
		float dX2[8];
		std::memset(dX2, 0, D * sizeof(float));
		std::vector<float> gGamma2(D, 0.0f);
		std::vector<float> gBeta2(D, 0.0f);
		layernorm_backward_rows_accum(x, dY, T, D, gamma, meanBuf, invStdBuf, dX2, gGamma2, gBeta2);
		layernorm_backward_rows_accum(x, dY, T, D, gamma, meanBuf, invStdBuf, dX2, gGamma2, gBeta2);

		bool ok = true;
		for (unsigned int i = 0; i < D; ++i)
		{
			if (std::fabs(gGamma2[i] - 2.0f * gGamma1[i]) > 1e-5f)
				ok = false;
			if (std::fabs(gBeta2[i] - 2.0f * gBeta1[i]) > 1e-5f)
				ok = false;
		}
		ASSERT("LayerNorm backward gGamma/gBeta accumulates (2x)", ok);
		printf("[PASS] E3: LayerNormBackwardGammaAccumulates\n");
	}

	// E4: RMSNormForwardUnitRMS
	{
		const unsigned int D = 16;
		float x[16], y[16];
		fill_random(x, D, seed);
		std::vector<float> gamma(D, 1.0f);
		std::vector<float> beta(D, 0.0f);
		rmsnorm_into(x, D, gamma, beta, 1e-5f, y);

		double sumsq = 0.0;
		for (unsigned int i = 0; i < D; ++i)
			sumsq += static_cast<double>(y[i]) * static_cast<double>(y[i]);
		double rms = std::sqrt(sumsq / static_cast<double>(D));

		ASSERT("RMSNorm forward: RMS should be ~1.0", std::fabs(rms - 1.0) < 1e-4);
		printf("[PASS] E4: RMSNormForwardUnitRMS (rms=%.6f)\n", rms);
	}

	// E5: RMSNormBackwardFiniteDiff
	{
		const unsigned int D = 8;
		const unsigned int T = 1;
		float x[8];
		unsigned int seed5 = 77u;
		fill_random(x, D, seed5);
		std::vector<float> gamma(D, 1.0f);
		std::vector<float> beta(D, 0.0f);
		const float eps = 1e-5f;
		const float delta = 1e-3f;

		// Forward at x
		float Y0[8], invRms0[1];
		rmsnorm_forward_rows(x, T, D, gamma, beta, eps, Y0, invRms0);

		// Perturb x[0]
		float xp[8];
		std::memcpy(xp, x, D * sizeof(float));
		xp[0] += delta;
		float Y1[8], invRms1[1];
		rmsnorm_forward_rows(xp, T, D, gamma, beta, eps, Y1, invRms1);

		// dY = ones
		float dY[8];
		for (unsigned int i = 0; i < D; ++i)
			dY[i] = 1.0f;

		// Numerical gradient
		double numGrad = 0.0;
		for (unsigned int i = 0; i < D; ++i)
			numGrad += static_cast<double>(Y1[i] - Y0[i]) * static_cast<double>(dY[i]);
		numGrad /= static_cast<double>(delta);

		// Analytic backward
		float dX[8];
		std::memset(dX, 0, D * sizeof(float));
		std::vector<float> gGamma(D, 0.0f);
		std::vector<float> gBeta(D, 0.0f);
		rmsnorm_backward_rows_accum(x, dY, T, D, gamma, invRms0, dX, gGamma, gBeta);

		double err = std::fabs(static_cast<double>(dX[0]) - numGrad);
		ASSERT("RMSNorm backward finite diff dX[0]", err < 5e-2);
		printf("[PASS] E5: RMSNormBackwardFiniteDiff (analytic=%.6f, numerical=%.6f, err=%.2e)\n",
		       dX[0], numGrad, err);
	}

	// E6: NormVecMatchesRowsVariant
	{
		const unsigned int D = 16;
		float x[16];
		unsigned int seed6 = 99u;
		fill_random(x, D, seed6);
		std::vector<float> gamma(D, 1.0f);
		std::vector<float> beta(D, 0.0f);
		const float eps = 1e-5f;

		// layernorm_vec (single vector)
		std::vector<float> yVec;
		layernorm_vec(x, D, gamma, beta, eps, yVec);

		// layernorm_forward_rows with T=1
		float yRows[16], meanBuf[1], invStdBuf[1];
		layernorm_forward_rows(x, 1, D, gamma, beta, eps, yRows, meanBuf, invStdBuf);

		bool ok = true;
		for (unsigned int i = 0; i < D; ++i)
		{
			if (std::fabs(yVec[i] - yRows[i]) > 1e-6f)
				ok = false;
		}
		ASSERT("layernorm_vec matches layernorm_forward_rows(T=1)", ok);
		printf("[PASS] E6: NormVecMatchesRowsVariant\n");
	}

	// ----------------------------------------------------------
	// Group F: RoPE
	// ----------------------------------------------------------
	printf("\n--- Group F: RoPE ---\n");

	// F1: RoPERoundTrip
	{
		const unsigned int dHead = 8;
		const unsigned int ropeDim = 8;
		const unsigned int T = 1;
		float buf[8], orig[8];
		unsigned int seedF1 = 55u;
		fill_random(buf, dHead, seedF1);
		std::memcpy(orig, buf, dHead * sizeof(float));

		std::vector<double> invFreq(ropeDim / 2);
		for (unsigned int i = 0; i < ropeDim / 2; ++i)
			invFreq[i] = pow(10000.0, -(2.0 * static_cast<double>(i)) / static_cast<double>(ropeDim));

		// Forward
		rope_apply_inplace(buf, T, dHead, ropeDim, invFreq, false);
		// Inverse
		rope_apply_inplace(buf, T, dHead, ropeDim, invFreq, true);

		bool ok = true;
		for (unsigned int i = 0; i < dHead; ++i)
		{
			if (std::fabs(buf[i] - orig[i]) > 1e-5f)
				ok = false;
		}
		ASSERT("RoPE round-trip recovers original", ok);
		printf("[PASS] F1: RoPERoundTrip\n");
	}

	// F2: RoPEVecMatchesInplace
	{
		const unsigned int dHead = 8;
		const unsigned int ropeDim = 8;
		float bufVec[8], bufInplace[8];
		unsigned int seedF2 = 66u;
		fill_random(bufVec, dHead, seedF2);
		std::memcpy(bufInplace, bufVec, dHead * sizeof(float));

		std::vector<double> invFreq(ropeDim / 2);
		for (unsigned int i = 0; i < ropeDim / 2; ++i)
			invFreq[i] = pow(10000.0, -(2.0 * static_cast<double>(i)) / static_cast<double>(ropeDim));

		// rope_apply_vec at pos=0
		rope_apply_vec(bufVec, dHead, ropeDim, invFreq, 0);
		// rope_apply_inplace with T=1 (tpos=0)
		rope_apply_inplace(bufInplace, 1, dHead, ropeDim, invFreq, false);

		bool ok = true;
		for (unsigned int i = 0; i < dHead; ++i)
		{
			if (std::fabs(bufVec[i] - bufInplace[i]) > 1e-6f)
				ok = false;
		}
		ASSERT("rope_apply_vec(pos=0) matches rope_apply_inplace(T=1)", ok);
		printf("[PASS] F2: RoPEVecMatchesInplace\n");
	}

	// F3: RoPEPositionZero
	{
		const unsigned int dHead = 8;
		const unsigned int ropeDim = 8;
		float buf[8], orig[8];
		unsigned int seedF3 = 77u;
		fill_random(buf, dHead, seedF3);
		std::memcpy(orig, buf, dHead * sizeof(float));

		std::vector<double> invFreq(ropeDim / 2);
		for (unsigned int i = 0; i < ropeDim / 2; ++i)
			invFreq[i] = pow(10000.0, -(2.0 * static_cast<double>(i)) / static_cast<double>(ropeDim));

		rope_apply_vec(buf, dHead, ropeDim, invFreq, 0);

		bool ok = true;
		for (unsigned int i = 0; i < dHead; ++i)
		{
			if (std::fabs(buf[i] - orig[i]) > 1e-6f)
				ok = false;
		}
		ASSERT("RoPE at pos=0 is identity", ok);
		printf("[PASS] F3: RoPEPositionZero\n");
	}

	// F4: RoPEDifferentPositions
	{
		const unsigned int dHead = 8;
		const unsigned int ropeDim = 8;
		float buf0[8], buf5[8];
		unsigned int seedF4 = 88u;
		fill_random(buf0, dHead, seedF4);
		std::memcpy(buf5, buf0, dHead * sizeof(float));

		std::vector<double> invFreq(ropeDim / 2);
		for (unsigned int i = 0; i < ropeDim / 2; ++i)
			invFreq[i] = pow(10000.0, -(2.0 * static_cast<double>(i)) / static_cast<double>(ropeDim));

		rope_apply_vec(buf0, dHead, ropeDim, invFreq, 0);
		rope_apply_vec(buf5, dHead, ropeDim, invFreq, 5);

		bool differ = false;
		for (unsigned int i = 0; i < dHead; ++i)
		{
			if (std::fabs(buf0[i] - buf5[i]) > 1e-6f)
				differ = true;
		}
		ASSERT("RoPE pos=0 vs pos=5 should differ", differ);
		printf("[PASS] F4: RoPEDifferentPositions\n");
	}

	// F5: RoPEStridedMatchesNonStrided
	{
		const unsigned int dHead = 8;
		const unsigned int ropeDim = 8;
		const unsigned int T = 4;
		float bufNonStrided[32], bufStrided[32];
		unsigned int seedF5 = 111u;
		fill_random(bufNonStrided, T * dHead, seedF5);
		std::memcpy(bufStrided, bufNonStrided, T * dHead * sizeof(float));

		std::vector<double> invFreq(ropeDim / 2);
		for (unsigned int i = 0; i < ropeDim / 2; ++i)
			invFreq[i] = pow(10000.0, -(2.0 * static_cast<double>(i)) / static_cast<double>(ropeDim));

		// Non-strided: contiguous [T, dHead]
		rope_apply_inplace(bufNonStrided, T, dHead, ropeDim, invFreq, false);
		// Strided with stride=dHead (equivalent to contiguous)
		rope_apply_inplace_strided(bufStrided, T, dHead, dHead, ropeDim, invFreq, false);

		bool ok = true;
		for (unsigned int i = 0; i < T * dHead; ++i)
		{
			if (std::fabs(bufNonStrided[i] - bufStrided[i]) > 1e-6f)
				ok = false;
		}
		ASSERT("RoPE strided(stride=dHead) matches non-strided", ok);
		printf("[PASS] F5: RoPEStridedMatchesNonStrided\n");
	}

	// ----------------------------------------------------------
	// Group G: FP16/BF16 Round-Trip
	// ----------------------------------------------------------
	printf("\n--- Group G: FP16/BF16 Round-Trip ---\n");

	// G1: FP16RoundTripExact
	{
		const float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 65504.0f};
		const int nVals = 6;
		bool ok = true;
		for (int i = 0; i < nVals; ++i)
		{
			uint16_t h = float_to_half_rn(vals[i]);
			float rt = half_to_float(h);
			if (rt != vals[i])
				ok = false;
		}
		ASSERT("FP16 round-trip exact for representable values", ok);
		printf("[PASS] G1: FP16RoundTripExact\n");
	}

	// G2: FP16OverflowToInf
	{
		uint16_t h = float_to_half_rn(65536.0f);
		float rt = half_to_float(h);
		ASSERT("FP16 overflow to Inf", std::isinf(rt));
		printf("[PASS] G2: FP16OverflowToInf\n");
	}

	// G3: FP16UnderflowToZero
	{
		uint16_t h = float_to_half_rn(1e-30f);
		float rt = half_to_float(h);
		ASSERT("FP16 underflow to zero", rt == 0.0f);
		printf("[PASS] G3: FP16UnderflowToZero\n");
	}

	// G4: FP16SpecialNaN
	{
		float nan_val = std::numeric_limits<float>::quiet_NaN();
		uint16_t h = float_to_half_rn(nan_val);
		float rt = half_to_float(h);
		ASSERT("FP16 NaN round-trip", std::isnan(rt));
		printf("[PASS] G4: FP16SpecialNaN\n");
	}

	// G5: FP16SpecialInf
	{
		float inf_val = std::numeric_limits<float>::infinity();
		uint16_t h = float_to_half_rn(inf_val);
		float rt = half_to_float(h);
		ASSERT("FP16 Inf round-trip", std::isinf(rt) && rt > 0.0f);
		printf("[PASS] G5: FP16SpecialInf\n");
	}

	// G6: BF16RoundTripExact
	{
		const float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 100.0f};
		const int nVals = 5;
		bool ok = true;
		for (int i = 0; i < nVals; ++i)
		{
			uint16_t b = float_to_bf16_rn(vals[i]);
			float rt = bf16_to_float(b);
			if (rt != vals[i])
				ok = false;
		}
		ASSERT("BF16 round-trip exact for representable values", ok);
		printf("[PASS] G6: BF16RoundTripExact\n");
	}

	// G7: BF16MantissaTruncation
	{
		float val = 1.001f;
		uint16_t b = float_to_bf16_rn(val);
		float rt = bf16_to_float(b);
		double err = std::fabs(static_cast<double>(rt) - static_cast<double>(val));
		ASSERT("BF16 mantissa truncation: close but not exact", err < 0.01);
		printf("[PASS] G7: BF16MantissaTruncation (rt=%.6f, err=%.2e)\n", rt, err);
	}

	// G8: LowpDispatch
	{
		// dtype 0 -> not LOWP_BF16 -> FP16 path
		// dtype LOWP_BF16 (=2) -> BF16 path
		uint16_t lowp_fp16 = float_to_lowp(1.0f, 0);
		uint16_t direct_fp16 = float_to_half_rn(1.0f);
		ASSERT("float_to_lowp(1.0, 0) matches float_to_half_rn(1.0)", lowp_fp16 == direct_fp16);

		uint16_t lowp_bf16 = float_to_lowp(1.0f, LOWP_BF16);
		uint16_t direct_bf16 = float_to_bf16_rn(1.0f);
		ASSERT("float_to_lowp(1.0, LOWP_BF16) matches float_to_bf16_rn(1.0)", lowp_bf16 == direct_bf16);

		float rt_fp16 = lowp_to_float(lowp_fp16, 0);
		float rt_fp16_direct = half_to_float(direct_fp16);
		ASSERT("lowp_to_float(h, 0) matches half_to_float(h)", rt_fp16 == rt_fp16_direct);

		float rt_bf16 = lowp_to_float(lowp_bf16, LOWP_BF16);
		float rt_bf16_direct = bf16_to_float(direct_bf16);
		ASSERT("lowp_to_float(b, LOWP_BF16) matches bf16_to_float(b)", rt_bf16 == rt_bf16_direct);

		printf("[PASS] G8: LowpDispatch\n");
	}

	// ----------------------------------------------------------
	// Group H: Linear/GEMM
	// ----------------------------------------------------------
	printf("\n--- Group H: Linear/GEMM ---\n");

	// H1: LinearForwardBatchParity
	{
		const unsigned int T = 4;
		const unsigned int inSize = 8;
		const unsigned int outSize = 4;
		std::vector<float> W(outSize * inSize);
		std::vector<float> b(outSize);
		float X[32]; // T * inSize
		fill_random(&W[0], outSize * inSize, seed);
		fill_random(&b[0], outSize, seed);
		fill_random(X, T * inSize, seed);

		// Batch forward
		float Y_batch[16]; // T * outSize
		linear_forward(X, T, inSize, W, b, outSize, Y_batch);

		// Per-row forward
		float Y_row[16];
		for (unsigned int t = 0; t < T; ++t)
		{
			const float* xt = X + t * inSize;
			float* yt = Y_row + t * outSize;
			linear_into(xt, inSize, W, b, outSize, yt);
		}

		bool ok = true;
		float maxDiff = 0.0f;
		for (unsigned int i = 0; i < T * outSize; ++i)
		{
			float diff = std::fabs(Y_batch[i] - Y_row[i]);
			if (diff > maxDiff)
				maxDiff = diff;
			if (diff > 1e-5f)
				ok = false;
		}
		ASSERT("linear_forward matches per-row linear_into", ok);
		printf("[PASS] H1: LinearForwardBatchParity (maxDiff=%.2e)\n", maxDiff);
	}

	// H2: LinearForwardLowpMatchesFP32
	{
		const unsigned int T = 4;
		const unsigned int inSize = 8;
		const unsigned int outSize = 4;
		std::vector<float> W(outSize * inSize);
		std::vector<float> b(outSize);
		float X[32];
		unsigned int seedH2 = 200u;
		fill_random(&W[0], outSize * inSize, seedH2);
		fill_random(&b[0], outSize, seedH2);
		fill_random(X, T * inSize, seedH2);

		// FP32 reference
		float Y_fp32[16];
		linear_forward(X, T, inSize, W, b, outSize, Y_fp32);

		// Convert W to FP16
		std::vector<uint16_t> W_half(outSize * inSize);
		for (unsigned int i = 0; i < outSize * inSize; ++i)
			W_half[i] = float_to_half_rn(W[i]);

		// Low-precision forward (dtype 0 = FP16)
		float Y_lowp[16];
		linear_forward_lowp(X, T, inSize, &W_half[0], 0, b, outSize, Y_lowp);

		float maxDiff = 0.0f;
		for (unsigned int i = 0; i < T * outSize; ++i)
		{
			float diff = std::fabs(Y_fp32[i] - Y_lowp[i]);
			if (diff > maxDiff)
				maxDiff = diff;
		}
		ASSERT("linear_forward_lowp(FP16) close to FP32", maxDiff < 1e-2f);
		printf("[PASS] H2: LinearForwardLowpMatchesFP32 (maxDiff=%.2e)\n", maxDiff);
	}

	// H3: GEMMRowMajorABtBias
	{
		// A[2x3], B[4x3], bias[4] -> C[2x4] = A * B^T + bias
		const unsigned int M = 2;
		const unsigned int K = 3;
		const unsigned int N = 4;
		float A[6], B[12], bias[4], C[8];
		unsigned int seedH3 = 300u;
		fill_random(A, M * K, seedH3);
		fill_random(B, N * K, seedH3);
		fill_random(bias, N, seedH3);

		gemm_rowmajor_ABt_bias(A, M, K, B, bias, N, N, C);

		// Naive reference: C[i,j] = bias[j] + sum_k A[i,k]*B[j,k]
		float C_ref[8];
		for (unsigned int i = 0; i < M; ++i)
		{
			for (unsigned int j = 0; j < N; ++j)
			{
				double acc = static_cast<double>(bias[j]);
				for (unsigned int k = 0; k < K; ++k)
					acc += static_cast<double>(A[i * K + k]) * static_cast<double>(B[j * K + k]);
				C_ref[i * N + j] = static_cast<float>(acc);
			}
		}

		float maxDiff = 0.0f;
		for (unsigned int i = 0; i < M * N; ++i)
		{
			float diff = std::fabs(C[i] - C_ref[i]);
			if (diff > maxDiff)
				maxDiff = diff;
		}
		ASSERT("gemm_rowmajor_ABt_bias matches naive reference", maxDiff < 1e-5f);
		printf("[PASS] H3: GEMMRowMajorABtBias (maxDiff=%.2e)\n", maxDiff);
	}

	// H4: TiedEmbeddingLogitsParity
	{
		const unsigned int vocabSize = 8;
		const unsigned int dModel = 4;
		std::vector<float> tokE(vocabSize * dModel);
		std::vector<float> lmBias; // empty bias
		float h[4];
		unsigned int seedH4 = 400u;
		fill_random(&tokE[0], vocabSize * dModel, seedH4);
		fill_random(h, dModel, seedH4);

		float logits[8];
		tied_embedding_logits_into(h, dModel, tokE, lmBias, vocabSize, logits);

		// Manual reference: logits[v] = dot(h, tokE[v, :])
		bool ok = true;
		for (unsigned int v = 0; v < vocabSize; ++v)
		{
			double acc = 0.0;
			for (unsigned int i = 0; i < dModel; ++i)
				acc += static_cast<double>(h[i]) * static_cast<double>(tokE[v * dModel + i]);
			float ref = static_cast<float>(acc);
			if (std::fabs(logits[v] - ref) > 1e-4f)
				ok = false;
		}
		ASSERT("tied_embedding_logits_into matches manual dot product", ok);
		printf("[PASS] H4: TiedEmbeddingLogitsParity\n");
	}

	// ----------------------------------------------------------
	// Group I: Positional Encoding
	// ----------------------------------------------------------
	printf("\n--- Group I: Positional Encoding ---\n");

	// I1: SinusoidalPos0
	{
		const unsigned int dModel = 4;
		float h[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		add_sinusoidal_positional_encoding_inplace(h, 0, dModel);

		// PE[0, 0] = sin(0) = 0
		// PE[0, 1] = cos(0) = 1
		// PE[0, 2] = sin(0) = 0
		// PE[0, 3] = cos(0) = 1
		ASSERT("Sinusoidal pos=0: h[0]==0", std::fabs(h[0] - 0.0f) < 1e-6f);
		ASSERT("Sinusoidal pos=0: h[1]==1", std::fabs(h[1] - 1.0f) < 1e-6f);
		ASSERT("Sinusoidal pos=0: h[2]==0", std::fabs(h[2] - 0.0f) < 1e-6f);
		ASSERT("Sinusoidal pos=0: h[3]==1", std::fabs(h[3] - 1.0f) < 1e-6f);
		printf("[PASS] I1: SinusoidalPos0 (h=[%.4f, %.4f, %.4f, %.4f])\n", h[0], h[1], h[2], h[3]);
	}

	// I2: SinusoidalSeqMatchesSingle
	{
		const unsigned int dModel = 8;
		const unsigned int T = 4;

		// Seq variant
		float bufSeq[32]; // T * dModel
		std::memset(bufSeq, 0, sizeof(bufSeq));
		add_sinusoidal_positional_encoding_seq_inplace(bufSeq, T, dModel);

		// Single-position variant for each t
		float bufSingle[32];
		std::memset(bufSingle, 0, sizeof(bufSingle));
		for (unsigned int t = 0; t < T; ++t)
			add_sinusoidal_positional_encoding_inplace(bufSingle + t * dModel, t, dModel);

		bool ok = true;
		for (unsigned int i = 0; i < T * dModel; ++i)
		{
			if (std::fabs(bufSeq[i] - bufSingle[i]) > 1e-6f)
				ok = false;
		}
		ASSERT("Sinusoidal seq matches single-position calls", ok);
		printf("[PASS] I2: SinusoidalSeqMatchesSingle\n");
	}

	// I3: SinusoidalCachedMatchesUncached
	{
		const unsigned int dModel = 8;
		const unsigned int pos = 5;

		// Uncached
		float h1[8];
		std::memset(h1, 0, sizeof(h1));
		add_sinusoidal_positional_encoding_inplace(h1, pos, dModel);

		// Build cache
		std::vector<double> invDenomPair;
		build_sinusoidal_inv_denom_pair(dModel, invDenomPair);

		// Cached
		float h2[8];
		std::memset(h2, 0, sizeof(h2));
		add_sinusoidal_positional_encoding_inplace(h2, pos, dModel, invDenomPair);

		bool ok = true;
		for (unsigned int i = 0; i < dModel; ++i)
		{
			if (std::fabs(h1[i] - h2[i]) > 1e-6f)
				ok = false;
		}
		ASSERT("Sinusoidal cached matches uncached", ok);
		printf("[PASS] I3: SinusoidalCachedMatchesUncached\n");
	}

	// I4: InvDenomPairMonotonic
	{
		const unsigned int dModel = 64;
		std::vector<double> invDenomPair;
		build_sinusoidal_inv_denom_pair(dModel, invDenomPair);

		const unsigned int nPairs = (dModel + 1u) / 2u;
		ASSERT("invDenomPair[0] == 1.0", std::fabs(invDenomPair[0] - 1.0) < 1e-12);

		bool monotonic = true;
		for (unsigned int i = 1; i < nPairs; ++i)
		{
			if (invDenomPair[i] > invDenomPair[i - 1])
				monotonic = false;
		}
		ASSERT("invDenomPair is monotonically non-increasing", monotonic);
		printf("[PASS] I4: InvDenomPairMonotonic (nPairs=%u, first=%.6f, last=%.6e)\n",
		       nPairs, invDenomPair[0], invDenomPair[nPairs - 1]);
	}

	printf("\n============================================================\n");
	printf("All Transformer Kernels Tests Passed\n");
	printf("============================================================\n");
}
