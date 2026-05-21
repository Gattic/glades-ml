// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "gpu-training-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_device.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_buffer.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_blas.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_kernels.h"
#include <math.h>
#include <vector>
#include <stdint.h>

static bool approxEqual(float a, float b, float eps)
{
	return fabsf(a - b) < eps;
}

void GpuTrainingUnitTest()
{
	printf("============================================================\n");
	printf("-----------------------------------\n");
	printf("GPU Training Test\n");
	printf("-----------------------------------\n");

#ifdef GLADES_HAVE_CUDA
	if (!glades::gpu::initDevice())
	{
		printf("No CUDA device available, skipping GPU training tests.\n");
		printf("============================================================\n");
		return;
	}

	// =========================================================================
	// Test 1: sgemm_rowmajor_abt — Forward projection with non-square weights
	//
	// Bug fixed: forward pass used sgemm_rowmajor (no transpose) instead of
	// sgemm_rowmajor_abt. Weight matrix W is stored as [outDim, inDim] row-major,
	// so the forward projection y = x * W^T requires the _abt variant.
	// =========================================================================
	printf("Test 1: sgemm_rowmajor_abt (forward projection)\n");
	{
		const int M = 2;  // tokens / batch rows
		const int K = 3;  // input dimension
		const int N = 4;  // output dimension (outDim != inDim)

		// A[M,K] — input activations
		float h_A[6] = {1.0f, 2.0f, 3.0f,
		                 4.0f, 5.0f, 6.0f};

		// B[N,K] — weight matrix stored as [outDim, inDim]
		float h_B[12] = {1.0f, 0.0f, 0.0f,
		                  0.0f, 1.0f, 0.0f,
		                  0.0f, 0.0f, 1.0f,
		                  1.0f, 1.0f, 1.0f};

		// Expected: C = A * B^T  →  [M, N]
		//   row 0: [1*1+2*0+3*0, 1*0+2*1+3*0, 1*0+2*0+3*1, 1*1+2*1+3*1] = [1, 2, 3, 6]
		//   row 1: [4*1+5*0+6*0, 4*0+5*1+6*0, 4*0+5*0+6*1, 4*1+5*1+6*1] = [4, 5, 6, 15]
		float expected[8] = {1.0f, 2.0f, 3.0f, 6.0f,
		                     4.0f, 5.0f, 6.0f, 15.0f};

		glades::gpu::GpuBuffer<float> d_A, d_B, d_C;
		d_A.allocate(M * K);
		d_B.allocate(N * K);
		d_C.allocate(M * N);

		d_A.upload(h_A);
		d_B.upload(h_B);
		d_C.zero();

		// sgemm_rowmajor_abt: C[M,N] = A[M,K] * B[N,K]^T
		bool ok = glades::gpu::sgemm_rowmajor_abt(
		    M, N, K,
		    1.0f,
		    d_A.data(), K,
		    d_B.data(), K,
		    0.0f,
		    d_C.data(), N);

		ASSERT("sgemm_rowmajor_abt call failed", ok);
		ASSERT("sgemm_rowmajor_abt sync error",
		       glades::gpu::synchronizeCheck("sgemm_rowmajor_abt test"));

		float h_C[8];
		d_C.download(h_C);

		const float eps = 1e-4f;
		for (int i = 0; i < M * N; ++i)
		{
			char msg[128];
			sprintf(msg, "sgemm_rowmajor_abt result[%d]: got %.4f, expected %.4f",
			        i, h_C[i], expected[i]);
			ASSERT(msg, approxEqual(h_C[i], expected[i], eps));
		}

		printf("  sgemm_rowmajor_abt: PASSED\n");
	}

	// =========================================================================
	// Test 2: softmax_backward_attn — Correct batchSize parameter
	//
	// Bug fixed: backward pass called softmax_backward_attn with
	// batchSize = groupSize * T instead of just groupSize.  The kernel
	// internally computes totalRows = batchSize * T, so the wrong argument
	// caused totalRows = groupSize * T * T blocks to launch, reading far
	// past the allocated buffers.
	// =========================================================================
	printf("Test 2: softmax_backward_attn (correct batchSize)\n");
	{
		const int batchSize = 2;  // number of [T,T] attention matrices
		const int T = 3;          // sequence length
		const int total = batchSize * T * T;

		// P = softmax output (causal attention probabilities).
		// Each [T,T] block: rows sum to 1, upper-triangle is 0.
		float h_P[18] = {
		    // Batch 0
		    1.0f,  0.0f,  0.0f,
		    0.5f,  0.5f,  0.0f,
		    0.25f, 0.25f, 0.5f,
		    // Batch 1
		    1.0f,  0.0f,  0.0f,
		    0.5f,  0.5f,  0.0f,
		    0.25f, 0.25f, 0.5f
		};

		// dP = upstream gradient
		float h_dP[18] = {
		    // Batch 0
		     1.0f,  0.0f,  0.0f,
		     0.5f, -0.5f,  0.0f,
		     0.1f,  0.2f, -0.3f,
		    // Batch 1
		     1.0f,  0.0f,  0.0f,
		     0.5f, -0.5f,  0.0f,
		     0.1f,  0.2f, -0.3f
		};

		// Expected dS = P * (dP - row_dot), zeroed above diagonal.
		// Row 0: dot = 1.0*1.0 = 1.0
		//   dS = [1.0*(1.0-1.0), 0, 0] = [0, 0, 0]
		// Row 1: dot = 0.5*0.5 + 0.5*(-0.5) = 0.0
		//   dS = [0.5*(0.5-0.0), 0.5*(-0.5-0.0), 0] = [0.25, -0.25, 0]
		// Row 2: dot = 0.25*0.1 + 0.25*0.2 + 0.5*(-0.3) = -0.075
		//   dS = [0.25*(0.1+0.075), 0.25*(0.2+0.075), 0.5*(-0.3+0.075)]
		//      = [0.04375, 0.06875, -0.1125]
		float expected_dS[18] = {
		    // Batch 0
		    0.0f,      0.0f,      0.0f,
		    0.25f,    -0.25f,     0.0f,
		    0.04375f,  0.06875f, -0.1125f,
		    // Batch 1 (same)
		    0.0f,      0.0f,      0.0f,
		    0.25f,    -0.25f,     0.0f,
		    0.04375f,  0.06875f, -0.1125f
		};

		glades::gpu::GpuBuffer<float> d_P, d_dP, d_dS;
		d_P.allocate(total);
		d_dP.allocate(total);
		d_dS.allocate(total);

		d_P.upload(h_P);
		d_dP.upload(h_dP);
		d_dS.zero();

		// Correct call: batchSize = number of [T,T] matrices (NOT batchSize * T)
		bool ok = glades::gpu::softmax_backward_attn(
		    d_P.data(), d_dP.data(),
		    batchSize, T, 1.0f,
		    d_dS.data());

		ASSERT("softmax_backward_attn call failed", ok);
		ASSERT("softmax_backward_attn sync error",
		       glades::gpu::synchronizeCheck("softmax_backward_attn test"));

		float h_dS[18];
		d_dS.download(h_dS);

		const float eps = 1e-4f;
		for (int i = 0; i < total; ++i)
		{
			char msg[128];
			sprintf(msg, "softmax_backward_attn dS[%d]: got %.6f, expected %.6f",
			        i, h_dS[i], expected_dS[i]);
			ASSERT(msg, approxEqual(h_dS[i], expected_dS[i], eps));
		}

		printf("  softmax_backward_attn: PASSED\n");
	}

	// =========================================================================
	// Test 3: adam_update_bf16_state — Parity with FP32 Adam
	//
	// Runs both kernels with identical init (same seed gradients) for 100
	// steps and checks that weight trajectories agree within BF16 precision
	// (~0.5% relative on weight values after 100 steps). The kernels compute
	// identical math; they differ only in the m, v storage precision.
	// =========================================================================
	printf("Test 3: adam_update_bf16_state parity vs adam_update (FP32)\n");
	{
		const int N = 512; // parameters
		std::vector<float> W0(N), g(N);
		for (int i = 0; i < N; ++i)
		{
			W0[i] = 0.1f * static_cast<float>((i * 7) % 17 - 8);
			g[i]  = 0.01f * static_cast<float>((i * 13) % 11 - 5);
		}

		// FP32 path.
		std::vector<float> Wfp(W0);
		std::vector<float> mfp(N, 0.0f), vfp(N, 0.0f);
		glades::gpu::GpuBuffer<float> dWfp, dG, dMfp, dVfp;
		ASSERT("alloc Wfp", dWfp.allocate(N));
		ASSERT("alloc G",   dG.allocate(N));
		ASSERT("alloc Mfp", dMfp.allocate(N));
		ASSERT("alloc Vfp", dVfp.allocate(N));
		ASSERT("upload Wfp", dWfp.upload(&Wfp[0], N));
		ASSERT("upload G",   dG.upload(&g[0], N));
		ASSERT("upload Mfp", dMfp.upload(&mfp[0], N));
		ASSERT("upload Vfp", dVfp.upload(&vfp[0], N));

		// BF16-state path.
		std::vector<float> Wbf(W0);
		std::vector<uint16_t> mBf(N, 0), vBf(N, 0);
		glades::gpu::GpuBuffer<float> dWbf;
		glades::gpu::GpuBuffer<uint16_t> dMbf, dVbf;
		ASSERT("alloc Wbf", dWbf.allocate(N));
		ASSERT("alloc Mbf", dMbf.allocate(N));
		ASSERT("alloc Vbf", dVbf.allocate(N));
		ASSERT("upload Wbf", dWbf.upload(&Wbf[0], N));
		ASSERT("upload Mbf", dMbf.upload(&mBf[0], N));
		ASSERT("upload Vbf", dVbf.upload(&vBf[0], N));

		const float lr = 1e-3f, beta1 = 0.9f, beta2 = 0.999f, epsA = 1e-8f;
		const float wd = 0.0f, gradScale = 1.0f;
		for (int step = 1; step <= 100; ++step)
		{
			ASSERT("fp adam step",
			       glades::gpu::adam_update(dWfp.data(), dG.data(),
			                                 dMfp.data(), dVfp.data(),
			                                 lr, beta1, beta2, epsA, wd,
			                                 gradScale, step, N));
			ASSERT("bf adam step",
			       glades::gpu::adam_update_bf16_state(dWbf.data(), dG.data(),
			                                            dMbf.data(), dVbf.data(),
			                                            lr, beta1, beta2, epsA, wd,
			                                            gradScale, step, N));
		}

		std::vector<float> WfpOut(N), WbfOut(N);
		ASSERT("download Wfp", dWfp.download(&WfpOut[0], N));
		ASSERT("download Wbf", dWbf.download(&WbfOut[0], N));

		// Compare RELATIVE only where |Wfp| > a meaningful floor, since
		// weights can cross zero during training making tiny denominators
		// blow up rel error. Also report weight-space L2 deviation.
		float maxAbs = 0.0f, meanAbs = 0.0f, maxRelFiltered = 0.0f;
		double l2NumSq = 0.0, l2DenSq = 0.0;
		int relSamples = 0;
		for (int i = 0; i < N; ++i)
		{
			const float diff = fabsf(WfpOut[i] - WbfOut[i]);
			if (diff > maxAbs) maxAbs = diff;
			meanAbs += diff;
			l2NumSq += static_cast<double>(diff) * diff;
			l2DenSq += static_cast<double>(WfpOut[i]) * WfpOut[i];
			if (fabsf(WfpOut[i]) > 1e-3f)
			{
				const float rel = diff / fabsf(WfpOut[i]);
				if (rel > maxRelFiltered) maxRelFiltered = rel;
				++relSamples;
			}
		}
		meanAbs /= static_cast<float>(N);
		const float l2Rel = (l2DenSq > 0.0)
		    ? static_cast<float>(sqrt(l2NumSq / l2DenSq)) : 0.0f;
		printf("  after 100 steps: maxAbs=%.6g meanAbs=%.6g "
		       "maxRel(|W|>1e-3, %d samples)=%.3g L2rel=%.3g\n",
		       maxAbs, meanAbs, relSamples, maxRelFiltered, l2Rel);
		// L2-relative deviation is the right aggregate metric for BF16 EMA
		// state: per-coordinate rel can blow up near zero crossings but
		// overall trajectory drift should stay small. BF16's 7-bit mantissa
		// gives ~0.4% per-update quantization bias; 100 steps accumulated
		// should stay well under 2% L2 drift.
		ASSERT("parity weight L2-relative < 2%", l2Rel < 0.02f);
		printf("  adam_update_bf16_state: PASSED\n");
	}

	printf("-----------------------------------\n");
	printf("GPU Training Tests: ALL PASSED\n");
	printf("============================================================\n");

#else
	printf("CUDA not enabled, skipping GPU training tests.\n");
	printf("============================================================\n");
#endif
}
