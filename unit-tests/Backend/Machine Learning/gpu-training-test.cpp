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
		    batchSize, T,
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

	printf("-----------------------------------\n");
	printf("GPU Training Tests: ALL PASSED\n");
	printf("============================================================\n");

#else
	printf("CUDA not enabled, skipping GPU training tests.\n");
	printf("============================================================\n");
#endif
}
