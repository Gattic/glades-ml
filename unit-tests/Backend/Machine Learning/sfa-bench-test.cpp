// SFA GPU performance benchmark — baseline for Phase 7 optimization.
//
// Measures wall-clock and FLOPs/sec of the L_F matvec kernel at production
// scale (T=16384, d_s=8, r=4, W=128, sinks=8). Establishes the baseline that
// Phase 7 (nsys/perf-guided optimization) must improve on.
//
// Also benchmarks at T=1024 and T=4096 for scaling-curve characterization.

#include "sfa-bench-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/Networks/transformer_sfa_ops.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_sfa.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_buffer.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_device.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef GLADES_HAVE_CUDA
#include <cuda_runtime.h>
#endif

using glades::transformer_sfa_ops::SFAParams;
using glades::transformer_sfa_ops::buildEdgeSet;

namespace {

unsigned int g_state_b = 0xCAFEBABEu;

float urand_b(float a, float b)
{
	g_state_b = g_state_b * 1664525u + 1013904223u;
	const float u = ((g_state_b >> 16) & 0xFFFFu) / 65535.0f;
	return a + u * (b - a);
}

void buildBenchSFAParams(SFAParams& p,
                          int T, int d_s, int d_h, int r, int W, int n_sinks)
{
	p.T = T; p.d_s = d_s; p.d_h = d_h; p.r = r;
	p.W = W; p.n_sinks = n_sinks;
	p.lambda = 1e-2f; p.gamma = 0.0f;

	buildEdgeSet(T, W, n_sinks, p.edge_src, p.edge_tgt);
	const int E = static_cast<int>(p.edge_src.size());

	// Simple orthonormal init (no GS).
	p.U.assign(static_cast<size_t>(T) * d_s * r, 0.0f);
	for (int i = 0; i < T; ++i)
		for (int beta = 0; beta < r; ++beta)
			for (int a = 0; a < d_s; ++a)
				p.U[i * d_s * r + a * r + beta] = (a == beta) ? 1.0f : 0.01f * urand_b(-1.f, 1.f);

	p.Sigma.assign(static_cast<size_t>(E) * r, 0.0f);
	for (size_t k = 0; k < p.Sigma.size(); ++k) p.Sigma[k] = 0.8f + 0.2f * urand_b(0.f, 1.f);

	p.P_q.assign(static_cast<size_t>(d_s) * d_h, 0.0f);
	p.P_v.assign(static_cast<size_t>(d_s) * d_h, 0.0f);
	p.P_o.assign(static_cast<size_t>(d_h) * d_s, 0.0f);
}

#ifdef GLADES_HAVE_CUDA
void benchOneShape(int T, int d_s, int d_h, int r, int W, int n_sinks, int n_iters)
{
	SFAParams p;
	buildBenchSFAParams(p, T, d_s, d_h, r, W, n_sinks);
	const int E   = static_cast<int>(p.edge_src.size());
	const int Tds = T * d_s;
	std::vector<float> s(Tds, 0.5f);

	glades::gpu::GpuBuffer<float> dU, dSigma, ds, dout;
	glades::gpu::GpuBuffer<int>   dE_src, dE_tgt;
	dU.allocate(p.U.size()); dU.upload(&p.U[0], p.U.size());
	dSigma.allocate(p.Sigma.size()); dSigma.upload(&p.Sigma[0], p.Sigma.size());
	dE_src.allocate(p.edge_src.size()); dE_src.upload(&p.edge_src[0], p.edge_src.size());
	dE_tgt.allocate(p.edge_tgt.size()); dE_tgt.upload(&p.edge_tgt[0], p.edge_tgt.size());
	ds.allocate(s.size()); ds.upload(&s[0], s.size());
	dout.allocate(s.size());

	// Warm-up.
	for (int it = 0; it < 3; ++it)
	{
		glades::gpu::sfa_laplacian_matvec_fp32(
		    dU.data(), dSigma.data(), dE_src.data(), dE_tgt.data(),
		    ds.data(), dout.data(), T, E, d_s, r);
	}
	cudaDeviceSynchronize();

	// Timed run with CUDA events ON THE SAME STREAM as kernel launches.
	cudaStream_t s_stream = glades::gpu::computeStream();
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);

	cudaEventRecord(start, s_stream);
	for (int it = 0; it < n_iters; ++it)
	{
		glades::gpu::sfa_laplacian_matvec_fp32(
		    dU.data(), dSigma.data(), dE_src.data(), dE_tgt.data(),
		    ds.data(), dout.data(), T, E, d_s, r);
	}
	cudaEventRecord(stop, s_stream);
	cudaEventSynchronize(stop);
	float ms_total = 0.0f;
	cudaEventElapsedTime(&ms_total, start, stop);
	const float ms_per_call = ms_total / n_iters;

	// Per-edge FLOPs: roughly (4 * d_s * r + d_s) per edge in body.
	const double flops_per_call = static_cast<double>(E) * (4.0 * d_s * r + d_s);
	const double gflops = (flops_per_call / 1e9) / (ms_per_call / 1000.0);

	std::printf("  T=%6d d_s=%d r=%d W=%d  |E|=%9d  L_F matvec %8.3f ms  %8.1f GFLOPs/sec\n",
	            T, d_s, r, W, E, ms_per_call, gflops);

	cudaEventDestroy(start); cudaEventDestroy(stop);

	// ====== CSR variant ======
	std::vector<int> out_off(T+1), in_off(T+1), out_edges(E), in_edges(E);
	glades::gpu::sfa_build_csr_host(&p.edge_src[0], &p.edge_tgt[0], E, T,
	    &out_off[0], &out_edges[0], &in_off[0], &in_edges[0]);

	glades::gpu::GpuBuffer<int> d_out_off, d_in_off, d_out_edges, d_in_edges;
	d_out_off.allocate(out_off.size());   d_out_off.upload(&out_off[0], out_off.size());
	d_in_off.allocate(in_off.size());     d_in_off.upload(&in_off[0], in_off.size());
	d_out_edges.allocate(out_edges.size()); d_out_edges.upload(&out_edges[0], out_edges.size());
	d_in_edges.allocate(in_edges.size());   d_in_edges.upload(&in_edges[0], in_edges.size());

	for (int it = 0; it < 3; ++it)
	{
		glades::gpu::sfa_laplacian_matvec_csr_fp32(
		    dU.data(), dSigma.data(), dE_src.data(), dE_tgt.data(),
		    d_out_off.data(), d_out_edges.data(), d_in_off.data(), d_in_edges.data(),
		    ds.data(), dout.data(), T, E, d_s, r);
	}
	cudaDeviceSynchronize();
	cudaEvent_t cs, ce; cudaEventCreate(&cs); cudaEventCreate(&ce);
	cudaEventRecord(cs, s_stream);
	for (int it = 0; it < n_iters; ++it)
	{
		glades::gpu::sfa_laplacian_matvec_csr_fp32(
		    dU.data(), dSigma.data(), dE_src.data(), dE_tgt.data(),
		    d_out_off.data(), d_out_edges.data(), d_in_off.data(), d_in_edges.data(),
		    ds.data(), dout.data(), T, E, d_s, r);
	}
	cudaEventRecord(ce, s_stream);
	cudaEventSynchronize(ce);
	float ms_csr_total = 0.0f;
	cudaEventElapsedTime(&ms_csr_total, cs, ce);
	const float ms_csr = ms_csr_total / n_iters;
	const double gflops_csr = (flops_per_call / 1e9) / (ms_csr / 1000.0);
	const float speedup = ms_per_call / ms_csr;
	std::printf("    -> CSR variant:                    %8.3f ms  %8.1f GFLOPs/sec  (%.2fx vs atomic)\n",
	            ms_csr, gflops_csr, speedup);
	cudaEventDestroy(cs); cudaEventDestroy(ce);
}
#endif

}  // anonymous namespace

void SFABenchUnitTest()
{
	std::printf("=== SFA GPU benchmark (baseline for Phase 7) ===\n");

#ifdef GLADES_HAVE_CUDA
	ASSERT("init CUDA device", glades::gpu::initDevice());

	// Scaling curve.
	std::printf("L_F matvec scaling (FP32, atomicAdd reductions, RTX 4080 SUPER):\n");
	benchOneShape(/*T*/ 1024, /*d_s*/ 8, /*d_h*/ 64, /*r*/ 4, /*W*/ 128, /*sinks*/ 8, /*n_iters*/ 100);
	benchOneShape(/*T*/ 4096, /*d_s*/ 8, /*d_h*/ 64, /*r*/ 4, /*W*/ 128, /*sinks*/ 8, /*n_iters*/ 50);
	benchOneShape(/*T*/16384, /*d_s*/ 8, /*d_h*/ 64, /*r*/ 4, /*W*/ 128, /*sinks*/ 8, /*n_iters*/ 20);

	// Wider stalk at T=16384.
	std::printf("L_F matvec wider d_s at T=16384:\n");
	benchOneShape(/*T*/16384, /*d_s*/16, /*d_h*/ 64, /*r*/ 4, /*W*/ 128, /*sinks*/ 8, /*n_iters*/ 10);
	benchOneShape(/*T*/16384, /*d_s*/32, /*d_h*/ 64, /*r*/ 4, /*W*/ 128, /*sinks*/ 8, /*n_iters*/ 10);

	std::printf("=== SFA benchmark done ===\n");
#else
	std::printf("  CUDA not enabled — skipping SFA benchmark.\n");
#endif
}
