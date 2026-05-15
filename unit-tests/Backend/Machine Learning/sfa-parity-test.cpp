// SFA CPU vs GPU parity test.
//
// Compares CPU reference (transformer_sfa_ops.h) against GPU primitives
// (gpu_sfa.h) for the three core SFA primitives:
//   1. L_F matvec
//   2. Source assembly (b_i = U_i U_i^T P_q q_i + gamma * P_v v_i)
//   3. Readout (y_i = P_o^T s_i)
//
// Tolerance: 5e-3 relative for FP32 with edge-parallel atomicAdd reductions.
// The atomicAdd ordering is non-deterministic, so per-call results vary at
// the FP32 epsilon level; we require ~5 mantissa-bit agreement.

#include "sfa-parity-test.h"
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

using glades::transformer_sfa_ops::SFAParams;
using glades::transformer_sfa_ops::buildEdgeSet;
using glades::transformer_sfa_ops::laplacianMatvec;
using glades::transformer_sfa_ops::sourceAssembly;
using glades::transformer_sfa_ops::readout;

namespace {

unsigned int g_state = 0x12345678u;

float urand(float a, float b)
{
	g_state = g_state * 1664525u + 1013904223u;
	const float u = ((g_state >> 16) & 0xFFFFu) / 65535.0f;
	return a + u * (b - a);
}

// Build a small SFA parameter set with random U and unit-ish Sigma (matches
// the synthetic builder in transformer_sfa_smoke_test.cpp).
void buildSyntheticSFAParams(SFAParams& p,
                              int T, int d_s, int d_h, int r, int W, int n_sinks)
{
	p.T = T; p.d_s = d_s; p.d_h = d_h; p.r = r;
	p.W = W; p.n_sinks = n_sinks;
	p.lambda = 1e-2f; p.gamma = 0.0f;

	buildEdgeSet(T, W, n_sinks, p.edge_src, p.edge_tgt);
	const int E = static_cast<int>(p.edge_src.size());

	// Gram-Schmidt orthonormal stalk frames per token.
	p.U.assign(static_cast<size_t>(T) * d_s * r, 0.0f);
	for (int i = 0; i < T; ++i)
	{
		for (int beta = 0; beta < r; ++beta)
		{
			for (int a = 0; a < d_s; ++a)
				p.U[i * d_s * r + a * r + beta] = urand(-1.0f, 1.0f);
			for (int prev = 0; prev < beta; ++prev)
			{
				float dot = 0.0f;
				for (int a = 0; a < d_s; ++a)
					dot += p.U[i * d_s * r + a * r + beta] *
					       p.U[i * d_s * r + a * r + prev];
				for (int a = 0; a < d_s; ++a)
					p.U[i * d_s * r + a * r + beta] -=
					    dot * p.U[i * d_s * r + a * r + prev];
			}
			float norm = 0.0f;
			for (int a = 0; a < d_s; ++a)
			{
				float v = p.U[i * d_s * r + a * r + beta];
				norm += v * v;
			}
			norm = std::sqrt(std::max(norm, 1e-12f));
			for (int a = 0; a < d_s; ++a)
				p.U[i * d_s * r + a * r + beta] /= norm;
		}
	}

	p.Sigma.assign(static_cast<size_t>(E) * r, 0.0f);
	for (int e = 0; e < E; ++e)
		for (int beta = 0; beta < r; ++beta)
			p.Sigma[e * r + beta] = urand(0.5f, 1.0f);

	p.P_q.assign(static_cast<size_t>(d_s) * d_h, 0.0f);
	p.P_v.assign(static_cast<size_t>(d_s) * d_h, 0.0f);
	p.P_o.assign(static_cast<size_t>(d_h) * d_s, 0.0f);
	for (int a = 0; a < d_s; ++a)
		for (int h = 0; h < d_h; ++h)
		{
			if (a == h) p.P_q[a * d_h + h] = 1.0f;
			if (a == h) p.P_v[a * d_h + h] = 1.0f;
			if (h == a) p.P_o[h * d_s + a] = 1.0f;
		}
}

double rel_err(double result, double ref)
{
	if (ref == 0.0 && result == 0.0) return 0.0;
	double denom = std::fabs(ref);
	if (denom < 1e-6) denom = 1e-6;
	return std::fabs(result - ref) / denom;
}

double max_rel_err(const std::vector<float>& gpu, const std::vector<float>& cpu)
{
	double max_re = 0.0;
	for (size_t k = 0; k < cpu.size(); ++k)
		max_re = std::max(max_re, rel_err(gpu[k], cpu[k]));
	return max_re;
}

}  // anonymous namespace

void SFAParityUnitTest()
{
	std::printf("=== SFA CPU vs GPU parity test ===\n");

#ifdef GLADES_HAVE_CUDA
	ASSERT("init CUDA device", glades::gpu::initDevice());

	const int T = 64, d_s = 8, d_h = 8, r = 4, W = 16, n_sinks = 2;
	SFAParams p;
	buildSyntheticSFAParams(p, T, d_s, d_h, r, W, n_sinks);

	const int E = static_cast<int>(p.edge_src.size());
	const int Tds = T * d_s;
	const int Tdh = T * d_h;
	std::printf("  config: T=%d d_s=%d d_h=%d r=%d W=%d sinks=%d |E|=%d\n",
	            T, d_s, d_h, r, W, n_sinks, E);

	// ====== Test 1: L_F matvec parity ======
	std::vector<float> s(Tds), out_cpu(Tds), out_gpu(Tds);
	for (int k = 0; k < Tds; ++k) s[k] = urand(-1.0f, 1.0f);
	laplacianMatvec(p, &s[0], &out_cpu[0]);

	glades::gpu::GpuBuffer<float> dU, dSigma, ds, dout;
	glades::gpu::GpuBuffer<int>   dE_src, dE_tgt;
	dU.allocate(p.U.size());        dU.upload(&p.U[0], p.U.size());
	dSigma.allocate(p.Sigma.size()); dSigma.upload(&p.Sigma[0], p.Sigma.size());
	dE_src.allocate(p.edge_src.size()); dE_src.upload(&p.edge_src[0], p.edge_src.size());
	dE_tgt.allocate(p.edge_tgt.size()); dE_tgt.upload(&p.edge_tgt[0], p.edge_tgt.size());
	ds.allocate(s.size());           ds.upload(&s[0], s.size());
	dout.allocate(out_cpu.size());

	ASSERT("sfa_laplacian_matvec_fp32 launch", glades::gpu::sfa_laplacian_matvec_fp32(
	    dU.data(), dSigma.data(), dE_src.data(), dE_tgt.data(),
	    ds.data(), dout.data(), T, E, d_s, r));
	dout.download(&out_gpu[0], out_cpu.size());

	double maxre_lap = max_rel_err(out_gpu, out_cpu);
	std::printf("  L_F matvec (atomic) max rel err = %.6e\n", maxre_lap);
	ASSERT("L_F matvec atomic parity (5e-3 tol)", maxre_lap < 5e-3);

	// CSR variant — deterministic, no atomics.
	std::vector<int> out_off(T+1), in_off(T+1), out_edges_l(E), in_edges_l(E);
	glades::gpu::sfa_build_csr_host(&p.edge_src[0], &p.edge_tgt[0], E, T,
	    &out_off[0], &out_edges_l[0], &in_off[0], &in_edges_l[0]);

	glades::gpu::GpuBuffer<int> d_out_off, d_in_off, d_out_e, d_in_e;
	d_out_off.allocate(out_off.size()); d_out_off.upload(&out_off[0], out_off.size());
	d_in_off.allocate(in_off.size()); d_in_off.upload(&in_off[0], in_off.size());
	d_out_e.allocate(out_edges_l.size()); d_out_e.upload(&out_edges_l[0], out_edges_l.size());
	d_in_e.allocate(in_edges_l.size()); d_in_e.upload(&in_edges_l[0], in_edges_l.size());

	std::vector<float> out_csr(Tds);
	ASSERT("sfa_laplacian_matvec_csr_fp32 launch", glades::gpu::sfa_laplacian_matvec_csr_fp32(
	    dU.data(), dSigma.data(), dE_src.data(), dE_tgt.data(),
	    d_out_off.data(), d_out_e.data(), d_in_off.data(), d_in_e.data(),
	    ds.data(), dout.data(), T, E, d_s, r));
	dout.download(&out_csr[0], out_cpu.size());
	double maxre_csr = max_rel_err(out_csr, out_cpu);
	std::printf("  L_F matvec (CSR)    max rel err = %.6e\n", maxre_csr);
	ASSERT("L_F matvec CSR parity (5e-3 tol)", maxre_csr < 5e-3);

	// ====== Test 2: source assembly parity ======
	std::vector<float> q(Tdh), v(Tdh), b_cpu(Tds), b_gpu(Tds);
	for (int k = 0; k < Tdh; ++k) { q[k] = urand(-1.0f, 1.0f); v[k] = urand(-1.0f, 1.0f); }
	p.gamma = 0.5f;
	sourceAssembly(p, &q[0], &v[0], &b_cpu[0]);

	glades::gpu::GpuBuffer<float> dPq, dPv, dq, dv, db;
	dPq.allocate(p.P_q.size()); dPq.upload(&p.P_q[0], p.P_q.size());
	dPv.allocate(p.P_v.size()); dPv.upload(&p.P_v[0], p.P_v.size());
	dq.allocate(q.size());      dq.upload(&q[0], q.size());
	dv.allocate(v.size());      dv.upload(&v[0], v.size());
	db.allocate(b_cpu.size());

	ASSERT("sfa_source_assembly_fp32 launch", glades::gpu::sfa_source_assembly_fp32(
	    dU.data(), dPq.data(), dPv.data(), dq.data(), dv.data(), p.gamma,
	    db.data(), T, d_s, d_h, r));
	db.download(&b_gpu[0], b_cpu.size());

	double maxre_src = max_rel_err(b_gpu, b_cpu);
	std::printf("  source asm  max rel err = %.6e\n", maxre_src);
	ASSERT("source assembly parity (5e-3 tol)", maxre_src < 5e-3);

	// ====== Test 3: readout parity ======
	std::vector<float> y_cpu(Tdh), y_gpu(Tdh);
	readout(p, &b_cpu[0], &y_cpu[0]);

	glades::gpu::GpuBuffer<float> dPo, dyout;
	dPo.allocate(p.P_o.size()); dPo.upload(&p.P_o[0], p.P_o.size());
	dyout.allocate(y_cpu.size());

	// Note: GPU readout reads from db (source-assembly output), CPU readout
	// reads from b_cpu. They should match if both source-assemblies matched.
	ASSERT("sfa_readout_fp32 launch", glades::gpu::sfa_readout_fp32(
	    dPo.data(), db.data(), dyout.data(), T, d_s, d_h));
	dyout.download(&y_gpu[0], y_cpu.size());

	double maxre_ro = max_rel_err(y_gpu, y_cpu);
	std::printf("  readout     max rel err = %.6e\n", maxre_ro);
	ASSERT("readout parity (5e-3 tol)", maxre_ro < 5e-3);

	std::printf("=== SFA parity: ALL 3 PRIMITIVES PASS ===\n");
#else
	std::printf("  CUDA not enabled — skipping SFA parity test.\n");
#endif
}
