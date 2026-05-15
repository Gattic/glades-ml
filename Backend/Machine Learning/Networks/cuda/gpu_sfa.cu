// SFA (Sheaf-Focal Attention) GPU primitives — paradigm #250.
//
// See gpu_sfa.h for declarations and research/PARADIGM_SHIFT_250_DESIGN.md
// for math. CPU reference: transformer_sfa_ops.h::laplacianMatvec /
// sourceAssembly / readout. This file implements the CUDA kernels and host
// wrappers used by the SFA forward path.
//
// Iter 21 (2026-05-15): initial GPU port. Edge-parallel kernel with
// atomicAdd reductions. Subsequent iters (22+) will:
//   - Restore bit-exact determinism via CSR-style per-vertex reduction.
//   - Cooperative block-level parallelism (one block per edge, threads
//     in block cooperate on the d_s outputs).
//   - BF16 mixed-precision via cublasGemmEx for the lift/project GEMVs.

#include "gpu_sfa.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

namespace glades {
namespace gpu {

namespace {

#define GLADES_CUDA_CHECK(call)                                               \
	do {                                                                      \
		cudaError_t err_ = (call);                                            \
		if (err_ != cudaSuccess) {                                            \
			fprintf(stderr, "[sfa-cuda] %s:%d  %s  -> %s\n",                  \
			        __FILE__, __LINE__, #call, cudaGetErrorString(err_));     \
			return false;                                                     \
		}                                                                     \
	} while (0)

static constexpr int kBlockElem = 256;

// ---------------------------------------------------------------------------
// L_F matvec kernel — one thread per edge.
//
// Per-edge work (~d_s · r ops + 2·d_s atomicAdds):
//   1. tmp_inner[beta] = (sum_a U_i[a,beta] * s_i[a]) * Sigma[e,beta]
//   2. (R s_i)[a]      = sum_beta U_j[a,beta] * tmp_inner[beta]
//   3. delta[a]        = s_j[a] - (R s_i)[a]
//   4. atomicAdd(out[j*d_s + a], delta[a])
//   5. tmp_delta[beta] = (sum_a U_j[a,beta] * delta[a]) * Sigma[e,beta]
//   6. (R^T delta)[a]  = sum_beta U_i[a,beta] * tmp_delta[beta]
//   7. atomicAdd(out[i*d_s + a], -(R^T delta)[a])
//
// Per-thread storage:
//   tmp_inner[MAX_R], tmp_delta[MAX_R] — at most MAX_R floats each.
//   delta[MAX_DS] — up to MAX_DS floats.
//
// Static limits to allow per-thread arrays:
constexpr int kMaxDS = 128;
constexpr int kMaxR  = 16;
// ---------------------------------------------------------------------------
__global__ void sfa_laplacian_kernel(const float* __restrict__ U,
                                      const float* __restrict__ Sigma,
                                      const int*   __restrict__ edge_src,
                                      const int*   __restrict__ edge_tgt,
                                      const float* __restrict__ s,
                                      float* __restrict__ out,
                                      int T, int E, int d_s, int r)
{
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= E) return;

	const int i = edge_src[e];
	const int j = edge_tgt[e];

	const float* Ui      = U + (size_t)i * d_s * r;
	const float* Uj      = U + (size_t)j * d_s * r;
	const float* Sig_e   = Sigma + (size_t)e * r;
	const float* s_i     = s + (size_t)i * d_s;
	const float* s_j     = s + (size_t)j * d_s;

	// Step 1: tmp_inner[beta] = sum_a U_i[a,beta] * s_i[a]; mul by Sigma.
	float tmp_inner[kMaxR];
	for (int beta = 0; beta < r; ++beta)
	{
		float acc = 0.0f;
		for (int a = 0; a < d_s; ++a)
			acc += Ui[a * r + beta] * s_i[a];
		tmp_inner[beta] = acc * Sig_e[beta];
	}

	// Step 2-4: compute delta and write +delta to out[j].
	// Also accumulate delta into local array for Step 5.
	float delta[kMaxDS];
	for (int a = 0; a < d_s; ++a)
	{
		float Rs = 0.0f;
		for (int beta = 0; beta < r; ++beta)
			Rs += Uj[a * r + beta] * tmp_inner[beta];
		delta[a] = s_j[a] - Rs;
		atomicAdd(&out[(size_t)j * d_s + a], delta[a]);
	}

	// Step 5: tmp_delta[beta] = sum_a U_j[a,beta] * delta[a]; mul by Sigma.
	float tmp_delta[kMaxR];
	for (int beta = 0; beta < r; ++beta)
	{
		float acc = 0.0f;
		for (int a = 0; a < d_s; ++a)
			acc += Uj[a * r + beta] * delta[a];
		tmp_delta[beta] = acc * Sig_e[beta];
	}

	// Step 6-7: (R^T delta)[a] = sum_beta U_i[a,beta] * tmp_delta[beta];
	// atomicAdd -(R^T delta) into out[i].
	for (int a = 0; a < d_s; ++a)
	{
		float Rt = 0.0f;
		for (int beta = 0; beta < r; ++beta)
			Rt += Ui[a * r + beta] * tmp_delta[beta];
		atomicAdd(&out[(size_t)i * d_s + a], -Rt);
	}
}

__global__ void sfa_zero_out_kernel(float* __restrict__ out, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) out[idx] = 0.0f;
}

} // anonymous namespace

bool sfa_laplacian_matvec_fp32(const float* U,
                                const float* Sigma,
                                const int* edge_src,
                                const int* edge_tgt,
                                const float* s,
                                float* out,
                                int T, int E, int d_s, int r,
                                cudaStream_t stream)
{
	if (T <= 0 || E <= 0 || d_s <= 0 || r <= 0) return true;
	if (d_s > kMaxDS || r > kMaxR)
	{
		fprintf(stderr, "[sfa-cuda] sfa_laplacian_matvec_fp32: d_s=%d (max %d) or r=%d (max %d) exceeded\n",
		        d_s, kMaxDS, r, kMaxR);
		return false;
	}

	cudaStream_t s_use = (stream != 0) ? stream : computeStream();

	// Zero output buffer.
	const int Tds = T * d_s;
	int zg = (Tds + kBlockElem - 1) / kBlockElem;
	sfa_zero_out_kernel<<<zg, kBlockElem, 0, s_use>>>(out, Tds);
	GLADES_CUDA_CHECK(cudaGetLastError());

	// Edge-parallel matvec.
	int grid = (E + kBlockElem - 1) / kBlockElem;
	sfa_laplacian_kernel<<<grid, kBlockElem, 0, s_use>>>(
	    U, Sigma, edge_src, edge_tgt, s, out, T, E, d_s, r);
	GLADES_CUDA_CHECK(cudaGetLastError());

	return true;
}

// ===========================================================================
//  Source assembly: b_i = U_i U_i^T P_q q_i + gamma * P_v v_i
// ===========================================================================
namespace {

__global__ void sfa_source_kernel(const float* __restrict__ U,
                                   const float* __restrict__ P_q,
                                   const float* __restrict__ P_v,
                                   const float* __restrict__ q,
                                   const float* __restrict__ v,
                                   float gamma,
                                   float* __restrict__ b,
                                   int T, int d_s, int d_h, int r)
{
	int i = blockIdx.x;       // one block per token
	int a = threadIdx.x;      // one thread per stalk-dim
	if (i >= T || a >= d_s) return;

	const float* Ui    = U + (size_t)i * d_s * r;
	const float* qi    = q + (size_t)i * d_h;
	const float* vi    = v + (size_t)i * d_h;
	float* bi          = b + (size_t)i * d_s;

	// Step 1: lifted_q[a] = sum_h P_q[a,h] * q_i[h]
	// Each thread computes its own component of lifted_q (length d_s).
	extern __shared__ float smem[];
	float* lifted_q = smem;            // size d_s
	float* tmp_r    = smem + d_s;      // size r

	float lifted = 0.0f;
	for (int h = 0; h < d_h; ++h)
		lifted += P_q[(size_t)a * d_h + h] * qi[h];
	lifted_q[a] = lifted;
	__syncthreads();

	// Step 2: tmp_r[beta] = sum_a U_i[a,beta] * lifted_q[a]
	// Thread 0 computes all r components (small r typically).
	if (a < r)
	{
		float acc = 0.0f;
		for (int aa = 0; aa < d_s; ++aa)
			acc += Ui[aa * r + a] * lifted_q[aa];
		tmp_r[a] = acc;
	}
	__syncthreads();

	// Step 3: b[a] = sum_beta U_i[a,beta] * tmp_r[beta]
	float proj = 0.0f;
	for (int beta = 0; beta < r; ++beta)
		proj += Ui[a * r + beta] * tmp_r[beta];

	// Step 4: optional value injection: + gamma * (P_v · v_i)[a]
	if (gamma != 0.0f)
	{
		float val_lift = 0.0f;
		for (int h = 0; h < d_h; ++h)
			val_lift += P_v[(size_t)a * d_h + h] * vi[h];
		proj += gamma * val_lift;
	}
	bi[a] = proj;
}

} // anonymous namespace

bool sfa_source_assembly_fp32(const float* U,
                               const float* P_q,
                               const float* P_v,
                               const float* q,
                               const float* v,
                               float gamma,
                               float* b,
                               int T, int d_s, int d_h, int r,
                               cudaStream_t stream)
{
	if (T <= 0 || d_s <= 0 || d_h <= 0 || r <= 0) return true;

	cudaStream_t s_use = (stream != 0) ? stream : computeStream();

	// Block size = d_s (must be at least r so the r-component reduction works).
	int block = d_s < r ? r : d_s;
	if (block > 1024)
	{
		fprintf(stderr, "[sfa-cuda] sfa_source_assembly_fp32: d_s=%d exceeds 1024 block size\n", d_s);
		return false;
	}
	size_t smem_bytes = sizeof(float) * (d_s + r);
	sfa_source_kernel<<<T, block, smem_bytes, s_use>>>(
	    U, P_q, P_v, q, v, gamma, b, T, d_s, d_h, r);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  Readout: y_i = P_o^T s_i  (output [T, d_h] from input [T, d_s])
// ===========================================================================
namespace {

__global__ void sfa_readout_kernel(const float* __restrict__ P_o,
                                    const float* __restrict__ s,
                                    float* __restrict__ y,
                                    int T, int d_s, int d_h)
{
	int i = blockIdx.x;
	int h = threadIdx.x;
	if (i >= T || h >= d_h) return;

	const float* si = s + (size_t)i * d_s;
	float acc = 0.0f;
	for (int a = 0; a < d_s; ++a)
		acc += P_o[(size_t)h * d_s + a] * si[a];
	y[(size_t)i * d_h + h] = acc;
}

} // anonymous namespace

bool sfa_readout_fp32(const float* P_o,
                       const float* s,
                       float* y,
                       int T, int d_s, int d_h,
                       cudaStream_t stream)
{
	if (T <= 0 || d_s <= 0 || d_h <= 0) return true;

	cudaStream_t s_use = (stream != 0) ? stream : computeStream();
	int block = d_h > 1024 ? 1024 : d_h;
	if (d_h > 1024)
	{
		// Fallback: scale grid across d_h chunks.
		fprintf(stderr, "[sfa-cuda] sfa_readout_fp32: d_h=%d > 1024 unsupported in v1\n", d_h);
		return false;
	}
	sfa_readout_kernel<<<T, block, 0, s_use>>>(P_o, s, y, T, d_s, d_h);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  CSR-format L_F matvec — deterministic, atomic-free, faster.
// ===========================================================================
namespace {

// Per-vertex L_F kernel. One block per vertex; threads loop over the d_s
// output components for that vertex.
//
// Each output entry out[i*d_s + a] accumulates:
//   sum over INCOMING edges e=(j,i):  (s_i[a] - R_{i<-j} s_j)[a]
//                                       = s_i[a] - (U_i (Sigma_e * (U_j^T s_j)))[a]
//   sum over OUTGOING edges e=(i,j):  -(R_{j<-i}^T (s_j - R_{j<-i} s_i))[a]
//
// For determinism, we iterate edges in CSR order (fixed at build time).
__global__ void sfa_laplacian_csr_kernel(const float* __restrict__ U,
                                          const float* __restrict__ Sigma,
                                          const int*   __restrict__ edge_src,
                                          const int*   __restrict__ edge_tgt,
                                          const int*   __restrict__ out_csr_off,
                                          const int*   __restrict__ out_csr_edges,
                                          const int*   __restrict__ in_csr_off,
                                          const int*   __restrict__ in_csr_edges,
                                          const float* __restrict__ s,
                                          float*       __restrict__ out,
                                          int T, int d_s, int r)
{
	const int i = blockIdx.x;          // one block per vertex i
	if (i >= T) return;
	const int a = threadIdx.x;         // one thread per stalk component
	if (a >= d_s) return;

	const float* Ui  = U + (size_t)i * d_s * r;
	const float* s_i = s + (size_t)i * d_s;
	float acc = 0.0f;

	// ====== Incoming edges (i is tgt; e = (j, i)) ======
	const int in_start = in_csr_off[i];
	const int in_end   = in_csr_off[i + 1];
	for (int k = in_start; k < in_end; ++k)
	{
		const int e = in_csr_edges[k];
		const int j = edge_src[e];
		const float* Uj    = U + (size_t)j * d_s * r;
		const float* Sig_e = Sigma + (size_t)e * r;
		const float* s_j   = s + (size_t)j * d_s;

		// R_{i<-j} s_j = U_i (Sigma_e * (U_j^T s_j)).
		// Compute U_j^T s_j once via shared memory? For now per-thread sum.
		float Rs_a = 0.0f;
		for (int beta = 0; beta < r; ++beta)
		{
			float inner = 0.0f;
			for (int aa = 0; aa < d_s; ++aa)
				inner += Uj[aa * r + beta] * s_j[aa];
			Rs_a += Ui[a * r + beta] * Sig_e[beta] * inner;
		}
		// delta[a] = s_i[a] - Rs_a; contribution to out[i*d_s + a] is +delta[a].
		acc += s_i[a] - Rs_a;
	}

	// ====== Outgoing edges (i is src; e = (i, j)) ======
	const int out_start = out_csr_off[i];
	const int out_end   = out_csr_off[i + 1];
	for (int k = out_start; k < out_end; ++k)
	{
		const int e = out_csr_edges[k];
		const int j = edge_tgt[e];
		const float* Uj    = U + (size_t)j * d_s * r;
		const float* Sig_e = Sigma + (size_t)e * r;
		const float* s_j   = s + (size_t)j * d_s;

		// delta[a] = s_j[a] - (R_{j<-i} s_i)[a]
		// Compute U_i^T s_i  (length r)
		float inner_arr[kMaxR];
		for (int beta = 0; beta < r; ++beta)
		{
			float inner = 0.0f;
			for (int aa = 0; aa < d_s; ++aa)
				inner += Ui[aa * r + beta] * s_i[aa];
			inner_arr[beta] = inner * Sig_e[beta];
		}
		// (R s_i)[a'] = sum_beta U_j[a',beta] inner_arr[beta], for all a'.
		// We need delta[a'] = s_j[a'] - Rs[a'] for the WHOLE row (because the
		// R^T delta computation needs all of delta).
		float delta[kMaxDS];
		for (int aa = 0; aa < d_s; ++aa)
		{
			float Rs = 0.0f;
			for (int beta = 0; beta < r; ++beta)
				Rs += Uj[aa * r + beta] * inner_arr[beta];
			delta[aa] = s_j[aa] - Rs;
		}
		// tmp_delta[beta] = (sum_a U_j[a,beta] delta[a]) * Sigma_e[beta]
		float tmp_delta[kMaxR];
		for (int beta = 0; beta < r; ++beta)
		{
			float td = 0.0f;
			for (int aa = 0; aa < d_s; ++aa)
				td += Uj[aa * r + beta] * delta[aa];
			tmp_delta[beta] = td * Sig_e[beta];
		}
		// (R^T delta)[a] = sum_beta U_i[a,beta] tmp_delta[beta]
		float Rt = 0.0f;
		for (int beta = 0; beta < r; ++beta)
			Rt += Ui[a * r + beta] * tmp_delta[beta];

		// Contribution to out[i*d_s + a] is -Rt.
		acc -= Rt;
	}

	out[(size_t)i * d_s + a] = acc;
}

} // anonymous namespace

bool sfa_laplacian_matvec_csr_fp32(const float* U,
                                    const float* Sigma,
                                    const int* edge_src,
                                    const int* edge_tgt,
                                    const int* out_csr_off,
                                    const int* out_csr_edges,
                                    const int* in_csr_off,
                                    const int* in_csr_edges,
                                    const float* s,
                                    float* out,
                                    int T, int E,
                                    int d_s, int r,
                                    cudaStream_t stream)
{
	if (T <= 0 || E <= 0 || d_s <= 0 || r <= 0) return true;
	if (d_s > kMaxDS || r > kMaxR)
	{
		fprintf(stderr, "[sfa-cuda] csr: d_s=%d (max %d) or r=%d (max %d) exceeded\n",
		        d_s, kMaxDS, r, kMaxR);
		return false;
	}
	cudaStream_t s_use = (stream != 0) ? stream : computeStream();

	// One block per vertex, d_s threads per block.
	int block = d_s;
	if (block < 32) block = 32;  // round up for warp utilization
	sfa_laplacian_csr_kernel<<<T, block, 0, s_use>>>(
	    U, Sigma, edge_src, edge_tgt,
	    out_csr_off, out_csr_edges, in_csr_off, in_csr_edges,
	    s, out, T, d_s, r);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Host-side CSR builder. Single-pass counting + filling.
void sfa_build_csr_host(const int* edge_src, const int* edge_tgt, int E, int T,
                        int* out_csr_off, int* out_csr_edges,
                        int* in_csr_off, int* in_csr_edges)
{
	// Count edges per source vertex (outgoing) and per target (incoming).
	for (int i = 0; i <= T; ++i) { out_csr_off[i] = 0; in_csr_off[i] = 0; }
	for (int e = 0; e < E; ++e)
	{
		out_csr_off[edge_src[e] + 1]++;
		in_csr_off[edge_tgt[e] + 1]++;
	}
	// Prefix sum.
	for (int i = 1; i <= T; ++i) { out_csr_off[i] += out_csr_off[i-1]; in_csr_off[i] += in_csr_off[i-1]; }
	// Fill (using a temp cursor array).
	std::vector<int> out_cursor(T, 0), in_cursor(T, 0);
	for (int e = 0; e < E; ++e)
	{
		int sv = edge_src[e]; int tv = edge_tgt[e];
		out_csr_edges[out_csr_off[sv] + out_cursor[sv]++] = e;
		in_csr_edges [in_csr_off [tv] + in_cursor [tv]++] = e;
	}
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
