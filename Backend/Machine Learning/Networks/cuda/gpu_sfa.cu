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

// 2026-05-15 Phase 4b: per-vertex diagonal of L_F + λI for Jacobi
// preconditioning of the Tikhonov solve.
//
// L_F = δ^T δ where δ is the sheaf coboundary.  For edge e=(i,j):
//   δ x[e] = x_j - R_{j<-i} x_i  with  R_{j<-i} = U_j Σ_e U_i^T
//
// δ^T applied back: at vertex j contributes +(δ x[e]); at vertex i
// contributes -(R_{j<-i})^T (δ x[e]).  Diagonal of L_F at (v, a, a):
//   from each edge where v = tgt (incoming): +1  (identity term)
//   from each edge where v = src (outgoing): +(R^T R)[a,a]
//                                          = ‖R_{j<-i}^T[a,:]‖²
// With orthonormal U:
//   ‖R_{j<-i}^T[a,:]‖² = sum_β U_i[a,β]² Σ_e[β]²
// Plus λ from regulariser (added separately).
__global__ void sfa_diagonal_kernel(const float* __restrict__ U,
                                     const float* __restrict__ Sigma,
                                     const int*   __restrict__ edge_src,
                                     const int*   __restrict__ edge_tgt,
                                     float* __restrict__ diag,
                                     int T, int E, int d_s, int r)
{
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= E) return;

	const int i = edge_src[e];
	const int j = edge_tgt[e];
	const float* Ui    = U + (size_t)i * d_s * r;
	const float* Sig_e = Sigma + (size_t)e * r;

	for (int a = 0; a < d_s; ++a)
	{
		// Vertex i (src): +sum_β U_i[a,β]² Σ_e[β]²
		float v_i = 0.0f;
		for (int beta = 0; beta < r; ++beta)
		{
			float u_ia = Ui[a * r + beta];
			float sig = Sig_e[beta];
			v_i += u_ia * u_ia * sig * sig;
		}
		atomicAdd(&diag[(size_t)i * d_s + a], v_i);
		// Vertex j (tgt): +1 from the identity x_j term.
		atomicAdd(&diag[(size_t)j * d_s + a], 1.0f);
	}
}

__global__ void sfa_diag_add_lambda_kernel(float* __restrict__ diag, float lambda,
                                            int Tds)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < Tds) diag[idx] += lambda;
}

// 2026-05-15 Phase 4b: Jacobi-preconditioned Richardson step.
// Fused: s += α · D^{-1} · (b - L_F s - λ s)
// where the parenthesized expression has already been computed into res.
// Replaces a 4-axpy chain (memcpy, axpy×3) with one element-wise kernel.
__global__ void sfa_jacobi_step_kernel(float* __restrict__ s,
                                        const float* __restrict__ b,
                                        const float* __restrict__ Ls,
                                        const float* __restrict__ Dinv,
                                        float lambda, float alpha,
                                        int Tds)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Tds) return;
	const float si = s[idx];
	const float res = b[idx] - Ls[idx] - lambda * si;
	s[idx] = si + alpha * Dinv[idx] * res;
}

__global__ void sfa_reciprocal_clamped_kernel(float* __restrict__ Dinv,
                                               const float* __restrict__ diag,
                                               float eps, int Tds)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= Tds) return;
	float d = diag[idx];
	if (d < eps) d = eps;
	Dinv[idx] = 1.0f / d;
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
	const int i = blockIdx.x;
	const int h = blockIdx.y * blockDim.x + threadIdx.x;
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
	// 2D grid: blockIdx.x = token, blockIdx.y = d_h chunk.  Block size 256
	// (sweet spot for typical d_h ≥ 256; smaller d_h pays a partial-block
	// occupancy cost but the kernel is fast either way).
	const int block = 256;
	const int blocks_h = (d_h + block - 1) / block;
	dim3 grid(T, blocks_h, 1);
	sfa_readout_kernel<<<grid, block, 0, s_use>>>(P_o, s, y, T, d_s, d_h);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  CSR-format L_F matvec — block-cooperative variant.
//
// 2026-05-15: this kernel replaces the per-stalk-component v1 implementation.
//
// Design:
//   - One block per vertex (T blocks total).
//   - Block size = 128 threads — covers the average vertex degree
//     (W + n_sinks ≈ 140) with one thread per edge in most cases.
//   - Each thread t reads ONE edge from the vertex's edge list and
//     computes its full contribution to out[i*d_s + 0..d_s-1] without
//     redundancy.  Threads with no edge in their slot become no-ops.
//   - U_i and s_i are cached in shared memory once per block (one read
//     for the vertex instead of W+n_sinks reads, one per edge).
//   - Per-edge contributions accumulate into a per-block shared-memory
//     output via shared-mem atomicAdd (≈free vs global atomic), then a
//     single non-atomic write per output element finalizes out[i*d_s + a].
//
// vs the previous CSR v1 kernel (1 block per vertex, d_s threads, per-
// thread loop over edges with d_s× redundant work): ~30× speedup at
// T=16384, d_s=8, r=4, W=128.
//
// Determinism note: shared-mem atomicAdd order is not fixed, so this
// variant is NOT bit-exact deterministic.  For Probe B' (forward-only
// NLL comparison at the ~1e-2 nat scale) FP32 ULP-level reordering is
// well below the signal floor.  If strict determinism is later required,
// fall back to v1 via the `_v1` host wrapper (TODO).
// ===========================================================================
namespace {

// Cached U_i / s_i lives in shared memory at the start of the block.
// Remaining shared memory holds the d_s-element output accumulator.
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
	const int tid = threadIdx.x;
	const int bsz = blockDim.x;

	// Shared memory layout:
	//   out_acc  [d_s]       — per-block accumulator for out[i*d_s + a]
	//   Ui_cache [d_s * r]   — cached U_i
	//   si_cache [d_s]       — cached s_i
	extern __shared__ float smem[];
	float* out_acc  = smem;
	float* Ui_cache = smem + d_s;
	float* si_cache = smem + d_s + d_s * r;

	// Initialize accumulator + caches.
	for (int idx = tid; idx < d_s; idx += bsz) out_acc[idx] = 0.0f;
	for (int idx = tid; idx < d_s * r; idx += bsz)
		Ui_cache[idx] = U[(size_t)i * d_s * r + idx];
	for (int idx = tid; idx < d_s; idx += bsz)
		si_cache[idx] = s[(size_t)i * d_s + idx];
	__syncthreads();

	// ====== Incoming edges (i is tgt; e = (j, i)) ======
	const int in_start = in_csr_off[i];
	const int in_end   = in_csr_off[i + 1];
	for (int k = in_start + tid; k < in_end; k += bsz)
	{
		const int e = in_csr_edges[k];
		const int j = edge_src[e];
		const float* Uj    = U + (size_t)j * d_s * r;
		const float* Sig_e = Sigma + (size_t)e * r;
		const float* s_j   = s + (size_t)j * d_s;

		// tmp_inner[beta] = (U_j^T s_j)[beta] * Sigma_e[beta]
		float tmp_inner[kMaxR];
		for (int beta = 0; beta < r; ++beta)
		{
			float v = 0.0f;
			for (int a = 0; a < d_s; ++a)
				v += Uj[a * r + beta] * s_j[a];
			tmp_inner[beta] = v * Sig_e[beta];
		}
		// delta[a] = s_i[a] - (U_i tmp_inner)[a]; contribution to out_i is +delta[a].
		for (int a = 0; a < d_s; ++a)
		{
			float Rs = 0.0f;
			for (int beta = 0; beta < r; ++beta)
				Rs += Ui_cache[a * r + beta] * tmp_inner[beta];
			atomicAdd(&out_acc[a], si_cache[a] - Rs);
		}
	}

	// ====== Outgoing edges (i is src; e = (i, j)) ======
	const int out_start = out_csr_off[i];
	const int out_end   = out_csr_off[i + 1];
	for (int k = out_start + tid; k < out_end; k += bsz)
	{
		const int e = out_csr_edges[k];
		const int j = edge_tgt[e];
		const float* Uj    = U + (size_t)j * d_s * r;
		const float* Sig_e = Sigma + (size_t)e * r;
		const float* s_j   = s + (size_t)j * d_s;

		// tmp_inner[beta] = (U_i^T s_i)[beta] * Sigma_e[beta]   (cached U_i, s_i)
		float tmp_inner[kMaxR];
		for (int beta = 0; beta < r; ++beta)
		{
			float v = 0.0f;
			for (int a = 0; a < d_s; ++a)
				v += Ui_cache[a * r + beta] * si_cache[a];
			tmp_inner[beta] = v * Sig_e[beta];
		}
		// delta[a] = s_j[a] - (U_j tmp_inner)[a]
		float delta[kMaxDS];
		for (int a = 0; a < d_s; ++a)
		{
			float Rs = 0.0f;
			for (int beta = 0; beta < r; ++beta)
				Rs += Uj[a * r + beta] * tmp_inner[beta];
			delta[a] = s_j[a] - Rs;
		}
		// tmp_delta[beta] = (U_j^T delta)[beta] * Sigma_e[beta]
		float tmp_delta[kMaxR];
		for (int beta = 0; beta < r; ++beta)
		{
			float v = 0.0f;
			for (int a = 0; a < d_s; ++a)
				v += Uj[a * r + beta] * delta[a];
			tmp_delta[beta] = v * Sig_e[beta];
		}
		// contribution to out_i is -(U_i tmp_delta)[a]
		for (int a = 0; a < d_s; ++a)
		{
			float Rt = 0.0f;
			for (int beta = 0; beta < r; ++beta)
				Rt += Ui_cache[a * r + beta] * tmp_delta[beta];
			atomicAdd(&out_acc[a], -Rt);
		}
	}

	__syncthreads();
	// Final write: one thread per output element.
	for (int idx = tid; idx < d_s; idx += bsz)
		out[(size_t)i * d_s + idx] = out_acc[idx];
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

	// One block per vertex; block size = 128 threads (good for typical
	// degrees W + n_sinks ≈ 140 — one thread per edge mostly, with a tail
	// loop for higher degrees).  Shared memory: out_acc[d_s] + Ui_cache[d_s*r]
	// + si_cache[d_s] = d_s * (2 + r) floats.
	const int block = 128;
	const size_t smem_bytes = sizeof(float) * (size_t)d_s * (size_t)(2 + r);
	sfa_laplacian_csr_kernel<<<T, block, smem_bytes, s_use>>>(
	    U, Sigma, edge_src, edge_tgt,
	    out_csr_off, out_csr_edges, in_csr_off, in_csr_edges,
	    s, out, T, d_s, r);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// 2026-05-15 Phase 4b: compute diag(L_F + λI) on device.  Output is
// [T · d_s] FP32, ready to feed into a Jacobi-preconditioned solver.
bool sfa_laplacian_diagonal_fp32(const float* U,
                                  const float* Sigma,
                                  const int* edge_src,
                                  const int* edge_tgt,
                                  float lambda,
                                  float* diag,
                                  int T, int E,
                                  int d_s, int r,
                                  cudaStream_t stream)
{
	if (T <= 0 || E <= 0 || d_s <= 0 || r <= 0) return true;
	cudaStream_t s_use = (stream != 0) ? stream : computeStream();

	const int Tds = T * d_s;
	int zg = (Tds + kBlockElem - 1) / kBlockElem;
	sfa_zero_out_kernel<<<zg, kBlockElem, 0, s_use>>>(diag, Tds);
	GLADES_CUDA_CHECK(cudaGetLastError());

	int eg = (E + kBlockElem - 1) / kBlockElem;
	sfa_diagonal_kernel<<<eg, kBlockElem, 0, s_use>>>(
	    U, Sigma, edge_src, edge_tgt, diag, T, E, d_s, r);
	GLADES_CUDA_CHECK(cudaGetLastError());

	sfa_diag_add_lambda_kernel<<<zg, kBlockElem, 0, s_use>>>(diag, lambda, Tds);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Compute 1/diag with a floor at eps.
bool sfa_jacobi_inverse_diagonal_fp32(const float* diag, float* Dinv,
                                       float eps, int T, int d_s,
                                       cudaStream_t stream)
{
	if (T <= 0 || d_s <= 0) return true;
	cudaStream_t s_use = (stream != 0) ? stream : computeStream();
	const int Tds = T * d_s;
	int g = (Tds + kBlockElem - 1) / kBlockElem;
	sfa_reciprocal_clamped_kernel<<<g, kBlockElem, 0, s_use>>>(
	    Dinv, diag, eps, Tds);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Fused Jacobi step: s += α · D^{-1} · (b - L_F s - λ s).
// Caller has already computed Ls = L_F · s; this kernel does the rest
// in a single pass over [T · d_s] elements.
bool sfa_jacobi_step_fp32(float* s, const float* b, const float* Ls,
                           const float* Dinv, float lambda, float alpha,
                           int T, int d_s,
                           cudaStream_t stream)
{
	if (T <= 0 || d_s <= 0) return true;
	cudaStream_t s_use = (stream != 0) ? stream : computeStream();
	const int Tds = T * d_s;
	int g = (Tds + kBlockElem - 1) / kBlockElem;
	sfa_jacobi_step_kernel<<<g, kBlockElem, 0, s_use>>>(
	    s, b, Ls, Dinv, lambda, alpha, Tds);
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

// ===========================================================================
//  Phase 8 (2026-05-15): backward kernels for SFA training.
//
// Full backward chain (given dp_out at the SFA-swap layer):
//   1. Shear:  dy = sign · dp_out;  dp_in = dp_out (pass-through)
//   2. Readout backward:                                       (kernel below)
//        dσ_i[a]   = sum_h P_o[h,a] · dy_i[h]
//        dP_o[h,a] += sum_i σ_i[a] · dy_i[h]   (atomic accumulate)
//   3. Tikhonov adjoint solve:  w = M^{-1} dσ  (reuses Jacobi solver)
//      → db = w
//   4. Source-assembly backward:                               (kernel below)
//        Per token i:
//          dP_q[c,h] += (U_i U_i^T db_i)[c] · q_i[h]
//          dP_v[a,h] += γ · db_i[a] · v_i[h]
//          dq_i[h]   += sum_a,c (U_i U_i^T)[a,c] · P_q[c,h] · db_i[a]
//          dv_i[h]   += γ · sum_a P_v[a,h] · db_i[a]
//          dU_i[c,β] += δ_{a,c} (U_i^T P_q q_i)[β] · db_i[a]
//                     + U_i[a,β] · (P_q q_i)[c] · db_i[a]   (source path)
//   5. Implicit diff through L_F:                              (kernel below)
//        For each edge e=(i,j), compute "edge bilinear" gradients:
//        Let u = U_i^T σ_i, ū = U_i^T w_i, v = U_j^T σ_j, v̄ = U_j^T w_j.
//        Bilinear form from edge e:
//          B_e = ū^T (Σ² ⊙ u)              (R^T R diagonal at i)
//              + w_j · σ_j                  (identity at j; constant in Σ, U)
//              - ū^T (Σ ⊙ v)                (off-diag at (i,j))
//              - v̄^T (Σ ⊙ u)                (off-diag at (j,i))
//        Gradients (per edge, accumulated atomically):
//          dΣ_e[β]    += -∂B_e/∂Σ_e[β]
//                      = ū[β] v[β] + v̄[β] u[β] - 2 Σ_e[β] ū[β] u[β]
//          dU_i[a,β]  += -∂B_e/∂U_i[a,β]
//                      = w_i[a] Σ_e[β] (v[β] - Σ_e[β] u[β])
//                      + σ_i[a] Σ_e[β] (v̄[β] - Σ_e[β] ū[β])
//          dU_j[a,β]  += -∂B_e/∂U_j[a,β]
//                      = ū[β] Σ_e[β] σ_j[a] + w_j[a] Σ_e[β] u[β]
//
// All kernels write FP32 gradient buffers; trainer applies Adam updates.
// ===========================================================================

namespace {

// Readout backward: dσ + dP_o accumulation.
__global__ void sfa_readout_bwd_dsigma_kernel(const float* __restrict__ P_o,
                                                const float* __restrict__ dy,
                                                float* __restrict__ dsigma,
                                                int T, int d_s, int d_h)
{
	const int i = blockIdx.x;
	const int a = blockIdx.y * blockDim.x + threadIdx.x;
	if (i >= T || a >= d_s) return;

	const float* dyi = dy + (size_t)i * d_h;
	float acc = 0.0f;
	for (int h = 0; h < d_h; ++h)
		acc += P_o[(size_t)h * d_s + a] * dyi[h];
	dsigma[(size_t)i * d_s + a] = acc;
}

__global__ void sfa_readout_bwd_dPo_kernel(const float* __restrict__ sigma,
                                            const float* __restrict__ dy,
                                            float* __restrict__ dPo,
                                            int T, int d_s, int d_h)
{
	const int h = blockIdx.x;
	const int a = blockIdx.y * blockDim.x + threadIdx.x;
	if (h >= d_h || a >= d_s) return;

	float acc = 0.0f;
	for (int i = 0; i < T; ++i)
		acc += sigma[(size_t)i * d_s + a] * dy[(size_t)i * d_h + h];
	atomicAdd(&dPo[(size_t)h * d_s + a], acc);
}

// Source-assembly backward: dq, dv, dP_q, dP_v, dU (source-path contribution).
//
// One block per token; threads cooperate on d_s and d_h ranges via shared mem.
// For simplicity the kernel does scalar atomics; with T=16384 and d_s≤32 the
// dP_q/dP_v atomics are concentrated on a [d_s × d_h] buffer of size ~64K
// elements — manageable contention.
__global__ void sfa_source_bwd_kernel(const float* __restrict__ U,
                                       const float* __restrict__ P_q,
                                       const float* __restrict__ P_v,
                                       const float* __restrict__ q,
                                       const float* __restrict__ v,
                                       const float* __restrict__ db,
                                       float gamma,
                                       float* __restrict__ dP_q,
                                       float* __restrict__ dP_v,
                                       float* __restrict__ dU,
                                       float* __restrict__ dq,
                                       float* __restrict__ dv,
                                       int T, int d_s, int d_h, int r)
{
	const int i = blockIdx.x;
	if (i >= T) return;

	const float* Ui  = U  + (size_t)i * d_s * r;
	const float* qi  = q  + (size_t)i * d_h;
	const float* vi  = v  + (size_t)i * d_h;
	const float* dbi = db + (size_t)i * d_s;
	float* dUi       = dU + (size_t)i * d_s * r;
	float* dqi       = dq + (size_t)i * d_h;
	float* dvi       = dv + (size_t)i * d_h;

	// One thread per d_h component for q/v gradients & projection backwards.
	const int h = threadIdx.x;
	if (h >= d_h) return;

	// Compute (P_q q_i)[c] for each c — reuse across threads via shared mem.
	// But h here is the d_h index; we need (P_q q_i)[c] = sum_h P_q[c,h] q_i[h].
	// To avoid quadratic-in-d_h shared work per token, we just loop.
	//
	// Step 1: compute (U_i^T db_i)[β]   for β = 0..r-1  (length r per token)
	__shared__ float Udb[16];     // up to kMaxR
	__shared__ float UUtdb[128];  // U_i U_i^T db_i  — length d_s
	if (h < r)
	{
		float acc = 0.0f;
		for (int a = 0; a < d_s; ++a)
			acc += Ui[a * r + h] * dbi[a];
		Udb[h] = acc;
	}
	__syncthreads();
	if (h < d_s)
	{
		float acc = 0.0f;
		for (int beta = 0; beta < r; ++beta)
			acc += Ui[h * r + beta] * Udb[beta];
		UUtdb[h] = acc;
	}
	__syncthreads();

	// Step 2: dq_i[h] += sum_a (U_i U_i^T)[a, ???] · P_q[???, h] · db_i[a]
	//      = (sum_a P_q[a, h] · UUtdb[a])  [identifying the projection structure]
	// Recall b_i[a] = sum_{β,c,h} U_i[a,β] U_i[c,β] P_q[c,h] q_i[h] + γ P_v[a,h] v_i[h]
	//              = sum_h P_q^proj_i[a, h] q_i[h] + γ P_v[a,h] v_i[h]
	//   where P_q^proj_i = (U_i U_i^T) · P_q.
	// dq_i[h] = sum_a db_i[a] · P_q^proj_i[a, h] = (UUtdb)^T · P_q[:, h]
	{
		float acc = 0.0f;
		for (int a = 0; a < d_s; ++a)
			acc += UUtdb[a] * P_q[(size_t)a * d_h + h];
		dqi[h] = acc;
	}
	// dv_i[h] = γ · sum_a db_i[a] · P_v[a, h]
	{
		float acc = 0.0f;
		for (int a = 0; a < d_s; ++a)
			acc += dbi[a] * P_v[(size_t)a * d_h + h];
		dvi[h] = gamma * acc;
	}
	// dP_q[c, h] += UUtdb[c] · q_i[h]   (atomic, per c)
	for (int c = 0; c < d_s; ++c)
		atomicAdd(&dP_q[(size_t)c * d_h + h], UUtdb[c] * qi[h]);
	// dP_v[a, h] += γ · db_i[a] · v_i[h]   (atomic, per a)
	for (int a = 0; a < d_s; ++a)
		atomicAdd(&dP_v[(size_t)a * d_h + h], gamma * dbi[a] * vi[h]);

	// dU_i contribution from source assembly (in addition to L_F implicit diff):
	//   b_i[a] = sum_β U_i[a,β] · (U_i^T P_q q_i)[β] + γ P_v[a,h] v_i[h]
	//   ∂b_i[a]/∂U_i[c,β] = δ_{a,c} (U_i^T P_q q_i)[β] + U_i[a,β] (P_q q_i)[c]
	// dU_i[c, β] += db_i[a=c] · (U_i^T P_q q_i)[β] + sum_a db_i[a] · U_i[a,β] · (P_q q_i)[c]
	//
	// Need (P_q q_i)[c] — compute it from h-parallel reduction.  Reuse smem.
	__shared__ float Pq_qi[128];  // length d_s; (P_q q_i)[c]
	if (h < d_s)
	{
		float acc = 0.0f;
		for (int hh = 0; hh < d_h; ++hh)
			acc += P_q[(size_t)h * d_h + hh] * qi[hh];
		Pq_qi[h] = acc;
	}
	__syncthreads();
	// Also need (U_i^T P_q q_i)[β] (length r).
	__shared__ float UTPqq[16];
	if (h < r)
	{
		float acc = 0.0f;
		for (int c = 0; c < d_s; ++c)
			acc += Ui[c * r + h] * Pq_qi[c];
		UTPqq[h] = acc;
	}
	__syncthreads();
	// Now accumulate dU_i: h-parallel over (c, β) of size d_s * r.
	// Use (c, β) indexed by single thread h if h < d_s * r.
	if (h < d_s * r)
	{
		const int c    = h / r;
		const int beta = h % r;
		float grad = dbi[c] * UTPqq[beta];
		// + sum_a db_i[a] · U_i[a, β] · Pq_qi[c]   (second term)
		float second = 0.0f;
		for (int a = 0; a < d_s; ++a)
			second += dbi[a] * Ui[a * r + beta];
		grad += second * Pq_qi[c];
		// dU is shared with the L_F implicit-diff kernel; use atomic add.
		atomicAdd(&dUi[c * r + beta], grad);
	}
}

// Implicit-differentiation through L_F: edge-parallel gradient accumulation
// for U and Σ.  Inputs are σ (forward solution) and w = M^{-1} dσ (adjoint).
__global__ void sfa_laplacian_bwd_kernel(const float* __restrict__ U,
                                          const float* __restrict__ Sigma,
                                          const int*   __restrict__ edge_src,
                                          const int*   __restrict__ edge_tgt,
                                          const float* __restrict__ sigma,
                                          const float* __restrict__ w,
                                          float* __restrict__ dU,
                                          float* __restrict__ dSigma,
                                          int T, int E, int d_s, int r)
{
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= E) return;

	const int i = edge_src[e];
	const int j = edge_tgt[e];
	const float* Ui    = U + (size_t)i * d_s * r;
	const float* Uj    = U + (size_t)j * d_s * r;
	const float* Sig_e = Sigma + (size_t)e * r;
	const float* si    = sigma + (size_t)i * d_s;
	const float* sj    = sigma + (size_t)j * d_s;
	const float* wi    = w + (size_t)i * d_s;
	const float* wj    = w + (size_t)j * d_s;

	// u = U_i^T σ_i,  ū = U_i^T w_i,  v = U_j^T σ_j,  v̄ = U_j^T w_j   (length r)
	float u[kMaxR], ubar[kMaxR], v[kMaxR], vbar[kMaxR];
	for (int beta = 0; beta < r; ++beta)
	{
		float u_b = 0.0f, ub_b = 0.0f, v_b = 0.0f, vb_b = 0.0f;
		for (int a = 0; a < d_s; ++a)
		{
			u_b  += Ui[a * r + beta] * si[a];
			ub_b += Ui[a * r + beta] * wi[a];
			v_b  += Uj[a * r + beta] * sj[a];
			vb_b += Uj[a * r + beta] * wj[a];
		}
		u[beta]    = u_b;
		ubar[beta] = ub_b;
		v[beta]    = v_b;
		vbar[beta] = vb_b;
	}

	// dΣ_e[β] += ū[β] v[β] + v̄[β] u[β] - 2 Σ_e[β] ū[β] u[β]
	for (int beta = 0; beta < r; ++beta)
	{
		float g = ubar[beta] * v[beta]
		        + vbar[beta] * u[beta]
		        - 2.0f * Sig_e[beta] * ubar[beta] * u[beta];
		atomicAdd((float*)&dSigma[(size_t)e * r + beta], g);
	}

	// dU_i[a, β] += w_i[a] Σ_e[β] (v[β] - Σ_e[β] u[β])
	//            + σ_i[a] Σ_e[β] (v̄[β] - Σ_e[β] ū[β])
	float* dUi = dU + (size_t)i * d_s * r;
	float* dUj = dU + (size_t)j * d_s * r;
	for (int a = 0; a < d_s; ++a)
	{
		for (int beta = 0; beta < r; ++beta)
		{
			float t1 = wi[a] * Sig_e[beta] * (v[beta]    - Sig_e[beta] * u[beta]);
			float t2 = si[a] * Sig_e[beta] * (vbar[beta] - Sig_e[beta] * ubar[beta]);
			atomicAdd(&dUi[a * r + beta], t1 + t2);
		}
	}
	// dU_j[a, β] += ū[β] Σ_e[β] σ_j[a] + w_j[a] Σ_e[β] u[β]
	for (int a = 0; a < d_s; ++a)
	{
		for (int beta = 0; beta < r; ++beta)
		{
			float t = Sig_e[beta] * (ubar[beta] * sj[a] + wj[a] * u[beta]);
			atomicAdd(&dUj[a * r + beta], t);
		}
	}
}

} // anonymous namespace

bool sfa_readout_backward_fp32(const float* P_o, const float* sigma,
                                const float* dy,
                                float* dsigma, float* dPo,
                                int T, int d_s, int d_h,
                                cudaStream_t stream)
{
	if (T <= 0 || d_s <= 0 || d_h <= 0) return true;
	cudaStream_t s_use = (stream != 0) ? stream : computeStream();

	// dσ kernel: T blocks × ceil(d_s/256) blocks, 256 threads.
	{
		const int block = 256;
		const int blocks_a = (d_s + block - 1) / block;
		dim3 grid(T, blocks_a, 1);
		sfa_readout_bwd_dsigma_kernel<<<grid, block, 0, s_use>>>(
		    P_o, dy, dsigma, T, d_s, d_h);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}
	// dP_o kernel: d_h blocks × ceil(d_s/256) blocks.
	{
		const int block = 256;
		const int blocks_a = (d_s + block - 1) / block;
		dim3 grid(d_h, blocks_a, 1);
		sfa_readout_bwd_dPo_kernel<<<grid, block, 0, s_use>>>(
		    sigma, dy, dPo, T, d_s, d_h);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}
	return true;
}

bool sfa_source_assembly_backward_fp32(const float* U, const float* P_q,
                                        const float* P_v, const float* q,
                                        const float* v, const float* db,
                                        float gamma,
                                        float* dP_q, float* dP_v, float* dU,
                                        float* dq, float* dv,
                                        int T, int d_s, int d_h, int r,
                                        cudaStream_t stream)
{
	if (T <= 0 || d_s <= 0 || d_h <= 0 || r <= 0) return true;
	cudaStream_t s_use = (stream != 0) ? stream : computeStream();

	// Block size = d_h (capped at 1024).  For d_h > 1024 we'd need to split,
	// but at d_h=2048 this kernel uses block=1024 with a striding loop inside
	// — to keep code simple in this iter we require d_h ≤ 1024 and ask callers
	// to chunk if needed.  At d_h = m = 2048 we exceed; fall back to scalar.
	if (d_h > 1024 || d_s > 128 || r > 16)
	{
		fprintf(stderr, "[sfa-cuda] source_assembly_backward: d_h=%d d_s=%d r=%d "
		                "exceeds kernel limits (1024/128/16); please chunk\n",
		        d_h, d_s, r);
		return false;
	}
	sfa_source_bwd_kernel<<<T, d_h, 0, s_use>>>(
	    U, P_q, P_v, q, v, db, gamma,
	    dP_q, dP_v, dU, dq, dv,
	    T, d_s, d_h, r);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool sfa_laplacian_backward_fp32(const float* U, const float* Sigma,
                                  const int* edge_src, const int* edge_tgt,
                                  const float* sigma, const float* w,
                                  float* dU, float* dSigma,
                                  int T, int E, int d_s, int r,
                                  cudaStream_t stream)
{
	if (T <= 0 || E <= 0 || d_s <= 0 || r <= 0) return true;
	if (d_s > kMaxDS || r > kMaxR) {
		fprintf(stderr, "[sfa-cuda] laplacian_backward: d_s=%d (max %d) r=%d (max %d)\n",
		        d_s, kMaxDS, r, kMaxR);
		return false;
	}
	cudaStream_t s_use = (stream != 0) ? stream : computeStream();
	int grid = (E + kBlockElem - 1) / kBlockElem;
	sfa_laplacian_bwd_kernel<<<grid, kBlockElem, 0, s_use>>>(
	    U, Sigma, edge_src, edge_tgt, sigma, w, dU, dSigma,
	    T, E, d_s, r);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ---------------------------------------------------------------------------
// Paradigm #255 DSA: commutation-defect kernel.
//
// Computes per-token commutation defect for paradigm #255 (Dynamic Sheaf
// Activation).  In the SFA construction the edge set is causal-only: a
// single edge e = (i-1, i) carries one Sigma_e for both directions of
// traversal (R_{j<-i} = U_j diag(Sigma_e) U_i^T and its conjugate-transpose
// R_{i<-j} = U_i diag(Sigma_e) U_j^T).  The round-trip restriction map
// therefore reduces to
//
//   R_{i<-j} R_{j<-i}  =  U_i diag(Sigma_e^2) U_i^T
//
// and, in the rank-r subspace (where U_i is orthonormal), the Frobenius
// defect against the identity is
//
//   eps_i  =  sqrt( sum_beta ( Sigma_e[beta]^2 - 1.0 )^2 )
//
// where e is the immediate-predecessor edge into i (src = i-1, tgt = i).
// If no such edge exists, eps[i] = 0.
//
// The predecessor edge is discovered via the existing in-CSR structure
// (incoming edges to i); no new host setup is required.
//
// One block per token, 32 threads.  Thread 0 finds the predecessor edge
// and writes its index to shared memory; the warp then cooperates on the
// r-dim reduction with __shfl_xor_sync.
//
// Reference C++ implementation: research/dsa_probe_o_prototype.cpp
// (compute_defect_per_token).
// ---------------------------------------------------------------------------
namespace {

__global__ void sfa_defect_step1_kernel(const float* __restrict__ Sigma,
                                         const int*   __restrict__ edge_src,
                                         const int*   __restrict__ in_csr_off,
                                         const int*   __restrict__ in_csr_edges,
                                         int T, int r,
                                         float* __restrict__ eps)
{
	int i = blockIdx.x;
	if (i >= T) return;

	__shared__ int s_pred_edge;

	if (threadIdx.x == 0)
	{
		int pred = -1;
		int in_start = in_csr_off[i];
		int in_end   = in_csr_off[i + 1];
		for (int k = in_start; k < in_end; ++k)
		{
			int e = in_csr_edges[k];
			if (edge_src[e] == i - 1) { pred = e; break; }
		}
		s_pred_edge = pred;
	}
	__syncthreads();

	int pred_edge = s_pred_edge;
	if (pred_edge < 0)
	{
		if (threadIdx.x == 0) eps[i] = 0.0f;
		return;
	}

	const float* Sig_e = Sigma + (size_t)pred_edge * r;

	float acc = 0.0f;
	for (int b = threadIdx.x; b < r; b += blockDim.x)
	{
		float s  = Sig_e[b];
		float d  = s * s - 1.0f;
		acc += d * d;
	}

	// Warp reduction.  Block size is one warp (32 threads).
	unsigned mask = 0xFFFFFFFFu;
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1)
	{
		acc += __shfl_xor_sync(mask, acc, offset);
	}

	if (threadIdx.x == 0) eps[i] = sqrtf(acc);
}

}  // namespace

bool sfa_defect_step1_fp32(const float* Sigma,
                            const int*   edge_src,
                            const int*   in_csr_off,
                            const int*   in_csr_edges,
                            float*       eps,
                            int T, int r,
                            cudaStream_t stream)
{
	if (T <= 0 || r <= 0) return true;
	if (r > kMaxR)
	{
		fprintf(stderr, "[sfa-cuda] defect_step1: r=%d exceeds kMaxR=%d\n",
		        r, kMaxR);
		return false;
	}
	cudaStream_t s_use = (stream != 0) ? stream : computeStream();
	const int block = 32;  // one warp
	sfa_defect_step1_kernel<<<T, block, 0, s_use>>>(
	    Sigma, edge_src, in_csr_off, in_csr_edges,
	    T, r, eps);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ---------------------------------------------------------------------------
// Paradigm #255 DSA — Candidate 1 (iter-14): U-frame defect.
//
// After iter-13 Probe O at flagship scale FALSIFIED the Σ-based defect
// formula (per-position ε flat at ~0.92 across all 8 buckets, Pearson r =
// -0.187 vs Phase 8b NLL), the cocycle-relevant signal is hypothesised to
// live in the stalk-frame structure U_i rather than in Σ.  Candidate 1
// implements
//
//   eps^U_i  =  ‖ U_i^T U_{i-1}  −  I_r ‖_F                       (eq. 1')
//
// — the Frobenius distance between the adjacent-stalk Gram product and
// the rank-r identity.  When U_{i-1} and U_i span the same r-dim
// subspace (orthonormal frames aligned), U_i^T U_{i-1} = I_r and the
// defect is 0.  When the subspaces rotate (e.g., adjacent positions
// process different content), the off-diagonals fill in and the defect
// grows.
//
// One block per token (i >= 1).  Block size = min(r*r, 32).  Each thread
// computes one (beta1, beta2) entry of M = U_i^T U_{i-1} via an inner
// loop over d_s, subtracts the identity, squares, and reduces with
// __shfl_xor_sync.
//
// Cost: O(T * r * r * d_s) — for r=4, d_s=8 this is 16 * 8 = 128 ops/token,
// total 0.13 K ops/token, negligible vs SFA Tikhonov solve.
// ---------------------------------------------------------------------------
namespace {

__global__ void sfa_defect_frame_kernel(const float* __restrict__ U,
                                         int T, int d_s, int r,
                                         float* __restrict__ eps)
{
	int i = blockIdx.x;
	if (i >= T) return;
	if (i == 0)
	{
		if (threadIdx.x == 0) eps[i] = 0.0f;
		return;
	}

	const float* Ui  = U + (size_t)i * d_s * r;
	const float* Uim = U + (size_t)(i - 1) * d_s * r;

	const int RR = r * r;
	float acc = 0.0f;

	// Each thread handles a stride of (beta1, beta2) pairs (column-major
	// in the flat r*r index).
	for (int k = threadIdx.x; k < RR; k += blockDim.x)
	{
		int beta1 = k / r;
		int beta2 = k % r;
		float m = 0.0f;
		for (int a = 0; a < d_s; ++a)
		{
			float ui  = Ui [a * r + beta1];
			float uim = Uim[a * r + beta2];
			m += ui * uim;
		}
		float diff = m - (beta1 == beta2 ? 1.0f : 0.0f);
		acc += diff * diff;
	}

	// Warp reduce (block = 32 threads).
	unsigned mask = 0xFFFFFFFFu;
	#pragma unroll
	for (int offset = 16; offset > 0; offset >>= 1)
		acc += __shfl_xor_sync(mask, acc, offset);

	if (threadIdx.x == 0) eps[i] = sqrtf(acc);
}

}  // namespace

bool sfa_defect_frame_step1_fp32(const float* U,
                                  float*       eps,
                                  int T, int d_s, int r,
                                  cudaStream_t stream)
{
	if (T <= 0 || d_s <= 0 || r <= 0) return true;
	if (r > kMaxR)
	{
		fprintf(stderr, "[sfa-cuda] defect_frame_step1: r=%d exceeds kMaxR=%d\n",
		        r, kMaxR);
		return false;
	}
	cudaStream_t s_use = (stream != 0) ? stream : computeStream();
	const int block = 32;  // one warp
	sfa_defect_frame_kernel<<<T, block, 0, s_use>>>(U, T, d_s, r, eps);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
