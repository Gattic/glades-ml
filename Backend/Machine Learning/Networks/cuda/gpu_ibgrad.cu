// IBGRAD (Information-Bottleneck Gradient Subspace, paradigm shift #19)
// GPU primitives.  See gpu_ibgrad.h, research/PARADIGM_SHIFT_19_SELECTION.md.

#include "gpu_ibgrad.h"
#include "gpu_blas.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>
#include <cstdio>

namespace glades {
namespace gpu {

namespace {

// ------------------------------------------------------------------------
// Gaussian init via Box-Muller on splitmix64 counter.  Scale by 1/√N so
// that Pᵀ·P ≈ I_r in expectation by CLT.
// ------------------------------------------------------------------------
__global__ void k_ibgrad_init(float* __restrict__ P,
                              int n_elems, uint64_t seed, float scale)
{
	const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int i0 = 2 * pair_idx;
	if (i0 >= n_elems) return;

	auto sm = [&](int k) -> uint64_t {
		uint64_t x = seed ^ (uint64_t)((int64_t)(pair_idx * 2 + k) * 0x9E3779B97F4A7C15ULL);
		x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
		x ^= x >> 27; x *= 0x94D049BB133111EBULL;
		x ^= x >> 31;
		return x;
	};
	uint64_t x1 = sm(0);
	uint64_t x2 = sm(1);
	const uint32_t m1 = (uint32_t)(x1 >> 9) & 0x007fffffu;
	const uint32_t m2 = (uint32_t)(x2 >> 9) & 0x007fffffu;
	float u1 = (float)m1 / (float)(1u << 23);
	float u2 = (float)m2 / (float)(1u << 23);
	if (u1 < 1e-7f) u1 = 1e-7f;
	if (u2 < 1e-7f) u2 = 1e-7f;

	const float r = sqrtf(-2.0f * logf(u1));
	const float th = 6.2831853071795864769f * u2;
	const float z0 = r * cosf(th) * scale;
	const float z1 = r * sinf(th) * scale;

	P[i0] = z0;
	if (i0 + 1 < n_elems) P[i0 + 1] = z1;
}

// ------------------------------------------------------------------------
// Row-major [N × r]  →  column-major [N × r]  (same as row-major [r × N]
// read as column-major [N × r]).  We need this for cuSOLVER which expects
// column-major input.
// ------------------------------------------------------------------------
// 1D grid linearization — avoids the 65535 gridDim.y limit at N > 1M.
// Each thread handles one element via linear index idx = i*cols + j.
__global__ void k_transpose_rowmajor_to_colmajor(const float* __restrict__ in,
                                                 float* __restrict__ out,
                                                 int rows, int cols)
{
	const size_t n = (size_t)rows * cols;
	for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	     idx < n;
	     idx += (size_t)gridDim.x * blockDim.x)
	{
		const int i = (int)(idx / cols);
		const int j = (int)(idx % cols);
		// Row-major in:  in[i*cols + j]
		// Column-major out of shape [rows × cols]: out[i + j*rows]
		out[i + (size_t)j * rows] = in[(size_t)i * cols + j];
	}
}

__global__ void k_transpose_colmajor_to_rowmajor(const float* __restrict__ in,
                                                 float* __restrict__ out,
                                                 int rows, int cols)
{
	const size_t n = (size_t)rows * cols;
	for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	     idx < n;
	     idx += (size_t)gridDim.x * blockDim.x)
	{
		const int i = (int)(idx / cols);
		const int j = (int)(idx % cols);
		out[(size_t)i * cols + j] = in[i + (size_t)j * rows];
	}
}

// ------------------------------------------------------------------------
// Rank-1 outer product update: P[i, j] += eta · g[i] · y[j]
// One block per row i; threads cooperate on columns j.
// ------------------------------------------------------------------------
__global__ void k_ibgrad_rank1_update(float* __restrict__ P,
                                      const float* __restrict__ g,
                                      const float* __restrict__ y,
                                      int N, int r, float eta)
{
	const int i = blockIdx.x;
	if (i >= N) return;

	const float gi = eta * g[i];
	float* row = P + (size_t)i * r;
	for (int j = threadIdx.x; j < r; j += blockDim.x)
		row[j] += gi * y[j];
}

} // anonymous namespace

bool ibgrad_init_projection(float* P_out,
                            unsigned int N, unsigned int r,
                            uint64_t seed)
{
	if (P_out == nullptr) return false;
	if (N == 0u || r == 0u) return false;

	const int n_elems = (int)((size_t)N * r);
	const int n_pairs = (n_elems + 1) / 2;
	const int block = 256;
	const int grid  = (n_pairs + block - 1) / block;
	const float scale = 1.0f / sqrtf((float)N);
	k_ibgrad_init<<<grid, block, 0, computeStream()>>>(
	    P_out, n_elems, seed, scale);
	return cudaGetLastError() == cudaSuccess;
}

bool ibgrad_project(const float* P, const float* g,
                    unsigned int N, unsigned int r,
                    float* g_sub_out)
{
	if (P == nullptr || g == nullptr || g_sub_out == nullptr) return false;
	if (N == 0u || r == 0u) return false;

	// g_sub = Pᵀ · g.  P is [N × r] row-major.  Treat g as [N × 1].
	// sgemm_rowmajor_atb(M, N, K, ...) computes out [M × N] = Aᵀ[M × K] · B[K × N]
	// We want out [r × 1] = Pᵀ[r × N] · g[N × 1], so M=r, N=1, K=N_total.
	return sgemm_rowmajor_atb((unsigned int)r, 1u, (unsigned int)N, 1.0f,
	                          P, (unsigned int)r,
	                          g, 1u,
	                          0.0f,
	                          g_sub_out, 1u);
}

bool ibgrad_unproject(const float* P, const float* update_sub,
                      unsigned int N, unsigned int r,
                      float* update_full_out)
{
	if (P == nullptr || update_sub == nullptr || update_full_out == nullptr) return false;
	if (N == 0u || r == 0u) return false;

	// update_full = P · update_sub.  P [N × r] · update_sub [r × 1] = [N × 1].
	return sgemm_rowmajor((unsigned int)N, 1u, (unsigned int)r, 1.0f,
	                      P, (unsigned int)r,
	                      update_sub, 1u,
	                      0.0f,
	                      update_full_out, 1u);
}

bool ibgrad_oja_rank1_update(float* P_inout,
                             const float* g, const float* y,
                             unsigned int N, unsigned int r,
                             float eta)
{
	if (P_inout == nullptr || g == nullptr || y == nullptr) return false;
	if (N == 0u || r == 0u) return false;

	const int block = (r >= 256) ? 256 : ((r >= 64) ? 64 : 32);
	k_ibgrad_rank1_update<<<(int)N, block, 0, computeStream()>>>(
	    P_inout, g, y, (int)N, (int)r, eta);
	return cudaGetLastError() == cudaSuccess;
}

// ------------------------------------------------------------------------
// Compute ‖g‖² via block-reduction kernel.  Result is written to a
// single-float device scalar; caller downloads if needed.
// ------------------------------------------------------------------------
namespace {
__global__ void k_ibgrad_norm_sq(const float* __restrict__ g,
                                 int N, float* __restrict__ out)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int block = blockDim.x;
	const int stride = gridDim.x * block;

	float local = 0.0f;
	for (int i = bid * block + tid; i < N; i += stride) {
		float v = g[i];
		local += v * v;
	}

	__shared__ float shm[32];
	const int lane = tid & 31;
	const int warp = tid >> 5;

	for (int off = 16; off > 0; off >>= 1)
		local += __shfl_down_sync(0xffffffffu, local, off);
	if (lane == 0) shm[warp] = local;
	__syncthreads();

	if (warp == 0) {
		const int nwarps = (block + 31) >> 5;
		float v = (tid < nwarps) ? shm[tid] : 0.0f;
		for (int off = 16; off > 0; off >>= 1)
			v += __shfl_down_sync(0xffffffffu, v, off);
		if (tid == 0) atomicAdd(out, v);
	}
}

// Write column 0 of P (row-major [N × r]) with g[i] / sqrt(g_norm_sq[0]).
__global__ void k_ibgrad_write_col0(float* __restrict__ P,
                                    const float* __restrict__ g,
                                    const float* __restrict__ g_norm_sq,
                                    int N, int r)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const float norm = sqrtf(g_norm_sq[0]) + 1e-8f;
	// P[i, 0] = g[i] / norm
	P[(size_t)i * r + 0] = g[i] / norm;
}
} // anonymous

// ------------------------------------------------------------------------
// QR reorthogonalization via cuSOLVER (one-shot handle, cached scratch).
// ------------------------------------------------------------------------
namespace {
static cusolverDnHandle_t g_ibgradSolver = nullptr;
static bool g_ibgradSolverReady = false;
static float* g_ibgradQrWork = nullptr;
static size_t g_ibgradQrWorkCap = 0;
static int*   g_ibgradInfo = nullptr;
static float* g_ibgradColScratch = nullptr;
static size_t g_ibgradColScratchCap = 0;
static float* g_ibgradTau = nullptr;
static size_t g_ibgradTauCap = 0;

static bool ibgrad_solver_init()
{
	if (!g_ibgradSolverReady) {
		if (cusolverDnCreate(&g_ibgradSolver) != CUSOLVER_STATUS_SUCCESS) return false;
		g_ibgradSolverReady = true;
	}
	cusolverDnSetStream(g_ibgradSolver, computeStream());
	if (g_ibgradInfo == nullptr) {
		if (cudaMalloc(&g_ibgradInfo, sizeof(int)) != cudaSuccess) return false;
	}
	return true;
}
} // anonymous

bool ibgrad_qr_reorthogonalize(float* P_inout, unsigned int N, unsigned int r)
{
	if (P_inout == nullptr) return false;
	if (N == 0u || r == 0u || r > N) return false;
	if (!ibgrad_solver_init()) return false;

	const size_t sz = (size_t)N * r;

	// Scratch for column-major P.
	if (sz > g_ibgradColScratchCap) {
		if (g_ibgradColScratch) cudaFree(g_ibgradColScratch);
		if (cudaMalloc(&g_ibgradColScratch, sz * sizeof(float)) != cudaSuccess) return false;
		g_ibgradColScratchCap = sz;
	}
	if ((size_t)r > g_ibgradTauCap) {
		if (g_ibgradTau) cudaFree(g_ibgradTau);
		if (cudaMalloc(&g_ibgradTau, r * sizeof(float)) != cudaSuccess) return false;
		g_ibgradTauCap = r;
	}

	// 1. Row-major → column-major (1D grid linearization, handles N > 1M).
	{
		const int block = 256;
		const size_t n_elems = (size_t)N * r;
		// Clamp grid to 65535 blocks; kernel loops to cover N·r elements.
		size_t grid_sz = (n_elems + block - 1) / (size_t)block;
		if (grid_sz > 65535u) grid_sz = 65535u;
		k_transpose_rowmajor_to_colmajor<<<(int)grid_sz, block, 0, computeStream()>>>(
		    P_inout, g_ibgradColScratch, (int)N, (int)r);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// 2. Query + allocate workspace.
	int lwork_geqrf = 0, lwork_orgqr = 0;
	if (cusolverDnSgeqrf_bufferSize(g_ibgradSolver, N, r,
	                                g_ibgradColScratch, N, &lwork_geqrf)
	    != CUSOLVER_STATUS_SUCCESS) return false;
	if (cusolverDnSorgqr_bufferSize(g_ibgradSolver, N, r, r,
	                                g_ibgradColScratch, N, g_ibgradTau, &lwork_orgqr)
	    != CUSOLVER_STATUS_SUCCESS) return false;
	const int lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
	if ((size_t)lwork > g_ibgradQrWorkCap) {
		if (g_ibgradQrWork) cudaFree(g_ibgradQrWork);
		if (cudaMalloc(&g_ibgradQrWork, lwork * sizeof(float)) != cudaSuccess) return false;
		g_ibgradQrWorkCap = lwork;
	}

	// 3. sgeqrf.
	if (cusolverDnSgeqrf(g_ibgradSolver, N, r, g_ibgradColScratch, N, g_ibgradTau,
	                     g_ibgradQrWork, lwork, g_ibgradInfo)
	    != CUSOLVER_STATUS_SUCCESS) return false;
	int host_info = 0;
	cudaMemcpy(&host_info, g_ibgradInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (host_info != 0) {
		fprintf(stderr, "[ibgrad-qr] sgeqrf info=%d N=%u r=%u\n", host_info, N, r);
		return false;
	}

	// 4. Form Q explicitly via sorgqr.
	if (cusolverDnSorgqr(g_ibgradSolver, N, r, r, g_ibgradColScratch, N, g_ibgradTau,
	                     g_ibgradQrWork, lwork, g_ibgradInfo)
	    != CUSOLVER_STATUS_SUCCESS) return false;
	cudaMemcpy(&host_info, g_ibgradInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (host_info != 0) {
		fprintf(stderr, "[ibgrad-qr] sorgqr info=%d N=%u r=%u\n", host_info, N, r);
		return false;
	}

	// 5. Column-major → row-major back into P (1D grid linearization).
	{
		const int block = 256;
		const size_t n_elems = (size_t)N * r;
		size_t grid_sz = (n_elems + block - 1) / (size_t)block;
		if (grid_sz > 65535u) grid_sz = 65535u;
		k_transpose_colmajor_to_rowmajor<<<(int)grid_sz, block, 0, computeStream()>>>(
		    g_ibgradColScratch, P_inout, (int)N, (int)r);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	return true;
}

bool ibgrad_refresh_first_column(float* P_inout, const float* g,
                                 unsigned int N, unsigned int r)
{
	if (P_inout == nullptr || g == nullptr) return false;
	if (N == 0u || r == 0u) return false;

	// Compute ‖g‖² into a temp device scalar.
	static thread_local float* d_scalar = nullptr;
	if (d_scalar == nullptr) {
		if (cudaMalloc(&d_scalar, sizeof(float)) != cudaSuccess) return false;
	}
	if (cudaMemsetAsync(d_scalar, 0, sizeof(float), computeStream()) != cudaSuccess) return false;

	{
		const int block = 256;
		const int grid = (((int)N + block - 1) / block);
		const int max_grid = 64;
		const int g_use = (grid < max_grid) ? grid : max_grid;
		k_ibgrad_norm_sq<<<g_use, block, 0, computeStream()>>>(
		    g, (int)N, d_scalar);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// Write column 0 of P with normalized g.
	{
		const int block = 256;
		const int grid = ((int)N + block - 1) / block;
		k_ibgrad_write_col0<<<grid, block, 0, computeStream()>>>(
		    P_inout, g, d_scalar, (int)N, (int)r);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
