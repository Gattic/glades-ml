// IBGRAD (Information-Bottleneck Gradient Subspace, paradigm shift #19)
// GPU primitives.  See gpu_ibgrad.h, research/PARADIGM_SHIFT_19_SELECTION.md.

#include "gpu_ibgrad.h"
#include "gpu_blas.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
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

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
