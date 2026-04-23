// MFIO (Moment-Free Implicit Optimizer) GPU primitives.
// See gpu_mfio.h and research/PARADIGM_SHIFT_11_DESIGN.md.

#include "gpu_mfio.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace glades {
namespace gpu {

namespace {

// Single-block reduction kernel that computes
//     ŝ = (1/T) · Σ_t ‖z[t]‖² · ‖δ[t]‖²
//     σ = 1 / (ŝ · beta + eps)
// in one pass.  Threads cooperatively reduce the per-row products,
// then thread 0 writes the final σ.
//
// Launch: <<<1, block, smemBytes>>>.  block = 256 typical.
__global__ void k_mfio_compute_sigma(const float* __restrict__ z,
                                     const float* __restrict__ delta,
                                     unsigned int T, unsigned int d_in, unsigned int d_out,
                                     float beta_schedule, float eps,
                                     float* __restrict__ sigma_out)
{
	extern __shared__ float smem[];
	const int tid = threadIdx.x;
	const int bs  = blockDim.x;

	// Accumulate ‖z[t]‖² · ‖δ[t]‖² across all t, with threads striding.
	float partial = 0.0f;
	for (unsigned int t = tid; t < T; t += bs)
	{
		float norm_z2 = 0.0f;
		const float* zr = z + (size_t)t * d_in;
		for (unsigned int k = 0; k < d_in; ++k)
		{
			const float v = zr[k];
			norm_z2 += v * v;
		}
		float norm_d2 = 0.0f;
		const float* dr = delta + (size_t)t * d_out;
		for (unsigned int k = 0; k < d_out; ++k)
		{
			const float v = dr[k];
			norm_d2 += v * v;
		}
		partial += norm_z2 * norm_d2;
	}

	// Warp reduce via __shfl_xor_sync, then cross-warp via shared.
	for (int off = 16; off > 0; off >>= 1)
		partial += __shfl_xor_sync(0xffffffffu, partial, off);
	const int lane = tid & 31;
	const int warp = tid >> 5;
	if (lane == 0) smem[warp] = partial;
	__syncthreads();

	if (warp == 0)
	{
		const int nwarps = (bs + 31) >> 5;
		float v = (tid < nwarps) ? smem[tid] : 0.0f;
		for (int off = 16; off > 0; off >>= 1)
			v += __shfl_xor_sync(0xffffffffu, v, off);
		if (tid == 0)
		{
			const float s_hat = v / (float)T;
			// sqrt-preconditioner to match Adam's 1/√v effective scaling;
			// 1/ŝ alone gives 1/|g|² which overshoots catastrophically as
			// gradients shrink near the optimum.
			const float sigma = 1.0f / (sqrtf(s_hat * beta_schedule) + eps);
			*sigma_out = sigma;
		}
	}
}

// Parameter update:  θ[i] = θ[i] · (1 − η · wd) − η · σ · g[i]
__global__ void k_mfio_update(float* __restrict__ theta,
                              const float* __restrict__ g,
                              const float* __restrict__ sigma,
                              float lr, float wd,
                              int n_params)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_params) return;
	const float s = *sigma;
	const float wd_scale = 1.0f - lr * wd;
	theta[i] = wd_scale * theta[i] - lr * s * g[i];
}

} // anonymous namespace

bool mfio_compute_sigma(const float* z, const float* delta,
                        unsigned int T, unsigned int d_in, unsigned int d_out,
                        float beta_schedule,
                        float eps,
                        float* sigma_out)
{
	if (z == nullptr || delta == nullptr || sigma_out == nullptr) return false;
	if (T == 0u || d_in == 0u || d_out == 0u) return false;

	const int block = 256;
	const int nwarps = (block + 31) >> 5;
	const size_t smemBytes = nwarps * sizeof(float);
	k_mfio_compute_sigma<<<1, block, smemBytes, computeStream()>>>(
	    z, delta, T, d_in, d_out, beta_schedule, eps, sigma_out);
	return cudaGetLastError() == cudaSuccess;
}

bool mfio_update(float* theta, const float* g,
                 const float* sigma,
                 float lr, float wd,
                 int n_params)
{
	if (theta == nullptr || g == nullptr || sigma == nullptr) return false;
	if (n_params <= 0) return false;

	const int block = 256;
	const int grid  = (n_params + block - 1) / block;
	k_mfio_update<<<grid, block, 0, computeStream()>>>(
	    theta, g, sigma, lr, wd, n_params);
	return cudaGetLastError() == cudaSuccess;
}

// ========================================================================
// Column-sum-of-squares reductions for z [T × d_in] and δ [T × d_out].
// One block per column, threads stride across T.
// ========================================================================
namespace {

__global__ void k_mfio_col_sqsum(const float* __restrict__ X,
                                 unsigned int T, unsigned int d,
                                 float* __restrict__ out)
{
	extern __shared__ float smem[];
	const unsigned int col = blockIdx.x;
	const unsigned int tid = threadIdx.x;
	if (col >= d) return;

	float partial = 0.0f;
	for (unsigned int t = tid; t < T; t += blockDim.x)
	{
		const float v = X[(size_t)t * d + col];
		partial += v * v;
	}
	// warp reduce
	for (int off = 16; off > 0; off >>= 1)
		partial += __shfl_xor_sync(0xffffffffu, partial, off);
	const int lane = tid & 31;
	const int warp = tid >> 5;
	if (lane == 0) smem[warp] = partial;
	__syncthreads();
	if (warp == 0)
	{
		const int nwarps = (blockDim.x + 31) >> 5;
		float v = (tid < nwarps) ? smem[tid] : 0.0f;
		for (int off = 16; off > 0; off >>= 1)
			v += __shfl_xor_sync(0xffffffffu, v, off);
		if (tid == 0) out[col] = v;
	}
}

// Per-weight MFIO update using row/col norms.
__global__ void k_mfio_update_rowcol(float* __restrict__ theta,
                                     const float* __restrict__ g,
                                     const float* __restrict__ zn,
                                     const float* __restrict__ dn,
                                     unsigned int d_in, unsigned int d_out,
                                     float T_normalizer,
                                     float lr, float beta, float eps, float wd)
{
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= d_in || j >= d_out) return;
	const float zi = zn[i];
	const float dj = dn[j];
	const float s  = zi * dj * beta / T_normalizer;
	const float sigma = 1.0f / (sqrtf(s) + eps);
	const size_t off = (size_t)i * d_out + j;
	const float wd_scale = 1.0f - lr * wd;
	theta[off] = wd_scale * theta[off] - lr * sigma * g[off];
}

} // anonymous namespace

bool mfio_compute_rowcol_norms(const float* z, const float* delta,
                               unsigned int T, unsigned int d_in, unsigned int d_out,
                               float* zn_out, float* dn_out)
{
	if (z == nullptr || delta == nullptr || zn_out == nullptr || dn_out == nullptr)
		return false;
	if (T == 0u || d_in == 0u || d_out == 0u) return false;

	const int block = 256;
	const int nwarps = (block + 31) >> 5;
	const size_t smemBytes = nwarps * sizeof(float);
	k_mfio_col_sqsum<<<d_in, block, smemBytes, computeStream()>>>(
	    z, T, d_in, zn_out);
	if (cudaGetLastError() != cudaSuccess) return false;
	k_mfio_col_sqsum<<<d_out, block, smemBytes, computeStream()>>>(
	    delta, T, d_out, dn_out);
	return cudaGetLastError() == cudaSuccess;
}

bool mfio_update_rowcol(float* theta, const float* g,
                        const float* zn, const float* dn,
                        unsigned int d_in, unsigned int d_out,
                        float T_normalizer,
                        float lr, float beta, float eps, float wd)
{
	if (theta == nullptr || g == nullptr || zn == nullptr || dn == nullptr)
		return false;
	if (d_in == 0u || d_out == 0u) return false;

	dim3 block(32, 8);
	dim3 grid((d_out + block.x - 1u) / block.x,
	          (d_in  + block.y - 1u) / block.y);
	k_mfio_update_rowcol<<<grid, block, 0, computeStream()>>>(
	    theta, g, zn, dn, d_in, d_out,
	    T_normalizer, lr, beta, eps, wd);
	return cudaGetLastError() == cudaSuccess;
}

// ========================================================================
// Gradient-space row/col norm reduction.  g is [d_in × d_out] row-major.
//
//   zn[i] = Σ_j g[i,j]²        (row reduction, one block per row)
//   dn[j] = Σ_i g[i,j]²        (col reduction, one block per col — reuses
//                               k_mfio_col_sqsum treating (d_in, d_out) as
//                               its (T, d) args).
// ========================================================================
namespace {

__global__ void k_mfio_row_sqsum(const float* __restrict__ X,
                                 unsigned int d_in, unsigned int d_out,
                                 float* __restrict__ out)
{
	extern __shared__ float smem[];
	const unsigned int row = blockIdx.x;
	const unsigned int tid = threadIdx.x;
	if (row >= d_in) return;

	float partial = 0.0f;
	const float* rp = X + (size_t)row * d_out;
	for (unsigned int j = tid; j < d_out; j += blockDim.x)
	{
		const float v = rp[j];
		partial += v * v;
	}
	// warp reduce
	for (int off = 16; off > 0; off >>= 1)
		partial += __shfl_xor_sync(0xffffffffu, partial, off);
	const int lane = tid & 31;
	const int warp = tid >> 5;
	if (lane == 0) smem[warp] = partial;
	__syncthreads();
	if (warp == 0)
	{
		const int nwarps = (blockDim.x + 31) >> 5;
		float v = (tid < nwarps) ? smem[tid] : 0.0f;
		for (int off = 16; off > 0; off >>= 1)
			v += __shfl_xor_sync(0xffffffffu, v, off);
		if (tid == 0) out[row] = v;
	}
}

} // anonymous namespace

bool mfio_compute_rowcol_norms_from_grad(const float* g,
                                         unsigned int d_in, unsigned int d_out,
                                         float* zn_out, float* dn_out)
{
	if (g == nullptr || zn_out == nullptr || dn_out == nullptr) return false;
	if (d_in == 0u || d_out == 0u) return false;

	const int block = 256;
	const int nwarps = (block + 31) >> 5;
	const size_t smemBytes = nwarps * sizeof(float);
	// zn[i] = Σ_j g[i,j]²  — one block per row.
	k_mfio_row_sqsum<<<d_in, block, smemBytes, computeStream()>>>(
	    g, d_in, d_out, zn_out);
	if (cudaGetLastError() != cudaSuccess) return false;
	// dn[j] = Σ_i g[i,j]²  — one block per col; reuse k_mfio_col_sqsum
	// with T=d_in, d=d_out (it strides rows and reduces per-column).
	k_mfio_col_sqsum<<<d_out, block, smemBytes, computeStream()>>>(
	    g, d_in, d_out, dn_out);
	return cudaGetLastError() == cudaSuccess;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
