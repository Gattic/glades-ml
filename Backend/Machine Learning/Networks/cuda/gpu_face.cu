// FACE (Frequency-Aware Column-normalized Embedding optimizer) GPU primitives.
// See gpu_face.h and research/PARADIGM_SHIFT_28_DESIGN.md.

#include "gpu_face.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace glades {
namespace gpu {

namespace {

// Row reduction: one block per row, each block computes zn[i] = Σ_j g[i,j]²
// and writes 1 to an "active" scratch slot if zn[i] > 0 (any nonzero entry).
// The per-block scratch is a single float per row containing 0 or 1.
__global__ void k_face_row_stats(const float* __restrict__ g,
                                 unsigned int V, unsigned int m,
                                 float* __restrict__ zn_out,
                                 float* __restrict__ active_flag_out)
{
	extern __shared__ float smem[];
	const unsigned int row = blockIdx.x;
	const unsigned int tid = threadIdx.x;
	if (row >= V) return;

	float partial = 0.0f;
	const float* rp = g + (size_t)row * m;
	for (unsigned int j = tid; j < m; j += blockDim.x) {
		const float v = rp[j];
		partial += v * v;
	}
	// Warp reduce.
	for (int off = 16; off > 0; off >>= 1)
		partial += __shfl_xor_sync(0xffffffffu, partial, off);
	const int lane = tid & 31;
	const int warp = tid >> 5;
	const int nw   = (blockDim.x + 31) >> 5;
	if (lane == 0) smem[warp] = partial;
	__syncthreads();
	if (warp == 0) {
		float v = (tid < nw) ? smem[tid] : 0.0f;
		for (int off = 16; off > 0; off >>= 1)
			v += __shfl_xor_sync(0xffffffffu, v, off);
		if (tid == 0) {
			zn_out[row] = v;
			active_flag_out[row] = (v > 0.0f) ? 1.0f : 0.0f;
		}
	}
}

// Column reduction: one block per column, dn_raw[j] = Σ_i g[i,j]²
// (identical shape to the existing MFIO k_mfio_col_sqsum).
__global__ void k_face_col_sqsum(const float* __restrict__ g,
                                 unsigned int V, unsigned int m,
                                 float* __restrict__ dn_out)
{
	extern __shared__ float smem[];
	const unsigned int col = blockIdx.x;
	const unsigned int tid = threadIdx.x;
	if (col >= m) return;

	float partial = 0.0f;
	for (unsigned int i = tid; i < V; i += blockDim.x) {
		const float v = g[(size_t)i * m + col];
		partial += v * v;
	}
	for (int off = 16; off > 0; off >>= 1)
		partial += __shfl_xor_sync(0xffffffffu, partial, off);
	const int lane = tid & 31;
	const int warp = tid >> 5;
	const int nw   = (blockDim.x + 31) >> 5;
	if (lane == 0) smem[warp] = partial;
	__syncthreads();
	if (warp == 0) {
		float v = (tid < nw) ? smem[tid] : 0.0f;
		for (int off = 16; off > 0; off >>= 1)
			v += __shfl_xor_sync(0xffffffffu, v, off);
		if (tid == 0) dn_out[col] = v;
	}
}

// Single-block reduce: sums a vector to a scalar.  Used to reduce zn[V] →
// gF (Frobenius² = sum of zn) and active_flag[V] → q (count of active rows).
__global__ void k_face_sum_vec(const float* __restrict__ in,
                               unsigned int n,
                               float* __restrict__ out)
{
	extern __shared__ float smem[];
	const int tid = threadIdx.x;
	const int bs  = blockDim.x;
	float acc = 0.0f;
	for (unsigned int i = tid; i < n; i += bs)
		acc += in[i];
	for (int off = 16; off > 0; off >>= 1)
		acc += __shfl_xor_sync(0xffffffffu, acc, off);
	const int lane = tid & 31;
	const int warp = tid >> 5;
	const int nw   = (bs + 31) >> 5;
	if (lane == 0) smem[warp] = acc;
	__syncthreads();
	if (warp == 0) {
		float v = (tid < nw) ? smem[tid] : 0.0f;
		for (int off = 16; off > 0; off >>= 1)
			v += __shfl_xor_sync(0xffffffffu, v, off);
		if (tid == 0) *out = v;
	}
}

} // anonymous namespace

// Preconditioned update kernel: one thread per (i, j) pair.  Because g is
// zero on inactive rows, the multiplicative update naturally zeros there.
__global__ void k_face_apply_update(float* __restrict__ theta,
                                    const float* __restrict__ g,
                                    const float* __restrict__ zn_bar,
                                    const float* __restrict__ dn_bar,
                                    const float* __restrict__ q_hat,
                                    const float* __restrict__ gF_hat,
                                    unsigned int V, unsigned int m,
                                    float lr, float eps_sq)
{
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= V || j >= m) return;

	// Denominator = zn̄[i] · dn̄[j] / (q̂ · gF̄).
	const float zn = zn_bar[i];
	const float dn = dn_bar[j];
	const float q  = *q_hat;
	const float gF = *gF_hat;
	const float den = (q * gF) + 1e-20f;        // avoid /0 during warmup
	const float s   = (zn * dn) / den;
	const float sigma = 1.0f / (sqrtf(s + eps_sq));
	const size_t off = (size_t)i * m + j;
	theta[off] -= lr * sigma * g[off];
}

bool face_apply_preconditioned_update(float* theta,
                                      const float* g,
                                      const float* zn_bar,
                                      const float* dn_bar,
                                      const float* q_hat,
                                      const float* gF_hat,
                                      unsigned int V, unsigned int m,
                                      float lr, float eps)
{
	if (theta == nullptr || g == nullptr) return false;
	if (zn_bar == nullptr || dn_bar == nullptr) return false;
	if (q_hat == nullptr || gF_hat == nullptr) return false;
	if (V == 0u || m == 0u) return false;

	dim3 block(32, 8);
	dim3 grid((m + block.x - 1u) / block.x,
	          (V + block.y - 1u) / block.y);
	const float eps_sq = eps * eps;
	k_face_apply_update<<<grid, block, 0, computeStream()>>>(
	    theta, g, zn_bar, dn_bar, q_hat, gF_hat, V, m, lr, eps_sq);
	return cudaGetLastError() == cudaSuccess;
}

bool face_compute_sparse_stats(const float* g,
                               unsigned int V, unsigned int m,
                               float* zn_out,
                               float* dn_raw_out,
                               float* q_out,
                               float* gF_out)
{
	if (g == nullptr || zn_out == nullptr || dn_raw_out == nullptr ||
	    q_out == nullptr || gF_out == nullptr)
		return false;
	if (V == 0u || m == 0u) return false;

	const int block = 256;
	const int nwarps = (block + 31) >> 5;
	const size_t smemBytes = nwarps * sizeof(float);

	// Allocate scratch for the V-dim active-flag vector (one float per row).
	float* d_active_flags = nullptr;
	cudaMalloc(&d_active_flags, V * sizeof(float));
	if (d_active_flags == nullptr) return false;

	// Row reduction: zn[V] + active_flag[V].
	k_face_row_stats<<<V, block, smemBytes, computeStream()>>>(
	    g, V, m, zn_out, d_active_flags);
	if (cudaGetLastError() != cudaSuccess) {
		cudaFree(d_active_flags); return false;
	}

	// Column reduction: dn_raw[m].
	k_face_col_sqsum<<<m, block, smemBytes, computeStream()>>>(
	    g, V, m, dn_raw_out);
	if (cudaGetLastError() != cudaSuccess) {
		cudaFree(d_active_flags); return false;
	}

	// gF = Σ_i zn[i] (also = Σ_{i,j} g[i,j]² = ‖g‖_F²).
	k_face_sum_vec<<<1, block, smemBytes, computeStream()>>>(
	    zn_out, V, gF_out);

	// q = Σ_i active_flag[i].
	k_face_sum_vec<<<1, block, smemBytes, computeStream()>>>(
	    d_active_flags, V, q_out);

	cudaFree(d_active_flags);
	return cudaGetLastError() == cudaSuccess;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
