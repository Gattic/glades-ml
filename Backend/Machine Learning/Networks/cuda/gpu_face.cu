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

// EMA update kernels.  One thread per element; conditional update for rows
// (skip inactive), unconditional for cols and scalars.
__global__ void k_face_ema_rows(float* __restrict__ zn_bar,
                                const float* __restrict__ zn_new,
                                unsigned int V, float beta_row)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= V) return;
	const float zn_i = zn_new[i];
	if (zn_i > 0.0f) {
		zn_bar[i] = beta_row * zn_bar[i] + (1.0f - beta_row) * zn_i;
	}
	// else: row inactive this step; preserve prior zn_bar[i].
}

__global__ void k_face_ema_cols(float* __restrict__ dn_bar,
                                const float* __restrict__ dn_raw,
                                const float* __restrict__ q,
                                unsigned int m, float beta_col)
{
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= m) return;
	(void)q;
	// FACE Phase 4 dimensional correction (vs design doc §3): store dn_raw
	// directly in the EMA (no frequency-debias division).  Combined with the
	// q-less apply kernel below, this yields σ = 1/√(zn·dn_raw/gF + ε²) —
	// scales as 1/σ_g like dense MFIO, not V/σ_g as the q-scaled formula.
	dn_bar[j] = beta_col * dn_bar[j] + (1.0f - beta_col) * dn_raw[j];
}

__global__ void k_face_ema_scalars(float* __restrict__ q_hat,
                                   float* __restrict__ gF_hat,
                                   const float* __restrict__ q,
                                   const float* __restrict__ gF,
                                   float beta_col)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;
	*q_hat  = beta_col * (*q_hat)  + (1.0f - beta_col) * (*q);
	*gF_hat = beta_col * (*gF_hat) + (1.0f - beta_col) * (*gF);
}

bool face_update_emas(float* zn_bar, float* dn_bar,
                      float* q_hat, float* gF_hat,
                      const float* zn_new, const float* dn_raw,
                      const float* q, const float* gF,
                      unsigned int V, unsigned int m,
                      float beta_row, float beta_col)
{
	if (zn_bar == nullptr || dn_bar == nullptr || q_hat == nullptr || gF_hat == nullptr)
		return false;
	if (zn_new == nullptr || dn_raw == nullptr || q == nullptr || gF == nullptr)
		return false;
	if (V == 0u || m == 0u) return false;

	const int block = 256;
	const int grid_V = (int)((V + (unsigned)block - 1u) / (unsigned)block);
	const int grid_m = (int)((m + (unsigned)block - 1u) / (unsigned)block);

	k_face_ema_rows<<<grid_V, block, 0, computeStream()>>>(
	    zn_bar, zn_new, V, beta_row);
	k_face_ema_cols<<<grid_m, block, 0, computeStream()>>>(
	    dn_bar, dn_raw, q, m, beta_col);
	k_face_ema_scalars<<<1, 1, 0, computeStream()>>>(
	    q_hat, gF_hat, q, gF, beta_col);
	return cudaGetLastError() == cudaSuccess;
}

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

	// FACE preconditioner (dimensionally-correct form — see gpu_face.cu
	// comment on k_face_ema_cols).  s = zn·dn_raw / gF, giving σ ~ 1/σ_g
	// like dense MFIO.  q_hat is unused in this form (tracked for diagnostics).
	const float zn = zn_bar[i];
	const float dn = dn_bar[j];
	const float gF = *gF_hat;
	(void)q_hat;
	const float den = gF + 1e-20f;
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
