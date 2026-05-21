// KV-FACE Gate-0 probe — GPU primitives.
// See gpu_kvface_probe.h and research/PARADIGM_SHIFT_36_DESIGN.md.

#include "gpu_kvface_probe.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

namespace glades {
namespace gpu {

namespace {

// ========================================================================
// k_kvface_popularity: compute per-head per-position popularity
//   p[h, t] = (1/T) * Σ_q P[h, q, t]
// Grid: (T, nHeads). Block: 256 threads.
// Each block reduces over the Q dimension for one (h, t) pair.
// ========================================================================
__global__ void k_kvface_popularity(const float* __restrict__ P,
                                    float* __restrict__ popularity,
                                    unsigned int nHeads,
                                    unsigned int T)
{
	extern __shared__ float smem[];
	const unsigned int t = blockIdx.x;
	const unsigned int h = blockIdx.y;
	const unsigned int tid = threadIdx.x;
	const unsigned int bs = blockDim.x;
	if (h >= nHeads || t >= T) return;

	const float* Ph = P + (size_t)h * T * T;
	const float inv_T = 1.0f / (float)T;

	// Each thread accumulates over strided Q indices.
	float acc = 0.0f;
	for (unsigned int q = tid; q < T; q += bs)
		acc += Ph[(size_t)q * T + t];

	smem[tid] = acc;
	__syncthreads();
	for (unsigned int off = bs >> 1; off > 0; off >>= 1) {
		if (tid < off) smem[tid] += smem[tid + off];
		__syncthreads();
	}
	if (tid == 0)
		popularity[(size_t)h * T + t] = smem[0] * inv_T;
}

// ========================================================================
// k_kvface_gini: per-head Gini coefficient via pairwise-difference formula.
//   G = (Σ_i Σ_j |x_i - x_j|) / (2 * T * Σ_i x_i)
// Grid: (nHeads). Block: 256 threads.
// Each block computes one Gini value.
// Uses shared memory to cache the T values (T ≤ 4096).
// Falls back to global loads when T too large.
// ========================================================================
__global__ void k_kvface_gini(const float* __restrict__ popularity,
                              float* __restrict__ gini_per_head,
                              unsigned int nHeads,
                              unsigned int T)
{
	extern __shared__ float sdata[];  // [T] values + [blockDim] reductions
	const unsigned int h = blockIdx.x;
	const unsigned int tid = threadIdx.x;
	const unsigned int bs = blockDim.x;
	if (h >= nHeads) return;

	float* svals = sdata;            // size T
	float* sred  = sdata + T;        // size bs

	const float* ph = popularity + (size_t)h * T;

	// Stage 1: load values to shared memory + compute Σ x_i (sum)
	float my_sum = 0.0f;
	for (unsigned int i = tid; i < T; i += bs) {
		const float v = ph[i];
		svals[i] = v;
		my_sum += v;
	}
	sred[tid] = my_sum;
	__syncthreads();
	for (unsigned int off = bs >> 1; off > 0; off >>= 1) {
		if (tid < off) sred[tid] += sred[tid + off];
		__syncthreads();
	}
	const float total_sum = sred[0];
	__syncthreads();

	// Stage 2: compute Σ_i Σ_j |x_i - x_j| via strided pairs.
	// Each thread sums a subset of (i, j) pairs with i < j.
	float my_pair = 0.0f;
	const unsigned int total_pairs = T * (T - 1u) / 2u;
	for (unsigned int k = tid; k < total_pairs; k += bs) {
		// Map linear k to (i, j) with i < j.
		// Closed-form inversion: i = floor((sqrt(1 + 8*k) - 1) / 2), j = k - i*(i+1)/2 + i + 1 — but this
		// only holds for a column-major triangular enumeration. Use the standard row-major enumeration:
		//   Given T, pairs are (0,1),(0,2),...,(0,T-1),(1,2),(1,3),...,(T-2,T-1).
		// Row i has (T - 1 - i) pairs; find i such that cumulative count ≤ k.
		unsigned int cum = 0;
		unsigned int i = 0;
		for (; i < T - 1; ++i) {
			const unsigned int row_count = T - 1u - i;
			if (cum + row_count > k) break;
			cum += row_count;
		}
		const unsigned int j = i + 1u + (k - cum);
		const float diff = svals[i] - svals[j];
		my_pair += (diff >= 0.0f) ? diff : -diff;
	}

	sred[tid] = my_pair;
	__syncthreads();
	for (unsigned int off = bs >> 1; off > 0; off >>= 1) {
		if (tid < off) sred[tid] += sred[tid + off];
		__syncthreads();
	}

	if (tid == 0) {
		// Σ_i Σ_j |x_i - x_j| = 2 * (Σ_{i<j} |x_i - x_j|)
		const float total_abs_diff = 2.0f * sred[0];
		const float denom = 2.0f * (float)T * total_sum;
		gini_per_head[h] = (denom > 1e-20f) ? (total_abs_diff / denom) : 0.0f;
	}
}

// ========================================================================
// k_kvface_reduce_stats: compute mean, min, max of a [nHeads]-length
// Gini array. Single block, block-reductions.
// ========================================================================
__global__ void k_kvface_reduce_stats(const float* __restrict__ gini_per_head,
                                      float* __restrict__ stats3,
                                      unsigned int nHeads)
{
	extern __shared__ float s_r[];  // [blockDim * 3]
	const unsigned int tid = threadIdx.x;
	const unsigned int bs = blockDim.x;
	float my_sum = 0.0f;
	float my_min = INFINITY;
	float my_max = -INFINITY;
	for (unsigned int h = tid; h < nHeads; h += bs) {
		const float v = gini_per_head[h];
		my_sum += v;
		if (v < my_min) my_min = v;
		if (v > my_max) my_max = v;
	}
	s_r[tid]             = my_sum;
	s_r[tid + bs]        = my_min;
	s_r[tid + 2 * bs]    = my_max;
	__syncthreads();
	for (unsigned int off = bs >> 1; off > 0; off >>= 1) {
		if (tid < off) {
			s_r[tid]          += s_r[tid + off];
			const float a      = s_r[tid + bs];
			const float b      = s_r[tid + bs + off];
			s_r[tid + bs]      = (a < b) ? a : b;
			const float c      = s_r[tid + 2 * bs];
			const float d      = s_r[tid + 2 * bs + off];
			s_r[tid + 2 * bs]  = (c > d) ? c : d;
		}
		__syncthreads();
	}
	if (tid == 0) {
		stats3[0] = s_r[0] / (float)nHeads;
		stats3[1] = s_r[bs];
		stats3[2] = s_r[2 * bs];
	}
}

} // namespace (anonymous)

bool kvface_probe_compute_popularity(const float* P,
                                     float* popularity,
                                     int nHeads,
                                     int T)
{
	if (!P || !popularity || nHeads <= 0 || T <= 0) return false;
	const int block = 256;
	const size_t smem = block * sizeof(float);
	dim3 grid((unsigned int)T, (unsigned int)nHeads);
	k_kvface_popularity<<<grid, block, smem>>>(P, popularity,
	                                           (unsigned int)nHeads,
	                                           (unsigned int)T);
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::fprintf(stderr, "kvface_probe_compute_popularity: launch failed: %s\n",
		             cudaGetErrorString(err));
		return false;
	}
	return true;
}

bool kvface_probe_gini_device(const float* popularity,
                              float* gini_per_head,
                              int nHeads,
                              int T)
{
	if (!popularity || !gini_per_head || nHeads <= 0 || T <= 0) return false;
	const int block = 256;
	// Shared memory: T floats for cached values + block floats for reduction.
	const size_t smem = ((size_t)T + (size_t)block) * sizeof(float);
	// Guard against GPU shared-memory limit (typically 48 KB ~ 12k floats).
	if (smem > 48u * 1024u) {
		std::fprintf(stderr,
		             "kvface_probe_gini_device: T=%d exceeds shared memory budget (%zu bytes)\n",
		             T, smem);
		return false;
	}
	k_kvface_gini<<<(unsigned int)nHeads, block, smem>>>(popularity, gini_per_head,
	                                                     (unsigned int)nHeads,
	                                                     (unsigned int)T);
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::fprintf(stderr, "kvface_probe_gini_device: launch failed: %s\n",
		             cudaGetErrorString(err));
		return false;
	}
	return true;
}

bool kvface_probe_reduce_stats(const float* gini_per_head,
                               float* stats3,
                               int nHeads)
{
	if (!gini_per_head || !stats3 || nHeads <= 0) return false;
	const int block = 128;
	const size_t smem = 3u * block * sizeof(float);
	k_kvface_reduce_stats<<<1u, block, smem>>>(gini_per_head, stats3,
	                                            (unsigned int)nHeads);
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::fprintf(stderr, "kvface_probe_reduce_stats: launch failed: %s\n",
		             cudaGetErrorString(err));
		return false;
	}
	return true;
}

float kvface_probe_gini_host(const float* popularity_host, int T)
{
	if (!popularity_host || T <= 1) return 0.0f;

	std::vector<float> x(popularity_host, popularity_host + T);
	std::sort(x.begin(), x.end());

	// Gini using sorted Lorenz curve formula:
	//   G = (2 * Σ_{i=1}^{T} i * x_i) / (T * Σ_i x_i) - (T+1)/T
	// where i is 1-indexed.
	double total_sum = 0.0;
	double weighted  = 0.0;
	for (int i = 0; i < T; ++i) {
		const double v = (double)x[i];
		total_sum += v;
		weighted  += v * (double)(i + 1);
	}
	if (total_sum < 1e-20) return 0.0f;
	const double G = (2.0 * weighted) / ((double)T * total_sum) - ((double)(T + 1)) / (double)T;
	return (float)G;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
