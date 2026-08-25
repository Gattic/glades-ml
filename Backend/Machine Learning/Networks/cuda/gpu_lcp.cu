// LCP (Lattice Compute Pooling, paradigm shift #16) GPU primitives.
// See gpu_lcp.h, research/PARADIGM_SHIFT_16_SELECTION.md.

#include "gpu_lcp.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace glades {
namespace gpu {

namespace {

// ------------------------------------------------------------------------
// LSH projection: per-token, compute h_t · R_j for j ∈ [0, K), pack the
// sign bits into a uint32 bucket id.  One block per token, K threads
// cooperate on the K hash bits; each thread handles one R column.
// ------------------------------------------------------------------------
__global__ void k_lcp_lsh_project(const float* __restrict__ h,
                                  const float* __restrict__ R,
                                  int T, int d, int K,
                                  unsigned int* __restrict__ bucket_out)
{
	const int t = blockIdx.x;
	if (t >= T) return;

	const int tid = threadIdx.x;
	const int block = blockDim.x;
	const float* h_row = h + (size_t)t * d;

	// Each thread computes dots for a strided subset of hash bits.
	extern __shared__ float sh_dots[];  // size = K floats
	if (tid < K) sh_dots[tid] = 0.0f;
	__syncthreads();

	for (int j = tid; j < K; j += block) {
		const float* R_col = R + (size_t)j * d;
		float s = 0.0f;
		for (int i = 0; i < d; ++i) s += h_row[i] * R_col[i];
		sh_dots[j] = s;
	}
	__syncthreads();

	if (tid == 0) {
		unsigned int b = 0u;
		for (int j = 0; j < K; ++j) {
			if (sh_dots[j] > 0.0f) b |= (1u << j);
		}
		bucket_out[t] = b;
	}
}

// ------------------------------------------------------------------------
// Gaussian init via Box-Muller on splitmix64 counter.  One thread per
// pair of entries; scale by 1/√d so ‖h‖·‖R_j‖ is O(1).
// ------------------------------------------------------------------------
__global__ void k_lcp_lsh_init(float* __restrict__ R,
                               int n_elems, uint64_t seed, float scale)
{
	// Each thread writes 2 entries using one Box-Muller draw.
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

	R[i0] = z0;
	if (i0 + 1 < n_elems) R[i0 + 1] = z1;
}

// ------------------------------------------------------------------------
// Gather rows.  h_reps_out[m, :] = h_in[rep_idx[m], :].
// ------------------------------------------------------------------------
__global__ void k_lcp_gather(const float* __restrict__ h_in,
                             const unsigned int* __restrict__ rep_idx,
                             int n_reps, int d,
                             float* __restrict__ h_reps_out)
{
	const int m = blockIdx.x;
	if (m >= n_reps) return;

	const unsigned int src = rep_idx[m];
	const float* src_row = h_in + (size_t)src * d;
	float* dst_row = h_reps_out + (size_t)m * d;

	for (int j = threadIdx.x; j < d; j += blockDim.x)
		dst_row[j] = src_row[j];
}

// ------------------------------------------------------------------------
// Scatter: h_out[t, :] = h_reps_in[cluster_of_token[t], :].
// ------------------------------------------------------------------------
__global__ void k_lcp_scatter(const float* __restrict__ h_reps_in,
                              const unsigned int* __restrict__ cluster,
                              int T, int d,
                              float* __restrict__ h_out)
{
	const int t = blockIdx.x;
	if (t >= T) return;

	const unsigned int c = cluster[t];
	const float* src_row = h_reps_in + (size_t)c * d;
	float* dst_row = h_out + (size_t)t * d;

	for (int j = threadIdx.x; j < d; j += blockDim.x)
		dst_row[j] = src_row[j];
}

// ------------------------------------------------------------------------
// Scatter backward: dh_reps[cluster[t], :] += dh_out[t, :].
// Uses atomicAdd since multiple tokens map to the same cluster.
// ------------------------------------------------------------------------
__global__ void k_lcp_scatter_backward(const float* __restrict__ dh_out,
                                       const unsigned int* __restrict__ cluster,
                                       int T, int d,
                                       float* __restrict__ dh_reps_out)
{
	const int t = blockIdx.x;
	if (t >= T) return;

	const unsigned int c = cluster[t];
	const float* src_row = dh_out + (size_t)t * d;
	float* dst_row = dh_reps_out + (size_t)c * d;

	for (int j = threadIdx.x; j < d; j += blockDim.x)
		atomicAdd(&dst_row[j], src_row[j]);
}

__global__ void k_zero_floats(float* p, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) p[i] = 0.0f;
}

// ------------------------------------------------------------------------
// delta[t, :] = h[t, :] - h_reps[cluster_of_token[t], :]
// ------------------------------------------------------------------------
__global__ void k_lcp_compute_delta(const float* __restrict__ h_in,
                                    const float* __restrict__ h_reps,
                                    const unsigned int* __restrict__ cluster,
                                    int T, int d,
                                    float* __restrict__ delta_out)
{
	const int t = blockIdx.x;
	if (t >= T) return;

	const unsigned int c = cluster[t];
	const float* h_row    = h_in    + (size_t)t * d;
	const float* rep_row  = h_reps  + (size_t)c * d;
	float*       out_row  = delta_out + (size_t)t * d;

	for (int j = threadIdx.x; j < d; j += blockDim.x)
		out_row[j] = h_row[j] - rep_row[j];
}

// ------------------------------------------------------------------------
// delta backward:
//   dh_in[t, :]   += d_delta[t, :]
//   dh_reps[c, :] -= Σ_{t : cluster[t]==c} d_delta[t, :]
// ------------------------------------------------------------------------
__global__ void k_lcp_delta_backward_dh_in(const float* __restrict__ d_delta,
                                           int T, int d,
                                           float* __restrict__ dh_in)
{
	const int t = blockIdx.x;
	if (t >= T) return;

	const float* src_row = d_delta + (size_t)t * d;
	float*       dst_row = dh_in   + (size_t)t * d;

	for (int j = threadIdx.x; j < d; j += blockDim.x)
		dst_row[j] += src_row[j];
}

__global__ void k_lcp_delta_backward_dh_reps(const float* __restrict__ d_delta,
                                             const unsigned int* __restrict__ cluster,
                                             int T, int d,
                                             float* __restrict__ dh_reps)
{
	const int t = blockIdx.x;
	if (t >= T) return;

	const unsigned int c = cluster[t];
	const float* src_row = d_delta + (size_t)t * d;
	float*       dst_row = dh_reps + (size_t)c * d;

	for (int j = threadIdx.x; j < d; j += blockDim.x)
		atomicAdd(&dst_row[j], -src_row[j]);
}

} // anonymous namespace

bool lcp_lsh_project(const float* h_in, const float* R,
                     unsigned int T, unsigned int d, unsigned int K,
                     unsigned int* bucket_out)
{
	if (h_in == nullptr || R == nullptr || bucket_out == nullptr) return false;
	if (T == 0u || d == 0u || K == 0u || K > 32u) return false;

	const int block = (K >= 32) ? 32 : (int)K;
	const size_t shared = (size_t)K * sizeof(float);
	k_lcp_lsh_project<<<(int)T, block, shared, computeStream()>>>(
	    h_in, R, (int)T, (int)d, (int)K, bucket_out);
	return cudaGetLastError() == cudaSuccess;
}

bool lcp_lsh_init_matrix(float* R_out, unsigned int d, unsigned int K,
                         uint64_t seed)
{
	if (R_out == nullptr) return false;
	if (d == 0u || K == 0u) return false;

	const int n_elems = (int)((size_t)d * K);
	const int n_pairs = (n_elems + 1) / 2;
	const int block = 256;
	const int grid  = (n_pairs + block - 1) / block;
	const float scale = 1.0f / sqrtf((float)d);
	k_lcp_lsh_init<<<grid, block, 0, computeStream()>>>(
	    R_out, n_elems, seed, scale);
	return cudaGetLastError() == cudaSuccess;
}

bool lcp_bucket_first_index(const unsigned int* bucket_in,
                            unsigned int T, unsigned int M_max,
                            unsigned int* rep_idx_out,
                            int* n_reps_out,
                            unsigned int* cluster_of_token_out)
{
	if (bucket_in == nullptr || rep_idx_out == nullptr ||
	    n_reps_out == nullptr || cluster_of_token_out == nullptr) return false;
	if (T == 0u || M_max == 0u) return false;

	// Implementation: download bucket_in, compute on host (T is small enough
	// that the copy dominates).  This is the simplest correct implementation
	// — a fully-GPU version would use a sorted-segment scan kernel.
	std::vector<unsigned int> buckets((size_t)T);
	cudaError_t err = cudaMemcpyAsync(&buckets[0], bucket_in,
	                                   sizeof(unsigned int) * T,
	                                   cudaMemcpyDeviceToHost,
	                                   computeStream());
	if (err != cudaSuccess) return false;
	cudaStreamSynchronize(computeStream());

	// Deterministic first-occurrence map: iterate tokens in order, assign
	// cluster ids in the order buckets first appear.
	std::vector<unsigned int> cluster_of((size_t)T);
	std::vector<unsigned int> reps;
	reps.reserve(M_max);
	std::vector<int> bucket_to_cluster(65536, -1);  // K≤16 case; for K=32 use unordered_map
	bool use_map = false;
	std::map<unsigned int, unsigned int>* map_ptr = nullptr;

	// Heuristic: if any bucket id exceeds 65535, switch to map.
	for (unsigned int t = 0; t < T; ++t) {
		if (buckets[t] >= 65536u) { use_map = true; break; }
	}
	if (use_map) {
		map_ptr = new std::map<unsigned int, unsigned int>();
	}

	for (unsigned int t = 0; t < T; ++t) {
		const unsigned int b = buckets[t];
		int c = -1;
		if (use_map) {
			auto it = map_ptr->find(b);
			if (it != map_ptr->end()) c = (int)it->second;
		} else {
			c = bucket_to_cluster[b];
		}
		if (c < 0) {
			if (reps.size() >= M_max) {
				// Spill: token falls into an arbitrary existing cluster (cluster 0).
				// This keeps the invariant n_reps ≤ M_max.  Acceptable for training;
				// warning worth logging at runtime.
				c = 0;
			} else {
				c = (int)reps.size();
				reps.push_back(t);
				if (use_map) (*map_ptr)[b] = (unsigned int)c;
				else         bucket_to_cluster[b] = c;
			}
		}
		cluster_of[t] = (unsigned int)c;
	}
	if (use_map) delete map_ptr;

	const int n_reps = (int)reps.size();
	*n_reps_out = n_reps;

	// Upload outputs.
	if (n_reps > 0) {
		err = cudaMemcpyAsync(rep_idx_out, &reps[0],
		                       sizeof(unsigned int) * (size_t)n_reps,
		                       cudaMemcpyHostToDevice,
		                       computeStream());
		if (err != cudaSuccess) return false;
	}
	err = cudaMemcpyAsync(cluster_of_token_out, &cluster_of[0],
	                       sizeof(unsigned int) * (size_t)T,
	                       cudaMemcpyHostToDevice,
	                       computeStream());
	if (err != cudaSuccess) return false;
	cudaStreamSynchronize(computeStream());
	return true;
}

bool lcp_gather(const float* h_in, const unsigned int* rep_idx,
                unsigned int n_reps, unsigned int d,
                float* h_reps_out)
{
	if (h_in == nullptr || rep_idx == nullptr || h_reps_out == nullptr) return false;
	if (n_reps == 0u || d == 0u) return false;

	const int block = (d >= 256) ? 256 : ((d >= 64) ? 64 : 32);
	k_lcp_gather<<<(int)n_reps, block, 0, computeStream()>>>(
	    h_in, rep_idx, (int)n_reps, (int)d, h_reps_out);
	return cudaGetLastError() == cudaSuccess;
}

bool lcp_scatter(const float* h_reps_in,
                 const unsigned int* cluster_of_token,
                 unsigned int T, unsigned int d,
                 float* h_out)
{
	if (h_reps_in == nullptr || cluster_of_token == nullptr || h_out == nullptr) return false;
	if (T == 0u || d == 0u) return false;

	const int block = (d >= 256) ? 256 : ((d >= 64) ? 64 : 32);
	k_lcp_scatter<<<(int)T, block, 0, computeStream()>>>(
	    h_reps_in, cluster_of_token, (int)T, (int)d, h_out);
	return cudaGetLastError() == cudaSuccess;
}

bool lcp_scatter_backward(const float* dh_out,
                          const unsigned int* cluster_of_token,
                          unsigned int T, unsigned int d,
                          unsigned int n_reps,
                          bool accumulate,
                          float* dh_reps_out)
{
	if (dh_out == nullptr || cluster_of_token == nullptr || dh_reps_out == nullptr) return false;
	if (T == 0u || d == 0u || n_reps == 0u) return false;

	if (!accumulate) {
		const int n_elems = (int)((size_t)n_reps * d);
		const int block = 256;
		const int grid  = (n_elems + block - 1) / block;
		k_zero_floats<<<grid, block, 0, computeStream()>>>(dh_reps_out, n_elems);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	const int block = (d >= 256) ? 256 : ((d >= 64) ? 64 : 32);
	k_lcp_scatter_backward<<<(int)T, block, 0, computeStream()>>>(
	    dh_out, cluster_of_token, (int)T, (int)d, dh_reps_out);
	return cudaGetLastError() == cudaSuccess;
}

bool lcp_compute_delta(const float* h_in, const float* h_reps_in,
                       const unsigned int* cluster_of_token,
                       unsigned int T, unsigned int d,
                       float* delta_out)
{
	if (h_in == nullptr || h_reps_in == nullptr ||
	    cluster_of_token == nullptr || delta_out == nullptr) return false;
	if (T == 0u || d == 0u) return false;

	const int block = (d >= 256) ? 256 : ((d >= 64) ? 64 : 32);
	k_lcp_compute_delta<<<(int)T, block, 0, computeStream()>>>(
	    h_in, h_reps_in, cluster_of_token, (int)T, (int)d, delta_out);
	return cudaGetLastError() == cudaSuccess;
}

bool lcp_compute_delta_backward(const float* d_delta,
                                const unsigned int* cluster_of_token,
                                unsigned int T, unsigned int d,
                                unsigned int n_reps,
                                float* dh_in_out, float* dh_reps_out)
{
	if (d_delta == nullptr || cluster_of_token == nullptr) return false;
	if (T == 0u || d == 0u || n_reps == 0u) return false;

	const int block = (d >= 256) ? 256 : ((d >= 64) ? 64 : 32);
	if (dh_in_out != nullptr) {
		k_lcp_delta_backward_dh_in<<<(int)T, block, 0, computeStream()>>>(
		    d_delta, (int)T, (int)d, dh_in_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	if (dh_reps_out != nullptr) {
		k_lcp_delta_backward_dh_reps<<<(int)T, block, 0, computeStream()>>>(
		    d_delta, cluster_of_token, (int)T, (int)d, dh_reps_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
