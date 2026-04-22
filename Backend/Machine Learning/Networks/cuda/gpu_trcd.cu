// TRCD (Token-Routed Conditional Depth, paradigm shift #13) GPU primitives.
// See gpu_trcd.h, research/PARADIGM_SHIFT_13_CANDIDATE_C_TRCD.md,
// research/PARADIGM_SHIFT_13_SELECTION.md.

#include "gpu_trcd.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

namespace glades {
namespace gpu {

namespace {

// ------------------------------------------------------------------------
// Block-per-token dot: u[t] = a · h[t, :] + b
// One thread block handles one token; block_size threads cooperate on d.
// ------------------------------------------------------------------------
__global__ void k_trcd_route_logits(const float* __restrict__ h,
                                    const float* __restrict__ a,
                                    float b,
                                    int T, int d,
                                    float* __restrict__ u_out)
{
	const int t = blockIdx.x;
	if (t >= T) return;

	const int tid = threadIdx.x;
	const int block = blockDim.x;

	const float* h_row = h + (size_t)t * d;

	float local = 0.0f;
	for (int j = tid; j < d; j += block)
		local += h_row[j] * a[j];

	// Warp+block reduction via shared memory.
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
		if (tid == 0) u_out[t] = v + b;
	}
}

// ------------------------------------------------------------------------
// Backward: dh[t, :] = dU[t] * a[:]; gA += dU^T · h_in; gB += sum(dU).
// Two kernels: one for dh (trivial scale-by-row), one for gA + gB (block
// per feature j, sum over t).
// ------------------------------------------------------------------------
__global__ void k_trcd_route_dh(const float* __restrict__ dU,
                                const float* __restrict__ a,
                                int T, int d,
                                float* __restrict__ dh)
{
	const int t = blockIdx.x;
	if (t >= T) return;

	const float s = dU[t];
	float* out_row = dh + (size_t)t * d;

	for (int j = threadIdx.x; j < d; j += blockDim.x)
		out_row[j] = s * a[j];
}

__global__ void k_trcd_route_gA_gB(const float* __restrict__ dU,
                                   const float* __restrict__ h,
                                   int T, int d,
                                   float* __restrict__ gA,
                                   float* __restrict__ gB)
{
	// Block j reduces over t: gA[j] += sum_t dU[t] * h[t, j].
	const int j = blockIdx.x;
	if (j >= d) return;

	const int tid = threadIdx.x;
	const int block = blockDim.x;

	float local = 0.0f;
	for (int t = tid; t < T; t += block)
		local += dU[t] * h[(size_t)t * d + j];

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
		if (tid == 0) {
			atomicAdd(&gA[j], v);
			// Only j == 0 accumulates the bias sum; avoid d-way contention.
			if (j == 0 && gB != nullptr) {
				float bsum = 0.0f;
				for (int t = 0; t < T; ++t) bsum += dU[t];
				atomicAdd(gB, bsum);
			}
		}
	}
}

// ------------------------------------------------------------------------
// Gumbel-softmax gate.
//
// Training (with fresh Gumbel sample g ~ Gumbel(0, 1) = -log(-log(u)),
// u ~ Uniform(0, 1)):
//   α = σ( (u_t − λ + g_t) / τ )
//
// Evaluation (deterministic, zero noise):
//   α = 1 iff u_t > λ else 0
// ------------------------------------------------------------------------
__device__ inline float trcd_gumbel_sample(uint64_t seed, int i)
{
	// splitmix64-style counter RNG → uniform [ε, 1).
	uint64_t x = seed ^ (uint64_t)(i * 0x9E3779B97F4A7C15ULL);
	x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
	x ^= x >> 27; x *= 0x94D049BB133111EBULL;
	x ^= x >> 31;

	const uint32_t mant = (uint32_t)(x >> 9) & 0x007fffffu;
	float u01 = (float)mant / (float)(1u << 23);
	// Clamp to avoid log(0).
	if (u01 < 1e-7f) u01 = 1e-7f;
	if (u01 > 1.0f - 1e-7f) u01 = 1.0f - 1e-7f;
	return -logf(-logf(u01));
}

__global__ void k_trcd_gumbel_gate_train(const float* __restrict__ u,
                                         float lambda, float tau,
                                         uint64_t seed,
                                         int T,
                                         float* __restrict__ alpha)
{
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= T) return;

	const float g = trcd_gumbel_sample(seed, t);
	const float x = (u[t] - lambda + g) / tau;
	// Logistic; avoid overflow.
	float a;
	if (x > 20.0f)      a = 1.0f;
	else if (x < -20.0f) a = 0.0f;
	else                 a = 1.0f / (1.0f + expf(-x));
	alpha[t] = a;
}

__global__ void k_trcd_gumbel_gate_eval(const float* __restrict__ u,
                                        float lambda,
                                        int T,
                                        float* __restrict__ alpha)
{
	const int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= T) return;
	alpha[t] = (u[t] > lambda) ? 1.0f : 0.0f;
}

// ------------------------------------------------------------------------
// Apply gate: h_out[t, :] = α[t] * h_in[t, :].
// Also handles the backward by row-parallel scale.
// ------------------------------------------------------------------------
__global__ void k_trcd_apply_gate(const float* __restrict__ h_in,
                                  const float* __restrict__ alpha,
                                  int T, int d,
                                  float* __restrict__ h_out)
{
	const int t = blockIdx.x;
	if (t >= T) return;

	const float s = alpha[t];
	const float* in_row = h_in + (size_t)t * d;
	float* out_row = h_out + (size_t)t * d;

	for (int j = threadIdx.x; j < d; j += blockDim.x)
		out_row[j] = s * in_row[j];
}

__global__ void k_trcd_apply_gate_dh(const float* __restrict__ dh_out,
                                     const float* __restrict__ alpha,
                                     int T, int d,
                                     float* __restrict__ dh_in)
{
	const int t = blockIdx.x;
	if (t >= T) return;

	const float s = alpha[t];
	const float* in_row = dh_out + (size_t)t * d;
	float* out_row = dh_in + (size_t)t * d;

	for (int j = threadIdx.x; j < d; j += blockDim.x)
		out_row[j] = s * in_row[j];
}

__global__ void k_trcd_apply_gate_dalpha(const float* __restrict__ dh_out,
                                         const float* __restrict__ h_in,
                                         int T, int d,
                                         float* __restrict__ dalpha)
{
	// Row t: dalpha[t] = sum_j h_in[t, j] * dh_out[t, j].
	const int t = blockIdx.x;
	if (t >= T) return;

	const int tid = threadIdx.x;
	const int block = blockDim.x;

	const float* hin_row = h_in   + (size_t)t * d;
	const float* dh_row  = dh_out + (size_t)t * d;

	float local = 0.0f;
	for (int j = tid; j < d; j += block)
		local += hin_row[j] * dh_row[j];

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
		if (tid == 0) dalpha[t] = v;
	}
}

} // anonymous namespace

bool trcd_route_logits(const float* h_in, const float* a_l, float b_l,
                       unsigned int T, unsigned int d,
                       float* u_out)
{
	if (h_in == nullptr || a_l == nullptr || u_out == nullptr) return false;
	if (T == 0u || d == 0u) return false;

	// Block size chosen to cover typical d (up to 8192) with good occupancy.
	int block = 256;
	if (d < 256) {
		if (d >= 128)      block = 128;
		else if (d >= 64)  block = 64;
		else               block = 32;
	}
	k_trcd_route_logits<<<(int)T, block, 0, computeStream()>>>(
	    h_in, a_l, b_l, (int)T, (int)d, u_out);
	return cudaGetLastError() == cudaSuccess;
}

bool trcd_route_logits_backward(const float* dU, const float* h_in,
                                const float* a_l,
                                unsigned int T, unsigned int d,
                                float* gA_out, float* gB_out,
                                float* dh_out)
{
	if (dU == nullptr || h_in == nullptr) return false;
	if (T == 0u || d == 0u) return false;

	if (dh_out != nullptr) {
		if (a_l == nullptr) return false;
		const int block = (d >= 256) ? 256 : ((d >= 64) ? 64 : 32);
		k_trcd_route_dh<<<(int)T, block, 0, computeStream()>>>(
		    dU, a_l, (int)T, (int)d, dh_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	if (gA_out != nullptr) {
		const int block = 256;
		k_trcd_route_gA_gB<<<(int)d, block, 0, computeStream()>>>(
		    dU, h_in, (int)T, (int)d, gA_out, gB_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	} else if (gB_out != nullptr) {
		// Fallback: user wanted bias grad only; compute via a 1-block launch.
		const int block = 256;
		k_trcd_route_gA_gB<<<1, block, 0, computeStream()>>>(
		    dU, h_in, (int)T, (int)d,
		    nullptr, gB_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	return true;
}

bool trcd_gumbel_gate(const float* u, float lambda, float tau,
                      uint64_t seed, bool training,
                      unsigned int T,
                      float* alpha_out)
{
	if (u == nullptr || alpha_out == nullptr) return false;
	if (T == 0u) return false;

	const int block = 256;
	const int grid  = ((int)T + block - 1) / block;
	if (training) {
		if (tau <= 0.0f) return false;
		k_trcd_gumbel_gate_train<<<grid, block, 0, computeStream()>>>(
		    u, lambda, tau, seed, (int)T, alpha_out);
	} else {
		k_trcd_gumbel_gate_eval<<<grid, block, 0, computeStream()>>>(
		    u, lambda, (int)T, alpha_out);
	}
	return cudaGetLastError() == cudaSuccess;
}

bool trcd_apply_gate(const float* h_in, const float* alpha,
                     unsigned int T, unsigned int d,
                     float* h_out)
{
	if (h_in == nullptr || alpha == nullptr || h_out == nullptr) return false;
	if (T == 0u || d == 0u) return false;

	const int block = (d >= 256) ? 256 : ((d >= 64) ? 64 : 32);
	k_trcd_apply_gate<<<(int)T, block, 0, computeStream()>>>(
	    h_in, alpha, (int)T, (int)d, h_out);
	return cudaGetLastError() == cudaSuccess;
}

bool trcd_apply_gate_backward(const float* dh_out, const float* h_in,
                              const float* alpha,
                              unsigned int T, unsigned int d,
                              float* dh_in_out, float* dalpha_out)
{
	if (dh_out == nullptr || h_in == nullptr || alpha == nullptr) return false;
	if (T == 0u || d == 0u) return false;

	const int block = (d >= 256) ? 256 : ((d >= 64) ? 64 : 32);
	if (dh_in_out != nullptr) {
		k_trcd_apply_gate_dh<<<(int)T, block, 0, computeStream()>>>(
		    dh_out, alpha, (int)T, (int)d, dh_in_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	if (dalpha_out != nullptr) {
		k_trcd_apply_gate_dalpha<<<(int)T, block, 0, computeStream()>>>(
		    dh_out, h_in, (int)T, (int)d, dalpha_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	return true;
}

void trcd_lambda_pi_update(float d_bar_obs, float d_bar_target,
                           float kp, float ki,
                           float& integral_inout, float& lambda_inout,
                           float lambda_max)
{
	const float error = d_bar_obs - d_bar_target;
	integral_inout += error;

	float lambda_new = lambda_inout + kp * error + ki * integral_inout;
	if (lambda_new < 0.0f) lambda_new = 0.0f;
	if (lambda_new > lambda_max) lambda_new = lambda_max;
	lambda_inout = lambda_new;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
