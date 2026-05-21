// NESR (Noise-Equilibrium Stochastic Resonance) GPU primitive.
// See gpu_nesr.h and research/FUTURE_PARADIGM_CANDIDATES.md §32.

#include "gpu_nesr.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace glades {
namespace gpu {

namespace {

// xorshift32 PRNG; returns a value in [0, 2^32).
__device__ __forceinline__ unsigned int xorshift32(unsigned int state)
{
	state ^= state << 13;
	state ^= state >> 17;
	state ^= state << 5;
	return state;
}

// Two uniform-[0,1) samples from one xorshift32 sequence.  Returns a
// standard-normal sample via Box-Muller.  Always takes two uniforms
// and returns the first Box-Muller component (the second is discarded).
__device__ __forceinline__ float gaussian_from_seed(unsigned int seed)
{
	unsigned int s = xorshift32(seed);
	const float u1 = (float)(s & 0x00ffffffu) / (float)(1u << 24);
	s = xorshift32(s);
	const float u2 = (float)(s & 0x00ffffffu) / (float)(1u << 24);
	// Box-Muller: z = √(-2 · ln u1) · cos(2π · u2)
	const float u1_clamped = (u1 < 1e-7f) ? 1e-7f : u1;
	const float r = sqrtf(-2.0f * logf(u1_clamped));
	const float theta = 6.283185307179586f * u2;
	return r * cosf(theta);
}

__global__ void k_nesr_inject_noise(float* __restrict__ theta,
                                    float scale,
                                    unsigned int base_seed,
                                    unsigned int step,
                                    int n_params)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_params) return;
	// Seed is a mix of base, parameter index, and step.  XOR is fine
	// for decorrelation; the xorshift32 inside Box-Muller re-whitens.
	const unsigned int seed = (base_seed ^ (unsigned)(i * 2654435761u)) ^ (step * 1013904223u);
	const float z = gaussian_from_seed(seed == 0u ? 1u : seed);  // guard against 0
	theta[i] += scale * z;
}

} // anonymous namespace

bool nesr_inject_noise(float* theta,
                       float scale,
                       unsigned int base_seed,
                       unsigned int step,
                       int n_params)
{
	if (theta == nullptr) return false;
	if (n_params <= 0) return false;
	if (scale == 0.0f) return true;  // no-op
	const int block = 256;
	const int grid  = (n_params + block - 1) / block;
	k_nesr_inject_noise<<<grid, block, 0, computeStream()>>>(
	    theta, scale, base_seed, step, n_params);
	return cudaGetLastError() == cudaSuccess;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
