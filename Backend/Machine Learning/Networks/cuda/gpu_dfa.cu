// DFA (Direct Feedback Alignment) GPU primitives.
// See gpu_dfa.h and research/PARADIGM_SHIFT_12_DESIGN.md.

#include "gpu_dfa.h"
#include "gpu_blas.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>

namespace glades {
namespace gpu {

namespace {

// Counter-based Philox-lite RNG — deterministic and cheap, good enough
// for the fixed-random matrices that never need cryptographic quality.
// One thread per element fills R_out with uniform [−scale, +scale].
__global__ void k_dfa_fill_uniform(float* __restrict__ R_out,
                                   size_t n, uint64_t seed, float scale)
{
	const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	// Simple splitmix64-style counter → float in [-scale, scale].
	uint64_t x = seed ^ (uint64_t)(i * 0x9E3779B97F4A7C15ULL);
	x ^= x >> 30;
	x *= 0xBF58476D1CE4E5B9ULL;
	x ^= x >> 27;
	x *= 0x94D049BB133111EBULL;
	x ^= x >> 31;

	const uint32_t mant = (uint32_t)(x >> 9) & 0x007fffffu;
	const float u01 = (float)mant / (float)(1u << 23);
	R_out[i] = (2.0f * u01 - 1.0f) * scale;
}

} // anonymous namespace

bool dfa_init_random_matrix(float* R_out,
                            unsigned int rows, unsigned int cols,
                            uint64_t seed,
                            float scale)
{
	if (R_out == nullptr) return false;
	if (rows == 0u || cols == 0u) return false;

	const size_t n = (size_t)rows * cols;
	const int block = 256;
	const int grid  = (int)((n + block - 1) / block);
	k_dfa_fill_uniform<<<grid, block, 0, computeStream()>>>(R_out, n, seed, scale);
	return cudaGetLastError() == cudaSuccess;
}

bool dfa_project_error(const float* e, const float* R,
                       unsigned int T,
                       unsigned int d_out, unsigned int d_hid,
                       float* e_proj_out)
{
	if (e == nullptr || R == nullptr || e_proj_out == nullptr) return false;
	if (T == 0u || d_out == 0u || d_hid == 0u) return false;

	// e [T × d_out] · R [d_out × d_hid] = e_proj [T × d_hid].
	return sgemm_rowmajor(T, d_hid, d_out,
	                      1.0f,
	                      e,    d_out,
	                      R,    d_hid,
	                      0.0f,
	                      e_proj_out, d_hid);
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
