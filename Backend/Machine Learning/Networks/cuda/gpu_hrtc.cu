// HRTC (Hierarchical Reversible Token Compression) GPU primitives.
// See gpu_hrtc.h and research/HRTC_DESIGN.md for the framework.

#include "gpu_hrtc.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>

namespace glades {
namespace gpu {

namespace {

// Haar at k=2: one element of the m-vector per thread.  Grid is
// [blocks = T/2, threads = m (capped)].  super[i] and residual[i] are
// computed from X[2i] and X[2i+1] using the orthogonal 2×2 Haar matrix
//   [ 1/√2   1/√2 ]
//   [ 1/√2  -1/√2 ]
// At FP32 this is invertible to ~7 decimal digits (1/√2 is not bit-exact
// reciprocal of √2 in IEEE 754, but the accumulated round-trip error stays
// well below BF16 ULP).
__global__ void k_hrtc_pool_haar_k2(const float* __restrict__ X,
                                    float* __restrict__ super,
                                    float* __restrict__ residual,
                                    unsigned int T, unsigned int m)
{
	unsigned int i = blockIdx.x;                 // output block index in [0, T/2)
	unsigned int j = blockIdx.y * blockDim.x + threadIdx.x;
	if (j >= m || i * 2u + 1u >= T) return;

	const float inv_sqrt2 = 0.70710678118654752f;
	const float x0 = X[(size_t)(2u * i) * m + j];
	const float x1 = X[(size_t)(2u * i + 1u) * m + j];
	super[(size_t)i * m + j]    = inv_sqrt2 * (x0 + x1);
	residual[(size_t)i * m + j] = inv_sqrt2 * (x0 - x1);
}

__global__ void k_hrtc_unpool_haar_k2(const float* __restrict__ super,
                                      const float* __restrict__ residual,
                                      float* __restrict__ X_out,
                                      unsigned int T, unsigned int m)
{
	unsigned int i = blockIdx.x;
	unsigned int j = blockIdx.y * blockDim.x + threadIdx.x;
	if (j >= m || i * 2u + 1u >= T) return;

	const float inv_sqrt2 = 0.70710678118654752f;
	const float s = super[(size_t)i * m + j];
	const float r = residual[(size_t)i * m + j];
	X_out[(size_t)(2u * i) * m + j]      = inv_sqrt2 * (s + r);
	X_out[(size_t)(2u * i + 1u) * m + j] = inv_sqrt2 * (s - r);
}

} // anonymous namespace

bool hrtc_pool_haar_k2(const float* X, float* super, float* residual,
                       unsigned int T, unsigned int m)
{
	if (X == nullptr || super == nullptr || residual == nullptr) return false;
	if ((T & 1u) != 0u || T == 0u || m == 0u) return false;

	dim3 block(256);
	dim3 grid(T / 2u, (m + block.x - 1u) / block.x);
	k_hrtc_pool_haar_k2<<<grid, block, 0, computeStream()>>>(X, super, residual, T, m);
	if (cudaGetLastError() != cudaSuccess) return false;
	return true;
}

bool hrtc_unpool_haar_k2(const float* super, const float* residual,
                        float* X_out, unsigned int T, unsigned int m)
{
	if (super == nullptr || residual == nullptr || X_out == nullptr) return false;
	if ((T & 1u) != 0u || T == 0u || m == 0u) return false;

	dim3 block(256);
	dim3 grid(T / 2u, (m + block.x - 1u) / block.x);
	k_hrtc_unpool_haar_k2<<<grid, block, 0, computeStream()>>>(super, residual, X_out, T, m);
	if (cudaGetLastError() != cudaSuccess) return false;
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
