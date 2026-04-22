// MPOT (Matrix Product Operator weight decomposition) GPU primitives.
// See gpu_mpot.h and research/PARADIGM_SHIFT_10_DESIGN.md.

#include "gpu_mpot.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>

namespace glades {
namespace gpu {

namespace {

// One thread per output element W[i, j].  Iterates α over bond dim D.
// Shapes:
//   A    [m_1, n_1, D]   row-major: A[i_1·n_1·D + j_1·D + α]
//   B    [D, m_2, n_2]   row-major: B[α·m_2·n_2 + i_2·n_2 + j_2]
//   W    [m_1·m_2, n_1·n_2] row-major
__global__ void k_mpot_reconstruct(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   unsigned int m_1, unsigned int m_2,
                                   unsigned int n_1, unsigned int n_2,
                                   unsigned int D,
                                   float* __restrict__ W_out)
{
	const unsigned int m = m_1 * m_2;
	const unsigned int n = n_1 * n_2;
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m || j >= n) return;

	const unsigned int i_1 = i / m_2;
	const unsigned int i_2 = i % m_2;
	const unsigned int j_1 = j / n_2;
	const unsigned int j_2 = j % n_2;

	// A row start for (i_1, j_1): [i_1·n_1·D + j_1·D]
	const size_t a_row = (size_t)i_1 * n_1 * D + (size_t)j_1 * D;
	// B base for (i_2, j_2) varies over α: offset = α·m_2·n_2 + i_2·n_2 + j_2
	// Inner loop touches α-strided elements in B — coalesced in α is not
	// trivial.  Inner loop is tiny (D ≤ 128 typical) so this is fine.

	float sum = 0.0f;
	for (unsigned int a = 0; a < D; ++a)
	{
		const float av = A[a_row + a];
		const float bv = B[(size_t)a * m_2 * n_2 + (size_t)i_2 * n_2 + j_2];
		sum += av * bv;
	}
	W_out[(size_t)i * n + j] = sum;
}

} // anonymous namespace

bool mpot_reconstruct_dense(const float* A, const float* B,
                            unsigned int m_1, unsigned int m_2,
                            unsigned int n_1, unsigned int n_2,
                            unsigned int D,
                            float* W_out)
{
	if (A == nullptr || B == nullptr || W_out == nullptr) return false;
	if (m_1 == 0u || m_2 == 0u || n_1 == 0u || n_2 == 0u || D == 0u) return false;

	const unsigned int m = m_1 * m_2;
	const unsigned int n = n_1 * n_2;

	dim3 block(32, 8);
	dim3 grid((n + block.x - 1u) / block.x,
	          (m + block.y - 1u) / block.y);
	k_mpot_reconstruct<<<grid, block, 0, computeStream()>>>(
	    A, B, m_1, m_2, n_1, n_2, D, W_out);
	return cudaGetLastError() == cudaSuccess;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
