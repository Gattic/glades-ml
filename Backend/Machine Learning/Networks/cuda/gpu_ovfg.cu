// OVFG (Operator-Valued Factored Gradient) GPU primitives.
// See gpu_ovfg.h and research/PARADIGM_SHIFT_9_CANDIDATE_C_OVFG.md.

#include "gpu_ovfg.h"
#include "gpu_blas.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>

namespace glades {
namespace gpu {

namespace {

// Row-major transpose: Y[b, a] = X[a, b].  One thread per output element.
// Grid is (rows_out + block - 1) / block along Y, (cols_out + block - 1)
// along X.  Uses tile-in-shared-mem to coalesce global accesses.
__global__ void k_transpose_2d(const float* __restrict__ X,
                               float* __restrict__ Y,
                               unsigned int rows_X, unsigned int cols_X)
{
	// Y has shape [cols_X, rows_X].
	const unsigned int TILE = 32;
	__shared__ float tile[32][33]; // +1 to avoid bank conflicts

	unsigned int x_row = blockIdx.y * TILE + threadIdx.y;
	unsigned int x_col = blockIdx.x * TILE + threadIdx.x;
	if (x_row < rows_X && x_col < cols_X)
		tile[threadIdx.y][threadIdx.x] = X[(size_t)x_row * cols_X + x_col];
	__syncthreads();

	// After transpose: out-element at (x_col, x_row) is tile[ty][tx]; when
	// we write out, swap block indices.
	unsigned int y_row = blockIdx.x * TILE + threadIdx.y; // now iterating cols_X
	unsigned int y_col = blockIdx.y * TILE + threadIdx.x; // now iterating rows_X
	if (y_row < cols_X && y_col < rows_X)
		Y[(size_t)y_row * rows_X + y_col] = tile[threadIdx.x][threadIdx.y];
}

static bool transpose_rowmajor(const float* X, float* Y,
                               unsigned int rows_X, unsigned int cols_X)
{
	if (X == nullptr || Y == nullptr || rows_X == 0u || cols_X == 0u)
		return false;
	dim3 block(32, 32);
	dim3 grid((cols_X + 31u) / 32u, (rows_X + 31u) / 32u);
	k_transpose_2d<<<grid, block, 0, computeStream()>>>(X, Y, rows_X, cols_X);
	return cudaGetLastError() == cudaSuccess;
}

} // anonymous namespace

// ========================================================================
// ovfg_compute_dense_from_factors — PARITY HELPER (tests only).
//
// G [m × n] = L [m × r] · R^T [r × n].  Since R is stored [n × r]
// row-major, R^T has shape [r × n] and is accessed via the ABT GEMM.
// ========================================================================
bool ovfg_compute_dense_from_factors(const float* L, const float* R,
                                     unsigned int m, unsigned int n,
                                     unsigned int r,
                                     float* G_out)
{
	if (L == nullptr || R == nullptr || G_out == nullptr) return false;
	if (m == 0u || n == 0u || r == 0u) return false;

	// sgemm_rowmajor_abt: C[M,N] = A[M,K] · B^T[K,N] where B is stored [N,K].
	// Here A=L [m,r], B=R [n,r], result C=G [m,n].  M=m, N=n, K=r.
	return sgemm_rowmajor_abt(static_cast<int>(m), static_cast<int>(n), static_cast<int>(r),
	                          1.0f,
	                          L, static_cast<int>(r),
	                          R, static_cast<int>(r),
	                          0.0f,
	                          G_out, static_cast<int>(n));
}

// ========================================================================
// ovfg_factored_grad_from_activation
//
// Input: A [T × m], D [T × n].
// Output: L [m × T] = A^T, R [n × T] = D^T.
// Then G = L · R^T = A^T · D, as required.
//
// Phase 1 uses explicit transposes.  Phase 2 will expose A, D directly
// (they already exist in the backward path) and skip the copy, recovering
// the full "zero extra memory" intent of OVFG.
// ========================================================================
bool ovfg_factored_grad_from_activation(const float* A, const float* D,
                                        unsigned int T,
                                        unsigned int m, unsigned int n,
                                        float* L_out, float* R_out)
{
	if (A == nullptr || D == nullptr || L_out == nullptr || R_out == nullptr)
		return false;
	if (T == 0u || m == 0u || n == 0u) return false;

	if (!transpose_rowmajor(A, L_out, T, m)) return false;
	if (!transpose_rowmajor(D, R_out, T, n)) return false;
	return true;
}

// ========================================================================
// ovfg_apply_update_dense
//
// W[m, n] ← W[m, n] + (-eta) · L[m, r] · R^T[r, n]
//
// One SGEMM.  Parity path for dense weights (LayerNorm γ/β, etc.).
// Stiefel-factored weights use ovfg_stiefel_tangent_grad in Phase 3.
// ========================================================================
bool ovfg_apply_update_dense(const float* L, const float* R,
                             unsigned int m, unsigned int n,
                             unsigned int r,
                             float eta,
                             float* W)
{
	if (L == nullptr || R == nullptr || W == nullptr) return false;
	if (m == 0u || n == 0u || r == 0u) return false;

	return sgemm_rowmajor_abt(static_cast<int>(m), static_cast<int>(n), static_cast<int>(r),
	                          -eta,
	                          L, static_cast<int>(r),
	                          R, static_cast<int>(r),
	                          1.0f,
	                          W, static_cast<int>(n));
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
