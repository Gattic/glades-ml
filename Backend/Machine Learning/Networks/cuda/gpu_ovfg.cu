// OVFG (Operator-Valued Factored Gradient) GPU primitives.
// See gpu_ovfg.h and research/PARADIGM_SHIFT_9_CANDIDATE_C_OVFG.md.

#include "gpu_ovfg.h"
#include "gpu_blas.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

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

// ========================================================================
// Row-dot accumulator for Adafactor moment update.
//
//   out[i] ← beta2 · out[i] + (1 - beta2) · Σ_k A[i,k] · B[i,k]
//
// Where A and B share shape [rows × cols], both row-major.  One thread
// block per row; threads cooperate on the inner sum via blockReduceSum.
//
// This is the final step of the Adafactor row-sum derivation:
//   Σ_j G[i,j]² = diag(L · (R^T R) · L^T)[i]
//                = (L · S) · L^T diag component
//                = Σ_k (L·S)[i,k] · L[i,k]
// with A = L·S and B = L.
// ========================================================================
__global__ void k_ovfg_adafactor_row_update(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            unsigned int rows,
                                            unsigned int cols,
                                            float beta2,
                                            float* __restrict__ out)
{
	extern __shared__ float smem[];
	const unsigned int i = blockIdx.x;
	if (i >= rows) return;

	const unsigned int tid = threadIdx.x;
	float partial = 0.0f;
	for (unsigned int k = tid; k < cols; k += blockDim.x)
	{
		const size_t off = (size_t)i * cols + k;
		partial += A[off] * B[off];
	}

	// Warp reduce, then inter-warp reduce through shared memory.
	const unsigned int lane = tid & 31u;
	const unsigned int warp = tid >> 5;
	for (int offset = 16; offset > 0; offset >>= 1)
		partial += __shfl_xor_sync(0xffffffffu, partial, offset);
	if (lane == 0) smem[warp] = partial;
	__syncthreads();

	if (warp == 0u)
	{
		const unsigned int nwarps = (blockDim.x + 31u) >> 5;
		float v = (tid < nwarps) ? smem[tid] : 0.0f;
		for (int offset = 16; offset > 0; offset >>= 1)
			v += __shfl_xor_sync(0xffffffffu, v, offset);
		if (tid == 0u)
		{
			const float prev = out[i];
			out[i] = beta2 * prev + (1.0f - beta2) * v;
		}
	}
}

// ========================================================================
// Scaled copy row-packed into an [rows × cols_out] destination at column
// offset `col_off`:
//     dst[i, col_off + j] = alpha * src[i, j]     for i<rows, j<cols_src
// Used to assemble the concatenated [√β1 L | √(1-β1) L_acc] layout.
// ========================================================================
__global__ void k_scaled_copy_into_columns(const float* __restrict__ src,
                                           unsigned int rows,
                                           unsigned int cols_src,
                                           unsigned int col_off,
                                           unsigned int cols_out,
                                           float alpha,
                                           float* __restrict__ dst)
{
	const unsigned int i = blockIdx.y;
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows || j >= cols_src) return;
	dst[(size_t)i * cols_out + (col_off + j)] = alpha * src[(size_t)i * cols_src + j];
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

// ========================================================================
// ovfg_adafactor_moments — Adafactor-style row/col 2nd moments from (L, R)
// WITHOUT materializing G = L · R^T.  See header for the derivation.
//
// Layout of the caller-provided scratch buffer:
//     [0 .. r*r)              S = R^T R      (first pass)   OR P = L^T L (second pass)
//     [r*r .. r*r + m*r)      T = L · S      (first pass)
//     [r*r + m*r .. ...)      — unused on first pass
// For the second pass we reuse the same scratch with a different meaning:
//     [0 .. r*r)              P = L^T L
//     [r*r .. r*r + n*r)      U = R · P
// Total scratch required: r*r + max(m, n) * r floats.
// ========================================================================
bool ovfg_adafactor_moments(const float* L, const float* R,
                            unsigned int m, unsigned int n,
                            unsigned int r,
                            float beta2,
                            float* c, float* d,
                            float* scratch)
{
	if (L == nullptr || R == nullptr || c == nullptr || d == nullptr ||
	    scratch == nullptr)
		return false;
	if (m == 0u || n == 0u || r == 0u) return false;

	// ---- Row (i) accumulator: c_new[i] = diag(L · (R^T R) · L^T)[i] -----
	float* S = scratch;                    // r × r
	float* T = scratch + (size_t)r * r;    // m × r

	// S = R^T · R.  R is [n × r]; result [r × r] row-major.
	if (!sgemm_rowmajor_atb(static_cast<int>(r), static_cast<int>(r), static_cast<int>(n),
	                        1.0f,
	                        R, static_cast<int>(r),
	                        R, static_cast<int>(r),
	                        0.0f,
	                        S, static_cast<int>(r)))
		return false;

	// T = L · S.  L is [m × r], S is [r × r]; result [m × r].
	if (!sgemm_rowmajor(static_cast<int>(m), static_cast<int>(r), static_cast<int>(r),
	                    1.0f,
	                    L, static_cast<int>(r),
	                    S, static_cast<int>(r),
	                    0.0f,
	                    T, static_cast<int>(r)))
		return false;

	// c[i] ← β2 c[i] + (1-β2) · Σ_k T[i,k] · L[i,k].
	{
		const unsigned int block = 128;
		const unsigned int nwarps = (block + 31u) >> 5;
		const size_t smemBytes = nwarps * sizeof(float);
		k_ovfg_adafactor_row_update<<<m, block, smemBytes, computeStream()>>>(
		    T, L, m, r, beta2, c);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// ---- Column (j) accumulator: d_new[j] = diag(R · (L^T L) · R^T)[j] --
	float* P = scratch;                    // r × r
	float* U = scratch + (size_t)r * r;    // n × r  (overlaps old T — reused safely)

	if (!sgemm_rowmajor_atb(static_cast<int>(r), static_cast<int>(r), static_cast<int>(m),
	                        1.0f,
	                        L, static_cast<int>(r),
	                        L, static_cast<int>(r),
	                        0.0f,
	                        P, static_cast<int>(r)))
		return false;
	if (!sgemm_rowmajor(static_cast<int>(n), static_cast<int>(r), static_cast<int>(r),
	                    1.0f,
	                    R, static_cast<int>(r),
	                    P, static_cast<int>(r),
	                    0.0f,
	                    U, static_cast<int>(r)))
		return false;
	{
		const unsigned int block = 128;
		const unsigned int nwarps = (block + 31u) >> 5;
		const size_t smemBytes = nwarps * sizeof(float);
		k_ovfg_adafactor_row_update<<<n, block, smemBytes, computeStream()>>>(
		    U, R, n, r, beta2, d);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	return true;
}

// ========================================================================
// ovfg_first_moment_append — assemble L_new = [√β1·L | √(1-β1)·L_acc]
// and R_new = [√β1·R | √(1-β1)·R_acc].  Pure copy + scale — no GEMM.
// ========================================================================
bool ovfg_first_moment_append(const float* L, const float* R,
                              unsigned int r,
                              const float* L_acc, const float* R_acc,
                              unsigned int r_acc,
                              unsigned int m, unsigned int n,
                              float beta1,
                              float* L_new, float* R_new)
{
	if (L_acc == nullptr || R_acc == nullptr ||
	    L_new == nullptr || R_new == nullptr)
		return false;
	if (m == 0u || n == 0u) return false;
	if (r == 0u && r_acc == 0u) return false;

	const unsigned int r_total = r + r_acc;
	const float s1 = std::sqrt(beta1);
	const float s2 = std::sqrt(1.0f - beta1);

	dim3 block(256);

	// Copy existing L[:, :r] → L_new[:, :r] scaled by s1.
	if (r > 0u && L != nullptr)
	{
		dim3 grid((r + block.x - 1u) / block.x, m);
		k_scaled_copy_into_columns<<<grid, block, 0, computeStream()>>>(
		    L, m, r, /*col_off=*/0u, r_total, s1, L_new);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	// Copy L_acc → L_new[:, r:r+r_acc] scaled by s2.
	if (r_acc > 0u)
	{
		dim3 grid((r_acc + block.x - 1u) / block.x, m);
		k_scaled_copy_into_columns<<<grid, block, 0, computeStream()>>>(
		    L_acc, m, r_acc, /*col_off=*/r, r_total, s2, L_new);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// Same for R.
	if (r > 0u && R != nullptr)
	{
		dim3 grid((r + block.x - 1u) / block.x, n);
		k_scaled_copy_into_columns<<<grid, block, 0, computeStream()>>>(
		    R, n, r, /*col_off=*/0u, r_total, s1, R_new);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	if (r_acc > 0u)
	{
		dim3 grid((r_acc + block.x - 1u) / block.x, n);
		k_scaled_copy_into_columns<<<grid, block, 0, computeStream()>>>(
		    R_acc, n, r_acc, /*col_off=*/r, r_total, s2, R_new);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
