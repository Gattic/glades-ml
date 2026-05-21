// OVFG (Operator-Valued Factored Gradient) GPU primitives.
// See gpu_ovfg.h and research/PARADIGM_SHIFT_9_CANDIDATE_C_OVFG.md.

#include "gpu_ovfg.h"
#include "gpu_blas.h"
#include "gpu_device.h"
#include "gpu_stiefel.h"
#include <cusolverDn.h>

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

// ========================================================================
// Column-broadcast scale: out[i, j] = in[i, j] * sigma[j].
// Used for dU_raw ← dU_raw · diag(Σ), same layout as Stiefel's
// k_scale_cols_by_diag.  One 2D grid (rows × ceil(cols/block)).
// ========================================================================
__global__ void k_ovfg_scale_cols_by_diag(float* __restrict__ M,
                                          const float* __restrict__ sigma,
                                          unsigned int rows,
                                          unsigned int cols)
{
	const unsigned int i = blockIdx.x;
	const unsigned int j = blockIdx.y * blockDim.x + threadIdx.x;
	if (i >= rows || j >= cols) return;
	M[(size_t)i * cols + j] *= sigma[j];
}

// ========================================================================
// Column-broadcast scaled copy with sqrt: dst[i, j] = src[i, j] * √diag[j].
// Both src and dst are row-major (rows × cols).  Used to assemble
// L_out = U_col · diag(√Σ) and R_out = V_col · diag(√Σ) in the
// truncate_factors path.
// ========================================================================
__global__ void k_ovfg_scale_cols_by_sqrt_diag_copy(
    const float* __restrict__ src,
    const float* __restrict__ diag,
    unsigned int rows,
    unsigned int cols,
    float* __restrict__ dst)
{
	const unsigned int i = blockIdx.x;
	const unsigned int j = blockIdx.y * blockDim.x + threadIdx.x;
	if (i >= rows || j >= cols) return;
	const float d = diag[j];
	const float s = d > 0.0f ? sqrtf(d) : 0.0f;
	dst[(size_t)i * cols + j] = src[(size_t)i * cols + j] * s;
}

// ========================================================================
// Extract the first `r_out` columns from a row-major (rows × r_in)
// matrix into a (rows × r_out) row-major matrix.  Used to truncate U_K
// / V_K after SVD.
// ========================================================================
__global__ void k_ovfg_copy_first_cols(const float* __restrict__ src,
                                       unsigned int rows,
                                       unsigned int r_in,
                                       unsigned int r_out,
                                       float* __restrict__ dst)
{
	const unsigned int i = blockIdx.x;
	const unsigned int j = blockIdx.y * blockDim.x + threadIdx.x;
	if (i >= rows || j >= r_out) return;
	dst[(size_t)i * r_out + j] = src[(size_t)i * r_in + j];
}

// ========================================================================
// Extract the upper r × r triangle of a [rows × r] column-major matrix
// (as produced by cusolverDnSgeqrf, where the factor R occupies the
// upper triangle of A_col with Householder reflectors below the diag).
// Writes R as a row-major [r × r] matrix with strictly-below-diagonal
// entries zeroed.
//
// Critical: A_col has column-major LEADING DIMENSION = rows (not r),
// so the column-major offset of element (i, j) with i in [0, r), j in
// [0, r) is j*rows + i.
// ========================================================================
__global__ void k_ovfg_extract_upper_triangle_colmajor_to_rowmajor(
    const float* __restrict__ A_col,
    unsigned int rows,
    float* __restrict__ R_row,
    unsigned int r)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // row
	const unsigned int j = blockIdx.y * blockDim.x + threadIdx.y; // col
	if (i >= r || j >= r) return;
	float v = (j >= i) ? A_col[(size_t)j * rows + i] : 0.0f;
	R_row[(size_t)i * r + j] = v;
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

// ========================================================================
// ovfg_stiefel_tangent_grad — Phase 3 payoff.
// Derivation is documented in gpu_ovfg.h.
// ========================================================================
bool ovfg_stiefel_tangent_grad(const GpuStiefelWeight& s,
                               const float* L, const float* R,
                               unsigned int r,
                               float* dU, float* dSigma, float* dV,
                               float* scratch)
{
	if (!s.allocated()) return false;
	if (L == nullptr || R == nullptr || scratch == nullptr) return false;
	if (dU == nullptr || dSigma == nullptr || dV == nullptr) return false;
	if (r == 0u) return false;

	// Refresh U, V FP32 cache — OVFG is an entry-point primitive like
	// stiefel_dense_grad_to_tangent.
	stiefel_refresh_fp32_cache(s);

	const unsigned int m   = s.m;
	const unsigned int n   = s.n;
	const unsigned int rho = s.r;                      // Stiefel rank
	const float* U_f32 = s.U_f32_cache.data();
	const float* V_f32 = s.V_f32_cache.data();
	const float* sig   = s.sigma.data();

	// Scratch layout:
	//   A           [rho × r]
	//   B           [rho × r]
	//   scratch_rr  [rho × rho]   (first  tangent-projection workspace)
	//   scratch_rr2 [rho × rho]   (second tangent-projection workspace)
	float* A           = scratch;
	float* B           = A           + (size_t)rho * r;
	float* scratch_rr  = B           + (size_t)rho * r;
	float* scratch_rr2 = scratch_rr  + (size_t)rho * rho;

	// (1) A = U^T · L       [rho × r]  via sgemm_rowmajor_atb (M=rho, N=r, K=m)
	if (!sgemm_rowmajor_atb(static_cast<int>(rho), static_cast<int>(r), static_cast<int>(m),
	                        1.0f,
	                        U_f32, static_cast<int>(rho),
	                        L,     static_cast<int>(r),
	                        0.0f,
	                        A,     static_cast<int>(r)))
		return false;

	// (2) B = V^T · R       [rho × r]
	if (!sgemm_rowmajor_atb(static_cast<int>(rho), static_cast<int>(r), static_cast<int>(n),
	                        1.0f,
	                        V_f32, static_cast<int>(rho),
	                        R,     static_cast<int>(r),
	                        0.0f,
	                        B,     static_cast<int>(r)))
		return false;

	// (3) dΣ[i] = Σ_k A[i,k] · B[i,k].  Reuse the row-dot kernel with
	//     beta2 = 0 so it produces the raw dot (no EMA blend).
	{
		const unsigned int block = 128;
		const unsigned int nwarps = (block + 31u) >> 5;
		const size_t smemBytes = nwarps * sizeof(float);
		k_ovfg_adafactor_row_update<<<rho, block, smemBytes, computeStream()>>>(
		    A, B, rho, r, 0.0f, dSigma);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// (4) dU_raw = L · B^T       [m × rho]  via sgemm_abt (M=m, N=rho, K=r).
	if (!sgemm_rowmajor_abt(static_cast<int>(m), static_cast<int>(rho), static_cast<int>(r),
	                        1.0f,
	                        L, static_cast<int>(r),
	                        B, static_cast<int>(r),
	                        0.0f,
	                        dU, static_cast<int>(rho)))
		return false;
	// dU_raw ← dU_raw · diag(Σ)   (column-broadcast scale)
	{
		dim3 block(64);
		dim3 grid(m, (rho + block.x - 1u) / block.x);
		k_ovfg_scale_cols_by_diag<<<grid, block, 0, computeStream()>>>(
		    dU, sig, m, rho);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// (5) dV_raw = R · A^T       [n × rho]  via sgemm_abt (M=n, N=rho, K=r).
	if (!sgemm_rowmajor_abt(static_cast<int>(n), static_cast<int>(rho), static_cast<int>(r),
	                        1.0f,
	                        R, static_cast<int>(r),
	                        A, static_cast<int>(r),
	                        0.0f,
	                        dV, static_cast<int>(rho)))
		return false;
	// dV_raw ← dV_raw · diag(Σ)
	{
		dim3 block(64);
		dim3 grid(n, (rho + block.x - 1u) / block.x);
		k_ovfg_scale_cols_by_diag<<<grid, block, 0, computeStream()>>>(
		    dV, sig, n, rho);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// (6) Finish the Stiefel tangent projection: dU, dV → proj(dU, dV).
	// This is exactly the step that stiefel_dense_grad_to_tangent calls
	// at its tail; reusing it guarantees byte-identical outputs.
	stiefel_tangent_project_grad(s, dU, dV, scratch_rr, scratch_rr2);

	return true;
}

// ========================================================================
// ovfg_stiefel_unconstrained_grad — RAW (pre-projection) variant, matches
// stiefel_backward_unconstrained byte-for-byte.  Intended to feed
// stiefel_adam_step which applies tangent projection as its first step.
//
// Scratch: 2·ρ·r floats for A, B (no r×r scratches needed).
// ========================================================================
bool ovfg_stiefel_unconstrained_grad(const GpuStiefelWeight& s,
                                     const float* L, const float* R,
                                     unsigned int r,
                                     float* dU, float* dSigma, float* dV,
                                     float* scratch)
{
	if (!s.allocated()) return false;
	if (L == nullptr || R == nullptr || scratch == nullptr) return false;
	if (dU == nullptr || dSigma == nullptr || dV == nullptr) return false;
	if (r == 0u) return false;

	stiefel_refresh_fp32_cache(s);

	const unsigned int m   = s.m;
	const unsigned int n   = s.n;
	const unsigned int rho = s.r;
	const float* U_f32 = s.U_f32_cache.data();
	const float* V_f32 = s.V_f32_cache.data();
	const float* sig   = s.sigma.data();

	float* A = scratch;                    // [rho × r]
	float* B = A + (size_t)rho * r;        // [rho × r]

	if (!sgemm_rowmajor_atb(static_cast<int>(rho), static_cast<int>(r), static_cast<int>(m),
	                        1.0f,
	                        U_f32, static_cast<int>(rho),
	                        L,     static_cast<int>(r),
	                        0.0f,
	                        A,     static_cast<int>(r)))
		return false;
	if (!sgemm_rowmajor_atb(static_cast<int>(rho), static_cast<int>(r), static_cast<int>(n),
	                        1.0f,
	                        V_f32, static_cast<int>(rho),
	                        R,     static_cast<int>(r),
	                        0.0f,
	                        B,     static_cast<int>(r)))
		return false;

	{
		const unsigned int block = 128;
		const unsigned int nwarps = (block + 31u) >> 5;
		const size_t smemBytes = nwarps * sizeof(float);
		k_ovfg_adafactor_row_update<<<rho, block, smemBytes, computeStream()>>>(
		    A, B, rho, r, 0.0f, dSigma);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	if (!sgemm_rowmajor_abt(static_cast<int>(m), static_cast<int>(rho), static_cast<int>(r),
	                        1.0f,
	                        L, static_cast<int>(r),
	                        B, static_cast<int>(r),
	                        0.0f,
	                        dU, static_cast<int>(rho)))
		return false;
	{
		dim3 block(64);
		dim3 grid(m, (rho + block.x - 1u) / block.x);
		k_ovfg_scale_cols_by_diag<<<grid, block, 0, computeStream()>>>(
		    dU, sig, m, rho);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	if (!sgemm_rowmajor_abt(static_cast<int>(n), static_cast<int>(rho), static_cast<int>(r),
	                        1.0f,
	                        R, static_cast<int>(r),
	                        A, static_cast<int>(r),
	                        0.0f,
	                        dV, static_cast<int>(rho)))
		return false;
	{
		dim3 block(64);
		dim3 grid(n, (rho + block.x - 1u) / block.x);
		k_ovfg_scale_cols_by_diag<<<grid, block, 0, computeStream()>>>(
		    dV, sig, n, rho);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// No tangent projection — caller (typically stiefel_adam_step) applies.
	return true;
}

// ========================================================================
// File-local cuSOLVER state for OVFG.  Kept separate from Stiefel's
// solver workspace so the two modules don't contend for cached
// allocations during concurrent backward passes.
// ========================================================================
namespace {
cusolverDnHandle_t g_ovfgSolver = nullptr;
bool g_ovfgSolverReady = false;
int* g_ovfgInfo = nullptr;

bool ovfg_solver_init()
{
	if (!g_ovfgSolverReady)
	{
		if (cusolverDnCreate(&g_ovfgSolver) != CUSOLVER_STATUS_SUCCESS)
			return false;
		g_ovfgSolverReady = true;
	}
	cusolverDnSetStream(g_ovfgSolver, computeStream());
	if (g_ovfgInfo == nullptr)
	{
		if (cudaMalloc(&g_ovfgInfo, sizeof(int)) != cudaSuccess) return false;
	}
	return true;
}
} // anonymous namespace

// ========================================================================
// ovfg_truncate_factors — PHASE 2b naive dense-SVD path.
//
// See gpu_ovfg.h for full API + rationale.  Current implementation:
//
//   (1) Materialize G = L · R^T  (m × n, row-major).  This is the
//       costly step that Phase 2c will eliminate.
//   (2) Transpose G to column-major (cuSOLVER convention).
//   (3) cusolverDnSgesvd(G_col) → U_col (m × m), Σ (min(m,n)), V^T_col (n × n).
//   (4) Transpose U_col → U (m × m, row-major).  V is obtained by
//       transposing V^T_col once (VT_col is [n × n] col-major = [n × n]
//       row-major of V itself, no transpose needed).
//   (5) Truncate: take top r_out columns of U and of V; take first r_out
//       entries of Σ.
//   (6) L_out = U_trunc · diag(√Σ_trunc)     (m × r_out)
//       R_out = V_trunc · diag(√Σ_trunc)     (n × r_out)
//       ⇒ L_out · R_out^T = U Σ V^T restricted to top r_out = best
//       rank-r_out approximation of G by Eckart–Young.
// ========================================================================
bool ovfg_truncate_factors(const float* L, const float* R,
                           unsigned int m, unsigned int n,
                           unsigned int r_in, unsigned int r_out,
                           float* L_out, float* R_out,
                           float* scratch)
{
	if (L == nullptr || R == nullptr || L_out == nullptr || R_out == nullptr)
		return false;
	if (scratch == nullptr) return false;
	if (m == 0u || n == 0u || r_in == 0u || r_out == 0u) return false;
	if (r_out > r_in) return false;
	if (!ovfg_solver_init()) return false;

	const unsigned int k_full = m < n ? m : n;
	if (r_out > k_full) return false;

	// Scratch layout (row-major unless noted):
	//   G       [m × n]
	//   G_col   [m × n] column-major (staging for SVD)
	//   U_col   [m × m] column-major (SVD output)
	//   Sigma   [k_full]
	//   VT_col  [n × n] column-major (= V row-major)
	//   U_rm    [m × m] row-major staging for truncation copy
	const size_t off_G     = 0;
	const size_t off_G_col = off_G   + (size_t)m * n;
	const size_t off_U_col = off_G_col + (size_t)m * n;
	const size_t off_Sigma = off_U_col + (size_t)m * m;
	const size_t off_VT    = off_Sigma + k_full;
	const size_t off_U_rm  = off_VT    + (size_t)n * n;

	float* G      = scratch + off_G;
	float* G_col  = scratch + off_G_col;
	float* U_col  = scratch + off_U_col;
	float* Sigma  = scratch + off_Sigma;
	float* VT_col = scratch + off_VT;
	float* U_rm   = scratch + off_U_rm;

	// (1) G = L · R^T using the existing parity helper.
	if (!ovfg_compute_dense_from_factors(L, R, m, n, r_in, G))
		return false;

	// (2) Transpose G (row-major [m × n]) to column-major [m × n] for cuSOLVER.
	// Column-major [m × n] with leading-dim m is the row-major transpose.
	{
		dim3 block(32, 32);
		dim3 grid((n + 31u) / 32u, (m + 31u) / 32u);
		k_transpose_2d<<<grid, block, 0, computeStream()>>>(G, G_col, m, n);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// (3) Query SVD workspace.
	int lwork = 0;
	if (cusolverDnSgesvd_bufferSize(g_ovfgSolver, m, n, &lwork)
	    != CUSOLVER_STATUS_SUCCESS)
		return false;
	static thread_local float* svd_work = nullptr;
	static thread_local size_t svd_work_cap = 0;
	if (static_cast<size_t>(lwork) > svd_work_cap)
	{
		if (svd_work) cudaFree(svd_work);
		if (cudaMalloc(&svd_work, lwork * sizeof(float)) != cudaSuccess)
			return false;
		svd_work_cap = lwork;
	}

	// (4) Full SVD (jobu='A', jobvt='A' → full m×m U and n×n V^T).
	cusolverStatus_t st = cusolverDnSgesvd(
	    g_ovfgSolver, 'A', 'A', m, n, G_col, m,
	    Sigma, U_col, m, VT_col, n,
	    svd_work, lwork, nullptr, g_ovfgInfo);
	if (st != CUSOLVER_STATUS_SUCCESS) return false;
	int host_info = 0;
	cudaMemcpyAsync(&host_info, g_ovfgInfo, sizeof(int),
	                cudaMemcpyDeviceToHost, computeStream());
	cudaStreamSynchronize(computeStream());
	if (host_info != 0) return false;

	// (5) Transpose U_col ([m × m] col-major = [m × m] row-major of U^T)
	// to U_rm (row-major U).
	// Actually: column-major element U_col[j * m + i] stores U[i, j] in the
	// mathematical sense (cuSOLVER's U with left singular vectors as cols).
	// In row-major layout, U[i, j] = U_col[j * m + i].  So the row-major
	// U has the same bytes but differently-indexed.  We just need to
	// "transpose" col-major to row-major — which is a plain transpose of
	// the (m × m) buffer.
	{
		dim3 block(32, 32);
		dim3 grid((m + 31u) / 32u, (m + 31u) / 32u);
		k_transpose_2d<<<grid, block, 0, computeStream()>>>(U_col, U_rm, m, m);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// VT_col is column-major [n × n], i.e., row-major V^T has element
	// V^T[i, j] = VT_col[j * n + i].  We want V (row-major, [n × n]),
	// which equals (V^T)^T, i.e., V[i, j] = V^T[j, i] = VT_col[i * n + j].
	// That means VT_col reinterpreted as row-major IS V row-major already.
	// No transpose needed — we can use VT_col directly as V_rm.
	float* V_rm = VT_col;

	// (6) Truncate to first r_out columns of U_rm (m × r_out) and first
	// r_out columns of V_rm (n × r_out); then scale by √Σ.
	// U_rm is (m × m); copy columns [0, r_out) into L_out (m × r_out)
	// first, then scale by √Σ in place (writing into L_out).
	// We use two kernels: copy_first_cols → scale_cols_by_sqrt_diag_copy.
	// Simpler: one fused kernel that reads U_rm and writes L_out scaled.
	{
		dim3 block(64);
		dim3 grid(m, (r_out + block.x - 1u) / block.x);
		k_ovfg_scale_cols_by_sqrt_diag_copy<<<grid, block, 0, computeStream()>>>(
		    U_rm, Sigma, m, r_out, L_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	// But we need the first r_out cols of U_rm (which is m × m).  The
	// kernel above reads U_rm[i, j] at offset i*m + j, but we want the
	// (m × r_out) strided read i*m + j for j < r_out.  The kernel uses
	// "cols" for the dst (r_out), but for src reads cols = r_out too.
	// That's WRONG — we'd read the wrong elements.  Fix by copying first.
	// Actually re-examining: the kernel does
	//     dst[i * cols + j] = src[i * cols + j] * √diag[j]
	// With cols = r_out and src = U_rm (m × m), that reads U_rm[i * r_out + j],
	// which for j < r_out is NOT the first r_out columns of U_rm (it would
	// be the first r_out of row i in a r_out-wide layout, not m-wide).
	// The right answer: first copy-compact U_rm[:, :r_out] into a temp,
	// then scale.  Let's do that cleanly with k_ovfg_copy_first_cols.

	// Redo: overwrite L_out with a compact copy of U_rm[:, :r_out].
	{
		dim3 block(64);
		dim3 grid(m, (r_out + block.x - 1u) / block.x);
		k_ovfg_copy_first_cols<<<grid, block, 0, computeStream()>>>(
		    U_rm, m, m, r_out, L_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	// Now L_out has shape (m × r_out) and contains U[:, :r_out].
	// Scale columns by √Σ in place (src = dst = L_out, cols = r_out).
	{
		dim3 block(64);
		dim3 grid(m, (r_out + block.x - 1u) / block.x);
		k_ovfg_scale_cols_by_sqrt_diag_copy<<<grid, block, 0, computeStream()>>>(
		    L_out, Sigma, m, r_out, L_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// Same for R_out = V[:, :r_out] · diag(√Σ).
	{
		dim3 block(64);
		dim3 grid(n, (r_out + block.x - 1u) / block.x);
		k_ovfg_copy_first_cols<<<grid, block, 0, computeStream()>>>(
		    V_rm, n, n, r_out, R_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	{
		dim3 block(64);
		dim3 grid(n, (r_out + block.x - 1u) / block.x);
		k_ovfg_scale_cols_by_sqrt_diag_copy<<<grid, block, 0, computeStream()>>>(
		    R_out, Sigma, n, r_out, R_out);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	return true;
}

// ========================================================================
// Thin QR factorization helper for OVFG.  Given a row-major [rows × r]
// matrix A, writes:
//   Q_row   [rows × r] row-major — orthonormal columns of A's column span
//   R_row   [r × r]    row-major — upper-triangular factor
// via cuSOLVER's geqrf + orgqr path (column-major internally).
//
// Workspace pointers (all caller-owned):
//   A_col      [rows × r]  — column-major staging (overwritten)
//   tau        [r]          — Householder scalars
//   qr_work    [qr_lwork]   — cuSOLVER workspace
//   slot       0 or 1       — selects info pointer (prevents contention)
//
// Returns false on cuSOLVER failure or OOM.
// ========================================================================
namespace {
int*  g_ovfgQrInfo[2] = {nullptr, nullptr};
float* g_ovfgQrWork[2] = {nullptr, nullptr};
size_t g_ovfgQrWorkCap[2] = {0, 0};

bool ovfg_thin_qr(const float* A_row, unsigned int rows, unsigned int r,
                  float* Q_row, float* R_row,
                  float* A_col, float* tau,
                  int slot)
{
	if (!ovfg_solver_init()) return false;
	if (slot < 0 || slot > 1) return false;
	if (g_ovfgQrInfo[slot] == nullptr)
	{
		if (cudaMalloc(&g_ovfgQrInfo[slot], sizeof(int)) != cudaSuccess)
			return false;
	}

	// Row-major [rows × r] → column-major [rows × r].
	// Column-major (rows, r) leading-dim rows has element (i, j) at j*rows+i;
	// row-major (rows, r) has element (i, j) at i*r+j.  So we need a
	// row-major ↔ column-major transpose, which k_transpose_2d handles.
	{
		dim3 block(32, 32);
		dim3 grid((r + 31u) / 32u, (rows + 31u) / 32u);
		k_transpose_2d<<<grid, block, 0, computeStream()>>>(A_row, A_col, rows, r);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// Query workspace.
	int lwork_geqrf = 0, lwork_orgqr = 0;
	if (cusolverDnSgeqrf_bufferSize(g_ovfgSolver, rows, r, A_col, rows, &lwork_geqrf)
	    != CUSOLVER_STATUS_SUCCESS)
		return false;
	if (cusolverDnSorgqr_bufferSize(g_ovfgSolver, rows, r, r, A_col, rows, tau, &lwork_orgqr)
	    != CUSOLVER_STATUS_SUCCESS)
		return false;
	const int lwork = lwork_geqrf > lwork_orgqr ? lwork_geqrf : lwork_orgqr;
	if (static_cast<size_t>(lwork) > g_ovfgQrWorkCap[slot])
	{
		if (g_ovfgQrWork[slot]) cudaFree(g_ovfgQrWork[slot]);
		if (cudaMalloc(&g_ovfgQrWork[slot], lwork * sizeof(float)) != cudaSuccess)
			return false;
		g_ovfgQrWorkCap[slot] = lwork;
	}

	// QR.  After this, A_col upper-triangle has R and below-diagonal
	// has Householder reflectors; tau has scalars.
	cusolverStatus_t st = cusolverDnSgeqrf(g_ovfgSolver, rows, r, A_col, rows, tau,
	                                       g_ovfgQrWork[slot], lwork, g_ovfgQrInfo[slot]);
	if (st != CUSOLVER_STATUS_SUCCESS) return false;
	int host_info = 0;
	cudaMemcpyAsync(&host_info, g_ovfgQrInfo[slot], sizeof(int),
	                cudaMemcpyDeviceToHost, computeStream());
	cudaStreamSynchronize(computeStream());
	if (host_info != 0) return false;

	// Extract R (upper triangle, col-major → row-major).
	{
		dim3 block(16, 16);
		dim3 grid((r + 15u) / 16u, (r + 15u) / 16u);
		k_ovfg_extract_upper_triangle_colmajor_to_rowmajor<<<grid, block, 0, computeStream()>>>(
		    A_col, rows, R_row, r);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// Build Q explicitly from Householder representation.
	st = cusolverDnSorgqr(g_ovfgSolver, rows, r, r, A_col, rows, tau,
	                      g_ovfgQrWork[slot], lwork, g_ovfgQrInfo[slot]);
	if (st != CUSOLVER_STATUS_SUCCESS) return false;
	cudaMemcpyAsync(&host_info, g_ovfgQrInfo[slot], sizeof(int),
	                cudaMemcpyDeviceToHost, computeStream());
	cudaStreamSynchronize(computeStream());
	if (host_info != 0) return false;

	// A_col now holds Q in column-major [rows × r].  Transpose to row-major.
	{
		dim3 block(32, 32);
		dim3 grid((rows + 31u) / 32u, (r + 31u) / 32u);
		k_transpose_2d<<<grid, block, 0, computeStream()>>>(A_col, Q_row, r, rows);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	return true;
}
} // anonymous namespace

// ========================================================================
// ovfg_truncate_factors_qr — PHASE 2c efficient truncation (no m×n ever).
// See gpu_ovfg.h for the algorithm sketch; implementation below.
//
// Scratch layout (floats):
//   Q_L            [m × r_in]
//   Q_R            [n × r_in]
//   R_L            [r_in × r_in]  (row-major)
//   R_R            [r_in × r_in]  (row-major)
//   K              [r_in × r_in]  (row-major, R_L · R_R^T)
//   K_col          [r_in × r_in]  (column-major staging for SVD)
//   U_K_col        [r_in × r_in]  (SVD out, col-major)
//   VT_K_col       [r_in × r_in]  (SVD out, col-major = V_K row-major)
//   Sigma_K        [r_in]
//   U_K_row_full   [r_in × r_in]  (row-major transpose of U_K_col)
//   A_col_L        [m × r_in]     (QR scratch, shared with A_col_R via slot)
//   A_col_R        [n × r_in]
//   tau_L          [r_in]
//   tau_R          [r_in]
// Total: 2(m+n)r_in + 5 r_in² + 3 r_in.
// ========================================================================
bool ovfg_truncate_factors_qr(const float* L, const float* R,
                              unsigned int m, unsigned int n,
                              unsigned int r_in, unsigned int r_out,
                              float* L_out, float* R_out,
                              float* scratch)
{
	if (L == nullptr || R == nullptr || L_out == nullptr || R_out == nullptr)
		return false;
	if (scratch == nullptr) return false;
	if (m == 0u || n == 0u || r_in == 0u || r_out == 0u) return false;
	if (r_out > r_in) return false;
	if (r_in > m || r_in > n) return false;  // thin QR requires r_in ≤ min(m,n)
	if (!ovfg_solver_init()) return false;

	const size_t r2 = (size_t)r_in * r_in;
	size_t off = 0;
	float* Q_L          = scratch + off; off += (size_t)m * r_in;
	float* Q_R          = scratch + off; off += (size_t)n * r_in;
	float* R_L          = scratch + off; off += r2;
	float* R_R          = scratch + off; off += r2;
	float* K            = scratch + off; off += r2;
	float* K_col        = scratch + off; off += r2;
	float* U_K_col      = scratch + off; off += r2;
	float* VT_K_col     = scratch + off; off += r2;
	float* Sigma_K      = scratch + off; off += r_in;
	float* U_K_row_full = scratch + off; off += r2;
	float* A_col_L      = scratch + off; off += (size_t)m * r_in;
	float* A_col_R      = scratch + off; off += (size_t)n * r_in;
	float* tau_L        = scratch + off; off += r_in;
	float* tau_R        = scratch + off; off += r_in;
	(void)off;

	// (1) Thin QR of L.
	if (!ovfg_thin_qr(L, m, r_in, Q_L, R_L, A_col_L, tau_L, /*slot=*/0))
		return false;
	// (2) Thin QR of R.
	if (!ovfg_thin_qr(R, n, r_in, Q_R, R_R, A_col_R, tau_R, /*slot=*/1))
		return false;

	// (3) K = R_L · R_R^T   [r_in × r_in]
	if (!sgemm_rowmajor_abt(static_cast<int>(r_in), static_cast<int>(r_in), static_cast<int>(r_in),
	                        1.0f,
	                        R_L, static_cast<int>(r_in),
	                        R_R, static_cast<int>(r_in),
	                        0.0f,
	                        K, static_cast<int>(r_in)))
		return false;

	// (4) SVD of small K.  Transpose to column-major, cusolverDnSgesvd.
	{
		dim3 block(32, 32);
		dim3 grid((r_in + 31u) / 32u, (r_in + 31u) / 32u);
		k_transpose_2d<<<grid, block, 0, computeStream()>>>(K, K_col, r_in, r_in);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	int lwork = 0;
	if (cusolverDnSgesvd_bufferSize(g_ovfgSolver, r_in, r_in, &lwork)
	    != CUSOLVER_STATUS_SUCCESS)
		return false;
	static thread_local float* svd_work_qr = nullptr;
	static thread_local size_t svd_work_qr_cap = 0;
	if (static_cast<size_t>(lwork) > svd_work_qr_cap)
	{
		if (svd_work_qr) cudaFree(svd_work_qr);
		if (cudaMalloc(&svd_work_qr, lwork * sizeof(float)) != cudaSuccess)
			return false;
		svd_work_qr_cap = lwork;
	}
	cusolverStatus_t st = cusolverDnSgesvd(
	    g_ovfgSolver, 'A', 'A', r_in, r_in, K_col, r_in,
	    Sigma_K, U_K_col, r_in, VT_K_col, r_in,
	    svd_work_qr, lwork, nullptr, g_ovfgInfo);
	if (st != CUSOLVER_STATUS_SUCCESS) return false;
	int host_info = 0;
	cudaMemcpyAsync(&host_info, g_ovfgInfo, sizeof(int),
	                cudaMemcpyDeviceToHost, computeStream());
	cudaStreamSynchronize(computeStream());
	if (host_info != 0) return false;

	// (5) Transpose U_K_col (col-major [r_in × r_in] = U_K^T row-major) to
	// U_K_row_full (row-major U_K).
	{
		dim3 block(32, 32);
		dim3 grid((r_in + 31u) / 32u, (r_in + 31u) / 32u);
		k_transpose_2d<<<grid, block, 0, computeStream()>>>(U_K_col, U_K_row_full, r_in, r_in);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	// VT_K_col reinterpreted row-major = V_K row-major (as in the 2b path).
	float* V_K_row_full = VT_K_col;

	// (6) Form truncated factors directly into L_out and R_out:
	//   L_out = Q_L · (U_K[:, :r_out] · diag(√Σ[:r_out]))
	//   R_out = Q_R · (V_K[:, :r_out] · diag(√Σ[:r_out]))
	//
	// We need intermediate (r_in × r_out) scaled matrices.  Reuse R_L
	// and R_R buffers (no longer needed after K was formed).
	float* U_K_trunc = R_L;   // [r_in × r_out], reuse r_in² scratch
	float* V_K_trunc = R_R;
	// Copy first r_out columns, then scale.
	{
		dim3 block(64);
		dim3 grid(r_in, (r_out + block.x - 1u) / block.x);
		k_ovfg_copy_first_cols<<<grid, block, 0, computeStream()>>>(
		    U_K_row_full, r_in, r_in, r_out, U_K_trunc);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	{
		dim3 block(64);
		dim3 grid(r_in, (r_out + block.x - 1u) / block.x);
		k_ovfg_scale_cols_by_sqrt_diag_copy<<<grid, block, 0, computeStream()>>>(
		    U_K_trunc, Sigma_K, r_in, r_out, U_K_trunc);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	{
		dim3 block(64);
		dim3 grid(r_in, (r_out + block.x - 1u) / block.x);
		k_ovfg_copy_first_cols<<<grid, block, 0, computeStream()>>>(
		    V_K_row_full, r_in, r_in, r_out, V_K_trunc);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	{
		dim3 block(64);
		dim3 grid(r_in, (r_out + block.x - 1u) / block.x);
		k_ovfg_scale_cols_by_sqrt_diag_copy<<<grid, block, 0, computeStream()>>>(
		    V_K_trunc, Sigma_K, r_in, r_out, V_K_trunc);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// L_out = Q_L · U_K_trunc:  (m × r_in) · (r_in × r_out) → (m × r_out).
	if (!sgemm_rowmajor(static_cast<int>(m), static_cast<int>(r_out), static_cast<int>(r_in),
	                    1.0f,
	                    Q_L,       static_cast<int>(r_in),
	                    U_K_trunc, static_cast<int>(r_out),
	                    0.0f,
	                    L_out,     static_cast<int>(r_out)))
		return false;
	// R_out = Q_R · V_K_trunc:  (n × r_in) · (r_in × r_out) → (n × r_out).
	if (!sgemm_rowmajor(static_cast<int>(n), static_cast<int>(r_out), static_cast<int>(r_in),
	                    1.0f,
	                    Q_R,       static_cast<int>(r_in),
	                    V_K_trunc, static_cast<int>(r_out),
	                    0.0f,
	                    R_out,     static_cast<int>(r_out)))
		return false;

	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
