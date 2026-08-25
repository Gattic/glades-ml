// MPOT (Matrix Product Operator weight decomposition) GPU primitives.
// See gpu_mpot.h and research/PARADIGM_SHIFT_10_DESIGN.md.

#include "gpu_mpot.h"
#include "gpu_device.h"
#include "gpu_blas.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cmath>

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

// Permute W[i, j] where i = i_1·m_2 + i_2, j = j_1·n_2 + j_2 into
// M[p, q] where p = i_1·n_1 + j_1, q = i_2·n_2 + j_2.  Output M is
// row-major with dims (m_1·n_1, m_2·n_2).  One thread per output cell.
__global__ void k_mpot_index_permute(const float* __restrict__ W,
                                     unsigned int m_1, unsigned int m_2,
                                     unsigned int n_1, unsigned int n_2,
                                     float* __restrict__ M)
{
	const unsigned int P = m_1 * n_1;
	const unsigned int Q = m_2 * n_2;
	const unsigned int p = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int q = blockIdx.x * blockDim.x + threadIdx.x;
	if (p >= P || q >= Q) return;

	const unsigned int i_1 = p / n_1;
	const unsigned int j_1 = p % n_1;
	const unsigned int i_2 = q / n_2;
	const unsigned int j_2 = q % n_2;
	const unsigned int i = i_1 * m_2 + i_2;
	const unsigned int j = j_1 * n_2 + j_2;
	const unsigned int n = n_1 * n_2;
	M[(size_t)p * Q + q] = W[(size_t)i * n + j];
}

// Transpose a row-major (rows × cols) matrix to column-major with
// leading-dim rows (equivalent to a row-major (cols × rows) transpose).
// Used to hand row-major matrices into cuSOLVER.  Simple non-tiled.
__global__ void k_mpot_rm_to_cm(const float* __restrict__ X,
                                unsigned int rows, unsigned int cols,
                                float* __restrict__ Y_col)
{
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows || j >= cols) return;
	// col-major: element (i, j) at offset j * rows + i.
	Y_col[(size_t)j * rows + i] = X[(size_t)i * cols + j];
}

// Transpose column-major (rows × cols) to row-major (rows × cols).
__global__ void k_mpot_cm_to_rm(const float* __restrict__ X_col,
                                unsigned int rows, unsigned int cols,
                                float* __restrict__ Y_row)
{
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows || j >= cols) return;
	Y_row[(size_t)i * cols + j] = X_col[(size_t)j * rows + i];
}

// Scatter the first D left singular vectors U_rm [P × D] into A:
//   A[i_1, j_1, α] = U_rm[i_1·n_1 + j_1, α] · √Σ[α]       for α < D
// Input U_full_rm is row-major [P × P] (only first D columns used).
__global__ void k_mpot_scatter_A(const float* __restrict__ U_full_rm,
                                 unsigned int P_ld,
                                 const float* __restrict__ Sigma,
                                 unsigned int m_1, unsigned int n_1,
                                 unsigned int D,
                                 float* __restrict__ A)
{
	const unsigned int p = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int P = m_1 * n_1;
	if (p >= P || a >= D) return;
	const unsigned int i_1 = p / n_1;
	const unsigned int j_1 = p % n_1;
	const float s = Sigma[a];
	const float sqs = s > 0.0f ? sqrtf(s) : 0.0f;
	// A[i_1, j_1, α] at row-major offset i_1*n_1*D + j_1*D + α.
	A[(size_t)i_1 * n_1 * D + (size_t)j_1 * D + a] =
	    U_full_rm[(size_t)p * P_ld + a] * sqs;
}

// Permute B from (D, m_2, n_2) row-major to (m_2, D, n_2) row-major.
// Needed so that X [T·m_1, m_2] · B_perm [m_2, D·n_2] is a standard
// matrix multiply producing T1 [T·m_1, D·n_2].
__global__ void k_mpot_perm_B_D_m2_n2__to__m2_D_n2(
    const float* __restrict__ B,
    unsigned int D, unsigned int m_2, unsigned int n_2,
    float* __restrict__ B_perm)
{
	const unsigned int alpha = blockIdx.z;
	const unsigned int i_2   = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j_2   = blockIdx.x * blockDim.x + threadIdx.x;
	if (alpha >= D || i_2 >= m_2 || j_2 >= n_2) return;
	B_perm[(size_t)i_2 * D * n_2 + (size_t)alpha * n_2 + j_2] =
	    B[(size_t)alpha * m_2 * n_2 + (size_t)i_2 * n_2 + j_2];
}

// Permute A from (m_1, n_1, D) row-major to (m_1, D, n_1) row-major.
// Needed so that T1_perm [T·n_2, m_1·D] · A_perm [m_1·D, n_1] is a
// standard matrix multiply producing Y_pre [T·n_2, n_1].
__global__ void k_mpot_perm_A_m1_n1_D__to__m1_D_n1(
    const float* __restrict__ A,
    unsigned int m_1, unsigned int n_1, unsigned int D,
    float* __restrict__ A_perm)
{
	const unsigned int i_1   = blockIdx.z;
	const unsigned int alpha = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j_1   = blockIdx.x * blockDim.x + threadIdx.x;
	if (i_1 >= m_1 || alpha >= D || j_1 >= n_1) return;
	A_perm[(size_t)i_1 * D * n_1 + (size_t)alpha * n_1 + j_1] =
	    A[(size_t)i_1 * n_1 * D + (size_t)j_1 * D + alpha];
}

// Permute T1 from (T, m_1, D, n_2) row-major to (T, n_2, m_1, D) row-major.
// One thread per output element, iterate the outer batch t over blockIdx.z.
__global__ void k_mpot_perm_T1_T_m1_D_n2__to__T_n2_m1_D(
    const float* __restrict__ T1,
    unsigned int T, unsigned int m_1, unsigned int D, unsigned int n_2,
    float* __restrict__ T1_perm)
{
	// Flatten (t, j_2) in y; (i_1, α) in x.
	const unsigned int tj = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int ia = blockIdx.x * blockDim.x + threadIdx.x;
	if (tj >= T * n_2 || ia >= m_1 * D) return;
	const unsigned int t   = tj / n_2;
	const unsigned int j_2 = tj % n_2;
	const unsigned int i_1 = ia / D;
	const unsigned int alpha = ia % D;
	// T1[t, i_1, α, j_2]  → T1_perm[t, j_2, i_1, α]
	T1_perm[((size_t)t * n_2 + j_2) * (m_1 * D) + (size_t)i_1 * D + alpha] =
	    T1[((size_t)t * m_1 + i_1) * (D * n_2) + (size_t)alpha * n_2 + j_2];
}

// Inverse permutations (used by mpot_backward).

// A_perm (m_1, D, n_1) → A (m_1, n_1, D)
__global__ void k_mpot_perm_A_inv_m1_D_n1__to__m1_n1_D(
    const float* __restrict__ A_perm,
    unsigned int m_1, unsigned int n_1, unsigned int D,
    float* __restrict__ A)
{
	const unsigned int i_1   = blockIdx.z;
	const unsigned int j_1   = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int alpha = blockIdx.x * blockDim.x + threadIdx.x;
	if (i_1 >= m_1 || j_1 >= n_1 || alpha >= D) return;
	A[(size_t)i_1 * n_1 * D + (size_t)j_1 * D + alpha] =
	    A_perm[(size_t)i_1 * D * n_1 + (size_t)alpha * n_1 + j_1];
}

// B_perm (m_2, D, n_2) → B (D, m_2, n_2)
__global__ void k_mpot_perm_B_inv_m2_D_n2__to__D_m2_n2(
    const float* __restrict__ B_perm,
    unsigned int D, unsigned int m_2, unsigned int n_2,
    float* __restrict__ B)
{
	const unsigned int alpha = blockIdx.z;
	const unsigned int i_2   = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j_2   = blockIdx.x * blockDim.x + threadIdx.x;
	if (alpha >= D || i_2 >= m_2 || j_2 >= n_2) return;
	B[(size_t)alpha * m_2 * n_2 + (size_t)i_2 * n_2 + j_2] =
	    B_perm[(size_t)i_2 * D * n_2 + (size_t)alpha * n_2 + j_2];
}

// T1_perm (T, n_2, m_1, D) → T1 (T, m_1, D, n_2)
__global__ void k_mpot_perm_T1_inv_T_n2_m1_D__to__T_m1_D_n2(
    const float* __restrict__ T1_perm,
    unsigned int T, unsigned int m_1, unsigned int D, unsigned int n_2,
    float* __restrict__ T1)
{
	const unsigned int tj = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int ia = blockIdx.x * blockDim.x + threadIdx.x;
	if (tj >= T * n_2 || ia >= m_1 * D) return;
	const unsigned int t   = tj / n_2;
	const unsigned int j_2 = tj % n_2;
	const unsigned int i_1 = ia / D;
	const unsigned int alpha = ia % D;
	T1[((size_t)t * m_1 + i_1) * (D * n_2) + (size_t)alpha * n_2 + j_2] =
	    T1_perm[((size_t)t * n_2 + j_2) * (m_1 * D) + (size_t)i_1 * D + alpha];
}

// Permute Y_pre (T, n_2, n_1) → Y (T, n_1, n_2).
__global__ void k_mpot_perm_Y_T_n2_n1__to__T_n1_n2(
    const float* __restrict__ Y_pre,
    unsigned int T, unsigned int n_1, unsigned int n_2,
    float* __restrict__ Y)
{
	const unsigned int t   = blockIdx.z;
	const unsigned int j_1 = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j_2 = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= T || j_1 >= n_1 || j_2 >= n_2) return;
	Y[((size_t)t * n_1 + j_1) * n_2 + j_2] =
	    Y_pre[((size_t)t * n_2 + j_2) * n_1 + j_1];
}

// Scatter the first D right singular vectors V_rm [Q × D] into B:
//   B[α, i_2, j_2] = √Σ[α] · V_rm[i_2·n_2 + j_2, α]        for α < D
// Input V_full_rm is row-major [Q × Q] (only first D columns used).
__global__ void k_mpot_scatter_B(const float* __restrict__ V_full_rm,
                                 unsigned int Q_ld,
                                 const float* __restrict__ Sigma,
                                 unsigned int m_2, unsigned int n_2,
                                 unsigned int D,
                                 float* __restrict__ B)
{
	const unsigned int q = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int a = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int Q = m_2 * n_2;
	if (q >= Q || a >= D) return;
	const unsigned int i_2 = q / n_2;
	const unsigned int j_2 = q % n_2;
	const float s = Sigma[a];
	const float sqs = s > 0.0f ? sqrtf(s) : 0.0f;
	// B[α, i_2, j_2] at offset α*m_2*n_2 + i_2*n_2 + j_2.
	B[(size_t)a * m_2 * n_2 + (size_t)i_2 * n_2 + j_2] =
	    sqs * V_full_rm[(size_t)q * Q_ld + a];
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

// ========================================================================
// File-local cuSOLVER state for MPOT.  Separate handle from OVFG/Stiefel
// to avoid workspace contention.
// ========================================================================
namespace {
cusolverDnHandle_t g_mpotSolver = nullptr;
bool g_mpotSolverReady = false;
int* g_mpotInfo = nullptr;

bool mpot_solver_init()
{
	if (!g_mpotSolverReady)
	{
		if (cusolverDnCreate(&g_mpotSolver) != CUSOLVER_STATUS_SUCCESS)
			return false;
		g_mpotSolverReady = true;
	}
	cusolverDnSetStream(g_mpotSolver, computeStream());
	if (g_mpotInfo == nullptr)
	{
		if (cudaMalloc(&g_mpotInfo, sizeof(int)) != cudaSuccess) return false;
	}
	return true;
}
} // anonymous namespace

// ========================================================================
// mpot_init_from_dense — full SVD of the permuted W matrix, truncate to
// bond D, scatter into (A, B).  See gpu_mpot.h for the math.
//
// Scratch layout (floats, caller-owned):
//   [0 .. P·Q)         M (permuted W, row-major)
//   [P·Q ..)           M_col (column-major copy for cuSOLVER)
//   then                U_col (P × P), Sigma (min(P,Q)), VT_col (Q × Q)
//   then                U_rm (P × P), V_rm (Q × Q) row-major staging
// where P = m_1·n_1, Q = m_2·n_2.
//
// Total: P·Q + P·Q + P·P + min(P,Q) + Q·Q + P·P + Q·Q
//      = 2 P·Q + 2 P² + 2 Q² + min(P,Q).
// ========================================================================
bool mpot_init_from_dense(const float* W,
                          unsigned int m_1, unsigned int m_2,
                          unsigned int n_1, unsigned int n_2,
                          unsigned int D,
                          float* A, float* B,
                          float* scratch)
{
	if (W == nullptr || A == nullptr || B == nullptr || scratch == nullptr)
		return false;
	if (m_1 == 0u || m_2 == 0u || n_1 == 0u || n_2 == 0u || D == 0u) return false;

	const unsigned int P = m_1 * n_1;
	const unsigned int Q = m_2 * n_2;
	const unsigned int K = P < Q ? P : Q;
	if (D > K) return false;
	if (!mpot_solver_init()) return false;

	// Scratch carve-out.
	size_t off = 0;
	float* M       = scratch + off; off += (size_t)P * Q;
	float* M_col   = scratch + off; off += (size_t)P * Q;
	float* U_col   = scratch + off; off += (size_t)P * P;
	float* Sigma   = scratch + off; off += K;
	float* VT_col  = scratch + off; off += (size_t)Q * Q;
	float* U_rm    = scratch + off; off += (size_t)P * P;
	float* V_rm    = scratch + off; off += (size_t)Q * Q;
	(void)off;

	// (1) Permute W into M.
	{
		dim3 block(32, 8);
		dim3 grid((Q + block.x - 1u) / block.x,
		          (P + block.y - 1u) / block.y);
		k_mpot_index_permute<<<grid, block, 0, computeStream()>>>(
		    W, m_1, m_2, n_1, n_2, M);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// (2) M (row-major P × Q) → M_col (column-major P × Q).
	{
		dim3 block(32, 8);
		dim3 grid((Q + block.x - 1u) / block.x,
		          (P + block.y - 1u) / block.y);
		k_mpot_rm_to_cm<<<grid, block, 0, computeStream()>>>(
		    M, P, Q, M_col);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// (3) Full SVD (jobu='A', jobvt='A').
	int lwork = 0;
	if (cusolverDnSgesvd_bufferSize(g_mpotSolver, P, Q, &lwork)
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

	cusolverStatus_t st = cusolverDnSgesvd(
	    g_mpotSolver, 'A', 'A', P, Q, M_col, P,
	    Sigma, U_col, P, VT_col, Q,
	    svd_work, lwork, nullptr, g_mpotInfo);
	if (st != CUSOLVER_STATUS_SUCCESS) return false;
	int host_info = 0;
	cudaMemcpyAsync(&host_info, g_mpotInfo, sizeof(int),
	                cudaMemcpyDeviceToHost, computeStream());
	cudaStreamSynchronize(computeStream());
	if (host_info != 0) return false;

	// (4) Transpose U_col (col-major P × P) to U_rm (row-major P × P).
	{
		dim3 block(32, 8);
		dim3 grid((P + block.x - 1u) / block.x,
		          (P + block.y - 1u) / block.y);
		k_mpot_cm_to_rm<<<grid, block, 0, computeStream()>>>(
		    U_col, P, P, U_rm);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	// cuSOLVER's VT_col is V^T in column-major layout: raw element at
	// offset (j*Q + i) holds V^T[i, j] = V[j, i].  Reinterpreted as
	// row-major, raw at (i*Q + j) = VT_col raw[i*Q + j] = V[i, j].
	// So VT_col reinterpreted as row-major IS V row-major — a plain
	// byte copy into V_rm suffices.
	cudaMemcpyAsync(V_rm, VT_col, (size_t)Q * Q * sizeof(float),
	                cudaMemcpyDeviceToDevice, computeStream());

	// (5) Scatter into A and B.
	{
		dim3 block(32, 8);
		dim3 grid((D + block.x - 1u) / block.x,
		          (P + block.y - 1u) / block.y);
		k_mpot_scatter_A<<<grid, block, 0, computeStream()>>>(
		    U_rm, P, Sigma, m_1, n_1, D, A);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	{
		dim3 block(32, 8);
		dim3 grid((D + block.x - 1u) / block.x,
		          (Q + block.y - 1u) / block.y);
		k_mpot_scatter_B<<<grid, block, 0, computeStream()>>>(
		    V_rm, Q, Sigma, m_2, n_2, D, B);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	return true;
}

// ========================================================================
// mpot_forward — see gpu_mpot.h for the algorithm.
// ========================================================================
bool mpot_forward(const float* X,
                  const float* A, const float* B,
                  unsigned int T,
                  unsigned int m_1, unsigned int m_2,
                  unsigned int n_1, unsigned int n_2,
                  unsigned int D,
                  float* Y,
                  float* scratch)
{
	if (X == nullptr || A == nullptr || B == nullptr || Y == nullptr || scratch == nullptr)
		return false;
	if (T == 0u || m_1 == 0u || m_2 == 0u || n_1 == 0u || n_2 == 0u || D == 0u)
		return false;

	// Scratch layout:
	//   A_perm      [m_1 · D · n_1]        (A permuted)
	//   B_perm      [m_2 · D · n_2]        (B permuted)
	//   T1          [T · m_1 · D · n_2]
	//   T1_perm     [T · n_2 · m_1 · D]    (same size as T1)
	//   Y_pre       [T · n_2 · n_1]        (same size as Y)
	size_t off = 0;
	float* A_perm   = scratch + off; off += (size_t)m_1 * D * n_1;
	float* B_perm   = scratch + off; off += (size_t)m_2 * D * n_2;
	float* T1       = scratch + off; off += (size_t)T * m_1 * D * n_2;
	float* T1_perm  = scratch + off; off += (size_t)T * n_2 * m_1 * D;
	float* Y_pre    = scratch + off; off += (size_t)T * n_1 * n_2;
	(void)off;

	// (1) Permute B  (D, m_2, n_2)  →  B_perm (m_2, D, n_2)
	{
		dim3 block(16, 8, 1);
		dim3 grid((n_2 + block.x - 1u) / block.x,
		          (m_2 + block.y - 1u) / block.y,
		          D);
		k_mpot_perm_B_D_m2_n2__to__m2_D_n2<<<grid, block, 0, computeStream()>>>(
		    B, D, m_2, n_2, B_perm);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// (2) Permute A  (m_1, n_1, D)  →  A_perm (m_1, D, n_1)
	{
		dim3 block(16, 8, 1);
		dim3 grid((n_1 + block.x - 1u) / block.x,
		          (D + block.y - 1u) / block.y,
		          m_1);
		k_mpot_perm_A_m1_n1_D__to__m1_D_n1<<<grid, block, 0, computeStream()>>>(
		    A, m_1, n_1, D, A_perm);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// (3) GEMM:  X [T·m_1, m_2]  ·  B_perm [m_2, D·n_2]  →  T1 [T·m_1, D·n_2]
	// X is stored as (T, m_1, m_2) row-major ≡ (T·m_1, m_2) row-major with
	// the same memory layout — no permutation needed.
	const int Tm1 = static_cast<int>(T) * static_cast<int>(m_1);
	const int Dn2 = static_cast<int>(D) * static_cast<int>(n_2);
	if (!sgemm_rowmajor(Tm1, Dn2, static_cast<int>(m_2),
	                    1.0f,
	                    X,       static_cast<int>(m_2),
	                    B_perm,  Dn2,
	                    0.0f,
	                    T1,      Dn2))
		return false;

	// (4) Permute T1  (T, m_1, D, n_2)  →  T1_perm (T, n_2, m_1, D)
	{
		dim3 block(16, 16, 1);
		dim3 grid((m_1 * D + block.x - 1u) / block.x,
		          (T * n_2 + block.y - 1u) / block.y,
		          1);
		k_mpot_perm_T1_T_m1_D_n2__to__T_n2_m1_D<<<grid, block, 0, computeStream()>>>(
		    T1, T, m_1, D, n_2, T1_perm);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// (5) GEMM:  T1_perm [T·n_2, m_1·D]  ·  A_perm [m_1·D, n_1]  →  Y_pre [T·n_2, n_1]
	const int Tn2 = static_cast<int>(T) * static_cast<int>(n_2);
	const int m1D = static_cast<int>(m_1) * static_cast<int>(D);
	if (!sgemm_rowmajor(Tn2, static_cast<int>(n_1), m1D,
	                    1.0f,
	                    T1_perm, m1D,
	                    A_perm,  static_cast<int>(n_1),
	                    0.0f,
	                    Y_pre,   static_cast<int>(n_1)))
		return false;

	// (6) Permute Y_pre (T, n_2, n_1)  →  Y (T, n_1, n_2)
	{
		dim3 block(16, 16, 1);
		dim3 grid((n_2 + block.x - 1u) / block.x,
		          (n_1 + block.y - 1u) / block.y,
		          T);
		k_mpot_perm_Y_T_n2_n1__to__T_n1_n2<<<grid, block, 0, computeStream()>>>(
		    Y_pre, T, n_1, n_2, Y);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	return true;
}

// ========================================================================
// mpot_backward — see gpu_mpot.h for the chain-rule derivation.
// ========================================================================
bool mpot_backward(const float* X,
                   const float* A, const float* B,
                   const float* dY,
                   unsigned int T,
                   unsigned int m_1, unsigned int m_2,
                   unsigned int n_1, unsigned int n_2,
                   unsigned int D,
                   float* dX, float* dA, float* dB,
                   float* scratch)
{
	if (X == nullptr || A == nullptr || B == nullptr || dY == nullptr)
		return false;
	if (dX == nullptr || dA == nullptr || dB == nullptr || scratch == nullptr)
		return false;
	if (T == 0u || m_1 == 0u || m_2 == 0u || n_1 == 0u || n_2 == 0u || D == 0u)
		return false;

	// Scratch layout (all floats):
	size_t off = 0;
	float* A_perm     = scratch + off; off += (size_t)m_1 * D * n_1;
	float* B_perm     = scratch + off; off += (size_t)m_2 * D * n_2;
	float* T1         = scratch + off; off += (size_t)T * m_1 * D * n_2;
	float* T1_perm    = scratch + off; off += (size_t)T * n_2 * m_1 * D;
	float* dY_pre     = scratch + off; off += (size_t)T * n_2 * n_1;
	float* dT1_perm   = scratch + off; off += (size_t)T * n_2 * m_1 * D;
	float* dT1        = scratch + off; off += (size_t)T * m_1 * D * n_2;
	float* dA_perm    = scratch + off; off += (size_t)m_1 * D * n_1;
	float* dB_perm    = scratch + off; off += (size_t)m_2 * D * n_2;
	(void)off;

	// --- Forward recomputation of intermediate buffers (self-contained) ---

	// A_perm
	{
		dim3 block(16, 8, 1);
		dim3 grid((n_1 + block.x - 1u) / block.x,
		          (D + block.y - 1u) / block.y,
		          m_1);
		k_mpot_perm_A_m1_n1_D__to__m1_D_n1<<<grid, block, 0, computeStream()>>>(
		    A, m_1, n_1, D, A_perm);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	// B_perm
	{
		dim3 block(16, 8, 1);
		dim3 grid((n_2 + block.x - 1u) / block.x,
		          (m_2 + block.y - 1u) / block.y,
		          D);
		k_mpot_perm_B_D_m2_n2__to__m2_D_n2<<<grid, block, 0, computeStream()>>>(
		    B, D, m_2, n_2, B_perm);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	// T1 = X · B_perm
	const int Tm1 = static_cast<int>(T) * static_cast<int>(m_1);
	const int Dn2 = static_cast<int>(D) * static_cast<int>(n_2);
	if (!sgemm_rowmajor(Tm1, Dn2, static_cast<int>(m_2),
	                    1.0f,
	                    X,       static_cast<int>(m_2),
	                    B_perm,  Dn2,
	                    0.0f,
	                    T1,      Dn2))
		return false;
	// T1 → T1_perm
	{
		dim3 block(16, 16, 1);
		dim3 grid((m_1 * D + block.x - 1u) / block.x,
		          (T * n_2 + block.y - 1u) / block.y,
		          1);
		k_mpot_perm_T1_T_m1_D_n2__to__T_n2_m1_D<<<grid, block, 0, computeStream()>>>(
		    T1, T, m_1, D, n_2, T1_perm);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// dY is (T, n_1, n_2); un-permute back to dY_pre (T, n_2, n_1) —
	// same swap direction as the forward Y_pre → Y kernel, re-used with
	// n_1 and n_2 arguments swapped.
	{
		dim3 block(16, 16, 1);
		dim3 grid((n_1 + block.x - 1u) / block.x,
		          (n_2 + block.y - 1u) / block.y,
		          T);
		k_mpot_perm_Y_T_n2_n1__to__T_n1_n2<<<grid, block, 0, computeStream()>>>(
		    dY, T, n_2, n_1, dY_pre);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	// --- GEMM 2 backward  (forward: Y_pre = T1_perm · A_perm) ---
	// dA_perm = T1_perm^T · dY_pre  (shape (m_1·D, n_1))
	const int Tn2 = static_cast<int>(T) * static_cast<int>(n_2);
	const int m1D = static_cast<int>(m_1) * static_cast<int>(D);
	if (!sgemm_rowmajor_atb(m1D, static_cast<int>(n_1), Tn2,
	                        1.0f,
	                        T1_perm, m1D,
	                        dY_pre,  static_cast<int>(n_1),
	                        0.0f,
	                        dA_perm, static_cast<int>(n_1)))
		return false;
	// dT1_perm = dY_pre · A_perm^T  (shape (T·n_2, m_1·D))
	if (!sgemm_rowmajor_abt(Tn2, m1D, static_cast<int>(n_1),
	                        1.0f,
	                        dY_pre,  static_cast<int>(n_1),
	                        A_perm,  static_cast<int>(n_1),
	                        0.0f,
	                        dT1_perm, m1D))
		return false;

	// --- GEMM 1 backward  (forward: T1 = X · B_perm) ---
	// dT1 = un-permute(dT1_perm)  (T, m_1, D, n_2)
	{
		dim3 block(16, 16, 1);
		dim3 grid((m_1 * D + block.x - 1u) / block.x,
		          (T * n_2 + block.y - 1u) / block.y,
		          1);
		k_mpot_perm_T1_inv_T_n2_m1_D__to__T_m1_D_n2<<<grid, block, 0, computeStream()>>>(
		    dT1_perm, T, m_1, D, n_2, dT1);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	// dX = dT1 · B_perm^T  (shape (T·m_1, m_2))
	if (!sgemm_rowmajor_abt(Tm1, static_cast<int>(m_2), Dn2,
	                        1.0f,
	                        dT1,     Dn2,
	                        B_perm,  Dn2,
	                        0.0f,
	                        dX,      static_cast<int>(m_2)))
		return false;
	// dB_perm = X^T · dT1  (shape (m_2, D·n_2))
	if (!sgemm_rowmajor_atb(static_cast<int>(m_2), Dn2, Tm1,
	                        1.0f,
	                        X,       static_cast<int>(m_2),
	                        dT1,     Dn2,
	                        0.0f,
	                        dB_perm, Dn2))
		return false;

	// Un-permute dA_perm and dB_perm to their native layouts.
	{
		dim3 block(16, 8, 1);
		dim3 grid((D + block.x - 1u) / block.x,
		          (n_1 + block.y - 1u) / block.y,
		          m_1);
		k_mpot_perm_A_inv_m1_D_n1__to__m1_n1_D<<<grid, block, 0, computeStream()>>>(
		    dA_perm, m_1, n_1, D, dA);
		if (cudaGetLastError() != cudaSuccess) return false;
	}
	{
		dim3 block(16, 8, 1);
		dim3 grid((n_2 + block.x - 1u) / block.x,
		          (m_2 + block.y - 1u) / block.y,
		          D);
		k_mpot_perm_B_inv_m2_D_n2__to__D_m2_n2<<<grid, block, 0, computeStream()>>>(
		    dB_perm, D, m_2, n_2, dB);
		if (cudaGetLastError() != cudaSuccess) return false;
	}

	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
