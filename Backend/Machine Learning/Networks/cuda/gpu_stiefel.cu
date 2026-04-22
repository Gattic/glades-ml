// GPU Stiefel × Σ manifold-factored weight primitives (paradigm shift #7).
//
// See gpu_stiefel.h and research/WEIGHT_MANIFOLD_DESIGN.md for the framework.
// This file ships the Phase-1 minimal prototype: allocation, forward-pass
// 3-chained SGEMM, and a reconstruct-dense helper for parity testing.
//
// Later phases will add: QR retraction (cuSOLVER), Cayley retraction,
// tangent-projected backward, Riemannian Adam, vector transport.

#include "gpu_stiefel.h"
#include "gpu_device.h"
#include "gpu_blas.h"
#include "gpu_kernels.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cstring>

namespace glades {
namespace gpu {

namespace {

#define GLADES_CUDA_CHECK(call)                                               \
	do {                                                                      \
		cudaError_t err_ = (call);                                            \
		if (err_ != cudaSuccess) {                                            \
			fprintf(stderr, "[stiefel-cuda] %s:%d  %s  -> %s\n",              \
			        __FILE__, __LINE__, #call, cudaGetErrorString(err_));     \
			return;                                                           \
		}                                                                     \
	} while (0)

// Scale each column of A[M,N] by d[N] (row-major, d broadcast across rows).
__global__ void k_scale_cols_by_diag(float* A, const float* d,
                                     unsigned int M, unsigned int N)
{
	unsigned int row = blockIdx.x;
	unsigned int col = threadIdx.x + blockIdx.y * blockDim.x;
	if (row >= M || col >= N) return;
	A[row * N + col] *= d[col];
}

// Elementwise: W[i,j] += U[i,k] * Σ[k] * V[j,k] summed over k. Used for the
// dense-reconstruction helper. Straight row-major [m × n] output.
// This is the explicit m×n×r contraction used only in parity testing — for
// the hot path we compute Y = X · V · diag(Σ) · U^T directly without
// materializing W.
__global__ void k_stiefel_reconstruct(const uint16_t* U, const float* sigma,
                                      const uint16_t* V, float* W,
                                      unsigned int m, unsigned int n,
                                      unsigned int r)
{
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m || j >= n) return;

	float acc = 0.0f;
	for (unsigned int k = 0; k < r; ++k)
	{
		// BF16 decode: shift to FP32 by placing the 16 bits in the high half
		unsigned int u_bits = ((unsigned int)U[i * r + k]) << 16;
		unsigned int v_bits = ((unsigned int)V[j * r + k]) << 16;
		float u_val, v_val;
		memcpy(&u_val, &u_bits, sizeof(float));
		memcpy(&v_val, &v_bits, sizeof(float));
		acc += u_val * sigma[k] * v_val;
	}
	W[i * n + j] = acc;
}

} // anonymous namespace

// ===========================================================================
// GpuStiefelWeight lifecycle
// ===========================================================================

GpuStiefelWeight::GpuStiefelWeight() : m(0), n(0), r(0) {}

void GpuStiefelWeight::allocate(unsigned int m_, unsigned int n_, unsigned int r_)
{
	m = m_;
	n = n_;
	r = r_;

	U.allocate(size_t(m) * r);
	sigma.allocate(r);
	V.allocate(size_t(n) * r);

	// Adam moments (FP32 for now; int8 packing is a follow-up phase).
	m_U.allocate(size_t(m) * r);
	m_sigma.allocate(r);
	m_V.allocate(size_t(n) * r);
	v_U.allocate(size_t(m) * r);
	v_sigma.allocate(r);
	v_V.allocate(size_t(n) * r);

	// Zero-initialize moments so bias correction works from step 1.
	cudaMemset(m_U.data(), 0, size_t(m) * r * sizeof(float));
	cudaMemset(m_V.data(), 0, size_t(n) * r * sizeof(float));
	cudaMemset(m_sigma.data(), 0, r * sizeof(float));
	cudaMemset(v_U.data(), 0, size_t(m) * r * sizeof(float));
	cudaMemset(v_V.data(), 0, size_t(n) * r * sizeof(float));
	cudaMemset(v_sigma.data(), 0, r * sizeof(float));

	qr_tau_U.allocate(r);
	qr_tau_V.allocate(r);

	orthogonality_drift.allocate(1);
}

void GpuStiefelWeight::release()
{
	U.free();
	sigma.free();
	V.free();
	m_U.free();
	m_sigma.free();
	m_V.free();
	v_U.free();
	v_sigma.free();
	v_V.free();
	qr_tau_U.free();
	qr_tau_V.free();
	qr_work.free();
	orthogonality_drift.free();
	m = n = r = 0;
}

bool GpuStiefelWeight::allocated() const
{
	return U.allocated() && sigma.allocated() && V.allocated() && r > 0;
}

// ===========================================================================
// stiefel_reconstruct_dense — materialize W = U · diag(Σ) · V^T for parity.
// ===========================================================================

void stiefel_reconstruct_dense(const GpuStiefelWeight& s, float* W_dense)
{
	if (!s.allocated() || W_dense == nullptr) return;

	dim3 block(16, 16);
	dim3 grid((s.n + block.x - 1) / block.x, (s.m + block.y - 1) / block.y);

	k_stiefel_reconstruct<<<grid, block, 0, computeStream()>>>(s.U.data(), s.sigma.data(),
	                                       s.V.data(), W_dense,
	                                       s.m, s.n, s.r);
	GLADES_CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// stiefel_forward — Y[B × m] = X[B × n] · V · diag(Σ) · U^T via 3 SGEMMs.
//
// Step 1: T1[B × r]  = X    · V         (X [B,n], V [n,r])
// Step 2: T1       *= Σ (broadcast per column)
// Step 3: Y [B × m]  = T1   · U^T       (T1 [B,r], U [m,r] stored row-major so
//                                        U^T access is a straightforward abt)
//
// FP32 fast-path first. BF16 path (x_bf16 = true) will be added in the next
// iteration once sgemm_rowmajor_bf16 wrappers accept Stiefel-shaped inputs.
// ===========================================================================

void stiefel_forward(const void* X, bool x_bf16,
                     const GpuStiefelWeight& s,
                     float* Y, float* scratch1, unsigned int B)
{
	if (!s.allocated() || X == nullptr || Y == nullptr || scratch1 == nullptr)
		return;
	if (x_bf16)
	{
		// BF16 path not yet implemented in Phase-1; treat as FP32.
		// TODO: route to sgemm_rowmajor_bf16 when the BF16 input wrappers
		// are extended to take BF16 inputs with FP32 outputs.
	}

	const float* Xf = static_cast<const float*>(X);
	const uint16_t* Ubf = s.U.data();
	const uint16_t* Vbf = s.V.data();
	const float*    sig = s.sigma.data();

	// Step 1: T1 = X · V. V is BF16; stage to FP32 via cast first.
	// (Next iteration: replace with BF16-input GEMM via sgemm_rowmajor_bf16.)
	static thread_local float* V_f32 = nullptr;
	static thread_local size_t V_f32_cap = 0;
	size_t V_size = size_t(s.n) * s.r;
	if (V_size > V_f32_cap)
	{
		if (V_f32) cudaFree(V_f32);
		if (cudaMalloc(&V_f32, V_size * sizeof(float)) != cudaSuccess)
		{
			V_f32 = nullptr; V_f32_cap = 0; return;
		}
		V_f32_cap = V_size;
	}
	cast_bf16_to_f32(Vbf, V_f32, V_size);

	if (!sgemm_rowmajor(B, s.r, s.n, 1.0f, Xf, s.n, V_f32, s.r,
	                    0.0f, scratch1, s.r))
		return;

	// Step 2: broadcast-scale each column by Σ[k]
	{
		dim3 block(64);
		dim3 grid(B, (s.r + block.x - 1) / block.x);
		k_scale_cols_by_diag<<<grid, block, 0, computeStream()>>>(scratch1, sig, B, s.r);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}

	// Step 3: Y = T1 · U^T.  U is [m,r] row-major, so U^T is accessed via the
	// abt variant: C[B,m] = A[B,r] · B^T[m,r] where B is U.
	static thread_local float* U_f32 = nullptr;
	static thread_local size_t U_f32_cap = 0;
	size_t U_size = size_t(s.m) * s.r;
	if (U_size > U_f32_cap)
	{
		if (U_f32) cudaFree(U_f32);
		if (cudaMalloc(&U_f32, U_size * sizeof(float)) != cudaSuccess)
		{
			U_f32 = nullptr; U_f32_cap = 0; return;
		}
		U_f32_cap = U_size;
	}
	cast_bf16_to_f32(Ubf, U_f32, U_size);

	if (!sgemm_rowmajor_abt(B, s.m, s.r, 1.0f, scratch1, s.r, U_f32, s.r,
	                        0.0f, Y, s.m))
		return;
}

// ===========================================================================
// stiefel_backward_unconstrained — raw chain-rule gradients
//
// Given Y = X · V · diag(Σ) · U^T:
//   T1 = dY · U             [B × r]     (abt: dY [B,m], U [m,r])
//   dsigma[k] = Σ_b (T1[b,k] · (X · V)[b,k])        — row-wise reduction
//   T2 = T1 · diag(Σ)       [B × r]     in-place
//   dV = X^T · T2           [n × r]     (atb: X [B,n], T2 [B,r])
//   dX (optional) = T2 · V^T [B × n]    (abt: T2 [B,r], V [n,r])
//
//   T3 = X · V · diag(Σ)    [B × r]     — we reconstruct via forward path
//   dU = dY^T · T3          [m × r]     (atb: dY [B,m], T3 [B,r])
// ===========================================================================

namespace {
__global__ void k_rowwise_sum_product(const float* A, const float* B,
                                       float* out, unsigned int rows,
                                       unsigned int cols)
{
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= cols) return;
	float acc = 0.0f;
	for (unsigned int r = 0; r < rows; ++r)
		acc += A[r * cols + col] * B[r * cols + col];
	out[col] = acc;
}
}

void stiefel_backward_unconstrained(
    const float* dY,
    const void* X,
    bool /*x_bf16*/,
    const GpuStiefelWeight& s,
    float* dX,
    float* dU,
    float* dsigma,
    float* dV,
    float* scratch_Br,
    unsigned int B)
{
	if (!s.allocated() || dY == nullptr || X == nullptr ||
	    dU == nullptr || dV == nullptr || dsigma == nullptr || scratch_Br == nullptr)
		return;

	const float* Xf = static_cast<const float*>(X);
	const uint16_t* Ubf = s.U.data();
	const uint16_t* Vbf = s.V.data();
	const float*    sig = s.sigma.data();

	// FP32 staging of U, V. (Same thread-local caches as the forward path.)
	static thread_local float* U_f32 = nullptr;
	static thread_local size_t U_f32_cap = 0;
	static thread_local float* V_f32 = nullptr;
	static thread_local size_t V_f32_cap = 0;
	size_t U_size = size_t(s.m) * s.r;
	size_t V_size = size_t(s.n) * s.r;
	if (U_size > U_f32_cap)
	{
		if (U_f32) cudaFree(U_f32);
		if (cudaMalloc(&U_f32, U_size * sizeof(float)) != cudaSuccess) return;
		U_f32_cap = U_size;
	}
	if (V_size > V_f32_cap)
	{
		if (V_f32) cudaFree(V_f32);
		if (cudaMalloc(&V_f32, V_size * sizeof(float)) != cudaSuccess) return;
		V_f32_cap = V_size;
	}
	cast_bf16_to_f32(Ubf, U_f32, U_size);
	cast_bf16_to_f32(Vbf, V_f32, V_size);

	// --- Compute T3 = X · V · diag(Σ)  [B × r] into a scratch buffer.
	// We re-use scratch_Br for this, since it is also [B × r]. Callers need
	// two [B × r] scratches if they want to keep T1 around; we allocate a
	// second thread-local buffer for T1.
	static thread_local float* T1_cache = nullptr;
	static thread_local size_t T1_cap = 0;
	size_t T1_size = size_t(B) * s.r;
	if (T1_size > T1_cap)
	{
		if (T1_cache) cudaFree(T1_cache);
		if (cudaMalloc(&T1_cache, T1_size * sizeof(float)) != cudaSuccess) return;
		T1_cap = T1_size;
	}
	float* T3 = scratch_Br;  // [B × r]
	if (!sgemm_rowmajor(B, s.r, s.n, 1.0f, Xf, s.n, V_f32, s.r,
	                    0.0f, T3, s.r))
		return;
	{
		dim3 block(64);
		dim3 grid(B, (s.r + block.x - 1) / block.x);
		k_scale_cols_by_diag<<<grid, block, 0, computeStream()>>>(T3, sig, B, s.r);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}

	// --- T1 = dY · U  [B × r]  (abt: dY [B,m], U [m,r])
	float* T1 = T1_cache;
	if (!sgemm_rowmajor(B, s.r, s.m, 1.0f, dY, s.m, U_f32, s.r,
	                    0.0f, T1, s.r))
		return;

	// --- dΣ[k] = Σ_b T1[b,k] * T3_preScale[b,k]  — but T3 already has Σ applied.
	// dΣ[k] = Σ_b T1[b,k] * (X·V)[b,k], so we need (X·V) = T3 / Σ. Simpler:
	// recompute (X·V) into T1_cache-like buffer; we actually already have
	// T1 = dY·U, so we need a separate buffer. Use a third thread-local.
	static thread_local float* XV_cache = nullptr;
	static thread_local size_t XV_cap = 0;
	if (T1_size > XV_cap)
	{
		if (XV_cache) cudaFree(XV_cache);
		if (cudaMalloc(&XV_cache, T1_size * sizeof(float)) != cudaSuccess) return;
		XV_cap = T1_size;
	}
	if (!sgemm_rowmajor(B, s.r, s.n, 1.0f, Xf, s.n, V_f32, s.r,
	                    0.0f, XV_cache, s.r))
		return;
	{
		dim3 block(64);
		dim3 grid((s.r + block.x - 1) / block.x);
		k_rowwise_sum_product<<<grid, block, 0, computeStream()>>>(T1, XV_cache, dsigma, B, s.r);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}

	// --- T1 <- T1 · diag(Σ)  in place
	{
		dim3 block(64);
		dim3 grid(B, (s.r + block.x - 1) / block.x);
		k_scale_cols_by_diag<<<grid, block, 0, computeStream()>>>(T1, sig, B, s.r);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}

	// --- dV = X^T · T1  [n × r]  (atb: X [B,n], T1 [B,r])
	if (!sgemm_rowmajor_atb(s.n, s.r, B, 1.0f, Xf, s.n, T1, s.r,
	                        0.0f, dV, s.r))
		return;

	// --- dX (optional) = T1 · V^T  [B × n]  (abt: T1 [B,r], V [n,r])
	if (dX != nullptr)
	{
		if (!sgemm_rowmajor_abt(B, s.n, s.r, 1.0f, T1, s.r, V_f32, s.r,
		                        0.0f, dX, s.n))
			return;
	}

	// --- dU = dY^T · T3  [m × r]  (atb: dY [B,m], T3 [B,r])
	if (!sgemm_rowmajor_atb(s.m, s.r, B, 1.0f, dY, s.m, T3, s.r,
	                        0.0f, dU, s.r))
		return;
}

// ===========================================================================
// stiefel_tangent_project_grad — canonical Stiefel tangent-space projection
//
// For each Stiefel factor (U or V) with gradient G, the canonical-metric
// tangent projection reduces to
//     proj_U(G) = G − U · sym(U^T G)
// where sym(A) = (A + A^T)/2.  Derivation:
//     proj_U(G) = (I − UU^T) G + U · skew(U^T G)
//               = G − U(U^T G) + (U(U^T G) − U G^T U)/2
//               = G − (1/2) U (U^T G + G^T U)
//
// Implemented with two cuBLAS calls + one r×r symmetrization kernel:
//   S = U^T · G                           (atb, r×r)
//   S ← (S + S^T) / 2                     (in-place kernel)
//   G ← G − U · S                         (standard GEMM, alpha=−1, beta=1)
// ===========================================================================

namespace {
__global__ void k_symmetrize_inplace(float* S, unsigned int r)
{
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= r || j >= r) return;
	if (i > j) return;
	const float a = S[i * r + j];
	const float b = S[j * r + i];
	const float sym = 0.5f * (a + b);
	S[i * r + j] = sym;
	if (i != j) S[j * r + i] = sym;
}

static void stiefel_project_one(const uint16_t* A_bf, unsigned int rows,
                                unsigned int r, float* G, float* scratch_rr)
{
	// FP32 staging of A (U or V).
	static thread_local float* A_f32 = nullptr;
	static thread_local size_t A_f32_cap = 0;
	size_t sz = size_t(rows) * r;
	if (sz > A_f32_cap)
	{
		if (A_f32) cudaFree(A_f32);
		if (cudaMalloc(&A_f32, sz * sizeof(float)) != cudaSuccess) return;
		A_f32_cap = sz;
	}
	cast_bf16_to_f32(A_bf, A_f32, sz);

	// S = A^T · G   (atb: A [rows,r], G [rows,r] → S [r,r])
	if (!sgemm_rowmajor_atb(r, r, rows, 1.0f, A_f32, r, G, r, 0.0f, scratch_rr, r))
		return;

	// S ← (S + S^T) / 2
	{
		dim3 block(16, 16);
		dim3 grid((r + block.x - 1) / block.x, (r + block.y - 1) / block.y);
		k_symmetrize_inplace<<<grid, block, 0, computeStream()>>>(scratch_rr, r);
		cudaGetLastError();
	}

	// G ← G − A · S
	if (!sgemm_rowmajor(rows, r, r, -1.0f, A_f32, r, scratch_rr, r,
	                    1.0f, G, r))
		return;
}
}

void stiefel_tangent_project_grad(const GpuStiefelWeight& s,
                                  float* grad_U, float* grad_V,
                                  float* scratch_UtGU, float* scratch_VtGV)
{
	if (!s.allocated()) return;
	if (grad_U != nullptr && scratch_UtGU != nullptr)
		stiefel_project_one(s.U.data(), s.m, s.r, grad_U, scratch_UtGU);
	if (grad_V != nullptr && scratch_VtGV != nullptr)
		stiefel_project_one(s.V.data(), s.n, s.r, grad_V, scratch_VtGV);
}

void stiefel_backward_project(const float*, const void*, bool,
                              const GpuStiefelWeight&,
                              float*, float*, float*, float*, unsigned int)
{
	// Legacy symbol — unused for now.
}

// ===========================================================================
// QR retraction U ← qf(U + η_U), V ← qf(V + η_V), Σ ← Σ ⊙ exp(η_Σ / Σ)
//
// cuSOLVER works in column-major. Our A [rows × r] stored row-major has the
// same byte pattern as A^T [r × rows] stored column-major. Calling sgeqrf
// with m=rows, n=r on the row-major buffer interpreted as column-major
// [rows × r] actually QRs the matrix whose columns are our rows. We want
// the thin QR of the row-major [rows × r] (orthonormal columns). Simplest:
// maintain a column-major scratch and transpose in/out via a trivial kernel.
// ===========================================================================

namespace {

__global__ void k_transpose_2d(const float* in, float* out,
                               unsigned int rows, unsigned int cols)
{
	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows || j >= cols) return;
	// out is [cols × rows] row-major = [rows × cols] column-major
	out[j * rows + i] = in[i * cols + j];
}

__global__ void k_add_inplace(float* dst, const float* add, size_t n)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) dst[idx] += add[idx];
}

__global__ void k_sigma_fisher_rao(float* sigma, const float* eta,
                                   unsigned int r)
{
	unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= r) return;
	const float sig = sigma[k];
	const float exp_arg = eta[k] / (sig + 1e-12f);
	// Clip extremely large exponents to keep Σ bounded during large updates.
	const float capped = (exp_arg > 10.0f) ? 10.0f :
	                     (exp_arg < -10.0f ? -10.0f : exp_arg);
	sigma[k] = sig * expf(capped);
}

static cusolverDnHandle_t g_stiefelSolver = 0;
static bool g_stiefelSolverReady = false;
// Per-factor workspace (U vs V). Separate buffers prevent the second
// cuSOLVER call from stomping on workspace the first is still using —
// without this, we saw ~10% cold-start flakiness even after stream sync.
static float* g_stiefelQrWork[2] = {0, 0};
static size_t g_stiefelQrWorkCap[2] = {0, 0};
static int*   g_stiefelInfo[2] = {0, 0};

static bool stiefel_solver_init()
{
	if (!g_stiefelSolverReady)
	{
		if (cusolverDnCreate(&g_stiefelSolver) != CUSOLVER_STATUS_SUCCESS)
			return false;
		g_stiefelSolverReady = true;
	}
	cusolverDnSetStream(g_stiefelSolver, computeStream());
	for (int k = 0; k < 2; ++k)
	{
		if (g_stiefelInfo[k] == nullptr)
		{
			if (cudaMalloc(&g_stiefelInfo[k], sizeof(int)) != cudaSuccess)
				return false;
		}
	}
	return true;
}

// slot 0 = U factor, slot 1 = V factor — separate workspaces required.
static bool stiefel_qr_retract_one(uint16_t* A_bf, float* tau,
                                   unsigned int rows, unsigned int r,
                                   const float* eta, int slot)
{
	if (!stiefel_solver_init()) return false;

	// Stage A [rows × r] FP32.
	static thread_local float* A_f32 = nullptr;
	static thread_local size_t A_f32_cap = 0;
	// Column-major scratch [rows × r].
	static thread_local float* A_col = nullptr;
	static thread_local size_t A_col_cap = 0;

	const size_t sz = size_t(rows) * r;
	if (sz > A_f32_cap)
	{
		if (A_f32) cudaFree(A_f32);
		if (cudaMalloc(&A_f32, sz * sizeof(float)) != cudaSuccess) return false;
		A_f32_cap = sz;
	}
	if (sz > A_col_cap)
	{
		if (A_col) cudaFree(A_col);
		if (cudaMalloc(&A_col, sz * sizeof(float)) != cudaSuccess) return false;
		A_col_cap = sz;
	}

	// 1. A_f32 = BF16(A) + eta
	cast_bf16_to_f32(A_bf, A_f32, sz);
	if (eta != nullptr)
	{
		const size_t n = sz;
		dim3 block(256);
		dim3 grid((n + block.x - 1) / block.x);
		k_add_inplace<<<grid, block, 0, computeStream()>>>(A_f32, eta, n);
		cudaGetLastError();
	}

	// 2. Transpose to column-major [rows × r] in A_col (so sgeqrf reads it
	// correctly as column-major [rows × r]).
	{
		dim3 block(16, 16);
		dim3 grid((r + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
		k_transpose_2d<<<grid, block, 0, computeStream()>>>(A_f32, A_col, rows, r);
		cudaGetLastError();
	}

	// 3. Query sgeqrf workspace.
	int lwork_geqrf = 0, lwork_orgqr = 0;
	if (cusolverDnSgeqrf_bufferSize(g_stiefelSolver, rows, r,
	                                A_col, rows, &lwork_geqrf)
	    != CUSOLVER_STATUS_SUCCESS)
		return false;
	if (cusolverDnSorgqr_bufferSize(g_stiefelSolver, rows, r, r,
	                                A_col, rows, tau, &lwork_orgqr)
	    != CUSOLVER_STATUS_SUCCESS)
		return false;
	const int lwork = lwork_geqrf > lwork_orgqr ? lwork_geqrf : lwork_orgqr;

	if (size_t(lwork) > g_stiefelQrWorkCap[slot])
	{
		if (g_stiefelQrWork[slot]) cudaFree(g_stiefelQrWork[slot]);
		if (cudaMalloc(&g_stiefelQrWork[slot], lwork * sizeof(float))
		    != cudaSuccess)
			return false;
		g_stiefelQrWorkCap[slot] = lwork;
	}

	// 4. QR factorization in place.
	cusolverStatus_t st = cusolverDnSgeqrf(g_stiefelSolver, rows, r, A_col, rows, tau,
	                     g_stiefelQrWork[slot], lwork, g_stiefelInfo[slot]);
	if (st != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "[stiefel-qr] sgeqrf status=%d rows=%u r=%u\n",
		        (int)st, rows, r);
		return false;
	}
	int host_info = 0;
	cudaMemcpy(&host_info, g_stiefelInfo[slot], sizeof(int), cudaMemcpyDeviceToHost);
	if (host_info != 0)
	{
		fprintf(stderr, "[stiefel-qr] sgeqrf info=%d rows=%u r=%u\n",
		        host_info, rows, r);
		return false;
	}

	// 5. Form Q explicitly.
	st = cusolverDnSorgqr(g_stiefelSolver, rows, r, r, A_col, rows, tau,
	                     g_stiefelQrWork[slot], lwork, g_stiefelInfo[slot]);
	if (st != CUSOLVER_STATUS_SUCCESS)
	{
		fprintf(stderr, "[stiefel-qr] sorgqr status=%d rows=%u r=%u\n",
		        (int)st, rows, r);
		return false;
	}
	cudaMemcpy(&host_info, g_stiefelInfo[slot], sizeof(int), cudaMemcpyDeviceToHost);
	if (host_info != 0)
	{
		fprintf(stderr, "[stiefel-qr] sorgqr info=%d rows=%u r=%u\n",
		        host_info, rows, r);
		return false;
	}

	// 6. Transpose back to row-major and cast to BF16.
	{
		dim3 block(16, 16);
		dim3 grid((rows + block.x - 1) / block.x, (r + block.y - 1) / block.y);
		// A_col is column-major [rows × r] = row-major [r × rows]; transpose
		// back into A_f32 row-major [rows × r].
		k_transpose_2d<<<grid, block, 0, computeStream()>>>(A_col, A_f32, r, rows);
		cudaGetLastError();
	}
	cast_f32_to_bf16(A_f32, A_bf, sz);
	return true;
}

} // anonymous namespace

void stiefel_retract_qr(GpuStiefelWeight& s,
                        const float* eta_U,
                        const float* eta_sigma,
                        const float* eta_V)
{
	if (!s.allocated()) return;
	stiefel_qr_retract_one(s.U.data(), s.qr_tau_U.data(), s.m, s.r, eta_U, /*slot=*/0);
	stiefel_qr_retract_one(s.V.data(), s.qr_tau_V.data(), s.n, s.r, eta_V, /*slot=*/1);
	if (eta_sigma != nullptr)
	{
		dim3 block(64);
		dim3 grid((s.r + block.x - 1) / block.x);
		k_sigma_fisher_rao<<<grid, block, 0, computeStream()>>>(s.sigma.data(), eta_sigma, s.r);
		cudaGetLastError();
	}
}

void stiefel_retract_cayley(GpuStiefelWeight&, const float*, const float*, const float*,
                            float*, float*)
{
	// TODO Phase-2: implement Cayley fast-path.
}

// ===========================================================================
// Adam moment update + tangent-space step assembly
//
//   m ← β1·m + (1−β1)·g
//   v ← β2·v + (1−β2)·g²
//   m̂ = m / (1 − β1^t),  v̂ = v / (1 − β2^t)
//   η = −lr · m̂ / (√v̂ + eps)
//
// Written as a single elementwise kernel over the flattened tensor.  Note
// the minus sign is folded into η so that the QR retraction receives a
// descent direction directly:  U_new = qf(U + η).
// ===========================================================================

namespace {

__global__ void k_adam_step_and_eta(
    const float* __restrict__ g,
    float* __restrict__ mom,
    float* __restrict__ vel,
    float* __restrict__ eta_out,
    float lr,
    float beta1, float beta2, float eps,
    float bc1, float bc2,  // 1 / (1 − β^t) for first/second moments
    size_t n)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float gi = g[i];
	const float m_new = beta1 * mom[i] + (1.0f - beta1) * gi;
	const float v_new = beta2 * vel[i] + (1.0f - beta2) * gi * gi;
	mom[i] = m_new;
	vel[i] = v_new;
	const float m_hat = m_new * bc1;
	const float v_hat = v_new * bc2;
	eta_out[i] = -lr * m_hat / (sqrtf(v_hat) + eps);
}

} // anonymous namespace

void stiefel_adam_step(GpuStiefelWeight& s,
                       float* dU, float* dsigma, float* dV,
                       float lr, float beta1, float beta2, float eps,
                       int step_1based,
                       float* scratch_rr_U, float* scratch_rr_V,
                       float* scratch_etaU, float* scratch_etaV,
                       float* scratch_etaS)
{
	if (!s.allocated()) return;
	if (step_1based < 1) step_1based = 1;

	// (1) Tangent-project the raw gradients in place.
	stiefel_tangent_project_grad(s, dU, dV, scratch_rr_U, scratch_rr_V);

	// (2–3) Adam moment update and η computation (per sub-tensor).
	const float bc1 = 1.0f / (1.0f - std::pow(beta1, step_1based));
	const float bc2 = 1.0f / (1.0f - std::pow(beta2, step_1based));

	{
		dim3 block(256);
		const size_t nU = size_t(s.m) * s.r;
		dim3 grid((nU + block.x - 1) / block.x);
		k_adam_step_and_eta<<<grid, block, 0, computeStream()>>>(dU, s.m_U.data(), s.v_U.data(),
		                                     scratch_etaU, lr, beta1, beta2,
		                                     eps, bc1, bc2, nU);
	}
	{
		dim3 block(256);
		const size_t nV = size_t(s.n) * s.r;
		dim3 grid((nV + block.x - 1) / block.x);
		k_adam_step_and_eta<<<grid, block, 0, computeStream()>>>(dV, s.m_V.data(), s.v_V.data(),
		                                     scratch_etaV, lr, beta1, beta2,
		                                     eps, bc1, bc2, nV);
	}
	{
		dim3 block(64);
		dim3 grid((s.r + block.x - 1) / block.x);
		k_adam_step_and_eta<<<grid, block, 0, computeStream()>>>(dsigma, s.m_sigma.data(),
		                                     s.v_sigma.data(),
		                                     scratch_etaS, lr, beta1, beta2,
		                                     eps, bc1, bc2, s.r);
	}

	// (4–5) Retract onto the manifold.
	stiefel_retract_qr(s, scratch_etaU, scratch_etaS, scratch_etaV);
}

void stiefel_vector_transport(GpuStiefelWeight&, float*)
{
	// TODO Phase-2f: Gram-Schmidt vector transport of moments onto new
	// tangent space (currently an approximation — keep moments in place).
}

void stiefel_check_orthogonality(GpuStiefelWeight&)
{
	// TODO Phase-2: compute ‖U^T U − I_r‖_F.
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
