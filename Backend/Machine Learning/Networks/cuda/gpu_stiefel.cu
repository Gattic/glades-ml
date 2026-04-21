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

	m_U.allocate(size_t(m) * r);
	m_U_scale.allocate(1);
	m_sigma.allocate(r);
	m_V.allocate(size_t(n) * r);
	m_V_scale.allocate(1);

	v_U.allocate(size_t(m) * r);
	v_U_scale.allocate(1);
	v_sigma.allocate(r);
	v_V.allocate(size_t(n) * r);
	v_V_scale.allocate(1);

	qr_tau_U.allocate(r);
	qr_tau_V.allocate(r);
	// qr_work sized lazily by the retraction wrappers once cuSOLVER is wired.

	orthogonality_drift.allocate(1);
}

void GpuStiefelWeight::release()
{
	U.free();
	sigma.free();
	V.free();
	m_U.free();
	m_U_scale.free();
	m_sigma.free();
	m_V.free();
	m_V_scale.free();
	v_U.free();
	v_U_scale.free();
	v_sigma.free();
	v_V.free();
	v_V_scale.free();
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

	k_stiefel_reconstruct<<<grid, block>>>(s.U.data(), s.sigma.data(),
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
		k_scale_cols_by_diag<<<grid, block>>>(scratch1, sig, B, s.r);
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
// Phase-1 stubs for later primitives — keep link graph intact.
// ===========================================================================

void stiefel_backward_project(const float*, const void*, bool,
                              const GpuStiefelWeight&,
                              float*, float*, float*, float*, unsigned int)
{
	// TODO Phase-2: implement tangent projection backward.
}

void stiefel_retract_qr(GpuStiefelWeight&, const float*, const float*, const float*)
{
	// TODO Phase-2: implement cuSOLVER sgeqrf + sorgqr retraction.
}

void stiefel_retract_cayley(GpuStiefelWeight&, const float*, const float*, const float*,
                            float*, float*)
{
	// TODO Phase-2: implement Cayley fast-path.
}

void stiefel_vector_transport(GpuStiefelWeight&, float*)
{
	// TODO Phase-2: modified Gram-Schmidt vector transport for Adam momenta.
}

void stiefel_check_orthogonality(GpuStiefelWeight&)
{
	// TODO Phase-2: compute ‖U^T U − I_r‖_F.
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
