// IGAA (Information-Geometric Attention Augmentation) — paradigm #260.
// Iter 25-26 (2026-05-16) Gate-0 prototype GPU kernels.
//
// See gpu_igaa.h for the host-facing API and the formula reference.

#include "gpu_igaa.h"
#include "gpu_device.h"
#include "gpu_blas.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

namespace glades {
namespace gpu {

// One block per token (gridDim.x = T).  Block size threadIdx.x indexes K (≤ 16
// at Gate-0; we pad up to 32 for warp friendliness).  Steps:
//   (a) load Z[i, k] from pi_buf (where the sgemm + bias have already written)
//   (b) divide by tau, find max for numerical stability, exponentiate, sum
//   (c) write softmax back into pi_buf (T × K)
//   (d) compute gate_i = Σ_{k < m_scfa} (Pi[i,k] − 1/K)
//        — convention: the first m_scfa columns correspond to SCFA modes; the
//          rest are identity (contribute 0 to Δy).  Trainer maintains this
//          permutation via the --igaa-modes argument.
//   (e) read α scalar, compute scale_i = α * gate_i
//   (f) write s = scale_i back into pi_buf[i, K] (we overflow by 1; pi_buf
//        was allocated T*(K+1) by the caller? — instead use a per-block shared
//        scalar and return scale_i implicit via the next kernel.)
//
// To keep the kernel simple at Gate-0, we write the per-token scale_i into
// a separate output buffer (gate_out, T floats).  A second kernel then
// applies p[i, :] *= (1 + scale_i) across all (i, d).
__global__ void igaa_softmax_and_gate_kernel(
    float* __restrict__       Z,        // [T, K]  → overwritten with π
    const float* __restrict__ tau_ptr,  // [1]
    const float* __restrict__ alpha_ptr,// [1]
    float* __restrict__       gate_out, // [T]
    int T, int K, int m_scfa)
{
	const int i = blockIdx.x;
	if (i >= T) return;

	extern __shared__ float smem[];
	float* row = smem;  // K floats

	const int k = threadIdx.x;
	const float tau = tau_ptr[0];
	const float alpha = alpha_ptr[0];

	// Load (and divide by tau) into shared.
	if (k < K)
		row[k] = Z[i * K + k] / fmaxf(tau, 1e-6f);
	__syncthreads();

	// Warp-friendly max reduction (K <= 32 typically; use a simple serial
	// reduction at thread 0 for simplicity).
	if (k == 0)
	{
		float mx = row[0];
		for (int kk = 1; kk < K; ++kk) if (row[kk] > mx) mx = row[kk];
		row[K + 0] = mx;  // overflow slot; assumes shared mem sized for K+2
	}
	__syncthreads();

	const float mx = row[K + 0];
	float exped = 0.0f;
	if (k < K)
	{
		exped = __expf(row[k] - mx);
		row[k] = exped;
	}
	__syncthreads();

	if (k == 0)
	{
		float s = 0.0f;
		for (int kk = 0; kk < K; ++kk) s += row[kk];
		row[K + 1] = s;
	}
	__syncthreads();

	const float denom = row[K + 1];
	float pi_ik = 0.0f;
	if (k < K)
	{
		pi_ik = row[k] / fmaxf(denom, 1e-12f);
		Z[i * K + k] = pi_ik;  // overwrite Z with π (for backward later)
	}
	__syncthreads();

	// Compute gate_i = Σ_{k < m_scfa} (π[i,k] − 1/K).  Single thread.
	if (k == 0)
	{
		const float inv_K = 1.0f / (float)K;
		float gate = 0.0f;
		for (int kk = 0; kk < m_scfa; ++kk) gate += (row[kk] / denom) - inv_K;
		gate_out[i] = alpha * gate;
	}
}

// Apply in-place: p[i, d] += scale[i] * p[i, d]   ↔   p[i,:] *= (1 + scale[i])
// Equivalent to a per-row scalar broadcast multiply-add.  Block per token,
// threads stride over m.
__global__ void igaa_apply_scale_kernel(
    float* __restrict__       p,        // [T, m]
    const float* __restrict__ scale,    // [T]
    int T, int m)
{
	const int i = blockIdx.x;
	if (i >= T) return;
	const float s = scale[i];
	const float k = 1.0f + s;
	float* row = p + (size_t)i * m;
	for (int d = threadIdx.x; d < m; d += blockDim.x)
		row[d] *= k;
}

bool igaa_apply_gate(const float* x,
                     float*       p,
                     const float* W_pi,
                     const float* b_pi,
                     const float* tau,
                     const float* alpha,
                     float*       pi_buf,
                     int          T,
                     int          m,
                     int          K,
                     int          m_scfa)
{
	if (!x || !p || !W_pi || !b_pi || !tau || !alpha || !pi_buf) return false;
	if (T <= 0 || m <= 0 || K <= 0 || K > 32 || m_scfa < 0 || m_scfa > K) return false;

	cudaStream_t stream = computeStream();

	// Step 1: Z[T, K] = X[T, m] · W_pi^T[m, K]
	// sgemm_rowmajor_atb: C[M,N] = A^T[M,K] B[K,N]  where A is stored [K, M] row-major.
	// Here: A = W_pi stored [K, m] row-major; B = X stored [T, m] row-major;
	// We want Z[T, K] = X[T, m] @ W_pi^T[m, K].
	//   = (W_pi @ X^T)^T  but easier: use sgemm_rowmajor with A=X, B=W_pi^T.
	//   sgemm_rowmajor expects A[M, K] B[K, N] → C[M, N] in row-major.
	//   With M=T, K_inner=m, N=K, A=X[T,m], B=W_pi has shape [K, m] row-major
	//   == W_pi^T has shape [m, K] column-major == W_pi[K, m] row-major used
	//   transposed.  Use sgemm_rowmajor_atb with A=W_pi[K, m] (so A^T = W_pi^T[m, K]),
	//   B=X^T??? — easier: use sgemm with explicit shapes.
	//
	// sgemm_rowmajor signature: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
	// We want C[T,K] = X[T,m] * W_pi^T[m,K]
	// Set M=T, N=K, K_inner=m, A=X (lda=m), B = ??? need W_pi^T but only have W_pi.
	// sgemm_rowmajor_atb: C[M,N] = A^T[M,K] B[K,N], A stored [K,M].
	// Reverse roles: let A = W_pi (stored [K, m]), then A^T[m, K]; want result
	// X · A^T... that's different.
	// Simplest: use sgemm with B = W_pi as if it were [K, m] but treat it as B^T.
	// cuBLAS uses column-major; sgemm_rowmajor wraps cublasSgemm with CUBLAS_OP_T
	// for both A and B and then transposes again — getting the right call is
	// fiddly without consulting gpu_blas.cu.
	//
	// Punt: do it via two calls.
	// First: Z[T, K] = X[T, m] @ W_pi^T[m, K].
	//   This is sgemm with A=X[T, m], B=W_pi[K, m] interpreted with B-transpose.
	// In sgemm_rowmajor_atb(M, N, K, A, B): C[M,N] = A^T[M,K] B[K,N], A stored [K, M].
	//   Let M=T, N=K, K_inner=m: A^T[T, m] B[m, K]. A stored [m, T]. We have X[T, m]
	//   stored row-major — interpreted as [m, T] column-major... no this is getting
	//   confusing.
	//
	// Cleanest: pre-transpose W_pi to W_pi_t[m, K] once (host-side at init) and
	// store it as a separate buffer.  But that doubles memory for W_pi.
	//
	// Alternative cleanest: write a tiny custom kernel for this small GEMM since
	// K is tiny (≤ 16).  T×m × m×K with K≤16 is a thin GEMM that doesn't benefit
	// much from cuBLAS anyway.  One block per (T, K), threads stride over m for the
	// inner product.
	//
	// For iter 26: use the custom kernel approach.  See igaa_thin_gemm_kernel below.

	// Step 1 via custom thin GEMM: Z[i, k] = Σ_d X[i, d] * W_pi[k, d] + b_pi[k]
	{
		const int block = 256;
		dim3 grid(T, K);
		extern __shared__ float smem_dummy[]; (void)smem_dummy;
		// Inline a custom thin GEMM call — see kernel below.
		extern void igaa_thin_gemm_launch(const float*, const float*, const float*,
		                                  float*, int, int, int, cudaStream_t);
		igaa_thin_gemm_launch(x, W_pi, b_pi, pi_buf, T, m, K, stream);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "[igaa_apply_gate] thin GEMM launch failed: %s\n",
			        cudaGetErrorString(err));
			return false;
		}
	}

	// Step 2-3: softmax + gate computation, in-place over pi_buf, writes gate to
	// a separate T-vector (allocated as the tail of pi_buf: pi_buf has T*(K+2)
	// or caller passes a separate scratch).  For iter 26 we expect the caller
	// to allocate pi_buf large enough; we'll use the last T floats for the
	// gate vector.  Caller convention: pi_buf size = T*K + T.
	float* gate_out = pi_buf + (size_t)T * (size_t)K;

	{
		const int threads = ((K + 31) / 32) * 32;
		const int shared_bytes = sizeof(float) * (K + 2);
		igaa_softmax_and_gate_kernel<<<T, threads, shared_bytes, stream>>>(
		    pi_buf, tau, alpha, gate_out, T, K, m_scfa);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "[igaa_apply_gate] softmax kernel failed: %s\n",
			        cudaGetErrorString(err));
			return false;
		}
	}

	// Step 4: p[i, :] += scale[i] · p[i, :]   ↔   p *= (1 + scale)
	{
		const int threads = 256;
		igaa_apply_scale_kernel<<<T, threads, 0, stream>>>(p, gate_out, T, m);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "[igaa_apply_gate] scale kernel failed: %s\n",
			        cudaGetErrorString(err));
			return false;
		}
	}

	return true;
}

// Thin GEMM kernel: Z[T, K] = X[T, m] @ W_pi^T[m, K] + bias[K]
// One block per (i, k) pair, threads cooperate on the m-dimension reduction.
__global__ void igaa_thin_gemm_kernel(
    const float* __restrict__ x,     // [T, m]
    const float* __restrict__ W_pi,  // [K, m]
    const float* __restrict__ b_pi,  // [K]
    float* __restrict__       Z,     // [T, K]
    int T, int m, int K)
{
	const int i = blockIdx.x;
	const int k = blockIdx.y;
	if (i >= T || k >= K) return;

	const float* xi = x + (size_t)i * m;
	const float* wk = W_pi + (size_t)k * m;

	float acc = 0.0f;
	for (int d = threadIdx.x; d < m; d += blockDim.x)
		acc += xi[d] * wk[d];

	// Block reduction via shared memory.
	__shared__ float ssum[256];
	const int tid = threadIdx.x;
	ssum[tid] = acc;
	__syncthreads();
	for (int off = blockDim.x / 2; off > 0; off >>= 1)
	{
		if (tid < off) ssum[tid] += ssum[tid + off];
		__syncthreads();
	}
	if (tid == 0) Z[(size_t)i * K + k] = ssum[0] + b_pi[k];
}

void igaa_thin_gemm_launch(const float* x, const float* W_pi, const float* b_pi,
                            float* Z, int T, int m, int K, cudaStream_t stream)
{
	const int threads = 256;
	dim3 grid(T, K);
	igaa_thin_gemm_kernel<<<grid, threads, 0, stream>>>(x, W_pi, b_pi, Z, T, m, K);
}

// ===================================================================
// BACKWARD KERNELS
// ===================================================================

// Compute dG[i] = Σ_d dp_out[i,d] · p_save[i,d]  (per-row inner product)
// And modify dp in place: dp[i,d] *= (1 + scale[i])
// One block per token; threads stride over m for the reduction.
__global__ void igaa_scale_bwd_kernel(
    float* __restrict__       dp,        // [T, m] in/out
    const float* __restrict__ p_save,    // [T, m]
    const float* __restrict__ scale,     // [T] (= α · gate_pure)
    float* __restrict__       dG_out,    // [T]
    int T, int m)
{
	const int i = blockIdx.x;
	if (i >= T) return;
	const float s = scale[i];
	const float k_factor = 1.0f + s;
	float* dp_row = dp + (size_t)i * m;
	const float* p_row = p_save + (size_t)i * m;

	float acc = 0.0f;
	const int tid = threadIdx.x;
	for (int d = tid; d < m; d += blockDim.x)
	{
		// Compute dG contribution BEFORE in-place scale modifies dp.
		acc += dp_row[d] * p_row[d];
	}

	__shared__ float ssum[256];
	ssum[tid] = acc;
	__syncthreads();
	for (int off = blockDim.x / 2; off > 0; off >>= 1)
	{
		if (tid < off) ssum[tid] += ssum[tid + off];
		__syncthreads();
	}
	if (tid == 0) dG_out[i] = ssum[0];
	__syncthreads();

	// Now modify dp in place.
	for (int d = tid; d < m; d += blockDim.x)
		dp_row[d] *= k_factor;
}

// Build dπ from dG and reduce dalpha.
// dπ[i, k] = α · dG[i]  for k < m_scfa, else 0
// dalpha[0] += Σ_i dG[i] · gate_pure[i]   where gate_pure[i] = scale[i] / α
//             (special case α=0: gate_pure is undefined; we use the analytic
//             form gate_pure = Σ_{k<m_scfa} (π[i,k] − 1/K) from π directly.)
__global__ void igaa_gate_bwd_kernel(
    const float* __restrict__ dG,        // [T]
    const float* __restrict__ pi,        // [T, K] from forward (softmax output)
    const float* __restrict__ alpha,     // [1]
    float* __restrict__       dpi,       // [T, K] out
    float* __restrict__       dalpha_atom, // [1] in/out — atomicAdd target
    int T, int K, int m_scfa)
{
	const int i = blockIdx.x;
	if (i >= T) return;
	const int k = threadIdx.x;
	const float a = alpha[0];
	const float dG_i = dG[i];

	// dπ[i, k] = α · dG_i for k < m_scfa else 0.
	if (k < K)
		dpi[(size_t)i * K + k] = (k < m_scfa) ? (a * dG_i) : 0.0f;

	__syncthreads();

	// gate_pure_i = Σ_{k<m_scfa} (π[i,k] − 1/K), computed analytically.
	if (k == 0)
	{
		const float inv_K = 1.0f / (float)K;
		float gate_pure = 0.0f;
		for (int kk = 0; kk < m_scfa; ++kk)
			gate_pure += pi[(size_t)i * K + kk] - inv_K;
		const float contrib = dG_i * gate_pure;
		atomicAdd(dalpha_atom, contrib);
	}
}

// Softmax backward: dz[i, k] = (π[i,k] / τ) · (dπ[i,k] − Σ_j π[i,j] · dπ[i,j])
// One block per token; threads index K (≤ 32 typically).
__global__ void igaa_softmax_bwd_kernel(
    const float* __restrict__ pi,        // [T, K]
    const float* __restrict__ dpi,       // [T, K]
    const float* __restrict__ tau_ptr,   // [1]
    float* __restrict__       dz,        // [T, K] out
    int T, int K)
{
	const int i = blockIdx.x;
	if (i >= T) return;
	const int k = threadIdx.x;
	const float tau = fmaxf(tau_ptr[0], 1e-6f);

	extern __shared__ float smem[];
	float* row_pi = smem;          // K floats
	float* row_dpi = smem + K;     // K floats

	if (k < K)
	{
		row_pi[k]  = pi[(size_t)i * K + k];
		row_dpi[k] = dpi[(size_t)i * K + k];
	}
	__syncthreads();

	// dot = Σ_j π[i,j] · dπ[i,j]
	float dot = 0.0f;
	if (k == 0)
	{
		for (int kk = 0; kk < K; ++kk) dot += row_pi[kk] * row_dpi[kk];
		smem[2 * K] = dot;
	}
	__syncthreads();
	dot = smem[2 * K];

	if (k < K)
		dz[(size_t)i * K + k] = (row_pi[k] / tau) * (row_dpi[k] - dot);
}

// Column-sum reduction: db_pi[k] += Σ_i dz[i, k]
// One block per k; threads stride over T.
__global__ void igaa_bias_grad_kernel(
    const float* __restrict__ dz,        // [T, K]
    float* __restrict__       db_pi,     // [K] in/out (accumulates)
    int T, int K)
{
	const int k = blockIdx.x;
	if (k >= K) return;
	const int tid = threadIdx.x;
	float acc = 0.0f;
	for (int i = tid; i < T; i += blockDim.x)
		acc += dz[(size_t)i * K + k];

	__shared__ float ssum[256];
	ssum[tid] = acc;
	__syncthreads();
	for (int off = blockDim.x / 2; off > 0; off >>= 1)
	{
		if (tid < off) ssum[tid] += ssum[tid + off];
		__syncthreads();
	}
	if (tid == 0) db_pi[k] += ssum[0];
}

// W_pi gradient: dW_pi[k, d] += Σ_i dz[i, k] · x[i, d]
// One block per (k, d-tile); threads reduce over i.
// For Gate-0 with K ≤ 16 and m=2048, total blocks = K · ceil(m/256) ≈ 128 — fine.
__global__ void igaa_dW_pi_kernel(
    const float* __restrict__ x,         // [T, m]
    const float* __restrict__ dz,        // [T, K]
    float* __restrict__       dW_pi,     // [K, m] in/out (accumulates)
    int T, int m, int K)
{
	const int k = blockIdx.x;
	const int d_blk = blockIdx.y * 256 + threadIdx.x;
	if (k >= K || d_blk >= m) return;
	float acc = 0.0f;
	for (int i = 0; i < T; ++i)
		acc += dz[(size_t)i * K + k] * x[(size_t)i * m + d_blk];
	dW_pi[(size_t)k * m + d_blk] += acc;
}

// dx accumulation: dx[i, d] += Σ_k dz[i, k] · W_pi[k, d]
// One block per (i, d-tile); thread does a single (i, d) output.
__global__ void igaa_dx_accum_kernel(
    const float* __restrict__ dz,        // [T, K]
    const float* __restrict__ W_pi,      // [K, m]
    float* __restrict__       dx,        // [T, m] in/out (accumulates)
    int T, int m, int K)
{
	const int i = blockIdx.x;
	const int d = blockIdx.y * 256 + threadIdx.x;
	if (i >= T || d >= m) return;
	float acc = 0.0f;
	for (int k = 0; k < K; ++k)
		acc += dz[(size_t)i * K + k] * W_pi[(size_t)k * m + d];
	dx[(size_t)i * m + d] += acc;
}

bool igaa_backward(float*       dp,
                   const float* p_save,
                   const float* pi_buf,
                   const float* x,
                   const float* W_pi,
                   const float* alpha,
                   const float* tau,
                   float*       dW_pi,
                   float*       db_pi,
                   float*       dalpha,
                   float*       dx,
                   float*       scratch,
                   int          T,
                   int          m,
                   int          K,
                   int          m_scfa)
{
	if (!dp || !p_save || !pi_buf || !x || !W_pi || !alpha || !tau ||
	    !dW_pi || !db_pi || !dalpha || !dx || !scratch) return false;
	if (T <= 0 || m <= 0 || K <= 0 || K > 32 || m_scfa < 0 || m_scfa > K) return false;

	cudaStream_t stream = computeStream();

	// scratch layout: [dpi : T*K | dG : T]  (same as pi_buf layout)
	float* dpi_buf = scratch;
	float* dG      = scratch + (size_t)T * (size_t)K;
	const float* pi_ro    = pi_buf;
	const float* scale_ro = pi_buf + (size_t)T * (size_t)K;

	// Step (a): compute dG, modify dp in place to dp_in.
	{
		const int threads = 256;
		igaa_scale_bwd_kernel<<<T, threads, 0, stream>>>(
		    dp, p_save, scale_ro, dG, T, m);
	}

	// Step (b): build dπ from dG, accumulate dalpha atomically.
	{
		const int threads = ((K + 31) / 32) * 32;
		igaa_gate_bwd_kernel<<<T, threads, 0, stream>>>(
		    dG, pi_ro, alpha, dpi_buf, dalpha, T, K, m_scfa);
	}

	// Step (c): softmax backward dπ → dz.  Reuse pi_buf as dz output too?
	// We have dπ in dpi_buf and need dz somewhere.  Overwrite dpi_buf in place
	// (since dz has the same shape as dπ).
	{
		const int threads = ((K + 31) / 32) * 32;
		const int shared_bytes = sizeof(float) * (2 * K + 2);
		igaa_softmax_bwd_kernel<<<T, threads, shared_bytes, stream>>>(
		    pi_ro, dpi_buf, tau, dpi_buf, T, K);
		// dpi_buf now holds dz.
	}

	// Step (d): db_pi += Σ_i dz[:, k]
	{
		const int threads = 256;
		igaa_bias_grad_kernel<<<K, threads, 0, stream>>>(dpi_buf, db_pi, T, K);
	}

	// Step (e): dW_pi[k, d] += Σ_i dz[i, k] · x[i, d]
	{
		const int d_tile = 256;
		const int d_grid = (m + d_tile - 1) / d_tile;
		dim3 grid(K, d_grid);
		igaa_dW_pi_kernel<<<grid, d_tile, 0, stream>>>(x, dpi_buf, dW_pi, T, m, K);
	}

	// Step (f): dx[i, d] += Σ_k dz[i, k] · W_pi[k, d]
	{
		const int d_tile = 256;
		const int d_grid = (m + d_tile - 1) / d_tile;
		dim3 grid(T, d_grid);
		igaa_dx_accum_kernel<<<grid, d_tile, 0, stream>>>(dpi_buf, W_pi, dx, T, m, K);
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "[igaa_backward] kernel launch failed: %s\n",
		        cudaGetErrorString(err));
		return false;
	}

	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
