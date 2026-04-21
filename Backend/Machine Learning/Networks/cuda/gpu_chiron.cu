// CHIRON reversible-flow transformer GPU primitives.
//
// See gpu_chiron.h for the wrapper declarations and research/CHIRON_framework.md
// for the math. This file implements the CUDA kernels and host wrappers.
//
// Design notes:
//   - Shears are trivially element-wise and use a simple saxpy-like kernel.
//     No shared memory, one thread per element.
//   - ReLN forward/inverse are per-row kernels: one block per token, warp
//     reductions for mean/variance. Direct port of the layernorm kernel
//     style from gpu_kernels.cu, extended with stats output/input.
//   - Sketch project/lift are batched matvecs. We route through cuBLAS
//     sgemm_rowmajor wrappers since the math is pure GEMM.

#include "gpu_chiron.h"
#include "gpu_device.h"
#include "gpu_blas.h"
#include "gpu_kernels.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>

namespace glades {
namespace gpu {

namespace {

#define GLADES_CUDA_CHECK(call)                                               \
	do {                                                                      \
		cudaError_t err_ = (call);                                            \
		if (err_ != cudaSuccess) {                                            \
			fprintf(stderr, "[chiron-cuda] %s:%d  %s  -> %s\n",               \
			        __FILE__, __LINE__, #call, cudaGetErrorString(err_));     \
			return false;                                                     \
		}                                                                     \
	} while (0)

static constexpr int kBlockElem = 256;
static constexpr int kMaxBlockRow = 1024;

inline int rowBlockSize(int cols)
{
	int b = cols < kMaxBlockRow ? cols : kMaxBlockRow;
	b = ((b + 31) / 32) * 32;
	if (b < 32) b = 32;
	if (b > kMaxBlockRow) b = kMaxBlockRow;
	return b;
}

// ---------------------------------------------------------------------------
// Warp/block reductions. Mirror the implementations in gpu_kernels.cu so each
// translation unit has its own static copy (separable compilation is off).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warpReduceSum(float val)
{
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}

__device__ float blockReduceSum(float val, float* smem)
{
	int lane = threadIdx.x & 31;
	int wid  = threadIdx.x >> 5;

	val = warpReduceSum(val);
	if (lane == 0) smem[wid] = val;
	__syncthreads();

	int numWarps = (blockDim.x + 31) / 32;
	val = (threadIdx.x < (unsigned)numWarps) ? smem[threadIdx.x] : 0.0f;
	if (wid == 0) val = warpReduceSum(val);
	return val;
}

} // anonymous namespace

// ===========================================================================
//  1. Symplectic shears — element-wise add/subtract.
// ===========================================================================

namespace {

__global__ void chiron_shear_add_kernel(float* __restrict__ p,
                                        const float* __restrict__ u, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) p[i] += u[i];
}

__global__ void chiron_shear_sub_kernel(float* __restrict__ p,
                                        const float* __restrict__ u, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) p[i] -= u[i];
}

} // anonymous namespace

bool chiron_shear_add(float* p, const float* u, int n)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	chiron_shear_add_kernel<<<grid, kBlockElem, 0, computeStream()>>>(p, u, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_shear_sub(float* p, const float* u, int n)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	chiron_shear_sub_kernel<<<grid, kBlockElem, 0, computeStream()>>>(p, u, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  2. Reversible LayerNorm (ReLN) forward.
// ===========================================================================
//
// One block per token.  Each block computes the row mean and variance via
// warp/block reductions, writes the normalized row, and records
// (mu, log(sigma)) into stats[row, 0..1].

namespace {

__global__ void chiron_reln_forward_rows(const float* __restrict__ q_in,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float eps, int cols,
                                         float* __restrict__ q_out,
                                         float* __restrict__ stats)
{
	int row = blockIdx.x;
	const float* xRow = q_in + (size_t)row * cols;
	float*       oRow = q_out + (size_t)row * cols;

	extern __shared__ float smem[];
	float* sSumA = smem;
	float* sSumB = smem + (blockDim.x / 32 + 1);

	__shared__ float sMean, sSigma;

	// Pass 1: mean.
	float s = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) s += xRow[i];
	s = blockReduceSum(s, sSumA);
	if (threadIdx.x == 0) sMean = s / (float)cols;
	__syncthreads();

	const float mu = sMean;

	// Pass 2: variance.
	float v = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float d = xRow[i] - mu;
		v += d * d;
	}
	v = blockReduceSum(v, sSumB);

	if (threadIdx.x == 0) {
		float var = v / (float)cols + eps;
		sSigma = sqrtf(var);
	}
	__syncthreads();

	const float sigma = sSigma;
	const float inv_sigma = 1.0f / sigma;

	// Pass 3: normalize + affine.
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		oRow[i] = gamma[i] * (xRow[i] - mu) * inv_sigma + beta[i];

	// Stats: { mu, log(sigma) } for this row.
	if (threadIdx.x == 0) {
		stats[(size_t)row * 2 + 0] = mu;
		stats[(size_t)row * 2 + 1] = logf(sigma);
	}
}

} // anonymous namespace

bool chiron_reln_forward(const float* q_in, float* q_out, float* stats,
                          const float* gamma, const float* beta,
                          int T, int m, float eps)
{
	if (T <= 0 || m <= 0) return true;
	int block = rowBlockSize(m);
	int smemBytes = (block / 32 + 2) * 2 * sizeof(float);
	chiron_reln_forward_rows<<<T, block, smemBytes, computeStream()>>>(
		q_in, gamma, beta, eps, m, q_out, stats);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  3. Reversible LayerNorm (ReLN) inverse.
// ===========================================================================
//
// Given q_out, stats[row, 0..1] = { mu, log(sigma) }, recover q_in.
// Single-pass per row.

namespace {

__global__ void chiron_reln_inverse_rows(const float* __restrict__ q_out,
                                         const float* __restrict__ stats,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         int cols,
                                         float* __restrict__ q_in)
{
	int row = blockIdx.x;
	const float* yRow = q_out + (size_t)row * cols;
	float*       xRow = q_in  + (size_t)row * cols;

	const float mu    = stats[(size_t)row * 2 + 0];
	const float sigma = expf(stats[(size_t)row * 2 + 1]);

	for (int i = threadIdx.x; i < cols; i += blockDim.x)
	{
		// x = sigma * (y - beta) / gamma + mu
		xRow[i] = sigma * (yRow[i] - beta[i]) / gamma[i] + mu;
	}
}

} // anonymous namespace

bool chiron_reln_inverse(const float* q_out, float* q_in, const float* stats,
                          const float* gamma, const float* beta,
                          int T, int m)
{
	if (T <= 0 || m <= 0) return true;
	int block = rowBlockSize(m);
	chiron_reln_inverse_rows<<<T, block, 0, computeStream()>>>(
		q_out, stats, gamma, beta, m, q_in);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  3b. Reversible LayerNorm backward (wraps layernorm_backward).
// ===========================================================================
//
// ReLN forward is numerically identical to LayerNorm forward — the only
// novelty is where (mu, log_sigma) are stored. For the backward pass
// we convert the external (mu, log_sigma) stats buffer into the
// (mean[T], invStd[T]) format that the existing layernorm_backward
// kernel expects, then defer to that kernel.

namespace {

// Kernel to split [T, 2] (mu, log_sigma) -> two separate [T] buffers
// (mean, invStd).  One thread per row.
__global__ void chiron_stats_split_kernel(const float* __restrict__ stats,
                                          int T,
                                          float* __restrict__ mean,
                                          float* __restrict__ invStd)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < T)
	{
		mean[i]   = stats[(size_t)i * 2 + 0];
		invStd[i] = expf(-stats[(size_t)i * 2 + 1]);
	}
}

} // anonymous namespace

bool chiron_reln_backward(const float* dq_out, const float* q_in,
                           const float* gamma, const float* stats,
                           int T, int m,
                           float* dq_in, float* dgamma, float* dbeta,
                           float* scratch_stats_split)
{
	if (T <= 0 || m <= 0) return true;

	// Split stats[T, 2] into mean[T] and invStd[T] via a small kernel.
	float* d_mean  = scratch_stats_split;
	float* d_invStd = scratch_stats_split + T;
	const int grid = (T + kBlockElem - 1) / kBlockElem;
	chiron_stats_split_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
	    stats, T, d_mean, d_invStd);
	GLADES_CUDA_CHECK(cudaGetLastError());

	// Delegate to the existing LayerNorm backward kernel, which
	// already handles the complex dx / dgamma / dbeta computation.
	return layernorm_backward(dq_out, q_in, gamma, d_mean, d_invStd,
	                           T, m, dq_in, dgamma, dbeta);
}

// ===========================================================================
//  4. Sketch project — Z = X · S^T   (X: [T, Ntok], S: [r, Ntok], Z: [T, r]).
// ===========================================================================
//
// Routed through cuBLAS sgemm with B-transpose: this is exactly
// sgemm_rowmajor_abt(M=T, N=r, K=Ntok, A=X, B=S, C=Z).

bool chiron_sketch_project(const float* X, const float* S,
                            int T, int Ntok, int r, float* Z)
{
	if (T <= 0 || Ntok <= 0 || r <= 0) return true;
	// Z = 1.0 * X · S^T + 0.0 * Z, all row-major, A is [T, Ntok] ld=Ntok,
	// B is [r, Ntok] ld=Ntok, C is [T, r] ld=r.
	return sgemm_rowmajor_abt(T, r, Ntok,
	                           1.0f,
	                           X, Ntok,
	                           S, Ntok,
	                           0.0f,
	                           Z, r);
}

// ===========================================================================
//  5. Sketch lift-add — X += (1/r) · R · S   (R: [T, r], S: [r, Ntok], X: [T, Ntok]).
// ===========================================================================

bool chiron_sketch_lift_add(float* X, const float* R, const float* S,
                             int T, int Ntok, int r)
{
	if (T <= 0 || Ntok <= 0 || r <= 0) return true;
	const float alpha = 1.0f / static_cast<float>(r);
	// Standard row-major GEMM: C = alpha * A · B + 1.0 * C.
	// A = R [T, r] ld=r, B = S [r, Ntok] ld=Ntok, C = X [T, Ntok] ld=Ntok.
	return sgemm_rowmajor(T, Ntok, r,
	                       alpha,
	                       R, r,
	                       S, Ntok,
	                       1.0f,
	                       X, Ntok);
}

// ===========================================================================
//  6. Symplectic attention shear — composition wrapper.
// ===========================================================================
//
// Layered on top of existing GPU primitives:
//   Q = q · Wq       (sgemm_rowmajor: [T,m] · [m,dH] -> [T,dH])
//   K = q · Wk
//   V = q · Wv
//   O = flash_attention_multihead_forward(Q, K, V)
//   p += ± O · Wo    (sgemm_rowmajor with alpha=±1, beta=1: accumulates into p)
//
// For the inverse shear, the caller passes invert=true and we use alpha=-1
// in the final GEMM (equivalent to p -= Y(q) since q is unchanged).

bool chiron_attention_shear(const float* q, float* p,
                             const float* Wq, const float* Wk,
                             const float* Wv, const float* Wo,
                             int T, int m, int nHeads, int nKVHeads, int dHead,
                             bool causal, bool invert,
                             float* scratch_Q, float* scratch_K,
                             float* scratch_V, float* scratch_O)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel    = nHeads   * dHead;
	const int dModelKV  = nKVHeads * dHead;

	// Q = q · Wq.  q: [T, m] ld=m; Wq: [m, dModel] ld=dModel; Q: [T, dModel] ld=dModel.
	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wq, dModel, 0.0f, scratch_Q, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wk, dModelKV, 0.0f, scratch_K, dModelKV))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wv, dModelKV, 0.0f, scratch_V, dModelKV))
		return false;

	// Flash attention: O[T, dModel] = softmax(Q K^T / sqrt(dHead)) · V.
	if (!flash_attention_multihead_forward(scratch_Q, scratch_K, scratch_V,
	                                        T, nHeads, nKVHeads,
	                                        dHead, dModel, dModelKV,
	                                        causal, scratch_O))
		return false;

	// p += ±  O · Wo.  O: [T, dModel] ld=dModel; Wo: [dModel, m] ld=m; p: [T, m] ld=m.
	const float sign = invert ? -1.0f : 1.0f;
	if (!sgemm_rowmajor(T, m, dModel, sign, scratch_O, dModel, Wo, m, 1.0f, p, m))
		return false;

	return true;
}

// ===========================================================================
//  6b. Symplectic attention shear — backward.
// ===========================================================================
//
// Backward through the forward:
//   Q = q · Wq,   K = q · Wk,   V = q · Wv
//   O = flash_attn(Q, K, V)
//   Y = O · Wo
//   p_new = p + Y
//
// Gradients (composition of chain rule):
//   dL/dp      = dL/dp_new                                         (identity)
//   dL/dY      = dL/dp_new                                         (identity)
//   dL/dO      = dL/dY · Wo^T          (sgemm_rowmajor_abt)
//   dL/dWo    += O^T · dL/dY           (sgemm_rowmajor_atb)
//   dL/dQ, dK, dV  ← flash_attention_multihead_backward(Q, K, V, O, dO)
//   dL/dq     += dL/dQ · Wq^T          (sgemm_rowmajor_abt, beta=1)
//   dL/dq     += dL/dK · Wk^T
//   dL/dq     += dL/dV · Wv^T
//   dL/dWq    += q^T · dL/dQ           (sgemm_rowmajor_atb, beta=1)
//   dL/dWk    += q^T · dL/dK
//   dL/dWv    += q^T · dL/dV
//
// All the sgemms are existing cuBLAS wrappers.  The flash-attention
// backward is the existing kernel.  This function is pure orchestration.

bool chiron_attention_shear_backward(
    const float* q, const float* dp_new,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    int T, int m, int nHeads, int nKVHeads, int dHead,
    bool causal,
    float* dq,
    float* dWq, float* dWk, float* dWv, float* dWo,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel    = nHeads   * dHead;
	const int dModelKV  = nKVHeads * dHead;

	// --- Recompute forward intermediates (Q, K, V, O) from q ---
	if (!sgemm_rowmajor(T, dModel,   m, 1.0f, q, m, Wq, dModel,   0.0f, sQ, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wk, dModelKV, 0.0f, sK, dModelKV))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wv, dModelKV, 0.0f, sV, dModelKV))
		return false;
	if (!flash_attention_multihead_forward(sQ, sK, sV,
	                                        T, nHeads, nKVHeads,
	                                        dHead, dModel, dModelKV,
	                                        causal, sO))
		return false;

	// --- Output-projection backward: dL/dO = dL/dp_new · Wo^T ---
	// dp_new: [T, m]; Wo: [dModel, m]; dO: [T, dModel]; dO = dp_new · Wo^T
	if (!sgemm_rowmajor_abt(T, dModel, m, 1.0f, dp_new, m, Wo, m, 0.0f, sdO, dModel))
		return false;

	// --- dL/dWo += O^T · dp_new ---
	// O: [T, dModel]; dp_new: [T, m]; dWo: [dModel, m]; dWo += O^T · dp_new
	if (!sgemm_rowmajor_atb(dModel, m, T, 1.0f, sO, dModel, dp_new, m, 1.0f, dWo, m))
		return false;

	// --- Attention backward: given dO, produce dQ, dK, dV ---
	// Note: flash_attention_multihead_backward WRITES to dQ and ACCUMULATES
	// into dK, dV (via atomicAdd for the shared K/V case).  Zero them first.
	// We use sdQ/sdK/sdV as LOCAL dQ/dK/dV accumulators.
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdQ, 0, sizeof(float) * T * dModel,   computeStream()));
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdK, 0, sizeof(float) * T * dModelKV, computeStream()));
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdV, 0, sizeof(float) * T * dModelKV, computeStream()));
	if (!flash_attention_multihead_backward(sQ, sK, sV, sO, sdO,
	                                         T, nHeads, nKVHeads,
	                                         dHead, dModel, dModelKV,
	                                         causal, sdQ, sdK, sdV))
		return false;

	// --- Project dQ/dK/dV back into q space.  dq += dQ · Wq^T (etc.) ---
	if (!sgemm_rowmajor_abt(T, m, dModel, 1.0f, sdQ, dModel, Wq, dModel, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_abt(T, m, dModelKV, 1.0f, sdK, dModelKV, Wk, dModelKV, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_abt(T, m, dModelKV, 1.0f, sdV, dModelKV, Wv, dModelKV, 1.0f, dq, m))
		return false;

	// --- Weight gradients: dWq += q^T · dQ (etc.) ---
	if (!sgemm_rowmajor_atb(m, dModel, T, 1.0f, q, m, sdQ, dModel, 1.0f, dWq, dModel))
		return false;
	if (!sgemm_rowmajor_atb(m, dModelKV, T, 1.0f, q, m, sdK, dModelKV, 1.0f, dWk, dModelKV))
		return false;
	if (!sgemm_rowmajor_atb(m, dModelKV, T, 1.0f, q, m, sdV, dModelKV, 1.0f, dWv, dModelKV))
		return false;

	return true;
}

// ===========================================================================
//  7. Symplectic attention shear — BF16-input flash-attention variant.
// ===========================================================================
//
// Same control flow as chiron_attention_shear, but the flash-attention core
// runs with BF16 Q/K/V. On profiling hardware (RTX 4080 SUPER), the BF16
// kernel is 50-200x faster than the FP32 variant at training-scale T and
// dHead; this variant closes the gap between CHIRON's attention path and
// cuBLAS's sgemm throughput.
//
// Cast overhead: one device-side BF16 cast per Q/K/V buffer, each O(T*dModel)
// — negligible compared to the T*T*dHead attention work.

bool chiron_attention_shear_bf16(const float* q, float* p,
                                  const float* Wq, const float* Wk,
                                  const float* Wv, const float* Wo,
                                  int T, int m, int nHeads, int nKVHeads, int dHead,
                                  bool causal, bool invert,
                                  float* scratch_Q, float* scratch_K,
                                  float* scratch_V, float* scratch_O,
                                  uint16_t* scratch_Qbf, uint16_t* scratch_Kbf,
                                  uint16_t* scratch_Vbf)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel    = nHeads   * dHead;
	const int dModelKV  = nKVHeads * dHead;

	// Projections in FP32 (fast on cuBLAS).
	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wq, dModel, 0.0f, scratch_Q, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wk, dModelKV, 0.0f, scratch_K, dModelKV))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wv, dModelKV, 0.0f, scratch_V, dModelKV))
		return false;

	// Cast Q/K/V FP32 -> BF16 for the flash-attention core.
	if (!cast_f32_to_bf16(scratch_Q, scratch_Qbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!cast_f32_to_bf16(scratch_K, scratch_Kbf, static_cast<size_t>(T) * dModelKV))
		return false;
	if (!cast_f32_to_bf16(scratch_V, scratch_Vbf, static_cast<size_t>(T) * dModelKV))
		return false;

	// BF16-input flash attention. Output stays FP32 (written to scratch_O).
	if (!flash_attention_multihead_forward_bf16(scratch_Qbf, scratch_Kbf, scratch_Vbf,
	                                             T, nHeads, nKVHeads,
	                                             dHead, dModel, dModelKV,
	                                             causal, scratch_O))
		return false;

	// Output projection (FP32).
	const float sign = invert ? -1.0f : 1.0f;
	if (!sgemm_rowmajor(T, m, dModel, sign, scratch_O, dModel, Wo, m, 1.0f, p, m))
		return false;

	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
