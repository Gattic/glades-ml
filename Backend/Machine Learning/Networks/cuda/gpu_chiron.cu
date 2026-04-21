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

// ---------------------------------------------------------------------------
// BF16-tensor-core variant of chiron_attention_shear_tiled.  Same math as
// chiron_attention_shear, but the attention-core GEMMs (QK^T and P·V) run
// with BF16 inputs via flash_attention_cublas_tiled_bf16.  On RTX 4080 SUPER
// this is ~2× over the TF32 variant on large shapes.  The Q/K/V projections
// and output projection remain FP32 (they're only 3+1 GEMMs per block and
// gain little from BF16).
//
// Extra scratch (all caller-owned):
//   scratch_Qbf16, scratch_Kbf16, scratch_Vbf16, scratch_Pbf16 — BF16
//     staging buffers for the BF16 attention kernel.  Sized [T, dModel]
//     for Q/K/V and [nH, T, T] for P.
bool chiron_attention_shear_bf16_tiled(const float* q, float* p,
                                         const float* Wq, const float* Wk,
                                         const float* Wv, const float* Wo,
                                         int T, int m, int nHeads, int dHead,
                                         bool causal, bool invert,
                                         float* scratch_Q, float* scratch_K,
                                         float* scratch_V, float* scratch_O,
                                         float* scratch_S,
                                         unsigned short* scratch_Qbf16,
                                         unsigned short* scratch_Kbf16,
                                         unsigned short* scratch_Vbf16,
                                         unsigned short* scratch_Pbf16)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel = nHeads * dHead;

	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wq, dModel, 0.0f, scratch_Q, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wk, dModel, 0.0f, scratch_K, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wv, dModel, 0.0f, scratch_V, dModel))
		return false;

	if (!flash_attention_cublas_tiled_bf16(
	        scratch_Q, scratch_K, scratch_V,
	        T, nHeads, dHead, dModel, causal,
	        scratch_O, scratch_S,
	        scratch_Qbf16, scratch_Kbf16, scratch_Vbf16, scratch_Pbf16))
		return false;

	const float sign = invert ? -1.0f : 1.0f;
	if (!sgemm_rowmajor(T, m, dModel, sign, scratch_O, dModel, Wo, m, 1.0f, p, m))
		return false;
	return true;
}

// BF16-weight variant of chiron_attention_shear_bf16_tiled.  Takes BF16
// weight pointers directly — no per-layer FP32 cast scratch needed for
// weights.  Q/K/V/O projections run through sgemm_rowmajor_bf16
// (BF16 x BF16 -> FP32 via BF16 tensor cores, ~2x TF32 throughput on
// Ampere/Ada).
//
// Extra scratch (caller-owned):
//   scratch_qbf   [T, m]     — BF16 cast of q (one cast per layer)
//   scratch_Obf   [T, dModel] — BF16 cast of attention output for Wo proj
//
// Combined with --bf16-weights on the trainer: eliminates 4 weight-cast
// kernels per layer and replaces 4 TF32-TC GEMMs with BF16-TC GEMMs.
// Projected 8% e2e throughput improvement at 2 B scale (see
// research/BF16_PROJECTION_OPT.md).
bool chiron_attention_shear_bf16w_tiled(const float* q, float* p,
                                          const unsigned short* Wq_bf,
                                          const unsigned short* Wk_bf,
                                          const unsigned short* Wv_bf,
                                          const unsigned short* Wo_bf,
                                          int T, int m, int nHeads, int dHead,
                                          bool causal, bool invert,
                                          unsigned short* scratch_qbf,
                                          unsigned short* scratch_Obf,
                                          float* scratch_Q, float* scratch_K,
                                          float* scratch_V, float* scratch_O,
                                          float* scratch_S,
                                          unsigned short* scratch_Qbf16,
                                          unsigned short* scratch_Kbf16,
                                          unsigned short* scratch_Vbf16,
                                          unsigned short* scratch_Pbf16)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel = nHeads * dHead;

	// Cast q FP32 -> BF16 once per layer.
	if (!cast_f32_to_bf16(q, scratch_qbf, static_cast<size_t>(T) * m))
		return false;

	// BF16 x BF16 -> FP32 projections via BF16 tensor cores.
	if (!sgemm_rowmajor_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wq_bf, dModel, 0.0f, scratch_Q, dModel))
		return false;
	if (!sgemm_rowmajor_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wk_bf, dModel, 0.0f, scratch_K, dModel))
		return false;
	if (!sgemm_rowmajor_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wv_bf, dModel, 0.0f, scratch_V, dModel))
		return false;

	// Attention core (BF16 inputs, FP32 output).
	if (!flash_attention_cublas_tiled_bf16(
	        scratch_Q, scratch_K, scratch_V,
	        T, nHeads, dHead, dModel, causal,
	        scratch_O, scratch_S,
	        scratch_Qbf16, scratch_Kbf16, scratch_Vbf16, scratch_Pbf16))
		return false;

	// Cast scratch_O -> BF16 for the output projection.
	if (!cast_f32_to_bf16(scratch_O, scratch_Obf, static_cast<size_t>(T) * dModel))
		return false;

	// Output projection: p += sign * scratch_Obf @ Wo_bf (BF16 TC, FP32 accum).
	const float sign = invert ? -1.0f : 1.0f;
	if (!sgemm_rowmajor_bf16(T, m, dModel, sign, scratch_Obf, dModel, Wo_bf, m, 1.0f, p, m))
		return false;
	return true;
}

// BF16-weight backward counterpart to chiron_attention_shear_bf16w_tiled.
// Eliminates weight casts on the backward path by using BF16-TC GEMMs for
// the projections, the dq-projection (abt), and the forward recompute.
// The weight-grad GEMMs (dWq += q^T · sdQ etc.) stay FP32 — caller may
// pair with --bf16-grads to accumulate them into BF16 persistent storage
// via bf16_accum_axpy downstream.
//
// Extra scratch (caller-owned):
//   scratch_qbf    [T, m]      — reused for q cast and for dp_new cast
//   scratch_sdbf   [T, dModel] — rotates through sdQ/sdK/sdV casts
//
// Savings vs chiron_attention_shear_backward_tiled (with bf16-weights
// caller casting each weight to FP32): 7 fewer weight casts per layer,
// 7 BF16-TC GEMMs instead of TF32-TC (projection + abt dq-projection),
// net ~200 MB HBM cast traffic saved per layer at 2 B scale.
bool chiron_attention_shear_backward_bf16w_tiled(
    const float* q, const float* dp_new,
    const unsigned short* Wq_bf, const unsigned short* Wk_bf,
    const unsigned short* Wv_bf, const unsigned short* Wo_bf,
    int T, int m, int nHeads, int dHead,
    bool causal,
    float* dq,
    float* dWq, float* dWk, float* dWv, float* dWo,
    unsigned short* scratch_qbf,
    unsigned short* scratch_sdbf,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV,
    float* scratch_P, float* scratch_dP)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel = nHeads * dHead;

	// 1. Cast q -> BF16 for projections.
	if (!cast_f32_to_bf16(q, scratch_qbf, static_cast<size_t>(T) * m))
		return false;

	// 2. Forward recompute: Q/K/V projections via BF16-TC GEMMs.
	if (!sgemm_rowmajor_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wq_bf, dModel, 0.0f, sQ, dModel))
		return false;
	if (!sgemm_rowmajor_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wk_bf, dModel, 0.0f, sK, dModel))
		return false;
	if (!sgemm_rowmajor_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wv_bf, dModel, 0.0f, sV, dModel))
		return false;
	if (!flash_attention_cublas_tiled(sQ, sK, sV, T, nHeads, dHead, dModel, causal,
	                                    sO, scratch_P))
		return false;

	// 3. Output-projection backward.  dO = dp_new · Wo^T.  Cast dp_new to
	//    BF16 (overwriting scratch_qbf — q_bf is no longer needed here)
	//    and use abt_bf16 for BF16 × BF16 × Wo^T.
	if (!cast_f32_to_bf16(dp_new, scratch_qbf, static_cast<size_t>(T) * m))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wo_bf, m, 0.0f, sdO, dModel))
		return false;

	// 4. dWo += sO^T · dp_new (FP32 × FP32 -> FP32; sO and dp_new are FP32).
	if (!sgemm_rowmajor_atb(dModel, m, T, 1.0f, sO, dModel, dp_new, m, 1.0f, dWo, m))
		return false;

	// 5. Attention backward: FP32 throughout; writes sdQ/sdK/sdV as FP32.
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdQ, 0, sizeof(float) * T * dModel, computeStream()));
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdK, 0, sizeof(float) * T * dModel, computeStream()));
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdV, 0, sizeof(float) * T * dModel, computeStream()));
	if (!flash_attention_backward_cublas_tiled(
	        sQ, sK, sV, sO, sdO,
	        T, nHeads, dHead, dModel, causal,
	        sdQ, sdK, sdV, scratch_P, scratch_dP))
		return false;

	// 6. dq += sdQ · Wq^T (and for sdK, sdV).  Use abt_bf16 — cast each sdX
	//    to BF16 one at a time into scratch_sdbf, then BF16 × BF16 -> FP32.
	if (!cast_f32_to_bf16(sdQ, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, m, dModel, 1.0f, scratch_sdbf, dModel, Wq_bf, dModel, 1.0f, dq, m))
		return false;
	if (!cast_f32_to_bf16(sdK, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, m, dModel, 1.0f, scratch_sdbf, dModel, Wk_bf, dModel, 1.0f, dq, m))
		return false;
	if (!cast_f32_to_bf16(sdV, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, m, dModel, 1.0f, scratch_sdbf, dModel, Wv_bf, dModel, 1.0f, dq, m))
		return false;

	// 7. Weight gradients: dWq += q^T · sdQ etc.  FP32 throughout.
	if (!sgemm_rowmajor_atb(m, dModel, T, 1.0f, q, m, sdQ, dModel, 1.0f, dWq, dModel))
		return false;
	if (!sgemm_rowmajor_atb(m, dModel, T, 1.0f, q, m, sdK, dModel, 1.0f, dWk, dModel))
		return false;
	if (!sgemm_rowmajor_atb(m, dModel, T, 1.0f, q, m, sdV, dModel, 1.0f, dWv, dModel))
		return false;
	return true;
}

// Tiled variant of chiron_attention_shear.  Same math as chiron_attention_shear
// but replaces the O(T²·dH) flash-attention core with flash_attention_cublas_tiled
// (TF32 tensor cores via cuBLAS batched strided GEMM).  Typical 5-10× wall-clock
// improvement at T≥512 on Ampere/Ada hardware, at the cost of an [nH, T, T]
// scratch buffer (caller-owned).
//
// Constraint: nHeads == nKVHeads (no GQA — the tiled kernel does not expand
// the KV head dimension).
bool chiron_attention_shear_tiled(const float* q, float* p,
                                    const float* Wq, const float* Wk,
                                    const float* Wv, const float* Wo,
                                    int T, int m, int nHeads, int dHead,
                                    bool causal, bool invert,
                                    float* scratch_Q, float* scratch_K,
                                    float* scratch_V, float* scratch_O,
                                    float* scratch_S)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel = nHeads * dHead;

	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wq, dModel, 0.0f, scratch_Q, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wk, dModel, 0.0f, scratch_K, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wv, dModel, 0.0f, scratch_V, dModel))
		return false;

	if (!flash_attention_cublas_tiled(scratch_Q, scratch_K, scratch_V,
	                                    T, nHeads, dHead, dModel, causal,
	                                    scratch_O, scratch_S))
		return false;

	const float sign = invert ? -1.0f : 1.0f;
	if (!sgemm_rowmajor(T, m, dModel, sign, scratch_O, dModel, Wo, m, 1.0f, p, m))
		return false;
	return true;
}

// Tiled variant of the shear backward.  Replaces flash_attention_multihead_backward
// with flash_attention_backward_cublas_tiled — TF32 tensor-core batched GEMMs for
// P = softmax(QK^T), dV += P^T dO, dP = dO V^T, dS = softmax_bwd(P, dP),
// dQ = dS K, dK += dS^T Q.  Extra scratch: scratch_P and scratch_dP, each [nH, T, T].
bool chiron_attention_shear_backward_tiled(
    const float* q, const float* dp_new,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    int T, int m, int nHeads, int dHead,
    bool causal,
    float* dq,
    float* dWq, float* dWk, float* dWv, float* dWo,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV,
    float* scratch_P, float* scratch_dP)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel = nHeads * dHead;

	// Recompute Q, K, V, O from q.
	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wq, dModel, 0.0f, sQ, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wk, dModel, 0.0f, sK, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wv, dModel, 0.0f, sV, dModel))
		return false;
	if (!flash_attention_cublas_tiled(sQ, sK, sV, T, nHeads, dHead, dModel, causal,
	                                    sO, scratch_P))
		return false;

	// dO = dp_new · Wo^T
	if (!sgemm_rowmajor_abt(T, dModel, m, 1.0f, dp_new, m, Wo, m, 0.0f, sdO, dModel))
		return false;
	// dWo += O^T · dp_new
	if (!sgemm_rowmajor_atb(dModel, m, T, 1.0f, sO, dModel, dp_new, m, 1.0f, dWo, m))
		return false;

	// Tiled attention backward: writes dQ, accumulates dK+=, dV+=.
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdQ, 0, sizeof(float) * T * dModel, computeStream()));
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdK, 0, sizeof(float) * T * dModel, computeStream()));
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdV, 0, sizeof(float) * T * dModel, computeStream()));
	if (!flash_attention_backward_cublas_tiled(
	        sQ, sK, sV, sO, sdO,
	        T, nHeads, dHead, dModel, causal,
	        sdQ, sdK, sdV, scratch_P, scratch_dP))
		return false;

	// dq += dQ · Wq^T + dK · Wk^T + dV · Wv^T
	if (!sgemm_rowmajor_abt(T, m, dModel, 1.0f, sdQ, dModel, Wq, dModel, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_abt(T, m, dModel, 1.0f, sdK, dModel, Wk, dModel, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_abt(T, m, dModel, 1.0f, sdV, dModel, Wv, dModel, 1.0f, dq, m))
		return false;

	// dWq, dWk, dWv += q^T · dQ, dK, dV
	if (!sgemm_rowmajor_atb(m, dModel, T, 1.0f, q, m, sdQ, dModel, 1.0f, dWq, dModel))
		return false;
	if (!sgemm_rowmajor_atb(m, dModel, T, 1.0f, q, m, sdK, dModel, 1.0f, dWk, dModel))
		return false;
	if (!sgemm_rowmajor_atb(m, dModel, T, 1.0f, q, m, sdV, dModel, 1.0f, dWv, dModel))
		return false;
	return true;
}

// ===========================================================================
//  5b. cuBLAS-tiled flash attention (Stage-1 tensor-core variant).
// ===========================================================================

bool flash_attention_cublas_tiled(const float* Q, const float* K, const float* V,
                                    int T, int nHeads, int dHead, int dModel,
                                    bool causal,
                                    float* O, float* scratch_S)
{
	if (T <= 0 || nHeads <= 0 || dHead <= 0) return true;

	const float invSqrtDH = 1.0f / sqrtf(static_cast<float>(dHead));

	// S[nH, T, T] = invSqrtDH · Q_h · K_h^T for each head h.
	// Q and K are laid out as [T, dModel] with heads packed along dModel.
	// For head h, the head slice is Q[:, h*dHead : (h+1)*dHead].
	// Batched strided: stride_between_batches = dHead (the slice starts
	// dHead elements later in the packed layout).
	if (!sgemm_batched_strided_abt(
	        T, T, dHead,
	        invSqrtDH,
	        Q, dModel, (long long)dHead,   // A = Q_h; lda=dModel, strideA=dHead
	        K, dModel, (long long)dHead,   // B = K_h; ldb=dModel, strideB=dHead
	        0.0f,
	        scratch_S, T, (long long)T * T,  // C = S_h; ldc=T, strideC=T*T
	        nHeads))
		return false;

	// Apply causal mask + row-softmax in place.
	// `causal_mask_softmax_inplace` expects [batch, T, T] — we treat
	// nHeads as batch.  (For non-causal we'd want a separate softmax
	// but CHIRON always trains causal; assume causal here.)
	if (causal)
	{
		if (!causal_mask_softmax_inplace(scratch_S, nHeads, T))
			return false;
	}
	else
	{
		// Fall back to plain rowwise softmax (still in-place).
		if (!softmax_forward(scratch_S, nHeads * T, T, scratch_S))
			return false;
	}

	// O[nH, T, dHead] = S · V  for each head.
	if (!sgemm_batched_strided(
	        T, dHead, T,
	        1.0f,
	        scratch_S, T, (long long)T * T,  // A = P_h; lda=T, strideA=T*T
	        V, dModel, (long long)dHead,     // B = V_h; ldb=dModel, strideB=dHead
	        0.0f,
	        O, dModel, (long long)dHead,     // C = O_h; ldc=dModel, strideC=dHead
	        nHeads))
		return false;

	return true;
}

// ===========================================================================
//  5b2. BF16 cuBLAS-tiled flash attention (tensor-core path).
// ===========================================================================
//
// Same algorithm as flash_attention_cublas_tiled but with Q/K/V cast to
// BF16 and GEMMs running on BF16 tensor cores (2x over TF32 on 4080 SUPER).
// Softmax runs on the FP32 scratch_S; P is cast to BF16 for the PV GEMM.

bool flash_attention_cublas_tiled_bf16(
    const float* Q, const float* K, const float* V,
    int T, int nHeads, int dHead, int dModel,
    bool causal,
    float* O,
    float* scratch_S,
    unsigned short* scratch_Qbf16, unsigned short* scratch_Kbf16,
    unsigned short* scratch_Vbf16, unsigned short* scratch_Pbf16)
{
	if (T <= 0 || nHeads <= 0 || dHead <= 0) return true;
	const float invSqrtDH = 1.0f / sqrtf(static_cast<float>(dHead));
	const size_t nPacked = static_cast<size_t>(T) * dModel;

	// Cast Q/K/V once.
	if (!cast_f32_to_bf16(Q, scratch_Qbf16, nPacked)) return false;
	if (!cast_f32_to_bf16(K, scratch_Kbf16, nPacked)) return false;
	if (!cast_f32_to_bf16(V, scratch_Vbf16, nPacked)) return false;

	// S = (1/sqrt(dH)) Q K^T via BF16 batched (_abt).
	if (!sgemm_batched_strided_abt_bf16(
	        T, T, dHead, invSqrtDH,
	        scratch_Qbf16, dModel, (long long)dHead,
	        scratch_Kbf16, dModel, (long long)dHead,
	        0.0f,
	        scratch_S, T, (long long)T * T,
	        nHeads))
		return false;

	// Softmax in FP32 (same as FP32 path).
	if (causal)
	{
		if (!causal_mask_softmax_inplace(scratch_S, nHeads, T)) return false;
	}
	else
	{
		if (!softmax_forward(scratch_S, nHeads * T, T, scratch_S)) return false;
	}

	// Cast P to BF16 for the PV GEMM.
	const size_t nScores = static_cast<size_t>(nHeads) * T * T;
	if (!cast_f32_to_bf16(scratch_S, scratch_Pbf16, nScores)) return false;

	// O = P · V via BF16 batched (plain NN).
	if (!sgemm_batched_strided_bf16(
	        T, dHead, T, 1.0f,
	        scratch_Pbf16, T, (long long)T * T,
	        scratch_Vbf16, dModel, (long long)dHead,
	        0.0f,
	        O, dModel, (long long)dHead,
	        nHeads))
		return false;

	return true;
}

// ===========================================================================
//  5c. cuBLAS-tiled flash attention — BACKWARD.
// ===========================================================================
//
// Math:
//   P   = softmax_row(causal((1/sqrt(dH)) Q K^T))   [recompute]
//   dV += P^T · dO
//   dP  = dO · V^T
//   dS  = softmax_backward_attn(P, dP)              [existing kernel]
//   dQ  = (1/sqrt(dH)) · dS · K
//   dK += (1/sqrt(dH)) · dS^T · Q

bool flash_attention_backward_cublas_tiled(
    const float* Q, const float* K, const float* V,
    const float* /*O unused*/, const float* dO,
    int T, int nHeads, int dHead, int dModel,
    bool causal,
    float* dQ, float* dK, float* dV,
    float* scratch_P, float* scratch_dP)
{
	if (T <= 0 || nHeads <= 0 || dHead <= 0) return true;
	const float invSqrtDH = 1.0f / sqrtf(static_cast<float>(dHead));

	// Recompute P.
	if (!sgemm_batched_strided_abt(
	        T, T, dHead, invSqrtDH,
	        Q, dModel, (long long)dHead,
	        K, dModel, (long long)dHead,
	        0.0f,
	        scratch_P, T, (long long)T * T,
	        nHeads))
		return false;
	if (causal)
	{
		if (!causal_mask_softmax_inplace(scratch_P, nHeads, T)) return false;
	}
	else
	{
		if (!softmax_forward(scratch_P, nHeads * T, T, scratch_P)) return false;
	}

	// dV += P^T · dO.
	if (!sgemm_batched_strided_atb(
	        T, dHead, T, 1.0f,
	        scratch_P, T, (long long)T * T,
	        dO, dModel, (long long)dHead,
	        1.0f,
	        dV, dModel, (long long)dHead,
	        nHeads))
		return false;

	// dP = dO · V^T.
	if (!sgemm_batched_strided_abt(
	        T, T, dHead, 1.0f,
	        dO, dModel, (long long)dHead,
	        V, dModel, (long long)dHead,
	        0.0f,
	        scratch_dP, T, (long long)T * T,
	        nHeads))
		return false;

	// dS = softmax_backward(P, dP) in place on scratch_dP.
	if (!softmax_backward_attn(scratch_P, scratch_dP, nHeads, T, 1.0f, scratch_dP))
		return false;

	// dQ = (1/sqrt(dH)) · dS · K.
	if (!sgemm_batched_strided(
	        T, dHead, T, invSqrtDH,
	        scratch_dP, T, (long long)T * T,
	        K, dModel, (long long)dHead,
	        0.0f,
	        dQ, dModel, (long long)dHead,
	        nHeads))
		return false;

	// dK += (1/sqrt(dH)) · dS^T · Q.
	if (!sgemm_batched_strided_atb(
	        T, dHead, T, invSqrtDH,
	        scratch_dP, T, (long long)T * T,
	        Q, dModel, (long long)dHead,
	        1.0f,
	        dK, dModel, (long long)dHead,
	        nHeads))
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

// Backward counterpart to chiron_attention_shear_bf16 (non-materialized
// flash attention).  Same orchestration as chiron_attention_shear_backward
// (FP32 projections, O recomputation, output-projection backward, attention
// backward, weight-grad accumulation) except the attention backward uses
// flash_attention_multihead_backward_bf16 — which recomputes softmax
// probabilities block-wise instead of reading a materialized probs tensor.
//
// Scratch: same as chiron_attention_shear_backward plus BF16 staging
// buffers for Q/K/V (same as the forward).  No scratch_P / scratch_dP
// (the whole point of flash attention).
//
// Intended for long-context (T >= 4096) training where the tiled variant's
// O(nH*T^2) scratch_P + scratch_dP exceeds the GPU's remaining VRAM.
bool chiron_attention_shear_backward_bf16(
    const float* q, const float* dp_new,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    int T, int m, int nHeads, int nKVHeads, int dHead,
    bool causal,
    float* dq,
    float* dWq, float* dWk, float* dWv, float* dWo,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV,
    uint16_t* scratch_Qbf, uint16_t* scratch_Kbf, uint16_t* scratch_Vbf)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel    = nHeads   * dHead;
	const int dModelKV  = nKVHeads * dHead;

	// Recompute forward intermediates.
	if (!sgemm_rowmajor(T, dModel,   m, 1.0f, q, m, Wq, dModel,   0.0f, sQ, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wk, dModelKV, 0.0f, sK, dModelKV))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wv, dModelKV, 0.0f, sV, dModelKV))
		return false;

	// Cast to BF16 for flash attention.
	if (!cast_f32_to_bf16(sQ, scratch_Qbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!cast_f32_to_bf16(sK, scratch_Kbf, static_cast<size_t>(T) * dModelKV))
		return false;
	if (!cast_f32_to_bf16(sV, scratch_Vbf, static_cast<size_t>(T) * dModelKV))
		return false;

	// Forward (recompute O in FP32) — needed by the tiled PV backward path
	// that flash_attention_multihead_backward_bf16 takes as input.
	if (!flash_attention_multihead_forward_bf16(scratch_Qbf, scratch_Kbf, scratch_Vbf,
	                                             T, nHeads, nKVHeads,
	                                             dHead, dModel, dModelKV,
	                                             causal, sO))
		return false;

	// Output-projection backward.
	if (!sgemm_rowmajor_abt(T, dModel, m, 1.0f, dp_new, m, Wo, m, 0.0f, sdO, dModel))
		return false;
	if (!sgemm_rowmajor_atb(dModel, m, T, 1.0f, sO, dModel, dp_new, m, 1.0f, dWo, m))
		return false;

	// Attention backward (flash, BF16 inputs).
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdQ, 0, sizeof(float) * T * dModel,   computeStream()));
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdK, 0, sizeof(float) * T * dModelKV, computeStream()));
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdV, 0, sizeof(float) * T * dModelKV, computeStream()));
	if (!flash_attention_multihead_backward_bf16(scratch_Qbf, scratch_Kbf, scratch_Vbf,
	                                              sO, sdO,
	                                              T, nHeads, nKVHeads,
	                                              dHead, dModel, dModelKV,
	                                              causal, sdQ, sdK, sdV))
		return false;

	// Project dQ/dK/dV back into q space.
	if (!sgemm_rowmajor_abt(T, m, dModel, 1.0f, sdQ, dModel, Wq, dModel, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_abt(T, m, dModelKV, 1.0f, sdK, dModelKV, Wk, dModelKV, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_abt(T, m, dModelKV, 1.0f, sdV, dModelKV, Wv, dModelKV, 1.0f, dq, m))
		return false;

	// Weight gradients.
	if (!sgemm_rowmajor_atb(m, dModel, T, 1.0f, q, m, sdQ, dModel, 1.0f, dWq, dModel))
		return false;
	if (!sgemm_rowmajor_atb(m, dModelKV, T, 1.0f, q, m, sdK, dModelKV, 1.0f, dWk, dModelKV))
		return false;
	if (!sgemm_rowmajor_atb(m, dModelKV, T, 1.0f, q, m, sdV, dModelKV, 1.0f, dWv, dModelKV))
		return false;
	return true;
}

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
