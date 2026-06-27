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
#include "gpu_blas_fp8.h"
#include "gpu_kernels.h"
#include "gpu_buffer.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cmath>
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
//  1b. SCFA stream-op fused kernels (ralph-loop iter 5, 2026-05-14).
// ===========================================================================
//
// The SCFA forward/backward chain in glades-trainer/trainer/chiron_main.cpp
// issues 6-7 axpy/memcpy operations per layer per direction over the FP32
// residual stream buffers (T·m floats = 67 MB at T=8192, m=2048).  Each
// memcpy_d2d that's immediately followed by an axpy can be folded into a
// single fused kernel that halves the memory traffic for that step.  The
// math is bit-identical (these are element-wise FP32 ops, no precision
// loss).  Behavior is gated behind --scfa-fuse-streams in the trainer.

namespace {

__global__ void chiron_scfa_sub_kernel(float* __restrict__ c,
                                       const float* __restrict__ a,
                                       const float* __restrict__ b, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = a[i] - b[i];
}

__global__ void chiron_scfa_axpy2_kernel(float* __restrict__ p,
                                         float alpha,
                                         const float* __restrict__ a,
                                         const float* __restrict__ b, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) p[i] += alpha * (a[i] + b[i]);
}

__global__ void chiron_scfa_scaled_copy_kernel(float* __restrict__ c,
                                               float alpha,
                                               const float* __restrict__ a, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = alpha * a[i];
}

// iter 63 (Arc 2, BF16 residual-p — 2026-05-16): BF16-p storage variants.
// Per PARADIGM_BF16_RESIDUAL_P_DESIGN.md §4.2:
//   p̂ := round_RN( bf16_to_fp32(p̂) + α·(a + b) )           (axpy2)
//   p̂ := round_RN( bf16_to_fp32(p̂) + α·x )                  (axpy)
//   c  := round_RN( α·a )                                    (scaled-copy)
// All accumulation is FP32-internal; only the final write rounds back to BF16.
// RN-even (deterministic) for iter 63; stochastic-rounding (SR) variant in iter 64.

__device__ __forceinline__ unsigned short fp32_to_bf16_rn_dev(float x)
{
	union { float f; unsigned int u; } v;
	v.f = x;
	if (isnan(x)) {
		// Quiet NaN; preserve sign.
		return (unsigned short)(((v.u & 0x80000000u) | 0x7FC00000u) >> 16);
	}
	// Round-to-nearest, ties-to-even.
	const unsigned int lsb = (v.u >> 16) & 1u;
	const unsigned int bias = 0x7FFFu + lsb;
	return (unsigned short)((v.u + bias) >> 16);
}

__device__ __forceinline__ float bf16_to_fp32_dev(unsigned short b)
{
	union { unsigned int u; float f; } v;
	v.u = ((unsigned int)b) << 16;
	return v.f;
}

__global__ void chiron_scfa_axpy2_bf16p_rn_kernel(unsigned short* __restrict__ p_bf,
                                                   float alpha,
                                                   const float* __restrict__ a,
                                                   const float* __restrict__ b, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float acc = bf16_to_fp32_dev(p_bf[i]) + alpha * (a[i] + b[i]);
	p_bf[i] = fp32_to_bf16_rn_dev(acc);
}

__global__ void chiron_axpy_bf16p_rn_kernel(unsigned short* __restrict__ p_bf,
                                             float alpha,
                                             const float* __restrict__ x, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float acc = bf16_to_fp32_dev(p_bf[i]) + alpha * x[i];
	p_bf[i] = fp32_to_bf16_rn_dev(acc);
}

__global__ void chiron_scfa_scaled_copy_bf16p_rn_kernel(unsigned short* __restrict__ c_bf,
                                                         float alpha,
                                                         const float* __restrict__ a, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	c_bf[i] = fp32_to_bf16_rn_dev(alpha * a[i]);
}

// Read BF16 p, add to FP32 q: q[i] += alpha * bf16_to_fp32(p_bf[i]).
// Used at the LN+axpy step (q += p) when --bf16-residual-p routes p to BF16
// storage but q stays FP32.
__global__ void chiron_bf16_to_fp32_axpy_kernel(float* __restrict__ q,
                                                 float alpha,
                                                 const unsigned short* __restrict__ p_bf,
                                                 int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	q[i] += alpha * bf16_to_fp32_dev(p_bf[i]);
}

// iter 64 (Arc 2): stochastic-rounding variants.  Same FP32-internal accum
// as RN, but final BF16 encode uses xorshift-mixed hash of (idx, step, seed)
// to decide round-up vs round-down — making per-element error mean-zero by
// construction.  Drift over L=12 reversible writes becomes random walk
// O(sqrt(L) * ULP_BF16) instead of biased O(L * ULP_BF16).  Same RNG infra
// as iter 49's cast_f32_to_bf16_stochastic.
__device__ __forceinline__ uint32_t sr_hash32_dev(uint32_t a, uint32_t b, uint32_t c)
{
	uint32_t x = a ^ (b * 0x9E3779B1u) ^ (c * 0x85EBCA6Bu);
	x ^= x >> 16; x *= 0x7FEB352Du;
	x ^= x >> 15; x *= 0x846CA68Bu;
	x ^= x >> 16;
	return x;
}

__device__ __forceinline__ unsigned short fp32_to_bf16_sr_dev(float x,
                                                               uint32_t idx,
                                                               uint32_t stepIdx,
                                                               uint32_t baseSeed)
{
	union { float f; uint32_t u; } v;
	v.f = x;
	if (isnan(x)) {
		return (unsigned short)(((v.u & 0x80000000u) | 0x7FC00000u) >> 16);
	}
	const uint32_t low16 = v.u & 0xFFFFu;
	const uint32_t rnd = sr_hash32_dev(idx, stepIdx, baseSeed) & 0xFFFFu;
	uint32_t high16 = v.u >> 16;
	if (rnd < low16) high16 += 1u;
	return (unsigned short)(high16 & 0xFFFFu);
}

__global__ void chiron_scfa_axpy2_bf16p_sr_kernel(unsigned short* __restrict__ p_bf,
                                                   float alpha,
                                                   const float* __restrict__ a,
                                                   const float* __restrict__ b,
                                                   int n,
                                                   uint32_t srBaseSeed,
                                                   uint32_t srStepIdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float acc = bf16_to_fp32_dev(p_bf[i]) + alpha * (a[i] + b[i]);
	p_bf[i] = fp32_to_bf16_sr_dev(acc, (uint32_t)i, srStepIdx, srBaseSeed);
}

// iter 70 (2026-05-19): Fused dual-output axpy2 for the iter 65 BF16-residual-p
// mirror.  Computes new FP32 p = p_fp32 + alpha*(a+b) in an FP32 register and
// writes BOTH the FP32 canonical (s.p) AND a BF16 SR-rounded mirror (s.p_bf16)
// in a single pass.  Replaces the two-kernel sequence at chiron_main.cpp:6439+
// (chiron_scfa_axpy2 then cast_f32_to_bf16_stochastic) used at every SCFA shear
// commit under --bf16-residual-p.  SR hash matches k_cast_f32_to_bf16_stochastic
// for bit-identical NLL when (srBaseSeed, srStepIdx) is preserved across calls.
// Saves: 1 launch + 1 HBM read of p (Tm × 4 B) per layer per direction.
__global__ void chiron_scfa_axpy2_dual_p_kernel(float* __restrict__ p_fp32,
                                                 unsigned short* __restrict__ p_bf16,
                                                 float alpha,
                                                 const float* __restrict__ a,
                                                 const float* __restrict__ b,
                                                 int n,
                                                 uint32_t srBaseSeed,
                                                 uint32_t srStepIdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	// Match chiron_scfa_axpy2_kernel's exact arithmetic form (p[i] += alpha *
	// (a[i] + b[i])) so NVCC contracts to the same FMA emit — otherwise the
	// register-pressure delta from the added BF16-SR computation can shift
	// FMA decisions, producing ULP-scale FP32 drift (iter 50 pattern).
	float p_val = p_fp32[i];
	p_val += alpha * (a[i] + b[i]);
	p_fp32[i] = p_val;
	p_bf16[i] = fp32_to_bf16_sr_dev(p_val, (uint32_t)i, srStepIdx, srBaseSeed);
}

__global__ void chiron_axpy_bf16p_sr_kernel(unsigned short* __restrict__ p_bf,
                                             float alpha,
                                             const float* __restrict__ x,
                                             int n,
                                             uint32_t srBaseSeed,
                                             uint32_t srStepIdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float acc = bf16_to_fp32_dev(p_bf[i]) + alpha * x[i];
	p_bf[i] = fp32_to_bf16_sr_dev(acc, (uint32_t)i, srStepIdx, srBaseSeed);
}

__global__ void chiron_scfa_scaled_copy_bf16p_sr_kernel(unsigned short* __restrict__ c_bf,
                                                         float alpha,
                                                         const float* __restrict__ a,
                                                         int n,
                                                         uint32_t srBaseSeed,
                                                         uint32_t srStepIdx)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	c_bf[i] = fp32_to_bf16_sr_dev(alpha * a[i], (uint32_t)i, srStepIdx, srBaseSeed);
}

// Reln forward: reads BF16 p row by row, computes FP32-internal mean / var /
// normalize, writes FP32 q_out.  Stats stay FP32.  Identical math to
// chiron_reln_forward_rows but with BF16-decoded reads.
__global__ void chiron_reln_forward_rows_bf16p_kernel(
    const unsigned short* __restrict__ p_bf_in,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float eps, int cols,
    float* __restrict__ q_out,
    float* __restrict__ stats)
{
	int row = blockIdx.x;
	const unsigned short* xRow = p_bf_in + (size_t)row * cols;
	float*       oRow = q_out + (size_t)row * cols;

	extern __shared__ float smem[];
	float* sSumA = smem;
	float* sSumB = smem + (blockDim.x / 32 + 1);

	__shared__ float sMean, sSigma;

	// Pass 1: mean (decode BF16 inline).
	float s = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) s += bf16_to_fp32_dev(xRow[i]);
	// reuse the blockReduceSum from gpu_chiron.cu by manual reduction:
	// SMEM tree-reduce over blockDim.x threads → warp tail.
	{
		// Single-block warp reduce; simpler form since the existing
		// blockReduceSum lives in anon namespace of this file.
		__shared__ float sShared[33];
		const int lane = threadIdx.x & 31;
		const int warpId = threadIdx.x >> 5;
		for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFFu, s, o);
		if (lane == 0) sShared[warpId] = s;
		__syncthreads();
		if (warpId == 0) {
			s = (threadIdx.x < (blockDim.x + 31) / 32) ? sShared[lane] : 0.0f;
			for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFFu, s, o);
		}
		if (threadIdx.x == 0) sMean = s / (float)cols;
	}
	__syncthreads();
	const float mu = sMean;
	(void)sSumA;

	// Pass 2: variance.
	float v = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float d = bf16_to_fp32_dev(xRow[i]) - mu;
		v += d * d;
	}
	{
		__shared__ float vShared[33];
		const int lane = threadIdx.x & 31;
		const int warpId = threadIdx.x >> 5;
		for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
		if (lane == 0) vShared[warpId] = v;
		__syncthreads();
		if (warpId == 0) {
			v = (threadIdx.x < (blockDim.x + 31) / 32) ? vShared[lane] : 0.0f;
			for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
		}
		if (threadIdx.x == 0) {
			float var = v / (float)cols + eps;
			sSigma = sqrtf(var);
		}
	}
	__syncthreads();
	const float sigma = sSigma;
	const float inv_sigma = 1.0f / sigma;
	(void)sSumB;

	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		oRow[i] = gamma[i] * (bf16_to_fp32_dev(xRow[i]) - mu) * inv_sigma + beta[i];

	if (threadIdx.x == 0) {
		stats[(size_t)row * 2 + 0] = mu;
		stats[(size_t)row * 2 + 1] = sigma;
	}
}

// Reln inverse: reads FP32 q_out + stats, undoes affine + LN, SR-writes BF16 p.
// p = ((q_out - beta) / gamma) * sigma + mu, then BF16 SR encode.
__global__ void chiron_reln_inverse_rows_bf16p_sr_kernel(
    const float* __restrict__ q_out,
    const float* __restrict__ stats,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int cols,
    unsigned short* __restrict__ p_bf_out,
    uint32_t srBaseSeed,
    uint32_t srStepIdx)
{
	int row = blockIdx.x;
	const float* oRow = q_out + (size_t)row * cols;
	unsigned short* pRow = p_bf_out + (size_t)row * cols;
	const float mu    = stats[(size_t)row * 2 + 0];
	const float sigma = stats[(size_t)row * 2 + 1];

	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		const float y = (oRow[i] - beta[i]) / gamma[i];
		const float v = fmaf(sigma, y, mu);
		pRow[i] = fp32_to_bf16_sr_dev(v, (uint32_t)((size_t)row * cols + i),
		                              srStepIdx, srBaseSeed);
	}
}

} // anonymous namespace

bool chiron_scfa_sub(float* c, const float* a, const float* b, int n,
                     cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_scfa_sub_kernel<<<grid, kBlockElem, 0, s>>>(c, a, b, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_scfa_axpy2(float* p, float alpha,
                       const float* a, const float* b, int n,
                       cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_scfa_axpy2_kernel<<<grid, kBlockElem, 0, s>>>(p, alpha, a, b, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// iter 63 (Arc 2): BF16-p host wrappers.
bool chiron_scfa_axpy2_bf16p_rn(unsigned short* p_bf, float alpha,
                                 const float* a, const float* b, int n,
                                 cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_scfa_axpy2_bf16p_rn_kernel<<<grid, kBlockElem, 0, s>>>(p_bf, alpha, a, b, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_axpy_bf16p_rn(unsigned short* p_bf, float alpha,
                           const float* x, int n,
                           cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_axpy_bf16p_rn_kernel<<<grid, kBlockElem, 0, s>>>(p_bf, alpha, x, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_scfa_scaled_copy_bf16p_rn(unsigned short* c_bf, float alpha,
                                       const float* a, int n,
                                       cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_scfa_scaled_copy_bf16p_rn_kernel<<<grid, kBlockElem, 0, s>>>(c_bf, alpha, a, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_bf16_to_fp32_axpy(float* q, float alpha,
                               const unsigned short* p_bf, int n,
                               cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_bf16_to_fp32_axpy_kernel<<<grid, kBlockElem, 0, s>>>(q, alpha, p_bf, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// iter 64: SR variants of the BF16-p kernels.  Same FP32 accum semantics as
// the RN variants but with mean-zero rounding.  Caller supplies a per-tensor
// seed + step counter (typically W.bf16WeightsSeed + step) so per-(model,
// step, element) randomness is reproducible across runs with the same seed.
bool chiron_scfa_axpy2_bf16p_sr(unsigned short* p_bf, float alpha,
                                 const float* a, const float* b, int n,
                                 unsigned int srBaseSeed,
                                 unsigned int srStepIdx,
                                 cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_scfa_axpy2_bf16p_sr_kernel<<<grid, kBlockElem, 0, s>>>(
	    p_bf, alpha, a, b, n, srBaseSeed, srStepIdx);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_axpy_bf16p_sr(unsigned short* p_bf, float alpha,
                           const float* x, int n,
                           unsigned int srBaseSeed,
                           unsigned int srStepIdx,
                           cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_axpy_bf16p_sr_kernel<<<grid, kBlockElem, 0, s>>>(
	    p_bf, alpha, x, n, srBaseSeed, srStepIdx);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// iter 70 host wrapper for the fused dual-output axpy2 kernel.
bool chiron_scfa_axpy2_dual_p(float* p_fp32, unsigned short* p_bf16,
                               float alpha,
                               const float* a, const float* b, int n,
                               unsigned int srBaseSeed,
                               unsigned int srStepIdx,
                               cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_scfa_axpy2_dual_p_kernel<<<grid, kBlockElem, 0, s>>>(
	    p_fp32, p_bf16, alpha, a, b, n, srBaseSeed, srStepIdx);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_scfa_scaled_copy_bf16p_sr(unsigned short* c_bf, float alpha,
                                       const float* a, int n,
                                       unsigned int srBaseSeed,
                                       unsigned int srStepIdx,
                                       cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_scfa_scaled_copy_bf16p_sr_kernel<<<grid, kBlockElem, 0, s>>>(
	    c_bf, alpha, a, n, srBaseSeed, srStepIdx);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_reln_forward_rows_bf16p(const unsigned short* p_bf_in,
                                     float* q_out, float* stats,
                                     const float* gamma, const float* beta,
                                     int T, int m, float eps)
{
	if (T <= 0 || m <= 0) return true;
	const int block = rowBlockSize(m);
	const int smemBytes = (block / 32 + 2) * 2 * sizeof(float);
	chiron_reln_forward_rows_bf16p_kernel<<<T, block, smemBytes, computeStream()>>>(
	    p_bf_in, gamma, beta, eps, m, q_out, stats);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_reln_inverse_rows_bf16p_sr(const float* q_out, const float* stats,
                                        const float* gamma, const float* beta,
                                        int T, int m,
                                        unsigned short* p_bf_out,
                                        unsigned int srBaseSeed,
                                        unsigned int srStepIdx)
{
	if (T <= 0 || m <= 0) return true;
	const int block = rowBlockSize(m);
	chiron_reln_inverse_rows_bf16p_sr_kernel<<<T, block, 0, computeStream()>>>(
	    q_out, stats, gamma, beta, m, p_bf_out, srBaseSeed, srStepIdx);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_scfa_scaled_copy(float* c, float alpha, const float* a, int n,
                              cudaStream_t stream)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	chiron_scfa_scaled_copy_kernel<<<grid, kBlockElem, 0, s>>>(c, alpha, a, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  2. Reversible LayerNorm (ReLN) forward.
// ===========================================================================
//
// One block per token.  Each block computes the row mean and variance via
// warp/block reductions, writes the normalized row, and records
// (mu, sigma) into stats[row, 0..1].

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

	// Stats: { mu, sigma } for this row.
	if (threadIdx.x == 0) {
		stats[(size_t)row * 2 + 0] = mu;
		stats[(size_t)row * 2 + 1] = sigma;
	}
}

// Port A (cast-elimination arc, 2026-06-12): reln forward with a BF16 mirror
// side-write.  Identical math to chiron_reln_forward_rows; pass 3 also
// writes the RNE-rounded BF16 encoding of each output element, bit-identical
// to running k_cast_f32_to_bf16 on q_out afterwards.  Lets the downstream
// FAST_16BF outer GEMM consume the mirror via the fast16bf constant table
// instead of launching a standalone T×m cast (iter 97/99/101 side-write
// mechanism class).  See research/CAST_CENSUS_2026_06_12.md.
__global__ void chiron_reln_forward_rows_dual(const float* __restrict__ q_in,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              float eps, int cols,
                                              float* __restrict__ q_out,
                                              unsigned short* __restrict__ q_out_bf16,
                                              float* __restrict__ stats)
{
	int row = blockIdx.x;
	const float* xRow = q_in + (size_t)row * cols;
	float*       oRow = q_out + (size_t)row * cols;
	unsigned short* bRow = q_out_bf16 + (size_t)row * cols;

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

	// Pass 3: normalize + affine, with BF16 RNE side-write (same encoding as
	// k_cast_f32_to_bf16: round-to-nearest-even via lsb bias; NaN flushed to
	// sign-preserving quiet NaN).
	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		const float o = gamma[i] * (xRow[i] - mu) * inv_sigma + beta[i];
		oRow[i] = o;
		union { float f; uint32_t u; } enc;
		enc.f = o;
		if (isnan(o)) {
			const uint32_t sign = enc.u & 0x80000000u;
			bRow[i] = (unsigned short)(((sign | 0x7FC00000u) >> 16) & 0xFFFFu);
		} else {
			const uint32_t lsb = (enc.u >> 16) & 1u;
			bRow[i] = (unsigned short)((enc.u + 0x7FFFu + lsb) >> 16);
		}
	}

	// Stats: { mu, sigma } for this row.
	if (threadIdx.x == 0) {
		stats[(size_t)row * 2 + 0] = mu;
		stats[(size_t)row * 2 + 1] = sigma;
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
	// iter 9 (2026-05-14): in-place safe — kernel reads xRow then writes oRow
	// per-element within a single thread iteration; if q_in == q_out the read
	// precedes the write per address.  __syncthreads between passes ensures
	// reductions complete before the normalize pass starts.  The 3-pass row
	// pattern (mean → variance → normalize+affine) is bit-identical whether
	// q_in == q_out or not.
	chiron_reln_forward_rows<<<T, block, smemBytes, computeStream()>>>(
		q_in, gamma, beta, eps, m, q_out, stats);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_reln_forward_dual(const float* q_in, float* q_out,
                              unsigned short* q_out_bf16, float* stats,
                              const float* gamma, const float* beta,
                              int T, int m, float eps)
{
	if (T <= 0 || m <= 0) return true;
	if (!q_out_bf16)
		return chiron_reln_forward(q_in, q_out, stats, gamma, beta, T, m, eps);
	int block = rowBlockSize(m);
	int smemBytes = (block / 32 + 2) * 2 * sizeof(float);
	// In-place safe for q_in == q_out by the same per-element read-then-write
	// argument as chiron_reln_forward (iter 9 note above).
	chiron_reln_forward_rows_dual<<<T, block, smemBytes, computeStream()>>>(
		q_in, gamma, beta, eps, m, q_out, q_out_bf16, stats);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  2b. Fused reln-forward + axpy-into-q (ralph-loop iter 9, 2026-05-14).
// ===========================================================================
//
// Replaces the 2-call pattern in CHIRON's per-layer-fuse path:
//     chiron_reln_forward(p, p_norm, stats, gamma_p, beta_p, T, m, eps);
//     axpy(alpha, p_norm, q, T*m);  // q += alpha · p_norm
// with a single kernel that computes normalized p AND accumulates it into q
// in one pass.  Eliminates the p_norm round-trip (1 write + 1 read of a
// T·m FP32 buffer = ~128 MB per call at T=8192 m=2048).
//
// Math is bit-identical FP32 modulo associativity of the inner FMA
// (alpha·(γ·(p-μ)/σ + β) is computed as one expression per element, so
// the rounding may differ by 1 ULP from the two-call form — sub-ULP at
// FP32 mantissa).
//
// Caller responsibility: alpha can be positive (forward q += α·reln(p))
// or negative (backward step 2: q -= α·reln(p) ≡ q += (-α)·reln(p)).
// stats[T, 2] is written exactly as chiron_reln_forward writes them.

namespace {

__global__ void chiron_reln_axpy_into_q_rows(const float* __restrict__ p,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              float alpha, float eps, int cols,
                                              float* __restrict__ q,
                                              float* __restrict__ stats)
{
	int row = blockIdx.x;
	const float* xRow = p + (size_t)row * cols;
	float*       qRow = q + (size_t)row * cols;

	extern __shared__ float smem[];
	float* sSumA = smem;
	float* sSumB = smem + (blockDim.x / 32 + 1);

	__shared__ float sMean, sSigma;

	// Pass 1: mean of p.
	float s = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) s += xRow[i];
	s = blockReduceSum(s, sSumA);
	if (threadIdx.x == 0) sMean = s / (float)cols;
	__syncthreads();

	const float mu = sMean;

	// Pass 2: variance of p.
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

	// Pass 3: q[i] += alpha · (gamma[i] · (p[i] - mu) / sigma + beta[i]).
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
	{
		float p_norm_i = gamma[i] * (xRow[i] - mu) * inv_sigma + beta[i];
		qRow[i] += alpha * p_norm_i;
	}

	// Stats: { mu, sigma } for this row (same format as chiron_reln_forward).
	if (threadIdx.x == 0) {
		stats[(size_t)row * 2 + 0] = mu;
		stats[(size_t)row * 2 + 1] = sigma;
	}
}

} // anonymous namespace

bool chiron_reln_axpy_into_q(const float* p, float* q, float* stats,
                              const float* gamma, const float* beta,
                              float alpha, int T, int m, float eps)
{
	if (T <= 0 || m <= 0) return true;
	int block = rowBlockSize(m);
	int smemBytes = (block / 32 + 2) * 2 * sizeof(float);
	chiron_reln_axpy_into_q_rows<<<T, block, smemBytes, computeStream()>>>(
		p, gamma, beta, alpha, eps, m, q, stats);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  2c. OBSD per-layer drift (richer symplectic block — Task 2).
// ===========================================================================
//
// Forward (sign=+1): q[i] += scale·a[i]·tanh(gamma[i]·x̂ + beta[i]),
//   x̂ = (p[i] − μ) / σ, with μ,σ the per-row mean/std of p (parameter-free
//   normalize, μ,σ over the m channels of the row).  Inverse (sign=−1)
//   subtracts the same term.  The drift never modifies p, so the inverse
//   recomputes x̂ from p exactly and reconstructs q.  No stats are emitted
//   (μ,σ are re-derived from p in both directions and in the backward).

namespace {

__global__ void chiron_drift_into_q_rows(const float* __restrict__ p,
                                         const float* __restrict__ a,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float sign, float scale, float eps, int cols,
                                         float* __restrict__ q)
{
	int row = blockIdx.x;
	const float* xRow = p + (size_t)row * cols;
	float*       qRow = q + (size_t)row * cols;
	extern __shared__ float smem[];
	float* sSumA = smem;
	float* sSumB = smem + (blockDim.x / 32 + 1);
	__shared__ float sMean, sSigma;

	float s = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) s += xRow[i];
	s = blockReduceSum(s, sSumA);
	if (threadIdx.x == 0) sMean = s / (float)cols;
	__syncthreads();
	const float mu = sMean;

	float v = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) { float d = xRow[i]-mu; v += d*d; }
	v = blockReduceSum(v, sSumB);
	if (threadIdx.x == 0) { float var = v/(float)cols + eps; sSigma = sqrtf(var); }
	__syncthreads();
	const float inv_sigma = 1.0f / sSigma;

	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float xhat = (xRow[i] - mu) * inv_sigma;
		float u = gamma[i] * xhat + beta[i];
		qRow[i] += sign * scale * a[i] * tanhf(u);
	}
}

} // anonymous namespace

bool chiron_drift_into_q(const float* p, float* q, const float* a,
                         const float* gamma, const float* beta,
                         float sign, float scale, int T, int m, float eps)
{
	if (T <= 0 || m <= 0) return true;
	int block = rowBlockSize(m);
	int smemBytes = (block / 32 + 2) * 2 * sizeof(float);
	chiron_drift_into_q_rows<<<T, block, smemBytes, computeStream()>>>(
	    p, a, gamma, beta, sign, scale, eps, m, q);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  3. Reversible LayerNorm (ReLN) inverse.
// ===========================================================================
//
// Given q_out, stats[row, 0..1] = { mu, sigma }, recover q_in.
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
	const float sigma = stats[(size_t)row * 2 + 1];

	for (int i = threadIdx.x; i < cols; i += blockDim.x)
	{
		// x = fma(sigma, (y - beta) / gamma, mu)
		const float y = (yRow[i] - beta[i]) / gamma[i];
		xRow[i] = fmaf(sigma, y, mu);
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
// novelty is where (mu, sigma) are stored. For the backward pass
// we convert the external (mu, sigma) stats buffer into the
// (mean[T], invStd[T]) format that the existing layernorm_backward
// kernel expects, then defer to that kernel.

namespace {

// Kernel to split [T, 2] (mu, sigma) -> two separate [T] buffers
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
		invStd[i] = 1.0f / stats[(size_t)i * 2 + 1];
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

// Phase 3 (q-side source cure, 2026-06-17): bounded ReLN backward — clamps the
// normalized xhat to [-xhatMax, xhatMax] in the dgamma/dbeta reduction, so the
// BF16-inverse reconstruction drift that inflates xhat (and overflows dgamma)
// is bounded at its source.  xhatMax<=0 = plain chiron_reln_backward.
bool chiron_reln_backward_bounded(const float* dq_out, const float* q_in,
                                   const float* gamma, const float* stats,
                                   int T, int m,
                                   float* dq_in, float* dgamma, float* dbeta,
                                   float* scratch_stats_split, float xhatMax)
{
	if (T <= 0 || m <= 0) return true;
	float* d_mean  = scratch_stats_split;
	float* d_invStd = scratch_stats_split + T;
	const int grid = (T + kBlockElem - 1) / kBlockElem;
	chiron_stats_split_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
	    stats, T, d_mean, d_invStd);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return layernorm_backward_bounded(dq_out, q_in, gamma, d_mean, d_invStd,
	                                   T, m, dq_in, dgamma, dbeta, xhatMax);
}

// ===========================================================================
//  3c. ReLN reverse-consistency backward (q-side instability cure, 2026-06-23).
// ===========================================================================
//
// Root cause (grad-trigger evidence, seed 2024 step 24070): the backward
// normalizes the recomputed activation q_in with the SAVED forward stats,
// which have drifted, inflating xhat ~13x BEFORE the sum-over-T forms dgamma
// and overflows it.  This backward re-derives (mean, invStd) from q_in itself
// (mirroring chiron_reln_forward_rows' two-pass reduction), so the xhat that
// layernorm_backward forms is unit-RMS by construction.  On a healthy step
// (recompute == forward) the re-derived stats equal the saved stats up to
// fp reduction order -> near-identity.  See
// docs/superpowers/specs/2026-06-23-reln-reverse-consistency-design.md.

namespace {

// One block per row: recompute mean and invStd from q_in over the m columns.
// Writes mean[T] into split[0..T) and invStd[T] into split[T..2T), matching the
// (mean, invStd) layout chiron_reln_backward feeds to layernorm_backward.
__global__ void chiron_reln_reanchor_stats_kernel(const float* __restrict__ q_in,
                                                  int cols, float eps,
                                                  float* __restrict__ mean,
                                                  float* __restrict__ invStd)
{
	int row = blockIdx.x;
	const float* xRow = q_in + (size_t)row * cols;

	extern __shared__ float smem[];
	float* sSumA = smem;
	float* sSumB = smem + (blockDim.x / 32 + 1);
	__shared__ float sMean, sInvStd;

	// Pass 1: mean.
	float s = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) s += xRow[i];
	s = blockReduceSum(s, sSumA);
	if (threadIdx.x == 0) sMean = s / (float)cols;
	__syncthreads();
	const float mu = sMean;

	// Pass 2: variance -> invStd.
	float v = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float d = xRow[i] - mu;
		v += d * d;
	}
	v = blockReduceSum(v, sSumB);
	if (threadIdx.x == 0) {
		float var = v / (float)cols + eps;
		sInvStd = 1.0f / sqrtf(var);
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		mean[row]   = mu;
		invStd[row] = sInvStd;
	}
}

} // anonymous namespace

// Re-anchored ReLN backward: identical interface to chiron_reln_backward, but
// derives (mean, invStd) from q_in rather than the saved stats.  `eps` must
// match the forward's eps_reln so healthy steps reproduce the saved stats.
bool chiron_reln_backward_reanchor(const float* dq_out, const float* q_in,
                                    const float* gamma,
                                    int T, int m, float eps,
                                    float* dq_in, float* dgamma, float* dbeta,
                                    float* scratch_stats_split)
{
	if (T <= 0 || m <= 0) return true;
	float* d_mean   = scratch_stats_split;
	float* d_invStd = scratch_stats_split + T;
	int block = rowBlockSize(m);
	size_t smemBytes = 2u * (size_t)(block / 32 + 1) * sizeof(float);
	chiron_reln_reanchor_stats_kernel<<<T, block, smemBytes, computeStream()>>>(
	    q_in, m, eps, d_mean, d_invStd);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return layernorm_backward(dq_out, q_in, gamma, d_mean, d_invStd,
	                          T, m, dq_in, dgamma, dbeta);
}

// ===========================================================================
//  3d. OBSD per-layer drift backward.
// ===========================================================================
//
// Chain: u = gamma·x̂ + beta ; s = tanh(u) ; q_out = q_in + scale·a·s, with
// x̂ = (p−μ)/σ and μ,σ per row of p (reanchored — recomputed from p).  Given
// dq_out this accumulates:
//   da     += Σ_t scale·s·dq_out
//   du      = scale·a·(1−s²)·dq_out ; dgamma += Σ_t du·x̂ ; dbeta += Σ_t du
//   dp      = parameter-free-normalize-backward of g = du⊙gamma
// The dp/dgamma/dbeta computation is delegated to chiron_reln_backward_reanchor
// called with dq_out:=du, gamma:=gamma_p — it forms g=du⊙gamma internally and
// also returns dgamma=Σ du·x̂ and dbeta=Σ du.  This pre-backward kernel only
// materializes du[T,m] and sdq[T,m]=scale·s·dq_out; da is the column-sum of sdq.

namespace {

// Per row: recompute μ,σ,x̂,u from p; write du and sdq.  Mirrors the two-pass
// reduction of chiron_drift_into_q_rows / chiron_reln_reanchor_stats_kernel.
__global__ void chiron_drift_pre_backward_rows(const float* __restrict__ p,
                                               const float* __restrict__ dq,
                                               const float* __restrict__ a,
                                               const float* __restrict__ gamma,
                                               const float* __restrict__ beta,
                                               float scale, float eps, int cols,
                                               float* __restrict__ du,
                                               float* __restrict__ sdq)
{
	int row = blockIdx.x;
	const float* pr = p + (size_t)row*cols; const float* dr = dq + (size_t)row*cols;
	float* duR = du + (size_t)row*cols; float* sdqR = sdq + (size_t)row*cols;
	extern __shared__ float smem[];
	float* sA = smem; float* sB = smem + (blockDim.x/32 + 1);
	__shared__ float sMean, sSigma;
	float s=0.f; for (int i=threadIdx.x;i<cols;i+=blockDim.x) s+=pr[i];
	s=blockReduceSum(s,sA); if(threadIdx.x==0) sMean=s/(float)cols; __syncthreads();
	const float mu=sMean;
	float v=0.f; for (int i=threadIdx.x;i<cols;i+=blockDim.x){ float d=pr[i]-mu; v+=d*d; }
	v=blockReduceSum(v,sB); if(threadIdx.x==0){ float var=v/(float)cols+eps; sSigma=sqrtf(var);} __syncthreads();
	const float inv=1.0f/sSigma;
	for (int i=threadIdx.x;i<cols;i+=blockDim.x){
		float xhat=(pr[i]-mu)*inv; float u=gamma[i]*xhat+beta[i]; float sa=tanhf(u); float sp=1.0f-sa*sa;
		duR[i]  = scale*a[i]*sp*dr[i];
		sdqR[i] = scale*sa*dr[i];
	}
}

// Deterministic column sum: one block per channel column, loop over rows.
//   out[j] += Σ_t in[t*cols+j].
__global__ void chiron_col_accumulate(const float* __restrict__ in, int rows, int cols,
                                      float* __restrict__ out)
{
	int j = blockIdx.x; if (j>=cols) return;
	float acc=0.f; for (int t=threadIdx.x; t<rows; t+=blockDim.x) acc += in[(size_t)t*cols + j];
	extern __shared__ float red[];
	acc = blockReduceSum(acc, red);
	if (threadIdx.x==0) out[j] += acc;
}

} // anonymous namespace

bool chiron_drift_backward(const float* dq_out, const float* p, const float* a,
                           const float* gamma, const float* beta, float scale,
                           int T, int m, float eps,
                           float* dp, float* da, float* dgamma, float* dbeta,
                           float* scratch_stats_split)
{
	if (T <= 0 || m <= 0) return true;
	// Internal scratch: du and sdq, each [T, m].  Raw cudaMalloc per call (see
	// header note); fine for the test, but Task 6 should pass pre-allocated
	// scratch on the training hot path.
	glades::gpu::GpuBuffer<float> du, sdq;
	if (!du.allocate((size_t)T * m) || !sdq.allocate((size_t)T * m)) return false;

	int block = rowBlockSize(m);
	int smemBytes = (block/32 + 2) * 2 * sizeof(float);
	chiron_drift_pre_backward_rows<<<T, block, smemBytes, computeStream()>>>(
	    p, dq_out, a, gamma, beta, scale, eps, m, du.data(), sdq.data());
	GLADES_CUDA_CHECK(cudaGetLastError());

	// da = colsum(sdq), accumulated.  One block per channel; deterministic.
	int rblock = 256; int rsmem = (rblock/32 + 1) * sizeof(float);
	chiron_col_accumulate<<<m, rblock, rsmem, computeStream()>>>(sdq.data(), T, m, da);
	GLADES_CUDA_CHECK(cudaGetLastError());

	// dp, dgamma, dbeta via the reanchor ReLN backward fed du + gamma_p.
	return chiron_reln_backward_reanchor(du.data(), p, gamma, T, m, eps,
	                                     dp, dgamma, dbeta, scratch_stats_split);
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
//  5b. SIRA terminal phase loss — final-state active loss + gradient.
// ===========================================================================

namespace {

__device__ __forceinline__ float sira_safe_positive_d(float x, float eps)
{
	const float e = (eps > 0.0f) ? eps : 1e-12f;
	return (x > e) ? x : e;
}

__device__ __forceinline__ float sira_pseudo_huber_d(float z, float tau)
{
	const float t = (tau > 0.0f) ? tau : 0.2f;
	const float r = z / t;
	return t * t * (sqrtf(1.0f + r * r) - 1.0f);
}

__device__ __forceinline__ float sira_pseudo_huber_grad_d(float z, float tau)
{
	const float t = (tau > 0.0f) ? tau : 0.2f;
	const float r = z / t;
	return z / sqrtf(1.0f + r * r);
}

__global__ void chiron_sira_terminal_stats_kernel(const float* __restrict__ q,
                                                  const float* __restrict__ p,
                                                  int n,
                                                  float* __restrict__ stats3)
{
	extern __shared__ float smem[];
	float sp = 0.0f;
	float sq = 0.0f;
	float sd = 0.0f;
	const int stride = gridDim.x * blockDim.x;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
	{
		const float pv = p[i];
		const float qv = q[i];
		sp += pv * pv;
		sq += qv * qv;
		sd += pv * qv;
	}

	const float bp = blockReduceSum(sp, smem);
	if (threadIdx.x == 0) atomicAdd(stats3 + 0, bp);
	__syncthreads();
	const float bq = blockReduceSum(sq, smem);
	if (threadIdx.x == 0) atomicAdd(stats3 + 1, bq);
	__syncthreads();
	const float bd = blockReduceSum(sd, smem);
	if (threadIdx.x == 0) atomicAdd(stats3 + 2, bd);
}

__global__ void chiron_sira_terminal_loss_kernel(const float* __restrict__ stats3,
                                                 int n,
                                                 float coef,
                                                 float energyWeight,
                                                 float balanceWeight,
                                                 float actionWeight,
                                                 float huberTau,
                                                 float eps,
                                                 float* __restrict__ loss1)
{
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	const float invN = 1.0f / static_cast<float>(n);
	const float p2MeanRaw = stats3[0] * invN;
	const float q2MeanRaw = stats3[1] * invN;
	const float pqMean    = stats3[2] * invN;
	const float p2 = sira_safe_positive_d(p2MeanRaw, eps);
	const float q2 = sira_safe_positive_d(q2MeanRaw, eps);
	const float energyZ = logf(sira_safe_positive_d(0.5f * (p2MeanRaw + q2MeanRaw), eps));
	const float balanceZ = 0.5f * (logf(p2) - logf(q2));
	const float actionDenom = sqrtf(p2 * q2);
	const float actionZ = (actionDenom > eps) ? (pqMean / actionDenom) : 0.0f;

	float loss = 0.0f;
	if (energyWeight > 0.0f)
		loss += energyWeight * sira_pseudo_huber_d(energyZ, huberTau);
	if (balanceWeight > 0.0f)
		loss += balanceWeight * sira_pseudo_huber_d(balanceZ, huberTau);
	if (actionWeight > 0.0f)
		loss += actionWeight * sira_pseudo_huber_d(actionZ, huberTau);
	loss1[0] = coef * loss;
}

__global__ void chiron_sira_terminal_grad_kernel(const float* __restrict__ q,
                                                 const float* __restrict__ p,
                                                 int n,
                                                 float coef,
                                                 float energyWeight,
                                                 float balanceWeight,
                                                 float actionWeight,
                                                 float huberTau,
                                                 float eps,
                                                 const float* __restrict__ stats3,
                                                 float gradScale,
                                                 float* __restrict__ dq,
                                                 float* __restrict__ dp)
{
	const float invN = 1.0f / static_cast<float>(n);
	const float p2MeanRaw = stats3[0] * invN;
	const float q2MeanRaw = stats3[1] * invN;
	const float pqMean    = stats3[2] * invN;
	const float p2 = sira_safe_positive_d(p2MeanRaw, eps);
	const float q2 = sira_safe_positive_d(q2MeanRaw, eps);
	const float energy = sira_safe_positive_d(0.5f * (p2MeanRaw + q2MeanRaw), eps);
	const float energyZ = logf(energy);
	const float balanceZ = 0.5f * (logf(p2) - logf(q2));
	const float actionDenom = sqrtf(p2 * q2);
	const float invActionDenom = (actionDenom > eps) ? (1.0f / actionDenom) : 0.0f;
	const float actionZ = (actionDenom > eps) ? (pqMean * invActionDenom) : 0.0f;

	const float energyCoeff = (energyWeight > 0.0f)
	    ? (coef * energyWeight * sira_pseudo_huber_grad_d(energyZ, huberTau))
	    : 0.0f;
	const float balanceCoeff = (balanceWeight > 0.0f)
	    ? (coef * balanceWeight * sira_pseudo_huber_grad_d(balanceZ, huberTau))
	    : 0.0f;
	const float actionCoeff = (actionWeight > 0.0f)
	    ? (coef * actionWeight * sira_pseudo_huber_grad_d(actionZ, huberTau))
	    : 0.0f;

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
	{
		const float pv = p[i];
		const float qv = q[i];
		float gp = 0.0f;
		float gq = 0.0f;
		if (energyCoeff != 0.0f)
		{
			const float c = energyCoeff * invN / energy;
			gp += c * pv;
			gq += c * qv;
		}
		if (balanceCoeff != 0.0f)
		{
			gp += balanceCoeff * invN * pv / p2;
			gq -= balanceCoeff * invN * qv / q2;
		}
		if (actionCoeff != 0.0f && invActionDenom > 0.0f)
		{
			gp += actionCoeff * invN * (qv * invActionDenom - actionZ * pv / p2);
			gq += actionCoeff * invN * (pv * invActionDenom - actionZ * qv / q2);
		}
		if (dp) dp[i] += gradScale * gp;
		if (dq) dq[i] += gradScale * gq;
	}
}

} // anonymous namespace

bool chiron_sira_terminal_forward(const float* q, const float* p,
                                  int n,
                                  float coef,
                                  float energyWeight,
                                  float balanceWeight,
                                  float actionWeight,
                                  float huberTau,
                                  float eps,
                                  float* stats3,
                                  float* loss1)
{
	if (coef <= 0.0f) return true;
	if (n <= 0) return true;
	if (energyWeight <= 0.0f && balanceWeight <= 0.0f && actionWeight <= 0.0f)
	{
		if (loss1) GLADES_CUDA_CHECK(cudaMemsetAsync(loss1, 0, sizeof(float), computeStream()));
		return true;
	}
	if (!q || !p || !stats3 || !loss1) return false;
	const float safeEps = (eps > 0.0f) ? eps : 1e-12f;
	GLADES_CUDA_CHECK(cudaMemsetAsync(stats3, 0, sizeof(float) * 3u, computeStream()));
	const int gridRaw = (n + kBlockElem - 1) / kBlockElem;
	const int grid = (gridRaw > 4096) ? 4096 : gridRaw;
	const int smemBytes = ((kBlockElem + 31) / 32) * sizeof(float);
	chiron_sira_terminal_stats_kernel<<<grid, kBlockElem, smemBytes, computeStream()>>>(
	    q, p, n, stats3);
	GLADES_CUDA_CHECK(cudaGetLastError());
	chiron_sira_terminal_loss_kernel<<<1, 1, 0, computeStream()>>>(
	    stats3, n, coef, energyWeight, balanceWeight, actionWeight,
	    huberTau, safeEps, loss1);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool chiron_sira_terminal_add_grad(const float* q, const float* p,
                                   int n,
                                   float coef,
                                   float energyWeight,
                                   float balanceWeight,
                                   float actionWeight,
                                   float huberTau,
                                   float eps,
                                   const float* stats3,
                                   float gradScale,
                                   float* dq,
                                   float* dp)
{
	if (coef <= 0.0f) return true;
	if (n <= 0 || gradScale == 0.0f) return true;
	if (energyWeight <= 0.0f && balanceWeight <= 0.0f && actionWeight <= 0.0f) return true;
	if (!q || !p || !stats3 || (!dq && !dp)) return false;
	const float safeEps = (eps > 0.0f) ? eps : 1e-12f;
	const int gridRaw = (n + kBlockElem - 1) / kBlockElem;
	const int grid = (gridRaw > 4096) ? 4096 : gridRaw;
	chiron_sira_terminal_grad_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
	    q, p, n, coef, energyWeight, balanceWeight, actionWeight,
	    huberTau, safeEps, stats3, gradScale, dq, dp);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
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

// Cast-elim Port C fwd slice (2026-06-12): library toggle.  When ON, the
// bf16w shear's Q/K/V projections write BF16-D directly into the
// scratch_*bf16 buffers (sgemm_rowmajor_bf16_dst_bf16) and the inner
// attention skips its three standalone casts.  The FP32 scratch_Q/K/V are
// then NOT materialized — callers that consume them (FP32-host checkpoint
// caches) must keep the toggle off; the trainer gates accordingly.
// Default OFF; set once at trainer init (not capture-safe to flip mid-run).
static bool g_cast_elim_inner_fwd = false;
void set_cast_elim_inner_fwd(bool on) { g_cast_elim_inner_fwd = on; }
bool get_cast_elim_inner_fwd() { return g_cast_elim_inner_fwd; }
static bool flash_attention_cublas_tiled_bf16_precast(
    const unsigned short* Qbf16, const unsigned short* Kbf16,
    const unsigned short* Vbf16,
    int T, int nHeads, int dHead, int dModel,
    bool causal,
    float* O,
    float* scratch_S,
    unsigned short* scratch_Pbf16);

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

	// Cast-elim Port C fwd slice: project straight to BF16-D and skip the
	// inner attention's standalone casts.  FP32 scratch_Q/K/V are NOT
	// written on this path (sole fwd consumers were the casts; checkpoint
	// saves read the BF16 scratches — trainer gates configs that need the
	// FP32 copies).  cuBLAS rounds the FP32 accumulator to BF16 (RNE) on
	// store — same rounding as the standalone cast; algorithm selection
	// for the D-type change is the gate-decided parity risk.
	if (g_cast_elim_inner_fwd)
	{
		if (!sgemm_rowmajor_bf16_dst_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wq_bf, dModel, 0.0f, scratch_Qbf16, dModel))
			return false;
		if (!sgemm_rowmajor_bf16_dst_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wk_bf, dModel, 0.0f, scratch_Kbf16, dModel))
			return false;
		if (!sgemm_rowmajor_bf16_dst_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wv_bf, dModel, 0.0f, scratch_Vbf16, dModel))
			return false;
		if (!flash_attention_cublas_tiled_bf16_precast(
		        scratch_Qbf16, scratch_Kbf16, scratch_Vbf16,
		        T, nHeads, dHead, dModel, causal,
		        scratch_O, scratch_S, scratch_Pbf16))
			return false;
	}
	else
	{
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
	}

	// Cast scratch_O -> BF16 for the output projection.
	if (!cast_f32_to_bf16(scratch_O, scratch_Obf, static_cast<size_t>(T) * dModel))
		return false;

	// Output projection: p += sign * scratch_Obf @ Wo_bf (BF16 TC, FP32 accum).
	const float sign = invert ? -1.0f : 1.0f;
	if (!sgemm_rowmajor_bf16(T, m, dModel, sign, scratch_Obf, dModel, Wo_bf, m, 1.0f, p, m))
		return false;
	return true;
}

// FP8 (E4M3) projection variant of chiron_attention_shear_bf16w_tiled.
// See gpu_chiron.h for the full contract.  All 4 projection GEMMs run
// through cuBLASLt's FP8 path with per-tensor scales computed on the fly
// via amax reductions.  Attention core stays BF16-TC (the tiled
// flash_attention path needs BF16 inputs, not FP8).  Each projection
// scale is computed once per call; in steady state weights change
// slowly enough that this is fine, and the amax kernels are tiny
// (T·m bytes total, sub-ms).
//
// Math (per projection):
//   stored_x = clamp(real_x * scale, ±448)  in E4M3
//   stored_w = clamp(real_w * scale_w, ±448)  in E4M3
//   real_out = (1 / (scale * scale_w)) · sum(stored_x · stored_w)
//
// The unscale is done in kernel_cast_bf16_to_fp32_scaled inside the
// sgemm_rowmajor_fp8_e4m3_bf16 wrapper.
bool chiron_attention_shear_fp8w_tiled(const float* q, float* p,
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
                                         unsigned short* scratch_Pbf16,
                                         float* d_scale_q,
                                         float* d_scale_Wq, float* d_scale_Wk,
                                         float* d_scale_Wv, float* d_scale_Wo,
                                         float* d_scale_O)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel = nHeads * dHead;

	// Cast q FP32 → BF16 once per layer (same as the BF16 path; the FP8
	// wrapper takes BF16 inputs and re-casts to E4M3 internally).
	if (!cast_f32_to_bf16(q, scratch_qbf, static_cast<size_t>(T) * m))
		return false;

	// Calibrate per-tensor scales from amax.  These are tiny reductions
	// (≤ 4 KB output per kernel) and pipeline on the same stream as the
	// GEMMs, so they don't add measurable latency vs. the projections.
	if (!fp8_calibrate_amax_e4m3_bf16(scratch_qbf, (size_t)T * m, d_scale_q)) return false;
	if (!fp8_calibrate_amax_e4m3_bf16(Wq_bf, (size_t)m * dModel, d_scale_Wq)) return false;
	if (!fp8_calibrate_amax_e4m3_bf16(Wk_bf, (size_t)m * dModel, d_scale_Wk)) return false;
	if (!fp8_calibrate_amax_e4m3_bf16(Wv_bf, (size_t)m * dModel, d_scale_Wv)) return false;
	if (!fp8_calibrate_amax_e4m3_bf16(Wo_bf, (size_t)dModel * m, d_scale_Wo)) return false;

	// FP8 Q/K/V projections (BF16 inputs, FP8 GEMM core, FP32 output).
	if (!sgemm_rowmajor_fp8_e4m3_bf16(T, dModel, m, 1.0f,
	        scratch_qbf, m, Wq_bf, dModel, 0.0f, scratch_Q, dModel,
	        d_scale_q, d_scale_Wq))
		return false;
	if (!sgemm_rowmajor_fp8_e4m3_bf16(T, dModel, m, 1.0f,
	        scratch_qbf, m, Wk_bf, dModel, 0.0f, scratch_K, dModel,
	        d_scale_q, d_scale_Wk))
		return false;
	if (!sgemm_rowmajor_fp8_e4m3_bf16(T, dModel, m, 1.0f,
	        scratch_qbf, m, Wv_bf, dModel, 0.0f, scratch_V, dModel,
	        d_scale_q, d_scale_Wv))
		return false;

	// Attention core stays BF16 (cuBLAS sgemm_batched_strided_bf16 + custom softmax).
	if (!flash_attention_cublas_tiled_bf16(
	        scratch_Q, scratch_K, scratch_V,
	        T, nHeads, dHead, dModel, causal,
	        scratch_O, scratch_S,
	        scratch_Qbf16, scratch_Kbf16, scratch_Vbf16, scratch_Pbf16))
		return false;

	// Cast scratch_O → BF16 for the output projection.
	if (!cast_f32_to_bf16(scratch_O, scratch_Obf, static_cast<size_t>(T) * dModel))
		return false;
	// Calibrate scratch_O scale (attention output magnitude is config-
	// dependent so we recalibrate per layer).
	if (!fp8_calibrate_amax_e4m3_bf16(scratch_Obf, (size_t)T * dModel, d_scale_O)) return false;

	// FP8 output projection: p += sign · scratch_Obf @ Wo_bf.
	const float sign = invert ? -1.0f : 1.0f;
	if (!sgemm_rowmajor_fp8_e4m3_bf16(T, m, dModel, sign,
	        scratch_Obf, dModel, Wo_bf, m, 1.0f, p, m,
	        d_scale_O, d_scale_Wo))
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

	// 4. dWo += sO^T · dp_new.  2026-05-13: switched to BF16 tensor cores —
	// scratch_qbf currently holds bf16(dp_new) from step 3 above; cast sO to
	// bf16 in scratch_sdbf and run atb_bf16.  ~2× faster than the previous
	// FP32 sgemm_rowmajor_atb on Ada.
	if (!cast_f32_to_bf16(sO, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_atb_bf16(dModel, m, T, 1.0f,
	        scratch_sdbf, dModel,  // sO^T   (BF16)
	        scratch_qbf, m,        // dp_new (BF16, from step 3)
	        1.0f, dWo, m))
		return false;

	// 5. Attention backward: FP32 throughout; writes sdQ/sdK/sdV as FP32.
	// iter 107 (2026-05-21): flash_attention_backward_cublas_tiled now uses
	// beta=0 (overwrite) for dV/dQ/dK — caller pre-zero is redundant.  Skipping
	// the 3 cudaMemsetAsync calls saves ~3 × 16 MB / layer (production scale).
	if (!flash_attention_backward_cublas_tiled(
	        sQ, sK, sV, sO, sdO,
	        T, nHeads, dHead, dModel, causal,
	        sdQ, sdK, sdV, scratch_P, scratch_dP))
		return false;

	// 2026-05-13 (paradigm-stack fix #1): weight gradients now also use BF16
	// tensor cores.  Pattern per X in {Q, K, V}: recast bf16(q) → scratch_qbf
	// only once (between steps 5 and 6), then for each X cast sdX → scratch_sdbf
	// and do both the activation grad (step 6) AND the weight grad (step 7) in
	// BF16 before moving to the next X.  This avoids needing extra scratches.
	if (!cast_f32_to_bf16(q, scratch_qbf, static_cast<size_t>(T) * m))
		return false;

	// 6+7 fused per direction.
	// Q:
	if (!cast_f32_to_bf16(sdQ, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, m, dModel, 1.0f, scratch_sdbf, dModel, Wq_bf, dModel, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_atb_bf16(m, dModel, T, 1.0f,
	        scratch_qbf, m, scratch_sdbf, dModel, 1.0f, dWq, dModel))
		return false;
	// K:
	if (!cast_f32_to_bf16(sdK, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, m, dModel, 1.0f, scratch_sdbf, dModel, Wk_bf, dModel, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_atb_bf16(m, dModel, T, 1.0f,
	        scratch_qbf, m, scratch_sdbf, dModel, 1.0f, dWk, dModel))
		return false;
	// V:
	if (!cast_f32_to_bf16(sdV, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, m, dModel, 1.0f, scratch_sdbf, dModel, Wv_bf, dModel, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_atb_bf16(m, dModel, T, 1.0f,
	        scratch_qbf, m, scratch_sdbf, dModel, 1.0f, dWv, dModel))
		return false;
	return true;
}

// iter 61 (2026-05-16): BF16-grad variant.  Same math as the FP32-grad
// _bf16w_tiled above, but the 4 dW weight-grad GEMMs route through
// sgemm_rowmajor_atb_bf16_dst_bf16 (cuBLAS gemmEx with D=BF16, beta=1) to
// write directly into the persistent BF16 dW buffers.  Eliminates the
// downstream bf16_accum_axpy commit kernel and the FP32 grad scratch
// traffic (~3.1% of GPU time, ~14 launches/step on the iter60 stack).
//
// Caller must pre-zero the BF16 dW buffers at the start of each
// gradient-accumulation window (matches the bf16_accum_axpy protocol).
// The FP32 internal accumulator inside cuBLAS gemmEx is identical to the
// old FP32-out path; only the final BF16 rounding step is folded into
// the GEMM rather than the standalone kernel.
bool chiron_attention_shear_backward_bf16w_bf16g_tiled(
    const float* q, const float* dp_new,
    const unsigned short* Wq_bf, const unsigned short* Wk_bf,
    const unsigned short* Wv_bf, const unsigned short* Wo_bf,
    int T, int m, int nHeads, int dHead,
    bool causal,
    float* dq,
    unsigned short* dWq_bf, unsigned short* dWk_bf,
    unsigned short* dWv_bf, unsigned short* dWo_bf,
    unsigned short* scratch_qbf,
    unsigned short* scratch_sdbf,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV,
    float* scratch_P, float* scratch_dP,
    bool dw_beta_zero)  // iter 108 FAIL: when true, dW_bf cuBLAS would use
                          // beta=0 (overwrite).  Math was non-bit-identical at
                          // production — NLL +0.5 nat drift.  Param retained
                          // for API stability but dw_beta forced to 1.0f
                          // (legacy behavior) regardless.
{
	(void)dw_beta_zero;  // iter 108 FAIL — param ignored, beta stays at 1.0f.
	const float dw_beta = 1.0f;
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
	// BF16G backward is parity-sensitive here: keep the score GEMM/softmax
	// path identical to flash_attention_cublas_tiled, but route only P·V
	// through strict FP32 cuBLAS math (no TF32 tensor-core contraction).
	{
		const float invSqrtDH = 1.0f / sqrtf(static_cast<float>(dHead));
		if (!sgemm_batched_strided_abt(
		        T, T, dHead,
		        invSqrtDH,
		        sQ, dModel, (long long)dHead,
		        sK, dModel, (long long)dHead,
		        0.0f,
		        scratch_P, T, (long long)T * T,
		        nHeads))
			return false;
		if (causal)
		{
			if (!causal_mask_softmax_inplace(scratch_P, nHeads, T))
				return false;
		}
		else if (!softmax_forward(scratch_P, nHeads * T, T, scratch_P))
		{
			return false;
		}
		if (!sgemm_batched_strided_exact(
		        T, dHead, T,
		        1.0f,
		        scratch_P, T, (long long)T * T,
		        sV, dModel, (long long)dHead,
		        0.0f,
		        sO, dModel, (long long)dHead,
		        nHeads))
			return false;
	}

	// 3. Output-projection backward.  dO = dp_new · Wo^T.
	if (!cast_f32_to_bf16(dp_new, scratch_qbf, static_cast<size_t>(T) * m))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, dModel, m, 1.0f, scratch_qbf, m, Wo_bf, m, 0.0f, sdO, dModel))
		return false;

	// 4. dWo += sO^T · dp_new.  BF16-out: commits to persistent dWo_bf.
	// iter 108: beta is dw_beta (0=overwrite when caller skips pre-zero
	// for accumSteps==1, 1=accumulate for multi-micro-batch grad accum).
	if (!cast_f32_to_bf16(sO, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_atb_bf16_dst_bf16(dModel, m, T, 1.0f,
	        scratch_sdbf, dModel,
	        scratch_qbf, m,
	        dw_beta, dWo_bf, m))
		return false;

	// 5. Attention backward: FP32 throughout; writes sdQ/sdK/sdV as FP32.
	// iter 107 (2026-05-21): caller-side memsets eliminated — beta=0 in
	// flash_attention_backward_cublas_tiled makes them redundant.
	if (!flash_attention_backward_cublas_tiled(
	        sQ, sK, sV, sO, sdO,
	        T, nHeads, dHead, dModel, causal,
	        sdQ, sdK, sdV, scratch_P, scratch_dP))
		return false;

	// Recast q to BF16 for the weight-grad GEMMs.
	if (!cast_f32_to_bf16(q, scratch_qbf, static_cast<size_t>(T) * m))
		return false;

	// 6+7 fused per direction.  dq accumulates FP32 (beta=1 — cross-Q/K/V dq
	// contributions sum into single dq buffer).  dW_bf uses dw_beta (iter 108:
	// 0=overwrite for single micro-batch, 1=accumulate for grad accum).
	// Q:
	if (!cast_f32_to_bf16(sdQ, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, m, dModel, 1.0f, scratch_sdbf, dModel, Wq_bf, dModel, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_atb_bf16_dst_bf16(m, dModel, T, 1.0f,
	        scratch_qbf, m, scratch_sdbf, dModel, dw_beta, dWq_bf, dModel))
		return false;
	// K:
	if (!cast_f32_to_bf16(sdK, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, m, dModel, 1.0f, scratch_sdbf, dModel, Wk_bf, dModel, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_atb_bf16_dst_bf16(m, dModel, T, 1.0f,
	        scratch_qbf, m, scratch_sdbf, dModel, dw_beta, dWk_bf, dModel))
		return false;
	// V:
	if (!cast_f32_to_bf16(sdV, scratch_sdbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!sgemm_rowmajor_abt_bf16(T, m, dModel, 1.0f, scratch_sdbf, dModel, Wv_bf, dModel, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_atb_bf16_dst_bf16(m, dModel, T, 1.0f,
	        scratch_qbf, m, scratch_sdbf, dModel, dw_beta, dWv_bf, dModel))
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

	// Tiled attention backward: writes dQ/dK/dV via beta=0 overwrite (iter 107).
	// iter 107 (2026-05-21): pre-zero memsets eliminated — flash_attention_backward
	// _cublas_tiled now writes with beta=0 (overwrite) for all 3 grads, making
	// the caller pre-zero redundant.
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
//
// iter 118 (2026-05-21): when set_iter118_fa_inner_fwd(true) is called,
// the cuBLAS+softmax+PV pipeline is replaced by a single FA-style fused
// kernel (flash_attention_multihead_forward).  FP32 compute (no tensor
// cores) — math validation Gate-0.  iter 119 will port to BF16/MMA for
// wall improvement.

namespace {
static bool g_iter118_fa_inner_fwd = false;
static bool g_iter119_fa_inner_bf16 = false;
}

void set_iter118_fa_inner_fwd(bool on)
{
	g_iter118_fa_inner_fwd = on;
}

// iter 119 (2026-05-21): BF16-input FA kernel.  Q/K/V are pre-cast to BF16
// by the cuBLAS pipeline already (scratch_Qbf16/Kbf16/Vbf16); when this
// toggle is on, we use those BF16 buffers directly with the BF16 FA kernel
// (flash_attention_multihead_forward_bf16) instead of the cuBLAS+softmax+PV
// pipeline.  Still FP32 compute inside the kernel (no tensor cores yet —
// iter 120+ for MMA wmma path).
void set_iter119_fa_inner_bf16(bool on)
{
	g_iter119_fa_inner_bf16 = on;
}

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
	// iter 118: drop-in FP32 FA kernel.  No BF16 casts; FP32 compute throughout.
	if (g_iter118_fa_inner_fwd)
	{
		return flash_attention_multihead_forward(
		    Q, K, V,
		    T, nHeads, /*nKVHeads=*/nHeads,
		    dHead, dModel, /*dModelKV=*/dModel,
		    causal, O);
	}
	// iter 119: cast Q/K/V to BF16 (same as cuBLAS pipeline), then use the
	// BF16-input FA kernel (still FP32 compute inside — no tensor cores yet).
	// Compared to iter 118: halves input memory bandwidth via BF16 loads.
	if (g_iter119_fa_inner_bf16)
	{
		if (!cast_f32_to_bf16(Q, scratch_Qbf16, nPacked)) return false;
		if (!cast_f32_to_bf16(K, scratch_Kbf16, nPacked)) return false;
		if (!cast_f32_to_bf16(V, scratch_Vbf16, nPacked)) return false;
		return flash_attention_multihead_forward_bf16(
		    scratch_Qbf16, scratch_Kbf16, scratch_Vbf16,
		    T, nHeads, /*nKVHeads=*/nHeads,
		    dHead, dModel, /*dModelKV=*/dModel,
		    causal, O);
	}

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

	// iter 102 (2026-05-21): when causal, fuse softmax + FP32→BF16 cast into
	// a single kernel that writes BF16 P directly to scratch_Pbf16.  Eliminates
	// the separate cast_f32_to_bf16 launch + the FP32 S → BF16 P memory
	// round-trip.  Math bit-identical to (softmax_inplace THEN cast).
	// Non-causal path keeps the legacy (softmax_forward + cast) chain since
	// the inner attention is always causal at CHIRON production (medalTrain=false).
	if (causal)
	{
		if (!causal_mask_softmax_bf16_out(scratch_S, scratch_Pbf16, nHeads, T))
			return false;
	}
	else
	{
		if (!softmax_forward(scratch_S, nHeads * T, T, scratch_S)) return false;
		const size_t nScores_nc = static_cast<size_t>(nHeads) * T * T;
		if (!cast_f32_to_bf16(scratch_S, scratch_Pbf16, nScores_nc)) return false;
	}

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

// Cast-elim Port C fwd slice (2026-06-12): pre-cast variant — identical
// pipeline to flash_attention_cublas_tiled_bf16 from the S GEMM onward,
// with Q/K/V already BF16 (the dst-BF16 projections wrote them).  The
// legacy function is untouched (no codegen risk to the default path).
static bool flash_attention_cublas_tiled_bf16_precast(
    const unsigned short* Qbf16, const unsigned short* Kbf16,
    const unsigned short* Vbf16,
    int T, int nHeads, int dHead, int dModel,
    bool causal,
    float* O,
    float* scratch_S,
    unsigned short* scratch_Pbf16)
{
	if (T <= 0 || nHeads <= 0 || dHead <= 0) return true;
	const float invSqrtDH = 1.0f / sqrtf(static_cast<float>(dHead));

	// S = (1/sqrt(dH)) Q K^T via BF16 batched (_abt).
	if (!sgemm_batched_strided_abt_bf16(
	        T, T, dHead, invSqrtDH,
	        Qbf16, dModel, (long long)dHead,
	        Kbf16, dModel, (long long)dHead,
	        0.0f,
	        scratch_S, T, (long long)T * T,
	        nHeads))
		return false;

	// Fused causal softmax + BF16 P write (iter 102 kernel); legacy split
	// path for the non-causal case (not hit at CHIRON production).
	if (causal)
	{
		if (!causal_mask_softmax_bf16_out(scratch_S, scratch_Pbf16, nHeads, T))
			return false;
	}
	else
	{
		if (!softmax_forward(scratch_S, nHeads * T, T, scratch_S)) return false;
		const size_t nScores_nc = static_cast<size_t>(nHeads) * T * T;
		if (!cast_f32_to_bf16(scratch_S, scratch_Pbf16, nScores_nc)) return false;
	}

	// O = P · V via BF16 batched (plain NN).
	if (!sgemm_batched_strided_bf16(
	        T, dHead, T, 1.0f,
	        scratch_Pbf16, T, (long long)T * T,
	        Vbf16, dModel, (long long)dHead,
	        0.0f,
	        O, dModel, (long long)dHead,
	        nHeads))
		return false;

	return true;
}

// Cast-elim V+O slice (2026-06-12): Task-4B (QK-Norm) production variant.
// Q and K arrive FP32 (post qknorm_forward_gpu + per-head scale — they MUST
// stay FP32 through QK-Norm) and are cast here exactly as the legacy
// pipeline does; V arrives PRE-CAST BF16 (the dst-BF16 projection wrote
// it); O is written directly as BF16 by the dst-BF16 P·V GEMM (RNE on
// store — same rounding as the legacy FP32-write + standalone cast).
// Eliminates the V input cast and the O output cast per call.
bool flash_attention_cublas_tiled_bf16_vpre_obf16(
    const float* Q, const float* K, const unsigned short* Vbf16,
    int T, int nHeads, int dHead, int dModel,
    bool causal,
    unsigned short* O_bf16,
    float* scratch_S,
    unsigned short* scratch_Qbf16, unsigned short* scratch_Kbf16,
    unsigned short* scratch_Pbf16)
{
	if (T <= 0 || nHeads <= 0 || dHead <= 0) return true;
	const float invSqrtDH = 1.0f / sqrtf(static_cast<float>(dHead));
	const size_t nPacked = static_cast<size_t>(T) * dModel;

	// Cast post-QK-Norm Q/K (required precision boundary); V is pre-cast.
	if (!cast_f32_to_bf16(Q, scratch_Qbf16, nPacked)) return false;
	if (!cast_f32_to_bf16(K, scratch_Kbf16, nPacked)) return false;

	// S = (1/sqrt(dH)) Q K^T via BF16 batched (_abt).
	if (!sgemm_batched_strided_abt_bf16(
	        T, T, dHead, invSqrtDH,
	        scratch_Qbf16, dModel, (long long)dHead,
	        scratch_Kbf16, dModel, (long long)dHead,
	        0.0f,
	        scratch_S, T, (long long)T * T,
	        nHeads))
		return false;

	if (causal)
	{
		if (!causal_mask_softmax_bf16_out(scratch_S, scratch_Pbf16, nHeads, T))
			return false;
	}
	else
	{
		if (!softmax_forward(scratch_S, nHeads * T, T, scratch_S)) return false;
		const size_t nScores_nc = static_cast<size_t>(nHeads) * T * T;
		if (!cast_f32_to_bf16(scratch_S, scratch_Pbf16, nScores_nc)) return false;
	}

	// O = P · V, written directly as BF16 (FP32 accumulate, RNE store).
	if (!sgemm_batched_strided_bf16_dst_bf16(
	        T, dHead, T, 1.0f,
	        scratch_Pbf16, T, (long long)T * T,
	        Vbf16, dModel, (long long)dHead,
	        0.0f,
	        O_bf16, dModel, (long long)dHead,
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

	// Recompute S = (1/sqrt(dH)) · Q · K^T into scratch_P.
	if (!sgemm_batched_strided_abt(
	        T, T, dHead, invSqrtDH,
	        Q, dModel, (long long)dHead,
	        K, dModel, (long long)dHead,
	        0.0f,
	        scratch_P, T, (long long)T * T,
	        nHeads))
		return false;

	// iter 115 (2026-05-21): reorder cuBLAS dP = dO · V^T to BEFORE the softmax
	// pass, so both inputs (S in scratch_P, dP in scratch_dP) are available for
	// the fused softmax+bwd_attn kernel.  cuBLAS is sequential on compute stream
	// — this is just a code-order reorder.  Saves 1 kernel launch per call
	// (24 calls/step at production) plus L2 cache benefit (P stays warm between
	// fused kernel's pass 3 write and pass A read, vs cross-kernel eviction
	// risk in the legacy split-kernel path).
	if (!sgemm_batched_strided_abt(
	        T, T, dHead, 1.0f,
	        dO, dModel, (long long)dHead,
	        V, dModel, (long long)dHead,
	        0.0f,
	        scratch_dP, T, (long long)T * T,
	        nHeads))
		return false;

	if (causal)
	{
		// iter 115: fused softmax (in-place S → P in scratch_P) + bwd_attn
		// (dP → dS in-place in scratch_dP).  Literal concatenation of the
		// two prior kernels' passes — math bit-identical at single-element
		// FP32 precision (no merged-pass FMA reorder, unlike iter 83 NEGATIVE
		// which merged the mask+max passes).
		if (!causal_softmax_with_bwd_attn(scratch_P, scratch_dP, nHeads, T, 1.0f, scratch_dP))
			return false;
	}
	else
	{
		// Non-causal: keep legacy split kernels (fused kernel only implements
		// the causal-mask path).
		if (!softmax_forward(scratch_P, nHeads * T, T, scratch_P)) return false;
		if (!softmax_backward_attn(scratch_P, scratch_dP, nHeads, T, 1.0f, scratch_dP))
			return false;
	}

	// iter 107 (2026-05-21): dV = P^T · dO (overwrite, was += accumulate).
	// All callers zero sdV before this call and don't depend on prior content.
	// Math bit-identical when dV starts at zero.
	if (!sgemm_batched_strided_atb(
	        T, dHead, T, 1.0f,
	        scratch_P, T, (long long)T * T,
	        dO, dModel, (long long)dHead,
	        0.0f,
	        dV, dModel, (long long)dHead,
	        nHeads))
		return false;

	// dQ = (1/sqrt(dH)) · dS · K.  Use strict FP32 cuBLAS math
	// for parity-sensitive replay (avoid TF32 tensor-core contraction).
	if (!sgemm_batched_strided_exact(
	        T, dHead, T, invSqrtDH,
	        scratch_dP, T, (long long)T * T,
	        K, dModel, (long long)dHead,
	        0.0f,
	        dQ, dModel, (long long)dHead,
	        nHeads))
		return false;

	// iter 107 (2026-05-21): dK = (1/sqrt(dH)) · dS^T · Q (overwrite, was +=).
	// Same rationale as dV above — all callers pre-zero sdK and don't depend
	// on prior content.  Math bit-identical when dK starts at zero.
	if (!sgemm_batched_strided_atb_exact(
	        T, dHead, T, invSqrtDH,
	        scratch_dP, T, (long long)T * T,
	        Q, dModel, (long long)dHead,
	        0.0f,
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

// Local-window BF16 shear forward.  Same orchestration as
// chiron_attention_shear_bf16 but uses the windowed flash kernel —
// per-query attention restricted to ±windowSize tokens.  At T=16384,
// windowSize=256 that's a 64× reduction on attention-core compute.
// windowSize <= 0 or >= T degenerates to the full-attention variant.
bool chiron_attention_shear_local_bf16(const float* q, float* p,
                                         const float* Wq, const float* Wk,
                                         const float* Wv, const float* Wo,
                                         int T, int m, int nHeads, int nKVHeads, int dHead,
                                         bool causal, bool invert, int windowSize,
                                         float* scratch_Q, float* scratch_K,
                                         float* scratch_V, float* scratch_O,
                                         uint16_t* scratch_Qbf, uint16_t* scratch_Kbf,
                                         uint16_t* scratch_Vbf)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel    = nHeads   * dHead;
	const int dModelKV  = nKVHeads * dHead;

	if (!sgemm_rowmajor(T, dModel, m, 1.0f, q, m, Wq, dModel, 0.0f, scratch_Q, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wk, dModelKV, 0.0f, scratch_K, dModelKV))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wv, dModelKV, 0.0f, scratch_V, dModelKV))
		return false;

	if (!cast_f32_to_bf16(scratch_Q, scratch_Qbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!cast_f32_to_bf16(scratch_K, scratch_Kbf, static_cast<size_t>(T) * dModelKV))
		return false;
	if (!cast_f32_to_bf16(scratch_V, scratch_Vbf, static_cast<size_t>(T) * dModelKV))
		return false;

	if (!flash_attention_multihead_forward_bf16_local(
	        scratch_Qbf, scratch_Kbf, scratch_Vbf,
	        T, nHeads, nKVHeads, dHead, dModel, dModelKV,
	        causal, windowSize, scratch_O))
		return false;

	const float sign = invert ? -1.0f : 1.0f;
	if (!sgemm_rowmajor(T, m, dModel, sign, scratch_O, dModel, Wo, m, 1.0f, p, m))
		return false;
	return true;
}

// Local-window BF16 shear backward.  Mirrors chiron_attention_shear_backward_bf16
// but swaps the flash backward for the windowed variant.
bool chiron_attention_shear_backward_local_bf16(
    const float* q, const float* dp_new,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    int T, int m, int nHeads, int nKVHeads, int dHead,
    bool causal, int windowSize,
    float* dq,
    float* dWq, float* dWk, float* dWv, float* dWo,
    float* sQ, float* sK, float* sV, float* sO,
    float* sdO, float* sdQ, float* sdK, float* sdV,
    uint16_t* scratch_Qbf, uint16_t* scratch_Kbf, uint16_t* scratch_Vbf)
{
	if (T <= 0 || m <= 0 || dHead <= 0 || nHeads <= 0) return true;
	const int dModel    = nHeads   * dHead;
	const int dModelKV  = nKVHeads * dHead;

	if (!sgemm_rowmajor(T, dModel,   m, 1.0f, q, m, Wq, dModel,   0.0f, sQ, dModel))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wk, dModelKV, 0.0f, sK, dModelKV))
		return false;
	if (!sgemm_rowmajor(T, dModelKV, m, 1.0f, q, m, Wv, dModelKV, 0.0f, sV, dModelKV))
		return false;

	if (!cast_f32_to_bf16(sQ, scratch_Qbf, static_cast<size_t>(T) * dModel))
		return false;
	if (!cast_f32_to_bf16(sK, scratch_Kbf, static_cast<size_t>(T) * dModelKV))
		return false;
	if (!cast_f32_to_bf16(sV, scratch_Vbf, static_cast<size_t>(T) * dModelKV))
		return false;

	if (!flash_attention_multihead_forward_bf16_local(
	        scratch_Qbf, scratch_Kbf, scratch_Vbf,
	        T, nHeads, nKVHeads, dHead, dModel, dModelKV,
	        causal, windowSize, sO))
		return false;

	if (!sgemm_rowmajor_abt(T, dModel, m, 1.0f, dp_new, m, Wo, m, 0.0f, sdO, dModel))
		return false;
	if (!sgemm_rowmajor_atb(dModel, m, T, 1.0f, sO, dModel, dp_new, m, 1.0f, dWo, m))
		return false;

	GLADES_CUDA_CHECK(cudaMemsetAsync(sdQ, 0, sizeof(float) * T * dModel,   computeStream()));
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdK, 0, sizeof(float) * T * dModelKV, computeStream()));
	GLADES_CUDA_CHECK(cudaMemsetAsync(sdV, 0, sizeof(float) * T * dModelKV, computeStream()));
	if (!flash_attention_multihead_backward_bf16_local(
	        scratch_Qbf, scratch_Kbf, scratch_Vbf,
	        sO, sdO,
	        T, nHeads, nKVHeads, dHead, dModel, dModelKV,
	        causal, windowSize, sdQ, sdK, sdV))
		return false;

	if (!sgemm_rowmajor_abt(T, m, dModel,   1.0f, sdQ, dModel,   Wq, dModel,   1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_abt(T, m, dModelKV, 1.0f, sdK, dModelKV, Wk, dModelKV, 1.0f, dq, m))
		return false;
	if (!sgemm_rowmajor_abt(T, m, dModelKV, 1.0f, sdV, dModelKV, Wv, dModelKV, 1.0f, dq, m))
		return false;

	if (!sgemm_rowmajor_atb(m, dModel,   T, 1.0f, q, m, sdQ, dModel,   1.0f, dWq, dModel))
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
