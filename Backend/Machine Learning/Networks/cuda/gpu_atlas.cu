// GPU-accelerated ATLAS optimizer (BRSP variant).
//
// Mirrors atlas_optimizer.cpp but operates entirely on device memory.
// Uses cuBLAS SGEMM for matrix products and custom CUDA kernels for
// elementwise / reduction operations.
//
// Requirements: CUDA 11+, SM 6.0+, cuBLAS.

#include "gpu_atlas.h"

#ifdef GLADES_HAVE_CUDA

#include "gpu_blas.h"
#include "gpu_device.h"
#include "Backend/Database/GLogger.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <vector>

namespace glades {
namespace gpu {

namespace {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define ATLAS_CUDA_CHECK(call)                                                \
	do {                                                                      \
		cudaError_t err_ = (call);                                            \
		if (err_ != cudaSuccess) {                                            \
			fprintf(stderr, "[atlas-gpu] %s:%d  %s  -> %s\n",                \
			        __FILE__, __LINE__, #call, cudaGetErrorString(err_));     \
			return false;                                                     \
		}                                                                     \
	} while (0)

static constexpr int kBlock = 256;

// Note: GPU memory tracking is done per-init via stderr logging.
// No global state is maintained — callers can aggregate if needed.

// ---------------------------------------------------------------------------
// Warp / block reduction primitives
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

__device__ __forceinline__ double warpReduceSumD(double val)
{
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}

__device__ double blockReduceSumD(double val, double* smem)
{
	int lane = threadIdx.x & 31;
	int wid  = threadIdx.x >> 5;

	val = warpReduceSumD(val);
	if (lane == 0) smem[wid] = val;
	__syncthreads();

	int numWarps = (blockDim.x + 31) / 32;
	val = (threadIdx.x < (unsigned)numWarps) ? smem[threadIdx.x] : 0.0;
	if (wid == 0) val = warpReduceSumD(val);
	return val;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// --- Gram-Schmidt orthonormalization ---
// Q is [m, r] row-major.  We process columns sequentially (j = 0..r-1).
// One block is launched; threads cooperate on dot products and normalization.
// Uses double accumulators for dot products and norms to match CPU path.
__global__ void atlas_gs_kernel(float* __restrict__ Q, int m, int r)
{
	extern __shared__ char smem_gs_bytes[];
	double* smem = reinterpret_cast<double*>(smem_gs_bytes);

	for (int j = 0; j < r; ++j)
	{
		// Subtract projections onto previous columns (modified GS)
		for (int p = 0; p < j; ++p)
		{
			double localDot = 0.0;
			for (int k = threadIdx.x; k < m; k += blockDim.x)
				localDot += (double)Q[k * r + j] * (double)Q[k * r + p];
			double dot = blockReduceSumD(localDot, smem);
			__syncthreads();

			for (int k = threadIdx.x; k < m; k += blockDim.x)
				Q[k * r + j] -= (float)(dot * (double)Q[k * r + p]);
			__syncthreads();
		}

		// Normalize column j
		double localNorm = 0.0;
		for (int k = threadIdx.x; k < m; k += blockDim.x)
		{
			double v = (double)Q[k * r + j];
			localNorm += v * v;
		}
		double norm = blockReduceSumD(localNorm, smem);
		__syncthreads();

		__shared__ float sInvNorm;
		if (threadIdx.x == 0)
		{
			norm = sqrt(norm);
			sInvNorm = (norm > 1e-12) ? (float)(1.0 / norm) : 0.0f;
		}
		__syncthreads();

		float inv = sInvNorm;
		if (inv > 0.0f)
		{
			for (int k = threadIdx.x; k < m; k += blockDim.x)
				Q[k * r + j] *= inv;
		}
		else
		{
			for (int k = threadIdx.x; k < m; k += blockDim.x)
				Q[k * r + j] = (k == j && j < m) ? 1.0f : 0.0f;
		}
		__syncthreads();
	}
}

// --- Weight decay: W[i] -= lr * (wd1*sign(W[i]) + wd2*W[i]) ---
__global__ void atlas_wd_kernel(float* __restrict__ W, size_t mn,
                                float lr, float wd1, float wd2)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= mn) return;
	float w = W[idx];
	float decay = 0.0f;
	if (wd1 != 0.0f)
	{
		float s = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
		decay += wd1 * s;
	}
	if (wd2 != 0.0f)
		decay += wd2 * w;
	W[idx] = w - lr * decay;
}

// --- Baseline update: W[i] -= baseScaled * gW[i] ---
__global__ void atlas_baseline_kernel(float* __restrict__ W,
                                      const float* __restrict__ gW,
                                      size_t mn, float baseScaled)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= mn) return;
	W[idx] -= baseScaled * gW[idx];
}

// --- Fisher diagonal update: EMA of mean-squared projected gradient ---
// Left subspace: gz[r, outerDim], direction c = contiguous row c.
// One block per subspace component.
__global__ void atlas_fisher_kernel(const float* __restrict__ gz,
                                    float* __restrict__ fisherDiag,
                                    int r, int outerDim,
                                    float beta, float oneMinusBeta)
{
	int c = blockIdx.x;
	if (c >= r) return;

	extern __shared__ char smem_fisher_bytes[];
	double* smem = reinterpret_cast<double*>(smem_fisher_bytes);
	const float* row = gz + (size_t)c * outerDim;

	double localSum = 0.0;
	for (int j = threadIdx.x; j < outerDim; j += blockDim.x)
	{
		double v = (double)row[j];
		localSum += v * v;
	}
	double sumsq = blockReduceSumD(localSum, smem);

	if (threadIdx.x == 0)
	{
		float meansq = (float)(sumsq / (double)outerDim);
		fisherDiag[c] = beta * fisherDiag[c] + oneMinusBeta * meansq;
	}
}

// Right subspace: gz[outerDim, r], direction c = column c (stride r).
// One block per subspace component.
// Each thread reads a contiguous chunk of the row (all r cols) but only
// accumulates column c — this gives coalesced reads when threads in a warp
// process consecutive rows.
__global__ void atlas_fisher_col_kernel(const float* __restrict__ gz,
                                         float* __restrict__ fisherDiag,
                                         int r, int outerDim,
                                         float beta, float oneMinusBeta)
{
	int c = blockIdx.x;
	if (c >= r) return;

	extern __shared__ char smem_fisher_bytes[];
	double* smem = reinterpret_cast<double*>(smem_fisher_bytes);

	// Threads stride by blockDim.x over rows — consecutive threads access
	// consecutive rows, so gz[i*r + c] and gz[(i+1)*r + c] differ by r floats.
	// With r=256, this is 1KB stride — not ideal for coalescing but acceptable
	// for the reduction pattern (compute-bound, not memory-bound).
	double localSum = 0.0;
	for (int i = threadIdx.x; i < outerDim; i += blockDim.x)
	{
		double v = (double)gz[(size_t)i * r + c];
		localSum += v * v;
	}
	double sumsq = blockReduceSumD(localSum, smem);

	if (threadIdx.x == 0)
	{
		float meansq = (float)(sumsq / (double)outerDim);
		fisherDiag[c] = beta * fisherDiag[c] + oneMinusBeta * meansq;
	}
}

// --- Prepare correction: fused PNG prediction + per-component scaling ---
// Left subspace: gz[r, outerDim], direction c = row c.
// Grid: (ceil(outerDim/block), r)
__global__ void atlas_correction_kernel(const float* __restrict__ gz,
                                        const float* __restrict__ prevGz,
                                        const float* __restrict__ fisherDiag,
                                        float* __restrict__ out,
                                        int r, int outerDim,
                                        float onePlusMu, float negMu,
                                        float baselineRate, float lr, float eps,
                                        float kappaLr, float bcFactor)
{
	int c = blockIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (c >= r || j >= outerDim) return;

	const float effFisher = fisherDiag[c] * bcFactor;
	float fisherLR = fminf(lr / (effFisher + eps), kappaLr);
	float corrScale = baselineRate - fisherLR;
	size_t idx = (size_t)c * outerDim + j;
	float gPred = onePlusMu * gz[idx] + negMu * prevGz[idx];
	out[idx] = corrScale * gPred;
}

// Right subspace: gz[outerDim, r] — each row has r elements (all directions).
// Process entire rows: each thread handles one element of a row, giving
// coalesced access when adjacent threads process adjacent columns.
// Grid: (ceil(r/block), outerDim) — each block row = one outerDim row.
__global__ void atlas_correction_col_kernel(const float* __restrict__ gz,
                                             const float* __restrict__ prevGz,
                                             const float* __restrict__ fisherDiag,
                                             float* __restrict__ out,
                                             int r, int outerDim,
                                             float onePlusMu, float negMu,
                                             float baselineRate, float lr, float eps,
                                             float kappaLr, float bcFactor)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;  // subspace direction
	int i = blockIdx.y;                               // row in outerDim
	if (c >= r || i >= outerDim) return;

	const float effFisher = fisherDiag[c] * bcFactor;
	float fisherLR = fminf(lr / (effFisher + eps), kappaLr);
	float corrScale = baselineRate - fisherLR;
	size_t idx = (size_t)i * r + c;
	float gPred = onePlusMu * gz[idx] + negMu * prevGz[idx];
	out[idx] = corrScale * gPred;
}

// --- Mu adaptation norms: dual reduction ---
// d_out[0] = sum((gz[i]-prevGz[i])^2)  (errNormSq)
// d_out[1] = sum(gz[i]^2)              (gzNormSq)
// Phase 1: each block writes partial sums to d_partials.
// d_partials layout: [errPartials(nBlocks), gzPartials(nBlocks)]
// where errPartials[blockIdx.x] = partial errNormSq,
//       gzPartials[blockIdx.x]  = partial gzNormSq.
__global__ void atlas_mu_norms_kernel(const float* __restrict__ gz,
                                      const float* __restrict__ prevGz,
                                      size_t rn, int nBlocks,
                                      float* __restrict__ d_partials)
{
	extern __shared__ float smem[];
	float* sErr = smem;
	float* sGz  = smem + (blockDim.x / 32 + 1);

	float localErr = 0.0f;
	float localGz  = 0.0f;
	for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rn;
	     i += (size_t)gridDim.x * blockDim.x)
	{
		float g = gz[i];
		float e = g - prevGz[i];
		localErr += e * e;
		localGz  += g * g;
	}

	float errSum = blockReduceSum(localErr, sErr);
	__syncthreads();
	float gzSum = blockReduceSum(localGz, sGz);

	if (threadIdx.x == 0)
	{
		d_partials[blockIdx.x] = errSum;
		d_partials[nBlocks + blockIdx.x] = gzSum;
	}
}

// Phase 2: single-block kernel reduces two channels of partials to d_out[0..1].
__global__ void atlas_reduce_partials2_kernel(const float* __restrict__ d_partials,
                                               int nBlocks,
                                               float* __restrict__ d_out)
{
	extern __shared__ float smem[];
	float* s0 = smem;
	float* s1 = smem + (blockDim.x / 32 + 1);

	float v0 = 0.0f;
	float v1 = 0.0f;
	for (int i = threadIdx.x; i < nBlocks; i += blockDim.x)
	{
		v0 += d_partials[i];
		v1 += d_partials[nBlocks + i];
	}
	float sum0 = blockReduceSum(v0, s0);
	__syncthreads();
	float sum1 = blockReduceSum(v1, s1);
	if (threadIdx.x == 0)
	{
		d_out[0] = sum0;
		d_out[1] = sum1;
	}
}

// --- EMA blend: dst[i] = (1-beta)*a[i] + beta*b[i] ---
// NOTE: dst may alias a or b. Each thread reads/writes only its own index
// so no cross-thread race exists, but we avoid __restrict__ to be safe.
__global__ void atlas_ema_kernel(float* dst,
                                 const float* a,
                                 const float* b,
                                 size_t count, float oneMinusBeta, float beta)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count) return;
	dst[idx] = oneMinusBeta * a[idx] + beta * b[idx];
}

// --- Transform Fisher diagonal into new basis ---
// f_new[c] = max(sum_j overlap[c*r+j]^2 * f_old[j], 1e-12)
// Small kernel: r threads, one block.
__global__ void atlas_transform_fisher_kernel(const float* __restrict__ overlap,
                                              const float* __restrict__ f_old,
                                              float* __restrict__ f_new,
                                              int r)
{
	int c = threadIdx.x;
	if (c >= r) return;

	float fNew = 0.0f;
	for (int j = 0; j < r; ++j)
	{
		float o = overlap[c * r + j];
		fNew += o * o * f_old[j];
	}
	f_new[c] = (fNew > 1e-12f) ? fNew : 1e-12f;
}

// --- Scale kernel: dst[i] = src[i] * scale ---
__global__ void atlas_scale_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    size_t n, float scale)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	dst[idx] = src[idx] * scale;
}

// --- NaN/Inf guard: clamp non-finite weights to zero ---
// Uses CUDA's isfinite() intrinsic which is safe under --use_fast_math,
// unlike the manual (x == x) && (x - x == 0) pattern which can be
// optimized away (same reasoning as the CPU atlas_isfinite comment).
__global__ void atlas_guard_kernel(float* __restrict__ W, size_t mn)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= mn) return;
	if (!isfinite(W[idx]))
		W[idx] = 0.0f;
}

// --- Sigma2 reduction: compute sum(gW[i]^2 * gScale^2) ---
// Phase 1: each block writes its partial sum to d_partials[blockIdx.x].
__global__ void atlas_sigma2_kernel(const float* __restrict__ gW,
                                    size_t mn, float gScale,
                                    float* __restrict__ d_partials)
{
	extern __shared__ float smem[];

	float localSum = 0.0f;
	for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < mn;
	     i += (size_t)gridDim.x * blockDim.x)
	{
		float v = gW[i] * gScale;
		localSum += v * v;
	}

	float sum = blockReduceSum(localSum, smem);
	if (threadIdx.x == 0)
		d_partials[blockIdx.x] = sum;
}

// Phase 2: single-block kernel reduces d_partials[0..nBlocks-1] to d_out[0].
// Deterministic: single block, fixed thread order, no atomics.
__global__ void atlas_reduce_partials_kernel(const float* __restrict__ d_partials,
                                              int nBlocks,
                                              float* __restrict__ d_out)
{
	extern __shared__ float smem[];
	float val = 0.0f;
	for (int i = threadIdx.x; i < nBlocks; i += blockDim.x)
		val += d_partials[i];
	float sum = blockReduceSum(val, smem);
	if (threadIdx.x == 0)
		d_out[0] = sum;
}

// --- On-GPU Cholesky factorization + inversion of a small [r x r] matrix ---
// Single-block kernel: thread 0 does the serial Cholesky/backsolve; all threads
// participate in loading/storing. For r <= 128 this is cheaper than two PCIe
// round-trips (D2H + H2D) that the old CPU path required.
// d_G: [r*r] symmetric positive-definite Gram matrix (overwritten with R^{-1})
// regularize: if true, adds 1e-4 diagonal shift for ill-conditioned inputs.
__global__ void atlas_cholesky_inv_kernel(float* __restrict__ d_G, int r,
                                           bool regularize)
{
	// Only thread 0 does the serial work (r is small — O(r^3) ≈ few ms for r=128).
	if (threadIdx.x != 0) return;

	float* G = d_G;  // in-place on global memory

	// 1. Column-norm pre-scaling: G_s[i,j] = G[i,j] / (sqrt(G[i,i]) * sqrt(G[j,j]))
	float maxDiag = 0.0f;
	for (int j = 0; j < r; ++j)
		if (G[j * r + j] > maxDiag) maxDiag = G[j * r + j];
	float diagFloor = 1e-12f * (maxDiag > 0.0f ? maxDiag : 1.0f);

	// Use d_G + r*r as scratch for colScale (caller ensures buffer is large enough).
	float* colScale = G + r * r;
	for (int j = 0; j < r; ++j)
	{
		float d = G[j * r + j] > diagFloor ? G[j * r + j] : diagFloor;
		colScale[j] = sqrtf(d);
	}
	for (int i = 0; i < r; ++i)
		for (int j = 0; j < r; ++j)
			G[i * r + j] /= (colScale[i] * colScale[j]);

	// 2. Regularize
	if (regularize)
		for (int j = 0; j < r; ++j)
			G[j * r + j] += 1e-4f;

	// 3. Cholesky: G_s = R^T R (R upper triangular, row-major)
	for (int j = 0; j < r; ++j)
	{
		float d = G[j * r + j];
		for (int k = 0; k < j; ++k)
			d -= G[k * r + j] * G[k * r + j];
		if (d < 1e-8f) d = 1e-8f;
		G[j * r + j] = sqrtf(d);
		float invD = 1.0f / G[j * r + j];
		for (int i = j + 1; i < r; ++i)
		{
			float v = G[j * r + i];
			for (int k = 0; k < j; ++k)
				v -= G[k * r + j] * G[k * r + i];
			G[j * r + i] = v * invD;
		}
	}

	// Zero lower triangle
	for (int i = 0; i < r; ++i)
		for (int j = 0; j < i; ++j)
			G[i * r + j] = 0.0f;

	// 4. Un-scale: R_true = R * diag(colScale)
	for (int i = 0; i < r; ++i)
		for (int j = i; j < r; ++j)
			G[i * r + j] *= colScale[j];

	// 5. In-place inversion of upper-triangular R via back-substitution.
	// Result overwrites G with R^{-1}.
	// First, copy R to colScale area as scratch (we need original R during inversion).
	float* R = colScale;  // reuse colScale area — we're done with it
	for (int i = 0; i < r * r; ++i)
		R[i] = G[i];

	// Zero G for accumulation
	for (int i = 0; i < r * r; ++i)
		G[i] = 0.0f;

	for (int j = r - 1; j >= 0; --j)
	{
		G[j * r + j] = 1.0f / R[j * r + j];
		for (int i = j - 1; i >= 0; --i)
		{
			float s = 0.0f;
			for (int k = i + 1; k <= j; ++k)
				s += R[i * r + k] * G[k * r + j];
			G[i * r + j] = -s / R[i * r + i];
		}
	}
}

} // anonymous namespace

// ===========================================================================
//  Host-side kernel wrappers
// ===========================================================================

bool atlas_gpu_gram_schmidt(float* d_Q, int m, int r)
{
	if (r <= 0 || m <= 0) return true;
	int blockSize = 256;
	if (blockSize > m) blockSize = ((m + 31) / 32) * 32;
	if (blockSize < 32) blockSize = 32;
	int smemBytes = ((blockSize / 32) + 1) * sizeof(double);
	atlas_gs_kernel<<<1, blockSize, smemBytes, computeStream()>>>(d_Q, m, r);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Single-pass Cholesky QR helper (fully on-GPU, no host round-trips).
// If `regularize` is true, adds diagonal shift for float32 stability.
// d_scratch_rr: device buffer of at least scratchBufElems floats (Gram matrix + scratch).
// d_qrTemp: device buffer [m*r].
static bool cholesky_qr_pass(float* d_Q, int m, int r,
                              float* d_scratch_rr, size_t scratchBufElems,
                              float* d_qrTemp,
                              bool regularize)
{
	// The Cholesky kernel uses d_scratch_rr[0..r*r-1] for the Gram matrix and
	// d_scratch_rr[r*r..2*r*r-1] as working scratch. Ensure the caller provided enough.
	const size_t requiredElems = (size_t)r * r * 2;
	if (scratchBufElems < requiredElems)
	{
		fprintf(stderr, "[atlas-gpu] cholesky_qr_pass: scratch buffer too small "
		        "(%zu < %zu for r=%d)\n", scratchBufElems, requiredElems, r);
		return false;
	}

	// 1. Gram matrix: G[r,r] = Q^T Q (on GPU via cuBLAS)
	if (!sgemm_rowmajor_atb(r, r, m, 1.0f, d_Q, r, d_Q, r,
	                         0.0f, d_scratch_rr, r))
		return false;

	// 2. Cholesky factorization + inversion entirely on GPU.
	// The kernel reads G from d_scratch_rr, writes R^{-1} back to d_scratch_rr.
	// It uses d_scratch_rr[r*r .. 2*r*r-1] as scratch for colScale and R copy.
	atlas_cholesky_inv_kernel<<<1, 1, 0, computeStream()>>>(d_scratch_rr, r, regularize);
	ATLAS_CUDA_CHECK(cudaGetLastError());

	// 3. Q_new = Q * R_inv via cuBLAS SGEMM
	if (!sgemm_rowmajor(m, r, r, 1.0f, d_Q, r, d_scratch_rr, r,
	                     0.0f, d_qrTemp, r))
		return false;

	// 4. Copy result back to Q
	ATLAS_CUDA_CHECK(cudaMemcpyAsync(d_Q, d_qrTemp,
	                                 (size_t)m * r * sizeof(float),
	                                 cudaMemcpyDeviceToDevice,
	                                 computeStream()));

	return true;
}

// CholeskyQR² — two-pass orthonormalization for machine-precision results.
// Pass 1 (regularized): handles ill-conditioned input from power iteration,
//   gets columns to approximately unit norm.
// Pass 2 (exact): Gram matrix is now well-conditioned (diag ≈ 1), so Cholesky
//   succeeds without regularization, producing exact orthonormality.
// d_scratch_rr must be a device buffer of at least 2*r*r floats
// (the Cholesky kernel uses d_scratch_rr[r*r .. 2*r*r-1] as scratch).
static bool cholesky_qr(float* d_Q, int m, int r,
                         float* d_scratch_rr, size_t scratchBufElems,
                         float* d_qrTemp)
{
	if (r <= 0 || m <= 0) return true;

	// Pass 1: regularized — handles ill-conditioned input
	if (!cholesky_qr_pass(d_Q, m, r, d_scratch_rr, scratchBufElems, d_qrTemp, true))
		return false;

	// Pass 2: exact — cleans up regularization artifacts
	if (!cholesky_qr_pass(d_Q, m, r, d_scratch_rr, scratchBufElems, d_qrTemp, false))
		return false;

	return true;
}

bool atlas_gpu_weight_decay(float* d_W, size_t mn, float lr, float wd1, float wd2)
{
	if (mn == 0) return true;
	if (wd1 == 0.0f && wd2 == 0.0f) return true;
	int grid = (int)((mn + kBlock - 1) / kBlock);
	atlas_wd_kernel<<<grid, kBlock, 0, computeStream()>>>(d_W, mn, lr, wd1, wd2);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_baseline_update(float* d_W, const float* d_gW, size_t mn, float baseScaled)
{
	if (mn == 0) return true;
	int grid = (int)((mn + kBlock - 1) / kBlock);
	atlas_baseline_kernel<<<grid, kBlock, 0, computeStream()>>>(d_W, d_gW, mn, baseScaled);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_fisher_update(const float* d_gz, float* d_fisherDiag,
                             int r, int outerDim, float beta, bool rightSubspace)
{
	if (r <= 0 || outerDim <= 0) return true;
	int blockSize = 256;
	if (blockSize > outerDim) blockSize = ((outerDim + 31) / 32) * 32;
	if (blockSize < 32) blockSize = 32;
	int smemBytes = ((blockSize / 32) + 1) * sizeof(double);
	if (rightSubspace)
		atlas_fisher_col_kernel<<<r, blockSize, smemBytes, computeStream()>>>(
			d_gz, d_fisherDiag, r, outerDim, beta, 1.0f - beta);
	else
		atlas_fisher_kernel<<<r, blockSize, smemBytes, computeStream()>>>(
			d_gz, d_fisherDiag, r, outerDim, beta, 1.0f - beta);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_prepare_correction(const float* d_gz, const float* d_prevGz,
                                   const float* d_fisherDiag,
                                   float* d_out, int r, int outerDim,
                                   float onePlusMu, float negMu,
                                   float baselineRate, float lr, float eps,
                                   float kappaLr, float bcFactor,
                                   bool rightSubspace)
{
	if (r <= 0 || outerDim <= 0) return true;
	dim3 block(kBlock);
	if (rightSubspace)
	{
		// Col kernel: x=direction (r), y=row (outerDim) — coalesced reads
		dim3 grid((r + kBlock - 1) / kBlock, outerDim);
		atlas_correction_col_kernel<<<grid, block, 0, computeStream()>>>(
			d_gz, d_prevGz, d_fisherDiag, d_out, r, outerDim,
			onePlusMu, negMu, baselineRate, lr, eps, kappaLr, bcFactor);
	}
	else
	{
		// Row kernel: x=element (outerDim), y=direction (r)
		dim3 grid((outerDim + kBlock - 1) / kBlock, r);
		atlas_correction_kernel<<<grid, block, 0, computeStream()>>>(
			d_gz, d_prevGz, d_fisherDiag, d_out, r, outerDim,
			onePlusMu, negMu, baselineRate, lr, eps, kappaLr, bcFactor);
	}
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_mu_norms(const float* d_gz, const float* d_prevGz,
                         size_t rn, float* d_out, float* d_partials)
{
	if (rn == 0) return true;
	int grid = (int)((rn + kBlock - 1) / kBlock);
	if (grid > 256) grid = 256;
	// Phase 1: per-block partial sums (layout: [errPartials(grid), gzPartials(grid)])
	int smemBytes = 2 * ((kBlock / 32) + 1) * sizeof(float);
	atlas_mu_norms_kernel<<<grid, kBlock, smemBytes, computeStream()>>>(d_gz, d_prevGz, rn, grid, d_partials);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	// Phase 2: single-block deterministic reduction
	int smemBytes2 = 2 * ((kBlock / 32) + 1) * sizeof(float);
	atlas_reduce_partials2_kernel<<<1, kBlock, smemBytes2, computeStream()>>>(d_partials, grid, d_out);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_ema_blend(float* d_dst, const float* d_a, const float* d_b,
                          size_t count, float beta)
{
	if (count == 0) return true;
	int grid = (int)((count + kBlock - 1) / kBlock);
	atlas_ema_kernel<<<grid, kBlock, 0, computeStream()>>>(d_dst, d_a, d_b, count, 1.0f - beta, beta);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_transform_fisher(const float* d_overlap, const float* d_f_old,
                                  float* d_f_new, int r)
{
	if (r <= 0) return true;
	int blockSize = ((r + 31) / 32) * 32;
	if (blockSize < 32) blockSize = 32;
	if (blockSize > 1024) blockSize = 1024;
	atlas_transform_fisher_kernel<<<1, blockSize, 0, computeStream()>>>(d_overlap, d_f_old, d_f_new, r);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_scale_grad(const float* d_gW, float* d_out, size_t mn, float gScale)
{
	if (mn == 0) return true;
	int grid = (int)((mn + kBlock - 1) / kBlock);
	atlas_scale_kernel<<<grid, kBlock, 0, computeStream()>>>(d_gW, d_out, mn, gScale);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_guard(float* d_W, size_t mn)
{
	if (mn == 0) return true;
	const int block = 256;
	int grid = (int)((mn + block - 1) / block);
	atlas_guard_kernel<<<grid, block, 0, computeStream()>>>(d_W, mn);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  atlas_gpu_init — allocate and initialize core state buffers
// ===========================================================================

bool atlas_gpu_init(GpuAtlasWeightState& state,
                    unsigned int m, unsigned int n,
                    unsigned int rank, float muInit,
                    glades::rng::Engine& rng)
{
	state.m = m;
	state.n = n;
	state.r = rank;
	if (state.r > m) state.r = m;
	if (state.r > n) state.r = n;
	if (state.r == 0u) state.r = 1u;

	// Dual-space selection: use whichever dimension is smaller for the subspace.
	state.rightSubspace = (m > n);

	// Dimension-proportional rank cap: subspace rank should not exceed 25% of
	// the subspace dimension. This prevents over-provisioning rank relative to
	// the available directions (e.g., for small layers where subDim < 4*rank).
	{
		const unsigned int subDim = state.rightSubspace ? n : m;
		const unsigned int maxRank = subDim / 4u;
		if (maxRank > 0u && state.r > maxRank)
			state.r = maxRank;
	}

	if (state.r > 256u)
	{
		fprintf(stderr, "[atlas-gpu] WARNING: rank %u > 256 may use significant GPU memory "
		        "(m=%u n=%u)\n", state.r, m, n);
	}

	const unsigned int r = state.r;
	const bool isRight = state.rightSubspace;
	const unsigned int subDim = isRight ? n : m;    // dimension the basis lives in
	const unsigned int outerDim = isRight ? m : n;  // the other dimension
	const size_t sr = (size_t)subDim * r;    // basis size: U/V [subDim, r]
	const size_t or_ = (size_t)outerDim * r; // projected gradient size: gz [outerDim, r] or [r, outerDim]

	// For left subspace: gz is [r, n] so or_ = r*n (but laid out as r*outerDim)
	// For right subspace: gz is [m, r] so or_ = m*r
	// Both cases: or_ = outerDim * r

	// Allocate persistent buffers
	if (!state.U.allocate(sr)) return false;
	if (!state.fisherDiag.allocate(r)) return false;
	if (!state.prevGz.allocate(or_)) return false;

	// Allocate per-step scratch buffers
	if (!state.gz.allocate(or_)) return false;
	if (!state.gPred.allocate(or_)) return false;
	if (!state.d_reduce.allocate(2)) return false;
	if (!state.d_partials.allocate(512)) return false;

	// Pre-allocate refresh scratch buffers
	if (!state.U_old.allocate(sr)) return false;
	if (!state.f_old.allocate(r)) return false;
	if (!state.B.allocate(or_)) return false;
	if (!state.overlap.allocate((size_t)r * r * 2)) return false;
	if (!state.prevGzOld.allocate(or_)) return false;
	state.refreshAllocated = true;

	// Pre-allocate Cholesky QR temp buffer
	if (state.qrTemp.size() < sr)
	{
		state.qrTemp.free();
		if (!state.qrTemp.allocate(sr)) return false;
	}

	// Initialize subspace basis with random Gaussian values on CPU, then upload.
	{
		std::vector<float> h_U(sr);
		const float scale = 1.0f / sqrtf((float)subDim);
		for (size_t i = 0; i < sr; ++i)
			h_U[i] = glades::rng::standard_normal(rng) * scale;

		if (!state.U.upload(h_U.data(), sr)) return false;
	}

	// Orthogonalize basis on device (Cholesky QR² for machine-precision results)
	if (!cholesky_qr(state.U.data(), (int)subDim, (int)r,
	                 state.overlap.data(), state.overlap.size(),
	                 state.qrTemp.data()))
		return false;

	// Initialize Fisher diagonal to eps (will be set from actual gradient
	// statistics on first step via bias correction or direct EMA).
	// Previous init of 1.0 caused ~7000 steps of wrong preconditioning
	// as Fisher decayed from 1.0 to the true value (~1e-6).
	{
		const float epsInit = 1e-8f;
		std::vector<float> initFisher(r, epsInit);
		if (!state.fisherDiag.upload(initFisher.data(), r)) return false;
	}

	// Zero prevGz
	if (!state.prevGz.zero()) return false;

	state.sigma2 = 1e-8f; // will be set from first step's actual gMeanSq
	state.mu = muInit;
	state.step = 0ULL;
	state.initialized = true;

	{
		const size_t totalElems = sr + r + or_
		    + or_ + or_ + 2 + 512
		    + sr + r + or_ + (size_t)r*r*2 + or_
		    + sr;
		fprintf(stderr, "[atlas-gpu] init m=%u n=%u r=%u %s total_gpu_bytes=%zu\n",
		        m, n, r, isRight ? "RIGHT" : "LEFT", totalElems * sizeof(float));
	}

	ATLAS_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
	return true;
}

// ===========================================================================
//  refreshSubspace — periodic subspace update via randomized power iteration
// ===========================================================================

// d_grad is the raw gradient [m*n], gScale folds into SGEMM alpha.
// Eigenvectors are scale-invariant, so we pass alpha=1 (no scaling).
//
// Left subspace (m <= n): power iteration finds top-r left singular vectors of G.
//   Q ∈ R^{m×r}: B = G^T Q, Z = G B, Q = orth(Z)
// Right subspace (m > n): power iteration finds top-r right singular vectors of G.
//   Q ∈ R^{n×r}: B = G Q, Z = G^T B, Q = orth(Z)
static bool refreshSubspace(GpuAtlasWeightState& state,
                            const float* d_grad,
                            unsigned int m, unsigned int n,
                            unsigned int powerIters, float betaRefresh)
{
	const unsigned int r = state.r;
	if (r == 0u || m == 0u || n == 0u) return true;

	const bool isRight = state.rightSubspace;
	const unsigned int subDim = isRight ? n : m;
	const unsigned int outerDim = isRight ? m : n;

	if (state.overlap.size() < (size_t)r * r * 2)
	{
		fprintf(stderr, "[atlas-gpu] FATAL: overlap buffer too small (%zu < %zu)\n",
		        state.overlap.size(), (size_t)r * r * 2);
		return false;
	}

	const size_t sr = (size_t)subDim * r;
	const size_t or_ = (size_t)outerDim * r;

	// Save old basis and Fisher
	ATLAS_CUDA_CHECK(cudaMemcpyAsync(state.U_old.data(), state.U.data(),
	                                 sr * sizeof(float), cudaMemcpyDeviceToDevice,
	                                 computeStream()));
	ATLAS_CUDA_CHECK(cudaMemcpyAsync(state.f_old.data(), state.fisherDiag.data(),
	                                 r * sizeof(float), cudaMemcpyDeviceToDevice,
	                                 computeStream()));

	float* d_Q = state.U.data();

	for (unsigned int p = 0; p < powerIters; ++p)
	{
		if (isRight)
		{
			// Right subspace: Q ∈ R^{n×r}
			// B[m,r] = G[m,n] * Q[n,r]
			if (!sgemm_rowmajor((int)m, (int)r, (int)n,
			                     1.0f,
			                     d_grad, (int)n,
			                     d_Q, (int)r,
			                     0.0f,
			                     state.B.data(), (int)r))
				return false;

			// Z[n,r] = G^T[n,m] * B[m,r]
			if (!sgemm_rowmajor_atb((int)n, (int)r, (int)m,
			                         1.0f,
			                         d_grad, (int)n,
			                         state.B.data(), (int)r,
			                         0.0f,
			                         d_Q, (int)r))
				return false;
		}
		else
		{
			// Left subspace: Q ∈ R^{m×r}
			// B[n,r] = G^T[n,m] * Q[m,r]
			if (!sgemm_rowmajor_atb((int)n, (int)r, (int)m,
			                         1.0f,
			                         d_grad, (int)n,
			                         d_Q, (int)r,
			                         0.0f,
			                         state.B.data(), (int)r))
				return false;

			// Z[m,r] = G[m,n] * B[n,r]
			if (!sgemm_rowmajor((int)m, (int)r, (int)n,
			                     1.0f,
			                     d_grad, (int)n,
			                     state.B.data(), (int)r,
			                     0.0f,
			                     d_Q, (int)r))
				return false;
		}

		if (!cholesky_qr(d_Q, (int)subDim, (int)r,
		                 state.overlap.data(), state.overlap.size(),
		                 state.qrTemp.data()))
			return false;
	}

	// --- EMA blend ---
	if (!atlas_gpu_ema_blend(d_Q, state.U_old.data(), d_Q, sr, betaRefresh))
		return false;

	// --- NaN guard on blended basis ---
	// The power iteration + CholeskyQR can produce NaN/Inf when the gradient
	// is nearly rank-deficient (e.g. layer 0 Wq with very small gradients).
	// CholeskyQR's R_inv amplifies tiny values by 1e10+, and cross-terms in
	// the back-substitution can overflow to Inf → NaN in the SGEMM output.
	// The EMA blend then propagates: 0.5*U_old + 0.5*NaN = NaN.
	//
	// Detection: sample a few elements of d_Q. If any are non-finite, the
	// entire matrix is likely corrupted (NaN propagates through SGEMM).
	// Recovery: restore U_old. This makes the refresh a no-op for this weight:
	// overlap = I, Fisher and prevGz are unchanged. The weight simply keeps
	// its current subspace until the next refresh with better-conditioned gradients.
	{
		float qCheck[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		ATLAS_CUDA_CHECK(cudaMemcpy(&qCheck[0], d_Q, sizeof(float), cudaMemcpyDeviceToHost));
		ATLAS_CUDA_CHECK(cudaMemcpy(&qCheck[1], d_Q + sr / 4, sizeof(float), cudaMemcpyDeviceToHost));
		ATLAS_CUDA_CHECK(cudaMemcpy(&qCheck[2], d_Q + sr / 2, sizeof(float), cudaMemcpyDeviceToHost));
		ATLAS_CUDA_CHECK(cudaMemcpy(&qCheck[3], d_Q + sr - 1, sizeof(float), cudaMemcpyDeviceToHost));
		if (!std::isfinite(qCheck[0]) || !std::isfinite(qCheck[1])
		    || !std::isfinite(qCheck[2]) || !std::isfinite(qCheck[3]))
		{
			// Restore old basis — refresh becomes no-op.
			ATLAS_CUDA_CHECK(cudaMemcpy(d_Q, state.U_old.data(),
			                            sr * sizeof(float), cudaMemcpyDeviceToDevice));
			// Restore old Fisher (f_old was saved earlier).
			ATLAS_CUDA_CHECK(cudaMemcpy(state.fisherDiag.data(), state.f_old.data(),
			                            r * sizeof(float), cudaMemcpyDeviceToDevice));
			// prevGz stays unchanged (we haven't modified it yet).
			return true; // skip overlap/transform — basis unchanged
		}
	}

	// Re-orthogonalize after blending
	if (!cholesky_qr(d_Q, (int)subDim, (int)r,
	                 state.overlap.data(), state.overlap.size(),
	                 state.qrTemp.data()))
		return false;

	// --- Overlap matrix O[r,r] = Q_new^T * Q_old ---
	if (!sgemm_rowmajor_atb((int)r, (int)r, (int)subDim,
	                         1.0f,
	                         d_Q, (int)r,
	                         state.U_old.data(), (int)r,
	                         0.0f,
	                         state.overlap.data(), (int)r))
		return false;

	// --- Transform Fisher diagonal into new basis ---
	if (!atlas_gpu_transform_fisher(state.overlap.data(), state.f_old.data(),
	                                  state.fisherDiag.data(), (int)r))
		return false;

	// --- Transform prevGz into new basis ---
	ATLAS_CUDA_CHECK(cudaMemcpyAsync(state.prevGzOld.data(), state.prevGz.data(),
	                                 or_ * sizeof(float), cudaMemcpyDeviceToDevice,
	                                 computeStream()));

	if (isRight)
	{
		// prevGz is [m, r]: prevGz_new[i,c] = sum_j prevGzOld[i,j] * overlap[j,c]
		// This is: prevGz_new[m,r] = prevGzOld[m,r] * overlap[r,r]
		if (!sgemm_rowmajor((int)outerDim, (int)r, (int)r,
		                     1.0f,
		                     state.prevGzOld.data(), (int)r,
		                     state.overlap.data(), (int)r,
		                     0.0f,
		                     state.prevGz.data(), (int)r))
			return false;
	}
	else
	{
		// prevGz is [r, n]: prevGz_new[c,j] = sum_k overlap[c,k] * prevGzOld[k,j]
		// This is: prevGz_new[r,n] = overlap[r,r] * prevGzOld[r,n]
		if (!sgemm_rowmajor((int)r, (int)outerDim, (int)r,
		                     1.0f,
		                     state.overlap.data(), (int)r,
		                     state.prevGzOld.data(), (int)outerDim,
		                     0.0f,
		                     state.prevGz.data(), (int)outerDim))
			return false;
	}

	return true;
}

// ===========================================================================
//  atlas_gpu_step — one full BRSP optimizer step on GPU
// ===========================================================================

bool atlas_gpu_step(GpuAtlasWeightState& state,
                    float* d_W, float* d_gW,
                    unsigned int m, unsigned int n,
                    float invBatch, float lr,
                    float wd1, float wd2, float gradScale,
                    const glades::ATLASConfig& ac,
                    shmea::GLogger* logger,
                    const char* tag)
{
	const float beta = ac.beta;
	const float muMin = ac.muMin;
	const float muMax = ac.muMax;
	const float eps = ac.eps;
	const unsigned int tSub = ac.tSub;
	const unsigned int powerIters = ac.powerIters;
	const float betaRefresh = ac.betaRefresh;
	const float kappaMax = ac.kappaMax;
	const float muGrowthRate = ac.muGrowthRate;
	if (!state.initialized) return false;

	const unsigned int r = state.r;
	const bool isRight = state.rightSubspace;
	const unsigned int outerDim = isRight ? m : n;
	const size_t mn = (size_t)m * n;
	const size_t or_ = (size_t)outerDim * r;  // gz/gPred/prevGz size
	state.step += 1ULL;

	// Guard incoming gradient against NaN/Inf before any computation.
	// A single NaN in d_gW would corrupt sigma2 (via reduction), the subspace
	// basis U (via power iteration at refresh), gz (via projection), and the
	// baseline update. Replacing NaN with 0 is safe: zero gradient entries
	// contribute nothing, and the weight matrix is unaffected by them.
	if (!atlas_gpu_guard(d_gW, mn))
		return false;

	const float gScale = invBatch * gradScale;

	// NaN diagnostic: verify guard worked (only at diagnostic steps)
	if (logger && tag && tSub > 0u && ((state.step % (unsigned long long)tSub) == 0ULL))
	{
		float gwSamples[3] = {0.0f, 0.0f, 0.0f};
		cudaMemcpy(&gwSamples[0], d_gW, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gwSamples[1], d_gW + mn / 2, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gwSamples[2], d_gW + mn - 1, sizeof(float), cudaMemcpyDeviceToHost);
		if (!std::isfinite(gwSamples[0]) || !std::isfinite(gwSamples[1]) || !std::isfinite(gwSamples[2]))
		{
			std::ostringstream oss;
			oss << "event=atlas_nan_diag tag=" << tag << " step=" << state.step
			    << " location=gW_after_guard"
			    << " gW[0]=" << gwSamples[0] << " gW[mid]=" << gwSamples[1]
			    << " gW[end]=" << gwSamples[2];
			logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
		}
	}

	// === Bias correction factor (matches CPU path) ===
	// Compensates for zero-initialization bias in EMA quantities.
	float bcFactor = 1.0f;
	if (ac.biasCorrection)
	{
		const double betaPow = pow((double)beta, (double)state.step);
		const double denom = 1.0 - betaPow;
		if (denom > 1e-15)
			bcFactor = (float)(1.0 / denom);
	}

	// === Step 1: Compute sigma2 (synchronous, matches CPU path) ===
	// Two-pass deterministic reduction: block partials → single-block sum.
	// NOTE: sigma2 uses gradScale (not gScale = invBatch*gradScale) to track
	// gradient variance on the accumulated-gradient scale. With gScale, the
	// invBatch² factor (~6e-10 for batch=40K) pushes sigma2 below the eps
	// floor for all weights, killing per-weight baseline rate adaptation.
	// Using gradScale keeps sigma2 ≈ mean(gW_accum²) which is O(0.01–0.1),
	// well above eps, allowing meaningful per-weight differentiation.
	// The invBatch normalization is applied later via baseScaled = rate*gScale.
	{
		int grid = (int)((mn + kBlock - 1) / kBlock);
		if (grid > 256) grid = 256;
		int smemBytes = ((kBlock / 32) + 1) * sizeof(float);
		// Phase 1: per-block partial sums
		atlas_sigma2_kernel<<<grid, kBlock, smemBytes>>>(d_gW, mn, gradScale,
		                                                  state.d_partials.data());
		ATLAS_CUDA_CHECK(cudaGetLastError());
		// Phase 2: single-block deterministic final reduction
		atlas_reduce_partials_kernel<<<1, kBlock, smemBytes>>>(
		    state.d_partials.data(), grid, state.d_reduce.data());
		ATLAS_CUDA_CHECK(cudaGetLastError());
		// Synchronous D2H — consume in the same step (matches CPU path).
		float h_sigma2Sum = 0.0f;
		ATLAS_CUDA_CHECK(cudaMemcpy(&h_sigma2Sum, state.d_reduce.data(),
		                              sizeof(float), cudaMemcpyDeviceToHost));
		float gMeanSq = h_sigma2Sum / (float)mn;
		if (state.step == 1ULL)
		{
			// First step: initialize sigma2 from actual gradient statistics
			// instead of decaying from arbitrary 1.0 initialization.
			state.sigma2 = (gMeanSq > eps) ? gMeanSq : eps;
		}
		else
		{
			state.sigma2 = beta * state.sigma2 + (1.0f - beta) * gMeanSq;
		}
		// Floor at eps to prevent sigma2 → 0 collapse that kills preconditioning.
		if (state.sigma2 < eps) state.sigma2 = eps;
		if (!std::isfinite(state.sigma2))
		{
			state.sigma2 = eps;
			if (logger)
			{
				std::ostringstream oss;
				oss << "event=atlas_gpu_sigma2_recovery step=" << state.step;
				if (tag) oss << " tag=" << tag;
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
		}
	}

	// === Step 2: Periodic subspace refresh ===
	// Pass raw d_gW directly — eigenvectors are scale-invariant.
	if (tSub > 0u && (state.step % (unsigned long long)tSub) == 0ULL)
	{
		// NaN diagnostic: check U BEFORE refresh
		if (logger && tag)
		{
			const unsigned int subDim = isRight ? n : m;
			const size_t sr = (size_t)subDim * r;
			float uPre[2] = {0.0f, 0.0f};
			cudaMemcpy(&uPre[0], state.U.data(), sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(&uPre[1], state.U.data() + sr - 1, sizeof(float), cudaMemcpyDeviceToHost);
			if (!std::isfinite(uPre[0]) || !std::isfinite(uPre[1]))
			{
				std::ostringstream oss;
				oss << "event=atlas_nan_diag tag=" << tag << " step=" << state.step
				    << " location=U_BEFORE_refresh"
				    << " U[0]=" << uPre[0] << " U[end]=" << uPre[1];
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
		}

		if (!refreshSubspace(state, d_gW, m, n, powerIters, betaRefresh))
		{
			if (logger)
			{
				std::ostringstream oss;
				oss << "event=atlas_gpu_refresh_failure step=" << state.step
				    << " m=" << m << " n=" << n << " rank=" << r;
				if (tag) oss << " tag=" << tag;
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
			return false;
		}

		// NaN diagnostic: check U after refresh
		if (logger && tag)
		{
			float uSample[2] = {0.0f, 0.0f};
			const unsigned int subDim = isRight ? n : m;
			const size_t sr = (size_t)subDim * r;
			cudaMemcpy(&uSample[0], state.U.data(), sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(&uSample[1], state.U.data() + sr / 2, sizeof(float), cudaMemcpyDeviceToHost);
			if (!std::isfinite(uSample[0]) || !std::isfinite(uSample[1]))
			{
				std::ostringstream oss;
				oss << "event=atlas_nan_diag tag=" << tag
				    << " step=" << state.step
				    << " location=U_after_refresh"
				    << " U[0]=" << uSample[0] << " U[mid]=" << uSample[1];
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
		}
	}

	// === Step 3: Decoupled weight decay ===
	if (!atlas_gpu_weight_decay(d_W, mn, lr, wd1, wd2))
		return false;

	// === Step 4: Project gradient to subspace ===
	if (isRight)
	{
		// Right subspace: gz[m,r] = gScale * G[m,n] * V[n,r]
		if (!sgemm_rowmajor((int)m, (int)r, (int)n,
		                     gScale,
		                     d_gW, (int)n,
		                     state.U.data(), (int)r,
		                     0.0f,
		                     state.gz.data(), (int)r))
			return false;
	}
	else
	{
		// Left subspace: gz[r,n] = gScale * U^T[r,m] * G[m,n]
		if (!sgemm_rowmajor_atb((int)r, (int)n, (int)m,
		                         gScale,
		                         state.U.data(), (int)r,
		                         d_gW, (int)n,
		                         0.0f,
		                         state.gz.data(), (int)n))
			return false;
	}

	// NaN diagnostic: check gz, prevGz, and U after projection
	if (logger && tag && (state.step % (unsigned long long)tSub) == 0ULL)
	{
		float gzSample[2] = {0.0f, 0.0f};
		float prevGzSample[2] = {0.0f, 0.0f};
		float uSample = 0.0f;
		float gwSample = 0.0f;
		cudaMemcpy(&gzSample[0], state.gz.data(), sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gzSample[1], state.gz.data() + or_ / 2, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&prevGzSample[0], state.prevGz.data(), sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&prevGzSample[1], state.prevGz.data() + or_ / 2, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&uSample, state.U.data(), sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gwSample, d_gW, sizeof(float), cudaMemcpyDeviceToHost);
		if (!std::isfinite(gzSample[0]) || !std::isfinite(gzSample[1])
		    || !std::isfinite(prevGzSample[0]) || !std::isfinite(prevGzSample[1]))
		{
			std::ostringstream oss;
			oss << "event=atlas_nan_diag tag=" << tag
			    << " step=" << state.step
			    << " location=gz_after_projection"
			    << " gz[0]=" << gzSample[0] << " gz[mid]=" << gzSample[1]
			    << " prevGz[0]=" << prevGzSample[0] << " prevGz[mid]=" << prevGzSample[1]
			    << " U[0]=" << uSample << " gW[0]=" << gwSample
			    << " gScale=" << gScale;
			logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
		}
	}

	// === Step 5: Update Fisher diagonal ===
	if (!atlas_gpu_fisher_update(state.gz.data(), state.fisherDiag.data(),
	                             (int)r, (int)outerDim, beta, isRight))
		return false;

	// === Step 6: Full-space baseline update ===
	{
		// Apply bias correction to sigma2 (matches CPU path).
		const float effSigma2 = state.sigma2 * bcFactor;
		const float kappaLr = kappaMax * lr;
		float rawBaselineRate = lr / (effSigma2 + eps);
		if (rawBaselineRate > kappaLr) rawBaselineRate = kappaLr;
		const float baselineRate = rawBaselineRate;
		state.lastBaselineRate = baselineRate;
		const float baseScaled = baselineRate * gScale;
		if (!atlas_gpu_baseline_update(d_W, d_gW, mn, baseScaled))
			return false;

		// === Step 7: Subspace correction with PNG ===
		const float onePlusMu = 1.0f + state.mu;
		const float negMu = -state.mu;

		if (!atlas_gpu_prepare_correction(state.gz.data(), state.prevGz.data(),
		                                   state.fisherDiag.data(),
		                                   state.gPred.data(), (int)r, (int)outerDim,
		                                   onePlusMu, negMu,
		                                   baselineRate, lr, eps, kappaLr,
		                                   bcFactor, isRight))
			return false;

		if (isRight)
		{
			// Right subspace: W[m,n] += gPred[m,r] * V^T[r,n]
			// sgemm_rowmajor_abt: C[M,N] = A[M,K] * B^T[K,N] → B is [N,K]
			if (!sgemm_rowmajor_abt((int)m, (int)n, (int)r,
			                         1.0f,
			                         state.gPred.data(), (int)r,
			                         state.U.data(), (int)r,
			                         1.0f,
			                         d_W, (int)n))
				return false;
		}
		else
		{
			// Left subspace: W[m,n] += U[m,r] * gPred[r,n]
			if (!sgemm_rowmajor((int)m, (int)n, (int)r,
			                     1.0f,
			                     state.U.data(), (int)r,
			                     state.gPred.data(), (int)n,
			                     1.0f,
			                     d_W, (int)n))
				return false;
		}

		// Guard against NaN/Inf propagation (matches CPU path)
		if (!atlas_gpu_guard(d_W, mn))
			return false;
	}

	// === Step 8: Compute mu adaptation (synchronous, matches CPU path) ===
	if (state.step > 1ULL && muMax > 0.0f)
	{
		if (!atlas_gpu_mu_norms(state.gz.data(), state.prevGz.data(),
		                         or_, state.d_reduce.data(), state.d_partials.data()))
			return false;
		// Synchronous D2H — consume in the same step (matches CPU path).
		float h_muNorms[2] = {0.0f, 0.0f};
		ATLAS_CUDA_CHECK(cudaMemcpy(h_muNorms, state.d_reduce.data(),
		                              2 * sizeof(float), cudaMemcpyDeviceToHost));
		float errNormSq = h_muNorms[0];
		float gzNormSq  = h_muNorms[1];
		float gzNorm = sqrtf(gzNormSq);
		if (gzNorm > 1e-12f)
		{
			float ratio = sqrtf(errNormSq) / (gzNorm + 1e-12f);
			float newMu = state.mu * (1.0f - ratio)
			            + muGrowthRate * (muMax - state.mu);
			if (newMu < muMin) newMu = muMin;
			if (newMu > muMax) newMu = muMax;
			state.mu = newMu;
		}
		if (!std::isfinite(state.mu))
		{
			state.mu = muMin;
			if (logger)
			{
				std::ostringstream oss;
				oss << "event=atlas_gpu_mu_recovery step=" << state.step;
				if (tag) oss << " tag=" << tag;
				logger->warning("ATLAS", shmea::GString(oss.str().c_str()));
			}
		}
	}

	// === Periodic diagnostics (mirrors CPU atlas_optimizer.cpp) ===
	const bool diagStep = logger && tSub > 0u
	    && (state.step % (unsigned long long)tSub) == 0ULL;
	if (diagStep)
	{
		AtlasGpuDiag diag = atlas_gpu_get_diag(state);
		std::ostringstream oss;
		oss << "event=atlas_gpu_step";
		if (tag) oss << " tag=" << tag;
		oss << " step=" << state.step
		    << " m=" << m << " n=" << n << " rank=" << r
		    << " subspace=" << (isRight ? "right" : "left")
		    << " lr=" << lr
		    << " mu=" << state.mu
		    << " sigma2=" << state.sigma2
		    << " bc_factor=" << bcFactor;
		if (diag.valid)
		{
			oss << " baseline_rate=" << diag.baselineRate
			    << " fisher_min=" << diag.fisherMin
			    << " fisher_max=" << diag.fisherMax
			    << " fisher_mean=" << diag.fisherMean
			    << " fisher_median=" << diag.fisherMedian
			    << " fisher_ratio=" << (diag.fisherMin > 1e-15f ? diag.fisherMax / diag.fisherMin : 0.0f)
			    << " sigma2_fisher_ratio=" << (diag.fisherMean > 1e-15f ? diag.sigma2 / diag.fisherMean : 0.0f)
			    << " effective_rank=" << diag.effectiveRank
			    << " spectral_efficiency=" << diag.spectralEfficiency
			    << " top1_concentration=" << diag.top1Concentration
			    << " top10_concentration=" << diag.top10Concentration
			    << " gz_norm=" << diag.gzNorm
			    << " pred_err_norm=" << diag.updateNorm;
		}
		logger->info("ATLAS", shmea::GString(oss.str().c_str()));
	}

	// === Step 9: Store compressed gradient for next step ===
	ATLAS_CUDA_CHECK(cudaMemcpyAsync(state.prevGz.data(), state.gz.data(),
	                                 or_ * sizeof(float), cudaMemcpyDeviceToDevice,
	                                 computeStream()));

	// === Step 10: Clear accumulated gradients ===
	ATLAS_CUDA_CHECK(cudaMemsetAsync(d_gW, 0, (size_t)mn * sizeof(float), computeStream()));

	return true;
}

bool atlas_gpu_update(GpuAtlasWeightState& state,
                      float* d_W, float* d_gW,
                      unsigned int m, unsigned int n,
                      float invBatch, float lr,
                      float wd1, float wd2, float gradScale,
                      const glades::ATLASConfig& ac,
                      glades::rng::Engine& rng,
                      shmea::GLogger* logger,
                      const char* tag)
{
	if (!state.initialized && m > 0u && n > 0u)
	{
		if (!atlas_gpu_init(state, m, n, ac.rank, ac.muMin, rng))
			return false;
	}
	return atlas_gpu_step(state, d_W, d_gW, m, n, invBatch, lr, wd1, wd2, gradScale,
	                      ac, logger, tag);
}

AtlasGpuDiag atlas_gpu_get_diag(const GpuAtlasWeightState& state)
{
	AtlasGpuDiag d;
	if (!state.initialized)
		return d;

	const unsigned int r = state.r;
	d.sigma2 = state.sigma2;
	d.mu = state.mu;
	d.step = state.step;
	d.baselineRate = state.lastBaselineRate;
	d.rightSubspace = state.rightSubspace;

	// Download Fisher diagonal to compute min/max/mean/median
	if (r > 0u)
	{
		std::vector<float> h_fisher(r);
		cudaError_t err = cudaMemcpy(h_fisher.data(), state.fisherDiag.data(),
		                              r * sizeof(float), cudaMemcpyDeviceToHost);
		if (err == cudaSuccess)
		{
			float fMin = h_fisher[0], fMax = h_fisher[0], fSum = 0.0f;
			for (unsigned int c = 0; c < r; ++c)
			{
				float f = h_fisher[c];
				if (f < fMin) fMin = f;
				if (f > fMax) fMax = f;
				fSum += f;
			}
			d.fisherMin = fMin;
			d.fisherMax = fMax;
			d.fisherMean = fSum / (float)r;

			// Median (partial sort)
			std::vector<float> sorted(h_fisher);
			std::nth_element(sorted.begin(), sorted.begin() + (int)(r/2), sorted.end());
			d.fisherMedian = sorted[r/2];

			// Amplification ratio range (sigma2/fisher gives relative LR multiplier)
			d.corrScaleMin = fMin > 1e-12f ? d.sigma2 / fMin : 0.0f;
			d.corrScaleMax = fMax > 1e-12f ? d.sigma2 / fMax : 0.0f;

			// --- Spectral efficiency: how well the rank is utilized ---
			if (fSum > 1e-30f)
			{
				// Fisher entropy: H = -Σ p_c ln(p_c)
				double entropy = 0.0;
				for (unsigned int c = 0; c < r; ++c)
				{
					double p = (double)h_fisher[c] / (double)fSum;
					if (p > 1e-30)
						entropy -= p * log(p);
				}
				d.effectiveRank = (float)exp(entropy);
				d.spectralEfficiency = d.effectiveRank / (float)r;

				// Top-1 concentration
				d.top1Concentration = fMax / fSum;

				// Top-10 concentration (or top-r if r < 10)
				const unsigned int topK = (r < 10u) ? r : 10u;
				std::nth_element(sorted.begin(), sorted.begin() + (int)(r - topK), sorted.end());
				double topKSum = 0.0;
				for (unsigned int c = r - topK; c < r; ++c)
					topKSum += (double)sorted[c];
				d.top10Concentration = (float)(topKSum / (double)fSum);
			}
		}
	}

	// Download gz norm (from last step's d_reduce[1] = gzNormSq)
	if (state.d_reduce.size() >= 2)
	{
		float h_norms[2] = {0.0f, 0.0f};
		cudaError_t err = cudaMemcpy(h_norms, state.d_reduce.data(),
		                              2 * sizeof(float), cudaMemcpyDeviceToHost);
		if (err == cudaSuccess)
		{
			d.gzNorm = sqrtf(h_norms[1]);
			d.updateNorm = sqrtf(h_norms[0]);
		}
	}

	d.valid = true;
	return d;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
