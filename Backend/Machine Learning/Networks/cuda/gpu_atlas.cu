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
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

// Shared temp buffer for Cholesky QR SGEMM output.
// Pre-allocated in atlas_gpu_init to avoid mid-training GPU allocations.
static GpuBuffer<float> s_qrTemp;

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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// --- Gram-Schmidt orthonormalization ---
// Q is [m, r] row-major.  We process columns sequentially (j = 0..r-1).
// One block is launched; threads cooperate on dot products and normalization.
__global__ void atlas_gs_kernel(float* __restrict__ Q, int m, int r)
{
	extern __shared__ float smem[];

	for (int j = 0; j < r; ++j)
	{
		// Subtract projections onto previous columns (modified GS)
		for (int p = 0; p < j; ++p)
		{
			float localDot = 0.0f;
			for (int k = threadIdx.x; k < m; k += blockDim.x)
				localDot += Q[k * r + j] * Q[k * r + p];
			float dot = blockReduceSum(localDot, smem);
			__syncthreads();

			for (int k = threadIdx.x; k < m; k += blockDim.x)
				Q[k * r + j] -= dot * Q[k * r + p];
			__syncthreads();
		}

		// Normalize column j
		float localNorm = 0.0f;
		for (int k = threadIdx.x; k < m; k += blockDim.x)
		{
			float v = Q[k * r + j];
			localNorm += v * v;
		}
		float norm = blockReduceSum(localNorm, smem);
		__syncthreads();

		__shared__ float sInvNorm;
		if (threadIdx.x == 0)
		{
			norm = sqrtf(norm);
			sInvNorm = (norm > 1e-12f) ? (1.0f / norm) : 0.0f;
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
__global__ void atlas_wd_kernel(float* __restrict__ W, int mn,
                                float lr, float wd1, float wd2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
                                      int mn, float baseScaled)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= mn) return;
	W[idx] -= baseScaled * gW[idx];
}

// --- Fisher diagonal update: EMA of row-wise mean-squared of gz[r, n] ---
// One block per row (subspace component).
__global__ void atlas_fisher_kernel(const float* __restrict__ gz,
                                    float* __restrict__ fisherDiag,
                                    int r, int n,
                                    float beta, float oneMinusBeta)
{
	int c = blockIdx.x;
	if (c >= r) return;

	extern __shared__ float smem[];
	const float* row = gz + (size_t)c * n;

	float localSum = 0.0f;
	for (int j = threadIdx.x; j < n; j += blockDim.x)
	{
		float v = row[j];
		localSum += v * v;
	}
	float sumsq = blockReduceSum(localSum, smem);

	if (threadIdx.x == 0)
	{
		float meansq = sumsq / (float)n;
		fisherDiag[c] = beta * fisherDiag[c] + oneMinusBeta * meansq;
	}
}

// --- Prepare correction: fused PNG prediction + per-component scaling ---
// out[c*n+j] = corrScale[c] * ((1+mu)*gz[c*n+j] - mu*prevGz[c*n+j])
// where corrScale[c] = baselineRate - lr/(max(fisherDiag[c], sigma2) + eps)
// Grid: (ceil(n/block), r)
__global__ void atlas_correction_kernel(const float* __restrict__ gz,
                                        const float* __restrict__ prevGz,
                                        const float* __restrict__ fisherDiag,
                                        float* __restrict__ out,
                                        int r, int n,
                                        float onePlusMu, float negMu,
                                        float baselineRate, float lr, float eps,
                                        float kappaLr)
{
	int c = blockIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (c >= r || j >= n) return;

	// Cap the per-direction Fisher LR at kappaMax*lr (= kappaLr).
	// The net effective LR in direction c is lr/(f_c+eps).  Without capping,
	// this can be 1000x the base lr when f_c is small (e.g. after basis refresh
	// or when sigma2 converges to small gradient variance).
	// With both baselineRate and fisherLR capped at kappaLr, corrScale ∈ [0, baselineRate]
	// and the correction can only reduce the effective LR, never amplify beyond kappaLr.
	float fisherLR = fminf(lr / (fisherDiag[c] + eps), kappaLr);
	float corrScale = baselineRate - fisherLR;
	size_t idx = (size_t)c * n + j;
	float gPred = onePlusMu * gz[idx] + negMu * prevGz[idx];
	out[idx] = corrScale * gPred;
}

// --- Mu adaptation norms: dual reduction ---
// d_out[0] = sum((gz[i]-prevGz[i])^2)  (errNormSq)
// d_out[1] = sum(gz[i]^2)              (gzNormSq)
__global__ void atlas_mu_norms_kernel(const float* __restrict__ gz,
                                      const float* __restrict__ prevGz,
                                      int rn, float* __restrict__ d_out)
{
	extern __shared__ float smem[];
	float* sErr = smem;
	float* sGz  = smem + (blockDim.x / 32 + 1);

	float localErr = 0.0f;
	float localGz  = 0.0f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rn;
	     i += gridDim.x * blockDim.x)
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
		atomicAdd(&d_out[0], errSum);
		atomicAdd(&d_out[1], gzSum);
	}
}

// --- EMA blend: dst[i] = (1-beta)*a[i] + beta*b[i] ---
// NOTE: dst may alias a or b. Each thread reads/writes only its own index
// so no cross-thread race exists, but we avoid __restrict__ to be safe.
__global__ void atlas_ema_kernel(float* dst,
                                 const float* a,
                                 const float* b,
                                 int count, float oneMinusBeta, float beta)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

// --- Sigma2 reduction: compute sum(gW[i]^2 * gScale^2) ---
// Outputs a single float to d_out[0].
__global__ void atlas_sigma2_kernel(const float* __restrict__ gW,
                                    int mn, float gScale,
                                    float* __restrict__ d_out)
{
	extern __shared__ float smem[];

	float localSum = 0.0f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < mn;
	     i += gridDim.x * blockDim.x)
	{
		float v = gW[i] * gScale;
		localSum += v * v;
	}

	float sum = blockReduceSum(localSum, smem);
	if (threadIdx.x == 0)
		atomicAdd(&d_out[0], sum);
}

// --- Fast multi-block Modified Gram-Schmidt ---
// Phase 1: compute dot products for column j.
// Block k computes Q[:,j] · Q[:,j+k] and writes to d_dots[k].
// Block 0 computes the self-dot (norm squared).
__global__ void atlas_gs_dots_kernel(const float* __restrict__ Q,
                                      int m, int r, int j,
                                      float* __restrict__ d_dots)
{
	extern __shared__ float smem[];
	int target = blockIdx.x;  // which dot product this block computes
	int colK = j + target;

	float localSum = 0.0f;
	for (int row = threadIdx.x; row < m; row += blockDim.x)
		localSum += Q[row * r + j] * Q[row * r + colK];

	float sum = blockReduceSum(localSum, smem);
	if (threadIdx.x == 0)
		d_dots[target] = sum;
}

// Phase 2: normalize column j, subtract projections from columns j+1..r-1.
// Each thread handles one row.  d_dots[0] = ||Q[:,j]||^2,
// d_dots[k] = Q[:,j]·Q[:,j+k] (pre-normalization).
__global__ void atlas_gs_update_kernel(float* __restrict__ Q,
                                        int m, int r, int j,
                                        const float* __restrict__ d_dots,
                                        int numRemaining)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= m) return;

	float invNorm = (d_dots[0] > 0.0f) ? rsqrtf(d_dots[0]) : 0.0f;
	float qj = Q[row * r + j] * invNorm;
	Q[row * r + j] = qj;

	int base = row * r + j;
	for (int k = 1; k < numRemaining; ++k)
		Q[base + k] -= (d_dots[k] * invNorm) * qj;
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
	int smemBytes = ((blockSize / 32) + 1) * sizeof(float);
	atlas_gs_kernel<<<1, blockSize, smemBytes>>>(d_Q, m, r);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Fast multi-block Modified Gram-Schmidt using full GPU parallelism.
// d_dots must be a device buffer of at least r floats (scratch for dot products).
static bool atlas_gpu_gram_schmidt_fast(float* d_Q, int m, int r, float* d_dots)
{
	if (r <= 0 || m <= 0) return true;
	int blockSize = 256;
	if (blockSize > m) blockSize = ((m + 31) / 32) * 32;
	if (blockSize < 32) blockSize = 32;
	int smemBytes = ((blockSize / 32) + 1) * sizeof(float);
	int gridUpdate = (m + blockSize - 1) / blockSize;

	for (int j = 0; j < r; ++j)
	{
		int numRemaining = r - j;

		// Phase 1: compute norm of col j and dot products with cols j+1..r-1
		atlas_gs_dots_kernel<<<numRemaining, blockSize, smemBytes>>>(
			d_Q, m, r, j, d_dots);

		// Phase 2: normalize col j and subtract projections
		atlas_gs_update_kernel<<<gridUpdate, blockSize>>>(
			d_Q, m, r, j, d_dots, numRemaining);
	}
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Single-pass Cholesky QR helper. If `regularize` is true, adds diagonal
// shift for float32 stability on ill-conditioned inputs. Returns false on error.
// d_scratch_rr: device buffer [r*r], d_qrTemp: device buffer [m*r].
static bool cholesky_qr_pass(float* d_Q, int m, int r,
                              float* d_scratch_rr, float* d_qrTemp,
                              bool regularize)
{
	// 1. Gram matrix: G[r,r] = Q^T Q
	if (!sgemm_rowmajor_atb(r, r, m, 1.0f, d_Q, r, d_Q, r,
	                         0.0f, d_scratch_rr, r))
		return false;

	// 2. Download G to host
	std::vector<float> R(r * r);
	ATLAS_CUDA_CHECK(cudaMemcpy(R.data(), d_scratch_rr,
	                            (size_t)r * r * sizeof(float),
	                            cudaMemcpyDeviceToHost));

	// 3. Column-norm pre-scaling: transform G into a correlation matrix.
	std::vector<float> colScale(r);
	float maxDiag = 0.0f;
	for (int j = 0; j < r; ++j)
		if (R[j * r + j] > maxDiag) maxDiag = R[j * r + j];
	float diagFloor = 1e-12f * (maxDiag > 0.0f ? maxDiag : 1.0f);
	for (int j = 0; j < r; ++j)
	{
		float d = R[j * r + j] > diagFloor ? R[j * r + j] : diagFloor;
		colScale[j] = sqrtf(d);
	}
	for (int i = 0; i < r; ++i)
		for (int j = 0; j < r; ++j)
			R[i * r + j] /= (colScale[i] * colScale[j]);

	// 4. Regularize if requested (first pass on ill-conditioned input)
	if (regularize)
		for (int j = 0; j < r; ++j)
			R[j * r + j] += 1e-4f;

	// 5. Cholesky: G_s = L^T L (L upper triangular, row-major)
	for (int j = 0; j < r; ++j)
	{
		float d = R[j * r + j];
		for (int k = 0; k < j; ++k)
			d -= R[k * r + j] * R[k * r + j];
		if (d < 1e-8f) d = 1e-8f;
		R[j * r + j] = sqrtf(d);
		float invD = 1.0f / R[j * r + j];
		for (int i = j + 1; i < r; ++i)
		{
			float v = R[j * r + i];
			for (int k = 0; k < j; ++k)
				v -= R[k * r + j] * R[k * r + i];
			R[j * r + i] = v * invD;
		}
	}

	// Zero lower triangle
	for (int i = 0; i < r; ++i)
		for (int j = 0; j < i; ++j)
			R[i * r + j] = 0.0f;

	// 6. Un-scale: R_true = L * diag(colScale)
	for (int i = 0; i < r; ++i)
		for (int j = i; j < r; ++j)
			R[i * r + j] *= colScale[j];

	// 7. Invert R on CPU via back-substitution (upper triangular)
	std::vector<float> Rinv(r * r, 0.0f);
	for (int j = r - 1; j >= 0; --j)
	{
		Rinv[j * r + j] = 1.0f / R[j * r + j];
		for (int i = j - 1; i >= 0; --i)
		{
			float s = 0.0f;
			for (int k = i + 1; k <= j; ++k)
				s += R[i * r + k] * Rinv[k * r + j];
			Rinv[i * r + j] = -s / R[i * r + i];
		}
	}

	// 8. Upload R_inv to device
	ATLAS_CUDA_CHECK(cudaMemcpy(d_scratch_rr, Rinv.data(),
	                            (size_t)r * r * sizeof(float),
	                            cudaMemcpyHostToDevice));

	// 9. Q_new = Q * R_inv via cuBLAS SGEMM
	if (!sgemm_rowmajor(m, r, r, 1.0f, d_Q, r, d_scratch_rr, r,
	                     0.0f, d_qrTemp, r))
		return false;

	// 10. Copy result back to Q
	ATLAS_CUDA_CHECK(cudaMemcpy(d_Q, d_qrTemp,
	                            (size_t)m * r * sizeof(float),
	                            cudaMemcpyDeviceToDevice));

	return true;
}

// CholeskyQR² — two-pass orthonormalization for machine-precision results.
// Pass 1 (regularized): handles ill-conditioned input from power iteration,
//   gets columns to approximately unit norm.
// Pass 2 (exact): Gram matrix is now well-conditioned (diag ≈ 1), so Cholesky
//   succeeds without regularization, producing exact orthonormality.
// d_scratch_rr must be a device buffer of at least r*r floats.
static bool cholesky_qr(float* d_Q, int m, int r, float* d_scratch_rr)
{
	if (r <= 0 || m <= 0) return true;

	// Pass 1: regularized — handles ill-conditioned input
	if (!cholesky_qr_pass(d_Q, m, r, d_scratch_rr, s_qrTemp.data(), true))
		return false;

	// Pass 2: exact — cleans up regularization artifacts
	if (!cholesky_qr_pass(d_Q, m, r, d_scratch_rr, s_qrTemp.data(), false))
		return false;

	return true;
}

bool atlas_gpu_weight_decay(float* d_W, int mn, float lr, float wd1, float wd2)
{
	if (mn <= 0) return true;
	if (wd1 == 0.0f && wd2 == 0.0f) return true;
	int grid = (mn + kBlock - 1) / kBlock;
	atlas_wd_kernel<<<grid, kBlock>>>(d_W, mn, lr, wd1, wd2);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_baseline_update(float* d_W, const float* d_gW, int mn, float baseScaled)
{
	if (mn <= 0) return true;
	int grid = (mn + kBlock - 1) / kBlock;
	atlas_baseline_kernel<<<grid, kBlock>>>(d_W, d_gW, mn, baseScaled);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_fisher_update(const float* d_gz, float* d_fisherDiag,
                             int r, int n, float beta)
{
	if (r <= 0 || n <= 0) return true;
	int blockSize = 256;
	if (blockSize > n) blockSize = ((n + 31) / 32) * 32;
	if (blockSize < 32) blockSize = 32;
	int smemBytes = ((blockSize / 32) + 1) * sizeof(float);
	atlas_fisher_kernel<<<r, blockSize, smemBytes>>>(
		d_gz, d_fisherDiag, r, n, beta, 1.0f - beta);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_prepare_correction(const float* d_gz, const float* d_prevGz,
                                   const float* d_fisherDiag,
                                   float* d_out, int r, int n,
                                   float onePlusMu, float negMu,
                                   float baselineRate, float lr, float eps,
                                   float kappaLr)
{
	if (r <= 0 || n <= 0) return true;
	dim3 block(kBlock);
	dim3 grid((n + kBlock - 1) / kBlock, r);
	atlas_correction_kernel<<<grid, block>>>(
		d_gz, d_prevGz, d_fisherDiag, d_out, r, n,
		onePlusMu, negMu, baselineRate, lr, eps, kappaLr);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_mu_norms(const float* d_gz, const float* d_prevGz,
                         int rn, float* d_out)
{
	if (rn <= 0) return true;
	ATLAS_CUDA_CHECK(cudaMemsetAsync(d_out, 0, 2 * sizeof(float)));
	int grid = (rn + kBlock - 1) / kBlock;
	if (grid > 256) grid = 256;
	int smemBytes = 2 * ((kBlock / 32) + 1) * sizeof(float);
	atlas_mu_norms_kernel<<<grid, kBlock, smemBytes>>>(d_gz, d_prevGz, rn, d_out);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_ema_blend(float* d_dst, const float* d_a, const float* d_b,
                          int count, float beta)
{
	if (count <= 0) return true;
	int grid = (count + kBlock - 1) / kBlock;
	atlas_ema_kernel<<<grid, kBlock>>>(d_dst, d_a, d_b, count, 1.0f - beta, beta);
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
	atlas_transform_fisher_kernel<<<1, blockSize>>>(d_overlap, d_f_old, d_f_new, r);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool atlas_gpu_scale_grad(const float* d_gW, float* d_out, int mn, float gScale)
{
	if (mn <= 0) return true;
	int grid = (mn + kBlock - 1) / kBlock;
	// Reuse the baseline kernel: out[i] = gW[i] * gScale is the same as
	// dst[i] -= (-gScale) * src[i] starting from dst=0, but just use a
	// dedicated approach via the sigma2 pattern. Actually simpler to just
	// reuse atlas_baseline_kernel with a trick: zero out, then subtract.
	// Cleaner: write a trivial scale kernel inline.
	// We already declared atlas_scale_kernel but removed it. Use baseline:
	// First zero out, then out[i] -= (-gScale)*gW[i] = gScale*gW[i].
	ATLAS_CUDA_CHECK(cudaMemsetAsync(d_out, 0, (size_t)mn * sizeof(float)));
	atlas_baseline_kernel<<<grid, kBlock>>>(d_out, d_gW, mn, -gScale);
	ATLAS_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  atlas_gpu_init — allocate and initialize core state buffers
// ===========================================================================

bool atlas_gpu_init(GpuAtlasWeightState& state,
                    unsigned int m, unsigned int n,
                    unsigned int rank, float muInit)
{
	state.m = m;
	state.n = n;
	state.r = rank;
	if (state.r > m) state.r = m;
	if (state.r > n) state.r = n;
	if (state.r == 0u) state.r = 1u;

	const unsigned int r = state.r;
	const size_t mr = (size_t)m * r;
	const size_t rn = (size_t)r * n;

	// Allocate persistent buffers
	if (!state.U.allocate(mr)) return false;
	if (!state.fisherDiag.allocate(r)) return false;
	if (!state.prevGz.allocate(rn)) return false;

	// Allocate per-step scratch buffers
	if (!state.gz.allocate(rn)) return false;
	if (!state.gPred.allocate(rn)) return false;
	if (!state.d_reduce.allocate(2)) return false;

	// Pre-allocate refresh scratch buffers (avoids memory spike at first refresh)
	if (!state.U_old.allocate(mr)) return false;
	if (!state.f_old.allocate(r)) return false;
	if (!state.B.allocate((size_t)n * r)) return false;
	if (!state.overlap.allocate((size_t)r * r)) return false;
	if (!state.prevGzOld.allocate(rn)) return false;
	state.refreshAllocated = true;

	// Initialize U with random Gaussian values on CPU, then upload
	{
		std::vector<float> h_U(mr);
		const float scale = 1.0f / sqrtf((float)m);

		// Simple deterministic PRNG (xorshift64) seeded per weight matrix
		unsigned long long seed = 42ULL + (unsigned long long)m * 1000003ULL
		                        + (unsigned long long)n * 999983ULL;
		for (size_t i = 0; i < mr; ++i)
		{
			// xorshift64
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			// Box-Muller (pairs) for approximate normal
			float u = ((float)(seed & 0xFFFFFFFF) + 1.0f) / 4294967297.0f;
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			float v = ((float)(seed & 0xFFFFFFFF) + 1.0f) / 4294967297.0f;
			h_U[i] = scale * sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v);
		}

		if (!state.U.upload(h_U.data(), mr)) return false;
	}

	// Orthogonalize U on device
	if (!atlas_gpu_gram_schmidt(state.U.data(), (int)m, (int)r))
		return false;

	// Initialize Fisher diagonal to 1.0
	{
		std::vector<float> ones(r, 1.0f);
		if (!state.fisherDiag.upload(ones.data(), r)) return false;
	}

	// Zero prevGz
	if (!state.prevGz.zero()) return false;

	// Pre-allocate shared Cholesky QR temp buffer (grows to max m*r across all weights)
	if (s_qrTemp.size() < mr)
	{
		s_qrTemp.free();
		if (!s_qrTemp.allocate(mr)) return false;
	}

	state.sigma2 = 1.0f;
	state.mu = muInit;
	state.step = 0ULL;
	state.initialized = true;

	ATLAS_CUDA_CHECK(cudaDeviceSynchronize());
	fprintf(stderr, "[atlas-gpu] init m=%u n=%u r=%u (%.1f MB)\n",
	        m, n, r, (float)(mr + rn * 3 + r + 2) * 4.0f / (1024.0f * 1024.0f));
	return true;
}

// ===========================================================================
//  ensureRefreshBuffers — lazy allocation of subspace refresh scratch
// ===========================================================================

static bool ensureRefreshBuffers(GpuAtlasWeightState& state)
{
	if (state.refreshAllocated) return true;

	const unsigned int m = state.m;
	const unsigned int n = state.n;
	const unsigned int r = state.r;
	const size_t mr = (size_t)m * r;
	const size_t rn = (size_t)r * n;

	if (!state.U_old.allocate(mr)) return false;
	if (!state.f_old.allocate(r)) return false;
	if (!state.B.allocate((size_t)n * r)) return false;
	if (!state.overlap.allocate((size_t)r * r)) return false;
	if (!state.prevGzOld.allocate(rn)) return false;

	state.refreshAllocated = true;
	fprintf(stderr, "[atlas-gpu] refresh buffers allocated m=%u n=%u r=%u (%.1f MB)\n",
	        m, n, r, (float)(mr + r + (size_t)n * r + (size_t)r * r + rn) * 4.0f / (1024.0f * 1024.0f));
	return true;
}

// ===========================================================================
//  refreshSubspace — periodic subspace update via randomized power iteration
// ===========================================================================

// d_grad is the raw gradient [m*n], gScale folds into SGEMM alpha.
// Eigenvectors of G*G^T are scale-invariant, so we pass alpha=1 (no scaling).
static bool refreshSubspace(GpuAtlasWeightState& state,
                            const float* d_grad,
                            unsigned int m, unsigned int n,
                            unsigned int powerIters, float betaRefresh)
{
	const unsigned int r = state.r;
	if (r == 0u || m == 0u || n == 0u) return true;

	const size_t mr = (size_t)m * r;
	const size_t rn = (size_t)r * n;

	// Save old basis and Fisher
	ATLAS_CUDA_CHECK(cudaMemcpy(state.U_old.data(), state.U.data(),
	                            mr * sizeof(float), cudaMemcpyDeviceToDevice));
	ATLAS_CUDA_CHECK(cudaMemcpy(state.f_old.data(), state.fisherDiag.data(),
	                            r * sizeof(float), cudaMemcpyDeviceToDevice));

	// --- Randomized power iteration (warm-started from current U) ---
	float* d_Q = state.U.data();

	for (unsigned int p = 0; p < powerIters; ++p)
	{
		// B[n,r] = grad^T[n,m] * Q[m,r]
		// sgemm_rowmajor_atb: C[M,N] = alpha * A^T[M,K] * B[K,N]
		// A=grad[m,n] (stored [K,M] with K=m,M=n), B=Q[m,r] (stored [K,N] with N=r), C=B[n,r]
		if (!sgemm_rowmajor_atb((int)n, (int)r, (int)m,
		                         1.0f,
		                         d_grad, (int)n,
		                         d_Q, (int)r,
		                         0.0f,
		                         state.B.data(), (int)r))
			return false;

		// Z[m,r] = grad[m,n] * B[n,r]
		// sgemm_rowmajor: C[M,N] = alpha * A[M,K] * B[K,N]
		if (!sgemm_rowmajor((int)m, (int)r, (int)n,
		                     1.0f,
		                     d_grad, (int)n,
		                     state.B.data(), (int)r,
		                     0.0f,
		                     d_Q, (int)r))
			return false;

		if (!atlas_gpu_gram_schmidt_fast(d_Q, (int)m, (int)r, state.overlap.data()))
			return false;
	}

	// --- EMA blend: U = (1-betaRefresh)*U_old + betaRefresh*U_new ---
	// dst=d_Q aliases b=d_Q. Safe because each thread reads/writes only its own idx.
	if (!atlas_gpu_ema_blend(d_Q, state.U_old.data(), d_Q, (int)mr, betaRefresh))
		return false;

	// Re-orthogonalize after blending
	if (!atlas_gpu_gram_schmidt_fast(d_Q, (int)m, (int)r, state.overlap.data()))
		return false;

	// --- Compute overlap matrix O[r,r] = U_final^T * U_old ---
	if (!sgemm_rowmajor_atb((int)r, (int)r, (int)m,
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
	ATLAS_CUDA_CHECK(cudaMemcpy(state.prevGzOld.data(), state.prevGz.data(),
	                            rn * sizeof(float), cudaMemcpyDeviceToDevice));

	// prevGz_new[r,n] = overlap[r,r] * prevGzOld[r,n]
	if (!sgemm_rowmajor((int)r, (int)n, (int)r,
	                     1.0f,
	                     state.overlap.data(), (int)r,
	                     state.prevGzOld.data(), (int)n,
	                     0.0f,
	                     state.prevGz.data(), (int)n))
		return false;

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
                    float beta, float muMin, float muMax,
                    float eps, unsigned int tSub,
                    unsigned int powerIters, float betaRefresh,
                    float kappaMax)
{
	if (!state.initialized) return false;

	const unsigned int r = state.r;
	const int mn = (int)((size_t)m * n);
	const int rn = (int)((size_t)r * n);
	state.step += 1ULL;

	const float gScale = invBatch * gradScale;

	// === Step 1: Update global second moment sigma2 ===
	{
		ATLAS_CUDA_CHECK(cudaMemsetAsync(state.d_reduce.data(), 0, sizeof(float)));
		int grid = (mn + kBlock - 1) / kBlock;
		if (grid > 256) grid = 256;
		int smemBytes = ((kBlock / 32) + 1) * sizeof(float);
		atlas_sigma2_kernel<<<grid, kBlock, smemBytes>>>(d_gW, mn, gScale,
		                                                  state.d_reduce.data());
		ATLAS_CUDA_CHECK(cudaGetLastError());

		float h_sumSq = 0.0f;
		ATLAS_CUDA_CHECK(cudaMemcpy(&h_sumSq, state.d_reduce.data(),
		                            sizeof(float), cudaMemcpyDeviceToHost));
		float gMeanSq = h_sumSq / (float)mn;
		state.sigma2 = beta * state.sigma2 + (1.0f - beta) * gMeanSq;
	}

	// === Step 2: Periodic subspace refresh ===
	// Pass raw d_gW directly — eigenvectors are scale-invariant.
	if (tSub > 0u && (state.step % (unsigned long long)tSub) == 0ULL)
	{
		if (!refreshSubspace(state, d_gW, m, n, powerIters, betaRefresh))
			return false;
	}

	// === Step 3: Decoupled weight decay ===
	if (!atlas_gpu_weight_decay(d_W, mn, lr, wd1, wd2))
		return false;

	// === Step 4: Project gradient to subspace ===
	// gz[r,n] = gScale * U^T[r,m] * gW[m,n]
	if (!sgemm_rowmajor_atb((int)r, (int)n, (int)m,
	                         gScale,
	                         state.U.data(), (int)r,
	                         d_gW, (int)n,
	                         0.0f,
	                         state.gz.data(), (int)n))
		return false;

	// === Step 5: Update Fisher diagonal ===
	if (!atlas_gpu_fisher_update(state.gz.data(), state.fisherDiag.data(),
	                             (int)r, (int)n, beta))
		return false;

	// === Step 6: Full-space baseline update ===
	{
		const float baselineRate = fminf(lr / (state.sigma2 + eps), kappaMax * lr);
		const float baseScaled = baselineRate * gScale;
		if (!atlas_gpu_baseline_update(d_W, d_gW, mn, baseScaled))
			return false;

		// === Step 7: Subspace correction with PNG ===
		const float onePlusMu = 1.0f + state.mu;
		const float negMu = -state.mu;

		if (!atlas_gpu_prepare_correction(state.gz.data(), state.prevGz.data(),
		                                   state.fisherDiag.data(),
		                                   state.gPred.data(), (int)r, (int)n,
		                                   onePlusMu, negMu,
		                                   baselineRate, lr, eps, kappaMax * lr))
			return false;

		// W[m,n] += U[m,r] * gPred[r,n]
		if (!sgemm_rowmajor((int)m, (int)n, (int)r,
		                     1.0f,
		                     state.U.data(), (int)r,
		                     state.gPred.data(), (int)n,
		                     1.0f,
		                     d_W, (int)n))
			return false;
	}

	// === Step 8: Adapt prediction coefficient mu ===
	if (state.step > 1ULL && muMax > 0.0f)
	{
		if (!atlas_gpu_mu_norms(state.gz.data(), state.prevGz.data(),
		                         rn, state.d_reduce.data()))
			return false;

		float h_norms[2] = {0.0f, 0.0f};
		ATLAS_CUDA_CHECK(cudaMemcpy(h_norms, state.d_reduce.data(),
		                            2 * sizeof(float), cudaMemcpyDeviceToHost));

		float errNormSq = h_norms[0];
		float gzNormSq  = h_norms[1];
		float gzNorm = sqrtf(gzNormSq);
		if (gzNorm > 1e-12f)
		{
			float ratio = sqrtf(errNormSq) / (gzNorm + 1e-12f);
			float newMu = state.mu * (1.0f - ratio);
			if (newMu < muMin) newMu = muMin;
			if (newMu > muMax) newMu = muMax;
			state.mu = newMu;
		}
	}

	// === Step 9: Store compressed gradient for next step ===
	ATLAS_CUDA_CHECK(cudaMemcpy(state.prevGz.data(), state.gz.data(),
	                            (size_t)rn * sizeof(float), cudaMemcpyDeviceToDevice));

	// === Step 10: Clear accumulated gradients ===
	ATLAS_CUDA_CHECK(cudaMemsetAsync(d_gW, 0, (size_t)mn * sizeof(float)));

	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
