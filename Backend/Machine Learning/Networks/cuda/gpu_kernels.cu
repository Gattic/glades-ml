// Custom CUDA kernel implementations for Glades ML.
//
// Contains normalization, activation, attention, embedding, optimizer, and
// utility kernels together with thin host-side wrapper functions that handle
// grid/block configuration.
//
// Requirements: CUDA 11+, SM 6.0+.

#include "gpu_kernels.h"
#include "gpu_device.h"
#include "gpu_blas.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cfloat>
#include <cmath>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace glades {
namespace gpu {

namespace {

// Convenience: check a CUDA call, print on error, return false.
#define GLADES_CUDA_CHECK(call)                                               \
	do {                                                                      \
		cudaError_t err_ = (call);                                            \
		if (err_ != cudaSuccess) {                                            \
			fprintf(stderr, "[glades-cuda] %s:%d  %s  -> %s\n",              \
			        __FILE__, __LINE__, #call, cudaGetErrorString(err_));     \
			return false;                                                     \
		}                                                                     \
	} while (0)

// Block size for simple element-wise kernels.
static constexpr int kBlockElem = 256;

// Maximum block size for per-row kernels.
static constexpr int kMaxBlockRow = 1024;

// Choose block size for per-row kernels: min(cols, kMaxBlockRow) rounded up to
// the nearest warp (32).
inline int rowBlockSize(int cols)
{
	int b = cols < kMaxBlockRow ? cols : kMaxBlockRow;
	b = ((b + 31) / 32) * 32;
	if (b < 32) b = 32;
	if (b > kMaxBlockRow) b = kMaxBlockRow;
	return b;
}

} // anonymous namespace

// ===========================================================================
//  Warp-level primitives
// ===========================================================================

namespace {

__device__ __forceinline__ float warpReduceSum(float val)
{
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}

__device__ __forceinline__ float warpReduceMax(float val)
{
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
		val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
	return val;
}

// Block-wide sum reduction using shared memory.  Caller must provide
// smem of size >= (blockDim.x / 32) floats.
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

// Block-wide max reduction using shared memory.
__device__ float blockReduceMax(float val, float* smem)
{
	int lane = threadIdx.x & 31;
	int wid  = threadIdx.x >> 5;

	val = warpReduceMax(val);
	if (lane == 0) smem[wid] = val;
	__syncthreads();

	int numWarps = (blockDim.x + 31) / 32;
	val = (threadIdx.x < (unsigned)numWarps) ? smem[threadIdx.x] : -FLT_MAX;
	if (wid == 0) val = warpReduceMax(val);
	return val;
}

} // anonymous namespace

// ===========================================================================
//  1. Layer normalization forward
// ===========================================================================

namespace {

__global__ void layernorm_forward_rows(const float* __restrict__ x,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float eps, int cols,
                                       float* __restrict__ out,
                                       float* __restrict__ meanOut,
                                       float* __restrict__ invStdOut)
{
	// One block per row.
	int row = blockIdx.x;
	const float* xRow = x + (size_t)row * cols;
	float* oRow       = out + (size_t)row * cols;

	extern __shared__ float smem[];  // at least (blockDim.x / 32) * 2 floats
	float* sSum  = smem;
	float* sSum2 = smem + (blockDim.x / 32 + 1);

	// Pass 1: compute mean.
	float localSum = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		localSum += xRow[i];
	localSum = blockReduceSum(localSum, sSum);

	__shared__ float sMean;
	__shared__ float sInvStd;
	if (threadIdx.x == 0)
		sMean = localSum / (float)cols;
	__syncthreads();

	float mu = sMean;

	// Pass 2: variance.
	float localVar = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float d = xRow[i] - mu;
		localVar += d * d;
	}
	localVar = blockReduceSum(localVar, sSum2);

	if (threadIdx.x == 0) {
		float var = localVar / (float)cols;
		sInvStd = rsqrtf(var + eps);
	}
	__syncthreads();

	float inv = sInvStd;

	// Write side-outputs.
	if (threadIdx.x == 0) {
		meanOut[row]   = mu;
		invStdOut[row] = inv;
	}

	// Pass 3: normalize.
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		oRow[i] = gamma[i] * (xRow[i] - mu) * inv + beta[i];
}

} // anonymous namespace

bool layernorm_forward(const float* x, const float* gamma, const float* beta,
                       float eps, int rows, int cols,
                       float* out, float* mean, float* invStd)
{
	if (rows <= 0 || cols <= 0) return true;
	int block = rowBlockSize(cols);
	int smemBytes = (block / 32 + 2) * 2 * sizeof(float);
	layernorm_forward_rows<<<rows, block, smemBytes, computeStream()>>>(
		x, gamma, beta, eps, cols, out, mean, invStd);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  2. Layer normalization backward
// ===========================================================================

namespace {

// dx kernel: one block per row, no atomics.
__global__ void layernorm_backward_dx(const float* __restrict__ dout,
                                      const float* __restrict__ x,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ mean,
                                      const float* __restrict__ invStd,
                                      int cols,
                                      float* __restrict__ dx)
{
	int row = blockIdx.x;
	const float* dRow = dout + (size_t)row * cols;
	const float* xRow = x    + (size_t)row * cols;
	float* dxRow      = dx   + (size_t)row * cols;

	float mu  = mean[row];
	float inv = invStd[row];

	extern __shared__ float smem[];
	float* s1 = smem;
	float* s2 = smem + (blockDim.x / 32 + 1);

	float localDs = 0.0f;
	float localDb = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float xhat = (xRow[i] - mu) * inv;
		float dg   = dRow[i] * gamma[i];
		localDs += dg * xhat;
		localDb += dg;
	}

	localDs = blockReduceSum(localDs, s1);
	__syncthreads();
	localDb = blockReduceSum(localDb, s2);

	__shared__ float sDs, sDb;
	if (threadIdx.x == 0) {
		sDs = localDs;
		sDb = localDb;
	}
	__syncthreads();

	float ds = sDs;
	float db = sDb;
	float invN = 1.0f / (float)cols;

	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float xhat = (xRow[i] - mu) * inv;
		float dg   = dRow[i] * gamma[i];
		dxRow[i] = inv * (dg - invN * (db + xhat * ds));
	}
}

// Legacy dgamma/dbeta kernel: one block per column, threads reduce across rows.
// Kept as dead code; iter 53/60 ships the 2-phase variant below.
__global__ void layernorm_backward_dgamma_dbeta(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ invStd,
    int rows, int cols,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta)
{
	int col = blockIdx.x;
	if (col >= cols) return;

	extern __shared__ float smem[];
	float* s1 = smem;
	float* s2 = smem + (blockDim.x / 32 + 1);

	float localDg = 0.0f;
	float localDb = 0.0f;
	for (int r = threadIdx.x; r < rows; r += blockDim.x) {
		float mu  = mean[r];
		float inv = invStd[r];
		float xhat = (x[(size_t)r * cols + col] - mu) * inv;
		float d = dout[(size_t)r * cols + col];
		localDg += d * xhat;
		localDb += d;
	}

	localDg = blockReduceSum(localDg, s1);
	__syncthreads();
	localDb = blockReduceSum(localDb, s2);

	if (threadIdx.x == 0) {
		dgamma[col] += localDg;
		dbeta[col]  += localDb;
	}
}

// Iter 53 / Iter 60: deterministic 2-phase parallel reduction for dgamma_dbeta.
// Phase 1 (this kernel): 2D-tiled coalesced access (BLOCK_C cols × BLOCK_T row-
// partitions reduced via SMEM).  T_PARTS blocks per col-tile produce partial
// sums in scratch[T_PARTS, cols] — no atomics needed (each (col, t_partition)
// is uniquely owned by one block).
template<int BLOCK_C, int BLOCK_T>
__global__ void layernorm_backward_dgamma_dbeta_partial(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ invStd,
    int rows, int cols,
    int rowsPerBlock,
    float* __restrict__ partial_dg,
    float* __restrict__ partial_db)
{
	const int col            = blockIdx.x * BLOCK_C + threadIdx.x;
	const int rowStart       = blockIdx.y * rowsPerBlock;
	const int rowEnd         = rowStart + rowsPerBlock;
	const int rowEndClamped  = (rowEnd > rows) ? rows : rowEnd;
	const int ty             = threadIdx.y;

	float dgAcc = 0.0f;
	float dbAcc = 0.0f;
	if (col < cols)
	{
		for (int r = rowStart + ty; r < rowEndClamped; r += BLOCK_T)
		{
			float mu   = mean[r];
			float inv  = invStd[r];
			float xhat = (x[(size_t)r * cols + col] - mu) * inv;
			float d    = dout[(size_t)r * cols + col];
			dgAcc += d * xhat;
			dbAcc += d;
		}
	}

	__shared__ float sDg[BLOCK_T][BLOCK_C];
	__shared__ float sDb[BLOCK_T][BLOCK_C];
	sDg[ty][threadIdx.x] = dgAcc;
	sDb[ty][threadIdx.x] = dbAcc;
	__syncthreads();

	for (int s = BLOCK_T / 2; s > 0; s >>= 1)
	{
		if (ty < s)
		{
			sDg[ty][threadIdx.x] += sDg[ty + s][threadIdx.x];
			sDb[ty][threadIdx.x] += sDb[ty + s][threadIdx.x];
		}
		__syncthreads();
	}

	if (ty == 0 && col < cols)
	{
		const size_t base = (size_t)blockIdx.y * (size_t)cols + (size_t)col;
		partial_dg[base] = sDg[0][threadIdx.x];
		partial_db[base] = sDb[0][threadIdx.x];
	}
}

// Phase 2: deterministic per-col reduce of T_PARTS partials in fixed loop
// order.  One thread per col; T_PARTS is small (≤8) so loop is cheap.
__global__ void layernorm_backward_dgamma_dbeta_reduce(
    const float* __restrict__ partial_dg,
    const float* __restrict__ partial_db,
    int t_parts, int cols,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta)
{
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= cols) return;
	float sumG = 0.0f;
	float sumB = 0.0f;
	for (int p = 0; p < t_parts; ++p)
	{
		sumG += partial_dg[(size_t)p * (size_t)cols + (size_t)col];
		sumB += partial_db[(size_t)p * (size_t)cols + (size_t)col];
	}
	dgamma[col] += sumG;
	dbeta[col]  += sumB;
}

} // anonymous namespace

// Iter 53/60 lazy scratch pool for the 2-phase dgamma_dbeta path.  Sized for
// the largest LN backward call we'll see (T_PARTS × cols floats × 2 for dg/db).
// Allocated once; never freed (small).
namespace {
static float* s_ln_partial_dg = 0;
static float* s_ln_partial_db = 0;
static int    s_ln_partial_cols_max = 0;
static int    s_ln_partial_t_parts_max = 0;

static bool ensure_ln_partial_scratch(int t_parts, int cols)
{
	if (t_parts <= s_ln_partial_t_parts_max && cols <= s_ln_partial_cols_max)
		return true;
	if (s_ln_partial_dg) { cudaFree(s_ln_partial_dg); s_ln_partial_dg = 0; }
	if (s_ln_partial_db) { cudaFree(s_ln_partial_db); s_ln_partial_db = 0; }
	const size_t bytes = (size_t)t_parts * (size_t)cols * sizeof(float);
	cudaError_t e = cudaMalloc(&s_ln_partial_dg, bytes);
	if (e != cudaSuccess) return false;
	e = cudaMalloc(&s_ln_partial_db, bytes);
	if (e != cudaSuccess) return false;
	s_ln_partial_cols_max = cols;
	s_ln_partial_t_parts_max = t_parts;
	return true;
}
} // anonymous namespace

bool layernorm_backward(const float* dout, const float* x,
                        const float* gamma, const float* mean,
                        const float* invStd, int rows, int cols,
                        float* dx, float* dgamma, float* dbeta)
{
	if (rows <= 0 || cols <= 0) return true;

	// Kernel 1: dx (one block per row).
	int block1 = rowBlockSize(cols);
	int smemBytes1 = (block1 / 32 + 2) * 2 * sizeof(float);
	layernorm_backward_dx<<<rows, block1, smemBytes1, computeStream()>>>(
		dout, x, gamma, mean, invStd, cols, dx);
	GLADES_CUDA_CHECK(cudaGetLastError());

	// Kernel 2: dgamma/dbeta — Iter 53/60 deterministic 2-phase parallel
	// reduction.  Replaces the legacy "1 block per col, threads loop strided
	// rows" kernel.  Coalesced access + parallel reduction + deterministic
	// per-col reduce over T_PARTS partials.
	{
		const int BLOCK_C = 64;
		const int BLOCK_T = 8;
		const int T_PARTS = 4;
		if (!ensure_ln_partial_scratch(T_PARTS, cols)) return false;
		int rowsPerBlock = (rows + T_PARTS - 1) / T_PARTS;
		// Phase 1.
		{
			dim3 grid((cols + BLOCK_C - 1) / BLOCK_C, T_PARTS);
			dim3 block(BLOCK_C, BLOCK_T);
			layernorm_backward_dgamma_dbeta_partial<64, 8>
			    <<<grid, block, 0, computeStream()>>>(
			        dout, x, mean, invStd, rows, cols, rowsPerBlock,
			        s_ln_partial_dg, s_ln_partial_db);
			GLADES_CUDA_CHECK(cudaGetLastError());
		}
		// Phase 2: deterministic per-col reduce.
		{
			const int RBLOCK = 256;
			int rgrid = (cols + RBLOCK - 1) / RBLOCK;
			layernorm_backward_dgamma_dbeta_reduce<<<rgrid, RBLOCK, 0, computeStream()>>>(
			    s_ln_partial_dg, s_ln_partial_db, T_PARTS, cols, dgamma, dbeta);
			GLADES_CUDA_CHECK(cudaGetLastError());
		}
	}

	return true;
}

// ===========================================================================
//  3. RMSNorm forward
// ===========================================================================

namespace {

__global__ void rmsnorm_forward_rows(const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     float eps, int cols,
                                     float* __restrict__ out,
                                     float* __restrict__ invRmsOut)
{
	int row = blockIdx.x;
	const float* xRow = x + (size_t)row * cols;
	float* oRow       = out + (size_t)row * cols;

	extern __shared__ float smem[];

	float localSS = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		localSS += xRow[i] * xRow[i];
	localSS = blockReduceSum(localSS, smem);

	__shared__ float sInvRms;
	if (threadIdx.x == 0) {
		float rms = localSS / (float)cols;
		sInvRms = rsqrtf(rms + eps);
		invRmsOut[row] = sInvRms;
	}
	__syncthreads();

	float inv = sInvRms;
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		oRow[i] = gamma[i] * xRow[i] * inv;
}

} // anonymous namespace

bool rmsnorm_forward(const float* x, const float* gamma, float eps,
                     int rows, int cols, float* out, float* invRms)
{
	if (rows <= 0 || cols <= 0) return true;
	int block = rowBlockSize(cols);
	int smemBytes = (block / 32 + 1) * sizeof(float);
	rmsnorm_forward_rows<<<rows, block, smemBytes, computeStream()>>>(
		x, gamma, eps, cols, out, invRms);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  4. RMSNorm backward
// ===========================================================================

namespace {

// dx kernel: one block per row, no atomics.
__global__ void rmsnorm_backward_dx(const float* __restrict__ dout,
                                    const float* __restrict__ x,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ invRms,
                                    int cols,
                                    float* __restrict__ dx)
{
	int row = blockIdx.x;
	const float* dRow = dout + (size_t)row * cols;
	const float* xRow = x    + (size_t)row * cols;
	float* dxRow      = dx   + (size_t)row * cols;
	float inv = invRms[row];

	extern __shared__ float smem[];

	float localDot = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		localDot += dRow[i] * gamma[i] * xRow[i];
	localDot = blockReduceSum(localDot, smem);

	__shared__ float sDot;
	if (threadIdx.x == 0)
		sDot = localDot;
	__syncthreads();

	float dot  = sDot;
	float invN = 1.0f / (float)cols;
	float inv3 = inv * inv * inv;

	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		dxRow[i] = inv * gamma[i] * dRow[i] - xRow[i] * inv3 * dot * invN;
}

// dgamma kernel: one block per column, threads reduce across rows.
__global__ void rmsnorm_backward_dgamma(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ invRms,
    int rows, int cols,
    float* __restrict__ dgamma)
{
	int col = blockIdx.x;
	if (col >= cols) return;

	extern __shared__ float smem[];

	float localDg = 0.0f;
	for (int r = threadIdx.x; r < rows; r += blockDim.x) {
		float inv = invRms[r];
		localDg += dout[(size_t)r * cols + col] * x[(size_t)r * cols + col] * inv;
	}
	localDg = blockReduceSum(localDg, smem);

	if (threadIdx.x == 0)
		dgamma[col] += localDg;
}

} // anonymous namespace

bool rmsnorm_backward(const float* dout, const float* x,
                      const float* gamma, const float* invRms,
                      int rows, int cols,
                      float* dx, float* dgamma)
{
	if (rows <= 0 || cols <= 0) return true;

	// Kernel 1: dx (one block per row).
	int block1 = rowBlockSize(cols);
	int smemBytes1 = (block1 / 32 + 1) * sizeof(float);
	rmsnorm_backward_dx<<<rows, block1, smemBytes1, computeStream()>>>(
		dout, x, gamma, invRms, cols, dx);
	GLADES_CUDA_CHECK(cudaGetLastError());

	// Kernel 2: dgamma (one block per column, reduce across rows).
	int block2 = rowBlockSize(rows);
	int smemBytes2 = (block2 / 32 + 1) * sizeof(float);
	rmsnorm_backward_dgamma<<<cols, block2, smemBytes2, computeStream()>>>(
		dout, x, invRms, rows, cols, dgamma);
	GLADES_CUDA_CHECK(cudaGetLastError());

	return true;
}

// ===========================================================================
//  5. Stable softmax (per-row)
// ===========================================================================

namespace {

__global__ void softmax_stable_rows(const float* __restrict__ x,
                                    int cols,
                                    float* __restrict__ out)
{
	int row = blockIdx.x;
	const float* xRow = x + (size_t)row * cols;
	float* oRow       = out + (size_t)row * cols;

	extern __shared__ float smem[];
	float* sMax = smem;
	float* sSum = smem + (blockDim.x / 32 + 1);

	// Pass 1: row max.
	float localMax = -FLT_MAX;
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		localMax = fmaxf(localMax, xRow[i]);
	localMax = blockReduceMax(localMax, sMax);

	__shared__ float sRowMax;
	if (threadIdx.x == 0) sRowMax = localMax;
	__syncthreads();
	float rowMax = sRowMax;

	// Pass 2: sum of exp(x - max).
	float localSum = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float e = expf(xRow[i] - rowMax);
		oRow[i] = e;  // store intermediate exp in output
		localSum += e;
	}
	localSum = blockReduceSum(localSum, sSum);

	__shared__ float sRowSum;
	if (threadIdx.x == 0) sRowSum = localSum;
	__syncthreads();
	float invSum = 1.0f / sRowSum;

	// Pass 3: normalize.
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		oRow[i] *= invSum;
}

} // anonymous namespace

bool softmax_forward(const float* x, int rows, int cols, float* out)
{
	if (rows <= 0 || cols <= 0) return true;
	int block = rowBlockSize(cols);
	int smemBytes = (block / 32 + 2) * 2 * sizeof(float);
	softmax_stable_rows<<<rows, block, smemBytes, computeStream()>>>(x, cols, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  6. Softmax cross-entropy backward
// ===========================================================================

namespace {

__global__ void softmax_cross_entropy_backward(const float* __restrict__ probs,
                                               const int* __restrict__ targets,
                                               int cols,
                                               float* __restrict__ dlogits)
{
	// Grid: (rows * ceil(cols / blockDim.x)).
	int row = blockIdx.x;
	int target = targets[row];
	const float* pRow = probs   + (size_t)row * cols;
	float* dRow       = dlogits + (size_t)row * cols;

	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float p = pRow[i];
		dRow[i] = (i == target) ? (p - 1.0f) : p;
	}
}

} // anonymous namespace

bool softmax_cross_entropy_bwd(const float* probs, const int* targets,
                               int rows, int cols, float* dlogits)
{
	if (rows <= 0 || cols <= 0) return true;
	int block = rowBlockSize(cols);
	softmax_cross_entropy_backward<<<rows, block, 0, computeStream()>>>(probs, targets, cols, dlogits);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  6b. DISTILL-FORWARD (paradigm shift #56) — KL + CE combined loss
// ===========================================================================
//
// Combined loss L = α · KL(p_T || p_S) + (1 - α) · CE(p_S, target)
// where p_T is teacher softmax, p_S is student softmax, target is the true
// next-token id.  The teacher distribution is treated as constant for the
// backward (frozen teacher).
//
// Backward derivation (per-row, single token t):
//   ∂CE/∂z_v       = p_S[v] - [v == target]
//   ∂KL/∂z_v       = p_S[v] - p_T[v]   (teacher constant; well-known)
//   ∂L/∂z_v        = α · (p_S[v] - p_T[v]) + (1 - α) · (p_S[v] - [v==target])
//                  = p_S[v] - α · p_T[v] - (1 - α) · [v == target]
//
// Forward scalar loss (for logging, optional):
//   CE  = -log p_S[target]
//   KL  = Σ_v p_T[v] · (log p_T[v] - log p_S[v])
//   L   = α · KL + (1 - α) · CE
//
// Both kernels skip rows whose target is padToken or out-of-range, mirroring
// the existing cross_entropy_nll_loss semantics.

namespace {

__global__ void distill_combined_backward(const float* __restrict__ probs_student,
                                          const float* __restrict__ probs_teacher,
                                          const int*   __restrict__ targets,
                                          int cols, float alpha,
                                          float* __restrict__ dlogits)
{
	// Grid: rows (one row per timestep).
	int row    = blockIdx.x;
	int target = targets[row];
	const float* pS = probs_student + (size_t)row * cols;
	const float* pT = probs_teacher + (size_t)row * cols;
	float*       dR = dlogits        + (size_t)row * cols;

	const float one_minus_alpha = 1.0f - alpha;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float val = pS[i] - alpha * pT[i];
		if (i == target) val -= one_minus_alpha;
		dR[i] = val;
	}
}

__global__ void distill_combined_loss_kernel(const float* __restrict__ probs_student,
                                             const float* __restrict__ probs_teacher,
                                             const int*   __restrict__ targets,
                                             int T, int vocabSize, int padToken,
                                             float alpha,
                                             float* __restrict__ loss_sum,
                                             int*   __restrict__ valid_count)
{
	extern __shared__ float smem[];
	float localLoss = 0.0f;
	int   localCnt  = 0;

	for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < T;
	     t += blockDim.x * gridDim.x)
	{
		int tgt = targets[t];
		if (padToken >= 0 && tgt == padToken) continue;
		if (tgt < 0 || tgt >= vocabSize) continue;

		const float* pS = probs_student + (size_t)t * vocabSize;
		const float* pT = probs_teacher + (size_t)t * vocabSize;

		// CE = -log p_S[target] with a tiny floor to avoid log(0).
		float pStgt = pS[tgt];
		if (pStgt < 1e-12f) pStgt = 1e-12f;
		float ce = -logf(pStgt);

		// KL(p_T || p_S) = Σ p_T[v] (log p_T[v] - log p_S[v]).
		float kl = 0.0f;
		for (int v = 0; v < vocabSize; ++v) {
			float pt = pT[v];
			if (pt <= 0.0f) continue;
			float ps = pS[v];
			if (ps < 1e-12f) ps = 1e-12f;
			kl += pt * (logf(pt) - logf(ps));
		}

		localLoss += alpha * kl + (1.0f - alpha) * ce;
		++localCnt;
	}

	float lossF = blockReduceSum(localLoss, smem);
	if (threadIdx.x == 0) atomicAdd(loss_sum, lossF);
	float cntF = (float)localCnt;
	cntF = blockReduceSum(cntF, smem);
	if (threadIdx.x == 0) atomicAdd(valid_count, (int)cntF);
}

} // anonymous namespace

bool distill_combined_bwd(const float* probs_student, const float* probs_teacher,
                          const int* targets, int rows, int cols, float alpha,
                          float* dlogits)
{
	if (rows <= 0 || cols <= 0) return true;
	if (probs_teacher == NULL) return false;
	int block = rowBlockSize(cols);
	distill_combined_backward<<<rows, block, 0, computeStream()>>>(
	    probs_student, probs_teacher, targets, cols, alpha, dlogits);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool distill_combined_loss(const float* probs_student, const float* probs_teacher,
                           const int* targets,
                           int T, int vocabSize, int padToken, float alpha,
                           float* loss_sum, int* valid_count)
{
	if (T <= 0 || vocabSize <= 0) return true;
	if (probs_teacher == NULL) return false;
	GLADES_CUDA_CHECK(cudaMemset(loss_sum,    0, sizeof(float)));
	GLADES_CUDA_CHECK(cudaMemset(valid_count, 0, sizeof(int)));
	int block = 256;
	int grid  = 1;
	if (T > 256) { grid = (T + block - 1) / block; if (grid > 128) grid = 128; }
	int smemBytes = (block / 32 + 2) * sizeof(float);
	distill_combined_loss_kernel<<<grid, block, smemBytes, computeStream()>>>(
	    probs_student, probs_teacher, targets, T, vocabSize, padToken, alpha,
	    loss_sum, valid_count);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  7. GELU (tanh approximation)
// ===========================================================================

namespace {

// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu_val(float x)
{
	const float kA = 0.7978845608f; // sqrt(2/pi)
	const float kB = 0.044715f;
	float x3 = x * x * x;
	float inner = kA * (x + kB * x3);
	return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float gelu_grad(float x)
{
	const float kA = 0.7978845608f;
	const float kB = 0.044715f;
	float x2 = x * x;
	float inner = kA * (x + kB * x * x2);
	float th = tanhf(inner);
	float sech2 = 1.0f - th * th;
	float dInner = kA * (1.0f + 3.0f * kB * x2);
	return 0.5f * (1.0f + th) + 0.5f * x * sech2 * dInner;
}

__global__ void gelu_forward_kernel(const float* __restrict__ x, int n,
                                    float* __restrict__ out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		out[idx] = gelu_val(x[idx]);
}

__global__ void gelu_backward_kernel(const float* __restrict__ dout,
                                     const float* __restrict__ x, int n,
                                     float* __restrict__ dx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		dx[idx] = dout[idx] * gelu_grad(x[idx]);
}

} // anonymous namespace

bool gelu_forward(const float* x, int n, float* out)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	gelu_forward_kernel<<<grid, kBlockElem, 0, computeStream()>>>(x, n, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool gelu_backward(const float* dout, const float* x, int n, float* dx)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	gelu_backward_kernel<<<grid, kBlockElem, 0, computeStream()>>>(dout, x, n, dx);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  8. SiLU (x * sigmoid(x))
// ===========================================================================

namespace {

__device__ __forceinline__ float sigmoid_val(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__global__ void silu_forward_kernel(const float* __restrict__ x, int n,
                                    float* __restrict__ out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		float s = sigmoid_val(x[idx]);
		out[idx] = x[idx] * s;
	}
}

__global__ void silu_backward_kernel(const float* __restrict__ dout,
                                     const float* __restrict__ x, int n,
                                     float* __restrict__ dx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		float s = sigmoid_val(x[idx]);
		// d/dx [x * sigma(x)] = sigma(x) + x * sigma(x) * (1 - sigma(x))
		//                      = sigma(x) * (1 + x * (1 - sigma(x)))
		dx[idx] = dout[idx] * s * (1.0f + x[idx] * (1.0f - s));
	}
}

} // anonymous namespace

bool silu_forward(const float* x, int n, float* out)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	silu_forward_kernel<<<grid, kBlockElem, 0, computeStream()>>>(x, n, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool silu_backward(const float* dout, const float* x, int n, float* dx)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	silu_backward_kernel<<<grid, kBlockElem, 0, computeStream()>>>(dout, x, n, dx);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  9. ReLU
// ===========================================================================

namespace {

__global__ void relu_forward_kernel(const float* __restrict__ x, int n,
                                    float* __restrict__ out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		out[idx] = fmaxf(0.0f, x[idx]);
}

__global__ void relu_backward_kernel(const float* __restrict__ dout,
                                     const float* __restrict__ x, int n,
                                     float* __restrict__ dx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		dx[idx] = (x[idx] > 0.0f) ? dout[idx] : 0.0f;
}

} // anonymous namespace

bool relu_forward(const float* x, int n, float* out)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	relu_forward_kernel<<<grid, kBlockElem, 0, computeStream()>>>(x, n, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool relu_backward(const float* dout, const float* x, int n, float* dx)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	relu_backward_kernel<<<grid, kBlockElem, 0, computeStream()>>>(dout, x, n, dx);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  10. SwiGLU
// ===========================================================================

namespace {

// gate_up layout: [n, 2*dFF].  First dFF columns = gate, last dFF = up.
// out[i, j] = silu(gate[i, j]) * up[i, j]
// 1-D grid over n * dFF elements.
__global__ void swiglu_fwd(const float* __restrict__ gate_up,
                           int n, int dFF,
                           float* __restrict__ out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n * dFF) return;
	int row = idx / dFF;
	int col = idx % dFF;

	int totalCols = 2 * dFF;
	const float* guRow = gate_up + (size_t)row * totalCols;
	float g = guRow[col];
	float u = guRow[col + dFF];
	float s = sigmoid_val(g);
	out[idx] = g * s * u;
}

__global__ void swiglu_bwd(const float* __restrict__ dout,
                           const float* __restrict__ gate_up,
                           int n, int dFF,
                           float* __restrict__ d_gate_up)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n * dFF) return;
	int row = idx / dFF;
	int col = idx % dFF;

	int totalCols = 2 * dFF;
	const float* guRow = gate_up + (size_t)row * totalCols;
	float g = guRow[col];
	float u = guRow[col + dFF];
	float s = sigmoid_val(g);
	float silu_g = g * s;

	float do_val = dout[idx];

	// d_gate = dout * up * d_silu(gate) = dout * up * sigma(g)*(1 + g*(1-sigma(g)))
	float dsilu = s * (1.0f + g * (1.0f - s));
	float dg = do_val * u * dsilu;
	// d_up = dout * silu(gate)
	float du = do_val * silu_g;

	float* dguRow = d_gate_up + (size_t)row * totalCols;
	dguRow[col]        = dg;
	dguRow[col + dFF]  = du;
}

} // anonymous namespace

bool swiglu_forward(const float* gate_up, int n, int dFF, float* out)
{
	if (n <= 0 || dFF <= 0) return true;
	int total = n * dFF;
	int grid = (total + kBlockElem - 1) / kBlockElem;
	swiglu_fwd<<<grid, kBlockElem, 0, computeStream()>>>(gate_up, n, dFF, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool swiglu_backward(const float* dout, const float* gate_up,
                     int n, int dFF, float* d_gate_up)
{
	if (n <= 0 || dFF <= 0) return true;
	int total = n * dFF;
	int grid = (total + kBlockElem - 1) / kBlockElem;
	swiglu_bwd<<<grid, kBlockElem, 0, computeStream()>>>(dout, gate_up, n, dFF, d_gate_up);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  11. Rotary positional encoding (RoPE)
// ===========================================================================

namespace {

__global__ void rope_apply_inplace(float* __restrict__ x,
                                   const float* __restrict__ invFreq,
                                   int T, int nHeads, int dHead,
                                   int halfDim, bool inverse)
{
	// Grid: one thread per (t, h, d) triple where d in [0, halfDim).
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = T * nHeads * halfDim;
	if (idx >= total) return;

	int d   = idx % halfDim;
	int tmp = idx / halfDim;
	int h   = tmp % nHeads;
	int t   = tmp / nHeads;

	float theta = (float)t * invFreq[d];
	float cosT  = cosf(theta);
	float sinT  = inverse ? -sinf(theta) : sinf(theta);

	// x layout: [T, nHeads, dHead]
	size_t base = ((size_t)t * nHeads + h) * dHead;
	float x0 = x[base + d];
	float x1 = x[base + d + halfDim];

	x[base + d]            = x0 * cosT - x1 * sinT;
	x[base + d + halfDim]  = x0 * sinT + x1 * cosT;
}

} // anonymous namespace

bool rope_apply(float* x, const float* invFreq,
                int T, int nHeads, int dHead,
                int halfDim, bool inverse)
{
	if (T <= 0 || nHeads <= 0 || dHead < 2) return true;
	int hd = (halfDim > 0 && halfDim <= dHead / 2) ? halfDim : (dHead / 2);
	int total = T * nHeads * hd;
	int grid = (total + kBlockElem - 1) / kBlockElem;
	rope_apply_inplace<<<grid, kBlockElem, 0, computeStream()>>>(x, invFreq, T, nHeads, dHead, hd, inverse);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Fused Q+K RoPE: apply RoPE to both Q and K in a single kernel launch.
// Grid: (ceil(total/blockDim.x), 2) — blockIdx.y==0 for Q, blockIdx.y==1 for K.
namespace {

__global__ void rope_apply_qk_kernel(float* __restrict__ Q,
                                      float* __restrict__ K,
                                      const float* __restrict__ invFreq,
                                      int T, int nQHeads, int nKVHeads,
                                      int dHead, int halfDim, bool inverse,
                                      int totalQ, int totalK)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int region = blockIdx.y; // 0=Q, 1=K

	float* x;
	int nHeads, total;
	if (region == 0) {
		x = Q; nHeads = nQHeads; total = totalQ;
	} else {
		x = K; nHeads = nKVHeads; total = totalK;
	}

	if (idx >= total) return;

	int d   = idx % halfDim;
	int tmp = idx / halfDim;
	int h   = tmp % nHeads;
	int t   = tmp / nHeads;

	float theta = (float)t * invFreq[d];
	float cosT  = cosf(theta);
	float sinT  = inverse ? -sinf(theta) : sinf(theta);

	size_t base = ((size_t)t * nHeads + h) * dHead;
	float x0 = x[base + d];
	float x1 = x[base + d + halfDim];

	x[base + d]            = x0 * cosT - x1 * sinT;
	x[base + d + halfDim]  = x0 * sinT + x1 * cosT;
}

} // anonymous namespace

bool rope_apply_qk(float* Q, float* K, const float* invFreq,
                    int T, int nQHeads, int nKVHeads, int dHead,
                    int halfDim, bool inverse)
{
	if (T <= 0 || dHead < 2) return true;
	int hd = (halfDim > 0 && halfDim <= dHead / 2) ? halfDim : (dHead / 2);
	int totalQ = T * nQHeads * hd;
	int totalK = T * nKVHeads * hd;
	int maxTotal = (totalQ > totalK) ? totalQ : totalK;
	int gridX = (maxTotal + kBlockElem - 1) / kBlockElem;
	dim3 grid(gridX, 2);
	rope_apply_qk_kernel<<<grid, kBlockElem, 0, computeStream()>>>(Q, K, invFreq, T, nQHeads, nKVHeads,
	                                            dHead, hd, inverse, totalQ, totalK);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  12. Simple vector ops: add_bias, add_residual, axpy
// ===========================================================================

namespace {

__global__ void add_bias_kernel(float* __restrict__ out,
                                const float* __restrict__ bias,
                                int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= rows * cols) return;
	int col = idx % cols;
	out[idx] += bias[col];
}

__global__ void add_residual_kernel(float* __restrict__ out,
                                    const float* __restrict__ residual,
                                    int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		out[idx] += residual[idx];
}

__global__ void axpy_kernel(float alpha,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		y[idx] += alpha * x[idx];
}

} // anonymous namespace

bool add_bias(float* out, const float* bias, int rows, int cols)
{
	if (rows <= 0 || cols <= 0) return true;
	int total = rows * cols;
	int grid = (total + kBlockElem - 1) / kBlockElem;
	add_bias_kernel<<<grid, kBlockElem, 0, computeStream()>>>(out, bias, rows, cols);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool add_residual(float* out, const float* residual, int n)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	add_residual_kernel<<<grid, kBlockElem, 0, computeStream()>>>(out, residual, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool axpy(float alpha, const float* x, float* y, int n)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	axpy_kernel<<<grid, kBlockElem, 0, computeStream()>>>(alpha, x, y, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

namespace {

__global__ void add_two_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               int n,
                               float* __restrict__ out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		out[idx] = a[idx] + b[idx];
}

} // anonymous namespace

bool add_two(float* out, const float* a, const float* b, int n)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	add_two_kernel<<<grid, kBlockElem, 0, computeStream()>>>(a, b, n, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

namespace {
__global__ void scale_array_kernel(float* __restrict__ x, float scale, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) x[idx] *= scale;
}
} // anonymous namespace

bool scale_array(float* x, float scale, int n)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	scale_array_kernel<<<grid, kBlockElem, 0, computeStream()>>>(x, scale, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  13. Embedding gather / scatter_add
// ===========================================================================

namespace {

__global__ void embedding_gather_kernel(const float* __restrict__ E,
                                        const int* __restrict__ tokenIds,
                                        int T, int vocabSize, int dModel,
                                        float* __restrict__ out)
{
	// 1-D grid over T * dModel elements.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= T * dModel) return;
	int t = idx / dModel;
	int d = idx % dModel;
	int tok = tokenIds[t];
	// Bounds check on vocabulary.
	if (tok >= 0 && tok < vocabSize)
		out[idx] = E[(size_t)tok * dModel + d];
	else
		out[idx] = 0.0f;
}

// BF16 variant: gather from a bf16 embedding table and write FP32 output.
// Used when bf16-weights mode retires the FP32 master so tokE only exists
// as the bf16 mirror.  Output stays FP32 because downstream activation
// path (h, x1, etc.) is FP32.
__global__ void embedding_gather_bf16_kernel(const uint16_t* __restrict__ E_bf16,
                                              const int* __restrict__ tokenIds,
                                              int T, int vocabSize, int dModel,
                                              float* __restrict__ out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= T * dModel) return;
	int t = idx / dModel;
	int d = idx % dModel;
	int tok = tokenIds[t];
	if (tok >= 0 && tok < vocabSize)
	{
		// Decode BF16 to FP32: zero-extend then shift left 16 bits.
		union { uint32_t u; float f; } uv;
		uv.u = static_cast<uint32_t>(E_bf16[(size_t)tok * dModel + d]) << 16;
		out[idx] = uv.f;
	}
	else
	{
		out[idx] = 0.0f;
	}
}

__global__ void embedding_scatter_add_kernel(float* __restrict__ dE,
                                             const int* __restrict__ tokenIds,
                                             const float* __restrict__ dout,
                                             int T, int vocabSize, int dModel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= T * dModel) return;
	int t = idx / dModel;
	int d = idx % dModel;
	int tok = tokenIds[t];
	if (tok >= 0 && tok < vocabSize)
		atomicAdd(&dE[(size_t)tok * dModel + d], dout[idx]);
}

// BF16 atomic-add via atomicCAS on uint16_t.  Used by Phase-3 BF16-grad path
// for the embedding-grad scatter-add (gTokE) — directly accumulates each
// token's dout slice into the persistent BF16 mirror, eliminating both the
// FP32 grad allocation and the Phase-1 cast pass for gTokE.
__global__ void embedding_scatter_add_bf16_kernel(uint16_t* __restrict__ dE_bf16,
                                                  const int* __restrict__ tokenIds,
                                                  const float* __restrict__ dout,
                                                  int T, int vocabSize, int dModel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= T * dModel) return;
	int t = idx / dModel;
	int d = idx % dModel;
	int tok = tokenIds[t];
	if (tok < 0 || tok >= vocabSize) return;

	const float addend = dout[idx];
	uint16_t* slot = &dE_bf16[(size_t)tok * dModel + d];

	// Loop until atomicCAS succeeds.  At vocab>=50K and T<=4096 the
	// expected per-slot contention is tiny (typical T/V ≈ 0.08).
	uint16_t old = *slot;
	for (;;)
	{
		// Decode old bf16 to f32, add, encode back to bf16 (RNE).
		union { uint32_t u; float f; } uv;
		uv.u = static_cast<uint32_t>(old) << 16;
		const float sum = uv.f + addend;
		union { float f; uint32_t u; } v;
		v.f = sum;
		uint16_t newBf16;
		if (isnan(sum))
		{
			const uint32_t sign = v.u & 0x80000000u;
			newBf16 = static_cast<uint16_t>(((sign | 0x7FC00000u) >> 16) & 0xFFFFu);
		}
		else
		{
			const uint32_t lsb = (v.u >> 16) & 1u;
			const uint32_t roundingBias = 0x7FFFu + lsb;
			newBf16 = static_cast<uint16_t>((v.u + roundingBias) >> 16);
		}
		if (old == newBf16) break;  // addend was zero or got rounded away
		// atomicCAS works on uint32_t; pack the 16-bit slot into a 32-bit word.
		uintptr_t slotAddr = reinterpret_cast<uintptr_t>(slot);
		uint32_t* base32 = reinterpret_cast<uint32_t*>(slotAddr & ~uintptr_t(2));
		const bool highHalf = (slotAddr & 2u) != 0u;
		uint32_t fullOld = *base32;
		uint16_t curOld = highHalf ? (uint16_t)(fullOld >> 16) : (uint16_t)(fullOld & 0xFFFFu);
		if (curOld != old) { old = curOld; continue; }
		uint32_t fullNew = highHalf
		    ? ((fullOld & 0x0000FFFFu) | (static_cast<uint32_t>(newBf16) << 16))
		    : ((fullOld & 0xFFFF0000u) | static_cast<uint32_t>(newBf16));
		uint32_t prev = atomicCAS(base32, fullOld, fullNew);
		if (prev == fullOld) break;
		// Lost the race; refresh and retry.
		old = highHalf ? (uint16_t)(prev >> 16) : (uint16_t)(prev & 0xFFFFu);
	}
}

} // anonymous namespace

bool embedding_gather(const float* E, const int* tokenIds,
                      int T, int vocabSize, int dModel, float* out)
{
	if (T <= 0 || dModel <= 0) return true;

	int total = T * dModel;
	int grid = (total + kBlockElem - 1) / kBlockElem;
	embedding_gather_kernel<<<grid, kBlockElem, 0, computeStream()>>>(E, tokenIds, T, vocabSize, dModel, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool embedding_gather_bf16(const uint16_t* E_bf16, const int* tokenIds,
                            int T, int vocabSize, int dModel, float* out)
{
	if (T <= 0 || dModel <= 0) return true;
	int total = T * dModel;
	int grid = (total + kBlockElem - 1) / kBlockElem;
	embedding_gather_bf16_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
	    E_bf16, tokenIds, T, vocabSize, dModel, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool embedding_scatter_add(float* dE, const int* tokenIds,
                           const float* dout,
                           int T, int vocabSize, int dModel)
{
	if (T <= 0 || dModel <= 0) return true;
	int total = T * dModel;
	int grid = (total + kBlockElem - 1) / kBlockElem;
	embedding_scatter_add_kernel<<<grid, kBlockElem, 0, computeStream()>>>(dE, tokenIds, dout, T, vocabSize, dModel);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool embedding_scatter_add_bf16(uint16_t* dE_bf16, const int* tokenIds,
                                const float* dout,
                                int T, int vocabSize, int dModel)
{
	if (T <= 0 || dModel <= 0) return true;
	int total = T * dModel;
	int grid = (total + kBlockElem - 1) / kBlockElem;
	embedding_scatter_add_bf16_kernel<<<grid, kBlockElem, 0, computeStream()>>>(dE_bf16, tokenIds, dout, T, vocabSize, dModel);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  14. Adam optimizer
// ===========================================================================

namespace {

__global__ void adam_update_kernel(float* __restrict__ param,
                                  const float* __restrict__ grad,
                                  float* __restrict__ m,
                                  float* __restrict__ v,
                                  float lr, float beta1, float beta2,
                                  float eps, float weightDecay,
                                  float gradScale,
                                  int step, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	float g = grad[idx] * gradScale;

	// Decoupled weight decay (AdamW).
	if (weightDecay != 0.0f)
		param[idx] -= lr * weightDecay * param[idx];

	// Update biased first and second moments.
	float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
	float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
	m[idx] = m_new;
	v[idx] = v_new;

	// Bias correction.
	float bc1 = 1.0f - powf(beta1, (float)step);
	float bc2 = 1.0f - powf(beta2, (float)step);
	float m_hat = m_new / bc1;
	float v_hat = v_new / bc2;

	param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// SOPHIA-G — paradigm shift #55 (Liu et al. 2023 "Sophia: A Scalable Stochastic
// Second-order Optimizer for Language Model Pre-training", Sophia-G variant
// using gradient-squared as a diagonal Hessian proxy in lieu of Hutchinson HVP).
//
// Sophia maintains:
//   m_t = β_1 m_{t-1} + (1-β_1) g_t                           (1st moment, same as Adam)
//   h_t = β_2 h_{t-1} + (1-β_2) g_t * g_t                     (Hessian proxy, = Adam's v)
// Update rule:
//   ratio_t = m_hat_t / max(γ · h_hat_t, ε)
//   ratio_t = clip(ratio_t, -ρ, ρ)
//   θ_{t+1} = θ_t - lr · ratio_t  (decoupled weight decay applied separately)
//
// Defaults: β_1=0.965, β_2=0.99, γ=0.05, ρ=1.0.  The clip is the key
// difference from Adam — caps update magnitude in directions where the
// Hessian is small and the ratio explodes.  Empirical 1.5-2× steps
// reduction vs Adam in the published paper.
//
// Sophia-G uses g² as a coarse Hessian proxy (vs Sophia-H's Hutchinson HVP);
// loses some of the speedup but no extra forward/backward passes required.
__global__ void sophia_g_update_kernel(float* __restrict__ param,
                                        const float* __restrict__ grad,
                                        float* __restrict__ m,
                                        float* __restrict__ h,
                                        float lr, float beta1, float beta2,
                                        float gamma, float rho, float eps,
                                        float weightDecay, float gradScale,
                                        int step, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	float g = grad[idx] * gradScale;

	// Decoupled weight decay (AdamW-style).
	if (weightDecay != 0.0f)
		param[idx] -= lr * weightDecay * param[idx];

	// EMAs: 1st moment (Adam-equiv) and Hessian proxy (= grad² EMA).
	float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
	float h_new = beta2 * h[idx] + (1.0f - beta2) * g * g;
	m[idx] = m_new;
	h[idx] = h_new;

	// Bias correction.
	float bc1 = 1.0f - powf(beta1, (float)step);
	float bc2 = 1.0f - powf(beta2, (float)step);
	float m_hat = m_new / bc1;
	float h_hat = h_new / bc2;

	// Sophia ratio: m_hat / max(γ · h_hat, ε), clipped to [-ρ, ρ].
	float denom = fmaxf(gamma * h_hat, eps);
	float ratio = m_hat / denom;
	if (ratio > rho) ratio = rho;
	else if (ratio < -rho) ratio = -rho;

	param[idx] -= lr * ratio;
}

// iter 181 — ASTRA paradigm #41 (Gate-0, m=1 stateless v).
// Replaces Adam's persistent v EMA with the within-step gradient
// magnitude v_t = g_t² + eps².  Persistent state collapses to momentum
// `m` only (no `v`, no Kahan `c`).  Premise test: does removing the
// long-horizon v EMA preserve convergence at small scale?
__global__ void astra_update_kernel(float* __restrict__ param,
                                    const float* __restrict__ grad,
                                    float* __restrict__ m,
                                    float lr, float beta1,
                                    float eps, float weightDecay,
                                    float gradScale,
                                    int step, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	float g = grad[idx] * gradScale;

	if (weightDecay != 0.0f)
		param[idx] -= lr * weightDecay * param[idx];

	float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
	m[idx] = m_new;

	float bc1 = 1.0f - powf(beta1, (float)step);
	float m_hat = m_new / bc1;

	// ASTRA: stateless v from instantaneous g², no EMA.
	float v_inst = g * g;
	param[idx] -= lr * m_hat / (sqrtf(v_inst) + eps);
}

} // anonymous namespace

bool sophia_g_update(float* param, const float* grad, float* m, float* h,
                      float lr, float beta1, float beta2,
                      float gamma, float rho, float eps,
                      float weightDecay, float gradScale, int step, int n)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	sophia_g_update_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
		param, grad, m, h, lr, beta1, beta2, gamma, rho, eps,
		weightDecay, gradScale, step, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool adam_update(float* param, const float* grad, float* m, float* v,
                 float lr, float beta1, float beta2, float eps,
                 float weightDecay, float gradScale, int step, int n)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	adam_update_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
		param, grad, m, v, lr, beta1, beta2, eps, weightDecay, gradScale, step, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool astra_update(float* param, const float* grad, float* m,
                  float lr, float beta1, float eps,
                  float weightDecay, float gradScale, int step, int n)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	astra_update_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
		param, grad, m, lr, beta1, eps, weightDecay, gradScale, step, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ---------------------------------------------------------------------------
// Adam with BF16-packed optimizer state
// ---------------------------------------------------------------------------
// Loads m, v from uint16_t (BF16 bit pattern in high half-word), casts to
// FP32 for EMA compute, stores back as BF16 with round-to-nearest-even.
// Weights and grads stay FP32 throughout. Mathematical semantics identical
// to adam_update; only EMA storage precision differs.

namespace {

__device__ __forceinline__ float bf16_load_as_f32(uint16_t b)
{
	union { uint32_t u; float f; } v;
	v.u = static_cast<uint32_t>(b) << 16;
	return v.f;
}

__device__ __forceinline__ uint16_t bf16_store_from_f32(float f)
{
	union { float f; uint32_t u; } v;
	v.f = f;
	if (isnan(f))
	{
		const uint32_t sign = v.u & 0x80000000u;
		return static_cast<uint16_t>(((sign | 0x7FC00000u) >> 16) & 0xFFFFu);
	}
	const uint32_t lsb = (v.u >> 16) & 1u;
	const uint32_t roundingBias = 0x7FFFu + lsb;
	return static_cast<uint16_t>((v.u + roundingBias) >> 16);
}

__global__ void adam_update_bf16_state_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    uint16_t* __restrict__ m_bf16,
    uint16_t* __restrict__ v_bf16,
    uint16_t* __restrict__ c_bf16,   // iter 171: Kahan compensation for v (nullable)
    float lr, float beta1, float beta2,
    float eps, float weightDecay,
    float gradScale,
    int step, int n)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	const float g = grad[idx] * gradScale;

	// Decoupled weight decay (AdamW) — weights are FP32.
	if (weightDecay != 0.0f)
		param[idx] -= lr * weightDecay * param[idx];

	// Load BF16 moments, upcast to FP32 for EMA arithmetic.
	const float m_old = bf16_load_as_f32(m_bf16[idx]);
	const float v_old = bf16_load_as_f32(v_bf16[idx]);

	const float m_new = beta1 * m_old + (1.0f - beta1) * g;

	// iter 171: Kahan-compensated v update.  Without compensation, BF16 v
	// silently loses contributions when (1-β₂)·g² « β₂·v_old (β₂=0.999
	// already shrinks the contribution 1000×; BF16's 3-digit mantissa
	// rounds the sum down to β₂·v_old).  Over 100k+ steps v drifts low,
	// Adam's m/√v becomes too large, weights overshoot.  Mid-Phase-C
	// divergence at 1.84B × 650k run-3 (2026-04-26) — see surprise #17.
	//
	// Compensation: track the bits truncated when storing v as BF16
	// last step, re-apply them this step.  Memory cost: +1 BF16/param.
	float v_new;
	uint16_t v_new_bf16_packed;
	if (c_bf16 != nullptr)
	{
		const float c_old = bf16_load_as_f32(c_bf16[idx]);
		const float v_decay = beta2 * v_old;
		const float input = (1.0f - beta2) * g * g + c_old;
		const float v_full = v_decay + input;
		v_new_bf16_packed = bf16_store_from_f32(v_full);
		const float v_stored = bf16_load_as_f32(v_new_bf16_packed);
		// Residual = full-precision sum minus what BF16 actually stored.
		// Carries forward into next step's update.
		c_bf16[idx] = bf16_store_from_f32(v_full - v_stored);
		v_new = v_stored;
	}
	else
	{
		v_new = beta2 * v_old + (1.0f - beta2) * g * g;
		v_new_bf16_packed = bf16_store_from_f32(v_new);
	}

	// Store back as BF16 (round-to-nearest-even).
	m_bf16[idx] = bf16_store_from_f32(m_new);
	v_bf16[idx] = v_new_bf16_packed;

	// Bias correction (matches FP32 adam_update_kernel).
	const float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
	const float m_hat = m_new / bc1;
	const float v_hat = v_new / bc2;

	param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// SOPHIA-G with bf16 m/h state.  Same structure as adam_update_bf16_state
// but with the Sophia clipped second-order rule replacing Adam's m/sqrt(v).
__global__ void sophia_g_update_bf16_state_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    uint16_t* __restrict__ m_bf16,
    uint16_t* __restrict__ h_bf16,
    float lr, float beta1, float beta2,
    float gamma, float rho, float eps,
    float weightDecay, float gradScale,
    int step, int n)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	const float g = grad[idx] * gradScale;

	// Decoupled weight decay (AdamW-style).
	if (weightDecay != 0.0f)
		param[idx] -= lr * weightDecay * param[idx];

	const float m_old = bf16_load_as_f32(m_bf16[idx]);
	const float h_old = bf16_load_as_f32(h_bf16[idx]);

	const float m_new = beta1 * m_old + (1.0f - beta1) * g;
	const float h_new = beta2 * h_old + (1.0f - beta2) * g * g;

	m_bf16[idx] = bf16_store_from_f32(m_new);
	h_bf16[idx] = bf16_store_from_f32(h_new);

	const float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
	const float m_hat = m_new / bc1;
	const float h_hat = h_new / bc2;

	const float denom = fmaxf(gamma * h_hat, eps);
	float ratio = m_hat / denom;
	if (ratio > rho) ratio = rho;
	else if (ratio < -rho) ratio = -rho;

	param[idx] -= lr * ratio;
}

} // anonymous namespace

bool adam_update_bf16_state(float* param, const float* grad,
                            uint16_t* m_bf16, uint16_t* v_bf16,
                            float lr, float beta1, float beta2, float eps,
                            float weightDecay, float gradScale,
                            int step, int n)
{
	if (n <= 0) return true;
	const int grid = (n + kBlockElem - 1) / kBlockElem;
	adam_update_bf16_state_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
	    param, grad, m_bf16, v_bf16, /*c_bf16=*/nullptr,
	    lr, beta1, beta2, eps, weightDecay, gradScale, step, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool sophia_g_update_bf16_state(float* param, const float* grad,
                                 uint16_t* m_bf16, uint16_t* h_bf16,
                                 float lr, float beta1, float beta2,
                                 float gamma, float rho, float eps,
                                 float weightDecay, float gradScale,
                                 int step, int n)
{
	if (n <= 0) return true;
	const int grid = (n + kBlockElem - 1) / kBlockElem;
	sophia_g_update_bf16_state_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
	    param, grad, m_bf16, h_bf16,
	    lr, beta1, beta2, gamma, rho, eps,
	    weightDecay, gradScale, step, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// iter 171: Kahan-compensated variant — same kernel, c_bf16 buffer carries
// the truncation residual from each step's v update into the next step.
// See adam_update_bf16_state_kernel for mechanism.
bool adam_update_bf16_kahan_state(float* param, const float* grad,
                                   uint16_t* m_bf16, uint16_t* v_bf16,
                                   uint16_t* c_bf16,
                                   float lr, float beta1, float beta2, float eps,
                                   float weightDecay, float gradScale,
                                   int step, int n)
{
	if (n <= 0) return true;
	const int grid = (n + kBlockElem - 1) / kBlockElem;
	adam_update_bf16_state_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
	    param, grad, m_bf16, v_bf16, c_bf16,
	    lr, beta1, beta2, eps, weightDecay, gradScale, step, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ---------------------------------------------------------------------------
// BF16-grad variants — same Adam math as above but read the gradient from a
// BF16 buffer instead of FP32.  Used when MixedPrecisionConfig::gradStorageBf16
// is true (paradigm-stack at ≥500M); halves grad-buffer VRAM at the cost of
// BF16's 7-bit mantissa quantization noise on each Adam step (averaged out
// by the EMA in m, v).
// ---------------------------------------------------------------------------
namespace {
__global__ void adam_update_bf16_state_bf16grad_kernel(
    float* __restrict__ param,
    const uint16_t* __restrict__ grad_bf16,
    uint16_t* __restrict__ m_bf16,
    uint16_t* __restrict__ v_bf16,
    uint16_t* __restrict__ c_bf16,   // Kahan compensation (nullable)
    float lr, float beta1, float beta2,
    float eps, float weightDecay,
    float gradScale,
    int step, int n)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	const float g = bf16_load_as_f32(grad_bf16[idx]) * gradScale;

	if (weightDecay != 0.0f)
		param[idx] -= lr * weightDecay * param[idx];

	const float m_old = bf16_load_as_f32(m_bf16[idx]);
	const float v_old = bf16_load_as_f32(v_bf16[idx]);

	const float m_new = beta1 * m_old + (1.0f - beta1) * g;

	float v_new;
	uint16_t v_new_bf16_packed;
	if (c_bf16 != nullptr)
	{
		const float c_old = bf16_load_as_f32(c_bf16[idx]);
		const float v_decay = beta2 * v_old;
		const float input = (1.0f - beta2) * g * g + c_old;
		const float v_full = v_decay + input;
		v_new_bf16_packed = bf16_store_from_f32(v_full);
		const float v_stored = bf16_load_as_f32(v_new_bf16_packed);
		c_bf16[idx] = bf16_store_from_f32(v_full - v_stored);
		v_new = v_stored;
	}
	else
	{
		v_new = beta2 * v_old + (1.0f - beta2) * g * g;
		v_new_bf16_packed = bf16_store_from_f32(v_new);
	}

	m_bf16[idx] = bf16_store_from_f32(m_new);
	v_bf16[idx] = v_new_bf16_packed;

	const float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
	const float m_hat = m_new / bc1;
	const float v_hat = v_new / bc2;

	param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}
}  // anonymous namespace

bool adam_update_bf16_state_bf16grad(float* param, const uint16_t* grad_bf16,
                                      uint16_t* m_bf16, uint16_t* v_bf16,
                                      float lr, float beta1, float beta2, float eps,
                                      float weightDecay, float gradScale,
                                      int step, int n)
{
	if (n <= 0) return true;
	const int grid = (n + kBlockElem - 1) / kBlockElem;
	adam_update_bf16_state_bf16grad_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
	    param, grad_bf16, m_bf16, v_bf16, /*c_bf16=*/nullptr,
	    lr, beta1, beta2, eps, weightDecay, gradScale, step, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ---------------------------------------------------------------------------
// Adam with int8-packed optimizer state (block-wise scale).
// ---------------------------------------------------------------------------
// Each block of ADAM_INT8_BS parameters stores one FP32 absmax scale plus
// ADAM_INT8_BS int8/uint8 values per moment.
//
// ASYMMETRIC ENCODING (fixes linear-quant divergence at lr ≥ 3e-5):
//   m moment — signed int8, range [-absmax, +absmax], step absmax/127
//   v moment — UNSIGNED uint8, range [0, absmax], step absmax/255
//
// v is always non-negative (sum of squared grads), so allocating the full
// uint8 range [0, 255] to the positive axis doubles v's resolution near
// zero.  That region is where 1/√v enters the Adam denominator — losing
// precision there was the cause of the earlier divergence.
//
// Memory per param per moment ≈ 1.016 bytes (2× smaller than BF16,
// 4× smaller than FP32).  Math is identical to adam_update up to
// quantization noise on the EMAs; param + grad stay FP32.

namespace {

#define ADAM_INT8_BS 256   // chosen so shmem = 2 * BS * 4 = 2 KB — fits in L1

__global__ void adam_update_int8_state_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    int8_t*  __restrict__ m_int8,
    uint8_t* __restrict__ v_uint8,
    float* __restrict__ m_scale,
    float* __restrict__ v_scale,
    float lr, float beta1, float beta2,
    float eps, float weightDecay,
    float gradScale,
    int step, int n, int numBlocks)
{
	const int blockId = blockIdx.x;
	if (blockId >= numBlocks) return;

	const int start = blockId * ADAM_INT8_BS;
	const int end = min(start + ADAM_INT8_BS, n);
	const int len = end - start;

	const float oldMScale = m_scale[blockId];
	const float oldVScale = v_scale[blockId];
	const float qinvM = 1.0f / 127.0f;  // signed [-127, 127]
	const float qinvV = 1.0f / 255.0f;  // unsigned [0, 255]

	__shared__ float sM[ADAM_INT8_BS];
	__shared__ float sV[ADAM_INT8_BS];

	float localMmax = 0.0f;
	float localVmax = 0.0f;

	for (int i = threadIdx.x; i < len; i += blockDim.x)
	{
		const int gi = start + i;
		const float g = grad[gi] * gradScale;

		// Dequantize: m as signed int8, v as unsigned uint8.
		const float mOld = (float)m_int8[gi]  * oldMScale * qinvM;
		const float vOld = (float)v_uint8[gi] * oldVScale * qinvV;

		const float mNew = beta1 * mOld + (1.0f - beta1) * g;
		const float vNew = beta2 * vOld + (1.0f - beta2) * g * g;

		sM[i] = mNew;
		sV[i] = vNew;

		const float am = fabsf(mNew);
		// vNew is non-negative by construction (sum of squared grads), so
		// its absmax is just max.
		if (am > localMmax) localMmax = am;
		if (vNew > localVmax) localVmax = vNew;
	}

	// Block-reduce absmax via shuffle.  Works for blockDim.x ≤ 1024 and a
	// power of two; we launch with blockDim.x = ADAM_INT8_BS.
	__shared__ float sMax[2];
	float mMax = localMmax;
	float vMax = localVmax;
	for (int off = warpSize / 2; off > 0; off /= 2)
	{
		float o = __shfl_xor_sync(0xFFFFFFFF, mMax, off);
		if (o > mMax) mMax = o;
		o = __shfl_xor_sync(0xFFFFFFFF, vMax, off);
		if (o > vMax) vMax = o;
	}
	// Write per-warp results to shared.
	if ((threadIdx.x & (warpSize - 1)) == 0)
	{
		const int warpId = threadIdx.x / warpSize;
		sMax[warpId == 0 ? 0 : 0] = mMax;  // we only use warp 0 below
	}
	__syncthreads();

	// Block-wide reduction (one warp since BS=256 = 8 warps; do a second
	// shuffle pass through shared memory).
	__shared__ float sWarpM[32], sWarpV[32];
	const int warpId = threadIdx.x / warpSize;
	const int lane = threadIdx.x % warpSize;
	if (lane == 0)
	{
		sWarpM[warpId] = mMax;
		sWarpV[warpId] = vMax;
	}
	__syncthreads();

	if (warpId == 0)
	{
		const int numWarps = blockDim.x / warpSize;
		float mm = (lane < numWarps) ? sWarpM[lane] : 0.0f;
		float vv = (lane < numWarps) ? sWarpV[lane] : 0.0f;
		for (int off = warpSize / 2; off > 0; off /= 2)
		{
			float o = __shfl_xor_sync(0xFFFFFFFF, mm, off);
			if (o > mm) mm = o;
			o = __shfl_xor_sync(0xFFFFFFFF, vv, off);
			if (o > vv) vv = o;
		}
		if (lane == 0)
		{
			sWarpM[0] = mm;
			sWarpV[0] = vv;
		}
	}
	__syncthreads();

	const float newMMax = fmaxf(sWarpM[0], 1e-20f);  // guard div-by-zero
	const float newVMax = fmaxf(sWarpV[0], 1e-20f);

	if (threadIdx.x == 0)
	{
		m_scale[blockId] = newMMax;
		v_scale[blockId] = newVMax;
	}

	const float bc1 = 1.0f - powf(beta1, (float)step);
	const float bc2 = 1.0f - powf(beta2, (float)step);
	const float invNewM = 127.0f / newMMax;  // signed int8 spacing
	const float invNewV = 255.0f / newVMax;  // unsigned uint8 spacing

	for (int i = threadIdx.x; i < len; i += blockDim.x)
	{
		const int gi = start + i;
		const float mNew = sM[i];
		const float vNew = sV[i];

		// Requantize: m -> signed int8 [-127, 127]; v -> unsigned uint8 [0, 255].
		// For v, round towards zero ONLY if strictly zero; otherwise clamp up
		// to 1 so that dequantization can never underestimate a nonzero v down
		// to 0 (which would drive 1/√v → 1/eps and blow up the update).
		float mq = mNew * invNewM;
		mq = fmaxf(-127.0f, fminf(127.0f, rintf(mq)));
		m_int8[gi]  = (int8_t)mq;
		if (vNew <= 0.0f)
		{
			v_uint8[gi] = 0;
		}
		else
		{
			float vq = rintf(vNew * invNewV);
			if (vq < 1.0f) vq = 1.0f;      // preserve "nonzero" semantics
			if (vq > 255.0f) vq = 255.0f;
			v_uint8[gi] = (uint8_t)vq;
		}

		// AdamW weight decay on the current param (FP32).
		if (weightDecay != 0.0f)
			param[gi] -= lr * weightDecay * param[gi];

		// Bias-corrected Adam update on param.  We clamp the denominator
		// so a bad-luck quantization of vNew down to ~0 can't drive the
		// step magnitude past a safety threshold.  When vNew was stored
		// with at least code 1 of the uint8 grid, sqrt(vNew) ≥ √(absmax/255)
		// which is always well-behaved; but we still floor at eps to stay
		// symmetric with the FP32 adam_update path.
		const float mHat = mNew / bc1;
		const float vHat = vNew / bc2;
		param[gi] -= lr * mHat / (sqrtf(vHat) + eps);
	}
}

// ---------------------------------------------------------------------------
// Iter 49: fused adam-int8 with BF16-weight + BF16-grad inline I/O.
// Replaces the 4-kernel chain `cast_bf16_to_f32(param) + cast_bf16_to_f32(grad)
// + adam_update_int8_state + cast_f32_to_bf16_stochastic(param)` with one
// kernel that decodes BF16 → FP32 on read, runs the same Adam math, and
// encodes FP32 → BF16 (stochastic, same RNG as cast_f32_to_bf16_stochastic)
// on write.  Eliminates 3 kernel launches and the param/grad/param FP32
// scratch round-trips per param per step.
//
// Cast helper is in another anonymous namespace later in the file; we duplicate
// the sr_hash32 device function here so this kernel can be inlined locally.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t adam_int8_sr_hash32(uint32_t a,
                                                         uint32_t b,
                                                         uint32_t c)
{
	uint32_t x = a ^ (b * 0x9E3779B1u) ^ (c * 0x85EBCA6Bu);
	x ^= x >> 16; x *= 0x7FEB352Du;
	x ^= x >> 15; x *= 0x846CA68Bu;
	x ^= x >> 16;
	return x;
}

__global__ void adam_update_int8_state_bf16w_bf16g_kernel(
    uint16_t* __restrict__ param_bf16,
    const uint16_t* __restrict__ grad_bf16,
    int8_t*  __restrict__ m_int8,
    uint8_t* __restrict__ v_uint8,
    float* __restrict__ m_scale,
    float* __restrict__ v_scale,
    float lr, float beta1, float beta2,
    float eps, float weightDecay,
    float gradScale,
    int step, int n, int numBlocks,
    uint32_t srBaseSeed, uint32_t srStepIdx)
{
	const int blockId = blockIdx.x;
	if (blockId >= numBlocks) return;

	const int start = blockId * ADAM_INT8_BS;
	const int end = min(start + ADAM_INT8_BS, n);
	const int len = end - start;

	const float oldMScale = m_scale[blockId];
	const float oldVScale = v_scale[blockId];
	const float qinvM = 1.0f / 127.0f;
	const float qinvV = 1.0f / 255.0f;

	__shared__ float sM[ADAM_INT8_BS];
	__shared__ float sV[ADAM_INT8_BS];

	float localMmax = 0.0f;
	float localVmax = 0.0f;

	// Pass 1: read bf16 grad, decode inline, compute m/v updates, find absmax.
	for (int i = threadIdx.x; i < len; i += blockDim.x)
	{
		const int gi = start + i;
		// Inline bf16 → fp32 cast for grad.
		union { uint32_t u; float f; } gv;
		gv.u = (uint32_t)grad_bf16[gi] << 16;
		const float g = gv.f * gradScale;

		const float mOld = (float)m_int8[gi]  * oldMScale * qinvM;
		const float vOld = (float)v_uint8[gi] * oldVScale * qinvV;

		const float mNew = beta1 * mOld + (1.0f - beta1) * g;
		const float vNew = beta2 * vOld + (1.0f - beta2) * g * g;

		sM[i] = mNew;
		sV[i] = vNew;

		const float am = fabsf(mNew);
		if (am > localMmax) localMmax = am;
		if (vNew > localVmax) localVmax = vNew;
	}

	// Block-reduce absmax via warp shuffle.
	__shared__ float sWarpM[32], sWarpV[32];
	float mMax = localMmax;
	float vMax = localVmax;
	for (int off = warpSize / 2; off > 0; off /= 2)
	{
		float o = __shfl_xor_sync(0xFFFFFFFF, mMax, off);
		if (o > mMax) mMax = o;
		o = __shfl_xor_sync(0xFFFFFFFF, vMax, off);
		if (o > vMax) vMax = o;
	}
	const int warpId = threadIdx.x / warpSize;
	const int lane = threadIdx.x % warpSize;
	if (lane == 0)
	{
		sWarpM[warpId] = mMax;
		sWarpV[warpId] = vMax;
	}
	__syncthreads();
	if (warpId == 0)
	{
		const int numWarps = blockDim.x / warpSize;
		float mm = (lane < numWarps) ? sWarpM[lane] : 0.0f;
		float vv = (lane < numWarps) ? sWarpV[lane] : 0.0f;
		for (int off = warpSize / 2; off > 0; off /= 2)
		{
			float o = __shfl_xor_sync(0xFFFFFFFF, mm, off);
			if (o > mm) mm = o;
			o = __shfl_xor_sync(0xFFFFFFFF, vv, off);
			if (o > vv) vv = o;
		}
		if (lane == 0)
		{
			sWarpM[0] = mm;
			sWarpV[0] = vv;
		}
	}
	__syncthreads();

	const float newMMax = fmaxf(sWarpM[0], 1e-20f);
	const float newVMax = fmaxf(sWarpV[0], 1e-20f);

	if (threadIdx.x == 0)
	{
		m_scale[blockId] = newMMax;
		v_scale[blockId] = newVMax;
	}

	const float bc1 = 1.0f - powf(beta1, (float)step);
	const float bc2 = 1.0f - powf(beta2, (float)step);
	const float invNewM = 127.0f / newMMax;
	const float invNewV = 255.0f / newVMax;

	// Pass 2: requantize m/v, read bf16 param, update, stochastic encode bf16 param.
	for (int i = threadIdx.x; i < len; i += blockDim.x)
	{
		const int gi = start + i;
		const float mNew = sM[i];
		const float vNew = sV[i];

		// Requantize m → signed int8.
		float mq = mNew * invNewM;
		mq = fmaxf(-127.0f, fminf(127.0f, rintf(mq)));
		m_int8[gi]  = (int8_t)mq;
		// Requantize v → unsigned uint8.
		if (vNew <= 0.0f)
		{
			v_uint8[gi] = 0;
		}
		else
		{
			float vq = rintf(vNew * invNewV);
			if (vq < 1.0f) vq = 1.0f;
			if (vq > 255.0f) vq = 255.0f;
			v_uint8[gi] = (uint8_t)vq;
		}

		// Decode bf16 param inline.
		union { uint32_t u; float f; } pv;
		pv.u = (uint32_t)param_bf16[gi] << 16;
		float pFp = pv.f;

		// AdamW weight decay.
		if (weightDecay != 0.0f)
			pFp -= lr * weightDecay * pFp;

		// Bias-corrected Adam update.
		const float mHat = mNew / bc1;
		const float vHat = vNew / bc2;
		pFp -= lr * mHat / (sqrtf(vHat) + eps);

		// Stochastic-rounded fp32 → bf16 encode (same RNG as cast_f32_to_bf16_stochastic).
		union { float f; uint32_t u; } v;
		v.f = pFp;
		if (isnan(pFp))
		{
			const uint32_t sign = v.u & 0x80000000u;
			param_bf16[gi] = (uint16_t)(((sign | 0x7FC00000u) >> 16) & 0xFFFFu);
		}
		else
		{
			const uint32_t low16 = v.u & 0xFFFFu;
			const uint32_t rnd = adam_int8_sr_hash32((uint32_t)gi, srStepIdx, srBaseSeed) & 0xFFFFu;
			uint32_t high16 = v.u >> 16;
			if (rnd < low16) high16 += 1u;
			param_bf16[gi] = (uint16_t)(high16 & 0xFFFFu);
		}
	}
}

} // anonymous namespace

bool adam_update_int8_state(float* param, const float* grad,
                             int8_t* m_int8, uint8_t* v_uint8,
                             float* m_scale, float* v_scale,
                             float lr, float beta1, float beta2, float eps,
                             float weightDecay, float gradScale,
                             int step, int n)
{
	if (n <= 0) return true;
	const int numBlocks = (n + ADAM_INT8_BS - 1) / ADAM_INT8_BS;
	adam_update_int8_state_kernel<<<numBlocks, ADAM_INT8_BS, 0, computeStream()>>>(
	    param, grad, m_int8, v_uint8, m_scale, v_scale,
	    lr, beta1, beta2, eps, weightDecay, gradScale, step, n, numBlocks);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// BF16-grad variant: caller provides a scratch FP32 buffer (size >= n).
// Casts BF16 → FP32 once into the scratch, then dispatches the existing
// int8 kernel.  Avoids the ~150-LOC duplication of the int8 kernel itself.
// Compute cost: one extra cast pass; bandwidth-bound so usually free
// alongside the Adam step which is also bandwidth-bound.
bool adam_update_int8_state_bf16grad(float* param, const uint16_t* grad_bf16,
                                      int8_t* m_int8, uint8_t* v_uint8,
                                      float* m_scale, float* v_scale,
                                      float* scratch_fp32,
                                      float lr, float beta1, float beta2, float eps,
                                      float weightDecay, float gradScale,
                                      int step, int n)
{
	if (n <= 0) return true;
	if (scratch_fp32 == 0 || grad_bf16 == 0) return false;
	if (!cast_bf16_to_f32(grad_bf16, scratch_fp32, (size_t)n)) return false;
	return adam_update_int8_state(param, scratch_fp32, m_int8, v_uint8,
	                              m_scale, v_scale, lr, beta1, beta2, eps,
	                              weightDecay, gradScale, step, n);
}

// BF16-WEIGHT variants — wrapper kernels for CHIRON-style --bf16-weights mode.
// param is a bf16 buffer (no FP32 master).  Per Adam step:
//   1. cast bf16(param) -> weight_scratch_fp32
//   2. existing adam kernel mutates weight_scratch_fp32 in place
//   3. cast weight_scratch_fp32 -> bf16(param) with stochastic rounding
//        (seed = srBaseSeed XOR (srStepIdx * 0x9E3779B1))
// Caller supplies stochastic-round seed + step counter so per-(model,step,
// tensor) randomness is deterministic and reproducible.

bool adam_update_bf16_state_bf16grad_bf16w(uint16_t* param_bf16,
                                            float* weight_scratch_fp32,
                                            const uint16_t* grad_bf16,
                                            uint16_t* m_bf16, uint16_t* v_bf16,
                                            float lr, float beta1, float beta2, float eps,
                                            float weightDecay, float gradScale,
                                            int step, int n,
                                            uint32_t srBaseSeed, uint32_t srStepIdx)
{
	if (n <= 0) return true;
	if (param_bf16 == 0 || weight_scratch_fp32 == 0 || grad_bf16 == 0) return false;
	if (!cast_bf16_to_f32(param_bf16, weight_scratch_fp32, (size_t)n)) return false;
	if (!adam_update_bf16_state_bf16grad(weight_scratch_fp32, grad_bf16,
	                                     m_bf16, v_bf16,
	                                     lr, beta1, beta2, eps,
	                                     weightDecay, gradScale, step, n))
		return false;
	return cast_f32_to_bf16_stochastic(weight_scratch_fp32, param_bf16, (size_t)n,
	                                    srBaseSeed, srStepIdx);
}

// Iter 49 fused: int8 Adam with BF16-weight + BF16-grad direct I/O.
// Replaces the 4-kernel chain (cast bf16→fp32 param, cast bf16→fp32 grad,
// adam_update_int8_state, cast fp32→bf16 stochastic param) with one kernel.
// Bit-equivalent to the chain modulo fp32 reduction-order in the absmax
// reduce; same stochastic rounding RNG (sr_hash32) keyed on (idx, srStepIdx,
// srBaseSeed) so training trajectory is bit-identical to the unfused path
// modulo sub-ULP FMA ordering.
bool adam_update_int8_state_bf16w_bf16g_fused(uint16_t* param_bf16,
                                               const uint16_t* grad_bf16,
                                               int8_t* m_int8, uint8_t* v_uint8,
                                               float* m_scale, float* v_scale,
                                               float lr, float beta1, float beta2, float eps,
                                               float weightDecay, float gradScale,
                                               int step, int n,
                                               uint32_t srBaseSeed, uint32_t srStepIdx)
{
	if (n <= 0) return true;
	if (param_bf16 == 0 || grad_bf16 == 0) return false;
	const int numBlocks = (n + ADAM_INT8_BS - 1) / ADAM_INT8_BS;
	adam_update_int8_state_bf16w_bf16g_kernel<<<numBlocks, ADAM_INT8_BS, 0, computeStream()>>>(
	    param_bf16, grad_bf16, m_int8, v_uint8, m_scale, v_scale,
	    lr, beta1, beta2, eps, weightDecay, gradScale, step, n, numBlocks,
	    srBaseSeed, srStepIdx);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool adam_update_int8_state_bf16grad_bf16w(uint16_t* param_bf16,
                                            float* weight_scratch_fp32,
                                            const uint16_t* grad_bf16,
                                            int8_t* m_int8, uint8_t* v_uint8,
                                            float* m_scale, float* v_scale,
                                            float* grad_scratch_fp32,
                                            float lr, float beta1, float beta2, float eps,
                                            float weightDecay, float gradScale,
                                            int step, int n,
                                            uint32_t srBaseSeed, uint32_t srStepIdx)
{
	if (n <= 0) return true;
	if (param_bf16 == 0 || weight_scratch_fp32 == 0 || grad_bf16 == 0) return false;
	if (grad_scratch_fp32 == 0) return false;
	if (!cast_bf16_to_f32(param_bf16, weight_scratch_fp32, (size_t)n)) return false;
	if (!adam_update_int8_state_bf16grad(weight_scratch_fp32, grad_bf16,
	                                     m_int8, v_uint8, m_scale, v_scale,
	                                     grad_scratch_fp32,
	                                     lr, beta1, beta2, eps,
	                                     weightDecay, gradScale, step, n))
		return false;
	return cast_f32_to_bf16_stochastic(weight_scratch_fp32, param_bf16, (size_t)n,
	                                    srBaseSeed, srStepIdx);
}

// Number of scale blocks (one FP32 scale per block of ADAM_INT8_BS params).
int adam_int8_scale_count(int n)
{
	return (n + ADAM_INT8_BS - 1) / ADAM_INT8_BS;
}

// Batched Adam: process all parameter groups in a single kernel launch.
// Each block handles one element range within one parameter group.
namespace {

__global__ void adam_group_scale_batch_kernel(
    float** __restrict__ params,
    float** __restrict__ grads,
    float** __restrict__ ms,
    float** __restrict__ vs,
    float* __restrict__ groupScales,
    float* __restrict__ prevStepRms,
    const int* __restrict__ sizes,
    float beta1, float beta2, float eps,
    float gradScale, int step, int groupCount,
    unsigned int minGroupSize,
    float stabilityScale, float snrScale, float ratioScale,
    float minScale, float maxScale)
{
	int grp = blockIdx.x;
	if (grp >= groupCount)
		return;

	const int n = sizes[grp];
	if (n <= 0)
		return;
	if (static_cast<unsigned int>(n) < minGroupSize)
	{
		if (threadIdx.x == 0)
			groupScales[grp] = 1.0f;
		return;
	}

	float* param = params[grp];
	float* grad = grads[grp];
	float* m_arr = ms[grp];
	float* v_arr = vs[grp];

	__shared__ float sharedStepSq[8];
	__shared__ float sharedWeightSq[8];
	__shared__ float sharedMHatSq[8];
	__shared__ float sharedVHat[8];

	float stepSq = 0.0f;
	float weightSq = 0.0f;
	float mHatSq = 0.0f;
	float vHatSum = 0.0f;

	const float bc1 = 1.0f - powf(beta1, (float)step);
	const float bc2 = 1.0f - powf(beta2, (float)step);
	for (int idx = threadIdx.x; idx < n; idx += blockDim.x)
	{
		const float g = grad[idx] * gradScale;
		const float m_new = beta1 * m_arr[idx] + (1.0f - beta1) * g;
		const float v_new = beta2 * v_arr[idx] + (1.0f - beta2) * g * g;
		const float m_hat = m_new / bc1;
		const float v_hat = v_new / bc2;
		const float denom = sqrtf(v_hat) + eps;
		const float stepVal = m_hat / denom;
		stepSq += stepVal * stepVal;
		const float w = param[idx];
		weightSq += w * w;
		mHatSq += m_hat * m_hat;
		vHatSum += v_hat;
	}

	stepSq = blockReduceSum(stepSq, sharedStepSq);
	weightSq = blockReduceSum(weightSq, sharedWeightSq);
	mHatSq = blockReduceSum(mHatSq, sharedMHatSq);
	vHatSum = blockReduceSum(vHatSum, sharedVHat);

	if (threadIdx.x == 0)
	{
		const float invN = 1.0f / static_cast<float>(n);
		const float stepRms = sqrtf(fmaxf(stepSq * invN, 0.0f));
		const float weightRms = sqrtf(fmaxf(weightSq * invN, 0.0f));
		const float snr = (mHatSq * invN) / ((vHatSum * invN) + eps);
		const float prev = prevStepRms[grp];
		const float stability = (prev > 0.0f)
		    ? (fminf(stepRms, prev) / (fmaxf(stepRms, prev) + eps))
		    : 1.0f;
		const float updateRatio = stepRms / (weightRms + eps);
		float scale = 1.0f
		    + stabilityScale * stability
		    + snrScale * log1pf(fmaxf(snr, 0.0f))
		    - ratioScale * updateRatio;
		scale = fminf(maxScale, fmaxf(minScale, scale));
		groupScales[grp] = scale;
		prevStepRms[grp] = stepRms;
	}
}

} // anonymous namespace

bool adam_group_scale_batch(float** d_params, float** d_grads,
                            float** d_ms, float** d_vs,
                            float* d_groupScales, float* d_groupPrevStepRms,
                            const int* d_sizes,
                            float beta1, float beta2, float eps,
                            float gradScale, int step, int groupCount,
                            unsigned int minGroupSize,
                            float stabilityScale, float snrScale, float ratioScale,
                            float minScale, float maxScale)
{
	if (groupCount <= 0)
		return true;
	adam_group_scale_batch_kernel<<<groupCount, kBlockElem, 0, computeStream()>>>(
	    d_params, d_grads, d_ms, d_vs,
	    d_groupScales, d_groupPrevStepRms, d_sizes,
	    beta1, beta2, eps, gradScale, step, groupCount,
	    minGroupSize, stabilityScale, snrScale, ratioScale, minScale, maxScale);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

namespace {

__global__ void adam_update_batch_kernel(
    float** __restrict__ params,
    float** __restrict__ grads,
    float** __restrict__ ms,
    float** __restrict__ vs,
    const float* __restrict__ baseLrs,
    const float* __restrict__ wds,
    float lrScale,
    const float* __restrict__ stepScales,
    const int* __restrict__ sizes,
    float** __restrict__ rowMetrics,
    float** __restrict__ colMetrics,
    float** __restrict__ rowStructMetrics,
    float** __restrict__ colStructMetrics,
    float** __restrict__ prevMhats,
    float** __restrict__ metricScratch,
    const int* __restrict__ metricRows,
    const int* __restrict__ metricCols,
    float beta1, float beta2, float eps,
    float gradScale, int step, int groupCount)
{
	int grp = blockIdx.y;
	if (grp >= groupCount) return;

	int n = sizes[grp];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	float* param = params[grp];
	float* grad = grads[grp];
	float* m_arr = ms[grp];
	float* v_arr = vs[grp];
	const float baseLr = baseLrs[grp] * lrScale;
	float lr = baseLr * (stepScales ? stepScales[grp] : 1.0f);
	float weightDecay = wds[grp];
	float* rowMetric = rowMetrics ? rowMetrics[grp] : NULL;
	float* colMetric = colMetrics ? colMetrics[grp] : NULL;
	float* rowStructMetric = rowStructMetrics ? rowStructMetrics[grp] : NULL;
	float* colStructMetric = colStructMetrics ? colStructMetrics[grp] : NULL;
	float* prevMhat = prevMhats ? prevMhats[grp] : NULL;
	float* metricStats = metricScratch ? metricScratch[grp] : NULL;
	const int rows = metricRows ? metricRows[grp] : 0;
	const int cols = metricCols ? metricCols[grp] : 0;
	const bool useMatrixMetric =
	    rowMetric && colMetric && rows > 0 && cols > 0 && (rows * cols) == n;
	const float predictiveWeight =
	    (useMatrixMetric && prevMhat && metricStats && step > 1)
	        ? fmaxf(metricStats[9], 0.0f)
	        : 0.0f;
	const float structuralWeight =
	    (useMatrixMetric && metricStats) ? fmaxf(metricStats[10], 0.0f) : 0.0f;
	const float baseWeight =
	    useMatrixMetric ? fmaxf(0.0f, 1.0f - predictiveWeight - structuralWeight) : 1.0f;

	float g = grad[idx] * gradScale;

	if (weightDecay != 0.0f)
		param[idx] -= lr * weightDecay * param[idx];

	float m_new = beta1 * m_arr[idx] + (1.0f - beta1) * g;
	float v_new = beta2 * v_arr[idx] + (1.0f - beta2) * g * g;
	m_arr[idx] = m_new;
	v_arr[idx] = v_new;

	float bc1 = 1.0f - powf(beta1, (float)step);
	float bc2 = 1.0f - powf(beta2, (float)step);
	float m_hat = m_new / bc1;
	float v_hat = v_new / bc2;
	float predictiveMhat = m_hat;
	if (predictiveWeight > 0.0f)
	{
		float delta = m_hat - prevMhat[idx];
		const float deltaCap = 0.5f * (fabsf(m_hat) + eps);
		if (delta > deltaCap)
			delta = deltaCap;
		else if (delta < -deltaCap)
			delta = -deltaCap;
		predictiveMhat += delta;
	}
	const float diagInv = 1.0f / (sqrtf(v_hat) + eps);
	if (prevMhat)
		prevMhat[idx] = m_hat;
	if (useMatrixMetric)
	{
		const int row = idx / cols;
		const int col = idx - row * cols;
		const float baseMetric = rowMetric[row] * colMetric[col];
		const float structMetric =
		    (rowStructMetric && colStructMetric)
		        ? (rowStructMetric[row] * colStructMetric[col])
		        : baseMetric;
		const float baseStep = m_hat * diagInv * baseMetric;
		const float predictiveStep = predictiveMhat * diagInv * baseMetric;
		const float structuralStep = m_hat * diagInv * structMetric;
		const float stepVal =
		    baseWeight * baseStep
		    + predictiveWeight * predictiveStep
		    + structuralWeight * structuralStep;
		param[idx] -= lr * stepVal;
		grad[idx] = 0.0f;
		return;
	}
	const float stepVal = m_hat * diagInv;
	param[idx] -= lr * stepVal;
}

// Sophia-G batched variant.  Same buffer-of-pointers layout as
// adam_update_batch_kernel but with the Sophia clipped second-order rule
// instead of Adam's m/sqrt(v).  No ECHO metric scaling — meant as an
// drop-in replacement that preserves the per-group learning rate +
// weight-decay + bias-correction semantics.  Param 'h' replaces Adam's
// 'v' (same shape, same buffers reused — gradient-squared EMA).
__global__ void sophia_g_update_batch_kernel(
    float** __restrict__ params,
    float** __restrict__ grads,
    float** __restrict__ ms,
    float** __restrict__ hs,
    const float* __restrict__ baseLrs,
    const float* __restrict__ wds,
    float lrScale,
    const int* __restrict__ sizes,
    float beta1, float beta2,
    float gamma, float rho, float eps,
    float gradScale, int step, int groupCount)
{
	int grp = blockIdx.y;
	if (grp >= groupCount) return;

	int n = sizes[grp];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	float* param = params[grp];
	float* grad = grads[grp];
	float* m_arr = ms[grp];
	float* h_arr = hs[grp];
	const float baseLr = baseLrs[grp] * lrScale;
	float weightDecay = wds[grp];

	float g = grad[idx] * gradScale;

	// Decoupled weight decay (AdamW-style).
	if (weightDecay != 0.0f)
		param[idx] -= baseLr * weightDecay * param[idx];

	float m_new = beta1 * m_arr[idx] + (1.0f - beta1) * g;
	float h_new = beta2 * h_arr[idx] + (1.0f - beta2) * g * g;
	m_arr[idx] = m_new;
	h_arr[idx] = h_new;

	float bc1 = 1.0f - powf(beta1, (float)step);
	float bc2 = 1.0f - powf(beta2, (float)step);
	float m_hat = m_new / bc1;
	float h_hat = h_new / bc2;

	float denom = fmaxf(gamma * h_hat, eps);
	float ratio = m_hat / denom;
	if (ratio > rho) ratio = rho;
	else if (ratio < -rho) ratio = -rho;

	param[idx] -= baseLr * ratio;
	grad[idx] = 0.0f;  // mirror adam_update_batch's grad clear
}

} // anonymous namespace

bool sophia_g_update_batch(float** d_params, float** d_grads,
                            float** d_ms, float** d_hs,
                            const float* d_baseLrs, const float* d_wds,
                            float lrScale, const int* d_sizes, int maxSize,
                            float beta1, float beta2, float gamma, float rho,
                            float eps, float gradScale, int step, int groupCount)
{
	if (groupCount <= 0 || maxSize <= 0) return true;
	dim3 block(kBlockElem, 1, 1);
	dim3 grid((maxSize + kBlockElem - 1) / kBlockElem, groupCount, 1);
	sophia_g_update_batch_kernel<<<grid, block, 0, computeStream()>>>(
	    d_params, d_grads, d_ms, d_hs,
	    d_baseLrs, d_wds, lrScale, d_sizes,
	    beta1, beta2, gamma, rho, eps,
	    gradScale, step, groupCount);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool adam_update_batch(float** d_params, float** d_grads,
                       float** d_ms, float** d_vs,
                       const float* d_baseLrs, const float* d_wds,
                       float lrScale,
                       const float* d_stepScales,
                       const int* d_sizes, int maxSize,
                       float** d_rowMetrics, float** d_colMetrics,
                       float** d_rowStructMetrics, float** d_colStructMetrics,
                       float** d_prevMhats, float** d_metricScratch,
                       const int* d_metricRows, const int* d_metricCols,
                       float beta1, float beta2, float eps,
                       float gradScale, int step, int groupCount)
{
	if (groupCount <= 0) return true;
	int gridX = (maxSize + kBlockElem - 1) / kBlockElem;
	dim3 grid(gridX, groupCount);
	adam_update_batch_kernel<<<grid, kBlockElem, 0, computeStream()>>>(
		d_params, d_grads, d_ms, d_vs,
		d_baseLrs, d_wds, lrScale, d_stepScales, d_sizes,
		d_rowMetrics, d_colMetrics, d_rowStructMetrics, d_colStructMetrics,
		d_prevMhats, d_metricScratch,
		d_metricRows, d_metricCols,
		beta1, beta2, eps, gradScale, step, groupCount);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  15a. reduce_rows_sum
// ===========================================================================

namespace {

// One block per column.  Threads cooperatively sum across all rows.
__global__ void reduce_rows_sum_kernel(const float* __restrict__ input,
                                       int rows, int cols,
                                       float beta,
                                       float* __restrict__ out)
{
    int col = blockIdx.x;
    if (col >= cols) return;

    extern __shared__ float smem[];

    float localSum = 0.0f;
    for (int r = threadIdx.x; r < rows; r += blockDim.x)
        localSum += input[(size_t)r * cols + col];
    localSum = blockReduceSum(localSum, smem);

    if (threadIdx.x == 0)
    {
        if (beta == 0.0f)
            out[col] = localSum;
        else
            out[col] = beta * out[col] + localSum;
    }
}

} // anonymous namespace

bool reduce_rows_sum(const float* input, int rows, int cols,
                     float beta, float* out)
{
    if (rows <= 0 || cols <= 0) return true;
    int block = rowBlockSize(rows);
    int smemBytes = (block / 32 + 1) * sizeof(float);
    reduce_rows_sum_kernel<<<cols, block, smemBytes, computeStream()>>>(input, rows, cols, beta, out);
    GLADES_CUDA_CHECK(cudaGetLastError());
    return true;
}

// ===========================================================================
//  15b. Causal-masked softmax (in-place, for attention scores)
// ===========================================================================

namespace {

// One block per (batch, row).  S is [batchSize, T, T] row-major.
__global__ void causal_mask_softmax_kernel(float* __restrict__ S,
                                           int T)
{
    int idx = blockIdx.x;  // linear index into (batch, row)
    int row = idx % T;
    float* sRow = S + (size_t)idx * T;

    extern __shared__ float smem[];
    float* sMax = smem;
    float* sSum = smem + (blockDim.x / 32 + 1);

    // Apply causal mask: set j > row to -FLT_MAX.
    for (int j = threadIdx.x; j < T; j += blockDim.x)
    {
        if (j > row)
            sRow[j] = -FLT_MAX;
    }
    __syncthreads();

    // Pass 1: row max.
    float localMax = -FLT_MAX;
    for (int j = threadIdx.x; j < T; j += blockDim.x)
        localMax = fmaxf(localMax, sRow[j]);
    localMax = blockReduceMax(localMax, sMax);

    __shared__ float sRowMax;
    if (threadIdx.x == 0) sRowMax = localMax;
    __syncthreads();
    float rowMax = sRowMax;

    // Pass 2: sum of exp(x - max).
    float localSum = 0.0f;
    for (int j = threadIdx.x; j < T; j += blockDim.x)
    {
        float e = expf(sRow[j] - rowMax);
        sRow[j] = e;
        localSum += e;
    }
    localSum = blockReduceSum(localSum, sSum);

    __shared__ float sRowSum;
    if (threadIdx.x == 0) sRowSum = localSum;
    __syncthreads();
    float invSum = 1.0f / sRowSum;

    // Pass 3: normalize.
    for (int j = threadIdx.x; j < T; j += blockDim.x)
        sRow[j] *= invSum;
}

} // anonymous namespace

bool causal_mask_softmax_inplace(float* S, int batchSize, int T)
{
    if (batchSize <= 0 || T <= 0) return true;
    int totalRows = batchSize * T;
    int block = rowBlockSize(T);
    int smemBytes = (block / 32 + 2) * 2 * sizeof(float);
    causal_mask_softmax_kernel<<<totalRows, block, smemBytes, computeStream()>>>(S, T);
    GLADES_CUDA_CHECK(cudaGetLastError());
    return true;
}

// ===========================================================================
//  15c. Softmax backward for attention
// ===========================================================================

namespace {

// dS = P * (dP - row_sum(dP * P)), zero above diagonal.
// One block per (batch, row).
__global__ void softmax_backward_attn_kernel(const float* __restrict__ P,
                                              const float* __restrict__ dP,
                                              int T,
                                              float outputScale,
                                              float* __restrict__ dS)
{
    int idx = blockIdx.x;
    int row = idx % T;
    const float* pRow  = P  + (size_t)idx * T;
    const float* dpRow = dP + (size_t)idx * T;
    float* dsRow       = dS + (size_t)idx * T;

    extern __shared__ float smem[];

    // Compute dot = sum_j(dP[j] * P[j]) for valid positions (j <= row).
    float localDot = 0.0f;
    for (int j = threadIdx.x; j <= row; j += blockDim.x)
        localDot += dpRow[j] * pRow[j];
    localDot = blockReduceSum(localDot, smem);

    __shared__ float sDot;
    if (threadIdx.x == 0) sDot = localDot;
    __syncthreads();
    float dot = sDot;

    // dS[j] = outputScale * P[j] * (dP[j] - dot) for j <= row, 0 otherwise.
    for (int j = threadIdx.x; j < T; j += blockDim.x)
    {
        if (j <= row)
            dsRow[j] = outputScale * pRow[j] * (dpRow[j] - dot);
        else
            dsRow[j] = 0.0f;
    }
}

} // anonymous namespace

bool softmax_backward_attn(const float* P, const float* dP,
                           int batchSize, int T, float outputScale, float* dS)
{
    if (batchSize <= 0 || T <= 0) return true;
    int totalRows = batchSize * T;
    int block = rowBlockSize(T);
    int smemBytes = (block / 32 + 1) * sizeof(float);
    softmax_backward_attn_kernel<<<totalRows, block, smemBytes, computeStream()>>>(P, dP, T, outputScale, dS);
    GLADES_CUDA_CHECK(cudaGetLastError());
    return true;
}

// ===========================================================================
//  15d. Cross-entropy NLL loss
// ===========================================================================

namespace {

// One block total (simple reduction).  Each thread handles a subset of rows.
__global__ void cross_entropy_nll_kernel(const float* __restrict__ probs,
                                         const int* __restrict__ targets,
                                         int T, int vocabSize, int padToken,
                                         float* __restrict__ loss_sum,
                                         int* __restrict__ valid_count)
{
    extern __shared__ float smem[];
    float* sLoss = smem;

    float localLoss = 0.0f;
    int localCount = 0;
    for (int t = threadIdx.x; t < T; t += blockDim.x)
    {
        int tgt = targets[t];
        if (padToken >= 0 && tgt == padToken) continue;
        if (tgt < 0 || tgt >= vocabSize) continue;
        float p = probs[(size_t)t * vocabSize + tgt];
        if (p < 1e-12f) p = 1e-12f;
        localLoss += -logf(p);
        ++localCount;
    }

    // Reduce loss.
    localLoss = blockReduceSum(localLoss, sLoss);
    if (threadIdx.x == 0)
        atomicAdd(loss_sum, localLoss);

    // Reduce count (reuse warp reduce as floats, cast back).
    __syncthreads();
    float countF = (float)localCount;
    countF = blockReduceSum(countF, sLoss);
    if (threadIdx.x == 0)
        atomicAdd(valid_count, (int)countF);
}

} // anonymous namespace

bool cross_entropy_nll_loss(const float* probs, const int* targets,
                            int T, int vocabSize, int padToken,
                            float* loss_sum, int* valid_count)
{
    if (T <= 0 || vocabSize <= 0) return true;
    GLADES_CUDA_CHECK(cudaMemset(loss_sum, 0, sizeof(float)));
    GLADES_CUDA_CHECK(cudaMemset(valid_count, 0, sizeof(int)));
    int block = 256;
    int grid = 1;
    if (T > 256) { grid = (T + block - 1) / block; if (grid > 128) grid = 128; }
    int smemBytes = (block / 32 + 2) * sizeof(float) + (block / 32 + 2) * sizeof(int);
    cross_entropy_nll_kernel<<<grid, block, smemBytes, computeStream()>>>(
        probs, targets, T, vocabSize, padToken, loss_sum, valid_count);
    GLADES_CUDA_CHECK(cudaGetLastError());
    return true;
}

// ===========================================================================
//  15e. Argmax accuracy
// ===========================================================================

namespace {

__global__ void argmax_count_kernel(const float* __restrict__ probs,
                                    const int* __restrict__ targets,
                                    int T, int vocabSize, int padToken,
                                    int* __restrict__ correct_count,
                                    int* __restrict__ valid_count)
{
    extern __shared__ float smem[];

    int localCorrect = 0;
    int localValid = 0;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < T;
         t += blockDim.x * gridDim.x)
    {
        int tgt = targets[t];
        if (padToken >= 0 && tgt == padToken) continue;
        if (tgt < 0 || tgt >= vocabSize) continue;
        ++localValid;

        const float* row = probs + (size_t)t * vocabSize;
        int bestIdx = 0;
        float bestVal = row[0];
        for (int v = 1; v < vocabSize; ++v)
        {
            if (row[v] > bestVal) { bestVal = row[v]; bestIdx = v; }
        }
        if (bestIdx == tgt) ++localCorrect;
    }

    float correctF = (float)localCorrect;
    correctF = blockReduceSum(correctF, smem);
    if (threadIdx.x == 0) atomicAdd(correct_count, (int)correctF);

    __syncthreads();
    float validF = (float)localValid;
    validF = blockReduceSum(validF, smem);
    if (threadIdx.x == 0) atomicAdd(valid_count, (int)validF);
}

} // anonymous namespace

bool argmax_count_matches(const float* probs, const int* targets,
                          int T, int vocabSize, int padToken,
                          int* correct_count, int* valid_count)
{
    if (T <= 0 || vocabSize <= 0) return true;
    GLADES_CUDA_CHECK(cudaMemset(correct_count, 0, sizeof(int)));
    GLADES_CUDA_CHECK(cudaMemset(valid_count, 0, sizeof(int)));
    // One thread per row for argmax (vocabSize may be large).
    int block = 128;
    int grid = (T + block - 1) / block;
    if (grid > 128) grid = 128;
    int smemBytes = (block / 32 + 1) * sizeof(float);
    argmax_count_kernel<<<grid, block, smemBytes, computeStream()>>>(
        probs, targets, T, vocabSize, padToken, correct_count, valid_count);
    GLADES_CUDA_CHECK(cudaGetLastError());
    return true;
}

// ===========================================================================
//  15e2. BF16-storage variants for the readout loss path (ralph-loop iter 10)
// ===========================================================================
//
// --bf16-logits-storage routes the three T·V buffers (logits, probs, dlogits)
// as BF16 (uint16_t) rather than FP32.  At T=16384 V=32000 this saves
// 3 × 2 GB = 6 GB → 3 × 1 GB = 3 GB, unlocking T=16384 on 16 GB hardware.
//
// All BF16 storage kernels cast to FP32 on load, do math in FP32, cast back
// on store (RNE rounding via __float2bfloat16).  Probs precision is the
// concern — softmax outputs sum to 1.0f, so each prob is in [0, 1] and the
// BF16 mantissa floor at ~3.9e-3 means values below that round to 0.  NLL
// loss already floors at 1e-12 (CE_KERNEL above) — the BF16 path applies
// the same floor AFTER cast, so the masking is identical.

namespace {

__device__ __forceinline__ float bf16_load(unsigned short bits)
{
	return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
}

__device__ __forceinline__ unsigned short bf16_store(float val)
{
	__nv_bfloat16 b = __float2bfloat16(val);
	return *reinterpret_cast<unsigned short*>(&b);
}

// BF16-in / BF16-out softmax.  3 passes per row: (1) max over BF16-loaded
// values, (2) sum of exp(x - max), (3) store exp(x - max) / sum as BF16.
// Pass 3 recomputes exp (vs the FP32 path's "store intermediate exp" trick)
// to avoid a BF16 round-trip on the intermediate exp.  Bandwidth identical
// because the BF16 output is half the bytes of FP32 output.
__global__ void softmax_stable_rows_bf16(const unsigned short* __restrict__ xb,
                                          int cols,
                                          unsigned short* __restrict__ ob)
{
	int row = blockIdx.x;
	const unsigned short* xRow = xb + (size_t)row * cols;
	unsigned short*       oRow = ob + (size_t)row * cols;

	extern __shared__ float smem[];
	float* sMax = smem;
	float* sSum = smem + (blockDim.x / 32 + 1);

	float localMax = -FLT_MAX;
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		localMax = fmaxf(localMax, bf16_load(xRow[i]));
	localMax = blockReduceMax(localMax, sMax);

	__shared__ float sRowMax;
	if (threadIdx.x == 0) sRowMax = localMax;
	__syncthreads();
	float rowMax = sRowMax;

	float localSum = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		localSum += expf(bf16_load(xRow[i]) - rowMax);
	localSum = blockReduceSum(localSum, sSum);

	__shared__ float sRowSum;
	if (threadIdx.x == 0) sRowSum = localSum;
	__syncthreads();
	float invSum = 1.0f / sRowSum;

	for (int i = threadIdx.x; i < cols; i += blockDim.x)
		oRow[i] = bf16_store(expf(bf16_load(xRow[i]) - rowMax) * invSum);
}

__global__ void softmax_cross_entropy_backward_bf16(
    const unsigned short* __restrict__ probs,
    const int* __restrict__ targets,
    int cols,
    unsigned short* __restrict__ dlogits)
{
	int row = blockIdx.x;
	int target = targets[row];
	const unsigned short* pRow = probs   + (size_t)row * cols;
	unsigned short*       dRow = dlogits + (size_t)row * cols;

	for (int i = threadIdx.x; i < cols; i += blockDim.x)
	{
		float p = bf16_load(pRow[i]);
		float v = (i == target) ? (p - 1.0f) : p;
		dRow[i] = bf16_store(v);
	}
}

__global__ void scale_array_bf16_kernel(unsigned short* __restrict__ x,
                                         float scale, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) x[idx] = bf16_store(bf16_load(x[idx]) * scale);
}

__global__ void cross_entropy_nll_bf16_kernel(
    const unsigned short* __restrict__ probs,
    const int* __restrict__ targets,
    int T, int vocabSize, int padToken,
    float* __restrict__ loss_sum,
    int* __restrict__ valid_count)
{
	extern __shared__ float smem[];
	float* sLoss = smem;

	float localLoss = 0.0f;
	int localCount = 0;
	for (int t = threadIdx.x; t < T; t += blockDim.x)
	{
		int tgt = targets[t];
		if (padToken >= 0 && tgt == padToken) continue;
		if (tgt < 0 || tgt >= vocabSize) continue;
		float p = bf16_load(probs[(size_t)t * vocabSize + tgt]);
		if (p < 1e-12f) p = 1e-12f;
		localLoss += -logf(p);
		++localCount;
	}

	localLoss = blockReduceSum(localLoss, sLoss);
	if (threadIdx.x == 0)
		atomicAdd(loss_sum, localLoss);

	__syncthreads();
	float countF = (float)localCount;
	countF = blockReduceSum(countF, sLoss);
	if (threadIdx.x == 0)
		atomicAdd(valid_count, (int)countF);
}

// Legacy 1-thread-per-row argmax — kept as dead code; iter 56/60 ships the
// warp-parallel variant below.
__global__ void argmax_count_bf16_kernel(
    const unsigned short* __restrict__ probs,
    const int* __restrict__ targets,
    int T, int vocabSize, int padToken,
    int* __restrict__ correct_count,
    int* __restrict__ valid_count)
{
	extern __shared__ float smem[];

	int localCorrect = 0;
	int localValid = 0;
	for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < T;
	     t += blockDim.x * gridDim.x)
	{
		int tgt = targets[t];
		if (padToken >= 0 && tgt == padToken) continue;
		if (tgt < 0 || tgt >= vocabSize) continue;
		++localValid;

		const unsigned short* row = probs + (size_t)t * vocabSize;
		int bestIdx = 0;
		float bestVal = bf16_load(row[0]);
		for (int v = 1; v < vocabSize; ++v)
		{
			float pv = bf16_load(row[v]);
			if (pv > bestVal) { bestVal = pv; bestIdx = v; }
		}
		if (bestIdx == tgt) ++localCorrect;
	}

	float correctF = (float)localCorrect;
	correctF = blockReduceSum(correctF, smem);
	if (threadIdx.x == 0) atomicAdd(correct_count, (int)correctF);

	__syncthreads();
	float validF = (float)localValid;
	validF = blockReduceSum(validF, smem);
	if (threadIdx.x == 0) atomicAdd(valid_count, (int)validF);
}

// Iter 56 / Iter 60: warp-parallel argmax — 1 warp per row.  Each warp's
// 32 lanes do a strided scan over V (lane handles v=lane, lane+32, ...),
// then a 5-step __shfl_down_sync reduction finds the global max+arg.
// Block: 32 warps × 32 lanes = 1024 threads = 32 rows per block.  Replaces
// the 1-thread-per-row V-loop pattern of the legacy kernel.  Train-accuracy
// values bit-identical; the kernel is logging-only, no NLL impact.
__global__ void argmax_count_bf16_warp_kernel(
    const unsigned short* __restrict__ probs,
    const int* __restrict__ targets,
    int T, int vocabSize, int padToken,
    int* __restrict__ correct_count,
    int* __restrict__ valid_count)
{
	const int warpsPerBlock = blockDim.x / 32;
	const int lane          = threadIdx.x & 31;
	const int warpId        = threadIdx.x >> 5;
	const int row           = blockIdx.x * warpsPerBlock + warpId;
	if (row >= T) return;

	int tgt = targets[row];
	bool isValid = (padToken < 0 || tgt != padToken) && (tgt >= 0 && tgt < vocabSize);

	int   bestIdx = -1;
	float bestVal = -1e38f;
	if (isValid)
	{
		const unsigned short* prow = probs + (size_t)row * vocabSize;
		for (int v = lane; v < vocabSize; v += 32)
		{
			float pv = bf16_load(prow[v]);
			if (pv > bestVal) { bestVal = pv; bestIdx = v; }
		}
		for (int off = 16; off > 0; off >>= 1)
		{
			float oVal = __shfl_down_sync(0xFFFFFFFF, bestVal, off);
			int   oIdx = __shfl_down_sync(0xFFFFFFFF, bestIdx, off);
			if (oVal > bestVal) { bestVal = oVal; bestIdx = oIdx; }
		}
	}

	if (lane == 0)
	{
		if (isValid)
		{
			atomicAdd(valid_count, 1);
			if (bestIdx == tgt) atomicAdd(correct_count, 1);
		}
	}
}

} // anonymous namespace

bool softmax_forward_bf16(const unsigned short* x, int rows, int cols,
                           unsigned short* out)
{
	if (rows <= 0 || cols <= 0) return true;
	int block = rowBlockSize(cols);
	int smemBytes = (block / 32 + 2) * 2 * sizeof(float);
	softmax_stable_rows_bf16<<<rows, block, smemBytes, computeStream()>>>(x, cols, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool softmax_cross_entropy_bwd_bf16(const unsigned short* probs,
                                     const int* targets,
                                     int rows, int cols,
                                     unsigned short* dlogits)
{
	if (rows <= 0 || cols <= 0) return true;
	int block = rowBlockSize(cols);
	softmax_cross_entropy_backward_bf16<<<rows, block, 0, computeStream()>>>(
	    probs, targets, cols, dlogits);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool scale_array_bf16(unsigned short* x, float scale, int n)
{
	if (n <= 0) return true;
	int grid = (n + kBlockElem - 1) / kBlockElem;
	scale_array_bf16_kernel<<<grid, kBlockElem, 0, computeStream()>>>(x, scale, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool cross_entropy_nll_loss_bf16(const unsigned short* probs,
                                  const int* targets,
                                  int T, int vocabSize, int padToken,
                                  float* loss_sum, int* valid_count)
{
	if (T <= 0 || vocabSize <= 0) return true;
	GLADES_CUDA_CHECK(cudaMemset(loss_sum, 0, sizeof(float)));
	GLADES_CUDA_CHECK(cudaMemset(valid_count, 0, sizeof(int)));
	int block = 256;
	int grid = 1;
	if (T > 256) { grid = (T + block - 1) / block; if (grid > 128) grid = 128; }
	int smemBytes = (block / 32 + 2) * sizeof(float) + (block / 32 + 2) * sizeof(int);
	cross_entropy_nll_bf16_kernel<<<grid, block, smemBytes, computeStream()>>>(
	    probs, targets, T, vocabSize, padToken, loss_sum, valid_count);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool argmax_count_matches_bf16(const unsigned short* probs,
                                const int* targets,
                                int T, int vocabSize, int padToken,
                                int* correct_count, int* valid_count)
{
	if (T <= 0 || vocabSize <= 0) return true;
	GLADES_CUDA_CHECK(cudaMemset(correct_count, 0, sizeof(int)));
	GLADES_CUDA_CHECK(cudaMemset(valid_count, 0, sizeof(int)));
	// Iter 56/60: warp-parallel argmax (1 warp per row).  Block = 1024 threads
	// = 32 warps = 32 rows per block.  Replaces the legacy 1-thread-per-row
	// V-loop pattern.
	const int WARPS_PER_BLOCK = 32;
	const int block = WARPS_PER_BLOCK * 32;        // 1024
	int grid = (T + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
	argmax_count_bf16_warp_kernel<<<grid, block, 0, computeStream()>>>(
	    probs, targets, T, vocabSize, padToken, correct_count, valid_count);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  15e3. Live-eval metrics — position-bucketed NLL + top-k accuracy
// ===========================================================================
//
// Backs the eval suite added 2026-05-14:
//   #1 position-stratified val NLL (cross_entropy_nll_bucketed_bf16)
//   #6 top-k accuracy at k ∈ {1, 5, 10} (topk_accuracy_bf16)
//
// Both operate on BF16 probs and respect padToken; both zero their outputs
// internally for caller convenience.

namespace {

// Per-position-bucket NLL accumulation.  Buckets are evenly sized partitions
// of the [0, T) range — bucket b covers positions [b*T/B, (b+1)*T/B).  Each
// row contributes its CE = -log p[target] (with the 1e-12 floor matching
// cross_entropy_nll_loss_bf16) to its bucket's loss_sum, and increments the
// bucket's valid_count.  Pad/OOR tokens are skipped.
//
// Grid: 1-D over T (one thread per row).  Bucket atomics keep contention low
// because B is small (typically 8) and each thread targets exactly one bucket.
__global__ void cross_entropy_nll_bucketed_bf16_kernel(
    const unsigned short* __restrict__ probs,
    const int* __restrict__ targets,
    int T, int vocabSize, int padToken,
    int numBuckets,
    float* __restrict__ loss_sum,   // [numBuckets]
    int*   __restrict__ valid_count) // [numBuckets]
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= T) return;
	int tgt = targets[t];
	if (padToken >= 0 && tgt == padToken) return;
	if (tgt < 0 || tgt >= vocabSize) return;

	float p = bf16_load(probs[(size_t)t * vocabSize + tgt]);
	if (p < 1e-12f) p = 1e-12f;
	float ce = -logf(p);

	// Compute bucket: bucket = t * numBuckets / T (integer division).
	// Equivalent to floor(t / (T/numBuckets)) but avoids float division.
	int bucket = (int)((long long)t * numBuckets / T);
	if (bucket >= numBuckets) bucket = numBuckets - 1;

	atomicAdd(&loss_sum[bucket], ce);
	atomicAdd(&valid_count[bucket], 1);
}

// Top-k accuracy: for each row, find whether the target is among the top-k
// highest-probability tokens.  Uses a single-thread partial sort with an
// in-register top-K buffer of size K_MAX (compile-time bound).
//
// The kernel takes an array of K values (sorted ascending) so callers can
// query multiple K simultaneously: e.g. k_values = {1, 5, 10}.  For each
// row, the kernel computes "rank of target token among all vocab", then for
// each requested k, increments correct_count[i] if rank < k_values[i].
//
// Implementation: scan vocab once, count tokens with prob >= prob[target].
// Rank = number of tokens with strictly greater probability.  This handles
// ties by counting the target as one of the equal-prob tokens.
//
// Grid: 1-D over T rows.  Per-row inner loop is O(V) which is fine at V=32k.
__global__ void topk_accuracy_bf16_kernel(
    const unsigned short* __restrict__ probs,
    const int* __restrict__ targets,
    int T, int vocabSize, int padToken,
    int numK,
    const int* __restrict__ k_values,    // [numK], sorted ascending
    int* __restrict__ correct_counts,    // [numK]
    int* __restrict__ valid_count)       // [1]
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= T) return;
	int tgt = targets[t];
	if (padToken >= 0 && tgt == padToken) return;
	if (tgt < 0 || tgt >= vocabSize) return;

	const unsigned short* row = probs + (size_t)t * vocabSize;
	float p_target = bf16_load(row[tgt]);

	// Rank = number of strictly-greater-probability tokens.  Target is at
	// rank 0 iff it's the unique max (or tied for max with no strictly
	// greater token).  Top-k acc = (rank < k).
	int rank = 0;
	for (int v = 0; v < vocabSize; ++v)
	{
		if (v == tgt) continue;
		float pv = bf16_load(row[v]);
		if (pv > p_target) ++rank;
	}

	atomicAdd(valid_count, 1);
	for (int i = 0; i < numK; ++i)
	{
		if (rank < k_values[i])
			atomicAdd(&correct_counts[i], 1);
	}
}

} // anonymous namespace

bool cross_entropy_nll_bucketed_bf16(const unsigned short* probs,
                                      const int* targets,
                                      int T, int vocabSize, int padToken,
                                      int numBuckets,
                                      float* loss_sum, int* valid_count)
{
	if (T <= 0 || vocabSize <= 0 || numBuckets <= 0) return true;
	GLADES_CUDA_CHECK(cudaMemset(loss_sum, 0, numBuckets * sizeof(float)));
	GLADES_CUDA_CHECK(cudaMemset(valid_count, 0, numBuckets * sizeof(int)));
	int block = 256;
	int grid = (T + block - 1) / block;
	cross_entropy_nll_bucketed_bf16_kernel<<<grid, block, 0, computeStream()>>>(
	    probs, targets, T, vocabSize, padToken, numBuckets, loss_sum, valid_count);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool topk_accuracy_bf16(const unsigned short* probs, const int* targets,
                         int T, int vocabSize, int padToken,
                         int numK, const int* k_values,
                         int* correct_counts, int* valid_count)
{
	if (T <= 0 || vocabSize <= 0 || numK <= 0) return true;
	GLADES_CUDA_CHECK(cudaMemset(correct_counts, 0, numK * sizeof(int)));
	GLADES_CUDA_CHECK(cudaMemset(valid_count, 0, sizeof(int)));
	int block = 128;
	int grid = (T + block - 1) / block;
	topk_accuracy_bf16_kernel<<<grid, block, 0, computeStream()>>>(
	    probs, targets, T, vocabSize, padToken, numK, k_values,
	    correct_counts, valid_count);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  15f. Chunked cross-entropy loss (large-vocab unlock, no T × V scratch)
// ===========================================================================
//
// Streaming log-sum-exp over vocab chunks.  See gpu_kernels.h for full API.

namespace {

// Initialize per-row streaming state: running_max = -inf, running_sum = 0,
// target_logit = NaN (sentinel for "target not yet observed").
__global__ void k_cce_init_state(float* __restrict__ running_max,
                                 float* __restrict__ running_sum,
                                 float* __restrict__ target_logit,
                                 int T)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    running_max[t]  = -INFINITY;
    running_sum[t]  = 0.0f;
    // NaN sentinel: a target that never appears in any chunk means
    // targets[t] < 0 or >= V (invalid / pad — skipped in the final reduce).
    target_logit[t] = nanf("");
}

// Per-chunk streaming update.  One thread block per row of logits_chunk.
// Computes chunk_max[t] and chunk_sum[t] = Σ_v exp(logits[t, v] - chunk_max),
// then merges with the running_max, running_sum via the log-sum-exp rule:
//     new_max = max(running_max, chunk_max)
//     new_sum = running_sum · exp(running_max - new_max)
//             + chunk_sum   · exp(chunk_max  - new_max)
//
// Additionally: if the target token for row t falls inside this chunk,
// capture target_logit[t] = logits_chunk[t, target[t] - chunk_start].
__global__ void k_cce_chunk_update(const float* __restrict__ logits_chunk,
                                   const int* __restrict__ targets,
                                   int T, int V_ch, int chunk_start, int chunk_end,
                                   int padToken,
                                   float* __restrict__ running_max,
                                   float* __restrict__ running_sum,
                                   float* __restrict__ target_logit)
{
    extern __shared__ float smem[];
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;
    if (t >= T) return;

    const float* row = logits_chunk + (size_t)t * V_ch;

    // Per-thread local max + sum-exp contribution.
    float local_max = -INFINITY;
    for (int v = tid; v < V_ch; v += blockDim.x)
    {
        const float x = row[v];
        if (x > local_max) local_max = x;
    }
    // Block reduce for max via shared memory + warp shuffle.
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    for (int off = 16; off > 0; off >>= 1)
    {
        const float o = __shfl_xor_sync(0xffffffffu, local_max, off);
        if (o > local_max) local_max = o;
    }
    if (lane == 0) smem[warp] = local_max;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    float chunk_max = -INFINITY;
    if (warp == 0)
    {
        float v = (tid < nwarps) ? smem[tid] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1)
        {
            const float o = __shfl_xor_sync(0xffffffffu, v, off);
            if (o > v) v = o;
        }
        if (tid == 0) smem[0] = v;
    }
    __syncthreads();
    chunk_max = smem[0];

    // Second pass: sum_exp(logits - chunk_max).
    float local_sum = 0.0f;
    for (int v = tid; v < V_ch; v += blockDim.x)
        local_sum += expf(row[v] - chunk_max);
    // Block reduce sum.
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(0xffffffffu, local_sum, off);
    if (lane == 0) smem[warp] = local_sum;
    __syncthreads();
    if (warp == 0)
    {
        float v = (tid < nwarps) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (tid == 0) smem[1] = v;
    }
    __syncthreads();
    const float chunk_sum = smem[1];

    // Thread 0 merges with running state and captures target if in chunk.
    if (tid == 0)
    {
        const float rm = running_max[t];
        const float rs = running_sum[t];
        const float new_max = rm > chunk_max ? rm : chunk_max;
        // If both rm and chunk_max are -inf, new_max is -inf — safe because
        // exp(-inf - -inf) = exp(NaN) issue is avoided by the isinf guard.
        float new_sum;
        if (isinf(new_max) && new_max < 0.0f)
            new_sum = 0.0f;
        else
            new_sum = rs * expf(rm - new_max) + chunk_sum * expf(chunk_max - new_max);
        running_max[t] = new_max;
        running_sum[t] = new_sum;

        const int tgt = targets[t];
        if (padToken >= 0 && tgt == padToken) return;
        if (tgt < chunk_start || tgt >= chunk_end) return;
        target_logit[t] = row[tgt - chunk_start];
    }
}

// Final reduction: per-row loss[t] = log(running_sum[t]) + running_max[t]
//                                    - target_logit[t]
// Accumulate into loss_sum, and count valid rows into valid_count.
__global__ void k_cce_finalize(const float* __restrict__ running_max,
                               const float* __restrict__ running_sum,
                               const float* __restrict__ target_logit,
                               const int* __restrict__ targets,
                               int T, int padToken,
                               float* __restrict__ loss_sum,
                               int* __restrict__ valid_count)
{
    extern __shared__ float smem[];
    const int tid = threadIdx.x;

    float local_loss = 0.0f;
    int local_valid  = 0;
    for (int t = tid; t < T; t += blockDim.x)
    {
        const int tgt = targets[t];
        if (padToken >= 0 && tgt == padToken) continue;
        if (!isfinite(target_logit[t])) continue;  // pad or out-of-range
        const float lse = logf(running_sum[t]) + running_max[t];
        local_loss += (lse - target_logit[t]);
        ++local_valid;
    }
    // Reduce via warp shuffle + shared.
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1)
        local_loss += __shfl_xor_sync(0xffffffffu, local_loss, off);
    if (lane == 0) smem[warp] = local_loss;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    if (warp == 0)
    {
        float v = (tid < nwarps) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (tid == 0) atomicAdd(loss_sum, v);
    }
    __syncthreads();
    // Count reduce.
    float vF = (float)local_valid;
    for (int off = 16; off > 0; off >>= 1)
        vF += __shfl_xor_sync(0xffffffffu, vF, off);
    if (lane == 0) smem[warp] = vF;
    __syncthreads();
    if (warp == 0)
    {
        float v = (tid < nwarps) ? smem[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (tid == 0) atomicAdd(valid_count, (int)v);
    }
}

} // anonymous namespace

namespace {

// Per-chunk softmax + (−onehot) fused kernel for the CE backward.  Replaces
// logits_chunk in place with dL/dlogits_chunk:
//     dL/dlogits[t, v] = inv_valid ·
//                        ( exp(logits[t, v] - running_max[t]) / running_sum[t]
//                        − (cs + v == target[t] ? 1 : 0) )
//
// Invalid rows (pad, OOR target, NaN target_logit) emit zero gradient
// across the whole chunk.  One block per row, threads tile the V axis.
__global__ void k_cce_softmax_minus_onehot_scaled(
    float* __restrict__ logits_chunk_inout,
    const int* __restrict__ targets,
    const float* __restrict__ running_max,
    const float* __restrict__ running_sum,
    int T, int V_ch, int chunk_start, int padToken,
    float inv_valid)
{
    const int t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x;

    const int tgt = targets[t];
    const bool valid_row = !(padToken >= 0 && tgt == padToken) &&
                           (tgt >= 0);
    // Note: even if tgt >= V, we still correctly zero the gradient here
    // because no column v in any chunk matches, and the onehot term never
    // fires.  But we still scale softmax by inv_valid, so we'd spread
    // non-zero grad onto what should be invalid.  For safety, mask hard.

    const float rmax  = running_max[t];
    const float rsum  = running_sum[t];
    const bool sum_ok = isfinite(rmax) && rsum > 0.0f;

    float* row = logits_chunk_inout + (size_t)t * V_ch;
    for (int v = tid; v < V_ch; v += blockDim.x)
    {
        float g;
        if (!valid_row || !sum_ok)
        {
            g = 0.0f;
        }
        else
        {
            const float p    = expf(row[v] - rmax) / rsum;
            const int   vidx = chunk_start + v;
            const float o    = (vidx == tgt) ? 1.0f : 0.0f;
            g = inv_valid * (p - o);
        }
        row[v] = g;
    }
}

} // anonymous namespace

bool chunked_cross_entropy_backward(const float* X, const float* W_lm,
                                    const int* targets,
                                    const float* running_max,
                                    const float* running_sum,
                                    int T, int V, int d, int padToken,
                                    int V_chunk_size,
                                    int valid_count,
                                    bool accumulate,
                                    float* dX, float* dW_lm,
                                    float* scratch)
{
    if (X == nullptr || W_lm == nullptr || targets == nullptr) return false;
    if (running_max == nullptr || running_sum == nullptr) return false;
    if (dX == nullptr || dW_lm == nullptr || scratch == nullptr) return false;
    if (T <= 0 || V <= 0 || d <= 0 || V_chunk_size <= 0) return false;

    // If there are no valid targets, gradients are zero — just honor
    // accumulate semantics and return.
    if (valid_count <= 0)
    {
        if (!accumulate)
        {
            GLADES_CUDA_CHECK(cudaMemsetAsync(dX,    0, (size_t)T * d * sizeof(float),
                                              computeStream()));
            GLADES_CUDA_CHECK(cudaMemsetAsync(dW_lm, 0, (size_t)V * d * sizeof(float),
                                              computeStream()));
        }
        return true;
    }

    const float inv_valid = 1.0f / (float)valid_count;
    float* logits_chunk = scratch;   // [T × V_chunk_size]

    // Initialize dX, dW_lm if not accumulating.
    if (!accumulate)
    {
        GLADES_CUDA_CHECK(cudaMemsetAsync(dX,    0, (size_t)T * d * sizeof(float),
                                          computeStream()));
        GLADES_CUDA_CHECK(cudaMemsetAsync(dW_lm, 0, (size_t)V * d * sizeof(float),
                                          computeStream()));
    }

    for (int cs = 0; cs < V; cs += V_chunk_size)
    {
        const int ce   = (cs + V_chunk_size < V) ? (cs + V_chunk_size) : V;
        const int V_ch = ce - cs;

        // (1) Re-compute logits_chunk [T × V_ch] = X · W_lm[cs:ce, :]^T.
        if (!sgemm_rowmajor_abt(T, V_ch, d,
                                1.0f,
                                X,                           d,
                                W_lm + (size_t)cs * d,       d,
                                0.0f,
                                logits_chunk,                V_ch))
            return false;

        // (2) In-place: logits_chunk → dL/dlogits_chunk (scaled).
        {
            const int block = 128;
            k_cce_softmax_minus_onehot_scaled<<<T, block, 0, computeStream()>>>(
                logits_chunk, targets, running_max, running_sum,
                T, V_ch, cs, padToken, inv_valid);
            GLADES_CUDA_CHECK(cudaGetLastError());
        }

        // (3) dX += dL/dlogits_chunk · W_lm_chunk.
        //     sgemm_rowmajor: C[M,N] = A[M,K] · B[K,N].
        //     A = dL/dlogits_chunk [T, V_ch], B = W_lm[cs:ce, :] [V_ch, d],
        //     result dX [T, d].  beta=1 accumulates across chunks.
        if (!sgemm_rowmajor(T, d, V_ch,
                            1.0f,
                            logits_chunk,                V_ch,
                            W_lm + (size_t)cs * d,       d,
                            1.0f,
                            dX,                          d))
            return false;

        // (4) dW_lm[cs:ce, :] += dL/dlogits_chunk^T · X.
        //     sgemm_rowmajor_atb: C[M,N] = A^T[M,K] · B[K,N], A stored [K,M].
        //     A = dL/dlogits_chunk [T, V_ch] (K=T, M=V_ch),
        //     B = X [T, d], result dW_lm_chunk [V_ch, d].
        //     beta=1 accumulates if caller chose accumulate=true; our
        //     cudaMemsetAsync already zeroed the chunk when !accumulate.
        if (!sgemm_rowmajor_atb(V_ch, d, T,
                                1.0f,
                                logits_chunk,                V_ch,
                                X,                           d,
                                1.0f,
                                dW_lm + (size_t)cs * d,      d))
            return false;
    }

    return true;
}

bool chunked_cross_entropy_loss(const float* X, const float* W_lm,
                                const int* targets,
                                int T, int V, int d, int padToken,
                                int V_chunk_size,
                                float* loss_sum, int* valid_count,
                                float* scratch)
{
    if (X == nullptr || W_lm == nullptr || targets == nullptr) return false;
    if (loss_sum == nullptr || valid_count == nullptr || scratch == nullptr) return false;
    if (T <= 0 || V <= 0 || d <= 0 || V_chunk_size <= 0) return false;

    float* logits_chunk = scratch;
    float* running_max  = scratch + (size_t)T * V_chunk_size;
    float* running_sum  = running_max + T;
    float* target_logit = running_sum + T;

    GLADES_CUDA_CHECK(cudaMemset(loss_sum, 0, sizeof(float)));
    GLADES_CUDA_CHECK(cudaMemset(valid_count, 0, sizeof(int)));

    {
        const int block = 256;
        const int grid  = (T + block - 1) / block;
        k_cce_init_state<<<grid, block, 0, computeStream()>>>(
            running_max, running_sum, target_logit, T);
        GLADES_CUDA_CHECK(cudaGetLastError());
    }

    // Process V in chunks.
    for (int cs = 0; cs < V; cs += V_chunk_size)
    {
        const int ce   = (cs + V_chunk_size < V) ? (cs + V_chunk_size) : V;
        const int V_ch = ce - cs;

        // logits_chunk [T × V_ch] = X [T × d] · W_lm[cs:ce, :]^T  [V_ch × d]^T
        // sgemm_rowmajor_abt: C[M,N] = A[M,K] · B^T[K,N], B stored as [N,K].
        // Here A=X [T,d], B=W_lm_chunk [V_ch, d], result logits_chunk [T, V_ch].
        if (!sgemm_rowmajor_abt(T, V_ch, d,
                                1.0f,
                                X,                           d,
                                W_lm + (size_t)cs * d,       d,
                                0.0f,
                                logits_chunk,                V_ch))
            return false;

        // Stream update: running max/sum + capture target_logit.
        {
            const int block  = 128;
            const int nwarps = (block + 31) >> 5;
            const size_t smemBytes = (nwarps + 2) * sizeof(float);
            k_cce_chunk_update<<<T, block, smemBytes, computeStream()>>>(
                logits_chunk, targets,
                T, V_ch, cs, ce, padToken,
                running_max, running_sum, target_logit);
            GLADES_CUDA_CHECK(cudaGetLastError());
        }
    }

    // Final loss reduction.
    {
        const int block  = 256;
        const int nwarps = (block + 31) >> 5;
        const size_t smemBytes = (nwarps + 1) * sizeof(float);
        k_cce_finalize<<<1, block, smemBytes, computeStream()>>>(
            running_max, running_sum, target_logit, targets,
            T, padToken, loss_sum, valid_count);
        GLADES_CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

// ===========================================================================
//  16. Flash attention (packed multi-head / GQA)
// ===========================================================================
//
// Training path:
// - Q[T, dModel]
// - K[T, dModelKV]
// - V[T, dModelKV]
// - O[T, dModel]
//
// Heads are packed contiguously inside the feature dimension. For GQA,
// multiple query heads map to the same KV head.
//
// Strategy: one block per (query row, query head). Iterate over K/V tiles
// held in shared memory, update the output row with online softmax, and never
// materialize a [T,T] score/probability matrix.

namespace {

// Default tile size for keys/values loaded into shared memory.
// The *runtime* tile may be smaller when `dHead` is large enough that the
// full-tile shared-memory requirement (`flashTile * 2 * dHead * sizeof(float)`)
// exceeds the device's per-block max-opt-in shared memory.
static constexpr int kFlashTile = 64;

// One warp per block for flash attention kernels. All reductions are
// warp-level via __shfl_down_sync — no __syncthreads in the inner j loop,
// which is the dominant cost when T is large (T×T dot products). The
// scalar softmax state is simply replicated across the 32 lanes.
static constexpr int kFlashBlock = 32;

// Warp-wide sum-reduction via shuffle. Broadcasts to all lanes.
// No shared memory, no __syncthreads required.
__device__ __forceinline__ float flash_warp_reduce_sum(float val)
{
	for (int o = 16; o > 0; o >>= 1)
		val += __shfl_down_sync(0xffffffffu, val, o);
	return __shfl_sync(0xffffffffu, val, 0);
}

// Forward kernel. One block per (query row, query head).
// Shared memory layout:
//   float sK[flashTile * dHead]    -- tile of keys
//   float sV[flashTile * dHead]    -- tile of values
//   float sReduce[blockDim.x]      -- scratch for block-wide reductions
// Passed as dynamic shared memory. `flashTile` is a runtime parameter so
// the launcher can shrink it when `dHead * flashTile * 2 * sizeof(float)`
// exceeds the device's shared-memory budget.
//
// Key optimization (2026-04): the Q·K dot product is now computed in parallel
// across threads (each handles dHead/blockDim.x elements, then block-reduce).
// Previously every thread redundantly computed the full dot product while
// blockDim.x was coupled to flashTile (as small as 8 at dHead=1024). Decoupling
// block size from flashTile lifts the effective parallelism from ~8 to 128.
__global__ void flash_attention_fwd_multihead_kernel(const float* __restrict__ Q,
                                                     const float* __restrict__ K,
                                                     const float* __restrict__ V,
                                                     int T, int nHeads, int nKVHeads,
                                                     int dHead, int dModel, int dModelKV,
                                                     int causal,
                                                     int flashTile,
                                                     float* __restrict__ O)
{
	const int q = static_cast<int>(blockIdx.x);
	const int h = static_cast<int>(blockIdx.y);
	if (q >= T || h >= nHeads) return;

	extern __shared__ float smem[];
	float* sK = smem;                              // [flashTile, dHead]
	float* sV = smem + flashTile * dHead;          // [flashTile, dHead]

	const int groupSize = (nKVHeads > 0) ? (nHeads / nKVHeads) : 1;
	const int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0 ? (h / groupSize) : 0);

	const float* qRow = Q + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead;
	float* oRow = O + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead;

	const int tid = threadIdx.x;

	// Online softmax state (replicated across the 32 lanes; all see the same
	// warp-reduced `dot` so they derive the same state).
	float runMax = -FLT_MAX;
	float runSum = 0.0f;

	for (int d = tid; d < dHead; d += blockDim.x)
		oRow[d] = 0.0f;

	const float scale = rsqrtf(static_cast<float>(dHead));
	const int numTiles = (T + flashTile - 1) / flashTile;

	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > T) tileLen = T - kStart;

		// Cooperatively load sK and sV (one warp striding over loadCount).
		const int loadCount = tileLen * dHead;
		for (int i = tid; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd];
		}
		for (int i = tid; i < loadCount; i += blockDim.x) {
			const int vr = i / dHead;
			const int vd = i % dHead;
			sV[i] = V[static_cast<size_t>(kStart + vr) * dModelKV + static_cast<size_t>(kvHead) * dHead + vd];
		}
		__syncwarp();

		for (int j = 0; j < tileLen; ++j) {
			const int kIdx = kStart + j;
			if (causal && kIdx > q) break;

			// Warp-parallel Q·K dot product.
			float partial = 0.0f;
			for (int d = tid; d < dHead; d += blockDim.x)
				partial += qRow[d] * sK[j * dHead + d];
			const float dot = flash_warp_reduce_sum(partial) * scale;

			// Scalar softmax state update (all 32 lanes compute identical values).
			const float prevMax = runMax;
			if (dot > runMax) runMax = dot;
			const float exp_prev = expf(prevMax - runMax);
			const float exp_cur  = expf(dot - runMax);
			runSum = runSum * exp_prev + exp_cur;

			// Parallel O update (each lane writes a disjoint strip of d).
			for (int d = tid; d < dHead; d += blockDim.x)
				oRow[d] = oRow[d] * exp_prev + exp_cur * sV[j * dHead + d];
		}
	}

	const float invSum = (runSum > 0.0f) ? (1.0f / runSum) : 0.0f;
	for (int d = tid; d < dHead; d += blockDim.x)
		oRow[d] *= invSum;
}

// Backward kernel. One block per (query row, query head).
// Recomputes attention on the fly (flash-style) to avoid storing the T*T matrix.
__global__ void flash_attention_bwd_multihead_kernel(const float* __restrict__ Q,
                                                     const float* __restrict__ K,
                                                     const float* __restrict__ V,
                                                     const float* __restrict__ O,
                                                     const float* __restrict__ dO,
                                                     int T, int nHeads, int nKVHeads,
                                                     int dHead, int dModel, int dModelKV,
                                                     int causal,
                                                     int flashTile,
                                                     float* __restrict__ dQ,
                                                     float* __restrict__ dK_out,
                                                     float* __restrict__ dV_out)
{
	const int q = static_cast<int>(blockIdx.x);
	const int h = static_cast<int>(blockIdx.y);
	if (q >= T || h >= nHeads) return;

	extern __shared__ float smem[];
	float* sK = smem;                             // [flashTile, dHead]
	float* sV = smem + flashTile * dHead;         // [flashTile, dHead]

	const int groupSize = (nKVHeads > 0) ? (nHeads / nKVHeads) : 1;
	const int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0 ? (h / groupSize) : 0);

	const float* qRow  = Q  + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead;
	const float* oRow  = O  + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead;
	const float* doRow = dO + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead;
	float* dqRow       = dQ + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead;

	const int tid = threadIdx.x;
	const float scale = rsqrtf(static_cast<float>(dHead));
	const int numTiles = (T + flashTile - 1) / flashTile;

	// ---- Pass 1: compute runMax, runSum (=> logSumExp) with warp-parallel dot products.
	float runMax = -FLT_MAX;
	float runSum = 0.0f;

	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > T) tileLen = T - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = tid; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd];
		}
		__syncwarp();

		for (int j = 0; j < tileLen; ++j) {
			const int kIdx = kStart + j;
			if (causal && kIdx > q) break;
			float partial = 0.0f;
			for (int d = tid; d < dHead; d += blockDim.x)
				partial += qRow[d] * sK[j * dHead + d];
			const float dot = flash_warp_reduce_sum(partial) * scale;
			if (dot > runMax) {
				runSum = runSum * expf(runMax - dot);
				runMax = dot;
			}
			runSum += expf(dot - runMax);
		}
	}

	const float logSumExp = runMax + logf(runSum + 1e-20f);

	// ---- D = dO · O (warp reduce).
	float Dpartial = 0.0f;
	for (int d = tid; d < dHead; d += blockDim.x)
		Dpartial += doRow[d] * oRow[d];
	const float D = flash_warp_reduce_sum(Dpartial);

	for (int d = tid; d < dHead; d += blockDim.x)
		dqRow[d] = 0.0f;

	// ---- Pass 2: compute dQ (accumulate), dK/dV via atomic adds.
	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > T) tileLen = T - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = tid; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd];
		}
		for (int i = tid; i < loadCount; i += blockDim.x) {
			const int vr = i / dHead;
			const int vd = i % dHead;
			sV[i] = V[static_cast<size_t>(kStart + vr) * dModelKV + static_cast<size_t>(kvHead) * dHead + vd];
		}
		__syncwarp();

		for (int j = 0; j < tileLen; ++j) {
			const int kIdx = kStart + j;
			if (causal && kIdx > q) break;

			// Warp-parallel Q·K dot product.
			float partialQK = 0.0f;
			for (int d = tid; d < dHead; d += blockDim.x)
				partialQK += qRow[d] * sK[j * dHead + d];
			const float dot = flash_warp_reduce_sum(partialQK) * scale;
			const float p = expf(dot - logSumExp);

			// Warp-parallel dO·V dot product.
			float partialDoV = 0.0f;
			for (int d = tid; d < dHead; d += blockDim.x)
				partialDoV += doRow[d] * sV[j * dHead + d];
			const float doV = flash_warp_reduce_sum(partialDoV);
			const float ds = p * (doV - D);

			// dQ accumulate (each lane writes a disjoint strip).
			for (int d = tid; d < dHead; d += blockDim.x)
				dqRow[d] += ds * scale * sK[j * dHead + d];

			// dK, dV atomically accumulated into global (many blocks target same kIdx).
			for (int d = tid; d < dHead; d += blockDim.x)
				atomicAdd(&dK_out[static_cast<size_t>(kIdx) * dModelKV + static_cast<size_t>(kvHead) * dHead + d],
				          ds * scale * qRow[d]);
			for (int d = tid; d < dHead; d += blockDim.x)
				atomicAdd(&dV_out[static_cast<size_t>(kIdx) * dModelKV + static_cast<size_t>(kvHead) * dHead + d],
				          p * doRow[d]);
		}
	}
}

// ===========================================================================
// Multi-query flash attention: one block processes QROWS query rows.
// K/V tile loads are amortized across all QROWS queries in the block, which
// is the dominant bandwidth cost at long T. Each of the QROWS warps within a
// block owns one query row and runs the warp-level dot+softmax independently.
// ===========================================================================

// Queries per block. 4 warps = 128 threads per block; per-query O lives in
// shared memory (QROWS*dHead floats), which at dHead=1024 is 16 KiB — fits
// alongside the 64 KiB K/V tiles in the 100 KiB per-block shmem budget.
static constexpr int kFlashQRows = 4;

template <int QROWS>
__global__ void flash_attention_fwd_multiq_kernel(const float* __restrict__ Q,
                                                   const float* __restrict__ K,
                                                   const float* __restrict__ V,
                                                   int T, int nHeads, int nKVHeads,
                                                   int dHead, int dModel, int dModelKV,
                                                   int causal,
                                                   int flashTile,
                                                   float* __restrict__ O)
{
	const int qBlock = static_cast<int>(blockIdx.x);
	const int h = static_cast<int>(blockIdx.y);
	if (h >= nHeads) return;

	const int warpId = static_cast<int>(threadIdx.x) >> 5;   // 0..QROWS-1
	const int laneId = static_cast<int>(threadIdx.x) & 31;
	const int q = qBlock * QROWS + warpId;
	const bool myRowActive = (q < T);

	extern __shared__ float smem[];
	float* sK = smem;                                         // [flashTile, dHead]
	float* sV = sK + flashTile * dHead;                       // [flashTile, dHead]
	float* sO = sV + flashTile * dHead;                       // [QROWS, dHead]
	float* myO = sO + warpId * dHead;                         // this warp's row

	const int groupSize = (nKVHeads > 0) ? (nHeads / nKVHeads) : 1;
	const int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0 ? (h / groupSize) : 0);

	const float* qRow = myRowActive
	    ? (Q + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : Q;  // safe placeholder
	float* oRowGlobal = myRowActive
	    ? (O + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : nullptr;

	// Initialize this warp's row of sO.
	for (int d = laneId; d < dHead; d += 32)
		myO[d] = 0.0f;

	float runMax = -FLT_MAX;
	float runSum = 0.0f;

	const float scale = rsqrtf(static_cast<float>(dHead));
	const int numTiles = (T + flashTile - 1) / flashTile;

	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > T) tileLen = T - kStart;

		// Cooperative K/V load — all QROWS*32 threads participate.
		const int loadCount = tileLen * dHead;
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd];
		}
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int vr = i / dHead;
			const int vd = i % dHead;
			sV[i] = V[static_cast<size_t>(kStart + vr) * dModelKV + static_cast<size_t>(kvHead) * dHead + vd];
		}
		__syncthreads();

		if (myRowActive) {
			for (int j = 0; j < tileLen; ++j) {
				const int kIdx = kStart + j;
				if (causal && kIdx > q) break;

				// Warp-level Q[q]·K[j].
				float partial = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partial += qRow[d] * sK[j * dHead + d];
				const float dot = flash_warp_reduce_sum(partial) * scale;

				// Online softmax update (replicated across 32 lanes).
				const float prevMax = runMax;
				if (dot > runMax) runMax = dot;
				const float exp_prev = expf(prevMax - runMax);
				const float exp_cur  = expf(dot - runMax);
				runSum = runSum * exp_prev + exp_cur;

				// Per-warp row of sO updated in place (32 lanes, stride 32).
				for (int d = laneId; d < dHead; d += 32)
					myO[d] = myO[d] * exp_prev + exp_cur * sV[j * dHead + d];
			}
		}
		__syncthreads();  // Before next tile overwrites sK / sV
	}

	if (myRowActive) {
		const float invSum = (runSum > 0.0f) ? (1.0f / runSum) : 0.0f;
		for (int d = laneId; d < dHead; d += 32)
			oRowGlobal[d] = myO[d] * invSum;
	}
}

template <int QROWS>
__global__ void flash_attention_bwd_multiq_kernel(const float* __restrict__ Q,
                                                   const float* __restrict__ K,
                                                   const float* __restrict__ V,
                                                   const float* __restrict__ O,
                                                   const float* __restrict__ dO,
                                                   int T, int nHeads, int nKVHeads,
                                                   int dHead, int dModel, int dModelKV,
                                                   int causal,
                                                   int flashTile,
                                                   float* __restrict__ dQ,
                                                   float* __restrict__ dK_out,
                                                   float* __restrict__ dV_out)
{
	const int qBlock = static_cast<int>(blockIdx.x);
	const int h = static_cast<int>(blockIdx.y);
	if (h >= nHeads) return;

	const int warpId = static_cast<int>(threadIdx.x) >> 5;
	const int laneId = static_cast<int>(threadIdx.x) & 31;
	const int q = qBlock * QROWS + warpId;
	const bool myRowActive = (q < T);

	extern __shared__ float smem[];
	float* sK = smem;
	float* sV = sK + flashTile * dHead;
	// Note: no sO in bwd; dqRow is accumulated directly in global, and
	// dK/dV go through atomicAdd. Saves QROWS*dHead*4 bytes of shmem.

	const int groupSize = (nKVHeads > 0) ? (nHeads / nKVHeads) : 1;
	const int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0 ? (h / groupSize) : 0);

	const float* qRow  = myRowActive
	    ? (Q  + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : Q;
	const float* oRow  = myRowActive
	    ? (O  + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : O;
	const float* doRow = myRowActive
	    ? (dO + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : dO;
	float* dqRow = myRowActive
	    ? (dQ + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : nullptr;

	const float scale = rsqrtf(static_cast<float>(dHead));
	const int numTiles = (T + flashTile - 1) / flashTile;

	// ---- Pass 1: compute runMax, runSum => logSumExp (only needs K).
	float runMax = -FLT_MAX;
	float runSum = 0.0f;

	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > T) tileLen = T - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd];
		}
		__syncthreads();

		if (myRowActive) {
			for (int j = 0; j < tileLen; ++j) {
				const int kIdx = kStart + j;
				if (causal && kIdx > q) break;
				float partial = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partial += qRow[d] * sK[j * dHead + d];
				const float dot = flash_warp_reduce_sum(partial) * scale;
				if (dot > runMax) {
					runSum = runSum * expf(runMax - dot);
					runMax = dot;
				}
				runSum += expf(dot - runMax);
			}
		}
		__syncthreads();
	}

	const float logSumExp = runMax + logf(runSum + 1e-20f);

	// ---- D = dO · O (warp reduce per active warp).
	float D = 0.0f;
	if (myRowActive) {
		float Dpartial = 0.0f;
		for (int d = laneId; d < dHead; d += 32)
			Dpartial += doRow[d] * oRow[d];
		D = flash_warp_reduce_sum(Dpartial);

		for (int d = laneId; d < dHead; d += 32)
			dqRow[d] = 0.0f;
	}

	// ---- Pass 2: compute dQ (accumulate), dK/dV via atomic adds.
	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > T) tileLen = T - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd];
		}
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int vr = i / dHead;
			const int vd = i % dHead;
			sV[i] = V[static_cast<size_t>(kStart + vr) * dModelKV + static_cast<size_t>(kvHead) * dHead + vd];
		}
		__syncthreads();

		if (myRowActive) {
			for (int j = 0; j < tileLen; ++j) {
				const int kIdx = kStart + j;
				if (causal && kIdx > q) break;

				// Warp Q·K.
				float partialQK = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partialQK += qRow[d] * sK[j * dHead + d];
				const float dot = flash_warp_reduce_sum(partialQK) * scale;
				const float p = expf(dot - logSumExp);

				// Warp dO·V.
				float partialDoV = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partialDoV += doRow[d] * sV[j * dHead + d];
				const float doV = flash_warp_reduce_sum(partialDoV);
				const float ds = p * (doV - D);

				// dQ accumulate (each lane writes disjoint strip).
				for (int d = laneId; d < dHead; d += 32)
					dqRow[d] += ds * scale * sK[j * dHead + d];

				// dK, dV atomic adds (many query blocks target same kIdx).
				for (int d = laneId; d < dHead; d += 32)
					atomicAdd(&dK_out[static_cast<size_t>(kIdx) * dModelKV + static_cast<size_t>(kvHead) * dHead + d],
					          ds * scale * qRow[d]);
				for (int d = laneId; d < dHead; d += 32)
					atomicAdd(&dV_out[static_cast<size_t>(kIdx) * dModelKV + static_cast<size_t>(kvHead) * dHead + d],
					          p * doRow[d]);
			}
		}
		__syncthreads();
	}
}

} // anonymous namespace

bool flash_attention_forward(const float* Q, const float* K, const float* V,
                             int T, int dK, int dV, bool causal,
                             float* O)
{
	if (dK != dV) return false;
	return flash_attention_multihead_forward(Q, K, V, T, 1, 1, dK, dK, dK, causal, O);
}

bool flash_attention_backward(const float* Q, const float* K, const float* V,
                              const float* O, const float* dO,
                              int T, int dK, int dV, bool causal,
                              float* dQ, float* dK_out, float* dV_out)
{
	if (dK != dV) return false;
	return flash_attention_multihead_backward(Q, K, V, O, dO, T, 1, 1, dK, dK, dK, causal,
	                                          dQ, dK_out, dV_out);
}

namespace {

// Compute a runtime flashTile that fits in the device's per-block shared-memory
// budget. `extraSmemBytes` covers additional per-block shared allocations
// (e.g., the sO[QROWS, dHead] buffer for the multi-query fwd kernel).
// Opts into the max dynamic shared memory for the given kernel so the
// large-dHead case works on Ampere+ / Ada / Hopper GPUs.
template <typename KernelT>
static int setup_flash_tile(KernelT kernel, int dHead, size_t extraSmemBytes = 0)
{
	// Query device's per-block opt-in max shared memory (bytes).
	int dev = 0;
	cudaGetDevice(&dev);
	int maxOptin = 48 * 1024; // default static limit
	cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

	// Start at kFlashTile and halve until the tile + extras fit.
	int tile = kFlashTile;
	auto required = [&](int t) {
		return static_cast<size_t>(t) * static_cast<size_t>(2 * dHead) * sizeof(float)
		       + extraSmemBytes;
	};
	while (tile > 4 && required(tile) > static_cast<size_t>(maxOptin))
		tile /= 2;

	// Opt into the required shared memory per block.
	const size_t smem = required(tile);
	if (smem > 48u * 1024u)
		cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
		                     cudaFuncAttributeMaxDynamicSharedMemorySize,
		                     static_cast<int>(smem));
	return tile;
}

// Inline BF16 -> FP32 cast used at the per-thread load point of the BF16
// flash-attention kernels. Software-only (the bf16 value becomes the upper
// half of the fp32 bit pattern), so this is essentially free vs a plain load.
__device__ __forceinline__ float bf16_as_float(uint16_t b)
{
	union { uint32_t u; float f; } v;
	v.u = static_cast<uint32_t>(b) << 16;
	return v.f;
}

// BF16-input single-query flash attention. Q/K/V stored BF16 in global memory;
// shared-memory tiles and all softmax/accumulation still FP32 for numerical
// stability. Kernel body mirrors flash_attention_fwd_multihead_kernel with the
// four global load sites swapped for inline bf16_as_float casts.
__global__ void flash_attention_fwd_multihead_kernel_bf16(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    int T, int nHeads, int nKVHeads,
    int dHead, int dModel, int dModelKV,
    int causal,
    int flashTile,
    float* __restrict__ O)
{
	const int q = static_cast<int>(blockIdx.x);
	const int h = static_cast<int>(blockIdx.y);
	if (q >= T || h >= nHeads) return;

	extern __shared__ float smem[];
	float* sK = smem;
	float* sV = smem + flashTile * dHead;

	const int groupSize = (nKVHeads > 0) ? (nHeads / nKVHeads) : 1;
	const int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0 ? (h / groupSize) : 0);

	const uint16_t* qRow = Q + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead;
	float* oRow = O + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead;

	const int tid = threadIdx.x;

	float runMax = -FLT_MAX;
	float runSum = 0.0f;

	for (int d = tid; d < dHead; d += blockDim.x)
		oRow[d] = 0.0f;

	const float scale = rsqrtf(static_cast<float>(dHead));
	const int numTiles = (T + flashTile - 1) / flashTile;

	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > T) tileLen = T - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = tid; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = bf16_as_float(K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd]);
		}
		for (int i = tid; i < loadCount; i += blockDim.x) {
			const int vr = i / dHead;
			const int vd = i % dHead;
			sV[i] = bf16_as_float(V[static_cast<size_t>(kStart + vr) * dModelKV + static_cast<size_t>(kvHead) * dHead + vd]);
		}
		__syncwarp();

		for (int j = 0; j < tileLen; ++j) {
			const int kIdx = kStart + j;
			if (causal && kIdx > q) break;

			float partial = 0.0f;
			for (int d = tid; d < dHead; d += blockDim.x)
				partial += bf16_as_float(qRow[d]) * sK[j * dHead + d];
			const float dot = flash_warp_reduce_sum(partial) * scale;

			const float prevMax = runMax;
			if (dot > runMax) runMax = dot;
			const float exp_prev = expf(prevMax - runMax);
			const float exp_cur  = expf(dot - runMax);
			runSum = runSum * exp_prev + exp_cur;

			for (int d = tid; d < dHead; d += blockDim.x)
				oRow[d] = oRow[d] * exp_prev + exp_cur * sV[j * dHead + d];
		}
	}

	const float invSum = (runSum > 0.0f) ? (1.0f / runSum) : 0.0f;
	for (int d = tid; d < dHead; d += blockDim.x)
		oRow[d] *= invSum;
}

// BF16-input multi-query flash attention — same amortized K/V tile reuse
// as the FP32 variant; loads reinterpret BF16 inputs as FP32 on the fly.
template <int QROWS>
__global__ void flash_attention_fwd_multiq_kernel_bf16(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    int T, int nHeads, int nKVHeads,
    int dHead, int dModel, int dModelKV,
    int causal,
    int flashTile,
    float* __restrict__ O)
{
	const int qBlock = static_cast<int>(blockIdx.x);
	const int h = static_cast<int>(blockIdx.y);
	if (h >= nHeads) return;

	const int warpId = static_cast<int>(threadIdx.x) >> 5;
	const int laneId = static_cast<int>(threadIdx.x) & 31;
	const int q = qBlock * QROWS + warpId;
	const bool myRowActive = (q < T);

	extern __shared__ float smem[];
	float* sK = smem;
	float* sV = sK + flashTile * dHead;
	float* sO = sV + flashTile * dHead;
	float* myO = sO + warpId * dHead;

	const int groupSize = (nKVHeads > 0) ? (nHeads / nKVHeads) : 1;
	const int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0 ? (h / groupSize) : 0);

	const uint16_t* qRow = myRowActive
	    ? (Q + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : Q;
	float* oRowGlobal = myRowActive
	    ? (O + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : nullptr;

	for (int d = laneId; d < dHead; d += 32)
		myO[d] = 0.0f;

	float runMax = -FLT_MAX;
	float runSum = 0.0f;

	const float scale = rsqrtf(static_cast<float>(dHead));
	const int numTiles = (T + flashTile - 1) / flashTile;

	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > T) tileLen = T - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = bf16_as_float(K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd]);
		}
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int vr = i / dHead;
			const int vd = i % dHead;
			sV[i] = bf16_as_float(V[static_cast<size_t>(kStart + vr) * dModelKV + static_cast<size_t>(kvHead) * dHead + vd]);
		}
		__syncthreads();

		if (myRowActive) {
			for (int j = 0; j < tileLen; ++j) {
				const int kIdx = kStart + j;
				if (causal && kIdx > q) break;

				float partial = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partial += bf16_as_float(qRow[d]) * sK[j * dHead + d];
				const float dot = flash_warp_reduce_sum(partial) * scale;

				const float prevMax = runMax;
				if (dot > runMax) runMax = dot;
				const float exp_prev = expf(prevMax - runMax);
				const float exp_cur  = expf(dot - runMax);
				runSum = runSum * exp_prev + exp_cur;

				for (int d = laneId; d < dHead; d += 32)
					myO[d] = myO[d] * exp_prev + exp_cur * sV[j * dHead + d];
			}
		}
		__syncthreads();
	}

	if (myRowActive) {
		const float invSum = (runSum > 0.0f) ? (1.0f / runSum) : 0.0f;
		for (int d = laneId; d < dHead; d += 32)
			oRowGlobal[d] = myO[d] * invSum;
	}
}

// Local-window BF16 flash attention forward.  Each query attends to keys in
// [q - windowSize, q] (causal) or [q - windowSize, q + windowSize] (non-causal).
// Skips tiles that fall entirely outside the window, giving an O(T·W) compute
// profile instead of O(T²).  For windowSize <= 0, degenerates to full
// attention (same output as the non-local kernel).  See
// research/SUBQUADRATIC_ATTENTION_DESIGN.md for the full rationale.
template <int QROWS>
__global__ void flash_attention_fwd_local_kernel_bf16(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    int T, int nHeads, int nKVHeads,
    int dHead, int dModel, int dModelKV,
    int causal, int windowSize,
    int sinkCount,
    int flashTile,
    float* __restrict__ O)
{
	const int qBlock = static_cast<int>(blockIdx.x);
	const int h = static_cast<int>(blockIdx.y);
	if (h >= nHeads) return;

	const int warpId = static_cast<int>(threadIdx.x) >> 5;
	const int laneId = static_cast<int>(threadIdx.x) & 31;
	const int q = qBlock * QROWS + warpId;
	const bool myRowActive = (q < T);

	extern __shared__ float smem[];
	float* sK = smem;
	float* sV = sK + flashTile * dHead;
	float* sO = sV + flashTile * dHead;
	float* myO = sO + warpId * dHead;

	const int groupSize = (nKVHeads > 0) ? (nHeads / nKVHeads) : 1;
	const int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0 ? (h / groupSize) : 0);

	const uint16_t* qRow = myRowActive
	    ? (Q + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : Q;
	float* oRowGlobal = myRowActive
	    ? (O + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : nullptr;

	for (int d = laneId; d < dHead; d += 32)
		myO[d] = 0.0f;

	float runMax = -FLT_MAX;
	float runSum = 0.0f;

	const float scale = rsqrtf(static_cast<float>(dHead));

	// Compute this query row's window: [kLo, kHi).
	// windowSize <= 0 → full attention.
	const int kLo_row = (windowSize > 0 && q > windowSize) ? (q - windowSize) : 0;
	const int kHi_row_raw = causal
	    ? (q + 1)
	    : (windowSize > 0 ? (q + windowSize + 1) : T);
	const int kHi_row = (kHi_row_raw > T) ? T : kHi_row_raw;

	// For cross-warp tile loading we use the UNION across the QROWS warps in this block:
	// the first query in the block has the lowest row-kLo; the last has the highest row-kHi.
	const int qBlockStart = qBlock * QROWS;
	const int qBlockEnd = qBlockStart + QROWS - 1;  // inclusive
	const int qBlockEndCap = (qBlockEnd < T - 1) ? qBlockEnd : (T - 1);
	// Paradigm #78 ATTENTION-SINK: when sinkCount > 0, the first sinkCount
	// keys are always allowed regardless of window. Block-level kLo collapses
	// to 0 to ensure those tiles are loaded; per-row check below distinguishes
	// sink keys from window keys.
	const int blockKLo = (sinkCount > 0)
	    ? 0
	    : ((windowSize > 0 && qBlockStart > windowSize) ? (qBlockStart - windowSize) : 0);
	const int blockKHi_raw = causal
	    ? (qBlockEndCap + 1)
	    : (windowSize > 0 ? (qBlockEndCap + windowSize + 1) : T);
	const int blockKHi = (blockKHi_raw > T) ? T : blockKHi_raw;

	const int numTiles = (blockKHi - blockKLo + flashTile - 1) / flashTile;

	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = blockKLo + tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > blockKHi) tileLen = blockKHi - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = bf16_as_float(K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd]);
		}
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int vr = i / dHead;
			const int vd = i % dHead;
			sV[i] = bf16_as_float(V[static_cast<size_t>(kStart + vr) * dModelKV + static_cast<size_t>(kvHead) * dHead + vd]);
		}
		__syncthreads();

		if (myRowActive) {
			for (int j = 0; j < tileLen; ++j) {
				const int kIdx = kStart + j;
				// Skip if outside this row's window AND outside the sink range.
				// Sinks: first `sinkCount` keys always allowed.
				if (kIdx < kLo_row && kIdx >= sinkCount) continue;
				if (kIdx >= kHi_row) break;   // sorted order; no more valid

				float partial = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partial += bf16_as_float(qRow[d]) * sK[j * dHead + d];
				const float dot = flash_warp_reduce_sum(partial) * scale;

				const float prevMax = runMax;
				if (dot > runMax) runMax = dot;
				const float exp_prev = expf(prevMax - runMax);
				const float exp_cur  = expf(dot - runMax);
				runSum = runSum * exp_prev + exp_cur;

				for (int d = laneId; d < dHead; d += 32)
					myO[d] = myO[d] * exp_prev + exp_cur * sV[j * dHead + d];
			}
		}
		__syncthreads();
	}

	if (myRowActive) {
		const float invSum = (runSum > 0.0f) ? (1.0f / runSum) : 0.0f;
		for (int d = laneId; d < dHead; d += 32)
			oRowGlobal[d] = myO[d] * invSum;
	}
}

// BF16-input multi-query flash attention backward. Q/K/V are BF16 in global
// memory (the dominant memory-traffic tensors in backward, loaded in both
// pass 1 and pass 2). O, dO, dQ, dK, dV stay FP32 since they are each loaded
// or written once. Softmax / accumulation all in FP32 for numerical
// stability.
template <int QROWS>
__global__ void flash_attention_bwd_multiq_kernel_bf16(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    int T, int nHeads, int nKVHeads,
    int dHead, int dModel, int dModelKV,
    int causal,
    int flashTile,
    float* __restrict__ dQ,
    float* __restrict__ dK_out,
    float* __restrict__ dV_out)
{
	const int qBlock = static_cast<int>(blockIdx.x);
	const int h = static_cast<int>(blockIdx.y);
	if (h >= nHeads) return;

	const int warpId = static_cast<int>(threadIdx.x) >> 5;
	const int laneId = static_cast<int>(threadIdx.x) & 31;
	const int q = qBlock * QROWS + warpId;
	const bool myRowActive = (q < T);

	extern __shared__ float smem[];
	float* sK = smem;
	float* sV = sK + flashTile * dHead;

	const int groupSize = (nKVHeads > 0) ? (nHeads / nKVHeads) : 1;
	const int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0 ? (h / groupSize) : 0);

	const uint16_t* qRow = myRowActive
	    ? (Q + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : Q;
	const float* oRow  = myRowActive
	    ? (O  + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : O;
	const float* doRow = myRowActive
	    ? (dO + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : dO;
	float* dqRow = myRowActive
	    ? (dQ + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : nullptr;

	const float scale = rsqrtf(static_cast<float>(dHead));
	const int numTiles = (T + flashTile - 1) / flashTile;

	// Pass 1: runMax, runSum.
	float runMax = -FLT_MAX;
	float runSum = 0.0f;

	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > T) tileLen = T - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = bf16_as_float(K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd]);
		}
		__syncthreads();

		if (myRowActive) {
			for (int j = 0; j < tileLen; ++j) {
				const int kIdx = kStart + j;
				if (causal && kIdx > q) break;
				float partial = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partial += bf16_as_float(qRow[d]) * sK[j * dHead + d];
				const float dot = flash_warp_reduce_sum(partial) * scale;
				if (dot > runMax) {
					runSum = runSum * expf(runMax - dot);
					runMax = dot;
				}
				runSum += expf(dot - runMax);
			}
		}
		__syncthreads();
	}

	const float logSumExp = runMax + logf(runSum + 1e-20f);

	float D = 0.0f;
	if (myRowActive) {
		float Dpartial = 0.0f;
		for (int d = laneId; d < dHead; d += 32)
			Dpartial += doRow[d] * oRow[d];
		D = flash_warp_reduce_sum(Dpartial);

		for (int d = laneId; d < dHead; d += 32)
			dqRow[d] = 0.0f;
	}

	// Pass 2: dQ accumulate, dK/dV via atomic adds.
	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > T) tileLen = T - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = bf16_as_float(K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd]);
		}
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int vr = i / dHead;
			const int vd = i % dHead;
			sV[i] = bf16_as_float(V[static_cast<size_t>(kStart + vr) * dModelKV + static_cast<size_t>(kvHead) * dHead + vd]);
		}
		__syncthreads();

		if (myRowActive) {
			for (int j = 0; j < tileLen; ++j) {
				const int kIdx = kStart + j;
				if (causal && kIdx > q) break;

				float partialQK = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partialQK += bf16_as_float(qRow[d]) * sK[j * dHead + d];
				const float dot = flash_warp_reduce_sum(partialQK) * scale;
				const float p = expf(dot - logSumExp);

				float partialDoV = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partialDoV += doRow[d] * sV[j * dHead + d];
				const float doV = flash_warp_reduce_sum(partialDoV);
				const float ds = p * (doV - D);

				for (int d = laneId; d < dHead; d += 32)
					dqRow[d] += ds * scale * sK[j * dHead + d];

				for (int d = laneId; d < dHead; d += 32)
					atomicAdd(&dK_out[static_cast<size_t>(kIdx) * dModelKV + static_cast<size_t>(kvHead) * dHead + d],
					          ds * scale * bf16_as_float(qRow[d]));
				for (int d = laneId; d < dHead; d += 32)
					atomicAdd(&dV_out[static_cast<size_t>(kIdx) * dModelKV + static_cast<size_t>(kvHead) * dHead + d],
					          p * doRow[d]);
			}
		}
		__syncthreads();
	}
}

// Local-window BF16 flash attention backward — mirrors the full BF16
// backward but applies the same per-row window bounds as the local forward
// kernel.  Tiles falling entirely outside the block-union window are
// skipped (no K/V load); inside a partial tile, individual j iterations
// outside the per-row window are skipped.
//
// Uses two passes like the full backward: pass 1 computes runMax + runSum
// (softmax normalizer) over the restricted window; pass 2 accumulates
// dQ/dK/dV using the same P = exp(S - logSumExp) formulation.
template <int QROWS>
__global__ void flash_attention_bwd_local_kernel_bf16(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    int T, int nHeads, int nKVHeads,
    int dHead, int dModel, int dModelKV,
    int causal, int windowSize,
    int sinkCount,
    int flashTile,
    float* __restrict__ dQ,
    float* __restrict__ dK_out,
    float* __restrict__ dV_out)
{
	const int qBlock = static_cast<int>(blockIdx.x);
	const int h = static_cast<int>(blockIdx.y);
	if (h >= nHeads) return;

	const int warpId = static_cast<int>(threadIdx.x) >> 5;
	const int laneId = static_cast<int>(threadIdx.x) & 31;
	const int q = qBlock * QROWS + warpId;
	const bool myRowActive = (q < T);

	extern __shared__ float smem[];
	float* sK = smem;
	float* sV = sK + flashTile * dHead;

	const int groupSize = (nKVHeads > 0) ? (nHeads / nKVHeads) : 1;
	const int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0 ? (h / groupSize) : 0);

	const uint16_t* qRow = myRowActive
	    ? (Q + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : Q;
	const float* oRow  = myRowActive
	    ? (O  + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : O;
	const float* doRow = myRowActive
	    ? (dO + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : dO;
	float* dqRow = myRowActive
	    ? (dQ + static_cast<size_t>(q) * dModel + static_cast<size_t>(h) * dHead)
	    : nullptr;

	const float scale = rsqrtf(static_cast<float>(dHead));

	// Per-row window [kLo_row, kHi_row).
	const int kLo_row = (windowSize > 0 && q > windowSize) ? (q - windowSize) : 0;
	const int kHi_row_raw = causal
	    ? (q + 1)
	    : (windowSize > 0 ? (q + windowSize + 1) : T);
	const int kHi_row = (kHi_row_raw > T) ? T : kHi_row_raw;

	// Block-union window (loosest bounds across the QROWS queries in this block).
	const int qBlockStart = qBlock * QROWS;
	const int qBlockEnd = qBlockStart + QROWS - 1;
	const int qBlockEndCap = (qBlockEnd < T - 1) ? qBlockEnd : (T - 1);
	// Paradigm #78 sinks: same logic as forward — when sinkCount > 0, expand
	// block-level kLo to 0 so sink tiles are loaded; per-row check distinguishes.
	const int blockKLo = (sinkCount > 0)
	    ? 0
	    : ((windowSize > 0 && qBlockStart > windowSize) ? (qBlockStart - windowSize) : 0);
	const int blockKHi_raw = causal
	    ? (qBlockEndCap + 1)
	    : (windowSize > 0 ? (qBlockEndCap + windowSize + 1) : T);
	const int blockKHi = (blockKHi_raw > T) ? T : blockKHi_raw;

	const int numTiles = (blockKHi - blockKLo + flashTile - 1) / flashTile;

	// Pass 1: compute runMax, runSum over the restricted window.
	float runMax = -FLT_MAX;
	float runSum = 0.0f;

	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = blockKLo + tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > blockKHi) tileLen = blockKHi - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = bf16_as_float(K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd]);
		}
		__syncthreads();

		if (myRowActive) {
			for (int j = 0; j < tileLen; ++j) {
				const int kIdx = kStart + j;
				if (kIdx < kLo_row && kIdx >= sinkCount) continue;
				if (kIdx >= kHi_row) break;
				float partial = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partial += bf16_as_float(qRow[d]) * sK[j * dHead + d];
				const float dot = flash_warp_reduce_sum(partial) * scale;
				if (dot > runMax) {
					runSum = runSum * expf(runMax - dot);
					runMax = dot;
				}
				runSum += expf(dot - runMax);
			}
		}
		__syncthreads();
	}

	const float logSumExp = runMax + logf(runSum + 1e-20f);

	float D = 0.0f;
	if (myRowActive) {
		float Dpartial = 0.0f;
		for (int d = laneId; d < dHead; d += 32)
			Dpartial += doRow[d] * oRow[d];
		D = flash_warp_reduce_sum(Dpartial);
		for (int d = laneId; d < dHead; d += 32) dqRow[d] = 0.0f;
	}

	// Pass 2: dQ accumulate; dK, dV via atomic adds (only for in-window keys).
	for (int tile = 0; tile < numTiles; ++tile) {
		const int kStart = blockKLo + tile * flashTile;
		int tileLen = flashTile;
		if (kStart + tileLen > blockKHi) tileLen = blockKHi - kStart;

		const int loadCount = tileLen * dHead;
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int kr = i / dHead;
			const int kd = i % dHead;
			sK[i] = bf16_as_float(K[static_cast<size_t>(kStart + kr) * dModelKV + static_cast<size_t>(kvHead) * dHead + kd]);
		}
		for (int i = threadIdx.x; i < loadCount; i += blockDim.x) {
			const int vr = i / dHead;
			const int vd = i % dHead;
			sV[i] = bf16_as_float(V[static_cast<size_t>(kStart + vr) * dModelKV + static_cast<size_t>(kvHead) * dHead + vd]);
		}
		__syncthreads();

		if (myRowActive) {
			for (int j = 0; j < tileLen; ++j) {
				const int kIdx = kStart + j;
				if (kIdx < kLo_row && kIdx >= sinkCount) continue;
				if (kIdx >= kHi_row) break;

				float partialQK = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partialQK += bf16_as_float(qRow[d]) * sK[j * dHead + d];
				const float dot = flash_warp_reduce_sum(partialQK) * scale;
				const float p = expf(dot - logSumExp);

				float partialDoV = 0.0f;
				for (int d = laneId; d < dHead; d += 32)
					partialDoV += doRow[d] * sV[j * dHead + d];
				const float doV = flash_warp_reduce_sum(partialDoV);
				const float ds = p * (doV - D);

				for (int d = laneId; d < dHead; d += 32)
					dqRow[d] += ds * scale * sK[j * dHead + d];

				for (int d = laneId; d < dHead; d += 32)
					atomicAdd(&dK_out[static_cast<size_t>(kIdx) * dModelKV + static_cast<size_t>(kvHead) * dHead + d],
					          ds * scale * bf16_as_float(qRow[d]));
				for (int d = laneId; d < dHead; d += 32)
					atomicAdd(&dV_out[static_cast<size_t>(kIdx) * dModelKV + static_cast<size_t>(kvHead) * dHead + d],
					          p * doRow[d]);
			}
		}
		__syncthreads();
	}
}

} // anonymous namespace

// Local-window BF16 flash attention backward wrapper.  windowSize <= 0 or
// >= T falls back to the full backward.
// Paradigm #78: sinkCount > 0 forces the local-window path so sinks are
// always preserved even when windowSize >= T degenerate.
bool flash_attention_multihead_backward_bf16_local(
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const float* O, const float* dO,
    int T, int nHeads, int nKVHeads,
    int dHead, int dModel, int dModelKV,
    bool causal, int windowSize,
    float* dQ, float* dK_out, float* dV_out,
    int sinkCount)
{
	if (T <= 0 || nHeads <= 0 || nKVHeads <= 0 || dHead <= 0 || dModel <= 0 || dModelKV <= 0)
		return true;
	if ((windowSize <= 0 || windowSize >= T) && sinkCount <= 0) {
		return flash_attention_multihead_backward_bf16(
		    Q, K, V, O, dO,
		    T, nHeads, nKVHeads, dHead, dModel, dModelKV,
		    causal, dQ, dK_out, dV_out);
	}

	GLADES_CUDA_CHECK(cudaMemset(dK_out, 0, static_cast<size_t>(T) * static_cast<size_t>(dModelKV) * sizeof(float)));
	GLADES_CUDA_CHECK(cudaMemset(dV_out, 0, static_cast<size_t>(T) * static_cast<size_t>(dModelKV) * sizeof(float)));

	int dev = 0;
	cudaGetDevice(&dev);
	int maxOptin = 48 * 1024;
	cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
	const size_t minMultiQ = static_cast<size_t>(4) * static_cast<size_t>(2 * dHead) * sizeof(float);
	const bool multiQFits = (minMultiQ <= static_cast<size_t>(maxOptin));

	if (!multiQFits) {
		return flash_attention_multihead_backward_bf16(
		    Q, K, V, O, dO,
		    T, nHeads, nKVHeads, dHead, dModel, dModelKV,
		    causal, dQ, dK_out, dV_out);
	}

	const int flashTile = setup_flash_tile(flash_attention_bwd_local_kernel_bf16<kFlashQRows>, dHead);
	const int block = 32 * kFlashQRows;
	const dim3 grid(static_cast<unsigned int>((T + kFlashQRows - 1) / kFlashQRows),
	                static_cast<unsigned int>(nHeads), 1u);
	const size_t smemBytes = static_cast<size_t>(flashTile) * static_cast<size_t>(2 * dHead) * sizeof(float);
	flash_attention_bwd_local_kernel_bf16<kFlashQRows><<<grid, block, smemBytes, computeStream()>>>(
	    Q, K, V, O, dO,
	    T, nHeads, nKVHeads, dHead, dModel, dModelKV,
	    causal ? 1 : 0, windowSize, sinkCount, flashTile, dQ, dK_out, dV_out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool flash_attention_multihead_backward_bf16(
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const float* O, const float* dO,
    int T, int nHeads, int nKVHeads,
    int dHead, int dModel, int dModelKV,
    bool causal,
    float* dQ, float* dK_out, float* dV_out)
{
	if (T <= 0 || nHeads <= 0 || nKVHeads <= 0 || dHead <= 0 || dModel <= 0 || dModelKV <= 0)
		return true;

	GLADES_CUDA_CHECK(cudaMemset(dK_out, 0, static_cast<size_t>(T) * static_cast<size_t>(dModelKV) * sizeof(float)));
	GLADES_CUDA_CHECK(cudaMemset(dV_out, 0, static_cast<size_t>(T) * static_cast<size_t>(dModelKV) * sizeof(float)));

	int dev = 0;
	cudaGetDevice(&dev);
	int maxOptin = 48 * 1024;
	cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
	const size_t minMultiQ = static_cast<size_t>(4) * static_cast<size_t>(2 * dHead) * sizeof(float);
	const bool multiQFits = (minMultiQ <= static_cast<size_t>(maxOptin));

	if (multiQFits) {
		const int flashTile = setup_flash_tile(flash_attention_bwd_multiq_kernel_bf16<kFlashQRows>, dHead);
		const int block = 32 * kFlashQRows;
		const dim3 grid(static_cast<unsigned int>((T + kFlashQRows - 1) / kFlashQRows),
		                static_cast<unsigned int>(nHeads), 1u);
		const size_t smemBytes = static_cast<size_t>(flashTile) * static_cast<size_t>(2 * dHead) * sizeof(float);
		flash_attention_bwd_multiq_kernel_bf16<kFlashQRows><<<grid, block, smemBytes, computeStream()>>>(
			Q, K, V, O, dO, T, nHeads, nKVHeads, dHead, dModel, dModelKV, causal ? 1 : 0,
			flashTile, dQ, dK_out, dV_out);
		GLADES_CUDA_CHECK(cudaGetLastError());
		return true;
	}

	// Single-query fallback not implemented yet for BF16 backward (large-dHead
	// path would need its own kernel copy); fall back to FP32 path by
	// returning false for the caller to route around.
	return false;
}

bool flash_attention_multihead_forward_bf16(const uint16_t* Q, const uint16_t* K,
                                            const uint16_t* V,
                                            int T, int nHeads, int nKVHeads,
                                            int dHead, int dModel, int dModelKV,
                                            bool causal, float* O)
{
	if (T <= 0 || nHeads <= 0 || nKVHeads <= 0 || dHead <= 0 || dModel <= 0 || dModelKV <= 0)
		return true;

	const size_t sOBytes = static_cast<size_t>(kFlashQRows) * static_cast<size_t>(dHead) * sizeof(float);
	int dev = 0;
	cudaGetDevice(&dev);
	int maxOptin = 48 * 1024;
	cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
	const size_t minMultiQ = static_cast<size_t>(4) * static_cast<size_t>(2 * dHead) * sizeof(float) + sOBytes;
	const bool multiQFits = (minMultiQ <= static_cast<size_t>(maxOptin));

	if (multiQFits) {
		const int flashTile = setup_flash_tile(flash_attention_fwd_multiq_kernel_bf16<kFlashQRows>, dHead, sOBytes);
		const int block = 32 * kFlashQRows;
		const dim3 grid(static_cast<unsigned int>((T + kFlashQRows - 1) / kFlashQRows),
		                static_cast<unsigned int>(nHeads), 1u);
		const size_t smemBytes = static_cast<size_t>(flashTile) * static_cast<size_t>(2 * dHead) * sizeof(float) + sOBytes;
		flash_attention_fwd_multiq_kernel_bf16<kFlashQRows><<<grid, block, smemBytes, computeStream()>>>(
			Q, K, V, T, nHeads, nKVHeads, dHead, dModel, dModelKV, causal ? 1 : 0, flashTile, O);
		GLADES_CUDA_CHECK(cudaGetLastError());
		return true;
	}

	const int flashTile = setup_flash_tile(flash_attention_fwd_multihead_kernel_bf16, dHead);
	const int block = kFlashBlock;
	const dim3 grid(static_cast<unsigned int>(T), static_cast<unsigned int>(nHeads), 1u);
	const size_t smemBytes = static_cast<size_t>(flashTile) * static_cast<size_t>(2 * dHead) * sizeof(float);
	flash_attention_fwd_multihead_kernel_bf16<<<grid, block, smemBytes, computeStream()>>>(
		Q, K, V, T, nHeads, nKVHeads, dHead, dModel, dModelKV, causal ? 1 : 0, flashTile, O);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Local-window BF16 flash attention forward wrapper.  windowSize <= 0 falls
// back to full attention via flash_attention_multihead_forward_bf16.
// Paradigm #78: sinkCount > 0 forces the local-window kernel even if window
// is degenerate, so that the first `sinkCount` keys are always retained.
bool flash_attention_multihead_forward_bf16_local(const uint16_t* Q, const uint16_t* K,
                                                   const uint16_t* V,
                                                   int T, int nHeads, int nKVHeads,
                                                   int dHead, int dModel, int dModelKV,
                                                   bool causal, int windowSize,
                                                   float* O, int sinkCount)
{
	if (T <= 0 || nHeads <= 0 || nKVHeads <= 0 || dHead <= 0 || dModel <= 0 || dModelKV <= 0)
		return true;
	if ((windowSize <= 0 || windowSize >= T) && sinkCount <= 0) {
		return flash_attention_multihead_forward_bf16(
		    Q, K, V, T, nHeads, nKVHeads, dHead, dModel, dModelKV, causal, O);
	}

	const size_t sOBytes = static_cast<size_t>(kFlashQRows) * static_cast<size_t>(dHead) * sizeof(float);
	int dev = 0;
	cudaGetDevice(&dev);
	int maxOptin = 48 * 1024;
	cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
	const size_t minMultiQ = static_cast<size_t>(4) * static_cast<size_t>(2 * dHead) * sizeof(float) + sOBytes;
	const bool multiQFits = (minMultiQ <= static_cast<size_t>(maxOptin));

	if (!multiQFits) {
		// Fall back to full attention — local-only single-query path not implemented.
		return flash_attention_multihead_forward_bf16(
		    Q, K, V, T, nHeads, nKVHeads, dHead, dModel, dModelKV, causal, O);
	}

	const int flashTile = setup_flash_tile(flash_attention_fwd_local_kernel_bf16<kFlashQRows>, dHead, sOBytes);
	const int block = 32 * kFlashQRows;
	const dim3 grid(static_cast<unsigned int>((T + kFlashQRows - 1) / kFlashQRows),
	                static_cast<unsigned int>(nHeads), 1u);
	const size_t smemBytes = static_cast<size_t>(flashTile) * static_cast<size_t>(2 * dHead) * sizeof(float) + sOBytes;
	flash_attention_fwd_local_kernel_bf16<kFlashQRows><<<grid, block, smemBytes, computeStream()>>>(
	    Q, K, V, T, nHeads, nKVHeads, dHead, dModel, dModelKV,
	    causal ? 1 : 0, windowSize, sinkCount, flashTile, O);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool flash_attention_multihead_forward(const float* Q, const float* K, const float* V,
                                       int T, int nHeads, int nKVHeads,
                                       int dHead, int dModel, int dModelKV,
                                       bool causal, float* O)
{
	if (T <= 0 || nHeads <= 0 || nKVHeads <= 0 || dHead <= 0 || dModel <= 0 || dModelKV <= 0)
		return true;

	// Multi-query: each block handles kFlashQRows query rows. sO lives in
	// shared memory alongside sK/sV. Fall back to single-query warp kernel
	// if the multi-query budget doesn't fit even at flashTile=4.
	const size_t sOBytes = static_cast<size_t>(kFlashQRows) * static_cast<size_t>(dHead) * sizeof(float);
	int dev = 0;
	cudaGetDevice(&dev);
	int maxOptin = 48 * 1024;
	cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
	const size_t minMultiQ = static_cast<size_t>(4) * static_cast<size_t>(2 * dHead) * sizeof(float) + sOBytes;
	const bool multiQFits = (minMultiQ <= static_cast<size_t>(maxOptin));

	if (multiQFits) {
		const int flashTile = setup_flash_tile(flash_attention_fwd_multiq_kernel<kFlashQRows>, dHead, sOBytes);
		const int block = 32 * kFlashQRows;  // one warp per query row
		const dim3 grid(static_cast<unsigned int>((T + kFlashQRows - 1) / kFlashQRows),
		                static_cast<unsigned int>(nHeads), 1u);
		const size_t smemBytes = static_cast<size_t>(flashTile) * static_cast<size_t>(2 * dHead) * sizeof(float) + sOBytes;
		flash_attention_fwd_multiq_kernel<kFlashQRows><<<grid, block, smemBytes, computeStream()>>>(
			Q, K, V, T, nHeads, nKVHeads, dHead, dModel, dModelKV, causal ? 1 : 0, flashTile, O);
		GLADES_CUDA_CHECK(cudaGetLastError());
		return true;
	}

	// Fallback: single-query warp kernel.
	const int flashTile = setup_flash_tile(flash_attention_fwd_multihead_kernel, dHead);
	const int block = kFlashBlock;
	const dim3 grid(static_cast<unsigned int>(T), static_cast<unsigned int>(nHeads), 1u);
	const size_t smemBytes = static_cast<size_t>(flashTile) * static_cast<size_t>(2 * dHead) * sizeof(float);
	flash_attention_fwd_multihead_kernel<<<grid, block, smemBytes, computeStream()>>>(
		Q, K, V, T, nHeads, nKVHeads, dHead, dModel, dModelKV, causal ? 1 : 0, flashTile, O);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool flash_attention_multihead_backward(const float* Q, const float* K, const float* V,
                                        const float* O, const float* dO,
                                        int T, int nHeads, int nKVHeads,
                                        int dHead, int dModel, int dModelKV,
                                        bool causal,
                                        float* dQ, float* dK_out, float* dV_out)
{
	if (T <= 0 || nHeads <= 0 || nKVHeads <= 0 || dHead <= 0 || dModel <= 0 || dModelKV <= 0)
		return true;

	GLADES_CUDA_CHECK(cudaMemset(dK_out, 0, static_cast<size_t>(T) * static_cast<size_t>(dModelKV) * sizeof(float)));
	GLADES_CUDA_CHECK(cudaMemset(dV_out, 0, static_cast<size_t>(T) * static_cast<size_t>(dModelKV) * sizeof(float)));

	// Multi-query backward: no sO in shmem (dqRow accumulated directly in global).
	int dev = 0;
	cudaGetDevice(&dev);
	int maxOptin = 48 * 1024;
	cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
	const size_t minMultiQ = static_cast<size_t>(4) * static_cast<size_t>(2 * dHead) * sizeof(float);
	const bool multiQFits = (minMultiQ <= static_cast<size_t>(maxOptin));

	if (multiQFits) {
		const int flashTile = setup_flash_tile(flash_attention_bwd_multiq_kernel<kFlashQRows>, dHead);
		const int block = 32 * kFlashQRows;
		const dim3 grid(static_cast<unsigned int>((T + kFlashQRows - 1) / kFlashQRows),
		                static_cast<unsigned int>(nHeads), 1u);
		const size_t smemBytes = static_cast<size_t>(flashTile) * static_cast<size_t>(2 * dHead) * sizeof(float);
		flash_attention_bwd_multiq_kernel<kFlashQRows><<<grid, block, smemBytes, computeStream()>>>(
			Q, K, V, O, dO, T, nHeads, nKVHeads, dHead, dModel, dModelKV, causal ? 1 : 0,
			flashTile, dQ, dK_out, dV_out);
		GLADES_CUDA_CHECK(cudaGetLastError());
		return true;
	}

	// Fallback: single-query warp kernel.
	const int flashTile = setup_flash_tile(flash_attention_bwd_multihead_kernel, dHead);
	const int block = kFlashBlock;
	const dim3 grid(static_cast<unsigned int>(T), static_cast<unsigned int>(nHeads), 1u);
	const size_t smemBytes = static_cast<size_t>(flashTile) * static_cast<size_t>(2 * dHead) * sizeof(float);
	flash_attention_bwd_multihead_kernel<<<grid, block, smemBytes, computeStream()>>>(
		Q, K, V, O, dO, T, nHeads, nKVHeads, dHead, dModel, dModelKV, causal ? 1 : 0,
		flashTile, dQ, dK_out, dV_out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  16. Incremental KV-cache attention (single-query, multi-head)
// ===========================================================================
//
// One block per query head.  Each block:
//   Phase 1: Compute scores[0..pos] = Q[hq] . K_cache[u, kvHead] * invSqrt
//   Phase 2: Parallel reduction to find max score
//   Phase 3: Compute exp(score - max), parallel reduction for sum
//   Phase 4: Normalize probabilities
//   Phase 5: Weighted sum of V_cache -> output head
//
// scores_scratch[nHeads, maxLen] is a global memory buffer to avoid
// shared-memory limits for long sequences.

namespace {

__global__ void kv_attn_incremental_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    float* __restrict__ scores,
    const unsigned char* __restrict__ keyValid,
    int nHeads, int nKVHeads, int dHead,
    int dModelKV, int maxLen, int pos,
    float invSqrt,
    float* __restrict__ out)
{
	const int hq = blockIdx.x;
	if (hq >= nHeads) return;

	const int groupSize = (nKVHeads > 0 && nKVHeads < nHeads) ? (nHeads / nKVHeads) : 1;
	const int kvHead = (nKVHeads == nHeads) ? hq : (hq / groupSize);
	const int seqLen = pos + 1;
	const int tid = threadIdx.x;
	const int nThreads = blockDim.x;

	const float* qh = Q + hq * dHead;
	const int kvOff = kvHead * dHead;
	float* myScores = scores + (size_t)hq * maxLen;

	// Shared memory for parallel reduction: [nThreads]
	extern __shared__ float reduce[];

	// Phase 1: Compute dot-product scores
	for (int u = tid; u < seqLen; u += nThreads) {
		if (keyValid && keyValid[u] == 0) {
			myScores[u] = -1e30f;
		} else {
			const float* ku = K_cache + (size_t)u * dModelKV + kvOff;
			float dot = 0.0f;
			for (int d = 0; d < dHead; ++d)
				dot += qh[d] * ku[d];
			myScores[u] = dot * invSqrt;
		}
	}
	__syncthreads();

	// Phase 2: Find max score (parallel reduction)
	float localMax = -1e30f;
	for (int u = tid; u < seqLen; u += nThreads)
		localMax = fmaxf(localMax, myScores[u]);
	reduce[tid] = localMax;
	__syncthreads();
	for (int s = nThreads / 2; s > 0; s >>= 1) {
		if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
		__syncthreads();
	}
	float maxVal = reduce[0];
	__syncthreads();

	// Phase 3: Exp and sum
	float localSum = 0.0f;
	for (int u = tid; u < seqLen; u += nThreads) {
		float e = expf(myScores[u] - maxVal);
		myScores[u] = e;
		localSum += e;
	}
	reduce[tid] = localSum;
	__syncthreads();
	for (int s = nThreads / 2; s > 0; s >>= 1) {
		if (tid < s) reduce[tid] += reduce[tid + s];
		__syncthreads();
	}
	float sumVal = reduce[0];
	__syncthreads();

	// Phase 4: Normalize to probabilities
	float invS = (sumVal > 0.0f) ? (1.0f / sumVal) : 0.0f;
	for (int u = tid; u < seqLen; u += nThreads)
		myScores[u] *= invS;
	__syncthreads();

	// Phase 5: Weighted sum of V values -> output
	float* oh = out + hq * dHead;
	for (int d = tid; d < dHead; d += nThreads) {
		float acc = 0.0f;
		for (int u = 0; u < seqLen; ++u)
			acc += myScores[u] * V_cache[(size_t)u * dModelKV + kvOff + d];
		oh[d] = acc;
	}
}

} // anonymous namespace

bool kv_attention_incremental(const float* Q,
                              const float* K_cache, const float* V_cache,
                              float* scores_scratch,
                              const unsigned char* keyValid,
                              int nHeads, int nKVHeads, int dHead,
                              int dModelKV, int maxLen, int pos,
                              float invSqrt, float* out)
{
	if (nHeads <= 0 || dHead <= 0 || pos < 0) return true;

	int block = 256;
	if (block > kMaxBlockRow) block = kMaxBlockRow;
	size_t smemBytes = (size_t)block * sizeof(float);

	kv_attn_incremental_kernel<<<nHeads, block, smemBytes, computeStream()>>>(
		Q, K_cache, V_cache, scores_scratch, keyValid,
		nHeads, nKVHeads, dHead, dModelKV, maxLen, pos,
		invSqrt, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ---------------------------------------------------------------------------
// Device memory operations
// ---------------------------------------------------------------------------

void device_memcpy_d2d(void* dst, const void* src, size_t bytes)
{
	cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, computeStream());
}

void device_memcpy_h2d(void* dst, const void* src, size_t bytes)
{
	cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, transferStream());
}

void device_memcpy_d2h(void* dst, const void* src, size_t bytes)
{
	cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, transferStream());
}

void device_memcpy_2d_d2d(void* dst, size_t dpitch, const void* src, size_t spitch,
                           size_t width, size_t height)
{
	cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, computeStream());
}

void device_memset_bytes(void* ptr, int value, size_t bytes)
{
	cudaMemsetAsync(ptr, value, bytes, computeStream());
}

// ===========================================================================
//  Batch zero: zero multiple GPU buffers with a single kernel launch
// ===========================================================================

namespace {

// Each block zeros one buffer. ptrs[blockIdx.x] is the pointer,
// sizes[blockIdx.x] is the number of floats to zero.
__global__ void zero_multi_buffers_kernel(float** __restrict__ ptrs,
                                          const int* __restrict__ sizes)
{
	float* buf = ptrs[blockIdx.x];
	int n = sizes[blockIdx.x];
	for (int i = threadIdx.x; i < n; i += blockDim.x)
		buf[i] = 0.0f;
}

} // anonymous namespace

bool zero_buffers_batch(float** d_ptrs, const int* d_sizes, int count)
{
	if (count <= 0) return true;
	zero_multi_buffers_kernel<<<count, 256, 0, computeStream()>>>(d_ptrs, d_sizes);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  Pack loss scalars: copy 4 scalar device values into a contiguous buffer
// ===========================================================================

namespace {

__global__ void pack_loss_scalars_kernel(const float* __restrict__ lossSum,
                                          const int* __restrict__ lossCount,
                                          const int* __restrict__ correctCount,
                                          const int* __restrict__ validCount,
                                          int* __restrict__ out)
{
	if (threadIdx.x == 0)
	{
		// Reinterpret float as int bits for the first slot.
		const int* lossBits = reinterpret_cast<const int*>(lossSum);
		out[0] = lossBits[0];
		out[1] = lossCount[0];
		out[2] = correctCount[0];
		out[3] = validCount[0];
	}
}

} // anonymous namespace

bool pack_loss_scalars(const float* lossSum, const int* lossCount,
                       const int* correctCount, const int* validCount,
                       int* out)
{
	pack_loss_scalars_kernel<<<1, 1, 0, computeStream()>>>(lossSum, lossCount, correctCount, validCount, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

namespace {

__global__ void collect_token_lm_metrics_kernel(const float* __restrict__ probs,
                                                const int* __restrict__ targets,
                                                int T, int vocabSize, int padToken,
                                                int* __restrict__ out)
{
	extern __shared__ float smem[];
	float* sLoss = smem;
	float* sValid = sLoss + blockDim.x;
	float* sCorrect = sValid + blockDim.x;

	float localLoss = 0.0f;
	float localValid = 0.0f;
	float localCorrect = 0.0f;

	for (int t = threadIdx.x; t < T; t += blockDim.x)
	{
		const int tgt = targets[t];
		if (padToken >= 0 && tgt == padToken)
			continue;
		if (tgt < 0 || tgt >= vocabSize)
			continue;

		const float* row = probs + static_cast<size_t>(t) * vocabSize;
		float p = row[tgt];
		if (p < 1e-12f)
			p = 1e-12f;
		localLoss += -logf(p);
		localValid += 1.0f;

		int bestIdx = 0;
		float bestVal = row[0];
		for (int v = 1; v < vocabSize; ++v)
		{
			if (row[v] > bestVal)
			{
				bestVal = row[v];
				bestIdx = v;
			}
		}
		if (bestIdx == tgt)
			localCorrect += 1.0f;
	}

	sLoss[threadIdx.x] = localLoss;
	sValid[threadIdx.x] = localValid;
	sCorrect[threadIdx.x] = localCorrect;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (threadIdx.x < stride)
		{
			sLoss[threadIdx.x] += sLoss[threadIdx.x + stride];
			sValid[threadIdx.x] += sValid[threadIdx.x + stride];
			sCorrect[threadIdx.x] += sCorrect[threadIdx.x + stride];
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		union
		{
			float f;
			int i;
		} lossBits;
		lossBits.f = sLoss[0];
		out[0] = lossBits.i;
		out[1] = static_cast<int>(sValid[0]);
		out[2] = static_cast<int>(sCorrect[0]);
		out[3] = static_cast<int>(sValid[0]);
	}
}

} // anonymous namespace

bool collect_token_lm_metrics(const float* probs, const int* targets,
                              int T, int vocabSize, int padToken,
                              int* out)
{
	if (T <= 0 || vocabSize <= 0)
		return true;
	// PERF FIX (see memory/perf_regression_apr16.md): the previous
	// collect_token_lm_metrics_kernel launched with <<<1, 256>>> — a single
	// thread block of 256 threads processing T×vocabSize work entirely
	// inside one block.  At T=2048, vocabSize=32000 this was ~24 ms/call,
	// consuming ~8% of GPU time per training step.
	//
	// The properly-parallelized implementation is already present as two
	// separate kernels (cross_entropy_nll_loss + argmax_count_matches) —
	// both use grid=(T+block-1)/block clamped at 128 blocks.  Route
	// collect_token_lm_metrics through those to restore the pre-eecdb97c1
	// throughput.
	//
	// Output layout (unchanged for call-site compatibility):
	//   out[0] = loss_sum  (bit-cast float)
	//   out[1] = valid_count
	//   out[2] = correct_count
	//   out[3] = samples_count  (= valid_count for now)
	float* loss_sum       = reinterpret_cast<float*>(&out[0]);
	int*   loss_count     = &out[1];
	int*   correct_count  = &out[2];
	int*   samples_count  = &out[3];
	if (!cross_entropy_nll_loss(probs, targets, T, vocabSize, padToken,
	                            loss_sum, loss_count))
		return false;
	if (!argmax_count_matches(probs, targets, T, vocabSize, padToken,
	                          correct_count, samples_count))
		return false;
	return true;
}

// ===========================================================================
//  17. Sum of squared elements (for gradient norm computation)
// ===========================================================================

namespace {

__global__ void sum_sq_kernel(const float* __restrict__ data, int n,
                              float* __restrict__ acc)
{
	extern __shared__ float smem[];
	float localSum = 0.0f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += gridDim.x * blockDim.x)
	{
		float v = data[i];
		localSum += v * v;
	}
	float sum = blockReduceSum(localSum, smem);
	if (threadIdx.x == 0)
		atomicAdd(acc, sum);
}

} // anonymous namespace

bool sum_squared_accumulate(const float* data, int n, float* d_accumulator)
{
	if (n <= 0) return true;
	int block = 256;
	int grid = (n + block - 1) / block;
	if (grid > 256) grid = 256;
	int smemBytes = ((block / 32) + 1) * sizeof(float);
	sum_sq_kernel<<<grid, block, smemBytes, computeStream()>>>(data, n, d_accumulator);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

namespace {

__global__ void sum_sq_bf16_kernel(const uint16_t* __restrict__ data, int n,
                                    float* __restrict__ acc)
{
	extern __shared__ float smem[];
	float localSum = 0.0f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += gridDim.x * blockDim.x)
	{
		union { uint32_t u; float f; } uv;
		uv.u = static_cast<uint32_t>(data[i]) << 16;
		float v = uv.f;
		localSum += v * v;
	}
	float sum = blockReduceSum(localSum, smem);
	if (threadIdx.x == 0)
		atomicAdd(acc, sum);
}

} // anonymous namespace

bool sum_squared_accumulate_bf16(const uint16_t* data, int n, float* d_accumulator)
{
	if (n <= 0) return true;
	int block = 256;
	int grid = (n + block - 1) / block;
	if (grid > 256) grid = 256;
	int smemBytes = ((block / 32) + 1) * sizeof(float);
	sum_sq_bf16_kernel<<<grid, block, smemBytes, computeStream()>>>(data, n, d_accumulator);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
// BF16 / FP32 cast kernels — foundation for mixed-precision training.
//
// BF16 keeps FP32's 8-bit exponent and truncates mantissa to 7 bits (vs FP16's
// 5-exp/10-mantissa). Because the exponent range matches FP32, conversion is
// a simple high-half-word extraction with round-to-nearest-even.
//
// These cast kernels are the primitives for:
//   * BF16 weight storage (halves weight VRAM; FP32 master weights in optimizer)
//   * BF16 gradient storage
//   * Any buffer where we want to sacrifice mantissa precision for capacity
// ===========================================================================

namespace {

__global__ void k_cast_f32_to_bf16(const float* __restrict__ src,
                                   uint16_t* __restrict__ dst,
                                   size_t n)
{
	const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	// Reinterpret float as uint32, apply RN-even rounding on the low half-word,
	// and take the upper 16 bits.
	const float f = src[idx];
	union { float f; uint32_t u; } v;
	v.f = f;
	// Flush NaN to BF16 quiet NaN (preserve sign, set high mantissa bit).
	if (isnan(f)) {
		const uint32_t sign = v.u & 0x80000000u;
		dst[idx] = static_cast<uint16_t>(((sign | 0x7FC00000u) >> 16) & 0xFFFFu);
		return;
	}
	const uint32_t lsb = (v.u >> 16) & 1u;
	const uint32_t roundingBias = 0x7FFFu + lsb;
	dst[idx] = static_cast<uint16_t>((v.u + roundingBias) >> 16);
}

__global__ void k_cast_bf16_to_f32(const uint16_t* __restrict__ src,
                                   float* __restrict__ dst,
                                   size_t n)
{
	const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	union { uint32_t u; float f; } v;
	v.u = static_cast<uint32_t>(src[idx]) << 16;
	dst[idx] = v.f;
}

} // anonymous namespace

bool cast_f32_to_bf16(const float* src, uint16_t* dst, size_t n)
{
	if (n == 0) return true;
	const unsigned int TPB = 256u;
	const size_t blocks = (n + TPB - 1u) / TPB;
	// Grid cap to avoid >2^31 block count on extremely large buffers; the
	// kernel strides aren't needed below that because we size n per the caller.
	if (blocks > 0x7FFFFFFFu) return false;
	k_cast_f32_to_bf16<<<static_cast<unsigned int>(blocks), TPB, 0, computeStream()>>>(src, dst, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool cast_bf16_to_f32(const uint16_t* src, float* dst, size_t n)
{
	if (n == 0) return true;
	const unsigned int TPB = 256u;
	const size_t blocks = (n + TPB - 1u) / TPB;
	if (blocks > 0x7FFFFFFFu) return false;
	k_cast_bf16_to_f32<<<static_cast<unsigned int>(blocks), TPB, 0, computeStream()>>>(src, dst, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  Stochastic-rounded FP32 -> BF16 cast
// ===========================================================================
// Deterministic RN-even rounding quantizes away updates smaller than one ULP
// (1 / 2^8 = 1/256 of the weight magnitude for BF16).  In training that stalls
// small gradient accumulations indefinitely.
//
// Stochastic rounding instead rounds:
//   - up   with probability  p = frac / (1 ULP)
//   - down with probability  1 - p
// where frac is the low 16 bits of the FP32 representation (the BF16 mantissa
// remainder).  Matches deterministic RN in expectation; preserves sub-ULP
// updates over many steps.
//
// RNG: a simple per-element splittable hash seeded from (baseSeed, idx, step).
// No global RNG state needed; each element's rounding is independent.

namespace {

__device__ __forceinline__ uint32_t sr_hash32(uint32_t a, uint32_t b, uint32_t c)
{
	// xorshift-mixed hash — cheap, good enough for rounding randomness.
	uint32_t x = a ^ (b * 0x9E3779B1u) ^ (c * 0x85EBCA6Bu);
	x ^= x >> 16; x *= 0x7FEB352Du;
	x ^= x >> 15; x *= 0x846CA68Bu;
	x ^= x >> 16;
	return x;
}

__global__ void k_cast_f32_to_bf16_stochastic(const float* __restrict__ src,
                                               uint16_t* __restrict__ dst,
                                               size_t n,
                                               uint32_t baseSeed,
                                               uint32_t stepIdx)
{
	const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	const float f = src[idx];
	union { float f; uint32_t u; } v;
	v.f = f;

	// NaN: emit BF16 quiet NaN preserving sign (same as RN path).
	if (isnan(f))
	{
		const uint32_t sign = v.u & 0x80000000u;
		dst[idx] = static_cast<uint16_t>(((sign | 0x7FC00000u) >> 16) & 0xFFFFu);
		return;
	}

	// Stochastic rounding: compare low 16 bits against a per-element hash.
	// If hash's low 16 bits < low16 of the float, we round up (= truncate high16 + 1).
	// Otherwise truncate toward zero w.r.t. the low bits (= high16 alone).
	const uint32_t low16 = v.u & 0xFFFFu;
	const uint32_t rnd = sr_hash32(static_cast<uint32_t>(idx),
	                                 stepIdx, baseSeed) & 0xFFFFu;
	uint32_t high16 = v.u >> 16;
	if (rnd < low16)
	{
		// Round up; propagate carry if mantissa overflows.  For BF16 encoding
		// the high16 IS {sign | exp | m[6..0]} so a simple +1 is correct for
		// non-NaN inputs (exp increments take care of mantissa overflow).
		high16 += 1u;
	}
	dst[idx] = static_cast<uint16_t>(high16 & 0xFFFFu);
}

} // anonymous namespace

bool cast_f32_to_bf16_stochastic(const float* src, uint16_t* dst, size_t n,
                                  uint32_t baseSeed, uint32_t stepIdx)
{
	if (n == 0) return true;
	const unsigned int TPB = 256u;
	const size_t blocks = (n + TPB - 1u) / TPB;
	if (blocks > 0x7FFFFFFFu) return false;
	k_cast_f32_to_bf16_stochastic<<<static_cast<unsigned int>(blocks), TPB, 0, computeStream()>>>(
	    src, dst, n, baseSeed, stepIdx);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  BF16 gradient accumulation: write dst_bf16 = bf16(alpha * src_f32 + fp32(dst_bf16) * beta)
// ===========================================================================
// Used for BF16 gradient accumulation across gradient-accumulation micro-steps:
//   - First micro-step of a window: beta=0, alpha=1 — writes bf16(src)
//   - Subsequent micro-steps: beta=1, alpha=1 — adds in src to existing bf16 accum
// The cast back to BF16 uses the same RN-even rule as cast_f32_to_bf16.

namespace {

__global__ void k_bf16_accum_axpy(uint16_t* __restrict__ dst_bf16,
                                   const float* __restrict__ src_f32,
                                   float alpha, float beta,
                                   size_t n)
{
	const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	// Decode existing BF16 accumulator.
	union { uint32_t u; float f; } uv;
	uv.u = static_cast<uint32_t>(dst_bf16[idx]) << 16;
	const float acc = uv.f * beta + alpha * src_f32[idx];

	// Re-encode with round-to-nearest-even.
	union { float f; uint32_t u; } v;
	v.f = acc;
	if (isnan(acc)) {
		const uint32_t sign = v.u & 0x80000000u;
		dst_bf16[idx] = static_cast<uint16_t>(((sign | 0x7FC00000u) >> 16) & 0xFFFFu);
		return;
	}
	const uint32_t lsb = (v.u >> 16) & 1u;
	const uint32_t roundingBias = 0x7FFFu + lsb;
	dst_bf16[idx] = static_cast<uint16_t>((v.u + roundingBias) >> 16);
}

} // anonymous namespace

bool bf16_accum_axpy(uint16_t* dst_bf16, const float* src_f32,
                      float alpha, float beta, size_t n)
{
	if (n == 0) return true;
	const unsigned int TPB = 256u;
	const size_t blocks = (n + TPB - 1u) / TPB;
	if (blocks > 0x7FFFFFFFu) return false;
	k_bf16_accum_axpy<<<static_cast<unsigned int>(blocks), TPB, 0, computeStream()>>>(
	    dst_bf16, src_f32, alpha, beta, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ===========================================================================
//  Paradigm-shift GPU kernels (impl(paradigm-74/76/78) round 2)
// ===========================================================================
// Mirror the CPU primitives in transformer_ops.h. Initial implementations
// are correctness-first; production tuning (warp-level reductions, tensor
// cores, async pipelines) can follow once parity is established.

namespace {

// ---------- #74 PHOENIX-1BIT GPU GEMM (column-major bit-packed weights) ----
// Each block computes a tile of Y[m*BM .. m*BM+BM, n*BN .. n*BN+BN]; one
// thread per output. Inner loop walks K bytes for the chosen column n.
// W_bits layout: column-major (bits for column n at bytes [n*Kbytes ..
// n*Kbytes+Kbytes)). bit k of column n = (W[k,n] >= 0).
__global__ void phoenix_binary_gemm_colmajor_kernel(const float* __restrict__ X,
                                                    const unsigned char* __restrict__ W_bits,
                                                    int M, int N, int K, int Kbytes,
                                                    float* __restrict__ Y)
{
	const int m = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (m >= M || n >= N) return;

	const float* xm = X + (size_t)m * K;
	const unsigned char* col = W_bits + (size_t)n * Kbytes;

	float rowSum = 0.0f;
	for (int k = 0; k < K; ++k)
		rowSum += xm[k];

	float maskedSum = 0.0f;
	int kBase = 0;
	for (int bb = 0; bb < Kbytes; ++bb, kBase += 8)
	{
		const unsigned char byte = col[bb];
		const int kEnd = (kBase + 8 <= K) ? (kBase + 8) : K;
		if (byte == 0u) continue;
		if (byte == 0xFFu && kBase + 8 <= K)
		{
			maskedSum += xm[kBase + 0] + xm[kBase + 1] + xm[kBase + 2] + xm[kBase + 3] +
			             xm[kBase + 4] + xm[kBase + 5] + xm[kBase + 6] + xm[kBase + 7];
			continue;
		}
		for (int j = 0; j < (kEnd - kBase); ++j)
			if (byte & (1u << j))
				maskedSum += xm[kBase + j];
	}

	Y[(size_t)m * N + n] = 2.0f * maskedSum - rowSum;
}

// ---------- #78 sink+window attention (FP32, single-head) ----------
// One block (one warp = 32 threads) per query position. The warp cooperatively
// reduces over dHead for the score dot-product and for the V-weighted output
// update. Thread 0 holds the scalar online-softmax state (m, l) and broadcasts
// {alpha, beta, newM} via warp shuffle.
//
// Optimization log (perf skill phases 1-4):
//   - Phase 1: profiled the single-thread-per-block reference kernel
//     (5.10x speedup at T=1024 dHead=64, dominated by a serial dHead=64
//     dot product and dHead=64 V-weighted update per key).
//   - Phase 3: parallelize the two dHead loops across 32 threads via
//     warp-stride access, warp-reduce the dot product, broadcast the
//     softmax scalars.
//
// O[T, dHead], Q/K/V[T, dHead] strided. invSqrt = 1/sqrt(dHead).
__global__ void sw_attention_forward_kernel(const float* __restrict__ Q,
                                            int qStride,
                                            const float* __restrict__ K,
                                            int kStride,
                                            const float* __restrict__ V,
                                            int vStride,
                                            int T, int dHead,
                                            int causal,
                                            int sinkCount,
                                            int windowSize,
                                            float invSqrt,
                                            float* __restrict__ O,
                                            int oStride)
{
	const int t = blockIdx.x;
	if (t >= T) return;
	const int tid = threadIdx.x;
	const unsigned mask = 0xFFFFFFFFu;

	const float* qt = Q + (size_t)t * qStride;
	float* ot = O + (size_t)t * oStride;
	const int maxU = causal ? t : (T - 1);

	// Initialize O cooperatively
	for (int d = tid; d < dHead; d += 32)
		ot[d] = 0.0f;

	// Scalar state (thread 0 only)
	float m_state = -1e30f;
	float l_state = 0.0f;
	bool any = false;

	for (int u = 0; u <= maxU; ++u)
	{
		bool allowed = false;
		if (sinkCount == 0 && windowSize == 0)
			allowed = true;
		else if (u < sinkCount)
			allowed = true;
		else if (windowSize > 0 && u + windowSize > t)
			allowed = true;
		if (!allowed) continue;

		const float* ku = K + (size_t)u * kStride;

		// Warp-parallel dot product: each thread accumulates a stride-32 slice.
		float partial = 0.0f;
		for (int d = tid; d < dHead; d += 32)
			partial += qt[d] * ku[d];
		// Warp-reduce
		for (int off = 16; off > 0; off >>= 1)
			partial += __shfl_xor_sync(mask, partial, off);
		// All threads now have the dot product in `partial`.
		const float s = partial * invSqrt;

		if (!any)
		{
			any = true;
			m_state = s;
			l_state = 1.0f;
			const float* vu = V + (size_t)u * vStride;
			for (int d = tid; d < dHead; d += 32)
				ot[d] = vu[d];
			continue;
		}
		const float newM = (s > m_state) ? s : m_state;
		const float alpha = expf(m_state - newM);
		const float beta = expf(s - newM);
		l_state = l_state * alpha + beta;
		m_state = newM;

		const float* vu = V + (size_t)u * vStride;
		for (int d = tid; d < dHead; d += 32)
			ot[d] = ot[d] * alpha + beta * vu[d];
	}

	if (!any || !(l_state > 0.0f)) return;
	const float invL = 1.0f / l_state;
	for (int d = tid; d < dHead; d += 32)
		ot[d] *= invL;
}

} // anonymous namespace

// #74 wrapper
bool phoenix_binary_gemm_gpu(const float* X,
                              const unsigned char* W_bits,
                              int M, int N, int K,
                              float* Y)
{
	if (M <= 0 || N <= 0 || K <= 0) return true;
	const int Kbytes = (K + 7) / 8;
	const dim3 block(16u, 16u, 1u);
	const dim3 grid((N + (int)block.x - 1) / (int)block.x,
	                (M + (int)block.y - 1) / (int)block.y, 1u);
	phoenix_binary_gemm_colmajor_kernel<<<grid, block, 0, computeStream()>>>(
	    X, W_bits, M, N, K, Kbytes, Y);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// #78 wrapper (single-head FP32; parity scope).
bool sw_attention_forward_gpu(const float* Q, int qStride,
                               const float* K, int kStride,
                               const float* V, int vStride,
                               int T, int dHead, bool causal,
                               int sinkCount, int windowSize,
                               float* O, int oStride)
{
	if (T <= 0 || dHead <= 0) return true;
	const float invSqrt = 1.0f / sqrtf((float)dHead);
	const dim3 grid((unsigned int)T, 1u, 1u);
	const dim3 block(32u, 1u, 1u);  // single-thread per block (correctness scope)
	sw_attention_forward_kernel<<<grid, block, 0, computeStream()>>>(
	    Q, qStride, K, kStride, V, vStride, T, dHead, causal ? 1 : 0,
	    sinkCount, windowSize, invSqrt, O, oStride);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ---------- #78 sink+window attention backward (FP32, single-head) -----
// Recompute-style backward (mirrors CPU
// scaled_dot_product_attention_backward_recompute_flash_strided_sw).
// One block per query; all dQ writes are local to that block. dK/dV writes
// across queries use atomicAdd.
__global__ void sw_attention_backward_kernel(const float* __restrict__ Q,
                                             int qStride,
                                             const float* __restrict__ K,
                                             int kStride,
                                             const float* __restrict__ V,
                                             int vStride,
                                             const float* __restrict__ dO,
                                             int dOStride,
                                             int T, int dHead,
                                             int causal,
                                             int sinkCount,
                                             int windowSize,
                                             float invSqrt,
                                             float* __restrict__ dQ,
                                             int dQStride,
                                             float* __restrict__ dK_out,
                                             int dKStride,
                                             float* __restrict__ dV_out,
                                             int dVStride)
{
	const int t = blockIdx.x;
	if (t >= T) return;
	const int tid = threadIdx.x;
	const unsigned mask = 0xFFFFFFFFu;

	const float* qt = Q + (size_t)t * qStride;
	const float* dOt = dO + (size_t)t * dOStride;
	float* dQt = dQ + (size_t)t * dQStride;
	const int maxU = causal ? t : (T - 1);

	// Pass 1: compute online softmax (m, l) over allowed keys (warp-parallel
	// dot product); cache scores via single-thread state.  Full caching of
	// per-key scores in shared memory would be ideal but T may exceed shmem;
	// we just pass over keys twice (recompute in pass 2/3).
	float m_state = -1e30f;
	float l_state = 0.0f;
	bool any = false;

	for (int u = 0; u <= maxU; ++u)
	{
		bool allowed = (sinkCount == 0 && windowSize == 0)
		                  || (u < sinkCount)
		                  || (windowSize > 0 && u + windowSize > t);
		if (!allowed) continue;

		const float* ku = K + (size_t)u * kStride;
		float partial = 0.0f;
		for (int d = tid; d < dHead; d += 32)
			partial += qt[d] * ku[d];
		for (int off = 16; off > 0; off >>= 1)
			partial += __shfl_xor_sync(mask, partial, off);
		const float s = partial * invSqrt;

		if (!any) { any = true; m_state = s; l_state = 1.0f; continue; }
		const float newM = (s > m_state) ? s : m_state;
		const float alpha = expf(m_state - newM);
		const float beta = expf(s - newM);
		l_state = l_state * alpha + beta;
		m_state = newM;
	}
	if (!any || !(l_state > 0.0f)) return;
	const float inv_l = 1.0f / l_state;

	// Pass 2: compute rowDot = sum_u p_u * dP_u and accumulate dV.
	// We recompute s and p per key.
	float rowDot_partial = 0.0f;
	for (int u = 0; u <= maxU; ++u)
	{
		bool allowed = (sinkCount == 0 && windowSize == 0)
		                  || (u < sinkCount)
		                  || (windowSize > 0 && u + windowSize > t);
		if (!allowed) continue;

		const float* ku = K + (size_t)u * kStride;
		const float* vu = V + (size_t)u * vStride;

		// Recompute s
		float partial = 0.0f;
		for (int d = tid; d < dHead; d += 32)
			partial += qt[d] * ku[d];
		for (int off = 16; off > 0; off >>= 1)
			partial += __shfl_xor_sync(mask, partial, off);
		const float s = partial * invSqrt;
		const float pf = expf(s - m_state) * inv_l;

		// dP = dot(dOt, vu)
		float dP_part = 0.0f;
		for (int d = tid; d < dHead; d += 32)
			dP_part += dOt[d] * vu[d];
		for (int off = 16; off > 0; off >>= 1)
			dP_part += __shfl_xor_sync(mask, dP_part, off);
		// dP_part now in all lanes.

		// rowDot accumulator (only thread 0 holds the running sum; we add
		// pf*dP, broadcast not needed since dP_part is the same across lanes).
		if (tid == 0) rowDot_partial += pf * dP_part;

		// dV[u] += pf * dOt; atomic per-lane elementwise.
		for (int d = tid; d < dHead; d += 32)
		{
			float* dVu = dV_out + (size_t)u * dVStride + d;
			atomicAdd(dVu, pf * dOt[d]);
		}
	}
	float rowDot = __shfl_sync(mask, rowDot_partial, 0);

	// Pass 3: compute ds_u = pf * (dP_u - rowDot) * invSqrt, then accumulate
	// dQ[t] += ds * ku and dK[u] += ds * qt.
	for (int u = 0; u <= maxU; ++u)
	{
		bool allowed = (sinkCount == 0 && windowSize == 0)
		                  || (u < sinkCount)
		                  || (windowSize > 0 && u + windowSize > t);
		if (!allowed) continue;

		const float* ku = K + (size_t)u * kStride;
		const float* vu = V + (size_t)u * vStride;

		// Recompute s, pf
		float partial = 0.0f;
		for (int d = tid; d < dHead; d += 32)
			partial += qt[d] * ku[d];
		for (int off = 16; off > 0; off >>= 1)
			partial += __shfl_xor_sync(mask, partial, off);
		const float s = partial * invSqrt;
		const float pf = expf(s - m_state) * inv_l;

		// Recompute dP
		float dP_part = 0.0f;
		for (int d = tid; d < dHead; d += 32)
			dP_part += dOt[d] * vu[d];
		for (int off = 16; off > 0; off >>= 1)
			dP_part += __shfl_xor_sync(mask, dP_part, off);
		const float ds = pf * (dP_part - rowDot) * invSqrt;
		if (ds == 0.0f) continue;

		// dQ[t] += ds * ku  (each lane contributes its slice)
		for (int d = tid; d < dHead; d += 32)
			atomicAdd(&dQt[d], ds * ku[d]);
		// dK[u] += ds * qt
		float* dKu = dK_out + (size_t)u * dKStride;
		for (int d = tid; d < dHead; d += 32)
			atomicAdd(&dKu[d], ds * qt[d]);
	}
}

// ---------- #74 PHOENIX-1BIT FFN-side helper: Y = X @ sign(W).T -----------
// W[N, K] row-major, treated as binary {-1, +1} via sign. Mirrors
// gpu_gemm_abt_mp's interface for in-place FFN replacement.
//
// Optimized tiled kernel:
//   - Shared-memory blocking on X and W tiles (BM × BK and BN × BK)
//   - Each thread accumulates one output (m, n)
//   - Outer K-loop walks tiles of width BK; inner K-loop inside shmem
//   - Sign extracted from float W on-the-fly (no separate packed buffer)
//
// At BM=BN=32, BK=32, this achieves 32-fold X-row reuse across N outputs
// and 32-fold W-col reuse across M outputs, bringing it within range of
// cuBLAS at moderate shapes while staying multiply-free.
namespace {

template <int BM, int BN, int BK>
__global__ void binary_gemm_abt_from_float_tiled_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    int M, int N, int K,
    float* __restrict__ Y)
{
	const int blkM = blockIdx.y * BM;
	const int blkN = blockIdx.x * BN;
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;
	const int gm = blkM + ty;
	const int gn = blkN + tx;

	__shared__ float Xtile[BM][BK];
	__shared__ float Wtile[BN][BK];

	float acc = 0.0f;

	for (int kBase = 0; kBase < K; kBase += BK)
	{
		// Cooperatively load X[blkM:blkM+BM, kBase:kBase+BK]
		// Each thread loads BM*BK / (BM*BN) = BK/BN entries
		#pragma unroll
		for (int kk = tx; kk < BK; kk += BN)
		{
			const int gk = kBase + kk;
			Xtile[ty][kk] = (gm < M && gk < K)
			    ? X[(size_t)gm * K + gk]
			    : 0.0f;
		}
		// Cooperatively load W[blkN:blkN+BN, kBase:kBase+BK]
		// Each thread loads similarly
		#pragma unroll
		for (int kk = ty; kk < BK; kk += BM)
		{
			const int gk = kBase + kk;
			Wtile[tx][kk] = (gn < N && gk < K)
			    ? W[(size_t)gn * K + gk]
			    : 0.0f;
		}
		__syncthreads();

		// Inner loop: my (gm, gn) accumulates over BK
		if (gm < M && gn < N)
		{
			#pragma unroll
			for (int kk = 0; kk < BK; ++kk)
			{
				const float xv = Xtile[ty][kk];
				const float wv = Wtile[tx][kk];
				// sign(wv) * xv via branchless select
				acc += (wv >= 0.0f) ? xv : -xv;
			}
		}
		__syncthreads();
	}

	if (gm < M && gn < N)
		Y[(size_t)gm * N + gn] = acc;
}

// Naive scalar kernel kept as a fallback for shapes that don't tile nicely.
__global__ void binary_gemm_abt_from_float_kernel(const float* __restrict__ X,
                                                  const float* __restrict__ W,
                                                  int M, int N, int K,
                                                  float* __restrict__ Y)
{
	const int t = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= M || n >= N) return;
	const float* xt = X + (size_t)t * K;
	const float* wn = W + (size_t)n * K;
	float sum = 0.0f;
	for (int k = 0; k < K; ++k)
		sum += (wn[k] >= 0.0f) ? xt[k] : -xt[k];
	Y[(size_t)t * N + n] = sum;
}

} // anonymous

bool binary_gemm_abt_from_float(const float* X, const float* W,
                                int M, int N, int K, float* Y)
{
	if (M <= 0 || N <= 0 || K <= 0) return true;

	// Use tiled kernel for shapes where M, N >= 32; otherwise naive.
	if (M >= 32 && N >= 32)
	{
		const int BM = 32, BN = 32, BK = 32;
		const dim3 block((unsigned int)BN, (unsigned int)BM, 1u);
		const dim3 grid((unsigned int)((N + BN - 1) / BN),
		                (unsigned int)((M + BM - 1) / BM), 1u);
		binary_gemm_abt_from_float_tiled_kernel<32, 32, 32>
		    <<<grid, block, 0, computeStream()>>>(X, W, M, N, K, Y);
		GLADES_CUDA_CHECK(cudaGetLastError());
		return true;
	}

	const dim3 block(16u, 16u, 1u);
	const dim3 grid((unsigned int)(N + (int)block.x - 1) / (int)block.x,
	                (unsigned int)(M + (int)block.y - 1) / (int)block.y, 1u);
	binary_gemm_abt_from_float_kernel<<<grid, block, 0, computeStream()>>>(
	    X, W, M, N, K, Y);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// BF16-binarized fast path: re-encode sign(W) as BF16 ±1.0 buffer once,
// then run cuBLAS bf16 sgemm via existing tensor-core path. This costs
// one binarization pass + one bf16 GEMM. The BF16 GEMM uses Ada tensor
// cores at full FP16/BF16 throughput, and the binarization step is
// memory-bound (cheap relative to GEMM).
//
// Storage: caller provides W_bf16_scratch [N * K] uint16_t buffer for
// the binarized form. The scratch is overwritten with sign(W) cast to
// BF16. Subsequent calls with the same W can reuse cached scratch.
namespace {
__global__ void k_binarize_to_bf16_signs(const float* __restrict__ W,
                                         uint16_t* __restrict__ W_bf16,
                                         size_t n)
{
	const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	// +1.0 BF16 = 0x3F80; -1.0 BF16 = 0xBF80
	W_bf16[i] = (W[i] >= 0.0f) ? (uint16_t)0x3F80u : (uint16_t)0xBF80u;
}
} // anonymous

bool binarize_to_bf16_signs(const float* W, uint16_t* W_bf16, size_t n)
{
	if (n == 0) return true;
	const unsigned int TPB = 256u;
	const size_t blocks = (n + TPB - 1u) / TPB;
	if (blocks > 0x7FFFFFFFu) return false;
	k_binarize_to_bf16_signs<<<(unsigned int)blocks, TPB, 0, computeStream()>>>(
	    W, W_bf16, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ============================================================================
// (b) WMMA B1 tensor-core binary GEMM (Recommendation 2 deep path)
// ============================================================================
//
// SM 7.5+ exposes B1 (sub-byte 1-bit) tensor cores via the WMMA C++ API.
// Fragment shape: 8 × 8 × 128. Operation: XOR-popcount accumulation into INT32.
//
// Inputs (device pointers):
//   A_bits: M × K bits, row-major; K must be divisible by 128.
//   B_bits: N × K bits, row-major; K must be divisible by 128.
//
// Output:
//   C: M × N int32, with c[m,n] = popcount( ~(A[m] ^ B[n]) ) over K bits.
//   To convert XOR-popcount to ±1 GEMM:
//     binary_dot(a, b) = K - 2 * popcount(a ^ b)
//   So we can compute the standard ±1 dot product:  signed = K - 2 * c.
//
// This kernel is a CORRECTNESS demonstration — it shows the WMMA B1 path works
// on Ada (SM 8.9). It is NOT yet wired into the trainer because the trainer's
// activations are FP32/BF16, not 1-bit. Production B1 binary GEMM at training
// time requires also binarizing X (BitNet b1.0 style).

#include <mma.h>

namespace {
using namespace nvcuda;

// One block computes a 8×8 output tile via single-warp WMMA fragment.
__global__ void wmma_b1_gemm_kernel(const unsigned int* __restrict__ A_bits,
                                    const unsigned int* __restrict__ B_bits,
                                    int M, int N, int K_bits,
                                    int* __restrict__ C)
{
	const int blkM = blockIdx.y * 8;
	const int blkN = blockIdx.x * 8;
	if (blkM >= M || blkN >= N) return;

#if __CUDA_ARCH__ >= 750
	wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
	wmma::fill_fragment(c_frag, 0);

	const int Kuint = K_bits / 32;
	for (int kBase = 0; kBase < Kuint; kBase += 4)
	{
		const unsigned int* a_ptr = A_bits + (size_t)blkM * Kuint + kBase;
		const unsigned int* b_ptr = B_bits + (size_t)blkN * Kuint + kBase;
		wmma::load_matrix_sync(a_frag, a_ptr, Kuint * 32);
		wmma::load_matrix_sync(b_frag, b_ptr, Kuint * 32);
		wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag,
		                wmma::experimental::bmmaBitOpAND,
		                wmma::experimental::bmmaAccumulateOpPOPC);
	}

	// Store result (8×8 row-major)
	const int rowsLeft = M - blkM;
	const int colsLeft = N - blkN;
	if (rowsLeft >= 8 && colsLeft >= 8)
	{
		wmma::store_matrix_sync(C + (size_t)blkM * N + blkN, c_frag, N, wmma::mem_row_major);
	}
#else
	(void)A_bits; (void)B_bits; (void)M; (void)N; (void)K_bits; (void)C;
#endif
}
} // anonymous

// WMMA B1 binary GEMM: C[M,N] = popcount(A_bits[M,K] AND B_bits[N,K]) over K bits.
// K_bits must be divisible by 128.
bool wmma_b1_gemm(const unsigned int* A_bits, const unsigned int* B_bits,
                   int M, int N, int K_bits, int* C)
{
	if (M <= 0 || N <= 0 || K_bits <= 0) return true;
	if ((K_bits % 128) != 0) return false;
	const dim3 block(32u, 1u, 1u);  // single warp per block
	const dim3 grid((unsigned int)((N + 7) / 8), (unsigned int)((M + 7) / 8), 1u);
	wmma_b1_gemm_kernel<<<grid, block, 0, computeStream()>>>(A_bits, B_bits, M, N, K_bits, C);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ============================================================================
// (b) Full BitNet b1.0-style binary inference path
// ============================================================================
// Production binary inference (Wang et al. 2024 BitNet b1.0):
//
//   X_q[m,k]   = sign(X[m,k])           (1 bit)
//   alpha_x[m] = mean(|X[m,:]|)         (per-row scale, FP32)
//   W_q[n,k]   = sign(W[n,k])           (1 bit)
//   alpha_w[n] = mean(|W[n,:]|)         (per-row scale, FP32)
//
//   Y[m,n] = alpha_x[m] * alpha_w[n] * (K - 2 * popcount(X_q[m] ^ W_q[n]))
//
// At inference, sign(X) loses ~5-15% quality vs full-precision; the scales
// recover most of it. cuBLAS doesn't ship a B1×B1 sgemm so we use WMMA
// directly for the popcount stage.

namespace {

// Per-row scale: alpha[m] = mean(|X[m,:]|). Block per row; warp reduction.
__global__ void k_quant_x_to_b1_with_scale(const float* __restrict__ X,
                                           int M, int K,
                                           unsigned int* __restrict__ X_bits,
                                           float* __restrict__ alpha)
{
	const int m = blockIdx.x;
	if (m >= M) return;
	const int tid = threadIdx.x;
	const int Kuint = K / 32;
	const float* xm = X + (size_t)m * K;
	unsigned int* xbm = X_bits + (size_t)m * Kuint;

	// Pass 1: per-thread mean(|x|) accumulation
	float local_sum = 0.0f;
	for (int k = tid; k < K; k += 32)
		local_sum += fabsf(xm[k]);
	for (int off = 16; off > 0; off >>= 1)
		local_sum += ::__shfl_xor_sync(0xFFFFFFFF, local_sum, off);
	const float scale = local_sum / (float)K;
	if (tid == 0) alpha[m] = scale;

	// Pass 2: pack sign bits into uint32. Each thread handles uint32 chunks.
	for (int u = tid; u < Kuint; u += 32)
	{
		unsigned int bits = 0u;
		const int kBase = u * 32;
		#pragma unroll
		for (int b = 0; b < 32; ++b)
		{
			if (xm[kBase + b] >= 0.0f)
				bits |= (1u << b);
		}
		xbm[u] = bits;
	}
}

// BitNet GEMM: combines WMMA B1 popcount with per-row scale recovery.
// Uses XOR (not AND) for ±1 GEMM: dot(±1, ±1) = K - 2*popcount(a^b).
// The WMMA bmma op supports XOR by setting bmmaBitOpXOR.
__global__ void wmma_b1_gemm_xor_kernel(const unsigned int* __restrict__ A_bits,
                                        const unsigned int* __restrict__ B_bits,
                                        int M, int N, int K_bits,
                                        int* __restrict__ C_pop)
{
	const int blkM = blockIdx.y * 8;
	const int blkN = blockIdx.x * 8;
	if (blkM >= M || blkN >= N) return;
#if __CUDA_ARCH__ >= 750
	wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag;
	wmma::fill_fragment(c_frag, 0);
	const int Kuint = K_bits / 32;
	for (int kBase = 0; kBase < Kuint; kBase += 4)
	{
		const unsigned int* a_ptr = A_bits + (size_t)blkM * Kuint + kBase;
		const unsigned int* b_ptr = B_bits + (size_t)blkN * Kuint + kBase;
		wmma::load_matrix_sync(a_frag, a_ptr, Kuint * 32);
		wmma::load_matrix_sync(b_frag, b_ptr, Kuint * 32);
		wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag,
		                wmma::experimental::bmmaBitOpXOR,
		                wmma::experimental::bmmaAccumulateOpPOPC);
	}
	const int rowsLeft = M - blkM;
	const int colsLeft = N - blkN;
	if (rowsLeft >= 8 && colsLeft >= 8)
	{
		wmma::store_matrix_sync(C_pop + (size_t)blkM * N + blkN, c_frag, N, wmma::mem_row_major);
	}
#else
	(void)A_bits; (void)B_bits; (void)M; (void)N; (void)K_bits; (void)C_pop;
#endif
}

// Recover Y[m,n] from popcount: y = alpha_x[m] * alpha_w[n] * (K - 2 * popcount).
__global__ void k_bitnet_recover_scale(const int* __restrict__ C_pop,
                                       const float* __restrict__ alpha_x,
                                       const float* __restrict__ alpha_w,
                                       int M, int N, int K_bits,
                                       float* __restrict__ Y)
{
	const int m = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (m >= M || n >= N) return;
	const int pop = C_pop[(size_t)m * N + n];
	const int signed_dot = K_bits - 2 * pop;
	Y[(size_t)m * N + n] = alpha_x[m] * alpha_w[n] * (float)signed_dot;
}

} // anonymous

// Quantize X[M,K] to binary bits + per-row scale alpha[M].
// K must be divisible by 32 (uint32 packing).
bool quantize_x_to_b1_with_scale(const float* X, int M, int K,
                                 unsigned int* X_bits, float* alpha)
{
	if (M <= 0 || K <= 0) return true;
	if ((K % 32) != 0) return false;
	k_quant_x_to_b1_with_scale<<<(unsigned int)M, 32u, 0, computeStream()>>>(
	    X, M, K, X_bits, alpha);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Single-call BitNet QAT FFN forward: takes float X, float W; allocates
// internal binary + scale scratch (or reuses passed scratch); runs the full
// quantize-X + WMMA-XOR + scale-recover pipeline. Caller passes scratch
// buffers sized appropriately:
//   X_bits_scratch  [M * (K/32)]   uint32
//   W_bits_scratch  [N * (K/32)]   uint32
//   alpha_x_scratch [M]            float
//   alpha_w_scratch [N]            float
//   C_pop_scratch   [M * N]        int32
// K must be a multiple of 128 (WMMA B1 fragment shape).
bool bitnet_ffn_forward_gpu(const float* X, const float* W,
                            int M, int N, int K,
                            unsigned int* X_bits_scratch,
                            unsigned int* W_bits_scratch,
                            float* alpha_x_scratch,
                            float* alpha_w_scratch,
                            int* C_pop_scratch,
                            float* Y)
{
	if (M <= 0 || N <= 0 || K <= 0) return true;
	if ((K % 128) != 0) return false;
	if (!quantize_x_to_b1_with_scale(X, M, K, X_bits_scratch, alpha_x_scratch))
		return false;
	if (!quantize_x_to_b1_with_scale(W, N, K, W_bits_scratch, alpha_w_scratch))
		return false;
	return bitnet_b1_forward(X_bits_scratch, W_bits_scratch,
	                          alpha_x_scratch, alpha_w_scratch,
	                          C_pop_scratch, M, N, K, Y);
}

// Full BitNet inference forward: Y[M,N] = scale_recover(WMMA-B1-XOR(X_bits, W_bits)).
// X_bits [M, K/32], W_bits [N, K/32], alpha_x [M], alpha_w [N], K_bits = K.
// The popcount intermediate is allocated in C_pop_scratch [M*N int32].
bool bitnet_b1_forward(const unsigned int* X_bits,
                       const unsigned int* W_bits,
                       const float* alpha_x,
                       const float* alpha_w,
                       int* C_pop_scratch,
                       int M, int N, int K_bits,
                       float* Y)
{
	if (M <= 0 || N <= 0 || K_bits <= 0) return true;
	if ((K_bits % 128) != 0) return false;

	// Pass 1: WMMA B1 XOR-popcount → C_pop[M,N] int32
	{
		const dim3 block(32u, 1u, 1u);
		const dim3 grid((unsigned int)((N + 7) / 8), (unsigned int)((M + 7) / 8), 1u);
		wmma_b1_gemm_xor_kernel<<<grid, block, 0, computeStream()>>>(
		    X_bits, W_bits, M, N, K_bits, C_pop_scratch);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}

	// Pass 2: scale recovery
	{
		const dim3 block(16u, 16u, 1u);
		const dim3 grid((unsigned int)((N + (int)block.x - 1) / (int)block.x),
		                (unsigned int)((M + (int)block.y - 1) / (int)block.y), 1u);
		k_bitnet_recover_scale<<<grid, block, 0, computeStream()>>>(
		    C_pop_scratch, alpha_x, alpha_w, M, N, K_bits, Y);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}
	return true;
}

// #78 backward wrapper.
bool sw_attention_backward_gpu(const float* Q, int qStride,
                                const float* K, int kStride,
                                const float* V, int vStride,
                                const float* dO, int dOStride,
                                int T, int dHead, bool causal,
                                int sinkCount, int windowSize,
                                float* dQ, int dQStride,
                                float* dK_out, int dKStride,
                                float* dV_out, int dVStride)
{
	if (T <= 0 || dHead <= 0) return true;
	// Caller is expected to zero dQ/dK/dV (atomicAdd accumulation semantics).
	const float invSqrt = 1.0f / sqrtf((float)dHead);
	const dim3 grid((unsigned int)T, 1u, 1u);
	const dim3 block(32u, 1u, 1u);
	sw_attention_backward_kernel<<<grid, block, 0, computeStream()>>>(
	    Q, qStride, K, kStride, V, vStride, dO, dOStride, T, dHead,
	    causal ? 1 : 0, sinkCount, windowSize, invSqrt,
	    dQ, dQStride, dK_out, dKStride, dV_out, dVStride);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// ============================================================================
// (a) Full MLA forward (paradigm #76 trainer-dispatch primitive)
// ============================================================================
//
// Single-call helper that combines compute_latent + decompress_kv into one
// trainer-ready primitive. When wired into sgd_transformer.cpp's GPU
// attention path under `mlaLatentDim > 0`, this replaces the standard
// per-head K, V projection with the low-rank latent path.
//
//   Standard MHA:  K = h @ W_K   (T, dKV);   V = h @ W_V   (T, dKV)
//   #76 MLA:       c = h @ W_DKV (T, d_c);   K = c @ W_UK; V = c @ W_UV
//
// At training time the gradient chain rule is:
//   dW_UK = c^T @ dK;   dW_UV = c^T @ dV
//   dc    = dK @ W_UK^T + dV @ W_UV^T
//   dW_DKV = h^T @ dc;  dh += dc @ W_DKV^T
// All five gradients are standard cuBLAS sgemm calls (no custom kernel).
//
// Cache memory at inference: with KV cache storing c instead of K, V, the
// per-token cache size drops from 2*dKV to d_c (4× compression at d_c =
// dKV/2; up to 8× at d_c = dKV/4).

bool mla_attention_forward_gpu(const float* h,
                                const float* W_DKV,
                                const float* W_UK,
                                const float* W_UV,
                                int T, int dHidden, int dC, int dKVtotal,
                                float* c_scratch,
                                float* K_out,
                                float* V_out)
{
	if (T <= 0 || dHidden <= 0 || dC <= 0 || dKVtotal <= 0) return true;
	if (!mla_compute_latent_gpu(h, W_DKV, T, dHidden, dC, c_scratch))
		return false;
	if (!mla_decompress_kv_gpu(c_scratch, W_UK, W_UV, T, dC, dKVtotal,
	                            K_out, V_out))
		return false;
	return true;
}

// 2D transpose: out[N, M] = in[M, N]^T. One thread per element.
namespace {
__global__ void k_transpose_2d(const float* __restrict__ in,
                               int M, int N,
                               float* __restrict__ out)
{
	const int n = blockIdx.x * blockDim.x + threadIdx.x;
	const int m = blockIdx.y * blockDim.y + threadIdx.y;
	if (m >= M || n >= N) return;
	out[(size_t)n * M + m] = in[(size_t)m * N + n];
}
} // anonymous

bool transpose_2d(const float* in, int M, int N, float* out)
{
	if (M <= 0 || N <= 0) return true;
	const dim3 block(16u, 16u, 1u);
	const dim3 grid((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16), 1u);
	k_transpose_2d<<<grid, block, 0, computeStream()>>>(in, M, N, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Backward through MLA latent: given dK, dV, h, the factored weights and c
// (cached from forward), produce dh, dW_DKV, dW_UK, dW_UV.
// All five gradients via cuBLAS chain rule:
bool mla_attention_backward_gpu(const float* h,
                                 const float* c_cached,
                                 const float* dK,
                                 const float* dV,
                                 const float* W_DKV,
                                 const float* W_UK,
                                 const float* W_UV,
                                 int T, int dHidden, int dC, int dKVtotal,
                                 float* dh_accum,    // [T * dHidden] add to existing
                                 float* dW_DKV,      // [dHidden * dC]
                                 float* dW_UK,       // [dC * dKVtotal]
                                 float* dW_UV,       // [dC * dKVtotal]
                                 float* dc_scratch,  // [T * dC]
                                 unsigned short* bf16_scratch_A,
                                 unsigned short* bf16_scratch_B)
{
	if (T <= 0 || dHidden <= 0 || dC <= 0 || dKVtotal <= 0) return true;

	// PERMANENT FIX: route through the BF16 atb path that the standard W_K
	// backward uses (gpu_gemm_atb_mp's useBf16=true branch). The FP32
	// sgemm_rowmajor_atb path has a latent cuBLAS issue at certain shapes
	// (M=192 N=384 K=1024 reproducibly fails with STATUS_EXECUTION_FAILED).
	// The bf16 path uses cublasGemmEx with explicit bf16 input typing, which
	// avoids the failure. Output stays FP32 in C; precision impact is
	// negligible (bf16 inputs match the rest of the trainer in --mp mode).
	//
	// Caller must pass two bf16 scratch buffers (sized for the largest
	// operand: max(T*dHidden, T*dKVtotal, dHidden*dC, dC*dKVtotal)).
	{
		cudaError_t pre = cudaDeviceSynchronize();
		if (pre != cudaSuccess) {
			fprintf(stderr, "[mla_bwd] PRE-call error: %d (%s)\n", (int)pre, cudaGetErrorString(pre));
			return false;
		}
	}
	(void)cudaGetLastError();

#define MLA_BWD_CHECK(label) do { \
	cudaError_t e = cudaDeviceSynchronize(); \
	if (e != cudaSuccess) { \
		fprintf(stderr, "[mla_bwd] %s err=%d (%s) T=%d dH=%d dC=%d dKVtot=%d\n", \
		        label, (int)e, cudaGetErrorString(e), T, dHidden, dC, dKVtotal); \
		return false; \
	} \
} while (0)

	// dW_UK[dC, dKVtotal] = c^T[dC, T] @ dK[T, dKVtotal]
	if (!cast_f32_to_bf16(c_cached, bf16_scratch_A, (size_t)T * dC)) return false;
	MLA_BWD_CHECK("cast_c");
	if (!cast_f32_to_bf16(dK,       bf16_scratch_B, (size_t)T * dKVtotal)) return false;
	MLA_BWD_CHECK("cast_dK");
	if (!sgemm_rowmajor_atb_bf16(dC, dKVtotal, T, 1.0f,
	                              bf16_scratch_A, dC,
	                              bf16_scratch_B, dKVtotal,
	                              0.0f, dW_UK, dKVtotal)) return false;
	MLA_BWD_CHECK("gemm_dW_UK");

	// dW_UV[dC, dKVtotal] = c^T @ dV
	// (c is still in bf16_scratch_A from above)
	if (!cast_f32_to_bf16(dV,       bf16_scratch_B, (size_t)T * dKVtotal)) return false;
	if (!sgemm_rowmajor_atb_bf16(dC, dKVtotal, T, 1.0f,
	                              bf16_scratch_A, dC,
	                              bf16_scratch_B, dKVtotal,
	                              0.0f, dW_UV, dKVtotal))
		return false;

	// dc[T, dC] = dK[T, dKVtotal] @ W_UK^T[dKVtotal, dC] + dV @ W_UV^T
	if (!cast_f32_to_bf16(dK,   bf16_scratch_A, (size_t)T * dKVtotal)) return false;
	if (!cast_f32_to_bf16(W_UK, bf16_scratch_B, (size_t)dC * dKVtotal)) return false;
	if (!sgemm_rowmajor_abt_bf16(T, dC, dKVtotal, 1.0f,
	                              bf16_scratch_A, dKVtotal,
	                              bf16_scratch_B, dKVtotal,
	                              0.0f, dc_scratch, dC))
		return false;
	if (!cast_f32_to_bf16(dV,   bf16_scratch_A, (size_t)T * dKVtotal)) return false;
	if (!cast_f32_to_bf16(W_UV, bf16_scratch_B, (size_t)dC * dKVtotal)) return false;
	if (!sgemm_rowmajor_abt_bf16(T, dC, dKVtotal, 1.0f,
	                              bf16_scratch_A, dKVtotal,
	                              bf16_scratch_B, dKVtotal,
	                              1.0f, dc_scratch, dC))
		return false;

	// dW_DKV[dHidden, dC] = h^T[dHidden, T] @ dc[T, dC]
	if (!cast_f32_to_bf16(h,           bf16_scratch_A, (size_t)T * dHidden)) return false;
	if (!cast_f32_to_bf16(dc_scratch,  bf16_scratch_B, (size_t)T * dC)) return false;
	if (!sgemm_rowmajor_atb_bf16(dHidden, dC, T, 1.0f,
	                              bf16_scratch_A, dHidden,
	                              bf16_scratch_B, dC,
	                              0.0f, dW_DKV, dC))
		return false;

	// dh[T, dHidden] += dc[T, dC] @ W_DKV^T[dC, dHidden]
	if (!cast_f32_to_bf16(dc_scratch, bf16_scratch_A, (size_t)T * dC)) return false;
	if (!cast_f32_to_bf16(W_DKV,      bf16_scratch_B, (size_t)dHidden * dC)) return false;
	if (!sgemm_rowmajor_abt_bf16(T, dHidden, dC, 1.0f,
	                              bf16_scratch_A, dC,
	                              bf16_scratch_B, dC,
	                              1.0f, dh_accum, dHidden))
		return false;

	// DEBUG: surface any GPU error before returning
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "[mla_bwd] cudaDeviceSynchronize err=%d (%s) T=%d dH=%d dC=%d dKVtot=%d\n",
		        (int)err, cudaGetErrorString(err), T, dHidden, dC, dKVtotal);
		return false;
	}
	return true;
}

// #76 wrappers — both are matrix multiplies; reuse cuBLAS sgemm_rowmajor.
// c[T, d_c] = h[T, d_h] @ W_DKV[d_h, d_c]
bool mla_compute_latent_gpu(const float* h, const float* W_DKV,
                             int T, int d_h, int d_c, float* c_out)
{
	if (T <= 0 || d_h <= 0 || d_c <= 0) return true;
	return sgemm_rowmajor(T, d_c, d_h, 1.0f,
	                      h, d_h, W_DKV, d_c, 0.0f, c_out, d_c);
}

// K[T, dKVtotal] = c[T, d_c] @ W_UK[d_c, dKVtotal]
// V[T, dKVtotal] = c[T, d_c] @ W_UV[d_c, dKVtotal]
bool mla_decompress_kv_gpu(const float* c,
                            const float* W_UK, const float* W_UV,
                            int T, int d_c, int dKVtotal,
                            float* K_out, float* V_out)
{
	if (T <= 0 || d_c <= 0 || dKVtotal <= 0) return true;
	if (!sgemm_rowmajor(T, dKVtotal, d_c, 1.0f,
	                    c, d_c, W_UK, dKVtotal, 0.0f, K_out, dKVtotal))
		return false;
	if (!sgemm_rowmajor(T, dKVtotal, d_c, 1.0f,
	                    c, d_c, W_UV, dKVtotal, 0.0f, V_out, dKVtotal))
		return false;
	return true;
}

// ===========================================================================
//  ORION (paradigm shift #43) — Galerkin model-order reduction primitives.
// ===========================================================================
//
// Block-diagonal-V variant: every parameter tensor has its own basis V of
// shape [n × r] (BF16 to fit memory).  Each tensor maintains a tiny α (r
// floats), α_anchor, g_∥ (r), H_∥ (r×r), A_∥ (r) state; the anchor step
// builds (g_∥, H_∥, A_∥) via FD-HVP and the K-1 reduced steps iterate
// α on this local quadratic surrogate without touching the model.
//
// Kernels in this section:
//
//   orion_proj_left      : α += V^⊤ g                      [r += n×r ⨯ n]
//   orion_lift_add       : θ += V · α                      [n += n×r ⨯ r]
//   orion_perturb_col    : θ_pert = θ + ε · V[:, k]        [n = n + ε·col_k]
//   orion_oja_tilt       : V += η · g_⊥ · (V^⊤ g)^⊤        [n×r += outer-r]
//   orion_grammat_init   : init V columns to random normal then Gram-Schmidt
//   orion_gs_step        : one inner Gram-Schmidt step (subtract projection)
//   orion_col_norm_recip : compute 1/||V[:, k]|| (single scalar)
//   orion_col_scale      : scale V[:, k] by a host-supplied scalar
//
// All kernels work with V stored in column-major within [n × r] flat layout:
// V[i, k] = V_flat[k * n + i].  This makes the column-wise dot products and
// scales coalesce nicely.

namespace {

// α[k] += Σ_i V[i, k] * g[i]  for k ∈ [0, r).  One block per column.
__global__ void orion_proj_left_bf16_kernel(const __nv_bfloat16* __restrict__ V,
                                            const float*         __restrict__ g,
                                            int n, int r,
                                            float* __restrict__ alpha)
{
	extern __shared__ float smem[];
	int col = blockIdx.x;
	if (col >= r) return;
	const __nv_bfloat16* V_col = V + (size_t)col * n;
	float acc = 0.0f;
	for (int i = threadIdx.x; i < n; i += blockDim.x) {
		acc += __bfloat162float(V_col[i]) * g[i];
	}
	float total = blockReduceSum(acc, smem);
	if (threadIdx.x == 0) ::atomicAdd(&alpha[col], total);
}

// θ[i] += Σ_k V[i, k] * α[k]  for i ∈ [0, n).
__global__ void orion_lift_add_bf16_kernel(float*               __restrict__ theta,
                                           const __nv_bfloat16* __restrict__ V,
                                           const float*         __restrict__ alpha,
                                           int n, int r)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	float acc = 0.0f;
	for (int k = 0; k < r; ++k) {
		acc += __bfloat162float(V[(size_t)k * n + i]) * alpha[k];
	}
	theta[i] += acc;
}

// θ_pert[i] = θ[i] + eps · V[i, col]
__global__ void orion_perturb_col_bf16_kernel(float*               __restrict__ theta_pert,
                                              const float*         __restrict__ theta,
                                              const __nv_bfloat16* __restrict__ V,
                                              int n, int col, float eps)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const __nv_bfloat16* V_col = V + (size_t)col * n;
	theta_pert[i] = theta[i] + eps * __bfloat162float(V_col[i]);
}

// V[i, dst_col] -= alpha * V[i, src_col]   (Gram-Schmidt subtraction).
__global__ void orion_gs_subtract_bf16_kernel(__nv_bfloat16* __restrict__ V,
                                              int n, int src_col, int dst_col,
                                              float alpha)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	__nv_bfloat16* V_dst = V + (size_t)dst_col * n;
	const __nv_bfloat16* V_src = V + (size_t)src_col * n;
	float v = __bfloat162float(V_dst[i]) - alpha * __bfloat162float(V_src[i]);
	V_dst[i] = __float2bfloat16(v);
}

// Compute ||V[:, col]||² (per-column reduction → single scalar via atomicAdd).
__global__ void orion_col_normsq_bf16_kernel(const __nv_bfloat16* __restrict__ V,
                                             int n, int col,
                                             float* __restrict__ out_normsq)
{
	extern __shared__ float smem[];
	const __nv_bfloat16* V_col = V + (size_t)col * n;
	float acc = 0.0f;
	for (int i = threadIdx.x; i < n; i += blockDim.x) {
		float v = __bfloat162float(V_col[i]);
		acc += v * v;
	}
	float total = blockReduceSum(acc, smem);
	if (threadIdx.x == 0) ::atomicAdd(out_normsq, total);
}

// V[i, col] *= scale  (used to renormalize after Gram-Schmidt).
__global__ void orion_col_scale_bf16_kernel(__nv_bfloat16* __restrict__ V,
                                            int n, int col, float scale)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	__nv_bfloat16* V_col = V + (size_t)col * n;
	float v = __bfloat162float(V_col[i]) * scale;
	V_col[i] = __float2bfloat16(v);
}

// Dot product of two BF16 columns of V → scalar.
__global__ void orion_col_dot_bf16_kernel(const __nv_bfloat16* __restrict__ V,
                                          int n, int col_a, int col_b,
                                          float* __restrict__ out_dot)
{
	extern __shared__ float smem[];
	const __nv_bfloat16* A = V + (size_t)col_a * n;
	const __nv_bfloat16* B = V + (size_t)col_b * n;
	float acc = 0.0f;
	for (int i = threadIdx.x; i < n; i += blockDim.x) {
		acc += __bfloat162float(A[i]) * __bfloat162float(B[i]);
	}
	float total = blockReduceSum(acc, smem);
	if (threadIdx.x == 0) ::atomicAdd(out_dot, total);
}

// Oja tilt: V[:, k] += eta * (g[i] - V[i,:]·(V^⊤g)) · (V^⊤g)[k]
// Operating column-major.  g_proj is the host-uploaded r-dim vector V^⊤g.
__global__ void orion_oja_tilt_bf16_kernel(__nv_bfloat16*       __restrict__ V,
                                           const float*         __restrict__ g,
                                           const float*         __restrict__ g_proj,
                                           int n, int r, float eta)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	// Residual ⊥ V: g_⊥[i] = g[i] - Σ_k V[i,k] · g_proj[k]
	float Vg = 0.0f;
	for (int k = 0; k < r; ++k) {
		Vg += __bfloat162float(V[(size_t)k * n + i]) * g_proj[k];
	}
	float gperp_i = g[i] - Vg;
	// For each column k: V[i, k] += eta * g_⊥[i] * g_proj[k]
	for (int k = 0; k < r; ++k) {
		float v = __bfloat162float(V[(size_t)k * n + i]);
		v += eta * gperp_i * g_proj[k];
		V[(size_t)k * n + i] = __float2bfloat16(v);
	}
}

} // anonymous namespace

bool orion_proj_left(const uint16_t* V, const float* g,
                     int n, int r, float* alpha_out)
{
	if (n <= 0 || r <= 0) return true;
	GLADES_CUDA_CHECK(cudaMemsetAsync(alpha_out, 0, r * sizeof(float), computeStream()));
	int block = rowBlockSize(n);
	int smemBytes = (block / 32 + 2) * sizeof(float);
	orion_proj_left_bf16_kernel<<<r, block, smemBytes, computeStream()>>>(
	    reinterpret_cast<const __nv_bfloat16*>(V), g, n, r, alpha_out);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool orion_lift_add(float* theta, const uint16_t* V,
                    const float* alpha, int n, int r)
{
	if (n <= 0 || r <= 0) return true;
	int block = kBlockElem;
	int grid  = (n + block - 1) / block;
	orion_lift_add_bf16_kernel<<<grid, block, 0, computeStream()>>>(
	    theta, reinterpret_cast<const __nv_bfloat16*>(V), alpha, n, r);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool orion_perturb_col(float* theta_pert, const float* theta,
                       const uint16_t* V, int n, int col, float eps)
{
	if (n <= 0) return true;
	int block = kBlockElem;
	int grid  = (n + block - 1) / block;
	orion_perturb_col_bf16_kernel<<<grid, block, 0, computeStream()>>>(
	    theta_pert, theta, reinterpret_cast<const __nv_bfloat16*>(V), n, col, eps);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

bool orion_oja_tilt(uint16_t* V, const float* g, const float* g_proj,
                    int n, int r, float eta)
{
	if (n <= 0 || r <= 0) return true;
	int block = kBlockElem;
	int grid  = (n + block - 1) / block;
	orion_oja_tilt_bf16_kernel<<<grid, block, 0, computeStream()>>>(
	    reinterpret_cast<__nv_bfloat16*>(V), g, g_proj, n, r, eta);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// Modified Gram-Schmidt re-orthogonalization of V columns IN PLACE.
// At r ≤ 8 the host-side loop over column pairs is trivially cheap; only
// the per-pair dot/subtract/norm kernels touch n elements.  Caller supplies
// two FP32 scratch scalars on device (scratch_dot, scratch_normsq).
bool orion_gram_schmidt(uint16_t* V, int n, int r,
                        float* scratch_dot, float* scratch_normsq)
{
	__nv_bfloat16* V_bf16 = reinterpret_cast<__nv_bfloat16*>(V);
	if (n <= 0 || r <= 0) return true;
	int block = rowBlockSize(n);
	int smemBytes = (block / 32 + 2) * sizeof(float);
	int grid_elem = (n + kBlockElem - 1) / kBlockElem;

	for (int k = 0; k < r; ++k)
	{
		// Subtract projections onto previous columns.
		for (int j = 0; j < k; ++j)
		{
			GLADES_CUDA_CHECK(cudaMemsetAsync(scratch_dot, 0, sizeof(float),
			                                   computeStream()));
			orion_col_dot_bf16_kernel<<<1, block, smemBytes, computeStream()>>>(
			    V_bf16, n, k, j, scratch_dot);
			float alpha_h = 0.0f;
			GLADES_CUDA_CHECK(cudaMemcpyAsync(&alpha_h, scratch_dot,
			                                   sizeof(float),
			                                   cudaMemcpyDeviceToHost,
			                                   computeStream()));
			GLADES_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
			orion_gs_subtract_bf16_kernel<<<grid_elem, kBlockElem, 0,
			                                  computeStream()>>>(
			    V_bf16, n, j, k, alpha_h);
			GLADES_CUDA_CHECK(cudaGetLastError());
		}
		// Renormalize column k.
		GLADES_CUDA_CHECK(cudaMemsetAsync(scratch_normsq, 0, sizeof(float),
		                                   computeStream()));
		orion_col_normsq_bf16_kernel<<<1, block, smemBytes, computeStream()>>>(
		    V_bf16, n, k, scratch_normsq);
		float normsq_h = 0.0f;
		GLADES_CUDA_CHECK(cudaMemcpyAsync(&normsq_h, scratch_normsq,
		                                   sizeof(float),
		                                   cudaMemcpyDeviceToHost,
		                                   computeStream()));
		GLADES_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
		float scale = (normsq_h > 1e-12f) ? (1.0f / sqrtf(normsq_h)) : 0.0f;
		orion_col_scale_bf16_kernel<<<grid_elem, kBlockElem, 0,
		                                computeStream()>>>(V_bf16, n, k, scale);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}
	return true;
}

// ===========================================================================
//  SCFA (paradigm shift #42) — Spectral Compressed Flow Attention primitives.
// ===========================================================================
//
// SCFA replaces full T-token attention with attention in a k-dim sequence-
// spectral basis B ∈ ℝ^{T×k} (k ≪ T) + a depthwise causal conv D covering
// the out-of-spectrum residual.  Forward:
//
//   q_compr = B^T q                       (T → k compression)
//   y_compr = SoftmaxAttn(q_compr ...)    (k-dim attention; existing kernel)
//   y_∥     = B · y_compr                 (k → T lift)
//   y_⊥     = D(q - B B^T q)              (depthwise causal conv on residual)
//   y       = y_∥ + y_⊥
//
// This file ships the two SCFA-specific primitives:
//   scfa_dct_basis_init    — fill B with normalized DCT-II basis (orthonormal)
//   scfa_depthwise_causal_conv_fwd — y_⊥ = D(x) with causal kernel size 2w+1,
//     one filter per channel; m channels, T positions, w half-width.
//
// The compression/lift steps reuse sgemm_rowmajor (in this same file).
// Theorem 3 reversibility integration with CHIRON shears is handled at the
// chiron_main.cpp level; these kernels are paradigm-agnostic linear algebra.

namespace {

// Apply 1-D depthwise causal conv: y[t, c] = Σ_{i=-w..0} K[c, w+i] · x[t+i, c]
// for t ∈ [0, T), c ∈ [0, m).  Out-of-bounds left taps zero-padded.
__global__ void scfa_depthwise_causal_conv_fwd_kernel(
    const float* __restrict__ x,    // [T, m]
    const float* __restrict__ K,    // [m, w+1]  (only causal half + center)
    int T, int m, int w,
    float* __restrict__ y)          // [T, m]
{
	int t = blockIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= T || c >= m) return;

	float acc = 0.0f;
	for (int i = 0; i <= w; ++i)
	{
		int src = t - i;
		if (src < 0) break;
		acc += K[(size_t)c * (size_t)(w + 1) + (size_t)i] *
		       x[(size_t)src * (size_t)m + (size_t)c];
	}
	y[(size_t)t * (size_t)m + (size_t)c] = acc;
}

} // anonymous namespace

bool scfa_depthwise_causal_conv_fwd(const float* x, const float* K,
                                     int T, int m, int w, float* y,
                                     cudaStream_t stream)
{
	if (T <= 0 || m <= 0 || w < 0) return true;
	int block = 256;
	dim3 grid((m + block - 1) / block, T);
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	scfa_depthwise_causal_conv_fwd_kernel<<<grid, block, 0, s>>>(
	    x, K, T, m, w, y);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

namespace {

// Backward through depthwise causal conv.
// Forward: y[t, c] = Σ_{i=0..w} K[c, i] · x[t-i, c]
// Gradients:
//   dx[t, c] += Σ_{i=0..w, t+i<T} K[c, i] · dy[t+i, c]
//   dK[c, i] += Σ_{t=i..T-1}    x[t-i, c] · dy[t, c]
// Caller must zero dx and dK before launch (kernel uses += accumulators).

__global__ void scfa_dwconv_dx_kernel(const float* __restrict__ dy,
                                      const float* __restrict__ K,
                                      int T, int m, int w,
                                      float* __restrict__ dx)
{
	int t = blockIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= T || c >= m) return;
	float acc = 0.0f;
	for (int i = 0; i <= w; ++i)
	{
		int src = t + i;
		if (src >= T) break;
		acc += K[(size_t)c * (size_t)(w + 1) + (size_t)i] *
		       dy[(size_t)src * (size_t)m + (size_t)c];
	}
	dx[(size_t)t * (size_t)m + (size_t)c] += acc;
}

__global__ void scfa_dwconv_dK_kernel(const float* __restrict__ x,
                                      const float* __restrict__ dy,
                                      int T, int m, int w,
                                      float* __restrict__ dK)
{
	// One thread per (c, i) pair; loops t.
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y;
	if (c >= m || i > w) return;
	float acc = 0.0f;
	for (int t = i; t < T; ++t)
	{
		acc += x[(size_t)(t - i) * (size_t)m + (size_t)c] *
		       dy[(size_t)t * (size_t)m + (size_t)c];
	}
	::atomicAdd(&dK[(size_t)c * (size_t)(w + 1) + (size_t)i], acc);
}

// 2D-tiled parallel variant: BLOCK_M consecutive channels (coalesced memory)
// and BLOCK_T parallel T-partitions per block, reduced via shared memory.
// Replaces the legacy "1 thread per (c,i), loop t" pattern which was both
// non-coalesced (threads in a block shared c, strided rows) and severely
// under-parallel at production scale.
template<int BLOCK_M, int BLOCK_T>
__global__ void scfa_dwconv_dK_kernel_par(const float* __restrict__ x,
                                          const float* __restrict__ dy,
                                          int T, int m, int w,
                                          float* __restrict__ dK)
{
	const int i = blockIdx.y;                              // tap in [0, w]
	const int c = blockIdx.x * BLOCK_M + threadIdx.x;      // channel (coalesced)
	const int ty = threadIdx.y;                            // T-partition
	if (i > w) return;

	float acc = 0.0f;
	// Strided sum over T: each ty handles t = i+ty, i+ty+BLOCK_T, ...
	if (c < m)
	{
		for (int t = i + ty; t < T; t += BLOCK_T)
		{
			acc += x[(size_t)(t - i) * (size_t)m + (size_t)c] *
			       dy[(size_t)t * (size_t)m + (size_t)c];
		}
	}

	// Reduction across the ty dimension via shared memory.
	__shared__ float sdata[BLOCK_T][BLOCK_M];
	sdata[ty][threadIdx.x] = acc;
	__syncthreads();

	for (int s = BLOCK_T / 2; s > 0; s >>= 1)
	{
		if (ty < s)
		{
			sdata[ty][threadIdx.x] += sdata[ty + s][threadIdx.x];
		}
		__syncthreads();
	}

	if (ty == 0 && c < m)
	{
		// Exactly one block-row writes to each (c, i); += for accumulate semantics.
		dK[(size_t)c * (size_t)(w + 1) + (size_t)i] += sdata[0][threadIdx.x];
	}
}

} // anonymous namespace

bool scfa_depthwise_causal_conv_bwd(const float* x, const float* K,
                                     const float* dy,
                                     int T, int m, int w,
                                     float* dx, float* dK,
                                     cudaStream_t stream)
{
	if (T <= 0 || m <= 0 || w < 0) return true;
	cudaStream_t s = (stream != 0) ? stream : computeStream();
	// dx kernel: zero-initialize is the caller's responsibility (kernel +=).
	{
		int block = 256;
		dim3 grid((m + block - 1) / block, T);
		scfa_dwconv_dx_kernel<<<grid, block, 0, s>>>(
		    dy, K, T, m, w, dx);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}
	// dK kernel: 2D-tiled parallel reduction.  BLOCK_M consecutive channels per
	// block (coalesced memory) × BLOCK_T parallel T-partitions, reduced via SMEM.
	// Replaces the legacy "1 thread per (c,i), loop t" kernel which was both
	// non-coalesced and severely under-parallel at production scale.
	{
		const int BLOCK_M = 64;
		const int BLOCK_T = 8;
		int wp1 = w + 1;
		dim3 grid((m + BLOCK_M - 1) / BLOCK_M, wp1);
		dim3 block(BLOCK_M, BLOCK_T);
		scfa_dwconv_dK_kernel_par<64, 8><<<grid, block, 0, s>>>(
		    x, dy, T, m, w, dK);
		GLADES_CUDA_CHECK(cudaGetLastError());
	}
	return true;
}

namespace {

// Fill B[T × k] (row-major) with orthonormal DCT-II basis on device.
__global__ void scfa_dct_basis_init_kernel(float* B_flat, int T, int k)
{
	int t = blockIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= T || j >= k) return;

	const float invT = 1.0f / (float)T;
	float alpha = (j == 0) ? sqrtf(invT) : sqrtf(2.0f * invT);
	const float pi_f = 3.14159265358979323846f;
	float arg = ((2.0f * (float)t + 1.0f) * (float)j * pi_f) /
	            (2.0f * (float)T);
	B_flat[(size_t)t * (size_t)k + (size_t)j] = alpha * cosf(arg);
}

} // anonymous namespace

// Fill B[T × k] (row-major, B[t, j] = B_flat[t*k + j]) with orthonormal DCT-II
// basis truncated to k components: B[t, j] = α_j · cos((2t+1)·j·π / (2T)),
// α_0 = 1/√T, α_{j>0} = √(2/T).  Natural sequence-spectral basis for language
// data (low-frequency components carry most signal).  Computed on device.
bool scfa_dct_basis_init(float* B_flat, int T, int k)
{
	if (T <= 0 || k <= 0) return true;
	int block = 256;
	dim3 grid((k + block - 1) / block, T);
	scfa_dct_basis_init_kernel<<<grid, block, 0, computeStream()>>>(
	    B_flat, T, k);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
