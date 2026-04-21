// Custom CUDA kernel implementations for Glades ML.
//
// Contains normalization, activation, attention, embedding, optimizer, and
// utility kernels together with thin host-side wrapper functions that handle
// grid/block configuration.
//
// Requirements: CUDA 11+, SM 6.0+.

#include "gpu_kernels.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
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

// dgamma/dbeta kernel: one block per column, threads reduce across rows.
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

	// Kernel 2: dgamma/dbeta (one block per column, reduce across rows).
	int block2 = rowBlockSize(rows);
	int smemBytes2 = (block2 / 32 + 2) * 2 * sizeof(float);
	layernorm_backward_dgamma_dbeta<<<cols, block2, smemBytes2, computeStream()>>>(
		dout, x, mean, invStd, rows, cols, dgamma, dbeta);
	GLADES_CUDA_CHECK(cudaGetLastError());

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

} // anonymous namespace

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
	const float v_new = beta2 * v_old + (1.0f - beta2) * g * g;

	// Store back as BF16 (round-to-nearest-even).
	m_bf16[idx] = bf16_store_from_f32(m_new);
	v_bf16[idx] = bf16_store_from_f32(v_new);

	// Bias correction (matches FP32 adam_update_kernel).
	const float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
	const float m_hat = m_new / bc1;
	const float v_hat = v_new / bc2;

	param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
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
	    param, grad, m_bf16, v_bf16,
	    lr, beta1, beta2, eps, weightDecay, gradScale, step, n);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
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

} // anonymous namespace

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

} // anonymous namespace

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
	const int block = 256;
	const int smemBytes = 3 * block * static_cast<int>(sizeof(float));
	collect_token_lm_metrics_kernel<<<1, block, smemBytes, computeStream()>>>(
	    probs, targets, T, vocabSize, padToken, out);
	GLADES_CUDA_CHECK(cudaGetLastError());
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

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
