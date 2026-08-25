// CUDA primitives for ORBIT, CHIRON's increment-space factored optimizer.
#include "gpu_orbit.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace glades {
namespace gpu {
namespace {

static const int ORBIT_BLOCK = 256;

__device__ __forceinline__ float orbit_bf16_to_float(uint16_t x)
{
	union { unsigned int u; float f; } v;
	v.u = (unsigned int)x << 16;
	return v.f;
}

__device__ __forceinline__ uint32_t orbit_sr_hash32(uint32_t a, uint32_t b, uint32_t c)
{
	uint32_t x = a ^ (b * 0x9E3779B1u) ^ (c * 0x85EBCA6Bu);
	x ^= x >> 16; x *= 0x7FEB352Du;
	x ^= x >> 15; x *= 0x846CA68Bu;
	x ^= x >> 16;
	return x;
}

__device__ __forceinline__ uint16_t orbit_float_to_bf16_sr(float x, int idx,
                                                            uint32_t seed, uint32_t step)
{
	union { float f; uint32_t u; } v;
	v.f = x;
	if (isnan(x))
	{
		const uint32_t sign = v.u & 0x80000000u;
		return (uint16_t)(((sign | 0x7FC00000u) >> 16) & 0xFFFFu);
	}
	const uint32_t low = v.u & 0xFFFFu;
	uint32_t high = v.u >> 16;
	if ((orbit_sr_hash32((uint32_t)idx, step, seed) & 0xFFFFu) < low) ++high;
	return (uint16_t)(high & 0xFFFFu);
}

__device__ __forceinline__ float warp_sum(float x)
{
	for (int off = 16; off > 0; off >>= 1)
		x += __shfl_down_sync(0xFFFFFFFFu, x, off);
	return x;
}

__device__ __forceinline__ float warp_min(float x)
{
	for (int off = 16; off > 0; off >>= 1)
		x = fminf(x, __shfl_down_sync(0xFFFFFFFFu, x, off));
	return x;
}

__device__ __forceinline__ float warp_max(float x)
{
	for (int off = 16; off > 0; off >>= 1)
		x = fmaxf(x, __shfl_down_sync(0xFFFFFFFFu, x, off));
	return x;
}

__device__ __forceinline__ void atomic_min_positive(float* p, float x)
{
	if (x >= 0.0f && isfinite(x)) atomicMin((unsigned int*)p, __float_as_uint(x));
}
__device__ __forceinline__ void atomic_max_positive(float* p, float x)
{
	if (x >= 0.0f && isfinite(x)) atomicMax((unsigned int*)p, __float_as_uint(x));
}

template <typename T>
__device__ __forceinline__ float orbit_load(const T* p, size_t i);
template <>
__device__ __forceinline__ float orbit_load<float>(const float* p, size_t i) { return p[i]; }
template <>
__device__ __forceinline__ float orbit_load<uint16_t>(const uint16_t* p, size_t i) { return orbit_bf16_to_float(p[i]); }

// One-read tile: 8 rows x 256 columns. The 32 KiB shared tile supports both
// row and column reductions without global atomics or a second matrix read.
template <typename T>
__global__ void orbit_row_col_tile_kernel(const T* __restrict__ g,
                                           int rows, int cols, float gradScale,
                                           float* __restrict__ rowPartial,
                                           float* __restrict__ colPartial)
{
	__shared__ float tile[8][256];
	const int tid = threadIdx.x;
	const int row0 = blockIdx.y * 8;
	const int col = blockIdx.x * 256 + tid;
#pragma unroll
	for (int r = 0; r < 8; ++r)
	{
		float q = 0.0f;
		if (row0 + r < rows && col < cols)
		{
			const float v = orbit_load<T>(g, (size_t)(row0 + r) * cols + col) * gradScale;
			q = v * v;
		}
		tile[r][tid] = q;
	}
	__syncthreads();

	if (col < cols)
	{
		float s = 0.0f;
#pragma unroll
		for (int r = 0; r < 8; ++r) s += tile[r][tid];
		const int rowTiles = (rows + 7) / 8;
		colPartial[(size_t)col * rowTiles + blockIdx.y] = s;
	}
	const int warp = tid >> 5;
	const int lane = tid & 31;
	if (warp < 8 && row0 + warp < rows)
	{
		float s = 0.0f;
		for (int c = lane; c < 256; c += 32) s += tile[warp][c];
		s = warp_sum(s);
		if (lane == 0)
		{
			const int colTiles = (cols + 255) / 256;
			rowPartial[(size_t)(row0 + warp) * colTiles + blockIdx.x] = s;
		}
	}
}

__global__ void orbit_finalize_rows_kernel(const float* __restrict__ partial,
                                            int rows, int cols, int tiles,
                                            float betaF, float* __restrict__ ema,
                                            float* __restrict__ emaSum,
                                            float* __restrict__ scaleSample)
{
	const int row = blockIdx.x;
	if (row >= rows) return;
	float s = 0.0f;
	for (int k = threadIdx.x; k < tiles; k += blockDim.x)
		s += partial[(size_t)row * tiles + k];
	s = warp_sum(s);
	__shared__ float w[8];
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	if (lane == 0) w[warp] = s;
	__syncthreads();
	if (warp == 0)
	{
		s = lane < 8 ? w[lane] : 0.0f;
		s = warp_sum(s);
		if (lane == 0)
		{
			const float sample = s / (float)cols;
			const float next = betaF * ema[row] + (1.0f - betaF) * sample;
			ema[row] = next;
			atomicAdd(emaSum, next);
			atomicAdd(scaleSample, sample);
		}
	}
}

__global__ void orbit_finalize_cols_kernel(const float* __restrict__ partial,
                                            int rows, int cols, int tiles,
                                            float betaF, float* __restrict__ ema,
                                            float* __restrict__ emaSum)
{
	const int col = blockIdx.x;
	if (col >= cols) return;
	float s = 0.0f;
	for (int k = threadIdx.x; k < tiles; k += blockDim.x)
		s += partial[(size_t)col * tiles + k];
	s = warp_sum(s);
	__shared__ float w[8];
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	if (lane == 0) w[warp] = s;
	__syncthreads();
	if (warp == 0)
	{
		s = lane < 8 ? w[lane] : 0.0f;
		s = warp_sum(s);
		if (lane == 0)
		{
			const float sample = s / (float)rows;
			const float next = betaF * ema[col] + (1.0f - betaF) * sample;
			ema[col] = next;
			atomicAdd(emaSum, next);
		}
	}
}

__global__ void orbit_scale_from_sample_kernel(float* scaleEma,
                                                const float* sampleSum,
                                                float divisor, float betaF)
{
	if (threadIdx.x == 0)
	{
		const float sample = *sampleSum / divisor;
		*scaleEma = betaF * (*scaleEma) + (1.0f - betaF) * sample;
	}
}

// Coalesced embedding columns: each thread owns a column and loops 64 rows.
__global__ void orbit_embedding_col_tiles_kernel(const float* __restrict__ g,
                                                  int rows, int cols,
                                                  float gradScale,
                                                  float* __restrict__ colSample)
{
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= cols) return;
	const int row0 = blockIdx.y * 64;
	const int row1 = min(rows, row0 + 64);
	float s = 0.0f;
	for (int r = row0; r < row1; ++r)
	{
		const float v = g[(size_t)r * cols + col] * gradScale;
		s += v * v;
	}
	atomicAdd(colSample + col, s);
}

__global__ void orbit_finalize_embedding_cols_kernel(const float* __restrict__ sample,
                                                       int rows, int cols, float betaF,
                                                       float* __restrict__ ema,
                                                       float* __restrict__ emaSum,
                                                       float* __restrict__ scaleSample)
{
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= cols) return;
	const float x = sample[col] / (float)rows;
	const float next = betaF * ema[col] + (1.0f - betaF) * x;
	ema[col] = next;
	atomicAdd(emaSum, next);
	atomicAdd(scaleSample, x);
}

template <typename T>
__global__ void orbit_scale_reduce_kernel(const T* __restrict__ g, int n,
                                           float gradScale, float* out)
{
	float s = 0.0f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
	     i < n; i += blockDim.x * gridDim.x)
	{
		const float x = orbit_load<T>(g, i) * gradScale;
		s += x * x;
	}
	s = warp_sum(s);
	if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

__global__ void orbit_adjoint_kernel(const float* __restrict__ x,
                                      int rows, int cols, float weight,
                                      float* __restrict__ sample)
{
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= cols) return;
	float s = 0.0f;
	for (int r = 0; r < rows; ++r)
	{
		const float v = x[(size_t)r * cols + col];
		s += v * v;
	}
	sample[col] += weight * s;
}

__global__ void orbit_h_strided_kernel(const float* __restrict__ h,
                                        int rows, int cols, int stride,
                                        float* __restrict__ sample)
{
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= cols) return;
	float s = 0.0f; int count = 0;
	for (int r = 0; r < rows; r += stride)
	{
		const float v = h[(size_t)r * cols + col];
		s += v * v; ++count;
	}
	sample[col] = s / (float)max(1, count);
}

template <typename T>
__global__ void orbit_occupancy_kernel(const T* __restrict__ probs,
                                        int rows, int cols,
                                        float* __restrict__ sample)
{
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= cols) return;
	float s = 0.0f;
	for (int r = 0; r < rows; ++r)
	{
		const float p = orbit_load<T>(probs, (size_t)r * cols + col);
		s += p * (1.0f - p);
	}
	sample[col] = s / (float)rows;
}

__global__ void orbit_frequency_kernel(const int* ids, int rows, int vocab,
                                        float* frequency)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
	     i += blockDim.x * gridDim.x)
	{
		const int id = ids[i];
		if (id >= 0 && id < vocab) atomicAdd(frequency + id, 1.0f);
	}
}

__global__ void orbit_dq_mean_sq_kernel(const float* dq, int n, float* out)
{
	float s = 0.0f;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += blockDim.x * gridDim.x)
	{
		const float v = dq[i]; s += v * v;
	}
	s = warp_sum(s);
	if ((threadIdx.x & 31) == 0) atomicAdd(out, s / (float)n);
}

__global__ void orbit_vector_ema_kernel(const float* sample, float* ema,
                                         int n, float betaF, float* sum)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += blockDim.x * gridDim.x)
	{
		const float x = betaF * ema[i] + (1.0f - betaF) * sample[i];
		ema[i] = x;
		atomicAdd(sum, x);
	}
}

__global__ void orbit_scalar_ema_kernel(const float* sample, float* ema, float betaF)
{
	if (blockIdx.x == 0 && threadIdx.x == 0)
		*ema = betaF * (*ema) + (1.0f - betaF) * (*sample);
}

__global__ void orbit_embedding_rows_kernel(const float* occupancy,
                                             const float* frequency,
                                             const float* dqMeanSq,
                                             int vocab, float kappa, float bc,
                                             float* factor, float* sum)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab;
	     i += blockDim.x * gridDim.x)
	{
		const float x = occupancy[i] / bc + kappa * frequency[i] * (*dqMeanSq / bc);
		factor[i] = fmaxf(x, 1e-30f);
		atomicAdd(sum, fmaxf(x, 1e-30f));
	}
}

__global__ void orbit_quantile_kernel(const float* values, int n, float q, float* out)
{
	__shared__ float a[1024];
	const int tid = threadIdx.x;
	const int samples = min(n, 1024);
	if (tid < samples)
	{
		const int idx = (int)(((long long)tid * (long long)n) / samples);
		a[tid] = values[min(n - 1, idx)];
	}
	else a[tid] = FLT_MAX;
	__syncthreads();
	// Bitonic sorting network, deterministic for finite non-negative factors.
	for (int k = 2; k <= 1024; k <<= 1)
		for (int j = k >> 1; j > 0; j >>= 1)
		{
			const int ixj = tid ^ j;
			if (ixj > tid)
			{
				const bool up = (tid & k) == 0;
				const float x = a[tid], y = a[ixj];
				if ((up && x > y) || (!up && x < y)) { a[tid] = y; a[ixj] = x; }
			}
			__syncthreads();
		}
	if (tid == 0)
	{
		int k = (int)floorf(q * (float)(samples - 1));
		if (k < 0) k = 0; if (k >= samples) k = samples - 1;
		*out = a[k];
	}
}

__global__ void orbit_floor_sum_kernel(const float* values, int n,
                                        const float* floorValue,
                                        float* prepared, float* sum)
{
	const float f = *floorValue;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	     i += blockDim.x * gridDim.x)
	{
		const float x = fmaxf(values[i], f);
		prepared[i] = x;
		atomicAdd(sum, x);
	}
}

__global__ void orbit_diag_reset_kernel(float* d)
{
	if (threadIdx.x < ORBIT_DIAG_SIZE) d[threadIdx.x] = 0.0f;
	if (threadIdx.x == 0) d[ORBIT_DIAG_FACTOR_MIN] = FLT_MAX;
}

template <typename ParamT, typename GradT>
__global__ void orbit_factored_step_kernel(ParamT* __restrict__ param,
                                            const GradT* __restrict__ grad,
                                            int8_t* __restrict__ momentum,
                                            float* __restrict__ momentumScale,
                                            const float* __restrict__ rowFactor,
                                            const float* __restrict__ rowSum,
                                            const float* __restrict__ colFactor,
                                            const float* __restrict__ colSum,
                                            const float* __restrict__ scaleEma,
                                            int rows, int cols,
                                            float lr, float beta1, float betaF,
                                            float eps, float wd, bool metricWd,
                                            float gradScale, int step,
                                            int freezeSteps, float deltaF,
                                            const float* __restrict__ delayedLengthSq,
                                            float* __restrict__ currentLengthSq,
                                            uint32_t srSeed, uint32_t srStep,
                                            float* __restrict__ diagnostics)
{
	const int blockId = blockIdx.x;
	const int start = blockId * ORBIT_BLOCK;
	const int idx = start + threadIdx.x;
	const int n = rows * cols;
	const int len = min(ORBIT_BLOCK, n - start);
	if (len <= 0) return;
	__shared__ float sm[ORBIT_BLOCK];
	const float oldScale = momentumScale[blockId];
	float mNew = 0.0f;
	if (threadIdx.x < len)
	{
		const float g = orbit_load<GradT>(grad, idx) * gradScale;
		const float mOld = (float)momentum[idx] * oldScale * (1.0f / 127.0f);
		mNew = beta1 * mOld + (1.0f - beta1) * g;
		sm[threadIdx.x] = mNew;
	}
	else sm[threadIdx.x] = 0.0f;
	__syncthreads();

	float mx = threadIdx.x < len ? fabsf(mNew) : 0.0f;
	mx = warp_max(mx);
	__shared__ float warpMax[8];
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	if (lane == 0) warpMax[warp] = mx;
	__syncthreads();
	if (warp == 0)
	{
		mx = lane < 8 ? warpMax[lane] : 0.0f;
		mx = warp_max(mx);
		if (lane == 0) warpMax[0] = fmaxf(mx, 1e-20f);
	}
	__syncthreads();
	const float newScale = warpMax[0];
	if (threadIdx.x == 0) momentumScale[blockId] = newScale;

	float metric = 0.0f;
	float dvals[8] = {0,0,0,0,0,0,0,0};
	float factorShape = 1.0f;
	if (threadIdx.x < len)
	{
		float mq = rintf(sm[threadIdx.x] * (127.0f / newScale));
		mq = fmaxf(-127.0f, fminf(127.0f, mq));
		momentum[idx] = (int8_t)mq;
		const float bc1 = fmaxf(1e-12f, 1.0f - powf(beta1, (float)step));
		const float bcf = fmaxf(1e-12f, 1.0f - powf(betaF, (float)step));
		const int r = idx / cols, c = idx - (idx / cols) * cols;
		if (step > freezeSteps)
		{
			const float rf = rowFactor ? rowFactor[r] * (float)rows / fmaxf(*rowSum, 1e-30f) : 1.0f;
			const float cf = colFactor ? colFactor[c] * (float)cols / fmaxf(*colSum, 1e-30f) : 1.0f;
			factorShape = fmaxf(rf * cf, 1e-30f);
		}
		const float vhat = fmaxf((*scaleEma / bcf) * factorShape, 1e-30f);
		const float mhat = sm[threadIdx.x] / bc1;
		const float u = mhat / (sqrtf(vhat) + eps);
		float global = 1.0f;
		if (deltaF > 0.0f && delayedLengthSq)
		{
			const float prev = *delayedLengthSq;
			if (prev > deltaF * deltaF) global = deltaF * rsqrtf(prev);
		}
		const float p0 = orbit_load<ParamT>(param, idx);
		const float decayDen = metricWd ? (sqrtf(vhat) + eps) : 1.0f;
		const float update = global * lr * u;
		const float p1 = p0 - lr * wd * p0 / decayDen - update;
		if (sizeof(ParamT) == sizeof(float))
			((float*)param)[idx] = p1;
		else
			((uint16_t*)param)[idx] = orbit_float_to_bf16_sr(p1, idx, srSeed, srStep);
		if (deltaF > 0.0f) metric = vhat * (lr * u) * (lr * u);
		if (diagnostics && (blockId & 15) == 0)
		{
			const float g = orbit_load<GradT>(grad, idx) * gradScale;
			const float g2 = g * g;
			dvals[0]=1.0f; dvals[1]=update*update; dvals[2]=p0*p0;
			dvals[3]=vhat; dvals[4]=g2; dvals[5]=vhat*vhat;
			dvals[6]=g2*g2; dvals[7]=vhat*g2;
		}
	}
	if (deltaF > 0.0f)
	{
		sm[threadIdx.x] = metric; __syncthreads();
		for (int stride = 128; stride > 0; stride >>= 1)
		{
			if (threadIdx.x < stride) sm[threadIdx.x] += sm[threadIdx.x + stride];
			__syncthreads();
		}
		if (threadIdx.x == 0) atomicAdd(currentLengthSq, sm[0]);
	}
	if (diagnostics && (blockId & 15) == 0)
	{
		__shared__ float ds[8][ORBIT_BLOCK];
		__shared__ float dmin[ORBIT_BLOCK], dmax[ORBIT_BLOCK];
#pragma unroll
		for (int k = 0; k < 8; ++k) ds[k][threadIdx.x] = dvals[k];
		dmin[threadIdx.x] = threadIdx.x < len ? factorShape : FLT_MAX;
		dmax[threadIdx.x] = threadIdx.x < len ? factorShape : 0.0f;
		__syncthreads();
		for (int stride = 128; stride > 0; stride >>= 1)
		{
			if (threadIdx.x < stride)
			{
#pragma unroll
				for (int k = 0; k < 8; ++k) ds[k][threadIdx.x] += ds[k][threadIdx.x + stride];
				dmin[threadIdx.x] = fminf(dmin[threadIdx.x], dmin[threadIdx.x + stride]);
				dmax[threadIdx.x] = fmaxf(dmax[threadIdx.x], dmax[threadIdx.x + stride]);
			}
			__syncthreads();
		}
		if (threadIdx.x == 0)
		{
#pragma unroll
			for (int k = 0; k < 8; ++k) atomicAdd(diagnostics + k, ds[k][0]);
			atomic_min_positive(diagnostics + ORBIT_DIAG_FACTOR_MIN, dmin[0]);
			atomic_max_positive(diagnostics + ORBIT_DIAG_FACTOR_MAX, dmax[0]);
		}
	}
}

template <typename T>
bool row_col_impl(const T* grad, int rows, int cols, float gradScale,
                   float betaF, float* rowEma, float* colEma, float* scaleEma,
                   float* rowSum, float* colSum,
                   float* rowPartial, float* colPartial, float* scalarScratch)
{
	if (!grad || rows <= 0 || cols <= 0 || !rowEma || !colEma || !scaleEma ||
	    !rowSum || !colSum || !rowPartial || !colPartial || !scalarScratch) return false;
	cudaStream_t stream = computeStream();
	cudaMemsetAsync(rowSum, 0, sizeof(float), stream);
	cudaMemsetAsync(colSum, 0, sizeof(float), stream);
	cudaMemsetAsync(scalarScratch, 0, sizeof(float), stream);
	dim3 grid((cols + 255) / 256, (rows + 7) / 8);
	orbit_row_col_tile_kernel<T><<<grid, ORBIT_BLOCK, 0, stream>>>(
	    grad, rows, cols, gradScale, rowPartial, colPartial);
	if (cudaGetLastError() != cudaSuccess) return false;
	const int colTiles = (cols + 255) / 256;
	const int rowTiles = (rows + 7) / 8;
	orbit_finalize_rows_kernel<<<rows, ORBIT_BLOCK, 0, stream>>>(
	    rowPartial, rows, cols, colTiles, betaF, rowEma, rowSum, scalarScratch);
	if (cudaGetLastError() != cudaSuccess) return false;
	orbit_finalize_cols_kernel<<<cols, ORBIT_BLOCK, 0, stream>>>(
	    colPartial, rows, cols, rowTiles, betaF, colEma, colSum);
	if (cudaGetLastError() != cudaSuccess) return false;
	orbit_scale_from_sample_kernel<<<1,1,0,stream>>>(scaleEma, scalarScratch, (float)rows, betaF);
	return cudaGetLastError() == cudaSuccess;
}

template <typename T>
bool scale_impl(const T* grad, int n, float gradScale, float betaF,
                float* scaleEma, float* scratch)
{
	if (!grad || n <= 0 || !scaleEma || !scratch) return false;
	cudaStream_t stream = computeStream();
	cudaMemsetAsync(scratch, 0, sizeof(float), stream);
	const int grid = min(1024, (n + ORBIT_BLOCK - 1) / ORBIT_BLOCK);
	orbit_scale_reduce_kernel<T><<<grid,ORBIT_BLOCK,0,stream>>>(grad,n,gradScale,scratch);
	if (cudaGetLastError() != cudaSuccess) return false;
	orbit_scale_from_sample_kernel<<<1,1,0,stream>>>(scaleEma,scratch,(float)n,betaF);
	return cudaGetLastError() == cudaSuccess;
}

} // anonymous namespace

size_t orbit_row_partial_count(int rows, int cols)
{
	return rows > 0 && cols > 0 ? (size_t)rows * (size_t)((cols + 255) / 256) : 0u;
}
size_t orbit_col_partial_count(int rows, int cols)
{
	return rows > 0 && cols > 0 ? (size_t)cols * (size_t)((rows + 7) / 8) : 0u;
}

bool orbit_row_col_sqsum(const float* g,int r,int c,float gs,float b,float* re,float* ce,float* se,float* rs,float* cs,float* rp,float* cp,float* sc)
{ return row_col_impl<float>(g,r,c,gs,b,re,ce,se,rs,cs,rp,cp,sc); }
bool orbit_row_col_sqsum_bf16(const uint16_t* g,int r,int c,float gs,float b,float* re,float* ce,float* se,float* rs,float* cs,float* rp,float* cp,float* sc)
{ return row_col_impl<uint16_t>(g,r,c,gs,b,re,ce,se,rs,cs,rp,cp,sc); }

bool orbit_col_sqsum_ema(const float* g, int rows, int cols, float gradScale,
                         float betaF, float* colEma, float* scaleEma,
                         float* colSum, float* colSample, float* scalarScratch)
{
	if (!g || rows<=0 || cols<=0 || !colEma || !scaleEma || !colSum || !colSample || !scalarScratch) return false;
	cudaStream_t st=computeStream();
	cudaMemsetAsync(colSample,0,(size_t)cols*sizeof(float),st);
	cudaMemsetAsync(colSum,0,sizeof(float),st);
	cudaMemsetAsync(scalarScratch,0,sizeof(float),st);
	dim3 grid((cols+255)/256,(rows+63)/64);
	orbit_embedding_col_tiles_kernel<<<grid,256,0,st>>>(g,rows,cols,gradScale,colSample);
	if(cudaGetLastError()!=cudaSuccess)return false;
	orbit_finalize_embedding_cols_kernel<<<(cols+255)/256,256,0,st>>>(colSample,rows,cols,betaF,colEma,colSum,scalarScratch);
	if(cudaGetLastError()!=cudaSuccess)return false;
	orbit_scale_from_sample_kernel<<<1,1,0,st>>>(scaleEma,scalarScratch,(float)cols,betaF);
	return cudaGetLastError()==cudaSuccess;
}

bool orbit_scale_ema(const float* g,int n,float gs,float b,float* e,float* s)
{ return scale_impl<float>(g,n,gs,b,e,s); }
bool orbit_scale_ema_bf16(const uint16_t* g,int n,float gs,float b,float* e,float* s)
{ return scale_impl<uint16_t>(g,n,gs,b,e,s); }

bool orbit_adjoint_rows(const float* dp,int rows,int cols,float weight,bool reset,float* sample)
{
	if(!dp||rows<=0||cols<=0||!sample)return false;
	cudaStream_t st=computeStream();
	if(reset)cudaMemsetAsync(sample,0,(size_t)cols*sizeof(float),st);
	orbit_adjoint_kernel<<<(cols+255)/256,256,0,st>>>(dp,rows,cols,weight,sample);
	return cudaGetLastError()==cudaSuccess;
}
bool orbit_h_colsq_strided(const float* h,int rows,int cols,int stride,float* sample)
{
	if(!h||rows<=0||cols<=0||!sample)return false;
	if(stride<1)stride=1;
	orbit_h_strided_kernel<<<(cols+255)/256,256,0,computeStream()>>>(h,rows,cols,stride,sample);
	return cudaGetLastError()==cudaSuccess;
}
bool orbit_occupancy_bwd(const float* p,int r,int c,float* s)
{
	if(!p||r<=0||c<=0||!s)return false;
	orbit_occupancy_kernel<float><<<(c+255)/256,256,0,computeStream()>>>(p,r,c,s);
	return cudaGetLastError()==cudaSuccess;
}
bool orbit_occupancy_bwd_bf16(const uint16_t* p,int r,int c,float* s)
{
	if(!p||r<=0||c<=0||!s)return false;
	orbit_occupancy_kernel<uint16_t><<<(c+255)/256,256,0,computeStream()>>>(p,r,c,s);
	return cudaGetLastError()==cudaSuccess;
}
bool orbit_embedding_input_stats(const int* ids,const float* dq,int rows,int vocab,int width,float* freq,float* mean)
{
	if(!ids||!dq||rows<=0||vocab<=0||width<=0||!freq||!mean)return false;
	cudaStream_t st=computeStream();
	cudaMemsetAsync(freq,0,(size_t)vocab*sizeof(float),st);
	cudaMemsetAsync(mean,0,sizeof(float),st);
	orbit_frequency_kernel<<<min(256,(rows+255)/256),256,0,st>>>(ids,rows,vocab,freq);
	if(cudaGetLastError()!=cudaSuccess)return false;
	const int n=rows*width;
	orbit_dq_mean_sq_kernel<<<min(1024,(n+255)/256),256,0,st>>>(dq,n,mean);
	return cudaGetLastError()==cudaSuccess;
}

bool orbit_vector_ema(const float* sample,float* ema,int n,float betaF,float* sum)
{
	if(!sample||!ema||n<=0||!sum)return false;
	cudaStream_t st=computeStream(); cudaMemsetAsync(sum,0,sizeof(float),st);
	orbit_vector_ema_kernel<<<min(1024,(n+255)/256),256,0,st>>>(sample,ema,n,betaF,sum);
	return cudaGetLastError()==cudaSuccess;
}
bool orbit_scalar_ema(const float* sample,float* ema,float betaF)
{
	if(!sample||!ema)return false;
	orbit_scalar_ema_kernel<<<1,1,0,computeStream()>>>(sample,ema,betaF);
	return cudaGetLastError()==cudaSuccess;
}
bool orbit_build_embedding_rows(const float* occ,const float* freq,const float* dq,int vocab,float kappa,float bc,float* factor,float* sum)
{
	if(!occ||!freq||!dq||vocab<=0||!factor||!sum)return false;
	cudaStream_t st=computeStream(); cudaMemsetAsync(sum,0,sizeof(float),st);
	orbit_embedding_rows_kernel<<<min(1024,(vocab+255)/256),256,0,st>>>(occ,freq,dq,vocab,kappa,fmaxf(bc,1e-12f),factor,sum);
	return cudaGetLastError()==cudaSuccess;
}
bool orbit_quantile(const float* v,int n,float q,float* out)
{
	if(!v||n<=0||!out||q<0.0f||q>1.0f)return false;
	orbit_quantile_kernel<<<1,1024,0,computeStream()>>>(v,n,q,out);
	return cudaGetLastError()==cudaSuccess;
}
bool orbit_floor_and_sum(const float* v,int n,const float* floor,float* prepared,float* sum)
{
	if(!v||n<=0||!floor||!prepared||!sum)return false;
	cudaStream_t st=computeStream(); cudaMemsetAsync(sum,0,sizeof(float),st);
	orbit_floor_sum_kernel<<<min(1024,(n+255)/256),256,0,st>>>(v,n,floor,prepared,sum);
	return cudaGetLastError()==cudaSuccess;
}
bool orbit_diagnostics_reset(float* d)
{
	if(!d)return false; orbit_diag_reset_kernel<<<1,32,0,computeStream()>>>(d);
	return cudaGetLastError()==cudaSuccess;
}

bool orbit_factored_step(float* p,const float* g,int8_t* m,float* ms,const float* rf,const float* rs,const float* cf,const float* cs,const float* se,int rows,int cols,float lr,float b1,float bf,float eps,float wd,bool metricWd,float gs,int step,int freeze,float delta,const float* prev,float* cur,float* diag)
{
	if(!p||!g||!m||!ms||!se||rows<=0||cols<=0||step<=0)return false;
	const int n=rows*cols;
	orbit_factored_step_kernel<float,float><<<(n+255)/256,256,0,computeStream()>>>(p,g,m,ms,rf,rs,cf,cs,se,rows,cols,lr,b1,bf,eps,wd,metricWd,gs,step,freeze,delta,prev,cur,0u,0u,diag);
	return cudaGetLastError()==cudaSuccess;
}
bool orbit_factored_step_bf16(uint16_t* p,const uint16_t* g,int8_t* m,float* ms,const float* rf,const float* rs,const float* cf,const float* cs,const float* se,int rows,int cols,float lr,float b1,float bf,float eps,float wd,bool metricWd,float gs,int step,int freeze,float delta,const float* prev,float* cur,uint32_t seed,uint32_t srStep,float* diag)
{
	if(!p||!g||!m||!ms||!se||rows<=0||cols<=0||step<=0)return false;
	const int n=rows*cols;
	orbit_factored_step_kernel<uint16_t,uint16_t><<<(n+255)/256,256,0,computeStream()>>>(p,g,m,ms,rf,rs,cf,cs,se,rows,cols,lr,b1,bf,eps,wd,metricWd,gs,step,freeze,delta,prev,cur,seed,srStep,diag);
	return cudaGetLastError()==cudaSuccess;
}

} // namespace gpu
} // namespace glades

#endif
