// ORION INT8 V buffer kernels — staged draft for paradigm #43 v4.
//
// Mirrors iter 49's int8-Adam blockscaled quantization pattern (block size 256,
// per-block FP32 scale).  V is stored column-major as INT8 with per-block scales;
// each column has its own scale array of length scale_count = ceil(n / 256).
// Total VRAM: n*r*1 (data) + ceil(n/256)*r*4 (scales) ≈ n*r*1.015 bytes.
// vs. BF16: n*r*2 bytes → ~50 % savings on V.
//
// Kernels:
//   orion_v_quantize_int8_column         FP32 → INT8 + scales (one column)
//   orion_v_dequantize_int8_column       INT8 + scales → FP32 (one column)
//   orion_axpy_int8_v_column             theta += alpha * V[:, k] (fused dequant + axpy)
//   orion_vt_dot_int8_columns            result[k] = V[:, k] · x   for all k
//
// All on the default CUDA stream; ChironParams patterns work via stream() helper.
//
// Build (after copying into the canonical gpu_kernels.cu, or compiling standalone):
//   nvcc -O2 -arch=sm_89 -dc -c orion_int8_v_kernels.cu -o orion_int8_v_kernels.o

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#ifndef ORION_V_BS
#define ORION_V_BS 256          // matches ADAM_INT8_BS (iter 49)
#endif

#ifndef ORION_V_TPB
#define ORION_V_TPB 256
#endif

static __device__ __forceinline__ int orion_v_scale_count_dev(int n) {
    return (n + ORION_V_BS - 1) / ORION_V_BS;
}

// ---------------------------------------------------------------------------
// 1. Quantize one FP32 column of V → INT8 + per-block FP32 scales.
//   Launch: <<<scale_count, ORION_V_TPB>>>
//   src:     FP32 column [n] on device
//   dst_q:   INT8 column [n] on device
//   scales:  FP32 [scale_count] on device
// ---------------------------------------------------------------------------
__global__ void orion_v_quantize_int8_column_kernel(
    const float* __restrict__ src,
    int8_t* __restrict__ dst_q,
    float* __restrict__ scales,
    int n)
{
    const int blockId = blockIdx.x;
    const int scale_count = orion_v_scale_count_dev(n);
    if (blockId >= scale_count) return;

    const int start = blockId * ORION_V_BS;
    const int end   = (start + ORION_V_BS < n) ? (start + ORION_V_BS) : n;
    const int len   = end - start;

    // Pass 1: absmax via warp shuffle.
    float localMax = 0.0f;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        const float v = src[start + i];
        const float a = fabsf(v);
        if (a > localMax) localMax = a;
    }
    for (int off = 16; off > 0; off /= 2) {
        const float o = __shfl_xor_sync(0xFFFFFFFFu, localMax, off);
        if (o > localMax) localMax = o;
    }
    __shared__ float sWarpMax[32];
    const int warpId = threadIdx.x / 32;
    const int lane   = threadIdx.x % 32;
    if (lane == 0) sWarpMax[warpId] = localMax;
    __syncthreads();

    if (warpId == 0) {
        float v = (lane < (blockDim.x / 32)) ? sWarpMax[lane] : 0.0f;
        for (int off = 16; off > 0; off /= 2) {
            const float o = __shfl_xor_sync(0xFFFFFFFFu, v, off);
            if (o > v) v = o;
        }
        if (lane == 0) scales[blockId] = v;
    }
    __syncthreads();

    const float blockMax = scales[blockId];
    const float invScale = (blockMax > 1e-30f) ? (127.0f / blockMax) : 0.0f;

    // Pass 2: quantize. Round to nearest, clamp to [-127, +127] (we leave -128
    // unused so a single FP32 invScale dequantizes symmetrically).
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        const float v = src[start + i];
        int q = (int)__float2int_rn(v * invScale);
        if (q < -127) q = -127;
        if (q >  127) q =  127;
        dst_q[start + i] = (int8_t)q;
    }
}

// ---------------------------------------------------------------------------
// 2. Dequantize one INT8 column → FP32.
//   Launch: <<< (n + ORION_V_TPB - 1) / ORION_V_TPB, ORION_V_TPB >>>
// ---------------------------------------------------------------------------
__global__ void orion_v_dequantize_int8_column_kernel(
    const int8_t* __restrict__ src_q,
    const float* __restrict__ scales,
    float* __restrict__ dst,
    int n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const int blockId = tid / ORION_V_BS;
    const float scale = scales[blockId];
    const float qinv  = scale * (1.0f / 127.0f);

    dst[tid] = (float)src_q[tid] * qinv;
}

// ---------------------------------------------------------------------------
// 3. Fused theta += alpha * V[:, col_idx] where V is column-major INT8.
//   V_q: int8 array [n*r] (column col_idx at &V_q[col_idx * n])
//   scales: float [scale_count * r]
//   Launch: <<< (n + ORION_V_TPB - 1) / ORION_V_TPB, ORION_V_TPB >>>
// Avoids a separate dequantize pass; the FMA happens in registers.
// ---------------------------------------------------------------------------
__global__ void orion_axpy_int8_v_column_kernel(
    float* __restrict__ theta,
    const int8_t* __restrict__ V_q,
    const float* __restrict__ scales,
    int n,
    int col_idx,
    int scale_count,
    float alpha)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const size_t base_q     = (size_t)col_idx * (size_t)n;
    const size_t base_scale = (size_t)col_idx * (size_t)scale_count;
    const int    blockId    = tid / ORION_V_BS;
    const float  scale      = scales[base_scale + blockId];
    const float  qinv       = scale * (1.0f / 127.0f);

    const float v_val = (float)V_q[base_q + tid] * qinv;
    theta[tid] += alpha * v_val;
}

// ---------------------------------------------------------------------------
// 4. result[k] = V[:, k] · x for all k in [0, r).  Fused dequant + dot.
//   V_q: int8 [n*r] column-major
//   scales: float [scale_count*r] column-major
//   x: float [n]
//   result: float [r]
//   Launch: <<< r, ORION_V_TPB >>>  (one block per column)
// ---------------------------------------------------------------------------
__global__ void orion_vt_dot_int8_columns_kernel(
    const int8_t* __restrict__ V_q,
    const float* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ result,
    int n,
    int scale_count)
{
    const int col_idx       = blockIdx.x;
    const size_t base_q     = (size_t)col_idx * (size_t)n;
    const size_t base_scale = (size_t)col_idx * (size_t)scale_count;

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int   blockId = i / ORION_V_BS;
        const float scale   = scales[base_scale + blockId];
        const float qinv    = scale * (1.0f / 127.0f);
        const float v_val   = (float)V_q[base_q + i] * qinv;
        localSum += v_val * x[i];
    }
    for (int off = 16; off > 0; off /= 2) {
        localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, off);
    }
    __shared__ float sWarpSum[32];
    const int warpId = threadIdx.x / 32;
    const int lane   = threadIdx.x % 32;
    if (lane == 0) sWarpSum[warpId] = localSum;
    __syncthreads();

    if (warpId == 0) {
        float v = (lane < (blockDim.x / 32)) ? sWarpSum[lane] : 0.0f;
        for (int off = 16; off > 0; off /= 2) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
        if (lane == 0) result[col_idx] = v;
    }
}

// ---------------------------------------------------------------------------
// Host-callable wrappers.
// ---------------------------------------------------------------------------
extern "C" {

int orion_v_scale_count(int n) { return (n + ORION_V_BS - 1) / ORION_V_BS; }

bool orion_v_quantize_int8_column(const float* src, int8_t* dst_q,
                                   float* scales, int n,
                                   cudaStream_t stream)
{
    const int scale_count = orion_v_scale_count(n);
    orion_v_quantize_int8_column_kernel<<<scale_count, ORION_V_TPB, 0, stream>>>(
        src, dst_q, scales, n);
    return cudaGetLastError() == cudaSuccess;
}

bool orion_v_dequantize_int8_column(const int8_t* src_q, const float* scales,
                                     float* dst, int n,
                                     cudaStream_t stream)
{
    const int grid = (n + ORION_V_TPB - 1) / ORION_V_TPB;
    orion_v_dequantize_int8_column_kernel<<<grid, ORION_V_TPB, 0, stream>>>(
        src_q, scales, dst, n);
    return cudaGetLastError() == cudaSuccess;
}

bool orion_axpy_int8_v_column(float* theta, const int8_t* V_q,
                               const float* scales, int n, int col_idx,
                               float alpha, cudaStream_t stream)
{
    const int scale_count = orion_v_scale_count(n);
    const int grid = (n + ORION_V_TPB - 1) / ORION_V_TPB;
    orion_axpy_int8_v_column_kernel<<<grid, ORION_V_TPB, 0, stream>>>(
        theta, V_q, scales, n, col_idx, scale_count, alpha);
    return cudaGetLastError() == cudaSuccess;
}

bool orion_vt_dot_int8_columns(const int8_t* V_q, const float* scales,
                                const float* x, float* result,
                                int n, int r, cudaStream_t stream)
{
    const int scale_count = orion_v_scale_count(n);
    orion_vt_dot_int8_columns_kernel<<<r, ORION_V_TPB, 0, stream>>>(
        V_q, scales, x, result, n, scale_count);
    return cudaGetLastError() == cudaSuccess;
}

} // extern "C"

// ---------------------------------------------------------------------------
// Integration sketch (for chiron_main.cpp's OrionTensor struct):
//
//   struct OrionTensor {
//       ...
//       // ORION v4 quantized storage (opt-in via --orion-int8-v):
//       glades::gpu::GpuBuffer<int8_t> V_int8;        // [n*r] column-major
//       glades::gpu::GpuBuffer<float>  V_scales;      // [scale_count*r]
//       bool use_int8_v;
//       ...
//   };
//
// At Oja-update time:
//   if (t->use_int8_v) {
//       // dequantize current V column → FP32 scratch
//       orion_v_dequantize_int8_column(V_int8.data() + col*n, V_scales.data() + col*sc, scratch.data(), n, stream);
//       // existing Oja update on scratch
//       ... (apply tilt, Gram-Schmidt etc.)
//       // quantize back
//       orion_v_quantize_int8_column(scratch.data(), V_int8.data() + col*n, V_scales.data() + col*sc, n, stream);
//   } else {
//       // existing BF16 path
//   }
//
// At lift-back time (theta += V · Δα):
//   if (t->use_int8_v) {
//       for (int k = 0; k < r; ++k)
//           orion_axpy_int8_v_column(t->theta, V_int8.data(), V_scales.data(),
//                                    n, k, delta_alpha[k], stream);
//   } else {
//       // existing BF16 path
//   }
//
// VRAM budget at 1B params r=4:
//   BF16 V:   871M * 4 * 2 = 6.97 GB
//   INT8 V:   871M * 4 * 1 = 3.49 GB  (data)
//           + 871M/256 * 4 * 4 = 53 MB (scales)
//   Net save: 3.43 GB (49.3 %)
// ---------------------------------------------------------------------------
