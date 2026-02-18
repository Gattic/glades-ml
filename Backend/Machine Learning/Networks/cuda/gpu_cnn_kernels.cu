// CNN CUDA kernel implementations.
#include "gpu_cnn_kernels.h"

#include <cuda_runtime.h>
#include <cstdio>

namespace glades {
namespace gpu {

// ============================================================
// im2col kernel
// ============================================================
__global__ void im2col_kernel(const float* __restrict__ input,
                              int inC, int inH, int inW,
                              int kH, int kW,
                              int strideH, int strideW,
                              int padH, int padW,
                              int outH, int outW,
                              long long totalElements,
                              float* __restrict__ output)
{
    // Each thread writes one element of the output matrix [outH*outW, inC*kH*kW].
    // Global index maps to (row, col) = (spatial_pos, channel_patch_idx).
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    const int totalCols = inC * kH * kW;
    const int spatialPos = (int)(idx / totalCols);
    const int patchIdx = (int)(idx % totalCols);

    const int oh = spatialPos / outW;
    const int ow = spatialPos % outW;

    const int c = patchIdx / (kH * kW);
    const int rem = patchIdx % (kH * kW);
    const int kh = rem / kW;
    const int kww = rem % kW;

    const int ih = oh * strideH - padH + kh;
    const int iw = ow * strideW - padW + kww;

    float val = 0.0f;
    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
        val = input[c * inH * inW + ih * inW + iw];

    output[idx] = val;
}

void im2col_gpu(const float* input,
                unsigned int inC, unsigned int inH, unsigned int inW,
                unsigned int kH, unsigned int kW,
                unsigned int strideH, unsigned int strideW,
                unsigned int padH, unsigned int padW,
                unsigned int outH, unsigned int outW,
                float* output)
{
    const long long totalElements = (long long)outH * outW * inC * kH * kW;
    if (totalElements <= 0) return;
    const int threads = 256;
    const int blocks = (int)((totalElements + threads - 1) / threads);
    im2col_kernel<<<blocks, threads>>>(input,
        (int)inC, (int)inH, (int)inW,
        (int)kH, (int)kW,
        (int)strideH, (int)strideW,
        (int)padH, (int)padW,
        (int)outH, (int)outW,
        totalElements,
        output);
}

// ============================================================
// col2im kernel
// ============================================================
__global__ void col2im_kernel(const float* __restrict__ cols,
                              int inC, int inH, int inW,
                              int kH, int kW,
                              int strideH, int strideW,
                              int padH, int padW,
                              int outH, int outW,
                              long long totalElements,
                              float* __restrict__ output)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;

    const int totalCols = inC * kH * kW;
    const int spatialPos = (int)(idx / totalCols);
    const int patchIdx = (int)(idx % totalCols);

    const int oh = spatialPos / outW;
    const int ow = spatialPos % outW;

    const int c = patchIdx / (kH * kW);
    const int rem = patchIdx % (kH * kW);
    const int kh = rem / kW;
    const int kww = rem % kW;

    const int ih = oh * strideH - padH + kh;
    const int iw = ow * strideW - padW + kww;

    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW)
        atomicAdd(&output[c * inH * inW + ih * inW + iw], cols[idx]);
}

void col2im_gpu(const float* cols,
                unsigned int inC, unsigned int inH, unsigned int inW,
                unsigned int kH, unsigned int kW,
                unsigned int strideH, unsigned int strideW,
                unsigned int padH, unsigned int padW,
                unsigned int outH, unsigned int outW,
                float* output)
{
    const long long totalElements = (long long)outH * outW * inC * kH * kW;
    if (totalElements <= 0) return;
    // Zero output first
    cudaMemset(output, 0, (size_t)inC * inH * inW * sizeof(float));
    const int threads = 256;
    const int blocks = (int)((totalElements + threads - 1) / threads);
    col2im_kernel<<<blocks, threads>>>(cols,
        (int)inC, (int)inH, (int)inW,
        (int)kH, (int)kW,
        (int)strideH, (int)strideW,
        (int)padH, (int)padW,
        (int)outH, (int)outW,
        totalElements,
        output);
}

// ============================================================
// Max pooling forward
// ============================================================
__global__ void maxpool_forward_kernel(const float* __restrict__ input,
                                       int C, int inH, int inW,
                                       int poolH, int poolW,
                                       int poolStrideH, int poolStrideW,
                                       int outH, int outW,
                                       float* __restrict__ output,
                                       int* __restrict__ argmax)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = C * outH * outW;
    if (idx >= total) return;

    const int c = idx / (outH * outW);
    const int rem = idx % (outH * outW);
    const int oh = rem / outW;
    const int ow = rem % outW;

    const int hStart = oh * poolStrideH;
    const int wStart = ow * poolStrideW;

    float maxVal = -1e30f;
    int maxIdx = 0;

    for (int ph = 0; ph < poolH; ++ph)
    {
        for (int pw = 0; pw < poolW; ++pw)
        {
            const int ih = hStart + ph;
            const int iw = wStart + pw;
            if (ih < inH && iw < inW)
            {
                const int srcIdx = c * inH * inW + ih * inW + iw;
                const float v = input[srcIdx];
                if (v > maxVal)
                {
                    maxVal = v;
                    maxIdx = srcIdx;
                }
            }
        }
    }

    output[idx] = maxVal;
    argmax[idx] = maxIdx;
}

void maxpool_forward_gpu(const float* input,
                         unsigned int C, unsigned int H, unsigned int W,
                         unsigned int poolH, unsigned int poolW,
                         unsigned int poolStrideH, unsigned int poolStrideW,
                         unsigned int outH, unsigned int outW,
                         float* output, int* argmax)
{
    const int total = (int)(C * outH * outW);
    if (total <= 0) return;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    maxpool_forward_kernel<<<blocks, threads>>>(input,
        (int)C, (int)H, (int)W,
        (int)poolH, (int)poolW,
        (int)poolStrideH, (int)poolStrideW,
        (int)outH, (int)outW,
        output, argmax);
}

// ============================================================
// Max pooling backward
// ============================================================
__global__ void maxpool_backward_kernel(const float* __restrict__ dOutput,
                                         const int* __restrict__ argmax,
                                         int total,
                                         float* __restrict__ dInput)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    atomicAdd(&dInput[argmax[idx]], dOutput[idx]);
}

void maxpool_backward_gpu(const float* dOutput,
                          const int* argmax,
                          unsigned int C, unsigned int inH, unsigned int inW,
                          unsigned int outH, unsigned int outW,
                          float* dInput)
{
    const int total = (int)(C * outH * outW);
    if (total <= 0) return;
    cudaMemset(dInput, 0, (size_t)C * inH * inW * sizeof(float));
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    maxpool_backward_kernel<<<blocks, threads>>>(dOutput, argmax, total, dInput);
}

// ============================================================
// BatchNorm forward (training)
// ============================================================
__global__ void batchnorm_forward_train_kernel(const float* __restrict__ input,
                                                int C, int N,
                                                const float* __restrict__ gamma,
                                                const float* __restrict__ beta,
                                                float eps,
                                                float* __restrict__ output,
                                                float* __restrict__ meanOut,
                                                float* __restrict__ invStdOut,
                                                float* __restrict__ normalizedOut)
{
    // One block per channel.
    const int c = blockIdx.x;
    if (c >= C) return;

    extern __shared__ float smem[];
    float* sMean = smem;
    float* sVar = smem + blockDim.x;

    const float* x = input + c * N;
    float* y = output + c * N;
    float* norm = normalizedOut + c * N;

    // Compute mean
    float localSum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        localSum += x[i];
    sMean[threadIdx.x] = localSum;
    __syncthreads();

    // Reduce to get mean
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            sMean[threadIdx.x] += sMean[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sMean[0] / (float)N;
    if (threadIdx.x == 0) meanOut[c] = mean;
    __syncthreads();

    // Compute variance
    float localVar = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        float diff = x[i] - mean;
        localVar += diff * diff;
    }
    sVar[threadIdx.x] = localVar;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            sVar[threadIdx.x] += sVar[threadIdx.x + s];
        __syncthreads();
    }
    float var = sVar[0] / (float)N;
    float invStd = rsqrtf(var + eps);
    if (threadIdx.x == 0) invStdOut[c] = invStd;
    __syncthreads();

    float g = gamma[c];
    float b = beta[c];
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        float n = (x[i] - mean) * invStd;
        norm[i] = n;
        y[i] = g * n + b;
    }
}

void batchnorm_forward_train_gpu(const float* input,
                                 unsigned int C, unsigned int N,
                                 const float* gamma, const float* beta,
                                 float eps,
                                 float* output,
                                 float* mean, float* invStd, float* normalized)
{
    if (C == 0 || N == 0) return;
    const int threads = (N < 256u) ? (int)N : 256;
    // Round to next power of 2 for reduction
    int tPow2 = 1;
    while (tPow2 < threads) tPow2 <<= 1;
    if (tPow2 > 1024) tPow2 = 1024;
    const size_t smemSize = 2 * tPow2 * sizeof(float);
    batchnorm_forward_train_kernel<<<C, tPow2, smemSize>>>(
        input, (int)C, (int)N, gamma, beta, eps,
        output, mean, invStd, normalized);
}

// ============================================================
// BatchNorm forward (inference)
// ============================================================
__global__ void batchnorm_forward_infer_kernel(const float* __restrict__ input,
                                                int C, int N,
                                                const float* __restrict__ gamma,
                                                const float* __restrict__ beta,
                                                const float* __restrict__ runMean,
                                                const float* __restrict__ runVar,
                                                float eps,
                                                float* __restrict__ output)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = C * N;
    if (idx >= total) return;

    const int c = idx / N;
    float invStd = rsqrtf(runVar[c] + eps);
    output[idx] = gamma[c] * (input[idx] - runMean[c]) * invStd + beta[c];
}

void batchnorm_forward_infer_gpu(const float* input,
                                 unsigned int C, unsigned int N,
                                 const float* gamma, const float* beta,
                                 const float* runMean, const float* runVar,
                                 float eps,
                                 float* output)
{
    const int total = (int)(C * N);
    if (total <= 0) return;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    batchnorm_forward_infer_kernel<<<blocks, threads>>>(
        input, (int)C, (int)N, gamma, beta, runMean, runVar, eps, output);
}

// ============================================================
// BatchNorm backward
// ============================================================
__global__ void batchnorm_backward_kernel(const float* __restrict__ dOutput,
                                           const float* __restrict__ normalized,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ invStd,
                                           int C, int N,
                                           float* __restrict__ dInput,
                                           float* __restrict__ dgamma,
                                           float* __restrict__ dbeta)
{
    // One block per channel.
    const int c = blockIdx.x;
    if (c >= C) return;

    extern __shared__ float smem[];
    float* sDgamma = smem;
    float* sDbeta = smem + blockDim.x;

    const float* dy = dOutput + c * N;
    const float* xn = normalized + c * N;
    float* dx = dInput + c * N;
    float g = gamma[c];
    float is = invStd[c];

    // Accumulate dgamma and dbeta
    float localDg = 0.0f, localDb = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        localDg += dy[i] * xn[i];
        localDb += dy[i];
    }
    sDgamma[threadIdx.x] = localDg;
    sDbeta[threadIdx.x] = localDb;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sDgamma[threadIdx.x] += sDgamma[threadIdx.x + s];
            sDbeta[threadIdx.x] += sDbeta[threadIdx.x + s];
        }
        __syncthreads();
    }

    float dgammaC = sDgamma[0];
    float dbetaC = sDbeta[0];
    if (threadIdx.x == 0)
    {
        dgamma[c] += dgammaC;
        dbeta[c] += dbetaC;
    }
    __syncthreads();

    // Compute dx
    float invN = 1.0f / (float)N;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        dx[i] = g * is * invN * ((float)N * dy[i] - dbetaC - xn[i] * dgammaC);
    }
}

void batchnorm_backward_gpu(const float* dOutput,
                            const float* normalized,
                            const float* gamma,
                            const float* invStd,
                            unsigned int C, unsigned int N,
                            float* dInput,
                            float* dgamma, float* dbeta)
{
    if (C == 0 || N == 0) return;
    int threads = (N < 256u) ? (int)N : 256;
    int tPow2 = 1;
    while (tPow2 < threads) tPow2 <<= 1;
    if (tPow2 > 1024) tPow2 = 1024;
    const size_t smemSize = 2 * tPow2 * sizeof(float);
    batchnorm_backward_kernel<<<C, tPow2, smemSize>>>(
        dOutput, normalized, gamma, invStd, (int)C, (int)N,
        dInput, dgamma, dbeta);
}

} // namespace gpu
} // namespace glades
