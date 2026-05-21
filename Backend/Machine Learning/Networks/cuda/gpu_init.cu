// GPU parameter init kernels (curand Philox 4x32-10).
//
// Replaces host xorshift64* Glorot/normal init for the --gpu path.

#include "gpu_init.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

namespace glades {
namespace gpu {

namespace {

// Fixed launch geometry => bit-exact reproducibility per (seed, tensor_id).
// The grid-stride loop handles tensors of any N >= kBlocks*kBlock without
// changing per-thread Philox streams.
constexpr int kBlock  = 256;
constexpr int kBlocks = 1024;  // 256k threads total — saturates Ada SMs.

// Mix constant: 2^64 / phi.  Avoids correlated streams when tensor_ids
// are sequential small integers.
__device__ __forceinline__ unsigned long long
mixSeed(unsigned long long seed, unsigned long long tensor_id)
{
    return seed ^ (tensor_id * 0x9E3779B97F4A7C15ULL);
}

__global__ void glorotUniformKernel(float* W, std::size_t N, float limit,
                                    unsigned long long seed,
                                    unsigned long long tid)
{
    const std::size_t t = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(mixSeed(seed, tid), /*subsequence=*/(unsigned long long)t,
                /*offset=*/0ULL, &state);

    for (std::size_t i = t; i < N; i += stride)
    {
        // curand_uniform returns (0, 1] — map to [-limit, +limit).
        const float u = curand_uniform(&state);
        W[i] = (u * 2.0f - 1.0f) * limit;
    }
}

__global__ void normalKernel(float* W, std::size_t N, float mean, float stddev,
                             unsigned long long seed,
                             unsigned long long tid)
{
    const std::size_t t = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(mixSeed(seed, tid), /*subsequence=*/(unsigned long long)t,
                /*offset=*/0ULL, &state);

    for (std::size_t i = t; i < N; i += stride)
    {
        const float n = curand_normal(&state);
        W[i] = mean + stddev * n;
    }
}

__global__ void constantKernel(float* W, std::size_t N, float val)
{
    const std::size_t t = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
    for (std::size_t i = t; i < N; i += stride)
        W[i] = val;
}

} // namespace

void initGlorotUniform(float* d_W, std::size_t N,
                       unsigned int fanIn, unsigned int fanOut,
                       uint64_t seed, uint64_t tensor_id)
{
    if (d_W == 0 || N == 0 || fanIn == 0u || fanOut == 0u)
        return;
    const float limit = (float)sqrt(6.0 / (double)((uint64_t)fanIn + (uint64_t)fanOut));
    glorotUniformKernel<<<kBlocks, kBlock>>>(d_W, N, limit,
                                             (unsigned long long)seed,
                                             (unsigned long long)tensor_id);
}

void initNormal(float* d_W, std::size_t N, float mean, float stddev,
                uint64_t seed, uint64_t tensor_id)
{
    if (d_W == 0 || N == 0)
        return;
    normalKernel<<<kBlocks, kBlock>>>(d_W, N, mean, stddev,
                                      (unsigned long long)seed,
                                      (unsigned long long)tensor_id);
}

void initZeros(float* d_W, std::size_t N)
{
    if (d_W == 0 || N == 0)
        return;
    cudaMemsetAsync(d_W, 0, sizeof(float) * N);
}

void initConstant(float* d_W, std::size_t N, float val)
{
    if (d_W == 0 || N == 0)
        return;
    if (val == 0.0f)
    {
        cudaMemsetAsync(d_W, 0, sizeof(float) * N);
        return;
    }
    constantKernel<<<kBlocks, kBlock>>>(d_W, N, val);
}

} // namespace gpu
} // namespace glades
