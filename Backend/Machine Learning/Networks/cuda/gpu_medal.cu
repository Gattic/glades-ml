// MEDAL GPU primitives — paradigm #262 iter 43.
//
// Implements the discrete absorbing-mask CTMC corruption sampler and the
// masked-CE loss-mask kernels.  See gpu_medal.h for API contract and
// research/PARADIGM_SHIFT_262_MEDAL_DESIGN.md for the full mathematical
// design.

#include "gpu_medal.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdint>

namespace glades {
namespace gpu {

// -------------------------------------------------------------------- //
// Kernel: medal_corrupt_tokens
//
// Per-thread: one position.  Loads original token, draws uniform [0,1),
// writes either V_mask or the original, and writes the mask bit.
// -------------------------------------------------------------------- //
__global__ void medal_corrupt_tokens_kernel(const int* __restrict__ tokens,
                                            int T, int V_mask, float alpha,
                                            uint64_t seed,
                                            int* __restrict__ tokens_corr,
                                            unsigned char* __restrict__ mask)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;

    // Per-thread Philox4 state seeded by (seed, position).  No subsequence
    // bookkeeping needed since each thread draws one sample.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)i, 0, &state);
    float u = curand_uniform(&state);  // (0, 1]

    bool is_masked = (u < alpha);
    tokens_corr[i] = is_masked ? V_mask : tokens[i];
    mask[i]        = is_masked ? (unsigned char)1 : (unsigned char)0;
}

bool medal_corrupt_tokens(const int* d_tokens, int T, int V_mask,
                          float alpha, uint64_t seed,
                          int* d_tokens_corr, unsigned char* d_mask)
{
    if (T <= 0) return false;
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    int threads = 256;
    int blocks  = (T + threads - 1) / threads;
    medal_corrupt_tokens_kernel<<<blocks, threads>>>(
        d_tokens, T, V_mask, alpha, seed, d_tokens_corr, d_mask);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[medal] corrupt_tokens launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

// -------------------------------------------------------------------- //
// Kernel: medal_mask_dlogits (FP32 in-place row-zero)
//
// One block per row (T blocks).  Within block, threads stride over V cols
// and zero out if mask[row] == 0.  This is memory-bandwidth-bound; the
// zero-write on mask==0 rows is the dominant cost.
// -------------------------------------------------------------------- //
__global__ void medal_mask_dlogits_kernel(float* __restrict__ dlogits,
                                          const unsigned char* __restrict__ mask,
                                          int T, int V)
{
    int row = blockIdx.x;
    if (row >= T) return;
    unsigned char m = mask[row];
    if (m != 0) return;  // keep masked-position dlogits (do nothing)

    float* row_ptr = dlogits + (size_t)row * V;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int j = tid; j < V; j += stride) row_ptr[j] = 0.0f;
}

bool medal_mask_dlogits(float* d_logits, const unsigned char* d_mask,
                        int T, int V)
{
    if (T <= 0 || V <= 0) return false;
    int threads = 256;
    medal_mask_dlogits_kernel<<<T, threads>>>(d_logits, d_mask, T, V);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[medal] mask_dlogits launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

// -------------------------------------------------------------------- //
// BF16 variant.
// -------------------------------------------------------------------- //
__global__ void medal_mask_dlogits_bf16_kernel(uint16_t* __restrict__ dlogits_bf,
                                               const unsigned char* __restrict__ mask,
                                               int T, int V)
{
    int row = blockIdx.x;
    if (row >= T) return;
    unsigned char m = mask[row];
    if (m != 0) return;

    uint16_t* row_ptr = dlogits_bf + (size_t)row * V;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    // BF16 zero = 0x0000.
    for (int j = tid; j < V; j += stride) row_ptr[j] = (uint16_t)0;
}

bool medal_mask_dlogits_bf16(uint16_t* d_logits_bf, const unsigned char* d_mask,
                             int T, int V)
{
    if (T <= 0 || V <= 0) return false;
    int threads = 256;
    medal_mask_dlogits_bf16_kernel<<<T, threads>>>(d_logits_bf, d_mask, T, V);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[medal] mask_dlogits_bf16 launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

// -------------------------------------------------------------------- //
// Kernel: medal_masked_nll
//
// One block per row.  Within block, each thread computes -log p[row,target]
// if mask[row] == 1; atomic-adds to d_nll_sum and increments d_n_masked.
// Only thread 0 in each block does the atomic to avoid contention.
// -------------------------------------------------------------------- //
__global__ void medal_masked_nll_kernel(const float* __restrict__ probs,
                                        const int* __restrict__ targets,
                                        const unsigned char* __restrict__ mask,
                                        int T, int V,
                                        float* __restrict__ nll_sum,
                                        int* __restrict__ n_masked)
{
    int row = blockIdx.x;
    if (row >= T || threadIdx.x != 0) return;

    if (mask[row] == 0) return;
    int t = targets[row];
    if (t < 0 || t >= V) return;
    float p = probs[(size_t)row * V + t];
    if (p < 1e-30f) p = 1e-30f;  // safety
    float nll = -__logf(p);

    atomicAdd(nll_sum, nll);
    atomicAdd(n_masked, 1);
}

bool medal_masked_nll(const float* d_probs, const int* d_targets,
                      const unsigned char* d_mask, int T, int V,
                      float* d_nll_sum, int* d_n_masked)
{
    if (T <= 0 || V <= 0) return false;
    // Zero accumulators first.
    cudaMemset(d_nll_sum,  0, sizeof(float));
    cudaMemset(d_n_masked, 0, sizeof(int));
    int threads = 32;  // only thread 0 active anyway; tiny block keeps SM happy
    medal_masked_nll_kernel<<<T, threads>>>(
        d_probs, d_targets, d_mask, T, V, d_nll_sum, d_n_masked);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[medal] masked_nll launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

// -------------------------------------------------------------------- //
// Kernel: medal_masked_nll_bf16
//
// BF16-storage variant of medal_masked_nll for the bf16-logits-storage path.
// One block per row; thread 0 looks up p[row, targets[row]] in BF16 storage,
// promotes to FP32, takes -log, and atomicAdd's to the scalar accumulator.
// -------------------------------------------------------------------- //
__global__ void medal_masked_nll_bf16_kernel(const uint16_t* __restrict__ probs_bf,
                                              const int* __restrict__ targets,
                                              const unsigned char* __restrict__ mask,
                                              int T, int V,
                                              float* __restrict__ nll_sum,
                                              int* __restrict__ n_masked)
{
    int row = blockIdx.x;
    if (row >= T || threadIdx.x != 0) return;

    if (mask[row] == 0) return;
    int t = targets[row];
    if (t < 0 || t >= V) return;
    uint16_t bits = probs_bf[(size_t)row * V + t];
    union { float f; uint32_t u; } v;
    v.u = ((uint32_t)bits) << 16;
    float p = v.f;
    if (p < 1e-30f) p = 1e-30f;
    float nll = -__logf(p);

    atomicAdd(nll_sum, nll);
    atomicAdd(n_masked, 1);
}

bool medal_masked_nll_bf16(const uint16_t* d_probs_bf, const int* d_targets,
                           const unsigned char* d_mask, int T, int V,
                           float* d_nll_sum, int* d_n_masked)
{
    if (T <= 0 || V <= 0) return false;
    cudaMemset(d_nll_sum,  0, sizeof(float));
    cudaMemset(d_n_masked, 0, sizeof(int));
    int threads = 32;
    medal_masked_nll_bf16_kernel<<<T, threads>>>(
        d_probs_bf, d_targets, d_mask, T, V, d_nll_sum, d_n_masked);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "[medal] masked_nll_bf16 launch failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
