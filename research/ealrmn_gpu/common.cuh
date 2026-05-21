// EALRMN Phase-1 GPU prototype — common infrastructure
// Tensor allocation, error checking, RNG, common kernels.
#ifndef EALRMN_COMMON_CUH
#define EALRMN_COMMON_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>

// ===== Error checking =====

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        std::exit(1);                                                  \
    }                                                                  \
} while (0)

#define CUBLAS_CHECK(call) do {                                        \
    cublasStatus_t _s = (call);                                        \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                 \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",                    \
                __FILE__, __LINE__, (int)_s);                          \
        std::exit(1);                                                  \
    }                                                                  \
} while (0)

// ===== Tensor helpers =====

struct Tensor {
    float* d = nullptr;          // device pointer
    int64_t numel = 0;
    std::vector<int> shape;
    void zero_() { CUDA_CHECK(cudaMemset(d, 0, numel * sizeof(float))); }
};

inline Tensor make_tensor(const std::vector<int>& shape) {
    Tensor t;
    t.shape = shape;
    int64_t n = 1;
    for (int s : shape) n *= s;
    t.numel = n;
    CUDA_CHECK(cudaMalloc(&t.d, n * sizeof(float)));
    return t;
}

inline void free_tensor(Tensor& t) {
    if (t.d) cudaFree(t.d);
    t.d = nullptr;
    t.numel = 0;
    t.shape.clear();
}

inline void copy_h2d(Tensor& t, const std::vector<float>& h) {
    CUDA_CHECK(cudaMemcpy(t.d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
}

inline void copy_d2h(std::vector<float>& h, const Tensor& t) {
    h.resize(t.numel);
    CUDA_CHECK(cudaMemcpy(h.data(), t.d, t.numel * sizeof(float), cudaMemcpyDeviceToHost));
}

inline std::vector<float> tensor_to_host(const Tensor& t) {
    std::vector<float> h(t.numel);
    CUDA_CHECK(cudaMemcpy(h.data(), t.d, t.numel * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// Norm utilities (for debugging / grad-checking)
__global__ void sum_sq_kernel(const float* x, int n, float* out) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    while (idx < n) {
        float v = x[idx];
        local += v * v;
        idx += blockDim.x * gridDim.x;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

inline float tensor_l2norm(const Tensor& t) {
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    int block = 256;
    int grid = (t.numel + block - 1) / block;
    if (grid > 1024) grid = 1024;
    sum_sq_kernel<<<grid, block, block * sizeof(float)>>>(t.d, (int)t.numel, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    float h;
    CUDA_CHECK(cudaMemcpy(&h, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    return std::sqrt(h);
}

// ===== RNG =====

// Host-side splitmix64 + xorshift for reproducible CPU-side random ops.
// Splitmix64 hash on input seed ensures distinct seeds produce distinct streams,
// including seed=0 (xorshift alone would collapse seed=0 → state=1).
struct HostRng {
    uint64_t state;
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    HostRng(uint64_t seed = 42) : state(splitmix64(seed + 0xCAFEBABEDEADBEEFULL)) {
        if (state == 0) state = 0xDEADBEEFCAFEBABEULL;
    }
    uint64_t next_u64() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    float next_uniform() {
        return (next_u64() >> 11) * (1.0f / (1ULL << 53));
    }
    float next_normal() {
        float u1 = next_uniform();
        float u2 = next_uniform();
        if (u1 < 1e-9f) u1 = 1e-9f;
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
    }
    int next_int(int hi) {
        return (int)(next_u64() % (uint64_t)hi);
    }
};

// Device-side RNG kernels for fast on-GPU initialization
__global__ void curand_init_kernel(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void fill_normal_kernel(float* x, int n, float mean, float stddev, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState s = states[idx];
        x[idx] = mean + stddev * curand_normal(&s);
        states[idx] = s;
    }
}

__global__ void fill_uniform_kernel(float* x, int n, float lo, float hi, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState s = states[idx];
        x[idx] = lo + (hi - lo) * curand_uniform(&s);
        states[idx] = s;
    }
}

struct GpuRng {
    curandState* states = nullptr;
    int n_states = 0;
    void init(int n, unsigned long long seed) {
        n_states = n;
        CUDA_CHECK(cudaMalloc(&states, n * sizeof(curandState)));
        int block = 256;
        int grid = (n + block - 1) / block;
        curand_init_kernel<<<grid, block>>>(states, n, seed);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void free() {
        if (states) cudaFree(states);
        states = nullptr; n_states = 0;
    }
    void normal_(Tensor& t, float mean, float stddev) {
        int n = (int)t.numel;
        if (n > n_states) {
            fprintf(stderr, "GpuRng: not enough states (%d < %d)\n", n_states, n);
            std::exit(1);
        }
        int block = 256;
        int grid = (n + block - 1) / block;
        fill_normal_kernel<<<grid, block>>>(t.d, n, mean, stddev, states);
    }
    void uniform_(Tensor& t, float lo, float hi) {
        int n = (int)t.numel;
        if (n > n_states) {
            fprintf(stderr, "GpuRng: not enough states\n");
            std::exit(1);
        }
        int block = 256;
        int grid = (n + block - 1) / block;
        fill_uniform_kernel<<<grid, block>>>(t.d, n, lo, hi, states);
    }
};

// Host-side init helpers (write FP32 to device via copy)
inline void init_xavier_uniform(Tensor& t, int fan_in, int fan_out, HostRng& rng) {
    float bound = std::sqrt(6.0f / (float)(fan_in + fan_out));
    std::vector<float> h(t.numel);
    for (int64_t i = 0; i < t.numel; ++i) h[i] = (rng.next_uniform() * 2.0f - 1.0f) * bound;
    copy_h2d(t, h);
}

inline void init_normal(Tensor& t, float mean, float stddev, HostRng& rng) {
    std::vector<float> h(t.numel);
    for (int64_t i = 0; i < t.numel; ++i) h[i] = mean + stddev * rng.next_normal();
    copy_h2d(t, h);
}

inline void init_zero(Tensor& t) { t.zero_(); }

inline void init_orthogonal_scale(Tensor& t, int dim, float scale, HostRng& rng) {
    // Initialize a (dim, dim) matrix with QR of random Gaussian, then scale.
    // For Koopman K init: dim×dim, spectral radius ~ scale (target < 1).
    std::vector<float> A(dim * dim);
    for (int i = 0; i < dim * dim; ++i) A[i] = rng.next_normal();
    // Modified Gram-Schmidt (rows interpreted as vectors).
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < i; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < dim; ++k) dot += A[i * dim + k] * A[j * dim + k];
            for (int k = 0; k < dim; ++k) A[i * dim + k] -= dot * A[j * dim + k];
        }
        float n = 0.0f;
        for (int k = 0; k < dim; ++k) n += A[i * dim + k] * A[i * dim + k];
        n = std::sqrt(n);
        if (n < 1e-9f) n = 1e-9f;
        for (int k = 0; k < dim; ++k) A[i * dim + k] /= n;
    }
    for (int i = 0; i < dim * dim; ++i) A[i] *= scale;
    copy_h2d(t, A);
}

// ===== Common elementwise kernels =====

__global__ void k_zero(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0f;
}

__global__ void k_add_inplace(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void k_scale_inplace(float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

__global__ void k_add_scaled(float* x, const float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a * y[i];
}

__global__ void k_copy(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

inline void launch_1d(int n, int block, int& grid) {
    grid = (n + block - 1) / block;
}

#define LAUNCH_1D(name, n, ...) do {                              \
    int _block = 256;                                             \
    int _grid = (((int)(n)) + _block - 1) / _block;               \
    name<<<_grid, _block>>>(__VA_ARGS__);                         \
} while (0)

// ===== Activation kernels =====

__device__ inline float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void k_sigmoid_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sigmoidf_dev(x[i]);
}

__global__ void k_sigmoid_bwd(const float* y, const float* dy, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * y[i] * (1.0f - y[i]);
}

__global__ void k_tanh_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}

__global__ void k_tanh_bwd(const float* y, const float* dy, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = dy[i] * (1.0f - y[i] * y[i]);
}

__global__ void k_gelu_fwd(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float c = 0.7978845608028654f; // sqrt(2/pi)
        float a = c * (v + 0.044715f * v * v * v);
        y[i] = 0.5f * v * (1.0f + tanhf(a));
    }
}

__global__ void k_gelu_bwd(const float* x, const float* dy, float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float c = 0.7978845608028654f;
        float u = c * (v + 0.044715f * v * v * v);
        float t = tanhf(u);
        float du = c * (1.0f + 3.0f * 0.044715f * v * v);
        dx[i] = dy[i] * (0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * du);
    }
}

#endif // EALRMN_COMMON_CUH
