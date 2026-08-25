// EALRMN Phase-1 GPU prototype — AdamW optimizer + grad clipping.
#ifndef EALRMN_OPTIMIZER_CUH
#define EALRMN_OPTIMIZER_CUH

#include "common.cuh"

__global__ void k_adamw_step(float* w, float* m, float* v, const float* g,
                             int n, float lr, float b1, float b2, float eps,
                             float wd, float bc1, float bc2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.0f - b1) * gi;
    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    float update = m_hat / (sqrtf(v_hat) + eps);
    // Decoupled weight decay
    w[i] -= lr * (update + wd * w[i]);
}

inline void adamw_step(Tensor& w, Tensor& m, Tensor& v, const Tensor& g,
                       float lr, float b1, float b2, float eps,
                       float wd, int step) {
    int n = (int)w.numel;
    float bc1 = 1.0f - std::pow(b1, (float)step);
    float bc2 = 1.0f - std::pow(b2, (float)step);
    int block = 256;
    int grid = (n + block - 1) / block;
    k_adamw_step<<<grid, block>>>(w.d, m.d, v.d, g.d, n, lr, b1, b2, eps, wd, bc1, bc2);
}

// Compute global L2 norm of all gradient tensors.
__global__ void k_sum_sq_into(const float* g, int n, float* out_acc) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local = 0.0f;
    while (idx < n) {
        float gi = g[idx];
        local += gi * gi;
        idx += blockDim.x * gridDim.x;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out_acc, sdata[0]);
}

inline float global_grad_norm(const std::vector<Tensor*>& grads) {
    float* d_acc;
    CUDA_CHECK(cudaMalloc(&d_acc, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_acc, 0, sizeof(float)));
    for (Tensor* g : grads) {
        int n = (int)g->numel;
        if (n == 0) continue;
        int block = 256;
        int grid = (n + block - 1) / block;
        if (grid > 1024) grid = 1024;
        k_sum_sq_into<<<grid, block, block * sizeof(float)>>>(g->d, n, d_acc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    float h;
    CUDA_CHECK(cudaMemcpy(&h, d_acc, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_acc);
    return std::sqrt(h);
}

__global__ void k_scale(float* x, int n, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

inline void clip_grads(const std::vector<Tensor*>& grads, float max_norm) {
    float gn = global_grad_norm(grads);
    if (gn <= max_norm || gn <= 1e-12f) return;
    float scale = max_norm / gn;
    for (Tensor* g : grads) {
        int n = (int)g->numel;
        if (n == 0) continue;
        int block = 256;
        int grid = (n + block - 1) / block;
        k_scale<<<grid, block>>>(g->d, n, scale);
    }
}

#endif // EALRMN_OPTIMIZER_CUH
