// EALRMN Phase-1 GPU prototype — recurrence + memory + attention readout kernels
// Specialized GPU kernels for the EALRMN model.
#ifndef EALRMN_RECURRENCE_CUH
#define EALRMN_RECURRENCE_CUH

#include "common.cuh"
#include "kernels.cuh"

// ===== Per-slot gate computation =====
// Forward: g[b, j] = sigmoid(<s[b, :], W_g[j, :]> + b_g[j])
// W_g: (J, m), b_g: (J,), s: (B, m), g_out: (B, J)
// Using cuBLAS: g_pre = s @ W_g^T  → (B, J).

__global__ void k_add_bias_J(float* x, const float* b, int B, int J) {
    int n = blockIdx.x;
    int j = threadIdx.x;
    if (n < B && j < J) x[n * J + j] += b[j];
}

__global__ void k_sigmoid_inplace(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = sigmoidf_dev(x[i]);
}

// ===== Memory update =====
// Forward: M[b, j, d] = (1 - lambda[j]) * M_prev[b, j, d] + lambda[j] * g[b, j] * z[b, d]
// M: (B, J, m), g: (B, J), z: (B, m), lambda: (J,)
__global__ void k_memory_update_fwd(float* M, const float* g, const float* z,
                                    const float* lambda, int B, int J, int m) {
    int b = blockIdx.x;
    int j = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b < B && j < J && d < m) {
        float lam = lambda[j];
        float gbj = g[b * J + j];
        float prev = M[(b * J + j) * m + d];
        M[(b * J + j) * m + d] = (1.0f - lam) * prev + lam * gbj * z[b * m + d];
    }
}

inline void launch_memory_update_fwd(float* M, const float* g, const float* z,
                                     const float* lambda, int B, int J, int m) {
    dim3 block(128);
    dim3 grid(B, J, (m + 127) / 128);
    k_memory_update_fwd<<<grid, block>>>(M, g, z, lambda, B, J, m);
}

// Backward through memory update:
//   dM_prev[b, j, d] = (1 - lambda[j]) * dM_curr[b, j, d]
//   dg[b, j] += sum_d (lambda[j] * z[b, d] * dM_curr[b, j, d])
//   dz[b, d] += sum_j (lambda[j] * g[b, j] * dM_curr[b, j, d])
//
// We accumulate dz and dg into provided buffers; dM_prev is written.
__global__ void k_memory_update_bwd_M(float* dM_prev, const float* dM_curr,
                                      const float* lambda, int B, int J, int m) {
    int b = blockIdx.x;
    int j = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b < B && j < J && d < m) {
        float lam = lambda[j];
        dM_prev[(b * J + j) * m + d] = (1.0f - lam) * dM_curr[(b * J + j) * m + d];
    }
}

// dg[b, j] += lambda[j] * <z[b, :], dM[b, j, :]>
// Use one block per (b, j) row, reduce within.
__global__ void k_memory_update_bwd_dg(const float* dM, const float* z,
                                       const float* lambda, float* dg,
                                       int B, int J, int m) {
    int b = blockIdx.x;
    int j = blockIdx.y;
    if (b >= B || j >= J) return;
    const float* dMrow = dM + (b * J + j) * m;
    const float* zrow = z + b * m;
    float lam = lambda[j];
    float s = 0.0f;
    for (int d = threadIdx.x; d < m; d += blockDim.x) {
        s += dMrow[d] * zrow[d];
    }
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) sh[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (threadIdx.x < (blockDim.x + 31) / 32) ? sh[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) atomicAdd(&dg[b * J + j], lam * s);
    }
}

// dz[b, d] += sum_j (lambda[j] * g[b, j] * dM[b, j, d])
__global__ void k_memory_update_bwd_dz(const float* dM, const float* g,
                                       const float* lambda, float* dz,
                                       int B, int J, int m) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || d >= m) return;
    float acc = 0.0f;
    for (int j = 0; j < J; ++j) {
        float lam = lambda[j];
        float gbj = g[b * J + j];
        acc += lam * gbj * dM[(b * J + j) * m + d];
    }
    atomicAdd(&dz[b * m + d], acc);
}

inline void launch_memory_update_bwd(float* dM_prev, const float* dM_curr,
                                     const float* z, const float* g,
                                     const float* lambda, float* dz, float* dg,
                                     int B, int J, int m) {
    {
        dim3 block(128);
        dim3 grid(B, J, (m + 127) / 128);
        k_memory_update_bwd_M<<<grid, block>>>(dM_prev, dM_curr, lambda, B, J, m);
    }
    {
        dim3 block(128);
        dim3 grid(B, J);
        k_memory_update_bwd_dg<<<grid, block>>>(dM_curr, z, lambda, dg, B, J, m);
    }
    {
        dim3 block(128);
        dim3 grid(B, (m + 127) / 128);
        k_memory_update_bwd_dz<<<grid, block>>>(dM_curr, g, lambda, dz, B, J, m);
    }
}

// ===== Sigmoid backward through gate (combined with d/dpre_g) =====
// dpre_g = dg * g * (1 - g). Here we want the chain: given d_g_post_sigmoid (called dg),
// and the post-sigmoid value g (stored), compute dpre_g.
__global__ void k_sigmoid_bwd_inplace(float* dg, const float* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        dg[i] = dg[i] * v * (1.0f - v);
    }
}

// ===== Attention readout =====
// q: (B, m), M_T: (B, J, m), output r: (B, m). Also save alpha (B, J) for backward.
// q = W_q @ s_T + b_q is done via standard GEMM + bias_add (not in this file).
//
// scores[b, j] = (q[b, :] · M_T[b, j, :]) / sqrt(m)
// alpha[b, :]  = softmax(scores[b, :])
// r[b, :]      = sum_j alpha[b, j] * M_T[b, j, :]

__global__ void k_att_scores(const float* q, const float* M, float* scores,
                             int B, int J, int m, float inv_sqrt_m) {
    int b = blockIdx.x;
    int j = blockIdx.y;
    if (b >= B || j >= J) return;
    const float* qrow = q + b * m;
    const float* Mrow = M + (b * J + j) * m;
    float s = 0.0f;
    for (int d = threadIdx.x; d < m; d += blockDim.x) s += qrow[d] * Mrow[d];
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) sh[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (threadIdx.x < (blockDim.x + 31) / 32) ? sh[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) scores[b * J + j] = s * inv_sqrt_m;
    }
}

__global__ void k_softmax_J(float* x, int B, int J) {
    int b = blockIdx.x;
    if (b >= B) return;
    float* row = x + b * J;
    // small J (J=4), do everything in thread 0
    if (threadIdx.x == 0) {
        float mx = row[0];
        for (int j = 1; j < J; ++j) if (row[j] > mx) mx = row[j];
        float s = 0.0f;
        for (int j = 0; j < J; ++j) { row[j] = expf(row[j] - mx); s += row[j]; }
        float inv = 1.0f / s;
        for (int j = 0; j < J; ++j) row[j] *= inv;
    }
}

__global__ void k_att_combine_v(const float* alpha, const float* M, float* r,
                                int B, int J, int m) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || d >= m) return;
    float acc = 0.0f;
    for (int j = 0; j < J; ++j) {
        acc += alpha[b * J + j] * M[(b * J + j) * m + d];
    }
    r[b * m + d] = acc;
}

inline void att_readout_fwd(const float* q, const float* M,
                            float* scores, float* alpha, float* r,
                            int B, int J, int m) {
    float inv_sqrt_m = 1.0f / std::sqrt((float)m);
    {
        dim3 block(128);
        dim3 grid(B, J);
        k_att_scores<<<grid, block>>>(q, M, scores, B, J, m, inv_sqrt_m);
    }
    // copy scores to alpha (we softmax alpha in-place)
    CUDA_CHECK(cudaMemcpy(alpha, scores, B * J * sizeof(float), cudaMemcpyDeviceToDevice));
    {
        dim3 block(32);
        dim3 grid(B);
        k_softmax_J<<<grid, block>>>(alpha, B, J);
    }
    {
        dim3 block(128);
        dim3 grid(B, (m + 127) / 128);
        k_att_combine_v<<<grid, block>>>(alpha, M, r, B, J, m);
    }
}

// Backward:
//   dM_T_attn[b, j, :] += alpha[b, j] * dr[b, :]
//   dalpha[b, j] = <dr[b, :], M_T[b, j, :]>
//   dscore[b, j] = sum_k alpha[b, k] * (delta_{j,k} - alpha[b, j]) * dalpha[b, k]
//                = alpha[b, j] * (dalpha[b, j] - sum_k alpha[b, k] * dalpha[b, k])
//   dq[b, :] = sum_j (dscore[b, j] / sqrt(m)) * M_T[b, j, :]
//   dM_T_attn[b, j, :] += (dscore[b, j] / sqrt(m)) * q[b, :]
__global__ void k_att_bwd_dM_dalpha(const float* dr, const float* M, const float* alpha,
                                    float* dM_attn, float* dalpha,
                                    int B, int J, int m) {
    int b = blockIdx.x;
    int j = blockIdx.y;
    if (b >= B || j >= J) return;
    const float* drrow = dr + b * m;
    const float* Mrow = M + (b * J + j) * m;
    float a = alpha[b * J + j];
    float s = 0.0f;
    for (int d = threadIdx.x; d < m; d += blockDim.x) {
        dM_attn[(b * J + j) * m + d] = a * drrow[d];
        s += drrow[d] * Mrow[d];
    }
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) sh[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (threadIdx.x < (blockDim.x + 31) / 32) ? sh[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) dalpha[b * J + j] = s;
    }
}

// dscore[b, j] = alpha[b, j] * (dalpha[b, j] - sum_k alpha[b, k] * dalpha[b, k])
__global__ void k_softmax_J_bwd(const float* alpha, const float* dalpha, float* dscore,
                                int B, int J) {
    int b = blockIdx.x;
    if (b >= B) return;
    if (threadIdx.x == 0) {
        float dot = 0.0f;
        for (int j = 0; j < J; ++j) dot += alpha[b * J + j] * dalpha[b * J + j];
        for (int j = 0; j < J; ++j) {
            float a = alpha[b * J + j];
            dscore[b * J + j] = a * (dalpha[b * J + j] - dot);
        }
    }
}

// dq[b, :] = sum_j (dscore[b, j] / sqrt(m)) * M[b, j, :]
// dM[b, j, :] += (dscore[b, j] / sqrt(m)) * q[b, :]
__global__ void k_att_bwd_dq_dM_scores(const float* dscore, const float* q, const float* M,
                                       float* dq, float* dM_attn,
                                       int B, int J, int m, float inv_sqrt_m) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || d >= m) return;
    float acc = 0.0f;
    for (int j = 0; j < J; ++j) {
        float ds = dscore[b * J + j] * inv_sqrt_m;
        acc += ds * M[(b * J + j) * m + d];
        atomicAdd(&dM_attn[(b * J + j) * m + d], ds * q[b * m + d]);
    }
    dq[b * m + d] = acc;
}

inline void att_readout_bwd(const float* dr, const float* q, const float* M, const float* alpha,
                            float* dq, float* dM_attn,
                            float* scratch_dalpha, float* scratch_dscore,
                            int B, int J, int m) {
    float inv_sqrt_m = 1.0f / std::sqrt((float)m);
    // Step 1: dM_attn = alpha * dr; dalpha = <dr, M>
    CUDA_CHECK(cudaMemset(dM_attn, 0, B * J * m * sizeof(float)));
    {
        dim3 block(128);
        dim3 grid(B, J);
        k_att_bwd_dM_dalpha<<<grid, block>>>(dr, M, alpha, dM_attn, scratch_dalpha, B, J, m);
    }
    // Step 2: dscore = softmax_bwd(dalpha)
    {
        dim3 block(32);
        dim3 grid(B);
        k_softmax_J_bwd<<<grid, block>>>(alpha, scratch_dalpha, scratch_dscore, B, J);
    }
    // Step 3: dq += sum_j (dscore[j]/sqrt(m)) M[j]; dM[j] += (dscore[j]/sqrt(m)) q
    {
        dim3 block(128);
        dim3 grid(B, (m + 127) / 128);
        k_att_bwd_dq_dM_scores<<<grid, block>>>(scratch_dscore, q, M, dq, dM_attn, B, J, m, inv_sqrt_m);
    }
}

// ===== Concat / split feat = [s_T, r] =====
__global__ void k_concat_2(const float* a, const float* b, float* out,
                           int B, int Da, int Db) {
    int n = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    int D = Da + Db;
    if (n >= B || d >= D) return;
    if (d < Da) out[n * D + d] = a[n * Da + d];
    else        out[n * D + d] = b[n * Db + (d - Da)];
}

inline void launch_concat_2(const float* a, const float* b, float* out,
                            int B, int Da, int Db) {
    dim3 block(128);
    dim3 grid(B, (Da + Db + 127) / 128);
    k_concat_2<<<grid, block>>>(a, b, out, B, Da, Db);
}

__global__ void k_split_2(const float* x, float* a, float* b,
                          int B, int Da, int Db) {
    int n = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    int D = Da + Db;
    if (n >= B || d >= D) return;
    if (d < Da) a[n * Da + d] = x[n * D + d];
    else        b[n * Db + (d - Da)] = x[n * D + d];
}

inline void launch_split_2(const float* x, float* a, float* b,
                           int B, int Da, int Db) {
    dim3 block(128);
    dim3 grid(B, (Da + Db + 127) / 128);
    k_split_2<<<grid, block>>>(x, a, b, B, Da, Db);
}

// ===== RNN tanh recurrence step (used by RNN baseline) =====
// s_t = tanh(W_h @ s_{t-1} + W_in @ z_t + b)
// Standard: compute the pre-activation via GEMM, then tanh.

#endif // EALRMN_RECURRENCE_CUH
