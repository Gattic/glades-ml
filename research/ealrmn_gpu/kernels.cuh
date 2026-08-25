// EALRMN Phase-1 GPU prototype — primary kernels
// GEMM wrappers, embedding, softmax, layer norm, cross-entropy, recurrence ops.
#ifndef EALRMN_KERNELS_CUH
#define EALRMN_KERNELS_CUH

#include "common.cuh"

// ===== cuBLAS GEMM (FP32 row-major) =====
//
// We use cuBLAS in column-major form. To compute C = A*B (row-major)
// for A (M,K), B (K,N), C (M,N), we can call:
//   cublasSgemm(N, N, N, M, K, ..., B_d, N, A_d, K, ..., C_d, N)
// because (A*B)^T = B^T * A^T, and column-major B with shape (N,K) is
// row-major B^T, etc.

inline void gemm_rm(cublasHandle_t handle,
                    bool transA, bool transB,
                    int M, int N, int K,
                    float alpha,
                    const float* A, int lda_rm,
                    const float* B, int ldb_rm,
                    float beta,
                    float* C, int ldc_rm) {
    // Treat row-major (M,K) A as column-major (K,M) with leading dim lda_rm.
    // If transA=true (A is K×M row-major), in col-major it's (M,K).
    // To compute C = op(A) * op(B) row-major (M,N) = transpose of col-major (N,M):
    //   col-major:  C^T = op(B)^T * op(A)^T  (sizes N×M)
    //   so call cublasSgemm with:
    //     m = N, n = M, k = K
    //     transA_col = !transB (because we pass B as the first operand in col-major)
    //     transB_col = !transA
    //
    // Actually the cleanest interpretation: row-major Z = X*Y means in col-major,
    // Z^T = Y^T * X^T. With X (M,K) row-major = X^T (K,M) col-major, transpose
    // operation in col-major gives X (M,K). Etc.
    //
    // Reference: just always do this swap.
    cublasOperation_t opA = transA ? CUBLAS_OP_N : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_N : CUBLAS_OP_N;
    (void)opA; (void)opB;
    // Use: cublasSgemm(handle, opB_col, opA_col, N, M, K, ..., B, ldb_col, A, lda_col, ..., C, ldc_col)
    // where ldX_col equals the row-stride of X in row-major (since column-major leading dim of X^T equals row-major leading dim of X).
    cublasOperation_t opA_col = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB_col = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    // For row-major Z (M,N) = X(M,K) * Y(K,N):
    //   Z^T (N,M) col-major = Y^T (N,K) col-major * X^T (K,M) col-major
    // We pass Y as first ("A_cublas"), X as second ("B_cublas"), with op = OP_N giving Y^T col-major access.
    // Sizes: m_cublas=N, n_cublas=M, k_cublas=K.
    CUBLAS_CHECK(cublasSgemm(handle,
        opB_col, opA_col,
        N, M, K,
        &alpha,
        B, ldb_rm,   // col-major leading dim of B^T = row-major leading dim of B
        A, lda_rm,
        &beta,
        C, ldc_rm));
}

// Convenience wrappers.
inline void gemm_nn(cublasHandle_t handle, int M, int N, int K,
                    float alpha, const float* A, const float* B,
                    float beta, float* C) {
    // C(M,N) = alpha * A(M,K) * B(K,N) + beta * C(M,N)
    gemm_rm(handle, false, false, M, N, K, alpha, A, K, B, N, beta, C, N);
}

inline void gemm_nt(cublasHandle_t handle, int M, int N, int K,
                    float alpha, const float* A, const float* B,
                    float beta, float* C) {
    // C(M,N) = alpha * A(M,K) * B(N,K)^T + beta * C(M,N)
    // A is (M,K) row-major, B is (N,K) row-major; we transpose B to use it as (K,N).
    gemm_rm(handle, false, true, M, N, K, alpha, A, K, B, K, beta, C, N);
}

inline void gemm_tn(cublasHandle_t handle, int M, int N, int K,
                    float alpha, const float* A, const float* B,
                    float beta, float* C) {
    // C(M,N) = alpha * A(K,M)^T * B(K,N) + beta * C(M,N)
    gemm_rm(handle, true, false, M, N, K, alpha, A, M, B, N, beta, C, N);
}

// ===== Embedding lookup =====

__global__ void k_embedding_fwd(const int* ids, const float* table,
                                float* out, int N, int D) {
    // ids: [N], table: [V, D], out: [N, D]
    int n = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (n < N && d < D) {
        int id = ids[n];
        out[n * D + d] = table[id * D + d];
    }
}

inline void launch_embedding_fwd(const int* ids, const float* table,
                                 float* out, int N, int D) {
    dim3 block(128);
    dim3 grid(N, (D + 127) / 128);
    k_embedding_fwd<<<grid, block>>>(ids, table, out, N, D);
}

// Backward: scatter-add d_out into d_table at id positions.
__global__ void k_embedding_bwd(const int* ids, const float* d_out,
                                float* d_table, int N, int D) {
    int n = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (n < N && d < D) {
        int id = ids[n];
        atomicAdd(&d_table[id * D + d], d_out[n * D + d]);
    }
}

inline void launch_embedding_bwd(const int* ids, const float* d_out,
                                 float* d_table, int N, int D) {
    dim3 block(128);
    dim3 grid(N, (D + 127) / 128);
    k_embedding_bwd<<<grid, block>>>(ids, d_out, d_table, N, D);
}

// ===== Softmax + cross-entropy (combined) =====

// Forward: per-row softmax. logits (B, C), labels (B,) → loss (scalar mean).
// Also write softmax probs into a buffer for backward.
__global__ void k_softmax_ce_fwd(const float* logits, const int* labels,
                                 float* probs, float* losses,
                                 int B, int C) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float* L = logits + b * C;
    float* P = probs + b * C;
    // Compute max
    float m = -INFINITY;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        m = fmaxf(m, L[c]);
    }
    __shared__ float sm[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    // warp reduce
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) sm[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (threadIdx.x < (blockDim.x + 31) / 32) ? sm[lane] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) sm[0] = m;
    }
    __syncthreads();
    float gmax = sm[0];
    // Compute exp + sum
    float s = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float e = expf(L[c] - gmax);
        P[c] = e;
        s += e;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) sm[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (threadIdx.x < (blockDim.x + 31) / 32) ? sm[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) sm[0] = s;
    }
    __syncthreads();
    float gs = sm[0];
    // Normalize + cross-entropy
    int y = labels[b];
    float inv = 1.0f / gs;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        P[c] *= inv;
    }
    if (threadIdx.x == 0) {
        // y is valid in [0, C). loss = -log(P[y]) = -((L[y]-gmax) - log(gs))
        float lpy = (L[y] - gmax) - logf(gs);
        losses[b] = -lpy;
    }
}

inline void launch_softmax_ce_fwd(const float* logits, const int* labels,
                                  float* probs, float* losses, int B, int C) {
    dim3 block(128);
    dim3 grid(B);
    k_softmax_ce_fwd<<<grid, block>>>(logits, labels, probs, losses, B, C);
}

__global__ void k_softmax_ce_bwd(const float* probs, const int* labels,
                                 float* d_logits, int B, int C, float scale) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float* P = probs + b * C;
    float* dL = d_logits + b * C;
    int y = labels[b];
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = P[c];
        if (c == y) v -= 1.0f;
        dL[c] = scale * v;
    }
}

inline void launch_softmax_ce_bwd(const float* probs, const int* labels,
                                  float* d_logits, int B, int C, float scale) {
    dim3 block(128);
    dim3 grid(B);
    k_softmax_ce_bwd<<<grid, block>>>(probs, labels, d_logits, B, C, scale);
}

// Argmax (top-1) for accuracy computation.
__global__ void k_argmax(const float* logits, int* preds, int B, int C) {
    int b = blockIdx.x;
    if (b >= B) return;
    const float* L = logits + b * C;
    int best = 0;
    float bv = L[0];
    for (int c = 1; c < C; ++c) {
        float v = L[c];
        if (v > bv) { bv = v; best = c; }
    }
    if (threadIdx.x == 0) preds[b] = best;
}

inline void launch_argmax(const float* logits, int* preds, int B, int C) {
    dim3 block(32);
    dim3 grid(B);
    k_argmax<<<grid, block>>>(logits, preds, B, C);
}

// ===== Layer norm =====
// For Transformer baseline.

__global__ void k_layer_norm_fwd(const float* x, const float* gamma, const float* beta,
                                 float* y, float* mean_out, float* rstd_out,
                                 int N, int D, float eps) {
    int n = blockIdx.x;
    if (n >= N) return;
    const float* X = x + n * D;
    float* Y = y + n * D;
    // mean
    float ms = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) ms += X[d];
    __shared__ float ssum[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) ms += __shfl_xor_sync(0xffffffff, ms, o);
    if (lane == 0) ssum[warp] = ms;
    __syncthreads();
    if (warp == 0) {
        ms = (threadIdx.x < (blockDim.x + 31) / 32) ? ssum[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) ms += __shfl_xor_sync(0xffffffff, ms, o);
        if (lane == 0) ssum[0] = ms;
    }
    __syncthreads();
    float mean = ssum[0] / (float)D;
    // var
    float vs = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = X[d] - mean;
        vs += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) vs += __shfl_xor_sync(0xffffffff, vs, o);
    if (lane == 0) ssum[warp] = vs;
    __syncthreads();
    if (warp == 0) {
        vs = (threadIdx.x < (blockDim.x + 31) / 32) ? ssum[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) vs += __shfl_xor_sync(0xffffffff, vs, o);
        if (lane == 0) ssum[0] = vs;
    }
    __syncthreads();
    float rstd = rsqrtf(ssum[0] / (float)D + eps);
    if (threadIdx.x == 0) {
        if (mean_out) mean_out[n] = mean;
        if (rstd_out) rstd_out[n] = rstd;
    }
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xh = (X[d] - mean) * rstd;
        Y[d] = xh * gamma[d] + beta[d];
    }
}

inline void launch_layer_norm_fwd(const float* x, const float* gamma, const float* beta,
                                  float* y, float* mean, float* rstd,
                                  int N, int D, float eps = 1e-5f) {
    dim3 block(128);
    dim3 grid(N);
    k_layer_norm_fwd<<<grid, block>>>(x, gamma, beta, y, mean, rstd, N, D, eps);
}

__global__ void k_layer_norm_bwd(const float* x, const float* dy, const float* gamma,
                                 const float* mean, const float* rstd,
                                 float* dx, float* dgamma, float* dbeta,
                                 int N, int D) {
    int n = blockIdx.x;
    if (n >= N) return;
    const float* X = x + n * D;
    const float* DY = dy + n * D;
    float* DX = dx + n * D;
    float m = mean[n];
    float r = rstd[n];
    // sum_dy_g and sum_dy_g_xh
    float s1 = 0.0f, s2 = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xh = (X[d] - m) * r;
        float dyg = DY[d] * gamma[d];
        s1 += dyg;
        s2 += dyg * xh;
    }
    __shared__ float sh1[32], sh2[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        s1 += __shfl_xor_sync(0xffffffff, s1, o);
        s2 += __shfl_xor_sync(0xffffffff, s2, o);
    }
    if (lane == 0) { sh1[warp] = s1; sh2[warp] = s2; }
    __syncthreads();
    if (warp == 0) {
        s1 = (threadIdx.x < (blockDim.x + 31) / 32) ? sh1[lane] : 0.0f;
        s2 = (threadIdx.x < (blockDim.x + 31) / 32) ? sh2[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            s1 += __shfl_xor_sync(0xffffffff, s1, o);
            s2 += __shfl_xor_sync(0xffffffff, s2, o);
        }
        if (lane == 0) { sh1[0] = s1; sh2[0] = s2; }
    }
    __syncthreads();
    float gs1 = sh1[0], gs2 = sh2[0];
    float invD = 1.0f / (float)D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xh = (X[d] - m) * r;
        float dyg = DY[d] * gamma[d];
        // dx = (dyg - mean_dyg - xh * mean_dyg_xh) * rstd
        DX[d] = (dyg - gs1 * invD - xh * gs2 * invD) * r;
        atomicAdd(&dgamma[d], DY[d] * xh);
        atomicAdd(&dbeta[d], DY[d]);
    }
}

inline void launch_layer_norm_bwd(const float* x, const float* dy, const float* gamma,
                                  const float* mean, const float* rstd,
                                  float* dx, float* dgamma, float* dbeta,
                                  int N, int D) {
    dim3 block(128);
    dim3 grid(N);
    k_layer_norm_bwd<<<grid, block>>>(x, dy, gamma, mean, rstd, dx, dgamma, dbeta, N, D);
}

// ===== Vector ops needed for recurrence and memory =====

// Bias-add: y[n, d] += bias[d]
__global__ void k_bias_add(float* x, const float* bias, int N, int D) {
    int n = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (n < N && d < D) {
        x[n * D + d] += bias[d];
    }
}

inline void launch_bias_add(float* x, const float* bias, int N, int D) {
    dim3 block(128);
    dim3 grid(N, (D + 127) / 128);
    k_bias_add<<<grid, block>>>(x, bias, N, D);
}

// Bias backward: db[d] = sum_n dx[n, d]
__global__ void k_bias_bwd(const float* dy, float* db, int N, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float s = 0.0f;
    for (int n = 0; n < N; ++n) s += dy[n * D + d];
    atomicAdd(&db[d], s);
}

inline void launch_bias_bwd(const float* dy, float* db, int N, int D) {
    int block = 128;
    int grid = (D + block - 1) / block;
    k_bias_bwd<<<grid, block>>>(dy, db, N, D);
}

#endif // EALRMN_KERNELS_CUH
