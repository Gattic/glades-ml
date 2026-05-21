// EALRMN Phase-1 GPU prototype — Multi-head Transformer baseline
// Pre-norm, causal mask, sinusoidal positional encoding, GELU MLP.
// 1L or 2L variants. Full O(T^2) attention (suitable up to T=4096).
#ifndef EALRMN_MODEL_TRANSFORMER_CUH
#define EALRMN_MODEL_TRANSFORMER_CUH

#include "common.cuh"
#include "kernels.cuh"
#include "recurrence_kernels.cuh"
#include "model_ealrmn.cuh"  // for embedding_fwd/bwd_col helpers

// ===== Strided batched GEMM wrappers (row-major) =====

inline void gemm_strided_batched_nn(cublasHandle_t handle,
                                    int M, int N, int K,
                                    float alpha, const float* A, int strideA,
                                    const float* B, int strideB,
                                    float beta, float* C, int strideC,
                                    int batchCount) {
    // C(M,N) = A(M,K) * B(K,N), batchCount times.
    // Row-major → col-major: cublasSgemmStridedBatched with swapped operands.
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N, strideB,
        A, K, strideA,
        &beta,
        C, N, strideC,
        batchCount));
}

inline void gemm_strided_batched_nt(cublasHandle_t handle,
                                    int M, int N, int K,
                                    float alpha, const float* A, int strideA,
                                    const float* B, int strideB,
                                    float beta, float* C, int strideC,
                                    int batchCount) {
    // C(M,N) = A(M,K) * B(N,K)^T
    // Equiv col-major: C^T(N,M) = (B^T)^T_col * A^T_col = B_col * A_col_T
    // → in col-major: opB=N (since B in col-major is K×N from row-major (N,K); we want N×K so transpose)
    // Wait let me redo. B is (N, K) row-major. In col-major view, B is (K, N). We want to use B^T_rm (which is K×N row-major) as multiplier; this is the same as B viewed in col-major, no transpose.
    // Standard derivation:
    //   row-major op: C = A * B^T_rm.  Sizes (M,N) = (M,K) * (N,K)^T.
    //   In col-major: C_T = (B^T_rm)_T * A_T   where _T denotes col-major-view-transpose
    //   = B_T * A_T (since (B^T)_T = B for col-major)
    // hmm let me just trust the pattern.
    // Without batch:  gemm_nt uses cublasSgemm(handle, OP_T, OP_N, N, M, K, ..., B, K, A, K, ..., C, N).
    // So: opB_col = OP_T, opA_col = OP_N, ldB = K, ldA = K.
    // Batched version follows the same.
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, K, strideB,
        A, K, strideA,
        &beta,
        C, N, strideC,
        batchCount));
}

inline void gemm_strided_batched_tn(cublasHandle_t handle,
                                    int M, int N, int K,
                                    float alpha, const float* A, int strideA,
                                    const float* B, int strideB,
                                    float beta, float* C, int strideC,
                                    int batchCount) {
    // C(M,N) = A(K,M)^T * B(K,N)
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        B, N, strideB,
        A, M, strideA,
        &beta,
        C, N, strideC,
        batchCount));
}

// ===== Reshape helpers (split/merge heads) =====
// X: (B, T, m) → (B, H, T, d_h) where m = H * d_h
// We do this via a permutation kernel. Memory layout:
//   X[b, t, h, d] (in (B,T,H,d_h) layout) ↔ X[b, h, t, d] in (B,H,T,d_h) layout

__global__ void k_split_heads_fwd(const float* x_in, float* x_out,
                                  int B, int T, int H, int d_h) {
    // in: (B, T, m) where m = H*d_h, indexed [b, t, h, d]
    // out: (B, H, T, d_h), indexed [b, h, t, d]
    int b = blockIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;
    int d = threadIdx.x;
    if (b < B && h < H && t < T && d < d_h) {
        x_out[((b * H + h) * T + t) * d_h + d] = x_in[((b * T + t) * H + h) * d_h + d];
    }
}

__global__ void k_merge_heads_fwd(const float* x_in, float* x_out,
                                  int B, int T, int H, int d_h) {
    // in: (B, H, T, d_h), out: (B, T, m)
    int b = blockIdx.x;
    int h = blockIdx.y;
    int t = blockIdx.z;
    int d = threadIdx.x;
    if (b < B && h < H && t < T && d < d_h) {
        x_out[((b * T + t) * H + h) * d_h + d] = x_in[((b * H + h) * T + t) * d_h + d];
    }
}

inline void launch_split_heads(const float* x_in, float* x_out, int B, int T, int H, int d_h) {
    dim3 block(d_h);  // assumed d_h <= 1024
    dim3 grid(B, H, T);
    k_split_heads_fwd<<<grid, block>>>(x_in, x_out, B, T, H, d_h);
}

inline void launch_merge_heads(const float* x_in, float* x_out, int B, int T, int H, int d_h) {
    dim3 block(d_h);
    dim3 grid(B, H, T);
    k_merge_heads_fwd<<<grid, block>>>(x_in, x_out, B, T, H, d_h);
}

// ===== Causal masked softmax =====
// in: scores (B*H, T, T), each row gets softmax with causal mask: scores[i, j > i] -> -inf

__global__ void k_softmax_causal_fwd(float* scores, int BH, int T) {
    int bh = blockIdx.x;
    int i = blockIdx.y;
    if (bh >= BH || i >= T) return;
    float* row = scores + ((int64_t)bh * T + i) * T;
    int last = i;  // attend to [0, i]
    // max
    float mx = -INFINITY;
    for (int j = threadIdx.x; j <= last; j += blockDim.x) {
        mx = fmaxf(mx, row[j]);
    }
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    if (lane == 0) sh[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        mx = (threadIdx.x < (blockDim.x + 31) / 32) ? sh[lane] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        if (lane == 0) sh[0] = mx;
    }
    __syncthreads();
    float gmax = sh[0];
    // sum
    float s = 0.0f;
    for (int j = threadIdx.x; j <= last; j += blockDim.x) {
        float e = expf(row[j] - gmax);
        row[j] = e;
        s += e;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) sh[warp] = s;
    __syncthreads();
    if (warp == 0) {
        s = (threadIdx.x < (blockDim.x + 31) / 32) ? sh[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) sh[0] = s;
    }
    __syncthreads();
    float gs = sh[0];
    float inv = 1.0f / gs;
    for (int j = threadIdx.x; j <= last; j += blockDim.x) {
        row[j] *= inv;
    }
    // zero out the future
    for (int j = threadIdx.x + last + 1; j < T; j += blockDim.x) {
        row[j] = 0.0f;
    }
}

inline void launch_softmax_causal_fwd(float* scores, int BH, int T) {
    dim3 block(128);
    dim3 grid(BH, T);
    k_softmax_causal_fwd<<<grid, block>>>(scores, BH, T);
}

// Backward: given P (post-softmax) and dP, compute dS = P * (dP - rowsum(P * dP))
// For causal, future entries are 0 in P, and we want dS = 0 there too.
__global__ void k_softmax_causal_bwd(const float* P, const float* dP, float* dS,
                                     int BH, int T) {
    int bh = blockIdx.x;
    int i = blockIdx.y;
    if (bh >= BH || i >= T) return;
    int64_t base = ((int64_t)bh * T + i) * T;
    const float* Prow = P + base;
    const float* dProw = dP + base;
    float* dSrow = dS + base;
    int last = i;
    float sumPdp = 0.0f;
    for (int j = threadIdx.x; j <= last; j += blockDim.x) {
        sumPdp += Prow[j] * dProw[j];
    }
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) sumPdp += __shfl_xor_sync(0xffffffff, sumPdp, o);
    if (lane == 0) sh[warp] = sumPdp;
    __syncthreads();
    if (warp == 0) {
        sumPdp = (threadIdx.x < (blockDim.x + 31) / 32) ? sh[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) sumPdp += __shfl_xor_sync(0xffffffff, sumPdp, o);
        if (lane == 0) sh[0] = sumPdp;
    }
    __syncthreads();
    float g = sh[0];
    for (int j = threadIdx.x; j <= last; j += blockDim.x) {
        dSrow[j] = Prow[j] * (dProw[j] - g);
    }
    for (int j = threadIdx.x + last + 1; j < T; j += blockDim.x) {
        dSrow[j] = 0.0f;
    }
}

inline void launch_softmax_causal_bwd(const float* P, const float* dP, float* dS,
                                      int BH, int T) {
    dim3 block(128);
    dim3 grid(BH, T);
    k_softmax_causal_bwd<<<grid, block>>>(P, dP, dS, BH, T);
}

// ===== Sinusoidal positional encoding =====
// pe[t, d]: sin(t / 10000^(2k/m)) for even d=2k, cos(...) for odd d=2k+1
__global__ void k_sin_pe(float* pe, int T, int m) {
    int t = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (t < T && d < m) {
        int k = d / 2;
        float exp = (2.0f * (float)k) / (float)m;
        float inv = powf(10000.0f, exp);
        float arg = (float)t / inv;
        pe[t * m + d] = (d % 2 == 0) ? sinf(arg) : cosf(arg);
    }
}

inline void make_sinusoidal_pe(Tensor& pe, int T, int m) {
    dim3 block(128);
    dim3 grid(T, (m + 127) / 128);
    k_sin_pe<<<grid, block>>>(pe.d, T, m);
}

// Add PE in place: x[b, t, d] += pe[t, d]
__global__ void k_add_pe(float* x, const float* pe, int B, int T, int m) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b < B && t < T && d < m) {
        x[(b * T + t) * m + d] += pe[t * m + d];
    }
}

inline void launch_add_pe(float* x, const float* pe, int B, int T, int m) {
    dim3 block(128);
    dim3 grid(B, T, (m + 127) / 128);
    k_add_pe<<<grid, block>>>(x, pe, B, T, m);
}

// ===== Helpers =====
__global__ void k_split_qkv(const float* qkv, float* Q, float* K, float* V,
                            int B, int T, int m) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b < B && t < T && d < m) {
        int64_t idx_in = ((int64_t)b * T + t) * 3 * m + d;
        int64_t idx_out = ((int64_t)b * T + t) * m + d;
        Q[idx_out] = qkv[idx_in];
        K[idx_out] = qkv[idx_in + m];
        V[idx_out] = qkv[idx_in + 2 * m];
    }
}

__global__ void k_concat_qkv(const float* dQ, const float* dK, const float* dV,
                             float* dQKV, int B, int T, int m) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b < B && t < T && d < m) {
        int64_t idx_in = ((int64_t)b * T + t) * m + d;
        int64_t idx_out = ((int64_t)b * T + t) * 3 * m + d;
        dQKV[idx_out] = dQ[idx_in];
        dQKV[idx_out + m] = dK[idx_in];
        dQKV[idx_out + 2 * m] = dV[idx_in];
    }
}

// ===== Single attention block =====

struct TransformerLayer {
    cublasHandle_t cublas;
    int m = 0, H = 0, d_h = 0;
    Tensor ln1_g, ln1_b;       // (m,)
    Tensor Wqkv;               // (m, 3m)
    Tensor bqkv;               // (3m,)
    Tensor Wo;                 // (m, m)
    Tensor bo;                 // (m,)
    Tensor ln2_g, ln2_b;       // (m,)
    Tensor W1, b1;             // (m, 4m), (4m,)
    Tensor W2, b2;             // (4m, m), (m,)

    Tensor dln1_g, dln1_b, dWqkv, dbqkv, dWo, dbo;
    Tensor dln2_g, dln2_b, dW1, db1, dW2, db2;

    Tensor mln1_g, mln1_b, mWqkv, mbqkv, mWo, mbo;
    Tensor mln2_g, mln2_b, mW1, mb1, mW2, mb2;
    Tensor vln1_g, vln1_b, vWqkv, vbqkv, vWo, vbo;
    Tensor vln2_g, vln2_b, vW1, vb1, vW2, vb2;

    std::vector<Tensor*> params() {
        return { &ln1_g, &ln1_b, &Wqkv, &bqkv, &Wo, &bo,
                 &ln2_g, &ln2_b, &W1, &b1, &W2, &b2 };
    }
    std::vector<Tensor*> grads() {
        return { &dln1_g, &dln1_b, &dWqkv, &dbqkv, &dWo, &dbo,
                 &dln2_g, &dln2_b, &dW1, &db1, &dW2, &db2 };
    }
    std::vector<Tensor*> ms() {
        return { &mln1_g, &mln1_b, &mWqkv, &mbqkv, &mWo, &mbo,
                 &mln2_g, &mln2_b, &mW1, &mb1, &mW2, &mb2 };
    }
    std::vector<Tensor*> vs() {
        return { &vln1_g, &vln1_b, &vWqkv, &vbqkv, &vWo, &vbo,
                 &vln2_g, &vln2_b, &vW1, &vb1, &vW2, &vb2 };
    }

    void init(cublasHandle_t handle, int m_, int H_, HostRng& rng) {
        cublas = handle; m = m_; H = H_; d_h = m / H;
        ln1_g = make_tensor({m}); ln1_b = make_tensor({m});
        Wqkv = make_tensor({m, 3 * m}); bqkv = make_tensor({3 * m});
        Wo = make_tensor({m, m}); bo = make_tensor({m});
        ln2_g = make_tensor({m}); ln2_b = make_tensor({m});
        W1 = make_tensor({m, 4 * m}); b1 = make_tensor({4 * m});
        W2 = make_tensor({4 * m, m}); b2 = make_tensor({m});

        // grads
        dln1_g = make_tensor({m}); dln1_b = make_tensor({m});
        dWqkv = make_tensor({m, 3 * m}); dbqkv = make_tensor({3 * m});
        dWo = make_tensor({m, m}); dbo = make_tensor({m});
        dln2_g = make_tensor({m}); dln2_b = make_tensor({m});
        dW1 = make_tensor({m, 4 * m}); db1 = make_tensor({4 * m});
        dW2 = make_tensor({4 * m, m}); db2 = make_tensor({m});

        // Adam
        mln1_g = make_tensor({m}); mln1_g.zero_();
        mln1_b = make_tensor({m}); mln1_b.zero_();
        mWqkv  = make_tensor({m, 3 * m}); mWqkv.zero_();
        mbqkv  = make_tensor({3 * m}); mbqkv.zero_();
        mWo    = make_tensor({m, m}); mWo.zero_();
        mbo    = make_tensor({m}); mbo.zero_();
        mln2_g = make_tensor({m}); mln2_g.zero_();
        mln2_b = make_tensor({m}); mln2_b.zero_();
        mW1    = make_tensor({m, 4 * m}); mW1.zero_();
        mb1    = make_tensor({4 * m}); mb1.zero_();
        mW2    = make_tensor({4 * m, m}); mW2.zero_();
        mb2    = make_tensor({m}); mb2.zero_();
        vln1_g = make_tensor({m}); vln1_g.zero_();
        vln1_b = make_tensor({m}); vln1_b.zero_();
        vWqkv  = make_tensor({m, 3 * m}); vWqkv.zero_();
        vbqkv  = make_tensor({3 * m}); vbqkv.zero_();
        vWo    = make_tensor({m, m}); vWo.zero_();
        vbo    = make_tensor({m}); vbo.zero_();
        vln2_g = make_tensor({m}); vln2_g.zero_();
        vln2_b = make_tensor({m}); vln2_b.zero_();
        vW1    = make_tensor({m, 4 * m}); vW1.zero_();
        vb1    = make_tensor({4 * m}); vb1.zero_();
        vW2    = make_tensor({4 * m, m}); vW2.zero_();
        vb2    = make_tensor({m}); vb2.zero_();

        // Initialize
        std::vector<float> ones(m, 1.0f);
        copy_h2d(ln1_g, ones); ln1_b.zero_();
        copy_h2d(ln2_g, ones); ln2_b.zero_();

        init_xavier_uniform(Wqkv, m, 3 * m, rng); bqkv.zero_();
        init_xavier_uniform(Wo, m, m, rng); bo.zero_();
        init_xavier_uniform(W1, m, 4 * m, rng); b1.zero_();
        init_xavier_uniform(W2, 4 * m, m, rng); b2.zero_();
    }

    void free_all() {
        for (Tensor* t : { &ln1_g, &ln1_b, &Wqkv, &bqkv, &Wo, &bo,
                          &ln2_g, &ln2_b, &W1, &b1, &W2, &b2,
                          &dln1_g, &dln1_b, &dWqkv, &dbqkv, &dWo, &dbo,
                          &dln2_g, &dln2_b, &dW1, &db1, &dW2, &db2,
                          &mln1_g, &mln1_b, &mWqkv, &mbqkv, &mWo, &mbo,
                          &mln2_g, &mln2_b, &mW1, &mb1, &mW2, &mb2,
                          &vln1_g, &vln1_b, &vWqkv, &vbqkv, &vWo, &vbo,
                          &vln2_g, &vln2_b, &vW1, &vb1, &vW2, &vb2 }) {
            free_tensor(*t);
        }
    }

    void zero_grads() { for (Tensor* g : grads()) g->zero_(); }
};

struct TransformerLayerCache {
    int B = 0, T = 0, m = 0, H = 0;
    Tensor x_in;        // (B, T, m) input to layer
    Tensor x_ln1;       // (B, T, m) output of LN1
    Tensor mean1, rstd1; // (B*T,)
    Tensor qkv;         // (B, T, 3m)
    Tensor Q, K, V;     // (B, H, T, d_h)
    Tensor S;           // (B, H, T, T) attention probs (post-softmax)
    Tensor O_heads;     // (B, H, T, d_h)
    Tensor O;           // (B, T, m) merged
    Tensor Y_attn;      // (B, T, m) = O @ Wo + bo
    Tensor x_mid;       // (B, T, m) = x_in + Y_attn (residual after attention)
    Tensor x_ln2;       // (B, T, m) LN2 output
    Tensor mean2, rstd2;
    Tensor h1;          // (B, T, 4m) after first MLP linear (pre-GELU)
    Tensor h1_act;      // (B, T, 4m) after GELU
    Tensor h2;          // (B, T, m) after second MLP linear
    Tensor x_out;       // (B, T, m) final residual

    void alloc(int B_, int T_, int m_, int H_) {
        B = B_; T = T_; m = m_; H = H_;
        int d_h = m / H;
        x_in    = make_tensor({B, T, m});
        x_ln1   = make_tensor({B, T, m});
        mean1   = make_tensor({B * T});
        rstd1   = make_tensor({B * T});
        qkv     = make_tensor({B, T, 3 * m});
        Q       = make_tensor({B, H, T, d_h});
        K       = make_tensor({B, H, T, d_h});
        V       = make_tensor({B, H, T, d_h});
        S       = make_tensor({B, H, T, T});
        O_heads = make_tensor({B, H, T, d_h});
        O       = make_tensor({B, T, m});
        Y_attn  = make_tensor({B, T, m});
        x_mid   = make_tensor({B, T, m});
        x_ln2   = make_tensor({B, T, m});
        mean2   = make_tensor({B * T});
        rstd2   = make_tensor({B * T});
        h1      = make_tensor({B, T, 4 * m});
        h1_act  = make_tensor({B, T, 4 * m});
        h2      = make_tensor({B, T, m});
        x_out   = make_tensor({B, T, m});
    }
    void free() {
        for (Tensor* t : { &x_in, &x_ln1, &mean1, &rstd1, &qkv,
                          &Q, &K, &V, &S, &O_heads, &O, &Y_attn,
                          &x_mid, &x_ln2, &mean2, &rstd2,
                          &h1, &h1_act, &h2, &x_out }) free_tensor(*t);
    }
};

// Forward: a single transformer layer.
inline void transformer_layer_forward(TransformerLayer& L, const float* x_in_d,
                                      TransformerLayerCache& cache) {
    int B = cache.B, T = cache.T, m = cache.m, H = cache.H;
    int d_h = m / H;
    int N = B * T;

    // Copy input
    CUDA_CHECK(cudaMemcpy(cache.x_in.d, x_in_d, B * T * m * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // LN1: x_ln1 = LN(x_in)
    launch_layer_norm_fwd(cache.x_in.d, L.ln1_g.d, L.ln1_b.d,
                          cache.x_ln1.d, cache.mean1.d, cache.rstd1.d, N, m);

    // qkv = x_ln1 @ Wqkv + bqkv  (B*T, 3m) = (B*T, m) @ (m, 3m)
    gemm_nn(L.cublas, N, 3 * m, m, 1.0f, cache.x_ln1.d, L.Wqkv.d, 0.0f, cache.qkv.d);
    launch_bias_add(cache.qkv.d, L.bqkv.d, N, 3 * m);

    // Split QKV into Q, K, V
    {
        dim3 block(128);
        dim3 grid(B, T, (m + 127) / 128);
        // We need temporary (B, T, m) per Q/K/V before splitting heads.
        // Use cache.Q.d, cache.K.d, cache.V.d as scratch (they will get overwritten by split_heads).
        // Actually we need separate scratch. Let me allocate.
    }
    Tensor Q_flat = make_tensor({B, T, m});
    Tensor K_flat = make_tensor({B, T, m});
    Tensor V_flat = make_tensor({B, T, m});
    {
        dim3 block(128);
        dim3 grid(B, T, (m + 127) / 128);
        k_split_qkv<<<grid, block>>>(cache.qkv.d, Q_flat.d, K_flat.d, V_flat.d, B, T, m);
    }
    launch_split_heads(Q_flat.d, cache.Q.d, B, T, H, d_h);
    launch_split_heads(K_flat.d, cache.K.d, B, T, H, d_h);
    launch_split_heads(V_flat.d, cache.V.d, B, T, H, d_h);
    free_tensor(Q_flat); free_tensor(K_flat); free_tensor(V_flat);

    // S = Q @ K^T / sqrt(d_h)  → (B*H, T, T)
    float inv_sqrt = 1.0f / std::sqrt((float)d_h);
    gemm_strided_batched_nt(L.cublas, T, T, d_h, inv_sqrt,
                            cache.Q.d, T * d_h,
                            cache.K.d, T * d_h,
                            0.0f, cache.S.d, T * T, B * H);

    // Causal softmax
    launch_softmax_causal_fwd(cache.S.d, B * H, T);

    // O_heads = S @ V  → (B*H, T, d_h)
    gemm_strided_batched_nn(L.cublas, T, d_h, T, 1.0f,
                            cache.S.d, T * T,
                            cache.V.d, T * d_h,
                            0.0f, cache.O_heads.d, T * d_h, B * H);

    // Merge heads → (B, T, m)
    launch_merge_heads(cache.O_heads.d, cache.O.d, B, T, H, d_h);

    // Y_attn = O @ Wo + bo
    gemm_nn(L.cublas, N, m, m, 1.0f, cache.O.d, L.Wo.d, 0.0f, cache.Y_attn.d);
    launch_bias_add(cache.Y_attn.d, L.bo.d, N, m);

    // x_mid = x_in + Y_attn
    CUDA_CHECK(cudaMemcpy(cache.x_mid.d, cache.x_in.d, B * T * m * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    {
        int n = B * T * m;
        int block = 256;
        int grid = (n + block - 1) / block;
        k_add_inplace<<<grid, block>>>(cache.x_mid.d, cache.Y_attn.d, n);
    }

    // LN2
    launch_layer_norm_fwd(cache.x_mid.d, L.ln2_g.d, L.ln2_b.d,
                          cache.x_ln2.d, cache.mean2.d, cache.rstd2.d, N, m);

    // h1 = x_ln2 @ W1 + b1  (N, 4m)
    gemm_nn(L.cublas, N, 4 * m, m, 1.0f, cache.x_ln2.d, L.W1.d, 0.0f, cache.h1.d);
    launch_bias_add(cache.h1.d, L.b1.d, N, 4 * m);

    // GELU
    {
        int n = N * 4 * m;
        int block = 256;
        int grid = (n + block - 1) / block;
        k_gelu_fwd<<<grid, block>>>(cache.h1.d, cache.h1_act.d, n);
    }

    // h2 = h1_act @ W2 + b2
    gemm_nn(L.cublas, N, m, 4 * m, 1.0f, cache.h1_act.d, L.W2.d, 0.0f, cache.h2.d);
    launch_bias_add(cache.h2.d, L.b2.d, N, m);

    // x_out = x_mid + h2
    CUDA_CHECK(cudaMemcpy(cache.x_out.d, cache.x_mid.d, B * T * m * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    {
        int n = B * T * m;
        int block = 256;
        int grid = (n + block - 1) / block;
        k_add_inplace<<<grid, block>>>(cache.x_out.d, cache.h2.d, n);
    }
}

// Backward: a single transformer layer.
// dx_out -> dx_in (returned as dx_in_d), accumulating into L's grad tensors.
inline void transformer_layer_backward(TransformerLayer& L, const float* dx_out_d,
                                       TransformerLayerCache& cache, float* dx_in_d) {
    int B = cache.B, T = cache.T, m = cache.m, H = cache.H;
    int d_h = m / H;
    int N = B * T;

    // x_out = x_mid + h2  → dh2 = dx_out; dx_mid_a = dx_out
    Tensor dh2 = make_tensor({B, T, m});
    CUDA_CHECK(cudaMemcpy(dh2.d, dx_out_d, B * T * m * sizeof(float), cudaMemcpyDeviceToDevice));
    Tensor dx_mid = make_tensor({B, T, m});
    CUDA_CHECK(cudaMemcpy(dx_mid.d, dx_out_d, B * T * m * sizeof(float), cudaMemcpyDeviceToDevice));

    // h2 = h1_act @ W2 + b2
    // dW2 += h1_act^T @ dh2; db2 += sum dh2; dh1_act = dh2 @ W2^T
    gemm_tn(L.cublas, 4 * m, m, N, 1.0f, cache.h1_act.d, dh2.d, 1.0f, L.dW2.d);
    launch_bias_bwd(dh2.d, L.db2.d, N, m);
    Tensor dh1_act = make_tensor({B, T, 4 * m});
    gemm_nt(L.cublas, N, 4 * m, m, 1.0f, dh2.d, L.W2.d, 0.0f, dh1_act.d);

    // GELU backward
    Tensor dh1 = make_tensor({B, T, 4 * m});
    {
        int n = N * 4 * m;
        int block = 256;
        int grid = (n + block - 1) / block;
        k_gelu_bwd<<<grid, block>>>(cache.h1.d, dh1_act.d, dh1.d, n);
    }
    free_tensor(dh1_act);

    // h1 = x_ln2 @ W1 + b1
    gemm_tn(L.cublas, m, 4 * m, N, 1.0f, cache.x_ln2.d, dh1.d, 1.0f, L.dW1.d);
    launch_bias_bwd(dh1.d, L.db1.d, N, 4 * m);
    Tensor dx_ln2 = make_tensor({B, T, m});
    gemm_nt(L.cublas, N, m, 4 * m, 1.0f, dh1.d, L.W1.d, 0.0f, dx_ln2.d);
    free_tensor(dh1);

    // LN2 backward
    Tensor dx_mid_from_ln = make_tensor({B, T, m});
    dx_mid_from_ln.zero_();
    launch_layer_norm_bwd(cache.x_mid.d, dx_ln2.d, L.ln2_g.d,
                          cache.mean2.d, cache.rstd2.d,
                          dx_mid_from_ln.d, L.dln2_g.d, L.dln2_b.d, N, m);
    free_tensor(dx_ln2);
    // dx_mid += dx_mid_from_ln
    {
        int n = B * T * m;
        int block = 256;
        int grid = (n + block - 1) / block;
        k_add_inplace<<<grid, block>>>(dx_mid.d, dx_mid_from_ln.d, n);
    }
    free_tensor(dx_mid_from_ln);

    // x_mid = x_in + Y_attn  → dY_attn = dx_mid; dx_in = dx_mid (accumulate later)
    Tensor dY_attn = make_tensor({B, T, m});
    CUDA_CHECK(cudaMemcpy(dY_attn.d, dx_mid.d, B * T * m * sizeof(float), cudaMemcpyDeviceToDevice));
    // dx_in_d will receive dx_mid as starting accumulator
    CUDA_CHECK(cudaMemcpy(dx_in_d, dx_mid.d, B * T * m * sizeof(float), cudaMemcpyDeviceToDevice));
    free_tensor(dx_mid);

    // Y_attn = O @ Wo + bo
    gemm_tn(L.cublas, m, m, N, 1.0f, cache.O.d, dY_attn.d, 1.0f, L.dWo.d);
    launch_bias_bwd(dY_attn.d, L.dbo.d, N, m);
    Tensor dO = make_tensor({B, T, m});
    gemm_nt(L.cublas, N, m, m, 1.0f, dY_attn.d, L.Wo.d, 0.0f, dO.d);
    free_tensor(dY_attn);

    // Split-heads: dO (B, T, m) → dO_heads (B, H, T, d_h)
    Tensor dO_heads = make_tensor({B, H, T, d_h});
    launch_split_heads(dO.d, dO_heads.d, B, T, H, d_h);
    free_tensor(dO);

    // O_heads = S @ V (batched)
    // dS = dO_heads @ V^T; dV = S^T @ dO_heads
    Tensor dS = make_tensor({B, H, T, T});
    gemm_strided_batched_nt(L.cublas, T, T, d_h, 1.0f,
                            dO_heads.d, T * d_h,
                            cache.V.d, T * d_h,
                            0.0f, dS.d, T * T, B * H);
    Tensor dV = make_tensor({B, H, T, d_h});
    gemm_strided_batched_tn(L.cublas, T, d_h, T, 1.0f,
                            cache.S.d, T * T,
                            dO_heads.d, T * d_h,
                            0.0f, dV.d, T * d_h, B * H);
    free_tensor(dO_heads);

    // Softmax backward (causal)
    Tensor dS_pre = make_tensor({B, H, T, T});
    launch_softmax_causal_bwd(cache.S.d, dS.d, dS_pre.d, B * H, T);
    free_tensor(dS);

    // S = Q @ K^T / sqrt(d_h)
    // dQ = dS_pre @ K / sqrt(d_h); dK = dS_pre^T @ Q / sqrt(d_h)
    float inv_sqrt = 1.0f / std::sqrt((float)d_h);
    Tensor dQ_h = make_tensor({B, H, T, d_h});
    Tensor dK_h = make_tensor({B, H, T, d_h});
    gemm_strided_batched_nn(L.cublas, T, d_h, T, inv_sqrt,
                            dS_pre.d, T * T,
                            cache.K.d, T * d_h,
                            0.0f, dQ_h.d, T * d_h, B * H);
    gemm_strided_batched_tn(L.cublas, T, d_h, T, inv_sqrt,
                            dS_pre.d, T * T,
                            cache.Q.d, T * d_h,
                            0.0f, dK_h.d, T * d_h, B * H);
    free_tensor(dS_pre);

    // Merge heads back
    Tensor dQ_flat = make_tensor({B, T, m});
    Tensor dK_flat = make_tensor({B, T, m});
    launch_merge_heads(dQ_h.d, dQ_flat.d, B, T, H, d_h);
    launch_merge_heads(dK_h.d, dK_flat.d, B, T, H, d_h);
    free_tensor(dQ_h); free_tensor(dK_h);

    Tensor dV_flat = make_tensor({B, T, m});
    launch_merge_heads(dV.d, dV_flat.d, B, T, H, d_h);
    free_tensor(dV);

    // Concat back into dQKV
    Tensor dQKV = make_tensor({B, T, 3 * m});
    {
        dim3 block(128);
        dim3 grid(B, T, (m + 127) / 128);
        k_concat_qkv<<<grid, block>>>(dQ_flat.d, dK_flat.d, dV_flat.d, dQKV.d, B, T, m);
    }
    free_tensor(dQ_flat); free_tensor(dK_flat); free_tensor(dV_flat);

    // qkv = x_ln1 @ Wqkv + bqkv
    gemm_tn(L.cublas, m, 3 * m, N, 1.0f, cache.x_ln1.d, dQKV.d, 1.0f, L.dWqkv.d);
    launch_bias_bwd(dQKV.d, L.dbqkv.d, N, 3 * m);
    Tensor dx_ln1 = make_tensor({B, T, m});
    gemm_nt(L.cublas, N, m, 3 * m, 1.0f, dQKV.d, L.Wqkv.d, 0.0f, dx_ln1.d);
    free_tensor(dQKV);

    // LN1 backward → accumulate into dx_in_d
    Tensor dx_in_from_ln = make_tensor({B, T, m});
    dx_in_from_ln.zero_();
    launch_layer_norm_bwd(cache.x_in.d, dx_ln1.d, L.ln1_g.d,
                          cache.mean1.d, cache.rstd1.d,
                          dx_in_from_ln.d, L.dln1_g.d, L.dln1_b.d, N, m);
    free_tensor(dx_ln1);
    {
        int n = B * T * m;
        int block = 256;
        int grid = (n + block - 1) / block;
        k_add_inplace<<<grid, block>>>(dx_in_d, dx_in_from_ln.d, n);
    }
    free_tensor(dx_in_from_ln);

    free_tensor(dh2);
}

// ===== Full Transformer model =====

struct TransformerModelCache {
    int B = 0, T = 0, m = 0, H = 0, L_layers = 0, n_classes = 0;
    const int* ids = nullptr;
    Tensor x0;          // (B, T, m) embedding + PE
    std::vector<TransformerLayerCache> layer_caches;
    Tensor x_final_ln;  // (B, T, m) final LN output
    Tensor mean_f, rstd_f;
    Tensor logits;
    Tensor probs;
    Tensor losses;

    void alloc(int B_, int T_, int m_, int H_, int L_, int n_classes_) {
        B = B_; T = T_; m = m_; H = H_; L_layers = L_; n_classes = n_classes_;
        x0 = make_tensor({B, T, m});
        layer_caches.resize(L_);
        for (int i = 0; i < L_; ++i) layer_caches[i].alloc(B, T, m, H);
        x_final_ln = make_tensor({B, T, m});
        mean_f = make_tensor({B * T});
        rstd_f = make_tensor({B * T});
        logits = make_tensor({B, n_classes});
        probs = make_tensor({B, n_classes});
        losses = make_tensor({B});
    }
    void free() {
        free_tensor(x0);
        for (auto& c : layer_caches) c.free();
        layer_caches.clear();
        free_tensor(x_final_ln); free_tensor(mean_f); free_tensor(rstd_f);
        free_tensor(logits); free_tensor(probs); free_tensor(losses);
    }
};

struct TransformerModel {
    cublasHandle_t cublas;
    int V = 0, m = 0, H = 0, L_layers = 0, n_classes = 0;
    Tensor E;       // (V, m)
    Tensor pe;      // (max_T, m) — precomputed, allocated once
    std::vector<TransformerLayer> layers;
    Tensor ln_f_g, ln_f_b;  // final LN
    Tensor W_out;  // (m, n_classes)
    Tensor b_out;

    Tensor dE, dln_f_g, dln_f_b, dW_out, db_out;
    Tensor mE, mln_f_g, mln_f_b, mW_out, mb_out;
    Tensor vE, vln_f_g, vln_f_b, vW_out, vb_out;

    int max_T = 0;

    std::vector<Tensor*> all_params() {
        std::vector<Tensor*> v = { &E, &ln_f_g, &ln_f_b, &W_out, &b_out };
        for (auto& L : layers) for (Tensor* p : L.params()) v.push_back(p);
        return v;
    }
    std::vector<Tensor*> all_grads() {
        std::vector<Tensor*> v = { &dE, &dln_f_g, &dln_f_b, &dW_out, &db_out };
        for (auto& L : layers) for (Tensor* p : L.grads()) v.push_back(p);
        return v;
    }
    std::vector<Tensor*> all_ms() {
        std::vector<Tensor*> v = { &mE, &mln_f_g, &mln_f_b, &mW_out, &mb_out };
        for (auto& L : layers) for (Tensor* p : L.ms()) v.push_back(p);
        return v;
    }
    std::vector<Tensor*> all_vs() {
        std::vector<Tensor*> v = { &vE, &vln_f_g, &vln_f_b, &vW_out, &vb_out };
        for (auto& L : layers) for (Tensor* p : L.vs()) v.push_back(p);
        return v;
    }

    void init(cublasHandle_t handle, int V_, int m_, int H_, int L_, int n_classes_,
              int max_T_, unsigned long long seed) {
        cublas = handle;
        V = V_; m = m_; H = H_; L_layers = L_; n_classes = n_classes_;
        max_T = max_T_;
        HostRng rng(seed);

        E = make_tensor({V, m});
        ln_f_g = make_tensor({m}); ln_f_b = make_tensor({m});
        W_out = make_tensor({m, n_classes}); b_out = make_tensor({n_classes});
        pe = make_tensor({max_T, m});

        dE = make_tensor({V, m});
        dln_f_g = make_tensor({m}); dln_f_b = make_tensor({m});
        dW_out = make_tensor({m, n_classes}); db_out = make_tensor({n_classes});

        mE = make_tensor({V, m}); mE.zero_();
        mln_f_g = make_tensor({m}); mln_f_g.zero_();
        mln_f_b = make_tensor({m}); mln_f_b.zero_();
        mW_out = make_tensor({m, n_classes}); mW_out.zero_();
        mb_out = make_tensor({n_classes}); mb_out.zero_();
        vE = make_tensor({V, m}); vE.zero_();
        vln_f_g = make_tensor({m}); vln_f_g.zero_();
        vln_f_b = make_tensor({m}); vln_f_b.zero_();
        vW_out = make_tensor({m, n_classes}); vW_out.zero_();
        vb_out = make_tensor({n_classes}); vb_out.zero_();

        init_normal(E, 0.0f, 1.0f / std::sqrt((float)m), rng);
        std::vector<float> ones(m, 1.0f);
        copy_h2d(ln_f_g, ones); ln_f_b.zero_();
        init_xavier_uniform(W_out, m, n_classes, rng); b_out.zero_();

        make_sinusoidal_pe(pe, max_T, m);

        layers.resize(L_);
        for (int i = 0; i < L_; ++i) {
            layers[i].init(handle, m, H, rng);
        }
    }

    void free_all() {
        free_tensor(E); free_tensor(ln_f_g); free_tensor(ln_f_b);
        free_tensor(W_out); free_tensor(b_out); free_tensor(pe);
        free_tensor(dE); free_tensor(dln_f_g); free_tensor(dln_f_b);
        free_tensor(dW_out); free_tensor(db_out);
        free_tensor(mE); free_tensor(mln_f_g); free_tensor(mln_f_b);
        free_tensor(mW_out); free_tensor(mb_out);
        free_tensor(vE); free_tensor(vln_f_g); free_tensor(vln_f_b);
        free_tensor(vW_out); free_tensor(vb_out);
        for (auto& L : layers) L.free_all();
        layers.clear();
    }

    void zero_grads() {
        dE.zero_(); dln_f_g.zero_(); dln_f_b.zero_(); dW_out.zero_(); db_out.zero_();
        for (auto& L : layers) L.zero_grads();
    }

    int64_t num_params() {
        int64_t n = 0;
        for (Tensor* p : all_params()) n += p->numel;
        return n;
    }

    void forward(const int* ids_device, int B, int T, TransformerModelCache& cache) {
        cache.ids = ids_device;
        cache.B = B; cache.T = T; cache.m = m; cache.H = H;
        cache.L_layers = L_layers; cache.n_classes = n_classes;

        // Embedding: x0[b, t, :] = E[ids[b, t]] for all (b, t) in one kernel.
        flat_embedding_fwd(ids_device, E.d, cache.x0.d, B, T, m);

        // Add positional encoding
        launch_add_pe(cache.x0.d, pe.d, B, T, m);

        // Layer stack
        const float* x_curr = cache.x0.d;
        for (int li = 0; li < L_layers; ++li) {
            transformer_layer_forward(layers[li], x_curr, cache.layer_caches[li]);
            x_curr = cache.layer_caches[li].x_out.d;
        }

        // Final LN over (B, T, m)
        int N = B * T;
        launch_layer_norm_fwd(x_curr, ln_f_g.d, ln_f_b.d, cache.x_final_ln.d,
                              cache.mean_f.d, cache.rstd_f.d, N, m);

        // Last-token pooling: take x_final_ln[:, T-1, :] → (B, m). Then logits = ... @ W_out + b_out
        // Use a kernel to extract last position.
        Tensor last = make_tensor({B, m});
        extract_last_pos(cache.x_final_ln.d, last.d, B, T, m);
        gemm_nn(cublas, B, n_classes, m, 1.0f, last.d, W_out.d, 0.0f, cache.logits.d);
        launch_bias_add(cache.logits.d, b_out.d, B, n_classes);
        free_tensor(last);
    }

    float compute_loss(const int* labels_device, TransformerModelCache& cache) {
        launch_softmax_ce_fwd(cache.logits.d, labels_device, cache.probs.d, cache.losses.d,
                              cache.B, cache.n_classes);
        std::vector<float> h_losses;
        copy_d2h(h_losses, cache.losses);
        float s = 0.0f;
        for (float v : h_losses) s += v;
        return s;
    }

    void backward(const int* labels_device, TransformerModelCache& cache) {
        const int B = cache.B, T = cache.T;
        const float scale = 1.0f / (float)B;

        Tensor d_logits = make_tensor({B, n_classes});
        launch_softmax_ce_bwd(cache.probs.d, labels_device, d_logits.d, B, n_classes, scale);
        launch_bias_bwd(d_logits.d, db_out.d, B, n_classes);

        // Get last_pos features
        Tensor last = make_tensor({B, m});
        extract_last_pos(cache.x_final_ln.d, last.d, B, T, m);
        // dW_out += last^T @ d_logits; d_last = d_logits @ W_out^T
        gemm_tn(cublas, m, n_classes, B, 1.0f, last.d, d_logits.d, 1.0f, dW_out.d);
        Tensor d_last = make_tensor({B, m});
        gemm_nt(cublas, B, m, n_classes, 1.0f, d_logits.d, W_out.d, 0.0f, d_last.d);
        free_tensor(last);

        // Scatter d_last back to d_x_final_ln (only the last position has nonzero grad)
        Tensor d_x_final_ln = make_tensor({B, T, m});
        d_x_final_ln.zero_();
        scatter_last_pos(d_last.d, d_x_final_ln.d, B, T, m);
        free_tensor(d_last);

        // Final LN backward
        int N = B * T;
        const float* x_last_layer = (L_layers > 0)
            ? cache.layer_caches[L_layers - 1].x_out.d
            : cache.x0.d;
        Tensor d_x_last_layer = make_tensor({B, T, m});
        d_x_last_layer.zero_();
        launch_layer_norm_bwd(x_last_layer, d_x_final_ln.d, ln_f_g.d,
                              cache.mean_f.d, cache.rstd_f.d,
                              d_x_last_layer.d, dln_f_g.d, dln_f_b.d, N, m);
        free_tensor(d_x_final_ln);

        // Layer stack backward (reverse order)
        Tensor d_x = d_x_last_layer;
        for (int li = L_layers - 1; li >= 0; --li) {
            Tensor d_x_in = make_tensor({B, T, m});
            transformer_layer_backward(layers[li], d_x.d, cache.layer_caches[li], d_x_in.d);
            free_tensor(d_x);
            d_x = d_x_in;
        }

        // Backward through PE add (PE has no grad)
        // Backward through embedding: dE[ids[b, t]] += d_x[b, t, :]
        flat_embedding_bwd(cache.ids, d_x.d, dE.d, B, T, m);

        free_tensor(d_x);
        free_tensor(d_logits);
    }

    // === helpers ===
    static void flat_embedding_fwd(const int* ids, const float* E, float* out,
                                   int B, int T, int m);
    static void flat_embedding_bwd(const int* ids, const float* d_out, float* dE,
                                   int B, int T, int m);
    static void extract_last_pos(const float* x, float* out, int B, int T, int m);
    static void scatter_last_pos(const float* in, float* x, int B, int T, int m);
};

// Implementation of helpers

__global__ void k_flat_embedding_fwd(const int* ids, const float* E, float* out,
                                     int B, int T, int m) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b < B && t < T && d < m) {
        int id = ids[b * T + t];
        out[((b * T) + t) * m + d] = E[id * m + d];
    }
}

__global__ void k_flat_embedding_bwd(const int* ids, const float* d_out, float* dE,
                                     int B, int T, int m) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;
    if (b < B && t < T && d < m) {
        int id = ids[b * T + t];
        atomicAdd(&dE[id * m + d], d_out[((b * T) + t) * m + d]);
    }
}

inline void TransformerModel::flat_embedding_fwd(const int* ids, const float* E, float* out,
                                                  int B, int T, int m) {
    dim3 block(128);
    dim3 grid(B, T, (m + 127) / 128);
    k_flat_embedding_fwd<<<grid, block>>>(ids, E, out, B, T, m);
}

inline void TransformerModel::flat_embedding_bwd(const int* ids, const float* d_out, float* dE,
                                                  int B, int T, int m) {
    dim3 block(128);
    dim3 grid(B, T, (m + 127) / 128);
    k_flat_embedding_bwd<<<grid, block>>>(ids, d_out, dE, B, T, m);
}

__global__ void k_extract_last_pos(const float* x, float* out, int B, int T, int m) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b < B && d < m) {
        out[b * m + d] = x[((b * T) + (T - 1)) * m + d];
    }
}

__global__ void k_scatter_last_pos(const float* in, float* x, int B, int T, int m) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b < B && d < m) {
        x[((b * T) + (T - 1)) * m + d] = in[b * m + d];
    }
}

inline void TransformerModel::extract_last_pos(const float* x, float* out, int B, int T, int m) {
    dim3 block(128);
    dim3 grid(B, (m + 127) / 128);
    k_extract_last_pos<<<grid, block>>>(x, out, B, T, m);
}

inline void TransformerModel::scatter_last_pos(const float* in, float* x, int B, int T, int m) {
    dim3 block(128);
    dim3 grid(B, (m + 127) / 128);
    k_scatter_last_pos<<<grid, block>>>(in, x, B, T, m);
}

#endif // EALRMN_MODEL_TRANSFORMER_CUH
