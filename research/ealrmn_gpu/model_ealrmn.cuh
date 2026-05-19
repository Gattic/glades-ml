// EALRMN Phase-1 GPU prototype — full EALRMN-attmem model
// Encoder → Koopman linear recurrence → 4-slot gated EMA memory → attention readout → MLP head.
#ifndef EALRMN_MODEL_EALRMN_CUH
#define EALRMN_MODEL_EALRMN_CUH

#include "common.cuh"
#include "kernels.cuh"
#include "recurrence_kernels.cuh"

struct EALRMNModelCache {
    int B = 0;
    int T = 0;
    int m = 0;
    int J = 0;
    int n_classes = 0;

    // Inputs (device pointer to ids, host owns it)
    const int* ids = nullptr;

    // Saved forward tensors
    Tensor z_all;     // (T, B, m)
    Tensor s_all;     // (T+1, B, m)
    Tensor g_all;     // (T, B, J)
    Tensor M_T;       // (B, J, m)
    Tensor q;         // (B, m)
    Tensor alpha;     // (B, J)
    Tensor scores;    // (B, J)  saved pre-softmax (unused after fwd but kept for debugging)
    Tensor r;         // (B, m)
    Tensor feat;      // (B, 2m)
    Tensor logits;    // (B, n_classes)
    Tensor probs;     // (B, n_classes) (post-softmax, for CE bwd)
    Tensor losses;    // (B,) per-sample CE loss

    void alloc(int B_, int T_, int m_, int J_, int n_classes_) {
        B = B_; T = T_; m = m_; J = J_; n_classes = n_classes_;
        z_all  = make_tensor({T, B, m});
        s_all  = make_tensor({T + 1, B, m});
        g_all  = make_tensor({T, B, J});
        M_T    = make_tensor({B, J, m});
        q      = make_tensor({B, m});
        alpha  = make_tensor({B, J});
        scores = make_tensor({B, J});
        r      = make_tensor({B, m});
        feat   = make_tensor({B, 2 * m});
        logits = make_tensor({B, n_classes});
        probs  = make_tensor({B, n_classes});
        losses = make_tensor({B});
    }
    void free() {
        free_tensor(z_all);  free_tensor(s_all);  free_tensor(g_all);
        free_tensor(M_T);    free_tensor(q);      free_tensor(alpha);
        free_tensor(scores); free_tensor(r);      free_tensor(feat);
        free_tensor(logits); free_tensor(probs);  free_tensor(losses);
    }
};

struct EALRMNModel {
    cublasHandle_t cublas;
    int V = 0, m = 0, J = 4, n_classes = 0;
    float spectral_radius = 0.95f;
    // Ablation knobs (set BEFORE init()).
    std::string init_K_method = "orthogonal"; // "orthogonal" | "xavier"
    bool use_attmem = true;                   // false → r forced to 0; attention dead

    // Parameters
    Tensor E;       // (V, m)
    Tensor K;       // (m, m)
    Tensor W_in;    // (m, m)
    Tensor W_g;     // (J, m)
    Tensor b_g;     // (J,)
    Tensor W_q;     // (m, m)
    Tensor b_q;     // (m,)
    Tensor W_out;   // (2m, n_classes)
    Tensor b_out;   // (n_classes,)
    Tensor lambda;  // (J,) — fixed multi-timescale decays

    // Gradients (same shapes)
    Tensor dE, dK, dW_in, dW_g, db_g, dW_q, db_q, dW_out, db_out;

    // Adam state
    Tensor mE, mK, mW_in, mW_g, mb_g, mW_q, mb_q, mW_out, mb_out;
    Tensor vE, vK, vW_in, vW_g, vb_g, vW_q, vb_q, vW_out, vb_out;

    std::vector<Tensor*> params() {
        return { &E, &K, &W_in, &W_g, &b_g, &W_q, &b_q, &W_out, &b_out };
    }
    std::vector<Tensor*> grads() {
        return { &dE, &dK, &dW_in, &dW_g, &db_g, &dW_q, &db_q, &dW_out, &db_out };
    }
    std::vector<Tensor*> ms() {
        return { &mE, &mK, &mW_in, &mW_g, &mb_g, &mW_q, &mb_q, &mW_out, &mb_out };
    }
    std::vector<Tensor*> vs() {
        return { &vE, &vK, &vW_in, &vW_g, &vb_g, &vW_q, &vb_q, &vW_out, &vb_out };
    }

    void init(cublasHandle_t handle, int V_, int m_, int J_, int n_classes_,
              unsigned long long seed) {
        cublas = handle;
        V = V_; m = m_; J = J_; n_classes = n_classes_;
        HostRng rng(seed);

        // Allocate
        E     = make_tensor({V, m});
        K     = make_tensor({m, m});
        W_in  = make_tensor({m, m});
        W_g   = make_tensor({J, m});
        b_g   = make_tensor({J});
        W_q   = make_tensor({m, m});
        b_q   = make_tensor({m});
        W_out = make_tensor({2 * m, n_classes});
        b_out = make_tensor({n_classes});
        lambda = make_tensor({J});

        // Gradients
        dE     = make_tensor({V, m});
        dK     = make_tensor({m, m});
        dW_in  = make_tensor({m, m});
        dW_g   = make_tensor({J, m});
        db_g   = make_tensor({J});
        dW_q   = make_tensor({m, m});
        db_q   = make_tensor({m});
        dW_out = make_tensor({2 * m, n_classes});
        db_out = make_tensor({n_classes});

        // Adam
        for (Tensor* p : params()) {
            // m, v aligned
        }
        mE     = make_tensor({V, m});      mE.zero_();
        mK     = make_tensor({m, m});      mK.zero_();
        mW_in  = make_tensor({m, m});      mW_in.zero_();
        mW_g   = make_tensor({J, m});      mW_g.zero_();
        mb_g   = make_tensor({J});         mb_g.zero_();
        mW_q   = make_tensor({m, m});      mW_q.zero_();
        mb_q   = make_tensor({m});         mb_q.zero_();
        mW_out = make_tensor({2 * m, n_classes}); mW_out.zero_();
        mb_out = make_tensor({n_classes}); mb_out.zero_();
        vE     = make_tensor({V, m});      vE.zero_();
        vK     = make_tensor({m, m});      vK.zero_();
        vW_in  = make_tensor({m, m});      vW_in.zero_();
        vW_g   = make_tensor({J, m});      vW_g.zero_();
        vb_g   = make_tensor({J});         vb_g.zero_();
        vW_q   = make_tensor({m, m});      vW_q.zero_();
        vb_q   = make_tensor({m});         vb_q.zero_();
        vW_out = make_tensor({2 * m, n_classes}); vW_out.zero_();
        vb_out = make_tensor({n_classes}); vb_out.zero_();

        // Init parameters
        float e_scale = 1.0f / std::sqrt((float)m);
        init_normal(E, 0.0f, e_scale, rng);

        // K: orthogonal scaled to spectral radius < 1, OR Xavier (ablation).
        if (init_K_method == "xavier") {
            init_xavier_uniform(K, m, m, rng);
        } else {
            init_orthogonal_scale(K, m, spectral_radius, rng);
        }

        // W_in: Xavier
        init_xavier_uniform(W_in, m, m, rng);
        // W_g: small init so gate starts ~ sigmoid(0) = 0.5
        {
            float bound = 0.1f * std::sqrt(6.0f / (float)(m + 1));
            std::vector<float> h(J * m);
            for (int i = 0; i < J * m; ++i) h[i] = (rng.next_uniform() * 2.0f - 1.0f) * bound;
            copy_h2d(W_g, h);
        }
        b_g.zero_();
        init_xavier_uniform(W_q, m, m, rng);
        b_q.zero_();
        init_xavier_uniform(W_out, 2 * m, n_classes, rng);
        b_out.zero_();

        // lambda: [0.5, 0.1, 0.01, 0.001] (truncate or extend if J != 4)
        std::vector<float> lam(J);
        float base[4] = { 0.5f, 0.1f, 0.01f, 0.001f };
        for (int j = 0; j < J; ++j) lam[j] = (j < 4) ? base[j] : 0.001f;
        copy_h2d(lambda, lam);
    }

    void free_all() {
        free_tensor(E); free_tensor(K); free_tensor(W_in);
        free_tensor(W_g); free_tensor(b_g); free_tensor(W_q); free_tensor(b_q);
        free_tensor(W_out); free_tensor(b_out); free_tensor(lambda);
        free_tensor(dE); free_tensor(dK); free_tensor(dW_in);
        free_tensor(dW_g); free_tensor(db_g); free_tensor(dW_q); free_tensor(db_q);
        free_tensor(dW_out); free_tensor(db_out);
        free_tensor(mE); free_tensor(mK); free_tensor(mW_in);
        free_tensor(mW_g); free_tensor(mb_g); free_tensor(mW_q); free_tensor(mb_q);
        free_tensor(mW_out); free_tensor(mb_out);
        free_tensor(vE); free_tensor(vK); free_tensor(vW_in);
        free_tensor(vW_g); free_tensor(vb_g); free_tensor(vW_q); free_tensor(vb_q);
        free_tensor(vW_out); free_tensor(vb_out);
    }

    void zero_grads() {
        for (Tensor* g : grads()) g->zero_();
    }

    int64_t num_params() {
        int64_t n = 0;
        for (Tensor* p : params()) n += p->numel;
        return n;
    }

    // Forward pass.
    // ids: device int array of length B*T (each id ∈ [0, V))
    // labels: not needed here (used in loss); cache.logits will be filled.
    //
    // Saves all needed tensors in cache for backward.
    void forward(const int* ids_device, int B, int T, EALRMNModelCache& cache) {
        cache.ids = ids_device;
        cache.B = B; cache.T = T; cache.m = m; cache.J = J; cache.n_classes = n_classes;

        // Zero initial state and memory
        Tensor s_prev = make_tensor({B, m});
        s_prev.zero_();
        Tensor M = make_tensor({B, J, m});
        M.zero_();

        // s_all[0] = 0
        CUDA_CHECK(cudaMemcpy(cache.s_all.d, s_prev.d, B * m * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        // Workspace
        Tensor z_t = make_tensor({B, m});
        Tensor s_t = make_tensor({B, m});
        Tensor pre_g = make_tensor({B, J});
        Tensor g_t = make_tensor({B, J});

        for (int t = 0; t < T; ++t) {
            // Embedding lookup: z_t = E[ids[:, t]]
            embedding_fwd_col(ids_device, E.d, z_t.d, B, T, t, m);

            // Save z_t
            CUDA_CHECK(cudaMemcpy(cache.z_all.d + (int64_t)t * B * m, z_t.d,
                                  B * m * sizeof(float), cudaMemcpyDeviceToDevice));

            // s_t = s_prev @ K^T + z_t @ W_in^T
            gemm_nt(cublas, B, m, m, 1.0f, s_prev.d, K.d, 0.0f, s_t.d);
            gemm_nt(cublas, B, m, m, 1.0f, z_t.d, W_in.d, 1.0f, s_t.d);

            // Save s_t into s_all[t+1]
            CUDA_CHECK(cudaMemcpy(cache.s_all.d + (int64_t)(t + 1) * B * m, s_t.d,
                                  B * m * sizeof(float), cudaMemcpyDeviceToDevice));

            // pre_g = s_t @ W_g^T + b_g
            gemm_nt(cublas, B, J, m, 1.0f, s_t.d, W_g.d, 0.0f, pre_g.d);
            {
                dim3 block(J);
                dim3 grid(B);
                k_add_bias_J<<<grid, block>>>(pre_g.d, b_g.d, B, J);
            }
            // g = sigmoid(pre_g)
            {
                int n = B * J;
                int block = 128;
                int grid = (n + block - 1) / block;
                k_sigmoid_fwd<<<grid, block>>>(pre_g.d, g_t.d, n);
            }
            // Save g_t
            CUDA_CHECK(cudaMemcpy(cache.g_all.d + (int64_t)t * B * J, g_t.d,
                                  B * J * sizeof(float), cudaMemcpyDeviceToDevice));

            // Memory update: M[b, j, d] = (1-λ_j) M[b, j, d] + λ_j g[b, j] z[b, d]
            launch_memory_update_fwd(M.d, g_t.d, z_t.d, lambda.d, B, J, m);

            // Advance s_prev = s_t (via swap of pointers — here just copy)
            CUDA_CHECK(cudaMemcpy(s_prev.d, s_t.d, B * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        // Save final memory
        CUDA_CHECK(cudaMemcpy(cache.M_T.d, M.d, B * J * m * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        // Attention readout at t=T:
        // q = s_T @ W_q^T + b_q
        gemm_nt(cublas, B, m, m, 1.0f, s_prev.d, W_q.d, 0.0f, cache.q.d);
        {
            dim3 block(128);
            dim3 grid(B, (m + 127) / 128);
            k_bias_add<<<grid, block>>>(cache.q.d, b_q.d, B, m);
        }
        att_readout_fwd(cache.q.d, cache.M_T.d, cache.scores.d, cache.alpha.d, cache.r.d, B, J, m);

        // Ablation: force r=0 so the attention path is dead.
        if (!use_attmem) {
            CUDA_CHECK(cudaMemset(cache.r.d, 0, B * m * sizeof(float)));
        }

        // feat = concat(s_T, r)
        launch_concat_2(s_prev.d, cache.r.d, cache.feat.d, B, m, m);

        // logits = feat @ W_out + b_out
        gemm_nn(cublas, B, n_classes, 2 * m, 1.0f, cache.feat.d, W_out.d, 0.0f, cache.logits.d);
        launch_bias_add(cache.logits.d, b_out.d, B, n_classes);

        free_tensor(s_prev); free_tensor(M);
        free_tensor(z_t); free_tensor(s_t); free_tensor(pre_g); free_tensor(g_t);
    }

    // Compute loss and softmax for backward.
    // Returns total loss (sum over B).
    float compute_loss(const int* labels_device, EALRMNModelCache& cache) {
        launch_softmax_ce_fwd(cache.logits.d, labels_device, cache.probs.d, cache.losses.d,
                              cache.B, cache.n_classes);
        std::vector<float> h_losses;
        copy_d2h(h_losses, cache.losses);
        float s = 0.0f;
        for (float v : h_losses) s += v;
        return s;
    }

    // Backward pass. Assumes forward+compute_loss already called.
    void backward(const int* labels_device, EALRMNModelCache& cache) {
        const int B = cache.B, T = cache.T;
        const float scale = 1.0f / (float)B;  // mean over batch

        // d_logits = (probs - one_hot(label)) * scale
        Tensor d_logits = make_tensor({B, n_classes});
        launch_softmax_ce_bwd(cache.probs.d, labels_device, d_logits.d, B, n_classes, scale);

        // db_out += sum_b d_logits[b, :]
        launch_bias_bwd(d_logits.d, db_out.d, B, n_classes);

        // dW_out += feat^T @ d_logits;  d_feat = d_logits @ W_out^T
        Tensor d_feat = make_tensor({B, 2 * m});
        gemm_tn(cublas, 2 * m, n_classes, B, 1.0f, cache.feat.d, d_logits.d, 1.0f, dW_out.d);
        gemm_nt(cublas, B, 2 * m, n_classes, 1.0f, d_logits.d, W_out.d, 0.0f, d_feat.d);

        // Split d_feat into d_s_T and d_r
        Tensor d_s_T = make_tensor({B, m});
        Tensor d_r = make_tensor({B, m});
        launch_split_2(d_feat.d, d_s_T.d, d_r.d, B, m, m);

        // Attention readout backward
        // Ablation: when use_attmem=false, the attention output was zeroed in forward,
        // so the whole attention block has no learning signal. Skip the backward to
        // keep W_q, b_q frozen at init values and dM_attn = 0.
        Tensor d_q = make_tensor({B, m});
        d_q.zero_();
        Tensor d_M_attn = make_tensor({B, J, m});
        Tensor dalpha = make_tensor({B, J});
        Tensor dscore = make_tensor({B, J});
        const float* s_T_ptr = cache.s_all.d + (int64_t)T * B * m;
        if (use_attmem) {
            att_readout_bwd(d_r.d, cache.q.d, cache.M_T.d, cache.alpha.d,
                            d_q.d, d_M_attn.d, dalpha.d, dscore.d, B, J, m);
            launch_bias_bwd(d_q.d, db_q.d, B, m);
            gemm_tn(cublas, m, m, B, 1.0f, d_q.d, s_T_ptr, 1.0f, dW_q.d);
            gemm_nn(cublas, B, m, m, 1.0f, d_q.d, W_q.d, 1.0f, d_s_T.d);
        } else {
            // Ablation: zero out d_M_attn so the memory backward chain receives no
            // signal from the (dead) attention block.
            CUDA_CHECK(cudaMemset(d_M_attn.d, 0, B * J * m * sizeof(float)));
        }

        // BPTT loop. Initialize d_s as d_s_T, d_M as d_M_attn.
        Tensor d_s = make_tensor({B, m});
        CUDA_CHECK(cudaMemcpy(d_s.d, d_s_T.d, B * m * sizeof(float), cudaMemcpyDeviceToDevice));
        Tensor d_M = make_tensor({B, J, m});
        CUDA_CHECK(cudaMemcpy(d_M.d, d_M_attn.d, B * J * m * sizeof(float), cudaMemcpyDeviceToDevice));

        // Workspace per step
        Tensor d_z = make_tensor({B, m});
        Tensor d_g = make_tensor({B, J});
        Tensor d_M_prev = make_tensor({B, J, m});
        Tensor d_s_prev = make_tensor({B, m});
        Tensor d_pre_g = make_tensor({B, J});
        Tensor d_s_from_gate = make_tensor({B, m});

        for (int t = T - 1; t >= 0; --t) {
            // At this point d_s is d(s_t), d_M is d(M_t).
            //
            // Backward through memory update at step t:
            //   dz_t += sum_j λ_j g[t,j] dM[t,j,:]
            //   dg[t, j] = λ_j <z_t, dM[t, j, :]>
            //   dM_{t-1}[b, j, d] = (1-λ_j) dM[t, b, j, d]
            d_z.zero_();
            d_g.zero_();
            const float* z_t_ptr = cache.z_all.d + (int64_t)t * B * m;
            const float* g_t_ptr = cache.g_all.d + (int64_t)t * B * J;
            launch_memory_update_bwd(d_M_prev.d, d_M.d, z_t_ptr, g_t_ptr,
                                     lambda.d, d_z.d, d_g.d, B, J, m);

            // Backward through sigmoid in gate: d_pre_g = d_g * g * (1 - g)
            CUDA_CHECK(cudaMemcpy(d_pre_g.d, d_g.d, B * J * sizeof(float), cudaMemcpyDeviceToDevice));
            {
                int n = B * J;
                int block = 128;
                int grid = (n + block - 1) / block;
                k_sigmoid_bwd_inplace<<<grid, block>>>(d_pre_g.d, g_t_ptr, n);
            }
            // pre_g = s_t @ W_g^T + b_g
            // db_g += sum_b d_pre_g[b, :]
            launch_bias_bwd(d_pre_g.d, db_g.d, B, J);
            // dW_g (J, m) += d_pre_g^T @ s_t
            const float* s_t_ptr = cache.s_all.d + (int64_t)(t + 1) * B * m;
            gemm_tn(cublas, J, m, B, 1.0f, d_pre_g.d, s_t_ptr, 1.0f, dW_g.d);
            // ds_t (from gate) += d_pre_g @ W_g
            gemm_nn(cublas, B, m, J, 1.0f, d_pre_g.d, W_g.d, 0.0f, d_s_from_gate.d);
            {
                int n = B * m;
                int block = 256;
                int grid = (n + block - 1) / block;
                k_add_inplace<<<grid, block>>>(d_s.d, d_s_from_gate.d, n);
            }

            // Backward through recurrence: s_t = s_{t-1} @ K^T + z_t @ W_in^T
            // dK (m, m) += d_s^T @ s_{t-1}
            const float* s_prev_ptr = cache.s_all.d + (int64_t)t * B * m;
            gemm_tn(cublas, m, m, B, 1.0f, d_s.d, s_prev_ptr, 1.0f, dK.d);
            // ds_{t-1} = d_s @ K
            gemm_nn(cublas, B, m, m, 1.0f, d_s.d, K.d, 0.0f, d_s_prev.d);
            // dW_in (m, m) += d_s^T @ z_t
            gemm_tn(cublas, m, m, B, 1.0f, d_s.d, z_t_ptr, 1.0f, dW_in.d);
            // dz_t += d_s @ W_in
            gemm_nn(cublas, B, m, m, 1.0f, d_s.d, W_in.d, 1.0f, d_z.d);

            // Embedding backward: dE[ids[:, t]] += d_z[b, :]
            embedding_bwd_col(cache.ids, d_z.d, dE.d, B, T, t, m);

            // Advance: d_s = d_s_prev, d_M = d_M_prev
            CUDA_CHECK(cudaMemcpy(d_s.d, d_s_prev.d, B * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_M.d, d_M_prev.d, B * J * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        // Free workspace
        free_tensor(d_logits); free_tensor(d_feat); free_tensor(d_s_T); free_tensor(d_r);
        free_tensor(d_q); free_tensor(d_M_attn); free_tensor(dalpha); free_tensor(dscore);
        free_tensor(d_s); free_tensor(d_M);
        free_tensor(d_z); free_tensor(d_g); free_tensor(d_M_prev); free_tensor(d_s_prev);
        free_tensor(d_pre_g); free_tensor(d_s_from_gate);
    }

    // Helper: embedding fwd at column t
    static inline void embedding_fwd_col(const int* ids, const float* table, float* out,
                                         int B, int T, int t, int D);
    static inline void embedding_bwd_col(const int* ids, const float* d_out, float* d_table,
                                         int B, int T, int t, int D);
};

// ===== Embedding helpers (column-wise across time) =====
__global__ void k_embedding_fwd_col(const int* ids, const float* table, float* out,
                                    int B, int T, int t, int D) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b < B && d < D) {
        int id = ids[b * T + t];
        out[b * D + d] = table[id * D + d];
    }
}

__global__ void k_embedding_bwd_col(const int* ids, const float* d_out, float* d_table,
                                    int B, int T, int t, int D) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (b < B && d < D) {
        int id = ids[b * T + t];
        atomicAdd(&d_table[id * D + d], d_out[b * D + d]);
    }
}

inline void EALRMNModel::embedding_fwd_col(const int* ids, const float* table, float* out,
                                           int B, int T, int t, int D) {
    dim3 block(128);
    dim3 grid(B, (D + 127) / 128);
    k_embedding_fwd_col<<<grid, block>>>(ids, table, out, B, T, t, D);
}

inline void EALRMNModel::embedding_bwd_col(const int* ids, const float* d_out, float* d_table,
                                           int B, int T, int t, int D) {
    dim3 block(128);
    dim3 grid(B, (D + 127) / 128);
    k_embedding_bwd_col<<<grid, block>>>(ids, d_out, d_table, B, T, t, D);
}

#endif // EALRMN_MODEL_EALRMN_CUH
