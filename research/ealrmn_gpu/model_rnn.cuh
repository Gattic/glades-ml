// EALRMN Phase-1 GPU prototype — RNN baseline
// Standard tanh RNN: s_t = tanh(W_h @ s_{t-1} + W_in @ z_t + b_h)
#ifndef EALRMN_MODEL_RNN_CUH
#define EALRMN_MODEL_RNN_CUH

#include "common.cuh"
#include "kernels.cuh"
#include "recurrence_kernels.cuh"

struct RNNModelCache {
    int B = 0, T = 0, m = 0, n_classes = 0;
    const int* ids = nullptr;
    Tensor z_all;     // (T, B, m)
    Tensor s_all;     // (T+1, B, m)
    Tensor logits;
    Tensor probs;
    Tensor losses;

    void alloc(int B_, int T_, int m_, int n_classes_) {
        B = B_; T = T_; m = m_; n_classes = n_classes_;
        z_all = make_tensor({T, B, m});
        s_all = make_tensor({T + 1, B, m});
        logits = make_tensor({B, n_classes});
        probs  = make_tensor({B, n_classes});
        losses = make_tensor({B});
    }
    void free() {
        free_tensor(z_all); free_tensor(s_all);
        free_tensor(logits); free_tensor(probs); free_tensor(losses);
    }
};

struct RNNModel {
    cublasHandle_t cublas;
    int V = 0, m = 0, n_classes = 0;

    Tensor E;       // (V, m)
    Tensor W_h;     // (m, m)
    Tensor W_in;    // (m, m)
    Tensor b_h;     // (m,)
    Tensor W_out;   // (m, n_classes)
    Tensor b_out;   // (n_classes,)

    Tensor dE, dW_h, dW_in, db_h, dW_out, db_out;
    Tensor mE, mW_h, mW_in, mb_h, mW_out, mb_out;
    Tensor vE, vW_h, vW_in, vb_h, vW_out, vb_out;

    std::vector<Tensor*> params() {
        return { &E, &W_h, &W_in, &b_h, &W_out, &b_out };
    }
    std::vector<Tensor*> grads() {
        return { &dE, &dW_h, &dW_in, &db_h, &dW_out, &db_out };
    }
    std::vector<Tensor*> ms() { return { &mE, &mW_h, &mW_in, &mb_h, &mW_out, &mb_out }; }
    std::vector<Tensor*> vs() { return { &vE, &vW_h, &vW_in, &vb_h, &vW_out, &vb_out }; }

    void init(cublasHandle_t handle, int V_, int m_, int n_classes_,
              unsigned long long seed) {
        cublas = handle;
        V = V_; m = m_; n_classes = n_classes_;
        HostRng rng(seed);

        E     = make_tensor({V, m});
        W_h   = make_tensor({m, m});
        W_in  = make_tensor({m, m});
        b_h   = make_tensor({m});
        W_out = make_tensor({m, n_classes});
        b_out = make_tensor({n_classes});

        dE     = make_tensor({V, m});
        dW_h   = make_tensor({m, m});
        dW_in  = make_tensor({m, m});
        db_h   = make_tensor({m});
        dW_out = make_tensor({m, n_classes});
        db_out = make_tensor({n_classes});

        mE     = make_tensor({V, m});      mE.zero_();
        mW_h   = make_tensor({m, m});      mW_h.zero_();
        mW_in  = make_tensor({m, m});      mW_in.zero_();
        mb_h   = make_tensor({m});         mb_h.zero_();
        mW_out = make_tensor({m, n_classes}); mW_out.zero_();
        mb_out = make_tensor({n_classes}); mb_out.zero_();
        vE     = make_tensor({V, m});      vE.zero_();
        vW_h   = make_tensor({m, m});      vW_h.zero_();
        vW_in  = make_tensor({m, m});      vW_in.zero_();
        vb_h   = make_tensor({m});         vb_h.zero_();
        vW_out = make_tensor({m, n_classes}); vW_out.zero_();
        vb_out = make_tensor({n_classes}); vb_out.zero_();

        init_normal(E, 0.0f, 1.0f / std::sqrt((float)m), rng);
        init_orthogonal_scale(W_h, m, 0.95f, rng);
        init_xavier_uniform(W_in, m, m, rng);
        b_h.zero_();
        init_xavier_uniform(W_out, m, n_classes, rng);
        b_out.zero_();
    }

    void free_all() {
        free_tensor(E); free_tensor(W_h); free_tensor(W_in);
        free_tensor(b_h); free_tensor(W_out); free_tensor(b_out);
        free_tensor(dE); free_tensor(dW_h); free_tensor(dW_in);
        free_tensor(db_h); free_tensor(dW_out); free_tensor(db_out);
        free_tensor(mE); free_tensor(mW_h); free_tensor(mW_in);
        free_tensor(mb_h); free_tensor(mW_out); free_tensor(mb_out);
        free_tensor(vE); free_tensor(vW_h); free_tensor(vW_in);
        free_tensor(vb_h); free_tensor(vW_out); free_tensor(vb_out);
    }

    void zero_grads() { for (Tensor* g : grads()) g->zero_(); }

    int64_t num_params() {
        int64_t n = 0;
        for (Tensor* p : params()) n += p->numel;
        return n;
    }

    void forward(const int* ids_device, int B, int T, RNNModelCache& cache) {
        cache.ids = ids_device;
        cache.B = B; cache.T = T; cache.m = m; cache.n_classes = n_classes;

        // s_0 = 0
        CUDA_CHECK(cudaMemset(cache.s_all.d, 0, B * m * sizeof(float)));
        Tensor s_prev = make_tensor({B, m});
        s_prev.zero_();
        Tensor z_t = make_tensor({B, m});
        Tensor pre_s = make_tensor({B, m});
        Tensor s_t = make_tensor({B, m});

        for (int t = 0; t < T; ++t) {
            EALRMNModel::embedding_fwd_col(ids_device, E.d, z_t.d, B, T, t, m);
            CUDA_CHECK(cudaMemcpy(cache.z_all.d + (int64_t)t * B * m, z_t.d,
                                  B * m * sizeof(float), cudaMemcpyDeviceToDevice));

            // pre_s = s_prev @ W_h^T + z_t @ W_in^T
            gemm_nt(cublas, B, m, m, 1.0f, s_prev.d, W_h.d, 0.0f, pre_s.d);
            gemm_nt(cublas, B, m, m, 1.0f, z_t.d, W_in.d, 1.0f, pre_s.d);
            launch_bias_add(pre_s.d, b_h.d, B, m);

            // s_t = tanh(pre_s)
            {
                int n = B * m;
                int block = 256;
                int grid = (n + block - 1) / block;
                k_tanh_fwd<<<grid, block>>>(pre_s.d, s_t.d, n);
            }
            CUDA_CHECK(cudaMemcpy(cache.s_all.d + (int64_t)(t + 1) * B * m, s_t.d,
                                  B * m * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(s_prev.d, s_t.d, B * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        // logits = s_T @ W_out + b_out
        const float* s_T_ptr = cache.s_all.d + (int64_t)T * B * m;
        gemm_nn(cublas, B, n_classes, m, 1.0f, s_T_ptr, W_out.d, 0.0f, cache.logits.d);
        launch_bias_add(cache.logits.d, b_out.d, B, n_classes);

        free_tensor(s_prev); free_tensor(z_t); free_tensor(pre_s); free_tensor(s_t);
    }

    float compute_loss(const int* labels_device, RNNModelCache& cache) {
        launch_softmax_ce_fwd(cache.logits.d, labels_device, cache.probs.d, cache.losses.d,
                              cache.B, cache.n_classes);
        std::vector<float> h_losses;
        copy_d2h(h_losses, cache.losses);
        float s = 0.0f;
        for (float v : h_losses) s += v;
        return s;
    }

    void backward(const int* labels_device, RNNModelCache& cache) {
        const int B = cache.B, T = cache.T;
        const float scale = 1.0f / (float)B;

        Tensor d_logits = make_tensor({B, n_classes});
        launch_softmax_ce_bwd(cache.probs.d, labels_device, d_logits.d, B, n_classes, scale);
        launch_bias_bwd(d_logits.d, db_out.d, B, n_classes);

        const float* s_T_ptr = cache.s_all.d + (int64_t)T * B * m;
        Tensor d_s = make_tensor({B, m});
        gemm_tn(cublas, m, n_classes, B, 1.0f, s_T_ptr, d_logits.d, 1.0f, dW_out.d);
        gemm_nt(cublas, B, m, n_classes, 1.0f, d_logits.d, W_out.d, 0.0f, d_s.d);

        Tensor d_pre_s = make_tensor({B, m});
        Tensor d_s_prev = make_tensor({B, m});

        for (int t = T - 1; t >= 0; --t) {
            // d_pre_s = d_s * (1 - s_t^2). Use saved s_t = s_all[t+1].
            const float* s_t_ptr = cache.s_all.d + (int64_t)(t + 1) * B * m;
            {
                int n = B * m;
                int block = 256;
                int grid = (n + block - 1) / block;
                k_tanh_bwd<<<grid, block>>>(s_t_ptr, d_s.d, d_pre_s.d, n);
            }
            // db_h += sum d_pre_s
            launch_bias_bwd(d_pre_s.d, db_h.d, B, m);

            const float* s_prev_ptr = cache.s_all.d + (int64_t)t * B * m;
            // dW_h += d_pre_s^T @ s_prev
            gemm_tn(cublas, m, m, B, 1.0f, d_pre_s.d, s_prev_ptr, 1.0f, dW_h.d);
            // d_s_prev = d_pre_s @ W_h
            gemm_nn(cublas, B, m, m, 1.0f, d_pre_s.d, W_h.d, 0.0f, d_s_prev.d);

            const float* z_t_ptr = cache.z_all.d + (int64_t)t * B * m;
            // dW_in += d_pre_s^T @ z_t
            gemm_tn(cublas, m, m, B, 1.0f, d_pre_s.d, z_t_ptr, 1.0f, dW_in.d);
            // d_z = d_pre_s @ W_in
            Tensor d_z = make_tensor({B, m});
            gemm_nn(cublas, B, m, m, 1.0f, d_pre_s.d, W_in.d, 0.0f, d_z.d);

            // Embedding bwd
            EALRMNModel::embedding_bwd_col(cache.ids, d_z.d, dE.d, B, T, t, m);

            CUDA_CHECK(cudaMemcpy(d_s.d, d_s_prev.d, B * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            free_tensor(d_z);
        }
        free_tensor(d_logits); free_tensor(d_s); free_tensor(d_pre_s); free_tensor(d_s_prev);
    }
};

#endif // EALRMN_MODEL_RNN_CUH
