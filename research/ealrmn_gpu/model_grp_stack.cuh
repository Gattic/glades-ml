// VESTA Phase-2 Gate-2B — multi-layer GRP-RNN stack with residual + MLP.
//
// Architecture:
//   x_0 = E[ids]                                      (T, B, m)
//   for ℓ = 0..L-1:
//     h = LN1_ℓ(x_ℓ)
//     h = GRP_recurrence_ℓ(h)                         per-position over t
//     x_ℓ' = x_ℓ + h                                  (residual)
//     h    = LN2_ℓ(x_ℓ')
//     h    = MLP_ℓ(h)         W1 → ReLU → W2
//     x_{ℓ+1} = x_ℓ' + h                              (residual)
//   x_final = LN_final(x_L)
//   logits[t] = x_final[t] @ W_out + b_out
//   loss = mean_t softmax_ce(logits[t], ids[t+1])     LM mode only
//
// Single-layer GRP recurrence (within one block):
//   s_0 = 0
//   for t in 0..T-1:
//     theta = tanh(W_a @ x_t + b_a) * phi_max          (B, K)
//     R_t   = prod_k Givens(theta_k)
//     s_t   = decay * R_t @ s_{t-1} + W_in @ x_t
//     (optional LN within the block, not here — outer block-LN handles it)
//     y_t   = s_t        // direct linear-state output, no tanh
//   block-output sequence: y_1..y_T  (length T)
//
// This is GRP-RNN+LN+MLP, the Gate-2B candidate. LM mode only (single-label
// classification not supported here — single-block model_grp_rnn.cuh covers that).
//
// Author: VESTA Phase 2, 2026-05-19.

#ifndef VESTA_MODEL_GRP_STACK_CUH
#define VESTA_MODEL_GRP_STACK_CUH

#include "common.cuh"
#include "kernels.cuh"
#include "model_ealrmn.cuh"   // for EALRMNModel::embedding_fwd_col / _bwd_col
#include "model_grp_rnn.cuh"  // for k_grp_angle_act_fwd/bwd, k_grp_givens_fwd/bwd, k_build_lm_labels

// =============================================================================
// MLP kernels
// =============================================================================

// y = ReLU(x), elementwise on N elements.
__global__ void k_relu_fwd(const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = fmaxf(x[i], 0.0f);
}
inline void launch_relu_fwd(const float* x, float* y, int N) {
    int block = 256, grid = (N + block - 1) / block;
    k_relu_fwd<<<grid, block>>>(x, y, N);
}

// dx = dy * I[x > 0]
__global__ void k_relu_bwd(const float* x, const float* dy, float* dx, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f;
}
inline void launch_relu_bwd(const float* x, const float* dy, float* dx, int N) {
    int block = 256, grid = (N + block - 1) / block;
    k_relu_bwd<<<grid, block>>>(x, dy, dx, N);
}

// =============================================================================
// Per-layer parameter struct
// =============================================================================

struct GRPStackLayer {
    // Per-layer GRP recurrence params.
    Tensor W_a;       // (K, m)
    Tensor b_a;       // (K,)
    Tensor W_in;      // (m, m)
    // Pre-norm LN before the recurrence.
    Tensor gamma_ln1; // (m,)
    Tensor beta_ln1;  // (m,)
    // Pre-norm LN before the MLP.
    Tensor gamma_ln2; // (m,)
    Tensor beta_ln2;  // (m,)
    // MLP params: m → 4m → m, ReLU between.
    Tensor W_mlp1;    // (m, 4m)
    Tensor b_mlp1;    // (4m,)
    Tensor W_mlp2;    // (4m, m)
    Tensor b_mlp2;    // (m,)

    Tensor dW_a, db_a, dW_in;
    Tensor dgamma_ln1, dbeta_ln1, dgamma_ln2, dbeta_ln2;
    Tensor dW_mlp1, db_mlp1, dW_mlp2, db_mlp2;

    Tensor mW_a, mb_a, mW_in, mgamma_ln1, mbeta_ln1, mgamma_ln2, mbeta_ln2;
    Tensor mW_mlp1, mb_mlp1, mW_mlp2, mb_mlp2;
    Tensor vW_a, vb_a, vW_in, vgamma_ln1, vbeta_ln1, vgamma_ln2, vbeta_ln2;
    Tensor vW_mlp1, vb_mlp1, vW_mlp2, vb_mlp2;
};

struct GRPStackCache {
    int B = 0, T = 0, m = 0, K = 0, L = 0, V = 0;
    int mlp_hidden = 0;
    const int* ids = nullptr;

    // Per-layer activations needed for backward.
    // residual_in[ℓ] (T, B, m) = input to layer ℓ (after embedding for ℓ=0).
    std::vector<Tensor> residual_in;       // L tensors
    // pre_ln1[ℓ] (T, B, m) = input to LN1 (== residual_in[ℓ]) — alias residual_in.
    // ln1_mean[ℓ], ln1_rstd[ℓ] (T, B)
    std::vector<Tensor> ln1_mean;
    std::vector<Tensor> ln1_rstd;
    // Per-step GRP recurrence cache:
    //   pre_theta[ℓ] (T, B, K), theta[ℓ] (T, B, K), s_post_rot[ℓ] (T, B, m),
    //   s_all[ℓ] (T+1, B, m), x_after_ln1[ℓ] (T, B, m) (input to recurrence)
    std::vector<Tensor> pre_theta;
    std::vector<Tensor> theta;
    std::vector<Tensor> s_post_rot;
    std::vector<Tensor> s_all;
    std::vector<Tensor> x_after_ln1;
    // Pre-MLP value: pre_ln2[ℓ] (T, B, m) = residual_in[ℓ] + s_seq[ℓ]   (the snapshot
    // captured before MLP is added; needed for LN2 backward AND as the d_pre_ln2 root
    // that gets routed back into both the GRP-recurrence and the skip branches).
    std::vector<Tensor> pre_ln2;
    // Final residual_out[ℓ] (T, B, m) = pre_ln2[ℓ] + mlp_out[ℓ]; fed to next layer.
    std::vector<Tensor> residual_out;
    // Pre-MLP LN: ln2_mean[ℓ], ln2_rstd[ℓ], pre_ln2 is the LN2 input.
    std::vector<Tensor> ln2_mean;
    std::vector<Tensor> ln2_rstd;
    // MLP cache: mlp_h_pre (T, B, 4m) before ReLU; mlp_h_act after ReLU.
    std::vector<Tensor> mlp_x_after_ln2;  // (T, B, m)
    std::vector<Tensor> mlp_h_pre;        // (T, B, 4m)
    std::vector<Tensor> mlp_h_act;        // (T, B, 4m)

    // Final LN cache.
    Tensor x_final_pre_ln;  // (T, B, m) — input to final LN
    Tensor ln_final_mean;   // (T, B)
    Tensor ln_final_rstd;   // (T, B)
    Tensor x_final;         // (T, B, m) — output of final LN, input to readout

    // LM-mode outputs.
    bool lm_mode = true;  // stack is LM-only
    Tensor logits_all;    // (T, B, V) — positions 0..T-2 used
    Tensor probs_all;
    Tensor losses_all;    // ((T-1)*B,)
    int* labels_lm = nullptr;  // (T-1, B) device int

    void alloc(int B_, int T_, int m_, int K_, int L_, int V_, int mlp_hidden_) {
        B = B_; T = T_; m = m_; K = K_; L = L_; V = V_; mlp_hidden = mlp_hidden_;
        residual_in.resize(L);
        ln1_mean.resize(L);
        ln1_rstd.resize(L);
        pre_theta.resize(L);
        theta.resize(L);
        s_post_rot.resize(L);
        s_all.resize(L);
        x_after_ln1.resize(L);
        pre_ln2.resize(L);
        residual_out.resize(L);
        ln2_mean.resize(L);
        ln2_rstd.resize(L);
        mlp_x_after_ln2.resize(L);
        mlp_h_pre.resize(L);
        mlp_h_act.resize(L);
        for (int l = 0; l < L; ++l) {
            residual_in[l]     = make_tensor({T, B, m});
            ln1_mean[l]        = make_tensor({T, B});
            ln1_rstd[l]        = make_tensor({T, B});
            pre_theta[l]       = make_tensor({T, B, K});
            theta[l]           = make_tensor({T, B, K});
            s_post_rot[l]      = make_tensor({T, B, m});
            s_all[l]           = make_tensor({T + 1, B, m});
            x_after_ln1[l]     = make_tensor({T, B, m});
            pre_ln2[l]         = make_tensor({T, B, m});
            residual_out[l]    = make_tensor({T, B, m});
            ln2_mean[l]        = make_tensor({T, B});
            ln2_rstd[l]        = make_tensor({T, B});
            mlp_x_after_ln2[l] = make_tensor({T, B, m});
            mlp_h_pre[l]       = make_tensor({T, B, mlp_hidden});
            mlp_h_act[l]       = make_tensor({T, B, mlp_hidden});
        }
        x_final_pre_ln = make_tensor({T, B, m});
        ln_final_mean  = make_tensor({T, B});
        ln_final_rstd  = make_tensor({T, B});
        x_final        = make_tensor({T, B, m});
        logits_all     = make_tensor({T, B, V});
        probs_all      = make_tensor({T, B, V});
        losses_all     = make_tensor({(T - 1) * B});
        CUDA_CHECK(cudaMalloc(&labels_lm, (T - 1) * B * sizeof(int)));
    }
    void free() {
        for (int l = 0; l < (int)residual_in.size(); ++l) {
            free_tensor(residual_in[l]);
            free_tensor(ln1_mean[l]);
            free_tensor(ln1_rstd[l]);
            free_tensor(pre_theta[l]);
            free_tensor(theta[l]);
            free_tensor(s_post_rot[l]);
            free_tensor(s_all[l]);
            free_tensor(x_after_ln1[l]);
            free_tensor(pre_ln2[l]);
            free_tensor(residual_out[l]);
            free_tensor(ln2_mean[l]);
            free_tensor(ln2_rstd[l]);
            free_tensor(mlp_x_after_ln2[l]);
            free_tensor(mlp_h_pre[l]);
            free_tensor(mlp_h_act[l]);
        }
        free_tensor(x_final_pre_ln);
        free_tensor(ln_final_mean);
        free_tensor(ln_final_rstd);
        free_tensor(x_final);
        free_tensor(logits_all);
        free_tensor(probs_all);
        free_tensor(losses_all);
        if (labels_lm) { cudaFree(labels_lm); labels_lm = nullptr; }
        residual_in.clear(); ln1_mean.clear(); ln1_rstd.clear();
        pre_theta.clear(); theta.clear(); s_post_rot.clear();
        s_all.clear(); x_after_ln1.clear();
        pre_ln2.clear(); residual_out.clear();
        ln2_mean.clear(); ln2_rstd.clear();
        mlp_x_after_ln2.clear(); mlp_h_pre.clear(); mlp_h_act.clear();
    }
};

// =============================================================================
// Stack model
// =============================================================================

struct GRPStackModel {
    cublasHandle_t cublas;
    int V = 0, m = 0, K = 0, L = 0;
    int mlp_hidden = 0;
    float phi_max = 1.5707963f;
    float decay = 0.95f;
    int stride = 3;
    // --linear-recurrence: force theta = 0 always (R_t = I, no Givens rotation).
    // Reduces the recurrence to a scalar-decay linear RNN: s_t = decay*s_{t-1} + W_in @ x_t.
    // This is the B5 baseline (linear-RNN+orthogonal floor) from the VESTA Phase-2 brief.
    bool linear_recurrence = false;

    std::vector<GRPStackLayer> layers;
    // Shared E and readout.
    Tensor E;         // (V, m)
    Tensor gamma_lnf; // (m,) final LN
    Tensor beta_lnf;  // (m,)
    Tensor W_out;     // (m, V)
    Tensor b_out;     // (V,)

    Tensor dE, dgamma_lnf, dbeta_lnf, dW_out, db_out;
    Tensor mE, mgamma_lnf, mbeta_lnf, mW_out, mb_out;
    Tensor vE, vgamma_lnf, vbeta_lnf, vW_out, vb_out;

    // Plane geometry.
    Tensor plane_p_dev;
    Tensor plane_q_dev;

    std::vector<Tensor*> params() {
        std::vector<Tensor*> v;
        v.push_back(&E);
        for (int l = 0; l < L; ++l) {
            v.push_back(&layers[l].gamma_ln1); v.push_back(&layers[l].beta_ln1);
            v.push_back(&layers[l].W_a);       v.push_back(&layers[l].b_a);
            v.push_back(&layers[l].W_in);
            v.push_back(&layers[l].gamma_ln2); v.push_back(&layers[l].beta_ln2);
            v.push_back(&layers[l].W_mlp1);    v.push_back(&layers[l].b_mlp1);
            v.push_back(&layers[l].W_mlp2);    v.push_back(&layers[l].b_mlp2);
        }
        v.push_back(&gamma_lnf); v.push_back(&beta_lnf);
        v.push_back(&W_out);     v.push_back(&b_out);
        return v;
    }
    std::vector<Tensor*> grads() {
        std::vector<Tensor*> v;
        v.push_back(&dE);
        for (int l = 0; l < L; ++l) {
            v.push_back(&layers[l].dgamma_ln1); v.push_back(&layers[l].dbeta_ln1);
            v.push_back(&layers[l].dW_a);       v.push_back(&layers[l].db_a);
            v.push_back(&layers[l].dW_in);
            v.push_back(&layers[l].dgamma_ln2); v.push_back(&layers[l].dbeta_ln2);
            v.push_back(&layers[l].dW_mlp1);    v.push_back(&layers[l].db_mlp1);
            v.push_back(&layers[l].dW_mlp2);    v.push_back(&layers[l].db_mlp2);
        }
        v.push_back(&dgamma_lnf); v.push_back(&dbeta_lnf);
        v.push_back(&dW_out);     v.push_back(&db_out);
        return v;
    }
    std::vector<Tensor*> ms() {
        std::vector<Tensor*> v;
        v.push_back(&mE);
        for (int l = 0; l < L; ++l) {
            v.push_back(&layers[l].mgamma_ln1); v.push_back(&layers[l].mbeta_ln1);
            v.push_back(&layers[l].mW_a);       v.push_back(&layers[l].mb_a);
            v.push_back(&layers[l].mW_in);
            v.push_back(&layers[l].mgamma_ln2); v.push_back(&layers[l].mbeta_ln2);
            v.push_back(&layers[l].mW_mlp1);    v.push_back(&layers[l].mb_mlp1);
            v.push_back(&layers[l].mW_mlp2);    v.push_back(&layers[l].mb_mlp2);
        }
        v.push_back(&mgamma_lnf); v.push_back(&mbeta_lnf);
        v.push_back(&mW_out);     v.push_back(&mb_out);
        return v;
    }
    std::vector<Tensor*> vs() {
        std::vector<Tensor*> v;
        v.push_back(&vE);
        for (int l = 0; l < L; ++l) {
            v.push_back(&layers[l].vgamma_ln1); v.push_back(&layers[l].vbeta_ln1);
            v.push_back(&layers[l].vW_a);       v.push_back(&layers[l].vb_a);
            v.push_back(&layers[l].vW_in);
            v.push_back(&layers[l].vgamma_ln2); v.push_back(&layers[l].vbeta_ln2);
            v.push_back(&layers[l].vW_mlp1);    v.push_back(&layers[l].vb_mlp1);
            v.push_back(&layers[l].vW_mlp2);    v.push_back(&layers[l].vb_mlp2);
        }
        v.push_back(&vgamma_lnf); v.push_back(&vbeta_lnf);
        v.push_back(&vW_out);     v.push_back(&vb_out);
        return v;
    }

    int64_t num_params() {
        int64_t n = 0;
        for (Tensor* p : params()) n += p->numel;
        return n;
    }

    void zero_grads() {
        for (Tensor* g : grads()) g->zero_();
    }

    void init(cublasHandle_t handle, int V_, int m_, int K_, int L_, int mlp_hidden_,
              unsigned long long seed) {
        cublas = handle;
        V = V_; m = m_; K = K_; L = L_; mlp_hidden = mlp_hidden_;
        HostRng rng(seed);

        E = make_tensor({V, m});
        init_normal(E, 0.0f, 1.0f / std::sqrt((float)m), rng);
        dE = make_tensor({V, m}); dE.zero_();
        mE = make_tensor({V, m}); mE.zero_();
        vE = make_tensor({V, m}); vE.zero_();

        // Final LN.
        gamma_lnf = make_tensor({m});
        beta_lnf  = make_tensor({m});
        std::vector<float> ones(m, 1.0f);
        copy_h2d(gamma_lnf, ones);
        beta_lnf.zero_();
        dgamma_lnf = make_tensor({m}); dgamma_lnf.zero_();
        dbeta_lnf  = make_tensor({m}); dbeta_lnf.zero_();
        mgamma_lnf = make_tensor({m}); mgamma_lnf.zero_();
        mbeta_lnf  = make_tensor({m}); mbeta_lnf.zero_();
        vgamma_lnf = make_tensor({m}); vgamma_lnf.zero_();
        vbeta_lnf  = make_tensor({m}); vbeta_lnf.zero_();

        // Readout.
        W_out = make_tensor({m, V});
        init_xavier_uniform(W_out, m, V, rng);
        b_out = make_tensor({V}); b_out.zero_();
        dW_out = make_tensor({m, V}); dW_out.zero_();
        db_out = make_tensor({V});    db_out.zero_();
        mW_out = make_tensor({m, V}); mW_out.zero_();
        mb_out = make_tensor({V});    mb_out.zero_();
        vW_out = make_tensor({m, V}); vW_out.zero_();
        vb_out = make_tensor({V});    vb_out.zero_();

        layers.resize(L);
        for (int l = 0; l < L; ++l) {
            GRPStackLayer& layer = layers[l];
            layer.W_a   = make_tensor({K, m});
            init_normal(layer.W_a, 0.0f, 1.0f / std::sqrt((float)m), rng);
            layer.b_a   = make_tensor({K}); layer.b_a.zero_();
            layer.W_in  = make_tensor({m, m});
            init_xavier_uniform(layer.W_in, m, m, rng);

            layer.gamma_ln1 = make_tensor({m});
            copy_h2d(layer.gamma_ln1, ones);
            layer.beta_ln1  = make_tensor({m}); layer.beta_ln1.zero_();
            layer.gamma_ln2 = make_tensor({m});
            copy_h2d(layer.gamma_ln2, ones);
            layer.beta_ln2  = make_tensor({m}); layer.beta_ln2.zero_();

            // MLP init: Xavier on each linear, biases zero. (Zero-init on W_mlp2 was tried
            // 2026-05-19 as a transformer-style "each block starts as identity" mitigation
            // for the no-depth-gain pathology; it did NOT unlock depth and made all configs
            // slightly worse — see VESTA_GATE2B_RESULT.md.)
            layer.W_mlp1 = make_tensor({m, mlp_hidden});
            init_xavier_uniform(layer.W_mlp1, m, mlp_hidden, rng);
            layer.b_mlp1 = make_tensor({mlp_hidden}); layer.b_mlp1.zero_();
            layer.W_mlp2 = make_tensor({mlp_hidden, m});
            init_xavier_uniform(layer.W_mlp2, mlp_hidden, m, rng);
            layer.b_mlp2 = make_tensor({m}); layer.b_mlp2.zero_();

            layer.dW_a   = make_tensor({K, m}); layer.dW_a.zero_();
            layer.db_a   = make_tensor({K});    layer.db_a.zero_();
            layer.dW_in  = make_tensor({m, m}); layer.dW_in.zero_();
            layer.dgamma_ln1 = make_tensor({m}); layer.dgamma_ln1.zero_();
            layer.dbeta_ln1  = make_tensor({m}); layer.dbeta_ln1.zero_();
            layer.dgamma_ln2 = make_tensor({m}); layer.dgamma_ln2.zero_();
            layer.dbeta_ln2  = make_tensor({m}); layer.dbeta_ln2.zero_();
            layer.dW_mlp1 = make_tensor({m, mlp_hidden}); layer.dW_mlp1.zero_();
            layer.db_mlp1 = make_tensor({mlp_hidden});    layer.db_mlp1.zero_();
            layer.dW_mlp2 = make_tensor({mlp_hidden, m}); layer.dW_mlp2.zero_();
            layer.db_mlp2 = make_tensor({m});             layer.db_mlp2.zero_();

            layer.mW_a       = make_tensor({K, m}); layer.mW_a.zero_();
            layer.vW_a       = make_tensor({K, m}); layer.vW_a.zero_();
            layer.mb_a       = make_tensor({K});    layer.mb_a.zero_();
            layer.vb_a       = make_tensor({K});    layer.vb_a.zero_();
            layer.mW_in      = make_tensor({m, m}); layer.mW_in.zero_();
            layer.vW_in      = make_tensor({m, m}); layer.vW_in.zero_();
            layer.mgamma_ln1 = make_tensor({m});    layer.mgamma_ln1.zero_();
            layer.vgamma_ln1 = make_tensor({m});    layer.vgamma_ln1.zero_();
            layer.mbeta_ln1  = make_tensor({m});    layer.mbeta_ln1.zero_();
            layer.vbeta_ln1  = make_tensor({m});    layer.vbeta_ln1.zero_();
            layer.mgamma_ln2 = make_tensor({m});    layer.mgamma_ln2.zero_();
            layer.vgamma_ln2 = make_tensor({m});    layer.vgamma_ln2.zero_();
            layer.mbeta_ln2  = make_tensor({m});    layer.mbeta_ln2.zero_();
            layer.vbeta_ln2  = make_tensor({m});    layer.vbeta_ln2.zero_();
            layer.mW_mlp1    = make_tensor({m, mlp_hidden}); layer.mW_mlp1.zero_();
            layer.vW_mlp1    = make_tensor({m, mlp_hidden}); layer.vW_mlp1.zero_();
            layer.mb_mlp1    = make_tensor({mlp_hidden});    layer.mb_mlp1.zero_();
            layer.vb_mlp1    = make_tensor({mlp_hidden});    layer.vb_mlp1.zero_();
            layer.mW_mlp2    = make_tensor({mlp_hidden, m}); layer.mW_mlp2.zero_();
            layer.vW_mlp2    = make_tensor({mlp_hidden, m}); layer.vW_mlp2.zero_();
            layer.mb_mlp2    = make_tensor({m});             layer.mb_mlp2.zero_();
            layer.vb_mlp2    = make_tensor({m});             layer.vb_mlp2.zero_();
        }

        // Plane geometry (shared across layers).
        std::vector<int> hp(K), hq(K);
        for (int k = 0; k < K; ++k) {
            hp[k] = k % m;
            hq[k] = (k + stride) % m;
        }
        Tensor pp; pp.shape = {K}; pp.numel = K;
        CUDA_CHECK(cudaMalloc(&pp.d, K * sizeof(int)));
        Tensor pq; pq.shape = {K}; pq.numel = K;
        CUDA_CHECK(cudaMalloc(&pq.d, K * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(pp.d, hp.data(), K * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pq.d, hq.data(), K * sizeof(int), cudaMemcpyHostToDevice));
        plane_p_dev = pp;
        plane_q_dev = pq;
    }

    void free_all() {
        free_tensor(E);
        free_tensor(gamma_lnf); free_tensor(beta_lnf);
        free_tensor(W_out); free_tensor(b_out);
        free_tensor(dE);
        free_tensor(dgamma_lnf); free_tensor(dbeta_lnf);
        free_tensor(dW_out); free_tensor(db_out);
        free_tensor(mE); free_tensor(mgamma_lnf); free_tensor(mbeta_lnf);
        free_tensor(mW_out); free_tensor(mb_out);
        free_tensor(vE); free_tensor(vgamma_lnf); free_tensor(vbeta_lnf);
        free_tensor(vW_out); free_tensor(vb_out);
        for (int l = 0; l < L; ++l) {
            GRPStackLayer& lay = layers[l];
            free_tensor(lay.W_a); free_tensor(lay.b_a); free_tensor(lay.W_in);
            free_tensor(lay.gamma_ln1); free_tensor(lay.beta_ln1);
            free_tensor(lay.gamma_ln2); free_tensor(lay.beta_ln2);
            free_tensor(lay.W_mlp1); free_tensor(lay.b_mlp1);
            free_tensor(lay.W_mlp2); free_tensor(lay.b_mlp2);
            free_tensor(lay.dW_a); free_tensor(lay.db_a); free_tensor(lay.dW_in);
            free_tensor(lay.dgamma_ln1); free_tensor(lay.dbeta_ln1);
            free_tensor(lay.dgamma_ln2); free_tensor(lay.dbeta_ln2);
            free_tensor(lay.dW_mlp1); free_tensor(lay.db_mlp1);
            free_tensor(lay.dW_mlp2); free_tensor(lay.db_mlp2);
            free_tensor(lay.mW_a); free_tensor(lay.mb_a); free_tensor(lay.mW_in);
            free_tensor(lay.mgamma_ln1); free_tensor(lay.mbeta_ln1);
            free_tensor(lay.mgamma_ln2); free_tensor(lay.mbeta_ln2);
            free_tensor(lay.mW_mlp1); free_tensor(lay.mb_mlp1);
            free_tensor(lay.mW_mlp2); free_tensor(lay.mb_mlp2);
            free_tensor(lay.vW_a); free_tensor(lay.vb_a); free_tensor(lay.vW_in);
            free_tensor(lay.vgamma_ln1); free_tensor(lay.vbeta_ln1);
            free_tensor(lay.vgamma_ln2); free_tensor(lay.vbeta_ln2);
            free_tensor(lay.vW_mlp1); free_tensor(lay.vb_mlp1);
            free_tensor(lay.vW_mlp2); free_tensor(lay.vb_mlp2);
        }
        if (plane_p_dev.d) cudaFree(plane_p_dev.d);
        if (plane_q_dev.d) cudaFree(plane_q_dev.d);
    }

    // ====== Forward ======
    // x_in (T, B, m) is the layer's pre-LN input. Stores all per-step caches.
    // Writes the layer's post-residual output to residual_post (handled by caller via cache.residual_post[l]).
    // After this call:
    //   cache.x_after_ln1[l] (T, B, m) holds LN(x_in)
    //   cache.pre_theta[l], cache.theta[l] (T, B, K) hold angle pre/post-act
    //   cache.s_all[l] (T+1, B, m): s_all[0]=0, s_all[1..T] = recurrence states
    //   cache.s_post_rot[l] (T, B, m): pre-decay rotated state per step (for Givens bwd)
    //   cache.residual_post[l] (T, B, m) = x_in + s_seq (where s_seq[t] = s_all[t+1])
    void forward_recurrence_block(GRPStackLayer& layer, GRPStackCache& cache, int l) {
        int B = cache.B, T = cache.T;
        // 1. LN1 over residual_in[l] → x_after_ln1[l]   (treats as N=T*B rows of m)
        const float* x_in = cache.residual_in[l].d;
        float* x_ln = cache.x_after_ln1[l].d;
        int N_rows = T * B;
        launch_layer_norm_fwd(x_in, layer.gamma_ln1.d, layer.beta_ln1.d,
                              x_ln, cache.ln1_mean[l].d, cache.ln1_rstd[l].d,
                              N_rows, m);

        // 2. pre_theta = x_ln @ W_a^T + b_a   (NxK). Skipped for linear-recurrence baseline.
        if (linear_recurrence) {
            CUDA_CHECK(cudaMemset(cache.pre_theta[l].d, 0, (int64_t)N_rows * K * sizeof(float)));
            CUDA_CHECK(cudaMemset(cache.theta[l].d, 0, (int64_t)N_rows * K * sizeof(float)));
        } else {
            gemm_nt(cublas, N_rows, K, m, 1.0f, x_ln, layer.W_a.d, 0.0f, cache.pre_theta[l].d);
            launch_bias_add(cache.pre_theta[l].d, layer.b_a.d, N_rows, K);

            // 3. theta = tanh(pre_theta) * phi_max  (elementwise on T*B*K entries).
            {
                int n = N_rows * K;
                int block = 256, grid = (n + block - 1) / block;
                k_grp_angle_act_fwd<<<grid, block>>>(cache.pre_theta[l].d, cache.theta[l].d,
                                                     N_rows, K, phi_max);
            }
        }

        // 4. Per-step recurrence: s_t = decay * R_t @ s_{t-1} + W_in @ x_ln[t]
        //    Save s_post_rot[t] = R_t @ s_{t-1} (pre-decay) per step.
        CUDA_CHECK(cudaMemset(cache.s_all[l].d, 0, B * m * sizeof(float)));  // s_0 = 0
        Tensor s_rot_t = make_tensor({B, m});
        Tensor s_t     = make_tensor({B, m});
        for (int t = 0; t < T; ++t) {
            const float* s_prev = cache.s_all[l].d + (int64_t)t * B * m;
            const float* x_ln_t   = x_ln + (int64_t)t * B * m;
            const float* theta_t  = cache.theta[l].d + (int64_t)t * B * K;
            float* s_post_rot_t   = cache.s_post_rot[l].d + (int64_t)t * B * m;

            // s_rot = R_t @ s_prev   (Givens sequential, B blocks)
            {
                int block = 32;
                k_grp_givens_fwd<<<B, block>>>(s_prev, theta_t,
                                               (int*)plane_p_dev.d, (int*)plane_q_dev.d,
                                               s_rot_t.d, B, m, K);
            }
            // Save pre-decay rotated state.
            CUDA_CHECK(cudaMemcpy(s_post_rot_t, s_rot_t.d, B * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            // Apply decay.
            if (decay != 1.0f) {
                int n = B * m;
                CUBLAS_CHECK(cublasSscal(cublas, n, &decay, s_rot_t.d, 1));
            }
            // s_t = s_rot + x_ln_t @ W_in^T   (in row-vector form: s_t = s_rot + W_in @ x_ln_t)
            CUDA_CHECK(cudaMemcpy(s_t.d, s_rot_t.d, B * m * sizeof(float), cudaMemcpyDeviceToDevice));
            gemm_nt(cublas, B, m, m, 1.0f, x_ln_t, layer.W_in.d, 1.0f, s_t.d);

            // Save s_t = s_all[l][t+1]
            CUDA_CHECK(cudaMemcpy(cache.s_all[l].d + (int64_t)(t + 1) * B * m, s_t.d,
                                  B * m * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        free_tensor(s_rot_t); free_tensor(s_t);

        // 5. pre_ln2[l] = residual_in[l] + s_seq[l]  (where s_seq[t] = s_all[t+1])
        //    This is the value u that LN2 reads from. It is preserved (not overwritten) so
        //    the backward through LN2 can use it.
        const float* s_seq = cache.s_all[l].d + (int64_t)B * m;  // skip s_0
        CUDA_CHECK(cudaMemcpy(cache.pre_ln2[l].d, cache.residual_in[l].d,
                              N_rows * m * sizeof(float), cudaMemcpyDeviceToDevice));
        {
            int n = N_rows * m;
            float one = 1.0f;
            CUBLAS_CHECK(cublasSaxpy(cublas, n, &one, s_seq, 1, cache.pre_ln2[l].d, 1));
        }
    }

    // MLP forward: input is cache.pre_ln2[l]. Writes residual_out[l] = pre_ln2 + MLP(LN2(pre_ln2)).
    void forward_mlp_block(GRPStackLayer& layer, GRPStackCache& cache, int l) {
        int B = cache.B, T = cache.T;
        int N_rows = T * B;
        // 1. LN2 over pre_ln2[l] → mlp_x_after_ln2[l]
        launch_layer_norm_fwd(cache.pre_ln2[l].d, layer.gamma_ln2.d, layer.beta_ln2.d,
                              cache.mlp_x_after_ln2[l].d,
                              cache.ln2_mean[l].d, cache.ln2_rstd[l].d, N_rows, m);
        // 2. h_pre = x_after_ln2 @ W_mlp1 + b_mlp1  (N, 4m)
        gemm_nn(cublas, N_rows, mlp_hidden, m, 1.0f,
                cache.mlp_x_after_ln2[l].d, layer.W_mlp1.d, 0.0f, cache.mlp_h_pre[l].d);
        launch_bias_add(cache.mlp_h_pre[l].d, layer.b_mlp1.d, N_rows, mlp_hidden);
        // 3. h_act = ReLU(h_pre)
        launch_relu_fwd(cache.mlp_h_pre[l].d, cache.mlp_h_act[l].d, N_rows * mlp_hidden);
        // 4. residual_out[l] = pre_ln2[l]; then residual_out += h_act @ W_mlp2 + b_mlp2.
        CUDA_CHECK(cudaMemcpy(cache.residual_out[l].d, cache.pre_ln2[l].d,
                              N_rows * m * sizeof(float), cudaMemcpyDeviceToDevice));
        // residual_out += h_act @ W_mlp2  (gemm_nn beta=1 accumulates).
        gemm_nn(cublas, N_rows, m, mlp_hidden, 1.0f, cache.mlp_h_act[l].d, layer.W_mlp2.d, 1.0f,
                cache.residual_out[l].d);
        // residual_out += b_mlp2 (elementwise broadcast).
        launch_bias_add(cache.residual_out[l].d, layer.b_mlp2.d, N_rows, m);
    }

    void forward(const int* ids_device, int B, int T, GRPStackCache& cache) {
        cache.ids = ids_device;
        cache.B = B; cache.T = T;

        // 1. Embed ids into residual_in[0] (T, B, m). Use embedding_fwd_col per t.
        for (int t = 0; t < T; ++t) {
            EALRMNModel::embedding_fwd_col(
                ids_device, E.d,
                cache.residual_in[0].d + (int64_t)t * B * m,
                B, T, t, m);
        }

        // 2. For each layer ℓ, compute the post-residual sequence and copy to next layer's input.
        for (int l = 0; l < L; ++l) {
            forward_recurrence_block(layers[l], cache, l);   // writes pre_ln2[l]
            forward_mlp_block(layers[l], cache, l);          // writes residual_out[l] = pre_ln2 + MLP
            // If not last layer, residual_in[l+1] = residual_out[l].
            if (l + 1 < L) {
                CUDA_CHECK(cudaMemcpy(cache.residual_in[l + 1].d, cache.residual_out[l].d,
                                      (int64_t)T * B * m * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }

        // 3. x_final_pre_ln = residual_out[L-1]. Apply final LN.
        CUDA_CHECK(cudaMemcpy(cache.x_final_pre_ln.d, cache.residual_out[L - 1].d,
                              (int64_t)T * B * m * sizeof(float), cudaMemcpyDeviceToDevice));
        int N_rows = T * B;
        launch_layer_norm_fwd(cache.x_final_pre_ln.d, gamma_lnf.d, beta_lnf.d,
                              cache.x_final.d, cache.ln_final_mean.d, cache.ln_final_rstd.d,
                              N_rows, m);

        // 4. Logits over (T-1)*B used rows. Compute all T rows then ignore last (cheaper than gather).
        int M = T * B;
        gemm_nn(cublas, M, V, m, 1.0f, cache.x_final.d, W_out.d, 0.0f, cache.logits_all.d);
        launch_bias_add(cache.logits_all.d, b_out.d, M, V);
    }

    // ====== Loss (LM only) ======
    float compute_loss(const int* /*labels_d_unused*/, GRPStackCache& cache) {
        int M_lm = (cache.T - 1) * cache.B;
        // Note: forward computed logits for ALL T positions. We only use the first T-1 rows
        // (which correspond to predicting positions 1..T-1 from contexts 0..T-2). The rows are
        // stored contiguously as (T*B, V) with row r = t*B + b — so the first (T-1)*B rows are
        // exactly the right slice.
        launch_softmax_ce_fwd(cache.logits_all.d, cache.labels_lm,
                              cache.probs_all.d, cache.losses_all.d, M_lm, V);
        std::vector<float> h_losses;
        copy_d2h(h_losses, cache.losses_all);
        float s = 0.0f;
        for (float v : h_losses) s += v;
        return s;  // sum over M_lm tokens; caller divides
    }

    // ====== Backward ======
    void backward(const int* /*labels_d_unused*/, GRPStackCache& cache) {
        int B = cache.B, T = cache.T;
        int N_rows = T * B;
        int M_lm = (T - 1) * B;
        float scale = 1.0f / (float)B;  // matches gradcheck convention (loss/B)

        // 1. softmax_ce backward → d_logits_all (M_lm, V). Rows T-1..T-1 (last B rows of T*B)
        //    are NOT in M_lm; we need to zero their d_logits contribution. Approach:
        //    allocate (T*B, V), call softmax_ce_bwd on first M_lm rows only, then zero the last B rows.
        Tensor d_logits_all = make_tensor({T, B, V});
        CUDA_CHECK(cudaMemset(d_logits_all.d, 0, (int64_t)N_rows * V * sizeof(float)));
        launch_softmax_ce_bwd(cache.probs_all.d, cache.labels_lm,
                              d_logits_all.d, M_lm, V, scale);
        // The last B rows of d_logits_all stay zero — they don't get a gradient from LM loss.

        // 2. dW_out += x_final^T @ d_logits_all   (m, V)
        gemm_tn(cublas, m, V, N_rows, 1.0f, cache.x_final.d, d_logits_all.d, 1.0f, dW_out.d);
        launch_bias_bwd(d_logits_all.d, db_out.d, N_rows, V);

        // 3. d_x_final = d_logits_all @ W_out^T  (N_rows, m)
        Tensor d_x_final = make_tensor({T, B, m});
        gemm_nt(cublas, N_rows, m, V, 1.0f, d_logits_all.d, W_out.d, 0.0f, d_x_final.d);
        free_tensor(d_logits_all);

        // 4. final-LN backward → d_x_final_pre_ln
        Tensor d_x_final_pre_ln = make_tensor({T, B, m});
        launch_layer_norm_bwd(cache.x_final_pre_ln.d, d_x_final.d, gamma_lnf.d,
                              cache.ln_final_mean.d, cache.ln_final_rstd.d,
                              d_x_final_pre_ln.d, dgamma_lnf.d, dbeta_lnf.d,
                              N_rows, m);
        free_tensor(d_x_final);

        // 5. d_residual_out[L-1] = d_x_final_pre_ln. Then descend through layers in reverse.
        //    d_residual_out accumulator: at the top of each layer-l backward, it holds
        //    d(loss)/d residual_out[l] (= d(loss)/d residual_in[l+1] for l < L-1, or = d
        //    x_final_pre_ln for l = L-1).
        Tensor d_residual_out = make_tensor({T, B, m});
        CUDA_CHECK(cudaMemcpy(d_residual_out.d, d_x_final_pre_ln.d,
                              (int64_t)N_rows * m * sizeof(float), cudaMemcpyDeviceToDevice));
        free_tensor(d_x_final_pre_ln);

        for (int l = L - 1; l >= 0; --l) {
            GRPStackLayer& layer = layers[l];
            // Forward in this layer:
            //   pre_ln2[l] = residual_in[l] + s_seq[l]                 (skip-pre-MLP)
            //   residual_out[l] = pre_ln2[l] + (h_act @ W_mlp2 + b_mlp2)   (post-MLP)
            //
            // Backward through the two-branch residual_out = pre_ln2 + mlp_out:
            //   d_pre_ln2 = d_residual_out  (skip path)  + d_pre_ln2_via_ln2 (MLP path through LN2)
            //   d_mlp_out = d_residual_out               (the "add" path to mlp_out)

            // (a) MLP-branch backward starting from d_mlp_out = d_residual_out.
            // dW_mlp2 += mlp_h_act^T @ d_residual_out  (mlp_hidden, m)
            gemm_tn(cublas, mlp_hidden, m, N_rows, 1.0f,
                    cache.mlp_h_act[l].d, d_residual_out.d, 1.0f, layer.dW_mlp2.d);
            launch_bias_bwd(d_residual_out.d, layer.db_mlp2.d, N_rows, m);
            // d_h_act = d_residual_out @ W_mlp2^T   (N, mlp_hidden)
            Tensor d_h_act = make_tensor({N_rows, mlp_hidden});
            gemm_nt(cublas, N_rows, mlp_hidden, m, 1.0f,
                    d_residual_out.d, layer.W_mlp2.d, 0.0f, d_h_act.d);
            // d_h_pre = d_h_act * I[h_pre > 0]
            Tensor d_h_pre = make_tensor({N_rows, mlp_hidden});
            launch_relu_bwd(cache.mlp_h_pre[l].d, d_h_act.d, d_h_pre.d, N_rows * mlp_hidden);
            free_tensor(d_h_act);
            // dW_mlp1 += mlp_x_after_ln2^T @ d_h_pre  (m, mlp_hidden)
            gemm_tn(cublas, m, mlp_hidden, N_rows, 1.0f,
                    cache.mlp_x_after_ln2[l].d, d_h_pre.d, 1.0f, layer.dW_mlp1.d);
            launch_bias_bwd(d_h_pre.d, layer.db_mlp1.d, N_rows, mlp_hidden);
            // d_x_after_ln2 = d_h_pre @ W_mlp1^T   (N, m)
            Tensor d_x_after_ln2 = make_tensor({N_rows, m});
            gemm_nt(cublas, N_rows, m, mlp_hidden, 1.0f,
                    d_h_pre.d, layer.W_mlp1.d, 0.0f, d_x_after_ln2.d);
            free_tensor(d_h_pre);

            // LN2 backward: input was pre_ln2[l], so dgamma_ln2/dbeta_ln2 read from pre_ln2[l].
            Tensor d_pre_ln2_via_ln2 = make_tensor({N_rows, m});
            launch_layer_norm_bwd(cache.pre_ln2[l].d, d_x_after_ln2.d, layer.gamma_ln2.d,
                                  cache.ln2_mean[l].d, cache.ln2_rstd[l].d,
                                  d_pre_ln2_via_ln2.d, layer.dgamma_ln2.d, layer.dbeta_ln2.d,
                                  N_rows, m);
            free_tensor(d_x_after_ln2);

            // Total d_pre_ln2 = d_residual_out (skip path) + d_pre_ln2_via_ln2.
            // We overwrite d_residual_out in place into d_pre_ln2.
            Tensor& d_pre_ln2 = d_residual_out;  // alias for clarity
            {
                int n = N_rows * m;
                float one = 1.0f;
                CUBLAS_CHECK(cublasSaxpy(cublas, n, &one, d_pre_ln2_via_ln2.d, 1,
                                         d_pre_ln2.d, 1));
            }
            free_tensor(d_pre_ln2_via_ln2);

            // Now d_pre_ln2 holds d(loss)/d(residual_in[l] + s_seq[l]).

            // (b) GRP recurrence branch backward.
            //   pre_ln2 = residual_in[l] + s_seq[l]
            //   d_residual_in[l] += d_pre_ln2 (skip)
            //   d_s_seq[l]       += d_pre_ln2 (recurrence output gradient)

            // d_s_seq[l] (T, B, m) — gradient w.r.t. s_all[1..T] (the recurrence outputs).
            Tensor d_s_seq = make_tensor({T, B, m});
            CUDA_CHECK(cudaMemcpy(d_s_seq.d, d_pre_ln2.d,
                                  (int64_t)N_rows * m * sizeof(float), cudaMemcpyDeviceToDevice));
            // d_residual_in[l] starts at d_pre_ln2 (skip), then LN1 backward accumulates the
            // input-to-recurrence gradient (via x_after_ln1) into it.  We'll reuse d_pre_ln2's
            // storage as d_residual_in.

            // Per-step recurrence backward.
            // d_s holds gradient w.r.t. s_t at iter t. At the START of iter t (just before processing
            // step t backward), d_s = gradient flowing FROM future steps + d_s_seq[t].
            Tensor d_s = make_tensor({B, m}); d_s.zero_();
            Tensor d_s_rot = make_tensor({B, m});
            Tensor d_x_ln_t = make_tensor({B, m});
            Tensor d_theta = make_tensor({B, K});
            Tensor d_pre_theta_t = make_tensor({B, K});
            Tensor s_workspace = make_tensor({B, m});
            Tensor theta_t_tmp = make_tensor({B, K});
            // d_x_after_ln1[l] accumulator (T, B, m). We'll backprop LN1 outside the loop.
            Tensor d_x_after_ln1 = make_tensor({T, B, m});
            d_x_after_ln1.zero_();

            for (int t = T - 1; t >= 0; --t) {
                // Inject d_s_seq[t] (= d(loss)/d s_all[t+1] via direct skip from residual_post).
                {
                    int n = B * m;
                    float one = 1.0f;
                    const float* d_s_seq_t = d_s_seq.d + (int64_t)t * B * m;
                    CUBLAS_CHECK(cublasSaxpy(cublas, n, &one, d_s_seq_t, 1, d_s.d, 1));
                }
                const float* x_ln_t = cache.x_after_ln1[l].d + (int64_t)t * B * m;
                const float* theta_t_ptr = cache.theta[l].d + (int64_t)t * B * K;
                const float* s_post_rot_t_ptr = cache.s_post_rot[l].d + (int64_t)t * B * m;

                // d_pre_tanh = d_s (no tanh on state in stack model; we go straight)
                Tensor d_pre_tanh = d_s;  // alias

                // d_s_rot_pre_decay = d_pre_tanh * decay
                CUDA_CHECK(cudaMemcpy(d_s_rot.d, d_pre_tanh.d, B * m * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
                if (decay != 1.0f) {
                    int n = B * m;
                    CUBLAS_CHECK(cublasSscal(cublas, n, &decay, d_s_rot.d, 1));
                }

                // dW_in += d_pre_tanh^T @ x_ln_t   (m, m)
                gemm_tn(cublas, m, m, B, 1.0f, d_pre_tanh.d, x_ln_t, 1.0f, layer.dW_in.d);
                // d_x_ln_from_Win = d_pre_tanh @ W_in   (B, m)
                gemm_nn(cublas, B, m, m, 1.0f, d_pre_tanh.d, layer.W_in.d, 0.0f, d_x_ln_t.d);

                // Givens backward: d_theta accumulator starts zero.
                d_theta.zero_();
                CUDA_CHECK(cudaMemcpy(theta_t_tmp.d, theta_t_ptr, B * K * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
                {
                    int block = 32;
                    k_grp_givens_bwd<<<B, block>>>(s_post_rot_t_ptr, theta_t_tmp.d,
                                                   (int*)plane_p_dev.d, (int*)plane_q_dev.d,
                                                   s_workspace.d, d_s_rot.d, d_theta.d,
                                                   B, m, K);
                }
                // After this, d_s_rot holds d_s_prev (the gradient w.r.t. s_{t-1}).

                // theta = tanh(pre_theta)*phi_max backward. Skipped for linear-recurrence
                // baseline (W_a/b_a have no gradient contribution since theta is forced to 0).
                if (!linear_recurrence) {
                    int n = B * K;
                    int block = 256, grid = (n + block - 1) / block;
                    k_grp_angle_act_bwd<<<grid, block>>>(theta_t_tmp.d, d_theta.d, d_pre_theta_t.d,
                                                         B, K, phi_max);
                    // db_a += sum d_pre_theta_t  (across B per K)
                    launch_bias_bwd(d_pre_theta_t.d, layer.db_a.d, B, K);
                    // dW_a += d_pre_theta_t^T @ x_ln_t  (K, m)
                    gemm_tn(cublas, K, m, B, 1.0f, d_pre_theta_t.d, x_ln_t, 1.0f, layer.dW_a.d);
                    // d_x_ln_from_Wa = d_pre_theta_t @ W_a   (B, m)
                    gemm_nn(cublas, B, m, K, 1.0f, d_pre_theta_t.d, layer.W_a.d, 1.0f, d_x_ln_t.d);
                }

                // Accumulate into d_x_after_ln1[l][t]
                float* d_x_after_ln1_t = d_x_after_ln1.d + (int64_t)t * B * m;
                {
                    int n = B * m;
                    float one = 1.0f;
                    CUBLAS_CHECK(cublasSaxpy(cublas, n, &one, d_x_ln_t.d, 1, d_x_after_ln1_t, 1));
                }

                // Roll d_s ← d_s_rot (d_s_prev) for the next iter.
                CUDA_CHECK(cudaMemcpy(d_s.d, d_s_rot.d, B * m * sizeof(float), cudaMemcpyDeviceToDevice));
            }
            free_tensor(d_s); free_tensor(d_s_rot); free_tensor(d_x_ln_t);
            free_tensor(d_theta); free_tensor(d_pre_theta_t);
            free_tensor(s_workspace); free_tensor(theta_t_tmp);
            free_tensor(d_s_seq);

            // LN1 backward: d_x_after_ln1[l] → d_residual_in_via_ln1[l]
            Tensor d_residual_in_via_ln1 = make_tensor({T, B, m});
            launch_layer_norm_bwd(cache.residual_in[l].d, d_x_after_ln1.d, layer.gamma_ln1.d,
                                  cache.ln1_mean[l].d, cache.ln1_rstd[l].d,
                                  d_residual_in_via_ln1.d, layer.dgamma_ln1.d, layer.dbeta_ln1.d,
                                  N_rows, m);
            free_tensor(d_x_after_ln1);

            // d_residual_in[l] = d_pre_ln2 (skip-around the recurrence)
            //                  + d_residual_in_via_ln1 (LN1 backward branch)
            // Reuse d_pre_ln2's storage as d_residual_in (which IS d_residual_out for next iter).
            {
                int n = N_rows * m;
                float one = 1.0f;
                CUBLAS_CHECK(cublasSaxpy(cublas, n, &one, d_residual_in_via_ln1.d, 1,
                                         d_pre_ln2.d, 1));
            }
            free_tensor(d_residual_in_via_ln1);

            // After the layer-l loop iteration, d_residual_out (== d_pre_ln2 alias) holds
            // d(loss)/d residual_in[l]. For l > 0, that's d_residual_out for layer l-1.
            // For l = 0, it feeds into embedding backward.
        }

        // 6. Embedding backward: d_residual_out (now d_residual_in[0]) flows back through E.
        for (int t = 0; t < T; ++t) {
            EALRMNModel::embedding_bwd_col(cache.ids,
                                           d_residual_out.d + (int64_t)t * B * m,
                                           dE.d, B, T, t, m);
        }
        free_tensor(d_residual_out);
    }
};

#endif // VESTA_MODEL_GRP_STACK_CUH
