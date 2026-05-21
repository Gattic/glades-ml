// VESTA Phase-1 GPU prototype — GRP-RNN (Group-Rotation Product Recurrent Network)
//
// Recurrence:
//   theta_k(x_t) = tanh(W_a[k, :] @ z_t + b_a[k]) * phi_max,  k = 0..K-1
//   R_t = prod_{k=0..K-1} G_k(theta_k(x_t)) in SO(m)
//   s_t = R_t @ s_{t-1} + W_in @ z_t
//   y_t = W_out @ s_t + b_out
//
// G_k(theta) is the m-by-m identity except for the 2x2 block at rows/cols
// (plane_p[k], plane_q[k]) which is [[cos theta, -sin theta], [sin theta, cos theta]].
//
// Plane configurations:
//   - default (interlocking, stride=3): (plane_p[k], plane_q[k]) =
//     (2k mod m, (2k + 6) mod m). K = m/2.
//   - --grp-disjoint-planes: (2k, 2k+1) for k=0..K-1. K = m/2. Reduces to a
//     shared-phase LRU baseline.
//
// Ablations:
//   - --grp-tanh-state: applies tanh after the linear update (replicates EALRMN
//     Phase-1 tanh-RNN strawman; expected to fail by 10-100x on needle).
//   - --grp-fixed-angles: theta_k = b_a[k] (no input dependence; W_a forced
//     to zero gradient). Reduces to fixed-phase LRU.
//
// Reverse-time recursion for backward: applies the inverse Givens chain to
// recompute intermediate states without storing them. Memory cost: O(B*m)
// per timestep, same as a standard RNN.
//
// Author: VESTA, 2026-05-19.

#ifndef VESTA_MODEL_GRP_RNN_CUH
#define VESTA_MODEL_GRP_RNN_CUH

#include "common.cuh"
#include "kernels.cuh"
#include "recurrence_kernels.cuh"
#include "model_ealrmn.cuh"  // for EALRMNModel::embedding_fwd_col / _bwd_col

// =============================================================================
// CUDA kernels
// =============================================================================

// Compute theta = tanh(pre_theta) * phi_max
// pre_theta is (B, K), result theta is (B, K). All elementwise.
__global__ void k_grp_angle_act_fwd(const float* pre_theta, float* theta,
                                    int B, int K, float phi_max) {
    int n = B * K;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    theta[i] = tanhf(pre_theta[i]) * phi_max;
}

// Backward through theta = tanh(pre_theta) * phi_max.
// d_pre_theta = d_theta * phi_max * (1 - tanh(pre_theta)^2) = d_theta * phi_max * (1 - (theta/phi_max)^2)
// But we have theta saved, so: d_pre_theta = d_theta * (phi_max - theta*theta/phi_max).
__global__ void k_grp_angle_act_bwd(const float* theta, const float* d_theta,
                                    float* d_pre_theta, int B, int K, float phi_max) {
    int n = B * K;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = theta[i] / phi_max;  // = tanh(pre_theta[i])
    d_pre_theta[i] = d_theta[i] * phi_max * (1.0f - t * t);
}

// Apply K Givens rotations sequentially to a state vector. Forward.
// One block per batch element. Sequential within the block (Givens may share
// coordinates with interlocking planes, so order matters).
//
// Block size: any (only thread 0 does the work after copy).
__global__ void k_grp_givens_fwd(const float* s_in, const float* theta,
                                 const int* plane_p, const int* plane_q,
                                 float* s_out, int B, int m, int K) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Copy s_in[b] to s_out[b] in parallel.
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    for (int i = tid; i < m; i += bsz) {
        s_out[b * m + i] = s_in[b * m + i];
    }
    __syncthreads();

    // Sequential Givens application. Thread 0 only.
    if (tid == 0) {
        const float* theta_b = theta + b * K;
        float* s_b = s_out + b * m;
        for (int k = 0; k < K; ++k) {
            int p = plane_p[k];
            int q = plane_q[k];
            float c = cosf(theta_b[k]);
            float si = sinf(theta_b[k]);
            float sp = s_b[p];
            float sq = s_b[q];
            s_b[p] = c * sp - si * sq;
            s_b[q] = si * sp + c * sq;
        }
    }
    __syncthreads();
}

// Backward through the K-Givens product. Reverse-time recursion:
//
// Given s_out_post = R_t @ s_in (the result of forward), and d_s_out_post,
// recompute intermediate states by inverse-Givens and accumulate
// d_theta and d_s_in.
//
// Inverse of G_k(theta): G_k(-theta), so:
//   s_k[p] = c*s_{k+1}[p] + si*s_{k+1}[q]
//   s_k[q] = -si*s_{k+1}[p] + c*s_{k+1}[q]
//
// For gradient w.r.t. theta_k, using:
//   s_{k+1}[p] = c*s_k[p] - si*s_k[q]
//   s_{k+1}[q] = si*s_k[p] + c*s_k[q]
//   d_theta_k = (-si*s_k[p] - c*s_k[q]) * d_s_{k+1}[p]
//             + (c*s_k[p] - si*s_k[q]) * d_s_{k+1}[q]
//
// And d_s_k via the inverse rotation (orthogonal: same as inverse-Givens):
//   d_s_k[p] = c*d_s_{k+1}[p] + si*d_s_{k+1}[q]
//   d_s_k[q] = -si*d_s_{k+1}[p] + c*d_s_{k+1}[q]
//
// We use the post-rotation state buffer s_workspace (B, m) as scratch.
// d_s_out_post is overwritten with d_s_in on exit.
__global__ void k_grp_givens_bwd(const float* s_out_post, const float* theta,
                                 const int* plane_p, const int* plane_q,
                                 float* s_workspace,  // (B, m), scratch
                                 float* d_s_inout,    // in: d_s_out_post, out: d_s_in
                                 float* d_theta,      // (B, K), accumulated
                                 int B, int m, int K) {
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;
    int bsz = blockDim.x;

    // Copy s_out_post[b] -> s_workspace[b] (working state).
    for (int i = tid; i < m; i += bsz) {
        s_workspace[b * m + i] = s_out_post[b * m + i];
    }
    __syncthreads();

    // Sequential reverse-Givens. Thread 0 only.
    if (tid == 0) {
        const float* theta_b = theta + b * K;
        float* s_b = s_workspace + b * m;
        float* d_s_b = d_s_inout + b * m;
        float* d_theta_b = d_theta + b * K;

        for (int k = K - 1; k >= 0; --k) {
            int p = plane_p[k];
            int q = plane_q[k];
            float c = cosf(theta_b[k]);
            float si = sinf(theta_b[k]);

            // s_{k}[p,q] = inverse-Givens(s_{k+1}[p,q])
            // where s_{k+1} is currently in s_b.
            float sp_post = s_b[p];
            float sq_post = s_b[q];
            float sp_pre = c * sp_post + si * sq_post;
            float sq_pre = -si * sp_post + c * sq_post;
            s_b[p] = sp_pre;
            s_b[q] = sq_pre;

            // Accumulate d_theta_k using s_k (pre-rotation, now in s_b) and
            // d_s_{k+1} (currently in d_s_b).
            float dsp = d_s_b[p];
            float dsq = d_s_b[q];
            float dt = (-si * sp_pre - c * sq_pre) * dsp
                     + ( c * sp_pre - si * sq_pre) * dsq;
            d_theta_b[k] = d_theta_b[k] + dt;

            // Propagate gradient: d_s_k[p,q] = inverse-Givens(d_s_{k+1}[p,q]).
            d_s_b[p] = c * dsp + si * dsq;
            d_s_b[q] = -si * dsp + c * dsq;
        }
    }
    __syncthreads();
}

// Optional post-rotation tanh (the --grp-tanh-state ablation).
__global__ void k_grp_state_tanh_fwd(const float* s_in, float* s_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    s_out[i] = tanhf(s_in[i]);
}

__global__ void k_grp_state_tanh_bwd(const float* s_out, const float* d_s_out,
                                     float* d_s_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = s_out[i];
    d_s_in[i] = d_s_out[i] * (1.0f - s * s);
}

// =============================================================================
// GRP-RNN model
// =============================================================================

struct GRPRNNModelCache {
    int B = 0, T = 0, m = 0, K = 0, n_classes = 0;
    const int* ids = nullptr;
    Tensor z_all;      // (T, B, m)
    Tensor s_all;      // (T+1, B, m)  -- post-rotation, post-input, post-LN (the s_t output)
    Tensor s_post_rot_all;  // (T, B, m) -- post-rotation, pre-input-add (needed for Givens backward)
    Tensor pre_ln_all;     // (T, B, m) -- pre_tanh BEFORE optional LN (needed for LN backward)
    Tensor ln_mean_all;    // (T, B) -- LN per-step mean
    Tensor ln_rstd_all;    // (T, B) -- LN per-step rstd
    Tensor theta_all;  // (T, B, K)
    Tensor pre_theta_all;  // (T, B, K)
    Tensor logits;
    Tensor probs;
    Tensor losses;
    // LM-mode (per-token next-token prediction) — allocated only when needed.
    bool lm_mode = false;
    Tensor logits_all;   // (T, B, V) — positions 0..T-2 used; position T-1 unused
    Tensor probs_all;    // (T, B, V)
    Tensor losses_all;   // (T-1)*B losses (one per position per batch)
    int* labels_lm = nullptr;  // (T-1, B) int device buffer; labels_lm[t*B + b] = ids[b*T + (t+1)]

    void alloc(int B_, int T_, int m_, int K_, int n_classes_, bool lm_mode_ = false) {
        B = B_; T = T_; m = m_; K = K_; n_classes = n_classes_; lm_mode = lm_mode_;
        z_all = make_tensor({T, B, m});
        s_all = make_tensor({T + 1, B, m});
        s_post_rot_all = make_tensor({T, B, m});
        pre_ln_all = make_tensor({T, B, m});
        ln_mean_all = make_tensor({T, B});
        ln_rstd_all = make_tensor({T, B});
        theta_all = make_tensor({T, B, K});
        pre_theta_all = make_tensor({T, B, K});
        logits = make_tensor({B, n_classes});
        probs  = make_tensor({B, n_classes});
        losses = make_tensor({B});
        if (lm_mode) {
            logits_all = make_tensor({T, B, n_classes});
            probs_all = make_tensor({T, B, n_classes});
            losses_all = make_tensor({(T - 1) * B});
            CUDA_CHECK(cudaMalloc(&labels_lm, (T - 1) * B * sizeof(int)));
        }
    }
    void free() {
        free_tensor(z_all); free_tensor(s_all); free_tensor(s_post_rot_all);
        free_tensor(pre_ln_all); free_tensor(ln_mean_all); free_tensor(ln_rstd_all);
        free_tensor(theta_all); free_tensor(pre_theta_all);
        free_tensor(logits); free_tensor(probs); free_tensor(losses);
        if (lm_mode) {
            free_tensor(logits_all); free_tensor(probs_all); free_tensor(losses_all);
            if (labels_lm) cudaFree(labels_lm);
            labels_lm = nullptr;
        }
    }
};

// Build per-token LM labels: labels_lm[t*B + b] = ids[b*T + (t+1)] for t=0..T-2.
__global__ void k_build_lm_labels(const int* ids, int* labels_lm, int B, int T) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    if (b < B && t < T - 1) {
        labels_lm[t * B + b] = ids[b * T + (t + 1)];
    }
}

struct GRPRNNModel {
    cublasHandle_t cublas;
    int V = 0, m = 0, K = 0, n_classes = 0;

    // Ablation knobs.
    bool tanh_state = false;       // --grp-tanh-state
    bool fixed_angles = false;     // --grp-fixed-angles
    int stride = 3;                // interlocking stride; 1 = disjoint adjacent (post-aware)
    bool disjoint_planes = false;  // --grp-disjoint-planes: use (2k, 2k+1)
    bool use_layernorm = false;    // --grp-layernorm: apply LN to s_t each step
    float phi_max = 1.5707963f;    // pi/2
    float decay = 0.95f;           // scalar Λ = decay * I after rotation. Matches the
                                   // reference linear-RNN's init_orthogonal_scale=0.95
                                   // spectral radius. Set 1.0 for pure orthogonal.

    // Parameters.
    Tensor E;       // (V, m)
    Tensor W_a;     // (K, m)   angle weights: pre_theta = z_t @ W_a^T + b_a
    Tensor b_a;     // (K,)
    Tensor W_in;    // (m, m)
    Tensor W_out;   // (m, n_classes)
    Tensor b_out;   // (n_classes,)
    // LayerNorm parameters (only allocated/used when use_layernorm=true).
    Tensor gamma_ln; // (m,)
    Tensor beta_ln;  // (m,)

    // Plane geometry (device-side, int32, immutable after init).
    Tensor plane_p_dev;
    Tensor plane_q_dev;

    // Gradients.
    Tensor dE, dW_a, db_a, dW_in, dW_out, db_out, dgamma_ln, dbeta_ln;
    // Adam moments.
    Tensor mE, mW_a, mb_a, mW_in, mW_out, mb_out, mgamma_ln, mbeta_ln;
    Tensor vE, vW_a, vb_a, vW_in, vW_out, vb_out, vgamma_ln, vbeta_ln;

    std::vector<Tensor*> params() {
        if (use_layernorm) return { &E, &W_a, &b_a, &W_in, &W_out, &b_out, &gamma_ln, &beta_ln };
        return { &E, &W_a, &b_a, &W_in, &W_out, &b_out };
    }
    std::vector<Tensor*> grads() {
        if (use_layernorm) return { &dE, &dW_a, &db_a, &dW_in, &dW_out, &db_out, &dgamma_ln, &dbeta_ln };
        return { &dE, &dW_a, &db_a, &dW_in, &dW_out, &db_out };
    }
    std::vector<Tensor*> ms() {
        if (use_layernorm) return { &mE, &mW_a, &mb_a, &mW_in, &mW_out, &mb_out, &mgamma_ln, &mbeta_ln };
        return { &mE, &mW_a, &mb_a, &mW_in, &mW_out, &mb_out };
    }
    std::vector<Tensor*> vs() {
        if (use_layernorm) return { &vE, &vW_a, &vb_a, &vW_in, &vW_out, &vb_out, &vgamma_ln, &vbeta_ln };
        return { &vE, &vW_a, &vb_a, &vW_in, &vW_out, &vb_out };
    }

    void init(cublasHandle_t handle, int V_, int m_, int K_, int n_classes_,
              unsigned long long seed) {
        cublas = handle;
        V = V_; m = m_; K = K_; n_classes = n_classes_;
        HostRng rng(seed);

        E      = make_tensor({V, m});
        W_a    = make_tensor({K, m});
        b_a    = make_tensor({K});
        W_in   = make_tensor({m, m});
        W_out  = make_tensor({m, n_classes});
        b_out  = make_tensor({n_classes});

        dE     = make_tensor({V, m});
        dW_a   = make_tensor({K, m});
        db_a   = make_tensor({K});
        dW_in  = make_tensor({m, m});
        dW_out = make_tensor({m, n_classes});
        db_out = make_tensor({n_classes});

        mE     = make_tensor({V, m});      mE.zero_();
        mW_a   = make_tensor({K, m});      mW_a.zero_();
        mb_a   = make_tensor({K});         mb_a.zero_();
        mW_in  = make_tensor({m, m});      mW_in.zero_();
        mW_out = make_tensor({m, n_classes}); mW_out.zero_();
        mb_out = make_tensor({n_classes}); mb_out.zero_();
        vE     = make_tensor({V, m});      vE.zero_();
        vW_a   = make_tensor({K, m});      vW_a.zero_();
        vb_a   = make_tensor({K});         vb_a.zero_();
        vW_in  = make_tensor({m, m});      vW_in.zero_();
        vW_out = make_tensor({m, n_classes}); vW_out.zero_();
        vb_out = make_tensor({n_classes}); vb_out.zero_();

        init_normal(E, 0.0f, 1.0f / std::sqrt((float)m), rng);
        // W_a init small so initial angles are near zero (R_t ~= I at init,
        // matching the "orthogonal init" prescription from EALRMN Phase-1).
        init_normal(W_a, 0.0f, 1.0f / std::sqrt((float)m), rng);
        b_a.zero_();
        init_xavier_uniform(W_in, m, m, rng);
        init_xavier_uniform(W_out, m, n_classes, rng);
        b_out.zero_();

        // LayerNorm parameters (allocated even if use_layernorm=false, so params()/grads()
        // can return them when flipped on dynamically; gradient is zero when unused).
        gamma_ln = make_tensor({m});
        beta_ln  = make_tensor({m});
        dgamma_ln = make_tensor({m});  dgamma_ln.zero_();
        dbeta_ln  = make_tensor({m});  dbeta_ln.zero_();
        mgamma_ln = make_tensor({m});  mgamma_ln.zero_();
        mbeta_ln  = make_tensor({m});  mbeta_ln.zero_();
        vgamma_ln = make_tensor({m});  vgamma_ln.zero_();
        vbeta_ln  = make_tensor({m});  vbeta_ln.zero_();
        // gamma init 1, beta init 0.
        std::vector<float> ones(m, 1.0f);
        copy_h2d(gamma_ln, ones);
        beta_ln.zero_();

        // Plane geometry. Build on host, copy to device.
        //   - disjoint adjacent: (p_k, q_k) = (2k, 2k+1). K = m/2 typical.
        //     This is the LRU reduction: m/2 disjoint Givens block-diagonal in 2x2 form.
        //   - interlocking: (p_k, q_k) = (k mod m, (k + stride) mod m). Default K = m.
        //     Each coord is the p of one Givens and the q of another; consecutive
        //     Givens share a coord, so they do NOT commute (this is the source of
        //     non-diagonality and the path to SO(m) generation).
        plane_p_dev = make_tensor({K});
        plane_q_dev = make_tensor({K});
        std::vector<int> hp(K), hq(K);
        if (disjoint_planes) {
            for (int k = 0; k < K; ++k) {
                hp[k] = (2 * k) % m;
                hq[k] = (2 * k + 1) % m;
            }
        } else {
            for (int k = 0; k < K; ++k) {
                hp[k] = k % m;
                hq[k] = (k + stride) % m;
            }
        }
        // Copy ints into the float tensors as raw bytes is fragile; allocate int tensors instead.
        free_tensor(plane_p_dev);
        free_tensor(plane_q_dev);
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
        // Always free all 8 tensor groups (params() may not include LN when use_layernorm=false).
        free_tensor(E); free_tensor(W_a); free_tensor(b_a);
        free_tensor(W_in); free_tensor(W_out); free_tensor(b_out);
        free_tensor(gamma_ln); free_tensor(beta_ln);
        free_tensor(dE); free_tensor(dW_a); free_tensor(db_a);
        free_tensor(dW_in); free_tensor(dW_out); free_tensor(db_out);
        free_tensor(dgamma_ln); free_tensor(dbeta_ln);
        free_tensor(mE); free_tensor(mW_a); free_tensor(mb_a);
        free_tensor(mW_in); free_tensor(mW_out); free_tensor(mb_out);
        free_tensor(mgamma_ln); free_tensor(mbeta_ln);
        free_tensor(vE); free_tensor(vW_a); free_tensor(vb_a);
        free_tensor(vW_in); free_tensor(vW_out); free_tensor(vb_out);
        free_tensor(vgamma_ln); free_tensor(vbeta_ln);
        if (plane_p_dev.d) cudaFree(plane_p_dev.d);
        if (plane_q_dev.d) cudaFree(plane_q_dev.d);
    }

    void zero_grads() { for (Tensor* g : grads()) g->zero_(); }

    int64_t num_params() {
        int64_t n = 0;
        for (Tensor* p : params()) n += p->numel;
        return n;
    }

    void forward(const int* ids_device, int B, int T, GRPRNNModelCache& cache) {
        cache.ids = ids_device;
        cache.B = B; cache.T = T; cache.m = m; cache.K = K; cache.n_classes = n_classes;

        // s_0 = 0
        CUDA_CHECK(cudaMemset(cache.s_all.d, 0, B * m * sizeof(float)));

        Tensor s_prev = make_tensor({B, m});  s_prev.zero_();
        Tensor z_t = make_tensor({B, m});
        Tensor pre_theta = make_tensor({B, K});
        Tensor theta = make_tensor({B, K});
        Tensor s_rot = make_tensor({B, m});
        Tensor s_post_rot = make_tensor({B, m});
        Tensor s_t = make_tensor({B, m});
        Tensor pre_tanh = make_tensor({B, m});

        for (int t = 0; t < T; ++t) {
            EALRMNModel::embedding_fwd_col(ids_device, E.d, z_t.d, B, T, t, m);
            CUDA_CHECK(cudaMemcpy(cache.z_all.d + (int64_t)t * B * m, z_t.d,
                                  B * m * sizeof(float), cudaMemcpyDeviceToDevice));

            // pre_theta = z_t @ W_a^T  (z_t is (B, m), W_a is (K, m), pre_theta is (B, K))
            if (fixed_angles) {
                CUDA_CHECK(cudaMemset(pre_theta.d, 0, B * K * sizeof(float)));
            } else {
                gemm_nt(cublas, B, K, m, 1.0f, z_t.d, W_a.d, 0.0f, pre_theta.d);
            }
            launch_bias_add(pre_theta.d, b_a.d, B, K);
            CUDA_CHECK(cudaMemcpy(cache.pre_theta_all.d + (int64_t)t * B * K, pre_theta.d,
                                  B * K * sizeof(float), cudaMemcpyDeviceToDevice));

            // theta = tanh(pre_theta) * phi_max
            {
                int n = B * K;
                int block = 256;
                int grid = (n + block - 1) / block;
                k_grp_angle_act_fwd<<<grid, block>>>(pre_theta.d, theta.d, B, K, phi_max);
            }
            CUDA_CHECK(cudaMemcpy(cache.theta_all.d + (int64_t)t * B * K, theta.d,
                                  B * K * sizeof(float), cudaMemcpyDeviceToDevice));

            // s_rot = R_t @ s_prev  (in-place via Givens kernel; pre-decay)
            {
                int block = 32;
                k_grp_givens_fwd<<<B, block>>>(s_prev.d, theta.d,
                                               (int*)plane_p_dev.d, (int*)plane_q_dev.d,
                                               s_rot.d, B, m, K);
            }
            // Save s_post_rot = s_rot (PRE-decay, needed by Givens backward).
            CUDA_CHECK(cudaMemcpy(s_post_rot.d, s_rot.d, B * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(cache.s_post_rot_all.d + (int64_t)t * B * m, s_post_rot.d,
                                  B * m * sizeof(float), cudaMemcpyDeviceToDevice));
            // Apply scalar decay: s_rot *= decay (now stale post-decay).
            if (decay != 1.0f) {
                int n = B * m;
                CUBLAS_CHECK(cublasSscal(cublas, n, &decay, s_rot.d, 1));
            }

            // pre_tanh = s_rot + W_in @ z_t  (linear addition, optionally followed by LN and tanh)
            CUDA_CHECK(cudaMemcpy(pre_tanh.d, s_rot.d, B * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            // pre_tanh += z_t @ W_in^T
            gemm_nt(cublas, B, m, m, 1.0f, z_t.d, W_in.d, 1.0f, pre_tanh.d);

            // Optional LayerNorm. Saves pre_ln (input to LN), mean, rstd per (t, b).
            // Overwrites pre_tanh with LN(pre_tanh).
            if (use_layernorm) {
                // Save pre-LN value for backward.
                CUDA_CHECK(cudaMemcpy(cache.pre_ln_all.d + (int64_t)t * B * m, pre_tanh.d,
                                      B * m * sizeof(float), cudaMemcpyDeviceToDevice));
                float* mean_t = cache.ln_mean_all.d + (int64_t)t * B;
                float* rstd_t = cache.ln_rstd_all.d + (int64_t)t * B;
                Tensor ln_out = make_tensor({B, m});
                launch_layer_norm_fwd(pre_tanh.d, gamma_ln.d, beta_ln.d,
                                      ln_out.d, mean_t, rstd_t, B, m);
                CUDA_CHECK(cudaMemcpy(pre_tanh.d, ln_out.d, B * m * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
                free_tensor(ln_out);
            }

            // Optional state-tanh (--grp-tanh-state).
            if (tanh_state) {
                int n = B * m;
                int block = 256;
                int grid = (n + block - 1) / block;
                k_grp_state_tanh_fwd<<<grid, block>>>(pre_tanh.d, s_t.d, n);
            } else {
                CUDA_CHECK(cudaMemcpy(s_t.d, pre_tanh.d, B * m * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
            }

            CUDA_CHECK(cudaMemcpy(cache.s_all.d + (int64_t)(t + 1) * B * m, s_t.d,
                                  B * m * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(s_prev.d, s_t.d, B * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        if (!cache.lm_mode) {
            // Single-label classification head: logits = s_T @ W_out + b_out.
            const float* s_T_ptr = cache.s_all.d + (int64_t)T * B * m;
            gemm_nn(cublas, B, n_classes, m, 1.0f, s_T_ptr, W_out.d, 0.0f, cache.logits.d);
            launch_bias_add(cache.logits.d, b_out.d, B, n_classes);
        } else {
            // LM mode: per-step logits[t] = s_all[t+1] @ W_out + b_out for t=0..T-2.
            // s_all[1..T-1] is a contiguous (T-1)*B*m slice starting at offset B*m.
            int M = (T - 1) * B;
            const float* s_lm_ptr = cache.s_all.d + (int64_t)B * m;
            float* logits_lm_ptr = cache.logits_all.d;
            gemm_nn(cublas, M, n_classes, m, 1.0f, s_lm_ptr, W_out.d, 0.0f, logits_lm_ptr);
            launch_bias_add(logits_lm_ptr, b_out.d, M, n_classes);
        }

        free_tensor(s_prev); free_tensor(z_t); free_tensor(pre_theta); free_tensor(theta);
        free_tensor(s_rot); free_tensor(s_post_rot); free_tensor(s_t); free_tensor(pre_tanh);
    }

    float compute_loss(const int* labels_device, GRPRNNModelCache& cache) {
        if (cache.lm_mode) {
            // Per-token LM: softmax+CE over (T-1)*B rows. labels_lm must be built before this.
            int M = (cache.T - 1) * cache.B;
            launch_softmax_ce_fwd(cache.logits_all.d, cache.labels_lm,
                                  cache.probs_all.d, cache.losses_all.d,
                                  M, cache.n_classes);
            std::vector<float> h_losses;
            copy_d2h(h_losses, cache.losses_all);
            float s = 0.0f;
            for (float v : h_losses) s += v;
            return s;  // total loss summed over (T-1)*B tokens; caller divides as needed
        }
        launch_softmax_ce_fwd(cache.logits.d, labels_device, cache.probs.d, cache.losses.d,
                              cache.B, cache.n_classes);
        std::vector<float> h_losses;
        copy_d2h(h_losses, cache.losses);
        float s = 0.0f;
        for (float v : h_losses) s += v;
        return s;
    }

    void backward(const int* labels_device, GRPRNNModelCache& cache) {
        const int B = cache.B, T = cache.T;
        Tensor d_s = make_tensor({B, m});
        Tensor d_logits_lm_all;  // optional, allocated only in LM mode
        Tensor d_s_lm_all;       // optional, (T-1, B, m) per-step logits-gradient contributions

        if (cache.lm_mode) {
            // LM mode: gradient flows from per-step logits, not from a single final readout.
            // d_logits_all = (probs_all - one_hot(labels_lm)) * scale.
            // scale = 1/B matches the gradcheck convention (compute_loss/B is the "average").
            // The per-token-average is then sum_loss / ((T-1)*B); we compensate at training
            // time by reducing the learning rate by ~(T-1) or by changing the reported loss.
            int M = (T - 1) * B;
            float scale = 1.0f / (float)B;
            d_logits_lm_all = make_tensor({(T - 1), B, n_classes});
            launch_softmax_ce_bwd(cache.probs_all.d, cache.labels_lm,
                                  d_logits_lm_all.d, M, n_classes, scale);
            // dW_out += s_lm^T @ d_logits_lm_all
            const float* s_lm_ptr = cache.s_all.d + (int64_t)B * m;  // s_all[1..T-1]
            gemm_tn(cublas, m, n_classes, M, 1.0f, s_lm_ptr, d_logits_lm_all.d, 1.0f, dW_out.d);
            // db_out += sum(d_logits_lm_all) across M rows
            launch_bias_bwd(d_logits_lm_all.d, db_out.d, M, n_classes);
            // d_s_lm_all = d_logits_lm_all @ W_out^T  (shape (T-1)*B, m)
            d_s_lm_all = make_tensor({(T - 1), B, m});
            gemm_nt(cublas, M, m, n_classes, 1.0f, d_logits_lm_all.d, W_out.d, 0.0f, d_s_lm_all.d);
            free_tensor(d_logits_lm_all);
            // In LM mode, d_s starts at zero (no readout at t=T-1).
            d_s.zero_();
        } else {
            // Single-label classification mode (original path).
            const float scale = 1.0f / (float)B;
            Tensor d_logits = make_tensor({B, n_classes});
            launch_softmax_ce_bwd(cache.probs.d, labels_device, d_logits.d, B, n_classes, scale);
            launch_bias_bwd(d_logits.d, db_out.d, B, n_classes);
            const float* s_T_ptr = cache.s_all.d + (int64_t)T * B * m;
            gemm_tn(cublas, m, n_classes, B, 1.0f, s_T_ptr, d_logits.d, 1.0f, dW_out.d);
            gemm_nt(cublas, B, m, n_classes, 1.0f, d_logits.d, W_out.d, 0.0f, d_s.d);
            free_tensor(d_logits);
        }

        // Per-step backward.
        Tensor d_pre_tanh = make_tensor({B, m});
        Tensor d_s_rot   = make_tensor({B, m});
        Tensor d_z       = make_tensor({B, m});
        Tensor d_theta   = make_tensor({B, K});
        Tensor d_pre_theta = make_tensor({B, K});
        Tensor s_workspace = make_tensor({B, m});

        Tensor theta_t = make_tensor({B, K});

        for (int t = T - 1; t >= 0; --t) {
            // LM mode: inject per-step logits gradient into d_s at iteration t < T-1.
            // The recurrence backward at iter t processes d_s = grad w.r.t. s_t = cache.s_all[t+1].
            // logits[t] = W_out @ s_all[t+1] for t=0..T-2, so gradient enters d_s for t=0..T-2.
            if (cache.lm_mode && t < T - 1) {
                int n = B * m;
                float one = 1.0f;
                const float* d_s_lm_t = d_s_lm_all.d + (int64_t)t * B * m;
                CUBLAS_CHECK(cublasSaxpy(cublas, n, &one, d_s_lm_t, 1, d_s.d, 1));
            }
            // Reload theta_t, s_post_rot_t (in-place pointers).
            const float* theta_t_ptr = cache.theta_all.d + (int64_t)t * B * K;
            const float* s_post_rot_t_ptr = cache.s_post_rot_all.d + (int64_t)t * B * m;

            // Step 1: backward through optional state-tanh: d_after_ln = d_s * (1 - s_t^2).
            const float* s_t_ptr = cache.s_all.d + (int64_t)(t + 1) * B * m;
            Tensor d_after_ln = make_tensor({B, m});
            if (tanh_state) {
                int n = B * m;
                int block = 256;
                int grid = (n + block - 1) / block;
                k_grp_state_tanh_bwd<<<grid, block>>>(s_t_ptr, d_s.d, d_after_ln.d, n);
            } else {
                CUDA_CHECK(cudaMemcpy(d_after_ln.d, d_s.d, B * m * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
            }

            // Step 1b: backward through optional LayerNorm.
            if (use_layernorm) {
                const float* pre_ln_t = cache.pre_ln_all.d + (int64_t)t * B * m;
                const float* mean_t = cache.ln_mean_all.d + (int64_t)t * B;
                const float* rstd_t = cache.ln_rstd_all.d + (int64_t)t * B;
                launch_layer_norm_bwd(pre_ln_t, d_after_ln.d, gamma_ln.d,
                                      mean_t, rstd_t,
                                      d_pre_tanh.d, dgamma_ln.d, dbeta_ln.d, B, m);
            } else {
                CUDA_CHECK(cudaMemcpy(d_pre_tanh.d, d_after_ln.d, B * m * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
            }
            free_tensor(d_after_ln);

            // Step 2: backward through pre_tanh = (decay * s_rot_pre_decay) + W_in @ z_t.
            // d_s_rot_pre_decay = d_pre_tanh * decay; d_(W_in @ z_t) = d_pre_tanh.
            CUDA_CHECK(cudaMemcpy(d_s_rot.d, d_pre_tanh.d, B * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            if (decay != 1.0f) {
                int n = B * m;
                CUBLAS_CHECK(cublasSscal(cublas, n, &decay, d_s_rot.d, 1));
            }

            const float* z_t_ptr = cache.z_all.d + (int64_t)t * B * m;
            // dW_in += d_pre_tanh^T @ z_t
            gemm_tn(cublas, m, m, B, 1.0f, d_pre_tanh.d, z_t_ptr, 1.0f, dW_in.d);
            // d_z = d_pre_tanh @ W_in
            gemm_nn(cublas, B, m, m, 1.0f, d_pre_tanh.d, W_in.d, 0.0f, d_z.d);

            // Step 3: backward through s_rot = R_t @ s_prev. Reverse-time recursion.
            // d_theta_t accumulated here (start zero).
            d_theta.zero_();
            {
                // Reload theta_t for the kernel.
                CUDA_CHECK(cudaMemcpy(theta_t.d, theta_t_ptr, B * K * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
                int block = 32;
                k_grp_givens_bwd<<<B, block>>>(s_post_rot_t_ptr, theta_t.d,
                                               (int*)plane_p_dev.d, (int*)plane_q_dev.d,
                                               s_workspace.d, d_s_rot.d, d_theta.d,
                                               B, m, K);
            }
            // After k_grp_givens_bwd, d_s_rot holds d_s_prev (the gradient w.r.t. input state).

            // Step 4: backward through theta = tanh(pre_theta) * phi_max.
            if (!fixed_angles) {
                int n = B * K;
                int block = 256;
                int grid = (n + block - 1) / block;
                k_grp_angle_act_bwd<<<grid, block>>>(theta_t.d, d_theta.d, d_pre_theta.d,
                                                     B, K, phi_max);

                // db_a += sum d_pre_theta (across B)
                launch_bias_bwd(d_pre_theta.d, db_a.d, B, K);
                // dW_a += d_pre_theta^T @ z_t  (K, m)
                gemm_tn(cublas, K, m, B, 1.0f, d_pre_theta.d, z_t_ptr, 1.0f, dW_a.d);
                // d_z += d_pre_theta @ W_a
                gemm_nn(cublas, B, m, K, 1.0f, d_pre_theta.d, W_a.d, 1.0f, d_z.d);
            } else {
                // Fixed-angles: pre_theta = b_a, then theta = tanh(b_a) * phi_max.
                // d_pre_theta = d_theta * phi_max * (1 - tanh(b_a)^2). Same backward as
                // the trainable case; only difference is no W_a update and no d_z contribution.
                int n = B * K;
                int block = 256;
                int grid = (n + block - 1) / block;
                k_grp_angle_act_bwd<<<grid, block>>>(theta_t.d, d_theta.d, d_pre_theta.d,
                                                     B, K, phi_max);
                launch_bias_bwd(d_pre_theta.d, db_a.d, B, K);
                // No dW_a (W_a frozen with fixed_angles).
                // No d_z contribution from angle path.
            }

            // Step 5: embedding backward.
            EALRMNModel::embedding_bwd_col(cache.ids, d_z.d, dE.d, B, T, t, m);

            // Step 6: roll d_s ← d_s_rot (== d_s_prev) for next iter.
            CUDA_CHECK(cudaMemcpy(d_s.d, d_s_rot.d, B * m * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        free_tensor(d_s); free_tensor(d_pre_tanh);
        free_tensor(d_s_rot); free_tensor(d_z); free_tensor(d_theta);
        free_tensor(d_pre_theta); free_tensor(s_workspace); free_tensor(theta_t);
        if (cache.lm_mode) free_tensor(d_s_lm_all);
    }
};

#endif // VESTA_MODEL_GRP_RNN_CUH
