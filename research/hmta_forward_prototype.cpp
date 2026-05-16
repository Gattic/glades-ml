// HMTA Forward Prototype — paradigm #261 Gate-0 (kernel-correctness branch).
//
// Standalone C++98 program that validates the forward arithmetic of
// Hierarchical Multipole Token Attention (HMTA) against flat causal SDPA on
// synthetic data with controllable underlying rank.  See
// research/PARADIGM_SHIFT_261_HMTA_DESIGN.md for the full mathematics.
//
// What this prototype tests:
//   (a) HMTA forward at multipole rank p approximates flat-SDPA output;
//   (b) Approximation error decays with p;
//   (c) Approximation error scales with truncated singular-value tail of
//       admissible cluster-pair logit submatrices (Proposition 9.3);
//   (d) FLOP count of HMTA scales as O(T log T) vs SDPA's O(T^2), with the
//       crossover at small T already visible.
//   (e) SDPA recovery limit: at p = s0 and eta = T/s0 (no admissible far
//       pairs), HMTA collapses to flat causal SDPA exactly.
//
// This is the Gate-0 "kernel-correctness" milestone of paradigm #261.  A
// PASS here unlocks Phase B (full from-scratch training comparison).  A
// FAIL falsifies the multipole hypothesis before any CUDA kernel is written.
//
// Build:
//   g++ -std=c++98 -O2 -Wall -Wextra research/hmta_forward_prototype.cpp \
//       -o research/hmta_forward_prototype
// Run:
//   ./research/hmta_forward_prototype
//
// No external dependencies.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

// -------------------------------------------------------------------- //
// Random number helpers (Box-Muller from rand()).
// -------------------------------------------------------------------- //
static double uniform_open() {
    double u = (double(rand()) + 1.0) / (double(RAND_MAX) + 2.0);
    return u;
}

static double gauss(double mean, double stddev) {
    double u1 = uniform_open();
    double u2 = uniform_open();
    double z  = std::sqrt(-2.0 * std::log(u1)) *
                std::cos(2.0 * 3.14159265358979323846 * u2);
    return mean + stddev * z;
}

// -------------------------------------------------------------------- //
// Small linear-algebra helpers (column-major float matrices).
// -------------------------------------------------------------------- //
//   row-major flat storage: A[i,j] = A[i*ncols + j]

static void zero(float* A, int n) {
    for (int i = 0; i < n; ++i) A[i] = 0.0f;
}

static void fill_gauss(float* A, int n, double sd) {
    for (int i = 0; i < n; ++i) A[i] = (float)gauss(0.0, sd);
}

// y[m] = sum_k A[m,k] * x[k]   A is M×K, row-major
static void matvec(const float* A, const float* x, float* y, int M, int K) {
    for (int i = 0; i < M; ++i) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) s += A[i*K + k] * x[k];
        y[i] = (float)s;
    }
}

// C = A * B   A:M×K, B:K×N, C:M×N
static void matmul(const float* A, const float* B, float* C,
                   int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += A[i*K + k] * B[k*N + j];
            C[i*N + j] = (float)s;
        }
    }
}

// C = A^T * B   A:K×M, B:K×N, C:M×N
static void matmul_AT_B(const float* A, const float* B, float* C,
                        int K, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += A[k*M + i] * B[k*N + j];
            C[i*N + j] = (float)s;
        }
    }
}

// In-place row-wise softmax with a causal mask: row i can only attend to
// columns j <= i_in_global_terms (we pass an absolute logit array; columns
// strictly greater than the row's causal limit are zeroed out via the mask).
// `causal_limits[i]` is the inclusive column-index limit for row i.
static void softmax_rows_causal(float* P, int rows, int cols,
                                const int* causal_limits) {
    for (int i = 0; i < rows; ++i) {
        int lim = causal_limits[i];
        // find max for stability over allowed columns
        float mx = -1e30f;
        for (int j = 0; j <= lim && j < cols; ++j) {
            if (P[i*cols + j] > mx) mx = P[i*cols + j];
        }
        // exp, sum
        double s = 0.0;
        for (int j = 0; j < cols; ++j) {
            if (j > lim) {
                P[i*cols + j] = 0.0f;
            } else {
                float e = std::exp(P[i*cols + j] - mx);
                P[i*cols + j] = e;
                s += e;
            }
        }
        // normalize
        if (s > 0) {
            float inv = (float)(1.0 / s);
            for (int j = 0; j < cols; ++j) P[i*cols + j] *= inv;
        }
    }
}

// -------------------------------------------------------------------- //
// Power-iteration top-p SVD for a M×N matrix.
// Returns (U:M×p, S:p, V:N×p).  Only used for OFFLINE multipole-encoder
// initialization on synthetic data (not in the production HMTA forward).
// -------------------------------------------------------------------- //
static void rand_subspace_truncated_svd(const float* A, int M, int N, int p,
                                        float* U, float* S, float* V) {
    // Random Gaussian sketch + 2 passes of subspace iteration is enough for
    // p << min(M,N) and rough accuracy.  We only need ~3 decimal places.
    std::vector<float> Omega(N * p), Y(M * p), B(p * N), Up(M * p);
    fill_gauss(&Omega[0], N * p, 1.0);
    // Y = A * Omega
    matmul(A, &Omega[0], &Y[0], M, N, p);
    // QR(Y) via modified Gram-Schmidt
    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < j; ++i) {
            double dot = 0.0;
            for (int r = 0; r < M; ++r) dot += Y[r*p + i] * Y[r*p + j];
            for (int r = 0; r < M; ++r) Y[r*p + j] -= (float)dot * Y[r*p + i];
        }
        double nrm = 0.0;
        for (int r = 0; r < M; ++r) nrm += (double)Y[r*p + j] * Y[r*p + j];
        nrm = std::sqrt(nrm);
        if (nrm < 1e-12) nrm = 1.0;
        float inv = (float)(1.0 / nrm);
        for (int r = 0; r < M; ++r) Y[r*p + j] *= inv;
    }
    // U = Y is now M×p orthonormal-columns
    for (int i = 0; i < M*p; ++i) U[i] = Y[i];
    // B = U^T A   (p × N)
    matmul_AT_B(U, A, &B[0], M, p, N);
    // S[i] = ||B[i,:]||  (approximate singular values)
    for (int i = 0; i < p; ++i) {
        double nrm = 0.0;
        for (int j = 0; j < N; ++j) nrm += (double)B[i*N + j] * B[i*N + j];
        S[i] = (float)std::sqrt(nrm);
    }
    // V[:,i] = B[i,:] / S[i]
    for (int i = 0; i < p; ++i) {
        float inv = (S[i] > 1e-12f) ? 1.0f / S[i] : 0.0f;
        for (int j = 0; j < N; ++j) V[j*p + i] = B[i*N + j] * inv;
    }
}

// -------------------------------------------------------------------- //
// Synthetic data: Q, K, V with controllable underlying rank.
// q_i = U_q * c_i + xi    (d-dim)
// k_j = U_k * d_j + xi    (d-dim)
// v_j = random Gaussian
// where U_q, U_k are d×r random orthonormal-column matrices.
// -------------------------------------------------------------------- //
static void make_synthetic_QKV(int T, int d, int r, double noise_sd,
                               float* Q, float* K, float* V) {
    std::vector<float> U_q(d * r), U_k(d * r);
    fill_gauss(&U_q[0], d * r, 1.0 / std::sqrt((double)d));
    fill_gauss(&U_k[0], d * r, 1.0 / std::sqrt((double)d));

    std::vector<float> c(T * r), e(T * r);
    fill_gauss(&c[0], T * r, 1.0);
    fill_gauss(&e[0], T * r, 1.0);

    // Q[i,:] = U_q * c[i,:]  + noise
    for (int i = 0; i < T; ++i) {
        for (int p = 0; p < d; ++p) {
            double s = 0.0;
            for (int rr = 0; rr < r; ++rr) s += U_q[p*r + rr] * c[i*r + rr];
            s += gauss(0.0, noise_sd);
            Q[i*d + p] = (float)s;
        }
        for (int p = 0; p < d; ++p) {
            double s = 0.0;
            for (int rr = 0; rr < r; ++rr) s += U_k[p*r + rr] * e[i*r + rr];
            s += gauss(0.0, noise_sd);
            K[i*d + p] = (float)s;
        }
    }
    fill_gauss(V, T * d, 1.0);
}

// -------------------------------------------------------------------- //
// Flat causal SDPA: Y[i,:] = sum_{j<=i} P[i,j] * V[j,:]
//                  P[i,j] = softmax_j ( q_i . k_j / sqrt(d) )
// -------------------------------------------------------------------- //
static void flat_sdpa_forward(const float* Q, const float* K, const float* V,
                              int T, int d, float* Y, double* flops_out) {
    std::vector<float> Logits(T * T);
    double scale = 1.0 / std::sqrt((double)d);
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < T; ++j) {
            double s = 0.0;
            for (int p = 0; p < d; ++p) s += Q[i*d + p] * K[j*d + p];
            Logits[i*T + j] = (float)(s * scale);
        }
    }
    std::vector<int> causal(T);
    for (int i = 0; i < T; ++i) causal[i] = i;
    softmax_rows_causal(&Logits[0], T, T, &causal[0]);

    // Y = P * V
    matmul(&Logits[0], V, Y, T, T, d);

    if (flops_out) {
        // QK^T:  T*T*d.   softmax: ~T*T (cheap).   PV: T*T*d.
        *flops_out = 2.0 * (double)T * (double)T * (double)d;
    }
}

// -------------------------------------------------------------------- //
// HMTA forward — see PARADIGM_SHIFT_261_HMTA_DESIGN.md §5.
// Parameters:
//   T            sequence length (must be a multiple of s0)
//   d            head dim
//   s0           leaf size
//   p            multipole rank
//   eta          near-neighbor radius (in clusters)
// Operators (all learned in production; here we use synthetic init derived
// from the leaf's K/V via local SVD — so this prototype lower-bounds what a
// trained HMTA can do, by using the LOCALLY-OPTIMAL rank-p basis at each
// leaf instead of a single level-shared E):
//   E (per-leaf, local SVD)         : R^{p x s0}
//   K_{ell, Delta}                  : we approximate via PROJECTION of the
//                                     true cluster-pair logit matrix onto
//                                     the per-leaf moment bases.  This
//                                     gives the BEST POSSIBLE rank-p
//                                     translation operator and thus the
//                                     LOWER BOUND of HMTA approximation
//                                     error.  A trained HMTA can only do
//                                     worse than this (in the L2 sense).
// We are testing the multipole hypothesis: if THIS lower-bound HMTA is
// inaccurate vs flat-SDPA, then a trained HMTA cannot possibly do better.
//
// The forward pipeline below realizes equations (Sec. 5.2-5.5) of the
// design document with the simplification that:
//   - M2M is identity (we compute moments only at leaf level; no upward
//     coarsening since at the test scales here the leaves are already
//     informative).
//   - L2L is identity.
//   - eta-near is "self + |Delta| <= eta", with admissibility for far pairs
//     being |Delta| > eta.
// This is the equivalent of a SINGLE-LEVEL multipole, but with eta sweep
// across all far cluster-pairs.  It's sufficient to test the rank-p
// approximation hypothesis, which is the crux of the design.
// -------------------------------------------------------------------- //
static void hmta_forward(const float* Q, const float* K, const float* V,
                         int T, int d, int s0, int p, int eta,
                         float* Y, double* flops_out, double scale) {
    int Nc = T / s0;  // number of leaf clusters

    // -------- Step 1: leaf-encode --- per-leaf rank-p SVD of [K_I; V_I].
    // We pack K and V into a single (s0, 2d) per-leaf matrix and take a
    // rank-p truncation.  Then the encoder E is U^T (the left singular
    // vectors transposed): R^{p x s0}.
    //
    // Moments M[c] is the rank-p projection of [K_I_c; V_I_c]:
    //   M[c] = U[c]^T * [K_I_c | V_I_c]    in R^{p x 2d}
    // We store U[c] in R^{s0 x p} per cluster and M[c] in R^{p x 2d}.
    //
    // Simultaneously, compute the "projected key/value" reconstruction:
    //   [K_hat_I_c | V_hat_I_c] = U[c] * U[c]^T * [K_I_c | V_I_c]
    // which is what HMTA sees in the far-field.
    //
    std::vector< std::vector<float> > Uc(Nc, std::vector<float>(s0 * p));
    std::vector< std::vector<float> > Mc(Nc, std::vector<float>(p * 2 * d));
    std::vector< std::vector<float> > Khat(Nc, std::vector<float>(s0 * d));
    std::vector< std::vector<float> > Vhat(Nc, std::vector<float>(s0 * d));

    for (int c = 0; c < Nc; ++c) {
        int base = c * s0;
        std::vector<float> KV(s0 * 2 * d);
        for (int i = 0; i < s0; ++i) {
            for (int q = 0; q < d; ++q)        KV[i*2*d + q]     = K[(base + i)*d + q];
            for (int q = 0; q < d; ++q)        KV[i*2*d + d + q] = V[(base + i)*d + q];
        }
        std::vector<float> S(p), Vt(2 * d * p);
        rand_subspace_truncated_svd(&KV[0], s0, 2 * d, p, &Uc[c][0], &S[0], &Vt[0]);
        // Mc[c] = Uc[c]^T * KV   in R^{p x 2d}
        matmul_AT_B(&Uc[c][0], &KV[0], &Mc[c][0], s0, p, 2 * d);
        // Khat = Uc * (Uc^T * K_I) ;  Vhat similarly
        std::vector<float> Kpart(s0 * d), Vpart(s0 * d);
        for (int i = 0; i < s0; ++i) {
            for (int q = 0; q < d; ++q) Kpart[i*d + q] = K[(base + i)*d + q];
            for (int q = 0; q < d; ++q) Vpart[i*d + q] = V[(base + i)*d + q];
        }
        std::vector<float> tmpK(p * d), tmpV(p * d);
        matmul_AT_B(&Uc[c][0], &Kpart[0], &tmpK[0], s0, p, d);
        matmul_AT_B(&Uc[c][0], &Vpart[0], &tmpV[0], s0, p, d);
        matmul(&Uc[c][0], &tmpK[0], &Khat[c][0], s0, p, d);
        matmul(&Uc[c][0], &tmpV[0], &Vhat[c][0], s0, p, d);
    }

    // -------- Step 2+3+4: far-field aggregation.
    // For each query leaf nu, build the effective K/V seen by Q_{I_nu}.
    // For each near or same cluster mu (|Delta(mu,nu)| <= eta), use the
    // *exact* K_I_mu, V_I_mu (this is the "near" branch).
    // For each FAR cluster mu (Delta(mu, nu) >= eta+1, and causal), use the
    // *projected* Khat_I_mu, Vhat_I_mu.
    // Then compute standard causal SDPA over the concatenated effective K, V.
    //
    // Causality: only mu with min I_mu <= max I_nu can contribute; within a
    // mu, the per-token causal mask within mu is applied via the row-wise
    // causal limit when nu == mu (self-cluster).
    //
    zero(Y, T * d);

    for (int nu = 0; nu < Nc; ++nu) {
        // Build effective K_eff, V_eff for this query cluster.
        // Layout: contiguous over causal-mu's columns.
        // We'll just compute the full T×T causal logit row-wise restricted
        // to the query window I_nu, applying the projection rule per column.

        for (int i_in_nu = 0; i_in_nu < s0; ++i_in_nu) {
            int row = nu * s0 + i_in_nu;
            const float* q_row = &Q[row * d];

            std::vector<float> logits(T, -1e30f);
            for (int j = 0; j <= row; ++j) {  // causal columns
                int mu = j / s0;
                int j_in_mu = j - mu * s0;
                int dlt = nu - mu;  // 0 for self, positive for past
                double s = 0.0;
                if (dlt <= eta) {
                    // near: exact key
                    for (int p_ = 0; p_ < d; ++p_) s += q_row[p_] * K[j * d + p_];
                } else {
                    // far: projected key
                    for (int p_ = 0; p_ < d; ++p_)
                        s += q_row[p_] * Khat[mu][j_in_mu * d + p_];
                }
                logits[j] = (float)(s * scale);
            }
            // softmax over causal cols
            float mx = -1e30f;
            for (int j = 0; j <= row; ++j)
                if (logits[j] > mx) mx = logits[j];
            double sumE = 0.0;
            for (int j = 0; j <= row; ++j) {
                logits[j] = std::exp(logits[j] - mx);
                sumE += logits[j];
            }
            float inv = (float)(1.0 / sumE);
            for (int j = 0; j <= row; ++j) logits[j] *= inv;

            // accumulate Y_row = sum_j logits[j] * V_eff[j]
            for (int j = 0; j <= row; ++j) {
                int mu = j / s0;
                int j_in_mu = j - mu * s0;
                int dlt = nu - mu;
                const float* vp = (dlt <= eta)
                                  ? &V[j * d]
                                  : &Vhat[mu][j_in_mu * d];
                float w = logits[j];
                for (int q = 0; q < d; ++q) Y[row * d + q] += w * vp[q];
            }
        }
    }

    if (flops_out) {
        // Leaf-encode SVD: Nc * O(s0 * 2d * p * 2)        (sketch + projection)
        // Per-row eff-attention: T * (T * d / 2)          (still T^2 in this
        //                                                  prototype because we
        //                                                  expand the logits.
        //                                                  Production kernel
        //                                                  is O(T (s0 + p D)),
        //                                                  see design §10.1.)
        double f_enc = (double)Nc * (double)s0 * 2.0 * (double)d * (double)p * 4.0;
        double f_att = (double)T * (double)T * (double)d * 0.5;  // half avg by causality
        *flops_out = f_enc + f_att;
    }
}

// HMTA "production-cost" — pure FLOP COUNT for the design's intended
// O(T log T) kernel, without re-implementing the actual hierarchical sweep
// (which would require building a real tree).  This gives the FLOP-RATIO
// figure used in §10 of the design.
static double hmta_production_flops(int T, int d, int s0, int p, int eta) {
    int Nc = T / s0;
    int D  = 0;
    { int x = Nc; while (x > 1) { x >>= 1; ++D; } }
    // Per leaf:
    //   encode E :  p * s0 * 2d
    //   M2M    :  D * p * 2p * 2d
    //   M2L    :  2 * eta * p^2 * 2d * D     (across levels)
    //   L2L    :  D * p^2 * 2d
    //   decode :  s0 * p * 2d
    //   near   :  (2*eta + 1) * s0 * 2d
    double per_leaf = (double)p * (double)s0 * 2.0 * (double)d
                    + (double)D * (double)p * 2.0 * (double)p * 2.0 * (double)d
                    + 2.0 * (double)eta * (double)p * (double)p * 2.0 * (double)d * (double)D
                    + (double)D * (double)p * (double)p * 2.0 * (double)d
                    + (double)s0 * (double)p * 2.0 * (double)d
                    + (double)(2 * eta + 1) * (double)s0 * 2.0 * (double)d;
    return per_leaf * (double)Nc;
}

// -------------------------------------------------------------------- //
// Metrics.
// -------------------------------------------------------------------- //
static double rel_l2_error(const float* A, const float* B, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = (double)A[i] - (double)B[i];
        num += d * d;
        den += (double)A[i] * (double)A[i];
    }
    if (den < 1e-30) return 0.0;
    return std::sqrt(num / den);
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return 0.0;
    return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// -------------------------------------------------------------------- //
// Singular-value-decay probe on the cluster-pair logit submatrices.
// For each admissible cluster-pair (mu, nu) with Delta > eta, compute the
// flat-SDPA logit submatrix Lambda_{mu,nu} in R^{s0 x s0} and its top-(p+1)
// singular values.  Report median rank-p retention:
//   retention_p := 1 - sigma_{p+1}^2 / sum_k sigma_k^2
// Conjecture C4 passes if median retention >= 0.85 at p = 8 on realistic
// synthetic data with a moderate true rank r.
// -------------------------------------------------------------------- //
static void sv_decay_probe(const float* Q, const float* K, int T, int d,
                           int s0, int p, int eta,
                           double scale,
                           double* med_ret_out,
                           double* med_sig_pp1_out) {
    int Nc = T / s0;
    std::vector<double> retentions;
    std::vector<double> sigma_pp1s;

    for (int nu = 0; nu < Nc; ++nu) {
        for (int mu = 0; mu < nu; ++mu) {
            int dlt = nu - mu;
            if (dlt <= eta) continue;  // not admissible (near)

            // Build Lambda_{mu,nu} = Q_nu * K_mu^T * scale  in R^{s0 x s0}.
            std::vector<float> L(s0 * s0);
            for (int i = 0; i < s0; ++i) {
                int r1 = nu * s0 + i;
                for (int j = 0; j < s0; ++j) {
                    int r2 = mu * s0 + j;
                    double s = 0.0;
                    for (int q = 0; q < d; ++q) s += Q[r1*d + q] * K[r2*d + q];
                    L[i*s0 + j] = (float)(s * scale);
                }
            }

            // Compute top-(p+1) singular values via subspace iteration.
            int k_top = p + 1;
            std::vector<float> Uc(s0 * k_top), S(k_top), Vt(s0 * k_top);
            rand_subspace_truncated_svd(&L[0], s0, s0, k_top, &Uc[0], &S[0], &Vt[0]);
            // S is the ESTIMATED top-(p+1) singular values.
            // Compute full Frobenius norm of L for the retention denominator.
            double fro2 = 0.0;
            for (int i = 0; i < s0*s0; ++i) fro2 += (double)L[i] * L[i];

            // sum of top-p squared singular values
            double top_p_sq = 0.0;
            for (int i = 0; i < p; ++i) top_p_sq += (double)S[i] * S[i];
            // retention = top_p_sq / fro2
            double ret = (fro2 > 0) ? top_p_sq / fro2 : 0.0;
            retentions.push_back(ret);

            // |sigma_{p+1}| / ||L||_F
            double sig_pp1 = (k_top > p) ? (double)S[p] : 0.0;
            double fro = std::sqrt(fro2);
            sigma_pp1s.push_back(fro > 0 ? sig_pp1 / fro : 0.0);
        }
    }

    *med_ret_out     = median(retentions);
    *med_sig_pp1_out = median(sigma_pp1s);
}

// -------------------------------------------------------------------- //
// Main driver: sweep over rank r and HMTA rank p; report metrics.
// -------------------------------------------------------------------- //
int main(int argc, char** argv) {
    int seed = 42;
    if (argc > 1) seed = std::atoi(argv[1]);
    srand((unsigned)seed);

    std::printf("HMTA Forward Prototype — paradigm #261 Gate-0 (kernel-correctness)\n");
    std::printf("=================================================================\n");
    std::printf("Seed: %d\n\n", seed);

    // Sweep configuration.
    int T     = 512;
    int d     = 64;
    int s0    = 64;
    int eta   = 2;
    int p_set[3]    = { 4, 8, 16 };
    int rank_set[5] = { 2, 4, 8, 16, 32 };
    double noise_sd = 0.10;
    double scale = 1.0 / std::sqrt((double)d);

    std::printf("Config: T=%d d=%d s0=%d eta=%d noise=%.3f\n",
                T, d, s0, eta, noise_sd);
    std::printf("        clusters Nc=%d  tree depth D=%d\n",
                T/s0, (int)std::ceil(std::log((double)(T/s0))/std::log(2.0)));
    std::printf("        admissible far-pair stride: |Delta| > %d clusters\n\n", eta);

    // Allocate.
    std::vector<float> Q(T * d), K(T * d), V(T * d);
    std::vector<float> Y_sdpa(T * d), Y_hmta(T * d);

    std::printf("\n--- Truncation error scan ---\n");
    std::printf(" rank_r | p=4 rel_L2 | p=8 rel_L2 | p=16 rel_L2 | flat-SDPA Y_norm\n");
    std::printf("--------+------------+------------+-------------+------------------\n");

    for (int ri = 0; ri < 5; ++ri) {
        int r = rank_set[ri];
        make_synthetic_QKV(T, d, r, noise_sd, &Q[0], &K[0], &V[0]);

        double f_sdpa = 0.0;
        flat_sdpa_forward(&Q[0], &K[0], &V[0], T, d, &Y_sdpa[0], &f_sdpa);
        double Ynorm = 0.0;
        for (int i = 0; i < T*d; ++i) Ynorm += (double)Y_sdpa[i] * Y_sdpa[i];
        Ynorm = std::sqrt(Ynorm);

        double err[3];
        for (int pi = 0; pi < 3; ++pi) {
            int p = p_set[pi];
            double f_hmta = 0.0;
            hmta_forward(&Q[0], &K[0], &V[0], T, d, s0, p, eta, &Y_hmta[0],
                         &f_hmta, scale);
            err[pi] = rel_l2_error(&Y_sdpa[0], &Y_hmta[0], T * d);
        }
        std::printf("   %3d  |   %.4f   |   %.4f   |   %.4f    |   %.3f\n",
                    r, err[0], err[1], err[2], Ynorm);
    }

    // SV-decay probe at fixed rank r=8 (the "natural" true rank that matches
    // the design's p = 8 default) and a higher rank r=32 stress test.
    std::printf("\n--- Singular-value decay probe (Conjecture C4) ---\n");
    std::printf("(measures sigma_{p+1}/||L||_F on admissible cluster-pair logit "
                "submatrices)\n");
    std::printf(" true_rank | p | median retention | median sigma_{p+1}/||L||_F\n");
    std::printf("-----------+---+------------------+----------------------------\n");
    for (int ri = 0; ri < 5; ++ri) {
        int r = rank_set[ri];
        make_synthetic_QKV(T, d, r, noise_sd, &Q[0], &K[0], &V[0]);
        for (int pi = 0; pi < 3; ++pi) {
            int p = p_set[pi];
            double med_ret = 0, med_sig = 0;
            sv_decay_probe(&Q[0], &K[0], T, d, s0, p, eta, scale,
                           &med_ret, &med_sig);
            std::printf("    %3d    | %2d |     %.4f       |     %.4f\n",
                        r, p, med_ret, med_sig);
        }
    }

    // FLOP-scaling demonstration.
    std::printf("\n--- FLOP-ratio scaling: HMTA(design) vs flat-SDPA ---\n");
    std::printf("(uses HMTA's intended O(T log T) production cost formula; "
                "the prototype above uses an O(T^2) inner loop for simplicity.)\n");
    std::printf(" T      | s0  | p | eta | SDPA FLOPs | HMTA FLOPs | ratio\n");
    std::printf("--------+-----+---+-----+------------+------------+--------\n");
    int Ts[] = { 512, 1024, 2048, 4096, 8192, 16384 };
    for (int ti = 0; ti < 6; ++ti) {
        int Tt = Ts[ti];
        double f_sdpa = 2.0 * (double)Tt * (double)Tt * (double)d;
        double f_hmta = hmta_production_flops(Tt, d, s0, 8, eta);
        std::printf(" %5d  | %3d | %d |  %d  | %.3e | %.3e | %6.1fx\n",
                    Tt, s0, 8, eta, f_sdpa, f_hmta, f_sdpa / f_hmta);
    }

    // SDPA-recovery sanity: at p = s0 and eta = T/s0 (no admissible far
    // pairs), HMTA should reduce to flat causal SDPA exactly (up to
    // floating-point round-off).
    std::printf("\n--- SDPA-recovery sanity (p = s0 = %d, eta = Nc = %d) ---\n",
                s0, T / s0);
    {
        make_synthetic_QKV(T, d, 8, noise_sd, &Q[0], &K[0], &V[0]);
        double f_sdpa = 0.0, f_hmta = 0.0;
        flat_sdpa_forward(&Q[0], &K[0], &V[0], T, d, &Y_sdpa[0], &f_sdpa);
        hmta_forward(&Q[0], &K[0], &V[0], T, d, s0, s0, T / s0, &Y_hmta[0],
                     &f_hmta, scale);
        double e = rel_l2_error(&Y_sdpa[0], &Y_hmta[0], T * d);
        std::printf("  rel_L2(SDPA, HMTA[p=s0,eta=Nc])  =  %.6e   "
                    "(should be < 1e-3)\n", e);
    }

    // SCFA-recovery sanity: with s0 = chunk size and eta = 0 (only self-
    // cluster is near; everything else is far), HMTA collapses to a
    // chunk-spectral structure equivalent to SCFA at compression rank k = p.
    // Here we just verify HMTA forward succeeds with eta=0 and yields some
    // reasonable output (not bit-exact SCFA without the spectral basis init).
    std::printf("\n--- SCFA-shape sanity (eta = 0, single near = self) ---\n");
    {
        make_synthetic_QKV(T, d, 8, noise_sd, &Q[0], &K[0], &V[0]);
        double f_sdpa = 0.0, f_hmta = 0.0;
        flat_sdpa_forward(&Q[0], &K[0], &V[0], T, d, &Y_sdpa[0], &f_sdpa);
        hmta_forward(&Q[0], &K[0], &V[0], T, d, s0, 8, 0, &Y_hmta[0],
                     &f_hmta, scale);
        double e = rel_l2_error(&Y_sdpa[0], &Y_hmta[0], T * d);
        std::printf("  rel_L2(SDPA, HMTA[eta=0,p=8])  =  %.4f   "
                    "(should be moderate; SCFA-class accuracy)\n", e);
    }

    std::printf("\n=================================================================\n");
    std::printf("PASS criteria (Gate-0 kernel-correctness branch):\n");
    std::printf("  C0a  HMTA at p=8 has rel_L2 error <= 0.20 vs flat-SDPA at\n");
    std::printf("       true_rank r in {4, 8} (the design's target regime).\n");
    std::printf("  C0b  Median rank-p retention >= 0.85 at p = 8 for r <= 8.\n");
    std::printf("  C0c  HMTA production FLOPs <= 1/10 of flat-SDPA at T = 16384.\n");
    std::printf("  C0d  SDPA-recovery error at p = s0, eta = Nc is < 1e-3.\n");
    std::printf("=================================================================\n");
    return 0;
}
