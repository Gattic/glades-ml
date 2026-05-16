// HMTA Forward Prototype v2 — paradigm #261 iter 38.
//
// Extensions over v1 (research/hmta_forward_prototype.cpp):
//   (i)  Separate K-only and V-only SVD bases per leaf (instead of joint
//        rank-p SVD of [K | V]).  This is the L2-optimal basis CHOICE for
//        the attention LOGIT (which involves K only) and the V-output
//        (which involves V only), and should give a strictly tighter
//        forward L2 error than v1's joint-basis prototype.
//   (ii) Multi-level multipole forward: builds the cluster tree to depth
//        D = ceil(log2(Nc)), with admissibility decided per level.  Far
//        pairs at level l use the LEVEL-l moments (which compress 2^l
//        leaves of tokens into a single rank-p moment).  Single-level v1
//        is the special case D = 0.
//   (iii) Comparison table: v1 (joint), K-only-V-only-single-level,
//        K-only-V-only-multi-level.
//
// Gate-0 conjecture C0a-v2 (stated BEFORE running):
//   With K-only and V-only separate-rank-p bases per leaf (single-level
//   or multi-level), HMTA forward L2 error vs flat-SDPA at p=8 across
//   true-rank r in {4, 8, 16} drops to median <= 0.15 (vs v1's 0.22).
//
// Build:
//   g++ -std=c++98 -O2 -Wall -Wextra research/hmta_forward_v2.cpp \
//       -o research/hmta_forward_v2
// Run:
//   ./research/hmta_forward_v2 <seed>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

// -------------------------------------------------------------------- //
// RNG helpers.
static double uniform_open() {
    double u = (double(rand()) + 1.0) / (double(RAND_MAX) + 2.0);
    return u;
}
static double gauss(double m, double s) {
    double u1 = uniform_open();
    double u2 = uniform_open();
    return m + s * std::sqrt(-2.0 * std::log(u1)) *
                  std::cos(2.0 * 3.14159265358979323846 * u2);
}
static void zero(float* A, int n) { for (int i = 0; i < n; ++i) A[i] = 0.0f; }
static void fill_gauss(float* A, int n, double sd) {
    for (int i = 0; i < n; ++i) A[i] = (float)gauss(0.0, sd);
}

// -------------------------------------------------------------------- //
// Matrix helpers (row-major).
static void matmul(const float* A, const float* B, float* C,
                   int M, int K, int N) {
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) s += A[i*K + k] * B[k*N + j];
        C[i*N + j] = (float)s;
    }
}
static void matmul_AT_B(const float* A, const float* B, float* C,
                        int K, int M, int N) {
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) s += A[k*M + i] * B[k*N + j];
        C[i*N + j] = (float)s;
    }
}

// Top-p left-singular subspace of M×N matrix via subspace iteration.
// Returns U:M×p (orthonormal cols).
static void left_subspace(const float* A, int M, int N, int p, float* U) {
    std::vector<float> Omega(N * p), Y(M * p);
    fill_gauss(&Omega[0], N * p, 1.0);
    matmul(A, &Omega[0], &Y[0], M, N, p);
    // 2 passes of subspace iteration for accuracy.
    for (int pass = 0; pass < 2; ++pass) {
        // QR via modified Gram-Schmidt.
        for (int j = 0; j < p; ++j) {
            for (int i = 0; i < j; ++i) {
                double dot = 0.0;
                for (int r = 0; r < M; ++r) dot += Y[r*p + i] * Y[r*p + j];
                for (int r = 0; r < M; ++r) Y[r*p + j] -= (float)dot * Y[r*p + i];
            }
            double n = 0.0;
            for (int r = 0; r < M; ++r) n += (double)Y[r*p + j] * Y[r*p + j];
            n = std::sqrt(n);
            if (n < 1e-12) n = 1.0;
            float inv = (float)(1.0 / n);
            for (int r = 0; r < M; ++r) Y[r*p + j] *= inv;
        }
        if (pass == 0) {
            // Y <- A (A^T Y)  (subspace iteration step)
            std::vector<float> tmp(N * p);
            matmul_AT_B(A, &Y[0], &tmp[0], M, N, p);
            matmul(A, &tmp[0], &Y[0], M, N, p);
        }
    }
    for (int i = 0; i < M*p; ++i) U[i] = Y[i];
}

// -------------------------------------------------------------------- //
// Synthetic Q, K, V with controllable underlying rank.
static void make_synthetic_QKV(int T, int d, int r, double noise_sd,
                               float* Q, float* K, float* V) {
    std::vector<float> U_q(d * r), U_k(d * r);
    fill_gauss(&U_q[0], d * r, 1.0 / std::sqrt((double)d));
    fill_gauss(&U_k[0], d * r, 1.0 / std::sqrt((double)d));
    std::vector<float> c(T * r), e(T * r);
    fill_gauss(&c[0], T * r, 1.0);
    fill_gauss(&e[0], T * r, 1.0);
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
// Causal SDPA (the reference).
static void flat_sdpa(const float* Q, const float* K, const float* V,
                      int T, int d, float* Y) {
    std::vector<float> L(T * T);
    double scale = 1.0 / std::sqrt((double)d);
    for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j) {
        double s = 0.0;
        for (int p = 0; p < d; ++p) s += Q[i*d + p] * K[j*d + p];
        L[i*T + j] = (float)(s * scale);
    }
    // Causal softmax per row.
    for (int i = 0; i < T; ++i) {
        float mx = -1e30f;
        for (int j = 0; j <= i; ++j) if (L[i*T + j] > mx) mx = L[i*T + j];
        double sum = 0.0;
        for (int j = 0; j < T; ++j) {
            if (j > i) { L[i*T + j] = 0.0f; }
            else { float e = std::exp(L[i*T + j] - mx); L[i*T + j] = e; sum += e; }
        }
        if (sum > 0) {
            float inv = (float)(1.0/sum);
            for (int j = 0; j <= i; ++j) L[i*T + j] *= inv;
        }
    }
    matmul(&L[0], V, Y, T, T, d);
}

// -------------------------------------------------------------------- //
// HMTA forward with configurable basis strategy.
//   basis = 0  : JOINT  (v1 prototype: SVD of [K | V] as one s0 x 2d matrix)
//   basis = 1  : SEPARATE single-level (K-only SVD for K, V-only SVD for V)
//   basis = 2  : SEPARATE multi-level (separate K/V SVDs at level 0;
//                level >=1 moments computed by re-encoding the union of
//                child token spans through a fresh SVD; admissibility
//                checked per-level so coarser pairs use coarser moments)
//
// At basis=0 this should reproduce v1 (mod RNG order).
// At basis=1, V is reconstructed using its OWN basis, so V error is
// minimized independently of K error.  Forward L2 should drop.
// At basis=2, coarser-level moments cover wider spans -> more compression
// -> POSSIBLY WORSE per-pair L2 but FEWER pairs -> aggregate similar.
//
// Production scale uses basis=2 (multi-level).  This prototype lets us
// quantify the cost of each step.
// -------------------------------------------------------------------- //

struct LeafCache {
    std::vector<float> Uk;   // s0 x p   (K basis)
    std::vector<float> Uv;   // s0 x p   (V basis)
    std::vector<float> Uj;   // s0 x p   (joint basis for basis=0)
    std::vector<float> Khat; // s0 x d   (projected K)
    std::vector<float> Vhat; // s0 x d   (projected V)
};

static void encode_leaf(const float* K_part, const float* V_part,
                        int s0, int d, int p, int basis,
                        LeafCache* out) {
    out->Uk.assign(s0 * p, 0.0f);
    out->Uv.assign(s0 * p, 0.0f);
    out->Uj.assign(s0 * p, 0.0f);
    out->Khat.assign(s0 * d, 0.0f);
    out->Vhat.assign(s0 * d, 0.0f);

    if (basis == 0) {
        // Joint [K | V] SVD.
        std::vector<float> KV(s0 * 2 * d);
        for (int i = 0; i < s0; ++i) {
            for (int q = 0; q < d; ++q) KV[i*2*d + q]     = K_part[i*d + q];
            for (int q = 0; q < d; ++q) KV[i*2*d + d + q] = V_part[i*d + q];
        }
        left_subspace(&KV[0], s0, 2 * d, p, &out->Uj[0]);
        // K_hat = U U^T K
        std::vector<float> tmp(p * d);
        matmul_AT_B(&out->Uj[0], K_part, &tmp[0], s0, p, d);
        matmul(&out->Uj[0], &tmp[0], &out->Khat[0], s0, p, d);
        matmul_AT_B(&out->Uj[0], V_part, &tmp[0], s0, p, d);
        matmul(&out->Uj[0], &tmp[0], &out->Vhat[0], s0, p, d);
    } else {
        // Separate K-only and V-only bases.
        left_subspace(K_part, s0, d, p, &out->Uk[0]);
        left_subspace(V_part, s0, d, p, &out->Uv[0]);
        // K_hat = U_k U_k^T K  ;  V_hat = U_v U_v^T V
        std::vector<float> tmp(p * d);
        matmul_AT_B(&out->Uk[0], K_part, &tmp[0], s0, p, d);
        matmul(&out->Uk[0], &tmp[0], &out->Khat[0], s0, p, d);
        matmul_AT_B(&out->Uv[0], V_part, &tmp[0], s0, p, d);
        matmul(&out->Uv[0], &tmp[0], &out->Vhat[0], s0, p, d);
    }
}

// Multi-level: pre-compute per-level moments by encoding the union of
// child token spans through a fresh SVD.  Per-level moments correspond to
// 2^l adjacent leaves combined.
static void build_multilevel(const float* K, const float* V,
                             int T, int d, int s0, int p,
                             std::vector< std::vector<LeafCache> >* tree) {
    int Nc = T / s0;
    int D = 0;
    { int x = Nc; while (x > 1) { x >>= 1; ++D; } }

    tree->clear();
    tree->resize(D + 1);

    // Level 0: per leaf.
    (*tree)[0].resize(Nc);
    for (int c = 0; c < Nc; ++c) {
        encode_leaf(&K[c*s0*d], &V[c*s0*d], s0, d, p, /*basis=*/1, &(*tree)[0][c]);
    }

    // Level l >= 1: encode the span of 2^l leaves with a fresh SVD.
    for (int l = 1; l <= D; ++l) {
        int span = s0 * (1 << l);
        int Ncl  = Nc >> l;
        if (Ncl < 1) Ncl = 1;
        (*tree)[l].resize(Ncl);
        for (int c = 0; c < Ncl; ++c) {
            const float* Kspan = &K[c * span * d];
            const float* Vspan = &V[c * span * d];
            // For consistency we reuse the encode_leaf with s0 := span.
            LeafCache cache;
            cache.Uk.assign(span * p, 0.0f);
            cache.Uv.assign(span * p, 0.0f);
            cache.Khat.assign(span * d, 0.0f);
            cache.Vhat.assign(span * d, 0.0f);
            left_subspace(Kspan, span, d, p, &cache.Uk[0]);
            left_subspace(Vspan, span, d, p, &cache.Uv[0]);
            std::vector<float> tmp(p * d);
            matmul_AT_B(&cache.Uk[0], Kspan, &tmp[0], span, p, d);
            matmul(&cache.Uk[0], &tmp[0], &cache.Khat[0], span, p, d);
            matmul_AT_B(&cache.Uv[0], Vspan, &tmp[0], span, p, d);
            matmul(&cache.Uv[0], &tmp[0], &cache.Vhat[0], span, p, d);
            (*tree)[l][c] = cache;
        }
    }
}

// HMTA forward routing:
//   For each query token i, partition its causal-attention budget [0..i]
//   into "near" (closest 2*eta+1 leaves) and "far" (rest).
//   The "far" region at depth l is the set of tokens whose level-l ancestor
//   cluster differs from i's level-l ancestor by EXACTLY one (i.e., they
//   become admissible at level l).  At basis=2 we look up the corresponding
//   level-l moment; at basis=1 we use level-0 only (everything beyond the
//   near window uses the per-leaf projection).
static void hmta_forward(const float* Q, const float* K, const float* V,
                         int T, int d, int s0, int p, int eta, int basis,
                         float* Y) {
    int Nc = T / s0;
    int D  = 0;
    { int x = Nc; while (x > 1) { x >>= 1; ++D; } }

    std::vector< std::vector<LeafCache> > tree;
    if (basis == 2) {
        build_multilevel(K, V, T, d, s0, p, &tree);
    } else {
        tree.resize(1);
        tree[0].resize(Nc);
        for (int c = 0; c < Nc; ++c) {
            encode_leaf(&K[c*s0*d], &V[c*s0*d], s0, d, p, basis, &tree[0][c]);
        }
    }

    double scale = 1.0 / std::sqrt((double)d);
    zero(Y, T * d);

    for (int row = 0; row < T; ++row) {
        int nu = row / s0;
        const float* q_row = &Q[row * d];

        // For each causal column j <= row, decide near/far AND, if far,
        // determine the level at which we summarize (basis=2 only).
        std::vector<float> logits(T, -1e30f);
        std::vector<const float*> Vsrc(T, (const float*)0);
        std::vector<const float*> Ksrc(T, (const float*)0);

        for (int j = 0; j <= row; ++j) {
            int mu = j / s0;
            int dlt = nu - mu;
            const float* kp = 0;
            const float* vp = 0;
            if (dlt <= eta) {
                // Near: exact.
                kp = &K[j * d];
                vp = &V[j * d];
            } else if (basis < 2) {
                // Far, single-level: use leaf-level projection.
                int j_in_mu = j - mu * s0;
                kp = &tree[0][mu].Khat[j_in_mu * d];
                vp = &tree[0][mu].Vhat[j_in_mu * d];
            } else {
                // Far, multi-level: pick the COARSEST level at which this
                // pair becomes admissible (admissibility at level l means
                // cluster distance at level l exceeds eta).  Levels finer
                // than the necessary one would still be admissible but use
                // more (i.e., redundant) information; we want the COARSEST
                // — that maximizes compression and tests the production
                // FLOP claim.
                int chosen_l = 0;
                for (int l = D; l >= 1; --l) {
                    int Ncl = Nc >> l;
                    if (Ncl < 1) Ncl = 1;
                    int nu_l = nu >> l;
                    int mu_l = mu >> l;
                    int dl   = nu_l - mu_l;
                    if (dl > eta) { chosen_l = l; break; }
                }
                int span = s0 * (1 << chosen_l);
                int cl   = j / span;
                int j_in_cl = j - cl * span;
                if ((int)tree[chosen_l].size() <= cl) {
                    // bounds guard
                    int j_in_mu = j - mu * s0;
                    kp = &tree[0][mu].Khat[j_in_mu * d];
                    vp = &tree[0][mu].Vhat[j_in_mu * d];
                } else {
                    kp = &tree[chosen_l][cl].Khat[j_in_cl * d];
                    vp = &tree[chosen_l][cl].Vhat[j_in_cl * d];
                }
            }
            double s = 0.0;
            for (int q = 0; q < d; ++q) s += q_row[q] * kp[q];
            logits[j] = (float)(s * scale);
            Vsrc[j] = vp;
            Ksrc[j] = kp;
        }

        // softmax over causal range
        float mx = -1e30f;
        for (int j = 0; j <= row; ++j) if (logits[j] > mx) mx = logits[j];
        double sum = 0.0;
        for (int j = 0; j <= row; ++j) {
            logits[j] = std::exp(logits[j] - mx);
            sum += logits[j];
        }
        float inv = (float)(1.0 / sum);
        for (int j = 0; j <= row; ++j) logits[j] *= inv;

        // accumulate Y
        for (int j = 0; j <= row; ++j) {
            float w = logits[j];
            const float* vp = Vsrc[j];
            for (int q = 0; q < d; ++q) Y[row * d + q] += w * vp[q];
        }
    }
}

// -------------------------------------------------------------------- //
// FLOP estimator for design-cost (matches v1).
static double hmta_production_flops(int T, int d, int s0, int p, int eta) {
    int Nc = T / s0;
    int D  = 0; { int x = Nc; while (x > 1) { x >>= 1; ++D; } }
    double per_leaf = (double)p * (double)s0 * 2.0 * (double)d
        + (double)D * (double)p * 2.0 * (double)p * 2.0 * (double)d
        + 2.0 * (double)eta * (double)p * (double)p * 2.0 * (double)d * (double)D
        + (double)D * (double)p * (double)p * 2.0 * (double)d
        + (double)s0 * (double)p * 2.0 * (double)d
        + (double)(2 * eta + 1) * (double)s0 * 2.0 * (double)d;
    return per_leaf * (double)Nc;
}

static double rel_l2(const float* A, const float* B, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        double dd = (double)A[i] - (double)B[i];
        num += dd * dd; den += (double)A[i] * (double)A[i];
    }
    return den < 1e-30 ? 0 : std::sqrt(num / den);
}

static double median_f(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return 0.0;
    return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// -------------------------------------------------------------------- //
int main(int argc, char** argv) {
    int seed = (argc > 1) ? std::atoi(argv[1]) : 42;
    srand((unsigned)seed);

    std::printf("HMTA Forward Prototype v2 — paradigm #261 iter 38 (basis study)\n");
    std::printf("===============================================================\n");
    std::printf("Seed: %d\n\n", seed);

    int T = 512, d = 64, s0 = 64, eta = 2;
    int p_set[3]    = { 4, 8, 16 };
    int rank_set[5] = { 2, 4, 8, 16, 32 };
    double noise_sd = 0.10;

    std::printf("Config: T=%d d=%d s0=%d eta=%d noise=%.3f\n",
                T, d, s0, eta, noise_sd);
    std::printf("Clusters Nc=%d, tree depth D=%d\n\n",
                T/s0, (int)std::ceil(std::log((double)(T/s0))/std::log(2.0)));

    std::vector<float> Q(T*d), K(T*d), V(T*d);
    std::vector<float> Y_sdpa(T*d), Y_hmta(T*d);

    const char* basis_name[3] = { "joint[K|V]", "K-only,V-only", "multi-level" };

    // Sweep over (basis, p, rank).
    std::printf("=== Forward L2 error table (5 seeds median) ===\n");
    for (int b = 0; b < 3; ++b) {
        std::printf("\n  basis = %s\n", basis_name[b]);
        std::printf("  r \\ p |   p=4   |   p=8   |   p=16\n");
        std::printf("  ------+---------+---------+--------\n");
        for (int ri = 0; ri < 5; ++ri) {
            int r = rank_set[ri];
            double errs[3] = {0,0,0};
            int seeds[5] = { 7, 13, 42, 100, 1729 };
            for (int si = 0; si < 5; ++si) {
                std::vector<double> per_seed(3, 0);
                srand((unsigned)seeds[si]);
                make_synthetic_QKV(T, d, r, noise_sd, &Q[0], &K[0], &V[0]);
                flat_sdpa(&Q[0], &K[0], &V[0], T, d, &Y_sdpa[0]);
                for (int pi = 0; pi < 3; ++pi) {
                    hmta_forward(&Q[0], &K[0], &V[0], T, d, s0, p_set[pi], eta, b,
                                 &Y_hmta[0]);
                    errs[pi] += rel_l2(&Y_sdpa[0], &Y_hmta[0], T*d);
                }
            }
            std::printf("  r=%2d  | %.4f  | %.4f  | %.4f\n",
                        r, errs[0]/5, errs[1]/5, errs[2]/5);
        }
    }
    srand((unsigned)seed);

    // SDPA recovery sanity for the new bases.
    std::printf("\n=== SDPA recovery sanity (p=s0=%d, eta=Nc=%d, rank=8) ===\n",
                s0, T/s0);
    make_synthetic_QKV(T, d, 8, noise_sd, &Q[0], &K[0], &V[0]);
    flat_sdpa(&Q[0], &K[0], &V[0], T, d, &Y_sdpa[0]);
    for (int b = 0; b < 3; ++b) {
        hmta_forward(&Q[0], &K[0], &V[0], T, d, s0, s0, T/s0, b, &Y_hmta[0]);
        double e = rel_l2(&Y_sdpa[0], &Y_hmta[0], T*d);
        std::printf("  %s : rel_L2 = %.6e\n", basis_name[b], e);
    }

    // FLOP scaling.
    std::printf("\n=== FLOP scaling (design O(T log T) formula, p=8 eta=2 s0=64) ===\n");
    std::printf(" T      | SDPA FLOPs   | HMTA FLOPs   | ratio\n");
    std::printf("--------+--------------+--------------+--------\n");
    int Ts[] = { 512, 1024, 2048, 4096, 8192, 16384 };
    for (int ti = 0; ti < 6; ++ti) {
        int Tt = Ts[ti];
        double f_sdpa = 2.0 * (double)Tt * (double)Tt * (double)d;
        double f_hmta = hmta_production_flops(Tt, d, s0, 8, eta);
        std::printf(" %5d  | %.3e   | %.3e   | %6.1fx\n",
                    Tt, f_sdpa, f_hmta, f_sdpa/f_hmta);
    }

    // C0a-v2 PASS/FAIL.
    std::printf("\n=== C0a-v2 verdict ===\n");
    std::printf("Target: median forward L2 at p=8, r in {4,8,16} <= 0.15.\n");
    {
        double med_v1 = 0, med_v2 = 0, med_v3 = 0;
        std::vector<double> v1_set, v2_set, v3_set;
        int rs[3] = { 4, 8, 16 };
        int seeds[5] = { 7, 13, 42, 100, 1729 };
        for (int ri = 0; ri < 3; ++ri) {
            for (int si = 0; si < 5; ++si) {
                srand((unsigned)seeds[si]);
                make_synthetic_QKV(T, d, rs[ri], noise_sd, &Q[0], &K[0], &V[0]);
                flat_sdpa(&Q[0], &K[0], &V[0], T, d, &Y_sdpa[0]);
                hmta_forward(&Q[0], &K[0], &V[0], T, d, s0, 8, eta, 0, &Y_hmta[0]);
                v1_set.push_back(rel_l2(&Y_sdpa[0], &Y_hmta[0], T*d));
                hmta_forward(&Q[0], &K[0], &V[0], T, d, s0, 8, eta, 1, &Y_hmta[0]);
                v2_set.push_back(rel_l2(&Y_sdpa[0], &Y_hmta[0], T*d));
                hmta_forward(&Q[0], &K[0], &V[0], T, d, s0, 8, eta, 2, &Y_hmta[0]);
                v3_set.push_back(rel_l2(&Y_sdpa[0], &Y_hmta[0], T*d));
            }
        }
        med_v1 = median_f(v1_set);
        med_v2 = median_f(v2_set);
        med_v3 = median_f(v3_set);
        std::printf("  basis=joint[K|V]   (v1)            median = %.4f\n", med_v1);
        std::printf("  basis=K-only,V-only single-level   median = %.4f\n", med_v2);
        std::printf("  basis=multi-level                  median = %.4f\n", med_v3);
        std::printf("\n  C0a-v2 PASS iff K-only median <= 0.15: %s\n",
                    med_v2 <= 0.15 ? "PASS" : "FAIL");
    }

    return 0;
}
