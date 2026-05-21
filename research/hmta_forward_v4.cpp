// HMTA Forward Prototype v4 — paradigm #261 iter 40 (F1: separate p_K, p_V).
//
// Iter 39 established that the iter-37/38 forward-L2 floor at p=8 is set
// by rank-p V-projection loss (discards (d-p)/d of V variance), NOT by
// noise or basis-choice issues.  This iter tests F1: use DIFFERENT ranks
// for K and V projections within HMTA's leaf encoder.
//
// Hypothesis: a small p_K (sufficient for similarity scoring) combined
// with a larger p_V (preserves V information) gives a Pareto-better
// design point than the unified p of iters 37-39.
//
// Sweep:
//   p_K in { 4, 8 }
//   p_V in { 8, 16, 32, 48, 64 }
//
// Tests:
//   (A) Gaussian synthetic at rank r=8, noise sd=0.10.
//   (B) Sink+local synthetic (4 sinks per T=512, sink scale 10x).
//
// Gate-0 conjecture C0-iter40:
//   For some (p_K, p_V), forward L2 vs flat-SDPA <= 0.15 on BOTH Gaussian
//   AND sink+local, while production FLOP ratio at T=16384 >= 10x over
//   flat-SDPA.
//
// Build:
//   g++ -std=c++98 -O2 -Wall -Wextra research/hmta_forward_v4.cpp \
//       -o research/hmta_forward_v4
// Run:
//   ./research/hmta_forward_v4 <seed>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

static double uniform_open() {
    return (double(rand()) + 1.0) / (double(RAND_MAX) + 2.0);
}
static double gauss(double m, double s) {
    double u1 = uniform_open(), u2 = uniform_open();
    return m + s * std::sqrt(-2.0 * std::log(u1)) *
                  std::cos(2.0 * 3.14159265358979323846 * u2);
}
static void fill_gauss(float* A, int n, double sd) {
    for (int i = 0; i < n; ++i) A[i] = (float)gauss(0.0, sd);
}
static void zero(float* A, int n) { for (int i = 0; i < n; ++i) A[i] = 0.0f; }

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

// Top-p left-singular subspace via subspace iteration.
static void left_subspace(const float* A, int M, int N, int p, float* U) {
    std::vector<float> Om(N * p), Y(M * p);
    fill_gauss(&Om[0], N * p, 1.0);
    matmul(A, &Om[0], &Y[0], M, N, p);
    for (int pass = 0; pass < 2; ++pass) {
        for (int j = 0; j < p; ++j) {
            for (int i = 0; i < j; ++i) {
                double dot = 0.0;
                for (int r = 0; r < M; ++r) dot += Y[r*p + i] * Y[r*p + j];
                for (int r = 0; r < M; ++r) Y[r*p + j] -= (float)dot * Y[r*p + i];
            }
            double n = 0;
            for (int r = 0; r < M; ++r) n += (double)Y[r*p + j] * Y[r*p + j];
            n = std::sqrt(n);
            if (n < 1e-12) n = 1.0;
            float inv = (float)(1.0 / n);
            for (int r = 0; r < M; ++r) Y[r*p + j] *= inv;
        }
        if (pass == 0) {
            std::vector<float> tmp(N * p);
            matmul_AT_B(A, &Y[0], &tmp[0], M, N, p);
            matmul(A, &tmp[0], &Y[0], M, N, p);
        }
    }
    for (int i = 0; i < M*p; ++i) U[i] = Y[i];
}

// ---- Synthetic data generators (same as v3) ----
static void make_gaussian(int T, int d, int r, double noise_sd,
                          float* Q, float* K, float* V) {
    std::vector<float> Uq(d * r), Uk(d * r), c(T * r), e(T * r);
    fill_gauss(&Uq[0], d * r, 1.0 / std::sqrt((double)d));
    fill_gauss(&Uk[0], d * r, 1.0 / std::sqrt((double)d));
    fill_gauss(&c[0], T * r, 1.0);
    fill_gauss(&e[0], T * r, 1.0);
    for (int i = 0; i < T; ++i) {
        for (int p = 0; p < d; ++p) {
            double s = 0.0;
            for (int rr = 0; rr < r; ++rr) s += Uq[p*r + rr] * c[i*r + rr];
            s += gauss(0.0, noise_sd);
            Q[i*d + p] = (float)s;
        }
        for (int p = 0; p < d; ++p) {
            double s = 0.0;
            for (int rr = 0; rr < r; ++rr) s += Uk[p*r + rr] * e[i*r + rr];
            s += gauss(0.0, noise_sd);
            K[i*d + p] = (float)s;
        }
    }
    fill_gauss(V, T * d, 1.0);
}

static void make_sink_local(int T, int d, int n_sinks, double sink_scale,
                            double bg_scale, float* Q, float* K, float* V) {
    fill_gauss(Q, T * d, 1.0);
    fill_gauss(K, T * d, bg_scale);
    fill_gauss(V, T * d, 1.0);
    for (int s = 0; s < n_sinks; ++s) {
        int p = ((s + 1) * T) / (n_sinks + 1);
        for (int q = 0; q < d; ++q) K[p*d + q] = (float)gauss(0.0, sink_scale);
    }
    // Position-aligned rank-2 component (local-decay structure).
    std::vector<float> pe(T * 2);
    fill_gauss(&pe[0], T * 2, 1.0);
    for (int i = 0; i < T; ++i) {
        Q[i*d + d - 1] += 0.5f * pe[i*2];
        Q[i*d + d - 2] += 0.5f * pe[i*2 + 1];
        K[i*d + d - 1] += 0.5f * pe[i*2];
        K[i*d + d - 2] += 0.5f * pe[i*2 + 1];
    }
}

// ---- Flat causal SDPA reference ----
static void flat_sdpa(const float* Q, const float* K, const float* V,
                      int T, int d, float* Y) {
    std::vector<float> L(T * T);
    double scale = 1.0 / std::sqrt((double)d);
    for (int i = 0; i < T; ++i) for (int j = 0; j < T; ++j) {
        double s = 0.0;
        for (int p = 0; p < d; ++p) s += Q[i*d + p] * K[j*d + p];
        L[i*T + j] = (float)(s * scale);
    }
    for (int i = 0; i < T; ++i) {
        float mx = -1e30f;
        for (int j = 0; j <= i; ++j) if (L[i*T + j] > mx) mx = L[i*T + j];
        double sum = 0;
        for (int j = 0; j < T; ++j) {
            if (j > i) L[i*T + j] = 0;
            else { float e = std::exp(L[i*T + j] - mx); L[i*T + j] = e; sum += e; }
        }
        if (sum > 0) {
            float inv = (float)(1.0 / sum);
            for (int j = 0; j <= i; ++j) L[i*T + j] *= inv;
        }
    }
    matmul(&L[0], V, Y, T, T, d);
}

// HMTA forward with SEPARATE p_K, p_V.
static void hmta_separate(const float* Q, const float* K, const float* V,
                          int T, int d, int s0, int p_K, int p_V, int eta,
                          float* Y) {
    int Nc = T / s0;
    std::vector< std::vector<float> > Uk(Nc), Uv(Nc), Kh(Nc), Vh(Nc);
    for (int c = 0; c < Nc; ++c) {
        Uk[c].assign(s0 * p_K, 0);
        Uv[c].assign(s0 * p_V, 0);
        Kh[c].assign(s0 * d, 0);
        Vh[c].assign(s0 * d, 0);
        left_subspace(&K[c*s0*d], s0, d, p_K, &Uk[c][0]);
        left_subspace(&V[c*s0*d], s0, d, p_V, &Uv[c][0]);
        std::vector<float> tmpK(p_K * d), tmpV(p_V * d);
        matmul_AT_B(&Uk[c][0], &K[c*s0*d], &tmpK[0], s0, p_K, d);
        matmul(&Uk[c][0], &tmpK[0], &Kh[c][0], s0, p_K, d);
        matmul_AT_B(&Uv[c][0], &V[c*s0*d], &tmpV[0], s0, p_V, d);
        matmul(&Uv[c][0], &tmpV[0], &Vh[c][0], s0, p_V, d);
    }
    double scale = 1.0 / std::sqrt((double)d);
    zero(Y, T * d);
    for (int row = 0; row < T; ++row) {
        int nu = row / s0;
        const float* qr = &Q[row * d];
        std::vector<float> lg(T, -1e30f);
        std::vector<const float*> Vs(T, (const float*)0);
        for (int j = 0; j <= row; ++j) {
            int mu = j / s0;
            int dlt = nu - mu;
            const float* kp;
            const float* vp;
            if (dlt <= eta) { kp = &K[j*d]; vp = &V[j*d]; }
            else { int jm = j - mu * s0; kp = &Kh[mu][jm*d]; vp = &Vh[mu][jm*d]; }
            double s = 0;
            for (int q = 0; q < d; ++q) s += qr[q] * kp[q];
            lg[j] = (float)(s * scale);
            Vs[j] = vp;
        }
        float mx = -1e30f;
        for (int j = 0; j <= row; ++j) if (lg[j] > mx) mx = lg[j];
        double sum = 0;
        for (int j = 0; j <= row; ++j) { lg[j] = std::exp(lg[j] - mx); sum += lg[j]; }
        float inv = (float)(1.0 / sum);
        for (int j = 0; j <= row; ++j) lg[j] *= inv;
        for (int j = 0; j <= row; ++j) {
            float w = lg[j];
            const float* vp = Vs[j];
            for (int q = 0; q < d; ++q) Y[row * d + q] += w * vp[q];
        }
    }
}

// Per-layer production FLOPs with separate p_K, p_V.
//   Leaf-K encode: p_K * s0 * d
//   Leaf-V encode: p_V * s0 * d
//   M2M K: D * p_K * 2 p_K * d
//   M2M V: D * p_V * 2 p_V * d
//   M2L K: 2 * eta * p_K^2 * d * D
//   M2L V: 2 * eta * p_V^2 * d * D
//   L2L K: D * p_K^2 * d
//   L2L V: D * p_V^2 * d
//   Decode (softmax(Q L_K^T) L_V): s0 * (p_K + p_V) * d
//   Near attention: (2 eta + 1) * s0 * 2 d  (K and V)
static double hmta_separate_flops(int T, int d, int s0, int p_K, int p_V, int eta) {
    int Nc = T / s0;
    int D = 0; { int x = Nc; while (x > 1) { x >>= 1; ++D; } }
    double per_leaf =
        (double)p_K * (double)s0 * (double)d
      + (double)p_V * (double)s0 * (double)d
      + (double)D * (double)p_K * 2.0 * (double)p_K * (double)d
      + (double)D * (double)p_V * 2.0 * (double)p_V * (double)d
      + 2.0 * (double)eta * (double)p_K * (double)p_K * (double)d * (double)D
      + 2.0 * (double)eta * (double)p_V * (double)p_V * (double)d * (double)D
      + (double)D * (double)p_K * (double)p_K * (double)d
      + (double)D * (double)p_V * (double)p_V * (double)d
      + (double)s0 * ((double)p_K + (double)p_V) * (double)d
      + (double)(2 * eta + 1) * (double)s0 * 2.0 * (double)d;
    return per_leaf * (double)Nc;
}

static double rl2(const float* A, const float* B, int n) {
    double nu = 0, de = 0;
    for (int i = 0; i < n; ++i) {
        double d = (double)A[i] - (double)B[i];
        nu += d*d; de += (double)A[i]*A[i];
    }
    return de < 1e-30 ? 0 : std::sqrt(nu / de);
}

// ---- Driver ----
int main(int argc, char** argv) {
    int seed = (argc > 1) ? std::atoi(argv[1]) : 42;
    srand((unsigned)seed);

    int T = 512, d = 64, s0 = 64, eta = 2;
    int seeds[5] = { 7, 13, 42, 100, 1729 };
    std::vector<float> Q(T*d), K(T*d), V(T*d), Ys(T*d), Yh(T*d);

    std::printf("HMTA v4 — paradigm #261 iter 40 (F1: separate p_K, p_V)\n");
    std::printf("=======================================================\n");
    std::printf("Config: T=%d d=%d s0=%d eta=%d, basis = K-only + V-only single-level\n\n",
                T, d, s0, eta);

    int p_K_set[2] = { 4, 8 };
    int p_V_set[5] = { 8, 16, 32, 48, 64 };

    // (A) Gaussian synthetic at r=8, noise=0.10.
    std::printf("(A) Gaussian synthetic (r=8, noise=0.10)\n");
    std::printf("    p_K \\ p_V |  pV=8  | pV=16  | pV=32  | pV=48  | pV=64\n");
    std::printf("    ----------+--------+--------+--------+--------+--------\n");
    for (int pki = 0; pki < 2; ++pki) {
        int p_K = p_K_set[pki];
        std::printf("    p_K=%d    ", p_K);
        for (int pvi = 0; pvi < 5; ++pvi) {
            int p_V = p_V_set[pvi];
            std::vector<double> l2s;
            for (int si = 0; si < 5; ++si) {
                srand((unsigned)seeds[si]);
                make_gaussian(T, d, 8, 0.10, &Q[0], &K[0], &V[0]);
                flat_sdpa(&Q[0], &K[0], &V[0], T, d, &Ys[0]);
                hmta_separate(&Q[0], &K[0], &V[0], T, d, s0, p_K, p_V, eta, &Yh[0]);
                l2s.push_back(rl2(&Ys[0], &Yh[0], T*d));
            }
            std::sort(l2s.begin(), l2s.end());
            std::printf("| %.4f ", l2s[2]);
        }
        std::printf("\n");
    }

    // (B) Sink+local synthetic.
    std::printf("\n(B) Sink+local synthetic (4 sinks, sink_scale=10x bg, bg=1)\n");
    std::printf("    p_K \\ p_V |  pV=8  | pV=16  | pV=32  | pV=48  | pV=64\n");
    std::printf("    ----------+--------+--------+--------+--------+--------\n");
    for (int pki = 0; pki < 2; ++pki) {
        int p_K = p_K_set[pki];
        std::printf("    p_K=%d    ", p_K);
        for (int pvi = 0; pvi < 5; ++pvi) {
            int p_V = p_V_set[pvi];
            std::vector<double> l2s;
            for (int si = 0; si < 5; ++si) {
                srand((unsigned)seeds[si]);
                make_sink_local(T, d, 4, 10.0, 1.0, &Q[0], &K[0], &V[0]);
                flat_sdpa(&Q[0], &K[0], &V[0], T, d, &Ys[0]);
                hmta_separate(&Q[0], &K[0], &V[0], T, d, s0, p_K, p_V, eta, &Yh[0]);
                l2s.push_back(rl2(&Ys[0], &Yh[0], T*d));
            }
            std::sort(l2s.begin(), l2s.end());
            std::printf("| %.4f ", l2s[2]);
        }
        std::printf("\n");
    }

    // (C) Production FLOP ratio at T=16384 for each (p_K, p_V).
    std::printf("\n(C) Production FLOP ratio at T=16384, d=64, s0=64, eta=2\n");
    std::printf("    p_K \\ p_V |  pV=8  | pV=16  | pV=32  | pV=48  | pV=64\n");
    std::printf("    ----------+--------+--------+--------+--------+--------\n");
    double f_sdpa = 2.0 * 16384.0 * 16384.0 * 64.0;
    for (int pki = 0; pki < 2; ++pki) {
        int p_K = p_K_set[pki];
        std::printf("    p_K=%d    ", p_K);
        for (int pvi = 0; pvi < 5; ++pvi) {
            int p_V = p_V_set[pvi];
            double f_h = hmta_separate_flops(16384, 64, 64, p_K, p_V, eta);
            std::printf("| %5.1fx ", f_sdpa / f_h);
        }
        std::printf("\n");
    }

    // (D) Pareto verdict.
    std::printf("\n(D) Iter-40 C0 verdict (joint Gaussian+sink+local at threshold)\n");
    std::printf("    Target: forward L2 <= 0.15 on BOTH Gaussian AND sink+local\n");
    std::printf("            AND FLOP ratio >= 10x at T=16384.\n\n");
    std::printf("    p_K | p_V | L2_gauss | L2_sink | FLOP ratio | PASS\n");
    std::printf("    ----+-----+----------+---------+------------+------\n");

    int pass_count = 0;
    for (int pki = 0; pki < 2; ++pki) {
        for (int pvi = 0; pvi < 5; ++pvi) {
            int p_K = p_K_set[pki];
            int p_V = p_V_set[pvi];
            std::vector<double> g_l2s, s_l2s;
            for (int si = 0; si < 5; ++si) {
                srand((unsigned)seeds[si]);
                make_gaussian(T, d, 8, 0.10, &Q[0], &K[0], &V[0]);
                flat_sdpa(&Q[0], &K[0], &V[0], T, d, &Ys[0]);
                hmta_separate(&Q[0], &K[0], &V[0], T, d, s0, p_K, p_V, eta, &Yh[0]);
                g_l2s.push_back(rl2(&Ys[0], &Yh[0], T*d));
                srand((unsigned)seeds[si]);
                make_sink_local(T, d, 4, 10.0, 1.0, &Q[0], &K[0], &V[0]);
                flat_sdpa(&Q[0], &K[0], &V[0], T, d, &Ys[0]);
                hmta_separate(&Q[0], &K[0], &V[0], T, d, s0, p_K, p_V, eta, &Yh[0]);
                s_l2s.push_back(rl2(&Ys[0], &Yh[0], T*d));
            }
            std::sort(g_l2s.begin(), g_l2s.end());
            std::sort(s_l2s.begin(), s_l2s.end());
            double f_h = hmta_separate_flops(16384, 64, 64, p_K, p_V, eta);
            double ratio = f_sdpa / f_h;
            bool pass = (g_l2s[2] <= 0.15) && (s_l2s[2] <= 0.15) && (ratio >= 10.0);
            std::printf("    %d   |  %d  |  %.4f  | %.4f  |  %5.1fx   | %s\n",
                        p_K, p_V, g_l2s[2], s_l2s[2], ratio,
                        pass ? "PASS" : "fail");
            if (pass) ++pass_count;
        }
    }
    std::printf("\n  Config(s) passing all 3 criteria: %d\n", pass_count);
    return 0;
}
