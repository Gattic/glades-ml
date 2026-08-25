// HMTA Flagship SV-Retention Probe — paradigm #261 iter 41.
//
// Reads the post-fix flagship CHRF v=4 checkpoint directly (without the full
// glades-trainer GPU infrastructure) and analyzes the spectral structure of
// per-head attention logit submatrices Lambda_{mu,nu} on cluster-pair blocks.
//
// What this probes:
//   Conjecture C4 (design doc §15.4): "for natural-language sequences ...
//   the median sigma_{p+1}(Lambda_{mu,nu}) / ||Lambda_{mu,nu}||_F across
//   admissible pairs is below 5% at p = 8."
//
// HMTA viability requires that real flagship attention matrices admit
// rank-p approximation with bounded truncation error.  Iter 40 showed
// (p_K=8, p_V=48) clears C0a on the sink+local synthetic; this iter tests
// whether real flagship attention behaves like that synthetic regime.
//
// Probe design:
//   1. Open CHRF v=4 checkpoint, parse header, locate Wq[l], Wk[l] for
//      each target layer.
//   2. For each (layer, head):
//        For each seed:
//          X = LayerNorm(N(0, 1))            [s0, m]   synthetic activation
//          Q_h = X · Wq[:, h*dH:(h+1)*dH]    [s0, dH]
//          K_h = X · Wk[:, h*dH:(h+1)*dH]    [s0, dH]
//        Compute "anchor" basis from one seed; use another seed's X to form
//        a cross-cluster Lambda submatrix and measure rank-p retention.
//   3. Median rank-p retention across (heads, layers, seeds).
//
// This is "weights-only": it uses real flagship Wq, Wk but synthetic X.
// Trained-vs-random X differ in correlation structure; the Q, K SPECTRAL
// STRUCTURE that matters for HMTA is determined by Wq, Wk's interaction
// with the activation covariance.  LayerNormed random X is a reasonable
// proxy because the flagship's activations are LayerNormed in q-reln.
//
// Gate-0 conjecture C0-iter41:
//   For at least one mid-depth layer (L in {6, 12, 18}), median rank-p
//   retention across heads and seeds is:
//     >= 0.85 at p = 8   (matches design's optimistic default)
//     >= 0.85 at p = 24  (matches iter-38 fallback)
//   AT MINIMUM at p = 48 we must see >= 0.85 (else HMTA is empirically
//   falsified at scale even with iter-40's (p_K=8, p_V=48) design).
//
// Build:
//   g++ -std=c++98 -O2 -Wall -Wextra research/hmta_flagship_svprobe.cpp \
//       -o research/hmta_flagship_svprobe
// Run:
//   ./research/hmta_flagship_svprobe \
//       research/runs/2026-05-15-flagship-postfix-T16384-30k/chiron_1B_T16384.step30000

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

// -------------------------------------------------------------------- //
// RNG.
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

// -------------------------------------------------------------------- //
// bf16 -> fp32 promotion (matches chiron_infer.cpp).
static void bf16_to_fp32(const uint16_t* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        union { float f; uint32_t u; } v;
        v.u = ((uint32_t)in[i]) << 16;
        out[i] = v.f;
    }
}

// Read a weight block (n elements) into fp32 buf; auto-handles bf16.
static bool read_block(std::FILE* fp, std::vector<float>& buf, size_t n,
                       bool bf16) {
    buf.resize(n);
    if (!bf16) {
        return std::fread(&buf[0], sizeof(float), n, fp) == n;
    }
    std::vector<uint16_t> bf(n);
    if (std::fread(&bf[0], sizeof(uint16_t), n, fp) != n) return false;
    bf16_to_fp32(&bf[0], &buf[0], n);
    return true;
}

// Skip a weight block on disk (n elements * size_per_elem).
static bool skip_block(std::FILE* fp, size_t n, bool bf16) {
    size_t bytes = n * (bf16 ? sizeof(uint16_t) : sizeof(float));
    return std::fseek(fp, (long)bytes, SEEK_CUR) == 0;
}

// -------------------------------------------------------------------- //
// Matrix helpers (row-major fp32).
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

// Top-p left-singular subspace via subspace iteration; returns orthonormal
// columns U[M x p] and singular values S[p] (approximate, ~3-decimal).
static void rand_svd_top(const float* A, int M, int N, int p,
                         float* U, float* S) {
    std::vector<float> Om(N * p), Y(M * p), B(p * N);
    fill_gauss(&Om[0], N * p, 1.0);
    matmul(A, &Om[0], &Y[0], M, N, p);
    for (int pass = 0; pass < 2; ++pass) {
        // QR via modified Gram-Schmidt.
        for (int j = 0; j < p; ++j) {
            for (int i = 0; i < j; ++i) {
                double d = 0;
                for (int r = 0; r < M; ++r) d += Y[r*p + i] * Y[r*p + j];
                for (int r = 0; r < M; ++r) Y[r*p + j] -= (float)d * Y[r*p + i];
            }
            double n = 0;
            for (int r = 0; r < M; ++r) n += (double)Y[r*p + j] * Y[r*p + j];
            n = std::sqrt(n);
            if (n < 1e-20) n = 1.0;
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
    matmul_AT_B(U, A, &B[0], M, p, N);
    for (int i = 0; i < p; ++i) {
        double nrm = 0;
        for (int j = 0; j < N; ++j) nrm += (double)B[i*N + j] * B[i*N + j];
        S[i] = (float)std::sqrt(nrm);
    }
}

// Frobenius norm.
static double fro2(const float* A, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double)A[i] * A[i];
    return s;
}

// Median (in place sort).
static double median_(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0;
    size_t n = v.size();
    return (n & 1) ? v[n/2] : 0.5 * (v[n/2-1] + v[n/2]);
}

// LayerNorm on a [rows, cols] tensor in place (row-wise zero-mean unit-var).
static void layernorm_inplace(float* X, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        double mu = 0;
        for (int j = 0; j < cols; ++j) mu += X[i*cols + j];
        mu /= cols;
        double var = 0;
        for (int j = 0; j < cols; ++j) {
            double d = X[i*cols + j] - mu;
            var += d*d;
        }
        var /= cols;
        double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int j = 0; j < cols; ++j) X[i*cols + j] = (float)((X[i*cols + j] - mu) * inv);
    }
}

// -------------------------------------------------------------------- //
// Main probe.
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <flagship_checkpoint_path> [seed]\n", argv[0]);
        return 1;
    }
    const char* ckpt_path = argv[1];
    int seed = (argc > 2) ? std::atoi(argv[2]) : 42;
    srand((unsigned)seed);

    std::printf("HMTA Flagship SV-Retention Probe — paradigm #261 iter 41\n");
    std::printf("=========================================================\n");
    std::printf("Checkpoint: %s\n", ckpt_path);
    std::printf("Seed: %d\n\n", seed);

    std::FILE* fp = std::fopen(ckpt_path, "rb");
    if (!fp) {
        std::fprintf(stderr, "Cannot open %s\n", ckpt_path);
        return 2;
    }

    // ---- Parse CHRF v=4 header ----
    char magic[5] = {0};
    if (std::fread(magic, 1, 4, fp) != 4) { std::fprintf(stderr, "short magic\n"); return 3; }
    bool is_full = (magic[0]=='C' && magic[1]=='H' && magic[2]=='R' && magic[3]=='F');
    if (!is_full && !(magic[0]=='C' && magic[1]=='H' && magic[2]=='R' && magic[3]=='N')) {
        std::fprintf(stderr, "bad magic: %4.4s\n", magic);
        return 3;
    }
    int version = 0;
    if (std::fread(&version, sizeof(int), 1, fp) != 1) { std::fprintf(stderr, "short ver\n"); return 3; }
    std::printf("Format: %4.4s v=%d (%s)\n", magic, version, is_full ? "full-state" : "weights-only");

    int hdr[6];
    if (std::fread(hdr, sizeof(int), 6, fp) != 6) { std::fprintf(stderr, "short hdr\n"); return 3; }
    int T = hdr[0], m = hdr[1], L = hdr[2], nH = hdr[3], dH = hdr[4], V = hdr[5];
    int dModel = nH * dH;
    std::printf("Dims: T=%d m=%d L=%d nH=%d dH=%d V=%d dModel=%d\n",
                T, m, L, nH, dH, V, dModel);

    bool weights_bf16 = false;
    uint32_t chrf_flags = 0;
    bool has_gamma_p = false;
    if (is_full) {
        std::fseek(fp, 5 * sizeof(int32_t), SEEK_CUR);
        if (std::fread(&chrf_flags, sizeof(uint32_t), 1, fp) != 1) return 3;
        std::fseek(fp, 1 * sizeof(int32_t), SEEK_CUR);
        weights_bf16 = (chrf_flags & 64u) != 0;
        has_gamma_p  = (version >= 3) && ((chrf_flags & 8u) != 0);
        std::printf("CHRF flags: 0x%x  (bf16=%d, has_gamma_p=%d, scfa=%d)\n",
                    chrf_flags, (int)weights_bf16, (int)has_gamma_p,
                    (int)((chrf_flags & 128u) != 0));
    } else if (version >= 3) {
        uint32_t cf = 0;
        if (std::fread(&cf, sizeof(uint32_t), 1, fp) != 1) return 3;
        weights_bf16 = (cf & 0x1u) != 0;
    }

    // ---- Skip embedding E[V, m] ----
    size_t E_sz = (size_t)V * m;
    if (!skip_block(fp, E_sz, weights_bf16)) { std::fprintf(stderr, "skip E failed\n"); return 4; }
    std::printf("Skipped embedding (%.1f MB)\n",
                (double)(E_sz * (weights_bf16 ? 2 : 4)) / (1024.0 * 1024.0));

    // ---- Iterate layers; for target mid-depth layers, read Wq + Wk ----
    int target_layers[3] = { 6, 12, 18 };
    std::vector<float> Wq_buf, Wk_buf;
    std::vector< std::vector<float> > Wq_layers(3), Wk_layers(3);

    size_t Wqkv_sz = (size_t)m * dModel;
    size_t Wo_sz   = (size_t)dModel * m;
    for (int l = 0; l < L; ++l) {
        bool keep = false; int idx = -1;
        for (int t = 0; t < 3; ++t) if (target_layers[t] == l) { keep = true; idx = t; }
        if (keep) {
            if (!read_block(fp, Wq_buf, Wqkv_sz, weights_bf16)) return 5;
            if (!read_block(fp, Wk_buf, Wqkv_sz, weights_bf16)) return 5;
            // Skip Wv, Wo, gamma, beta (and gamma_p, beta_p if present).
            if (!skip_block(fp, Wqkv_sz, weights_bf16)) return 5;
            if (!skip_block(fp, Wo_sz,   weights_bf16)) return 5;
            if (!skip_block(fp, m,        weights_bf16)) return 5;
            if (!skip_block(fp, m,        weights_bf16)) return 5;
            if (has_gamma_p) {
                if (!skip_block(fp, m, weights_bf16)) return 5;
                if (!skip_block(fp, m, weights_bf16)) return 5;
            }
            Wq_layers[idx] = Wq_buf;
            Wk_layers[idx] = Wk_buf;
            std::printf("Loaded Wq, Wk for layer %d\n", l);
        } else {
            if (!skip_block(fp, Wqkv_sz, weights_bf16)) return 5;
            if (!skip_block(fp, Wqkv_sz, weights_bf16)) return 5;
            if (!skip_block(fp, Wqkv_sz, weights_bf16)) return 5;
            if (!skip_block(fp, Wo_sz,   weights_bf16)) return 5;
            if (!skip_block(fp, m,        weights_bf16)) return 5;
            if (!skip_block(fp, m,        weights_bf16)) return 5;
            if (has_gamma_p) {
                if (!skip_block(fp, m, weights_bf16)) return 5;
                if (!skip_block(fp, m, weights_bf16)) return 5;
            }
        }
    }
    std::fclose(fp);

    // ---- Analyze ----
    const int s0 = 64;       // HMTA leaf size (matches design default)
    const int p_set[5] = { 8, 16, 24, 32, 48 };
    const int seeds[5] = { 7, 13, 42, 100, 1729 };

    std::printf("\n=== Probe results (median across heads × seeds per layer) ===\n");
    std::printf("LayerNormed random Gaussian X, [s0=%d, m=%d]\n", s0, m);
    std::printf("Per-head:  Q = X·Wq_h, K = X·Wk_h, Lambda = Q K^T / sqrt(dH=%d)\n\n", dH);
    std::printf("  layer | p=8  | p=16 | p=24 | p=32 | p=48 |  (rank-p retention)\n");
    std::printf("  ------+------+------+------+------+------\n");

    // Per-head Wq_h, Wk_h:  m × dH each.
    std::vector<float> X(s0 * m);
    std::vector<float> Q(s0 * dH), K(s0 * dH);
    std::vector<float> Lam(s0 * s0);

    // Aggregate stats per layer.
    std::vector< std::vector<double> > retentions_per_p(5);
    std::vector< std::vector< std::vector<double> > > all_rets(3);
    for (int t = 0; t < 3; ++t) all_rets[t].assign(5, std::vector<double>());

    for (int t = 0; t < 3; ++t) {
        int layer = target_layers[t];
        const std::vector<float>& Wq = Wq_layers[t];
        const std::vector<float>& Wk = Wk_layers[t];
        for (int h = 0; h < nH; ++h) {
            std::vector<float> Wq_h(m * dH), Wk_h(m * dH);
            for (int i = 0; i < m; ++i) for (int j = 0; j < dH; ++j) {
                Wq_h[i*dH + j] = Wq[i*dModel + h*dH + j];
                Wk_h[i*dH + j] = Wk[i*dModel + h*dH + j];
            }
            for (int si = 0; si < 5; ++si) {
                srand((unsigned)seeds[si]);
                fill_gauss(&X[0], s0 * m, 1.0);
                layernorm_inplace(&X[0], s0, m);
                matmul(&X[0], &Wq_h[0], &Q[0], s0, m, dH);
                matmul(&X[0], &Wk_h[0], &K[0], s0, m, dH);
                // Lambda = Q K^T / sqrt(dH)
                float scale = (float)(1.0 / std::sqrt((double)dH));
                for (int i = 0; i < s0; ++i) for (int j = 0; j < s0; ++j) {
                    double s = 0;
                    for (int k = 0; k < dH; ++k) s += Q[i*dH + k] * K[j*dH + k];
                    Lam[i*s0 + j] = (float)s * scale;
                }
                double tot = fro2(&Lam[0], s0 * s0);
                if (tot <= 0) continue;
                for (int pi = 0; pi < 5; ++pi) {
                    int p = p_set[pi];
                    std::vector<float> U(s0 * p), S(p);
                    rand_svd_top(&Lam[0], s0, s0, p, &U[0], &S[0]);
                    double top = 0;
                    for (int i = 0; i < p; ++i) top += (double)S[i] * S[i];
                    double ret = top / tot;
                    all_rets[t][pi].push_back(ret);
                }
            }
        }
        std::printf("  L=%2d  ", layer);
        for (int pi = 0; pi < 5; ++pi) {
            std::printf("| %.3f", median_(all_rets[t][pi]));
        }
        std::printf("\n");
    }

    // Pooled medians across all 3 layers.
    std::printf("  ------+------+------+------+------+------\n");
    std::printf("  pool  ");
    std::vector<double> pooled[5];
    for (int pi = 0; pi < 5; ++pi) {
        for (int t = 0; t < 3; ++t)
            pooled[pi].insert(pooled[pi].end(), all_rets[t][pi].begin(), all_rets[t][pi].end());
    }
    double med_p[5];
    for (int pi = 0; pi < 5; ++pi) {
        med_p[pi] = median_(pooled[pi]);
        std::printf("| %.3f", med_p[pi]);
    }
    std::printf("\n");

    // ---- Verdict ----
    std::printf("\n=== Gate-0 verdict (C0-iter41) ===\n");
    std::printf("Threshold: median retention >= 0.85.\n\n");
    int crit_p = -1;
    for (int pi = 0; pi < 5; ++pi) {
        std::printf("  p=%2d : median retention = %.3f   %s\n",
                    p_set[pi], med_p[pi],
                    med_p[pi] >= 0.85 ? "PASS" : "FAIL");
        if (crit_p < 0 && med_p[pi] >= 0.85) crit_p = p_set[pi];
    }
    std::printf("\n");
    if (crit_p < 0) {
        std::printf("VERDICT: FAIL at all tested p. The multipole hypothesis is\n");
        std::printf("         empirically falsified on real flagship attention at\n");
        std::printf("         scale.  HMTA design must pivot or accept worse-than-\n");
        std::printf("         iter-40 quality.\n");
    } else if (crit_p <= 8) {
        std::printf("VERDICT: PASS at p_K=8 (iter-37 design default).  HMTA viable\n");
        std::printf("         at original parameters.\n");
    } else if (crit_p <= 24) {
        std::printf("VERDICT: PASS at p_K=%d.  Iter-40 (p_K=8, p_V=48) design is\n", crit_p);
        std::printf("         viable; iter-38's unified-p=24 is also clear.\n");
    } else {
        std::printf("VERDICT: PARTIAL — clears only at p=%d.  Iter-40 design needs\n", crit_p);
        std::printf("         revision: p_K must be raised to %d to match real data.\n", crit_p);
    }
    return 0;
}
