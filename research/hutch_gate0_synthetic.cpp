// HUTCH-DIAG Gate-0 probe: synthetic validation.
//
// Generates an SPD Hessian H = A^T A + λI with known structure.
// Computes: (a) true diag(H) analytically, (b) Hutchinson estimate
// v ⊙ Hv with N_probes ∈ {1, 4, 16, 64, 256} probes.
// Measures: correlation ρ and relative-error norm.
//
// The test probes variance properties under three Hessian regimes:
//   Regime 1: STRONGLY diagonal (off-diag magnitude ≪ diag)
//   Regime 2: MIXED (off-diag comparable to diag)
//   Regime 3: DENSE (Gaussian, strong off-diag correlations)
//
// Accept criterion: ρ(N=16) ≥ 0.6 for realistic (mixed) Hessian regime.
//
// Build: g++ -O2 -std=c++17 hutch_gate0.cpp -o /tmp/hutch_gate0

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using V = std::vector<float>;
using M = std::vector<V>;

// Compute C = A * v
V matvec(const M& A, const V& v) {
    const int n = (int)A.size();
    V out((size_t)n, 0.0f);
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int j = 0; j < n; ++j) s += A[i][j] * v[j];
        out[i] = s;
    }
    return out;
}

float correlation(const V& x, const V& y) {
    const int n = (int)x.size();
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int i = 0; i < n; ++i) {
        const double dx = x[i] - mx;
        const double dy = y[i] - my;
        sxx += dx*dx; syy += dy*dy; sxy += dx*dy;
    }
    if (sxx < 1e-20 || syy < 1e-20) return 0.0f;
    return (float)(sxy / std::sqrt(sxx * syy));
}

float rel_error(const V& est, const V& truth) {
    const int n = (int)truth.size();
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        const double e = (double)est[i] - (double)truth[i];
        num += e * e;
        den += (double)truth[i] * (double)truth[i];
    }
    if (den < 1e-20) return 0.0f;
    return (float)std::sqrt(num / den);
}

M build_hessian(int n, float off_diag_scale, std::mt19937& rng) {
    // H = A^T A + λ I.  A is dense Gaussian with std ~ off_diag_scale / sqrt(n).
    std::normal_distribution<float> N01(0.0f, off_diag_scale / std::sqrt((float)n));
    M A(n, V(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = N01(rng);
    M H(n, V(n, 0.0f));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int k = 0; k < n; ++k) s += A[k][i] * A[k][j];
            H[i][j] = s;
            if (i == j) H[i][j] += 1.0f;  // λ = 1
        }
    return H;
}

V true_diag(const M& H) {
    const int n = (int)H.size();
    V d((size_t)n);
    for (int i = 0; i < n; ++i) d[i] = H[i][i];
    return d;
}

// Single-probe Hutchinson: given v ~ Rademacher, returns v ⊙ Hv.
V hutch_probe(const M& H, std::mt19937& rng) {
    const int n = (int)H.size();
    std::bernoulli_distribution coin(0.5);
    V v((size_t)n);
    for (int i = 0; i < n; ++i) v[i] = coin(rng) ? 1.0f : -1.0f;
    V Hv = matvec(H, v);
    V out((size_t)n);
    for (int i = 0; i < n; ++i) out[i] = v[i] * Hv[i];
    return out;
}

int main() {
    std::mt19937 rng(1337);
    const int n = 200;

    std::printf("=== HUTCH-DIAG Gate-0 synthetic probe ===\n");
    std::printf("Hessian H = A^T A + I, n=%d, three regimes of off-diagonal scale\n\n", n);

    const float scales[] = { 0.1f, 0.5f, 2.0f };
    const char* names[]  = { "STRONG-DIAG (scale=0.1)",
                             "MIXED      (scale=0.5)",
                             "DENSE      (scale=2.0)" };
    const int N_probes_list[] = { 1, 4, 16, 64, 256 };

    for (int r = 0; r < 3; ++r) {
        std::printf("--- %s ---\n", names[r]);
        M H = build_hessian(n, scales[r], rng);
        V d_true = true_diag(H);

        // Also measure off-diagonal / diagonal mass ratio for context.
        double diag_mass = 0.0, off_mass = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) diag_mass += H[i][i] * H[i][i];
                else        off_mass  += H[i][j] * H[i][j];
            }
        std::printf("  ||off-diag|| / ||diag|| = %.4f\n",
                    std::sqrt(off_mass / diag_mass));

        // For each N, average N probes and compute correlation to true diagonal.
        std::printf("  %-8s %-10s %-14s\n", "N_probes", "ρ_correl", "rel_err");
        for (int N : N_probes_list) {
            V avg((size_t)n, 0.0f);
            for (int k = 0; k < N; ++k) {
                V pk = hutch_probe(H, rng);
                for (int i = 0; i < n; ++i) avg[i] += pk[i];
            }
            for (int i = 0; i < n; ++i) avg[i] /= (float)N;
            const float rho = correlation(avg, d_true);
            const float err = rel_error(avg, d_true);
            std::printf("  %-8d %-10.4f %-14.4f\n", N, rho, err);
        }
        std::printf("\n");
    }

    std::printf("Accept: ρ(N=16) ≥ 0.6 in MIXED regime → Gate-0 PASSES.\n");
    return 0;
}
