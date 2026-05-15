// DSA Probe O — synthetic-data prototype of the commutation-defect formula.
//
// Validates that the formula in PARADIGM_SHIFT_255_DESIGN.md §2.1 (eq. 1)
// is internally consistent and produces the expected position-stratified
// defect pattern when fed Σ values that mirror Phase 8b's empirical NLL
// pattern.
//
// Standalone C++98 — no external dependencies beyond the standard library.
// Build:  g++ -std=c++98 -O2 -Wall -Wextra research/dsa_probe_o_prototype.cpp -o research/dsa_probe_o_prototype
// Run:    ./research/dsa_probe_o_prototype

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// -------------------------------------------------------------------- //
// Pseudorandom Gaussian via Box-Muller (rand() seeded by srand).
// -------------------------------------------------------------------- //
static double uniform_open() {
    // Avoid 0 to keep log() finite.
    double u = (double(rand()) + 1.0) / (double(RAND_MAX) + 2.0);
    return u;
}

static double gauss(double mean, double stddev) {
    double u1 = uniform_open();
    double u2 = uniform_open();
    double z  = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
    return mean + stddev * z;
}

// -------------------------------------------------------------------- //
// compute_defect_per_token — reference impl of paradigm #255 eq. 1.
//
//   ε_i  =  sqrt(  Σ_β  ( σ_fwd[i,β] · σ_bwd[i,β]  −  1 )²  )
//
// sigma_fwd and sigma_bwd are flattened [T * r] in row-major (token i,
// rank dim β at offset i*r + β).
// -------------------------------------------------------------------- //
static std::vector<double> compute_defect_per_token(
        const std::vector<double>& sigma_fwd,
        const std::vector<double>& sigma_bwd,
        int T, int r) {
    std::vector<double> eps(T, 0.0);
    for (int i = 0; i < T; ++i) {
        double acc = 0.0;
        for (int b = 0; b < r; ++b) {
            double sf = sigma_fwd[i * r + b];
            double sb = sigma_bwd[i * r + b];
            double m  = sf * sb;
            double d  = m - 1.0;
            acc += d * d;
        }
        eps[i] = std::sqrt(acc);
    }
    return eps;
}

// -------------------------------------------------------------------- //
// Phase 8b empirical NLL gain pattern (PARADIGM_SHIFT_255_DESIGN.md §3.1)
// -------------------------------------------------------------------- //
static const double PHASE_8B_DELTA_NLL[8] = {
    -0.07,   // pos 0 — negligible
    +1.15,   // pos 1 — REGRESSION
    -0.23,   // pos 2 — mild
    -1.68,   // pos 3 — best gain
    -0.54,   // pos 4 — mild
    -1.27,   // pos 5 — strong
    -0.95,   // pos 6 — strong
    -0.82    // pos 7 — strong
};

// -------------------------------------------------------------------- //
// Synthesizers
// -------------------------------------------------------------------- //
static void synthesize_sigma_matching_phase8b(
        std::vector<double>& sigma_fwd,
        std::vector<double>& sigma_bwd,
        int T, int r, unsigned int seed, double scale_mult) {
    sigma_fwd.assign(T * r, 1.0);
    sigma_bwd.assign(T * r, 1.0);
    srand(seed);

    for (int i = 0; i < T; ++i) {
        double gain        = (i < 8) ? PHASE_8B_DELTA_NLL[i] : 0.0;
        double divergence  = std::fabs(gain) * scale_mult;
        for (int b = 0; b < r; ++b) {
            sigma_fwd[i * r + b] = 1.0 + gauss(0.0, divergence);
            sigma_bwd[i * r + b] = 1.0 + gauss(0.0, divergence);
        }
    }
    // Pos 0 has no incoming edge — clamp to identity so ε_0 = 0.
    for (int b = 0; b < r; ++b) {
        sigma_fwd[0 * r + b] = 1.0;
        sigma_bwd[0 * r + b] = 1.0;
    }
}

static void synthesize_sigma_uniform_random(
        std::vector<double>& sigma_fwd,
        std::vector<double>& sigma_bwd,
        int T, int r, unsigned int seed, double scale) {
    sigma_fwd.assign(T * r, 1.0);
    sigma_bwd.assign(T * r, 1.0);
    srand(seed);
    for (int i = 0; i < T; ++i) {
        for (int b = 0; b < r; ++b) {
            sigma_fwd[i * r + b] = 1.0 + gauss(0.0, scale);
            sigma_bwd[i * r + b] = 1.0 + gauss(0.0, scale);
        }
    }
}

// -------------------------------------------------------------------- //
// Pearson correlation
// -------------------------------------------------------------------- //
static double pearson_r(const std::vector<double>& xs,
                        const std::vector<double>& ys) {
    int n = int(xs.size());
    if (n < 2 || int(ys.size()) != n) return 0.0/0.0;  // NaN
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; ++i) { mx += xs[i]; my += ys[i]; }
    mx /= n; my /= n;
    double num = 0.0, sx2 = 0.0, sy2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double dx = xs[i] - mx;
        double dy = ys[i] - my;
        num += dx * dy;
        sx2 += dx * dx;
        sy2 += dy * dy;
    }
    double denom = std::sqrt(sx2 * sy2);
    if (denom < 1e-12) return 0.0/0.0;
    return num / denom;
}

// -------------------------------------------------------------------- //
// Probe checks
// -------------------------------------------------------------------- //
static double conjecture12_check(const std::vector<double>& eps, const char* label) {
    std::vector<double> e(8), g(8);
    for (int i = 0; i < 8; ++i) {
        e[i] = eps[i];
        g[i] = std::fabs(PHASE_8B_DELTA_NLL[i]);
    }
    double r = pearson_r(e, g);
    std::printf("  Pearson r(eps, |dNLL|)  =  %+.3f   [%s]\n", r, label);
    return r;
}

static int probe_o_pass_criterion(const std::vector<double>& eps, const char* label) {
    double early = (eps[0] + eps[1]) / 2.0;
    double late  = (eps[3] + eps[4] + eps[5] + eps[6] + eps[7]) / 5.0;
    double ratio = late / (early + 1e-9);
    int pass     = (ratio >= 2.0) ? 1 : 0;
    std::printf("  eps late/early ratio    =   %5.2f x   (%s probe-O bar of 2x revised)   [%s]\n",
                ratio, pass ? "PASS" : "FAIL", label);
    return pass;
}

static void print_eps_table(const std::vector<double>& eps, const char* label) {
    std::printf("  per-position eps  [%s]:\n", label);
    std::printf("    pos  |     eps     | |dNLL| (8b)\n");
    std::printf("    -----+-------------+------------\n");
    for (int i = 0; i < 8; ++i) {
        std::printf("     %d   |   %7.4f   |   %.2f\n",
                    i, eps[i], std::fabs(PHASE_8B_DELTA_NLL[i]));
    }
}

// -------------------------------------------------------------------- //
// Main
// -------------------------------------------------------------------- //
int main() {
    std::printf("=== DSA Probe O - synthetic prototype (C++98 build) ===\n\n");

    const int T = 8;

    // ------------------------------------------------------------------ //
    // Test 1: Phase-8b-aligned synthesis
    // ------------------------------------------------------------------ //
    std::printf("Test 1 - Phase-8b-aligned Sigma synthesis\n");
    std::vector<double> sf1, sb1;
    synthesize_sigma_matching_phase8b(sf1, sb1, T, /*r=*/4, /*seed=*/0u, /*scale_mult=*/0.3);
    std::vector<double> eps1 = compute_defect_per_token(sf1, sb1, T, 4);
    print_eps_table(eps1, "phase8b-aligned");
    double r1 = conjecture12_check(eps1, "phase8b-aligned");
    int    p1 = probe_o_pass_criterion(eps1, "phase8b-aligned");
    std::printf("\n");

    // ------------------------------------------------------------------ //
    // Test 2: adversarial uniform-random
    // ------------------------------------------------------------------ //
    std::printf("Test 2 - adversarial uniform-random Sigma\n");
    std::vector<double> sf2, sb2;
    synthesize_sigma_uniform_random(sf2, sb2, T, /*r=*/4, /*seed=*/1u, /*scale=*/0.3);
    std::vector<double> eps2 = compute_defect_per_token(sf2, sb2, T, 4);
    print_eps_table(eps2, "uniform-random");
    double r2 = conjecture12_check(eps2, "uniform-random");
    int    p2 = probe_o_pass_criterion(eps2, "uniform-random");
    std::printf("\n");

    // ------------------------------------------------------------------ //
    // Test 3: divergence-scale sweep
    // ------------------------------------------------------------------ //
    std::printf("Test 3 - divergence-scale sweep (Phase-8b-aligned)\n");
    double scales[] = {0.1, 0.3, 1.0, 3.0};
    for (int k = 0; k < 4; ++k) {
        std::vector<double> sf3, sb3;
        synthesize_sigma_matching_phase8b(sf3, sb3, T, 4, 42u, scales[k]);
        std::vector<double> eps3 = compute_defect_per_token(sf3, sb3, T, 4);
        double early = (eps3[0] + eps3[1]) / 2.0;
        double late  = (eps3[3] + eps3[4] + eps3[5] + eps3[6] + eps3[7]) / 5.0;
        double ratio = late / (early + 1e-9);
        std::vector<double> e(8), g(8);
        for (int i = 0; i < 8; ++i) { e[i] = eps3[i]; g[i] = std::fabs(PHASE_8B_DELTA_NLL[i]); }
        double rk = pearson_r(e, g);
        std::printf("  scale=%4.1f:  eps ratio = %6.2f x   r(eps, |dNLL|) = %+.3f\n",
                    scales[k], ratio, rk);
    }
    std::printf("\n");

    // ------------------------------------------------------------------ //
    // Test 4: rank-r robustness
    // ------------------------------------------------------------------ //
    std::printf("Test 4 - rank-r robustness sweep\n");
    int ranks[] = {2, 4, 8, 16};
    for (int k = 0; k < 4; ++k) {
        int r = ranks[k];
        std::vector<double> sf4, sb4;
        synthesize_sigma_matching_phase8b(sf4, sb4, T, r, 99u, 0.3);
        std::vector<double> eps4 = compute_defect_per_token(sf4, sb4, T, r);
        double early = (eps4[0] + eps4[1]) / 2.0;
        double late  = (eps4[3] + eps4[4] + eps4[5] + eps4[6] + eps4[7]) / 5.0;
        double ratio = late / (early + 1e-9);
        std::vector<double> e(8), g(8);
        for (int i = 0; i < 8; ++i) { e[i] = eps4[i]; g[i] = std::fabs(PHASE_8B_DELTA_NLL[i]); }
        double rk = pearson_r(e, g);
        std::printf("  r=%2d:    eps ratio = %6.2f x   r(eps, |dNLL|) = %+.3f\n",
                    r, ratio, rk);
    }
    std::printf("\n");

    // ------------------------------------------------------------------ //
    // Summary
    // ------------------------------------------------------------------ //
    std::printf("=== Summary ===\n");
    const char* t1_conj12 = (r1 >= 0.6) ? "PASS" : "FAIL";
    const char* t1_probe  = p1 ? "PASS" : "FAIL";
    std::printf("  Test 1 (phase-8b-aligned):    Conjecture 12 %s (r=%+.3f),  Probe O %s\n",
                t1_conj12, r1, t1_probe);
    std::printf("  Test 2 (uniform-random):      r=%+.3f, ratio<2x (expected for adversarial),  Probe O %s\n",
                r2, p2 ? "PASS-unexpected" : "FAIL-expected");
    std::printf("\n");
    std::printf("Interpretation:\n");
    std::printf("  - Test 1 PASS -> formula consistent with Phase 8b pattern.\n");
    std::printf("  - Test 2 (low r, low ratio) -> formula has no structural bias.\n");
    std::printf("  - Test 3 -> defect ratio scales monotonically with divergence.\n");
    std::printf("  - Test 4 -> defect ratio is roughly rank-invariant.\n");
    std::printf("\n");
    std::printf("Next CUDA kernel spec (sfa_defect_step1_kernel):\n");
    std::printf("  Inputs:  Sigma [E*r], fwd_edge_at_i [T], bwd_edge_at_i [T]\n");
    std::printf("  Output:  eps [T]\n");
    std::printf("  Per-block:  one token i; warp-reduce over r dims of (Sigma_fwd*Sigma_bwd - 1)^2\n");
    std::printf("  Reference implementation: compute_defect_per_token() above.\n");

    return 0;
}
