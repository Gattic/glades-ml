// ORION Gate-0 — slow-rank hypothesis probe.
//
// Reads a sequence of CHRF v=4 (BF16-weights) checkpoints, computes the
// trajectory matrix M = [Δθ_1 | Δθ_2 | ... | Δθ_N] where Δθ_k = θ_k - θ_anchor,
// then computes the singular spectrum of M via the small Gram matrix M^T M
// (eigendecomposed with classical Jacobi rotations).  Reports cumulative
// energy at r=1, 2, 4, 8.  Decision rule: if r=4 captures ≥0.95 of the
// energy, the slow-rank hypothesis holds at this scale and ORION is viable.
//
// To avoid loading 30+ GB of weights at full d=870M, the probe uses
// reproducible random subsampling: SAMPLE_COUNT indices drawn uniformly
// at random with fixed seed.  The eigenspectrum of a random projection
// of the trajectory preserves the rank-energy ratio statistically.
//
// Build (from glades-ml repo root):
//   g++ -O2 -std=c++14 -o research/orion_slow_rank_probe research/orion_slow_rank_probe.cpp
//
// Usage:
//   research/orion_slow_rank_probe path/to/probe.step500 path/to/probe.step1000 ...

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// Subsample size.  10M float doubles = 80 MB per checkpoint; total
// memory for N=10 checkpoints ~ 800 MB.
static const size_t SAMPLE_COUNT = 10000000;

// xorshift32 — reproducible random indices.
static uint32_t xorshift32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

// BF16 → FP32 (bit-pack, no rounding).
static inline float bf16_to_f32(uint16_t b) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}

struct CHRFHeader {
    int32_t T, m, L, nH, dH, V;
    int32_t step;
    int32_t slcLast;
    int32_t runtimeT, runtimeL;
    float   runtimeAlpha;
    uint32_t flags;
    int32_t faceStepCount;
    uint32_t version;
    // Derived:
    size_t weights_count;  // total FP32 param count in the weight blob
    size_t weight_bytes;   // size in bytes (×2 for BF16, ×4 for FP32)
    bool   bf16_on_disk;
};

static bool read_chrf_header(std::FILE* fp, CHRFHeader& h) {
    char magic[4];
    if (std::fread(magic, 1, 4, fp) != 4 || std::memcmp(magic, "CHRF", 4) != 0) {
        std::fprintf(stderr, "not CHRF\n");
        return false;
    }
    if (std::fread(&h.version, 4, 1, fp) != 1) return false;
    int32_t hdr[6];
    if (std::fread(hdr, 4, 6, fp) != 6) return false;
    h.T = hdr[0]; h.m = hdr[1]; h.L = hdr[2];
    h.nH = hdr[3]; h.dH = hdr[4]; h.V = hdr[5];
    int32_t stepInfo[2];
    if (std::fread(stepInfo, 4, 2, fp) != 2) return false;
    h.step = stepInfo[0]; h.slcLast = stepInfo[1];
    int32_t runtime[2];
    if (std::fread(runtime, 4, 2, fp) != 2) return false;
    h.runtimeT = runtime[0]; h.runtimeL = runtime[1];
    if (std::fread(&h.runtimeAlpha, 4, 1, fp) != 1) return false;
    if (std::fread(&h.flags, 4, 1, fp) != 1) return false;
    if (std::fread(&h.faceStepCount, 4, 1, fp) != 1) return false;

    h.bf16_on_disk = (h.flags & 64u) != 0u;
    const int dModel = h.nH * h.dH;
    // Per layer: Wq + Wk + Wv (3 × m·dModel) + Wo (dModel·m) + γ + β (2 × m)
    // Plus optional gamma_p, beta_p if flag bit 8.
    const size_t per_layer_main = 3ull * h.m * dModel + (size_t)dModel * h.m + 2ull * h.m;
    const size_t per_layer_gp   = (h.flags & 8u) ? 2ull * h.m : 0ull;
    const size_t per_layer = per_layer_main + per_layer_gp;
    h.weights_count = (size_t)h.V * h.m + (size_t)h.L * per_layer;
    h.weight_bytes  = h.weights_count * (h.bf16_on_disk ? 2 : 4);
    return true;
}

// Read sampled indices' worth of weight data into out[] (already sized SAMPLE_COUNT).
// indices must be sorted ascending for streaming.
static bool read_sampled_weights(std::FILE* fp, const CHRFHeader& h,
                                  const std::vector<size_t>& indices,
                                  std::vector<float>& out) {
    out.resize(indices.size());
    if (!h.bf16_on_disk) {
        std::fprintf(stderr, "probe expects v=4 (bf16) checkpoints; got FP32\n");
        return false;
    }
    // Stream through file, capturing only the bytes at sampled indices.
    const size_t n = h.weights_count;
    const size_t k = indices.size();
    size_t idx_i = 0;
    const size_t CHUNK = 1u << 20;  // 1M elements at a time
    std::vector<uint16_t> chunk(CHUNK);
    size_t pos = 0;
    while (pos < n && idx_i < k) {
        const size_t take = std::min(CHUNK, n - pos);
        if (std::fread(&chunk[0], 2, take, fp) != take) {
            std::fprintf(stderr, "fread short at pos=%zu\n", pos);
            return false;
        }
        const size_t chunk_end = pos + take;
        while (idx_i < k && indices[idx_i] < chunk_end) {
            out[idx_i] = bf16_to_f32(chunk[indices[idx_i] - pos]);
            ++idx_i;
        }
        pos = chunk_end;
    }
    return idx_i == k;
}

// Symmetric eigendecomp via classical Jacobi rotations.
// A is N×N symmetric (row-major), modified in-place to diagonal.
// Returns eigenvalues sorted descending.
static std::vector<double> jacobi_eigen(std::vector<double>& A, int N, int max_sweeps=100) {
    auto idx = [N](int i, int j) { return i * N + j; };
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off_diag = 0.0;
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j)
                off_diag += A[idx(i,j)] * A[idx(i,j)];
        if (off_diag < 1e-18 * (double)N) break;
        for (int p = 0; p < N - 1; ++p) {
            for (int q = p + 1; q < N; ++q) {
                const double app = A[idx(p,p)];
                const double aqq = A[idx(q,q)];
                const double apq = A[idx(p,q)];
                if (std::fabs(apq) < 1e-15) continue;
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (std::fabs(theta) > 1e15) t = 1.0 / (2.0 * theta);
                else t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta*theta + 1.0));
                double c = 1.0 / std::sqrt(1.0 + t*t);
                double s = t * c;
                A[idx(p,p)] = app - t * apq;
                A[idx(q,q)] = aqq + t * apq;
                A[idx(p,q)] = A[idx(q,p)] = 0.0;
                for (int i = 0; i < N; ++i) {
                    if (i == p || i == q) continue;
                    const double aip = A[idx(i,p)];
                    const double aiq = A[idx(i,q)];
                    A[idx(i,p)] = A[idx(p,i)] = c * aip - s * aiq;
                    A[idx(i,q)] = A[idx(q,i)] = s * aip + c * aiq;
                }
            }
        }
    }
    std::vector<double> eigs(N);
    for (int i = 0; i < N; ++i) eigs[i] = A[idx(i,i)];
    std::sort(eigs.begin(), eigs.end(), std::greater<double>());
    return eigs;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s checkpoint1 checkpoint2 [checkpoint3 ...]\n", argv[0]);
        std::fprintf(stderr, "  first checkpoint is the anchor; Δθ_k = θ_k - θ_anchor.\n");
        return 1;
    }
    const int N = argc - 1;  // number of checkpoints
    std::vector<std::string> paths;
    for (int i = 1; i < argc; ++i) paths.push_back(argv[i]);

    // Read first checkpoint's header to learn d.
    CHRFHeader h0;
    {
        std::FILE* fp = std::fopen(paths[0].c_str(), "rb");
        if (!fp) { std::fprintf(stderr, "open %s\n", paths[0].c_str()); return 1; }
        if (!read_chrf_header(fp, h0)) { std::fclose(fp); return 1; }
        std::fclose(fp);
    }
    const size_t d = h0.weights_count;
    std::printf("== ORION slow-rank probe ==\n");
    std::printf("checkpoints: %d\n", N);
    std::printf("d (total weights) = %zu  (V=%d m=%d L=%d dModel=%d)\n",
                d, h0.V, h0.m, h0.L, h0.nH * h0.dH);
    std::printf("subsample = %zu indices (xorshift32 seed=1337)\n", SAMPLE_COUNT);

    // Generate reproducible sorted random indices.
    std::vector<size_t> indices(SAMPLE_COUNT);
    uint32_t rng = 1337u;
    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        uint32_t lo = xorshift32(&rng);
        uint32_t hi = xorshift32(&rng);
        uint64_t r = ((uint64_t)hi << 32) | lo;
        indices[i] = (size_t)(r % d);
    }
    std::sort(indices.begin(), indices.end());

    // Load all checkpoints' subsamples into RAM.  Memory: N * SAMPLE_COUNT * 4 bytes.
    std::vector< std::vector<float> > theta(N);
    for (int k = 0; k < N; ++k) {
        std::FILE* fp = std::fopen(paths[k].c_str(), "rb");
        if (!fp) { std::fprintf(stderr, "open %s\n", paths[k].c_str()); return 1; }
        CHRFHeader hk;
        if (!read_chrf_header(fp, hk)) { std::fclose(fp); return 1; }
        if (hk.weights_count != d) {
            std::fprintf(stderr, "shape mismatch at %s: %zu vs %zu\n",
                        paths[k].c_str(), hk.weights_count, d);
            std::fclose(fp); return 1;
        }
        if (!read_sampled_weights(fp, hk, indices, theta[k])) {
            std::fclose(fp); return 1;
        }
        std::fclose(fp);
        std::printf("[%d/%d] %s  step=%d  ‖θ‖=%.4e  loaded %zu samples\n",
                    k + 1, N, paths[k].c_str(), hk.step,
                    [&]() { double s=0; for (float v : theta[k]) s += (double)v*v; return std::sqrt(s); }(),
                    theta[k].size());
    }

    // Build trajectory matrix Δ = [Δθ_1 | Δθ_2 | ... | Δθ_{N-1}] with Δθ_k = θ_{k} - θ_0.
    // Compute Gram = Δ^T Δ as (N-1)×(N-1) symmetric matrix.
    const int M = N - 1;
    std::vector<double> Gram((size_t)M * M, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int j = i; j < M; ++j) {
            double dot = 0.0;
            const std::vector<float>& a = theta[i + 1];
            const std::vector<float>& b = theta[j + 1];
            const std::vector<float>& base = theta[0];
            const size_t k = a.size();
            for (size_t t = 0; t < k; ++t) {
                const double da = (double)a[t] - (double)base[t];
                const double db = (double)b[t] - (double)base[t];
                dot += da * db;
            }
            Gram[(size_t)i * M + j] = Gram[(size_t)j * M + i] = dot;
        }
    }

    std::printf("\nGram matrix Δ^T Δ (%dx%d):\n", M, M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) std::printf(" %+9.3e", Gram[(size_t)i * M + j]);
        std::printf("\n");
    }

    // Eigendecompose; singular values of Δ are sqrt(eigenvalues of Gram).
    std::vector<double> eigs = jacobi_eigen(Gram, M);
    std::vector<double> sv(M);
    for (int i = 0; i < M; ++i) sv[i] = (eigs[i] > 0.0 ? std::sqrt(eigs[i]) : 0.0);

    double total = 0.0;
    for (int i = 0; i < M; ++i) total += eigs[i];
    std::printf("\nSingular spectrum (sorted desc):\n");
    for (int i = 0; i < M; ++i) {
        std::printf("  σ[%d] = %.4e   energy = %.6f   cumulative = %.6f\n",
                    i, sv[i], eigs[i] / total,
                    [&]() { double c=0; for (int j=0; j<=i; ++j) c += eigs[j]; return c / total; }());
    }

    std::printf("\nCumulative energy by rank r:\n");
    const int probe_ranks[] = {1, 2, 4, 8};
    for (int idx = 0; idx < 4; ++idx) {
        const int r = probe_ranks[idx];
        if (r > M) break;
        double sum = 0.0;
        for (int i = 0; i < r; ++i) sum += eigs[i];
        std::printf("  r=%d : %.6f  %s\n", r, sum / total,
                    (sum / total >= 0.95) ? "(≥0.95 ✓ slow-rank holds)"
                    : (sum / total >= 0.80) ? "(0.80-0.95: marginal)"
                    : "(<0.80: slow-rank FAILS)");
    }

    std::printf("\nVerdict: ");
    double r4 = 0.0;
    for (int i = 0; i < std::min(4, M); ++i) r4 += eigs[i];
    r4 /= total;
    if (r4 >= 0.95)      std::printf("PASS — ORION viable at r=4 (energy %.4f)\n", r4);
    else if (r4 >= 0.80) std::printf("MARGINAL — try larger r (energy %.4f at r=4)\n", r4);
    else                 std::printf("FAIL — slow-rank hypothesis does not hold at this scale (energy %.4f at r=4)\n", r4);
    return 0;
}
