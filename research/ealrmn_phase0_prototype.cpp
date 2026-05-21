// EALRMN-v1 Phase-0 prototype — narrowest test of Claim 2 in EALRMN_DESIGN.md.
//
// Tests whether a closed-form Koopman latent predictor (ẑ_{i+1} = K * s_i)
// trained on stop-gradded latent MSE recovers an HMM's hidden state via a
// linear probe of s_i to h_t faster than the same architecture trained on
// raw next-token NLL ("token" baseline). Single expert, fixed patches,
// full-rank r = m, no spectral memory, no IB contrastive head. The point
// is to isolate the latent-vs-token objective.
//
// Pass criterion: linear-probe accuracy ≥ 0.85 in latent mode by step 500,
// with token mode ≤ 0.70 at the same step. Falsifying outcome: both modes
// converge to similar accuracy at similar speed.
//
// Build:
//   g++ -std=c++98 -O2 -Wall -Wextra research/ealrmn_phase0_prototype.cpp -o research/ealrmn_phase0_prototype
// Run:
//   ./research/ealrmn_phase0_prototype --mode latent     --seed 42 --steps 500
//   ./research/ealrmn_phase0_prototype --mode token      --seed 42 --steps 500
//   ./research/ealrmn_phase0_prototype --mode latent_nce --seed 42 --steps 500   (Phase-0b)
//
// Output (tab-separated, one line per logged step):
//   step  train_loss  held_loss  probe_acc  z_var

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

// =============== Hyperparameters ===============
static const int kHmmS         = 4;     // hidden states
static const int kVocab        = 16;    // vocabulary
static const int kPatchLen     = 4;     // raw tokens per patch
static const int kNumPatches   = 16;    // patches per stream
static const int kStream       = kPatchLen * kNumPatches; // = 64
static const int kEmbDim       = 16;    // token embedding dim
static const int kM            = 32;    // lifted observable dim (r=m, full rank)
static const int kBatch        = 16;    // streams per training step
static const double kNormTarget = 1.0;  // anti-collapse: target ||z||^2 ≥ kNormTarget per sample
static const double kNormWeight = 1.0;  // weight of per-sample norm hinge
static const double kGradClip   = 10.0; // L2 norm clip per parameter group (lenient)
static const double kEmissionOverlap = 0.1; // HMM emission overlap (0=disjoint, 1=uniform)

// InfoNCE hyperparameters (only used in MODE_LATENT_NCE).
// Anchor:  student E_θ(p_i)
// Positive: stopgrad teacher E_θ̄(p_{i+1})  with E_θ̄ = EMA of E_θ, momentum kInfoNceMomentum
// Negatives: in-batch teacher embeddings at the same future patch index across streams.
static const double kInfoNceTau      = 0.2;   // similarity temperature (cosine sim / τ)
static const double kInfoNceWeight   = 1.0;   // β_z — weight of NCE term
static const double kInfoNceMomentum = 0.95;  // EMA momentum for the teacher (faster updates)
static const int    kInfoNceWindow   = 1;     // future-pair window (fixed at 1 in Phase-0b)
// Reconstruction loss on patch tokens from z_i (bootstrapping signal for the encoder).
// Without this, the encoder receives no clear "predict your input" gradient and the
// NCE term cannot escape the random-init plateau.
static const double kReconWeight     = 1.0;

// Phase-0c-A: recurrence-stability fixes.
// kMseWeight reduces the latent-MSE pull in MODE_LATENT_NCE so that K does not fight
// the encoder. kIdentityReg pulls K toward identity each SGD step, keeping the
// Koopman recurrence's spectrum bounded. kKopLrScale uses a smaller effective lr for
// K and B so they track z-dynamics smoothly rather than chasing them.
static const double kMseWeight       = 0.1;
static const double kIdentityReg     = 0.01;
static const double kKopLrScale      = 0.3;

// Phase-0c-B: simplified spectral memory.
// Four memory slots, each m-dim, updated as fixed-decay EMA of z (no gates, no backprop).
// Decay constants span fast-transient to slow-persistent — this is the operator-theoretic
// realization of bounded memory in the design memo §4.4, simplified for Phase-0c-B by
// fixing the eigenvalues rather than learning them. Probe joint (s, M) and M alone to
// localize whether memory carries state info.
static const int    kMemSlots        = 4;
static const double kMemDecays[4]    = { 0.50, 0.80, 0.95, 0.99 };
// Write gate: g_i = sigmoid(W_g · s_i + b_g) — scalar gate per step controls memory
// update intensity. Write cost = mean g over the stream, weight kWriteWeight.
// At gate=0: memory is frozen. At gate=1: full EMA update.
static const double kWriteWeight     = 0.05;

// =============== Random helpers ===============
static double rand_unif() {
    return (double(rand()) + 1.0) / (double(RAND_MAX) + 2.0);
}
static double rand_gauss() {
    double u1 = rand_unif(), u2 = rand_unif();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
}

// =============== HMM data generator ===============
struct HMM {
    std::vector<double> P;   // S x S transitions, row-major
    std::vector<double> E;   // S x V emissions, row-major
    std::vector<double> pi0;
    int S, V;
    HMM() : S(0), V(0) {}

    void init(int s_count, int v_count, double overlap, unsigned seed) {
        S = s_count; V = v_count;
        srand(seed);
        P.assign(S * S, 0.0);
        E.assign(S * V, 0.0);
        pi0.assign(S, 1.0 / S);
        // Transitions with diagonal boost (mean dwell ~4).
        for (int s = 0; s < S; ++s) {
            double sum = 0.0;
            for (int s2 = 0; s2 < S; ++s2) {
                double w = rand_unif();
                if (s == s2) w += 3.0;
                P[s * S + s2] = w;
                sum += w;
            }
            for (int s2 = 0; s2 < S; ++s2) P[s * S + s2] /= sum;
        }
        // Emissions: contiguous preferred range per state + overlap floor.
        int width = V / S;
        for (int s = 0; s < S; ++s) {
            int lo = s * width, hi = lo + width;
            double sum = 0.0;
            for (int v = 0; v < V; ++v) {
                double base = overlap;
                if (v >= lo && v < hi) base += 1.0;
                E[s * V + v] = base;
                sum += base;
            }
            for (int v = 0; v < V; ++v) E[s * V + v] /= sum;
        }
    }

    int sample_cat(const double* row, int n) const {
        double r = rand_unif(), cum = 0.0;
        for (int i = 0; i < n; ++i) { cum += row[i]; if (r < cum) return i; }
        return n - 1;
    }

    void generate(int T, std::vector<int>& x, std::vector<int>& h) const {
        x.resize(T); h.resize(T);
        int s = sample_cat(&pi0[0], S);
        for (int t = 0; t < T; ++t) {
            h[t] = s;
            x[t] = sample_cat(&E[s * V], V);
            s = sample_cat(&P[s * S], S);
        }
    }
};

// =============== Matrix helpers (row-major) ===============
static void matvec(const double* A, const double* x, double* y, int M, int N) {
    for (int m = 0; m < M; ++m) {
        const double* row = A + m * N;
        double acc = 0.0;
        for (int n = 0; n < N; ++n) acc += row[n] * x[n];
        y[m] = acc;
    }
}
static void matvec_T(const double* A, const double* x, double* y, int M, int N) {
    for (int n = 0; n < N; ++n) y[n] = 0.0;
    for (int m = 0; m < M; ++m) {
        const double* row = A + m * N;
        double xm = x[m];
        for (int n = 0; n < N; ++n) y[n] += row[n] * xm;
    }
}
static void outer_add(double* dA, double alpha, const double* x, const double* y, int M, int N) {
    for (int m = 0; m < M; ++m) {
        double ax = alpha * x[m];
        for (int n = 0; n < N; ++n) dA[m * N + n] += ax * y[n];
    }
}
static void axpy(double* y, double alpha, const double* x, int n) {
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

// Clip an array g of length n so that its L2 norm is at most max_norm.
// Returns the original L2 norm (so the caller can log / monitor).
static double clip_l2(double* g, int n, double max_norm) {
    double sq = 0.0;
    for (int i = 0; i < n; ++i) sq += g[i] * g[i];
    double norm = std::sqrt(sq);
    if (norm > max_norm && norm > 0) {
        double scale = max_norm / norm;
        for (int i = 0; i < n; ++i) g[i] *= scale;
    }
    return norm;
}

// =============== Encoder ===============
// z_i = W * mean(Emb[tok_l]) + b   (W is m x d_emb, b is m)
struct Encoder {
    std::vector<double> Emb, W, b;
    std::vector<double> dEmb, dW, db;
    int V_, d_emb_, m_;
    Encoder() : V_(0), d_emb_(0), m_(0) {}

    void init(int V, int d_emb, int m, unsigned seed) {
        V_ = V; d_emb_ = d_emb; m_ = m;
        Emb.assign(V * d_emb, 0.0);
        W.assign(d_emb * m, 0.0);
        b.assign(m, 0.0);
        srand(seed);
        double se = std::sqrt(1.0 / d_emb);
        double sw = std::sqrt(2.0 / (d_emb + m));
        for (int i = 0; i < V * d_emb; ++i) Emb[i] = rand_gauss() * se;
        for (int i = 0; i < m * d_emb; ++i) W[i] = rand_gauss() * sw;
        zero_grad();
    }
    void zero_grad() {
        dEmb.assign(V_ * d_emb_, 0.0);
        dW.assign(m_ * d_emb_, 0.0);
        db.assign(m_, 0.0);
    }
    void forward(const int* tokens, int patch_len, double* z_out, double* e_mean_cache) const {
        for (int j = 0; j < d_emb_; ++j) e_mean_cache[j] = 0.0;
        for (int l = 0; l < patch_len; ++l) {
            int tok = tokens[l];
            for (int j = 0; j < d_emb_; ++j) e_mean_cache[j] += Emb[tok * d_emb_ + j];
        }
        double inv_L = 1.0 / double(patch_len);
        for (int j = 0; j < d_emb_; ++j) e_mean_cache[j] *= inv_L;
        matvec(&W[0], e_mean_cache, z_out, m_, d_emb_);
        for (int i = 0; i < m_; ++i) z_out[i] += b[i];
    }
    void backward(const int* tokens, int patch_len, const double* d_z, const double* e_mean_cache) {
        axpy(&db[0], 1.0, d_z, m_);
        outer_add(&dW[0], 1.0, d_z, e_mean_cache, m_, d_emb_);
        std::vector<double> d_emean(d_emb_, 0.0);
        matvec_T(&W[0], d_z, &d_emean[0], m_, d_emb_);
        double inv_L = 1.0 / double(patch_len);
        for (int l = 0; l < patch_len; ++l) {
            int tok = tokens[l];
            for (int j = 0; j < d_emb_; ++j) dEmb[tok * d_emb_ + j] += d_emean[j] * inv_L;
        }
    }
    void sgd_update(double lr) {
        for (int i = 0; i < V_ * d_emb_; ++i) Emb[i] -= lr * dEmb[i];
        for (int i = 0; i < m_ * d_emb_; ++i) W[i] -= lr * dW[i];
        for (int i = 0; i < m_; ++i) b[i] -= lr * db[i];
    }
};

// =============== Koopman recurrence ===============
struct Koopman {
    std::vector<double> Kop, Bop;
    std::vector<double> dK, dB;
    int m_;
    Koopman() : m_(0) {}

    void init(int m, unsigned seed) {
        m_ = m;
        Kop.assign(m * m, 0.0);
        Bop.assign(m * m, 0.0);
        srand(seed);
        double sk = std::sqrt(1.0 / m);
        for (int i = 0; i < m * m; ++i) Kop[i] = rand_gauss() * sk * 0.1;
        for (int i = 0; i < m; ++i) Kop[i * m + i] += 0.95;     // near-identity init (Lyapunov margin)
        for (int i = 0; i < m * m; ++i) Bop[i] = rand_gauss() * sk;
        zero_grad();
    }
    void zero_grad() {
        dK.assign(m_ * m_, 0.0);
        dB.assign(m_ * m_, 0.0);
    }
    void step(const double* s_in, const double* z, double* s_out) const {
        std::vector<double> Ks(m_), Bz(m_);
        matvec(&Kop[0], s_in, &Ks[0], m_, m_);
        matvec(&Bop[0], z,    &Bz[0], m_, m_);
        for (int i = 0; i < m_; ++i) s_out[i] = Ks[i] + Bz[i];
    }
    void predict_h1(const double* s_in, double* pred) const {
        matvec(&Kop[0], s_in, pred, m_, m_);
    }
    void backward_step(const double* s_in, const double* z, const double* d_s_out,
                       double* d_s_in, double* d_z) {
        outer_add(&dK[0], 1.0, d_s_out, s_in, m_, m_);
        outer_add(&dB[0], 1.0, d_s_out, z,    m_, m_);
        matvec_T(&Kop[0], d_s_out, d_s_in, m_, m_);
        matvec_T(&Bop[0], d_s_out, d_z,    m_, m_);
    }
    void backward_predict(const double* s_in, const double* d_pred, double* d_s_in_from_pred) {
        outer_add(&dK[0], 1.0, d_pred, s_in, m_, m_);
        matvec_T(&Kop[0], d_pred, d_s_in_from_pred, m_, m_);
    }
    void sgd_update(double lr) {
        for (int i = 0; i < m_ * m_; ++i) Kop[i] -= lr * dK[i];
        for (int i = 0; i < m_ * m_; ++i) Bop[i] -= lr * dB[i];
    }
};

// =============== Token-mode decoder ===============
struct Decoder {
    std::vector<double> Wd, bd;
    std::vector<double> dWd, dbd;
    int V_, m_;
    Decoder() : V_(0), m_(0) {}

    void init(int V, int m, unsigned seed) {
        V_ = V; m_ = m;
        Wd.assign(V * m, 0.0);
        bd.assign(V, 0.0);
        srand(seed);
        double sw = std::sqrt(2.0 / (V + m));
        for (int i = 0; i < V * m; ++i) Wd[i] = rand_gauss() * sw;
        zero_grad();
    }
    void zero_grad() {
        dWd.assign(V_ * m_, 0.0);
        dbd.assign(V_, 0.0);
    }
    void forward(const double* s, double* logits) const {
        matvec(&Wd[0], s, logits, V_, m_);
        for (int v = 0; v < V_; ++v) logits[v] += bd[v];
    }
    // Loss = sum over next_tokens of -log softmax(logits)[tok].
    double loss_and_dlogits(const double* logits, const int* next_tokens, int next_len,
                            double* d_logits) const {
        double max_l = logits[0];
        for (int v = 1; v < V_; ++v) if (logits[v] > max_l) max_l = logits[v];
        std::vector<double> p(V_, 0.0);
        double sum_exp = 0.0;
        for (int v = 0; v < V_; ++v) { p[v] = std::exp(logits[v] - max_l); sum_exp += p[v]; }
        for (int v = 0; v < V_; ++v) p[v] /= sum_exp;
        double loss = 0.0;
        for (int v = 0; v < V_; ++v) d_logits[v] = 0.0;
        for (int l = 0; l < next_len; ++l) {
            int tok = next_tokens[l];
            loss += -std::log(std::max(p[tok], 1e-30));
            for (int v = 0; v < V_; ++v) d_logits[v] += p[v];
            d_logits[tok] -= 1.0;
        }
        return loss;
    }
    void backward(const double* s, const double* d_logits, double* d_s) {
        outer_add(&dWd[0], 1.0, d_logits, s, V_, m_);
        axpy(&dbd[0], 1.0, d_logits, V_);
        matvec_T(&Wd[0], d_logits, d_s, V_, m_);
    }
    void sgd_update(double lr) {
        for (int i = 0; i < V_ * m_; ++i) Wd[i] -= lr * dWd[i];
        for (int v = 0; v < V_; ++v) bd[v] -= lr * dbd[v];
    }
};

// =============== Trainer ===============
struct Trainer {
    HMM hmm;
    Encoder enc;            // student encoder
    Encoder teacher_enc;    // EMA teacher encoder (only used in MODE_LATENT_NCE)
    Koopman kop;
    Decoder dec;
    // Phase-0c-B: write-gate parameters W_g (m,) and b_g (scalar). Forward-only memory
    // means we don't need d_M, but W_g and b_g are learned.
    std::vector<double> Wg, dWg;
    double bg, dbg;
    bool use_memory;  // turned on for MODE_LATENT_NCE only when --use-memory is set
    enum Mode { MODE_LATENT, MODE_TOKEN, MODE_LATENT_NCE };
    Mode mode;
    double lr;
    int m;

    Trainer() : bg(0.0), dbg(0.0), use_memory(false), mode(MODE_LATENT), lr(0.01), m(kM) {}

    void init(int seed) {
        hmm.init(kHmmS, kVocab, kEmissionOverlap, (unsigned)(seed));
        enc.init(kVocab, kEmbDim, kM, (unsigned)(seed + 100));
        // Teacher allocated the same shape; weights copied from student so that
        // at step 0 the teacher equals the student.
        teacher_enc.init(kVocab, kEmbDim, kM, (unsigned)(seed + 100));
        for (int i = 0; i < (int)enc.Emb.size(); ++i) teacher_enc.Emb[i] = enc.Emb[i];
        for (int i = 0; i < (int)enc.W.size();   ++i) teacher_enc.W[i]   = enc.W[i];
        for (int i = 0; i < (int)enc.b.size();   ++i) teacher_enc.b[i]   = enc.b[i];
        kop.init(kM, (unsigned)(seed + 200));
        dec.init(kVocab, kM, (unsigned)(seed + 300));
        // Init memory write-gate parameters.
        Wg.assign(kM, 0.0);
        dWg.assign(kM, 0.0);
        bg = 0.0; dbg = 0.0;
        srand((unsigned)(seed + 400));
        double sg = std::sqrt(1.0 / kM);
        for (int i = 0; i < kM; ++i) Wg[i] = rand_gauss() * sg * 0.1;
        bg = 0.0; // initial gate ~ sigmoid(0) = 0.5, neutral
        srand((unsigned)(seed + 999));
    }

    // EMA update of the teacher encoder: teacher = mom * teacher + (1-mom) * student.
    void teacher_ema_step() {
        const double mom = kInfoNceMomentum;
        for (int i = 0; i < (int)enc.Emb.size(); ++i)
            teacher_enc.Emb[i] = mom * teacher_enc.Emb[i] + (1.0 - mom) * enc.Emb[i];
        for (int i = 0; i < (int)enc.W.size();   ++i)
            teacher_enc.W[i]   = mom * teacher_enc.W[i]   + (1.0 - mom) * enc.W[i];
        for (int i = 0; i < (int)enc.b.size();   ++i)
            teacher_enc.b[i]   = mom * teacher_enc.b[i]   + (1.0 - mom) * enc.b[i];
    }

    // One forward+backward over a batch. Returns avg per-stream loss + var reg.
    double train_step(const std::vector<std::vector<int> >& batch_x) {
        enc.zero_grad(); kop.zero_grad(); dec.zero_grad();
        int B = (int)batch_x.size();

        // Per-stream forward: collect z's and e_means. Also accumulate batch-level
        // z stats for the diagnostic z_var report (NOT used as the regularizer).
        std::vector<std::vector<std::vector<double> > > zS(B);
        std::vector<std::vector<std::vector<double> > > emS(B);
        std::vector<double> z_sum(m, 0.0), z_sq(m, 0.0);
        int z_count = 0;
        for (int b = 0; b < B; ++b) {
            zS[b].assign(kNumPatches, std::vector<double>(m, 0.0));
            emS[b].assign(kNumPatches, std::vector<double>(kEmbDim, 0.0));
            for (int i = 0; i < kNumPatches; ++i) {
                enc.forward(&batch_x[b][i * kPatchLen], kPatchLen,
                            &zS[b][i][0], &emS[b][i][0]);
                for (int j = 0; j < m; ++j) {
                    z_sum[j] += zS[b][i][j];
                    z_sq[j]  += zS[b][i][j] * zS[b][i][j];
                }
                z_count++;
            }
        }
        (void)z_sum; (void)z_sq; (void)z_count; // (used only for optional diagnostics)

        // Anti-collapse: per-sample norm hinge L_norm = sum_{b,i} max(0, kNormTarget - ||z_bi||^2)
        // Stronger than the batch-variance hinge (gradient is not divided by batch size).
        // Applied inside the per-stream loop below.

        // Teacher forward (for InfoNCE positives). Forward-only — no gradient.
        std::vector<std::vector<std::vector<double> > > tzS;
        if (mode == MODE_LATENT_NCE) {
            tzS.assign(B, std::vector<std::vector<double> >());
            std::vector<double> em_buf(kEmbDim);
            for (int b = 0; b < B; ++b) {
                tzS[b].assign(kNumPatches, std::vector<double>(m, 0.0));
                for (int i = 0; i < kNumPatches; ++i) {
                    teacher_enc.forward(&batch_x[b][i * kPatchLen], kPatchLen,
                                        &tzS[b][i][0], &em_buf[0]);
                }
            }
        }

        double total = 0.0;
        double norm_loss_acc = 0.0;
        double nce_loss_acc = 0.0;
        int    nce_count    = 0;
        double recon_loss_acc = 0.0;
        int    recon_count    = 0;
        std::vector<double> s_init(m, 0.0);
        std::vector<double> d_s_in_buf(m), d_z_buf(m), d_pred_buf(m);
        std::vector<double> logits(kVocab), d_logits(kVocab), d_s_dec(m);

        for (int b = 0; b < B; ++b) {
            // Forward through Koopman recurrence.
            std::vector<std::vector<double> > s(kNumPatches, std::vector<double>(m, 0.0));
            kop.step(&s_init[0], &zS[b][0][0], &s[0][0]);
            for (int i = 1; i < kNumPatches; ++i)
                kop.step(&s[i-1][0], &zS[b][i][0], &s[i][0]);

            std::vector<std::vector<double> > d_s(kNumPatches, std::vector<double>(m, 0.0));
            std::vector<std::vector<double> > d_z(kNumPatches, std::vector<double>(m, 0.0));

            double stream_loss = 0.0;

            if (mode == MODE_LATENT || mode == MODE_LATENT_NCE) {
                // Predict ẑ_i = K * s_{i-1} for i = 1..N-1.
                // In MODE_LATENT_NCE the MSE pull is downweighted by kMseWeight so that K
                // does not fight the richer encoder signal from NCE+recon.
                double mse_w = (mode == MODE_LATENT_NCE) ? kMseWeight : 1.0;
                for (int i = 1; i < kNumPatches; ++i) {
                    std::vector<double> pred(m);
                    kop.predict_h1(&s[i-1][0], &pred[0]);
                    double l = 0.0;
                    std::vector<double> d_pred(m);
                    for (int j = 0; j < m; ++j) {
                        double diff = pred[j] - zS[b][i][j]; // stopgrad on zS[b][i] as target
                        l += diff * diff;
                        d_pred[j] = mse_w * 2.0 * diff / m;
                    }
                    stream_loss += mse_w * l / m;
                    // backward predictor: dK += d_pred * s_{i-1}^T, d_s_{i-1} += K^T d_pred
                    kop.backward_predict(&s[i-1][0], &d_pred[0], &d_pred_buf[0]);
                    for (int j = 0; j < m; ++j) d_s[i-1][j] += d_pred_buf[j];
                }
                total += stream_loss / (kNumPatches - 1);
            } else { // MODE_TOKEN
                // Decode tokens of patch i+1 from s[i].
                for (int i = 0; i < kNumPatches - 1; ++i) {
                    dec.forward(&s[i][0], &logits[0]);
                    double l = dec.loss_and_dlogits(&logits[0],
                                                    &batch_x[b][(i+1) * kPatchLen],
                                                    kPatchLen, &d_logits[0]);
                    stream_loss += l / kPatchLen;
                    for (int v = 0; v < kVocab; ++v) d_logits[v] /= kPatchLen;
                    for (int j = 0; j < m; ++j) d_s_dec[j] = 0.0;
                    dec.backward(&s[i][0], &d_logits[0], &d_s_dec[0]);
                    for (int j = 0; j < m; ++j) d_s[i][j] += d_s_dec[j];
                }
                total += stream_loss / (kNumPatches - 1);
            }

            // Backward through recurrence (BPTT).
            for (int i = kNumPatches - 1; i > 0; --i) {
                kop.backward_step(&s[i-1][0], &zS[b][i][0], &d_s[i][0],
                                  &d_s_in_buf[0], &d_z_buf[0]);
                for (int j = 0; j < m; ++j) {
                    d_s[i-1][j] += d_s_in_buf[j];
                    d_z[i][j]   += d_z_buf[j];
                }
            }
            kop.backward_step(&s_init[0], &zS[b][0][0], &d_s[0][0],
                              &d_s_in_buf[0], &d_z_buf[0]);
            for (int j = 0; j < m; ++j) d_z[0][j] += d_z_buf[j];

            // Reconstruction loss on z_i: predict the OWN patch's tokens from z_i via dec.
            // Provides a clean bootstrapping signal for the encoder: z_i must encode the
            // patch's token distribution. Combined with the NCE term, this gives the
            // encoder enough structure for NCE to specialize on hidden-state features.
            if (mode == MODE_LATENT_NCE) {
                std::vector<double> logits_z(kVocab), d_logits_z(kVocab), d_z_recon(m);
                for (int i = 0; i < kNumPatches; ++i) {
                    dec.forward(&zS[b][i][0], &logits_z[0]);
                    double l_recon = dec.loss_and_dlogits(&logits_z[0],
                                                          &batch_x[b][i * kPatchLen],
                                                          kPatchLen, &d_logits_z[0]);
                    recon_loss_acc += l_recon / kPatchLen;
                    recon_count++;
                    for (int v = 0; v < kVocab; ++v) d_logits_z[v] /= kPatchLen;
                    for (int j = 0; j < m; ++j) d_z_recon[j] = 0.0;
                    dec.backward(&zS[b][i][0], &d_logits_z[0], &d_z_recon[0]);
                    for (int j = 0; j < m; ++j) d_z[i][j] += kReconWeight * d_z_recon[j];
                }
            }

            // Per-sample norm hinge: L_norm_bi = max(0, kNormTarget - ||z_bi||^2)
            //   d L_norm_bi / d z_bij = -2 * kNormWeight * z_bij  (if ||z||^2 < kNormTarget)
            for (int i = 0; i < kNumPatches; ++i) {
                double zn2 = 0.0;
                for (int j = 0; j < m; ++j) zn2 += zS[b][i][j] * zS[b][i][j];
                if (zn2 < kNormTarget) {
                    norm_loss_acc += (kNormTarget - zn2);
                    for (int j = 0; j < m; ++j) d_z[i][j] += -2.0 * kNormWeight * zS[b][i][j];
                }
            }

            // InfoNCE contrastive term with cosine similarity + L2 normalization.
            //   Anchor:  zS[b][i] (student); zhat = zS[b][i] / ||zS[b][i]||
            //   Pos:     tzS[b][i+W] (teacher, stopgrad); thaht analogous
            //   Negs:    tzS[bp][i+W] for bp ≠ b (in-batch teacher candidates)
            //   sim[bp]  = cos(z_anchor, t_pos[bp]) / τ
            //   L_bi     = -log softmax(sim)[b]
            // Cosine + normalization breaks the magnitude-collapse symmetry:
            // z = const is no longer a trivial solution because cosine ignores magnitude
            // and the encoder must produce DIRECTIONALLY varied outputs.
            // Gradient through normalization:
            //   d L / d z[k] = ( d L / d zhat[k] - zhat[k] · ⟨zhat, d L / d zhat⟩ ) / ||z||
            if (mode == MODE_LATENT_NCE) {
                std::vector<double> sim(B), p(B);
                int W = kInfoNceWindow;
                double inv_tau = 1.0 / kInfoNceTau;
                for (int i = 0; i + W < kNumPatches; ++i) {
                    // Anchor norm.
                    double z_n2 = 0.0;
                    for (int j = 0; j < m; ++j) z_n2 += zS[b][i][j] * zS[b][i][j];
                    double z_norm = std::sqrt(z_n2) + 1e-8;
                    // Teacher norms (per candidate).
                    std::vector<double> t_norm(B, 1e-8);
                    for (int bp = 0; bp < B; ++bp) {
                        double tn2 = 0.0;
                        for (int j = 0; j < m; ++j) tn2 += tzS[bp][i + W][j] * tzS[bp][i + W][j];
                        t_norm[bp] = std::sqrt(tn2) + 1e-8;
                    }
                    // Cosine sim / τ.
                    double max_sim = -1e18;
                    for (int bp = 0; bp < B; ++bp) {
                        double dot = 0.0;
                        for (int j = 0; j < m; ++j) dot += zS[b][i][j] * tzS[bp][i + W][j];
                        sim[bp] = (dot / (z_norm * t_norm[bp])) * inv_tau;
                        if (sim[bp] > max_sim) max_sim = sim[bp];
                    }
                    double sum_e = 0.0;
                    for (int bp = 0; bp < B; ++bp) { p[bp] = std::exp(sim[bp] - max_sim); sum_e += p[bp]; }
                    for (int bp = 0; bp < B; ++bp) p[bp] /= sum_e;
                    double L_bi = -(sim[b] - max_sim - std::log(sum_e));
                    nce_loss_acc += L_bi;
                    nce_count++;
                    // d L / d zhat[j] = (1/τ) Σ_bp (p[bp] - 1{bp==b}) · t_zhat[bp][j]
                    //                 = (1/τ) Σ_bp (p[bp] - 1{bp==b}) · tzS[bp][i+W][j] / t_norm[bp]
                    std::vector<double> d_zhat(m, 0.0);
                    for (int j = 0; j < m; ++j) {
                        double g_j = 0.0;
                        for (int bp = 0; bp < B; ++bp) {
                            double delta = (bp == b) ? (p[bp] - 1.0) : p[bp];
                            g_j += delta * tzS[bp][i + W][j] / t_norm[bp];
                        }
                        d_zhat[j] = inv_tau * g_j;
                    }
                    // Convert d_zhat to d_z via the chain rule.
                    double dot_zd = 0.0;
                    for (int j = 0; j < m; ++j) dot_zd += (zS[b][i][j] / z_norm) * d_zhat[j];
                    for (int j = 0; j < m; ++j) {
                        double d_z_j = (d_zhat[j] - (zS[b][i][j] / z_norm) * dot_zd) / z_norm;
                        d_z[i][j] += kInfoNceWeight * d_z_j;
                    }
                }
            }

            // Backward through encoder for each patch.
            for (int i = 0; i < kNumPatches; ++i) {
                enc.backward(&batch_x[b][i * kPatchLen], kPatchLen, &d_z[i][0], &emS[b][i][0]);
            }
        }

        total /= B;
        total += kNormWeight * norm_loss_acc / (B * kNumPatches);
        if (mode == MODE_LATENT_NCE && nce_count > 0) {
            total += kInfoNceWeight * nce_loss_acc / nce_count;
        }
        if (mode == MODE_LATENT_NCE && recon_count > 0) {
            total += kReconWeight * recon_loss_acc / recon_count;
        }

        // Average gradients across batch.
        double invB = 1.0 / double(B);
        for (int i = 0; i < (int)enc.dEmb.size(); ++i) enc.dEmb[i] *= invB;
        for (int i = 0; i < (int)enc.dW.size();   ++i) enc.dW[i]   *= invB;
        for (int i = 0; i < (int)enc.db.size();   ++i) enc.db[i]   *= invB;
        for (int i = 0; i < (int)kop.dK.size();   ++i) kop.dK[i]   *= invB;
        for (int i = 0; i < (int)kop.dB.size();   ++i) kop.dB[i]   *= invB;
        if (mode == MODE_TOKEN || mode == MODE_LATENT_NCE) {
            for (int i = 0; i < (int)dec.dWd.size(); ++i) dec.dWd[i] *= invB;
            for (int i = 0; i < (int)dec.dbd.size(); ++i) dec.dbd[i] *= invB;
        }

        // Gradient clipping (per parameter group, L2 norm ≤ kGradClip).
        clip_l2(&enc.dEmb[0], (int)enc.dEmb.size(), kGradClip);
        clip_l2(&enc.dW[0],   (int)enc.dW.size(),   kGradClip);
        clip_l2(&enc.db[0],   (int)enc.db.size(),   kGradClip);
        clip_l2(&kop.dK[0],   (int)kop.dK.size(),   kGradClip);
        clip_l2(&kop.dB[0],   (int)kop.dB.size(),   kGradClip);
        if (mode == MODE_TOKEN || mode == MODE_LATENT_NCE) {
            clip_l2(&dec.dWd[0], (int)dec.dWd.size(), kGradClip);
            clip_l2(&dec.dbd[0], (int)dec.dbd.size(), kGradClip);
        }

        enc.sgd_update(lr);
        // Phase-0c-A: identity-regularization on K and per-group lr scaling for K, B in MODE_LATENT_NCE.
        // dK += kIdentityReg * (K - I)  pulls K toward identity each step.
        if (mode == MODE_LATENT_NCE) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    double target = (i == j) ? 1.0 : 0.0;
                    kop.dK[i * m + j] += kIdentityReg * (kop.Kop[i * m + j] - target);
                }
            }
            kop.sgd_update(lr * kKopLrScale);
        } else {
            kop.sgd_update(lr);
        }
        if (mode == MODE_TOKEN || mode == MODE_LATENT_NCE) dec.sgd_update(lr);
        // Teacher tracks student via EMA, AFTER the student step.
        if (mode == MODE_LATENT_NCE) teacher_ema_step();

        return total;
    }

    // Encode a single stream and return s[i] for all patches.
    void encode_stream(const std::vector<int>& x, std::vector<std::vector<double> >& s_out) const {
        std::vector<std::vector<double> > z(kNumPatches, std::vector<double>(m, 0.0));
        std::vector<double> e_mean(kEmbDim, 0.0);
        for (int i = 0; i < kNumPatches; ++i)
            enc.forward(&x[i * kPatchLen], kPatchLen, &z[i][0], &e_mean[0]);
        s_out.assign(kNumPatches, std::vector<double>(m, 0.0));
        std::vector<double> s_init(m, 0.0);
        kop.step(&s_init[0], &z[0][0], &s_out[0][0]);
        for (int i = 1; i < kNumPatches; ++i)
            kop.step(&s_out[i-1][0], &z[i][0], &s_out[i][0]);
    }

    // Encode a stream and also produce the spectral memory state M[i][j][k] for each
    // patch i, slot j ∈ [0, kMemSlots), feature k ∈ [0, m). Memory uses gated multi-
    // timescale EMA: M[i][j] = M[i-1][j] - g_i * (1 - λ_j) * (M[i-1][j] - z[i]).
    // Returns also the average write gate (write-cost diagnostic).
    double encode_stream_with_memory(const std::vector<int>& x,
                                     std::vector<std::vector<double> >& s_out,
                                     std::vector<std::vector<std::vector<double> > >& M_out) const {
        std::vector<std::vector<double> > z(kNumPatches, std::vector<double>(m, 0.0));
        std::vector<double> e_mean(kEmbDim, 0.0);
        for (int i = 0; i < kNumPatches; ++i)
            enc.forward(&x[i * kPatchLen], kPatchLen, &z[i][0], &e_mean[0]);
        s_out.assign(kNumPatches, std::vector<double>(m, 0.0));
        std::vector<double> s_init(m, 0.0);
        kop.step(&s_init[0], &z[0][0], &s_out[0][0]);
        for (int i = 1; i < kNumPatches; ++i)
            kop.step(&s_out[i-1][0], &z[i][0], &s_out[i][0]);
        // Memory: kMemSlots slots × m dims, all initialised to 0.
        M_out.assign(kNumPatches, std::vector<std::vector<double> >(kMemSlots, std::vector<double>(m, 0.0)));
        double gate_sum = 0.0;
        for (int i = 0; i < kNumPatches; ++i) {
            // Write gate: scalar g_i = sigmoid(Wg · s_i + bg).
            double pre = bg;
            for (int j = 0; j < m; ++j) pre += Wg[j] * s_out[i][j];
            double g = 1.0 / (1.0 + std::exp(-pre));
            gate_sum += g;
            // For each slot, gated EMA toward z[i].
            for (int s_idx = 0; s_idx < kMemSlots; ++s_idx) {
                double lambda = kMemDecays[s_idx];
                double prev_factor = 1.0 - g * (1.0 - lambda);
                double new_factor  = g * (1.0 - lambda);
                if (i == 0) {
                    for (int k = 0; k < m; ++k) M_out[0][s_idx][k] = new_factor * z[0][k];
                } else {
                    for (int k = 0; k < m; ++k)
                        M_out[i][s_idx][k] = prev_factor * M_out[i-1][s_idx][k] + new_factor * z[i][k];
                }
            }
        }
        return gate_sum / kNumPatches;  // average gate per step
    }
};

// =============== Linear probe (multinomial logistic on frozen s) ===============
struct LinearProbe {
    std::vector<double> Wp, bp;
    int S_, m_;

    void init(int S, int m, unsigned seed) {
        S_ = S; m_ = m;
        Wp.assign(S * m, 0.0);
        bp.assign(S, 0.0);
        srand(seed);
        double sw = std::sqrt(2.0 / (S + m));
        for (int i = 0; i < S * m; ++i) Wp[i] = rand_gauss() * sw;
    }
    double accuracy(const std::vector<std::vector<double> >& xs, const std::vector<int>& ys) const {
        int N = (int)xs.size();
        if (N == 0) return 0.0;
        std::vector<double> logits(S_);
        int correct = 0;
        for (int i = 0; i < N; ++i) {
            matvec(&Wp[0], &xs[i][0], &logits[0], S_, m_);
            for (int s = 0; s < S_; ++s) logits[s] += bp[s];
            int pred = 0; double lmax = logits[0];
            for (int s = 1; s < S_; ++s) if (logits[s] > lmax) { lmax = logits[s]; pred = s; }
            if (pred == ys[i]) correct++;
        }
        return double(correct) / N;
    }
    double train(const std::vector<std::vector<double> >& xs, const std::vector<int>& ys,
                 int steps_p, double lr_p) {
        int N = (int)xs.size();
        if (N == 0) return 0.0;
        std::vector<double> dW(S_ * m_), db(S_), logits(S_), dlog(S_), p(S_);
        for (int step = 0; step < steps_p; ++step) {
            for (int i = 0; i < S_ * m_; ++i) dW[i] = 0.0;
            for (int s = 0; s < S_; ++s) db[s] = 0.0;
            for (int i = 0; i < N; ++i) {
                matvec(&Wp[0], &xs[i][0], &logits[0], S_, m_);
                for (int s = 0; s < S_; ++s) logits[s] += bp[s];
                double lmax = logits[0];
                for (int s = 1; s < S_; ++s) if (logits[s] > lmax) lmax = logits[s];
                double sum_e = 0.0;
                for (int s = 0; s < S_; ++s) { p[s] = std::exp(logits[s] - lmax); sum_e += p[s]; }
                for (int s = 0; s < S_; ++s) p[s] /= sum_e;
                for (int s = 0; s < S_; ++s) dlog[s] = p[s];
                dlog[ys[i]] -= 1.0;
                outer_add(&dW[0], 1.0 / N, &dlog[0], &xs[i][0], S_, m_);
                axpy(&db[0], 1.0 / N, &dlog[0], S_);
            }
            for (int i = 0; i < S_ * m_; ++i) Wp[i] -= lr_p * dW[i];
            for (int s = 0; s < S_; ++s) bp[s] -= lr_p * db[s];
        }
        return accuracy(xs, ys);
    }
};

// =============== Evaluation: linear-probe accuracy on held-out streams ===============
// Returns acc_s, acc_z (existing probes) and — if use_memory — acc_m (probe of memory
// only, the kMemSlots*m vector flattened) and acc_sm (probe of concat(s, M)).
// Also returns avg_gate (write rate diagnostic, 0 if memory not used).
static void eval_probes(const Trainer& tr, int n_train_streams, int n_test_streams,
                        unsigned eval_seed, double& acc_s_out, double& acc_z_out,
                        double& acc_m_out, double& acc_sm_out, double& avg_gate_out) {
    srand(eval_seed);
    std::vector<int> x, h;
    std::vector<std::vector<double> > xs_s_tr, xs_s_te, xs_z_tr, xs_z_te;
    std::vector<std::vector<double> > xs_m_tr, xs_m_te, xs_sm_tr, xs_sm_te;
    std::vector<int> y_tr, y_te;
    std::vector<double> e_mean(kEmbDim);
    int mem_dim = kMemSlots * tr.m;     // dim of flattened memory
    int sm_dim  = tr.m + mem_dim;       // dim of (s, M) concatenated
    double gate_total = 0.0;
    int gate_streams = 0;
    for (int n = 0; n < n_train_streams + n_test_streams; ++n) {
        tr.hmm.generate(kStream, x, h);
        std::vector<std::vector<double> > s_seq, z_seq(kNumPatches, std::vector<double>(tr.m, 0.0));
        std::vector<std::vector<std::vector<double> > > M_seq;
        if (tr.use_memory) {
            double g = tr.encode_stream_with_memory(x, s_seq, M_seq);
            gate_total += g; gate_streams++;
        } else {
            tr.encode_stream(x, s_seq);
        }
        for (int i = 0; i < kNumPatches; ++i)
            tr.enc.forward(&x[i * kPatchLen], kPatchLen, &z_seq[i][0], &e_mean[0]);
        for (int i = 0; i < kNumPatches; ++i) {
            int counts[16] = {0};
            for (int l = 0; l < kPatchLen; ++l) counts[h[i * kPatchLen + l]]++;
            int best = 0, bc = counts[0];
            for (int s = 1; s < tr.hmm.S; ++s) if (counts[s] > bc) { bc = counts[s]; best = s; }
            // s and z (always present).
            std::vector<double> sm_flat(sm_dim, 0.0);
            std::vector<double> m_flat(mem_dim, 0.0);
            if (tr.use_memory) {
                // Flatten M_seq[i] into m_flat: [slot0, slot1, slot2, slot3] concatenated.
                int idx = 0;
                for (int sl = 0; sl < kMemSlots; ++sl)
                    for (int k = 0; k < tr.m; ++k) m_flat[idx++] = M_seq[i][sl][k];
                // sm_flat: [s, M].
                for (int k = 0; k < tr.m; ++k) sm_flat[k] = s_seq[i][k];
                for (int k = 0; k < mem_dim; ++k) sm_flat[tr.m + k] = m_flat[k];
            }
            if (n < n_train_streams) {
                xs_s_tr.push_back(s_seq[i]); xs_z_tr.push_back(z_seq[i]); y_tr.push_back(best);
                if (tr.use_memory) { xs_m_tr.push_back(m_flat); xs_sm_tr.push_back(sm_flat); }
            } else {
                xs_s_te.push_back(s_seq[i]); xs_z_te.push_back(z_seq[i]); y_te.push_back(best);
                if (tr.use_memory) { xs_m_te.push_back(m_flat); xs_sm_te.push_back(sm_flat); }
            }
        }
    }
    LinearProbe probe_s, probe_z;
    probe_s.init(tr.hmm.S, tr.m, eval_seed + 1);
    probe_s.train(xs_s_tr, y_tr, 200, 0.5);
    acc_s_out = probe_s.accuracy(xs_s_te, y_te);
    probe_z.init(tr.hmm.S, tr.m, eval_seed + 2);
    probe_z.train(xs_z_tr, y_tr, 200, 0.5);
    acc_z_out = probe_z.accuracy(xs_z_te, y_te);
    if (tr.use_memory) {
        LinearProbe probe_m, probe_sm;
        probe_m.init(tr.hmm.S, mem_dim, eval_seed + 3);
        probe_m.train(xs_m_tr, y_tr, 200, 0.2);
        acc_m_out = probe_m.accuracy(xs_m_te, y_te);
        probe_sm.init(tr.hmm.S, sm_dim, eval_seed + 4);
        probe_sm.train(xs_sm_tr, y_tr, 200, 0.2);
        acc_sm_out = probe_sm.accuracy(xs_sm_te, y_te);
        avg_gate_out = (gate_streams > 0) ? gate_total / gate_streams : 0.0;
    } else {
        acc_m_out = 0.0; acc_sm_out = 0.0; avg_gate_out = 0.0;
    }
}

// =============== Held-out loss eval ===============
static double eval_held_loss(const Trainer& tr, int n_streams, unsigned eval_seed) {
    srand(eval_seed);
    std::vector<int> x, h;
    double total = 0.0; int count = 0;
    std::vector<double> e_mean(kEmbDim);
    for (int n = 0; n < n_streams; ++n) {
        tr.hmm.generate(kStream, x, h);
        std::vector<std::vector<double> > s_seq;
        tr.encode_stream(x, s_seq);
        std::vector<std::vector<double> > z(kNumPatches, std::vector<double>(tr.m, 0.0));
        for (int i = 0; i < kNumPatches; ++i)
            tr.enc.forward(&x[i * kPatchLen], kPatchLen, &z[i][0], &e_mean[0]);
        double stream_loss = 0.0;
        if (tr.mode == Trainer::MODE_LATENT) {
            std::vector<double> pred(tr.m);
            for (int i = 1; i < kNumPatches; ++i) {
                tr.kop.predict_h1(&s_seq[i-1][0], &pred[0]);
                double l = 0.0;
                for (int j = 0; j < tr.m; ++j) { double d = pred[j] - z[i][j]; l += d*d; }
                stream_loss += l / tr.m;
            }
            stream_loss /= (kNumPatches - 1);
        } else {
            std::vector<double> logits(kVocab);
            for (int i = 0; i < kNumPatches - 1; ++i) {
                tr.dec.forward(&s_seq[i][0], &logits[0]);
                double lmax = logits[0];
                for (int v = 1; v < kVocab; ++v) if (logits[v] > lmax) lmax = logits[v];
                double sum_e = 0.0; std::vector<double> p(kVocab);
                for (int v = 0; v < kVocab; ++v) { p[v] = std::exp(logits[v] - lmax); sum_e += p[v]; }
                for (int v = 0; v < kVocab; ++v) p[v] /= sum_e;
                double l = 0.0;
                for (int l_t = 0; l_t < kPatchLen; ++l_t) {
                    int tok = x[(i+1) * kPatchLen + l_t];
                    l += -std::log(std::max(p[tok], 1e-30));
                }
                stream_loss += l / kPatchLen;
            }
            stream_loss /= (kNumPatches - 1);
        }
        total += stream_loss; count++;
    }
    return total / count;
}

// =============== z-variance diagnostic ===============
static double eval_z_var(const Trainer& tr, unsigned eval_seed) {
    srand(eval_seed);
    std::vector<int> x, h;
    std::vector<double> sum_z(tr.m, 0.0), sq_z(tr.m, 0.0);
    std::vector<double> z(tr.m), e_mean(kEmbDim);
    int count = 0;
    const int N = 8;
    for (int n = 0; n < N; ++n) {
        tr.hmm.generate(kStream, x, h);
        for (int i = 0; i < kNumPatches; ++i) {
            tr.enc.forward(&x[i * kPatchLen], kPatchLen, &z[0], &e_mean[0]);
            for (int j = 0; j < tr.m; ++j) { sum_z[j] += z[j]; sq_z[j] += z[j]*z[j]; }
            count++;
        }
    }
    double avg_var = 0.0;
    for (int j = 0; j < tr.m; ++j) {
        double mean = sum_z[j] / count;
        double var = sq_z[j] / count - mean * mean;
        if (var < 0) var = 0;
        avg_var += var;
    }
    return avg_var / tr.m;
}

// =============== Main ===============
int main(int argc, char** argv) {
    std::string mode_str = "latent";
    int seed = 42;
    int steps = 500;
    int print_every = 50;
    double lr = 0.01;

    bool use_memory = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--mode"        && i+1 < argc) mode_str = argv[++i];
        else if (a == "--seed"        && i+1 < argc) seed = std::atoi(argv[++i]);
        else if (a == "--steps"       && i+1 < argc) steps = std::atoi(argv[++i]);
        else if (a == "--lr"          && i+1 < argc) lr = std::atof(argv[++i]);
        else if (a == "--print-every" && i+1 < argc) print_every = std::atoi(argv[++i]);
        else if (a == "--use-memory")                use_memory = true;
        else { std::fprintf(stderr, "Unknown arg %s\n", argv[i]); return 1; }
    }

    Trainer tr;
    tr.init(seed);
    tr.use_memory = use_memory;
    tr.lr = lr;
    if      (mode_str == "latent")     tr.mode = Trainer::MODE_LATENT;
    else if (mode_str == "token")      tr.mode = Trainer::MODE_TOKEN;
    else if (mode_str == "latent_nce") tr.mode = Trainer::MODE_LATENT_NCE;
    else { std::fprintf(stderr, "--mode must be 'latent', 'token', or 'latent_nce'\n"); return 1; }

    std::printf("# EALRMN Phase-0  mode=%s seed=%d steps=%d lr=%.4f use_memory=%d\n",
                mode_str.c_str(), seed, steps, lr, tr.use_memory ? 1 : 0);
    if (tr.use_memory)
        std::printf("# step\ttrain_loss\theld_loss\tprobe_s\tprobe_z\tprobe_m\tprobe_sm\tgate\tz_var\n");
    else
        std::printf("# step\ttrain_loss\theld_loss\tprobe_s\tprobe_z\tz_var\n");

    std::vector<int> x_tmp, h_tmp;
    for (int step = 0; step < steps; ++step) {
        srand((unsigned)(seed * 31 + step * 7919 + 1));
        std::vector<std::vector<int> > batch(kBatch);
        for (int b = 0; b < kBatch; ++b) {
            tr.hmm.generate(kStream, x_tmp, h_tmp);
            batch[b] = x_tmp;
        }
        double tloss = tr.train_step(batch);

        if (step % print_every == 0 || step == steps - 1) {
            double hloss = eval_held_loss(tr, 16, (unsigned)(seed + 1000 + step));
            double acc_s = 0.0, acc_z = 0.0, acc_m = 0.0, acc_sm = 0.0, gate = 0.0;
            eval_probes(tr, 8, 8, (unsigned)(seed + 2000 + step),
                        acc_s, acc_z, acc_m, acc_sm, gate);
            double zvar  = eval_z_var(tr, (unsigned)(seed + 3000 + step));
            if (tr.use_memory) {
                std::printf("%6d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                            step, tloss, hloss, acc_s, acc_z, acc_m, acc_sm, gate, zvar);
            } else {
                std::printf("%6d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                            step, tloss, hloss, acc_s, acc_z, zvar);
            }
            std::fflush(stdout);
        }
    }
    return 0;
}
