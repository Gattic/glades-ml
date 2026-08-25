// EALRMN-v1 Phase-0e prototype — per-token needle-in-haystack.
//
// Phase-0d revealed that the mean-pool patch encoder was the bottleneck:
// KV "patches" (markers + IDs) and filler "patches" produce statistically
// similar z's at random init, so neither the bounded-memory write gate nor
// cosine attention can discriminate them.
//
// Phase-0e changes ONE thing: kPatchLen=1. Each token is its own "patch":
// z_t = Emb[x_t] · W + b is just an embedding lookup followed by a linear
// projection. Distinct token types now have distinct z's even at random init.
// The recurrence, memory mechanism, and attention readout are unchanged.
//
// Pre-registered prediction (from Phase-0d failure analysis):
//   At T=64 tokens, the per-token EALRMN model should:
//     (a) Learn a gate that fires high on marker/key/value tokens, low on filler.
//     (b) Achieve significantly higher retrieval accuracy than per-patch baseline.
//   The per-token ATTN model should reach near-perfect accuracy at T=64.
//   The per-token RNN model should match EALRMN closely (still no memory advantage)
//   OR EALRMN should win if memory specifically helps long-range retention.
//
// Build:
//   g++ -std=c++98 -O2 -Wall -Wextra research/ealrmn_phase0e_pertoken.cpp -o research/ealrmn_phase0e_pertoken
// Run:
//   ./research/ealrmn_phase0e_pertoken --model ealrmn --seed 42 --steps 2000 --T 64

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

// =============== Task constants ===============
static const int kFillerMax    = 16;
static const int kKeyMarker    = 16;
static const int kValMarker    = 17;
static const int kQueryMarker  = 18;
static const int kKeyStart     = 19;
static const int kNumKeys      = 4;
static const int kValStart     = 23;
static const int kNumVals      = 4;
static const int kVocab        = kValStart + kNumVals;

// =============== Architecture constants ===============
static const int kEmbDim       = 16;
static const int kM            = 32;
static const int kBatch        = 16;
static const int kMemSlots     = 4;
static const double kMemDecays[4] = { 0.50, 0.80, 0.95, 0.99 };
static const double kGradClip  = 10.0;
static const double kWritePenalty = 0.001;   // very weak — gate must learn first

// =============== Random helpers ===============
static double rand_unif() { return (double(rand()) + 1.0) / (double(RAND_MAX) + 2.0); }
static double rand_gauss() {
    double u1 = rand_unif(), u2 = rand_unif();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
}
static int rand_int(int n) { return std::rand() % n; }

// =============== Matrix helpers ===============
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

// =============== Needle task (per-token) ===============
// Stream of T tokens, 2 KV pairs (8 tokens) at random non-overlapping positions,
// query (2 tokens) at the end. Label is the value for the queried key.
struct NeedleTask {
    int T;
    NeedleTask() : T(64) {}

    void generate(std::vector<int>& x, int& label) const {
        x.assign(T, 0);
        for (int i = 0; i < T; ++i) x[i] = rand_int(kFillerMax);

        // Query is at positions T-2, T-1.
        // KV pairs occupy 4 consecutive tokens starting at some position p,
        // and must end before T-2 to not overlap the query.
        // Pick 2 distinct, non-overlapping positions in [0, T-2-4+1) = [0, T-5).
        int max_start = T - 2 - 4;  // last legal start position (inclusive)
        if (max_start < 4) {
            // Stream too short — degenerate; just pick (0, 4).
            // For T=64 this never triggers; max_start=58.
        }
        int p1, p2;
        do {
            p1 = rand_int(max_start + 1);
            p2 = rand_int(max_start + 1);
            // Non-overlap: |p1 - p2| >= 4.
        } while (std::abs(p1 - p2) < 4);

        int keys[2], vals[2];
        keys[0] = rand_int(kNumKeys);
        do { keys[1] = rand_int(kNumKeys); } while (keys[1] == keys[0]);
        vals[0] = rand_int(kNumVals);
        vals[1] = rand_int(kNumVals);

        for (int k = 0; k < 2; ++k) {
            int pos = (k == 0 ? p1 : p2);
            x[pos]     = kKeyMarker;
            x[pos + 1] = kKeyStart + keys[k];
            x[pos + 2] = kValMarker;
            x[pos + 3] = kValStart + vals[k];
        }

        int q_choice = rand_int(2);
        x[T - 2] = kQueryMarker;
        x[T - 1] = kKeyStart + keys[q_choice];

        label = vals[q_choice];
    }
};

// =============== Per-token encoder ===============
// z_t = W · Emb[x_t] + b  — no mean-pool, just per-token embedding + linear.
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
    // Forward: e = Emb[tok];  z = W · e + b.  e is cached for backward.
    void forward(int tok, double* z_out, double* e_cache) const {
        for (int j = 0; j < d_emb_; ++j) e_cache[j] = Emb[tok * d_emb_ + j];
        matvec(&W[0], e_cache, z_out, m_, d_emb_);
        for (int i = 0; i < m_; ++i) z_out[i] += b[i];
    }
    // Backward: accumulates dEmb[tok], dW, db given d_z.
    void backward(int tok, const double* d_z, const double* e_cache) {
        axpy(&db[0], 1.0, d_z, m_);
        outer_add(&dW[0], 1.0, d_z, e_cache, m_, d_emb_);
        std::vector<double> d_e(d_emb_, 0.0);
        matvec_T(&W[0], d_z, &d_e[0], m_, d_emb_);
        for (int j = 0; j < d_emb_; ++j) dEmb[tok * d_emb_ + j] += d_e[j];
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
        for (int i = 0; i < m; ++i) Kop[i * m + i] += 0.95;
        for (int i = 0; i < m * m; ++i) Bop[i] = rand_gauss() * sk;
        zero_grad();
    }
    void zero_grad() { dK.assign(m_ * m_, 0.0); dB.assign(m_ * m_, 0.0); }
    void step(const double* s_in, const double* z, double* s_out) const {
        std::vector<double> Ks(m_), Bz(m_);
        matvec(&Kop[0], s_in, &Ks[0], m_, m_);
        matvec(&Bop[0], z,    &Bz[0], m_, m_);
        for (int i = 0; i < m_; ++i) s_out[i] = Ks[i] + Bz[i];
    }
    void backward_step(const double* s_in, const double* z, const double* d_s_out,
                       double* d_s_in, double* d_z) {
        outer_add(&dK[0], 1.0, d_s_out, s_in, m_, m_);
        outer_add(&dB[0], 1.0, d_s_out, z,    m_, m_);
        matvec_T(&Kop[0], d_s_out, d_s_in, m_, m_);
        matvec_T(&Bop[0], d_s_out, d_z,    m_, m_);
    }
    void sgd_update(double lr) {
        for (int i = 0; i < m_ * m_; ++i) Kop[i] -= lr * dK[i];
        for (int i = 0; i < m_ * m_; ++i) Bop[i] -= lr * dB[i];
    }
};

// =============== Readout head ===============
struct Readout {
    std::vector<double> W, b;
    std::vector<double> dW, db;
    int feat_dim_, n_classes_;
    Readout() : feat_dim_(0), n_classes_(0) {}

    void init(int feat_dim, int n_classes, unsigned seed) {
        feat_dim_ = feat_dim; n_classes_ = n_classes;
        W.assign(n_classes * feat_dim, 0.0);
        b.assign(n_classes, 0.0);
        srand(seed);
        double sw = std::sqrt(2.0 / (n_classes + feat_dim));
        for (int i = 0; i < n_classes * feat_dim; ++i) W[i] = rand_gauss() * sw;
        zero_grad();
    }
    void zero_grad() {
        dW.assign(n_classes_ * feat_dim_, 0.0);
        db.assign(n_classes_, 0.0);
    }
    void forward(const double* feat, double* logits) const {
        matvec(&W[0], feat, logits, n_classes_, feat_dim_);
        for (int c = 0; c < n_classes_; ++c) logits[c] += b[c];
    }
    double loss_and_grad(const double* logits, int label, double* d_logits) const {
        double max_l = logits[0];
        for (int c = 1; c < n_classes_; ++c) if (logits[c] > max_l) max_l = logits[c];
        std::vector<double> p(n_classes_, 0.0);
        double sum_e = 0.0;
        for (int c = 0; c < n_classes_; ++c) { p[c] = std::exp(logits[c] - max_l); sum_e += p[c]; }
        for (int c = 0; c < n_classes_; ++c) p[c] /= sum_e;
        double loss = -std::log(std::max(p[label], 1e-30));
        for (int c = 0; c < n_classes_; ++c) d_logits[c] = p[c];
        d_logits[label] -= 1.0;
        return loss;
    }
    void backward(const double* feat, const double* d_logits, double* d_feat) {
        outer_add(&dW[0], 1.0, d_logits, feat, n_classes_, feat_dim_);
        axpy(&db[0], 1.0, d_logits, n_classes_);
        matvec_T(&W[0], d_logits, d_feat, n_classes_, feat_dim_);
    }
    int predict(const double* logits) const {
        int best = 0; double bm = logits[0];
        for (int c = 1; c < n_classes_; ++c) if (logits[c] > bm) { bm = logits[c]; best = c; }
        return best;
    }
    void sgd_update(double lr) {
        for (int i = 0; i < n_classes_ * feat_dim_; ++i) W[i] -= lr * dW[i];
        for (int c = 0; c < n_classes_; ++c) b[c] -= lr * db[c];
    }
};

// =============== Trainer ===============
struct Trainer {
    NeedleTask task;
    Encoder enc;
    Koopman kop;
    Readout readout;
    // Gate parameters (EALRMN only).
    std::vector<double> Wg, dWg;
    double bg, dbg;
    enum Model { MODEL_EALRMN, MODEL_RNN, MODEL_ATTN };
    Model model;
    int m;
    double lr;
    int feat_dim;

    Trainer() : bg(0.0), dbg(0.0), model(MODEL_EALRMN), m(kM), lr(0.005), feat_dim(0) {}

    void init(int seed, int T, Model mdl) {
        model = mdl;
        task.T = T;
        enc.init(kVocab, kEmbDim, kM, (unsigned)(seed + 100));
        kop.init(kM, (unsigned)(seed + 200));
        Wg.assign(kM, 0.0);
        dWg.assign(kM, 0.0);
        bg = 0.0; dbg = 0.0;
        srand((unsigned)(seed + 400));
        for (int i = 0; i < kM; ++i) Wg[i] = rand_gauss() * 0.1 / std::sqrt((double)kM);
        switch (model) {
            case MODEL_EALRMN: feat_dim = kM + kMemSlots * kM; break;
            case MODEL_RNN:    feat_dim = kM; break;
            case MODEL_ATTN:   feat_dim = kM; break;
        }
        readout.init(feat_dim, kNumVals, (unsigned)(seed + 300));
        srand((unsigned)(seed + 999));
    }

    // Run recurrence and (optionally) memory over T per-token z_seq.
    void run_recurrence(const std::vector<std::vector<double> >& z_seq,
                        std::vector<std::vector<double> >& s_seq,
                        std::vector<std::vector<std::vector<double> > >& M_seq,
                        std::vector<double>& gate_seq,
                        bool use_memory) const {
        int T = z_seq.size();
        s_seq.assign(T, std::vector<double>(m, 0.0));
        std::vector<double> s_init(m, 0.0);
        kop.step(&s_init[0], &z_seq[0][0], &s_seq[0][0]);
        for (int t = 1; t < T; ++t)
            kop.step(&s_seq[t-1][0], &z_seq[t][0], &s_seq[t][0]);
        gate_seq.assign(use_memory ? T : 0, 0.0);
        if (use_memory) {
            M_seq.assign(T, std::vector<std::vector<double> >(kMemSlots, std::vector<double>(m, 0.0)));
            for (int t = 0; t < T; ++t) {
                double pre = bg;
                for (int k = 0; k < m; ++k) pre += Wg[k] * z_seq[t][k];
                double g = 1.0 / (1.0 + std::exp(-pre));
                gate_seq[t] = g;
                for (int sl = 0; sl < kMemSlots; ++sl) {
                    double lambda = kMemDecays[sl];
                    double write_factor = g * (1.0 - lambda);
                    double carry_factor = 1.0 - write_factor;
                    if (t == 0) {
                        for (int k = 0; k < m; ++k) M_seq[0][sl][k] = write_factor * z_seq[0][k];
                    } else {
                        for (int k = 0; k < m; ++k)
                            M_seq[t][sl][k] = carry_factor * M_seq[t-1][sl][k] + write_factor * z_seq[t][k];
                    }
                }
            }
        }
    }

    void build_feat(const std::vector<std::vector<double> >& z_seq,
                    const std::vector<std::vector<double> >& s_seq,
                    const std::vector<std::vector<std::vector<double> > >& M_seq,
                    std::vector<double>& feat_out,
                    std::vector<double>* attn_weights_out) const {
        int T = z_seq.size();
        feat_out.assign(feat_dim, 0.0);
        switch (model) {
            case MODEL_EALRMN: {
                for (int k = 0; k < m; ++k) feat_out[k] = s_seq[T-1][k];
                int idx = m;
                for (int sl = 0; sl < kMemSlots; ++sl)
                    for (int k = 0; k < m; ++k) feat_out[idx++] = M_seq[T-1][sl][k];
                break;
            }
            case MODEL_RNN: {
                for (int k = 0; k < m; ++k) feat_out[k] = s_seq[T-1][k];
                break;
            }
            case MODEL_ATTN: {
                const std::vector<double>& q = z_seq[T-1];
                double qn2 = 0.0;
                for (int k = 0; k < m; ++k) qn2 += q[k] * q[k];
                double qn = std::sqrt(qn2) + 1e-8;
                std::vector<double> sims(T-1, 0.0);
                double max_s = -1e18;
                for (int t = 0; t < T-1; ++t) {
                    double zn2 = 0.0, dot = 0.0;
                    for (int k = 0; k < m; ++k) { zn2 += z_seq[t][k] * z_seq[t][k]; dot += q[k] * z_seq[t][k]; }
                    double zn = std::sqrt(zn2) + 1e-8;
                    sims[t] = (dot / (qn * zn)) / 0.2;
                    if (sims[t] > max_s) max_s = sims[t];
                }
                double sum_e = 0.0;
                std::vector<double> alpha(T-1, 0.0);
                for (int t = 0; t < T-1; ++t) { alpha[t] = std::exp(sims[t] - max_s); sum_e += alpha[t]; }
                for (int t = 0; t < T-1; ++t) alpha[t] /= sum_e;
                for (int k = 0; k < m; ++k) {
                    double acc = 0.0;
                    for (int t = 0; t < T-1; ++t) acc += alpha[t] * z_seq[t][k];
                    feat_out[k] = acc;
                }
                if (attn_weights_out != NULL) *attn_weights_out = alpha;
                break;
            }
        }
    }

    double train_step(int B) {
        enc.zero_grad(); kop.zero_grad(); readout.zero_grad();
        for (int i = 0; i < (int)Wg.size(); ++i) dWg[i] = 0.0;
        dbg = 0.0;
        double total_loss = 0.0;
        double write_loss_acc = 0.0;
        int write_count = 0;
        std::vector<int> x; int label;
        for (int b = 0; b < B; ++b) {
            task.generate(x, label);
            int T = task.T;
            std::vector<std::vector<double> > z_seq(T, std::vector<double>(m, 0.0));
            std::vector<std::vector<double> > emS(T, std::vector<double>(kEmbDim, 0.0));
            for (int t = 0; t < T; ++t) enc.forward(x[t], &z_seq[t][0], &emS[t][0]);

            std::vector<std::vector<double> > s_seq;
            std::vector<std::vector<std::vector<double> > > M_seq;
            std::vector<double> gate_seq;
            bool use_memory = (model == MODEL_EALRMN);
            if (model == MODEL_EALRMN || model == MODEL_RNN)
                run_recurrence(z_seq, s_seq, M_seq, gate_seq, use_memory);

            std::vector<double> feat, attn_w;
            build_feat(z_seq, s_seq, M_seq, feat, (model == MODEL_ATTN) ? &attn_w : NULL);

            std::vector<double> logits(kNumVals), d_logits(kNumVals);
            readout.forward(&feat[0], &logits[0]);
            double l = readout.loss_and_grad(&logits[0], label, &d_logits[0]);
            total_loss += l;

            std::vector<double> d_feat(feat_dim, 0.0);
            readout.backward(&feat[0], &d_logits[0], &d_feat[0]);

            std::vector<std::vector<double> > d_z_seq(T, std::vector<double>(m, 0.0));
            switch (model) {
                case MODEL_EALRMN: {
                    std::vector<std::vector<double> > d_s_seq(T, std::vector<double>(m, 0.0));
                    for (int k = 0; k < m; ++k) d_s_seq[T-1][k] = d_feat[k];
                    std::vector<std::vector<double> > d_M(kMemSlots, std::vector<double>(m, 0.0));
                    for (int sl = 0; sl < kMemSlots; ++sl) {
                        int idx = m + sl * m;
                        for (int k = 0; k < m; ++k) d_M[sl][k] = d_feat[idx + k];
                    }
                    for (int t = T - 1; t >= 0; --t) {
                        double g = gate_seq[t];
                        double d_g = 0.0;
                        for (int sl = 0; sl < kMemSlots; ++sl) {
                            double u_t_sl = g * (1.0 - kMemDecays[sl]);
                            double one_minus_u = 1.0 - u_t_sl;
                            for (int k = 0; k < m; ++k) {
                                double M_prev_k = (t > 0) ? M_seq[t-1][sl][k] : 0.0;
                                d_z_seq[t][k] += u_t_sl * d_M[sl][k];
                                d_g          += (1.0 - kMemDecays[sl]) * (z_seq[t][k] - M_prev_k) * d_M[sl][k];
                            }
                            for (int k = 0; k < m; ++k) d_M[sl][k] *= one_minus_u;
                        }
                        d_g += kWritePenalty;
                        double d_pre = d_g * g * (1.0 - g);
                        for (int k = 0; k < m; ++k) {
                            dWg[k]      += d_pre * z_seq[t][k];
                            d_z_seq[t][k] += d_pre * Wg[k];
                        }
                        dbg += d_pre;
                    }
                    for (int t = 0; t < T; ++t) write_loss_acc += gate_seq[t];
                    write_count += T;
                    std::vector<double> s_init(m, 0.0);
                    std::vector<double> d_s_in_buf(m), d_z_buf(m);
                    for (int t = T-1; t > 0; --t) {
                        kop.backward_step(&s_seq[t-1][0], &z_seq[t][0], &d_s_seq[t][0],
                                          &d_s_in_buf[0], &d_z_buf[0]);
                        for (int k = 0; k < m; ++k) {
                            d_s_seq[t-1][k] += d_s_in_buf[k];
                            d_z_seq[t][k] += d_z_buf[k];
                        }
                    }
                    kop.backward_step(&s_init[0], &z_seq[0][0], &d_s_seq[0][0],
                                      &d_s_in_buf[0], &d_z_buf[0]);
                    for (int k = 0; k < m; ++k) d_z_seq[0][k] += d_z_buf[k];
                    break;
                }
                case MODEL_RNN: {
                    std::vector<std::vector<double> > d_s_seq(T, std::vector<double>(m, 0.0));
                    for (int k = 0; k < m; ++k) d_s_seq[T-1][k] = d_feat[k];
                    std::vector<double> s_init(m, 0.0);
                    std::vector<double> d_s_in_buf(m), d_z_buf(m);
                    for (int t = T-1; t > 0; --t) {
                        kop.backward_step(&s_seq[t-1][0], &z_seq[t][0], &d_s_seq[t][0],
                                          &d_s_in_buf[0], &d_z_buf[0]);
                        for (int k = 0; k < m; ++k) {
                            d_s_seq[t-1][k] += d_s_in_buf[k];
                            d_z_seq[t][k] += d_z_buf[k];
                        }
                    }
                    kop.backward_step(&s_init[0], &z_seq[0][0], &d_s_seq[0][0],
                                      &d_s_in_buf[0], &d_z_buf[0]);
                    for (int k = 0; k < m; ++k) d_z_seq[0][k] += d_z_buf[k];
                    break;
                }
                case MODEL_ATTN: {
                    const double tau = 0.2;
                    const int Tq = (int)attn_w.size();
                    const std::vector<double>& q = z_seq[T-1];
                    double qn2 = 0.0;
                    for (int k = 0; k < m; ++k) qn2 += q[k] * q[k];
                    double qn = std::sqrt(qn2) + 1e-8;
                    std::vector<double> qhat(m);
                    for (int k = 0; k < m; ++k) qhat[k] = q[k] / qn;
                    std::vector<double> d_alpha(Tq, 0.0);
                    for (int t = 0; t < Tq; ++t) {
                        double s = 0.0;
                        for (int k = 0; k < m; ++k) s += z_seq[t][k] * d_feat[k];
                        d_alpha[t] = s;
                        for (int k = 0; k < m; ++k) d_z_seq[t][k] += attn_w[t] * d_feat[k];
                    }
                    double sum_ada = 0.0;
                    for (int t = 0; t < Tq; ++t) sum_ada += attn_w[t] * d_alpha[t];
                    std::vector<double> d_sim(Tq, 0.0);
                    for (int t = 0; t < Tq; ++t) d_sim[t] = attn_w[t] * (d_alpha[t] - sum_ada);
                    std::vector<double> d_qhat(m, 0.0);
                    for (int t = 0; t < Tq; ++t) {
                        double zn2 = 0.0;
                        for (int k = 0; k < m; ++k) zn2 += z_seq[t][k] * z_seq[t][k];
                        double zn = std::sqrt(zn2) + 1e-8;
                        std::vector<double> khat(m);
                        for (int k = 0; k < m; ++k) khat[k] = z_seq[t][k] / zn;
                        std::vector<double> d_khat(m, 0.0);
                        for (int k = 0; k < m; ++k) {
                            d_qhat[k]   += (d_sim[t] / tau) * khat[k];
                            d_khat[k]    = (d_sim[t] / tau) * qhat[k];
                        }
                        double dot_zd = 0.0;
                        for (int k = 0; k < m; ++k) dot_zd += khat[k] * d_khat[k];
                        for (int k = 0; k < m; ++k)
                            d_z_seq[t][k] += (d_khat[k] - khat[k] * dot_zd) / zn;
                    }
                    double dot_qd = 0.0;
                    for (int k = 0; k < m; ++k) dot_qd += qhat[k] * d_qhat[k];
                    for (int k = 0; k < m; ++k)
                        d_z_seq[T-1][k] += (d_qhat[k] - qhat[k] * dot_qd) / qn;
                    break;
                }
            }

            // Per-token encoder backward.
            for (int t = 0; t < T; ++t) enc.backward(x[t], &d_z_seq[t][0], &emS[t][0]);
        }

        total_loss /= B;
        if (model == MODEL_EALRMN && write_count > 0)
            total_loss += kWritePenalty * (write_loss_acc / write_count);

        double invB = 1.0 / double(B);
        for (int i = 0; i < (int)enc.dEmb.size(); ++i) enc.dEmb[i] *= invB;
        for (int i = 0; i < (int)enc.dW.size();   ++i) enc.dW[i]   *= invB;
        for (int i = 0; i < (int)enc.db.size();   ++i) enc.db[i]   *= invB;
        if (model == MODEL_EALRMN || model == MODEL_RNN) {
            for (int i = 0; i < (int)kop.dK.size(); ++i) kop.dK[i] *= invB;
            for (int i = 0; i < (int)kop.dB.size(); ++i) kop.dB[i] *= invB;
        }
        if (model == MODEL_EALRMN) {
            for (int i = 0; i < (int)dWg.size(); ++i) dWg[i] *= invB;
            dbg *= invB;
        }
        for (int i = 0; i < (int)readout.dW.size(); ++i) readout.dW[i] *= invB;
        for (int i = 0; i < (int)readout.db.size(); ++i) readout.db[i] *= invB;

        clip_l2(&enc.dEmb[0], (int)enc.dEmb.size(), kGradClip);
        clip_l2(&enc.dW[0],   (int)enc.dW.size(),   kGradClip);
        clip_l2(&enc.db[0],   (int)enc.db.size(),   kGradClip);
        if (model == MODEL_EALRMN || model == MODEL_RNN) {
            clip_l2(&kop.dK[0], (int)kop.dK.size(), kGradClip);
            clip_l2(&kop.dB[0], (int)kop.dB.size(), kGradClip);
        }
        if (model == MODEL_EALRMN) clip_l2(&dWg[0], (int)dWg.size(), kGradClip);
        clip_l2(&readout.dW[0], (int)readout.dW.size(), kGradClip);
        clip_l2(&readout.db[0], (int)readout.db.size(), kGradClip);

        enc.sgd_update(lr);
        if (model == MODEL_EALRMN || model == MODEL_RNN) kop.sgd_update(lr * 0.3);
        if (model == MODEL_EALRMN) {
            for (int i = 0; i < (int)Wg.size(); ++i) Wg[i] -= lr * dWg[i];
            bg -= lr * dbg;
        }
        readout.sgd_update(lr);
        return total_loss;
    }

    // Evaluate: returns loss, accuracy, average gate, and gate-by-token-type breakdown.
    void eval(int n_streams, double& loss_out, double& acc_out, double& gate_out,
              double& gate_marker_out, double& gate_filler_out) const {
        std::vector<int> x; int label;
        double total_loss = 0.0; int correct = 0;
        double gate_total = 0.0; int gate_steps = 0;
        double gate_marker = 0.0; int marker_count = 0;
        double gate_filler = 0.0; int filler_count = 0;
        std::vector<double> e_cache(kEmbDim);
        for (int n = 0; n < n_streams; ++n) {
            task.generate(x, label);
            int T = task.T;
            std::vector<std::vector<double> > z_seq(T, std::vector<double>(m, 0.0));
            for (int t = 0; t < T; ++t) enc.forward(x[t], &z_seq[t][0], &e_cache[0]);
            std::vector<std::vector<double> > s_seq;
            std::vector<std::vector<std::vector<double> > > M_seq;
            std::vector<double> gate_seq;
            bool use_mem = (model == MODEL_EALRMN);
            if (model == MODEL_EALRMN || model == MODEL_RNN)
                run_recurrence(z_seq, s_seq, M_seq, gate_seq, use_mem);
            for (int t = 0; t < (int)gate_seq.size(); ++t) {
                gate_total += gate_seq[t]; gate_steps++;
                bool is_marker = (x[t] >= kKeyMarker && x[t] <= kQueryMarker)
                              || (x[t] >= kKeyStart && x[t] < kKeyStart + kNumKeys)
                              || (x[t] >= kValStart && x[t] < kValStart + kNumVals);
                if (is_marker) { gate_marker += gate_seq[t]; marker_count++; }
                else           { gate_filler += gate_seq[t]; filler_count++; }
            }
            std::vector<double> feat, attn_w;
            build_feat(z_seq, s_seq, M_seq, feat, (model == MODEL_ATTN) ? &attn_w : NULL);
            std::vector<double> logits(kNumVals), d_logits(kNumVals);
            readout.forward(&feat[0], &logits[0]);
            total_loss += readout.loss_and_grad(&logits[0], label, &d_logits[0]);
            if (readout.predict(&logits[0]) == label) correct++;
        }
        loss_out = total_loss / n_streams;
        acc_out  = double(correct) / n_streams;
        gate_out = (gate_steps > 0) ? gate_total / gate_steps : 0.0;
        gate_marker_out = (marker_count > 0) ? gate_marker / marker_count : 0.0;
        gate_filler_out = (filler_count > 0) ? gate_filler / filler_count : 0.0;
    }
};

int main(int argc, char** argv) {
    std::string model_str = "ealrmn";
    int seed = 42;
    int steps = 2000;
    int print_every = 200;
    int T = 64;
    double lr = 0.005;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"       && i+1 < argc) model_str = argv[++i];
        else if (a == "--seed"        && i+1 < argc) seed = std::atoi(argv[++i]);
        else if (a == "--steps"       && i+1 < argc) steps = std::atoi(argv[++i]);
        else if (a == "--lr"          && i+1 < argc) lr = std::atof(argv[++i]);
        else if (a == "--T"           && i+1 < argc) T = std::atoi(argv[++i]);
        else if (a == "--print-every" && i+1 < argc) print_every = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "Unknown arg %s\n", argv[i]); return 1; }
    }

    Trainer::Model mdl;
    if      (model_str == "ealrmn") mdl = Trainer::MODEL_EALRMN;
    else if (model_str == "rnn")    mdl = Trainer::MODEL_RNN;
    else if (model_str == "attn")   mdl = Trainer::MODEL_ATTN;
    else { std::fprintf(stderr, "--model must be 'ealrmn', 'rnn', or 'attn'\n"); return 1; }

    Trainer tr;
    tr.init(seed, T, mdl);
    tr.lr = lr;

    std::printf("# EALRMN Phase-0e  per-token needle-haystack model=%s seed=%d steps=%d T=%d lr=%.4f\n",
                model_str.c_str(), seed, steps, T, lr);
    std::printf("# Task: 2 KV pairs (8 tokens) + 2-token query in %d-token stream\n", T);
    if (mdl == Trainer::MODEL_EALRMN)
        std::printf("# step\ttrain_loss\theld_loss\tacc\tgate\tgate_marker\tgate_filler\n");
    else
        std::printf("# step\ttrain_loss\theld_loss\tacc\n");

    for (int step = 0; step < steps; ++step) {
        srand((unsigned)(seed * 31 + step * 7919 + 1));
        double tloss = tr.train_step(kBatch);
        if (step % print_every == 0 || step == steps - 1) {
            srand((unsigned)(seed + 1000 + step));
            double hloss, hacc, hgate, hgate_m, hgate_f;
            tr.eval(64, hloss, hacc, hgate, hgate_m, hgate_f);
            if (mdl == Trainer::MODEL_EALRMN)
                std::printf("%6d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                            step, tloss, hloss, hacc, hgate, hgate_m, hgate_f);
            else
                std::printf("%6d\t%.4f\t%.4f\t%.4f\n", step, tloss, hloss, hacc);
            std::fflush(stdout);
        }
    }
    return 0;
}
