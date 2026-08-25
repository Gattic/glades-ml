// EALRMN-v1 Phase-0d prototype — needle-in-haystack retrieval test.
//
// Tests Claim 3 from EALRMN_DESIGN.md §8: bounded associative memory of capacity
// K_mem matches an unbounded-context attention baseline on retrieval tasks where
// the relevant-information rank ρ_rel ≤ K_mem.
//
// Task structure (one stream):
//   - N patches, each 4 tokens; total stream length T = N · 4.
//   - 2 distinct (key, value) pairs inserted at random non-overlapping patch
//     positions in patches [0, N-2). A K-V patch is the 4-token sequence
//     [KEY_MARKER, k_id, VAL_MARKER, v_id].
//   - Last patch (N-1) is the query patch: [QUERY_MARKER, k_id_query, 0, 0].
//   - All other patches are filler (uniform random from [0, kFillerMax)).
//   - Label: the v_id paired with k_id_query in the inserted KV pairs.
//
// Three trained models, all sharing the encoder and final readout head:
//   (1) EALRMN  — encoder + Koopman recurrence + 4-slot spectral memory; readout
//                  from (s_N, M_N) concatenated (m + kMemSlots·m = 5m dims).
//   (2) RNN     — encoder + Koopman recurrence; readout from s_N only (m dims).
//   (3) ATTN    — encoder; final readout uses cosine-similarity attention from
//                  the query patch's z onto all earlier z's; readout from the
//                  attended sum.
//
// Build:
//   g++ -std=c++98 -O2 -Wall -Wextra research/ealrmn_phase0d_needle.cpp \
//       -o research/ealrmn_phase0d_needle
// Run:
//   ./research/ealrmn_phase0d_needle --model ealrmn --seed 42 --steps 1500 --N 16
//   ./research/ealrmn_phase0d_needle --model rnn    --seed 42 --steps 1500 --N 16
//   ./research/ealrmn_phase0d_needle --model attn   --seed 42 --steps 1500 --N 16
//
// Output: per-logging-step tab-separated row: step, train_loss, held_loss, acc.
// acc is retrieval accuracy on held-out streams. Random baseline = 1/kNumVals = 0.25.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

// =============== Task constants ===============
static const int kPatchLen     = 4;
static const int kFillerMax    = 16;     // tokens 0..15 are filler
static const int kKeyMarker    = 16;
static const int kValMarker    = 17;
static const int kQueryMarker  = 18;
static const int kKeyStart     = 19;     // key ids: 19..19+kNumKeys-1
static const int kNumKeys      = 4;
static const int kValStart     = 23;     // value ids: 23..23+kNumVals-1
static const int kNumVals      = 4;
static const int kVocab        = kValStart + kNumVals;   // = 27

// =============== Architecture constants ===============
static const int kEmbDim       = 16;
static const int kM            = 32;
static const int kBatch        = 16;
static const int kMemSlots     = 4;
static const double kMemDecays[4] = { 0.50, 0.80, 0.95, 0.99 };
static const double kGradClip  = 10.0;
// Learned write gate g_i = σ(W_g · s_i + b_g) controls per-step EMA update.
// g_i ≈ 0 freezes memory; g_i ≈ 1 updates at the slot's natural rate.
// Write penalty kWritePenalty · mean(g) added to the loss — implements design memo's
// M6 (memory-write regularization). Strong penalty + gradient-on-gate forces the
// model to write selectively to memory rather than indiscriminately accumulating.
static const double kWritePenalty = 0.002;  // weak (gate must learn WHEN to write before penalty refines it)

// =============== Random helpers ===============
static double rand_unif() {
    return (double(rand()) + 1.0) / (double(RAND_MAX) + 2.0);
}
static double rand_gauss() {
    double u1 = rand_unif(), u2 = rand_unif();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
}
static int rand_int(int n) { return std::rand() % n; }

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

// =============== Needle-in-haystack task generator ===============
// Generates a stream of length N·kPatchLen with 2 KV pairs and 1 query.
// Label is the value paired with the queried key.
struct NeedleTask {
    int N;          // total number of patches per stream
    NeedleTask() : N(16) {}

    // Generate one stream. label is the queried value's index in [0, kNumVals).
    void generate(std::vector<int>& x, int& label) const {
        int T = N * kPatchLen;
        x.assign(T, 0);
        // Fill with random filler tokens.
        for (int i = 0; i < T; ++i) x[i] = rand_int(kFillerMax);

        // Pick 2 distinct patch positions from [0, N-1) — last patch is reserved for query.
        int p1 = rand_int(N - 1);
        int p2;
        do { p2 = rand_int(N - 1); } while (p2 == p1);

        // Pick 2 distinct keys and 2 random values.
        int keys[2], vals[2];
        keys[0] = rand_int(kNumKeys);
        do { keys[1] = rand_int(kNumKeys); } while (keys[1] == keys[0]);
        vals[0] = rand_int(kNumVals);
        vals[1] = rand_int(kNumVals);

        // Place KV patches.
        for (int k = 0; k < 2; ++k) {
            int pos = (k == 0 ? p1 : p2) * kPatchLen;
            x[pos]     = kKeyMarker;
            x[pos + 1] = kKeyStart + keys[k];
            x[pos + 2] = kValMarker;
            x[pos + 3] = kValStart + vals[k];
        }

        // Query patch at position N-1.
        int q_choice = rand_int(2);   // which of the two inserted keys to query
        int q_pos = (N - 1) * kPatchLen;
        x[q_pos]     = kQueryMarker;
        x[q_pos + 1] = kKeyStart + keys[q_choice];
        // Pad remaining 2 tokens with filler so the query patch doesn't look constant.
        x[q_pos + 2] = rand_int(kFillerMax);
        x[q_pos + 3] = rand_int(kFillerMax);

        label = vals[q_choice];
    }
};

// =============== Encoder ===============
// z_i = W · mean(Emb[tok_l]) + b
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

// =============== Readout head: linear from feat to kNumVals classes ===============
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
    // Learned write-gate parameters (EALRMN only).
    std::vector<double> Wg, dWg;
    double bg, dbg;
    enum Model { MODEL_EALRMN, MODEL_RNN, MODEL_ATTN };
    Model model;
    int m;
    double lr;
    int feat_dim;       // dim of input feature to readout head

    Trainer() : bg(0.0), dbg(0.0), model(MODEL_EALRMN), m(kM), lr(0.005), feat_dim(0) {}

    void init(int seed, int N_patches, Model mdl) {
        model = mdl;
        task.N = N_patches;
        enc.init(kVocab, kEmbDim, kM, (unsigned)(seed + 100));
        kop.init(kM, (unsigned)(seed + 200));
        // Init learned write-gate (random small Wg, bg=0 ⇒ gate ~0.5 at start).
        Wg.assign(kM, 0.0);
        dWg.assign(kM, 0.0);
        bg = 0.0; dbg = 0.0;
        srand((unsigned)(seed + 400));
        for (int i = 0; i < kM; ++i) Wg[i] = rand_gauss() * 0.1 / std::sqrt((double)kM);
        // Feature dim depends on model.
        switch (model) {
            case MODEL_EALRMN: feat_dim = kM + kMemSlots * kM; break;   // (s_N, M_N)
            case MODEL_RNN:    feat_dim = kM; break;                    // s_N
            case MODEL_ATTN:   feat_dim = kM; break;                    // attended sum
        }
        readout.init(feat_dim, kNumVals, (unsigned)(seed + 300));
        srand((unsigned)(seed + 999));
    }

    // Compute s_seq (Koopman recurrence), M_seq (memory), gate_seq (per-step gates).
    // For EALRMN model: memory uses gated EMA: M_i[sl] = (1 − g_i(1−λ_sl)) M_{i−1}[sl] + g_i(1−λ_sl) z_i.
    // For RNN model: M_seq is empty.
    void run_recurrence(const std::vector<std::vector<double> >& z_seq,
                        std::vector<std::vector<double> >& s_seq,
                        std::vector<std::vector<std::vector<double> > >& M_seq,
                        std::vector<double>& gate_seq,
                        bool use_memory) const {
        int N = z_seq.size();
        s_seq.assign(N, std::vector<double>(m, 0.0));
        std::vector<double> s_init(m, 0.0);
        kop.step(&s_init[0], &z_seq[0][0], &s_seq[0][0]);
        for (int i = 1; i < N; ++i)
            kop.step(&s_seq[i-1][0], &z_seq[i][0], &s_seq[i][0]);
        gate_seq.assign(use_memory ? N : 0, 0.0);
        if (use_memory) {
            M_seq.assign(N, std::vector<std::vector<double> >(kMemSlots, std::vector<double>(m, 0.0)));
            for (int i = 0; i < N; ++i) {
                // Compute gate from z_i (local patch content) — not s_i (integrated state).
                // The decision "is this patch a KV pair vs filler" is a function of the
                // patch tokens themselves; s_i would dilute it with all past distractors.
                double pre = bg;
                for (int k = 0; k < m; ++k) pre += Wg[k] * z_seq[i][k];
                double g = 1.0 / (1.0 + std::exp(-pre));
                gate_seq[i] = g;
                for (int sl = 0; sl < kMemSlots; ++sl) {
                    double lambda = kMemDecays[sl];
                    double write_factor = g * (1.0 - lambda);
                    double carry_factor = 1.0 - write_factor;
                    if (i == 0) {
                        for (int k = 0; k < m; ++k) M_seq[0][sl][k] = write_factor * z_seq[0][k];
                    } else {
                        for (int k = 0; k < m; ++k)
                            M_seq[i][sl][k] = carry_factor * M_seq[i-1][sl][k] + write_factor * z_seq[i][k];
                    }
                }
            }
        }
    }

    // Build the readout feature for one stream from its z_seq, s_seq, M_seq.
    // For ATTN model: query = z_seq[N-1]; attn over z_seq[0..N-2]; feat = weighted sum.
    void build_feat(const std::vector<std::vector<double> >& z_seq,
                    const std::vector<std::vector<double> >& s_seq,
                    const std::vector<std::vector<std::vector<double> > >& M_seq,
                    std::vector<double>& feat_out,
                    std::vector<double>* attn_weights_out  /* may be NULL */) const {
        int N = z_seq.size();
        feat_out.assign(feat_dim, 0.0);
        switch (model) {
            case MODEL_EALRMN: {
                // feat = concat(s_N-1, M_N-1[0], M_N-1[1], M_N-1[2], M_N-1[3])
                for (int k = 0; k < m; ++k) feat_out[k] = s_seq[N-1][k];
                int idx = m;
                for (int sl = 0; sl < kMemSlots; ++sl)
                    for (int k = 0; k < m; ++k) feat_out[idx++] = M_seq[N-1][sl][k];
                break;
            }
            case MODEL_RNN: {
                for (int k = 0; k < m; ++k) feat_out[k] = s_seq[N-1][k];
                break;
            }
            case MODEL_ATTN: {
                // Cosine attention from query z_{N-1} to z_{0..N-2}.
                const std::vector<double>& q = z_seq[N-1];
                double qn = 0.0;
                for (int k = 0; k < m; ++k) qn += q[k] * q[k];
                qn = std::sqrt(qn) + 1e-8;
                std::vector<double> sims(N-1, 0.0);
                double max_s = -1e18;
                for (int i = 0; i < N-1; ++i) {
                    double zn = 0.0, dot = 0.0;
                    for (int k = 0; k < m; ++k) { zn += z_seq[i][k] * z_seq[i][k]; dot += q[k] * z_seq[i][k]; }
                    zn = std::sqrt(zn) + 1e-8;
                    sims[i] = (dot / (qn * zn)) / 0.2;   // τ=0.2 for attention sharpness
                    if (sims[i] > max_s) max_s = sims[i];
                }
                double sum_e = 0.0;
                std::vector<double> alpha(N-1, 0.0);
                for (int i = 0; i < N-1; ++i) { alpha[i] = std::exp(sims[i] - max_s); sum_e += alpha[i]; }
                for (int i = 0; i < N-1; ++i) alpha[i] /= sum_e;
                // feat = sum_i alpha_i * z_seq[i]
                for (int k = 0; k < m; ++k) {
                    double acc = 0.0;
                    for (int i = 0; i < N-1; ++i) acc += alpha[i] * z_seq[i][k];
                    feat_out[k] = acc;
                }
                if (attn_weights_out != NULL) *attn_weights_out = alpha;
                break;
            }
        }
    }

    // One training step: forward + backward + SGD on B streams.
    // Returns average loss.
    double train_step(int B) {
        enc.zero_grad(); kop.zero_grad(); readout.zero_grad();
        for (int i = 0; i < (int)Wg.size(); ++i) dWg[i] = 0.0;
        dbg = 0.0;
        double total_loss = 0.0;
        double write_loss_acc = 0.0;
        int    write_count    = 0;
        std::vector<int> x; int label;
        for (int b = 0; b < B; ++b) {
            task.generate(x, label);
            int N = task.N;
            // Forward: encode each patch.
            std::vector<std::vector<double> > z_seq(N, std::vector<double>(m, 0.0));
            std::vector<std::vector<double> > emS(N, std::vector<double>(kEmbDim, 0.0));
            for (int i = 0; i < N; ++i)
                enc.forward(&x[i * kPatchLen], kPatchLen, &z_seq[i][0], &emS[i][0]);

            std::vector<std::vector<double> > s_seq;
            std::vector<std::vector<std::vector<double> > > M_seq;
            std::vector<double> gate_seq;
            bool use_memory = (model == MODEL_EALRMN);
            if (model == MODEL_EALRMN || model == MODEL_RNN) {
                run_recurrence(z_seq, s_seq, M_seq, gate_seq, use_memory);
            }

            std::vector<double> feat;
            std::vector<double> attn_w;   // attention weights (for ATTN backward)
            build_feat(z_seq, s_seq, M_seq, feat, (model == MODEL_ATTN) ? &attn_w : NULL);

            // Readout forward + loss.
            std::vector<double> logits(kNumVals), d_logits(kNumVals);
            readout.forward(&feat[0], &logits[0]);
            double l = readout.loss_and_grad(&logits[0], label, &d_logits[0]);
            total_loss += l;

            // Backward through readout.
            std::vector<double> d_feat(feat_dim, 0.0);
            readout.backward(&feat[0], &d_logits[0], &d_feat[0]);

            // Now backprop through the feature builder + recurrence + encoder.
            std::vector<std::vector<double> > d_z_seq(N, std::vector<double>(m, 0.0));
            switch (model) {
                case MODEL_EALRMN: {
                    // d_feat = [d_s_N | d_M_N[0] | d_M_N[1] | d_M_N[2] | d_M_N[3]]
                    std::vector<std::vector<double> > d_s_seq(N, std::vector<double>(m, 0.0));
                    for (int k = 0; k < m; ++k) d_s_seq[N-1][k] = d_feat[k];
                    // Gated-memory backward: at each i,
                    //   M_i[sl] = (1 − u_i_sl) M_{i−1}[sl] + u_i_sl z_i,    u_i_sl = g_i (1−λ_sl)
                    // d M_{i−1}[sl] = (1 − u_i_sl) · d M_i[sl]
                    // d z_i        += Σ_sl u_i_sl · d M_i[sl]
                    // d u_i_sl     = (z_i − M_{i−1}[sl]) · d M_i[sl]
                    // d g_i        = Σ_sl (1 − λ_sl) · d u_i_sl
                    // d g_i then routes back to d_pre_i = d g_i · g_i (1 − g_i)
                    //   → d_W_g, d_b_g, d s_i (since gate uses s_i).
                    // Initialize d_M[sl] from d_feat then walk back through chain.
                    std::vector<std::vector<double> > d_M(kMemSlots, std::vector<double>(m, 0.0));
                    for (int sl = 0; sl < kMemSlots; ++sl) {
                        int idx = m + sl * m;
                        for (int k = 0; k < m; ++k) d_M[sl][k] = d_feat[idx + k];
                    }
                    // Walk back in time.
                    for (int i = N - 1; i >= 0; --i) {
                        double g = gate_seq[i];
                        double d_g = 0.0;
                        // M_prev[sl]: either M_seq[i-1][sl] (if i>0) or all-zeros (if i==0).
                        for (int sl = 0; sl < kMemSlots; ++sl) {
                            double u_i_sl = g * (1.0 - kMemDecays[sl]);
                            double one_minus_u = 1.0 - u_i_sl;
                            for (int k = 0; k < m; ++k) {
                                double M_prev_k = (i > 0) ? M_seq[i-1][sl][k] : 0.0;
                                // Accumulate gradients.
                                d_z_seq[i][k] += u_i_sl * d_M[sl][k];
                                d_g          += (1.0 - kMemDecays[sl]) * (z_seq[i][k] - M_prev_k) * d_M[sl][k];
                            }
                            // Now propagate d_M[sl] to d_M_{i-1}[sl]: d_M[sl] ← one_minus_u · d_M[sl]
                            // (done in-place after we've consumed it for this step)
                            for (int k = 0; k < m; ++k) d_M[sl][k] *= one_minus_u;
                        }
                        // Add write-penalty gradient on g_i: penalty contributes (kWritePenalty / N) to d_g per stream.
                        d_g += kWritePenalty;
                        // Sigmoid bwd: d_pre = d_g · g · (1 − g)
                        double d_pre = d_g * g * (1.0 - g);
                        // Gate uses z_i now: d_W_g += d_pre · z_i;  d_b_g += d_pre; d_z_i += d_pre · W_g
                        for (int k = 0; k < m; ++k) {
                            dWg[k]      += d_pre * z_seq[i][k];
                            d_z_seq[i][k] += d_pre * Wg[k];
                        }
                        dbg += d_pre;
                    }
                    // Write-loss accounting for reporting.
                    for (int i = 0; i < N; ++i) write_loss_acc += gate_seq[i];
                    write_count += N;
                    // Now backward through Koopman recurrence using d_s_seq.
                    std::vector<double> s_init(m, 0.0);
                    std::vector<double> d_s_in_buf(m), d_z_buf(m);
                    for (int i = N-1; i > 0; --i) {
                        kop.backward_step(&s_seq[i-1][0], &z_seq[i][0], &d_s_seq[i][0],
                                          &d_s_in_buf[0], &d_z_buf[0]);
                        for (int k = 0; k < m; ++k) {
                            d_s_seq[i-1][k] += d_s_in_buf[k];
                            d_z_seq[i][k] += d_z_buf[k];
                        }
                    }
                    kop.backward_step(&s_init[0], &z_seq[0][0], &d_s_seq[0][0],
                                      &d_s_in_buf[0], &d_z_buf[0]);
                    for (int k = 0; k < m; ++k) d_z_seq[0][k] += d_z_buf[k];
                    break;
                }
                case MODEL_RNN: {
                    std::vector<std::vector<double> > d_s_seq(N, std::vector<double>(m, 0.0));
                    for (int k = 0; k < m; ++k) d_s_seq[N-1][k] = d_feat[k];
                    std::vector<double> s_init(m, 0.0);
                    std::vector<double> d_s_in_buf(m), d_z_buf(m);
                    for (int i = N-1; i > 0; --i) {
                        kop.backward_step(&s_seq[i-1][0], &z_seq[i][0], &d_s_seq[i][0],
                                          &d_s_in_buf[0], &d_z_buf[0]);
                        for (int k = 0; k < m; ++k) {
                            d_s_seq[i-1][k] += d_s_in_buf[k];
                            d_z_seq[i][k] += d_z_buf[k];
                        }
                    }
                    kop.backward_step(&s_init[0], &z_seq[0][0], &d_s_seq[0][0],
                                      &d_s_in_buf[0], &d_z_buf[0]);
                    for (int k = 0; k < m; ++k) d_z_seq[0][k] += d_z_buf[k];
                    break;
                }
                case MODEL_ATTN: {
                    // feat = sum_i alpha_i * z_i; backward through both the per-key
                    // multiplication AND the softmax over cosine similarities.
                    //
                    // Forward (recap):
                    //   q = z_{N-1};  qhat = q / ||q||
                    //   khat_i = z_i / ||z_i||
                    //   sim_i = (qhat · khat_i) / τ
                    //   alpha = softmax(sim)
                    //   feat = Σ_i alpha_i · z_i
                    //
                    // Gradients:
                    //   d alpha_i = z_i · d_feat   (a scalar)
                    //   d sim_j  = α_j Σ_i α_i (d alpha_i) − α_j d alpha_j  (softmax bwd, but with the −sign convention)
                    //           = α_j (d alpha_j − Σ_i α_i d alpha_i)         (clean form)
                    //   Through cosine: d sim_j / d qhat = khat_j / τ
                    //                   d sim_j / d khat_j = qhat / τ
                    //   Through L2 normalization (chain rule from existing latent_nce code).
                    //   Per-key d z_i has two contributions: from feat's alpha_i · z_i term AND
                    //   from sim_i's cosine.
                    const double tau = 0.2;
                    const int Nq = (int)attn_w.size();  // == N-1
                    const std::vector<double>& q = z_seq[N-1];
                    double qn2 = 0.0;
                    for (int k = 0; k < m; ++k) qn2 += q[k] * q[k];
                    double qn = std::sqrt(qn2) + 1e-8;
                    std::vector<double> qhat(m);
                    for (int k = 0; k < m; ++k) qhat[k] = q[k] / qn;
                    // d alpha_i scalars and d feat / d z_i contribution.
                    std::vector<double> d_alpha(Nq, 0.0);
                    for (int i = 0; i < Nq; ++i) {
                        double s = 0.0;
                        for (int k = 0; k < m; ++k) s += z_seq[i][k] * d_feat[k];
                        d_alpha[i] = s;
                        for (int k = 0; k < m; ++k)
                            d_z_seq[i][k] += attn_w[i] * d_feat[k];   // value-path gradient
                    }
                    // d sim_j = α_j (d_alpha_j − Σ_i α_i d_alpha_i)
                    double sum_ada = 0.0;
                    for (int i = 0; i < Nq; ++i) sum_ada += attn_w[i] * d_alpha[i];
                    std::vector<double> d_sim(Nq, 0.0);
                    for (int j = 0; j < Nq; ++j)
                        d_sim[j] = attn_w[j] * (d_alpha[j] - sum_ada);
                    // d qhat = Σ_j d_sim_j · khat_j / τ;  d khat_i = d_sim_i · qhat / τ
                    std::vector<double> d_qhat(m, 0.0);
                    std::vector<std::vector<double> > d_khat(Nq, std::vector<double>(m, 0.0));
                    for (int i = 0; i < Nq; ++i) {
                        double zn2 = 0.0;
                        for (int k = 0; k < m; ++k) zn2 += z_seq[i][k] * z_seq[i][k];
                        double zn = std::sqrt(zn2) + 1e-8;
                        std::vector<double> khat(m);
                        for (int k = 0; k < m; ++k) khat[k] = z_seq[i][k] / zn;
                        for (int k = 0; k < m; ++k) {
                            d_qhat[k]      += (d_sim[i] / tau) * khat[k];
                            d_khat[i][k]    = (d_sim[i] / tau) * qhat[k];
                        }
                        // Backprop d_khat[i] through L2-normalization to d_z_seq[i].
                        double dot_zd = 0.0;
                        for (int k = 0; k < m; ++k) dot_zd += khat[k] * d_khat[i][k];
                        for (int k = 0; k < m; ++k)
                            d_z_seq[i][k] += (d_khat[i][k] - khat[k] * dot_zd) / zn;
                    }
                    // Backprop d_qhat to d_z_seq[N-1] via the same chain rule.
                    {
                        double dot_qd = 0.0;
                        for (int k = 0; k < m; ++k) dot_qd += qhat[k] * d_qhat[k];
                        for (int k = 0; k < m; ++k)
                            d_z_seq[N-1][k] += (d_qhat[k] - qhat[k] * dot_qd) / qn;
                    }
                    break;
                }
            }

            // Encoder backward.
            for (int i = 0; i < N; ++i) {
                enc.backward(&x[i * kPatchLen], kPatchLen, &d_z_seq[i][0], &emS[i][0]);
            }
        }
        total_loss /= B;
        if (model == MODEL_EALRMN && write_count > 0)
            total_loss += kWritePenalty * (write_loss_acc / write_count);

        // Average gradients.
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

        // Gradient clipping per group.
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

        // SGD updates.
        enc.sgd_update(lr);
        if (model == MODEL_EALRMN || model == MODEL_RNN) kop.sgd_update(lr * 0.3);
        if (model == MODEL_EALRMN) {
            for (int i = 0; i < (int)Wg.size(); ++i) Wg[i] -= lr * dWg[i];
            bg -= lr * dbg;
        }
        readout.sgd_update(lr);

        return total_loss;
    }

    // Evaluate on a held-out set: return (loss, accuracy, avg gate if EALRMN).
    void eval(int n_streams, double& loss_out, double& acc_out, double& gate_out) const {
        std::vector<int> x; int label;
        double total_loss = 0.0;
        int correct = 0;
        double gate_total = 0.0;
        int gate_steps = 0;
        std::vector<double> e_mean(kEmbDim);
        for (int n = 0; n < n_streams; ++n) {
            task.generate(x, label);
            int N = task.N;
            std::vector<std::vector<double> > z_seq(N, std::vector<double>(m, 0.0));
            for (int i = 0; i < N; ++i)
                enc.forward(&x[i * kPatchLen], kPatchLen, &z_seq[i][0], &e_mean[0]);
            std::vector<std::vector<double> > s_seq;
            std::vector<std::vector<std::vector<double> > > M_seq;
            std::vector<double> gate_seq;
            bool use_mem = (model == MODEL_EALRMN);
            if (model == MODEL_EALRMN || model == MODEL_RNN)
                run_recurrence(z_seq, s_seq, M_seq, gate_seq, use_mem);
            for (int i = 0; i < (int)gate_seq.size(); ++i) { gate_total += gate_seq[i]; gate_steps++; }
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
    }
};

// =============== Main ===============
int main(int argc, char** argv) {
    std::string model_str = "ealrmn";
    int seed = 42;
    int steps = 1500;
    int print_every = 100;
    int N = 16;        // patches per stream (default: 16 patches × 4 = 64 tokens)
    double lr = 0.005;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"       && i+1 < argc) model_str = argv[++i];
        else if (a == "--seed"        && i+1 < argc) seed = std::atoi(argv[++i]);
        else if (a == "--steps"       && i+1 < argc) steps = std::atoi(argv[++i]);
        else if (a == "--lr"          && i+1 < argc) lr = std::atof(argv[++i]);
        else if (a == "--N"           && i+1 < argc) N = std::atoi(argv[++i]);
        else if (a == "--print-every" && i+1 < argc) print_every = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "Unknown arg %s\n", argv[i]); return 1; }
    }

    Trainer::Model mdl;
    if      (model_str == "ealrmn") mdl = Trainer::MODEL_EALRMN;
    else if (model_str == "rnn")    mdl = Trainer::MODEL_RNN;
    else if (model_str == "attn")   mdl = Trainer::MODEL_ATTN;
    else { std::fprintf(stderr, "--model must be 'ealrmn', 'rnn', or 'attn'\n"); return 1; }

    Trainer tr;
    tr.init(seed, N, mdl);
    tr.lr = lr;

    std::printf("# EALRMN Phase-0d  needle-haystack model=%s seed=%d steps=%d N=%d lr=%.4f\n",
                model_str.c_str(), seed, steps, N, lr);
    std::printf("# Task: 2 K-V pairs in %d patches, query at last patch, 4-way value prediction\n", N);
    std::printf("# step\ttrain_loss\theld_loss\theld_acc\tgate\n");

    for (int step = 0; step < steps; ++step) {
        srand((unsigned)(seed * 31 + step * 7919 + 1));
        double tloss = tr.train_step(kBatch);
        if (step % print_every == 0 || step == steps - 1) {
            srand((unsigned)(seed + 1000 + step));
            double hloss, hacc, hgate;
            tr.eval(64, hloss, hacc, hgate);
            std::printf("%6d\t%.4f\t%.4f\t%.4f\t%.4f\n", step, tloss, hloss, hacc, hgate);
            std::fflush(stdout);
        }
    }
    return 0;
}
