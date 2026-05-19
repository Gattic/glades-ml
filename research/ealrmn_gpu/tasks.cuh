// EALRMN Phase-1 GPU prototype — task data generators
// All tasks generate sequences of token ids of length T plus a single target label.
// CPU-side generation, then bulk H2D copy.
#ifndef EALRMN_TASKS_CUH
#define EALRMN_TASKS_CUH

#include "common.cuh"

// ===== needle-in-haystack =====
// Vocab: [0, V):
//   0..(N_KEYS-1)         : marker tokens M_k
//   N_KEYS..(2*N_KEYS-1)  : value tokens V_k
//   2*N_KEYS..(3*N_KEYS-1): query tokens Q_k
//   3*N_KEYS..(V-1)       : filler tokens
// For a sequence:
//   - Pick a key k ∈ [0, N_KEYS).
//   - Choose position p ∈ [0, T-4) for the marker.
//   - Sequence is filler tokens except:
//       seq[p]   = marker M_k
//       seq[p+2] = value V_k
//       seq[T-1] = query Q_k
//   - Label = value index (or k).
//
// Classification target: predict the value token id (V_k). Number of classes = N_KEYS.

struct NeedleTask {
    int V_vocab;   // total vocab size
    int N_KEYS;    // number of keys
    int T;
    NeedleTask(int n_keys, int T_, int filler_size = 64)
        : N_KEYS(n_keys), T(T_) {
        V_vocab = 3 * N_KEYS + filler_size;
    }
    int n_classes() const { return N_KEYS; }
    int vocab_size() const { return V_vocab; }

    // Generate a batch into host arrays.
    void generate(std::vector<int>& ids, std::vector<int>& labels,
                  int B, HostRng& rng) const {
        ids.assign(B * T, 0);
        labels.assign(B, 0);
        int filler_lo = 3 * N_KEYS;
        int filler_hi = V_vocab;
        int filler_n = filler_hi - filler_lo;
        for (int b = 0; b < B; ++b) {
            int k = rng.next_int(N_KEYS);
            int p = rng.next_int(T - 4);  // marker position in [0, T-4)
            // Fill with fillers
            for (int t = 0; t < T; ++t) {
                ids[b * T + t] = filler_lo + rng.next_int(filler_n);
            }
            ids[b * T + p]     = k;                    // marker
            ids[b * T + p + 2] = N_KEYS + k;           // value
            ids[b * T + T - 1] = 2 * N_KEYS + k;       // query
            labels[b] = k;
        }
    }
};

// ===== hmm =====
// Random transition matrix sampled once per task instance.
// Observation = state with 20% probability of random replacement.
// Target: predict hidden state at position T-1.

struct HmmTask {
    int n_states;
    int T;
    float noise_p;
    HmmTask(int n_states_, int T_, float noise_p_ = 0.2f)
        : n_states(n_states_), T(T_), noise_p(noise_p_) {}
    int n_classes() const { return n_states; }
    int vocab_size() const { return n_states; }  // observation symbols same as states

    void generate(std::vector<int>& ids, std::vector<int>& labels,
                  int B, HostRng& rng) const {
        ids.assign(B * T, 0);
        labels.assign(B, 0);
        // Per-batch transition matrix
        for (int b = 0; b < B; ++b) {
            std::vector<std::vector<float>> trans(n_states, std::vector<float>(n_states, 0.0f));
            for (int s = 0; s < n_states; ++s) {
                float total = 0.0f;
                for (int s2 = 0; s2 < n_states; ++s2) {
                    float u = rng.next_uniform();
                    trans[s][s2] = u * u;  // bias toward sparser
                    total += trans[s][s2];
                }
                for (int s2 = 0; s2 < n_states; ++s2) trans[s][s2] /= total;
            }
            int state = rng.next_int(n_states);
            for (int t = 0; t < T; ++t) {
                int obs = state;
                if (rng.next_uniform() < noise_p) {
                    obs = rng.next_int(n_states);
                }
                ids[b * T + t] = obs;
                if (t == T - 1) labels[b] = state;
                // Sample next state from transition matrix
                float u = rng.next_uniform();
                float cum = 0.0f;
                int next_state = n_states - 1;
                for (int s2 = 0; s2 < n_states; ++s2) {
                    cum += trans[state][s2];
                    if (u < cum) { next_state = s2; break; }
                }
                state = next_state;
            }
        }
    }
};

// ===== syntheticlm =====
// Order-2 Markov chain over V states.
// Token x_t depends on (x_{t-1}, x_{t-2}). We give the model x_0..x_{T-1} and ask it to
// predict x_T (next token).
// Goal: predict the LAST observed token's next.

struct SynthLMTask {
    int V_vocab;
    int T;
    int order;
    SynthLMTask(int V_, int T_, int order_ = 2)
        : V_vocab(V_), T(T_), order(order_) {}
    int n_classes() const { return V_vocab; }
    int vocab_size() const { return V_vocab; }

    void generate(std::vector<int>& ids, std::vector<int>& labels,
                  int B, HostRng& rng) const {
        // Number of conditioning context combinations
        int n_ctx = 1;
        for (int o = 0; o < order; ++o) n_ctx *= V_vocab;
        if (n_ctx > 1 << 20) {
            // too big — fall back to order=1
            n_ctx = V_vocab;
        }
        ids.assign(B * T, 0);
        labels.assign(B, 0);
        for (int b = 0; b < B; ++b) {
            // Build per-batch CDF: trans[ctx, :] over V_vocab
            std::vector<std::vector<float>> trans(n_ctx, std::vector<float>(V_vocab, 0.0f));
            for (int c = 0; c < n_ctx; ++c) {
                float total = 0.0f;
                for (int v = 0; v < V_vocab; ++v) {
                    float u = rng.next_uniform();
                    trans[c][v] = u * u * u;  // sparser
                    total += trans[c][v];
                }
                for (int v = 0; v < V_vocab; ++v) trans[c][v] /= total;
            }
            // Initialize first `order` tokens
            int hist[8] = {0,0,0,0,0,0,0,0};  // up to order=8
            for (int o = 0; o < order; ++o) {
                hist[o] = rng.next_int(V_vocab);
                ids[b * T + o] = hist[o];
            }
            int ord_used = (order <= 8) ? order : 1;
            for (int t = order; t < T; ++t) {
                int ctx = 0;
                for (int o = 0; o < ord_used; ++o) ctx = ctx * V_vocab + hist[o];
                ctx %= n_ctx;
                float u = rng.next_uniform();
                float cum = 0.0f;
                int next = V_vocab - 1;
                for (int v = 0; v < V_vocab; ++v) {
                    cum += trans[ctx][v];
                    if (u < cum) { next = v; break; }
                }
                ids[b * T + t] = next;
                // Shift history
                for (int o = 0; o < ord_used - 1; ++o) hist[o] = hist[o + 1];
                hist[ord_used - 1] = next;
            }
            // Label: predict next token (x_T) from same conditional
            {
                int ctx = 0;
                for (int o = 0; o < ord_used; ++o) ctx = ctx * V_vocab + hist[o];
                ctx %= n_ctx;
                float u = rng.next_uniform();
                float cum = 0.0f;
                int next = V_vocab - 1;
                for (int v = 0; v < V_vocab; ++v) {
                    cum += trans[ctx][v];
                    if (u < cum) { next = v; break; }
                }
                labels[b] = next;
            }
        }
    }
};

// ===== Generic batch holder =====
struct BatchData {
    Tensor ids_d;       // (B, T) int but allocated as float — actually we need int allocator
    Tensor labels_d;
    int B = 0, T = 0;
    void alloc(int B_, int T_) {
        B = B_; T = T_;
        // We need int storage. Allocate via raw cudaMalloc.
    }
};

// Helpers for int tensors.
inline int* make_int_tensor(int n) {
    int* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    return d;
}

inline void copy_ints_h2d(int* d, const std::vector<int>& h) {
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
}

#endif // EALRMN_TASKS_CUH
