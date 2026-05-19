// EALRMN Phase-1 GPU prototype — task data generators
// All tasks generate sequences of token ids of length T plus a single target label.
// CPU-side generation, then bulk H2D copy.
#ifndef EALRMN_TASKS_CUH
#define EALRMN_TASKS_CUH

#include "common.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <stdexcept>

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

// ===== A_5 word recognition (state-tracking expressivity, VESTA Claim N1) =====
// Two generators of A_5 ⊂ S_5:
//   g_0 = (1 2 3)   in 0-indexed array form [1, 2, 0, 3, 4]
//   g_1 = (3 4 5)   in 0-indexed array form [0, 1, 3, 4, 2]
// Both are 3-cycles, both even, and together they generate the full A_5.
//
// For each batch element b:
//   - Sample T tokens uniformly from {0, 1}
//   - Compose the corresponding permutations to get final product π in A_5
//   - Label = lex rank of π among the 60 even-parity permutations of {0..4}
//
// Random baseline on this task is 1/60 ≈ 0.0167 accuracy.

struct A5WordTask {
    int T;
    std::vector<int> a5_lookup;  // S_5 lex rank (0..119) -> A_5 rank (0..59) or -1 if odd

    A5WordTask(int T_ = 64) : T(T_) {
        build_a5_lookup();
    }
    int n_classes() const { return 60; }
    int vocab_size() const { return 2; }

    static void compose(const int* a, const int* b, int* out) {
        // out = a then b: apply a first, then b. out[i] = b[a[i]].
        for (int i = 0; i < 5; ++i) out[i] = b[a[i]];
    }
    static int parity(const int* p) {
        int inv = 0;
        for (int i = 0; i < 5; ++i)
            for (int j = i + 1; j < 5; ++j)
                if (p[i] > p[j]) inv++;
        return inv & 1;
    }
    static int lex_rank(const int* p) {
        int rank = 0;
        int fact[5] = {24, 6, 2, 1, 1};  // 4!, 3!, 2!, 1!, 0!
        bool used[5] = {false, false, false, false, false};
        for (int i = 0; i < 5; ++i) {
            int count_smaller = 0;
            for (int j = 0; j < p[i]; ++j) if (!used[j]) count_smaller++;
            rank += count_smaller * fact[i];
            used[p[i]] = true;
        }
        return rank;
    }
    void build_a5_lookup() {
        a5_lookup.assign(120, -1);
        int p[5] = {0, 1, 2, 3, 4};
        int idx = 0;
        do {
            if (parity(p) == 0) {
                a5_lookup[lex_rank(p)] = idx;
                idx++;
            }
        } while (std::next_permutation(p, p + 5));
    }

    void generate(std::vector<int>& ids, std::vector<int>& labels,
                  int B, HostRng& rng) const {
        const int g0[5] = {1, 2, 0, 3, 4};
        const int g1[5] = {0, 1, 3, 4, 2};
        ids.assign(B * T, 0);
        labels.assign(B, 0);
        for (int b = 0; b < B; ++b) {
            int p[5] = {0, 1, 2, 3, 4};
            int tmp[5];
            for (int t = 0; t < T; ++t) {
                int tok = rng.next_int(2);
                ids[b * T + t] = tok;
                const int* g = (tok == 0) ? g0 : g1;
                compose(p, g, tmp);
                for (int i = 0; i < 5; ++i) p[i] = tmp[i];
            }
            int lr = lex_rank(p);
            int a5 = a5_lookup[lr];
            // a5 should always be >= 0 since A_5 is closed under composition.
            labels[b] = (a5 >= 0) ? a5 : 0;
        }
    }
};

// ===== Pretokenized real corpus =====
// Reads a .tok.bin file (Glades trainer format):
//   24-byte header: magic(4)=0x544F4B42 | version(4)=1 | vocab_size(4) | reserved(4)=0 | token_count(8)
//   payload: uint16_le[token_count]
//
// For each batch element we sample a random offset into the corpus and read
// T consecutive tokens. The LM task predicts the next token at each position,
// so caller should set up labels_lm via k_build_lm_labels as usual; the per-batch
// `labels` vector (single-label) is set to a dummy value here.

struct CorpusTask {
    int T;
    int V_vocab;
    std::vector<uint16_t> tokens;  // entire file in memory, payload only
    explicit CorpusTask(const std::string& path, int T_) : T(T_), V_vocab(0) {
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "CorpusTask: could not open %s\n", path.c_str());
            std::exit(1);
        }
        uint32_t magic = 0, version = 0, vsz = 0, reserved = 0;
        uint64_t tok_count = 0;
        std::fread(&magic, 4, 1, fp);
        std::fread(&version, 4, 1, fp);
        std::fread(&vsz, 4, 1, fp);
        std::fread(&reserved, 4, 1, fp);
        std::fread(&tok_count, 8, 1, fp);
        if (magic != 0x544F4B42u) {
            std::fprintf(stderr, "CorpusTask: bad magic 0x%08x in %s\n", magic, path.c_str());
            std::exit(1);
        }
        V_vocab = (int)vsz;
        tokens.resize((size_t)tok_count);
        size_t got = std::fread(tokens.data(), sizeof(uint16_t), (size_t)tok_count, fp);
        if (got != (size_t)tok_count) {
            std::fprintf(stderr, "CorpusTask: short read %zu < %llu in %s\n",
                         got, (unsigned long long)tok_count, path.c_str());
            std::exit(1);
        }
        std::fclose(fp);
        std::fprintf(stderr, "CorpusTask: loaded %llu tokens, vocab=%d from %s\n",
                     (unsigned long long)tok_count, V_vocab, path.c_str());
    }
    int n_classes() const { return V_vocab; }
    int vocab_size() const { return V_vocab; }

    void generate(std::vector<int>& ids, std::vector<int>& labels,
                  int B, HostRng& rng) const {
        ids.assign(B * T, 0);
        labels.assign(B, 0);
        size_t N = tokens.size();
        if (N < (size_t)T + 1) {
            std::fprintf(stderr, "CorpusTask: corpus has only %zu tokens < T+1=%d\n", N, T + 1);
            std::exit(1);
        }
        size_t max_off = N - (size_t)T;
        for (int b = 0; b < B; ++b) {
            // Uniform offset in [0, max_off]
            uint32_t r1 = (uint32_t)rng.next_int(1 << 30);
            uint32_t r2 = (uint32_t)rng.next_int(1 << 30);
            uint64_t r = ((uint64_t)r1 << 30) | (uint64_t)r2;
            size_t off = (size_t)(r % (uint64_t)(max_off + 1));
            for (int t = 0; t < T; ++t) {
                ids[b * T + t] = (int)tokens[off + (size_t)t];
            }
            // Single-label fallback: last-token's next (for back-compat with non-LM heads).
            labels[b] = (off + (size_t)T < N) ? (int)tokens[off + (size_t)T] : 0;
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
