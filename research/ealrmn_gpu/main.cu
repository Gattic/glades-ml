// EALRMN Phase-1 GPU prototype — main entry point
// Modes: --mode=train (default), --mode=gradcheck, --mode=smoke
//
// Args:
//   --model=<ealrmn_attmem|rnn|transformer_1l|transformer_2l>
//   --task=<needle|hmm|syntheticlm>
//   --m=256   --T=2048  --seed=0
//   --batch=8 --steps=2000 --lr=3e-4
//   --eval-batch=16 --eval-every=200
//   --grad-clip=1.0 --wd=0.01
//   --warmup=200 --print-every=100
//   --J=4 (memory slots for ealrmn)
//   --H=8 (heads for transformer)
//   --jsonl=path/to/out.jsonl   --tag=descriptor

#include "common.cuh"
#include "kernels.cuh"
#include "recurrence_kernels.cuh"
#include "model_ealrmn.cuh"
#include "model_rnn.cuh"
#include "model_transformer.cuh"
#include "model_grp_rnn.cuh"
#include "model_grp_stack.cuh"
#include "tasks.cuh"
#include "optimizer.cuh"

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

struct Args {
    std::string mode = "train";
    std::string model = "ealrmn_attmem";
    std::string task = "needle";
    int m = 128;
    int T = 256;
    unsigned long long seed = 0;
    int batch = 4;
    int steps = 1000;
    float lr = 3e-4f;
    int eval_batch = 16;
    int eval_every = 200;
    float grad_clip = 1.0f;
    float wd = 0.01f;
    int warmup = 200;
    int print_every = 50;
    int J = 4;
    int H = 4;
    std::string jsonl = "";
    std::string tag = "";
    int n_keys = 8;        // for needle: number of distinct keys
    int n_states = 8;      // for hmm
    int lm_vocab = 64;     // for syntheticlm (must be small for order-2 to fit)
    // Ablation knobs
    std::string init_K = "orthogonal";   // EALRMN K init: "orthogonal" | "xavier"
    std::string init_Wh = "orthogonal";  // RNN W_h init: "orthogonal" | "xavier"
    std::string readout = "attmem";      // EALRMN readout: "attmem" (default) | "s_only"
    int use_tanh = 1;                    // RNN: 1 = tanh recurrence (default), 0 = linear
    // GRP-RNN ablation knobs.
    int grp_K = -1;                      // K Givens per step (default = m/2 in train fn)
    int grp_stride = 3;                  // interlocking stride (1 = adjacent disjoint via flag)
    int grp_tanh_state = 0;              // --grp-tanh-state: apply tanh after linear update (strawman)
    int grp_fixed_angles = 0;            // --grp-fixed-angles: angles input-independent
    int grp_disjoint_planes = 0;         // --grp-disjoint-planes: (2k, 2k+1) plane layout
    int grp_layernorm = 0;               // --grp-layernorm: apply LN to s_t each step
    int lm_mode = 0;                     // --lm-mode: per-token next-token loss (for GRP-RNN)
    // GRP-stack (multi-layer) extras.
    int n_layers = 1;                    // --n-layers: number of GRP-stack layers
    int mlp_hidden = -1;                 // --mlp-hidden: MLP hidden dim (default 4*m)
    int linear_recurrence = 0;           // --linear-recurrence: stack baseline w/o Givens rotation
    // Curriculum learning: train first `curriculum_steps` steps at `curriculum_T`,
    // then switch to a.T for the remainder. Default 0 = disabled.
    int curriculum_T = 0;
    int curriculum_steps = 0;
    // Multi-phase curriculum schedule: comma-separated `T:steps` pairs.
    // E.g., --curriculum-schedule=8:4000,32:8000  -> 4000 steps at T=8, 8000 at T=32,
    // then remaining steps at a.T. Overrides curriculum_T/curriculum_steps when set.
    std::string curriculum_schedule;
    // Corpus path for --task=corpus (a .tok.bin file in the Glades trainer format).
    std::string corpus_path;
    std::string corpus_val_path;  // optional separate val/eval path
};

static bool parse_int(const std::string& v, int& out) {
    try { out = std::stoi(v); return true; } catch (...) { return false; }
}
static bool parse_ull(const std::string& v, unsigned long long& out) {
    try { out = std::stoull(v); return true; } catch (...) { return false; }
}
static bool parse_float(const std::string& v, float& out) {
    try { out = std::stof(v); return true; } catch (...) { return false; }
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        size_t eq = s.find('=');
        if (eq == std::string::npos) continue;
        std::string key = s.substr(0, eq);
        std::string val = s.substr(eq + 1);
        if (key == "--mode") a.mode = val;
        else if (key == "--model") a.model = val;
        else if (key == "--task") a.task = val;
        else if (key == "--m") parse_int(val, a.m);
        else if (key == "--T") parse_int(val, a.T);
        else if (key == "--seed") parse_ull(val, a.seed);
        else if (key == "--batch") parse_int(val, a.batch);
        else if (key == "--steps") parse_int(val, a.steps);
        else if (key == "--lr") parse_float(val, a.lr);
        else if (key == "--eval-batch") parse_int(val, a.eval_batch);
        else if (key == "--eval-every") parse_int(val, a.eval_every);
        else if (key == "--grad-clip") parse_float(val, a.grad_clip);
        else if (key == "--wd") parse_float(val, a.wd);
        else if (key == "--warmup") parse_int(val, a.warmup);
        else if (key == "--print-every") parse_int(val, a.print_every);
        else if (key == "--J") parse_int(val, a.J);
        else if (key == "--H") parse_int(val, a.H);
        else if (key == "--jsonl") a.jsonl = val;
        else if (key == "--tag") a.tag = val;
        else if (key == "--n-keys") parse_int(val, a.n_keys);
        else if (key == "--n-states") parse_int(val, a.n_states);
        else if (key == "--lm-vocab") parse_int(val, a.lm_vocab);
        else if (key == "--init-K") a.init_K = val;
        else if (key == "--init-Wh") a.init_Wh = val;
        else if (key == "--readout") a.readout = val;
        else if (key == "--use-tanh") parse_int(val, a.use_tanh);
        else if (key == "--grp-K") parse_int(val, a.grp_K);
        else if (key == "--grp-stride") parse_int(val, a.grp_stride);
        else if (key == "--grp-tanh-state") parse_int(val, a.grp_tanh_state);
        else if (key == "--grp-fixed-angles") parse_int(val, a.grp_fixed_angles);
        else if (key == "--grp-disjoint-planes") parse_int(val, a.grp_disjoint_planes);
        else if (key == "--grp-layernorm") parse_int(val, a.grp_layernorm);
        else if (key == "--lm-mode") parse_int(val, a.lm_mode);
        else if (key == "--curriculum-T") parse_int(val, a.curriculum_T);
        else if (key == "--curriculum-steps") parse_int(val, a.curriculum_steps);
        else if (key == "--curriculum-schedule") a.curriculum_schedule = val;
        else if (key == "--corpus") a.corpus_path = val;
        else if (key == "--corpus-val") a.corpus_val_path = val;
        else if (key == "--n-layers") parse_int(val, a.n_layers);
        else if (key == "--mlp-hidden") parse_int(val, a.mlp_hidden);
        else if (key == "--linear-recurrence") parse_int(val, a.linear_recurrence);
        else {
            std::cerr << "Unknown arg: " << key << "\n";
        }
    }
    return a;
}

// ===== Task adapter =====
// Unifies generation interface across tasks.
struct TaskWrapper {
    int kind = 0;  // 0=needle, 1=hmm, 2=syntheticlm, 3=a5_word, 4=corpus, 5=corpus_val
    NeedleTask needle{8, 256};
    HmmTask hmm{8, 256};
    SynthLMTask lm{64, 256};
    A5WordTask a5{64};
    CorpusTask* corpus = nullptr;       // owns train-corpus tokens
    CorpusTask* corpus_val = nullptr;   // optional separate val corpus (shares same vocab)
    int V = 0;
    int n_classes = 0;

    void init(const Args& a) {
        if (a.task == "needle") {
            kind = 0;
            needle = NeedleTask(a.n_keys, a.T);
            V = needle.vocab_size();
            n_classes = needle.n_classes();
        } else if (a.task == "hmm") {
            kind = 1;
            hmm = HmmTask(a.n_states, a.T);
            V = hmm.vocab_size();
            n_classes = hmm.n_classes();
        } else if (a.task == "syntheticlm") {
            kind = 2;
            lm = SynthLMTask(a.lm_vocab, a.T);
            V = lm.vocab_size();
            n_classes = lm.n_classes();
        } else if (a.task == "a5_word") {
            kind = 3;
            a5 = A5WordTask(a.T);
            V = a5.vocab_size();
            n_classes = a5.n_classes();
        } else if (a.task == "corpus") {
            kind = 4;
            if (a.corpus_path.empty()) {
                std::cerr << "Task 'corpus' requires --corpus=<path>\n";
                std::exit(1);
            }
            corpus = new CorpusTask(a.corpus_path, a.T);
            if (!a.corpus_val_path.empty()) {
                corpus_val = new CorpusTask(a.corpus_val_path, a.T);
            }
            V = corpus->vocab_size();
            n_classes = corpus->n_classes();
        } else {
            std::cerr << "Unknown task: " << a.task << "\n";
            std::exit(1);
        }
    }

    void generate(std::vector<int>& ids, std::vector<int>& labels,
                  int B, HostRng& rng) const {
        if (kind == 0) needle.generate(ids, labels, B, rng);
        else if (kind == 1) hmm.generate(ids, labels, B, rng);
        else if (kind == 2) lm.generate(ids, labels, B, rng);
        else if (kind == 3) a5.generate(ids, labels, B, rng);
        else                corpus->generate(ids, labels, B, rng);
    }
    // Generate a val-batch. Uses corpus_val if available, otherwise falls through to generate().
    void generate_val(std::vector<int>& ids, std::vector<int>& labels,
                      int B, HostRng& rng) const {
        if (kind == 4 && corpus_val) {
            corpus_val->generate(ids, labels, B, rng);
        } else {
            generate(ids, labels, B, rng);
        }
    }
};

// ===== Training utilities =====

// Cosine-with-warmup LR schedule
inline float lr_schedule(float base_lr, int step, int warmup, int total_steps) {
    if (step <= warmup) return base_lr * (float)step / (float)warmup;
    float progress = (float)(step - warmup) / (float)std::max(1, total_steps - warmup);
    if (progress > 1.0f) progress = 1.0f;
    float cos_v = 0.5f * (1.0f + std::cos((float)M_PI * progress));
    return base_lr * (0.1f + 0.9f * cos_v);
}

template <typename CacheT>
static float compute_accuracy(const CacheT& cache, const std::vector<int>& labels_h) {
    int B = cache.B;
    int C = cache.n_classes;
    int* preds_int;
    CUDA_CHECK(cudaMalloc(&preds_int, B * sizeof(int)));
    launch_argmax(cache.logits.d, preds_int, B, C);
    std::vector<int> ph(B);
    CUDA_CHECK(cudaMemcpy(ph.data(), preds_int, B * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(preds_int);
    int correct = 0;
    for (int i = 0; i < B; ++i) if (ph[i] == labels_h[i]) correct++;
    return (float)correct / (float)B;
}

static void emit_jsonl(std::ofstream* jsonl, const Args& a, int step,
                       float train_loss, float eval_loss, float eval_acc,
                       double tok_per_sec, int64_t n_params, double wall_s) {
    if (!jsonl) return;
    (*jsonl) << "{\"model\":\"" << a.model << "\""
             << ",\"task\":\"" << a.task << "\""
             << ",\"m\":" << a.m
             << ",\"T\":" << a.T
             << ",\"seed\":" << a.seed
             << ",\"step\":" << step
             << ",\"train_loss\":" << train_loss
             << ",\"val_loss\":" << eval_loss
             << ",\"val_acc\":" << eval_acc
             << ",\"tok_per_sec\":" << tok_per_sec
             << ",\"n_params\":" << n_params
             << ",\"wall_s\":" << wall_s
             << ",\"tag\":\"" << a.tag << "\"}\n";
    jsonl->flush();
}

static int train_ealrmn(Args& a, TaskWrapper& tw, std::ofstream* jsonl) {
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    EALRMNModel model;
    model.init_K_method = a.init_K;
    model.use_attmem = (a.readout != "s_only");
    model.init(cublas, tw.V, a.m, a.J, tw.n_classes, a.seed);

    HostRng rng(a.seed);
    HostRng eval_rng(a.seed + 17);
    std::vector<int> ids_h(a.batch * a.T), labels_h(a.batch);
    std::vector<int> eval_ids_h(a.eval_batch * a.T), eval_labels_h(a.eval_batch);
    int* ids_d = make_int_tensor(a.batch * a.T);
    int* labels_d = make_int_tensor(a.batch);
    int* eval_ids_d = make_int_tensor(a.eval_batch * a.T);
    int* eval_labels_d = make_int_tensor(a.eval_batch);

    EALRMNModelCache cache, eval_cache;
    cache.alloc(a.batch, a.T, a.m, a.J, tw.n_classes);
    eval_cache.alloc(a.eval_batch, a.T, a.m, a.J, tw.n_classes);

    auto wall_start = std::chrono::high_resolution_clock::now();
    int64_t tokens_seen = 0;
    int64_t n_params = model.num_params();
    fprintf(stderr, "Model %s task %s m=%d T=%d B=%d steps=%d seed=%llu n_params=%lld\n",
            a.model.c_str(), a.task.c_str(), a.m, a.T, a.batch, a.steps,
            (unsigned long long)a.seed, (long long)n_params);

    for (int step = 1; step <= a.steps; ++step) {
        tw.generate(ids_h, labels_h, a.batch, rng);
        copy_ints_h2d(ids_d, ids_h);
        copy_ints_h2d(labels_d, labels_h);

        model.zero_grads();
        model.forward(ids_d, a.batch, a.T, cache);
        float train_loss = model.compute_loss(labels_d, cache) / (float)a.batch;
        model.backward(labels_d, cache);
        clip_grads(model.grads(), a.grad_clip);

        float lr_now = lr_schedule(a.lr, step, a.warmup, a.steps);
        auto ps = model.params();
        auto gs = model.grads();
        auto ms = model.ms();
        auto vs = model.vs();
        for (size_t i = 0; i < ps.size(); ++i) {
            adamw_step(*ps[i], *ms[i], *vs[i], *gs[i], lr_now, 0.9f, 0.95f, 1e-8f, a.wd, step);
        }
        tokens_seen += (int64_t)a.batch * a.T;

        if (step % a.print_every == 0 || step == 1) {
            auto wall_now = std::chrono::high_resolution_clock::now();
            double wall_s = std::chrono::duration<double>(wall_now - wall_start).count();
            double tps = tokens_seen / std::max(wall_s, 1e-9);
            fprintf(stderr, "[%s/%s m=%d T=%d s=%llu] step %5d train_loss=%.4f lr=%.6f tok/s=%.1f\n",
                    a.model.c_str(), a.task.c_str(), a.m, a.T,
                    (unsigned long long)a.seed, step, train_loss, lr_now, tps);
        }

        if (step % a.eval_every == 0 || step == a.steps) {
            tw.generate(eval_ids_h, eval_labels_h, a.eval_batch, eval_rng);
            copy_ints_h2d(eval_ids_d, eval_ids_h);
            copy_ints_h2d(eval_labels_d, eval_labels_h);
            model.forward(eval_ids_d, a.eval_batch, a.T, eval_cache);
            float eval_loss = model.compute_loss(eval_labels_d, eval_cache) / (float)a.eval_batch;
            float eval_acc = compute_accuracy(eval_cache, eval_labels_h);
            auto wall_now = std::chrono::high_resolution_clock::now();
            double wall_s = std::chrono::duration<double>(wall_now - wall_start).count();
            double tps = tokens_seen / std::max(wall_s, 1e-9);
            fprintf(stderr, "  EVAL step %5d val_loss=%.4f val_acc=%.4f wall_s=%.1f\n",
                    step, eval_loss, eval_acc, wall_s);
            emit_jsonl(jsonl, a, step, train_loss, eval_loss, eval_acc, tps, n_params, wall_s);
        }
    }

    cache.free(); eval_cache.free();
    cudaFree(ids_d); cudaFree(labels_d); cudaFree(eval_ids_d); cudaFree(eval_labels_d);
    model.free_all();
    cublasDestroy(cublas);
    return 0;
}

static int train_rnn(Args& a, TaskWrapper& tw, std::ofstream* jsonl) {
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    RNNModel model;
    model.init_Wh_method = a.init_Wh;
    model.use_tanh = (a.use_tanh != 0);
    model.init(cublas, tw.V, a.m, tw.n_classes, a.seed);

    HostRng rng(a.seed);
    HostRng eval_rng(a.seed + 17);
    std::vector<int> ids_h(a.batch * a.T), labels_h(a.batch);
    std::vector<int> eval_ids_h(a.eval_batch * a.T), eval_labels_h(a.eval_batch);
    int* ids_d = make_int_tensor(a.batch * a.T);
    int* labels_d = make_int_tensor(a.batch);
    int* eval_ids_d = make_int_tensor(a.eval_batch * a.T);
    int* eval_labels_d = make_int_tensor(a.eval_batch);

    RNNModelCache cache, eval_cache;
    cache.alloc(a.batch, a.T, a.m, tw.n_classes);
    eval_cache.alloc(a.eval_batch, a.T, a.m, tw.n_classes);

    auto wall_start = std::chrono::high_resolution_clock::now();
    int64_t tokens_seen = 0;
    int64_t n_params = model.num_params();
    fprintf(stderr, "Model %s task %s m=%d T=%d B=%d steps=%d seed=%llu n_params=%lld\n",
            a.model.c_str(), a.task.c_str(), a.m, a.T, a.batch, a.steps,
            (unsigned long long)a.seed, (long long)n_params);

    for (int step = 1; step <= a.steps; ++step) {
        tw.generate(ids_h, labels_h, a.batch, rng);
        copy_ints_h2d(ids_d, ids_h);
        copy_ints_h2d(labels_d, labels_h);

        model.zero_grads();
        model.forward(ids_d, a.batch, a.T, cache);
        float train_loss = model.compute_loss(labels_d, cache) / (float)a.batch;
        model.backward(labels_d, cache);
        clip_grads(model.grads(), a.grad_clip);

        float lr_now = lr_schedule(a.lr, step, a.warmup, a.steps);
        auto ps = model.params();
        auto gs = model.grads();
        auto ms = model.ms();
        auto vs = model.vs();
        for (size_t i = 0; i < ps.size(); ++i) {
            adamw_step(*ps[i], *ms[i], *vs[i], *gs[i], lr_now, 0.9f, 0.95f, 1e-8f, a.wd, step);
        }
        tokens_seen += (int64_t)a.batch * a.T;

        if (step % a.print_every == 0 || step == 1) {
            auto wall_now = std::chrono::high_resolution_clock::now();
            double wall_s = std::chrono::duration<double>(wall_now - wall_start).count();
            double tps = tokens_seen / std::max(wall_s, 1e-9);
            fprintf(stderr, "[%s/%s m=%d T=%d s=%llu] step %5d train_loss=%.4f lr=%.6f tok/s=%.1f\n",
                    a.model.c_str(), a.task.c_str(), a.m, a.T,
                    (unsigned long long)a.seed, step, train_loss, lr_now, tps);
        }

        if (step % a.eval_every == 0 || step == a.steps) {
            tw.generate(eval_ids_h, eval_labels_h, a.eval_batch, eval_rng);
            copy_ints_h2d(eval_ids_d, eval_ids_h);
            copy_ints_h2d(eval_labels_d, eval_labels_h);
            model.forward(eval_ids_d, a.eval_batch, a.T, eval_cache);
            float eval_loss = model.compute_loss(eval_labels_d, eval_cache) / (float)a.eval_batch;
            float eval_acc = compute_accuracy(eval_cache, eval_labels_h);
            auto wall_now = std::chrono::high_resolution_clock::now();
            double wall_s = std::chrono::duration<double>(wall_now - wall_start).count();
            double tps = tokens_seen / std::max(wall_s, 1e-9);
            fprintf(stderr, "  EVAL step %5d val_loss=%.4f val_acc=%.4f wall_s=%.1f\n",
                    step, eval_loss, eval_acc, wall_s);
            emit_jsonl(jsonl, a, step, train_loss, eval_loss, eval_acc, tps, n_params, wall_s);
        }
    }

    cache.free(); eval_cache.free();
    cudaFree(ids_d); cudaFree(labels_d); cudaFree(eval_ids_d); cudaFree(eval_labels_d);
    model.free_all();
    cublasDestroy(cublas);
    return 0;
}

// Transformer-specific train loop (separate cache type)
static int train_grp_rnn(Args& a, TaskWrapper& tw, std::ofstream* jsonl) {
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    GRPRNNModel model;
    model.tanh_state = (a.grp_tanh_state != 0);
    model.fixed_angles = (a.grp_fixed_angles != 0);
    model.disjoint_planes = (a.grp_disjoint_planes != 0);
    model.use_layernorm = (a.grp_layernorm != 0);
    model.stride = a.grp_stride;
    // Default K: m for interlocking (each coord visits 2 Givens), m/2 for disjoint pairs.
    int K = a.grp_K > 0 ? a.grp_K : (a.grp_disjoint_planes ? (a.m / 2) : a.m);
    // In LM mode, the readout predicts the next token, so n_classes = vocab size.
    bool lm_mode_init = (a.lm_mode != 0);
    int head_classes = lm_mode_init ? tw.V : tw.n_classes;
    model.init(cublas, tw.V, a.m, K, head_classes, a.seed);

    HostRng rng(a.seed);
    HostRng eval_rng(a.seed + 17);

    // Multi-phase curriculum schedule. Each entry: (T, step_threshold_inclusive).
    // The last entry is implicitly the final phase at a.T with threshold a.steps.
    // Examples:
    //   --curriculum-T=8 --curriculum-steps=4000 (legacy 2-phase) -> [(8, 4000), (a.T, a.steps)]
    //   --curriculum-schedule=8:4000,32:12000   -> [(8, 4000), (32, 12000), (a.T, a.steps)]
    struct CurrPhase { int T; int step_threshold; };
    std::vector<CurrPhase> sched;
    if (!a.curriculum_schedule.empty()) {
        std::stringstream ss(a.curriculum_schedule);
        std::string item;
        while (std::getline(ss, item, ',')) {
            size_t colon = item.find(':');
            if (colon == std::string::npos) continue;
            int t_val = std::stoi(item.substr(0, colon));
            int s_val = std::stoi(item.substr(colon + 1));
            sched.push_back({t_val, s_val});
        }
    } else if (a.curriculum_T > 0 && a.curriculum_steps > 0 && a.curriculum_T != a.T) {
        sched.push_back({a.curriculum_T, a.curriculum_steps});
    }
    sched.push_back({a.T, a.steps});  // final phase

    bool use_curriculum = (sched.size() > 1);
    if (use_curriculum && tw.kind != 3) {
        fprintf(stderr, "WARNING: curriculum only supported for a5_word task; ignoring.\n");
        sched.clear();
        sched.push_back({a.T, a.steps});
        use_curriculum = false;
    }

    int phase_idx = 0;
    int T_curr = sched[0].T;
    if (use_curriculum) {
        tw.a5.T = T_curr;
        fprintf(stderr, "Curriculum schedule (%zu phases):", sched.size());
        for (auto& p : sched) fprintf(stderr, " T=%d-until-step-%d;", p.T, p.step_threshold);
        fprintf(stderr, "\n");
    }

    // Allocate input buffers for the largest T in the schedule.
    int T_max = 0;
    for (auto& p : sched) T_max = std::max(T_max, p.T);
    std::vector<int> ids_h(a.batch * T_max);
    std::vector<int> labels_h(a.batch);
    std::vector<int> eval_ids_h(a.eval_batch * T_max);
    std::vector<int> eval_labels_h(a.eval_batch);
    int* ids_d = make_int_tensor(a.batch * T_max);
    int* labels_d = make_int_tensor(a.batch);
    int* eval_ids_d = make_int_tensor(a.eval_batch * T_max);
    int* eval_labels_d = make_int_tensor(a.eval_batch);

    GRPRNNModelCache cache, eval_cache;
    bool lm_mode = (a.lm_mode != 0);
    cache.alloc(a.batch, T_curr, a.m, K, head_classes, lm_mode);
    eval_cache.alloc(a.eval_batch, T_curr, a.m, K, head_classes, lm_mode);

    auto wall_start = std::chrono::high_resolution_clock::now();
    int64_t tokens_seen = 0;
    int64_t n_params = model.num_params();
    fprintf(stderr, "Model %s task %s m=%d T=%d K=%d B=%d steps=%d seed=%llu n_params=%lld stride=%d disjoint=%d tanh_state=%d fixed_angles=%d\n",
            a.model.c_str(), a.task.c_str(), a.m, a.T, K, a.batch, a.steps,
            (unsigned long long)a.seed, (long long)n_params,
            a.grp_stride, a.grp_disjoint_planes, a.grp_tanh_state, a.grp_fixed_angles);

    for (int step = 1; step <= a.steps; ++step) {
        // Curriculum: advance phase when step exceeds current phase's threshold.
        if (use_curriculum && phase_idx + 1 < (int)sched.size() &&
            step > sched[phase_idx].step_threshold) {
            phase_idx++;
            cache.free(); eval_cache.free();
            T_curr = sched[phase_idx].T;
            tw.a5.T = T_curr;
            cache.alloc(a.batch, T_curr, a.m, K, head_classes, lm_mode);
            eval_cache.alloc(a.eval_batch, T_curr, a.m, K, head_classes, lm_mode);
            fprintf(stderr, "[curriculum] switching to T=%d at step %d (phase %d/%zu)\n",
                    T_curr, step, phase_idx + 1, sched.size());
        }
        // generate sized for T_curr (resize the input buffer accordingly)
        ids_h.assign(a.batch * T_curr, 0);
        tw.generate(ids_h, labels_h, a.batch, rng);
        CUDA_CHECK(cudaMemcpy(ids_d, ids_h.data(), ids_h.size() * sizeof(int), cudaMemcpyHostToDevice));
        copy_ints_h2d(labels_d, labels_h);

        // LM mode: build per-step labels labels_lm[t*B + b] = ids[b*T + (t+1)] for t=0..T-2.
        if (lm_mode) {
            dim3 block(1);
            dim3 grid(a.batch, T_curr - 1);
            k_build_lm_labels<<<grid, block>>>(ids_d, cache.labels_lm, a.batch, T_curr);
        }

        model.zero_grads();
        model.forward(ids_d, a.batch, T_curr, cache);
        float train_loss_sum = model.compute_loss(labels_d, cache);
        float train_loss = lm_mode
            ? train_loss_sum / (float)((T_curr - 1) * a.batch)
            : train_loss_sum / (float)a.batch;
        model.backward(labels_d, cache);
        clip_grads(model.grads(), a.grad_clip);

        float lr_now = lr_schedule(a.lr, step, a.warmup, a.steps);
        auto ps = model.params();
        auto gs = model.grads();
        auto ms = model.ms();
        auto vs = model.vs();
        for (size_t i = 0; i < ps.size(); ++i) {
            adamw_step(*ps[i], *ms[i], *vs[i], *gs[i], lr_now, 0.9f, 0.95f, 1e-8f, a.wd, step);
        }
        tokens_seen += (int64_t)a.batch * a.T;

        tokens_seen += (int64_t)a.batch * T_curr;

        if (step % a.print_every == 0 || step == 1) {
            auto wall_now = std::chrono::high_resolution_clock::now();
            double wall_s = std::chrono::duration<double>(wall_now - wall_start).count();
            double tps = tokens_seen / std::max(wall_s, 1e-9);
            fprintf(stderr, "[%s/%s m=%d T=%d K=%d s=%llu] step %5d train_loss=%.4f lr=%.6f tok/s=%.1f\n",
                    a.model.c_str(), a.task.c_str(), a.m, T_curr, K,
                    (unsigned long long)a.seed, step, train_loss, lr_now, tps);
        }

        if (step % a.eval_every == 0 || step == a.steps) {
            eval_ids_h.assign(a.eval_batch * T_curr, 0);
            tw.generate_val(eval_ids_h, eval_labels_h, a.eval_batch, eval_rng);
            CUDA_CHECK(cudaMemcpy(eval_ids_d, eval_ids_h.data(), eval_ids_h.size() * sizeof(int), cudaMemcpyHostToDevice));
            copy_ints_h2d(eval_labels_d, eval_labels_h);
            if (lm_mode) {
                dim3 block(1);
                dim3 grid(a.eval_batch, T_curr - 1);
                k_build_lm_labels<<<grid, block>>>(eval_ids_d, eval_cache.labels_lm, a.eval_batch, T_curr);
            }
            model.forward(eval_ids_d, a.eval_batch, T_curr, eval_cache);
            float eval_loss_sum = model.compute_loss(eval_labels_d, eval_cache);
            float eval_loss = lm_mode
                ? eval_loss_sum / (float)((T_curr - 1) * a.eval_batch)
                : eval_loss_sum / (float)a.eval_batch;
            float eval_acc = compute_accuracy(eval_cache, eval_labels_h);
            auto wall_now = std::chrono::high_resolution_clock::now();
            double wall_s = std::chrono::duration<double>(wall_now - wall_start).count();
            double tps = tokens_seen / std::max(wall_s, 1e-9);
            fprintf(stderr, "  EVAL step %5d T_curr=%d val_loss=%.4f val_acc=%.4f wall_s=%.1f\n",
                    step, T_curr, eval_loss, eval_acc, wall_s);
            emit_jsonl(jsonl, a, step, train_loss, eval_loss, eval_acc, tps, n_params, wall_s);
        }
    }

    cache.free(); eval_cache.free();
    cudaFree(ids_d); cudaFree(labels_d); cudaFree(eval_ids_d); cudaFree(eval_labels_d);
    model.free_all();
    cublasDestroy(cublas);
    return 0;
}

// Train loop for multi-layer GRP-RNN stack (LM-only).
static int train_grp_stack(Args& a, TaskWrapper& tw, std::ofstream* jsonl) {
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    GRPStackModel model;
    model.stride = a.grp_stride;
    model.linear_recurrence = (a.linear_recurrence != 0);
    int K = a.grp_K > 0 ? a.grp_K : a.m;
    int mlp_h = (a.mlp_hidden > 0) ? a.mlp_hidden : 4 * a.m;
    model.init(cublas, tw.V, a.m, K, a.n_layers, mlp_h, a.seed);

    HostRng rng(a.seed);
    HostRng eval_rng(a.seed + 17);

    std::vector<int> ids_h(a.batch * a.T);
    std::vector<int> labels_h(a.batch);
    int* ids_d = make_int_tensor(a.batch * a.T);
    int* labels_d = make_int_tensor(a.batch);
    std::vector<int> eval_ids_h(a.eval_batch * a.T);
    std::vector<int> eval_labels_h(a.eval_batch);
    int* eval_ids_d = make_int_tensor(a.eval_batch * a.T);
    int* eval_labels_d = make_int_tensor(a.eval_batch);

    GRPStackCache cache, eval_cache;
    cache.alloc(a.batch, a.T, a.m, K, a.n_layers, tw.V, mlp_h);
    eval_cache.alloc(a.eval_batch, a.T, a.m, K, a.n_layers, tw.V, mlp_h);

    auto wall_start = std::chrono::high_resolution_clock::now();
    int64_t tokens_seen = 0;
    int64_t n_params = model.num_params();
    fprintf(stderr, "Model %s task %s m=%d T=%d K=%d L=%d mlp_h=%d B=%d steps=%d seed=%llu n_params=%lld\n",
            a.model.c_str(), a.task.c_str(), a.m, a.T, K, a.n_layers, mlp_h, a.batch, a.steps,
            (unsigned long long)a.seed, (long long)n_params);

    for (int step = 1; step <= a.steps; ++step) {
        tw.generate(ids_h, labels_h, a.batch, rng);
        CUDA_CHECK(cudaMemcpy(ids_d, ids_h.data(), ids_h.size() * sizeof(int), cudaMemcpyHostToDevice));
        copy_ints_h2d(labels_d, labels_h);
        // Build per-step LM labels.
        {
            dim3 block(1);
            dim3 grid(a.batch, a.T - 1);
            k_build_lm_labels<<<grid, block>>>(ids_d, cache.labels_lm, a.batch, a.T);
        }

        model.zero_grads();
        model.forward(ids_d, a.batch, a.T, cache);
        float train_loss_sum = model.compute_loss(labels_d, cache);
        float train_loss = train_loss_sum / (float)((a.T - 1) * a.batch);
        model.backward(labels_d, cache);
        clip_grads(model.grads(), a.grad_clip);

        float lr_now = lr_schedule(a.lr, step, a.warmup, a.steps);
        auto ps = model.params();
        auto gs = model.grads();
        auto ms = model.ms();
        auto vs = model.vs();
        for (size_t i = 0; i < ps.size(); ++i) {
            adamw_step(*ps[i], *ms[i], *vs[i], *gs[i], lr_now, 0.9f, 0.95f, 1e-8f, a.wd, step);
        }
        tokens_seen += (int64_t)a.batch * a.T;

        if (step % a.print_every == 0 || step == 1) {
            auto wall_now = std::chrono::high_resolution_clock::now();
            double wall_s = std::chrono::duration<double>(wall_now - wall_start).count();
            double tps = tokens_seen / std::max(wall_s, 1e-9);
            fprintf(stderr, "[%s/%s m=%d T=%d L=%d s=%llu] step %5d train_loss=%.4f lr=%.6f tok/s=%.1f\n",
                    a.model.c_str(), a.task.c_str(), a.m, a.T, a.n_layers,
                    (unsigned long long)a.seed, step, train_loss, lr_now, tps);
        }

        if (step % a.eval_every == 0 || step == a.steps) {
            tw.generate_val(eval_ids_h, eval_labels_h, a.eval_batch, eval_rng);
            CUDA_CHECK(cudaMemcpy(eval_ids_d, eval_ids_h.data(), eval_ids_h.size() * sizeof(int), cudaMemcpyHostToDevice));
            copy_ints_h2d(eval_labels_d, eval_labels_h);
            {
                dim3 block(1);
                dim3 grid(a.eval_batch, a.T - 1);
                k_build_lm_labels<<<grid, block>>>(eval_ids_d, eval_cache.labels_lm, a.eval_batch, a.T);
            }
            model.forward(eval_ids_d, a.eval_batch, a.T, eval_cache);
            float eval_loss_sum = model.compute_loss(eval_labels_d, eval_cache);
            float eval_loss = eval_loss_sum / (float)((a.T - 1) * a.eval_batch);
            auto wall_now = std::chrono::high_resolution_clock::now();
            double wall_s = std::chrono::duration<double>(wall_now - wall_start).count();
            double tps = tokens_seen / std::max(wall_s, 1e-9);
            fprintf(stderr, "  EVAL step %5d val_loss=%.4f wall_s=%.1f\n", step, eval_loss, wall_s);
            emit_jsonl(jsonl, a, step, train_loss, eval_loss, 0.0f, tps, n_params, wall_s);
        }
    }

    cache.free(); eval_cache.free();
    cudaFree(ids_d); cudaFree(labels_d); cudaFree(eval_ids_d); cudaFree(eval_labels_d);
    model.free_all();
    cublasDestroy(cublas);
    return 0;
}

static int train_transformer(Args& a, TaskWrapper& tw, std::ofstream* jsonl) {
    cublasHandle_t cublas_t;
    CUBLAS_CHECK(cublasCreate(&cublas_t));
    int L_layers = (a.model == "transformer_2l") ? 2 : 1;
    TransformerModel model;
    model.init(cublas_t, tw.V, a.m, a.H, L_layers, tw.n_classes, a.T, a.seed);

    HostRng rng(a.seed);
    HostRng eval_rng(a.seed + 17);

    std::vector<int> ids_h(a.batch * a.T);
    std::vector<int> labels_h(a.batch);
    int* ids_d = make_int_tensor(a.batch * a.T);
    int* labels_d = make_int_tensor(a.batch);
    std::vector<int> eval_ids_h(a.eval_batch * a.T);
    std::vector<int> eval_labels_h(a.eval_batch);
    int* eval_ids_d = make_int_tensor(a.eval_batch * a.T);
    int* eval_labels_d = make_int_tensor(a.eval_batch);

    TransformerModelCache cache;
    cache.alloc(a.batch, a.T, a.m, a.H, L_layers, tw.n_classes);
    TransformerModelCache eval_cache;
    eval_cache.alloc(a.eval_batch, a.T, a.m, a.H, L_layers, tw.n_classes);

    auto wall_start = std::chrono::high_resolution_clock::now();
    int64_t tokens_seen = 0;

    int64_t n_params = model.num_params();
    fprintf(stderr, "Model %s task %s m=%d T=%d B=%d steps=%d seed=%llu n_params=%lld\n",
            a.model.c_str(), a.task.c_str(), a.m, a.T, a.batch, a.steps,
            (unsigned long long)a.seed, (long long)n_params);

    for (int step = 1; step <= a.steps; ++step) {
        tw.generate(ids_h, labels_h, a.batch, rng);
        copy_ints_h2d(ids_d, ids_h);
        copy_ints_h2d(labels_d, labels_h);

        model.zero_grads();
        model.forward(ids_d, a.batch, a.T, cache);
        float loss_sum = model.compute_loss(labels_d, cache);
        float train_loss = loss_sum / (float)a.batch;
        model.backward(labels_d, cache);

        clip_grads(model.all_grads(), a.grad_clip);

        float lr_now = lr_schedule(a.lr, step, a.warmup, a.steps);
        auto ps = model.all_params();
        auto gs = model.all_grads();
        auto ms = model.all_ms();
        auto vs = model.all_vs();
        for (size_t i = 0; i < ps.size(); ++i) {
            adamw_step(*ps[i], *ms[i], *vs[i], *gs[i], lr_now, 0.9f, 0.95f, 1e-8f, a.wd, step);
        }

        tokens_seen += (int64_t)a.batch * a.T;

        if (step % a.print_every == 0 || step == 1) {
            auto wall_now = std::chrono::high_resolution_clock::now();
            double wall_s = std::chrono::duration<double>(wall_now - wall_start).count();
            double tps = tokens_seen / std::max(wall_s, 1e-9);
            fprintf(stderr, "[%s/%s m=%d T=%d s=%llu] step %5d train_loss=%.4f lr=%.6f tok/s=%.1f\n",
                    a.model.c_str(), a.task.c_str(), a.m, a.T,
                    (unsigned long long)a.seed, step, train_loss, lr_now, tps);
        }

        if (step % a.eval_every == 0 || step == a.steps) {
            tw.generate(eval_ids_h, eval_labels_h, a.eval_batch, eval_rng);
            copy_ints_h2d(eval_ids_d, eval_ids_h);
            copy_ints_h2d(eval_labels_d, eval_labels_h);
            model.forward(eval_ids_d, a.eval_batch, a.T, eval_cache);
            float eval_loss_sum = model.compute_loss(eval_labels_d, eval_cache);
            float eval_loss = eval_loss_sum / (float)a.eval_batch;
            float eval_acc = compute_accuracy(eval_cache, eval_labels_h);
            auto wall_now = std::chrono::high_resolution_clock::now();
            double wall_s = std::chrono::duration<double>(wall_now - wall_start).count();
            double tps = tokens_seen / std::max(wall_s, 1e-9);

            fprintf(stderr, "  EVAL step %5d val_loss=%.4f val_acc=%.4f wall_s=%.1f\n",
                    step, eval_loss, eval_acc, wall_s);

            if (jsonl) {
                (*jsonl) << "{\"model\":\"" << a.model << "\""
                         << ",\"task\":\"" << a.task << "\""
                         << ",\"m\":" << a.m
                         << ",\"T\":" << a.T
                         << ",\"seed\":" << a.seed
                         << ",\"step\":" << step
                         << ",\"train_loss\":" << train_loss
                         << ",\"val_loss\":" << eval_loss
                         << ",\"val_acc\":" << eval_acc
                         << ",\"tok_per_sec\":" << tps
                         << ",\"n_params\":" << n_params
                         << ",\"wall_s\":" << wall_s
                         << ",\"tag\":\"" << a.tag << "\"}\n";
                jsonl->flush();
            }
        }
    }

    cache.free();
    eval_cache.free();
    cudaFree(ids_d); cudaFree(labels_d);
    cudaFree(eval_ids_d); cudaFree(eval_labels_d);
    model.free_all();
    cublasDestroy(cublas_t);
    return 0;
}

// ===== Gradient checking =====
// For a small config, compute (loss(w + eps) - loss(w - eps)) / (2 eps) and compare to analytic dW.
// We pick a few random indices in each parameter tensor.

// Adaptive gradient check: per index, scale eps to gradient magnitude when possible.
// FP32-noise-aware tolerance: pass if rel < tol OR abs_err < abs_tol.
template <typename ModelT, typename CacheT>
static bool gradcheck_model(ModelT& model, TaskWrapper& tw, CacheT& cache,
                            const int* ids_d, const int* labels_d,
                            int B, int T,
                            float eps = 5e-3f, float tol = 5e-2f, float abs_tol = 5e-4f) {
    // Run forward + backward once to populate analytic grads.
    model.zero_grads();
    model.forward(ids_d, B, T, cache);
    model.compute_loss(labels_d, cache);
    model.backward(labels_d, cache);

    auto ps = model.params();
    auto gs = model.grads();
    HostRng rng(12345);
    int n_checks = 6;
    int n_failed = 0;
    int n_total = 0;

    for (size_t pi = 0; pi < ps.size(); ++pi) {
        Tensor* W = ps[pi];
        Tensor* G = gs[pi];
        int n = (int)W->numel;
        if (n == 0) continue;
        for (int c = 0; c < n_checks; ++c) {
            int idx = rng.next_int(n);
            float w_orig, analytic;
            CUDA_CHECK(cudaMemcpy(&w_orig, W->d + idx, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&analytic, G->d + idx, sizeof(float), cudaMemcpyDeviceToHost));

            float w_plus = w_orig + eps;
            CUDA_CHECK(cudaMemcpy(W->d + idx, &w_plus, sizeof(float), cudaMemcpyHostToDevice));
            model.forward(ids_d, B, T, cache);
            float l_plus = model.compute_loss(labels_d, cache) / (float)B;

            float w_minus = w_orig - eps;
            CUDA_CHECK(cudaMemcpy(W->d + idx, &w_minus, sizeof(float), cudaMemcpyHostToDevice));
            model.forward(ids_d, B, T, cache);
            float l_minus = model.compute_loss(labels_d, cache) / (float)B;

            CUDA_CHECK(cudaMemcpy(W->d + idx, &w_orig, sizeof(float), cudaMemcpyHostToDevice));

            float numeric = (l_plus - l_minus) / (2.0f * eps);
            float denom = std::max(std::fabs(numeric) + std::fabs(analytic), 1e-6f);
            float rel = std::fabs(numeric - analytic) / denom;
            n_total++;
            bool ok = rel < tol || std::fabs(numeric - analytic) < abs_tol;
            if (!ok) {
                fprintf(stderr, "  param %zu idx %d: num=%.6e ana=%.6e rel=%.4f  FAIL\n",
                        pi, idx, numeric, analytic, rel);
                n_failed++;
            }
        }
    }
    fprintf(stderr, "gradcheck: %d/%d passed\n", n_total - n_failed, n_total);
    return n_failed == 0;
}

static int mode_gradcheck(Args& a, TaskWrapper& tw) {
    // small config: B=2, T=8, m=8
    a.batch = 2; a.T = 8; a.m = 8;
    tw.init(a);
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    HostRng rng(a.seed);
    std::vector<int> ids_h(a.batch * a.T);
    std::vector<int> labels_h(a.batch);
    tw.generate(ids_h, labels_h, a.batch, rng);
    int* ids_d = make_int_tensor(a.batch * a.T);
    int* labels_d = make_int_tensor(a.batch);
    copy_ints_h2d(ids_d, ids_h);
    copy_ints_h2d(labels_d, labels_h);

    bool ok = false;
    if (a.model == "ealrmn_attmem") {
        EALRMNModel model;
        model.init(cublas, tw.V, a.m, a.J, tw.n_classes, a.seed);
        EALRMNModelCache cache;
        cache.alloc(a.batch, a.T, a.m, a.J, tw.n_classes);
        ok = gradcheck_model(model, tw, cache, ids_d, labels_d, a.batch, a.T);
        cache.free();
        model.free_all();
    } else if (a.model == "rnn") {
        RNNModel model;
        model.init(cublas, tw.V, a.m, tw.n_classes, a.seed);
        RNNModelCache cache;
        cache.alloc(a.batch, a.T, a.m, tw.n_classes);
        ok = gradcheck_model(model, tw, cache, ids_d, labels_d, a.batch, a.T);
        cache.free();
        model.free_all();
    } else if (a.model == "grp_rnn") {
        GRPRNNModel model;
        model.tanh_state = (a.grp_tanh_state != 0);
        model.fixed_angles = (a.grp_fixed_angles != 0);
        model.disjoint_planes = (a.grp_disjoint_planes != 0);
        model.use_layernorm = (a.grp_layernorm != 0);
        model.stride = a.grp_stride;
        // Default K: m for interlocking (each coord visits 2 Givens), m/2 for disjoint pairs.
    int K = a.grp_K > 0 ? a.grp_K : (a.grp_disjoint_planes ? (a.m / 2) : a.m);
        bool lm_mode = (a.lm_mode != 0);
        int head_classes = lm_mode ? tw.V : tw.n_classes;
        model.init(cublas, tw.V, a.m, K, head_classes, a.seed);
        GRPRNNModelCache cache;
        cache.alloc(a.batch, a.T, a.m, K, head_classes, lm_mode);
        if (lm_mode) {
            // Build labels_lm[t*B + b] = ids[b*T + (t+1)] for t=0..T-2
            dim3 block(1);
            dim3 grid(a.batch, a.T - 1);
            k_build_lm_labels<<<grid, block>>>(ids_d, cache.labels_lm, a.batch, a.T);
        }
        ok = gradcheck_model(model, tw, cache, ids_d, labels_d, a.batch, a.T);
        cache.free();
        model.free_all();
    } else if (a.model == "grp_stack") {
        GRPStackModel model;
        model.stride = a.grp_stride;
        model.linear_recurrence = (a.linear_recurrence != 0);
        int K = a.grp_K > 0 ? a.grp_K : a.m;
        int mlp_h = (a.mlp_hidden > 0) ? a.mlp_hidden : 4 * a.m;
        model.init(cublas, tw.V, a.m, K, a.n_layers, mlp_h, a.seed);
        GRPStackCache cache;
        cache.alloc(a.batch, a.T, a.m, K, a.n_layers, tw.V, mlp_h);
        // Build labels_lm[t*B + b] = ids[b*T + (t+1)] for t=0..T-2 (stack is LM-only)
        {
            dim3 block(1);
            dim3 grid(a.batch, a.T - 1);
            k_build_lm_labels<<<grid, block>>>(ids_d, cache.labels_lm, a.batch, a.T);
        }
        ok = gradcheck_model(model, tw, cache, ids_d, labels_d, a.batch, a.T);
        cache.free();
        model.free_all();
    } else if (a.model == "transformer_1l" || a.model == "transformer_2l") {
        int L_layers = (a.model == "transformer_2l") ? 2 : 1;
        TransformerModel model;
        model.init(cublas, tw.V, a.m, a.H, L_layers, tw.n_classes, a.T, a.seed);
        TransformerModelCache cache;
        cache.alloc(a.batch, a.T, a.m, a.H, L_layers, tw.n_classes);
        // Custom gradcheck for transformer (params accessor different)
        // Use a variant that uses all_params / all_grads.
        // We inline an adapted version.
        HostRng rng2(54321);
        auto ps = model.all_params();
        auto gs = model.all_grads();
        // Run forward+backward
        model.zero_grads();
        model.forward(ids_d, a.batch, a.T, cache);
        model.compute_loss(labels_d, cache);
        model.backward(labels_d, cache);
        int n_total = 0, n_failed = 0;
        int n_checks = 6;
        for (size_t pi = 0; pi < ps.size(); ++pi) {
            Tensor* W = ps[pi];
            Tensor* G = gs[pi];
            int n = (int)W->numel;
            if (n == 0) continue;
            for (int c = 0; c < n_checks; ++c) {
                int idx = rng2.next_int(n);
                float w_orig, analytic;
                CUDA_CHECK(cudaMemcpy(&w_orig, W->d + idx, sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&analytic, G->d + idx, sizeof(float), cudaMemcpyDeviceToHost));
                float eps = 5e-3f;
                float w_plus = w_orig + eps;
                CUDA_CHECK(cudaMemcpy(W->d + idx, &w_plus, sizeof(float), cudaMemcpyHostToDevice));
                model.forward(ids_d, a.batch, a.T, cache);
                float l_plus = model.compute_loss(labels_d, cache) / (float)a.batch;
                float w_minus = w_orig - eps;
                CUDA_CHECK(cudaMemcpy(W->d + idx, &w_minus, sizeof(float), cudaMemcpyHostToDevice));
                model.forward(ids_d, a.batch, a.T, cache);
                float l_minus = model.compute_loss(labels_d, cache) / (float)a.batch;
                CUDA_CHECK(cudaMemcpy(W->d + idx, &w_orig, sizeof(float), cudaMemcpyHostToDevice));
                float numeric = (l_plus - l_minus) / (2.0f * eps);
                float denom = std::max(std::fabs(numeric) + std::fabs(analytic), 1e-6f);
                float rel = std::fabs(numeric - analytic) / denom;
                bool ok2 = rel < 5e-2f || std::fabs(numeric - analytic) < 5e-4f;
                n_total++;
                if (!ok2) {
                    fprintf(stderr, "  param %zu idx %d: num=%.6e ana=%.6e rel=%.4f  FAIL\n",
                            pi, idx, numeric, analytic, rel);
                    n_failed++;
                }
            }
        }
        fprintf(stderr, "transformer gradcheck: %d/%d passed\n", n_total - n_failed, n_total);
        ok = (n_failed == 0);
        cache.free();
        model.free_all();
    } else {
        fprintf(stderr, "Unknown model for gradcheck: %s\n", a.model.c_str());
        ok = false;
    }
    cudaFree(ids_d); cudaFree(labels_d);
    cublasDestroy(cublas);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    TaskWrapper tw;
    tw.init(a);

    if (a.mode == "gradcheck") {
        return mode_gradcheck(a, tw);
    }

    std::ofstream* jsonl_p = nullptr;
    std::ofstream jsonl_f;
    if (!a.jsonl.empty()) {
        jsonl_f.open(a.jsonl, std::ios::app);
        jsonl_p = &jsonl_f;
    }

    int rc = 1;
    if (a.model == "ealrmn_attmem") {
        rc = train_ealrmn(a, tw, jsonl_p);
    } else if (a.model == "rnn") {
        rc = train_rnn(a, tw, jsonl_p);
    } else if (a.model == "grp_rnn") {
        rc = train_grp_rnn(a, tw, jsonl_p);
    } else if (a.model == "grp_stack") {
        rc = train_grp_stack(a, tw, jsonl_p);
    } else if (a.model == "transformer_1l" || a.model == "transformer_2l") {
        rc = train_transformer(a, tw, jsonl_p);
    } else {
        fprintf(stderr, "Unknown model: %s\n", a.model.c_str());
    }

    if (jsonl_p) jsonl_f.close();
    return rc;
}
