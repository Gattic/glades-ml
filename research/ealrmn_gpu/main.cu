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
        else {
            std::cerr << "Unknown arg: " << key << "\n";
        }
    }
    return a;
}

// ===== Task adapter =====
// Unifies generation interface across tasks.
struct TaskWrapper {
    int kind = 0;  // 0=needle, 1=hmm, 2=syntheticlm
    NeedleTask needle{8, 256};
    HmmTask hmm{8, 256};
    SynthLMTask lm{64, 256};
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
        } else {
            std::cerr << "Unknown task: " << a.task << "\n";
            std::exit(1);
        }
    }

    void generate(std::vector<int>& ids, std::vector<int>& labels,
                  int B, HostRng& rng) const {
        if (kind == 0) needle.generate(ids, labels, B, rng);
        else if (kind == 1) hmm.generate(ids, labels, B, rng);
        else                lm.generate(ids, labels, B, rng);
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
    } else if (a.model == "transformer_1l" || a.model == "transformer_2l") {
        rc = train_transformer(a, tw, jsonl_p);
    } else {
        fprintf(stderr, "Unknown model: %s\n", a.model.c_str());
    }

    if (jsonl_p) jsonl_f.close();
    return rc;
}
