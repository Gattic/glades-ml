// Markdown table generator for EALRMN Phase-1 GPU results.
// Reads JSONL, emits per-(T) tables grouped by model with mean ± stddev across seeds.
//
// Build: g++ -std=c++17 -O2 md_table.cpp -o md_table
// Usage: ./md_table results/sweep_prod_v1.jsonl

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>

struct Row {
    std::string model, task, tag;
    int m=0, T=0, step=0;
    long long seed=0;
    double train_loss=0, val_loss=0, val_acc=0, tok_per_sec=0, wall_s=0;
    long long n_params=0;
};

static std::string get_str(const std::string& json, const std::string& key) {
    std::string k = "\"" + key + "\":\"";
    auto p = json.find(k);
    if (p == std::string::npos) return "";
    p += k.size();
    auto q = json.find('"', p);
    return json.substr(p, q - p);
}

static double get_num(const std::string& json, const std::string& key) {
    std::string k = "\"" + key + "\":";
    auto p = json.find(k);
    if (p == std::string::npos) return 0.0;
    p += k.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) p++;
    auto q = p;
    while (q < json.size() && json[q] != ',' && json[q] != '}' && json[q] != '\n') q++;
    return std::stod(json.substr(p, q - p));
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <file.jsonl>\n", argv[0]); return 1; }
    std::ifstream f(argv[1]);
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    std::vector<Row> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        Row r;
        r.model = get_str(line, "model");
        r.task  = get_str(line, "task");
        r.m     = (int)get_num(line, "m");
        r.T     = (int)get_num(line, "T");
        r.step  = (int)get_num(line, "step");
        r.seed  = (long long)get_num(line, "seed");
        r.train_loss = get_num(line, "train_loss");
        r.val_loss   = get_num(line, "val_loss");
        r.val_acc    = get_num(line, "val_acc");
        r.tok_per_sec= get_num(line, "tok_per_sec");
        r.n_params   = (long long)get_num(line, "n_params");
        rows.push_back(r);
    }

    // Build set of (T) values seen, and last-step row per (model, T, seed).
    std::set<int> Ts;
    for (auto& r : rows) Ts.insert(r.T);

    // Map (model, T, seed) → last-step row.
    struct K3 { std::string model; int T; long long seed; bool operator<(const K3& o) const {
        if (model != o.model) return model < o.model;
        if (T != o.T) return T < o.T;
        return seed < o.seed;
    } };
    std::map<K3, Row> last;
    for (auto& r : rows) {
        K3 k{r.model, r.T, r.seed};
        auto it = last.find(k);
        if (it == last.end() || r.step > it->second.step) last[k] = r;
    }

    // For each T, emit a markdown table.
    auto fmt_pm = [](double m, double s) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.4f ± %.4f", m, s);
        return std::string(buf);
    };
    auto mean = [](const std::vector<double>& v) {
        double s = 0; for (double x : v) s += x; return s / v.size();
    };
    auto stddev = [](const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        double mu = 0; for (double x : v) mu += x; mu /= v.size();
        double s = 0; for (double x : v) s += (x - mu) * (x - mu);
        return std::sqrt(s / (v.size() - 1));
    };

    printf("# EALRMN Phase-1 GPU sweep results (auto-generated)\n\n");

    // Per (model, T), find the maximum step reached by any seed of that config.
    // Only include seeds whose final step matches the max (they've completed).
    struct C2 { std::string model; int m; int T; bool operator<(const C2& o) const {
        if (model != o.model) return model < o.model;
        if (m != o.m) return m < o.m;
        return T < o.T;
    } };
    std::map<C2, int> max_step;
    for (auto& [k, r] : last) {
        C2 c{r.model, r.m, r.T};
        auto it = max_step.find(c);
        if (it == max_step.end() || r.step > it->second) max_step[c] = r.step;
    }

    for (int T : Ts) {
        printf("## T = %d\n\n", T);
        printf("| model | m | n_params | n_seeds (complete/partial) | val_loss (final) | val_acc (final) | tok/s |\n");
        printf("|-------|---|---------:|---------------------------:|------------------:|----------------:|------:|\n");

        std::set<std::pair<std::string, int>> mods;
        for (auto& [k, r] : last) if (k.T == T) mods.insert({r.model, r.m});
        for (auto& [model, m] : mods) {
            std::vector<double> vl, va, ts; long long np = 0;
            int n_complete = 0, n_partial = 0;
            C2 cc{model, m, T};
            int target_step = max_step[cc];
            for (auto& [k, r] : last) {
                if (k.model == model && k.T == T && r.m == m) {
                    if (std::isnan(r.val_loss) || std::isinf(r.val_loss)) continue;
                    if (r.step < target_step) { n_partial++; continue; }
                    vl.push_back(r.val_loss);
                    va.push_back(r.val_acc);
                    ts.push_back(r.tok_per_sec);
                    np = r.n_params;
                    n_complete++;
                }
            }
            if (vl.empty()) {
                printf("| %s | %d | %lld | 0 / %d | (in progress) | — | — |\n",
                       model.c_str(), m, np, n_partial);
                continue;
            }
            printf("| %s | %d | %lld | %d / %d | %s | %s | %.0f |\n",
                   model.c_str(), m, np, n_complete, n_partial,
                   fmt_pm(mean(vl), stddev(vl)).c_str(),
                   fmt_pm(mean(va), stddev(va)).c_str(),
                   mean(ts));
        }
        printf("\n");
    }

    // Cross-T comparison: gap of EALRMN vs RNN at each T
    printf("## EALRMN vs RNN gap (val_loss difference) by T\n\n");
    printf("| m | T | ealrmn val_loss | rnn val_loss | diff (ealrmn − rnn) |\n");
    printf("|---|---|----------------:|-------------:|--------------------:|\n");

    std::set<int> mvals;
    for (auto& [k, r] : last) mvals.insert(r.m);
    for (int m : mvals) {
        for (int T : Ts) {
            std::vector<double> evl, rvl;
            for (auto& [k, r] : last) {
                if (k.T != T || r.m != m) continue;
                if (std::isnan(r.val_loss)) continue;
                if (k.model == "ealrmn_attmem") evl.push_back(r.val_loss);
                else if (k.model == "rnn") rvl.push_back(r.val_loss);
            }
            if (evl.empty() || rvl.empty()) continue;
            double me = mean(evl), se = stddev(evl);
            double mr = mean(rvl), sr = stddev(rvl);
            double diff = me - mr;
            printf("| %d | %d | %s | %s | %+.4f |\n",
                   m, T,
                   fmt_pm(me, se).c_str(), fmt_pm(mr, sr).c_str(), diff);
        }
    }
    return 0;
}
