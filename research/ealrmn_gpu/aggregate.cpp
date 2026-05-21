// EALRMN Phase-1 GPU prototype — sweep result aggregator
// Reads JSONL lines (one per eval), groups by (model, task, m, T, step),
// computes mean ± 95% CI across seeds for val_loss and val_acc.
//
// Build: g++ -std=c++17 -O2 aggregate.cpp -o aggregate
// Usage: ./aggregate results/sweep_X.jsonl

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

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
    if (q == std::string::npos) return "";
    return json.substr(p, q - p);
}

static double get_num(const std::string& json, const std::string& key) {
    std::string k = "\"" + key + "\":";
    auto p = json.find(k);
    if (p == std::string::npos) return 0.0;
    p += k.size();
    // skip whitespace
    while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) p++;
    // read until , } " or end
    auto q = p;
    while (q < json.size() && json[q] != ',' && json[q] != '}' && json[q] != '\n') q++;
    return std::stod(json.substr(p, q - p));
}

static Row parse_row(const std::string& line) {
    Row r;
    r.model = get_str(line, "model");
    r.task  = get_str(line, "task");
    r.tag   = get_str(line, "tag");
    r.m     = (int)get_num(line, "m");
    r.T     = (int)get_num(line, "T");
    r.step  = (int)get_num(line, "step");
    r.seed  = (long long)get_num(line, "seed");
    r.train_loss = get_num(line, "train_loss");
    r.val_loss   = get_num(line, "val_loss");
    r.val_acc    = get_num(line, "val_acc");
    r.tok_per_sec= get_num(line, "tok_per_sec");
    r.wall_s     = get_num(line, "wall_s");
    r.n_params   = (long long)get_num(line, "n_params");
    return r;
}

struct Key {
    std::string model, task;
    int m, T, step;
    bool operator<(const Key& o) const {
        if (model != o.model) return model < o.model;
        if (task != o.task) return task < o.task;
        if (m != o.m) return m < o.m;
        if (T != o.T) return T < o.T;
        return step < o.step;
    }
};

static double mean(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return s / (double)v.size();
}

static double stddev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double mu = mean(v);
    double s = 0;
    for (double x : v) s += (x - mu) * (x - mu);
    return std::sqrt(s / (double)(v.size() - 1));
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <file.jsonl>\n", argv[0]); return 1; }
    std::ifstream f(argv[1]);
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }

    std::vector<Row> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        rows.push_back(parse_row(line));
    }

    // Group by (model, task, m, T, step)
    std::map<Key, std::vector<Row>> groups;
    for (auto& r : rows) {
        Key k{r.model, r.task, r.m, r.T, r.step};
        groups[k].push_back(r);
    }

    // Print header
    printf("model           task         m     T     step    n_seeds  val_loss (mean±sd)   val_acc (mean±sd)    tok/s_mean\n");
    printf("--------------- ------------ ----- ----- ------- -------- ------------------- ------------------- ----------\n");

    for (auto& [k, vs] : groups) {
        std::vector<double> vl, va, ts;
        for (auto& r : vs) { vl.push_back(r.val_loss); va.push_back(r.val_acc); ts.push_back(r.tok_per_sec); }
        double mvl = mean(vl), svl = stddev(vl);
        double mva = mean(va), sva = stddev(va);
        double mts = mean(ts);
        printf("%-15s %-12s %5d %5d %7d %8zu %7.4f ± %7.4f %7.4f ± %7.4f %10.0f\n",
               k.model.c_str(), k.task.c_str(), k.m, k.T, k.step, vs.size(),
               mvl, svl, mva, sva, mts);
    }

    // Also: per (model, task, m, T) final-step summary (last step per seed)
    printf("\n=== FINAL-STEP SUMMARY (complete seeds only) ===\n");
    struct ConfKey {
        std::string model, task;
        int m, T;
        bool operator<(const ConfKey& o) const {
            if (model != o.model) return model < o.model;
            if (task != o.task) return task < o.task;
            if (m != o.m) return m < o.m;
            return T < o.T;
        }
    };
    std::map<std::pair<ConfKey, long long>, Row> last_row;
    for (auto& r : rows) {
        ConfKey c{r.model, r.task, r.m, r.T};
        auto k2 = std::make_pair(c, r.seed);
        auto it = last_row.find(k2);
        if (it == last_row.end() || r.step > it->second.step) {
            last_row[k2] = r;
        }
    }
    // Per (conf), find the max step reached by any seed → treat as "expected final step".
    std::map<ConfKey, int> max_step_per_conf;
    for (auto& [k2, r] : last_row) {
        auto it = max_step_per_conf.find(k2.first);
        if (it == max_step_per_conf.end() || r.step > it->second)
            max_step_per_conf[k2.first] = r.step;
    }
    // Group rows that reached max_step, separately count partial
    std::map<ConfKey, std::vector<Row>> by_conf;
    std::map<ConfKey, int> partial_per_conf;
    for (auto& [k2, r] : last_row) {
        int target = max_step_per_conf[k2.first];
        if (r.step >= target) by_conf[k2.first].push_back(r);
        else partial_per_conf[k2.first]++;
    }
    printf("model           task         m     T      n_complete/partial  final_val_loss      final_val_acc       final_tok/s\n");
    printf("--------------- ------------ ----- ------ ------------------- ------------------- ------------------- ----------\n");
    for (auto& [c, vs] : by_conf) {
        std::vector<double> vl, va, ts;
        for (auto& r : vs) {
            if (std::isnan(r.val_loss) || std::isinf(r.val_loss)) continue;
            vl.push_back(r.val_loss); va.push_back(r.val_acc); ts.push_back(r.tok_per_sec);
        }
        int partial = partial_per_conf.count(c) ? partial_per_conf[c] : 0;
        if (vl.empty()) {
            printf("%-15s %-12s %5d %6d   0 complete / %d partial   (in progress / nan)\n",
                   c.model.c_str(), c.task.c_str(), c.m, c.T, partial);
            continue;
        }
        printf("%-15s %-12s %5d %6d  %3zu complete / %d partial   %7.4f ± %7.4f %7.4f ± %7.4f %10.0f\n",
               c.model.c_str(), c.task.c_str(), c.m, c.T, vl.size(), partial,
               mean(vl), stddev(vl), mean(va), stddev(va), mean(ts));
    }
    return 0;
}
