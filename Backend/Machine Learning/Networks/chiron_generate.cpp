// chiron_generate.cpp — CHIRON token generation, sampling, TF eval.
// Ports of the pre-2026-07-03 chiron_infer implementations (see header for the
// bit-parity contracts; MT19937/canonical-double is toolchain-coupled).
// C++98.

#include "chiron_generate.h"

#include <cstdio>     // snprintf, printf
#include <cstdlib>    // getenv
#include <cstring>    // memset
#include <cmath>      // nextafter, std::exp, std::log
#include <algorithm>  // std::partial_sort, std::sort
#include <set>
#include <string>
#include <vector>

namespace glades {
namespace chiron {

// ---------------------------------------------------------------------------
// ChironMt19937 — C++98 MT19937 bit-compatible with std::mt19937, plus a
// canonical-double generator bit-compatible with this toolchain's
// std::uniform_real_distribution<double>(0,1) over std::mt19937.
//
// Replicated from libstdc++ 13 (g++ 13.3.0), verbatim behavior of:
//   /usr/include/c++/13/bits/random.tcc
//     - mersenne_twister_engine<...>::seed(result_type)          [line ~328]
//     - mersenne_twister_engine<...>::_M_gen_rand()  (the twist) [line ~399]
//     - mersenne_twister_engine<...>::operator()     (tempering) [line ~455]
//     - generate_canonical<_RealType,__bits,_URNG>()             [line ~3349]
//   /usr/include/c++/13/bits/random.h
//     - __detail::_Adaptor<mt19937,double>::operator() routes through
//       generate_canonical<double, numeric_limits<double>::digits, mt19937>
//     - uniform_real_distribution(0,1) => __urng min/max == 0/1
//
// mt19937 template params: w=32, n=624, m=397, r=31, a=0x9908b0df,
//   u=11, d=0xffffffff, s=7, b=0x9d2c5680, t=15, c=0xefc60000, l=18,
//   f=1812433253.  All arithmetic is naturally mod 2^32 in uint32_t (identical
//   low-32 bits to libstdc++'s uint_fast32_t-with-mod-2^32 computation).
// ---------------------------------------------------------------------------

namespace {
    const int    MT_N = 624;
    const int    MT_M = 397;
    const uint32_t MT_MATRIX_A   = 0x9908b0dfu;
    const uint32_t MT_UPPER_MASK = 0x80000000u;  // (~0u) << r, r=31
    const uint32_t MT_LOWER_MASK = 0x7fffffffu;  // ~upper
}

ChironMt19937::ChironMt19937(uint32_t seed)
{
    // seed(): _M_x[0] = sd; _M_x[i] = f*(x^(x>>(w-2))) + i,  f=1812433253, w=32.
    mt[0] = seed;
    for (int i = 1; i < MT_N; ++i)
    {
        uint32_t x = mt[i - 1];
        x ^= x >> 30;                       // >> (w-2)
        x = 1812433253u * x + (uint32_t)i;  // i < 624 == (i mod n); wraps mod 2^32
        mt[i] = x;
    }
    mti = MT_N;  // _M_p = state_size -> regenerate on first draw
}

uint32_t ChironMt19937::next_u32()
{
    // Reload the vector (twist) when exhausted — _M_gen_rand().
    if (mti >= MT_N)
    {
        int k;
        for (k = 0; k < MT_N - MT_M; ++k)
        {
            uint32_t y = (mt[k] & MT_UPPER_MASK) | (mt[k + 1] & MT_LOWER_MASK);
            mt[k] = mt[k + MT_M] ^ (y >> 1) ^ ((y & 0x1u) ? MT_MATRIX_A : 0u);
        }
        for (; k < MT_N - 1; ++k)
        {
            uint32_t y = (mt[k] & MT_UPPER_MASK) | (mt[k + 1] & MT_LOWER_MASK);
            mt[k] = mt[k + (MT_M - MT_N)] ^ (y >> 1) ^ ((y & 0x1u) ? MT_MATRIX_A : 0u);
        }
        uint32_t y = (mt[MT_N - 1] & MT_UPPER_MASK) | (mt[0] & MT_LOWER_MASK);
        mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ ((y & 0x1u) ? MT_MATRIX_A : 0u);
        mti = 0;
    }

    // Tempering: u=11,d=0xffffffff, s=7,b=0x9d2c5680, t=15,c=0xefc60000, l=18.
    uint32_t z = mt[mti++];
    z ^= (z >> 11) & 0xffffffffu;
    z ^= (z << 7) & 0x9d2c5680u;
    z ^= (z << 15) & 0xefc60000u;
    z ^= (z >> 18);
    return z;
}

double ChironMt19937::next_canonical_double()
{
    // generate_canonical<double, 53, mt19937>:
    //   b = min(digits=53, bits=53) = 53
    //   r = (max - min + 1) = 2^32,  log2r = 32
    //   m = max(1, ceil(b/log2r)) = ceil(53/32) = 2 engine draws
    //   sum = u0*1 + u1*2^32 ; tmp = 2^64 ; ret = sum/tmp
    //   if (ret >= 1) ret = nextafter(1, 0)      [_GLIBCXX_USE_C99_MATH_TR1]
    // The u1*2^32 product is exact (power-of-two scale); the rounding matches
    // libstdc++ at the sum and the divide.  (FMA-neutral: the multiply is exact.)
    const double r = 4294967296.0;   // 2^32
    double sum = 0.0;
    double tmp = 1.0;
    for (int k = 2; k != 0; --k)
    {
        sum += (double)next_u32() * tmp;
        tmp *= r;
    }
    double ret = sum / tmp;
    if (ret >= 1.0)
        ret = nextafter(1.0, 0.0);
    return ret;
}

// ---------------------------------------------------------------------------
// ChironGenParams — real defaults (load-bearing; match CLI defaults exactly).
// ---------------------------------------------------------------------------

ChironGenParams::ChironGenParams()
    : maxTokens(100)
    , temperature(0.8f)
    , topK(40)
    , topP(0.95f)
    , repWindow(256)
    , repPenalty(1.0f)
    , freqPenalty(1.2f)
    , presPenalty(0.4f)
    , noRepeatN(3)
    , seed(1337)
{
}

// ---------------------------------------------------------------------------
// chiron_sample_token — verbatim port of chiron_infer::sampleToken (lines 80-182).
// C++98: lambda comparators become functor structs.
// ---------------------------------------------------------------------------

namespace {
// Comparator: sort int indices by descending probability in p[].
// Used by both std::partial_sort (top-k) and std::sort (top-p).
struct CmpByProbDesc {
    const std::vector<double>& p;
    explicit CmpByProbDesc(const std::vector<double>& v) : p(v) {}
    bool operator()(int a, int b) const { return p[a] > p[b]; }
};
} // anonymous namespace

int chiron_sample_token(const std::vector<float>& logitsIn,
                        const ChironGenParams& gp,
                        const std::vector<int>& context,
                        ChironMt19937& rng)
{
    const int V = (int)logitsIn.size();

    // Repetition handling over the last repWindow tokens of the context.
    // Counters CHIRON's runaway repetition attractor (a generated token's logit
    // climbs with each repeat). Frequency penalty (subtractive, xcount) is the
    // key — it scales with the growing logit, unlike the multiplicative form.
    std::vector<float> logits(logitsIn);
    if (gp.repWindow > 0 &&
        (gp.repPenalty != 1.0f || gp.freqPenalty != 0.0f || gp.presPenalty != 0.0f) &&
        !context.empty())
    {
        std::vector<unsigned char> seen(V, 0);
        int start = (int)context.size() - gp.repWindow;
        if (start < 0) start = 0;
        for (int i = start; i < (int)context.size(); ++i)
        {
            int v = context[i]; if (v < 0 || v >= V) continue;
            logits[v] -= gp.freqPenalty;                 // per occurrence
            if (!seen[v])
            {
                seen[v] = 1;
                logits[v] -= gp.presPenalty;             // once if present
                if (gp.repPenalty != 1.0f)
                    logits[v] = (logits[v] > 0.0f)
                                ? logits[v] / gp.repPenalty
                                : logits[v] * gp.repPenalty;
            }
        }
    }

    // No-repeat n-gram: forbid any token that would re-form an n-gram already
    // in the context. The robust cure for phrase loops the frequency penalty
    // can't fully break.
    if (gp.noRepeatN > 1 && (int)context.size() >= gp.noRepeatN)
    {
        const int plen = gp.noRepeatN - 1;
        const int csz  = (int)context.size();
        for (int i = 0; i + gp.noRepeatN <= csz; ++i)
        {
            bool match = true;
            for (int j = 0; j < plen; ++j)
                if (context[i + j] != context[csz - plen + j]) { match = false; break; }
            if (match)
            {
                int b = context[i + plen];
                if (b >= 0 && b < V) logits[b] = -1e30f;
            }
        }
    }

    // Apply temperature.
    std::vector<float> adj(V);
    const float invT = (gp.temperature > 0.0f) ? 1.0f / gp.temperature : 1.0f;
    float maxL = -1e30f;
    for (int i = 0; i < V; ++i)
    {
        adj[i] = logits[i] * invT;
        if (adj[i] > maxL) maxL = adj[i];
    }
    // Softmax.
    double Z = 0.0;
    std::vector<double> p(V);
    for (int i = 0; i < V; ++i) { p[i] = std::exp((double)(adj[i] - maxL)); Z += p[i]; }
    for (int i = 0; i < V; ++i) p[i] /= Z;

    // Top-K filter.
    if (gp.topK > 0 && gp.topK < V)
    {
        std::vector<int> idx(V);
        for (int i = 0; i < V; ++i) idx[i] = i;
        CmpByProbDesc cmp(p);
        std::partial_sort(idx.begin(), idx.begin() + gp.topK, idx.end(), cmp);
        std::vector<double> kept(V, 0.0);
        for (int i = 0; i < gp.topK; ++i) kept[idx[i]] = p[idx[i]];
        double s = 0.0;
        for (int i = 0; i < V; ++i) s += kept[i];
        if (s > 0.0) for (int i = 0; i < V; ++i) kept[i] /= s;
        p.swap(kept);
    }

    // Top-P (nucleus) filter.
    if (gp.topP > 0.0f && gp.topP < 1.0f)
    {
        std::vector<int> idx(V);
        for (int i = 0; i < V; ++i) idx[i] = i;
        CmpByProbDesc cmp(p);
        std::sort(idx.begin(), idx.end(), cmp);
        double cum = 0.0;
        std::vector<double> kept(V, 0.0);
        for (int i = 0; i < V; ++i)
        {
            kept[idx[i]] = p[idx[i]];
            cum += p[idx[i]];
            if (cum >= (double)gp.topP) break;
        }
        double s = 0.0;
        for (int i = 0; i < V; ++i) s += kept[i];
        if (s > 0.0) for (int i = 0; i < V; ++i) kept[i] /= s;
        p.swap(kept);
    }

    // Sample: single canonical draw + CDF walk.
    double u = rng.next_canonical_double();
    double cum = 0.0;
    for (int i = 0; i < V; ++i)
    {
        cum += p[i];
        if (u < cum) return i;
    }
    return V - 1;
}

// ---------------------------------------------------------------------------
// chiron_generate — generation loop.
// Ported from chiron_infer.cpp generate-lambda (lines 416-483), minus CLI
// concerns (BPE decode, printf, dumpTokens, genMetrics).
// C++98: lambda comparators become functor structs.
// ---------------------------------------------------------------------------

namespace {
// Descending float comparator for the CHIRON_DBG top-5 partial_sort.
struct CmpByFloatDesc {
    const std::vector<float>& v;
    explicit CmpByFloatDesc(const std::vector<float>& vec) : v(vec) {}
    bool operator()(int a, int b) const { return v[a] > v[b]; }
};
} // anonymous namespace

bool chiron_generate(const ChironModelDims& dims,
                     const ChironModelWeights& w,
                     const ChironServingConfig& cfg,
                     ChironEvalScratch& s,
                     const std::vector<int>& promptTokens,
                     const ChironGenParams& gp,
                     ChironTokenSink sink,
                     void* sinkCtx,
                     std::vector<int>* outTokens)
{
#ifndef GLADES_HAVE_CUDA
    (void)dims; (void)w; (void)cfg; (void)s;
    (void)promptTokens; (void)gp; (void)sink; (void)sinkCtx; (void)outTokens;
    return false;
#else
    // Empty prompt is unsupported — matches chiron_infer CLI behavior.
    if (promptTokens.empty()) return false;

    if (outTokens) outTokens->clear();

    // Working buffer: starts as prompt, grows one token per generation step.
    std::vector<int> tokens(promptTokens);

    // Seeded per generate call. NOTE: the pre-2026-07-03 CLI seeded one
    // process-level mt19937 shared across REPL prompts; one-shot paths are
    // identical, multi-prompt REPL streams differ (acknowledged behavior change).
    ChironMt19937 rng(gp.seed);

    std::vector<int>   input((size_t)dims.T, 0);
    std::vector<float> logitsAll((size_t)dims.T * (size_t)dims.V);
    std::vector<float> logitsRow((size_t)dims.V);

    for (int gen = 0; gen < gp.maxTokens; ++gen)
    {
        // Fill window: pad to T zeros, then place the last min(|tokens|,T)
        // tokens (clamped to [0,V)) starting at position 0.
        const int useLen = (int)tokens.size() < dims.T ? (int)tokens.size() : dims.T;
        for (int i = 0; i < dims.T; ++i) input[i] = 0;
        const int offset = (int)tokens.size() - useLen;
        for (int i = 0; i < useLen; ++i)
        {
            int tk = tokens[offset + i];
            if (tk < 0 || tk >= dims.V) tk = 0;
            input[i] = tk;
        }

        if (!s.d_tokens.upload(&input[0], (size_t)dims.T)) return false;
        if (!chiron_eval_forward(dims, w, cfg, s)) return false;

        // Extract logits row for the last valid position (useLen-1).
        if (!s.logits.download(&logitsAll[0], logitsAll.size())) return false;
        const int lastPos = useLen - 1;
        for (int v = 0; v < dims.V; ++v)
            logitsRow[v] = logitsAll[(size_t)lastPos * (size_t)dims.V + v];

        // CHIRON_DBG: env-gated top-5 logit dump (harmless in lib; matches
        // chiron_infer.cpp lines 462-471 with lambda → functor).
        if (std::getenv("CHIRON_DBG") && gen < 4)
        {
            std::vector<int> idx(dims.V);
            for (int v = 0; v < dims.V; ++v) idx[v] = v;
            const int top5 = dims.V < 5 ? dims.V : 5;
            CmpByFloatDesc cmp(logitsRow);
            std::partial_sort(idx.begin(), idx.begin() + top5, idx.end(), cmp);
            std::printf("[dbg-gen %d] pos=%d top5: ", gen, lastPos);
            for (int j = 0; j < top5; ++j)
                std::printf("tok%d=%.3g  ", idx[j], logitsRow[idx[j]]);
            std::printf("\n");
        }

        // Sample: passes the FULL accumulated tokens (prompt + generated so
        // far) as context — exactly what chiron_infer passed as `context`.
        const int next = chiron_sample_token(logitsRow, gp, tokens, rng);

        // Append to working buffer and record in output.
        tokens.push_back(next);
        if (outTokens) outTokens->push_back(next);

        // Emit to sink.  The refused token is already appended/recorded above;
        // no further iterations run if sink returns false.
        if (sink && !sink(sinkCtx, next)) break;
    }
    return true;
#endif
}

// ---------------------------------------------------------------------------
// chiron_tf_eval — verbatim port of chiron_infer.cpp tf-check block (536-581).
// One forward over the padded window; per-position argmax + double-precision
// log-sum-exp NLL.  logitsAllOut receives the full [T,V] host logits when
// non-null (for --dump-logits callers), avoiding a second forward.
// ---------------------------------------------------------------------------

bool chiron_tf_eval(const ChironModelDims& dims,
                    const ChironModelWeights& w,
                    const ChironServingConfig& cfg,
                    ChironEvalScratch& s,
                    const std::vector<int>& tokens,
                    ChironTfResult& out,
                    std::vector<float>* logitsAllOut)
{
#ifndef GLADES_HAVE_CUDA
    (void)dims; (void)w; (void)cfg; (void)s;
    (void)tokens; (void)out; (void)logitsAllOut;
    return false;
#else
    // Pad / truncate to T.
    const int useLen = ((int)tokens.size() < dims.T) ? (int)tokens.size() : dims.T;
    std::vector<int> input(dims.T, 0);
    for (int i = 0; i < useLen; ++i) input[i] = tokens[i];

    if (!s.d_tokens.upload(&input[0], (size_t)dims.T)) return false;
    if (!chiron_eval_forward(dims, w, cfg, s)) return false;

    // Download the full [T, V] logits to host — either into *logitsAllOut or a
    // local buffer.  The NLL loop reads from whichever vector was filled.
    const size_t nLogits = (size_t)dims.T * (size_t)dims.V;
    std::vector<float>* hostPtr;
    std::vector<float>  localBuf;
    if (logitsAllOut)
    {
        logitsAllOut->resize(nLogits);
        if (!s.logits.download(&(*logitsAllOut)[0], nLogits)) return false;
        hostPtr = logitsAllOut;
    }
    else
    {
        localBuf.resize(nLogits);
        if (!s.logits.download(&localBuf[0], nLogits)) return false;
        hostPtr = &localBuf;
    }

    // Per-position argmax + double-precision log-sum-exp NLL.
    // Matches chiron_infer.cpp:562-574 exactly (same scan order, same types).
    long correct = 0, total = 0;
    double nllSum = 0.0;
    for (int i = 0; i < useLen - 1; ++i)
    {
        const float* row = &(*hostPtr)[(size_t)i * dims.V];
        int am = 0; float best = row[0];
        double mx = row[0];
        for (int v = 1; v < dims.V; ++v) if (row[v] > mx) mx = row[v];
        double Z = 0.0;
        for (int v = 0; v < dims.V; ++v) Z += std::exp((double)row[v] - mx);
        for (int v = 1; v < dims.V; ++v) if (row[v] > best) { best = row[v]; am = v; }
        const int tgt = tokens[i + 1];
        if (am == tgt) ++correct;
        nllSum += -((double)row[tgt] - mx - std::log(Z));
        ++total;
    }

    out.positions = total;
    out.top1Acc   = total ? (double)correct / (double)total : 0.0;
    out.meanNll   = total ? nllSum / (double)total : 0.0;
    return true;
#endif
}

// ---------------------------------------------------------------------------
// chiron_degeneration_metrics — verbatim port of chiron_infer.cpp:57-77.
//   distinct4 = unique 4-grams / total 4-grams  (low => repetitive/collapsed)
//   maxRun    = longest run of identical consecutive tokens (high => stuck)
// Edge cases match the source exactly: empty → distinct4=1.0 maxRun=0;
//   <4 tokens → distinct4=1.0 (no 4-grams).
// ---------------------------------------------------------------------------

void chiron_degeneration_metrics(const std::vector<int>& gen,
                                 double& distinct4,
                                 int& maxRun)
{
    maxRun = gen.empty() ? 0 : 1;
    int run = 1;
    for (size_t i = 1; i < gen.size(); ++i)
    {
        if (gen[i] == gen[i - 1]) { ++run; if (run > maxRun) maxRun = run; }
        else run = 1;
    }
    if (gen.size() < 4) { distinct4 = 1.0; return; }
    std::set<std::string> seen;
    long total = 0;
    char buf[64];
    for (size_t i = 0; i + 3 < gen.size(); ++i)
    {
        snprintf(buf, sizeof(buf), "%d,%d,%d,%d",
                 gen[i], gen[i + 1], gen[i + 2], gen[i + 3]);
        seen.insert(std::string(buf));
        ++total;
    }
    distinct4 = total ? (double)seen.size() / (double)total : 1.0;
}

} // namespace chiron
} // namespace glades
