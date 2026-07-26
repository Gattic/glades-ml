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
// Pure CPU ARREST repetition detector (frozen detector contract v1).
// ---------------------------------------------------------------------------

namespace {

const char REPETITION_CONFIG_FORMAT[] = "chiron-arrest-detector-config";
const uint64_t REPETITION_CONFIG_SCALE = UINT64_C(1000000);
const int REPETITION_CONFIG_LIMIT = 1048576;

void repetition_config_set_error(std::string* error, const std::string& message)
{
    if (error) *error = message;
}

bool repetition_config_int_range(const char* name, int value,
                                 int minimum, int maximum,
                                 std::string* error)
{
    if (value >= minimum && value <= maximum) return true;
    repetition_config_set_error(error, std::string(name) + " is out of range");
    return false;
}

bool repetition_config_float_units(const char* name, float value,
                                   uint64_t& units, std::string* error)
{
    if (!(value == value) || value < 0.0f ||
        value > (float)REPETITION_CONFIG_LIMIT)
    {
        repetition_config_set_error(error, std::string(name) + " is not finite and bounded");
        return false;
    }
    const double scaled = (double)value * (double)REPETITION_CONFIG_SCALE;
    const double rounded = std::floor(scaled + 0.5);
    units = (uint64_t)rounded;
    const float reconstructed = (float)((double)units /
                                        (double)REPETITION_CONFIG_SCALE);
    if (reconstructed != value)
    {
        repetition_config_set_error(error, std::string(name) +
                                           " is not canonical to six decimal places");
        return false;
    }
    return true;
}

bool repetition_config_float_range(const char* name, float value,
                                   float minimumExclusive, float maximumInclusive,
                                   std::string* error)
{
    uint64_t units = 0;
    if (!repetition_config_float_units(name, value, units, error)) return false;
    if (value > minimumExclusive && value <= maximumInclusive) return true;
    repetition_config_set_error(error, std::string(name) + " is out of range");
    return false;
}

void repetition_append_uint(std::string& out, uint64_t value)
{
    char digits[32];
    int count = 0;
    do
    {
        digits[count++] = (char)('0' + (value % 10));
        value /= 10;
    }
    while (value != 0);
    while (count > 0) out.push_back(digits[--count]);
}

void repetition_append_int_line(std::string& out, const char* name, int value)
{
    out += name;
    out.push_back('=');
    repetition_append_uint(out, (uint64_t)value);
    out.push_back('\n');
}

void repetition_append_float_line(std::string& out, const char* name, float value)
{
    uint64_t units = 0;
    repetition_config_float_units(name, value, units, NULL);
    out += name;
    out.push_back('=');
    repetition_append_uint(out, units / REPETITION_CONFIG_SCALE);
    out.push_back('.');
    uint64_t fraction = units % REPETITION_CONFIG_SCALE;
    uint64_t divisor = UINT64_C(100000);
    for (int i = 0; i < 6; ++i)
    {
        out.push_back((char)('0' + (fraction / divisor) % 10));
        divisor /= 10;
    }
    out.push_back('\n');
}

void repetition_config_serialize_valid(const ChironRepetitionConfig& config,
                                       std::string& out)
{
    out.clear();
    out += "format=";
    out += REPETITION_CONFIG_FORMAT;
    out += "\nversion=";
    repetition_append_uint(out, CHIRON_REPETITION_CONFIG_VERSION);
    out.push_back('\n');
    repetition_append_int_line(out, "maxPeriod", config.maxPeriod);
    repetition_append_int_line(out, "minCycleSupport", config.minCycleSupport);
    repetition_append_float_line(out, "cycleThreshold", config.cycleThreshold);
    repetition_append_int_line(out, "repeatLookback", config.repeatLookback);
    repetition_append_int_line(out, "repeatedSpanWindow", config.repeatedSpanWindow);
    repetition_append_int_line(out, "minRepeatedSpan", config.minRepeatedSpan);
    repetition_append_int_line(out, "diversityWindow", config.diversityWindow);
    repetition_append_int_line(out, "maxRunThreshold", config.maxRunThreshold);
    repetition_append_float_line(out, "repeatedSpanThreshold", config.repeatedSpanThreshold);
    repetition_append_float_line(out, "distinct1Threshold", config.distinct1Threshold);
    repetition_append_float_line(out, "distinct4Threshold", config.distinct4Threshold);
    repetition_append_int_line(out, "ngramWindow", config.ngramWindow);
    repetition_append_int_line(out, "maxHazards", config.maxHazards);
    repetition_append_int_line(out, "minPeriodHazardSupport", config.minPeriodHazardSupport);
    repetition_append_float_line(out, "hazardThreshold", config.hazardThreshold);
    repetition_append_int_line(out, "runConfidenceSpan", config.runConfidenceSpan);
    repetition_append_int_line(out, "ngramConfidenceCount", config.ngramConfidenceCount);
    repetition_append_float_line(out, "postOnsetDecay", config.postOnsetDecay);
}

bool repetition_parse_uint(const std::string& text, int& value)
{
    if (text.empty() || (text.size() > 1 && text[0] == '0')) return false;
    uint64_t parsed = 0;
    for (size_t i = 0; i < text.size(); ++i)
    {
        if (text[i] < '0' || text[i] > '9') return false;
        parsed = parsed * 10 + (uint64_t)(text[i] - '0');
        if (parsed > UINT64_C(2147483647)) return false;
    }
    value = (int)parsed;
    return true;
}

bool repetition_parse_float6(const std::string& text, float& value)
{
    const size_t dot = text.find('.');
    if (dot == std::string::npos || dot == 0 || text.size() - dot - 1 != 6)
        return false;
    const std::string wholeText = text.substr(0, dot);
    if (wholeText.size() > 1 && wholeText[0] == '0') return false;
    uint64_t whole = 0;
    for (size_t i = 0; i < wholeText.size(); ++i)
    {
        if (wholeText[i] < '0' || wholeText[i] > '9') return false;
        whole = whole * 10 + (uint64_t)(wholeText[i] - '0');
        if (whole > (uint64_t)REPETITION_CONFIG_LIMIT) return false;
    }
    uint64_t fraction = 0;
    for (size_t i = dot + 1; i < text.size(); ++i)
    {
        if (text[i] < '0' || text[i] > '9') return false;
        fraction = fraction * 10 + (uint64_t)(text[i] - '0');
    }
    const uint64_t units = whole * REPETITION_CONFIG_SCALE + fraction;
    value = (float)((double)units / (double)REPETITION_CONFIG_SCALE);
    return true;
}

bool repetition_parse_int_line(const std::string& line, const char* name,
                               int& value, std::string* error)
{
    const std::string prefix = std::string(name) + "=";
    if (line.compare(0, prefix.size(), prefix) != 0 ||
        !repetition_parse_uint(line.substr(prefix.size()), value))
    {
        repetition_config_set_error(error, std::string("invalid canonical field: ") + name);
        return false;
    }
    return true;
}

bool repetition_parse_float_line(const std::string& line, const char* name,
                                 float& value, std::string* error)
{
    const std::string prefix = std::string(name) + "=";
    if (line.compare(0, prefix.size(), prefix) != 0 ||
        !repetition_parse_float6(line.substr(prefix.size()), value))
    {
        repetition_config_set_error(error, std::string("invalid canonical field: ") + name);
        return false;
    }
    return true;
}

uint32_t repetition_sha_rotr(uint32_t value, int bits)
{
    return (value >> bits) | (value << (32 - bits));
}

struct RepetitionSha256
{
    uint32_t state[8];
    uint64_t totalBytes;
    unsigned char block[64];
    size_t used;
};

void repetition_sha_transform(RepetitionSha256& sha, const unsigned char* block)
{
    static const uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    uint32_t words[64];
    for (int i = 0; i < 16; ++i)
    {
        words[i] = ((uint32_t)block[4 * i] << 24) |
                   ((uint32_t)block[4 * i + 1] << 16) |
                   ((uint32_t)block[4 * i + 2] << 8) |
                   (uint32_t)block[4 * i + 3];
    }
    for (int i = 16; i < 64; ++i)
    {
        const uint32_t s0 = repetition_sha_rotr(words[i - 15], 7) ^
                            repetition_sha_rotr(words[i - 15], 18) ^
                            (words[i - 15] >> 3);
        const uint32_t s1 = repetition_sha_rotr(words[i - 2], 17) ^
                            repetition_sha_rotr(words[i - 2], 19) ^
                            (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    uint32_t a = sha.state[0], b = sha.state[1], c = sha.state[2], d = sha.state[3];
    uint32_t e = sha.state[4], f = sha.state[5], g = sha.state[6], h = sha.state[7];
    for (int i = 0; i < 64; ++i)
    {
        const uint32_t sum1 = repetition_sha_rotr(e, 6) ^
                              repetition_sha_rotr(e, 11) ^
                              repetition_sha_rotr(e, 25);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t t1 = h + sum1 + choose + constants[i] + words[i];
        const uint32_t sum0 = repetition_sha_rotr(a, 2) ^
                              repetition_sha_rotr(a, 13) ^
                              repetition_sha_rotr(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = sum0 + majority;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    sha.state[0] += a; sha.state[1] += b; sha.state[2] += c; sha.state[3] += d;
    sha.state[4] += e; sha.state[5] += f; sha.state[6] += g; sha.state[7] += h;
}

void repetition_sha_init(RepetitionSha256& sha)
{
    const uint32_t initial[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    for (int i = 0; i < 8; ++i) sha.state[i] = initial[i];
    sha.totalBytes = 0;
    sha.used = 0;
}

void repetition_sha_update(RepetitionSha256& sha,
                           const unsigned char* data, size_t length)
{
    sha.totalBytes += (uint64_t)length;
    while (length > 0)
    {
        const size_t room = 64 - sha.used;
        const size_t take = std::min(room, length);
        std::memcpy(sha.block + sha.used, data, take);
        sha.used += take;
        data += take;
        length -= take;
        if (sha.used == 64)
        {
            repetition_sha_transform(sha, sha.block);
            sha.used = 0;
        }
    }
}

void repetition_sha_finish(RepetitionSha256& sha, unsigned char digest[32])
{
    const uint64_t bitLength = sha.totalBytes * UINT64_C(8);
    sha.block[sha.used++] = 0x80u;
    if (sha.used > 56)
    {
        while (sha.used < 64) sha.block[sha.used++] = 0;
        repetition_sha_transform(sha, sha.block);
        sha.used = 0;
    }
    while (sha.used < 56) sha.block[sha.used++] = 0;
    for (int i = 7; i >= 0; --i)
        sha.block[sha.used++] = (unsigned char)(bitLength >> (8 * i));
    repetition_sha_transform(sha, sha.block);
    for (int i = 0; i < 8; ++i)
    {
        digest[4 * i] = (unsigned char)(sha.state[i] >> 24);
        digest[4 * i + 1] = (unsigned char)(sha.state[i] >> 16);
        digest[4 * i + 2] = (unsigned char)(sha.state[i] >> 8);
        digest[4 * i + 3] = (unsigned char)sha.state[i];
    }
}

void repetition_sha256_hex(const std::string& bytes, std::string& hex)
{
    RepetitionSha256 sha;
    repetition_sha_init(sha);
    repetition_sha_update(sha, (const unsigned char*)bytes.data(), bytes.size());
    unsigned char digest[32];
    repetition_sha_finish(sha, digest);
    static const char digits[] = "0123456789abcdef";
    hex.clear();
    hex.reserve(64);
    for (int i = 0; i < 32; ++i)
    {
        hex.push_back(digits[digest[i] >> 4]);
        hex.push_back(digits[digest[i] & 15]);
    }
}

} // anonymous namespace

ChironRepetitionConfig::ChironRepetitionConfig()
    : maxPeriod(64)
    , minCycleSupport(32)
    , cycleThreshold(0.80f)
    , repeatLookback(64)
    , repeatedSpanWindow(128)
    , minRepeatedSpan(8)
    , diversityWindow(64)
    , maxRunThreshold(8)
    , repeatedSpanThreshold(0.60f)
    , distinct1Threshold(0.15f)
    , distinct4Threshold(0.35f)
    , ngramWindow(128)
    , maxHazards(16)
    , minPeriodHazardSupport(8)
    , hazardThreshold(0.70f)
    , runConfidenceSpan(8)
    , ngramConfidenceCount(4)
    , postOnsetDecay(32.0f)
{
}

bool chiron_repetition_config_validate(const ChironRepetitionConfig& config,
                                       std::string* error)
{
    if (error) error->clear();
    if (!repetition_config_int_range("maxPeriod", config.maxPeriod, 1, 64, error) ||
        !repetition_config_int_range("minCycleSupport", config.minCycleSupport,
                                     1, REPETITION_CONFIG_LIMIT, error) ||
        !repetition_config_float_range("cycleThreshold", config.cycleThreshold,
                                       0.0f, 1.0f, error) ||
        !repetition_config_int_range("repeatLookback", config.repeatLookback,
                                     1, REPETITION_CONFIG_LIMIT, error) ||
        !repetition_config_int_range("repeatedSpanWindow", config.repeatedSpanWindow,
                                     1, REPETITION_CONFIG_LIMIT, error) ||
        !repetition_config_int_range("minRepeatedSpan", config.minRepeatedSpan,
                                     1, REPETITION_CONFIG_LIMIT, error) ||
        !repetition_config_int_range("diversityWindow", config.diversityWindow,
                                     4, REPETITION_CONFIG_LIMIT, error) ||
        !repetition_config_int_range("maxRunThreshold", config.maxRunThreshold,
                                     1, REPETITION_CONFIG_LIMIT, error) ||
        !repetition_config_float_range("repeatedSpanThreshold",
                                       config.repeatedSpanThreshold, 0.0f, 1.0f, error) ||
        !repetition_config_float_range("distinct1Threshold", config.distinct1Threshold,
                                       0.0f, 1.0f, error) ||
        !repetition_config_float_range("distinct4Threshold", config.distinct4Threshold,
                                       0.0f, 1.0f, error) ||
        !repetition_config_int_range("ngramWindow", config.ngramWindow,
                                     2, REPETITION_CONFIG_LIMIT, error) ||
        !repetition_config_int_range("maxHazards", config.maxHazards, 1, 16, error) ||
        !repetition_config_int_range("minPeriodHazardSupport",
                                     config.minPeriodHazardSupport,
                                     1, REPETITION_CONFIG_LIMIT, error) ||
        !repetition_config_float_range("hazardThreshold", config.hazardThreshold,
                                       0.0f, 1.0f, error) ||
        !repetition_config_int_range("runConfidenceSpan", config.runConfidenceSpan,
                                     1, REPETITION_CONFIG_LIMIT, error) ||
        !repetition_config_int_range("ngramConfidenceCount", config.ngramConfidenceCount,
                                     2, REPETITION_CONFIG_LIMIT, error) ||
        !repetition_config_float_range("postOnsetDecay", config.postOnsetDecay,
                                       0.0f, (float)REPETITION_CONFIG_LIMIT, error))
        return false;

    if (config.minRepeatedSpan > config.repeatedSpanWindow)
    {
        repetition_config_set_error(error,
            "minRepeatedSpan exceeds repeatedSpanWindow");
        return false;
    }
    const int maximumPeriodEvidence =
        std::max(config.minCycleSupport, 2 * config.maxPeriod);
    if (config.minPeriodHazardSupport > maximumPeriodEvidence)
    {
        repetition_config_set_error(error,
            "minPeriodHazardSupport exceeds available configured support");
        return false;
    }
    return true;
}

bool chiron_repetition_config_serialize(const ChironRepetitionConfig& config,
                                        std::string& canonicalBytes,
                                        std::string* error)
{
    canonicalBytes.clear();
    if (!chiron_repetition_config_validate(config, error)) return false;
    repetition_config_serialize_valid(config, canonicalBytes);
    return true;
}

bool chiron_repetition_config_parse(const std::string& canonicalBytes,
                                    ChironRepetitionConfig& config,
                                    std::string* error)
{
    if (canonicalBytes.empty() || canonicalBytes.size() > 4096 ||
        canonicalBytes[canonicalBytes.size() - 1] != '\n' ||
        canonicalBytes.find('\r') != std::string::npos)
    {
        repetition_config_set_error(error,
            "config must be at most 4096 bytes, LF-only, and LF-terminated");
        return false;
    }
    // Copy only after the size bound so canonicalBytes may safely alias error
    // without allowing malformed oversized input to force a second allocation.
    const std::string input(canonicalBytes);
    if (error) error->clear();

    std::vector<std::string> lines;
    size_t begin = 0;
    while (begin < input.size())
    {
        const size_t end = input.find('\n', begin);
        if (end == std::string::npos)
        {
            repetition_config_set_error(error, "unterminated config line");
            return false;
        }
        lines.push_back(input.substr(begin, end - begin));
        begin = end + 1;
    }
    if (lines.size() != 20)
    {
        repetition_config_set_error(error, "config must contain exactly 20 lines");
        return false;
    }
    if (lines[0] != std::string("format=") + REPETITION_CONFIG_FORMAT)
    {
        repetition_config_set_error(error, "unsupported detector config format");
        return false;
    }
    std::string expectedVersion("version=");
    repetition_append_uint(expectedVersion, CHIRON_REPETITION_CONFIG_VERSION);
    if (lines[1] != expectedVersion)
    {
        repetition_config_set_error(error, "unsupported detector config version");
        return false;
    }

    ChironRepetitionConfig parsed;
    if (!repetition_parse_int_line(lines[2], "maxPeriod", parsed.maxPeriod, error) ||
        !repetition_parse_int_line(lines[3], "minCycleSupport",
                                   parsed.minCycleSupport, error) ||
        !repetition_parse_float_line(lines[4], "cycleThreshold",
                                     parsed.cycleThreshold, error) ||
        !repetition_parse_int_line(lines[5], "repeatLookback",
                                   parsed.repeatLookback, error) ||
        !repetition_parse_int_line(lines[6], "repeatedSpanWindow",
                                   parsed.repeatedSpanWindow, error) ||
        !repetition_parse_int_line(lines[7], "minRepeatedSpan",
                                   parsed.minRepeatedSpan, error) ||
        !repetition_parse_int_line(lines[8], "diversityWindow",
                                   parsed.diversityWindow, error) ||
        !repetition_parse_int_line(lines[9], "maxRunThreshold",
                                   parsed.maxRunThreshold, error) ||
        !repetition_parse_float_line(lines[10], "repeatedSpanThreshold",
                                     parsed.repeatedSpanThreshold, error) ||
        !repetition_parse_float_line(lines[11], "distinct1Threshold",
                                     parsed.distinct1Threshold, error) ||
        !repetition_parse_float_line(lines[12], "distinct4Threshold",
                                     parsed.distinct4Threshold, error) ||
        !repetition_parse_int_line(lines[13], "ngramWindow",
                                   parsed.ngramWindow, error) ||
        !repetition_parse_int_line(lines[14], "maxHazards",
                                   parsed.maxHazards, error) ||
        !repetition_parse_int_line(lines[15], "minPeriodHazardSupport",
                                   parsed.minPeriodHazardSupport, error) ||
        !repetition_parse_float_line(lines[16], "hazardThreshold",
                                     parsed.hazardThreshold, error) ||
        !repetition_parse_int_line(lines[17], "runConfidenceSpan",
                                   parsed.runConfidenceSpan, error) ||
        !repetition_parse_int_line(lines[18], "ngramConfidenceCount",
                                   parsed.ngramConfidenceCount, error) ||
        !repetition_parse_float_line(lines[19], "postOnsetDecay",
                                     parsed.postOnsetDecay, error))
        return false;

    if (!chiron_repetition_config_validate(parsed, error)) return false;
    std::string regenerated;
    repetition_config_serialize_valid(parsed, regenerated);
    if (regenerated != input)
    {
        repetition_config_set_error(error, "config bytes are not canonical");
        return false;
    }
    config = parsed;
    return true;
}

bool chiron_repetition_config_sha256(const ChironRepetitionConfig& config,
                                     std::string& lowercaseHex,
                                     std::string* error)
{
    lowercaseHex.clear();
    std::string canonical;
    if (!chiron_repetition_config_serialize(config, canonical, error)) return false;
    repetition_sha256_hex(canonical, lowercaseHex);
    return true;
}

ChironRepetitionMetrics::ChironRepetitionMetrics()
    : distinct1(1.0)
    , distinct2(1.0)
    , distinct4(1.0)
    , repeatFraction(0.0)
    , cycleMax(0.0)
    , repeatedSpanCoverage(0.0)
    , cyclePeriod(0)
    , longestSuffixCopy(0)
    , maxRun(0)
    , collapseOnset(-1)
    , collapsed(false)
{
}

ChironHazardRow::ChironHazardRow()
    : count(0), overflow(false)
{
    for (int i = 0; i < 16; ++i)
    {
        tokenIds[i] = -1;
        confidence[i] = 0.0f;
    }
}

namespace {

double repetition_distinct_range(const std::vector<int>& x,
                                  int begin, int end, int ngram)
{
    const int length = end - begin;
    if (ngram <= 0 || length < ngram) return 1.0;
    std::set< std::vector<int> > unique;
    for (int i = begin; i + ngram <= end; ++i)
        unique.insert(std::vector<int>(x.begin() + i, x.begin() + i + ngram));
    const int total = length - ngram + 1;
    return total > 0 ? (double)unique.size() / (double)total : 1.0;
}

void repetition_lcp_table(const std::vector<int>& x,
                          std::vector< std::vector<int> >& lcp)
{
    const int n = (int)x.size();
    lcp.assign((size_t)n + 1, std::vector<int>((size_t)n + 1, 0));
    for (int i = n - 1; i >= 0; --i)
        for (int j = n - 1; j >= 0; --j)
            if (x[i] == x[j]) lcp[i][j] = 1 + lcp[i + 1][j + 1];
}

double repetition_span_coverage(const std::vector< std::vector<int> >& lcp,
                                int prefixLength,
                                const ChironRepetitionConfig& config)
{
    if (prefixLength <= 0 || config.repeatedSpanWindow <= 0 ||
        config.minRepeatedSpan <= 0)
        return 0.0;

    const int windowBegin = std::max(0, prefixLength - config.repeatedSpanWindow);
    const int windowLength = prefixLength - windowBegin;
    std::vector<int> difference((size_t)windowLength + 1, 0);

    for (int current = windowBegin; current < prefixLength; ++current)
    {
        for (int earlier = 0; earlier < current; ++earlier)
        {
            int length = lcp[earlier][current];
            const int available = prefixLength - current;
            if (length > available) length = available;
            if (length < config.minRepeatedSpan) continue;

            int a0 = std::max(windowBegin, earlier);
            int a1 = std::min(prefixLength, earlier + length);
            if (a0 < a1)
            {
                ++difference[a0 - windowBegin];
                --difference[a1 - windowBegin];
            }
            int b0 = current;
            int b1 = std::min(prefixLength, current + length);
            if (b0 < b1)
            {
                ++difference[b0 - windowBegin];
                --difference[b1 - windowBegin];
            }
        }
    }

    int active = 0, covered = 0;
    for (int i = 0; i < windowLength; ++i)
    {
        active += difference[i];
        if (active > 0) ++covered;
    }
    return windowLength > 0 ? (double)covered / (double)windowLength : 0.0;
}

double repetition_cycle_score(const std::vector<int>& x,
                              int prefixLength, int period, int support)
{
    if (period <= 0 || support <= 0 || prefixLength < period + support)
        return -1.0;
    const int start = prefixLength - support;
    int matches = 0;
    for (int i = start; i < prefixLength; ++i)
        if (x[i] == x[i - period]) ++matches;
    return (double)matches / (double)support;
}

int repetition_longest_suffix_copy(const std::vector< std::vector<int> >& lcp,
                                   int length)
{
    int best = 0;
    for (int suffix = 1; suffix < length; ++suffix)
    {
        const int suffixLength = length - suffix;
        if (suffixLength <= best) continue;
        for (int earlier = 0; earlier < suffix; ++earlier)
        {
            if (lcp[earlier][suffix] >= suffixLength)
            {
                best = suffixLength;
                break;
            }
        }
    }
    return best;
}

struct RepetitionCandidate
{
    int token;
    float confidence;
    RepetitionCandidate(int t, float c) : token(t), confidence(c) {}
};

void repetition_add_candidate(std::vector<RepetitionCandidate>& candidates,
                              int token, float confidence)
{
    if (confidence < 0.0f) confidence = 0.0f;
    if (confidence > 1.0f) confidence = 1.0f;
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        if (candidates[i].token == token)
        {
            if (confidence > candidates[i].confidence)
                candidates[i].confidence = confidence;
            return;
        }
    }
    candidates.push_back(RepetitionCandidate(token, confidence));
}

struct RepetitionCandidateOrder
{
    bool operator()(const RepetitionCandidate& a,
                    const RepetitionCandidate& b) const
    {
        if (a.confidence != b.confidence) return a.confidence > b.confidence;
        return a.token < b.token;
    }
};

int repetition_suffix_run(const std::vector<int>& x, int prefixLength)
{
    if (prefixLength <= 0) return 0;
    int run = 1;
    for (int i = prefixLength - 1; i > 0 && x[i] == x[i - 1]; --i) ++run;
    return run;
}

} // anonymous namespace

void chiron_repetition_metrics(const std::vector<int>& generated,
                               const ChironRepetitionConfig& config,
                               ChironRepetitionMetrics& out)
{
    out = ChironRepetitionMetrics();
    const int n = (int)generated.size();
    out.distinct1 = repetition_distinct_range(generated, 0, n, 1);
    out.distinct2 = repetition_distinct_range(generated, 0, n, 2);
    out.distinct4 = repetition_distinct_range(generated, 0, n, 4);
    if (n == 0) return;

    out.maxRun = 1;
    int run = 1, repeated = 0;
    for (int i = 0; i < n; ++i)
    {
        if (i > 0)
        {
            if (generated[i] == generated[i - 1]) ++run;
            else run = 1;
            if (run > out.maxRun) out.maxRun = run;
        }
        if (config.repeatLookback > 0)
        {
            const int begin = std::max(0, i - config.repeatLookback);
            for (int j = begin; j < i; ++j)
            {
                if (generated[j] == generated[i]) { ++repeated; break; }
            }
        }
    }
    out.repeatFraction = (double)repeated / (double)n;

    std::vector< std::vector<int> > lcp;
    repetition_lcp_table(generated, lcp);
    out.longestSuffixCopy = repetition_longest_suffix_copy(lcp, n);

    int currentRun = 0;
    const int maxPeriod = std::max(0, std::min(64, config.maxPeriod));
    for (int prefix = 1; prefix <= n; ++prefix)
    {
        if (prefix == 1 || generated[prefix - 1] != generated[prefix - 2]) currentRun = 1;
        else ++currentRun;

        bool cycleCollapsed = false;
        if (maxPeriod > 0 && config.minCycleSupport > 0)
        {
            for (int period = 1; period <= maxPeriod; ++period)
            {
                const int support = std::max(config.minCycleSupport, 2 * period);
                const double score = repetition_cycle_score(generated, prefix, period, support);
                if (score < 0.0) continue;
                if (score > out.cycleMax ||
                    (score == out.cycleMax &&
                     (out.cyclePeriod == 0 || period < out.cyclePeriod)))
                {
                    out.cycleMax = score;
                    out.cyclePeriod = period;
                }
                if (score >= (double)config.cycleThreshold) cycleCollapsed = true;
            }
        }

        const double coverage = repetition_span_coverage(lcp, prefix, config);
        if (coverage > out.repeatedSpanCoverage) out.repeatedSpanCoverage = coverage;

        bool diversityCollapsed = false;
        if (config.diversityWindow > 0 && prefix >= config.diversityWindow)
        {
            const int begin = prefix - config.diversityWindow;
            const double d1 = repetition_distinct_range(generated, begin, prefix, 1);
            const double d4 = repetition_distinct_range(generated, begin, prefix, 4);
            diversityCollapsed = d1 <= (double)config.distinct1Threshold &&
                                 d4 <= (double)config.distinct4Threshold;
        }

        const bool runCollapsed = config.maxRunThreshold > 0 &&
                                  currentRun >= config.maxRunThreshold;
        const bool spanCollapsed = config.repeatedSpanWindow > 0 &&
                                   config.minRepeatedSpan > 0 &&
                                   coverage >= (double)config.repeatedSpanThreshold;
        if (out.collapseOnset < 0 &&
            (runCollapsed || cycleCollapsed || spanCollapsed || diversityCollapsed))
            out.collapseOnset = prefix - 1;
    }
    out.collapsed = out.collapseOnset >= 0;
}

void chiron_repetition_hazards(const std::vector<int>& generated,
                               const ChironRepetitionConfig& config,
                               std::vector<ChironHazardRow>& rows,
                               std::vector<float>& rowWeights)
{
    const int n = (int)generated.size();
    rows.assign((size_t)n, ChironHazardRow());
    rowWeights.assign((size_t)n, 0.0f);

    ChironRepetitionMetrics metrics;
    chiron_repetition_metrics(generated, config, metrics);
    const int maxPeriod = std::max(0, std::min(64, config.maxPeriod));
    const int effectiveCap = std::max(0, std::min(16, config.maxHazards));

    for (int t = 0; t < n; ++t)
    {
        std::vector<RepetitionCandidate> candidates;

        const int suffixRun = repetition_suffix_run(generated, t);
        if (suffixRun >= 2 && config.runConfidenceSpan > 0)
        {
            float confidence = (float)suffixRun / (float)config.runConfidenceSpan;
            repetition_add_candidate(candidates, generated[t - 1], confidence);
        }

        if (maxPeriod > 0 && config.minCycleSupport > 0 &&
            config.minPeriodHazardSupport > 0)
        {
            for (int period = 1; period <= maxPeriod && period < t; ++period)
            {
                const int target = std::max(config.minCycleSupport, 2 * period);
                const int available = std::min(target, t - period);
                if (available < config.minPeriodHazardSupport) continue;
                const double score = repetition_cycle_score(generated, t, period, available);
                if (score >= (double)config.hazardThreshold)
                    repetition_add_candidate(candidates, generated[t - period], (float)score);
            }
        }

        if (config.ngramWindow > 0 && config.ngramConfidenceCount > 0)
        {
            const int windowBegin = std::max(0, t - config.ngramWindow);
            for (int ngram = 2; ngram <= 8; ++ngram)
            {
                const int suffixLength = ngram - 1;
                const int suffixBegin = t - suffixLength;
                if (suffixBegin < windowBegin) continue;
                std::vector<RepetitionCandidate> counts;
                for (int i = windowBegin; i + ngram <= t; ++i)
                {
                    bool match = true;
                    for (int j = 0; j < suffixLength; ++j)
                    {
                        if (generated[i + j] != generated[suffixBegin + j])
                        {
                            match = false;
                            break;
                        }
                    }
                    if (!match) continue;
                    const int token = generated[i + suffixLength];
                    bool found = false;
                    for (size_t c = 0; c < counts.size(); ++c)
                    {
                        if (counts[c].token == token)
                        {
                            counts[c].confidence += 1.0f;
                            found = true;
                            break;
                        }
                    }
                    if (!found) counts.push_back(RepetitionCandidate(token, 1.0f));
                }
                for (size_t c = 0; c < counts.size(); ++c)
                {
                    const int occurrences = (int)counts[c].confidence;
                    if (occurrences < 2) continue;
                    const float confidence = (float)occurrences /
                                             (float)config.ngramConfidenceCount;
                    repetition_add_candidate(candidates, counts[c].token, confidence);
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), RepetitionCandidateOrder());
        ChironHazardRow& row = rows[t];
        row.overflow = (int)candidates.size() > effectiveCap;
        row.count = std::min((int)candidates.size(), effectiveCap);
        for (int i = 0; i < row.count; ++i)
        {
            row.tokenIds[i] = candidates[i].token;
            row.confidence[i] = candidates[i].confidence;
        }

        const float rowConfidence = candidates.empty() ? 0.0f : candidates[0].confidence;
        if (metrics.collapseOnset >= 0 && t > metrics.collapseOnset)
        {
            if (config.postOnsetDecay > 0.0f)
            {
                const int age = t - (metrics.collapseOnset + 1);
                rowWeights[t] = rowConfidence *
                    (float)std::exp(-(double)age / (double)config.postOnsetDecay);
            }
        }
        else if (rowConfidence >= config.hazardThreshold)
            rowWeights[t] = rowConfidence;
    }
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
