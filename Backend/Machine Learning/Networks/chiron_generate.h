// chiron_generate.h — CHIRON token generation, sampling, TF eval.
// Single source of truth for CHIRON inference-side generation (2026-07-03 arc).
// C++98.
#ifndef _GLADES_CHIRON_GENERATE_H_
#define _GLADES_CHIRON_GENERATE_H_

#include <stdint.h>
#include <cstddef>
#include <vector>
#include "chiron_serving.h"

namespace glades {
namespace chiron {

// C++98 MT19937 bit-compatible with std::mt19937, plus a canonical-double
// generator bit-compatible with this toolchain's
// std::uniform_real_distribution<double>(0,1) over std::mt19937
// (libstdc++ generate_canonical<double,53>: TOOLCHAIN-COUPLED — pinned by the
// rng-stream goldens in the chiron-generate unit test; a libstdc++ behavior
// change would shift stochastic streams, not correctness).
struct ChironMt19937
{
	uint32_t mt[624];
	int mti;
	explicit ChironMt19937(uint32_t seed);
	uint32_t next_u32();
	double next_canonical_double();   // one variate == one U(0,1) draw of the old sampler
};

struct ChironGenParams
{
	int maxTokens;        // 100
	float temperature;    // 0.8f
	int topK;             // 40
	float topP;           // 0.95f
	int repWindow;        // 256
	float repPenalty;     // 1.0f
	float freqPenalty;    // 1.2f
	float presPenalty;    // 0.4f
	int noRepeatN;        // 3
	uint32_t seed;        // 1337
	ChironGenParams();    // sets exactly the defaults above (today's CLI defaults)
};

// Streaming sink: called once per generated token; return false to stop early.
typedef bool (*ChironTokenSink)(void* ctx, int token);

// Verbatim port of chiron_infer::sampleToken (penalties -> ngram ban ->
// temperature/softmax -> top-k -> top-p -> single CDF draw).
int chiron_sample_token(const std::vector<float>& logitsIn,
                        const ChironGenParams& gp,
                        const std::vector<int>& context,
                        ChironMt19937& rng);

// Generation loop (pad-to-T window, keep-last-T slide, one eval forward per
// token, sample at position useLen-1, append, emit to sink).  promptTokens are
// clamped to [0,V) as today.  Returns false on forward/scratch failure.
// outTokens (optional) receives ONLY the generated tokens.
bool chiron_generate(const ChironModelDims& dims, const ChironModelWeights& w,
                     const ChironServingConfig& cfg, ChironEvalScratch& s,
                     const std::vector<int>& promptTokens,
                     const ChironGenParams& gp,
                     ChironTokenSink sink, void* sinkCtx,
                     std::vector<int>* outTokens);

struct ChironTfResult
{
	long positions;
	double top1Acc;
	double meanNll;
	ChironTfResult() : positions(0), top1Acc(0.0), meanNll(0.0) {}
};

// Teacher-forcing eval: one forward over the (padded) window, per-position
// argmax + double-precision log-sum-exp NLL vs tokens[i+1].
// logitsAllOut (optional): receives the full [T,V] host logits from the
// forward (for --dump-logits) so callers need not run a second forward.
bool chiron_tf_eval(const ChironModelDims& dims, const ChironModelWeights& w,
                    const ChironServingConfig& cfg, ChironEvalScratch& s,
                    const std::vector<int>& tokens,
                    ChironTfResult& out,
                    std::vector<float>* logitsAllOut);

// Pure CPU ARREST detector configuration. Defaults are the frozen detector-v1
// contract; G0a may version these fields before their hashes are frozen.
struct ChironRepetitionConfig
{
	int maxPeriod;
	int minCycleSupport;
	float cycleThreshold;
	int repeatLookback;
	int repeatedSpanWindow;
	int minRepeatedSpan;
	int diversityWindow;
	int maxRunThreshold;
	float repeatedSpanThreshold;
	float distinct1Threshold;
	float distinct4Threshold;
	int ngramWindow;
	int maxHazards;
	int minPeriodHazardSupport;
	float hazardThreshold;
	int runConfidenceSpan;
	int ngramConfidenceCount;
	float postOnsetDecay;
	ChironRepetitionConfig();
};

struct ChironRepetitionMetrics
{
	double distinct1;
	double distinct2;
	double distinct4;
	double repeatFraction;
	double cycleMax;
	double repeatedSpanCoverage;
	int cyclePeriod;
	int longestSuffixCopy;
	int maxRun;
	int collapseOnset;
	bool collapsed;
	ChironRepetitionMetrics();
};

struct ChironHazardRow
{
	int tokenIds[16];
	float confidence[16];
	int count;
	bool overflow;
	ChironHazardRow();
};

// Pure CPU trajectory metrics. No decoded text, logits, corpus labels, model
// state, or GPU state enter this API.
void chiron_repetition_metrics(const std::vector<int>& generated,
                               const ChironRepetitionConfig& config,
                               ChironRepetitionMetrics& out);

// Emits generated.size() strict-prefix rows: row t uses generated[0:t].
void chiron_repetition_hazards(const std::vector<int>& generated,
                               const ChironRepetitionConfig& config,
                               std::vector<ChironHazardRow>& rows,
                               std::vector<float>& rowWeights);

// Compatibility API: distinct4 = unique 4-grams / total; maxRun = longest
// identical-token run. Outputs remain unchanged by the ARREST detector.
void chiron_degeneration_metrics(const std::vector<int>& gen,
                                 double& distinct4, int& maxRun);

} // namespace chiron
} // namespace glades

#endif
