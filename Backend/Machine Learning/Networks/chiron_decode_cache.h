// chiron_decode_cache.h — exact no-slide incremental decode for causal SCFA CHIRON.
//
// Correctness-first API. The cache is bound to one model/configuration and
// rejects dense attention, legacy non-causal SCFA, geometry changes, and
// positions at or beyond the native context limit.
//
// C++98.
#ifndef _GLADES_CHIRON_DECODE_CACHE_H_
#define _GLADES_CHIRON_DECODE_CACHE_H_

#include "chiron_serving.h"

#include <vector>

namespace glades {
namespace chiron {

class ChironDecodeSnapshot;

class ChironDecodeCache
{
public:
    ChironDecodeCache();
    ~ChironDecodeCache();

    bool allocate(const ChironModelDims& d, const ChironModelWeights& w,
                  const ChironServingConfig& cfg);
    bool reset();
    bool cloneFrom(const ChironDecodeCache& source);

    bool ready() const;
    int position() const;
    int completedBlocks() const;
    int contextLimit() const;
    bool slidingSupported() const { return false; }

private:
    struct Impl;
    Impl* impl_;

    ChironDecodeCache(const ChironDecodeCache&);
    ChironDecodeCache& operator=(const ChironDecodeCache&);

    friend bool chiron_decode_prefill(const ChironModelDims&,
                                      const ChironModelWeights&,
                                      const ChironServingConfig&,
                                      ChironEvalScratch&,
                                      const std::vector<int>&,
                                      ChironDecodeCache&);
    friend bool chiron_decode_step(const ChironModelDims&,
                                   const ChironModelWeights&,
                                   const ChironServingConfig&,
                                   ChironEvalScratch&, int,
                                   ChironDecodeCache&);
    friend class ChironDecodeSnapshot;
};

// Lightweight branch point for one prefilled cache. Completed q/K/V prefix
// rows remain immutable in the owning cache; only the mutable partial block,
// convolution rings, delayed y row, metadata, and prompt logits are copied.
// A snapshot restores only its original cache instance.
class ChironDecodeSnapshot
{
public:
    ChironDecodeSnapshot();
    ~ChironDecodeSnapshot();
    bool capture(ChironDecodeCache& cache, ChironEvalScratch& scratch);
    bool restore(ChironDecodeCache& cache, ChironEvalScratch& scratch) const;
    bool ready() const;
private:
    struct Impl;
    Impl* impl_;
    ChironDecodeSnapshot(const ChironDecodeSnapshot&);
    ChironDecodeSnapshot& operator=(const ChironDecodeSnapshot&);
};

// Resets cache, consumes a non-empty prompt shorter than d.T, and leaves logits
// for the final prompt token in s.logits[0..V). Input IDs outside [0,V) are
// clamped to token 0, matching chiron_generate.
bool chiron_decode_prefill(const ChironModelDims& d,
                           const ChironModelWeights& w,
                           const ChironServingConfig& cfg,
                           ChironEvalScratch& s,
                           const std::vector<int>& promptTokens,
                           ChironDecodeCache& cache);

// Consumes one token and leaves its logits in s.logits[0..V). Fails without
// advancing when the native no-slide context is full or cache/model geometry
// does not match.
bool chiron_decode_step(const ChironModelDims& d,
                        const ChironModelWeights& w,
                        const ChironServingConfig& cfg,
                        ChironEvalScratch& s,
                        int token,
                        ChironDecodeCache& cache);

} // namespace chiron
} // namespace glades

#endif
