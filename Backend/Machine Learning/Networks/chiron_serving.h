// chiron_serving.h — CHIRON serving feature-resolution module.
//
// Applies the serving decision table: turns (checkpoint contents + CLI overrides)
// into a validated ChironServingConfig.  Ported from chiron_infer.cpp:1040–1234.
//
// Does NOT implement the eval forward (Task 6).
//
// C++98.

#ifndef _GLADES_CHIRON_SERVING_H_
#define _GLADES_CHIRON_SERVING_H_

#include "chiron_checkpoint.h"

#include <string>
#include <vector>

namespace glades {
namespace chiron {

// CLI/caller overrides that control feature resolution.
struct ChironServingOverrides
{
    int   fuseAttnPerLayer;   // -1 unset (default-on heuristic applies), 0 --no-fuse-attn, 1 --fuse-attn-per-layer
    bool  fuseAttnReln;       // --fuse-attn-reln
    bool  qkNorm;             // --qk-norm
    float qkNormGamma;        // --qk-norm-gamma; <=0 => auto log2(T)
    int   scfaForceMode;      // 0 auto, 1 --scfa, -1 --no-scfa
    int   scfaKOverride;      // 0 = checkpoint's k
    int   scfaWOverride;      // -1 = checkpoint's w
    bool  whiscCoupling;      // --whisc-coupling
    float rotThetaMax;        // default 0.07f (flagship recipe)
    float whiscClamp;         // default 8.0f
    int   seqLen;             // 0 = checkpoint T
    ChironServingOverrides(); // defaults: {-1,false,false,0.0f,0,0,-1,false,0.07f,8.0f,0}
};

// Resolved serving configuration (ready for use by the eval forward, Task 6).
struct ChironServingConfig
{
    bool  fuseAttnPerLayer, fuseAttnReln, qkNorm, useScfa, whiscCoupling;
    float rotThetaMax, whiscClamp;
    float epsReln;                        // 1e-4f
    std::vector<float> qknormGammaScale;  // L*nH of gamma*sqrt(dH), ready for upload; empty if !qkNorm
    ChironServingConfig();
};

// Applies the serving decision table.  May mutate dims.T (seqLen) and
// w.scfa (k/w resolution, B build, D=0 fallback).  Returns 0 = ok,
// 6 = SCFA init failure, 7 = interlock violation (message in err).
int chiron_resolve_serving(ChironModelDims& dims, ChironModelWeights& w,
                           const ChironServingOverrides& o,
                           ChironServingConfig& cfg, std::string& err);

} // namespace chiron
} // namespace glades

#endif
