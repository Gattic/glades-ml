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

// ---------------------------------------------------------------------------
// Eval forward (Task 6).  Ported 1:1 from chiron_infer.cpp Scratch (359-457) /
// forwardInfer (580-740) with the kernel call sequence preserved exactly — the
// later bit-parity gate depends on it.
// ---------------------------------------------------------------------------

// Per-forward scratch.  Recomputes q from d_tokens each call (q is overwritten),
// so the caller uploads d_tokens and downloads logits between forwards.
struct ChironEvalScratch
{
    glades::gpu::GpuBuffer<int>   d_tokens;   // [T] caller uploads
    glades::gpu::GpuBuffer<float> q;          // [T, m]
    glades::gpu::GpuBuffer<float> p;          // [T, m]
    glades::gpu::GpuBuffer<float> q_tmp;      // [T, m]
    glades::gpu::GpuBuffer<float> stats;      // [L, T, 2]
    // Dense-attention scratch (allocated when SCFA is OFF).
    glades::gpu::GpuBuffer<float> sQ, sK, sV, sO;  // [T, dModel]
    glades::gpu::GpuBuffer<float> scratch_P;       // [nH, T, T]
    glades::gpu::GpuBuffer<float> logits;          // [T, V] caller downloads
    // paradigm #32 fuse-attn-per-layer scratch.
    glades::gpu::GpuBuffer<float> p_norm;          // [T, m]
    glades::gpu::GpuBuffer<float> stats_p;         // [L, T, 2]
    // SCFA scratch (allocated when SCFA is ON).
    glades::gpu::GpuBuffer<float> scfa_qcompr;  // [k, m]
    glades::gpu::GpuBuffer<float> scfa_qpar;    // [T, m]
    glades::gpu::GpuBuffer<float> scfa_qperp;   // [T, m]
    glades::gpu::GpuBuffer<float> scfa_yperp;   // [T, m]
    glades::gpu::GpuBuffer<float> scfa_ycompr;  // [k, m]
    glades::gpu::GpuBuffer<float> scfa_ypar;    // [T, m]
    glades::gpu::GpuBuffer<float> scfa_inner_p; // [k, m]
    glades::gpu::GpuBuffer<float> scfa_inner_sQ;
    glades::gpu::GpuBuffer<float> scfa_inner_sK;
    glades::gpu::GpuBuffer<float> scfa_inner_sV;
    glades::gpu::GpuBuffer<float> scfa_inner_sO;  // each [k, dModel]
    glades::gpu::GpuBuffer<float> scfa_inner_sP;  // [nH, k, k]
    glades::gpu::GpuBuffer<float> qknorm_invNorm;     // [k*nH] throwaway
    glades::gpu::GpuBuffer<float> qknorm_gamma_scale; // [L*nH] = gamma*sqrt(dH), prefilled
    // WhiSC-D coupling scratch. `a` is fixed checkpoint state, per layer,
    // rather than being recalibrated from the current potentially padded window.
    glades::gpu::GpuBuffer<float> rot_a;       // [m] SORC coeff a (folded in place)
    glades::gpu::GpuBuffer<float> rot_c;       // [m] SORC coeff c (folded in place)
    glades::gpu::GpuBuffer<float> whisc_a;     // [L*m] fixed whitening scale
    // Owned per-layer rot_phi angle buffers [m], uploaded from w.rotPhi when whisc.
    std::vector<glades::gpu::GpuBuffer<float>*> rotPhiGpu;

    ChironEvalScratch();
    ~ChironEvalScratch();   // deletes rotPhiGpu[i]

    // Allocates all scratch and uploads qknormGammaScale + rot_phi.  Returns true
    // on success.  useScfa/scfaK/whisc are taken from cfg/w.
    bool allocate(const ChironModelDims& d, const ChironModelWeights& w,
                  const ChironServingConfig& cfg);
private:
    ChironEvalScratch(const ChironEvalScratch&);
    ChironEvalScratch& operator=(const ChironEvalScratch&);
};

// Forward pass: q_0 = embed(tokens) -> logits [T, V] in s.logits.  Preserves the
// exact glades::gpu call sequence of chiron_infer.cpp::forwardInfer.
bool chiron_eval_forward(const ChironModelDims& d, const ChironModelWeights& w,
                         const ChironServingConfig& cfg, ChironEvalScratch& s);

} // namespace chiron
} // namespace glades

#endif
