// chiron_serving.cpp — CHIRON serving feature-resolution decision table.
//
// Ports chiron_infer.cpp:1040–1234 into a reusable library module.
//
// C++98.

#include "chiron_serving.h"
#include "cuda/gpu_kernels.h"  // scfa_dct_basis_init

#include <cstdio>
#include <cmath>
#include <cstring>

namespace glades {
namespace chiron {

// ---------------------------------------------------------------------------
// ChironServingOverrides ctor
// ---------------------------------------------------------------------------

ChironServingOverrides::ChironServingOverrides()
    : fuseAttnPerLayer(-1)
    , fuseAttnReln(false)
    , qkNorm(false)
    , qkNormGamma(0.0f)
    , scfaForceMode(0)
    , scfaKOverride(0)
    , scfaWOverride(-1)
    , whiscCoupling(false)
    , rotThetaMax(0.07f)
    , whiscClamp(8.0f)
    , seqLen(0)
{
}

// ---------------------------------------------------------------------------
// ChironServingConfig ctor
// ---------------------------------------------------------------------------

ChironServingConfig::ChironServingConfig()
    : fuseAttnPerLayer(false)
    , fuseAttnReln(false)
    , qkNorm(false)
    , useScfa(false)
    , whiscCoupling(false)
    , rotThetaMax(0.0f)
    , whiscClamp(0.0f)
    , epsReln(1e-4f)
{
}

// ---------------------------------------------------------------------------
// chiron_resolve_serving
// ---------------------------------------------------------------------------

int chiron_resolve_serving(ChironModelDims& dims, ChironModelWeights& w,
                           const ChironServingOverrides& o,
                           ChironServingConfig& cfg, std::string& err)
{
    // --- WhiSC interlocks (2026-07-01) ---
    // Both directions are hard errors: a mismatch silently produces wrong logits.
    const bool ckptHasWhisc = !w.rotPhi.empty();
    if (ckptHasWhisc && !o.whiscCoupling)
    {
        err = "FATAL — checkpoint carries WhiSC-D coupling (CHRF flag bit 1024)\n"
              "  but --whisc-coupling was NOT passed.  Serving without it silently omits the\n"
              "  trained per-layer whitened rotation coupling and produces WRONG logits.\n"
              "  Re-run with:  --whisc-coupling --rot-theta-max 0.07  (flagship recipe value)";
        return 7;
    }
    if (o.whiscCoupling && !ckptHasWhisc)
    {
        err = "FATAL — --whisc-coupling was passed but the checkpoint has NO\n"
              "  WhiSC-D state (CHRF flag bit 1024 clear).  Applying the coupling to a\n"
              "  non-WhiSC checkpoint corrupts the forward.  Drop --whisc-coupling.";
        return 7;
    }

    // --- New bit-512 a_drift refusal ---
    if (w.hasADrift)
    {
        err = "FATAL — checkpoint carries OBSD a_drift (CHRF flag bit 512) but the eval "
              "forward does not apply per-layer drift. Serving would be silently wrong. "
              "(OBSD is a closed NO-GO arc; retrain without --per-layer-drift or extend "
              "chiron_serving.)";
        return 7;
    }

    // --- gamma_p auto-disable of per-layer fuse ---
    // fuseAttnPerLayer: -1 = unset (default-on heuristic), 0 = forced off, 1 = forced on.
    // "default-on" means ON unless a heuristic disables it.
    bool fuseAttnPerLayer = (o.fuseAttnPerLayer != 0);  // -1 or 1 → true initially; 0 → false
    bool fusePerLayerExplicit = (o.fuseAttnPerLayer == 0 || o.fuseAttnPerLayer == 1);
    bool fuseAttnReln = o.fuseAttnReln;

    if (fuseAttnPerLayer && w.gamma_p.empty())
    {
        std::printf("[chiron-serving] note: checkpoint has no per-layer gamma_p/beta_p "
                    "(pre-2026-05-12 format) — fuse-attn-per-layer auto-disabled\n");
        fuseAttnPerLayer = false;
    }

    // --- 2026-06-27 fuse-attn-reln auto-enable ---
    // SCFA production models train with --no-fuse-attn --fuse-attn-reln and carry
    // no per-layer gamma_p.  Auto-enable reln-fuse for such checkpoints.
    if (!fuseAttnReln && w.gamma_p.empty() && w.scfa.dLoaded)
    {
        std::printf("[chiron-serving] fuse-attn-reln auto-enabled (SCFA checkpoint with no "
                    "per-layer gamma_p — trained with --no-fuse-attn --fuse-attn-reln)\n");
        fuseAttnReln = true;
    }

    // --- 2026-07-01: WhiSC-D fuse-mode dead-gamma_p guard ---
    // WhiSC/SORC/OBSD checkpoints allocate gamma_p/beta_p as coupling scratch
    // (at dead init: gamma_p=1, beta_p=0) even when training with --no-fuse-attn
    // --fuse-attn-reln.  If gamma_p is at dead init AND the operator didn't
    // explicitly set fuse mode, auto-switch to reln-fuse.
    if (ckptHasWhisc && !w.gamma_p.empty() && w.scfa.dLoaded && !fusePerLayerExplicit
        && fuseAttnPerLayer && (int)w.gamma_p.size() == dims.L && (int)w.beta_p.size() == dims.L)
    {
#ifdef GLADES_HAVE_CUDA
        std::vector<float> gp((size_t)dims.m), bp((size_t)dims.m);
        bool dlOk = w.gamma_p[0]->download(&gp[0], (size_t)dims.m)
                 && w.beta_p[0]->download(&bp[0], (size_t)dims.m);
        if (dlOk)
        {
            bool deadInit = true;
            for (int i = 0; i < dims.m; ++i)
            {
                if (std::fabs(gp[i] - 1.0f) > 1e-4f || std::fabs(bp[i]) > 1e-4f)
                {
                    deadInit = false;
                    break;
                }
            }
            if (deadInit)
            {
                std::printf("[chiron-serving] WhiSC-D checkpoint has DEAD (init) gamma_p — the flagship recipe\n"
                            "  trains with --no-fuse-attn --fuse-attn-reln; auto-switching to reln-fuse and\n"
                            "  disabling per-layer fuse (avoids the gamma_p-present garbage path, tf-nll~20).\n"
                            "  Override with explicit --fuse-attn-per-layer if you trained WhiSC WITH it.\n");
                fuseAttnPerLayer = false;
                fuseAttnReln = true;
            }
            else
            {
                std::fprintf(stderr, "[chiron-serving] WARNING: WhiSC-D checkpoint has a TRAINED gamma_p — keeping\n"
                             "  per-layer fuse. If serving looks wrong, pass --no-fuse-attn --fuse-attn-reln.\n");
            }
        }
#endif
    }

    std::printf("[chiron-serving] fuse-attn-per-layer: %s\n",
                fuseAttnPerLayer ? "ENABLED (paradigm #32)" : "disabled");
    std::printf("[chiron-serving] fuse-attn-reln: %s\n",
                fuseAttnReln ? "ENABLED (single-layer q+=p at L-1)" : "disabled");

    // --- seqLen override ---
    if (o.seqLen > 0 && o.seqLen <= dims.T)
        dims.T = o.seqLen;

    // --- SCFA mode/k/w resolution ---
    bool useScfa = false;
    if (o.scfaForceMode == 1)       useScfa = true;
    else if (o.scfaForceMode == -1) useScfa = false;
    else                            useScfa = w.scfa.dLoaded;

    if (useScfa)
    {
        // Resolve k and w.  Priority: CLI override > loaded blob > defaults.
        const int defaultK = (dims.T >= 16) ? (dims.T / 16) : 4;
        const int defaultW = 8;
        w.scfa.k = (o.scfaKOverride > 0) ? o.scfaKOverride
                   : (w.scfa.dLoaded ? w.scfa.k : defaultK);
        w.scfa.w = (o.scfaWOverride >= 0) ? o.scfaWOverride
                   : (w.scfa.dLoaded ? w.scfa.w : defaultW);
        // Sanity bounds.
        if (w.scfa.k <= 0 || w.scfa.k > dims.T) w.scfa.k = defaultK;
        if (w.scfa.w < 0) w.scfa.w = defaultW;

#ifdef GLADES_HAVE_CUDA
        // Build B (DCT-II basis).
        const size_t B_sz = (size_t)dims.T * (size_t)w.scfa.k;
        if (!w.scfa.B.allocate(B_sz)
            || !glades::gpu::scfa_dct_basis_init(w.scfa.B.data(), dims.T, w.scfa.k))
        {
            std::fprintf(stderr, "[chiron-serving] SCFA B init failed (T=%d k=%d)\n",
                         dims.T, w.scfa.k);
            return 6;
        }
        // If D wasn't loaded, allocate and zero.
        if (!w.scfa.dLoaded)
        {
            const size_t D_sz = (size_t)dims.m * (size_t)(w.scfa.w + 1);
            w.scfa.D.assign((size_t)dims.L, (glades::gpu::GpuBuffer<float>*)0);
            for (int l = 0; l < dims.L; ++l)
            {
                w.scfa.D[l] = new glades::gpu::GpuBuffer<float>();
                if (!w.scfa.D[l]->allocate(D_sz) || !w.scfa.D[l]->zero())
                {
                    std::fprintf(stderr, "[chiron-serving] SCFA D[%d] alloc failed\n", l);
                    return 6;
                }
            }
            std::printf("[chiron-serving] WARNING: SCFA on with D=0 (checkpoint had no SCFA state).\n"
                        "  The y_perp residual branch is dropped; only the low-frequency rank-k\n"
                        "  attention contributes.  Model output will be lower quality than the\n"
                        "  trained-with-D NLL.  Retrain or fine-tune to recover full fidelity.\n");
        }
#endif
        w.scfa.present = true;
        std::printf("[chiron-serving] SCFA: ENABLED (k=%d w=%d, D %s)\n",
                    w.scfa.k, w.scfa.w, w.scfa.dLoaded ? "loaded" : "zero-fallback");
    }
    else
    {
        std::printf("[chiron-serving] SCFA: disabled (dense O(T^2) attention)\n");
    }

    // --- QK-Norm auto-enable ---
    bool qkNorm = o.qkNorm;
    float qkNormGamma = o.qkNormGamma;

    if (!qkNorm && (int)w.qknormGamma.size() == dims.L * dims.nH)
    {
        qkNorm = true;
        std::printf("[chiron-serving] QK-Norm auto-enabled (checkpoint carries per-head gamma)\n");
    }

    // --- QK-Norm: prefill gamma*sqrt(dH) scale vector ---
    if (qkNorm)
    {
        const float sqrtDh = std::sqrt((float)dims.dH);
        const bool haveExact = ((int)w.qknormGamma.size() == dims.L * dims.nH);
        cfg.qknormGammaScale.resize((size_t)dims.L * dims.nH);
        if (haveExact)
        {
            for (size_t i = 0; i < cfg.qknormGammaScale.size(); ++i)
                cfg.qknormGammaScale[i] = w.qknormGamma[i] * sqrtDh;
            std::printf("[chiron-serving] QK-Norm: ENABLED (EXACT per-head gamma from checkpoint, L*nH=%d)\n",
                        dims.L * dims.nH);
        }
        else
        {
            if (qkNormGamma <= 0.0f)
                qkNormGamma = std::log((float)dims.T) / std::log(2.0f);  // log2(T)
            for (size_t i = 0; i < cfg.qknormGammaScale.size(); ++i)
                cfg.qknormGammaScale[i] = qkNormGamma * sqrtDh;
            std::printf("[chiron-serving] QK-Norm: ENABLED (gamma=%.3f APPROX — checkpoint lacks per-head gamma; "
                        "retrain with the bit-256 save for exact)\n", qkNormGamma);
        }
    }
    else
    {
        std::printf("[chiron-serving] QK-Norm: disabled\n");
    }

    // --- Populate cfg ---
    cfg.fuseAttnPerLayer = fuseAttnPerLayer;
    cfg.fuseAttnReln     = fuseAttnReln;
    cfg.qkNorm           = qkNorm;
    cfg.useScfa          = useScfa;
    cfg.whiscCoupling    = o.whiscCoupling;
    cfg.rotThetaMax      = o.rotThetaMax;
    cfg.whiscClamp       = o.whiscClamp;
    cfg.epsReln          = 1e-4f;

    return 0;
}

} // namespace chiron
} // namespace glades
