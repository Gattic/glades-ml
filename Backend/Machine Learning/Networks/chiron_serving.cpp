// chiron_serving.cpp — CHIRON serving feature-resolution decision table.
//
// Ports chiron_infer.cpp:1040–1234 into a reusable library module.
//
// C++98.

#include "chiron_serving.h"
#include "cuda/gpu_kernels.h"  // scfa_dct_basis_init, embedding_gather, axpy, qknorm, device_memcpy_d2d
#include "cuda/gpu_chiron.h"   // chiron_attention_shear_tiled, chiron_reln_forward, rot/whisc kernels
#include "cuda/gpu_blas.h"     // sgemm_rowmajor / _atb / _abt

#include <cstdio>
#include <cstdlib>  // std::getenv
#include <cmath>
#include <cstring>
#include <algorithm>  // std::min, std::max

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
    // In-memory callers predating GQA populate nH/dModel only. Preserve that
    // legacy construction contract by resolving an omitted KV geometry to MHA.
    if(dims.nKVH<=0)dims.nKVH=dims.nH;
    if(dims.dModelKV<=0)dims.dModelKV=dims.dModel;
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

    // --- bit-512 (a_drift) interlock (2026-07-03: refines commit 1ad8baeb2) ---
    // a_drift (the OBSD per-layer symplectic drift gate) is co-allocated with
    // rot_phi (bit 1024) whenever --rot-coupling / --whisc-coupling is used, even
    // when --per-layer-drift is NOT active — in which case it stays at zero-init
    // and is an EXACT no-op (OBSD drift is q += a(x)tanh(...), so a=0 => identity).
    // Production WhiSC-D and PIED checkpoints carry bit 512 with a_drift ALL-ZERO.
    // Commit 1ad8baeb2 removed the blanket refusal (which bricked PIED serving) but
    // reopened the real hazard: a genuinely drift-TRAINED checkpoint (nonzero
    // a_drift) would serve silently WRONG, since the eval forward does not apply
    // per-layer drift.  Refuse ONLY that case; serve the zero-init case normally.
    if (w.hasADrift)
    {
        bool anyNonzero = false;
        for (size_t i = 0; i < w.aDrift.size(); ++i)
            if (w.aDrift[i] != 0.0f) { anyNonzero = true; break; }
        if (anyNonzero)
        {
            err = "FATAL — checkpoint carries TRAINED OBSD a_drift (CHRF flag bit 512, nonzero).\n"
                  "  The eval forward does NOT apply the per-layer drift, so serving would produce\n"
                  "  silently WRONG logits.  OBSD (per-layer symplectic drift) is a CLOSED NO-GO arc —\n"
                  "  no production checkpoint should carry trained a_drift.  Refusing to serve (code 7).";
            return 7;
        }
        std::printf("[chiron-serving] a_drift present but all-zero (bit 512 co-allocated by rot "
                    "coupling) -- serving without drift is exact\n");
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
        if (w.scfa.dLoaded && !w.scfa.causalBlock)
        {
            err = "FATAL — checkpoint carries legacy global-DCT SCFA state without the "
                  "causal-block marker (bit 4096). That operator leaks future tokens and is "
                  "incompatible with autoregressive serving. Retrain with causal-block SCFA.";
            return 7;
        }
        // Resolve k and w.  Priority: CLI override > loaded blob > defaults.
        int defaultK = dims.T / 16;
        if (defaultK < 4) defaultK = dims.T < 4 ? dims.T : 4;
        const int defaultW = 8;
        w.scfa.k = (o.scfaKOverride > 0) ? o.scfaKOverride
                   : (w.scfa.dLoaded ? w.scfa.k : defaultK);
        w.scfa.w = (o.scfaWOverride >= 0) ? o.scfaWOverride
                   : (w.scfa.dLoaded ? w.scfa.w : defaultW);
        // Sanity bounds.
        if (w.scfa.k <= 0 || w.scfa.k > dims.T) w.scfa.k = defaultK;
        if (w.scfa.w < 0) w.scfa.w = defaultW;

#ifdef GLADES_HAVE_CUDA
        // Causal SCFA uses implicit contiguous block compression and lagged
        // lifts; the old dense DCT B[T,k] allocation is intentionally gone.
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
        w.scfa.causalBlock = true;
        std::printf("[chiron-serving] causal-block SCFA: ENABLED (k=%d w=%d, D %s)\n",
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

// ---------------------------------------------------------------------------
// ChironEvalScratch — ported from chiron_infer.cpp Scratch (359-457).
// ---------------------------------------------------------------------------

ChironEvalScratch::ChironEvalScratch()
{
}

ChironEvalScratch::~ChironEvalScratch()
{
    for (size_t i = 0; i < rotPhiGpu.size(); ++i) delete rotPhiGpu[i];
    rotPhiGpu.clear();
}

bool ChironEvalScratch::allocate(const ChironModelDims& d, const ChironModelWeights& w,
                                 const ChironServingConfig& cfg)
{
    const int T=d.T, m=d.m, L=d.L, dModel=d.dModel;
    const int dModelKV=d.dModelKV>0?d.dModelKV:dModel;
    const bool useScfa = cfg.useScfa;
    const int  scfaK   = w.scfa.k;
    const bool whisc   = cfg.whiscCoupling;

    if (!d_tokens.allocate(T)) return false;
    if (!q.allocate((size_t)T * m)) return false;
    if (!p.allocate((size_t)T * m)) return false;
    if (!q_tmp.allocate((size_t)T * m)) return false;
    if (!stats.allocate((size_t)L * T * 2u)) return false;
    if (!logits.allocate((size_t)T * d.V)) return false;
    if (!p_norm.allocate((size_t)T * m)) return false;
    if (!stats_p.allocate((size_t)L * T * 2u)) return false;
    if (useScfa)
    {
        const size_t Tm  = (size_t)T * (size_t)m;
        const size_t km  = (size_t)scfaK * (size_t)m;
        const size_t kdM = (size_t)scfaK * (size_t)dModel;
        const size_t kdMKV = (size_t)scfaK * (size_t)dModelKV;
        const size_t Skk = (size_t)d.nH * (size_t)scfaK * (size_t)scfaK;
        if (!scfa_qcompr.allocate(km)) return false;
        if (!scfa_qpar.allocate(Tm)) return false;
        if (!scfa_qperp.allocate(Tm)) return false;
        if (!scfa_yperp.allocate(Tm)) return false;
        if (!scfa_ycompr.allocate(km)) return false;
        if (!scfa_ypar.allocate(Tm)) return false;
        if (!scfa_inner_p.allocate(km)) return false;
        if (!scfa_inner_sQ.allocate(kdM)) return false;
        if (!scfa_inner_sK.allocate(kdMKV)) return false;
        if (!scfa_inner_sV.allocate(kdMKV)) return false;
        if (!scfa_inner_sO.allocate(kdM)) return false;
        if (!scfa_inner_sP.allocate(Skk)) return false;
        if (!qknorm_invNorm.allocate((size_t)scfaK * d.nH)) return false;
        if (!qknorm_gamma_scale.allocate((size_t)d.L * d.nH)) return false;  // per-layer gamma*sqrt(dH)
    }
    else
    {
        if (!sQ.allocate((size_t)T * dModel)) return false;
        if (!sK.allocate((size_t)T * dModelKV)) return false;
        if (!sV.allocate((size_t)T * dModelKV)) return false;
        if (!sO.allocate((size_t)T * dModel)) return false;
        if (!scratch_P.allocate((size_t)d.nH * T * T)) return false;
    }
    if (d.ffnHidden > 0)
    {
        const size_t n=(size_t)T*d.ffnHidden;
        if (!ffnGate.allocate(n) || !ffnUp.allocate(n) || !ffnHidden.allocate(n)) return false;
    }
    if (whisc)
    {
        if (!rot_a.allocate((size_t)m)) return false;
        if (!rot_c.allocate((size_t)m)) return false;
        const size_t Lm = (size_t)L * (size_t)m;
        if (!whisc_a.allocate(Lm)) return false;
        if (w.whiscA.size() == Lm)
        {
            if (!whisc_a.upload(&w.whiscA[0], Lm)) return false;
        }
        else
        {
            // New causal checkpoints persist a. Identity is safe for freshly
            // initialized/manual models; legacy SCFA checkpoints are refused.
            std::vector<float> ones(Lm, 1.0f);
            if (!whisc_a.upload(&ones[0], Lm)) return false;
        }
    }

    // QK-Norm: upload the prefilled per-layer per-head gamma*sqrt(dH) scale
    // (resolve fills cfg.qknormGammaScale host-side).  Port of chiron_infer.cpp:1231.
    // The buffer is only allocated on the SCFA path (where qkNorm is consumed);
    // the .allocated() guard preserves chiron_infer's behavior for the never-used
    // dense+qkNorm combo (where its upload return was ignored — dense attention
    // never reads the scale), instead of failing scratch allocation.
    if (!cfg.qknormGammaScale.empty() && qknorm_gamma_scale.allocated())
    {
        if (!qknorm_gamma_scale.upload(&cfg.qknormGammaScale[0], cfg.qknormGammaScale.size()))
            return false;
    }

    // 2026-07-01: upload WhiSC-D rot_phi angles to per-layer GPU buffers [m].
    // rotPhi is L*m layer-major.  Port of chiron_infer.cpp:1238-1259 (tag rebranded
    // to [chiron-serving]; glyphs ASCII-normalized).
    if (whisc)
    {
        rotPhiGpu.assign(d.L, (glades::gpu::GpuBuffer<float>*)0);
        double rotAbsSum = 0.0; float rotAbsMax = 0.0f;
        for (int l = 0; l < d.L; ++l)
        {
            rotPhiGpu[l] = new glades::gpu::GpuBuffer<float>();
            if (!rotPhiGpu[l]->allocate((size_t)d.m))
            { std::fprintf(stderr, "chiron_serving: rot_phi GPU alloc failed (layer %d)\n", l); return false; }
            if (!w.rotPhi.empty())
            {
                if (!rotPhiGpu[l]->upload(&w.rotPhi[(size_t)l * d.m], (size_t)d.m)) return false;
                for (int i = 0; i < d.m; ++i)
                { float v = std::fabs(w.rotPhi[(size_t)l * d.m + i]); rotAbsSum += v; if (v > rotAbsMax) rotAbsMax = v; }
            }
        }
        std::printf("[chiron-serving] WhiSC-D coupling ENABLED (theta_max=%.4g clamp=%.4g) -- "
                    "rot_phi |.|max=%.4g mean=%.4g (nonzero confirms angles loaded)\n",
                    cfg.rotThetaMax, cfg.whiscClamp, rotAbsMax, rotAbsSum / ((double)d.L * d.m));
        if (rotAbsMax == 0.0f)
            std::fprintf(stderr, "[chiron-serving] WARNING: rot_phi is all-zero -- coupling is identity "
                         "(checkpoint may predate WhiSC-D training or loaded wrong region)\n");
    }
    return true;
}

// ---------------------------------------------------------------------------
// scfa_shear_eval — ported from chiron_infer.cpp scfa_shear_infer (484-577).
// Same glades::gpu calls, same order, same args.
// ---------------------------------------------------------------------------

static void dbg_mag(int layer, const char* tag, const glades::gpu::GpuBuffer<float>& b, size_t n)
{
    if (layer != 0 || !std::getenv("CHIRON_DBG")) return;
    size_t s = std::min(n, (size_t)8192);
    std::vector<float> h(s);
    const_cast<glades::gpu::GpuBuffer<float>&>(b).download(&h[0], s);
    float mx = 0; double sum = 0;
    for (size_t i = 0; i < s; ++i) { mx = std::max(mx, std::fabs(h[i])); sum += std::fabs(h[i]); }
    std::printf("[dbg-scfa L%02d]   %-10s |.|max=%.4g mean=%.4g\n", layer, tag, mx, sum / s);
}

static bool scfa_shear_eval(ChironEvalScratch& s, const ChironScfaState& scfa,
                            const float* Wq, const float* Wk,
                            const float* Wv, const float* Wo,
                            int layer, int T, int m, int nH, int nKVH, int dH,
                            bool invert, bool qkNorm)
{
    const int k = scfa.k;
    const int w = scfa.w;
    const int dModel = nH * dH;
    const int dModelKV = nKVH * dH;
    const size_t Tm = (size_t)T * (size_t)m;
    const size_t km = (size_t)k * (size_t)m;

    // 1. Causal block compression. Summary b contains only its contiguous
    // source block; lagged lifts expose it starting in the following block.
    if (!glades::gpu::scfa_block_compress(
            s.q.data(), T, m, k, 1.0f, 0.0f,
            s.scfa_qcompr.data())) return false;
    dbg_mag(layer, "q_in", s.q, Tm);
    dbg_mag(layer, "q_compr", s.scfa_qcompr, km);

    // 2. Causal lag lift: q_par for block b uses only summary b-1.
    if (!glades::gpu::scfa_causal_lag_lift(
            s.scfa_qcompr.data(), T, m, k, 1.0f, 0.0f,
            s.scfa_qpar.data())) return false;

    // 3. q_perp = q - q_par.
    glades::gpu::device_memcpy_d2d(s.scfa_qperp.data(), s.q.data(), sizeof(float) * Tm);
    if (!glades::gpu::axpy(-1.0f, s.scfa_qpar.data(), s.scfa_qperp.data(), (int)Tm)) return false;

    // 4. y_perp = D . q_perp (depthwise causal conv).
    if (!glades::gpu::scfa_depthwise_causal_conv_fwd(
            s.scfa_qperp.data(), scfa.D[layer]->data(),
            T, m, w, s.scfa_yperp.data())) return false;

    // 5. Inner attention on compressed length k.
    if (!s.scfa_inner_p.zero()) return false;
    if (qkNorm)
    {
        // QK-Norm decomposed path (replicates chiron_main.cpp:8734-8787 FP32 split).
        if (!glades::gpu::sgemm_rowmajor(k, dModel, m, 1.0f,
                s.scfa_qcompr.data(), m, Wq, dModel, 0.0f, s.scfa_inner_sQ.data(), dModel)) return false;
        if (!glades::gpu::sgemm_rowmajor(k, dModelKV, m, 1.0f,
                s.scfa_qcompr.data(), m, Wk, dModelKV, 0.0f, s.scfa_inner_sK.data(), dModelKV)) return false;
        if (!glades::gpu::sgemm_rowmajor(k, dModelKV, m, 1.0f,
                s.scfa_qcompr.data(), m, Wv, dModelKV, 0.0f, s.scfa_inner_sV.data(), dModelKV)) return false;
        if (!glades::gpu::qknorm_forward_gpu(s.scfa_inner_sQ.data(), s.qknorm_invNorm.data(), k, nH, dH, 1e-6f)) return false;
        if (!glades::gpu::qknorm_forward_gpu(s.scfa_inner_sK.data(), s.qknorm_invNorm.data(), k, nKVH, dH, 1e-6f)) return false;
        if (!glades::gpu::scale_q_per_head(s.scfa_inner_sQ.data(), s.qknorm_gamma_scale.data() + (size_t)layer * nH, k, nH, dH)) return false;
        if (!glades::gpu::flash_attention_cublas_tiled(
                s.scfa_inner_sQ.data(), s.scfa_inner_sK.data(), s.scfa_inner_sV.data(),
                k, nH, nKVH, dH, dModel, dModelKV, /*causal=*/true,
                s.scfa_inner_sO.data(), s.scfa_inner_sP.data())) return false;
        if (!glades::gpu::sgemm_rowmajor(k, m, dModel, 1.0f,
                s.scfa_inner_sO.data(), dModel, Wo, m, 0.0f, s.scfa_inner_p.data(), m)) return false;
    }
    else if (!glades::gpu::chiron_attention_shear_tiled(
            s.scfa_qcompr.data(), s.scfa_inner_p.data(),
            Wq, Wk, Wv, Wo,
            k, m, nH, nKVH, dH, /*causal=*/true, /*invert=*/false,
            s.scfa_inner_sQ.data(), s.scfa_inner_sK.data(),
            s.scfa_inner_sV.data(), s.scfa_inner_sO.data(),
            s.scfa_inner_sP.data())) return false;
    glades::gpu::device_memcpy_d2d(s.scfa_ycompr.data(), s.scfa_inner_p.data(),
                                    sizeof(float) * km);
    dbg_mag(layer, "q_par", s.scfa_qpar, Tm);
    dbg_mag(layer, "q_perp", s.scfa_qperp, Tm);
    dbg_mag(layer, "y_perp", s.scfa_yperp, Tm);
    dbg_mag(layer, "y_compr", s.scfa_ycompr, km);

    // 6. Causal lag lift: compressed output b is visible in block b+1.
    if (!glades::gpu::scfa_causal_lag_lift(
            s.scfa_ycompr.data(), T, m, k, 1.0f, 0.0f,
            s.scfa_ypar.data())) return false;

    dbg_mag(layer, "y_par", s.scfa_ypar, Tm);

    // 7-8. p +- (y_par + y_perp).
    const float sign = invert ? -1.0f : 1.0f;
    if (!glades::gpu::axpy(1.0f, s.scfa_yperp.data(), s.scfa_ypar.data(), (int)Tm)) return false;
    if (!glades::gpu::axpy(sign, s.scfa_ypar.data(), s.p.data(), (int)Tm)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// chiron_eval_forward — ported from chiron_infer.cpp forwardInfer (580-740).
// The glades::gpu call sequence (including device_memcpy_d2d and .zero()) is
// preserved verbatim; the later refactored-inference bit-parity gate depends
// on it.  CHIRON_DBG env-gated debug blocks are kept.
// ---------------------------------------------------------------------------

static bool append_trace_row(const ChironEvalScratch& s, int T, int m,
                             int traceRow, ChironEvalTrace& trace)
{
    const size_t count = (size_t)T * (size_t)m;
    std::vector<float> q(count), p(count);
    if (!s.q.download(&q[0], count) || !s.p.download(&p[0], count)) return false;
    const float* qr = &q[(size_t)traceRow * (size_t)m];
    const float* pr = &p[(size_t)traceRow * (size_t)m];
    trace.lastQ.insert(trace.lastQ.end(), qr, qr + m);
    trace.lastP.insert(trace.lastP.end(), pr, pr + m);
    return true;
}

static bool chiron_eval_forward_impl(const ChironModelDims& d,
                                     const ChironModelWeights& w,
                                     const ChironServingConfig& cfg,
                                     ChironEvalScratch& s,
                                     ChironEvalScfaLayerObserver observer,
                                     void* observerContext,
                                     int traceRow,
                                     ChironEvalTrace* trace)
{
    const int T=d.T,m=d.m,V=d.V,L=d.L,nH=d.nH,nKVH=d.nKVH>0?d.nKVH:d.nH,dH=d.dH;
    if (trace)
    {
        trace->clear();
        if (traceRow < 0 || traceRow >= T) return false;
        trace->hiddenSize = m;
        trace->layers = L;
        trace->row = traceRow;
        trace->lastQ.reserve((size_t)(L + 1) * (size_t)m);
        trace->lastP.reserve((size_t)(L + 1) * (size_t)m);
    }

    // Fuse is only active if requested AND the checkpoint provided per-layer params.
    const bool applyFuse = cfg.fuseAttnPerLayer
                           && !w.gamma_p.empty() && !w.beta_p.empty()
                           && (int)w.gamma_p.size() == L && (int)w.beta_p.size() == L;
    // 2026-07-03 divergence-fix vs chiron_infer's plain 1/sqrt(L): on the SCFA
    // forward path the per-layer fuse alpha adopts the trainer's k/T-damped
    // formula, replicated EXACTLY from glades-trainer chiron_main.cpp:13304-13306
    // (non-SFA-swap SCFA branch):
    //     alpha = (1/sqrt(L)) * (scfa_k / T)
    // The dense path keeps the plain 1/sqrt(L).  NOTE: fuse-attn-per-layer is OFF
    // for all production checkpoints (they serve --no-fuse-attn --fuse-attn-reln),
    // so this does NOT affect the bit-parity gate; it aligns eval with training
    // whenever per-layer fuse IS used.
    float fuseAlpha = 0.0f;
    if (applyFuse)
    {
        if (w.scfa.present)
            fuseAlpha = (1.0f / std::sqrt((float)L)) * ((float)w.scfa.k / (float)T);
        else
            fuseAlpha = 1.0f / std::sqrt((float)L);
    }

    // q_0 = embed(tokens); p_0 = 0.
    if (!glades::gpu::embedding_gather(w.E.data(), s.d_tokens.data(), T, V, m, s.q.data())) return false;
    if (!s.p.zero()) return false;
    if (trace && !append_trace_row(s, T, m, traceRow, *trace)) return false;

    if (std::getenv("CHIRON_DBG"))
    {
        std::vector<int> tk(T);
        s.d_tokens.download(&tk[0], (size_t)T);
        std::vector<float> qfull((size_t)T * m);
        s.q.download(&qfull[0], (size_t)T * m);
        const int positions[4] = {0, 5000, 10000, 16000};
        for (int pi = 0; pi < 4; ++pi)
        {
            int pos = positions[pi]; if (pos >= T) pos = T - 1;
            const float* row = &qfull[(size_t)pos * m];
            std::printf("[dbg-embed] pos=%5d tok=%6d  q[0..4]=%.4g,%.4g,%.4g,%.4g,%.4g\n",
                        pos, tk[pos], row[0], row[1], row[2], row[3], row[4]);
        }
    }

    for (int l = 0; l < L; ++l)
    {
        if (cfg.useScfa)
        {
            if (!scfa_shear_eval(s, w.scfa,
                                 w.Wq[l]->data(), w.Wk[l]->data(), w.Wv[l]->data(), w.Wo[l]->data(),
                                 l,T,m,nH,nKVH,dH,/*invert=*/false,cfg.qkNorm)) return false;
            if (observer && !observer(observerContext, l, s)) return false;
        }
        else if (!glades::gpu::chiron_attention_shear_tiled(
                s.q.data(), s.p.data(),
                w.Wq[l]->data(), w.Wk[l]->data(), w.Wv[l]->data(), w.Wo[l]->data(),
                T,m,nH,nKVH,dH,/*causal=*/true,/*invert=*/false,
                s.sQ.data(),s.sK.data(),s.sV.data(),s.sO.data(),
                s.scratch_P.data())) return false;

        if (d.ffnHidden > 0 && !glades::gpu::chiron_ffn_shear_forward(
                s.q.data(),s.p.data(),w.ffnGate[l]->data(),w.ffnUp[l]->data(),
                w.ffnDown[l]->data(),T,m,d.ffnHidden,1.0f,
                s.ffnGate.data(),s.ffnUp.data(),s.ffnHidden.data())) return false;

        // Single-layer fuse: at layer L-1 only, q += p BEFORE the final q-reln.
        if (cfg.fuseAttnReln && l == L - 1)
        {
            if (!glades::gpu::axpy(1.0f, s.p.data(), s.q.data(), T * m)) return false;
        }

        // Paradigm #32 fuse-attn-per-layer: at every layer, before the q-reln,
        //   p_norm = reln_p(p, gamma_p[l], beta_p[l]); q += fuseAlpha * p_norm.
        if (applyFuse)
        {
            if (!glades::gpu::chiron_reln_forward(
                    s.p.data(), s.p_norm.data(),
                    s.stats_p.data() + (size_t)l * T * 2u,
                    w.gamma_p[l]->data(), w.beta_p[l]->data(),
                    T, m, cfg.epsReln)) return false;
            if (!glades::gpu::axpy(fuseAlpha, s.p_norm.data(), s.q.data(), T * m)) return false;
        }

        // 2026-07-01: WhiSC-D coupling (CHRF bit 1024) — AFTER the attention shear
        // + fuse and BEFORE the q-reln.  s_warm=1 (no warmup ramp), ema=1.0.
        //   1. rot_coeffs: learned angle rot_phi[l] -> SORC coeffs (rot_a, rot_c)
        //   2. update_stats (ema=1.0): per-channel a = clamp((E[q^2]/E[p^2])^0.25)
        //   3. fold: absorb whitening into the coeffs (rot_a*=a^2, rot_c/=a^2)
        //   4. rot_forward(+1): the folded 3-shear W^-1 R(theta) W in one pass, in place
        if (cfg.whiscCoupling)
        {
            const float* aw = s.whisc_a.data() + (size_t)l * m;
            if (!glades::gpu::chiron_rot_coeffs(
                    s.rotPhiGpu[l]->data(), cfg.rotThetaMax, /*s_warm=*/1.0f, m,
                    s.rot_a.data(), s.rot_c.data())) return false;
            // Fixed checkpoint state: current-window/future tokens never
            // recalibrate coefficients used by this or a later decode step.
            if (!glades::gpu::chiron_whisc_fold_coeffs(
                    s.rot_a.data(), s.rot_c.data(), aw, m)) return false;
            if (!glades::gpu::chiron_rot_forward(
                    s.q.data(), s.p.data(), s.rot_a.data(), s.rot_c.data(),
                    /*sign=*/+1.0f, T, m)) return false;
        }

        if (!glades::gpu::chiron_reln_forward(
                s.q.data(), s.q_tmp.data(),
                s.stats.data() + (size_t)l * T * 2u,
                w.gamma[l]->data(), w.beta[l]->data(),
                T, m, cfg.epsReln)) return false;
        glades::gpu::device_memcpy_d2d(s.q.data(), s.q_tmp.data(), sizeof(float) * T * m);
        if (trace && !append_trace_row(s, T, m, traceRow, *trace)) return false;

        if (std::getenv("CHIRON_DBG"))
        {
            std::vector<float> hq(64), hp(64);
            s.q.download(&hq[0], 64); s.p.download(&hp[0], 64);
            float qmx=0,pmx=0; double qs=0,ps=0;
            for (int i=0;i<64;++i){ qmx=std::max(qmx,std::fabs(hq[i])); pmx=std::max(pmx,std::fabs(hp[i])); qs+=std::fabs(hq[i]); ps+=std::fabs(hp[i]); }
            std::printf("[dbg] L%02d  |q|max=%.4g mean=%.4g   |p|max=%.4g mean=%.4g   q[0..3]=%.3g,%.3g,%.3g,%.3g\n",
                        l, qmx, qs/64, pmx, ps/64, hq[0],hq[1],hq[2],hq[3]);
        }
    }

    // Readout: logits = q_L . E^T
    if (!glades::gpu::sgemm_rowmajor_abt(
        T, V, m, 1.0f,
        s.q.data(), m, w.E.data(), m,
        0.0f, s.logits.data(), V)) return false;
    if (std::getenv("CHIRON_DBG"))
    {
        std::vector<float> lg(d.V);
        s.logits.download(&lg[0], d.V);
        int am=0; float mx=lg[0]; for (int v=1;v<d.V;++v) if(lg[v]>mx){mx=lg[v];am=v;}
        double mean=0; for(int v=0;v<d.V;++v) mean+=lg[v];
        std::printf("[dbg] logits[pos0]: max=%.4g argmax=%d mean=%.4g\n", mx, am, mean/d.V);
    }
    return true;
}

bool chiron_eval_forward_observed(const ChironModelDims& d,
                                  const ChironModelWeights& w,
                                  const ChironServingConfig& cfg,
                                  ChironEvalScratch& s,
                                  ChironEvalScfaLayerObserver observer,
                                  void* observerContext)
{
    return chiron_eval_forward_impl(d, w, cfg, s, observer, observerContext,
                                    -1, NULL);
}

bool chiron_eval_forward(const ChironModelDims& d, const ChironModelWeights& w,
                         const ChironServingConfig& cfg, ChironEvalScratch& s)
{
    return chiron_eval_forward_impl(d, w, cfg, s, NULL, NULL, -1, NULL);
}

bool chiron_eval_forward_trace(const ChironModelDims& d,
                               const ChironModelWeights& w,
                               const ChironServingConfig& cfg,
                               ChironEvalScratch& s,
                               int traceRow,
                               ChironEvalTrace& trace)
{
    return chiron_eval_forward_impl(d, w, cfg, s, NULL, NULL,
                                    traceRow, &trace);
}

} // namespace chiron
} // namespace glades
