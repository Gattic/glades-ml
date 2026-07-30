// chiron_decode_cache.cpp — exact no-slide incremental decode for causal SCFA CHIRON.
// C++98.

#include "chiron_decode_cache.h"
#include "cuda/gpu_blas.h"
#include "cuda/gpu_chiron.h"
#include "cuda/gpu_device.h"
#include "cuda/gpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace glades {
namespace chiron {

struct ChironDecodeCache::Impl
{
    struct Layer
    {
        glades::gpu::GpuBuffer<float> qCompr;    // [k,m]
        glades::gpu::GpuBuffer<float> yLast;     // [m], most recently committed y_compr
        glades::gpu::GpuBuffer<float> kCache;    // [k,dModelKV], projected/normalized
        glades::gpu::GpuBuffer<float> vCache;    // [k,dModelKV]
        glades::gpu::GpuBuffer<float> qBlock;    // [maxBlockWidth,m]
        glades::gpu::GpuBuffer<float> qPerpRing; // [w+1,m]

        bool allocate(int k, int m, int dModelKV, int maxBlockWidth, int historyCapacity)
        {
            return qCompr.allocate((size_t)k * m)
                && yLast.allocate((size_t)m)
                && kCache.allocate((size_t)k * dModelKV)
                && vCache.allocate((size_t)k * dModelKV)
                && qBlock.allocate((size_t)maxBlockWidth * m)
                && qPerpRing.allocate((size_t)historyCapacity * m);
        }

        bool zero()
        {
            return qCompr.zero() && yLast.zero() && kCache.zero() && vCache.zero()
                && qBlock.zero() && qPerpRing.zero();
        }
    };

    std::vector<Layer*> layers;
    const ChironModelWeights* modelIdentity;
    int T, m, V, L, nH, nKVH, dH, dModel, dModelKV, ffnHidden;
    int k, w, maxBlockWidth, historyCapacity;
    int position, completedBlocks, blockFill, historyCount, historyWrite;
    bool qkNorm, fuseAttnReln, fuseAttnPerLayer, whiscCoupling;
    float epsReln, rotThetaMax, whiscClamp;
    bool isReady;

    Impl()
        : modelIdentity(NULL), T(0), m(0), V(0), L(0), nH(0), nKVH(0), dH(0)
        , dModel(0), dModelKV(0), ffnHidden(0), k(0), w(0), maxBlockWidth(0)
        , historyCapacity(0), position(0), completedBlocks(0), blockFill(0)
        , historyCount(0), historyWrite(0), qkNorm(false), fuseAttnReln(false)
        , fuseAttnPerLayer(false), whiscCoupling(false), epsReln(0.0f)
        , rotThetaMax(0.0f), whiscClamp(0.0f), isReady(false)
    {
    }

    ~Impl() { clear(); }

    void clear()
    {
        for (size_t i = 0; i < layers.size(); ++i) delete layers[i];
        layers.clear();
        modelIdentity = NULL;
        isReady = false;
    }

    int blockBegin(int b) const
    {
        return (int)(((long long)b * (long long)T) / (long long)k);
    }

    int blockWidth(int b) const
    {
        return blockBegin(b + 1) - blockBegin(b);
    }

    bool scratchMatches(const ChironEvalScratch& s) const
    {
        return s.d_tokens.size() >= 1u
            && s.q.size() >= (size_t)T * m
            && s.p.size() >= (size_t)T * m
            && s.q_tmp.size() >= (size_t)T * m
            && s.logits.size() >= (size_t)T * V
            && s.stats.size() >= (size_t)L * T * 2u
            && s.stats_p.size() >= (size_t)L * T * 2u
            && s.p_norm.size() >= (size_t)T * m
            && s.scfa_qpar.size() >= (size_t)T * m
            && s.scfa_qperp.size() >= (size_t)T * m
            && s.scfa_yperp.size() >= (size_t)T * m
            && s.scfa_ypar.size() >= (size_t)T * m
            && s.scfa_inner_sQ.size() >= (size_t)k * dModel
            && s.scfa_inner_sO.size() >= (size_t)k * dModel
            && s.scfa_inner_sP.size() >= (size_t)nH * k * k
            && (!qkNorm || (s.qknorm_invNorm.size() >= (size_t)k * nH
                            && s.qknorm_gamma_scale.size() >= (size_t)L * nH))
            && (ffnHidden <= 0
                || (s.ffnGate.size() >= (size_t)T * ffnHidden
                    && s.ffnUp.size() >= (size_t)T * ffnHidden
                    && s.ffnHidden.size() >= (size_t)T * ffnHidden));
    }

    bool matches(const ChironModelDims& d, const ChironModelWeights& weights,
                 const ChironServingConfig& cfg, const ChironEvalScratch& s) const
    {
        const int resolvedNKVH = d.nKVH > 0 ? d.nKVH : d.nH;
        const int resolvedDMKV = d.dModelKV > 0 ? d.dModelKV : resolvedNKVH * d.dH;
        return isReady && modelIdentity == &weights
            && d.T == T && d.m == m && d.V == V && d.L == L
            && d.nH == nH && resolvedNKVH == nKVH && d.dH == dH
            && d.dModel == dModel && resolvedDMKV == dModelKV
            && d.ffnHidden == ffnHidden
            && weights.scfa.present && weights.scfa.dLoaded && weights.scfa.causalBlock
            && weights.scfa.k == k && weights.scfa.w == w
            && cfg.useScfa && cfg.qkNorm == qkNorm
            && cfg.fuseAttnReln == fuseAttnReln
            && cfg.fuseAttnPerLayer == fuseAttnPerLayer
            && cfg.whiscCoupling == whiscCoupling
            && cfg.epsReln == epsReln && cfg.rotThetaMax == rotThetaMax
            && cfg.whiscClamp == whiscClamp
            && scratchMatches(s);
    }

    bool allocateFor(const ChironModelDims& d, const ChironModelWeights& weights,
                     const ChironServingConfig& cfg)
    {
        clear();
        const int resolvedNKVH = d.nKVH > 0 ? d.nKVH : d.nH;
        const int resolvedDMKV = d.dModelKV > 0 ? d.dModelKV : resolvedNKVH * d.dH;
        if (!cfg.useScfa || !weights.scfa.present || !weights.scfa.dLoaded
            || !weights.scfa.causalBlock || weights.scfa.k <= 0
            || weights.scfa.k > d.T || weights.scfa.w < 0
            || (int)weights.scfa.D.size() != d.L || d.T <= 1 || d.m <= 0
            || d.V <= 0 || d.L <= 0 || d.nH <= 0 || resolvedNKVH <= 0
            || d.dH <= 0 || d.dModel != d.nH * d.dH
            || resolvedDMKV != resolvedNKVH * d.dH)
            return false;

        T = d.T; m = d.m; V = d.V; L = d.L; nH = d.nH; nKVH = resolvedNKVH;
        dH = d.dH; dModel = d.dModel; dModelKV = resolvedDMKV;
        ffnHidden = d.ffnHidden; k = weights.scfa.k; w = weights.scfa.w;
        historyCapacity = w + 1;
        maxBlockWidth = 0;
        for (int b = 0; b < k; ++b)
            maxBlockWidth = std::max(maxBlockWidth, blockWidth(b));
        if (maxBlockWidth <= 0 || historyCapacity <= 0) return false;

        qkNorm = cfg.qkNorm;
        fuseAttnReln = cfg.fuseAttnReln;
        fuseAttnPerLayer = cfg.fuseAttnPerLayer;
        whiscCoupling = cfg.whiscCoupling;
        epsReln = cfg.epsReln;
        rotThetaMax = cfg.rotThetaMax;
        whiscClamp = cfg.whiscClamp;
        modelIdentity = &weights;

        for (int l = 0; l < L; ++l)
        {
            Layer* layer = new Layer();
            layers.push_back(layer);
            if (!layer->allocate(k, m, dModelKV, maxBlockWidth, historyCapacity))
            {
                clear();
                return false;
            }
        }
        isReady = true;
        return resetState();
    }

    bool resetState()
    {
        if (!isReady) return false;
        for (size_t l = 0; l < layers.size(); ++l)
            if (!layers[l]->zero()) return false;
        position = 0;
        completedBlocks = 0;
        blockFill = 0;
        historyCount = 0;
        historyWrite = 0;
        return true;
    }

    bool prepareCapture(int promptLength)
    {
        if (!isReady || promptLength <= 0 || promptLength >= T) return false;
        if (!resetState()) return false;
        position = promptLength;
        completedBlocks = 0;
        while (completedBlocks + 1 < k
               && promptLength >= blockBegin(completedBlocks + 1))
            ++completedBlocks;
        blockFill = promptLength - blockBegin(completedBlocks);
        historyCount = std::min(promptLength, historyCapacity);
        historyWrite = historyCount % historyCapacity;
        return blockFill >= 0 && blockFill < blockWidth(completedBlocks);
    }

    bool captureLayer(int layerIndex, const ChironEvalScratch& s)
    {
        if (layerIndex < 0 || layerIndex >= L || position <= 0 || position >= T)
            return false;
        Layer& layer = *layers[layerIndex];
        if (completedBlocks > 0)
        {
            glades::gpu::device_memcpy_d2d(
                layer.qCompr.data(), s.scfa_qcompr.data(),
                (size_t)completedBlocks * m * sizeof(float));
            glades::gpu::device_memcpy_d2d(
                layer.kCache.data(), s.scfa_inner_sK.data(),
                (size_t)completedBlocks * dModelKV * sizeof(float));
            glades::gpu::device_memcpy_d2d(
                layer.vCache.data(), s.scfa_inner_sV.data(),
                (size_t)completedBlocks * dModelKV * sizeof(float));
            glades::gpu::device_memcpy_d2d(
                layer.yLast.data(),
                s.scfa_ycompr.data() + (size_t)(completedBlocks - 1) * m,
                (size_t)m * sizeof(float));
        }
        if (blockFill > 0)
            glades::gpu::device_memcpy_d2d(
                layer.qBlock.data(),
                s.q.data() + (size_t)blockBegin(completedBlocks) * m,
                (size_t)blockFill * m * sizeof(float));
        if (historyCount > 0)
            glades::gpu::device_memcpy_d2d(
                layer.qPerpRing.data(),
                s.scfa_qperp.data() + (size_t)(position - historyCount) * m,
                (size_t)historyCount * m * sizeof(float));
        return true;
    }

    static bool captureLayerCallback(void* context, int layerIndex,
                                     const ChironEvalScratch& s)
    {
        Impl* self = static_cast<Impl*>(context);
        return self && self->captureLayer(layerIndex, s);
    }

    bool cloneState(const Impl& source)
    {
        clear();
        if (!source.isReady) return false;

        T=source.T; m=source.m; V=source.V; L=source.L; nH=source.nH;
        nKVH=source.nKVH; dH=source.dH; dModel=source.dModel;
        dModelKV=source.dModelKV; ffnHidden=source.ffnHidden;
        k=source.k; w=source.w; maxBlockWidth=source.maxBlockWidth;
        historyCapacity=source.historyCapacity; position=source.position;
        completedBlocks=source.completedBlocks; blockFill=source.blockFill;
        historyCount=source.historyCount; historyWrite=source.historyWrite;
        qkNorm=source.qkNorm; fuseAttnReln=source.fuseAttnReln;
        fuseAttnPerLayer=source.fuseAttnPerLayer;
        whiscCoupling=source.whiscCoupling; epsReln=source.epsReln;
        rotThetaMax=source.rotThetaMax; whiscClamp=source.whiscClamp;
        modelIdentity=source.modelIdentity;

        for (int l = 0; l < L; ++l)
        {
            Layer* dst = new Layer();
            layers.push_back(dst);
            if (!dst->allocate(k, m, dModelKV, maxBlockWidth, historyCapacity))
            {
                clear();
                return false;
            }
            const Layer& src = *source.layers[l];
            glades::gpu::device_memcpy_d2d(dst->qCompr.data(), src.qCompr.data(), src.qCompr.bytes());
            glades::gpu::device_memcpy_d2d(dst->yLast.data(), src.yLast.data(), src.yLast.bytes());
            glades::gpu::device_memcpy_d2d(dst->kCache.data(), src.kCache.data(), src.kCache.bytes());
            glades::gpu::device_memcpy_d2d(dst->vCache.data(), src.vCache.data(), src.vCache.bytes());
            glades::gpu::device_memcpy_d2d(dst->qBlock.data(), src.qBlock.data(), src.qBlock.bytes());
            glades::gpu::device_memcpy_d2d(dst->qPerpRing.data(), src.qPerpRing.data(), src.qPerpRing.bytes());
        }
        isReady = true;
        return true;
    }

    bool commitBlock(Layer& layer, const ChironModelWeights& weights,
                     ChironEvalScratch& s, int layerIndex, int blockIndex,
                     int width)
    {
        float* qRow = layer.qCompr.data() + (size_t)blockIndex * m;
        float* kRow = layer.kCache.data() + (size_t)blockIndex * dModelKV;
        float* vRow = layer.vCache.data() + (size_t)blockIndex * dModelKV;
        if (!glades::gpu::scfa_block_compress(layer.qBlock.data(), width, m, 1,
                                               1.0f, 0.0f, qRow)) return false;
        if (!glades::gpu::sgemm_rowmajor(1, dModel, m, 1.0f,
                qRow, m, weights.Wq[layerIndex]->data(), dModel,
                0.0f, s.scfa_inner_sQ.data(), dModel)) return false;
        if (!glades::gpu::sgemm_rowmajor(1, dModelKV, m, 1.0f,
                qRow, m, weights.Wk[layerIndex]->data(), dModelKV,
                0.0f, kRow, dModelKV)) return false;
        if (!glades::gpu::sgemm_rowmajor(1, dModelKV, m, 1.0f,
                qRow, m, weights.Wv[layerIndex]->data(), dModelKV,
                0.0f, vRow, dModelKV)) return false;
        if (qkNorm)
        {
            if (!glades::gpu::qknorm_forward_gpu(s.scfa_inner_sQ.data(),
                    s.qknorm_invNorm.data(), 1, nH, dH, 1e-6f)) return false;
            if (!glades::gpu::qknorm_forward_gpu(kRow,
                    s.qknorm_invNorm.data(), 1, nKVH, dH, 1e-6f)) return false;
            if (!glades::gpu::scale_q_per_head(s.scfa_inner_sQ.data(),
                    s.qknorm_gamma_scale.data() + (size_t)layerIndex * nH,
                    1, nH, dH)) return false;
        }
        const float invSqrt = 1.0f / std::sqrt((float)dH);
        if (!glades::gpu::kv_attention_incremental(
                s.scfa_inner_sQ.data(), layer.kCache.data(), layer.vCache.data(),
                s.scfa_inner_sP.data(), NULL, nH, nKVH, dH, dModelKV,
                k, blockIndex, invSqrt, s.scfa_inner_sO.data())) return false;
        if (!glades::gpu::sgemm_rowmajor(1, m, dModel, 1.0f,
                s.scfa_inner_sO.data(), dModel, weights.Wo[layerIndex]->data(), m,
                0.0f, layer.yLast.data(), m)) return false;
        return true;
    }

    bool consume(const ChironModelDims& d, const ChironModelWeights& weights,
                 const ChironServingConfig& cfg, ChironEvalScratch& s,
                 int token, bool produceLogits)
    {
        if (!matches(d, weights, cfg, s) || position < 0 || position >= T
            || completedBlocks < 0 || completedBlocks >= k)
            return false;

        const int blockIndex = completedBlocks;
        const int width = blockWidth(blockIndex);
        if (blockFill < 0 || blockFill >= width
            || position != blockBegin(blockIndex) + blockFill)
            return false;

        if (token < 0 || token >= V) token = 0;
        if (!s.d_tokens.upload(&token, 1)) return false;
        if (!glades::gpu::embedding_gather(weights.E.data(), s.d_tokens.data(),
                                            1, V, m, s.q.data())) return false;
        glades::gpu::device_memset_bytes(s.p.data(), 0, (size_t)m * sizeof(float));

        const int nextHistoryCount = std::min(historyCount + 1, historyCapacity);
        const int nextHistoryWrite = (historyWrite + 1) % historyCapacity;
        const int historyStart = (nextHistoryWrite - nextHistoryCount + historyCapacity)
                               % historyCapacity;
        const bool completesBlock = blockFill + 1 == width;

        const bool applyFuse = fuseAttnPerLayer
            && !weights.gamma_p.empty() && !weights.beta_p.empty()
            && (int)weights.gamma_p.size() == L && (int)weights.beta_p.size() == L;
        float fuseAlpha = 0.0f;
        if (applyFuse)
            fuseAlpha = (1.0f / std::sqrt((float)L)) * ((float)k / (float)T);

        for (int l = 0; l < L; ++l)
        {
            Layer& layer = *layers[l];
            glades::gpu::device_memcpy_d2d(
                layer.qBlock.data() + (size_t)blockFill * m,
                s.q.data(), (size_t)m * sizeof(float));

            if (blockIndex == 0)
                glades::gpu::device_memset_bytes(s.scfa_qpar.data(), 0,
                                                 (size_t)m * sizeof(float));
            else if (!glades::gpu::scfa_lag_row(
                    layer.qCompr.data() + (size_t)(blockIndex - 1) * m,
                    m, blockWidth(blockIndex - 1), s.scfa_qpar.data())) return false;

            glades::gpu::device_memcpy_d2d(s.q_tmp.data(), s.q.data(),
                                            (size_t)m * sizeof(float));
            if (!glades::gpu::axpy(-1.0f, s.scfa_qpar.data(),
                                   s.q_tmp.data(), m)) return false;
            glades::gpu::device_memcpy_d2d(
                layer.qPerpRing.data() + (size_t)historyWrite * m,
                s.q_tmp.data(), (size_t)m * sizeof(float));

            const int firstRows = std::min(nextHistoryCount,
                                            historyCapacity - historyStart);
            glades::gpu::device_memcpy_d2d(
                s.scfa_qperp.data(),
                layer.qPerpRing.data() + (size_t)historyStart * m,
                (size_t)firstRows * m * sizeof(float));
            if (firstRows < nextHistoryCount)
                glades::gpu::device_memcpy_d2d(
                    s.scfa_qperp.data() + (size_t)firstRows * m,
                    layer.qPerpRing.data(),
                    (size_t)(nextHistoryCount - firstRows) * m * sizeof(float));

            if (!glades::gpu::scfa_depthwise_causal_conv_fwd(
                    s.scfa_qperp.data(), weights.scfa.D[l]->data(),
                    nextHistoryCount, m, w, s.scfa_yperp.data())) return false;

            if (blockIndex == 0)
                glades::gpu::device_memset_bytes(s.scfa_ypar.data(), 0,
                                                 (size_t)m * sizeof(float));
            else if (!glades::gpu::scfa_lag_row(
                    layer.yLast.data(), m, blockWidth(blockIndex - 1),
                    s.scfa_ypar.data())) return false;
            if (!glades::gpu::axpy(1.0f,
                    s.scfa_yperp.data() + (size_t)(nextHistoryCount - 1) * m,
                    s.scfa_ypar.data(), m)) return false;
            if (!glades::gpu::axpy(1.0f, s.scfa_ypar.data(), s.p.data(), m)) return false;

            if (ffnHidden > 0 && !glades::gpu::chiron_ffn_shear_forward(
                    s.q.data(), s.p.data(), weights.ffnGate[l]->data(),
                    weights.ffnUp[l]->data(), weights.ffnDown[l]->data(),
                    1, m, ffnHidden, 1.0f, s.ffnGate.data(),
                    s.ffnUp.data(), s.ffnHidden.data())) return false;

            if (fuseAttnReln && l == L - 1)
                if (!glades::gpu::axpy(1.0f, s.p.data(), s.q.data(), m)) return false;

            if (applyFuse)
            {
                if (!glades::gpu::chiron_reln_forward(
                        s.p.data(), s.p_norm.data(),
                        s.stats_p.data() + (size_t)l * T * 2u,
                        weights.gamma_p[l]->data(), weights.beta_p[l]->data(),
                        1, m, epsReln)) return false;
                if (!glades::gpu::axpy(fuseAlpha, s.p_norm.data(), s.q.data(), m)) return false;
            }

            if (whiscCoupling)
            {
                const float* aw = s.whisc_a.data() + (size_t)l * m;
                if (!glades::gpu::chiron_rot_coeffs(
                        s.rotPhiGpu[l]->data(), rotThetaMax, 1.0f, m,
                        s.rot_a.data(), s.rot_c.data())) return false;
                if (!glades::gpu::chiron_whisc_fold_coeffs(
                        s.rot_a.data(), s.rot_c.data(), aw, m)) return false;
                if (!glades::gpu::chiron_rot_forward(
                        s.q.data(), s.p.data(), s.rot_a.data(), s.rot_c.data(),
                        +1.0f, 1, m)) return false;
            }

            if (!glades::gpu::chiron_reln_forward(
                    s.q.data(), s.q_tmp.data(),
                    s.stats.data() + (size_t)l * T * 2u,
                    weights.gamma[l]->data(), weights.beta[l]->data(),
                    1, m, epsReln)) return false;
            glades::gpu::device_memcpy_d2d(s.q.data(), s.q_tmp.data(),
                                            (size_t)m * sizeof(float));

            if (completesBlock
                && !commitBlock(layer, weights, s, l, blockIndex, width)) return false;
        }

        if (produceLogits && !glades::gpu::sgemm_rowmajor_abt(
                1, V, m, 1.0f, s.q.data(), m, weights.E.data(), m,
                0.0f, s.logits.data(), V)) return false;

        ++position;
        historyCount = nextHistoryCount;
        historyWrite = nextHistoryWrite;
        if (completesBlock)
        {
            ++completedBlocks;
            blockFill = 0;
        }
        else
            ++blockFill;
        return true;
    }
};

ChironDecodeCache::ChironDecodeCache() : impl_(new Impl()) {}
ChironDecodeCache::~ChironDecodeCache() { delete impl_; }

bool ChironDecodeCache::allocate(const ChironModelDims& d,
                                 const ChironModelWeights& w,
                                 const ChironServingConfig& cfg)
{
#ifndef GLADES_HAVE_CUDA
    (void)d; (void)w; (void)cfg;
    return false;
#else
    if (!glades::gpu::isAvailable() && !glades::gpu::initDevice()) return false;
    return impl_->allocateFor(d, w, cfg);
#endif
}

bool ChironDecodeCache::reset() { return impl_->resetState(); }
bool ChironDecodeCache::cloneFrom(const ChironDecodeCache& source)
{
    if (&source == this) return true;
    return impl_->cloneState(*source.impl_);
}
bool ChironDecodeCache::ready() const { return impl_->isReady; }
int ChironDecodeCache::position() const { return impl_->position; }
int ChironDecodeCache::completedBlocks() const { return impl_->completedBlocks; }
int ChironDecodeCache::contextLimit() const { return impl_->T; }

bool chiron_decode_step(const ChironModelDims& d,
                        const ChironModelWeights& w,
                        const ChironServingConfig& cfg,
                        ChironEvalScratch& s,
                        int token,
                        ChironDecodeCache& cache)
{
#ifndef GLADES_HAVE_CUDA
    (void)d; (void)w; (void)cfg; (void)s; (void)token; (void)cache;
    return false;
#else
    return cache.impl_->consume(d, w, cfg, s, token, true);
#endif
}

bool chiron_decode_prefill(const ChironModelDims& d,
                           const ChironModelWeights& w,
                           const ChironServingConfig& cfg,
                           ChironEvalScratch& s,
                           const std::vector<int>& promptTokens,
                           ChironDecodeCache& cache)
{
#ifndef GLADES_HAVE_CUDA
    (void)d; (void)w; (void)cfg; (void)s; (void)promptTokens; (void)cache;
    return false;
#else
    const int promptLength = (int)promptTokens.size();
    if (promptLength <= 0 || promptLength >= d.T
        || !cache.impl_->matches(d, w, cfg, s)) return false;
    std::vector<int> input((size_t)d.T, 0);
    for (int i = 0; i < promptLength; ++i)
    {
        int token = promptTokens[(size_t)i];
        input[(size_t)i] = (token >= 0 && token < d.V) ? token : 0;
    }
    if (!cache.impl_->prepareCapture(promptLength)) return false;
    if (!s.d_tokens.upload(&input[0], input.size())) return false;
    if (!chiron_eval_forward_observed(d, w, cfg, s,
            ChironDecodeCache::Impl::captureLayerCallback, cache.impl_)) return false;
    if (promptLength > 1)
        glades::gpu::device_memcpy_d2d(
            s.logits.data(),
            s.logits.data() + (size_t)(promptLength - 1) * d.V,
            (size_t)d.V * sizeof(float));
    return true;
#endif
}

} // namespace chiron
} // namespace glades
