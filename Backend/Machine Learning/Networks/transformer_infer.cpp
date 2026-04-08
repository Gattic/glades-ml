// Transformer decoder-only KV-cache inference for token LM mode.
//
// This is a forward-only incremental decode path:
// - append one token at a time
// - cache K/V for each layer
// - compute logits for the appended token position
//
#include "network.h"
#include "transformer_config.h"
#include "transformer_common_utils.h"
#include "../GMath/gmath.h"
#include "transformer_kernels.h"
#include "tensor_view.h"
#include "Backend/Database/GLogger.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <vector>

#include "logfmt_utils.h"

#ifdef GLADES_HAVE_CUDA
#include "cuda/gpu_device.h"
#include "cuda/gpu_buffer.h"
#include "cuda/gpu_blas.h"
#include "cuda/gpu_kernels.h"
#include "cuda/gpu_dispatch.h"
#include "cuda/gpu_transformer_state.h"
#include <cuda_runtime.h>
#endif

using namespace glades;

namespace {
static inline bool is_finite(float x) { return glades::transformer_kernels::is_finite(x); }
using namespace glades::logfmt;
using glades::transformer_common::bytes_to_human;
using glades::transformer_common::checked_add_size;
using glades::transformer_common::checked_mul_size;
using glades::transformer_common::parse_u64_env;
using glades::transformer_common::ScopedTimerMs;

// === Allocation sizing hardening ===
//
// KV-cache session allocations scale with:
//   O(batch * layers * maxSeqLen * dModelKV)
// and are user-configurable via API inputs (maxSeqLen) and model config.
//
// We MUST:
// - compute sizes with overflow checks
// - enforce a hard cap to avoid OOM / wraparound -> OOB writes
//
// Cap policy:
// - Default cap is conservative and intended for safety, not "train real LLMs".
// - Override via environment variable:
//     GLADES_TRANSFORMER_KV_SESSION_MAX_BYTES
//   where value is an integer byte count (e.g. 4294967296 for 4GiB).
static inline unsigned long long kv_session_max_bytes(const glades::TransformerRunConfig& cfg)
{
	if (cfg.kvSessionMaxBytes > 0ULL)
		return cfg.kvSessionMaxBytes;
	// Default: 2 GiB.
	// Rationale: large enough for typical CPU demo/serving, small enough to prevent
	// accidental multi-tens-of-GB allocations from a bad maxSeqLen.
	static const unsigned long long kDefault = 2ULL * 1024ULL * 1024ULL * 1024ULL;
	unsigned long long v = 0ULL;
	if (parse_u64_env("GLADES_TRANSFORMER_KV_SESSION_MAX_BYTES", v) && v > 0ULL)
		return v;
	return kDefault;
}

static void log_kv_append_event(shmea::GLogger* logger,
                                const char* event,
                                unsigned int seqLen,
                                unsigned int maxSeqLen,
                                unsigned int activeCount,
                                bool emittedLogits,
                                const glades::NNetwork::TransformerKvPerfBreakdown* perf)
{
	if (!logger || !event)
		return;
	std::ostringstream oss;
	oss << "event=" << event;
	append_logfmt_kv(oss, "seq_len", seqLen);
	append_logfmt_kv(oss, "max_seq_len", maxSeqLen);
	append_logfmt_kv(oss, "active", activeCount);
	append_logfmt_kv(oss, "emit_logits", emittedLogits);
	if (perf)
	{
		append_logfmt_kv(oss, "kv_appends", static_cast<unsigned long long>(perf->kvAppends));
		append_logfmt_kv(oss, "kv_ms_total", perf->msTotal);
		append_logfmt_kv(oss, "non_finite_hidden", static_cast<unsigned long long>(perf->nonFiniteHiddenState));
		append_logfmt_kv(oss, "gpu_kernel_launches", perf->gpu.counters.kernelLaunches);
		append_logfmt_kv(oss, "gpu_sync_points", perf->gpu.counters.syncPoints);
		append_logfmt_kv(oss, "gpu_bytes_h2d", perf->gpu.counters.bytesH2D);
		append_logfmt_kv(oss, "gpu_bytes_d2h", perf->gpu.counters.bytesD2H);
		append_logfmt_kv(oss, "gpu_bytes_d2d", perf->gpu.counters.bytesD2D);
		append_logfmt_kv(oss, "gpu_ms_total", perf->gpu.msTotal);
	}
	logger->info("Transformer", shmea::GString(oss.str().c_str()));
}

struct TransformerSessionCommonConfig
{
	unsigned int dModel;
	unsigned int dFF;
	unsigned int nHeads;
	unsigned int nKVHeads;
	unsigned int nLayers;
	unsigned int dHead;
	unsigned int dModelKV;
	unsigned int ffnKind;
	unsigned int ff1Width;
	bool metricsEnabled;
	bool metricsBreakdownEnabled;
	bool metricsLogPerKvAppend;
	bool metricsGpuPerfEnabled;
	float layerNormEps;
	unsigned int normType;
	unsigned int positionalEncoding;
	int ropeDimOverride;
	float ropeTheta;
	unsigned int ffnActivation;
	int padTokenId;
	shmea::GLogger* logger;

	TransformerSessionCommonConfig()
	    : dModel(0u),
	      dFF(0u),
	      nHeads(0u),
	      nKVHeads(0u),
	      nLayers(0u),
	      dHead(0u),
	      dModelKV(0u),
	      ffnKind(0u),
	      ff1Width(0u),
	      metricsEnabled(false),
	      metricsBreakdownEnabled(false),
	      metricsLogPerKvAppend(false),
	      metricsGpuPerfEnabled(false),
	      layerNormEps(0.0f),
	      normType(0u),
	      positionalEncoding(0u),
	      ropeDimOverride(0),
	      ropeTheta(0.0f),
	      ffnActivation(0u),
	      padTokenId(-1),
	      logger(NULL)
	{
	}
};

static NNetworkStatus build_transformer_session_common_config(const char* where,
                                                             unsigned int dModel,
                                                             unsigned int dFF,
                                                             unsigned int nHeads,
                                                             unsigned int nKVHeadsRaw,
                                                             unsigned int nLayers,
                                                             unsigned int ffnKind,
                                                             int padTokenId,
                                                             const glades::TransformerRunConfig& runtimeCfg,
                                                             const glades::NNetwork::TransformerMetricsConfig& metricsCfg,
                                                             shmea::GLogger* logger,
                                                             TransformerSessionCommonConfig& out)
{
	TransformerRuntimeConfigSnapshot runtime;
	NNetworkStatus st = buildTransformerRuntimeConfigSnapshot(where, runtimeCfg, runtime);
	if (!st.ok())
		return st;

	out.dModel = dModel;
	out.dFF = dFF;
	out.nHeads = nHeads;
	out.nKVHeads = (nKVHeadsRaw > 0u ? nKVHeadsRaw : nHeads);
	out.dHead = (out.nHeads > 0u ? (out.dModel / out.nHeads) : 0u);
	out.dModelKV = out.nKVHeads * out.dHead;
	if (out.dHead == 0u || out.dModelKV == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, std::string(where) + ": invalid head dimensions");
	if (out.nHeads > 0u && (out.dModel % out.nHeads) != 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, std::string(where) + ": dModel is not divisible by nHeads");
	if (out.nKVHeads > 0u && (out.nHeads % out.nKVHeads) != 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, std::string(where) + ": nHeads is not divisible by nKVHeads");

	out.nLayers = nLayers;
	out.ffnKind = ffnKind;
	out.ff1Width =
	    (ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFF) : dFF;
	out.metricsEnabled = metricsCfg.enable;
	out.metricsBreakdownEnabled = metricsCfg.enableKvKernelBreakdown;
	out.metricsLogPerKvAppend = metricsCfg.logPerKvAppend;
	out.metricsGpuPerfEnabled = (metricsCfg.enable && metricsCfg.enableGpuPerf);
	out.layerNormEps = runtime.layerNormEps;
	out.normType = runtime.normType;
	out.positionalEncoding = runtime.positionalEncoding;
	out.ropeDimOverride = runtime.ropeDimOverride;
	out.ropeTheta = runtime.ropeTheta;
	out.ffnActivation = runtime.ffnActivation;
	out.padTokenId = padTokenId;
	out.logger = metricsCfg.enable ? logger : NULL;
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

// Use the shared scalar kernels directly (no wrapper shims).
using glades::transformer_kernels::layernorm_into;
using glades::transformer_kernels::layernorm_vec;
using glades::transformer_kernels::linear_into_opt;
using glades::transformer_kernels::linear_vec;
using glades::transformer_kernels::rmsnorm_into;
using glades::transformer_kernels::rmsnorm_vec;
using glades::transformer_kernels::rope_apply_vec;
using glades::transformer_kernels::softmax_stable_inplace;
using glades::transformer_kernels::tied_embedding_logits_into;

// Stable softmax with an optional key mask (keyAllowed[i]==0 => score is ignored and prob is 0).
// Operates on the first `n` entries of `scores` and does not allocate/resize.
static inline void softmax_stable_keymask_inplace(float* scores, size_t n, const unsigned char* keyAllowed)
{
	if (!scores || n == 0u)
		return;

	if (!keyAllowed)
	{
		// Unmasked stable softmax on raw buffer (no allocations).
		float maxv = scores[0];
		for (size_t i = 1; i < n; ++i)
			if (scores[i] > maxv) maxv = scores[i];

		double sum = 0.0;
		for (size_t i = 0; i < n; ++i)
		{
			const double e = exp(static_cast<double>(scores[i] - maxv));
			scores[i] = static_cast<float>(e);
			sum += e;
		}
		if (sum <= 0.0)
		{
			const float inv = 1.0f / static_cast<float>(n);
			for (size_t i = 0; i < n; ++i)
				scores[i] = inv;
			return;
		}

		const float inv = static_cast<float>(1.0 / sum);
		for (size_t i = 0; i < n; ++i)
			scores[i] *= inv;
		return;
	}

	// max over allowed
	float maxv = 0.0f;
	bool maxInit = false;
	unsigned int allowedCount = 0u;
	for (size_t i = 0; i < n; ++i)
	{
		if (keyAllowed[i] == 0u)
			continue;
		if (!maxInit)
		{
			maxv = scores[i];
			maxInit = true;
		}
		else if (scores[i] > maxv)
			maxv = scores[i];
		++allowedCount;
	}
	if (!maxInit || allowedCount == 0u)
	{
		for (size_t i = 0; i < n; ++i)
			scores[i] = 0.0f;
		return;
	}

	double sum = 0.0;
	for (size_t i = 0; i < n; ++i)
	{
		if (keyAllowed[i] == 0u)
		{
			scores[i] = 0.0f;
			continue;
		}
		const double e = exp(static_cast<double>(scores[i] - maxv));
		scores[i] = static_cast<float>(e);
		sum += e;
	}
	if (sum <= 0.0)
	{
		const float inv = 1.0f / static_cast<float>(allowedCount);
		for (size_t i = 0; i < n; ++i)
			scores[i] = (keyAllowed[i] == 0u) ? 0.0f : inv;
		return;
	}

	const float inv = static_cast<float>(1.0 / sum);
	for (size_t i = 0; i < n; ++i)
		scores[i] *= inv;
}

static inline void softmax_stable_keymask_inplace(std::vector<float>& scores, size_t n, const unsigned char* keyAllowed)
{
	if (scores.empty() || n == 0u)
		return;
	if (n > scores.size())
		n = scores.size();
	softmax_stable_keymask_inplace(&scores[0], n, keyAllowed);
}

// Fused attention for a single head (decoder causal attention at one query position):
// - Pass 1: compute scaled dot scores into `scoreBuf[0..pos]` and track max over allowed keys.
// - Pass 2: compute exp(score-max), accumulate weighted V into `outHead`, normalize by sum.
//
// This replaces the old 3-pass approach (scores -> softmax -> per-dimension gather) and is
// significantly more cache-friendly while remaining deterministic and allocation-free.
static inline void attention_head_fused_softmax_weighted_sum(float* outHead,
                                                            float* scoreBuf,
                                                            const float* qh,
                                                            const glades::Tensor2DView<const float>& kLayer,
                                                            const glades::Tensor2DView<const float>& vLayer,
                                                            unsigned int dHead,
                                                            unsigned int kvHead,
                                                            unsigned int pos,
                                                            const unsigned char* keyAllowed,
                                                            float invSqrt)
{
	if (!outHead || !scoreBuf || !qh || !kLayer.ok() || !vLayer.ok() || dHead == 0u)
		return;
	if (kLayer.cols != vLayer.cols || kLayer.rowStride != vLayer.rowStride || kLayer.rows != vLayer.rows)
		return;

	const size_t colOff = static_cast<size_t>(kvHead) * static_cast<size_t>(dHead);
	if (colOff + static_cast<size_t>(dHead) > kLayer.cols)
		return;

	for (unsigned int i = 0; i < dHead; ++i)
		outHead[i] = 0.0f;

	const size_t scoreN = static_cast<size_t>(pos) + 1u;
	float maxScore = -std::numeric_limits<float>::infinity();
	bool any = false;

	for (unsigned int u = 0; u <= pos; ++u)
	{
		if (static_cast<size_t>(u) >= kLayer.rows)
			return;
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		const float* kRow = kLayer.row(static_cast<size_t>(u)) + colOff;
		const float s = glades::transformer_kernels::dot_f32(qh, kRow, dHead) * invSqrt;
		scoreBuf[u] = s;
		if (!any || s > maxScore)
			maxScore = s;
		any = true;
	}

	if (!any)
		return;

	double sum = 0.0;
	for (unsigned int u = 0; u <= pos; ++u)
	{
		if (static_cast<size_t>(u) >= vLayer.rows)
			return;
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		const double w = exp(static_cast<double>(scoreBuf[u] - maxScore));
		sum += w;
		const float* vRow = vLayer.row(static_cast<size_t>(u)) + colOff;
		const float wf = static_cast<float>(w);
		glades::transformer_kernels::axpy_f32(outHead, vRow, wf, dHead);
	}

	if (!(sum > 0.0) || !std::isfinite(sum))
	{
		for (unsigned int i = 0; i < dHead; ++i)
			outHead[i] = 0.0f;
		return;
	}

	const float inv = static_cast<float>(1.0 / sum);
	for (unsigned int i = 0; i < dHead; ++i)
		outHead[i] *= inv;
}

// FP16 KV-cache variant of fused attention (kLayer/vLayer contain IEEE754 binary16).
static inline void attention_head_fused_softmax_weighted_sum_f16(float* outHead,
                                                                float* scoreBuf,
                                                                const float* qh,
                                                                const glades::Tensor2DView<const uint16_t>& kLayer,
                                                                const glades::Tensor2DView<const uint16_t>& vLayer,
                                                                unsigned int dHead,
                                                                unsigned int kvHead,
                                                                unsigned int pos,
                                                                const unsigned char* keyAllowed,
                                                                float invSqrt)
{
	if (!outHead || !scoreBuf || !qh || !kLayer.ok() || !vLayer.ok() || dHead == 0u)
		return;
	if (kLayer.cols != vLayer.cols || kLayer.rowStride != vLayer.rowStride || kLayer.rows != vLayer.rows)
		return;

	const size_t colOff = static_cast<size_t>(kvHead) * static_cast<size_t>(dHead);
	if (colOff + static_cast<size_t>(dHead) > kLayer.cols)
		return;

	for (unsigned int i = 0; i < dHead; ++i)
		outHead[i] = 0.0f;

	float maxScore = -std::numeric_limits<float>::infinity();
	bool any = false;

	for (unsigned int u = 0; u <= pos; ++u)
	{
		if (static_cast<size_t>(u) >= kLayer.rows)
			return;
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		const uint16_t* kRow = kLayer.row(static_cast<size_t>(u)) + colOff;
		double dot = 0.0;
		for (unsigned int i = 0; i < dHead; ++i)
		{
			const float kv = glades::transformer_kernels::half_to_float(kRow[i]);
			dot += static_cast<double>(qh[i]) * static_cast<double>(kv);
		}
		const float s = static_cast<float>(dot) * invSqrt;
		scoreBuf[u] = s;
		if (!any || s > maxScore)
			maxScore = s;
		any = true;
	}

	if (!any)
		return;

	double sum = 0.0;
	for (unsigned int u = 0; u <= pos; ++u)
	{
		if (static_cast<size_t>(u) >= vLayer.rows)
			return;
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		const double w = exp(static_cast<double>(scoreBuf[u] - maxScore));
		sum += w;
		const uint16_t* vRow = vLayer.row(static_cast<size_t>(u)) + colOff;
		const float wf = static_cast<float>(w);
		for (unsigned int i = 0; i < dHead; ++i)
		{
			const float vv = glades::transformer_kernels::half_to_float(vRow[i]);
			outHead[i] += wf * vv;
		}
	}

	if (!(sum > 0.0) || !std::isfinite(sum))
	{
		for (unsigned int i = 0; i < dHead; ++i)
			outHead[i] = 0.0f;
		return;
	}

	const float inv = static_cast<float>(1.0 / sum);
	for (unsigned int i = 0; i < dHead; ++i)
		outHead[i] *= inv;
}

// Low-precision KV-cache variant of fused attention:
// - kLayer/vLayer contain uint16_t values interpreted according to lowpDType:
//   - transformer_kernels::LOWP_F16: IEEE754 binary16
//   - transformer_kernels::LOWP_BF16: bfloat16
static inline void attention_head_fused_softmax_weighted_sum_lowp(float* outHead,
                                                                  float* scoreBuf,
                                                                  const float* qh,
                                                                  const glades::Tensor2DView<const uint16_t>& kLayer,
                                                                  const glades::Tensor2DView<const uint16_t>& vLayer,
                                                                  int lowpDType,
                                                                  unsigned int dHead,
                                                                  unsigned int kvHead,
                                                                  unsigned int pos,
                                                                  const unsigned char* keyAllowed,
                                                                  float invSqrt)
{
	if (!outHead || !scoreBuf || !qh || !kLayer.ok() || !vLayer.ok() || dHead == 0u)
		return;
	if (kLayer.cols != vLayer.cols || kLayer.rowStride != vLayer.rowStride || kLayer.rows != vLayer.rows)
		return;

	const size_t colOff = static_cast<size_t>(kvHead) * static_cast<size_t>(dHead);
	if (colOff + static_cast<size_t>(dHead) > kLayer.cols)
		return;

	for (unsigned int i = 0; i < dHead; ++i)
		outHead[i] = 0.0f;

	float maxScore = -std::numeric_limits<float>::infinity();
	bool any = false;

	for (unsigned int u = 0; u <= pos; ++u)
	{
		if (static_cast<size_t>(u) >= kLayer.rows)
			return;
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		const uint16_t* kRow = kLayer.row(static_cast<size_t>(u)) + colOff;
		double dot = 0.0;
		for (unsigned int i = 0; i < dHead; ++i)
		{
			const float kv = glades::transformer_kernels::lowp_to_float(kRow[i], lowpDType);
			dot += static_cast<double>(qh[i]) * static_cast<double>(kv);
		}
		const float s = static_cast<float>(dot) * invSqrt;
		scoreBuf[u] = s;
		if (!any || s > maxScore)
			maxScore = s;
		any = true;
	}

	if (!any)
		return;

	double sum = 0.0;
	for (unsigned int u = 0; u <= pos; ++u)
	{
		if (static_cast<size_t>(u) >= vLayer.rows)
			return;
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		const double w = exp(static_cast<double>(scoreBuf[u] - maxScore));
		sum += w;
		const uint16_t* vRow = vLayer.row(static_cast<size_t>(u)) + colOff;
		const float wf = static_cast<float>(w);
		for (unsigned int i = 0; i < dHead; ++i)
		{
			const float vv = glades::transformer_kernels::lowp_to_float(vRow[i], lowpDType);
			outHead[i] += wf * vv;
		}
	}

	if (!(sum > 0.0) || !std::isfinite(sum))
	{
		for (unsigned int i = 0; i < dHead; ++i)
			outHead[i] = 0.0f;
		return;
	}

	const float inv = static_cast<float>(1.0 / sum);
	for (unsigned int i = 0; i < dHead; ++i)
		outHead[i] *= inv;
}

// ---------------------------------------------------------------------------
// GPU inference state (KV cache + scratch on device)
// ---------------------------------------------------------------------------
#ifdef GLADES_HAVE_CUDA

struct GpuInferState
{
	bool initialized;

	// Shared weights (NOT owned - points to NNetwork::gpuTransformerWeights).
	glades::gpu::GpuTransformerWeights* weights;

	// Per-session KV cache: each [nLayers * maxLen * dModelKV]
	glades::gpu::GpuBuffer<float> k;
	glades::gpu::GpuBuffer<float> v;

	// Key validity mask on GPU: [maxLen]
	glades::gpu::GpuBuffer<unsigned char> keyValid;

	// Single-token scratch buffers.
	glades::gpu::GpuBuffer<float> h;           // [dModel]
	glades::gpu::GpuBuffer<float> x1;          // [dModel]
	glades::gpu::GpuBuffer<float> q;           // [dModel]
	glades::gpu::GpuBuffer<float> kvec;        // [dModelKV]
	glades::gpu::GpuBuffer<float> vvec;        // [dModelKV]
	glades::gpu::GpuBuffer<float> attnConcat;  // [dModel]
	glades::gpu::GpuBuffer<float> attnOut;     // [dModel]
	glades::gpu::GpuBuffer<float> ffPre;       // [ff1Width]
	glades::gpu::GpuBuffer<float> ffAct;       // [dFF]
	glades::gpu::GpuBuffer<float> ffOut;       // [dModel]
	glades::gpu::GpuBuffer<float> logits;      // [vocabSize]

	// Attention scores scratch: [nHeads * maxLen]
	glades::gpu::GpuBuffer<float> attnScores;

	// RoPE invFreq on GPU: [ropeDim/2]
	glades::gpu::GpuBuffer<float> ropeInvFreq;

	// LN mean/invStd scratch (single-row norms): [1] each
	glades::gpu::GpuBuffer<float> lnMean;
	glades::gpu::GpuBuffer<float> lnInvStd;

	// Sinusoidal PE scratch: [dModel]
	glades::gpu::GpuBuffer<float> peScratch;

	// Token ID scratch: [1]
	glades::gpu::GpuBuffer<int> tokenIdBuf;

	// Cached dimensions.
	unsigned int maxLen;
	unsigned int dModel;
	unsigned int dFF;
	unsigned int dModelKV;
	unsigned int nHeads;
	unsigned int nKVHeads;
	unsigned int nLayers;
	unsigned int vocabSize;

	GpuInferState()
	    : initialized(false), weights(NULL),
	      maxLen(0), dModel(0), dFF(0), dModelKV(0),
	      nHeads(0), nKVHeads(0), nLayers(0), vocabSize(0) {}
	~GpuInferState() { freeState(); }

	bool allocate(glades::gpu::GpuTransformerWeights* w,
	              unsigned int maxLen_, unsigned int dModel_, unsigned int dFF_,
	              unsigned int dModelKV_, unsigned int nHeads_, unsigned int nKVHeads_,
	              unsigned int nLayers_, unsigned int ff1Width_, unsigned int vocabSize_)
	{
		weights = w;
		maxLen = maxLen_;
		dModel = dModel_;
		dFF = dFF_;
		dModelKV = dModelKV_;
		nHeads = nHeads_;
		nKVHeads = nKVHeads_;
		nLayers = nLayers_;
		vocabSize = vocabSize_;

		size_t kvTotal = (size_t)nLayers * maxLen * dModelKV;
		if (!k.allocate(kvTotal) || !v.allocate(kvTotal)) return false;
		k.zero();
		v.zero();

		if (!keyValid.allocate(maxLen)) return false;
		// Initialize all positions to valid (1).
		std::vector<unsigned char> ones(maxLen, 1u);
		if (!keyValid.upload(&ones[0], maxLen)) return false;

		if (!h.allocate(dModel) || !x1.allocate(dModel) || !q.allocate(dModel)) return false;
		if (!kvec.allocate(dModelKV) || !vvec.allocate(dModelKV)) return false;
		if (!attnConcat.allocate(dModel) || !attnOut.allocate(dModel)) return false;
		if (!ffPre.allocate(ff1Width_) || !ffAct.allocate(dFF) || !ffOut.allocate(dModel)) return false;
		if (!logits.allocate(vocabSize)) return false;
		if (!attnScores.allocate((size_t)nHeads * maxLen)) return false;
		if (!lnMean.allocate(1) || !lnInvStd.allocate(1)) return false;
		if (!peScratch.allocate(dModel)) return false;
		if (!tokenIdBuf.allocate(1)) return false;

		initialized = true;
		return true;
	}

	void freeState()
	{
		k.free(); v.free(); keyValid.free();
		h.free(); x1.free(); q.free();
		kvec.free(); vvec.free();
		attnConcat.free(); attnOut.free();
		ffPre.free(); ffAct.free(); ffOut.free();
		logits.free(); attnScores.free();
		lnMean.free(); lnInvStd.free();
		peScratch.free(); tokenIdBuf.free();
		ropeInvFreq.free();
		initialized = false;
		weights = NULL;
	}
};

static void freeGpuInferState(void*& ptr)
{
	if (ptr)
	{
		delete static_cast<GpuInferState*>(ptr);
		ptr = 0;
	}
}

static bool gpu_infer_run_layer_step(const std::vector<double>& ropeInvFreq,
                                     const glades::gpu::GpuTransformerWeights::Block& blk,
                                     GpuInferState& gs,
                                     unsigned int li,
                                     unsigned int pos,
                                     unsigned int dModel,
                                     unsigned int dFF,
                                     unsigned int nHeads,
                                     unsigned int nKVHeads,
                                     unsigned int dHead,
                                     unsigned int dModelKV,
                                     unsigned int ff1Width,
                                     unsigned int ffnKind,
                                     unsigned int ffnActivation,
                                     int normType,
                                     float eps,
                                     bool useRope,
                                     unsigned int ropeDim,
                                     float ropeTheta,
                                     glades::NNetwork::TransformerGpuPerfBreakdown* gpuPerf)
{
	namespace gg = glades::gpu;
	const int dM = static_cast<int>(dModel);
	const int dMKV = static_cast<int>(dModelKV);
	const int dFFi = static_cast<int>(dFF);
	const int ff1W = static_cast<int>(ff1Width);

	if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::rmsnorm_forward(gs.h.data(), blk.ln1Gamma.data(), eps,
		                    1, dM, gs.x1.data(), gs.lnInvStd.data());
	}
	else
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::layernorm_forward(gs.h.data(), blk.ln1Gamma.data(), blk.ln1Beta.data(),
		                      eps, 1, dM, gs.x1.data(), gs.lnMean.data(), gs.lnInvStd.data());
	}

	if (gpuPerf)
		gg::perfRecordKernel(&gpuPerf->counters, 1u);
	gg::sgemv_rowmajor(dM, dM, 1.0f, blk.Wq.data(), dM, gs.x1.data(), 0.0f, gs.q.data());
	if (blk.bq.allocated())
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::add_bias(gs.q.data(), blk.bq.data(), 1, dM);
	}
	if (gpuPerf)
		gg::perfRecordKernel(&gpuPerf->counters, 1u);
	gg::sgemv_rowmajor(dMKV, dM, 1.0f, blk.Wk.data(), dM, gs.x1.data(), 0.0f, gs.kvec.data());
	if (blk.bk.allocated())
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::add_bias(gs.kvec.data(), blk.bk.data(), 1, dMKV);
	}
	if (gpuPerf)
		gg::perfRecordKernel(&gpuPerf->counters, 1u);
	gg::sgemv_rowmajor(dMKV, dM, 1.0f, blk.Wv.data(), dM, gs.x1.data(), 0.0f, gs.vvec.data());
	if (blk.bv.allocated())
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::add_bias(gs.vvec.data(), blk.bv.data(), 1, dMKV);
	}

	if (useRope && ropeDim >= 2u)
	{
		std::vector<float> qCpu(dModel);
		std::vector<float> kvCpu(dModelKV);
		if (gpuPerf)
		{
			gg::perfRecordBytesD2H(&gpuPerf->counters, static_cast<size_t>(dModel + dModelKV) * sizeof(float));
			gg::perfRecordBytesH2D(&gpuPerf->counters, static_cast<size_t>(dModel + dModelKV) * sizeof(float));
		}
		gs.q.download(&qCpu[0], dModel);
		gs.kvec.download(&kvCpu[0], dModelKV);
		for (unsigned int hq = 0; hq < nHeads; ++hq)
			rope_apply_vec(&qCpu[static_cast<size_t>(hq) * dHead], dHead, ropeDim,
			               ropeInvFreq, pos);
		for (unsigned int hk = 0; hk < nKVHeads; ++hk)
			rope_apply_vec(&kvCpu[static_cast<size_t>(hk) * dHead], dHead, ropeDim,
			               ropeInvFreq, pos);
		gs.q.upload(&qCpu[0], dModel);
		gs.kvec.upload(&kvCpu[0], dModelKV);
	}

	{
		const size_t perLayer = static_cast<size_t>(gs.maxLen) * dModelKV;
		const size_t base = static_cast<size_t>(li) * perLayer + static_cast<size_t>(pos) * dModelKV;
		if (gpuPerf)
			gg::perfRecordBytesD2D(&gpuPerf->counters, 2u * static_cast<size_t>(dModelKV) * sizeof(float));
		gpu::device_memcpy_d2d(gs.k.data() + base, gs.kvec.data(), dModelKV * sizeof(float));
		gpu::device_memcpy_d2d(gs.v.data() + base, gs.vvec.data(), dModelKV * sizeof(float));
	}

	{
		glades::gpu::ScopedPerfTimerMs gpuStage(gpuPerf ? &gpuPerf->msAttention : NULL);
		const size_t perLayer = static_cast<size_t>(gs.maxLen) * dModelKV;
		const float* kLayer = gs.k.data() + static_cast<size_t>(li) * perLayer;
		const float* vLayer = gs.v.data() + static_cast<size_t>(li) * perLayer;
		const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dHead)));
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::kv_attention_incremental(
		    gs.q.data(), kLayer, vLayer,
		    gs.attnScores.data(), gs.keyValid.data(),
		    static_cast<int>(nHeads), static_cast<int>(nKVHeads),
		    static_cast<int>(dHead), dMKV, static_cast<int>(gs.maxLen),
		    static_cast<int>(pos), invSqrt, gs.attnConcat.data());
	}

	if (gpuPerf)
		gg::perfRecordKernel(&gpuPerf->counters, 1u);
	gg::sgemv_rowmajor(dM, dM, 1.0f, blk.Wo.data(), dM,
	                   gs.attnConcat.data(), 0.0f, gs.attnOut.data());
	if (blk.bo.allocated())
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::add_bias(gs.attnOut.data(), blk.bo.data(), 1, dM);
	}
	if (gpuPerf)
		gg::perfRecordKernel(&gpuPerf->counters, 1u);
	gg::add_residual(gs.h.data(), gs.attnOut.data(), dM);

	if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::rmsnorm_forward(gs.h.data(), blk.ln2Gamma.data(), eps,
		                    1, dM, gs.x1.data(), gs.lnInvStd.data());
	}
	else
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::layernorm_forward(gs.h.data(), blk.ln2Gamma.data(), blk.ln2Beta.data(),
		                      eps, 1, dM, gs.x1.data(), gs.lnMean.data(), gs.lnInvStd.data());
	}

	if (gpuPerf)
		gg::perfRecordKernel(&gpuPerf->counters, 1u);
	gg::sgemv_rowmajor(ff1W, dM, 1.0f, blk.W1.data(), dM,
	                   gs.x1.data(), 0.0f, gs.ffPre.data());
	if (blk.b1.allocated())
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::add_bias(gs.ffPre.data(), blk.b1.data(), 1, ff1W);
	}
	if (ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU))
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::swiglu_forward(gs.ffPre.data(), 1, dFFi, gs.ffAct.data());
	}
	else if (static_cast<int>(ffnActivation) == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::gelu_forward(gs.ffPre.data(), dFFi, gs.ffAct.data());
	}
	else
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::relu_forward(gs.ffPre.data(), dFFi, gs.ffAct.data());
	}

	if (gpuPerf)
		gg::perfRecordKernel(&gpuPerf->counters, 1u);
	gg::sgemv_rowmajor(dM, dFFi, 1.0f, blk.W2.data(), dFFi,
	                   gs.ffAct.data(), 0.0f, gs.ffOut.data());
	if (blk.b2.allocated())
	{
		if (gpuPerf)
			gg::perfRecordKernel(&gpuPerf->counters, 1u);
		gg::add_bias(gs.ffOut.data(), blk.b2.data(), 1, dM);
	}
	if (gpuPerf)
		gg::perfRecordKernel(&gpuPerf->counters, 1u);
	gg::add_residual(gs.h.data(), gs.ffOut.data(), dM);
	return true;
}

#endif // GLADES_HAVE_CUDA

} // namespace

// Destructor for TransformerLmSession: frees GPU inference state if allocated.
glades::NNetwork::TransformerLmSession::~TransformerLmSession()
{
#ifdef GLADES_HAVE_CUDA
	if (gpuInferState)
	{
		delete static_cast<GpuInferState*>(gpuInferState);
		gpuInferState = 0;
	}
#endif
}

glades::NNetworkStatus glades::NNetwork::transformerLmSessionReset(glades::NNetwork::TransformerLmSession& session, unsigned int maxSeqLen) const
{
	if (netType != TYPE_TRANSFORMER_DECODER)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: requires TYPE_TRANSFORMER_DECODER");
	const glades::TransformerRunConfig& runtimeCfg = trainingConfig.transformer;
	const TransformerMetricsConfig metricsCfg = getTransformerMetricsConfig();
	if (!runtimeCfg.enableTokenEmbedding)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: requires transformer.enableTokenEmbedding");
	if (!tensorTransformer.initialized || !tensorTransformer.tokenModel)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmSessionReset: transformer tensors not initialized for token LM (loadModel or run once)");
	if (maxSeqLen == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: maxSeqLen is 0");

	const TensorTransformerState& tt = tensorTransformer;
	TransformerSessionCommonConfig commonCfg;
	{
		const NNetworkStatus st = build_transformer_session_common_config("transformerLmSessionReset",
		                                                                 tt.dModel,
		                                                                 tt.dFF,
		                                                                 tt.nHeads,
		                                                                 tt.nKVHeads,
		                                                                 tt.nLayers,
		                                                                 tt.ffnKind,
		                                                                 tt.padTokenId,
		                                                                 runtimeCfg,
		                                                                 metricsCfg,
		                                                                 getLogger(),
		                                                                 commonCfg);
		if (!st.ok())
			return st;
	}

	// Allocation sizing safety: overflow checks + hard cap.
	{
		const size_t L = static_cast<size_t>(tt.nLayers);
		const size_t S = static_cast<size_t>(maxSeqLen);
		const size_t K = static_cast<size_t>(commonCfg.dModelKV);
		size_t perLayerElems = 0u;
		size_t totalElems = 0u;
		if (!checked_mul_size(S, K, perLayerElems) || !checked_mul_size(L, perLayerElems, totalElems))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: KV-cache size overflow (maxSeqLen too large)");

		const bool kvLowp = (runtimeCfg.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_F16) ||
		                    (runtimeCfg.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_BF16);
		const unsigned long long elemBytes = kvLowp ? static_cast<unsigned long long>(sizeof(uint16_t))
		                                            : static_cast<unsigned long long>(sizeof(float));
		// K and V both stored: *2.
		const unsigned long long kvBytes =
		    static_cast<unsigned long long>(totalElems) * elemBytes * 2ULL;

		// Scratch (rough upper bound) in bytes (floats + a few byte masks).
		// This is small compared to KV, but include it so the cap reflects total session footprint.
		size_t scratchFloats = 0u;
		// h,x1,x2,q,attnConcat,attnOut,ffOut: 7*dModel
		// kvec,vvec: 2*dModelKV
		// ffPre: ff1Width
		// ffAct: dFF
		// scores: maxSeqLen
		if (!checked_mul_size(static_cast<size_t>(commonCfg.dModel), 7u, scratchFloats))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: scratch size overflow");
		size_t tmp = 0u;
		if (!checked_add_size(scratchFloats, static_cast<size_t>(commonCfg.dModelKV) * 2u, scratchFloats))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: scratch size overflow");
		if (!checked_add_size(scratchFloats, static_cast<size_t>(commonCfg.ff1Width), scratchFloats))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: scratch size overflow");
		if (!checked_add_size(scratchFloats, static_cast<size_t>(commonCfg.dFF), scratchFloats))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: scratch size overflow");
		if (!checked_add_size(scratchFloats, static_cast<size_t>(maxSeqLen), scratchFloats))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: scratch size overflow");

		const unsigned long long scratchBytes =
		    static_cast<unsigned long long>(scratchFloats) * static_cast<unsigned long long>(sizeof(float)) +
		    // keyValid mask: maxSeqLen bytes
		    static_cast<unsigned long long>(maxSeqLen);

		const unsigned long long wantBytes = kvBytes + scratchBytes;
		const unsigned long long cap = kv_session_max_bytes(runtimeCfg);
		if (cap > 0ULL && wantBytes > cap)
		{
			std::ostringstream oss;
			oss << "transformerLmSessionReset: session allocation exceeds cap (want "
			    << bytes_to_human(wantBytes) << ", cap " << bytes_to_human(cap)
			    << "). Reduce maxSeqLen or set transformer.kvSessionMaxBytes / GLADES_TRANSFORMER_KV_SESSION_MAX_BYTES.";
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, oss.str());
		}
	}

	session.reset();
	session.initialized = true;
	session.maxLen = maxSeqLen;
	session.curLen = 0u;
	session.metricsEnabled = commonCfg.metricsEnabled;
	session.metricsBreakdownEnabled = commonCfg.metricsBreakdownEnabled;
	session.metricsLogPerKvAppend = commonCfg.metricsLogPerKvAppend;
	session.metricsGpuPerfEnabled = commonCfg.metricsGpuPerfEnabled;
	session.perf.reset();
	session.dModel = commonCfg.dModel;
	session.dFF = commonCfg.dFF;
	session.nHeads = commonCfg.nHeads;
	session.nKVHeads = commonCfg.nKVHeads;
	session.nLayers = commonCfg.nLayers;
	session.dHead = commonCfg.dHead;
	session.dModelKV = commonCfg.dModelKV;
	session.ffnKind = commonCfg.ffnKind;
	session.ff1Width = commonCfg.ff1Width;
	session.layerNormEps = commonCfg.layerNormEps;
	session.normType = commonCfg.normType;
	session.positionalEncoding = commonCfg.positionalEncoding;
	session.ropeDimOverride = commonCfg.ropeDimOverride;
	session.ropeTheta = commonCfg.ropeTheta;
	session.ffnActivation = commonCfg.ffnActivation;
	session.padTokenId = commonCfg.padTokenId;
	session.logger = commonCfg.logger;

	size_t perLayer = 0u;
	(void)checked_mul_size(static_cast<size_t>(maxSeqLen), static_cast<size_t>(commonCfg.dModelKV), perLayer);
	if (runtimeCfg.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_F16)
		session.kvCacheDType = glades::NNetwork::TransformerLmSession::KV_CACHE_F16;
	else if (runtimeCfg.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_BF16)
		session.kvCacheDType = glades::NNetwork::TransformerLmSession::KV_CACHE_BF16;
	else
		session.kvCacheDType = glades::NNetwork::TransformerLmSession::KV_CACHE_F32;
	if (session.kvCacheDType != glades::NNetwork::TransformerLmSession::KV_CACHE_F32)
	{
		session.k.clear();
		session.v.clear();
		session.k16.assign(static_cast<size_t>(tt.nLayers) * perLayer, static_cast<uint16_t>(0u));
		session.v16.assign(static_cast<size_t>(tt.nLayers) * perLayer, static_cast<uint16_t>(0u));
	}
	else
	{
		session.k16.clear();
		session.v16.clear();
		session.k.assign(static_cast<size_t>(tt.nLayers) * perLayer, 0.0f);
		session.v.assign(static_cast<size_t>(tt.nLayers) * perLayer, 0.0f);
	}
	session.keyValid.assign(static_cast<size_t>(maxSeqLen), 1u);
	session.h.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.x1.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.x2.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.q.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.kvec.assign(static_cast<size_t>(commonCfg.dModelKV), 0.0f);
	session.vvec.assign(static_cast<size_t>(commonCfg.dModelKV), 0.0f);
	session.attnConcat.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.attnOut.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.ffPre.assign(static_cast<size_t>(commonCfg.ff1Width), 0.0f);
	session.ffAct.assign(static_cast<size_t>(commonCfg.dFF), 0.0f);
	session.ffOut.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.scores.assign(static_cast<size_t>(maxSeqLen), 0.0f);
	if (commonCfg.positionalEncoding == static_cast<unsigned int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
	{
		if (session.metricsEnabled)
		{
			const bool hit = (session.posEncCache.sinDModelCached == commonCfg.dModel && !session.posEncCache.sinInvDenomPair.empty());
			if (hit)
				++session.perf.sinCacheHits;
			else
				++session.perf.sinCacheMisses;
		}
		session.posEncCache.ensureSinusoidal(commonCfg.dModel);
	}

	// === GPU inference state allocation ===
#ifdef GLADES_HAVE_CUDA
	// Free any existing GPU infer state from a previous session.
	freeGpuInferState(session.gpuInferState);

	// If GPU weights are already resident (from training or ensureGpuState),
	// allocate a GPU inference state to run the forward pass on GPU.
	if (gpuStateReady && gpuTransformerWeights && gpuTransformerWeights->initialized)
	{
		GpuInferState* gs = new GpuInferState();
		if (gs->allocate(gpuTransformerWeights, maxSeqLen, commonCfg.dModel, commonCfg.dFF,
		                 commonCfg.dModelKV, commonCfg.nHeads, commonCfg.nKVHeads, commonCfg.nLayers,
		                 session.ff1Width, tt.vocabSize))
		{
			session.gpuInferState = static_cast<void*>(gs);
		}
		else
		{
			// GPU allocation failed; fall back to CPU.
			delete gs;
		}
	}
#endif

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::NNetwork::transformerLmSessionAppend(glades::NNetwork::TransformerLmSession& session,
                                                                   unsigned int tokenId,
                                                                   std::vector<float>* outLogits) const
{
	if (!session.initialized)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmSessionAppend: session not initialized (call transformerLmSessionReset)");
	if (session.curLen >= session.maxLen)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmSessionAppend: cache full (maxSeqLen exceeded)");

	const TensorTransformerState& tt = tensorTransformer;
	const unsigned int vocab = tt.vocabSize;
	if (tokenId >= vocab)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionAppend: tokenId out of range");

	// Sanity: ensure the session matches the current model dims.
	if (session.dModel != tt.dModel || session.dFF != tt.dFF || session.nLayers != tt.nLayers || session.nHeads != tt.nHeads)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmSessionAppend: session shape mismatch (reset required)");

	const unsigned int pos = session.curLen;
	const unsigned int dModel = session.dModel;
	const unsigned int dFF = session.dFF;
	const unsigned int nHeads = session.nHeads;
	const unsigned int nKVHeads = session.nKVHeads;
	const unsigned int dHead = session.dHead;
	const unsigned int dModelKV = session.dModelKV;
	const unsigned int groupSize = (nKVHeads > 0u ? (nHeads / nKVHeads) : 0u);
	const int padTokenId = session.padTokenId;

	const bool metricsOn = session.metricsEnabled;
	const bool breakdown = metricsOn && session.metricsBreakdownEnabled;
	if (metricsOn)
		++session.perf.kvAppends;
	ScopedTimerMs tTotal(this, metricsOn, &session.perf.msTotal);

	// Validate scratch sizes (Append must be allocation-free).
	if (session.h.size() != dModel ||
	    session.x1.size() != dModel ||
	    session.x2.size() != dModel ||
	    session.q.size() != dModel ||
	    session.kvec.size() != dModelKV ||
	    session.vvec.size() != dModelKV ||
	    session.attnConcat.size() != dModel ||
	    session.attnOut.size() != dModel ||
	    session.ffPre.size() != session.ff1Width ||
	    session.ffAct.size() != dFF ||
	    session.ffOut.size() != dModel ||
	    session.scores.size() != session.maxLen)
	{
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmSessionAppend: session scratch size mismatch (reset required)");
	}

	// Mark this position as padding/non-padding for attention masking.
	if (pos < session.keyValid.size())
		session.keyValid[pos] = (padTokenId >= 0 && static_cast<int>(tokenId) == padTokenId) ? 0u : 1u;

	// Config knobs
	const float eps = session.layerNormEps;
	const int normType = static_cast<int>(session.normType);
	const int posEnc = static_cast<int>(session.positionalEncoding);
	const int ropeDimOverride = session.ropeDimOverride;
	const float ropeTheta = session.ropeTheta;
	const bool useRope = (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE));

	unsigned int ropeDim = dHead;
	if (ropeDimOverride > 0)
	{
		const unsigned int rd = static_cast<unsigned int>(ropeDimOverride);
		ropeDim = (rd < ropeDim) ? rd : ropeDim;
	}
	if ((ropeDim % 2u) != 0u)
		ropeDim -= 1u;

	// === GPU inference fast-path ===
	// If GPU state is ready, run the entire forward pass on GPU and return.
#ifdef GLADES_HAVE_CUDA
	{
		GpuInferState* gs = static_cast<GpuInferState*>(session.gpuInferState);
		if (gs && gs->initialized && gs->weights && gs->weights->initialized)
		{
			namespace gg = glades::gpu;
			const gg::GpuTransformerWeights& gw = *gs->weights;
			glades::NNetwork::TransformerGpuPerfBreakdown* gpuPerf =
			    (session.metricsGpuPerfEnabled ? &session.perf.gpu : NULL);
			glades::gpu::ScopedPerfTimerMs gpuTotal(gpuPerf ? &gpuPerf->msTotal : NULL);
			const int dM = static_cast<int>(dModel);
			const int dMKV = static_cast<int>(dModelKV);
			const int dH = static_cast<int>(dHead);
			const unsigned int ff1Width = session.ff1Width;
			const int ff1W = static_cast<int>(ff1Width);
			const int dFFi = static_cast<int>(dFF);

			// Update key validity mask on GPU for this position.
			{
				if (gpuPerf)
					gg::perfRecordBytesH2D(&gpuPerf->counters, 1u);
				unsigned char kv = (padTokenId >= 0 && static_cast<int>(tokenId) == padTokenId) ? 0u : 1u;
				gpu::device_memcpy_h2d(gs->keyValid.data() + pos, &kv, 1);
			}

			// --- Embedding ---
			// h[dModel] = tokE[tokenId, :]
			{
				glades::gpu::ScopedPerfTimerMs gpuStage(gpuPerf ? &gpuPerf->msEmbed : NULL);
				int tid = static_cast<int>(tokenId);
				if (gpuPerf)
				{
					gg::perfRecordBytesH2D(&gpuPerf->counters, sizeof(int));
					gg::perfRecordKernel(&gpuPerf->counters, 1u);
				}
				gs->tokenIdBuf.upload(&tid, 1);
				gg::embedding_gather(gw.tokE.data(), gs->tokenIdBuf.data(),
				                     1, static_cast<int>(vocab), dM, gs->h.data());
			}

			// --- Positional encoding ---
			if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
			{
				glades::gpu::ScopedPerfTimerMs gpuStage(gpuPerf ? &gpuPerf->msPosEnc : NULL);
				// Compute sinusoidal PE on CPU, upload, add to h on GPU.
				session.posEncCache.ensureSinusoidal(dModel);
				std::vector<float> peVec(dModel, 0.0f);
				for (unsigned int i = 0; i < dModel; ++i)
				{
					const unsigned int pairIdx = i / 2u;
					if (pairIdx < session.posEncCache.sinInvDenomPair.size())
					{
						const double angle = static_cast<double>(pos) * session.posEncCache.sinInvDenomPair[pairIdx];
						peVec[i] = (i % 2u == 0u) ? static_cast<float>(sin(angle)) : static_cast<float>(cos(angle));
					}
				}
				if (gpuPerf)
				{
					gg::perfRecordBytesH2D(&gpuPerf->counters, static_cast<size_t>(dModel) * sizeof(float));
					gg::perfRecordKernel(&gpuPerf->counters, 1u);
				}
				gs->peScratch.upload(&peVec[0], dModel);
				gg::add_residual(gs->h.data(), gs->peScratch.data(), dM);
			}
			// RoPE is applied after QKV projection, not here.
			// POSENC_NONE: nothing to do.
			if (useRope && ropeDim >= 2u)
				session.posEncCache.ensureRope(ropeDim, ropeTheta);

			// --- Per-layer forward ---
			for (unsigned int li = 0; li < tt.nLayers; ++li)
			{
				const gg::GpuTransformerWeights::Block& gb = gw.blocks[li];
				gpu_infer_run_layer_step(session.posEncCache.ropeInvFreq, gb, *gs, li, pos, dModel, dFF,
				                        nHeads, nKVHeads, dHead, dModelKV,
				                        session.ff1Width, session.ffnKind, session.ffnActivation,
				                        normType, eps, useRope, ropeDim, ropeTheta, gpuPerf);
			}

			// --- Logits: tied embedding ---
			if (outLogits)
			{
				glades::gpu::ScopedPerfTimerMs gpuStage(gpuPerf ? &gpuPerf->msAttention : NULL);
				// logits[vocab] = tokE[vocab, dModel] * h[dModel] + lmBias[vocab]
				if (gpuPerf)
					gg::perfRecordKernel(&gpuPerf->counters, 1u);
				gg::sgemv_rowmajor(static_cast<int>(vocab), dM, 1.0f,
				                   gw.tokE.data(), dM, gs->h.data(),
				                   0.0f, gs->logits.data());
				if (gw.lmBias.allocated())
				{
					if (gpuPerf)
						gg::perfRecordKernel(&gpuPerf->counters, 1u);
					gg::add_bias(gs->logits.data(), gw.lmBias.data(), 1, static_cast<int>(vocab));
				}

				// Download logits to CPU.
				if (outLogits->size() != vocab)
					outLogits->resize(vocab);
				if (gpuPerf)
					gg::perfRecordBytesD2H(&gpuPerf->counters, static_cast<size_t>(vocab) * sizeof(float));
				gs->logits.download(&(*outLogits)[0], vocab);
			}

			session.curLen += 1u;
			if (gpuPerf)
				lastTransformerInferGpuPerf = *gpuPerf;
			if (gpuPerf && metricsOn && session.logger && transformerMetricsCfg.logGpuInferSummary)
				log_kv_append_event(session.logger, "transformer_gpu_kv_append", session.curLen, session.maxLen, 1u, outLogits != NULL, &session.perf);
			return NNetworkStatus(NNetworkStatus::OK, std::string());
		}
	}
#endif // GLADES_HAVE_CUDA

	// RoPE invFreq cache (depends only on ropeDim and ropeTheta).
	if (useRope && ropeDim >= 2u)
	{
		if (metricsOn)
		{
			const bool hit =
			    (session.posEncCache.ropeDimCached == ropeDim && session.posEncCache.ropeThetaCached == ropeTheta && !session.posEncCache.ropeInvFreq.empty());
			if (hit)
				++session.perf.ropeCacheHits;
			else
				++session.perf.ropeCacheMisses;
		}
		session.posEncCache.ensureRope(ropeDim, ropeTheta);
	}

	// h: current token hidden state (starts as embedding)
	std::vector<float, glades::AlignedAllocator<float, 64> >& h = session.h;
	{
		ScopedTimerMs t(this, breakdown, &session.perf.msEmbed);
		const size_t eOff = static_cast<size_t>(tokenId) * static_cast<size_t>(dModel);
		for (unsigned int i = 0; i < dModel; ++i)
			h[i] = tt.tokE[eOff + i];
	}

	// Positional encoding must match training.
	if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
	{
		ScopedTimerMs t(this, breakdown, &session.perf.msPosEnc);
		if (metricsOn)
		{
			const bool hit = (session.posEncCache.sinDModelCached == dModel && !session.posEncCache.sinInvDenomPair.empty());
			if (hit)
				++session.perf.sinCacheHits;
			else
				++session.perf.sinCacheMisses;
		}
		session.posEncCache.ensureSinusoidal(dModel);
		glades::transformer_kernels::add_sinusoidal_positional_encoding_inplace(&h[0], pos, dModel, session.posEncCache.sinInvDenomPair);
	}
	else if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_NONE) ||
	         posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE))
	{
		// no-op here
	}
	else
	{
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionAppend: unknown positionalEncoding");
	}

	std::vector<float, glades::AlignedAllocator<float, 64> >& x1 = session.x1;
	std::vector<float, glades::AlignedAllocator<float, 64> >& x2 = session.x2;
	std::vector<float, glades::AlignedAllocator<float, 64> >& q = session.q;
	std::vector<float, glades::AlignedAllocator<float, 64> >& kvec = session.kvec;
	std::vector<float, glades::AlignedAllocator<float, 64> >& vvec = session.vvec;
	std::vector<float, glades::AlignedAllocator<float, 64> >& attnConcat = session.attnConcat;
	std::vector<float, glades::AlignedAllocator<float, 64> >& attnOut = session.attnOut;
	std::vector<float, glades::AlignedAllocator<float, 64> >& ffPre = session.ffPre;
	std::vector<float, glades::AlignedAllocator<float, 64> >& ffAct = session.ffAct;
	std::vector<float, glades::AlignedAllocator<float, 64> >& ffOut = session.ffOut;
	std::vector<float, glades::AlignedAllocator<float, 64> >& scores = session.scores;

	for (unsigned int li = 0; li < tt.nLayers; ++li)
	{
		const TensorTransformerState::Block& b = tt.blocks[li];

		// Norm1
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msNorm);
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
				rmsnorm_into(&h[0], dModel, b.ln1Gamma, b.ln1Beta, eps, &x1[0]);
			else
				layernorm_into(&h[0], dModel, b.ln1Gamma, b.ln1Beta, eps, &x1[0]);
		}

		// Projections
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msProjQKV);
			linear_into_opt(&x1[0], dModel, b.Wq, b.bq, dModel, &q[0]);
			linear_into_opt(&x1[0], dModel, b.Wk, b.bk, dModelKV, &kvec[0]);
			linear_into_opt(&x1[0], dModel, b.Wv, b.bv, dModelKV, &vvec[0]);
		}

		// RoPE on Q and K
		if (useRope && ropeDim >= 2u)
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msRoPE);
			for (unsigned int hq = 0; hq < nHeads; ++hq)
				rope_apply_vec(&q[static_cast<size_t>(hq) * static_cast<size_t>(dHead)], dHead, ropeDim, session.posEncCache.ropeInvFreq, pos);
			for (unsigned int hk = 0; hk < nKVHeads; ++hk)
				rope_apply_vec(&kvec[static_cast<size_t>(hk) * static_cast<size_t>(dHead)], dHead, ropeDim, session.posEncCache.ropeInvFreq, pos);
		}

		// Store K/V into cache at [li, pos]
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msKVStore);
			const size_t perLayer = static_cast<size_t>(session.maxLen) * static_cast<size_t>(dModelKV);
			const size_t base = static_cast<size_t>(li) * perLayer + static_cast<size_t>(pos) * static_cast<size_t>(dModelKV);
			if (session.kvCacheDType != glades::NNetwork::TransformerLmSession::KV_CACHE_F32)
			{
				const int lowpDType =
				    (session.kvCacheDType == glades::NNetwork::TransformerLmSession::KV_CACHE_BF16)
				        ? glades::transformer_kernels::LOWP_BF16
				        : glades::transformer_kernels::LOWP_F16;
				for (unsigned int i = 0; i < dModelKV; ++i)
				{
					session.k16[base + i] = glades::transformer_kernels::float_to_lowp(kvec[i], lowpDType);
					session.v16[base + i] = glades::transformer_kernels::float_to_lowp(vvec[i], lowpDType);
				}
			}
			else
			{
				for (unsigned int i = 0; i < dModelKV; ++i)
				{
					session.k[base + i] = kvec[i];
					session.v[base + i] = vvec[i];
				}
			}
		}

		// Attention for this token against cached K/V up to pos (fused softmax + weighted sum).
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msAttention);
			const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dHead)));
			float* scoreBuf = scores.empty() ? NULL : &scores[0];
			const unsigned char* keyAllowed = session.keyValid.empty() ? NULL : &session.keyValid[0];
			const size_t perLayer = static_cast<size_t>(session.maxLen) * static_cast<size_t>(dModelKV);
			const float* kLayer = NULL;
			const float* vLayer = NULL;
			const uint16_t* kLayer16 = NULL;
			const uint16_t* vLayer16 = NULL;
			const int lowpDType =
			    (session.kvCacheDType == glades::NNetwork::TransformerLmSession::KV_CACHE_BF16)
			        ? glades::transformer_kernels::LOWP_BF16
			        : glades::transformer_kernels::LOWP_F16;
			if (session.kvCacheDType != glades::NNetwork::TransformerLmSession::KV_CACHE_F32)
			{
				kLayer16 =
				    (session.k16.size() >= static_cast<size_t>(tt.nLayers) * perLayer) ? (&session.k16[static_cast<size_t>(li) * perLayer]) : NULL;
				vLayer16 =
				    (session.v16.size() >= static_cast<size_t>(tt.nLayers) * perLayer) ? (&session.v16[static_cast<size_t>(li) * perLayer]) : NULL;
				if (!scoreBuf || !kLayer16 || !vLayer16)
					return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmSessionAppend: invalid attention scratch/cache pointers (lowp)");
			}
			else
			{
				kLayer =
				    (session.k.size() >= static_cast<size_t>(tt.nLayers) * perLayer) ? (&session.k[static_cast<size_t>(li) * perLayer]) : NULL;
				vLayer =
				    (session.v.size() >= static_cast<size_t>(tt.nLayers) * perLayer) ? (&session.v[static_cast<size_t>(li) * perLayer]) : NULL;
				if (!scoreBuf || !kLayer || !vLayer)
					return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmSessionAppend: invalid attention scratch/cache pointers");
			}
			for (unsigned int hq = 0; hq < nHeads; ++hq)
			{
				const unsigned int kvHead = (nKVHeads == nHeads) ? hq : (groupSize > 0u ? (hq / groupSize) : 0u);
				const float* qh = &q[static_cast<size_t>(hq) * static_cast<size_t>(dHead)];
				float* outHead = &attnConcat[static_cast<size_t>(hq) * static_cast<size_t>(dHead)];
				if (session.kvCacheDType != glades::NNetwork::TransformerLmSession::KV_CACHE_F32)
				{
					const glades::Tensor2DView<const uint16_t> kMat(kLayer16, static_cast<size_t>(pos) + 1u, static_cast<size_t>(dModelKV),
					                                               static_cast<size_t>(dModelKV), perLayer);
					const glades::Tensor2DView<const uint16_t> vMat(vLayer16, static_cast<size_t>(pos) + 1u, static_cast<size_t>(dModelKV),
					                                               static_cast<size_t>(dModelKV), perLayer);
					attention_head_fused_softmax_weighted_sum_lowp(outHead,
					                                              scoreBuf,
					                                              qh,
					                                              kMat,
					                                              vMat,
					                                              lowpDType,
					                                              dHead,
					                                              kvHead,
					                                              pos,
					                                              keyAllowed,
					                                              invSqrt);
				}
				else
				{
					const glades::Tensor2DView<const float> kMat(kLayer, static_cast<size_t>(pos) + 1u, static_cast<size_t>(dModelKV),
					                                            static_cast<size_t>(dModelKV), perLayer);
					const glades::Tensor2DView<const float> vMat(vLayer, static_cast<size_t>(pos) + 1u, static_cast<size_t>(dModelKV),
					                                            static_cast<size_t>(dModelKV), perLayer);
					attention_head_fused_softmax_weighted_sum(outHead,
					                                         scoreBuf,
					                                         qh,
					                                         kMat,
					                                         vMat,
					                                         dHead,
					                                         kvHead,
					                                         pos,
					                                         keyAllowed,
					                                         invSqrt);
				}
			}
		}

		// Wo + residual
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msWo);
			linear_into_opt(&attnConcat[0], dModel, b.Wo, b.bo, dModel, &attnOut[0]);
		}
		for (unsigned int i = 0; i < dModel; ++i)
			h[i] = h[i] + attnOut[i];

		// Norm2
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msNorm);
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
				rmsnorm_into(&h[0], dModel, b.ln2Gamma, b.ln2Beta, eps, &x2[0]);
			else
				layernorm_into(&h[0], dModel, b.ln2Gamma, b.ln2Beta, eps, &x2[0]);
		}

		// FFN
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msFFN);
			linear_into_opt(&x2[0], dModel, b.W1, b.b1, session.ff1Width, &ffPre[0]);
			if (session.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU))
			{
				for (unsigned int i = 0; i < dFF; ++i)
				{
					const float gate = glades::transformer_ops::silu(ffPre[i]);
					const float up = ffPre[static_cast<size_t>(dFF) + i];
					ffAct[i] = gate * up;
				}
			}
			else
			{
				const int act = static_cast<int>(session.ffnActivation);
				for (unsigned int i = 0; i < dFF; ++i)
				{
					const float x = ffPre[i];
					ffAct[i] = (act == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
					               ? glades::transformer_ops::gelu(x)
					               : glades::transformer_ops::relu(x);
				}
			}
			linear_into_opt(&ffAct[0], dFF, b.W2, b.b2, dModel, &ffOut[0]);
		}
		for (unsigned int i = 0; i < dModel; ++i)
			h[i] = h[i] + ffOut[i];

		for (unsigned int i = 0; i < dModel; ++i)
			if (!is_finite(h[i]))
			{
				if (metricsOn)
				{
					++session.perf.nonFiniteHiddenState;
					session.perf.lastNonFiniteLayer = li;
					session.perf.lastNonFinitePos = pos;
				}
				std::ostringstream oss;
				oss << "transformerLmSessionAppend: non-finite hidden state at layer " << li
				    << " position " << pos << " dim " << i;
				return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, oss.str());
			}
	}

	if (outLogits)
	{
		ScopedTimerMs t(this, breakdown, &session.perf.msLogits);

		// Apply final LayerNorm before logits (in-place on h, single position).
		if (!tt.lnFinalGamma.empty())
		{
			float invStd = 0.0f;
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
				glades::transformer_kernels::rmsnorm_forward_rows(&h[0], 1u, dModel, tt.lnFinalGamma, tt.lnFinalBeta, eps, &h[0], &invStd);
			else
			{
				float mean = 0.0f;
				glades::transformer_kernels::layernorm_forward_rows(&h[0], 1u, dModel, tt.lnFinalGamma, tt.lnFinalBeta, eps, &h[0], &mean, &invStd);
			}
		}

		// Avoid per-token reallocations if caller reuses the same vector across steps.
		if (outLogits->capacity() < static_cast<size_t>(vocab))
			outLogits->reserve(static_cast<size_t>(vocab));
		if (outLogits->size() != vocab)
			outLogits->resize(vocab);
		if (!outLogits->empty())
			tied_embedding_logits_into(&h[0], dModel, tt.tokE, tt.lmBias, vocab, &(*outLogits)[0]);
	}

	session.curLen += 1u;
	if (metricsOn && session.metricsLogPerKvAppend && session.logger)
		log_kv_append_event(session.logger,
		                   "transformer_kv_append",
		                   session.curLen,
		                   session.maxLen,
		                   1u,
		                   outLogits != NULL,
		                   &session.perf);
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::NNetwork::transformerLmBatchSessionReset(glades::NNetwork::TransformerLmBatchSession& session,
                                                                       unsigned int batchSize,
                                                                       unsigned int maxSeqLen) const
{
	if (netType != TYPE_TRANSFORMER_DECODER)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: requires TYPE_TRANSFORMER_DECODER");
	const glades::TransformerRunConfig& runtimeCfg = trainingConfig.transformer;
	const TransformerMetricsConfig metricsCfg = getTransformerMetricsConfig();
	if (!runtimeCfg.enableTokenEmbedding)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: requires transformer.enableTokenEmbedding");
	if (!tensorTransformer.initialized || !tensorTransformer.tokenModel)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionReset: transformer tensors not initialized for token LM (loadModel or run once)");
	if (batchSize == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: batchSize is 0");
	if (maxSeqLen == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: maxSeqLen is 0");

	const TensorTransformerState& tt = tensorTransformer;
	TransformerSessionCommonConfig commonCfg;
	{
		const NNetworkStatus st = build_transformer_session_common_config("transformerLmBatchSessionReset",
		                                                                 tt.dModel,
		                                                                 tt.dFF,
		                                                                 tt.nHeads,
		                                                                 tt.nKVHeads,
		                                                                 tt.nLayers,
		                                                                 tt.ffnKind,
		                                                                 tt.padTokenId,
		                                                                 runtimeCfg,
		                                                                 metricsCfg,
		                                                                 getLogger(),
		                                                                 commonCfg);
		if (!st.ok())
			return st;
	}

	// Allocation sizing safety: overflow checks + hard cap.
	{
		const size_t B = static_cast<size_t>(batchSize);
		const size_t L = static_cast<size_t>(tt.nLayers);
		const size_t S = static_cast<size_t>(maxSeqLen);
		const size_t K = static_cast<size_t>(commonCfg.dModelKV);
		size_t perSeqElems = 0u;
		size_t tmp = 0u;
		if (!checked_mul_size(L, S, tmp) || !checked_mul_size(tmp, K, perSeqElems))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: KV-cache size overflow (maxSeqLen too large)");
		size_t totalElems = 0u;
		if (!checked_mul_size(B, perSeqElems, totalElems))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: KV-cache size overflow (batchSize too large)");

		const bool kvLowp = (runtimeCfg.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_F16) ||
		                    (runtimeCfg.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_BF16);
		const unsigned long long elemBytes = kvLowp ? static_cast<unsigned long long>(sizeof(uint16_t))
		                                            : static_cast<unsigned long long>(sizeof(float));
		const unsigned long long kvBytes =
		    static_cast<unsigned long long>(totalElems) * elemBytes * 2ULL; // K+V

		// Shared scratch is O(dModel) and small; keyValid is B*maxSeqLen bytes.
		unsigned long long keyMaskBytes = 0ULL;
		{
			size_t keyMask = 0u;
			if (!checked_mul_size(static_cast<size_t>(batchSize), static_cast<size_t>(maxSeqLen), keyMask))
				return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: keyValid size overflow");
			keyMaskBytes = static_cast<unsigned long long>(keyMask);
		}
		const unsigned long long cap = kv_session_max_bytes(runtimeCfg);
		const unsigned long long wantBytes = kvBytes + keyMaskBytes;
		if (cap > 0ULL && wantBytes > cap)
		{
			std::ostringstream oss;
			oss << "transformerLmBatchSessionReset: session allocation exceeds cap (want "
			    << bytes_to_human(wantBytes) << ", cap " << bytes_to_human(cap)
			    << "). Reduce batchSize/maxSeqLen or set transformer.kvSessionMaxBytes / GLADES_TRANSFORMER_KV_SESSION_MAX_BYTES.";
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, oss.str());
		}
	}

	session.reset();
	session.initialized = true;
	session.batchSize = batchSize;
	session.maxLen = maxSeqLen;
	session.curLen.assign(static_cast<size_t>(batchSize), 0u);
	session.metricsEnabled = commonCfg.metricsEnabled;
	session.metricsBreakdownEnabled = commonCfg.metricsBreakdownEnabled;
	session.metricsLogPerKvAppend = commonCfg.metricsLogPerKvAppend;
	session.metricsGpuPerfEnabled = commonCfg.metricsGpuPerfEnabled;
	session.perf.reset();
	session.dModel = commonCfg.dModel;
	session.dFF = commonCfg.dFF;
	session.nHeads = commonCfg.nHeads;
	session.nKVHeads = commonCfg.nKVHeads;
	session.nLayers = commonCfg.nLayers;
	session.dHead = commonCfg.dHead;
	session.dModelKV = commonCfg.dModelKV;
	session.ffnKind = commonCfg.ffnKind;
	session.ff1Width = commonCfg.ff1Width;
	session.layerNormEps = commonCfg.layerNormEps;
	session.normType = commonCfg.normType;
	session.positionalEncoding = commonCfg.positionalEncoding;
	session.ropeDimOverride = commonCfg.ropeDimOverride;
	session.ropeTheta = commonCfg.ropeTheta;
	session.ffnActivation = commonCfg.ffnActivation;
	session.padTokenId = commonCfg.padTokenId;
	session.logger = commonCfg.logger;

	size_t perSeq = 0u;
	{
		size_t tmp = 0u;
		(void)checked_mul_size(static_cast<size_t>(tt.nLayers), static_cast<size_t>(maxSeqLen), tmp);
		(void)checked_mul_size(tmp, static_cast<size_t>(commonCfg.dModelKV), perSeq);
	}
	if (runtimeCfg.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_F16)
		session.kvCacheDType = glades::NNetwork::TransformerLmBatchSession::KV_CACHE_F16;
	else if (runtimeCfg.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_BF16)
		session.kvCacheDType = glades::NNetwork::TransformerLmBatchSession::KV_CACHE_BF16;
	else
		session.kvCacheDType = glades::NNetwork::TransformerLmBatchSession::KV_CACHE_F32;
	if (session.kvCacheDType != glades::NNetwork::TransformerLmBatchSession::KV_CACHE_F32)
	{
		session.k.clear();
		session.v.clear();
		session.k16.assign(static_cast<size_t>(batchSize) * perSeq, static_cast<uint16_t>(0u));
		session.v16.assign(static_cast<size_t>(batchSize) * perSeq, static_cast<uint16_t>(0u));
	}
	else
	{
		session.k16.clear();
		session.v16.clear();
		session.k.assign(static_cast<size_t>(batchSize) * perSeq, 0.0f);
		session.v.assign(static_cast<size_t>(batchSize) * perSeq, 0.0f);
	}
	session.keyValid.assign(static_cast<size_t>(batchSize) * static_cast<size_t>(maxSeqLen), 1u);
	session.h.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.x1.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.x2.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.q.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.kvec.assign(static_cast<size_t>(commonCfg.dModelKV), 0.0f);
	session.vvec.assign(static_cast<size_t>(commonCfg.dModelKV), 0.0f);
	session.attnConcat.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.attnOut.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.ffPre.assign(static_cast<size_t>(commonCfg.ff1Width), 0.0f);
	session.ffAct.assign(static_cast<size_t>(commonCfg.dFF), 0.0f);
	session.ffOut.assign(static_cast<size_t>(commonCfg.dModel), 0.0f);
	session.scores.assign(static_cast<size_t>(maxSeqLen), 0.0f);
	if (commonCfg.positionalEncoding == static_cast<unsigned int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
	{
		if (session.metricsEnabled)
		{
			const bool hit = (session.posEncCache.sinDModelCached == commonCfg.dModel && !session.posEncCache.sinInvDenomPair.empty());
			if (hit)
				++session.perf.sinCacheHits;
			else
				++session.perf.sinCacheMisses;
		}
		session.posEncCache.ensureSinusoidal(commonCfg.dModel);
	}

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::NNetwork::transformerLmBatchSessionAppendSelective(glades::NNetwork::TransformerLmBatchSession& session,
                                                                                const std::vector<unsigned int>& tokenIds,
                                                                                const std::vector<unsigned char>* tokenValid,
                                                                                const std::vector<unsigned char>& active,
                                                                                std::vector<float>* outLogitsFlat) const
{
	if (!session.initialized)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionAppendSelective: session not initialized (call transformerLmBatchSessionReset)");

	const unsigned int B = session.batchSize;
	if (tokenIds.size() != static_cast<size_t>(B))
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionAppendSelective: tokenIds.size != batchSize");
	if (tokenValid && tokenValid->size() != static_cast<size_t>(B))
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionAppendSelective: tokenValid.size != batchSize");
	if (active.size() != static_cast<size_t>(B))
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionAppendSelective: active.size != batchSize");

	const TensorTransformerState& tt = tensorTransformer;
	const unsigned int vocab = tt.vocabSize;
	if (vocab == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionAppendSelective: vocabSize is 0");

	// Output buffer [B, vocab]
	float* outPtr = NULL;
	if (outLogitsFlat)
	{
		outLogitsFlat->resize(static_cast<size_t>(B) * static_cast<size_t>(vocab));
		outPtr = outLogitsFlat->empty() ? NULL : &(*outLogitsFlat)[0];
	}

	// Sanity: ensure session matches current model dims.
	if (session.dModel != tt.dModel || session.dFF != tt.dFF || session.nLayers != tt.nLayers || session.nHeads != tt.nHeads)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionAppendSelective: session shape mismatch (reset required)");

	const unsigned int maxLen = session.maxLen;
	const unsigned int nLayers = session.nLayers;
	const unsigned int dModelKV = session.dModelKV;
	const size_t perSeq = static_cast<size_t>(nLayers) * static_cast<size_t>(maxLen) * static_cast<size_t>(dModelKV);
	const int padTokenId = session.padTokenId;

	const bool metricsOn = session.metricsEnabled;
	const bool breakdown = metricsOn && session.metricsBreakdownEnabled;
	ScopedTimerMs tTotal(this, metricsOn, &session.perf.msTotal);

	// Config knobs
	const float eps = session.layerNormEps;
	const int normType = static_cast<int>(session.normType);
	const int posEnc = static_cast<int>(session.positionalEncoding);
	const int ropeDimOverride = session.ropeDimOverride;
	const float ropeTheta = session.ropeTheta;
	const bool useRope = (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE));

	// RoPE cache for the session (shared across batch elements).
	unsigned int ropeDim = session.dHead;
	if (ropeDimOverride > 0)
	{
		const unsigned int rd = static_cast<unsigned int>(ropeDimOverride);
		ropeDim = (rd < ropeDim) ? rd : ropeDim;
	}
	if ((ropeDim % 2u) != 0u)
		ropeDim -= 1u;
	if (useRope && ropeDim >= 2u)
	{
		if (metricsOn)
		{
			const bool hit =
			    (session.posEncCache.ropeDimCached == ropeDim && session.posEncCache.ropeThetaCached == ropeTheta && !session.posEncCache.ropeInvFreq.empty());
			if (hit)
				++session.perf.ropeCacheHits;
			else
				++session.perf.ropeCacheMisses;
		}
		session.posEncCache.ensureRope(ropeDim, ropeTheta);
	}

	// Ensure sinusoidal cache if needed.
	if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
	{
		if (metricsOn)
		{
			const bool hit = (session.posEncCache.sinDModelCached == session.dModel && !session.posEncCache.sinInvDenomPair.empty());
			if (hit)
				++session.perf.sinCacheHits;
			else
				++session.perf.sinCacheMisses;
		}
		session.posEncCache.ensureSinusoidal(session.dModel);
	}

	// Reused scratch refs.
	std::vector<float, glades::AlignedAllocator<float, 64> >& h = session.h;
	std::vector<float, glades::AlignedAllocator<float, 64> >& x1 = session.x1;
	std::vector<float, glades::AlignedAllocator<float, 64> >& x2 = session.x2;
	std::vector<float, glades::AlignedAllocator<float, 64> >& q = session.q;
	std::vector<float, glades::AlignedAllocator<float, 64> >& kvec = session.kvec;
	std::vector<float, glades::AlignedAllocator<float, 64> >& vvec = session.vvec;
	std::vector<float, glades::AlignedAllocator<float, 64> >& attnConcat = session.attnConcat;
	std::vector<float, glades::AlignedAllocator<float, 64> >& attnOut = session.attnOut;
	std::vector<float, glades::AlignedAllocator<float, 64> >& ffPre = session.ffPre;
	std::vector<float, glades::AlignedAllocator<float, 64> >& ffAct = session.ffAct;
	std::vector<float, glades::AlignedAllocator<float, 64> >& ffOut = session.ffOut;
	std::vector<float, glades::AlignedAllocator<float, 64> >& scores = session.scores;

	for (unsigned int b = 0; b < B; ++b)
	{
		if (active[b] == 0u)
		{
			if (outPtr)
			{
				float* outRow = &outPtr[static_cast<size_t>(b) * static_cast<size_t>(vocab)];
				std::fill(outRow, outRow + vocab, 0.0f);
			}
			continue;
		}

		unsigned int tok = tokenIds[b];
		if (tok >= vocab)
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionAppendSelective: tokenId out of range");

		const unsigned char valid = tokenValid ? ((*tokenValid)[b] ? 1u : 0u) : 1u;
		if (valid == 0u && padTokenId >= 0)
			tok = static_cast<unsigned int>(padTokenId);

		unsigned char keyIsValid = valid;
		if (padTokenId >= 0 && static_cast<int>(tok) == padTokenId)
			keyIsValid = 0u;

		unsigned int& cur = session.curLen[b];
		if (cur >= maxLen)
			return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionAppendSelective: cache full (maxSeqLen exceeded)");
		const unsigned int pos = cur;

		if (metricsOn)
			++session.perf.kvAppends;

		float* kSeq = NULL;
		float* vSeq = NULL;
		uint16_t* kSeq16 = NULL;
		uint16_t* vSeq16 = NULL;
		if (session.kvCacheDType != glades::NNetwork::TransformerLmBatchSession::KV_CACHE_F32)
		{
			kSeq16 = session.k16.empty() ? NULL : &session.k16[static_cast<size_t>(b) * perSeq];
			vSeq16 = session.v16.empty() ? NULL : &session.v16[static_cast<size_t>(b) * perSeq];
		}
		else
		{
			kSeq = session.k.empty() ? NULL : &session.k[static_cast<size_t>(b) * perSeq];
			vSeq = session.v.empty() ? NULL : &session.v[static_cast<size_t>(b) * perSeq];
		}
		unsigned char* keyValidSeq = session.keyValid.empty() ? NULL : &session.keyValid[static_cast<size_t>(b) * static_cast<size_t>(maxLen)];
		float* outRow = outPtr ? &outPtr[static_cast<size_t>(b) * static_cast<size_t>(vocab)] : NULL;

		// keyValid for this position
		if (keyValidSeq)
			keyValidSeq[pos] = (keyIsValid ? 1u : 0u);
		if (valid == 0u && padTokenId < 0)
		{
			const size_t perLayer = static_cast<size_t>(maxLen) * static_cast<size_t>(session.dModelKV);
			const size_t basePos = static_cast<size_t>(pos) * static_cast<size_t>(session.dModelKV);
			for (unsigned int li = 0; li < session.nLayers; ++li)
			{
				const size_t base = static_cast<size_t>(li) * perLayer + basePos;
				if (session.kvCacheDType != glades::NNetwork::TransformerLmBatchSession::KV_CACHE_F32)
				{
					if (!kSeq16 || !vSeq16)
						return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionAppendSelective: invalid K/V cache pointers (lowp)");
					std::fill(kSeq16 + base, kSeq16 + base + session.dModelKV, static_cast<uint16_t>(0u));
					std::fill(vSeq16 + base, vSeq16 + base + session.dModelKV, static_cast<uint16_t>(0u));
				}
				else
				{
					if (!kSeq || !vSeq)
						return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionAppendSelective: invalid K/V cache pointers");
					std::fill(kSeq + base, kSeq + base + session.dModelKV, 0.0f);
					std::fill(vSeq + base, vSeq + base + session.dModelKV, 0.0f);
				}
			}
			if (outRow)
				std::fill(outRow, outRow + vocab, 0.0f);
			cur += 1u;
			continue;
		}

		// Embed
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msEmbed);
			const size_t eOff = static_cast<size_t>(tok) * static_cast<size_t>(session.dModel);
			for (unsigned int i = 0; i < session.dModel; ++i)
				h[i] = tt.tokE[eOff + i];
		}

		// Positional encoding (sinusoidal only; RoPE is applied to Q/K).
		if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msPosEnc);
			glades::transformer_kernels::add_sinusoidal_positional_encoding_inplace(&h[0], pos, session.dModel, session.posEncCache.sinInvDenomPair);
		}
		else if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_NONE) ||
		         posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE))
		{
			// no-op
		}
		else
		{
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionAppendSelective: unknown positionalEncoding");
		}

		for (unsigned int li = 0; li < session.nLayers; ++li)
		{
			const TensorTransformerState::Block& blk = tt.blocks[li];

			// Norm1
			{
				ScopedTimerMs t(this, breakdown, &session.perf.msNorm);
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
					rmsnorm_into(&h[0], session.dModel, blk.ln1Gamma, blk.ln1Beta, eps, &x1[0]);
				else
					layernorm_into(&h[0], session.dModel, blk.ln1Gamma, blk.ln1Beta, eps, &x1[0]);
			}

			// Projections
			{
				ScopedTimerMs t(this, breakdown, &session.perf.msProjQKV);
				linear_into_opt(&x1[0], session.dModel, blk.Wq, blk.bq, session.dModel, &q[0]);
				linear_into_opt(&x1[0], session.dModel, blk.Wk, blk.bk, session.dModelKV, &kvec[0]);
				linear_into_opt(&x1[0], session.dModel, blk.Wv, blk.bv, session.dModelKV, &vvec[0]);
			}

			// RoPE on Q and K
			if (useRope && ropeDim >= 2u)
			{
				ScopedTimerMs t(this, breakdown, &session.perf.msRoPE);
				for (unsigned int hq_i = 0; hq_i < session.nHeads; ++hq_i)
					rope_apply_vec(&q[static_cast<size_t>(hq_i) * static_cast<size_t>(session.dHead)], session.dHead, ropeDim, session.posEncCache.ropeInvFreq, pos);
				for (unsigned int hk_i = 0; hk_i < session.nKVHeads; ++hk_i)
					rope_apply_vec(&kvec[static_cast<size_t>(hk_i) * static_cast<size_t>(session.dHead)], session.dHead, ropeDim, session.posEncCache.ropeInvFreq, pos);
			}

			// Store K/V
			{
				ScopedTimerMs t(this, breakdown, &session.perf.msKVStore);
				const size_t perLayer = static_cast<size_t>(maxLen) * static_cast<size_t>(session.dModelKV);
				const size_t base = static_cast<size_t>(li) * perLayer + static_cast<size_t>(pos) * static_cast<size_t>(session.dModelKV);
				if (session.kvCacheDType != glades::NNetwork::TransformerLmBatchSession::KV_CACHE_F32)
				{
					const int lowpDType =
					    (session.kvCacheDType == glades::NNetwork::TransformerLmBatchSession::KV_CACHE_BF16)
					        ? glades::transformer_kernels::LOWP_BF16
					        : glades::transformer_kernels::LOWP_F16;
					for (unsigned int i = 0; i < session.dModelKV; ++i)
					{
						kSeq16[base + i] = glades::transformer_kernels::float_to_lowp(kvec[i], lowpDType);
						vSeq16[base + i] = glades::transformer_kernels::float_to_lowp(vvec[i], lowpDType);
					}
				}
				else
				{
					for (unsigned int i = 0; i < session.dModelKV; ++i)
					{
						kSeq[base + i] = kvec[i];
						vSeq[base + i] = vvec[i];
					}
				}
			}

			// Attention
			{
				ScopedTimerMs t(this, breakdown, &session.perf.msAttention);
				const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(session.dHead)));
				float* scoreBuf = scores.empty() ? NULL : &scores[0];
				const unsigned int groupSize = (session.nKVHeads > 0u ? (session.nHeads / session.nKVHeads) : 0u);
				if (!scoreBuf)
					return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionAppendSelective: invalid attention scratch pointer");
				for (unsigned int hq_i = 0; hq_i < session.nHeads; ++hq_i)
				{
					const unsigned int kvHead =
					    (session.nKVHeads == session.nHeads) ? hq_i : (groupSize > 0u ? (hq_i / groupSize) : 0u);
					const float* qh = &q[static_cast<size_t>(hq_i) * static_cast<size_t>(session.dHead)];
					float* outHead = &attnConcat[static_cast<size_t>(hq_i) * static_cast<size_t>(session.dHead)];
					const size_t perLayer = static_cast<size_t>(maxLen) * static_cast<size_t>(session.dModelKV);
					const float* kLayer = NULL;
					const float* vLayer = NULL;
					const uint16_t* kLayer16 = NULL;
					const uint16_t* vLayer16 = NULL;
					const int lowpDType =
					    (session.kvCacheDType == glades::NNetwork::TransformerLmBatchSession::KV_CACHE_BF16)
					        ? glades::transformer_kernels::LOWP_BF16
					        : glades::transformer_kernels::LOWP_F16;
					if (session.kvCacheDType != glades::NNetwork::TransformerLmBatchSession::KV_CACHE_F32)
					{
						kLayer16 = kSeq16 ? (kSeq16 + static_cast<size_t>(li) * perLayer) : NULL;
						vLayer16 = vSeq16 ? (vSeq16 + static_cast<size_t>(li) * perLayer) : NULL;
						if (!kLayer16 || !vLayer16)
							return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionAppendSelective: invalid K/V cache pointers (lowp)");
						const glades::Tensor2DView<const uint16_t> kMat(kLayer16, static_cast<size_t>(pos) + 1u, static_cast<size_t>(session.dModelKV),
						                                               static_cast<size_t>(session.dModelKV), perLayer);
						const glades::Tensor2DView<const uint16_t> vMat(vLayer16, static_cast<size_t>(pos) + 1u, static_cast<size_t>(session.dModelKV),
						                                               static_cast<size_t>(session.dModelKV), perLayer);
						attention_head_fused_softmax_weighted_sum_lowp(outHead,
						                                              scoreBuf,
						                                              qh,
						                                              kMat,
						                                              vMat,
						                                              lowpDType,
						                                              session.dHead,
						                                              kvHead,
						                                              pos,
						                                              keyValidSeq,
						                                              invSqrt);
					}
					else
					{
						kLayer = kSeq ? (kSeq + static_cast<size_t>(li) * perLayer) : NULL;
						vLayer = vSeq ? (vSeq + static_cast<size_t>(li) * perLayer) : NULL;
						if (!kLayer || !vLayer)
							return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionAppendSelective: invalid K/V cache pointers");
						const glades::Tensor2DView<const float> kMat(kLayer, static_cast<size_t>(pos) + 1u, static_cast<size_t>(session.dModelKV),
						                                            static_cast<size_t>(session.dModelKV), perLayer);
						const glades::Tensor2DView<const float> vMat(vLayer, static_cast<size_t>(pos) + 1u, static_cast<size_t>(session.dModelKV),
						                                            static_cast<size_t>(session.dModelKV), perLayer);
						attention_head_fused_softmax_weighted_sum(outHead,
						                                         scoreBuf,
						                                         qh,
						                                         kMat,
						                                         vMat,
						                                         session.dHead,
						                                         kvHead,
						                                         pos,
						                                         keyValidSeq,
						                                         invSqrt);
					}
				}
			}

			{
				ScopedTimerMs t(this, breakdown, &session.perf.msWo);
				linear_into_opt(&attnConcat[0], session.dModel, blk.Wo, blk.bo, session.dModel, &attnOut[0]);
			}
			for (unsigned int i = 0; i < session.dModel; ++i)
				h[i] = h[i] + attnOut[i];

			// Norm2
			{
				ScopedTimerMs t(this, breakdown, &session.perf.msNorm);
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
					rmsnorm_into(&h[0], session.dModel, blk.ln2Gamma, blk.ln2Beta, eps, &x2[0]);
				else
					layernorm_into(&h[0], session.dModel, blk.ln2Gamma, blk.ln2Beta, eps, &x2[0]);
			}

			// FFN
			{
				ScopedTimerMs t(this, breakdown, &session.perf.msFFN);
				linear_into_opt(&x2[0], session.dModel, blk.W1, blk.b1, session.ff1Width, &ffPre[0]);
				if (session.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU))
				{
					for (unsigned int i = 0; i < session.dFF; ++i)
					{
						const float gate = glades::transformer_ops::silu(ffPre[i]);
						const float up = ffPre[static_cast<size_t>(session.dFF) + i];
						ffAct[i] = gate * up;
					}
				}
				else
				{
					const int act = static_cast<int>(session.ffnActivation);
					for (unsigned int i = 0; i < session.dFF; ++i)
					{
						const float x = ffPre[i];
						ffAct[i] = (act == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
						               ? glades::transformer_ops::gelu(x)
						               : glades::transformer_ops::relu(x);
					}
				}
				linear_into_opt(&ffAct[0], session.dFF, blk.W2, blk.b2, session.dModel, &ffOut[0]);
			}
			for (unsigned int i = 0; i < session.dModel; ++i)
				h[i] = h[i] + ffOut[i];

			for (unsigned int i = 0; i < session.dModel; ++i)
				if (!is_finite(h[i]))
				{
					if (metricsOn)
					{
						++session.perf.nonFiniteHiddenState;
						session.perf.lastNonFiniteLayer = li;
						session.perf.lastNonFinitePos = pos;
					}
					std::ostringstream oss;
					oss << "transformerLmBatchSessionAppendSelective: non-finite hidden state at batch " << b
					    << " layer " << li << " position " << pos << " dim " << i;
					return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, oss.str());
				}
		}

		// Logits
		if (outRow)
		{
			ScopedTimerMs t(this, breakdown, &session.perf.msLogits);

			// Apply final LayerNorm before logits (in-place on h, single position).
			if (!tt.lnFinalGamma.empty())
			{
				float invStd = 0.0f;
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
					glades::transformer_kernels::rmsnorm_forward_rows(&h[0], 1u, session.dModel, tt.lnFinalGamma, tt.lnFinalBeta, eps, &h[0], &invStd);
				else
				{
					float mean = 0.0f;
					glades::transformer_kernels::layernorm_forward_rows(&h[0], 1u, session.dModel, tt.lnFinalGamma, tt.lnFinalBeta, eps, &h[0], &mean, &invStd);
				}
			}

			tied_embedding_logits_into(&h[0], session.dModel, tt.tokE, tt.lmBias, vocab, outRow);
		}

		cur += 1u;
	}

	if (metricsOn && session.metricsLogPerKvAppend && session.logger)
	{
		unsigned int activeCount = 0u;
		unsigned int maxCurLen = 0u;
		for (size_t i = 0u; i < active.size(); ++i)
			if (active[i] != 0u)
				++activeCount;
		for (size_t i = 0u; i < session.curLen.size(); ++i)
			if (session.curLen[i] > maxCurLen)
				maxCurLen = session.curLen[i];
		log_kv_append_event(session.logger,
		                   "transformer_kv_batch_append",
		                   maxCurLen,
		                   session.maxLen,
		                   activeCount,
		                   outLogitsFlat != NULL,
		                   &session.perf);
	}
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::NNetwork::transformerLmForwardLastLogits(const std::vector<unsigned int>& tokenIds,
                                                                        std::vector<float>& outLogits) const
{
	glades::NNetwork::RunLockGuard runGuard(*const_cast<glades::NNetwork*>(this));
	if (!runGuard.ok())
		return NNetworkStatus(NNetworkStatus::INVALID_STATE,
		                     "transformerLmForwardLastLogits: NNetwork is already running (training/eval/inference are not re-entrant)");

	if (netType != TYPE_TRANSFORMER_DECODER)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmForwardLastLogits: requires TYPE_TRANSFORMER_DECODER");
	if (!trainingConfig.transformer.enableTokenEmbedding)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmForwardLastLogits: requires transformer.enableTokenEmbedding");
	if (!tensorTransformer.initialized || !tensorTransformer.tokenModel)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmForwardLastLogits: transformer tensors not initialized for token LM");
	if (tokenIds.empty())
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmForwardLastLogits: tokenIds empty");

	const TensorTransformerState& tt = tensorTransformer;
	const unsigned int vocab = tt.vocabSize;
	const unsigned int dModel = tt.dModel;
	const unsigned int nHeads = tt.nHeads;
	const unsigned int nKVHeads = (tt.nKVHeads > 0u ? tt.nKVHeads : nHeads);
	const unsigned int dHead = (nHeads > 0u ? (dModel / nHeads) : 0u);
	const unsigned int dModelKV = nKVHeads * dHead;
	if (vocab == 0u || dModel == 0u || nHeads == 0u || dHead == 0u || dModelKV == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmForwardLastLogits: invalid transformer dimensions");
	if ((dModel % nHeads) != 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmForwardLastLogits: dModel is not divisible by nHeads");
	if (nKVHeads > 0u && (nHeads % nKVHeads) != 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmForwardLastLogits: nHeads is not divisible by nKVHeads");
	if (tt.blocks.size() != static_cast<size_t>(tt.nLayers))
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmForwardLastLogits: invalid block count");

	const size_t T = tokenIds.size();
	for (size_t t = 0; t < T; ++t)
	{
		if (tokenIds[t] >= vocab)
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmForwardLastLogits: tokenId out of range");
	}

	// Config knobs
	const float eps = (trainingConfig.transformer.layerNormEps > 0.0f ? trainingConfig.transformer.layerNormEps : 1e-5f);
	const int normType = static_cast<int>(trainingConfig.transformer.normType);
	const int posEnc = static_cast<int>(trainingConfig.transformer.positionalEncoding);
	const int ropeDimOverride = trainingConfig.transformer.ropeDimOverride;
	const float ropeTheta = (trainingConfig.transformer.ropeTheta > 0.0f ? trainingConfig.transformer.ropeTheta : 10000.0f);
	const bool useRope = (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE));

	// Optional padding mask: when padTokenId>=0, positions whose tokenId==padTokenId
	// must not contribute keys/values to attention.
	const int padTokenId = tt.padTokenId;
	std::vector<unsigned char> keyAllowed;
	if (padTokenId >= 0)
	{
		keyAllowed.assign(T, 1u);
		for (size_t t = 0; t < T; ++t)
			if (static_cast<int>(tokenIds[t]) == padTokenId)
				keyAllowed[t] = 0u;
	}

	unsigned int ropeDim = dHead;
	if (ropeDimOverride > 0)
	{
		const unsigned int rd = static_cast<unsigned int>(ropeDimOverride);
		ropeDim = (rd < ropeDim) ? rd : ropeDim;
	}
	if ((ropeDim % 2u) != 0u)
		ropeDim -= 1u;
	const std::vector<double>* ropeInvFreq = NULL;
	if (useRope && ropeDim >= 2u)
	{
		// Avoid recomputing pow()-derived invFreq each call.
		transformerPosEncCache.ensureRope(ropeDim, ropeTheta);
		ropeInvFreq = &transformerPosEncCache.ropeInvFreq;
	}

	// h[t, :]
	std::vector<float> h(T * static_cast<size_t>(dModel), 0.0f);
	for (size_t t = 0; t < T; ++t)
	{
		const unsigned int tok = tokenIds[t];
		const size_t eOff = static_cast<size_t>(tok) * static_cast<size_t>(dModel);
		float* ht = &h[t * static_cast<size_t>(dModel)];
		for (unsigned int i = 0; i < dModel; ++i)
			ht[i] = tt.tokE[eOff + i];
	}

	// Positional encoding must match training.
	if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
	{
		// Avoid recomputing pow()-derived denominators per position.
		transformerPosEncCache.ensureSinusoidal(dModel);
		glades::transformer_kernels::add_sinusoidal_positional_encoding_seq_inplace(&h[0], static_cast<unsigned int>(T), dModel,
		                                                                           transformerPosEncCache.sinInvDenomPair);
	}
	else if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_NONE) ||
	         posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE))
	{
		// no-op here
	}
	else
	{
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmForwardLastLogits: unknown positionalEncoding");
	}

	// Per-layer buffers reused across layers
	std::vector<float> x1(T * static_cast<size_t>(dModel), 0.0f);
	std::vector<float> q(T * static_cast<size_t>(dModel), 0.0f);
	std::vector<float> k(T * static_cast<size_t>(dModelKV), 0.0f);
	std::vector<float> v(T * static_cast<size_t>(dModelKV), 0.0f);

	// Reuse one scores buffer for all heads/positions. We only use a prefix [0..t].
	std::vector<float> scores(T, 0.0f);
	std::vector<float> attnConcat(dModel, 0.0f);
	std::vector<float> attnOut(dModel, 0.0f);
	std::vector<float> x2(dModel, 0.0f);
	std::vector<float> ffPre;
	std::vector<float> ffAct;
	std::vector<float> ffOut(dModel, 0.0f);

	const unsigned int groupSize = (nKVHeads > 0u ? (nHeads / nKVHeads) : 0u);
	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dHead)));

	for (unsigned int li = 0; li < tt.nLayers; ++li)
	{
		const TensorTransformerState::Block& b = tt.blocks[li];

		// Norm1: x1[t] = norm(h[t])
		for (size_t t = 0; t < T; ++t)
		{
			const float* ht = &h[t * static_cast<size_t>(dModel)];
			float* x1t = &x1[t * static_cast<size_t>(dModel)];
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
				rmsnorm_into(ht, dModel, b.ln1Gamma, b.ln1Beta, eps, x1t);
			else
				layernorm_into(ht, dModel, b.ln1Gamma, b.ln1Beta, eps, x1t);
		}

		// Projections (Q,K,V) from x1
		for (size_t t = 0; t < T; ++t)
		{
			const float* x1t = &x1[t * static_cast<size_t>(dModel)];
			float* qt = &q[t * static_cast<size_t>(dModel)];
			float* kt = &k[t * static_cast<size_t>(dModelKV)];
			float* vt = &v[t * static_cast<size_t>(dModelKV)];

			linear_into_opt(x1t, dModel, b.Wq, b.bq, dModel, qt);
			linear_into_opt(x1t, dModel, b.Wk, b.bk, dModelKV, kt);
			linear_into_opt(x1t, dModel, b.Wv, b.bv, dModelKV, vt);

			// RoPE on Q and K (per position)
			if (useRope && ropeInvFreq)
			{
				const unsigned int pos = static_cast<unsigned int>(t);
				for (unsigned int hq = 0; hq < nHeads; ++hq)
					rope_apply_vec(&qt[static_cast<size_t>(hq) * static_cast<size_t>(dHead)], dHead, ropeDim, *ropeInvFreq, pos);
				for (unsigned int hk = 0; hk < nKVHeads; ++hk)
					rope_apply_vec(&kt[static_cast<size_t>(hk) * static_cast<size_t>(dHead)], dHead, ropeDim, *ropeInvFreq, pos);
			}
		}

		// Attention + Wo residual (per position)
		for (size_t t = 0; t < T; ++t)
		{
			std::fill(attnConcat.begin(), attnConcat.end(), 0.0f);

			for (unsigned int hq = 0; hq < nHeads; ++hq)
			{
				const unsigned int kvHead = (nKVHeads == nHeads) ? hq : (groupSize > 0u ? (hq / groupSize) : 0u);
				size_t headOff = 0u;
				if (!checked_mul_size(static_cast<size_t>(hq), static_cast<size_t>(dHead), headOff))
					return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "transformerLmForwardLastLogits: head offset overflow");
				const float* qh = &q[t * static_cast<size_t>(dModel) + headOff];

				// causal scores[u] for u=0..t
				float* scoreBuf = scores.empty() ? NULL : &scores[0];
				if (!scoreBuf)
					return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmForwardLastLogits: invalid attention scratch pointer");
				float* outHead = &attnConcat[headOff];
				const glades::Tensor2DView<const float> kMat(&k[0], t + 1u, static_cast<size_t>(dModelKV), static_cast<size_t>(dModelKV), k.size());
				const glades::Tensor2DView<const float> vMat(&v[0], t + 1u, static_cast<size_t>(dModelKV), static_cast<size_t>(dModelKV), v.size());
				attention_head_fused_softmax_weighted_sum(outHead,
				                                         scoreBuf,
				                                         qh,
				                                         kMat,
				                                         vMat,
				                                         dHead,
				                                         kvHead,
				                                         static_cast<unsigned int>(t),
				                                         keyAllowed.empty() ? NULL : &keyAllowed[0],
				                                         invSqrt);
			}

			linear_into_opt(&attnConcat[0], dModel, b.Wo, b.bo, dModel, &attnOut[0]);
			float* ht = &h[t * static_cast<size_t>(dModel)];
			for (unsigned int i = 0; i < dModel; ++i)
				ht[i] = ht[i] + attnOut[i];
		}

		// Norm2 + FFN residual (per position)
		const unsigned int dFF = tt.dFF;
		const unsigned int ff1Width =
		    (tt.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFF) : dFF;
		if (ffPre.size() != static_cast<size_t>(ff1Width))
			ffPre.resize(static_cast<size_t>(ff1Width));
		if (ffAct.size() != static_cast<size_t>(dFF))
			ffAct.resize(static_cast<size_t>(dFF));

		const int act = static_cast<int>(trainingConfig.transformer.ffnActivation);
		for (size_t t = 0; t < T; ++t)
		{
			float* ht = &h[t * static_cast<size_t>(dModel)];
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
				rmsnorm_into(ht, dModel, b.ln2Gamma, b.ln2Beta, eps, &x2[0]);
			else
				layernorm_into(ht, dModel, b.ln2Gamma, b.ln2Beta, eps, &x2[0]);

			linear_into_opt(&x2[0], dModel, b.W1, b.b1, ff1Width, &ffPre[0]);

			if (tt.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU))
			{
				for (unsigned int i = 0; i < dFF; ++i)
				{
					const float gate = glades::transformer_ops::silu(ffPre[i]);
					const float up = ffPre[static_cast<size_t>(dFF) + i];
					ffAct[i] = gate * up;
				}
			}
			else
			{
				for (unsigned int i = 0; i < dFF; ++i)
				{
					const float x = ffPre[i];
					ffAct[i] = (act == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
					               ? glades::transformer_ops::gelu(x)
					               : glades::transformer_ops::relu(x);
				}
			}

			linear_into_opt(&ffAct[0], dFF, b.W2, b.b2, dModel, &ffOut[0]);
			for (unsigned int i = 0; i < dModel; ++i)
			{
				ht[i] = ht[i] + ffOut[i];
				if (!is_finite(ht[i]))
				{
					std::ostringstream oss;
					oss << "transformerLmForwardLastLogits: non-finite hidden state at layer " << li << " position " << t << " dim " << i;
					return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, oss.str());
				}
			}
		}
	}

	// Apply final LayerNorm to last position before logits.
	std::vector<float> hLastLN(dModel, 0.0f);
	{
		const float* hLastRaw = &h[(T - 1u) * static_cast<size_t>(dModel)];
		if (!tt.lnFinalGamma.empty())
		{
			float invStd = 0.0f;
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
				glades::transformer_kernels::rmsnorm_forward_rows(hLastRaw, 1u, dModel, tt.lnFinalGamma, tt.lnFinalBeta, eps, &hLastLN[0], &invStd);
			else
			{
				float mean = 0.0f;
				glades::transformer_kernels::layernorm_forward_rows(hLastRaw, 1u, dModel, tt.lnFinalGamma, tt.lnFinalBeta, eps, &hLastLN[0], &mean, &invStd);
			}
		}
		else
		{
			std::copy(hLastRaw, hLastRaw + dModel, hLastLN.begin());
		}
	}

	// Logits for last position: h_last * E^T + bias
	outLogits.assign(vocab, 0.0f);
	if (!outLogits.empty())
		tied_embedding_logits_into(&hLastLN[0], dModel, tt.tokE, tt.lmBias, vocab, &outLogits[0]);

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}
