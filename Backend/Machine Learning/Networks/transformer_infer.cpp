// Transformer decoder-only KV-cache inference for token LM mode.
//
// This is a forward-only incremental decode path:
// - append one token at a time
// - cache K/V for each layer
// - compute logits for the appended token position
//
#include "network.h"
#include "../GMath/gmath.h"
#include "transformer_kernels.h"
#include "tensor_view.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <vector>

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
static inline bool checked_mul_size(size_t a, size_t b, size_t& out)
{
	if (a == 0u || b == 0u)
	{
		out = 0u;
		return true;
	}
	if (a > (std::numeric_limits<size_t>::max() / b))
		return false;
	out = a * b;
	return true;
}

static inline bool checked_add_size(size_t a, size_t b, size_t& out)
{
	if (a > (std::numeric_limits<size_t>::max() - b))
		return false;
	out = a + b;
	return true;
}

static inline bool parse_u64_env(const char* name, unsigned long long& out)
{
	out = 0ULL;
	if (!name || !name[0])
		return false;
	const char* v = ::getenv(name);
	if (!v || !v[0])
		return false;
	errno = 0;
	char* end = NULL;
	const unsigned long long x = ::strtoull(v, &end, 10);
	if (errno != 0 || end == v || (end && *end != '\0'))
		return false;
	out = x;
	return true;
}

static inline unsigned long long kv_session_max_bytes()
{
	// Default: 2 GiB.
	// Rationale: large enough for typical CPU demo/serving, small enough to prevent
	// accidental multi-tens-of-GB allocations from a bad maxSeqLen.
	static const unsigned long long kDefault = 2ULL * 1024ULL * 1024ULL * 1024ULL;
	unsigned long long v = 0ULL;
	if (parse_u64_env("GLADES_TRANSFORMER_KV_SESSION_MAX_BYTES", v) && v > 0ULL)
		return v;
	return kDefault;
}

static std::string bytes_to_human(unsigned long long bytes)
{
	std::ostringstream oss;
	const double b = static_cast<double>(bytes);
	const double mib = b / (1024.0 * 1024.0);
	const double gib = mib / 1024.0;
	oss.setf(std::ios::fixed);
	oss.precision(2);
	if (gib >= 1.0)
		oss << gib << " GiB";
	else
		oss << mib << " MiB";
	return oss.str();
}

struct ScopedTimerMs
{
	const glades::NNetwork* net;
	double* acc;
	int64_t t0ms;
	explicit ScopedTimerMs(const glades::NNetwork* n, bool enabled, double* outAcc)
	    : net(n), acc((enabled && outAcc && n) ? outAcc : NULL), t0ms(0)
	{
		if (acc)
			t0ms = net->getCurrentTimeMilliseconds();
	}
	~ScopedTimerMs()
	{
		if (!acc)
			return;
		const int64_t t1ms = net->getCurrentTimeMilliseconds();
		*acc += static_cast<double>(t1ms - t0ms);
	}
};

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
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		if (static_cast<size_t>(u) >= kLayer.rows)
			return;
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
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		const double w = exp(static_cast<double>(scoreBuf[u] - maxScore));
		sum += w;
		if (static_cast<size_t>(u) >= vLayer.rows)
			return;
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
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		if (static_cast<size_t>(u) >= kLayer.rows)
			return;
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
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		const double w = exp(static_cast<double>(scoreBuf[u] - maxScore));
		sum += w;
		if (static_cast<size_t>(u) >= vLayer.rows)
			return;
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
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		if (static_cast<size_t>(u) >= kLayer.rows)
			return;
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
		if (keyAllowed && keyAllowed[u] == 0u)
			continue;
		const double w = exp(static_cast<double>(scoreBuf[u] - maxScore));
		sum += w;
		if (static_cast<size_t>(u) >= vLayer.rows)
			return;
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

static inline void ensure_sinusoidal_cache(unsigned int dModel,
                                           unsigned int& cachedDModel,
                                           std::vector<double>& invDenomPair)
{
	if (dModel == 0u)
	{
		cachedDModel = 0u;
		invDenomPair.clear();
		return;
	}
	if (cachedDModel == dModel && !invDenomPair.empty())
		return;

	cachedDModel = dModel;
	const unsigned int nPairs = (dModel + 1u) / 2u;
	invDenomPair.assign(static_cast<size_t>(nPairs), 0.0);
	for (unsigned int ii = 0; ii < nPairs; ++ii)
	{
		const double exponent = (2.0 * static_cast<double>(ii)) / static_cast<double>(dModel);
		invDenomPair[static_cast<size_t>(ii)] = pow(10000.0, -exponent);
	}
}

static inline void ensure_rope_cache(unsigned int ropeDimEven,
                                     float ropeTheta,
                                     unsigned int& cachedRopeDim,
                                     float& cachedRopeTheta,
                                     std::vector<double>& invFreq)
{
	if (ropeDimEven < 2u || ropeTheta <= 0.0f)
	{
		cachedRopeDim = 0u;
		cachedRopeTheta = 0.0f;
		invFreq.clear();
		return;
	}
	if ((ropeDimEven % 2u) != 0u)
		ropeDimEven -= 1u;
	if (cachedRopeDim == ropeDimEven && cachedRopeTheta == ropeTheta && !invFreq.empty())
		return;

	cachedRopeDim = ropeDimEven;
	cachedRopeTheta = ropeTheta;
	invFreq.assign(static_cast<size_t>(ropeDimEven / 2u), 0.0);
	for (unsigned int ii = 0; ii < (ropeDimEven / 2u); ++ii)
	{
		const double frac = (2.0 * static_cast<double>(ii)) / static_cast<double>(ropeDimEven);
		invFreq[static_cast<size_t>(ii)] = pow(static_cast<double>(ropeTheta), -frac);
	}
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
	if (!trainingConfig.transformer.enableTokenEmbedding)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: requires transformer.enableTokenEmbedding");
	if (!tensorTransformer.initialized || !tensorTransformer.tokenModel)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmSessionReset: transformer tensors not initialized for token LM (loadModel or run once)");
	if (maxSeqLen == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: maxSeqLen is 0");

	// Fail fast on unsupported positional encodings. KV-cache inference must match training.
	{
		const int posEnc = static_cast<int>(trainingConfig.transformer.positionalEncoding);
		if (posEnc != static_cast<int>(glades::TransformerRunConfig::POSENC_NONE) &&
		    posEnc != static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL) &&
		    posEnc != static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE))
		{
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: unknown positionalEncoding");
		}
	}

	const TensorTransformerState& tt = tensorTransformer;
	const unsigned int dModel = tt.dModel;
	const unsigned int nHeads = tt.nHeads;
	const unsigned int nKVHeads = (tt.nKVHeads > 0u ? tt.nKVHeads : nHeads);
	const unsigned int dHead = (nHeads > 0u ? (dModel / nHeads) : 0u);
	const unsigned int dModelKV = nKVHeads * dHead;
	if (dHead == 0u || dModelKV == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmSessionReset: invalid head dimensions");
	const unsigned int ff1WidthLocal =
	    (tt.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * tt.dFF) : tt.dFF;

	// Allocation sizing safety: overflow checks + hard cap.
	{
		const size_t L = static_cast<size_t>(tt.nLayers);
		const size_t S = static_cast<size_t>(maxSeqLen);
		const size_t K = static_cast<size_t>(dModelKV);
		size_t perLayerElems = 0u;
		size_t totalElems = 0u;
		if (!checked_mul_size(S, K, perLayerElems) || !checked_mul_size(L, perLayerElems, totalElems))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: KV-cache size overflow (maxSeqLen too large)");

		const bool kvLowp = (trainingConfig.transformer.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_F16) ||
		                    (trainingConfig.transformer.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_BF16);
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
		if (!checked_mul_size(static_cast<size_t>(dModel), 7u, scratchFloats))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: scratch size overflow");
		size_t tmp = 0u;
		if (!checked_add_size(scratchFloats, static_cast<size_t>(dModelKV) * 2u, scratchFloats))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: scratch size overflow");
		if (!checked_add_size(scratchFloats, static_cast<size_t>(ff1WidthLocal), scratchFloats))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: scratch size overflow");
		if (!checked_add_size(scratchFloats, static_cast<size_t>(tt.dFF), scratchFloats))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: scratch size overflow");
		if (!checked_add_size(scratchFloats, static_cast<size_t>(maxSeqLen), scratchFloats))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmSessionReset: scratch size overflow");

		const unsigned long long scratchBytes =
		    static_cast<unsigned long long>(scratchFloats) * static_cast<unsigned long long>(sizeof(float)) +
		    // keyValid mask: maxSeqLen bytes
		    static_cast<unsigned long long>(maxSeqLen);

		const unsigned long long wantBytes = kvBytes + scratchBytes;
		const unsigned long long cap = kv_session_max_bytes();
		if (cap > 0ULL && wantBytes > cap)
		{
			std::ostringstream oss;
			oss << "transformerLmSessionReset: session allocation exceeds cap (want "
			    << bytes_to_human(wantBytes) << ", cap " << bytes_to_human(cap)
			    << "). Reduce maxSeqLen or set GLADES_TRANSFORMER_KV_SESSION_MAX_BYTES.";
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, oss.str());
		}
	}

	session.reset();
	session.initialized = true;
	session.metricsEnabled = transformerMetricsCfg.enable;
	session.perf.reset();
	session.maxLen = maxSeqLen;
	session.curLen = 0u;
	session.dModel = dModel;
	session.dFF = tt.dFF;
	session.nHeads = nHeads;
	session.nKVHeads = nKVHeads;
	session.nLayers = tt.nLayers;
	session.dHead = dHead;
	session.dModelKV = dModelKV;
	session.ffnKind = tt.ffnKind;
	session.ff1Width =
	    (tt.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * tt.dFF) : tt.dFF;

	size_t perLayer = 0u;
	(void)checked_mul_size(static_cast<size_t>(maxSeqLen), static_cast<size_t>(dModelKV), perLayer);
	// KV-cache dtype for this session.
	if (trainingConfig.transformer.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_F16)
		session.kvCacheDType = glades::NNetwork::TransformerLmSession::KV_CACHE_F16;
	else if (trainingConfig.transformer.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_BF16)
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

	// Pre-size scratch buffers to guarantee allocation-free Append().
	session.h.assign(static_cast<size_t>(dModel), 0.0f);
	session.x1.assign(static_cast<size_t>(dModel), 0.0f);
	session.x2.assign(static_cast<size_t>(dModel), 0.0f);
	session.q.assign(static_cast<size_t>(dModel), 0.0f);
	session.kvec.assign(static_cast<size_t>(dModelKV), 0.0f);
	session.vvec.assign(static_cast<size_t>(dModelKV), 0.0f);
	session.attnConcat.assign(static_cast<size_t>(dModel), 0.0f);
	session.attnOut.assign(static_cast<size_t>(dModel), 0.0f);
	session.ffPre.assign(static_cast<size_t>(session.ff1Width), 0.0f);
	session.ffAct.assign(static_cast<size_t>(tt.dFF), 0.0f);
	session.ffOut.assign(static_cast<size_t>(dModel), 0.0f);
	session.scores.assign(static_cast<size_t>(maxSeqLen), 0.0f);

	// Cache sinusoidal PE frequency terms (pow-free) for this dModel when needed.
	if (trainingConfig.transformer.positionalEncoding == glades::TransformerRunConfig::POSENC_SINUSOIDAL)
	{
		if (session.metricsEnabled)
		{
			const bool hit = (session.sinDModelCached == dModel && !session.sinInvDenomPair.empty());
			if (hit)
				++session.perf.sinCacheHits;
			else
				++session.perf.sinCacheMisses;
		}
		ensure_sinusoidal_cache(dModel, session.sinDModelCached, session.sinInvDenomPair);
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
		if (gs->allocate(gpuTransformerWeights, maxSeqLen, dModel, tt.dFF,
		                 dModelKV, nHeads, nKVHeads, tt.nLayers,
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
	const int padTokenId = tt.padTokenId;

	const bool metricsOn = session.metricsEnabled;
	const bool breakdown = metricsOn && transformerMetricsCfg.enableKvKernelBreakdown;
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
	const float eps = (trainingConfig.transformer.layerNormEps > 0.0f ? trainingConfig.transformer.layerNormEps : 1e-5f);
	const int normType = static_cast<int>(trainingConfig.transformer.normType);
	const int posEnc = static_cast<int>(trainingConfig.transformer.positionalEncoding);
	const int ropeDimOverride = trainingConfig.transformer.ropeDimOverride;
	const float ropeTheta = (trainingConfig.transformer.ropeTheta > 0.0f ? trainingConfig.transformer.ropeTheta : 10000.0f);
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
			const int dM = static_cast<int>(dModel);
			const int dMKV = static_cast<int>(dModelKV);
			const int dH = static_cast<int>(dHead);
			const unsigned int ff1Width = session.ff1Width;
			const int ff1W = static_cast<int>(ff1Width);
			const int dFFi = static_cast<int>(dFF);

			// Update key validity mask on GPU for this position.
			{
				unsigned char kv = (padTokenId >= 0 && static_cast<int>(tokenId) == padTokenId) ? 0u : 1u;
				gpu::device_memcpy_h2d(gs->keyValid.data() + pos, &kv, 1);
			}

			// --- Embedding ---
			// h[dModel] = tokE[tokenId, :]
			{
				int tid = static_cast<int>(tokenId);
				gs->tokenIdBuf.upload(&tid, 1);
				gg::embedding_gather(gw.tokE.data(), gs->tokenIdBuf.data(),
				                     1, static_cast<int>(vocab), dM, gs->h.data());
			}

			// --- Positional encoding ---
			if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
			{
				// Compute sinusoidal PE on CPU, upload, add to h on GPU.
				ensure_sinusoidal_cache(dModel, session.sinDModelCached, session.sinInvDenomPair);
				std::vector<float> peVec(dModel, 0.0f);
				for (unsigned int i = 0; i < dModel; ++i)
				{
					const unsigned int pairIdx = i / 2u;
					if (pairIdx < session.sinInvDenomPair.size())
					{
						const double angle = static_cast<double>(pos) * session.sinInvDenomPair[pairIdx];
						peVec[i] = (i % 2u == 0u) ? static_cast<float>(sin(angle)) : static_cast<float>(cos(angle));
					}
				}
				gs->peScratch.upload(&peVec[0], dModel);
				gg::add_residual(gs->h.data(), gs->peScratch.data(), dM);
			}
			// RoPE is applied after QKV projection, not here.
			// POSENC_NONE: nothing to do.

			// --- Per-layer forward ---
			for (unsigned int li = 0; li < tt.nLayers; ++li)
			{
				const gg::GpuTransformerWeights::Block& gb = gw.blocks[li];

				// Norm1: h[dModel] -> x1[dModel]
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
					gg::rmsnorm_forward(gs->h.data(), gb.ln1Gamma.data(), eps,
					                    1, dM, gs->x1.data(), gs->lnInvStd.data());
				else
					gg::layernorm_forward(gs->h.data(), gb.ln1Gamma.data(), gb.ln1Beta.data(),
					                      eps, 1, dM, gs->x1.data(), gs->lnMean.data(), gs->lnInvStd.data());

				// Q/K/V projections: sgemv y = W * x + b
				// q[dModel] = Wq[dModel, dModel] * x1[dModel]
				gg::sgemv_rowmajor(dM, dM, 1.0f, gb.Wq.data(), dM, gs->x1.data(), 0.0f, gs->q.data());
				if (gb.bq.allocated())
					gg::add_bias(gs->q.data(), gb.bq.data(), 1, dM);

				// kvec[dModelKV] = Wk[dModelKV, dModel] * x1[dModel]
				gg::sgemv_rowmajor(dMKV, dM, 1.0f, gb.Wk.data(), dM, gs->x1.data(), 0.0f, gs->kvec.data());
				if (gb.bk.allocated())
					gg::add_bias(gs->kvec.data(), gb.bk.data(), 1, dMKV);

				// vvec[dModelKV] = Wv[dModelKV, dModel] * x1[dModel]
				gg::sgemv_rowmajor(dMKV, dM, 1.0f, gb.Wv.data(), dM, gs->x1.data(), 0.0f, gs->vvec.data());
				if (gb.bv.allocated())
					gg::add_bias(gs->vvec.data(), gb.bv.data(), 1, dMKV);

				// RoPE on Q and K for this single token.
				// The GPU rope_apply kernel uses theta = t * invFreq[d] with t as the row
				// index. For T=1, t=0 always, giving no rotation. For incremental inference
				// we need theta = pos * invFreq[d]. We handle this by applying RoPE on CPU
				// (download, rotate, re-upload). The overhead is small since Q and K are
				// single vectors, and RoPE is not the bottleneck in inference.
				if (useRope && ropeDim >= 2u)
				{
					ensure_rope_cache(ropeDim, ropeTheta, session.ropeDimCached,
					                  session.ropeThetaCached, session.ropeInvFreq);
					std::vector<float> qCpu(dModel);
					std::vector<float> kvCpu(dModelKV);
					gs->q.download(&qCpu[0], dModel);
					gs->kvec.download(&kvCpu[0], dModelKV);
					for (unsigned int hq = 0; hq < nHeads; ++hq)
						rope_apply_vec(&qCpu[static_cast<size_t>(hq) * dHead], dHead, ropeDim,
						               session.ropeInvFreq, pos);
					for (unsigned int hk = 0; hk < nKVHeads; ++hk)
						rope_apply_vec(&kvCpu[static_cast<size_t>(hk) * dHead], dHead, ropeDim,
						               session.ropeInvFreq, pos);
					gs->q.upload(&qCpu[0], dModel);
					gs->kvec.upload(&kvCpu[0], dModelKV);
				}

				// Store K/V into GPU cache at [li, pos]
				{
					const size_t perLayer = static_cast<size_t>(gs->maxLen) * dModelKV;
					const size_t base = static_cast<size_t>(li) * perLayer + static_cast<size_t>(pos) * dModelKV;
					gpu::device_memcpy_d2d(gs->k.data() + base, gs->kvec.data(),
					                       dModelKV * sizeof(float));
					gpu::device_memcpy_d2d(gs->v.data() + base, gs->vvec.data(),
					                       dModelKV * sizeof(float));
				}

				// Attention: Q[nHeads*dHead] against KV cache -> attnConcat[nHeads*dHead]
				{
					const size_t perLayer = static_cast<size_t>(gs->maxLen) * dModelKV;
					const float* kLayer = gs->k.data() + static_cast<size_t>(li) * perLayer;
					const float* vLayer = gs->v.data() + static_cast<size_t>(li) * perLayer;
					const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dHead)));

					gg::kv_attention_incremental(
						gs->q.data(), kLayer, vLayer,
						gs->attnScores.data(), gs->keyValid.data(),
						static_cast<int>(nHeads), static_cast<int>(nKVHeads),
						dH, dMKV, static_cast<int>(gs->maxLen),
						static_cast<int>(pos), invSqrt, gs->attnConcat.data());
				}

				// Wo projection + residual: attnOut = Wo * attnConcat + bo; h += attnOut
				gg::sgemv_rowmajor(dM, dM, 1.0f, gb.Wo.data(), dM,
				                   gs->attnConcat.data(), 0.0f, gs->attnOut.data());
				if (gb.bo.allocated())
					gg::add_bias(gs->attnOut.data(), gb.bo.data(), 1, dM);
				gg::add_residual(gs->h.data(), gs->attnOut.data(), dM);

				// Norm2: h -> x1 (reuse x1 as scratch for post-LN2 output)
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
					gg::rmsnorm_forward(gs->h.data(), gb.ln2Gamma.data(), eps,
					                    1, dM, gs->x1.data(), gs->lnInvStd.data());
				else
					gg::layernorm_forward(gs->h.data(), gb.ln2Gamma.data(), gb.ln2Beta.data(),
					                      eps, 1, dM, gs->x1.data(), gs->lnMean.data(), gs->lnInvStd.data());

				// FFN: x1 -> ffPre -> ffAct -> ffOut
				// ffPre = W1 * x1 + b1
				gg::sgemv_rowmajor(ff1W, dM, 1.0f, gb.W1.data(), dM,
				                   gs->x1.data(), 0.0f, gs->ffPre.data());
				if (gb.b1.allocated())
					gg::add_bias(gs->ffPre.data(), gb.b1.data(), 1, ff1W);

				// Activation
				if (session.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU))
				{
					// SwiGLU: ffPre[2*dFF] -> ffAct[dFF]
					gg::swiglu_forward(gs->ffPre.data(), 1, dFFi, gs->ffAct.data());
				}
				else
				{
					const int act = static_cast<int>(trainingConfig.transformer.ffnActivation);
					if (act == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
						gg::gelu_forward(gs->ffPre.data(), dFFi, gs->ffAct.data());
					else
						gg::relu_forward(gs->ffPre.data(), dFFi, gs->ffAct.data());
				}

				// ffOut = W2 * ffAct + b2
				gg::sgemv_rowmajor(dM, dFFi, 1.0f, gb.W2.data(), dFFi,
				                   gs->ffAct.data(), 0.0f, gs->ffOut.data());
				if (gb.b2.allocated())
					gg::add_bias(gs->ffOut.data(), gb.b2.data(), 1, dM);

				// h += ffOut
				gg::add_residual(gs->h.data(), gs->ffOut.data(), dM);
			}

			// --- Logits: tied embedding ---
			if (outLogits)
			{
				// logits[vocab] = tokE[vocab, dModel] * h[dModel] + lmBias[vocab]
				gg::sgemv_rowmajor(static_cast<int>(vocab), dM, 1.0f,
				                   gw.tokE.data(), dM, gs->h.data(),
				                   0.0f, gs->logits.data());
				if (gw.lmBias.allocated())
					gg::add_bias(gs->logits.data(), gw.lmBias.data(), 1, static_cast<int>(vocab));

				// Download logits to CPU.
				if (outLogits->size() != vocab)
					outLogits->resize(vocab);
				gs->logits.download(&(*outLogits)[0], vocab);
			}

			session.curLen += 1u;
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
			    (session.ropeDimCached == ropeDim && session.ropeThetaCached == ropeTheta && !session.ropeInvFreq.empty());
			if (hit)
				++session.perf.ropeCacheHits;
			else
				++session.perf.ropeCacheMisses;
		}
		ensure_rope_cache(ropeDim, ropeTheta, session.ropeDimCached, session.ropeThetaCached, session.ropeInvFreq);
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
			const bool hit = (session.sinDModelCached == dModel && !session.sinInvDenomPair.empty());
			if (hit)
				++session.perf.sinCacheHits;
			else
				++session.perf.sinCacheMisses;
		}
		ensure_sinusoidal_cache(dModel, session.sinDModelCached, session.sinInvDenomPair);
		glades::transformer_kernels::add_sinusoidal_positional_encoding_inplace(&h[0], pos, dModel, session.sinInvDenomPair);
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
				rope_apply_vec(&q[static_cast<size_t>(hq) * static_cast<size_t>(dHead)], dHead, ropeDim, session.ropeInvFreq, pos);
			for (unsigned int hk = 0; hk < nKVHeads; ++hk)
				rope_apply_vec(&kvec[static_cast<size_t>(hk) * static_cast<size_t>(dHead)], dHead, ropeDim, session.ropeInvFreq, pos);
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
					                                               static_cast<size_t>(dModelKV));
					const glades::Tensor2DView<const uint16_t> vMat(vLayer16, static_cast<size_t>(pos) + 1u, static_cast<size_t>(dModelKV),
					                                               static_cast<size_t>(dModelKV));
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
					                                            static_cast<size_t>(dModelKV));
					const glades::Tensor2DView<const float> vMat(vLayer, static_cast<size_t>(pos) + 1u, static_cast<size_t>(dModelKV),
					                                            static_cast<size_t>(dModelKV));
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
				const int act = static_cast<int>(trainingConfig.transformer.ffnActivation);
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
				return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "transformerLmSessionAppend: non-finite hidden state");
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
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::NNetwork::transformerLmBatchSessionReset(glades::NNetwork::TransformerLmBatchSession& session,
                                                                       unsigned int batchSize,
                                                                       unsigned int maxSeqLen) const
{
	if (netType != TYPE_TRANSFORMER_DECODER)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: requires TYPE_TRANSFORMER_DECODER");
	if (!trainingConfig.transformer.enableTokenEmbedding)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: requires transformer.enableTokenEmbedding");
	if (!tensorTransformer.initialized || !tensorTransformer.tokenModel)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionReset: transformer tensors not initialized for token LM (loadModel or run once)");
	if (batchSize == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: batchSize is 0");
	if (maxSeqLen == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: maxSeqLen is 0");

	{
		const int posEnc = static_cast<int>(trainingConfig.transformer.positionalEncoding);
		if (posEnc != static_cast<int>(glades::TransformerRunConfig::POSENC_NONE) &&
		    posEnc != static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL) &&
		    posEnc != static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE))
		{
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: unknown positionalEncoding");
		}
	}

	const TensorTransformerState& tt = tensorTransformer;
	const unsigned int dModel = tt.dModel;
	const unsigned int nHeads = tt.nHeads;
	const unsigned int nKVHeads = (tt.nKVHeads > 0u ? tt.nKVHeads : nHeads);
	const unsigned int dHead = (nHeads > 0u ? (dModel / nHeads) : 0u);
	const unsigned int dModelKV = nKVHeads * dHead;
	if (dHead == 0u || dModelKV == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmBatchSessionReset: invalid head dimensions");

	// Allocation sizing safety: overflow checks + hard cap.
	{
		const size_t B = static_cast<size_t>(batchSize);
		const size_t L = static_cast<size_t>(tt.nLayers);
		const size_t S = static_cast<size_t>(maxSeqLen);
		const size_t K = static_cast<size_t>(dModelKV);
		size_t perSeqElems = 0u;
		size_t tmp = 0u;
		if (!checked_mul_size(L, S, tmp) || !checked_mul_size(tmp, K, perSeqElems))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: KV-cache size overflow (maxSeqLen too large)");
		size_t totalElems = 0u;
		if (!checked_mul_size(B, perSeqElems, totalElems))
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "transformerLmBatchSessionReset: KV-cache size overflow (batchSize too large)");

		const bool kvLowp = (trainingConfig.transformer.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_F16) ||
		                    (trainingConfig.transformer.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_BF16);
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
		const unsigned long long cap = kv_session_max_bytes();
		const unsigned long long wantBytes = kvBytes + keyMaskBytes;
		if (cap > 0ULL && wantBytes > cap)
		{
			std::ostringstream oss;
			oss << "transformerLmBatchSessionReset: session allocation exceeds cap (want "
			    << bytes_to_human(wantBytes) << ", cap " << bytes_to_human(cap)
			    << "). Reduce batchSize/maxSeqLen or set GLADES_TRANSFORMER_KV_SESSION_MAX_BYTES.";
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, oss.str());
		}
	}

	session.reset();
	session.initialized = true;
	session.metricsEnabled = transformerMetricsCfg.enable;
	session.perf.reset();
	session.batchSize = batchSize;
	session.maxLen = maxSeqLen;
	session.curLen.assign(static_cast<size_t>(batchSize), 0u);
	session.dModel = dModel;
	session.dFF = tt.dFF;
	session.nHeads = nHeads;
	session.nKVHeads = nKVHeads;
	session.nLayers = tt.nLayers;
	session.dHead = dHead;
	session.dModelKV = dModelKV;
	session.ffnKind = tt.ffnKind;
	session.ff1Width =
	    (tt.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * tt.dFF) : tt.dFF;

	size_t perSeq = 0u;
	{
		size_t tmp = 0u;
		(void)checked_mul_size(static_cast<size_t>(tt.nLayers), static_cast<size_t>(maxSeqLen), tmp);
		(void)checked_mul_size(tmp, static_cast<size_t>(dModelKV), perSeq);
	}
	// KV-cache dtype for this session.
	if (trainingConfig.transformer.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_F16)
		session.kvCacheDType = glades::NNetwork::TransformerLmBatchSession::KV_CACHE_F16;
	else if (trainingConfig.transformer.kvCacheDType == glades::TransformerRunConfig::KV_CACHE_BF16)
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

	// Shared scratch sized once (reused while looping).
	session.h.assign(static_cast<size_t>(dModel), 0.0f);
	session.x1.assign(static_cast<size_t>(dModel), 0.0f);
	session.x2.assign(static_cast<size_t>(dModel), 0.0f);
	session.q.assign(static_cast<size_t>(dModel), 0.0f);
	session.kvec.assign(static_cast<size_t>(dModelKV), 0.0f);
	session.vvec.assign(static_cast<size_t>(dModelKV), 0.0f);
	session.attnConcat.assign(static_cast<size_t>(dModel), 0.0f);
	session.attnOut.assign(static_cast<size_t>(dModel), 0.0f);
	session.ffPre.assign(static_cast<size_t>(session.ff1Width), 0.0f);
	session.ffAct.assign(static_cast<size_t>(tt.dFF), 0.0f);
	session.ffOut.assign(static_cast<size_t>(dModel), 0.0f);
	session.scores.assign(static_cast<size_t>(maxSeqLen), 0.0f);

	if (trainingConfig.transformer.positionalEncoding == glades::TransformerRunConfig::POSENC_SINUSOIDAL)
	{
		if (session.metricsEnabled)
		{
			const bool hit = (session.sinDModelCached == dModel && !session.sinInvDenomPair.empty());
			if (hit)
				++session.perf.sinCacheHits;
			else
				++session.perf.sinCacheMisses;
		}
		ensure_sinusoidal_cache(dModel, session.sinDModelCached, session.sinInvDenomPair);
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
	const int padTokenId = tt.padTokenId;

	const bool metricsOn = session.metricsEnabled;
	const bool breakdown = metricsOn && transformerMetricsCfg.enableKvKernelBreakdown;
	ScopedTimerMs tTotal(this, metricsOn, &session.perf.msTotal);

	// Config knobs
	const float eps = (trainingConfig.transformer.layerNormEps > 0.0f ? trainingConfig.transformer.layerNormEps : 1e-5f);
	const int normType = static_cast<int>(trainingConfig.transformer.normType);
	const int posEnc = static_cast<int>(trainingConfig.transformer.positionalEncoding);
	const int ropeDimOverride = trainingConfig.transformer.ropeDimOverride;
	const float ropeTheta = (trainingConfig.transformer.ropeTheta > 0.0f ? trainingConfig.transformer.ropeTheta : 10000.0f);
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
			    (session.ropeDimCached == ropeDim && session.ropeThetaCached == ropeTheta && !session.ropeInvFreq.empty());
			if (hit)
				++session.perf.ropeCacheHits;
			else
				++session.perf.ropeCacheMisses;
		}
		ensure_rope_cache(ropeDim, ropeTheta, session.ropeDimCached, session.ropeThetaCached, session.ropeInvFreq);
	}

	// Ensure sinusoidal cache if needed.
	if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
	{
		if (metricsOn)
		{
			const bool hit = (session.sinDModelCached == session.dModel && !session.sinInvDenomPair.empty());
			if (hit)
				++session.perf.sinCacheHits;
			else
				++session.perf.sinCacheMisses;
		}
		ensure_sinusoidal_cache(session.dModel, session.sinDModelCached, session.sinInvDenomPair);
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
			glades::transformer_kernels::add_sinusoidal_positional_encoding_inplace(&h[0], pos, session.dModel, session.sinInvDenomPair);
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
					rope_apply_vec(&q[static_cast<size_t>(hq_i) * static_cast<size_t>(session.dHead)], session.dHead, ropeDim, session.ropeInvFreq, pos);
				for (unsigned int hk_i = 0; hk_i < session.nKVHeads; ++hk_i)
					rope_apply_vec(&kvec[static_cast<size_t>(hk_i) * static_cast<size_t>(session.dHead)], session.dHead, ropeDim, session.ropeInvFreq, pos);
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
						                                               static_cast<size_t>(session.dModelKV));
						const glades::Tensor2DView<const uint16_t> vMat(vLayer16, static_cast<size_t>(pos) + 1u, static_cast<size_t>(session.dModelKV),
						                                               static_cast<size_t>(session.dModelKV));
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
						                                            static_cast<size_t>(session.dModelKV));
						const glades::Tensor2DView<const float> vMat(vLayer, static_cast<size_t>(pos) + 1u, static_cast<size_t>(session.dModelKV),
						                                            static_cast<size_t>(session.dModelKV));
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
					const int act = static_cast<int>(trainingConfig.transformer.ffnActivation);
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
					return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "transformerLmBatchSessionAppendSelective: non-finite hidden state");
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

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::NNetwork::transformerLmForwardLastLogits(const std::vector<unsigned int>& tokenIds,
                                                                        std::vector<float>& outLogits) const
{
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
				const float* qh = &q[t * static_cast<size_t>(dModel) + static_cast<size_t>(hq) * static_cast<size_t>(dHead)];

				// causal scores[u] for u=0..t
				float* scoreBuf = scores.empty() ? NULL : &scores[0];
				if (!scoreBuf)
					return NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerLmForwardLastLogits: invalid attention scratch pointer");
				float* outHead = &attnConcat[static_cast<size_t>(hq) * static_cast<size_t>(dHead)];
				const glades::Tensor2DView<const float> kMat(&k[0], t + 1u, static_cast<size_t>(dModelKV), static_cast<size_t>(dModelKV));
				const glades::Tensor2DView<const float> vMat(&v[0], t + 1u, static_cast<size_t>(dModelKV), static_cast<size_t>(dModelKV));
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
					return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "transformerLmForwardLastLogits: non-finite hidden state");
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

