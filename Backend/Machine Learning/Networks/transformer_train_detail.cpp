// Internal transformer training helpers extracted from sgd_transformer.cpp.
#include "transformer_train_detail.h"

#include "glades_thread_pool.h"
#include "sgd_utils.h"
#include "transformer_common_utils.h"
#include "transformer_kernels.h"
#include "transformer_ops.h"

#include <algorithm>
#include <vector>

namespace glades {
namespace transformer_train_detail {

using glades::transformer_common::checked_mul_size;

namespace {

struct LinearFwdCtx;
void linear_forward_opt_parallel(const float* X, unsigned int T, unsigned int inSize,
                                 const std::vector<float>& W, const std::vector<float>& b,
                                 unsigned int outSize, float* Y);

struct LinearBwdGWCtx;
void linear_bwd_gw_body(void* ud, unsigned int begin, unsigned int end);

struct LinearBwdDXCtx
{
	const float* dY;
	const float* W;
	unsigned int inSize;
	unsigned int outSize;
	float* dX;
};

void linear_bwd_dx_body(void* ud, unsigned int begin, unsigned int end);
void linear_backward_accum_parallel(const float* X, const float* dY, unsigned int T,
                                    unsigned int inSize, unsigned int outSize,
                                    std::vector<float>& gW, std::vector<float>& gB,
                                    const std::vector<float>& W, float* dXOut);

struct LinearFwdCtx
{
	const float* X;
	unsigned int inSize;
	const float* W;
	const float* b;
	unsigned int outSize;
	unsigned int bSize;
	float* Y;
};

static const unsigned int LINEAR_FWD_TILE = 32u;

struct LinearFwdTiledCtx
{
	const float* X;
	unsigned int T;
	unsigned int inSize;
	const float* W;
	const float* b;
	unsigned int outSize;
	unsigned int bSize;
	float* Y;
};

struct LinearBwdGWCtx
{
	const float* X;
	const float* dY;
	unsigned int T;
	unsigned int inSize;
	unsigned int outSize;
	float* gW;
	float* gB;
};

} // namespace

float clip_maybe(float v, float limit)
{
	return glades::sgd_detail::clipf_maybe(v, limit);
}

float transformer_schedule_multiplier(const glades::LearningRateScheduleConfig& schedule,
                                      int epochIdx,
                                      unsigned int stepInEpoch,
                                      unsigned int totalStepsInEpoch)
{
	if (schedule.type == glades::LearningRateScheduleConfig::NONE)
		return 1.0f;

	if (totalStepsInEpoch == 0u)
		return schedule.multiplier(epochIdx);

	if (stepInEpoch > totalStepsInEpoch)
		stepInEpoch = totalStepsInEpoch;

	const double epochProgress =
	    static_cast<double>(epochIdx) +
	    (static_cast<double>(stepInEpoch) / static_cast<double>(totalStepsInEpoch));
	return schedule.multiplierFractionalEpoch(epochProgress);
}

void add_positional_encoding(float* h,
                             unsigned int T,
                             unsigned int dModel,
                             const std::vector<double>& invDenomPair)
{
	if (T == 0u || dModel == 0u || !h)
		return;
	glades::transformer_kernels::add_sinusoidal_positional_encoding_seq_inplace(h, T, dModel, invDenomPair);
}

void linear_forward_maybe_lowp(const float* X,
                               unsigned int T,
                               unsigned int inSize,
                               const std::vector<float>& W,
                               const std::vector<uint16_t>& WLowp,
                               bool useLowp,
                               int lowpDType,
                               const std::vector<float>& b,
                               unsigned int outSize,
                               float* Y)
{
	if (useLowp)
	{
		if (WLowp.size() == static_cast<size_t>(outSize) * static_cast<size_t>(inSize))
			glades::transformer_kernels::linear_forward_lowp(X, T, inSize, &WLowp[0], lowpDType, b, outSize, Y);
		else
			linear_forward_opt_parallel(X, T, inSize, W, b, outSize, Y);
	}
	else
	{
		linear_forward_opt_parallel(X, T, inSize, W, b, outSize, Y);
	}
}

static void linear_backward_accum(const float* X,
                                  const float* dY,
                                  unsigned int T,
                                  unsigned int inSize,
                                  unsigned int outSize,
                                  std::vector<float>& gW,
                                  std::vector<float>& gB,
                                  const std::vector<float>& W,
                                  float* dXOut)
{
	linear_backward_accum_parallel(X, dY, T, inSize, outSize, gW, gB, W, dXOut);
}

void linear_backward_accum_maybe_lowp(const float* X,
                                      const float* dY,
                                      unsigned int T,
                                      unsigned int inSize,
                                      unsigned int outSize,
                                      std::vector<float>& gW,
                                      std::vector<float>& gB,
                                      const std::vector<float>& WMaster,
                                      const std::vector<uint16_t>& WLowp,
                                      bool useLowp,
                                      int lowpDType,
                                      float* dXOut)
{
	linear_backward_accum(X, dY, T, inSize, outSize, gW, gB, WMaster, NULL);

	if (!dXOut)
		return;
	std::fill(dXOut, dXOut + (static_cast<size_t>(T) * static_cast<size_t>(inSize)), 0.0f);

	const bool haveLowp = useLowp && (WLowp.size() == static_cast<size_t>(outSize) * static_cast<size_t>(inSize));
	if (!haveLowp)
	{
		const bool worthParallel = (static_cast<unsigned long long>(T) * outSize * inSize >= 500000ULL);
		glades::ThreadPool& pool = glades::ThreadPool::instance();
		if (worthParallel && T > 1u && pool.numThreads() > 1u)
		{
			LinearBwdDXCtx ctx;
			ctx.dY = dY;
			ctx.W = WMaster.empty() ? NULL : &WMaster[0];
			ctx.inSize = inSize;
			ctx.outSize = outSize;
			ctx.dX = dXOut;
			pool.parallel_for(T, linear_bwd_dx_body, &ctx);
		}
		else
		{
			for (unsigned int t = 0; t < T; ++t)
			{
				const size_t dyOff = static_cast<size_t>(t) * static_cast<size_t>(outSize);
				const size_t dxOff = static_cast<size_t>(t) * static_cast<size_t>(inSize);
				for (unsigned int o = 0; o < outSize; ++o)
				{
					const float dy = dY[dyOff + o];
					glades::transformer_kernels::axpy_f32(
					    dXOut + dxOff,
					    &WMaster[static_cast<size_t>(o) * static_cast<size_t>(inSize)],
					    dy, inSize);
				}
			}
		}
	}
	else
	{
		std::vector<float> wRowBuf(inSize);
		for (unsigned int t = 0; t < T; ++t)
		{
			const size_t dyOff = static_cast<size_t>(t) * static_cast<size_t>(outSize);
			const size_t dxOff = static_cast<size_t>(t) * static_cast<size_t>(inSize);
			for (unsigned int o = 0; o < outSize; ++o)
			{
				const float dy = dY[dyOff + o];
				const size_t wRowOff = static_cast<size_t>(o) * static_cast<size_t>(inSize);
				for (unsigned int i = 0; i < inSize; ++i)
					wRowBuf[i] = glades::transformer_kernels::lowp_to_float(WLowp[wRowOff + i], lowpDType);
				glades::transformer_kernels::axpy_f32(dXOut + dxOff, &wRowBuf[0], dy, inSize);
			}
		}
	}
}

void attn_fwd_body(void* ud, unsigned int begin, unsigned int end)
{
	const AttnFwdCtx& c = *static_cast<const AttnFwdCtx*>(ud);
	for (unsigned int h = begin; h < end; ++h)
	{
		const unsigned int kvHead = (c.nKVHeads == c.nHeads) ? h : (c.groupSize > 0u ? (h / c.groupSize) : 0u);
		size_t qOff = 0u, kOff = 0u, vOff = 0u, oOff = 0u;
		if (!checked_mul_size(static_cast<size_t>(h), static_cast<size_t>(c.dHead), qOff) ||
		    !checked_mul_size(static_cast<size_t>(kvHead), static_cast<size_t>(c.dHead), kOff) ||
		    !checked_mul_size(static_cast<size_t>(kvHead), static_cast<size_t>(c.dHead), vOff) ||
		    !checked_mul_size(static_cast<size_t>(h), static_cast<size_t>(c.dHead), oOff))
			return;
		glades::transformer_ops::scaled_dot_product_attention_forward_flash_strided(
		    c.Q + qOff,
		    c.dModel,
		    c.K + kOff,
		    c.dModelKV,
		    c.V + vOff,
		    c.dModelKV,
		    c.T,
		    c.dHead,
		    c.dHead,
		    c.causal,
		    c.O + oOff,
		    c.dModel,
		    c.keyAllowed);
	}
}

void attn_bwd_body(void* ud, unsigned int begin, unsigned int end)
{
	const AttnBwdCtx& c = *static_cast<const AttnBwdCtx*>(ud);
	const bool chunked = (c.nChunksPerHead > 1u && c.dKVscratch != NULL);

	for (unsigned int item = begin; item < end; ++item)
	{
		unsigned int h, tBegin, tEnd;

		if (chunked)
		{
			h = item / c.nChunksPerHead;
			const unsigned int chunkIdx = item % c.nChunksPerHead;
			const unsigned int rowsPerChunk = (c.T + c.nChunksPerHead - 1u) / c.nChunksPerHead;
			tBegin = chunkIdx * rowsPerChunk;
			tEnd = tBegin + rowsPerChunk;
			if (tEnd > c.T) tEnd = c.T;
			if (tBegin >= tEnd) continue;
		}
		else
		{
			const unsigned int kvh = item;
			const unsigned int hStart = kvh * c.groupSize;
			const unsigned int hEnd = hStart + c.groupSize;
			size_t kvhOff = 0u;
			if (!checked_mul_size(static_cast<size_t>(kvh), static_cast<size_t>(c.dHead), kvhOff))
				continue;
			for (unsigned int hh = hStart; hh < hEnd && hh < c.nHeads; ++hh)
			{
				size_t hhOff = 0u;
				if (!checked_mul_size(static_cast<size_t>(hh), static_cast<size_t>(c.dHead), hhOff))
					continue;
				glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_strided(
				    c.Q + hhOff,
				    c.dModel,
				    c.K + kvhOff,
				    c.dModelKV,
				    c.V + kvhOff,
				    c.dModelKV,
				    c.dO + hhOff,
				    c.dModel,
				    c.T, c.dHead, c.dHead, c.causal,
				    c.dQ + hhOff,
				    c.dModel,
				    c.dK + kvhOff,
				    c.dModelKV,
				    c.dV + kvhOff,
				    c.dModelKV,
				    c.keyAllowed);
			}
			continue;
		}

		const unsigned int kvh = (c.nKVHeads == c.nHeads) ? h : (c.groupSize > 0u ? (h / c.groupSize) : 0u);
		size_t scratchSize = 0u;
		if (!checked_mul_size(static_cast<size_t>(c.T), static_cast<size_t>(c.dHead), scratchSize))
			continue;
		size_t scratchItemOff = 0u;
		if (!checked_mul_size(static_cast<size_t>(item), scratchSize * 2u, scratchItemOff))
			continue;
		float* dKlocal = c.dKVscratch + scratchItemOff;
		float* dVlocal = dKlocal + scratchSize;

		size_t hOff = 0u, kvhOff = 0u;
		if (!checked_mul_size(static_cast<size_t>(h), static_cast<size_t>(c.dHead), hOff) ||
		    !checked_mul_size(static_cast<size_t>(kvh), static_cast<size_t>(c.dHead), kvhOff))
			continue;

		glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_chunk(
		    c.Q + hOff,
		    c.dModel,
		    c.K + kvhOff,
		    c.dModelKV,
		    c.V + kvhOff,
		    c.dModelKV,
		    c.dO + hOff,
		    c.dModel,
		    tBegin, tEnd,
		    c.T, c.dHead, c.dHead, c.causal,
		    c.dQ + hOff,
		    c.dModel,
		    dKlocal, dVlocal,
		    c.keyAllowed);
	}
}

void attn_bwd_reduce_body(void* ud, unsigned int begin, unsigned int end)
{
	const AttnBwdReduceCtx& c = *static_cast<const AttnBwdReduceCtx*>(ud);
	const size_t scratchSize = static_cast<size_t>(c.T) * static_cast<size_t>(c.dHead);

	for (unsigned int kvh = begin; kvh < end; ++kvh)
	{
		const unsigned int hStart = kvh * c.groupSize;
		const unsigned int hEnd = (hStart + c.groupSize < c.nHeads) ? (hStart + c.groupSize) : c.nHeads;

		float* dKout = c.dK + static_cast<size_t>(kvh) * static_cast<size_t>(c.dHead);
		float* dVout = c.dV + static_cast<size_t>(kvh) * static_cast<size_t>(c.dHead);

		for (unsigned int h = hStart; h < hEnd; ++h)
		{
			for (unsigned int ch = 0; ch < c.nChunksPerHead; ++ch)
			{
				const unsigned int item = h * c.nChunksPerHead + ch;
				const float* dKlocal = c.dKVscratch + static_cast<size_t>(item) * scratchSize * 2u;
				const float* dVlocal = dKlocal + scratchSize;

				for (unsigned int u = 0; u < c.T; ++u)
				{
					const size_t localOff = static_cast<size_t>(u) * c.dHead;
					const size_t stridedOff = static_cast<size_t>(u) * c.dModelKV;
					glades::transformer_kernels::axpy_f32(
					    dKout + stridedOff,
					    dKlocal + localOff,
					    1.0f, c.dHead);
					glades::transformer_kernels::axpy_f32(
					    dVout + stridedOff,
					    dVlocal + localOff,
					    1.0f, c.dHead);
				}
			}
		}
	}
}

void rope_body(void* ud, unsigned int begin, unsigned int end)
{
	const RopeFwdCtx& c = *static_cast<const RopeFwdCtx*>(ud);
	for (unsigned int h = begin; h < end; ++h)
	{
		glades::transformer_kernels::rope_apply_inplace_strided(
		    c.buf + static_cast<size_t>(h) * static_cast<size_t>(c.dHead),
		    c.T, c.rowStride, c.dHead, c.ropeDim, *c.invFreq, c.inverse);
	}
}

void norm_fwd_body(void* ud, unsigned int begin, unsigned int end)
{
	const NormFwdCtx& c = *static_cast<const NormFwdCtx*>(ud);
	for (unsigned int t = begin; t < end; ++t)
	{
		const size_t off = static_cast<size_t>(t) * static_cast<size_t>(c.D);
		if (c.isRmsNorm)
		{
			double sumsq = 0.0;
			for (unsigned int i = 0; i < c.D; ++i)
			{
				const double xd = static_cast<double>(c.X[off + i]);
				sumsq += xd * xd;
			}
			const double mean2 = sumsq / static_cast<double>(c.D);
			const double invRms = 1.0 / sqrt(mean2 + static_cast<double>(c.eps));
			c.invStdOut[t] = static_cast<float>(invRms);
			for (unsigned int i = 0; i < c.D; ++i)
			{
				const float g = (i < c.gammaSize) ? c.gamma[i] : 1.0f;
				const float b = (i < c.betaSize) ? c.beta[i] : 0.0f;
				c.Y[off + i] = (c.X[off + i] * static_cast<float>(invRms)) * g + b;
			}
		}
		else
		{
			double sum = 0.0;
			for (unsigned int i = 0; i < c.D; ++i)
				sum += static_cast<double>(c.X[off + i]);
			const double mean = sum / static_cast<double>(c.D);
			double var = 0.0;
			for (unsigned int i = 0; i < c.D; ++i)
			{
				const double d = static_cast<double>(c.X[off + i]) - mean;
				var += d * d;
			}
			var /= static_cast<double>(c.D);
			const double invStd = 1.0 / sqrt(var + static_cast<double>(c.eps));
			if (c.meanOut)
				c.meanOut[t] = static_cast<float>(mean);
			c.invStdOut[t] = static_cast<float>(invStd);
			for (unsigned int i = 0; i < c.D; ++i)
			{
				const float xn = static_cast<float>((static_cast<double>(c.X[off + i]) - mean) * invStd);
				const float g = (i < c.gammaSize) ? c.gamma[i] : 1.0f;
				const float b = (i < c.betaSize) ? c.beta[i] : 0.0f;
				c.Y[off + i] = xn * g + b;
			}
		}
	}
}

void tied_emb_logits_body(void* ud, unsigned int begin, unsigned int end)
{
	const TiedEmbLogitsCtx& c = *static_cast<const TiedEmbLogitsCtx*>(ud);
	for (unsigned int t = begin; t < end; ++t)
	{
		const float* ht = c.H + static_cast<size_t>(t) * static_cast<size_t>(c.dModel);
		float* zt = c.logitsOut + static_cast<size_t>(t) * static_cast<size_t>(c.vocab);
		glades::transformer_kernels::gemv_rowmajor_bias_block4_unroll8_into(
		    ht, c.dModel, c.tokE, c.vocab,
		    c.lmBias, c.lmBiasSize, zt);
	}
}

namespace {

void linear_fwd_tiled_body(void* ud, unsigned int begin, unsigned int end)
{
	const LinearFwdTiledCtx& c = *static_cast<const LinearFwdTiledCtx*>(ud);
	for (unsigned int o = begin; o < end; ++o)
	{
		const float* wRow = c.W + static_cast<size_t>(o) * static_cast<size_t>(c.inSize);
		const float bias = (o < c.bSize) ? c.b[o] : 0.0f;
		for (unsigned int t = 0; t < c.T; ++t)
		{
			const float* xt = c.X + static_cast<size_t>(t) * static_cast<size_t>(c.inSize);
			float* yt = c.Y + static_cast<size_t>(t) * static_cast<size_t>(c.outSize);
			yt[o] = bias + glades::transformer_kernels::dot_f32(xt, wRow, c.inSize);
		}
	}
}

void linear_forward_opt_parallel(const float* X,
                                 unsigned int T,
                                 unsigned int inSize,
                                 const std::vector<float>& W,
                                 const std::vector<float>& b,
                                 unsigned int outSize,
                                 float* Y)
{
	if (!X || !Y || T == 0u || inSize == 0u || outSize == 0u)
		return;

	const bool worthParallel = (static_cast<unsigned long long>(T) * outSize * inSize >= 500000ULL);
	glades::ThreadPool& pool = glades::ThreadPool::instance();
	if (worthParallel && pool.numThreads() > 1u)
	{
		LinearFwdTiledCtx ctx;
		ctx.X = X;
		ctx.T = T;
		ctx.inSize = inSize;
		ctx.W = W.empty() ? NULL : &W[0];
		ctx.b = b.empty() ? NULL : &b[0];
		ctx.bSize = static_cast<unsigned int>(b.size());
		ctx.outSize = outSize;
		ctx.Y = Y;
		pool.parallel_for(outSize, linear_fwd_tiled_body, &ctx);
	}
	else
	{
		glades::transformer_kernels::linear_forward_opt(X, T, inSize, W, b, outSize, Y);
	}
}

void linear_bwd_gw_body(void* ud, unsigned int begin, unsigned int end)
{
	const LinearBwdGWCtx& c = *static_cast<const LinearBwdGWCtx*>(ud);
	for (unsigned int o = begin; o < end; ++o)
	{
		float* gWrow = c.gW + static_cast<size_t>(o) * static_cast<size_t>(c.inSize);
		float gbAcc = 0.0f;
		for (unsigned int t = 0; t < c.T; ++t)
		{
			const float dy = c.dY[static_cast<size_t>(t) * static_cast<size_t>(c.outSize) + o];
			gbAcc += dy;
			glades::transformer_kernels::axpy_f32(gWrow,
			    c.X + static_cast<size_t>(t) * static_cast<size_t>(c.inSize), dy, c.inSize);
		}
		c.gB[o] += gbAcc;
	}
}

void linear_bwd_dx_body(void* ud, unsigned int begin, unsigned int end)
{
	const LinearBwdDXCtx& c = *static_cast<const LinearBwdDXCtx*>(ud);
	static const unsigned int DX_TILE = 32u;
	for (unsigned int o0 = 0; o0 < c.outSize; o0 += DX_TILE)
	{
		const unsigned int o1 = (o0 + DX_TILE < c.outSize) ? (o0 + DX_TILE) : c.outSize;
		for (unsigned int t = begin; t < end; ++t)
		{
			const size_t dyOff = static_cast<size_t>(t) * static_cast<size_t>(c.outSize);
			const size_t dxOff = static_cast<size_t>(t) * static_cast<size_t>(c.inSize);
			for (unsigned int o = o0; o < o1; ++o)
			{
				const float dy = c.dY[dyOff + o];
				glades::transformer_kernels::axpy_f32(
				    c.dX + dxOff,
				    &c.W[static_cast<size_t>(o) * static_cast<size_t>(c.inSize)],
				    dy, c.inSize);
			}
		}
	}
}

void linear_backward_accum_parallel(const float* X,
                                    const float* dY,
                                    unsigned int T,
                                    unsigned int inSize,
                                    unsigned int outSize,
                                    std::vector<float>& gW,
                                    std::vector<float>& gB,
                                    const std::vector<float>& W,
                                    float* dXOut)
{
	if (gW.size() != static_cast<size_t>(outSize) * static_cast<size_t>(inSize))
		gW.assign(static_cast<size_t>(outSize) * static_cast<size_t>(inSize), 0.0f);
	if (gB.size() != outSize)
		gB.assign(outSize, 0.0f);

	const bool worthParallel = (static_cast<unsigned long long>(T) * outSize * inSize >= 500000ULL);
	glades::ThreadPool& pool = glades::ThreadPool::instance();
	const bool doParallel = worthParallel && pool.numThreads() > 1u;

	if (doParallel && outSize > 1u)
	{
		LinearBwdGWCtx ctx;
		ctx.X = X;
		ctx.dY = dY;
		ctx.T = T;
		ctx.inSize = inSize;
		ctx.outSize = outSize;
		ctx.gW = &gW[0];
		ctx.gB = &gB[0];
		pool.parallel_for(outSize, linear_bwd_gw_body, &ctx);
	}
	else
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const size_t xOff = static_cast<size_t>(t) * static_cast<size_t>(inSize);
			const size_t dyOff = static_cast<size_t>(t) * static_cast<size_t>(outSize);
			for (unsigned int o = 0; o < outSize; ++o)
			{
				const float dy = dY[dyOff + o];
				gB[o] += dy;
				glades::transformer_kernels::axpy_f32(
				    &gW[static_cast<size_t>(o) * static_cast<size_t>(inSize)],
				    X + xOff, dy, inSize);
			}
		}
	}

	if (!dXOut)
		return;
	std::fill(dXOut, dXOut + (static_cast<size_t>(T) * static_cast<size_t>(inSize)), 0.0f);
	if (doParallel && T > 1u)
	{
		LinearBwdDXCtx ctx;
		ctx.dY = dY;
		ctx.W = W.empty() ? NULL : &W[0];
		ctx.inSize = inSize;
		ctx.outSize = outSize;
		ctx.dX = dXOut;
		pool.parallel_for(T, linear_bwd_dx_body, &ctx);
	}
	else
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const size_t dyOff = static_cast<size_t>(t) * static_cast<size_t>(outSize);
			const size_t dxOff = static_cast<size_t>(t) * static_cast<size_t>(inSize);
			for (unsigned int o = 0; o < outSize; ++o)
			{
				const float dy = dY[dyOff + o];
				glades::transformer_kernels::axpy_f32(
				    dXOut + dxOff,
				    &W[static_cast<size_t>(o) * static_cast<size_t>(inSize)],
				    dy, inSize);
			}
		}
	}
}

} // namespace

} // namespace transformer_train_detail
} // namespace glades
