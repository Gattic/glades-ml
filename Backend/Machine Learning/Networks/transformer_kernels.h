// Shared Transformer math kernels used by both training and inference paths.
//
// Goal:
// - Avoid having subtly different implementations in `sgd_transformer.cpp` vs `transformer_infer.cpp`.
// - Keep this header-only (inline) so it can be used from multiple translation units safely.
//
// NOTE:
// - These are reference (scalar) implementations intended to be correct and deterministic.
// - Higher-performance vectorized/fused kernels can be introduced behind the same APIs later.
// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdint.h>
#if defined(__AVX2__) || defined(__SSE2__)
#include <immintrin.h>
#endif
#include <vector>

namespace glades {
namespace transformer_kernels {

inline bool is_finite(float x) { return std::isfinite(x); }

// === Optional SIMD helpers (x86) ===
//
// These are used to speed up hot inference/training kernels while keeping a scalar fallback.
// They intentionally use float accumulation (not double) for throughput.
static inline float dot_f32(const float* a, const float* b, unsigned int n)
{
	if (!a || !b || n == 0u)
		return 0.0f;
#if defined(__AVX2__)
	__m256 acc = _mm256_setzero_ps();
	unsigned int i = 0u;
	for (; (i + 7u) < n; i += 8u)
	{
		const __m256 va = _mm256_loadu_ps(a + i);
		const __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
		acc = _mm256_fmadd_ps(va, vb, acc);
#else
		acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
#endif
	}
	// Horizontal sum
	__m128 lo = _mm256_castps256_ps128(acc);
	__m128 hi = _mm256_extractf128_ps(acc, 1);
	__m128 sum = _mm_add_ps(lo, hi);
	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);
	float out = _mm_cvtss_f32(sum);
	for (; i < n; ++i)
		out += a[i] * b[i];
	return out;
#else
	double acc = 0.0;
	for (unsigned int i = 0; i < n; ++i)
		acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
	return static_cast<float>(acc);
#endif
}

static inline void axpy_f32(float* y, const float* x, float a, unsigned int n)
{
	if (!y || !x || n == 0u)
		return;
#if defined(__AVX2__)
	const __m256 va = _mm256_set1_ps(a);
	unsigned int i = 0u;
	for (; (i + 7u) < n; i += 8u)
	{
		const __m256 vx = _mm256_loadu_ps(x + i);
		__m256 vy = _mm256_loadu_ps(y + i);
#if defined(__FMA__)
		vy = _mm256_fmadd_ps(va, vx, vy);
#else
		vy = _mm256_add_ps(vy, _mm256_mul_ps(va, vx));
#endif
		_mm256_storeu_ps(y + i, vy);
	}
	for (; i < n; ++i)
		y[i] += a * x[i];
#else
	for (unsigned int i = 0; i < n; ++i)
		y[i] += a * x[i];
#endif
}

// === Debug-mode precondition checking ===
//
// Many kernels in this header are leaf routines used in hot paths. When a caller violates
// a precondition (e.g. provides an undersized cache buffer), silent early-returns make
// debugging extremely difficult.
//
// Policy:
// - In debug builds, fail fast with an assert so the bug is loud and local.
// - In release builds, we still return early to avoid undefined behavior.
#ifndef GLADES_KERNEL_ASSERT
#define GLADES_KERNEL_ASSERT(expr) assert(expr)
#endif

// === FP16 (IEEE 754 binary16) helpers ===
//
// These are used for KV-cache compression in inference sessions.
// They are dependency-free and deterministic.
inline uint16_t float_to_half_rn(float f)
{
	// Based on well-known float<->half bit conversions (round-to-nearest-even).
	union
	{
		float f;
		uint32_t u;
	} v;
	v.f = f;

	const uint32_t sign = (v.u >> 16) & 0x8000u;
	uint32_t exp = (v.u >> 23) & 0xFFu;
	uint32_t mant = v.u & 0x7FFFFFu;

	// NaN/Inf
	if (exp == 0xFFu)
	{
		if (mant == 0u)
			return static_cast<uint16_t>(sign | 0x7C00u); // Inf
		// Quiet NaN: keep some payload.
		mant >>= 13;
		if (mant == 0u) mant = 1u;
		return static_cast<uint16_t>(sign | 0x7C00u | static_cast<uint16_t>(mant));
	}

	// Zero / subnormal float
	if (exp == 0u)
		return static_cast<uint16_t>(sign); // flush subnormals to zero

	// Normalized float: adjust exponent bias (127 -> 15)
	int32_t halfExp = static_cast<int32_t>(exp) - 127 + 15;
	if (halfExp >= 31)
	{
		// Overflow => Inf
		return static_cast<uint16_t>(sign | 0x7C00u);
	}
	if (halfExp <= 0)
	{
		// Underflow => subnormal half or zero
		if (halfExp < -10)
			return static_cast<uint16_t>(sign); // too small => 0
		// Add implicit leading 1
		mant |= 0x800000u;
		const uint32_t shift = static_cast<uint32_t>(1 - halfExp);
		// Round mantissa
		uint32_t mantRounded = mant >> (shift + 13u);
		const uint32_t rem = mant & ((1u << (shift + 13u)) - 1u);
		const uint32_t halfway = 1u << (shift + 12u);
		if (rem > halfway || (rem == halfway && (mantRounded & 1u)))
			++mantRounded;
		return static_cast<uint16_t>(sign | static_cast<uint16_t>(mantRounded));
	}

	// Normal half
	// Round mantissa from 23 bits to 10 bits
	uint32_t mantRounded = mant >> 13;
	const uint32_t rem = mant & 0x1FFFu;
	if (rem > 0x1000u || (rem == 0x1000u && (mantRounded & 1u)))
	{
		++mantRounded;
		if (mantRounded == 0x400u)
		{
			// mantissa overflow -> increment exponent
			mantRounded = 0u;
			++halfExp;
			if (halfExp >= 31)
				return static_cast<uint16_t>(sign | 0x7C00u);
		}
	}

	return static_cast<uint16_t>(sign | (static_cast<uint32_t>(halfExp) << 10) | (mantRounded & 0x3FFu));
}

inline float half_to_float(uint16_t h)
{
	const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
	uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1Fu;
	uint32_t mant = static_cast<uint32_t>(h) & 0x3FFu;

	uint32_t out;
	if (exp == 0u)
	{
		if (mant == 0u)
		{
			out = sign; // +/-0
		}
		else
		{
			// subnormal half -> normalize
			exp = 1u;
			while ((mant & 0x400u) == 0u)
			{
				mant <<= 1u;
				--exp;
			}
			mant &= 0x3FFu;
			const uint32_t fexp = (exp + (127u - 15u)) << 23;
			const uint32_t fmant = mant << 13;
			out = sign | fexp | fmant;
		}
	}
	else if (exp == 31u)
	{
		// Inf/NaN
		const uint32_t fexp = 0xFFu << 23;
		const uint32_t fmant = mant ? (mant << 13) : 0u;
		out = sign | fexp | fmant;
	}
	else
	{
		// Normal
		const uint32_t fexp = (exp + (127u - 15u)) << 23;
		const uint32_t fmant = mant << 13;
		out = sign | fexp | fmant;
	}

	union
	{
		uint32_t u;
		float f;
	} v;
	v.u = out;
	return v.f;
}

// === BF16 (bfloat16) helpers ===
//
// BF16 keeps the float exponent and truncates mantissa to 7 bits.
// Conversions are dependency-free and deterministic.
inline uint16_t float_to_bf16_rn(float f)
{
	union
	{
		float f;
		uint32_t u;
	} v;
	v.f = f;
	// Round-to-nearest-even on the truncated bits.
	// Add 0x7FFF + lsb of upper half before shifting.
	const uint32_t lsb = (v.u >> 16) & 1u;
	const uint32_t roundingBias = 0x7FFFu + lsb;
	return static_cast<uint16_t>((v.u + roundingBias) >> 16);
}

inline float bf16_to_float(uint16_t b)
{
	union
	{
		uint32_t u;
		float f;
	} v;
	v.u = static_cast<uint32_t>(b) << 16;
	return v.f;
}

// Low-precision weight dtype selector used by mixed-precision training helpers.
enum LowpDType
{
	LOWP_F16 = 1,
	LOWP_BF16 = 2
};

inline uint16_t float_to_lowp(float f, int lowpDType)
{
	return (lowpDType == LOWP_BF16) ? float_to_bf16_rn(f) : float_to_half_rn(f);
}

inline float lowp_to_float(uint16_t v, int lowpDType)
{
	return (lowpDType == LOWP_BF16) ? bf16_to_float(v) : half_to_float(v);
}

// Compiler hint for non-aliasing pointers (best-effort).
#if defined(__GNUC__) || defined(__clang__)
#define GLADES_RESTRICT __restrict__
#else
#define GLADES_RESTRICT
#endif

// === Positional encodings ===

// Build invDenomPair for sinusoidal PE:
// invDenomPair[ii] = 1 / 10000^(2*ii/dModel), length ceil(dModel/2).
inline void build_sinusoidal_inv_denom_pair(unsigned int dModel, std::vector<double>& invDenomPair)
{
	invDenomPair.clear();
	if (dModel == 0u)
		return;
	const unsigned int nPairs = (dModel + 1u) / 2u;
	invDenomPair.assign(static_cast<size_t>(nPairs), 0.0);
	for (unsigned int ii = 0; ii < nPairs; ++ii)
	{
		const double exponent = (2.0 * static_cast<double>(ii)) / static_cast<double>(dModel);
		invDenomPair[static_cast<size_t>(ii)] = pow(10000.0, -exponent);
	}
}

// Apply sinusoidal positional encoding for a single position in-place:
//   h[i] += PE[pos, i]
inline void add_sinusoidal_positional_encoding_inplace(float* h, unsigned int pos, unsigned int dModel)
{
	if (!h || dModel == 0u)
		return;
	for (unsigned int i = 0; i < dModel; ++i)
	{
		const unsigned int idx = i / 2u;
		const double exponent = (2.0 * static_cast<double>(idx)) / static_cast<double>(dModel);
		const double denom = pow(10000.0, exponent);
		const double angle = static_cast<double>(pos) / denom;
		const float pe = ((i % 2u) == 0u) ? static_cast<float>(sin(angle)) : static_cast<float>(cos(angle));
		h[i] += pe;
	}
}

// Overload using a precomputed invDenomPair (avoids pow() in inner loop).
inline void add_sinusoidal_positional_encoding_inplace(float* h,
                                                       unsigned int pos,
                                                       unsigned int dModel,
                                                       const std::vector<double>& invDenomPair)
{
	if (!h || dModel == 0u)
		return;
	const unsigned int needPairs = (dModel + 1u) / 2u;
	if (invDenomPair.size() < static_cast<size_t>(needPairs))
	{
		// Precondition violation: caller must provide a correctly-sized cache.
		GLADES_KERNEL_ASSERT(false && "add_sinusoidal_positional_encoding_inplace: invDenomPair cache too small");
		return;
	}
	for (unsigned int i = 0; i < dModel; ++i)
	{
		const unsigned int idx = i / 2u;
		const double angle = static_cast<double>(pos) * invDenomPair[static_cast<size_t>(idx)];
		const float pe = ((i % 2u) == 0u) ? static_cast<float>(sin(angle)) : static_cast<float>(cos(angle));
		h[i] += pe;
	}
}

inline void add_sinusoidal_positional_encoding_inplace(std::vector<float>& h, unsigned int pos, unsigned int dModel)
{
	if (h.size() < static_cast<size_t>(dModel))
		return;
	add_sinusoidal_positional_encoding_inplace(&h[0], pos, dModel);
}

inline void add_sinusoidal_positional_encoding_inplace(std::vector<float>& h,
                                                       unsigned int pos,
                                                       unsigned int dModel,
                                                       const std::vector<double>& invDenomPair)
{
	if (h.size() < static_cast<size_t>(dModel))
		return;
	add_sinusoidal_positional_encoding_inplace(&h[0], pos, dModel, invDenomPair);
}

// Apply sinusoidal positional encoding to a contiguous [T, dModel] buffer in-place.
inline void add_sinusoidal_positional_encoding_seq_inplace(float* h, unsigned int T, unsigned int dModel)
{
	if (!h || T == 0u || dModel == 0u)
		return;
	for (unsigned int t = 0; t < T; ++t)
	{
		float* ht = h + (static_cast<size_t>(t) * static_cast<size_t>(dModel));
		add_sinusoidal_positional_encoding_inplace(ht, t, dModel);
	}
}

inline void add_sinusoidal_positional_encoding_seq_inplace(float* h,
                                                           unsigned int T,
                                                           unsigned int dModel,
                                                           const std::vector<double>& invDenomPair)
{
	if (!h || T == 0u || dModel == 0u)
		return;
	for (unsigned int t = 0; t < T; ++t)
	{
		float* ht = h + (static_cast<size_t>(t) * static_cast<size_t>(dModel));
		add_sinusoidal_positional_encoding_inplace(ht, t, dModel, invDenomPair);
	}
}

// === Linear ===

// y[out] = b[out] + sum_in W[out,in] * x[in]
inline void linear_into(const float* x,
                        unsigned int inSize,
                        const std::vector<float>& W,
                        const std::vector<float>& b,
                        unsigned int outSize,
                        float* y)
{
	if (!x || !y)
		return;
	{
		const size_t need = static_cast<size_t>(outSize) * static_cast<size_t>(inSize);
		if (W.size() < need)
		{
			GLADES_KERNEL_ASSERT(false && "linear_into: W is smaller than outSize*inSize");
			return;
		}
	}
	for (unsigned int o = 0; o < outSize; ++o)
	{
		double acc = (o < b.size()) ? static_cast<double>(b[o]) : 0.0;
		const size_t wOff = static_cast<size_t>(o) * static_cast<size_t>(inSize);
		for (unsigned int i = 0; i < inSize; ++i)
			acc += static_cast<double>(W[wOff + i]) * static_cast<double>(x[i]);
		y[o] = static_cast<float>(acc);
	}
}

// Optimized matvec (float accumulation + optional AVX2).
// Intended for inference/training hot paths where throughput matters more than double-accum parity.
inline void linear_into_opt(const float* x,
                            unsigned int inSize,
                            const std::vector<float>& W,
                            const std::vector<float>& b,
                            unsigned int outSize,
                            float* y)
{
	if (!x || !y)
		return;
	{
		const size_t need = static_cast<size_t>(outSize) * static_cast<size_t>(inSize);
		if (W.size() < need)
		{
			GLADES_KERNEL_ASSERT(false && "linear_into_opt: W is smaller than outSize*inSize");
			return;
		}
	}
	for (unsigned int o = 0; o < outSize; ++o)
	{
		const float bias = (o < b.size()) ? b[o] : 0.0f;
		const float* wRow = &W[static_cast<size_t>(o) * static_cast<size_t>(inSize)];
		y[o] = bias + dot_f32(wRow, x, inSize);
	}
}

// Blocked GEMV specialization for row-major W:
//   y[o] = b[o] + dot(W[o, :], x)
//
// This is tuned for the token-LM tied head (vocab can be large, inSize=dModel).
// It is still dependency-free and deterministic, but significantly faster than the
// naive scalar implementation due to:
// - processing 4 output rows at once (reduces x loads / loop overhead)
// - unrolling the inner loop (encourages autovectorization)
inline void gemv_rowmajor_bias_block4_unroll8_into(const float* GLADES_RESTRICT x,
                                                  unsigned int inSize,
                                                  const float* GLADES_RESTRICT W,
                                                  unsigned int outSize,
                                                  const float* GLADES_RESTRICT b,
                                                  unsigned int bSize,
                                                  float* GLADES_RESTRICT y)
{
	if (!x || !W || !y)
		return;
	if (inSize == 0u || outSize == 0u)
		return;

	unsigned int o = 0u;
	for (; (o + 3u) < outSize; o += 4u)
	{
		const float* w0 = W + static_cast<size_t>(o + 0u) * static_cast<size_t>(inSize);
		const float* w1 = W + static_cast<size_t>(o + 1u) * static_cast<size_t>(inSize);
		const float* w2 = W + static_cast<size_t>(o + 2u) * static_cast<size_t>(inSize);
		const float* w3 = W + static_cast<size_t>(o + 3u) * static_cast<size_t>(inSize);

		float acc0 = (b && (o + 0u) < bSize) ? b[o + 0u] : 0.0f;
		float acc1 = (b && (o + 1u) < bSize) ? b[o + 1u] : 0.0f;
		float acc2 = (b && (o + 2u) < bSize) ? b[o + 2u] : 0.0f;
		float acc3 = (b && (o + 3u) < bSize) ? b[o + 3u] : 0.0f;

		unsigned int i = 0u;
		for (; (i + 7u) < inSize; i += 8u)
		{
			const float x0 = x[i + 0u];
			const float x1 = x[i + 1u];
			const float x2 = x[i + 2u];
			const float x3 = x[i + 3u];
			const float x4 = x[i + 4u];
			const float x5 = x[i + 5u];
			const float x6 = x[i + 6u];
			const float x7 = x[i + 7u];

			acc0 += w0[i + 0u] * x0 + w0[i + 1u] * x1 + w0[i + 2u] * x2 + w0[i + 3u] * x3 +
			        w0[i + 4u] * x4 + w0[i + 5u] * x5 + w0[i + 6u] * x6 + w0[i + 7u] * x7;
			acc1 += w1[i + 0u] * x0 + w1[i + 1u] * x1 + w1[i + 2u] * x2 + w1[i + 3u] * x3 +
			        w1[i + 4u] * x4 + w1[i + 5u] * x5 + w1[i + 6u] * x6 + w1[i + 7u] * x7;
			acc2 += w2[i + 0u] * x0 + w2[i + 1u] * x1 + w2[i + 2u] * x2 + w2[i + 3u] * x3 +
			        w2[i + 4u] * x4 + w2[i + 5u] * x5 + w2[i + 6u] * x6 + w2[i + 7u] * x7;
			acc3 += w3[i + 0u] * x0 + w3[i + 1u] * x1 + w3[i + 2u] * x2 + w3[i + 3u] * x3 +
			        w3[i + 4u] * x4 + w3[i + 5u] * x5 + w3[i + 6u] * x6 + w3[i + 7u] * x7;
		}
		for (; i < inSize; ++i)
		{
			const float xi = x[i];
			acc0 += w0[i] * xi;
			acc1 += w1[i] * xi;
			acc2 += w2[i] * xi;
			acc3 += w3[i] * xi;
		}

		y[o + 0u] = acc0;
		y[o + 1u] = acc1;
		y[o + 2u] = acc2;
		y[o + 3u] = acc3;
	}

	// Tail rows.
	for (; o < outSize; ++o)
	{
		const float* w = W + static_cast<size_t>(o) * static_cast<size_t>(inSize);
		float acc = (b && o < bSize) ? b[o] : 0.0f;
		unsigned int i = 0u;
		for (; (i + 7u) < inSize; i += 8u)
		{
			acc += w[i + 0u] * x[i + 0u];
			acc += w[i + 1u] * x[i + 1u];
			acc += w[i + 2u] * x[i + 2u];
			acc += w[i + 3u] * x[i + 3u];
			acc += w[i + 4u] * x[i + 4u];
			acc += w[i + 5u] * x[i + 5u];
			acc += w[i + 6u] * x[i + 6u];
			acc += w[i + 7u] * x[i + 7u];
		}
		for (; i < inSize; ++i)
			acc += w[i] * x[i];
		y[o] = acc;
	}
}

// Token-LM tied embedding head:
//   logits[v] = dot(h, tokE[v, :]) + lmBias[v]
inline void tied_embedding_logits_into(const float* h,
                                       unsigned int dModel,
                                       const std::vector<float>& tokE,
                                       const std::vector<float>& lmBias,
                                       unsigned int vocab,
                                       float* outLogits)
{
	if (!h || !outLogits)
		return;
	if (vocab == 0u || dModel == 0u)
		return;
	const size_t need = static_cast<size_t>(vocab) * static_cast<size_t>(dModel);
	if (tokE.size() < need)
	{
		GLADES_KERNEL_ASSERT(false && "tied_embedding_logits_into: tokE is smaller than vocab*dModel (model not initialized/corrupt)");
		return; // avoid OOB in release builds
	}

	gemv_rowmajor_bias_block4_unroll8_into(h, dModel, &tokE[0], vocab, lmBias.empty() ? NULL : &lmBias[0],
	                                      static_cast<unsigned int>(lmBias.size()), outLogits);
}

// Low-precision variant of the tied embedding head (tokE packed as uint16_t).
inline void tied_embedding_logits_into_lowp(const float* h,
                                            unsigned int dModel,
                                            const uint16_t* GLADES_RESTRICT tokE,
                                            int lowpDType,
                                            const std::vector<float>& lmBias,
                                            unsigned int vocab,
                                            float* outLogits)
{
	if (!h || !tokE || !outLogits)
		return;
	if (vocab == 0u || dModel == 0u)
		return;
	for (unsigned int v = 0; v < vocab; ++v)
	{
		const size_t eOff = static_cast<size_t>(v) * static_cast<size_t>(dModel);
		double acc = (v < lmBias.size()) ? static_cast<double>(lmBias[v]) : 0.0;
		for (unsigned int i = 0; i < dModel; ++i)
			acc += static_cast<double>(lowp_to_float(tokE[eOff + i], lowpDType)) * static_cast<double>(h[i]);
		outLogits[v] = static_cast<float>(acc);
	}
}

// Batched variant: logits[t, v] = dot(H[t], tokE[v]) + lmBias[v]
inline void tied_embedding_logits_forward_rows(const float* H,
                                               unsigned int T,
                                               unsigned int dModel,
                                               const std::vector<float>& tokE,
                                               const std::vector<float>& lmBias,
                                               unsigned int vocab,
                                               float* logitsOut)
{
	if (!H || !logitsOut)
		return;
	if (T == 0u || vocab == 0u || dModel == 0u)
		return;
	const size_t need = static_cast<size_t>(vocab) * static_cast<size_t>(dModel);
	if (tokE.size() < need)
	{
		GLADES_KERNEL_ASSERT(false && "tied_embedding_logits_forward_rows: tokE is smaller than vocab*dModel (model not initialized/corrupt)");
		return;
	}

	for (unsigned int t = 0; t < T; ++t)
	{
		const float* ht = H + static_cast<size_t>(t) * static_cast<size_t>(dModel);
		float* zt = logitsOut + static_cast<size_t>(t) * static_cast<size_t>(vocab);
		gemv_rowmajor_bias_block4_unroll8_into(ht, dModel, &tokE[0], vocab, lmBias.empty() ? NULL : &lmBias[0],
		                                      static_cast<unsigned int>(lmBias.size()), zt);
	}
}

// Batched low-precision tied head.
inline void tied_embedding_logits_forward_rows_lowp(const float* H,
                                                    unsigned int T,
                                                    unsigned int dModel,
                                                    const uint16_t* GLADES_RESTRICT tokE,
                                                    int lowpDType,
                                                    const std::vector<float>& lmBias,
                                                    unsigned int vocab,
                                                    float* logitsOut)
{
	if (!H || !tokE || !logitsOut)
		return;
	if (T == 0u || vocab == 0u || dModel == 0u)
		return;
	for (unsigned int t = 0; t < T; ++t)
	{
		const float* ht = H + static_cast<size_t>(t) * static_cast<size_t>(dModel);
		float* zt = logitsOut + static_cast<size_t>(t) * static_cast<size_t>(vocab);
		tied_embedding_logits_into_lowp(ht, dModel, tokE, lowpDType, lmBias, vocab, zt);
	}
}

inline void linear_vec(const float* x,
                       unsigned int inSize,
                       const std::vector<float>& W,
                       const std::vector<float>& b,
                       unsigned int outSize,
                       std::vector<float>& y)
{
	if (outSize == 0u)
	{
		y.clear();
		return;
	}
	if (y.size() != static_cast<size_t>(outSize))
		y.resize(outSize);
	linear_into(x, inSize, W, b, outSize, &y[0]);
}

// Optimized forward helper (calls linear_into_opt per row).
inline void linear_forward_opt(const float* X,
                               unsigned int T,
                               unsigned int inSize,
                               const std::vector<float>& W,
                               const std::vector<float>& b,
                               unsigned int outSize,
                               float* Y)
{
	if (!X || !Y)
		return;
	if (T == 0u || inSize == 0u || outSize == 0u)
		return;
	for (unsigned int t = 0; t < T; ++t)
	{
		const float* xt = X + static_cast<size_t>(t) * static_cast<size_t>(inSize);
		float* yt = Y + static_cast<size_t>(t) * static_cast<size_t>(outSize);
		linear_into_opt(xt, inSize, W, b, outSize, yt);
	}
}

// Y[t,out] = b[out] + sum_in W[out,in] * X[t,in]
inline void linear_forward(const float* X,
                           unsigned int T,
                           unsigned int inSize,
                           const std::vector<float>& W,
                           const std::vector<float>& b,
                           unsigned int outSize,
                           float* Y)
{
	if (!X || !Y)
		return;
	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t xOff = static_cast<size_t>(t) * static_cast<size_t>(inSize);
		const size_t yOff = static_cast<size_t>(t) * static_cast<size_t>(outSize);
		for (unsigned int o = 0; o < outSize; ++o)
		{
			double acc = (o < b.size() ? static_cast<double>(b[o]) : 0.0);
			const size_t wOff = static_cast<size_t>(o) * static_cast<size_t>(inSize);
			for (unsigned int i = 0; i < inSize; ++i)
				acc += static_cast<double>(W[wOff + i]) * static_cast<double>(X[xOff + i]);
			Y[yOff + o] = static_cast<float>(acc);
		}
	}
}

// Low-precision linear forward:
// Y[t,out] = b[out] + sum_in W_lowp[out,in] * X[t,in]
inline void linear_forward_lowp(const float* X,
                                unsigned int T,
                                unsigned int inSize,
                                const uint16_t* GLADES_RESTRICT W,
                                int lowpDType,
                                const std::vector<float>& b,
                                unsigned int outSize,
                                float* Y)
{
	if (!X || !W || !Y)
		return;
	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t xOff = static_cast<size_t>(t) * static_cast<size_t>(inSize);
		const size_t yOff = static_cast<size_t>(t) * static_cast<size_t>(outSize);
		for (unsigned int o = 0; o < outSize; ++o)
		{
			double acc = (o < b.size() ? static_cast<double>(b[o]) : 0.0);
			const size_t wOff = static_cast<size_t>(o) * static_cast<size_t>(inSize);
			for (unsigned int i = 0; i < inSize; ++i)
				acc += static_cast<double>(lowp_to_float(W[wOff + i], lowpDType)) * static_cast<double>(X[xOff + i]);
			Y[yOff + o] = static_cast<float>(acc);
		}
	}
}

// === Normalization ===

inline void layernorm_into(const float* x,
                           unsigned int D,
                           const std::vector<float>& gamma,
                           const std::vector<float>& beta,
                           float eps,
                           float* y)
{
	if (!x || !y || D == 0u)
		return;
	double sum = 0.0;
	for (unsigned int i = 0; i < D; ++i)
		sum += static_cast<double>(x[i]);
	const double mean = sum / static_cast<double>(D);
	double var = 0.0;
	for (unsigned int i = 0; i < D; ++i)
	{
		const double d = static_cast<double>(x[i]) - mean;
		var += d * d;
	}
	var /= static_cast<double>(D);
	const double invStd = 1.0 / sqrt(var + static_cast<double>(eps));
	for (unsigned int i = 0; i < D; ++i)
	{
		const float xn = static_cast<float>((static_cast<double>(x[i]) - mean) * invStd);
		const float g = (i < gamma.size() ? gamma[i] : 1.0f);
		const float bb = (i < beta.size() ? beta[i] : 0.0f);
		y[i] = xn * g + bb;
	}
}

inline void layernorm_vec(const float* x,
                          unsigned int D,
                          const std::vector<float>& gamma,
                          const std::vector<float>& beta,
                          float eps,
                          std::vector<float>& y)
{
	if (D == 0u)
	{
		y.clear();
		return;
	}
	if (y.size() != static_cast<size_t>(D))
		y.resize(D);
	layernorm_into(x, D, gamma, beta, eps, &y[0]);
}

inline void rmsnorm_into(const float* x,
                         unsigned int D,
                         const std::vector<float>& gamma,
                         const std::vector<float>& beta,
                         float eps,
                         float* y)
{
	if (!x || !y || D == 0u)
		return;
	double sumsq = 0.0;
	for (unsigned int i = 0; i < D; ++i)
	{
		const double xd = static_cast<double>(x[i]);
		sumsq += xd * xd;
	}
	const double mean2 = sumsq / static_cast<double>(D);
	const double invRms = 1.0 / sqrt(mean2 + static_cast<double>(eps));
	for (unsigned int i = 0; i < D; ++i)
	{
		const float g = (i < gamma.size() ? gamma[i] : 1.0f);
		const float bb = (i < beta.size() ? beta[i] : 0.0f);
		y[i] = (x[i] * static_cast<float>(invRms)) * g + bb;
	}
}

inline void rmsnorm_vec(const float* x,
                        unsigned int D,
                        const std::vector<float>& gamma,
                        const std::vector<float>& beta,
                        float eps,
                        std::vector<float>& y)
{
	if (D == 0u)
	{
		y.clear();
		return;
	}
	if (y.size() != static_cast<size_t>(D))
		y.resize(D);
	rmsnorm_into(x, D, gamma, beta, eps, &y[0]);
}

// === RoPE ===

// Apply Rotary Positional Embeddings (RoPE) in-place to a contiguous [T, dHead] buffer.
// Only the first ropeDim (must be even) dimensions are rotated in pairs.
// invFreq must have length ropeDim/2 with invFreq[i] = theta^(-2i/ropeDim).
inline void rope_apply_inplace(float* buf,
                               unsigned int T,
                               unsigned int dHead,
                               unsigned int ropeDim,
                               const std::vector<double>& invFreq,
                               bool inverse)
{
	if (!buf)
		return;
	if (T == 0u || dHead == 0u || ropeDim < 2u)
		return;
	if ((ropeDim % 2u) != 0u)
		ropeDim -= 1u;
	if (ropeDim > dHead)
		ropeDim = dHead - (dHead % 2u);
	if (invFreq.size() < static_cast<size_t>(ropeDim / 2u))
	{
		GLADES_KERNEL_ASSERT(false && "rope_apply_inplace: invFreq cache too small");
		return;
	}

	for (unsigned int tpos = 0; tpos < T; ++tpos)
	{
		const size_t base = static_cast<size_t>(tpos) * static_cast<size_t>(dHead);
		for (unsigned int j = 0; j < ropeDim; j += 2u)
		{
			const unsigned int ii = j / 2u;
			const double ang = static_cast<double>(tpos) * invFreq[static_cast<size_t>(ii)];
			const double c = cos(ang);
			double s = sin(ang);
			if (inverse)
				s = -s;
			const float x0 = buf[base + j];
			const float x1 = buf[base + j + 1u];
			buf[base + j] = static_cast<float>(static_cast<double>(x0) * c - static_cast<double>(x1) * s);
			buf[base + j + 1u] = static_cast<float>(static_cast<double>(x0) * s + static_cast<double>(x1) * c);
		}
	}
}

inline void rope_apply_vec(float* vec,
                           unsigned int dHead,
                           unsigned int ropeDim,
                           const std::vector<double>& invFreq,
                           unsigned int pos)
{
	// Single-position specialization: apply with T=1 but position != 0.
	// We implement directly to avoid constructing a fake [T,dHead] where tpos=pos.
	if (!vec)
		return;
	if (dHead == 0u || ropeDim < 2u)
		return;
	if ((ropeDim % 2u) != 0u)
		ropeDim -= 1u;
	if (ropeDim > dHead)
		ropeDim = dHead - (dHead % 2u);
	if (invFreq.size() < static_cast<size_t>(ropeDim / 2u))
	{
		GLADES_KERNEL_ASSERT(false && "rope_apply_vec: invFreq cache too small");
		return;
	}

	for (unsigned int j = 0; j < ropeDim; j += 2u)
	{
		const unsigned int ii = j / 2u;
		const double ang = static_cast<double>(pos) * invFreq[static_cast<size_t>(ii)];
		const double c = cos(ang);
		const double s = sin(ang);
		const float x0 = vec[j];
		const float x1 = vec[j + 1u];
		vec[j] = static_cast<float>(static_cast<double>(x0) * c - static_cast<double>(x1) * s);
		vec[j + 1u] = static_cast<float>(static_cast<double>(x0) * s + static_cast<double>(x1) * c);
	}
}

// Strided RoPE apply: rotate a [T, dHead] view where each timestep vector has stride `rowStride`.
// This allows applying RoPE in-place to packed Q/K buffers laid out as [T, dModel] (stride=dModel)
// or [T, dModelKV] (stride=dModelKV) without gathering into contiguous temporaries.
inline void rope_apply_inplace_strided(float* buf,
                                      unsigned int T,
                                      unsigned int rowStride,
                                      unsigned int dHead,
                                      unsigned int ropeDim,
                                      const std::vector<double>& invFreq,
                                      bool inverse)
{
	if (!buf)
		return;
	if (T == 0u || dHead == 0u || ropeDim < 2u || rowStride == 0u)
		return;
	if ((ropeDim % 2u) != 0u)
		ropeDim -= 1u;
	if (ropeDim > dHead)
		ropeDim = dHead - (dHead % 2u);
	if (invFreq.size() < static_cast<size_t>(ropeDim / 2u))
	{
		GLADES_KERNEL_ASSERT(false && "rope_apply_inplace_strided: invFreq cache too small");
		return;
	}

	for (unsigned int tpos = 0; tpos < T; ++tpos)
	{
		float* vec = buf + static_cast<size_t>(tpos) * static_cast<size_t>(rowStride);
		for (unsigned int j = 0; j < ropeDim; j += 2u)
		{
			const unsigned int ii = j / 2u;
			const double ang = static_cast<double>(tpos) * invFreq[static_cast<size_t>(ii)];
			const double c = cos(ang);
			double s = sin(ang);
			if (inverse)
				s = -s;
			const float x0 = vec[j];
			const float x1 = vec[j + 1u];
			vec[j] = static_cast<float>(static_cast<double>(x0) * c - static_cast<double>(x1) * s);
			vec[j + 1u] = static_cast<float>(static_cast<double>(x0) * s + static_cast<double>(x1) * c);
		}
	}
}

// === Softmax ===

inline void softmax_stable_inplace(std::vector<float>& scores)
{
	if (scores.empty())
		return;
	float maxv = scores[0];
	for (size_t i = 1; i < scores.size(); ++i)
		if (scores[i] > maxv) maxv = scores[i];
	double sum = 0.0;
	for (size_t i = 0; i < scores.size(); ++i)
	{
		const double e = exp(static_cast<double>(scores[i] - maxv));
		scores[i] = static_cast<float>(e);
		sum += e;
	}
	if (sum <= 0.0)
	{
		const float inv = 1.0f / static_cast<float>(scores.size());
		for (size_t i = 0; i < scores.size(); ++i)
			scores[i] = inv;
		return;
	}
	const float inv = static_cast<float>(1.0 / sum);
	for (size_t i = 0; i < scores.size(); ++i)
		scores[i] *= inv;
}

// Stable softmax on a raw buffer in-place.
// Operates on the first `n` entries of `scores`.
inline void softmax_stable_inplace(float* scores, size_t n)
{
	if (!scores || n == 0u)
		return;
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
}

inline void softmax_stable(const std::vector<float>& logits, std::vector<float>& probs)
{
	probs.assign(logits.size(), 0.0f);
	if (logits.empty())
		return;
	float maxv = logits[0];
	for (size_t i = 1; i < logits.size(); ++i)
		if (logits[i] > maxv)
			maxv = logits[i];
	double sum = 0.0;
	for (size_t i = 0; i < logits.size(); ++i)
	{
		const double e = exp(static_cast<double>(logits[i] - maxv));
		probs[i] = static_cast<float>(e);
		sum += e;
	}
	if (sum <= 0.0)
	{
		const float u = 1.0f / static_cast<float>(logits.size());
		for (size_t i = 0; i < logits.size(); ++i)
			probs[i] = u;
		return;
	}
	const float inv = static_cast<float>(1.0 / sum);
	for (size_t i = 0; i < probs.size(); ++i)
		probs[i] *= inv;
}

// Stable softmax: probsOut[i] = softmax(logits)[i]
// No allocations. Safe to call with probsOut == logits (in-place).
inline void softmax_stable_into(const float* logits, size_t n, float* probsOut)
{
	if (!logits || !probsOut || n == 0u)
		return;
	float maxv = logits[0];
	for (size_t i = 1; i < n; ++i)
		if (logits[i] > maxv) maxv = logits[i];
	double sum = 0.0;
	for (size_t i = 0; i < n; ++i)
	{
		const double e = exp(static_cast<double>(logits[i] - maxv));
		probsOut[i] = static_cast<float>(e);
		sum += e;
	}
	if (sum <= 0.0)
	{
		const float inv = 1.0f / static_cast<float>(n);
		for (size_t i = 0; i < n; ++i)
			probsOut[i] = inv;
		return;
	}
	const float inv = static_cast<float>(1.0 / sum);
	for (size_t i = 0; i < n; ++i)
		probsOut[i] *= inv;
}

// === Normalization backward kernels (training) ===
//
// These are kept in this shared header so training and inference use the same
// math conventions and so future optimized implementations can share the same API.

// LayerNorm forward for each timestep independently:
//   y = ((x - mean) * invStd) * gamma + beta
// Where mean/invStd are computed per row and returned to the caller.
inline void layernorm_forward_rows(const float* X,
                                  unsigned int T,
                                  unsigned int D,
                                  const std::vector<float>& gamma,
                                  const std::vector<float>& beta,
                                  float eps,
                                  float* Y,
                                  float* meanOut,
                                  float* invStdOut)
{
	if (!X || !Y || !meanOut || !invStdOut || T == 0u || D == 0u)
		return;
	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t off = static_cast<size_t>(t) * static_cast<size_t>(D);
		double sum = 0.0;
		for (unsigned int i = 0; i < D; ++i)
			sum += static_cast<double>(X[off + i]);
		const double mean = sum / static_cast<double>(D);
		double var = 0.0;
		for (unsigned int i = 0; i < D; ++i)
		{
			const double d = static_cast<double>(X[off + i]) - mean;
			var += d * d;
		}
		var /= static_cast<double>(D);
		const double invStd = 1.0 / sqrt(var + static_cast<double>(eps));
		meanOut[t] = static_cast<float>(mean);
		invStdOut[t] = static_cast<float>(invStd);
		for (unsigned int i = 0; i < D; ++i)
		{
			const float xn = static_cast<float>((static_cast<double>(X[off + i]) - mean) * invStd);
			const float g = (i < gamma.size() ? gamma[i] : 1.0f);
			const float b = (i < beta.size() ? beta[i] : 0.0f);
			Y[off + i] = xn * g + b;
		}
	}
}

// LayerNorm backward per timestep.
// Accumulates gGamma/gBeta across rows.
inline void layernorm_backward_rows_accum(const float* X,
                                         const float* dY,
                                         unsigned int T,
                                         unsigned int D,
                                         const std::vector<float>& gamma,
                                         const float* mean,
                                         const float* invStd,
                                         float* dX,
                                         std::vector<float>& gGamma,
                                         std::vector<float>& gBeta)
{
	if (!X || !dY || !mean || !invStd || !dX || T == 0u || D == 0u)
		return;
	if (gGamma.size() != D) gGamma.assign(D, 0.0f);
	if (gBeta.size() != D) gBeta.assign(D, 0.0f);

	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t off = static_cast<size_t>(t) * static_cast<size_t>(D);
		const float m = mean[t];
		const float is = invStd[t];

		// dxn = dY * gamma
		double sum_dxn = 0.0;
		double sum_dxn_xmu = 0.0;
		for (unsigned int i = 0; i < D; ++i)
		{
			const float g = (i < gamma.size() ? gamma[i] : 1.0f);
			const float xmu = X[off + i] - m;
			const float xn = xmu * is;
			const float dy = dY[off + i];
			gGamma[i] += dy * xn;
			gBeta[i] += dy;

			const float dxn = dy * g;
			sum_dxn += static_cast<double>(dxn);
			sum_dxn_xmu += static_cast<double>(dxn) * static_cast<double>(xmu);
		}

		// dX = (1/D) * invStd * (D*dxn - sum(dxn) - xmu*invStd^2*sum(dxn*xmu))
		const double invD = 1.0 / static_cast<double>(D);
		const double isd = static_cast<double>(is);
		const double is2 = isd * isd;
		for (unsigned int i = 0; i < D; ++i)
		{
			const double xmu = static_cast<double>(X[off + i]) - static_cast<double>(m);
			const float g = (i < gamma.size() ? gamma[i] : 1.0f);
			const float dxn = dY[off + i] * g;
			const double v = invD * isd *
			                 (static_cast<double>(D) * static_cast<double>(dxn) - sum_dxn - xmu * is2 * sum_dxn_xmu);
			dX[off + i] = static_cast<float>(v);
		}
	}
}

// RMSNorm forward (per timestep):
//   y = (x * invRms) * gamma + beta
// invRms = 1 / sqrt(mean(x^2) + eps)
inline void rmsnorm_forward_rows(const float* X,
                                unsigned int T,
                                unsigned int D,
                                const std::vector<float>& gamma,
                                const std::vector<float>& beta,
                                float eps,
                                float* Y,
                                float* invRmsOut)
{
	if (!X || !Y || !invRmsOut || T == 0u || D == 0u)
		return;
	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t off = static_cast<size_t>(t) * static_cast<size_t>(D);
		double sumsq = 0.0;
		for (unsigned int i = 0; i < D; ++i)
		{
			const double xd = static_cast<double>(X[off + i]);
			sumsq += xd * xd;
		}
		const double mean2 = sumsq / static_cast<double>(D);
		const double invRms = 1.0 / sqrt(mean2 + static_cast<double>(eps));
		invRmsOut[t] = static_cast<float>(invRms);
		for (unsigned int i = 0; i < D; ++i)
		{
			const float g = (i < gamma.size() ? gamma[i] : 1.0f);
			const float b = (i < beta.size() ? beta[i] : 0.0f);
			Y[off + i] = (X[off + i] * static_cast<float>(invRms)) * g + b;
		}
	}
}

// RMSNorm backward (per timestep). Accumulates gGamma/gBeta.
inline void rmsnorm_backward_rows_accum(const float* X,
                                       const float* dY,
                                       unsigned int T,
                                       unsigned int D,
                                       const std::vector<float>& gamma,
                                       const float* invRms,
                                       float* dX,
                                       std::vector<float>& gGamma,
                                       std::vector<float>& gBeta)
{
	if (!X || !dY || !invRms || !dX || T == 0u || D == 0u)
		return;
	if (gGamma.size() != D) gGamma.assign(D, 0.0f);
	if (gBeta.size() != D) gBeta.assign(D, 0.0f);

	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t off = static_cast<size_t>(t) * static_cast<size_t>(D);
		const float inv = invRms[t];

		// dxhat = dY * gamma
		// sum = sum_i dxhat_i * x_i
		double sum_dxhat_x = 0.0;
		for (unsigned int i = 0; i < D; ++i)
		{
			const float g = (i < gamma.size() ? gamma[i] : 1.0f);
			const float dy = dY[off + i];
			gBeta[i] += dy;
			gGamma[i] += dy * (X[off + i] * inv);
			sum_dxhat_x += static_cast<double>(dy * g) * static_cast<double>(X[off + i]);
		}

		// dX_i = dxhat_i * inv - x_i * inv^3 / D * sum(dxhat * x)
		const double invd = static_cast<double>(inv);
		const double inv3 = invd * invd * invd;
		const double invD = 1.0 / static_cast<double>(D);
		for (unsigned int i = 0; i < D; ++i)
		{
			const float g = (i < gamma.size() ? gamma[i] : 1.0f);
			const double dxhat = static_cast<double>(dY[off + i] * g);
			const double xi = static_cast<double>(X[off + i]);
			const double v = dxhat * invd - xi * inv3 * invD * sum_dxhat_x;
			dX[off + i] = static_cast<float>(v);
		}
	}
}

} // namespace transformer_kernels
} // namespace glades

