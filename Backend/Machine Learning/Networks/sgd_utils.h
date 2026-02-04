// Shared SGD utilities for NNetwork training steps.
// Kept header-only (inline) to avoid ODR issues across translation units.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace glades {
namespace sgd_detail {

inline bool is_finite(float v)
{
#if defined(__cplusplus) && __cplusplus >= 201103L
	return std::isfinite(v);
#else
	// C++98 fallback: reject NaN and infinities.
	const float inf = std::numeric_limits<float>::infinity();
	return (v == v) && (v != inf) && (v != -inf);
#endif
}

inline bool is_finite_double(double v)
{
#if defined(__cplusplus) && __cplusplus >= 201103L
	return std::isfinite(v);
#else
	const double inf = std::numeric_limits<double>::infinity();
	return (v == v) && (v != inf) && (v != -inf);
#endif
}

// Check a vector for non-finite values using a bounded sampling strategy.
// This is intended for potentially huge input rows (e.g., images).
inline bool vector_all_finite_bounded(const std::vector<float>& v, unsigned int maxChecks = 16u)
{
	if (v.empty())
		return true;
	if (maxChecks == 0u)
		maxChecks = 1u;
	const unsigned int n = static_cast<unsigned int>(v.size());
	const unsigned int checks = (n < maxChecks ? n : maxChecks);
	const unsigned int denom = (checks > 1u ? (checks - 1u) : 1u);
	for (unsigned int k = 0; k < checks; ++k)
	{
		const unsigned int idx = (n == 1u) ? 0u
		                                   : static_cast<unsigned int>((static_cast<unsigned long long>(k) * static_cast<unsigned long long>(n - 1u)) / denom);
		if (!is_finite(v[idx]))
			return false;
	}
	return true;
}

// Overload for shmea::GVector<float> (API-compatible container used throughout the engine).
template <typename GVectorLike>
inline bool gvector_all_finite_bounded(const GVectorLike& v, unsigned int maxChecks = 16u)
{
	if (v.size() == 0)
		return true;
	if (maxChecks == 0u)
		maxChecks = 1u;
	const unsigned int n = static_cast<unsigned int>(v.size());
	const unsigned int checks = (n < maxChecks ? n : maxChecks);
	const unsigned int denom = (checks > 1u ? (checks - 1u) : 1u);
	for (unsigned int k = 0; k < checks; ++k)
	{
		const unsigned int idx = (n == 1u) ? 0u
		                                   : static_cast<unsigned int>((static_cast<unsigned long long>(k) * static_cast<unsigned long long>(n - 1u)) / denom);
		if (!is_finite(v[idx]))
			return false;
	}
	return true;
}

// Pointer/span variant (for DataInput::*View APIs).
inline bool span_all_finite_bounded(const float* data, unsigned int size, unsigned int maxChecks = 16u)
{
	if (!data || size == 0u)
		return true;
	if (maxChecks == 0u)
		maxChecks = 1u;
	const unsigned int n = size;
	const unsigned int checks = (n < maxChecks ? n : maxChecks);
	const unsigned int denom = (checks > 1u ? (checks - 1u) : 1u);
	for (unsigned int k = 0; k < checks; ++k)
	{
		const unsigned int idx = (n == 1u) ? 0u
		                                   : static_cast<unsigned int>((static_cast<unsigned long long>(k) * static_cast<unsigned long long>(n - 1u)) / denom);
		if (!is_finite(data[idx]))
			return false;
	}
	return true;
}

inline float clipf(float v, float limit)
{
	if (v > limit) return limit;
	if (v < -limit) return -limit;
	return v;
}

// Conditional clip: if limit <= 0, clipping is disabled.
inline float clipf_maybe(float v, float limit)
{
	if (limit <= 0.0f)
		return v;
	return clipf(v, limit);
}

inline float clipf_range(float v, float lo, float hi)
{
	if (v < lo) return lo;
	if (v > hi) return hi;
	return v;
}

inline float clamp_prob01(float p)
{
	// Keep probabilities away from {0,1} to avoid log(0) / division blowups.
	const float eps = 1e-7f;
	if (p < eps) return eps;
	if (p > 1.0f - eps) return 1.0f - eps;
	return p;
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
	for (size_t i = 0; i < logits.size(); ++i)
		probs[i] *= inv;
}

} // namespace sgd_detail
} // namespace glades

