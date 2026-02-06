// Minimal Transformer math ops (forward/backward) used by Glades ML.
//
// These are intentionally small, dependency-free kernels implemented in terms of
// contiguous row-major buffers so they can be reused by the training loop and unit tests.
//
// Conventions:
// - Matrices are flattened row-major.
// - Sequence tensors are shaped [T, D] and flattened as t*D + d.
// - Attention probabilities are shaped [T, T] and flattened as t*T + u.
//
// NOTE: This is not a general-purpose tensor library; it is a compact set of
// primitives sufficient for Transformer encoder/decoder blocks in this codebase.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace glades {
namespace transformer_ops {

inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }
inline float relu_deriv_from_y(float y) { return (y > 0.0f) ? 1.0f : 0.0f; }

// SiLU / Swish: x * sigmoid(x)
inline float sigmoid(float x)
{
	// Stable enough for typical float ranges used in this codebase.
	return 1.0f / (1.0f + static_cast<float>(exp(-static_cast<double>(x))));
}

inline float silu(float x)
{
	const float s = sigmoid(x);
	return x * s;
}

inline float silu_deriv(float x)
{
	// d/dx (x*sigmoid(x)) = sigmoid(x) * (1 + x*(1-sigmoid(x)))
	const float s = sigmoid(x);
	return s * (1.0f + x * (1.0f - s));
}

// GELU (approx) and derivative.
//
// tanh approximation (Hendrycks & Gimpel):
//   gelu(x) ≈ 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
//
// Derivative of the above approximation:
//   let u = sqrt(2/pi) * (x + 0.044715*x^3)
//   gelu'(x) = 0.5*(1 + tanh(u)) + 0.5*x*(1 - tanh(u)^2)*du/dx
//   du/dx = sqrt(2/pi) * (1 + 3*0.044715*x^2)
inline float gelu(float x)
{
	const double xd = static_cast<double>(x);
	const double c = 0.79788456080286535588; // sqrt(2/pi)
	const double u = c * (xd + 0.044715 * xd * xd * xd);
	const double t = tanh(u);
	return static_cast<float>(0.5 * xd * (1.0 + t));
}

inline float gelu_deriv(float x)
{
	const double xd = static_cast<double>(x);
	const double c = 0.79788456080286535588; // sqrt(2/pi)
	const double x2 = xd * xd;
	const double u = c * (xd + 0.044715 * xd * x2);
	const double t = tanh(u);
	const double sech2 = 1.0 - (t * t);
	const double du = c * (1.0 + 3.0 * 0.044715 * x2);
	const double g = 0.5 * (1.0 + t) + 0.5 * xd * sech2 * du;
	return static_cast<float>(g);
}

inline void softmax_masked_row_stable(const float* scoresRow,
                                      unsigned int T,
                                      unsigned int rowT,
                                      bool causal,
                                      std::vector<float>& probsRowOut)
{
	probsRowOut.assign(T, 0.0f);
	if (T == 0u)
		return;

	// Determine allowed range.
	const unsigned int maxU = causal ? rowT : (T - 1u);

	// max over allowed
	float maxv = scoresRow[0];
	bool maxInit = false;
	for (unsigned int u = 0; u < T; ++u)
	{
		if (causal && u > maxU)
			continue;
		if (!maxInit)
		{
			maxv = scoresRow[u];
			maxInit = true;
		}
		else if (scoresRow[u] > maxv)
			maxv = scoresRow[u];
	}
	if (!maxInit)
	{
		// Nothing allowed (shouldn't happen), return zeros.
		return;
	}

	double sum = 0.0;
	for (unsigned int u = 0; u < T; ++u)
	{
		if (causal && u > maxU)
		{
			probsRowOut[u] = 0.0f;
			continue;
		}
		const double e = exp(static_cast<double>(scoresRow[u] - maxv));
		probsRowOut[u] = static_cast<float>(e);
		sum += e;
	}
	if (sum <= 0.0)
	{
		// Uniform over allowed.
		const float inv = 1.0f / static_cast<float>(maxU + 1u);
		for (unsigned int u = 0; u <= maxU; ++u)
			probsRowOut[u] = inv;
		return;
	}
	const float inv = static_cast<float>(1.0 / sum);
	for (unsigned int u = 0; u < T; ++u)
		probsRowOut[u] *= inv;
}

// Stable softmax for a single attention row with:
// - optional causal masking (u > rowT disallowed)
// - optional key mask (keyAllowed[u] == 0 disallowed)
//
// If there are no allowed keys for this row, this returns all zeros (not uniform).
inline void softmax_masked_row_stable_keymask(const float* scoresRow,
                                              unsigned int T,
                                              unsigned int rowT,
                                              bool causal,
                                              const unsigned char* keyAllowed,
                                              std::vector<float>& probsRowOut)
{
	if (!keyAllowed)
	{
		softmax_masked_row_stable(scoresRow, T, rowT, causal, probsRowOut);
		return;
	}

	probsRowOut.assign(T, 0.0f);
	if (T == 0u)
		return;

	const unsigned int maxU = causal ? rowT : (T - 1u);

	// max over allowed
	float maxv = 0.0f;
	bool maxInit = false;
	for (unsigned int u = 0; u < T; ++u)
	{
		if (causal && u > maxU)
			continue;
		if (keyAllowed[u] == 0u)
			continue;
		if (!maxInit)
		{
			maxv = scoresRow[u];
			maxInit = true;
		}
		else if (scoresRow[u] > maxv)
			maxv = scoresRow[u];
	}
	if (!maxInit)
	{
		// Nothing allowed.
		return;
	}

	double sum = 0.0;
	unsigned int allowedCount = 0u;
	for (unsigned int u = 0; u < T; ++u)
	{
		if (causal && u > maxU)
		{
			probsRowOut[u] = 0.0f;
			continue;
		}
		if (keyAllowed[u] == 0u)
		{
			probsRowOut[u] = 0.0f;
			continue;
		}
		const double e = exp(static_cast<double>(scoresRow[u] - maxv));
		probsRowOut[u] = static_cast<float>(e);
		sum += e;
		++allowedCount;
	}
	if (sum <= 0.0 || allowedCount == 0u)
	{
		// Uniform over allowed.
		const float inv = (allowedCount > 0u) ? (1.0f / static_cast<float>(allowedCount)) : 0.0f;
		for (unsigned int u = 0; u < T; ++u)
		{
			if (causal && u > maxU)
				continue;
			if (keyAllowed[u] == 0u)
				continue;
			probsRowOut[u] = inv;
		}
		return;
	}
	const float inv = static_cast<float>(1.0 / sum);
	for (unsigned int u = 0; u < T; ++u)
		probsRowOut[u] *= inv;
}

// Scaled dot-product attention for a single head.
//
// Inputs:
// - Q: [T, dK]
// - K: [T, dK]
// - V: [T, dV] (typically dV == dK)
//
// Outputs:
// - O: [T, dV]
// - probs (optional): [T, T]
inline void scaled_dot_product_attention_forward(const float* Q,
                                                 const float* K,
                                                 const float* V,
                                                 unsigned int T,
                                                 unsigned int dK,
                                                 unsigned int dV,
                                                 bool causal,
                                                 std::vector<float>& O,
                                                 std::vector<float>* probsCache)
{
	O.assign(static_cast<size_t>(T) * static_cast<size_t>(dV), 0.0f);
	if (probsCache)
		probsCache->assign(static_cast<size_t>(T) * static_cast<size_t>(T), 0.0f);
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	std::vector<float> scores(T, 0.0f);
	std::vector<float> probsRow;
	for (unsigned int t = 0; t < T; ++t)
	{
		// scores[u] = dot(Q[t], K[u]) * invSqrt
		for (unsigned int u = 0; u < T; ++u)
		{
			double dot = 0.0;
			const size_t qOff = static_cast<size_t>(t) * static_cast<size_t>(dK);
			const size_t kOff = static_cast<size_t>(u) * static_cast<size_t>(dK);
			for (unsigned int k = 0; k < dK; ++k)
				dot += static_cast<double>(Q[qOff + k]) * static_cast<double>(K[kOff + k]);
			scores[u] = static_cast<float>(dot) * invSqrt;
		}

		softmax_masked_row_stable(scores.data(), T, t, causal, probsRow);
		if (probsCache)
		{
			const size_t pOff = static_cast<size_t>(t) * static_cast<size_t>(T);
			for (unsigned int u = 0; u < T; ++u)
				(*probsCache)[pOff + u] = probsRow[u];
		}

		// O[t] = sum_u probs[t,u] * V[u]
		const size_t oOff = static_cast<size_t>(t) * static_cast<size_t>(dV);
		for (unsigned int u = 0; u < T; ++u)
		{
			const float p = probsRow[u];
			if (p == 0.0f)
				continue;
			const size_t vOff = static_cast<size_t>(u) * static_cast<size_t>(dV);
			for (unsigned int dv = 0; dv < dV; ++dv)
				O[oOff + dv] += p * V[vOff + dv];
		}
	}
}

// Overload with key mask (keyAllowed[u]==0 excludes the key/value at timestep u).
inline void scaled_dot_product_attention_forward(const float* Q,
                                                 const float* K,
                                                 const float* V,
                                                 unsigned int T,
                                                 unsigned int dK,
                                                 unsigned int dV,
                                                 bool causal,
                                                 std::vector<float>& O,
                                                 std::vector<float>* probsCache,
                                                 const unsigned char* keyAllowed)
{
	O.assign(static_cast<size_t>(T) * static_cast<size_t>(dV), 0.0f);
	if (probsCache)
		probsCache->assign(static_cast<size_t>(T) * static_cast<size_t>(T), 0.0f);
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	std::vector<float> scores(T, 0.0f);
	std::vector<float> probsRow;
	for (unsigned int t = 0; t < T; ++t)
	{
		for (unsigned int u = 0; u < T; ++u)
		{
			double dot = 0.0;
			const size_t qOff = static_cast<size_t>(t) * static_cast<size_t>(dK);
			const size_t kOff = static_cast<size_t>(u) * static_cast<size_t>(dK);
			for (unsigned int k = 0; k < dK; ++k)
				dot += static_cast<double>(Q[qOff + k]) * static_cast<double>(K[kOff + k]);
			scores[u] = static_cast<float>(dot) * invSqrt;
		}

		softmax_masked_row_stable_keymask(scores.data(), T, t, causal, keyAllowed, probsRow);
		if (probsCache)
		{
			const size_t pOff = static_cast<size_t>(t) * static_cast<size_t>(T);
			for (unsigned int u = 0; u < T; ++u)
				(*probsCache)[pOff + u] = probsRow[u];
		}

		const size_t oOff = static_cast<size_t>(t) * static_cast<size_t>(dV);
		for (unsigned int u = 0; u < T; ++u)
		{
			const float p = probsRow[u];
			if (p == 0.0f)
				continue;
			const size_t vOff = static_cast<size_t>(u) * static_cast<size_t>(dV);
			for (unsigned int dv = 0; dv < dV; ++dv)
				O[oOff + dv] += p * V[vOff + dv];
		}
	}
}

// Backward pass for scaled dot-product attention for a single head.
//
// Inputs:
// - Q,K,V as in forward
// - dO: upstream gradient [T, dV]
// - probs: attention probabilities [T, T] from forward
//
// Outputs (accumulated into):
// - dQ [T, dK]
// - dK [T, dK]
// - dV [T, dV]
inline void scaled_dot_product_attention_backward(const float* Q,
                                                  const float* K,
                                                  const float* V,
                                                  const float* dO,
                                                  const float* probs,
                                                  unsigned int T,
                                                  unsigned int dK,
                                                  unsigned int dV,
                                                  bool causal,
                                                  std::vector<float>& dQ,
                                                  std::vector<float>& dKOut,
                                                  std::vector<float>& dVOut)
{
	dQ.assign(static_cast<size_t>(T) * static_cast<size_t>(dK), 0.0f);
	dKOut.assign(static_cast<size_t>(T) * static_cast<size_t>(dK), 0.0f);
	dVOut.assign(static_cast<size_t>(T) * static_cast<size_t>(dV), 0.0f);
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	// dV[u] += sum_t probs[t,u] * dO[t]
	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t pOff = static_cast<size_t>(t) * static_cast<size_t>(T);
		const size_t dOOff = static_cast<size_t>(t) * static_cast<size_t>(dV);
		for (unsigned int u = 0; u < T; ++u)
		{
			const float p = probs[pOff + u];
			if (p == 0.0f)
				continue;
			const size_t dVOff = static_cast<size_t>(u) * static_cast<size_t>(dV);
			for (unsigned int dv = 0; dv < dV; ++dv)
				dVOut[dVOff + dv] += p * dO[dOOff + dv];
		}
	}

	// dProbs[t,u] = dot(dO[t], V[u])
	std::vector<float> dProbs(static_cast<size_t>(T) * static_cast<size_t>(T), 0.0f);
	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t dOOff = static_cast<size_t>(t) * static_cast<size_t>(dV);
		const size_t dpOff = static_cast<size_t>(t) * static_cast<size_t>(T);
		for (unsigned int u = 0; u < T; ++u)
		{
			if (causal && u > t)
				continue;
			const size_t vOff = static_cast<size_t>(u) * static_cast<size_t>(dV);
			double dot = 0.0;
			for (unsigned int dv = 0; dv < dV; ++dv)
				dot += static_cast<double>(dO[dOOff + dv]) * static_cast<double>(V[vOff + dv]);
			dProbs[dpOff + u] = static_cast<float>(dot);
		}
	}

	// dScores via softmax Jacobian per row:
	// dS_u = p_u * (dP_u - sum_j p_j * dP_j)
	std::vector<float> dScores(static_cast<size_t>(T) * static_cast<size_t>(T), 0.0f);
	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t pOff = static_cast<size_t>(t) * static_cast<size_t>(T);
		double rowDot = 0.0;
		const unsigned int maxU = causal ? t : (T - 1u);
		for (unsigned int u = 0; u <= maxU; ++u)
			rowDot += static_cast<double>(probs[pOff + u]) * static_cast<double>(dProbs[pOff + u]);
		for (unsigned int u = 0; u < T; ++u)
		{
			if (causal && u > maxU)
			{
				dScores[pOff + u] = 0.0f;
				continue;
			}
			const float p = probs[pOff + u];
			dScores[pOff + u] = p * (dProbs[pOff + u] - static_cast<float>(rowDot));
		}
	}

	// dQ and dK from scores = Q K^T / sqrt(dK)
	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t qOff = static_cast<size_t>(t) * static_cast<size_t>(dK);
		const size_t dsOff = static_cast<size_t>(t) * static_cast<size_t>(T);
		for (unsigned int u = 0; u < T; ++u)
		{
			if (causal && u > t)
				continue;
			const float ds = dScores[dsOff + u] * invSqrt;
			if (ds == 0.0f)
				continue;
			const size_t kOff = static_cast<size_t>(u) * static_cast<size_t>(dK);
			for (unsigned int k = 0; k < dK; ++k)
			{
				dQ[qOff + k] += ds * K[kOff + k];
				dKOut[kOff + k] += ds * Q[qOff + k];
			}
		}
	}
}

// Memory-efficient backward pass for scaled dot-product attention for a single head.
//
// This variant **does not require** the [T,T] probability matrix from forward and does not
// allocate any [T,T] intermediates. It recomputes the masked softmax per row and performs the
// backward pass row-wise using only O(T) extra memory.
//
// Inputs:
// - Q,K,V as in forward
// - dO: upstream gradient [T, dV]
//
// Outputs (written into):
// - dQ [T, dK]
// - dK [T, dK]
// - dV [T, dV]
inline void scaled_dot_product_attention_backward_recompute(const float* Q,
                                                            const float* K,
                                                            const float* V,
                                                            const float* dO,
                                                            unsigned int T,
                                                            unsigned int dK,
                                                            unsigned int dV,
                                                            bool causal,
                                                            std::vector<float>& dQ,
                                                            std::vector<float>& dKOut,
                                                            std::vector<float>& dVOut)
{
	dQ.assign(static_cast<size_t>(T) * static_cast<size_t>(dK), 0.0f);
	dKOut.assign(static_cast<size_t>(T) * static_cast<size_t>(dK), 0.0f);
	dVOut.assign(static_cast<size_t>(T) * static_cast<size_t>(dV), 0.0f);
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	std::vector<float> scores(T, 0.0f);
	std::vector<float> probsRow;
	std::vector<float> dProbsRow(T, 0.0f);

	for (unsigned int t = 0; t < T; ++t)
	{
		// scores[u] = dot(Q[t], K[u]) * invSqrt
		const size_t qOff = static_cast<size_t>(t) * static_cast<size_t>(dK);
		for (unsigned int u = 0; u < T; ++u)
		{
			double dot = 0.0;
			const size_t kOff = static_cast<size_t>(u) * static_cast<size_t>(dK);
			for (unsigned int k = 0; k < dK; ++k)
				dot += static_cast<double>(Q[qOff + k]) * static_cast<double>(K[kOff + k]);
			scores[u] = static_cast<float>(dot) * invSqrt;
		}

		softmax_masked_row_stable(scores.data(), T, t, causal, probsRow);

		// dV[u] += probs[t,u] * dO[t]
		const size_t dOOff = static_cast<size_t>(t) * static_cast<size_t>(dV);
		for (unsigned int u = 0; u < T; ++u)
		{
			const float p = probsRow[u];
			if (p == 0.0f)
				continue;
			const size_t dVOff = static_cast<size_t>(u) * static_cast<size_t>(dV);
			for (unsigned int dv = 0; dv < dV; ++dv)
				dVOut[dVOff + dv] += p * dO[dOOff + dv];
		}

		// dProbsRow[u] = dot(dO[t], V[u])
		const unsigned int maxU = causal ? t : (T - 1u);
		for (unsigned int u = 0; u < T; ++u)
		{
			if (causal && u > maxU)
			{
				dProbsRow[u] = 0.0f;
				continue;
			}
			const size_t vOff = static_cast<size_t>(u) * static_cast<size_t>(dV);
			double dot = 0.0;
			for (unsigned int dv = 0; dv < dV; ++dv)
				dot += static_cast<double>(dO[dOOff + dv]) * static_cast<double>(V[vOff + dv]);
			dProbsRow[u] = static_cast<float>(dot);
		}

		// rowDot = sum_u p_u * dP_u
		double rowDot = 0.0;
		for (unsigned int u = 0; u <= maxU; ++u)
			rowDot += static_cast<double>(probsRow[u]) * static_cast<double>(dProbsRow[u]);

		// dScores_u = p_u * (dP_u - rowDot)
		// dQ[t] and dK[u] from scores = Q K^T / sqrt(dK)
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			const float p = probsRow[u];
			if (p == 0.0f)
				continue;
			const float ds = (p * (dProbsRow[u] - static_cast<float>(rowDot))) * invSqrt;
			if (ds == 0.0f)
				continue;

			const size_t kOff = static_cast<size_t>(u) * static_cast<size_t>(dK);
			for (unsigned int k = 0; k < dK; ++k)
			{
				dQ[qOff + k] += ds * K[kOff + k];
				dKOut[kOff + k] += ds * Q[qOff + k];
			}
		}
	}
}

// Overload with key mask (keyAllowed[u]==0 excludes the key/value at timestep u).
inline void scaled_dot_product_attention_backward_recompute(const float* Q,
                                                            const float* K,
                                                            const float* V,
                                                            const float* dO,
                                                            unsigned int T,
                                                            unsigned int dK,
                                                            unsigned int dV,
                                                            bool causal,
                                                            std::vector<float>& dQ,
                                                            std::vector<float>& dKOut,
                                                            std::vector<float>& dVOut,
                                                            const unsigned char* keyAllowed)
{
	if (!keyAllowed)
	{
		scaled_dot_product_attention_backward_recompute(Q, K, V, dO, T, dK, dV, causal, dQ, dKOut, dVOut);
		return;
	}

	dQ.assign(static_cast<size_t>(T) * static_cast<size_t>(dK), 0.0f);
	dKOut.assign(static_cast<size_t>(T) * static_cast<size_t>(dK), 0.0f);
	dVOut.assign(static_cast<size_t>(T) * static_cast<size_t>(dV), 0.0f);
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	std::vector<float> scores(T, 0.0f);
	std::vector<float> probsRow;
	std::vector<float> dProbsRow(T, 0.0f);

	for (unsigned int t = 0; t < T; ++t)
	{
		const size_t qOff = static_cast<size_t>(t) * static_cast<size_t>(dK);
		for (unsigned int u = 0; u < T; ++u)
		{
			double dot = 0.0;
			const size_t kOff = static_cast<size_t>(u) * static_cast<size_t>(dK);
			for (unsigned int k = 0; k < dK; ++k)
				dot += static_cast<double>(Q[qOff + k]) * static_cast<double>(K[kOff + k]);
			scores[u] = static_cast<float>(dot) * invSqrt;
		}

		softmax_masked_row_stable_keymask(scores.data(), T, t, causal, keyAllowed, probsRow);

		// dV[u] += probs[t,u] * dO[t]
		const size_t dOOff = static_cast<size_t>(t) * static_cast<size_t>(dV);
		for (unsigned int u = 0; u < T; ++u)
		{
			if (keyAllowed[u] == 0u)
				continue;
			const float p = probsRow[u];
			if (p == 0.0f)
				continue;
			const size_t dVOff = static_cast<size_t>(u) * static_cast<size_t>(dV);
			for (unsigned int dv = 0; dv < dV; ++dv)
				dVOut[dVOff + dv] += p * dO[dOOff + dv];
		}

		// dProbsRow[u] = dot(dO[t], V[u])
		const unsigned int maxU = causal ? t : (T - 1u);
		for (unsigned int u = 0; u < T; ++u)
		{
			if (causal && u > maxU)
			{
				dProbsRow[u] = 0.0f;
				continue;
			}
			if (keyAllowed[u] == 0u)
			{
				dProbsRow[u] = 0.0f;
				continue;
			}
			const size_t vOff = static_cast<size_t>(u) * static_cast<size_t>(dV);
			double dot = 0.0;
			for (unsigned int dv = 0; dv < dV; ++dv)
				dot += static_cast<double>(dO[dOOff + dv]) * static_cast<double>(V[vOff + dv]);
			dProbsRow[u] = static_cast<float>(dot);
		}

		// rowDot = sum_u p_u * dP_u over allowed u
		double rowDot = 0.0;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed[u] == 0u)
				continue;
			rowDot += static_cast<double>(probsRow[u]) * static_cast<double>(dProbsRow[u]);
		}

		// dScores_u = p_u * (dP_u - rowDot)
		// dQ[t] and dK[u] from scores = Q K^T / sqrt(dK)
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed[u] == 0u)
				continue;
			const float p = probsRow[u];
			if (p == 0.0f)
				continue;
			const float ds = (p * (dProbsRow[u] - static_cast<float>(rowDot))) * invSqrt;
			if (ds == 0.0f)
				continue;

			const size_t kOff = static_cast<size_t>(u) * static_cast<size_t>(dK);
			for (unsigned int k = 0; k < dK; ++k)
			{
				dQ[qOff + k] += ds * K[kOff + k];
				dKOut[kOff + k] += ds * Q[qOff + k];
			}
		}
	}
}

// ============================
// Strided attention helpers
// ============================
//
// These variants operate on *strided* [T, D] views so callers can run multi-head attention
// directly on packed Q/K/V buffers (without gathering per-head contiguous temporaries).
//
// Conventions:
// - Q(t,k) = Qbase[t*qStride + k]
// - K(u,k) = Kbase[u*kStride + k]
// - V(u,d) = Vbase[u*vStride + d]
// - O(t,d) = Obase[t*oStride + d]
//
// NOTE:
// - These functions do not allocate per-head output buffers.
// - The backward variant **accumulates** into dQ/dK/dV.

inline void scaled_dot_product_attention_forward_strided(const float* Qbase,
                                                         unsigned int qStride,
                                                         const float* Kbase,
                                                         unsigned int kStride,
                                                         const float* Vbase,
                                                         unsigned int vStride,
                                                         unsigned int T,
                                                         unsigned int dK,
                                                         unsigned int dV,
                                                         bool causal,
                                                         float* Obase,
                                                         unsigned int oStride,
                                                         std::vector<float>& scoresScratch,
                                                         std::vector<float>& probsRowScratch)
{
	if (!Qbase || !Kbase || !Vbase || !Obase)
		return;
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	if (scoresScratch.size() < T)
		scoresScratch.assign(T, 0.0f);
	if (probsRowScratch.size() < T)
		probsRowScratch.assign(T, 0.0f);

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	for (unsigned int t = 0; t < T; ++t)
	{
		// scores[u] = dot(Q[t], K[u]) * invSqrt
		const float* qt = Qbase + static_cast<size_t>(t) * static_cast<size_t>(qStride);
		for (unsigned int u = 0; u < T; ++u)
		{
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			double dot = 0.0;
			for (unsigned int k = 0; k < dK; ++k)
				dot += static_cast<double>(qt[k]) * static_cast<double>(ku[k]);
			scoresScratch[u] = static_cast<float>(dot) * invSqrt;
		}

		softmax_masked_row_stable(scoresScratch.data(), T, t, causal, probsRowScratch);

		// O[t] = sum_u p[u] * V[u]
		float* ot = Obase + static_cast<size_t>(t) * static_cast<size_t>(oStride);
		for (unsigned int dv = 0; dv < dV; ++dv)
			ot[dv] = 0.0f;

		const unsigned int maxU = causal ? t : (T - 1u);
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			const float p = probsRowScratch[u];
			if (p == 0.0f)
				continue;
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			for (unsigned int dv = 0; dv < dV; ++dv)
				ot[dv] += p * vu[dv];
		}
	}
}

inline void scaled_dot_product_attention_forward_strided(const float* Qbase,
                                                         unsigned int qStride,
                                                         const float* Kbase,
                                                         unsigned int kStride,
                                                         const float* Vbase,
                                                         unsigned int vStride,
                                                         unsigned int T,
                                                         unsigned int dK,
                                                         unsigned int dV,
                                                         bool causal,
                                                         float* Obase,
                                                         unsigned int oStride,
                                                         std::vector<float>& scoresScratch,
                                                         std::vector<float>& probsRowScratch,
                                                         const unsigned char* keyAllowed)
{
	if (!keyAllowed)
	{
		scaled_dot_product_attention_forward_strided(Qbase, qStride, Kbase, kStride, Vbase, vStride, T, dK, dV, causal, Obase, oStride,
		                                            scoresScratch, probsRowScratch);
		return;
	}
	if (!Qbase || !Kbase || !Vbase || !Obase)
		return;
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	if (scoresScratch.size() < T)
		scoresScratch.assign(T, 0.0f);
	if (probsRowScratch.size() < T)
		probsRowScratch.assign(T, 0.0f);

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	for (unsigned int t = 0; t < T; ++t)
	{
		const float* qt = Qbase + static_cast<size_t>(t) * static_cast<size_t>(qStride);
		for (unsigned int u = 0; u < T; ++u)
		{
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			double dot = 0.0;
			for (unsigned int k = 0; k < dK; ++k)
				dot += static_cast<double>(qt[k]) * static_cast<double>(ku[k]);
			scoresScratch[u] = static_cast<float>(dot) * invSqrt;
		}

		softmax_masked_row_stable_keymask(scoresScratch.data(), T, t, causal, keyAllowed, probsRowScratch);

		float* ot = Obase + static_cast<size_t>(t) * static_cast<size_t>(oStride);
		for (unsigned int dv = 0; dv < dV; ++dv)
			ot[dv] = 0.0f;

		const unsigned int maxU = causal ? t : (T - 1u);
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed[u] == 0u)
				continue;
			const float p = probsRowScratch[u];
			if (p == 0.0f)
				continue;
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			for (unsigned int dv = 0; dv < dV; ++dv)
				ot[dv] += p * vu[dv];
		}
	}
}

// ============================
// FlashAttention-style strided attention (online softmax)
// ============================
//
// These variants are "FlashAttention-like" in the sense that they:
// - never materialize a [T,T] attention score/probability matrix
// - avoid even a per-row scores/probs scratch vector
// - compute softmax normalization online (max + running sum of exp)
//
// Memory: O(dV) per row (plus caller-owned Q/K/V/O)
// Time:   O(T^2 * dK) like the reference kernel (scalar CPU)
//
// If a row has zero allowed keys, output is set to zeros and gradients are zero.
inline void scaled_dot_product_attention_forward_flash_strided(const float* Qbase,
                                                               unsigned int qStride,
                                                               const float* Kbase,
                                                               unsigned int kStride,
                                                               const float* Vbase,
                                                               unsigned int vStride,
                                                               unsigned int T,
                                                               unsigned int dK,
                                                               unsigned int dV,
                                                               bool causal,
                                                               float* Obase,
                                                               unsigned int oStride,
                                                               const unsigned char* keyAllowed)
{
	if (!Qbase || !Kbase || !Vbase || !Obase)
		return;
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	for (unsigned int t = 0; t < T; ++t)
	{
		const float* qt = Qbase + static_cast<size_t>(t) * static_cast<size_t>(qStride);
		float* ot = Obase + static_cast<size_t>(t) * static_cast<size_t>(oStride);
		for (unsigned int dv = 0; dv < dV; ++dv)
			ot[dv] = 0.0f;

		const unsigned int maxU = causal ? t : (T - 1u);

		// Online softmax state.
		float m = -1e30f;
		double l = 0.0;
		bool any = false;

		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed && keyAllowed[u] == 0u)
				continue;
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);

			double dot = 0.0;
			for (unsigned int k = 0; k < dK; ++k)
				dot += static_cast<double>(qt[k]) * static_cast<double>(ku[k]);
			const float s = static_cast<float>(dot) * invSqrt;

			if (!any)
			{
				any = true;
				m = s;
				const double beta = 1.0; // exp(s - m) where m==s
				l = beta;
				const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
				for (unsigned int dv = 0; dv < dV; ++dv)
					ot[dv] = static_cast<float>(beta) * vu[dv];
				continue;
			}

			const float newM = (s > m) ? s : m;
			const double alpha = exp(static_cast<double>(m - newM));
			const double beta = exp(static_cast<double>(s - newM));
			l = l * alpha + beta;

			// Scale old accumulator + add new contribution.
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			for (unsigned int dv = 0; dv < dV; ++dv)
				ot[dv] = (ot[dv] * static_cast<float>(alpha)) + (static_cast<float>(beta) * vu[dv]);
			m = newM;
		}

		if (!any || !(l > 0.0))
		{
			// All masked: keep zeros.
			continue;
		}

		const float invL = static_cast<float>(1.0 / l);
		for (unsigned int dv = 0; dv < dV; ++dv)
			ot[dv] *= invL;
	}
}

inline void scaled_dot_product_attention_forward_flash_strided(const float* Qbase,
                                                               unsigned int qStride,
                                                               const float* Kbase,
                                                               unsigned int kStride,
                                                               const float* Vbase,
                                                               unsigned int vStride,
                                                               unsigned int T,
                                                               unsigned int dK,
                                                               unsigned int dV,
                                                               bool causal,
                                                               float* Obase,
                                                               unsigned int oStride)
{
	scaled_dot_product_attention_forward_flash_strided(Qbase, qStride, Kbase, kStride, Vbase, vStride, T, dK, dV, causal, Obase, oStride, NULL);
}

// FlashAttention-style backward (recompute) for strided views.
//
// This is a memory-efficient backward that:
// - recomputes the online softmax normalizer per row
// - accumulates dV and computes dQ/dK without allocating O(T) scratch vectors
// - performs 3 passes over keys per row (still scalar CPU)
//
// IMPORTANT: dQ/dK/dV are accumulated into (not cleared).
inline void scaled_dot_product_attention_backward_recompute_flash_strided(const float* Qbase,
                                                                          unsigned int qStride,
                                                                          const float* Kbase,
                                                                          unsigned int kStride,
                                                                          const float* Vbase,
                                                                          unsigned int vStride,
                                                                          const float* dObase,
                                                                          unsigned int dOStride,
                                                                          unsigned int T,
                                                                          unsigned int dK,
                                                                          unsigned int dV,
                                                                          bool causal,
                                                                          float* dQbase,
                                                                          unsigned int dQStride,
                                                                          float* dKbase,
                                                                          unsigned int dKStride,
                                                                          float* dVbase,
                                                                          unsigned int dVStride,
                                                                          const unsigned char* keyAllowed)
{
	if (!Qbase || !Kbase || !Vbase || !dObase || !dQbase || !dKbase || !dVbase)
		return;
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	for (unsigned int t = 0; t < T; ++t)
	{
		const float* qt = Qbase + static_cast<size_t>(t) * static_cast<size_t>(qStride);
		const float* dOt = dObase + static_cast<size_t>(t) * static_cast<size_t>(dOStride);
		float* dQt = dQbase + static_cast<size_t>(t) * static_cast<size_t>(dQStride);

		const unsigned int maxU = causal ? t : (T - 1u);

		// Pass 1: compute softmax normalizer (m, l) online.
		float m = -1e30f;
		double l = 0.0;
		bool any = false;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed && keyAllowed[u] == 0u)
				continue;
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			double dot = 0.0;
			for (unsigned int k = 0; k < dK; ++k)
				dot += static_cast<double>(qt[k]) * static_cast<double>(ku[k]);
			const float s = static_cast<float>(dot) * invSqrt;

			if (!any)
			{
				any = true;
				m = s;
				l = 1.0;
				continue;
			}
			const float newM = (s > m) ? s : m;
			const double alpha = exp(static_cast<double>(m - newM));
			const double beta = exp(static_cast<double>(s - newM));
			l = l * alpha + beta;
			m = newM;
		}
		if (!any || !(l > 0.0))
		{
			// All masked: gradients are zero.
			continue;
		}
		const double invL = 1.0 / l;

		// Pass 2: accumulate rowDot and dV.
		double rowDot = 0.0;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed && keyAllowed[u] == 0u)
				continue;
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			double dotQK = 0.0;
			for (unsigned int k = 0; k < dK; ++k)
				dotQK += static_cast<double>(qt[k]) * static_cast<double>(ku[k]);
			const float s = static_cast<float>(dotQK) * invSqrt;
			const double p = exp(static_cast<double>(s - m)) * invL;

			// dP = dot(dO[t], V[u])
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			double dP = 0.0;
			for (unsigned int dv = 0; dv < dV; ++dv)
				dP += static_cast<double>(dOt[dv]) * static_cast<double>(vu[dv]);
			rowDot += p * dP;

			// dV[u] += p * dO[t]
			float* dVu = dVbase + static_cast<size_t>(u) * static_cast<size_t>(dVStride);
			const float pf = static_cast<float>(p);
			for (unsigned int dv = 0; dv < dV; ++dv)
				dVu[dv] += pf * dOt[dv];
		}

		// Pass 3: dQ and dK from dScores.
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed && keyAllowed[u] == 0u)
				continue;
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			double dotQK = 0.0;
			for (unsigned int k = 0; k < dK; ++k)
				dotQK += static_cast<double>(qt[k]) * static_cast<double>(ku[k]);
			const float s = static_cast<float>(dotQK) * invSqrt;
			const double p = exp(static_cast<double>(s - m)) * invL;
			const float pf = static_cast<float>(p);
			if (pf == 0.0f)
				continue;

			// dP = dot(dO[t], V[u]) again (no scratch)
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			double dP = 0.0;
			for (unsigned int dv = 0; dv < dV; ++dv)
				dP += static_cast<double>(dOt[dv]) * static_cast<double>(vu[dv]);

			const float ds = (pf * (static_cast<float>(dP - rowDot))) * invSqrt;
			if (ds == 0.0f)
				continue;

			float* dKu = dKbase + static_cast<size_t>(u) * static_cast<size_t>(dKStride);
			for (unsigned int k = 0; k < dK; ++k)
			{
				dQt[k] += ds * ku[k];
				dKu[k] += ds * qt[k];
			}
		}
	}
}

inline void scaled_dot_product_attention_backward_recompute_flash_strided(const float* Qbase,
                                                                          unsigned int qStride,
                                                                          const float* Kbase,
                                                                          unsigned int kStride,
                                                                          const float* Vbase,
                                                                          unsigned int vStride,
                                                                          const float* dObase,
                                                                          unsigned int dOStride,
                                                                          unsigned int T,
                                                                          unsigned int dK,
                                                                          unsigned int dV,
                                                                          bool causal,
                                                                          float* dQbase,
                                                                          unsigned int dQStride,
                                                                          float* dKbase,
                                                                          unsigned int dKStride,
                                                                          float* dVbase,
                                                                          unsigned int dVStride)
{
	scaled_dot_product_attention_backward_recompute_flash_strided(Qbase, qStride, Kbase, kStride, Vbase, vStride,
	                                                              dObase, dOStride, T, dK, dV, causal,
	                                                              dQbase, dQStride, dKbase, dKStride, dVbase, dVStride,
	                                                              NULL);
}

inline void scaled_dot_product_attention_backward_recompute_strided(const float* Qbase,
                                                                    unsigned int qStride,
                                                                    const float* Kbase,
                                                                    unsigned int kStride,
                                                                    const float* Vbase,
                                                                    unsigned int vStride,
                                                                    const float* dObase,
                                                                    unsigned int dOStride,
                                                                    unsigned int T,
                                                                    unsigned int dK,
                                                                    unsigned int dV,
                                                                    bool causal,
                                                                    float* dQbase,
                                                                    unsigned int dQStride,
                                                                    float* dKbase,
                                                                    unsigned int dKStride,
                                                                    float* dVbase,
                                                                    unsigned int dVStride,
                                                                    std::vector<float>& scoresScratch,
                                                                    std::vector<float>& probsRowScratch,
                                                                    std::vector<float>& dProbsRowScratch)
{
	if (!Qbase || !Kbase || !Vbase || !dObase || !dQbase || !dKbase || !dVbase)
		return;
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	if (scoresScratch.size() < T)
		scoresScratch.assign(T, 0.0f);
	if (probsRowScratch.size() < T)
		probsRowScratch.assign(T, 0.0f);
	if (dProbsRowScratch.size() < T)
		dProbsRowScratch.assign(T, 0.0f);

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	for (unsigned int t = 0; t < T; ++t)
	{
		const float* qt = Qbase + static_cast<size_t>(t) * static_cast<size_t>(qStride);

		// scores[u] = dot(Q[t], K[u]) * invSqrt
		for (unsigned int u = 0; u < T; ++u)
		{
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			double dot = 0.0;
			for (unsigned int k = 0; k < dK; ++k)
				dot += static_cast<double>(qt[k]) * static_cast<double>(ku[k]);
			scoresScratch[u] = static_cast<float>(dot) * invSqrt;
		}

		softmax_masked_row_stable(scoresScratch.data(), T, t, causal, probsRowScratch);

		const float* dOt = dObase + static_cast<size_t>(t) * static_cast<size_t>(dOStride);

		// dV[u] += p[u] * dO[t]
		const unsigned int maxU = causal ? t : (T - 1u);
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			const float p = probsRowScratch[u];
			if (p == 0.0f)
				continue;
			float* dVu = dVbase + static_cast<size_t>(u) * static_cast<size_t>(dVStride);
			for (unsigned int dv = 0; dv < dV; ++dv)
				dVu[dv] += p * dOt[dv];
		}

		// dProbsRow[u] = dot(dO[t], V[u])
		for (unsigned int u = 0; u < T; ++u)
		{
			if (causal && u > maxU)
			{
				dProbsRowScratch[u] = 0.0f;
				continue;
			}
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			double dot = 0.0;
			for (unsigned int dv = 0; dv < dV; ++dv)
				dot += static_cast<double>(dOt[dv]) * static_cast<double>(vu[dv]);
			dProbsRowScratch[u] = static_cast<float>(dot);
		}

		// rowDot = sum_u p_u * dP_u
		double rowDot = 0.0;
		for (unsigned int u = 0; u <= maxU; ++u)
			rowDot += static_cast<double>(probsRowScratch[u]) * static_cast<double>(dProbsRowScratch[u]);

		// dScores_u = p_u * (dP_u - rowDot)
		// dQ[t] and dK[u] from scores = Q K^T / sqrt(dK)
		float* dQt = dQbase + static_cast<size_t>(t) * static_cast<size_t>(dQStride);
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			const float p = probsRowScratch[u];
			if (p == 0.0f)
				continue;
			const float ds = (p * (dProbsRowScratch[u] - static_cast<float>(rowDot))) * invSqrt;
			if (ds == 0.0f)
				continue;

			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			float* dKu = dKbase + static_cast<size_t>(u) * static_cast<size_t>(dKStride);
			for (unsigned int k = 0; k < dK; ++k)
			{
				dQt[k] += ds * ku[k];
				dKu[k] += ds * qt[k];
			}
		}
	}
}

inline void scaled_dot_product_attention_backward_recompute_strided(const float* Qbase,
                                                                    unsigned int qStride,
                                                                    const float* Kbase,
                                                                    unsigned int kStride,
                                                                    const float* Vbase,
                                                                    unsigned int vStride,
                                                                    const float* dObase,
                                                                    unsigned int dOStride,
                                                                    unsigned int T,
                                                                    unsigned int dK,
                                                                    unsigned int dV,
                                                                    bool causal,
                                                                    float* dQbase,
                                                                    unsigned int dQStride,
                                                                    float* dKbase,
                                                                    unsigned int dKStride,
                                                                    float* dVbase,
                                                                    unsigned int dVStride,
                                                                    std::vector<float>& scoresScratch,
                                                                    std::vector<float>& probsRowScratch,
                                                                    std::vector<float>& dProbsRowScratch,
                                                                    const unsigned char* keyAllowed)
{
	if (!keyAllowed)
	{
		scaled_dot_product_attention_backward_recompute_strided(Qbase, qStride, Kbase, kStride, Vbase, vStride,
		                                                       dObase, dOStride, T, dK, dV, causal,
		                                                       dQbase, dQStride, dKbase, dKStride, dVbase, dVStride,
		                                                       scoresScratch, probsRowScratch, dProbsRowScratch);
		return;
	}
	if (!Qbase || !Kbase || !Vbase || !dObase || !dQbase || !dKbase || !dVbase)
		return;
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	if (scoresScratch.size() < T)
		scoresScratch.assign(T, 0.0f);
	if (probsRowScratch.size() < T)
		probsRowScratch.assign(T, 0.0f);
	if (dProbsRowScratch.size() < T)
		dProbsRowScratch.assign(T, 0.0f);

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	for (unsigned int t = 0; t < T; ++t)
	{
		const float* qt = Qbase + static_cast<size_t>(t) * static_cast<size_t>(qStride);

		for (unsigned int u = 0; u < T; ++u)
		{
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			double dot = 0.0;
			for (unsigned int k = 0; k < dK; ++k)
				dot += static_cast<double>(qt[k]) * static_cast<double>(ku[k]);
			scoresScratch[u] = static_cast<float>(dot) * invSqrt;
		}

		softmax_masked_row_stable_keymask(scoresScratch.data(), T, t, causal, keyAllowed, probsRowScratch);

		const float* dOt = dObase + static_cast<size_t>(t) * static_cast<size_t>(dOStride);
		const unsigned int maxU = causal ? t : (T - 1u);

		// dV[u] += p[u] * dO[t]
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed[u] == 0u)
				continue;
			const float p = probsRowScratch[u];
			if (p == 0.0f)
				continue;
			float* dVu = dVbase + static_cast<size_t>(u) * static_cast<size_t>(dVStride);
			for (unsigned int dv = 0; dv < dV; ++dv)
				dVu[dv] += p * dOt[dv];
		}

		// dProbsRow[u] = dot(dO[t], V[u])
		for (unsigned int u = 0; u < T; ++u)
		{
			if (causal && u > maxU)
			{
				dProbsRowScratch[u] = 0.0f;
				continue;
			}
			if (keyAllowed[u] == 0u)
			{
				dProbsRowScratch[u] = 0.0f;
				continue;
			}
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			double dot = 0.0;
			for (unsigned int dv = 0; dv < dV; ++dv)
				dot += static_cast<double>(dOt[dv]) * static_cast<double>(vu[dv]);
			dProbsRowScratch[u] = static_cast<float>(dot);
		}

		double rowDot = 0.0;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed[u] == 0u)
				continue;
			rowDot += static_cast<double>(probsRowScratch[u]) * static_cast<double>(dProbsRowScratch[u]);
		}

		float* dQt = dQbase + static_cast<size_t>(t) * static_cast<size_t>(dQStride);
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed[u] == 0u)
				continue;
			const float p = probsRowScratch[u];
			if (p == 0.0f)
				continue;
			const float ds = (p * (dProbsRowScratch[u] - static_cast<float>(rowDot))) * invSqrt;
			if (ds == 0.0f)
				continue;

			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			float* dKu = dKbase + static_cast<size_t>(u) * static_cast<size_t>(dKStride);
			for (unsigned int k = 0; k < dK; ++k)
			{
				dQt[k] += ds * ku[k];
				dKu[k] += ds * qt[k];
			}
		}
	}
}

} // namespace transformer_ops
} // namespace glades

