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

#include "transformer_kernels.h"

namespace glades {
namespace transformer_ops {

inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }
inline float relu_deriv_from_y(float y) { return (y > 0.0f) ? 1.0f : 0.0f; }

// SiLU / Swish: x * sigmoid(x)
inline float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
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

inline unsigned int attention_row_max_u(unsigned int T, unsigned int rowT, bool causal)
{
	return causal ? rowT : (T - 1u);
}

inline double attention_softmax_row_dot(const std::vector<float>& probsRow,
                                        const std::vector<float>& dProbsRow,
                                        unsigned int maxU)
{
	double rowDot = 0.0;
	for (unsigned int u = 0; u <= maxU; ++u)
		rowDot += static_cast<double>(probsRow[u]) * static_cast<double>(dProbsRow[u]);
	return rowDot;
}

inline float attention_score_grad(float prob,
                                  float dProb,
                                  double rowDot,
                                  float invSqrt)
{
	return (prob * (dProb - static_cast<float>(rowDot))) * invSqrt;
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
		const unsigned int maxU = attention_row_max_u(T, t, causal);
		double rowDot = 0.0;
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
			const unsigned int maxU = attention_row_max_u(T, t, causal);
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
			const double rowDot = attention_softmax_row_dot(probsRow, dProbsRow, maxU);

		// dScores_u = p_u * (dP_u - rowDot)
		// dQ[t] and dK[u] from scores = Q K^T / sqrt(dK)
		for (unsigned int u = 0; u <= maxU; ++u)
			{
				const float p = probsRow[u];
				if (p == 0.0f)
					continue;
				const float ds = attention_score_grad(p, dProbsRow[u], rowDot, invSqrt);
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
		const unsigned int maxU = attention_row_max_u(T, t, causal);
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
#if defined(__SSE2__)
			// Software prefetch: bring next key block into L1 cache
			if ((u & 31u) == 0u && (u + 32u) <= maxU)
			{
				const float* kpf = Kbase + static_cast<size_t>(u + 32u) * static_cast<size_t>(kStride);
				_mm_prefetch(reinterpret_cast<const char*>(kpf), _MM_HINT_T0);
			}
#endif
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);

			const float s = glades::transformer_kernels::dot_f32(qt, ku, dK) * invSqrt;

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
			const float alpha = expf(m - newM);
			const float beta = expf(s - newM);
			l = l * static_cast<double>(alpha) + static_cast<double>(beta);

			// Scale old accumulator + add new contribution.
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			for (unsigned int dv = 0; dv < dV; ++dv)
				ot[dv] = (ot[dv] * alpha) + (beta * vu[dv]);
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

	// Per-row scratch: cache scores, probabilities, and dP to avoid recomputation across passes.
	// This trades O(T) memory for eliminating redundant Q*K dot products, dO*V dot products,
	// and exp() calls across the 3 passes (down from 3x to 1x for Q*K, 2x to 1x for dO*V).
	std::vector<float> sCache(T);
	std::vector<float> pCache(T);
	std::vector<float> dPCache(T);

	for (unsigned int t = 0; t < T; ++t)
	{
		const float* qt = Qbase + static_cast<size_t>(t) * static_cast<size_t>(qStride);
		const float* dOt = dObase + static_cast<size_t>(t) * static_cast<size_t>(dOStride);
		float* dQt = dQbase + static_cast<size_t>(t) * static_cast<size_t>(dQStride);

		const unsigned int maxU = causal ? t : (T - 1u);

		// Pass 1: compute scores (cached) and softmax normalizer (m, l) online.
		float m = -1e30f;
		double l = 0.0;
		bool any = false;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed && keyAllowed[u] == 0u)
			{
				sCache[u] = -1e30f;
				continue;
			}
#if defined(__SSE2__)
			if ((u & 31u) == 0u && (u + 32u) <= maxU)
			{
				const float* kpf = Kbase + static_cast<size_t>(u + 32u) * static_cast<size_t>(kStride);
				_mm_prefetch(reinterpret_cast<const char*>(kpf), _MM_HINT_T0);
			}
#endif
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			const float s = glades::transformer_kernels::dot_f32(qt, ku, dK) * invSqrt;
			sCache[u] = s;

			if (!any)
			{
				any = true;
				m = s;
				l = 1.0;
				continue;
			}
			const float newM = (s > m) ? s : m;
			const float alpha = expf(m - newM);
			const float beta = expf(s - newM);
			l = l * static_cast<double>(alpha) + static_cast<double>(beta);
			m = newM;
		}
		if (!any || !(l > 0.0))
		{
			// All masked: gradients are zero.
			continue;
		}
		const double invL = 1.0 / l;

		// Pass 2: accumulate dV and rowDot using cached scores.
		// Cache probabilities and dP for reuse in pass 3.
		double rowDot = 0.0;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed && keyAllowed[u] == 0u)
			{
				pCache[u] = 0.0f;
				dPCache[u] = 0.0f;
				continue;
			}
			const float pf = static_cast<float>(static_cast<double>(expf(sCache[u] - m)) * invL);
			pCache[u] = pf;

			// dP = dot(dO[t], V[u])
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			const float dP = glades::transformer_kernels::dot_f32(dOt, vu, dV);
			dPCache[u] = dP;
			rowDot += static_cast<double>(pf) * static_cast<double>(dP);

			// dV[u] += p * dO[t]
			float* dVu = dVbase + static_cast<size_t>(u) * static_cast<size_t>(dVStride);
			glades::transformer_kernels::axpy_f32(dVu, dOt, pf, dV);
		}

		// Pass 3: dQ and dK from dScores, using cached p and dP.
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed && keyAllowed[u] == 0u)
				continue;
			const float pf = pCache[u];
			if (pf == 0.0f)
				continue;

			const float ds = attention_score_grad(pf, dPCache[u], rowDot, invSqrt);
			if (ds == 0.0f)
				continue;

			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			float* dKu = dKbase + static_cast<size_t>(u) * static_cast<size_t>(dKStride);
			glades::transformer_kernels::axpy_f32(dQt, ku, ds, dK);
			glades::transformer_kernels::axpy_f32(dKu, qt, ds, dK);
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

// Chunk variant: processes query rows [tBegin, tEnd) only.
// dQ writes go to dQbase (strided, only rows tBegin..tEnd-1 are touched).
// dK/dV writes go to contiguous local buffers (stride = dK/dV, not dKStride/dVStride).
// Caller must zero dKlocal/dVlocal before calling.
inline void scaled_dot_product_attention_backward_recompute_flash_chunk(
    const float* Qbase, unsigned int qStride,
    const float* Kbase, unsigned int kStride,
    const float* Vbase, unsigned int vStride,
    const float* dObase, unsigned int dOStride,
    unsigned int tBegin, unsigned int tEnd,
    unsigned int T, unsigned int dK, unsigned int dV,
    bool causal,
    float* dQbase, unsigned int dQStride,
    float* dKlocal, float* dVlocal,
    const unsigned char* keyAllowed)
{
	if (!Qbase || !Kbase || !Vbase || !dObase || !dQbase || !dKlocal || !dVlocal)
		return;
	if (T == 0u || dK == 0u || dV == 0u || tBegin >= tEnd)
		return;

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	std::vector<float> sCache(T);
	std::vector<float> pCache(T);
	std::vector<float> dPCache(T);

	for (unsigned int t = tBegin; t < tEnd; ++t)
	{
		const float* qt = Qbase + static_cast<size_t>(t) * static_cast<size_t>(qStride);
		const float* dOt = dObase + static_cast<size_t>(t) * static_cast<size_t>(dOStride);
		float* dQt = dQbase + static_cast<size_t>(t) * static_cast<size_t>(dQStride);

		const unsigned int maxU = attention_row_max_u(T, t, causal);

		float m = -1e30f;
		double l = 0.0;
		bool any = false;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed && keyAllowed[u] == 0u)
			{
				sCache[u] = -1e30f;
				continue;
			}
#if defined(__SSE2__)
			if ((u & 31u) == 0u && (u + 32u) <= maxU)
			{
				const float* kpf = Kbase + static_cast<size_t>(u + 32u) * static_cast<size_t>(kStride);
				_mm_prefetch(reinterpret_cast<const char*>(kpf), _MM_HINT_T0);
			}
#endif
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			const float s = glades::transformer_kernels::dot_f32(qt, ku, dK) * invSqrt;
			sCache[u] = s;

			if (!any) { any = true; m = s; l = 1.0; continue; }
			const float newM = (s > m) ? s : m;
			const float alpha = expf(m - newM);
			const float beta = expf(s - newM);
			l = l * static_cast<double>(alpha) + static_cast<double>(beta);
			m = newM;
		}
		if (!any || !(l > 0.0))
			continue;
		const double invL = 1.0 / l;

		double rowDot = 0.0;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed && keyAllowed[u] == 0u)
			{
				pCache[u] = 0.0f;
				dPCache[u] = 0.0f;
				continue;
			}
			const float pf = static_cast<float>(static_cast<double>(expf(sCache[u] - m)) * invL);
			pCache[u] = pf;
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			const float dP = glades::transformer_kernels::dot_f32(dOt, vu, dV);
			dPCache[u] = dP;
			rowDot += static_cast<double>(pf) * static_cast<double>(dP);
			// dV into contiguous local buffer (stride = dV)
			float* dVu = dVlocal + static_cast<size_t>(u) * static_cast<size_t>(dV);
			glades::transformer_kernels::axpy_f32(dVu, dOt, pf, dV);
		}

		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (keyAllowed && keyAllowed[u] == 0u)
				continue;
			const float pf = pCache[u];
			if (pf == 0.0f) continue;
			const float ds = attention_score_grad(pf, dPCache[u], rowDot, invSqrt);
			if (ds == 0.0f) continue;
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			// dK into contiguous local buffer (stride = dK)
			float* dKu = dKlocal + static_cast<size_t>(u) * static_cast<size_t>(dK);
			glades::transformer_kernels::axpy_f32(dQt, ku, ds, dK);
			glades::transformer_kernels::axpy_f32(dKu, qt, ds, dK);
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

// ============================================================================
// Paradigm shift #78 — ATTENTION-SINK-DISTILL-CHIRON (StreamingLLM-style)
// ============================================================================
//
// Sink+window attention. For each query position t and key position u
// (with causal restriction u <= t), key u is allowed iff:
//
//   (sinkCount == 0 AND windowSize == 0)  // both disabled => full causal
//   OR  u < sinkCount                     // sink (always allowed)
//   OR  (windowSize > 0 AND u + windowSize > t)  // within sliding window
//
// At t large with windowSize=W and sinkCount=S: visited keys = S + W (constant).
// KV cache for this query references (S + W) entries, INDEPENDENT of T.
//
// Reference: Xiao et al. 2023, "Efficient Streaming Language Models with
// Attention Sinks" (StreamingLLM). vLLM/lmdeploy/llama.cpp/MLC-LLM/TGI all
// ship native attention-sink in their inference paths.

inline bool sw_key_allowed(unsigned int t,
                           unsigned int u,
                           unsigned int sinkCount,
                           unsigned int windowSize,
                           const unsigned char* keyAllowed)
{
	if (keyAllowed && keyAllowed[u] == 0u)
		return false;
	if (sinkCount == 0u && windowSize == 0u)
		return true;
	if (u < sinkCount)
		return true;
	if (windowSize > 0u && (u + windowSize) > t)
		return true;
	return false;
}

inline unsigned int sw_visited_count(unsigned int t,
                                     unsigned int T,
                                     bool causal,
                                     unsigned int sinkCount,
                                     unsigned int windowSize,
                                     const unsigned char* keyAllowed)
{
	const unsigned int maxU = causal ? t : (T - 1u);
	unsigned int n = 0u;
	for (unsigned int u = 0; u <= maxU && u < T; ++u)
	{
		if (sw_key_allowed(t, u, sinkCount, windowSize, keyAllowed))
			++n;
	}
	return n;
}

inline void scaled_dot_product_attention_forward_flash_strided_sw(const float* Qbase,
                                                                  unsigned int qStride,
                                                                  const float* Kbase,
                                                                  unsigned int kStride,
                                                                  const float* Vbase,
                                                                  unsigned int vStride,
                                                                  unsigned int T,
                                                                  unsigned int dK,
                                                                  unsigned int dV,
                                                                  bool causal,
                                                                  unsigned int sinkCount,
                                                                  unsigned int windowSize,
                                                                  float* Obase,
                                                                  unsigned int oStride,
                                                                  const unsigned char* keyAllowed)
{
	if (sinkCount == 0u && windowSize == 0u)
	{
		scaled_dot_product_attention_forward_flash_strided(Qbase, qStride, Kbase, kStride, Vbase, vStride,
		                                                   T, dK, dV, causal, Obase, oStride, keyAllowed);
		return;
	}
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

		float m = -1e30f;
		double l = 0.0;
		bool any = false;

		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (!sw_key_allowed(t, u, sinkCount, windowSize, keyAllowed))
				continue;
#if defined(__SSE2__)
			if ((u & 31u) == 0u && (u + 32u) <= maxU)
			{
				const float* kpf = Kbase + static_cast<size_t>(u + 32u) * static_cast<size_t>(kStride);
				_mm_prefetch(reinterpret_cast<const char*>(kpf), _MM_HINT_T0);
			}
#endif
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			const float s = glades::transformer_kernels::dot_f32(qt, ku, dK) * invSqrt;

			if (!any)
			{
				any = true;
				m = s;
				l = 1.0;
				const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
				for (unsigned int dv = 0; dv < dV; ++dv)
					ot[dv] = vu[dv];
				continue;
			}

			const float newM = (s > m) ? s : m;
			const float alpha = expf(m - newM);
			const float beta = expf(s - newM);
			l = l * static_cast<double>(alpha) + static_cast<double>(beta);
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			for (unsigned int dv = 0; dv < dV; ++dv)
				ot[dv] = (ot[dv] * alpha) + (beta * vu[dv]);
			m = newM;
		}

		if (!any || !(l > 0.0))
			continue;

		const float invL = static_cast<float>(1.0 / l);
		for (unsigned int dv = 0; dv < dV; ++dv)
			ot[dv] *= invL;
	}
}

inline void scaled_dot_product_attention_backward_recompute_flash_strided_sw(const float* Qbase,
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
                                                                             unsigned int sinkCount,
                                                                             unsigned int windowSize,
                                                                             float* dQbase,
                                                                             unsigned int dQStride,
                                                                             float* dKbase,
                                                                             unsigned int dKStride,
                                                                             float* dVbase,
                                                                             unsigned int dVStride,
                                                                             const unsigned char* keyAllowed)
{
	if (sinkCount == 0u && windowSize == 0u)
	{
		scaled_dot_product_attention_backward_recompute_flash_strided(Qbase, qStride, Kbase, kStride, Vbase, vStride,
		                                                              dObase, dOStride, T, dK, dV, causal,
		                                                              dQbase, dQStride, dKbase, dKStride, dVbase, dVStride,
		                                                              keyAllowed);
		return;
	}
	if (!Qbase || !Kbase || !Vbase || !dObase || !dQbase || !dKbase || !dVbase)
		return;
	if (T == 0u || dK == 0u || dV == 0u)
		return;

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	std::vector<float> sCache(T);
	std::vector<float> pCache(T);
	std::vector<float> dPCache(T);

	for (unsigned int t = 0; t < T; ++t)
	{
		const float* qt = Qbase + static_cast<size_t>(t) * static_cast<size_t>(qStride);
		const float* dOt = dObase + static_cast<size_t>(t) * static_cast<size_t>(dOStride);
		float* dQt = dQbase + static_cast<size_t>(t) * static_cast<size_t>(dQStride);

		const unsigned int maxU = causal ? t : (T - 1u);

		float m = -1e30f;
		double l = 0.0;
		bool any = false;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (!sw_key_allowed(t, u, sinkCount, windowSize, keyAllowed))
			{
				sCache[u] = -1e30f;
				continue;
			}
#if defined(__SSE2__)
			if ((u & 31u) == 0u && (u + 32u) <= maxU)
			{
				const float* kpf = Kbase + static_cast<size_t>(u + 32u) * static_cast<size_t>(kStride);
				_mm_prefetch(reinterpret_cast<const char*>(kpf), _MM_HINT_T0);
			}
#endif
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			const float s = glades::transformer_kernels::dot_f32(qt, ku, dK) * invSqrt;
			sCache[u] = s;

			if (!any) { any = true; m = s; l = 1.0; continue; }
			const float newM = (s > m) ? s : m;
			const float alpha = expf(m - newM);
			const float beta = expf(s - newM);
			l = l * static_cast<double>(alpha) + static_cast<double>(beta);
			m = newM;
		}
		if (!any || !(l > 0.0))
			continue;
		const double invL = 1.0 / l;

		double rowDot = 0.0;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (!sw_key_allowed(t, u, sinkCount, windowSize, keyAllowed))
			{
				pCache[u] = 0.0f;
				dPCache[u] = 0.0f;
				continue;
			}
			const float pf = static_cast<float>(static_cast<double>(expf(sCache[u] - m)) * invL);
			pCache[u] = pf;
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			const float dP = glades::transformer_kernels::dot_f32(dOt, vu, dV);
			dPCache[u] = dP;
			rowDot += static_cast<double>(pf) * static_cast<double>(dP);
			float* dVu = dVbase + static_cast<size_t>(u) * static_cast<size_t>(dVStride);
			glades::transformer_kernels::axpy_f32(dVu, dOt, pf, dV);
		}

		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (!sw_key_allowed(t, u, sinkCount, windowSize, keyAllowed))
				continue;
			const float pf = pCache[u];
			if (pf == 0.0f) continue;
			const float ds = attention_score_grad(pf, dPCache[u], rowDot, invSqrt);
			if (ds == 0.0f) continue;
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			float* dKu = dKbase + static_cast<size_t>(u) * static_cast<size_t>(dKStride);
			glades::transformer_kernels::axpy_f32(dQt, ku, ds, dK);
			glades::transformer_kernels::axpy_f32(dKu, qt, ds, dK);
		}
	}
}

inline void scaled_dot_product_attention_backward_recompute_flash_chunk_sw(
    const float* Qbase, unsigned int qStride,
    const float* Kbase, unsigned int kStride,
    const float* Vbase, unsigned int vStride,
    const float* dObase, unsigned int dOStride,
    unsigned int tBegin, unsigned int tEnd,
    unsigned int T, unsigned int dK, unsigned int dV,
    bool causal,
    unsigned int sinkCount,
    unsigned int windowSize,
    float* dQbase, unsigned int dQStride,
    float* dKlocal, float* dVlocal,
    const unsigned char* keyAllowed)
{
	if (sinkCount == 0u && windowSize == 0u)
	{
		scaled_dot_product_attention_backward_recompute_flash_chunk(Qbase, qStride, Kbase, kStride, Vbase, vStride,
		                                                            dObase, dOStride, tBegin, tEnd,
		                                                            T, dK, dV, causal, dQbase, dQStride,
		                                                            dKlocal, dVlocal, keyAllowed);
		return;
	}
	if (!Qbase || !Kbase || !Vbase || !dObase || !dQbase || !dKlocal || !dVlocal)
		return;
	if (T == 0u || dK == 0u || dV == 0u || tBegin >= tEnd)
		return;

	const float invSqrt = 1.0f / static_cast<float>(sqrt(static_cast<double>(dK)));

	std::vector<float> sCache(T);
	std::vector<float> pCache(T);
	std::vector<float> dPCache(T);

	for (unsigned int t = tBegin; t < tEnd; ++t)
	{
		const float* qt = Qbase + static_cast<size_t>(t) * static_cast<size_t>(qStride);
		const float* dOt = dObase + static_cast<size_t>(t) * static_cast<size_t>(dOStride);
		float* dQt = dQbase + static_cast<size_t>(t) * static_cast<size_t>(dQStride);

		const unsigned int maxU = attention_row_max_u(T, t, causal);

		float m = -1e30f;
		double l = 0.0;
		bool any = false;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (!sw_key_allowed(t, u, sinkCount, windowSize, keyAllowed))
			{
				sCache[u] = -1e30f;
				continue;
			}
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			const float s = glades::transformer_kernels::dot_f32(qt, ku, dK) * invSqrt;
			sCache[u] = s;
			if (!any) { any = true; m = s; l = 1.0; continue; }
			const float newM = (s > m) ? s : m;
			const float alpha = expf(m - newM);
			const float beta = expf(s - newM);
			l = l * static_cast<double>(alpha) + static_cast<double>(beta);
			m = newM;
		}
		if (!any || !(l > 0.0))
			continue;
		const double invL = 1.0 / l;

		double rowDot = 0.0;
		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (!sw_key_allowed(t, u, sinkCount, windowSize, keyAllowed))
			{
				pCache[u] = 0.0f;
				dPCache[u] = 0.0f;
				continue;
			}
			const float pf = static_cast<float>(static_cast<double>(expf(sCache[u] - m)) * invL);
			pCache[u] = pf;
			const float* vu = Vbase + static_cast<size_t>(u) * static_cast<size_t>(vStride);
			const float dP = glades::transformer_kernels::dot_f32(dOt, vu, dV);
			dPCache[u] = dP;
			rowDot += static_cast<double>(pf) * static_cast<double>(dP);
			float* dVu = dVlocal + static_cast<size_t>(u) * static_cast<size_t>(dV);
			glades::transformer_kernels::axpy_f32(dVu, dOt, pf, dV);
		}

		for (unsigned int u = 0; u <= maxU; ++u)
		{
			if (!sw_key_allowed(t, u, sinkCount, windowSize, keyAllowed))
				continue;
			const float pf = pCache[u];
			if (pf == 0.0f) continue;
			const float ds = attention_score_grad(pf, dPCache[u], rowDot, invSqrt);
			if (ds == 0.0f) continue;
			const float* ku = Kbase + static_cast<size_t>(u) * static_cast<size_t>(kStride);
			float* dKu = dKlocal + static_cast<size_t>(u) * static_cast<size_t>(dK);
			glades::transformer_kernels::axpy_f32(dQt, ku, ds, dK);
			glades::transformer_kernels::axpy_f32(dKu, qt, ds, dK);
		}
	}
}

// ============================================================================
// Paradigm shift #76 — MLA-DISTILL-CHIRON (Multi-Latent Attention)
// ============================================================================
//
// DeepSeek-V2/V3 Multi-Latent Attention. Compress KV cache via low-rank
// latent projection. Cache stores c_t (d_c ≈ 384-512) instead of full
// K_t, V_t (n_heads * d_kv ≈ 2048). Decompression at attention time:
//
//   c_t = h_t @ W_DKV     [d_c]                         (cached)
//   K_t = c_t @ W_UK      [n_heads * d_kv]              (recomputed)
//   V_t = c_t @ W_UV      [n_heads * d_kv]              (recomputed)
//
// KV cache compression: (n_heads * d_kv * 2) / d_c. At standard config
// (n_heads=16, d_kv=128, d_c=512): 4096/512 = 8x compression.
//
// Theorem 2: The MLA forward pass is bit-exact equivalent to a standard-MHA
// forward where W_K = W_DKV @ W_UK and W_V = W_DKV @ W_UV. The "low-rank"
// constraint is on the JOINT projection h -> K (rank d_c) and h -> V (rank d_c),
// but the attention math is unchanged.
//
// Reference: DeepSeek-V2 (Liu et al. 2024), DeepSeek-V3 (DeepSeek 2024).

// Decompress KV latent c_t into K and V tensors.
//   c        [T, d_c]
//   W_UK     [d_c, dKVtotal]   where dKVtotal = n_heads * d_kv (or nKVHeads * d_kv for GQA)
//   W_UV     [d_c, dKVtotal]
//   K_out    [T, dKVtotal]
//   V_out    [T, dKVtotal]
//
// Pure CPU primitive; production GPU path uses fused gemm.
inline void mla_decompress_kv(const float* c,
                              const float* W_UK,
                              const float* W_UV,
                              unsigned int T,
                              unsigned int d_c,
                              unsigned int dKVtotal,
                              float* K_out,
                              float* V_out)
{
	if (!c || !W_UK || !W_UV || !K_out || !V_out)
		return;
	if (T == 0u || d_c == 0u || dKVtotal == 0u)
		return;
	for (unsigned int t = 0; t < T; ++t)
	{
		const float* ct = c + static_cast<size_t>(t) * d_c;
		float* Kt = K_out + static_cast<size_t>(t) * dKVtotal;
		float* Vt = V_out + static_cast<size_t>(t) * dKVtotal;
		for (unsigned int j = 0; j < dKVtotal; ++j)
		{
			double kSum = 0.0;
			double vSum = 0.0;
			for (unsigned int i = 0; i < d_c; ++i)
			{
				const double ci = static_cast<double>(ct[i]);
				kSum += ci * static_cast<double>(W_UK[static_cast<size_t>(i) * dKVtotal + j]);
				vSum += ci * static_cast<double>(W_UV[static_cast<size_t>(i) * dKVtotal + j]);
			}
			Kt[j] = static_cast<float>(kSum);
			Vt[j] = static_cast<float>(vSum);
		}
	}
}

// Compute the KV latent c from hidden h and down-projection W_DKV.
//   h        [T, d_h]
//   W_DKV    [d_h, d_c]
//   c_out    [T, d_c]
inline void mla_compute_latent(const float* h,
                               const float* W_DKV,
                               unsigned int T,
                               unsigned int d_h,
                               unsigned int d_c,
                               float* c_out)
{
	if (!h || !W_DKV || !c_out)
		return;
	if (T == 0u || d_h == 0u || d_c == 0u)
		return;
	for (unsigned int t = 0; t < T; ++t)
	{
		const float* ht = h + static_cast<size_t>(t) * d_h;
		float* ct = c_out + static_cast<size_t>(t) * d_c;
		for (unsigned int j = 0; j < d_c; ++j)
		{
			double s = 0.0;
			for (unsigned int i = 0; i < d_h; ++i)
				s += static_cast<double>(ht[i]) *
				     static_cast<double>(W_DKV[static_cast<size_t>(i) * d_c + j]);
			ct[j] = static_cast<float>(s);
		}
	}
}

// Compute the effective compression ratio MHA cache size / MLA cache size.
//   nHeads, dKV: standard MHA config.
//   d_c:         MLA latent rank.
//   d_rope:      decoupled-RoPE rank (separately cached).
inline float mla_compression_ratio(unsigned int nHeads,
                                   unsigned int dKV,
                                   unsigned int d_c,
                                   unsigned int d_rope)
{
	const unsigned int mhaPerToken = nHeads * dKV * 2u;  // K + V
	const unsigned int mlaPerToken = d_c + d_rope;
	if (mlaPerToken == 0u) return 0.0f;
	return static_cast<float>(mhaPerToken) / static_cast<float>(mlaPerToken);
}

} // namespace transformer_ops
} // namespace glades
