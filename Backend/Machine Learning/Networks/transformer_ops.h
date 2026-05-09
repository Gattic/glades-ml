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

// ============================================================================
// Paradigm shift #74 — PHOENIX-1BIT-DISTILL-COMBO-CHIRON (binary kernels)
// ============================================================================
//
// Binary {-1, +1} weights with 1 bit per weight (16x compression vs BF16,
// 32x vs FP32). Forward GEMM Y[m,n] = sum_k X[m,k] * W[k,n] where
// W[k,n] in {-1, +1}, packed as: bit=1 if W=+1, bit=0 if W=-1.
//
// Decoding identity:
//   sum_k X[m,k] * W[k,n] = 2 * sum_{k : bit[k,n]==1} X[m,k] - sum_k X[m,k]
//
// I.e., one masked-sum + one row-sum per (m,n). Per row m, the row-sum is
// reused across all n. Compute reduction: ~2x over float GEMM at this
// granularity (no multiply in the inner loop).
//
// Reference: BitNet b1.58 / b1.0 (Microsoft 2024). XNOR-popcount form is
// for fully-binary GEMM (X also {-1,+1}); we keep X float for QAT.

// Pack ±1 weights into bits. For w >= 0: bit=1; for w < 0: bit=0.
// W:        [K * N] floats, expected in [-1, +1] range, row-major (k major, n minor).
// W_bits:   [(K * N + 7) / 8] bytes; bit (k*N + n) = (W[k,n] >= 0).
//
// NOTE: This is the "row-major bits" layout — convenient for unpacking/diagnostics
// but slow for binary GEMM since the bits for a single output column n are
// strided in memory. For fast GEMM use phoenix_pack_signs_colmajor below.
inline void phoenix_pack_signs(const float* W,
                               unsigned int K,
                               unsigned int N,
                               unsigned char* W_bits)
{
	if (!W || !W_bits || K == 0u || N == 0u)
		return;
	const size_t total = static_cast<size_t>(K) * static_cast<size_t>(N);
	const size_t bytes = (total + 7u) >> 3;
	for (size_t b = 0; b < bytes; ++b)
		W_bits[b] = 0u;
	for (size_t i = 0; i < total; ++i)
	{
		if (W[i] >= 0.0f)
			W_bits[i >> 3] |= (unsigned char)(1u << (i & 7u));
	}
}

// Column-major bit packing: for each output column n, pack K bits sequentially.
// Layout: bits for column n start at byte offset n * ((K+7)/8). Bit index k of
// column n is at byte (n * Kbytes + k/8), bit position (k & 7).
//
// This layout makes binary GEMM cache-friendly: the K bits for one (m, n) output
// are contiguous in memory and can be processed 8 bits at a time.
inline void phoenix_pack_signs_colmajor(const float* W,
                                        unsigned int K,
                                        unsigned int N,
                                        unsigned char* W_bits)
{
	if (!W || !W_bits || K == 0u || N == 0u)
		return;
	const size_t Kbytes = (static_cast<size_t>(K) + 7u) >> 3;
	const size_t totalBytes = Kbytes * static_cast<size_t>(N);
	for (size_t b = 0; b < totalBytes; ++b)
		W_bits[b] = 0u;
	for (unsigned int n = 0; n < N; ++n)
	{
		unsigned char* col = W_bits + static_cast<size_t>(n) * Kbytes;
		for (unsigned int k = 0; k < K; ++k)
		{
			if (W[static_cast<size_t>(k) * N + n] >= 0.0f)
				col[k >> 3] |= (unsigned char)(1u << (k & 7u));
		}
	}
}

// Unpack bits back to ±1 floats.
inline void phoenix_unpack_signs(const unsigned char* W_bits,
                                 unsigned int K,
                                 unsigned int N,
                                 float* W_out)
{
	if (!W_bits || !W_out || K == 0u || N == 0u)
		return;
	const size_t total = static_cast<size_t>(K) * static_cast<size_t>(N);
	for (size_t i = 0; i < total; ++i)
	{
		const unsigned char bit = (W_bits[i >> 3] >> (i & 7u)) & 1u;
		W_out[i] = bit ? 1.0f : -1.0f;
	}
}

// Compression ratio in bytes: float weights / packed bits.
inline float phoenix_compression_ratio_bf16()
{
	// BF16 (2 bytes/weight) vs 1 bit/weight = 16x.
	return 16.0f;
}

inline float phoenix_compression_ratio_fp32()
{
	// FP32 (4 bytes/weight) vs 1 bit/weight = 32x.
	return 32.0f;
}

// Binary GEMM with row-major bit packing (slow but layout-compatible with
// phoenix_pack_signs). Provided for correctness-test parity; for speed use
// phoenix_binary_gemm_colmajor with phoenix_pack_signs_colmajor.
//   X:      [M, K]
//   W_bits: bit-packed [(K*N+7)/8] bytes (bit (k*N+n) == (W[k,n] >= 0))
//   Y:      [M, N]   (output, overwritten)
//
// Algorithm: per row m, precompute rowSum[m] = sum_k X[m,k]; per output
// (m,n), maskedSum = sum_{k : bit[k,n]==1} X[m,k]; Y[m,n] = 2*maskedSum - rowSum[m].
inline void phoenix_binary_gemm(const float* X,
                                const unsigned char* W_bits,
                                unsigned int M,
                                unsigned int N,
                                unsigned int K,
                                float* Y)
{
	if (!X || !W_bits || !Y || M == 0u || N == 0u || K == 0u)
		return;

	std::vector<float> rowSum(M, 0.0f);
	for (unsigned int m = 0; m < M; ++m)
	{
		const float* xm = X + static_cast<size_t>(m) * K;
		double s = 0.0;
		for (unsigned int k = 0; k < K; ++k)
			s += static_cast<double>(xm[k]);
		rowSum[m] = static_cast<float>(s);
	}

	for (unsigned int m = 0; m < M; ++m)
	{
		const float* xm = X + static_cast<size_t>(m) * K;
		float* ym = Y + static_cast<size_t>(m) * N;
		const float rs = rowSum[m];

		for (unsigned int n = 0; n < N; ++n)
		{
			double maskedSum = 0.0;
			for (unsigned int k = 0; k < K; ++k)
			{
				const size_t bitIdx = static_cast<size_t>(k) * static_cast<size_t>(N) +
				                      static_cast<size_t>(n);
				const unsigned char bit = (W_bits[bitIdx >> 3] >> (bitIdx & 7u)) & 1u;
				if (bit)
					maskedSum += static_cast<double>(xm[k]);
			}
			ym[n] = static_cast<float>(2.0 * maskedSum) - rs;
		}
	}
}

// Fast binary GEMM with column-major bit packing.
//   X:        [M, K]
//   W_bits:   layout from phoenix_pack_signs_colmajor (each column packs K bits
//             contiguously; total N * ((K+7)/8) bytes).
//   Y:        [M, N]
//
// Algorithm: same masked-sum identity but processes 8 bits per byte sequentially
// over X[m, k]. Pre-computes rowSum once. Inner loop is byte-by-byte: each byte
// accumulates X[m, k..k+7] into maskedSum where the byte's bits are 1.
inline void phoenix_binary_gemm_colmajor(const float* X,
                                         const unsigned char* W_bits,
                                         unsigned int M,
                                         unsigned int N,
                                         unsigned int K,
                                         float* Y)
{
	if (!X || !W_bits || !Y || M == 0u || N == 0u || K == 0u)
		return;
	const size_t Kbytes = (static_cast<size_t>(K) + 7u) >> 3;

	std::vector<float> rowSum(M, 0.0f);
	for (unsigned int m = 0; m < M; ++m)
	{
		const float* xm = X + static_cast<size_t>(m) * K;
		float s = 0.0f;
		for (unsigned int k = 0; k < K; ++k)
			s += xm[k];
		rowSum[m] = s;
	}

	for (unsigned int m = 0; m < M; ++m)
	{
		const float* xm = X + static_cast<size_t>(m) * K;
		float* ym = Y + static_cast<size_t>(m) * N;
		const float rs = rowSum[m];

		for (unsigned int n = 0; n < N; ++n)
		{
			const unsigned char* col = W_bits + static_cast<size_t>(n) * Kbytes;
			float maskedSum = 0.0f;
			for (size_t bb = 0; bb < Kbytes; ++bb)
			{
				const unsigned char byte = col[bb];
				const unsigned int kBase = static_cast<unsigned int>(bb) * 8u;
				const unsigned int kEnd = (kBase + 8u <= K) ? (kBase + 8u) : K;
				if (byte == 0u) continue;
				if (byte == 0xFFu && (kBase + 8u) <= K)
				{
					maskedSum += xm[kBase + 0u] + xm[kBase + 1u] + xm[kBase + 2u] + xm[kBase + 3u] +
					             xm[kBase + 4u] + xm[kBase + 5u] + xm[kBase + 6u] + xm[kBase + 7u];
					continue;
				}
				for (unsigned int j = 0; j < (kEnd - kBase); ++j)
					if (byte & (1u << j))
						maskedSum += xm[kBase + j];
			}
			ym[n] = 2.0f * maskedSum - rs;
		}
	}
}

// ============================================================================
// Paradigm shift #93 — ASTRA-KAHAN-DISTILL (Path 2 round 2)
// ============================================================================
//
// Stateless-v Adam (#41 ASTRA recomposition) with Kahan compensation on the
// momentum accumulator. Per-step update:
//
//   m_t = β1 · m_{t-1} + (1-β1) · g_t      (with Kahan compensation in c_t)
//   v_t = g_t²                              (stateless — no persistent v)
//   θ_t = θ_{t-1} - lr · m_t / (sqrt(v_t) + ε)
//
// Memory:
//   - Standard Adam: 2 floats per param (m, v).
//   - ASTRA-KAHAN: 2 floats per param (m, c_kahan). v is recomputed each step.
//
// Naive memory cost is identical, but the Kahan compensator c is well-suited
// to BF16 storage (it's a small residual) whereas v needs more dynamic range.
// In a BF16-Adam-state regime, Kahan-bf16 gives ≈10x precision of plain v-bf16
// at the same byte cost — the design's "1.8 GB Adam savings" claim is the
// difference between Kahan-bf16 m + Kahan-bf16 c (4 bytes) vs full-Adam-bf16
// m + v + Kahan c (6 bytes). Here we stay in FP32 for unit-test scope.
//
// Reference: ASTRA #41 (Liu 2023 stateless variance), Kahan summation
// (Kahan 1965) for compensated accumulation.

inline void astra_kahan_step(float* param,
                             const float* grad,
                             float* m,
                             float* c_kahan,
                             unsigned int n,
                             float lr,
                             float beta1,
                             float eps,
                             float weightDecay)
{
	if (!param || !grad || !m || !c_kahan || n == 0u)
		return;
	const float oneMinusB1 = 1.0f - beta1;
	for (unsigned int i = 0; i < n; ++i)
	{
		const float g = grad[i];

		// Kahan-compensated update of m:
		//   m_t = β1·m_{t-1} + (1-β1)·g_t
		// Track residual lost-to-rounding via c_kahan.
		const float y = oneMinusB1 * g - c_kahan[i];
		const float mPrev = beta1 * m[i];
		const float t = mPrev + y;
		c_kahan[i] = (t - mPrev) - y;
		m[i] = t;

		// Stateless-v: v_t = g_t², recomputed each step.
		const float v = g * g;

		// AdamW-style param update (decoupled weight decay).
		const float denom = sqrtf(v) + eps;
		const float upd = lr * m[i] / denom;
		float p = param[i];
		if (weightDecay > 0.0f)
			p -= lr * weightDecay * p;
		p -= upd;
		param[i] = p;
	}
}

// Reference standard Adam (with full v EMA) for parity comparison.
inline void adam_step_reference(float* param,
                                const float* grad,
                                float* m,
                                float* v,
                                unsigned int n,
                                float lr,
                                float beta1,
                                float beta2,
                                float eps,
                                float weightDecay,
                                int step)
{
	if (!param || !grad || !m || !v || n == 0u || step <= 0)
		return;
	const float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(step));
	for (unsigned int i = 0; i < n; ++i)
	{
		const float g = grad[i];
		m[i] = beta1 * m[i] + (1.0f - beta1) * g;
		v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
		const float mhat = m[i] / bc1;
		const float vhat = v[i] / bc2;
		const float denom = sqrtf(vhat) + eps;
		const float upd = lr * mhat / denom;
		float p = param[i];
		if (weightDecay > 0.0f)
			p -= lr * weightDecay * p;
		p -= upd;
		param[i] = p;
	}
}

// ============================================================================
// Paradigm shift #99 — NEURAL-CACHE-COMPRESSION (Path 2 round 3)
// ============================================================================
//
// Extends #76 MLA's linear W_DKV projection with a 2-layer MLP:
//
//   c_t^KV = MLP_compress(h_t)
//          = (GELU(h_t · W_dc1 + b_dc1)) · W_dc2 + b_dc2
//   K_t    = MLP_decompress_K(c_t^KV)
//          = (GELU(c_t^KV · W_uk1 + b_uk1)) · W_uk2 + b_uk2
//   V_t    = MLP_decompress_V(c_t^KV)
//          = (GELU(c_t^KV · W_uv1 + b_uv1)) · W_uv2 + b_uv2
//
// Headline: target d_c=256 (vs MLA's 384 = ~1.5× more aggressive compression).
// Bijectivity preservation: c → K, V is deterministic; same hash → same K, V.

inline void neural_compress_2layer(const float* h,
                                   const float* W1, const float* b1,
                                   const float* W2, const float* b2,
                                   unsigned int T,
                                   unsigned int d_in,
                                   unsigned int d_hidden,
                                   unsigned int d_out,
                                   float* c_out)
{
	if (!h || !W1 || !W2 || !c_out) return;
	if (T == 0u || d_in == 0u || d_hidden == 0u || d_out == 0u) return;

	std::vector<float> hidden(static_cast<size_t>(T) * d_hidden);
	for (unsigned int t = 0; t < T; ++t)
	{
		const float* ht = h + static_cast<size_t>(t) * d_in;
		float* hidT = &hidden[static_cast<size_t>(t) * d_hidden];
		for (unsigned int j = 0; j < d_hidden; ++j)
		{
			double s = 0.0;
			for (unsigned int i = 0; i < d_in; ++i)
				s += static_cast<double>(ht[i]) *
				     static_cast<double>(W1[static_cast<size_t>(i) * d_hidden + j]);
			float pre = static_cast<float>(s);
			if (b1) pre += b1[j];
			hidT[j] = gelu(pre);
		}
		float* ct = c_out + static_cast<size_t>(t) * d_out;
		for (unsigned int k = 0; k < d_out; ++k)
		{
			double s = 0.0;
			for (unsigned int j = 0; j < d_hidden; ++j)
				s += static_cast<double>(hidT[j]) *
				     static_cast<double>(W2[static_cast<size_t>(j) * d_out + k]);
			float v = static_cast<float>(s);
			if (b2) v += b2[k];
			ct[k] = v;
		}
	}
}

// Determines effective compression ratio for #99 vs MHA baseline.
//   nHeads, dKV : standard MHA config
//   d_c         : neural-compressor bottleneck output dim
inline float neural_cache_compression_ratio(unsigned int nHeads,
                                            unsigned int dKV,
                                            unsigned int d_c)
{
	if (d_c == 0u) return 0.0f;
	const unsigned int mhaPerToken = nHeads * dKV * 2u;  // K + V floats per token
	return static_cast<float>(mhaPerToken) / static_cast<float>(d_c);
}

// ============================================================================
// Paradigm shift #77 — MOEFICATION-DISTILL-CHIRON (Path 2 round 5)
// ============================================================================
//
// Post-hoc Mixture-of-Experts: route per-token to top-k of E experts via
// router softmax, then compute weighted sum of selected experts' outputs.
// Per #77 design: typical config E=8, k=2 (active ratio = 0.25).
//
// Headline: 256B effective parameters at ~25% of #74-dense compute.
// (Mixtral 8x22B ≈ 141B params, ~39B active.)

// Top-k selection: writes top-k indices into out_indices and corresponding
// renormalized softmax weights into out_weights. Indices are in original
// router-logit positions; weights sum to 1 over the top-k subset.
//
//   logits:  [E]
//   k:       number of experts to keep (1 ≤ k ≤ E)
inline void moe_topk_router(const float* logits,
                            unsigned int E,
                            unsigned int k,
                            unsigned int* out_indices,
                            float* out_weights)
{
	if (!logits || !out_indices || !out_weights || E == 0u || k == 0u || k > E)
		return;

	// O(k * E) selection: simple but correct for E small (typical E=8).
	std::vector<bool> taken(E, false);
	for (unsigned int i = 0; i < k; ++i)
	{
		unsigned int best = E;
		float bestv = -1e30f;
		for (unsigned int e = 0; e < E; ++e)
		{
			if (taken[e]) continue;
			if (logits[e] > bestv) { bestv = logits[e]; best = e; }
		}
		out_indices[i] = best;
		taken[best] = true;
	}

	// Softmax over the selected k logits (renormalize within top-k).
	float maxL = logits[out_indices[0]];
	for (unsigned int i = 1; i < k; ++i)
		if (logits[out_indices[i]] > maxL) maxL = logits[out_indices[i]];
	double sum = 0.0;
	for (unsigned int i = 0; i < k; ++i)
	{
		const double e = exp(static_cast<double>(logits[out_indices[i]] - maxL));
		out_weights[i] = static_cast<float>(e);
		sum += e;
	}
	if (sum <= 0.0) return;
	const float inv = static_cast<float>(1.0 / sum);
	for (unsigned int i = 0; i < k; ++i) out_weights[i] *= inv;
}

// Compute the active-parameters fraction for top-k of E gating.
inline float moe_active_fraction(unsigned int E, unsigned int k)
{
	if (E == 0u) return 0.0f;
	return static_cast<float>(k) / static_cast<float>(E);
}

// Sparse mixture forward: given pre-computed expert outputs (one per expert,
// each [d_out]) and the top-k routing decision, write the weighted sum.
//
//   expert_outputs: [E * d_out]
//   indices:        [k]      from moe_topk_router
//   weights:        [k]      from moe_topk_router
//   y_out:          [d_out]  overwritten with Σ_{i in topk} w_i · expert_i(x)
//
// In production the K experts beyond the top-k would NOT be evaluated; the
// caller is responsible for that compute saving. This function just sums.
inline void moe_combine_topk_outputs(const float* expert_outputs,
                                     unsigned int E,
                                     unsigned int d_out,
                                     const unsigned int* indices,
                                     const float* weights,
                                     unsigned int k,
                                     float* y_out)
{
	if (!expert_outputs || !indices || !weights || !y_out || E == 0u || d_out == 0u || k == 0u)
		return;
	for (unsigned int d = 0; d < d_out; ++d) y_out[d] = 0.0f;
	for (unsigned int i = 0; i < k; ++i)
	{
		const unsigned int e = indices[i];
		if (e >= E) continue;
		const float w = weights[i];
		const float* ex = expert_outputs + static_cast<size_t>(e) * d_out;
		for (unsigned int d = 0; d < d_out; ++d)
			y_out[d] += w * ex[d];
	}
}

// ============================================================================
// Paradigm shift #73 — PHOENIX-1.58BIT (ternary weights, BitNet b1.58)
// ============================================================================
//
// Ternary {-1, 0, +1} weights at 1.58 bits per weight (log₂ 3). Two bits per
// weight in storage (with 1 bit padding) gives 16x compression vs FP32 on
// average. The intermediate "0" value provides sparsity not present in #74's
// binary {-1, +1}.
//
// Storage: 2 bits per weight, packed 4 weights per byte:
//   00 → 0
//   01 → +1
//   10 → -1
//   11 → reserved (unused)

inline void phoenix158_pack_ternary(const float* W,
                                    unsigned int n,
                                    unsigned char* out_2bit,
                                    float zero_threshold = 0.01f)
{
	if (!W || !out_2bit || n == 0u) return;
	const size_t bytes = (n + 3u) / 4u;
	for (size_t b = 0; b < bytes; ++b) out_2bit[b] = 0u;
	for (unsigned int i = 0; i < n; ++i)
	{
		unsigned char code = 0u;
		if (W[i] > zero_threshold) code = 1u;       // +1
		else if (W[i] < -zero_threshold) code = 2u; // -1
		// else 0 (encoded as 00)
		out_2bit[i >> 2] |= (unsigned char)(code << ((i & 3u) << 1u));
	}
}

inline void phoenix158_unpack_ternary(const unsigned char* in_2bit,
                                      unsigned int n,
                                      float* W_out)
{
	if (!in_2bit || !W_out || n == 0u) return;
	for (unsigned int i = 0; i < n; ++i)
	{
		const unsigned char code = (in_2bit[i >> 2] >> ((i & 3u) << 1u)) & 3u;
		if (code == 1u) W_out[i] = 1.0f;
		else if (code == 2u) W_out[i] = -1.0f;
		else W_out[i] = 0.0f;
	}
}

// Compression ratio for #73 vs FP32 (1.58 bits per weight in entropy; 2 bits
// in our packed storage, so the practical ratio is 32/2 = 16x).
inline float phoenix158_compression_ratio_fp32()
{
	return 16.0f;  // 32 bits FP32 / 2 bits packed
}

inline float phoenix158_compression_ratio_bf16()
{
	return 8.0f;   // 16 bits BF16 / 2 bits packed
}

// Ternary GEMM: Y = X · W_ternary. Per-output algorithm:
//   y[m,n] = Σ_{k : W[k,n] = +1} X[m,k] - Σ_{k : W[k,n] = -1} X[m,k]
// Zero-coded weights skip naturally (no contribution).
inline void phoenix158_ternary_gemm(const float* X,
                                    const unsigned char* W_2bit,
                                    unsigned int M,
                                    unsigned int N,
                                    unsigned int K,
                                    float* Y)
{
	if (!X || !W_2bit || !Y || M == 0u || N == 0u || K == 0u) return;
	for (unsigned int m = 0; m < M; ++m)
	{
		const float* xm = X + static_cast<size_t>(m) * K;
		float* ym = Y + static_cast<size_t>(m) * N;
		for (unsigned int n = 0; n < N; ++n)
		{
			float pos = 0.0f;
			float neg = 0.0f;
			for (unsigned int k = 0; k < K; ++k)
			{
				const size_t bitIdx = static_cast<size_t>(k) * static_cast<size_t>(N) +
				                      static_cast<size_t>(n);
				const unsigned char code = (W_2bit[bitIdx >> 2] >> ((bitIdx & 3u) << 1u)) & 3u;
				if (code == 1u) pos += xm[k];
				else if (code == 2u) neg += xm[k];
			}
			ym[n] = pos - neg;
		}
	}
}

// Ternary sparsity: count zero-weight fraction (not just nominal "skip").
inline float phoenix158_zero_fraction(const unsigned char* W_2bit, unsigned int n)
{
	if (!W_2bit || n == 0u) return 0.0f;
	unsigned int zeros = 0u;
	for (unsigned int i = 0; i < n; ++i)
	{
		const unsigned char code = (W_2bit[i >> 2] >> ((i & 3u) << 1u)) & 3u;
		if (code == 0u) ++zeros;
	}
	return static_cast<float>(zeros) / static_cast<float>(n);
}

} // namespace transformer_ops
} // namespace glades
