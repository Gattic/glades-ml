// CHIRON reversible-flow building blocks.
//
// CHIRON ("Canonical Hamiltonian Involutive Reversible Operator Network")
// expresses a transformer block as a composition of symplectic diffeomorphisms
// on a paired hidden state (q, p) ∈ R^{T × m} × R^{T × m}, giving an explicit
// block inverse that reconstructs activations from outputs without storing
// them. See research/CHIRON_framework.md for the full derivation.
//
// This header is the CPU reference implementation. It is header-only and
// C++98-compatible, matching the style of transformer_ops.h.
//
// Building blocks:
//   - Shear^p (momentum kick):   (q, p) → (q, p + u)   for u a pure function of q
//   - Shear^q (position drift):  (q, p) → (q + u, p)   for u a pure function of p
//   - ReLN forward/inverse:      reversible LayerNorm on q with (μ, log σ)
//                                written to a caller-provided external stats
//                                buffer (primary API) OR into reserved coords
//                                of p (future optimization).
//
// Invariants (block inverses rely on these):
//   - All shear updates must be a pure function of the OPPOSITE branch only.
//   - For the external-stats ReLN, the caller sizes an [L, T, 2] FP32 buffer
//     and passes a distinct slice per block. Memory cost is 2 FP32 per token
//     per block — negligible compared to the O(L·T·d) activations CHIRON
//     eliminates.
//
// Convention:
//   - Tensors are row-major, flattened as [T, m].
//   - All ops work per-token; the token-axis loop is explicit in the caller.

#pragma once

#include <cmath>
#include <cstddef>

namespace glades {
namespace chiron {

// Shear^p: p += u. u must be a pure function of q (not of p).
// Forward: p[i] += u[i] for i in [0, m).
inline void shear_add_to_p(float* p, const float* u, unsigned int m)
{
	for (unsigned int i = 0; i < m; ++i)
		p[i] += u[i];
}

// Shear^p inverse: p -= u. Caller must pass the SAME u as forward (re-computed
// from q_new which equals q_old in the shear, so recomputation is exact).
inline void shear_sub_from_p(float* p, const float* u, unsigned int m)
{
	for (unsigned int i = 0; i < m; ++i)
		p[i] -= u[i];
}

// Shear^q: q += u. u must be a pure function of p (not of q).
inline void shear_add_to_q(float* q, const float* u, unsigned int m)
{
	for (unsigned int i = 0; i < m; ++i)
		q[i] += u[i];
}

inline void shear_sub_from_q(float* q, const float* u, unsigned int m)
{
	for (unsigned int i = 0; i < m; ++i)
		q[i] -= u[i];
}

// Zero reserved coordinates on a token-row vector. Used to enforce the
// invariant that nonlinear maps do not write to reserved coords.
inline void zero_reserved_coords_row(float* v, unsigned int r0, unsigned int r1)
{
	v[r0] = 0.0f;
	v[r1] = 0.0f;
}

// Zero reserved coordinates across all T tokens of a row-major [T, m] tensor.
inline void zero_reserved_coords_all_tokens(float* v, unsigned int T, unsigned int m,
                                            unsigned int r0, unsigned int r1)
{
	for (unsigned int t = 0; t < T; ++t)
	{
		v[t * m + r0] = 0.0f;
		v[t * m + r1] = 0.0f;
	}
}

// Reversible LayerNorm (ReLN), per-token forward — "external stats" variant.
//
// This is the primary ReLN API used by the prototype: stats (μ, log σ) are
// written to a caller-provided `stats_out[2]` buffer rather than packed into
// reserved coordinates of p. This decouples the ReLN round-trip from the
// per-block channel-permutation bookkeeping and makes composition trivially
// block-local: the caller sizes a `[L, T, 2]` stats buffer, block ℓ writes
// to its own slice, and the inverse reads from that slice.
//
// Given:
//   q_in        row-major [m], input q coordinates of one token
//   gamma, beta row-major [m], LN affine parameters
//   m           hidden width per branch (= dModel / 2 in CHIRON)
//   eps         numerical epsilon added to variance
// Writes:
//   q_out       row-major [m], normalized + affine-transformed q
//   stats_out   [2], { μ, log(σ) } for this token
//
// Math:
//   μ   = (1/m) Σ q_in[i]
//   σ²  = (1/m) Σ (q_in[i] − μ)² + eps
//   σ   = √σ²
//   q_out[i] = gamma[i] * (q_in[i] − μ) / σ + beta[i]
//
// Memory cost per block: 2 FP32 per token. For L=96, T=4096 that is
// ~3 MB total, which is negligible compared to the O(L·T·d) activations
// that CHIRON eliminates.
inline void reln_forward_row(const float* q_in, float* q_out, float* stats_out,
                             const float* gamma, const float* beta,
                             unsigned int m, float eps)
{
	// Mean.
	double sum = 0.0;
	for (unsigned int i = 0; i < m; ++i)
		sum += q_in[i];
	const float mu = static_cast<float>(sum / static_cast<double>(m));

	// Variance.
	double var_sum = 0.0;
	for (unsigned int i = 0; i < m; ++i)
	{
		const float d = q_in[i] - mu;
		var_sum += static_cast<double>(d) * static_cast<double>(d);
	}
	const float var = static_cast<float>(var_sum / static_cast<double>(m)) + eps;
	const float sigma = sqrtf(var);
	const float inv_sigma = 1.0f / sigma;

	// Normalize + affine.
	for (unsigned int i = 0; i < m; ++i)
		q_out[i] = gamma[i] * (q_in[i] - mu) * inv_sigma + beta[i];

	// Record (μ, log σ) into the caller-owned stats buffer for this token.
	stats_out[0] = mu;
	stats_out[1] = logf(sigma);
}

// Reversible LayerNorm inverse, per-token — "external stats" variant.
//
// Given the outputs of reln_forward_row:
//   q_out       [m], post-ReLN q
//   stats_in    [2], { μ, log σ } recorded at forward time
//   gamma, beta [m], same as forward
// Writes:
//   q_in        [m], recovered original q (bit-exact in FP32)
//
// Math:
//   μ          = stats_in[0]
//   log σ      = stats_in[1]
//   σ          = exp(log σ)
//   q_in[i]    = σ · (q_out[i] − beta[i]) / gamma[i]  +  μ
inline void reln_inverse_row(const float* q_out, float* q_in, const float* stats_in,
                             const float* gamma, const float* beta, unsigned int m)
{
	const float mu = stats_in[0];
	const float log_sigma = stats_in[1];
	const float sigma = expf(log_sigma);

	for (unsigned int i = 0; i < m; ++i)
	{
		// gamma[i] may be tuned small; caller should avoid gamma[i] ≈ 0.
		q_in[i] = sigma * (q_out[i] - beta[i]) / gamma[i] + mu;
	}
}

// Whole-tensor wrappers for a [T, m] q tensor and a [T, 2] stats tensor.
// Convenience for the block-level forward/inverse.
inline void reln_forward(const float* q_in, float* q_out, float* stats_out,
                         const float* gamma, const float* beta,
                         unsigned int T, unsigned int m, float eps)
{
	for (unsigned int t = 0; t < T; ++t)
	{
		reln_forward_row(q_in + t * m, q_out + t * m, stats_out + t * 2u,
		                 gamma, beta, m, eps);
	}
}

inline void reln_inverse(const float* q_out, float* q_in, const float* stats_in,
                         const float* gamma, const float* beta,
                         unsigned int T, unsigned int m)
{
	for (unsigned int t = 0; t < T; ++t)
	{
		reln_inverse_row(q_out + t * m, q_in + t * m, stats_in + t * 2u,
		                 gamma, beta, m);
	}
}

} // namespace chiron
} // namespace glades
