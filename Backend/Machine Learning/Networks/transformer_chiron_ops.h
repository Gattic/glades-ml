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

// ------------------------------------------------------------------
// Sketch residual correction (framework §4.4).
//
// To prevent BF16 round-off from compounding across the block-inverse
// chain, each forward pass stores a tiny FP32 "sketch" z_ℓ = S_ℓ · x_ℓ,
// where S_ℓ ∈ R^{r × N} is a Gaussian random matrix and x_ℓ is the flat
// block-input state of size N (typically N = 2·T·m for the paired (q, p)
// hidden state). During backward, after running the BF16 inverse to get
// an approximation x̃_ℓ, the stored sketch z_ℓ is compared against
// S_ℓ · x̃_ℓ and the residual is lifted back into x-space to correct
// x̃_ℓ:
//
//   z_ℓ_fresh   = S_ℓ · vec(x̃_ℓ)
//   residual    = z_ℓ − z_ℓ_fresh
//   correction  = (S_ℓ^T / r) · residual
//   x̂_ℓ        = x̃_ℓ + correction
//
// Theorem (unbiasedness). If S_ℓ has i.i.d. N(0, 1) entries and S_ℓ is
// independent of (x_ℓ, x̃_ℓ), then E[x̂_ℓ] = x_ℓ. Variance of each
// coordinate of x̂_ℓ is O(||x − x̃||² / r).
//
// Proof sketch. E[S_ℓ^T S_ℓ] = r · I_N (each entry is a sum of r
// independent mean-zero products of variance 1). Thus
// E[(S_ℓ^T S_ℓ / r) · (x − x̃)] = x − x̃ identically.
//
// The sketch matrix can be seeded (e.g., hash of layer index) so neither
// S_ℓ nor the random seed needs to be stored beyond a short seed word
// per layer. This prototype stores S_ℓ explicitly for clarity; GPU
// production kernels should regenerate S_ℓ on-the-fly per layer from
// a deterministic seed.

// sketch_project: compute z = S · x, where S is [r, N] row-major,
// x is [N] row-major.  Output z is [r] row-major.
inline void sketch_project(const float* S, const float* x,
                           unsigned int r, unsigned int N, float* z_out)
{
	for (unsigned int k = 0; k < r; ++k)
	{
		float acc = 0.0f;
		const float* row = S + k * N;
		for (unsigned int i = 0; i < N; ++i)
			acc += row[i] * x[i];
		z_out[k] = acc;
	}
}

// sketch_lift: compute correction = (S^T / r) · residual, adding in-place
// to x_out.  S is [r, N], residual is [r], x_out is [N].
//
// This is the minimum-norm lift of residual back to the full state space;
// E[correction] = (1/r) E[S^T S] (x - x̃) = (x - x̃) by the identity
// above.
inline void sketch_lift_add(const float* S, const float* residual,
                            unsigned int r, unsigned int N, float* x_out)
{
	const float inv_r = 1.0f / static_cast<float>(r);
	for (unsigned int i = 0; i < N; ++i)
	{
		float acc = 0.0f;
		for (unsigned int k = 0; k < r; ++k)
			acc += S[k * N + i] * residual[k];
		x_out[i] += acc * inv_r;
	}
}

// -------------- Batched + tiled CPU variants (Phase 3.5 optimized) --------
//
// The per-token sketch ops above are correct but CPU-inefficient: they load
// each row of S once per token, yielding ~3 GFLOP/s on modern x86 because
// the inner loop is memory-bound and not vectorizable across tokens.
//
// The batched variants process all T tokens in one blocked SIMD-friendly
// matrix multiply:
//   - sketch_project_batched: Z [T, r] = X [T, N] · S^T [N, r]
//   - sketch_lift_add_batched: X [T, N] += (1/r) · Z [T, r] · S [r, N]
//
// Tiling: T/M_TILE outer, r or N inner. The `K_TILE` inner loop accumulates
// FMAs with S streamed into L1. On a 4-wide SSE or 8-wide AVX host, GCC can
// auto-vectorize the inner K loop; we keep the loop structure compiler-
// friendly (contiguous strided loads, known trip counts via tile bounds).

#ifndef GLADES_CHIRON_TILE_M
#define GLADES_CHIRON_TILE_M 8
#endif
#ifndef GLADES_CHIRON_TILE_N
#define GLADES_CHIRON_TILE_N 32
#endif
#ifndef GLADES_CHIRON_TILE_K
#define GLADES_CHIRON_TILE_K 64
#endif

// Z [T, r] = X [T, N] · S^T [N, r]   (i.e. Z[t, k] = Σ_i X[t, i] * S[k, i])
//
// Performance: blocked tiling on (M, N, K) plus an unrolled inner FMA loop.
// Outer M dim is the token index (trivially independent) — can be parallelized
// by the caller via OpenMP around the mb loop. We keep the function itself
// serial-safe to retain header-only reusability.
inline void sketch_project_batched(const float* S, const float* X,
                                    unsigned int T, unsigned int N, unsigned int r,
                                    float* Z)
{
	// Zero output.
	const size_t total = static_cast<size_t>(T) * static_cast<size_t>(r);
	for (size_t i = 0; i < total; ++i) Z[i] = 0.0f;

	const unsigned int TM = GLADES_CHIRON_TILE_M;
	const unsigned int TN = GLADES_CHIRON_TILE_N; // tile over r (output cols)
	const unsigned int TK = GLADES_CHIRON_TILE_K;

	for (unsigned int mb = 0; mb < T; mb += TM)
	{
		const unsigned int me = (mb + TM < T) ? (mb + TM) : T;
		for (unsigned int nb = 0; nb < r; nb += TN)
		{
			const unsigned int ne = (nb + TN < r) ? (nb + TN) : r;
			for (unsigned int kb = 0; kb < N; kb += TK)
			{
				const unsigned int ke = (kb + TK < N) ? (kb + TK) : N;

				// Inner: Z[mb:me, nb:ne] += X[mb:me, kb:ke] · S[nb:ne, kb:ke]^T.
				// Token-major outer: better Z locality across the k loop.
				for (unsigned int t = mb; t < me; ++t)
				{
					const float* xRow = X + static_cast<size_t>(t) * N;
					float*       zRow = Z + static_cast<size_t>(t) * r;
					for (unsigned int k = nb; k < ne; ++k)
					{
						const float* sRow = S + static_cast<size_t>(k) * N;
						// Accumulate into four parallel lanes so the compiler
						// can emit 4-wide FMAs even without SIMD intrinsics.
						float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
						unsigned int i = kb;
						const unsigned int ke4 = kb + ((ke - kb) & ~3u);
						for (; i < ke4; i += 4)
						{
							acc0 += xRow[i + 0] * sRow[i + 0];
							acc1 += xRow[i + 1] * sRow[i + 1];
							acc2 += xRow[i + 2] * sRow[i + 2];
							acc3 += xRow[i + 3] * sRow[i + 3];
						}
						float acc = acc0 + acc1 + acc2 + acc3;
						for (; i < ke; ++i) acc += xRow[i] * sRow[i];
						zRow[k] += acc;
					}
				}
			}
		}
	}
}

// X [T, N] += (1/r) · Z [T, r] · S [r, N]  (i.e. X[t, i] += (1/r) · Σ_k Z[t, k] · S[k, i])
//
// Loop order: outer-product with k outermost inside a token.  Each row of
// S[k, :] is loaded once and multiplied by Z[t, k] into a contiguous stripe
// of X[t, :].  This gives SEQUENTIAL access to both X[t, :] and S[k, :],
// which is dramatically more cache-friendly than the sum-k-over-stride-N
// access pattern of a straightforward dot-product formulation.
inline void sketch_lift_add_batched(const float* S, const float* Z,
                                     unsigned int T, unsigned int N, unsigned int r,
                                     float* X)
{
	const float inv_r = 1.0f / static_cast<float>(r);
	const unsigned int TM = GLADES_CHIRON_TILE_M;
	const unsigned int TN = GLADES_CHIRON_TILE_N;
	const unsigned int TK = GLADES_CHIRON_TILE_K;

	for (unsigned int mb = 0; mb < T; mb += TM)
	{
		const unsigned int me = (mb + TM < T) ? (mb + TM) : T;
		for (unsigned int nb = 0; nb < N; nb += TN)
		{
			const unsigned int ne = (nb + TN < N) ? (nb + TN) : N;
			for (unsigned int kb = 0; kb < r; kb += TK)
			{
				const unsigned int ke = (kb + TK < r) ? (kb + TK) : r;

				for (unsigned int t = mb; t < me; ++t)
				{
					float*       xRow = X + static_cast<size_t>(t) * N;
					const float* zRow = Z + static_cast<size_t>(t) * r;
					// Outer over k: each k-row of S is contiguous in i.
					for (unsigned int k = kb; k < ke; ++k)
					{
						const float* sRow = S + static_cast<size_t>(k) * N;
						const float  scale = inv_r * zRow[k];
						unsigned int i = nb;
						const unsigned int ne4 = nb + ((ne - nb) & ~3u);
						for (; i < ne4; i += 4)
						{
							xRow[i + 0] += scale * sRow[i + 0];
							xRow[i + 1] += scale * sRow[i + 1];
							xRow[i + 2] += scale * sRow[i + 2];
							xRow[i + 3] += scale * sRow[i + 3];
						}
						for (; i < ne; ++i)
							xRow[i] += scale * sRow[i];
					}
				}
			}
		}
	}
}

} // namespace chiron
} // namespace glades
