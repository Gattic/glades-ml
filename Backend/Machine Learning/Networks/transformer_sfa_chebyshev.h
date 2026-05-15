// SFA Chebyshev solver — applies (L_F + lambda*I)^{-1} or exp(-tau*L_F) to a
// source vector via M-degree Chebyshev polynomial recurrence on the sheaf
// Laplacian, with optional diagonal-Jacobi preconditioning.
//
// Paradigm #250 from research/PARADIGM_SHIFT_250_DESIGN.md §5.
// Theoretical analysis in research/PARADIGM_SHIFT_250_PROOFS.md §3.
//
// Theorem 3 (info-loss bound, iter 2 §3) shows that without preconditioning,
// L_F's condition number κ = (μ_max + λ) / λ = O(10^6) and Chebyshev M=8
// gives ε ≈ 0.98 — insufficient. With diagonal-Jacobi preconditioning,
// κ reduces to ~30, and M=8 gives ε ≈ 0.055 — borderline acceptable.
// M=16 gives ε ≈ 3e-3 — solid.
//
// This is a CPU prototype. GPU port will use cuBLAS for the inner GEMV and
// fuse the recurrence steps per ATLAS-COMPILE (paradigm #51).
//
// Conventions match transformer_sfa_ops.h.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "transformer_sfa_ops.h"

namespace glades
{
namespace transformer_sfa_chebyshev
{

using transformer_sfa_ops::SFAParams;
using transformer_sfa_ops::laplacianMatvec;

// Compute the block-Jacobi preconditioner D_F (the block-diagonal of L_F).
//
// For SFA: [L_F]_{ii} = sum over edges incident to i of R^T R contributions.
// We approximate D_F by its scalar trace per token: D_F_ii ≈ (deg_i + sum_j ||Sigma||^2).
//
// Returns inv_diag[T] where inv_diag[i] = 1 / D_F_ii (clamped to [1e-3, 1e3] for stability).
inline void computeJacobiPreconditioner(const SFAParams& p,
                                         std::vector<float>& inv_diag)
{
	const int T = p.T;
	const int r = p.r;
	const int E = static_cast<int>(p.edge_src.size());

	std::vector<float> diag(T, 0.0f);
	std::vector<int> deg_in(T, 0);
	std::vector<int> deg_out(T, 0);

	for (int e = 0; e < E; ++e)
	{
		const int i = p.edge_src[e];
		const int j = p.edge_tgt[e];
		deg_out[i]++;
		deg_in[j]++;

		// Sigma^2 contribution: each diagonal entry's square sums into D_F_ii.
		const float* Sigma_e = &p.Sigma[static_cast<size_t>(e) * r];
		float sigma_sq = 0.0f;
		for (int beta = 0; beta < r; ++beta)
		{
			sigma_sq += Sigma_e[beta] * Sigma_e[beta];
		}
		diag[i] += sigma_sq;  // R_{j<-i}^T R_{j<-i} contributes to ii block
	}

	// Add deg-I contribution (each edge incident to i adds I_{d_s} ≈ scalar 1).
	for (int i = 0; i < T; ++i)
	{
		diag[i] += static_cast<float>(deg_in[i] + deg_out[i]);
	}

	// Invert with stability clamping.
	inv_diag.resize(T);
	for (int i = 0; i < T; ++i)
	{
		const float d = std::max(diag[i] + 1e-3f, 1e-3f);
		inv_diag[i] = 1.0f / std::min(d, 1e3f);
	}
}

// Apply preconditioner D_F^{-1} to a section s (scalar-per-token).
inline void applyPreconditioner(const SFAParams& p,
                                 const std::vector<float>& inv_diag,
                                 float* s)
{
	const int T = p.T;
	const int d_s = p.d_s;
	for (int i = 0; i < T; ++i)
	{
		const float scale = inv_diag[i];
		float* s_i = &s[static_cast<size_t>(i) * d_s];
		for (int a = 0; a < d_s; ++a)
		{
			s_i[a] *= scale;
		}
	}
}

// Estimate mu_max of the preconditioned operator D_F^{-1} L_F via power iteration.
inline float estimatePreconditionedMuMax(const SFAParams& p,
                                          const std::vector<float>& inv_diag,
                                          int n_iters,
                                          unsigned int seed)
{
	const int Tds = p.T * p.d_s;
	std::vector<float> v(Tds);
	std::vector<float> Lv(Tds);

	unsigned int state = seed ? seed : 0xC0FFEEu;
	for (int k = 0; k < Tds; ++k)
	{
		state = state * 1664525u + 1013904223u;
		v[k] = ((state >> 16) & 0xFFFFu) / 65535.0f - 0.5f;
	}

	float norm = 0.0f;
	for (int k = 0; k < Tds; ++k)
		norm += v[k] * v[k];
	norm = std::sqrt(norm);
	const float inv_norm = 1.0f / std::max(norm, 1e-12f);
	for (int k = 0; k < Tds; ++k)
		v[k] *= inv_norm;

	float lambda_est = 0.0f;
	for (int it = 0; it < n_iters; ++it)
	{
		// Lv = L_F v, then Lv = D_F^{-1} Lv
		laplacianMatvec(p, &v[0], &Lv[0]);
		applyPreconditioner(p, inv_diag, &Lv[0]);

		float dot = 0.0f;
		float Lnorm = 0.0f;
		for (int k = 0; k < Tds; ++k)
		{
			dot += v[k] * Lv[k];
			Lnorm += Lv[k] * Lv[k];
		}
		Lnorm = std::sqrt(Lnorm);
		lambda_est = dot;
		const float inv2 = 1.0f / std::max(Lnorm, 1e-12f);
		for (int k = 0; k < Tds; ++k)
			v[k] = Lv[k] * inv2;
	}

	return lambda_est * 1.10f;
}

// Chebyshev coefficients for f(x) = 1 / (mu_max * 0.5*(x + 1) + lambda) on x in [-1, 1].
// I.e., we map L_F → A = 2/mu_max * L_F - I so A has eigenvalues in [-1, 1] approximately
// (assuming L_F's spectrum is in [0, mu_max] after preconditioning).
//
// Coefficients computed via the Chebyshev-T inner product:
//   c_n = (2/π) ∫_{-1}^{1} f(x) T_n(x) / sqrt(1 - x^2) dx
// approximated by Chebyshev nodes (Clenshaw-Curtis quadrature).
inline void computeChebyshevCoeffs(float mu_max, float lambda,
                                    int M,
                                    std::vector<float>& coeffs)
{
	coeffs.assign(M, 0.0f);
	const int N_quad = 4 * M;  // Quadrature points for accurate inner product.
	const float pi = 3.14159265358979323846f;

	// Chebyshev-T nodes: x_k = cos((2k+1) π / 2N), k=0..N-1.
	for (int n = 0; n < M; ++n)
	{
		double sum = 0.0;
		for (int k = 0; k < N_quad; ++k)
		{
			const double theta = (2 * k + 1) * pi / (2 * N_quad);
			const double x = std::cos(theta);
			const double T_n = std::cos(n * theta);
			// Map x ∈ [-1, 1] back to the original L_F-spectrum: mu = mu_max * 0.5 * (x + 1).
			const double mu = static_cast<double>(mu_max) * 0.5 * (x + 1.0);
			const double f = 1.0 / (mu + static_cast<double>(lambda));
			sum += f * T_n;
		}
		// Standard Chebyshev coefficient normalisation.
		const double norm = (n == 0) ? 1.0 / static_cast<double>(N_quad)
		                              : 2.0 / static_cast<double>(N_quad);
		coeffs[n] = static_cast<float>(sum * norm);
	}
}

// Solve (L_F + lambda*I) s = b using M-degree Chebyshev approximation of
// the Tikhonov-regularised inverse, with optional diagonal-Jacobi preconditioning.
//
// CHEBYSHEV RECURRENCE: Clenshaw's BACKWARD recurrence (numerically stable).
// The forward recurrence w_n = 2 A w_{n-1} - w_{n-2} amplifies round-off and
// fails beyond M ~= 8 in FP32 (validated empirically iter 14).
//
// Clenshaw backward recurrence for p(A) v = sum_{n=0}^{N} c_n T_n(A) v:
//   d_{N+1} = d_{N+2} = 0      (zero vectors)
//   For k = N down to 1:
//     d_k = 2 A d_{k+1} - d_{k+2} + c_k v
//   p(A) v = A d_1 - d_2 + c_0 v
//
// This is stable because the c_k * v term is added at each step rather than
// accumulating into the running w_n vector.
//
// Algorithm:
//   1. Optionally apply diag-Jacobi preconditioner D_F^{-1} to b.
//   2. Map: A = (2/mu_max) L_F - I, so spec(A) approx in [-1, 1].
//   3. Clenshaw backward recurrence with coefficients c_n for f(mu) = 1/(mu + lambda).
//
// Cost: M matvecs (each O(|E| * d_s * r)) + M axpy (each O(T * d_s)).
//
// Result placed in `result`.
inline void solveTikhonov(const SFAParams& p,
                           const float* b,
                           int M,
                           bool use_preconditioner,
                           float* result)
{
	const int T = p.T;
	const int d_s = p.d_s;
	const int Tds = T * d_s;

	// Step 1: preconditioning.
	std::vector<float> inv_diag;
	std::vector<float> b_precond(Tds);
	if (use_preconditioner)
	{
		computeJacobiPreconditioner(p, inv_diag);
		std::memcpy(&b_precond[0], b, sizeof(float) * Tds);
		applyPreconditioner(p, inv_diag, &b_precond[0]);
	}
	const float* b_eff = use_preconditioner ? &b_precond[0] : b;

	// Step 2: estimate mu_max (preconditioned spectrum).
	// Use 20 power iterations (converges for typical L_F spectra) with 25%
	// safety margin to ensure A = (2/mu_max) L_F - I has spectrum strictly
	// in [-1, 1]. Underestimating mu_max causes Chebyshev/Clenshaw divergence
	// since |T_n(x)| grows exponentially for |x| > 1.
	float mu_max;
	if (use_preconditioner)
		mu_max = estimatePreconditionedMuMax(p, inv_diag, 20, 0xC0FFEEu);
	else
		mu_max = transformer_sfa_ops::estimateMuMax(p, 20, 0xC0FFEEu);
	mu_max = std::max(mu_max * 1.25f, 1e-3f);  // 25% safety margin on top of estimateMuMax's 10%

	// Step 3: Chebyshev coefficients.
	std::vector<float> coeffs;
	computeChebyshevCoeffs(mu_max, p.lambda, M, coeffs);

	// Step 4: Clenshaw backward recurrence (stable for arbitrary M).
	// Compute p(A) v where p(A) = sum_{k=0}^{M-1} c_k T_k(A), v = b_eff.
	//
	// Backward recurrence:
	//   d_M = d_{M+1} = 0
	//   For k = M-1 down to 1:
	//     d_k = 2 A d_{k+1} - d_{k+2} + c_k v
	//   result = A d_1 - d_2 + c_0 v
	//
	// Each iter does one L_F matvec + axpy operations.
	const float two_over_mu = 2.0f / mu_max;

	std::vector<float> d_curr(Tds, 0.0f);      // d_{k+1}, initially 0 (d_M)
	std::vector<float> d_next(Tds, 0.0f);      // d_{k+2}, initially 0 (d_{M+1})
	std::vector<float> d_new(Tds);             // working d_k
	std::vector<float> tmp(Tds);

	// Loop k = M-1 down to 1.
	for (int k = M - 1; k >= 1; --k)
	{
		// d_new = 2 A d_curr - d_next + c_k * b_eff
		// where A = (2/mu_max) L_F - I (preconditioned if requested).
		//
		// 2 A d_curr = (4/mu_max) L_F d_curr - 2 d_curr
		laplacianMatvec(p, &d_curr[0], &tmp[0]);
		if (use_preconditioner)
			applyPreconditioner(p, inv_diag, &tmp[0]);
		const float four_over_mu = 4.0f / mu_max;
		const float ck = coeffs[k];
		for (int idx = 0; idx < Tds; ++idx)
			d_new[idx] = four_over_mu * tmp[idx] - 2.0f * d_curr[idx]
			             - d_next[idx] + ck * b_eff[idx];

		// rotate: d_{k+2} ← d_{k+1}, d_{k+1} ← d_k
		std::swap(d_next, d_curr);
		std::swap(d_curr, d_new);
	}

	// Final: result = A d_1 - d_2 + c_0 b_eff
	//                = (2/mu_max) L_F d_1 - d_1 - d_2 + c_0 b_eff
	laplacianMatvec(p, &d_curr[0], &tmp[0]);
	if (use_preconditioner)
		applyPreconditioner(p, inv_diag, &tmp[0]);
	const float c0 = coeffs[0];
	for (int idx = 0; idx < Tds; ++idx)
		result[idx] = two_over_mu * tmp[idx] - d_curr[idx]
		              - d_next[idx] + c0 * b_eff[idx];
}

// Compute the residual ||r|| / ||b|| for diagnostic / Probe D purposes.
//   r = (L_F + lambda * I) s - b
inline float residualNorm(const SFAParams& p,
                           const float* b,
                           const float* s)
{
	const int T = p.T;
	const int d_s = p.d_s;
	const int Tds = T * d_s;

	std::vector<float> Ls(Tds);
	laplacianMatvec(p, s, &Ls[0]);

	float r_norm = 0.0f;
	float b_norm = 0.0f;
	for (int k = 0; k < Tds; ++k)
	{
		const float r = Ls[k] + p.lambda * s[k] - b[k];
		r_norm += r * r;
		b_norm += b[k] * b[k];
	}
	r_norm = std::sqrt(r_norm);
	b_norm = std::sqrt(b_norm);
	return r_norm / std::max(b_norm, 1e-12f);
}

}  // namespace transformer_sfa_chebyshev
}  // namespace glades
