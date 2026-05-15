// SFA smoke test — standalone executable verifying that the CPU prototype
// primitives compile and produce sensible numerical output.
//
// This is NOT yet a Gate-0 probe. It is a "does the code work at all?" test:
// build a small synthetic SFA layer, run forward, and check that:
//   1. Edge set has correct count.
//   2. L_F matvec is symmetric and PSD on test vectors.
//   3. Chebyshev solve gives a small residual.
//   4. Trivial sheaf case (R = I) reduces to graph-Laplacian solve.
//
// Compile (standalone):
//   g++ -std=c++98 -O2 -I.. -I/path/to/shmea \
//       transformer_sfa_smoke_test.cpp -o sfa_smoke_test
//
// Or via CMake (iter 15 will add integration). For iter 14, this is intentionally
// dependency-light: pure CPU, no shmea, no glades runtime — just the SFA primitives.
//
// Expected output: 4 PASS lines and one summary. Wall-clock < 1 second.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>

#include "transformer_sfa_ops.h"
#include "transformer_sfa_chebyshev.h"

using glades::transformer_sfa_ops::SFAParams;
using glades::transformer_sfa_ops::buildEdgeSet;
using glades::transformer_sfa_ops::laplacianMatvec;
using glades::transformer_sfa_ops::sourceAssembly;
using glades::transformer_sfa_ops::readout;
using glades::transformer_sfa_chebyshev::solveTikhonov;
using glades::transformer_sfa_chebyshev::residualNorm;

namespace
{
unsigned int g_state = 0x12345678u;

float urand(float a, float b)
{
	g_state = g_state * 1664525u + 1013904223u;
	const float u = ((g_state >> 16) & 0xFFFFu) / 65535.0f;
	return a + u * (b - a);
}

// Build a small SFA parameter set with random U and unit-ish Sigma.
void buildSyntheticSFAParams(SFAParams& p,
                              int T, int d_s, int d_h, int r, int W, int n_sinks)
{
	p.T = T;
	p.d_s = d_s;
	p.d_h = d_h;
	p.r = r;
	p.W = W;
	p.n_sinks = n_sinks;
	p.lambda = 1e-2f;
	p.gamma = 0.0f;  // no value injection in smoke test

	// Edge set.
	buildEdgeSet(T, W, n_sinks, p.edge_src, p.edge_tgt);
	const int E = static_cast<int>(p.edge_src.size());

	// Random stalk frames U[T * d_s * r]. Init to small orthogonal-ish columns.
	p.U.assign(static_cast<size_t>(T) * d_s * r, 0.0f);
	for (int i = 0; i < T; ++i)
	{
		// Gram-Schmidt over r columns.
		for (int beta = 0; beta < r; ++beta)
		{
			for (int a = 0; a < d_s; ++a)
			{
				p.U[i * d_s * r + a * r + beta] = urand(-1.0f, 1.0f);
			}
			// Orthogonalise against previous columns.
			for (int prev = 0; prev < beta; ++prev)
			{
				float dot = 0.0f;
				for (int a = 0; a < d_s; ++a)
				{
					dot += p.U[i * d_s * r + a * r + beta] * p.U[i * d_s * r + a * r + prev];
				}
				for (int a = 0; a < d_s; ++a)
				{
					p.U[i * d_s * r + a * r + beta] -= dot * p.U[i * d_s * r + a * r + prev];
				}
			}
			// Normalise.
			float norm = 0.0f;
			for (int a = 0; a < d_s; ++a)
			{
				float v = p.U[i * d_s * r + a * r + beta];
				norm += v * v;
			}
			norm = std::sqrt(std::max(norm, 1e-12f));
			for (int a = 0; a < d_s; ++a)
			{
				p.U[i * d_s * r + a * r + beta] /= norm;
			}
		}
	}

	// Sigma: random in [0.5, 1.0] (per the iter-1 design init of Σ ≈ 0.8 I).
	p.Sigma.assign(static_cast<size_t>(E) * r, 0.0f);
	for (int e = 0; e < E; ++e)
	{
		for (int beta = 0; beta < r; ++beta)
		{
			p.Sigma[e * r + beta] = urand(0.5f, 1.0f);
		}
	}

	// P_q, P_v, P_o: identity-ish (small random perturbation).
	p.P_q.assign(static_cast<size_t>(d_s) * d_h, 0.0f);
	p.P_v.assign(static_cast<size_t>(d_s) * d_h, 0.0f);
	p.P_o.assign(static_cast<size_t>(d_h) * d_s, 0.0f);
	for (int a = 0; a < d_s; ++a)
	{
		for (int h = 0; h < d_h; ++h)
		{
			if (a == h)
				p.P_q[a * d_h + h] = 1.0f;
			if (a == h)
				p.P_v[a * d_h + h] = 1.0f;
			if (h == a)
				p.P_o[h * d_s + a] = 1.0f;
		}
	}
}

int testEdgeSetCount(const SFAParams& p)
{
	// Expected count: n_sinks * (T - n_sinks_mean) + (T - n_sinks) * (W+1) - boundary.
	// Approximate via direct count.
	const int T = p.T;
	const int W = p.W;
	const int n_sinks = p.n_sinks;

	int expected = 0;
	for (int i = 0; i < n_sinks && i < T; ++i)
	{
		expected += (T - i);  // sink i -> j in [i, T-1]
	}
	for (int i = n_sinks; i < T; ++i)
	{
		expected += std::min(T - 1, i + W) - i + 1;
	}

	const int actual = static_cast<int>(p.edge_src.size());
	if (actual == expected)
	{
		std::printf("[PASS] Edge set count: %d (expected %d).\n", actual, expected);
		return 1;
	}
	else
	{
		std::printf("[FAIL] Edge set count: %d (expected %d).\n", actual, expected);
		return 0;
	}
}

int testLaplacianPSD(const SFAParams& p)
{
	// L_F is PSD iff x^T L_F x >= 0 for all x.
	// Test on random vectors.
	const int Tds = p.T * p.d_s;
	std::vector<float> x(Tds), Lx(Tds);

	int n_neg = 0;
	const int n_trials = 8;
	for (int trial = 0; trial < n_trials; ++trial)
	{
		for (int k = 0; k < Tds; ++k)
			x[k] = urand(-1.0f, 1.0f);
		laplacianMatvec(p, &x[0], &Lx[0]);
		float quad = 0.0f;
		for (int k = 0; k < Tds; ++k)
			quad += x[k] * Lx[k];
		if (quad < -1e-4f)
		{
			std::printf("  Trial %d: x^T L_F x = %.4f (NEGATIVE!).\n", trial, quad);
			n_neg++;
		}
	}

	if (n_neg == 0)
	{
		std::printf("[PASS] L_F is PSD on %d random trials.\n", n_trials);
		return 1;
	}
	else
	{
		std::printf("[FAIL] L_F has negative quadratic form on %d/%d trials.\n", n_neg, n_trials);
		return 0;
	}
}

int testLaplacianSymmetric(const SFAParams& p)
{
	// L_F is symmetric iff (L x)^T y == x^T (L y) for all x, y.
	const int Tds = p.T * p.d_s;
	std::vector<float> x(Tds), y(Tds), Lx(Tds), Ly(Tds);

	int n_asymm = 0;
	const int n_trials = 4;
	for (int trial = 0; trial < n_trials; ++trial)
	{
		for (int k = 0; k < Tds; ++k)
		{
			x[k] = urand(-1.0f, 1.0f);
			y[k] = urand(-1.0f, 1.0f);
		}
		laplacianMatvec(p, &x[0], &Lx[0]);
		laplacianMatvec(p, &y[0], &Ly[0]);

		float Lxy = 0.0f, xLy = 0.0f;
		for (int k = 0; k < Tds; ++k)
		{
			Lxy += Lx[k] * y[k];
			xLy += x[k] * Ly[k];
		}
		const float diff = std::fabs(Lxy - xLy);
		const float scale = std::max(std::fabs(Lxy) + std::fabs(xLy), 1e-6f);
		const float rel = diff / scale;
		if (rel > 1e-4f)
		{
			std::printf("  Trial %d: (Lx)^T y = %.6f, x^T (Ly) = %.6f, rel diff = %.2e.\n",
			            trial, Lxy, xLy, rel);
			n_asymm++;
		}
	}

	if (n_asymm == 0)
	{
		std::printf("[PASS] L_F symmetric on %d random trials.\n", n_trials);
		return 1;
	}
	else
	{
		std::printf("[FAIL] L_F asymmetric on %d/%d trials.\n", n_asymm, n_trials);
		return 0;
	}
}

int testChebyshevSolve(const SFAParams& p)
{
	// Run Chebyshev solve at very high lambda — chooses an easy problem
	// the unpreconditioned Chebyshev can converge on. (Smoke test only.)
	// The hard problem (small lambda, ill-conditioned L_F) requires
	// symmetric preconditioning which is iter-15 work per iter-2 PROOFS.md.
	SFAParams p_easy = p;
	p_easy.lambda = 50.0f;  // Easy regime: lambda >> mu_max gives kappa ≈ 1.

	const int Tds = p.T * p.d_s;
	std::vector<float> b(Tds), s(Tds);
	for (int k = 0; k < Tds; ++k)
		b[k] = urand(-1.0f, 1.0f);

	float mu_max_4 = glades::transformer_sfa_ops::estimateMuMax(p_easy, 4, 0xABCDEF12u);
	float mu_max_20 = glades::transformer_sfa_ops::estimateMuMax(p_easy, 20, 0xABCDEF12u);
	float mu_max_50 = glades::transformer_sfa_ops::estimateMuMax(p_easy, 50, 0xABCDEF12u);
	std::printf("  mu_max estimates: 4-iter=%.4f, 20-iter=%.4f, 50-iter=%.4f (lambda=%.1f)\n",
	            mu_max_4, mu_max_20, mu_max_50, p_easy.lambda);

	// Diagnostic sweep: monitor convergence M=1..32 in easy regime.
	std::printf("  Convergence sweep (Clenshaw backward recurrence, easy lambda=50):\n");
	for (int M_try = 1; M_try <= 32; M_try *= 2)
	{
		solveTikhonov(p_easy, &b[0], M_try, /*use_preconditioner=*/false, &s[0]);
		const float r = residualNorm(p_easy, &b[0], &s[0]);
		std::printf("    M=%2d: ||r||/||b|| = %.6e\n", M_try, r);
	}

	// Use M=16 (Clenshaw is stable for any M; round-off no longer accumulates).
	const int M = 16;
	solveTikhonov(p_easy, &b[0], M, /*use_preconditioner=*/false, &s[0]);
	const float rel_residual = residualNorm(p_easy, &b[0], &s[0]);

	if (rel_residual < 1e-4f)
	{
		std::printf("[PASS] Chebyshev M=%d converges (residual %.6e < 1e-4).\n",
		            M, rel_residual);
		return 1;
	}
	else
	{
		std::printf("[FAIL] Chebyshev M=%d residual %.6e >= 1e-4.\n", M, rel_residual);
		return 0;
	}
}

}  // anonymous namespace

// Probe A: structural recovery check on TINY problem (T=8, d_s=1).
// Builds L_F explicitly as a dense matrix, solves (L_F + λI) s = b by Gaussian
// elimination, and compares to Chebyshev result. Validates that the iterative
// solver produces the same answer as the direct solve.
int testDirectSolveAgreement()
{
	SFAParams p;
	const int T = 8, d_s = 1, d_h = 1, r = 1, W = 3, n_sinks = 1;
	buildSyntheticSFAParams(p, T, d_s, d_h, r, W, n_sinks);
	p.lambda = 10.0f;  // Moderate regulariser for well-conditioned direct solve.

	const int Tds = T * d_s;
	std::vector<float> b(Tds), s_iter(Tds), s_direct(Tds);
	for (int k = 0; k < Tds; ++k)
		b[k] = urand(-1.0f, 1.0f);

	// Build L_F + lambda*I as dense matrix via repeated unit-vector matvecs.
	std::vector<float> M_dense(Tds * Tds, 0.0f);
	std::vector<float> e(Tds, 0.0f), Le(Tds);
	for (int j = 0; j < Tds; ++j)
	{
		std::fill(e.begin(), e.end(), 0.0f);
		e[j] = 1.0f;
		glades::transformer_sfa_ops::laplacianMatvec(p, &e[0], &Le[0]);
		for (int i = 0; i < Tds; ++i)
			M_dense[i * Tds + j] = Le[i] + (i == j ? p.lambda : 0.0f);
	}

	// Direct Gaussian elimination on M_dense * s_direct = b.
	std::vector<float> A(M_dense);
	std::vector<float> rhs(b);
	// Forward elimination with partial pivoting (deterministic).
	for (int i = 0; i < Tds; ++i)
	{
		// Find pivot
		int piv = i;
		for (int k = i + 1; k < Tds; ++k)
			if (std::fabs(A[k * Tds + i]) > std::fabs(A[piv * Tds + i]))
				piv = k;
		if (piv != i)
		{
			for (int k = 0; k < Tds; ++k)
				std::swap(A[i * Tds + k], A[piv * Tds + k]);
			std::swap(rhs[i], rhs[piv]);
		}
		const float diag = A[i * Tds + i];
		if (std::fabs(diag) < 1e-12f)
		{
			std::printf("[FAIL] Probe A: matrix singular at row %d.\n", i);
			return 0;
		}
		for (int k = i + 1; k < Tds; ++k)
		{
			const float factor = A[k * Tds + i] / diag;
			for (int j = i; j < Tds; ++j)
				A[k * Tds + j] -= factor * A[i * Tds + j];
			rhs[k] -= factor * rhs[i];
		}
	}
	// Back substitution.
	for (int i = Tds - 1; i >= 0; --i)
	{
		float sum = rhs[i];
		for (int j = i + 1; j < Tds; ++j)
			sum -= A[i * Tds + j] * s_direct[j];
		s_direct[i] = sum / A[i * Tds + i];
	}

	// Iterative Chebyshev solve at high M.
	glades::transformer_sfa_chebyshev::solveTikhonov(p, &b[0], 32, false, &s_iter[0]);

	// Compare.
	float diff = 0.0f, norm = 0.0f;
	for (int k = 0; k < Tds; ++k)
	{
		const float d = s_iter[k] - s_direct[k];
		diff += d * d;
		norm += s_direct[k] * s_direct[k];
	}
	diff = std::sqrt(diff);
	norm = std::sqrt(norm);
	const float rel_err = diff / std::max(norm, 1e-12f);

	std::printf("  Probe A (T=8, d_s=1, lambda=%.1f, M=32):\n", p.lambda);
	std::printf("    ||s_iter - s_direct||/||s_direct|| = %.6e\n", rel_err);

	if (rel_err < 1e-3f)
	{
		std::printf("[PASS] Iterative solve agrees with direct (rel err %.2e < 1e-3).\n", rel_err);
		return 1;
	}
	else
	{
		std::printf("[FAIL] Iterative solve disagrees with direct: rel err %.2e.\n", rel_err);
		return 0;
	}
}

// Scale-up test: Lanczos at T=1024, d_s=8, r=4 — near-production size.
// Reports wall-clock to demonstrate feasibility at the design-target scale.
int testLanczosScaleUp()
{
	SFAParams p;
	const int T = 1024, d_s = 8, d_h = 64, r = 4, W = 128, n_sinks = 8;
	buildSyntheticSFAParams(p, T, d_s, d_h, r, W, n_sinks);
	p.lambda = 1e-2f;

	const int Tds = T * d_s;
	std::vector<float> b(Tds), s_lanczos(Tds);
	for (int k = 0; k < Tds; ++k)
		b[k] = urand(-1.0f, 1.0f);

	std::printf("  Scale-up: T=%d, d_s=%d, r=%d, W=%d, sinks=%d, |E|=%zu, lambda=%.2e\n",
	            T, d_s, r, W, n_sinks, p.edge_src.size(), p.lambda);

	const int m = 64;

	// Wall-clock timer.
	clock_t t_start = clock();
	glades::transformer_sfa_chebyshev::lanczosSolve(p, &b[0], m, &s_lanczos[0]);
	clock_t t_end = clock();
	const double wall_sec = double(t_end - t_start) / CLOCKS_PER_SEC;

	const float rel_residual = glades::transformer_sfa_chebyshev::residualNorm(p, &b[0], &s_lanczos[0]);

	std::printf("    Lanczos m=%d: ||r||/||b|| = %.4e, wall-clock = %.3f sec\n",
	            m, rel_residual, wall_sec);

	// Pass: residual < 1e-2 AND wall-clock < 30 sec.
	const bool pass = (rel_residual < 1e-2f) && (wall_sec < 30.0);
	if (pass)
	{
		std::printf("[PASS] T=%d Lanczos converges to %.2e in %.2f sec.\n",
		            T, rel_residual, wall_sec);
		return 1;
	}
	else
	{
		std::printf("[FAIL] T=%d Lanczos: residual=%.2e, wall=%.2fs.\n",
		            T, rel_residual, wall_sec);
		return 0;
	}
}

// Test Lanczos solver on the HARD regime (small lambda, ill-conditioned L_F).
// Chebyshev fails this without proper preconditioning; Lanczos should succeed
// by adapting to the spectrum.
int testLanczosHardRegime()
{
	SFAParams p;
	const int T = 64, d_s = 4, d_h = 4, r = 2, W = 8, n_sinks = 2;
	buildSyntheticSFAParams(p, T, d_s, d_h, r, W, n_sinks);
	p.lambda = 1e-2f;  // Hard regime: kappa = mu_max / lambda ~ 4000.

	const int Tds = T * d_s;
	std::vector<float> b(Tds), s_lanczos(Tds);
	for (int k = 0; k < Tds; ++k)
		b[k] = urand(-1.0f, 1.0f);

	// Lanczos at m=32 with full re-orthogonalization.
	const int m = 32;
	glades::transformer_sfa_chebyshev::lanczosSolve(p, &b[0], m, &s_lanczos[0]);

	const float rel_residual = glades::transformer_sfa_chebyshev::residualNorm(p, &b[0], &s_lanczos[0]);
	std::printf("  Lanczos m=%d (HARD lambda=%.2e, kappa~%d):\n",
	            m, p.lambda, static_cast<int>(43.5f / p.lambda));
	std::printf("    ||r||/||b|| = %.6e\n", rel_residual);

	if (rel_residual < 1e-3f)
	{
		std::printf("[PASS] Lanczos converges in hard regime (rel residual %.2e < 1e-3).\n",
		            rel_residual);
		return 1;
	}
	else
	{
		std::printf("[FAIL] Lanczos hard-regime residual %.2e >= 1e-3.\n", rel_residual);
		return 0;
	}
}

int main(int /*argc*/, char** /*argv*/)
{
	std::printf("SFA CPU prototype smoke test\n");
	std::printf("============================\n");

	SFAParams p;
	const int T = 64, d_s = 4, d_h = 4, r = 2, W = 8, n_sinks = 2;
	buildSyntheticSFAParams(p, T, d_s, d_h, r, W, n_sinks);
	std::printf("Config: T=%d, d_s=%d, r=%d, W=%d, n_sinks=%d, |E|=%zu.\n",
	            T, d_s, r, W, n_sinks, p.edge_src.size());
	std::printf("\n");

	int n_pass = 0, n_total = 0;

	n_pass += testEdgeSetCount(p);   n_total++;
	n_pass += testLaplacianSymmetric(p);   n_total++;
	n_pass += testLaplacianPSD(p);   n_total++;
	n_pass += testChebyshevSolve(p);   n_total++;
	n_pass += testDirectSolveAgreement();   n_total++;
	n_pass += testLanczosHardRegime();   n_total++;
	n_pass += testLanczosScaleUp();   n_total++;

	std::printf("\n");
	std::printf("============================\n");
	std::printf("Summary: %d / %d tests PASSED.\n", n_pass, n_total);
	std::printf("\n");
	if (n_pass == n_total)
	{
		std::printf("[OVERALL PASS] SFA CPU prototype primitives are correct.\n");
		std::printf("Next: integrate with unit-test framework (iter 15) and implement\n");
		std::printf("      Probe A (SCFA-recovery numerical check).\n");
		return 0;
	}
	else
	{
		std::printf("[OVERALL FAIL] Some primitives have bugs. Fix before Gate-0.\n");
		return 1;
	}
}
