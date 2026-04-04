// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef GLADES_KELLY_H
#define GLADES_KELLY_H

#include <cmath>
#include <cstddef>
#include <vector>

namespace glades
{

struct KellyResult
{
	std::vector<double> weights;           // Optimal portfolio weights
	double growthRate;                     // Expected log-growth rate G(w*)
	std::vector<double> factorGrowthRates; // Per-factor growth contribution
	double unconstrainedGrowthRate;        // G without constraints (upper bound)

	KellyResult()
		: growthRate(0.0)
		, unconstrainedGrowthRate(0.0)
	{
	}
};

class Kelly
{
public:

	// ---- Univariate Kelly ----

	// Kelly fraction for a single bet: f* = mu / sigma^2
	// mu: expected return, sigma2: variance of return
	static double fraction(double mu, double sigma2);

	// Kelly growth rate for a single bet at fraction f:
	// G(f) = f * mu - 0.5 * f^2 * sigma^2
	static double growthRate(double f, double mu, double sigma2);

	// Optimal growth rate: mu^2 / (2 * sigma^2)
	static double optimalGrowthRate(double mu, double sigma2);

	// ---- Multivariate Kelly (uncorrelated factors) ----

	// Kelly fractions for K uncorrelated factors.
	// alphas: K expected returns
	// variances: K variances (diagonal of factor covariance)
	// Returns K Kelly fractions: g_j = alpha_j / variance_j
	static std::vector<double> factorFractions(const std::vector<double>& alphas,
	                                           const std::vector<double>& variances);

	// Growth rate for uncorrelated factors at given fractions.
	// g: K fractions, alphas: K expected returns, variances: K variances
	static double factorGrowthRate(const std::vector<double>& g,
	                               const std::vector<double>& alphas,
	                               const std::vector<double>& variances);

	// Per-factor growth rate contributions at optimal fractions.
	// Returns K values: alpha_j^2 / (2 * variance_j)
	static std::vector<double> factorGrowthAttribution(const std::vector<double>& alphas,
	                                                   const std::vector<double>& variances);

	// ---- Multivariate Kelly (general covariance) ----

	// Unconstrained Kelly weights: w* = Sigma^{-1} * mu
	// mu: N expected returns
	// covFlat: N x N covariance matrix (row-major flat)
	// Returns N weights. Returns empty vector if Sigma is singular.
	static std::vector<double> unconstrainedWeights(const std::vector<double>& mu,
	                                                const std::vector<double>& covFlat,
	                                                size_t N);

	// Kelly growth rate at given weights:
	// G(w) = w' * mu - 0.5 * w' * Sigma * w
	static double portfolioGrowthRate(const std::vector<double>& w,
	                                  const std::vector<double>& mu,
	                                  const std::vector<double>& covFlat,
	                                  size_t N);

	// Fractional Kelly: scale weights by a fraction f in (0, 1]
	static std::vector<double> fractionalKelly(const std::vector<double>& fullKelly,
	                                           double fraction);

	// ---- Utility ----

	// Solve a symmetric positive-definite linear system Ax = b via Cholesky decomposition.
	// A: N x N flat row-major. b: N-vector. Returns x. Returns empty on failure.
	static std::vector<double> choleskySolve(const std::vector<double>& A,
	                                         const std::vector<double>& b,
	                                         size_t N);

	// Compute matrix-vector product: y = A * x (A is N x N flat row-major, x is N-vector)
	static std::vector<double> matVecMul(const std::vector<double>& A,
	                                     const std::vector<double>& x,
	                                     size_t N);

private:

	// Cholesky decomposition: A = L * L^T. Returns lower-triangular L as flat N x N.
	// Returns empty on failure (not positive definite).
	static std::vector<double> choleskyDecomp(const std::vector<double>& A, size_t N);

	// Forward substitution: solve L * y = b
	static std::vector<double> forwardSub(const std::vector<double>& L,
	                                      const std::vector<double>& b,
	                                      size_t N);

	// Back substitution: solve L^T * x = y
	static std::vector<double> backSub(const std::vector<double>& L,
	                                   const std::vector<double>& y,
	                                   size_t N);
};

}; // namespace glades

#endif
