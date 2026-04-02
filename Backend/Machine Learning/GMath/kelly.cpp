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
#include "kelly.h"
#include <cstdio>

namespace glades
{

// ==== Univariate Kelly ====

double Kelly::fraction(double mu, double sigma2)
{
	if (sigma2 <= 1e-15)
	{
		printf("[Kelly] Warning: variance near zero (%.2e), returning 0\n", sigma2);
		return 0.0;
	}
	return mu / sigma2;
}

double Kelly::growthRate(double f, double mu, double sigma2)
{
	return f * mu - 0.5 * f * f * sigma2;
}

double Kelly::optimalGrowthRate(double mu, double sigma2)
{
	if (sigma2 <= 1e-15)
		return 0.0;
	return (mu * mu) / (2.0 * sigma2);
}

// ==== Multivariate Kelly (uncorrelated factors) ====

std::vector<double> Kelly::factorFractions(const std::vector<double>& alphas,
                                           const std::vector<double>& variances)
{
	size_t K = alphas.size();
	std::vector<double> g(K);
	for (size_t j = 0; j < K; ++j)
	{
		if (variances[j] <= 1e-15)
			g[j] = 0.0;
		else
			g[j] = alphas[j] / variances[j];
	}
	return g;
}

double Kelly::factorGrowthRate(const std::vector<double>& g,
                               const std::vector<double>& alphas,
                               const std::vector<double>& variances)
{
	double G = 0.0;
	for (size_t j = 0; j < g.size(); ++j)
		G += g[j] * alphas[j] - 0.5 * g[j] * g[j] * variances[j];
	return G;
}

std::vector<double> Kelly::factorGrowthAttribution(const std::vector<double>& alphas,
                                                   const std::vector<double>& variances)
{
	size_t K = alphas.size();
	std::vector<double> contrib(K);
	for (size_t j = 0; j < K; ++j)
	{
		if (variances[j] <= 1e-15)
			contrib[j] = 0.0;
		else
			contrib[j] = (alphas[j] * alphas[j]) / (2.0 * variances[j]);
	}
	return contrib;
}

// ==== Linear algebra utilities ====

std::vector<double> Kelly::matVecMul(const std::vector<double>& A,
                                     const std::vector<double>& x,
                                     size_t N)
{
	std::vector<double> y(N, 0.0);
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			y[i] += A[i * N + j] * x[j];
	return y;
}

std::vector<double> Kelly::choleskyDecomp(const std::vector<double>& A, size_t N)
{
	std::vector<double> L(N * N, 0.0);

	for (size_t i = 0; i < N; ++i)
	{
		for (size_t j = 0; j <= i; ++j)
		{
			double sum = 0.0;
			for (size_t k = 0; k < j; ++k)
				sum += L[i * N + k] * L[j * N + k];

			if (i == j)
			{
				double val = A[i * N + i] - sum;
				if (val <= 0.0)
				{
					printf("[Kelly] Cholesky failed: matrix not positive definite "
					       "(pivot %lu = %.2e)\n", (unsigned long)i, val);
					return std::vector<double>();
				}
				L[i * N + j] = std::sqrt(val);
			}
			else
			{
				if (std::fabs(L[j * N + j]) < 1e-30)
				{
					printf("[Kelly] Cholesky failed: zero diagonal at %lu\n",
					       (unsigned long)j);
					return std::vector<double>();
				}
				L[i * N + j] = (A[i * N + j] - sum) / L[j * N + j];
			}
		}
	}
	return L;
}

std::vector<double> Kelly::forwardSub(const std::vector<double>& L,
                                      const std::vector<double>& b,
                                      size_t N)
{
	std::vector<double> y(N);
	for (size_t i = 0; i < N; ++i)
	{
		double sum = 0.0;
		for (size_t j = 0; j < i; ++j)
			sum += L[i * N + j] * y[j];
		y[i] = (b[i] - sum) / L[i * N + i];
	}
	return y;
}

std::vector<double> Kelly::backSub(const std::vector<double>& L,
                                   const std::vector<double>& y,
                                   size_t N)
{
	std::vector<double> x(N);
	for (size_t i = N; i > 0; --i)
	{
		size_t idx = i - 1;
		double sum = 0.0;
		for (size_t j = idx + 1; j < N; ++j)
			sum += L[j * N + idx] * x[j]; // L^T element at (idx, j) = L(j, idx)
		x[idx] = (y[idx] - sum) / L[idx * N + idx];
	}
	return x;
}

std::vector<double> Kelly::choleskySolve(const std::vector<double>& A,
                                         const std::vector<double>& b,
                                         size_t N)
{
	std::vector<double> L = choleskyDecomp(A, N);
	if (L.empty())
		return std::vector<double>();

	// Solve L y = b
	std::vector<double> y = forwardSub(L, b, N);
	// Solve L^T x = y
	return backSub(L, y, N);
}

// ==== Multivariate Kelly (general covariance) ====

std::vector<double> Kelly::unconstrainedWeights(const std::vector<double>& mu,
                                                const std::vector<double>& covFlat,
                                                size_t N)
{
	// w* = Sigma^{-1} * mu, solved via Cholesky
	return choleskySolve(covFlat, mu, N);
}

double Kelly::portfolioGrowthRate(const std::vector<double>& w,
                                  const std::vector<double>& mu,
                                  const std::vector<double>& covFlat,
                                  size_t N)
{
	// G(w) = w' * mu - 0.5 * w' * Sigma * w
	double wMu = 0.0;
	for (size_t i = 0; i < N; ++i)
		wMu += w[i] * mu[i];

	std::vector<double> Sw = matVecMul(covFlat, w, N);
	double wSw = 0.0;
	for (size_t i = 0; i < N; ++i)
		wSw += w[i] * Sw[i];

	return wMu - 0.5 * wSw;
}

std::vector<double> Kelly::fractionalKelly(const std::vector<double>& fullKelly,
                                           double frac)
{
	std::vector<double> result(fullKelly.size());
	for (size_t i = 0; i < fullKelly.size(); ++i)
		result[i] = fullKelly[i] * frac;
	return result;
}

}; // namespace glades
