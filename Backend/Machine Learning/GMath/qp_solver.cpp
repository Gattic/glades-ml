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
#include "qp_solver.h"
#include "kelly.h"
#include <algorithm>
#include <cstdio>

namespace glades
{

double QPSolver::objective(const std::vector<double>& mu,
                           const std::vector<double>& covFlat,
                           const std::vector<double>& w,
                           size_t N)
{
	return Kelly::portfolioGrowthRate(w, mu, covFlat, N);
}

std::vector<double> QPSolver::gradient(const std::vector<double>& mu,
                                       const std::vector<double>& covFlat,
                                       const std::vector<double>& w,
                                       size_t N)
{
	// grad = mu - Sigma * w
	std::vector<double> Sw = Kelly::matVecMul(covFlat, w, N);
	std::vector<double> grad(N);
	for (size_t i = 0; i < N; ++i)
		grad[i] = mu[i] - Sw[i];
	return grad;
}

std::vector<double> QPSolver::projectOntoConstraints(const std::vector<double>& w,
                                                     const QPConstraints& constraints,
                                                     size_t N)
{
	std::vector<double> proj = w;

	// Apply box constraints first
	if (constraints.boxConstraints)
	{
		for (size_t i = 0; i < N; ++i)
		{
			if (i < constraints.lowerBounds.size() && proj[i] < constraints.lowerBounds[i])
				proj[i] = constraints.lowerBounds[i];
			if (i < constraints.upperBounds.size() && proj[i] > constraints.upperBounds[i])
				proj[i] = constraints.upperBounds[i];
		}
	}

	// Project onto budget constraint: sum(w) = target
	// Using iterative projection with clipping (handles box + budget jointly)
	if (constraints.budgetConstraint)
	{
		for (int iter = 0; iter < 50; ++iter)
		{
			double sum = 0.0;
			int numFree = 0;
			for (size_t i = 0; i < N; ++i)
			{
				sum += proj[i];
				// Check if this variable is at a box bound
				bool atLower = constraints.boxConstraints &&
				               i < constraints.lowerBounds.size() &&
				               proj[i] <= constraints.lowerBounds[i] + 1e-12;
				bool atUpper = constraints.boxConstraints &&
				               i < constraints.upperBounds.size() &&
				               proj[i] >= constraints.upperBounds[i] - 1e-12;
				if (!atLower && !atUpper)
					++numFree;
			}

			double excess = sum - constraints.budgetTarget;
			if (std::fabs(excess) < 1e-12)
				break;

			if (numFree == 0)
			{
				// All at bounds, distribute evenly anyway
				double shift = excess / (double)N;
				for (size_t i = 0; i < N; ++i)
					proj[i] -= shift;
			}
			else
			{
				double shift = excess / (double)numFree;
				for (size_t i = 0; i < N; ++i)
				{
					bool atLower = constraints.boxConstraints &&
					               i < constraints.lowerBounds.size() &&
					               proj[i] <= constraints.lowerBounds[i] + 1e-12;
					bool atUpper = constraints.boxConstraints &&
					               i < constraints.upperBounds.size() &&
					               proj[i] >= constraints.upperBounds[i] - 1e-12;
					if (!atLower && !atUpper)
						proj[i] -= shift;
				}
			}

			// Re-clip
			if (constraints.boxConstraints)
			{
				for (size_t i = 0; i < N; ++i)
				{
					if (i < constraints.lowerBounds.size() &&
					    proj[i] < constraints.lowerBounds[i])
						proj[i] = constraints.lowerBounds[i];
					if (i < constraints.upperBounds.size() &&
					    proj[i] > constraints.upperBounds[i])
						proj[i] = constraints.upperBounds[i];
				}
			}
		}
	}

	// Leverage constraint: if ||w||_1 > L_max, scale toward budget target
	if (constraints.leverageConstraint)
	{
		double l1norm = 0.0;
		for (size_t i = 0; i < N; ++i)
			l1norm += std::fabs(proj[i]);

		if (l1norm > constraints.maxLeverage)
		{
			double scale = constraints.maxLeverage / l1norm;
			for (size_t i = 0; i < N; ++i)
				proj[i] *= scale;
		}
	}

	return proj;
}

QPResult QPSolver::solveBudgetOnly(const std::vector<double>& mu,
                                   const std::vector<double>& covFlat,
                                   size_t N)
{
	QPResult result;

	// Closed form: w* = Sigma^{-1}(mu - lambda * 1)
	// where lambda = (1^T Sigma^{-1} mu - 1) / (1^T Sigma^{-1} 1)

	// Solve Sigma^{-1} * mu
	std::vector<double> SinvMu = Kelly::choleskySolve(covFlat, mu, N);
	if (SinvMu.empty())
	{
		printf("[QPSolver] Failed: covariance matrix not positive definite\n");
		result.converged = false;
		return result;
	}

	// Solve Sigma^{-1} * 1
	std::vector<double> ones(N, 1.0);
	std::vector<double> SinvOne = Kelly::choleskySolve(covFlat, ones, N);
	if (SinvOne.empty())
	{
		result.converged = false;
		return result;
	}

	// Compute lambda
	double oneT_SinvMu = 0.0;
	double oneT_SinvOne = 0.0;
	for (size_t i = 0; i < N; ++i)
	{
		oneT_SinvMu += SinvMu[i];
		oneT_SinvOne += SinvOne[i];
	}

	double lambda = 0.0;
	if (std::fabs(oneT_SinvOne) > 1e-15)
		lambda = (oneT_SinvMu - 1.0) / oneT_SinvOne;

	// w* = Sigma^{-1}(mu - lambda * 1)
	result.weights.resize(N);
	for (size_t i = 0; i < N; ++i)
		result.weights[i] = SinvMu[i] - lambda * SinvOne[i];

	result.budgetMultiplier = lambda;
	result.objectiveValue = objective(mu, covFlat, result.weights, N);
	result.iterations = 1;
	result.converged = true;

	return result;
}

QPResult QPSolver::solve(const std::vector<double>& mu,
                         const std::vector<double>& covFlat,
                         size_t N,
                         const QPConstraints& constraints,
                         int maxIter,
                         double tol)
{
	QPResult result;

	// If only budget constraint and no box/leverage, use closed form
	if (constraints.budgetConstraint && !constraints.boxConstraints &&
	    !constraints.leverageConstraint)
	{
		return solveBudgetOnly(mu, covFlat, N);
	}

	// Projected gradient ascent for the general case.
	// We maximize G(w) = mu^T w - 0.5 w^T Sigma w subject to constraints.

	// Initialize with equal weights
	std::vector<double> w(N);
	if (constraints.budgetConstraint)
	{
		double initVal = constraints.budgetTarget / (double)N;
		for (size_t i = 0; i < N; ++i)
			w[i] = initVal;
	}

	// Apply initial projection
	w = projectOntoConstraints(w, constraints, N);

	// Estimate Lipschitz constant from Sigma (sum of absolute row values as upper bound)
	double L = 0.0;
	for (size_t i = 0; i < N; ++i)
	{
		double rowSum = 0.0;
		for (size_t j = 0; j < N; ++j)
			rowSum += std::fabs(covFlat[i * N + j]);
		if (rowSum > L)
			L = rowSum;
	}
	if (L < 1e-15)
		L = 1.0;
	double stepSize = 1.0 / L;

	double prevObj = objective(mu, covFlat, w, N);
	int iter;

	for (iter = 0; iter < maxIter; ++iter)
	{
		// Compute gradient
		std::vector<double> grad = gradient(mu, covFlat, w, N);

		// Gradient ascent step with projection
		std::vector<double> wNew(N);
		for (size_t i = 0; i < N; ++i)
			wNew[i] = w[i] + stepSize * grad[i];

		// Project onto feasible set
		wNew = projectOntoConstraints(wNew, constraints, N);

		double newObj = objective(mu, covFlat, wNew, N);

		// Backtracking: if objective did not improve, halve step size
		int backtrack = 0;
		while (newObj < prevObj && backtrack < 30)
		{
			stepSize *= 0.5;
			for (size_t i = 0; i < N; ++i)
				wNew[i] = w[i] + stepSize * grad[i];
			wNew = projectOntoConstraints(wNew, constraints, N);
			newObj = objective(mu, covFlat, wNew, N);
			++backtrack;
		}

		// Check convergence on weight change
		double wDiff = 0.0;
		for (size_t i = 0; i < N; ++i)
		{
			double d = wNew[i] - w[i];
			wDiff += d * d;
		}

		w = wNew;
		prevObj = newObj;

		if (std::sqrt(wDiff) < tol)
			break;

		// Gentle step size recovery (don't overshoot)
		if (backtrack == 0 && stepSize < 1.0 / L)
			stepSize = std::min(stepSize * 1.2, 1.0 / L);
	}

	result.weights = w;
	result.objectiveValue = prevObj;
	result.iterations = iter + 1;
	result.converged = (iter < maxIter);

	// Estimate budget multiplier from KKT conditions:
	// grad_L = mu - Sigma*w - lambda*1 = 0 for unconstrained components
	// So lambda = mean(mu - Sigma*w) over free components
	std::vector<double> grad = gradient(mu, covFlat, w, N);
	double lambdaSum = 0.0;
	int numFree = 0;
	for (size_t i = 0; i < N; ++i)
	{
		bool atBound = false;
		if (constraints.boxConstraints)
		{
			if (i < constraints.lowerBounds.size() &&
			    std::fabs(w[i] - constraints.lowerBounds[i]) < 1e-8)
				atBound = true;
			if (i < constraints.upperBounds.size() &&
			    std::fabs(w[i] - constraints.upperBounds[i]) < 1e-8)
				atBound = true;
		}
		if (!atBound)
		{
			lambdaSum += grad[i];
			++numFree;
		}
	}
	result.budgetMultiplier = (numFree > 0) ? (lambdaSum / (double)numFree) : 0.0;

	// Estimate box multipliers
	result.boxMultipliersLower.resize(N, 0.0);
	result.boxMultipliersUpper.resize(N, 0.0);
	if (constraints.boxConstraints)
	{
		for (size_t i = 0; i < N; ++i)
		{
			if (i < constraints.lowerBounds.size() &&
			    std::fabs(w[i] - constraints.lowerBounds[i]) < 1e-8)
			{
				// At lower bound: multiplier = -(grad_i - lambda)
				result.boxMultipliersLower[i] =
					-(grad[i] - result.budgetMultiplier);
			}
			if (i < constraints.upperBounds.size() &&
			    std::fabs(w[i] - constraints.upperBounds[i]) < 1e-8)
			{
				// At upper bound: multiplier = grad_i - lambda
				result.boxMultipliersUpper[i] =
					grad[i] - result.budgetMultiplier;
			}
		}
	}

	return result;
}

}; // namespace glades
