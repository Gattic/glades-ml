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
#ifndef GLADES_QP_SOLVER_H
#define GLADES_QP_SOLVER_H

#include <cmath>
#include <cstddef>
#include <vector>

namespace glades
{

// Result of a quadratic programming solve
struct QPResult
{
	std::vector<double> weights;             // Optimal w* (N-vector)
	double objectiveValue;                   // G(w*) at the optimum
	double budgetMultiplier;                 // Lagrange multiplier for budget constraint
	std::vector<double> boxMultipliersLower; // Lagrange multipliers for lower box bounds
	std::vector<double> boxMultipliersUpper; // Lagrange multipliers for upper box bounds
	int iterations;
	bool converged;

	QPResult()
		: objectiveValue(0.0)
		, budgetMultiplier(0.0)
		, iterations(0)
		, converged(false)
	{
	}
};

// Constraints for the portfolio QP problem
struct QPConstraints
{
	bool budgetConstraint;    // 1^T w = 1
	double budgetTarget;      // Target sum (usually 1.0)

	bool boxConstraints;      // w_min <= w_i <= w_max
	std::vector<double> lowerBounds;
	std::vector<double> upperBounds;

	bool leverageConstraint;  // ||w||_1 <= L_max
	double maxLeverage;

	QPConstraints()
		: budgetConstraint(true)
		, budgetTarget(1.0)
		, boxConstraints(false)
		, leverageConstraint(false)
		, maxLeverage(1.0)
	{
	}
};

class QPSolver
{
public:

	// Solve the Kelly portfolio QP:
	//   max  mu^T w - 0.5 w^T Sigma w
	//   s.t. constraints
	//
	// mu: N-vector of expected returns
	// covFlat: N x N covariance matrix (row-major flat)
	// N: number of assets
	// constraints: portfolio constraints
	// maxIter: maximum iterations for the active-set solver
	// tol: convergence tolerance
	static QPResult solve(const std::vector<double>& mu,
	                      const std::vector<double>& covFlat,
	                      size_t N,
	                      const QPConstraints& constraints,
	                      int maxIter = 500,
	                      double tol = 1e-10);

	// Convenience: solve with only a budget constraint (1^T w = 1).
	// Returns the closed-form constrained Kelly: w* = Sigma^{-1}(mu - lambda * 1)
	static QPResult solveBudgetOnly(const std::vector<double>& mu,
	                                const std::vector<double>& covFlat,
	                                size_t N);

private:

	// Project a vector onto the simplex {x : sum(x) = target, lb <= x <= ub}
	// via iterative clipping (Duchi et al. generalization)
	static std::vector<double> projectOntoConstraints(const std::vector<double>& w,
	                                                  const QPConstraints& constraints,
	                                                  size_t N);

	// Compute gradient of the objective: grad = mu - Sigma * w
	static std::vector<double> gradient(const std::vector<double>& mu,
	                                    const std::vector<double>& covFlat,
	                                    const std::vector<double>& w,
	                                    size_t N);

	// Compute objective value: mu^T w - 0.5 w^T Sigma w
	static double objective(const std::vector<double>& mu,
	                        const std::vector<double>& covFlat,
	                        const std::vector<double>& w,
	                        size_t N);
};

}; // namespace glades

#endif
