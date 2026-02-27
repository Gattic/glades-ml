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
#ifndef _GBAYESOPTIMIZER
#define _GBAYESOPTIMIZER

#include "Backend/Database/GTable.h"
#include "../GMath/OHE.h"
#include "../rng.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <algorithm>

namespace glades {

// Gaussian Process for Regression (supports N-dimensional inputs)
class GaussianProcess
{
public:
	GaussianProcess(float length_scale = 1.0f, float variance = 1.0f, float noise = 1e-5f)
		: variance_(variance), noise_(noise), ndim_(1)
	{
		length_scales_.push_back(length_scale);
	}

	// N-dim constructor with ARD length scales
	GaussianProcess(const std::vector<float>& length_scales, float variance = 1.0f, float noise = 1e-5f)
		: length_scales_(length_scales), variance_(variance), noise_(noise),
		  ndim_(length_scales.size())
	{}

	// N-dim interface
	void addSample(const std::vector<float>& x, float y);
	void fit();
	std::pair<float, float> predict(const std::vector<float>& x) const;

	// Legacy 1D wrappers for backward compatibility
	void addSample(float x, float y);
	std::pair<float, float> predict(float x) const;

	void printInput() const;
	void print() const;

	unsigned int ndim() const { return ndim_; }
	unsigned int numSamples() const { return static_cast<unsigned int>(X_.size()); }

private:
	std::vector<std::vector<float> > X_;       // Observations (inputs), each is ndim-vector
	std::vector<float> y_;                     // Observations (outputs)
	std::vector<std::vector<float> > L_;       // Lower-triangular Cholesky factor of K
	std::vector<float> alpha_;                 // alpha = L^T \ (L \ y)
	std::vector<float> length_scales_;         // ARD length scales (one per dimension)
	float variance_;                           // Signal variance of the RBF kernel
	float noise_;                              // Noise level (added to diagonal)
	unsigned int ndim_;                        // Input dimensionality

	// Cholesky decomposition: returns lower-triangular L such that A = L * L^T
	// Returns false if decomposition fails (matrix not positive definite)
	bool choleskyDecompose(const std::vector<std::vector<float> >& A,
	                       std::vector<std::vector<float> >& L) const;

	// Solve L * x = b for x (forward substitution)
	std::vector<float> choleskySolveLower(const std::vector<std::vector<float> >& L,
	                                      const std::vector<float>& b) const;

	// Solve L^T * x = b for x (back substitution)
	std::vector<float> choleskySolveUpper(const std::vector<std::vector<float> >& L,
	                                      const std::vector<float>& b) const;

	// ARD RBF kernel: k(x1,x2) = variance * exp(-0.5 * sum((x1[d]-x2[d])^2 / ls[d]^2))
	float rbfKernelND(const std::vector<float>& x1, const std::vector<float>& x2) const;

	// Legacy 1D kernel (kept for reference, delegates to rbfKernelND)
	static float rbfKernel(float x1, float x2, float length_scale = 1.0f, float variance = 1.0f)
	{
		return variance * exp(-0.5f * pow((x1 - x2) / length_scale, 2));
	}
};

// Bayesian Optimization with Gaussian Process
class BayesianOptimizer
{
private:
	std::vector<float> best_params_;   // Best parameters found (N-dim)
	float best_score_;                 // Best score found (for minimization: lower is better)
	GaussianProcess gp_;
	unsigned int ndim_;
	glades::rng::Engine rng_;

public:
	// Legacy 1D constructor
	BayesianOptimizer()
		: best_score_(std::numeric_limits<float>::max()), ndim_(1)
	{
		best_params_.push_back(0.0f);
	}

	// N-dim constructor
	BayesianOptimizer(unsigned int ndim)
		: best_score_(std::numeric_limits<float>::max()), ndim_(ndim)
	{
		std::vector<float> ls(ndim, 1.0f);
		gp_ = GaussianProcess(ls);
		best_params_.resize(ndim, 0.0f);
	}

	// N-dim interface
	void addObservation(const std::vector<float>& x, float y);
	void fit();
	std::vector<float> suggestNext();

	// Legacy 1D interface
	float optimize(const std::vector<std::pair<float, float> >);
	void update(const std::pair<float, float>);

	// Legacy 1D getters
	float getBestParam() const { return best_params_.empty() ? 0.0f : best_params_[0]; }
	float getBestScore() const { return best_score_; }
	const GaussianProcess& getGP() const { return gp_; }

	// N-dim getters
	const std::vector<float>& getBestParams() const { return best_params_; }

	// CDF and PDF functions for the Gaussian distribution (now static)
	static float cdf(float x)
	{
		return 0.5f * (1.0f + erf(x / sqrt(2.0f)));
	}

	static float pdf(float x)
	{
		return exp(-0.5f * x * x) / sqrt(2.0f * M_PI);
	}

	// Legacy 1D Expected Improvement (maximization)
	float expectedImprovement(float x, const GaussianProcess& gp, float best_y)
	{
		std::pair<float, float> prediction = gp.predict(x);
		float mu = prediction.first;
		float sigma2 = prediction.second;
		float sigma = sqrt(sigma2);

		float z = (mu - best_y) / sigma;
		return (mu - best_y) * cdf(z) + sigma * pdf(z);
	}

	// N-dim Expected Improvement for MINIMIZATION
	// EI(x) = (best_y - mu) * CDF(z) + sigma * PDF(z) where z = (best_y - mu) / sigma
	float expectedImprovementND(const std::vector<float>& x, float best_y)
	{
		std::pair<float, float> prediction = gp_.predict(x);
		float mu = prediction.first;
		float sigma2 = prediction.second;
		if (sigma2 < 1e-12f)
			return 0.0f;
		float sigma = sqrt(sigma2);
		float z = (best_y - mu) / sigma;
		return (best_y - mu) * cdf(z) + sigma * pdf(z);
	}

	void print() const
	{
		std::cout << "Best score: " << best_score_ << std::endl;
		std::cout << "Best params:";
		for (unsigned int i = 0; i < best_params_.size(); ++i)
			std::cout << " " << best_params_[i];
		std::cout << std::endl;
		gp_.print();
	}
};

};

#endif
