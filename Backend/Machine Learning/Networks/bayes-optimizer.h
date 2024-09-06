// Copyright 2020 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <algorithm>

namespace glades {

// Gaussian Process for Regression
class GaussianProcess
{
public:
    GaussianProcess(float length_scale = 1.0, float variance = 1.0, float noise = 1e-5)
        : length_scale_(length_scale), variance_(variance), noise_(noise)
{}

    void addSample(float x, float y);
    void fit();
    void printInput() const;
    void print() const;

    std::pair<float, float> predict(float x) const;

private:
    std::vector<float> X_;                 // Observations (inputs)
    std::vector<float> y_;                 // Observations (outputs)
    std::vector<std::vector<float> > K_;        // Covariance matrix
    std::vector<std::vector<float> > K_inv_;    // Inverse of covariance matrix
    float length_scale_;              // Length scale of the RBF kernel
    float variance_;                  // Variance of the RBF kernel
    float noise_;                     // Noise level

    // Matrix inversion using Gaussian elimination (for small matrices)
    std::vector<std::vector<float> > invertMatrix(const std::vector<std::vector<float> >& matrix) const;

    // Kernel function for Gaussian Process (RBF Kernel)
    static float rbfKernel(float x1, float x2, float length_scale = 1.0, float variance = 1.0)
    {
	return variance * exp(-0.5 * pow((x1 - x2) / length_scale, 2));
    }

};

// Bayesian Optimization with Gaussian Process
class BayesianOptimizer
{
private:
    float best_param_;
    float best_score_;
    GaussianProcess gp_;

public:
    BayesianOptimizer()
    {
        srand(static_cast<unsigned>(time(0)));
    }

    float optimize(const std::vector<std::pair<float, float> >);
    void update(const std::pair<float, float>);
    float getBestParam() const { return best_param_; }
    float getBestScore() const { return best_score_; }
    const GaussianProcess& getGP() const { return gp_; }

    // CDF and PDF functions for the Gaussian distribution
    float cdf(float x)
    {
	return 0.5 * (1 + erf(x / sqrt(2.0)));
    }

    float pdf(float x)
    {
	return exp(-0.5 * x * x) / sqrt(2 * M_PI);
    }

    // Acquisition Function: Expected Improvement
    float expectedImprovement(float x, const GaussianProcess& gp, float best_y)
    {
	std::pair<float, float> prediction = gp.predict(x);
	float mu = prediction.first;
	float sigma2 = prediction.second;
	float sigma = sqrt(sigma2);

	float z = (mu - best_y) / sigma;
	return (mu - best_y) * cdf(z) + sigma * pdf(z);
    }

    void print() const
    {
	std::cout << "Best parameter: " << best_param_ << std::endl;
	std::cout << "Best score: " << best_score_ << std::endl;

	gp_.print();
    }
};

};

#endif
