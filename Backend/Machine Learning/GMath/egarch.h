// Copyright 2024 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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
#ifndef _GEGARCH
#define _GEGARCH

#include <vector>
#include <cmath>
#include <cstdio>
#include <utility>

namespace glades
{

struct EgarchResult
{
	double logLikelihood;
	double AIC;
	double BIC;
	int iterations;
	bool converged;

	EgarchResult()
		: logLikelihood(0.0)
		, AIC(0.0)
		, BIC(0.0)
		, iterations(0)
		, converged(false)
	{
	}
};

class EGARCH
{
private:

	int p;
	int q;
	int maxIterations;
	double tolerance;
	bool fitted;

	// Fitted parameters
	// log(sigma2_t) = omega
	//   + sum_i [ alpha_i * (|z_{t-i}| - E|z|) + gamma_i * z_{t-i} ]
	//   + sum_j beta_j * log(sigma2_{t-j})
	double omega;
	std::vector<double> alpha;   // magnitude effect
	std::vector<double> gamma;   // asymmetry / leverage
	std::vector<double> beta;    // log-variance persistence

	// Fitted state
	std::vector<double> returns;
	std::vector<double> conditionalVariances;
	std::vector<double> residuals;
	EgarchResult result;

	// Parameter transformation (only beta needs tanh for |beta_j| < 1)
	std::vector<double> fromUnconstrained(const std::vector<double>& phi) const;
	std::vector<double> toUnconstrained() const;
	void transformGradient(const std::vector<double>& phi,
		const std::vector<double>& gradTheta,
		std::vector<double>& gradPhi) const;

	// MLE core
	double negLogLikelihood(const std::vector<double>& theta,
		std::vector<double>& gradient) const;

	// BFGS optimizer
	bool bfgsMinimize(std::vector<double>& phi, double& fVal, int& iters);
	double lineSearch(const std::vector<double>& phi,
		const std::vector<double>& direction,
		double fCurrent,
		const std::vector<double>& gradCurrent,
		double& stepSize) const;

	// Initialization
	void initializeParameters();

public:

	EGARCH(int p = 1, int q = 1, int maxIterations = 500, double tolerance = 1e-6);

	// Fit the model to return series
	EgarchResult fit(const std::vector<double>& returns);

	// Forecast h-step-ahead conditional variance
	std::vector<double> forecast(int h) const;

	// Forecast h-step-ahead volatility (sqrt of variance)
	std::vector<double> forecastVolatility(int h) const;

	// Accessors
	double getOmega() const;
	const std::vector<double>& getAlpha() const;
	const std::vector<double>& getGamma() const;
	const std::vector<double>& getBeta() const;
	double getPersistence() const;
	double getUnconditionalVariance() const;
	const std::vector<double>& getConditionalVariances() const;
	const std::vector<double>& getResiduals() const;
	const EgarchResult& getResult() const;
	bool isFitted() const;
	bool hasLeverageEffect() const;

	// Order selection via BIC
	static std::pair<int, int> selectOrder(const std::vector<double>& returns,
		int maxP = 3, int maxQ = 3);
};

}; // namespace glades

#endif
