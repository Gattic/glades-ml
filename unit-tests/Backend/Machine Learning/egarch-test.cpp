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
#include "egarch-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/GMath/egarch.h"
#include <cmath>
#include <vector>
#include <cstdio>

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

namespace
{

static const double E_ABS_Z = 0.7978845608028654; // sqrt(2/pi)

// Simple LCG for deterministic pseudo-random numbers
struct LCG
{
	unsigned long state;
	LCG(unsigned long seed) : state(seed) {}
	unsigned long next()
	{
		state = state * 1103515245UL + 12345UL;
		return (state >> 16) & 0x7FFF;
	}
	double uniform()
	{
		return next() / 32768.0;
	}
	double normal()
	{
		double u1 = uniform();
		double u2 = uniform();
		if (u1 < 1e-10) u1 = 1e-10;
		return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
	}
};

// Generate synthetic EGARCH(1,1) returns from known parameters
// log(sigma2_t) = omega + alpha*(|z_{t-1}|-E|z|) + gamma*z_{t-1} + beta*log(sigma2_{t-1})
std::vector<double> generateEgarch11(double omega, double alpha1, double gamma1,
	double beta1, int T, unsigned long seed)
{
	LCG rng(seed);
	// Unconditional log-variance: h = omega / (1 - beta1)
	double hInit = omega / (1.0 - beta1);
	double h = hInit;
	double z = 0.0;

	std::vector<double> returns(T);
	for (int t = 0; t < T; ++t)
	{
		double sigma = sqrt(exp(h));
		z = rng.normal();
		double eps = sigma * z;
		returns[t] = eps;

		// Update log-variance
		double hNew = omega
			+ alpha1 * (fabs(z) - E_ABS_Z)
			+ gamma1 * z
			+ beta1 * h;
		h = hNew;

		// Safety clamp
		if (h < -50.0) h = -50.0;
		if (h > 50.0) h = 50.0;
	}
	return returns;
}

// Generate EGARCH data with strong leverage effect
std::vector<double> generateLeverageData(int T, unsigned long seed)
{
	// gamma = -0.15 means negative returns increase volatility more
	return generateEgarch11(-0.3, 0.15, -0.15, 0.97, T, seed);
}

} // anonymous namespace

void EGARCHUnitTest()
{
	printf("============================================================\n");
	printf("EGARCH Unit Test Suite\n");
	printf("============================================================\n");

	// --- Test 1: Basic EGARCH(1,1) fit convergence ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 1: Basic EGARCH(1,1) fit convergence\n");
	printf("-----------------------------------\n");
	{
		std::vector<double> returns = generateEgarch11(
			-0.2, 0.10, -0.08, 0.98, 5000, 42);

		glades::EGARCH model(1, 1);
		glades::EgarchResult res = model.fit(returns);

		printf("  Converged: %s (iterations: %d)\n",
			res.converged ? "yes" : "no", res.iterations);
		printf("  Log-likelihood: %.4f\n", res.logLikelihood);
		printf("  AIC: %.4f  BIC: %.4f\n", res.AIC, res.BIC);
		printf("  omega=%.4f  alpha=%.4f  gamma=%.4f  beta=%.4f\n",
			model.getOmega(), model.getAlpha()[0],
			model.getGamma()[0], model.getBeta()[0]);
		printf("  persistence=%.4f\n", model.getPersistence());

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Fit should converge==============",
			res.converged);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Model should be fitted==============",
			model.isFitted());
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Log-likelihood should be finite==============",
			res.logLikelihood == res.logLikelihood && res.logLikelihood < 1e15);
	}

	// --- Test 2: Leverage effect detection ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 2: Leverage effect detection (gamma < 0)\n");
	printf("-----------------------------------\n");
	{
		std::vector<double> returns = generateLeverageData(5000, 123);

		glades::EGARCH model(1, 1);
		model.fit(returns);

		double gammaVal = model.getGamma()[0];
		printf("  gamma=%.6f  hasLeverageEffect=%s\n",
			gammaVal, model.hasLeverageEffect() ? "yes" : "no");

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::gamma should be negative for leverage data==============",
			gammaVal < 0.0);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::hasLeverageEffect() should return true==============",
			model.hasLeverageEffect());
	}

	// --- Test 3: Conditional variances always positive ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 3: Conditional variances always positive (exp guarantees this)\n");
	printf("-----------------------------------\n");
	{
		std::vector<double> returns = generateEgarch11(
			-0.15, 0.12, -0.06, 0.97, 3000, 77);

		glades::EGARCH model(1, 1);
		model.fit(returns);

		const std::vector<double>& condVar = model.getConditionalVariances();
		const std::vector<double>& resids = model.getResiduals();

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Conditional variances size must match returns==============",
			condVar.size() == returns.size());
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Residuals size must match returns==============",
			resids.size() == returns.size());

		bool allPositive = true;
		for (size_t t = 0; t < condVar.size(); ++t)
		{
			if (condVar[t] <= 0.0)
			{
				allPositive = false;
				break;
			}
		}
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::All conditional variances must be positive==============",
			allPositive);
	}

	// --- Test 4: Standardized residuals variance near 1 ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 4: Standardized residuals variance near 1\n");
	printf("-----------------------------------\n");
	{
		std::vector<double> returns = generateEgarch11(
			-0.2, 0.10, -0.05, 0.97, 4000, 88);

		glades::EGARCH model(1, 1);
		model.fit(returns);

		const std::vector<double>& resids = model.getResiduals();

		double residMean = 0.0;
		for (size_t t = 0; t < resids.size(); ++t)
			residMean += resids[t];
		residMean /= resids.size();

		double residVar = 0.0;
		for (size_t t = 0; t < resids.size(); ++t)
			residVar += (resids[t] - residMean) * (resids[t] - residMean);
		residVar /= resids.size();

		printf("  Standardized residuals: mean=%.6f  variance=%.6f\n",
			residMean, residVar);

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Standardized residual variance should be near 1==============",
			fabs(residVar - 1.0) < 0.3);
	}

	// --- Test 5: Forecast convergence to unconditional variance ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 5: Forecast converges to unconditional variance\n");
	printf("-----------------------------------\n");
	{
		std::vector<double> returns = generateEgarch11(
			-0.3, 0.10, -0.05, 0.95, 3000, 55);

		glades::EGARCH model(1, 1);
		model.fit(returns);

		double unconditionalVar = model.getUnconditionalVariance();
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Unconditional variance must be positive==============",
			unconditionalVar > 0.0);

		std::vector<double> fcast = model.forecast(500);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Forecast length must match horizon==============",
			(int)fcast.size() == 500);

		double lastFcast = fcast[499];
		double relError = fabs(lastFcast - unconditionalVar) / unconditionalVar;
		printf("  Unconditional variance: %.8f\n", unconditionalVar);
		printf("  Forecast at h=500:     %.8f\n", lastFcast);
		printf("  Relative error:        %.6f\n", relError);

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Long-horizon forecast should converge to unconditional var==============",
			relError < 0.05);

		// Forecast volatility should be sqrt of variance forecast
		std::vector<double> volFcast = model.forecastVolatility(10);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Volatility forecast length must match horizon==============",
			(int)volFcast.size() == 10);

		std::vector<double> varFcast = model.forecast(10);
		bool volCorrect = true;
		for (int h = 0; h < 10; ++h)
		{
			if (fabs(volFcast[h] - sqrt(varFcast[h])) > 1e-10)
			{
				volCorrect = false;
				break;
			}
		}
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Volatility forecast must equal sqrt of variance forecast==============",
			volCorrect);
	}

	// --- Test 6: Unfitted model guards ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 6: Unfitted model returns empty forecasts\n");
	printf("-----------------------------------\n");
	{
		glades::EGARCH model(1, 1);

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Unfitted model isFitted() should be false==============",
			!model.isFitted());

		std::vector<double> fcast = model.forecast(10);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Unfitted model forecast should be empty==============",
			fcast.empty());
	}

	// --- Test 7: Insufficient data ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 7: Insufficient data handling\n");
	printf("-----------------------------------\n");
	{
		std::vector<double> tinyData;
		tinyData.push_back(0.01);
		tinyData.push_back(-0.02);
		tinyData.push_back(0.005);

		glades::EGARCH model(1, 1);
		glades::EgarchResult res = model.fit(tinyData);

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Insufficient data should not mark fitted==============",
			!model.isFitted());
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Insufficient data should not converge==============",
			!res.converged);
	}

	// --- Test 8: Higher order EGARCH(2,1) fit ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 8: Higher order EGARCH(2,1) fit\n");
	printf("-----------------------------------\n");
	{
		std::vector<double> returns = generateEgarch11(
			-0.2, 0.10, -0.08, 0.96, 3000, 99);

		glades::EGARCH model(1, 2);
		glades::EgarchResult res = model.fit(returns);

		printf("  Converged: %s (iterations: %d)\n",
			res.converged ? "yes" : "no", res.iterations);
		printf("  omega=%.4f  alpha=[%.4f, %.4f]  gamma=[%.4f, %.4f]  beta=[%.4f]\n",
			model.getOmega(),
			model.getAlpha()[0], model.getAlpha()[1],
			model.getGamma()[0], model.getGamma()[1],
			model.getBeta()[0]);
		printf("  persistence=%.4f\n", model.getPersistence());

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Higher order model should converge==============",
			res.converged);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Higher order persistence must be < 1==============",
			model.getPersistence() < 1.0);
	}

	// --- Test 9: AIC/BIC consistency ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 9: AIC/BIC consistency\n");
	printf("-----------------------------------\n");
	{
		std::vector<double> returns = generateEgarch11(
			-0.2, 0.10, -0.05, 0.97, 2000, 31);

		glades::EGARCH model11(1, 1);
		glades::EgarchResult res11 = model11.fit(returns);

		glades::EGARCH model21(1, 2);
		glades::EgarchResult res21 = model21.fit(returns);

		printf("  EGARCH(1,1): AIC=%.4f  BIC=%.4f\n", res11.AIC, res11.BIC);
		printf("  EGARCH(1,2): AIC=%.4f  BIC=%.4f\n", res21.AIC, res21.BIC);

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::AIC(1,1) should be finite==============",
			res11.AIC == res11.AIC && res11.AIC > -1e15 && res11.AIC < 1e15);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::BIC(1,1) should be finite==============",
			res11.BIC == res11.BIC && res11.BIC > -1e15 && res11.BIC < 1e15);

		// BIC should favor simpler model for data from EGARCH(1,1)
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::BIC should favor true model order==============",
			res11.BIC <= res21.BIC);
	}

	// --- Test 10: Order selection ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 10: Automatic order selection\n");
	printf("-----------------------------------\n");
	{
		std::vector<double> returns = generateEgarch11(
			-0.2, 0.10, -0.05, 0.97, 4000, 7);

		std::pair<int, int> bestOrder = glades::EGARCH::selectOrder(returns, 2, 2);
		printf("  Selected order: EGARCH(%d,%d)\n",
			bestOrder.first, bestOrder.second);

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Order selection p must be >= 1==============",
			bestOrder.first >= 1);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Order selection p must be <= maxP==============",
			bestOrder.first <= 2);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Order selection q must be >= 1==============",
			bestOrder.second >= 1);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Order selection q must be <= maxQ==============",
			bestOrder.second <= 2);

		glades::EGARCH selectedModel(bestOrder.first, bestOrder.second);
		glades::EgarchResult res = selectedModel.fit(returns);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Selected order model should converge==============",
			res.converged);
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Selected order model persistence must be < 1==============",
			selectedModel.getPersistence() < 1.0);
	}

	// --- Test 11: No leverage effect when gamma = 0 ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 11: No leverage when generated without asymmetry\n");
	printf("-----------------------------------\n");
	{
		// Generate with gamma=0 (symmetric)
		std::vector<double> returns = generateEgarch11(
			-0.2, 0.10, 0.0, 0.97, 5000, 2024);

		glades::EGARCH model(1, 1);
		model.fit(returns);

		double gammaVal = model.getGamma()[0];
		printf("  gamma=%.6f (generated with gamma=0)\n", gammaVal);

		// Fitted gamma should be near zero for symmetric data
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::gamma should be near zero for symmetric data==============",
			fabs(gammaVal) < 0.10);
	}

	// --- Test 12: Forecast mean-reversion behavior ---
	printf("-----------------------------------\n");
	printf("EGARCH Test 12: Forecast mean-reversion behavior\n");
	printf("-----------------------------------\n");
	{
		std::vector<double> returns = generateEgarch11(
			-0.5, 0.12, -0.06, 0.90, 3000, 444);

		glades::EGARCH model(1, 1);
		model.fit(returns);

		std::vector<double> fcast = model.forecast(100);
		double unconditionalVar = model.getUnconditionalVariance();

		// All forecasts should be positive (guaranteed by exp)
		bool allPos = true;
		for (size_t h = 0; h < fcast.size(); ++h)
		{
			if (fcast[h] <= 0.0)
			{
				allPos = false;
				break;
			}
		}
		G_assert(__FILE__, __LINE__,
			"==============EGARCH::All forecast values must be positive==============",
			allPos);

		double earlyDiff = fabs(fcast[0] - unconditionalVar);
		double lateDiff = fabs(fcast[99] - unconditionalVar);
		printf("  Early diff from uncond var: %.8f\n", earlyDiff);
		printf("  Late diff from uncond var:  %.8f\n", lateDiff);

		G_assert(__FILE__, __LINE__,
			"==============EGARCH::Late forecast should be closer to unconditional var==============",
			lateDiff <= earlyDiff + 1e-10);
	}

	printf("============================================================\n");
	printf("EGARCH Unit Tests Complete\n");
	printf("============================================================\n");
}
