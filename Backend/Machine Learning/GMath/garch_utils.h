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
#ifndef GLADES_GARCH_UTILS_H
#define GLADES_GARCH_UTILS_H

#include <cmath>
#include <cstddef>
#include <vector>

namespace glades
{

// Generic GARCH/EGARCH diagnostic and risk utilities.
// Works with any model that provides standardized residuals and conditional variances.
class GarchUtils
{
public:

	// Ljung-Box test for residual autocorrelation.
	// Tests H0: residuals are independently distributed (no remaining autocorrelation).
	// residuals: standardized residuals (eps_t / sigma_t) from a fitted GARCH/EGARCH model.
	// maxLag: number of lags to test (typically 10-20).
	// Returns the Q-statistic. Compare against chi-squared(maxLag - p - q) critical values:
	//   5% critical: ~chi2_inv(0.95, df). If Q > critical, reject H0 (model misspecified).
	static double ljungBoxStatistic(const std::vector<double>& residuals, int maxLag = 10);

	// Sample autocorrelation function of a series at a given lag.
	static double autocorrelation(const std::vector<double>& series, int lag);

	// Value at Risk (VaR) at confidence level alpha (e.g., 0.95 or 0.99).
	// Uses the conditional variance forecast and assumes Gaussian innovations.
	// forecastVariance: 1-step-ahead conditional variance from GARCH.forecast(1)[0]
	// alpha: confidence level (0.95 = 5% tail)
	// Returns the VaR as a positive number (the loss threshold).
	static double valueAtRisk(double forecastVariance, double alpha = 0.95);

	// Conditional Value at Risk (CVaR / Expected Shortfall) at confidence level alpha.
	// The expected loss given that loss exceeds VaR. More tail-sensitive than VaR.
	// Assumes Gaussian innovations.
	static double conditionalVaR(double forecastVariance, double alpha = 0.95);

	// Standardize a return series using GARCH conditional variances.
	// returns: raw return series
	// conditionalVariances: sigma2_t from a fitted model
	// Output: z_t = r_t / sqrt(sigma2_t)
	// Useful for feeding to spectral analysis (removes heteroskedasticity).
	static std::vector<double> standardize(const std::vector<double>& returns,
	                                       const std::vector<double>& conditionalVariances);

	// Inverse standard normal CDF (probit function).
	// Used internally for VaR computation. Rational approximation, accurate to ~1e-9.
	static double normalQuantile(double p);

	// Standard normal PDF.
	static double normalPDF(double x);
};

}; // namespace glades

#endif
