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
#include "garch_utils.h"
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace glades
{

double GarchUtils::autocorrelation(const std::vector<double>& series, int lag)
{
	size_t n = series.size();
	if (n < 2 || lag <= 0 || static_cast<size_t>(lag) >= n)
		return 0.0;

	// Mean
	double mean = 0.0;
	for (size_t i = 0; i < n; ++i)
		mean += series[i];
	mean /= static_cast<double>(n);

	// Autocovariance at given lag and lag 0
	double cov_lag = 0.0;
	double cov_0 = 0.0;
	for (size_t i = 0; i < n; ++i)
	{
		double d = series[i] - mean;
		cov_0 += d * d;
		if (i >= static_cast<size_t>(lag))
		{
			double d_lag = series[i - lag] - mean;
			cov_lag += d * d_lag;
		}
	}

	if (cov_0 < 1e-30)
		return 0.0;
	return cov_lag / cov_0;
}

double GarchUtils::ljungBoxStatistic(const std::vector<double>& residuals, int maxLag)
{
	size_t n = residuals.size();
	if (n < 10 || maxLag <= 0)
		return 0.0;

	// Squared residuals (test for remaining ARCH effects)
	std::vector<double> sq(n);
	for (size_t i = 0; i < n; ++i)
		sq[i] = residuals[i] * residuals[i];

	// Ljung-Box Q-statistic: Q = n(n+2) * sum_{k=1}^{m} rho_k^2 / (n-k)
	double Q = 0.0;
	double nn = static_cast<double>(n);
	for (int k = 1; k <= maxLag; ++k)
	{
		double rho_k = autocorrelation(sq, k);
		Q += (rho_k * rho_k) / (nn - static_cast<double>(k));
	}
	Q *= nn * (nn + 2.0);

	return Q;
}

double GarchUtils::normalPDF(double x)
{
	return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

double GarchUtils::normalQuantile(double p)
{
	// Peter Acklam's rational approximation to the inverse normal CDF.
	// Accurate to ~1.15e-9 for p in (0, 1).
	if (p <= 0.0) return -1e15;
	if (p >= 1.0) return 1e15;

	static const double a[] = {
		-3.969683028665376e+01, 2.209460984245205e+02,
		-2.759285104469687e+02, 1.383577518672690e+02,
		-3.066479806614716e+01, 2.506628277459239e+00
	};
	static const double b[] = {
		-5.447609879822406e+01, 1.615858368580409e+02,
		-1.556989798598866e+02, 6.680131188771972e+01,
		-1.328068155288572e+01
	};
	static const double c[] = {
		-7.784894002430293e-03, -3.223964580411365e-01,
		-2.400758277161838e+00, -2.549732539343734e+00,
		4.374664141464968e+00, 2.938163982698783e+00
	};
	static const double d[] = {
		7.784695709041462e-03, 3.224671290700398e-01,
		2.445134137142996e+00, 3.754408661907416e+00
	};

	double q, r;
	if (p < 0.02425)
	{
		q = std::sqrt(-2.0 * std::log(p));
		return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
		       ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
	}
	else if (p <= 0.97575)
	{
		q = p - 0.5;
		r = q * q;
		return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
		       (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
	}
	else
	{
		q = std::sqrt(-2.0 * std::log(1.0 - p));
		return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
		        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
	}
}

double GarchUtils::valueAtRisk(double forecastVariance, double alpha)
{
	if (forecastVariance <= 0.0) return 0.0;
	double sigma = std::sqrt(forecastVariance);
	double z = normalQuantile(alpha);
	return sigma * z; // Positive number representing the loss threshold
}

double GarchUtils::conditionalVaR(double forecastVariance, double alpha)
{
	if (forecastVariance <= 0.0) return 0.0;
	double sigma = std::sqrt(forecastVariance);
	double z_alpha = normalQuantile(alpha);
	// CVaR = sigma * phi(z_alpha) / (1 - alpha) for Gaussian
	double phi_z = normalPDF(z_alpha);
	return sigma * phi_z / (1.0 - alpha);
}

std::vector<double> GarchUtils::standardize(const std::vector<double>& returns,
                                            const std::vector<double>& conditionalVariances)
{
	size_t n = returns.size();
	size_t m = conditionalVariances.size();
	size_t len = (n < m) ? n : m;

	std::vector<double> standardized(len);
	for (size_t i = 0; i < len; ++i)
	{
		double sigma = std::sqrt(conditionalVariances[i]);
		if (sigma > 1e-15)
			standardized[i] = returns[i] / sigma;
		else
			standardized[i] = 0.0;
	}
	return standardized;
}

}; // namespace glades
