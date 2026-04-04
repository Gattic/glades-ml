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
#ifndef GLADES_FISHER_TRANSFORM_H
#define GLADES_FISHER_TRANSFORM_H

#include <cmath>
#include <cstddef>
#include <vector>

namespace glades
{

class FisherTransform
{
public:

	// Scalar Fisher transform: z = arctanh(rho) = 0.5 * ln((1+rho)/(1-rho))
	// Input rho must be in (-1, 1). Values are clamped to [-1+eps, 1-eps].
	static double transform(double rho);

	// Scalar inverse Fisher transform: rho = tanh(z)
	static double inverse(double z);

	// Vector Fisher transform: apply arctanh element-wise
	static std::vector<double> transformVec(const std::vector<double>& rho);

	// Vector inverse Fisher transform: apply tanh element-wise
	static std::vector<double> inverseVec(const std::vector<double>& z);

	// Fisher transform variance: 1 / (n - 3), where n is the effective sample size
	static double variance(size_t effectiveSampleSize);

	// Signal-to-noise ratio: z * sqrt(n - 3)
	static double snr(double z, size_t effectiveSampleSize);

	// Shrinkage function: psi(s) = 1 - exp(-s^2 / 2)
	// Maps SNR to a [0, 1] confidence weight.
	// Small SNR -> 0 (kill the signal), large SNR -> 1 (keep the signal).
	static double shrinkage(double snrValue);

	// Apply Fisher shrinkage to an expected return estimate.
	// adjustedAlpha = alpha * psi(SNR), where SNR = snr(z, n)
	static double adjustedReturn(double alpha, double z, size_t effectiveSampleSize);

	// Compute the Pearson correlation matrix from a data matrix.
	// data: M x N (M samples, N features)
	// Returns a flat N x N correlation matrix (row-major).
	static std::vector<double> correlationMatrix(
		const std::vector<std::vector<double> >& data);

	// Compute the covariance matrix from a data matrix.
	// data: M x N (M samples, N features)
	// Returns a flat N x N covariance matrix (row-major).
	static std::vector<double> covarianceMatrix(
		const std::vector<std::vector<double> >& data);

	// Compute column means of a data matrix.
	// data: M x N (M samples, N features)
	static std::vector<double> columnMeans(const std::vector<std::vector<double> >& data);

	// Compute column standard deviations (sample, using N-1 denominator).
	// data: M x N
	// means: precomputed column means
	static std::vector<double> columnStdDevs(const std::vector<std::vector<double> >& data,
	                                         const std::vector<double>& means);
};

}; // namespace glades

#endif
