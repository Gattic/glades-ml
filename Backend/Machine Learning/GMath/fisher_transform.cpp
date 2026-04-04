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
#include "fisher_transform.h"
#include <cstdio>

namespace glades
{

static const double FISHER_CLAMP_EPS = 1e-10;

double FisherTransform::transform(double rho)
{
	// Clamp to (-1, 1)
	if (rho >= 1.0 - FISHER_CLAMP_EPS)
		rho = 1.0 - FISHER_CLAMP_EPS;
	if (rho <= -1.0 + FISHER_CLAMP_EPS)
		rho = -1.0 + FISHER_CLAMP_EPS;

	// arctanh(rho) = 0.5 * ln((1 + rho) / (1 - rho))
	return 0.5 * std::log((1.0 + rho) / (1.0 - rho));
}

double FisherTransform::inverse(double z)
{
	// tanh(z) = (e^2z - 1) / (e^2z + 1)
	return std::tanh(z);
}

std::vector<double> FisherTransform::transformVec(const std::vector<double>& rho)
{
	std::vector<double> z(rho.size());
	for (size_t i = 0; i < rho.size(); ++i)
		z[i] = transform(rho[i]);
	return z;
}

std::vector<double> FisherTransform::inverseVec(const std::vector<double>& z)
{
	std::vector<double> rho(z.size());
	for (size_t i = 0; i < z.size(); ++i)
		rho[i] = inverse(z[i]);
	return rho;
}

double FisherTransform::variance(size_t effectiveSampleSize)
{
	if (effectiveSampleSize <= 3)
	{
		printf("[FisherTransform] Warning: effective sample size %lu <= 3, "
		       "variance is undefined\n", (unsigned long)effectiveSampleSize);
		return 1e10; // Return large variance to indicate unreliable
	}
	return 1.0 / (double)(effectiveSampleSize - 3);
}

double FisherTransform::snr(double z, size_t effectiveSampleSize)
{
	if (effectiveSampleSize <= 3)
		return 0.0;
	return z * std::sqrt((double)(effectiveSampleSize - 3));
}

double FisherTransform::shrinkage(double snrValue)
{
	// psi(s) = 1 - exp(-s^2 / 2)
	return 1.0 - std::exp(-0.5 * snrValue * snrValue);
}

double FisherTransform::adjustedReturn(double alpha, double z, size_t effectiveSampleSize)
{
	double s = snr(z, effectiveSampleSize);
	double psi = shrinkage(s);
	return alpha * psi;
}

std::vector<double> FisherTransform::columnMeans(
	const std::vector<std::vector<double> >& data)
{
	if (data.empty())
		return std::vector<double>();

	size_t M = data.size();
	size_t N = data[0].size();
	std::vector<double> means(N, 0.0);

	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
			means[j] += data[i][j];

	for (size_t j = 0; j < N; ++j)
		means[j] /= (double)M;

	return means;
}

std::vector<double> FisherTransform::columnStdDevs(
	const std::vector<std::vector<double> >& data,
	const std::vector<double>& means)
{
	if (data.empty())
		return std::vector<double>();

	size_t M = data.size();
	size_t N = data[0].size();
	std::vector<double> stddevs(N, 0.0);

	for (size_t i = 0; i < M; ++i)
		for (size_t j = 0; j < N; ++j)
		{
			double d = data[i][j] - means[j];
			stddevs[j] += d * d;
		}

	for (size_t j = 0; j < N; ++j)
	{
		if (M > 1)
			stddevs[j] = std::sqrt(stddevs[j] / (double)(M - 1));
		else
			stddevs[j] = 0.0;
	}

	return stddevs;
}

std::vector<double> FisherTransform::covarianceMatrix(
	const std::vector<std::vector<double> >& data)
{
	if (data.empty())
		return std::vector<double>();

	size_t M = data.size();
	size_t N = data[0].size();
	std::vector<double> means = columnMeans(data);
	std::vector<double> cov(N * N, 0.0);

	for (size_t s = 0; s < M; ++s)
	{
		for (size_t i = 0; i < N; ++i)
		{
			double di = data[s][i] - means[i];
			for (size_t j = i; j < N; ++j)
			{
				double dj = data[s][j] - means[j];
				cov[i * N + j] += di * dj;
			}
		}
	}

	double denom = (M > 1) ? (double)(M - 1) : 1.0;
	for (size_t i = 0; i < N; ++i)
	{
		for (size_t j = i; j < N; ++j)
		{
			cov[i * N + j] /= denom;
			cov[j * N + i] = cov[i * N + j]; // Symmetric
		}
	}

	return cov;
}

std::vector<double> FisherTransform::correlationMatrix(
	const std::vector<std::vector<double> >& data)
{
	if (data.empty())
		return std::vector<double>();

	size_t M = data.size();
	size_t N = data[0].size();
	std::vector<double> means = columnMeans(data);
	std::vector<double> stddevs = columnStdDevs(data, means);
	std::vector<double> corr(N * N, 0.0);

	// Compute correlation as cov(i,j) / (std(i) * std(j))
	std::vector<double> cov = covarianceMatrix(data);

	for (size_t i = 0; i < N; ++i)
	{
		for (size_t j = 0; j < N; ++j)
		{
			if (stddevs[i] > 1e-15 && stddevs[j] > 1e-15)
				corr[i * N + j] = cov[i * N + j] / (stddevs[i] * stddevs[j]);
			else if (i == j)
				corr[i * N + j] = 1.0;
			else
				corr[i * N + j] = 0.0;
		}
	}

	return corr;
}

}; // namespace glades
