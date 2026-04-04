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
#include "dwt.h"
#include <cmath>
#include <algorithm>

using namespace glades;

std::vector<double> DWT::lowPassFilter(FilterType type)
{
	if (type == Haar)
	{
		double s = 1.0 / std::sqrt(2.0);
		std::vector<double> h(2);
		h[0] = s;
		h[1] = s;
		return h;
	}

	// Daubechies-4 (8 taps)
	static const double c[] = {
		 0.23037781330885523,
		 0.71484657055291565,
		 0.63088076792959036,
		-0.02798376941685985,
		-0.18703481171888114,
		 0.03084138183598697,
		 0.03288301166698295,
		-0.01059740178499664
	};
	std::vector<double> h(8);
	for (int i = 0; i < 8; ++i)
		h[i] = c[i];
	return h;
}

std::vector<double> DWT::highPassFilter(FilterType type)
{
	std::vector<double> h = lowPassFilter(type);
	int L = (int)h.size();
	std::vector<double> g(L);
	for (int k = 0; k < L; ++k)
	{
		double sign = (k % 2 == 0) ? 1.0 : -1.0;
		g[k] = sign * h[L - 1 - k];
	}
	return g;
}

double DWT::detailCoeff(const std::vector<double>& approx, int n, int j,
                        const std::vector<double>& /*h*/,
                        const std::vector<double>& g)
{
	int L = (int)g.size();
	int step = 1;
	for (int s = 1; s < j; ++s)
		step *= 2;
	// step = 2^(j-1)

	double val = 0.0;
	int N = (int)approx.size();
	for (int m = 0; m < L; ++m)
	{
		int idx = n - m * step;
		// Periodic boundary (wrap around)
		while (idx < 0)
			idx += N;
		val += g[m] * approx[idx];
	}
	return val;
}

double DWT::approxCoeff(const std::vector<double>& approx, int n, int j,
                        const std::vector<double>& h)
{
	int L = (int)h.size();
	int step = 1;
	for (int s = 1; s < j; ++s)
		step *= 2;

	double val = 0.0;
	int N = (int)approx.size();
	for (int m = 0; m < L; ++m)
	{
		int idx = n - m * step;
		while (idx < 0)
			idx += N;
		val += h[m] * approx[idx];
	}
	return val;
}

std::vector<std::vector<double> > DWT::decompose(
	const std::vector<double>& signal, int J, FilterType type)
{
	std::vector<double> h = lowPassFilter(type);
	std::vector<double> g = highPassFilter(type);
	int N = (int)signal.size();

	// result[0..J-1] = detail coefficients at scales 1..J
	// result[J] = approximation at level J
	std::vector<std::vector<double> > result(J + 1);
	for (int i = 0; i <= J; ++i)
		result[i].resize(N, 0.0);

	// Current approximation (starts as the input signal)
	std::vector<double> a_prev = signal;

	for (int j = 1; j <= J; ++j)
	{
		std::vector<double> a_next(N, 0.0);
		for (int n = 0; n < N; ++n)
		{
			result[j - 1][n] = detailCoeff(a_prev, n, j, h, g);
			a_next[n] = approxCoeff(a_prev, n, j, h);
		}
		a_prev = a_next;
	}

	result[J] = a_prev;
	return result;
}

double DWT::scaleEnergy(const std::vector<double>& detail,
                        size_t windowEnd, size_t windowSize)
{
	if (detail.empty() || windowSize == 0)
		return 0.0;

	size_t N = detail.size();
	if (windowEnd >= N)
		windowEnd = N - 1;

	size_t count = 0;
	double sum = 0.0;
	for (size_t i = 0; i < windowSize; ++i)
	{
		if (windowEnd < i)
			break;
		size_t idx = windowEnd - i;
		sum += detail[idx] * detail[idx];
		++count;
	}

	return (count > 0) ? sum / (double)count : 0.0;
}

double DWT::waveletVariance(const std::vector<std::vector<double> >& details,
                            int J, size_t windowEnd, size_t windowSize)
{
	double total = 0.0;
	for (int j = 0; j < J; ++j)
		total += scaleEnergy(details[j], windowEnd, windowSize);
	return total;
}
