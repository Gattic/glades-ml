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
#include "permutation_entropy.h"
#include <cmath>
#include <algorithm>

using namespace glades;

int PermutationEntropy::factorial(int m)
{
	int f = 1;
	for (int i = 2; i <= m; ++i)
		f *= i;
	return f;
}

// Encode a permutation as an index in [0, m!-1] using the Lehmer code.
// For each position i, count how many elements to the right are smaller
// (this is the inversion count). The index is:
//   idx = Σ c_i * (m-1-i)!
int PermutationEntropy::ordinalPattern(const double* vec, int m)
{
	// Build rank array: rank[i] = rank of vec[i] among all m elements
	// Ties broken by index (earlier index = lower rank)
	int idx = 0;
	int fact = factorial(m - 1);

	// For each position, count elements to the right that are smaller
	for (int i = 0; i < m - 1; ++i)
	{
		int count = 0;
		for (int j = i + 1; j < m; ++j)
		{
			if (vec[j] < vec[i] || (vec[j] == vec[i] && j < i))
				++count;
		}
		idx += count * fact;
		if (m - 2 - i > 0)
			fact /= (m - 1 - i);
	}
	return idx;
}

double PermutationEntropy::compute(const std::vector<double>& series, int m, int tau)
{
	return computeSmoothed(series, m, tau, series.size(), 0.0);
}

double PermutationEntropy::computeWindowed(const std::vector<double>& series,
                                           int m, int tau, size_t W)
{
	return computeSmoothed(series, m, tau, W, 0.0);
}

double PermutationEntropy::computeSmoothed(const std::vector<double>& series,
                                           int m, int tau, size_t W, double alpha)
{
	int N = (int)series.size();
	int mFact = factorial(m);
	double logMFact = std::log((double)mFact);

	if (logMFact < 1e-15)
		return 0.0;

	// Determine window bounds
	int start = 0;
	if ((size_t)N > W)
		start = N - (int)W;

	// Minimum length needed: (m-1)*tau + 1 vectors
	int minLen = (m - 1) * tau + 1;
	int windowLen = N - start;
	if (windowLen < minLen)
		return 0.0;

	// Count ordinal patterns
	std::vector<int> counts(mFact, 0);
	int numVectors = 0;

	// Construct delay vectors and count patterns
	// A delay vector at position t is: (series[t], series[t-tau], ..., series[t-(m-1)*tau])
	for (int t = start + (m - 1) * tau; t < N; ++t)
	{
		// Build the delay vector
		double vec[8]; // m <= 7 is practical
		for (int k = 0; k < m; ++k)
			vec[k] = series[t - k * tau];

		int pattern = ordinalPattern(vec, m);
		if (pattern >= 0 && pattern < mFact)
			counts[pattern]++;
		numVectors++;
	}

	if (numVectors == 0)
		return 0.0;

	// Compute Shannon entropy with optional Laplace smoothing
	double totalCount = (double)numVectors + alpha * (double)mFact;
	double entropy = 0.0;
	for (int i = 0; i < mFact; ++i)
	{
		double p = ((double)counts[i] + alpha) / totalCount;
		if (p > 0.0)
			entropy -= p * std::log(p);
	}

	// Normalize to [0, 1]
	return entropy / logMFact;
}
