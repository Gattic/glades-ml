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
#ifndef GLADES_PERMUTATION_ENTROPY_H
#define GLADES_PERMUTATION_ENTROPY_H

#include <cstddef>
#include <vector>

namespace glades
{

// Permutation Entropy (Bandt & Pompe, 2002)
//
// Measures the complexity/predictability of a time series by counting
// the relative frequency of ordinal patterns in delay-embedded vectors.
// Returns H ∈ [0, 1] where 1 = maximum complexity (i.i.d.) and 0 = fully
// deterministic (single repeating pattern).
//
// The ordinal pattern of a vector (x₁, x₂, ..., xₘ) is the permutation
// that sorts its components. For embedding dimension m, there are m! possible
// patterns.
class PermutationEntropy
{
public:

	// Compute the normalized permutation entropy of a time series.
	// series: input data (at least m values long)
	// m: embedding dimension (typically 3, 4, or 5)
	// tau: delay (typically 1)
	// Returns H ∈ [0, 1], normalized by ln(m!).
	static double compute(const std::vector<double>& series, int m, int tau = 1);

	// Compute over a window: uses only the last W elements of the series.
	// If series.size() < W, uses all available data.
	static double computeWindowed(const std::vector<double>& series,
	                              int m, int tau, size_t W);

	// Compute with Laplace smoothing (pseudocount alpha added to each pattern).
	// Eliminates zero-frequency discontinuities at small cost in bias.
	static double computeSmoothed(const std::vector<double>& series,
	                              int m, int tau, size_t W, double alpha);

	// Map a delay vector of dimension m to its ordinal pattern index.
	// The index is in [0, m!-1] and uniquely identifies the permutation.
	// Uses factorial number system encoding.
	static int ordinalPattern(const double* vec, int m);

	// Factorial: m! (for normalization)
	static int factorial(int m);
};

}; // namespace glades

#endif
