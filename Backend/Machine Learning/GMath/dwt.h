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
#ifndef GLADES_DWT_H
#define GLADES_DWT_H

#include <cstddef>
#include <vector>

namespace glades
{

// Discrete Wavelet Transform — à trous (stationary / non-decimated) variant.
//
// The à trous algorithm produces one detail coefficient per scale per sample,
// making it shift-invariant and suitable for online / streaming use.
// Supports Haar and Daubechies-4 filter banks.
class DWT
{
public:

	// Filter bank types
	enum FilterType { Haar, Daubechies4 };

	// Get the low-pass (scaling) filter coefficients for a given type.
	static std::vector<double> lowPassFilter(FilterType type);

	// Get the high-pass (wavelet) filter via quadrature mirror relation:
	//   g[k] = (-1)^k * h[L-1-k]
	static std::vector<double> highPassFilter(FilterType type);

	// Full à trous decomposition of a signal into J levels.
	// Returns detail coefficients: result[j] is the detail at scale j (0-indexed),
	// each of length signal.size(). The approximation at level J is returned
	// in the last element result[J].
	// Total output: J+1 vectors, each of length signal.size().
	static std::vector<std::vector<double> > decompose(
		const std::vector<double>& signal, int J, FilterType type = Daubechies4);

	// Compute a single à trous detail coefficient at scale j for the sample at
	// position n, given the approximation coefficients at level j-1.
	// approx: the approximation signal at level j-1 (or the original signal for j=0).
	// n: sample index (must be >= 0)
	// j: scale level (1-indexed: j=1 is finest detail)
	// h, g: low-pass and high-pass filters
	// Returns the detail coefficient d_j[n].
	static double detailCoeff(const std::vector<double>& approx, int n, int j,
	                          const std::vector<double>& h,
	                          const std::vector<double>& g);

	// Compute the approximation coefficient a_j[n] from a_{j-1}.
	static double approxCoeff(const std::vector<double>& approx, int n, int j,
	                          const std::vector<double>& h);

	// Scale energy: mean of squared detail coefficients over a window.
	// detail: detail coefficients at one scale
	// windowEnd: last index (inclusive) in the window
	// windowSize: number of coefficients to use
	static double scaleEnergy(const std::vector<double>& detail,
	                          size_t windowEnd, size_t windowSize);

	// Total wavelet variance: sum of scale energies across J levels.
	static double waveletVariance(const std::vector<std::vector<double> >& details,
	                              int J, size_t windowEnd, size_t windowSize);
};

}; // namespace glades

#endif
