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
#ifndef GLADES_WASSERSTEIN_H
#define GLADES_WASSERSTEIN_H

#include <cstddef>
#include <vector>

namespace glades
{

// Wasserstein (Earth Mover's) distance for 1D and multivariate distributions.
//
// The 1-Wasserstein distance W₁(P,Q) measures the minimum "work" needed to
// transform distribution P into Q. For 1D distributions it has a closed-form
// via sorted quantile matching. For higher dimensions, we use the sliced
// Wasserstein approximation (random 1D projections).
class Wasserstein
{
public:

	// 1D Wasserstein distance between two equal-sized samples.
	// Both vectors are sorted internally (non-destructive).
	// Returns (1/n) * Σ|x_(i) - y_(i)| for sorted samples.
	static double distance1D(const std::vector<double>& a, const std::vector<double>& b);

	// 1D Wasserstein for unequal-sized samples via CDF integration.
	// Returns ∫|F_a(x) - F_b(x)| dx over the merged support.
	static double distance1DUnequal(const std::vector<double>& a,
	                                const std::vector<double>& b);

	// Sliced Wasserstein distance for d-dimensional samples.
	// P: n_p × d matrix (row-major, each row is a sample point)
	// Q: n_q × d matrix
	// d: dimensionality
	// projections: K × d matrix of unit vectors (pre-generated)
	// K: number of projection directions
	// Returns (1/K) * Σ_k W₁(proj_k(P), proj_k(Q))
	static double slicedDistance(const std::vector<double>& P, size_t n_p,
	                             const std::vector<double>& Q, size_t n_q,
	                             size_t d,
	                             const std::vector<double>& projections, size_t K);

	// Generate K random unit vectors on the d-dimensional unit sphere.
	// Returns K × d matrix (row-major). Uses Gaussian sampling + normalization.
	static std::vector<double> randomProjections(size_t d, size_t K, unsigned seed = 42);

	// Tail asymmetry signal: measures whether the right tail or left tail
	// of the probe distribution has expanded more relative to the reference.
	// a: probe sample (will be sorted)
	// b: reference sample (will be sorted)
	// alpha: tail emphasis exponent (>=1; higher = more tail emphasis)
	// Returns A* = weighted(right_tail_shift) - weighted(left_tail_shift)
	// Positive = right tail growing (bullish), negative = left tail growing (bearish).
	static double tailAsymmetry(const std::vector<double>& a,
	                            const std::vector<double>& b,
	                            double alpha = 2.0);
};

}; // namespace glades

#endif
