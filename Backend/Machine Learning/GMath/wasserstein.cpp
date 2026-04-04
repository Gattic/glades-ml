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
#include "wasserstein.h"
#include <cmath>
#include <algorithm>
#include <cstdlib>

using namespace glades;

double Wasserstein::distance1D(const std::vector<double>& a, const std::vector<double>& b)
{
	if (a.empty() || b.empty())
		return 0.0;

	std::vector<double> sa = a;
	std::vector<double> sb = b;
	std::sort(sa.begin(), sa.end());
	std::sort(sb.begin(), sb.end());

	// Equal size: simple sorted matching
	size_t n = std::min(sa.size(), sb.size());
	if (sa.size() == sb.size())
	{
		double sum = 0.0;
		for (size_t i = 0; i < n; ++i)
			sum += std::fabs(sa[i] - sb[i]);
		return sum / (double)n;
	}

	return distance1DUnequal(a, b);
}

double Wasserstein::distance1DUnequal(const std::vector<double>& a,
                                      const std::vector<double>& b)
{
	if (a.empty() || b.empty())
		return 0.0;

	std::vector<double> sa = a;
	std::vector<double> sb = b;
	std::sort(sa.begin(), sa.end());
	std::sort(sb.begin(), sb.end());

	size_t na = sa.size();
	size_t nb = sb.size();

	// Merge all unique values
	std::vector<double> merged;
	merged.reserve(na + nb);
	for (size_t i = 0; i < na; ++i) merged.push_back(sa[i]);
	for (size_t i = 0; i < nb; ++i) merged.push_back(sb[i]);
	std::sort(merged.begin(), merged.end());

	// Integrate |F_a(x) - F_b(x)| dx over intervals between consecutive points
	double integral = 0.0;
	size_t ia = 0, ib = 0;
	for (size_t k = 1; k < merged.size(); ++k)
	{
		double x_prev = merged[k - 1];
		double x_curr = merged[k];
		double dx = x_curr - x_prev;
		if (dx <= 0.0) continue;

		// CDF values at x_prev
		while (ia < na && sa[ia] <= x_prev) ++ia;
		while (ib < nb && sb[ib] <= x_prev) ++ib;
		double fa = (double)ia / (double)na;
		double fb = (double)ib / (double)nb;

		integral += std::fabs(fa - fb) * dx;
	}

	return integral;
}

double Wasserstein::slicedDistance(const std::vector<double>& P, size_t n_p,
                                  const std::vector<double>& Q, size_t n_q,
                                  size_t d,
                                  const std::vector<double>& projections, size_t K)
{
	if (n_p == 0 || n_q == 0 || d == 0 || K == 0)
		return 0.0;

	double total = 0.0;

	for (size_t k = 0; k < K; ++k)
	{
		// Project P and Q onto direction k
		const double* theta = &projections[k * d];

		std::vector<double> proj_p(n_p);
		for (size_t i = 0; i < n_p; ++i)
		{
			double dot = 0.0;
			for (size_t j = 0; j < d; ++j)
				dot += P[i * d + j] * theta[j];
			proj_p[i] = dot;
		}

		std::vector<double> proj_q(n_q);
		for (size_t i = 0; i < n_q; ++i)
		{
			double dot = 0.0;
			for (size_t j = 0; j < d; ++j)
				dot += Q[i * d + j] * theta[j];
			proj_q[i] = dot;
		}

		total += distance1D(proj_p, proj_q);
	}

	return total / (double)K;
}

// Simple LCG for reproducible random generation (C++98 compatible)
static double lcgNextDouble(unsigned& state)
{
	state = state * 1103515245u + 12345u;
	return (double)((state >> 16) & 0x7FFF) / 32767.0;
}

static double lcgGauss(unsigned& state)
{
	double u1 = lcgNextDouble(state);
	double u2 = lcgNextDouble(state);
	if (u1 < 1e-10) u1 = 1e-10;
	return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

std::vector<double> Wasserstein::randomProjections(size_t d, size_t K, unsigned seed)
{
	unsigned state = seed;
	std::vector<double> proj(K * d);
	for (size_t k = 0; k < K; ++k)
	{
		double norm = 0.0;
		for (size_t j = 0; j < d; ++j)
		{
			double g = lcgGauss(state);
			proj[k * d + j] = g;
			norm += g * g;
		}
		norm = std::sqrt(norm);
		if (norm > 1e-15)
		{
			for (size_t j = 0; j < d; ++j)
				proj[k * d + j] /= norm;
		}
	}

	return proj;
}

double Wasserstein::tailAsymmetry(const std::vector<double>& a,
                                  const std::vector<double>& b,
                                  double alpha)
{
	if (a.empty() || b.empty())
		return 0.0;

	std::vector<double> sa = a;
	std::vector<double> sb = b;
	std::sort(sa.begin(), sa.end());
	std::sort(sb.begin(), sb.end());

	// Use the smaller size for quantile matching
	size_t n = std::min(sa.size(), sb.size());
	if (n < 4) return 0.0;

	// Resample to equal size if needed (linear interpolation of quantiles)
	std::vector<double> qa(n), qb(n);
	for (size_t i = 0; i < n; ++i)
	{
		double u = ((double)i + 0.5) / (double)n;

		// Interpolate quantile from sa
		double pos_a = u * (double)(sa.size() - 1);
		size_t lo_a = (size_t)pos_a;
		double frac_a = pos_a - (double)lo_a;
		if (lo_a >= sa.size() - 1) { qa[i] = sa.back(); }
		else { qa[i] = sa[lo_a] * (1.0 - frac_a) + sa[lo_a + 1] * frac_a; }

		double pos_b = u * (double)(sb.size() - 1);
		size_t lo_b = (size_t)pos_b;
		double frac_b = pos_b - (double)lo_b;
		if (lo_b >= sb.size() - 1) { qb[i] = sb.back(); }
		else { qb[i] = sb[lo_b] * (1.0 - frac_b) + sb[lo_b + 1] * frac_b; }
	}

	// Compute tail-weighted asymmetry: A* = Σ w(u) * sign(u-0.5) * (qa-qb) / Σ w(u)
	double num = 0.0, den = 0.0;
	for (size_t i = 0; i < n; ++i)
	{
		double u = ((double)i + 0.5) / (double)n;
		double w = std::pow(std::fabs(u - 0.5) / 0.5, alpha);
		double sign = (u >= 0.5) ? 1.0 : -1.0;
		double delta = qa[i] - qb[i];
		num += w * sign * delta;
		den += w;
	}

	return (den > 1e-15) ? num / den : 0.0;
}
