// Copyright 2024 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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
#include "pca.h"
#include <cmath>
#include <cstdio>

using namespace glades;

static bool compare_eig_pairs_desc(const std::pair<double, size_t>& a, const std::pair<double, size_t>& b)
{
	return a.first > b.first;
}

PCA::PCA()
	: active_components(0)
	, num_features_(0)
	, converged_(false)
	, iteration_count_(0)
	, incremental_count_(0)
	, incremental_target_components_(0)
{
}

void PCA::clearState()
{
	mean_vec.clear();
	eigenvectors_flat.clear();
	eigenvalues.clear();
	variance_explained.clear();
	transformed_data.clear();
	reconstructed_data.clear();
	component_mapping.clear();
	active_components = 0;
	num_features_ = 0;
	converged_ = false;
	iteration_count_ = 0;
}

double PCA::eigvec(size_t component, size_t feature) const
{
	return eigenvectors_flat[component * num_features_ + feature];
}

// Accessors
const std::vector<double>& PCA::getMean() const { return mean_vec; }

std::vector<std::vector<double> > PCA::getEigenvectors() const
{
	std::vector<std::vector<double> > result(num_features_, std::vector<double>(num_features_));
	for (size_t i = 0; i < num_features_; ++i)
		for (size_t j = 0; j < num_features_; ++j)
			result[i][j] = eigenvectors_flat[i * num_features_ + j];
	return result;
}

const std::vector<double>& PCA::getEigenvalues() const { return eigenvalues; }
const std::vector<double>& PCA::getVarianceExplained() const { return variance_explained; }
const std::vector<std::vector<double> >& PCA::getTransformedData() const { return transformed_data; }
const std::vector<std::vector<double> >& PCA::getReconstructedData() const { return reconstructed_data; }
size_t PCA::getNumComponents() const { return active_components; }
bool PCA::converged() const { return converged_; }

void PCA::computeMean(const std::vector<std::vector<double> >& data)
{
	size_t num_samples = data.size();
	mean_vec.assign(num_features_, 0.0);
	for (size_t i = 0; i < num_samples; ++i)
	{
		const std::vector<double>& row = data[i];
		for (size_t j = 0; j < num_features_; ++j)
			mean_vec[j] += row[j];
	}
	for (size_t j = 0; j < num_features_; ++j)
		mean_vec[j] /= num_samples;
}

void PCA::computeCovarianceFlat(const std::vector<std::vector<double> >& data,
                                std::vector<double>& cov_flat) const
{
	size_t num_samples = data.size();
	size_t n = num_features_;

	// Pre-center data into a contiguous flat buffer (samples x features)
	std::vector<double> centered(num_samples * n);
	for (size_t k = 0; k < num_samples; ++k)
	{
		const std::vector<double>& row = data[k];
		double* dst = &centered[k * n];
		for (size_t j = 0; j < n; ++j)
			dst[j] = row[j] - mean_vec[j];
	}

	// Accumulate covariance via rank-1 updates: cov += centered_row * centered_row^T
	cov_flat.assign(n * n, 0.0);
	for (size_t k = 0; k < num_samples; ++k)
	{
		const double* row = &centered[k * n];
		for (size_t i = 0; i < n; ++i)
		{
			double ci = row[i];
			double* cov_row = &cov_flat[i * n];
			for (size_t j = i; j < n; ++j)
				cov_row[j] += ci * row[j];
		}
	}

	// Normalize and mirror
	double inv_denom = 1.0 / (num_samples - 1);
	for (size_t i = 0; i < n; ++i)
	{
		cov_flat[i * n + i] *= inv_denom;
		for (size_t j = i + 1; j < n; ++j)
		{
			cov_flat[i * n + j] *= inv_denom;
			cov_flat[j * n + i] = cov_flat[i * n + j];
		}
	}
}

// Householder tridiagonalization of a symmetric matrix.
// A: input symmetric matrix (flat n*n, row-major), destroyed on exit.
// diag: output diagonal (n elements).
// offdiag: output sub-diagonal (n elements, offdiag[0..n-2] used).
// Q: output orthogonal transformation (flat n*n), A = Q * T * Q^T.
void PCA::householderTridiag(double* A, size_t n, double* diag, double* offdiag, double* Q)
{
	// Initialize Q = I
	for (size_t i = 0; i < n * n; ++i)
		Q[i] = 0.0;
	for (size_t i = 0; i < n; ++i)
		Q[i * n + i] = 1.0;

	// Pre-allocate work buffers (max sub-column length is n-1)
	size_t max_m = (n > 1) ? n - 1 : 0;
	std::vector<double> u(max_m);
	std::vector<double> p(max_m);
	std::vector<double> q(max_m);

	for (size_t k = 0; k + 2 < n; ++k)
	{
		size_t m = n - k - 1; // length of the sub-column

		// x = A(k+1:n-1, k)
		// Compute ||x||
		double norm_x = 0.0;
		for (size_t i = 0; i < m; ++i)
		{
			double val = A[(k + 1 + i) * n + k];
			norm_x += val * val;
		}
		norm_x = std::sqrt(norm_x);

		if (norm_x < 1e-15)
		{
			offdiag[k] = 0.0;
			continue;
		}

		// sigma = -sign(x[0]) * ||x||
		double x0 = A[(k + 1) * n + k];
		double sigma = (x0 >= 0.0) ? -norm_x : norm_x;

		// Householder vector u = x; u[0] -= sigma
		for (size_t i = 0; i < m; ++i)
			u[i] = A[(k + 1 + i) * n + k];
		u[0] -= sigma;

		// beta = u^T * u
		double beta = 0.0;
		for (size_t i = 0; i < m; ++i)
			beta += u[i] * u[i];

		if (beta < 1e-30)
		{
			offdiag[k] = sigma;
			continue;
		}

		double two_over_beta = 2.0 / beta;

		// Set the tridiagonal element
		offdiag[k] = sigma;

		// Zero out the sub-column below k+1
		A[(k + 1) * n + k] = sigma;
		A[k * n + (k + 1)] = sigma;
		for (size_t i = 1; i < m; ++i)
		{
			A[(k + 1 + i) * n + k] = 0.0;
			A[k * n + (k + 1 + i)] = 0.0;
		}

		// p = (2/beta) * A(k+1:n-1, k+1:n-1) * u
		for (size_t i = 0; i < m; ++i)
		{
			double sum = 0.0;
			for (size_t j = 0; j < m; ++j)
				sum += A[(k + 1 + i) * n + (k + 1 + j)] * u[j];
			p[i] = two_over_beta * sum;
		}

		// K = u^T * p / 2
		double K = 0.0;
		for (size_t i = 0; i < m; ++i)
			K += u[i] * p[i];
		K *= 0.5;

		// q = p - K * (2/beta) * u
		// Note: K already has the 1/2 factor; we need q = p - (u^T*p / beta) * u
		double K_scaled = K * two_over_beta;
		for (size_t i = 0; i < m; ++i)
			q[i] = p[i] - K_scaled * u[i];

		// Symmetric rank-2 update: A(k+1:n-1, k+1:n-1) -= u*q^T + q*u^T
		for (size_t i = 0; i < m; ++i)
		{
			for (size_t j = i; j < m; ++j)
			{
				double update = u[i] * q[j] + q[i] * u[j];
				A[(k + 1 + i) * n + (k + 1 + j)] -= update;
				A[(k + 1 + j) * n + (k + 1 + i)] = A[(k + 1 + i) * n + (k + 1 + j)];
			}
		}

		// Accumulate Q: Q(:, k+1:n-1) -= (2/beta) * (Q(:, k+1:n-1) * u) * u^T
		for (size_t row = 0; row < n; ++row)
		{
			double dot = 0.0;
			for (size_t j = 0; j < m; ++j)
				dot += Q[row * n + (k + 1 + j)] * u[j];
			dot *= two_over_beta;
			for (size_t j = 0; j < m; ++j)
				Q[row * n + (k + 1 + j)] -= dot * u[j];
		}
	}

	// Extract diagonal and last offdiagonal element
	for (size_t i = 0; i < n; ++i)
		diag[i] = A[i * n + i];
	if (n >= 2)
		offdiag[n - 2] = A[(n - 1) * n + (n - 2)];
}

// QL algorithm with implicit shifts for symmetric tridiagonal eigenvalue problem.
// Based on the EISPACK tql2 routine.
// diag: diagonal elements (n), eigenvalues on exit.
// offdiag: sub-diagonal elements (n-1, offdiag[0..n-2]), destroyed on exit.
// Q: orthogonal matrix (flat n*n), eigenvectors accumulated on exit.
// Returns true if converged.
bool PCA::tridiagonalQL(double* diag, double* offdiag, double* Q, size_t n)
{
	const int MAX_ITER_PER_EIGENVALUE = 30;

	for (size_t l = 0; l < n; ++l)
	{
		int iter = 0;

		for (;;)
		{
			// Find the smallest m >= l such that offdiag[m] is negligible
			size_t m = l;
			while (m + 1 < n)
			{
				double dd = std::fabs(diag[m]) + std::fabs(diag[m + 1]);
				if (dd + std::fabs(offdiag[m]) == dd)
					break;
				++m;
			}

			if (m == l)
				break; // Converged for this eigenvalue

			if (iter++ >= MAX_ITER_PER_EIGENVALUE)
				return false;

			// QL shift: eigenvalue of bottom-right 2x2 closer to diag[l]
			double g = (diag[l + 1] - diag[l]) / (2.0 * offdiag[l]);
			double r = std::sqrt(g * g + 1.0);
			double sign_g = (g >= 0.0) ? r : -r;
			g = diag[m] - diag[l] + offdiag[l] / (g + sign_g);

			double s = 1.0;
			double c = 1.0;
			double p = 0.0;

			// QL rotation chase from m down to l+1
			bool underflow = false;
			for (size_t ii = m; ii > l; --ii)
			{
				size_t i = ii; // current index
				double f = s * offdiag[i - 1];
				double b = c * offdiag[i - 1];

				if (std::fabs(f) >= std::fabs(g))
				{
					c = g / f;
					r = std::sqrt(c * c + 1.0);
					offdiag[i] = f * r;
					s = 1.0 / r;
					c *= s;
				}
				else
				{
					s = f / g;
					r = std::sqrt(s * s + 1.0);
					offdiag[i] = g * r;
					c = 1.0 / r;
					s *= c;
				}

				if (std::fabs(offdiag[i]) < 1e-30)
				{
					// Underflow recovery
					diag[i] -= p;
					offdiag[m] = 0.0;
					underflow = true;
					break;
				}

				g = diag[i] - p;
				r = (diag[i - 1] - g) * s + 2.0 * c * b;
				p = s * r;
				diag[i] = g + p;
				g = c * r - b;

				// Accumulate eigenvectors
				for (size_t k = 0; k < n; ++k)
				{
					double qki = Q[k * n + i];
					double qkm = Q[k * n + (i - 1)];
					Q[k * n + i] = s * qkm + c * qki;
					Q[k * n + (i - 1)] = c * qkm - s * qki;
				}
			}

			if (underflow)
				continue; // Restart iteration for this l

			diag[l] -= p;
			offdiag[l] = g;
			offdiag[m] = 0.0;
		}
	}

	return true;
}

bool PCA::eigendecompose(std::vector<double>& cov_flat)
{
	size_t n = num_features_;

	// Allocate working arrays
	std::vector<double> diag(n);
	std::vector<double> offdiag(n, 0.0);
	std::vector<double> Q(n * n, 0.0);

	// Householder tridiagonalization: cov_flat -> tridiagonal form
	householderTridiag(&cov_flat[0], n, &diag[0], &offdiag[0], &Q[0]);

	// QL iteration on the tridiagonal matrix
	converged_ = tridiagonalQL(&diag[0], &offdiag[0], &Q[0], n);
	iteration_count_ = 0; // QL doesn't expose sweep count in the same way

	if (!converged_)
	{
		printf("[PCA] Warning: QL iteration did not converge.\n");
		return false;
	}

	// Check for and warn about large negative eigenvalues before clamping
	for (size_t i = 0; i < n; ++i)
	{
		if (diag[i] < -1e-6)
		{
			printf("[PCA] Warning: eigenvalue %u is %.6e (large negative). "
			       "Input may be ill-conditioned.\n",
			       (unsigned int)i, diag[i]);
		}
	}

	// Extract eigenvalues (clamp numerical noise to 0) and eigenvectors
	std::vector<double> eig_vals(n);
	// Eigenvectors are columns of Q (Q was accumulated as row-major, columns are eigenvectors)
	std::vector<double> eig_vecs_flat(n * n);
	for (size_t i = 0; i < n; ++i)
	{
		eig_vals[i] = (diag[i] > 0.0) ? diag[i] : 0.0;
		// Row i of eig_vecs = column i of Q (transpose)
		for (size_t j = 0; j < n; ++j)
			eig_vecs_flat[i * n + j] = Q[j * n + i];
	}

	// Sort by eigenvalue descending
	std::vector<std::pair<double, size_t> > eig_pairs(n);
	for (size_t i = 0; i < n; ++i)
		eig_pairs[i] = std::make_pair(eig_vals[i], i);

	std::sort(eig_pairs.begin(), eig_pairs.end(), compare_eig_pairs_desc);

	eigenvalues.resize(n);
	eigenvectors_flat.resize(n * n);
	for (size_t i = 0; i < n; ++i)
	{
		size_t idx = eig_pairs[i].second;
		eigenvalues[i] = eig_vals[idx];
		for (size_t j = 0; j < n; ++j)
			eigenvectors_flat[i * n + j] = eig_vecs_flat[idx * n + j];
	}

	// Normalize eigenvectors (safety measure against numerical drift)
	for (size_t i = 0; i < n; ++i)
	{
		double mag = 0.0;
		for (size_t j = 0; j < n; ++j)
		{
			double v = eigenvectors_flat[i * n + j];
			mag += v * v;
		}
		mag = std::sqrt(mag);

		if (mag > 1e-10)
		{
			for (size_t j = 0; j < n; ++j)
				eigenvectors_flat[i * n + j] /= mag;
		}
	}

	computeComponentMapping();
	computeVarianceExplained();

	return true;
}

void PCA::computeComponentMapping()
{
	size_t n = num_features_;

	// Map each PC to the original feature with the highest absolute loading.
	// This is an APPROXIMATE, BEST-EFFORT heuristic. When multiple PCs share
	// the same highest-loading feature, duplicates are reassigned to arbitrary
	// unused features. Use getFeatureImportance() or the raw eigenvectors for
	// rigorous analysis.
	component_mapping.resize(n);
	for (size_t i = 0; i < n; ++i)
	{
		size_t max_idx = 0;
		double max_val = std::fabs(eigenvectors_flat[i * n]);
		for (size_t j = 1; j < n; ++j)
		{
			double val = std::fabs(eigenvectors_flat[i * n + j]);
			if (val > max_val)
			{
				max_val = val;
				max_idx = j;
			}
		}
		component_mapping[i] = max_idx;
	}

	// Resolve duplicate mappings
	std::vector<bool> mapped(n, false);
	std::vector<size_t> duplicates;
	for (size_t i = 0; i < n; ++i)
	{
		if (mapped[component_mapping[i]])
			duplicates.push_back(i);
		else
			mapped[component_mapping[i]] = true;
	}
	for (size_t i = 0; i < duplicates.size(); ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			if (!mapped[j])
			{
				component_mapping[duplicates[i]] = j;
				mapped[j] = true;
				break;
			}
		}
	}
}

void PCA::computeVarianceExplained()
{
	size_t n = num_features_;
	double total_variance = 0.0;
	for (size_t i = 0; i < n; ++i)
		total_variance += eigenvalues[i];

	variance_explained.resize(n);
	if (total_variance > 1e-10)
	{
		for (size_t i = 0; i < n; ++i)
			variance_explained[i] = eigenvalues[i] / total_variance;
	}
	else
	{
		double equal = 1.0 / n;
		for (size_t i = 0; i < n; ++i)
			variance_explained[i] = equal;
	}
}

void PCA::project(const std::vector<std::vector<double> >& data)
{
	size_t num_samples = data.size();
	transformed_data.resize(num_samples, std::vector<double>(active_components, 0.0));
	for (size_t i = 0; i < num_samples; ++i)
	{
		const std::vector<double>& row = data[i];
		for (size_t c = 0; c < active_components; ++c)
		{
			double dot = 0.0;
			const double* ev = &eigenvectors_flat[c * num_features_];
			for (size_t f = 0; f < num_features_; ++f)
				dot += (row[f] - mean_vec[f]) * ev[f];
			transformed_data[i][c] = dot;
		}
	}
}

void PCA::reconstruct()
{
	size_t num_samples = transformed_data.size();
	reconstructed_data.resize(num_samples, std::vector<double>(num_features_));
	for (size_t i = 0; i < num_samples; ++i)
	{
		std::vector<double>& out = reconstructed_data[i];
		for (size_t j = 0; j < num_features_; ++j)
			out[j] = mean_vec[j];

		for (size_t c = 0; c < active_components; ++c)
		{
			double coeff = transformed_data[i][c];
			const double* ev = &eigenvectors_flat[c * num_features_];
			for (size_t j = 0; j < num_features_; ++j)
				out[j] += coeff * ev[j];
		}
	}
}

bool PCA::fit(const std::vector<std::vector<double> >& data, size_t num_components)
{
	clearState();

	size_t num_samples = data.size();
	if (num_samples <= 1)
		return false;

	num_features_ = data[0].size();
	if (num_features_ == 0)
		return false;

	// Validate all rows have the same number of features
	for (size_t i = 1; i < num_samples; ++i)
	{
		if (data[i].size() != num_features_)
		{
			printf("[PCA] Error: row %u has %u features, expected %u. Aborting.\n",
				(unsigned int)i, (unsigned int)data[i].size(), (unsigned int)num_features_);
			num_features_ = 0;
			return false;
		}
	}

	active_components = (num_components > 0 && num_components <= num_features_)
		? num_components : num_features_;

	computeMean(data);

	std::vector<double> cov_flat;
	computeCovarianceFlat(data, cov_flat);

	return eigendecompose(cov_flat);
}

bool PCA::compute(const std::vector<std::vector<double> >& data, size_t num_components)
{
	if (!fit(data, num_components))
		return false;

	project(data);
	reconstruct();
	return true;
}

std::vector<std::vector<double> > PCA::transform(const std::vector<std::vector<double> >& new_data) const
{
	std::vector<std::vector<double> > result;
	if (mean_vec.empty() || eigenvectors_flat.empty() || active_components == 0)
		return result;

	result.resize(new_data.size(), std::vector<double>(active_components, 0.0));

	for (size_t i = 0; i < new_data.size(); ++i)
	{
		if (new_data[i].size() != num_features_)
			continue;

		const std::vector<double>& row = new_data[i];
		for (size_t c = 0; c < active_components; ++c)
		{
			double dot = 0.0;
			const double* ev = &eigenvectors_flat[c * num_features_];
			for (size_t f = 0; f < num_features_; ++f)
				dot += (row[f] - mean_vec[f]) * ev[f];
			result[i][c] = dot;
		}
	}
	return result;
}

std::vector<std::vector<double> > PCA::inverseTransform(const std::vector<std::vector<double> >& projected_data) const
{
	std::vector<std::vector<double> > result;
	if (mean_vec.empty() || eigenvectors_flat.empty() || active_components == 0)
		return result;

	result.resize(projected_data.size(), std::vector<double>(num_features_));

	for (size_t i = 0; i < projected_data.size(); ++i)
	{
		std::vector<double>& out = result[i];
		for (size_t j = 0; j < num_features_; ++j)
			out[j] = mean_vec[j];

		size_t k = projected_data[i].size();
		if (k > active_components)
			k = active_components;

		for (size_t c = 0; c < k; ++c)
		{
			double coeff = projected_data[i][c];
			const double* ev = &eigenvectors_flat[c * num_features_];
			for (size_t j = 0; j < num_features_; ++j)
				out[j] += coeff * ev[j];
		}
	}
	return result;
}

void PCA::partialFit(const std::vector<std::vector<double> >& batch, size_t num_components)
{
	if (batch.empty())
		return;

	size_t batch_features = batch[0].size();
	if (batch_features == 0)
		return;

	// Count valid rows
	size_t batch_n = 0;
	for (size_t i = 0; i < batch.size(); ++i)
	{
		if (batch[i].size() == batch_features)
			++batch_n;
	}
	if (batch_n == 0)
		return;

	// First call: initialize
	if (incremental_count_ == 0)
	{
		num_features_ = batch_features;
		incremental_mean_.assign(num_features_, 0.0);
		incremental_m2_.assign(num_features_ * num_features_, 0.0);
	}

	if (batch_features != num_features_)
	{
		printf("[PCA] Error: batch has %u features, expected %u. Skipping.\n",
			(unsigned int)batch_features, (unsigned int)num_features_);
		return;
	}

	if (num_components > 0)
		incremental_target_components_ = num_components;

	// Compute batch mean (only valid rows)
	std::vector<double> batch_mean(num_features_, 0.0);
	for (size_t i = 0; i < batch.size(); ++i)
	{
		if (batch[i].size() != num_features_)
			continue;
		for (size_t j = 0; j < num_features_; ++j)
			batch_mean[j] += batch[i][j];
	}
	for (size_t j = 0; j < num_features_; ++j)
		batch_mean[j] /= batch_n;

	// Compute batch M2 (unnormalized covariance)
	size_t n = num_features_;
	std::vector<double> batch_m2(n * n, 0.0);
	for (size_t i = 0; i < batch.size(); ++i)
	{
		if (batch[i].size() != num_features_)
			continue;
		for (size_t j = 0; j < n; ++j)
		{
			double dj = batch[i][j] - batch_mean[j];
			for (size_t k = j; k < n; ++k)
			{
				double dk = batch[i][k] - batch_mean[k];
				batch_m2[j * n + k] += dj * dk;
			}
		}
	}
	// Mirror upper triangle
	for (size_t j = 0; j < n; ++j)
		for (size_t k = j + 1; k < n; ++k)
			batch_m2[k * n + j] = batch_m2[j * n + k];

	if (incremental_count_ == 0)
	{
		// First batch
		incremental_mean_ = batch_mean;
		incremental_m2_ = batch_m2;
		incremental_count_ = batch_n;
	}
	else
	{
		// Merge using Chan et al. parallel algorithm
		size_t n_a = incremental_count_;
		size_t n_b = batch_n;
		size_t n_total = n_a + n_b;

		std::vector<double> delta(num_features_);
		for (size_t j = 0; j < num_features_; ++j)
			delta[j] = batch_mean[j] - incremental_mean_[j];

		// Update M2: M2 = M2_a + M2_b + delta * delta^T * (n_a * n_b / n_total)
		double factor = (double)(n_a) * (double)(n_b) / (double)(n_total);
		for (size_t j = 0; j < n; ++j)
		{
			for (size_t k = j; k < n; ++k)
			{
				incremental_m2_[j * n + k] += batch_m2[j * n + k] + delta[j] * delta[k] * factor;
				incremental_m2_[k * n + j] = incremental_m2_[j * n + k];
			}
		}

		// Update mean
		for (size_t j = 0; j < num_features_; ++j)
			incremental_mean_[j] = (n_a * incremental_mean_[j] + n_b * batch_mean[j]) / n_total;

		incremental_count_ = n_total;
	}
}

bool PCA::finalizeFit()
{
	if (incremental_count_ <= 1)
	{
		printf("[PCA] Error: need at least 2 samples for finalizeFit().\n");
		return false;
	}

	// Clear previous results (but preserve incremental state temporarily)
	size_t saved_count = incremental_count_;
	std::vector<double> saved_mean = incremental_mean_;
	std::vector<double> saved_m2 = incremental_m2_;
	size_t saved_target = incremental_target_components_;

	clearState();

	// Restore
	num_features_ = saved_mean.size();
	mean_vec = saved_mean;

	active_components = (saved_target > 0 && saved_target <= num_features_)
		? saved_target : num_features_;

	// Covariance = M2 / (n - 1)
	size_t n = num_features_;
	std::vector<double> cov_flat(n * n);
	double denom = (double)(saved_count - 1);
	for (size_t i = 0; i < n * n; ++i)
		cov_flat[i] = saved_m2[i] / denom;

	bool ok = eigendecompose(cov_flat);

	// Clear incremental state
	incremental_count_ = 0;
	incremental_mean_.clear();
	incremental_m2_.clear();
	incremental_target_components_ = 0;

	return ok;
}

std::vector<double> PCA::getFeatureImportance() const
{
	std::vector<double> feature_importance(num_features_, 0.0);

	if (num_features_ == 0)
		return feature_importance;

	for (size_t i = 0; i < num_features_; ++i)
	{
		for (size_t j = 0; j < num_features_; ++j)
		{
			double weight = std::fabs(eigenvectors_flat[i * num_features_ + j]);
			feature_importance[j] += weight * weight * variance_explained[i];
		}
	}

	double total_importance = 0.0;
	for (size_t i = 0; i < num_features_; ++i)
		total_importance += feature_importance[i];

	if (total_importance > 1e-10)
	{
		for (size_t i = 0; i < num_features_; ++i)
			feature_importance[i] /= total_importance;
	}
	else
	{
		double equal = 1.0 / num_features_;
		for (size_t i = 0; i < num_features_; ++i)
			feature_importance[i] = equal;
	}

	return feature_importance;
}

size_t PCA::getOriginalFeatureIndex(size_t component_index) const
{
	if (component_index >= component_mapping.size())
		return num_features_; // sentinel: invalid index
	return component_mapping[component_index];
}

void PCA::printComponentMapping() const
{
	printf("\n--------- PCA Component Mapping ---------\n");

	if (component_mapping.empty())
	{
		printf("No component mapping available. Run compute() first.\n");
		return;
	}

	printf("Converged: %s\n", converged_ ? "yes" : "no");
	printf("PC Index | Original Feature | Variance Explained\n");
	printf("------------------------------------------\n");

	for (size_t i = 0; i < component_mapping.size(); ++i)
	{
		printf("  %3u    |       %3u        |     %.2f%%\n",
			static_cast<unsigned int>(i),
			static_cast<unsigned int>(component_mapping[i]),
			variance_explained[i] * 100.0);
	}

	printf("\nFeature Importance (based on variance explained):\n");
	std::vector<double> importance = getFeatureImportance();
	for (size_t i = 0; i < importance.size(); ++i)
	{
		printf("  Feature %3u: %.2f%%\n",
			static_cast<unsigned int>(i),
			importance[i] * 100.0);
	}

	printf("------------------------------------------\n\n");
}
