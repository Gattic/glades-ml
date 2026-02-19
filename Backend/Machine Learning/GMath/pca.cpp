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

double PCA::dot_product(const std::vector<double>& vec1, const std::vector<double>& vec2) const
{
	if (vec1.size() != vec2.size() || vec1.empty())
		return 0.0;

	double result = 0.0;
	for (size_t i = 0; i < vec1.size(); ++i)
		result += vec1[i] * vec2[i];
	return result;
}

// Accessors
const std::vector<double>& PCA::getMean() const { return mean_vec; }
const std::vector<std::vector<double> >& PCA::getEigenvectors() const { return eigenvectors; }
const std::vector<double>& PCA::getEigenvalues() const { return eigenvalues; }
const std::vector<double>& PCA::getVarianceExplained() const { return variance_explained; }
const std::vector<std::vector<double> >& PCA::getTransformedData() const { return transformed_data; }
const std::vector<std::vector<double> >& PCA::getReconstructedData() const { return reconstructed_data; }
size_t PCA::getNumComponents() const { return active_components; }

void PCA::compute(const std::vector<std::vector<double> >& data, size_t num_components)
{
	// Clear previous results
	mean_vec.clear();
	eigenvectors.clear();
	eigenvalues.clear();
	variance_explained.clear();
	transformed_data.clear();
	reconstructed_data.clear();
	component_mapping.clear();
	active_components = 0;

	size_t num_samples = data.size();
	if (num_samples <= 1)
		return;

	size_t num_features = data[0].size();
	if (num_features == 0)
		return;

	active_components = (num_components > 0 && num_components <= num_features)
		? num_components : num_features;

	// Step 1: Compute the mean of each feature
	mean_vec.resize(num_features, 0.0);
	for (size_t j = 0; j < num_features; ++j)
	{
		double sum = 0.0;
		for (size_t i = 0; i < num_samples; ++i)
			sum += data[i][j];
		mean_vec[j] = sum / num_samples;
	}

	// Step 2: Compute covariance matrix (exploit symmetry)
	std::vector<std::vector<double> > cov_mat(num_features, std::vector<double>(num_features, 0.0));
	for (size_t i = 0; i < num_features; ++i)
	{
		for (size_t j = i; j < num_features; ++j)
		{
			double sum = 0.0;
			for (size_t k = 0; k < num_samples; ++k)
				sum += (data[k][i] - mean_vec[i]) * (data[k][j] - mean_vec[j]);
			cov_mat[i][j] = sum / (num_samples - 1);
			cov_mat[j][i] = cov_mat[i][j];
		}
	}

	// Step 3: Jacobi eigendecomposition
	std::vector<std::vector<double> > A = cov_mat;
	std::vector<std::vector<double> > V(num_features, std::vector<double>(num_features, 0.0));
	for (size_t i = 0; i < num_features; ++i)
		V[i][i] = 1.0;

	const double epsilon = 1e-10;
	const int MAX_ITERATIONS = 1000;

	for (int iter = 0; iter < MAX_ITERATIONS; ++iter)
	{
		// Find largest off-diagonal element
		double max_off_diag = 0.0;
		size_t p = 0, q = 1;
		for (size_t i = 0; i < num_features; ++i)
		{
			for (size_t j = i + 1; j < num_features; ++j)
			{
				double val = std::fabs(A[i][j]);
				if (val > max_off_diag)
				{
					max_off_diag = val;
					p = i;
					q = j;
				}
			}
		}

		if (max_off_diag <= epsilon)
			break;

		double app = A[p][p];
		double aqq = A[q][q];
		double apq = A[p][q];

		double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
		double c = std::cos(theta);
		double s = std::sin(theta);

		// Update rows/columns of A for indices != p, q
		for (size_t i = 0; i < num_features; ++i)
		{
			if (i != p && i != q)
			{
				double aip = A[i][p];
				double aiq = A[i][q];
				A[i][p] = aip * c - aiq * s;
				A[p][i] = A[i][p];
				A[i][q] = aiq * c + aip * s;
				A[q][i] = A[i][q];
			}
		}

		A[p][p] = app * c * c - 2.0 * apq * c * s + aqq * s * s;
		A[q][q] = app * s * s + 2.0 * apq * c * s + aqq * c * c;
		A[p][q] = 0.0;
		A[q][p] = 0.0;

		// Accumulate eigenvectors
		for (size_t i = 0; i < num_features; ++i)
		{
			double vip = V[i][p];
			double viq = V[i][q];
			V[i][p] = vip * c - viq * s;
			V[i][q] = viq * c + vip * s;
		}
	}

	// Extract eigenvalues (clamp numerical noise to 0) and eigenvectors
	std::vector<double> eig_vals(num_features);
	std::vector<std::vector<double> > eig_vecs(num_features, std::vector<double>(num_features));
	for (size_t i = 0; i < num_features; ++i)
	{
		eig_vals[i] = std::max(0.0, A[i][i]);
		for (size_t j = 0; j < num_features; ++j)
			eig_vecs[i][j] = V[j][i]; // transpose: rows of eig_vecs = columns of V
	}

	// Step 4: Sort by eigenvalue descending
	std::vector<std::pair<double, size_t> > eig_pairs(num_features);
	for (size_t i = 0; i < num_features; ++i)
		eig_pairs[i] = std::make_pair(eig_vals[i], i);

	std::sort(eig_pairs.begin(), eig_pairs.end(), compare_eig_pairs_desc);

	eigenvalues.resize(num_features);
	eigenvectors.resize(num_features, std::vector<double>(num_features));
	for (size_t i = 0; i < num_features; ++i)
	{
		size_t idx = eig_pairs[i].second;
		eigenvalues[i] = eig_vals[idx];
		eigenvectors[i] = eig_vecs[idx];
	}

	// Normalize eigenvectors (safety measure against numerical drift)
	for (size_t i = 0; i < num_features; ++i)
	{
		double mag = 0.0;
		for (size_t j = 0; j < num_features; ++j)
			mag += eigenvectors[i][j] * eigenvectors[i][j];
		mag = std::sqrt(mag);

		if (mag > 1e-10)
		{
			for (size_t j = 0; j < num_features; ++j)
				eigenvectors[i][j] /= mag;
		}
	}

	// Component mapping: map each PC to the original feature with highest loading
	component_mapping.resize(num_features);
	for (size_t i = 0; i < num_features; ++i)
	{
		size_t max_idx = 0;
		double max_val = std::fabs(eigenvectors[i][0]);
		for (size_t j = 1; j < num_features; ++j)
		{
			double val = std::fabs(eigenvectors[i][j]);
			if (val > max_val)
			{
				max_val = val;
				max_idx = j;
			}
		}
		component_mapping[i] = max_idx;
	}

	// Resolve duplicate mappings
	std::vector<bool> mapped(num_features, false);
	std::vector<size_t> duplicates;
	for (size_t i = 0; i < num_features; ++i)
	{
		if (mapped[component_mapping[i]])
			duplicates.push_back(i);
		else
			mapped[component_mapping[i]] = true;
	}
	for (size_t i = 0; i < duplicates.size(); ++i)
	{
		for (size_t j = 0; j < num_features; ++j)
		{
			if (!mapped[j])
			{
				component_mapping[duplicates[i]] = j;
				mapped[j] = true;
				break;
			}
		}
	}

	// Step 5: Variance explained
	double total_variance = 0.0;
	for (size_t i = 0; i < num_features; ++i)
		total_variance += eigenvalues[i];

	variance_explained.resize(num_features);
	if (total_variance > 1e-10)
	{
		for (size_t i = 0; i < num_features; ++i)
			variance_explained[i] = eigenvalues[i] / total_variance;
	}
	else
	{
		double equal = 1.0 / num_features;
		for (size_t i = 0; i < num_features; ++i)
			variance_explained[i] = equal;
	}

	// Step 6: Transform (project mean-centered data onto top-k eigenvectors)
	transformed_data.resize(num_samples, std::vector<double>(active_components, 0.0));
	std::vector<double> centered(num_features);
	for (size_t i = 0; i < num_samples; ++i)
	{
		for (size_t j = 0; j < num_features; ++j)
			centered[j] = data[i][j] - mean_vec[j];

		for (size_t j = 0; j < active_components; ++j)
			transformed_data[i][j] = dot_product(centered, eigenvectors[j]);
	}

	// Step 7: Reconstruct (using top-k components, add mean back)
	reconstructed_data.resize(num_samples, std::vector<double>(num_features, 0.0));
	for (size_t i = 0; i < num_samples; ++i)
	{
		for (size_t j = 0; j < num_features; ++j)
		{
			double sum = 0.0;
			for (size_t k = 0; k < active_components; ++k)
				sum += transformed_data[i][k] * eigenvectors[k][j];
			reconstructed_data[i][j] = sum + mean_vec[j];
		}
	}
}

std::vector<double> PCA::getFeatureImportance() const
{
	size_t num_features = variance_explained.size();
	std::vector<double> feature_importance(num_features, 0.0);

	if (num_features == 0)
		return feature_importance;

	for (size_t i = 0; i < num_features; ++i)
	{
		for (size_t j = 0; j < num_features; ++j)
		{
			double weight = std::fabs(eigenvectors[i][j]);
			feature_importance[j] += weight * weight * variance_explained[i];
		}
	}

	double total_importance = 0.0;
	for (size_t i = 0; i < num_features; ++i)
		total_importance += feature_importance[i];

	if (total_importance > 1e-10)
	{
		for (size_t i = 0; i < num_features; ++i)
			feature_importance[i] /= total_importance;
	}
	else
	{
		double equal = 1.0 / num_features;
		for (size_t i = 0; i < num_features; ++i)
			feature_importance[i] = equal;
	}

	return feature_importance;
}

size_t PCA::getOriginalFeatureIndex(size_t component_index) const
{
	if (component_index >= component_mapping.size())
		return 0;
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
