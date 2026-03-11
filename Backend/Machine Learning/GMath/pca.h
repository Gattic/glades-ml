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
#ifndef GLADES_PCA_H
#define GLADES_PCA_H

#include <algorithm>
#include <cmath>
#include <vector>

namespace glades
{

class PCA
{
private:
	std::vector<double> mean_vec;
	std::vector<double> eigenvectors_flat; // num_features x num_features, row-major
	std::vector<double> eigenvalues; // sorted descending
	std::vector<double> variance_explained;
	std::vector<std::vector<double> > transformed_data;
	std::vector<std::vector<double> > reconstructed_data;
	// Approximate mapping from PC index to original feature index.
	// Best-effort heuristic; see compute() for limitations.
	std::vector<size_t> component_mapping;
	size_t active_components;
	size_t num_features_;
	bool converged_;
	int iteration_count_;

	// Incremental fitting state
	size_t incremental_count_;
	std::vector<double> incremental_mean_;
	std::vector<double> incremental_m2_; // flat n*n, unnormalized covariance
	size_t incremental_target_components_;

	void computeMean(const std::vector<std::vector<double> >& data);
	void computeCovarianceFlat(const std::vector<std::vector<double> >& data,
	                           std::vector<double>& cov_flat) const;
	bool eigendecompose(std::vector<double>& cov_flat);
	void computeComponentMapping();
	void computeVarianceExplained();
	void project(const std::vector<std::vector<double> >& data);
	void reconstruct();
	void clearState();

	// Eigensolver internals
	static void householderTridiag(double* A, size_t n, double* diag, double* offdiag, double* Q);
	static bool tridiagonalQL(double* diag, double* offdiag, double* Q, size_t n);

	// Flat eigenvector access: eigenvectors_flat[component * num_features_ + feature]
	double eigvec(size_t component, size_t feature) const;

public:
	PCA();

	// Fit + transform + reconstruct (backward compatible, returns false on failure)
	bool compute(const std::vector<std::vector<double> >& data, size_t num_components = 0);

	// Fit only: computes mean, eigenvectors, eigenvalues, variance explained
	bool fit(const std::vector<std::vector<double> >& data, size_t num_components = 0);

	// Incremental fitting: accumulate batches, then finalize
	void partialFit(const std::vector<std::vector<double> >& batch, size_t num_components = 0);
	bool finalizeFit();

	const std::vector<double>& getMean() const;
	std::vector<std::vector<double> > getEigenvectors() const;
	const std::vector<double>& getEigenvalues() const;
	const std::vector<double>& getVarianceExplained() const;
	const std::vector<std::vector<double> >& getTransformedData() const;
	const std::vector<std::vector<double> >& getReconstructedData() const;
	size_t getNumComponents() const;
	bool converged() const;

	std::vector<std::vector<double> > transform(const std::vector<std::vector<double> >& new_data) const;
	std::vector<std::vector<double> > inverseTransform(const std::vector<std::vector<double> >& projected_data) const;

	std::vector<double> getFeatureImportance() const;
	size_t getOriginalFeatureIndex(size_t component_index) const;
	void printComponentMapping() const;
};

};

#endif
