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
#ifndef _GPCA
#define _GPCA

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace glades
{

class PCA
{
private:
	std::vector<double> mean_vec;
	std::vector<std::vector<double> > eigenvectors; // rows are eigenvectors, sorted desc
	std::vector<double> eigenvalues; // sorted descending
	std::vector<double> variance_explained;
	std::vector<std::vector<double> > transformed_data;
	std::vector<std::vector<double> > reconstructed_data;
	std::vector<size_t> component_mapping;
	size_t active_components;

	double dot_product(const std::vector<double>& vec1, const std::vector<double>& vec2) const;

public:
	void compute(const std::vector<std::vector<double> >& data, size_t num_components = 0);

	const std::vector<double>& getMean() const;
	const std::vector<std::vector<double> >& getEigenvectors() const;
	const std::vector<double>& getEigenvalues() const;
	const std::vector<double>& getVarianceExplained() const;
	const std::vector<std::vector<double> >& getTransformedData() const;
	const std::vector<std::vector<double> >& getReconstructedData() const;
	size_t getNumComponents() const;

	std::vector<double> getFeatureImportance() const;
	size_t getOriginalFeatureIndex(size_t component_index) const;
	void printComponentMapping() const;
};

};

#endif
