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

using namespace glades;

// Helper function to compute the mean of a vector of numbers
double PCA::compute_mean(const std::vector<double>& data)
{
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i)
    {
        sum += data[i];
    }
    return sum / data.size();
}

// Helper function to compute the dot product of two vectors
double PCA::dot_product(const std::vector<double>& vec1, const std::vector<double>& vec2)
{
    double result = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Helper function to perform matrix-vector multiplication
std::vector<double> PCA::matrix_vector_multiply(const std::vector<std::vector<double> >& matrix, const std::vector<double>& vec)
{
    std::vector<double> result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        result[i] = dot_product(matrix[i], vec);
    }
    return result;
}

// Custom comparison function for sorting in descending order
bool PCA::compare_pairs(const std::pair<double, std::vector<double> >& pair1, const std::pair<double, std::vector<double> >& pair2)
{
    return pair1.first > pair2.first;
}

// New comparison function for sorting eigenvalue pairs in descending order by absolute value
struct CompareEigPairs {
    bool operator()(const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) const {
        return std::fabs(a.first) > std::fabs(b.first);
    }
};

// Multiply two matrices: C = A * B
std::vector<std::vector<double> > PCA::matrixMultiply(const std::vector<std::vector<double> >& A,
	const std::vector<std::vector<double> >& B)
{
    size_t rows_A = A.size();
    size_t cols_A = A[0].size();
    size_t cols_B = B[0].size();

    std::vector<std::vector<double> > C(rows_A, std::vector<double>(cols_B, 0.0));

    for (size_t i = 0; i < rows_A; ++i)
    {
        for (size_t j = 0; j < cols_B; ++j)
	{
            for (size_t k = 0; k < cols_A; ++k)
	    {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// Gram-Schmidt orthogonalization
void PCA::gramSchmidt(std::vector<std::vector<double> >& matrix)
{
    size_t num_cols = matrix[0].size();

    for (size_t i = 1; i < num_cols; ++i)
    {
        for (size_t j = 0; j < i; ++j)
	{
            double dot_product = 0.0;
            for (size_t k = 0; k < num_cols; ++k)
	    {
                dot_product += matrix[k][i] * matrix[k][j];
            }

            for (size_t k = 0; k < num_cols; ++k)
	    {
                matrix[k][i] -= dot_product * matrix[k][j];
            }
        }
    }
}

// Main function to compute PCA
void PCA::compute(const std::vector<std::vector<double> >& data)
{
    transformed_data.clear();
    sorted_eig_vecs.clear();
    variance_explained.clear();
    reconstructed_data.clear();
    component_mapping.clear(); // Clear the component mapping

    size_t num_samples = data.size();
    if (num_samples == 0) {
        printf("Error: Empty dataset provided to PCA.\n");
        return;
    }
    
    size_t num_features = data[0].size();
    if (num_features == 0) {
        printf("Error: Dataset contains empty feature vectors.\n");
        return;
    }

    // Step 1: Compute the mean of the data
    printf("----------\n");
    printf("Computing the mean of the data...\n");
    std::vector<double> mean_vec(num_features, 0.0);
    for (size_t i = 0; i < num_features; ++i)
    {
        std::vector<double> feature_data(num_samples, 0.0);
        for (size_t j = 0; j < num_samples; ++j)
        {
            feature_data[j] = data[j][i];
        }
        mean_vec[i] = compute_mean(feature_data);
    }

    // Step 2: Compute the covariance matrix
    printf("----------\n");
    printf("Computing the covariance matrix...\n");
    std::vector<std::vector<double> > cov_mat(num_features, std::vector<double>(num_features, 0.0));
    if (num_samples <= 1) {
        printf("Error: Need at least 2 samples to compute covariance matrix.\n");
        return;
    }
    
    for (size_t i = 0; i < num_features; ++i)
    {
	std::cout << "Covariance Matrix Row " << i << ": [";
        for (size_t j = 0; j < num_features; ++j)
        {
            std::vector<double> data_i(num_samples, 0.0);
            std::vector<double> data_j(num_samples, 0.0);
            for (size_t k = 0; k < num_samples; ++k)
            {
                data_i[k] = data[k][i] - mean_vec[i];
                data_j[k] = data[k][j] - mean_vec[j];
            }
            cov_mat[i][j] = dot_product(data_i, data_j) / (num_samples - 1);

	    // Print the covariance matrix
	    if(j == num_features - 1)
		std::cout << cov_mat[i][j];
	    else
		std::cout << cov_mat[i][j] << ", ";
        }
	std::cout << "]" << std::endl;
    }
    printf("----------\n");

    // Step 3: Compute the eigenvectors and eigenvalues of the covariance matrix
    printf("Computing the eigenvectors and eigenvalues of the covariance matrix...\n");
    std::vector<std::vector<double> > A = cov_mat;
    std::vector<std::vector<double> > V(num_features, std::vector<double>(num_features, 0.0));
    for (size_t i = 0; i < num_features; ++i)
    {
        V[i][i] = 1.0;
    }

    double epsilon = 1e-10;
    double max_off_diag = 1.0;
    
    const int MAX_ITERATIONS = 1000;
    int iteration_count = 0;
    
    while (max_off_diag > epsilon && iteration_count < MAX_ITERATIONS)
    {
        iteration_count++;
        
        if (iteration_count % 100 == 0 || iteration_count == 1) {
            printf("Iteration %d, Max off diag: %.10f\n", iteration_count, max_off_diag);
        }
        
        max_off_diag = 0.0;
        size_t p = 0;
        size_t q = 0;

        for (size_t i = 0; i < num_features; ++i)
        {
            for (size_t j = i + 1; j < num_features; ++j)
            {
                double off_diag = std::fabs(A[i][j]);
                if (off_diag > max_off_diag)
                {
                    max_off_diag = off_diag;
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
        
        for (size_t i = 0; i < num_features; ++i) {
            if (i != p && i != q) {
                double aip = A[i][p];
                double aiq = A[i][q];
                A[i][p] = aip * c - aiq * s;
                A[p][i] = A[i][p];
                A[i][q] = aiq * c + aip * s;
                A[q][i] = A[i][q];
            }
        }
        
        double new_app = app * c * c - 2.0 * apq * c * s + aqq * s * s;
        double new_aqq = app * s * s + 2.0 * apq * c * s + aqq * c * c;
        A[p][p] = new_app;
        A[q][q] = new_aqq;
        
        A[p][q] = 0.0;
        A[q][p] = 0.0;
        
        for (size_t i = 0; i < num_features; ++i) {
            double vip = V[i][p];
            double viq = V[i][q];
            V[i][p] = vip * c - viq * s;
            V[i][q] = viq * c + vip * s;
        }
    }

    if (iteration_count >= MAX_ITERATIONS) {
        printf("Warning: Jacobi algorithm did not converge after %d iterations.\n", MAX_ITERATIONS);
        printf("Final max off-diagonal element: %.10f\n", max_off_diag);
    } else {
        printf("Jacobi algorithm converged after %d iterations.\n", iteration_count);
    }

    std::vector<double> eig_vals(num_features, 0.0);
    std::vector<std::vector<double> > eig_vecs(num_features, std::vector<double>(num_features, 0.0));
    for (size_t i = 0; i < num_features; ++i)
    {
        eig_vals[i] = A[i][i];
        for (size_t j = 0; j < num_features; ++j)
	{
            eig_vecs[i][j] = V[j][i];
        }
    }

    std::cout << "Eigenvalues: [";
    for (size_t i = 0; i < num_features; ++i)
	{
	    if(i == num_features - 1)
		std::cout << eig_vals[i];
	    else
		std::cout << eig_vals[i] << ", ";
	}
	std::cout << "]" << std::endl;

    // Step 4: Sort eigenvectors based on eigenvalues
    printf("----------\n");
    printf("Sorting eigenvectors based on eigenvalues...\n");
    std::vector<std::pair<double, size_t> > eig_pairs;
    for (size_t i = 0; i < num_features; ++i)
    {
        eig_pairs.push_back(std::make_pair(eig_vals[i], i));
    }

    std::sort(eig_pairs.begin(), eig_pairs.end(), CompareEigPairs());

    std::vector<double> sorted_eig_vals(num_features, 0.0);
    sorted_eig_vecs.resize(num_features, std::vector<double>(num_features, 0.0));
    component_mapping.resize(num_features); // Resize the mapping vector
    
    for (size_t i = 0; i < num_features; ++i)
    {
        size_t index = eig_pairs[i].second;
        component_mapping[i] = index; // Store the original index for each sorted component
        sorted_eig_vals[i] = eig_vals[index];
        for (size_t j = 0; j < num_features; ++j)
        {
            sorted_eig_vecs[i][j] = eig_vecs[index][j];
        }
    }

    for (size_t i = 0; i < num_features; ++i)
    {
        double mag = 0.0;
        for (size_t j = 0; j < num_features; ++j)
        {
            mag += sorted_eig_vecs[i][j] * sorted_eig_vecs[i][j];
        }
        mag = std::sqrt(mag);
        
        if (mag > 1e-10)
        {
            for (size_t j = 0; j < num_features; ++j)
            {
                sorted_eig_vecs[i][j] /= mag;
            }
        }
        else
        {
            printf("Warning: Found zero-magnitude eigenvector, skipping normalization.\n");
        }
    }

    printf("Running Gram-Schmidt orthogonalization on the eigenvectors...\n");
    gramSchmidt(sorted_eig_vecs);

    printf("----------\n");
    for (size_t i = 0; i < num_features; ++i)
    {
	std::cout << "Eigenvector " << i << ": [";
	for (size_t j = 0; j < num_features; ++j)
	{
	    if(j == num_features - 1)
		std::cout << sorted_eig_vecs[i][j];
	    else
		std::cout << sorted_eig_vecs[i][j] << ", ";
	}
	std::cout << "]" << std::endl;
    }
    printf("----------\n");

    printf("Transforming the data using the eigenvectors...\n");
    transformed_data.resize(num_samples, std::vector<double>(num_features, 0.0));
    for (size_t i = 0; i < num_samples; ++i)
    {
        for (size_t j = 0; j < num_features; ++j)
        {
            transformed_data[i][j] = dot_product(data[i], sorted_eig_vecs[j]);
        }
    }

    printf("Computing the percentage of variance explained by each principal component...\n");
    double total_variance = 0.0;
    for (size_t i = 0; i < num_features; ++i)
    {
	total_variance += std::fabs(sorted_eig_vals[i]);
    }

    variance_explained = std::vector<double>(num_features, 0.0);
    if (total_variance > 1e-10)
    {
        for (size_t i = 0; i < num_features; ++i)
        {
            variance_explained[i] = std::fabs(sorted_eig_vals[i]) / total_variance;
        }
    }
    else
    {
        printf("Warning: Total variance is zero or near-zero, setting all variance explained to equal values.\n");
        double equal_variance = 1.0 / num_features;
        for (size_t i = 0; i < num_features; ++i)
        {
            variance_explained[i] = equal_variance;
        }
    }

    std::cout << "Variance explained by each principal component: " << std::endl;
    for (size_t i = 0; i < num_features; ++i)
    {
	std::cout << "Principal Component " << i << ": " << variance_explained[i] * 100 << "%" << std::endl;
    }

    std::cout << std::endl;

    printf("Reconstructing the original data from the transformed data...\n");
    reconstructed_data.resize(num_samples, std::vector<double>(num_features, 0.0));
    for (size_t i = 0; i < num_samples; ++i)
    {
	for (size_t j = 0; j < num_features; ++j)
	{
	    for (size_t k = 0; k < num_features; ++k)
	    {
	        reconstructed_data[i][j] += transformed_data[i][k] * sorted_eig_vecs[k][j];
	    }
	}
    }

    double reconstruction_error = 0.0;
    for (size_t i = 0; i < num_samples; ++i)
    {
	for (size_t j = 0; j < num_features; ++j)
	{
	    double diff = data[i][j] - reconstructed_data[i][j];
	    reconstruction_error += diff * diff;
	}
    }

    std::cout << "Reconstruction error: " << reconstruction_error << std::endl;
}

void PCA::calculate_arrow_head(double x1, double y1, double x2, double y2)
{
    double angle = std::atan2(static_cast<double>(y2 - y1), static_cast<double>(x2 - x1));
    double arrowSize = 10.0;
    double arrowX1 = x2 - arrowSize * std::cos(angle + M_PI / 6);
    double arrowY1 = y2 - arrowSize * std::sin(angle + M_PI / 6);
    double arrowX2 = x2 - arrowSize * std::cos(angle - M_PI / 6);
    double arrowY2 = y2 - arrowSize * std::sin(angle - M_PI / 6);
}

void pca_example(const std::vector<std::vector<double> >& data, std::vector<std::vector<double> >& transformed_data, std::vector<std::vector<double> >& sorted_eig_vecs)
{
    std::vector<std::vector<double> > dataset;
    int graphSize = 200;
    for (int i = -graphSize; i < graphSize; ++i)
    {
	double x = static_cast<double>(i) / graphSize * 10.0;
	double y = 0.5 * x + 0.5 * std::sin(3.0 * x) + 0.5 * std::cos(2.0 * x) + 0.5 * std::sin(5.0 * x) + 0.5 * std::cos(7.0 * x);

	std::vector<double> point;
	point.push_back(x);
	point.push_back(y);
	dataset.push_back(point);
    }

    PCA pca;
    pca.compute(dataset);

    for (size_t i = 0; i < sorted_eig_vecs.size(); ++i)
    {
	double arrowX1 = 0;
	double arrowY1 = 0;

	double normX = sorted_eig_vecs[i][0];
	double normY = sorted_eig_vecs[i][1];

	double arrowX2 = arrowX1 + normX;
	double arrowY2 = arrowY1 - normY;

	PCA::calculate_arrow_head(arrowX1, arrowY1, arrowX2, arrowY2);

	PCA::calculate_arrow_head(arrowX1 + 1, arrowY1, arrowX2 + 1, arrowY2);
	PCA::calculate_arrow_head(arrowX1 - 1, arrowY1, arrowX2 - 1, arrowY2);
	PCA::calculate_arrow_head(arrowX1, arrowY1 + 1, arrowX2, arrowY2 + 1);
	PCA::calculate_arrow_head(arrowX1, arrowY1 - 1, arrowX2, arrowY2 - 1);
    }
}
