#include <vector>
#include <algorithm>
#include <cmath>
#include <cmath>
#include <fstream>
#include <iostream>

// Helper function to compute the mean of a vector of numbers
double compute_mean(const std::vector<double>& data)
{
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i)
    {
        sum += data[i];
    }
    return sum / data.size();
}

// Helper function to compute the dot product of two vectors
double dot_product(const std::vector<double>& vec1, const std::vector<double>& vec2)
{
    double result = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Helper function to perform matrix-vector multiplication
std::vector<double> matrix_vector_multiply(const std::vector<std::vector<double> >& matrix, const std::vector<double>& vec)
{
    std::vector<double> result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        result[i] = dot_product(matrix[i], vec);
    }
    return result;
}

// Custom comparison function for sorting in descending order
bool compare_pairs(const std::pair<double, std::vector<double> >& pair1, const std::pair<double, std::vector<double> >& pair2)
{
    return pair1.first > pair2.first;
}

// Multiply two matrices: C = A * B
std::vector<std::vector<double> > matrixMultiply(const std::vector<std::vector<double> >& A,
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
void gramSchmidt(std::vector<std::vector<double> >& matrix)
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
void compute_pca(const std::vector<std::vector<double> >& data, std::vector<std::vector<double> >& transformed_data, std::vector<std::vector<double> >& sorted_eig_vecs)
{
    size_t num_samples = data.size();
    size_t num_features = data[0].size();

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

    double epsilon = 0.0001;
    double max_off_diag = 1.0;

    while (max_off_diag > epsilon)
    {
        max_off_diag = 0.0;
        size_t p = 0;
        size_t q = 0;

        for (size_t i = 0; i < num_features; ++i)
	{
            for (size_t j = i + 1; j < num_features; ++j)
	    {
                double off_diag = std::abs(A[i][j]);
                if (off_diag > max_off_diag)
		{
                    max_off_diag = off_diag;
                    p = i;
                    q = j;
                }
            }
        }

        double theta = 0.5 * std::atan2(2.0 * A[p][q], A[q][q] - A[p][p]);

        std::vector<std::vector<double> > J(num_features, std::vector<double>(num_features, 0.0));
        for (size_t i = 0; i < num_features; ++i)
	{
            J[i][i] = 1.0;
        }
        J[p][p] = std::cos(theta);
        J[p][q] = -std::sin(theta);
        J[q][p] = std::sin(theta);
        J[q][q] = std::cos(theta);

        // Update A: A = J^T * A * J
        A = matrixMultiply(matrixMultiply(J, A), J);

        // Update V: V = V * J
        V = matrixMultiply(V, J);
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

    std::sort(eig_pairs.begin(), eig_pairs.end(), compare_pairs);

    // Copy sorted eigenvectors into sorted_eig_vecs
    sorted_eig_vecs.resize(num_features, std::vector<double>(num_features, 0.0));
    for (size_t i = 0; i < num_features; ++i)
    {
        size_t index = eig_pairs[i].second;
        for (size_t j = 0; j < num_features; ++j)
        {
            sorted_eig_vecs[i][j] = eig_vecs[index][j];
        }
    }

    // Step 5: Run Gram-Schmidt orthogonalization on the eigenvectors
    // This is done to ensure that the eigenvectors are orthogonal to each other
    // This is necessary because the eigenvectors are not guaranteed to be orthogonal
    // The eigenvectors are orthogonalized in descending order of eigenvalues
    printf("Running Gram-Schmidt orthogonalization on the eigenvectors...\n");
    gramSchmidt(sorted_eig_vecs);

    // Print the sorted_eig_vecs
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

    // Step 6: Reduce the dimensionality of the data
    // Transform the data using the eigenvectors
    // The transformed data is the dot product of the original data and the eigenvectors
    // The transformed data has the same number of samples as the original data
    // The transformed data has the same number of features as the number of eigenvectors
    // The principal components are the directions of maximum variance in the data.
    // The dimensionality of the data is reduced by projecting the data onto the first k principal components
    // The first k principal components are the eigenvectors with the k largest eigenvalues
    printf("Transforming the data using the eigenvectors...\n");
    transformed_data.resize(num_samples, std::vector<double>(num_features, 0.0));
    for (size_t i = 0; i < num_samples; ++i)
    {
        for (size_t j = 0; j < num_features; ++j)
        {
            transformed_data[i][j] = dot_product(data[i], sorted_eig_vecs[j]);
        }
    }

    // Step 7: Compute the percentage of variance explained by each principal component
    printf("Computing the percentage of variance explained by each principal component...\n");
    double total_variance = 0.0;
    for (size_t i = 0; i < num_features; ++i)
    {
	total_variance += eig_vals[i];
    }

    std::vector<double> variance_explained(num_features, 0.0);
    for (size_t i = 0; i < num_features; ++i)
    {
	variance_explained[i] = eig_vals[i] / total_variance;
    }

    std::cout << "Variance explained by each principal component: " << std::endl;
    for (size_t i = 0; i < num_features; ++i)
    {
	std::cout << "Principal Component " << i << ": " << variance_explained[i] * 100 << "%" << std::endl;
    }

    std::cout << std::endl;

    // Step 8: Reconstruct the original data from the transformed data
    // The reconstructed data is the dot product of the transformed data and the eigenvectors
    // The reconstructed data has the same number of samples as the original data
    // The reconstructed data has the same number of features as the original data
    // The reconstructed data is an approximation of the original data
    std::vector<std::vector<double> > reconstructed_data(num_samples, std::vector<double>(num_features, 0.0));
    for (size_t i = 0; i < num_samples; ++i)
    {
	for (size_t j = 0; j < num_features; ++j)
	{
	    reconstructed_data[i][j] = dot_product(transformed_data[i], sorted_eig_vecs[j]);
	}
    }

    // Step 9: Compute the reconstruction error
    // The reconstruction error is the difference between the original data and the reconstructed data
    // The reconstruction error is the sum of the squared differences between the original data and the reconstructed data
    // The reconstruction error is the Frobenius norm of the difference between the original data and the reconstructed data
    // The reconstruction error is the sum of the squared singular values that were discarded
    double reconstruction_error = 0.0;
    for (size_t i = 0; i < num_samples; ++i)
    {
	for (size_t j = 0; j < num_features; ++j)
	{
	    reconstruction_error += std::pow(data[i][j] - reconstructed_data[i][j], 2);
	}
    }

    std::cout << "Reconstruction error: " << reconstruction_error << std::endl;
}

void calculate_arrow_head(double x1, double y1, double x2, double y2)
{
    // Arrow line: x1, y1, x2, y2

    // Calculate arrow head
    double angle = std::atan2(static_cast<double>(y2 - y1), static_cast<double>(x2 - x1));
    double arrowSize = 10.0;
    double arrowX1 = x2 - arrowSize * std::cos(angle + M_PI / 6); // M_PI / 6 = 30 degrees
    double arrowY1 = y2 - arrowSize * std::sin(angle + M_PI / 6);
    double arrowX2 = x2 - arrowSize * std::cos(angle - M_PI / 6);
    double arrowY2 = y2 - arrowSize * std::sin(angle - M_PI / 6);

    // Arrow head
    // x2, y2, arrowX1, arrowY1
    // x2, y2, arrowX2, arrowY2
}

void pca(const std::vector<std::vector<double> >& data, std::vector<std::vector<double> >& transformed_data, std::vector<std::vector<double> >& sorted_eig_vecs)
{
    // Generate example data
    std::vector<std::vector<double> > dataset;
    int graphSize = 200; // pos and neg
    for (int i = -graphSize; i < graphSize; ++i)
    {
	double x = static_cast<double>(i) / graphSize * 10.0;
	double y = 0.5 * x + 0.5 * std::sin(3.0 * x) + 0.5 * std::cos(2.0 * x) + 0.5 * std::sin(5.0 * x) + 0.5 * std::cos(7.0 * x);
	//double y = x; // easy visual example for testing

	std::vector<double> point;
	point.push_back(x);
	point.push_back(y);
	dataset.push_back(point);
    }

    // Compute PCA
    compute_pca(dataset, transformed_data, sorted_eig_vecs);

    for (size_t i = 0; i < sorted_eig_vecs.size(); ++i)
    {
	double arrowX1 = 0;
	double arrowY1 = 0;

	// Normalize the eigenvectors to fit the screen dimensions
	/*double normX = sorted_eig_vecs[i][0] * (screenWidth / 2);
	double normY = sorted_eig_vecs[i][1] * (screenHeight / 2);

	// Scale the normalized values by a factor for visibility
	double scaleFactor = 0.5;  // Adjust this factor as needed
	double arrowX2 = arrowX1 + normX * scaleFactor;
	double arrowY2 = arrowY1 - normY * scaleFactor;  // Flip the Y-coordinate
	*/

	double normX = sorted_eig_vecs[i][0];
	double normY = sorted_eig_vecs[i][1];

	// Scale the normalized values by a factor for visibility
	double arrowX2 = arrowX1 + normX;
	double arrowY2 = arrowY1 - normY;

	calculate_arrow_head(arrowX1, arrowY1, arrowX2, arrowY2);

	// Add thickness to the arrows
	calculate_arrow_head(arrowX1 + 1, arrowY1, arrowX2 + 1, arrowY2);
	calculate_arrow_head(arrowX1 - 1, arrowY1, arrowX2 - 1, arrowY2);
	calculate_arrow_head(arrowX1, arrowY1 + 1, arrowX2, arrowY2 + 1);
	calculate_arrow_head(arrowX1, arrowY1 - 1, arrowX2, arrowY2 - 1);
    }
}
