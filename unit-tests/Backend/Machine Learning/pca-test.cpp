// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "pca-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/GMath/pca.h"

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void PCAUnitTest()
{
    // Generate example data
    std::vector<std::vector<double> > example_data;
    int graphSize = 200; // pos and neg
    for (int i = -graphSize; i < graphSize; ++i)
    {
	double x = static_cast<double>(i) / graphSize * 10.0;
	double y = x; // easy visual example for testing

	std::vector<double> point;
	point.push_back(x);
	point.push_back(y);
	example_data.push_back(point);
    }

    std::vector<std::vector<double> > transformed_data;
    std::vector<std::vector<double> > sorted_eig_vecs;
    compute_pca(example_data, transformed_data, sorted_eig_vecs);

    /* Expected Output:
     * ----------
     *  Computing the mean of the data...
     *  ----------
     *  Computing the covariance matrix...
     *  Covariance Matrix Row 0: [33.4167, 33.4167]
     *  Covariance Matrix Row 1: [33.4167, 33.4167]
     *  ----------
     *  Computing the eigenvectors and eigenvalues of the covariance matrix...
     *  Eigenvalues: [-7.64242e-15, -5.56129e-15]
     *  ----------
     *  Sorting eigenvectors based on eigenvalues...
     *  Running Gram-Schmidt orthogonalization on the eigenvectors...
     *  ----------
     *  Eigenvector 0: [-0.707107, 0.707107]
     *  Eigenvector 1: [0.707107, 0.707107]
     *  ----------
     *  Transforming the data using the eigenvectors...
     *  Computing the percentage of variance explained by each principal component...
     *  Variance explained by each principal component: 
     *  Principal Component 0: 57.8809%
     *  Principal Component 1: 42.1191%
     *
     *  Reconstruction error: 4.6284e-28
     */

    printf("============================================================\n");

    // Generate example data
    example_data.clear();
    for (int i = -graphSize; i < graphSize; ++i)
    {
	double x = static_cast<double>(i) / graphSize * 10.0;
	double y = 0.5 * x + 0.5 * std::sin(3.0 * x) + 0.5 * std::cos(2.0 * x) + 0.5 * std::sin(5.0 * x) + 0.5 * std::cos(7.0 * x);
	//double y = x; // easy visual example for testing

	std::vector<double> point;
	point.push_back(x);
	point.push_back(y);
	example_data.push_back(point);
    }

    transformed_data.clear();
    sorted_eig_vecs.clear();
    compute_pca(example_data, transformed_data, sorted_eig_vecs);

    /* Expected Output:
     * ----------
     * Computing the mean of the data...
     * ----------
     *  Computing the covariance matrix...
     *  Covariance Matrix Row 0: [33.4167, 16.568]
     *  Covariance Matrix Row 1: [16.568, 8.73592]
     *  ----------
     *  Computing the eigenvectors and eigenvalues of the covariance matrix...
     *  Eigenvalues: [-0.687157, -25.3679]
     *  ----------
     *  Sorting eigenvectors based on eigenvalues...
     *  Running Gram-Schmidt orthogonalization on the eigenvectors...
     *  ----------
     *  Eigenvector 0: [-0.436971, 0.899476]
     *  Eigenvector 1: [-0.899476, -0.436971]
     *  ----------
     *  Transforming the data using the eigenvectors...
     *  Computing the percentage of variance explained by each principal component...
     *  Variance explained by each principal component: 
     *  Principal Component 0: 2.63732%
     *  Principal Component 1: 97.3627%
     *
     *  Reconstruction error: 54430.9
     */
}
