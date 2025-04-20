// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "pca-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/image.h"
#include "../../../Backend/Machine Learning/GMath/pca.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "Backend/Plotter/Plotter.h"

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void createPCAImage(shmea::GString newImageName, const std::vector<std::vector<double> >& compute_data, const glades::PCA& pca)
{
    std::string imagePath = "datasets";
    std::string nameStr = newImageName.c_str(); // Convert GString to std::string

    // Create a plotter with default dimensions and supersampling factor of 4
    shmea::Plotter plotter(1800, 1000, 4);
    
    // Convert 2D data points to Plotter's Point format for original data
    std::vector<shmea::Point> original_points;
    for (size_t i = 0; i < compute_data.size(); ++i) {
        shmea::Point p;
        if (compute_data[i].size() >= 2) {
            p.x = compute_data[i][0];
            p.y = compute_data[i][1];
            original_points.push_back(p);
        }
    }
    
    // Save the original data plot
    plotter.chart()
        .title(nameStr + ": Original Data", 36)
        .grid(true)
        .axes(true)
        .cornerRadius(15)
        .logo("logo.png")
        .axisLabels("Feature 1", "Feature 2", 28)
        .originAxes(true)  // Enable four quadrant origin axes
        .autoMargins(shmea::CHART_SCATTER)
        .addSeries("Original Data", original_points, shmea::RGBA(0xFF, 0x00, 0x00, 0xFF), shmea::SERIES_SCATTER, 2, 8)
        .saveAs(nameStr + "_pca_original.png", imagePath);
    
    printf("[PCA] Original data image saved\n");
    
    // Create a second plotter for transformed data
    shmea::Plotter plotter2(1800, 1000, 4);
    
    // Convert 2D data points to Plotter's Point format for transformed data
    std::vector<shmea::Point> transformed_points;
    for (size_t i = 0; i < pca.transformed_data.size(); ++i) {
        shmea::Point p;
        if (pca.transformed_data[i].size() >= 2) {
            p.x = pca.transformed_data[i][0];
            p.y = pca.transformed_data[i][1];
            transformed_points.push_back(p);
        }
    }

    // Use the pca.sorted_eigen_vecs to create arrows for the diagram
    std::vector<shmea::Arrow> diagramArrows;
    for(unsigned int i = 0; i < pca.sorted_eig_vecs.size(); ++i)
    {
	shmea::Arrow arrow;
	arrow.start.x = 0.0f;
	arrow.start.y = 0.0f;
	arrow.end.x = pca.sorted_eig_vecs[i][0];
	arrow.end.y = pca.sorted_eig_vecs[i][1];
	arrow.color = shmea::RGBA(0x00, 0x00, 0xFF, 0xFF); // Blue color for arrows
	diagramArrows.push_back(arrow);
    }
    
    // Save the transformed data plot
    plotter2.chart()
        .title(nameStr + ": Transformed Data", 36)
        .grid(true)
        .axes(true)
        .cornerRadius(15)
        .logo("logo.png")
        .axisLabels("Principal Component 1", "Principal Component 2", 28)
        .originAxes(true)  // Enable four quadrant origin axes
        .autoMargins(shmea::CHART_SCATTER)
        .addSeries("Transformed Data", transformed_points, shmea::RGBA(0x00, 0xFF, 0x00, 0xFF), shmea::SERIES_SCATTER, 2, 8)
        .addArrows(diagramArrows)
        .saveAs(nameStr + "_pca_transformed.png", imagePath);
    
    printf("[PCA] Transformed data image saved\n");
    
    // Create a third plotter for reconstructed data
    shmea::Plotter plotter3(1800, 1000, 4);
    
    // Convert 2D data points to Plotter's Point format for reconstructed data
    std::vector<shmea::Point> reconstructed_points;
    for (size_t i = 0; i < pca.reconstructed_data.size(); ++i) {
        shmea::Point p;
        if (pca.reconstructed_data[i].size() >= 2) {
            p.x = pca.reconstructed_data[i][0];
            p.y = pca.reconstructed_data[i][1];
            reconstructed_points.push_back(p);
        }
    }
    
    // Save the reconstructed data plot
    plotter3.chart()
        .title(nameStr + ": Reconstructed Data", 36)
        .grid(true)
        .axes(true)
        .cornerRadius(15)
        .logo("logo.png")
        .axisLabels("Feature 1", "Feature 2", 28)
        .originAxes(true)  // Enable four quadrant origin axes
        .autoMargins(shmea::CHART_SCATTER)
        .addSeries("Reconstructed Data", reconstructed_points, shmea::RGBA(0xFF, 0x00, 0xFF, 0xFF), shmea::SERIES_SCATTER, 2, 8)
        .saveAs(nameStr + "_pca_reconstructed.png", imagePath);
    
    printf("[PCA] Reconstructed data image saved\n");
    
    // Combine the features to cluster the classes using the first two principal components
    std::vector<shmea::Point> pca_score_points;
    for(unsigned int i = 0; i < pca.transformed_data.size(); ++i)
    {
        shmea::Point p;
        if (pca.transformed_data[i].size() >= 2) {
            p.x = pca.transformed_data[i][0];
            p.y = pca.transformed_data[i][1];
            pca_score_points.push_back(p);
        }
    }
    
    // Create a fourth plotter for score plot (first two principal components)
    shmea::Plotter plotter4(1800, 1000, 4);
    
    // Save the score plot
    plotter4.chart()
        .title(nameStr + ": PCA Score Plot (PC1 vs PC2)", 36)
        .grid(true)
        .axes(true)
        .cornerRadius(15)
        .logo("logo.png")
        .axisLabels("Principal Component 1", "Principal Component 2", 28)
        .originAxes(true)  // Enable four quadrant origin axes
        .autoMargins(shmea::CHART_SCATTER)
        .addSeries("PC Score", pca_score_points, shmea::RGBA(0xFF, 0x00, 0xFF, 0xFF), shmea::SERIES_SCATTER, 2, 8)
        .saveAs(nameStr + "_pca_score.png", imagePath);
    
    printf("[PCA] Score plot image saved\n");
}

void PCAUnitTest()
{
    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test y=x\n");
    printf("-----------------------------------\n");
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

    glades::PCA pca1;
    pca1.compute(example_data);

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

    createPCAImage("yequalsx", example_data, pca1);

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test x*x\n");
    printf("-----------------------------------\n");

    // Generate example data
    example_data.clear();
    for (int i = -graphSize; i < graphSize; ++i)
    {
	double x = static_cast<double>(i) / graphSize * 10.0;
	double y = x*x;

	std::vector<double> point;
	point.push_back(x);
	point.push_back(y);
	example_data.push_back(point);
    }

    glades::PCA pca2;
    pca2.compute(example_data);

    createPCAImage("xsquared", example_data, pca2);

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test Trig\n");
    printf("-----------------------------------\n");

    // Generate example data
    example_data.clear();
    for (int i = -graphSize; i < graphSize; ++i)
    {
	double x = static_cast<double>(i) / graphSize * 10.0;
	double y = 0.5 * x + 0.5 * std::sin(3.0 * x) + 0.5 * std::cos(2.0 * x) + 0.5 * std::sin(5.0 * x) + 0.5 * std::cos(7.0 * x);

	std::vector<double> point;
	point.push_back(x);
	point.push_back(y);
	example_data.push_back(point);
    }

    glades::PCA pca3;
    pca3.compute(example_data);

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

    createPCAImage("trig", example_data, pca3);

    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("PCA Test Iris\n");
    printf("-----------------------------------\n");

    shmea::GString path = "datasets/iris.data";
    glades::DataInput* di = new glades::NumberInput();
    di->import(path);

    std::vector<std::vector<double> > compute_data;
    unsigned int trainSize = di->getTrainSize();
    for(unsigned int i = 0; i < trainSize; ++i)
    {
	shmea::GVector<float> cRow = di->getTrainRow(i);
	std::vector<double> dataVec;
	for(unsigned int j = 0; j < cRow.size(); ++j)
	{
	    dataVec.push_back(cRow[j]);
	}

	compute_data.push_back(dataVec);
    }

    glades::PCA pca4;
    pca4.compute(compute_data);

    // Save a PNG representation of the PCA
    createPCAImage("iris", compute_data, pca4);

    std::cout << "Variance explained by each principal component: " << std::endl;
    for(unsigned int i = 0; i < pca4.variance_explained.size(); ++i)
    {
	std::cout << "Principal Component " << i << ": " << pca4.variance_explained[i]*100.0f << "%" << std::endl;
    }

    printf("============================================================\n");
    return;
    // This test takes forever
    printf("-----------------------------------\n");
    printf("PCA Test MNIST\n");
    printf("-----------------------------------\n");

    //shmea::GString path = "datasets/images/MNIST/train/0/1000.png";
    path = "MNIST";
    glades::DataInput* di2 = new glades::ImageInput();
    di2->import(path);

    std::vector<std::vector<double> > img_data;
    trainSize = di2->getTrainSize();
    for(unsigned int i = 0; i < trainSize; ++i)
    {
	shmea::GVector<float> flattenedImg = di2->getTrainRow(i);
	std::vector<double> imgVec;
	for(unsigned int j = 0; j < flattenedImg.size(); ++j)
	{
	    imgVec.push_back(flattenedImg[j]);
	}

	img_data.push_back(imgVec);
    }

    glades::PCA pca5;
    pca5.compute(img_data);
}
