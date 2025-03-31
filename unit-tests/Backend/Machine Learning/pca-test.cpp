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
#include "Backend/Plotter/PNGPlotter.h"

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void createPCAImage(shmea::GString newImageName, const std::vector<std::vector<double> >& compute_data, const glades::PCA& pca)
{
    std::string imagePath = "datasets";

    shmea::GString imageName = newImageName + "_pca_original.png";
    shmea::GString imageName2 = newImageName + "_pca_transformed.png";
    shmea::GString imageName3 = newImageName + "_pca_reconstructed.png";

    int margin_top = 0;
    int margin_right = 0;
    int margin_bottom = 0;
    int margin_left = 0;

    float min_compute = 0.0f;
    float max_compute = 0.0f;

    // Save the original points
    shmea::PNGPlotter plotterPNG(shmea::PNGPlotter::SUPERSAMPLE_WIDTH, shmea::PNGPlotter::SUPERSAMPLE_HEIGHT, compute_data.size(), max_compute, min_compute, 0, margin_top, margin_right, margin_bottom, margin_left, true);

    shmea::RGBA RED(0xFF, 0x00, 0x00, 0xFF);
    plotterPNG.addDataPointsPCA(compute_data, RED);

    shmea::RGBA BLUE(0x00, 0x00, 0xFF, 0xFF);
    plotterPNG.addArrow(pca.sorted_eig_vecs, pca.variance_explained, BLUE);

    printf("[PCA] Image Saved\n");
    plotterPNG.SavePNG(imageName.c_str(), imagePath);

    shmea::PNGPlotter plotterPNG2(shmea::PNGPlotter::SUPERSAMPLE_WIDTH, shmea::PNGPlotter::SUPERSAMPLE_HEIGHT, compute_data.size(), max_compute, min_compute, 0, margin_top, margin_right, margin_bottom, margin_left, true);

    shmea::RGBA GREEN(0x00, 0xFF, 0x00, 0xFF);
    plotterPNG2.addDataPointsPCA(pca.transformed_data, GREEN);

    plotterPNG2.addArrow(pca.sorted_eig_vecs, pca.variance_explained, BLUE);

    printf("[PCA] Image Saved\n");
    plotterPNG2.SavePNG(imageName2.c_str(), imagePath);

    shmea::PNGPlotter plotterPNG3(shmea::PNGPlotter::SUPERSAMPLE_WIDTH, shmea::PNGPlotter::SUPERSAMPLE_HEIGHT, compute_data.size(), max_compute, min_compute, 0, margin_top, margin_right, margin_bottom, margin_left, true);

    shmea::RGBA PURPLE(0xFF, 0x00, 0xFF, 0xFF);
    plotterPNG3.addDataPointsPCA(pca.transformed_data, PURPLE);

    plotterPNG3.addArrow(pca.sorted_eig_vecs, pca.variance_explained, BLUE);

    printf("[PCA] Image Saved\n");
    plotterPNG3.SavePNG(imageName3.c_str(), imagePath);

    // Combine the features to cluster the classes using the first two principal components
    std::vector<std::vector<double> > combined_features;
    for(unsigned int i = 0; i < pca.transformed_data.size(); ++i)
    {
	std::vector<double> point;
	point.push_back(pca.transformed_data[i][0]);
	point.push_back(pca.transformed_data[i][1]);
	combined_features.push_back(point);
    }

    // Show the classes in the first two principal components
    shmea::PNGPlotter plotterPNG4(shmea::PNGPlotter::SUPERSAMPLE_WIDTH, shmea::PNGPlotter::SUPERSAMPLE_HEIGHT, compute_data.size(), max_compute, min_compute, 0, margin_top, margin_right, margin_bottom, margin_left, true);

    plotterPNG4.addDataPointsPCA(combined_features, PURPLE);

    //plotterPNG4.addArrow(pca.sorted_eig_vecs, pca.variance_explained, BLUE);

    printf("[PCA] Image Saved\n");
    shmea::GString imageName4 = newImageName + "_pca_score.png";
    plotterPNG4.SavePNG(imageName4.c_str(), imagePath);
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
