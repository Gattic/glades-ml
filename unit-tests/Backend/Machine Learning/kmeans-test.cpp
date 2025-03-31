// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "kmeans-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/image.h"
#include "../../../Backend/Machine Learning/GMath/kmeans.h"
#include "../../../Backend/Machine Learning/GMath/pca.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "Backend/Plotter/PNGPlotter.h"

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void createKMeansImage(shmea::GString newImageName, const glades::PCA& pca, const glades::KMeans& kmeans)
{
    std::string imagePath = "datasets";

    shmea::GString imageName = newImageName + "_kmeans_original.png";

    int margin_top = 0;
    int margin_right = 0;
    int margin_bottom = 0;
    int margin_left = 0;

    float min_compute = 0.0f;
    float max_compute = 0.0f;

    // Combine the features to cluster the classes using the first two principal components
    std::vector<std::vector<double> > combined_features;
    for(unsigned int i = 0; i < pca.transformed_data.size(); ++i)
    {
	std::vector<double> point;
	point.push_back(pca.transformed_data[i][0]);
	point.push_back(pca.transformed_data[i][1]);
	combined_features.push_back(point);
    }

    shmea::PNGPlotter plotterPNG(shmea::PNGPlotter::SUPERSAMPLE_WIDTH, shmea::PNGPlotter::SUPERSAMPLE_HEIGHT, combined_features.size(), max_compute, min_compute, 0, margin_top, margin_right, margin_bottom, margin_left, true);

    plotterPNG.addDataPointsKMeans(newImageName.c_str(), combined_features, kmeans.labels, kmeans.getCentroids());

    printf("[KMEANS] Image Saved\n");
    plotterPNG.SavePNG(imageName.c_str(), imagePath);
}

void KMeansUnitTest()
{
    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("KMeans Test\n");
    printf("-----------------------------------\n");

    shmea::GString path = "datasets/iris.data";
    glades::DataInput* di = new glades::NumberInput();
    di->import(path);

    std::vector<std::vector<float> > points;
    std::vector<std::vector<double> > compute_data;
    unsigned int trainSize = di->getTrainSize();
    for(unsigned int i = 0; i < trainSize; ++i)
    {
	shmea::GVector<float> cRow = di->getTrainRow(i);
	std::vector<float> dataVec;
	std::vector<double> dblVec;
	for(unsigned int j = 0; j < cRow.size(); ++j)
	{
	    dataVec.push_back(cRow[j]);
	    dblVec.push_back(cRow[j]);
	}

	points.push_back(dataVec);
	compute_data.push_back(dblVec);
    }

    glades::PCA pca;
    pca.compute(compute_data);

    // Determine the optimal K using silhouette score
    int maxK = 5; // Define the maximum number of clusters to evaluate
    int optimalK = glades::KMeans::determineOptimalK(points, maxK);

    std::cout << "Optimal K determined by silhouette score: " << optimalK << "\n";

    // Run K-Means with the optimal K
    glades::KMeans kmeans(optimalK, 100, 1e-4);
    kmeans.fit(points);

    // Display clustering results
    std::cout << "\nCluster Assignments:\n";
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        std::cout << "Point (" << i << ") => Cluster " << kmeans.labels[i] << "\n";
    }

    createKMeansImage("iris", pca, kmeans);
}
