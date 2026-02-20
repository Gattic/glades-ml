// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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
#include "kmeans-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/image.h"
#include "../../../Backend/Machine Learning/GMath/kmeans.h"
#include "../../../Backend/Machine Learning/GMath/pca.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "Backend/Plotter/Plotter.h"
#include <sstream>

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void createKMeansImage(shmea::GString newImageName, const glades::PCA& pca, const glades::KMeans& kmeans)
{
    std::string imagePath = "datasets";
    std::string nameStr = newImageName.c_str(); // Convert GString to std::string

    // Create a string for the K value (C++03 compatible)
    std::stringstream ss;
    ss << kmeans.getClassCount(); // Use kmeans.k to access the K value directly
    std::string kValueStr = ss.str();

    // Combine the features to cluster the classes using the first two principal components
    const std::vector<std::vector<double> >& tdata = pca.getTransformedData();
    std::vector<std::vector<double> > clusterData;
    std::vector<int> clusterLabels;
    clusterData.reserve(tdata.size());
    clusterLabels.reserve(tdata.size());
    const std::vector<int>& kLabels = kmeans.getLabels();
    for(unsigned int i = 0; i < tdata.size(); ++i)
    {
        std::vector<double> point;
        point.push_back(tdata[i][0]);
        point.push_back(tdata[i][1]);
        clusterData.push_back(point);
        clusterLabels.push_back(kLabels[i]);
    }

    // Get centroids as vector<vector<double>>
    std::vector<std::vector<double> > centroids;
    std::vector<std::vector<float> > floatCentroids = kmeans.getCentroids();
    for (unsigned int i = 0; i < floatCentroids.size(); ++i) {
        std::vector<double> centroid;
        for (unsigned int j = 0; j < 2 && j < floatCentroids[i].size(); ++j) {
            centroid.push_back(static_cast<double>(floatCentroids[i][j]));
        }
        centroids.push_back(centroid);
    }

    // Create a plotter with default dimensions and supersampling factor of 4
    shmea::Plotter plotter(1800, 1000, 4);
    
    // Plot the K-means clusters
    plotter.chart()
        .title(nameStr + ": K-Means Clustering (K=" + kValueStr + ")", 36)
        .grid(true)
        .axes(true)
        .originAxes(true)  // Enable four quadrant origin axes
        .cornerRadius(15)
        .logo("logo.png")
        .axisLabels("Principal Component 1", "Principal Component 2", 28)
        .autoMargins(shmea::CHART_CLUSTER)
        .addClusterData(clusterData, clusterLabels, centroids)
        .saveAs(nameStr + "_kmeans_original.png", imagePath);

    printf("[KMEANS] Image Saved\n");
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
    points.reserve(trainSize);
    compute_data.reserve(trainSize);
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
    const std::vector<int>& finalLabels = kmeans.getLabels();
    std::cout << "\nCluster Assignments:\n";
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        std::cout << "Point (" << i << ") => Cluster " << finalLabels[i] << "\n";
    }

    createKMeansImage("iris", pca, kmeans);
}
