#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <limits>

namespace glades
{

class KMeans
{
private:
    int k;                  // Number of clusters
    int maxIterations;       // Maximum iterations
    float tolerance;         // Stopping threshold
    std::vector<std::vector<float> > centroids;  // Centroid positions

    float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) const;
    void assignClusters(const std::vector<std::vector<float> >& points);
    bool updateCentroids(const std::vector<std::vector<float> >& points);
    void initializeCentroids(const std::vector<std::vector<float> >& points);

public:

    std::vector<int> labels;

    KMeans(int clusters, int iterations = 100, float tol = 1e-4);
    void fit(const std::vector<std::vector<float> >& points);
    int predict(const std::vector<float>& point) const;
    std::vector<std::vector<float> > getCentroids() const;
    unsigned int getClassCount() const;

    static int determineOptimalK(const std::vector<std::vector<float> >& points, int maxK);
};
};

#endif // KMEANS_HPP

