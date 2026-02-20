#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <limits>

#include "../rng.h"

namespace glades
{

class KMeans
{
private:
    int k;                  // Number of clusters
    int dims_;              // Cached feature dimensionality
    int maxIterations;       // Maximum iterations
    float tolerance;         // Stopping threshold
    std::vector<float> centroids_;   // Flat storage: k * dims_ contiguous floats
    std::vector<int> labels;
    // Optional explicit RNG engine for deterministic behavior.
    // If NULL, initialization falls back to glades::rng::default_engine().
    glades::rng::Engine* rngEngine_;

    float squaredDistToCentroid(const std::vector<float>& point, int centroidIdx) const;
    void assignClusters(const std::vector<std::vector<float> >& points);
    bool updateCentroids(const std::vector<std::vector<float> >& points);
    void initializeCentroids(const std::vector<std::vector<float> >& points);

public:

    KMeans(int clusters, int iterations = 100, float tol = 1e-4);
    void setRngEngine(glades::rng::Engine* e) { rngEngine_ = e; }
    void fit(const std::vector<std::vector<float> >& points);
    int predict(const std::vector<float>& point) const;
    std::vector<std::vector<float> > getCentroids() const;
    const std::vector<int>& getLabels() const;
    unsigned int getClassCount() const;

    static int determineOptimalK(const std::vector<std::vector<float> >& points, int maxK);
};
};

#endif // KMEANS_HPP
