#include "kmeans.h"
#include "../rng.h"

using namespace glades;

// Constructor
KMeans::KMeans(int clusters, int iterations, float tol)
{
    k = clusters;
    dims_ = 0;
    maxIterations = iterations;
    tolerance = tol;
    rngEngine_ = NULL;
}

// Squared distance from a point vector to flat centroid at index j
float KMeans::squaredDistToCentroid(const std::vector<float>& point, int j) const
{
    const float* c = &centroids_[j * dims_];
    const float* p = &point[0];
    float sum = 0.0f;
    for (int d = 0; d < dims_; ++d)
    {
        float diff = p[d] - c[d];
        sum += diff * diff;
    }
    return sum;
}

// K-Means++ Initialization
void KMeans::initializeCentroids(const std::vector<std::vector<float> >& points)
{
    int n = static_cast<int>(points.size());
    centroids_.resize(k * dims_);
    glades::rng::Engine& eng = (rngEngine_ ? *rngEngine_ : glades::rng::default_engine());

    // Select first centroid randomly
    int firstIndex = glades::rng::uniform_int(eng, 0, n - 1);
    const float* src = &points[firstIndex][0];
    for (int d = 0; d < dims_; ++d)
        centroids_[d] = src[d];

    // Pre-allocate distances buffer (hoisted out of loop)
    std::vector<float> distances(n);

    // Select remaining centroids using distance-based probability
    for (int c = 1; c < k; ++c)
    {
        // Compute min squared distance to any already-chosen centroid
        for (int i = 0; i < n; ++i)
        {
            const float* p = &points[i][0];
            float minDist = std::numeric_limits<float>::max();
            for (int j = 0; j < c; ++j)
            {
                const float* ctr = &centroids_[j * dims_];
                float dist = 0.0f;
                for (int d = 0; d < dims_; ++d)
                {
                    float diff = p[d] - ctr[d];
                    dist += diff * diff;
                }
                if (dist < minDist)
                    minDist = dist;
            }
            distances[i] = minDist;
        }

        // Weighted random selection (probabilities proportional to squared distance)
        float totalDist = 0.0f;
        for (int i = 0; i < n; ++i)
            totalDist += distances[i];

        float r = glades::rng::uniform_float(eng, 0.0f, totalDist);
        float cumulative = 0.0f;
        int sel = n - 1; // fallback for float rounding

        for (int i = 0; i < n; ++i)
        {
            cumulative += distances[i];
            if (cumulative >= r)
            {
                sel = i;
                break;
            }
        }

        // Copy selected point into centroid c
        const float* selPt = &points[sel][0];
        float* dest = &centroids_[c * dims_];
        for (int d = 0; d < dims_; ++d)
            dest[d] = selPt[d];
    }
}

// Assign each point to the nearest centroid
void KMeans::assignClusters(const std::vector<std::vector<float> >& points)
{
    int n = static_cast<int>(points.size());
    for (int i = 0; i < n; ++i)
    {
        const float* p = &points[i][0];
        float minDist = std::numeric_limits<float>::max();
        int bestCluster = 0;

        for (int j = 0; j < k; ++j)
        {
            const float* c = &centroids_[j * dims_];
            float dist = 0.0f;
            for (int d = 0; d < dims_; ++d)
            {
                float diff = p[d] - c[d];
                dist += diff * diff;
            }
            if (dist < minDist)
            {
                minDist = dist;
                bestCluster = j;
            }
        }
        labels[i] = bestCluster;
    }
}

// Update centroids and return true if centroids moved significantly
bool KMeans::updateCentroids(const std::vector<std::vector<float> >& points)
{
    int flatSize = k * dims_;
    std::vector<float> newCentroids(flatSize, 0.0f);
    std::vector<int> counts(k, 0);

    int n = static_cast<int>(points.size());
    for (int i = 0; i < n; ++i)
    {
        int cluster = labels[i];
        ++counts[cluster];
        const float* p = &points[i][0];
        float* nc = &newCentroids[cluster * dims_];
        for (int d = 0; d < dims_; ++d)
            nc[d] += p[d];
    }

    bool converged = true;
    for (int j = 0; j < k; ++j)
    {
        float* nc = &newCentroids[j * dims_];
        const float* oc = &centroids_[j * dims_];

        if (counts[j] == 0)
        {
            // Retain previous centroid for empty clusters
            for (int d = 0; d < dims_; ++d)
                nc[d] = oc[d];
            continue;
        }

        float invCount = 1.0f / counts[j];
        for (int d = 0; d < dims_; ++d)
        {
            nc[d] *= invCount;

            // Check for convergence using tolerance
            if (std::fabs(oc[d] - nc[d]) > tolerance)
                converged = false;
        }
    }

    centroids_.swap(newCentroids);
    return !converged;
}

// Fit the model to the dataset
void KMeans::fit(const std::vector<std::vector<float> >& points)
{
    centroids_.clear();
    labels.clear();

    if (points.empty() || k <= 0 || static_cast<int>(points.size()) < k)
        return;

    dims_ = static_cast<int>(points[0].size());
    labels.assign(points.size(), -1);

    initializeCentroids(points);

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        assignClusters(points);

        if (!updateCentroids(points))
            break;
    }
}

int KMeans::predict(const std::vector<float>& point) const
{
    if (centroids_.empty())
        return -1;

    float minDist = std::numeric_limits<float>::max();
    int bestCluster = -1;

    for (int j = 0; j < k; ++j)
    {
        float dist = squaredDistToCentroid(point, j);
        if (dist < minDist)
        {
            minDist = dist;
            bestCluster = j;
        }
    }
    return bestCluster;
}

// Reconstruct vector-of-vectors from flat storage (rare, post-fit call)
std::vector<std::vector<float> > KMeans::getCentroids() const
{
    std::vector<std::vector<float> > result;
    result.reserve(k);
    for (int j = 0; j < k; ++j)
    {
        const float* c = &centroids_[j * dims_];
        result.push_back(std::vector<float>(c, c + dims_));
    }
    return result;
}

const std::vector<int>& KMeans::getLabels() const
{
    return labels;
}

unsigned int KMeans::getClassCount() const
{
    return k;
}

// Determine the best k using the elbow method on SSE
int KMeans::determineOptimalK(const std::vector<std::vector<float> >& points, int maxK)
{
    if (points.empty() || maxK < 2)
        return 2;

    std::vector<float> sseValues;
    sseValues.reserve(maxK - 1);

    for (int k = 2; k <= maxK; ++k)
    {
        KMeans kmeans(k, 100, 1e-4);
        kmeans.fit(points);

        // Compute SSE (sum of squared distances to assigned centroid)
        float sse = 0.0f;
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            sse += kmeans.squaredDistToCentroid(points[i], kmeans.labels[i]);
        }

        sseValues.push_back(sse);
    }

    // Find "elbow" using second derivative (maximum curvature)
    int bestK = 2;
    float maxSecondDeriv = 0.0f;

    for (std::size_t i = 1; i + 1 < sseValues.size(); ++i)
    {
        float secondDeriv = sseValues[i - 1] - 2.0f * sseValues[i] + sseValues[i + 1];
        if (secondDeriv > maxSecondDeriv)
        {
            maxSecondDeriv = secondDeriv;
            bestK = static_cast<int>(i) + 2; // Since k starts at 2
        }
    }

    return bestK;
}
