#include "kmeans.h"
#include "../rng.h"

using namespace glades;

// Constructor
KMeans::KMeans(int clusters, int iterations, float tol)
{
    k = clusters;
    maxIterations = iterations;
    tolerance = tol;
}

// Compute Euclidean distance (optimized loop)
float KMeans::euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) const
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// K-Means++ Initialization
void KMeans::initializeCentroids(const std::vector<std::vector<float> >& points)
{
    centroids.clear();
    int n = points.size();

    // Select first centroid randomly
    int firstIndex = glades::rng::uniform_int(0, n - 1);
    centroids.push_back(points[firstIndex]);

    // Select remaining centroids using distance-based probability
    while ((int)centroids.size() < k)
    {
        std::vector<float> distances(n, std::numeric_limits<float>::max());

        // Compute distances to the nearest centroid
        for (int i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j < centroids.size(); ++j)
            {
                float dist = euclideanDistance(points[i], centroids[j]);
                if (dist < distances[i])
                {
                    distances[i] = dist;
                }
            }
        }

        // Weighted random selection based on squared distances
        float totalDist = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            totalDist += distances[i] * distances[i];
        }

        float r = glades::rng::uniform_float(0.0f, totalDist);
        float cumulative = 0.0f;

        for (int i = 0; i < n; ++i)
        {
            cumulative += distances[i] * distances[i];
            if (cumulative >= r)
            {
                centroids.push_back(points[i]);
                break;
            }
        }
    }
}

// Assign each point to the nearest centroid
void KMeans::assignClusters(const std::vector<std::vector<float> >& points)
{
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        float minDist = std::numeric_limits<float>::max();
        int bestCluster = -1;

        for (std::size_t j = 0; j < centroids.size(); ++j)
        {
            float dist = euclideanDistance(points[i], centroids[j]);
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
    std::vector<std::vector<float> > newCentroids(k, std::vector<float>(points[0].size(), 0.0f));
    std::vector<int> counts(k, 0);

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        int cluster = labels[i];
        ++counts[cluster];

        for (std::size_t d = 0; d < points[i].size(); ++d)
        {
            newCentroids[cluster][d] += points[i][d];
        }
    }

    bool converged = true;
    for (int j = 0; j < k; ++j)
    {
        if (counts[j] == 0) continue; // Avoid division by zero

        for (std::size_t d = 0; d < centroids[j].size(); ++d)
        {
            newCentroids[j][d] /= counts[j];

            // Check for convergence using tolerance
            if (std::fabs(centroids[j][d] - newCentroids[j][d]) > tolerance)
            {
                converged = false;
            }
        }
    }

    centroids = newCentroids;
    return !converged;
}

// Fit the model to the dataset
void KMeans::fit(const std::vector<std::vector<float> >& points)
{
    labels.clear();
    labels.resize(points.size());

    initializeCentroids(points);
    labels.assign(points.size(), -1);

    int iterations = 0;
    while (iterations < maxIterations)
    {
        assignClusters(points);

        if (!updateCentroids(points))
        {
            std::cout << "Converged after " << iterations << " iterations.\n";
            break;
        }

        ++iterations;
    }
}

int KMeans::predict(const std::vector<float>& point) const
{
    float minDist = std::numeric_limits<float>::max();
	int bestCluster = -1;

	for (std::size_t j = 0; j < centroids.size(); ++j)
	{
		float dist = euclideanDistance(point, centroids[j]);
		if (dist < minDist)
		{
			minDist = dist;
			bestCluster = j;
		}
	}
	return bestCluster;
}

std::vector<std::vector<float> > KMeans::getCentroids() const
{
    return centroids;
}

unsigned int KMeans::getClassCount() const
{
    return k;
}

// Determine the best k using silhouette score
int KMeans::determineOptimalK(const std::vector<std::vector<float> >& points, int maxK)
{
    std::vector<float> sseValues;
    
    for (int k = 2; k <= maxK; ++k)
    {
        KMeans kmeans(k, 100, 1e-4);
        kmeans.fit(points);

        // Compute SSE (inertia)
        float sse = 0.0f;
        std::vector<std::vector<float> > centroids = kmeans.getCentroids();

        for (std::size_t i = 0; i < points.size(); ++i)
        {
            sse += kmeans.euclideanDistance(points[i], centroids[kmeans.labels[i]]);
        }

        sseValues.push_back(sse);
    }

    // Find "elbow" where SSE drops sharply
    int bestK = 2;
    float maxDiff = 0.0f;

    for (std::size_t i = 1; i < sseValues.size() - 1; ++i)
    {
        float diff = sseValues[i - 1] - sseValues[i];
        if (diff > maxDiff)
        {
            maxDiff = diff;
            bestK = i + 2; // Since k starts at 2
        }
    }

    return bestK;
}

