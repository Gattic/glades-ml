#include "bayes-optimizer.h"

using namespace glades;

void GaussianProcess::addSample(float x, float y)
{
    X_.push_back(x);
    y_.push_back(y);
}

void GaussianProcess::fit()
{
    int n = X_.size();
    K_.resize(n);
    for (int i = 0; i < n; ++i)
    {
        K_[i].resize(n);
    }

    // Build the covariance matrix
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            K_[i][j] = rbfKernel(X_[i], X_[j], length_scale_, variance_);
        }
        K_[i][i] += noise_;  // Add noise to the diagonal
    }

    // Compute the inverse of the covariance matrix (K_inv = K^-1)
    K_inv_ = invertMatrix(K_);
}

std::pair<float, float> GaussianProcess::predict(float x) const
{
    int n = X_.size();
    std::vector<float> k(n);

    // Compute k vector
    for (int i = 0; i < n; ++i)
    {
        k[i] = rbfKernel(X_[i], x, length_scale_, variance_);
    }

    // Compute mean prediction (mu)
    float mu = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            mu += k[i] * K_inv_[i][j] * y_[j];
        }
    }

    // Compute variance (sigma^2)
    float sigma2 = rbfKernel(x, x, length_scale_, variance_);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            sigma2 -= k[i] * K_inv_[i][j] * k[j];
        }
    }

    return std::make_pair(mu, sigma2);
}

// Matrix inversion using Gaussian elimination (for small matrices)
std::vector<std::vector<float> > GaussianProcess::invertMatrix(const std::vector<std::vector<float> >& matrix) const
{
    int n = matrix.size();
    std::vector<std::vector<float> > inv_matrix(n, std::vector<float>(n, 0.0f));
    std::vector<std::vector<float> > A = matrix;

    // Initialize the identity matrix
    for (int i = 0; i < n; ++i)
    {
        inv_matrix[i][i] = 1.0f;
    }

    // Gaussian elimination
    for (int i = 0; i < n; ++i)
    {
        float diag_element = A[i][i];
        for (int j = 0; j < n; ++j)
        {
            A[i][j] /= diag_element;
            inv_matrix[i][j] /= diag_element;
        }
        for (int k = 0; k < n; ++k)
        {
            if (k != i)
    	{
                float factor = A[k][i];
                for (int j = 0; j < n; ++j)
    	    {
                    A[k][j] -= factor * A[i][j];
                    inv_matrix[k][j] -= factor * inv_matrix[i][j];
                }
            }
        }
    }

    return inv_matrix;
}

void BayesianOptimizer::optimize()
{
    // Initial random sampling
    for (int i = 0; i < 10; ++i)
    {
        float param = randomSample();
        float score = objectiveFunction(param);
        gp_.addSample(param, score);
        if (score > best_score_)
        {
            best_score_ = score;
            best_param_ = param;
        }
    }

    gp_.fit();

    // Iteratively sample using EI acquisition function
    for (size_t i = 0; i < iterations; ++i)
    {
        float param = findNextSample();
        float score = objectiveFunction(param);
        gp_.addSample(param, score);
        gp_.fit();

        if (score > best_score_)
        {
            best_score_ = score;
            best_param_ = param;
        }

        // Print out the current iteration
	printf("[BAYES OPT] Iteration %lu: param = %f, score = %f\n", i + 1, param, score);
    }

    printf("[BAYES OPT] Best parameter found: %f with score: %f\n", best_param_, best_score_);
}

float BayesianOptimizer::randomSample() const
{
    return min_val + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_val - min_val)));
}

float BayesianOptimizer::findNextSample()
{
    float best_ei = -std::numeric_limits<float>::max();
    float best_x = min_val;

    for (float x = min_val; x <= max_val; x += 0.01f)
    {
        float ei = expectedImprovement(x, gp_, best_score_);
        if (ei > best_ei)
        {
            best_ei = ei;
            best_x = x;
        }
    }

    return best_x;
}
