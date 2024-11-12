#include "bayes-optimizer.h"

using namespace glades;

void GaussianProcess::addSample(float x, float y)
{
    X_.push_back(x);
    y_.push_back(y);
}

void GaussianProcess::fit()
{
    unsigned int n = X_.size();
    K_.resize(n);
    for (unsigned int i = 0; i < n; ++i)
    {
        K_[i].resize(n);
    }

    // Build the covariance matrix
    for (unsigned int i = 0; i < n; ++i)
    {
        for (unsigned int j = 0; j < n; ++j)
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
    unsigned int n = X_.size();
    std::vector<float> k(n);

    // Compute k vector
    for (unsigned int i = 0; i < n; ++i)
    {
        k[i] = rbfKernel(X_[i], x, length_scale_, variance_);
    }

    // Compute mean prediction (mu)
    float mu = 0.0f;
    for (unsigned int i = 0; i < n; ++i)
    {
        for (unsigned int j = 0; j < n; ++j)
        {
            mu += k[i] * K_inv_[i][j] * y_[j];
        }
    }

    // Compute variance (sigma^2)
    float sigma2 = rbfKernel(x, x, length_scale_, variance_);
    for (unsigned int i = 0; i < n; ++i)
    {
        for (unsigned int j = 0; j < n; ++j)
        {
            sigma2 -= k[i] * K_inv_[i][j] * k[j];
        }
    }

    return std::make_pair(mu, sigma2);
}

// Matrix inversion using Gaussian elimination (for small matrices)
std::vector<std::vector<float> > GaussianProcess::invertMatrix(const std::vector<std::vector<float> >& matrix) const
{
    unsigned int n = matrix.size();
    std::vector<std::vector<float> > inv_matrix(n, std::vector<float>(n, 0.0f));
    std::vector<std::vector<float> > A = matrix;

    // Initialize the identity matrix
    for (unsigned int i = 0; i < n; ++i)
    {
        inv_matrix[i][i] = 1.0f;
    }

    // Gaussian elimination
    for (unsigned int i = 0; i < n; ++i)
    {
        float diag_element = A[i][i];
        for (unsigned int j = 0; j < n; ++j)
        {
            A[i][j] /= diag_element;
            inv_matrix[i][j] /= diag_element;
        }
        for (unsigned int k = 0; k < n; ++k)
        {
            if (k != i)
    	{
                float factor = A[k][i];
                for (unsigned int j = 0; j < n; ++j)
    	    {
                    A[k][j] -= factor * A[i][j];
                    inv_matrix[k][j] -= factor * inv_matrix[i][j];
                }
            }
        }
    }

    return inv_matrix;
}

void GaussianProcess::printInput() const
{
	printf("X: ");
	for (unsigned int i = 0; i < X_.size(); ++i)
	{
		printf("%f ", X_[i]);
	}
	printf("\n");

	printf("Y: ");
	for (unsigned int i = 0; i < y_.size(); ++i)
	{
		printf("%f ", y_[i]);
	}
	printf("\n");
}

void GaussianProcess::print() const
{
	printf("K: \n");
	for (unsigned int i = 0; i < K_.size(); ++i)
	{
		for (unsigned int j = 0; j < K_[i].size(); ++j)
		{
			printf("%f ", K_[i][j]);
		}
		printf("\n");
	}

	printf("K_inv: \n");
	for (unsigned int i = 0; i < K_inv_.size(); ++i)
	{
		for (unsigned int j = 0; j < K_inv_[i].size(); ++j)
		{
			printf("%f ", K_inv_[i][j]);
		}
		printf("\n");
	}
}

float BayesianOptimizer::optimize(std::vector<std::pair<float, float> > data)
{
	// Step 1: Initialize the Gaussian Process
	for (unsigned int i = 0; i < data.size(); ++i)
	{
		//printf("Adding sample: %f\t %f\n", data[i].first, data[i].second);
		gp_.addSample(data[i].first, data[i].second);
	}
	gp_.fit();

	printf("--------------------\n");

	// Step 2: Optimize the acquisition function
	float best_score = -1.0f;
	float best_param = 0.0f;
	for (float x = 0.001f; x <= 0.3; x += 0.001f)
	{
		float score = expectedImprovement(x, gp_, best_score);
		//printf("x:score: %f\t: %f\n", x, score);
		if (score > best_score)
		{
			//printf("New Best Score: %f\n", score);
			best_score = score;
			best_param = x;
		}
	}

	std::cout << "Best parameter: " << best_param << std::endl;
	std::cout << "Best score: " << best_score << std::endl;

    return best_param;
}

void BayesianOptimizer::update(std::pair<float, float> row)
{
	// Step 3: Evaluate the objective function
	float y = row.second;
	gp_.addSample(row.first, y);
	gp_.fit();

	// Repeat Steps 2 and 3 until convergence
	
	// Step 2: Optimize the acquisition function
	float best_score = -1.0f;
	float best_param = 0.0f;
	for (float x = 0.0f; x <= 10.0f; x += 0.01f)
	{
		float score = expectedImprovement(x, gp_, best_score);
		if (score > best_score)
		{
			best_score = score;
			best_param = x;
		}
	}

	std::cout << "Best parameter: " << best_param << std::endl;
	std::cout << "Best score: " << best_score << std::endl;
}
