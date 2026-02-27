#include "bayes-optimizer.h"

using namespace glades;

// ==================== GaussianProcess ====================

// N-dim addSample
void GaussianProcess::addSample(const std::vector<float>& x, float y)
{
	X_.push_back(x);
	y_.push_back(y);
}

// Legacy 1D addSample wrapper
void GaussianProcess::addSample(float x, float y)
{
	std::vector<float> xv(1, x);
	addSample(xv, y);
}

// ARD RBF kernel
float GaussianProcess::rbfKernelND(const std::vector<float>& x1, const std::vector<float>& x2) const
{
	float sum = 0.0f;
	for (unsigned int d = 0; d < ndim_; ++d)
	{
		float diff = x1[d] - x2[d];
		float ls = length_scales_[d];
		sum += (diff * diff) / (ls * ls);
	}
	return variance_ * exp(-0.5f * sum);
}

// Cholesky decomposition: A = L * L^T
bool GaussianProcess::choleskyDecompose(const std::vector<std::vector<float> >& A,
                                        std::vector<std::vector<float> >& L) const
{
	unsigned int n = A.size();
	L.resize(n);
	for (unsigned int i = 0; i < n; ++i)
	{
		L[i].resize(n, 0.0f);
	}

	for (unsigned int i = 0; i < n; ++i)
	{
		for (unsigned int j = 0; j <= i; ++j)
		{
			float sum = 0.0f;
			for (unsigned int k = 0; k < j; ++k)
			{
				sum += L[i][k] * L[j][k];
			}

			if (i == j)
			{
				float diag = A[i][i] - sum;
				if (diag <= 0.0f)
					return false;
				L[i][j] = sqrt(diag);
			}
			else
			{
				L[i][j] = (A[i][j] - sum) / L[j][j];
			}
		}
	}
	return true;
}

// Forward substitution: solve L * x = b
std::vector<float> GaussianProcess::choleskySolveLower(const std::vector<std::vector<float> >& L,
                                                       const std::vector<float>& b) const
{
	unsigned int n = b.size();
	std::vector<float> x(n, 0.0f);
	for (unsigned int i = 0; i < n; ++i)
	{
		float sum = 0.0f;
		for (unsigned int j = 0; j < i; ++j)
		{
			sum += L[i][j] * x[j];
		}
		x[i] = (b[i] - sum) / L[i][i];
	}
	return x;
}

// Back substitution: solve L^T * x = b
std::vector<float> GaussianProcess::choleskySolveUpper(const std::vector<std::vector<float> >& L,
                                                       const std::vector<float>& b) const
{
	unsigned int n = b.size();
	std::vector<float> x(n, 0.0f);
	for (int i = (int)n - 1; i >= 0; --i)
	{
		float sum = 0.0f;
		for (unsigned int j = (unsigned int)i + 1; j < n; ++j)
		{
			sum += L[j][i] * x[j]; // L^T[i][j] = L[j][i]
		}
		x[i] = (b[i] - sum) / L[i][i];
	}
	return x;
}

void GaussianProcess::fit()
{
	unsigned int n = X_.size();
	if (n == 0) return;

	// Build the covariance matrix K
	std::vector<std::vector<float> > K(n);
	for (unsigned int i = 0; i < n; ++i)
	{
		K[i].resize(n);
	}

	for (unsigned int i = 0; i < n; ++i)
	{
		for (unsigned int j = 0; j < n; ++j)
		{
			K[i][j] = rbfKernelND(X_[i], X_[j]);
		}
		K[i][i] += noise_;  // Add noise to the diagonal
	}

	// Cholesky decomposition with jitter fallback
	bool success = choleskyDecompose(K, L_);
	if (!success)
	{
		// Add increasing jitter until decomposition succeeds
		float jitter = 1e-4f;
		for (int attempt = 0; attempt < 10; ++attempt)
		{
			for (unsigned int i = 0; i < n; ++i)
			{
				K[i][i] += jitter;
			}
			success = choleskyDecompose(K, L_);
			if (success) break;
			jitter *= 10.0f;
		}
	}

	// Compute alpha = L^T \ (L \ y)
	std::vector<float> z = choleskySolveLower(L_, y_);
	alpha_ = choleskySolveUpper(L_, z);
}

// N-dim predict
std::pair<float, float> GaussianProcess::predict(const std::vector<float>& x) const
{
	unsigned int n = X_.size();
	if (n == 0)
		return std::make_pair(0.0f, variance_);

	std::vector<float> k(n);

	// Compute k vector (covariance between x and training points)
	for (unsigned int i = 0; i < n; ++i)
	{
		k[i] = rbfKernelND(X_[i], x);
	}

	// Compute mean: mu = k^T * alpha
	float mu = 0.0f;
	for (unsigned int i = 0; i < n; ++i)
	{
		mu += k[i] * alpha_[i];
	}

	// Compute variance: sigma^2 = k(x,x) - k^T * (K^-1) * k
	// Using Cholesky: v = L \ k, then sigma^2 = k(x,x) - v^T * v
	std::vector<float> v = choleskySolveLower(L_, k);
	float sigma2 = rbfKernelND(x, x);
	for (unsigned int i = 0; i < n; ++i)
	{
		sigma2 -= v[i] * v[i];
	}

	// Clamp to avoid negative variance due to numerical issues
	if (sigma2 < 0.0f)
		sigma2 = 0.0f;

	return std::make_pair(mu, sigma2);
}

// Legacy 1D predict wrapper
std::pair<float, float> GaussianProcess::predict(float x) const
{
	std::vector<float> xv(1, x);
	return predict(xv);
}

void GaussianProcess::printInput() const
{
	printf("X (%u samples, %u dims):\n", (unsigned int)X_.size(), ndim_);
	for (unsigned int i = 0; i < X_.size(); ++i)
	{
		printf("  [");
		for (unsigned int d = 0; d < X_[i].size(); ++d)
		{
			if (d > 0) printf(", ");
			printf("%f", X_[i][d]);
		}
		printf("]\n");
	}

	printf("Y: ");
	for (unsigned int i = 0; i < y_.size(); ++i)
	{
		printf("%f ", y_[i]);
	}
	printf("\n");
}

void GaussianProcess::print() const
{
	printf("L (Cholesky factor): \n");
	for (unsigned int i = 0; i < L_.size(); ++i)
	{
		for (unsigned int j = 0; j < L_[i].size(); ++j)
		{
			printf("%f ", L_[i][j]);
		}
		printf("\n");
	}

	printf("Alpha: ");
	for (unsigned int i = 0; i < alpha_.size(); ++i)
	{
		printf("%f ", alpha_[i]);
	}
	printf("\n");
}

// ==================== BayesianOptimizer ====================

// N-dim addObservation
void BayesianOptimizer::addObservation(const std::vector<float>& x, float y)
{
	gp_.addSample(x, y);

	// Track best (minimization)
	if (y < best_score_)
	{
		best_score_ = y;
		best_params_ = x;
	}
}

// Fit the GP model
void BayesianOptimizer::fit()
{
	gp_.fit();
}

// Suggest next point to evaluate using Latin Hypercube Sampling + local hill-climbing
// Works in [0,1]^N space
std::vector<float> BayesianOptimizer::suggestNext()
{
	unsigned int nCandidates = 200;

	// Latin Hypercube Sampling: divide [0,1] into nCandidates bins per dimension
	// For each dimension, create a permuted list of bin indices
	std::vector<std::vector<float> > candidates(nCandidates);
	for (unsigned int i = 0; i < nCandidates; ++i)
	{
		candidates[i].resize(ndim_);
	}

	// Generate LHS samples
	for (unsigned int d = 0; d < ndim_; ++d)
	{
		// Create permutation array [0, 1, ..., nCandidates-1]
		std::vector<unsigned int> perm(nCandidates);
		for (unsigned int i = 0; i < nCandidates; ++i)
		{
			perm[i] = i;
		}
		// Fisher-Yates shuffle
		for (unsigned int i = nCandidates - 1; i > 0; --i)
		{
			unsigned int j = glades::rng::uniform_uint(rng_, 0, i);
			unsigned int tmp = perm[i];
			perm[i] = perm[j];
			perm[j] = tmp;
		}
		// Assign sample values within each bin
		for (unsigned int i = 0; i < nCandidates; ++i)
		{
			float lo = (float)perm[i] / (float)nCandidates;
			float hi = ((float)perm[i] + 1.0f) / (float)nCandidates;
			candidates[i][d] = glades::rng::uniform_float(rng_, lo, hi);
		}
	}

	// Evaluate EI at all candidates, find best
	float bestEI = -1.0f;
	unsigned int bestIdx = 0;
	for (unsigned int i = 0; i < nCandidates; ++i)
	{
		float ei = expectedImprovementND(candidates[i], best_score_);
		if (ei > bestEI)
		{
			bestEI = ei;
			bestIdx = i;
		}
	}

	std::vector<float> best = candidates[bestIdx];

	// Local hill-climbing around the best candidate
	float step = 0.01f;
	for (int iter = 0; iter < 50; ++iter)
	{
		bool improved = false;
		for (unsigned int d = 0; d < ndim_; ++d)
		{
			// Try +step
			std::vector<float> trial = best;
			trial[d] += step;
			if (trial[d] <= 1.0f)
			{
				float ei = expectedImprovementND(trial, best_score_);
				if (ei > bestEI)
				{
					bestEI = ei;
					best = trial;
					improved = true;
				}
			}

			// Try -step
			trial = best;
			trial[d] -= step;
			if (trial[d] >= 0.0f)
			{
				float ei = expectedImprovementND(trial, best_score_);
				if (ei > bestEI)
				{
					bestEI = ei;
					best = trial;
					improved = true;
				}
			}
		}
		if (!improved)
		{
			step *= 0.5f;
			if (step < 1e-6f)
				break;
		}
	}

	return best;
}

// Legacy 1D optimize
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

// Legacy 1D update
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
