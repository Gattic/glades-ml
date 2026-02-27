# Bayesian Hyperparameter Optimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the existing 1D BayesianOptimizer to a multi-dimensional GP with ARD kernel and Cholesky decomposition, then build a HyperparameterTuner for outer-loop trial search and an adaptive Bayesian LR schedule for inner-loop tuning.

**Architecture:** The existing `GaussianProcess` and `BayesianOptimizer` classes in `bayes-optimizer.h/cpp` are upgraded in-place from 1D to N-D. A new `hyperparameter_tuner.h/cpp` provides `SearchSpace` (parameter encoding/decoding) and `HyperparameterTuner` (trial-based optimization loop). The training config gains a `BAYESIAN` LR schedule type for inner-loop adaptive LR. All pure C++, no new dependencies.

**Tech Stack:** C++ (C++98 compatible with the project), CMake build system, project-local `G_assert` test framework.

**Design doc:** `docs/plans/2026-02-22-bayesian-hyperparameter-optimizer-design.md`

---

### Task 1: Upgrade GaussianProcess to Multi-Dimensional with Cholesky

**Files:**
- Modify: `Backend/Machine Learning/Networks/bayes-optimizer.h`
- Modify: `Backend/Machine Learning/Networks/bayes-optimizer.cpp`
- Modify: `unit-tests/Backend/Machine Learning/bayes-optimizer-test.cpp`
- Modify: `unit-tests/Backend/Machine Learning/bayes-optimizer-test.h`

**Step 1: Write failing tests for multi-dimensional GP**

Add new test functions to `bayes-optimizer-test.cpp`. These test the N-D GP, Cholesky, and ARD kernel. Keep the existing `BayesOptimizerUnitTest()` working.

In `bayes-optimizer-test.h`, add:

```cpp
void BayesOptimizerMultiDimTest();
```

In `bayes-optimizer-test.cpp`, add:

```cpp
void BayesOptimizerMultiDimTest()
{
    printf("============================================================\n");
    printf("BayesOptimizer Multi-Dimensional GP Test\n");
    printf("============================================================\n");

    // Test 1: 2D GP with known function f(x1,x2) = -(x1^2 + x2^2)
    // Minimum at (0,0) = 0, samples around it
    {
        std::vector<float> ls;
        ls.push_back(1.0f);
        ls.push_back(1.0f);
        glades::GaussianProcess gp(ls, 1.0f, 1e-5f);

        std::vector<float> p1; p1.push_back(0.0f); p1.push_back(0.0f);
        gp.addSample(p1, 0.0f);  // f(0,0) = 0

        std::vector<float> p2; p2.push_back(1.0f); p2.push_back(0.0f);
        gp.addSample(p2, -1.0f); // f(1,0) = -1

        std::vector<float> p3; p3.push_back(0.0f); p3.push_back(1.0f);
        gp.addSample(p3, -1.0f); // f(0,1) = -1

        std::vector<float> p4; p4.push_back(1.0f); p4.push_back(1.0f);
        gp.addSample(p4, -2.0f); // f(1,1) = -2

        gp.fit();

        // Predict at training points: mean should be close to observed values
        std::pair<float, float> pred = gp.predict(p1);
        float mu = pred.first;
        G_assert(__FILE__, __LINE__,
            "==============GP-MultiDim::predict(0,0) mean ~0==============",
            fabs(mu - 0.0f) < 0.1f);

        // Predict at an unseen point (0.5, 0.5): should be between 0 and -2
        std::vector<float> mid; mid.push_back(0.5f); mid.push_back(0.5f);
        std::pair<float, float> predMid = gp.predict(mid);
        G_assert(__FILE__, __LINE__,
            "==============GP-MultiDim::predict(0.5,0.5) interpolates==============",
            predMid.first > -2.5f && predMid.first < 0.5f);

        // Variance at unseen point should be positive
        G_assert(__FILE__, __LINE__,
            "==============GP-MultiDim::predict(0.5,0.5) variance > 0==============",
            predMid.second > 0.0f);

        printf("GP Multi-Dim 2D test passed\n");
    }

    // Test 2: Cholesky produces valid decomposition (L * L^T ≈ K)
    {
        std::vector<float> ls;
        ls.push_back(1.0f);
        glades::GaussianProcess gp(ls, 1.0f, 1e-4f);

        std::vector<float> a; a.push_back(0.0f);
        std::vector<float> b; b.push_back(1.0f);
        std::vector<float> c; c.push_back(2.0f);
        gp.addSample(a, 1.0f);
        gp.addSample(b, 0.5f);
        gp.addSample(c, 0.1f);
        gp.fit();

        // If fit() succeeds with Cholesky without crashing, decomposition is valid.
        // Predict and verify consistency.
        std::pair<float, float> predA = gp.predict(a);
        G_assert(__FILE__, __LINE__,
            "==============GP-Cholesky::predict at training point==============",
            fabs(predA.first - 1.0f) < 0.1f);

        printf("GP Cholesky decomposition test passed\n");
    }

    // Test 3: ARD kernel - different length scales per dimension
    {
        // Dimension 0 has tight length scale (0.1), dimension 1 has loose (10.0)
        // GP should be more sensitive to changes in dim 0
        std::vector<float> ls;
        ls.push_back(0.1f);
        ls.push_back(10.0f);
        glades::GaussianProcess gp(ls, 1.0f, 1e-5f);

        std::vector<float> origin; origin.push_back(0.0f); origin.push_back(0.0f);
        gp.addSample(origin, 1.0f);

        std::vector<float> shiftD0; shiftD0.push_back(0.5f); shiftD0.push_back(0.0f);
        gp.addSample(shiftD0, 0.0f);

        std::vector<float> shiftD1; shiftD1.push_back(0.0f); shiftD1.push_back(0.5f);
        gp.addSample(shiftD1, 0.9f);

        gp.fit();

        // Predict at a point shifted in dim 0: should differ more from origin
        std::vector<float> testD0; testD0.push_back(0.25f); testD0.push_back(0.0f);
        std::pair<float, float> pD0 = gp.predict(testD0);

        // Predict at a point shifted in dim 1: should be closer to origin
        std::vector<float> testD1; testD1.push_back(0.0f); testD1.push_back(0.25f);
        std::pair<float, float> pD1 = gp.predict(testD1);

        // Dim 1 prediction should be closer to 1.0 (origin value) than dim 0 prediction
        G_assert(__FILE__, __LINE__,
            "==============GP-ARD::dim1 closer to origin than dim0==============",
            fabs(pD1.first - 1.0f) < fabs(pD0.first - 1.0f));

        printf("GP ARD kernel test passed\n");
    }

    printf("============================================================\n");
}
```

Register in `main.cpp`: add `else if (strcmp(argv[1], "bayes-optimizer-nd") == 0) BayesOptimizerMultiDimTest();` and include the header.

**Step 2: Run tests to verify they fail**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && make -j$(nproc) 2>&1 | tail -20
```

Expected: Compilation errors because `GaussianProcess` constructor doesn't accept `vector<float>` length scales yet.

**Step 3: Upgrade GaussianProcess header to multi-dimensional**

Replace the class in `bayes-optimizer.h`:

```cpp
class GaussianProcess
{
public:
    // Multi-dimensional constructor (ARD kernel with per-dimension length scales)
    GaussianProcess(const std::vector<float>& length_scales, float variance = 1.0f, float noise = 1e-5f)
        : length_scales_(length_scales), variance_(variance), noise_(noise)
    {}

    // Legacy 1D constructor (for backward compatibility and inner-loop adaptive LR)
    GaussianProcess(float length_scale = 1.0f, float variance = 1.0f, float noise = 1e-5f)
        : length_scales_(1, length_scale), variance_(variance), noise_(noise)
    {}

    // Multi-dimensional sample interface
    void addSample(const std::vector<float>& x, float y);
    void fit();
    std::pair<float, float> predict(const std::vector<float>& x) const;

    // Legacy 1D interface (wraps the multi-dim version)
    void addSample(float x, float y);
    std::pair<float, float> predict(float x) const;

    unsigned int numSamples() const { return X_.size(); }
    unsigned int numDimensions() const { return length_scales_.size(); }

    void printInput() const;
    void print() const;

private:
    std::vector<std::vector<float> > X_;       // Observations: each is N-dim
    std::vector<float> y_;                      // Observation values
    std::vector<std::vector<float> > L_;        // Lower Cholesky factor of K
    std::vector<float> alpha_;                  // L^T \ (L \ y) for fast prediction
    std::vector<float> length_scales_;          // Per-dimension length scales (ARD)
    float variance_;
    float noise_;

    // Cholesky decomposition: K = L * L^T
    // Returns false if matrix is not positive definite.
    bool choleskyDecompose(const std::vector<std::vector<float> >& matrix,
                           std::vector<std::vector<float> >& L) const;

    // Solve L * x = b (forward substitution)
    std::vector<float> choleskySolveL(const std::vector<std::vector<float> >& L,
                                       const std::vector<float>& b) const;

    // Solve L^T * x = b (backward substitution)
    std::vector<float> choleskySolveLT(const std::vector<std::vector<float> >& L,
                                        const std::vector<float>& b) const;

    // ARD RBF kernel
    float rbfKernel(const std::vector<float>& x1, const std::vector<float>& x2) const;
};
```

**Step 4: Implement multi-dimensional GP in bayes-optimizer.cpp**

Replace the implementation:

```cpp
// --- ARD RBF Kernel ---
float GaussianProcess::rbfKernel(const std::vector<float>& x1, const std::vector<float>& x2) const
{
    float sq_dist = 0.0f;
    for (unsigned int d = 0; d < x1.size() && d < x2.size(); ++d)
    {
        float diff = (x1[d] - x2[d]) / length_scales_[d];
        sq_dist += diff * diff;
    }
    return variance_ * exp(-0.5f * sq_dist);
}

// --- Multi-dim sample interface ---
void GaussianProcess::addSample(const std::vector<float>& x, float y)
{
    X_.push_back(x);
    y_.push_back(y);
}

// --- Legacy 1D wrappers ---
void GaussianProcess::addSample(float x, float y)
{
    std::vector<float> xv(1, x);
    X_.push_back(xv);
    y_.push_back(y);
}

std::pair<float, float> GaussianProcess::predict(float x) const
{
    std::vector<float> xv(1, x);
    return predict(xv);
}

// --- Cholesky decomposition ---
bool GaussianProcess::choleskyDecompose(const std::vector<std::vector<float> >& matrix,
                                         std::vector<std::vector<float> >& Lout) const
{
    unsigned int n = matrix.size();
    Lout.assign(n, std::vector<float>(n, 0.0f));
    for (unsigned int i = 0; i < n; ++i)
    {
        for (unsigned int j = 0; j <= i; ++j)
        {
            float sum = 0.0f;
            for (unsigned int k = 0; k < j; ++k)
                sum += Lout[i][k] * Lout[j][k];

            if (i == j)
            {
                float diag = matrix[i][i] - sum;
                if (diag <= 0.0f)
                    return false; // not positive definite
                Lout[i][j] = sqrt(diag);
            }
            else
            {
                Lout[i][j] = (matrix[i][j] - sum) / Lout[j][j];
            }
        }
    }
    return true;
}

// --- Forward substitution: solve L * x = b ---
std::vector<float> GaussianProcess::choleskySolveL(const std::vector<std::vector<float> >& Lm,
                                                     const std::vector<float>& b) const
{
    unsigned int n = b.size();
    std::vector<float> x(n, 0.0f);
    for (unsigned int i = 0; i < n; ++i)
    {
        float sum = 0.0f;
        for (unsigned int j = 0; j < i; ++j)
            sum += Lm[i][j] * x[j];
        x[i] = (b[i] - sum) / Lm[i][i];
    }
    return x;
}

// --- Backward substitution: solve L^T * x = b ---
std::vector<float> GaussianProcess::choleskySolveLT(const std::vector<std::vector<float> >& Lm,
                                                      const std::vector<float>& b) const
{
    unsigned int n = b.size();
    std::vector<float> x(n, 0.0f);
    for (int i = (int)n - 1; i >= 0; --i)
    {
        float sum = 0.0f;
        for (unsigned int j = (unsigned int)(i + 1); j < n; ++j)
            sum += Lm[j][i] * x[j]; // L^T[i][j] = L[j][i]
        x[i] = (b[i] - sum) / Lm[i][i];
    }
    return x;
}

// --- Fit: build covariance matrix, decompose, precompute alpha ---
void GaussianProcess::fit()
{
    unsigned int n = X_.size();
    std::vector<std::vector<float> > K(n, std::vector<float>(n, 0.0f));

    for (unsigned int i = 0; i < n; ++i)
    {
        for (unsigned int j = 0; j < n; ++j)
            K[i][j] = rbfKernel(X_[i], X_[j]);
        K[i][i] += noise_; // Add noise to diagonal
    }

    if (!choleskyDecompose(K, L_))
    {
        // Fallback: add more jitter and retry
        for (unsigned int i = 0; i < n; ++i)
            K[i][i] += 1e-3f;
        choleskyDecompose(K, L_);
    }

    // alpha = L^T \ (L \ y)
    std::vector<float> Ly = choleskySolveL(L_, y_);
    alpha_ = choleskySolveLT(L_, Ly);
}

// --- Predict: return (mean, variance) ---
std::pair<float, float> GaussianProcess::predict(const std::vector<float>& x) const
{
    unsigned int n = X_.size();
    std::vector<float> k_star(n);
    for (unsigned int i = 0; i < n; ++i)
        k_star[i] = rbfKernel(X_[i], x);

    // Mean: mu = k_star^T * alpha
    float mu = 0.0f;
    for (unsigned int i = 0; i < n; ++i)
        mu += k_star[i] * alpha_[i];

    // Variance: sigma2 = k(x,x) - k_star^T * K^-1 * k_star
    //                    = k(x,x) - v^T * v  where v = L \ k_star
    float k_xx = rbfKernel(x, x);
    std::vector<float> v = choleskySolveL(L_, k_star);
    float v_dot = 0.0f;
    for (unsigned int i = 0; i < n; ++i)
        v_dot += v[i] * v[i];

    float sigma2 = k_xx - v_dot;
    if (sigma2 < 0.0f) sigma2 = 0.0f; // numerical floor

    return std::make_pair(mu, sigma2);
}
```

Remove the old `invertMatrix` method and the old static `rbfKernel`. Remove the old `K_` and `K_inv_` members. Update `printInput()` and `print()` to work with multi-dim X_ and the new L_ / alpha_ members.

**Step 5: Update BayesianOptimizer for multi-dimensional EI**

In `bayes-optimizer.h`, update the `BayesianOptimizer` class:

```cpp
class BayesianOptimizer
{
private:
    std::vector<float> best_params_;
    float best_score_;
    GaussianProcess gp_;
    unsigned int ndim_;

public:
    // Multi-dimensional constructor
    explicit BayesianOptimizer(unsigned int ndim)
        : best_score_(std::numeric_limits<float>::max()),
          gp_(std::vector<float>(ndim, 1.0f), 1.0f, 1e-5f),
          ndim_(ndim)
    {}

    // Legacy 1D constructor
    BayesianOptimizer()
        : best_score_(std::numeric_limits<float>::max()),
          gp_(1.0f, 1.0f, 1e-5f),
          ndim_(1)
    {}

    // Multi-dimensional interface
    void addObservation(const std::vector<float>& params, float score);
    void fit();
    std::vector<float> suggestNext() const;

    // Legacy 1D interface
    float optimize(const std::vector<std::pair<float, float> >);
    void update(const std::pair<float, float>);

    const std::vector<float>& getBestParams() const { return best_params_; }
    float getBestScore() const { return best_score_; }
    const GaussianProcess& getGP() const { return gp_; }

    // Gaussian PDF/CDF
    static float cdf(float x);
    static float pdf(float x);

    // Expected Improvement (minimization: lower score = better)
    float expectedImprovement(const std::vector<float>& x) const;

    void print() const;
};
```

In `bayes-optimizer.cpp`, implement:

```cpp
void BayesianOptimizer::addObservation(const std::vector<float>& params, float score)
{
    gp_.addSample(params, score);
    if (score < best_score_)
    {
        best_score_ = score;
        best_params_ = params;
    }
}

void BayesianOptimizer::fit()
{
    gp_.fit();
}

float BayesianOptimizer::cdf(float x)
{
    return 0.5f * (1.0f + erf(x / sqrt(2.0f)));
}

float BayesianOptimizer::pdf(float x)
{
    return exp(-0.5f * x * x) / sqrt(2.0f * (float)M_PI);
}

// Expected Improvement for MINIMIZATION
// EI(x) = (best_y - mu) * CDF(z) + sigma * PDF(z)
// where z = (best_y - mu) / sigma
float BayesianOptimizer::expectedImprovement(const std::vector<float>& x) const
{
    std::pair<float, float> pred = gp_.predict(x);
    float mu = pred.first;
    float sigma2 = pred.second;
    float sigma = sqrt(sigma2);
    if (sigma < 1e-8f) return 0.0f;

    float z = (best_score_ - mu) / sigma;
    return (best_score_ - mu) * cdf(z) + sigma * pdf(z);
}

// Suggest next point using Latin Hypercube Sampling + local hill-climbing
std::vector<float> BayesianOptimizer::suggestNext() const
{
    // Phase 1: Latin Hypercube Sampling of 200 candidates in [0,1]^ndim
    const unsigned int nCandidates = 200;
    const unsigned int nTopForHillClimb = 5;
    const unsigned int hillClimbSteps = 20;
    const float hillClimbStepSize = 0.05f;

    std::vector<std::vector<float> > candidates(nCandidates);

    // Simple LHS: for each dimension, create a shuffled grid
    // Use a deterministic but varied pattern (no srand dependency)
    for (unsigned int i = 0; i < nCandidates; ++i)
    {
        candidates[i].resize(ndim_);
        for (unsigned int d = 0; d < ndim_; ++d)
        {
            // Stratified random: bin i of nCandidates, with pseudo-random offset
            unsigned int shuffled = (i * 7 + d * 13 + 37) % nCandidates;
            float lo = (float)shuffled / (float)nCandidates;
            float hi = ((float)shuffled + 1.0f) / (float)nCandidates;
            candidates[i][d] = (lo + hi) * 0.5f;
        }
    }

    // Evaluate EI for all candidates
    float bestEI = -1.0f;
    std::vector<float> bestCandidate;
    for (unsigned int i = 0; i < nCandidates; ++i)
    {
        float ei = expectedImprovement(candidates[i]);
        if (ei > bestEI)
        {
            bestEI = ei;
            bestCandidate = candidates[i];
        }
    }

    // Phase 2: Local hill-climbing from best candidate
    std::vector<float> current = bestCandidate;
    float currentEI = bestEI;
    for (unsigned int step = 0; step < hillClimbSteps; ++step)
    {
        float stepSize = hillClimbStepSize / (1.0f + 0.1f * (float)step);
        for (unsigned int d = 0; d < ndim_; ++d)
        {
            // Try +step and -step in dimension d
            std::vector<float> candidate = current;

            candidate[d] = current[d] + stepSize;
            if (candidate[d] <= 1.0f)
            {
                float ei = expectedImprovement(candidate);
                if (ei > currentEI)
                {
                    currentEI = ei;
                    current = candidate;
                }
            }

            candidate[d] = current[d] - stepSize;
            if (candidate[d] >= 0.0f)
            {
                float ei = expectedImprovement(candidate);
                if (ei > currentEI)
                {
                    currentEI = ei;
                    current = candidate;
                }
            }
        }
    }

    return current;
}
```

Keep the legacy `optimize()` and `update()` methods working by wrapping the 1D API internally.

**Step 6: Run tests to verify they pass**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && make -j$(nproc) && make run bayes-optimizer-nd
```

Expected: All three multi-dim tests pass. Also verify legacy test still passes:

```bash
make run bayes-optimizer
```

**Step 7: Commit**

```bash
git add Backend/Machine\ Learning/Networks/bayes-optimizer.h \
        Backend/Machine\ Learning/Networks/bayes-optimizer.cpp \
        unit-tests/Backend/Machine\ Learning/bayes-optimizer-test.cpp \
        unit-tests/Backend/Machine\ Learning/bayes-optimizer-test.h \
        unit-tests/main.cpp
git commit -m "Upgrade GaussianProcess to multi-dimensional with ARD kernel and Cholesky"
```

---

### Task 2: Implement SearchSpace (Parameter Encoding/Decoding)

**Files:**
- Create: `Backend/Machine Learning/Networks/hyperparameter_tuner.h`
- Create: `Backend/Machine Learning/Networks/hyperparameter_tuner.cpp`
- Modify: `Backend/Machine Learning/Networks/CMakeLists.txt` (add new files)
- Create: `unit-tests/Backend/Machine Learning/hyperparameter-tuner-test.h`
- Create: `unit-tests/Backend/Machine Learning/hyperparameter-tuner-test.cpp`
- Modify: `unit-tests/Backend/Machine Learning/CMakeLists.txt` (add test files)
- Modify: `unit-tests/main.cpp` (register test)

**Step 1: Write failing tests for SearchSpace**

Create `unit-tests/Backend/Machine Learning/hyperparameter-tuner-test.h`:

```cpp
#ifndef _UT_HYPERPARAMETER_TUNER
#define _UT_HYPERPARAMETER_TUNER

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>

void SearchSpaceUnitTest();
void HyperparameterTunerUnitTest();

#endif
```

Create `unit-tests/Backend/Machine Learning/hyperparameter-tuner-test.cpp`:

```cpp
#include "hyperparameter-tuner-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/Networks/hyperparameter_tuner.h"
#include <cmath>

void SearchSpaceUnitTest()
{
    printf("============================================================\n");
    printf("SearchSpace Unit Test\n");
    printf("============================================================\n");

    glades::SearchSpace space;

    // Add a log-scale continuous parameter (like learning rate)
    glades::HyperParameter lr;
    lr.name = "learningRate";
    lr.type = glades::HyperParameter::CONTINUOUS;
    lr.low = 1e-5f;
    lr.high = 0.1f;
    lr.logScale = true;
    space.params.push_back(lr);

    // Add a linear continuous parameter
    glades::HyperParameter gamma;
    gamma.name = "gamma";
    gamma.type = glades::HyperParameter::CONTINUOUS;
    gamma.low = 0.1f;
    gamma.high = 0.999f;
    gamma.logScale = false;
    space.params.push_back(gamma);

    // Add an integer parameter
    glades::HyperParameter warmup;
    warmup.name = "warmupSteps";
    warmup.type = glades::HyperParameter::INTEGER;
    warmup.low = 0.0f;
    warmup.high = 5000.0f;
    warmup.logScale = false;
    space.params.push_back(warmup);

    // Add a categorical parameter
    glades::HyperParameter optType;
    optType.name = "optimizer";
    optType.type = glades::HyperParameter::CATEGORICAL;
    optType.choices.push_back(0.0f); // SGD_MOMENTUM
    optType.choices.push_back(1.0f); // ADAMW
    optType.logScale = false;
    space.params.push_back(optType);

    // Test 1: encode then decode should round-trip
    // Raw values: lr=0.001, gamma=0.5, warmup=1000, optimizer=ADAMW(1.0)
    std::vector<float> raw;
    raw.push_back(0.001f);
    raw.push_back(0.5f);
    raw.push_back(1000.0f);
    raw.push_back(1.0f);

    std::vector<float> encoded = space.encode(raw);

    // Encoded should be in [0,1] for all dims
    for (unsigned int i = 0; i < encoded.size(); ++i)
    {
        G_assert(__FILE__, __LINE__,
            "==============SearchSpace::encode in [0,1]==============",
            encoded[i] >= 0.0f && encoded[i] <= 1.0f);
    }

    std::vector<float> decoded = space.decode(encoded);

    // Log-scale LR: should round-trip within tolerance
    G_assert(__FILE__, __LINE__,
        "==============SearchSpace::round-trip LR==============",
        fabs(decoded[0] - 0.001f) < 0.0001f);

    // Linear gamma: should round-trip within tolerance
    G_assert(__FILE__, __LINE__,
        "==============SearchSpace::round-trip gamma==============",
        fabs(decoded[1] - 0.5f) < 0.01f);

    // Integer warmup: should round to nearest int
    G_assert(__FILE__, __LINE__,
        "==============SearchSpace::round-trip warmup==============",
        fabs(decoded[2] - 1000.0f) < 1.0f);

    // Categorical optimizer: should round to nearest choice
    G_assert(__FILE__, __LINE__,
        "==============SearchSpace::round-trip optimizer==============",
        fabs(decoded[3] - 1.0f) < 0.01f);

    // Test 2: boundaries
    std::vector<float> zeros(4, 0.0f);
    std::vector<float> decodedMin = space.decode(zeros);
    G_assert(__FILE__, __LINE__,
        "==============SearchSpace::decode(0) = low bounds==============",
        decodedMin[0] >= 1e-5f && decodedMin[0] <= 1.1e-5f);

    std::vector<float> ones(4, 1.0f);
    std::vector<float> decodedMax = space.decode(ones);
    G_assert(__FILE__, __LINE__,
        "==============SearchSpace::decode(1) = high bounds==============",
        decodedMax[0] >= 0.09f && decodedMax[0] <= 0.101f);

    printf("SearchSpace tests passed\n");
    printf("============================================================\n");
}
```

Add to `unit-tests/Backend/Machine Learning/CMakeLists.txt`:

```
hyperparameter-tuner-test.cpp
```

Add to `unit-tests/main.cpp`:

```cpp
#include "Backend/Machine Learning/hyperparameter-tuner-test.h"
// In the else-if chain:
else if (strcmp(argv[1], "search-space") == 0)
    SearchSpaceUnitTest();
else if (strcmp(argv[1], "hp-tuner") == 0)
    HyperparameterTunerUnitTest();
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && cmake .. && make -j$(nproc) 2>&1 | tail -20
```

Expected: Compilation error, `hyperparameter_tuner.h` not found.

**Step 3: Implement SearchSpace**

Create `Backend/Machine Learning/Networks/hyperparameter_tuner.h`:

```cpp
#ifndef _GHYPERPARAMETER_TUNER
#define _GHYPERPARAMETER_TUNER

#include "bayes-optimizer.h"
#include "training_config.h"
#include <string>
#include <vector>
#include <cmath>

namespace glades {

class NNetwork;
class DataInput;

struct HyperParameter
{
    enum Type { CONTINUOUS = 0, INTEGER = 1, CATEGORICAL = 2 };

    std::string name;
    Type type;
    float low;
    float high;
    std::vector<float> choices; // for CATEGORICAL
    bool logScale;

    HyperParameter()
        : type(CONTINUOUS), low(0.0f), high(1.0f), logScale(false)
    {}
};

struct SearchSpace
{
    std::vector<HyperParameter> params;

    unsigned int dimensions() const { return params.size(); }

    // Encode raw parameter values to normalized [0,1]^N
    std::vector<float> encode(const std::vector<float>& raw) const;

    // Decode normalized [0,1]^N back to raw parameter values
    std::vector<float> decode(const std::vector<float>& normalized) const;

    // Build default search space for TrainingConfig
    static SearchSpace defaultTrainingSearchSpace();

    // Apply decoded raw values to a TrainingConfig
    TrainingConfig applyToConfig(const std::vector<float>& raw,
                                  const TrainingConfig& base) const;
};

} // namespace glades

#endif
```

Create `Backend/Machine Learning/Networks/hyperparameter_tuner.cpp`:

```cpp
#include "hyperparameter_tuner.h"
#include <cmath>
#include <algorithm>

using namespace glades;

std::vector<float> SearchSpace::encode(const std::vector<float>& raw) const
{
    std::vector<float> normalized(params.size(), 0.0f);
    for (unsigned int i = 0; i < params.size() && i < raw.size(); ++i)
    {
        const HyperParameter& p = params[i];
        switch (p.type)
        {
        case HyperParameter::CONTINUOUS:
        case HyperParameter::INTEGER:
        {
            if (p.logScale)
            {
                float logLow = log(p.low);
                float logHigh = log(p.high);
                float logVal = log(raw[i]);
                normalized[i] = (logVal - logLow) / (logHigh - logLow);
            }
            else
            {
                normalized[i] = (raw[i] - p.low) / (p.high - p.low);
            }
            // Clamp to [0,1]
            if (normalized[i] < 0.0f) normalized[i] = 0.0f;
            if (normalized[i] > 1.0f) normalized[i] = 1.0f;
            break;
        }
        case HyperParameter::CATEGORICAL:
        {
            // Find index of the closest choice
            float bestDist = 1e30f;
            unsigned int bestIdx = 0;
            for (unsigned int c = 0; c < p.choices.size(); ++c)
            {
                float dist = fabs(raw[i] - p.choices[c]);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = c;
                }
            }
            if (p.choices.size() <= 1)
                normalized[i] = 0.0f;
            else
                normalized[i] = (float)bestIdx / (float)(p.choices.size() - 1);
            break;
        }
        }
    }
    return normalized;
}

std::vector<float> SearchSpace::decode(const std::vector<float>& normalized) const
{
    std::vector<float> raw(params.size(), 0.0f);
    for (unsigned int i = 0; i < params.size() && i < normalized.size(); ++i)
    {
        const HyperParameter& p = params[i];
        float val = normalized[i];
        // Clamp to [0,1]
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;

        switch (p.type)
        {
        case HyperParameter::CONTINUOUS:
        {
            if (p.logScale)
            {
                float logLow = log(p.low);
                float logHigh = log(p.high);
                raw[i] = exp(logLow + val * (logHigh - logLow));
            }
            else
            {
                raw[i] = p.low + val * (p.high - p.low);
            }
            break;
        }
        case HyperParameter::INTEGER:
        {
            float continuous = p.low + val * (p.high - p.low);
            raw[i] = floor(continuous + 0.5f); // round to nearest int
            break;
        }
        case HyperParameter::CATEGORICAL:
        {
            if (p.choices.empty())
            {
                raw[i] = 0.0f;
            }
            else
            {
                unsigned int idx = (unsigned int)(val * (float)(p.choices.size() - 1) + 0.5f);
                if (idx >= p.choices.size()) idx = p.choices.size() - 1;
                raw[i] = p.choices[idx];
            }
            break;
        }
        }
    }
    return raw;
}

SearchSpace SearchSpace::defaultTrainingSearchSpace()
{
    SearchSpace space;

    // Learning rate (log scale)
    HyperParameter lr;
    lr.name = "learningRate";
    lr.type = HyperParameter::CONTINUOUS;
    lr.low = 1e-5f;
    lr.high = 0.1f;
    lr.logScale = true;
    space.params.push_back(lr);

    // Optimizer type (categorical)
    HyperParameter opt;
    opt.name = "optimizer";
    opt.type = HyperParameter::CATEGORICAL;
    opt.choices.push_back(0.0f); // SGD_MOMENTUM
    opt.choices.push_back(1.0f); // ADAMW
    space.params.push_back(opt);

    // LR schedule type (categorical)
    HyperParameter sched;
    sched.name = "lrScheduleType";
    sched.type = HyperParameter::CATEGORICAL;
    sched.choices.push_back(0.0f); // NONE
    sched.choices.push_back(1.0f); // STEP
    sched.choices.push_back(2.0f); // EXP
    sched.choices.push_back(3.0f); // COSINE
    space.params.push_back(sched);

    // LR schedule gamma
    HyperParameter gamma;
    gamma.name = "lrScheduleGamma";
    gamma.type = HyperParameter::CONTINUOUS;
    gamma.low = 0.1f;
    gamma.high = 0.999f;
    gamma.logScale = false;
    space.params.push_back(gamma);

    // Adam beta1
    HyperParameter b1;
    b1.name = "adamBeta1";
    b1.type = HyperParameter::CONTINUOUS;
    b1.low = 0.8f;
    b1.high = 0.99f;
    b1.logScale = false;
    space.params.push_back(b1);

    // Adam beta2
    HyperParameter b2;
    b2.name = "adamBeta2";
    b2.type = HyperParameter::CONTINUOUS;
    b2.low = 0.99f;
    b2.high = 0.9999f;
    b2.logScale = false;
    space.params.push_back(b2);

    // Global grad clip norm
    HyperParameter clip;
    clip.name = "globalGradClipNorm";
    clip.type = HyperParameter::CONTINUOUS;
    clip.low = 0.0f;
    clip.high = 10.0f;
    clip.logScale = false;
    space.params.push_back(clip);

    // Warmup steps
    HyperParameter wu;
    wu.name = "warmupSteps";
    wu.type = HyperParameter::INTEGER;
    wu.low = 0.0f;
    wu.high = 5000.0f;
    wu.logScale = false;
    space.params.push_back(wu);

    // Weight decay (log scale)
    HyperParameter wd;
    wd.name = "weightDecay";
    wd.type = HyperParameter::CONTINUOUS;
    wd.low = 1e-6f;
    wd.high = 0.1f;
    wd.logScale = true;
    space.params.push_back(wd);

    return space;
}

TrainingConfig SearchSpace::applyToConfig(const std::vector<float>& raw,
                                           const TrainingConfig& base) const
{
    TrainingConfig cfg = base;

    for (unsigned int i = 0; i < params.size() && i < raw.size(); ++i)
    {
        const std::string& name = params[i].name;
        float val = raw[i];

        if (name == "learningRate")
        {
            // Learning rate is set on the NNInfo skeleton, not TrainingConfig.
            // This value is returned for the caller to apply separately.
            continue;
        }
        else if (name == "optimizer")
        {
            cfg.optimizer.type = (val < 0.5f)
                ? OptimizerConfig::SGD_MOMENTUM
                : OptimizerConfig::ADAMW;
        }
        else if (name == "lrScheduleType")
        {
            int t = (int)(val + 0.5f);
            if (t <= 0) cfg.lrSchedule.setNone();
            else if (t == 1) cfg.lrSchedule.type = LearningRateScheduleConfig::STEP;
            else if (t == 2) cfg.lrSchedule.type = LearningRateScheduleConfig::EXP;
            else cfg.lrSchedule.type = LearningRateScheduleConfig::COSINE;
        }
        else if (name == "lrScheduleGamma")
        {
            cfg.lrSchedule.gamma = val;
        }
        else if (name == "adamBeta1")
        {
            cfg.optimizer.adamBeta1 = val;
        }
        else if (name == "adamBeta2")
        {
            cfg.optimizer.adamBeta2 = val;
        }
        else if (name == "globalGradClipNorm")
        {
            cfg.globalGradClipNorm = val;
        }
        else if (name == "warmupSteps")
        {
            cfg.warmup.type = (val > 0.5f)
                ? WarmupConfig::WARMUP_LINEAR
                : WarmupConfig::WARMUP_NONE;
            cfg.warmup.warmupSteps = (int)val;
        }
        else if (name == "weightDecay")
        {
            // Weight decay is applied via L2 regularization on NNInfo.
            // This value is returned for the caller to apply separately.
            continue;
        }
    }

    return cfg;
}
```

Add `hyperparameter_tuner.cpp` and `hyperparameter_tuner.h` to `Backend/Machine Learning/Networks/CMakeLists.txt`.

**Step 4: Run tests to verify they pass**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && cmake .. && make -j$(nproc) && make run search-space
```

Expected: All SearchSpace tests pass.

**Step 5: Commit**

```bash
git add Backend/Machine\ Learning/Networks/hyperparameter_tuner.h \
        Backend/Machine\ Learning/Networks/hyperparameter_tuner.cpp \
        Backend/Machine\ Learning/Networks/CMakeLists.txt \
        unit-tests/Backend/Machine\ Learning/hyperparameter-tuner-test.h \
        unit-tests/Backend/Machine\ Learning/hyperparameter-tuner-test.cpp \
        unit-tests/Backend/Machine\ Learning/CMakeLists.txt \
        unit-tests/main.cpp
git commit -m "Add SearchSpace with encode/decode and default training search space"
```

---

### Task 3: Implement HyperparameterTuner (Outer Loop)

**Files:**
- Modify: `Backend/Machine Learning/Networks/hyperparameter_tuner.h`
- Modify: `Backend/Machine Learning/Networks/hyperparameter_tuner.cpp`
- Modify: `unit-tests/Backend/Machine Learning/hyperparameter-tuner-test.cpp`

**Step 1: Write failing test for HyperparameterTuner**

Add to `hyperparameter-tuner-test.cpp`:

```cpp
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/DataInput.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"

void HyperparameterTunerUnitTest()
{
    printf("============================================================\n");
    printf("HyperparameterTuner Unit Test\n");
    printf("============================================================\n");

    // Build a simple DFF classification network as the template
    glades::NNInfo info;
    info.addInputLayer(2);
    info.addHiddenLayer(4);
    info.addOutputLayer(2, glades::GMath::CLASSIFICATION);
    info.setLearningRate(0, 0.01f);
    info.setLearningRate(1, 0.01f);

    glades::NNetwork templateNet(&info, glades::NNetwork::TYPE_DFF);

    // Create a simple XOR-like dataset
    shmea::GTable trainTable;
    // ... (populate with XOR data)
    // This test verifies the tuner API compiles and runs without crash.
    // Full convergence testing requires real data.

    // Test 1: Tuner construction
    glades::SearchSpace space;
    glades::HyperParameter lr;
    lr.name = "learningRate";
    lr.type = glades::HyperParameter::CONTINUOUS;
    lr.low = 0.001f;
    lr.high = 0.1f;
    lr.logScale = true;
    space.params.push_back(lr);

    glades::HyperparameterTuner tuner(space, 5, 10); // 5 trials, 10 epochs each
    G_assert(__FILE__, __LINE__,
        "==============HyperparameterTuner::construction==============",
        tuner.getMaxTrials() == 5);

    // Test 2: suggestNext returns valid normalized point
    std::vector<float> suggestion = tuner.suggestNext();
    G_assert(__FILE__, __LINE__,
        "==============HyperparameterTuner::suggestNext dim==============",
        suggestion.size() == 1);
    G_assert(__FILE__, __LINE__,
        "==============HyperparameterTuner::suggestNext in [0,1]==============",
        suggestion[0] >= 0.0f && suggestion[0] <= 1.0f);

    // Test 3: reportResult updates best
    tuner.reportResult(suggestion, 0.5f);
    tuner.reportResult(suggestion, 0.3f);
    G_assert(__FILE__, __LINE__,
        "==============HyperparameterTuner::bestScore==============",
        tuner.getBestScore() < 0.4f);

    printf("HyperparameterTuner tests passed\n");
    printf("============================================================\n");
}
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && make -j$(nproc) 2>&1 | tail -20
```

Expected: Compilation error, `HyperparameterTuner` class not defined.

**Step 3: Implement HyperparameterTuner**

Add to `hyperparameter_tuner.h`:

```cpp
class HyperparameterTuner
{
public:
    struct Trial
    {
        int id;
        std::vector<float> normalizedParams;
        std::vector<float> rawParams;
        float score;
        bool completed;
        bool pruned;

        Trial() : id(0), score(0.0f), completed(false), pruned(false) {}
    };

    HyperparameterTuner(const SearchSpace& space, int maxTrials, int epochsPerTrial);

    // Suggest next hyperparameter point (normalized [0,1]^N)
    std::vector<float> suggestNext();

    // Report the result of a trial
    void reportResult(const std::vector<float>& normalizedParams, float valLoss);

    // Run the full optimization loop on a template network
    TrainingConfig optimize(const NNetwork& templateNet,
                           const DataInput* trainData,
                           const DataInput* valData);

    int getMaxTrials() const { return maxTrials_; }
    int getEpochsPerTrial() const { return epochsPerTrial_; }
    float getBestScore() const { return optimizer_.getBestScore(); }
    const std::vector<float>& getBestParams() const { return optimizer_.getBestParams(); }
    const std::vector<Trial>& getTrials() const { return trials_; }

private:
    SearchSpace space_;
    BayesianOptimizer optimizer_;
    int maxTrials_;
    int epochsPerTrial_;
    int nInitialRandom_;
    int trialCounter_;
    std::vector<Trial> trials_;

    // Generate a random point in [0,1]^N for initial exploration
    std::vector<float> randomPoint() const;
};
```

Add to `hyperparameter_tuner.cpp`:

```cpp
HyperparameterTuner::HyperparameterTuner(const SearchSpace& space, int maxTrials, int epochsPerTrial)
    : space_(space),
      optimizer_(space.dimensions()),
      maxTrials_(maxTrials),
      epochsPerTrial_(epochsPerTrial),
      nInitialRandom_(5),
      trialCounter_(0)
{
    if (nInitialRandom_ > maxTrials_)
        nInitialRandom_ = maxTrials_;
}

std::vector<float> HyperparameterTuner::randomPoint() const
{
    unsigned int ndim = space_.dimensions();
    std::vector<float> point(ndim);
    for (unsigned int d = 0; d < ndim; ++d)
    {
        point[d] = (float)rand() / (float)RAND_MAX;
    }
    return point;
}

std::vector<float> HyperparameterTuner::suggestNext()
{
    if (trialCounter_ < nInitialRandom_)
    {
        return randomPoint();
    }
    return optimizer_.suggestNext();
}

void HyperparameterTuner::reportResult(const std::vector<float>& normalizedParams, float valLoss)
{
    Trial trial;
    trial.id = trialCounter_++;
    trial.normalizedParams = normalizedParams;
    trial.rawParams = space_.decode(normalizedParams);
    trial.score = valLoss;
    trial.completed = true;
    trial.pruned = false;
    trials_.push_back(trial);

    optimizer_.addObservation(normalizedParams, valLoss);
    optimizer_.fit();
}

TrainingConfig HyperparameterTuner::optimize(const NNetwork& templateNet,
                                              const DataInput* trainData,
                                              const DataInput* valData)
{
    // This method clones the template network for each trial,
    // trains it, evaluates on validation data, and reports results.
    //
    // Full implementation requires NNetwork::clone() which we add in Task 5.
    // For now, this is a stub that returns the base config.
    return templateNet.getTrainingConfig();
}
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && cmake .. && make -j$(nproc) && make run hp-tuner
```

Expected: All HyperparameterTuner tests pass.

**Step 5: Commit**

```bash
git add Backend/Machine\ Learning/Networks/hyperparameter_tuner.h \
        Backend/Machine\ Learning/Networks/hyperparameter_tuner.cpp \
        unit-tests/Backend/Machine\ Learning/hyperparameter-tuner-test.cpp
git commit -m "Add HyperparameterTuner with trial-based Bayesian optimization loop"
```

---

### Task 4: Add BAYESIAN LR Schedule (Inner Loop)

**Files:**
- Modify: `Backend/Machine Learning/Networks/training_config.h`
- Modify: `Backend/Machine Learning/Networks/network.h`
- Modify: `Backend/Machine Learning/Networks/network.cpp`
- Modify: `unit-tests/Backend/Machine Learning/hyperparameter-tuner-test.cpp`

**Step 1: Write failing test for Bayesian LR schedule**

Add to `hyperparameter-tuner-test.cpp`:

```cpp
void BayesianLRScheduleTest()
{
    printf("============================================================\n");
    printf("Bayesian LR Schedule Test\n");
    printf("============================================================\n");

    // Test that the BAYESIAN schedule type exists and the config struct works
    glades::LearningRateScheduleConfig sched;
    sched.type = glades::LearningRateScheduleConfig::BAYESIAN;

    // At epoch 0, before any observations, multiplier should be 1.0
    float mult = sched.multiplier(0);
    G_assert(__FILE__, __LINE__,
        "==============BayesianLR::initial multiplier == 1.0==============",
        fabs(mult - 1.0f) < 0.01f);

    // Test BayesianLRConfig defaults
    glades::BayesianLRConfig blr;
    G_assert(__FILE__, __LINE__,
        "==============BayesianLRConfig::defaults==============",
        blr.windowEpochs == 10 && blr.minLR > 0.0f && blr.maxLR > blr.minLR);

    printf("Bayesian LR Schedule tests passed\n");
    printf("============================================================\n");
}
```

Register as `"bayes-lr"` in `main.cpp`.

**Step 2: Run tests to verify they fail**

Expected: Compilation error, `BAYESIAN` not a valid enum value.

**Step 3: Add BAYESIAN to LearningRateScheduleConfig**

In `training_config.h`, modify `LearningRateScheduleConfig`:

```cpp
enum Type
{
    NONE = 0,
    STEP = 1,
    EXP = 2,
    COSINE = 3,
    BAYESIAN = 4
};
```

Add `BayesianLRConfig` struct after `LearningRateScheduleConfig`:

```cpp
struct BayesianLRConfig
{
    int windowEpochs;
    float minLR;
    float maxLR;

    BayesianLRConfig()
        : windowEpochs(10),
          minLR(1e-6f),
          maxLR(0.1f)
    {}
};
```

Add `BayesianLRConfig bayesianLR;` to `TrainingConfig` (after `lrSchedule`), and add it to `TrainingConfig`'s constructor initializer list.

In the `multiplier()` method of `LearningRateScheduleConfig`, add:

```cpp
case BAYESIAN:
    // Bayesian adaptive LR is managed externally by the trainer.
    // The schedule multiplier function returns 1.0 (neutral);
    // the actual multiplier is computed by the Bayesian LR controller
    // in the training loop and stored in lrScheduleMultiplier.
    return 1.0f;
```

**Step 4: Wire Bayesian LR into the training loop**

In `network.h`, add a forward declaration and member (behind `#include "bayes-optimizer.h"` or forward-declared):

```cpp
// Inside NNetwork private section:
float bayesianLRMultiplier_; // current Bayesian LR multiplier (1.0 default)
float bayesianLRLastLoss_;   // last observed loss for Bayesian LR
int bayesianLREpochCounter_; // epochs since last LR adjustment
```

In `network.cpp`, update `computeLearningRateMultiplier()`:

```cpp
float glades::NNetwork::computeLearningRateMultiplier(int epochFromStart) const
{
    if (trainingConfig.lrSchedule.type == LearningRateScheduleConfig::BAYESIAN)
        return bayesianLRMultiplier_;
    return trainingConfig.lrSchedule.multiplier(epochFromStart);
}
```

In `trainer.cpp`, after the epoch metrics are computed (around line 440), add the Bayesian LR update logic:

```cpp
// Bayesian adaptive LR update
if (isTrainRun && net.trainingConfig.lrSchedule.type == LearningRateScheduleConfig::BAYESIAN)
{
    ++net.bayesianLREpochCounter_;
    if (net.bayesianLREpochCounter_ >= net.trainingConfig.bayesianLR.windowEpochs)
    {
        net.bayesianLREpochCounter_ = 0;
        float currentLoss = metrics.totalError;
        float currentLR = net.bayesianLRMultiplier_;

        // Feed observation to internal GP
        net.bayesianLRGP_.addSample(log(currentLR), currentLoss);
        net.bayesianLRGP_.fit();

        // Use EI to suggest next LR (in log space)
        // Simple 1D grid search over [log(minLR), log(maxLR)]
        float logMin = log(net.trainingConfig.bayesianLR.minLR);
        float logMax = log(net.trainingConfig.bayesianLR.maxLR);
        float bestEI = -1.0f;
        float bestLogLR = log(currentLR);
        float bestLoss = currentLoss;

        // Need at least 2 observations before using EI
        if (net.bayesianLRGP_.numSamples() >= 2)
        {
            // Find best observed loss so far
            // (already tracked by observations)
            for (float logLR = logMin; logLR <= logMax; logLR += (logMax - logMin) / 100.0f)
            {
                std::pair<float, float> pred = net.bayesianLRGP_.predict(logLR);
                float mu = pred.first;
                float sigma = sqrt(pred.second);
                if (sigma < 1e-8f) continue;
                float z = (bestLoss - mu) / sigma;
                float ei = (bestLoss - mu) * BayesianOptimizer::cdf(z)
                         + sigma * BayesianOptimizer::pdf(z);
                if (ei > bestEI)
                {
                    bestEI = ei;
                    bestLogLR = logLR;
                }
            }
        }

        net.bayesianLRMultiplier_ = exp(bestLogLR);
        // Clamp
        if (net.bayesianLRMultiplier_ < net.trainingConfig.bayesianLR.minLR)
            net.bayesianLRMultiplier_ = net.trainingConfig.bayesianLR.minLR;
        if (net.bayesianLRMultiplier_ > net.trainingConfig.bayesianLR.maxLR)
            net.bayesianLRMultiplier_ = net.trainingConfig.bayesianLR.maxLR;
    }
}
```

Note: `bayesianLRGP_` is a `GaussianProcess` member on NNetwork (1D, for the inner loop). Add it to the private section of `NNetwork` alongside the other Bayesian LR members. Initialize `bayesianLRMultiplier_` to 1.0f, `bayesianLREpochCounter_` to 0 in the NNetwork constructor.

**Step 5: Run tests to verify they pass**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && cmake .. && make -j$(nproc) && make run bayes-lr
```

Expected: Bayesian LR schedule tests pass.

**Step 6: Commit**

```bash
git add Backend/Machine\ Learning/Networks/training_config.h \
        Backend/Machine\ Learning/Networks/network.h \
        Backend/Machine\ Learning/Networks/network.cpp \
        Backend/Machine\ Learning/Networks/trainer.cpp \
        unit-tests/Backend/Machine\ Learning/hyperparameter-tuner-test.cpp \
        unit-tests/main.cpp
git commit -m "Add BAYESIAN adaptive LR schedule type with inner-loop GP controller"
```

---

### Task 5: Wire HyperparameterTuner into NNetwork (Full Loop)

**Files:**
- Modify: `Backend/Machine Learning/Networks/network.h`
- Modify: `Backend/Machine Learning/Networks/network.cpp`
- Modify: `Backend/Machine Learning/Networks/hyperparameter_tuner.cpp`
- Modify: `unit-tests/Backend/Machine Learning/hyperparameter-tuner-test.cpp`

**Step 1: Write failing test for full optimize loop**

Add to `hyperparameter-tuner-test.cpp`:

```cpp
void HyperparameterTunerFullLoopTest()
{
    printf("============================================================\n");
    printf("HyperparameterTuner Full Loop Test\n");
    printf("============================================================\n");

    // Build a simple DFF network for XOR
    glades::NNInfo info;
    info.addInputLayer(2);
    info.addHiddenLayer(4);
    info.addOutputLayer(2, glades::GMath::CLASSIFICATION);
    info.setLearningRate(0, 0.1f);
    info.setLearningRate(1, 0.1f);

    glades::NNetwork templateNet(&info, glades::NNetwork::TYPE_DFF);

    // Load XOR dataset
    shmea::GString fname = "datasets/xor.csv";
    shmea::GTable table(fname, ',', shmea::GTable::TYPE_FILE);
    glades::DataInput di;
    di.load(table);

    // Create a minimal search space (just LR)
    glades::SearchSpace space;
    glades::HyperParameter lr;
    lr.name = "learningRate";
    lr.type = glades::HyperParameter::CONTINUOUS;
    lr.low = 0.01f;
    lr.high = 0.5f;
    lr.logScale = true;
    space.params.push_back(lr);

    // Run tuner: 3 trials, 50 epochs each
    glades::HyperparameterTuner tuner(space, 3, 50);
    glades::TrainingConfig bestConfig = tuner.optimize(templateNet, &di, &di);

    // Verify that the tuner found something (completed 3 trials)
    G_assert(__FILE__, __LINE__,
        "==============HyperparameterTuner::fullLoop completed trials==============",
        tuner.getTrials().size() == 3);

    // Best score should be a valid finite number
    G_assert(__FILE__, __LINE__,
        "==============HyperparameterTuner::fullLoop bestScore finite==============",
        std::isfinite(tuner.getBestScore()));

    printf("HyperparameterTuner full loop test passed\n");
    printf("============================================================\n");
}
```

Register as `"hp-tuner-full"` in `main.cpp`.

**Step 2: Run tests to verify they fail**

Expected: The `optimize()` stub returns immediately without running trials.

**Step 3: Add NNetwork cloning support**

In `network.h`, add to the public section:

```cpp
// Create a deep copy of this network (architecture + fresh weights).
// The clone has the same architecture and training config, but resets
// epoch counters and optimizer state for a fresh training run.
NNetwork* cloneForTrial() const;
```

In `network.cpp`, implement:

```cpp
glades::NNetwork* glades::NNetwork::cloneForTrial() const
{
    NNetwork* net = new NNetwork(skeleton, netType);
    net->trainingConfig = trainingConfig;
    net->terminator = terminator;
    // Fresh seed for each trial (offset by trial count to vary)
    net->setSeed(rngSeed + 1);
    return net;
}
```

**Step 4: Implement full optimize loop**

In `hyperparameter_tuner.cpp`, replace the `optimize()` stub. Add necessary includes:

```cpp
#include "network.h"
#include "../DataObjects/DataInput.h"
#include "training_callbacks.h"
```

```cpp
namespace {
// Internal callback for pruning support
struct TunerTrialCallback : public glades::ITrainingCallbacks
{
    float lastLoss;
    TunerTrialCallback() : lastLoss(0.0f) {}
    bool onEpochEnd(const glades::NNetwork& net, const glades::NNetworkEpochMetrics& metrics)
    {
        lastLoss = metrics.totalError;
        return false; // don't stop
    }
};
} // anon namespace

TrainingConfig HyperparameterTuner::optimize(const NNetwork& templateNet,
                                              const DataInput* trainData,
                                              const DataInput* valData)
{
    for (int t = 0; t < maxTrials_; ++t)
    {
        // Suggest next point
        std::vector<float> normalized = suggestNext();
        std::vector<float> raw = space_.decode(normalized);

        // Clone template network
        NNetwork* trialNet = templateNet.cloneForTrial();

        // Apply hyperparameters
        TrainingConfig trialConfig = space_.applyToConfig(raw, templateNet.getTrainingConfig());
        trialNet->setTrainingConfig(trialConfig);

        // Apply learning rate if present in search space
        for (unsigned int i = 0; i < space_.params.size(); ++i)
        {
            if (space_.params[i].name == "learningRate")
            {
                const NNInfo* info = trialNet->getNNInfo();
                if (info)
                {
                    // Set LR on all transitions
                    for (unsigned int layer = 0; layer <= (unsigned int)info->numHiddenLayers(); ++layer)
                        const_cast<NNInfo*>(info)->setLearningRate(layer, raw[i]);
                }
            }
        }

        // Set epoch limit via terminator
        Terminator term;
        term.setEpochLimit(epochsPerTrial_);
        trialNet->setTerminator(term);

        // Train
        TunerTrialCallback cb;
        trialNet->train(trainData, &cb);

        // Evaluate on validation set
        TunerTrialCallback valCb;
        trialNet->test(valData, &valCb);
        float valLoss = valCb.lastLoss;

        // Report result
        reportResult(normalized, valLoss);

        delete trialNet;
    }

    // Return the best config
    std::vector<float> bestRaw = space_.decode(optimizer_.getBestParams());
    return space_.applyToConfig(bestRaw, templateNet.getTrainingConfig());
}
```

**Step 5: Run tests to verify they pass**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && cmake .. && make -j$(nproc) && make run hp-tuner-full
```

Expected: Full loop test passes (3 trials completed, finite best score).

Also run existing tests to verify no regressions:

```bash
make run bayes-optimizer && make run bayes-optimizer-nd && make run search-space && make run hp-tuner
```

**Step 6: Commit**

```bash
git add Backend/Machine\ Learning/Networks/network.h \
        Backend/Machine\ Learning/Networks/network.cpp \
        Backend/Machine\ Learning/Networks/hyperparameter_tuner.h \
        Backend/Machine\ Learning/Networks/hyperparameter_tuner.cpp \
        unit-tests/Backend/Machine\ Learning/hyperparameter-tuner-test.cpp \
        unit-tests/main.cpp
git commit -m "Wire HyperparameterTuner full optimization loop with NNetwork cloning"
```

---

### Task 6: Regression Testing and Cleanup

**Files:**
- All modified files from Tasks 1-5

**Step 1: Build the main library**

```bash
cd /home/robert/dev/glades-ml/build && cmake .. && make -j$(nproc)
```

Expected: Clean build with no warnings related to new code.

**Step 2: Build and run full unit test suite**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && cmake .. && make -j$(nproc) && make run bayes-optimizer
```

Expected: Existing bayes-optimizer test passes (backward compatibility).

```bash
make run nn && make run bayes
```

Expected: All existing tests pass.

**Step 3: Run all new tests**

```bash
make run bayes-optimizer-nd && make run search-space && make run hp-tuner && make run bayes-lr && make run hp-tuner-full
```

Expected: All new tests pass.

**Step 4: Final commit if any cleanup needed**

```bash
git add -A && git commit -m "Final cleanup and regression test verification"
```

---

## Notes for the Implementing Engineer

### Build system
- Main library: `cd build && cmake .. && make -j$(nproc)`
- Unit tests: `cd unit-tests/build && cmake .. && make -j$(nproc)`
- Run specific test: `make run <test-name>` (e.g., `make run bayes-optimizer-nd`)
- C++98 standard — no `auto`, no range-for, no brace-init, no `nullptr`

### Key files to understand before starting
- `bayes-optimizer.h/cpp` — the existing 1D GP implementation you're upgrading
- `training_config.h` — all config structs, especially `LearningRateScheduleConfig` and `TrainingConfig`
- `trainer.cpp` — the training loop where LR schedule is applied (line ~286-293)
- `network.h` — NNetwork class (1500+ lines), specifically the LR-related methods around line 1446
- `training_callbacks.h` — `ITrainingCallbacks` and `NNetworkEpochMetrics`

### Testing conventions
- Tests use `G_assert(__FILE__, __LINE__, "message", expr)` — no gtest/catch
- Each test function is registered in `main.cpp` with a string key
- Test files need both `.h` and `.cpp`, registered in CMakeLists.txt

### Important constraints
- NNetwork is not copyable via copy constructor. Use the `cloneForTrial()` method.
- The codebase uses tabs for indentation (except CUDA .cu files which use 4 spaces).
- The GP's Cholesky can fail if the covariance matrix isn't positive definite. The jitter fallback (adding 1e-3 to diagonal) handles this.
- The Bayesian LR inner loop modifies `net.bayesianLRMultiplier_` during training — this makes `trainer.cpp` non-const w.r.t. the network. This is consistent with how other mutable state (epoch counters, metrics) is handled.
