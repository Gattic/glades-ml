# Bayesian Hyperparameter Optimizer Design

**Date:** 2026-02-22
**Status:** Approved
**Branch:** gan1

## Motivation

- Manual learning rate tuning is tedious and error-prone
- Suboptimal hyperparameters hurt model convergence
- Need to tune all TrainingConfig parameters, not just LR
- Existing `BayesianOptimizer` class is 1D-only and not integrated into training

## Requirements

- Full hyperparameter optimization across all tunable TrainingConfig parameters
- Both outer-loop (trial-based search) and inner-loop (adaptive LR mid-training)
- Pure C++, no new external dependencies
- Build on existing `BayesianOptimizer` / `GaussianProcess` classes

## Design

### 1. Multi-Dimensional Gaussian Process

Upgrade existing `GaussianProcess` from 1D to N-D:

- **Samples:** `vector<vector<float>>` instead of `vector<float>` for inputs
- **ARD Kernel:** Per-dimension length scales so the GP learns which hyperparameters matter more
  - `k(x1, x2) = variance * exp(-0.5 * sum((x1[d]-x2[d])^2 / length_scale[d]^2))`
  - `length_scale_` becomes `vector<float>` (one per dimension)
- **Cholesky Decomposition:** Replace Gaussian elimination matrix inversion with Cholesky for numerical stability on symmetric positive-definite covariance matrices
- **Prediction:** `predict(vector<float> x)` returns `{mu, sigma2}` as before

**Acquisition function optimization:** Replace brute-force grid scan with Latin Hypercube random sampling (200 candidates) + local hill-climbing on the top candidates. Avoids exponential blowup in N dimensions.

### 2. Search Space Definition

```cpp
struct HyperParameter {
    enum Type { CONTINUOUS, INTEGER, CATEGORICAL };
    std::string name;
    Type type;
    float low, high;                    // for CONTINUOUS/INTEGER
    std::vector<float> choices;         // for CATEGORICAL (encoded as floats)
    bool logScale;                      // for CONTINUOUS (e.g., learning rate)
};

struct SearchSpace {
    std::vector<HyperParameter> params;
    std::vector<float> encode(const std::vector<float>& raw) const;
    std::vector<float> decode(const std::vector<float>& normalized) const;
};
```

The GP operates in normalized `[0,1]^N` space. SearchSpace handles mapping to/from actual values (log-scale for LR, integer rounding for batch size, index mapping for categoricals).

**Default search space:**

| Parameter | Type | Range | Scale |
|-----------|------|-------|-------|
| learningRate | continuous | [1e-5, 0.1] | log |
| optimizer | categorical | {SGD_MOMENTUM, ADAMW} | - |
| lrSchedule.type | categorical | {NONE, STEP, EXP, COSINE} | - |
| lrSchedule.gamma | continuous | [0.1, 0.999] | linear |
| adamBeta1 | continuous | [0.8, 0.99] | linear |
| adamBeta2 | continuous | [0.99, 0.9999] | linear |
| globalGradClipNorm | continuous | [0.0, 10.0] | linear |
| warmup.steps | integer | [0, 5000] | linear |
| weightDecay (L2) | continuous | [1e-6, 0.1] | log |

### 3. HyperparameterTuner (Outer Loop)

```cpp
class HyperparameterTuner {
    SearchSpace space_;
    BayesianOptimizer optimizer_;
    int maxTrials_;
    int epochsPerTrial_;

    struct Trial {
        int id;
        std::vector<float> params;     // normalized [0,1]^N
        TrainingConfig config;          // decoded actual values
        float score;                    // validation loss (lower = better)
        bool pruned;
    };

    std::vector<Trial> trials_;

public:
    TrainingConfig optimize(const NNetwork& templateNet,
                           const DataInput* trainData,
                           const DataInput* valData);
    TrainingConfig suggestNext();
    void reportResult(int trialId, float valLoss);
};
```

**Flow:**
1. Run `n_initial` random trials (default 5) to seed the GP
2. For each subsequent trial: use GP + Expected Improvement to suggest next config
3. Train a fresh clone of the template network with that config for `epochsPerTrial_` epochs
4. Evaluate on validation set, record validation loss
5. Feed result back to GP, repeat until `maxTrials_` reached
6. Optional: median pruning -- if a trial's loss at epoch E is worse than the median of completed trials at epoch E, prune it early

**Integration:** Uses `NNetwork::clone()` + existing `train()` / `test()` API. Each trial is a standalone training run. The `ITrainingCallbacks::onEpochEnd` callback is used for early stopping and pruning checks.

### 4. Adaptive LR Schedule (Inner Loop)

New `LearningRateScheduleConfig::Type::BAYESIAN` for mid-training LR adaptation:

- Every `window` epochs (default 10), evaluate current validation loss
- Maintain a 1D GP mapping `log(lr) -> validation_loss` from the trajectory so far
- Use Expected Improvement to suggest the next LR for the next window
- Clamp to `[minLR, maxLR]` bounds

Plugs into existing `computeLearningRateMultiplier()` -- returns a multiplier just like STEP/EXP/COSINE. Does not restart training; adapts LR of a running session based on observed loss trajectory.

**Config additions:**
```cpp
struct BayesianLRConfig {
    int windowEpochs = 10;      // epochs between LR adjustments
    float minLR = 1e-6f;
    float maxLR = 0.1f;
};
```

### 5. File Organization

| File | Purpose |
|------|---------|
| `bayes-optimizer.h/cpp` | Multi-D GP + BayesianOptimizer (upgraded from existing) |
| `hyperparameter_tuner.h/cpp` | SearchSpace, HyperparameterTuner (new) |
| `training_config.h` | Add BAYESIAN to LR schedule enum, add BayesianLRConfig |
| `trainer.cpp` | Wire in BAYESIAN schedule multiplier computation |
| `network.h/cpp` | Add clone() method if not present, add tuner integration |

### 6. Technical Decisions

- **Cholesky over Gaussian elimination:** More numerically stable for SPD matrices, natural fit for GP covariance matrices
- **ARD kernel over isotropic:** Per-dimension length scales let the GP automatically learn which hyperparameters are important
- **Latin Hypercube + hill-climbing for acquisition:** Avoids exponential grid blowup in N-D while being simple to implement in pure C++
- **Normalized [0,1]^N space:** Keeps the GP well-conditioned regardless of actual parameter scales
- **Median pruning:** Simple early stopping heuristic that doesn't require additional surrogate models
- **~50 trial limit:** GP with Cholesky is O(n^3) which is fine for n<50; beyond that consider approximations

### 7. Out of Scope (Future Work)

- Multi-fidelity optimization (Hyperband-style successive halving)
- Parallel/async trial execution
- GP kernel hyperparameter optimization (length scales fixed or set by heuristic)
- Distributed hyperparameter search across machines
- TPE as alternative surrogate model
