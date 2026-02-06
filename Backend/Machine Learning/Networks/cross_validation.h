// Cross-validation utilities for the Glades ML engine.
//
// Goals:
// - Deterministic fold construction (seeded shuffling).
// - No data leakage: standardization/OHE are fitted on TRAIN and applied to TEST.
// - No weight leakage: each fold trains a fresh model clone (architecture + hyperparams).
// - No side effects: no filesystem I/O and no stdout/stderr printing.
//
// This is the only supported Cross Validation API.
#pragma once

#include "Backend/Database/GTable.h"
#include <stdint.h>
#include <vector>

namespace glades {

class NNetwork;
struct NNetworkStatus;

struct CrossValidationConfig
{
	// Number of folds for k-fold CV.
	// Must be >= 2 for meaningful CV. If <= 1, the runner will return INVALID_ARGUMENT.
	unsigned int kFolds;

	// If true and !timeSeries, shuffle rows before splitting into folds.
	bool shuffle;

	// RNG seed used when shuffle is enabled.
	uint64_t seed;

	// If true, use walk-forward splits:
	// - Rows are NOT shuffled.
	// - Each fold tests on a contiguous block, and trains on all rows before that block.
	// The first fold may be skipped if it would produce an empty training set.
	bool timeSeries;

	// If true and !timeSeries, attempt stratified splitting based on the first output column.
	// Stratification is best-effort; it is disabled automatically if a single output column
	// cannot be identified.
	bool stratify;

	// Standardization mode applied to numeric columns:
	// - NONE: no scaling
	// - MINMAX / ZSCORE: fit on train fold, apply to train+test
	int standardizeFlag;

	CrossValidationConfig();
};

struct CrossValidationResults
{
	// Per-network mean test accuracy across folds (in % units used by NNetwork::getAccuracy()).
	std::vector<float> meanTestAccuracy;
	// Per-network per-fold test accuracies [net][fold].
	std::vector< std::vector<float> > foldTestAccuracy;

	// Metadata for debugging/introspection.
	unsigned int foldsUsed;
	unsigned int totalRows;

	CrossValidationResults() : foldsUsed(0u), totalRows(0u) {}
};

// Run k-fold cross validation over an input table for one or more model templates.
//
// Each element in `modelTemplates` serves only as an architectural+hyperparameter template.
// The runner trains and evaluates *fresh clones* per fold; the passed templates are NOT mutated.
//
// Returns status.OK on success, otherwise an error code with message.
NNetworkStatus crossValidateTableCSV(const shmea::GTable& input,
                                    const std::vector<glades::NNetwork*>& modelTemplates,
                                    const CrossValidationConfig& cfg,
                                    CrossValidationResults* outResults);

// Train and evaluate model templates on explicit train/test tables.
// This is useful for holdout validation evaluation after CV.
NNetworkStatus trainTestTableCSV(const shmea::GTable& trainTbl,
                                 const shmea::GTable& testTbl,
                                 const std::vector<glades::NNetwork*>& modelTemplates,
                                 const CrossValidationConfig& cfg,
                                 std::vector<float>* outTestAccuracies);

} // namespace glades

