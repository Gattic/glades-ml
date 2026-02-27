// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef _GHYPERPARAMETER_TUNER
#define _GHYPERPARAMETER_TUNER

#include "training_config.h"
#include "bayes-optimizer.h"
#include "../rng.h"
#include <vector>
#include <string>
#include <cmath>
#include <limits>

namespace glades {

class NNetwork;
class DataInput;

// Describes a single hyperparameter dimension in the search space.
struct HyperParameter
{
	enum Type { CONTINUOUS = 0, INTEGER = 1, CATEGORICAL = 2 };

	std::string name;
	Type type;
	float low, high;
	std::vector<float> choices; // for CATEGORICAL
	bool logScale;

	HyperParameter()
		: name(""),
		  type(CONTINUOUS),
		  low(0.0f),
		  high(1.0f),
		  choices(),
		  logScale(false)
	{
	}
};

// Defines the search space for hyperparameter tuning: a collection of HyperParameter
// dimensions with encode/decode between raw values and normalized [0,1]^N representation.
struct SearchSpace
{
	std::vector<HyperParameter> params;

	unsigned int dimensions() const;

	// Raw values -> normalized [0,1]^N.
	// For log-scale params, encoding is done in log space.
	// For categorical params, the choice index is mapped to [0,1].
	std::vector<float> encode(const std::vector<float>& raw) const;

	// Normalized [0,1]^N -> raw values.
	// Integer params are rounded. Categorical params snap to nearest choice.
	std::vector<float> decode(const std::vector<float>& normalized) const;

	// Create a default search space for training config hyperparameters.
	static SearchSpace defaultTrainingSearchSpace();

	// Apply decoded raw values to a TrainingConfig copy.
	// "learningRate" and "weightDecay" are skipped (those go on NNInfo, not TrainingConfig).
	TrainingConfig applyToConfig(const std::vector<float>& raw, const TrainingConfig& base) const;
};

// Outer loop for hyperparameter optimization using Bayesian optimization.
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

		Trial()
			: id(0),
			  normalizedParams(),
			  rawParams(),
			  score(std::numeric_limits<float>::max()),
			  completed(false),
			  pruned(false)
		{
		}
	};

	HyperparameterTuner(const SearchSpace& space, int maxTrials, int epochsPerTrial);

	// Suggest the next point in normalized [0,1]^N to evaluate.
	// For the first nInitialRandom_ trials, returns a random point.
	// After that, uses Bayesian optimization.
	std::vector<float> suggestNext();

	// Report the result of evaluating a point.
	void reportResult(const std::vector<float>& normalizedParams, float valLoss);

	// Run the full optimization loop. STUB: returns templateNet's training config.
	TrainingConfig optimize(const NNetwork& templateNet,
	                        const DataInput* trainData,
	                        const DataInput* valData);

	// Getters
	int getMaxTrials() const;
	int getEpochsPerTrial() const;
	float getBestScore() const;
	const std::vector<float>& getBestParams() const;
	const std::vector<Trial>& getTrials() const;

private:
	SearchSpace space_;
	BayesianOptimizer optimizer_;
	int maxTrials_;
	int epochsPerTrial_;
	int nInitialRandom_;
	int trialCounter_;
	std::vector<Trial> trials_;
	glades::rng::Engine rng_;

	// Generate a random point in [0,1]^N.
	std::vector<float> randomPoint() const;
};

} // namespace glades

#endif
