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
#include "hyperparameter_tuner.h"
#include "network.h"
#include "../DataObjects/DataInput.h"
#include "training_callbacks.h"
#include <cstdio>
#include <cmath>

namespace glades {

// ============================================================================
// SearchSpace
// ============================================================================

unsigned int SearchSpace::dimensions() const
{
	return static_cast<unsigned int>(params.size());
}

std::vector<float> SearchSpace::encode(const std::vector<float>& raw) const
{
	std::vector<float> normalized(params.size(), 0.0f);
	for (unsigned int i = 0; i < params.size(); ++i)
	{
		const HyperParameter& hp = params[i];
		float v = raw[i];

		switch (hp.type)
		{
		case HyperParameter::CONTINUOUS:
		case HyperParameter::INTEGER:
		{
			if (hp.logScale)
			{
				// Encode in log space: normalized = (log(v) - log(low)) / (log(high) - log(low))
				float logLow = logf(hp.low);
				float logHigh = logf(hp.high);
				float logV = logf(v);
				float denom = logHigh - logLow;
				if (denom < 1e-12f)
					normalized[i] = 0.0f;
				else
					normalized[i] = (logV - logLow) / denom;
			}
			else
			{
				// Linear: normalized = (v - low) / (high - low)
				float denom = hp.high - hp.low;
				if (denom < 1e-12f)
					normalized[i] = 0.0f;
				else
					normalized[i] = (v - hp.low) / denom;
			}
			break;
		}
		case HyperParameter::CATEGORICAL:
		{
			// Find the index of the choice that matches v
			int idx = 0;
			float bestDist = 1e30f;
			for (unsigned int c = 0; c < hp.choices.size(); ++c)
			{
				float dist = fabsf(v - hp.choices[c]);
				if (dist < bestDist)
				{
					bestDist = dist;
					idx = static_cast<int>(c);
				}
			}
			// Map index to [0,1]: idx / (nChoices - 1), or 0 if only 1 choice
			if (hp.choices.size() <= 1)
				normalized[i] = 0.0f;
			else
				normalized[i] = static_cast<float>(idx) / static_cast<float>(hp.choices.size() - 1);
			break;
		}
		}
	}
	return normalized;
}

std::vector<float> SearchSpace::decode(const std::vector<float>& normalized) const
{
	std::vector<float> raw(params.size(), 0.0f);
	for (unsigned int i = 0; i < params.size(); ++i)
	{
		const HyperParameter& hp = params[i];
		float t = normalized[i];

		// Clamp to [0,1]
		if (t < 0.0f) t = 0.0f;
		if (t > 1.0f) t = 1.0f;

		switch (hp.type)
		{
		case HyperParameter::CONTINUOUS:
		{
			if (hp.logScale)
			{
				// Decode from log space: v = exp(log(low) + t * (log(high) - log(low)))
				float logLow = logf(hp.low);
				float logHigh = logf(hp.high);
				raw[i] = expf(logLow + t * (logHigh - logLow));
			}
			else
			{
				raw[i] = hp.low + t * (hp.high - hp.low);
			}
			break;
		}
		case HyperParameter::INTEGER:
		{
			float v;
			if (hp.logScale)
			{
				float logLow = logf(hp.low);
				float logHigh = logf(hp.high);
				v = expf(logLow + t * (logHigh - logLow));
			}
			else
			{
				v = hp.low + t * (hp.high - hp.low);
			}
			// Round to nearest integer
			raw[i] = floorf(v + 0.5f);
			// Clamp to bounds
			if (raw[i] < hp.low) raw[i] = hp.low;
			if (raw[i] > hp.high) raw[i] = hp.high;
			break;
		}
		case HyperParameter::CATEGORICAL:
		{
			if (hp.choices.empty())
			{
				raw[i] = 0.0f;
				break;
			}
			// Snap to nearest choice index
			float fidx = t * static_cast<float>(hp.choices.size() - 1);
			int idx = static_cast<int>(floorf(fidx + 0.5f));
			if (idx < 0) idx = 0;
			if (idx >= static_cast<int>(hp.choices.size()))
				idx = static_cast<int>(hp.choices.size()) - 1;
			raw[i] = hp.choices[idx];
			break;
		}
		}
	}
	return raw;
}

SearchSpace SearchSpace::defaultTrainingSearchSpace()
{
	SearchSpace space;

	// learningRate: continuous, [1e-5, 0.1], log scale
	{
		HyperParameter hp;
		hp.name = "learningRate";
		hp.type = HyperParameter::CONTINUOUS;
		hp.low = 1e-5f;
		hp.high = 0.1f;
		hp.logScale = true;
		space.params.push_back(hp);
	}

	// optimizer: categorical, {0.0=SGD_MOMENTUM, 1.0=ADAMW}
	{
		HyperParameter hp;
		hp.name = "optimizer";
		hp.type = HyperParameter::CATEGORICAL;
		hp.low = 0.0f;
		hp.high = 1.0f;
		hp.logScale = false;
		hp.choices.push_back(0.0f); // SGD_MOMENTUM
		hp.choices.push_back(1.0f); // ADAMW
		space.params.push_back(hp);
	}

	// lrScheduleType: categorical, {0=NONE, 1=STEP, 2=EXP, 3=COSINE}
	{
		HyperParameter hp;
		hp.name = "lrScheduleType";
		hp.type = HyperParameter::CATEGORICAL;
		hp.low = 0.0f;
		hp.high = 3.0f;
		hp.logScale = false;
		hp.choices.push_back(0.0f); // NONE
		hp.choices.push_back(1.0f); // STEP
		hp.choices.push_back(2.0f); // EXP
		hp.choices.push_back(3.0f); // COSINE
		space.params.push_back(hp);
	}

	// lrScheduleGamma: continuous, [0.1, 0.999], linear
	{
		HyperParameter hp;
		hp.name = "lrScheduleGamma";
		hp.type = HyperParameter::CONTINUOUS;
		hp.low = 0.1f;
		hp.high = 0.999f;
		hp.logScale = false;
		space.params.push_back(hp);
	}

	// adamBeta1: continuous, [0.8, 0.99], linear
	{
		HyperParameter hp;
		hp.name = "adamBeta1";
		hp.type = HyperParameter::CONTINUOUS;
		hp.low = 0.8f;
		hp.high = 0.99f;
		hp.logScale = false;
		space.params.push_back(hp);
	}

	// adamBeta2: continuous, [0.99, 0.9999], linear
	{
		HyperParameter hp;
		hp.name = "adamBeta2";
		hp.type = HyperParameter::CONTINUOUS;
		hp.low = 0.99f;
		hp.high = 0.9999f;
		hp.logScale = false;
		space.params.push_back(hp);
	}

	// globalGradClipNorm: continuous, [0.0, 10.0], linear
	{
		HyperParameter hp;
		hp.name = "globalGradClipNorm";
		hp.type = HyperParameter::CONTINUOUS;
		hp.low = 0.0f;
		hp.high = 10.0f;
		hp.logScale = false;
		space.params.push_back(hp);
	}

	// warmupSteps: integer, [0, 5000], linear
	{
		HyperParameter hp;
		hp.name = "warmupSteps";
		hp.type = HyperParameter::INTEGER;
		hp.low = 0.0f;
		hp.high = 5000.0f;
		hp.logScale = false;
		space.params.push_back(hp);
	}

	// weightDecay: continuous, [1e-6, 0.1], log scale
	{
		HyperParameter hp;
		hp.name = "weightDecay";
		hp.type = HyperParameter::CONTINUOUS;
		hp.low = 1e-6f;
		hp.high = 0.1f;
		hp.logScale = true;
		space.params.push_back(hp);
	}

	return space;
}

TrainingConfig SearchSpace::applyToConfig(const std::vector<float>& raw,
                                          const TrainingConfig& base) const
{
	TrainingConfig cfg = base;

	for (unsigned int i = 0; i < params.size(); ++i)
	{
		const std::string& name = params[i].name;
		float val = raw[i];

		// Skip learningRate and weightDecay (those go on NNInfo, not TrainingConfig)
		if (name == "learningRate" || name == "weightDecay")
			continue;

		if (name == "optimizer")
		{
			int ival = static_cast<int>(val + 0.5f);
			if (ival == 0)
				cfg.optimizer.type = OptimizerConfig::SGD_MOMENTUM;
			else
				cfg.optimizer.type = OptimizerConfig::ADAMW;
		}
		else if (name == "lrScheduleType")
		{
			int ival = static_cast<int>(val + 0.5f);
			if (ival == 0)
				cfg.lrSchedule.type = LearningRateScheduleConfig::NONE;
			else if (ival == 1)
				cfg.lrSchedule.type = LearningRateScheduleConfig::STEP;
			else if (ival == 2)
				cfg.lrSchedule.type = LearningRateScheduleConfig::EXP;
			else
				cfg.lrSchedule.type = LearningRateScheduleConfig::COSINE;
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
			int steps = static_cast<int>(val + 0.5f);
			cfg.warmup.warmupSteps = steps;
			cfg.warmup.type = (steps > 0) ? WarmupConfig::WARMUP_LINEAR : WarmupConfig::WARMUP_NONE;
		}
	}

	return cfg;
}

// ============================================================================
// HyperparameterTuner
// ============================================================================

HyperparameterTuner::HyperparameterTuner(const SearchSpace& space, int maxTrials, int epochsPerTrial)
	: space_(space),
	  optimizer_(space.dimensions()),
	  maxTrials_(maxTrials),
	  epochsPerTrial_(epochsPerTrial),
	  nInitialRandom_(5),
	  trialCounter_(0),
	  trials_(),
	  rng_()
{
	// If maxTrials is small, reduce the number of initial random trials
	if (nInitialRandom_ > maxTrials_)
		nInitialRandom_ = maxTrials_;
}

std::vector<float> HyperparameterTuner::randomPoint() const
{
	// NOTE: randomPoint() is const but needs to generate random numbers.
	// We use a mutable-like pattern by casting away const on the rng.
	// This is safe because randomPoint does not modify observable state.
	glades::rng::Engine& rngRef = const_cast<glades::rng::Engine&>(rng_);

	unsigned int ndim = space_.dimensions();
	std::vector<float> point(ndim);
	for (unsigned int i = 0; i < ndim; ++i)
	{
		point[i] = glades::rng::unit_float01(rngRef);
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
	trial.id = trialCounter_;
	trial.normalizedParams = normalizedParams;
	trial.rawParams = space_.decode(normalizedParams);
	trial.score = valLoss;
	trial.completed = true;
	trial.pruned = false;
	trials_.push_back(trial);

	optimizer_.addObservation(normalizedParams, valLoss);
	optimizer_.fit();

	++trialCounter_;
}

namespace {
struct TunerTrialCallback : public glades::ITrainingCallbacks
{
	float lastLoss;
	TunerTrialCallback() : lastLoss(0.0f) {}
	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		lastLoss = m.totalError;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}
};
} // anonymous namespace

TrainingConfig HyperparameterTuner::optimize(const NNetwork& templateNet,
                                             const DataInput* trainData,
                                             const DataInput* valData)
{
	TrainingConfig bestConfig = templateNet.getTrainingConfig();
	float bestLoss = std::numeric_limits<float>::max();

	for (int trial = 0; trial < maxTrials_; ++trial)
	{
		// 1. Suggest next point in normalized space
		std::vector<float> normalized = suggestNext();

		// 2. Decode to raw hyperparameter values
		std::vector<float> raw = space_.decode(normalized);

		// 3. Clone the template network for this trial
		NNetwork* trialNet = templateNet.cloneForTrial();

		// 4. Apply hyperparameters to the cloned network's training config
		TrainingConfig trialConfig = space_.applyToConfig(raw, templateNet.getTrainingConfig());
		trialNet->setTrainingConfig(trialConfig);

		// 5. Set learning rate on all layers if "learningRate" is in the search space
		for (unsigned int i = 0; i < space_.params.size(); ++i)
		{
			if (space_.params[i].name == "learningRate")
			{
				const NNInfo* info = trialNet->getNNInfo();
				if (info)
				{
					int nHidden = info->numHiddenLayers();
					for (int li = 0; li <= nHidden; ++li)
					{
						const_cast<NNInfo*>(info)->setLearningRate(
							static_cast<unsigned int>(li), raw[i]);
					}
				}
				break;
			}
		}

		// 6. Set epoch limit via Terminator
		trialNet->getTerminatorMutable().setEpoch(static_cast<int64_t>(epochsPerTrial_));
		trialNet->getTerminatorMutable().setAccuracy(0);

		// 7. Train on training data
		TunerTrialCallback trainCb;
		trialNet->train(trainData, &trainCb);

		// 8. Evaluate on validation data
		float valLoss = trainCb.lastLoss;
		if (valData)
		{
			TunerTrialCallback valCb;
			trialNet->test(valData, &valCb);
			valLoss = valCb.lastLoss;
		}

		// 9. Report result
		reportResult(normalized, valLoss);

		// 10. Track best config
		if (valLoss < bestLoss)
		{
			bestLoss = valLoss;
			bestConfig = trialConfig;
			// Also store the learning rate in best config (for reference)
			for (unsigned int i = 0; i < space_.params.size(); ++i)
			{
				if (space_.params[i].name == "learningRate")
				{
					// Store in a comment-like way; the caller can inspect getBestParams()
					break;
				}
			}
		}

		delete trialNet;
	}

	return bestConfig;
}

int HyperparameterTuner::getMaxTrials() const
{
	return maxTrials_;
}

int HyperparameterTuner::getEpochsPerTrial() const
{
	return epochsPerTrial_;
}

float HyperparameterTuner::getBestScore() const
{
	return optimizer_.getBestScore();
}

const std::vector<float>& HyperparameterTuner::getBestParams() const
{
	return optimizer_.getBestParams();
}

const std::vector<HyperparameterTuner::Trial>& HyperparameterTuner::getTrials() const
{
	return trials_;
}

} // namespace glades
