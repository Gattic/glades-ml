// Centralized training configuration for NNetwork runs.
//
// This consolidates previously-scattered knobs (LR scheduling, grad clipping, TBPTT window,
// minibatch sizing) into a single struct that callers can treat as the "run config".
//
// Defaults preserve historical behavior:
// - LR schedule: none (multiplier == 1)
// - Grad-norm clipping: disabled
// - Per-element gradient clipping: enabled at 10 (used historically in all SGD paths)
// - TBPTT/minibatch overrides: disabled (use NNInfo)
#pragma once

#include <cmath>

namespace glades {

struct LearningRateScheduleConfig
{
	enum Type
	{
		NONE = 0,
		STEP = 1,
		EXP = 2,
		COSINE = 3
	};

	Type type;

	// STEP: multiplier = gamma ^ floor(t / stepSizeEpochs)
	int stepSizeEpochs;
	float gamma;

	// COSINE: multiplier = minMultiplier + 0.5*(1-minMultiplier)*(1+cos(pi*t/T))
	int cosineTMaxEpochs;
	float minMultiplier;

	LearningRateScheduleConfig()
	    : type(NONE),
	      stepSizeEpochs(0),
	      gamma(1.0f),
	      cosineTMaxEpochs(0),
	      minMultiplier(0.0f)
	{
	}

	inline void setNone()
	{
		type = NONE;
		stepSizeEpochs = 0;
		gamma = 1.0f;
		cosineTMaxEpochs = 0;
		minMultiplier = 0.0f;
	}

	inline void setStep(int stepSize, float g)
	{
		type = STEP;
		stepSizeEpochs = stepSize;
		gamma = g;
	}

	inline void setExp(float g)
	{
		type = EXP;
		stepSizeEpochs = 0;
		gamma = g;
	}

	inline void setCosine(int tMax, float minMult)
	{
		type = COSINE;
		cosineTMaxEpochs = tMax;
		minMultiplier = minMult;
	}

	inline float multiplier(int epochFromStart) const
	{
		if (epochFromStart < 0)
			epochFromStart = 0;

		switch (type)
		{
		case STEP:
		{
			if (stepSizeEpochs <= 0)
				return 1.0f;
			const int k = epochFromStart / stepSizeEpochs;
			if (k <= 0)
				return 1.0f;
			double m = 1.0;
			for (int i = 0; i < k; ++i)
				m *= static_cast<double>(gamma);
			return static_cast<float>(m);
		}
		case EXP:
		{
			if (epochFromStart <= 0)
				return 1.0f;
			double m = 1.0;
			for (int i = 0; i < epochFromStart; ++i)
				m *= static_cast<double>(gamma);
			return static_cast<float>(m);
		}
		case COSINE:
		{
			if (cosineTMaxEpochs <= 0)
				return 1.0f;
			const int t = (epochFromStart > cosineTMaxEpochs) ? cosineTMaxEpochs : epochFromStart;
			const double T = static_cast<double>(cosineTMaxEpochs);
			const double tt = static_cast<double>(t);
			const double minM = static_cast<double>(minMultiplier);
			const double cosv = cos(3.14159265358979323846 * (tt / T));
			const double m = minM + 0.5 * (1.0 - minM) * (1.0 + cosv);
			return static_cast<float>(m);
		}
		case NONE:
		default:
			return 1.0f;
		}
	}
};

struct TrainingConfig
{
	// If > 0, overrides NNInfo::batchSize for this run.
	// If <= 0, use NNInfo::batchSize.
	int minibatchSizeOverride;

	// If > 0, overrides NNInfo::TBPTTWindow for recurrent nets.
	// If <= 0, use NNInfo::TBPTTWindow.
	int tbpttWindowOverride;

	// Global grad-norm clipping (0 disables).
	float globalGradClipNorm;

	// Per-element gradient clipping (<= 0 disables).
	//
	// Historical engine behavior clipped many intermediate gradients/deltas to +/-10.
	// This remains enabled by default for backward compatibility, and is applied across
	// DFF/RNN/GRU/LSTM paths.
	float perElementGradClip;

	// Learning rate schedule multiplier configuration.
	LearningRateScheduleConfig lrSchedule;

	TrainingConfig()
	    : minibatchSizeOverride(0),
	      tbpttWindowOverride(0),
	      globalGradClipNorm(0.0f),
	      perElementGradClip(10.0f),
	      lrSchedule()
	{
	}
};

} // namespace glades

