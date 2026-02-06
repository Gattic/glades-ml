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
#include <stdint.h> // uint64_t (C++98-friendly)

namespace glades {

// Transformer-specific run configuration.
//
// NOTE:
// - The transformer architecture still derives dModel and nLayers from NNInfo hidden layer sizes/count.
// - These knobs control transformer behavior without overloading unrelated NNInfo fields.
// - Defaults preserve existing behavior in sgd_transformer.cpp (sinusoidal pos-enc, ReLU FFN, LN eps=1e-5).
struct TransformerRunConfig
{
	enum PositionalEncodingType
	{
		POSENC_NONE = 0,
		POSENC_SINUSOIDAL = 1,
		// Rotary positional embeddings (RoPE): applied to Q/K per head.
		POSENC_ROPE = 2
	};

	// Normalization type.
	enum NormType
	{
		NORM_LAYERNORM = 0,
		// RMSNorm (LLaMA-style): normalize by RMS instead of mean/variance.
		NORM_RMSNORM = 1
	};

	// Feed-forward kind.
	enum FFNKind
	{
		// Classic 2-layer MLP: W1 -> activation -> W2.
		FFN_MLP = 0,
		// SwiGLU: out = SiLU(gate) * up; then W2 projects back to dModel.
		// This uses a single packed W1 that outputs [gate, up] with width 2*dFF.
		FFN_SWIGLU = 1
	};

	enum FFNActivationType
	{
		FFN_RELU = 0,
		FFN_GELU = 1
	};

	// KV-cache storage dtype for token-LM incremental inference sessions.
	// This does not affect training; it controls memory/speed tradeoffs in KV sessions.
	//
	// Notes:
	// - FP16 halves KV-cache memory at a small numerical cost.
	// - BF16 halves KV-cache memory and can be faster than FP16 on CPU because
	//   BF16->FP32 conversion is a simple bit operation (no FP16 exponent handling).
	// - FP32 preserves historical behavior.
	enum KVCacheDType
	{
		KV_CACHE_F32 = 0,
		KV_CACHE_F16 = 1,
		KV_CACHE_BF16 = 2
	};

	// Overrides (<=0 => use built-in defaults).
	int nHeadsOverride;
	// Grouped-query attention: number of KV heads (<=0 => nHeads).
	// Must divide nHeads when enabled.
	int nKVHeadsOverride;
	int dFFOverride;

	// === Language model (token) mode ===
	//
	// If enabled, the transformer interprets each input row as a single token id (integer semantics),
	// uses an embedding table to map tokens -> dModel, and produces vocab logits with a softmax loss.
	//
	// Expected rows are interpreted as a single token id (next-token target). This avoids huge one-hot
	// expected vectors and requires special handling in Trainer/SGD.
	bool enableTokenEmbedding;
	// If > 0, defines the vocabulary size for the embedding + softmax head.
	// If <= 0, vocab size is derived from NNInfo output layer size.
	int vocabSizeOverride;
	// If true, tie embedding weights and LM head weights (E used for both input and output).
	bool tieEmbeddings;
	// Token id used as padding/ignore label (<= -1 disables ignore).
	// When enabled, training skips loss/grad for timesteps whose target == padTokenId.
	int padTokenId;

	// Token-LM loss mode.
	//
	// FULL_SOFTMAX computes exact softmax over the full vocab and reports true perplexity, but
	// requires O(T*vocab) memory and O(T*vocab*dModel) compute. This is not viable for real LLM
	// scales on the CPU reference backend.
	//
	// SAMPLED_SOFTMAX computes a sampled-softmax objective over {target + negatives}. This makes
	// training feasible at larger vocab sizes, but the loss is NOT exact NLL and perplexity is not
	// meaningful.
	enum TokenLMLossKind
	{
		TOKEN_LM_FULL_SOFTMAX = 0,
		TOKEN_LM_SAMPLED_SOFTMAX = 1
	};
	TokenLMLossKind tokenLmLossKind;
	// Number of negative samples per token when TOKEN_LM_SAMPLED_SOFTMAX is selected.
	// (Ignored for full softmax.)
	int tokenLmSampledNegatives;
	// If false (default), training will hard-fail when token-LM full softmax would require
	// unreasonably large allocations/compute. This is a deliberate "trainability triage"
	// guardrail to avoid pretending CPU full-softmax is an LLM training solution.
	bool tokenLmAllowHugeFullSoftmax;

	// LayerNorm epsilon.
	float layerNormEps;

	// Norm type (LayerNorm vs RMSNorm).
	NormType normType;

	// Positional encoding.
	PositionalEncodingType positionalEncoding;

	// KV-cache dtype for inference sessions (see KVCacheDType).
	KVCacheDType kvCacheDType;

	// RoPE parameters (used only when positionalEncoding==POSENC_ROPE).
	// If ropeDimOverride <= 0, use dHead (full head dim). Will be rounded down to even.
	int ropeDimOverride;
	// RoPE base theta (typical: 10000).
	float ropeTheta;

	// FFN kind (MLP vs SwiGLU).
	FFNKind ffnKind;

	// FFN activation (ReLU or GELU).
	// NOTE: Ignored when ffnKind==FFN_SWIGLU (SwiGLU uses SiLU).
	FFNActivationType ffnActivation;

	TransformerRunConfig()
	    : nHeadsOverride(0),
	      nKVHeadsOverride(0),
	      dFFOverride(0),
	      enableTokenEmbedding(false),
	      vocabSizeOverride(0),
	      tieEmbeddings(true),
	      padTokenId(-1),
	      tokenLmLossKind(TOKEN_LM_FULL_SOFTMAX),
	      tokenLmSampledNegatives(64),
	      tokenLmAllowHugeFullSoftmax(false),
	      layerNormEps(1e-5f),
	      normType(NORM_LAYERNORM),
	      positionalEncoding(POSENC_SINUSOIDAL),
	      kvCacheDType(KV_CACHE_F32),
	      ropeDimOverride(0),
	      ropeTheta(10000.0f),
	      ffnKind(FFN_MLP),
	      ffnActivation(FFN_RELU)
	{
	}
};

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
			// Previously implemented as an O(k) loop; use pow() for O(1).
			// Preserve behavior for negative gamma as well (pow handles sign for integer exponents).
			const double m = pow(static_cast<double>(gamma), static_cast<double>(k));
			return static_cast<float>(m);
		}
		case EXP:
		{
			if (epochFromStart <= 0)
				return 1.0f;
			// Previously implemented as an O(epochFromStart) loop; use pow() for O(1).
			const double m = pow(static_cast<double>(gamma), static_cast<double>(epochFromStart));
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

// Optimizer configuration.
//
// NOTE:
// - Default preserves historical behavior (SGD with momentum from NNInfo).
// - For transformers, ADAMW is strongly recommended.
struct OptimizerConfig
{
	enum Type
	{
		SGD_MOMENTUM = 0,
		ADAMW = 1
	};

	Type type;

	// AdamW parameters (used when type==ADAMW).
	float adamBeta1;
	float adamBeta2;
	float adamEps;
	bool adamBiasCorrection;

	OptimizerConfig()
	    : type(SGD_MOMENTUM),
	      adamBeta1(0.9f),
	      adamBeta2(0.999f),
	      adamEps(1e-8f),
	      adamBiasCorrection(true)
	{
	}
};

// Mixed precision configuration (primarily for Transformer training).
//
// Design:
// - Keep FP32 "master" weights as the single source of truth for updates/serialization.
// - Maintain optional low-precision (FP16/BF16) copies of weight matrices used by forward/backward.
// - Use loss scaling to avoid FP16/BF16 gradient underflow when low-precision compute is used.
struct MixedPrecisionConfig
{
	enum WeightDType
	{
		WEIGHT_F32 = 0,
		WEIGHT_F16 = 1,
		WEIGHT_BF16 = 2
	};

	// If false, training runs in FP32 only (historical behavior).
	bool enable;
	// Low-precision dtype for weight copies used in forward/backward.
	WeightDType weightDType;

	// Loss scaling:
	// - If enable==true and useLossScaling==true, backprop deltas are multiplied by lossScale
	//   and the optimizer divides gradients by lossScale before applying updates.
	// - If dynamicLossScaling==true, the engine will back off on NaN/Inf and grow lossScale
	//   after `growthInterval` successful steps.
	bool useLossScaling;
	bool dynamicLossScaling;
	float lossScaleInit;
	float lossScaleMin;
	float lossScaleMax;
	int growthInterval;
	float growthFactor;
	float backoffFactor;

	MixedPrecisionConfig()
	    : enable(false),
	      weightDType(WEIGHT_F16),
	      useLossScaling(true),
	      dynamicLossScaling(true),
	      lossScaleInit(1024.0f),
	      lossScaleMin(1.0f),
	      lossScaleMax(65536.0f),
	      growthInterval(2000),
	      growthFactor(2.0f),
	      backoffFactor(0.5f)
	{
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

	// Optimizer configuration.
	OptimizerConfig optimizer;

	// Learning rate schedule multiplier configuration.
	LearningRateScheduleConfig lrSchedule;

	// Transformer run config (used only for transformer net types).
	TransformerRunConfig transformer;

	// Mixed precision (used primarily for transformer net types).
	MixedPrecisionConfig mixedPrecision;

	TrainingConfig()
	    : minibatchSizeOverride(0),
	      tbpttWindowOverride(0),
	      globalGradClipNorm(0.0f),
	      perElementGradClip(10.0f),
	      optimizer(),
	      lrSchedule(),
	      transformer(),
	      mixedPrecision()
	{
	}
};

} // namespace glades

