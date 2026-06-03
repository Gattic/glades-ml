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
#include <vector>

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
	// Benchmark-only hook: capture extra optimizer-gap diagnostics such as grouped
	// update norms, head share, and token-margin summaries. This is intentionally
	// off by default because it may snapshot pre/post weights around apply steps.
	bool captureOptimizerGapDiagnostics;

	// LayerNorm epsilon.
	float layerNormEps;

	// Norm type (LayerNorm vs RMSNorm).
	NormType normType;

	// Positional encoding.
	PositionalEncodingType positionalEncoding;

	// KV-cache dtype for inference sessions (see KVCacheDType).
	KVCacheDType kvCacheDType;
	// Optional hard cap for KV-session allocations in bytes.
	// 0 => use the environment/default policy.
	uint64_t kvSessionMaxBytes;
	// Optional hard cap for serving logits scratch/storage in bytes.
	// 0 => use the environment/default policy.
	uint64_t serveLogitsMaxBytes;

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

	// Dropout rates (training only; disabled by default).
	// Embedding dropout: applied after positional encoding, before the first block.
	float embeddingDropoutRate;
	// Residual dropout: applied to attention output and FFN output before residual adds.
	float residualDropoutRate;

	// Local-window attention (paradigm shift #6 extension for main transformer).
	// If > 0, each query attends to ±localAttnWindow tokens (plus BOS), giving
	// O(T·W) compute instead of O(T²).  If 0, full attention.  Requires the
	// flash path; silently ignored on the non-flash path.
	// Validated via CHIRONLocalAttentionFullWindowParityTest + chiron_train
	// `--local-attn` benchmarks (5.2-38.1× speedup at T=2k-16k).
	int localAttnWindow;

	// Attention-sink count (paradigm shift #78 ATTENTION-SINK-DISTILL-CHIRON).
	// If > 0, the first `attnSinkCount` positions are always allowed in
	// attention regardless of sliding-window restriction. Composes with
	// localAttnWindow (W) to give StreamingLLM-style infinite-context
	// attention: keys are allowed iff (u < S) OR (u + W > t), with causal
	// restriction u <= t. KV cache is bounded at S+W positions independent
	// of total context length T. Default 0 = disabled.
	// Reference: Xiao et al. 2023 "Efficient Streaming Language Models with
	// Attention Sinks". Production-validated by vLLM, lmdeploy, llama.cpp,
	// MLC-LLM, TGI.
	int attnSinkCount;

	// Binary FFN forward (paradigm shift #74 PHOENIX-1BIT-DISTILL-COMBO).
	// When true, the FFN W2 (output) projection is computed as Y = X @
	// sign(W2).T via an on-the-fly binary kernel — no multiplies on the
	// hot path. Float master weights are still kept for backward (STE)
	// and Adam updates; the speedup is FLOP-side only at training time.
	// At inference the float weights can be dropped. Default false.
	bool binaryFFN;

	// MLA latent rank (paradigm shift #76 MLA-DISTILL-CHIRON). When > 0,
	// switch the K/V projections from standard MHA W_K, W_V to a low-rank
	// latent: K = h @ W_DKV @ W_UK ; V = h @ W_DKV @ W_UV with latent
	// dimension d_c = mlaLatentDim. Cache stores c (T × d_c) instead of
	// K, V (T × n_heads × d_kv × 2). Default 0 = standard MHA.
	int mlaLatentDim;

	// FACE Adafactor on the token embedding (paradigm shift #28).  When true,
	// the tokE [V × dModel] table is updated via a frequency-debiased,
	// row/col-normalized preconditioner instead of dense Adam.  State drops
	// from ~8·V·dModel bytes (FP32 m+v) to 4·(V + dModel + 2) bytes
	// (zn̄, dn̄, q̂, gF̄ all FP32 scalars/vectors) — typically ~250-1000×
	// compression on the embedding optimizer state.  Default false.
	bool faceEmbedding;
	// FACE EMA decays.  betaRow=0.98 keeps inactive (rare) tokens'
	// per-row stats stable; betaCol=0.95 lets column/scalar EMAs adapt
	// faster.  Defaults match CHIRON's FACE preset.
	float faceBetaRow;
	float faceBetaCol;
	// Numerical floor on the FACE preconditioner denominator.  Default
	// 1e-8 (matches paradigm-28 design).
	float faceEps;

	// Paradigm shift #39 RLG (Reversible Layer Growth) — port from CHIRON.
	// When > 0 and < nLayers, the trainer initializes layers
	// [rlgInitialLayers, nLayers) with Wo = W2 = bo = b2 = 0, making each
	// such block bit-exact identity to the residual stream at step 0.
	// The forward pass through these layers contributes zero; gradient
	// flow at init is therefore equivalent to a smaller L = rlgInitialLayers
	// model, sidestepping the depth-amplified gradient variance that
	// breaks deep transformers at 1.84B. The optimizer naturally grows
	// the inactive layers as ∂loss/∂Wo is non-zero whenever the residual
	// has non-zero norm. Default 0 = disabled (all layers Glorot-init).
	int rlgInitialLayers;

	// Z-loss auxiliary objective (paradigm shift: PaLM/T5/Gemini-style logit
	// regularization). When > 0, adds zlossCoef * mean_t(log²(Z_t)) to the
	// readout loss, where Z_t = sum_v exp(logit_{t,v}). Default 0.0f =
	// disabled, math bit-identical to baseline. Recommended: 1e-4 (PaLM
	// default). Improves training stability under FP8 readout by bounding
	// logit magnitudes.
	float zlossCoef;

	// QK-Norm (paradigm shift: DeepSeek-V3 / modern Llama). When true, Q and
	// K are L2-normalized per-head before the attention dot product, and the
	// constant `1/sqrt(dHead)` scale is replaced by a learnable per-head
	// scalar γ. Default false = disabled, math bit-identical to baseline.
	bool qkNormEnabled;

	// Initial value for the QK-Norm γ scalar (per-head, shared across all
	// blocks at init). When <= 0 (default), initialized to log2(T) at first
	// forward pass per DeepSeek-V3 init. Set explicitly to override.
	float qkNormGammaInit;

	// Multi-token prediction depth (paradigm shift: DeepSeek-V3). When > 0,
	// adds N auxiliary heads each predicting the token at offset +k for
	// k in {2, ..., N+1}. The current implementation supports depth = 1 (a
	// single +2-offset head). Default 0 = disabled, math bit-identical.
	int mtpDepth;

	// MTP auxiliary loss coefficient. Each MTP head's CE is weighted by
	// (mtpCoef / mtpDepth) and added to the main CE loss. DeepSeek-V3 uses
	// 0.1 after a brief warmup at 0.3. Default 0.1f.
	float mtpCoef;

	// LayerDrop / stochastic depth (paradigm shift: Fan 2019 / Huang 2016 /
	// timm). When > 0, each transformer block l ∈ {0..L-1} is dropped with
	// probability p_l. Linear-rising schedule: p_l = (l/(L-1)) · layerDropPMax,
	// so layer 0 never drops and the deepest layer drops with probability
	// layerDropPMax. Kept layers' sub-residuals are scaled by 1/(1-p_l)
	// (inverted-dropout convention). Default 0.0f = disabled, math
	// bit-identical to baseline. Recommended for CHIRON 1B: 0.1.
	float layerDropPMax;

	// LayerDrop schedule type. true = linear-rising (p_l = (l/(L-1)) · pMax,
	// timm convention). false = constant (p_l = pMax for all l). Default
	// true. The constant variant is reserved for a possible future arc; this
	// arc uses linear-rising exclusively.
	bool layerDropLinearSchedule;

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
	      captureOptimizerGapDiagnostics(false),
	      layerNormEps(1e-5f),
	      normType(NORM_LAYERNORM),
	      positionalEncoding(POSENC_SINUSOIDAL),
	      kvCacheDType(KV_CACHE_F32),
	      kvSessionMaxBytes(0ULL),
	      serveLogitsMaxBytes(0ULL),
	      ropeDimOverride(0),
	      ropeTheta(10000.0f),
	      ffnKind(FFN_MLP),
	      ffnActivation(FFN_RELU),
	      embeddingDropoutRate(0.0f),
	      residualDropoutRate(0.0f),
	      localAttnWindow(0),
	      attnSinkCount(0),
	      binaryFFN(false),
	      mlaLatentDim(0),
	      faceEmbedding(false),
	      faceBetaRow(0.98f),
	      faceBetaCol(0.95f),
	      faceEps(1e-8f),
	      rlgInitialLayers(0),
	      zlossCoef(0.0f),
	      qkNormEnabled(false),
	      qkNormGammaInit(0.0f),
	      mtpDepth(0),
	      mtpCoef(0.1f),
	      layerDropPMax(0.0f),
	      layerDropLinearSchedule(true)
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
		COSINE = 3,
		BAYESIAN = 4
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

	// Step-level multiplier: progress is a float in [0, 1] representing
	// fractional progress through total training (e.g. step/totalSteps).
	inline float multiplierSmooth(float progress) const
	{
		if (progress < 0.0f) progress = 0.0f;
		if (progress > 1.0f) progress = 1.0f;

		switch (type)
		{
		case COSINE:
		{
			const double minM = static_cast<double>(minMultiplier);
			const double cosv = cos(3.14159265358979323846 * static_cast<double>(progress));
			return static_cast<float>(minM + 0.5 * (1.0 - minM) * (1.0 + cosv));
		}
		case NONE:
		default:
			return 1.0f;
		}
	}

	// Fractional-epoch multiplier: epochProgressFromStart is measured in epochs
	// from the start of training and may include intra-epoch progress.
	inline float multiplierFractionalEpoch(double epochProgressFromStart) const
	{
		if (epochProgressFromStart < 0.0)
			epochProgressFromStart = 0.0;

		switch (type)
		{
		case COSINE:
		{
			if (cosineTMaxEpochs <= 0)
				return 1.0f;
			const double T = static_cast<double>(cosineTMaxEpochs);
			double t = epochProgressFromStart;
			if (t > T)
				t = T;
			const double minM = static_cast<double>(minMultiplier);
			const double cosv = cos(3.14159265358979323846 * (t / T));
			const double m = minM + 0.5 * (1.0 - minM) * (1.0 + cosv);
			return static_cast<float>(m);
		}
		case STEP:
		case EXP:
		case BAYESIAN:
		case NONE:
		default:
			return multiplier(static_cast<int>(epochProgressFromStart));
		}
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
		case BAYESIAN:
			return 1.0f;
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
struct BayesianLRConfig
{
	int windowEpochs;   // evaluate every N epochs (default 10)
	float minLR;        // lower bound for LR multiplier (default 1e-6)
	float maxLR;        // upper bound for LR multiplier (default 0.1)

	BayesianLRConfig()
	    : windowEpochs(10),
	      minLR(1e-6f),
	      maxLR(0.1f)
	{
	}
};

struct OptimizerConfig
{
	enum Type
	{
		SGD_MOMENTUM = 0,
		ADAMW = 1,
		ATLAS = 2,
		VESTA = 3,
		HELIOS = 4,
		// Paradigm shift #55 SOPHIA-G (Liu et al. 2023, gradient-squared variant).
		// Drop-in Adam variant with clipped second-order update; published
		// 1.5-2× steps reduction to fixed final NLL.
		SOPHIA_G = 5
	};

	Type type;

	// AdamW parameters (used when type==ADAMW or SOPHIA_G inherits beta1/beta2).
	float adamBeta1;
	float adamBeta2;
	float adamEps;
	bool adamBiasCorrection;

	// SOPHIA_G parameters (Liu et al. 2023 defaults: gamma=0.05, rho=1.0,
	// beta1=0.965, beta2=0.99 — the latter two override adamBeta1/adamBeta2
	// when type==SOPHIA_G).
	float sophiaGamma;   // denominator scale on Hessian proxy
	float sophiaRho;     // update clip magnitude

	// Enable groupwise AdamW modulation. This keeps the exact AdamW update law
	// but multiplies each parameter group's effective step size by a cheap
	// scalar derived from group momentum stability, SNR, and update-to-weight
	// ratio. When false, the optimizer is exact AdamW.
	bool adamGroupwiseEnabled;

	// Positive multiplier on the groupwise stability/coherence signal.
	float adamGroupStabilityScale;

	// Positive multiplier on the groupwise signal-to-noise statistic.
	float adamGroupSnrScale;

	// Positive multiplier on the update-to-weight penalty term.
	float adamGroupRatioScale;

	// Clamp range for the final groupwise step multiplier.
	float adamGroupMinScale;
	float adamGroupMaxScale;

	// Minimum group size required before the groupwise multiplier is allowed to
	// differ from 1. Small groups (biases, norms, tiny vectors) stay on exact
	// AdamW.
	unsigned int adamGroupMinSize;

	OptimizerConfig()
	    : type(SGD_MOMENTUM),
	      adamBeta1(0.9f),
	      adamBeta2(0.999f),
	      adamEps(1e-8f),
	      adamBiasCorrection(true),
	      sophiaGamma(0.05f),
	      // Sophia paper (Liu 2023) uses rho=0.04 for LLM pre-training.
	      // ρ=1.0 (initially set per the design doc) caused 14e-2 nat
	      // divergence at 250 steps in 213M smoke — too aggressive.
	      sophiaRho(0.04f),
	      adamGroupwiseEnabled(false),
	      adamGroupStabilityScale(0.05f),
	      adamGroupSnrScale(0.05f),
	      adamGroupRatioScale(0.50f),
	      adamGroupMinScale(0.90f),
	      adamGroupMaxScale(1.15f),
	      adamGroupMinSize(256u)
	{
	}
};

// ATLAS optimizer configuration (BRSP variant).
//
// ATLAS (Adaptive Temporally-Predictive Learning in Active Subspaces) with
// Baseline-Regularized Subspace Preconditioning (BRSP):
// - Fisher-diagonal preconditioning in a low-rank subspace
// - Data-driven complement closure (sigma2) derived from the same normalized
//   covariance operator as the active Fisher statistics
// - EMA-blended subspace refresh with Fisher transform (no catastrophic resets)
// - Optional Predictive Natural Gradient (PNG) temporal extrapolation
struct ATLASConfig
{
	// Subspace rank per weight matrix.
	// The actual rank is clamped to min(rank, min(m, n)) for each weight matrix.
	// Larger rank captures more curvature information at higher compute/memory cost.
	unsigned int rank;

	// Residual complement-block rank cap. 0 disables the anisotropic complement
	// path; positive values allocate up to this many dense residual modes on
	// eligible hidden FC-style layers, with an online active rank selected
	// inside the cap and an isotropic tail closure on the remaining complement.
	unsigned int complementRank;

	// Relative learning-rate scale applied only to the anisotropic complement
	// sector update. Values below 1 damp the residual-sector correction so it
	// does not inherit the more aggressive nominal lr used for the active space.
	float complementLrScale;

	// Maximum ratio of the effective complement-sector rate to the sector's own
	// nominal lr. This caps complementRate =
	// (complementLrScale * lr) / (sectorFisher + eps) at
	// complementKappaMax * complementLrScale * lr.
	float complementKappaMax;

	// Enable the PRISM-style active-memory / predictive-edge prototype.
	// When false, ATLAS uses the historical active/complement logic unchanged.
	bool prismEnabled;

	// Number of lagged compressed gradients used by the PRISM active-memory
	// correction. The current prototype supports up to 2 lags.
	unsigned int prismLagHorizon;

	// Strength of the active-space memory correction. Larger values apply more
	// unresolved-bulk friction to the PNG-style extrapolated active update.
	float prismMemoryScale;

		// Minimum normalized predictive-edge score required before the complement
		// controller is allowed to activate residual complement modes.
		float prismPredictiveEdgeThreshold;

		// Enable the RESOLVE prototype: a lagged transfer-edge gate plus a stable
		// active-space memory kernel fit from compressed-gradient history.
		bool resolveEnabled;

		// Number of lagged compressed-gradient slices used by the RESOLVE
		// prototype. The current implementation supports up to 4 lags.
		unsigned int resolveLagHorizon;

		// Strength of the RESOLVE active-space memory kernel.
		float resolveMemoryScale;

	// Minimum normalized transfer-edge score required before the adaptive
	// complement controller is allowed to retain explicit residual modes.
	float resolvePredictiveEdgeThreshold;

	// Enable the HERO prototype: a Hankel-edge gate on residual complement
	// activation plus a memory-only fallback on the active coordinates.
	bool heroEnabled;

	// Number of lagged active/scout history slices used by the HERO Hankel
	// sketch. The current implementation supports up to 4 lags.
	unsigned int heroLagHorizon;

	// Strength of the HERO active-space memory kernel.
	float heroMemoryScale;

	// Minimum normalized Hankel-edge score required before the adaptive
	// complement controller is allowed to retain explicit residual modes.
	float heroEdgeThreshold;

	// Enable the COBALT prototype: a stacked active/scout transfer-edge gate
	// plus a transfer-weighted active-memory kernel. This is a CPU-side minimal
	// prototype of the broader cross-layer transfer idea; the current
	// implementation stays layer-local and reuses the existing compressed
	// active/scout histories.
	bool cobaltEnabled;

	// Number of lagged active/scout history slices used by the COBALT transfer
	// sketch. The current implementation supports up to 4 lags.
	unsigned int cobaltLagHorizon;

	// Strength of the COBALT active-space memory kernel. The effective memory
	// gain is further scaled by the observed transfer singular value so weak
	// transfer regimes fall back toward plain ATLAS.
	float cobaltMemoryScale;

	// Minimum normalized transfer-edge score required before the adaptive
	// complement controller is allowed to retain explicit residual modes.
	float cobaltEdgeThreshold;

	// Enable the BIRCH prototype: a local Hankel-style transfer gate on stacked
	// active/scout histories plus a memory-only fallback on the active
	// coordinates. This is the minimal ATLAS-side approximation of the broader
	// balanced transfer idea; explicit complement reactivation stays disabled.
	bool birchEnabled;

	// Number of lagged past slices used in the BIRCH past-state sketch. The
	// current implementation supports up to 3 past slices in addition to the
	// current and one previous future slice.
	unsigned int birchPastHorizon;

	// Number of active future slices used in the BIRCH Hankel sketch. The
	// current implementation supports up to 2 future slices (current + lag1).
	unsigned int birchFutureHorizon;

	// Strength of the BIRCH active-space memory kernel. The effective gain is
	// further gated by the supercritical part of the Hankel transfer score so
	// weak transfer regimes fall back toward plain ATLAS.
	float birchMemoryScale;

	// Minimum normalized Hankel transfer-edge score required before the BIRCH
	// memory fallback activates. Explicit complement modes remain disabled in
	// this minimal prototype even when the edge is supercritical.
	float birchEdgeThreshold;

	// Enable the GHOST prototype: approximate gauge-horizontal projection in the
	// compressed ATLAS state plus a rank-1 biorthogonal transfer mode extracted
	// from lagged active/scout history. The minimal prototype is memory-only and
	// keeps explicit complement activation disabled.
	bool ghostEnabled;

	// Number of lagged active/scout history slices used by the GHOST transfer
	// sketch. The current implementation supports up to 4 lagged slices.
	unsigned int ghostLagHorizon;

	// Strength of the GHOST active-space memory correction after quotient-style
	// horizontalization and balanced transfer weighting.
	float ghostMemoryScale;

	// Minimum normalized GHOST transfer-edge score required before the
	// memory-only correction activates.
	float ghostEdgeThreshold;

	// Enable SPARROW: a cheaper streaming quotient-transfer observer that keeps
	// the GHOST lesson (horizontalization + biorthogonal transfer) but replaces
	// lag-stack operator extraction with a rank-1 streaming latent memory model.
	bool sparrowEnabled;

	// Number of retained SPARROW streaming transfer modes. The current
	// implementation supports rank-1 and rank-2 observers.
	unsigned int sparrowModeRank;

	// When enabled, sparrowModeRank is treated as a cap rather than an exact
	// retained rank. Higher modes are only activated if their edge is strong
	// enough relative to the leading mode.
	bool sparrowAutoModeGate;

	// Strength of the SPARROW active-space memory correction.
	float sparrowMemoryScale;

	// Minimum SPARROW canonical-edge score required before the memory-only
	// correction activates.
	float sparrowEdgeThreshold;

	// Minimum raw mode-2 SPARROW edge required before the second streaming mode
	// is allowed to contribute when auto-gating is enabled.
	float sparrowSecondEdgeThreshold;

	// Minimum mode-2 / mode-1 SPARROW edge ratio required before the second
	// streaming mode is allowed to contribute when auto-gating is enabled.
	float sparrowSecondEdgeFraction;

	// Stability clamp for the SPARROW latent memory pole. The pole is projected
	// to [-sparrowPoleMax, sparrowPoleMax] each update.
	float sparrowPoleMax;

	// Enable QBRT: a quotient-balanced transfer controller that reuses the
	// existing active/scout lag histories, extracts a balanced rank-1 mode, and
	// applies only a memory correction in active coordinates.
	bool qbrtEnabled;

	// Number of lagged active/scout history slices used by the QBRT balanced
	// transfer sketch. The current implementation supports up to 4 lagged
	// slices, matching the existing GHOST/HERO history budget.
	unsigned int qbrtLagHorizon;

	// Strength of the QBRT active-space memory correction.
	float qbrtMemoryScale;

	// Minimum normalized QBRT transfer-edge score required before the
	// memory-only correction activates.
	float qbrtEdgeThreshold;

	// Stability clamp for the QBRT latent memory pole. The pole is projected to
	// [-qbrtPoleMax, qbrtPoleMax] each update.
	float qbrtPoleMax;

	// Enable QRC: a quotient resolvent controller that fits a tiny reduced
	// active/scout plant on compressed history and applies only a memory/control
	// correction in active coordinates.
	bool qrcEnabled;

	// Number of lagged active/scout history slices used by the QRC reduced-plant
	// fit. The current implementation supports up to 4 lagged slices.
	unsigned int qrcLagHorizon;

	// Strength of the QRC active-space control correction.
	float qrcMemoryScale;

	// Minimum normalized QRC closed-loop edge score required before the
	// controller activates.
	float qrcEdgeThreshold;

	// Stability clamp for the QRC latent plant pole. The pole is projected to
	// [-qrcPoleMax, qrcPoleMax] each update.
	float qrcPoleMax;

	// Enable RIFT: a quotient-horizontal path-signature memory correction that
	// uses low-order active/scout path features instead of explicit complement
	// geometry. The minimal prototype is CPU-side, memory-only, and keeps
	// explicit complement activation disabled.
	bool riftEnabled;

	// Number of lagged active/scout history slices used to build the RIFT path
	// segment. The current implementation supports up to 4 lagged slices.
	unsigned int riftLagHorizon;

	// Strength of the RIFT active-space memory correction.
	float riftMemoryScale;

	// Minimum RIFT path-edge score required before the memory-only correction
	// activates.
	float riftEdgeThreshold;

	// Stability clamp for the RIFT latent memory pole. The pole is projected to
	// [-riftPoleMax, riftPoleMax] each update.
	float riftPoleMax;

	// Enable ORBIT-Lite: an output-head-only function-space memory correction
	// that approximates quotient output modes directly in classifier row-space
	// instead of modeling parameter-space complement geometry.
	bool orbitEnabled;

	// Strength of the ORBIT-Lite active-space memory correction.
	float orbitMemoryScale;

	// Minimum normalized ORBIT functional-edge score required before the
	// memory-only correction activates.
	float orbitEdgeThreshold;

	// Stability clamp for the ORBIT-Lite latent memory pole. The pole is
	// projected to [-orbitPoleMax, orbitPoleMax] each update.
	float orbitPoleMax;

	// Enable HELM: a hidden/output transfer observer applied only on the DFF
	// output head. The current prototype is CPU-side, memory-style, and leaves
	// explicit complement modeling disabled.
	bool helmEnabled;

	// Strength of the HELM output-head memory correction.
	float helmMemoryScale;

	// Minimum HELM transfer-edge score required before the output-head
	// correction activates.
	float helmEdgeThreshold;

	// Maximum retained HELM transfer modes. Values above 1 enable a small
	// multi-mode hidden-to-output observer instead of the original rank-1 probe.
	unsigned int helmModeRank;

	// Number of trailing hidden layers to stack into the HELM observable.
	// 1 reproduces the original last-hidden-only probe; 2 enables HELM-v2.
	unsigned int helmHiddenStackDepth;

	// Stability clamp for the HELM latent memory pole. The pole is
	// projected to [-helmPoleMax, helmPoleMax] each update.
	float helmPoleMax;

	// Enable ASTER: a reduced output-space innovation state-space observer that
	// uses transported hidden controls and output innovations instead of
	// parameter-space observables. The minimal prototype is DFF-only,
	// output-head-only, and keeps explicit complement modeling disabled.
	bool asterEnabled;

	// Enable AEGIS: an evidence-gated fusion branch that keeps the ATLAS-BSRP
	// spatial base, enables SPARROW and ASTER together, and attenuates the
	// output-space correction when predictive parameter-space evidence is
	// already stronger on the current model state.
	bool aegisEnabled;

	// Enable CITADEL: a contextual trust-region refinement of AEGIS that
	// downweights residual channels in disagreement-heavy or context-hard
	// regimes and pushes the update back toward the spatial ATLAS base.
	bool citadelEnabled;

	// Enable RAMPART: a posterior-style residual fusion pass on top of the
	// existing AEGIS signals. The minimal prototype keeps the current ATLAS
	// backbone, infers a small three-channel posterior over
	// {spatial, predictive, output}, and projects the non-spatial correction
	// into an explicit residual budget.
	bool rampartEnabled;

	// Enable MERIT: a geometry-aware residual fusion branch that keeps the
	// current ATLAS/BSRP backbone, treats spatial structure as geometry instead
	// of a competing residual sensor, and solves only over predictive/output
	// residual evidence inside an explicit trust budget.
	bool meritEnabled;

	// Enable STRATA: a sparse mode-selection controller around the existing
	// AdamW/ATLAS backbone that chooses among {null, predictive, output,
	// coupled} residual modes instead of densely blending all channels every
	// step. The minimal prototype reuses SPARROW and ASTER as the actuators and
	// only treats ATLAS/BSRP as mode-conditional geometry.
	bool strataEnabled;

	// Enable AURORA: a receding-horizon residual controller that forecasts
	// predictive/output evidence one small step forward before solving a bounded
	// two-sensor posterior around the existing ATLAS base.
	bool auroraEnabled;

	// Enable SEAM: a mirror-descent simplex controller over
	// {spatial, predictive, output} coordinates. The current prototype updates a
	// small coordinate system from delayed evidence instead of solving a dense
	// posterior every boundary.
	bool seamEnabled;

	// Enable QUASAR: an entropy-regularized residual controller that maintains a
	// soft distribution over {null, predictive, output, coupled} residual modes
	// rather than selecting one mode hard.
	bool quasarEnabled;

	// Strength of the ASTER output-head memory / innovation correction.
	float asterMemoryScale;

	// Minimum ASTER transfer-edge score required before the output-head
	// correction activates.
	float asterEdgeThreshold;

	// Maximum retained ASTER transfer modes. The current prototype supports a
	// small rank-1 or rank-2 output-space realization.
	unsigned int asterStateRank;

	// Number of trailing hidden layers to transport into ASTER output-space
	// controls.
	unsigned int asterHiddenStackDepth;

	// Stability clamp for the ASTER latent pole surrogate. The pole is
	// projected to [-asterPoleMax, asterPoleMax] each update.
	float asterPoleMax;

	// Enable KAPPA: a transformer-only retrieval-state observable that augments
	// ASTER/AEGIS with compressed lagged KV summaries from the last decoder
	// blocks. The minimal prototype is head-only and only affects token-LM runs.
	bool kappaEnabled;

	// Number of query heads per tracked decoder block that contribute to the
	// compressed KAPPA retrieval observable.
	unsigned int kappaHeads;

	// Number of lag buckets used when compressing retrieval summaries. The
	// current implementation supports up to 4 buckets.
	unsigned int kappaLagBuckets;

	// Number of projected value channels kept per head/lag KAPPA observable.
	unsigned int kappaRank;

	// Forecast blending coefficient used by AURORA when combining filtered and
	// current predictive/output evidence into a short-horizon residual proposal.
	float auroraHorizonBlend;

	// Maximum D-metric residual budget for the AURORA predictive/output solve.
	float auroraBudgetMax;

	// When true, AURORA keeps its predictive/output controller but applies the
	// resulting gradients through an AdamW backbone instead of the ATLAS/BSRP
	// weight update. This is transformer-only in the current prototype.
	bool auroraAdamwBackbone;

	// Multiplicative gain applied to AURORA's head-local output correction when
	// retrieval and margin signals indicate the current residual should act more
	// like a head-dominant transformer adjustment.
	float auroraHeadGain;

	// Retention factor applied to non-head SPARROW trust inside AURORA once the
	// controller decides to bias the step toward the token head. Values in
	// [0, 1] keep the body closer to the ATLAS base while the head absorbs more
	// of the residual budget.
	float auroraBodyTrustScale;

	// Enable GEODE: a transformer-only Adam-style optimizer that reuses ATLAS
	// active subspace tracking as a low-rank geometry field and optionally
	// blends a small SPARROW-style predictive correction inside that geometry.
	bool geodeEnabled;

	// Multiplicative strength of the ATLAS low-rank geometry term inside the
	// GEODE Woodbury solve. 0 reduces GEODE to its diagonal Adam-style limit.
	float geodeGeometryScale;

	// Strength of the retained SPARROW active-mode correction when GEODE blends
	// a small predictive adjustment into the active coordinates before solving
	// the low-rank preconditioned step.
	float geodePredictiveScale;

	// Enable ECHO: Epilogue Curvature Harvesting Optimizer. ECHO keeps the
	// exact AdamW backbone but applies a separable row/column metric estimated
	// directly from backward operands rather than from post-hoc gradient passes.
	bool echoEnabled;

	// Strength of ECHO's two-sided diagonal geometry. 0 reduces ECHO exactly to
	// the Adam-style diagonal backbone.
	float echoGeometryScale;

	// Final ECHO geometry strength reached after the decay schedule completes.
	// Equal to echoGeometryScale when no schedule is active.
	float echoGeometryScaleFinal;

	// Number of optimizer steps over which ECHO linearly decays from
	// echoGeometryScale to echoGeometryScaleFinal. 0 disables scheduling.
	unsigned int echoGeometryDecaySteps;

	// Number of optimizer steps between ECHO metric refreshes. 1 refreshes on
	// every step. Larger values reuse the previously prepared row/column metric
	// vectors on skipped steps.
	unsigned int echoMetricCadence;

	// Optional scalar trust gate applied to ECHO geometry. 0 keeps the current
	// always-on ECHO metric. Positive values scale geometry down when observed
	// anisotropy is weak or unstable across steps.
	float echoTrustScale;

	// Optional bounded one-step predictive blend applied to the Adam first
	// moment before ECHO's row/column metric is applied. 0 disables the blend.
	float echoPredictiveScale;

	// Optional grouped structural factor strength. Positive values multiply a
	// coarse contiguous row/column chunk factor into the existing ECHO metric,
	// approximating a fixed-basis structural prior without low-rank solves.
	float echoStructuralScale;

	// Number of contiguous row/column groups used by the structural factor. 1
	// disables grouping.
	unsigned int echoStructuralGroups;

	// Scope of ECHO matrix activation:
	// 0 = all eligible matrices,
	// 1 = large-only (matrix area >= internal cutoff),
	// 2 = late-head (last decoder block plus head/output),
	// 3 = late-head-large (late-head restricted to large matrices).
	enum
	{
		ECHO_SCOPE_ALL = 0u,
		ECHO_SCOPE_LARGE_ONLY = 1u,
		ECHO_SCOPE_LATE_HEAD = 2u,
		ECHO_SCOPE_LATE_HEAD_LARGE = 3u
	};
	unsigned int echoScope;

	// Enable BiMAP: a transformer-only blockwise matrix preconditioner that
	// keeps Adam-style moments but scales matrix updates through row/column
	// second-moment factors instead of a low-rank residual solve.
	bool bimapEnabled;

	// Scope of BiMAP matrix promotion:
	// 0 = all eligible matrices,
	// 1 = head-only (tied token embedding / output projection),
	// 2 = late-only (last decoder block matrices),
	// 3 = late-head (last decoder block plus head/output).
	enum
	{
		BIMAP_SCOPE_ALL = 0u,
		BIMAP_SCOPE_HEAD_ONLY = 1u,
		BIMAP_SCOPE_LATE_ONLY = 2u,
		BIMAP_SCOPE_LATE_HEAD = 3u
	};
	unsigned int bimapScope;

	// Enable BiMAP-v2 low-rank row/column factors on top of the BiMAP-lite
	// diagonal row/column scaling backbone. When false, BiMAP reduces to the
	// original scale-only prototype.
	bool bimapLowRankEnabled;

	// Strength of the row/column matrix anisotropy term inside the BiMAP
	// two-sided preconditioner. 0 reduces BiMAP to its Adam-style diagonal
	// backbone.
	float bimapGeometryScale;

	// Strength of the bounded one-step predictive extrapolation blended into the
	// BiMAP first-moment signal.
	float bimapPredictiveScale;

	// Number of optimizer steps between BiMAP row/column factor refreshes.
	unsigned int bimapFactorCadence;

	// Enable PACT: Promoted Adaptive Compressed Tensor-preconditioner. PACT
	// keeps an exact AdamW fallback and only promotes matrix blocks into a
	// two-sided blockwise preconditioner when predicted gain clears a compute
	// penalty proxy.
	bool pactEnabled;

	// Enable low-rank row/column factors inside the promoted PACT metric. When
	// false, promoted blocks use only diagonal row/column anisotropy.
	bool pactLowRankEnabled;

	// Strength of the promoted row/column anisotropy term. 0 reduces promoted
	// PACT blocks to the exact AdamW fallback.
	float pactGeometryScale;

	// Strength of the bounded secant-style transport blended into the promoted
	// block signal before two-sided preconditioning.
	float pactPredictiveScale;

	// Number of optimizer steps between PACT factor refreshes on promoted
	// matrix blocks.
	unsigned int pactFactorCadence;

	// Enable RACER-lite: Risk-Adjusted Compute-Efficient Reconditioner. RACER
	// keeps exact AdamW fallback and only promotes matrix blocks into a BiMAP-
	// style two-sided preconditioner when stable-signal reward minus
	// curvature/noise penalties clears a compute-cost threshold.
	bool racerEnabled;

	// Strength of RACER-lite's row/column anisotropy term. 0 reduces RACER to
	// the exact AdamW fallback.
	float racerGeometryScale;

	// Strength of the bounded secant-style transport blended into RACER's
	// first-moment signal.
	float racerPredictiveScale;

	// Number of optimizer steps between RACER row/column factor refreshes.
	unsigned int racerFactorCadence;

	// Multiplier applied to RACER's residual-noise penalty when comparing the
	// promoted matrix step against the exact AdamW fallback.
	float racerRiskScale;

	// Multiplier applied to RACER's analytical optimizer-overhead proxy.
	float racerCostScale;

	// Promote a block when its EMA'd RACER reward margin rises above this
	// threshold.
	float racerPromoteThreshold;

	// Demote a previously promoted block when its EMA'd RACER reward margin
	// falls below this threshold.
	float racerDemoteThreshold;

	// Enable KRON: a true blockwise row/column factor preconditioner that keeps
	// Adam-style moments but replaces the matrix-block update with a two-sided
	// inverse-square-root factor apply on the current momentum signal.
	bool kronEnabled;

	// Strength of the KRON two-sided matrix factor apply. 1.0 uses the full
	// preconditioned step; 0.0 degenerates exactly to the Adam-style fallback.
	float kronGeometryScale;

	// Strength of the bounded secant-style transport blended into the KRON
	// block signal before applying row/column inverse-square-root factors.
	float kronPredictiveScale;

	// Number of optimizer steps between KRON factor refreshes.
	unsigned int kronFactorCadence;

	// Additive normalized diagonal floor used when forming KRON row/column
	// inverse-square-root factors from the EMA covariance blocks.
	float kronDamping;

	// Enable MATRA: Manifold-Admissible Trust-Region Adam. MATRA keeps the
	// exact AdamW backbone, blends in a cheap two-sided matrix-geometry
	// candidate on anisotropic blocks, and optionally adds a trusted
	// orthogonal matrix residual on eligible shapes.
	bool matraEnabled;

	// Maximum trust weight assigned to MATRA's two-sided geometry candidate.
	// 0 reduces MATRA to the Adam-style backbone (up to orthogonal residuals).
	float matraGeometryScale;

	// Maximum trust weight assigned to MATRA's orthogonal matrix candidate.
	// 0 disables the orthogonal branch and leaves only the geometry residual.
	float matraOrthogonalScale;

	// Strength of the bounded one-step predictive transport blended into the
	// first-moment signal before forming MATRA's candidates.
	float matraPredictiveScale;

	// Maximum total structured-update budget. MATRA constrains the sum of its
	// geometry and orthogonal trust weights to this value on every block.
	float matraTrustRadius;

	// Number of optimizer steps between MATRA row/column second-moment refreshes.
	unsigned int matraMetricCadence;

	// Number of optimizer steps between exact MATRA orthogonal residual solves.
	// 1 runs the orthogonal branch on every eligible step; larger values keep
	// predictive + geometry active every step while sparsifying the exact solve.
	unsigned int matraOrthCadence;

	// Only allow MATRA's orthogonal branch on matrix blocks whose aspect ratio
	// max(m, n) / min(m, n) does not exceed this limit.
	float matraMaxAspect;

	// Minimum block side length required before MATRA's orthogonal branch can engage.
	unsigned int matraMinDim;

	// Additive floor used when inverting the small Gram matrix inside MATRA's
	// orthogonal candidate.
	float matraDamping;

	// Enable ARGOS: Actuation-Routed Geometry with Observability Steering.
	// ARGOS keeps the exact AdamW backbone, reuses MATRA/MUON-style structured
	// candidates, and routes the residual budget toward head-observable blocks
	// when the measured block reward clears the Adam anchor.
	bool argosEnabled;

	// Scope of ARGOS matrix promotion:
	// 0 = all eligible matrices,
	// 1 = head-only (tied token embedding / output projection),
	// 2 = late-only (last decoder block matrices),
	// 3 = late-head (last decoder block plus head/output).
	enum
	{
		ARGOS_SCOPE_ALL = 0u,
		ARGOS_SCOPE_HEAD_ONLY = 1u,
		ARGOS_SCOPE_LATE_ONLY = 2u,
		ARGOS_SCOPE_LATE_HEAD = 3u
	};
	unsigned int argosScope;

	// Maximum trust weight assigned to ARGOS's two-sided geometry candidate.
	// 0 reduces ARGOS to the Adam-style predictive anchor (up to orthogonal
	// trust, if enabled).
	float argosGeometryScale;

	// Maximum trust weight assigned to ARGOS's orthogonal matrix candidate.
	float argosOrthogonalScale;

	// Strength of the bounded one-step predictive transport blended into the
	// first-moment signal before ARGOS evaluates structured candidates.
	float argosPredictiveScale;

	// Maximum total structured-update budget after observability routing.
	float argosTrustRadius;

	// Number of completed optimizer steps over which ARGOS linearly warms its
	// predictive anchor and structured trust budget from argosWarmupStartScale
	// to their configured strengths. 0 disables warmup.
	unsigned int argosWarmupSteps;

	// Starting multiplier used on the first ARGOS step when warmup is enabled.
	// 0 preserves exact AdamW on step 1; 1 keeps full ARGOS strength throughout.
	float argosWarmupStartScale;

	// Final actuation scale applied to ARGOS's deviation away from the exact
	// AdamW backbone after candidate/trust computation. 0 keeps exact AdamW
	// updates while still refreshing ARGOS state; 1 applies full ARGOS.
	float argosActuationScale;

	// Number of optimizer steps between ARGOS row/column metric refreshes.
	unsigned int argosMetricCadence;

	// Number of optimizer steps between exact ARGOS orthogonal residual solves.
	unsigned int argosOrthCadence;

	// Only allow ARGOS's orthogonal branch on matrix blocks whose aspect ratio
	// max(m, n) / min(m, n) does not exceed this limit.
	float argosMaxAspect;

	// Minimum block side length required before ARGOS's orthogonal branch can engage.
	unsigned int argosMinDim;

	// Additive floor used when inverting the small Gram matrix inside ARGOS's
	// orthogonal candidate.
	float argosDamping;

	// Multiplier applied to the block observability signal before routing the
	// residual budget. Larger values make ARGOS more willing to spend trust on
	// anisotropic, reward-positive blocks.
	float argosObservabilityScale;

	// Additional observability bonus for head/output blocks.
	float argosHeadBonus;

	// Additional observability bonus for blocks in the final decoder layer.
	float argosLateBonus;

	// Enable MUON-lite: selective orthogonalized-momentum updates on eligible
	// matrix blocks with exact AdamW fallback on all other parameters.
	bool muonEnabled;

	// Blend strength for the orthogonalized-momentum direction. 0 reduces
	// MUON-lite exactly to the Adam-style fallback.
	float muonGeometryScale;

	// Strength of the bounded secant-style transport blended into the Adam
	// first-moment signal before orthogonalization.
	float muonPredictiveScale;

	// Only apply MUON-lite to matrix blocks whose aspect ratio
	// max(m, n) / min(m, n) does not exceed this limit.
	float muonMaxAspect;

	// Minimum block side length required before MUON-lite is allowed to engage.
	unsigned int muonMinDim;

	// Additive floor used when inverting the small Gram matrix inside the polar
	// factor computation.
	float muonDamping;

	// Multiplier applied to PACT's analytical optimizer-overhead proxy when
	// deciding whether a block should remain promoted.
	float pactCostScale;

	// Promote a block when its EMA'd predicted-gain margin rises above this
	// threshold.
	float pactPromoteThreshold;

	// Demote a previously promoted block when its EMA'd predicted-gain margin
	// falls below this threshold.
	float pactDemoteThreshold;

	// Mirror-descent step size used by SEAM when updating its
	// {spatial, predictive, output} coordinate simplex.
	float seamMirrorStep;

	// Maximum D-metric residual budget for SEAM's predictive/output correction.
	float seamBudgetMax;

	// Temperature used by QUASAR when turning delayed mode evidence into a soft
	// residual-mode distribution. Lower values make the controller more peaked.
	float quasarTemperature;

	// Maximum D-metric residual budget for QUASAR's probabilistic residual.
	float quasarBudgetMax;

	// Relative scaling for AEGIS predictive evidence when comparing SPARROW and
	// ASTER channel confidence.
	float aegisPredictiveScale;

	// Relative scaling for AEGIS output-space evidence when comparing SPARROW
	// and ASTER channel confidence.
	float aegisOutputScale;

	// Baseline CITADEL anchor applied before contextual trust adjustments.
	float citadelAnchorBase;

	// Additional CITADEL anchor strength for hard transformer regimes.
	float citadelHardRegimeScale;

	// Additional CITADEL anchor strength from predictive/output disagreement.
	float citadelDisagreementScale;

	// Spatial-prior boost applied by CITADEL once the contextual anchor is
	// computed.
	float citadelSpatialScale;

	// Minimum backbone precision used by the RAMPART posterior. Larger values
	// keep the solution closer to the ATLAS/BSRP base even when the residual
	// channels are confident.
	float rampartTauMin;

	// Maximum backbone precision used by the RAMPART posterior in uncertain or
	// disagreement-heavy regimes.
	float rampartTauMax;

	// Maximum D-metric residual budget for the combined predictive/output
	// correction after the posterior solve. Values in [0, 1] keep the residual
	// bounded relative to the base step.
	float rampartBudgetMax;

	// Mixing coefficient for predictive/output covariance inside the RAMPART
	// posterior. Higher values reduce double-counting when both residual sensors
	// are strong and agree.
	float rampartCovarianceMix;

	// Multiplicative strength applied to MERIT's spatial geometry proxy when
	// converting ATLAS active capture into a residual-budget modifier.
	float meritGeometryScale;

	// Minimum backbone precision used by the MERIT posterior. Larger values keep
	// the residual correction closer to the ATLAS/BSRP geometry even when the
	// predictive/output sensors are confident.
	float meritTauMin;

	// Maximum backbone precision used by the MERIT posterior in uncertain or
	// disagreement-heavy regimes.
	float meritTauMax;

	// Maximum D-metric residual budget for MERIT's predictive/output correction.
	float meritBudgetMax;

	// Mixing coefficient for predictive/output covariance inside MERIT's
	// two-sensor posterior.
	float meritCovarianceMix;

	// Baseline score offset for STRATA's null mode. Larger values make the
	// controller stay closer to the AdamW/ATLAS backbone in uncertain regimes.
	float strataNullBias;

	// Penalty applied when STRATA switches away from the previous dominant mode.
	// Larger values increase dwell time and reduce mode oscillation.
	float strataDwellPenalty;

	// Maximum D-metric residual budget for STRATA's selected predictive/output
	// mode. Values in [0, 1] keep the residual bounded relative to the base
	// step.
	float strataBudgetMax;

	// Multiplicative strength applied to ATLAS active-capture geometry when
	// STRATA evaluates predictive-only mode candidates.
	float strataPredictiveGeometryScale;

	// Multiplicative strength applied to ATLAS active-capture geometry when
	// STRATA evaluates coupled predictive/output mode candidates.
	float strataCoupledGeometryScale;

	// Fisher EMA decay rate. Controls how quickly the Fisher diagonal and
	// normalized covariance trace adapt. Higher values (closer to 1) give more
	// stable estimates.
	float beta;

	// Prediction coefficient bounds. The adaptive mu is clamped to [muMin, muMax].
	// mu=0 disables temporal prediction (falls back to plain natural gradient).
	float muMin;
	float muMax;

	// Subspace refresh interval (optimizer steps). Every tSub steps, the subspace
	// basis U is refreshed via randomized power iteration with EMA blending.
	// Set to 0 to disable periodic refresh (use initial subspace only).
	unsigned int tSub;

	// Number of power iteration steps for subspace SVD computation.
	// More iterations give better subspace approximation at higher cost.
	unsigned int powerIters;

	// Regularization epsilon added to Fisher diagonal and sigma2 before inversion.
	float eps;

	// Maximum ratio of effective baseline learning rate to nominal lr.
	// Caps baselineRate = lr/(sigma2+eps) at kappaMax*lr to prevent divergence
	// as sigma2 converges to small gradient variance during training.
	// Higher values allow more aggressive preconditioning; 10 is a safe default.
	float kappaMax;

	// EMA blending coefficient for subspace refresh. Controls how much the new
	// power-iteration basis replaces the old basis at each refresh.
	// 0 = keep old basis (no refresh), 1 = full replacement (old ATLAS behavior).
	// Recommended: 0.5 (balanced blending preserves Fisher/prevGz continuity).
	float betaRefresh;

	// Growth rate for mu recovery. After transient instability drives mu to muMin,
	// this additive term allows mu to slowly recover toward muMax.
	// newMu = mu * (1 - ratio) + muGrowthRate * (muMax - mu)
	// Set to 0 to disable recovery (old behavior: mu can only shrink).
	float muGrowthRate;

	// When true, weight the refresh seed by the current Fisher diagonal so the
	// tracked subspace follows the same curvature signal used for preconditioning.
	bool fisherWeightedRefresh;

	// When true, ATLAS can shrink the active subspace rank when the observed
	// Fisher mass is concentrated in fewer directions than the configured rank.
	bool adaptiveRank;

	// Lower bound for the dynamically active rank. The allocated rank is still
	// `rank`; this only controls how many leading directions are used each step.
	unsigned int minActiveRank;

	// Fisher mass fraction retained by the active rank. Example: 0.95 means use
	// the smallest prefix of Fisher directions whose cumulative mass is >= 95%.
	float rankCapture;

	// If fisher_max / fisher_min stays below this threshold, treat the active
	// spectrum as effectively flat and shrink the sketch budget conservatively
	// at refresh boundaries. Set <= 1 to disable the flat-spectrum heuristic.
	float flatSpectrumThreshold;

	// Enable bias correction for EMA quantities (sigma2, fisherDiag, totalTrace).
	// When true, applies the standard correction factor 1/(1 - beta^step)
	// to compensate for zero-initialization bias in early steps. This
	// allows the optimizer to deliver meaningful preconditioning from step 1
	// instead of waiting ~1/(1-beta) steps for the EMA to converge.
	bool biasCorrection;

	ATLASConfig()
	    : rank(128u),
	      complementRank(0u),
	      complementLrScale(0.25f),
	      complementKappaMax(0.5f),
	      prismEnabled(false),
	      prismLagHorizon(2u),
	      prismMemoryScale(0.15f),
	      prismPredictiveEdgeThreshold(0.05f),
	      resolveEnabled(false),
	      resolveLagHorizon(4u),
	      resolveMemoryScale(0.10f),
	      resolvePredictiveEdgeThreshold(0.05f),
	      heroEnabled(false),
	      heroLagHorizon(4u),
	      heroMemoryScale(0.10f),
	      heroEdgeThreshold(0.10f),
	      cobaltEnabled(false),
	      cobaltLagHorizon(4u),
	      cobaltMemoryScale(0.08f),
	      cobaltEdgeThreshold(0.10f),
	      birchEnabled(false),
	      birchPastHorizon(3u),
	      birchFutureHorizon(2u),
	      birchMemoryScale(0.08f),
	      birchEdgeThreshold(0.10f),
	      ghostEnabled(false),
	      ghostLagHorizon(4u),
	      ghostMemoryScale(0.05f),
	      ghostEdgeThreshold(0.10f),
	      sparrowEnabled(false),
	      sparrowModeRank(1u),
	      sparrowAutoModeGate(false),
	      sparrowMemoryScale(0.05f),
	      sparrowEdgeThreshold(0.10f),
	      sparrowSecondEdgeThreshold(0.10f),
	      sparrowSecondEdgeFraction(0.50f),
	      sparrowPoleMax(0.95f),
	      qbrtEnabled(false),
	      qbrtLagHorizon(4u),
	      qbrtMemoryScale(0.05f),
	      qbrtEdgeThreshold(0.10f),
	      qbrtPoleMax(0.95f),
	      qrcEnabled(false),
	      qrcLagHorizon(4u),
	      qrcMemoryScale(0.05f),
	      qrcEdgeThreshold(0.10f),
	      qrcPoleMax(0.95f),
	      riftEnabled(false),
	      riftLagHorizon(4u),
	      riftMemoryScale(0.05f),
	      riftEdgeThreshold(0.05f),
	      riftPoleMax(0.95f),
	      orbitEnabled(false),
	      orbitMemoryScale(0.04f),
	      orbitEdgeThreshold(0.05f),
	      orbitPoleMax(0.95f),
	      helmEnabled(false),
	      helmMemoryScale(0.05f),
	      helmEdgeThreshold(0.10f),
	      helmModeRank(2u),
	      helmHiddenStackDepth(2u),
	      helmPoleMax(0.95f),
	      asterEnabled(false),
	      aegisEnabled(false),
	      citadelEnabled(false),
	      rampartEnabled(false),
	      meritEnabled(false),
	      strataEnabled(false),
	      auroraEnabled(false),
	      seamEnabled(false),
	      quasarEnabled(false),
	      asterMemoryScale(0.05f),
	      asterEdgeThreshold(0.10f),
	      asterStateRank(2u),
	      asterHiddenStackDepth(2u),
	      asterPoleMax(0.95f),
	      kappaEnabled(false),
	      kappaHeads(1u),
	      kappaLagBuckets(4u),
	      kappaRank(2u),
	      auroraHorizonBlend(0.65f),
	      auroraBudgetMax(0.70f),
	      auroraAdamwBackbone(false),
	      auroraHeadGain(1.0f),
	      auroraBodyTrustScale(1.0f),
	      geodeEnabled(false),
	      geodeGeometryScale(1.0f),
	      geodePredictiveScale(0.25f),
	      echoEnabled(false),
	      echoGeometryScale(1.0f),
	      echoGeometryScaleFinal(1.0f),
	      echoGeometryDecaySteps(0u),
	      echoMetricCadence(1u),
	      echoTrustScale(0.0f),
	      echoPredictiveScale(0.0f),
	      echoStructuralScale(0.0f),
	      echoStructuralGroups(1u),
	      echoScope(ECHO_SCOPE_ALL),
	      bimapEnabled(false),
	      bimapScope(BIMAP_SCOPE_ALL),
	      bimapLowRankEnabled(true),
	      bimapGeometryScale(1.0f),
	      bimapPredictiveScale(0.15f),
	      bimapFactorCadence(8u),
	      pactEnabled(false),
	      pactLowRankEnabled(true),
	      pactGeometryScale(1.0f),
	      pactPredictiveScale(0.10f),
	      pactFactorCadence(8u),
	      racerEnabled(false),
	      racerGeometryScale(1.0f),
	      racerPredictiveScale(0.05f),
	      racerFactorCadence(8u),
	      racerRiskScale(0.50f),
	      racerCostScale(0.0010f),
	      racerPromoteThreshold(0.0f),
	      racerDemoteThreshold(-0.0005f),
	      kronEnabled(false),
	      kronGeometryScale(1.0f),
	      kronPredictiveScale(0.05f),
	      kronFactorCadence(8u),
	      kronDamping(0.10f),
	      matraEnabled(false),
	      matraGeometryScale(1.0f),
	      matraOrthogonalScale(0.5f),
	      matraPredictiveScale(0.05f),
	      matraTrustRadius(0.50f),
	      matraMetricCadence(1u),
	      matraOrthCadence(1u),
	      matraMaxAspect(1.50f),
	      matraMinDim(8u),
	      matraDamping(0.01f),
	      argosEnabled(false),
	      argosScope(ARGOS_SCOPE_ALL),
	      argosGeometryScale(1.0f),
	      argosOrthogonalScale(0.5f),
	      argosPredictiveScale(0.05f),
	      argosTrustRadius(0.60f),
	      argosWarmupSteps(0u),
	      argosWarmupStartScale(0.0f),
	      argosActuationScale(1.0f),
	      argosMetricCadence(1u),
	      argosOrthCadence(1u),
	      argosMaxAspect(1.50f),
	      argosMinDim(8u),
	      argosDamping(0.01f),
	      argosObservabilityScale(0.75f),
	      argosHeadBonus(0.35f),
	      argosLateBonus(0.20f),
	      muonEnabled(false),
	      muonGeometryScale(1.0f),
	      muonPredictiveScale(0.05f),
	      muonMaxAspect(1.50f),
	      muonMinDim(8u),
	      muonDamping(0.01f),
	      pactCostScale(0.0010f),
	      pactPromoteThreshold(0.0f),
	      pactDemoteThreshold(-0.0005f),
	      seamMirrorStep(0.35f),
	      seamBudgetMax(0.65f),
	      quasarTemperature(0.60f),
	      quasarBudgetMax(0.70f),
	      aegisPredictiveScale(1.0f),
	      aegisOutputScale(1.0f),
	      citadelAnchorBase(0.0f),
	      citadelHardRegimeScale(0.85f),
	      citadelDisagreementScale(0.75f),
	      citadelSpatialScale(4.0f),
	      rampartTauMin(0.50f),
	      rampartTauMax(4.00f),
	      rampartBudgetMax(0.60f),
	      rampartCovarianceMix(0.35f),
	      meritGeometryScale(1.25f),
	      meritTauMin(0.35f),
	      meritTauMax(3.00f),
	      meritBudgetMax(0.70f),
	      meritCovarianceMix(0.30f),
	      strataNullBias(0.20f),
	      strataDwellPenalty(0.15f),
	      strataBudgetMax(0.65f),
	      strataPredictiveGeometryScale(0.75f),
	      strataCoupledGeometryScale(0.45f),
	      beta(0.999f),
	      muMin(0.01f),
	      muMax(0.3f),
	      tSub(200u),
	      powerIters(2u),
	      eps(1e-8f),
	      kappaMax(10.0f),
	      betaRefresh(0.5f),
	      muGrowthRate(0.001f),
	      fisherWeightedRefresh(true),
	      adaptiveRank(false),
	      minActiveRank(1u),
	      rankCapture(0.95f),
	      flatSpectrumThreshold(1.05f),
	      biasCorrection(true)
	{
	}

	inline float echoEffectiveGeometryScale(unsigned long long optimizerStep) const
	{
		const float start = (echoGeometryScale > 0.0f) ? echoGeometryScale : 0.0f;
		const float finalScale = (echoGeometryScaleFinal > 0.0f) ? echoGeometryScaleFinal : 0.0f;
		if (echoGeometryDecaySteps == 0u)
			return start;
		if (optimizerStep <= 1ULL)
			return start;
		double progress = static_cast<double>(optimizerStep - 1ULL)
		                / static_cast<double>(echoGeometryDecaySteps);
		if (progress < 0.0)
			progress = 0.0;
		if (progress > 1.0)
			progress = 1.0;
		return start + static_cast<float>(progress) * (finalScale - start);
	}

	inline bool echoShouldRefresh(unsigned long long optimizerStep) const
	{
		const unsigned long long cadence =
		    static_cast<unsigned long long>(std::max(1u, echoMetricCadence));
		if (cadence <= 1ULL)
			return true;
		const unsigned long long stepIndex = (optimizerStep > 0ULL) ? optimizerStep : 1ULL;
		return ((stepIndex - 1ULL) % cadence) == 0ULL;
	}

	inline float argosWarmupMultiplier(unsigned long long optimizerStep) const
	{
		if (argosWarmupSteps == 0u)
			return 1.0f;
		const float startScale = std::max(0.0f, std::min(1.0f, argosWarmupStartScale));
		if (optimizerStep >= static_cast<unsigned long long>(argosWarmupSteps))
			return 1.0f;
		const float progress =
		    static_cast<float>(optimizerStep)
		    / static_cast<float>(std::max(1u, argosWarmupSteps));
		return startScale + progress * (1.0f - startScale);
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

	// Store AdamW optimizer state (m, v) in BF16 instead of FP32. Halves
	// optimizer-state VRAM (4 bytes -> 2 bytes per parameter per moment,
	// so 4x params -> 2x params per GB). Compute (EMA, bias-correction,
	// sqrt, division) still happens in FP32 — only storage is BF16. BF16's
	// 8-bit exponent avoids FP16's underflow in v; 7-bit mantissa incurs
	// ~0.4% per-update quantization bias that the EMA averages out.
	// Requires GPU; takes the non-batched Adam path.
	bool adamStateBf16;

	// Store AdamW optimizer state (m as int8, v as uint8) with per-256-element
	// FP32 absmax scales.  ~1.016 bytes/param/moment vs. 4 bytes FP32 (4× drop)
	// or 2 bytes BF16 (~2× drop).  Mechanism: dequant on read with
	// (val/127 or val/255) * scale, EMA in FP32, requantize on write via
	// block-wide absmax reduction.  v uses unsigned [0,255] — doubles
	// precision near zero where 1/√v matters most.  Ported from CHIRON
	// (paradigm #11 MFIO mechanism).  Mutually exclusive with adamStateBf16
	// when both are set; int8 wins (it's the more aggressive compression).
	bool adamStateInt8;

	// Store gradients as BF16 instead of FP32.  Halves grad-buffer VRAM
	// (~3.7 GB savings at 1.84B).  Mechanism: each backward GEMM writes
	// into a shared scratch FP32 buffer (sized to the widest weight tensor),
	// then bf16_accum_axpy commits the result into the persistent BF16 grad
	// buffer with stochastic-rounding-equivalent rebanding.  Adam reads the
	// BF16 grad directly (uses adam_update_*_bf16grad kernels).  Required
	// alongside adamStateBf16 or adamStateInt8 to fit 1.84B-class on a 16 GB
	// GPU.  Ported from CHIRON's BF16-grads path (paradigm-stack at ≥500M).
	bool gradStorageBf16;

	// Phase-2 of BF16 grad storage: backward GEMMs commit DIRECTLY to the BF16
	// mirrors via a single shared FP32 scratch buffer; the Phase-1 cast pass
	// (FP32 grad -> BF16 mirror) is skipped; grad-norm reads BF16 mirrors.
	// Implies gradStorageBf16=true.  Mutually exclusive with non-compressed
	// Adam paths (full Adam batch, atlas/geode/muon) — those still want FP32
	// grads.  When phase2 active, the FP32 grad buffers are still allocated
	// (other code paths reference them) but never written to by backward;
	// retiring those allocations is a follow-on memory-cleanup task.
	bool gradStorageBf16Phase2;

	// CHIRON-style BF16 weight storage: per-block weights (Wq/Wk/Wv/Wo/W1/W2)
	// + tokE + WIn + WOut persist as BF16 on GPU (no FP32 master).  Adam reads
	// BF16 → casts to one shared FP32 scratch → applies update → casts back to
	// BF16 with stochastic rounding.  Saves another ~50% on weight VRAM (~3.7
	// GB at 1.84B) on top of the bf16-grad savings.  Forward already uses BF16
	// mirrors via gpu_gemm_mp, so no forward-side change is needed.
	bool weightStorageBf16;

	// Activation gradient checkpointing (sqrt-L scheme).  When enabled the
	// per-layer activation scratches (x1/Q/K/V/attnConcat/attnOut/hAfterAttn/
	// x2/ff1/ff1Act/ffOut/hAfterFF and LN stats) are sized to K = ⌈√L⌉ slots
	// instead of L; layer li writes into slot `li % K`.  At checkpoint
	// boundaries (every K layers) the layer-input residual hAfterFF is copied
	// into a `checkpoints[c]` buffer.  Backward walks segments from highest
	// down to 0 — for each segment it re-runs forward starting from the
	// checkpoint to repopulate the K slots, then runs backward in reverse over
	// the segment.  Saves ~3.6 GB at 1.84B (4.2 GB stash → 0.55 GB scratch +
	// checkpoints) at the cost of ~33% extra compute (one extra forward per
	// step).  Composes with bf16-weights and bf16-grads.
	bool activationCheckpoint;

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
	      adamStateBf16(false),
	      adamStateInt8(false),
	      gradStorageBf16(false),
	      gradStorageBf16Phase2(false),
	      weightStorageBf16(false),
	      activationCheckpoint(false),
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

struct WarmupConfig
{
	enum Type { WARMUP_NONE = 0, WARMUP_LINEAR = 1 };
	Type type;
	int warmupSteps; // optimizer steps (not epochs)
	WarmupConfig() : type(WARMUP_NONE), warmupSteps(0) {}
	inline float multiplier(int optimizerStep) const
	{
		if (type == WARMUP_NONE || warmupSteps <= 0) return 1.0f;
		if (optimizerStep >= warmupSteps) return 1.0f;
		return static_cast<float>(optimizerStep) / static_cast<float>(warmupSteps);
	}
};

struct DDPConfig
{
	bool enable;            // master switch (default false)
	bool linearLRScaling;   // scale LR by worldSize (default true)
	int rank;               // this worker's rank (0 = root)
	int worldSize;          // total workers
	int rootPort;           // port root listens on
	int compressionMode;    // 0=none, 1=FP16, 2=FP16+TopK
	float topKRatio;        // fraction of gradients to keep (default 0.01)
	int topKWarmupSteps;    // use FP16-only for first N steps (default 0)
	DDPConfig() : enable(false), linearLRScaling(true),
	              rank(0), worldSize(1), rootPort(9200),
	              compressionMode(0), topKRatio(0.01f), topKWarmupSteps(0) {}
};

// CNN-specific run configuration.
//
// Image dimensions (inputH * inputW * inputC) must equal the DataInput feature count.
// NNInfo hidden layers define FC layers after flatten (same as DFF).
// NNInfo output layer defines number of classes.
struct CNNConfig
{
	unsigned int inputH, inputW, inputC; // image dimensions (NCHW)

	struct ConvLayerSpec
	{
		unsigned int outChannels;   // filter count
		unsigned int kernelH, kernelW;
		unsigned int strideH, strideW;
		unsigned int padH, padW;    // zero-padding
		bool useBatchNorm;          // BN after conv
		bool useMaxPool;            // pooling after activation
		unsigned int poolH, poolW, poolStrideH, poolStrideW;

		ConvLayerSpec()
		    : outChannels(0u),
		      kernelH(3u), kernelW(3u),
		      strideH(1u), strideW(1u),
		      padH(1u), padW(1u),
		      useBatchNorm(false),
		      useMaxPool(false),
		      poolH(2u), poolW(2u),
		      poolStrideH(2u), poolStrideW(2u)
		{
		}
	};

	std::vector<ConvLayerSpec> convLayers;
	float batchNormEps;       // default 1e-5
	float batchNormMomentum;  // default 0.1 (EMA for running stats)

	CNNConfig()
	    : inputH(0u), inputW(0u), inputC(0u),
	      convLayers(),
	      batchNormEps(1e-5f),
	      batchNormMomentum(0.1f)
	{
	}
};

// Transposed-convolution (deconvolution) generator configuration for GANs.
struct DeconvConfig
{
	unsigned int projectChannels; // channels after FC projection
	unsigned int projectH;       // spatial height after projection
	unsigned int projectW;       // spatial width after projection

	struct DeconvLayerSpec
	{
		unsigned int outChannels;
		unsigned int kernelH, kernelW;
		unsigned int strideH, strideW;
		unsigned int padH, padW;
		bool useBatchNorm;
		bool useReLU; // false for last layer (use sigmoid)
		bool useUpsampleConv; // nearest-neighbor upsample + standard conv (no checkerboard)
		bool useTanh;

		DeconvLayerSpec()
		    : outChannels(0u),
		      kernelH(4u), kernelW(4u),
		      strideH(2u), strideW(2u),
		      padH(1u), padW(1u),
		      useBatchNorm(false),
		      useReLU(true),
		      useUpsampleConv(false),
		      useTanh(false)
		{
		}
	};

	std::vector<DeconvLayerSpec> layers;

	DeconvConfig()
	    : projectChannels(128u),
	      projectH(7u),
	      projectW(7u),
	      layers()
	{
	}
};

// VESTA optimizer configuration (Variational Entropy-Spectral Trust-region Adaptation).
//
// Per-weight-matrix Bregman mirror descent on the von Neumann spectral entropy
// potential Phi(W) = -1/2 tr(W^T W log W^T W) + mu/2 |W|_F^2. Tracks the top-r
// SVD of each weight matrix; updates are derived from the KKT conditions of a
// trust-region-constrained mirror step with spectral-homeostatic regularization.
// No second-moment EMA of squared gradients is maintained.
struct VestaConfig
{
	// Sketch rank per weight matrix. Clamped to min(rank, min(m, n)) at init.
	unsigned int rank;

	// Frobenius stabilizer of the spectral entropy potential Phi(W).
	// Must satisfy mu >= 3.0 for strict convexity near sigma=1; default 4.0.
	float mu;

	// Strength of the spectral-control regularizer R(W) = 1/2 * sum (ell - ellStar)^2.
	// 0 disables spectral homeostasis; default 0.1.
	float tau;

	// Operator-norm trust-region radius: max exp(ell[0]) is clamped to
	// (1 + rho) * prev_max. Default 0.05.
	float rho;

	// Scale of the signed complement step (for gradient components outside the
	// tracked subspace). Default 0.2.
	float lambdaPerp;

	// EMA rate for log-scale momentum beta = (1-gamma)*beta + gamma*ell. Default 0.01.
	float gamma;

	// Feedback rate from beta to ell: ell = (1-kappa)*ell + kappa*beta. Default 0.1.
	float kappa;

	// Homeostasis rate for ellStar update: ellStar = (1-nu)*ellStar + nu*ell.
	// Default 0.01.
	float nu;

	// Subspace refresh period (steps between sketched SVD refresh). Default 4.
	unsigned int tSk;

	// Homeostasis update period (steps between ellStar update). Default 1000.
	unsigned int tHom;

	// Power iteration count inside the sketch refresh. Default 2.
	unsigned int powerIters;

	// Clamp range on ell = log(sigma) to prevent over/underflow.
	float ellMin; // default -10.0
	float ellMax; // default   4.0

	// Numerical floor on phi_dd = -2*ell - 3 + mu to avoid division by near-zero.
	float phiDdFloor; // default 0.1

	// Lion-style momentum on the signed complement step. When enabled, VESTA
	// tracks an EMA of the out-of-subspace gradient and signs the EMA rather
	// than the instantaneous gradient. Costs one extra [m*n] buffer per weight
	// matrix, but empirically closes a large fraction of the AdamW gap.
	bool complementMomentumEnabled; // default false (opt-in)
	float complementBeta;           // default 0.9 (heavy-ball-style EMA rate)

	// When complementMomentumEnabled is true, controls whether the step is
	//   sign(m_perp)  — Lion-style, fixed-magnitude (default, good short horizons)
	// or
	//   m_perp        — classical heavy-ball, gradient-magnitude-aware (good long horizons)
	// Set to false to switch to raw-momentum mode. The lr * lambdaPerp product
	// typically needs re-tuning: raw-mode optima have lambdaPerp 3-10x larger
	// than sign-mode optima because the step magnitude now scales with the
	// gradient's own EMA.
	bool complementUseSign;         // default true (Lion-style)

	// Sign-stabilized tracked update: maintain an EMA of the tracked-subspace
	// diagonal A[i,i] = (U^T g V)_ii and use the EMA (rather than the
	// instantaneous value) in the log-scale mirror step. Kills per-step
	// variance in ell without changing the Bregman-mirror geometry.
	// Costs r fp32 floats of extra state per weight matrix.
	bool trackedEmaEnabled;         // default false (opt-in)
	float trackedEmaBeta;           // default 0.9

	// Basis source for the tracked subspace:
	//   0 = weights  (sketched SVD of W; original VESTA design)
	//   1 = gradient (sketched SVD of EMA(g); Fisher-adjacent)
	// When 1, an m*n buffer per matrix tracks the gradient EMA.
	unsigned int basisSource;       // default 0 (weights)
	float basisEmaBeta;             // default 0.99 (EMA rate for gradientEma)

	// If true, the GPU sketched-SVD refresh is executed on-device (cuBLAS GEMMs,
	// on-device modified Gram-Schmidt). Only the small B matrix [rp x n] is
	// downloaded for the Jacobi eigendecomposition and the right singular
	// vectors are uploaded back. If false, the GPU path downloads W to host,
	// runs the CPU sketched SVD, and uploads U/V/ell -- useful for strict
	// CPU/GPU parity tests. Default true (production speedup; at dModel=2048
	// the host roundtrip is ~50% of VESTA wall-clock per step).
	bool gpuRefreshOnDevice;

	VestaConfig()
	    : rank(32u),
	      mu(4.0f),
	      tau(0.1f),
	      rho(0.05f),
	      lambdaPerp(0.2f),
	      gamma(0.01f),
	      kappa(0.1f),
	      nu(0.01f),
	      tSk(4u),
	      tHom(1000u),
	      powerIters(2u),
	      ellMin(-10.0f),
	      ellMax(4.0f),
	      phiDdFloor(0.1f),
	      complementMomentumEnabled(false),
	      complementBeta(0.9f),
	      complementUseSign(true),
	      trackedEmaEnabled(false),
	      trackedEmaBeta(0.9f),
	      basisSource(0u),
	      basisEmaBeta(0.99f),
	      gpuRefreshOnDevice(true)
	{
	}
};

// HELIOS optimizer configuration (Hamiltonian Ensemble Langevin Integrator with
// Sharpness-adaptive Thermostat).
//
// HELIOS discretizes an underdamped Langevin-Nose-Hoover SDE with the BAOAB
// stochastic-symplectic splitting. Minimum viable instantiation uses a scalar
// global temperature, a single thermostat variable, no sharpness feedback, and
// no anchor regularizer. The full design (per-group T, sharpness probe,
// anchor EMA, Li-Sato-Tan noise correction) is described in
// research/HELIOS_framework.md.
//
// Per-weight-matrix state: momentum p [m*n] (and optional anchor thetaBar [m*n]).
// Per-group scalar state: {xi, T, kappa, m_mass}.
struct HeliosConfig
{
	// Step size h. Used only inside the integrator (the outer learning-rate
	// schedule still multiplies it). Default 1.0 so that `lr` passed to
	// applyStep acts as the effective step.
	float h;

	// Base friction floor gamma_0 (>= 0). The effective per-parameter friction
	// is Gamma = gamma_0 + xi + alpha * kappa. Default 0.1.
	float gamma0;

	// Target temperature T_0. Sets the invariant-measure scale: at static T,
	// the theta-marginal is prop. exp(-U(theta)/T). Default 1e-4.
	// For deterministic descent behavior set to 0.
	float T0;

	// Scalar group mass m (units of mass). Momentum has kinetic energy
	// (1/2) * ||p||^2 / m, so effective step on theta is (h/m) * p. Default 1.0.
	float mass;

	// Nose-Hoover thermostat inertia Q. Standard tuning gives Q = N * T /
	// omega_xi^2 with omega_xi ~ 1/(10 h) (i.e. xi relaxes ~10 steps).
	// When 0, thermostat is disabled (pure underdamped Langevin). Default 0.
	float Q;

	// Sharpness feedback coefficient alpha (>= 0). Scales the contribution of
	// kappa to Gamma. Set 0 to disable (minimum viable instantiation). Default 0.
	float alpha;

	// Upper clamp on the sharpness probe kappa_g (prevents runaway friction).
	// Only used when alpha > 0. Default 1e4.
	float kappaMax;

	// Sharpness probe refresh period K_hvp (one HVP per K_hvp steps, round-
	// robined across groups when per-group is enabled). 0 disables HVP
	// probing entirely (kappa stays at 0 and alpha is effectively ignored).
	// Default 0 (off in minimum viable instantiation).
	unsigned int kHvp;

	// Anchor EMA decay beta_a. theta_bar <- beta_a * theta_bar + (1-beta_a) * theta.
	// Anchor penalty in U is (lambda_a/2) * ||theta - theta_bar||^2. When
	// lambda_a == 0 the anchor is disabled and theta_bar is not allocated.
	// Defaults: beta_a = 0.999, lambda_a = 0 (off).
	float betaAnchor;
	float lambdaAnchor;

	// Li-Sato-Tan noise-temperature correction: T_eff = T0 + (h/4) *
	// tr(Sigma_B) / N, where tr(Sigma_B) is the mini-batch gradient noise
	// covariance trace. 0 disables the correction. Default 0 (off in minimum
	// viable instantiation).
	float noiseCorrection;

	HeliosConfig()
	    : h(1.0f),
	      gamma0(0.1f),
	      T0(1e-4f),
	      mass(1.0f),
	      Q(0.0f),
	      alpha(0.0f),
	      kappaMax(1e4f),
	      kHvp(0u),
	      betaAnchor(0.999f),
	      lambdaAnchor(0.0f),
	      noiseCorrection(0.0f)
	{
	}
};

// CHIRON reversible-flow transformer configuration.  When enabled, the
// training loop treats the transformer as a sequence of symplectic
// bijective blocks and reconstructs activations during backward via the
// block inverse rather than storing them (framework:
// research/CHIRON_framework.md).
//
// See research/CHIRON_PROGRESS.md for the current implementation phase.
// Default = disabled: existing code paths are untouched when enable=false.
struct ChironConfig
{
	// Master enable.  When false, CHIRON machinery is inert and the
	// standard transformer block path is used unchanged.
	bool enable;

	// Rank of the per-layer sketch used for BF16 reconstruction-error
	// correction (framework §4.4 / amendment §11a). Ignored when enable
	// is false or when anchorPeriod == 1.
	int sketchRank;

	// Per-token local sketch vs global block sketch. Per-token is the
	// practical choice (framework amendment §11a mitigation 1): the
	// effective sketch input dim shrinks from 2·T·m to 2·m, giving
	// √(T)× tighter per-coord correction at the same rank.
	bool perTokenSketch;

	// Anchor period k (framework §6.4 remedy 2 / amendment §11a
	// mitigation 2). Every k-th block stores a full BF16 activation
	// "anchor" so drift accumulation is capped at length k. With k=1
	// every block is an anchor (degenerates to full-activation
	// training); with k=L no anchors (pure sketch-corrected inverse).
	// Default 8 is the practical sweet spot per the framework.
	int anchorPeriod;

	// Deterministic seed base for sketch matrix generation. The actual
	// per-layer seed is `sketchSeed + layerIndex`. Sketches are
	// regenerated on demand from this seed rather than being stored.
	unsigned int sketchSeed;

	ChironConfig()
	    : enable(false),
	      sketchRank(256),
	      perTokenSketch(true),
	      anchorPeriod(8),
	      sketchSeed(0xC4120Fu)
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

	// Paradigm shift #38 SLC mini-LR-warmup (port from CHIRON iter-178).
	// At each T-schedule transition (managed by the trainer), the trainer
	// calls NNetwork::setSLCTransitionStep(optimizerStep, slcMiniWarmupSteps)
	// to mark the transition.  The per-step LR multiplier in sgd_transformer
	// then applies min(warmupMult, (optimizerStep - slcLastTransitionStep) /
	// slcMiniWarmupSteps) for the first slcMiniWarmupSteps after each
	// transition.  This linear ramp from 0 to full lr after a T jump lets
	// the optimizer's m/v EMAs adapt to the new gradient covariance and
	// prevents the post-transition gradient spike that drove flagship
	// 1.84B Phase-2 (T=512) to clipped-stagnation in the SLC test.
	// Default -1 (disabled).  Set to 0 by the trainer at chunk-1 to skip
	// the warmup before any transition has occurred.
	long long slcLastTransitionStep;
	int slcMiniWarmupSteps;

	// Per-element gradient clipping (<= 0 disables).
	//
	// Historical engine behavior clipped many intermediate gradients/deltas to +/-10.
	// This remains enabled by default for backward compatibility, and is applied across
	// DFF/RNN/GRU/LSTM paths.
	float perElementGradClip;

	// Optimizer configuration.
	OptimizerConfig optimizer;

	// ATLAS optimizer configuration (used when optimizer.type==ATLAS).
	ATLASConfig atlas;

	// VESTA optimizer configuration (used when optimizer.type==VESTA).
	VestaConfig vesta;

	// HELIOS optimizer configuration (used when optimizer.type==HELIOS).
	HeliosConfig helios;

	// Learning rate schedule multiplier configuration.
	LearningRateScheduleConfig lrSchedule;

	// Bayesian adaptive LR configuration (used when lrSchedule.type == BAYESIAN).
	BayesianLRConfig bayesianLR;

	// Transformer run config (used only for transformer net types).
	TransformerRunConfig transformer;

	// Mixed precision (used primarily for transformer net types).
	MixedPrecisionConfig mixedPrecision;

	// GPU offloading configuration (requires GLADES_HAVE_CUDA).
	struct GpuConfig
	{
		// If true, attempt to use GPU when available.
		bool enable;
		// CUDA device ID to use (0 = first GPU).
		int deviceId;
		// Minimum problem size (total floats) before offloading to GPU.
		// Small problems may be faster on CPU due to kernel launch overhead.
		// 0 = always use GPU when enabled.
		size_t minProblemSize;

		GpuConfig()
		    : enable(false),
		      deviceId(0),
		      minProblemSize(0)
		{
		}
	};

	GpuConfig gpu;

	WarmupConfig warmup;

	DDPConfig ddp;

	// Gradient checkpointing: trade compute for memory by recomputing activations
	// during backward instead of storing all per-layer intermediates.
	bool gradientCheckpointing;

	// CHIRON reversible-flow transformer configuration. When enabled, the
	// transformer backward reconstructs activations via the block inverse
	// rather than storing them (research/CHIRON_framework.md).
	ChironConfig chiron;

	// CNN run config (used only for TYPE_CNN).
	CNNConfig cnn;

	TrainingConfig()
	    : minibatchSizeOverride(0),
	      tbpttWindowOverride(0),
	      globalGradClipNorm(0.0f),
	      slcLastTransitionStep(-1),
	      slcMiniWarmupSteps(0),
	      perElementGradClip(10.0f),
	      optimizer(),
	      atlas(),
	      vesta(),
	      lrSchedule(),
	      bayesianLR(),
	      transformer(),
	      mixedPrecision(),
	      gpu(),
	      warmup(),
	      ddp(),
	      gradientCheckpointing(false),
	      chiron(),
	      cnn()
	{
	}
};

} // namespace glades
