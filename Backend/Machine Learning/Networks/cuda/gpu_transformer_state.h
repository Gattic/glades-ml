// GPU mirror of TensorTransformerState + TransformerScratch.
//
// Keeps all weights, optimizer state, and forward/backward scratch buffers
// GPU-resident so only inputs/outputs cross PCIe.
#pragma once

#include "gpu_buffer.h"
#include "gpu_atlas.h"
#include "gpu_vesta.h"
#include "gpu_helios.h"
#include <cstddef>

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

struct HostFloatBufferView
{
	float* data;
	size_t size;

	HostFloatBufferView()
	    : data(0), size(0)
	{
	}
};

struct TransformerHostBlockWeightsView
{
	HostFloatBufferView ln1Gamma;
	HostFloatBufferView ln1Beta;
	HostFloatBufferView Wq;
	HostFloatBufferView Wk;
	HostFloatBufferView Wv;
	HostFloatBufferView Wo;
	HostFloatBufferView bq;
	HostFloatBufferView bk;
	HostFloatBufferView bv;
	HostFloatBufferView bo;
	HostFloatBufferView ln2Gamma;
	HostFloatBufferView ln2Beta;
	HostFloatBufferView W1;
	HostFloatBufferView W2;
	HostFloatBufferView b1;
	HostFloatBufferView b2;
};

struct TransformerHostWeightsView
{
	HostFloatBufferView tokE;
	HostFloatBufferView WIn;
	HostFloatBufferView bIn;
	HostFloatBufferView WOut;
	HostFloatBufferView bOut;
	HostFloatBufferView lmBias;
	HostFloatBufferView lnFinalGamma;
	HostFloatBufferView lnFinalBeta;
	TransformerHostBlockWeightsView* blocks;
	unsigned int blockCount;

	TransformerHostWeightsView()
	    : blocks(0),
	      blockCount(0u)
	{
	}
};

struct TransformerGpuScratchConfig
{
	unsigned int T;
	unsigned int inputSize;
	unsigned int outSize;
	unsigned int dModel;
	unsigned int dFF;
	unsigned int dModelKV;
	unsigned int nHeads;
	unsigned int nLayers;
	unsigned int ff1Width;

	TransformerGpuScratchConfig()
	    : T(0u),
	      inputSize(0u),
	      outSize(0u),
	      dModel(0u),
	      dFF(0u),
	      dModelKV(0u),
	      nHeads(0u),
	      nLayers(0u),
	      ff1Width(0u)
	{
	}
};

// GPU-resident copy of all transformer weights + optimizer state.
// Layout mirrors NNetwork::TensorTransformerState.
struct GpuTransformerWeights
{
	bool initialized;

	// Model config (cached for kernel launches).
	unsigned int dModel;
	unsigned int dFF;
	unsigned int nHeads;
	unsigned int nKVHeads;
	unsigned int nLayers;
	unsigned int vocabSize;
	unsigned int inputSize;
	unsigned int outSize;
	unsigned int ffnKind;
	bool tokenModel;
	bool tieEmbeddings;

	// Token embedding: [vocabSize, dModel]
	GpuBuffer<float> tokE;
	GpuBuffer<float> vTokE;   // Adam m1
	GpuBuffer<float> v2TokE;  // Adam m2
	GpuBuffer<float> gTokE;   // gradients
	// BF16 low-precision mirror, kept in sync with the FP32 master after each
	// optimizer step (ensureGpuLowpMirrors). Empty when mixed precision is off.
	GpuBuffer<uint16_t> tokELowp;
	// BF16 Adam state (m1, m2) — used when MixedPrecisionConfig::adamStateBf16
	// is true. Exactly one of {vTokE, vTokE_bf16} and {v2TokE, v2TokE_bf16}
	// is allocated at a time to save VRAM; allocate() picks based on config.
	GpuBuffer<uint16_t> vTokE_bf16;
	GpuBuffer<uint16_t> v2TokE_bf16;

	// LM head bias: [vocabSize]
	GpuBuffer<float> lmBias;
	GpuBuffer<float> mLmBias;
	GpuBuffer<float> v2LmBias;
	GpuBuffer<float> gLmBias;

	// Input projection: [dModel, inputSize]
	GpuBuffer<float> WIn;
	GpuBuffer<float> vWIn;
	GpuBuffer<float> v2WIn;
	GpuBuffer<float> gWIn;
	GpuBuffer<uint16_t> WInLowp;
	GpuBuffer<uint16_t> vWIn_bf16;
	GpuBuffer<uint16_t> v2WIn_bf16;
	GpuBuffer<float> bIn;    // [dModel]
	GpuBuffer<float> mBIn;
	GpuBuffer<float> v2BIn;
	GpuBuffer<float> gBIn;

	// Output projection: [outSize, dModel]
	GpuBuffer<float> WOut;
	GpuBuffer<float> vWOut;
	GpuBuffer<float> v2WOut;
	GpuBuffer<float> gWOut;
	GpuBuffer<uint16_t> WOutLowp;
	GpuBuffer<uint16_t> vWOut_bf16;
	GpuBuffer<uint16_t> v2WOut_bf16;
	GpuBuffer<float> bOut;   // [outSize]
	GpuBuffer<float> mBOut;
	GpuBuffer<float> v2BOut;
	GpuBuffer<float> gBOut;

	// Final LayerNorm: [dModel]
	GpuBuffer<float> lnFinalGamma;
	GpuBuffer<float> lnFinalBeta;
	GpuBuffer<float> mLnFinalGamma;
	GpuBuffer<float> v2LnFinalGamma;
	GpuBuffer<float> mLnFinalBeta;
	GpuBuffer<float> v2LnFinalBeta;
	GpuBuffer<float> gLnFinalGamma;
	GpuBuffer<float> gLnFinalBeta;

	// Per-layer block weights.
	struct Block
	{
		// Pre-LN 1
		GpuBuffer<float> ln1Gamma;   // [dModel]
		GpuBuffer<float> ln1Beta;    // [dModel]
		GpuBuffer<float> mLn1Gamma;
		GpuBuffer<float> v2Ln1Gamma;
		GpuBuffer<float> mLn1Beta;
		GpuBuffer<float> v2Ln1Beta;
		GpuBuffer<float> gLn1Gamma;
		GpuBuffer<float> gLn1Beta;

		// QKV + output projections: [dModel, dModel] or [dModel, dModelKV]
		GpuBuffer<float> Wq, Wk, Wv, Wo;
		GpuBuffer<float> vWq, vWk, vWv, vWo;
		GpuBuffer<float> v2Wq, v2Wk, v2Wv, v2Wo;
		GpuBuffer<float> gWq, gWk, gWv, gWo;
		GpuBuffer<uint16_t> WqLowp, WkLowp, WvLowp, WoLowp;
		// BF16 Adam state (used when adamStateBf16=true, saves ~2x VRAM).
		GpuBuffer<uint16_t> vWq_bf16, vWk_bf16, vWv_bf16, vWo_bf16;
		GpuBuffer<uint16_t> v2Wq_bf16, v2Wk_bf16, v2Wv_bf16, v2Wo_bf16;
		GpuBuffer<float> bq, bk, bv, bo;     // [dModel] or [dModelKV]
		GpuBuffer<float> mBq, mBk, mBv, mBo;
		GpuBuffer<float> v2Bq, v2Bk, v2Bv, v2Bo;
		GpuBuffer<float> gBq, gBk, gBv, gBo;

		// Pre-LN 2
		GpuBuffer<float> ln2Gamma;   // [dModel]
		GpuBuffer<float> ln2Beta;    // [dModel]
		GpuBuffer<float> mLn2Gamma;
		GpuBuffer<float> v2Ln2Gamma;
		GpuBuffer<float> mLn2Beta;
		GpuBuffer<float> v2Ln2Beta;
		GpuBuffer<float> gLn2Gamma;
		GpuBuffer<float> gLn2Beta;

		// FFN: W1 [dFF or 2*dFF, dModel], W2 [dModel, dFF]
		GpuBuffer<float> W1, W2;
		GpuBuffer<float> vW1, vW2;
		GpuBuffer<float> v2W1, v2W2;
		GpuBuffer<uint16_t> vW1_bf16, vW2_bf16;
		GpuBuffer<uint16_t> v2W1_bf16, v2W2_bf16;
		GpuBuffer<float> gW1, gW2;
		GpuBuffer<uint16_t> W1Lowp, W2Lowp;
		GpuBuffer<float> b1, b2;     // [dFF or 2*dFF], [dModel]
		GpuBuffer<float> mB1, mB2;
		GpuBuffer<float> v2B1, v2B2;
		GpuBuffer<float> gB1, gB2;

		// ATLAS optimizer state (one per weight matrix)
		GpuAtlasWeightState atlasWq, atlasWk, atlasWv, atlasWo;
		GpuAtlasWeightState atlasW1, atlasW2;
		GpuVestaWeightState vestaWq, vestaWk, vestaWv, vestaWo;
		GpuVestaWeightState vestaW1, vestaW2;
		GpuHeliosWeightState heliosWq, heliosWk, heliosWv, heliosWo;
		GpuHeliosWeightState heliosW1, heliosW2;
		GpuEchoWeightState echoWq, echoWk, echoWv, echoWo;
		GpuEchoWeightState echoW1, echoW2;
		GpuBiMAPWeightState bimapWq, bimapWk, bimapWv, bimapWo;
		GpuBiMAPWeightState bimapW1, bimapW2;
		GpuPactWeightState pactWq, pactWk, pactWv, pactWo;
		GpuPactWeightState pactW1, pactW2;
		GpuRacerWeightState racerWq, racerWk, racerWv, racerWo;
		GpuRacerWeightState racerW1, racerW2;
		GpuMatraWeightState matraWq, matraWk, matraWv, matraWo;
		GpuMatraWeightState matraW1, matraW2;
		GpuArgosWeightState argosWq, argosWk, argosWv, argosWo;
		GpuArgosWeightState argosW1, argosW2;
		GpuMuonWeightState muonWq, muonWk, muonWv, muonWo;
		GpuMuonWeightState muonW1, muonW2;
	};

	Block* blocks;  // array of nLayers blocks

	// Persistent device arrays for batched Adam optimizer.
	// Pointer arrays (device arrays of float*): param, grad, m, v.
	float** d_adamParams;
	float** d_adamGrads;
	float** d_adamM;
	float** d_adamV;
	float** d_adamRowSecond;
	float** d_adamColSecond;
	float** d_adamRowMetric;
	float** d_adamColMetric;
	float** d_adamRowStructMetric;
	float** d_adamColStructMetric;
	float** d_adamPrevMhat;
	float** d_adamMetricScratch;
	GpuEchoObserveEntry* d_echoObserveEntries;
	GpuMatraBatchItem* d_matraBatchItems; // device scratch array of MATRA batch descriptors
	float* d_matraStatsBatch; // device scratch array of packed MATRA scalar stats [batch, 20]
	float** d_matraCoreBatchPtrs; // device scratch array of MATRA coreScratch pointers for batched factorization
	float** d_matraStepBatchPtrs; // device scratch array of MATRA orthStep pointers for batched triangular solves
	int* d_matraInfoBatch; // device scratch array of batched MATRA Cholesky status codes
	GpuMuonBatchItem* d_muonBatchItems; // device scratch array of MUON batch descriptors
	float** d_muonCoreBatchPtrs; // device scratch array of coreScratch pointers for batched MUON factorization
	float** d_muonStepBatchPtrs; // device scratch array of muonStep pointers for batched MUON triangular solves
	int* d_muonInfoBatch; // device scratch array of batched MUON Cholesky status codes
	// Per-group scalars (device arrays of float): static base lr and wd.
	float* d_adamBaseLr;
	float* d_adamWd;
	float* d_adamGroupScales;
	float* d_adamGroupPrevStepRms;
	// Per-group element counts (device array of int).
	int* d_adamSizes;
	int* d_adamMetricRows;
	int* d_adamMetricCols;
	int adamGroupCount;   // number of parameter groups
	int adamMaxSize;      // largest element count across groups
	int echoObserveCapacity; // capacity of the batched ECHO observe descriptor buffer
	int matraCoreBatchCapacity; // capacity of the batched MATRA core pointer scratch array
	int muonCoreBatchCapacity; // capacity of the batched MUON core pointer scratch array
	int echoObserveEntryCount; // cached descriptor count for the current scratch shape
	int echoObserveTotalFeatures; // total row+col features across cached observe descriptors
	unsigned int echoObserveSeqLen; // sequence length used to build cached descriptors
	unsigned int echoObserveScope; // ECHO scope used to build cached descriptors
	bool echoObserveTokenModel; // token-lm vs projection mode for cached descriptors
	bool echoObserveMetaUploaded; // true after cached observe descriptors uploaded
	bool matraBatchDescriptorsUploaded; // true after MATRA static batch descriptors uploaded
	bool adamPtrsUploaded; // true after pointer arrays uploaded once
	bool adamMetricMetaUploaded; // true after static ECHO metric metadata uploaded
	unsigned int adamMetricScope; // ECHO scope for the uploaded static metric metadata
	int matraBatchDescriptorCount; // cached MATRA descriptor count
	unsigned long long matraBatchDescriptorHash; // cached hash of uploaded MATRA descriptor layout

	// ATLAS optimizer state (one per weight matrix, biases use Adam).
	GpuAtlasWeightState atlasTokE;
	GpuAtlasWeightState atlasWIn;
	GpuAtlasWeightState atlasWOut;
	GpuVestaWeightState vestaTokE;
	GpuVestaWeightState vestaWIn;
	GpuVestaWeightState vestaWOut;
	GpuHeliosWeightState heliosTokE;
	GpuHeliosWeightState heliosWIn;
	GpuHeliosWeightState heliosWOut;
	GpuEchoWeightState echoTokE;
	GpuEchoWeightState echoWIn;
	GpuEchoWeightState echoWOut;
	GpuBiMAPWeightState bimapTokE;
	GpuBiMAPWeightState bimapWIn;
	GpuBiMAPWeightState bimapWOut;
	GpuPactWeightState pactTokE;
	GpuPactWeightState pactWIn;
	GpuPactWeightState pactWOut;
	GpuRacerWeightState racerTokE;
	GpuRacerWeightState racerWIn;
	GpuRacerWeightState racerWOut;
	GpuMatraWeightState matraTokE;
	GpuMatraWeightState matraWIn;
	GpuMatraWeightState matraWOut;
	GpuArgosWeightState argosTokE;
	GpuArgosWeightState argosWIn;
	GpuArgosWeightState argosWOut;
	GpuMuonWeightState muonTokE;
	GpuMuonWeightState muonWIn;
	GpuMuonWeightState muonWOut;

	// BF16 mixed-precision state.
	// When mixed precision is enabled (training_config.mixedPrecision.enable),
	// the *Lowp buffers above hold a BF16 mirror of every major weight matrix,
	// refreshed from the FP32 master after every optimizer step. Inference and
	// BF16 matmul paths read from the Lowp mirrors; gradients and master
	// weights stay FP32 throughout.
	bool lowpReady;      // true after ensureLowpMirrors has populated all Lowp buffers
	int  lowpDType;      // glades::transformer_kernels::LOWP_BF16 (others unsupported on GPU for now)

	GpuTransformerWeights();
	~GpuTransformerWeights();

	// Allocate all GPU buffers for the given model config.
	// When skipAdamBufs is true, Adam moment buffers (v*/v2*/m*) are not
	// allocated on GPU.  Used when the optimizer is ATLAS (which maintains
	// its own per-matrix state) to avoid wasting ~2x model-size in VRAM.
	// When adamStateBf16 is true, the m/v moments for the 9 large weight
	// matrices (tokE, WIn, WOut, Wq/Wk/Wv/Wo/W1/W2 per block) are stored
	// in BF16 (uint16_t) instead of FP32, halving their VRAM cost. Biases
	// and LN params keep FP32 state (their size is negligible).
	bool allocate(unsigned int dModel, unsigned int dFF, unsigned int nHeads,
	              unsigned int nKVHeads, unsigned int nLayers,
	              unsigned int vocabSize, unsigned int inputSize, unsigned int outSize,
	              unsigned int ffnKind, bool tokenModel, bool tieEmbeddings,
	              bool skipAdamBufs = false,
	              bool adamStateBf16 = false);

	// Free all GPU memory.
	void free();

	// Allocate ECHO-specific batched observe / metric metadata buffers on demand.
	// Plain AdamW and non-ECHO ATLAS variants do not need these arrays.
	bool ensureEchoBuffers();

	// Allocate every BF16 Lowp mirror (if not already sized) and populate each
	// one from its FP32 master via the on-device cast kernel. Should be called
	// once after the FP32 weights have been uploaded (ensureGpuState) and then
	// after every optimizer step while mixed precision is enabled.
	// Returns false if any device allocation or cast kernel dispatch fails.
	bool ensureLowpMirrors();
};

// GPU-resident forward/backward scratch buffers for transformer training.
// Layout mirrors NNetwork::TransformerScratch.
struct GpuTransformerScratch
{
	bool initialized;

	unsigned int T;
	unsigned int dModel;
	unsigned int dFF;
	unsigned int dModelKV;
	unsigned int nHeads;
	unsigned int nLayers;
	unsigned int inputSize;
	unsigned int outSize;
	unsigned int ff1Width;

	// Forward scratch
	GpuBuffer<float> x;          // [T, inputSize]
	GpuBuffer<float> h;          // [T, dModel]
	GpuBuffer<float> ln1Mean;    // [nLayers, T]
	GpuBuffer<float> ln1InvStd;  // [nLayers, T]
	GpuBuffer<float> x1;         // [nLayers, T, dModel]
	GpuBuffer<float> Q;          // [nLayers, T, dModel]
	GpuBuffer<float> K;          // [nLayers, T, dModelKV]
	GpuBuffer<float> V;          // [nLayers, T, dModelKV]
	GpuBuffer<float> attnConcat; // [nLayers, T, dModel]
	GpuBuffer<float> attnOut;    // [nLayers, T, dModel]
	GpuBuffer<float> hAfterAttn; // [nLayers, T, dModel]
	GpuBuffer<float> ln2Mean;    // [nLayers, T]
	GpuBuffer<float> ln2InvStd;  // [nLayers, T]
	GpuBuffer<float> x2;         // [nLayers, T, dModel]
	GpuBuffer<float> ff1;        // [nLayers, T, ff1Width]
	GpuBuffer<float> ff1Act;     // [nLayers, T, dFF]
	GpuBuffer<float> ffOut;      // [nLayers, T, dModel]
	GpuBuffer<float> hAfterFF;   // [nLayers, T, dModel]
	GpuBuffer<float> hPostFinalLN; // [T, dModel]
	GpuBuffer<float> lnFinalMean;  // [T]
	GpuBuffer<float> lnFinalInvStd; // [T]
	GpuBuffer<float> logits;     // [T, outSize]
	GpuBuffer<float> probs;      // [T, outSize]

	// Backward scratch
	GpuBuffer<float> dLogits;    // [T, outSize]
	GpuBuffer<float> dH;         // [T, dModel]
	GpuBuffer<float> dH2;        // [T, dModel]
	GpuBuffer<float> dFF1Act;    // [T, dFF]
	GpuBuffer<float> dFF1Cat;    // [T, ff1Width]
	GpuBuffer<float> dX2;        // [T, dModel]
	GpuBuffer<float> dHAfterAttnFromLN; // [T, dModel]
	GpuBuffer<float> dAttnConcat; // [T, dModel]
	GpuBuffer<float> dQfull;     // [T, dModel]
	GpuBuffer<float> dKfull;     // [T, dModelKV]
	GpuBuffer<float> dVfull;     // [T, dModelKV]
	GpuBuffer<float> dX1;        // [T, dModel]
	GpuBuffer<float> dXtmp;      // [T, dModel]
	GpuBuffer<float> dHInFromLN; // [T, dModel]
	GpuBuffer<float> dInput;     // [T, inputSize]

	// Token IDs (for embedding gather/scatter)
	GpuBuffer<int> tokenIds;     // [T]

	// Persistent buffers to avoid per-step allocations
	GpuBuffer<float> gpuInvFreq; // [dHead/2]  (RoPE inverse frequencies)
	GpuBuffer<int> gpuTargetsT;  // [T]        (target token IDs for loss/backward)

	// BF16 activation staging for mixed-precision GEMMs. Sized to T*max(dModel,
	// ff1Width) so a single buffer can hold any single activation tile. Used
	// at forward/backward GEMM call sites to cast FP32 activations to BF16
	// right before feeding cublasGemmEx with BF16 weight mirrors.
	GpuBuffer<uint16_t> activationLowp;
	// Secondary BF16 activation staging used when two distinct activations
	// must be live at once (e.g. dY and X for the weight-grad GEMM).
	GpuBuffer<uint16_t> activationLowp2;
	// BF16 scratches for Q/K/V that feed flash_attention_multihead_forward_bf16.
	// Only live for the current layer's attention forward; cast from the FP32
	// Q/K/V buffers right before the attention call.
	GpuBuffer<uint16_t> qLowp;
	GpuBuffer<uint16_t> kLowp;
	GpuBuffer<uint16_t> vLowp;

	// Full attention scores matrix [nHeads, T, T] used by the
	// cuBLAS-tiled flash_attention path (research/WMMA_ATTENTION_PLAN.md).
	// Trades O(nH*T^2) memory for tensor-core throughput.  Allocated
	// lazily on first use when the fast path is eligible (nHeads == nKVHeads).
	// Left unallocated (empty) otherwise to keep VRAM budget unchanged.
	GpuBuffer<float> attnScoresScratch;

	// Second [nHeads, T, T] scratch used by the cuBLAS-tiled
	// flash_attention BACKWARD kernel for the dP intermediate.
	// Allocated lazily.
	GpuBuffer<float> attnDPScratch;

	// BF16 scratches for the cuBLAS-tiled BF16 flash-attention variant
	// (research/WMMA_ATTENTION_PLAN.md — doubles the attention GEMM
	// throughput by running on BF16 tensor cores).  Q/K/V are cast
	// from the FP32 inputs once per forward.  P is cast after softmax.
	// Allocated lazily and shared across the FWD path; the BWD path
	// still uses the FP32 cuBLAS variant for numerical safety on dP.
	GpuBuffer<uint16_t> attnQbf16;      // [T, dModel]
	GpuBuffer<uint16_t> attnKbf16;      // [T, dModelKV]
	GpuBuffer<uint16_t> attnVbf16;      // [T, dModelKV]
	GpuBuffer<uint16_t> attnPbf16;      // [nHeads, T, T]

	// GPU loss computation scalars
	GpuBuffer<float> lossSum;    // [1]
	GpuBuffer<int> lossCount;    // [1]  (valid token count)
	GpuBuffer<int> correctCount; // [1]  (argmax matches)
	GpuBuffer<int> validCount;   // [1]  (valid tokens for accuracy)
	GpuBuffer<int> lossPack;     // [4]  (packed loss scalars for single D2H download)

	// Persistent device arrays for batch-zeroing dK/dV (2 pointers + 2 sizes).
	// Raw device pointers (not GpuBuffer) to avoid needing a float* specialization.
	float** d_dKdVZeroPtrs;  // device array of 2 float*
	int*    d_dKdVZeroSizes; // device array of 2 ints

	GpuTransformerScratch();
	~GpuTransformerScratch();

	bool allocate(unsigned int T, unsigned int inputSize, unsigned int outSize,
	              unsigned int dModel, unsigned int dFF, unsigned int dModelKV,
	              unsigned int nHeads, unsigned int nLayers, unsigned int ff1Width);
	void free();
};

bool ensureTransformerScratch(GpuTransformerScratch*& scratch,
                              const TransformerGpuScratchConfig& cfg);

// Upload CPU TensorTransformerState weights -> GPU.
// Assumes gpu weights are already allocated with matching dimensions.
// The cpu_* parameters are pointers to the CPU-side weight arrays.
// Returns true on success.
bool uploadTransformerWeights(GpuTransformerWeights& gpu,
                               const float* tokE, size_t tokESize,
                               const float* WIn, size_t WInSize,
                               const float* bIn, size_t bInSize,
                               const float* WOut, size_t WOutSize,
                               const float* bOut, size_t bOutSize,
                               const float* lmBias, size_t lmBiasSize,
                               const float* lnFinalGamma, size_t lnFinalGammaSize,
                               const float* lnFinalBeta, size_t lnFinalBetaSize);

// Download GPU weights -> CPU arrays.
bool downloadTransformerWeights(const GpuTransformerWeights& gpu,
                                 float* tokE, size_t tokESize,
                                 float* WIn, size_t WInSize,
                                 float* bIn, size_t bInSize,
                                 float* WOut, size_t WOutSize,
                                 float* bOut, size_t bOutSize,
                                 float* lmBias, size_t lmBiasSize,
                                 float* lnFinalGamma, size_t lnFinalGammaSize,
                                 float* lnFinalBeta, size_t lnFinalBetaSize);

// Upload/download a single block's weights.
bool uploadTransformerBlockWeights(GpuTransformerWeights::Block& gpuBlock,
                                    unsigned int dModel, unsigned int dModelKV,
                                    unsigned int ff1Width, unsigned int dFF,
                                    const float* ln1Gamma, const float* ln1Beta,
                                    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                                    const float* bq, const float* bk, const float* bv, const float* bo,
                                    const float* ln2Gamma, const float* ln2Beta,
                                    const float* W1, const float* W2,
                                    const float* b1, const float* b2);

bool uploadTransformerTokenIds(GpuTransformerScratch& scratch,
                               const int* tokenIds, size_t count);
bool uploadTransformerDenseInputs(GpuTransformerScratch& scratch,
                                  const float* hostInputs, size_t count);
bool uploadTransformerRopeInvFreq(GpuTransformerScratch& scratch,
                                  const float* invFreq, size_t count);
bool downloadTransformerWeightsToHost(const GpuTransformerWeights& gpu,
                                      const TransformerHostWeightsView& host);

// Zero all gradient buffers on GPU.
bool zeroTransformerGradients(GpuTransformerWeights& gpu);

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

struct GpuTransformerWeights
{
	bool initialized;
	GpuTransformerWeights() : initialized(false) {}
};

struct GpuTransformerScratch
{
	bool initialized;
	GpuTransformerScratch() : initialized(false) {}
};

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
