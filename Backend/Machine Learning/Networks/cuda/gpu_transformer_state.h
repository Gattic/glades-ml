// GPU mirror of TensorTransformerState + TransformerScratch.
//
// Keeps all weights, optimizer state, and forward/backward scratch buffers
// GPU-resident so only inputs/outputs cross PCIe.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

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
	GpuBuffer<float> bIn;    // [dModel]
	GpuBuffer<float> mBIn;
	GpuBuffer<float> v2BIn;
	GpuBuffer<float> gBIn;

	// Output projection: [outSize, dModel]
	GpuBuffer<float> WOut;
	GpuBuffer<float> vWOut;
	GpuBuffer<float> v2WOut;
	GpuBuffer<float> gWOut;
	GpuBuffer<float> bOut;   // [outSize]
	GpuBuffer<float> mBOut;
	GpuBuffer<float> v2BOut;
	GpuBuffer<float> gBOut;

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
		GpuBuffer<float> gW1, gW2;
		GpuBuffer<float> b1, b2;     // [dFF or 2*dFF], [dModel]
		GpuBuffer<float> mB1, mB2;
		GpuBuffer<float> v2B1, v2B2;
		GpuBuffer<float> gB1, gB2;
	};

	Block* blocks;  // array of nLayers blocks

	// Persistent device arrays for batched Adam optimizer.
	// Pointer arrays (device arrays of float*): param, grad, m, v.
	float** d_adamParams;
	float** d_adamGrads;
	float** d_adamM;
	float** d_adamV;
	// Per-group scalars (device arrays of float): lr, wd.
	float* d_adamLr;
	float* d_adamWd;
	// Per-group element counts (device array of int).
	int* d_adamSizes;
	int adamGroupCount;   // number of parameter groups
	int adamMaxSize;      // largest element count across groups
	bool adamPtrsUploaded; // true after pointer arrays uploaded once

	GpuTransformerWeights();
	~GpuTransformerWeights();

	// Allocate all GPU buffers for the given model config.
	bool allocate(unsigned int dModel, unsigned int dFF, unsigned int nHeads,
	              unsigned int nKVHeads, unsigned int nLayers,
	              unsigned int vocabSize, unsigned int inputSize, unsigned int outSize,
	              unsigned int ffnKind, bool tokenModel, bool tieEmbeddings);

	// Free all GPU memory.
	void free();
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

	// Attention scores/probs (materialized for batched GEMM attention path)
	GpuBuffer<float> attnScores; // [nHeads * T * T]
	GpuBuffer<float> attnProbs;  // [nHeads * T * T]

	// Persistent buffers to avoid per-step allocations
	GpuBuffer<float> gpuInvFreq; // [dHead/2]  (RoPE inverse frequencies)
	GpuBuffer<int> gpuTargetsT;  // [T]        (target token IDs for loss/backward)

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
                               const float* lmBias, size_t lmBiasSize);

// Download GPU weights -> CPU arrays.
bool downloadTransformerWeights(const GpuTransformerWeights& gpu,
                                 float* tokE, size_t tokESize,
                                 float* WIn, size_t WInSize,
                                 float* bIn, size_t bInSize,
                                 float* WOut, size_t WOutSize,
                                 float* bOut, size_t bOutSize,
                                 float* lmBias, size_t lmBiasSize);

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
