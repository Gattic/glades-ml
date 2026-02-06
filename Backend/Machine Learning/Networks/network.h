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
#ifndef _NNETWORK
#define _NNETWORK

#include "Backend/Database/GPointer.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "../State/Terminator.h"
#include "../GMath/cmatrix.h"
#include "../Structure/nninfo.h" // ensure NNInfo is a complete type for ownedSkeleton deletion
#include "../rng.h"
#include "training_callbacks.h"
#include "training_config.h"
#include "../nnetwork_status.h"
#include "bayes.h"
#include "transformer_ops.h"
#include "aligned_allocator.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <stdint.h>

// Concurrency primitives:
// Prefer standard C++ atomics when available; fall back to legacy builtins otherwise.
#if __cplusplus >= 201103L
#include <atomic>
#define GLADES_HAVE_STD_ATOMICS 1
#else
#define GLADES_HAVE_STD_ATOMICS 0
#endif

class Point2;

namespace shmea {
class GTable;
class GLogger;
};

namespace GNet {
class GServer;
class Connection;
};

namespace glades {

class DataInput;
class CMatrix;
class MetaNetwork;
class TrainingCore;
class Trainer;

class NNetwork
{
private:
	friend MetaNetwork;
	friend TrainingCore;
	friend Trainer;

	// Tensor-based DFF training state.
	//
	// This is a contiguous-buffer rewrite of the historical (graph-based) training core,
	// but implemented purely in packed vectors/matrices for cache-friendly execution.
	struct TensorDFFState
	{
		bool initialized;
		// Layer sizes including input and output: [in, h1, ..., hH, out]
		std::vector<unsigned int> sizes;

		// Per-transition weights (mapping sizes[t] -> sizes[t+1]) and optim state.
		// W is row-major [out][in].
		struct Transition
		{
			unsigned int in;
			unsigned int out;
			std::vector<float> W;
			std::vector<float> vW; // momentum/velocity
			// Per-neuron bias (stored in the Node graph as the final edge weight).
			// bias[j] corresponds to output unit j for this transition.
			std::vector<float> bias;

			// Accumulated gradients for current minibatch
			std::vector<float> gW;
			std::vector<float> gBias;

			Transition() : in(0), out(0) {}
		};

		std::vector<Transition> T;

		// Cached activations and deltas for a single sample (forward/backward).
		std::vector< std::vector<float> > a;      // a[layerIndex][i]
		std::vector< std::vector<float> > delta;  // delta for non-input layers; delta[li] aligns with a[li]

		// Minibatch accumulation count.
		unsigned int batchCount;

		TensorDFFState() : initialized(false), batchCount(0) {}

		void reset()
		{
			initialized = false;
			sizes.clear();
			T.clear();
			a.clear();
			delta.clear();
			batchCount = 0;
		}
	};

	// Tensorized recurrent training state (RNN/GRU/LSTM).
	//
	// These store parameters in packed contiguous arrays so the recurrent SGD helpers can run
	// mostly on cache-friendly kernels rather than pointer chasing through heap objects.
	// Momentum and weight decay semantics match the DFF tensor path (v = mf*v + lr*g; w -= v).
	struct TensorRNNState
	{
		bool initialized;
		unsigned int inputSize;
		unsigned int outSize;
		std::vector<unsigned int> hiddenSizes;

		struct Hidden
		{
			unsigned int in;
			unsigned int h;
			// Wxh: [h, in], Whh: [h, h] row-major
			std::vector<float> Wxh;
			std::vector<float> Whh;
			std::vector<float> vWxh;
			std::vector<float> vWhh;
			std::vector<float> gWxh;
			std::vector<float> gWhh;
			// Per-unit bias
			std::vector<float> bias;
			std::vector<float> gBias;
			Hidden() : in(0u), h(0u) {}
		};

		struct Out
		{
			unsigned int in;
			unsigned int out;
			// Why: [out, in]
			std::vector<float> Why;
			std::vector<float> vWhy;
			std::vector<float> gWhy;
			std::vector<float> bias;
			std::vector<float> gBias;
			Out() : in(0u), out(0u) {}
		};

		std::vector<Hidden> H;
		Out O;

		TensorRNNState() : initialized(false), inputSize(0u), outSize(0u) {}

		void reset()
		{
			initialized = false;
			inputSize = 0u;
			outSize = 0u;
			hiddenSizes.clear();
			H.clear();
			O = Out();
		}
	};

	struct TensorGatedState
	{
		bool initialized;
		unsigned int inputSize;
		unsigned int outSize;
		unsigned int gateCount;
		std::vector<unsigned int> hiddenSizes;

		struct Hidden
		{
			unsigned int in;
			unsigned int h;
			// Packed W: [gateCount, h, in] and U: [gateCount, h, h] row-major
			// Bias: [gateCount, h]
			std::vector<float> W;
			std::vector<float> U;
			std::vector<float> vW;
			std::vector<float> vU;
			std::vector<float> gW;
			std::vector<float> gU;
			std::vector<float> bias;
			std::vector<float> gBias;
			Hidden() : in(0u), h(0u) {}
		};

		struct Out
		{
			unsigned int in;
			unsigned int out;
			std::vector<float> Why;
			std::vector<float> vWhy;
			std::vector<float> gWhy;
			std::vector<float> bias;
			std::vector<float> gBias;
			Out() : in(0u), out(0u) {}
		};

		std::vector<Hidden> H;
		Out O;

		TensorGatedState(unsigned int g = 1u) : initialized(false), inputSize(0u), outSize(0u), gateCount(g) {}

		void reset()
		{
			initialized = false;
			inputSize = 0u;
			outSize = 0u;
			hiddenSizes.clear();
			H.clear();
			O = Out();
		}
	};

	TensorRNNState tensorRnn;
	// GRU: gateCount=3, LSTM: gateCount=4
	TensorGatedState tensorGru;
	TensorGatedState tensorLstm;

	// Reusable scratch buffers for recurrent (RNN/GRU/LSTM) forward/backward passes.
	// This avoids per-window nested-vector allocations in the hot path.
	struct RecurrentScratch
	{
		unsigned int winLen;
		unsigned int inputSize;
		unsigned int outSize;
		std::vector<unsigned int> hiddenSizes;

		// Common buffers
		std::vector<float> x; // [winLen, inputSize]
		std::vector<float> y; // [winLen, outSize]
		std::vector<float> outLogits; // [outSize]
		std::vector<float> outProbs;  // [outSize]

		// Per hidden layer buffers (flattened [winLen, hiddenSize])
		std::vector< std::vector<float> > h;
		std::vector< std::vector<float> > hPrevAtT;

		// GRU-only buffers
		std::vector< std::vector<float> > z;
		std::vector< std::vector<float> > r;
		std::vector< std::vector<float> > hTilde;

		// LSTM-only buffers
		std::vector< std::vector<float> > c;
		std::vector< std::vector<float> > cPrevAtT;
		std::vector< std::vector<float> > iGate;
		std::vector< std::vector<float> > fGate;
		std::vector< std::vector<float> > oGate;
		std::vector< std::vector<float> > gGate;
		std::vector< std::vector<float> > tanhC;

		RecurrentScratch() : winLen(0u), inputSize(0u), outSize(0u) {}

		static void resizeAndZero(std::vector<float>& v, size_t n)
		{
			if (v.size() != n)
				v.resize(n);
			std::fill(v.begin(), v.end(), 0.0f);
		}

		static void resizeAndZero2D(std::vector< std::vector<float> >& vv, size_t rows, const std::vector<unsigned int>& widths, unsigned int winLen)
		{
			if (vv.size() != rows)
				vv.resize(rows);
			for (size_t i = 0; i < rows; ++i)
			{
				const size_t n = static_cast<size_t>(winLen) * static_cast<size_t>(widths[i]);
				resizeAndZero(vv[i], n);
			}
		}

		void ensureCommon(unsigned int newWinLen,
		                  unsigned int newInputSize,
		                  unsigned int newOutSize,
		                  const std::vector<unsigned int>& newHiddenSizes)
		{
			winLen = newWinLen;
			inputSize = newInputSize;
			outSize = newOutSize;
			hiddenSizes = newHiddenSizes;

			resizeAndZero(x, static_cast<size_t>(winLen) * static_cast<size_t>(inputSize));
			resizeAndZero(y, static_cast<size_t>(winLen) * static_cast<size_t>(outSize));

			// Softmax temps (reused per timestep)
			if (outLogits.size() != outSize)
				outLogits.resize(outSize);
			if (outProbs.size() != outSize)
				outProbs.resize(outSize);
		}

		void ensureRNN(unsigned int newWinLen,
		               unsigned int newInputSize,
		               unsigned int newOutSize,
		               const std::vector<unsigned int>& newHiddenSizes)
		{
			ensureCommon(newWinLen, newInputSize, newOutSize, newHiddenSizes);
			resizeAndZero2D(h, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(hPrevAtT, newHiddenSizes.size(), newHiddenSizes, winLen);
		}

		void ensureGRU(unsigned int newWinLen,
		               unsigned int newInputSize,
		               unsigned int newOutSize,
		               const std::vector<unsigned int>& newHiddenSizes)
		{
			ensureRNN(newWinLen, newInputSize, newOutSize, newHiddenSizes);
			resizeAndZero2D(z, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(r, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(hTilde, newHiddenSizes.size(), newHiddenSizes, winLen);
		}

		void ensureLSTM(unsigned int newWinLen,
		                unsigned int newInputSize,
		                unsigned int newOutSize,
		                const std::vector<unsigned int>& newHiddenSizes)
		{
			ensureRNN(newWinLen, newInputSize, newOutSize, newHiddenSizes);
			resizeAndZero2D(c, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(cPrevAtT, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(iGate, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(fGate, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(oGate, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(gGate, newHiddenSizes.size(), newHiddenSizes, winLen);
			resizeAndZero2D(tanhC, newHiddenSizes.size(), newHiddenSizes, winLen);
		}
	};

	CMatrix confusionMatrix;
	GNet::GServer* serverInstance;
	GNet::Connection* cConnection;
	// Optional logger override. If NULL, use the attached server logger (if any),
	// otherwise fall back to a default logger instance.
	shmea::GLogger* loggerOverride;
	glades::NaiveBayes bModel;

	bool running;
	int netType;
	int epochs;
	bool saveInstance;
	float overallTotalError;
	float overallTotalAccuracy;
	float overallClassAccuracy;
	float overallClassPrecision;
	float overallClassRecall;
	float overallClassSpecificity;
	float overallClassF1;
	int minibatchSize;
	int64_t id;
	uint64_t rngSeed;
	// Per-network RNG engine. Training installs this as the "current" RNG via a scoped guard
	// so random draws are deterministic per-network.
	glades::rng::Engine rngEngine;
	// Concurrency guard for train/test runs.
	//
	// Policy:
	// - A single NNetwork instance is NOT re-entrant: do not call train()/test() concurrently
	//   on the same object from multiple threads.
	// - Different NNetwork instances MAY be run concurrently in different threads (subject to
	//   DataInput thread-safety; see determinism/concurrency policy docs).
	//
	// This lock prevents two threads from mutating shared scratch buffers / tensor state / metrics.
	// Implementation:
	// - C++11+: std::atomic_flag spinlock (portable)
	// - pre-C++11: volatile int with GCC/Clang __sync builtins (legacy)
#if GLADES_HAVE_STD_ATOMICS
	std::atomic_flag runLock;
#else
	volatile int runLock;
#endif

	// Low-level lock primitives (implemented in network.cpp; GCC/Clang use atomic builtins).
	bool tryAcquireRunLock();
	void releaseRunLock();

	// Epoch-scoped metric accumulators (reset at the start of each epoch).
	// Regression:
	// - SSE/SAE across all samples and outputs (used to compute MSE/MAE/RMSE/R^2).
	// - SumY/SumY2 used to compute SST without a second pass.
	double regSSE;
	double regSAE;
	double regSumY;
	double regSumY2;
	unsigned long long regCount;

	// Classification/KL: top-1 accuracy across samples.
	unsigned long long clsCorrect;
	unsigned long long clsTotal;

	bool firstRunActivation;
	NNetworkStatus lastStatus;
	TensorDFFState tensorDff;
	RecurrentScratch recScratch;

	// === Transformer (encoder/decoder) packed parameters ===
	//
	// Transformer models are sequence models, like recurrent nets, but without recurrence.
	// We store all parameters in packed vectors and run explicit forward/backward kernels.
	struct TensorTransformerState
	{
		bool initialized;
		// Dataset shapes
		unsigned int inputSize; // featureCount per timestep
		unsigned int outSize;   // expected output size per timestep

		// Model config
		unsigned int dModel;
		unsigned int dFF;
		unsigned int nHeads;
		// Grouped-query attention: number of KV heads (<= nHeads).
		// If equal to nHeads, this is standard multi-head attention.
		unsigned int nKVHeads;
		unsigned int nLayers;
		// If true, use causal self-attention (decoder-only / autoregressive).
		bool causal;
		// FFN kind (see TransformerRunConfig::FFNKind). Stored for shape consistency.
		unsigned int ffnKind;

		// Language-model (token) mode: embedding + vocab head.
		bool tokenModel;
		unsigned int vocabSize;
		int padTokenId;
		bool tieEmbeddings;
		// Optimizer update step (used for Adam bias correction).
		unsigned long long optimizerStep;
		// Token embedding table E: [vocabSize, dModel]
		std::vector<float> tokE;
		// Low-precision weight copy of tokE used by mixed-precision training (optional).
		std::vector<uint16_t> tokELowp;
		std::vector<float> vTokE;
		std::vector<float> v2TokE;
		std::vector<float> gTokE;
		// LM head bias: [vocabSize]
		std::vector<float> lmBias;
		std::vector<float> mLmBias;
		std::vector<float> v2LmBias;
		std::vector<float> gLmBias;

		// Input projection: x[t,inputSize] -> h[t,dModel]
		// WIn: [dModel, inputSize]
		std::vector<float> WIn;
		// Low-precision weight copy of WIn used by mixed-precision training (optional).
		std::vector<uint16_t> WInLowp;
		std::vector<float> vWIn;
		std::vector<float> v2WIn;
		std::vector<float> gWIn;
		std::vector<float> bIn;   // [dModel]
		std::vector<float> mBIn;
		std::vector<float> v2BIn;
		std::vector<float> gBIn;

		// Output projection: h[t,dModel] -> y[t,outSize]
		// WOut: [outSize, dModel]
		std::vector<float> WOut;
		// Low-precision weight copy of WOut used by mixed-precision training (optional).
		std::vector<uint16_t> WOutLowp;
		std::vector<float> vWOut;
		std::vector<float> v2WOut;
		std::vector<float> gWOut;
		std::vector<float> bOut;  // [outSize]
		std::vector<float> mBOut;
		std::vector<float> v2BOut;
		std::vector<float> gBOut;

		struct Block
		{
			// Pre-LN 1
			std::vector<float> ln1Gamma; // [dModel]
			std::vector<float> ln1Beta;  // [dModel]
			std::vector<float> mLn1Gamma;
			std::vector<float> v2Ln1Gamma;
			std::vector<float> mLn1Beta;
			std::vector<float> v2Ln1Beta;
			std::vector<float> gLn1Gamma;
			std::vector<float> gLn1Beta;

			// Self-attention linear projections (packed as [dModel, dModel])
			std::vector<float> Wq, Wk, Wv, Wo;
			// Low-precision copies (optional; used when TrainingConfig::mixedPrecision.enable).
			std::vector<uint16_t> WqLowp, WkLowp, WvLowp, WoLowp;
			std::vector<float> vWq, vWk, vWv, vWo;
			std::vector<float> v2Wq, v2Wk, v2Wv, v2Wo;
			std::vector<float> gWq, gWk, gWv, gWo;
			std::vector<float> bq, bk, bv, bo; // [dModel]
			std::vector<float> mBq, mBk, mBv, mBo;
			std::vector<float> v2Bq, v2Bk, v2Bv, v2Bo;
			std::vector<float> gBq, gBk, gBv, gBo;

			// Pre-LN 2
			std::vector<float> ln2Gamma; // [dModel]
			std::vector<float> ln2Beta;  // [dModel]
			std::vector<float> mLn2Gamma;
			std::vector<float> v2Ln2Gamma;
			std::vector<float> mLn2Beta;
			std::vector<float> v2Ln2Beta;
			std::vector<float> gLn2Gamma;
			std::vector<float> gLn2Beta;

			// Feed-forward network
			// W1: [dFF, dModel], b1: [dFF]
			// W2: [dModel, dFF], b2: [dModel]
			std::vector<float> W1, W2;
			// Low-precision copies (optional; used when TrainingConfig::mixedPrecision.enable).
			std::vector<uint16_t> W1Lowp, W2Lowp;
			std::vector<float> vW1, vW2;
			std::vector<float> v2W1, v2W2;
			std::vector<float> gW1, gW2;
			std::vector<float> b1, b2;
			std::vector<float> mB1, mB2;
			std::vector<float> v2B1, v2B2;
			std::vector<float> gB1, gB2;
		};

		std::vector<Block> blocks;

		// === Mixed precision runtime state (Transformer training) ===
		//
		// Master weights remain the FP32 vectors above.
		// If mixed precision is enabled, the training path uses the low-precision weight copies
		// (tokELowp/WInLowp/WOutLowp + per-block lowp matrices) for forward/backward GEMMs/GEMVs.
		bool mpLowpReady;
		// Low-precision dtype selector (transformer_kernels::LOWP_F16 or transformer_kernels::LOWP_BF16).
		int mpLowpDType;
		// Dynamic loss scaling state.
		float mpLossScale;
		int mpLossScaleGoodSteps;

		TensorTransformerState()
		    : initialized(false),
		      inputSize(0u),
		      outSize(0u),
		      dModel(0u),
		      dFF(0u),
		      nHeads(0u),
		      nKVHeads(0u),
		      nLayers(0u),
		      causal(false)
		      ,
		      ffnKind(0u)
		      ,
		      tokenModel(false),
		      vocabSize(0u),
		      padTokenId(-1),
		      tieEmbeddings(true),
		      optimizerStep(0ULL),
		      mpLowpReady(false),
		      mpLowpDType(0),
		      mpLossScale(1.0f),
		      mpLossScaleGoodSteps(0)
		{
		}

		void reset()
		{
			initialized = false;
			inputSize = 0u;
			outSize = 0u;
			dModel = 0u;
			dFF = 0u;
			nHeads = 0u;
			nKVHeads = 0u;
			nLayers = 0u;
			causal = false;
			ffnKind = 0u;
			tokenModel = false;
			vocabSize = 0u;
			padTokenId = -1;
			tieEmbeddings = true;
			optimizerStep = 0ULL;
			mpLowpReady = false;
			mpLowpDType = 0;
			mpLossScale = 1.0f;
			mpLossScaleGoodSteps = 0;
			tokE.clear(); tokELowp.clear(); vTokE.clear(); v2TokE.clear(); gTokE.clear();
			lmBias.clear(); mLmBias.clear(); v2LmBias.clear(); gLmBias.clear();
			WIn.clear(); WInLowp.clear(); vWIn.clear(); v2WIn.clear(); gWIn.clear();
			bIn.clear(); mBIn.clear(); v2BIn.clear(); gBIn.clear();
			WOut.clear(); WOutLowp.clear(); vWOut.clear(); v2WOut.clear(); gWOut.clear();
			bOut.clear(); mBOut.clear(); v2BOut.clear(); gBOut.clear();
			blocks.clear();
		}
	};

	struct TransformerScratch
	{
		unsigned int T;
		unsigned int inputSize;
		unsigned int outSize;
		unsigned int dModel;
		unsigned int dFF;
		// Width of K/V projections: nKVHeads * dHead.
		unsigned int dModelKV;
		unsigned int nHeads;
		unsigned int nLayers;
		// Width of FF1 pre-activation buffer (dFF for MLP, 2*dFF for SwiGLU).
		unsigned int ff1Width;

		// Common (aligned for SIMD-friendly kernels)
		std::vector<float, glades::AlignedAllocator<float, 64> > x; // [T, inputSize]
		std::vector<float, glades::AlignedAllocator<float, 64> > h; // [T, dModel] after input projection + pos enc

		// Per-layer caches (flattened by layer index)
		// LN1
		std::vector<float, glades::AlignedAllocator<float, 64> > ln1Mean;   // [nLayers, T]
		std::vector<float, glades::AlignedAllocator<float, 64> > ln1InvStd; // [nLayers, T]
		std::vector<float, glades::AlignedAllocator<float, 64> > x1;        // [nLayers, T, dModel] (LN1 output)
		// Q,K,V and attention
		std::vector<float, glades::AlignedAllocator<float, 64> > Q; // [nLayers, T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > K; // [nLayers, T, dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > V; // [nLayers, T, dModelKV]
		// Concatenated per-head attention outputs before Wo.
		// This is stored for backward so we don't need to cache/store full [T,T] attention probabilities.
		std::vector<float, glades::AlignedAllocator<float, 64> > attnConcat; // [nLayers, T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > attnOut;    // [nLayers, T, dModel] (after Wo)
		std::vector<float, glades::AlignedAllocator<float, 64> > hAfterAttn; // [nLayers, T, dModel] (residual)

		// LN2
		std::vector<float, glades::AlignedAllocator<float, 64> > ln2Mean;   // [nLayers, T]
		std::vector<float, glades::AlignedAllocator<float, 64> > ln2InvStd; // [nLayers, T]
		std::vector<float, glades::AlignedAllocator<float, 64> > x2;        // [nLayers, T, dModel] (LN2 output)
		// FFN caches
		std::vector<float, glades::AlignedAllocator<float, 64> > ff1;    // [nLayers, T, ff1Width] pre-activation
		std::vector<float, glades::AlignedAllocator<float, 64> > ff1Act; // [nLayers, T, dFF] post-activation
		std::vector<float, glades::AlignedAllocator<float, 64> > ffOut;  // [nLayers, T, dModel] (after W2)
		// Output of each block
		std::vector<float, glades::AlignedAllocator<float, 64> > hAfterFF; // [nLayers, T, dModel]

		// Output head
		std::vector<float, glades::AlignedAllocator<float, 64> > logits; // [T, outSize]
		std::vector<float, glades::AlignedAllocator<float, 64> > probs;  // [T, outSize] (softmax if needed)
		// Token LM sampled-softmax: per-timestep sampled token ids corresponding to logits/probs.
		// When sampled-softmax is enabled, logits/probs are sized [T, (1+K)] and tokenLmSampleIds
		// holds the vocabulary indices for each sampled column (col 0 is always the target id).
		std::vector<int> tokenLmSampleIds; // [T, outSize] (only used for token LM sampled-softmax)

		// === Backward scratch (reused across sequences/layers; aligned) ===
		// These buffers eliminate per-sequence/per-layer allocations in transformer backward.
		std::vector<float, glades::AlignedAllocator<float, 64> > dLogits; // [T, outSize]
		std::vector<float, glades::AlignedAllocator<float, 64> > dH;      // [T, dModel] upstream gradient
		std::vector<float, glades::AlignedAllocator<float, 64> > dH2;     // [T, dModel] secondary buffer
		// FFN backward temps
		std::vector<float, glades::AlignedAllocator<float, 64> > dFF1Act;          // [T, dFF]
		std::vector<float, glades::AlignedAllocator<float, 64> > dFF1Cat;          // [T, ff1Width]
		std::vector<float, glades::AlignedAllocator<float, 64> > dX2;              // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dHAfterAttnFromLN; // [T, dModel]
		// Attention/backprop temps
		std::vector<float, glades::AlignedAllocator<float, 64> > dAttnConcat; // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dQfull;      // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dKfull;      // [T, dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > dVfull;      // [T, dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > dX1;         // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dXtmp;       // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dHInFromLN;  // [T, dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > dInput;      // [T, inputSize]

		TransformerScratch()
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

		template <typename VecT>
		static void resize_and_zero(VecT& v, size_t n)
		{
			if (v.size() != n)
				v.resize(n);
			std::fill(v.begin(), v.end(), 0.0f);
		}

		void ensure(unsigned int newT,
		            unsigned int newInputSize,
		            unsigned int newOutSize,
		            unsigned int newDModel,
		            unsigned int newDFF,
		            unsigned int newDModelKV,
		            unsigned int newNHeads,
		            unsigned int newNLayers,
		            unsigned int newFF1Width)
		{
			T = newT;
			inputSize = newInputSize;
			outSize = newOutSize;
			dModel = newDModel;
			dFF = newDFF;
			dModelKV = newDModelKV;
			nHeads = newNHeads;
			nLayers = newNLayers;
			ff1Width = newFF1Width;

			resize_and_zero(x, static_cast<size_t>(T) * static_cast<size_t>(inputSize));
			resize_and_zero(h, static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(ln1Mean, static_cast<size_t>(nLayers) * static_cast<size_t>(T));
			resize_and_zero(ln1InvStd, static_cast<size_t>(nLayers) * static_cast<size_t>(T));
			resize_and_zero(x1, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(Q, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(K, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
			resize_and_zero(V, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));

			resize_and_zero(attnConcat, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(attnOut, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(hAfterAttn, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(ln2Mean, static_cast<size_t>(nLayers) * static_cast<size_t>(T));
			resize_and_zero(ln2InvStd, static_cast<size_t>(nLayers) * static_cast<size_t>(T));
			resize_and_zero(x2, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(ff1, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
			resize_and_zero(ff1Act, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dFF));
			resize_and_zero(ffOut, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(hAfterFF, static_cast<size_t>(nLayers) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

			resize_and_zero(logits, static_cast<size_t>(T) * static_cast<size_t>(outSize));
			resize_and_zero(probs, static_cast<size_t>(T) * static_cast<size_t>(outSize));
			if (tokenLmSampleIds.size() != static_cast<size_t>(T) * static_cast<size_t>(outSize))
				tokenLmSampleIds.resize(static_cast<size_t>(T) * static_cast<size_t>(outSize));
			std::fill(tokenLmSampleIds.begin(), tokenLmSampleIds.end(), 0);

			// Backward scratch (not per-layer; reused across the backward pass)
			// Note: we do not rely on these being zeroed except where explicitly filled in the hot path.
			resize_and_zero(dLogits, static_cast<size_t>(T) * static_cast<size_t>(outSize));
			resize_and_zero(dH, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dH2, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dFF1Act, static_cast<size_t>(T) * static_cast<size_t>(dFF));
			resize_and_zero(dFF1Cat, static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
			resize_and_zero(dX2, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dHAfterAttnFromLN, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dAttnConcat, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dQfull, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dKfull, static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
			resize_and_zero(dVfull, static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
			resize_and_zero(dX1, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dXtmp, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dHInFromLN, static_cast<size_t>(T) * static_cast<size_t>(dModel));
			resize_and_zero(dInput, static_cast<size_t>(T) * static_cast<size_t>(inputSize));
		}
	};

	TensorTransformerState tensorTransformer;
	TransformerScratch transformerScratch;

	// === Positional encoding caches (Transformer) ===
	//
	// These caches avoid recomputing expensive pow()-derived frequency terms (sinusoidal PE and RoPE).
	// They intentionally do NOT cache full [T,dModel] sin/cos tables (which can be enormous).
	//
	// NOTE: This cache is mutable so const inference helpers (e.g. transformerLmForwardLastLogits)
	// can reuse it without allocating each call. NNetwork is not re-entrant; callers should not
	// invoke transformer inference concurrently on the same instance.
	struct TransformerPosEncCache
	{
		// Sinusoidal positional encoding cache:
		// invDenomPair[ii] = 1 / 10000^(2*ii/dModel), length ceil(dModel/2).
		unsigned int sinDModelCached;
		std::vector<double> sinInvDenomPair;

		// RoPE cache:
		// invFreq[ii] = theta^(-2*ii/ropeDim), length ropeDim/2.
		unsigned int ropeDimCached;
		float ropeThetaCached;
		std::vector<double> ropeInvFreq;

		TransformerPosEncCache()
		    : sinDModelCached(0u),
		      sinInvDenomPair(),
		      ropeDimCached(0u),
		      ropeThetaCached(0.0f),
		      ropeInvFreq()
		{
		}

		void reset()
		{
			sinDModelCached = 0u;
			sinInvDenomPair.clear();
			ropeDimCached = 0u;
			ropeThetaCached = 0.0f;
			ropeInvFreq.clear();
		}

		void ensureSinusoidal(unsigned int dModel)
		{
			if (dModel == 0u)
			{
				sinDModelCached = 0u;
				sinInvDenomPair.clear();
				return;
			}
			if (sinDModelCached == dModel && !sinInvDenomPair.empty())
				return;

			sinDModelCached = dModel;
			const unsigned int nPairs = (dModel + 1u) / 2u;
			sinInvDenomPair.assign(static_cast<size_t>(nPairs), 0.0);
			for (unsigned int ii = 0; ii < nPairs; ++ii)
			{
				// invDenom = 10000^(-2*ii/dModel)
				const double exponent = (2.0 * static_cast<double>(ii)) / static_cast<double>(dModel);
				sinInvDenomPair[static_cast<size_t>(ii)] = pow(10000.0, -exponent);
			}
		}

		void ensureRope(unsigned int ropeDimEven, float ropeTheta)
		{
			if (ropeDimEven < 2u || ropeTheta <= 0.0f)
			{
				ropeDimCached = 0u;
				ropeThetaCached = 0.0f;
				ropeInvFreq.clear();
				return;
			}
			if ((ropeDimEven % 2u) != 0u)
				ropeDimEven -= 1u;
			if (ropeDimCached == ropeDimEven && ropeThetaCached == ropeTheta && !ropeInvFreq.empty())
				return;

			ropeDimCached = ropeDimEven;
			ropeThetaCached = ropeTheta;
			ropeInvFreq.assign(static_cast<size_t>(ropeDimEven / 2u), 0.0);
			for (unsigned int ii = 0; ii < (ropeDimEven / 2u); ++ii)
			{
				const double frac = (2.0 * static_cast<double>(ii)) / static_cast<double>(ropeDimEven);
				ropeInvFreq[static_cast<size_t>(ii)] = pow(static_cast<double>(ropeTheta), -frac);
			}
		}
	};
	mutable TransformerPosEncCache transformerPosEncCache;

	// Ensure packed tensor parameters are initialized from the attached DataInput shape.
	// Returns false and sets lastStatus on failure.
	bool ensureTensorParametersInitialized();

	// Tensor-first persistence for model packages (manifest version >= 2).
	// These write/read packed tensors directly.
	NNetworkStatus saveTensorWeightsToFile(const std::string& filePath) const;
	NNetworkStatus loadTensorWeightsFromFile(const std::string& filePath);

	// for tables & graphs
	std::vector<Point2*> rocCurve;
	shmea::GList results;
	shmea::GTable nbRecord;
	//Only for sending on the network
	shmea::GList cNodeActivations;

	// === Modern training loop features (minimal, backward-compatible) ===
	//
	// These features are implemented in the training core and (for DFF) in the tensor SGD step.
	// Defaults preserve historical behavior.
	TrainingConfig trainingConfig;
	float lrScheduleMultiplier; // computed each epoch by the scheduler; starts at 1
	float lastGradNorm;
	float lastGradNormScale;

	// === Tokenizer + vocabulary artifacts (deployment metadata) ===
	//
	// Glades models operate on token IDs. To make model packages self-contained for deployment,
	// callers may attach tokenizer/vocab artifacts to the network and persist them alongside
	// the model weights/architecture.
	//
	// IMPORTANT:
	// - This is metadata only. Glades does not implement BPE/SentencePiece tokenization here.
	// - The artifact format is intentionally dependency-free and validated strictly on load.
public:
	struct TokenizerArtifacts
	{
		// Opaque tokenizer type identifier (examples: "bpe", "sentencepiece", "wordpiece", "custom").
		// This is intended for consumers to route to the appropriate tokenizer implementation.
		std::string type;
		// Vocabulary table mapping token id -> token bytes (UTF-8 recommended but not required).
		std::vector<std::string> vocab;

		// Special token ids (optional; -1 means "not set").
		// These are NOT automatically forced to match trainingConfig.transformer.padTokenId, etc.
		int padTokenId;
		int bosTokenId;
		int eosTokenId;
		int unkTokenId;

		TokenizerArtifacts()
		    : type(),
		      vocab(),
		      padTokenId(-1),
		      bosTokenId(-1),
		      eosTokenId(-1),
		      unkTokenId(-1)
		{
		}

		void reset() { *this = TokenizerArtifacts(); }
	};

private:
	// Whether tokenizerArtifacts is present/meaningful.
	bool tokenizerArtifactsPresent;
	TokenizerArtifacts tokenizerArtifacts;

	// Internal helper used by TrainingCore (friend) to compute the schedule multiplier.
	float computeLearningRateMultiplier(int epochFromStart) const;

	NNetworkStatus run(const DataInput*, int, ITrainingCallbacks*);
	NNetworkStatus failStatus(NNetworkStatus::Code code, const std::string& message);
	// Reset core state (used by constructors/destructor and internal load/build paths).
	void clean();
	// Reset graph/curve outputs (train runs only).
	void resetGraphs();
	// Per-sample forward/backprop/update.
	// Returns explicit status; callers MUST stop on failure.
	NNetworkStatus SGDHelper(unsigned int, int); // Stochastic Gradient Descent
	// Net-type-specific SGD implementations (split into separate translation units).
	// These functions preserve the historical behavior of the corresponding blocks
	// that previously lived inside SGDHelper().
	void SGDHelper_DFF(unsigned int inputRowCounter, int runType);
	void SGDHelper_RNN(unsigned int inputRowCounter, int runType);
	void SGDHelper_GRU(unsigned int inputRowCounter, int runType);
	void SGDHelper_LSTM(unsigned int inputRowCounter, int runType);
	void SGDHelper_TRANSFORMER(unsigned int inputRowCounter, int runType);

	// Owned resources (used only in some construction paths)
	shmea::GPointer<NNInfo> ownedSkeleton;

	// === Core owned state (private; access via explicit API) ===
	//
	// Attached dataset for the active run (train/test) only.
	// This MUST NOT be used as a long-lived pointer: the caller owns DataInput and may delete
	// it immediately after train()/test() returns. `Trainer` is responsible for attaching and
	// detaching this pointer for the duration of a run.
	const DataInput* di;

	// Network architecture ("skeleton").
	//
	// The authoritative lifetime is owned by `ownedSkeleton` when present.
	// `skeleton` is a convenience alias for fast access.
	NNInfo* skeleton;

	// Train loop termination controls (epoch/accuracy/time, etc.).
	Terminator terminator;

	// Internal run-scope guard used by Trainer and internal APIs to enforce the concurrency policy.
	// If ok()==false, the caller must not touch or mutate the network (it is already running).
	class RunLockGuard
	{
	public:
		explicit RunLockGuard(NNetwork& n) : net(&n), acquired(false)
		{
			acquired = (net ? net->tryAcquireRunLock() : false);
		}
		~RunLockGuard()
		{
			if (acquired && net)
				net->releaseRunLock();
		}
		bool ok() const { return acquired; }
	private:
		NNetwork* net;
		bool acquired;
		// non-copyable
		RunLockGuard(const RunLockGuard&);
		RunLockGuard& operator=(const RunLockGuard&);
	};
public:
	enum
	{
		TYPE_DFF = 0,
		TYPE_RNN = 1,
		TYPE_GRU = 2,
		TYPE_LSTM = 3,
		// Transformer encoder: bidirectional self-attention over sequences.
		TYPE_TRANSFORMER_ENCODER = 4,
		// Transformer decoder-only: causal self-attention over sequences.
		TYPE_TRANSFORMER_DECODER = 5
	};

	enum
	{
		RUN_TRAIN = 0,
		RUN_TEST = 1,
		RUN_VALIDATE = 2
	};

	NNetwork(int=TYPE_DFF);
	// Construction from an external NNInfo is non-owning: the network clones and owns it internally.
	explicit NNetwork(const NNInfo* newNNInfo, int newNetType=TYPE_DFF);
	virtual ~NNetwork();
	void setSeed(uint64_t seed);
	uint64_t getSeed() const { return rngSeed; }
	// Returns the network's architecture type (TYPE_DFF, TYPE_RNN, etc.).
	int getNetType() const { return netType; }
	int64_t getCurrentTimeMilliseconds() const;
	bool getRunning() const;
	int getEpochs() const;
	void stop();
	// Unified, versioned persistence (architecture + weights).
	//
	// This is the production-facing API. It stores a self-contained "model package" under:
	//   database/models/<modelName>/{manifest.txt, nninfo.csv, weights.bin}
	//
	// Loading requires a DataInput instance to provide the input feature count so the
	// network tensors can be shaped before applying weights.
	NNetworkStatus saveModel(const std::string& modelName) const;
	NNetworkStatus loadModel(const std::string& modelName, const DataInput* forShape, int netTypeOverride = -1);

	// === Tokenizer/vocab artifacts (optional) ===
	//
	// These APIs manage deployment metadata stored with model packages:
	//   database/models/<modelName>/tokenizer/{manifest.txt,vocab.bin}
	//
	// Thread-safety:
	// - Not safe to mutate while the network is running (same as trainingConfig/terminator).
	bool hasTokenizerArtifacts() const { return tokenizerArtifactsPresent; }
	const TokenizerArtifacts& getTokenizerArtifacts() const { return tokenizerArtifacts; }
	NNetworkStatus setTokenizerArtifacts(const TokenizerArtifacts& a);
	void clearTokenizerArtifacts();

	// === Scalable training checkpointing (resumable) ===
	//
	// Unlike saveModel/loadModel, checkpoints may include optimizer state and are stored in a
	// sharded format to avoid huge single files for large transformers.
	//
	// Layout:
	//   database/checkpoints/<checkpointName>/{manifest.txt, nninfo.csv, shard_000.bin, ...}
	//
	// Notes:
	// - This API is intended for resuming training. It is stricter than saveModel/loadModel:
	//   optimizer state is validated by tensor name and exact element count.
	// - Loading requires a DataInput instance to allocate tensors with the correct input feature
	//   count before applying checkpoint tensors.
	struct CheckpointConfig
	{
		// Maximum bytes per shard file. A value of 0 defaults to 1 GiB.
		size_t maxShardBytes;
		// If true, include optimizer state (momentum / Adam moments) in the checkpoint.
		bool includeOptimizerState;
		CheckpointConfig()
		    : maxShardBytes(static_cast<size_t>(1024ull * 1024ull * 1024ull)),
		      includeOptimizerState(true)
		{
		}
	};

	NNetworkStatus saveCheckpoint(const std::string& checkpointName, const CheckpointConfig& cfg = CheckpointConfig()) const;
	NNetworkStatus loadCheckpoint(const std::string& checkpointName, const DataInput* forShape, int netTypeOverride = -1);
	void setServer(GNet::GServer*, GNet::Connection*);
	// Structured logging support.
	// If no logger override is set, this will use the attached server logger (if any),
	// otherwise a default logger.
	void setLogger(shmea::GLogger* logger);
	shmea::GLogger* getLogger() const;

	// Stochastic Gradient Descent
	NNetworkStatus train(const DataInput*);
	NNetworkStatus test(const DataInput*);
	NNetworkStatus train(const DataInput*, ITrainingCallbacks*);
	NNetworkStatus test(const DataInput*, ITrainingCallbacks*);
	const NNetworkStatus& getLastStatus() const { return lastStatus; }

	// Training loop controls (optional).
	// These are intentionally simple knobs that do not require modifying NNInfo persistence.
	void setLearningRateScheduleNone();
	void setLearningRateScheduleStep(int stepSizeEpochs, float gamma);
	void setLearningRateScheduleExp(float gamma);
	void setLearningRateScheduleCosine(int tMaxEpochs, float minMultiplier);
	float getLearningRateMultiplier() const { return lrScheduleMultiplier; }
	void setGlobalGradClipNorm(float clipNorm);
	float getGlobalGradClipNorm() const { return trainingConfig.globalGradClipNorm; }
	void setPerElementGradClip(float clipLimit);
	float getPerElementGradClip() const { return trainingConfig.perElementGradClip; }
	float getLastGradNorm() const { return lastGradNorm; }
	float getLastGradNormScale() const { return lastGradNormScale; }
	const TrainingConfig& getTrainingConfig() const { return trainingConfig; }
	// Mutable access to training config is supported for backwards compatibility.
	//
	// WARNING:
	// - Do not mutate this while the network is running.
	// - Prefer setTrainingConfig() for a single, validated update point.
	TrainingConfig& getTrainingConfigMutable() { return trainingConfig; }
	// Replace the training config as a single operation.
	// This fails if the network is currently running in another thread.
	NNetworkStatus setTrainingConfig(const TrainingConfig& cfg);

	int64_t getID() const;
	shmea::GString getName() const;
	// Architecture accessors.
	// - getNNInfo(): read-only view of the owned skeleton (may be NULL before load/build).
	const NNInfo* getNNInfo() const;
	// Run-scoped attached dataset (NULL when idle).
	const DataInput* getAttachedDataInput() const { return di; }
	// Terminator accessors.
	const Terminator& getTerminator() const { return terminator; }
	// Mutable access is supported for backwards compatibility.
	//
	// WARNING:
	// - Do not mutate this while the network is running.
	// - Prefer setTerminator() when possible.
	Terminator& getTerminatorMutable() { return terminator; }
	// Replace the terminator settings as a single operation.
	// This fails if the network is currently running in another thread.
	NNetworkStatus setTerminator(const Terminator& t);
	// Primary "accuracy-like" score used by Terminator and UI:
	// - Regression: R^2 expressed as percent in [0,100] (computed in Trainer).
	// - Classification/KL: top-1 accuracy expressed as percent in [0,100] (computed in Trainer).
	//
	// IMPORTANT: This must not return MCC. MCC has its own accessor.
	float getAccuracy() const;
	// Matthews correlation coefficient for classification/KL.
	// Returned in the same units produced by CMatrix (typically percent in this codebase).
	float getMCC() const;
	const CMatrix& getConfusionMatrix() const;
	const shmea::GList& getNodeActivations() const;

	// graphing
	shmea::GList getResults() const;

	// Returns weights in the same GUI serialization format used historically by the UI
	// followed by bias summaries.
	// This reads directly from tensor parameters.
	shmea::GList getWeightsForGui() const;

	// === Transformer token LM KV-cache inference sessions (re-entrant) ===
	//
	// These session APIs allow callers to own KV cache + scratch buffers per request (or per batch),
	// enabling:
	// - concurrent inference across threads using a shared, read-only model
	// - explicit memory ownership and reuse across requests
	// - allocation-free per-token append in hot loops (after Reset)
	//
	// Thread-safety:
	// - The session objects are owned by the caller and are not shared unless you share them.
	// - The NNetwork must not be mutated concurrently with session inference (i.e., do not train while serving).
	//
	// === Transformer structured metrics (serving/inference) ===
	//
	// This is a lightweight, dependency-free metrics surface intended for production serving.
	// It is disabled by default. When enabled, KV-cache session Reset/Append will accumulate
	// timing/counter data and the generation APIs will emit structured log lines via the network logger.
	struct TransformerMetricsConfig
	{
		// Master switch. When false, no extra timing/counters are collected.
		bool enable;
		// If true, collect coarse per-kernel timing breakdowns inside KV append.
		// This adds overhead and should only be enabled when diagnosing performance.
		bool enableKvKernelBreakdown;
		// Emit one log line per request result in batched serving APIs.
		bool logPerRequest;
		// Emit one log line per KV append (very noisy; intended for debugging only).
		bool logPerKvAppend;

		TransformerMetricsConfig()
		    : enable(false),
		      enableKvKernelBreakdown(true),
		      logPerRequest(true),
		      logPerKvAppend(false)
		{
		}
	};

	struct TransformerKvPerfBreakdown
	{
		unsigned long long kvAppends;
		// Cache effectiveness (positional encoding caches owned by sessions).
		unsigned long long sinCacheHits;
		unsigned long long sinCacheMisses;
		unsigned long long ropeCacheHits;
		unsigned long long ropeCacheMisses;
		// NaN/Inf detection
		unsigned long long nonFiniteHiddenState;
		unsigned int lastNonFiniteLayer;
		unsigned int lastNonFinitePos;

		// Wall time (ms) across KV appends (and optionally per-kernel breakdown).
		double msTotal;
		double msEmbed;
		double msPosEnc;
		double msNorm;
		double msProjQKV;
		double msRoPE;
		double msKVStore;
		double msAttention;
		double msWo;
		double msFFN;
		double msLogits;

		TransformerKvPerfBreakdown()
		    : kvAppends(0ULL),
		      sinCacheHits(0ULL),
		      sinCacheMisses(0ULL),
		      ropeCacheHits(0ULL),
		      ropeCacheMisses(0ULL),
		      nonFiniteHiddenState(0ULL),
		      lastNonFiniteLayer(0u),
		      lastNonFinitePos(0u),
		      msTotal(0.0),
		      msEmbed(0.0),
		      msPosEnc(0.0),
		      msNorm(0.0),
		      msProjQKV(0.0),
		      msRoPE(0.0),
		      msKVStore(0.0),
		      msAttention(0.0),
		      msWo(0.0),
		      msFFN(0.0),
		      msLogits(0.0)
		{
		}

		void reset() { *this = TransformerKvPerfBreakdown(); }
	};

private:
	// Transformer serving/inference metrics configuration (default: disabled).
	TransformerMetricsConfig transformerMetricsCfg;

public:
	// Configure structured transformer metrics/logging.
	// - When enabled, KV inference sessions will accumulate perf counters and the generation
	//   APIs will emit structured log lines through getLogger().
	void setTransformerMetricsConfig(const TransformerMetricsConfig& cfg) { transformerMetricsCfg = cfg; }
	const TransformerMetricsConfig& getTransformerMetricsConfig() const { return transformerMetricsCfg; }

	struct TransformerLmSession
	{
		enum KVCacheDType
		{
			KV_CACHE_F32 = 0,
			KV_CACHE_F16 = 1,
			KV_CACHE_BF16 = 2
		};

		bool initialized;
		unsigned int maxLen;
		unsigned int curLen;
		// Cached model dims (for indexing and sanity).
		unsigned int dModel;
		unsigned int dFF;
		unsigned int nHeads;
		unsigned int nKVHeads;
		unsigned int nLayers;
		unsigned int dHead;
		unsigned int dModelKV;
		unsigned int ffnKind;
		unsigned int ff1Width;

		// KV-cache storage dtype (owned by the session).
		KVCacheDType kvCacheDType;

		// Cached K/V per layer: [nLayers, maxLen, dModelKV]
		// Exactly one storage is used based on kvCacheDType.
		std::vector<float, glades::AlignedAllocator<float, 64> > k;
		std::vector<float, glades::AlignedAllocator<float, 64> > v;
		// Low-precision KV cache storage:
		// - When kvCacheDType==KV_CACHE_F16: values are IEEE754 binary16 (FP16)
		// - When kvCacheDType==KV_CACHE_BF16: values are bfloat16 (BF16)
		std::vector<uint16_t, glades::AlignedAllocator<uint16_t, 64> > k16;
		std::vector<uint16_t, glades::AlignedAllocator<uint16_t, 64> > v16;
		// keyValid[pos] == 1 => real token, 0 => padding (masked out of attention)
		std::vector<unsigned char, glades::AlignedAllocator<unsigned char, 64> > keyValid; // [maxLen]

		// Scratch buffers sized in Reset and reused across appends.
		std::vector<float, glades::AlignedAllocator<float, 64> > h;          // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > x1;         // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > x2;         // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > q;          // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > kvec;       // [dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > vvec;       // [dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > attnConcat; // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > attnOut;    // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffPre;      // [ff1Width]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffAct;      // [dFF]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffOut;      // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > scores;     // [maxLen]

		// Positional encoding caches (owned by the session to avoid mutating NNetwork).
		unsigned int sinDModelCached;
		std::vector<double> sinInvDenomPair;
		unsigned int ropeDimCached;
		float ropeThetaCached;
		std::vector<double> ropeInvFreq;

		// Optional performance counters/timers (populated only when enabled).
		bool metricsEnabled;
		TransformerKvPerfBreakdown perf;

		TransformerLmSession()
		    : initialized(false),
		      maxLen(0u),
		      curLen(0u),
		      dModel(0u),
		      dFF(0u),
		      nHeads(0u),
		      nKVHeads(0u),
		      nLayers(0u),
		      dHead(0u),
		      dModelKV(0u),
		      ffnKind(0u),
		      ff1Width(0u),
		      kvCacheDType(KV_CACHE_F32),
		      k(),
		      v(),
		      k16(),
		      v16(),
		      keyValid(),
		      h(),
		      x1(),
		      x2(),
		      q(),
		      kvec(),
		      vvec(),
		      attnConcat(),
		      attnOut(),
		      ffPre(),
		      ffAct(),
		      ffOut(),
		      scores(),
		      sinDModelCached(0u),
		      sinInvDenomPair(),
		      ropeDimCached(0u),
		      ropeThetaCached(0.0f),
		      ropeInvFreq(),
		      metricsEnabled(false),
		      perf()
		{
		}

		void reset()
		{
			initialized = false;
			maxLen = 0u;
			curLen = 0u;
			dModel = dFF = nHeads = nKVHeads = nLayers = dHead = dModelKV = 0u;
			ffnKind = 0u;
			ff1Width = 0u;
			kvCacheDType = KV_CACHE_F32;
			k.clear();
			v.clear();
			k16.clear();
			v16.clear();
			keyValid.clear();
			h.clear();
			x1.clear();
			x2.clear();
			q.clear();
			kvec.clear();
			vvec.clear();
			attnConcat.clear();
			attnOut.clear();
			ffPre.clear();
			ffAct.clear();
			ffOut.clear();
			scores.clear();
			sinDModelCached = 0u;
			sinInvDenomPair.clear();
			ropeDimCached = 0u;
			ropeThetaCached = 0.0f;
			ropeInvFreq.clear();
			metricsEnabled = false;
			perf.reset();
		}
	};

	struct TransformerLmBatchSession
	{
		enum KVCacheDType
		{
			KV_CACHE_F32 = 0,
			KV_CACHE_F16 = 1,
			KV_CACHE_BF16 = 2
		};

		bool initialized;
		unsigned int batchSize;
		unsigned int maxLen;
		// Per-sequence current lengths.
		std::vector<unsigned int> curLen; // [batchSize]

		// Cached model dims.
		unsigned int dModel;
		unsigned int dFF;
		unsigned int nHeads;
		unsigned int nKVHeads;
		unsigned int nLayers;
		unsigned int dHead;
		unsigned int dModelKV;
		unsigned int ffnKind;
		unsigned int ff1Width;

		// KV-cache storage dtype (owned by the session).
		KVCacheDType kvCacheDType;

		// Cached K/V per sequence:
		// - k/v are laid out as [batchSize, nLayers, maxLen, dModelKV] in a contiguous buffer.
		// Exactly one storage is used based on kvCacheDType.
		std::vector<float, glades::AlignedAllocator<float, 64> > k;
		std::vector<float, glades::AlignedAllocator<float, 64> > v;
		// Low-precision KV cache storage (same encoding as TransformerLmSession::k16/v16).
		std::vector<uint16_t, glades::AlignedAllocator<uint16_t, 64> > k16;
		std::vector<uint16_t, glades::AlignedAllocator<uint16_t, 64> > v16;
		// keyValid per sequence: [batchSize, maxLen]
		std::vector<unsigned char, glades::AlignedAllocator<unsigned char, 64> > keyValid;

		// Shared scratch buffers (reused while looping over batch elements).
		std::vector<float, glades::AlignedAllocator<float, 64> > h;          // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > x1;         // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > x2;         // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > q;          // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > kvec;       // [dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > vvec;       // [dModelKV]
		std::vector<float, glades::AlignedAllocator<float, 64> > attnConcat; // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > attnOut;    // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffPre;      // [ff1Width]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffAct;      // [dFF]
		std::vector<float, glades::AlignedAllocator<float, 64> > ffOut;      // [dModel]
		std::vector<float, glades::AlignedAllocator<float, 64> > scores;     // [maxLen]

		// Positional encoding caches (session-owned).
		unsigned int sinDModelCached;
		std::vector<double> sinInvDenomPair;
		unsigned int ropeDimCached;
		float ropeThetaCached;
		std::vector<double> ropeInvFreq;

		// Optional performance counters/timers (populated only when enabled).
		bool metricsEnabled;
		TransformerKvPerfBreakdown perf;

		TransformerLmBatchSession()
		    : initialized(false),
		      batchSize(0u),
		      maxLen(0u),
		      curLen(),
		      dModel(0u),
		      dFF(0u),
		      nHeads(0u),
		      nKVHeads(0u),
		      nLayers(0u),
		      dHead(0u),
		      dModelKV(0u),
		      ffnKind(0u),
		      ff1Width(0u),
		      kvCacheDType(KV_CACHE_F32),
		      k(),
		      v(),
		      k16(),
		      v16(),
		      keyValid(),
		      h(),
		      x1(),
		      x2(),
		      q(),
		      kvec(),
		      vvec(),
		      attnConcat(),
		      attnOut(),
		      ffPre(),
		      ffAct(),
		      ffOut(),
		      scores(),
		      sinDModelCached(0u),
		      sinInvDenomPair(),
		      ropeDimCached(0u),
		      ropeThetaCached(0.0f),
		      ropeInvFreq(),
		      metricsEnabled(false),
		      perf()
		{
		}

		void reset()
		{
			initialized = false;
			batchSize = 0u;
			maxLen = 0u;
			curLen.clear();
			dModel = dFF = nHeads = nKVHeads = nLayers = dHead = dModelKV = 0u;
			ffnKind = 0u;
			ff1Width = 0u;
			kvCacheDType = KV_CACHE_F32;
			k.clear();
			v.clear();
			k16.clear();
			v16.clear();
			keyValid.clear();
			h.clear();
			x1.clear();
			x2.clear();
			q.clear();
			kvec.clear();
			vvec.clear();
			attnConcat.clear();
			attnOut.clear();
			ffPre.clear();
			ffAct.clear();
			ffOut.clear();
			scores.clear();
			sinDModelCached = 0u;
			sinInvDenomPair.clear();
			ropeDimCached = 0u;
			ropeThetaCached = 0.0f;
			ropeInvFreq.clear();
			metricsEnabled = false;
			perf.reset();
		}
	};

	// Session APIs (const: do not mutate NNetwork inference state).
	NNetworkStatus transformerLmSessionReset(TransformerLmSession& session, unsigned int maxSeqLen) const;
	NNetworkStatus transformerLmSessionAppend(TransformerLmSession& session, unsigned int tokenId, std::vector<float>* outLogits /* optional */) const;

	NNetworkStatus transformerLmBatchSessionReset(TransformerLmBatchSession& session, unsigned int batchSize, unsigned int maxSeqLen) const;
	// Append one token for each active batch element (ragged-safe):
	// - Only active[b]!=0 advances session.curLen[b]
	// - If tokenValid is provided and tokenValid[b]==0, the position is treated as padding and masked out of attention
	// - If outLogitsFlat is provided, it is resized to [batchSize * vocabSize] and filled row-major; inactive rows are zeros
	NNetworkStatus transformerLmBatchSessionAppendSelective(TransformerLmBatchSession& session,
	                                                       const std::vector<unsigned int>& tokenIds,
	                                                       const std::vector<unsigned char>* tokenValid /* optional */,
	                                                       const std::vector<unsigned char>& active,
	                                                       std::vector<float>* outLogitsFlat /* optional */) const;
	NNetworkStatus transformerLmBatchSessionAppendSelective(TransformerLmBatchSession& session,
	                                                       const std::vector<unsigned int>& tokenIds,
	                                                       const std::vector<unsigned char>& active,
	                                                       std::vector<float>* outLogitsFlat /* optional */) const
	{
		return transformerLmBatchSessionAppendSelective(session, tokenIds, NULL, active, outLogitsFlat);
	}

	// === Transformer token LM generation API (decoder-only, KV-cache) ===
	//
	// This is the production-facing "real inference" API:
	// - KV-cache prefill on a prompt
	// - iterative decode with greedy or sampling (temperature/top-k/top-p)
	// - streaming callbacks for token emission / cancellation
	//
	// IMPORTANT:
	// - This API allocates and uses a per-call KV session (no internal KV state is retained).
	// - It is still not safe to call concurrently with training/mutation on the same NNetwork instance.
	// - This API requires token LM mode (enableTokenEmbedding==true) and decoder net type.
	struct TransformerGenerateConfig
	{
		// Maximum number of new tokens to generate (excluding the prompt).
		unsigned int maxNewTokens;
		// Total KV cache length cap. If 0, defaults to promptLen + maxNewTokens.
		// If provided, it must be >= promptLen + maxNewTokens.
		unsigned int maxSeqLen;

		// Sampling controls:
		// - temperature <= 0 => greedy (argmax)
		// - topK == 0 => disabled
		// - topP <= 0 or > 1 => disabled
		float temperature;
		unsigned int topK;
		float topP;
		// Nucleus (top-p) implementation policy:
		//
		// When topP < 1 and topK == 0, a "pure" nucleus implementation would need to:
		// - sort the full vocabulary by logit each step (O(V log V)), then
		// - take the smallest prefix whose cumulative probability >= topP.
		//
		// That can be prohibitively expensive for large vocabularies on CPU.
		//
		// Glades defaults to an explicit approximation:
		// - if topP < 1 and topK == 0, we first cap candidates to the top-K tokens where
		//   K = min(vocabSize, topPTopKCap), then apply top-p within those candidates.
		//
		// Set topPTopKCap to 0 to disable this approximation (full-vocab nucleus).
		unsigned int topPTopKCap;

		// Stop controls:
		// - eosTokenId < 0 => disabled
		// - if stopOnEos==true and eosTokenId is produced, generation stops after emitting it
		int eosTokenId;
		bool stopOnEos;

		// Output formatting:
		// - includePromptInOutput==true => out.tokens includes prompt first, then generated tokens
		// - otherwise out.tokens contains only generated tokens
		bool includePromptInOutput;

		// RNG control:
		// - rngSeedOverride!=0 => seed the per-call RNG with this value
		// - rngSeedOverride==0 => derive a deterministic seed from the network seed + prompt tokens
		//
		// IMPORTANT:
		// - Generation does not mutate or depend on the shared `NNetwork::rngEngine`.
		// - If you want stochastic variation across calls, you must supply different rngSeedOverride values.
		uint64_t rngSeedOverride;

		TransformerGenerateConfig()
		    : maxNewTokens(0u),
		      maxSeqLen(0u),
		      temperature(1.0f),
		      topK(0u),
		      topP(1.0f),
		      topPTopKCap(256u),
		      eosTokenId(-1),
		      stopOnEos(true),
		      includePromptInOutput(false),
		      rngSeedOverride(0ULL)
		{
		}
	};

	struct TransformerGenerateResult
	{
		// Tokens returned (see includePromptInOutput).
		std::vector<unsigned int> tokens;
		// Why generation ended.
		bool stoppedOnEos;
		// Stopped because a non-EOS stop token was encountered (TransformerServeRequest::stopTokenIds).
		// This is distinct from stoppedOnEos so serving telemetry can distinguish these cases.
		bool stoppedByStopToken;
		bool stoppedByCallback;
		bool stoppedByLimit;
		// Last token emitted (undefined if no tokens were emitted).
		unsigned int lastToken;

		TransformerGenerateResult()
		    : tokens(),
		      stoppedOnEos(false),
		      stoppedByStopToken(false),
		      stoppedByCallback(false),
		      stoppedByLimit(false),
		      lastToken(0u)
		{
		}
	};

	class ITransformerGenerateCallbacks
	{
	public:
		virtual ~ITransformerGenerateCallbacks() {}
		// Called after a token is emitted (and appended to the KV cache).
		// Return true to stop generation early.
		virtual bool onToken(const NNetwork& /*net*/, unsigned int /*tokenId*/, unsigned int /*generatedIndex*/) { return false; }
		// Polled once per step; return true to cancel generation.
		virtual bool shouldStop(const NNetwork& /*net*/) { return false; }
	};

	// Generate tokens given a prompt (token IDs).
	// - `promptTokens` must be non-empty (callers should include a BOS token if needed).
	// - `out` is always overwritten.
	NNetworkStatus transformerLmGenerate(const std::vector<unsigned int>& promptTokens,
	                                    const TransformerGenerateConfig& cfg,
	                                    TransformerGenerateResult& out,
	                                    ITransformerGenerateCallbacks* cb /* optional */) const;

	// === Serving-grade generation (batched, ragged prompts, continuous decode) ===
	//
	// This API is designed for "real serving" needs:
	// - Multiple requests in one call (batching)
	// - Ragged prompts without positional-encoding distortion
	// - Per-request early stop (EOS/limits/callback cancellation)
	// - Token streaming callbacks with request index
	//
	// Implementation notes:
	// - Uses the internal batched KV cache with selective appends (no fake padding positions).
	// - Still scalar (loops requests), but allocation-free per decode step.
	struct TransformerServeRequest
	{
		std::vector<unsigned int> promptTokens;
		TransformerGenerateConfig cfg;
		// Optional additional stop tokens (besides eosTokenId).
		// If any token in stopTokenIds is generated, generation stops after emitting it.
		std::vector<unsigned int> stopTokenIds;
	};

	struct TransformerServeBatchResult
	{
		// One result per request (aligned to input order).
		std::vector<TransformerGenerateResult> results;
	};

	class ITransformerServeCallbacks
	{
	public:
		virtual ~ITransformerServeCallbacks() {}
		// Called after a token is emitted for a request.
		// Return true to stop that request early.
		virtual bool onToken(const NNetwork& /*net*/, unsigned int /*requestIndex*/, unsigned int /*tokenId*/, unsigned int /*generatedIndex*/) { return false; }
		// Polled once per global decode step; return true to cancel all requests.
		virtual bool shouldStopAll(const NNetwork& /*net*/) { return false; }
		// Polled before sampling for a request each step; return true to cancel that request.
		virtual bool shouldStopRequest(const NNetwork& /*net*/, unsigned int /*requestIndex*/) { return false; }
	};

	// === Continuous batching scheduler (persistent) ===
	//
	// This is the "real batching architecture" primitive used by serving stacks:
	// - Create a batcher with a fixed capacity and max sequence length.
	// - Submit requests into slots (join) and remove them when done (leave).
	// - Call Step() repeatedly to advance all active requests by one token append:
	//   - requests still in prompt prefill append one prompt token
	//   - requests in decode sample + append one generated token
	//
	// Ownership:
	// - The batcher owns all request state (prompt tokens, stop tokens, results) per slot.
	// - The caller owns the batcher object and can reuse it across multiple batches.
	//
	// Thread-safety:
	// - A batcher is not internally synchronized; do not call Step/Submit/Remove concurrently
	//   on the same batcher from multiple threads.
	// - Multiple batchers may be used concurrently with the same NNetwork as long as the network
	//   is not being mutated (trained) concurrently.
	struct TransformerServeBatcherConfig
	{
		// Maximum number of concurrent requests (slots) in this batcher.
		unsigned int maxBatchSize;
		// Maximum KV cache length per request. Requests with larger maxSeqLen are rejected.
		unsigned int maxSeqLen;
		// If true, zero-out the used KV prefix when removing a slot.
		// This is more secure but can be expensive for large models/long sequences.
		bool wipeKvOnRemove;
		// Seed for the batcher's shared RNG stream (used when a request does not provide rngSeedOverride).
		// 0 => derive from this network's seed.
		uint64_t rngSeed;

		TransformerServeBatcherConfig()
		    : maxBatchSize(0u),
		      maxSeqLen(0u),
		      wipeKvOnRemove(false),
		      rngSeed(0ULL)
		{
		}
	};

	struct TransformerServeBatcher
	{
		bool initialized;
		unsigned int vocab;
		unsigned int maxBatchSize;
		unsigned int maxSeqLen;
		bool wipeKvOnRemove;

		// One KV-cache session sized for [maxBatchSize, maxSeqLen].
		TransformerLmBatchSession session;

		// Per-slot state (size maxBatchSize).
		std::vector<unsigned char> inUse;
		std::vector<unsigned char> done;
		std::vector<unsigned int> promptPos;
		std::vector<unsigned int> promptLen;
		std::vector<unsigned int> generated;
		std::vector<unsigned int> reqMaxNew;
		std::vector<unsigned int> reqMaxLen;

		// Request payload per slot (owned).
		std::vector<TransformerServeRequest> req;
		// Results per slot (owned).
		std::vector<TransformerGenerateResult> results;

		// RNG: one shared stream for non-overridden requests, and optional per-slot overrides.
		glades::rng::Engine batchEngine;
		std::vector<glades::rng::Engine> overrideEngines;
		std::vector<unsigned char> hasOverride;

		// Hot-loop buffers (no per-step allocations after Reset).
		std::vector<unsigned int> tokenIds;
		std::vector<unsigned char> active;
		std::vector<unsigned int> sampledTok;     // only meaningful for decode slots in the current step
		std::vector<unsigned char> sampledIsValid;
		std::vector<float> prevLogitsFlat; // [B, vocab]
		std::vector<float> logitsFlat;     // [B, vocab]
		// Sampling scratch (reused across slots; Step processes slots sequentially for sampling).
		std::vector<unsigned int> idxScratch;
		std::vector<float> weightScratch;

		TransformerServeBatcher()
		    : initialized(false),
		      vocab(0u),
		      maxBatchSize(0u),
		      maxSeqLen(0u),
		      wipeKvOnRemove(false),
		      session(),
		      inUse(),
		      done(),
		      promptPos(),
		      promptLen(),
		      generated(),
		      reqMaxNew(),
		      reqMaxLen(),
		      req(),
		      results(),
		      batchEngine(),
		      overrideEngines(),
		      hasOverride(),
		      tokenIds(),
		      active(),
		      sampledTok(),
		      sampledIsValid(),
		      prevLogitsFlat(),
		      logitsFlat(),
		      idxScratch(),
		      weightScratch()
		{
		}

		void reset()
		{
			initialized = false;
			vocab = 0u;
			maxBatchSize = 0u;
			maxSeqLen = 0u;
			wipeKvOnRemove = false;
			session.reset();

			inUse.clear();
			done.clear();
			promptPos.clear();
			promptLen.clear();
			generated.clear();
			reqMaxNew.clear();
			reqMaxLen.clear();
			req.clear();
			results.clear();

			overrideEngines.clear();
			hasOverride.clear();

			tokenIds.clear();
			active.clear();
			sampledTok.clear();
			sampledIsValid.clear();
			prevLogitsFlat.clear();
			logitsFlat.clear();
			idxScratch.clear();
			weightScratch.clear();
		}
	};

	// Initialize/reset a persistent continuous batcher.
	// After reset, the batcher has no active requests; callers may Submit() requests into free slots.
	NNetworkStatus transformerLmServeBatcherReset(TransformerServeBatcher& batcher,
	                                             const TransformerServeBatcherConfig& cfg) const;
	// Submit a request into a free slot. Returns the slot index in outSlot.
	NNetworkStatus transformerLmServeBatcherSubmit(TransformerServeBatcher& batcher,
	                                              const TransformerServeRequest& request,
	                                              unsigned int& outSlot) const;
	// Remove (free) a slot. Safe to call on done or cancelled slots.
	NNetworkStatus transformerLmServeBatcherRemove(TransformerServeBatcher& batcher, unsigned int slot) const;
	// Advance all active slots by one append step (prompt prefill or decode).
	// - Emits callbacks for generated tokens.
	// - Does not allocate on the hot path after Reset (subject to request submission copying).
	NNetworkStatus transformerLmServeBatcherStep(TransformerServeBatcher& batcher,
	                                            ITransformerServeCallbacks* cb /* optional */) const;

	// Batched generation entrypoint.
	// - Requests must be non-empty; each request must have a non-empty promptTokens.
	// - out.results is always overwritten and sized to requests.size().
	NNetworkStatus transformerLmServeGenerateBatch(const std::vector<TransformerServeRequest>& requests,
	                                              TransformerServeBatchResult& out,
	                                              ITransformerServeCallbacks* cb /* optional */) const;

	// === Transformer token LM full forward inference (debug/test) ===
	//
	// Computes the logits for the *last* token position of a full forward pass over `tokenIds`.
	// This is intended for unit testing (e.g., validating KV-cache parity) and small-scale debugging.
	//
	// Preconditions:
	// - netType == TYPE_TRANSFORMER_DECODER
	// - trainingConfig.transformer.enableTokenEmbedding == true
	// - tensorTransformer.initialized == true
	//
	// Output:
	// - outLogits is resized to vocabSize and filled with unnormalized logits.
	NNetworkStatus transformerLmForwardLastLogits(const std::vector<unsigned int>& tokenIds,
	                                             std::vector<float>& outLogits) const;
};
};

#endif
