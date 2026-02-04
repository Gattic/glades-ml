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
#include "../State/LayerBuilder.h"
#include "../Structure/nninfo.h" // ensure NNInfo is a complete type for ownedSkeleton deletion
#include "../rng.h"
#include "training_callbacks.h"
#include "training_config.h"
#include "bayes.h"
#include <algorithm>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <sys/time.h>
#include <stdint.h>

class Point2;

namespace shmea {
class GTable;
};

namespace GNet {
class GServer;
class Connection;
};

namespace glades {

class DataInput;
class Layer;
class Node;
class NetworkState;
class LayerBuilder;
class CMatrix;
class MetaNetwork;
class TrainingCore;
class Trainer;

// Lightweight status type for train/test runs.
// This replaces silent early-returns with an explicit error code + message.
struct NNetworkStatus
{
	enum Code
	{
		OK = 0,
		INVALID_ARGUMENT,
		INVALID_STATE,
		EMPTY_DATA,
		BUILD_FAILED,
		INTERNAL_ERROR
	};

	Code code;
	std::string message;

	NNetworkStatus(Code c = OK, const std::string& msg = std::string()) : code(c), message(msg) {}

	bool ok() const { return code == OK; }
};

class NNetwork
{
private:
	friend MetaNetwork;
	friend TrainingCore;
	friend Trainer;

	// Tensor-based DFF training state.
	//
	// This is a contiguous-buffer rewrite of the original Node/Edge training core.
	// We keep LayerBuilder ("meat") as the weight/bias storage + dropout mask generator,
	// but perform forward/backward and SGD updates using explicit vectors/matrices.
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
	// These mirror the Node/Edge graph weights into packed contiguous arrays so the
	// recurrent SGD helpers can run mostly on cache-friendly kernels rather than
	// pointer chasing through Edge objects.
	//
	// IMPORTANT:
	// - These are *training-time* accelerators. During training, the authoritative weights live
	//   in these packed tensors for performance. The Node/Edge graph in `meat` is treated as a
	//   lazily-synchronized view used for persistence/visualization/debugging.
	// - We only sync tensors -> graph at safe "inspection points" (e.g., end-of-epoch) to avoid
	//   catastrophic per-update pointer chasing.
	// - Momentum and weight decay semantics match the DFF tensor path (v = mf*v + lr*g; w -= v).
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
	// so legacy call sites (glades::rng::uniform_*) become per-network without invasive refactors.
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
	volatile int runLock;

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

    bool changeInputLayers;
	NNetworkStatus lastStatus;
	TensorDFFState tensorDff;
	RecurrentScratch recScratch;
	// Whether `meat`'s Node/Edge weights are stale relative to tensor weights.
	// Set true whenever tensor weights are updated; cleared when we sync tensors -> graph.
	bool graphWeightsDirty;

	// Synchronize tensor weights into the Node/Edge graph (`meat`) if they are dirty.
	// This is intentionally *not* done inside the hot SGD loops.
	void syncGraphWeightsFromTensorsIfDirty();

	// Initialize tensor parameters from the legacy Node/Edge graph if tensors are not yet initialized.
	//
	// This is a bootstrap mechanism only:
	// - After initialization, the authoritative parameters live in tensors.
	// - The Node/Edge graph is treated as a derived debug snapshot and must not be mutated
	//   to "change the model" during training/inference.
	//
	// Returns false and sets lastStatus on failure.
	bool ensureTensorParametersInitializedFromGraph();

	// Tensor-first persistence for model packages (manifest version >= 2).
	// These write/read packed tensors directly and do NOT rely on the Node/Edge graph being in sync.
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

	// Internal helper used by TrainingCore (friend) to compute the schedule multiplier.
	float computeLearningRateMultiplier(int epochFromStart) const;

	NNetworkStatus run(const DataInput*, int, ITrainingCallbacks*);
	NNetworkStatus failStatus(NNetworkStatus::Code code, const std::string& message);
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

    bool mustBuildMeat;

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

	// Legacy graph representation used for dropout masks, visualization/debugging, and
	// compatibility with older persistence paths.
	LayerBuilder meat;

	// Train loop termination controls (epoch/accuracy/time, etc.).
	Terminator terminator;
public:
	enum
	{
		TYPE_DFF = 0,
		TYPE_RNN = 1,
		TYPE_GRU = 2,
		TYPE_LSTM = 3
	};

	// Internal run-scope guard used by Trainer to enforce the concurrency policy.
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

	enum
	{
		RUN_TRAIN = 0,
		RUN_TEST = 1,
		RUN_VALIDATE = 2
	};

	NNetwork(int=TYPE_DFF);
	// Construction from an external NNInfo is non-owning: the network clones and owns it internally.
	explicit NNetwork(const NNInfo* newNNInfo, int newNetType=TYPE_DFF);
	// Convenience overload for callers with a reference.
	explicit NNetwork(const NNInfo& newNNInfo, int newNetType=TYPE_DFF) : NNetwork(&newNNInfo, newNetType) {}
	virtual ~NNetwork();
	void setSeed(uint64_t seed);
	uint64_t getSeed() const { return rngSeed; }
	int64_t getCurrentTimeMilliseconds() const;
	bool getRunning() const;
	int getEpochs() const;
	void stop();
    bool getChangeInputLayers() const;
    void setChangeInputLayers(bool);

	// Database
	bool load(const shmea::GString&);
	bool save() const;
	// Unified, versioned persistence (architecture + weights).
	//
	// This is the production-facing API. It stores a self-contained "model package" under:
	//   database/models/<modelName>/{manifest.txt, nninfo.csv, weights.txt}
	//
	// Loading requires a DataInput instance to provide the input feature count so the
	// network graph can be rebuilt before applying weights.
	NNetworkStatus saveModel(const std::string& modelName) const;
	NNetworkStatus loadModel(const std::string& modelName, const DataInput* forShape, int netTypeOverride = -1);
	void setServer(GNet::GServer*, GNet::Connection*);

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
	TrainingConfig& getTrainingConfigMutable() { return trainingConfig; }

	int64_t getID() const;
	shmea::GString getName() const;
	// Architecture accessors.
	// - getNNInfo(): read-only view of the owned skeleton (may be NULL before load/build).
	// - getNNInfoMutable(): advanced use only; mutating NNInfo changes training hyperparams and/or shape.
	const NNInfo* getNNInfo() const;
	NNInfo* getNNInfoMutable();
	// Run-scoped attached dataset (NULL when idle).
	const DataInput* getAttachedDataInput() const { return di; }
	// Legacy graph accessors (debug/compatibility).
	const LayerBuilder& graph() const { return meat; }
	// IMPORTANT (production safety):
	// The legacy Node/Edge graph (`LayerBuilder`) is *not* the authoritative parameter store during a run.
	// During train/test, parameters live in packed tensors and the graph is a lazily-synchronized debug view.
	//
	// To prevent accidental races/corruption, mutable graph access is forbidden while the network is running.
	// - Use `materializeGraphParameters()` + `graph()` for read-only inspection.
	// - If you need to mutate the graph (e.g. to set deterministic initial weights in a unit test),
	//   do it before calling train()/test().
	LayerBuilder& graphMutable();
	// Terminator accessors.
	const Terminator& getTerminator() const { return terminator; }
	Terminator& getTerminatorMutable() { return terminator; }
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
	void clean();
	void resetGraphs();

    bool getMustdBuildMeat() const;
    void setMustdBuildMeat(bool);

	// === Tensor-first parameter representation ===
	//
	// The authoritative parameters live in packed tensor state (TensorDFFState / TensorRNNState /
	// TensorGatedState). The legacy Node/Edge graph in `meat` is treated as a derived view for
	// visualization/debugging/persistence only.
	//
	// Call this when you need to inspect weights through Node/Edge APIs (e.g., unit tests,
	// legacy save paths). It is intentionally not done automatically each epoch.
	void materializeGraphParameters();

	// Returns weights in the same legacy GUI serialization format as `LayerBuilder::getWeights()`
	// followed by bias summaries (same as `LayerBuilder::addBiasWeights()`).
	// This reads directly from tensor parameters when available.
	shmea::GList getWeightsForGui() const;
};
};

#endif
