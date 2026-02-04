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
#include "network.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Database/ServiceData.h"
#include "Backend/Networking/main.h"
#include "../GMath/OHE.h"
#include "../GMath/cmatrix.h"
#include "../GMath/gmath.h"
#include "../State/LayerBuilder.h"
#include "../State/NetworkState.h"
#include "../State/Terminator.h"
#include "../State/edge.h"
#include "../State/layer.h"
#include "../State/node.h"
#include "../Structure/nninfo.h"
#include "../DataObjects/NumberInput.h"
#include "../DataObjects/ImageInput.h"
#include "training_core.h"
#include "param_layout.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace glades;

// for stopping ml  training instances

/*!
 * @brief NNetwork constructor
 * @details creates an empty nnetwork
 */
glades::NNetwork::NNetwork(int newNetType)
{
	running = false;
	runLock = 0;
	di = NULL;
	skeleton = NULL;
	ownedSkeleton.reset();
	serverInstance = NULL;
	cConnection = NULL;
	// Modern training loop defaults (preserve behavior)
	trainingConfig = TrainingConfig();
	lrScheduleMultiplier = 1.0f;
	lastGradNorm = 0.0f;
	lastGradNormScale = 1.0f;
	// Initialize tensor gate packs (gateCount is fixed by architecture type).
	tensorGru = TensorGatedState(3u);
	tensorLstm = TensorGatedState(4u);
	clean();
	netType = newNetType;
	minibatchSize = NNInfo::BATCH_STOCHASTIC;
    mustBuildMeat = true;
	rngSeed = glades::rng::current_seed();
	glades::rng::seed_engine(rngEngine, rngSeed);
}

/*!
 * @brief NNetwork destructor
 * @details destroys the NNetwork object
 */
glades::NNetwork::NNetwork(const NNInfo* newNNInfo, int newNetType)
{
	// Constructors must always fully initialize the object. Never early-return.
	running = false;
	runLock = 0;
	changeInputLayers = false;
	di = NULL;
	skeleton = NULL;
	ownedSkeleton.reset();
	serverInstance = NULL;
	cConnection = NULL;
	// Modern training loop defaults (preserve behavior)
	trainingConfig = TrainingConfig();
	lrScheduleMultiplier = 1.0f;
	lastGradNorm = 0.0f;
	lastGradNormScale = 1.0f;
	tensorGru = TensorGatedState(3u);
	tensorLstm = TensorGatedState(4u);
	clean();

	// Lifetime safety: clone and own the NNInfo rather than borrowing a raw pointer.
	// Many call sites allocate an NNInfo, pass it into NNetwork, and later delete it.
	// Borrowing would leave `skeleton` dangling.
	if (newNNInfo)
	{
		ownedSkeleton = shmea::GPointer<NNInfo>(new NNInfo(newNNInfo->getName(), newNNInfo->toGTable()));
		skeleton = ownedSkeleton.get();
	}
	netType = newNetType;
	minibatchSize = (skeleton ? skeleton->getBatchSize() : NNInfo::BATCH_STOCHASTIC);
	mustBuildMeat = true;
	rngSeed = glades::rng::current_seed();
	glades::rng::seed_engine(rngEngine, rngSeed);
}

glades::NNetwork::~NNetwork()
{
	clean();
	resetGraphs();
}

bool glades::NNetwork::tryAcquireRunLock()
{
#if defined(__GNUC__) || defined(__clang__)
	// Atomic test-and-set.
	return __sync_bool_compare_and_swap(&runLock, 0, 1);
#else
	// Best-effort fallback (NOT thread-safe without atomic primitives).
	if (runLock != 0)
		return false;
	runLock = 1;
	return true;
#endif
}

void glades::NNetwork::releaseRunLock()
{
#if defined(__GNUC__) || defined(__clang__)
	__sync_lock_release(&runLock);
#else
	runLock = 0;
#endif
}

void glades::NNetwork::materializeGraphParameters()
{
	// Legacy Node/Edge graph is a derived view of packed tensor parameters.
	// This function is intentionally explicit so callers control when the sync cost is paid.
	syncGraphWeightsFromTensorsIfDirty();
}

shmea::GList glades::NNetwork::getWeightsForGui() const
{
	// Preserve historical serialization format used by GUI:
	// - Per-layer, per-node weights (excluding per-neuron bias edges), with ',' between nodes and ';' between layers
	// - Then a 'B' marker followed by per-layer bias summaries.
	//
	// IMPORTANT: For recurrent nets, this matches historical behavior of LayerBuilder::getWeights():
	// it includes only the "input" weights (Wx/W) stored on the hidden/output nodes, and does not
	// include recurrent matrices stored on context nodes (Wh/U).

	shmea::GList weights;

	// If tensor state isn't initialized yet (e.g. before first run), fall back to the graph view.
	// This keeps legacy code paths working.
	const bool hasDff = (netType == TYPE_DFF) && tensorDff.initialized;
	const bool hasRnn = (netType == TYPE_RNN) && tensorRnn.initialized;
	const bool hasGru = (netType == TYPE_GRU) && tensorGru.initialized;
	const bool hasLstm = (netType == TYPE_LSTM) && tensorLstm.initialized;
	if (!hasDff && !hasRnn && !hasGru && !hasLstm)
	{
		weights = meat.getWeights();
		meat.addBiasWeights(weights);
		return weights;
	}

	if (hasDff)
	{
		for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
		{
			const TensorDFFState::Transition& tr = tensorDff.T[t];
			for (unsigned int j = 0; j < tr.out; ++j)
			{
				const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(tr.in);
				for (unsigned int i = 0; i < tr.in; ++i)
					weights.addFloat(tr.W[rowOff + i]);
				weights.addString(',');
			}
			weights.addString(';');
		}

		weights.addString('B');
		for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
		{
			const TensorDFFState::Transition& tr = tensorDff.T[t];
			double sum = 0.0;
			for (size_t i = 0; i < tr.bias.size(); ++i)
				sum += static_cast<double>(tr.bias[i]);
			const float mean = (tr.bias.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(tr.bias.size())));
			weights.addFloat(mean);
		}
		return weights;
	}

	if (hasRnn)
	{
		// Hidden layers: Wxh only
		for (size_t l = 0; l < tensorRnn.H.size(); ++l)
		{
			const TensorRNNState::Hidden& hl = tensorRnn.H[l];
			const unsigned int prevSize = hl.in;
			const unsigned int curSize = hl.h;
			for (unsigned int i = 0; i < curSize; ++i)
			{
				const size_t rowOff = static_cast<size_t>(i) * static_cast<size_t>(prevSize);
				for (unsigned int p = 0; p < prevSize; ++p)
					weights.addFloat(hl.Wxh[rowOff + p]);
				weights.addString(',');
			}
			weights.addString(';');
		}

		// Output: Why
		{
			const unsigned int prevSize = tensorRnn.O.in;
			const unsigned int outSize = tensorRnn.O.out;
			for (unsigned int k = 0; k < outSize; ++k)
			{
				const size_t rowOff = static_cast<size_t>(k) * static_cast<size_t>(prevSize);
				for (unsigned int i = 0; i < prevSize; ++i)
					weights.addFloat(tensorRnn.O.Why[rowOff + i]);
				weights.addString(',');
			}
			weights.addString(';');
		}

		weights.addString('B');
		for (size_t l = 0; l < tensorRnn.H.size(); ++l)
		{
			const TensorRNNState::Hidden& hl = tensorRnn.H[l];
			double sum = 0.0;
			for (size_t i = 0; i < hl.bias.size(); ++i)
				sum += static_cast<double>(hl.bias[i]);
			const float mean = (hl.bias.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(hl.bias.size())));
			weights.addFloat(mean);
		}
		{
			double sum = 0.0;
			for (size_t i = 0; i < tensorRnn.O.bias.size(); ++i)
				sum += static_cast<double>(tensorRnn.O.bias[i]);
			const float mean = (tensorRnn.O.bias.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(tensorRnn.O.bias.size())));
			weights.addFloat(mean);
		}
		return weights;
	}

	const TensorGatedState& tg = hasGru ? tensorGru : tensorLstm;
	// Hidden layers: W only (input-side weights); omit U (recurrent)
	for (size_t l = 0; l < tg.H.size(); ++l)
	{
		const TensorGatedState::Hidden& hl = tg.H[l];
		const unsigned int prevSize = hl.in;
		const unsigned int curSize = hl.h;
		const unsigned int gateCount = tg.gateCount;
		for (unsigned int u = 0; u < curSize; ++u)
		{
			for (unsigned int g = 0; g < gateCount; ++g)
			{
				const size_t wBase =
				    (static_cast<size_t>(g) * static_cast<size_t>(curSize) + static_cast<size_t>(u)) * static_cast<size_t>(prevSize);
				for (unsigned int p = 0; p < prevSize; ++p)
					weights.addFloat(hl.W[wBase + p]);
			}
			weights.addString(',');
		}
		weights.addString(';');
	}

	// Output: Why
	{
		const unsigned int prevSize = tg.O.in;
		const unsigned int outSize = tg.O.out;
		for (unsigned int k = 0; k < outSize; ++k)
		{
			const size_t rowOff = static_cast<size_t>(k) * static_cast<size_t>(prevSize);
			for (unsigned int i = 0; i < prevSize; ++i)
				weights.addFloat(tg.O.Why[rowOff + i]);
			weights.addString(',');
		}
		weights.addString(';');
	}

	weights.addString('B');
	for (size_t l = 0; l < tg.H.size(); ++l)
	{
		const TensorGatedState::Hidden& hl = tg.H[l];
		double sum = 0.0;
		for (size_t i = 0; i < hl.bias.size(); ++i)
			sum += static_cast<double>(hl.bias[i]);
		const float mean = (hl.bias.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(hl.bias.size())));
		weights.addFloat(mean);
	}
	{
		double sum = 0.0;
		for (size_t i = 0; i < tg.O.bias.size(); ++i)
			sum += static_cast<double>(tg.O.bias[i]);
		const float mean = (tg.O.bias.empty() ? 0.0f : static_cast<float>(sum / static_cast<double>(tg.O.bias.size())));
		weights.addFloat(mean);
	}
	return weights;
}

glades::LayerBuilder& glades::NNetwork::graphMutable()
{
	// Mutable access to the legacy Node/Edge graph is forbidden during an active run.
	//
	// Rationale:
	// - During train/test, packed tensors are authoritative and may be updated many times per epoch.
	// - The graph is a derived view synchronized only at explicit inspection points.
	// - Allowing callers to mutate the graph mid-run can silently desynchronize tensors vs graph
	//   and can introduce data races if another thread is running this network.
	//
	// Enforce this by using the run lock as the source-of-truth for "is a run active?".
	if (!tryAcquireRunLock())
	{
		throw std::runtime_error(
		    "NNetwork::graphMutable() is not available during train/test (legacy graph is debug-only while running). "
		    "Use materializeGraphParameters() + graph() for read-only inspection, or mutate the graph only when idle.");
	}

	// We only needed the lock as an atomic probe; release immediately.
	releaseRunLock();
	return meat;
}

int64_t NNetwork::getCurrentTimeMilliseconds() const
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<unsigned long long>(tv.tv_sec) * 1000ULL + tv.tv_usec / 1000ULL;
}

bool glades::NNetwork::getRunning() const
{
	return running;
}

int glades::NNetwork::getEpochs() const
{
	return epochs;
}

void glades::NNetwork::stop()
{
	running = false;
}

void glades::NNetwork::setSeed(uint64_t seed)
{
	rngSeed = seed;
	glades::rng::seed_engine(rngEngine, seed);
}

glades::NNetworkStatus glades::NNetwork::train(const DataInput* newDataInput)
{
    return train(newDataInput, NULL);
}

glades::NNetworkStatus glades::NNetwork::test(const DataInput* newDataInput)
{
    return test(newDataInput, NULL);
}

namespace {
class CompositeCallbacks : public glades::ITrainingCallbacks
{
public:
	CompositeCallbacks(glades::ITrainingCallbacks* a, glades::ITrainingCallbacks* b) : cbA(a), cbB(b) {}

	virtual void onRunStart(const glades::NNetwork& net, int runType)
	{
		if (cbA) cbA->onRunStart(net, runType);
		if (cbB) cbB->onRunStart(net, runType);
	}

	virtual bool onEpochEnd(const glades::NNetwork& net, const glades::NNetworkEpochMetrics& m)
	{
		bool stop = false;
		if (cbA) stop = cbA->onEpochEnd(net, m) || stop;
		if (cbB) stop = cbB->onEpochEnd(net, m) || stop;
		return stop;
	}

	virtual void onRunEnd(const glades::NNetwork& net, int runType)
	{
		if (cbA) cbA->onRunEnd(net, runType);
		if (cbB) cbB->onRunEnd(net, runType);
	}

private:
	glades::ITrainingCallbacks* cbA;
	glades::ITrainingCallbacks* cbB;
};

class ConsoleCallbacks : public glades::ITrainingCallbacks
{
public:
	virtual void onRunStart(const glades::NNetwork& net, int runType)
	{
		const glades::NNInfo* sk = net.getNNInfo();
		if (!sk)
			return;

		if (runType == glades::NNetwork::RUN_TRAIN)
			printf("[NN] Training...\n");
		else if (runType == glades::NNetwork::RUN_TEST)
			printf("[NN] Testing...\n");

		if (runType == glades::NNetwork::RUN_TRAIN)
		{
			if (sk->getOutputType() == glades::GMath::REGRESSION)
				printf("[NN] Epochs\tR2\t\tMSE\t\tMAE\t\tRMSE\t\tLR(mult)\tGradNorm(scale)\n");

			if ((sk->getOutputType() == glades::GMath::CLASSIFICATION) ||
				(sk->getOutputType() == glades::GMath::KL))
				printf("[NN] Epochs\tAccuracy\tMCC\t\tPrecision\tRecall\t\tSpecificity\tF1 Score\n");
		}
	}

	virtual bool onEpochEnd(const glades::NNetwork& net, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		const glades::NNInfo* sk = net.getNNInfo();
		if (!sk)
			return false;

		if (m.runType != glades::NNetwork::RUN_TRAIN)
			return false;

		if (m.outputType == glades::GMath::REGRESSION)
		{
			printf("\33[2K[NN] %d\t%f%%\t%f\t%f\t%f\t%g(%g)\t%g(%g)\r",
			       m.epoch,
			       m.totalAccuracy,
			       m.totalError,
			       m.regMAE,
			       m.regRMSE,
			       m.learningRate,
			       m.lrMultiplier,
			       m.gradNorm,
			       m.gradNormScale);
			fflush(stdout);
		}
		else if ((m.outputType == glades::GMath::CLASSIFICATION) || (m.outputType == glades::GMath::KL))
		{
			if (m.epoch < 100)
			{
				printf("\33[2K[NN] %d\t\t%f%%\t%f%%\t%f%%\t%f%%\t%f%%\t%f%%\r", m.epoch,
					   m.classAccuracy, m.classMCC, m.classPrecision, m.classRecall,
					   m.classSpecificity, m.classF1);
				fflush(stdout);
			}
			else
			{
				printf("\33[2K[NN] %d\t%f%%\t%f%%\t%f%%\t%f%%\t%f%%\t%f%%\r", m.epoch,
					   m.classAccuracy, m.classMCC, m.classPrecision, m.classRecall,
					   m.classSpecificity, m.classF1);
				fflush(stdout);
			}
		}

		return false;
	}

	virtual void onRunEnd(const glades::NNetwork& net, int runType)
	{
		const glades::NNInfo* sk = net.getNNInfo();
		if (!sk)
			return;

		if (runType == glades::NNetwork::RUN_TRAIN)
			printf("\n");
		else if (runType == glades::NNetwork::RUN_TEST)
		{
			if (sk->getOutputType() == glades::GMath::REGRESSION)
			{
				printf("[NN] %s R2: %f%%\n", sk->getName().c_str(), last.totalAccuracy);
				printf("[NN] %s MSE: %f\n", sk->getName().c_str(), last.totalError);
			}
			else
			{
				printf("[NN] %s Accuracy: %f%%\n", sk->getName().c_str(), last.totalAccuracy);
			}
			if (sk->getOutputType() == glades::GMath::CLASSIFICATION)
				printf("[NN] %s MCC: %f%%\n", sk->getName().c_str(), last.classMCC);
		}

		// Historical behavior: always print model summary at end.
		// NOTE: parameters are tensor-first; materialize the legacy graph view on demand.
		const_cast<glades::NNetwork&>(net).materializeGraphParameters();
		net.graph().print(sk);
		printf("\n");
	}

private:
	glades::NNetworkEpochMetrics last;
};

class GuiCallbacks : public glades::ITrainingCallbacks
{
public:
	GuiCallbacks(GNet::GServer* s, GNet::Connection* c)
	    : server(s), conn(c), lastUpdateTime(0), sentLayerSizes(false)
	{
	}

	virtual bool onEpochEnd(const glades::NNetwork& net, const glades::NNetworkEpochMetrics& m)
	{
		// Only send during training.
		if (m.runType != glades::NNetwork::RUN_TRAIN)
			return false;

		if (!server || !conn)
			return false;

		// Rate limit: first few epochs always, then ~60fps equivalent.
		const int64_t ms = net.getCurrentTimeMilliseconds();
		const int64_t timeDiff = ms - lastUpdateTime;
		if (!((m.epoch - m.startingEpoch < 10) || (timeDiff > 16)))
			return false;

		// First epoch is random (historical behavior: skip plotting for epoch 0).
		if (m.epoch > 0)
		{
			// Learning curve point: (epoch-1, totalError)
			shmea::GList wData;
			wData.addInt(m.epoch - 1);
			wData.addFloat(m.totalError);

			shmea::GList argData;
			argData.addString("PROGRESSIVE");

			shmea::ServiceData* cData = new shmea::ServiceData(conn, "GUI_Callback");
			cData->set(wData);
			cData->setArgList(argData);
			server->send(cData);

			// Activations: first message sends layer sizes, subsequent sends activations list
			argData.clear();
			argData.addString("ACTIVATIONS");
			cData = new shmea::ServiceData(conn, "GUI_Callback");

			if (!sentLayerSizes)
			{
				sentLayerSizes = true;
				shmea::GList layerSizes;
				const glades::NNInfo* sk = net.getNNInfo();
				if (sk)
				{
					for (unsigned int cLayerCounter = 0; cLayerCounter < sk->numHiddenLayers() + 2; ++cLayerCounter)
						layerSizes.addInt(net.graph().getLayerSize(cLayerCounter));
				}
				cData->set(layerSizes);
			}
			else
			{
				cData->set(net.getNodeActivations());
			}
			cData->setArgList(argData);
			server->send(cData);

			// Weights: layer weights + bias weights
			argData.clear();
			shmea::GList obtainedWeights = net.getWeightsForGui();

			argData.addString("WEIGHTS");
			cData = new shmea::ServiceData(conn, "GUI_Callback");
			cData->set(obtainedWeights);
			cData->setArgList(argData);
			server->send(cData);
		}

		// Accuracy label
		{
			shmea::GList argData;
			argData.addString("ACC");

			shmea::GList wData;
			wData.addInt(m.epoch);
			wData.addFloat(m.totalAccuracy);

			shmea::ServiceData* cData = new shmea::ServiceData(conn, "GUI_Callback");
			cData->set(wData);
			cData->setArgList(argData);
			server->send(cData);
		}

		// Confusion matrix
		if ((m.outputType == glades::GMath::CLASSIFICATION) || (m.outputType == glades::GMath::KL))
		{
			const glades::CMatrix& cm = net.getConfusionMatrix();
			shmea::GList argData;
			argData.addString("CONF");
			argData.addFloat(cm.getOverallFalseAlarm());
			argData.addFloat(cm.getOverallRecall());

			shmea::ServiceData* cData = new shmea::ServiceData(conn, "GUI_Callback");
			cData->set(cm.getMatrix());
			cData->setArgList(argData);
			server->send(cData);
		}

		lastUpdateTime = ms;
		return false;
	}

private:
	GNet::GServer* server;
	GNet::Connection* conn;
	int64_t lastUpdateTime;
	bool sentLayerSizes;
};
} // namespace

glades::NNetworkStatus glades::NNetwork::train(const DataInput* newDataInput, ITrainingCallbacks* callbacks)
{
    return run(newDataInput, RUN_TRAIN, callbacks);
}

glades::NNetworkStatus glades::NNetwork::test(const DataInput* newDataInput, ITrainingCallbacks* callbacks)
{
    return run(newDataInput, RUN_TEST, callbacks);
}

glades::NNetworkStatus glades::NNetwork::run(const DataInput* newDataInput, int runType, ITrainingCallbacks* callbacks)
{
	// Default callbacks preserve historical behavior: console + optional GUI adapter.
	// NOTE: The *core* run loop is now in TrainingCore, which is side-effect-free and
	// does not create default callbacks.
	if (callbacks)
	{
		const glades::NNetworkStatus st = glades::TrainingCore::run(*this, newDataInput, runType, callbacks);
		if (!st.ok())
		{
			shmea::GLogger* logger = NULL;
			if (serverInstance && serverInstance->logger)
				logger = serverInstance->logger.get();
			static shmea::GLogger defaultLogger(shmea::GLogger::LOG_INFO);
			if (!logger)
				logger = &defaultLogger;
			logger->error("NNetwork", st.message.c_str());
		}
		return st;
	}

	ConsoleCallbacks consoleCb;
	GuiCallbacks guiCb(serverInstance, cConnection);
	CompositeCallbacks defaultCb(&consoleCb, (serverInstance && cConnection) ? static_cast<glades::ITrainingCallbacks*>(&guiCb) : NULL);
	{
		const glades::NNetworkStatus st = glades::TrainingCore::run(*this, newDataInput, runType, static_cast<glades::ITrainingCallbacks*>(&defaultCb));
		if (!st.ok())
		{
			shmea::GLogger* logger = NULL;
			if (serverInstance && serverInstance->logger)
				logger = serverInstance->logger.get();
			static shmea::GLogger defaultLogger(shmea::GLogger::LOG_INFO);
			if (!logger)
				logger = &defaultLogger;
			logger->error("NNetwork", st.message.c_str());
		}
		return st;
	}
}

glades::NNetworkStatus glades::NNetwork::failStatus(glades::NNetworkStatus::Code code, const std::string& message)
{
	lastStatus = glades::NNetworkStatus(code, message);
	running = false;
	return lastStatus;
}

glades::NNetworkStatus glades::NNetwork::SGDHelper(unsigned int inputRowCounter, int runType)
{
	if (!skeleton)
		return failStatus(NNetworkStatus::INVALID_STATE, "NNetwork::SGDHelper: skeleton is NULL");

	if (meat.getLayersSize() <= 0)
		return failStatus(NNetworkStatus::INVALID_STATE, "NNetwork::SGDHelper: network has no layers (LayerBuilder not built)");

	// Net-type-specific SGD implementations live in separate translation units.
	// This preserves behavior while reducing the size/complexity of this file.
	lastStatus = NNetworkStatus(NNetworkStatus::OK, std::string());
	switch (netType)
	{
	case TYPE_DFF:
		SGDHelper_DFF(inputRowCounter, runType);
		return lastStatus;
	case TYPE_RNN:
		// Recurrent helpers only run once per epoch (on row 0).
		if (inputRowCounter != 0)
			return lastStatus;
		SGDHelper_RNN(inputRowCounter, runType);
		return lastStatus;
	case TYPE_GRU:
		if (inputRowCounter != 0)
			return lastStatus;
		SGDHelper_GRU(inputRowCounter, runType);
		return lastStatus;
	case TYPE_LSTM:
		if (inputRowCounter != 0)
			return lastStatus;
		SGDHelper_LSTM(inputRowCounter, runType);
		return lastStatus;
	default:
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "NNetwork::SGDHelper: unknown netType");
	}
}

/*!
 * @brief get ID
 * @details get the network's unique ID
 * @return the network's ID
 */
int64_t glades::NNetwork::getID() const
{
	return id;
}

shmea::GString glades::NNetwork::getName() const
{
	if (!skeleton)
		return "";

	return skeleton->getName();
}

const NNInfo* glades::NNetwork::getNNInfo() const
{
	if (!skeleton)
		return NULL;

	return skeleton;
}

NNInfo* glades::NNetwork::getNNInfoMutable()
{
	return skeleton;
}

float glades::NNetwork::getAccuracy() const
{
	if (!skeleton)
		return 0.0f;

	// NOTE:
	// This method is intentionally named "getAccuracy" for historical/UI reasons.
	// It must return the primary accuracy-like score computed by the training loop:
	// - Regression: R^2 (%) in overallTotalAccuracy
	// - Classification/KL: top-1 accuracy (%) in overallTotalAccuracy
	//
	// MCC is available separately via getMCC().
	return overallTotalAccuracy;
}

float glades::NNetwork::getMCC() const
{
	if (!skeleton)
		return 0.0f;

	if ((skeleton->getOutputType() == GMath::CLASSIFICATION) ||
	    (skeleton->getOutputType() == GMath::KL))
		return confusionMatrix.getOverallMCC();

	return 0.0f;
}

const glades::CMatrix& glades::NNetwork::getConfusionMatrix() const
{
	return confusionMatrix;
}

const shmea::GList& glades::NNetwork::getNodeActivations() const
{
	return cNodeActivations;
}

bool glades::NNetwork::load(const shmea::GString& netName)
{
	// Lifetime safety: loading a network always creates an owned NNInfo instance.
	// This avoids leaving `skeleton` pointing at a borrowed NNInfo with unclear lifetime.
	ownedSkeleton = shmea::GPointer<NNInfo>(new NNInfo(netName));
	skeleton = ownedSkeleton.get();

	// Load the nn state information
	if (!skeleton->load(netName))
		return false;

	//return meat.load(netName);
	return true;
}

bool glades::NNetwork::save() const
{
	if (!skeleton)
		return false;

	skeleton->save();
	return true;
	/// return meat.save(skeleton->getName());
}

void glades::NNetwork::setServer(GNet::GServer* newServer, GNet::Connection* newConnection)
{
	serverInstance = newServer;
	cConnection = newConnection;
}

shmea::GList glades::NNetwork::getResults() const
{
	return results;
}

void glades::NNetwork::clean()
{
	id = -1;
	ownedSkeleton.reset();
	skeleton = NULL;
	di = NULL;
	// Safety: LayerBuilder stores a non-owning dataset pointer used by getInputLayer().
	// Ensure it is detached when the network is cleaned.
	meat.detachDataInput();
	confusionMatrix.clean();
	serverInstance = NULL;
	cConnection = NULL;
	results.clear();
	nbRecord.clear();
	epochs = 0;
	overallTotalError = 0.0f;
	overallTotalAccuracy = 0.0f;
	overallClassAccuracy = 0.0f;
	overallClassPrecision = 0.0f;
	overallClassRecall = 0.0f;
	overallClassF1 = 0.0f;
	minibatchSize = NNInfo::BATCH_STOCHASTIC;
	changeInputLayers = false;
	running = false;
	lastStatus = NNetworkStatus(NNetworkStatus::OK, std::string());
	graphWeightsDirty = false;
	// Reset training configuration to defaults.
	trainingConfig = TrainingConfig();
	// Schedule bookkeeping resets each run
	lrScheduleMultiplier = 1.0f;
	lastGradNorm = 0.0f;
	lastGradNormScale = 1.0f;

	// Reset tensor training state caches (they will be re-initialized on next run).
	tensorDff.reset();
	tensorRnn.reset();
	tensorGru.reset();
	tensorLstm.reset();

	// Epoch-scoped metric accumulators
	regSSE = 0.0;
	regSAE = 0.0;
	regSumY = 0.0;
	regSumY2 = 0.0;
	regCount = 0ULL;
	clsCorrect = 0ULL;
	clsTotal = 0ULL;
}

void glades::NNetwork::syncGraphWeightsFromTensorsIfDirty()
{
	using namespace glades::param_layout;
	if (!graphWeightsDirty)
		return;

	// If there's no graph built yet, keep the dirty flag set; callers should build first.
	if (meat.getLayersSize() <= 0)
		return;

	// DFF: sync dense tensors into Node edges.
	if (netType == TYPE_DFF)
	{
		if (!tensorDff.initialized)
		{
			graphWeightsDirty = false;
			return;
		}

		for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
		{
			const TensorDFFState::Transition& tr = tensorDff.T[t];
			Layer* outLayer = meat.getOutputLayer(t + 1);
			if (!outLayer)
				continue;

			for (unsigned int j = 0; j < tr.out; ++j)
			{
				Node* node = meat.getOutputNode(outLayer, j);
				if (!node)
					continue;
				const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(tr.in);
				for (unsigned int i = 0; i < tr.in; ++i)
					node->setEdgeWeight(dense_weight_edge(i), tr.W[rowOff + i]);
				// Bias edge (final edge).
				if (j < tr.bias.size())
					node->setEdgeWeight(dense_bias_edge(tr.in), tr.bias[j]);
			}
		}

		graphWeightsDirty = false;
		return;
	}

	// RNN: sync packed Wxh/Whh + biases.
	if (netType == TYPE_RNN)
	{
		if (!tensorRnn.initialized)
		{
			graphWeightsDirty = false;
			return;
		}
		const int H = static_cast<int>(tensorRnn.H.size());
		for (int l = 0; l < H; ++l)
		{
			const TensorRNNState::Hidden& hl = tensorRnn.H[static_cast<size_t>(l)];
			const unsigned int prevSize = hl.in;
			const unsigned int curSize = hl.h;
			Layer* layer = meat.getOutputLayer(static_cast<unsigned int>(l) + 1);
			if (!layer)
				continue;

			for (unsigned int i = 0; i < curSize; ++i)
			{
				Node* node = meat.getOutputNode(layer, i);
				if (!node)
					continue;
				for (unsigned int p = 0; p < prevSize; ++p)
					node->setEdgeWeight(rnn_wx_edge(p), hl.Wxh[static_cast<size_t>(i) * static_cast<size_t>(prevSize) + p]);
				if (i < hl.bias.size())
					node->setEdgeWeight(rnn_bias_edge(prevSize), hl.bias[i]);

				Node* ctx = node->getContextNode();
				if (ctx)
				{
					for (unsigned int j = 0; j < curSize; ++j)
						ctx->setEdgeWeight(rnn_wh_edge(j), hl.Whh[static_cast<size_t>(i) * static_cast<size_t>(curSize) + j]);

					// Preserve conservative stability clamp for linear recurrent weights.
					if (skeleton)
					{
						const int actFx = skeleton->getActivationType(static_cast<unsigned int>(l));
						if (actFx == GMath::LINEAR)
						{
							for (unsigned int j = 0; j < curSize; ++j)
							{
								const float w = ctx->getEdgeWeight(rnn_wh_edge(j));
								float ww = w;
								if (ww < -1.0f)
									ww = -1.0f;
								else if (ww > 1.0f)
									ww = 1.0f;
								ctx->setEdgeWeight(rnn_wh_edge(j), ww);
							}
						}
					}
				}
			}
		}

		// Output transition
		{
			Layer* outLayer = meat.getOutputLayer(static_cast<unsigned int>(H) + 1);
			if (outLayer)
			{
				const unsigned int prevSize = tensorRnn.O.in;
				const unsigned int outSize = tensorRnn.O.out;
				for (unsigned int k = 0; k < outSize; ++k)
				{
					Node* node = meat.getOutputNode(outLayer, k);
					if (!node)
						continue;
					for (unsigned int i = 0; i < prevSize; ++i)
						node->setEdgeWeight(dense_weight_edge(i), tensorRnn.O.Why[static_cast<size_t>(k) * static_cast<size_t>(prevSize) + i]);
					if (k < tensorRnn.O.bias.size())
						node->setEdgeWeight(dense_bias_edge(prevSize), tensorRnn.O.bias[k]);
				}
			}
		}

		graphWeightsDirty = false;
		return;
	}

	// GRU/LSTM: sync packed gate tensors.
	if (netType == TYPE_GRU || netType == TYPE_LSTM)
	{
		const TensorGatedState& tg = (netType == TYPE_GRU) ? tensorGru : tensorLstm;
		const unsigned int gateCount = (netType == TYPE_GRU) ? 3u : 4u;
		if (!tg.initialized)
		{
			graphWeightsDirty = false;
			return;
		}

		const int H = static_cast<int>(tg.H.size());
		for (int l = 0; l < H; ++l)
		{
			const TensorGatedState::Hidden& hl = tg.H[static_cast<size_t>(l)];
			const unsigned int prevSize = hl.in;
			const unsigned int curSize = hl.h;
			const Gated layout = {prevSize, curSize, gateCount};

			Layer* layer = meat.getOutputLayer(static_cast<unsigned int>(l) + 1);
			if (!layer)
				continue;

			for (unsigned int i = 0; i < curSize; ++i)
			{
				Node* node = meat.getOutputNode(layer, i);
				if (!node)
					continue;
				Node* ctx = node->getContextNode();

				for (unsigned int g = 0; g < gateCount; ++g)
				{
					const size_t wBase = (static_cast<size_t>(g) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(prevSize);
					for (unsigned int p = 0; p < prevSize; ++p)
						node->setEdgeWeight(layout.w_edge(g, p), hl.W[wBase + p]);

					const size_t bIdx = static_cast<size_t>(g) * static_cast<size_t>(curSize) + static_cast<size_t>(i);
					if (bIdx < hl.bias.size())
						node->setEdgeWeight(layout.b_edge(g), hl.bias[bIdx]);

					if (ctx)
					{
						const size_t uBase = (static_cast<size_t>(g) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(curSize);
						for (unsigned int j = 0; j < curSize; ++j)
							ctx->setEdgeWeight(layout.u_edge(g, j), hl.U[uBase + j]);
					}
				}
			}
		}

		// Output transition
		{
			Layer* outLayer = meat.getOutputLayer(static_cast<unsigned int>(H) + 1);
			if (outLayer)
			{
				const unsigned int prevSize = tg.O.in;
				const unsigned int outSize = tg.O.out;
				for (unsigned int k = 0; k < outSize; ++k)
				{
					Node* node = meat.getOutputNode(outLayer, k);
					if (!node)
						continue;
					for (unsigned int i = 0; i < prevSize; ++i)
						node->setEdgeWeight(dense_weight_edge(i), tg.O.Why[static_cast<size_t>(k) * static_cast<size_t>(prevSize) + i]);
					if (k < tg.O.bias.size())
						node->setEdgeWeight(dense_bias_edge(prevSize), tg.O.bias[k]);
				}
			}
		}

		graphWeightsDirty = false;
		return;
	}

	// Unknown net type: clear flag to avoid repeated work.
	graphWeightsDirty = false;
}

bool glades::NNetwork::ensureTensorParametersInitializedFromGraph()
{
	using namespace glades::param_layout;
	if (!skeleton)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitializedFromGraph: skeleton is NULL");
		return false;
	}
	if (meat.getLayersSize() <= 0)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitializedFromGraph: graph (meat) is not built");
		return false;
	}

	// DFF
	if (netType == TYPE_DFF)
	{
		if (tensorDff.initialized)
			return true;

		const unsigned int inSize = meat.getLayerSize(0);
		const int H = skeleton->numHiddenLayers();
		const unsigned int outSize = skeleton->getOutputLayerSize();
		if (inSize == 0u || outSize == 0u)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitializedFromGraph: invalid DFF sizes (input/output == 0)");
			return false;
		}

		std::vector<unsigned int> wantSizes;
		wantSizes.reserve(static_cast<size_t>(H) + 2u);
		wantSizes.push_back(inSize);
		for (int l = 0; l < H; ++l)
			wantSizes.push_back(meat.getLayerSize(static_cast<unsigned int>(l) + 1u));
		wantSizes.push_back(outSize);

		tensorDff.reset();
		tensorDff.sizes = wantSizes;

		const unsigned int numTransitions = (wantSizes.size() >= 2u) ? static_cast<unsigned int>(wantSizes.size() - 1u) : 0u;
		tensorDff.T.resize(numTransitions);
		tensorDff.a.resize(wantSizes.size());
		tensorDff.delta.resize(wantSizes.size());

		for (unsigned int li = 0; li < wantSizes.size(); ++li)
		{
			tensorDff.a[li].assign(wantSizes[li], 0.0f);
			if (li == 0u)
				tensorDff.delta[li].clear();
			else
				tensorDff.delta[li].assign(wantSizes[li], 0.0f);
		}

		for (unsigned int t = 0; t < numTransitions; ++t)
		{
			const unsigned int in = wantSizes[t];
			const unsigned int out = wantSizes[t + 1u];
			TensorDFFState::Transition& tr = tensorDff.T[t];
			tr.in = in;
			tr.out = out;
			tr.W.assign(static_cast<size_t>(out) * static_cast<size_t>(in), 0.0f);
			tr.vW.assign(tr.W.size(), 0.0f);
			tr.gW.assign(tr.W.size(), 0.0f);
			tr.bias.assign(out, 0.0f);
			tr.gBias.assign(out, 0.0f);

			Layer* outLayer = meat.getOutputLayer(t + 1u);
			if (!outLayer)
				continue;
			for (unsigned int j = 0; j < out; ++j)
			{
				Node* node = meat.getOutputNode(outLayer, j);
				if (!node)
					continue;
				for (unsigned int i = 0; i < in; ++i)
					tr.W[static_cast<size_t>(j) * static_cast<size_t>(in) + i] = node->getEdgeWeight(dense_weight_edge(i));
				tr.bias[j] = node->getEdgeWeight(dense_bias_edge(in));
			}
		}

		tensorDff.batchCount = 0u;
		tensorDff.initialized = true;
		graphWeightsDirty = false; // graph was the source for this bootstrap
		return true;
	}

	// RNN
	if (netType == TYPE_RNN)
	{
		if (tensorRnn.initialized)
			return true;

		const unsigned int inputSize = meat.getLayerSize(0);
		const int H = skeleton->numHiddenLayers();
		const unsigned int outSize = skeleton->getOutputLayerSize();
		if (inputSize == 0u || outSize == 0u || H <= 0)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitializedFromGraph: invalid RNN sizes (hidden/input/output)");
			return false;
		}

		std::vector<unsigned int> hiddenSizes;
		hiddenSizes.resize(static_cast<size_t>(H));
		for (int l = 0; l < H; ++l)
			hiddenSizes[static_cast<size_t>(l)] = meat.getLayerSize(static_cast<unsigned int>(l) + 1u);

		tensorRnn.reset();
		tensorRnn.initialized = true;
		tensorRnn.inputSize = inputSize;
		tensorRnn.outSize = outSize;
		tensorRnn.hiddenSizes = hiddenSizes;
		tensorRnn.H.resize(static_cast<size_t>(H));

		for (int l = 0; l < H; ++l)
		{
			const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[static_cast<size_t>(l - 1)];
			const unsigned int curSize = hiddenSizes[static_cast<size_t>(l)];
			TensorRNNState::Hidden& hl = tensorRnn.H[static_cast<size_t>(l)];
			hl.in = prevSize;
			hl.h = curSize;
			hl.Wxh.assign(static_cast<size_t>(curSize) * static_cast<size_t>(prevSize), 0.0f);
			hl.Whh.assign(static_cast<size_t>(curSize) * static_cast<size_t>(curSize), 0.0f);
			hl.vWxh.assign(hl.Wxh.size(), 0.0f);
			hl.vWhh.assign(hl.Whh.size(), 0.0f);
			hl.gWxh.assign(hl.Wxh.size(), 0.0f);
			hl.gWhh.assign(hl.Whh.size(), 0.0f);
			hl.bias.assign(curSize, 0.0f);
			hl.gBias.assign(curSize, 0.0f);

			Layer* layer = meat.getOutputLayer(static_cast<unsigned int>(l) + 1u);
			if (!layer)
				continue;
			for (unsigned int i = 0; i < curSize; ++i)
			{
				Node* node = meat.getOutputNode(layer, i);
				if (!node)
					continue;
				for (unsigned int p = 0; p < prevSize; ++p)
					hl.Wxh[static_cast<size_t>(i) * static_cast<size_t>(prevSize) + p] = node->getEdgeWeight(rnn_wx_edge(p));
				hl.bias[i] = node->getEdgeWeight(rnn_bias_edge(prevSize));
				Node* ctx = node->getContextNode();
				if (ctx)
				{
					for (unsigned int j = 0; j < curSize; ++j)
						hl.Whh[static_cast<size_t>(i) * static_cast<size_t>(curSize) + j] = ctx->getEdgeWeight(rnn_wh_edge(j));
				}
			}
		}

		// Output transition
		{
			const unsigned int prevSize = hiddenSizes[static_cast<size_t>(H - 1)];
			tensorRnn.O.in = prevSize;
			tensorRnn.O.out = outSize;
			tensorRnn.O.Why.assign(static_cast<size_t>(outSize) * static_cast<size_t>(prevSize), 0.0f);
			tensorRnn.O.vWhy.assign(tensorRnn.O.Why.size(), 0.0f);
			tensorRnn.O.gWhy.assign(tensorRnn.O.Why.size(), 0.0f);
			tensorRnn.O.bias.assign(outSize, 0.0f);
			tensorRnn.O.gBias.assign(outSize, 0.0f);

			Layer* outLayer = meat.getOutputLayer(static_cast<unsigned int>(H) + 1u);
			if (outLayer)
			{
				for (unsigned int k = 0; k < outSize; ++k)
				{
					Node* node = meat.getOutputNode(outLayer, k);
					if (!node)
						continue;
					for (unsigned int i = 0; i < prevSize; ++i)
						tensorRnn.O.Why[static_cast<size_t>(k) * static_cast<size_t>(prevSize) + i] = node->getEdgeWeight(dense_weight_edge(i));
					tensorRnn.O.bias[k] = node->getEdgeWeight(dense_bias_edge(prevSize));
				}
			}
		}

		graphWeightsDirty = false;
		return true;
	}

	// GRU / LSTM (gated)
	if (netType == TYPE_GRU || netType == TYPE_LSTM)
	{
		TensorGatedState& tg = (netType == TYPE_GRU) ? tensorGru : tensorLstm;
		const unsigned int gateCount = (netType == TYPE_GRU) ? 3u : 4u;
		if (tg.initialized)
			return true;

		const unsigned int inputSize = meat.getLayerSize(0);
		const int H = skeleton->numHiddenLayers();
		const unsigned int outSize = skeleton->getOutputLayerSize();
		if (inputSize == 0u || outSize == 0u || H <= 0)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "ensureTensorParametersInitializedFromGraph: invalid gated sizes (hidden/input/output)");
			return false;
		}

		std::vector<unsigned int> hiddenSizes;
		hiddenSizes.resize(static_cast<size_t>(H));
		for (int l = 0; l < H; ++l)
			hiddenSizes[static_cast<size_t>(l)] = meat.getLayerSize(static_cast<unsigned int>(l) + 1u);

		tg.reset();
		tg.initialized = true;
		tg.inputSize = inputSize;
		tg.outSize = outSize;
		tg.gateCount = gateCount;
		tg.hiddenSizes = hiddenSizes;
		tg.H.resize(static_cast<size_t>(H));

		for (int l = 0; l < H; ++l)
		{
			const unsigned int prevSize = (l == 0) ? inputSize : hiddenSizes[static_cast<size_t>(l - 1)];
			const unsigned int curSize = hiddenSizes[static_cast<size_t>(l)];
			TensorGatedState::Hidden& hl = tg.H[static_cast<size_t>(l)];
			hl.in = prevSize;
			hl.h = curSize;
			hl.W.assign(static_cast<size_t>(gateCount) * static_cast<size_t>(curSize) * static_cast<size_t>(prevSize), 0.0f);
			hl.U.assign(static_cast<size_t>(gateCount) * static_cast<size_t>(curSize) * static_cast<size_t>(curSize), 0.0f);
			hl.vW.assign(hl.W.size(), 0.0f);
			hl.vU.assign(hl.U.size(), 0.0f);
			hl.gW.assign(hl.W.size(), 0.0f);
			hl.gU.assign(hl.U.size(), 0.0f);
			hl.bias.assign(static_cast<size_t>(gateCount) * static_cast<size_t>(curSize), 0.0f);
			hl.gBias.assign(hl.bias.size(), 0.0f);

			Layer* layer = meat.getOutputLayer(static_cast<unsigned int>(l) + 1u);
			if (!layer)
				continue;
			const Gated layout = {prevSize, curSize, gateCount};
			for (unsigned int i = 0; i < curSize; ++i)
			{
				Node* node = meat.getOutputNode(layer, i);
				if (!node)
					continue;
				Node* ctx = node->getContextNode();
				for (unsigned int g = 0; g < gateCount; ++g)
				{
					for (unsigned int p = 0; p < prevSize; ++p)
					{
						const size_t wIdx =
						    (static_cast<size_t>(g) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(prevSize) +
						    static_cast<size_t>(p);
						hl.W[wIdx] = node->getEdgeWeight(layout.w_edge(g, p));
					}
					hl.bias[static_cast<size_t>(g) * static_cast<size_t>(curSize) + static_cast<size_t>(i)] = node->getEdgeWeight(layout.b_edge(g));
					if (ctx)
					{
						for (unsigned int j = 0; j < curSize; ++j)
						{
							const size_t uIdx =
							    (static_cast<size_t>(g) * static_cast<size_t>(curSize) + static_cast<size_t>(i)) * static_cast<size_t>(curSize) +
							    static_cast<size_t>(j);
							hl.U[uIdx] = ctx->getEdgeWeight(layout.u_edge(g, j));
						}
					}
				}
			}
		}

		// Output transition
		{
			const unsigned int prevSize = hiddenSizes[static_cast<size_t>(H - 1)];
			tg.O.in = prevSize;
			tg.O.out = outSize;
			tg.O.Why.assign(static_cast<size_t>(outSize) * static_cast<size_t>(prevSize), 0.0f);
			tg.O.vWhy.assign(tg.O.Why.size(), 0.0f);
			tg.O.gWhy.assign(tg.O.Why.size(), 0.0f);
			tg.O.bias.assign(outSize, 0.0f);
			tg.O.gBias.assign(outSize, 0.0f);

			Layer* outLayer = meat.getOutputLayer(static_cast<unsigned int>(H) + 1u);
			if (outLayer)
			{
				for (unsigned int k = 0; k < outSize; ++k)
				{
					Node* node = meat.getOutputNode(outLayer, k);
					if (!node)
						continue;
					for (unsigned int i = 0; i < prevSize; ++i)
						tg.O.Why[static_cast<size_t>(k) * static_cast<size_t>(prevSize) + i] = node->getEdgeWeight(dense_weight_edge(i));
					tg.O.bias[k] = node->getEdgeWeight(dense_bias_edge(prevSize));
				}
			}
		}

		graphWeightsDirty = false;
		return true;
	}

	lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "ensureTensorParametersInitializedFromGraph: unknown netType");
	return false;
}

namespace {
static const char* kTensorWeightsMagic = "GLADES_TENSOR_WEIGHTS";
static const int kTensorWeightsVersion = 1;

static bool write_vec(std::ostream& out, const char* tag, const std::vector<float>& v)
{
	out << tag << " " << v.size();
	for (size_t i = 0; i < v.size(); ++i)
		out << " " << v[i];
	out << "\n";
	return static_cast<bool>(out);
}

static bool read_tag(std::istream& in, std::string& tagOut)
{
	tagOut.clear();
	return static_cast<bool>(in >> tagOut);
}

static bool read_vec(std::istream& in, std::vector<float>& v)
{
	size_t n = 0;
	if (!(in >> n))
		return false;
	v.assign(n, 0.0f);
	for (size_t i = 0; i < n; ++i)
	{
		if (!(in >> v[i]))
			return false;
	}
	return true;
}
} // namespace

glades::NNetworkStatus glades::NNetwork::saveTensorWeightsToFile(const std::string& filePath) const
{
	if (filePath.empty())
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveTensorWeightsToFile: filePath is empty");
	if (!skeleton)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "saveTensorWeightsToFile: skeleton is null");

	// Ensure tensors exist (bootstrap from graph if needed).
	if (!const_cast<glades::NNetwork*>(this)->ensureTensorParametersInitializedFromGraph())
		return lastStatus;

	std::ofstream out(filePath.c_str());
	if (!out)
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: unable to open file for writing");

	out << kTensorWeightsMagic << "\n";
	out << "version " << kTensorWeightsVersion << "\n";
	out << "netType " << netType << "\n";

	if (netType == TYPE_DFF)
	{
		out << "dff.transitions " << tensorDff.T.size() << "\n";
		for (size_t t = 0; t < tensorDff.T.size(); ++t)
		{
			const TensorDFFState::Transition& tr = tensorDff.T[t];
			out << "dff.t " << t << " " << tr.in << " " << tr.out << "\n";
			if (!write_vec(out, "W", tr.W)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (DFF W)");
			if (!write_vec(out, "bias", tr.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (DFF bias)");
		}
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_RNN)
	{
		out << "rnn.hiddenLayers " << tensorRnn.H.size() << "\n";
		out << "rnn.inputSize " << tensorRnn.inputSize << "\n";
		out << "rnn.outSize " << tensorRnn.outSize << "\n";
		for (size_t l = 0; l < tensorRnn.H.size(); ++l)
		{
			const TensorRNNState::Hidden& hl = tensorRnn.H[l];
			out << "rnn.h " << l << " " << hl.in << " " << hl.h << "\n";
			if (!write_vec(out, "Wxh", hl.Wxh)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (RNN Wxh)");
			if (!write_vec(out, "Whh", hl.Whh)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (RNN Whh)");
			if (!write_vec(out, "bias", hl.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (RNN bias)");
		}
		out << "rnn.o " << tensorRnn.O.in << " " << tensorRnn.O.out << "\n";
		if (!write_vec(out, "Why", tensorRnn.O.Why)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (RNN Why)");
		if (!write_vec(out, "bias", tensorRnn.O.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (RNN out bias)");
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_GRU || netType == TYPE_LSTM)
	{
		const TensorGatedState& tg = (netType == TYPE_GRU) ? tensorGru : tensorLstm;
		out << "gated.gateCount " << tg.gateCount << "\n";
		out << "gated.hiddenLayers " << tg.H.size() << "\n";
		out << "gated.inputSize " << tg.inputSize << "\n";
		out << "gated.outSize " << tg.outSize << "\n";
		for (size_t l = 0; l < tg.H.size(); ++l)
		{
			const TensorGatedState::Hidden& hl = tg.H[l];
			out << "gated.h " << l << " " << hl.in << " " << hl.h << "\n";
			if (!write_vec(out, "W", hl.W)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (gated W)");
			if (!write_vec(out, "U", hl.U)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (gated U)");
			if (!write_vec(out, "bias", hl.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (gated bias)");
		}
		out << "gated.o " << tg.O.in << " " << tg.O.out << "\n";
		if (!write_vec(out, "Why", tg.O.Why)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (gated Why)");
		if (!write_vec(out, "bias", tg.O.bias)) return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveTensorWeightsToFile: write failed (gated out bias)");
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveTensorWeightsToFile: unknown netType");
}

glades::NNetworkStatus glades::NNetwork::loadTensorWeightsFromFile(const std::string& filePath)
{
	if (filePath.empty())
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadTensorWeightsFromFile: filePath is empty");
	if (!skeleton)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: skeleton is null");

	std::ifstream in(filePath.c_str());
	if (!in)
		return failStatus(NNetworkStatus::INTERNAL_ERROR, "loadTensorWeightsFromFile: unable to open file");

	std::string magic;
	std::getline(in, magic);
	if (magic != kTensorWeightsMagic)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: magic mismatch");

	std::string tag;
	int version = 0;
	if (!read_tag(in, tag) || tag != "version" || !(in >> version))
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing version");
	if (version != kTensorWeightsVersion)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: unsupported tensor weights version");

	int fileNetType = -1;
	if (!read_tag(in, tag) || tag != "netType" || !(in >> fileNetType))
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing netType");
	if (fileNetType != netType)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: netType mismatch vs manifest");

	if (netType == TYPE_DFF)
	{
		size_t transitions = 0;
		if (!read_tag(in, tag) || tag != "dff.transitions" || !(in >> transitions))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing dff.transitions");

		tensorDff.reset();
		tensorDff.T.resize(transitions);
		tensorDff.sizes.clear();
		tensorDff.sizes.reserve(transitions + 1u);

		for (size_t t = 0; t < transitions; ++t)
		{
			size_t tIdx = 0;
			unsigned int inSize = 0, outSize = 0;
			if (!read_tag(in, tag) || tag != "dff.t" || !(in >> tIdx >> inSize >> outSize))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed dff.t header");
			if (tIdx != t)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: dff transition index mismatch");

			TensorDFFState::Transition& tr = tensorDff.T[t];
			tr.in = inSize;
			tr.out = outSize;

			if (t == 0u)
				tensorDff.sizes.push_back(inSize);
			tensorDff.sizes.push_back(outSize);

			if (!read_tag(in, tag) || tag != "W" || !read_vec(in, tr.W))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read DFF W");
			if (!read_tag(in, tag) || tag != "bias" || !read_vec(in, tr.bias))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read DFF bias");

			// Reset optimizer/grads.
			tr.vW.assign(tr.W.size(), 0.0f);
			tr.gW.assign(tr.W.size(), 0.0f);
			tr.gBias.assign(tr.bias.size(), 0.0f);
		}

		// Allocate activations/deltas for shape (training will overwrite).
		tensorDff.a.resize(tensorDff.sizes.size());
		tensorDff.delta.resize(tensorDff.sizes.size());
		for (size_t li = 0; li < tensorDff.sizes.size(); ++li)
		{
			tensorDff.a[li].assign(tensorDff.sizes[li], 0.0f);
			if (li == 0u) tensorDff.delta[li].clear();
			else tensorDff.delta[li].assign(tensorDff.sizes[li], 0.0f);
		}
		tensorDff.batchCount = 0u;
		tensorDff.initialized = true;
		graphWeightsDirty = true; // graph is now a derived view
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_RNN)
	{
		size_t hiddenLayers = 0;
		unsigned int inputSize = 0, outSize = 0;
		if (!read_tag(in, tag) || tag != "rnn.hiddenLayers" || !(in >> hiddenLayers))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing rnn.hiddenLayers");
		if (!read_tag(in, tag) || tag != "rnn.inputSize" || !(in >> inputSize))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing rnn.inputSize");
		if (!read_tag(in, tag) || tag != "rnn.outSize" || !(in >> outSize))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing rnn.outSize");

		tensorRnn.reset();
		tensorRnn.initialized = true;
		tensorRnn.inputSize = inputSize;
		tensorRnn.outSize = outSize;
		tensorRnn.hiddenSizes.assign(hiddenLayers, 0u);
		tensorRnn.H.resize(hiddenLayers);

		for (size_t l = 0; l < hiddenLayers; ++l)
		{
			size_t lIdx = 0;
			unsigned int inSize = 0, hSize = 0;
			if (!read_tag(in, tag) || tag != "rnn.h" || !(in >> lIdx >> inSize >> hSize))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed rnn.h header");
			if (lIdx != l)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: rnn hidden index mismatch");

			TensorRNNState::Hidden& hl = tensorRnn.H[l];
			hl.in = inSize;
			hl.h = hSize;
			tensorRnn.hiddenSizes[l] = hSize;

			if (!read_tag(in, tag) || tag != "Wxh" || !read_vec(in, hl.Wxh))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read RNN Wxh");
			if (!read_tag(in, tag) || tag != "Whh" || !read_vec(in, hl.Whh))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read RNN Whh");
			if (!read_tag(in, tag) || tag != "bias" || !read_vec(in, hl.bias))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read RNN bias");

			hl.vWxh.assign(hl.Wxh.size(), 0.0f);
			hl.vWhh.assign(hl.Whh.size(), 0.0f);
			hl.gWxh.assign(hl.Wxh.size(), 0.0f);
			hl.gWhh.assign(hl.Whh.size(), 0.0f);
			hl.gBias.assign(hl.bias.size(), 0.0f);
		}

		unsigned int oIn = 0, oOut = 0;
		if (!read_tag(in, tag) || tag != "rnn.o" || !(in >> oIn >> oOut))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing rnn.o header");
		tensorRnn.O.in = oIn;
		tensorRnn.O.out = oOut;
		if (!read_tag(in, tag) || tag != "Why" || !read_vec(in, tensorRnn.O.Why))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read RNN Why");
		if (!read_tag(in, tag) || tag != "bias" || !read_vec(in, tensorRnn.O.bias))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read RNN out bias");
		tensorRnn.O.vWhy.assign(tensorRnn.O.Why.size(), 0.0f);
		tensorRnn.O.gWhy.assign(tensorRnn.O.Why.size(), 0.0f);
		tensorRnn.O.gBias.assign(tensorRnn.O.bias.size(), 0.0f);

		graphWeightsDirty = true;
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	if (netType == TYPE_GRU || netType == TYPE_LSTM)
	{
		unsigned int gateCount = 0;
		size_t hiddenLayers = 0;
		unsigned int inputSize = 0, outSize = 0;
		if (!read_tag(in, tag) || tag != "gated.gateCount" || !(in >> gateCount))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing gated.gateCount");
		if (!read_tag(in, tag) || tag != "gated.hiddenLayers" || !(in >> hiddenLayers))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing gated.hiddenLayers");
		if (!read_tag(in, tag) || tag != "gated.inputSize" || !(in >> inputSize))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing gated.inputSize");
		if (!read_tag(in, tag) || tag != "gated.outSize" || !(in >> outSize))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing gated.outSize");

		TensorGatedState& tg = (netType == TYPE_GRU) ? tensorGru : tensorLstm;
		tg.reset();
		tg.initialized = true;
		tg.inputSize = inputSize;
		tg.outSize = outSize;
		tg.gateCount = gateCount;
		tg.hiddenSizes.assign(hiddenLayers, 0u);
		tg.H.resize(hiddenLayers);

		for (size_t l = 0; l < hiddenLayers; ++l)
		{
			size_t lIdx = 0;
			unsigned int inSize = 0, hSize = 0;
			if (!read_tag(in, tag) || tag != "gated.h" || !(in >> lIdx >> inSize >> hSize))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: malformed gated.h header");
			if (lIdx != l)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: gated hidden index mismatch");

			TensorGatedState::Hidden& hl = tg.H[l];
			hl.in = inSize;
			hl.h = hSize;
			tg.hiddenSizes[l] = hSize;

			if (!read_tag(in, tag) || tag != "W" || !read_vec(in, hl.W))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read gated W");
			if (!read_tag(in, tag) || tag != "U" || !read_vec(in, hl.U))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read gated U");
			if (!read_tag(in, tag) || tag != "bias" || !read_vec(in, hl.bias))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read gated bias");

			hl.vW.assign(hl.W.size(), 0.0f);
			hl.vU.assign(hl.U.size(), 0.0f);
			hl.gW.assign(hl.W.size(), 0.0f);
			hl.gU.assign(hl.U.size(), 0.0f);
			hl.gBias.assign(hl.bias.size(), 0.0f);
		}

		unsigned int oIn = 0, oOut = 0;
		if (!read_tag(in, tag) || tag != "gated.o" || !(in >> oIn >> oOut))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: missing gated.o header");
		tg.O.in = oIn;
		tg.O.out = oOut;
		if (!read_tag(in, tag) || tag != "Why" || !read_vec(in, tg.O.Why))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read gated Why");
		if (!read_tag(in, tag) || tag != "bias" || !read_vec(in, tg.O.bias))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadTensorWeightsFromFile: failed to read gated out bias");
		tg.O.vWhy.assign(tg.O.Why.size(), 0.0f);
		tg.O.gWhy.assign(tg.O.Why.size(), 0.0f);
		tg.O.gBias.assign(tg.O.bias.size(), 0.0f);

		graphWeightsDirty = true;
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadTensorWeightsFromFile: unknown netType");
}

void glades::NNetwork::setLearningRateScheduleNone()
{
	trainingConfig.lrSchedule.setNone();
	lrScheduleMultiplier = 1.0f;
}

void glades::NNetwork::setLearningRateScheduleStep(int stepSizeEpochs, float gamma)
{
	trainingConfig.lrSchedule.setStep(stepSizeEpochs, gamma);
	lrScheduleMultiplier = 1.0f;
}

void glades::NNetwork::setLearningRateScheduleExp(float gamma)
{
	trainingConfig.lrSchedule.setExp(gamma);
	lrScheduleMultiplier = 1.0f;
}

void glades::NNetwork::setLearningRateScheduleCosine(int tMaxEpochs, float minMultiplier)
{
	trainingConfig.lrSchedule.setCosine(tMaxEpochs, minMultiplier);
	lrScheduleMultiplier = 1.0f;
}

void glades::NNetwork::setGlobalGradClipNorm(float clipNorm)
{
	trainingConfig.globalGradClipNorm = clipNorm;
}

void glades::NNetwork::setPerElementGradClip(float clipLimit)
{
	trainingConfig.perElementGradClip = clipLimit;
}

float glades::NNetwork::computeLearningRateMultiplier(int epochFromStart) const
{
	return trainingConfig.lrSchedule.multiplier(epochFromStart);
}

void glades::NNetwork::resetGraphs()
{
	// create the results again
	results.clear();
}

bool glades::NNetwork::getChangeInputLayers() const
{
    return changeInputLayers;
}

void glades::NNetwork::setChangeInputLayers(bool cIL)
{
    changeInputLayers = cIL;
}

bool glades::NNetwork::getMustdBuildMeat() const
{
    return mustBuildMeat;
}

void glades::NNetwork::setMustdBuildMeat(bool newMustdBuildMeat)
{
    mustBuildMeat = newMustdBuildMeat;
}

