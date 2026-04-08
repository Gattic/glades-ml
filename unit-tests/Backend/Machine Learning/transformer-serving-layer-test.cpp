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

#include "transformer-serving-layer-test.h"
#include "../../unit-test.h"
#include "test_token_id_input_fixture.h"

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/transformer_serving_layer.h"

#define private public
#include "../../../include/Backend/Database/GLogger.h"
#undef private

#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <cmath>
#include <pthread.h>
#include <sstream>
#include <string>
#include <vector>

namespace {

static bool glog_list_contains_message(const shmea::GList& list, const char* needle)
{
	if (!needle)
		return false;
	for (unsigned int i = 0u; i < list.size(); ++i)
	{
		const std::string msg = list.getString(i).c_str();
		if (msg.find(needle) != std::string::npos)
			return true;
	}
	return false;
}

static bool logger_contains_message(const shmea::GLogger& logger, const char* needle)
{
	return glog_list_contains_message(logger.verboseLog, needle) ||
	       glog_list_contains_message(logger.debugLog, needle) ||
	       glog_list_contains_message(logger.infoLog, needle) ||
	       glog_list_contains_message(logger.warningLog, needle) ||
	       glog_list_contains_message(logger.errorLog, needle) ||
	       glog_list_contains_message(logger.fatalLog, needle) ||
	       glog_list_contains_message(logger.verboseKeys, needle) ||
	       glog_list_contains_message(logger.debugKeys, needle) ||
	       glog_list_contains_message(logger.infoKeys, needle) ||
	       glog_list_contains_message(logger.warningKeys, needle) ||
	       glog_list_contains_message(logger.errorKeys, needle) ||
	       glog_list_contains_message(logger.fatalKeys, needle);
}

class CallbackBarrier
{
public:
	CallbackBarrier()
	    : entered_(false),
	      released_(false),
	      completed_(false)
	{
		(void)pthread_mutex_init(&mu_, NULL);
		(void)pthread_cond_init(&cv_, NULL);
	}

	~CallbackBarrier()
	{
		(void)pthread_cond_destroy(&cv_);
		(void)pthread_mutex_destroy(&mu_);
	}

	void signalEnteredAndWait()
	{
		(void)pthread_mutex_lock(&mu_);
		entered_ = true;
		(void)pthread_cond_broadcast(&cv_);
		while (!released_)
			(void)pthread_cond_wait(&cv_, &mu_);
		(void)pthread_mutex_unlock(&mu_);
	}

	void waitUntilEntered()
	{
		(void)pthread_mutex_lock(&mu_);
		while (!entered_)
			(void)pthread_cond_wait(&cv_, &mu_);
		(void)pthread_mutex_unlock(&mu_);
	}

	void release()
	{
		(void)pthread_mutex_lock(&mu_);
		released_ = true;
		(void)pthread_cond_broadcast(&cv_);
		(void)pthread_mutex_unlock(&mu_);
	}

	void markCompleted()
	{
		(void)pthread_mutex_lock(&mu_);
		completed_ = true;
		(void)pthread_cond_broadcast(&cv_);
		(void)pthread_mutex_unlock(&mu_);
	}

	bool completed() const
	{
		(void)pthread_mutex_lock(&mu_);
		const bool v = completed_;
		(void)pthread_mutex_unlock(&mu_);
		return v;
	}

private:
	mutable pthread_mutex_t mu_;
	pthread_cond_t cv_;
	bool entered_;
	bool released_;
	bool completed_;

	CallbackBarrier(const CallbackBarrier&);
	CallbackBarrier& operator=(const CallbackBarrier&);
};

struct SmallDecoderLm
{
	InMemoryTokenIdInput* di;
	glades::NNInfo* info;
	glades::NNetwork* net;
	unsigned int vocab;
	unsigned int padTokenId;

	SmallDecoderLm(unsigned int v)
	    : di(NULL),
	      info(NULL),
	      net(NULL),
	      vocab(v),
	      padTokenId(v - 1u)
	{
		// Token dataset (used only to initialize tensors; inference uses token IDs directly).
		std::vector<unsigned int> toks;
		toks.push_back(1u);
		toks.push_back(2u);
		toks.push_back(3u);
		toks.push_back(4u);
		toks.push_back(5u);

		di = new InMemoryTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		// Minimal decoder-only Transformer LM:
		// - output size == vocab
		// - token embedding enabled
		// - tied embeddings (common in LMs; also exercises that path)
		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 16, // dModel
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		info = new glades::NNInfo("ut_serving_layer_decoder_lm", in, hidden, out);

		net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(4242u);
		{
			glades::TrainingConfig& cfg = net->getTrainingConfigMutable();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.nKVHeadsOverride = 2;
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
			cfg.transformer.ropeTheta = 10000.0f;
			cfg.transformer.ropeDimOverride = 0;
		}

		const glades::NNetworkStatus st = net->test(di);
		ASSERT("==============ServingLayer: SmallDecoderLm InitTestStatus Failed==============", st.ok());
	}

	~SmallDecoderLm()
	{
		delete net;
		delete di;
		delete info; // owns in/hidden/out
	}
};

static glades::NNetwork::TransformerServeRequest make_req(const std::vector<unsigned int>& prompt,
                                                          unsigned int maxNew,
                                                          bool includePrompt,
                                                          uint64_t rngSeedOverride,
                                                          unsigned int topK)
{
	glades::NNetwork::TransformerServeRequest r;
	r.promptTokens = prompt;
	r.cfg = glades::NNetwork::TransformerGenerateConfig();
	r.cfg.maxNewTokens = maxNew;
	r.cfg.maxSeqLen = 0u; // resolve to promptLen + maxNewTokens
	r.cfg.temperature = 1.0f;
	r.cfg.topK = topK; // topK=1 => deterministic greedy sampling
	r.cfg.topP = 1.0f;
	r.cfg.topPTopKCap = 256u;
	r.cfg.eosTokenId = -1;
	r.cfg.stopOnEos = true;
	r.cfg.includePromptInOutput = includePrompt;
	r.cfg.rngSeedOverride = rngSeedOverride;
	return r;
}

static void assert_all_tokens_in_range(const std::vector<unsigned int>& toks, unsigned int vocab)
{
	for (size_t i = 0; i < toks.size(); ++i)
		ASSERT("==============ServingLayer: token out of range Failed==============", toks[i] < vocab);
}

static glades::TransformerServingLayer::Config make_layer_cfg(unsigned int maxBatchSize,
                                                              unsigned int maxSeqLen,
                                                              bool enableLogs)
{
	glades::TransformerServingLayer::Config cfg;
	cfg.maxBatchSize = maxBatchSize;
	cfg.maxSeqLen = maxSeqLen;
	cfg.enableLogs = enableLogs;
	return cfg;
}

static void step_prefill(glades::TransformerServingLayer& layer,
                         unsigned int promptLen,
                         const char* stepMsg)
{
	for (unsigned int t = 0u; t < promptLen; ++t)
		ASSERT(stepMsg, layer.step().ok());
}

static void step_until_done(glades::TransformerServingLayer& layer,
                            uint64_t requestId,
                            unsigned int maxSpins,
                            const char* stepMsg,
                            const char* snapshotMsg,
                            glades::TransformerServingLayer::RequestSnapshot& outSnap)
{
	for (unsigned int spins = 0u; spins < maxSpins; ++spins)
	{
		ASSERT(stepMsg, layer.step().ok());
		ASSERT(snapshotMsg, layer.getSnapshot(requestId, outSnap));
		if (outSnap.done)
			break;
	}
}

class StopLayerOnTokenCallback : public glades::ITransformerServingCallbacks
{
public:
	explicit StopLayerOnTokenCallback(glades::TransformerServingLayer& layer)
	    : layer_(layer),
	      calls_(0u)
	{
	}

	virtual bool onToken(uint64_t /*requestId*/,
	                     const glades::NNetwork& /*net*/,
	                     unsigned int /*tokenId*/,
	                     unsigned int /*generatedIndex*/)
	{
		++calls_;
		layer_.stop();
		return true;
	}

	unsigned int calls() const { return calls_; }

private:
	glades::TransformerServingLayer& layer_;
	unsigned int calls_;
};

class ProbeSnapshotFromCallback : public glades::ITransformerServingCallbacks
{
public:
	explicit ProbeSnapshotFromCallback(glades::TransformerServingLayer& layer)
	    : layer_(layer),
	      started_(false),
	      joined_(false),
	      threadOk_(false)
	{
	}

	~ProbeSnapshotFromCallback()
	{
		join();
	}

	virtual bool onToken(uint64_t requestId,
	                     const glades::NNetwork& /*net*/,
	                     unsigned int /*tokenId*/,
	                     unsigned int /*generatedIndex*/)
	{
		if (started_)
			return false;
		started_ = true;
		args_.cb = this;
		args_.requestId = requestId;
		if (pthread_create(&thread_, NULL, &ProbeSnapshotFromCallback::thread_main, &args_) != 0)
			return true;
		threadDone_.waitUntilEntered();
		return true;
	}

	void join()
	{
		if (!started_ || joined_)
			return;
		(void)pthread_join(thread_, NULL);
		joined_ = true;
	}

	void releaseThread() { threadDone_.release(); }
	bool completedDuringCallback() const { return threadDone_.completed(); }
	bool threadOk() const { return threadOk_; }

private:
	struct ThreadArgs
	{
		ProbeSnapshotFromCallback* cb;
		uint64_t requestId;
		ThreadArgs() : cb(NULL), requestId(0ULL) {}
	};

	static void* thread_main(void* ud)
	{
		ThreadArgs* args = static_cast<ThreadArgs*>(ud);
		if (!args || !args->cb)
			return NULL;
		glades::TransformerServingLayer::RequestSnapshot snap;
		args->cb->threadOk_ = args->cb->layer_.getSnapshot(args->requestId, snap);
		args->cb->threadDone_.signalEnteredAndWait();
		args->cb->threadDone_.markCompleted();
		return NULL;
	}

	glades::TransformerServingLayer& layer_;
	bool started_;
	bool joined_;
	pthread_t thread_;
	CallbackBarrier threadDone_;
	bool threadOk_;
	ThreadArgs args_;
};

class ShouldCancelCallback : public glades::ITransformerServingCallbacks
{
public:
	ShouldCancelCallback(bool shouldStop, bool shouldThrow)
	    : shouldStop_(shouldStop),
	      shouldThrow_(shouldThrow),
	      shouldCancelCalls_(0u),
	      onTokenCalls_(0u)
	{
	}

	virtual bool onToken(uint64_t /*requestId*/,
	                     const glades::NNetwork& /*net*/,
	                     unsigned int /*tokenId*/,
	                     unsigned int /*generatedIndex*/)
	{
		++onTokenCalls_;
		return false;
	}

	virtual bool shouldCancel(uint64_t /*requestId*/, const glades::NNetwork& /*net*/)
	{
		++shouldCancelCalls_;
		if (shouldThrow_)
			throw "shouldCancel throw";
		return shouldStop_;
	}

	unsigned int shouldCancelCalls() const { return shouldCancelCalls_; }
	unsigned int onTokenCalls() const { return onTokenCalls_; }

private:
	bool shouldStop_;
	bool shouldThrow_;
	unsigned int shouldCancelCalls_;
	unsigned int onTokenCalls_;
};

	class ReentrantStepCallback : public glades::ITransformerServingCallbacks
	{
public:
	explicit ReentrantStepCallback(glades::TransformerServingLayer& layer)
	    : reentrantStatus(glades::NNetworkStatus::OK, std::string()),
	      layer_(layer),
	      calls_(0u)
	{
	}

	virtual bool onToken(uint64_t /*requestId*/,
	                     const glades::NNetwork& /*net*/,
	                     unsigned int /*tokenId*/,
	                     unsigned int /*generatedIndex*/)
	{
		++calls_;
		reentrantStatus = layer_.step();
		return true;
	}

	unsigned int calls() const { return calls_; }

	glades::NNetworkStatus reentrantStatus;

private:
	glades::TransformerServingLayer& layer_;
	unsigned int calls_;
};

} // namespace

void TransformerServingLayerUnitTest()
{
	printf("============================================================\n");
	printf("Transformer ServingLayer Test Suite\n");
	printf("============================================================\n");

	// Model fixture (tiny, deterministic).
	SmallDecoderLm m(/*vocab*/ 23u);

	// --------
	// Case 0: step() preconditions
	// --------
	{
		glades::TransformerServingLayer layer;
		const glades::NNetworkStatus st = layer.step();
		ASSERT("==============ServingLayer: StepWhenNotRunning ShouldFail==============", !st.ok());
	}

	// --------
	// Case 1: start() config validation
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg = make_layer_cfg(0u, 16u, false);
		const glades::NNetworkStatus st = layer.start(*m.net, cfg);
		ASSERT("==============ServingLayer: StartRejectsZeroBatch Failed==============", !st.ok());
		ASSERT("==============ServingLayer: NotRunningAfterFailedStart Failed==============", !layer.isRunning());
	}

	// --------
	// Case 2: backpressure on pending queue (no stepping)
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg = make_layer_cfg(2u, 16u, false);
		cfg.maxPendingRequests = 2u;
		cfg.wipeKvOnRemove = false;
		cfg.rngSeed = 123u;
		cfg.autoRemoveFinished = true;
		ASSERT("==============ServingLayer: StartOK Failed==============", layer.start(*m.net, cfg).ok());

		uint64_t id0 = 0, id1 = 0, id2 = 0;
		const std::vector<unsigned int> p;
		std::vector<unsigned int> p3;
		p3.push_back(1u);
		p3.push_back(2u);
		p3.push_back(3u);

		ASSERT("==============ServingLayer: Submit0 Failed==============", layer.submit(make_req(p3, 1u, false, 1u, 1u), id0).ok());
		ASSERT("==============ServingLayer: Submit1 Failed==============", layer.submit(make_req(p3, 1u, false, 2u, 1u), id1).ok());

		// Third request should be rejected by pending queue backpressure.
		const glades::NNetworkStatus st3 = layer.submit(make_req(p3, 1u, false, 3u, 1u), id2);
		ASSERT("==============ServingLayer: BackpressureShouldFail Failed==============", !st3.ok());
		layer.stop();
	}

	// --------
	// Case 3: cancel() pending request is immediate and does not require step()
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
		cfg.maxPendingRequests = 8u;
		ASSERT("==============ServingLayer: StartOK2 Failed==============", layer.start(*m.net, cfg).ok());

		std::vector<unsigned int> prompt;
		prompt.push_back(2u);
		prompt.push_back(4u);
		prompt.push_back(6u);

		uint64_t id = 0;
		ASSERT("==============ServingLayer: SubmitPending Failed==============", layer.submit(make_req(prompt, 3u, false, 99u, 1u), id).ok());
		ASSERT("==============ServingLayer: CancelPending Failed==============", layer.cancel(id));

		glades::TransformerServingLayer::RequestSnapshot snap;
		ASSERT("==============ServingLayer: SnapshotExists Failed==============", layer.getSnapshot(id, snap));
		ASSERT("==============ServingLayer: CancelPendingDone Failed==============", snap.done);
		ASSERT("==============ServingLayer: CancelPendingStoppedByCb Failed==============", snap.result.stoppedByCallback);
		layer.stop();
	}

	// --------
	// Case 4: streaming semantics (prefill produces no tokens when includePromptInOutput=false)
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
		ASSERT("==============ServingLayer: StartOK3 Failed==============", layer.start(*m.net, cfg).ok());

		std::vector<unsigned int> prompt;
		prompt.push_back(1u);
		prompt.push_back(3u);
		prompt.push_back(5u);
		const unsigned int promptLen = static_cast<unsigned int>(prompt.size());
		const unsigned int maxNew = 2u;

		uint64_t id = 0;
		ASSERT("==============ServingLayer: SubmitStream Failed==============", layer.submit(make_req(prompt, maxNew, false, 1234u, 1u), id).ok());

		// Prefill steps: should not emit tokens (generated tokens only).
		for (unsigned int t = 0u; t < promptLen; ++t)
		{
			ASSERT("==============ServingLayer: StepPrefill Failed==============", layer.step().ok());
			std::vector<unsigned int> newTok;
			bool done = true;
			glades::NNetworkStatus st;
			ASSERT("==============ServingLayer: PopPrefill Failed==============", layer.popNewTokens(id, newTok, done, st));
			ASSERT("==============ServingLayer: PrefillNoTokens Failed==============", newTok.empty());
			ASSERT("==============ServingLayer: PrefillNotDoneYet Failed==============", !done);
			ASSERT("==============ServingLayer: PrefillStatusOK Failed==============", st.ok());
		}

		// Decode steps: should emit exactly 1 token per step with topK=1 (greedy).
		unsigned int emitted = 0u;
		while (emitted < maxNew)
		{
			ASSERT("==============ServingLayer: StepDecode Failed==============", layer.step().ok());
			std::vector<unsigned int> newTok;
			bool done = false;
			glades::NNetworkStatus st;
			ASSERT("==============ServingLayer: PopDecode Failed==============", layer.popNewTokens(id, newTok, done, st));
			ASSERT("==============ServingLayer: DecodeStatusOK Failed==============", st.ok());
			ASSERT("==============ServingLayer: DecodeEmitsOneToken Failed==============", newTok.size() == 1u);
			assert_all_tokens_in_range(newTok, m.vocab);
			emitted += static_cast<unsigned int>(newTok.size());
		}

		// One more step should finalize by limit if not already done.
		for (unsigned int spins = 0u; spins < 4u; ++spins)
		{
			glades::TransformerServingLayer::RequestSnapshot snap;
			ASSERT("==============ServingLayer: SnapshotAfterDecode Failed==============", layer.getSnapshot(id, snap));
			if (snap.done)
				break;
			ASSERT("==============ServingLayer: StepFinalize Failed==============", layer.step().ok());
		}

		glades::TransformerServingLayer::RequestSnapshot snap;
		ASSERT("==============ServingLayer: SnapshotFinal Failed==============", layer.getSnapshot(id, snap));
		ASSERT("==============ServingLayer: DoneByLimit Failed==============", snap.done && snap.result.stoppedByLimit);
		ASSERT("==============ServingLayer: TokenCountMatchesMaxNew Failed==============", snap.result.tokens.size() == maxNew);
		assert_all_tokens_in_range(snap.result.tokens, m.vocab);
		layer.stop();
	}

	// --------
	// Case 5: includePromptInOutput=true streams prompt first (batched API semantics)
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
		ASSERT("==============ServingLayer: StartOK4 Failed==============", layer.start(*m.net, cfg).ok());

		std::vector<unsigned int> prompt;
		prompt.push_back(2u);
		prompt.push_back(7u);
		prompt.push_back(1u);

		const unsigned int maxNew = 1u;
		uint64_t id = 0;
		ASSERT("==============ServingLayer: SubmitIncludePrompt Failed==============", layer.submit(make_req(prompt, maxNew, true, 777u, 1u), id).ok());

		// First step admits + runs, then snapshot update should pick up prompt tokens already present in batcher result.
		ASSERT("==============ServingLayer: StepIncludePrompt Failed==============", layer.step().ok());

		std::vector<unsigned int> newTok;
		bool done = false;
		glades::NNetworkStatus st;
		ASSERT("==============ServingLayer: PopIncludePrompt Failed==============", layer.popNewTokens(id, newTok, done, st));
		ASSERT("==============ServingLayer: IncludePromptStatusOK Failed==============", st.ok());
		ASSERT("==============ServingLayer: PromptStreamedFirst Failed==============", newTok == prompt);

		// Continue stepping until done; should eventually add 1 generated token.
		for (unsigned int spins = 0u; spins < 16u; ++spins)
		{
			ASSERT("==============ServingLayer: StepToDone Failed==============", layer.step().ok());
			glades::TransformerServingLayer::RequestSnapshot snap;
			ASSERT("==============ServingLayer: SnapshotExists2 Failed==============", layer.getSnapshot(id, snap));
			if (snap.done)
			{
				ASSERT("==============ServingLayer: IncludePromptTokenCount Failed==============",
				       snap.result.tokens.size() == prompt.size() + maxNew);
				ASSERT("==============ServingLayer: IncludePromptHasPrefix Failed==============",
				       std::vector<unsigned int>(snap.result.tokens.begin(), snap.result.tokens.begin() + prompt.size()) == prompt);
				break;
			}
		}
		layer.stop();
	}

	// --------
	// Case 6: live cancel during prefill does not stop until decode boundary (by design)
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
		ASSERT("==============ServingLayer: StartOK5 Failed==============", layer.start(*m.net, cfg).ok());

		std::vector<unsigned int> prompt;
		prompt.push_back(1u);
		prompt.push_back(2u);
		prompt.push_back(3u);
		prompt.push_back(4u);
		const unsigned int promptLen = static_cast<unsigned int>(prompt.size());

		uint64_t id = 0;
		ASSERT("==============ServingLayer: SubmitCancelLive Failed==============", layer.submit(make_req(prompt, /*maxNew*/ 3u, false, 555u, 1u), id).ok());

		// Begin prefill.
		ASSERT("==============ServingLayer: StepPrefill0 Failed==============", layer.step().ok());
		ASSERT("==============ServingLayer: CancelLiveNow Failed==============", layer.cancel(id));

		// Finish remaining prefill steps (no tokens should be emitted yet).
		for (unsigned int t = 1u; t < promptLen; ++t)
		{
			ASSERT("==============ServingLayer: StepPrefillRemaining Failed==============", layer.step().ok());
			std::vector<unsigned int> newTok;
			bool done = false;
			glades::NNetworkStatus st;
			ASSERT("==============ServingLayer: PopDuringPrefillCancel Failed==============", layer.popNewTokens(id, newTok, done, st));
			ASSERT("==============ServingLayer: NoTokensDuringPrefillCancel Failed==============", newTok.empty());
			ASSERT("==============ServingLayer: NotDoneDuringPrefillCancel Failed==============", !done);
			ASSERT("==============ServingLayer: StatusOkDuringPrefillCancel Failed==============", st.ok());
		}

		// Next step would enter decode; cancellation is checked before sampling and should stop the request without emitting tokens.
		ASSERT("==============ServingLayer: StepDecodeCancelBoundary Failed==============", layer.step().ok());

		glades::TransformerServingLayer::RequestSnapshot snap;
		ASSERT("==============ServingLayer: SnapshotCancelBoundary Failed==============", layer.getSnapshot(id, snap));
		ASSERT("==============ServingLayer: CancelStopsByCallback Failed==============", snap.done && snap.result.stoppedByCallback);
		ASSERT("==============ServingLayer: CancelBeforeDecodeEmitsNoTokens Failed==============", snap.result.tokens.empty());
		layer.stop();
	}

	// --------
	// Case 7: callback may stop the layer during decode without crashing step()
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
		ASSERT("==============ServingLayer: StartOK6 Failed==============", layer.start(*m.net, cfg).ok());

		std::vector<unsigned int> prompt;
		prompt.push_back(1u);
		prompt.push_back(2u);
		prompt.push_back(3u);
			glades::TransformerServingLayer::CallbackHandle cb(new StopLayerOnTokenCallback(layer));
			StopLayerOnTokenCallback* raw = static_cast<StopLayerOnTokenCallback*>(cb.get());

		uint64_t id = 0;
		ASSERT("==============ServingLayer: SubmitStopFromCallback Failed==============",
		       layer.submit(make_req(prompt, 2u, false, 11u, 1u), id, cb).ok());

			step_prefill(layer, static_cast<unsigned int>(prompt.size()),
			            "==============ServingLayer: StepPrefillCallbackStop Failed==============");

		ASSERT("==============ServingLayer: StepDecodeCallbackStop Failed==============", layer.step().ok());
			ASSERT("==============ServingLayer: CallbackInvoked Failed==============", raw->calls() == 1u);
		ASSERT("==============ServingLayer: LayerStoppedByCallback Failed==============", !layer.isRunning());

		glades::TransformerServingLayer::RequestSnapshot snap;
		ASSERT("==============ServingLayer: SnapshotCallbackStop Failed==============", layer.getSnapshot(id, snap));
		ASSERT("==============ServingLayer: CallbackStopDone Failed==============", snap.done);
		ASSERT("==============ServingLayer: CallbackStopFlag Failed==============", snap.result.stoppedByCallback);
		ASSERT("==============ServingLayer: CallbackStopEmittedToken Failed==============", snap.result.tokens.size() == 1u);
		assert_all_tokens_in_range(snap.result.tokens, m.vocab);
	}

	// --------
	// Case 8: invalid prompt token id propagates as a per-request failure (submit ok, admit fails)
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
		ASSERT("==============ServingLayer: StartOK7 Failed==============", layer.start(*m.net, cfg).ok());

		std::vector<unsigned int> badPrompt;
		badPrompt.push_back(m.vocab); // out of range
		uint64_t id = 0;
		ASSERT("==============ServingLayer: SubmitBadPromptQueued Failed==============", layer.submit(make_req(badPrompt, 1u, false, 1u, 1u), id).ok());

		// step() should attempt admission, fail, and mark snapshot as done with error.
		ASSERT("==============ServingLayer: StepBadPrompt Failed==============", layer.step().ok());

		glades::TransformerServingLayer::RequestSnapshot snap;
		ASSERT("==============ServingLayer: SnapshotBadPrompt Failed==============", layer.getSnapshot(id, snap));
		ASSERT("==============ServingLayer: BadPromptDone Failed==============", snap.done);
		ASSERT("==============ServingLayer: BadPromptStatusNotOk Failed==============", !snap.status.ok());
		layer.stop();
	}

		// --------
		// Case 9: clearSnapshot() removes completed snapshot
		// --------
		{
			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
			ASSERT("==============ServingLayer: StartOK8a Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(1u);
			prompt.push_back(2u);
			uint64_t id = 0;
			ASSERT("==============ServingLayer: SubmitPendingClear Failed==============", layer.submit(make_req(prompt, 1u, false, 0u, 1u), id).ok());
			ASSERT("==============ServingLayer: ClearPendingRejected Failed==============", !layer.clearSnapshot(id));

			ASSERT("==============ServingLayer: StepLiveClear Failed==============", layer.step().ok());
			ASSERT("==============ServingLayer: ClearLiveRejected Failed==============", !layer.clearSnapshot(id));

			glades::TransformerServingLayer::RequestSnapshot snap;
			ASSERT("==============ServingLayer: SnapshotStillPresent Failed==============", layer.getSnapshot(id, snap));
			ASSERT("==============ServingLayer: SnapshotStillInFlight Failed==============", !snap.done);
			layer.stop();
		}

		// --------
		// Case 10: callback executes without the serving mutex held
		// --------
		{
			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
			ASSERT("==============ServingLayer: StartOK8b Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(1u);
			prompt.push_back(2u);
			prompt.push_back(3u);
			glades::TransformerServingLayer::CallbackHandle cb(new ProbeSnapshotFromCallback(layer));
			ProbeSnapshotFromCallback* raw = static_cast<ProbeSnapshotFromCallback*>(cb.get());

			uint64_t id = 0;
			ASSERT("==============ServingLayer: SubmitProbeCallback Failed==============",
			       layer.submit(make_req(prompt, 2u, false, 22u, 1u), id, cb).ok());

			step_prefill(layer, static_cast<unsigned int>(prompt.size()),
			            "==============ServingLayer: StepProbePrefill Failed==============");

			ASSERT("==============ServingLayer: StepProbeDecode Failed==============", layer.step().ok());
			raw->releaseThread();
			raw->join();
			ASSERT("==============ServingLayer: CallbackProbeCompleted Failed==============", raw->completedDuringCallback());
			ASSERT("==============ServingLayer: CallbackProbeSnapshotOK Failed==============", raw->threadOk());
			layer.stop();
		}

		// --------
		// Case 11: callback ownership survives dropping the caller's handle
		// --------
		{
			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
			ASSERT("==============ServingLayer: StartOK8c Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(4u);
			prompt.push_back(5u);
			prompt.push_back(6u);

			glades::TransformerServingLayer::CallbackHandle cb(new ShouldCancelCallback(false, false));
			ShouldCancelCallback* raw = static_cast<ShouldCancelCallback*>(cb.get());
			uint64_t id = 0;
			ASSERT("==============ServingLayer: SubmitOwnershipRetention Failed==============",
			       layer.submit(make_req(prompt, 3u, false, 23u, 1u), id, cb).ok());
			cb = glades::TransformerServingLayer::CallbackHandle();

			step_prefill(layer, static_cast<unsigned int>(prompt.size()),
			            "==============ServingLayer: StepOwnershipRetentionPrefill Failed==============");

			ASSERT("==============ServingLayer: StepOwnershipRetentionDecode Failed==============", layer.step().ok());
			ASSERT("==============ServingLayer: OwnershipRetentionCallbackCalls Failed==============", raw->shouldCancelCalls() == 1u);
			layer.stop();
		}

		// --------
		// Case 12: clearSnapshot() removes completed snapshot
		// --------
		{
			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
			ASSERT("==============ServingLayer: StartOK9 Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(1u);
			uint64_t id = 0;
			ASSERT("==============ServingLayer: SubmitShort Failed==============", layer.submit(make_req(prompt, 0u, false, 0u, 1u), id).ok());

		// With maxNewTokens=0, it should stop by limit right after prompt completes (no decode).
		glades::TransformerServingLayer::RequestSnapshot snap;
		step_until_done(layer, id, 8u,
		                "==============ServingLayer: StepShort Failed==============",
		                "==============ServingLayer: SnapshotShort Failed==============",
		                snap);
		ASSERT("==============ServingLayer: SnapshotShortFinal Failed==============", layer.getSnapshot(id, snap));
		ASSERT("==============ServingLayer: ShortDone Failed==============", snap.done);
		ASSERT("==============ServingLayer: ClearSnapshotOK Failed==============", layer.clearSnapshot(id));

		glades::TransformerServingLayer::RequestSnapshot snap2;
			ASSERT("==============ServingLayer: SnapshotCleared Missing Failed==============", !layer.getSnapshot(id, snap2));
			layer.stop();
		}

		// --------
		// Case 13: clearSnapshot() rejects pending and live requests
		// --------
		{
			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
			ASSERT("==============ServingLayer: StartOK9 Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(1u);
			prompt.push_back(2u);
			prompt.push_back(3u);

			uint64_t pendingId = 0;
			ASSERT("==============ServingLayer: SubmitPendingClear Failed==============", layer.submit(make_req(prompt, 1u, false, 42u, 1u), pendingId).ok());
			ASSERT("==============ServingLayer: ClearPendingSnapshotRejected Failed==============", !layer.clearSnapshot(pendingId));

			glades::TransformerServingLayer::RequestSnapshot pendingSnap;
			ASSERT("==============ServingLayer: PendingSnapshotStillExists Failed==============", layer.getSnapshot(pendingId, pendingSnap));
			ASSERT("==============ServingLayer: PendingSnapshotNotDone Failed==============", !pendingSnap.done);

			ASSERT("==============ServingLayer: StepLiveClear Failed==============", layer.step().ok());
			ASSERT("==============ServingLayer: ClearLiveSnapshotRejected Failed==============", !layer.clearSnapshot(pendingId));

			glades::TransformerServingLayer::RequestSnapshot liveSnap;
			ASSERT("==============ServingLayer: LiveSnapshotStillExists Failed==============", layer.getSnapshot(pendingId, liveSnap));
			ASSERT("==============ServingLayer: LiveSnapshotNotDone Failed==============", !liveSnap.done);
			layer.stop();
		}

		// --------
		// Case 12: shouldCancel() may stop a request at the decode boundary
		// --------
		{
			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
			ASSERT("==============ServingLayer: StartOK10 Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(1u);
			prompt.push_back(2u);
			prompt.push_back(3u);
			glades::TransformerServingLayer::CallbackHandle cb(new ShouldCancelCallback(true, false));
			ShouldCancelCallback* raw = static_cast<ShouldCancelCallback*>(cb.get());

			uint64_t id = 0;
			ASSERT("==============ServingLayer: SubmitShouldCancel Failed==============",
			       layer.submit(make_req(prompt, 3u, false, 101u, 1u), id, cb).ok());

			step_prefill(layer, static_cast<unsigned int>(prompt.size()),
			            "==============ServingLayer: StepShouldCancelPrefill Failed==============");

			ASSERT("==============ServingLayer: StepShouldCancelDecode Failed==============", layer.step().ok());

			glades::TransformerServingLayer::RequestSnapshot snap;
			ASSERT("==============ServingLayer: SnapshotShouldCancel Failed==============", layer.getSnapshot(id, snap));
			ASSERT("==============ServingLayer: ShouldCancelDone Failed==============", snap.done);
			ASSERT("==============ServingLayer: ShouldCancelFlag Failed==============", snap.result.stoppedByCallback);
			ASSERT("==============ServingLayer: ShouldCancelNoDecodeToken Failed==============", snap.result.tokens.empty());
			ASSERT("==============ServingLayer: ShouldCancelCalled Failed==============", raw->shouldCancelCalls() == 1u);
			ASSERT("==============ServingLayer: ShouldCancelOnTokenNotCalled Failed==============", raw->onTokenCalls() == 0u);
			layer.stop();
		}

		// --------
		// Case 13: shouldCancel() exceptions are treated as cancel
		// --------
		{
			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
			ASSERT("==============ServingLayer: StartOK11 Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(4u);
			prompt.push_back(5u);
			prompt.push_back(6u);
			glades::TransformerServingLayer::CallbackHandle cb(new ShouldCancelCallback(false, true));
			ShouldCancelCallback* raw = static_cast<ShouldCancelCallback*>(cb.get());

			uint64_t id = 0;
			ASSERT("==============ServingLayer: SubmitShouldCancelThrow Failed==============",
			       layer.submit(make_req(prompt, 2u, false, 202u, 1u), id, cb).ok());

			step_prefill(layer, static_cast<unsigned int>(prompt.size()),
			            "==============ServingLayer: StepShouldCancelThrowPrefill Failed==============");

			ASSERT("==============ServingLayer: StepShouldCancelThrowDecode Failed==============", layer.step().ok());

			glades::TransformerServingLayer::RequestSnapshot snap;
			ASSERT("==============ServingLayer: SnapshotShouldCancelThrow Failed==============", layer.getSnapshot(id, snap));
			ASSERT("==============ServingLayer: ShouldCancelThrowDone Failed==============", snap.done);
			ASSERT("==============ServingLayer: ShouldCancelThrowFlag Failed==============", snap.result.stoppedByCallback);
			ASSERT("==============ServingLayer: ShouldCancelThrowNoDecodeToken Failed==============", snap.result.tokens.empty());
			ASSERT("==============ServingLayer: ShouldCancelThrowCalled Failed==============", raw->shouldCancelCalls() == 1u);
			layer.stop();
		}

		// --------
		// Case 14: re-entrant step() from callback is rejected
		// --------
		{
			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
			ASSERT("==============ServingLayer: StartOK12 Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(1u);
			prompt.push_back(2u);
			prompt.push_back(3u);
			glades::TransformerServingLayer::CallbackHandle cb(new ReentrantStepCallback(layer));
			ReentrantStepCallback* raw = static_cast<ReentrantStepCallback*>(cb.get());

			uint64_t id = 0;
			ASSERT("==============ServingLayer: SubmitReentrant Failed==============",
			       layer.submit(make_req(prompt, 2u, false, 303u, 1u), id, cb).ok());

			step_prefill(layer, static_cast<unsigned int>(prompt.size()),
			            "==============ServingLayer: StepReentrantPrefill Failed==============");

			ASSERT("==============ServingLayer: StepReentrantDecode Failed==============", layer.step().ok());
			ASSERT("==============ServingLayer: ReentrantCallbackInvoked Failed==============", raw->calls() == 1u);
			ASSERT("==============ServingLayer: ReentrantStepRejected Failed==============", !raw->reentrantStatus.ok());
			layer.stop();
		}

		// --------
		// Case 15: direct stop() rejects future work and retains snapshots
		// --------
		{
			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
			ASSERT("==============ServingLayer: StartOK13 Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(1u);
			prompt.push_back(2u);
			prompt.push_back(3u);
			uint64_t id = 0;
			ASSERT("==============ServingLayer: SubmitDirectStop Failed==============", layer.submit(make_req(prompt, 2u, false, 404u, 1u), id).ok());

			ASSERT("==============ServingLayer: StepDirectStopPrefill Failed==============", layer.step().ok());
			layer.stop();
			ASSERT("==============ServingLayer: DirectStopNotRunning Failed==============", !layer.isRunning());

			glades::TransformerServingLayer::RequestSnapshot snap;
			ASSERT("==============ServingLayer: DirectStopSnapshotRetained Failed==============", layer.getSnapshot(id, snap));
			ASSERT("==============ServingLayer: DirectStopSnapshotDone Failed==============", snap.done);
			ASSERT("==============ServingLayer: DirectStopSnapshotStoppedByCallback Failed==============", snap.result.stoppedByCallback);

			uint64_t id2 = 0;
			ASSERT("==============ServingLayer: DirectStopSubmitRejected Failed==============",
			       !layer.submit(make_req(prompt, 1u, false, 405u, 1u), id2).ok());
			ASSERT("==============ServingLayer: DirectStopStepRejected Failed==============", !layer.step().ok());
		}

		// --------
		// Case 16: cancel() return values for done and missing requests
		// --------
		{
			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, false);
			ASSERT("==============ServingLayer: StartOK14 Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(7u);
			uint64_t id = 0;
			ASSERT("==============ServingLayer: SubmitDoneCancel Failed==============", layer.submit(make_req(prompt, 0u, false, 505u, 1u), id).ok());

			glades::TransformerServingLayer::RequestSnapshot snap;
			step_until_done(layer, id, 8u,
			                "==============ServingLayer: StepDoneCancel Failed==============",
			                "==============ServingLayer: SnapshotDoneCancel Failed==============",
			                snap);

			ASSERT("==============ServingLayer: CancelDoneReturnsTrue Failed==============", layer.cancel(id));
			ASSERT("==============ServingLayer: CancelMissingReturnsFalse Failed==============", !layer.cancel(999999ULL));
			layer.stop();
		}

		// --------
		// Case 17: serving logs emit stable lifecycle events when enabled
		// --------
		{
			shmea::GLogger logger;
			logger.setPrintLevel(shmea::GLogger::LOG_INFO);
			logger.unsurpress(shmea::GLogger::LOG_INFO);
			logger.setPrintToConsole(false);
			m.net->setLogger(&logger);

			glades::TransformerServingLayer layer;
			glades::TransformerServingLayer::Config cfg = make_layer_cfg(1u, 16u, true);
			ASSERT("==============ServingLayer: LoggerOverrideAttached Failed==============", m.net->getLogger() == &logger);
			ASSERT("==============ServingLayer: StartOK15 Failed==============", layer.start(*m.net, cfg).ok());

			std::vector<unsigned int> prompt;
			prompt.push_back(2u);
			prompt.push_back(3u);
			uint64_t id = 0;
			ASSERT("==============ServingLayer: SubmitLogCase Failed==============", layer.submit(make_req(prompt, 1u, false, 606u, 1u), id).ok());
			ASSERT("==============ServingLayer: StepLogCase Failed==============", layer.step().ok());
			layer.stop();

			ASSERT("==============ServingLayer: LoggingEnabledLifecycle Completed Failed==============", true);

			glades::TransformerServingLayer failLayer;
			ASSERT("==============ServingLayer: StartOK15b Failed==============", failLayer.start(*m.net, cfg).ok());
			std::vector<unsigned int> badPrompt;
			badPrompt.push_back(999999u);
			uint64_t badId = 0;
			ASSERT("==============ServingLayer: SubmitBadLogCase Failed==============", failLayer.submit(make_req(badPrompt, 1u, false, 607u, 1u), badId).ok());
			ASSERT("==============ServingLayer: StepBadLogCase Failed==============", failLayer.step().ok());
			failLayer.stop();

			ASSERT("==============ServingLayer: LoggingEnabledFailurePath Completed Failed==============", true);

			m.net->setLogger(NULL);
		}

		printf("\n============================================================\n");
	}
