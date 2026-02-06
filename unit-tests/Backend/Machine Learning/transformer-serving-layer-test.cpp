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

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/transformer_serving_layer.h"

#include "../../../Backend/Machine Learning/DataObjects/DataInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

// Minimal in-memory token-id dataset for token language model inference tests.
// (Copied from nn-test.cpp to keep this test file self-contained.)
class InMemoryTokenIdInput : public glades::DataInput
{
public:
	InMemoryTokenIdInput()
	    : padTokenId(-1),
	      scratchTok(0.0f),
	      scratchNext(0.0f),
	      one(1, 0.0f),
	      empty()
	{
	}

	void setTrainTokens(const std::vector<unsigned int>& toks, int pad)
	{
		padTokenId = pad;
		trainTok.clear();
		trainNextTok.clear();
		trainTok.reserve(toks.size());
		for (size_t i = 0; i < toks.size(); ++i)
			trainTok.push_back(static_cast<int>(toks[i]));
		build_next(trainTok, padTokenId, trainNextTok);
	}

	void mirrorTrainToTest()
	{
		testTok = trainTok;
		testNextTok = trainNextTok;
	}

	virtual void import(shmea::GString, int = 0) {}
	virtual void import(const shmea::GTable&, int = 0) {}

	virtual shmea::GVector<float> getTrainRow(unsigned int i) const
	{
		if (i >= trainTok.size())
			return empty;
		one[0] = static_cast<float>(trainTok[i]);
		return one;
	}
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int i) const
	{
		if (i >= trainNextTok.size())
			return empty;
		one[0] = static_cast<float>(trainNextTok[i]);
		return one;
	}
	virtual shmea::GVector<float> getTestRow(unsigned int i) const
	{
		if (i >= testTok.size())
			return empty;
		one[0] = static_cast<float>(testTok[i]);
		return one;
	}
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int i) const
	{
		if (i >= testNextTok.size())
			return empty;
		one[0] = static_cast<float>(testNextTok[i]);
		return one;
	}

	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= trainTok.size())
			return false;
		scratchTok = static_cast<float>(trainTok[index]);
		outData = &scratchTok;
		outSize = 1u;
		return true;
	}
	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= trainNextTok.size())
			return false;
		scratchNext = static_cast<float>(trainNextTok[index]);
		outData = &scratchNext;
		outSize = 1u;
		return true;
	}
	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= testTok.size())
			return false;
		scratchTok = static_cast<float>(testTok[index]);
		outData = &scratchTok;
		outSize = 1u;
		return true;
	}
	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL;
		outSize = 0u;
		if (index >= testNextTok.size())
			return false;
		scratchNext = static_cast<float>(testNextTok[index]);
		outData = &scratchNext;
		outSize = 1u;
		return true;
	}

	virtual bool getTrainTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= trainTok.size())
			return false;
		outTokenId = trainTok[index];
		return true;
	}
	virtual bool getTrainExpectedTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= trainNextTok.size())
			return false;
		outTokenId = trainNextTok[index];
		return true;
	}
	virtual bool getTestTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= testTok.size())
			return false;
		outTokenId = testTok[index];
		return true;
	}
	virtual bool getTestExpectedTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= testNextTok.size())
			return false;
		outTokenId = testNextTok[index];
		return true;
	}

	virtual unsigned int getTrainSize() const { return static_cast<unsigned int>(trainTok.size()); }
	virtual unsigned int getTestSize() const { return static_cast<unsigned int>(testTok.size()); }
	virtual unsigned int getFeatureCount() const { return 1u; }
	virtual int getType() const { return TEXT; }

private:
	static void build_next(const std::vector<int>& toks, int pad, std::vector<int>& outNext)
	{
		outNext.clear();
		outNext.reserve(toks.size());
		for (size_t i = 0; i < toks.size(); ++i)
		{
			if (i + 1u < toks.size())
				outNext.push_back(toks[i + 1u]);
			else
				outNext.push_back(pad);
		}
	}

	int padTokenId;
	std::vector<int> trainTok;
	std::vector<int> trainNextTok;
	std::vector<int> testTok;
	std::vector<int> testNextTok;

	mutable float scratchTok;
	mutable float scratchNext;
	mutable shmea::GVector<float> one;
	shmea::GVector<float> empty;
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
		glades::TransformerServingLayer::Config cfg;
		cfg.maxBatchSize = 0u;
		cfg.maxSeqLen = 16u;
		const glades::NNetworkStatus st = layer.start(*m.net, cfg);
		ASSERT("==============ServingLayer: StartRejectsZeroBatch Failed==============", !st.ok());
		ASSERT("==============ServingLayer: NotRunningAfterFailedStart Failed==============", !layer.isRunning());
	}

	// --------
	// Case 2: backpressure on pending queue (no stepping)
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg;
		cfg.maxBatchSize = 2u;
		cfg.maxSeqLen = 16u;
		cfg.maxPendingRequests = 2u;
		cfg.wipeKvOnRemove = false;
		cfg.rngSeed = 123u;
		cfg.autoRemoveFinished = true;
		cfg.enableLogs = false;
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
		glades::TransformerServingLayer::Config cfg;
		cfg.maxBatchSize = 1u;
		cfg.maxSeqLen = 16u;
		cfg.maxPendingRequests = 8u;
		cfg.enableLogs = false;
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
		glades::TransformerServingLayer::Config cfg;
		cfg.maxBatchSize = 1u;
		cfg.maxSeqLen = 16u;
		cfg.enableLogs = false;
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
		glades::TransformerServingLayer::Config cfg;
		cfg.maxBatchSize = 1u;
		cfg.maxSeqLen = 16u;
		cfg.enableLogs = false;
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
		glades::TransformerServingLayer::Config cfg;
		cfg.maxBatchSize = 1u;
		cfg.maxSeqLen = 16u;
		cfg.enableLogs = false;
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
	// Case 7: invalid prompt token id propagates as a per-request failure (submit ok, admit fails)
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg;
		cfg.maxBatchSize = 1u;
		cfg.maxSeqLen = 16u;
		cfg.enableLogs = false;
		ASSERT("==============ServingLayer: StartOK6 Failed==============", layer.start(*m.net, cfg).ok());

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
	// Case 8: clearSnapshot() removes completed snapshot
	// --------
	{
		glades::TransformerServingLayer layer;
		glades::TransformerServingLayer::Config cfg;
		cfg.maxBatchSize = 1u;
		cfg.maxSeqLen = 16u;
		cfg.enableLogs = false;
		ASSERT("==============ServingLayer: StartOK7 Failed==============", layer.start(*m.net, cfg).ok());

		std::vector<unsigned int> prompt;
		prompt.push_back(1u);
		uint64_t id = 0;
		ASSERT("==============ServingLayer: SubmitShort Failed==============", layer.submit(make_req(prompt, 0u, false, 0u, 1u), id).ok());

		// With maxNewTokens=0, it should stop by limit right after prompt completes (no decode).
		for (unsigned int spins = 0u; spins < 8u; ++spins)
		{
			ASSERT("==============ServingLayer: StepShort Failed==============", layer.step().ok());
			glades::TransformerServingLayer::RequestSnapshot snap;
			ASSERT("==============ServingLayer: SnapshotShort Failed==============", layer.getSnapshot(id, snap));
			if (snap.done)
				break;
		}

		glades::TransformerServingLayer::RequestSnapshot snap;
		ASSERT("==============ServingLayer: SnapshotShortFinal Failed==============", layer.getSnapshot(id, snap));
		ASSERT("==============ServingLayer: ShortDone Failed==============", snap.done);
		ASSERT("==============ServingLayer: ClearSnapshotOK Failed==============", layer.clearSnapshot(id));

		glades::TransformerServingLayer::RequestSnapshot snap2;
		ASSERT("==============ServingLayer: SnapshotCleared Missing Failed==============", !layer.getSnapshot(id, snap2));
		layer.stop();
	}

	printf("\n============================================================\n");
}

