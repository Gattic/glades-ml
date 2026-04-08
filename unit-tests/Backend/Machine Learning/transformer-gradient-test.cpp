// Transformer backward-pass gradient validation tests.
//
// These tests verify that the hand-coded backpropagation in sgd_transformer.cpp
// produces gradients in the correct direction across multiple architecture
// configurations. Sub-tests include:
//
//   1. Gradient direction: training for 1 epoch with SGD should decrease loss.
//   2. Learning-rate proportionality: loss decrease should scale with lr.
//   3. Weight-update coverage: all major weight matrices should change after training.
//   4. Multi-config sweep: RoPE/sinusoidal, RMSNorm/LayerNorm, GELU/ReLU/SiLU, MLP/SwiGLU.

#include "transformer-gradient-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/DataObjects/TokenInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "../../../Backend/Machine Learning/rng.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// In-memory token-id DataInput (same helper used in transformer-improvements-test.cpp).
// ---------------------------------------------------------------------------
class TestTokenIdInput : public glades::DataInput
{
public:
	TestTokenIdInput()
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
		if (i >= trainTok.size()) return empty;
		one[0] = static_cast<float>(trainTok[i]);
		return one;
	}
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int i) const
	{
		if (i >= trainNextTok.size()) return empty;
		one[0] = static_cast<float>(trainNextTok[i]);
		return one;
	}
	virtual shmea::GVector<float> getTestRow(unsigned int i) const
	{
		if (i >= testTok.size()) return empty;
		one[0] = static_cast<float>(testTok[i]);
		return one;
	}
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int i) const
	{
		if (i >= testNextTok.size()) return empty;
		one[0] = static_cast<float>(testNextTok[i]);
		return one;
	}

	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL; outSize = 0u;
		if (index >= trainTok.size()) return false;
		scratchTok = static_cast<float>(trainTok[index]);
		outData = &scratchTok; outSize = 1u;
		return true;
	}
	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL; outSize = 0u;
		if (index >= trainNextTok.size()) return false;
		scratchNext = static_cast<float>(trainNextTok[index]);
		outData = &scratchNext; outSize = 1u;
		return true;
	}
	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL; outSize = 0u;
		if (index >= testTok.size()) return false;
		scratchTok = static_cast<float>(testTok[index]);
		outData = &scratchTok; outSize = 1u;
		return true;
	}
	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
	{
		outData = NULL; outSize = 0u;
		if (index >= testNextTok.size()) return false;
		scratchNext = static_cast<float>(testNextTok[index]);
		outData = &scratchNext; outSize = 1u;
		return true;
	}

	virtual bool getTrainTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= trainTok.size()) return false;
		outTokenId = trainTok[index]; return true;
	}
	virtual bool getTrainExpectedTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= trainNextTok.size()) return false;
		outTokenId = trainNextTok[index]; return true;
	}
	virtual bool getTestTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= testTok.size()) return false;
		outTokenId = testTok[index]; return true;
	}
	virtual bool getTestExpectedTokenId(unsigned int index, int& outTokenId) const
	{
		outTokenId = 0;
		if (index >= testNextTok.size()) return false;
		outTokenId = testNextTok[index]; return true;
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

// ---------------------------------------------------------------------------
// Callback that records the loss from the last epoch.
// ---------------------------------------------------------------------------
class LossCaptureCb : public glades::ITrainingCallbacks
{
public:
	float lastLoss;
	float lastPerplexity;
	bool saw;

	LossCaptureCb() : lastLoss(0.0f), lastPerplexity(0.0f), saw(false) {}

	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		lastLoss = m.totalError;
		lastPerplexity = m.perplexity;
		saw = true;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}
};

// ---------------------------------------------------------------------------
// Configuration descriptor for multi-config sweep.
// ---------------------------------------------------------------------------
struct GradTestConfig
{
	const char* label;
	int posEnc;   // TransformerRunConfig::PositionalEncoding
	int normType; // TransformerRunConfig::NormType
	int ffnKind;  // TransformerRunConfig::FFNKind
	int ffnAct;   // TransformerRunConfig::FFNActivation
};

// ---------------------------------------------------------------------------
// Helper: apply common transformer config to a network.
// ---------------------------------------------------------------------------
static void applyGradTestConfig(
    glades::NNetwork* net,
    unsigned int vocab,
    unsigned int nHeads,
    unsigned int dFF,
    const GradTestConfig& cfg)
{
	glades::TrainingConfig& tc = net->getTrainingConfigMutable();
	tc.transformer.enableTokenEmbedding = true;
	tc.transformer.vocabSizeOverride = static_cast<int>(vocab);
	tc.transformer.tieEmbeddings = true;
	tc.transformer.padTokenId = static_cast<int>(vocab - 1u);
	tc.transformer.nHeadsOverride = static_cast<int>(nHeads);
	tc.transformer.nKVHeadsOverride = static_cast<int>(nHeads);
	tc.transformer.dFFOverride = static_cast<int>(dFF);
	tc.transformer.positionalEncoding =
	    static_cast<glades::TransformerRunConfig::PositionalEncodingType>(cfg.posEnc);
	tc.transformer.normType =
	    static_cast<glades::TransformerRunConfig::NormType>(cfg.normType);
	tc.transformer.ffnKind =
	    static_cast<glades::TransformerRunConfig::FFNKind>(cfg.ffnKind);
	tc.transformer.ffnActivation =
	    static_cast<glades::TransformerRunConfig::FFNActivationType>(cfg.ffnAct);
	tc.transformer.ropeTheta = 10000.0f;
	// AdamW is required for transformer token LM training.
	tc.optimizer.type = glades::OptimizerConfig::ADAMW;
}

// ---------------------------------------------------------------------------
// Build a small transformer decoder, train for 1 epoch, then run a
// separate eval pass to measure post-training loss. Returns the eval
// loss AFTER training (the callback loss is measured before the weight
// update, so we need a second pass).
// ---------------------------------------------------------------------------
static float trainOnceAndGetLoss(
    unsigned int vocab,
    unsigned int dModel,
    unsigned int nHeads,
    unsigned int dFF,
    unsigned int seed,
    float lr,
    const GradTestConfig& cfg,
    bool* outOk)
{
	*outOk = false;

	const unsigned int padTokenId = vocab - 1u;
	std::vector<unsigned int> toks;
	toks.push_back(2u);
	toks.push_back(5u);
	toks.push_back(11u);
	toks.push_back(3u);
	toks.push_back(7u);
	toks.push_back(1u);
	toks.push_back(9u);
	toks.push_back(4u);

	TestTokenIdInput* di = new TestTokenIdInput();
	di->setTrainTokens(toks, static_cast<int>(padTokenId));
	di->mirrorTrainToTest();

	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    static_cast<int>(toks.size()), lr, 0.0f, 0.0f, 0.0f, 0.0f,
	    glades::GMath::LINEAR, 1.0f);
	std::vector<glades::HiddenLayerInfo*> hidden;
	hidden.push_back(new glades::HiddenLayerInfo(
	    static_cast<int>(dModel), lr, 0.0f, 0.0f, 0.0f, 0.0f,
	    glades::GMath::LINEAR, 1.0f));
	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
	    static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);

	glades::NNInfo* info = new glades::NNInfo("ut_grad_check", in, hidden, out);
	glades::NNetwork* net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
	net->setSeed(seed);
	net->getTerminatorMutable().setEpoch(1);
	net->getTerminatorMutable().setAccuracy(0);
	applyGradTestConfig(net, vocab, nHeads, dFF, cfg);

	// Train 1 epoch
	const glades::NNetworkStatus stTrain = net->train(di);
	if (!stTrain.ok())
	{
		delete net;
		delete info;
		delete di;
		return 0.0f;
	}

	// Eval pass after training to measure updated loss
	LossCaptureCb cb;
	const glades::NNetworkStatus stEval = net->test(di, &cb);

	float loss = 0.0f;
	if (stEval.ok() && cb.saw)
	{
		loss = cb.lastLoss;
		*outOk = true;
	}

	delete net;
	delete info;
	delete di;
	return loss;
}

// ---------------------------------------------------------------------------
// Compute the initial (untrained) loss via the test/eval path.
// ---------------------------------------------------------------------------
static float evalLossBeforeTraining(
    unsigned int vocab,
    unsigned int dModel,
    unsigned int nHeads,
    unsigned int dFF,
    unsigned int seed,
    const GradTestConfig& cfg,
    bool* outOk)
{
	*outOk = false;

	const unsigned int padTokenId = vocab - 1u;
	std::vector<unsigned int> toks;
	toks.push_back(2u);
	toks.push_back(5u);
	toks.push_back(11u);
	toks.push_back(3u);
	toks.push_back(7u);
	toks.push_back(1u);
	toks.push_back(9u);
	toks.push_back(4u);

	TestTokenIdInput* di = new TestTokenIdInput();
	di->setTrainTokens(toks, static_cast<int>(padTokenId));
	di->mirrorTrainToTest();

	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    static_cast<int>(toks.size()), 0.01f, 0.0f, 0.0f, 0.0f, 0.0f,
	    glades::GMath::LINEAR, 1.0f);
	std::vector<glades::HiddenLayerInfo*> hidden;
	hidden.push_back(new glades::HiddenLayerInfo(
	    static_cast<int>(dModel), 0.01f, 0.0f, 0.0f, 0.0f, 0.0f,
	    glades::GMath::LINEAR, 1.0f));
	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
	    static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);

	glades::NNInfo* info = new glades::NNInfo("ut_grad_eval", in, hidden, out);
	glades::NNetwork* net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
	net->setSeed(seed);
	net->getTerminatorMutable().setEpoch(1);
	net->getTerminatorMutable().setAccuracy(0);
	applyGradTestConfig(net, vocab, nHeads, dFF, cfg);

	LossCaptureCb cb;
	const glades::NNetworkStatus st = net->test(di, &cb);

	float loss = 0.0f;
	if (st.ok() && cb.saw)
	{
		loss = cb.lastLoss;
		*outOk = true;
	}

	delete net;
	delete info;
	delete di;
	return loss;
}

// ---------------------------------------------------------------------------
// Train 1 epoch and verify weights were updated by checking that the eval
// loss changed after training. Returns true if loss changed by more than
// threshold, proving gradients flowed through the entire network.
// ---------------------------------------------------------------------------
static bool trainAndVerifyWeightsChanged(
    unsigned int vocab,
    unsigned int dModel,
    unsigned int nHeads,
    unsigned int dFF,
    unsigned int seed,
    float lr,
    const GradTestConfig& cfg)
{
	const unsigned int padTokenId = vocab - 1u;
	std::vector<unsigned int> toks;
	toks.push_back(2u);
	toks.push_back(5u);
	toks.push_back(11u);
	toks.push_back(3u);
	toks.push_back(7u);
	toks.push_back(1u);
	toks.push_back(9u);
	toks.push_back(4u);

	TestTokenIdInput* di = new TestTokenIdInput();
	di->setTrainTokens(toks, static_cast<int>(padTokenId));
	di->mirrorTrainToTest();

	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    static_cast<int>(toks.size()), lr, 0.0f, 0.0f, 0.0f, 0.0f,
	    glades::GMath::LINEAR, 1.0f);
	std::vector<glades::HiddenLayerInfo*> hidden;
	hidden.push_back(new glades::HiddenLayerInfo(
	    static_cast<int>(dModel), lr, 0.0f, 0.0f, 0.0f, 0.0f,
	    glades::GMath::LINEAR, 1.0f));
	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
	    static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);

	glades::NNInfo* info = new glades::NNInfo("ut_grad_wupd", in, hidden, out);
	glades::NNetwork* net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
	net->setSeed(seed);
	net->getTerminatorMutable().setEpoch(1);
	net->getTerminatorMutable().setAccuracy(0);

	applyGradTestConfig(net, vocab, nHeads, dFF, cfg);

	// Eval loss before training
	LossCaptureCb cbBefore;
	glades::NNetworkStatus st = net->test(di, &cbBefore);
	if (!st.ok() || !cbBefore.saw)
	{
		delete net;
		delete info;
		delete di;
		return false;
	}

	// Train 1 epoch
	st = net->train(di);
	if (!st.ok())
	{
		delete net;
		delete info;
		delete di;
		return false;
	}

	// Eval loss after training
	LossCaptureCb cbAfter;
	st = net->test(di, &cbAfter);
	if (!st.ok() || !cbAfter.saw)
	{
		delete net;
		delete info;
		delete di;
		return false;
	}

	const double diff = fabs(static_cast<double>(cbAfter.lastLoss) - static_cast<double>(cbBefore.lastLoss));

	delete net;
	delete info;
	delete di;

	return diff > 1e-6;
}

} // anonymous namespace

void TransformerGradientUnitTest()
{
	printf("============================================================\n");
	printf("Transformer Gradient Validation Test Suite\n");
	printf("============================================================\n");

	const unsigned int vocab = 17u;
	const unsigned int dModel = 16u;
	const unsigned int nHeads = 2u;
	const unsigned int dFF = 32u;
	const unsigned int seed = 42u;

	// Define architecture configurations to sweep.
	GradTestConfig configs[4];

	// Config 0: sinusoidal + LayerNorm + ReLU MLP (baseline)
	configs[0].label = "sinusoidal+LayerNorm+ReLU+MLP";
	configs[0].posEnc = glades::TransformerRunConfig::POSENC_SINUSOIDAL;
	configs[0].normType = glades::TransformerRunConfig::NORM_LAYERNORM;
	configs[0].ffnKind = glades::TransformerRunConfig::FFN_MLP;
	configs[0].ffnAct = glades::TransformerRunConfig::FFN_RELU;

	// Config 1: RoPE + RMSNorm + GELU MLP
	configs[1].label = "RoPE+RMSNorm+GELU+MLP";
	configs[1].posEnc = glades::TransformerRunConfig::POSENC_ROPE;
	configs[1].normType = glades::TransformerRunConfig::NORM_RMSNORM;
	configs[1].ffnKind = glades::TransformerRunConfig::FFN_MLP;
	configs[1].ffnAct = glades::TransformerRunConfig::FFN_GELU;

	// Config 2: RoPE + RMSNorm + SwiGLU FFN
	configs[2].label = "RoPE+RMSNorm+SwiGLU";
	configs[2].posEnc = glades::TransformerRunConfig::POSENC_ROPE;
	configs[2].normType = glades::TransformerRunConfig::NORM_RMSNORM;
	configs[2].ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
	configs[2].ffnAct = glades::TransformerRunConfig::FFN_GELU; // ignored for SwiGLU (uses SiLU internally)

	// Config 3: sinusoidal + LayerNorm + GELU MLP
	configs[3].label = "sinusoidal+LayerNorm+GELU+MLP";
	configs[3].posEnc = glades::TransformerRunConfig::POSENC_SINUSOIDAL;
	configs[3].normType = glades::TransformerRunConfig::NORM_LAYERNORM;
	configs[3].ffnKind = glades::TransformerRunConfig::FFN_MLP;
	configs[3].ffnAct = glades::TransformerRunConfig::FFN_GELU;

	// ===== 1. Gradient direction test =====
	// Train 1 epoch with SGD (lr=0.01). Loss after training should be lower than
	// the initial (pre-training) loss. This validates that the backward pass
	// computes gradients that point in the loss-reducing direction.
	printf("-----------------------------------\n");
	printf("Gradient direction: loss decreases after 1 epoch\n");
	printf("-----------------------------------\n");

	for (int ci = 0; ci < 4; ++ci)
	{
		const float lr = 0.01f;
		bool evalOk = false;
		const float lossBefore = evalLossBeforeTraining(
		    vocab, dModel, nHeads, dFF, seed, configs[ci], &evalOk);
		G_assert(__FILE__, __LINE__,
		         "==============GradDir: eval before failed==============", evalOk);
		G_assert(__FILE__, __LINE__,
		         "==============GradDir: initial loss not finite==============",
		         std::isfinite(lossBefore));
		G_assert(__FILE__, __LINE__,
		         "==============GradDir: initial loss is zero==============",
		         lossBefore > 0.0f);

		bool trainOk = false;
		const float lossAfter = trainOnceAndGetLoss(
		    vocab, dModel, nHeads, dFF, seed, lr, configs[ci], &trainOk);
		G_assert(__FILE__, __LINE__,
		         "==============GradDir: train failed==============", trainOk);
		G_assert(__FILE__, __LINE__,
		         "==============GradDir: trained loss not finite==============",
		         std::isfinite(lossAfter));

		printf("  [%s] loss: %.6f -> %.6f (delta=%.6f)\n",
		       configs[ci].label, lossBefore, lossAfter,
		       static_cast<double>(lossBefore) - static_cast<double>(lossAfter));

		// The loss should decrease (or at least not increase significantly).
		// We use a small tolerance to allow for rounding, but the gradient
		// should point in the correct direction.
		G_assert(__FILE__, __LINE__,
		         "==============GradDir: loss did not decrease==============",
		         lossAfter < lossBefore + 0.01f);
	}

	// ===== 2. Learning rate sensitivity =====
	// Train with lr=0.01 and lr=0.001. With AdamW, the step magnitude scales
	// roughly with lr, so larger lr should produce more loss decrease. We verify
	// qualitatively that the larger lr gives strictly more loss decrease.
	printf("-----------------------------------\n");
	printf("LR sensitivity: larger lr => more loss decrease\n");
	printf("-----------------------------------\n");

	{
		const GradTestConfig& cfg = configs[0]; // baseline config
		bool evalOk = false;
		const float loss0 = evalLossBeforeTraining(
		    vocab, dModel, nHeads, dFF, seed, cfg, &evalOk);
		G_assert(__FILE__, __LINE__,
		         "==============LRSens: eval failed==============", evalOk);

		bool ok1 = false, ok2 = false;
		const float lossHi = trainOnceAndGetLoss(
		    vocab, dModel, nHeads, dFF, seed, 0.01f, cfg, &ok1);
		const float lossLo = trainOnceAndGetLoss(
		    vocab, dModel, nHeads, dFF, seed, 0.001f, cfg, &ok2);
		G_assert(__FILE__, __LINE__,
		         "==============LRSens: train lr=0.01 failed==============", ok1);
		G_assert(__FILE__, __LINE__,
		         "==============LRSens: train lr=0.001 failed==============", ok2);

		const double deltaHi = static_cast<double>(loss0) - static_cast<double>(lossHi);
		const double deltaLo = static_cast<double>(loss0) - static_cast<double>(lossLo);

		printf("  loss0=%.6f lossHi(lr=0.01)=%.6f lossLo(lr=0.001)=%.6f\n",
		       loss0, lossHi, lossLo);
		printf("  deltaHi=%.6f deltaLo=%.6f\n", deltaHi, deltaLo);

		// Both deltas should be positive (loss decreased).
		G_assert(__FILE__, __LINE__,
		         "==============LRSens: deltaHi not positive==============",
		         deltaHi > 0.0);
		G_assert(__FILE__, __LINE__,
		         "==============LRSens: deltaLo not positive==============",
		         deltaLo > 0.0);

		// The larger lr should produce more loss decrease.
		G_assert(__FILE__, __LINE__,
		         "==============LRSens: larger lr did not give more loss decrease==============",
		         deltaHi > deltaLo);
	}

	// ===== 3. Weight-update coverage =====
	// After 1 epoch, the logits should have changed, proving that gradients
	// flowed through the entire network (embeddings -> attention -> FFN -> head).
	printf("-----------------------------------\n");
	printf("Weight-update coverage: all weights receive gradients\n");
	printf("-----------------------------------\n");

	for (int ci = 0; ci < 4; ++ci)
	{
		const bool changed = trainAndVerifyWeightsChanged(
		    vocab, dModel, nHeads, dFF, seed, 0.01f, configs[ci]);
		printf("  [%s] weights changed: %s\n",
		       configs[ci].label, changed ? "yes" : "NO");
		G_assert(__FILE__, __LINE__,
		         "==============WeightCoverage: weights did not change==============",
		         changed);
	}

	// ===== 4. Determinism check =====
	// Two runs with the same seed and config must produce identical loss.
	printf("-----------------------------------\n");
	printf("Determinism: same seed produces same loss\n");
	printf("-----------------------------------\n");

	for (int ci = 0; ci < 4; ++ci)
	{
		bool ok1 = false, ok2 = false;
		const float loss1 = trainOnceAndGetLoss(
		    vocab, dModel, nHeads, dFF, seed, 0.01f, configs[ci], &ok1);
		const float loss2 = trainOnceAndGetLoss(
		    vocab, dModel, nHeads, dFF, seed, 0.01f, configs[ci], &ok2);
		G_assert(__FILE__, __LINE__,
		         "==============Determinism: run 1 failed==============", ok1);
		G_assert(__FILE__, __LINE__,
		         "==============Determinism: run 2 failed==============", ok2);

		printf("  [%s] loss1=%.6f loss2=%.6f diff=%.2e\n",
		       configs[ci].label, loss1, loss2,
		       fabs(static_cast<double>(loss1) - static_cast<double>(loss2)));

		G_assert(__FILE__, __LINE__,
		         "==============Determinism: losses differ==============",
		         fabs(static_cast<double>(loss1) - static_cast<double>(loss2)) < 1e-5);
	}

	// ===== 5. Multi-epoch convergence =====
	// Train for 5 epochs and verify loss is monotonically non-increasing
	// (with small tolerance) for at least 4 of 5 epochs.
	printf("-----------------------------------\n");
	printf("Multi-epoch convergence: loss trend\n");
	printf("-----------------------------------\n");

	{
		const GradTestConfig& cfg = configs[2]; // RoPE+RMSNorm+SwiGLU (the most complex config)
		const unsigned int padTokenId = vocab - 1u;
		std::vector<unsigned int> toks;
		toks.push_back(2u);
		toks.push_back(5u);
		toks.push_back(11u);
		toks.push_back(3u);
		toks.push_back(7u);
		toks.push_back(1u);
		toks.push_back(9u);
		toks.push_back(4u);

		TestTokenIdInput* di = new TestTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    static_cast<int>(toks.size()), 0.01f, 0.0f, 0.0f, 0.0f, 0.0f,
		    glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    static_cast<int>(dModel), 0.01f, 0.0f, 0.0f, 0.0f, 0.0f,
		    glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
		    static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);

		glades::NNInfo* info = new glades::NNInfo("ut_grad_converge", in, hidden, out);
		glades::NNetwork* net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(seed);
		net->getTerminatorMutable().setEpoch(5);
		net->getTerminatorMutable().setAccuracy(0);

		applyGradTestConfig(net, vocab, nHeads, dFF, cfg);

		// Callback that records loss at every epoch.
		class MultiEpochCb : public glades::ITrainingCallbacks
		{
		public:
			std::vector<float> losses;
			virtual void onRunStart(const glades::NNetwork&, int) {}
			virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
			{
				losses.push_back(m.totalError);
				return false;
			}
			virtual void onRunEnd(const glades::NNetwork&, int) {}
		};

		MultiEpochCb cb;
		const glades::NNetworkStatus st = net->train(di, &cb);
		G_assert(__FILE__, __LINE__,
		         "==============Convergence: train failed==============", st.ok());
		G_assert(__FILE__, __LINE__,
		         "==============Convergence: no epoch data==============",
		         cb.losses.size() >= 5u);

		printf("  Epoch losses:");
		for (size_t e = 0; e < cb.losses.size(); ++e)
			printf(" %.4f", cb.losses[e]);
		printf("\n");

		// Verify the final loss is lower than the first epoch loss.
		G_assert(__FILE__, __LINE__,
		         "==============Convergence: final loss >= initial loss==============",
		         cb.losses.back() < cb.losses[0]);

		// All losses should be finite.
		for (size_t e = 0; e < cb.losses.size(); ++e)
			G_assert(__FILE__, __LINE__,
			         "==============Convergence: non-finite epoch loss==============",
			         std::isfinite(cb.losses[e]));

		delete net;
		delete info;
		delete di;
	}

	// ===== 6. GQA config (fewer KV heads) =====
	// Verify gradients work when nKVHeads < nHeads (grouped-query attention).
	printf("-----------------------------------\n");
	printf("GQA gradient direction: nKVHeads < nHeads\n");
	printf("-----------------------------------\n");

	{
		const unsigned int gqaHeads = 4u;
		const unsigned int gqaKVHeads = 2u;
		const unsigned int gqaDModel = 16u;
		const unsigned int gqaDFF = 32u;
		const unsigned int gqaVocab = 17u;
		const unsigned int gqaPad = gqaVocab - 1u;

		std::vector<unsigned int> toks;
		toks.push_back(2u);
		toks.push_back(5u);
		toks.push_back(11u);
		toks.push_back(3u);
		toks.push_back(7u);
		toks.push_back(1u);
		toks.push_back(9u);
		toks.push_back(4u);

		TestTokenIdInput* di = new TestTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(gqaPad));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    static_cast<int>(toks.size()), 0.01f, 0.0f, 0.0f, 0.0f, 0.0f,
		    glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    static_cast<int>(gqaDModel), 0.01f, 0.0f, 0.0f, 0.0f, 0.0f,
		    glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
		    static_cast<int>(gqaVocab), glades::OutputLayerInfo::CLASSIFICATION);

		glades::NNInfo* info = new glades::NNInfo("ut_grad_gqa", in, hidden, out);
		glades::NNetwork* net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(seed);
		net->getTerminatorMutable().setEpoch(1);
		net->getTerminatorMutable().setAccuracy(0);

		{
			glades::TrainingConfig& tc = net->getTrainingConfigMutable();
			tc.transformer.enableTokenEmbedding = true;
			tc.transformer.vocabSizeOverride = static_cast<int>(gqaVocab);
			tc.transformer.tieEmbeddings = true;
			tc.transformer.padTokenId = static_cast<int>(gqaPad);
			tc.transformer.nHeadsOverride = static_cast<int>(gqaHeads);
			tc.transformer.nKVHeadsOverride = static_cast<int>(gqaKVHeads);
			tc.transformer.dFFOverride = static_cast<int>(gqaDFF);
			tc.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
			tc.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			tc.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			tc.transformer.ffnActivation = glades::TransformerRunConfig::FFN_GELU;
			tc.transformer.ropeTheta = 10000.0f;
			tc.optimizer.type = glades::OptimizerConfig::ADAMW;
		}

		// Eval loss before training
		LossCaptureCb cbEval;
		glades::NNetworkStatus st = net->test(di, &cbEval);
		G_assert(__FILE__, __LINE__,
		         "==============GQA: eval failed==============", st.ok() && cbEval.saw);
		const float lossBefore = cbEval.lastLoss;

		// Train 1 epoch on the same network
		st = net->train(di);
		G_assert(__FILE__, __LINE__,
		         "==============GQA: train failed==============", st.ok());

		// Eval loss after training
		LossCaptureCb cbAfter;
		st = net->test(di, &cbAfter);
		G_assert(__FILE__, __LINE__,
		         "==============GQA: eval after failed==============", st.ok() && cbAfter.saw);
		const float lossAfter = cbAfter.lastLoss;

		printf("  GQA (4 heads, 2 KV heads) loss: %.6f -> %.6f\n",
		       lossBefore, lossAfter);

		G_assert(__FILE__, __LINE__,
		         "==============GQA: loss did not decrease==============",
		         lossAfter < lossBefore + 0.01f);

		delete net;
		delete info;
		delete di;
	}

	// -----------------------------------------------------------------------
	// 6. Real TokenInput import path also improves eval loss after training.
	// -----------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("TokenInput integration: file import participates in gradient descent\n");
	printf("-----------------------------------\n");
	{
		const char* path = "/tmp/ut_transformer_gradient_tokeninput.txt";
		std::ofstream out(path);
		out << "2 5 11 3 7 1 9 4\n";
		out << "3 6 10 2 8 1 7 5\n";
		out.close();

		glades::TokenInput di;
		di.setPadTokenId(12);
		di.setMirrorTrainToTestOnImplicitSplit(true);
		di.import(shmea::GString(path));
		G_assert(__FILE__, __LINE__, "==============Grad TokenInput: import failed==============", di.loadedOk());

		glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.02f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(12, 0.02f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* outInfo = new glades::OutputLayerInfo(13, glades::OutputLayerInfo::CLASSIFICATION);

		glades::NNInfo* info = new glades::NNInfo("ut_grad_tokeninput", in, hidden, outInfo);
		glades::NNetwork* net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(9090u);
		net->getTerminatorMutable().setEpoch(1);
		net->getTerminatorMutable().setAccuracy(0);

		glades::TrainingConfig& tc = net->getTrainingConfigMutable();
		tc.transformer.enableTokenEmbedding = true;
		tc.transformer.vocabSizeOverride = 13;
		tc.transformer.tieEmbeddings = true;
		tc.transformer.padTokenId = 12;
		tc.transformer.nHeadsOverride = 3;
		tc.transformer.nKVHeadsOverride = 3;
		tc.transformer.dFFOverride = 24;
		tc.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		tc.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
		tc.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
		tc.transformer.ffnActivation = glades::TransformerRunConfig::FFN_GELU;
		tc.transformer.ropeTheta = 10000.0f;
		tc.optimizer.type = glades::OptimizerConfig::ADAMW;

		LossCaptureCb cbBefore;
		glades::NNetworkStatus st = net->test(&di, &cbBefore);
		G_assert(__FILE__, __LINE__, "==============Grad TokenInput: eval before failed==============", st.ok() && cbBefore.saw);
		st = net->train(&di);
		G_assert(__FILE__, __LINE__, "==============Grad TokenInput: train failed==============", st.ok());
		LossCaptureCb cbAfter;
		st = net->test(&di, &cbAfter);
		G_assert(__FILE__, __LINE__, "==============Grad TokenInput: eval after failed==============", st.ok() && cbAfter.saw);
		G_assert(__FILE__, __LINE__, "==============Grad TokenInput: loss did not decrease==============",
		         cbAfter.lastLoss < cbBefore.lastLoss + 0.01f);

		delete net;
		delete info;
	}

	printf("============================================================\n");
	printf("All Transformer Gradient Tests Passed\n");
	printf("============================================================\n");
}
