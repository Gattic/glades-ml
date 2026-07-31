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

#include "transformer-verification-test.h"
#include "../../unit-test.h"
#include "test_token_id_input_fixture.h"

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/Networks/transformer_config.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

struct SmallDecoderLm
{
	InMemoryTokenIdInput* di;
	glades::NNInfo* info;
	glades::NNetwork* net;
	unsigned int vocab;
	unsigned int padTokenId;

	SmallDecoderLm(unsigned int newVocab)
	    : di(NULL),
	      info(NULL),
	      net(NULL),
	      vocab(newVocab),
	      padTokenId(newVocab - 1u)
	{
		std::vector<unsigned int> toks;
		toks.push_back(1u);
		toks.push_back(2u);
		toks.push_back(3u);
		toks.push_back(4u);
		toks.push_back(5u);

		di = new InMemoryTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		info = new glades::NNInfo("ut_transformer_verification_decoder_lm", in, hidden, out);

		net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(9001u);
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
			cfg.transformer.ffnActivation = glades::TransformerRunConfig::FFN_GELU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
			cfg.transformer.ropeTheta = 10000.0f;
		}

		const glades::NNetworkStatus st = net->test(di);
		ASSERT("==============TransformerVerification: SmallDecoderLm InitTestStatus Failed==============", st.ok());
	}

	~SmallDecoderLm()
	{
		delete net;
		delete di;
		delete info;
	}
};

static glades::TrainingConfig make_valid_training_cfg(unsigned int vocab);

struct TrainableDecoderLm
{
	InMemoryTokenIdInput* di;
	glades::NNInfo* info;
	glades::NNetwork* net;
	unsigned int vocab;
	unsigned int padTokenId;

	TrainableDecoderLm(unsigned int newVocab,
	                   unsigned int seqLen,
	                   unsigned int seed,
	                   bool bootstrapNow = true)
	    : di(NULL),
	      info(NULL),
	      net(NULL),
	      vocab(newVocab),
	      padTokenId(newVocab - 1u)
	{
		const unsigned int tokenSpan = (newVocab > 2u) ? (newVocab - 2u) : 1u;
		std::vector<unsigned int> toks;
		toks.reserve(seqLen);
		for (unsigned int i = 0u; i < seqLen; ++i)
			toks.push_back(1u + (i % tokenSpan));

		di = new InMemoryTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    1, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    16, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		info = new glades::NNInfo("ut_transformer_verification_train_decoder_lm", in, hidden, out);

		net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(seed);
		net->getTerminatorMutable().setEpoch(1);
		net->getTerminatorMutable().setAccuracy(0);

		glades::TrainingConfig cfg = make_valid_training_cfg(vocab);
		cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
		const glades::NNetworkStatus cfgSt = net->setTrainingConfig(cfg);
		ASSERT("==============TransformerVerification: TrainableDecoderLm SetTrainingConfig Failed==============", cfgSt.ok());

		if (bootstrapNow)
		{
			const glades::NNetworkStatus st = net->test(di);
			ASSERT("==============TransformerVerification: TrainableDecoderLm InitTestStatus Failed==============", st.ok());
		}
	}

	~TrainableDecoderLm()
	{
		delete net;
		delete di;
		delete info;
	}
};

static glades::TrainingConfig make_valid_training_cfg(unsigned int vocab)
{
	glades::TrainingConfig cfg;
	cfg.transformer.enableTokenEmbedding = true;
	cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
	cfg.transformer.tieEmbeddings = true;
	cfg.transformer.padTokenId = static_cast<int>(vocab - 1u);
	cfg.transformer.nHeadsOverride = 4;
	cfg.transformer.nKVHeadsOverride = 2;
	cfg.transformer.dFFOverride = 32;
	cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
	cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
	cfg.transformer.kvCacheDType = glades::TransformerRunConfig::KV_CACHE_BF16;
	cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
	cfg.transformer.ffnActivation = glades::TransformerRunConfig::FFN_GELU;
	return cfg;
}

static glades::NNetwork::TransformerServeRequest make_request(const std::vector<unsigned int>& prompt,
                                                              unsigned int maxNewTokens,
                                                              bool includePromptInOutput,
                                                              uint64_t seed)
{
	glades::NNetwork::TransformerServeRequest req;
	req.promptTokens = prompt;
	req.cfg = glades::NNetwork::TransformerGenerateConfig();
	req.cfg.maxNewTokens = maxNewTokens;
	req.cfg.maxSeqLen = 0u;
	req.cfg.temperature = 1.0f;
	req.cfg.topK = 1u;
	req.cfg.topP = 1.0f;
	req.cfg.topPTopKCap = 256u;
	req.cfg.eosTokenId = -1;
	req.cfg.stopOnEos = true;
	req.cfg.includePromptInOutput = includePromptInOutput;
	req.cfg.rngSeedOverride = seed;
	return req;
}

static void assert_tokens_in_range(const std::vector<unsigned int>& toks, unsigned int vocab)
{
	for (size_t i = 0u; i < toks.size(); ++i)
		ASSERT("==============TransformerVerification: token out of range Failed==============", toks[i] < vocab);
}

static void assert_status_error(const glades::NNetworkStatus& st,
                                glades::NNetworkStatus::Code expectedCode,
                                const char* expectedPrefix,
                                const char* expectedFragment)
{
	ASSERT("==============TransformerVerification: expected error status Failed==============", !st.ok());
	ASSERT("==============TransformerVerification: unexpected status code Failed==============", st.code == expectedCode);
	ASSERT("==============TransformerVerification: status prefix mismatch Failed==============",
	       st.message.find(expectedPrefix) == 0u);
	ASSERT("==============TransformerVerification: status fragment missing Failed==============",
	       st.message.find(expectedFragment) != std::string::npos);
}

static std::vector<unsigned int> make_hidden_sizes(unsigned int dModel, unsigned int nLayers)
{
	return std::vector<unsigned int>(nLayers, dModel);
}

class StopAfterFirstTokenCallback : public glades::ITransformerServeCallbacks
{
public:
	StopAfterFirstTokenCallback() : onTokenCalls_(0u) {}

	virtual bool onToken(const glades::NNetwork& /*net*/,
	                     unsigned int /*requestIndex*/,
	                     unsigned int /*tokenId*/,
	                     unsigned int generatedIndex)
	{
		++onTokenCalls_;
		return (generatedIndex == 0u);
	}

	unsigned int onTokenCalls() const { return onTokenCalls_; }

private:
	unsigned int onTokenCalls_;
};

class StopAllImmediatelyCallback : public glades::ITransformerServeCallbacks
{
public:
	StopAllImmediatelyCallback() : shouldStopAllCalls_(0u) {}

	virtual bool shouldStopAll(const glades::NNetwork& /*net*/)
	{
		++shouldStopAllCalls_;
		return true;
	}

	unsigned int shouldStopAllCalls() const { return shouldStopAllCalls_; }

private:
	unsigned int shouldStopAllCalls_;
};

class CaptureEpochMetricsCallback : public glades::ITrainingCallbacks
{
public:
	CaptureEpochMetricsCallback()
	    : last(),
	      sawRunStart(false),
	      sawEpoch(false),
	      sawRunEnd(false)
	{
	}

	virtual void onRunStart(const glades::NNetwork& /*net*/, int /*runType*/)
	{
		sawRunStart = true;
	}

	virtual bool onEpochEnd(const glades::NNetwork& /*net*/, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		sawEpoch = true;
		return false;
	}

	virtual void onRunEnd(const glades::NNetwork& /*net*/, int /*runType*/)
	{
		sawRunEnd = true;
	}

	glades::NNetworkEpochMetrics last;
	bool sawRunStart;
	bool sawEpoch;
	bool sawRunEnd;
};

class SetTrainingConfigWhileRunningCallback : public glades::ITrainingCallbacks
{
public:
	explicit SetTrainingConfigWhileRunningCallback(const glades::TrainingConfig& cfg)
	    : sawRunStart(false),
	      setConfigStatus(glades::NNetworkStatus::OK, std::string()),
	      cfg_(cfg)
	{
	}

	virtual void onRunStart(const glades::NNetwork& net, int /*runType*/)
	{
		sawRunStart = true;
		glades::NNetwork& mutNet = const_cast<glades::NNetwork&>(net);
		setConfigStatus = mutNet.setTrainingConfig(cfg_);
	}

	virtual bool onEpochEnd(const glades::NNetwork& /*net*/, const glades::NNetworkEpochMetrics& /*m*/)
	{
		return false;
	}

	bool sawRunStart;
	glades::NNetworkStatus setConfigStatus;

private:
	glades::TrainingConfig cfg_;
};

static void test_runtime_config_snapshot_validation()
{
	printf("  [A1] RuntimeConfigSnapshotValidation ...\n");

	glades::TransformerRunConfig runtime;
	runtime.enableTokenEmbedding = true;
	runtime.normType = glades::TransformerRunConfig::NORM_RMSNORM;
	runtime.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
	runtime.kvCacheDType = glades::TransformerRunConfig::KV_CACHE_BF16;
	runtime.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
	runtime.ffnActivation = glades::TransformerRunConfig::FFN_GELU;
	runtime.padTokenId = 17;
	runtime.layerNormEps = 0.0f;
	runtime.ropeTheta = 0.0f;

	glades::TransformerRuntimeConfigSnapshot snap;
	glades::NNetworkStatus st = glades::buildTransformerRuntimeConfigSnapshot("ut_transformer_runtime", runtime, snap);
	ASSERT("runtime snapshot should succeed", st.ok());
	ASSERT("runtime snapshot token model", snap.tokenModel);
	ASSERT("runtime snapshot norm type", snap.normType == static_cast<unsigned int>(glades::TransformerRunConfig::NORM_RMSNORM));
	ASSERT("runtime snapshot pos enc", snap.positionalEncoding == static_cast<unsigned int>(glades::TransformerRunConfig::POSENC_ROPE));
	ASSERT("runtime snapshot kv dtype", snap.kvCacheDType == static_cast<unsigned int>(glades::TransformerRunConfig::KV_CACHE_BF16));
	ASSERT("runtime snapshot ffn kind", snap.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU));
	ASSERT("runtime snapshot ffn activation", snap.ffnActivation == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_GELU));
	ASSERT("runtime snapshot layerNormEps default", std::fabs(snap.layerNormEps - 1e-5f) < 1e-8f);
	ASSERT("runtime snapshot ropeTheta default", std::fabs(snap.ropeTheta - 10000.0f) < 1e-4f);
	ASSERT("runtime snapshot pad token", snap.padTokenId == 17);

	runtime = glades::TransformerRunConfig();
	runtime.positionalEncoding = static_cast<glades::TransformerRunConfig::PositionalEncodingType>(99);
	st = glades::buildTransformerRuntimeConfigSnapshot("ut_transformer_runtime", runtime, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_runtime: ",
	                    "unknown positionalEncoding");

	runtime = glades::TransformerRunConfig();
	runtime.normType = static_cast<glades::TransformerRunConfig::NormType>(99);
	st = glades::buildTransformerRuntimeConfigSnapshot("ut_transformer_runtime", runtime, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_runtime: ",
	                    "unknown normType");

	runtime = glades::TransformerRunConfig();
	runtime.ffnKind = static_cast<glades::TransformerRunConfig::FFNKind>(99);
	st = glades::buildTransformerRuntimeConfigSnapshot("ut_transformer_runtime", runtime, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_runtime: ",
	                    "unknown ffnKind");

	runtime = glades::TransformerRunConfig();
	runtime.ffnActivation = static_cast<glades::TransformerRunConfig::FFNActivationType>(99);
	st = glades::buildTransformerRuntimeConfigSnapshot("ut_transformer_runtime", runtime, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_runtime: ",
	                    "unknown ffnActivation");

	runtime = glades::TransformerRunConfig();
	runtime.kvCacheDType = static_cast<glades::TransformerRunConfig::KVCacheDType>(99);
	st = glades::buildTransformerRuntimeConfigSnapshot("ut_transformer_runtime", runtime, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_runtime: ",
	                    "unknown kvCacheDType");

	runtime = glades::TransformerRunConfig();
	runtime.embeddingDropoutRate = -0.01f;
	st = glades::buildTransformerRuntimeConfigSnapshot("ut_transformer_runtime", runtime, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_runtime: ",
	                    "embeddingDropoutRate must be in [0,1)");

	runtime = glades::TransformerRunConfig();
	runtime.residualDropoutRate = 1.0f;
	st = glades::buildTransformerRuntimeConfigSnapshot("ut_transformer_runtime", runtime, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_runtime: ",
	                    "residualDropoutRate must be in [0,1)");

	printf("    PASSED\n");
}

static void test_runtime_config_snapshot_boundary_contract()
{
	printf("  [A2] RuntimeConfigSnapshotBoundaryContract ...\n");

	glades::TransformerRunConfig runtime;
	runtime.enableTokenEmbedding = false;
	runtime.layerNormEps = 2.5e-4f;
	runtime.normType = glades::TransformerRunConfig::NORM_LAYERNORM;
	runtime.positionalEncoding = glades::TransformerRunConfig::POSENC_NONE;
	runtime.kvCacheDType = glades::TransformerRunConfig::KV_CACHE_F16;
	runtime.ropeDimOverride = 14;
	runtime.ropeTheta = 321.0f;
	runtime.ffnKind = glades::TransformerRunConfig::FFN_MLP;
	runtime.ffnActivation = glades::TransformerRunConfig::FFN_RELU;
	runtime.padTokenId = -1;
	runtime.embeddingDropoutRate = 0.0f;
	runtime.residualDropoutRate = 0.999f;

	glades::TransformerRuntimeConfigSnapshot snap;
	glades::NNetworkStatus st = glades::buildTransformerRuntimeConfigSnapshot("ut_transformer_runtime_boundaries", runtime, snap);
	ASSERT("runtime boundary snapshot should succeed", st.ok());
	ASSERT("runtime boundary snapshot token model disabled", !snap.tokenModel);
	ASSERT("runtime boundary snapshot layerNormEps preserved", std::fabs(snap.layerNormEps - runtime.layerNormEps) < 1e-8f);
	ASSERT("runtime boundary snapshot ropeDimOverride preserved", snap.ropeDimOverride == runtime.ropeDimOverride);
	ASSERT("runtime boundary snapshot ropeTheta preserved", std::fabs(snap.ropeTheta - runtime.ropeTheta) < 1e-6f);
	ASSERT("runtime boundary snapshot kv dtype preserved",
	       snap.kvCacheDType == static_cast<unsigned int>(glades::TransformerRunConfig::KV_CACHE_F16));
	ASSERT("runtime boundary snapshot ffn kind preserved",
	       snap.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_MLP));
	ASSERT("runtime boundary snapshot ffn activation preserved",
	       snap.ffnActivation == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_RELU));

	runtime = glades::TransformerRunConfig();
	runtime.embeddingDropoutRate = 1.0f;
	st = glades::buildTransformerRuntimeConfigSnapshot("ut_transformer_runtime_boundaries", runtime, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_runtime_boundaries: ",
	                    "embeddingDropoutRate must be in [0,1)");

	runtime = glades::TransformerRunConfig();
	runtime.residualDropoutRate = -0.01f;
	st = glades::buildTransformerRuntimeConfigSnapshot(NULL, runtime, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "transformer_config: ",
	                    "residualDropoutRate must be in [0,1)");

	printf("    PASSED\n");
}

static void test_training_config_validation_contract()
{
	printf("  [A3] TrainingConfigValidationContract ...\n");

	const unsigned int vocab = 33u;
	glades::TrainingConfig cfg = make_valid_training_cfg(vocab);
	glades::NNetworkStatus st = glades::validateTransformerTrainingConfig("ut_transformer_training", cfg);
	ASSERT("training config valid baseline", st.ok());

	cfg.transformer.tokenLmLossKind = static_cast<glades::TransformerRunConfig::TokenLMLossKind>(99);
	st = glades::validateTransformerTrainingConfig("ut_transformer_training", cfg);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_training: ",
	                    "unknown tokenLmLossKind");

	cfg = make_valid_training_cfg(vocab);
	cfg.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX;
	cfg.transformer.tokenLmSampledNegatives = 0;
	st = glades::validateTransformerTrainingConfig("ut_transformer_training", cfg);
	ASSERT("training config full softmax ignores sampled negative count", st.ok());

	cfg = make_valid_training_cfg(vocab);
	cfg.transformer.embeddingDropoutRate = 1.0f;
	st = glades::validateTransformerTrainingConfig("ut_transformer_training", cfg);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_training: ",
	                    "embeddingDropoutRate must be in [0,1)");

	printf("    PASSED\n");
}

static void test_training_config_validation_and_model_snapshot()
{
	printf("  [A4] TrainingConfigValidationAndModelSnapshot ...\n");

	const unsigned int vocab = 33u;
	glades::TrainingConfig cfg = make_valid_training_cfg(vocab);
	cfg.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX;
	cfg.transformer.tokenLmSampledNegatives = 0;

	glades::NNetworkStatus st = glades::validateTransformerTrainingConfig("ut_transformer_training", cfg);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_training: ",
	                    "tokenLmSampledNegatives must be >= 1 for sampled softmax");

	cfg = make_valid_training_cfg(vocab);
	cfg.transformer.layerNormEps = 0.0f;
	cfg.transformer.ropeTheta = 0.0f;

	std::vector<unsigned int> hiddenSizes;
	hiddenSizes.push_back(16u);
	hiddenSizes.push_back(16u);

	glades::TransformerModelConfigSnapshot snap;
	st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model", cfg, hiddenSizes, vocab, true, true, snap);
	ASSERT("model snapshot should succeed", st.ok());
	ASSERT("model snapshot token model", snap.tokenModel);
	ASSERT("model snapshot dModel", snap.dModel == 16u);
	ASSERT("model snapshot dFF", snap.dFF == 32u);
	ASSERT("model snapshot ff1Width", snap.ff1Width == 64u);
	ASSERT("model snapshot nHeads", snap.nHeads == 4u);
	ASSERT("model snapshot nKVHeads", snap.nKVHeads == 2u);
	ASSERT("model snapshot nLayers", snap.nLayers == 2u);
	ASSERT("model snapshot vocabSize", snap.vocabSize == vocab);
	ASSERT("model snapshot causal", snap.causal);
	ASSERT("model snapshot tieEmbeddings", snap.tieEmbeddings);
	ASSERT("model snapshot layerNormEps default", std::fabs(snap.layerNormEps - 1e-5f) < 1e-8f);
	ASSERT("model snapshot ropeTheta default", std::fabs(snap.ropeTheta - 10000.0f) < 1e-4f);

	std::vector<unsigned int> badHiddenSizes;
	st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model", cfg, badHiddenSizes, vocab, true, true, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_STATE,
	                    "ut_transformer_model: ",
	                    "transformer requires >= 1 hidden layer (blocks)");

	cfg = make_valid_training_cfg(vocab);
	hiddenSizes.clear();
	hiddenSizes.push_back(10u);
	st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model", cfg, hiddenSizes, vocab, true, true, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_model: ",
	                    "transformer dModel must be divisible by nHeads");

	cfg = make_valid_training_cfg(vocab);
	cfg.transformer.nKVHeadsOverride = 3;
	hiddenSizes.clear();
	hiddenSizes.push_back(12u);
	st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model", cfg, hiddenSizes, vocab, true, true, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_model: ",
	                    "transformer nKVHeads must divide nHeads");

	cfg = make_valid_training_cfg(vocab);
	hiddenSizes.clear();
	hiddenSizes.push_back(16u);
	hiddenSizes.push_back(12u);
	st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model", cfg, hiddenSizes, vocab, true, true, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_model: ",
	                    "transformer requires constant hidden size (dModel) across all blocks");

	cfg = make_valid_training_cfg(vocab);
	cfg.transformer.tieEmbeddings = false;
	hiddenSizes.clear();
	hiddenSizes.push_back(16u);
	st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model", cfg, hiddenSizes, vocab, true, true, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_model: ",
	                    "token LM mode currently requires tieEmbeddings=true");

	cfg = make_valid_training_cfg(vocab);
	st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model", cfg, hiddenSizes, vocab, false, true, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_model: ",
	                    "token LM mode requires DataInput token-id accessors");

	cfg = make_valid_training_cfg(vocab);
	st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model", cfg, hiddenSizes, vocab + 1u, true, true, snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_model: ",
	                    "token LM vocabSizeOverride must match NNInfo output layer size");

	printf("    PASSED\n");
}

static void test_model_snapshot_default_and_non_token_contract()
{
	printf("  [A5] ModelSnapshotDefaultAndNonTokenContract ...\n");

	const std::vector<unsigned int> hiddenSizes = make_hidden_sizes(16u, 2u);
	glades::TransformerModelConfigSnapshot snap;

	glades::TrainingConfig nonTokenCfg;
	nonTokenCfg.transformer.tieEmbeddings = false;
	nonTokenCfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
	nonTokenCfg.transformer.ffnActivation = glades::TransformerRunConfig::FFN_RELU;
	nonTokenCfg.transformer.kvCacheDType = glades::TransformerRunConfig::KV_CACHE_F16;
	nonTokenCfg.transformer.layerNormEps = 2.0e-5f;
	nonTokenCfg.transformer.ropeDimOverride = 6;
	nonTokenCfg.transformer.ropeTheta = 5000.0f;

	glades::NNetworkStatus st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model_defaults",
	                                                                        nonTokenCfg,
	                                                                        hiddenSizes,
	                                                                        0u,
	                                                                        false,
	                                                                        false,
	                                                                        snap);
	ASSERT("non-token model snapshot should succeed", st.ok());
	ASSERT("non-token model flag disabled", !snap.tokenModel);
	ASSERT("non-token default heads", snap.nHeads == 4u);
	ASSERT("non-token default kv heads", snap.nKVHeads == 4u);
	ASSERT("non-token default dFF", snap.dFF == 64u);
	ASSERT("non-token mlp ff1 width", snap.ff1Width == 64u);
	ASSERT("non-token allows zero vocab", snap.vocabSize == 0u);
	ASSERT("non-token tieEmbeddings passthrough", !snap.tieEmbeddings);
	ASSERT("non-token causal flag", !snap.causal);
	ASSERT("non-token runtime fields copied", snap.kvCacheDType == static_cast<unsigned int>(glades::TransformerRunConfig::KV_CACHE_F16));
	ASSERT("non-token ropeDimOverride copied", snap.ropeDimOverride == 6);
	ASSERT("non-token positive ropeTheta preserved", std::fabs(snap.ropeTheta - 5000.0f) < 1e-4f);

	glades::TrainingConfig tokenDefaultCfg = make_valid_training_cfg(7u);
	tokenDefaultCfg.transformer.nHeadsOverride = 0;
	tokenDefaultCfg.transformer.nKVHeadsOverride = 0;
	tokenDefaultCfg.transformer.dFFOverride = 0;
	tokenDefaultCfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
	tokenDefaultCfg.transformer.vocabSizeOverride = 0;
	st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model_defaults",
	                                                 tokenDefaultCfg,
	                                                 hiddenSizes,
	                                                 11u,
	                                                 true,
	                                                 false,
	                                                 snap);
	ASSERT("token model default derivation should succeed", st.ok());
	ASSERT("token model default heads", snap.nHeads == 4u);
	ASSERT("token model default kv heads", snap.nKVHeads == 4u);
	ASSERT("token model default dFF", snap.dFF == 64u);
	ASSERT("token model default mlp ff1 width", snap.ff1Width == 64u);
	ASSERT("token model vocab falls back to output size", snap.vocabSize == 11u);
	ASSERT("token model causal flag copied", !snap.causal);

	glades::TrainingConfig zeroVocabCfg = make_valid_training_cfg(7u);
	zeroVocabCfg.transformer.vocabSizeOverride = 0;
	st = glades::buildTransformerModelConfigSnapshot("ut_transformer_model_defaults",
	                                                 zeroVocabCfg,
	                                                 hiddenSizes,
	                                                 0u,
	                                                 true,
	                                                 true,
	                                                 snap);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "ut_transformer_model_defaults: ",
	                    "token LM mode requires vocabSize > 0");

	printf("    PASSED\n");
}

static void test_batcher_submit_validation_matrix()
{
	printf("  [B1] BatcherSubmitValidationMatrix ...\n");

	SmallDecoderLm m(19u);

	glades::NNetwork::TransformerServeBatcher batcher;
	glades::NNetwork::TransformerServeBatcherConfig cfg;
	cfg.maxBatchSize = 1u;
	cfg.maxSeqLen = 4u;
	ASSERT("batcher reset ok", m.net->transformerLmServeBatcherReset(batcher, cfg).ok());

	unsigned int slot = 77u;

	glades::NNetwork::TransformerServeRequest emptyPrompt = make_request(std::vector<unsigned int>(), 1u, false, 101u);
	ASSERT("batcher rejects empty prompt", !m.net->transformerLmServeBatcherSubmit(batcher, emptyPrompt, slot).ok());

	std::vector<unsigned int> badPrompt;
	badPrompt.push_back(m.vocab);
	glades::NNetwork::TransformerServeRequest badPromptReq = make_request(badPrompt, 1u, false, 102u);
	ASSERT("batcher rejects prompt token oob", !m.net->transformerLmServeBatcherSubmit(batcher, badPromptReq, slot).ok());

	std::vector<unsigned int> prompt;
	prompt.push_back(1u);
	glades::NNetwork::TransformerServeRequest badStopReq = make_request(prompt, 1u, false, 103u);
	badStopReq.stopTokenIds.push_back(m.vocab);
	ASSERT("batcher rejects stop token oob", !m.net->transformerLmServeBatcherSubmit(batcher, badStopReq, slot).ok());

	std::vector<unsigned int> longPrompt;
	longPrompt.push_back(1u);
	longPrompt.push_back(2u);
	glades::NNetwork::TransformerServeRequest shortMaxReq = make_request(longPrompt, 0u, false, 104u);
	shortMaxReq.cfg.maxSeqLen = 1u;
	ASSERT("batcher rejects maxSeqLen shorter than prompt", !m.net->transformerLmServeBatcherSubmit(batcher, shortMaxReq, slot).ok());

	glades::NNetwork::TransformerServeRequest insufficientReq = make_request(prompt, 3u, false, 105u);
	insufficientReq.cfg.maxSeqLen = 2u;
	ASSERT("batcher rejects maxSeqLen shorter than prompt plus maxNew", !m.net->transformerLmServeBatcherSubmit(batcher, insufficientReq, slot).ok());

	glades::NNetwork::TransformerServeRequest exceedReq = make_request(longPrompt, 2u, false, 106u);
	exceedReq.cfg.maxSeqLen = 5u;
	ASSERT("batcher rejects maxSeqLen above batcher cap", !m.net->transformerLmServeBatcherSubmit(batcher, exceedReq, slot).ok());

	glades::NNetwork::TransformerServeRequest validReq = make_request(prompt, 1u, false, 107u);
	ASSERT("batcher accepts valid request", m.net->transformerLmServeBatcherSubmit(batcher, validReq, slot).ok());
	ASSERT("batcher first slot is zero", slot == 0u);

	unsigned int slot2 = 88u;
	ASSERT("batcher rejects second request when full", !m.net->transformerLmServeBatcherSubmit(batcher, validReq, slot2).ok());

	ASSERT("batcher remove frees slot", m.net->transformerLmServeBatcherRemove(batcher, slot).ok());
	ASSERT("batcher slot free after remove", batcher.slotFree(slot));
	ASSERT("batcher remove is idempotent on free slot", m.net->transformerLmServeBatcherRemove(batcher, slot).ok());
	ASSERT("batcher remove rejects slot out of range", !m.net->transformerLmServeBatcherRemove(batcher, 99u).ok());
	ASSERT("batcher cancel rejects slot out of range", !m.net->transformerLmServeBatcherCancelSlot(batcher, 99u).ok());
	ASSERT("batcher cancel rejects free slot", !m.net->transformerLmServeBatcherCancelSlot(batcher, slot).ok());

	printf("    PASSED\n");
}

static void test_batcher_slot_lifecycle_and_reuse()
{
	printf("  [B2] BatcherSlotLifecycleAndReuse ...\n");

	SmallDecoderLm m(23u);

	glades::NNetwork::TransformerServeBatcher batcher;
	glades::NNetwork::TransformerServeBatcherConfig cfg;
	cfg.maxBatchSize = 2u;
	cfg.maxSeqLen = 8u;
	ASSERT("batcher reset ok", m.net->transformerLmServeBatcherReset(batcher, cfg).ok());
	ASSERT("batcher initialized", batcher.isInitialized());
	ASSERT("batcher capacity", batcher.capacity() == 2u);

	std::vector<unsigned int> promptA;
	promptA.push_back(1u);
	promptA.push_back(2u);

	std::vector<unsigned int> promptB;
	promptB.push_back(3u);

	glades::NNetwork::TransformerServeRequest reqA = make_request(promptA, 2u, false, 201u);
	glades::NNetwork::TransformerServeRequest reqB = make_request(promptB, 0u, true, 202u);

	unsigned int slotA = 99u;
	unsigned int slotB = 99u;
	ASSERT("submit request A", m.net->transformerLmServeBatcherSubmit(batcher, reqA, slotA).ok());
	ASSERT("request A uses slot 0", slotA == 0u);
	ASSERT("request A starts in prefill", batcher.slotLifecycle(slotA) == glades::NNetwork::TransformerServeBatcher::SLOT_PREFILL);
	ASSERT("request A output starts empty", batcher.slotResult(slotA) && batcher.slotResult(slotA)->tokens.empty());

	ASSERT("submit request B", m.net->transformerLmServeBatcherSubmit(batcher, reqB, slotB).ok());
	ASSERT("request B uses slot 1", slotB == 1u);
	ASSERT("request B starts in prefill", batcher.slotLifecycle(slotB) == glades::NNetwork::TransformerServeBatcher::SLOT_PREFILL);
	ASSERT("request B output begins with prompt", batcher.slotResult(slotB) && batcher.slotResult(slotB)->tokens == promptB);

	ASSERT("step prefill #1", m.net->transformerLmServeBatcherStep(batcher, NULL).ok());
	ASSERT("request A still prefill after first token", batcher.slotLifecycle(slotA) == glades::NNetwork::TransformerServeBatcher::SLOT_PREFILL);
	ASSERT("request B done after prompt-only prefill", batcher.slotDone(slotB));
	ASSERT("request B remains in use until remove", batcher.slotInUse(slotB));
	ASSERT("request B stopped by limit", batcher.slotResult(slotB) && batcher.slotResult(slotB)->stoppedByLimit);
	ASSERT("request B prompt preserved", batcher.slotResult(slotB) && batcher.slotResult(slotB)->tokens == promptB);

	ASSERT("remove request B slot", m.net->transformerLmServeBatcherRemove(batcher, slotB).ok());
	ASSERT("request B slot becomes free", batcher.slotFree(slotB));
	ASSERT("removed slot result is cleared", batcher.slotResult(slotB) && batcher.slotResult(slotB)->tokens.empty());

	std::vector<unsigned int> promptC;
	promptC.push_back(4u);
	glades::NNetwork::TransformerServeRequest reqC = make_request(promptC, 1u, false, 203u);
	unsigned int slotC = 99u;
	ASSERT("submit request C", m.net->transformerLmServeBatcherSubmit(batcher, reqC, slotC).ok());
	ASSERT("request C reuses freed slot", slotC == slotB);

	ASSERT("step prefill #2", m.net->transformerLmServeBatcherStep(batcher, NULL).ok());
	ASSERT("request A ready to decode", batcher.slotLifecycle(slotA) == glades::NNetwork::TransformerServeBatcher::SLOT_DECODE);
	ASSERT("request A can decode", batcher.slotCanDecode(slotA));
	ASSERT("request C ready to decode", batcher.slotLifecycle(slotC) == glades::NNetwork::TransformerServeBatcher::SLOT_DECODE);
	ASSERT("request C can decode", batcher.slotCanDecode(slotC));

	ASSERT("step decode #1", m.net->transformerLmServeBatcherStep(batcher, NULL).ok());
	ASSERT("request A still active after first decode token", batcher.slotInUse(slotA) && !batcher.slotDone(slotA));
	ASSERT("request C done after one generated token", batcher.slotDone(slotC));
	ASSERT("request C stopped by limit", batcher.slotResult(slotC) && batcher.slotResult(slotC)->stoppedByLimit);
	ASSERT("request C emitted one token", batcher.slotResult(slotC) && batcher.slotResult(slotC)->tokens.size() == 1u);
	assert_tokens_in_range(batcher.slotResult(slotC)->tokens, m.vocab);

	ASSERT("step decode #2", m.net->transformerLmServeBatcherStep(batcher, NULL).ok());
	ASSERT("request A done after second decode token", batcher.slotDone(slotA));
	ASSERT("request A stopped by limit", batcher.slotResult(slotA) && batcher.slotResult(slotA)->stoppedByLimit);
	ASSERT("request A emitted two tokens", batcher.slotResult(slotA) && batcher.slotResult(slotA)->tokens.size() == 2u);
	assert_tokens_in_range(batcher.slotResult(slotA)->tokens, m.vocab);

	ASSERT("remove request A slot", m.net->transformerLmServeBatcherRemove(batcher, slotA).ok());
	ASSERT("remove request C slot", m.net->transformerLmServeBatcherRemove(batcher, slotC).ok());
	ASSERT("all slots free after removal", batcher.slotFree(0u) && batcher.slotFree(1u));

	printf("    PASSED\n");
}

static void test_batcher_callback_stop_and_manual_cancel()
{
	printf("  [B3] BatcherCallbackStopAndManualCancel ...\n");

	SmallDecoderLm m(17u);

	glades::NNetwork::TransformerServeBatcher batcher;
	glades::NNetwork::TransformerServeBatcherConfig cfg;
	cfg.maxBatchSize = 2u;
	cfg.maxSeqLen = 8u;
	ASSERT("batcher reset ok", m.net->transformerLmServeBatcherReset(batcher, cfg).ok());

	std::vector<unsigned int> promptA;
	promptA.push_back(1u);
	std::vector<unsigned int> promptB;
	promptB.push_back(2u);
	promptB.push_back(3u);

	unsigned int slotA = 99u;
	unsigned int slotB = 99u;
	ASSERT("submit request A", m.net->transformerLmServeBatcherSubmit(batcher, make_request(promptA, 3u, false, 301u), slotA).ok());
	ASSERT("submit request B", m.net->transformerLmServeBatcherSubmit(batcher, make_request(promptB, 3u, false, 302u), slotB).ok());

	StopAllImmediatelyCallback stopAll;
	ASSERT("shouldStopAll callback step succeeds", m.net->transformerLmServeBatcherStep(batcher, &stopAll).ok());
	ASSERT("shouldStopAll called once", stopAll.shouldStopAllCalls() == 1u);
	ASSERT("slot A done by callback", batcher.slotDone(slotA) && batcher.slotResult(slotA) && batcher.slotResult(slotA)->stoppedByCallback);
	ASSERT("slot B done by callback", batcher.slotDone(slotB) && batcher.slotResult(slotB) && batcher.slotResult(slotB)->stoppedByCallback);
	ASSERT("slot A emitted no decode tokens", batcher.slotResult(slotA)->tokens.empty());
	ASSERT("slot B emitted no decode tokens", batcher.slotResult(slotB)->tokens.empty());
	ASSERT("remove slot A after stopAll", m.net->transformerLmServeBatcherRemove(batcher, slotA).ok());
	ASSERT("remove slot B after stopAll", m.net->transformerLmServeBatcherRemove(batcher, slotB).ok());

	unsigned int slotC = 99u;
	ASSERT("submit request C", m.net->transformerLmServeBatcherSubmit(batcher, make_request(promptA, 3u, false, 303u), slotC).ok());
	ASSERT("manual cancel slot succeeds", m.net->transformerLmServeBatcherCancelSlot(batcher, slotC).ok());
	ASSERT("manual cancel marks done", batcher.slotDone(slotC));
	ASSERT("manual cancel marks callback stop", batcher.slotResult(slotC) && batcher.slotResult(slotC)->stoppedByCallback);
	ASSERT("manual cancel does not free slot", batcher.slotInUse(slotC));
	ASSERT("remove slot after manual cancel", m.net->transformerLmServeBatcherRemove(batcher, slotC).ok());
	ASSERT("slot free after remove", batcher.slotFree(slotC));

	glades::NNetwork::TransformerServeBatcher decodeBatcher;
	ASSERT("decode batcher reset ok", m.net->transformerLmServeBatcherReset(decodeBatcher, cfg).ok());
	unsigned int decodeSlot = 99u;
	ASSERT("submit decode request", m.net->transformerLmServeBatcherSubmit(decodeBatcher, make_request(promptA, 3u, false, 304u), decodeSlot).ok());
	ASSERT("decode request prefill", m.net->transformerLmServeBatcherStep(decodeBatcher, NULL).ok());
	ASSERT("decode request can now decode", decodeBatcher.slotCanDecode(decodeSlot));

	StopAfterFirstTokenCallback stopOnToken;
	ASSERT("onToken stop step succeeds", m.net->transformerLmServeBatcherStep(decodeBatcher, &stopOnToken).ok());
	ASSERT("onToken callback invoked once", stopOnToken.onTokenCalls() == 1u);
	ASSERT("decode request done after callback stop", decodeBatcher.slotDone(decodeSlot));
	ASSERT("decode request stopped by callback", decodeBatcher.slotResult(decodeSlot) && decodeBatcher.slotResult(decodeSlot)->stoppedByCallback);
	ASSERT("decode request emitted one token before stop", decodeBatcher.slotResult(decodeSlot) && decodeBatcher.slotResult(decodeSlot)->tokens.size() == 1u);
	assert_tokens_in_range(decodeBatcher.slotResult(decodeSlot)->tokens, m.vocab);

	printf("    PASSED\n");
}

static void test_training_set_training_config_runlock_contract()
{
	printf("  [C1] TrainingSetTrainingConfigRunLockContract ...\n");

	TrainableDecoderLm m(17u, 6u, 401u);

	glades::TrainingConfig cfg = m.net->getTrainingConfig();
	cfg.transformer.residualDropoutRate = 0.05f;

	SetTrainingConfigWhileRunningCallback cb(cfg);
	const glades::NNetworkStatus st = m.net->train(m.di, &cb);
	ASSERT("training run should succeed", st.ok());
	ASSERT("run-start callback should fire", cb.sawRunStart);
	assert_status_error(cb.setConfigStatus,
	                    glades::NNetworkStatus::INVALID_STATE,
	                    "setTrainingConfig: ",
	                    "network is running");

	printf("    PASSED\n");
}

static void test_training_optimizer_gate_and_eval_contract()
{
	printf("  [C2] TrainingOptimizerGateAndEvalContract ...\n");

	TrainableDecoderLm m(19u, 6u, 402u);

	glades::TrainingConfig cfg = m.net->getTrainingConfig();
	cfg.optimizer.type = glades::OptimizerConfig::SGD_MOMENTUM;
	cfg.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX;
	ASSERT("set training config for optimizer gate", m.net->setTrainingConfig(cfg).ok());

	CaptureEpochMetricsCallback evalCb;
	const glades::NNetworkStatus evalSt = m.net->test(m.di, &evalCb);
	ASSERT("eval path should allow unsupported optimizer", evalSt.ok());
	ASSERT("eval callback should see run start", evalCb.sawRunStart);
	ASSERT("eval callback should capture metrics", evalCb.sawEpoch);
	ASSERT("eval callback should see run end", evalCb.sawRunEnd);

	CaptureEpochMetricsCallback trainCb;
	const glades::NNetworkStatus trainSt = m.net->train(m.di, &trainCb);
	assert_status_error(trainSt,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "SGDHelper_TRANSFORMER: ",
	                    "requires optimizer=ADAMW, ATLAS, VESTA, HELIOS, or SOPHIA_G");
	ASSERT("optimizer-gated train should not emit epoch metrics", !trainCb.sawEpoch);

	printf("    PASSED\n");
}

static void test_training_full_softmax_metrics_contract()
{
	printf("  [C3] TrainingFullSoftmaxMetricsContract ...\n");

	TrainableDecoderLm m(17u, 6u, 403u);

	glades::TrainingConfig cfg = m.net->getTrainingConfig();
	cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
	cfg.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX;
	ASSERT("set training config for full softmax", m.net->setTrainingConfig(cfg).ok());

	CaptureEpochMetricsCallback cb;
	const glades::NNetworkStatus st = m.net->train(m.di, &cb);
	ASSERT("full softmax train should succeed", st.ok());
	ASSERT("full softmax callback should see run start", cb.sawRunStart);
	ASSERT("full softmax callback should capture metrics", cb.sawEpoch);
	ASSERT("full softmax callback should see run end", cb.sawRunEnd);
	ASSERT("full softmax totalError finite and non-negative", cb.last.totalError >= 0.0f && cb.last.totalError < 1.0e6f);
	ASSERT("full softmax perplexity reported", cb.last.perplexity > 0.0f && cb.last.perplexity < 1.0e6f);

	printf("    PASSED\n");
}

static void test_training_sampled_softmax_metrics_contract()
{
	printf("  [C4] TrainingSampledSoftmaxMetricsContract ...\n");

	TrainableDecoderLm m(17u, 6u, 404u);

	glades::TrainingConfig cfg = m.net->getTrainingConfig();
	cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
	cfg.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX;
	cfg.transformer.tokenLmSampledNegatives = 4;
	ASSERT("set training config for sampled softmax", m.net->setTrainingConfig(cfg).ok());

	CaptureEpochMetricsCallback cb;
	const glades::NNetworkStatus st = m.net->train(m.di, &cb);
	ASSERT("sampled softmax train should succeed", st.ok());
	ASSERT("sampled softmax callback should capture metrics", cb.sawEpoch);
	ASSERT("sampled softmax totalError finite and non-negative", cb.last.totalError >= 0.0f && cb.last.totalError < 1.0e6f);
	ASSERT("sampled softmax does not report perplexity", std::fabs(cb.last.perplexity) < 1e-7f);

	printf("    PASSED\n");
}

static void test_training_gradient_checkpointing_parity()
{
	printf("  [C5] TrainingGradientCheckpointingParity ...\n");

	TrainableDecoderLm noCheckpoint(17u, 6u, 405u);
	TrainableDecoderLm withCheckpoint(17u, 6u, 405u);

	glades::TrainingConfig noCheckpointCfg = noCheckpoint.net->getTrainingConfig();
	noCheckpointCfg.optimizer.type = glades::OptimizerConfig::ADAMW;
	noCheckpointCfg.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX;
	noCheckpointCfg.gradientCheckpointing = false;
	ASSERT("set training config without checkpointing", noCheckpoint.net->setTrainingConfig(noCheckpointCfg).ok());

	glades::TrainingConfig checkpointCfg = withCheckpoint.net->getTrainingConfig();
	checkpointCfg.optimizer.type = glades::OptimizerConfig::ADAMW;
	checkpointCfg.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX;
	checkpointCfg.gradientCheckpointing = true;
	ASSERT("set training config with checkpointing", withCheckpoint.net->setTrainingConfig(checkpointCfg).ok());

	CaptureEpochMetricsCallback noCheckpointCb;
	CaptureEpochMetricsCallback checkpointCb;
	ASSERT("train without checkpointing", noCheckpoint.net->train(noCheckpoint.di, &noCheckpointCb).ok());
	ASSERT("train with checkpointing", withCheckpoint.net->train(withCheckpoint.di, &checkpointCb).ok());
	ASSERT("checkpointing parity emits metrics", noCheckpointCb.sawEpoch && checkpointCb.sawEpoch);
	ASSERT("checkpointing totalError parity", std::fabs(noCheckpointCb.last.totalError - checkpointCb.last.totalError) < 1e-4f);

	std::vector<unsigned int> probe;
	probe.push_back(1u);
	probe.push_back(2u);
	probe.push_back(3u);

	std::vector<float> logitsNoCheckpoint;
	std::vector<float> logitsCheckpoint;
	ASSERT("forward after no-checkpoint train", noCheckpoint.net->transformerLmForwardLastLogits(probe, logitsNoCheckpoint).ok());
	ASSERT("forward after checkpoint train", withCheckpoint.net->transformerLmForwardLastLogits(probe, logitsCheckpoint).ok());
	ASSERT("checkpointing logits size parity", logitsNoCheckpoint.size() == logitsCheckpoint.size());

	std::vector<float> traceLogits;
	glades::TransformerForwardTrace trace;
	ASSERT("diagnostic trace forward", noCheckpoint.net->transformerLmForwardLastTrace(
			probe, traceLogits, trace).ok());
	ASSERT("diagnostic trace logits size", traceLogits.size() == logitsNoCheckpoint.size());
	ASSERT("diagnostic trace hidden size", trace.hiddenSize > 0u);
	ASSERT("diagnostic trace has layers", trace.layers > 0u);
	ASSERT("diagnostic trace stage shape",
	       trace.lastHidden.size() == (static_cast<size_t>(trace.layers) + 1u) * trace.hiddenSize);
	for (size_t i = 0u; i < traceLogits.size(); ++i)
		ASSERT("diagnostic trace logits exactly match normal forward", traceLogits[i] == logitsNoCheckpoint[i]);

	double maxAbs = 0.0;
	for (size_t i = 0u; i < logitsNoCheckpoint.size(); ++i)
	{
		const double d = std::fabs(static_cast<double>(logitsNoCheckpoint[i]) - static_cast<double>(logitsCheckpoint[i]));
		if (d > maxAbs) maxAbs = d;
	}
	ASSERT("checkpointing logits parity", maxAbs < 1e-4);

	printf("    PASSED\n");
}

static void test_training_full_softmax_huge_scratch_guard()
{
	printf("  [C6] TrainingFullSoftmaxHugeScratchGuard ...\n");

	TrainableDecoderLm m(16384u, 1400u, 406u, false);

	glades::TrainingConfig cfg = m.net->getTrainingConfig();
	cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
	cfg.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX;
	cfg.transformer.tokenLmAllowHugeFullSoftmax = false;
	ASSERT("set training config for huge softmax guard", m.net->setTrainingConfig(cfg).ok());

	const glades::NNetworkStatus st = m.net->train(m.di);
	assert_status_error(st,
	                    glades::NNetworkStatus::INVALID_ARGUMENT,
	                    "SGDHelper_TRANSFORMER: ",
	                    "token LM full softmax would allocate");

	printf("    PASSED\n");
}

} // namespace

void TransformerVerificationUnitTest()
{
	printf("============================================================\n");
	printf("Transformer Verification Test Suite\n");
	printf("============================================================\n");

	printf("\n--- Group A: Config Snapshots ---\n");
	test_runtime_config_snapshot_validation();
	test_runtime_config_snapshot_boundary_contract();
	test_training_config_validation_contract();
	test_training_config_validation_and_model_snapshot();
	test_model_snapshot_default_and_non_token_contract();

	printf("\n--- Group B: Persistent Batcher ---\n");
	test_batcher_submit_validation_matrix();
	test_batcher_slot_lifecycle_and_reuse();
	test_batcher_callback_stop_and_manual_cancel();

	printf("\n--- Group C: Training Config ---\n");
	test_training_set_training_config_runlock_contract();
	test_training_optimizer_gate_and_eval_contract();
	test_training_full_softmax_metrics_contract();
	test_training_sampled_softmax_metrics_contract();
	test_training_gradient_checkpointing_parity();
	test_training_full_softmax_huge_scratch_guard();

	printf("\n============================================================\n");
}
