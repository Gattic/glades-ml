// Numerical edge-case tests for the transformer inference/generation/sampling APIs.
//
// These tests target boundary conditions that are handled in code but had no explicit
// test coverage: T=0, T=1, T==maxSeqLen, NaN/Inf logits, extreme activations, etc.

#include "numerical-edge-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/rng.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/Networks/sampling_utils.h"
#include "../../../Backend/Machine Learning/Networks/transformer_types.h"
#include "../../../Backend/Machine Learning/DataObjects/DataInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// In-memory token-id DataInput (same pattern as transformer-improvements-test).
// ---------------------------------------------------------------------------
class TestTokenInput : public glades::DataInput
{
public:
	TestTokenInput() : padId(-1), scratchT(0.0f), scratchN(0.0f), one(1, 0.0f), empty() {}

	void setTrainTokens(const std::vector<unsigned int>& toks, int pad)
	{
		padId = pad;
		trainTok.clear(); trainNext.clear();
		for (size_t i = 0; i < toks.size(); ++i)
			trainTok.push_back(static_cast<int>(toks[i]));
		for (size_t i = 0; i < toks.size(); ++i)
			trainNext.push_back((i + 1u < toks.size()) ? trainTok[i + 1u] : pad);
	}
	void mirrorTrainToTest() { testTok = trainTok; testNext = trainNext; }

	virtual void import(shmea::GString, int = 0) {}
	virtual void import(const shmea::GTable&, int = 0) {}

	virtual shmea::GVector<float> getTrainRow(unsigned int i) const
	{ if (i >= trainTok.size()) return empty; one[0] = static_cast<float>(trainTok[i]); return one; }
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int i) const
	{ if (i >= trainNext.size()) return empty; one[0] = static_cast<float>(trainNext[i]); return one; }
	virtual shmea::GVector<float> getTestRow(unsigned int i) const
	{ if (i >= testTok.size()) return empty; one[0] = static_cast<float>(testTok[i]); return one; }
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int i) const
	{ if (i >= testNext.size()) return empty; one[0] = static_cast<float>(testNext[i]); return one; }

	virtual bool getTrainRowView(unsigned int i, const float*& d, unsigned int& n) const
	{ d = NULL; n = 0u; if (i >= trainTok.size()) return false; scratchT = static_cast<float>(trainTok[i]); d = &scratchT; n = 1u; return true; }
	virtual bool getTrainExpectedRowView(unsigned int i, const float*& d, unsigned int& n) const
	{ d = NULL; n = 0u; if (i >= trainNext.size()) return false; scratchN = static_cast<float>(trainNext[i]); d = &scratchN; n = 1u; return true; }
	virtual bool getTestRowView(unsigned int i, const float*& d, unsigned int& n) const
	{ d = NULL; n = 0u; if (i >= testTok.size()) return false; scratchT = static_cast<float>(testTok[i]); d = &scratchT; n = 1u; return true; }
	virtual bool getTestExpectedRowView(unsigned int i, const float*& d, unsigned int& n) const
	{ d = NULL; n = 0u; if (i >= testNext.size()) return false; scratchN = static_cast<float>(testNext[i]); d = &scratchN; n = 1u; return true; }

	virtual bool getTrainTokenId(unsigned int i, int& out) const { out = 0; if (i >= trainTok.size()) return false; out = trainTok[i]; return true; }
	virtual bool getTrainExpectedTokenId(unsigned int i, int& out) const { out = 0; if (i >= trainNext.size()) return false; out = trainNext[i]; return true; }
	virtual bool getTestTokenId(unsigned int i, int& out) const { out = 0; if (i >= testTok.size()) return false; out = testTok[i]; return true; }
	virtual bool getTestExpectedTokenId(unsigned int i, int& out) const { out = 0; if (i >= testNext.size()) return false; out = testNext[i]; return true; }

	virtual unsigned int getTrainSize() const { return static_cast<unsigned int>(trainTok.size()); }
	virtual unsigned int getTestSize() const { return static_cast<unsigned int>(testTok.size()); }
	virtual unsigned int getFeatureCount() const { return 1u; }
	virtual int getType() const { return TEXT; }

private:
	int padId;
	std::vector<int> trainTok, trainNext, testTok, testNext;
	mutable float scratchT, scratchN;
	mutable shmea::GVector<float> one;
	shmea::GVector<float> empty;
};

// ---------------------------------------------------------------------------
// Fixture: smallest useful transformer (dModel=16, 2 heads, 1 layer, vocab=8).
// ---------------------------------------------------------------------------
static const unsigned int VOCAB = 8u;
static const unsigned int DMODEL = 16u;
static const unsigned int NHEADS = 2u;
static const unsigned int DFF = 32u;
static const unsigned int SEED = 7777u;

struct SmallNet
{
	glades::NNInfo* info;
	glades::NNetwork* net;
	TestTokenInput* di;

	SmallNet() : info(NULL), net(NULL), di(NULL)
	{
		const unsigned int pad = VOCAB - 1u;
		std::vector<unsigned int> toks;
		for (unsigned int i = 0; i < 8u; ++i)
			toks.push_back(i % (VOCAB - 1u));

		di = new TestTokenInput();
		di->setTrainTokens(toks, static_cast<int>(pad));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
			1, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
			static_cast<int>(DMODEL), 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
			static_cast<int>(VOCAB), glades::OutputLayerInfo::CLASSIFICATION);

		info = new glades::NNInfo("ut_numerical_edge", in, hidden, out);
		net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(SEED);
		net->getTerminatorMutable().setEpoch(1);
		net->getTerminatorMutable().setAccuracy(0);

		glades::TrainingConfig& cfg = net->getTrainingConfigMutable();
		cfg.transformer.enableTokenEmbedding = true;
		cfg.transformer.vocabSizeOverride = static_cast<int>(VOCAB);
		cfg.transformer.tieEmbeddings = true;
		cfg.transformer.padTokenId = static_cast<int>(pad);
		cfg.transformer.nHeadsOverride = static_cast<int>(NHEADS);
		cfg.transformer.nKVHeadsOverride = static_cast<int>(NHEADS);
		cfg.transformer.dFFOverride = static_cast<int>(DFF);
		cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
		cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
		cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		cfg.transformer.ropeTheta = 10000.0f;
		cfg.optimizer.type = glades::OptimizerConfig::ADAMW;

		// Initialize weights via test() then train 1 epoch.
		ASSERT("numerical-edge: init failed", net->test(di).ok());
		ASSERT("numerical-edge: train failed", net->train(di).ok());
	}

	~SmallNet()
	{
		delete net;
		delete info;
		delete di;
	}
};

// ============================================================
// Group A: Sampling edge cases (unit tests on raw logit buffers)
// ============================================================

static void test_sampling_all_nan()
{
	printf("  [A1] SamplingAllNaNLogits ...\n");
	const float nan = std::numeric_limits<float>::quiet_NaN();
	float logits[] = {nan, nan, nan, nan};
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 100);
	unsigned int tok = 99u;
	std::vector<unsigned int> idx;
	std::vector<float> ws;

	// Temperature sampling: exp(NaN) = NaN → sum = NaN → return false.
	bool ok = glades::sampling::sample_token_from_logits_ptr(
		logits, rng, 4u, 1.0f, 0u, 1.0f, 256u, tok, idx, ws);
	ASSERT("all-NaN temp sampling should fail", !ok);
	printf("    PASSED\n");
}

static void test_sampling_all_inf()
{
	printf("  [A2] SamplingAllInfLogits ...\n");
	const float inf = std::numeric_limits<float>::infinity();
	float logits[] = {inf, inf, inf, inf};
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 101);
	unsigned int tok = 99u;
	std::vector<unsigned int> idx;
	std::vector<float> ws;

	// exp(Inf - Inf) = exp(NaN) = NaN → sum is NaN → return false.
	bool ok = glades::sampling::sample_token_from_logits_ptr(
		logits, rng, 4u, 1.0f, 0u, 1.0f, 256u, tok, idx, ws);
	ASSERT("all-Inf temp sampling should fail", !ok);
	printf("    PASSED\n");
}

static void test_sampling_mixed_nan()
{
	printf("  [A3] SamplingMixedNaNNormal ...\n");
	const float nan = std::numeric_limits<float>::quiet_NaN();
	float logits[] = {1.0f, nan, 2.0f, nan};
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 102);
	unsigned int tok = 99u;
	std::vector<unsigned int> idx;
	std::vector<float> ws;

	bool ok = glades::sampling::sample_token_from_logits_ptr(
		logits, rng, 4u, 1.0f, 0u, 1.0f, 256u, tok, idx, ws);
	// exp(NaN - max) = NaN → sum includes NaN → !isfinite(sum) → false.
	ASSERT("mixed NaN temp sampling should fail", !ok);
	printf("    PASSED\n");
}

static void test_sampling_all_nan_greedy()
{
	printf("  [A4] SamplingAllNaNGreedy ...\n");
	const float nan = std::numeric_limits<float>::quiet_NaN();
	float logits[] = {nan, nan, nan, nan};
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 103);
	unsigned int tok = 99u;
	std::vector<unsigned int> idx;
	std::vector<float> ws;

	// Greedy: scans for max. NaN comparisons always false → bestIdx stays 0.
	bool ok = glades::sampling::sample_token_from_logits_ptr(
		logits, rng, 4u, 0.0f, 0u, 1.0f, 256u, tok, idx, ws);
	// Greedy path returns true (no isfinite check) — document this behavior.
	ASSERT("all-NaN greedy should return true (no isfinite guard)", ok);
	printf("    greedy tok=%u (expected: 0, NaN comparison gives first index)\n", tok);
	printf("    PASSED\n");
}

static void test_sampling_all_same()
{
	printf("  [A5] SamplingAllSameLogits ...\n");
	float logits[] = {5.0f, 5.0f, 5.0f, 5.0f};
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 104);
	unsigned int tok = 99u;
	std::vector<unsigned int> idx;
	std::vector<float> ws;

	bool ok = glades::sampling::sample_token_from_logits_ptr(
		logits, rng, 4u, 1.0f, 0u, 1.0f, 256u, tok, idx, ws);
	ASSERT("all-same logits should succeed", ok);
	ASSERT("token in range", tok < 4u);
	printf("    PASSED\n");
}

static void test_sampling_extreme_positive()
{
	printf("  [A6] SamplingExtremePositiveLogits ...\n");
	float logits[] = {1e30f, 1e30f - 1.0f, 0.0f, 0.0f};
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 105);
	unsigned int tok = 99u;
	std::vector<unsigned int> idx;
	std::vector<float> ws;

	bool ok = glades::sampling::sample_token_from_logits_ptr(
		logits, rng, 4u, 1.0f, 0u, 1.0f, 256u, tok, idx, ws);
	ASSERT("extreme positive logits should succeed (max-shift stable)", ok);
	ASSERT("token in range", tok < 4u);
	printf("    PASSED\n");
}

static void test_sampling_extreme_negative()
{
	printf("  [A7] SamplingExtremeNegativeLogits ...\n");
	float logits[] = {-1e30f, -1e30f, -1e30f, -1e30f};
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 106);
	unsigned int tok = 99u;
	std::vector<unsigned int> idx;
	std::vector<float> ws;

	// exp(0) = 1 after max-shift, so sum > 0 → should succeed.
	bool ok = glades::sampling::sample_token_from_logits_ptr(
		logits, rng, 4u, 1.0f, 0u, 1.0f, 256u, tok, idx, ws);
	ASSERT("extreme negative logits should succeed", ok);
	ASSERT("token in range", tok < 4u);
	printf("    PASSED\n");
}

static void test_sampling_vocab1()
{
	printf("  [A8] SamplingVocab1 ...\n");
	float logits[] = {0.0f};
	glades::rng::Engine rng;
	glades::rng::seed_engine(rng, 107);
	unsigned int tok = 99u;
	std::vector<unsigned int> idx;
	std::vector<float> ws;

	bool ok = glades::sampling::sample_token_from_logits_ptr(
		logits, rng, 1u, 1.0f, 0u, 1.0f, 256u, tok, idx, ws);
	ASSERT("vocab=1 should succeed", ok);
	ASSERT("vocab=1 returns token 0", tok == 0u);
	printf("    PASSED\n");
}

// ============================================================
// Group B: Generate API boundary conditions
// ============================================================

static void test_generate_empty_prompt(const glades::NNetwork& net)
{
	printf("  [B1] GenerateEmptyPrompt ...\n");
	std::vector<unsigned int> prompt; // empty
	glades::NNetwork::TransformerGenerateConfig cfg;
	cfg.maxNewTokens = 1u;
	cfg.temperature = 0.0f; // greedy
	glades::NNetwork::TransformerGenerateResult out;

	const glades::NNetworkStatus st = net.transformerLmGenerate(prompt, cfg, out, NULL);
	ASSERT("empty prompt should return error", !st.ok());
	printf("    status: %s\n", st.message.c_str());
	printf("    PASSED\n");
}

static void test_generate_t1_prompt(const glades::NNetwork& net)
{
	printf("  [B2] GenerateT1Prompt ...\n");
	std::vector<unsigned int> prompt;
	prompt.push_back(0u); // single token
	glades::NNetwork::TransformerGenerateConfig cfg;
	cfg.maxNewTokens = 2u;
	cfg.temperature = 0.0f; // greedy
	cfg.rngSeedOverride = 1234ULL;
	glades::NNetwork::TransformerGenerateResult out;

	const glades::NNetworkStatus st = net.transformerLmGenerate(prompt, cfg, out, NULL);
	ASSERT("T=1 prompt generate should succeed", st.ok());
	ASSERT("should have generated tokens", !out.tokens.empty());
	for (size_t i = 0; i < out.tokens.size(); ++i)
		ASSERT("generated token in range", out.tokens[i] < VOCAB);
	printf("    generated %u tokens\n", static_cast<unsigned int>(out.tokens.size()));
	printf("    PASSED\n");
}

static void test_generate_t1_no_new(const glades::NNetwork& net)
{
	printf("  [B3] GenerateT1PromptNoNew ...\n");
	std::vector<unsigned int> prompt;
	prompt.push_back(0u);
	glades::NNetwork::TransformerGenerateConfig cfg;
	cfg.maxNewTokens = 0u; // generate nothing
	cfg.temperature = 0.0f;
	glades::NNetwork::TransformerGenerateResult out;

	const glades::NNetworkStatus st = net.transformerLmGenerate(prompt, cfg, out, NULL);
	ASSERT("maxNewTokens=0 should succeed", st.ok());
	ASSERT("should generate 0 tokens", out.tokens.empty());
	printf("    PASSED\n");
}

static void test_generate_token_oob(const glades::NNetwork& net)
{
	printf("  [B4] GenerateTokenIdEqVocab ...\n");
	std::vector<unsigned int> prompt;
	prompt.push_back(VOCAB); // == vocabSize, off-by-one
	glades::NNetwork::TransformerGenerateConfig cfg;
	cfg.maxNewTokens = 1u;
	cfg.temperature = 0.0f;
	glades::NNetwork::TransformerGenerateResult out;

	const glades::NNetworkStatus st = net.transformerLmGenerate(prompt, cfg, out, NULL);
	ASSERT("token ID == vocabSize should fail", !st.ok());
	printf("    status: %s\n", st.message.c_str());
	printf("    PASSED\n");
}

static void test_generate_max_valid_token(const glades::NNetwork& net)
{
	printf("  [B5] GenerateTokenIdMaxValid ...\n");
	std::vector<unsigned int> prompt;
	prompt.push_back(VOCAB - 1u); // largest valid token
	glades::NNetwork::TransformerGenerateConfig cfg;
	cfg.maxNewTokens = 1u;
	cfg.temperature = 0.0f;
	cfg.rngSeedOverride = 5678ULL;
	glades::NNetwork::TransformerGenerateResult out;

	const glades::NNetworkStatus st = net.transformerLmGenerate(prompt, cfg, out, NULL);
	ASSERT("max valid token ID should succeed", st.ok());
	printf("    PASSED\n");
}

// ============================================================
// Group C: Session API boundary conditions
// ============================================================

static void test_session_maxseqlen_zero(const glades::NNetwork& net)
{
	printf("  [C1] SessionResetMaxSeqLen0 ...\n");
	glades::NNetwork::TransformerLmSession session;
	const glades::NNetworkStatus st = net.transformerLmSessionReset(session, 0u);
	ASSERT("maxSeqLen=0 should fail", !st.ok());
	printf("    status: %s\n", st.message.c_str());
	printf("    PASSED\n");
}

static void test_session_append_beyond_capacity(const glades::NNetwork& net)
{
	printf("  [C2] SessionAppendBeyondCapacity ...\n");
	glades::NNetwork::TransformerLmSession session;
	const unsigned int maxLen = 2u;
	glades::NNetworkStatus st = net.transformerLmSessionReset(session, maxLen);
	ASSERT("session reset should succeed", st.ok());

	// Append exactly maxLen tokens — should succeed.
	std::vector<float> logits;
	st = net.transformerLmSessionAppend(session, 0u, &logits);
	ASSERT("append 1 should succeed", st.ok());
	st = net.transformerLmSessionAppend(session, 1u, &logits);
	ASSERT("append 2 should succeed", st.ok());

	// Third append should fail (beyond capacity).
	st = net.transformerLmSessionAppend(session, 2u, &logits);
	ASSERT("append beyond capacity should fail", !st.ok());
	printf("    status: %s\n", st.message.c_str());
	printf("    PASSED\n");
}

static void test_session_append_at_exact_capacity(const glades::NNetwork& net)
{
	printf("  [C3] SessionAppendAtExactCapacity ...\n");
	glades::NNetwork::TransformerLmSession session;
	const unsigned int maxLen = 3u;
	glades::NNetworkStatus st = net.transformerLmSessionReset(session, maxLen);
	ASSERT("session reset should succeed", st.ok());

	std::vector<float> logits;
	for (unsigned int i = 0; i < maxLen; ++i)
	{
		st = net.transformerLmSessionAppend(session, i % VOCAB, &logits);
		ASSERT("append within capacity should succeed", st.ok());
	}

	// Verify logits are valid.
	ASSERT("logits should have vocab entries", logits.size() == VOCAB);
	bool allFinite = true;
	for (size_t i = 0; i < logits.size(); ++i)
		if (!std::isfinite(logits[i])) allFinite = false;
	ASSERT("logits should all be finite", allFinite);
	printf("    PASSED\n");
}

static void test_session_append_token_oob(const glades::NNetwork& net)
{
	printf("  [C4] SessionAppendTokenOutOfRange ...\n");
	glades::NNetwork::TransformerLmSession session;
	glades::NNetworkStatus st = net.transformerLmSessionReset(session, 4u);
	ASSERT("session reset should succeed", st.ok());

	st = net.transformerLmSessionAppend(session, VOCAB, NULL);
	ASSERT("token == vocabSize should fail", !st.ok());
	printf("    status: %s\n", st.message.c_str());
	printf("    PASSED\n");
}

static void test_session_reset_clears_state(const glades::NNetwork& net)
{
	printf("  [C5] SessionAppendAfterReset ...\n");
	glades::NNetwork::TransformerLmSession session;
	glades::NNetworkStatus st = net.transformerLmSessionReset(session, 4u);
	ASSERT("first reset ok", st.ok());

	// Append one token.
	st = net.transformerLmSessionAppend(session, 0u, NULL);
	ASSERT("first append ok", st.ok());

	// Reset again.
	st = net.transformerLmSessionReset(session, 4u);
	ASSERT("second reset ok", st.ok());

	// Append should work at position 0 again.
	std::vector<float> logits;
	st = net.transformerLmSessionAppend(session, 0u, &logits);
	ASSERT("append after reset should succeed at pos 0", st.ok());
	ASSERT("logits after reset should have vocab entries", logits.size() == VOCAB);
	printf("    PASSED\n");
}

// ============================================================
// Group D: ForwardLastLogits edge cases
// ============================================================

static void test_forward_empty(const glades::NNetwork& net)
{
	printf("  [D1] ForwardLastLogitsEmpty ...\n");
	std::vector<unsigned int> tokenIds; // empty
	std::vector<float> logits;

	const glades::NNetworkStatus st = net.transformerLmForwardLastLogits(tokenIds, logits);
	ASSERT("empty tokenIds should fail", !st.ok());
	printf("    status: %s\n", st.message.c_str());
	printf("    PASSED\n");
}

static void test_forward_t1(const glades::NNetwork& net)
{
	printf("  [D2] ForwardLastLogitsT1 ...\n");
	std::vector<unsigned int> tokenIds;
	tokenIds.push_back(0u); // single token
	std::vector<float> logits;

	const glades::NNetworkStatus st = net.transformerLmForwardLastLogits(tokenIds, logits);
	ASSERT("T=1 forward should succeed", st.ok());
	ASSERT("logits size should be vocab", logits.size() == VOCAB);

	bool allFinite = true;
	for (size_t i = 0; i < logits.size(); ++i)
		if (!std::isfinite(logits[i])) allFinite = false;
	ASSERT("T=1 logits should all be finite", allFinite);
	printf("    PASSED\n");
}

static void test_forward_token_oob(const glades::NNetwork& net)
{
	printf("  [D3] ForwardLastLogitsTokenOOB ...\n");
	std::vector<unsigned int> tokenIds;
	tokenIds.push_back(VOCAB); // off-by-one
	std::vector<float> logits;

	const glades::NNetworkStatus st = net.transformerLmForwardLastLogits(tokenIds, logits);
	ASSERT("OOB token should fail", !st.ok());
	printf("    status: %s\n", st.message.c_str());
	printf("    PASSED\n");
}

static void test_forward_all_same_token(const glades::NNetwork& net)
{
	printf("  [D4] ForwardLastLogitsAllSameToken ...\n");
	std::vector<unsigned int> tokenIds;
	for (unsigned int i = 0; i < 4u; ++i)
		tokenIds.push_back(0u);
	std::vector<float> logits;

	const glades::NNetworkStatus st = net.transformerLmForwardLastLogits(tokenIds, logits);
	ASSERT("all-same token forward should succeed", st.ok());
	ASSERT("logits size should be vocab", logits.size() == VOCAB);

	bool allFinite = true;
	for (size_t i = 0; i < logits.size(); ++i)
		if (!std::isfinite(logits[i])) allFinite = false;
	ASSERT("all-same token logits should be finite", allFinite);
	printf("    PASSED\n");
}

// ============================================================
// Group E: KV-cache parity and session logits match forward
// ============================================================

static void test_session_logits_match_forward(const glades::NNetwork& net)
{
	printf("  [E1] SessionLogitsMatchForwardLastLogits ...\n");
	const unsigned int T = 4u;
	std::vector<unsigned int> tokenIds;
	for (unsigned int i = 0; i < T; ++i)
		tokenIds.push_back(i % (VOCAB - 1u));

	// Full forward.
	std::vector<float> logitsFull;
	glades::NNetworkStatus st = net.transformerLmForwardLastLogits(tokenIds, logitsFull);
	ASSERT("forward should succeed", st.ok());

	// Incremental session.
	glades::NNetwork::TransformerLmSession session;
	st = net.transformerLmSessionReset(session, T);
	ASSERT("session reset ok", st.ok());

	std::vector<float> logitsSession;
	for (unsigned int i = 0; i < T; ++i)
	{
		st = net.transformerLmSessionAppend(session, tokenIds[i], (i == T - 1u) ? &logitsSession : NULL);
		ASSERT("session append ok", st.ok());
	}

	ASSERT("both logits should have same size", logitsFull.size() == logitsSession.size());

	float maxDiff = 0.0f;
	for (size_t i = 0; i < logitsFull.size(); ++i)
	{
		float d = std::fabs(logitsFull[i] - logitsSession[i]);
		if (d > maxDiff) maxDiff = d;
	}
	printf("    max diff between full and session logits: %e\n", maxDiff);
	ASSERT("session logits should match forward within 1e-3", maxDiff < 1e-3f);
	printf("    PASSED\n");
}

} // anonymous namespace

// ============================================================
// Main entry
// ============================================================

void NumericalEdgeUnitTest()
{
	printf("============================================================\n");
	printf("Numerical Edge Case Test Suite\n");
	printf("============================================================\n");

	// --- Group A: Sampling edge cases (no network needed) ---
	printf("--- Group A: Sampling Edge Cases ---\n");
	test_sampling_all_nan();
	test_sampling_all_inf();
	test_sampling_mixed_nan();
	test_sampling_all_nan_greedy();
	test_sampling_all_same();
	test_sampling_extreme_positive();
	test_sampling_extreme_negative();
	test_sampling_vocab1();

	// --- Build a small network for Groups B-E ---
	printf("--- Building small transformer for API tests ---\n");
	SmallNet m;

	// --- Group B: Generate API boundaries ---
	printf("--- Group B: Generate API Boundary Conditions ---\n");
	test_generate_empty_prompt(*m.net);
	test_generate_t1_prompt(*m.net);
	test_generate_t1_no_new(*m.net);
	test_generate_token_oob(*m.net);
	test_generate_max_valid_token(*m.net);

	// --- Group C: Session API boundaries ---
	printf("--- Group C: Session API Boundary Conditions ---\n");
	test_session_maxseqlen_zero(*m.net);
	test_session_append_beyond_capacity(*m.net);
	test_session_append_at_exact_capacity(*m.net);
	test_session_append_token_oob(*m.net);
	test_session_reset_clears_state(*m.net);

	// --- Group D: ForwardLastLogits edge cases ---
	printf("--- Group D: ForwardLastLogits Edge Cases ---\n");
	test_forward_empty(*m.net);
	test_forward_t1(*m.net);
	test_forward_token_oob(*m.net);
	test_forward_all_same_token(*m.net);

	// --- Group E: KV-cache parity ---
	printf("--- Group E: Session/Forward Parity ---\n");
	test_session_logits_match_forward(*m.net);

	printf("============================================================\n");
	printf("All Numerical Edge Case Tests Passed\n");
	printf("============================================================\n");
}
