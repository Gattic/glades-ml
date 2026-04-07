// libFuzzer harness (opt-in build) for Transformer inference/generation APIs.
//
// This file is NOT compiled by default in the normal unit-tests build.
//
// Build example (clang):
//   clang++ -std=c++98 -O1 -g -fsanitize=fuzzer,address,undefined \
//     -I. -I./include -I./Backend -I./services \
//     unit-tests/Backend/Machine\ Learning/fuzz_transformer_infer.cpp \
//     -lglades -lshmea -o fuzz_transformer_infer
//
// Then run:
//   ./fuzz_transformer_infer -runs=100000 corpus_dir/
//
// Notes:
// - This harness avoids filesystem IO.
// - It keeps maxSeqLen small to avoid large allocations.

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "../../../Backend/Machine Learning/DataObjects/TokenInput.h"

#include "Backend/Database/GTable.h"
#include "Backend/Database/GList.h"

#include <stdint.h>
#include <vector>
#include <string>

namespace {
static inline unsigned int read_u8(const uint8_t* data, size_t& off, size_t n)
{
	if (off >= n) return 0u;
	return static_cast<unsigned int>(data[off++]);
}
static inline bool read_bool(const uint8_t* data, size_t& off, size_t n)
{
	return (read_u8(data, off, n) & 1u) != 0u;
}

struct Fixture
{
	glades::TokenInput di;
	glades::NNInfo* info;
	glades::NNetwork* net;
	unsigned int vocab;
	unsigned int padTokenId;

	Fixture()
	    : info(0), net(0), vocab(64u), padTokenId(63u)
	{
		// Create tiny prompt table to initialize tensors.
		shmea::GVector<shmea::GString> headers;
		headers.push_back("tok");
		shmea::GTable tbl(',', headers);
		for (unsigned int i = 0u; i < 8u; ++i)
		{
			shmea::GList row;
			row.addInt(static_cast<int>(1u + (i % 10u)));
			tbl.addRow(row);
		}
		di.setPadTokenId(static_cast<int>(padTokenId));
		di.import(tbl);

		glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(32, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		hidden.push_back(new glades::HiddenLayerInfo(32, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		info = new glades::NNInfo("fuzz_transformer_infer", in, hidden, out);

		net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(123u);
		{
			glades::TrainingConfig& cfg = net->getTrainingConfigMutable();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.nKVHeadsOverride = 2;
			cfg.transformer.dFFOverride = 64;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		}
		(void)net->test(&di);
	}

	~Fixture()
	{
		delete net;
		delete info;
	}
};

static Fixture& fixture()
{
	static Fixture f;
	return f;
}
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
	if (!data || size == 0u)
		return 0;
	size_t off = 0u;

	Fixture& fx = fixture();

	// Prompt tokens.
	const unsigned int promptLen = 1u + (read_u8(data, off, size) % 16u);
	std::vector<unsigned int> prompt;
	prompt.reserve(promptLen);
	for (unsigned int i = 0u; i < promptLen; ++i)
	{
		const unsigned int x = read_u8(data, off, size);
		// Allow pad occasionally.
		const unsigned int tok = ((x % 11u) == 0u) ? fx.padTokenId : (x % fx.vocab);
		prompt.push_back(tok);
	}

	// Full forward last logits.
	std::vector<float> logits;
	(void)fx.net->transformerLmForwardLastLogits(prompt, logits);

	// Generation (bounded).
	glades::NNetwork::TransformerGenerateConfig cfg;
	cfg.maxNewTokens = (read_u8(data, off, size) % 8u);
	cfg.maxSeqLen = promptLen + cfg.maxNewTokens + 2u; // small bound
	cfg.temperature = read_bool(data, off, size) ? 0.0f : 1.0f;
	cfg.topK = (read_u8(data, off, size) % 16u);
	cfg.topP = 1.0f;
	cfg.topPTopKCap = 64u;
	cfg.eosTokenId = -1;
	cfg.stopOnEos = true;
	cfg.includePromptInOutput = read_bool(data, off, size);
	cfg.rngSeedOverride = static_cast<uint64_t>(read_u8(data, off, size));

	glades::NNetwork::TransformerGenerateResult out;
	(void)fx.net->transformerLmGenerate(prompt, cfg, out, NULL);

	std::vector<glades::NNetwork::TransformerServeRequest> requests(1);
	requests[0].promptTokens = prompt;
	requests[0].cfg = cfg;
	glades::NNetwork::TransformerServeBatchResult serveOut;
	(void)fx.net->transformerLmServeGenerateBatch(requests, serveOut, NULL);

	// KV session append.
	glades::NNetwork::TransformerLmSession sess;
	(void)fx.net->transformerLmSessionReset(sess, cfg.maxSeqLen);
	for (size_t i = 0u; i < prompt.size(); ++i)
	{
		std::vector<float> step;
		(void)fx.net->transformerLmSessionAppend(sess, prompt[i], &step);
	}

	glades::NNetwork::TransformerServeBatcher batcher;
	glades::NNetwork::TransformerServeBatcherConfig batcherCfg;
	batcherCfg.maxBatchSize = 2u;
	batcherCfg.maxSeqLen = cfg.maxSeqLen;
	if (fx.net->transformerLmServeBatcherReset(batcher, batcherCfg).ok())
	{
		unsigned int slot = 0u;
		if (fx.net->transformerLmServeBatcherSubmit(batcher, requests[0], slot).ok())
		{
			const unsigned int maxSteps = static_cast<unsigned int>(prompt.size()) + cfg.maxNewTokens + 2u;
			for (unsigned int stepIdx = 0u; stepIdx < maxSteps; ++stepIdx)
			{
				const glades::NNetworkStatus st = fx.net->transformerLmServeBatcherStep(batcher, NULL);
				if (!st.ok())
					break;
				if (slot < batcher.done.size() && batcher.done[slot] != 0u)
					break;
			}
		}
	}

	return 0;
}
