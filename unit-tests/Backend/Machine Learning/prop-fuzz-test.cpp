// Property-based + fuzz-like randomized tests (opt-in).
//
// Goals:
// - Exercise hostile / edge-case inputs without hand-writing exhaustive cases.
// - Assert invariants and "no-crash" behavior for high-risk surfaces:
//   - TokenInput table import + view APIs
//   - Transformer generation / KV inference robustness under random prompts/configs
//   - Checkpoint load hardening against manifest corruption
//
// This file uses the existing unit-test framework (ASSERT/G_assert) and is wired to the
// unit-tests runner under the "prop-fuzz" command.

#include "prop-fuzz-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/DataObjects/TokenInput.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include "Backend/Database/GTable.h"
#include "Backend/Database/GList.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

// Property tests should be high-signal: the legacy ASSERT macro prints on success,
// which would spam output and slow tests to a crawl. Use this helper to report
// only failures.
static inline void prop_require(const char* file, int line, const char* msg, bool cond)
{
	if (!cond)
		G_assert(file, line, msg, cond);
}
#define PROP_REQUIRE(msg, cond) prop_require(__FILE__, __LINE__, msg, (cond))

// Deterministic PRNG for property tests (xorshift64*).
static inline uint64_t xorshift64s(uint64_t& state)
{
	if (state == 0ULL)
		state = 0x9e3779b97f4a7c15ULL;
	uint64_t x = state;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	state = x;
	return x * 2685821657736338717ULL;
}

static inline unsigned int urand(uint64_t& s, unsigned int bound)
{
	if (bound == 0u)
		return 0u;
	return static_cast<unsigned int>(xorshift64s(s) % static_cast<uint64_t>(bound));
}

static inline bool rand_bool(uint64_t& s) { return (xorshift64s(s) & 1ULL) != 0ULL; }

static std::string u64_to_string(uint64_t v)
{
	std::ostringstream oss;
	oss << static_cast<unsigned long long>(v);
	return oss.str();
}

static bool stat_is_dir(const std::string& path)
{
	struct stat st;
	if (::stat(path.c_str(), &st) != 0)
		return false;
	return S_ISDIR(st.st_mode) != 0;
}

static bool stat_is_file(const std::string& path)
{
	struct stat st;
	if (::stat(path.c_str(), &st) != 0)
		return false;
	return S_ISREG(st.st_mode) != 0;
}

static std::string join_dir(const std::string& a, const std::string& b)
{
	if (a.empty())
		return b;
	if (b.empty())
		return a;
	if (a[a.size() - 1u] == '/')
		return a + b;
	return a + "/" + b;
}

// Best-effort recursive delete (POSIX). Used only for test cleanup.
static bool remove_tree_recursive(const std::string& path)
{
	DIR* d = ::opendir(path.c_str());
	if (!d)
	{
		if (stat_is_file(path))
			return ::unlink(path.c_str()) == 0;
		return true;
	}
	struct dirent* ent = NULL;
	while ((ent = ::readdir(d)) != NULL)
	{
		const char* name = ent->d_name;
		if (!name)
			continue;
		if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
			continue;
		const std::string child = join_dir(path, std::string(name));
		struct stat st;
		if (::lstat(child.c_str(), &st) != 0)
			continue;
		if (S_ISDIR(st.st_mode))
		{
			(void)remove_tree_recursive(child);
			(void)::rmdir(child.c_str());
		}
		else
		{
			(void)::unlink(child.c_str());
		}
	}
	::closedir(d);
	return ::rmdir(path.c_str()) == 0;
}

static std::string checkpoint_root_dir()
{
	const char* env = ::getenv("GLADES_CHECKPOINT_ROOT");
	if (env && env[0] != '\0')
		return std::string(env);
	return std::string("database/checkpoints");
}

static bool replace_manifest_kv(std::string& manifest, const std::string& key, const std::string& newVal)
{
	const std::string needle = key + "=";
	size_t pos = manifest.find(needle);
	if (pos == std::string::npos)
		return false;
	size_t lineEnd = manifest.find('\n', pos);
	if (lineEnd == std::string::npos)
		lineEnd = manifest.size();
	const size_t startVal = pos + needle.size();
	manifest.replace(startVal, lineEnd - startVal, newVal);
	return true;
}

static bool write_text_file(const std::string& path, const std::string& contents)
{
	std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
	if (!out)
		return false;
	out << contents;
	out.flush();
	return static_cast<bool>(out);
}

static bool read_text_file(const std::string& path, std::string& out)
{
	out.clear();
	std::ifstream in(path.c_str(), std::ios::in);
	if (!in)
		return false;
	std::ostringstream oss;
	oss << in.rdbuf();
	out = oss.str();
	return true;
}

// Build a minimal deterministic decoder-only Transformer LM used by multiple property tests.
struct SmallDecoderLmFixture
{
	glades::TokenInput di;
	glades::NNInfo* info;
	glades::NNetwork* net;
	unsigned int vocab;
	unsigned int padTokenId;

	SmallDecoderLmFixture(unsigned int v)
	    : info(NULL), net(NULL), vocab(v), padTokenId(v - 1u)
	{
		// Build a small prompt table (1 column => single sequence).
		shmea::GVector<shmea::GString> headers;
		headers.push_back("tok");
		shmea::GTable tbl(',', headers);
		for (unsigned int i = 0u; i < 8u; ++i)
		{
			shmea::GList row;
			row.addInt(static_cast<int>((i % (vocab - 2u)) + 1u));
			tbl.addRow(row);
		}

		di.setPadTokenId(static_cast<int>(padTokenId));
		di.import(tbl);
		PROP_REQUIRE("==============PropFuzz: fixture TokenInput import failed==============", di.loadedOk());

		auto in = shmea::make_gpointer<glades::InputLayerInfo>(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);

		std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
		hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
		    /*size*/ 32, // dModel
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
		    /*size*/ 32,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));

		auto out = shmea::make_gpointer<glades::OutputLayerInfo>(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		info = new glades::NNInfo("ut_prop_fuzz_decoder_lm", in, hidden, out);

		net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(1337u);

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
			cfg.transformer.ropeTheta = 10000.0f;
			cfg.transformer.ropeDimOverride = 0;
		}

		// Init tensors via a no-op eval pass.
		const glades::NNetworkStatus st = net->test(&di);
		PROP_REQUIRE("==============PropFuzz: fixture net init test failed==============", st.ok());
	}

	~SmallDecoderLmFixture()
	{
		delete net;
		delete info; // owns in/hidden/out
	}
};

static void prop_tokeninput_table_import(uint64_t seed, unsigned int cases)
{
	uint64_t s = seed;
	for (unsigned int it = 0u; it < cases; ++it)
	{
		glades::TokenInput ti;
		const bool usePad = rand_bool(s);
		const int pad = usePad ? static_cast<int>(urand(s, 64u)) : -1;
		ti.setPadTokenId(pad);

		const unsigned int R = urand(s, 24u);     // allow 0 rows
		const unsigned int C = 1u + urand(s, 12u);

		shmea::GVector<shmea::GString> headers;
		for (unsigned int c = 0u; c < C; ++c)
		{
			std::ostringstream oss;
			oss << "c" << c;
			headers.push_back(oss.str().c_str());
		}
		shmea::GTable tbl(',', headers);

		// Fill rows with mixed cell types, including some intentionally invalid strings.
		for (unsigned int r = 0u; r < R; ++r)
		{
			shmea::GList row;
			for (unsigned int c = 0u; c < C; ++c)
			{
				const unsigned int kind = urand(s, 8u);
				if (kind == 0u)
				{
					// Valid small int token (non-negative).
					row.addInt(static_cast<int>(urand(s, 64u)));
				}
				else if (kind == 1u)
				{
					// Negative (should be rejected).
					row.addInt(-static_cast<int>(1 + urand(s, 3u)));
				}
				else if (kind == 2u)
				{
					// Float that is an integer value.
					row.addFloat(static_cast<float>(static_cast<int>(urand(s, 64u))));
				}
				else if (kind == 3u)
				{
					// String numeric.
					row.addString(u64_to_string(urand(s, 64u)).c_str());
				}
				else if (kind == 4u)
				{
					// Invalid string.
					row.addString("not_a_number");
				}
				else if (kind == 5u)
				{
					// Large value near int32 max to test bounds.
					row.addString("2147483647");
				}
				else if (kind == 6u)
				{
					// Overflowing value (should be rejected).
					row.addString("9999999999999999999999999");
				}
				else
				{
					// Boolean.
					row.addBoolean(rand_bool(s));
				}
			}
			tbl.addRow(row);
		}

		ti.import(tbl);
		const glades::NNetworkStatus st = ti.getLastStatus();
		if (ti.loadedOk())
		{
			PROP_REQUIRE("==============PropTokenInput: loadedOk but status not ok==============", st.ok());
			PROP_REQUIRE("==============PropTokenInput: train size should be >0==============", ti.getTrainSize() > 0u);
			PROP_REQUIRE("==============PropTokenInput: feature count must be 1==============", ti.getFeatureCount() == 1u);

			std::string err;
			PROP_REQUIRE("==============PropTokenInput: validateTrainRowShapes failed==============", ti.validateTrainRowShapes(1u, 1u, &err));
			PROP_REQUIRE("==============PropTokenInput: validateTrainSequences failed==============", ti.validateTrainSequences(&err));

			const unsigned int N = ti.getTrainSize();
			const unsigned int checks = std::min(N, 8u);
			for (unsigned int k = 0u; k < checks; ++k)
			{
				const unsigned int idx = (checks <= 1u) ? 0u : static_cast<unsigned int>((static_cast<unsigned long long>(k) * static_cast<unsigned long long>(N - 1u)) / (checks - 1u));
				int tok = 0, nextTok = 0;
				PROP_REQUIRE("==============PropTokenInput: getTrainTokenId must succeed==============", ti.getTrainTokenId(idx, tok));
				PROP_REQUIRE("==============PropTokenInput: getTrainExpectedTokenId must succeed==============", ti.getTrainExpectedTokenId(idx, nextTok));

				const float* p = NULL;
				unsigned int sz = 0u;
				PROP_REQUIRE("==============PropTokenInput: getTrainRowView must succeed==============", ti.getTrainRowView(idx, p, sz));
				PROP_REQUIRE("==============PropTokenInput: row view size must be 1==============", sz == 1u);
				PROP_REQUIRE("==============PropTokenInput: row view ptr must be non-null==============", p != NULL);
				PROP_REQUIRE("==============PropTokenInput: row view must match token id==============", static_cast<int>(*p) == tok);

				const float* q = NULL;
				unsigned int qsz = 0u;
				PROP_REQUIRE("==============PropTokenInput: getTrainExpectedRowView must succeed==============", ti.getTrainExpectedRowView(idx, q, qsz));
				PROP_REQUIRE("==============PropTokenInput: expected view size must be 1==============", qsz == 1u);
				PROP_REQUIRE("==============PropTokenInput: expected view ptr must be non-null==============", q != NULL);
				PROP_REQUIRE("==============PropTokenInput: expected view must match expected token id==============", static_cast<int>(*q) == nextTok);
			}
		}
		else
		{
			// If not loaded, status must explain why (not silently OK).
			PROP_REQUIRE("==============PropTokenInput: !loadedOk should not report OK==============", !st.ok());
		}
	}
}

static void prop_transformer_generate_and_kv_parity(uint64_t seed, unsigned int cases)
{
	uint64_t s = seed;
	SmallDecoderLmFixture fx(/*vocab*/ 67u);
	const unsigned int vocab = fx.vocab;

	for (unsigned int it = 0u; it < cases; ++it)
	{
		// Random prompt length (>=1).
		const unsigned int T = 1u + urand(s, 16u);
		std::vector<unsigned int> prompt;
		prompt.reserve(T);
		for (unsigned int i = 0u; i < T; ++i)
		{
			// Include pad token sometimes to exercise key masking.
			const unsigned int tok = (urand(s, 10u) == 0u) ? fx.padTokenId : urand(s, vocab);
			prompt.push_back(tok);
		}

		// (A) KV parity for last-logits at each prefix (core invariant).
		{
			glades::NNetwork::TransformerLmSession sess;
			const glades::NNetworkStatus stReset = fx.net->transformerLmSessionReset(sess, T);
			PROP_REQUIRE("==============PropTransformer: session reset failed==============", stReset.ok());

			std::vector<unsigned int> prefix;
			std::vector<float> logitsFull;
			std::vector<float> logitsKv;
			for (unsigned int t = 0u; t < T; ++t)
			{
				prefix.push_back(prompt[t]);
				const glades::NNetworkStatus stF = fx.net->transformerLmForwardLastLogits(prefix, logitsFull);
				const glades::NNetworkStatus stK = fx.net->transformerLmSessionAppend(sess, prompt[t], &logitsKv);
				PROP_REQUIRE("==============PropTransformer: forward failed==============", stF.ok());
				PROP_REQUIRE("==============PropTransformer: kv append failed==============", stK.ok());
				PROP_REQUIRE("==============PropTransformer: logits size mismatch==============", logitsFull.size() == vocab && logitsKv.size() == vocab);

				double maxAbs = 0.0;
				for (unsigned int i = 0u; i < vocab; ++i)
				{
					PROP_REQUIRE("==============PropTransformer: non-finite logits==============", std::isfinite(logitsFull[i]) && std::isfinite(logitsKv[i]));
					const double d = fabs(static_cast<double>(logitsFull[i]) - static_cast<double>(logitsKv[i]));
					if (d > maxAbs) maxAbs = d;
				}
				// Same tolerance used by deterministic gate tests.
				PROP_REQUIRE("==============PropTransformer: KV parity mismatch==============", maxAbs < 1e-3);
			}
		}

		// (B) Random generation configs: must not crash and must obey basic output invariants.
		{
			glades::NNetwork::TransformerGenerateConfig cfg;
			cfg.maxNewTokens = urand(s, 8u);
			cfg.maxSeqLen = 0u; // resolve to promptLen + maxNewTokens
			cfg.temperature = rand_bool(s) ? 0.0f : (0.5f + (static_cast<float>(urand(s, 200u)) / 100.0f)); // 0 => greedy
			cfg.topK = urand(s, 32u); // may be 0 => no top-k
			cfg.topP = 1.0f;
			if (rand_bool(s))
				cfg.topP = static_cast<float>(urand(s, 100u) + 1u) / 100.0f; // (0,1]
			cfg.topPTopKCap = 64u;
			cfg.eosTokenId = -1;
			cfg.stopOnEos = true;
			cfg.includePromptInOutput = rand_bool(s);
			cfg.rngSeedOverride = static_cast<uint64_t>(urand(s, 1000000u));

			glades::NNetwork::TransformerGenerateResult out;
			const glades::NNetworkStatus st = fx.net->transformerLmGenerate(prompt, cfg, out, NULL);
			PROP_REQUIRE("==============PropTransformer: generate status not ok==============", st.ok());

			const unsigned int expectMin = cfg.includePromptInOutput ? static_cast<unsigned int>(prompt.size()) : 0u;
			PROP_REQUIRE("==============PropTransformer: output shorter than expected prefix policy==============", out.tokens.size() >= expectMin);
			if (cfg.includePromptInOutput)
			{
				// Verify prefix matches prompt exactly.
				PROP_REQUIRE("==============PropTransformer: output prompt prefix mismatch==============",
				             std::vector<unsigned int>(out.tokens.begin(), out.tokens.begin() + prompt.size()) == prompt);
			}
			// Generated tokens are always within vocab.
			for (size_t i = 0; i < out.tokens.size(); ++i)
				PROP_REQUIRE("==============PropTransformer: generated token out of range==============", out.tokens[i] < vocab);
		}
	}
}

static void prop_checkpoint_manifest_corruption(uint64_t seed, unsigned int cases)
{
	uint64_t s = seed;

	// Keep this test modest: checkpointing uses filesystem IO.
	SmallDecoderLmFixture fx(/*vocab*/ 41u);
	glades::NNetwork& net = *fx.net;

	for (unsigned int it = 0u; it < cases; ++it)
	{
		// Create a unique checkpoint name (safe characters only).
		const unsigned long long pid = static_cast<unsigned long long>(::getpid());
		const unsigned long long tag = static_cast<unsigned long long>(xorshift64s(s));
		const std::string name = std::string("ut_prop_ckpt_") + u64_to_string(pid) + "_" + u64_to_string(tag);

		// Best-effort cleanup if it already exists.
		const std::string root = checkpoint_root_dir();
		const std::string dir = join_dir(root, name);
		if (stat_is_dir(dir))
			(void)remove_tree_recursive(dir);

		glades::NNetwork::CheckpointConfig cfg;
		cfg.includeOptimizerState = true;
		cfg.maxShardBytes = static_cast<size_t>(16ull * 1024ull * 1024ull); // small shards for test

		const glades::NNetworkStatus stSave = net.saveCheckpoint(name, cfg);
		PROP_REQUIRE("==============PropCheckpoint: saveCheckpoint failed==============", stSave.ok());

		// Load should succeed for the intact checkpoint.
		{
			glades::NNetwork net2(glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			const glades::NNetworkStatus stLoad = net2.loadCheckpoint(name, &fx.di, /*netTypeOverride*/ glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			PROP_REQUIRE("==============PropCheckpoint: loadCheckpoint intact failed==============", stLoad.ok());
		}

		// Corrupt manifest: make tensor.0.count mismatched vs allocated tensor size.
		{
			const std::string manifestPath = join_dir(dir, "manifest.txt");
			std::string manifest;
			PROP_REQUIRE("==============PropCheckpoint: read manifest failed==============", read_text_file(manifestPath, manifest));

			// tensor.0.count exists and is validated against allocated tensor vec size (strict).
			const bool okReplace = replace_manifest_kv(manifest, "tensor.0.count", "999999");
			PROP_REQUIRE("==============PropCheckpoint: manifest missing tensor.0.count==============", okReplace);
			// Keep bytes consistent with count*4 so the loader reaches the strict size check.
			(void)replace_manifest_kv(manifest, "tensor.0.bytes", "3999996"); // 999999*4

			PROP_REQUIRE("==============PropCheckpoint: write corrupted manifest failed==============", write_text_file(manifestPath, manifest));

			glades::NNetwork net3(glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			const glades::NNetworkStatus stLoad = net3.loadCheckpoint(name, &fx.di, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			PROP_REQUIRE("==============PropCheckpoint: corrupted manifest should fail==============", !stLoad.ok());
		}

		// Cleanup.
		(void)remove_tree_recursive(dir);
	}
}

} // namespace

void PropFuzzUnitTest()
{
	printf("============================================================\n");
	printf("Property-based + fuzz-like randomized tests (opt-in)\n");
	printf("============================================================\n");

	// Use a fixed seed for determinism/reproducibility.
	const uint64_t seed = 0xC0FFEEULL;

	printf("-----------------------------------\n");
	printf("Prop: TokenInput table import invariants\n");
	printf("-----------------------------------\n");
	prop_tokeninput_table_import(seed ^ 0xA5A5A5A5ULL, /*cases*/ 200u);

	printf("-----------------------------------\n");
	printf("Prop: Transformer KV parity + generate invariants\n");
	printf("-----------------------------------\n");
	prop_transformer_generate_and_kv_parity(seed ^ 0x5A5A5A5AULL, /*cases*/ 50u);

	printf("-----------------------------------\n");
	printf("Prop: Checkpoint manifest corruption hardening\n");
	printf("-----------------------------------\n");
	// Keep IO-heavy cases low.
	prop_checkpoint_manifest_corruption(seed ^ 0x12345678ULL, /*cases*/ 3u);

	printf("\n============================================================\n");
}

