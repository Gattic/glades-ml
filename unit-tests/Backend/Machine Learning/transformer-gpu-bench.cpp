#include "transformer-gpu-bench.h"
#include "test_token_id_input_fixture.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_device.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <string>
#include <vector>

namespace {

static int64_t now_ms()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return static_cast<int64_t>(tv.tv_sec) * 1000LL + static_cast<int64_t>(tv.tv_usec) / 1000LL;
}

static bool streq(const char* a, const char* b)
{
	return (a && b && strcmp(a, b) == 0);
}

static unsigned int mix_u32(unsigned int x)
{
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

struct CaptureMetricsCallbacks : public glades::ITrainingCallbacks
{
	glades::NNetworkEpochMetrics last;
	bool saw;

	CaptureMetricsCallbacks() : last(), saw(false) {}

	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}
};

struct BenchConfig
{
	enum Mode
	{
		MODE_ALL = 0,
		MODE_TRAIN = 1,
		MODE_INFER = 2
	};

	unsigned int vocab;
	unsigned int dModel;
	unsigned int dFF;
	unsigned int layers;
	unsigned int heads;
	unsigned int kvHeads;
	unsigned int seqLen;
	unsigned int trainSeqs;
	unsigned int inferPromptLen;
	unsigned int inferSteps;
	unsigned int repeats;
	unsigned int epochs;
	unsigned int sampledNegatives;
	unsigned int seed;
	float learningRate;
	Mode mode;

	BenchConfig()
	    : vocab(257u),
	      dModel(128u),
	      dFF(512u),
	      layers(4u),
	      heads(8u),
	      kvHeads(8u),
	      seqLen(64u),
	      trainSeqs(8u),
	      inferPromptLen(64u),
	      inferSteps(64u),
	      repeats(3u),
	      epochs(1u),
	      sampledNegatives(0u),
	      seed(1337u),
	      learningRate(0.001f),
	      mode(MODE_ALL)
	{
	}
};

struct DatasetBundle
{
	InMemoryTokenIdInput di;
	std::vector<unsigned int> prompt;
	unsigned int padTokenId;
	unsigned long long trainTokensPerEpoch;

	DatasetBundle()
	    : di(),
	      prompt(),
	      padTokenId(0u),
	      trainTokensPerEpoch(0ULL)
	{
	}
};

struct TrainBenchResult
{
	long long wallMs;
	double tokensPerSec;
	float finalLoss;
	float finalPerplexity;
	glades::NNetwork::TransformerGpuPerfBreakdown gpuPerf;
	bool ok;
	std::string err;

	TrainBenchResult()
	    : wallMs(0LL),
	      tokensPerSec(0.0),
	      finalLoss(0.0f),
	      finalPerplexity(0.0f),
	      gpuPerf(),
	      ok(true),
	      err()
	{
	}
};

struct InferBenchResult
{
	long long wallMs;
	double tokensPerSec;
	unsigned int appendedTokens;
	glades::NNetwork::TransformerGpuPerfBreakdown gpuPerf;
	bool ok;
	std::string err;

	InferBenchResult()
	    : wallMs(0LL),
	      tokensPerSec(0.0),
	      appendedTokens(0u),
	      gpuPerf(),
	      ok(true),
	      err()
	{
	}
};

class NetworkOwner
{
public:
	glades::NNInfo* info;
	glades::NNetwork* net;

	NetworkOwner()
	    : info(NULL),
	      net(NULL)
	{
	}

	~NetworkOwner()
	{
		delete net;
		delete info;
	}

private:
	NetworkOwner(const NetworkOwner&);
	NetworkOwner& operator=(const NetworkOwner&);
};

static void print_usage()
{
	printf("Usage: glades-unit-tests transformer-gpu-bench [options]\n");
	printf("Options:\n");
	printf("  --mode all|train|infer     Benchmark train path, infer path, or both (default: all)\n");
	printf("  --vocab N                  Vocabulary size including pad token (default: 257)\n");
	printf("  --dmodel N                 Transformer model width (default: 128)\n");
	printf("  --dff N                    FFN hidden width (default: 512)\n");
	printf("  --layers N                 Number of decoder layers (default: 4)\n");
	printf("  --heads N                  Number of attention heads (default: 8)\n");
	printf("  --kv-heads N               Number of KV heads (default: 8)\n");
	printf("  --seq-len N                Tokens per training sequence excluding pad sentinel (default: 64)\n");
	printf("  --train-seqs N             Number of training sequences per epoch (default: 8)\n");
	printf("  --infer-prompt N           Prompt tokens for KV append microbench (default: 64)\n");
	printf("  --infer-steps N            Additional incremental decode steps (default: 64)\n");
	printf("  --epochs N                 Training epochs per repeat (default: 1)\n");
	printf("  --repeats N                Benchmark repeats per mode (default: 3)\n");
	printf("  --lr X                     Learning rate (default: 0.001)\n");
	printf("  --sampled-negatives N      Enable sampled softmax with N negatives (default: 0 => full softmax)\n");
	printf("  --seed N                   Base seed (default: 1337)\n");
	printf("  --help                     Show this message\n");
}

static bool parse_uint_arg(const char* text, unsigned int& outValue)
{
	if (!text || !*text)
		return false;
	char* end = NULL;
	const unsigned long v = strtoul(text, &end, 10);
	if (!end || *end != '\0')
		return false;
	outValue = static_cast<unsigned int>(v);
	return true;
}

static bool parse_float_arg(const char* text, float& outValue)
{
	if (!text || !*text)
		return false;
	char* end = NULL;
	const double v = strtod(text, &end);
	if (!end || *end != '\0')
		return false;
	outValue = static_cast<float>(v);
	return true;
}

static bool parse_args(int argc, char* argv[], BenchConfig& cfg, std::string& err)
{
	err.clear();
	for (int i = 2; i < argc; ++i)
	{
		if (streq(argv[i], "--help"))
			return false;
		else if (streq(argv[i], "--mode") && i + 1 < argc)
		{
			++i;
			if (streq(argv[i], "all"))
				cfg.mode = BenchConfig::MODE_ALL;
			else if (streq(argv[i], "train"))
				cfg.mode = BenchConfig::MODE_TRAIN;
			else if (streq(argv[i], "infer"))
				cfg.mode = BenchConfig::MODE_INFER;
			else
			{
				err = "invalid --mode value";
				return false;
			}
		}
		else if (streq(argv[i], "--vocab") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.vocab)) { err = "invalid --vocab"; return false; }
		}
		else if (streq(argv[i], "--dmodel") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.dModel)) { err = "invalid --dmodel"; return false; }
		}
		else if (streq(argv[i], "--dff") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.dFF)) { err = "invalid --dff"; return false; }
		}
		else if (streq(argv[i], "--layers") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.layers)) { err = "invalid --layers"; return false; }
		}
		else if (streq(argv[i], "--heads") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.heads)) { err = "invalid --heads"; return false; }
		}
		else if (streq(argv[i], "--kv-heads") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.kvHeads)) { err = "invalid --kv-heads"; return false; }
		}
		else if (streq(argv[i], "--seq-len") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.seqLen)) { err = "invalid --seq-len"; return false; }
		}
		else if (streq(argv[i], "--train-seqs") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.trainSeqs)) { err = "invalid --train-seqs"; return false; }
		}
		else if (streq(argv[i], "--infer-prompt") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.inferPromptLen)) { err = "invalid --infer-prompt"; return false; }
		}
		else if (streq(argv[i], "--infer-steps") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.inferSteps)) { err = "invalid --infer-steps"; return false; }
		}
		else if (streq(argv[i], "--epochs") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.epochs)) { err = "invalid --epochs"; return false; }
		}
		else if (streq(argv[i], "--repeats") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.repeats)) { err = "invalid --repeats"; return false; }
		}
		else if (streq(argv[i], "--sampled-negatives") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.sampledNegatives)) { err = "invalid --sampled-negatives"; return false; }
		}
		else if (streq(argv[i], "--seed") && i + 1 < argc)
		{
			if (!parse_uint_arg(argv[++i], cfg.seed)) { err = "invalid --seed"; return false; }
		}
		else if (streq(argv[i], "--lr") && i + 1 < argc)
		{
			if (!parse_float_arg(argv[++i], cfg.learningRate)) { err = "invalid --lr"; return false; }
		}
		else
		{
			err = std::string("unknown or incomplete option: ") + argv[i];
			return false;
		}
	}

	if (cfg.vocab < 4u)
	{
		err = "vocab must be at least 4";
		return false;
	}
	if (cfg.layers == 0u || cfg.dModel == 0u || cfg.dFF == 0u || cfg.seqLen == 0u || cfg.trainSeqs == 0u || cfg.repeats == 0u)
	{
		err = "layers, dmodel, dff, seq-len, train-seqs, and repeats must be non-zero";
		return false;
	}
	if (cfg.heads == 0u || cfg.kvHeads == 0u || (cfg.dModel % cfg.heads) != 0u || (cfg.heads % cfg.kvHeads) != 0u)
	{
		err = "invalid head configuration";
		return false;
	}
	if (cfg.inferPromptLen == 0u && cfg.mode != BenchConfig::MODE_TRAIN)
	{
		err = "infer prompt length must be non-zero";
		return false;
	}
	if (cfg.learningRate <= 0.0f)
	{
		err = "learning rate must be positive";
		return false;
	}
	return true;
}

static void build_dataset(const BenchConfig& cfg, DatasetBundle& out)
{
	out = DatasetBundle();
	out.padTokenId = cfg.vocab - 1u;

	std::vector<unsigned int> tokens;
	std::vector<glades::DataInput::SequenceSpan> spans;
	tokens.reserve(static_cast<size_t>(cfg.trainSeqs) * static_cast<size_t>(cfg.seqLen + 1u));
	spans.reserve(cfg.trainSeqs);

	for (unsigned int seq = 0u; seq < cfg.trainSeqs; ++seq)
	{
		const unsigned int start = static_cast<unsigned int>(tokens.size());
		for (unsigned int t = 0u; t < cfg.seqLen; ++t)
		{
			const unsigned int raw = mix_u32(cfg.seed + seq * 131u + t * 977u);
			tokens.push_back(raw % (cfg.vocab - 1u));
		}
		spans.push_back(glades::DataInput::SequenceSpan(start, cfg.seqLen));
		tokens.push_back(out.padTokenId);
	}

	out.di.setTrainTokens(tokens, static_cast<int>(out.padTokenId));
	out.di.setTestTokens(tokens, static_cast<int>(out.padTokenId));
	(void)out.di.setTrainSequences(spans);
	(void)out.di.setTestSequences(spans);
	out.trainTokensPerEpoch = static_cast<unsigned long long>(cfg.trainSeqs) * static_cast<unsigned long long>(cfg.seqLen);

	out.prompt.reserve(cfg.inferPromptLen);
	for (unsigned int i = 0u; i < cfg.inferPromptLen; ++i)
	{
		const unsigned int raw = mix_u32(cfg.seed + 17u * (i + 1u));
		out.prompt.push_back(raw % (cfg.vocab - 1u));
	}
}

static glades::NNInfo* build_info(const BenchConfig& cfg, const char* name)
{
	auto in = shmea::make_gpointer<glades::InputLayerInfo>(
	    1,
	    cfg.learningRate,
	    0.0f,
	    0.0f,
	    0.0f,
	    0.0f,
	    glades::GMath::LINEAR,
	    1.0f);

	std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
	for (unsigned int i = 0u; i < cfg.layers; ++i)
	{
		hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
		    static_cast<int>(cfg.dModel),
		    cfg.learningRate,
		    0.0f,
		    0.0f,
		    0.0f,
		    0.0f,
		    glades::GMath::LINEAR,
		    1.0f));
	}

	auto out = shmea::make_gpointer<glades::OutputLayerInfo>(static_cast<int>(cfg.vocab), glades::OutputLayerInfo::CLASSIFICATION);
	return new glades::NNInfo(name, in, hidden, out);
}

static bool configure_network(glades::NNetwork& net, const BenchConfig& cfg, unsigned int padTokenId, std::string& err)
{
	err.clear();
	net.getTerminatorMutable().setEpoch(static_cast<int>(cfg.epochs));
	net.getTerminatorMutable().setAccuracy(0.0f);

	glades::TrainingConfig tc = net.getTrainingConfig();
	tc.transformer.enableTokenEmbedding = true;
	tc.transformer.vocabSizeOverride = static_cast<int>(cfg.vocab);
	tc.transformer.tieEmbeddings = true;
	tc.transformer.padTokenId = static_cast<int>(padTokenId);
	tc.transformer.nHeadsOverride = static_cast<int>(cfg.heads);
	tc.transformer.nKVHeadsOverride = static_cast<int>(cfg.kvHeads);
	tc.transformer.dFFOverride = static_cast<int>(cfg.dFF);
	tc.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
	tc.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
	tc.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
	tc.transformer.ropeTheta = 10000.0f;
	tc.optimizer.type = glades::OptimizerConfig::ADAMW;
	if (cfg.sampledNegatives > 0u)
	{
		tc.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX;
		tc.transformer.tokenLmSampledNegatives = static_cast<int>(cfg.sampledNegatives);
	}

	const glades::NNetworkStatus stCfg = net.setTrainingConfig(tc);
	if (!stCfg.ok())
	{
		err = stCfg.message;
		return false;
	}

	glades::NNetwork::TransformerMetricsConfig metricsCfg = net.getTransformerMetricsConfig();
	metricsCfg.enable = true;
	metricsCfg.enableGpuPerf = true;
	metricsCfg.logGpuTrainSummary = false;
	metricsCfg.logGpuInferSummary = false;
	metricsCfg.logPerKvAppend = false;
	net.setTransformerMetricsConfig(metricsCfg);
	return true;
}

static bool make_network(const BenchConfig& cfg, unsigned int runSeed, unsigned int padTokenId, const char* name, NetworkOwner& out, std::string& err)
{
	out.info = build_info(cfg, name);
	out.net = new glades::NNetwork(out.info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
	out.net->setSeed(static_cast<uint64_t>(runSeed));
	if (!configure_network(*out.net, cfg, padTokenId, err))
		return false;
	return true;
}

static bool warmup_model(glades::NNetwork& net, glades::DataInput& di, std::string& err)
{
	const glades::NNetworkStatus st = net.test(&di);
	if (!st.ok())
	{
		err = st.message;
		return false;
	}
	return true;
}

static TrainBenchResult run_train_bench(const BenchConfig& cfg, const DatasetBundle& data, unsigned int runSeed)
{
	TrainBenchResult result;
	NetworkOwner owner;
	if (!make_network(cfg, runSeed, data.padTokenId, "bench_transformer_gpu_train", owner, result.err))
	{
		result.ok = false;
		return result;
	}
	if (!warmup_model(*owner.net, const_cast<InMemoryTokenIdInput&>(data.di), result.err))
	{
		result.ok = false;
		return result;
	}

	CaptureMetricsCallbacks cb;
	const int64_t t0 = now_ms();
	const glades::NNetworkStatus st = owner.net->train(const_cast<InMemoryTokenIdInput*>(&data.di), &cb);
	const int64_t t1 = now_ms();

	result.wallMs = static_cast<long long>(t1 - t0);
	if (!st.ok())
	{
		result.ok = false;
		result.err = st.message;
		return result;
	}

	result.gpuPerf = owner.net->getLastTransformerTrainGpuPerf();
	if (cb.saw)
	{
		result.finalLoss = cb.last.totalError;
		result.finalPerplexity = cb.last.perplexity;
	}

	const double seconds = static_cast<double>(result.wallMs) / 1000.0;
	const double tokens = static_cast<double>(data.trainTokensPerEpoch) * static_cast<double>(cfg.epochs);
	result.tokensPerSec = (seconds > 0.0) ? (tokens / seconds) : 0.0;
	return result;
}

static InferBenchResult run_infer_bench(const BenchConfig& cfg, const DatasetBundle& data, unsigned int runSeed)
{
	InferBenchResult result;
	NetworkOwner owner;
	if (!make_network(cfg, runSeed, data.padTokenId, "bench_transformer_gpu_infer", owner, result.err))
	{
		result.ok = false;
		return result;
	}
	if (!warmup_model(*owner.net, const_cast<InMemoryTokenIdInput&>(data.di), result.err))
	{
		result.ok = false;
		return result;
	}

	glades::NNetwork::TransformerLmSession session;
	const glades::NNetworkStatus stReset = owner.net->transformerLmSessionReset(session, cfg.inferPromptLen + cfg.inferSteps);
	if (!stReset.ok())
	{
		result.ok = false;
		result.err = stReset.message;
		return result;
	}

	std::vector<float> logits;
	const int64_t t0 = now_ms();
	for (unsigned int i = 0u; i < data.prompt.size(); ++i)
	{
		const glades::NNetworkStatus stAppend = owner.net->transformerLmSessionAppend(session, data.prompt[i], &logits);
		if (!stAppend.ok())
		{
			result.ok = false;
			result.err = stAppend.message;
			return result;
		}
		++result.appendedTokens;
	}
	for (unsigned int step = 0u; step < cfg.inferSteps; ++step)
	{
		const unsigned int nextToken = data.prompt[step % data.prompt.size()];
		const glades::NNetworkStatus stAppend = owner.net->transformerLmSessionAppend(session, nextToken, &logits);
		if (!stAppend.ok())
		{
			result.ok = false;
			result.err = stAppend.message;
			return result;
		}
		++result.appendedTokens;
	}
	const int64_t t1 = now_ms();

	result.wallMs = static_cast<long long>(t1 - t0);
	result.gpuPerf = owner.net->getLastTransformerInferGpuPerf();
	const double seconds = static_cast<double>(result.wallMs) / 1000.0;
	result.tokensPerSec = (seconds > 0.0) ? (static_cast<double>(result.appendedTokens) / seconds) : 0.0;
	return result;
}

static void print_config(const BenchConfig& cfg)
{
	printf("Config: mode=%s vocab=%u dModel=%u dFF=%u layers=%u heads=%u kvHeads=%u seqLen=%u trainSeqs=%u inferPrompt=%u inferSteps=%u epochs=%u repeats=%u lr=%g sampledNegatives=%u seed=%u\n",
	       (cfg.mode == BenchConfig::MODE_TRAIN) ? "train" : ((cfg.mode == BenchConfig::MODE_INFER) ? "infer" : "all"),
	       cfg.vocab, cfg.dModel, cfg.dFF, cfg.layers, cfg.heads, cfg.kvHeads,
	       cfg.seqLen, cfg.trainSeqs, cfg.inferPromptLen, cfg.inferSteps,
	       cfg.epochs, cfg.repeats, cfg.learningRate, cfg.sampledNegatives, cfg.seed);
}

static void print_train_header()
{
	printf("Train Repeat\tWall(ms)\tTok/s\tLoss\tPPL\tKernel\tSync\tH2D\tD2H\tD2D\tGPU ms\n");
}

static void print_infer_header()
{
	printf("Infer Repeat\tWall(ms)\tTok/s\tAppended\tKernel\tSync\tH2D\tD2H\tD2D\tGPU ms\n");
}

static void print_train_row(unsigned int rep, const TrainBenchResult& r)
{
	printf("%u\t\t%lld\t\t%.2f\t%.6f\t%.4f\t%llu\t%llu\t%llu\t%llu\t%llu\t%.3f\n",
	       rep + 1u,
	       r.wallMs,
	       r.tokensPerSec,
	       r.finalLoss,
	       r.finalPerplexity,
	       r.gpuPerf.counters.kernelLaunches,
	       r.gpuPerf.counters.syncPoints,
	       r.gpuPerf.counters.bytesH2D,
	       r.gpuPerf.counters.bytesD2H,
	       r.gpuPerf.counters.bytesD2D,
	       r.gpuPerf.msTotal);
}

static void print_infer_row(unsigned int rep, const InferBenchResult& r)
{
	printf("%u\t\t%lld\t\t%.2f\t%u\t\t%llu\t%llu\t%llu\t%llu\t%llu\t%.3f\n",
	       rep + 1u,
	       r.wallMs,
	       r.tokensPerSec,
	       r.appendedTokens,
	       r.gpuPerf.counters.kernelLaunches,
	       r.gpuPerf.counters.syncPoints,
	       r.gpuPerf.counters.bytesH2D,
	       r.gpuPerf.counters.bytesD2H,
	       r.gpuPerf.counters.bytesD2D,
	       r.gpuPerf.msTotal);
}

static void print_train_summary(const std::vector<TrainBenchResult>& runs)
{
	long long sumWall = 0LL;
	double sumTokPerSec = 0.0;
	float sumLoss = 0.0f;
	float sumPpl = 0.0f;
	unsigned long long sumKernel = 0ULL;
	unsigned long long sumSync = 0ULL;
	for (size_t i = 0; i < runs.size(); ++i)
	{
		sumWall += runs[i].wallMs;
		sumTokPerSec += runs[i].tokensPerSec;
		sumLoss += runs[i].finalLoss;
		sumPpl += runs[i].finalPerplexity;
		sumKernel += runs[i].gpuPerf.counters.kernelLaunches;
		sumSync += runs[i].gpuPerf.counters.syncPoints;
	}
	const double n = runs.empty() ? 1.0 : static_cast<double>(runs.size());
	printf("Train Avg\t%.2f\t\t%.2f\t%.6f\t%.4f\t%llu\t%llu\n",
	       static_cast<double>(sumWall) / n,
	       sumTokPerSec / n,
	       sumLoss / static_cast<float>(n),
	       sumPpl / static_cast<float>(n),
	       static_cast<unsigned long long>(sumKernel / runs.size()),
	       static_cast<unsigned long long>(sumSync / runs.size()));
}

static void print_infer_summary(const std::vector<InferBenchResult>& runs)
{
	long long sumWall = 0LL;
	double sumTokPerSec = 0.0;
	unsigned int sumTokens = 0u;
	unsigned long long sumKernel = 0ULL;
	unsigned long long sumSync = 0ULL;
	for (size_t i = 0; i < runs.size(); ++i)
	{
		sumWall += runs[i].wallMs;
		sumTokPerSec += runs[i].tokensPerSec;
		sumTokens += runs[i].appendedTokens;
		sumKernel += runs[i].gpuPerf.counters.kernelLaunches;
		sumSync += runs[i].gpuPerf.counters.syncPoints;
	}
	const double n = runs.empty() ? 1.0 : static_cast<double>(runs.size());
	printf("Infer Avg\t%.2f\t\t%.2f\t%.2f\t\t%llu\t%llu\n",
	       static_cast<double>(sumWall) / n,
	       sumTokPerSec / n,
	       static_cast<double>(sumTokens) / n,
	       static_cast<unsigned long long>(sumKernel / runs.size()),
	       static_cast<unsigned long long>(sumSync / runs.size()));
}

} // namespace

void TransformerGpuBenchmark(int argc, char* argv[])
{
	printf("============================================================\n");
	printf("Transformer GPU Microbenchmark\n");
	printf("============================================================\n");

	BenchConfig cfg;
	std::string err;
	if (!parse_args(argc, argv, cfg, err))
	{
		if (!err.empty())
			printf("Argument error: %s\n\n", err.c_str());
		print_usage();
		printf("============================================================\n");
		return;
	}

#ifdef GLADES_HAVE_CUDA
	if (!glades::gpu::initDevice())
	{
		printf("No CUDA device available, skipping transformer GPU benchmark.\n");
		printf("============================================================\n");
		return;
	}
#else
	printf("CUDA not enabled, skipping transformer GPU benchmark.\n");
	printf("============================================================\n");
	return;
#endif

	DatasetBundle data;
	build_dataset(cfg, data);
	print_config(cfg);
	printf("\n");

	if (cfg.mode == BenchConfig::MODE_ALL || cfg.mode == BenchConfig::MODE_TRAIN)
	{
		std::vector<TrainBenchResult> trainRuns;
		trainRuns.reserve(cfg.repeats);
		print_train_header();
		for (unsigned int rep = 0u; rep < cfg.repeats; ++rep)
		{
			TrainBenchResult r = run_train_bench(cfg, data, cfg.seed + rep);
			trainRuns.push_back(r);
			if (!r.ok)
			{
				printf("%u\t\tERR\t\t0\t%s\n", rep + 1u, r.err.c_str());
				printf("============================================================\n");
				return;
			}
			print_train_row(rep, r);
		}
		if (!trainRuns.empty())
			print_train_summary(trainRuns);
		printf("\n");
	}

	if (cfg.mode == BenchConfig::MODE_ALL || cfg.mode == BenchConfig::MODE_INFER)
	{
		std::vector<InferBenchResult> inferRuns;
		inferRuns.reserve(cfg.repeats);
		print_infer_header();
		for (unsigned int rep = 0u; rep < cfg.repeats; ++rep)
		{
			InferBenchResult r = run_infer_bench(cfg, data, cfg.seed + 1000u + rep);
			inferRuns.push_back(r);
			if (!r.ok)
			{
				printf("%u\t\tERR\t\t0\t%s\n", rep + 1u, r.err.c_str());
				printf("============================================================\n");
				return;
			}
			print_infer_row(rep, r);
		}
		if (!inferRuns.empty())
			print_infer_summary(inferRuns);
	}

	printf("============================================================\n");
}
