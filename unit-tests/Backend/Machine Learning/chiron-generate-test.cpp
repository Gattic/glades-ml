// chiron-generate-test.cpp — CHIRON generation-side RNG unit tests (gate G1).
//
// Pins ChironMt19937 to be bit-compatible with std::mt19937 AND with this
// machine's libstdc++ std::uniform_real_distribution<double>(0,1) draw path.
// If G1 is subtly wrong, gate G4 (same-seed stochastic generation identity)
// would fail silently much later, so these goldens are load-bearing.
//
// Golden source: Task-1 capture on the current binary/toolchain,
//   ~/dev/glades-trainer/logs/chiron-unify-goldens/gen/rng_and_sampler.txt
// — three seeds (1337 / 2024 / 4242), each 64 lines of %a-format hex doubles,
// exactly what std::uniform_real_distribution<double>(0,1)(std::mt19937(seed))
// produced.  Transcribed below verbatim as %a string literals and parsed with
// std::strtod (hex-float parsing supported by this toolchain's strtod).
//
// The replicated reference is libstdc++ 13 (g++ 13.3.0, C++98 build):
//   /usr/include/c++/13/bits/random.tcc  (mersenne_twister_engine seed/twist,
//     operator() tempering, and generate_canonical<double,53,mt19937>).
//   /usr/include/c++/13/bits/random.h    (_Adaptor::operator() routing +
//     uniform_real_distribution min/max == 0/1).
// generate_canonical for double over mt19937: b=53, r=2^32, m=2 draws,
//   ret = (u0 + u1*2^32) / 2^64, then `if (ret >= 1.0) ret = nextafter(1,0)`.
//
// C++98.

#include "chiron-generate-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/Networks/chiron_generate.h"
#include "../../../Backend/Machine Learning/Networks/chiron_serving.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_kernels.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_chiron.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_blas.h"

#include <cstdio>
#include <cstring>   // memcmp
#include <cstdlib>   // strtod
#include <stdint.h>
#include <cmath>     // std::sin, std::exp, std::log, std::sqrt
#include <limits>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Task-1 goldens: std::uniform_real_distribution<double>(0,1)(std::mt19937(s))
// first 64 draws per seed, in %a hex-float (bit-exact double serialization).
// ---------------------------------------------------------------------------

// seed 1337
static const char* const kGold_1337[64] = {
	"0x1.1efdc17a86282p-1",
	"0x1.b361274144fc2p-3",
	"0x1.160eabb28e66ap-1",
	"0x1.164b2b4eb2b95p-5",
	"0x1.e21382929168cp-3",
	"0x1.2f297091096adp-1",
	"0x1.571146190c3acp-2",
	"0x1.14f9356be782ep-2",
	"0x1.3ad4ea62ee66fp-2",
	"0x1.9ad026363b054p-1",
	"0x1.85b9b0662e2ecp-4",
	"0x1.73db05ca8395dp-2",
	"0x1.7b78bcd240079p-1",
	"0x1.4793265fde4ecp-3",
	"0x1.b9acf078e2ee6p-1",
	"0x1.946d25279441p-1",
	"0x1.5e2151cb4b5afp-8",
	"0x1.dab3696d71ee9p-2",
	"0x1.8636dce5aa172p-2",
	"0x1.0a7f03292b23ep-1",
	"0x1.7ad4b9230a6a8p-2",
	"0x1.e70d391cc050ep-2",
	"0x1.ba260eb6938aap-1",
	"0x1.9ebdcb12ae4dep-2",
	"0x1.c312df85ffa3dp-2",
	"0x1.20781c316db57p-3",
	"0x1.a3eebfe8d3874p-1",
	"0x1.cb994bea65fcfp-1",
	"0x1.db582def103aep-1",
	"0x1.3d354c25aa2c8p-1",
	"0x1.ea334d865ed93p-1",
	"0x1.c3127d45ea2bp-1",
	"0x1.a08d4588d9cffp-1",
	"0x1.626f4411020d3p-1",
	"0x1.62cf3fcd055c7p-1",
	"0x1.ace5beb00815dp-1",
	"0x1.3bc45d1176a01p-1",
	"0x1.5487868ffc928p-1",
	"0x1.d4b29b8ca6c96p-2",
	"0x1.a9ddb2f481ac9p-2",
	"0x1.431b93e2bfea4p-1",
	"0x1.9cb8f29b62edbp-1",
	"0x1.5b306898017c4p-1",
	"0x1.b232225abd0b9p-1",
	"0x1.851633701e0b3p-1",
	"0x1.ebb325439423ep-1",
	"0x1.b1afe00b314f3p-5",
	"0x1.0d5b989367b13p-1",
	"0x1.5b3823fefb826p-1",
	"0x1.7f847dc5f21c5p-1",
	"0x1.13f09fe1ac04p-1",
	"0x1.0f6e17f270e21p-2",
	"0x1.ad07f07b2108ap-1",
	"0x1.784e84edfea9dp-1",
	"0x1.b897244c826b3p-1",
	"0x1.37016acc0eb91p-2",
	"0x1.3813cc202d975p-1",
	"0x1.1b73f34bc184ap-2",
	"0x1.c449e5d3f2694p-1",
	"0x1.26d90ec1f746fp-2",
	"0x1.047db0baae8f3p-1",
	"0x1.2c6cd6bdc8699p-3",
	"0x1.6031750c05944p-1",
	"0x1.c79f01db85189p-1"
};

// seed 2024
static const char* const kGold_2024[64] = {
	"0x1.83a99bf52d104p-1",
	"0x1.7a3d150165f19p-1",
	"0x1.30b2ec4860557p-1",
	"0x1.c93d816e166e1p-1",
	"0x1.f97f98fe68f84p-1",
	"0x1.3200278364dep-5",
	"0x1.9768c0957458dp-1",
	"0x1.56cfebc15bda6p-1",
	"0x1.77d08bf8f29bep-1",
	"0x1.138613ce58708p-5",
	"0x1.0f4587b827219p-3",
	"0x1.d55b74dd81549p-1",
	"0x1.24064b7cd1d09p-3",
	"0x1.ff98fe39ec6e2p-1",
	"0x1.9c48409ea8504p-2",
	"0x1.f08da394da60ap-3",
	"0x1.2c9ad25cbee53p-6",
	"0x1.8ee195b47361ap-1",
	"0x1.f8c3f8aaae423p-2",
	"0x1.82402186f16cdp-2",
	"0x1.81cecc6d083p-2",
	"0x1.a383c4ee30eaep-1",
	"0x1.aa0eea77d7f8cp-2",
	"0x1.b7c2bf9a039e9p-3",
	"0x1.02a2de1120efep-2",
	"0x1.466654ad89589p-1",
	"0x1.ea4e6ffb3112dp-2",
	"0x1.6efefafd168c1p-1",
	"0x1.58e95e24c3f24p-1",
	"0x1.b35b5d6986cf8p-2",
	"0x1.5cf82b0e9257fp-1",
	"0x1.4026add17b047p-1",
	"0x1.57c704bc7a5c8p-1",
	"0x1.638750c0e01d1p-1",
	"0x1.259f560388be6p-2",
	"0x1.a283f73528395p-2",
	"0x1.ead9fc2d91aaep-1",
	"0x1.e0b5b22f092b2p-2",
	"0x1.42920983579fep-3",
	"0x1.488cfcc7397f7p-5",
	"0x1.7d1402ced7172p-1",
	"0x1.6c51373441e89p-2",
	"0x1.25d460b4c57e8p-3",
	"0x1.37d70c61acb5bp-1",
	"0x1.6894579c8f3fap-5",
	"0x1.d6dcead666ab9p-1",
	"0x1.d43e28110139fp-1",
	"0x1.b64b1b43ca6f8p-1",
	"0x1.eacaae5a83081p-1",
	"0x1.712dc319bc05ep-1",
	"0x1.3439803821c47p-3",
	"0x1.e29ce5171ae0ep-1",
	"0x1.256dfdb70e49ap-1",
	"0x1.2d86d7aec60b1p-4",
	"0x1.2e4f48ecfbd9fp-2",
	"0x1.ff1ec10420c7p-1",
	"0x1.bc3f54afcde0bp-1",
	"0x1.80a5b34efe089p-3",
	"0x1.c59787f253cc3p-1",
	"0x1.8adaa1e9ffde1p-1",
	"0x1.ec31b17eb1905p-1",
	"0x1.6040deaea0319p-1",
	"0x1.3b66003d8dbcfp-4",
	"0x1.6ba84ef1c2adap-1"
};

// seed 4242
static const char* const kGold_4242[64] = {
	"0x1.9dfcc3dca65fcp-1",
	"0x1.eda7148f85f81p-3",
	"0x1.cfabda8ba7f1bp-2",
	"0x1.55cfd6452562bp-2",
	"0x1.425a23ee4121cp-3",
	"0x1.aafafabb5083dp-1",
	"0x1.0743c6910f179p-1",
	"0x1.53f35754aba45p-1",
	"0x1.38ff7525c95cp-3",
	"0x1.d22c81850eb32p-1",
	"0x1.cf8a6d92feaa3p-5",
	"0x1.c65364ec87171p-2",
	"0x1.5242bb32fc75p-1",
	"0x1.d89ed260bf873p-2",
	"0x1.d4ced498295a2p-3",
	"0x1.60ef0ec3d9e87p-2",
	"0x1.c1b43aaa1caadp-4",
	"0x1.f4ae972a2eb26p-4",
	"0x1.cff3368396ae2p-1",
	"0x1.b9d00809f6cf3p-4",
	"0x1.3b5f66f95514ep-2",
	"0x1.4ceef76c974dap-1",
	"0x1.ce8a7f7cf1207p-1",
	"0x1.0656078d1fbf7p-7",
	"0x1.ebfe6f5681b7p-3",
	"0x1.405d6c32bbe18p-2",
	"0x1.28724c700a9ebp-1",
	"0x1.919fc213b2055p-1",
	"0x1.3ff1ed6c6e98bp-1",
	"0x1.a84f6f8bb0e2ep-2",
	"0x1.df4041a926badp-1",
	"0x1.e7a3e49667232p-1",
	"0x1.1742c20fe67fep-2",
	"0x1.96ae97defb99bp-2",
	"0x1.b9542afee822ep-1",
	"0x1.18574f8eb6d3ep-1",
	"0x1.cffc469eab27p-1",
	"0x1.7ce11fd34bc4ep-1",
	"0x1.563e433f77b12p-2",
	"0x1.58c9882f51e1dp-1",
	"0x1.2466814fc6c8p-1",
	"0x1.d7f0c035f1827p-2",
	"0x1.36441d8f6ab94p-2",
	"0x1.adee9bda11f15p-1",
	"0x1.43401c59bd5a6p-1",
	"0x1.8b0ebcfb1a66p-2",
	"0x1.f91cd21e6ceebp-4",
	"0x1.95da1e6b0b1cdp-1",
	"0x1.ca1ae53fd17d6p-3",
	"0x1.201215d791e41p-2",
	"0x1.20c51b69e18ebp-1",
	"0x1.2578f68e50d68p-2",
	"0x1.df187a7611b9fp-1",
	"0x1.147792149d634p-2",
	"0x1.014402a0b966ap-3",
	"0x1.a2d1e1e25d0c4p-1",
	"0x1.8e191b511c17dp-1",
	"0x1.72fe9e05152a9p-1",
	"0x1.f4d5b63b5e7e7p-8",
	"0x1.b844e0ff129eep-1",
	"0x1.91567a442335ep-3",
	"0x1.4d99b83341e9p-1",
	"0x1.23953e0e4b138p-1",
	"0x1.dda556191b0b5p-1"
};

// ---------------------------------------------------------------------------
// Helper: bit-exact double compare via 8-byte memcmp (avoids NaN edge cases
// and any == fuzz — these are supposed to be the identical bit pattern).
// ---------------------------------------------------------------------------
static bool bits_equal(double a, double b)
{
	return memcmp(&a, &b, sizeof(double)) == 0;
}

static void check_seed(uint32_t seed, const char* const gold[64])
{
	glades::chiron::ChironMt19937 rng(seed);
	for (int i = 0; i < 64; ++i)
	{
		double want = std::strtod(gold[i], (char**)0);
		// Guard against silent hex-float parse failure: every golden is a
		// canonical double in (0,1), strictly positive and < 1.
		char first_msg[96];
		std::sprintf(first_msg, "seed %u golden[%d] parsed into (0,1)", (unsigned)seed, i);
		ASSERT(first_msg, want > 0.0 && want < 1.0);

		double got = rng.next_canonical_double();
		if (!bits_equal(got, want))
		{
			char msg[160];
			std::sprintf(msg,
				"seed %u canonical-double[%d] bit-mismatch: got=%a want=%a",
				(unsigned)seed, i, got, want);
			ASSERT(msg, false);
		}
	}
}

// ---------------------------------------------------------------------------
// Test 1: raw next_u32() core generator against the published MT19937
// reference for the default seed 5489 — catches seed/twist/temper bugs
// independently of the canonical-double combine.
// ---------------------------------------------------------------------------
void CHIRONMt19937RawTest()
{
	glades::chiron::ChironMt19937 rng(5489u);
	uint32_t a = rng.next_u32();
	uint32_t b = rng.next_u32();
	uint32_t c = rng.next_u32();
	ASSERT("mt19937 seed5489 u32[0] == 3499211612", a == 3499211612u);
	ASSERT("mt19937 seed5489 u32[1] == 581869302",  b == 581869302u);
	ASSERT("mt19937 seed5489 u32[2] == 3890346734", c == 3890346734u);
}

// ---------------------------------------------------------------------------
// Test 2: canonical-double stream bit-exact vs Task-1 goldens (all 3 seeds).
// ---------------------------------------------------------------------------
void CHIRONMt19937GoldenTest()
{
	check_seed(1337u, kGold_1337);
	check_seed(2024u, kGold_2024);
	check_seed(4242u, kGold_4242);
}

// ---------------------------------------------------------------------------
// Test 3: chiron_sample_token golden picks (G2 gate).
// V=64 logits sin(0.37*i)*4, initial history {3,7,3,9,3,7}, 6 configs,
// fresh ChironMt19937(1337) per config, 16 sequential picks each appended
// to the history copy.
// Goldens transcribed from:
//   ~/dev/glades-trainer/logs/chiron-unify-goldens/gen/rng_and_sampler.txt
//   lines 196-201 (cfg NAME: p0 p1 ... p15).
// ---------------------------------------------------------------------------
void CHIRONSamplerGoldenTest()
{
	const int V = 64;
	std::vector<float> logits(V);
	for (int i = 0; i < V; ++i)
		logits[i] = (float)(std::sin(0.37 * i) * 4.0);

	const int initHist[] = {3, 7, 3, 9, 3, 7};
	const int nHist = 6;

	// Struct holds one config + 16 golden picks.
	struct SamplerCfg {
		const char* name;
		float  temperature;
		int    topK;
		float  topP;
		int    repWindow;
		float  repPenalty;
		float  freqPenalty;
		float  presPenalty;
		int    noRepeatN;
		int    golden[16];
	};

	// 6 configs from the task brief; goldens from rng_and_sampler.txt lines 196-201.
	static const SamplerCfg kCfgs[] = {
		// cfg defaults: 38 6 37 4 21 40 22 20 23 55 5 39 56 19 56 54
		{ "defaults",       0.8f,  40, 0.95f, 256, 1.0f, 1.2f, 0.4f, 3,
		  {38, 6, 37, 4, 21, 40, 22, 20, 23, 55, 5, 39, 56, 19, 56, 54} },
		// cfg no-penalties: 37 5 37 3 6 38 21 19 20 54 4 21 40 5 55 54
		{ "no-penalties",   0.8f,  40, 0.95f,   0, 1.0f, 0.0f, 0.0f, 0,
		  {37, 5, 37, 3, 6, 38, 21, 19, 20, 54, 4, 21, 40, 5, 55, 54} },
		// cfg topk1: 55 38 21 4 5 22 39 56 54 37 20 6 23 40 57 53
		{ "topk1",          0.8f,   1, 0.95f, 256, 1.0f, 1.2f, 0.4f, 3,
		  {55, 38, 21, 4, 5, 22, 39, 56, 54, 37, 20, 6, 23, 40, 57, 53} },
		// cfg topp-only: 22 5 39 4 37 55 21 20 38 56 6 40 54 19 57 53
		{ "topp-only",      0.8f,   0, 0.50f, 256, 1.0f, 1.2f, 0.4f, 3,
		  {22, 5, 39, 4, 37, 55, 21, 20, 38, 56, 6, 40, 54, 19, 57, 53} },
		// cfg hot-ngram2: 38 5 39 4 21 54 22 23 37 56 6 36 55 19 57 53
		{ "hot-ngram2",     1.5f,   8, 1.00f, 256, 1.5f, 0.7f, 0.2f, 2,
		  {38, 5, 39, 4, 21, 54, 22, 23, 37, 56, 6, 36, 55, 19, 57, 53} },
		// cfg cold-heavy-pen: 38 5 37 4 21 54 38 22 5 55 4 21 39 5 55 38
		{ "cold-heavy-pen", 0.2f,  64, 0.99f,   4, 1.0f, 2.0f, 1.0f, 0,
		  {38, 5, 37, 4, 21, 54, 38, 22, 5, 55, 4, 21, 39, 5, 55, 38} },
	};
	const int nCfgs = 6;

	for (int c = 0; c < nCfgs; ++c)
	{
		const SamplerCfg& sc = kCfgs[c];

		// Build gen params, override relevant fields.
		glades::chiron::ChironGenParams gp;
		gp.temperature  = sc.temperature;
		gp.topK         = sc.topK;
		gp.topP         = sc.topP;
		gp.repWindow    = sc.repWindow;
		gp.repPenalty   = sc.repPenalty;
		gp.freqPenalty  = sc.freqPenalty;
		gp.presPenalty  = sc.presPenalty;
		gp.noRepeatN    = sc.noRepeatN;

		// Fresh RNG per config (seed 1337).
		glades::chiron::ChironMt19937 rng(1337u);

		// History copy, extended one pick at a time.
		std::vector<int> hist(initHist, initHist + nHist);

		for (int pick = 0; pick < 16; ++pick)
		{
			int got = glades::chiron::chiron_sample_token(logits, gp, hist, rng);
			char msg[160];
			std::sprintf(msg,
				"sampler cfg=%s pick[%d]: got %d want %d",
				sc.name, pick, got, sc.golden[pick]);
			ASSERT(msg, got == sc.golden[pick]);
			hist.push_back(got);
		}
	}
}

// ---------------------------------------------------------------------------
// Test 4: CHIRONDegenMetricsTest — chiron_degeneration_metrics.
// Four cases from the task brief + CHIRON_INFER_SELFTEST block.
// ---------------------------------------------------------------------------
void CHIRONDegenMetricsTest()
{
	// Case 1: 40 copies of token 2 → maxRun >= 30, distinct4 < 0.1
	{
		std::vector<int> rep;
		for (int i = 0; i < 40; ++i) rep.push_back(2);
		double d4 = 0.0; int run = 0;
		glades::chiron::chiron_degeneration_metrics(rep, d4, run);
		ASSERT("degen rep distinct4 < 0.1", d4 < 0.1);
		ASSERT("degen rep maxRun >= 30",    run >= 30);
	}

	// Case 2: varied (i*7+3)%50 sequence → distinct4 > 0.8, maxRun <= 2
	{
		std::vector<int> coh;
		for (int i = 0; i < 40; ++i) coh.push_back((i * 7 + 3) % 50);
		double d4 = 0.0; int run = 0;
		glades::chiron::chiron_degeneration_metrics(coh, d4, run);
		ASSERT("degen coh distinct4 > 0.8", d4 > 0.8);
		ASSERT("degen coh maxRun <= 2",      run <= 2);
	}

	// Case 3: empty → distinct4 == 1.0, maxRun == 0
	{
		std::vector<int> empty;
		double d4 = 0.0; int run = -1;
		glades::chiron::chiron_degeneration_metrics(empty, d4, run);
		ASSERT("degen empty distinct4 == 1.0", d4 == 1.0);
		ASSERT("degen empty maxRun == 0",       run == 0);
	}

	// Case 4: <4 tokens → distinct4 == 1.0
	{
		std::vector<int> short3;
		short3.push_back(7); short3.push_back(3); short3.push_back(7);
		double d4 = 0.0; int run = 0;
		glades::chiron::chiron_degeneration_metrics(short3, d4, run);
		ASSERT("degen short distinct4 == 1.0", d4 == 1.0);
	}
}

// ---------------------------------------------------------------------------
// Tests 4b-4d: ARREST detector-v1 pure CPU contract.
// ---------------------------------------------------------------------------

static int hazard_token_index(const glades::chiron::ChironHazardRow& row, int token)
{
	for (int i = 0; i < row.count; ++i) if (row.tokenIds[i] == token) return i;
	return -1;
}

static void assert_config_parse_rejected(const char* message, const std::string& bytes)
{
	glades::chiron::ChironRepetitionConfig out;
	out.maxPeriod = 17;
	std::string error;
	const bool ok = glades::chiron::chiron_repetition_config_parse(bytes, out, &error);
	ASSERT(message, !ok && !error.empty() && out.maxPeriod == 17);
}

static void assert_config_value_rejected(
	const char* message, const glades::chiron::ChironRepetitionConfig& config)
{
	std::string error;
	ASSERT(message, !glades::chiron::chiron_repetition_config_validate(config, &error) &&
	       !error.empty());
	std::string bytes("stale");
	ASSERT(message, !glades::chiron::chiron_repetition_config_serialize(config, bytes, &error) &&
	       bytes.empty());
	std::string hash("stale");
	ASSERT(message, !glades::chiron::chiron_repetition_config_sha256(config, hash, &error) &&
	       hash.empty());
}

void CHIRONRepetitionConfigContractTest()
{
	const char* expected =
		"format=chiron-arrest-detector-config\n"
		"version=1\n"
		"maxPeriod=64\n"
		"minCycleSupport=32\n"
		"cycleThreshold=0.800000\n"
		"repeatLookback=64\n"
		"repeatedSpanWindow=128\n"
		"minRepeatedSpan=8\n"
		"diversityWindow=64\n"
		"maxRunThreshold=8\n"
		"repeatedSpanThreshold=0.600000\n"
		"distinct1Threshold=0.150000\n"
		"distinct4Threshold=0.350000\n"
		"ngramWindow=128\n"
		"maxHazards=16\n"
		"minPeriodHazardSupport=8\n"
		"hazardThreshold=0.700000\n"
		"runConfidenceSpan=8\n"
		"ngramConfidenceCount=4\n"
		"postOnsetDecay=32.000000\n";

	glades::chiron::ChironRepetitionConfig config;
	std::string error("stale"), bytes;
	ASSERT("config schema version", glades::chiron::CHIRON_REPETITION_CONFIG_VERSION == 1);
	ASSERT("config defaults validate",
	       glades::chiron::chiron_repetition_config_validate(config, &error) && error.empty());
	ASSERT("config optional error API",
	       glades::chiron::chiron_repetition_config_validate(config));
	ASSERT("config defaults serialize",
	       glades::chiron::chiron_repetition_config_serialize(config, bytes, &error));
	ASSERT("config canonical bytes", bytes == expected && bytes.size() == 434);

	glades::chiron::ChironRepetitionConfig parsed;
	parsed.maxPeriod = 1;
	ASSERT("config canonical parse",
	       glades::chiron::chiron_repetition_config_parse(bytes, parsed, &error));
	std::string roundTrip;
	ASSERT("config canonical round trip",
	       glades::chiron::chiron_repetition_config_serialize(parsed, roundTrip, &error) &&
	       roundTrip == bytes);
	std::string aliasedInput(bytes);
	glades::chiron::ChironRepetitionConfig aliasParsed;
	ASSERT("config parse permits input/error alias",
	       glades::chiron::chiron_repetition_config_parse(
	           aliasedInput, aliasParsed, &aliasedInput) && aliasedInput.empty() &&
	       aliasParsed.maxPeriod == config.maxPeriod);

	std::string hash, hashAgain;
	ASSERT("config sha256",
	       glades::chiron::chiron_repetition_config_sha256(config, hash, &error));
	ASSERT("config sha256 golden",
	       hash == "8a90e0790e0a5571374cb30dc8bae53aada84e2f3ed03277781ad035baeb9c04");
	ASSERT("config sha256 deterministic",
	       glades::chiron::chiron_repetition_config_sha256(parsed, hashAgain, &error) &&
	       hashAgain == hash);
	glades::chiron::ChironRepetitionConfig changed(config);
	changed.maxRunThreshold = 9;
	ASSERT("config sha256 content identity",
	       glades::chiron::chiron_repetition_config_sha256(changed, hashAgain, &error) &&
	       hashAgain != hash);
	glades::chiron::ChironRepetitionConfig paddingBoundary(config);
	paddingBoundary.minCycleSupport = 1048576;
	paddingBoundary.repeatLookback = 1048576;
	ASSERT("config sha256 two-block padding golden",
	       glades::chiron::chiron_repetition_config_serialize(
	           paddingBoundary, roundTrip, &error) && roundTrip.size() == 444 &&
	       roundTrip.size() % 64 == 60 &&
	       glades::chiron::chiron_repetition_config_sha256(
	           paddingBoundary, hashAgain, &error) &&
	       hashAgain == "7e09525266becbdefdebb766237770954504975f54f78eccdd50ce9cf33f21ae");

	glades::chiron::ChironRepetitionConfig minimum(config);
	minimum.maxPeriod = 1; minimum.minCycleSupport = 1; minimum.cycleThreshold = 0.000001f;
	minimum.repeatLookback = 1; minimum.repeatedSpanWindow = 1; minimum.minRepeatedSpan = 1;
	minimum.diversityWindow = 4; minimum.maxRunThreshold = 1;
	minimum.repeatedSpanThreshold = 0.000001f; minimum.distinct1Threshold = 0.000001f;
	minimum.distinct4Threshold = 0.000001f; minimum.ngramWindow = 2; minimum.maxHazards = 1;
	minimum.minPeriodHazardSupport = 1; minimum.hazardThreshold = 0.000001f;
	minimum.runConfidenceSpan = 1; minimum.ngramConfidenceCount = 2;
	minimum.postOnsetDecay = 0.000001f;
	ASSERT("config accepts all lower boundaries",
	       glades::chiron::chiron_repetition_config_serialize(minimum, roundTrip, &error) &&
	       glades::chiron::chiron_repetition_config_parse(roundTrip, parsed, &error));

	glades::chiron::ChironRepetitionConfig maximum(config);
	maximum.minCycleSupport = 1048576; maximum.cycleThreshold = 1.0f;
	maximum.repeatLookback = 1048576; maximum.repeatedSpanWindow = 1048576;
	maximum.minRepeatedSpan = 1048576; maximum.diversityWindow = 1048576;
	maximum.maxRunThreshold = 1048576; maximum.repeatedSpanThreshold = 1.0f;
	maximum.distinct1Threshold = 1.0f; maximum.distinct4Threshold = 1.0f;
	maximum.ngramWindow = 1048576; maximum.minPeriodHazardSupport = 1048576;
	maximum.hazardThreshold = 1.0f; maximum.runConfidenceSpan = 1048576;
	maximum.ngramConfidenceCount = 1048576; maximum.postOnsetDecay = 1048576.0f;
	ASSERT("config accepts all upper boundaries",
	       glades::chiron::chiron_repetition_config_serialize(maximum, roundTrip, &error) &&
	       glades::chiron::chiron_repetition_config_parse(roundTrip, parsed, &error));

	std::string malformed(bytes);
	malformed.replace(malformed.find("version=1"), 9, "version=2");
	assert_config_parse_rejected("config rejects future version", malformed);
	malformed = bytes.substr(0, bytes.size() - 1);
	assert_config_parse_rejected("config rejects missing final LF", malformed);
	malformed = bytes + "trailing=true\n";
	assert_config_parse_rejected("config rejects trailing data", malformed);
	malformed = std::string("\xEF\xBB\xBF") + bytes;
	assert_config_parse_rejected("config rejects UTF-8 BOM", malformed);
	malformed = " " + bytes;
	assert_config_parse_rejected("config rejects whitespace", malformed);
	malformed = bytes;
	malformed.erase(malformed.find("maxPeriod=64\n"), 13);
	assert_config_parse_rejected("config rejects missing field", malformed);
	malformed.assign(4097, 'x'); malformed[4096] = '\n';
	assert_config_parse_rejected("config rejects oversized bytes", malformed);
	malformed = bytes;
	malformed.insert(malformed.find('\n'), 1, '\r');
	assert_config_parse_rejected("config rejects CRLF", malformed);
	malformed = bytes;
	malformed.replace(malformed.find("maxPeriod"), 9, "unknownxx");
	assert_config_parse_rejected("config rejects unknown key", malformed);
	malformed = bytes;
	malformed.insert(malformed.find("minCycleSupport="), "maxPeriod=64\n");
	assert_config_parse_rejected("config rejects duplicate key", malformed);
	malformed = bytes;
	const std::string ordered = "maxPeriod=64\nminCycleSupport=32\n";
	const std::string reversed = "minCycleSupport=32\nmaxPeriod=64\n";
	malformed.replace(malformed.find(ordered), ordered.size(), reversed);
	assert_config_parse_rejected("config rejects reordered fields", malformed);
	malformed = bytes;
	malformed.replace(malformed.find("maxPeriod=64"), 12, "maxPeriod=064");
	assert_config_parse_rejected("config rejects leading zero", malformed);
	malformed = bytes;
	malformed.replace(malformed.find("0.800000"), 8, "0.80000");
	assert_config_parse_rejected("config rejects noncanonical decimal", malformed);
	malformed = bytes;
	malformed.insert(malformed.find("cycleThreshold"), 1, '\0');
	assert_config_parse_rejected("config rejects embedded NUL", malformed);

	glades::chiron::ChironRepetitionConfig invalid(config);
	invalid.maxPeriod = 65;
	assert_config_value_rejected("config rejects period clamp alias", invalid);
	invalid = config; invalid.maxPeriod = 0;
	assert_config_value_rejected("config rejects zero period", invalid);
	invalid = config; invalid.maxHazards = 17;
	assert_config_value_rejected("config rejects hazard cap alias", invalid);
	invalid = config; invalid.maxHazards = 0;
	assert_config_value_rejected("config rejects zero hazard cap", invalid);
	invalid = config; invalid.repeatLookback = 0;
	assert_config_value_rejected("config rejects zero window", invalid);
	invalid = config; invalid.hazardThreshold = 0.0f;
	assert_config_value_rejected("config rejects zero threshold", invalid);
	invalid = config; invalid.repeatedSpanThreshold = 1.000001f;
	assert_config_value_rejected("config rejects threshold above one", invalid);
	invalid = config; invalid.postOnsetDecay = 0.0f;
	assert_config_value_rejected("config rejects zero decay", invalid);
	invalid = config; invalid.minRepeatedSpan = invalid.repeatedSpanWindow + 1;
	assert_config_value_rejected("config rejects impossible repeated span", invalid);
	invalid = config; invalid.minPeriodHazardSupport = 129;
	assert_config_value_rejected("config rejects impossible period support", invalid);
	invalid = config; invalid.cycleThreshold = std::numeric_limits<float>::quiet_NaN();
	assert_config_value_rejected("config rejects NaN", invalid);
	invalid = config; invalid.hazardThreshold = std::numeric_limits<float>::infinity();
	assert_config_value_rejected("config rejects infinity", invalid);
	invalid = config; invalid.distinct1Threshold = 0.1234567f;
	assert_config_value_rejected("config rejects sub-micro precision", invalid);
}

void CHIRONRepetitionMetricsTest()
{
	glades::chiron::ChironRepetitionConfig cfg;
	ASSERT("detector max period default", cfg.maxPeriod == 64);
	ASSERT("detector cycle support default", cfg.minCycleSupport == 32);
	ASSERT("detector cycle threshold default", cfg.cycleThreshold == 0.80f);
	ASSERT("detector repeat lookback default", cfg.repeatLookback == 64);
	ASSERT("detector span window default", cfg.repeatedSpanWindow == 128);
	ASSERT("detector min span default", cfg.minRepeatedSpan == 8);
	ASSERT("detector diversity window default", cfg.diversityWindow == 64);
	ASSERT("detector onset threshold default", cfg.maxRunThreshold == 8);
	ASSERT("detector span threshold default", cfg.repeatedSpanThreshold == 0.60f);
	ASSERT("detector distinct1 threshold default", cfg.distinct1Threshold == 0.15f);
	ASSERT("detector distinct4 threshold default", cfg.distinct4Threshold == 0.35f);
	ASSERT("detector ngram window default", cfg.ngramWindow == 128);
	ASSERT("detector hazard cap default", cfg.maxHazards == 16);
	ASSERT("detector period evidence default", cfg.minPeriodHazardSupport == 8);
	ASSERT("detector hazard threshold default", cfg.hazardThreshold == 0.70f);
	ASSERT("detector run confidence default", cfg.runConfidenceSpan == 8);
	ASSERT("detector ngram confidence default", cfg.ngramConfidenceCount == 4);
	ASSERT("detector post-onset decay default", cfg.postOnsetDecay == 32.0f);

	// Empty and short input stay finite and non-collapsed.
	{
		std::vector<int> empty;
		glades::chiron::ChironRepetitionMetrics m;
		glades::chiron::chiron_repetition_metrics(empty, cfg, m);
		ASSERT("detector empty d1", m.distinct1 == 1.0);
		ASSERT("detector empty d4", m.distinct4 == 1.0);
		ASSERT("detector empty run", m.maxRun == 0);
		ASSERT("detector empty onset", m.collapseOnset == -1 && !m.collapsed);
		const int shortValues[] = {1, 2, 1};
		std::vector<int> shortInput(shortValues, shortValues + 3);
		glades::chiron::chiron_repetition_metrics(shortInput, cfg, m);
		ASSERT("detector short d4", m.distinct4 == 1.0);
		ASSERT("detector short no collapse", !m.collapsed);
	}

	// Constant collapse is first-hit by the eighth token (zero-based index 7).
	{
		std::vector<int> x(40, 5);
		glades::chiron::ChironRepetitionMetrics m;
		glades::chiron::chiron_repetition_metrics(x, cfg, m);
		ASSERT("detector constant max run", m.maxRun == 40);
		ASSERT("detector constant onset", m.collapseOnset == 7 && m.collapsed);
		ASSERT("detector constant repeat fraction", std::fabs(m.repeatFraction - 39.0 / 40.0) < 1e-12);
	}

	// Every supported fundamental period reaches a perfect full-support cycle.
	const int periods[] = {2, 3, 8, 16, 32, 64};
	for (int pi = 0; pi < 6; ++pi)
	{
		const int period = periods[pi];
		std::vector<int> x;
		for (int i = 0; i < 200; ++i) x.push_back(100 + (i % period));
		glades::chiron::ChironRepetitionMetrics m;
		glades::chiron::chiron_repetition_metrics(x, cfg, m);
		char msg[96];
		std::sprintf(msg, "detector period %d score", period);
		ASSERT(msg, m.cycleMax == 1.0);
		std::sprintf(msg, "detector period %d identity", period);
		ASSERT(msg, m.cyclePeriod == period);
		std::sprintf(msg, "detector period %d collapsed", period);
		ASSERT(msg, m.collapsed);
	}

	// Two exact 64-token spans are fully covered, without requiring cycle-64
	// support (which needs 192 tokens).
	{
		std::vector<int> x;
		for (int i = 0; i < 64; ++i) x.push_back(1000 + i);
		for (int i = 0; i < 64; ++i) x.push_back(1000 + i);
		glades::chiron::ChironRepetitionMetrics m;
		glades::chiron::chiron_repetition_metrics(x, cfg, m);
		ASSERT("detector repeated span coverage", m.repeatedSpanCoverage == 1.0);
		ASSERT("detector repeated span collapse", m.collapsed);
		ASSERT("detector suffix copy overlap", m.longestSuffixCopy == 64);
	}

	// A template may retain high distinct-4 while containing repeated local
	// structure; it must not trip the diversity conjunction by itself.
	{
		std::vector<int> x;
		for (int block = 0; block < 8; ++block)
		{
			for (int i = 0; i < 8; ++i) x.push_back(200 + i);
			for (int i = 0; i < 24; ++i) x.push_back(10000 + block * 24 + i);
		}
		glades::chiron::ChironRepetitionMetrics m;
		glades::chiron::chiron_repetition_metrics(x, cfg, m);
		ASSERT("detector high-d4 template", m.distinct4 > 0.8);
		ASSERT("detector high-d4 template not collapsed", !m.collapsed);
	}

	// Varied soup and a short code-like continuation remain clean.
	{
		std::vector<int> soup;
		for (int i = 0; i < 256; ++i) soup.push_back(5000 + i);
		glades::chiron::ChironRepetitionMetrics m;
		glades::chiron::chiron_repetition_metrics(soup, cfg, m);
		ASSERT("detector soup d1", m.distinct1 == 1.0);
		ASSERT("detector soup d4", m.distinct4 == 1.0);
		ASSERT("detector soup no collapse", !m.collapsed);
		const int codeValues[] = {10, 20, 30, 123, 40, 41, 59, 11, 21, 31, 123, 42, 43, 59};
		std::vector<int> code(codeValues, codeValues + 14);
		glades::chiron::chiron_repetition_metrics(code, cfg, m);
		ASSERT("detector code delimiters no collapse", !m.collapsed);
		const int proseValues[] = {
			301, 7, 302, 11, 303, 7, 304, 12, 305, 7, 306, 13,
			307, 8, 308, 14, 309, 7, 310, 15, 311, 9, 312, 16,
			313, 7, 314, 17, 315, 10, 316, 18, 317, 7, 318, 19
		};
		std::vector<int> prose(proseValues, proseValues + 36);
		glades::chiron::chiron_repetition_metrics(prose, cfg, m);
		ASSERT("detector real-like continuation no collapse", !m.collapsed);
	}

	// Repeat fraction uses only the preceding configured lookback.
	{
		const int values[] = {1, 2, 1, 3};
		std::vector<int> x(values, values + 4);
		glades::chiron::ChironRepetitionMetrics m;
		glades::chiron::chiron_repetition_metrics(x, cfg, m);
		ASSERT("detector repeat fraction", m.repeatFraction == 0.25);
	}

	// The legacy API remains exactly equal to the new whole-trajectory fields.
	{
		std::vector<int> x;
		for (int i = 0; i < 80; ++i) x.push_back((i * 7 + 3) % 23);
		double legacyD4 = 0.0; int legacyRun = -1;
		glades::chiron::chiron_degeneration_metrics(x, legacyD4, legacyRun);
		glades::chiron::ChironRepetitionMetrics m;
		glades::chiron::chiron_repetition_metrics(x, cfg, m);
		ASSERT("detector legacy distinct4 exact", legacyD4 == m.distinct4);
		ASSERT("detector legacy max run exact", legacyRun == m.maxRun);
	}
}

void CHIRONRepetitionHazardsTest()
{
	// Run confidence and strictly causal pre/post-onset row weights.
	{
		glades::chiron::ChironRepetitionConfig cfg;
		cfg.maxPeriod = 0;
		cfg.ngramWindow = 0;
		std::vector<int> x(10, 5);
		x.push_back(9); // supplies a row for the state after ten repeated tokens
		std::vector<glades::chiron::ChironHazardRow> rows;
		std::vector<float> weights;
		glades::chiron::chiron_repetition_hazards(x, cfg, rows, weights);
		ASSERT("hazard row count", rows.size() == x.size() && weights.size() == x.size());
		ASSERT("hazard row zero empty", rows[0].count == 0 && weights[0] == 0.0f);
		int at2 = hazard_token_index(rows[2], 5);
		ASSERT("hazard run starts after two", at2 >= 0 && rows[2].confidence[at2] == 0.25f);
		ASSERT("hazard weak run has zero weight", weights[2] == 0.0f);
		int at6 = hazard_token_index(rows[6], 5);
		ASSERT("hazard run candidate", at6 >= 0);
		ASSERT("hazard run confidence", rows[6].confidence[at6] == 0.75f);
		ASSERT("hazard run pre-onset weight", weights[6] == 0.75f);
		ASSERT("hazard onset-emitting row no hindsight", std::fabs(weights[7] - 0.875f) < 1e-7f);
		ASSERT("hazard first post-onset weight", weights[8] == 1.0f);
		ASSERT("hazard decayed post-onset weight",
		       std::fabs(weights[9] - (float)std::exp(-1.0 / 32.0)) < 1e-6f);
	}

	// A confident period predicts exactly the token p positions back.
	{
		glades::chiron::ChironRepetitionConfig cfg;
		cfg.ngramWindow = 0;
		cfg.repeatedSpanWindow = 0;
		cfg.diversityWindow = 0;
		cfg.maxRunThreshold = 0;
		std::vector<int> x;
		for (int i = 0; i < 14; ++i) x.push_back(10 + (i % 2));
		x.push_back(99);
		std::vector<glades::chiron::ChironHazardRow> rows;
		std::vector<float> weights;
		glades::chiron::chiron_repetition_hazards(x, cfg, rows, weights);
		int index = hazard_token_index(rows[14], x[12]);
		ASSERT("hazard period candidate", index >= 0);
		ASSERT("hazard period confidence", rows[14].confidence[index] == 1.0f);
		ASSERT("hazard period weight", weights[14] == 1.0f);
	}

	// N-gram closure requires two observed completions and deduplicates the
	// same token across n=2 and n=3 at maximum confidence.
	{
		glades::chiron::ChironRepetitionConfig cfg;
		cfg.maxPeriod = 0;
		const int values[] = {1, 2, 3, 1, 2, 3, 1, 2, 99};
		std::vector<int> x(values, values + 9);
		std::vector<glades::chiron::ChironHazardRow> rows;
		std::vector<float> weights;
		glades::chiron::chiron_repetition_hazards(x, cfg, rows, weights);
		ASSERT("hazard ngram dedup count", rows[8].count == 1);
		ASSERT("hazard ngram token", rows[8].tokenIds[0] == 3);
		ASSERT("hazard ngram confidence", rows[8].confidence[0] == 0.5f);
		ASSERT("hazard ngram below weight threshold", weights[8] == 0.0f);
	}

	// Confidence sorts before token ID, while duplicate evidence retains the
	// maximum confidence for one token.
	{
		glades::chiron::ChironRepetitionConfig cfg;
		cfg.maxPeriod = 0;
		cfg.ngramConfidenceCount = 8;
		const int values[] = {5, 3, 5, 3, 5, 3, 5, 5, 5, 5, 5, 5, 99};
		std::vector<int> x(values, values + 13);
		std::vector<glades::chiron::ChironHazardRow> rows;
		std::vector<float> weights;
		glades::chiron::chiron_repetition_hazards(x, cfg, rows, weights);
		ASSERT("hazard confidence order has candidates", rows[12].count >= 2);
		ASSERT("hazard confidence order first token", rows[12].tokenIds[0] == 5);
		ASSERT("hazard confidence order descending",
		       rows[12].confidence[0] > rows[12].confidence[1]);
		ASSERT("hazard confidence order keeps lower token",
		       hazard_token_index(rows[12], 3) >= 0);
	}

	// More than sixteen unique bigram closures sets overflow and retains the
	// deterministic lowest token IDs when all confidences tie.
	{
		glades::chiron::ChironRepetitionConfig cfg;
		cfg.maxPeriod = 0;
		std::vector<int> x;
		for (int token = 100; token < 120; ++token)
		{
			x.push_back(7); x.push_back(token);
			x.push_back(7); x.push_back(token);
		}
		x.push_back(7);
		const int state = (int)x.size();
		x.push_back(999);
		std::vector<glades::chiron::ChironHazardRow> rows;
		std::vector<float> weights;
		glades::chiron::chiron_repetition_hazards(x, cfg, rows, weights);
		ASSERT("hazard overflow flag", rows[state].overflow);
		ASSERT("hazard overflow cap", rows[state].count == 16);
		for (int i = 0; i < 16; ++i)
		{
			char msg[80]; std::sprintf(msg, "hazard overflow order %d", i);
			ASSERT(msg, rows[state].tokenIds[i] == 100 + i);
		}
		cfg.maxHazards = 0;
		glades::chiron::chiron_repetition_hazards(x, cfg, rows, weights);
		ASSERT("hazard zero cap empty", rows[state].count == 0);
		ASSERT("hazard zero cap overflow", rows[state].overflow);
	}

	// Invalid/nonpositive evidence settings stay bounded and empty.
	{
		glades::chiron::ChironRepetitionConfig cfg;
		cfg.maxPeriod = 0; cfg.minCycleSupport = 0; cfg.ngramWindow = 0;
		cfg.runConfidenceSpan = 0; cfg.maxHazards = -1;
		cfg.repeatedSpanWindow = 0; cfg.diversityWindow = 0; cfg.maxRunThreshold = 0;
		std::vector<int> x(12, 4);
		std::vector<glades::chiron::ChironHazardRow> rows;
		std::vector<float> weights;
		glades::chiron::chiron_repetition_hazards(x, cfg, rows, weights);
		for (size_t i = 0; i < rows.size(); ++i)
			ASSERT("hazard invalid config empty", rows[i].count == 0 && weights[i] == 0.0f);
		glades::chiron::ChironRepetitionMetrics m;
		glades::chiron::chiron_repetition_metrics(x, cfg, m);
		ASSERT("metrics invalid config bounded", !m.collapsed && m.maxRun == 12);
	}
}

void CHIRONRepetitionAppendInvariantTest()
{
	glades::chiron::ChironRepetitionConfig cfg;
	std::vector<int> base;
	for (int i = 0; i < 96; ++i) base.push_back((i % 11) < 8 ? (i % 4) : 1000 + i);
	std::vector<int> extended(base);
	for (int i = 0; i < 37; ++i) extended.push_back(7000 + i * 13);

	std::vector<glades::chiron::ChironHazardRow> baseRows, extendedRows;
	std::vector<float> baseWeights, extendedWeights;
	glades::chiron::chiron_repetition_hazards(base, cfg, baseRows, baseWeights);
	glades::chiron::chiron_repetition_hazards(extended, cfg, extendedRows, extendedWeights);
	ASSERT("hazard append row prefix size", extendedRows.size() > baseRows.size());
	for (size_t t = 0; t < baseRows.size(); ++t)
	{
		ASSERT("hazard append count", baseRows[t].count == extendedRows[t].count);
		ASSERT("hazard append overflow", baseRows[t].overflow == extendedRows[t].overflow);
		ASSERT("hazard append weight", baseWeights[t] == extendedWeights[t]);
		for (int i = 0; i < 16; ++i)
		{
			ASSERT("hazard append token", baseRows[t].tokenIds[i] == extendedRows[t].tokenIds[i]);
			ASSERT("hazard append confidence", baseRows[t].confidence[i] == extendedRows[t].confidence[i]);
		}
	}
}

// ---------------------------------------------------------------------------
// Tiny-model fixture helpers for CHIRONTfEvalTest.
// Replicated from chiron-model-test.cpp (ev_* helpers) — do NOT refactor that
// file; these are an independent copy, as required by the task brief.
// ---------------------------------------------------------------------------

static const int TF_T  = 8;
static const int TF_M  = 4;
static const int TF_L  = 2;
static const int TF_NH = 1;
static const int TF_DH = 4;
static const int TF_V  = 16;
static const int TF_DM = 4;  // nH * dH

static glades::gpu::GpuBuffer<float>* tf_alloc_buf(size_t n, float base, float step)
{
	glades::gpu::GpuBuffer<float>* buf = new glades::gpu::GpuBuffer<float>();
	std::vector<float> h(n);
	for (size_t i = 0; i < n; ++i) h[i] = std::sin(step * (float)i + base);
	bool ok = buf->allocate(n) && buf->upload(&h[0], n);
	ASSERT("tf_alloc_buf ok", ok);
	return buf;
}

static void tf_fill_core_weights(glades::chiron::ChironModelWeights& w,
                                 const glades::chiron::ChironModelDims& d)
{
	const size_t Esize   = (size_t)d.V * d.m;
	const size_t Wqkv_sz = (size_t)d.m * d.dModel;
	const size_t Wo_sz   = (size_t)d.dModel * d.m;
	std::vector<float> e(Esize);
	for (size_t i = 0; i < Esize; ++i) e[i] = std::sin(0.05f * (float)i);
	ASSERT("tf E alloc", w.E.allocate(Esize) && w.E.upload(&e[0], Esize));
	for (int l = 0; l < d.L; ++l)
	{
		float off = 0.3f + (float)l * 0.7f;
		w.Wq.push_back(tf_alloc_buf(Wqkv_sz, off + 0.0f, 0.017f));
		w.Wk.push_back(tf_alloc_buf(Wqkv_sz, off + 0.1f, 0.019f));
		w.Wv.push_back(tf_alloc_buf(Wqkv_sz, off + 0.2f, 0.023f));
		w.Wo.push_back(tf_alloc_buf(Wo_sz,   off + 0.3f, 0.029f));
		w.gamma.push_back(tf_alloc_buf((size_t)d.m, off + 0.4f, 0.03f));
		w.beta.push_back( tf_alloc_buf((size_t)d.m, off + 0.5f, 0.04f));
	}
}

static glades::chiron::ChironModelDims tf_dims()
{
	glades::chiron::ChironModelDims d;
	d.T = TF_T; d.m = TF_M; d.L = TF_L; d.nH = TF_NH;
	d.dH = TF_DH; d.V = TF_V; d.dModel = TF_DM;
	return d;
}

// ---------------------------------------------------------------------------
// Test 5: CHIRONTfEvalTest — chiron_tf_eval parity.
//
// Tiny shape: T=8, m=4, L=2, nH=1, dH=4, V=16, dModel=4.
// Dense path + fuseAttnReln (same as CHIRONEvalForwardParityTest Case 1).
// Fixed 8-token window: {1,3,5,7,2,4,6,0}.
// ---------------------------------------------------------------------------
void CHIRONTfEvalTest()
{
	glades::chiron::ChironModelDims d = tf_dims();
	glades::chiron::ChironModelWeights w;
	tf_fill_core_weights(w, d);

	// Dense path + fuseAttnReln (the production serving path for PIED/WhiSC).
	glades::chiron::ChironServingConfig cfg;
	cfg.fuseAttnReln = true;
	cfg.epsReln      = 1e-4f;
	// useScfa=false, qkNorm=false, whiscCoupling=false, fuseAttnPerLayer=false

	// Fixed 8-token window.
	const int tokArr[] = {1, 3, 5, 7, 2, 4, 6, 0};
	std::vector<int> tokens(tokArr, tokArr + TF_T);

	// Allocate eval scratch + run chiron_tf_eval with logitsAllOut.
	glades::chiron::ChironEvalScratch s;
	ASSERT("tf scratch alloc", s.allocate(d, w, cfg));
	std::vector<float> logitsAllOut;
	glades::chiron::ChironTfResult result;
	ASSERT("tf_eval returns true", glades::chiron::chiron_tf_eval(d, w, cfg, s, tokens, result, &logitsAllOut));

	// Reference: run chiron_eval_forward directly, download logits, compute
	// NLL/top1 with the same double-precision log-sum-exp loop.
	glades::chiron::ChironEvalScratch rs;
	ASSERT("tf ref scratch alloc", rs.allocate(d, w, cfg));
	// Pad window (useLen = min(tokens.size(), T) = 8 = T).
	const int useLen = (int)tokens.size() < d.T ? (int)tokens.size() : d.T;
	std::vector<int> input(d.T, 0);
	for (int i = 0; i < useLen; ++i) input[i] = tokens[i];
	ASSERT("tf ref tokens upload", rs.d_tokens.upload(&input[0], d.T));
	ASSERT("tf ref forward", glades::chiron::chiron_eval_forward(d, w, cfg, rs));
	const size_t nLogits = (size_t)d.T * d.V;
	std::vector<float> refLogits(nLogits);
	ASSERT("tf ref logits download", rs.logits.download(&refLogits[0], nLogits));

	// Hand-loop: per-position argmax + double log-sum-exp NLL (same order as chiron_tf_eval).
	long refCorrect = 0, refTotal = 0;
	double refNllSum = 0.0;
	for (int i = 0; i < useLen - 1; ++i)
	{
		const float* row = &refLogits[(size_t)i * d.V];
		int am = 0; float best = row[0];
		double mx = row[0];
		for (int v = 1; v < d.V; ++v) if (row[v] > mx) mx = row[v];
		double Z = 0.0;
		for (int v = 0; v < d.V; ++v) Z += std::exp((double)row[v] - mx);
		for (int v = 1; v < d.V; ++v) if (row[v] > best) { best = row[v]; am = v; }
		const int tgt = tokens[i + 1];
		if (am == tgt) ++refCorrect;
		refNllSum += -((double)row[tgt] - mx - std::log(Z));
		++refTotal;
	}
	const double refTop1Acc = refTotal ? (double)refCorrect / (double)refTotal : 0.0;
	const double refMeanNll = refTotal ? refNllSum / (double)refTotal : 0.0;

	// Structural checks.
	ASSERT("tf positions == 7",        result.positions == 7);
	ASSERT("tf refTotal == 7",         refTotal == 7);

	// Bit-exact equality (same math, same order → same double result).
	ASSERT("tf meanNll bit-equal",     result.meanNll  == refMeanNll);
	ASSERT("tf top1Acc bit-equal",     result.top1Acc  == refTop1Acc);

	// logitsAllOut must equal the direct download element-exact.
	ASSERT("tf logitsAllOut size", logitsAllOut.size() == nLogits);
	for (size_t i = 0; i < nLogits; ++i)
		ASSERT("tf logitsAllOut element", logitsAllOut[i] == refLogits[i]);
}

// ---------------------------------------------------------------------------
// Sink early-stop helper (C++98: static function + plain int context).
// Counts down from initial value; returns true while count > 0 AFTER decrement.
// So with initial count=2: call 1 → true (count=1), call 2 → false (count=0).
// ---------------------------------------------------------------------------
static bool sinkCountdown(void* ctx, int /*token*/)
{
	int* pCount = static_cast<int*>(ctx);
	return --(*pCount) > 0;
}

// ---------------------------------------------------------------------------
// Reference generation loop: same window/slide/clamp/sample logic as
// chiron_generate, run on a SEPARATE scratch so the two paths are independent.
// outRef receives only generated tokens (maxSteps of them).
// ---------------------------------------------------------------------------
static void gen_reference_loop(
	const glades::chiron::ChironModelDims& dims,
	const glades::chiron::ChironModelWeights& w,
	const glades::chiron::ChironServingConfig& cfg,
	glades::chiron::ChironEvalScratch& s,
	const std::vector<int>& promptTokens,
	const glades::chiron::ChironGenParams& gp,
	int maxSteps,
	std::vector<int>& outRef)
{
	outRef.clear();
	std::vector<int>   tokens(promptTokens);
	glades::chiron::ChironMt19937 rng(gp.seed);
	std::vector<int>   input((size_t)dims.T, 0);
	std::vector<float> logitsAll((size_t)dims.T * (size_t)dims.V);
	std::vector<float> logitsRow((size_t)dims.V);

	for (int gen = 0; gen < maxSteps; ++gen)
	{
		const int useLen = (int)tokens.size() < dims.T ? (int)tokens.size() : dims.T;
		for (int i = 0; i < dims.T; ++i) input[i] = 0;
		const int offset = (int)tokens.size() - useLen;
		for (int i = 0; i < useLen; ++i)
		{
			int tk = tokens[offset + i];
			if (tk < 0 || tk >= dims.V) tk = 0;
			input[i] = tk;
		}
		ASSERT("ref upload ok",   s.d_tokens.upload(&input[0], (size_t)dims.T));
		ASSERT("ref forward ok",  glades::chiron::chiron_eval_forward(dims, w, cfg, s));
		ASSERT("ref download ok", s.logits.download(&logitsAll[0], logitsAll.size()));

		const int lastPos = useLen - 1;
		for (int v = 0; v < dims.V; ++v)
			logitsRow[v] = logitsAll[(size_t)lastPos * (size_t)dims.V + v];

		int next = glades::chiron::chiron_sample_token(logitsRow, gp, tokens, rng);
		tokens.push_back(next);
		outRef.push_back(next);
	}
}

// ---------------------------------------------------------------------------
// Test 6a: chiron_generate topK=1 determinism.
// Prompt {1,3,5}, maxTokens=6, topK=1.
// Compare API output against reference loop built from the same primitives.
// topK=1 collapses the distribution to one prob-1.0 candidate, making the
// CDF pick draw-independent: the correct token is always selected regardless
// of the draw value.  This validates deterministic loop mechanics
// (window/clamp/slide) but NOT draw-count parity — see
// CHIRONGenerateStochasticDrawParityTest for that.
// ---------------------------------------------------------------------------
void CHIRONGenerateTopK1Test()
{
	glades::chiron::ChironModelDims d = tf_dims();
	glades::chiron::ChironModelWeights w;
	tf_fill_core_weights(w, d);
	glades::chiron::ChironServingConfig cfg;
	cfg.fuseAttnReln = true;
	cfg.epsReln      = 1e-4f;

	glades::chiron::ChironGenParams gp;
	gp.topK      = 1;
	gp.maxTokens = 6;
	gp.seed      = 1337u;

	const int promptArr[] = {1, 3, 5};
	std::vector<int> prompt(promptArr, promptArr + 3);

	// Two independent scratches — API and reference must not share state.
	glades::chiron::ChironEvalScratch sApi;
	ASSERT("topK1 api scratch alloc", sApi.allocate(d, w, cfg));
	glades::chiron::ChironEvalScratch sRef;
	ASSERT("topK1 ref scratch alloc", sRef.allocate(d, w, cfg));

	// Run chiron_generate.
	std::vector<int> outTokens;
	bool ok = glades::chiron::chiron_generate(d, w, cfg, sApi, prompt, gp, NULL, NULL, &outTokens);
	ASSERT("topK1 generate returns true", ok);
	ASSERT("topK1 outTokens size == 6", (int)outTokens.size() == gp.maxTokens);

	// Run reference.
	std::vector<int> refTokens;
	gen_reference_loop(d, w, cfg, sRef, prompt, gp, gp.maxTokens, refTokens);
	ASSERT("topK1 refTokens size == 6", (int)refTokens.size() == gp.maxTokens);

	// Token-by-token equality.
	for (int i = 0; i < gp.maxTokens; ++i)
	{
		char msg[64]; std::sprintf(msg, "topK1 token[%d] match", i);
		ASSERT(msg, outTokens[i] == refTokens[i]);
	}
}

// ---------------------------------------------------------------------------
// Test 6b: chiron_generate window slide.
// Prompt of 12 tokens (> T=8), maxTokens=3.  Exercises offset = size - T > 0.
// ---------------------------------------------------------------------------
void CHIRONGenerateWindowSlideTest()
{
	glades::chiron::ChironModelDims d = tf_dims();
	glades::chiron::ChironModelWeights w;
	tf_fill_core_weights(w, d);
	glades::chiron::ChironServingConfig cfg;
	cfg.fuseAttnReln = true;
	cfg.epsReln      = 1e-4f;

	glades::chiron::ChironGenParams gp;
	gp.topK      = 1;
	gp.maxTokens = 3;
	gp.seed      = 1337u;

	// 12-token prompt with all IDs in [0, V=16).
	std::vector<int> prompt;
	for (int i = 0; i < 12; ++i) prompt.push_back(i % TF_V);

	glades::chiron::ChironEvalScratch sApi;
	ASSERT("slide api scratch alloc", sApi.allocate(d, w, cfg));
	glades::chiron::ChironEvalScratch sRef;
	ASSERT("slide ref scratch alloc", sRef.allocate(d, w, cfg));

	std::vector<int> outTokens;
	bool ok = glades::chiron::chiron_generate(d, w, cfg, sApi, prompt, gp, NULL, NULL, &outTokens);
	ASSERT("slide generate returns true", ok);
	ASSERT("slide outTokens size == 3", (int)outTokens.size() == gp.maxTokens);

	std::vector<int> refTokens;
	gen_reference_loop(d, w, cfg, sRef, prompt, gp, gp.maxTokens, refTokens);
	ASSERT("slide refTokens size == 3", (int)refTokens.size() == gp.maxTokens);

	for (int i = 0; i < gp.maxTokens; ++i)
	{
		char msg[64]; std::sprintf(msg, "slide token[%d] match", i);
		ASSERT(msg, outTokens[i] == refTokens[i]);
	}
}

// ---------------------------------------------------------------------------
// Test 6c: chiron_generate sink early-stop.
// Sink returns false after the 2nd token.  The 2nd token IS in outTokens
// (already appended before sink is called) but no further iterations run.
// With maxTokens=6, outTokens->size() == 2 proves early stop.
// ---------------------------------------------------------------------------
void CHIRONGenerateSinkStopTest()
{
	glades::chiron::ChironModelDims d = tf_dims();
	glades::chiron::ChironModelWeights w;
	tf_fill_core_weights(w, d);
	glades::chiron::ChironServingConfig cfg;
	cfg.fuseAttnReln = true;
	cfg.epsReln      = 1e-4f;

	glades::chiron::ChironGenParams gp;
	gp.topK      = 1;
	gp.maxTokens = 6;
	gp.seed      = 1337u;

	const int promptArr[] = {1, 3, 5};
	std::vector<int> prompt(promptArr, promptArr + 3);

	glades::chiron::ChironEvalScratch sApi;
	ASSERT("sink api scratch alloc", sApi.allocate(d, w, cfg));

	// count=2: call 1 returns true (count→1), call 2 returns false (count→0).
	int count = 2;
	std::vector<int> outTokens;
	bool ok = glades::chiron::chiron_generate(d, w, cfg, sApi, prompt, gp,
	                                           sinkCountdown, &count, &outTokens);
	ASSERT("sink generate returns true", ok);
	ASSERT("sink outTokens size == 2 (early stop)", (int)outTokens.size() == 2);
}

// ---------------------------------------------------------------------------
// Test 6d: chiron_generate token-id clamp.
// Prompt {-5, 20, 1}: -5 (< 0) and 20 (>= V=16) both clamp to 0.
// Reference applies the same clamp, so both loops must agree.
// ---------------------------------------------------------------------------
void CHIRONGenerateClampTest()
{
	glades::chiron::ChironModelDims d = tf_dims();
	glades::chiron::ChironModelWeights w;
	tf_fill_core_weights(w, d);
	glades::chiron::ChironServingConfig cfg;
	cfg.fuseAttnReln = true;
	cfg.epsReln      = 1e-4f;

	glades::chiron::ChironGenParams gp;
	gp.topK      = 1;
	gp.maxTokens = 3;
	gp.seed      = 1337u;

	const int promptArr[] = {-5, 20, 1};
	std::vector<int> prompt(promptArr, promptArr + 3);

	glades::chiron::ChironEvalScratch sApi;
	ASSERT("clamp api scratch alloc", sApi.allocate(d, w, cfg));
	glades::chiron::ChironEvalScratch sRef;
	ASSERT("clamp ref scratch alloc", sRef.allocate(d, w, cfg));

	std::vector<int> outTokens;
	bool ok = glades::chiron::chiron_generate(d, w, cfg, sApi, prompt, gp, NULL, NULL, &outTokens);
	ASSERT("clamp generate returns true", ok);
	ASSERT("clamp outTokens size == 3", (int)outTokens.size() == gp.maxTokens);

	std::vector<int> refTokens;
	gen_reference_loop(d, w, cfg, sRef, prompt, gp, gp.maxTokens, refTokens);
	ASSERT("clamp refTokens size == 3", (int)refTokens.size() == gp.maxTokens);

	for (int i = 0; i < gp.maxTokens; ++i)
	{
		char msg[64]; std::sprintf(msg, "clamp token[%d] match", i);
		ASSERT(msg, outTokens[i] == refTokens[i]);
	}
}

// ---------------------------------------------------------------------------
// Test 6e: chiron_generate stochastic draw-parity.
// Same tiny fixture, gp.topK=0 / topP=1.0 / temperature=1.0 (full-softmax CDF
// sampling — every pick genuinely depends on the draw value).  Any draw-count
// divergence inside chiron_generate's loop desynchronises the RNG stream and
// produces mismatched tokens.  This is the discriminating test that topK=1
// cases cannot provide (topK=1 picks are draw-independent, so a double- or
// zero-draw bug passes silently there).
// ---------------------------------------------------------------------------
void CHIRONGenerateStochasticDrawParityTest()
{
	glades::chiron::ChironModelDims d = tf_dims();
	glades::chiron::ChironModelWeights w;
	tf_fill_core_weights(w, d);
	glades::chiron::ChironServingConfig cfg;
	cfg.fuseAttnReln = true;
	cfg.epsReln      = 1e-4f;

	glades::chiron::ChironGenParams gp;
	gp.topK        = 0;
	gp.topP        = 1.0f;
	gp.temperature = 1.0f;
	gp.maxTokens   = 6;
	gp.seed        = 1337u;

	const int promptArr[] = {1, 3, 5};
	std::vector<int> prompt(promptArr, promptArr + 3);

	// Two independent scratches — API and reference must not share state.
	glades::chiron::ChironEvalScratch sApi;
	ASSERT("stoch api scratch alloc", sApi.allocate(d, w, cfg));
	glades::chiron::ChironEvalScratch sRef;
	ASSERT("stoch ref scratch alloc", sRef.allocate(d, w, cfg));

	// Run chiron_generate.
	std::vector<int> outTokens;
	bool ok = glades::chiron::chiron_generate(d, w, cfg, sApi, prompt, gp, NULL, NULL, &outTokens);
	ASSERT("stoch generate returns true", ok);
	ASSERT("stoch outTokens size == 6", (int)outTokens.size() == gp.maxTokens);

	// Run reference loop with the same gp/seed on an independent scratch.
	std::vector<int> refTokens;
	gen_reference_loop(d, w, cfg, sRef, prompt, gp, gp.maxTokens, refTokens);
	ASSERT("stoch refTokens size == 6", (int)refTokens.size() == gp.maxTokens);

	// Token-by-token equality.  With full-softmax CDF sampling, any draw-count
	// mismatch in chiron_generate desynchronises the RNG stream and fails here.
	for (int i = 0; i < gp.maxTokens; ++i)
	{
		char msg[64]; std::sprintf(msg, "stoch token[%d] match", i);
		ASSERT(msg, outTokens[i] == refTokens[i]);
	}
}

// ---------------------------------------------------------------------------
// CPU-only and full aggregate entries.
// ---------------------------------------------------------------------------
void CHIRONGenerateCpuUnitTest()
{
	CHIRONMt19937RawTest();
	CHIRONMt19937GoldenTest();
	CHIRONSamplerGoldenTest();
	CHIRONDegenMetricsTest();
	CHIRONRepetitionConfigContractTest();
	CHIRONRepetitionMetricsTest();
	CHIRONRepetitionHazardsTest();
	CHIRONRepetitionAppendInvariantTest();
}

void CHIRONGenerateUnitTest()
{
	CHIRONGenerateCpuUnitTest();
	CHIRONTfEvalTest();
	CHIRONGenerateTopK1Test();
	CHIRONGenerateWindowSlideTest();
	CHIRONGenerateSinkStopTest();
	CHIRONGenerateClampTest();
	CHIRONGenerateStochasticDrawParityTest();
}
