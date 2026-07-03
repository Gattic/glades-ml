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
// Aggregate entry.
// ---------------------------------------------------------------------------
void CHIRONGenerateUnitTest()
{
	CHIRONMt19937RawTest();
	CHIRONMt19937GoldenTest();
	CHIRONSamplerGoldenTest();
	CHIRONDegenMetricsTest();
	CHIRONTfEvalTest();
}
