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

#include "nn-test.h"
#include "../../unit-test.h"
#include "test_token_id_input_fixture.h"

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/transformer_public_api.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/DataObjects/TokenInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Networks/transformer_ops.h"
#include "../../../Backend/Machine Learning/Networks/transformer_kernels.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace {
struct EnvVarGuard
{
	std::string name;
	bool hadOld;
	std::string oldValue;

	explicit EnvVarGuard(const char* n)
	    : name(n ? n : ""),
	      hadOld(false),
	      oldValue()
	{
		if (!name.empty())
		{
			const char* v = ::getenv(name.c_str());
			if (v)
			{
				hadOld = true;
				oldValue = v;
			}
		}
	}

	void set(const char* v)
	{
		if (name.empty())
			return;
#if defined(_WIN32)
		// Windows CRT: set/unset via putenv style.
		// _putenv_s(name, "") unsets.
		(void)::_putenv_s(name.c_str(), v ? v : "");
#else
		(void)::setenv(name.c_str(), v ? v : "", 1);
#endif
	}

	void unset()
	{
		if (name.empty())
			return;
#if defined(_WIN32)
		(void)::_putenv_s(name.c_str(), "");
#else
		(void)::unsetenv(name.c_str());
#endif
	}

	~EnvVarGuard()
	{
		if (name.empty())
			return;
		if (hadOld)
			set(oldValue.c_str());
		else
			unset();
	}

private:
	EnvVarGuard(const EnvVarGuard&);
	EnvVarGuard& operator=(const EnvVarGuard&);
};

struct CaptureEpochMetricsCb : public glades::ITrainingCallbacks
{
	glades::NNetworkEpochMetrics last;
	bool saw;
	CaptureEpochMetricsCb() : last(), saw(false) {}
	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}
};

struct StopAtMinThen100Cb : public glades::ITrainingCallbacks
{
	glades::NNetworkEpochMetrics last;
	bool saw;
	bool reached;
	int minEpochs;
	int maxEpochs;
	StopAtMinThen100Cb(int minE, int maxE) : last(), saw(false), reached(false), minEpochs(minE), maxEpochs(maxE) {}
	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;

		if (maxEpochs > 0 && m.epoch >= maxEpochs)
			return true;

		if (m.epoch >= minEpochs && m.totalAccuracy >= 100.0f - 1e-6f)
		{
			reached = true;
			return true;
		}
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}
};

static bool read_u32_le(std::istream& in, unsigned int& outV)
{
	unsigned char b[4];
	in.read(reinterpret_cast<char*>(b), 4);
	if (!in)
		return false;
	outV = (static_cast<unsigned int>(b[0]) << 0) |
	       (static_cast<unsigned int>(b[1]) << 8) |
	       (static_cast<unsigned int>(b[2]) << 16) |
	       (static_cast<unsigned int>(b[3]) << 24);
	return true;
}

static bool read_u64_le(std::istream& in, unsigned long long& outV)
{
	unsigned char b[8];
	in.read(reinterpret_cast<char*>(b), 8);
	if (!in)
		return false;
	outV =
	    (static_cast<unsigned long long>(b[0]) << 0) |
	    (static_cast<unsigned long long>(b[1]) << 8) |
	    (static_cast<unsigned long long>(b[2]) << 16) |
	    (static_cast<unsigned long long>(b[3]) << 24) |
	    (static_cast<unsigned long long>(b[4]) << 32) |
	    (static_cast<unsigned long long>(b[5]) << 40) |
	    (static_cast<unsigned long long>(b[6]) << 48) |
	    (static_cast<unsigned long long>(b[7]) << 56);
	return true;
}

static bool read_f32_le(std::istream& in, float& outF)
{
	unsigned int bits = 0u;
	if (!read_u32_le(in, bits))
		return false;
	std::memcpy(&outF, &bits, sizeof(float));
	return true;
}

static void write_u32_le(std::ostream& out, unsigned int v)
{
	unsigned char b[4];
	b[0] = static_cast<unsigned char>((v >> 0) & 0xFFu);
	b[1] = static_cast<unsigned char>((v >> 8) & 0xFFu);
	b[2] = static_cast<unsigned char>((v >> 16) & 0xFFu);
	b[3] = static_cast<unsigned char>((v >> 24) & 0xFFu);
	out.write(reinterpret_cast<const char*>(b), 4);
}

static void write_u64_le(std::ostream& out, unsigned long long v)
{
	unsigned char b[8];
	b[0] = static_cast<unsigned char>((v >> 0) & 0xFFull);
	b[1] = static_cast<unsigned char>((v >> 8) & 0xFFull);
	b[2] = static_cast<unsigned char>((v >> 16) & 0xFFull);
	b[3] = static_cast<unsigned char>((v >> 24) & 0xFFull);
	b[4] = static_cast<unsigned char>((v >> 32) & 0xFFull);
	b[5] = static_cast<unsigned char>((v >> 40) & 0xFFull);
	b[6] = static_cast<unsigned char>((v >> 48) & 0xFFull);
	b[7] = static_cast<unsigned char>((v >> 56) & 0xFFull);
	out.write(reinterpret_cast<const char*>(b), 8);
}

static void write_f32_le(std::ostream& out, float f)
{
	unsigned int bits = 0u;
	std::memcpy(&bits, &f, sizeof(float));
	write_u32_le(out, bits);
}

static bool write_vec_f32(std::ostream& out, const std::vector<float>& v)
{
	write_u64_le(out, static_cast<unsigned long long>(v.size()));
	for (size_t i = 0; i < v.size(); ++i)
		write_f32_le(out, v[i]);
	return static_cast<bool>(out);
}

static bool read_and_accum_vec_l2(std::istream& in, double& sumsq)
{
	unsigned long long n = 0ull;
	if (!read_u64_le(in, n))
		return false;
	if (n > (1ull << 31))
		return false;
	for (unsigned long long i = 0; i < n; ++i)
	{
		float f = 0.0f;
		if (!read_f32_le(in, f))
			return false;
		const double d = static_cast<double>(f);
		sumsq += d * d;
	}
	return true;
}

static bool write_weights_header_bin(std::ostream& out, unsigned int netType)
{
	char magic[32];
	std::memset(magic, 0, sizeof(magic));
	const char* want = "GLADES_TENSOR_WEIGHTS_BIN";
	const size_t wantLen = strlen(want);
	std::memcpy(magic, want, (wantLen < sizeof(magic) ? wantLen : sizeof(magic)));
	out.write(magic, 32);
	write_u32_le(out, 1u);        // version
	write_u32_le(out, netType);   // netType
	write_u32_le(out, 0u);        // reserved
	write_u32_le(out, 0u);        // reserved
	return static_cast<bool>(out);
}

static bool transformer_weights_l2_from_file(const std::string& weightsPath, double& outL2)
{
	outL2 = 0.0;
	std::ifstream in(weightsPath.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;

	// header
	char magic[32];
	in.read(magic, 32);
	if (!in)
		return false;
	const char* want = "GLADES_TENSOR_WEIGHTS_BIN";
	const size_t wantLen = strlen(want);
	if (wantLen > sizeof(magic) || std::memcmp(magic, want, wantLen) != 0)
		return false;
	unsigned int version = 0u, netType = 0u, r0 = 0u, r1 = 0u;
	if (!read_u32_le(in, version) || !read_u32_le(in, netType) || !read_u32_le(in, r0) || !read_u32_le(in, r1))
		return false;
	if (version != 1u)
		return false;
	// Transformer net types: encoder=4, decoder=5
	if (!(netType == 4u || netType == 5u))
		return false;

	// transformer scalar config
	unsigned int causal = 0u, nLayers = 0u;
	unsigned int inputSize = 0u, dModel = 0u, dFF = 0u, nHeads = 0u, outSize = 0u;
	if (!read_u32_le(in, causal) || !read_u32_le(in, nLayers) ||
	    !read_u32_le(in, inputSize) || !read_u32_le(in, dModel) || !read_u32_le(in, dFF) ||
	    !read_u32_le(in, nHeads) || !read_u32_le(in, outSize))
		return false;
	{
		unsigned int nKVHeads = 0u, ffnKind = 0u;
		unsigned int tokenModel = 0u, vocabSize = 0u, padTok = 0u, tieEmb = 0u;
		if (!read_u32_le(in, nKVHeads) || !read_u32_le(in, ffnKind))
			return false;
		if (!read_u32_le(in, tokenModel) || !read_u32_le(in, vocabSize) || !read_u32_le(in, padTok) || !read_u32_le(in, tieEmb))
			return false;
		(void)nKVHeads; (void)ffnKind;
		(void)tokenModel; (void)vocabSize; (void)padTok; (void)tieEmb;
	}
	(void)causal; (void)inputSize; (void)dModel; (void)dFF; (void)nHeads; (void)outSize;

	double ss = 0.0;
	// global vectors
	if (!read_and_accum_vec_l2(in, ss)) return false; // WIn
	if (!read_and_accum_vec_l2(in, ss)) return false; // bIn
	if (!read_and_accum_vec_l2(in, ss)) return false; // WOut
	if (!read_and_accum_vec_l2(in, ss)) return false; // bOut
	if (!read_and_accum_vec_l2(in, ss)) return false; // tokE
	if (!read_and_accum_vec_l2(in, ss)) return false; // lmBias
	if (!read_and_accum_vec_l2(in, ss)) return false; // lnFinalGamma
	if (!read_and_accum_vec_l2(in, ss)) return false; // lnFinalBeta
	for (unsigned int l = 0; l < nLayers; ++l)
	{
		// block vectors (fixed order, 18 vectors)
		if (!read_and_accum_vec_l2(in, ss)) return false; // ln1Gamma
		if (!read_and_accum_vec_l2(in, ss)) return false; // ln1Beta
		if (!read_and_accum_vec_l2(in, ss)) return false; // Wq
		if (!read_and_accum_vec_l2(in, ss)) return false; // Wk
		if (!read_and_accum_vec_l2(in, ss)) return false; // Wv
		if (!read_and_accum_vec_l2(in, ss)) return false; // Wo
		if (!read_and_accum_vec_l2(in, ss)) return false; // bq
		if (!read_and_accum_vec_l2(in, ss)) return false; // bk
		if (!read_and_accum_vec_l2(in, ss)) return false; // bv
		if (!read_and_accum_vec_l2(in, ss)) return false; // bo
		if (!read_and_accum_vec_l2(in, ss)) return false; // ln2Gamma
		if (!read_and_accum_vec_l2(in, ss)) return false; // ln2Beta
		if (!read_and_accum_vec_l2(in, ss)) return false; // W1
		if (!read_and_accum_vec_l2(in, ss)) return false; // b1
		if (!read_and_accum_vec_l2(in, ss)) return false; // W2
		if (!read_and_accum_vec_l2(in, ss)) return false; // b2
	}
	outL2 = sqrt(ss);
	return true;
}

static void mkdir_if_missing(const std::string& path)
{
	// Best-effort (matches style used by other UTs).
	::mkdir(path.c_str(), 0777);
}

static std::vector<float> tokE_identity(unsigned int vocab, unsigned int dModel)
{
	std::vector<float> E(static_cast<size_t>(vocab) * static_cast<size_t>(dModel), 0.0f);
	const unsigned int diag = (vocab < dModel) ? vocab : dModel;
	for (unsigned int i = 0; i < diag; ++i)
		E[static_cast<size_t>(i) * static_cast<size_t>(dModel) + i] = 1.0f;
	return E;
}

static float sinusoidal_pe(unsigned int pos, unsigned int i, unsigned int dModel)
{
	// Match the formula used by the training path (sgd_transformer.cpp) and inference path (transformer_infer.cpp).
	const unsigned int idx = i / 2u;
	const double exponent = (2.0 * static_cast<double>(idx)) / static_cast<double>(dModel);
	const double denom = pow(10000.0, exponent);
	const double angle = static_cast<double>(pos) / denom;
	const double v = ((i % 2u) == 0u) ? sin(angle) : cos(angle);
	return static_cast<float>(v);
}

static bool write_transformer_decoder_tokenlm_weights(const std::string& weightsPath,
                                                     unsigned int nLayers,
                                                     unsigned int dModel,
                                                     unsigned int dFF,
                                                     unsigned int nHeads,
                                                     unsigned int nKVHeads,
                                                     unsigned int vocabSize,
                                                     unsigned int padTokenId,
                                                     unsigned int ffnKind, // TransformerRunConfig::FFNKind
                                                     const std::vector<float>& tokE, // [vocab, dModel]
                                                     const std::vector<float>& lmBias, // [vocab]
                                                     bool zeroAllBlocks)
{
	if (nLayers == 0u || dModel == 0u || dFF == 0u || nHeads == 0u || nKVHeads == 0u || vocabSize == 0u)
		return false;
	if ((dModel % nHeads) != 0u)
		return false;
	if ((nHeads % nKVHeads) != 0u)
		return false;

	const unsigned int dHead = dModel / nHeads;
	const unsigned int dModelKV = nKVHeads * dHead;
	const unsigned int ff1Width = (ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFF) : dFF;

	if (tokE.size() != static_cast<size_t>(vocabSize) * static_cast<size_t>(dModel))
		return false;
	if (!lmBias.empty() && lmBias.size() != static_cast<size_t>(vocabSize))
		return false;

	std::ofstream fp(weightsPath.c_str(), std::ios::out | std::ios::binary);
	if (!fp)
		return false;
	// netType=5 (Transformer decoder)
	if (!write_weights_header_bin(fp, /*netType*/ 5u))
		return false;

	// Transformer scalar config (must match `network.cpp` save/load order).
	write_u32_le(fp, /*causal*/ 1u);
	write_u32_le(fp, /*nLayers*/ nLayers);
	write_u32_le(fp, /*inputSize*/ 1u);
	write_u32_le(fp, /*dModel*/ dModel);
	write_u32_le(fp, /*dFF*/ dFF);
	write_u32_le(fp, /*nHeads*/ nHeads);
	write_u32_le(fp, /*outSize*/ vocabSize);
	write_u32_le(fp, /*nKVHeads*/ nKVHeads);
	write_u32_le(fp, /*ffnKind*/ ffnKind);
	write_u32_le(fp, /*tokenModel*/ 1u);
	write_u32_le(fp, /*vocabSize*/ vocabSize);
	write_u32_le(fp, /*padTokenId*/ padTokenId);
	write_u32_le(fp, /*tieEmbeddings*/ 1u);

	// Global vectors: WIn, bIn, WOut, bOut, tokE, lmBias
	{
		std::vector<float> WIn(static_cast<size_t>(dModel) * 1u, 0.0f);
		std::vector<float> bIn(static_cast<size_t>(dModel), 0.0f);
		std::vector<float> WOut(static_cast<size_t>(vocabSize) * static_cast<size_t>(dModel), 0.0f);
		std::vector<float> bOut(static_cast<size_t>(vocabSize), 0.0f);
		std::vector<float> bLm = lmBias;
		if (bLm.empty())
			bLm.assign(vocabSize, 0.0f);

		if (!write_vec_f32(fp, WIn)) return false;
		if (!write_vec_f32(fp, bIn)) return false;
		if (!write_vec_f32(fp, WOut)) return false;
		if (!write_vec_f32(fp, bOut)) return false;
		if (!write_vec_f32(fp, tokE)) return false;
		if (!write_vec_f32(fp, bLm)) return false;
	}

	// Final LayerNorm (gamma=1, beta=0 => identity)
	{
		std::vector<float> lnFinalGamma(dModel, 1.0f);
		std::vector<float> lnFinalBeta(dModel, 0.0f);
		if (!write_vec_f32(fp, lnFinalGamma)) return false;
		if (!write_vec_f32(fp, lnFinalBeta)) return false;
	}

	for (unsigned int l = 0; l < nLayers; ++l)
	{
		// ln1Gamma/beta
		std::vector<float> ln1Gamma(dModel, 1.0f);
		std::vector<float> ln1Beta(dModel, 0.0f);
		// projections
		std::vector<float> Wq(static_cast<size_t>(dModel) * static_cast<size_t>(dModel), 0.0f);
		std::vector<float> Wk(static_cast<size_t>(dModelKV) * static_cast<size_t>(dModel), 0.0f);
		std::vector<float> Wv(static_cast<size_t>(dModelKV) * static_cast<size_t>(dModel), 0.0f);
		std::vector<float> Wo(static_cast<size_t>(dModel) * static_cast<size_t>(dModel), 0.0f);
		std::vector<float> bq(dModel, 0.0f);
		std::vector<float> bk(dModelKV, 0.0f);
		std::vector<float> bv(dModelKV, 0.0f);
		std::vector<float> bo(dModel, 0.0f);
		// ln2Gamma/beta
		std::vector<float> ln2Gamma(dModel, 1.0f);
		std::vector<float> ln2Beta(dModel, 0.0f);
		// ffn
		std::vector<float> W1(static_cast<size_t>(ff1Width) * static_cast<size_t>(dModel), 0.0f);
		std::vector<float> b1(ff1Width, 0.0f);
		std::vector<float> W2(static_cast<size_t>(dModel) * static_cast<size_t>(dFF), 0.0f);
		std::vector<float> b2(dModel, 0.0f);

		// Optionally make blocks "non-trivial" (for tests that want to ensure blocks are actually loaded).
		// Otherwise, leaving them all-zero is a useful "no-op block" baseline.
		if (!zeroAllBlocks)
		{
			// A tiny deterministic perturbation so the block changes h but stays numerically stable.
			// Wq/Wk/Wv/Wo will remain 0 for simplicity; we use FFN biases only.
			if (b1.size() >= 1u) b1[0] = 0.01f;
			if (b2.size() >= 1u) b2[0] = 0.01f;
		}

		if (!write_vec_f32(fp, ln1Gamma)) return false;
		if (!write_vec_f32(fp, ln1Beta)) return false;
		if (!write_vec_f32(fp, Wq)) return false;
		if (!write_vec_f32(fp, Wk)) return false;
		if (!write_vec_f32(fp, Wv)) return false;
		if (!write_vec_f32(fp, Wo)) return false;
		if (!write_vec_f32(fp, bq)) return false;
		if (!write_vec_f32(fp, bk)) return false;
		if (!write_vec_f32(fp, bv)) return false;
		if (!write_vec_f32(fp, bo)) return false;
		if (!write_vec_f32(fp, ln2Gamma)) return false;
		if (!write_vec_f32(fp, ln2Beta)) return false;
		if (!write_vec_f32(fp, W1)) return false;
		if (!write_vec_f32(fp, b1)) return false;
		if (!write_vec_f32(fp, W2)) return false;
		if (!write_vec_f32(fp, b2)) return false;
	}

	return static_cast<bool>(fp);
}

static bool write_dff_weights_dense(const std::string& weightsPath,
                                   unsigned int in,
                                   unsigned int out,
                                   const std::vector<float>& W_rowMajor_out_in,
                                   const std::vector<float>& bias)
{
	if (in == 0u || out == 0u)
		return false;
	if (W_rowMajor_out_in.size() != static_cast<size_t>(in) * static_cast<size_t>(out))
		return false;
	if (bias.size() != static_cast<size_t>(out))
		return false;

	std::ofstream fp(weightsPath.c_str(), std::ios::out | std::ios::binary);
	if (!fp)
		return false;

	if (!write_weights_header_bin(fp, /*netType*/ 0u))
		return false;

	// transitions
	write_u32_le(fp, 1u);
	// transition 0
	write_u32_le(fp, in);
	write_u32_le(fp, out);
	if (!write_vec_f32(fp, W_rowMajor_out_in))
		return false;
	if (!write_vec_f32(fp, bias))
		return false;
	return static_cast<bool>(fp);
}

static bool write_dff_weights(const std::string& weightsPath, float w, float b)
{
	std::vector<float> W(1, w);
	std::vector<float> bias(1, b);
	return write_dff_weights_dense(weightsPath, /*in*/ 1u, /*out*/ 1u, W, bias);
}

static bool read_first_dff_weight(const std::string& weightsPath, float& outW)
{
	outW = 0.0f;
	std::ifstream in(weightsPath.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;

	char magic[32];
	in.read(magic, 32);
	if (!in)
		return false;
	const char* want = "GLADES_TENSOR_WEIGHTS_BIN";
	const size_t wantLen = strlen(want);
	if (wantLen > sizeof(magic) || std::memcmp(magic, want, wantLen) != 0)
		return false;
	unsigned int version = 0u, netType = 0u, r0 = 0u, r1 = 0u;
	if (!read_u32_le(in, version) || !read_u32_le(in, netType) || !read_u32_le(in, r0) || !read_u32_le(in, r1))
		return false;
	if (version != 1u || netType != 0u)
		return false;

	unsigned int transitions = 0u;
	if (!read_u32_le(in, transitions) || transitions == 0u)
		return false;
	unsigned int inSize = 0u, outSize = 0u;
	if (!read_u32_le(in, inSize) || !read_u32_le(in, outSize))
		return false;
	(void)inSize; (void)outSize;

	// W vector: read count then first float
	unsigned long long n = 0ull;
	if (!read_u64_le(in, n) || n == 0ull)
		return false;
	if (!read_f32_le(in, outW))
		return false;
	return true;
}

static bool write_rnn_weights_1x1x1(const std::string& weightsPath,
                                   float Wxh, float Whh, float bh,
                                   float Why, float by)
{
	std::ofstream out(weightsPath.c_str(), std::ios::out | std::ios::binary);
	if (!out)
		return false;

	if (!write_weights_header_bin(out, /*netType*/ 1u))
		return false;
	// hiddenLayers, inputSize, outSize
	write_u32_le(out, 1u);
	write_u32_le(out, 1u);
	write_u32_le(out, 1u);
	// hidden layer 0: in=1, h=1
	write_u32_le(out, 1u);
	write_u32_le(out, 1u);
	{
		std::vector<float> v(1);
		v[0] = Wxh;
		if (!write_vec_f32(out, v)) return false;
		v[0] = Whh;
		if (!write_vec_f32(out, v)) return false;
		v[0] = bh;
		if (!write_vec_f32(out, v)) return false;
	}
	// output: in=1, out=1
	write_u32_le(out, 1u);
	write_u32_le(out, 1u);
	{
		std::vector<float> v(1);
		v[0] = Why;
		if (!write_vec_f32(out, v)) return false;
		v[0] = by;
		if (!write_vec_f32(out, v)) return false;
	}
	return static_cast<bool>(out);
}

static bool read_rnn_out_bias_1x1x1(const std::string& weightsPath, float& outBy)
{
	outBy = 0.0f;
	std::ifstream in(weightsPath.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;

	// header
	char magic[32];
	in.read(magic, 32);
	if (!in)
		return false;
	const char* want = "GLADES_TENSOR_WEIGHTS_BIN";
	const size_t wantLen = strlen(want);
	if (wantLen > sizeof(magic) || std::memcmp(magic, want, wantLen) != 0)
		return false;
	unsigned int version = 0u, netType = 0u, r0 = 0u, r1 = 0u;
	if (!read_u32_le(in, version) || !read_u32_le(in, netType) || !read_u32_le(in, r0) || !read_u32_le(in, r1))
		return false;
	if (version != 1u || netType != 1u)
		return false;

	// rnn header: hiddenLayers, inputSize, outSize
	unsigned int hiddenLayers = 0u, inputSize = 0u, outSize = 0u;
	if (!read_u32_le(in, hiddenLayers) || !read_u32_le(in, inputSize) || !read_u32_le(in, outSize))
		return false;
	(void)inputSize;

	// hidden layer(s)
	for (unsigned int l = 0u; l < hiddenLayers; ++l)
	{
		unsigned int inSize = 0u, hSize = 0u;
		if (!read_u32_le(in, inSize) || !read_u32_le(in, hSize))
			return false;
		(void)inSize; (void)hSize;
		// Wxh, Whh, bh
		double dummy = 0.0;
		if (!read_and_accum_vec_l2(in, dummy)) return false;
		if (!read_and_accum_vec_l2(in, dummy)) return false;
		if (!read_and_accum_vec_l2(in, dummy)) return false;
	}

	// output: in, out, Why, by
	unsigned int oIn = 0u, oOut = 0u;
	if (!read_u32_le(in, oIn) || !read_u32_le(in, oOut))
		return false;
	(void)oIn; (void)oOut;
	{
		double dummy = 0.0;
		if (!read_and_accum_vec_l2(in, dummy)) return false; // Why
	}
	{
		unsigned long long n = 0ull;
		if (!read_u64_le(in, n) || n != static_cast<unsigned long long>(outSize))
			return false;
		// outSize==1 in this test helper
		if (!read_f32_le(in, outBy))
			return false;
	}
	return true;
}

static bool write_gated_weights_1layer_1x1x1(const std::string& weightsPath,
                                            unsigned int netType,
                                            unsigned int gateCount,
                                            float Why,
                                            float by)
{
	std::ofstream out(weightsPath.c_str(), std::ios::out | std::ios::binary);
	if (!out)
		return false;
	if (!write_weights_header_bin(out, netType))
		return false;
	// gated header: gateCount, hiddenLayers, inputSize, outSize
	write_u32_le(out, gateCount);
	write_u32_le(out, 1u);
	write_u32_le(out, 1u);
	write_u32_le(out, 1u);
	// hidden layer 0: in=1, h=1
	write_u32_le(out, 1u);
	write_u32_le(out, 1u);
	{
		std::vector<float> v(gateCount, 0.0f);
		if (!write_vec_f32(out, v)) return false; // W
		if (!write_vec_f32(out, v)) return false; // U
		if (!write_vec_f32(out, v)) return false; // bias
	}
	// output: in=1, out=1
	write_u32_le(out, 1u);
	write_u32_le(out, 1u);
	{
		std::vector<float> v(1);
		v[0] = Why;
		if (!write_vec_f32(out, v)) return false;
		v[0] = by;
		if (!write_vec_f32(out, v)) return false;
	}
	return static_cast<bool>(out);
}

static bool read_gated_out_bias_1layer_1x1x1(const std::string& weightsPath,
                                            unsigned int wantNetType,
                                            float& outBy)
{
	outBy = 0.0f;
	std::ifstream in(weightsPath.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;

	// header
	char magic[32];
	in.read(magic, 32);
	if (!in)
		return false;
	const char* want = "GLADES_TENSOR_WEIGHTS_BIN";
	const size_t wantLen = strlen(want);
	if (wantLen > sizeof(magic) || std::memcmp(magic, want, wantLen) != 0)
		return false;
	unsigned int version = 0u, netType = 0u, r0 = 0u, r1 = 0u;
	if (!read_u32_le(in, version) || !read_u32_le(in, netType) || !read_u32_le(in, r0) || !read_u32_le(in, r1))
		return false;
	if (version != 1u || netType != wantNetType)
		return false;

	// gated header: gateCount, hiddenLayers, inputSize, outSize
	unsigned int gateCount = 0u, hiddenLayers = 0u, inputSize = 0u, outSize = 0u;
	if (!read_u32_le(in, gateCount) || !read_u32_le(in, hiddenLayers) || !read_u32_le(in, inputSize) || !read_u32_le(in, outSize))
		return false;
	(void)gateCount; (void)inputSize;

	for (unsigned int l = 0u; l < hiddenLayers; ++l)
	{
		unsigned int inSize = 0u, hSize = 0u;
		if (!read_u32_le(in, inSize) || !read_u32_le(in, hSize))
			return false;
		(void)inSize; (void)hSize;
		// W, U, bias
		double dummy = 0.0;
		if (!read_and_accum_vec_l2(in, dummy)) return false;
		if (!read_and_accum_vec_l2(in, dummy)) return false;
		if (!read_and_accum_vec_l2(in, dummy)) return false;
	}

	// output: in, out, Why, by
	unsigned int oIn = 0u, oOut = 0u;
	if (!read_u32_le(in, oIn) || !read_u32_le(in, oOut))
		return false;
	(void)oIn; (void)oOut;
	{
		double dummy = 0.0;
		if (!read_and_accum_vec_l2(in, dummy)) return false; // Why
	}
	{
		unsigned long long n = 0ull;
		if (!read_u64_le(in, n) || n != static_cast<unsigned long long>(outSize))
			return false;
		// outSize==1 in this test helper
		if (!read_f32_le(in, outBy))
			return false;
	}
	return true;
}

static bool read_last_result_pred(const glades::NNetwork& net, float& outPred)
{
	outPred = 0.0f;
	const shmea::GList r = net.getResults();
	if (r.size() < 2u)
		return false;
	// Results are stored as [expectation, prediction] for the last processed sample/timestep.
	outPred = r.getFloat(1);
	return true;
}

static glades::NNetwork load_with_overridden_weights_DFF(const glades::NNInfo* info,
                                                        const glades::NumberInput* di,
                                                        const std::string& modelName,
                                                        unsigned int seed,
                                                        float w,
                                                        float b)
{
	// These override-style tests intentionally patch weights.bin after saveModel().
	// Ensure file integrity verification is disabled regardless of caller environment.
	EnvVarGuard verify("GLADES_MODEL_VERIFY_FILES");
	verify.unset();

	// 1) Create a model package from the architecture.
	glades::NNetwork bootstrap(info, glades::NNetwork::TYPE_DFF);
	bootstrap.setSeed(seed);
	bootstrap.getTerminatorMutable().setEpoch(1);
	bootstrap.getTerminatorMutable().setAccuracy(0);
	// Ensure test split is present (the runtime uses test split for net.test()).
	// These unit tests often only populate trainMatrix/trainExpectedMatrix.
	glades::NumberInput* diMut = const_cast<glades::NumberInput*>(di);
	if (diMut && diMut->getTestSize() == 0u && diMut->getTrainSize() > 0u)
	{
		diMut->testMatrix = diMut->trainMatrix;
		diMut->testExpectedMatrix = diMut->trainExpectedMatrix;
	}

	const glades::NNetworkStatus stInit = bootstrap.test(di);
	G_assert(__FILE__, __LINE__, "==============NN::DFF_InitTestStatus() Failed==============", stInit.ok());
	const glades::NNetworkStatus stSave0 = bootstrap.saveModel(modelName);
	G_assert(__FILE__, __LINE__, "==============NN::DFF_SaveBootstrapModel() Failed==============", stSave0.ok());

	// 2) Override weights.bin with deterministic packed tensors.
	G_assert(__FILE__, __LINE__, "==============NN::DFF_WriteOverrideWeights() Failed==============",
	         write_dff_weights("database/models/" + modelName + "/weights.bin", w, b));

	// 3) Load into a fresh net (ensures runtime uses the overridden packed weights).
	glades::NNetwork net(glades::NNetwork::TYPE_DFF);
	const glades::NNetworkStatus stLoad = net.loadModel(modelName, di);
	G_assert(__FILE__, __LINE__, "==============NN::DFF_LoadOverrideModel() Failed==============", stLoad.ok());
	return net;
}

static glades::NNetwork load_with_overridden_weights_RNN_1x1x1(const glades::NNInfo* info,
                                                               const glades::NumberInput* di,
                                                               const std::string& modelName,
                                                               unsigned int seed,
                                                               float Wxh, float Whh, float bh,
                                                               float Why, float by)
{
	// These override-style tests intentionally patch weights.bin after saveModel().
	// Ensure file integrity verification is disabled regardless of caller environment.
	EnvVarGuard verify("GLADES_MODEL_VERIFY_FILES");
	verify.unset();

	glades::NNetwork bootstrap(info, glades::NNetwork::TYPE_RNN);
	bootstrap.setSeed(seed);
	bootstrap.getTerminatorMutable().setEpoch(1);
	bootstrap.getTerminatorMutable().setAccuracy(0);
	glades::NumberInput* diMut = const_cast<glades::NumberInput*>(di);
	if (diMut && diMut->getTestSize() == 0u && diMut->getTrainSize() > 0u)
	{
		diMut->testMatrix = diMut->trainMatrix;
		diMut->testExpectedMatrix = diMut->trainExpectedMatrix;
	}

	const glades::NNetworkStatus stInit = bootstrap.test(di);
	G_assert(__FILE__, __LINE__, "==============NN::RNN_InitTestStatus() Failed==============", stInit.ok());
	const glades::NNetworkStatus stSave0 = bootstrap.saveModel(modelName);
	G_assert(__FILE__, __LINE__, "==============NN::RNN_SaveBootstrapModel() Failed==============", stSave0.ok());

	G_assert(__FILE__, __LINE__, "==============NN::RNN_WriteOverrideWeights() Failed==============",
	         write_rnn_weights_1x1x1("database/models/" + modelName + "/weights.bin", Wxh, Whh, bh, Why, by));

	glades::NNetwork net(glades::NNetwork::TYPE_RNN);
	const glades::NNetworkStatus stLoad = net.loadModel(modelName, di);
	G_assert(__FILE__, __LINE__, "==============NN::RNN_LoadOverrideModel() Failed==============", stLoad.ok());
	return net;
}

static glades::NNetwork load_with_overridden_weights_Gated_1layer_1x1x1(const glades::NNInfo* info,
                                                                        const glades::NumberInput* di,
                                                                        const std::string& modelName,
                                                                        int netType,
                                                                        unsigned int seed,
                                                                        unsigned int gateCount,
                                                                        float Why,
                                                                        float by)
{
	// Ensure file integrity verification is disabled regardless of caller environment.
	EnvVarGuard verify("GLADES_MODEL_VERIFY_FILES");
	verify.unset();

	glades::NNetwork bootstrap(info, netType);
	bootstrap.setSeed(seed);
	bootstrap.getTerminatorMutable().setEpoch(1);
	bootstrap.getTerminatorMutable().setAccuracy(0);
	glades::NumberInput* diMut = const_cast<glades::NumberInput*>(di);
	if (diMut && diMut->getTestSize() == 0u && diMut->getTrainSize() > 0u)
	{
		diMut->testMatrix = diMut->trainMatrix;
		diMut->testExpectedMatrix = diMut->trainExpectedMatrix;
	}

	const glades::NNetworkStatus stInit = bootstrap.test(di);
	G_assert(__FILE__, __LINE__, "==============NN::Gated_InitTestStatus() Failed==============", stInit.ok());
	const glades::NNetworkStatus stSave0 = bootstrap.saveModel(modelName);
	G_assert(__FILE__, __LINE__, "==============NN::Gated_SaveBootstrapModel() Failed==============", stSave0.ok());

	G_assert(__FILE__, __LINE__, "==============NN::Gated_WriteOverrideWeights() Failed==============",
	         write_gated_weights_1layer_1x1x1("database/models/" + modelName + "/weights.bin",
	                                          static_cast<unsigned int>(netType),
	                                          gateCount,
	                                          Why,
	                                          by));

	glades::NNetwork net(netType);
	const glades::NNetworkStatus stLoad = net.loadModel(modelName, di);
	G_assert(__FILE__, __LINE__, "==============NN::Gated_LoadOverrideModel() Failed==============", stLoad.ok());
	return net;
}

static bool text_file_contains_prefix(const std::string& path, const std::string& prefix)
{
	std::ifstream in(path.c_str());
	if (!in)
		return false;
	std::string line;
	while (std::getline(in, line))
	{
		if (line.size() >= prefix.size() && std::memcmp(line.data(), prefix.data(), prefix.size()) == 0)
			return true;
	}
	return false;
}

static void ModelPackageIntegrityVerificationUnitTest()
{
	EnvVarGuard verify("GLADES_MODEL_VERIFY_FILES");
	verify.set("1"); // opt-in verification

	glades::NumberInput* di = new glades::NumberInput();
	di->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
	di->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
	di->trainMatrix[0][0] = 1.0f;
	di->trainExpectedMatrix[0][0] = 0.0f;
	// Make test split available.
	di->testMatrix = di->trainMatrix;
	di->testExpectedMatrix = di->trainExpectedMatrix;

	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    /*batchSize*/ 1,
	    /*learningRate*/ 0.1f,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::LINEAR,
	    /*activationParam*/ 1.0f);
	std::vector<glades::HiddenLayerInfo*> hidden;
	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
	glades::NNInfo* info = new glades::NNInfo("ut_pkg_verify_integrity", in, hidden, out);

	const std::string modelName = "ut_pkg_verify_integrity";
	{
		glades::NNetwork bootstrap(info, glades::NNetwork::TYPE_DFF);
		bootstrap.setSeed(123u);
		bootstrap.getTerminatorMutable().setEpoch(1);
		bootstrap.getTerminatorMutable().setAccuracy(0);
		const glades::NNetworkStatus stInit = bootstrap.test(di);
		G_assert(__FILE__, __LINE__, "==============NN::PkgVerify_InitTestStatus() Failed==============", stInit.ok());
		const glades::NNetworkStatus stSave = bootstrap.saveModel(modelName);
		G_assert(__FILE__, __LINE__, "==============NN::PkgVerify_SaveModel() Failed==============", stSave.ok());
	}

	// Ensure v3+ packaging fields exist (best-effort; do not hardcode exact version).
	G_assert(__FILE__, __LINE__, "==============NN::PkgVerify_ManifestMissingWeightsHash() Failed==============",
	         text_file_contains_prefix("database/models/" + modelName + "/manifest.txt", "weights.fnv1a64="));
	G_assert(__FILE__, __LINE__, "==============NN::PkgVerify_ManifestMissingWeightsBytes() Failed==============",
	         text_file_contains_prefix("database/models/" + modelName + "/manifest.txt", "weights.bytes="));

	// Baseline: load must succeed with verification enabled.
	{
		glades::NNetwork net(glades::NNetwork::TYPE_DFF);
		const glades::NNetworkStatus stLoad = net.loadModel(modelName, di);
		G_assert(__FILE__, __LINE__, "==============NN::PkgVerify_LoadBaseline() Failed==============", stLoad.ok());
	}

	// Tamper weights.bin without updating manifest: load must fail when verification is enabled.
	G_assert(__FILE__, __LINE__, "==============NN::PkgVerify_TamperWeights() Failed==============",
	         write_dff_weights("database/models/" + modelName + "/weights.bin", /*w*/ 2.0f, /*b*/ 0.0f));
	{
		glades::NNetwork net(glades::NNetwork::TYPE_DFF);
		const glades::NNetworkStatus stLoad = net.loadModel(modelName, di);
		G_assert(__FILE__, __LINE__, "==============NN::PkgVerify_LoadTamperedShouldFail() Failed==============", !stLoad.ok());
	}

	delete di;
	delete info;
}
} // namespace

static void NumberInputTrainTestSplitUnitTest();

void NNUnitTest()
{
	printf("============================================================\n");
	printf("NN Unit Test Suite (modern-only)\n");
	printf("============================================================\n");

	printf("-----------------------------------\n");
	printf("NN Test: model package integrity verification\n");
	printf("-----------------------------------\n");
	{
		ModelPackageIntegrityVerificationUnitTest();
		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
	}

    printf("-----------------------------------\n");
	printf("NN Test A (DFF minibatch end-of-batch update)\n");
    printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->trainMatrix[0][0] = 1.0f;
		di->trainExpectedMatrix[0][0] = 0.0f;
		di->trainMatrix[1][0] = 2.0f;
		di->trainExpectedMatrix[1][0] = 1.0f;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
            /*batchSize*/ 10,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_minibatch_timing", in, hidden, out);

		glades::NNetwork net = load_with_overridden_weights_DFF(info, di, "ut_pkg_dff_minibatch", 123u, /*w*/ 1.0f, /*b*/ 0.0f);
		net.getTerminatorMutable().setEpoch(1);
		net.getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net.train(di);
		G_assert(__FILE__, __LINE__, "==============NN::DFF_Minibatch TrainStatus() Failed==============", st.ok());
		const glades::NNetworkStatus stSave = net.saveModel("ut_pkg_dff_minibatch_after");
		G_assert(__FILE__, __LINE__, "==============NN::DFF_Minibatch SaveModel() Failed==============", stSave.ok());

		float wFinal = 0.0f;
		const bool okW = read_first_dff_weight("database/models/ut_pkg_dff_minibatch_after/weights.bin", wFinal);
		G_assert(__FILE__, __LINE__, "==============NN::DFF_Minibatch ReadWeight() Failed==============", okW);
        const float expectedW = 0.7f;
        const float tol = 1e-3f;
		printf("[UT] DFF minibatch final weight = %f (expected ~%f)\n", wFinal, expectedW);
		G_assert(__FILE__, __LINE__, "==============NN::DFF_MinibatchUpdate() Failed==============",
                 (wFinal > expectedW - tol) && (wFinal < expectedW + tol));

		delete di;
		delete info;
    }

    printf("-----------------------------------\n");
	printf("NN Test B (DFF weight decay L2)\n");
    printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
            /*batchSize*/ 1,
		    /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 1.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_weight_decay_l2", in, hidden, out);

		glades::NNetwork net = load_with_overridden_weights_DFF(info, di, "ut_pkg_dff_l2", 321u, /*w*/ 1.0f, /*b*/ 0.0f);
		net.getTerminatorMutable().setEpoch(1);
		net.getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net.train(di);
		G_assert(__FILE__, __LINE__, "==============NN::DFF_L2 TrainStatus() Failed==============", st.ok());
		const glades::NNetworkStatus stSave = net.saveModel("ut_pkg_dff_l2_after");
		G_assert(__FILE__, __LINE__, "==============NN::DFF_L2 SaveModel() Failed==============", stSave.ok());

		float wFinal = 0.0f;
		const bool okW = read_first_dff_weight("database/models/ut_pkg_dff_l2_after/weights.bin", wFinal);
		G_assert(__FILE__, __LINE__, "==============NN::DFF_L2 ReadWeight() Failed==============", okW);
        const float expectedW = 0.9f;
        const float tol = 1e-3f;
		printf("[UT] DFF L2 final weight = %f (expected ~%f)\n", wFinal, expectedW);
		G_assert(__FILE__, __LINE__, "==============NN::DFF_L2WeightDecay() Failed==============",
                 (wFinal > expectedW - tol) && (wFinal < expectedW + tol));

		delete di;
		delete info;
    }

    printf("-----------------------------------\n");
	printf("NN Test C (DFF weight decay L1)\n");
    printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));

        const float lr = 0.1f;
        const float lambda1 = 1.0f;
        const float tol = 1e-3f;

		// w0=+1 => w1=0.9
        {
            glades::InputLayerInfo* in = new glades::InputLayerInfo(
                /*batchSize*/ 1,
                /*learningRate*/ lr,
                /*momentumFactor*/ 0.0f,
                /*weightDecay1*/ lambda1,
                /*weightDecay2*/ 0.0f,
                /*pDropout*/ 0.0f,
                /*activationType*/ glades::GMath::LINEAR,
			    /*activationParam*/ 1.0f);
            std::vector<glades::HiddenLayerInfo*> hidden;
            glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
            glades::NNInfo* info = new glades::NNInfo("ut_weight_decay_l1_pos", in, hidden, out);
			glades::NNetwork net = load_with_overridden_weights_DFF(info, di, "ut_pkg_dff_l1_pos", 777u, /*w*/ 1.0f, /*b*/ 0.0f);
            net.getTerminatorMutable().setEpoch(1);
            net.getTerminatorMutable().setAccuracy(0);
			G_assert(__FILE__, __LINE__, "==============NN::DFF_L1(+1) TrainStatus() Failed==============", net.train(di).ok());
			G_assert(__FILE__, __LINE__, "==============NN::DFF_L1(+1) SaveModel() Failed==============", net.saveModel("ut_pkg_dff_l1_pos_after").ok());
			float wFinal = 0.0f;
			G_assert(__FILE__, __LINE__, "==============NN::DFF_L1(+1) ReadWeight() Failed==============",
			         read_first_dff_weight("database/models/ut_pkg_dff_l1_pos_after/weights.bin", wFinal));
            const float expectedW = 0.9f;
			printf("[UT] DFF L1(+1) final weight = %f (expected ~%f)\n", wFinal, expectedW);
			G_assert(__FILE__, __LINE__, "==============NN::DFF_L1(+1) WeightDecay() Failed==============",
                     (wFinal > expectedW - tol) && (wFinal < expectedW + tol));
            delete info;
        }

		// w0=-1 => w1=-0.9
        {
            glades::InputLayerInfo* in = new glades::InputLayerInfo(
                /*batchSize*/ 1,
                /*learningRate*/ lr,
                /*momentumFactor*/ 0.0f,
                /*weightDecay1*/ lambda1,
                /*weightDecay2*/ 0.0f,
                /*pDropout*/ 0.0f,
                /*activationType*/ glades::GMath::LINEAR,
			    /*activationParam*/ 1.0f);
            std::vector<glades::HiddenLayerInfo*> hidden;
            glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
            glades::NNInfo* info = new glades::NNInfo("ut_weight_decay_l1_neg", in, hidden, out);
			glades::NNetwork net = load_with_overridden_weights_DFF(info, di, "ut_pkg_dff_l1_neg", 778u, /*w*/ -1.0f, /*b*/ 0.0f);
            net.getTerminatorMutable().setEpoch(1);
            net.getTerminatorMutable().setAccuracy(0);
			G_assert(__FILE__, __LINE__, "==============NN::DFF_L1(-1) TrainStatus() Failed==============", net.train(di).ok());
			G_assert(__FILE__, __LINE__, "==============NN::DFF_L1(-1) SaveModel() Failed==============", net.saveModel("ut_pkg_dff_l1_neg_after").ok());
			float wFinal = 0.0f;
			G_assert(__FILE__, __LINE__, "==============NN::DFF_L1(-1) ReadWeight() Failed==============",
			         read_first_dff_weight("database/models/ut_pkg_dff_l1_neg_after/weights.bin", wFinal));
            const float expectedW = -0.9f;
			printf("[UT] DFF L1(-1) final weight = %f (expected ~%f)\n", wFinal, expectedW);
			G_assert(__FILE__, __LINE__, "==============NN::DFF_L1(-1) WeightDecay() Failed==============",
                     (wFinal > expectedW - tol) && (wFinal < expectedW + tol));
            delete info;
        }

		delete di;
    }

    printf("-----------------------------------\n");
	printf("NN Test D (Regression metrics on test)\n");
    printf("-----------------------------------\n");
    {
        class CaptureMetricsCb : public glades::ITrainingCallbacks
        {
        public:
			glades::NNetworkEpochMetrics last;
			bool saw;
			CaptureMetricsCb() : last(), saw(false) {}
			virtual void onRunStart(const glades::NNetwork&, int) {}
            virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
            {
				last = m;
				saw = true;
                return false;
            }
			virtual void onRunEnd(const glades::NNetwork&, int) {}
		};

		glades::NumberInput* di = new glades::NumberInput();
		di->testMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->testExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->testMatrix[0][0] = 1.0f;
		di->testExpectedMatrix[0][0] = 2.0f;
		di->testMatrix[1][0] = 2.0f;
		di->testExpectedMatrix[1][0] = 4.0f;

		// Mirror test->train to keep the DataInput fully populated.
		di->trainMatrix = di->testMatrix;
		di->trainExpectedMatrix = di->testExpectedMatrix;

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
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_reg_metrics", in, hidden, out);

		glades::NNetwork net = load_with_overridden_weights_DFF(info, di, "ut_pkg_dff_reg_metrics", 999u, /*w*/ 1.0f, /*b*/ 0.0f);

        CaptureMetricsCb cb;
		const glades::NNetworkStatus st = net.test(di, &cb);
		G_assert(__FILE__, __LINE__, "==============NN::RegMetrics TestStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============NN::RegMetrics SawMetrics() Failed==============", cb.saw);

		const float expMSE = 2.5f;
		const float expMAE = 1.5f;
		const float expRMSE = static_cast<float>(sqrt(2.5));
		const float tol = 1e-4f;
		if (cb.saw)
		{
			printf("[UT] Reg metrics: MSE=%f MAE=%f RMSE=%f\n", cb.last.totalError, cb.last.regMAE, cb.last.regRMSE);
			G_assert(__FILE__, __LINE__, "==============NN::RegMetrics MSE() Failed==============", fabs(cb.last.totalError - expMSE) < tol);
			G_assert(__FILE__, __LINE__, "==============NN::RegMetrics MAE() Failed==============", fabs(cb.last.regMAE - expMAE) < tol);
			G_assert(__FILE__, __LINE__, "==============NN::RegMetrics RMSE() Failed==============", fabs(cb.last.regRMSE - expRMSE) < tol);
		}

		delete di;
		delete info;
    }

    printf("-----------------------------------\n");
	printf("NN Test E (Grad clip disabled => scale=1)\n");
    printf("-----------------------------------\n");
    {
        class CaptureMetricsCb : public glades::ITrainingCallbacks
        {
        public:
            glades::NNetworkEpochMetrics last;
            bool saw;
			CaptureMetricsCb() : last(), saw(false) {}
			virtual void onRunStart(const glades::NNetwork&, int) {}
            virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
            {
                last = m;
                saw = true;
                return false;
            }
			virtual void onRunEnd(const glades::NNetwork&, int) {}
		};

		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(1, 0.0f));
		di->trainMatrix[0][0] = 10.0f;
		di->trainExpectedMatrix[0][0] = 0.0f;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
            /*batchSize*/ 1,
            /*learningRate*/ 0.1f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_grad_clip_disabled", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.getTerminatorMutable().setEpoch(1);
		net.getTerminatorMutable().setAccuracy(0);

        CaptureMetricsCb cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		G_assert(__FILE__, __LINE__, "==============NN::GradClipDisabled TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============NN::GradClipDisabled SawMetrics() Failed==============", cb.saw);
        if (cb.saw)
        {
			G_assert(__FILE__, __LINE__, "==============NN::GradClipDisabled ScaleIsOne() Failed==============", fabs(cb.last.gradNormScale - 1.0f) < 1e-6f);
			G_assert(__FILE__, __LINE__, "==============NN::GradClipDisabled NormZero() Failed==============", fabs(cb.last.gradNorm - 0.0f) < 1e-6f);
		}

		delete di;
		delete info;
    }

    printf("-----------------------------------\n");
	printf("NN Test F (DFF binary classification trains from Xavier to 100%%)\n");
    printf("-----------------------------------\n");
	{
		// 2D linearly separable classification with a *narrow margin*.
		// We intentionally use many points near the decision boundary so Xavier init is very unlikely to be perfect.
		glades::NumberInput* di = new glades::NumberInput();
		const unsigned int N = 40u;
		di->trainMatrix = shmea::GMatrix(N, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(N, shmea::GVector<float>(2, 0.0f));
		for (unsigned int i = 0; i < N / 2u; ++i)
		{
			const float y = static_cast<float>(static_cast<int>(i) - 9);
			// class 0: x slightly negative
			di->trainMatrix[i][0] = -0.1f;
			di->trainMatrix[i][1] = y;
			di->trainExpectedMatrix[i][0] = 1.0f;
			di->trainExpectedMatrix[i][1] = 0.0f;
		}
		for (unsigned int i = N / 2u; i < N; ++i)
		{
			const float y = static_cast<float>(static_cast<int>(i - N / 2u) - 9);
			// class 1: x slightly positive
			di->trainMatrix[i][0] = 0.1f;
			di->trainMatrix[i][1] = y;
			di->trainExpectedMatrix[i][0] = 0.0f;
			di->trainExpectedMatrix[i][1] = 1.0f;
		}
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ static_cast<int>(N),
		    /*learningRate*/ 0.15f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(2, glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_class_bin_train_100", in, hidden, out);

		// Pick a deterministic seed where Xavier init is NOT already perfect.
		glades::NNetwork* net = NULL;
		float initAcc = 0.0f;
		const uint64_t seeds[] = {424242u, 424243u, 424244u, 424245u, 424246u, 424247u};
		const int seedCount = static_cast<int>(sizeof(seeds) / sizeof(seeds[0]));
		for (int si = 0; si < seedCount; ++si)
		{
			glades::NNetwork* cand = new glades::NNetwork(info, glades::NNetwork::TYPE_DFF);
			cand->setSeed(seeds[si]);
			cand->getTerminatorMutable().setEpoch(0);
			cand->getTerminatorMutable().setAccuracy(0.0f);
			CaptureEpochMetricsCb cb0;
			G_assert(__FILE__, __LINE__, "==============NN::ClassBin InitTest() Failed==============", cand->test(di, &cb0).ok());
			G_assert(__FILE__, __LINE__, "==============NN::ClassBin InitTest SawMetrics() Failed==============", cb0.saw);
			initAcc = cb0.last.totalAccuracy;
			if (initAcc < 100.0f - 1e-6f)
			{
				net = cand;
				break;
			}
			delete cand;
		}
		G_assert(__FILE__, __LINE__, "==============NN::ClassBin NonPerfectInitSeedNotFound() Failed==============", net != NULL);
		printf("[UT] Class(bin) initAccuracy=%f\n", initAcc);

		StopAtMinThen100Cb cb(/*minEpochs*/ 25, /*maxEpochs*/ 5000);
		const glades::NNetworkStatus st = net->train(di, &cb);
		G_assert(__FILE__, __LINE__, "==============NN::ClassBin TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============NN::ClassBin SawMetrics() Failed==============", cb.saw);
		G_assert(__FILE__, __LINE__, "==============NN::ClassBin Reached100() Failed==============", cb.reached);
		if (cb.saw)
		{
			printf("[UT] Class(bin) epochs=%d totalAccuracy=%f\n", cb.last.epoch, cb.last.totalAccuracy);
			G_assert(__FILE__, __LINE__, "==============NN::ClassBin EpochsGE25() Failed==============", cb.last.epoch >= 25);
			G_assert(__FILE__, __LINE__, "==============NN::ClassBin AccuracyIs100() Failed==============", fabs(cb.last.totalAccuracy - 100.0f) < 1e-6f);
		}
		delete net;

		delete di;
		delete info;
    }

    printf("-----------------------------------\n");
	printf("NN Test G (DFF 3-class identity trains from Xavier to 100%%)\n");
    printf("-----------------------------------\n");
	{
		// 3-class identity-like mapping with many samples per class.
		// This is linearly separable but very unlikely to be perfect at Xavier init.
		glades::NumberInput* di = new glades::NumberInput();
		const unsigned int perClass = 10u;
		const unsigned int N = 3u * perClass;
		di->trainMatrix = shmea::GMatrix(N, shmea::GVector<float>(3, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(N, shmea::GVector<float>(3, 0.0f));
		for (unsigned int k = 0; k < 3u; ++k)
		{
			for (unsigned int j = 0; j < perClass; ++j)
			{
				const unsigned int r = k * perClass + j;
				const float t = static_cast<float>(static_cast<int>(j) - 5) * 0.02f;
				// Base direction is one-hot, with tiny deterministic "shape" noise in other coords.
				di->trainMatrix[r][k] = 1.0f;
				di->trainMatrix[r][(k + 1u) % 3u] = t;
				di->trainMatrix[r][(k + 2u) % 3u] = -t;
				di->trainExpectedMatrix[r][k] = 1.0f;
			}
		}
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ static_cast<int>(N),
		    /*learningRate*/ 0.10f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(3, glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_class_3way_train_100", in, hidden, out);

		glades::NNetwork* net = NULL;
		float initAcc = 0.0f;
		const uint64_t seeds[] = {424250u, 424251u, 424252u, 424253u, 424254u, 424255u};
		const int seedCount = static_cast<int>(sizeof(seeds) / sizeof(seeds[0]));
		for (int si = 0; si < seedCount; ++si)
		{
			glades::NNetwork* cand = new glades::NNetwork(info, glades::NNetwork::TYPE_DFF);
			cand->setSeed(seeds[si]);
			cand->getTerminatorMutable().setEpoch(0);
			cand->getTerminatorMutable().setAccuracy(0.0f);
			CaptureEpochMetricsCb cb0;
			G_assert(__FILE__, __LINE__, "==============NN::Class3 InitTest() Failed==============", cand->test(di, &cb0).ok());
			G_assert(__FILE__, __LINE__, "==============NN::Class3 InitTest SawMetrics() Failed==============", cb0.saw);
			initAcc = cb0.last.totalAccuracy;
			if (initAcc < 100.0f - 1e-6f)
			{
				net = cand;
				break;
			}
			delete cand;
		}
		G_assert(__FILE__, __LINE__, "==============NN::Class3 NonPerfectInitSeedNotFound() Failed==============", net != NULL);
		printf("[UT] Class(3-way) initAccuracy=%f\n", initAcc);

		StopAtMinThen100Cb cb(/*minEpochs*/ 25, /*maxEpochs*/ 8000);
		const glades::NNetworkStatus st = net->train(di, &cb);
		G_assert(__FILE__, __LINE__, "==============NN::Class3 TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============NN::Class3 SawMetrics() Failed==============", cb.saw);
		G_assert(__FILE__, __LINE__, "==============NN::Class3 Reached100() Failed==============", cb.reached);
        if (cb.saw)
        {
			printf("[UT] Class(3-way) epochs=%d totalAccuracy=%f\n", cb.last.epoch, cb.last.totalAccuracy);
			G_assert(__FILE__, __LINE__, "==============NN::Class3 EpochsGE25() Failed==============", cb.last.epoch >= 25);
			G_assert(__FILE__, __LINE__, "==============NN::Class3 AccuracyIs100() Failed==============", fabs(cb.last.totalAccuracy - 100.0f) < 1e-6f);
		}
		delete net;

		delete di;
		delete info;
    }

    printf("-----------------------------------\n");
	printf("NN Test H (DFF hidden-layer classification trains from Xavier to 100%%)\n");
    printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		// XOR-style data cloud (not linearly separable), requires the hidden layer to solve.
		// We generate a small deterministic cloud around each corner to make "perfect at init" extremely unlikely.
		const unsigned int perCorner = 4u;
		const unsigned int N = 4u * perCorner;
		di->trainMatrix = shmea::GMatrix(N, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(N, shmea::GVector<float>(2, 0.0f));
		unsigned int r = 0u;
		for (unsigned int j = 0; j < perCorner; ++j)
		{
			const float t = static_cast<float>(static_cast<int>(j) - 2) * 0.05f;
			// (0,0) -> class 0
			di->trainMatrix[r][0] = 0.0f + t; di->trainMatrix[r][1] = 0.0f - t;
			di->trainExpectedMatrix[r][0] = 1.0f; di->trainExpectedMatrix[r][1] = 0.0f; ++r;
			// (0,1) -> class 1
			di->trainMatrix[r][0] = 0.0f + t; di->trainMatrix[r][1] = 1.0f - t;
			di->trainExpectedMatrix[r][0] = 0.0f; di->trainExpectedMatrix[r][1] = 1.0f; ++r;
			// (1,0) -> class 1
			di->trainMatrix[r][0] = 1.0f - t; di->trainMatrix[r][1] = 0.0f + t;
			di->trainExpectedMatrix[r][0] = 0.0f; di->trainExpectedMatrix[r][1] = 1.0f; ++r;
			// (1,1) -> class 0
			di->trainMatrix[r][0] = 1.0f - t; di->trainMatrix[r][1] = 1.0f + t;
			di->trainExpectedMatrix[r][0] = 1.0f; di->trainExpectedMatrix[r][1] = 0.0f; ++r;
		}
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ static_cast<int>(N),
		    /*learningRate*/ 0.20f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::SIGMOID,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 4,
		    /*learningRate*/ 0.20f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(2, glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_class_hidden_train_100", in, hidden, out);

		glades::NNetwork* net = NULL;
		float initAcc = 0.0f;
		const uint64_t seeds[] = {424260u, 424261u, 424262u, 424263u, 424264u, 424265u};
		const int seedCount = static_cast<int>(sizeof(seeds) / sizeof(seeds[0]));
		for (int si = 0; si < seedCount; ++si)
		{
			glades::NNetwork* cand = new glades::NNetwork(info, glades::NNetwork::TYPE_DFF);
			cand->setSeed(seeds[si]);
			cand->getTerminatorMutable().setEpoch(0);
			cand->getTerminatorMutable().setAccuracy(0.0f);
			CaptureEpochMetricsCb cb0;
			G_assert(__FILE__, __LINE__, "==============NN::ClassHidden InitTest() Failed==============", cand->test(di, &cb0).ok());
			G_assert(__FILE__, __LINE__, "==============NN::ClassHidden InitTest SawMetrics() Failed==============", cb0.saw);
			initAcc = cb0.last.totalAccuracy;
			if (initAcc < 100.0f - 1e-6f)
			{
				net = cand;
				break;
			}
			delete cand;
		}
		G_assert(__FILE__, __LINE__, "==============NN::ClassHidden NonPerfectInitSeedNotFound() Failed==============", net != NULL);
		printf("[UT] Class(hidden) initAccuracy=%f\n", initAcc);

		StopAtMinThen100Cb cb(/*minEpochs*/ 50, /*maxEpochs*/ 15000);
		const glades::NNetworkStatus st = net->train(di, &cb);
		G_assert(__FILE__, __LINE__, "==============NN::ClassHidden TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============NN::ClassHidden SawMetrics() Failed==============", cb.saw);
		G_assert(__FILE__, __LINE__, "==============NN::ClassHidden Reached100() Failed==============", cb.reached);
        if (cb.saw)
        {
			printf("[UT] Class(hidden) epochs=%d totalAccuracy=%f\n", cb.last.epoch, cb.last.totalAccuracy);
			G_assert(__FILE__, __LINE__, "==============NN::ClassHidden EpochsGE50() Failed==============", cb.last.epoch >= 50);
			G_assert(__FILE__, __LINE__, "==============NN::ClassHidden AccuracyIs100() Failed==============", fabs(cb.last.totalAccuracy - 100.0f) < 1e-6f);
        }
		delete net;

		delete di;
		delete info;
    }

	NumberInputTrainTestSplitUnitTest();

    printf("\n============================================================\n");
}

static void NumberInputTrainTestSplitUnitTest()
{
	printf("-----------------------------------\n");
	printf("NumberInput train/test split (fit-on-train) smoke test\n");
	printf("-----------------------------------\n");

	// Build a tiny in-memory dataset:
	// - 2 numeric inputs
	// - 1 string label output column (binary)
	shmea::GVector<shmea::GString> headers;
	headers.push_back("x1");
	headers.push_back("x2");
	headers.push_back("label");
	shmea::GTable tbl(',', headers);
	tbl.toggleOutput(2u);
	for (int i = 0; i < 20; ++i)
	{
		shmea::GList row;
		row.addFloat((i < 10) ? -1.0f : 1.0f);
		row.addFloat(static_cast<float>(i));
		row.addString((i < 10) ? "A" : "B");
		tbl.addRow(row);
	}

	glades::NumberInput di;
	glades::NumberInput::TrainTestSplitConfig cfg;
	cfg.testFraction = 0.25f;
	cfg.shuffle = true;
	cfg.stratify = true;
	cfg.seed = 2026u;

	const bool ok = di.importWithSplit(tbl, cfg, glades::GMath::ZSCORE);
	G_assert(__FILE__, __LINE__, "==============NumberInput::importWithSplit() Failed==============", ok);
	G_assert(__FILE__, __LINE__, "==============NumberInput Split TrainSize NonZero Failed==============", di.getTrainSize() > 0u);
	G_assert(__FILE__, __LINE__, "==============NumberInput Split TestSize NonZero Failed==============", di.getTestSize() > 0u);
	G_assert(__FILE__, __LINE__, "==============NumberInput Split TotalRows Match Failed==============", (di.getTrainSize() + di.getTestSize()) == tbl.numberOfRows());
	G_assert(__FILE__, __LINE__, "==============NumberInput Split FeatureCount Match Failed==============", di.getFeatureCount() == 2u);

	// Expected output should be 2-wide (A/B).
	const float* y = NULL;
	unsigned int yN = 0u;
	G_assert(__FILE__, __LINE__, "==============NumberInput Split ExpectedRowView Failed==============", di.getTrainExpectedRowView(0u, y, yN));
	G_assert(__FILE__, __LINE__, "==============NumberInput Split ExpectedDims Failed==============", yN == 2u);
}

void NNRecurrentUnitTest()
{
    printf("============================================================\n");
	printf("NN Recurrent Test Suite (modern-only)\n");
    printf("============================================================\n");

    printf("-----------------------------------\n");
	printf("RNN forward recurrence (1x1x1)\n");
    printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->testMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->testExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->testMatrix[0][0] = 2.0f;
		di->testMatrix[1][0] = 3.0f;
		// Expected outputs for the configured weights:
		// h1 = x1 = 2, y1 = 2
		// h2 = x2 + h1 = 3 + 2 = 5, y2 = 5
		di->testExpectedMatrix[0][0] = 2.0f;
		di->testExpectedMatrix[1][0] = 5.0f;
		di->trainMatrix = di->testMatrix;
		di->trainExpectedMatrix = di->testExpectedMatrix;

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
            /*size*/ 1,
            /*learningRate*/ 0.0f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_rnn_forward", in, hidden, out);

		glades::NNetwork net = load_with_overridden_weights_RNN_1x1x1(info, di, "ut_pkg_rnn_forward", 4242u,
		                                                              /*Wxh*/ 1.0f, /*Whh*/ 1.0f, /*bh*/ 0.0f,
		                                                              /*Why*/ 1.0f, /*by*/ 0.0f);
		const glades::NNetworkStatus st = net.test(di);
		G_assert(__FILE__, __LINE__, "==============NN::RNN Forward TestStatus() Failed==============", st.ok());

		float pred = 0.0f;
		G_assert(__FILE__, __LINE__, "==============NN::RNN Forward GetPred() Failed==============", read_last_result_pred(net, pred));
		// h1=2, h2=3 + 2 = 5, y2=5
		const float expected = 5.0f;
		const float tol = 1e-4f;
		printf("[UT] RNN last pred = %f (expected %f)\n", pred, expected);
		G_assert(__FILE__, __LINE__, "==============NN::RNN Forward PredMismatch() Failed==============", fabs(pred - expected) < tol);

		// End-to-end persistence: save model, reload, and re-check the same prediction.
		G_assert(__FILE__, __LINE__, "==============NN::RNN Forward SaveModel() Failed==============",
		         net.saveModel("ut_pkg_rnn_forward_after").ok());
		glades::NNetwork net2(glades::NNetwork::TYPE_RNN);
		G_assert(__FILE__, __LINE__, "==============NN::RNN Forward ReloadModel() Failed==============",
		         net2.loadModel("ut_pkg_rnn_forward_after", di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::RNN Forward Reload TestStatus() Failed==============", net2.test(di).ok());
		float pred2 = 0.0f;
		G_assert(__FILE__, __LINE__, "==============NN::RNN Forward Reload GetPred() Failed==============", read_last_result_pred(net2, pred2));
		printf("[UT] RNN reload last pred = %f (expected %f)\n", pred2, expected);
		G_assert(__FILE__, __LINE__, "==============NN::RNN Forward Reload PredMismatch() Failed==============", fabs(pred2 - expected) < tol);

		delete di;
		delete info;
    }

	printf("-----------------------------------\n");
	printf("RNN minibatch semantics (average by sequences/windows, not timesteps)\n");
	printf("-----------------------------------\n");
	{
		// Two sequences with different lengths:
		// - seq0 length 1 has expected=1
		// - seq1 length 3 has expected=0,0,0
		//
		// With y=0 and MSE dL/dy = 2(y - y*) this yields per-timestep deltas:
		// - seq0: -2
		// - seq1:  0
		//
		// Correct "sample averaging" semantics:
		// mean(seq means) = (-2 + 0)/2 = -1  => by -= lr * (-1) => by = +1 (with lr=1, by0=0)
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix[0][0] = 1.0f;
		di->trainExpectedMatrix[1][0] = 0.0f;
		di->trainExpectedMatrix[2][0] = 0.0f;
		di->trainExpectedMatrix[3][0] = 0.0f;
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		std::vector<glades::DataInput::SequenceSpan> spans;
		spans.push_back(glades::DataInput::SequenceSpan(0u, 1u));
		spans.push_back(glades::DataInput::SequenceSpan(1u, 3u));
		G_assert(__FILE__, __LINE__, "==============NN::RNN Minibatch SetTrainSequences Failed==============", di->setTrainSequences(spans));
		G_assert(__FILE__, __LINE__, "==============NN::RNN Minibatch SetTestSequences Failed==============", di->setTestSequences(spans));

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 2,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 1,
		    // NOTE: In the current NNInfo indexing scheme, recurrent output hyperparams
		    // are retrieved via layer index == numHiddenLayers, which maps to the *last hidden*
		    // layer (NNInfo does not expose output-layer LR/momentum/decay via getLearningRate()).
		    // Therefore, we set the hidden-layer learning rate here to drive the output update.
		    /*learningRate*/ 1.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_rnn_minibatch_semantics", in, hidden, out);

		glades::NNetwork net = load_with_overridden_weights_RNN_1x1x1(info, di, "ut_pkg_rnn_minibatch_semantics", 1234u,
		                                                              /*Wxh*/ 0.0f, /*Whh*/ 0.0f, /*bh*/ 0.0f,
		                                                              /*Why*/ 0.0f, /*by*/ 0.0f);
		net.getTerminatorMutable().setEpoch(1);
		net.getTerminatorMutable().setAccuracy(0);
		G_assert(__FILE__, __LINE__, "==============NN::RNN Minibatch TrainStatus() Failed==============", net.train(di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::RNN Minibatch SaveModel() Failed==============", net.saveModel("ut_pkg_rnn_minibatch_semantics_after").ok());

		float byFinal = 0.0f;
		G_assert(__FILE__, __LINE__, "==============NN::RNN Minibatch ReadOutBias() Failed==============",
		         read_rnn_out_bias_1x1x1("database/models/ut_pkg_rnn_minibatch_semantics_after/weights.bin", byFinal));
		const float expectedBy = 1.0f;
		const float tol = 1e-3f;
		printf("[UT] RNN minibatch final out bias = %f (expected ~%f)\n", byFinal, expectedBy);
		G_assert(__FILE__, __LINE__, "==============NN::RNN Minibatch OutBiasMismatch() Failed==============",
		         (byFinal > expectedBy - tol) && (byFinal < expectedBy + tol));

		delete di;
		delete info;
	}

	printf("-----------------------------------\n");
	printf("GRU minibatch semantics (average by sequences/windows, not timesteps)\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix[0][0] = 1.0f;
		di->trainExpectedMatrix[1][0] = 0.0f;
		di->trainExpectedMatrix[2][0] = 0.0f;
		di->trainExpectedMatrix[3][0] = 0.0f;
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		std::vector<glades::DataInput::SequenceSpan> spans;
		spans.push_back(glades::DataInput::SequenceSpan(0u, 1u));
		spans.push_back(glades::DataInput::SequenceSpan(1u, 3u));
		G_assert(__FILE__, __LINE__, "==============NN::GRU Minibatch SetTrainSequences Failed==============", di->setTrainSequences(spans));
		G_assert(__FILE__, __LINE__, "==============NN::GRU Minibatch SetTestSequences Failed==============", di->setTestSequences(spans));

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 2,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 1,
		    /*learningRate*/ 1.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_gru_minibatch_semantics", in, hidden, out);

		glades::NNetwork net = load_with_overridden_weights_Gated_1layer_1x1x1(info, di, "ut_pkg_gru_minibatch_semantics",
		                                                                       glades::NNetwork::TYPE_GRU,
		                                                                       2233u,
		                                                                       /*gateCount*/ 3u,
		                                                                       /*Why*/ 0.0f,
		                                                                       /*by*/ 0.0f);
		net.getTerminatorMutable().setEpoch(1);
		net.getTerminatorMutable().setAccuracy(0);
		G_assert(__FILE__, __LINE__, "==============NN::GRU Minibatch TrainStatus() Failed==============", net.train(di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::GRU Minibatch SaveModel() Failed==============", net.saveModel("ut_pkg_gru_minibatch_semantics_after").ok());

		float byFinal = 0.0f;
		G_assert(__FILE__, __LINE__, "==============NN::GRU Minibatch ReadOutBias() Failed==============",
		         read_gated_out_bias_1layer_1x1x1("database/models/ut_pkg_gru_minibatch_semantics_after/weights.bin",
		                                          static_cast<unsigned int>(glades::NNetwork::TYPE_GRU),
		                                          byFinal));
		const float expectedBy = 1.0f;
		const float tol = 1e-3f;
		printf("[UT] GRU minibatch final out bias = %f (expected ~%f)\n", byFinal, expectedBy);
		G_assert(__FILE__, __LINE__, "==============NN::GRU Minibatch OutBiasMismatch() Failed==============",
		         (byFinal > expectedBy - tol) && (byFinal < expectedBy + tol));

		delete di;
		delete info;
	}

	printf("-----------------------------------\n");
	printf("LSTM minibatch semantics (average by sequences/windows, not timesteps)\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix[0][0] = 1.0f;
		di->trainExpectedMatrix[1][0] = 0.0f;
		di->trainExpectedMatrix[2][0] = 0.0f;
		di->trainExpectedMatrix[3][0] = 0.0f;
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		std::vector<glades::DataInput::SequenceSpan> spans;
		spans.push_back(glades::DataInput::SequenceSpan(0u, 1u));
		spans.push_back(glades::DataInput::SequenceSpan(1u, 3u));
		G_assert(__FILE__, __LINE__, "==============NN::LSTM Minibatch SetTrainSequences Failed==============", di->setTrainSequences(spans));
		G_assert(__FILE__, __LINE__, "==============NN::LSTM Minibatch SetTestSequences Failed==============", di->setTestSequences(spans));

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 2,
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 1,
		    /*learningRate*/ 1.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_lstm_minibatch_semantics", in, hidden, out);

		glades::NNetwork net = load_with_overridden_weights_Gated_1layer_1x1x1(info, di, "ut_pkg_lstm_minibatch_semantics",
		                                                                       glades::NNetwork::TYPE_LSTM,
		                                                                       3344u,
		                                                                       /*gateCount*/ 4u,
		                                                                       /*Why*/ 0.0f,
		                                                                       /*by*/ 0.0f);
		net.getTerminatorMutable().setEpoch(1);
		net.getTerminatorMutable().setAccuracy(0);
		G_assert(__FILE__, __LINE__, "==============NN::LSTM Minibatch TrainStatus() Failed==============", net.train(di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::LSTM Minibatch SaveModel() Failed==============", net.saveModel("ut_pkg_lstm_minibatch_semantics_after").ok());

		float byFinal = 0.0f;
		G_assert(__FILE__, __LINE__, "==============NN::LSTM Minibatch ReadOutBias() Failed==============",
		         read_gated_out_bias_1layer_1x1x1("database/models/ut_pkg_lstm_minibatch_semantics_after/weights.bin",
		                                          static_cast<unsigned int>(glades::NNetwork::TYPE_LSTM),
		                                          byFinal));
		const float expectedBy = 1.0f;
		const float tol = 1e-3f;
		printf("[UT] LSTM minibatch final out bias = %f (expected ~%f)\n", byFinal, expectedBy);
		G_assert(__FILE__, __LINE__, "==============NN::LSTM Minibatch OutBiasMismatch() Failed==============",
		         (byFinal > expectedBy - tol) && (byFinal < expectedBy + tol));

		delete di;
		delete info;
	}

	printf("-----------------------------------\n");
	printf("GRU smoke test (save/load)\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		// A single sequence length 3.
		di->trainMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
		for (int t = 0; t < 3; ++t)
		{
			di->trainMatrix[t][0] = static_cast<float>(t);
			di->trainExpectedMatrix[t][0] = static_cast<float>(t);
		}
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.05f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 2,
		    /*learningRate*/ 0.05f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_gru_smoke", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_GRU);
		net.setSeed(9001u);
		net.getTerminatorMutable().setEpoch(2);
		net.getTerminatorMutable().setAccuracy(0);
		G_assert(__FILE__, __LINE__, "==============NN::GRU Smoke TrainStatus() Failed==============", net.train(di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::GRU Smoke SaveModel() Failed==============", net.saveModel("ut_pkg_gru_smoke").ok());

		glades::NNetwork net2(glades::NNetwork::TYPE_GRU);
		G_assert(__FILE__, __LINE__, "==============NN::GRU Smoke LoadModel() Failed==============", net2.loadModel("ut_pkg_gru_smoke", di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::GRU Smoke TestStatus() Failed==============", net2.test(di).ok());

		delete di;
		delete info;
	}

	printf("-----------------------------------\n");
	printf("LSTM smoke test (save/load)\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
		for (int t = 0; t < 3; ++t)
		{
			di->trainMatrix[t][0] = static_cast<float>(t);
			di->trainExpectedMatrix[t][0] = static_cast<float>(t);
		}
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.05f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 2,
		    /*learningRate*/ 0.05f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_lstm_smoke", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_LSTM);
		net.setSeed(9002u);
		net.getTerminatorMutable().setEpoch(2);
		net.getTerminatorMutable().setAccuracy(0);
		G_assert(__FILE__, __LINE__, "==============NN::LSTM Smoke TrainStatus() Failed==============", net.train(di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::LSTM Smoke SaveModel() Failed==============", net.saveModel("ut_pkg_lstm_smoke").ok());

		glades::NNetwork net2(glades::NNetwork::TYPE_LSTM);
		G_assert(__FILE__, __LINE__, "==============NN::LSTM Smoke LoadModel() Failed==============", net2.loadModel("ut_pkg_lstm_smoke", di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::LSTM Smoke TestStatus() Failed==============", net2.test(di).ok());

		delete di;
		delete info;
	}

    printf("\n============================================================\n");
}

void NNTransformerUnitTest()
{
    printf("============================================================\n");
	printf("NN Transformer Test Suite (modern-only)\n");
	printf("============================================================\n");

	// ===== Correctness gates (5) =====
	//
	// These are "stop-the-line" invariants. They are intentionally tiny and deterministic.
	// If any of these fail, inference/training correctness is not trustworthy.
	printf("-----------------------------------\n");
	printf("Transformer correctness gates (5)\n");
	printf("-----------------------------------\n");
	{
		// Gate 1: KV-cache incremental decode matches full forward logits (RoPE path).
		{
			const unsigned int vocab = 17u;
			const unsigned int padTokenId = vocab - 1u;
			const unsigned int T = 5u;
			std::vector<unsigned int> toks;
			toks.push_back(1u);
			toks.push_back(2u);
			toks.push_back(3u);
			toks.push_back(4u);
			toks.push_back(5u);

			InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
			di->setTrainTokens(toks, static_cast<int>(padTokenId));
			di->mirrorTrainToTest();

			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f)); // dModel=16
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_gate_kv_parity_rope", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.setSeed(4242u);
			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
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

			G_assert(__FILE__, __LINE__, "==============Gate1 InitTestStatus Failed==============", net.test(di).ok());
			glades::NNetwork::TransformerLmSession session;
			G_assert(__FILE__, __LINE__, "==============Gate1 SessionReset Failed==============", net.transformerLmSessionReset(session, T).ok());

			std::vector<unsigned int> prefix;
			std::vector<float> logitsFull;
			std::vector<float> logitsKv;
			for (unsigned int t = 0; t < T; ++t)
			{
				prefix.push_back(toks[t]);
				G_assert(__FILE__, __LINE__, "==============Gate1 ForwardLastLogits Failed==============", net.transformerLmForwardLastLogits(prefix, logitsFull).ok());
				G_assert(__FILE__, __LINE__, "==============Gate1 SessionAppend Failed==============", net.transformerLmSessionAppend(session, toks[t], &logitsKv).ok());
				G_assert(__FILE__, __LINE__, "==============Gate1 LogitsSize Failed==============", logitsFull.size() == vocab && logitsKv.size() == vocab);
				double maxAbs = 0.0;
				for (unsigned int i = 0; i < vocab; ++i)
				{
					const double d = fabs((double)logitsFull[i] - (double)logitsKv[i]);
					if (d > maxAbs) maxAbs = d;
					G_assert(__FILE__, __LINE__, "==============Gate1 LogitFinite Failed==============", std::isfinite(logitsFull[i]) && std::isfinite(logitsKv[i]));
				}
				G_assert(__FILE__, __LINE__, "==============Gate1 ParityMismatch Failed==============", maxAbs < 1e-3);
			}

			delete di;
			delete info;
		}

		// Gate 2: Determinism (same seed + config => same logits).
		{
			const unsigned int vocab = 19u;
			const unsigned int padTokenId = vocab - 1u;
			std::vector<unsigned int> prefix;
			prefix.push_back(2u);
			prefix.push_back(4u);
			prefix.push_back(6u);

			InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
			di->setTrainTokens(prefix, static_cast<int>(padTokenId));
			di->mirrorTrainToTest();

			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(12, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f)); // dModel=12
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_gate_determinism", in, hidden, out);

			glades::NNetwork a(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			glades::NNetwork b(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			a.setSeed(9u);
			b.setSeed(9u);
			{
				glades::TrainingConfig& cfgA = a.getTrainingConfigMutable();
				cfgA.transformer.enableTokenEmbedding = true;
				cfgA.transformer.vocabSizeOverride = static_cast<int>(vocab);
				cfgA.transformer.tieEmbeddings = true;
				cfgA.transformer.padTokenId = static_cast<int>(padTokenId);
				cfgA.transformer.nHeadsOverride = 3; // dHead=4
				cfgA.transformer.nKVHeadsOverride = 3;
				cfgA.transformer.dFFOverride = 24;
				cfgA.transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
				cfgA.transformer.ffnActivation = glades::TransformerRunConfig::FFN_GELU;
				cfgA.transformer.normType = glades::TransformerRunConfig::NORM_LAYERNORM;
				cfgA.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_SINUSOIDAL;
			}
			b.getTrainingConfigMutable() = a.getTrainingConfigMutable();

			G_assert(__FILE__, __LINE__, "==============Gate2 InitA Failed==============", a.test(di).ok());
			G_assert(__FILE__, __LINE__, "==============Gate2 InitB Failed==============", b.test(di).ok());

			std::vector<float> la, lb;
			G_assert(__FILE__, __LINE__, "==============Gate2 ForwardA Failed==============", a.transformerLmForwardLastLogits(prefix, la).ok());
			G_assert(__FILE__, __LINE__, "==============Gate2 ForwardB Failed==============", b.transformerLmForwardLastLogits(prefix, lb).ok());
			G_assert(__FILE__, __LINE__, "==============Gate2 Size Failed==============", la.size() == vocab && lb.size() == vocab);
			for (unsigned int i = 0; i < vocab; ++i)
				G_assert(__FILE__, __LINE__, "==============Gate2 LogitsMismatch Failed==============", fabs((double)la[i] - (double)lb[i]) < 1e-7);

			delete di;
			delete info;
		}

		// Gate 3: Padding key-mask correctness (attention forward ignores masked keys).
		{
			const unsigned int T = 3u;
			const unsigned int dK = 1u;
			const unsigned int dV = 1u;
			const float Q[T * dK] = {1.0f, 1.0f, 1.0f};
			const float K[T * dK] = {0.0f, 100.0f, 0.0f};  // masked key has huge score if not masked
			const float V[T * dV] = {1.0f, 999.0f, 3.0f};  // masked value would dominate if included
			unsigned char keyAllowed[T] = {1u, 0u, 1u};    // middle position is padding/masked

			float O[T * dV] = {0.0f, 0.0f, 0.0f};
			glades::transformer_ops::scaled_dot_product_attention_forward_flash_strided(
			    Q, /*qStride*/ dK,
			    K, /*kStride*/ dK,
			    V, /*vStride*/ dV,
			    T, dK, dV,
			    /*causal*/ false,
			    O, /*oStride*/ dV,
			    keyAllowed);

			// Only keys {0,2} are allowed and have equal scores (0), so output is uniform average: (1+3)/2 = 2.
			for (unsigned int t = 0; t < T; ++t)
			{
				G_assert(__FILE__, __LINE__, "==============Gate3 O Finite Failed==============", std::isfinite(O[t]));
				G_assert(__FILE__, __LINE__, "==============Gate3 MaskedForward Wrong Failed==============", fabs((double)O[t] - 2.0) < 1e-5);
			}
		}

		// Gate 4: Padding key-mask correctness (attention backward produces zero grads for masked keys).
		{
			const unsigned int T = 3u;
			const unsigned int dK = 1u;
			const unsigned int dV = 1u;
			const float Q[T * dK] = {1.0f, 1.0f, 1.0f};
			const float K[T * dK] = {0.0f, 100.0f, 0.0f};
			const float V[T * dV] = {1.0f, 999.0f, 3.0f};
			const float dO[T * dV] = {1.0f, -2.0f, 3.0f};
			unsigned char keyAllowed[T] = {1u, 0u, 1u};

			float dQ[T * dK] = {0.0f, 0.0f, 0.0f};
			float dKbuf[T * dK] = {0.0f, 0.0f, 0.0f};
			float dVbuf[T * dV] = {0.0f, 0.0f, 0.0f};
			glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_strided(
			    Q, /*qStride*/ dK,
			    K, /*kStride*/ dK,
			    V, /*vStride*/ dV,
			    dO, /*dOStride*/ dV,
			    T, dK, dV,
			    /*causal*/ false,
			    dQ, /*dQStride*/ dK,
			    dKbuf, /*dKStride*/ dK,
			    dVbuf, /*dVStride*/ dV,
			    keyAllowed);

			// Masked key/value at u=1 must have exactly zero gradients.
			G_assert(__FILE__, __LINE__, "==============Gate4 Masked dK NotZero Failed==============", fabs((double)dKbuf[1]) == 0.0);
			G_assert(__FILE__, __LINE__, "==============Gate4 Masked dV NotZero Failed==============", fabs((double)dVbuf[1]) == 0.0);
			for (unsigned int i = 0; i < T; ++i)
			{
				G_assert(__FILE__, __LINE__, "==============Gate4 dQ Finite Failed==============", std::isfinite(dQ[i]));
				G_assert(__FILE__, __LINE__, "==============Gate4 dK Finite Failed==============", std::isfinite(dKbuf[i]));
				G_assert(__FILE__, __LINE__, "==============Gate4 dV Finite Failed==============", std::isfinite(dVbuf[i]));
			}
		}

		// Gate 5: RoPE forward+inverse is identity (strided in-place kernel).
		{
			const unsigned int T = 4u;
			const unsigned int dHead = 4u;
			const unsigned int ropeDim = 4u;
			const double theta = 10000.0;
			std::vector<double> invFreq(ropeDim / 2u, 0.0);
			for (unsigned int i = 0; i < ropeDim / 2u; ++i)
				invFreq[i] = pow(theta, -2.0 * (double)i / (double)ropeDim);

			std::vector<float> buf(T * dHead, 0.0f);
			for (unsigned int t = 0; t < T; ++t)
			{
				for (unsigned int j = 0; j < dHead; ++j)
					buf[t * dHead + j] = (float)(0.1 * (double)(1u + t) + 0.01 * (double)j);
			}
			const std::vector<float> orig = buf;

			glades::transformer_kernels::rope_apply_inplace_strided(&buf[0], T, /*rowStride*/ dHead, dHead, ropeDim, invFreq, /*inverse*/ false);
			glades::transformer_kernels::rope_apply_inplace_strided(&buf[0], T, /*rowStride*/ dHead, dHead, ropeDim, invFreq, /*inverse*/ true);

			double maxAbs = 0.0;
			for (size_t i = 0; i < buf.size(); ++i)
			{
				const double d = fabs((double)buf[i] - (double)orig[i]);
				if (d > maxAbs) maxAbs = d;
			}
			G_assert(__FILE__, __LINE__, "==============Gate5 RoPE Invertibility Failed==============", maxAbs < 1e-6);
		}
	}

	// Minimal smoke test: ensure transformer can initialize tensors and run.
	printf("-----------------------------------\n");
	printf("Transformer smoke test\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		// A single sequence of length 3 (default DataInput sequence semantics: all rows are one sequence).
		di->trainMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(3, shmea::GVector<float>(1, 0.0f));
		for (int t = 0; t < 3; ++t)
		{
			di->trainMatrix[t][0] = static_cast<float>(t);
			// Learn identity: y = x (simple but non-trivial; validates gradients are non-zero).
			di->trainExpectedMatrix[t][0] = static_cast<float>(t);
		}
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

        glades::InputLayerInfo* in = new glades::InputLayerInfo(
            /*batchSize*/ 1,
		    /*learningRate*/ 0.01f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
            /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);

		// Transformer blocks are represented as hidden layers with constant size == dModel.
		// Heads and dFF are configured via TrainingConfig.transformer overrides.
        std::vector<glades::HiddenLayerInfo*> hidden;
        hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 8,                 // dModel
		    /*learningRate*/ 0.01f,
            /*momentumFactor*/ 0.0f,
            /*weightDecay1*/ 0.0f,
            /*weightDecay2*/ 0.0f,
            /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));

        glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_smoke", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
		net.setSeed(2026u);
        net.getTerminatorMutable().setEpoch(5);
        net.getTerminatorMutable().setAccuracy(0);

		// Exercise "modern LLM-style" transformer options:
		// - RoPE positional encoding
		// - RMSNorm
		// - SwiGLU FFN
		// - Grouped-query attention (KV heads < Q heads)
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.transformer.nHeadsOverride = 2;
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.nKVHeadsOverride = 1; // with nHeads=2 => 2 query heads share 1 KV head
			cfg.transformer.ropeTheta = 10000.0f;
			cfg.transformer.ropeDimOverride = 0; // full head dim
		}

		// Force init (no updates), then snapshot weights to verify training updates parameters.
		G_assert(__FILE__, __LINE__, "==============NN::Transformer Smoke InitTestStatus() Failed==============", net.test(di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::Transformer Smoke SaveModel(before) Failed==============", net.saveModel("ut_pkg_transformer_smoke_before").ok());
		double l2Before = 0.0;
		G_assert(__FILE__, __LINE__, "==============NN::Transformer Smoke ReadWeights(before) Failed==============",
		         transformer_weights_l2_from_file("database/models/ut_pkg_transformer_smoke_before/weights.bin", l2Before));
		G_assert(__FILE__, __LINE__, "==============NN::Transformer Smoke L2BeforeNonZero() Failed==============", l2Before > 0.0);

		const glades::NNetworkStatus st = net.train(di);
		G_assert(__FILE__, __LINE__, "==============NN::Transformer Smoke TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============NN::Transformer Smoke SaveModel(after) Failed==============", net.saveModel("ut_pkg_transformer_smoke_after").ok());
		double l2After = 0.0;
		G_assert(__FILE__, __LINE__, "==============NN::Transformer Smoke ReadWeights(after) Failed==============",
		         transformer_weights_l2_from_file("database/models/ut_pkg_transformer_smoke_after/weights.bin", l2After));
		// Assert that training actually updated parameters (non-trivial gradient path).
		G_assert(__FILE__, __LINE__, "==============NN::Transformer Smoke ParamsUpdated() Failed==============", fabs(l2After - l2Before) > 1e-9);

		delete di;
		delete info;
    }

	// Parity test: full forward (recompute) vs KV-cache incremental decode.
	// This is the single most important correctness invariant for autoregressive inference.
	printf("-----------------------------------\n");
	printf("Transformer decoder KV parity (full forward vs KV-cache)\n");
	printf("-----------------------------------\n");
	{
		// Small deterministic setup to keep this test fast and stable.
		const unsigned int vocab = 32u;
		const unsigned int padTokenId = vocab - 1u;
		const unsigned int T = 6u;
		std::vector<unsigned int> toks;
		for (unsigned int i = 0; i < T; ++i)
			toks.push_back(1u + i); // 1..6 (avoid padTokenId)

		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.01f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);

		// Transformer blocks are represented as hidden layers with constant size == dModel.
		// Heads/dFF are configured via TrainingConfig.transformer overrides.
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 16,                // dModel
		    /*learningRate*/ 0.01f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 16,
		    /*learningRate*/ 0.01f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_decoder_kv_parity", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(2026u);

		// Configure token LM mode explicitly.
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			// Ensure we do not depend on legacy NNInfo hidden-layer encoding for heads/dFF.
			// (Some unit-test hidden layer rows use activation metadata for other purposes.)
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.dFFOverride = 32;

			// Use RoPE here because KV-cache inference applies RoPE to Q/K and should be parity-safe.
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.nKVHeadsOverride = 2; // GQA: 4 Q heads share 2 KV heads
			cfg.transformer.ropeTheta = 10000.0f;
			cfg.transformer.ropeDimOverride = 0;
		}

		// Initialize weights/tensors.
		G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity InitTestStatus() Failed==============", net.test(di).ok());

		// Compare logits for each growing prefix.
		glades::NNetwork::TransformerLmSession session;
		G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity SessionReset() Failed==============",
		         net.transformerLmSessionReset(session, T).ok());
		std::vector<unsigned int> prefix;
		std::vector<float> logitsFull;
		std::vector<float> logitsKv;
		for (unsigned int t = 0; t < T; ++t)
		{
			prefix.push_back(toks[t]);
			G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity ForwardLastLogits() Failed==============",
			         net.transformerLmForwardLastLogits(prefix, logitsFull).ok());
			G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity SessionAppend() Failed==============",
			         net.transformerLmSessionAppend(session, toks[t], &logitsKv).ok());

			G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity LogitsSizeMismatch() Failed==============",
			         logitsFull.size() == logitsKv.size() && logitsFull.size() == vocab);

			double maxAbsDiff = 0.0;
			for (unsigned int i = 0; i < vocab; ++i)
			{
				const double d = fabs(static_cast<double>(logitsFull[i]) - static_cast<double>(logitsKv[i]));
				if (d > maxAbsDiff) maxAbsDiff = d;
			}
			G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity LogitsMismatch() Failed==============",
			         maxAbsDiff < 1e-3);
		}

		delete di;
		delete info;
	}

	// Parity test variants: exercise the other positional encoding modes too.
	printf("-----------------------------------\n");
	printf("Transformer decoder KV parity variants (NONE + SINUSOIDAL)\n");
	printf("-----------------------------------\n");
	{
		struct ParityCase
		{
			static void run(glades::TransformerRunConfig::PositionalEncodingType pe)
			{
				const unsigned int vocab = 24u;
				const unsigned int padTokenId = vocab - 1u;
				const unsigned int T = 5u;
				std::vector<unsigned int> toks;
				for (unsigned int i = 0; i < T; ++i)
					toks.push_back(2u + i); // avoid 0/1/pad

				InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
				di->setTrainTokens(toks, static_cast<int>(padTokenId));
				di->mirrorTrainToTest();

				glades::InputLayerInfo* in = new glades::InputLayerInfo(
				    /*batchSize*/ 1,
				    /*learningRate*/ 0.01f,
				    /*momentumFactor*/ 0.0f,
				    /*weightDecay1*/ 0.0f,
				    /*weightDecay2*/ 0.0f,
				    /*pDropout*/ 0.0f,
				    /*activationType*/ glades::GMath::LINEAR,
				    /*activationParam*/ 1.0f);

				std::vector<glades::HiddenLayerInfo*> hidden;
				hidden.push_back(new glades::HiddenLayerInfo(
				    /*size*/ 12,                // dModel
				    /*learningRate*/ 0.01f,
				    /*momentumFactor*/ 0.0f,
				    /*weightDecay1*/ 0.0f,
				    /*weightDecay2*/ 0.0f,
				    /*pDropout*/ 0.0f,
				    /*activationType*/ glades::GMath::LINEAR,
				    /*activationParam*/ 1.0f));

				glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
				glades::NNInfo* info = new glades::NNInfo("ut_transformer_decoder_kv_parity_variants", in, hidden, out);

				glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
				net.setSeed(2026u);
				{
					glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
					cfg.transformer.enableTokenEmbedding = true;
					cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
					cfg.transformer.tieEmbeddings = true;
					cfg.transformer.padTokenId = static_cast<int>(padTokenId);
					cfg.transformer.nHeadsOverride = 3; // dModel=12 -> dHead=4
					cfg.transformer.nKVHeadsOverride = 3;
					cfg.transformer.dFFOverride = 24;
					cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
					cfg.transformer.ffnActivation = glades::TransformerRunConfig::FFN_GELU;
					cfg.transformer.normType = glades::TransformerRunConfig::NORM_LAYERNORM;
					cfg.transformer.positionalEncoding = pe;
				}

				G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity Variants InitTestStatus() Failed==============", net.test(di).ok());
				glades::NNetwork::TransformerLmSession session;
				G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity Variants SessionReset() Failed==============",
				         net.transformerLmSessionReset(session, T).ok());

				std::vector<unsigned int> prefix;
				std::vector<float> logitsFull;
				std::vector<float> logitsKv;
				for (unsigned int t = 0; t < T; ++t)
				{
					prefix.push_back(toks[t]);
					G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity Variants ForwardLastLogits() Failed==============",
					         net.transformerLmForwardLastLogits(prefix, logitsFull).ok());
					G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity Variants SessionAppend() Failed==============",
					         net.transformerLmSessionAppend(session, toks[t], &logitsKv).ok());

					double maxAbsDiff = 0.0;
					for (unsigned int i = 0; i < vocab; ++i)
					{
						const double d = fabs(static_cast<double>(logitsFull[i]) - static_cast<double>(logitsKv[i]));
						if (d > maxAbsDiff) maxAbsDiff = d;
					}
					G_assert(__FILE__, __LINE__, "==============NN::Transformer Decoder KV Parity Variants LogitsMismatch() Failed==============",
					         maxAbsDiff < 1e-3);
				}

				delete di;
				delete info;
			}
		};

		ParityCase::run(glades::TransformerRunConfig::POSENC_NONE);
		ParityCase::run(glades::TransformerRunConfig::POSENC_SINUSOIDAL);
	}

	// KV parity with padding tokens interspersed:
	// - padded positions must be masked out of attention as KEYS
	// - full forward vs KV-cache incremental decode must remain identical
	printf("-----------------------------------\n");
	printf("Transformer decoder KV parity (padding token key-mask)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 23u;
		const unsigned int padTokenId = vocab - 1u;
		const unsigned int T = 7u;
		// Include padding tokens in the prompt to validate key-masking parity.
		// (Pad tokens are still processed as queries; they just can't be attended-to as keys.)
		const unsigned int toksArr[T] = {2u, 3u, padTokenId, 4u, 5u, padTokenId, 6u};
		std::vector<unsigned int> toks(toksArr, toksArr + T);

		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_decoder_kv_parity_padding_mask", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(2026u);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;  // dHead=4
			cfg.transformer.nKVHeadsOverride = 2; // GQA path
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
			cfg.transformer.ropeTheta = 10000.0f;
			cfg.transformer.ropeDimOverride = 0;
		}

		G_assert(__FILE__, __LINE__, "==============NN::KVPadMask InitTestStatus() Failed==============", net.test(di).ok());

		glades::NNetwork::TransformerLmSession session;
		G_assert(__FILE__, __LINE__, "==============NN::KVPadMask SessionReset Failed==============", net.transformerLmSessionReset(session, T).ok());

		std::vector<unsigned int> prefix;
		std::vector<float> logitsFull;
		std::vector<float> logitsKv;
		for (unsigned int t = 0; t < T; ++t)
		{
			prefix.push_back(toks[t]);
			G_assert(__FILE__, __LINE__, "==============NN::KVPadMask ForwardLastLogits Failed==============", net.transformerLmForwardLastLogits(prefix, logitsFull).ok());
			G_assert(__FILE__, __LINE__, "==============NN::KVPadMask SessionAppend Failed==============", net.transformerLmSessionAppend(session, toks[t], &logitsKv).ok());
			G_assert(__FILE__, __LINE__, "==============NN::KVPadMask Size Failed==============", logitsFull.size() == vocab && logitsKv.size() == vocab);
			double maxAbsDiff = 0.0;
			for (unsigned int i = 0; i < vocab; ++i)
			{
				G_assert(__FILE__, __LINE__, "==============NN::KVPadMask Finite Failed==============", std::isfinite(logitsFull[i]) && std::isfinite(logitsKv[i]));
				const double d = fabs(static_cast<double>(logitsFull[i]) - static_cast<double>(logitsKv[i]));
				if (d > maxAbsDiff) maxAbsDiff = d;
			}
			G_assert(__FILE__, __LINE__, "==============NN::KVPadMask ParityMismatch Failed==============", maxAbsDiff < 1e-3);
		}

		delete di;
		delete info;
	}

	// KV parity for batched KV-cache sessions:
	// - batch session must match per-sequence sessions for ragged prompts
	// - both codepaths are used in serving (single request vs batch API)
	printf("-----------------------------------\n");
	printf("Transformer decoder KV parity (batch session vs per-sequence sessions)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 31u;
		const unsigned int padTokenId = vocab - 1u;
		const unsigned int batchSize = 2u;
		const unsigned int maxLen = 6u;

		// Ragged prompts.
		const unsigned int p0Arr[6] = {2u, 3u, 4u, 5u, 6u, 7u};
		const unsigned int p1Arr[3] = {8u, 9u, 10u};
		std::vector<unsigned int> p0(p0Arr, p0Arr + 6u);
		std::vector<unsigned int> p1(p1Arr, p1Arr + 3u);

		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		di->setTrainTokens(p0, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_decoder_kv_parity_batch_session", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(2026u);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.nKVHeadsOverride = 2; // exercise GQA
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
			cfg.transformer.ropeTheta = 10000.0f;
			cfg.transformer.ropeDimOverride = 0;
		}
		G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity InitTestStatus() Failed==============", net.test(di).ok());

		// Reference: per-sequence sessions.
		std::vector<float> ref0, ref1;
		{
			glades::NNetwork::TransformerLmSession s0;
			G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity S0 Reset Failed==============", net.transformerLmSessionReset(s0, maxLen).ok());
			for (size_t t = 0; t < p0.size(); ++t)
				G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity S0 Append Failed==============", net.transformerLmSessionAppend(s0, p0[t], (t + 1u == p0.size()) ? &ref0 : NULL).ok());

			glades::NNetwork::TransformerLmSession s1;
			G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity S1 Reset Failed==============", net.transformerLmSessionReset(s1, maxLen).ok());
			for (size_t t = 0; t < p1.size(); ++t)
				G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity S1 Append Failed==============", net.transformerLmSessionAppend(s1, p1[t], (t + 1u == p1.size()) ? &ref1 : NULL).ok());
		}
		G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity RefSize0 Failed==============", ref0.size() == vocab);
		G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity RefSize1 Failed==============", ref1.size() == vocab);

		// Batched session: ragged-safe append with active mask.
		glades::NNetwork::TransformerLmBatchSession bs;
		G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity BatchReset Failed==============", net.transformerLmBatchSessionReset(bs, batchSize, maxLen).ok());
		std::vector<float> logitsFlat;
		std::vector<float> got0, got1;
		got0.assign(vocab, 0.0f);
		got1.assign(vocab, 0.0f);
		for (unsigned int step = 0u; step < maxLen; ++step)
		{
			std::vector<unsigned int> tokenIds(batchSize, padTokenId);
			std::vector<unsigned char> active(batchSize, 0u);

			if (step < static_cast<unsigned int>(p0.size()))
			{
				active[0] = 1u;
				tokenIds[0] = p0[step];
			}
			if (step < static_cast<unsigned int>(p1.size()))
			{
				active[1] = 1u;
				tokenIds[1] = p1[step];
			}

			G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity BatchAppend Failed==============",
			         net.transformerLmBatchSessionAppendSelective(bs, tokenIds, active, &logitsFlat).ok());
			G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity FlatSize Failed==============", logitsFlat.size() == static_cast<size_t>(batchSize) * static_cast<size_t>(vocab));

			// Inactive rows must be zeros by contract (helps downstream code avoid branching).
			for (unsigned int b = 0u; b < batchSize; ++b)
			{
				if (active[b] == 0u)
				{
					for (unsigned int i = 0u; i < vocab; ++i)
						G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity InactiveNotZero Failed==============", logitsFlat[b * vocab + i] == 0.0f);
				}
			}

			if (step + 1u == static_cast<unsigned int>(p0.size()))
			{
				for (unsigned int i = 0u; i < vocab; ++i)
					got0[i] = logitsFlat[0u * vocab + i];
			}
			if (step + 1u == static_cast<unsigned int>(p1.size()))
			{
				for (unsigned int i = 0u; i < vocab; ++i)
					got1[i] = logitsFlat[1u * vocab + i];
			}
		}

		// Compare batch vs per-sequence logits at the last prompt token.
		double maxAbs0 = 0.0;
		double maxAbs1 = 0.0;
		for (unsigned int i = 0u; i < vocab; ++i)
		{
			const double d0 = fabs(static_cast<double>(got0[i]) - static_cast<double>(ref0[i]));
			const double d1 = fabs(static_cast<double>(got1[i]) - static_cast<double>(ref1[i]));
			if (d0 > maxAbs0) maxAbs0 = d0;
			if (d1 > maxAbs1) maxAbs1 = d1;
		}
		G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity LogitsMismatch0 Failed==============", maxAbs0 < 1e-3);
		G_assert(__FILE__, __LINE__, "==============NN::BatchKVParity LogitsMismatch1 Failed==============", maxAbs1 < 1e-3);

		delete di;
		delete info;
	}

	// KV-cache FP16 storage parity:
	// - logits won't be bit-identical vs full forward (FP16 quantization), but should remain close
	// - argmax should remain stable for small deterministic models
	printf("-----------------------------------\n");
	printf("Transformer decoder KV-cache FP16 storage parity (close + argmax stable)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 29u;
		const unsigned int padTokenId = vocab - 1u;
		const unsigned int T = 6u;
		std::vector<unsigned int> toks;
		for (unsigned int i = 0u; i < T; ++i)
			toks.push_back(2u + i);

		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_decoder_kv_cache_fp16_parity", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(2026u);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
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
			cfg.transformer.kvCacheDType = glades::TransformerRunConfig::KV_CACHE_F16;
		}
		G_assert(__FILE__, __LINE__, "==============NN::KVFP16Parity InitTestStatus() Failed==============", net.test(di).ok());

		glades::NNetwork::TransformerLmSession session;
		G_assert(__FILE__, __LINE__, "==============NN::KVFP16Parity SessionReset Failed==============", net.transformerLmSessionReset(session, T).ok());

		std::vector<unsigned int> prefix;
		std::vector<float> logitsFull;
		std::vector<float> logitsKv;
		for (unsigned int t = 0u; t < T; ++t)
		{
			prefix.push_back(toks[t]);
			G_assert(__FILE__, __LINE__, "==============NN::KVFP16Parity ForwardLastLogits Failed==============", net.transformerLmForwardLastLogits(prefix, logitsFull).ok());
			G_assert(__FILE__, __LINE__, "==============NN::KVFP16Parity SessionAppend Failed==============", net.transformerLmSessionAppend(session, toks[t], &logitsKv).ok());
			G_assert(__FILE__, __LINE__, "==============NN::KVFP16Parity Size Failed==============", logitsFull.size() == vocab && logitsKv.size() == vocab);

			// Argmax stability + reasonably small error bound.
			unsigned int a0 = 0u, a1 = 0u;
			for (unsigned int i = 1u; i < vocab; ++i)
			{
				if (logitsFull[i] > logitsFull[a0]) a0 = i;
				if (logitsKv[i] > logitsKv[a1]) a1 = i;
			}
			G_assert(__FILE__, __LINE__, "==============NN::KVFP16Parity ArgmaxMismatch Failed==============", a0 == a1);

			double maxAbsDiff = 0.0;
			for (unsigned int i = 0u; i < vocab; ++i)
			{
				G_assert(__FILE__, __LINE__, "==============NN::KVFP16Parity Finite Failed==============", std::isfinite(logitsFull[i]) && std::isfinite(logitsKv[i]));
				const double d = fabs(static_cast<double>(logitsFull[i]) - static_cast<double>(logitsKv[i]));
				if (d > maxAbsDiff) maxAbsDiff = d;
			}
			G_assert(__FILE__, __LINE__, "==============NN::KVFP16Parity TooFar Failed==============", maxAbsDiff < 5e-2);
		}

		delete di;
		delete info;
	}

	// KV-cache BF16 storage parity:
	// - logits won't be bit-identical vs full forward (BF16 quantization), but should remain close
	// - argmax should remain stable for small deterministic models
	printf("-----------------------------------\n");
	printf("Transformer decoder KV-cache BF16 storage parity (close + argmax stable)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 29u;
		const unsigned int padTokenId = vocab - 1u;
		const unsigned int T = 6u;
		std::vector<unsigned int> toks;
		for (unsigned int i = 0u; i < T; ++i)
			toks.push_back(2u + i);

		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_decoder_kv_cache_bf16_parity", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(2026u);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
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
			cfg.transformer.kvCacheDType = glades::TransformerRunConfig::KV_CACHE_BF16;
		}
		G_assert(__FILE__, __LINE__, "==============NN::KVBF16Parity InitTestStatus() Failed==============", net.test(di).ok());

		glades::NNetwork::TransformerLmSession session;
		G_assert(__FILE__, __LINE__, "==============NN::KVBF16Parity SessionReset Failed==============", net.transformerLmSessionReset(session, T).ok());

		std::vector<unsigned int> prefix;
		std::vector<float> logitsFull;
		std::vector<float> logitsKv;
		for (unsigned int t = 0u; t < T; ++t)
		{
			prefix.push_back(toks[t]);
			G_assert(__FILE__, __LINE__, "==============NN::KVBF16Parity ForwardLastLogits Failed==============", net.transformerLmForwardLastLogits(prefix, logitsFull).ok());
			G_assert(__FILE__, __LINE__, "==============NN::KVBF16Parity SessionAppend Failed==============", net.transformerLmSessionAppend(session, toks[t], &logitsKv).ok());
			G_assert(__FILE__, __LINE__, "==============NN::KVBF16Parity Size Failed==============", logitsFull.size() == vocab && logitsKv.size() == vocab);

			// Argmax stability + reasonably small error bound.
			unsigned int a0 = 0u, a1 = 0u;
			for (unsigned int i = 1u; i < vocab; ++i)
			{
				if (logitsFull[i] > logitsFull[a0]) a0 = i;
				if (logitsKv[i] > logitsKv[a1]) a1 = i;
			}
			G_assert(__FILE__, __LINE__, "==============NN::KVBF16Parity ArgmaxMismatch Failed==============", a0 == a1);

			double maxAbsDiff = 0.0;
			for (unsigned int i = 0u; i < vocab; ++i)
			{
				G_assert(__FILE__, __LINE__, "==============NN::KVBF16Parity Finite Failed==============", std::isfinite(logitsFull[i]) && std::isfinite(logitsKv[i]));
				const double d = fabs(static_cast<double>(logitsFull[i]) - static_cast<double>(logitsKv[i]));
				if (d > maxAbsDiff) maxAbsDiff = d;
			}
			G_assert(__FILE__, __LINE__, "==============NN::KVBF16Parity TooFar Failed==============", maxAbsDiff < 5e-2);
		}

		delete di;
		delete info;
	}

	// RoPE edge-case: odd/small ropeDimOverride should not crash and must preserve KV-cache parity.
	printf("-----------------------------------\n");
	printf("Transformer decoder KV parity (RoPE ropeDimOverride edge cases)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 20u;
		const unsigned int padTokenId = vocab - 1u;
		const unsigned int T = 5u;
		std::vector<unsigned int> toks;
		for (unsigned int i = 0; i < T; ++i)
			toks.push_back(2u + i);

		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.01f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 12,                // dModel
		    /*learningRate*/ 0.01f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_decoder_rope_dim_override", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(2026u);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 3; // dHead=4
			cfg.transformer.nKVHeadsOverride = 1; // GQA
			cfg.transformer.dFFOverride = 24;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
			cfg.transformer.ropeTheta = 10000.0f;
			cfg.transformer.ropeDimOverride = 1; // odd + <2 => rounds to 0, effectively disabling rotation but exercising the branch
		}

		G_assert(__FILE__, __LINE__, "==============NN::RoPEDimOverride InitTestStatus() Failed==============", net.test(di).ok());
		glades::NNetwork::TransformerLmSession session;
		G_assert(__FILE__, __LINE__, "==============NN::RoPEDimOverride SessionReset Failed==============",
		         net.transformerLmSessionReset(session, T).ok());

		std::vector<unsigned int> prefix;
		std::vector<float> logitsFull;
		std::vector<float> logitsKv;
		for (unsigned int t = 0; t < T; ++t)
		{
			prefix.push_back(toks[t]);
			G_assert(__FILE__, __LINE__, "==============NN::RoPEDimOverride ForwardLastLogits Failed==============",
			         net.transformerLmForwardLastLogits(prefix, logitsFull).ok());
			G_assert(__FILE__, __LINE__, "==============NN::RoPEDimOverride SessionAppend Failed==============",
			         net.transformerLmSessionAppend(session, toks[t], &logitsKv).ok());
			double maxAbsDiff = 0.0;
			for (unsigned int i = 0; i < vocab; ++i)
			{
				const double d = fabs(static_cast<double>(logitsFull[i]) - static_cast<double>(logitsKv[i]));
				if (d > maxAbsDiff) maxAbsDiff = d;
			}
			G_assert(__FILE__, __LINE__, "==============NN::RoPEDimOverride LogitsMismatch Failed==============", maxAbsDiff < 1e-3);
		}

		delete di;
		delete info;
	}

	// RoPE parameter sensitivity: changing ropeTheta should change logits for positions > 0.
	printf("-----------------------------------\n");
	printf("Transformer decoder RoPE sensitivity (ropeTheta changes logits)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 16u;
		const unsigned int padTokenId = vocab - 1u;
		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		{
			std::vector<unsigned int> toks;
			toks.push_back(2u);
			toks.push_back(3u);
			di->setTrainTokens(toks, static_cast<int>(padTokenId));
			di->mirrorTrainToTest();
		}

		glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_rope_theta_sensitivity", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(2026u);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
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
			cfg.transformer.ropeDimOverride = 0;
		}
		G_assert(__FILE__, __LINE__, "==============NN::RoPEThetaSensitivity InitTestStatus() Failed==============", net.test(di).ok());

		std::vector<unsigned int> prefix;
		prefix.push_back(2u);
		prefix.push_back(3u);

		std::vector<float> logitsA, logitsB;
		net.getTrainingConfigMutable().transformer.ropeTheta = 10000.0f;
		G_assert(__FILE__, __LINE__, "==============NN::RoPEThetaSensitivity ForwardA Failed==============", net.transformerLmForwardLastLogits(prefix, logitsA).ok());
		net.getTrainingConfigMutable().transformer.ropeTheta = 500.0f;
		G_assert(__FILE__, __LINE__, "==============NN::RoPEThetaSensitivity ForwardB Failed==============", net.transformerLmForwardLastLogits(prefix, logitsB).ok());
		G_assert(__FILE__, __LINE__, "==============NN::RoPEThetaSensitivity Size Failed==============", logitsA.size() == logitsB.size() && logitsA.size() == vocab);

		double maxAbsDiff = 0.0;
		for (unsigned int i = 0; i < vocab; ++i)
		{
			const double d = fabs(static_cast<double>(logitsA[i]) - static_cast<double>(logitsB[i]));
			if (d > maxAbsDiff) maxAbsDiff = d;
		}
		// If this fails, it likely means RoPE is not being applied, or the model collapsed into a degenerate state.
		G_assert(__FILE__, __LINE__, "==============NN::RoPEThetaSensitivity NoEffect Failed==============", maxAbsDiff > 1e-6);

		delete di;
		delete info;
	}

	// Deterministic toy model: identity embedding + no-op blocks => logits are exactly predictable.
	printf("-----------------------------------\n");
	printf("Transformer decoder token LM deterministic logits (toy identity embedding)\n");
	printf("-----------------------------------\n");
	{
		// Design:
		// - vocab == dModel == 3
		// - embedding E is identity (one-hot basis)
		// - all block weights are zeros, so the transformer block is a no-op and the hidden state stays == embedding
		// - logits are computed as h * E^T + bias => logits == h exactly (since E is identity and bias=0)
		const unsigned int vocab = 3u;
		const unsigned int dModel = 3u;
		const unsigned int dFF = 4u;
		const unsigned int nHeads = 1u;
		const unsigned int nKVHeads = 1u;
		const unsigned int nLayers = 1u;
		const unsigned int padTokenId = vocab - 1u;

		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		{
			std::vector<unsigned int> toks;
			toks.push_back(0u);
			toks.push_back(1u);
			di->setTrainTokens(toks, static_cast<int>(padTokenId));
			di->mirrorTrainToTest();
		}

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
		    /*size*/ static_cast<int>(dModel),
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_toy_identity_logits", in, hidden, out);

		// 1) Bootstrap a package with the right manifest config.
		glades::NNetwork bootstrap(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		bootstrap.setSeed(123u);
		{
			glades::TrainingConfig& cfg = bootstrap.getTrainingConfigMutable();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = static_cast<int>(nHeads);
			cfg.transformer.nKVHeadsOverride = static_cast<int>(nKVHeads);
			cfg.transformer.dFFOverride = static_cast<int>(dFF);
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_LAYERNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_NONE;
		}
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity InitTestStatus() Failed==============", bootstrap.test(di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity SaveModel(bootstrap) Failed==============", bootstrap.saveModel("ut_pkg_transformer_toy_identity").ok());

		// 2) Override weights with deterministic values.
		const std::vector<float> E = tokE_identity(vocab, dModel); // identity
		const std::vector<float> bLm(vocab, 0.0f);
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity WriteOverrideWeights Failed==============",
		         write_transformer_decoder_tokenlm_weights("database/models/ut_pkg_transformer_toy_identity/weights.bin",
		                                                   nLayers, dModel, dFF, nHeads, nKVHeads,
		                                                   vocab, padTokenId,
		                                                   static_cast<unsigned int>(glades::TransformerRunConfig::FFN_MLP),
		                                                   E, bLm,
		                                                   /*zeroAllBlocks*/ true));

		// 3) Load the overridden model into a fresh net and assert exact logits.
		glades::NNetwork net(glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity LoadModel Failed==============", net.loadModel("ut_pkg_transformer_toy_identity", di).ok());

		std::vector<float> logits;
		std::vector<unsigned int> toks;
		toks.push_back(2u);
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity ForwardLastLogits Failed==============", net.transformerLmForwardLastLogits(toks, logits).ok());
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity LogitsSize Failed==============", logits.size() == vocab);
		// After final LayerNorm (gamma=1,beta=0) on [0,0,1]: h_norm = [-1/sqrt(2), -1/sqrt(2), sqrt(2)]
		// Tied identity embedding gives logits = h_norm.
		const float invSqrt2 = static_cast<float>(1.0 / sqrt(2.0));
		const float sqrt2    = static_cast<float>(sqrt(2.0));
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity Logit0 Failed==============", fabs(logits[0] - (-invSqrt2)) < 1e-4f);
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity Logit1 Failed==============", fabs(logits[1] - (-invSqrt2)) < 1e-4f);
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity Logit2 Failed==============", fabs(logits[2] - sqrt2) < 1e-4f);

		// Same via session-based incremental decode.
		glades::NNetwork::TransformerLmSession session;
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity SessionReset Failed==============",
		         net.transformerLmSessionReset(session, /*maxSeqLen*/ 4u).ok());
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity SessionLen0 Failed==============", session.getCurrentLength() == 0u);
		std::vector<float> logitsKv;
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity SessionAppend Failed==============",
		         net.transformerLmSessionAppend(session, /*tokenId*/ 2u, &logitsKv).ok());
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity SessionLen1 Failed==============", session.getCurrentLength() == 1u);
		G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity KvLogitsSize Failed==============", logitsKv.size() == vocab);
		for (unsigned int i = 0; i < vocab; ++i)
			G_assert(__FILE__, __LINE__, "==============NN::ToyIdentity KvLogitsMismatch Failed==============", fabs(logitsKv[i] - logits[i]) < 1e-6f);

		delete di;
		delete info;
	}

	// Deterministic positional encoding test: with identity embeddings and no-op blocks, the logits are:
	// logits[v] = 1_{v==token} + PE[pos, v]
	printf("-----------------------------------\n");
	printf("Transformer decoder sinusoidal positional encoding exactness (toy model)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 3u;
		const unsigned int dModel = 3u;
		const unsigned int dFF = 4u;
		const unsigned int nHeads = 1u;
		const unsigned int nKVHeads = 1u;
		const unsigned int nLayers = 1u;
		const unsigned int padTokenId = vocab - 1u;

		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		{
			std::vector<unsigned int> toks;
			toks.push_back(0u);
			toks.push_back(1u);
			toks.push_back(2u);
			di->setTrainTokens(toks, static_cast<int>(padTokenId));
			di->mirrorTrainToTest();
		}

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
		    /*size*/ static_cast<int>(dModel),
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_toy_sinusoidal", in, hidden, out);

		glades::NNetwork bootstrap(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		{
			glades::TrainingConfig& cfg = bootstrap.getTrainingConfigMutable();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = static_cast<int>(nHeads);
			cfg.transformer.nKVHeadsOverride = static_cast<int>(nKVHeads);
			cfg.transformer.dFFOverride = static_cast<int>(dFF);
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_SINUSOIDAL;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_LAYERNORM;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
		}
		G_assert(__FILE__, __LINE__, "==============NN::ToySinusoidal InitTestStatus() Failed==============", bootstrap.test(di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::ToySinusoidal SaveModel(bootstrap) Failed==============", bootstrap.saveModel("ut_pkg_transformer_toy_sinusoidal").ok());

		const std::vector<float> E = tokE_identity(vocab, dModel);
		G_assert(__FILE__, __LINE__, "==============NN::ToySinusoidal WriteOverrideWeights Failed==============",
		         write_transformer_decoder_tokenlm_weights("database/models/ut_pkg_transformer_toy_sinusoidal/weights.bin",
		                                                   nLayers, dModel, dFF, nHeads, nKVHeads,
		                                                   vocab, padTokenId,
		                                                   static_cast<unsigned int>(glades::TransformerRunConfig::FFN_MLP),
		                                                   E, std::vector<float>(vocab, 0.0f),
		                                                   /*zeroAllBlocks*/ true));

		glades::NNetwork net(glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		G_assert(__FILE__, __LINE__, "==============NN::ToySinusoidal LoadModel Failed==============", net.loadModel("ut_pkg_transformer_toy_sinusoidal", di).ok());

		// Compare full-forward vs incremental session decode and against the closed-form expected logits.
		glades::NNetwork::TransformerLmSession session;
		G_assert(__FILE__, __LINE__, "==============NN::ToySinusoidal SessionReset Failed==============",
		         net.transformerLmSessionReset(session, /*maxSeqLen*/ 8u).ok());
		std::vector<unsigned int> prefix;
		std::vector<float> logitsFull;
		std::vector<float> logitsKv;
		for (unsigned int t = 0; t < 3u; ++t)
		{
			const unsigned int tok = static_cast<unsigned int>(t);
			prefix.push_back(tok);
			G_assert(__FILE__, __LINE__, "==============NN::ToySinusoidal ForwardLastLogits Failed==============",
			         net.transformerLmForwardLastLogits(prefix, logitsFull).ok());
			G_assert(__FILE__, __LINE__, "==============NN::ToySinusoidal SessionAppend Failed==============",
			         net.transformerLmSessionAppend(session, tok, &logitsKv).ok());
			G_assert(__FILE__, __LINE__, "==============NN::ToySinusoidal LogitsSize Failed==============", logitsFull.size() == vocab && logitsKv.size() == vocab);

			// Compute expected hidden state, then apply LayerNorm (gamma=1, beta=0)
			// to get the expected logits (tied identity embedding).
			double h[3];
			double hMean = 0.0;
			for (unsigned int v = 0; v < vocab; ++v)
			{
				h[v] = ((v == tok) ? 1.0 : 0.0) + static_cast<double>(sinusoidal_pe(/*pos*/ t, /*i*/ v, /*dModel*/ dModel));
				hMean += h[v];
			}
			hMean /= static_cast<double>(vocab);
			double hVar = 0.0;
			for (unsigned int v = 0; v < vocab; ++v)
			{
				const double d = h[v] - hMean;
				hVar += d * d;
			}
			hVar /= static_cast<double>(vocab);
			const double hInvStd = 1.0 / sqrt(hVar + 1e-5);
			for (unsigned int v = 0; v < vocab; ++v)
			{
				const float exp = static_cast<float>((h[v] - hMean) * hInvStd);
				G_assert(__FILE__, __LINE__, "==============NN::ToySinusoidal FullExpectedMismatch Failed==============", fabs(logitsFull[v] - exp) < 1e-4f);
				G_assert(__FILE__, __LINE__, "==============NN::ToySinusoidal KvExpectedMismatch Failed==============", fabs(logitsKv[v] - exp) < 1e-4f);
			}
		}

		delete di;
		delete info;
	}

	// LLM inference API validation: verify that error paths return explicit statuses rather than silent no-ops.
	printf("-----------------------------------\n");
	printf("Transformer inference API validation (error paths)\n");
	printf("-----------------------------------\n");
	{
		// Cache reset: maxSeqLen must be > 0.
		{
			glades::NNetwork dec(glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			dec.getTrainingConfigMutable().transformer.enableTokenEmbedding = true;
			glades::NNetwork::TransformerLmSession session;
			const glades::NNetworkStatus st = dec.transformerLmSessionReset(session, 0u);
			G_assert(__FILE__, __LINE__, "==============NN::InferApi MaxSeqLen0() Failed==============", !st.ok());
		}

		// Wrong net type.
		{
			glades::NNetwork dff(glades::NNetwork::TYPE_DFF);
			glades::NNetwork::TransformerLmSession session;
			const glades::NNetworkStatus st = dff.transformerLmSessionReset(session, 4u);
			G_assert(__FILE__, __LINE__, "==============NN::InferApi WrongNetType() Failed==============", !st.ok());
		}

		// Decoder net but token LM not enabled.
		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(8, glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_infer_api", in, hidden, out);

			glades::NNetwork dec(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			glades::NNetwork::TransformerLmSession session;
			const glades::NNetworkStatus st = dec.transformerLmSessionReset(session, 4u);
			G_assert(__FILE__, __LINE__, "==============NN::InferApi TokenLMDisabled() Failed==============", !st.ok());

			delete info;
		}

		// Unknown positional encoding should fail fast in both reset() and append().
		{
			const unsigned int vocab = 8u;
			InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
			{
				std::vector<unsigned int> toks;
				toks.push_back(1u);
				toks.push_back(2u);
				// padTokenId is irrelevant in this error-path test; use -1 to skip last target.
				di->setTrainTokens(toks, -1);
				di->mirrorTrainToTest();
			}

			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_infer_bad_posenc", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			{
				glades::TrainingConfig cfg = net.getTrainingConfig();
				cfg.transformer.enableTokenEmbedding = true;
				cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
				cfg.transformer.tieEmbeddings = true;
				cfg.transformer.nHeadsOverride = 2;
				cfg.transformer.dFFOverride = 16;
				cfg.transformer.positionalEncoding = static_cast<glades::TransformerRunConfig::PositionalEncodingType>(999);
				const glades::NNetworkStatus stCfg = net.setTrainingConfig(cfg);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi BadPosEnc SetTrainingConfigShouldFail Failed==============", !stCfg.ok());
			}

			delete di;
			delete info;
		}

			// Append/forward argument validation (uninitialized cache, tokenId bounds, empty prefix).
			{
			const unsigned int vocab = 8u;
			InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
			{
				std::vector<unsigned int> toks;
				toks.push_back(1u);
				toks.push_back(2u);
				di->setTrainTokens(toks, -1);
				di->mirrorTrainToTest();
			}

			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_infer_append_bounds", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.transformer.enableTokenEmbedding = true;
				cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
				cfg.transformer.tieEmbeddings = true;
				cfg.transformer.nHeadsOverride = 2;
				cfg.transformer.dFFOverride = 16;
				cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_NONE;
			}
			G_assert(__FILE__, __LINE__, "==============NN::InferApi AppendBounds InitTestStatus() Failed==============", net.test(di).ok());

			// Append before reset should fail.
			{
				glades::NNetwork::TransformerLmSession session; // not initialized
				std::vector<float> logits;
				const glades::NNetworkStatus st = net.transformerLmSessionAppend(session, 1u, &logits);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi AppendBeforeReset ShouldFail Failed==============", !st.ok());
			}

			// Empty prefix forward should fail.
			{
				std::vector<unsigned int> empty;
				std::vector<float> logits;
				const glades::NNetworkStatus st = net.transformerLmForwardLastLogits(empty, logits);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi EmptyPrefix ShouldFail Failed==============", !st.ok());
			}

			// Reset with maxLen=1 then append twice => second append should fail (cache full).
			{
				glades::NNetwork::TransformerLmSession session;
				G_assert(__FILE__, __LINE__, "==============NN::InferApi SessionReset1 Failed==============", net.transformerLmSessionReset(session, 1u).ok());
				std::vector<float> logits;
				G_assert(__FILE__, __LINE__, "==============NN::InferApi SessionAppend0 Failed==============", net.transformerLmSessionAppend(session, 1u, &logits).ok());
				const glades::NNetworkStatus st2 = net.transformerLmSessionAppend(session, 2u, &logits);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi KvAppendOverflow ShouldFail Failed==============", !st2.ok());
			}

			// Token id out of range should fail.
			{
				glades::NNetwork::TransformerLmSession session;
				G_assert(__FILE__, __LINE__, "==============NN::InferApi SessionReset2 Failed==============", net.transformerLmSessionReset(session, 2u).ok());
				std::vector<float> logits;
				const glades::NNetworkStatus st = net.transformerLmSessionAppend(session, /*tokenId*/ vocab, &logits);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi TokenOutOfRange ShouldFail Failed==============", !st.ok());
			}

			// Session reset snapshots transformer runtime knobs used by append.
			{
				glades::TrainingConfig savedCfg = net.getTrainingConfig();

				glades::NNetwork::TransformerLmSession session;
				G_assert(__FILE__, __LINE__, "==============NN::InferApi SessionSnapshotReset Failed==============", net.transformerLmSessionReset(session, 3u).ok());
				net.getTrainingConfigMutable().transformer.positionalEncoding = static_cast<glades::TransformerRunConfig::PositionalEncodingType>(999);
				net.getTrainingConfigMutable().transformer.ffnActivation = glades::TransformerRunConfig::FFN_RELU;
				net.getTrainingConfigMutable().transformer.layerNormEps = 0.0f;

				std::vector<float> logits;
				const glades::NNetworkStatus st = net.transformerLmSessionAppend(session, 1u, &logits);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi SessionSnapshotAppend Failed==============", st.ok());
				G_assert(__FILE__, __LINE__, "==============NN::InferApi SessionSnapshotRestore Failed==============", net.setTrainingConfig(savedCfg).ok());
			}

			// Batch session reset snapshots the same runtime knobs.
			{
				glades::TrainingConfig savedCfg = net.getTrainingConfig();

				glades::NNetwork::TransformerLmBatchSession session;
				G_assert(__FILE__, __LINE__, "==============NN::InferApi BatchSessionSnapshotReset Failed==============", net.transformerLmBatchSessionReset(session, 1u, 3u).ok());
				net.getTrainingConfigMutable().transformer.positionalEncoding = static_cast<glades::TransformerRunConfig::PositionalEncodingType>(999);
				net.getTrainingConfigMutable().transformer.ffnActivation = glades::TransformerRunConfig::FFN_RELU;
				net.getTrainingConfigMutable().transformer.layerNormEps = 0.0f;

				std::vector<unsigned int> tokenIds(1, 1u);
				std::vector<unsigned char> active(1, 1u);
				std::vector<float> logits;
				const glades::NNetworkStatus st = net.transformerLmBatchSessionAppendSelective(session, tokenIds, active, &logits);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi BatchSessionSnapshotAppend Failed==============", st.ok());
				G_assert(__FILE__, __LINE__, "==============NN::InferApi BatchSessionSnapshotLogits Failed==============", logits.size() == vocab);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi BatchSessionSnapshotRestore Failed==============", net.setTrainingConfig(savedCfg).ok());
			}

			// Typed KV-session allocation cap should fail fast without relying on env state.
			{
				glades::TrainingConfig savedCfg = net.getTrainingConfig();
				glades::TrainingConfig cappedCfg = savedCfg;
				cappedCfg.transformer.kvSessionMaxBytes = 64ULL;
				G_assert(__FILE__, __LINE__, "==============NN::InferApi TypedKvCapConfig Failed==============", net.setTrainingConfig(cappedCfg).ok());

				glades::NNetwork::TransformerLmSession session;
				const glades::NNetworkStatus st = net.transformerLmSessionReset(session, 64u);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi TypedKvCapResetShouldFail Failed==============", !st.ok());
				G_assert(__FILE__, __LINE__, "==============NN::InferApi TypedKvCapRestore Failed==============", net.setTrainingConfig(savedCfg).ok());
			}

				delete di;
				delete info;
			}

			// Serving buffer hardening: fail fast when the configured logits-buffer cap is too small.
			{
				const unsigned int vocab = 8u;
				InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
				{
					std::vector<unsigned int> toks;
					toks.push_back(1u);
					toks.push_back(2u);
					di->setTrainTokens(toks, -1);
					di->mirrorTrainToTest();
				}

				glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
				std::vector<glades::HiddenLayerInfo*> hidden;
				hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
				glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
				glades::NNInfo* info = new glades::NNInfo("ut_transformer_serve_buffer_cap", in, hidden, out);

				glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
				{
					glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
					cfg.transformer.enableTokenEmbedding = true;
					cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
					cfg.transformer.tieEmbeddings = true;
					cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
					cfg.transformer.nHeadsOverride = 2;
					cfg.transformer.dFFOverride = 16;
					cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_NONE;
				}
				G_assert(__FILE__, __LINE__, "==============NN::InferApi ServeCap InitTestStatus() Failed==============", net.test(di).ok());

				net.getTrainingConfigMutable().transformer.serveLogitsMaxBytes = 64ULL;

				std::vector<glades::NNetwork::TransformerServeRequest> reqs;
				reqs.resize(2);
				reqs[0].promptTokens.push_back(1u);
				reqs[1].promptTokens.push_back(2u);
				reqs[0].cfg.maxNewTokens = 1u;
				reqs[1].cfg.maxNewTokens = 1u;

				glades::NNetwork::TransformerServeBatchResult outBatch;
				const glades::NNetworkStatus stServe = net.transformerLmServeGenerateBatch(reqs, outBatch, NULL);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi ServeCap GenerateShouldFail Failed==============", !stServe.ok());

				glades::NNetwork::TransformerServeBatcher batcher;
				glades::NNetwork::TransformerServeBatcherConfig bcfg;
				bcfg.maxBatchSize = 2u;
				bcfg.maxSeqLen = 4u;
				const glades::NNetworkStatus stBatcher = net.transformerLmServeBatcherReset(batcher, bcfg);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi ServeCap BatcherResetShouldFail Failed==============", !stBatcher.ok());
				G_assert(__FILE__, __LINE__, "==============NN::InferApi ServeCap BatcherResetShouldLeaveUninitialized Failed==============", !batcher.isInitialized());

				glades::NNetwork::TransformerServeRequest serveReq;
				serveReq.promptTokens.push_back(1u);
				serveReq.cfg.maxNewTokens = 1u;
				unsigned int submitSlot = 123u;
				const glades::NNetworkStatus stSubmit = net.transformerLmServeBatcherSubmit(batcher, serveReq, submitSlot);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi ServeCap BatcherSubmitAfterResetFailureShouldFail Failed==============", !stSubmit.ok());
				G_assert(__FILE__, __LINE__, "==============NN::InferApi ServeCap BatcherSubmitAfterResetFailureSlotReset Failed==============", submitSlot == 0u);

				delete di;
				delete info;
			}

			// Token LM enabled but tensors not initialized yet.
			{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(8, glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_infer_api2", in, hidden, out);

			glades::NNetwork dec(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			dec.getTrainingConfigMutable().transformer.enableTokenEmbedding = true;
			dec.getTrainingConfigMutable().transformer.vocabSizeOverride = 8;
			dec.getTrainingConfigMutable().transformer.tieEmbeddings = true;
			glades::NNetwork::TransformerLmSession session;
				const glades::NNetworkStatus st = dec.transformerLmSessionReset(session, 4u);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi UninitializedTensors() Failed==============", !st.ok());

				delete info;
			}

			// Public infer/generate entry points should reject re-entry while the network is already running.
			{
				struct InferWhileRunningCb : public glades::ITrainingCallbacks
				{
					bool sawRunStart;
					glades::NNetworkStatus genStatus;
					glades::NNetworkStatus batchStatus;
					glades::NNetworkStatus forwardStatus;

					InferWhileRunningCb()
					    : sawRunStart(false),
					      genStatus(glades::NNetworkStatus::OK, std::string()),
					      batchStatus(glades::NNetworkStatus::OK, std::string()),
					      forwardStatus(glades::NNetworkStatus::OK, std::string())
					{
					}

					virtual void onRunStart(const glades::NNetwork& net, int)
					{
						sawRunStart = true;
						std::vector<unsigned int> prompt;
						prompt.push_back(1u);
						prompt.push_back(2u);

						glades::NNetwork::TransformerGenerateConfig cfg;
						cfg.maxNewTokens = 1u;

						glades::NNetwork::TransformerGenerateResult out;
						genStatus = net.transformerLmGenerate(prompt, cfg, out, NULL);

						std::vector<glades::NNetwork::TransformerServeRequest> reqs(1);
						reqs[0].promptTokens = prompt;
						reqs[0].cfg = cfg;
						glades::NNetwork::TransformerServeBatchResult outBatch;
						batchStatus = net.transformerLmServeGenerateBatch(reqs, outBatch, NULL);

						std::vector<float> logits;
						forwardStatus = net.transformerLmForwardLastLogits(prompt, logits);
					}

					virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics&)
					{
						return true;
					}

					virtual void onRunEnd(const glades::NNetwork&, int) {}
				};

				const unsigned int vocab = 8u;
				InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
				{
					std::vector<unsigned int> toks;
					toks.push_back(1u);
					toks.push_back(2u);
					toks.push_back(3u);
					di->setTrainTokens(toks, -1);
					di->mirrorTrainToTest();
				}

				glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
				std::vector<glades::HiddenLayerInfo*> hidden;
				hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
				glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
				glades::NNInfo* info = new glades::NNInfo("ut_transformer_infer_runlock", in, hidden, out);

				glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
				{
					glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
					cfg.transformer.enableTokenEmbedding = true;
					cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
					cfg.transformer.tieEmbeddings = true;
					cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
					cfg.transformer.nHeadsOverride = 2;
					cfg.transformer.dFFOverride = 16;
					cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_NONE;
				}
				G_assert(__FILE__, __LINE__, "==============NN::InferApi RunLock InitTestStatus() Failed==============", net.test(di).ok());

				InferWhileRunningCb cb;
				G_assert(__FILE__, __LINE__, "==============NN::InferApi RunLock TrainStatus Failed==============", net.train(di, &cb).ok());
				G_assert(__FILE__, __LINE__, "==============NN::InferApi RunLock CallbackSeen Failed==============", cb.sawRunStart);
				G_assert(__FILE__, __LINE__, "==============NN::InferApi RunLock GenerateRejected Failed==============", !cb.genStatus.ok());
				G_assert(__FILE__, __LINE__, "==============NN::InferApi RunLock BatchRejected Failed==============", !cb.batchStatus.ok());
				G_assert(__FILE__, __LINE__, "==============NN::InferApi RunLock ForwardRejected Failed==============", !cb.forwardStatus.ok());

				delete di;
				delete info;
			}
		}

	// TokenInput dataset parsing + sequence semantics (LLM data pipeline).
	printf("-----------------------------------\n");
	printf("TokenInput parsing + explicit split semantics\n");
	printf("-----------------------------------\n");
	{
		mkdir_if_missing("database");
		mkdir_if_missing("database/tokeninput_ut");

		// Directory semantics: require explicit train.tok and test.tok.
		{
			{
				std::ofstream tr("database/tokeninput_ut/train.tok");
				tr << "1 2 3\n";
				tr << "\n";          // empty line should be ignored
				tr << "4 5\n";
			}
			{
				std::ofstream te("database/tokeninput_ut/test.tok");
				te << "7 8 9 10\n";
			}

			glades::TokenInput di;
			di.setPadTokenId(99);
			di.import(shmea::GString("database/tokeninput_ut/"), 0);

			G_assert(__FILE__, __LINE__, "==============TokenInput Dir TrainSize Failed==============", di.getTrainSize() == 5u);
			G_assert(__FILE__, __LINE__, "==============TokenInput Dir TestSize Failed==============", di.getTestSize() == 4u);
			G_assert(__FILE__, __LINE__, "==============TokenInput Dir TokenInputCapability Failed==============", di.hasTokenIdInput());
			G_assert(__FILE__, __LINE__, "==============TokenInput Dir FeatureCount Failed==============", di.getFeatureCount() == 1u);
			G_assert(__FILE__, __LINE__, "==============TokenInput Dir TrainSeqCount Failed==============", di.getTrainSequenceCount() == 2u);
			G_assert(__FILE__, __LINE__, "==============TokenInput Dir TestSeqCount Failed==============", di.getTestSequenceCount() == 1u);
			G_assert(__FILE__, __LINE__, "==============TokenInput Dir TrainSeq0Len Failed==============", di.getTrainSequenceLength(0u) == 3u);
			G_assert(__FILE__, __LINE__, "==============TokenInput Dir TrainSeq1Len Failed==============", di.getTrainSequenceLength(1u) == 2u);

			// Verify next-token shift and pad at sequence end.
			int tok = 0, nxt = 0;
			G_assert(__FILE__, __LINE__, "==============TokenInput TokId0 Failed==============", di.getTrainTokenId(0u, tok) && tok == 1);
			G_assert(__FILE__, __LINE__, "==============TokenInput NextId0 Failed==============", di.getTrainExpectedTokenId(0u, nxt) && nxt == 2);
			G_assert(__FILE__, __LINE__, "==============TokenInput TokId2 Failed==============", di.getTrainTokenId(2u, tok) && tok == 3);
			G_assert(__FILE__, __LINE__, "==============TokenInput NextId2 Pad Failed==============", di.getTrainExpectedTokenId(2u, nxt) && nxt == 99);
			G_assert(__FILE__, __LINE__, "==============TokenInput TokId4 Failed==============", di.getTrainTokenId(4u, tok) && tok == 5);
			G_assert(__FILE__, __LINE__, "==============TokenInput NextId4 Pad Failed==============", di.getTrainExpectedTokenId(4u, nxt) && nxt == 99);
		}

		// File semantics: train only by default; test split stays empty unless mirroring is explicitly enabled.
		{
			{
				std::ofstream tr("database/tokeninput_ut/onefile.tok");
				tr << "2 3 4\n";
			}
			glades::TokenInput di;
			di.setPadTokenId(-1);
			di.import(shmea::GString("database/tokeninput_ut/onefile.tok"), 0);
			// padTokenId < 0 => do not emit final timestep (avoids negative expected token ids).
			G_assert(__FILE__, __LINE__, "==============TokenInput File TrainSize Failed==============", di.getTrainSize() == 2u);
			G_assert(__FILE__, __LINE__, "==============TokenInput File TestEmpty Failed==============", di.getTestSize() == 0u);
			G_assert(__FILE__, __LINE__, "==============TokenInput File SeqCount Failed==============", di.getTrainSequenceCount() == 1u && di.getTestSequenceCount() == 0u);
		}

		// Single-input mirroring is still available, but it must be requested explicitly.
		{
			glades::TokenInput di;
			di.setPadTokenId(-1);
			di.setMirrorTrainToTestOnImplicitSplit(true);
			di.import(shmea::GString("database/tokeninput_ut/onefile.tok"), 0);
			G_assert(__FILE__, __LINE__, "==============TokenInput File MirrorOptIn TestSize Failed==============", di.getTestSize() == 2u);
			G_assert(__FILE__, __LINE__, "==============TokenInput File MirrorOptIn SeqCount Failed==============",
			         di.getTrainSequenceCount() == 1u && di.getTestSequenceCount() == 1u);
		}

		// Directory mode should fail if the explicit test split is missing.
		{
			mkdir_if_missing("database/tokeninput_ut_missing_test");
			{
				std::ofstream tr("database/tokeninput_ut_missing_test/train.tok");
				tr << "1 2 3\n";
			}
			glades::TokenInput di;
			di.import(shmea::GString("database/tokeninput_ut_missing_test/"), 0);
			G_assert(__FILE__, __LINE__, "==============TokenInput MissingTest Failed==============", di.getTrainSize() == 0u);
			G_assert(__FILE__, __LINE__, "==============TokenInput MissingTest Status Failed==============", !di.getLastStatus().ok());
			G_assert(__FILE__, __LINE__, "==============TokenInput MissingTest Message Failed==============",
			         di.getLastStatus().message.find("unable to open file") != std::string::npos);
		}

		// Directory mode should also fail if test.tok exists but contains malformed tokens.
		{
			mkdir_if_missing("database/tokeninput_ut_bad_test");
			{
				std::ofstream tr("database/tokeninput_ut_bad_test/train.tok");
				tr << "1 2 3\n";
			}
			{
				std::ofstream te("database/tokeninput_ut_bad_test/test.tok");
				te << "7 nope 9\n";
			}
			glades::TokenInput di;
			di.import(shmea::GString("database/tokeninput_ut_bad_test/"), 0);
			G_assert(__FILE__, __LINE__, "==============TokenInput BadTest Failed==============", di.getTrainSize() == 0u && di.getTestSize() == 0u);
			G_assert(__FILE__, __LINE__, "==============TokenInput BadTest Status Failed==============", !di.getLastStatus().ok());
			G_assert(__FILE__, __LINE__, "==============TokenInput BadTest Message Failed==============",
			         di.getLastStatus().message.find("invalid token") != std::string::npos);
		}

		// Invalid token id (does not fit in int) should fail to load (trainSize stays 0).
		{
			{
				std::ofstream tr("database/tokeninput_ut/bad.tok");
				tr << "99999999999999999999\n";
			}
			glades::TokenInput di;
			di.import(shmea::GString("database/tokeninput_ut/bad.tok"), 0);
			G_assert(__FILE__, __LINE__, "==============TokenInput Bad ShouldBeEmpty Failed==============", di.getTrainSize() == 0u);
		}
	}

	// Token LM perplexity + pad skipping: make logits uniform (all zeros) so NLL is exactly ln(vocab).
	printf("-----------------------------------\n");
	printf("Transformer token LM perplexity + pad skipping (uniform toy model)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 3u;
		const unsigned int dModel = 4u;
		const unsigned int dFF = 8u;
		const unsigned int nHeads = 1u;
		const unsigned int nKVHeads = 1u;
		const unsigned int nLayers = 1u;
		const unsigned int padTokenId = vocab - 1u;

		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		{
			// One sequence length 4; last target is pad => skipped.
			std::vector<unsigned int> toks;
			toks.push_back(0u);
			toks.push_back(1u);
			toks.push_back(0u);
			toks.push_back(2u);
			di->setTestTokens(toks, static_cast<int>(padTokenId));
			di->setTrainTokens(toks, static_cast<int>(padTokenId));
		}

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
		    /*size*/ static_cast<int>(dModel),
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_toy_uniform", in, hidden, out);

		glades::NNetwork bootstrap(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		bootstrap.setSeed(7u);
		{
			glades::TrainingConfig& cfg = bootstrap.getTrainingConfigMutable();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = static_cast<int>(nHeads);
			cfg.transformer.nKVHeadsOverride = static_cast<int>(nKVHeads);
			cfg.transformer.dFFOverride = static_cast<int>(dFF);
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_NONE;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
		}
		G_assert(__FILE__, __LINE__, "==============NN::ToyUniform InitTestStatus() Failed==============", bootstrap.test(di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::ToyUniform SaveModel(bootstrap) Failed==============", bootstrap.saveModel("ut_pkg_transformer_toy_uniform").ok());

		// Override weights: all zeros (including tokE) => logits are all zeros => uniform softmax.
		const std::vector<float> Ez(static_cast<size_t>(vocab) * static_cast<size_t>(dModel), 0.0f);
		G_assert(__FILE__, __LINE__, "==============NN::ToyUniform WriteOverrideWeights Failed==============",
		         write_transformer_decoder_tokenlm_weights("database/models/ut_pkg_transformer_toy_uniform/weights.bin",
		                                                   nLayers, dModel, dFF, nHeads, nKVHeads,
		                                                   vocab, padTokenId,
		                                                   static_cast<unsigned int>(glades::TransformerRunConfig::FFN_MLP),
		                                                   Ez, std::vector<float>(vocab, 0.0f),
		                                                   /*zeroAllBlocks*/ true));

		glades::NNetwork net(glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		G_assert(__FILE__, __LINE__, "==============NN::ToyUniform LoadModel Failed==============", net.loadModel("ut_pkg_transformer_toy_uniform", di).ok());

		CaptureEpochMetricsCb cb;
		G_assert(__FILE__, __LINE__, "==============NN::ToyUniform TestStatus Failed==============", net.test(di, &cb).ok());
		G_assert(__FILE__, __LINE__, "==============NN::ToyUniform SawMetrics Failed==============", cb.saw);
		if (cb.saw)
		{
			const float expectedNll = static_cast<float>(log(static_cast<double>(vocab))); // uniform softmax
			const float expectedPpl = static_cast<float>(vocab);
			G_assert(__FILE__, __LINE__, "==============NN::ToyUniform NLLMismatch Failed==============", fabs(cb.last.totalError - expectedNll) < 1e-4f);
			G_assert(__FILE__, __LINE__, "==============NN::ToyUniform PerplexityMismatch Failed==============", fabs(cb.last.perplexity - expectedPpl) < 1e-3f);
		}

		delete di;
		delete info;
	}

	// Batch generation correctness + determinism: serve-batch must match per-request generate under per-request RNG overrides.
	// Also validates ragged prompt prefill, stop tokens, and per-request early stop behavior.
	printf("-----------------------------------\n");
	printf("Transformer serving: batch generation correctness + determinism\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 7u;
		const unsigned int dModel = 8u;
		const unsigned int dFF = 16u;
		const unsigned int nHeads = 1u;
		const unsigned int nKVHeads = 1u;
		const unsigned int nLayers = 1u;
		const unsigned int padTokenId = vocab - 1u; // 6
		const unsigned int stopTok = 5u;

		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		{
			// Any tokens are fine; we just need a TokenInput-like DataInput to initialize and load a token LM package.
			std::vector<unsigned int> toks;
			toks.push_back(0u);
			toks.push_back(1u);
			toks.push_back(2u);
			di->setTrainTokens(toks, static_cast<int>(padTokenId));
			di->mirrorTrainToTest();
		}

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
		    /*size*/ static_cast<int>(dModel),
		    /*learningRate*/ 0.0f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_transformer_toy_bias_only", in, hidden, out);

		// 1) Bootstrap package.
		glades::NNetwork bootstrap(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		bootstrap.setSeed(42u);
		{
			glades::TrainingConfig& cfg = bootstrap.getTrainingConfigMutable();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = static_cast<int>(nHeads);
			cfg.transformer.nKVHeadsOverride = static_cast<int>(nKVHeads);
			cfg.transformer.dFFOverride = static_cast<int>(dFF);
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_NONE;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
		}
		G_assert(__FILE__, __LINE__, "==============NN::ServeBatch Bootstrap InitTestStatus() Failed==============", bootstrap.test(di).ok());
		G_assert(__FILE__, __LINE__, "==============NN::ServeBatch Bootstrap SaveModel Failed==============", bootstrap.saveModel("ut_pkg_transformer_toy_bias_only").ok());

		// 2) Override weights: tokE all zeros => hidden state is zero; logits are exactly lmBias.
		const std::vector<float> Ez(static_cast<size_t>(vocab) * static_cast<size_t>(dModel), 0.0f);
		std::vector<float> bLm(vocab, 0.0f);
		for (unsigned int i = 0u; i < vocab; ++i)
			bLm[i] = 0.1f * static_cast<float>(i);
		// Make stopTok the greedy argmax and make pad extremely unlikely.
		bLm[stopTok] = 3.0f;
		bLm[padTokenId] = -100.0f;
		G_assert(__FILE__, __LINE__, "==============NN::ServeBatch WriteOverrideWeights Failed==============",
		         write_transformer_decoder_tokenlm_weights("database/models/ut_pkg_transformer_toy_bias_only/weights.bin",
		                                                   nLayers, dModel, dFF, nHeads, nKVHeads,
		                                                   vocab, padTokenId,
		                                                   static_cast<unsigned int>(glades::TransformerRunConfig::FFN_MLP),
		                                                   Ez, bLm,
		                                                   /*zeroAllBlocks*/ true));

		// 3) Load as a fresh net for inference/generation tests.
		glades::NNetwork net(glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		G_assert(__FILE__, __LINE__, "==============NN::ServeBatch LoadModel Failed==============", net.loadModel("ut_pkg_transformer_toy_bias_only", di).ok());

		struct AssertSameGenerateResult
		{
			static void run(const glades::NNetwork::TransformerGenerateResult& a,
			                const glades::NNetwork::TransformerGenerateResult& b,
			                const char* msg)
			{
				G_assert(__FILE__, __LINE__, msg, a.tokens == b.tokens);
				G_assert(__FILE__, __LINE__, msg, a.stoppedOnEos == b.stoppedOnEos);
				G_assert(__FILE__, __LINE__, msg, a.stoppedByStopToken == b.stoppedByStopToken);
				G_assert(__FILE__, __LINE__, msg, a.stoppedByCallback == b.stoppedByCallback);
				G_assert(__FILE__, __LINE__, msg, a.stoppedByLimit == b.stoppedByLimit);
				G_assert(__FILE__, __LINE__, msg, a.lastToken == b.lastToken);
			}
		};

			// Case A: ragged prompts + per-request RNG overrides => batch == per-request generate exactly.
			{
				const glades::TransformerPublicAPI::Runtime api = glades::TransformerPublicAPI::runtime(net);
				glades::NNetwork::TransformerGenerateConfig cfgA;
				cfgA.includePromptInOutput = true;
			cfgA.maxNewTokens = 6u;
			cfgA.maxSeqLen = 0u; // promptLen + maxNewTokens
			cfgA.temperature = 1.0f;
			cfgA.topK = 0u;
			cfgA.topP = 1.0f;
			cfgA.eosTokenId = -1;
			cfgA.stopOnEos = false;

			std::vector<glades::NNetwork::TransformerServeRequest> reqs;
			reqs.resize(2);
			reqs[0].promptTokens.clear(); // len 2
			reqs[0].promptTokens.push_back(0u);
			reqs[0].promptTokens.push_back(1u);
			reqs[1].promptTokens.clear(); // len 3
			reqs[1].promptTokens.push_back(2u);
			reqs[1].promptTokens.push_back(3u);
			reqs[1].promptTokens.push_back(4u);
			reqs[0].cfg = cfgA;
			reqs[1].cfg = cfgA;
			reqs[0].cfg.rngSeedOverride = 111ULL;
			reqs[1].cfg.rngSeedOverride = 222ULL;

			glades::NNetwork::TransformerServeBatchResult outBatch;
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseA BatchStatus Failed==============",
				         glades::TransformerPublicAPI::generateBatch(net, reqs, outBatch, NULL).ok());
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseA BatchSize Failed==============", outBatch.results.size() == reqs.size());

				for (unsigned int r = 0u; r < static_cast<unsigned int>(reqs.size()); ++r)
				{
					glades::NNetwork::TransformerGenerateResult outSingle;
					G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseA SingleStatus Failed==============",
					         api.generate(reqs[r].promptTokens, reqs[r].cfg, outSingle, NULL).ok());
					AssertSameGenerateResult::run(outBatch.results[r], outSingle, "==============NN::ServeBatch CaseA BatchVsSingleMismatch Failed==============");
				}

				std::vector<float> logitsDirect;
				std::vector<float> logitsFacade;
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseA DirectForwardStatus Failed==============",
				         net.transformerLmForwardLastLogits(reqs[0].promptTokens, logitsDirect).ok());
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseA FacadeForwardStatus Failed==============",
				         api.forwardLastLogits(reqs[0].promptTokens, logitsFacade).ok());
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseA ForwardSize Failed==============",
				         logitsDirect.size() == logitsFacade.size());
				for (size_t i = 0u; i < logitsDirect.size(); ++i)
				{
					G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseA ForwardParity Failed==============",
					         std::fabs(logitsDirect[i] - logitsFacade[i]) < 1e-6f);
				}

				// Determinism: same requests twice => identical outputs (since per-request overrides are fixed).
				glades::NNetwork::TransformerServeBatchResult outBatch2;
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseA Batch2Status Failed==============",
				         api.generateBatch(reqs, outBatch2, NULL).ok());
			for (unsigned int r = 0u; r < static_cast<unsigned int>(reqs.size()); ++r)
				AssertSameGenerateResult::run(outBatch.results[r], outBatch2.results[r], "==============NN::ServeBatch CaseA DeterminismMismatch Failed==============");

			// Override isolation: request0 alone should match request0 in the 2-request batch.
			std::vector<glades::NNetwork::TransformerServeRequest> req1;
			req1.push_back(reqs[0]);
			glades::NNetwork::TransformerServeBatchResult outSolo;
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseA SoloStatus Failed==============",
			         net.transformerLmServeGenerateBatch(req1, outSolo, NULL).ok());
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseA SoloSize Failed==============", outSolo.results.size() == 1u);
			AssertSameGenerateResult::run(outBatch.results[0], outSolo.results[0], "==============NN::ServeBatch CaseA OverrideIsolationMismatch Failed==============");
		}

		// Case B: stop tokens work per-request and do not affect other requests.
		{
			glades::NNetwork::TransformerGenerateConfig cfgB;
			cfgB.includePromptInOutput = true;
			cfgB.maxNewTokens = 10u;
			cfgB.temperature = 0.0f; // greedy => always emits stopTok
			cfgB.topK = 0u;
			cfgB.topP = 1.0f;
			cfgB.eosTokenId = -1;
			cfgB.stopOnEos = false;
			cfgB.rngSeedOverride = 999ULL; // irrelevant for greedy, but keep explicit

			glades::NNetwork::TransformerGenerateConfig cfgC = cfgB;
			cfgC.maxNewTokens = 3u;   // short request to ensure it runs past genIdx=0
			cfgC.temperature = 1.0f; // stochastic (still deterministic via override)
			cfgC.rngSeedOverride = 1234ULL;

			std::vector<glades::NNetwork::TransformerServeRequest> reqs;
			reqs.resize(2);
			reqs[0].promptTokens.clear();
			reqs[0].promptTokens.push_back(1u);
			reqs[0].cfg = cfgB;
			reqs[0].stopTokenIds.clear();
			reqs[0].stopTokenIds.push_back(stopTok);

			reqs[1].promptTokens.clear();
			reqs[1].promptTokens.push_back(2u);
			reqs[1].promptTokens.push_back(3u);
			reqs[1].cfg = cfgC;

			glades::NNetwork::TransformerServeBatchResult outBatch;
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseB BatchStatus Failed==============",
			         net.transformerLmServeGenerateBatch(reqs, outBatch, NULL).ok());
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseB Size Failed==============", outBatch.results.size() == 2u);

			// Request0 should stop immediately after emitting stopTok.
			const glades::NNetwork::TransformerGenerateResult& r0 = outBatch.results[0];
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseB StopFlag Failed==============", r0.stoppedByStopToken);
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseB StopByLimit False Failed==============", !r0.stoppedByLimit);
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseB StopByCallback False Failed==============", !r0.stoppedByCallback);
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseB StopToken Last Failed==============", r0.lastToken == stopTok);
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseB StopToken Emitted Failed==============",
			         !r0.tokens.empty() && r0.tokens[r0.tokens.size() - 1u] == stopTok);
			// Exactly one generated token (plus prompt if included).
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseB StopToken Count Failed==============", r0.tokens.size() == (reqs[0].promptTokens.size() + 1u));

			// Request1 should run to its maxNewTokens limit.
			const glades::NNetwork::TransformerGenerateResult& r1 = outBatch.results[1];
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseB OtherReqLimit Failed==============", r1.stoppedByLimit);
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseB OtherReqTokenCount Failed==============", r1.tokens.size() == (reqs[1].promptTokens.size() + cfgC.maxNewTokens));
		}

			// Case C: per-request callback early stop.
			{
			struct StopAfterOneCb : public glades::ITransformerServeCallbacks
			{
				virtual bool onToken(const glades::NNetwork& /*net*/, unsigned int requestIndex, unsigned int /*tokenId*/, unsigned int generatedIndex)
				{
					// Stop request 0 after emitting its first generated token.
					return (requestIndex == 0u) && (generatedIndex == 0u);
				}
			};

			glades::NNetwork::TransformerGenerateConfig cfg;
			cfg.includePromptInOutput = true;
			cfg.maxNewTokens = 5u;
			cfg.temperature = 1.0f;
			cfg.topK = 0u;
			cfg.topP = 1.0f;
			cfg.eosTokenId = -1;
			cfg.stopOnEos = false;
			cfg.rngSeedOverride = 2026ULL;

			std::vector<glades::NNetwork::TransformerServeRequest> reqs;
			reqs.resize(2);
			reqs[0].promptTokens.clear();
			reqs[0].promptTokens.push_back(0u);
			reqs[1].promptTokens.clear();
			reqs[1].promptTokens.push_back(1u);
			reqs[0].cfg = cfg;
			reqs[1].cfg = cfg;
			reqs[1].cfg.rngSeedOverride = 2027ULL;

			glades::NNetwork::TransformerServeBatchResult outBatch;
			StopAfterOneCb cb;
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseC BatchStatus Failed==============",
			         net.transformerLmServeGenerateBatch(reqs, outBatch, &cb).ok());
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseC Size Failed==============", outBatch.results.size() == 2u);

			// Request0 stopped by callback after 1 generated token.
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseC Req0 StopByCallback Failed==============", outBatch.results[0].stoppedByCallback);
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseC Req0 TokenCount Failed==============",
			         outBatch.results[0].tokens.size() == (reqs[0].promptTokens.size() + 1u));

			// Request1 should run to limit.
			G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseC Req1 Limit Failed==============", outBatch.results[1].stoppedByLimit);
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseC Req1 TokenCount Failed==============",
				         outBatch.results[1].tokens.size() == (reqs[1].promptTokens.size() + cfg.maxNewTokens));
			}

			// Case D: serving logits storage cap is enforced for one-shot batch generation and persistent batcher reset.
			{
				EnvVarGuard cap("GLADES_TRANSFORMER_SERVE_MAX_BYTES");
				cap.set("1");

				glades::NNetwork::TransformerGenerateConfig cfg;
				cfg.includePromptInOutput = false;
				cfg.maxNewTokens = 1u;
				cfg.maxSeqLen = 0u;
				cfg.temperature = 1.0f;
				cfg.topK = 1u;
				cfg.topP = 1.0f;
				cfg.eosTokenId = -1;
				cfg.stopOnEos = false;
				cfg.rngSeedOverride = 7ULL;

				std::vector<glades::NNetwork::TransformerServeRequest> reqs;
				reqs.resize(1);
				reqs[0].promptTokens.clear();
				reqs[0].promptTokens.push_back(1u);
				reqs[0].cfg = cfg;

				glades::NNetwork::TransformerServeBatchResult outBatch;
				const glades::NNetworkStatus stBatch = net.transformerLmServeGenerateBatch(reqs, outBatch, NULL);
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseD BatchCapFail Failed==============", !stBatch.ok());
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseD BatchCapMessage Failed==============",
				         stBatch.message.find("serving logits buffers require") != std::string::npos);

				glades::NNetwork::TransformerServeBatcher batcher;
				glades::NNetwork::TransformerServeBatcherConfig bcfg;
				bcfg.maxBatchSize = 1u;
				bcfg.maxSeqLen = 4u;
				const glades::NNetworkStatus stReset = net.transformerLmServeBatcherReset(batcher, bcfg);
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseD BatcherCapFail Failed==============", !stReset.ok());
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseD BatcherCapMessage Failed==============",
				         stReset.message.find("serving logits buffers require") != std::string::npos);
			}

			// Case E: single-call generation is deterministic even without rngSeedOverride.
			// Policy: seed is derived from (network rngSeed ^ prompt hash) when rngSeedOverride==0.
			{
			glades::NNetwork::TransformerGenerateConfig cfg;
			cfg.includePromptInOutput = true;
			cfg.maxNewTokens = 8u;
			cfg.maxSeqLen = 0u;
			cfg.temperature = 1.0f; // stochastic
			cfg.topK = 0u;
			cfg.topP = 1.0f;
			cfg.topPTopKCap = 0u; // disable approximation to ensure "pure" top-p policy (topP==1 anyway)
			cfg.eosTokenId = -1;
			cfg.stopOnEos = false;
			cfg.rngSeedOverride = 0ULL; // derive from net seed + prompt

			std::vector<unsigned int> prompt;
			prompt.push_back(0u);
			prompt.push_back(1u);

			glades::NNetwork::TransformerGenerateResult a, b;
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseE GenAStatus Failed==============",
				         net.transformerLmGenerate(prompt, cfg, a, NULL).ok());
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseE GenBStatus Failed==============",
				         net.transformerLmGenerate(prompt, cfg, b, NULL).ok());
				AssertSameGenerateResult::run(a, b, "==============NN::ServeBatch CaseE DeterminismNoOverrideMismatch Failed==============");
			}

			// Case F: fast-by-design top-p cap semantics are explicit and equivalent:
			//   (topP<1, topK==0, topPTopKCap=C) must behave identically to (topP<1, topK=C).
			{
			const unsigned int cap = 7u;
			glades::NNetwork::TransformerGenerateConfig cfgCap;
			cfgCap.includePromptInOutput = true;
			cfgCap.maxNewTokens = 10u;
			cfgCap.maxSeqLen = 0u;
			cfgCap.temperature = 1.0f;
			cfgCap.topK = 0u;        // enable cap path
			cfgCap.topP = 0.80f;     // nucleus
			cfgCap.topPTopKCap = cap; // approximation: cap candidate set
			cfgCap.eosTokenId = -1;
			cfgCap.stopOnEos = false;
			cfgCap.rngSeedOverride = 424242ULL;

			glades::NNetwork::TransformerGenerateConfig cfgExplicit = cfgCap;
			cfgExplicit.topK = cap;
			// Make the "cap" field irrelevant in the explicit-topK config to guard against bugs
			// where both topK and cap might accidentally interact.
			cfgExplicit.topPTopKCap = 0u;

			std::vector<unsigned int> prompt;
			prompt.push_back(2u);
			prompt.push_back(3u);

			glades::NNetwork::TransformerGenerateResult a, b;
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseF GenCapStatus Failed==============",
				         net.transformerLmGenerate(prompt, cfgCap, a, NULL).ok());
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseF GenExplicitStatus Failed==============",
				         net.transformerLmGenerate(prompt, cfgExplicit, b, NULL).ok());
				AssertSameGenerateResult::run(a, b, "==============NN::ServeBatch CaseF TopPCapSemanticsMismatch Failed==============");
			}

			// Case G: greedy semantics equivalence:
			//   temperature<=0 (greedy) must equal temperature>0 with topK=1 (deterministic argmax).
			{
			glades::NNetwork::TransformerGenerateConfig greedy;
			greedy.includePromptInOutput = true;
			greedy.maxNewTokens = 6u;
			greedy.maxSeqLen = 0u;
			greedy.temperature = 0.0f; // greedy
			greedy.topK = 0u;
			greedy.topP = 1.0f;
			greedy.eosTokenId = -1;
			greedy.stopOnEos = false;
			greedy.rngSeedOverride = 1ULL;

			glades::NNetwork::TransformerGenerateConfig top1 = greedy;
			top1.temperature = 1.0f;
			top1.topK = 1u; // deterministic argmax
			top1.rngSeedOverride = 999ULL; // should be irrelevant for deterministic topK=1

			std::vector<unsigned int> prompt;
			prompt.push_back(4u);
			prompt.push_back(5u);

			glades::NNetwork::TransformerGenerateResult a, b;
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseG GreedyStatus Failed==============",
				         net.transformerLmGenerate(prompt, greedy, a, NULL).ok());
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseG Top1Status Failed==============",
				         net.transformerLmGenerate(prompt, top1, b, NULL).ok());
				AssertSameGenerateResult::run(a, b, "==============NN::ServeBatch CaseG GreedyVsTop1Mismatch Failed==============");
			}

			// Case H: full-vocab sampling parity:
			//   topK==0 with topP==1 uses a specialized fast path; it must match explicit topK==vocab.
			{
			glades::NNetwork::TransformerGenerateConfig fast;
			fast.includePromptInOutput = true;
			fast.maxNewTokens = 10u;
			fast.maxSeqLen = 0u;
			fast.temperature = 1.0f;
			fast.topK = 0u;   // fast full-vocab path
			fast.topP = 1.0f; // no nucleus
			fast.eosTokenId = -1;
			fast.stopOnEos = false;
			fast.rngSeedOverride = 777ULL;

			glades::NNetwork::TransformerGenerateConfig explicitK = fast;
			explicitK.topK = vocab; // explicit candidate list is full vocab

			std::vector<unsigned int> prompt;
			// Tokens must be in [0, vocab). Use a non-trivial prompt length >= 2.
			prompt.push_back(1u);
			prompt.push_back(2u);
			prompt.push_back(3u);

			glades::NNetwork::TransformerGenerateResult a, b;
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseH FastStatus Failed==============",
				         net.transformerLmGenerate(prompt, fast, a, NULL).ok());
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseH ExplicitStatus Failed==============",
				         net.transformerLmGenerate(prompt, explicitK, b, NULL).ok());
				AssertSameGenerateResult::run(a, b, "==============NN::ServeBatch CaseH FullVocabParityMismatch Failed==============");
			}

			// Case I: greedy sampling must ignore rngSeedOverride.
			{
			glades::NNetwork::TransformerGenerateConfig g0;
			g0.includePromptInOutput = true;
			g0.maxNewTokens = 5u;
			g0.maxSeqLen = 0u;
			g0.temperature = 0.0f; // greedy
			g0.topK = 0u;
			g0.topP = 1.0f;
			g0.eosTokenId = -1;
			g0.stopOnEos = false;
			g0.rngSeedOverride = 1ULL;

			glades::NNetwork::TransformerGenerateConfig g1 = g0;
			g1.rngSeedOverride = 2ULL;

			std::vector<unsigned int> prompt;
			prompt.push_back(3u);
			prompt.push_back(4u);

			glades::NNetwork::TransformerGenerateResult a, b;
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseI GreedyAStatus Failed==============",
				         net.transformerLmGenerate(prompt, g0, a, NULL).ok());
				G_assert(__FILE__, __LINE__, "==============NN::ServeBatch CaseI GreedyBStatus Failed==============",
				         net.transformerLmGenerate(prompt, g1, b, NULL).ok());
				AssertSameGenerateResult::run(a, b, "==============NN::ServeBatch CaseI GreedySeedAffectsOutput Failed==============");
			}

		delete di;
		delete info;
	}

	// Shape/config validation for transformer/LLM modes.
	printf("-----------------------------------\n");
	printf("Transformer config validation (reject invalid shapes)\n");
	printf("-----------------------------------\n");
	{
		// Token LM mode requires integer token-id accessors; do not use NumberInput here.
		InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
		{
			std::vector<unsigned int> toks;
			toks.push_back(0u);
			di->setTrainTokens(toks, /*pad*/ -1);
			di->mirrorTrainToTest();
		}

		// dModel must be divisible by nHeads.
		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(10, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f)); // dModel=10
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(8, glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_bad_heads", in, hidden, out);
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.getTrainingConfigMutable().transformer.nHeadsOverride = 4; // 10 % 4 != 0
			net.getTrainingConfigMutable().transformer.enableTokenEmbedding = true;
			net.getTrainingConfigMutable().transformer.vocabSizeOverride = 8;
			net.getTrainingConfigMutable().transformer.tieEmbeddings = true;
			const glades::NNetworkStatus st = net.test(di);
			G_assert(__FILE__, __LINE__, "==============NN::BadHeads ShouldFail Failed==============", !st.ok());
			delete info;
		}

		// nKVHeads must divide nHeads (GQA grouping).
		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(12, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f)); // dModel=12
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(8, glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_bad_kv_heads", in, hidden, out);
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.getTrainingConfigMutable().transformer.nHeadsOverride = 4;
			net.getTrainingConfigMutable().transformer.nKVHeadsOverride = 3; // 4 % 3 != 0
			net.getTrainingConfigMutable().transformer.enableTokenEmbedding = true;
			net.getTrainingConfigMutable().transformer.vocabSizeOverride = 8;
			net.getTrainingConfigMutable().transformer.tieEmbeddings = true;
			const glades::NNetworkStatus st = net.test(di);
			G_assert(__FILE__, __LINE__, "==============NN::BadKVHeads ShouldFail Failed==============", !st.ok());
			delete info;
		}

		// Transformer requires constant hidden size across blocks.
		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
			hidden.push_back(new glades::HiddenLayerInfo(10, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f)); // mismatch
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(8, glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_bad_hidden_sizes", in, hidden, out);
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.getTrainingConfigMutable().transformer.nHeadsOverride = 2;
			net.getTrainingConfigMutable().transformer.enableTokenEmbedding = true;
			net.getTrainingConfigMutable().transformer.vocabSizeOverride = 8;
			net.getTrainingConfigMutable().transformer.tieEmbeddings = true;
			const glades::NNetworkStatus st = net.test(di);
			G_assert(__FILE__, __LINE__, "==============NN::BadHiddenSizes ShouldFail Failed==============", !st.ok());
			delete info;
		}

		// Token LM mode currently requires tieEmbeddings=true (explicit invariant).
		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(8, glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_bad_tie", in, hidden, out);
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.getTrainingConfigMutable().transformer.enableTokenEmbedding = true;
			net.getTrainingConfigMutable().transformer.vocabSizeOverride = 8;
			net.getTrainingConfigMutable().transformer.tieEmbeddings = false;
			const glades::NNetworkStatus st = net.test(di);
			G_assert(__FILE__, __LINE__, "==============NN::BadTieEmbeddings ShouldFail Failed==============", !st.ok());
			delete info;
		}

		// setTrainingConfig should reject invalid runtime config before execution starts.
		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(8, glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_invalid_config_boundary", in, hidden, out);
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			glades::TrainingConfig cfg = net.getTrainingConfig();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = 8;
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.positionalEncoding = static_cast<glades::TransformerRunConfig::PositionalEncodingType>(999);
			const glades::NNetworkStatus st = net.setTrainingConfig(cfg);
			G_assert(__FILE__, __LINE__, "==============NN::SetTrainingConfig InvalidPosEnc Failed==============", !st.ok());
			delete info;
		}

		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(8, glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_gpu_metrics_cfg", in, hidden, out);
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			glades::NNetwork::TransformerMetricsConfig metricsCfg = net.getTransformerMetricsConfig();
			metricsCfg.enable = true;
			metricsCfg.enableGpuPerf = true;
			metricsCfg.logGpuTrainSummary = true;
			metricsCfg.logGpuInferSummary = true;
			net.setTransformerMetricsConfig(metricsCfg);
			const glades::NNetwork::TransformerMetricsConfig appliedCfg = net.getTransformerMetricsConfig();
			G_assert(__FILE__, __LINE__, "==============NN::TransformerMetricsCfg EnableGpuPerf Failed==============",
			         appliedCfg.enableGpuPerf && appliedCfg.logGpuTrainSummary && appliedCfg.logGpuInferSummary);
			G_assert(__FILE__, __LINE__, "==============NN::TransformerMetricsCfg TrainSnapshotZero Failed==============",
			         net.getLastTransformerTrainGpuPerf().counters.kernelLaunches == 0ULL);
			G_assert(__FILE__, __LINE__, "==============NN::TransformerMetricsCfg InferSnapshotZero Failed==============",
			         net.getLastTransformerInferGpuPerf().counters.bytesH2D == 0ULL);
			delete info;
		}

		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(8, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(8, glades::OutputLayerInfo::CLASSIFICATION);
			glades::NNInfo* info = new glades::NNInfo("ut_transformer_invalid_sampled_softmax", in, hidden, out);
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			glades::TrainingConfig cfg = net.getTrainingConfig();
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = 8;
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.tokenLmLossKind = glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX;
			cfg.transformer.tokenLmSampledNegatives = 0;
			const glades::NNetworkStatus st = net.setTrainingConfig(cfg);
			G_assert(__FILE__, __LINE__, "==============NN::SetTrainingConfig InvalidSampledSoftmax Failed==============", !st.ok());
			delete info;
		}

		delete di;
	}

	// Transformer math-kernel unit tests (attention + numerics). These test the "LLM core" primitives directly.
	printf("-----------------------------------\n");
	printf("Transformer ops: attention forward/backward (causal + recompute parity)\n");
	printf("-----------------------------------\n");
	{
		// Simple, exactly-solvable causal attention: Q=K=0 => uniform over allowed prefix.
		{
			const unsigned int T = 3u;
			const unsigned int dK = 1u;
			const unsigned int dV = 1u;
			const float Q[T * dK] = {0.0f, 0.0f, 0.0f};
			const float K[T * dK] = {0.0f, 0.0f, 0.0f};
			const float V[T * dV] = {1.0f, 2.0f, 3.0f};
			std::vector<float> O;
			std::vector<float> probs;
			glades::transformer_ops::scaled_dot_product_attention_forward(Q, K, V, T, dK, dV, /*causal*/ true, O, &probs);
			G_assert(__FILE__, __LINE__, "==============Ops::Attn Simple O size Failed==============", O.size() == T * dV);
			// Expected:
			// t=0: only u=0 => O=1
			// t=1: mean of {1,2} => 1.5
			// t=2: mean of {1,2,3} => 2
			G_assert(__FILE__, __LINE__, "==============Ops::Attn Simple O0 Failed==============", fabs(O[0] - 1.0f) < 1e-6f);
			G_assert(__FILE__, __LINE__, "==============Ops::Attn Simple O1 Failed==============", fabs(O[1] - 1.5f) < 1e-6f);
			G_assert(__FILE__, __LINE__, "==============Ops::Attn Simple O2 Failed==============", fabs(O[2] - 2.0f) < 1e-6f);
			// Mask property: probs[t,u]=0 for u>t in causal mode.
			G_assert(__FILE__, __LINE__, "==============Ops::Attn Simple Mask p01 Failed==============", fabs(probs[0u * T + 1u] - 0.0f) < 1e-6f);
			G_assert(__FILE__, __LINE__, "==============Ops::Attn Simple Mask p02 Failed==============", fabs(probs[0u * T + 2u] - 0.0f) < 1e-6f);
			G_assert(__FILE__, __LINE__, "==============Ops::Attn Simple Mask p12 Failed==============", fabs(probs[1u * T + 2u] - 0.0f) < 1e-6f);
		}

		// Stable masked softmax: extreme values should stay finite and sum to 1 over allowed.
		{
			const unsigned int T = 3u;
			const float scores[T] = {1000.0f, 0.0f, -1000.0f};
			std::vector<float> p;
			glades::transformer_ops::softmax_masked_row_stable(scores, T, /*rowT*/ 2u, /*causal*/ false, p);
			G_assert(__FILE__, __LINE__, "==============Ops::Softmax Extreme Size Failed==============", p.size() == T);
			const float sum = p[0] + p[1] + p[2];
			G_assert(__FILE__, __LINE__, "==============Ops::Softmax Extreme Sum Failed==============", fabs(sum - 1.0f) < 1e-6f);
			G_assert(__FILE__, __LINE__, "==============Ops::Softmax Extreme Finite Failed==============", std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]));
			G_assert(__FILE__, __LINE__, "==============Ops::Softmax Extreme Argmax Failed==============", p[0] > 0.999f);
		}

		// Backward parity: cached-probs backward should match recompute backward.
		{
			const unsigned int T = 3u;
			const unsigned int dK = 2u;
			const unsigned int dV = 2u;
			const float Q[T * dK] = {
			    0.1f, -0.2f,
			    0.0f, 0.3f,
			    -0.4f, 0.5f};
			const float K[T * dK] = {
			    -0.1f, 0.2f,
			    0.4f, -0.3f,
			    0.2f, 0.1f};
			const float V[T * dV] = {
			    0.2f, 0.0f,
			    -0.1f, 0.3f,
			    0.4f, -0.2f};
			const float dO[T * dV] = {
			    1.0f, 0.5f,
			    -0.25f, 0.75f,
			    0.1f, -0.2f};

			std::vector<float> O;
			std::vector<float> probs;
			glades::transformer_ops::scaled_dot_product_attention_forward(Q, K, V, T, dK, dV, /*causal*/ true, O, &probs);

			std::vector<float> dQ1, dK1, dV1;
			glades::transformer_ops::scaled_dot_product_attention_backward(Q, K, V, dO, probs.data(), T, dK, dV, /*causal*/ true, dQ1, dK1, dV1);

			std::vector<float> dQ2, dK2, dV2;
			glades::transformer_ops::scaled_dot_product_attention_backward_recompute(Q, K, V, dO, T, dK, dV, /*causal*/ true, dQ2, dK2, dV2);

			G_assert(__FILE__, __LINE__, "==============Ops::Attn Backward Size dQ Failed==============", dQ1.size() == dQ2.size());
			G_assert(__FILE__, __LINE__, "==============Ops::Attn Backward Size dK Failed==============", dK1.size() == dK2.size());
			G_assert(__FILE__, __LINE__, "==============Ops::Attn Backward Size dV Failed==============", dV1.size() == dV2.size());
			double maxAbs = 0.0;
			for (size_t i = 0; i < dQ1.size(); ++i) { const double d = fabs((double)dQ1[i] - (double)dQ2[i]); if (d > maxAbs) maxAbs = d; }
			for (size_t i = 0; i < dK1.size(); ++i) { const double d = fabs((double)dK1[i] - (double)dK2[i]); if (d > maxAbs) maxAbs = d; }
			for (size_t i = 0; i < dV1.size(); ++i) { const double d = fabs((double)dV1[i] - (double)dV2[i]); if (d > maxAbs) maxAbs = d; }
			G_assert(__FILE__, __LINE__, "==============Ops::Attn Backward RecomputeMismatch Failed==============", maxAbs < 1e-4);
		}
	}

	printf("-----------------------------------\n");
	printf("Transformer ops: activation derivatives (finite-difference checks)\n");
	printf("-----------------------------------\n");
	{
		// These are small numerical checks to catch accidental derivative regressions.
		const double eps = 1e-3;
		const float xs[] = {-3.0f, -1.0f, -0.2f, 0.0f, 0.3f, 1.0f, 3.0f};
		const int N = static_cast<int>(sizeof(xs) / sizeof(xs[0]));
		for (int i = 0; i < N; ++i)
		{
			const double x = static_cast<double>(xs[i]);

			// SiLU
			{
				const double f1 = (double)glades::transformer_ops::silu((float)(x + eps));
				const double f0 = (double)glades::transformer_ops::silu((float)(x - eps));
				const double num = (f1 - f0) / (2.0 * eps);
				const double ana = (double)glades::transformer_ops::silu_deriv((float)x);
				G_assert(__FILE__, __LINE__, "==============Ops::SiLU DerivMismatch Failed==============", fabs(num - ana) < 5e-3);
			}

			// GELU
			{
				const double f1 = (double)glades::transformer_ops::gelu((float)(x + eps));
				const double f0 = (double)glades::transformer_ops::gelu((float)(x - eps));
				const double num = (f1 - f0) / (2.0 * eps);
				const double ana = (double)glades::transformer_ops::gelu_deriv((float)x);
				G_assert(__FILE__, __LINE__, "==============Ops::GELU DerivMismatch Failed==============", fabs(num - ana) < 5e-3);
			}
		}
	}

    printf("\n============================================================\n");
}
